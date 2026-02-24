"""
CTrackAI — MQTT Data Ingestion

MQTT subscriber for receiving sensor readings from ESP32 devices
via an MQTT broker. This is an alternative ingestion path to REST.

Topic Structure:
    ctrackai/readings/{device_id}  → per-device topic
    ctrackai/readings/#            → wildcard subscription

Payload Format (JSON):
    {
        "device_id": "LAB_01",
        "timestamp": "2026-02-23T10:30:00Z",
        "wattage": 2450.5,
        "circuit_readings": [...]
    }

Architecture:
    ESP32 → MQTT Broker → MQTTSubscriber → ReadingBuffer → Pipeline

Enterprise Notes:
    - Automatic reconnection with exponential backoff
    - JSON payload validation via Pydantic
    - Dead letter logging for malformed messages
    - Graceful shutdown support
    - Configurable QoS levels
"""

import json
import threading
from datetime import datetime, timezone
from typing import Optional, Callable

import paho.mqtt.client as mqtt
from pydantic import ValidationError

from models.schemas import SensorReading
from ingestion.aggregator import reading_buffer
from config.settings import settings
from loguru import logger


class MQTTSubscriber:
    """
    MQTT subscriber that listens for sensor readings and feeds them
    into the shared ReadingBuffer.

    The subscriber runs in a background thread and automatically
    reconnects on disconnection.

    Usage:
        subscriber = MQTTSubscriber()
        subscriber.start()
        # ... application runs ...
        subscriber.stop()
    """

    def __init__(
        self,
        broker_host: str = None,
        broker_port: int = None,
        topic: str = None,
        client_id: str = "ctrackai_ingest",
        qos: int = 1,
        on_reading_received: Optional[Callable] = None,
    ):
        """
        Args:
            broker_host: MQTT broker hostname (default: from config)
            broker_port: MQTT broker port (default: from config)
            topic: MQTT topic to subscribe to (default: from config)
            client_id: MQTT client identifier
            qos: Quality of Service level (0, 1, or 2)
            on_reading_received: Optional callback when a valid reading is parsed
        """
        self.broker_host = broker_host or settings.MQTT_BROKER_HOST
        self.broker_port = broker_port or settings.MQTT_BROKER_PORT
        self.topic = topic or settings.MQTT_TOPIC
        self.client_id = client_id
        self.qos = qos
        self.on_reading_received = on_reading_received

        # MQTT client setup (v2 API)
        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=self.client_id,
            protocol=mqtt.MQTTv5,
        )
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

        # State tracking
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self._stats = {
            "messages_received": 0,
            "messages_parsed": 0,
            "messages_failed": 0,
            "reconnections": 0,
        }

        logger.info(
            f"MQTTSubscriber initialized: broker={self.broker_host}:{self.broker_port}, "
            f"topic={self.topic}, qos={self.qos}"
        )

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        """
        Callback when connected to the MQTT broker.
        Subscribes to the configured topic.
        """
        if reason_code == 0:
            logger.info(
                f"MQTT connected to {self.broker_host}:{self.broker_port}"
            )
            # Subscribe on connect (also handles re-subscription after reconnect)
            client.subscribe(self.topic, qos=self.qos)
            logger.info(f"MQTT subscribed to topic: {self.topic}")
        else:
            logger.error(f"MQTT connection failed with code: {reason_code}")

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None):
        """
        Callback when disconnected from the MQTT broker.
        Logs the disconnection; paho-mqtt handles automatic reconnection.
        """
        if reason_code != 0:
            logger.warning(
                f"MQTT unexpected disconnection (code: {reason_code}). "
                f"Auto-reconnect will be attempted."
            )
            self._stats["reconnections"] += 1
        else:
            logger.info("MQTT disconnected cleanly")

    def _on_message(self, client, userdata, message):
        """
        Callback when a message is received on the subscribed topic.

        Parses the JSON payload into a SensorReading and feeds it
        into the shared ReadingBuffer.

        Malformed messages are logged but not re-raised to prevent
        the subscriber from crashing.
        """
        self._stats["messages_received"] += 1

        try:
            # Decode payload
            payload_str = message.payload.decode("utf-8")
            payload = json.loads(payload_str)

            logger.debug(
                f"MQTT message on {message.topic}: {payload_str[:200]}"
            )

            # Extract device_id from topic if not in payload
            # Topic format: ctrackai/readings/{device_id}
            if "device_id" not in payload:
                topic_parts = message.topic.split("/")
                if len(topic_parts) >= 3:
                    payload["device_id"] = topic_parts[-1]

            # Ensure timestamp exists
            if "timestamp" not in payload:
                payload["timestamp"] = datetime.now(timezone.utc).isoformat()

            # Validate and parse via Pydantic
            reading = SensorReading(**payload)

            # Add to buffer
            completed_windows = reading_buffer.add_reading(reading)

            self._stats["messages_parsed"] += 1

            if completed_windows:
                logger.info(
                    f"MQTT ingestion triggered {len(completed_windows)} "
                    f"window aggregation(s)"
                )

            # Optional callback
            if self.on_reading_received:
                self.on_reading_received(reading)

        except json.JSONDecodeError as e:
            self._stats["messages_failed"] += 1
            logger.error(
                f"MQTT payload JSON decode error on {message.topic}: {e}. "
                f"Payload: {message.payload[:200]}"
            )

        except ValidationError as e:
            self._stats["messages_failed"] += 1
            logger.error(
                f"MQTT payload validation error on {message.topic}: {e}. "
                f"Payload: {payload_str[:200]}"
            )

        except Exception as e:
            self._stats["messages_failed"] += 1
            logger.error(
                f"MQTT message processing error on {message.topic}: {e}"
            )

    def start(self) -> None:
        """
        Start the MQTT subscriber in a background thread.

        Connects to the broker and begins listening for messages.
        The MQTT client loop runs in its own thread via loop_start().
        """
        if self._is_running:
            logger.warning("MQTT subscriber is already running")
            return

        try:
            logger.info(
                f"Starting MQTT subscriber: {self.broker_host}:{self.broker_port}"
            )
            self._client.connect(
                self.broker_host,
                self.broker_port,
                keepalive=60,
            )
            self._client.loop_start()
            self._is_running = True
            logger.info("MQTT subscriber started successfully")

        except ConnectionRefusedError:
            logger.error(
                f"MQTT broker connection refused at "
                f"{self.broker_host}:{self.broker_port}. "
                f"Is the broker running?"
            )

        except Exception as e:
            logger.error(f"MQTT subscriber failed to start: {e}")

    def stop(self) -> None:
        """
        Stop the MQTT subscriber gracefully.

        Unsubscribes, disconnects from the broker, and stops the
        background thread.
        """
        if not self._is_running:
            return

        try:
            logger.info("Stopping MQTT subscriber...")
            self._client.unsubscribe(self.topic)
            self._client.loop_stop()
            self._client.disconnect()
            self._is_running = False
            logger.info("MQTT subscriber stopped")

        except Exception as e:
            logger.error(f"MQTT subscriber stop error: {e}")
            self._is_running = False

    def get_stats(self) -> dict:
        """
        Get MQTT subscriber statistics.

        Returns:
            Dict with message counts and connection stats
        """
        return {
            **self._stats,
            "is_running": self._is_running,
            "broker": f"{self.broker_host}:{self.broker_port}",
            "topic": self.topic,
        }

    @property
    def is_running(self) -> bool:
        """Whether the subscriber is currently running."""
        return self._is_running


# ── Module-level singleton ────────────────────────────────────────
# Created but NOT started — call mqtt_subscriber.start() when ready
mqtt_subscriber = MQTTSubscriber()
