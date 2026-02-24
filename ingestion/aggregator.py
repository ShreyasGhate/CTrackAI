"""
CTrackAI — Data Aggregator

Aggregates raw 10-second sensor readings into 15-minute windows.
ESP32 sensors send readings every 10 seconds. This module buffers
them and produces a single AggregatedReading per 15-minute window
per device/circuit combination.

Architecture:
    ESP32 (10s) → Buffer → Aggregator (15-min window) → Pipeline

Thread Safety:
    Uses threading.Lock for buffer access. Safe for concurrent
    REST + MQTT ingestion into the same buffer.

Enterprise Notes:
    - Handles late-arriving readings (up to 1 window behind)
    - Detects gaps in sensor data (missing readings)
    - Computes: avg, min, max wattage + total kWh for each window
    - Auto-flushes stale windows that exceed 2× window duration
"""

import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict

from models.schemas import SensorReading, AggregatedReading, CircuitReading
from config.settings import settings
from loguru import logger


class ReadingBuffer:
    """
    Thread-safe buffer for raw sensor readings.

    Buffers incoming 10-second readings and groups them by device/circuit
    and time window. When a window completes (or is force-flushed),
    it produces an AggregatedReading.

    Usage:
        buffer = ReadingBuffer(window_minutes=15)
        buffer.add_reading(sensor_reading)
        completed = buffer.flush_completed_windows()
    """

    def __init__(
        self,
        window_minutes: int = None,
        sensor_interval_seconds: int = None,
        on_window_complete: Optional[Callable] = None,
    ):
        """
        Args:
            window_minutes: Aggregation window size in minutes (default: from config)
            sensor_interval_seconds: Expected interval between readings (default: from config)
            on_window_complete: Optional callback when a window completes,
                                receives List[AggregatedReading]
        """
        self.window_minutes = window_minutes or settings.AGGREGATION_WINDOW_MINUTES
        self.sensor_interval = sensor_interval_seconds or settings.SENSOR_INTERVAL_SECONDS
        self.on_window_complete = on_window_complete

        # Buffer structure: {(device_id, circuit_id, window_start): [readings]}
        self._buffer: Dict[Tuple[str, Optional[str], datetime], List[dict]] = defaultdict(list)
        self._lock = threading.Lock()
        self._last_known_good: Dict[Tuple[str, Optional[str]], float] = {}

        # Expected readings per window
        self.expected_readings_per_window = (self.window_minutes * 60) // self.sensor_interval

        logger.info(
            f"ReadingBuffer initialized: window={self.window_minutes}min, "
            f"sensor_interval={self.sensor_interval}s, "
            f"expected_readings_per_window={self.expected_readings_per_window}"
        )

    def _get_window_start(self, timestamp: datetime) -> datetime:
        """
        Compute the start of the aggregation window for a given timestamp.

        Windows align to clock boundaries. E.g., for 15-min windows:
            10:00:00 – 10:14:59 → window_start = 10:00:00
            10:15:00 – 10:29:59 → window_start = 10:15:00

        Args:
            timestamp: The reading timestamp

        Returns:
            The aligned window start time
        """
        # Ensure timezone-aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        minutes = timestamp.minute
        window_start_minute = (minutes // self.window_minutes) * self.window_minutes

        return timestamp.replace(
            minute=window_start_minute,
            second=0,
            microsecond=0,
        )

    def add_reading(self, reading: SensorReading) -> List[AggregatedReading]:
        """
        Add a raw sensor reading to the buffer.

        If the reading has circuit_readings, each circuit is buffered separately.
        If it has a single wattage, it's buffered as a device-level reading.

        After adding, checks for and flushes any completed windows.

        Args:
            reading: Raw sensor reading from ESP32

        Returns:
            List of AggregatedReadings from any windows that completed
        """
        completed = []

        with self._lock:
            if reading.circuit_readings:
                # Multi-circuit reading — buffer each circuit separately
                for circuit in reading.circuit_readings:
                    if circuit.wattage is not None:
                        self._buffer_single(
                            device_id=reading.device_id,
                            circuit_id=circuit.circuit_id,
                            timestamp=reading.timestamp,
                            wattage=circuit.wattage,
                        )
            elif reading.wattage is not None:
                # Single device-level reading
                self._buffer_single(
                    device_id=reading.device_id,
                    circuit_id=None,
                    timestamp=reading.timestamp,
                    wattage=reading.wattage,
                )
            else:
                logger.warning(
                    f"Reading from {reading.device_id} has no wattage data, skipping"
                )
                return completed

            # Check for completed windows
            completed = self._flush_completed_windows_unsafe()

        # Fire callback outside the lock to prevent deadlocks
        if completed and self.on_window_complete:
            try:
                self.on_window_complete(completed)
            except Exception as e:
                logger.error(f"Window complete callback failed: {e}")

        return completed

    def _buffer_single(
        self,
        device_id: str,
        circuit_id: Optional[str],
        timestamp: datetime,
        wattage: float,
    ) -> None:
        """
        Buffer a single wattage reading. Must be called inside the lock.

        Args:
            device_id: Device identifier
            circuit_id: Circuit identifier (None for device-level)
            timestamp: Reading timestamp
            wattage: Power reading in watts
        """
        window_start = self._get_window_start(timestamp)
        key = (device_id, circuit_id, window_start)

        self._buffer[key].append({
            "timestamp": timestamp,
            "wattage": wattage,
        })

        # Update last known good value
        if wattage >= 0:
            self._last_known_good[(device_id, circuit_id)] = wattage

    def _flush_completed_windows_unsafe(self) -> List[AggregatedReading]:
        """
        Flush windows that are complete (past their end time).
        Must be called inside the lock.

        A window is considered complete when the current time is past
        window_start + window_duration.

        Returns:
            List of AggregatedReadings from completed windows
        """
        now = datetime.now(timezone.utc)
        window_duration = timedelta(minutes=self.window_minutes)
        completed = []
        keys_to_remove = []

        for key, readings in self._buffer.items():
            device_id, circuit_id, window_start = key
            window_end = window_start + window_duration

            if now >= window_end:
                # Window is complete — aggregate
                aggregated = self._aggregate_readings(
                    device_id=device_id,
                    circuit_id=circuit_id,
                    window_start=window_start,
                    window_end=window_end,
                    readings=readings,
                )
                completed.append(aggregated)
                keys_to_remove.append(key)

        # Remove flushed windows
        for key in keys_to_remove:
            del self._buffer[key]

        if completed:
            logger.info(f"Flushed {len(completed)} completed aggregation windows")

        return completed

    def _aggregate_readings(
        self,
        device_id: str,
        circuit_id: Optional[str],
        window_start: datetime,
        window_end: datetime,
        readings: List[dict],
    ) -> AggregatedReading:
        """
        Aggregate a list of raw readings into a single AggregatedReading.

        Computes: avg wattage, min wattage, max wattage, total energy (kWh),
        reading count, and whether gaps were detected.

        Energy formula:
            energy_kwh = avg_wattage × window_hours / 1000

        Args:
            device_id: Device identifier
            circuit_id: Circuit identifier
            window_start: Start of the aggregation window
            window_end: End of the aggregation window
            readings: List of raw readings in this window

        Returns:
            AggregatedReading with computed statistics
        """
        wattages = [r["wattage"] for r in readings]

        avg_wattage = sum(wattages) / len(wattages)
        max_wattage = max(wattages)
        min_wattage = min(wattages)

        # Energy = average power × time duration
        window_hours = self.window_minutes / 60.0
        energy_kwh = (avg_wattage * window_hours) / 1000.0

        # Detect gaps: if we received significantly fewer readings than expected
        has_gaps = len(readings) < (self.expected_readings_per_window * 0.8)

        if has_gaps:
            logger.warning(
                f"Gap detected for {device_id}/{circuit_id}: "
                f"received {len(readings)}/{self.expected_readings_per_window} readings "
                f"in window {window_start.isoformat()}"
            )

        return AggregatedReading(
            device_id=device_id,
            circuit_id=circuit_id,
            timestamp_start=window_start,
            timestamp_end=window_end,
            avg_wattage=round(avg_wattage, 2),
            max_wattage=round(max_wattage, 2),
            min_wattage=round(min_wattage, 2),
            energy_kwh=round(energy_kwh, 6),
            reading_count=len(readings),
            has_gaps=has_gaps,
        )

    def force_flush_all(self) -> List[AggregatedReading]:
        """
        Force-flush ALL buffered windows regardless of completion status.
        Used during shutdown or for testing.

        Returns:
            List of AggregatedReadings from all buffered windows
        """
        with self._lock:
            completed = []
            for key, readings in self._buffer.items():
                device_id, circuit_id, window_start = key
                window_end = window_start + timedelta(minutes=self.window_minutes)

                if readings:  # Only aggregate if we have data
                    aggregated = self._aggregate_readings(
                        device_id=device_id,
                        circuit_id=circuit_id,
                        window_start=window_start,
                        window_end=window_end,
                        readings=readings,
                    )
                    completed.append(aggregated)

            self._buffer.clear()
            logger.info(f"Force-flushed {len(completed)} windows")
            return completed

    def flush_stale_windows(self, max_age_minutes: int = None) -> List[AggregatedReading]:
        """
        Flush windows that are older than max_age.
        Prevents memory leaks from orphaned windows.

        Args:
            max_age_minutes: Max age before force-flush (default: 2× window)

        Returns:
            List of AggregatedReadings from stale windows
        """
        if max_age_minutes is None:
            max_age_minutes = self.window_minutes * 2

        now = datetime.now(timezone.utc)
        max_age = timedelta(minutes=max_age_minutes)

        with self._lock:
            completed = []
            keys_to_remove = []

            for key, readings in self._buffer.items():
                device_id, circuit_id, window_start = key

                if now - window_start > max_age:
                    window_end = window_start + timedelta(minutes=self.window_minutes)
                    if readings:
                        aggregated = self._aggregate_readings(
                            device_id, circuit_id, window_start, window_end, readings
                        )
                        completed.append(aggregated)
                    keys_to_remove.append(key)
                    logger.warning(
                        f"Flushing stale window: {device_id}/{circuit_id} "
                        f"from {window_start.isoformat()} (age: {now - window_start})"
                    )

            for key in keys_to_remove:
                del self._buffer[key]

            return completed

    def get_buffer_stats(self) -> dict:
        """
        Get current buffer statistics for monitoring.

        Returns:
            Dict with buffer stats (active windows, total readings, etc.)
        """
        with self._lock:
            total_readings = sum(len(r) for r in self._buffer.values())
            return {
                "active_windows": len(self._buffer),
                "total_buffered_readings": total_readings,
                "window_minutes": self.window_minutes,
                "sensor_interval_seconds": self.sensor_interval,
                "expected_readings_per_window": self.expected_readings_per_window,
            }

    def get_last_known_good(
        self, device_id: str, circuit_id: Optional[str] = None
    ) -> Optional[float]:
        """
        Get the last known good wattage value for a device/circuit.
        Used by the Data Quality Handler for replacing bad readings.

        Args:
            device_id: Device identifier
            circuit_id: Circuit identifier (None for device-level)

        Returns:
            Last known good wattage, or None if no history
        """
        return self._last_known_good.get((device_id, circuit_id))


# ── Module-level singleton ────────────────────────────────────────
# Shared across REST + MQTT ingestion
reading_buffer = ReadingBuffer()
