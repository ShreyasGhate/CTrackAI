"""
CTrackAI — Synthetic Data Generator (Stage 12)

Generates realistic lab sensor data for testing the full pipeline.

Simulates a typical Indian university CS lab with:
    - 30 PCs                (category: computer)
    - 4 Air Conditioners    (category: ac)
    - 2 Servers             (category: server)
    - 6 Printers            (category: printer)
    - 10 Ceiling Fans       (category: fan)
    - 8 Monitors            (category: monitor)
    = 60 devices

Patterns:
    - Diurnal: high usage 9 AM - 6 PM, low at night
    - Weekly: lower on weekends
    - Device-type specific idle/active ranges
    - Gaussian noise for realism

Anomaly Injection:
    - Spikes (random high-power events)
    - After-hours usage (device left on at night)
    - Zero-power on servers (simulated outage)
    - Baseline breach (gradual drift)
"""

import random
import math
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Tuple

from loguru import logger

from models.schemas import AggregatedReading


# ══════════════════════════════════════════════════════════════
# LAB DEVICE DEFINITIONS
# ══════════════════════════════════════════════════════════════

DEVICE_CATALOG = [
    # (prefix, category, count, idle_W, active_W, always_on)
    ("LAB_PC",     "computer", 30, 25,  200,  False),
    ("LAB_AC",     "ac",        4, 10,  1800, False),
    ("LAB_SRV",    "server",    2, 200, 750,  True),
    ("LAB_PTR",    "printer",   6, 5,   400,  False),
    ("LAB_FAN",    "fan",      10, 0,   75,   False),
    ("LAB_MON",    "monitor",   8, 2,   45,   False),
]


def _build_device_list() -> List[dict]:
    """Build the full device list from catalog."""
    devices = []
    for prefix, category, count, idle_w, active_w, always_on in DEVICE_CATALOG:
        for i in range(1, count + 1):
            devices.append({
                "device_id": f"{prefix}_{i:02d}",
                "category": category,
                "circuit_id": f"CIRCUIT_{category.upper()}",
                "idle_wattage": idle_w,
                "active_wattage": active_w,
                "always_on": always_on,
            })
    return devices


DEVICES = _build_device_list()


class SyntheticDataGenerator:
    """
    Generate realistic time-series sensor data for testing.

    Usage:
        gen = SyntheticDataGenerator(seed=42)
        readings = gen.generate_day(date=datetime(2026, 2, 23))
        for reading in readings:
            result = pipeline.process(reading)
    """

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._anomaly_log: List[dict] = []

        logger.info(
            f"SyntheticDataGenerator initialized: "
            f"{len(DEVICES)} devices, seed={seed}"
        )

    def generate_day(
        self,
        date: Optional[datetime] = None,
        anomaly_rate: float = 0.02,
    ) -> List[AggregatedReading]:
        """
        Generate one full day of 15-min readings for all devices.

        Args:
            date: Start date (default: today)
            anomaly_rate: Probability of injecting an anomaly per reading

        Returns:
            List of AggregatedReading (60 devices × 96 windows = 5760)
        """
        if date is None:
            date = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        elif date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)

        is_weekend = date.weekday() >= 5
        readings = []

        for window_idx in range(96):  # 96 × 15min = 24h
            window_start = date + timedelta(minutes=15 * window_idx)
            window_end = window_start + timedelta(minutes=15)
            hour = window_start.hour + window_start.minute / 60.0

            for device in DEVICES:
                wattage = self._generate_wattage(
                    device, hour, is_weekend
                )

                # Anomaly injection
                if self._rng.random() < anomaly_rate:
                    wattage, anomaly_info = self._inject_anomaly(
                        device, wattage, hour, is_weekend
                    )
                    if anomaly_info:
                        anomaly_info["timestamp"] = window_start.isoformat()
                        self._anomaly_log.append(anomaly_info)

                # Add Gaussian noise (±3%)
                noise = self._rng.gauss(0, wattage * 0.03)
                wattage = max(0, wattage + noise)

                # Build aggregated reading
                energy_kwh = (wattage * 0.25) / 1000.0  # 15min = 0.25h

                readings.append(AggregatedReading(
                    device_id=device["device_id"],
                    circuit_id=device["circuit_id"],
                    timestamp_start=window_start,
                    timestamp_end=window_end,
                    avg_wattage=round(wattage, 2),
                    max_wattage=round(wattage * 1.1, 2),
                    min_wattage=round(wattage * 0.9, 2),
                    energy_kwh=round(energy_kwh, 6),
                    reading_count=90,  # 90 × 10s readings per 15min
                ))

        logger.info(
            f"Generated {len(readings)} readings for "
            f"{date.strftime('%Y-%m-%d')} "
            f"({len(self._anomaly_log)} anomalies injected)"
        )
        return readings

    def _generate_wattage(
        self,
        device: dict,
        hour: float,
        is_weekend: bool,
    ) -> float:
        """Generate realistic wattage based on time-of-day pattern."""
        idle = device["idle_wattage"]
        active = device["active_wattage"]
        always_on = device["always_on"]

        # Always-on devices (servers) stay near active
        if always_on:
            load = 0.6 + self._rng.random() * 0.3  # 60-90% load
            return idle + (active - idle) * load

        # Working hours: 9 AM - 6 PM with ramp
        if 9 <= hour < 18 and not is_weekend:
            # Bell curve centered at 13:00 (peak usage)
            time_factor = math.exp(-0.5 * ((hour - 13) / 3) ** 2)
            load = 0.3 + 0.7 * time_factor
            return idle + (active - idle) * load

        # Shoulder hours: 7-9 AM, 6-8 PM (ramp up/down)
        if (7 <= hour < 9 or 18 <= hour < 20) and not is_weekend:
            load = 0.1 + self._rng.random() * 0.2
            return idle + (active - idle) * load

        # Night / weekend: idle or off
        if idle > 0:
            return idle * (0.5 + self._rng.random() * 0.5)
        return 0.0

    def _inject_anomaly(
        self,
        device: dict,
        current_wattage: float,
        hour: float,
        is_weekend: bool,
    ) -> Tuple[float, Optional[dict]]:
        """Inject a random anomaly into the reading."""
        anomaly_type = self._rng.choice([
            "spike", "after_hours", "zero_server", "baseline_breach"
        ])

        device_id = device["device_id"]

        if anomaly_type == "spike":
            # 2×-4× normal wattage
            multiplier = 2.0 + self._rng.random() * 2.0
            new_wattage = current_wattage * multiplier
            return new_wattage, {
                "device_id": device_id,
                "type": "spike",
                "original": round(current_wattage, 1),
                "injected": round(new_wattage, 1),
                "multiplier": round(multiplier, 2),
            }

        if anomaly_type == "after_hours" and (hour < 7 or hour >= 20):
            # Device running at full power during off-hours
            new_wattage = device["active_wattage"] * 0.8
            return new_wattage, {
                "device_id": device_id,
                "type": "after_hours",
                "original": round(current_wattage, 1),
                "injected": round(new_wattage, 1),
                "hour": round(hour, 1),
            }

        if anomaly_type == "zero_server" and device["always_on"]:
            # Server drops to 0W — simulated outage
            return 0.0, {
                "device_id": device_id,
                "type": "zero_always_on",
                "original": round(current_wattage, 1),
                "injected": 0.0,
            }

        if anomaly_type == "baseline_breach":
            # 50-100% above normal
            boost = 1.5 + self._rng.random() * 0.5
            new_wattage = current_wattage * boost
            return new_wattage, {
                "device_id": device_id,
                "type": "baseline_breach",
                "original": round(current_wattage, 1),
                "injected": round(new_wattage, 1),
                "boost": round(boost, 2),
            }

        # Didn't match conditions — return unchanged
        return current_wattage, None

    def get_anomaly_log(self) -> List[dict]:
        """Get log of all injected anomalies."""
        return self._anomaly_log

    def get_device_list(self) -> List[dict]:
        """Get the full device catalog."""
        return DEVICES


# ── Module-level singleton ────────────────────────────────────────
synthetic_generator = SyntheticDataGenerator()
