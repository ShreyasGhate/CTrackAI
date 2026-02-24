"""
CTrackAI — Data Quality Handler (Stage 2)

Validates and cleans aggregated sensor readings before they enter
the ML pipeline. Catches SEVEN categories of data issues:

1. TIMESTAMP — Future timestamps, duplicates, out-of-order
2. MISSING — Null/None wattage values or empty readings
3. CORRUPT — Physically impossible values (negative, NaN, extreme)
4. OUTLIER — Statistically unusual values (z-score based)
5. STUCK SENSOR — Same value repeated 10+ times (sensor frozen)
6. RATE OF CHANGE — Physically impossible jumps between windows
7. ZERO WATTAGE — Always-on devices (servers) reading 0W

Replacement Strategy:
    Bad readings are replaced with the last known good value for
    that device/circuit. If no history exists, the device's rated
    idle wattage (from RAG specs) is used. If neither is available,
    the reading is flagged but passed through unchanged.

Design Principles:
    - Never silently drop a reading — always flag and document
    - Every quality decision is traceable in the output
    - Multiple issues can be detected on a single reading

Enterprise Notes:
    - All quality flags are structured for audit compliance
    - Quality statistics are tracked for monitoring dashboards
    - Configurable thresholds via settings
"""

import math
from typing import Optional, List, Tuple
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np
from loguru import logger

from models.schemas import (
    AggregatedReading,
    QualityCheckedReading,
    DataQualityStatus,
    DeviceSpec,
)
from config.settings import settings


class DataQualityHandler:
    """
    Validates and cleans sensor readings with 7 enterprise-grade checks.

    Maintains per-device/circuit state for:
        - Rolling mean/std for z-score outlier detection
        - Last known good values for replacement
        - Consecutive value tracking for stuck sensor detection
        - Timestamp ordering for duplicate/out-of-order detection

    Usage:
        handler = DataQualityHandler()
        checked = handler.check(aggregated_reading, device_spec=optional_spec)
        # checked.quality_status → "clean" | "flagged" | "replaced"
    """

    # ── Class Constants ───────────────────────────────────────
    # Device categories expected to always be powered on
    ALWAYS_ON_CATEGORIES = {"server", "network_switch", "ups", "router", "nas"}

    # Max rate-of-change multiplier (current / previous)
    MAX_RATE_OF_CHANGE = 3.0

    # Consecutive identical readings before flagging as stuck
    STUCK_SENSOR_THRESHOLD = 10

    # Max future timestamp tolerance (ESP32 clock drift)
    MAX_FUTURE_TIMESTAMP_SECONDS = 300  # 5 minutes

    def __init__(
        self,
        max_plausible_wattage: float = None,
        min_plausible_wattage: float = None,
        zscore_threshold: float = None,
        history_window: int = 100,
    ):
        """
        Args:
            max_plausible_wattage: Upper bound for valid wattage (default: from config)
            min_plausible_wattage: Lower bound for valid wattage (default: from config)
            zscore_threshold: Z-score threshold for outlier detection (default: from config)
            history_window: Number of recent readings to keep for statistics
        """
        self.max_plausible = max_plausible_wattage or settings.MAX_PLAUSIBLE_WATTAGE
        self.min_plausible = min_plausible_wattage or settings.MIN_PLAUSIBLE_WATTAGE
        self.zscore_threshold = zscore_threshold or settings.OUTLIER_ZSCORE_THRESHOLD
        self.history_window = history_window

        # Per-device/circuit rolling history for z-score calculation
        self._history: dict[Tuple[str, Optional[str]], List[float]] = defaultdict(list)

        # Last known good values
        self._last_known_good: dict[Tuple[str, Optional[str]], float] = {}

        # Consecutive identical reading counter for stuck sensor detection
        self._consecutive_tracker: dict[Tuple[str, Optional[str]], dict] = {}

        # Last timestamp per device for duplicate/ordering checks
        self._last_timestamp: dict[Tuple[str, Optional[str]], datetime] = {}

        # Quality statistics
        self._stats = {
            "total_checked": 0,
            "clean": 0,
            "flagged": 0,
            "replaced": 0,
            "issues": defaultdict(int),
        }

        logger.info(
            f"DataQualityHandler initialized: "
            f"max_wattage={self.max_plausible}W, "
            f"min_wattage={self.min_plausible}W, "
            f"zscore_threshold={self.zscore_threshold}, "
            f"history_window={self.history_window}"
        )

    # ══════════════════════════════════════════════════════════
    # MAIN CHECK ENTRY POINT
    # ══════════════════════════════════════════════════════════

    def check(
        self,
        reading: AggregatedReading,
        device_spec: Optional[DeviceSpec] = None,
    ) -> QualityCheckedReading:
        """
        Run all 7 quality checks on an aggregated reading.

        Checks are applied in order:
            0. Timestamp validation (future, duplicate, out-of-order)
            1. Missing value check (null wattage, zero readings)
            2. Corrupt value check (NaN, negative, extreme)
            3. Outlier check (z-score against device history)
            4. Stuck sensor check (identical consecutive readings)
            5. Rate-of-change check (impossible jumps)
            6. Zero wattage for always-on devices

        Args:
            reading: The aggregated reading to validate
            device_spec: Optional device specs from RAG layer

        Returns:
            QualityCheckedReading with status, flags, and corrected values
        """
        self._stats["total_checked"] += 1
        key = (reading.device_id, reading.circuit_id)

        original_wattage = reading.avg_wattage
        corrected_wattage = original_wattage
        flags: List[str] = []
        needs_replacement = False
        replacement_method = None

        # ── Check 0: Timestamp Validation ─────────────────────
        timestamp_flags = self._check_timestamp(reading)
        if timestamp_flags:
            flags.extend(timestamp_flags)
            # Future/duplicate timestamps → flag but keep data
            # (the wattage itself may still be valid)

        # ── Check 1: Missing Values ───────────────────────────
        missing_flags = self._check_missing(reading)
        if missing_flags:
            flags.extend(missing_flags)
            # Only trigger replacement for truly missing data,
            # not for partial gaps (which still have valid wattage)
            if "missing_wattage_null" in missing_flags or "missing_no_readings" in missing_flags:
                needs_replacement = True

        # ── Check 2: Corrupt Values ───────────────────────────
        if not needs_replacement:
            corrupt_flags = self._check_corrupt(reading, device_spec)
            if corrupt_flags:
                flags.extend(corrupt_flags)
                needs_replacement = True

        # ── Check 3: Outlier Detection ────────────────────────
        if not needs_replacement:
            outlier_flags = self._check_outlier(reading)
            if outlier_flags:
                flags.extend(outlier_flags)
                if self._is_extreme_outlier(reading):
                    needs_replacement = True

        # ── Check 4: Stuck / Frozen Sensor ────────────────────
        if not needs_replacement:
            stuck_flags = self._check_stuck_sensor(reading)
            if stuck_flags:
                flags.extend(stuck_flags)
                # Flagged only — stuck data is suspicious but
                # might still be valid (device genuinely idle)

        # ── Check 5: Rate of Change ───────────────────────────
        if not needs_replacement:
            roc_flags = self._check_rate_of_change(reading)
            if roc_flags:
                flags.extend(roc_flags)
                needs_replacement = True  # Impossible jump → replace

        # ── Check 6: Zero Wattage for Always-On Devices ───────
        if not needs_replacement and device_spec:
            zero_flags = self._check_zero_always_on(reading, device_spec)
            if zero_flags:
                flags.extend(zero_flags)
                # Flag but don't replace — could be a real power
                # loss that the anomaly detector should catch

        # ── Apply Replacement ─────────────────────────────────
        if needs_replacement:
            corrected_wattage, replacement_method = self._get_replacement(
                key, device_spec
            )
            if corrected_wattage is None:
                corrected_wattage = original_wattage
                replacement_method = "none_available"
                flags.append("no_replacement_available")
                logger.warning(
                    f"No replacement available for {reading.device_id}/"
                    f"{reading.circuit_id}. Keeping original value."
                )

        # ── Determine Final Status ────────────────────────────
        if not flags:
            status = DataQualityStatus.CLEAN
            self._stats["clean"] += 1
        elif needs_replacement and replacement_method != "none_available":
            status = DataQualityStatus.REPLACED
            self._stats["replaced"] += 1
        else:
            status = DataQualityStatus.FLAGGED
            self._stats["flagged"] += 1

        # ── Update History ────────────────────────────────────
        if status != DataQualityStatus.REPLACED:
            self._update_history(key, original_wattage)

        if status == DataQualityStatus.CLEAN:
            self._last_known_good[key] = original_wattage

        # Track issue types
        for flag in flags:
            self._stats["issues"][flag] += 1

        # ── Build Result ──────────────────────────────────────
        result = QualityCheckedReading(
            device_id=reading.device_id,
            circuit_id=reading.circuit_id,
            timestamp_start=reading.timestamp_start,
            timestamp_end=reading.timestamp_end,
            original_wattage=round(original_wattage, 2),
            corrected_wattage=round(corrected_wattage, 2),
            energy_kwh=self._recalculate_energy(corrected_wattage, reading),
            quality_status=status,
            quality_flags=flags,
            replacement_method=replacement_method,
        )

        if flags:
            logger.info(
                f"Quality check [{status.value}] for "
                f"{reading.device_id}/{reading.circuit_id}: "
                f"flags={flags}, original={original_wattage}W, "
                f"corrected={corrected_wattage}W"
            )

        return result

    # ══════════════════════════════════════════════════════════
    # CHECK 0: TIMESTAMP VALIDATION
    # ══════════════════════════════════════════════════════════

    def _check_timestamp(self, reading: AggregatedReading) -> List[str]:
        """
        Validate reading timestamps.

        Checks:
            - Future timestamp (ESP32 clock drift)
            - Duplicate timestamp (same window re-submitted)
            - Out-of-order timestamp (arrived before previous)
        """
        flags = []
        key = (reading.device_id, reading.circuit_id)
        now = datetime.now(timezone.utc)
        reading_time = reading.timestamp_start

        # Ensure timezone-aware
        if reading_time.tzinfo is None:
            reading_time = reading_time.replace(tzinfo=timezone.utc)

        # Future timestamp check
        future_limit = now + timedelta(seconds=self.MAX_FUTURE_TIMESTAMP_SECONDS)
        if reading_time > future_limit:
            offset = (reading_time - now).total_seconds()
            flags.append(f"timestamp_future_{offset:.0f}s_ahead")
            logger.warning(
                f"Future timestamp from {reading.device_id}/{reading.circuit_id}: "
                f"{reading_time.isoformat()} is {offset:.0f}s ahead of server time"
            )

        # Duplicate / out-of-order check
        last_ts = self._last_timestamp.get(key)
        if last_ts is not None:
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)

            if reading_time == last_ts:
                flags.append("timestamp_duplicate")
            elif reading_time < last_ts:
                delay = (last_ts - reading_time).total_seconds()
                flags.append(f"timestamp_out_of_order_{delay:.0f}s_behind_previous")

        # Update last timestamp
        self._last_timestamp[key] = reading_time
        return flags

    # ══════════════════════════════════════════════════════════
    # CHECK 1: MISSING VALUES
    # ══════════════════════════════════════════════════════════

    def _check_missing(self, reading: AggregatedReading) -> List[str]:
        """
        Check for missing values: null wattage, zero readings, data gaps.
        """
        flags = []

        if reading.avg_wattage is None:
            flags.append("missing_wattage_null")

        if reading.reading_count == 0:
            flags.append("missing_no_readings")

        if reading.has_gaps:
            flags.append("partial_data_gaps_detected")

        return flags

    # ══════════════════════════════════════════════════════════
    # CHECK 2: CORRUPT VALUES
    # ══════════════════════════════════════════════════════════

    def _check_corrupt(
        self,
        reading: AggregatedReading,
        device_spec: Optional[DeviceSpec] = None,
    ) -> List[str]:
        """
        Check for physically impossible values:
        NaN/Inf, negative, exceeds max plausible, exceeds device rated.
        """
        flags = []
        wattage = reading.avg_wattage

        if math.isnan(wattage) or math.isinf(wattage):
            flags.append("corrupt_nan_or_inf")
            return flags

        if wattage < self.min_plausible:
            flags.append(f"corrupt_negative_wattage_{wattage:.1f}W")

        if wattage > self.max_plausible:
            flags.append(
                f"corrupt_exceeds_max_plausible_{wattage:.1f}W"
                f"_limit_{self.max_plausible:.0f}W"
            )

        # Device-specific bound (allow 50% above rated for transient spikes)
        if device_spec and device_spec.rated_wattage:
            absolute_max = device_spec.rated_wattage * 1.5
            if wattage > absolute_max:
                flags.append(
                    f"corrupt_exceeds_device_rated_{wattage:.1f}W"
                    f"_max_{absolute_max:.0f}W"
                )

        return flags

    # ══════════════════════════════════════════════════════════
    # CHECK 3: OUTLIER DETECTION (Z-SCORE)
    # ══════════════════════════════════════════════════════════

    def _check_outlier(self, reading: AggregatedReading) -> List[str]:
        """
        Z-score outlier detection. Requires 10+ historical readings.
        """
        flags = []
        key = (reading.device_id, reading.circuit_id)
        history = self._history.get(key, [])

        if len(history) < 10:
            return flags

        mean = np.mean(history)
        std = np.std(history)

        if std == 0:
            return flags

        zscore = abs(reading.avg_wattage - mean) / std

        if zscore > self.zscore_threshold:
            flags.append(
                f"outlier_zscore_{zscore:.2f}_threshold_{self.zscore_threshold}"
            )

        return flags

    def _is_extreme_outlier(self, reading: AggregatedReading) -> bool:
        """Extreme outlier = z-score > 2× threshold. Triggers replacement."""
        key = (reading.device_id, reading.circuit_id)
        history = self._history.get(key, [])

        if len(history) < 10:
            return False

        mean = np.mean(history)
        std = np.std(history)

        if std == 0:
            return False

        zscore = abs(reading.avg_wattage - mean) / std
        return zscore > (self.zscore_threshold * 2)

    # ══════════════════════════════════════════════════════════
    # CHECK 4: STUCK / FROZEN SENSOR
    # ══════════════════════════════════════════════════════════

    def _check_stuck_sensor(self, reading: AggregatedReading) -> List[str]:
        """
        Detect a stuck/frozen sensor.

        If the same wattage (within ±0.1W tolerance) is reported for
        STUCK_SENSOR_THRESHOLD consecutive readings, the sensor is
        likely frozen. A truly idle device still has minor fluctuations
        from measurement noise — perfectly identical values suggest
        a stuck sensor or hardcoded fallback.
        """
        flags = []
        key = (reading.device_id, reading.circuit_id)
        wattage = reading.avg_wattage
        tolerance = 0.1  # Watts

        tracker = self._consecutive_tracker.get(key)

        if tracker is None:
            self._consecutive_tracker[key] = {"value": wattage, "count": 1}
        elif abs(wattage - tracker["value"]) <= tolerance:
            tracker["count"] += 1
            if tracker["count"] >= self.STUCK_SENSOR_THRESHOLD:
                flags.append(
                    f"stuck_sensor_{tracker['count']}_consecutive_identical_"
                    f"readings_at_{wattage:.1f}W"
                )
                logger.warning(
                    f"Stuck sensor detected for {reading.device_id}/"
                    f"{reading.circuit_id}: {tracker['count']} consecutive "
                    f"readings at {wattage:.1f}W"
                )
        else:
            self._consecutive_tracker[key] = {"value": wattage, "count": 1}

        return flags

    # ══════════════════════════════════════════════════════════
    # CHECK 5: RATE OF CHANGE
    # ══════════════════════════════════════════════════════════

    def _check_rate_of_change(self, reading: AggregatedReading) -> List[str]:
        """
        Check for physically impossible rate of change.

        If wattage jumps by more than MAX_RATE_OF_CHANGE × previous
        value in one window, it's likely a sensor error. Real devices
        don't go from 200W → 4000W instantly.

        Exception: Low-to-high transitions (prev < 50W) are allowed
        because devices legitimately power on from idle.
        """
        flags = []
        key = (reading.device_id, reading.circuit_id)
        current = reading.avg_wattage
        history = self._history.get(key, [])

        if not history:
            return flags

        previous = history[-1]

        # Skip for low previous values (device powering on is normal)
        if previous < 50.0:
            return flags

        # Check for impossible jump
        if current > 0 and previous > 0:
            ratio = current / previous
            if ratio > self.MAX_RATE_OF_CHANGE:
                flags.append(
                    f"rate_of_change_spike_{ratio:.1f}x_"
                    f"from_{previous:.0f}W_to_{current:.0f}W"
                )
                logger.warning(
                    f"Rate-of-change spike for {reading.device_id}/"
                    f"{reading.circuit_id}: {previous:.0f}W → {current:.0f}W "
                    f"({ratio:.1f}× increase)"
                )

        return flags

    # ══════════════════════════════════════════════════════════
    # CHECK 6: ZERO WATTAGE FOR ALWAYS-ON DEVICES
    # ══════════════════════════════════════════════════════════

    def _check_zero_always_on(
        self,
        reading: AggregatedReading,
        device_spec: DeviceSpec,
    ) -> List[str]:
        """
        Check for zero wattage on always-on devices.

        Servers, network switches, UPS, and routers should always have
        baseline power draw. A 0W reading indicates either a real power
        loss (critical info for anomaly detector) or a sensor fault.

        ONLY flags — does NOT replace, because a real power loss is
        critical information the anomaly detector needs to catch.
        """
        flags = []

        if device_spec.category.lower() not in self.ALWAYS_ON_CATEGORIES:
            return flags

        if reading.avg_wattage <= 0.5:  # ~0W threshold
            flags.append(
                f"zero_wattage_always_on_device_{device_spec.category}_"
                f"expected_min_{device_spec.idle_wattage:.0f}W"
            )
            logger.warning(
                f"Zero wattage for always-on {device_spec.category} "
                f"{reading.device_id}/{reading.circuit_id}: "
                f"reading={reading.avg_wattage}W, "
                f"expected_idle={device_spec.idle_wattage}W"
            )

        return flags

    # ══════════════════════════════════════════════════════════
    # REPLACEMENT METHODS
    # ══════════════════════════════════════════════════════════

    def _get_replacement(
        self,
        key: Tuple[str, Optional[str]],
        device_spec: Optional[DeviceSpec] = None,
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Get a replacement value for a bad reading.

        Priority: last-known-good → device idle wattage → historical mean → None
        """
        # Priority 1: Last known good
        if key in self._last_known_good:
            return (self._last_known_good[key], "last_known_good")

        # Priority 2: Device idle wattage from specs
        if device_spec and device_spec.idle_wattage > 0:
            return (device_spec.idle_wattage, "device_spec_idle_wattage")

        # Priority 3: Historical mean
        history = self._history.get(key, [])
        if len(history) >= 5:
            return (round(float(np.mean(history)), 2), "historical_mean")

        # Priority 4: No replacement available
        return (None, None)

    def _recalculate_energy(
        self,
        corrected_wattage: float,
        reading: AggregatedReading,
    ) -> float:
        """Recalculate energy (kWh) using corrected wattage."""
        duration = reading.timestamp_end - reading.timestamp_start
        window_hours = duration.total_seconds() / 3600.0
        return round((corrected_wattage * window_hours) / 1000.0, 6)

    # ══════════════════════════════════════════════════════════
    # HISTORY MANAGEMENT
    # ══════════════════════════════════════════════════════════

    def _update_history(
        self,
        key: Tuple[str, Optional[str]],
        wattage: float,
    ) -> None:
        """Add a wattage value to rolling history. Trims to window size."""
        self._history[key].append(wattage)
        if len(self._history[key]) > self.history_window:
            self._history[key] = self._history[key][-self.history_window:]

    def get_device_stats(
        self, device_id: str, circuit_id: Optional[str] = None
    ) -> Optional[dict]:
        """Get quality statistics for a specific device/circuit."""
        key = (device_id, circuit_id)
        history = self._history.get(key, [])

        if not history:
            return None

        return {
            "device_id": device_id,
            "circuit_id": circuit_id,
            "reading_count": len(history),
            "mean_wattage": round(float(np.mean(history)), 2),
            "std_wattage": round(float(np.std(history)), 2),
            "min_wattage": round(float(np.min(history)), 2),
            "max_wattage": round(float(np.max(history)), 2),
            "last_known_good": self._last_known_good.get(key),
        }

    def get_quality_stats(self) -> dict:
        """Get overall data quality statistics."""
        total = self._stats["total_checked"]
        return {
            "total_checked": total,
            "clean": self._stats["clean"],
            "flagged": self._stats["flagged"],
            "replaced": self._stats["replaced"],
            "clean_ratio": round(self._stats["clean"] / total, 4) if total > 0 else 0,
            "issue_breakdown": dict(self._stats["issues"]),
            "tracked_devices": len(self._history),
        }

    def reset_history(self, device_id: str = None, circuit_id: str = None) -> None:
        """Reset history for a specific device or all devices."""
        if device_id:
            key = (device_id, circuit_id)
            self._history.pop(key, None)
            self._last_known_good.pop(key, None)
            self._consecutive_tracker.pop(key, None)
            self._last_timestamp.pop(key, None)
            logger.info(f"Reset quality history for {device_id}/{circuit_id}")
        else:
            self._history.clear()
            self._last_known_good.clear()
            self._consecutive_tracker.clear()
            self._last_timestamp.clear()
            logger.info("Reset all quality history")


# ── Module-level singleton ────────────────────────────────────────
data_quality_handler = DataQualityHandler()
