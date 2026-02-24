"""
CTrackAI — Context Engine (Stage 5 — Production Grade)

Enriches each reading with temporal and contextual features
that the anomaly detector and severity scorer need.

Production Features:
    1. UTC → IST timezone conversion (Fix #1)
    2. Thread-safe baseline storage (Fix #2)
    3. Device-type-aware risk scoring (Fix #3)
    4. Holiday-excluded baselines (Fix #4)
    5. Exponentially weighted moving average baseline (Fix #5)
    6. Dynamic holiday loading from JSON (Fix #6)

Why This Matters:
    A lab consuming 2000W at 2 PM on a Tuesday is normal.
    The same 2000W at 3 AM on a Sunday is a HIGH anomaly.
    A server at 2000W at 3 AM is normal (always-on).
    Context is the difference between noise and a real alert.
"""

import json
import threading
from collections import defaultdict
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
from typing import Optional, List, Tuple

from loguru import logger

from models.schemas import CarbonResult, ContextFeatures, DeviceSpec
from config.settings import settings


# ══════════════════════════════════════════════════════════════
# INDIAN HOLIDAY CALENDAR
# ══════════════════════════════════════════════════════════════

# Default holidays (national + Maharashtra state)
# Fix #6: Can be overridden by loading from JSON at runtime
_DEFAULT_HOLIDAYS = {
    # 2025
    "2025-01-26": "Republic Day",
    "2025-03-14": "Holi",
    "2025-03-31": "Id-ul-Fitr (Eid)",
    "2025-04-06": "Gudi Padwa",
    "2025-04-14": "Dr. Ambedkar Jayanti",
    "2025-04-18": "Good Friday",
    "2025-05-01": "Maharashtra Day",
    "2025-05-12": "Buddha Purnima",
    "2025-08-15": "Independence Day",
    "2025-08-27": "Ganesh Chaturthi",
    "2025-10-02": "Gandhi Jayanti",
    "2025-10-20": "Diwali",
    "2025-11-05": "Guru Nanak Jayanti",
    "2025-12-25": "Christmas",
    # 2026
    "2026-01-26": "Republic Day",
    "2026-03-10": "Maha Shivaratri",
    "2026-03-17": "Holi",
    "2026-03-31": "Id-ul-Fitr (Eid)",
    "2026-04-02": "Good Friday",
    "2026-04-06": "Gudi Padwa",
    "2026-04-10": "Mahavir Jayanti",
    "2026-04-14": "Dr. Ambedkar Jayanti",
    "2026-05-01": "Maharashtra Day",
    "2026-05-12": "Buddha Purnima",
    "2026-06-07": "Eid-ul-Adha (Bakrid)",
    "2026-07-07": "Muharram",
    "2026-08-15": "Independence Day",
    "2026-08-25": "Ganesh Chaturthi",
    "2026-09-05": "Milad-un-Nabi",
    "2026-10-02": "Gandhi Jayanti",
    "2026-10-20": "Dussehra",
    "2026-10-21": "Dussehra (Second day)",
    "2026-11-09": "Diwali (Lakshmi Puja)",
    "2026-11-10": "Diwali (Balipratipada)",
    "2026-11-11": "Bhai Dooj",
    "2026-11-30": "Guru Nanak Jayanti",
    "2026-12-25": "Christmas",
    # 2027
    "2027-01-26": "Republic Day",
    "2027-03-08": "Holi",
    "2027-04-14": "Dr. Ambedkar Jayanti",
    "2027-05-01": "Maharashtra Day",
    "2027-08-15": "Independence Day",
    "2027-10-02": "Gandhi Jayanti",
    "2027-12-25": "Christmas",
}


def _parse_holidays(raw: dict) -> dict:
    """Convert string-keyed holiday dict to date-keyed."""
    parsed = {}
    for date_str, name in raw.items():
        try:
            parsed[date.fromisoformat(date_str)] = name
        except ValueError:
            logger.warning(f"Invalid holiday date: {date_str}")
    return parsed


# Device categories that are expected to run 24/7
ALWAYS_ON_CATEGORIES = {"server", "network_switch", "ups", "router", "nas",
                         "camera", "biometric"}


class ContextEngine:
    """
    Production-grade context engine with timezone handling,
    thread-safe baselines, device-aware risk scoring, and
    holiday-excluded exponentially weighted baselines.

    Usage:
        engine = ContextEngine()
        features = engine.enrich(carbon_result, device_spec=optional)
    """

    def __init__(
        self,
        baseline_weeks: int = None,
        holidays_json_path: str = None,
    ):
        """
        Args:
            baseline_weeks: Weeks of history for rolling baseline
            holidays_json_path: Optional JSON file to load holidays from
        """
        self.baseline_weeks = baseline_weeks or settings.ROLLING_BASELINE_WEEKS

        # Fix #1: Timezone offset for UTC → local conversion
        self._tz_offset = timedelta(hours=settings.LOCAL_TIMEZONE_OFFSET_HOURS)

        # Fix #2: Thread-safe lock for baseline storage
        self._lock = threading.Lock()

        # Rolling baseline: (device, circuit, day, hour) → list[float]
        self._baselines: dict[tuple, List[float]] = defaultdict(list)

        # Fix #4: Track which readings were on holidays (excluded from baseline)
        self._holiday_dates_seen: set = set()

        # Min readings before baseline is reliable
        self._min_readings = settings.ROLLING_BASELINE_MIN_READINGS

        # Working hours config
        self._work_start = settings.WORKING_HOURS_START
        self._work_end = settings.WORKING_HOURS_END
        self._working_days = settings.WORKING_DAYS

        # Fix #6: Load holidays (default + optional JSON override)
        self._holidays = _parse_holidays(_DEFAULT_HOLIDAYS)
        if holidays_json_path:
            self._load_holidays_from_json(holidays_json_path)

        self._stats = {
            "total_enriched": 0,
            "baseline_available": 0,
            "baseline_missing": 0,
            "holiday_readings_excluded": 0,
        }

        logger.info(
            f"ContextEngine initialized: tz_offset={self._tz_offset}, "
            f"baseline_weeks={self.baseline_weeks}, "
            f"holidays_loaded={len(self._holidays)}, "
            f"working={self._work_start}-{self._work_end}"
        )

    def _load_holidays_from_json(self, path: str) -> None:
        """Fix #6: Load holidays from external JSON file."""
        try:
            with open(path, "r") as f:
                raw = json.load(f)
            new_holidays = _parse_holidays(raw)
            self._holidays.update(new_holidays)
            logger.info(f"Loaded {len(new_holidays)} holidays from {path}")
        except Exception as e:
            logger.warning(f"Failed to load holidays from {path}: {e}")

    # ══════════════════════════════════════════════════════════
    # MAIN ENTRY POINT
    # ══════════════════════════════════════════════════════════

    def enrich(
        self,
        carbon_result: CarbonResult,
        device_spec: Optional[DeviceSpec] = None,
    ) -> ContextFeatures:
        """
        Generate context features for a carbon result.

        Args:
            carbon_result: Output from the Carbon Math Engine
            device_spec: Optional device specs for device-aware risk

        Returns:
            ContextFeatures with temporal, baseline, and risk data
        """
        self._stats["total_enriched"] += 1

        # Use midpoint of reading window
        midpoint_utc = carbon_result.timestamp_start + (
            carbon_result.timestamp_end - carbon_result.timestamp_start
        ) / 2

        # Fix #1: Convert UTC → local time (IST)
        local_time = self._to_local(midpoint_utc)

        # ── Temporal Features ─────────────────────────────────
        hour = local_time.hour
        day_of_week = local_time.weekday()  # 0=Monday
        day_name = local_time.strftime("%A")
        is_weekend = day_of_week >= 5

        # ── Working Hours ─────────────────────────────────────
        is_working_hour = (
            self._work_start <= hour < self._work_end
            and day_of_week in self._working_days
        )

        # ── Shift Classification ──────────────────────────────
        shift = self._classify_shift(hour, day_of_week)

        # ── Holiday Check ─────────────────────────────────────
        reading_date = local_time.date()
        is_holiday, holiday_name = self._check_holiday(reading_date)
        if is_holiday:
            is_working_hour = False

        # ── Rolling Baseline ──────────────────────────────────
        wattage = carbon_result.calculation_details.get("input_wattage", 0)

        baseline, deviation = self._get_baseline_deviation(
            device_id=carbon_result.device_id,
            circuit_id=carbon_result.circuit_id,
            day_of_week=day_of_week,
            hour=hour,
            current_wattage=wattage,
            is_holiday=is_holiday,
        )

        # ── Context Risk Score (Fix #3: device-type aware) ────
        risk_score = self._compute_risk_score(
            is_working_hour=is_working_hour,
            is_weekend=is_weekend,
            is_holiday=is_holiday,
            shift=shift,
            deviation=deviation,
            wattage=wattage,
            device_spec=device_spec,
        )

        features = ContextFeatures(
            hour=hour,
            day_of_week=day_of_week,
            day_name=day_name,
            is_weekend=is_weekend,
            is_working_hour=is_working_hour,
            is_holiday=is_holiday,
            holiday_name=holiday_name,
            shift=shift,
            baseline_wattage=baseline,
            deviation_from_baseline=deviation,
            context_risk_score=round(risk_score, 3),
        )

        logger.debug(
            f"Context for {carbon_result.device_id}: "
            f"{day_name} {hour}:00 IST, shift={shift}, "
            f"risk={risk_score:.3f}, holiday={is_holiday}"
        )

        return features

    # ══════════════════════════════════════════════════════════
    # FIX #1: TIMEZONE CONVERSION
    # ══════════════════════════════════════════════════════════

    def _to_local(self, dt: datetime) -> datetime:
        """
        Convert a datetime to local time (IST by default).

        Handles both timezone-aware and naive datetimes.
        Naive datetimes are assumed to be UTC.
        """
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt + self._tz_offset

    # ══════════════════════════════════════════════════════════
    # SHIFT CLASSIFICATION
    # ══════════════════════════════════════════════════════════

    def _classify_shift(self, hour: int, day_of_week: int) -> str:
        """
        Classify local time into shift categories.

        Shifts (using local time — IST):
            working:       9 AM - 6 PM, Mon-Fri
            early_morning: 6 AM - 9 AM, Mon-Fri
            after_hours:   6 PM - 10 PM, Mon-Fri
            night:         10 PM - 6 AM, any day
            weekend_day:   6 AM - 10 PM, Sat-Sun
        """
        is_weekday = day_of_week in self._working_days

        if 22 <= hour or hour < 6:
            return "night"
        elif is_weekday:
            if self._work_start <= hour < self._work_end:
                return "working"
            elif 6 <= hour < self._work_start:
                return "early_morning"
            else:
                return "after_hours"
        else:
            return "weekend_day"

    # ══════════════════════════════════════════════════════════
    # HOLIDAY DETECTION
    # ══════════════════════════════════════════════════════════

    def _check_holiday(self, reading_date: date) -> Tuple[bool, Optional[str]]:
        """Check if date is a holiday."""
        holiday_name = self._holidays.get(reading_date)
        return (holiday_name is not None, holiday_name)

    def add_holiday(self, holiday_date: date, name: str) -> None:
        """Dynamically add a holiday (Fix #6)."""
        self._holidays[holiday_date] = name
        logger.info(f"Added holiday: {holiday_date} = {name}")

    def get_holidays(self) -> dict:
        """Get all loaded holidays."""
        return {d.isoformat(): n for d, n in sorted(self._holidays.items())}

    # ══════════════════════════════════════════════════════════
    # FIX #4 + #5: HOLIDAY-EXCLUDED EWMA BASELINE
    # ══════════════════════════════════════════════════════════

    def _get_baseline_deviation(
        self,
        device_id: str,
        circuit_id: Optional[str],
        day_of_week: int,
        hour: int,
        current_wattage: float,
        is_holiday: bool,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate exponentially weighted rolling baseline and deviation.

        Fix #4: Holiday readings are NOT added to the baseline.
        Fix #5: Uses EWMA instead of simple mean (recent data = higher weight).

        EWMA formula:
            baseline = α × current + (1-α) × previous_baseline
            α = 2 / (span + 1), span = number of readings

        Deviation:
            deviation = (current - baseline) / baseline
        """
        key = (device_id, circuit_id, day_of_week, hour)

        # Fix #4: Don't add holiday readings to baseline
        if is_holiday:
            self._stats["holiday_readings_excluded"] += 1
            logger.debug(
                f"Holiday reading excluded from baseline: "
                f"{device_id}/{circuit_id} day={day_of_week} hour={hour}"
            )
        else:
            # Fix #2: Thread-safe write
            with self._lock:
                self._baselines[key].append(current_wattage)
                # Trim to max window
                max_readings = self.baseline_weeks * 4
                if len(self._baselines[key]) > max_readings:
                    self._baselines[key] = self._baselines[key][-max_readings:]

        # Read baseline (thread-safe)
        with self._lock:
            history = list(self._baselines.get(key, []))

        if len(history) < self._min_readings:
            self._stats["baseline_missing"] += 1
            return None, None

        self._stats["baseline_available"] += 1

        # Fix #5: EWMA calculation (more recent = higher weight)
        baseline = self._compute_ewma(history)

        if baseline == 0:
            deviation = 0.0 if current_wattage == 0 else float("inf")
        else:
            deviation = (current_wattage - baseline) / baseline

        return round(baseline, 2), round(deviation, 4)

    def _compute_ewma(self, values: List[float]) -> float:
        """
        Compute Exponentially Weighted Moving Average.

        More recent values have higher weight.
        α = 2 / (span + 1) where span = min(len(values), 20)
        """
        if not values:
            return 0.0

        span = min(len(values), 20)
        alpha = 2.0 / (span + 1)

        ewma = values[0]
        for v in values[1:]:
            ewma = alpha * v + (1 - alpha) * ewma

        return ewma

    # ══════════════════════════════════════════════════════════
    # FIX #3: DEVICE-TYPE-AWARE RISK SCORING
    # ══════════════════════════════════════════════════════════

    def _compute_risk_score(
        self,
        is_working_hour: bool,
        is_weekend: bool,
        is_holiday: bool,
        shift: str,
        deviation: Optional[float],
        wattage: float,
        device_spec: Optional[DeviceSpec] = None,
    ) -> float:
        """
        Compute composite context risk score (0-1).

        Fix #3: Device-type aware — always-on devices (servers, UPS,
        routers) get reduced time-based risk since they're expected
        to run 24/7.

        Scoring logic:
            Working hours + normal deviation → 0.0
            After hours + modest deviation → 0.3
            Night/weekend + high deviation → 0.7
            Holiday/night + very high → 0.9+
            SERVER at night → 0.0 (expected)
        """
        risk = 0.0

        # Fix #3: Check if device is always-on
        is_always_on = False
        if device_spec and device_spec.category.lower() in ALWAYS_ON_CATEGORIES:
            is_always_on = True

        # ── Time-based risk ───────────────────────────────────
        shift_risk = {
            "working": 0.0,
            "early_morning": 0.15,
            "after_hours": 0.25,
            "weekend_day": 0.30,
            "night": 0.45,
        }
        time_risk = shift_risk.get(shift, 0.0)

        # Fix #3: Always-on devices get ZERO time-based risk
        # (it's normal for them to run at night/weekends)
        if is_always_on:
            time_risk = 0.0

        risk += time_risk

        if is_holiday and not is_always_on:
            risk += 0.20

        if is_weekend and not is_working_hour and not is_always_on:
            risk += 0.10

        # ── Deviation-based risk ──────────────────────────────
        if deviation is not None and deviation > 0:
            if deviation > 0.5:     # 50% above baseline
                risk += 0.15
            if deviation > 1.0:     # 100% above baseline
                risk += 0.15
            if deviation > 2.0:     # 200% above baseline
                risk += 0.15

        return min(risk, 1.0)

    # ══════════════════════════════════════════════════════════
    # STATISTICS & UTILITIES
    # ══════════════════════════════════════════════════════════

    def get_context_stats(self) -> dict:
        """Get context engine statistics."""
        total = self._stats["total_enriched"]
        with self._lock:
            timeslots = len(self._baselines)
        return {
            **self._stats,
            "baseline_coverage": round(
                self._stats["baseline_available"] / total, 4
            ) if total > 0 else 0,
            "tracked_timeslots": timeslots,
            "holidays_loaded": len(self._holidays),
            "timezone_offset": str(self._tz_offset),
        }

    def get_baseline(
        self,
        device_id: str,
        circuit_id: Optional[str],
        day_of_week: int,
        hour: int,
    ) -> Optional[float]:
        """Get current EWMA baseline for a device/timeslot."""
        key = (device_id, circuit_id, day_of_week, hour)
        with self._lock:
            history = list(self._baselines.get(key, []))
        if len(history) < self._min_readings:
            return None
        return round(self._compute_ewma(history), 2)

    def reset_baselines(self, device_id: str = None) -> None:
        """Reset baselines for a device or all devices."""
        with self._lock:
            if device_id:
                keys = [k for k in self._baselines if k[0] == device_id]
                for k in keys:
                    del self._baselines[k]
                logger.info(f"Reset baselines for {device_id}")
            else:
                self._baselines.clear()
                logger.info("Reset all baselines")


# ── Module-level singleton ────────────────────────────────────────
context_engine = ContextEngine()
