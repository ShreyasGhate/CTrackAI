"""
CTrackAI — Forecasting Engine (Stage 9 — Production Grade)

Provides energy and CO₂ forecasts using a 3-tier approach:

Tier 1 — Prophet (primary):
    Facebook Prophet for time-series decomposition.
    Handles seasonality, holidays, and trend.
    Requires: 2+ weeks of 15-min data (~1344 readings).

Tier 2 — ARIMA fallback:
    AutoARIMA from pmdarima. Uses seasonal mode when enough
    data exists (3+ cycles of m), otherwise non-seasonal.
    Works with 100+ readings.

Tier 3 — Linear EWMA fallback:
    Exponentially weighted moving average for very short series.

Production Features:
    - Thread-safe time series storage
    - Graceful cascade through tiers
    - Smart ARIMA seasonal/non-seasonal selection
    - Negative forecast clamping (energy can't be < 0)
    - All tiers produce confidence intervals
"""

import threading
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Tuple

import numpy as np
from loguru import logger

from models.schemas import ForecastResult
from config.settings import settings


class ForecastEngine:
    """
    3-tier forecasting: Prophet → ARIMA → Linear EWMA.

    Thread-safe per-device time series storage.

    Usage:
        engine = ForecastEngine()
        engine.add_observation(device_id, timestamp, energy_kwh, co2_kg)
        forecast = engine.forecast(device_id)
    """

    PROPHET_MIN_READINGS = 1344    # ~2 weeks of 15-min data
    ARIMA_MIN_READINGS = 100       # ~25 hours
    LINEAR_MIN_READINGS = 10       # Minimum for any forecast
    ARIMA_SEASONAL_MIN_CYCLES = 3  # Need 3+ seasonal cycles for seasonal

    def __init__(self):
        self._lock = threading.Lock()

        # Per-device time series
        self._series: dict[str, List[Tuple[datetime, float, float]]] = \
            defaultdict(list)
        self._max_history = 8064  # 12 weeks of 15-min data

        self._prophet_available = self._check_prophet()
        self._arima_available = self._check_arima()

        self._stats = {
            "total_forecasts": 0,
            "prophet_used": 0,
            "arima_used": 0,
            "arima_seasonal": 0,
            "arima_nonseasonal": 0,
            "linear_used": 0,
            "insufficient_data": 0,
        }

        logger.info(
            f"ForecastEngine initialized: "
            f"prophet={self._prophet_available}, "
            f"arima={self._arima_available}"
        )

    def _check_prophet(self) -> bool:
        try:
            from prophet import Prophet  # noqa: F401
            return True
        except ImportError:
            logger.info("Prophet not installed — will use ARIMA/linear")
            return False

    def _check_arima(self) -> bool:
        try:
            import pmdarima  # noqa: F401
            return True
        except ImportError:
            logger.info("pmdarima not installed — will use linear")
            return False

    # ══════════════════════════════════════════════════════════
    # DATA INPUT (Thread-Safe)
    # ══════════════════════════════════════════════════════════

    def add_observation(
        self,
        device_id: str,
        timestamp: datetime,
        energy_kwh: float,
        co2_kg: float,
    ) -> None:
        """Thread-safe observation addition."""
        with self._lock:
            self._series[device_id].append((timestamp, energy_kwh, co2_kg))
            if len(self._series[device_id]) > self._max_history:
                self._series[device_id] = \
                    self._series[device_id][-self._max_history:]

    def forecast(self, device_id: str) -> ForecastResult:
        """
        Generate forecast for a device.

        Cascade: Prophet → ARIMA → Linear → insufficient
        """
        self._stats["total_forecasts"] += 1

        with self._lock:
            series = list(self._series.get(device_id, []))

        n = len(series)

        if n < self.LINEAR_MIN_READINGS:
            self._stats["insufficient_data"] += 1
            return ForecastResult(
                sufficient_history=False,
                forecast_model="none",
                forecast_timestamp=datetime.now(timezone.utc),
            )

        if self._prophet_available and n >= self.PROPHET_MIN_READINGS:
            return self._forecast_prophet(device_id, series)

        if self._arima_available and n >= self.ARIMA_MIN_READINGS:
            return self._forecast_arima(device_id, series)

        return self._forecast_linear(device_id, series)

    # ══════════════════════════════════════════════════════════
    # TIER 1: PROPHET
    # ══════════════════════════════════════════════════════════

    def _forecast_prophet(
        self,
        device_id: str,
        series: List[Tuple[datetime, float, float]],
    ) -> ForecastResult:
        """Prophet forecast with daily/weekly seasonality."""
        try:
            import pandas as pd
            from prophet import Prophet

            data = [(ts, kwh) for ts, kwh, _ in series]
            df = pd.DataFrame(data, columns=["ds", "y"])

            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                changepoint_prior_scale=settings.FORECAST_CHANGEPOINT_PRIOR,
            )
            model.fit(df)

            future = model.make_future_dataframe(periods=672, freq="15min")
            pred = model.predict(future)

            # Only use future predictions (after last training timestamp)
            last_ts = df["ds"].max()
            future_pred = pred[pred["ds"] > last_ts]

            next_day = future_pred.head(96)   # First 96 = next 24h
            next_week = future_pred.head(672)  # First 672 = next 7 days

            ef = settings.EMISSION_FACTOR
            self._stats["prophet_used"] += 1

            return self._build_result(
                model_name="prophet",
                next_day_kwh=max(0, float(next_day["yhat"].sum())),
                next_week_kwh=max(0, float(next_week["yhat"].sum())),
                ci_lower=max(0, float(next_day["yhat_lower"].sum())),
                ci_upper=float(next_day["yhat_upper"].sum()),
                emission_factor=ef,
            )

        except Exception as e:
            logger.warning(f"Prophet failed for {device_id}: {e}")
            if self._arima_available:
                return self._forecast_arima(device_id, series)
            return self._forecast_linear(device_id, series)

    # ══════════════════════════════════════════════════════════
    # TIER 2: ARIMA (Smart Seasonal Selection)
    # ══════════════════════════════════════════════════════════

    def _forecast_arima(
        self,
        device_id: str,
        series: List[Tuple[datetime, float, float]],
    ) -> ForecastResult:
        """
        Auto-ARIMA with smart seasonal mode selection.

        If we have 3+ full daily cycles (288+ readings), use
        seasonal ARIMA with m=96. Otherwise, use non-seasonal
        ARIMA which is much faster and works with less data.
        """
        try:
            import pmdarima as pm

            values = np.array([kwh for _, kwh, _ in series])
            n = len(values)

            # Smart seasonal selection
            m = 96  # Daily cycle = 96 × 15 min
            use_seasonal = n >= m * self.ARIMA_SEASONAL_MIN_CYCLES

            if use_seasonal:
                self._stats["arima_seasonal"] += 1
                model = pm.auto_arima(
                    values,
                    seasonal=True,
                    m=m,
                    suppress_warnings=True,
                    stepwise=True,
                    error_action="ignore",
                    max_order=5,
                )
            else:
                self._stats["arima_nonseasonal"] += 1
                model = pm.auto_arima(
                    values,
                    seasonal=False,
                    suppress_warnings=True,
                    stepwise=True,
                    error_action="ignore",
                    max_order=5,
                )

            forecasts, conf_int = model.predict(
                n_periods=672, return_conf_int=True
            )

            ef = settings.EMISSION_FACTOR
            self._stats["arima_used"] += 1

            mode = "arima_seasonal" if use_seasonal else "arima_nonseasonal"

            return self._build_result(
                model_name=mode,
                next_day_kwh=max(0, float(np.sum(forecasts[:96]))),
                next_week_kwh=max(0, float(np.sum(forecasts[:672]))),
                ci_lower=max(0, float(np.sum(conf_int[:96, 0]))),
                ci_upper=float(np.sum(conf_int[:96, 1])),
                emission_factor=ef,
            )

        except Exception as e:
            logger.warning(f"ARIMA failed for {device_id}: {e}")
            return self._forecast_linear(device_id, series)

    # ══════════════════════════════════════════════════════════
    # TIER 3: LINEAR EWMA FALLBACK
    # ══════════════════════════════════════════════════════════

    def _forecast_linear(
        self,
        device_id: str,
        series: List[Tuple[datetime, float, float]],
    ) -> ForecastResult:
        """EWMA-based forecast for short series."""
        values = np.array([kwh for _, kwh, _ in series])

        span = min(len(values), 20)
        alpha = 2.0 / (span + 1)
        ewma = values[0]
        for v in values[1:]:
            ewma = alpha * v + (1 - alpha) * ewma

        avg_kwh = max(0, float(ewma))  # Clamp non-negative

        next_day_kwh = avg_kwh * 96
        next_week_kwh = avg_kwh * 672

        ef = settings.EMISSION_FACTOR

        std = float(np.std(values[-span:])) if len(values) > 1 else 0.0
        ci_range = std * 96

        self._stats["linear_used"] += 1

        return self._build_result(
            model_name="linear_ewma",
            next_day_kwh=next_day_kwh,
            next_week_kwh=next_week_kwh,
            ci_lower=max(0, next_day_kwh - ci_range),
            ci_upper=next_day_kwh + ci_range,
            emission_factor=ef,
        )

    # ══════════════════════════════════════════════════════════
    # RESULT BUILDER (DRY)
    # ══════════════════════════════════════════════════════════

    def _build_result(
        self,
        model_name: str,
        next_day_kwh: float,
        next_week_kwh: float,
        ci_lower: float,
        ci_upper: float,
        emission_factor: float,
    ) -> ForecastResult:
        """Build a ForecastResult with consistent rounding and CO₂ conversion."""
        return ForecastResult(
            next_day_estimate_kwh=round(next_day_kwh, 4),
            next_day_estimate_kg_co2=round(next_day_kwh * emission_factor, 4),
            next_week_estimate_kwh=round(next_week_kwh, 4),
            next_week_estimate_kg_co2=round(
                next_week_kwh * emission_factor, 4
            ),
            confidence_interval_lower=round(ci_lower, 4),
            confidence_interval_upper=round(ci_upper, 4),
            forecast_model=model_name,
            forecast_timestamp=datetime.now(timezone.utc),
            sufficient_history=True,
        )

    # ══════════════════════════════════════════════════════════
    # STATISTICS
    # ══════════════════════════════════════════════════════════

    def get_forecast_stats(self) -> dict:
        """Get forecasting statistics."""
        with self._lock:
            tracked = list(self._series.keys())
            lengths = {k: len(v) for k, v in self._series.items()}
        return {
            **self._stats,
            "prophet_available": self._prophet_available,
            "arima_available": self._arima_available,
            "tracked_devices": tracked,
            "series_lengths": lengths,
        }

    def get_series_length(self, device_id: str) -> int:
        with self._lock:
            return len(self._series.get(device_id, []))


# ── Module-level singleton ────────────────────────────────────────
forecast_engine = ForecastEngine()
