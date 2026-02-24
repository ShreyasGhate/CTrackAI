"""
CTrackAI — Severity Scoring Engine (Stage 7)

Assigns severity levels to detected anomalies using a weighted
multi-factor formula.

Severity Formula:
    score = (w_delta × delta_factor) +
            (w_context × context_penalty) +
            (w_anomaly × anomaly_score)

Severity Bands:
    LOW:      0.00 - 0.30   (log only, no alert)
    MEDIUM:   0.31 - 0.55   (email notification)
    HIGH:     0.56 - 0.80   (SMS/push alert)
    CRITICAL: 0.81 - 1.00   (immediate action required)

Factors:
    - Delta wattage: how far above expected (normalized)
    - Context penalty: time-of-day risk multiplier
    - Anomaly score: detection confidence from Stage 6
    - Cost factor: estimated excess CO₂ cost
"""

from typing import Optional
from loguru import logger

from models.schemas import (
    CarbonResult,
    ContextFeatures,
    DeviceSpec,
    AnomalyResult,
    SeverityResult,
    SeverityLevel,
)
from config.settings import settings


class SeverityScoringEngine:
    """
    Multi-factor severity scoring for anomalies.

    Usage:
        scorer = SeverityScoringEngine()
        result = scorer.score(carbon, context, anomaly, device_spec)
    """

    def __init__(self):
        self._w_delta = settings.SEVERITY_WEIGHT_DELTA_WATTAGE
        self._w_context = settings.SEVERITY_WEIGHT_CONTEXT_PENALTY
        self._w_anomaly = settings.SEVERITY_WEIGHT_ANOMALY_SCORE

        # Severity bands from config
        self._low_max = settings.SEVERITY_LOW_MAX
        self._medium_max = settings.SEVERITY_MEDIUM_MAX
        self._high_max = settings.SEVERITY_HIGH_MAX

        self._stats = {
            "total_scored": 0,
            "by_level": {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0},
        }

        logger.info(
            f"SeverityScoringEngine initialized: "
            f"weights=({self._w_delta}, {self._w_context}, {self._w_anomaly}), "
            f"bands=[{self._low_max}/{self._medium_max}/{self._high_max}]"
        )

    def score(
        self,
        carbon: CarbonResult,
        context: ContextFeatures,
        anomaly: AnomalyResult,
        device_spec: Optional[DeviceSpec] = None,
    ) -> SeverityResult:
        """
        Calculate severity score for a detected anomaly.

        Args:
            carbon: Carbon calculation result
            context: Context features
            anomaly: Anomaly detection result
            device_spec: Optional device specifications

        Returns:
            SeverityResult with score, level, and contributing factors
        """
        self._stats["total_scored"] += 1

        wattage = carbon.calculation_details.get("input_wattage", 0)

        # ── Factor 1: Delta Wattage ───────────────────────────
        delta_factor = self._calc_delta_factor(
            wattage, context, device_spec
        )

        # ── Factor 2: Context Penalty ─────────────────────────
        context_penalty = self._calc_context_penalty(context)

        # ── Factor 3: Anomaly Score ───────────────────────────
        anomaly_score = anomaly.combined_score

        # ── Weighted Combination ──────────────────────────────
        severity_score = (
            self._w_delta * delta_factor +
            self._w_context * context_penalty +
            self._w_anomaly * anomaly_score
        )
        severity_score = round(min(max(severity_score, 0.0), 1.0), 4)

        # ── Severity Level ────────────────────────────────────
        level = self._score_to_level(severity_score)

        # ── Anomaly-type boost ────────────────────────────────
        # Certain types intrinsically warrant higher severity
        if anomaly.anomaly_type == "zero_always_on":
            severity_score = max(severity_score, 0.70)
            level = self._score_to_level(severity_score)

        self._stats["by_level"][level.value] += 1

        contributing = {
            "delta_wattage": round(delta_factor, 4),
            "context_penalty": round(context_penalty, 4),
            "anomaly_score": round(anomaly_score, 4),
        }

        result = SeverityResult(
            severity_score=severity_score,
            severity_level=level,
            contributing_factors=contributing,
        )

        logger.debug(
            f"Severity for {carbon.device_id}: "
            f"{level.value} ({severity_score:.3f}) — "
            f"Δ={delta_factor:.2f}, ctx={context_penalty:.2f}, "
            f"anom={anomaly_score:.2f}"
        )

        return result

    def _calc_delta_factor(
        self,
        wattage: float,
        context: ContextFeatures,
        device_spec: Optional[DeviceSpec],
    ) -> float:
        """
        Calculate how far the wattage is from expected.

        Uses baseline first, then device spec as fallback.
        Normalized to 0-1 range.
        """
        # Try baseline comparison
        if context.baseline_wattage and context.baseline_wattage > 0:
            delta = abs(wattage - context.baseline_wattage)
            return min(delta / (context.baseline_wattage * 2), 1.0)

        # Fallback: device spec comparison
        if device_spec and device_spec.rated_wattage > 0:
            if wattage > device_spec.rated_wattage:
                excess = wattage - device_spec.rated_wattage
                return min(excess / device_spec.rated_wattage, 1.0)
            return 0.0

        # No reference — use absolute scale (normalized to 5kW)
        return min(wattage / 5000.0, 1.0)

    def _calc_context_penalty(self, context: ContextFeatures) -> float:
        """
        Calculate context-based penalty.

        Higher penalty for anomalies in suspicious contexts.
        """
        penalty = context.context_risk_score

        # Extra penalty for high baseline deviation
        if (context.deviation_from_baseline is not None
                and context.deviation_from_baseline > 1.5):
            penalty += 0.2

        return min(penalty, 1.0)

    def _score_to_level(self, score: float) -> SeverityLevel:
        """Map severity score to severity level."""
        if score <= self._low_max:
            return SeverityLevel.LOW
        elif score <= self._medium_max:
            return SeverityLevel.MEDIUM
        elif score <= self._high_max:
            return SeverityLevel.HIGH
        else:
            return SeverityLevel.CRITICAL

    def get_severity_stats(self) -> dict:
        """Get severity scoring statistics."""
        return {
            **self._stats,
            "weights": {
                "delta": self._w_delta,
                "context": self._w_context,
                "anomaly": self._w_anomaly,
            },
        }


# ── Module-level singleton ────────────────────────────────────────
severity_engine = SeverityScoringEngine()
