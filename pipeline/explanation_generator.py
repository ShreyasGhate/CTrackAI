"""
CTrackAI — Explanation Generator (Stage 8)

Generates human-readable explanations for anomalies using:
    1. Template engine: Pre-built explanation templates per anomaly type
    2. SHAP integration: Feature importance from anomaly detection
       (falls back to rule-based explanations if SHAP unavailable)
    3. Actionable recommendations: What to do about each anomaly type

Every explanation includes:
    - One-sentence summary (e.g., "Lab PC_Row_A spiked to 400W,
      exceeding its 200W rating by 100%")
    - Top contributing features
    - Recommended action
    - Carbon impact estimate
"""

from typing import Optional, List, Dict, Any
from loguru import logger

from models.schemas import (
    CarbonResult,
    ContextFeatures,
    DeviceSpec,
    AnomalyResult,
    SeverityResult,
    SeverityLevel,
    Explanation,
)
from config.settings import settings


# ══════════════════════════════════════════════════════════════
# EXPLANATION TEMPLATES
# ══════════════════════════════════════════════════════════════

TEMPLATES = {
    "spike": {
        "summary": (
            "{device_id} spiked to {wattage:.0f}W, exceeding its "
            "{rated}W rating by {overshoot:.0f}%. This caused an "
            "estimated {excess_co2:.4f} kg excess CO₂."
        ),
        "action": (
            "Check for device malfunction, overloaded circuits, or "
            "unauthorized high-power equipment connected to this circuit. "
            "If persistent, schedule maintenance inspection."
        ),
    },
    "after_hours": {
        "summary": (
            "{device_id} was consuming {wattage:.0f}W during "
            "{shift} ({day_name} at {hour}:00). Devices on this circuit "
            "should normally be powered down or on standby during this period."
        ),
        "action": (
            "Verify if the device was intentionally left on. "
            "Consider implementing automatic power-down schedules "
            "or smart power strips to reduce after-hours waste."
        ),
    },
    "baseline_breach": {
        "summary": (
            "{device_id} consumed {wattage:.0f}W, which is "
            "{deviation:.0f}% above its normal baseline of "
            "{baseline:.0f}W at this time."
        ),
        "action": (
            "Investigate if new equipment was added or if there's "
            "a change in usage patterns. Review if the baseline "
            "needs recalibration."
        ),
    },
    "zero_always_on": {
        "summary": (
            "{device_id} ({device_name}) reported {wattage:.1f}W — "
            "this {category} device should always have power. "
            "Possible power failure or sensor disconnect."
        ),
        "action": (
            "URGENT: Check physical power connection to the device. "
            "Verify UPS status. Inspect sensor wiring. This could "
            "indicate a critical infrastructure failure."
        ),
    },
    "excessive_idle": {
        "summary": (
            "{device_id} was drawing only {wattage:.0f}W during "
            "working hours — near its idle threshold of {idle}W. "
            "Expected active usage at this time."
        ),
        "action": (
            "Check if the device is operating correctly. It may be "
            "in sleep mode, disconnected, or experiencing a fault. "
            "Verify user activity on this device."
        ),
    },
    "statistical": {
        "summary": (
            "{device_id} was flagged by statistical models with "
            "anomaly score {anomaly_score:.2f}. The power consumption "
            "of {wattage:.0f}W deviates from expected patterns."
        ),
        "action": (
            "Monitor this device — the anomaly was detected by "
            "machine learning models rather than explicit rules. "
            "If it persists, investigate the usage pattern change."
        ),
    },
}


class ExplanationGenerator:
    """
    Generates human-readable explanations for anomalies.

    Usage:
        generator = ExplanationGenerator()
        explanation = generator.explain(
            carbon, context, anomaly, severity, device_spec
        )
    """

    def __init__(self):
        self._shap_available = False
        self._check_shap()
        self._stats = {
            "total_explanations": 0,
            "shap_used": 0,
            "template_used": 0,
            "by_type": {},
        }
        logger.info(
            f"ExplanationGenerator initialized: "
            f"shap_available={self._shap_available}"
        )

    def _check_shap(self):
        """Check if SHAP library is available."""
        try:
            import shap  # noqa: F401
            self._shap_available = True
        except ImportError:
            self._shap_available = False
            logger.info(
                "SHAP not available — using rule-based explanations"
            )

    def explain(
        self,
        carbon: CarbonResult,
        context: ContextFeatures,
        anomaly: AnomalyResult,
        severity: SeverityResult,
        device_spec: Optional[DeviceSpec] = None,
    ) -> Explanation:
        """
        Generate explanation for an anomaly.

        Args:
            carbon: Carbon result
            context: Context features
            anomaly: Anomaly detection result
            severity: Severity scoring result
            device_spec: Optional device specifications

        Returns:
            Explanation with summary, features, and template info
        """
        self._stats["total_explanations"] += 1

        wattage = carbon.calculation_details.get("input_wattage", 0)
        anomaly_type = (anomaly.anomaly_type or "statistical").split(" (")[0]

        # Track by type
        self._stats["by_type"][anomaly_type] = \
            self._stats["by_type"].get(anomaly_type, 0) + 1

        # ── Generate Summary ──────────────────────────────────
        summary = self._render_template(
            anomaly_type, carbon, context, anomaly, device_spec, wattage
        )

        # ── Add severity context ──────────────────────────────
        severity_prefix = {
            SeverityLevel.LOW: "ℹ️",
            SeverityLevel.MEDIUM: "⚠️",
            SeverityLevel.HIGH: "🔴",
            SeverityLevel.CRITICAL: "🚨",
        }
        prefix = severity_prefix.get(severity.severity_level, "")
        summary = f"{prefix} [{severity.severity_level.value}] {summary}"

        # ── Top Contributing Features ─────────────────────────
        top_features = self._get_top_features(
            anomaly, severity, context, device_spec, wattage
        )

        # ── Determine template used ───────────────────────────
        template_used = f"template_{anomaly_type}"
        shap_used = False

        if self._shap_available:
            # In production, we'd compute SHAP values here
            shap_used = False  # Placeholder until IF model is trained
            self._stats["template_used"] += 1
        else:
            self._stats["template_used"] += 1

        result = Explanation(
            summary=summary,
            top_features=top_features,
            template_used=template_used,
            shap_available=shap_used,
        )

        logger.debug(
            f"Explanation for {carbon.device_id}: "
            f"[{severity.severity_level.value}] {anomaly_type}"
        )

        return result

    def _render_template(
        self,
        anomaly_type: str,
        carbon: CarbonResult,
        context: ContextFeatures,
        anomaly: AnomalyResult,
        device_spec: Optional[DeviceSpec],
        wattage: float,
    ) -> str:
        """Render the explanation template with actual values."""
        template = TEMPLATES.get(anomaly_type, TEMPLATES["statistical"])

        # Build template variables
        rated = device_spec.rated_wattage if device_spec else wattage
        overshoot = ((wattage - rated) / rated * 100) if rated > 0 else 0
        excess_co2 = 0.0
        if device_spec and wattage > device_spec.rated_wattage:
            # Use actual window duration, not hardcoded
            window_hours = (
                carbon.timestamp_end - carbon.timestamp_start
            ).total_seconds() / 3600.0
            excess_kwh = (
                (wattage - device_spec.rated_wattage) * window_hours / 1000
            )
            excess_co2 = excess_kwh * carbon.emission_factor

        try:
            summary = template["summary"].format(
                device_id=carbon.device_id,
                wattage=wattage,
                rated=rated,
                overshoot=max(overshoot, 0),
                excess_co2=excess_co2,
                shift=context.shift,
                day_name=context.day_name,
                hour=context.hour,
                deviation=(
                    (context.deviation_from_baseline or 0) * 100
                ),
                baseline=context.baseline_wattage or 0,
                device_name=(
                    device_spec.device_name if device_spec else "Unknown"
                ),
                category=(
                    device_spec.category if device_spec else "unknown"
                ),
                idle=device_spec.idle_wattage if device_spec else 0,
                anomaly_score=anomaly.combined_score,
            )
        except (KeyError, ValueError) as e:
            logger.warning(f"Template rendering error: {e}")
            summary = (
                f"{carbon.device_id} anomaly: {wattage:.0f}W, "
                f"type={anomaly_type}, score={anomaly.combined_score:.2f}"
            )

        # Append action
        action = template.get("action", "Monitor and investigate.")
        summary += f"\n\nRecommended action: {action}"

        return summary

    def _get_top_features(
        self,
        anomaly: AnomalyResult,
        severity: SeverityResult,
        context: ContextFeatures,
        device_spec: Optional[DeviceSpec],
        wattage: float,
    ) -> List[Dict[str, Any]]:
        """
        Get the top contributing features for the anomaly.

        Falls back to rule-based feature extraction if SHAP
        isn't available.
        """
        features = []

        # Severity contributing factors
        for factor_name, factor_value in severity.contributing_factors.items():
            if factor_value > 0.1:
                features.append({
                    "feature": factor_name,
                    "importance": round(factor_value, 3),
                    "value": factor_value,
                    "source": "severity_model",
                })

        # Context features
        if context.context_risk_score > 0.2:
            features.append({
                "feature": "context_risk",
                "importance": round(context.context_risk_score, 3),
                "value": f"shift={context.shift}, risk={context.context_risk_score}",
                "source": "context_engine",
            })

        if (context.deviation_from_baseline is not None
                and abs(context.deviation_from_baseline) > 0.5):
            features.append({
                "feature": "baseline_deviation",
                "importance": round(
                    min(abs(context.deviation_from_baseline) / 3, 1.0), 3
                ),
                "value": f"{context.deviation_from_baseline:.1%}",
                "source": "context_engine",
            })

        # Device spec features
        if device_spec and device_spec.rated_wattage > 0:
            load = wattage / device_spec.rated_wattage
            if load > 1.0:
                features.append({
                    "feature": "load_factor",
                    "importance": round(min((load - 1.0), 1.0), 3),
                    "value": f"{load:.1%} of rated",
                    "source": "device_spec",
                })

        # Sort by importance, take top N
        features.sort(key=lambda x: x["importance"], reverse=True)
        return features[:settings.SHAP_TOP_FEATURES]

    def get_explanation_stats(self) -> dict:
        """Get explanation generator statistics."""
        return {
            **self._stats,
            "shap_available": self._shap_available,
        }


# ── Module-level singleton ────────────────────────────────────────
explanation_generator = ExplanationGenerator()
