"""
CTrackAI — Pipeline Orchestrator (Stage 11)

Orchestrates the full 10-stage ML pipeline:

    AggregatedReading → Data Quality → RAG Lookup → Carbon Math →
    Context → Anomaly Detection → Severity → Explanation →
    Forecast → Traceability Log

Single entry point that connects all pipeline stages.
The API and MQTT handlers call this.

Production Features:
    - Graceful degradation at every stage
    - Processing time tracking
    - Structured PipelineOutput
    - Unique log IDs for traceability
"""

import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from models.schemas import (
    AggregatedReading,
    QualityCheckedReading,
    CarbonResult,
    ContextFeatures,
    AnomalyResult,
    SeverityResult,
    SeverityLevel,
    Explanation,
    ForecastResult,
    PipelineOutput,
    DataQualityStatus,
    DeviceSpec,
)
from config.settings import settings


class PipelineOrchestrator:
    """
    Full pipeline orchestration — single entry point.

    Usage:
        orchestrator = PipelineOrchestrator()
        result = orchestrator.process(aggregated_reading)
    """

    def __init__(self):
        self._stats = {
            "total_processed": 0,
            "errors": 0,
            "degraded": 0,
            "avg_processing_ms": 0.0,
        }
        logger.info("PipelineOrchestrator initialized")

    def process(self, reading: AggregatedReading) -> PipelineOutput:
        """
        Process an aggregated reading through the full pipeline.

        Args:
            reading: AggregatedReading from the ingestion layer

        Returns:
            PipelineOutput with full results from all stages
        """
        start_time = time.perf_counter()
        self._stats["total_processed"] += 1

        log_id = f"CTK-{uuid.uuid4().hex[:12]}"
        is_degraded = False
        degradation_notes = []

        # ══════════════════════════════════════════════════════
        # STAGE 2: DATA QUALITY
        # ══════════════════════════════════════════════════════
        quality_result: Optional[QualityCheckedReading] = None
        try:
            from pipeline.data_quality import data_quality_handler
            quality_result = data_quality_handler.check(reading)
        except Exception as e:
            logger.error(f"[{log_id}] Data quality failed: {e}")
            is_degraded = True
            degradation_notes.append(f"data_quality: {e}")
            # Minimal fallback
            quality_result = QualityCheckedReading(
                device_id=reading.device_id,
                circuit_id=reading.circuit_id,
                timestamp_start=reading.timestamp_start,
                timestamp_end=reading.timestamp_end,
                original_wattage=reading.avg_wattage,
                corrected_wattage=reading.avg_wattage,
                energy_kwh=reading.energy_kwh,
                quality_status=DataQualityStatus.FLAGGED,
                quality_flags=[f"dq_bypass: {str(e)[:100]}"],
            )

        # ══════════════════════════════════════════════════════
        # STAGE 3: RAG DEVICE LOOKUP
        # ══════════════════════════════════════════════════════
        device_spec: Optional[DeviceSpec] = None
        try:
            from rag.retriever import device_retriever
            # .get() returns (Optional[DeviceSpec], similarity_score)
            device_spec, _similarity = device_retriever.get(
                quality_result.device_id
            )
        except Exception as e:
            logger.warning(f"[{log_id}] RAG lookup failed: {e}")
            degradation_notes.append(f"rag: {e}")

        # ══════════════════════════════════════════════════════
        # STAGE 4: CARBON MATH
        # ══════════════════════════════════════════════════════
        carbon_result: Optional[CarbonResult] = None
        try:
            from pipeline.carbon_math import carbon_engine
            carbon_result = carbon_engine.calculate(
                quality_result, device_spec
            )
        except Exception as e:
            logger.error(f"[{log_id}] Carbon math failed: {e}")
            is_degraded = True
            degradation_notes.append(f"carbon_math: {e}")

        if carbon_result is None:
            elapsed = (time.perf_counter() - start_time) * 1000
            self._stats["errors"] += 1
            self._stats["degraded"] += 1
            return self._build_error_output(
                log_id, reading, elapsed, degradation_notes
            )

        # ══════════════════════════════════════════════════════
        # STAGE 5: CONTEXT ENGINE
        # ══════════════════════════════════════════════════════
        context: Optional[ContextFeatures] = None
        try:
            from pipeline.context_engine import context_engine
            context = context_engine.enrich(carbon_result, device_spec)
        except Exception as e:
            logger.warning(f"[{log_id}] Context engine failed: {e}")
            is_degraded = True
            degradation_notes.append(f"context: {e}")
            now = datetime.now(timezone.utc)
            context = ContextFeatures(
                hour=now.hour,
                day_of_week=now.weekday(),
                day_name=now.strftime("%A"),
                is_weekend=now.weekday() >= 5,
                is_working_hour=False,
                is_holiday=False,
                shift="unknown",
                context_risk_score=0.0,
            )

        # ══════════════════════════════════════════════════════
        # STAGE 6: ANOMALY DETECTION
        # ══════════════════════════════════════════════════════
        anomaly: Optional[AnomalyResult] = None
        try:
            from pipeline.anomaly_detection import anomaly_detector
            anomaly = anomaly_detector.detect(
                carbon_result, context, device_spec
            )
        except Exception as e:
            logger.warning(f"[{log_id}] Anomaly detection failed: {e}")
            is_degraded = True
            degradation_notes.append(f"anomaly: {e}")
            anomaly = AnomalyResult(
                is_anomaly=False,
                combined_score=0.0,
                layer_results=[],
            )

        # ══════════════════════════════════════════════════════
        # STAGE 7: SEVERITY SCORING (only if anomaly)
        # ══════════════════════════════════════════════════════
        severity: Optional[SeverityResult] = None
        if anomaly.is_anomaly:
            try:
                from pipeline.severity_scoring import severity_engine
                severity = severity_engine.score(
                    carbon_result, context, anomaly, device_spec
                )
            except Exception as e:
                logger.warning(f"[{log_id}] Severity scoring failed: {e}")
                degradation_notes.append(f"severity: {e}")

        # ══════════════════════════════════════════════════════
        # STAGE 8: EXPLANATION (only if anomaly)
        # ══════════════════════════════════════════════════════
        explanation: Optional[Explanation] = None
        if anomaly.is_anomaly and severity:
            try:
                from pipeline.explanation_generator import (
                    explanation_generator,
                )
                explanation = explanation_generator.explain(
                    carbon_result, context, anomaly, severity, device_spec
                )
            except Exception as e:
                logger.warning(f"[{log_id}] Explanation gen failed: {e}")
                degradation_notes.append(f"explanation: {e}")

        # ══════════════════════════════════════════════════════
        # STAGE 9: FORECAST UPDATE
        # ══════════════════════════════════════════════════════
        try:
            from pipeline.forecast_engine import forecast_engine
            forecast_engine.add_observation(
                carbon_result.device_id,
                carbon_result.timestamp_start,
                carbon_result.energy_kwh,
                carbon_result.co2_kg,
            )
        except Exception as e:
            logger.warning(f"[{log_id}] Forecast update failed: {e}")
            degradation_notes.append(f"forecast: {e}")

        # ══════════════════════════════════════════════════════
        # BUILD OUTPUT
        # ══════════════════════════════════════════════════════
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        wattage = carbon_result.calculation_details.get("input_wattage", 0)

        output = PipelineOutput(
            log_id=log_id,
            timestamp=carbon_result.timestamp_start,
            device_id=carbon_result.device_id,
            circuit_id=quality_result.circuit_id,
            sensor_reading_watts=wattage,
            energy_kwh=carbon_result.energy_kwh,
            co2_kg=carbon_result.co2_kg,
            emission_factor_used=carbon_result.emission_factor,
            data_quality_status=quality_result.quality_status,
            quality_flags=quality_result.quality_flags,
            anomaly_detected=anomaly.is_anomaly,
            anomaly_score=anomaly.combined_score if anomaly else None,
            anomaly_type=anomaly.anomaly_type if anomaly else None,
            severity=severity.severity_level if severity else None,
            severity_score=severity.severity_score if severity else None,
            explanation=explanation.summary[:500] if explanation else None,
            context=context,
            is_degraded=is_degraded,
            processing_time_ms=elapsed_ms,
        )

        # ══════════════════════════════════════════════════════
        # STAGE 10: TRACEABILITY LOG
        # ══════════════════════════════════════════════════════
        try:
            from pipeline.traceability import trace_logger
            trace_logger.log_result(output)
        except Exception as e:
            logger.warning(f"[{log_id}] Traceability log failed: {e}")

        # Update running stats
        n = self._stats["total_processed"]
        old_avg = self._stats["avg_processing_ms"]
        self._stats["avg_processing_ms"] = round(
            old_avg + (elapsed_ms - old_avg) / n, 2
        )
        if is_degraded:
            self._stats["degraded"] += 1

        logger.debug(
            f"[{log_id}] Pipeline complete: {output.device_id} "
            f"{elapsed_ms}ms, anomaly={output.anomaly_detected}"
        )

        return output

    def _build_error_output(
        self,
        log_id: str,
        reading: AggregatedReading,
        elapsed_ms: float,
        notes: list,
    ) -> PipelineOutput:
        """Build degraded output when critical stages fail."""
        return PipelineOutput(
            log_id=log_id,
            timestamp=reading.timestamp_start,
            device_id=reading.device_id,
            circuit_id=reading.circuit_id,
            sensor_reading_watts=reading.avg_wattage,
            energy_kwh=0.0,
            co2_kg=0.0,
            emission_factor_used=settings.EMISSION_FACTOR,
            data_quality_status=DataQualityStatus.FLAGGED,
            quality_flags=[f"pipeline_error: {', '.join(str(n) for n in notes)}"],
            anomaly_detected=False,
            is_degraded=True,
            processing_time_ms=int(elapsed_ms),
        )

    def get_orchestrator_stats(self) -> dict:
        """Get pipeline orchestration statistics."""
        return self._stats


# ── Module-level singleton ────────────────────────────────────────
pipeline = PipelineOrchestrator()
