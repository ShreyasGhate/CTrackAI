"""
CTrackAI — API v1 Router (Stage 11 — Production)

Full API serving layer with:
    /api/v1/readings/*     → Data ingestion (Stage 1)
    /api/v1/pipeline/*     → Full pipeline processing
    /api/v1/devices/*      → Device queries
    /api/v1/anomalies/*    → Anomaly queries
    /api/v1/forecast/*     → Forecast queries
    /api/v1/stats/*        → Pipeline statistics
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from loguru import logger

from ingestion.rest_ingest import router as readings_router


# ── v1 Router ─────────────────────────────────────────────────
api_v1_router = APIRouter()

# Include Stage 1 ingestion
api_v1_router.include_router(readings_router)


# ══════════════════════════════════════════════════════════════
# PIPELINE ENDPOINTS
# ══════════════════════════════════════════════════════════════

pipeline_router = APIRouter(prefix="/pipeline", tags=["Pipeline"])


@pipeline_router.get("/stats")
async def pipeline_stats():
    """Get full pipeline statistics from all stages."""
    try:
        stats = {}

        try:
            from pipeline.anomaly_detection import anomaly_detector
            stats["anomaly_detector"] = anomaly_detector.get_detector_stats()
        except Exception:
            pass

        try:
            from pipeline.severity_scoring import severity_engine
            stats["severity_engine"] = severity_engine.get_severity_stats()
        except Exception:
            pass

        try:
            from pipeline.explanation_generator import explanation_generator
            stats["explanation_generator"] = explanation_generator.get_explanation_stats()
        except Exception:
            pass

        try:
            from pipeline.forecast_engine import forecast_engine
            stats["forecast_engine"] = forecast_engine.get_forecast_stats()
        except Exception:
            pass

        try:
            from pipeline.traceability import trace_logger
            stats["traceability"] = trace_logger.get_logger_stats()
        except Exception:
            pass

        try:
            from pipeline.orchestrator import pipeline
            stats["orchestrator"] = pipeline.get_orchestrator_stats()
        except Exception:
            pass

        return stats

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


api_v1_router.include_router(pipeline_router)


# ══════════════════════════════════════════════════════════════
# DEVICE ENDPOINTS
# ══════════════════════════════════════════════════════════════

device_router = APIRouter(prefix="/devices", tags=["Devices"])


@device_router.get("/{device_id}/summary")
async def device_summary(device_id: str):
    """Get summary statistics for a device."""
    try:
        from pipeline.traceability import trace_logger
        summary = trace_logger.get_device_summary(device_id)
        if not summary or summary.get("total_readings") == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for device '{device_id}'"
            )
        return summary
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@device_router.get("/{device_id}/history")
async def device_history(
    device_id: str,
    limit: int = Query(20, ge=1, le=100),
    anomalies_only: bool = False,
):
    """Get recent pipeline results for a device."""
    try:
        from pipeline.traceability import trace_logger
        results = trace_logger.query_device(
            device_id, limit=limit, anomalies_only=anomalies_only
        )
        return {"device_id": device_id, "results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@device_router.get("/{device_id}/forecast")
async def device_forecast(device_id: str):
    """Get forecast for a device."""
    try:
        from pipeline.forecast_engine import forecast_engine
        result = forecast_engine.forecast(device_id)
        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


api_v1_router.include_router(device_router)


# ══════════════════════════════════════════════════════════════
# ANOMALY ENDPOINTS
# ══════════════════════════════════════════════════════════════

anomaly_router = APIRouter(prefix="/anomalies", tags=["Anomalies"])


@anomaly_router.get("/recent")
async def recent_anomalies(
    severity: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
):
    """Get recent anomalies across all devices."""
    try:
        from pipeline.traceability import trace_logger
        results = trace_logger.query_anomalies(
            severity=severity, limit=limit
        )
        return {"anomalies": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


api_v1_router.include_router(anomaly_router)
