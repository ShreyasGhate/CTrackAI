"""
CTrackAI — REST Data Ingestion

FastAPI router for receiving sensor readings via HTTP POST.
This is the primary data ingestion path for ESP32 sensors
configured to send data via REST API.

Endpoints:
    POST /readings           → Ingest a single sensor reading
    POST /readings/batch     → Ingest multiple readings at once
    GET  /readings/buffer    → Monitor buffer status

Flow:
    ESP32 → POST /api/v1/readings → ReadingBuffer → Aggregator → Pipeline

Enterprise Notes:
    - Input validation via Pydantic (FastAPI handles automatically)
    - Structured error responses
    - Buffer statistics endpoint for monitoring
    - Supports both single-device and multi-circuit payloads
"""

from fastapi import APIRouter, HTTPException, status
from typing import List

from models.schemas import (
    SensorReading,
    AggregatedReading,
    BatchReadingRequest,
    APIResponse,
)
from ingestion.aggregator import reading_buffer
from loguru import logger

# ── Router ────────────────────────────────────────────────────────
router = APIRouter(prefix="/readings", tags=["Data Ingestion"])


@router.post(
    "",
    response_model=APIResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest a single sensor reading",
    description=(
        "Accepts a single sensor reading from an ESP32 device. "
        "The reading is buffered and aggregated into 15-minute windows. "
        "Returns any completed aggregated windows if the reading "
        "triggers a window flush."
    ),
)
async def ingest_reading(reading: SensorReading) -> APIResponse:
    """
    Ingest a single sensor reading from an ESP32 device.

    The reading is added to the aggregation buffer. If this reading
    completes a 15-minute window, the aggregated result is returned
    in the response.

    Args:
        reading: SensorReading with device_id, timestamp, and wattage/circuit data

    Returns:
        APIResponse with any completed aggregation windows
    """
    try:
        logger.info(
            f"REST ingestion: device={reading.device_id}, "
            f"timestamp={reading.timestamp.isoformat()}, "
            f"wattage={reading.wattage}, "
            f"circuits={len(reading.circuit_readings) if reading.circuit_readings else 0}"
        )

        # Add to buffer and check for completed windows
        completed_windows = reading_buffer.add_reading(reading)

        return APIResponse(
            success=True,
            data={
                "accepted": True,
                "device_id": reading.device_id,
                "completed_windows": [w.model_dump() for w in completed_windows],
                "completed_window_count": len(completed_windows),
            },
            processing_time_ms=0,
        )

    except Exception as e:
        logger.error(f"REST ingestion failed for device {reading.device_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process reading: {str(e)}",
        )


@router.post(
    "/batch",
    response_model=APIResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest a batch of sensor readings",
    description=(
        "Accepts multiple sensor readings at once. Useful for "
        "bulk uploads or when sensors batch readings locally "
        "before sending."
    ),
)
async def ingest_batch(request: BatchReadingRequest) -> APIResponse:
    """
    Ingest a batch of sensor readings.

    Each reading is processed individually through the buffer.
    Returns all aggregated windows that completed during processing.

    Args:
        request: BatchReadingRequest containing a list of SensorReadings

    Returns:
        APIResponse with processing summary and completed windows
    """
    try:
        logger.info(f"REST batch ingestion: {len(request.readings)} readings")

        all_completed = []
        accepted_count = 0
        failed_count = 0

        for reading in request.readings:
            try:
                completed = reading_buffer.add_reading(reading)
                all_completed.extend(completed)
                accepted_count += 1
            except Exception as e:
                logger.warning(
                    f"Failed to process reading from {reading.device_id}: {e}"
                )
                failed_count += 1

        return APIResponse(
            success=True,
            data={
                "accepted_count": accepted_count,
                "failed_count": failed_count,
                "total_submitted": len(request.readings),
                "completed_windows": [w.model_dump() for w in all_completed],
                "completed_window_count": len(all_completed),
            },
            processing_time_ms=0,
        )

    except Exception as e:
        logger.error(f"REST batch ingestion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing failed: {str(e)}",
        )


@router.get(
    "/buffer/stats",
    response_model=APIResponse,
    summary="Get aggregation buffer statistics",
    description=(
        "Returns current buffer state: active windows, "
        "total buffered readings, configuration. "
        "Useful for monitoring and debugging."
    ),
)
async def get_buffer_stats() -> APIResponse:
    """
    Get current aggregation buffer statistics.

    Returns:
        APIResponse with buffer stats
    """
    stats = reading_buffer.get_buffer_stats()
    return APIResponse(
        success=True,
        data=stats,
        processing_time_ms=0,
    )


@router.post(
    "/buffer/flush",
    response_model=APIResponse,
    summary="Force-flush all buffered windows",
    description=(
        "Force-flushes all buffered windows regardless of "
        "completion time. Primarily for testing and debugging."
    ),
)
async def flush_buffer() -> APIResponse:
    """
    Force-flush all aggregation windows.
    Returns all aggregated readings from the buffer.

    Returns:
        APIResponse with all flushed aggregated readings
    """
    try:
        completed = reading_buffer.force_flush_all()
        return APIResponse(
            success=True,
            data={
                "flushed_windows": [w.model_dump() for w in completed],
                "flushed_count": len(completed),
            },
            processing_time_ms=0,
        )
    except Exception as e:
        logger.error(f"Buffer flush failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Buffer flush failed: {str(e)}",
        )
