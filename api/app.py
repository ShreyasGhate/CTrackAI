"""
CTrackAI — FastAPI Application Factory

Creates and configures the FastAPI application with:
    - API versioning via app.mount()
    - CORS middleware
    - Startup/shutdown lifecycle events
    - Health check endpoint

Architecture:
    main.py → create_api_app() → FastAPI app with /api/v1 mounted

Enterprise Notes:
    - Structured for future /api/v2 addition without breaking v1
    - MQTT subscriber starts/stops with app lifecycle
    - Health endpoint returns service status and buffer stats
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from loguru import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.

    Startup:
        - Initialize logging
        - Start MQTT subscriber (if enabled)
        - Log service info

    Shutdown:
        - Stop MQTT subscriber
        - Flush aggregation buffer
    """
    # ── Startup ───────────────────────────────────────────────
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.PIPELINE_VERSION}")
    logger.info(f"Region: {settings.REGION}, Emission Factor: {settings.EMISSION_FACTOR}")

    # Start MQTT if enabled
    if settings.MQTT_ENABLED:
        try:
            from ingestion.mqtt_ingest import mqtt_subscriber
            mqtt_subscriber.start()
            logger.info("MQTT subscriber started")
        except Exception as e:
            logger.warning(f"MQTT subscriber failed to start: {e}. Continuing without MQTT.")

    yield

    # ── Shutdown ──────────────────────────────────────────────
    logger.info("Shutting down...")

    # Stop MQTT
    if settings.MQTT_ENABLED:
        try:
            from ingestion.mqtt_ingest import mqtt_subscriber
            mqtt_subscriber.stop()
        except Exception:
            pass

    # Flush remaining buffered readings
    try:
        from ingestion.aggregator import reading_buffer
        remaining = reading_buffer.force_flush_all()
        if remaining:
            logger.info(f"Flushed {len(remaining)} remaining windows on shutdown")
    except Exception:
        pass

    logger.info(f"{settings.PROJECT_NAME} stopped")


def create_api_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI app instance
    """
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description=(
            "AI-powered carbon footprint tracking model service. "
            "Processes IoT sensor data through an 11-stage ML pipeline "
            "for carbon math, anomaly detection, and forecasting."
        ),
        version=settings.PIPELINE_VERSION,
        lifespan=lifespan,
    )

    # ── CORS Middleware ───────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Mount API v1 ──────────────────────────────────────────
    from api.v1.router import api_v1_router
    app.include_router(api_v1_router, prefix="/api/v1")

    # ── Root Health Check ─────────────────────────────────────
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Service health check endpoint."""
        from ingestion.aggregator import reading_buffer

        return {
            "status": "healthy",
            "service": settings.PROJECT_NAME,
            "version": settings.PIPELINE_VERSION,
            "region": settings.REGION,
            "emission_factor": settings.EMISSION_FACTOR,
            "buffer": reading_buffer.get_buffer_stats(),
        }

    logger.info("FastAPI app created with /api/v1 routes")
    return app
