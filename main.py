"""
CTrackAI — AI Model Service for Carbon Footprint Tracking

Entry point: starts the FastAPI server via Uvicorn.

Usage:
    python main.py
    OR
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import uvicorn
from config.settings import settings


def create_app():
    """Create and configure the FastAPI application."""
    # Import here to avoid circular imports during setup
    from api.app import create_api_app
    return create_api_app()


# Create the app instance (used by uvicorn main:app)
app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level="info",
    )
