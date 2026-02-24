"""
CTrackAI — Central Configuration

All settings loaded from environment variables with sensible defaults.
Uses Pydantic BaseSettings for validation and .env file support.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List
from pathlib import Path


class Settings(BaseSettings):
    """Central configuration for CTrackAI AI Model Service."""

    # ── Project Info ──────────────────────────────────────────────
    PROJECT_NAME: str = "CTrackAI"
    PIPELINE_VERSION: str = "1.0.0"

    # ── Emission Factors (kg CO₂ per kWh) ─────────────────────────
    # Source: Central Electricity Authority (CEA), India
    EMISSION_FACTOR_MAHARASHTRA: float = 0.727
    EMISSION_FACTOR_INDIA_DEFAULT: float = 0.82
    EMISSION_FACTOR: float = 0.727  # Active emission factor

    # ── Regional Config ───────────────────────────────────────────
    REGION: str = "maharashtra"
    COUNTRY_CODE: str = "IN"

    # ── Working Hours ─────────────────────────────────────────────
    WORKING_HOURS_START: int = 9   # 9 AM
    WORKING_HOURS_END: int = 18    # 6 PM
    WORKING_DAYS: List[int] = Field(default=[0, 1, 2, 3, 4])  # Mon=0 to Fri=4
    LOCAL_TIMEZONE_OFFSET_HOURS: float = 5.5  # IST = UTC + 5:30

    # ── Data Aggregation ──────────────────────────────────────────
    SENSOR_INTERVAL_SECONDS: int = 10
    AGGREGATION_WINDOW_MINUTES: int = 15

    # ── Data Quality Thresholds ───────────────────────────────────
    MAX_PLAUSIBLE_WATTAGE: float = 50000.0  # 50kW — max plausible for a lab
    MIN_PLAUSIBLE_WATTAGE: float = -1.0     # Negative = sensor error
    OUTLIER_ZSCORE_THRESHOLD: float = 3.0

    # ── Anomaly Detection ─────────────────────────────────────────
    # Rule engine thresholds
    ANOMALY_SPIKE_MULTIPLIER: float = 1.3   # wattage > rated × 1.3 = anomaly
    ANOMALY_SEVERE_MULTIPLIER: float = 2.5  # wattage > baseline × 2.5 = HIGH

    # Weighted voting
    ANOMALY_WEIGHT_RULES: float = 0.4
    ANOMALY_WEIGHT_ISOFOREST: float = 0.3
    ANOMALY_WEIGHT_AUTOENCODER: float = 0.3
    ANOMALY_THRESHOLD: float = 0.5  # score >= 0.5 → anomaly

    # Isolation Forest
    ISOFOREST_CONTAMINATION: float = 0.05
    ISOFOREST_N_ESTIMATORS: int = 100
    ISOFOREST_RANDOM_STATE: int = 42

    # ── Severity Scoring ──────────────────────────────────────────
    SEVERITY_WEIGHT_DELTA_WATTAGE: float = 0.4
    SEVERITY_WEIGHT_CONTEXT_PENALTY: float = 0.3
    SEVERITY_WEIGHT_ANOMALY_SCORE: float = 0.3

    # Severity bands
    SEVERITY_LOW_MAX: float = 0.30
    SEVERITY_MEDIUM_MAX: float = 0.55
    SEVERITY_HIGH_MAX: float = 0.80
    # Above 0.80 = CRITICAL

    # ── Rolling Baseline ──────────────────────────────────────────
    ROLLING_BASELINE_WEEKS: int = 4
    ROLLING_BASELINE_MIN_READINGS: int = 10

    # ── Forecasting ───────────────────────────────────────────────
    FORECAST_MIN_HISTORY_DAYS: int = 14  # Prophet needs ~2 weeks minimum
    FORECAST_CHANGEPOINT_PRIOR: float = 0.05
    FORECAST_SEASONALITY_MODE: str = "multiplicative"

    # ── SHAP / Explanation ────────────────────────────────────────
    SHAP_MAX_BACKGROUND_SAMPLES: int = 100
    SHAP_TOP_FEATURES: int = 3
    SHAP_TIMEOUT_SECONDS: float = 5.0

    # ── API Settings ──────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_TIMEOUT_SECONDS: float = 10.0

    # ── MQTT Settings ─────────────────────────────────────────────
    MQTT_BROKER_HOST: str = "localhost"
    MQTT_BROKER_PORT: int = 1883
    MQTT_TOPIC: str = "ctrackai/readings/#"
    MQTT_ENABLED: bool = False

    # ── Storage ───────────────────────────────────────────────────
    SQLITE_DB_PATH: str = "./data/ctrackai_logs.db"
    DATA_DIR: str = "./data"

    # ── RAG / Vector Store ────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    RAG_SIMILARITY_THRESHOLD: float = 0.30  # Min similarity to accept a match

    # ── Time-of-Use Emission Factors ──────────────────────────────
    # Grid carbon intensity varies by hour. During peak demand (daytime),
    # more coal plants are dispatched → higher emission factor.
    # Source: CEA Load Dispatch data, Maharashtra SLDC
    EMISSION_FACTOR_PEAK: float = 0.82       # 10 AM - 6 PM (coal-heavy)
    EMISSION_FACTOR_SHOULDER: float = 0.727   # 6 AM - 10 AM, 6 PM - 10 PM
    EMISSION_FACTOR_OFFPEAK: float = 0.65     # 10 PM - 6 AM (renewables/hydro)
    EMISSION_TOU_ENABLED: bool = True         # Enable time-of-use adjustment

    # ── Logging ───────────────────────────────────────────────────
    LOG_DIR: str = "./logs"
    LOG_RETENTION_DAYS: int = 90
    LOG_ROTATION: str = "10 MB"
    LOG_FORMAT: str = "json"

    # ── MLflow ────────────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = "./mlruns"

    # ── Formula Citations ─────────────────────────────────────────
    CITATION_GHG_PROTOCOL: str = (
        "GHG Protocol Corporate Standard, "
        "World Resources Institute & WBCSD, "
        "https://ghgprotocol.org/corporate-standard"
    )
    CITATION_FRONTIERS_PAPER: str = (
        "Wang et al. (2022), "
        "'Efficient whole-process carbon intensity calculation method "
        "for power users in active distribution networks', "
        "Frontiers in Energy Research, "
        "DOI: 10.3389/fenrg.2022.974365"
    )
    CITATION_CEA_INDIA: str = (
        "Central Electricity Authority (CEA), "
        "CO2 Baseline Database for the Indian Power Sector, "
        "Ministry of Power, Government of India"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# ── Singleton instance ────────────────────────────────────────────
settings = Settings()
