"""
CTrackAI — Pydantic Schemas

All data models used across the pipeline stages.
Each schema corresponds to the output of a specific pipeline stage.

Schema Flow:
  SensorReading → QualityCheckedReading → CarbonResult → ContextFeatures
  → AnomalyResult → SeverityResult → Explanation → Forecast → PipelineOutput
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import uuid


# ═══════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════

class DataQualityStatus(str, Enum):
    """Status of a sensor reading after quality checks."""
    CLEAN = "clean"
    FLAGGED = "flagged"
    REPLACED = "replaced"


class SeverityLevel(str, Enum):
    """Severity classification for anomalies."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DegradationLevel(str, Enum):
    """Level of service degradation."""
    NONE = "none"
    PARTIAL = "partial"
    SIGNIFICANT = "significant"


# ═══════════════════════════════════════════════════════════════════
# STAGE 1 — SENSOR INPUT
# ═══════════════════════════════════════════════════════════════════

class CircuitReading(BaseModel):
    """A single circuit-level reading from the ESP32 sensor."""
    circuit_id: str = Field(..., description="Circuit identifier, e.g. 'PC_Row_A', 'Fans'")
    wattage: Optional[float] = Field(None, description="Instantaneous power in watts")
    energy_kwh: Optional[float] = Field(None, description="Energy consumed in kWh (if pre-calculated)")


class SensorReading(BaseModel):
    """
    Raw sensor reading from an ESP32 IoT device.
    This is the entry point to the pipeline.
    """
    device_id: str = Field(..., description="Unique device/lab identifier, e.g. 'LAB_01'")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Reading timestamp (ISO 8601)")
    wattage: Optional[float] = Field(None, description="Total wattage (if single device)")
    circuit_readings: Optional[List[CircuitReading]] = Field(
        None, description="Circuit-level readings (if multi-circuit lab setup)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "device_id": "LAB_01",
                "timestamp": "2026-02-23T10:30:00Z",
                "wattage": 2450.5,
                "circuit_readings": [
                    {"circuit_id": "PC_Row_A", "wattage": 1200.0},
                    {"circuit_id": "Fans", "wattage": 350.5},
                    {"circuit_id": "AC_Units", "wattage": 900.0}
                ]
            }
        }


class AggregatedReading(BaseModel):
    """
    Aggregated reading from the 10s → 15-min window aggregator.
    This is what enters the main pipeline.
    """
    device_id: str
    circuit_id: Optional[str] = Field(None, description="Circuit ID if circuit-level")
    timestamp_start: datetime
    timestamp_end: datetime
    avg_wattage: float = Field(..., description="Average wattage over the window")
    max_wattage: float = Field(..., description="Peak wattage in the window")
    min_wattage: float = Field(..., description="Minimum wattage in the window")
    energy_kwh: float = Field(..., description="Total energy consumed in the window (kWh)")
    reading_count: int = Field(..., description="Number of 10s readings in this window")
    has_gaps: bool = Field(False, description="True if some expected readings were missing")


# ═══════════════════════════════════════════════════════════════════
# STAGE 2 — DATA QUALITY
# ═══════════════════════════════════════════════════════════════════

class QualityCheckedReading(BaseModel):
    """Reading after data quality checks."""
    device_id: str
    circuit_id: Optional[str] = None
    timestamp_start: datetime
    timestamp_end: datetime
    original_wattage: float = Field(..., description="Original reading before any correction")
    corrected_wattage: float = Field(..., description="Wattage after quality correction (may equal original)")
    energy_kwh: float
    quality_status: DataQualityStatus
    quality_flags: List[str] = Field(default_factory=list, description="List of quality issues found")
    replacement_method: Optional[str] = Field(None, description="Method used for replacement, if any")


# ═══════════════════════════════════════════════════════════════════
# STAGE 3 — RAG LAYER (DEVICE SPECS)
# ═══════════════════════════════════════════════════════════════════

class DeviceSpec(BaseModel):
    """Device specifications retrieved from the RAG layer."""
    device_name: str = Field(..., description="Full device name, e.g. 'HP LaserJet Pro M404dn'")
    category: str = Field(..., description="Device category: printer, computer, fan, ac, server, etc.")
    rated_wattage: float = Field(..., description="Maximum rated power in watts")
    idle_wattage: float = Field(0.0, description="Idle/standby power draw in watts")
    typical_wattage: Optional[float] = Field(None, description="Typical operating wattage")
    duty_cycle: Optional[float] = Field(None, description="Fraction of time device is active (0-1)")
    typical_usage_hours: Optional[float] = Field(None, description="Expected daily usage hours")
    energy_star_rated: bool = Field(False, description="Whether the device is Energy Star certified")
    source: str = Field("seed_data", description="Where the spec came from: seed_data, energy_star, web_scrape")
    confidence: float = Field(1.0, description="Confidence in the spec accuracy (0-1)")


# ═══════════════════════════════════════════════════════════════════
# STAGE 4 — CARBON MATH
# ═══════════════════════════════════════════════════════════════════

class CarbonResult(BaseModel):
    """Result of the carbon math calculation."""
    device_id: str
    circuit_id: Optional[str] = None
    timestamp_start: datetime
    timestamp_end: datetime
    energy_kwh: float = Field(..., description="Energy consumed (kWh)")
    emission_factor: float = Field(..., description="Emission factor used (kg CO₂/kWh)")
    emission_factor_source: str = Field("CEA Maharashtra Grid (2023)", description="Source of emission factor")
    co2_kg: float = Field(..., description="Calculated CO₂ in kilograms")
    region: str = Field(..., description="Region used for emission factor")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence in this calculation (0-1)")
    device_context_applied: bool = Field(False, description="Whether RAG device specs were used")
    device_spec: Optional[DeviceSpec] = None

    # Carbon equivalents (makes CO₂ relatable)
    equivalent_trees_monthly: Optional[float] = Field(None, description="Trees needed to absorb this CO₂ per month")
    equivalent_car_km: Optional[float] = Field(None, description="Equivalent km driven by average car")
    equivalent_phone_charges: Optional[float] = Field(None, description="Equivalent smartphone full charges")

    # Annualized projection
    annualized_co2_tonnes: Optional[float] = Field(None, description="Projected annual CO₂ in tonnes at this rate")
    annualized_energy_mwh: Optional[float] = Field(None, description="Projected annual energy in MWh at this rate")

    formula_citations: List[str] = Field(default_factory=list, description="Citation references for the formula")
    calculation_details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed breakdown of the calculation layers"
    )


# ═══════════════════════════════════════════════════════════════════
# STAGE 5 — CONTEXT ENGINE
# ═══════════════════════════════════════════════════════════════════

class ContextFeatures(BaseModel):
    """Temporal and contextual features for a reading."""
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6, description="0=Monday, 6=Sunday")
    day_name: str = Field(..., description="e.g. 'Monday'")
    is_weekend: bool
    is_working_hour: bool
    is_holiday: bool
    holiday_name: Optional[str] = None
    shift: str = Field(..., description="'working', 'after_hours', 'night'")
    baseline_wattage: Optional[float] = Field(
        None, description="Rolling baseline for this device at this time slot"
    )
    deviation_from_baseline: Optional[float] = Field(
        None, description="How far the current reading is from baseline (ratio)"
    )
    context_risk_score: float = Field(
        0.0, description="Higher if reading is in an unusual context (0-1)"
    )


# ═══════════════════════════════════════════════════════════════════
# STAGE 6 — ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════

class AnomalyLayerResult(BaseModel):
    """Result from a single anomaly detection layer."""
    layer_name: str = Field(..., description="'rule_engine', 'isolation_forest', 'autoencoder'")
    is_anomaly: bool
    score: float = Field(..., ge=0.0, le=1.0, description="Anomaly score from this layer")
    details: Optional[str] = Field(None, description="What triggered the anomaly (if any)")


class AnomalyResult(BaseModel):
    """Combined anomaly detection result from all 3 layers."""
    is_anomaly: bool
    combined_score: float = Field(..., ge=0.0, le=1.0, description="Weighted anomaly score")
    layer_results: List[AnomalyLayerResult] = Field(default_factory=list)
    anomaly_type: Optional[str] = Field(
        None, description="Type: 'after_hours', 'spike', 'sustained_high', 'server_drop', etc."
    )


# ═══════════════════════════════════════════════════════════════════
# STAGE 7 — SEVERITY SCORING
# ═══════════════════════════════════════════════════════════════════

class SeverityResult(BaseModel):
    """Severity assessment for an anomaly."""
    severity_score: float = Field(..., ge=0.0, le=1.0)
    severity_level: SeverityLevel
    contributing_factors: Dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown: {'delta_wattage': 0.3, 'context_penalty': 0.2, 'anomaly_score': 0.4}"
    )


# ═══════════════════════════════════════════════════════════════════
# STAGE 8 — EXPLANATION
# ═══════════════════════════════════════════════════════════════════

class Explanation(BaseModel):
    """Human-readable explanation for an anomaly."""
    summary: str = Field(..., description="One-sentence human-readable explanation")
    top_features: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top contributing features from SHAP"
    )
    template_used: str = Field(..., description="Which template generated this explanation")
    shap_available: bool = Field(True, description="Whether SHAP was used or fell back to rules")


# ═══════════════════════════════════════════════════════════════════
# STAGE 9 — FORECASTING
# ═══════════════════════════════════════════════════════════════════

class ForecastResult(BaseModel):
    """Forecast output for energy and CO₂."""
    next_day_estimate_kwh: Optional[float] = None
    next_day_estimate_kg_co2: Optional[float] = None
    next_week_estimate_kwh: Optional[float] = None
    next_week_estimate_kg_co2: Optional[float] = None
    confidence_interval_lower: Optional[float] = None
    confidence_interval_upper: Optional[float] = None
    forecast_model: str = Field("prophet", description="Model used: 'prophet' or 'arima'")
    forecast_timestamp: datetime = Field(default_factory=datetime.utcnow)
    sufficient_history: bool = Field(True, description="Whether enough historical data was available")


# ═══════════════════════════════════════════════════════════════════
# PIPELINE OUTPUT — COMBINES ALL STAGES
# ═══════════════════════════════════════════════════════════════════

class DegradedComponent(BaseModel):
    """Information about a degraded pipeline component."""
    component: str = Field(..., description="Stage name that degraded")
    reason: str = Field(..., description="Why it degraded")
    fallback_used: str = Field(..., description="What fallback was used instead")


class PipelineOutput(BaseModel):
    """
    Full output from a single pipeline cycle.
    Combines results from all stages.
    This is the structured log entry AND the API response body.
    """
    # ── Metadata ──
    log_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    pipeline_version: str = Field(default="1.0.0")
    processing_time_ms: int = Field(0, description="Total pipeline processing time in milliseconds")

    # ── Input ──
    device_id: str
    circuit_id: Optional[str] = None
    sensor_reading_watts: float

    # ── Stage 2: Data Quality ──
    data_quality_status: DataQualityStatus
    quality_flags: List[str] = Field(default_factory=list)

    # ── Stage 4: Carbon Math ──
    energy_kwh: float
    co2_kg: float
    emission_factor_used: float
    formula_citations: List[str] = Field(default_factory=list)

    # ── Stage 5: Context ──
    context: Optional[ContextFeatures] = None

    # ── Stage 6: Anomaly Detection ──
    anomaly_detected: bool = False
    anomaly_score: Optional[float] = None
    anomaly_type: Optional[str] = None

    # ── Stage 7: Severity (only if anomaly) ──
    severity: Optional[SeverityLevel] = None
    severity_score: Optional[float] = None

    # ── Stage 8: Explanation (only if anomaly) ──
    explanation: Optional[str] = None

    # ── Stage 9: Forecast ──
    forecast: Optional[ForecastResult] = None

    # ── Degradation Info ──
    degraded_components: List[DegradedComponent] = Field(default_factory=list)
    is_degraded: bool = Field(False, description="True if any component fell back to degraded mode")

    class Config:
        json_schema_extra = {
            "example": {
                "log_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "timestamp": "2026-02-23T10:45:00Z",
                "pipeline_version": "1.0.0",
                "processing_time_ms": 245,
                "device_id": "LAB_01",
                "circuit_id": "PC_Row_A",
                "sensor_reading_watts": 1200.0,
                "data_quality_status": "clean",
                "quality_flags": [],
                "energy_kwh": 0.3,
                "co2_kg": 0.2181,
                "emission_factor_used": 0.727,
                "formula_citations": [
                    "GHG Protocol Corporate Standard",
                    "Frontiers (DOI: 10.3389/fenrg.2022.974365)"
                ],
                "anomaly_detected": False,
                "severity": None,
                "explanation": None,
                "degraded_components": [],
                "is_degraded": False
            }
        }


# ═══════════════════════════════════════════════════════════════════
# API RESPONSE WRAPPER
# ═══════════════════════════════════════════════════════════════════

class APIResponse(BaseModel):
    """Standard API response wrapper."""
    success: bool = True
    data: Optional[Any] = None
    error: Optional[str] = None
    degraded_components: List[DegradedComponent] = Field(default_factory=list)
    processing_time_ms: int = 0


class BatchReadingRequest(BaseModel):
    """Batch sensor reading submission."""
    readings: List[SensorReading]


class DeviceReportResponse(BaseModel):
    """Device-level CO₂ report."""
    device_id: str
    period_start: datetime
    period_end: datetime
    total_energy_kwh: float
    total_co2_kg: float
    total_readings: int
    anomaly_count: int
    severity_breakdown: Dict[str, int] = Field(
        default_factory=dict,
        description="Count per severity level: {'LOW': 2, 'HIGH': 1}"
    )
    avg_daily_co2_kg: float
    forecast: Optional[ForecastResult] = None


class AnomalyListResponse(BaseModel):
    """Response for latest anomalies endpoint."""
    anomalies: List[PipelineOutput]
    total_count: int
    page: int = 1
    page_size: int = 20
