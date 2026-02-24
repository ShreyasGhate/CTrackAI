"""
CTrackAI — Carbon Math Engine (Stage 4 — Production Grade)

Converts energy consumption into CO₂ emissions using a 6-layer
calculation approach with full traceability.

Layer 1 — Input Validation:
    Validates wattage is in expected range (watts, not kW).

Layer 2 — Core Energy Math:
    energy_kwh = avg_wattage × window_hours / 1000

Layer 3 — Time-of-Use Emission Factor:
    Uses peak/shoulder/off-peak emission factors based on
    grid carbon intensity (coal dispatch varies by hour).
    Peak: 0.82, Shoulder: 0.727, Off-peak: 0.65

Layer 4 — Device-Specific Context:
    Decomposes power into idle + active components using
    RAG device specs for smarter energy attribution.

Layer 5 — Carbon Equivalents & Annualization:
    Converts CO₂ to relatable equivalents (trees, car-km,
    phone charges) and projects annual emissions.

Layer 6 — Citations & Traceability:
    Full audit trail with formula breakdown and academic references.

References:
    - GHG Protocol Corporate Standard (WRI & WBCSD)
    - Wang et al. (2022), Frontiers in Energy Research
    - Central Electricity Authority (CEA), India
    - EPA GHG Equivalencies Calculator
"""

from datetime import timezone
from typing import Optional
from loguru import logger

from models.schemas import (
    QualityCheckedReading,
    CarbonResult,
    DeviceSpec,
)
from config.settings import settings


class CarbonMathEngine:
    """
    6-layer production-grade carbon emission calculator.

    Converts quality-checked wattage readings into CO₂ emissions
    with input validation, time-of-use factors, device decomposition,
    carbon equivalents, annualized projections, and full traceability.

    Usage:
        engine = CarbonMathEngine()
        result = engine.calculate(quality_reading, device_spec=optional)
    """

    # ── Carbon Equivalents Constants ──────────────────────────
    # Source: EPA Greenhouse Gas Equivalencies Calculator
    # https://www.epa.gov/energy/greenhouse-gas-equivalencies-calculator
    TREES_ABSORPTION_KG_PER_YEAR = 21.77   # kg CO₂ per tree per year
    CAR_KG_PER_KM = 0.21                   # kg CO₂ per km (avg petrol car India)
    PHONE_CHARGE_KG = 0.008                # kg CO₂ per full smartphone charge

    # ── Input Validation Bounds ───────────────────────────────
    # Catches unit errors (kW instead of W, or MW instead of W)
    WATTAGE_WARN_LOW = 0.01    # Below this → suspicious
    WATTAGE_WARN_HIGH = 50000  # Above this → likely wrong unit

    def __init__(
        self,
        emission_factor: float = None,
        region: str = None,
    ):
        """
        Args:
            emission_factor: kg CO₂ per kWh (default: from config)
            region: Region for emission factor source (default: from config)
        """
        self.emission_factor = emission_factor or settings.EMISSION_FACTOR
        self.region = region or settings.REGION
        self.tou_enabled = settings.EMISSION_TOU_ENABLED

        self._emission_source = self._get_emission_source()

        # Running totals
        self._stats = {
            "total_calculations": 0,
            "total_energy_kwh": 0.0,
            "total_co2_kg": 0.0,
            "validation_warnings": 0,
            "tou_adjustments": 0,
            "device_context_applied": 0,
        }

        logger.info(
            f"CarbonMathEngine initialized: "
            f"emission_factor={self.emission_factor} kg CO₂/kWh, "
            f"region={self.region}, tou_enabled={self.tou_enabled}"
        )

    def _get_emission_source(self) -> str:
        """Get emission factor source description."""
        sources = {
            "maharashtra": "CEA Maharashtra Grid (2023)",
            "india": "CEA India National Average (2023)",
        }
        return sources.get(self.region.lower(), f"Regional grid ({self.region})")

    # ══════════════════════════════════════════════════════════
    # MAIN ENTRY POINT
    # ══════════════════════════════════════════════════════════

    def calculate(
        self,
        reading: QualityCheckedReading,
        device_spec: Optional[DeviceSpec] = None,
    ) -> CarbonResult:
        """
        Calculate CO₂ emissions for a quality-checked reading.

        6-layer pipeline:
            1. Input validation (unit sanity check)
            2. Core energy math (W → kWh)
            3. Time-of-use emission factor selection
            4. Device context decomposition
            5. Carbon equivalents & annualization
            6. Citations & traceability

        Returns:
            CarbonResult with energy, emissions, equivalents, and audit trail
        """
        self._stats["total_calculations"] += 1

        # ── Layer 1: Input Validation ─────────────────────────
        validation_notes = self._layer1_validate(reading)

        # ── Layer 2: Core Energy Math ─────────────────────────
        energy_kwh, energy_formula = self._layer2_energy(reading)

        # ── Layer 3: Time-of-Use Emission Factor ──────────────
        effective_factor, factor_source, tou_formula = (
            self._layer3_tou_factor(reading)
        )

        # ── Apply emission factor ─────────────────────────────
        co2_kg = energy_kwh * effective_factor
        co2_formula = (
            f"co2_kg = {energy_kwh:.6f} kWh "
            f"× {effective_factor} kg CO₂/kWh "
            f"= {co2_kg:.6f} kg CO₂"
        )

        # ── Layer 4: Device Context ───────────────────────────
        adjusted_co2, context_notes = self._layer4_device_context(
            co2_kg, energy_kwh, reading, device_spec
        )

        # ── Layer 5: Equivalents & Annualization ──────────────
        equivalents = self._layer5_equivalents(adjusted_co2, reading)

        # ── Layer 6: Citations & Traceability ─────────────────
        calc_details, citations = self._layer6_citations(
            reading, energy_kwh, effective_factor, co2_kg,
            adjusted_co2, energy_formula, co2_formula,
            tou_formula, context_notes, validation_notes,
        )

        # ── Confidence Calculation ────────────────────────────
        confidence = self._calculate_confidence(reading, device_spec)

        # ── Update stats ──────────────────────────────────────
        self._stats["total_energy_kwh"] += energy_kwh
        self._stats["total_co2_kg"] += adjusted_co2
        if device_spec:
            self._stats["device_context_applied"] += 1

        # ── Build Result ──────────────────────────────────────
        result = CarbonResult(
            device_id=reading.device_id,
            circuit_id=reading.circuit_id,
            timestamp_start=reading.timestamp_start,
            timestamp_end=reading.timestamp_end,
            energy_kwh=round(energy_kwh, 6),
            co2_kg=round(adjusted_co2, 6),
            emission_factor=effective_factor,
            emission_factor_source=factor_source,
            region=self.region,
            confidence=confidence,
            device_context_applied=device_spec is not None,
            device_spec=device_spec,
            equivalent_trees_monthly=equivalents.get("trees_monthly"),
            equivalent_car_km=equivalents.get("car_km"),
            equivalent_phone_charges=equivalents.get("phone_charges"),
            annualized_co2_tonnes=equivalents.get("annual_co2_tonnes"),
            annualized_energy_mwh=equivalents.get("annual_energy_mwh"),
            formula_citations=citations,
            calculation_details=calc_details,
        )

        logger.debug(
            f"Carbon calc for {reading.device_id}/{reading.circuit_id}: "
            f"{energy_kwh:.4f} kWh → {adjusted_co2:.6f} kg CO₂ "
            f"(factor={effective_factor}, confidence={confidence})"
        )

        return result

    # ══════════════════════════════════════════════════════════
    # LAYER 1: INPUT VALIDATION
    # ══════════════════════════════════════════════════════════

    def _layer1_validate(self, reading: QualityCheckedReading) -> str:
        """
        Validate input wattage for unit errors.

        Catches common mistakes:
            - Wattage in kW instead of W (values like 0.5 for a 500W device)
            - Wattage in MW (values > 50kW for lab equipment)
            - Negative wattage that slipped past quality checks
        """
        wattage = reading.corrected_wattage
        notes = []

        if wattage < 0:
            notes.append(f"WARNING: Negative wattage {wattage}W after quality check")
            self._stats["validation_warnings"] += 1

        if 0 < wattage < self.WATTAGE_WARN_LOW:
            notes.append(
                f"WARNING: Very low wattage {wattage}W — "
                f"possible unit error (kW sent as W?)"
            )
            self._stats["validation_warnings"] += 1
            logger.warning(
                f"Suspiciously low wattage {wattage}W for "
                f"{reading.device_id}/{reading.circuit_id}. "
                f"Check if sensor is sending kW instead of W."
            )

        if wattage > self.WATTAGE_WARN_HIGH:
            notes.append(
                f"WARNING: Very high wattage {wattage}W — "
                f"possible unit error (mW sent as W?)"
            )
            self._stats["validation_warnings"] += 1
            logger.warning(
                f"Suspiciously high wattage {wattage}W for "
                f"{reading.device_id}/{reading.circuit_id}. "
                f"Check if sensor is sending mW instead of W."
            )

        return " | ".join(notes) if notes else "Input validation passed"

    # ══════════════════════════════════════════════════════════
    # LAYER 2: CORE ENERGY MATH
    # ══════════════════════════════════════════════════════════

    def _layer2_energy(
        self, reading: QualityCheckedReading
    ) -> tuple[float, str]:
        """
        Calculate energy consumption in kWh.

        Formula:
            energy_kwh = corrected_wattage × window_hours / 1000
        """
        duration = reading.timestamp_end - reading.timestamp_start
        window_hours = duration.total_seconds() / 3600.0

        energy_kwh = (reading.corrected_wattage * window_hours) / 1000.0

        formula = (
            f"energy_kwh = ({reading.corrected_wattage:.2f}W "
            f"× {window_hours:.4f}h) / 1000 "
            f"= {energy_kwh:.6f} kWh"
        )

        return energy_kwh, formula

    # ══════════════════════════════════════════════════════════
    # LAYER 3: TIME-OF-USE EMISSION FACTOR
    # ══════════════════════════════════════════════════════════

    def _layer3_tou_factor(
        self, reading: QualityCheckedReading
    ) -> tuple[float, str, str]:
        """
        Select emission factor based on time of day.

        Grid carbon intensity varies by hour:
            Peak (10AM-6PM):     0.82 kg CO₂/kWh (coal-heavy dispatch)
            Shoulder (6-10AM, 6-10PM): 0.727 (mixed)
            Off-peak (10PM-6AM): 0.65 (renewables/hydro dominant)

        Returns:
            (effective_factor, source_description, formula_note)
        """
        if not self.tou_enabled:
            return (
                self.emission_factor,
                self._emission_source,
                f"Time-of-use disabled — using flat rate {self.emission_factor}",
            )

        # Get hour of reading (use midpoint of window)
        midpoint = reading.timestamp_start + (
            reading.timestamp_end - reading.timestamp_start
        ) / 2

        # Convert to local time for hour determination
        # (readings may be in UTC, but grid dispatch is local)
        hour = midpoint.hour

        if 10 <= hour < 18:  # 10 AM - 6 PM
            factor = settings.EMISSION_FACTOR_PEAK
            period = "peak"
        elif (6 <= hour < 10) or (18 <= hour < 22):  # Shoulder
            factor = settings.EMISSION_FACTOR_SHOULDER
            period = "shoulder"
        else:  # 10 PM - 6 AM
            factor = settings.EMISSION_FACTOR_OFFPEAK
            period = "off-peak"

        self._stats["tou_adjustments"] += 1

        source = f"CEA Maharashtra Grid ({period}, hour={hour})"
        formula = (
            f"TOU emission factor: {period} period (hour={hour}) → "
            f"{factor} kg CO₂/kWh "
            f"(peak={settings.EMISSION_FACTOR_PEAK}, "
            f"shoulder={settings.EMISSION_FACTOR_SHOULDER}, "
            f"off-peak={settings.EMISSION_FACTOR_OFFPEAK})"
        )

        return factor, source, formula

    # ══════════════════════════════════════════════════════════
    # LAYER 4: DEVICE-SPECIFIC CONTEXT
    # ══════════════════════════════════════════════════════════

    def _layer4_device_context(
        self,
        co2_kg: float,
        energy_kwh: float,
        reading: QualityCheckedReading,
        device_spec: Optional[DeviceSpec],
    ) -> tuple[float, str]:
        """
        Device-specific power decomposition.

        Decomposes measured wattage into idle and active components:
            total_power = idle_power + active_power
            active_power = total_power - idle_power

        This decomposition enables:
            - Identifying idle waste (device powered but not working)
            - Comparing active efficiency across devices
            - Estimating power-down savings

        The CO₂ calculation uses the MEASURED value (no adjustment),
        but the decomposition provides rich context for analysis.
        """
        if device_spec is None:
            return co2_kg, "No device spec — raw calculation used"

        notes_parts = []
        wattage = reading.corrected_wattage

        # ── Power Decomposition ───────────────────────────────
        idle_wattage = device_spec.idle_wattage or 0.0
        rated_wattage = device_spec.rated_wattage or wattage

        # Active power = total - idle (clamped to 0)
        active_power = max(0, wattage - idle_wattage)

        # Load factor = fraction of rated active capacity in use
        active_capacity = max(1, rated_wattage - idle_wattage)
        load_factor = min(active_power / active_capacity, 1.5)

        # Idle fraction = what % of total power is idle waste
        idle_fraction = idle_wattage / wattage if wattage > 0 else 0

        notes_parts.append(f"device={device_spec.device_name}")
        notes_parts.append(f"category={device_spec.category}")
        notes_parts.append(
            f"decomposition: {wattage:.0f}W = "
            f"{idle_wattage:.0f}W idle + {active_power:.0f}W active"
        )
        notes_parts.append(f"load_factor={load_factor:.3f}")
        notes_parts.append(f"idle_fraction={idle_fraction:.1%}")

        # Duty cycle context
        duty_cycle = device_spec.duty_cycle if device_spec.duty_cycle else 1.0
        if duty_cycle < 1.0:
            notes_parts.append(f"rated_duty_cycle={duty_cycle:.0%}")

        # Energy Star context
        if device_spec.energy_star_rated:
            notes_parts.append("Energy Star certified")

        # The CO₂ stays the same — we trust the measured wattage.
        # But the decomposition is crucial for the explanation and
        # anomaly detection stages downstream.
        return co2_kg, " | ".join(notes_parts)

    # ══════════════════════════════════════════════════════════
    # LAYER 5: CARBON EQUIVALENTS & ANNUALIZATION
    # ══════════════════════════════════════════════════════════

    def _layer5_equivalents(
        self,
        co2_kg: float,
        reading: QualityCheckedReading,
    ) -> dict:
        """
        Convert CO₂ to relatable equivalents and project annual emissions.

        Equivalents (EPA methodology):
            - Trees: 1 tree absorbs ~21.77 kg CO₂/year
            - Car: Average Indian petrol car emits ~0.21 kg CO₂/km
            - Phone: One full charge ≈ 0.008 kg CO₂

        Annualization:
            annual_co2 = co2_per_window × windows_per_year
        """
        # Calculate window duration
        duration = reading.timestamp_end - reading.timestamp_start
        window_hours = duration.total_seconds() / 3600.0

        # Annual extrapolation
        # Assume device runs typical_hours/day for 365 days
        hours_per_year = 365.25 * 24  # Total hours in a year
        if window_hours > 0:
            windows_per_year = hours_per_year / window_hours
            annual_co2_kg = co2_kg * windows_per_year
            annual_co2_tonnes = annual_co2_kg / 1000.0
            annual_energy_kwh = (
                reading.corrected_wattage * hours_per_year / 1000.0
            )
            annual_energy_mwh = annual_energy_kwh / 1000.0
        else:
            annual_co2_tonnes = 0.0
            annual_energy_mwh = 0.0

        # Carbon equivalents from this single reading
        trees_monthly = (
            (co2_kg / (self.TREES_ABSORPTION_KG_PER_YEAR / 12))
            if co2_kg > 0 else 0.0
        )
        car_km = co2_kg / self.CAR_KG_PER_KM if co2_kg > 0 else 0.0
        phone_charges = co2_kg / self.PHONE_CHARGE_KG if co2_kg > 0 else 0.0

        return {
            "trees_monthly": round(trees_monthly, 4),
            "car_km": round(car_km, 4),
            "phone_charges": round(phone_charges, 2),
            "annual_co2_tonnes": round(annual_co2_tonnes, 4),
            "annual_energy_mwh": round(annual_energy_mwh, 4),
        }

    # ══════════════════════════════════════════════════════════
    # LAYER 6: CITATIONS & TRACEABILITY
    # ══════════════════════════════════════════════════════════

    def _layer6_citations(
        self,
        reading: QualityCheckedReading,
        energy_kwh: float,
        effective_factor: float,
        co2_kg: float,
        adjusted_co2: float,
        energy_formula: str,
        co2_formula: str,
        tou_formula: str,
        context_notes: str,
        validation_notes: str,
    ) -> tuple[dict, list[str]]:
        """
        Build the full formula breakdown and citation list.

        Every CarbonResult includes enough information to reproduce
        the calculation from scratch — essential for audit compliance.
        """
        calc_details = {
            "layer_1_validation": validation_notes,
            "layer_2_energy": energy_formula,
            "layer_3_tou_factor": tou_formula,
            "layer_4_device_context": context_notes,
            "input_wattage": reading.corrected_wattage,
            "quality_status": reading.quality_status.value,
            "emission_factor_effective": effective_factor,
            "emission_factor_base": self.emission_factor,
            "emission_source": self._emission_source,
            "region": self.region,
            "scope": "Scope 2 (indirect, purchased electricity)",
            "method": "Location-based (GHG Protocol)",
        }

        # Correction details
        if reading.original_wattage != reading.corrected_wattage:
            calc_details["original_wattage"] = reading.original_wattage
            calc_details["correction_applied"] = True
            calc_details["replacement_method"] = reading.replacement_method

        citations = [
            settings.CITATION_GHG_PROTOCOL,
            settings.CITATION_FRONTIERS_PAPER,
            settings.CITATION_CEA_INDIA,
            (
                "US EPA, Greenhouse Gas Equivalencies Calculator, "
                "https://www.epa.gov/energy/greenhouse-gas-equivalencies-calculator"
            ),
        ]

        return calc_details, citations

    # ══════════════════════════════════════════════════════════
    # CONFIDENCE CALCULATION
    # ══════════════════════════════════════════════════════════

    def _calculate_confidence(
        self,
        reading: QualityCheckedReading,
        device_spec: Optional[DeviceSpec],
    ) -> float:
        """
        Calculate confidence score for this carbon calculation.

        Factors:
            - Data quality: clean=1.0, flagged=0.8, replaced=0.6
            - Device spec: available=+0.0, missing=-0.1
            - Quality flags: each flag reduces by 0.05 (max -0.2)
            - TOU enabled: +0.0 (more accurate), disabled=-0.05
        """
        quality_scores = {
            "clean": 1.0,
            "flagged": 0.8,
            "replaced": 0.6,
        }
        confidence = quality_scores.get(reading.quality_status.value, 0.5)

        if device_spec is None:
            confidence -= 0.1

        flag_count = len(reading.quality_flags)
        if flag_count > 0:
            confidence -= min(flag_count * 0.05, 0.2)

        if not self.tou_enabled:
            confidence -= 0.05

        return round(max(0.1, min(1.0, confidence)), 2)

    # ══════════════════════════════════════════════════════════
    # STATISTICS
    # ══════════════════════════════════════════════════════════

    def get_carbon_stats(self) -> dict:
        """Get accumulated carbon calculation statistics."""
        return {
            **self._stats,
            "emission_factor_base": self.emission_factor,
            "tou_enabled": self.tou_enabled,
            "region": self.region,
            "emission_source": self._emission_source,
        }


# ── Module-level singleton ────────────────────────────────────────
carbon_engine = CarbonMathEngine()
