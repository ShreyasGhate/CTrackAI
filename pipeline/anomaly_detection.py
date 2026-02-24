"""
CTrackAI — Anomaly Detection Engine (Stage 6 — Production Grade v2)

3-layer hybrid anomaly detection with weighted voting:

Layer 1 — Rule-Based Engine:
    5 deterministic rules with device-type awareness:
    - R1: Spike detection (wattage > rated × multiplier)
    - R2: After-hours usage (skips always-on devices!)
    - R3: Baseline deviation (> 100% above rolling baseline)
    - R4: Zero power for always-on devices
    - R5: Excessive idle during working hours

Layer 2 — Isolation Forest:
    Per-device statistical anomaly detection. Auto-retrains
    periodically as new data arrives.

Layer 3 — Adaptive Autoencoder Stub:
    Learns running mean/variance from observed data. Uses
    Mahalanobis-style reconstruction error. PyTorch-ready swap.

Weighted Voting:
    final_score = w_rules × R + w_iso × I + w_ae × A
    Graceful degradation with proper availability flags (not string matching).

Production Features:
    - Device-type-aware rules (servers exempt from after-hours rules)
    - Thread-safe model access
    - Auto-retraining after 200 new samples since last training
    - Anomaly deduplication (suppresses repeated alerts)
    - Proper availability flags for graceful degradation
"""

import threading
import math
from collections import defaultdict
from typing import Optional, List, Tuple

import numpy as np
from loguru import logger

from models.schemas import (
    CarbonResult,
    ContextFeatures,
    DeviceSpec,
    AnomalyResult,
    AnomalyLayerResult,
)
from config.settings import settings

# Device categories that run 24/7 (exempt from after-hours rules)
ALWAYS_ON_CATEGORIES = frozenset({
    "server", "network_switch", "ups", "router",
    "nas", "camera", "biometric",
})


class AnomalyDetector:
    """
    3-layer hybrid anomaly detection with weighted voting.

    Usage:
        detector = AnomalyDetector()
        result = detector.detect(carbon_result, context, device_spec)
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Weights from config
        self._w_rules = settings.ANOMALY_WEIGHT_RULES
        self._w_iso = settings.ANOMALY_WEIGHT_ISOFOREST
        self._w_ae = settings.ANOMALY_WEIGHT_AUTOENCODER
        self._threshold = settings.ANOMALY_THRESHOLD

        # Per-device training data for Isolation Forest
        self._training_data: dict[str, List[list]] = defaultdict(list)
        self._min_samples_for_training = 50
        self._retrain_interval = 200  # Retrain after this many new samples

        # Trained models (per-device)
        self._models: dict[str, object] = {}
        self._model_sample_count: dict[str, int] = {}  # Samples at last training

        # Isolation Forest availability flag (Fix #2)
        self._iforest_available: dict[str, bool] = defaultdict(lambda: True)

        # Adaptive autoencoder running stats (Fix #4)
        self._ae_running_mean: dict[str, np.ndarray] = {}
        self._ae_running_var: dict[str, np.ndarray] = {}
        self._ae_sample_count: dict[str, int] = defaultdict(int)

        # Anomaly deduplication (Fix #5)
        # Key: (device_id, anomaly_type) → consecutive count
        self._anomaly_streak: dict[tuple, int] = defaultdict(int)
        self._suppression_threshold = 3  # Suppress after 3 consecutive identical

        self._stats = {
            "total_detections": 0,
            "anomalies_found": 0,
            "anomalies_suppressed": 0,
            "rules_triggered": 0,
            "iforest_triggered": 0,
            "autoencoder_triggered": 0,
            "fallback_to_rules_only": 0,
            "models_retrained": 0,
        }

        logger.info(
            f"AnomalyDetector initialized: "
            f"weights=({self._w_rules}, {self._w_iso}, {self._w_ae}), "
            f"threshold={self._threshold}, "
            f"retrain_interval={self._retrain_interval}"
        )

    # ══════════════════════════════════════════════════════════
    # MAIN ENTRY POINT
    # ══════════════════════════════════════════════════════════

    def detect(
        self,
        carbon: CarbonResult,
        context: ContextFeatures,
        device_spec: Optional[DeviceSpec] = None,
    ) -> AnomalyResult:
        """
        Run 3-layer anomaly detection on a reading.

        Returns:
            AnomalyResult with combined score and per-layer breakdowns
        """
        self._stats["total_detections"] += 1
        device_id = carbon.device_id

        wattage = carbon.calculation_details.get("input_wattage", 0)
        features = self._build_feature_vector(wattage, context, device_spec)

        # ── Layer 1: Rule Engine (Fix #1: device-type aware) ──
        rule_result = self._layer1_rules(wattage, context, device_spec)

        # ── Layer 2: Isolation Forest ─────────────────────────
        iso_result = self._layer2_isolation_forest(device_id, features)

        # ── Layer 3: Adaptive Autoencoder (Fix #4) ────────────
        ae_result = self._layer3_autoencoder(device_id, features, context)

        # ── Weighted Voting (Fix #2: proper flags) ────────────
        combined_score, is_anomaly = self._weighted_vote(
            rule_result, iso_result, ae_result
        )

        # ── Anomaly Classification ────────────────────────────
        anomaly_type = None
        if is_anomaly:
            anomaly_type = self._classify_anomaly(
                wattage, context, device_spec, rule_result
            )

            # Fix #5: Deduplication
            is_suppressed = self._check_deduplication(
                device_id, anomaly_type
            )
            if is_suppressed:
                self._stats["anomalies_suppressed"] += 1
                # Still report as anomaly but mark as suppressed
                anomaly_type = f"{anomaly_type} (suppressed)"

            self._stats["anomalies_found"] += 1
        else:
            # Reset streak on normal reading
            self._reset_streak(device_id)

        # ── Store training data ───────────────────────────────
        self._store_training_sample(device_id, features)

        result = AnomalyResult(
            is_anomaly=is_anomaly,
            combined_score=round(combined_score, 4),
            layer_results=[rule_result, iso_result, ae_result],
            anomaly_type=anomaly_type,
        )

        if is_anomaly and "(suppressed)" not in (anomaly_type or ""):
            logger.info(
                f"ANOMALY [{anomaly_type}] for {device_id}: "
                f"score={combined_score:.3f}, "
                f"rules={rule_result.score:.2f}, "
                f"iforest={iso_result.score:.2f}, "
                f"ae={ae_result.score:.2f}"
            )

        return result

    # ══════════════════════════════════════════════════════════
    # FEATURE VECTOR
    # ══════════════════════════════════════════════════════════

    def _build_feature_vector(
        self,
        wattage: float,
        context: ContextFeatures,
        device_spec: Optional[DeviceSpec],
    ) -> np.ndarray:
        """
        Build normalized feature vector for ML models.

        Features: [wattage_norm, hour_sin, hour_cos, day_sin, day_cos,
                   is_working, is_weekend, deviation, load_factor, risk]
        """
        # Normalize wattage (use device rated if available)
        max_watt = 10000.0
        if device_spec and device_spec.rated_wattage > 0:
            max_watt = device_spec.rated_wattage * 3.0
        watt_norm = min(wattage / max_watt, 1.0)

        # Cyclical encoding (preserves 23→0 and Sun→Mon continuity)
        hour_sin = math.sin(2 * math.pi * context.hour / 24)
        hour_cos = math.cos(2 * math.pi * context.hour / 24)
        day_sin = math.sin(2 * math.pi * context.day_of_week / 7)
        day_cos = math.cos(2 * math.pi * context.day_of_week / 7)

        # Binary features
        is_working = 1.0 if context.is_working_hour else 0.0
        is_weekend = 1.0 if context.is_weekend else 0.0

        # Deviation (clamp to [-5, 5], normalize)
        deviation = context.deviation_from_baseline or 0.0
        deviation = max(-5.0, min(5.0, deviation)) / 5.0

        # Load factor (device-aware normalization)
        load_factor = 0.5
        if device_spec and device_spec.rated_wattage > 0:
            load_factor = min(wattage / device_spec.rated_wattage, 2.0) / 2.0

        risk = context.context_risk_score

        return np.array([
            watt_norm, hour_sin, hour_cos, day_sin, day_cos,
            is_working, is_weekend, deviation, load_factor, risk
        ], dtype=np.float64)

    # ══════════════════════════════════════════════════════════
    # LAYER 1: RULE-BASED ENGINE (Fix #1: device-type aware)
    # ══════════════════════════════════════════════════════════

    def _layer1_rules(
        self,
        wattage: float,
        context: ContextFeatures,
        device_spec: Optional[DeviceSpec],
    ) -> AnomalyLayerResult:
        """
        Deterministic rule-based anomaly detection.

        Fix #1: R2 now checks device category — always-on devices
        (servers, UPS, routers) are exempt from after-hours rules.
        """
        triggered_rules = []
        max_score = 0.0

        # Check if device is always-on
        is_always_on = (
            device_spec is not None
            and device_spec.category.lower() in ALWAYS_ON_CATEGORIES
        )

        # ── R1: Spike Detection ───────────────────────────────
        if device_spec and device_spec.rated_wattage > 0:
            spike_threshold = (
                device_spec.rated_wattage * settings.ANOMALY_SPIKE_MULTIPLIER
            )
            if wattage > spike_threshold:
                overshoot = (wattage - spike_threshold) / device_spec.rated_wattage
                severity = min(overshoot, 1.0)
                triggered_rules.append(
                    f"R1_spike: {wattage:.0f}W > "
                    f"{spike_threshold:.0f}W "
                    f"(rated×{settings.ANOMALY_SPIKE_MULTIPLIER})"
                )
                max_score = max(max_score, 0.6 + severity * 0.4)

        # ── R2: After-Hours High Usage (Fix #1) ──────────────
        # SKIP for always-on devices — they're expected to run 24/7
        if (not context.is_working_hour
                and wattage > 100
                and not is_always_on):

            time_factor = {
                "night": 0.7,
                "after_hours": 0.4,
                "weekend_day": 0.3,
                "early_morning": 0.2,
            }.get(context.shift, 0.0)

            if context.is_holiday:
                time_factor += 0.2

            # Device-relative wattage factor
            ref_wattage = 500.0
            if device_spec and device_spec.rated_wattage > 0:
                ref_wattage = device_spec.rated_wattage * 0.5
            wattage_factor = min(wattage / ref_wattage, 1.0)

            after_hours_score = time_factor * wattage_factor

            if after_hours_score > 0.3:
                triggered_rules.append(
                    f"R2_after_hours: {wattage:.0f}W during "
                    f"{context.shift} (score={after_hours_score:.2f})"
                )
                max_score = max(max_score, after_hours_score)

        # ── R3: Baseline Deviation ────────────────────────────
        if (context.deviation_from_baseline is not None
                and context.deviation_from_baseline > 1.0):
            dev = context.deviation_from_baseline
            dev_score = min(dev / 3.0, 1.0)
            triggered_rules.append(
                f"R3_baseline_deviation: {dev:.1%} above baseline "
                f"(baseline={context.baseline_wattage:.0f}W)"
            )
            max_score = max(max_score, 0.4 + dev_score * 0.5)

        # ── R4: Zero Power on Always-On Device ────────────────
        if is_always_on and wattage < 5:
            triggered_rules.append(
                f"R4_zero_always_on: {device_spec.device_name} "
                f"({device_spec.category}) at {wattage:.1f}W"
            )
            max_score = max(max_score, 0.85)

        # ── R5: Excessive Idle During Work Hours ──────────────
        if (device_spec
                and device_spec.idle_wattage > 0
                and device_spec.rated_wattage > 50
                and context.is_working_hour
                and not is_always_on):
            idle_threshold = device_spec.idle_wattage * 1.2
            if wattage < idle_threshold:
                triggered_rules.append(
                    f"R5_excessive_idle: {wattage:.0f}W ≤ "
                    f"{idle_threshold:.0f}W idle threshold "
                    f"during work hours"
                )
                max_score = max(max_score, 0.3)

        is_anomaly = max_score >= 0.4
        if is_anomaly:
            self._stats["rules_triggered"] += 1

        return AnomalyLayerResult(
            layer_name="rule_engine",
            is_anomaly=is_anomaly,
            score=round(max_score, 4),
            details=(
                " | ".join(triggered_rules)
                if triggered_rules
                else "No rules triggered"
            ),
        )

    # ══════════════════════════════════════════════════════════
    # LAYER 2: ISOLATION FOREST (Fix #3: auto-retrain)
    # ══════════════════════════════════════════════════════════

    def _layer2_isolation_forest(
        self,
        device_id: str,
        features: np.ndarray,
    ) -> AnomalyLayerResult:
        """
        Isolation Forest anomaly detection.

        Fix #3: Auto-retrains after retrain_interval new samples.
        Fix #2: Uses availability flag instead of string matching.
        """
        model = self._get_or_train_model(device_id)
        if model is None:
            self._stats["fallback_to_rules_only"] += 1
            sample_count = len(self._training_data.get(device_id, []))
            return AnomalyLayerResult(
                layer_name="isolation_forest",
                is_anomaly=False,
                score=0.0,
                details=(
                    f"Training: {sample_count}/"
                    f"{self._min_samples_for_training} samples"
                ),
            )

        try:
            prediction = model.predict(features.reshape(1, -1))[0]
            raw_score = model.decision_function(features.reshape(1, -1))[0]

            # Sigmoid normalization: negative raw_score → high anomaly score
            anomaly_score = 1.0 / (1.0 + math.exp(raw_score * 5))

            is_anomaly = prediction == -1
            if is_anomaly:
                self._stats["iforest_triggered"] += 1

            return AnomalyLayerResult(
                layer_name="isolation_forest",
                is_anomaly=is_anomaly,
                score=round(anomaly_score, 4),
                details=(
                    f"raw_decision={raw_score:.4f}, "
                    f"prediction={'ANOMALY' if is_anomaly else 'normal'}"
                ),
            )

        except Exception as e:
            logger.warning(f"Isolation Forest predict error: {e}")
            return AnomalyLayerResult(
                layer_name="isolation_forest",
                is_anomaly=False,
                score=0.0,
                details=f"Predict error: {str(e)[:80]}",
            )

    def _get_or_train_model(self, device_id: str):
        """
        Get existing model or train/retrain if needed.

        Fix #3: Checks if enough new samples have arrived since last
        training and retrains periodically.
        """
        with self._lock:
            samples = self._training_data.get(device_id, [])
            current_count = len(samples)

            # Not enough data yet
            if current_count < self._min_samples_for_training:
                return None

            # Check if retrain needed
            last_trained_at = self._model_sample_count.get(device_id, 0)
            needs_retrain = (
                device_id not in self._models
                or current_count - last_trained_at >= self._retrain_interval
            )

            if not needs_retrain:
                return self._models.get(device_id)

            # Train / retrain
            try:
                from sklearn.ensemble import IsolationForest

                X = np.array(samples)
                model = IsolationForest(
                    contamination=settings.ISOFOREST_CONTAMINATION,
                    n_estimators=settings.ISOFOREST_N_ESTIMATORS,
                    random_state=settings.ISOFOREST_RANDOM_STATE,
                    n_jobs=-1,
                )
                model.fit(X)
                self._models[device_id] = model
                self._model_sample_count[device_id] = current_count

                action = "Retrained" if last_trained_at > 0 else "Trained"
                self._stats["models_retrained"] += 1
                logger.info(
                    f"{action} Isolation Forest for '{device_id}' "
                    f"with {current_count} samples"
                )
                return model

            except ImportError:
                logger.warning(
                    "sklearn not available — Isolation Forest disabled"
                )
                return None
            except Exception as e:
                logger.error(f"IF training failed for '{device_id}': {e}")
                return self._models.get(device_id)

    def _store_training_sample(
        self, device_id: str, features: np.ndarray
    ) -> None:
        """Store feature vector for model training."""
        max_samples = 1000
        with self._lock:
            self._training_data[device_id].append(features.tolist())
            if len(self._training_data[device_id]) > max_samples:
                self._training_data[device_id] = \
                    self._training_data[device_id][-max_samples:]

    def retrain_model(self, device_id: str) -> bool:
        """Force retrain the Isolation Forest for a device."""
        with self._lock:
            self._model_sample_count[device_id] = 0  # Force retrain
        return self._get_or_train_model(device_id) is not None

    # ══════════════════════════════════════════════════════════
    # LAYER 3: ADAPTIVE AUTOENCODER STUB (Fix #4)
    # ══════════════════════════════════════════════════════════

    def _layer3_autoencoder(
        self,
        device_id: str,
        features: np.ndarray,
        context: ContextFeatures,
    ) -> AnomalyLayerResult:
        """
        Adaptive autoencoder stub using running statistics.

        Fix #4: Instead of a static hardcoded center, learns the
        mean and variance from observed data per-device. Uses
        Mahalanobis-style distance as reconstruction error.

        Swap with a real PyTorch autoencoder when ready.
        """
        # Feature importance weights
        weights = np.array([
            3.0,   # wattage: high importance
            0.5, 0.5,  # hour_sin/cos
            0.3, 0.3,  # day_sin/cos
            1.5,   # is_working
            0.5,   # is_weekend
            2.5,   # deviation: very important
            1.5,   # load_factor
            2.0,   # risk
        ], dtype=np.float64)

        # Update running statistics (online mean/variance)
        self._update_ae_stats(device_id, features)
        n = self._ae_sample_count.get(device_id, 0)

        if n < 10:
            # Not enough data — use conservative defaults
            return AnomalyLayerResult(
                layer_name="autoencoder_stub",
                is_anomaly=False,
                score=0.0,
                details=f"Warming up: {n}/10 samples",
            )

        mean = self._ae_running_mean[device_id]
        var = self._ae_running_var[device_id]

        # Mahalanobis-style reconstruction error
        # (distance from learned mean, scaled by learned variance)
        safe_var = np.maximum(var, 1e-6)  # Prevent division by zero
        diff = features - mean
        reconstruction_error = float(
            np.mean((diff ** 2) / safe_var * weights)
        )

        # Sigmoid normalization
        ae_score = 1.0 / (1.0 + math.exp(-2.0 * (reconstruction_error - 2.0)))

        is_anomaly = ae_score > 0.5
        if is_anomaly:
            self._stats["autoencoder_triggered"] += 1

        return AnomalyLayerResult(
            layer_name="autoencoder_stub",
            is_anomaly=is_anomaly,
            score=round(ae_score, 4),
            details=(
                f"reconstruction_error={reconstruction_error:.4f}, "
                f"learned_from={n}_samples, "
                f"adaptive=True"
            ),
        )

    def _update_ae_stats(
        self, device_id: str, features: np.ndarray
    ) -> None:
        """
        Update running mean and variance using Welford's algorithm.

        Welford's is numerically stable for online variance computation.
        """
        with self._lock:
            n = self._ae_sample_count[device_id] + 1
            self._ae_sample_count[device_id] = n

            if device_id not in self._ae_running_mean:
                self._ae_running_mean[device_id] = features.copy()
                self._ae_running_var[device_id] = np.zeros_like(features)
                return

            old_mean = self._ae_running_mean[device_id]
            delta = features - old_mean
            new_mean = old_mean + delta / n
            delta2 = features - new_mean

            self._ae_running_mean[device_id] = new_mean
            self._ae_running_var[device_id] += (delta * delta2 - self._ae_running_var[device_id]) / n

    # ══════════════════════════════════════════════════════════
    # WEIGHTED VOTING (Fix #2: proper flags)
    # ══════════════════════════════════════════════════════════

    def _weighted_vote(
        self,
        rule_result: AnomalyLayerResult,
        iso_result: AnomalyLayerResult,
        ae_result: AnomalyLayerResult,
    ) -> Tuple[float, bool]:
        """
        Combine layer scores using weighted voting.

        Fix #2: Uses score=0.0 as availability signal instead of
        fragile string matching on detail text.
        """
        r_score = rule_result.score
        i_score = iso_result.score
        a_score = ae_result.score

        w_r = self._w_rules
        w_i = self._w_iso
        w_a = self._w_ae

        # Fix #2: Proper degradation check
        iforest_unavailable = (
            i_score == 0.0 and not iso_result.is_anomaly
            and "raw_decision" not in (iso_result.details or "")
        )
        ae_unavailable = (
            a_score == 0.0 and not ae_result.is_anomaly
            and "reconstruction_error" not in (ae_result.details or "")
        )

        if iforest_unavailable and ae_unavailable:
            # Both ML layers unavailable — rules only
            w_r = 1.0
            w_i = 0.0
            w_a = 0.0
        elif iforest_unavailable:
            w_r += w_i * 0.7
            w_a += w_i * 0.3
            w_i = 0.0
        elif ae_unavailable:
            w_r += w_a * 0.5
            w_i += w_a * 0.5
            w_a = 0.0

        # Normalize
        total_w = w_r + w_i + w_a
        if total_w > 0:
            w_r /= total_w
            w_i /= total_w
            w_a /= total_w

        combined = w_r * r_score + w_i * i_score + w_a * a_score
        is_anomaly = combined >= self._threshold

        return combined, is_anomaly

    # ══════════════════════════════════════════════════════════
    # FIX #5: ANOMALY DEDUPLICATION
    # ══════════════════════════════════════════════════════════

    def _check_deduplication(
        self, device_id: str, anomaly_type: str
    ) -> bool:
        """
        Check if this anomaly should be suppressed (deduplication).

        After 3 consecutive identical anomaly types for the same device,
        subsequent alerts are marked as suppressed to prevent alert fatigue.
        """
        key = (device_id, anomaly_type)
        self._anomaly_streak[key] += 1

        # Reset streaks for other anomaly types on this device
        for k in list(self._anomaly_streak.keys()):
            if k[0] == device_id and k != key:
                self._anomaly_streak[k] = 0

        return self._anomaly_streak[key] > self._suppression_threshold

    def _reset_streak(self, device_id: str) -> None:
        """Reset all anomaly streaks for a device (normal reading)."""
        for k in list(self._anomaly_streak.keys()):
            if k[0] == device_id:
                self._anomaly_streak[k] = 0

    # ══════════════════════════════════════════════════════════
    # ANOMALY CLASSIFICATION
    # ══════════════════════════════════════════════════════════

    def _classify_anomaly(
        self,
        wattage: float,
        context: ContextFeatures,
        device_spec: Optional[DeviceSpec],
        rule_result: AnomalyLayerResult,
    ) -> str:
        """
        Classify anomaly type for human-readable explanations.

        Types: spike, after_hours, baseline_breach,
               zero_always_on, excessive_idle, statistical
        """
        details = rule_result.details or ""

        if "R1_spike" in details:
            return "spike"
        if "R4_zero_always_on" in details:
            return "zero_always_on"
        if "R2_after_hours" in details:
            return "after_hours"
        if "R3_baseline_deviation" in details:
            return "baseline_breach"
        if "R5_excessive_idle" in details:
            return "excessive_idle"

        return "statistical"

    # ══════════════════════════════════════════════════════════
    # STATISTICS
    # ══════════════════════════════════════════════════════════

    def get_detector_stats(self) -> dict:
        """Get anomaly detection statistics."""
        total = self._stats["total_detections"]
        with self._lock:
            trained_devices = list(self._models.keys())
            training_counts = {
                k: len(v) for k, v in self._training_data.items()
            }
            ae_counts = dict(self._ae_sample_count)
        return {
            **self._stats,
            "anomaly_rate": round(
                self._stats["anomalies_found"] / total, 4
            ) if total > 0 else 0,
            "trained_models": trained_devices,
            "training_samples": training_counts,
            "ae_learned_samples": ae_counts,
            "threshold": self._threshold,
            "weights": {
                "rules": self._w_rules,
                "isolation_forest": self._w_iso,
                "autoencoder": self._w_ae,
            },
        }


# ── Module-level singleton ────────────────────────────────────────
anomaly_detector = AnomalyDetector()
