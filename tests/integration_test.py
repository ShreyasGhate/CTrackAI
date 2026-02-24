"""
CTrackAI — Integration Test (Stage 13)

Exercises the full 10-stage pipeline with synthetic data.

Test Plan:
    1. Generate 1 day of synthetic readings (5760 readings)
    2. Process a sample of 20 readings through the orchestrator
    3. Validate all pipeline outputs
    4. Verify anomaly detection, severity, and explanations
    5. Check traceability logs
    6. Print comprehensive statistics

Run:
    python -m tests.integration_test
"""

import warnings
warnings.filterwarnings("ignore")

import time
import random
from datetime import datetime, timezone, timedelta

from models.schemas import AggregatedReading, DataQualityStatus
from synthetic.generator import SyntheticDataGenerator
from pipeline.orchestrator import PipelineOrchestrator


def run_integration_test():
    """Run full pipeline integration test."""
    print("=" * 60)
    print("  CTrackAI — FULL PIPELINE INTEGRATION TEST")
    print("=" * 60)
    print()

    # ── Setup ─────────────────────────────────────────────
    gen = SyntheticDataGenerator(seed=42)
    orch = PipelineOrchestrator()

    # ── Generate synthetic data ───────────────────────────
    print("📊 Generating synthetic data...")
    date = datetime(2026, 2, 23, tzinfo=timezone.utc)
    all_readings = gen.generate_day(date=date, anomaly_rate=0.03)
    print(f"   Generated {len(all_readings)} readings")
    print(f"   Injected anomalies: {len(gen.get_anomaly_log())}")
    print()

    # ── Sample diverse readings ───────────────────────────
    # Pick readings that cover different device types and times
    rng = random.Random(42)
    sample_size = 20
    sample = rng.sample(all_readings, min(sample_size, len(all_readings)))

    # ── Process through pipeline ──────────────────────────
    print(f"🔄 Processing {len(sample)} readings through full pipeline...")
    print("-" * 60)

    results = []
    anomaly_count = 0
    degraded_count = 0
    total_time = 0

    for i, reading in enumerate(sample, 1):
        result = orch.process(reading)
        results.append(result)
        total_time += result.processing_time_ms

        if result.anomaly_detected:
            anomaly_count += 1

        if result.is_degraded:
            degraded_count += 1

        status = "🔴 ANOMALY" if result.anomaly_detected else "✅ NORMAL"
        severity = f" [{result.severity.value}]" if result.severity else ""
        anomaly_type = f" ({result.anomaly_type})" if result.anomaly_type else ""

        print(
            f"  [{i:2d}/{sample_size}] {reading.device_id:12s} "
            f"{reading.avg_wattage:7.1f}W → "
            f"{result.co2_kg:.4f} kg CO₂ | "
            f"{status}{severity}{anomaly_type} "
            f"({result.processing_time_ms}ms)"
        )

    # ── Results Summary ───────────────────────────────────
    print()
    print("=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)

    print(f"\n  📈 Readings processed:  {len(results)}")
    print(f"  🔴 Anomalies detected: {anomaly_count}")
    print(f"  ⚠️  Degraded outputs:   {degraded_count}")
    print(f"  ⏱️  Avg processing:     {total_time / len(results):.0f}ms")
    print(f"  ⏱️  Total time:         {total_time}ms")

    # ── Validate outputs ──────────────────────────────────
    print("\n  📋 VALIDATION CHECKS:")

    # Check 1: All outputs have valid fields
    all_valid = all(
        r.log_id and r.device_id and r.energy_kwh >= 0 and r.co2_kg >= 0
        for r in results
    )
    print(f"  ✅ All outputs have valid fields: {all_valid}")

    # Check 2: All outputs have quality status
    all_quality = all(
        r.data_quality_status in DataQualityStatus
        for r in results
    )
    print(f"  ✅ All have quality status: {all_quality}")

    # Check 3: Anomalies have severity
    anomaly_results = [r for r in results if r.anomaly_detected]
    all_severity = all(
        r.severity is not None and r.severity_score is not None
        for r in anomaly_results
    )
    print(f"  ✅ All anomalies have severity: {all_severity}")

    # Check 4: Anomalies have explanations
    all_explained = all(
        r.explanation is not None and len(r.explanation) > 10
        for r in anomaly_results
    )
    print(f"  ✅ All anomalies have explanations: {all_explained}")

    # Check 5: CO₂ calculations are reasonable
    co2_values = [r.co2_kg for r in results]
    co2_reasonable = all(0 <= c <= 10 for c in co2_values)
    print(f"  ✅ CO₂ values reasonable (0-10 kg): {co2_reasonable}")

    # Check 6: Processing times are acceptable
    times = [r.processing_time_ms for r in results]
    # First call is slow (model loading), subsequent should be fast
    fast_calls = [t for t in times[1:] if t < 500]
    print(f"  ✅ Fast processing (<500ms after warmup): {len(fast_calls)}/{len(times)-1}")

    # ── Pipeline component stats ──────────────────────────
    print("\n  📊 PIPELINE COMPONENT STATS:")

    stats = orch.get_orchestrator_stats()
    print(f"  Orchestrator: {stats}")

    try:
        from pipeline.forecast_engine import forecast_engine
        fc_stats = forecast_engine.get_forecast_stats()
        print(f"  Forecasting:  {fc_stats['linear_used']} linear, "
              f"{fc_stats['tracked_devices'][:5]}... tracked")
    except Exception:
        pass

    try:
        from pipeline.traceability import trace_logger
        tl_stats = trace_logger.get_logger_stats()
        print(f"  Traceability: {tl_stats['total_logged']} logged, "
              f"{tl_stats['anomalies_logged']} anomalies")
    except Exception:
        pass

    # ── Anomaly detail ────────────────────────────────────
    if anomaly_results:
        print("\n  🔴 ANOMALY DETAILS:")
        for r in anomaly_results:
            print(
                f"    {r.device_id}: {r.anomaly_type} "
                f"(score={r.anomaly_score:.2f}, "
                f"severity={r.severity.value}, "
                f"score={r.severity_score:.3f})"
            )
            if r.explanation:
                # Show first 120 chars
                print(f"      → {r.explanation[:120]}...")

    # ── Forecast check ────────────────────────────────────
    print("\n  📈 FORECAST CHECK:")
    try:
        from pipeline.forecast_engine import forecast_engine
        for device in ["LAB_PC_01", "LAB_SRV_01"]:
            fc = forecast_engine.forecast(device)
            if fc.sufficient_history:
                print(
                    f"    {device}: {fc.forecast_model}, "
                    f"next_day={fc.next_day_estimate_kwh:.2f} kWh"
                )
            else:
                print(f"    {device}: insufficient history "
                      f"({forecast_engine.get_series_length(device)} readings)")
    except Exception as e:
        print(f"    Forecast check failed: {e}")

    # ── Final Verdict ─────────────────────────────────────
    print()
    print("=" * 60)
    passed = all([all_valid, all_quality, all_severity, all_explained, co2_reasonable])
    if passed:
        print("  🎉 ALL INTEGRATION TESTS PASSED!")
    else:
        print("  ❌ SOME TESTS FAILED — review output above")
    print("=" * 60)

    return passed


if __name__ == "__main__":
    run_integration_test()
