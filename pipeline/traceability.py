"""
CTrackAI — Traceability & Logging (Stage 10)

SQLite-backed audit trail for all pipeline outputs.
Every reading → pipeline result is stored with full traceability.

Features:
    - Async-compatible SQLite storage (thread-safe)
    - Auto-create tables on init
    - Queryable by device_id, time range, severity, anomaly type
    - Structured JSON logging via loguru
    - Stats and health monitoring
"""

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional, List

from loguru import logger

from models.schemas import PipelineOutput
from config.settings import settings


class TraceabilityLogger:
    """
    SQLite-backed structured logger for pipeline results.

    Thread-safe — uses a single writer connection with locks.

    Usage:
        tlogger = TraceabilityLogger()
        tlogger.log_result(pipeline_output)
        results = tlogger.query_device("LAB_01", limit=20)
    """

    def __init__(self, db_path: str = None):
        self._db_path = db_path or settings.SQLITE_DB_PATH
        self._lock = threading.Lock()

        # Ensure data directory exists
        import os
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)

        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

        self._stats = {
            "total_logged": 0,
            "anomalies_logged": 0,
            "errors": 0,
        }

        logger.info(f"TraceabilityLogger initialized: db={self._db_path}")

    def _create_tables(self):
        """Create pipeline_logs table if it doesn't exist."""
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    log_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    circuit_id TEXT,
                    sensor_reading_watts REAL,
                    energy_kwh REAL,
                    co2_kg REAL,
                    emission_factor_used REAL,
                    data_quality_status TEXT,
                    anomaly_detected INTEGER DEFAULT 0,
                    anomaly_score REAL,
                    anomaly_type TEXT,
                    severity TEXT,
                    severity_score REAL,
                    explanation TEXT,
                    is_degraded INTEGER DEFAULT 0,
                    processing_time_ms INTEGER,
                    full_output_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_device_ts
                ON pipeline_logs(device_id, timestamp)
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_anomaly
                ON pipeline_logs(anomaly_detected, severity)
            """)
            self._conn.commit()

    def log_result(self, output: PipelineOutput) -> bool:
        """
        Store a pipeline result in SQLite.

        Args:
            output: Complete pipeline output

        Returns:
            True if stored successfully
        """
        try:
            with self._lock:
                self._conn.execute("""
                    INSERT OR REPLACE INTO pipeline_logs (
                        log_id, timestamp, device_id, circuit_id,
                        sensor_reading_watts, energy_kwh, co2_kg,
                        emission_factor_used, data_quality_status,
                        anomaly_detected, anomaly_score, anomaly_type,
                        severity, severity_score, explanation,
                        is_degraded, processing_time_ms, full_output_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    output.log_id,
                    output.timestamp.isoformat(),
                    output.device_id,
                    output.circuit_id,
                    output.sensor_reading_watts,
                    output.energy_kwh,
                    output.co2_kg,
                    output.emission_factor_used,
                    output.data_quality_status.value,
                    1 if output.anomaly_detected else 0,
                    output.anomaly_score,
                    output.anomaly_type,
                    output.severity.value if output.severity else None,
                    output.severity_score,
                    output.explanation,
                    1 if output.is_degraded else 0,
                    output.processing_time_ms,
                    output.model_dump_json(),
                ))
                self._conn.commit()

            self._stats["total_logged"] += 1
            if output.anomaly_detected:
                self._stats["anomalies_logged"] += 1

            return True

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Failed to log result: {e}")
            return False

    def query_device(
        self,
        device_id: str,
        limit: int = 20,
        anomalies_only: bool = False,
    ) -> List[dict]:
        """Query pipeline logs for a device."""
        with self._lock:
            sql = "SELECT * FROM pipeline_logs WHERE device_id = ?"
            params = [device_id]

            if anomalies_only:
                sql += " AND anomaly_detected = 1"

            sql += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            rows = self._conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def query_anomalies(
        self,
        severity: Optional[str] = None,
        limit: int = 50,
    ) -> List[dict]:
        """Query latest anomalies across all devices."""
        with self._lock:
            sql = "SELECT * FROM pipeline_logs WHERE anomaly_detected = 1"
            params = []

            if severity:
                sql += " AND severity = ?"
                params.append(severity)

            sql += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            rows = self._conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def get_device_summary(self, device_id: str) -> dict:
        """Get summary statistics for a device."""
        with self._lock:
            row = self._conn.execute("""
                SELECT
                    COUNT(*) as total_readings,
                    SUM(energy_kwh) as total_energy_kwh,
                    SUM(co2_kg) as total_co2_kg,
                    SUM(anomaly_detected) as anomaly_count,
                    AVG(co2_kg) as avg_co2_per_reading,
                    MIN(timestamp) as first_reading,
                    MAX(timestamp) as last_reading
                FROM pipeline_logs
                WHERE device_id = ?
            """, (device_id,)).fetchone()
            return dict(row) if row else {}

    def get_logger_stats(self) -> dict:
        """Get logger statistics."""
        with self._lock:
            count_row = self._conn.execute(
                "SELECT COUNT(*) as total FROM pipeline_logs"
            ).fetchone()
            total_stored = count_row["total"] if count_row else 0

        return {
            **self._stats,
            "total_stored": total_stored,
            "db_path": self._db_path,
        }

    def close(self):
        """Close the SQLite connection."""
        with self._lock:
            self._conn.close()
        logger.info("TraceabilityLogger closed")


# ── Module-level singleton ────────────────────────────────────────
trace_logger = TraceabilityLogger()
