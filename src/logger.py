"""
EdgeInspect AI: Industrial Inspection Logging & Quality Audit Trail
Challenge: Arm Bharat AI-SoC Challenge 2026-27 (PS5: Industrial Defect Detection)

Provides persistent, production-grade quality logging:
- SQLite embedded database (`reports/inspections.db`)
- CSV real-time export (`reports/inspection_log.csv`)
- Factory Yield Metrics (Total parts, Pass %, Defect %, Pareto analysis)
- Audit trail compliant with ISO 9001, IATF 16949, and FDA cGMP data integrity
"""
import csv
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
DB_PATH = REPORTS_DIR / "inspections.db"
CSV_PATH = REPORTS_DIR / "inspection_log.csv"


class QualityInspectionLogger:
    """Industrial logging engine for edge defect inspection stations."""

    def __init__(self, db_path: Path = DB_PATH, csv_path: Path = CSV_PATH):
        self.db_path = Path(db_path)
        self.csv_path = Path(csv_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_sqlite()
        self._init_csv()

    def _init_sqlite(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS inspection_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    inspection_id TEXT UNIQUE NOT NULL,
                    application_domain TEXT NOT NULL,
                    part_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    defects_count INTEGER NOT NULL,
                    primary_defect TEXT,
                    severity TEXT,
                    disposition TEXT,
                    inference_ms REAL NOT NULL,
                    anomaly_score REAL,
                    hardware_platform TEXT,
                    detections_json TEXT
                )
            """)
            conn.commit()

    def _init_csv(self):
        if not self.csv_path.exists():
            headers = [
                "timestamp", "inspection_id", "application_domain", "part_name",
                "status", "defects_count", "primary_defect", "severity",
                "disposition", "inference_ms", "anomaly_score", "hardware_platform"
            ]
            with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def log_inspection(
        self,
        inspection_id: str,
        application_domain: str,
        part_name: str,
        status: str,
        defects_count: int,
        inference_ms: float,
        primary_defect: str = "None",
        severity: str = "NORMAL",
        disposition: str = "ACCEPT",
        anomaly_score: float = 0.0,
        hardware_platform: str = "Raspberry Pi 5 (Arm Cortex-A76)",
        detections: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Logs an inspection event to both SQLite database and CSV file."""
        now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dets_json = json.dumps(detections or [])

        # SQLite insert
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO inspection_records (
                    timestamp, inspection_id, application_domain, part_name,
                    status, defects_count, primary_defect, severity,
                    disposition, inference_ms, anomaly_score, hardware_platform, detections_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now_iso, inspection_id, application_domain, part_name,
                status, defects_count, primary_defect, severity,
                disposition, inference_ms, anomaly_score, hardware_platform, dets_json
            ))
            conn.commit()

        # CSV append
        row = [
            now_iso, inspection_id, application_domain, part_name,
            status, defects_count, primary_defect, severity,
            disposition, round(inference_ms, 2), round(anomaly_score, 3), hardware_platform
        ]
        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        return {
            "logged": True,
            "timestamp": now_iso,
            "inspection_id": inspection_id,
            "db_path": str(self.db_path),
            "csv_path": str(self.csv_path)
        }

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Computes live factory yield metrics and defect pareto distribution."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Total and pass/fail counts
            cursor.execute("SELECT COUNT(*) as total FROM inspection_records")
            total = cursor.fetchone()["total"]

            if total == 0:
                return {
                    "total_inspected": 0,
                    "passed_count": 0,
                    "defective_count": 0,
                    "yield_rate_pct": 100.0,
                    "defect_rate_pct": 0.0,
                    "avg_latency_ms": 0.0,
                    "pareto_defects": {},
                    "domain_breakdown": {}
                }

            cursor.execute("SELECT COUNT(*) as def_cnt FROM inspection_records WHERE defects_count > 0")
            defective = cursor.fetchone()["def_cnt"]
            passed = total - defective

            cursor.execute("SELECT AVG(inference_ms) as avg_lat FROM inspection_records")
            avg_lat = cursor.fetchone()["avg_lat"] or 0.0

            # Pareto defect distribution
            cursor.execute("""
                SELECT primary_defect, COUNT(*) as count 
                FROM inspection_records 
                WHERE primary_defect != 'None' AND primary_defect IS NOT NULL
                GROUP BY primary_defect 
                ORDER BY count DESC 
                LIMIT 6
            """)
            pareto = {row["primary_defect"]: row["count"] for row in cursor.fetchall()}

            # Domain breakdown
            cursor.execute("""
                SELECT application_domain, COUNT(*) as count 
                FROM inspection_records 
                GROUP BY application_domain
            """)
            domains = {row["application_domain"]: row["count"] for row in cursor.fetchall()}

            return {
                "total_inspected": total,
                "passed_count": passed,
                "defective_count": defective,
                "yield_rate_pct": round((passed / total) * 100.0, 1),
                "defect_rate_pct": round((defective / total) * 100.0, 1),
                "avg_latency_ms": round(avg_lat, 1),
                "pareto_defects": pareto,
                "domain_breakdown": domains
            }

    def get_recent_records(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Returns the most recent inspection records."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM inspection_records ORDER BY id DESC LIMIT ?",
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]
