"""SQLite storage for session grading data.

All grading results live in ``~/.routellect/grades.db`` so they can be
inspected locally, exported for sharing, or synced to a server later.
"""

from __future__ import annotations

import csv
import io
import sqlite3
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_ROUTELLECT_DIR = Path.home() / ".routellect"
_DB_PATH = _ROUTELLECT_DIR / "grades.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    message_count INTEGER DEFAULT 0,
    batch_size INTEGER,
    grading_cost_usd REAL,
    grader_model TEXT,
    avg_confidence REAL
);

CREATE TABLE IF NOT EXISTS grades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    message_index INTEGER NOT NULL,
    model_used TEXT NOT NULL,
    provider TEXT NOT NULL,
    grade TEXT NOT NULL,
    confidence REAL,
    reason TEXT,
    is_exploration INTEGER DEFAULT 0,
    graded_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS routing_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    message_index INTEGER NOT NULL,
    model_used TEXT NOT NULL,
    provider TEXT NOT NULL,
    is_exploration INTEGER DEFAULT 0,
    latency_ms INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_grades_session ON grades(session_id);
CREATE INDEX IF NOT EXISTS idx_grades_model ON grades(model_used);
CREATE INDEX IF NOT EXISTS idx_routing_session ON routing_log(session_id);
"""


@dataclass
class GradeRecord:
    session_id: str
    message_index: int
    model_used: str
    provider: str
    grade: str  # "pass", "mixed", "fail"
    confidence: float
    reason: str
    is_exploration: bool = False


@dataclass
class RoutingRecord:
    session_id: str
    message_index: int
    model_used: str
    provider: str
    is_exploration: bool
    latency_ms: int
    input_tokens: int
    output_tokens: int


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_db(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or _DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def ensure_session(session_id: str, db_path: Path | None = None) -> None:
    conn = _get_db(db_path)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, started_at) VALUES (?, ?)",
            (session_id, _now_iso()),
        )
        conn.commit()
    finally:
        conn.close()


def log_routing(record: RoutingRecord, db_path: Path | None = None) -> None:
    conn = _get_db(db_path)
    try:
        ensure_session(record.session_id, db_path)
        conn.execute(
            """INSERT INTO routing_log
               (session_id, message_index, model_used, provider, is_exploration,
                latency_ms, input_tokens, output_tokens, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.session_id,
                record.message_index,
                record.model_used,
                record.provider,
                int(record.is_exploration),
                record.latency_ms,
                record.input_tokens,
                record.output_tokens,
                _now_iso(),
            ),
        )
        conn.execute(
            "UPDATE sessions SET message_count = message_count + 1 WHERE session_id = ?",
            (record.session_id,),
        )
        conn.commit()
    finally:
        conn.close()


def save_grades(
    grades: list[GradeRecord],
    session_id: str,
    batch_size: int,
    grading_cost_usd: float,
    grader_model: str,
    avg_confidence: float,
    db_path: Path | None = None,
) -> None:
    conn = _get_db(db_path)
    try:
        now = _now_iso()
        for g in grades:
            conn.execute(
                """INSERT INTO grades
                   (session_id, message_index, model_used, provider, grade,
                    confidence, reason, is_exploration, graded_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    g.session_id,
                    g.message_index,
                    g.model_used,
                    g.provider,
                    g.grade,
                    g.confidence,
                    g.reason,
                    int(g.is_exploration),
                    now,
                ),
            )
        conn.execute(
            """UPDATE sessions
               SET ended_at = ?, batch_size = ?, grading_cost_usd = ?,
                   grader_model = ?, avg_confidence = ?
               WHERE session_id = ?""",
            (now, batch_size, grading_cost_usd, grader_model, avg_confidence, session_id),
        )
        conn.commit()
    finally:
        conn.close()


def query_recent_grades(limit: int = 30, db_path: Path | None = None) -> list[dict]:
    conn = _get_db(db_path)
    try:
        rows = conn.execute(
            """SELECT g.session_id, g.message_index, g.model_used, g.provider,
                      g.grade, g.confidence, g.reason, g.is_exploration, g.graded_at
               FROM grades g
               ORDER BY g.graded_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def query_model_stats(db_path: Path | None = None) -> list[dict]:
    conn = _get_db(db_path)
    try:
        rows = conn.execute(
            """SELECT model_used, provider, grade, COUNT(*) as count,
                      AVG(confidence) as avg_confidence
               FROM grades
               GROUP BY model_used, provider, grade
               ORDER BY model_used, grade""",
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _table_to_csv(conn: sqlite3.Connection, table: str) -> str:
    cursor = conn.execute(f"SELECT * FROM {table}")  # noqa: S608 — table name is hardcoded below
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(columns)
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()


def export_zip(output_path: Path, db_path: Path | None = None) -> Path:
    """Export all grading data as a ZIP containing CSV files.

    Produces a clean ZIP with:
      - sessions.csv
      - grades.csv
      - routing_log.csv
      - model_summary.csv

    Args:
        output_path: Where to write the ZIP file.
        db_path: Override DB path (for testing).

    Returns:
        Path to the written ZIP file.
    """
    conn = _get_db(db_path)
    try:
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for table in ("sessions", "grades", "routing_log"):
                zf.writestr(f"{table}.csv", _table_to_csv(conn, table))

            # Model summary
            stats = conn.execute(
                """SELECT
                       model_used,
                       provider,
                       COUNT(*) as total_grades,
                       SUM(CASE WHEN grade = 'pass' THEN 1 ELSE 0 END) as pass_count,
                       SUM(CASE WHEN grade = 'fail' THEN 1 ELSE 0 END) as fail_count,
                       SUM(CASE WHEN grade = 'mixed' THEN 1 ELSE 0 END) as mixed_count,
                       AVG(confidence) as avg_confidence,
                       ROUND(CAST(SUM(CASE WHEN grade = 'pass' THEN 1 ELSE 0 END) AS REAL)
                             / COUNT(*) * 100, 1) as pass_rate_pct
                   FROM grades
                   GROUP BY model_used, provider
                   ORDER BY pass_rate_pct DESC"""
            )
            columns = [desc[0] for desc in stats.description]
            rows = stats.fetchall()
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(columns)
            for row in rows:
                writer.writerow(row)
            zf.writestr("model_summary.csv", buf.getvalue())

        return output_path
    finally:
        conn.close()
