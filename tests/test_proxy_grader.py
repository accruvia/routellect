"""Tests for the session grader and grades DB."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from routellect.protocols import RoutingDecision


class TestGradesDB:
    def test_save_and_query_grades(self, tmp_path: Path):
        from routellect.proxy._grades_db import (
            GradeRecord,
            ensure_session,
            query_model_stats,
            query_recent_grades,
            save_grades,
        )

        db = tmp_path / "grades.db"
        ensure_session("sess-1", db_path=db)
        grades = [
            GradeRecord("sess-1", 0, "gpt-4o", "openai", "pass", 0.9, "user continued"),
            GradeRecord("sess-1", 1, "claude-sonnet-4-6", "anthropic", "fail", 0.8, "user corrected"),
        ]
        save_grades(grades, "sess-1", batch_size=10, grading_cost_usd=0.005,
                     grader_model="haiku", avg_confidence=0.85, db_path=db)

        recent = query_recent_grades(limit=10, db_path=db)
        assert len(recent) == 2
        assert recent[0]["grade"] in ("pass", "fail")

        stats = query_model_stats(db_path=db)
        assert len(stats) == 2
        models = {s["model_used"] for s in stats}
        assert "gpt-4o" in models
        assert "claude-sonnet-4-6" in models

    def test_routing_log(self, tmp_path: Path):
        from routellect.proxy._grades_db import RoutingRecord, log_routing, ensure_session

        db = tmp_path / "grades.db"
        ensure_session("sess-2", db_path=db)
        log_routing(RoutingRecord(
            session_id="sess-2",
            message_index=0,
            model_used="gpt-4o",
            provider="openai",
            is_exploration=False,
            latency_ms=500,
            input_tokens=100,
            output_tokens=50,
        ), db_path=db)

        import sqlite3
        conn = sqlite3.connect(str(db))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM routing_log WHERE session_id = 'sess-2'").fetchall()
        conn.close()
        assert len(rows) == 1
        assert dict(rows[0])["model_used"] == "gpt-4o"

    def test_export_zip(self, tmp_path: Path):
        import zipfile

        from routellect.proxy._grades_db import (
            GradeRecord,
            ensure_session,
            export_zip,
            save_grades,
        )

        db = tmp_path / "grades.db"
        ensure_session("sess-export", db_path=db)
        save_grades(
            [GradeRecord("sess-export", 0, "gpt-4o", "openai", "pass", 0.9, "good")],
            "sess-export", batch_size=5, grading_cost_usd=0.003,
            grader_model="haiku", avg_confidence=0.9, db_path=db,
        )

        zip_path = tmp_path / "export.zip"
        export_zip(zip_path, db_path=db)

        assert zip_path.exists()
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            assert "sessions.csv" in names
            assert "grades.csv" in names
            assert "routing_log.csv" in names
            assert "model_summary.csv" in names

            summary = zf.read("model_summary.csv").decode()
            assert "gpt-4o" in summary
            assert "pass_rate_pct" in summary


class TestGrader:
    def _make_grader(self):
        from routellect.proxy._grader import Grader

        return Grader(
            credentials={"openai": "sk-test"},
            selector=None,
            batch_size=3,
            idle_seconds=1,
        )

    def test_record_exchange_creates_session(self):
        grader = self._make_grader()
        decision = RoutingDecision(model_id="gpt-4o", backend="openai", confidence=0.8)

        with patch("routellect.proxy._grader.ensure_session"), \
             patch("routellect.proxy._grader.log_routing"):
            grader.record_exchange(
                session_id="sess-a",
                message_index=0,
                user_message="hello",
                assistant_response="hi there",
                decision=decision,
            )

        assert "sess-a" in grader._sessions
        assert grader._sessions["sess-a"].size == 1

    def test_should_grade_on_batch_full(self):
        grader = self._make_grader()
        decision = RoutingDecision(model_id="gpt-4o", backend="openai", confidence=0.8)

        with patch("routellect.proxy._grader.ensure_session"), \
             patch("routellect.proxy._grader.log_routing"):
            for i in range(3):
                grader.record_exchange(
                    session_id="sess-b",
                    message_index=i,
                    user_message=f"msg {i}",
                    assistant_response=f"resp {i}",
                    decision=decision,
                )

        assert grader.should_grade("sess-b") is True

    def test_should_not_grade_below_batch(self):
        grader = self._make_grader()
        decision = RoutingDecision(model_id="gpt-4o", backend="openai", confidence=0.8)

        with patch("routellect.proxy._grader.ensure_session"), \
             patch("routellect.proxy._grader.log_routing"):
            grader.record_exchange(
                session_id="sess-c",
                message_index=0,
                user_message="msg",
                assistant_response="resp",
                decision=decision,
            )

        assert grader.should_grade("sess-c") is False

    def test_parse_grades_valid(self):
        grader = self._make_grader()
        from routellect.proxy._grader import SessionBuffer, _Exchange

        buf = SessionBuffer(session_id="sess-parse")
        buf.exchanges = [
            _Exchange(
                message_index=0, user_message="hi", assistant_response="hello",
                model_used="gpt-4o", provider="openai",
                decision=RoutingDecision(model_id="gpt-4o", backend="openai", confidence=0.8),
                latency_ms=100, input_tokens=10, output_tokens=5,
            ),
            _Exchange(
                message_index=1, user_message="no that's wrong", assistant_response="sorry",
                model_used="claude-sonnet-4-6", provider="anthropic",
                decision=RoutingDecision(model_id="claude-sonnet-4-6", backend="anthropic", confidence=0.7),
                latency_ms=200, input_tokens=20, output_tokens=10,
            ),
        ]

        raw = json.dumps([
            {"index": 0, "grade": "pass", "confidence": 0.9, "reason": "user continued"},
            {"index": 1, "grade": "fail", "confidence": 0.85, "reason": "user corrected"},
        ])

        records = grader._parse_grades(raw, buf)
        assert len(records) == 2
        assert records[0].grade == "pass"
        assert records[0].model_used == "gpt-4o"
        assert records[1].grade == "fail"
        assert records[1].model_used == "claude-sonnet-4-6"

    def test_parse_grades_handles_markdown_fences(self):
        grader = self._make_grader()
        from routellect.proxy._grader import SessionBuffer, _Exchange

        buf = SessionBuffer(session_id="sess-md")
        buf.exchanges = [
            _Exchange(
                message_index=0, user_message="hi", assistant_response="hello",
                model_used="gpt-4o", provider="openai",
                decision=RoutingDecision(model_id="gpt-4o", backend="openai", confidence=0.8),
                latency_ms=100, input_tokens=10, output_tokens=5,
            ),
        ]

        raw = '```json\n[{"index": 0, "grade": "pass", "confidence": 0.9, "reason": "good"}]\n```'
        records = grader._parse_grades(raw, buf)
        assert len(records) == 1
        assert records[0].grade == "pass"

    def test_parse_grades_invalid_json(self):
        grader = self._make_grader()
        from routellect.proxy._grader import SessionBuffer

        buf = SessionBuffer(session_id="sess-bad")
        records = grader._parse_grades("not json at all", buf)
        assert records == []

    def test_flush_session(self):
        grader = self._make_grader()
        decision = RoutingDecision(model_id="gpt-4o", backend="openai", confidence=0.8)

        with patch("routellect.proxy._grader.ensure_session"), \
             patch("routellect.proxy._grader.log_routing"):
            grader.record_exchange(
                session_id="sess-flush",
                message_index=0,
                user_message="msg",
                assistant_response="resp",
                decision=decision,
            )

        assert "sess-flush" in grader._sessions
        grader.flush_session("sess-flush")
        assert "sess-flush" not in grader._sessions
