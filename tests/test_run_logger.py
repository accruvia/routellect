"""Tests for RunLogger A/B run telemetry logging."""

import datetime as dt
import json
from unittest.mock import patch

from routellect.telemetry.run_logger import MAX_RESPONSE_BYTES, TRUNCATION_MARKER, RunLogger


def test_init_creates_output_directory(tmp_path):
    out_dir = tmp_path / "ab_run"
    assert not out_dir.exists()
    RunLogger(out_dir)
    assert out_dir.is_dir()


def test_log_iteration_appends_jsonl_entries(tmp_path):
    logger = RunLogger(tmp_path / "ab_run")
    fixed_time = dt.datetime(2026, 3, 7, 12, 0, 0, tzinfo=dt.UTC)

    with patch("routellect.telemetry.run_logger.dt") as mock_dt:
        mock_dt.datetime.now.return_value = fixed_time
        mock_dt.UTC = dt.UTC
        logger.log_iteration(
            iteration=1,
            prompt="prompt-1",
            raw_response="response-1",
            tokens={"input": 10, "output": 5},
            validation={"test_code": {"valid": True, "errors": []}},
            artifacts={"test_code": "assert True"},
        )
        logger.log_iteration(
            iteration=2,
            prompt="prompt-2",
            raw_response="response-2",
            tokens={"input": 12, "output": 6},
            validation={"source_code": {"valid": False, "errors": ["bad import"]}},
            artifacts={"source_code": "def x():\n    return 1"},
        )

    lines = (tmp_path / "ab_run" / "iterations.jsonl").read_text().splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    assert first["iteration"] == 1
    assert first["prompt"] == "prompt-1"
    assert first["tokens"] == {"input": 10, "output": 5}
    assert first["timestamp"] == fixed_time.isoformat()

    second = json.loads(lines[1])
    assert second["iteration"] == 2


def test_log_iteration_truncates_large_response(tmp_path):
    logger = RunLogger(tmp_path / "ab_run")
    large = "x" * (MAX_RESPONSE_BYTES + 256)

    logger.log_iteration(1, "p", large, {}, {}, {})

    entry = json.loads((tmp_path / "ab_run" / "iterations.jsonl").read_text())
    assert TRUNCATION_MARKER in entry["raw_response"]
    assert entry["raw_response"] != large


def test_log_result_writes_json(tmp_path):
    logger = RunLogger(tmp_path / "ab_run")
    result = {"mode": "schema", "success": True, "iterations": 2}
    logger.log_result(result)
    saved = json.loads((tmp_path / "ab_run" / "result.json").read_text())
    assert saved == result


def test_log_diff_writes_patch(tmp_path):
    logger = RunLogger(tmp_path / "ab_run")
    diff = "diff --git a/a.py b/a.py\n+print('hello')\n"
    logger.log_diff(diff)
    assert (tmp_path / "ab_run" / "diff.patch").read_text() == diff


def test_log_artifacts_writes_files(tmp_path):
    logger = RunLogger(tmp_path / "ab_run")
    artifacts = {"test_code": "assert True\n", "source_code": "def run():\n    return 1\n"}
    logger.log_artifacts(artifacts)
    arts_dir = tmp_path / "ab_run" / "artifacts"
    assert (arts_dir / "test_code.py").read_text() == artifacts["test_code"]
    assert (arts_dir / "source_code.py").read_text() == artifacts["source_code"]
