from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import routellect.harness_worker as harness_worker
from routellect.harness_worker import run_worker


def _init_repo(project_root: Path) -> None:
    subprocess.run(["git", "init"], cwd=project_root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=project_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=project_root, check=True)
    (project_root / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n*.pyc\n", encoding="utf-8")
    (project_root / "src").mkdir()
    (project_root / "tests").mkdir()
    (project_root / "src" / "demo.py").write_text("VALUE = 1\n", encoding="utf-8")
    (project_root / "tests" / "test_demo.py").write_text(
        "from src.demo import VALUE\n\n\ndef test_value():\n    assert VALUE == 1\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=project_root, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=project_root, check=True, capture_output=True)


def test_harness_worker_reports_success_for_real_repo(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    _init_repo(project_root)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    monkeypatch.setenv(
        "ROUTELLECT_HARNESS_WORKER_COMMAND",
        "printf '\\n# harmless change\\n' >> src/demo.py",
    )

    report = run_worker(project_root=project_root, run_dir=run_dir, objective="Change demo value")

    assert report["worker_outcome"] == "success"
    assert "src/demo.py" in report["changed_files"]
    assert report["compile_check"]["passed"] is True
    assert report["test_check"]["passed"] is True
    assert (run_dir / "plan.txt").exists()
    assert json.loads((run_dir / "report.json").read_text(encoding="utf-8"))["worker_outcome"] == "success"


def test_harness_worker_blocks_when_no_changes_are_made(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    _init_repo(project_root)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    monkeypatch.setenv("ROUTELLECT_HARNESS_WORKER_COMMAND", "true")

    report = run_worker(project_root=project_root, run_dir=run_dir, objective="Do nothing")

    assert report["worker_outcome"] == "blocked"
    assert report["changed_files"] == []


def test_harness_worker_times_out_executor(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    _init_repo(project_root)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    monkeypatch.setattr(harness_worker, "DEFAULT_EXECUTOR_TIMEOUT_SECONDS", 1)
    monkeypatch.setenv(
        "ROUTELLECT_HARNESS_WORKER_COMMAND",
        "python3 - <<'PY'\nimport time\ntime.sleep(2)\nPY",
    )

    report = run_worker(project_root=project_root, run_dir=run_dir, objective="Timeout")

    assert report["worker_outcome"] == "failed"
    assert report["executor_timed_out"] is True
