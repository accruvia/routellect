from __future__ import annotations

import json
import os
import signal
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
    monkeypatch.setattr(harness_worker, "DEFAULT_EXECUTOR_IDLE_TIMEOUT_SECONDS", 0.25)
    monkeypatch.setattr(harness_worker, "DEFAULT_EXECUTOR_MAX_EXTENSION_SECONDS", 0.0)
    monkeypatch.setattr(harness_worker, "DEFAULT_EXECUTOR_POLL_SECONDS", 0.05)
    monkeypatch.setenv(
        "ROUTELLECT_HARNESS_WORKER_COMMAND",
        "python3 - <<'PY'\nimport time\ntime.sleep(2)\nPY",
    )

    report = run_worker(project_root=project_root, run_dir=run_dir, objective="Timeout")

    assert report["worker_outcome"] == "failed"
    assert report["executor_timed_out"] is True


def test_harness_worker_extends_timeout_when_progress_is_observed(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    _init_repo(project_root)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    monkeypatch.setattr(harness_worker, "DEFAULT_EXECUTOR_TIMEOUT_SECONDS", 0.3)
    monkeypatch.setattr(harness_worker, "DEFAULT_EXECUTOR_IDLE_TIMEOUT_SECONDS", 0.5)
    monkeypatch.setattr(harness_worker, "DEFAULT_EXECUTOR_MAX_EXTENSION_SECONDS", 1.0)
    monkeypatch.setattr(harness_worker, "DEFAULT_EXECUTOR_POLL_SECONDS", 0.05)
    monkeypatch.setenv(
        "ROUTELLECT_HARNESS_WORKER_COMMAND",
        "python3 - <<'PY'\n"
        "import sys, time\n"
        "from pathlib import Path\n"
        "print('working', flush=True)\n"
        "time.sleep(0.4)\n"
        "path = Path('src/demo.py')\n"
        "path.write_text(path.read_text(encoding='utf-8') + '\\n# progress\\n', encoding='utf-8')\n"
        "print('done', flush=True)\n"
        "PY",
    )

    report = run_worker(project_root=project_root, run_dir=run_dir, objective="Use progress-aware extension")

    assert report["worker_outcome"] == "success"
    assert report["executor_timed_out"] is False
    assert report["executor_timeout_details"]["progress_observed"] is True
    assert report["executor_timeout_details"]["extension_used_seconds"] > 0


def test_harness_worker_blocks_when_changes_escape_scope(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    _init_repo(project_root)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    monkeypatch.setenv("ROUTELLECT_HARNESS_WORKER_COMMAND", "printf '\\n# oops\\n' >> README.md")
    monkeypatch.setenv(
        "ACCRUVIA_TASK_SCOPE_JSON",
        json.dumps({"allowed_paths": ["src", "tests"], "forbidden_paths": ["README.md"]}),
    )

    report = run_worker(project_root=project_root, run_dir=run_dir, objective="Stay in scope")

    assert report["worker_outcome"] == "blocked"
    assert report["scope_violation"]["forbidden_path_hits"] == ["README.md"]


def test_harness_worker_ignores_preexisting_dirty_files_for_scope(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    _init_repo(project_root)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (project_root / "README.md").write_text("dirty\n", encoding="utf-8")

    monkeypatch.setenv("ROUTELLECT_HARNESS_WORKER_COMMAND", "printf '\\n# scoped change\\n' >> src/demo.py")
    monkeypatch.setenv(
        "ACCRUVIA_TASK_SCOPE_JSON",
        json.dumps({"allowed_paths": ["src/demo.py"], "forbidden_paths": ["README.md"]}),
    )

    report = run_worker(project_root=project_root, run_dir=run_dir, objective="Stay in scope with preexisting dirt")

    assert report["worker_outcome"] == "success"
    assert "README.md" in report["changed_files"]
    assert report["new_changed_files"] == ["src/demo.py"]
    assert report["scope_violation"] is None


def test_harness_worker_kills_process_group_on_timeout(monkeypatch, tmp_path: Path) -> None:
    class FakeProc:
        def __init__(self) -> None:
            self.pid = 4242
            self.returncode = None
            self._wait_calls = 0

        def poll(self):
            return None

        def wait(self, timeout=None):
            self._wait_calls += 1
            if timeout is not None and self._wait_calls == 1:
                raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
            self.returncode = -signal.SIGKILL
            return self.returncode

    proc = FakeProc()
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(harness_worker.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(harness_worker.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(harness_worker, "_git_changed_files_for_progress", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(harness_worker, "DEFAULT_EXECUTOR_TIMEOUT_SECONDS", 0.0)
    monkeypatch.setattr(harness_worker, "DEFAULT_EXECUTOR_IDLE_TIMEOUT_SECONDS", 0.0)
    monkeypatch.setattr(harness_worker, "DEFAULT_EXECUTOR_MAX_EXTENSION_SECONDS", 0.0)
    monkeypatch.setattr(harness_worker, "DEFAULT_EXECUTOR_POLL_SECONDS", 0.0)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    completed, timed_out, _ = harness_worker._run_command_with_progress("fake", tmp_path, run_dir)

    assert timed_out is True
    assert completed.returncode == 124
    assert signals == [(4242, signal.SIGTERM), (4242, signal.SIGKILL)]
