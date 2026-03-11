from __future__ import annotations

import sys
from pathlib import Path

from routellect.harness_plugins import RoutellectProjectAdapter


class _Project:
    id = "project-1"
    name = "Routellect"


class _Task:
    id = "task-1"


class _Run:
    id = "run-1"


def test_project_adapter_uses_repo_root_as_workspace_symlink(tmp_path: Path, monkeypatch) -> None:
    harness_src = Path(__file__).resolve().parents[2] / "accruvia-harness" / "src"
    sys.path.insert(0, str(harness_src))
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "README.md").write_text("# Demo\n", encoding="utf-8")
    (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    monkeypatch.setenv("ROUTELLECT_REPO_ROOT", str(repo_root))

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    workspace = RoutellectProjectAdapter().prepare_workspace(_Project(), _Task(), _Run(), run_dir)

    assert (run_dir / "workspace").is_symlink()
    assert (run_dir / "workspace").resolve() == repo_root
    assert workspace.project_root == repo_root
    assert (run_dir / "routellect_workspace_manifest.json").exists()


def test_project_adapter_worker_preserves_command_passthrough(tmp_path: Path, monkeypatch) -> None:
    harness_src = Path(__file__).resolve().parents[2] / "accruvia-harness" / "src"
    sys.path.insert(0, str(harness_src))
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "README.md").write_text("# Demo\n", encoding="utf-8")
    (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    monkeypatch.setenv("ROUTELLECT_REPO_ROOT", str(repo_root))

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    adapter = RoutellectProjectAdapter()
    workspace = adapter.prepare_workspace(_Project(), _Task(), _Run(), run_dir)
    worker = adapter.build_worker(_Project(), _Task(), _Run(), workspace, default_worker=None)

    assert "ROUTELLECT_HARNESS_WORKER_COMMAND" in worker.env_passthrough
    assert "harness_worker.py" in worker.command
