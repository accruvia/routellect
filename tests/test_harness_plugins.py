from __future__ import annotations

import json
import subprocess
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


def _init_repo(repo_root: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_root, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo_root, check=True, capture_output=True, text=True)


def test_project_adapter_creates_disposable_git_worktree(tmp_path: Path, monkeypatch) -> None:
    harness_src = Path(__file__).resolve().parents[2] / "accruvia-harness" / "src"
    sys.path.insert(0, str(harness_src))
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "README.md").write_text("# Demo\n", encoding="utf-8")
    (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    _init_repo(repo_root)
    monkeypatch.setenv("ROUTELLECT_REPO_ROOT", str(repo_root))

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    workspace = RoutellectProjectAdapter().prepare_workspace(_Project(), _Task(), _Run(), run_dir)

    worktree_root = run_dir / "workspace"
    manifest = json.loads((run_dir / "routellect_workspace_manifest.json").read_text(encoding="utf-8"))

    assert worktree_root.exists()
    assert not worktree_root.is_symlink()
    assert workspace.project_root == worktree_root
    assert manifest["workspace_mode"] == "disposable_git_worktree"
    assert manifest["routellect_repo_root"] == str(repo_root)
    assert manifest["routellect_worktree_root"] == str(worktree_root)
    assert workspace.environment["ROUTELLECT_REPO_ROOT"] == str(worktree_root)
    assert workspace.environment["ROUTELLECT_SOURCE_REPO_ROOT"] == str(repo_root)

    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=worktree_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert Path(result.stdout.strip()) == worktree_root


def test_project_adapter_changes_in_worktree_do_not_dirty_source_repo(tmp_path: Path, monkeypatch) -> None:
    harness_src = Path(__file__).resolve().parents[2] / "accruvia-harness" / "src"
    sys.path.insert(0, str(harness_src))
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "README.md").write_text("# Demo\n", encoding="utf-8")
    (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (repo_root / "src").mkdir()
    (repo_root / "src" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _init_repo(repo_root)
    monkeypatch.setenv("ROUTELLECT_REPO_ROOT", str(repo_root))

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    workspace = RoutellectProjectAdapter().prepare_workspace(_Project(), _Task(), _Run(), run_dir)

    (workspace.project_root / "src" / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    worktree_status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=workspace.project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    source_status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "src/module.py" in worktree_status.stdout
    assert source_status.stdout.strip() == ""


def test_project_adapter_worker_preserves_command_passthrough(tmp_path: Path, monkeypatch) -> None:
    harness_src = Path(__file__).resolve().parents[2] / "accruvia-harness" / "src"
    sys.path.insert(0, str(harness_src))
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "README.md").write_text("# Demo\n", encoding="utf-8")
    (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    _init_repo(repo_root)
    monkeypatch.setenv("ROUTELLECT_REPO_ROOT", str(repo_root))

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    adapter = RoutellectProjectAdapter()
    workspace = adapter.prepare_workspace(_Project(), _Task(), _Run(), run_dir)
    worker = adapter.build_worker(_Project(), _Task(), _Run(), workspace, default_worker=None)

    assert "ROUTELLECT_HARNESS_WORKER_COMMAND" in worker.env_passthrough
    assert "harness_worker.py" in worker.command
