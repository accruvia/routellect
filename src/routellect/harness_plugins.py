from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path


def _resolve_repo_root() -> Path:
    configured = os.environ.get("ROUTELLECT_REPO_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


class RoutellectProjectAdapter:
    name = "routellect"

    def prepare_workspace(self, project, task, run, run_dir: Path):
        from accruvia_harness.project_adapters import ProjectWorkspace

        source_repo_root = _resolve_repo_root()
        workspace = run_dir / "workspace"
        if workspace.exists():
            raise RuntimeError(f"Refusing to reuse existing workspace path: {workspace}")
        branch_name = self._worktree_branch_name(project.id, task.id, run.id)
        self._create_disposable_worktree(source_repo_root, workspace, branch_name)
        manifest_path = run_dir / "routellect_workspace_manifest.json"
        docs = [path for path in (source_repo_root / "README.md", source_repo_root / "pyproject.toml") if path.exists()]
        manifest_path.write_text(
            json.dumps(
                {
                    "project_id": project.id,
                    "task_id": task.id,
                    "run_id": run.id,
                    "adapter_name": self.name,
                    "workspace_mode": "disposable_git_worktree",
                    "reason": "Blocked or failed runs must never dirty the main Routellect checkout.",
                    "routellect_repo_root": str(source_repo_root),
                    "routellect_worktree_root": str(workspace),
                    "branch_name": branch_name,
                    "brain_sources": [str(path) for path in docs],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return ProjectWorkspace(
            project_root=workspace,
            workspace_mode="git_worktree",
            source_repo_root=source_repo_root,
            branch_name=branch_name,
            metadata_files=[manifest_path, *docs],
            environment={
                "ACCRUVIA_PROJECT_WORKSPACE": str(workspace),
                "ACCRUVIA_PROJECT_MANIFEST_PATH": str(manifest_path),
                "ROUTELLECT_REPO_ROOT": str(workspace),
                "ROUTELLECT_SOURCE_REPO_ROOT": str(source_repo_root),
                "ROUTELLECT_WORKTREE_BRANCH": branch_name,
            },
            diagnostics={
                "project_adapter": self.name,
                "workspace_mode": "disposable_git_worktree",
                "routellect_repo_root": str(source_repo_root),
                "routellect_worktree_root": str(workspace),
                "branch_name": branch_name,
            },
        )

    @staticmethod
    def _create_disposable_worktree(source_repo_root: Path, workspace: Path, branch_name: str) -> None:
        workspace.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["git", "worktree", "add", "-b", branch_name, str(workspace), "HEAD"],
            cwd=str(source_repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Failed to create disposable Routellect worktree: "
                f"{result.stderr.strip() or result.stdout.strip() or 'unknown git worktree error'}"
            )

    @staticmethod
    def _worktree_branch_name(project_id: str, task_id: str, run_id: str) -> str:
        raw = f"harness/{project_id}/{task_id}/{run_id}"
        return re.sub(r"[^A-Za-z0-9._/-]+", "-", raw)

    def build_worker(self, project, task, run, workspace, default_worker):
        from accruvia_harness.workers import ShellCommandWorker

        default_entrypoint = f"{shlex.quote(sys.executable)} {shlex.quote(str(Path(__file__).with_name('harness_worker.py')))}"
        command = os.environ.get(
            "ROUTELLECT_HARNESS_WORKER_ENTRYPOINT",
            default_entrypoint,
        )
        return ShellCommandWorker(
            command,
            env_passthrough=("ROUTELLECT_HARNESS_WORKER_COMMAND",),
        )


class RoutellectCognitionAdapter:
    name = "routellect"

    def resolve_project_root(self, project) -> Path:
        return _resolve_repo_root()

    def list_brain_paths(self, project, project_root: Path) -> list[Path]:
        candidates = [
            project_root / "README.md",
            project_root / "pyproject.toml",
        ]
        for folder_name in ("docs", "specs"):
            folder = project_root / folder_name
            if folder.exists():
                candidates.extend(sorted(path for path in folder.rglob("*.md") if path.is_file()))
        return [path for path in candidates if path.exists()][:16]

    def build_context(self, project, project_root: Path, project_summary, context_packet, source_documents):
        return {
            "project": {
                "id": project.id,
                "name": project.name,
                "description": project.description,
                "adapter_name": project.adapter_name,
                "project_root": str(project_root),
            },
            "objective": {
                "repo_name": "Routellect",
                "product_shape": "Open source LLM router and issue-runner module",
                "hosted_service_note": "Any server in the repo is scaffolding for velocity, not the durable product center.",
                "data_product_note": "Hashed routing requests and execution telemetry are strategically important downstream data assets.",
            },
            "project_summary": project_summary,
            "context_packet": context_packet,
            "brain_sources": [asdict(source) for source in source_documents],
        }

    def build_prompt(self, project, context: dict) -> str:
        return "\n\n".join(
            [
                "You are the project brain for Routellect.",
                "Analyze the objectives, repo documents, open task state, and recent work.",
                "Issues must be atomic. If a task is too broad, split it into smaller executable child tasks instead of retrying the broad task.",
                "Decide whether new issues/tasks should exist and what the highest-priority next work is.",
                "Return strict JSON with keys:",
                "summary, priority_focus, issue_creation_needed, proposed_tasks, risks.",
                "Each proposed_tasks item must contain title, objective, priority, rationale.",
                "If splitting an existing task, include split_of_task_id plus allowed_paths/forbidden_paths whenever you can bound the work safely.",
                json.dumps(context, indent=2, sort_keys=True),
            ]
        )

    def parse_response(self, response_text: str) -> dict:
        stripped = response_text.strip()
        for candidate in [stripped, *re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)]:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return {
                    "summary": str(payload.get("summary") or ""),
                    "priority_focus": str(payload.get("priority_focus") or ""),
                    "issue_creation_needed": bool(payload.get("issue_creation_needed", False)),
                    "proposed_tasks": list(payload.get("proposed_tasks") or []),
                    "risks": list(payload.get("risks") or []),
                    "raw_response": response_text,
                }
        return {
            "summary": stripped.splitlines()[0].strip() if stripped else "No heartbeat response returned.",
            "priority_focus": "",
            "issue_creation_needed": "create" in stripped.lower() and "issue" in stripped.lower(),
            "proposed_tasks": [],
            "risks": [],
            "raw_response": response_text,
        }


def register_project_adapters(registry) -> None:
    registry.register(RoutellectProjectAdapter())


def register_cognition_adapters(registry) -> None:
    registry.register(RoutellectCognitionAdapter())
