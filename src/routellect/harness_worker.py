from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

DEFAULT_EXECUTOR_TIMEOUT_SECONDS = 120
DEFAULT_VALIDATION_TIMEOUT_SECONDS = 60


def _python_command(project_root: Path) -> list[str]:
    repo_python = project_root / ".venv" / "bin" / "python"
    if repo_python.exists():
        return [str(repo_python)]
    return [sys.executable]


def _run_command(command: str, cwd: Path, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        shell=True,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def _run_subprocess(args: list[str], cwd: Path, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def _git_changed_files(project_root: Path) -> list[str]:
    result = _run_subprocess(["git", "status", "--porcelain"], project_root, timeout=10)
    changed: list[str] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        path = line[3:]
        if path.startswith("__pycache__/") or path.startswith(".pytest_cache/") or path.endswith(".pyc"):
            continue
        changed.append(path)
    return changed


def _executor_command(prompt: str) -> str:
    override = os.environ.get("ROUTELLECT_HARNESS_WORKER_COMMAND")
    if override:
        return override
    quoted_prompt = shlex.quote(prompt)
    return f"codex exec --dangerously-bypass-approvals-and-sandbox {quoted_prompt}"


def _write_plan(run_dir: Path, objective: str, project_root: Path) -> Path:
    plan_path = run_dir / "plan.txt"
    plan_path.write_text(
        f"objective={objective}\nproject_root={project_root}\n",
        encoding="utf-8",
    )
    return plan_path


def _run_compile_check(project_root: Path) -> dict[str, object]:
    src_root = project_root / "src"
    if not src_root.exists():
        return {"passed": True, "framework": "none", "returncode": 0, "stdout": "", "stderr": ""}
    try:
        result = _run_subprocess(
            [*_python_command(project_root), "-m", "compileall", "-q", "src"],
            project_root,
            timeout=DEFAULT_VALIDATION_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "framework": "compileall",
            "returncode": None,
            "stdout": "",
            "stderr": "compile check timed out",
            "timed_out": True,
        }
    return {
        "passed": result.returncode == 0,
        "framework": "compileall",
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _run_test_check(project_root: Path) -> dict[str, object]:
    tests_root = project_root / "tests"
    if not tests_root.exists():
        return {"passed": True, "framework": "none", "returncode": 0, "stdout": "", "stderr": ""}
    try:
        result = _run_subprocess(
            [*_python_command(project_root), "-m", "pytest", "tests", "-q"],
            project_root,
            timeout=DEFAULT_VALIDATION_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "framework": "pytest",
            "returncode": None,
            "stdout": "",
            "stderr": "test check timed out",
            "timed_out": True,
        }
    return {
        "passed": result.returncode == 0,
        "framework": "pytest",
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def run_worker(project_root: Path, run_dir: Path, objective: str) -> dict[str, object]:
    prompt = (
        "Work only inside this Routellect repository.\n"
        "Implement the task objective directly in the codebase, then leave the repo in a testable state.\n"
        f"Objective: {objective}\n"
    )
    _write_plan(run_dir, objective, project_root)
    prompt_path = run_dir / "worker_prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    response_path = run_dir / "worker_response.txt"
    command = _executor_command(prompt)
    try:
        execution = _run_command(command, project_root, timeout=DEFAULT_EXECUTOR_TIMEOUT_SECONDS)
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        execution = subprocess.CompletedProcess(
            args=command,
            returncode=124,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "worker executor timed out",
        )
        timed_out = True
    response_path.write_text(execution.stdout, encoding="utf-8")
    compile_check = _run_compile_check(project_root)
    test_check = _run_test_check(project_root)
    changed_files = _git_changed_files(project_root)

    if timed_out:
        outcome = "failed"
        summary = "Worker executor timed out."
    elif execution.returncode != 0:
        outcome = "failed"
        summary = "Worker executor failed."
    elif not changed_files:
        outcome = "blocked"
        summary = "Worker made no repository changes."
    elif not compile_check["passed"] or not test_check["passed"]:
        outcome = "failed"
        summary = "Worker changed the repo but validation failed."
    else:
        outcome = "success"
        summary = "Worker changed the repo and validation passed."

    report = {
        "task_objective": objective,
        "worker_backend": "routellect_project",
        "worker_outcome": outcome,
        "summary": summary,
        "executor_command": command,
        "executor_returncode": execution.returncode,
        "executor_timed_out": timed_out,
        "changed_files": changed_files,
        "compile_check": compile_check,
        "test_check": test_check,
        "response_path": str(response_path),
    }
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    run_dir = Path(os.environ["ACCRUVIA_RUN_DIR"])
    objective = os.environ["ACCRUVIA_TASK_OBJECTIVE"]
    project_root = Path(os.environ["ACCRUVIA_PROJECT_WORKSPACE"])
    report = run_worker(project_root=project_root, run_dir=run_dir, objective=objective)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
