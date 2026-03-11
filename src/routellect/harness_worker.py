from __future__ import annotations

import json
import os
import signal
import shlex
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_EXECUTOR_TIMEOUT_SECONDS = 120
DEFAULT_EXECUTOR_IDLE_TIMEOUT_SECONDS = 30
DEFAULT_EXECUTOR_MAX_EXTENSION_SECONDS = 120
DEFAULT_EXECUTOR_POLL_SECONDS = 1.0
DEFAULT_VALIDATION_TIMEOUT_SECONDS = 60
_VALIDATION_ENV_BLOCKLIST = {
    "ACCRUVIA_TASK_SCOPE_JSON",
    "ROUTELLECT_HARNESS_WORKER_COMMAND",
    "ROUTELLECT_HARNESS_EXECUTOR_TIMEOUT_SECONDS",
    "ROUTELLECT_HARNESS_EXECUTOR_IDLE_TIMEOUT_SECONDS",
    "ROUTELLECT_HARNESS_EXECUTOR_MAX_EXTENSION_SECONDS",
}


def _python_command(project_root: Path) -> list[str]:
    repo_python = project_root / ".venv" / "bin" / "python"
    if repo_python.exists():
        return [str(repo_python)]
    return [sys.executable]


def _executor_timeouts() -> tuple[float, float, float]:
    base = float(os.environ.get("ROUTELLECT_HARNESS_EXECUTOR_TIMEOUT_SECONDS", DEFAULT_EXECUTOR_TIMEOUT_SECONDS))
    idle = float(os.environ.get("ROUTELLECT_HARNESS_EXECUTOR_IDLE_TIMEOUT_SECONDS", DEFAULT_EXECUTOR_IDLE_TIMEOUT_SECONDS))
    extension = float(
        os.environ.get("ROUTELLECT_HARNESS_EXECUTOR_MAX_EXTENSION_SECONDS", DEFAULT_EXECUTOR_MAX_EXTENSION_SECONDS)
    )
    return base, idle, extension


def _run_subprocess(args: list[str], cwd: Path, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for key in _VALIDATION_ENV_BLOCKLIST:
        env.pop(key, None)
    return subprocess.run(
        args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        env=env,
    )


def _git_changed_files_for_progress(project_root: Path) -> list[str]:
    return _git_changed_files(project_root)


def _run_command_with_progress(command: str, cwd: Path, run_dir: Path) -> tuple[subprocess.CompletedProcess[str], bool, dict[str, object]]:
    stdout_path = run_dir / "worker_response.txt"
    stderr_path = run_dir / "worker.executor.stderr.txt"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")

    base_timeout, idle_timeout, max_extension = _executor_timeouts()
    hard_deadline = time.monotonic() + base_timeout + max_extension
    soft_deadline = time.monotonic() + base_timeout
    last_progress_at = time.monotonic()
    last_stdout_size = 0
    last_stderr_size = 0
    last_changed_files: tuple[str, ...] = tuple(_git_changed_files_for_progress(cwd))
    extension_used = 0.0
    progress_observed = False

    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=str(cwd),
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            preexec_fn=os.setsid,
        )

        timed_out = False
        while True:
            returncode = process.poll()
            now = time.monotonic()
            stdout_size = stdout_path.stat().st_size if stdout_path.exists() else 0
            stderr_size = stderr_path.stat().st_size if stderr_path.exists() else 0
            changed_files = tuple(_git_changed_files_for_progress(cwd))
            if stdout_size != last_stdout_size or stderr_size != last_stderr_size or changed_files != last_changed_files:
                progress_observed = True
                last_progress_at = now
                last_stdout_size = stdout_size
                last_stderr_size = stderr_size
                last_changed_files = changed_files

            if returncode is not None:
                break

            if now >= hard_deadline:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break

            if now >= soft_deadline:
                can_extend = extension_used < max_extension
                recently_active = (now - last_progress_at) <= idle_timeout
                if can_extend and recently_active:
                    remaining = max_extension - extension_used
                    extension_step = min(idle_timeout, remaining)
                    extension_used += extension_step
                    soft_deadline = now + extension_step
                else:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    break

            time.sleep(DEFAULT_EXECUTOR_POLL_SECONDS)

    stdout = stdout_path.read_text(encoding="utf-8")
    stderr = stderr_path.read_text(encoding="utf-8")
    completed = subprocess.CompletedProcess(
        args=command,
        returncode=process.returncode if not timed_out else 124,
        stdout=stdout,
        stderr=stderr if not timed_out else (stderr or "worker executor timed out"),
    )
    return completed, timed_out, {
        "base_timeout_seconds": base_timeout,
        "idle_timeout_seconds": idle_timeout,
        "max_extension_seconds": max_extension,
        "extension_used_seconds": extension_used,
        "progress_observed": progress_observed,
    }


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


def _load_scope() -> dict[str, object]:
    raw = os.environ.get("ACCRUVIA_TASK_SCOPE_JSON", "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _scope_violation(changed_files: list[str], scope: dict[str, object]) -> dict[str, object] | None:
    allowed = [str(item) for item in scope.get("allowed_paths", []) if item]
    forbidden = [str(item) for item in scope.get("forbidden_paths", []) if item]
    if not allowed and not forbidden:
        return None

    outside_allowed: list[str] = []
    forbidden_hits: list[str] = []
    if allowed:
        for path in changed_files:
            if not any(path == item or path.startswith(f"{item.rstrip('/')}/") for item in allowed):
                outside_allowed.append(path)
    if forbidden:
        for path in changed_files:
            if any(path == item or path.startswith(f"{item.rstrip('/')}/") for item in forbidden):
                forbidden_hits.append(path)
    if outside_allowed or forbidden_hits:
        return {
            "allowed_paths": allowed,
            "forbidden_paths": forbidden,
            "outside_allowed_paths": outside_allowed,
            "forbidden_path_hits": forbidden_hits,
        }
    return None


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
    initial_changed_files = _git_changed_files(project_root)
    command = _executor_command(prompt)
    execution, timed_out, timeout_details = _run_command_with_progress(command, project_root, run_dir)
    response_path = run_dir / "worker_response.txt"
    compile_check = _run_compile_check(project_root)
    test_check = _run_test_check(project_root)
    changed_files = _git_changed_files(project_root)
    new_changed_files = [path for path in changed_files if path not in initial_changed_files]
    scope = _load_scope()
    scope_violation = _scope_violation(new_changed_files, scope)

    if timed_out:
        outcome = "failed"
        summary = "Worker executor timed out."
    elif execution.returncode != 0:
        outcome = "failed"
        summary = "Worker executor failed."
    elif scope_violation is not None:
        outcome = "blocked"
        summary = "Worker changed files outside the task scope."
    elif not new_changed_files:
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
        "executor_timeout_details": timeout_details,
        "task_scope": scope,
        "scope_violation": scope_violation,
        "initial_changed_files": initial_changed_files,
        "changed_files": changed_files,
        "new_changed_files": new_changed_files,
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
