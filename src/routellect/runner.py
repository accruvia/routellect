"""A/B test runner for CLI issue resolution.

Modes:
  --mode baseline    : Agent writes files with standard prompt (issue + context + rules)
  --mode constrained : Agent writes files with constraint-heavy prompt (signatures, checklist, import whitelist)
  --mode direct      : Legacy alias for baseline

Both modes use the same execution, validation, blame analysis, and retry logic.
The only variable is prompt construction strategy.

Usage:
  PYTHONPATH=src python scripts/ab_runner.py --mode baseline
  PYTHONPATH=src python scripts/ab_runner.py --mode constrained
"""

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

try:
    import dotenv

    _repo_root = Path(__file__).resolve().parents[2]
    dotenv.load_dotenv(_repo_root / ".env", override=True)
except ImportError:
    pass

from routellect.validators.import_validator import ImportValidator
from routellect.validators.name_checker import NameChecker
from routellect.qa_panel import QAPanel, QAVerdict, QA_OUTPUT_SCHEMA

# Safety cap to avoid unbounded retry loops on hard failures.
MAX_ITERATIONS = 10

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class IssueSpec:
    issue_id: str
    title: str
    description: str
    labels: list[str] | None = None


@dataclass
class TokenUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def net_input(self) -> int:
        return self.input_tokens - self.cached_input_tokens

    def to_dict(self) -> dict:
        return {
            "input": self.input_tokens,
            "cached": self.cached_input_tokens,
            "output": self.output_tokens,
            "total": self.total,
            "net": self.net_input,
        }


@dataclass
class ExecutionStatus:
    ok: bool
    output: str
    tokens: TokenUsage
    session_id: str | None = None
    failure_kind: str | None = None
    blocked_reason: str | None = None
    auth_diagnosis: dict | None = None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "output": self.output,
            "tokens": self.tokens.to_dict(),
            "session_id": self.session_id,
            "failure_kind": self.failure_kind,
            "blocked_reason": self.blocked_reason,
            "auth_diagnosis": self.auth_diagnosis,
        }


# ---------------------------------------------------------------------------
# Artifact bank
# ---------------------------------------------------------------------------


class ArtifactBank:
    """Tracks accepted and rejected artifacts across iterations."""

    def __init__(self):
        self._accepted: dict[str, str] = {}
        self._rejections: dict[str, list[str]] = {}

    def accept(self, name: str, content: str):
        self._accepted[name] = content
        self._rejections.pop(name, None)

    def reject(self, name: str, errors: list[str]):
        self._accepted.pop(name, None)
        self._rejections[name] = errors

    def eject(self, name: str, reason: str):
        self._accepted.pop(name, None)
        self._rejections[name] = [reason]

    def is_accepted(self, name: str) -> bool:
        return name in self._accepted

    def get(self, name: str) -> str | None:
        return self._accepted.get(name)

    def all_accepted(self, required: list[str]) -> bool:
        return all(name in self._accepted for name in required)

    def pending(self, required: list[str]) -> list[str]:
        return [name for name in required if name not in self._accepted]

    def get_rejection_feedback(self, name: str) -> str:
        errors = self._rejections.get(name, [])
        return "\n".join(errors)

    def get_all_accepted(self) -> dict[str, str]:
        return dict(self._accepted)

    def get_all_feedback(self) -> str:
        parts = []
        for name, content in self._accepted.items():
            parts.append(f"ACCEPTED [{name}] (locked — do not regenerate):\n{content[:2000]}")
        for name, errors in self._rejections.items():
            parts.append(f"REJECTED [{name}] — fix these errors:\n" + "\n".join(errors))
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_artifact(
    name: str,
    content: str,
    src_root: Path,
    import_allowlist: set[str] | None = None,
) -> tuple[bool, list[str]]:
    """Validate a single artifact. Returns (passed, error_messages).

    import_allowlist: module paths to skip import validation for (e.g. when
    source_code creates a new module that test_code imports).
    """
    if name == "code_schema":
        return _validate_code_schema(content, src_root)
    elif name == "test_code":
        return _validate_code(
            content,
            src_root,
            require_asserts=True,
            import_allowlist=import_allowlist,
        )
    elif name == "source_code":
        return _validate_code(content, src_root, require_logic=True)
    return False, [f"Unknown artifact type: {name}"]


def _validate_code(
    code: str,
    src_root: Path,
    require_asserts: bool = False,
    require_logic: bool = False,
    import_allowlist: set[str] | None = None,
) -> tuple[bool, list[str]]:
    errors = []

    # ast.parse
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"SyntaxError: {e}"]

    # ImportValidator
    validator = ImportValidator(src_root=src_root)
    valid, invalid = validator.validate(code, allowlist=import_allowlist)
    if not valid:
        errors.append(f"Hallucinated imports: {', '.join(invalid)}")

    # NameChecker
    checker = NameChecker()
    undefined = checker.check(code)
    if undefined:
        errors.append(f"Undefined names: {', '.join(str(u) for u in undefined)}")

    # Require assert statements in test code
    if require_asserts:
        has_assert = any(isinstance(node, ast.Assert) for node in ast.walk(tree))
        if not has_assert:
            errors.append("Test code has no assert statements")

    # Require actual logic in source code (not just pass/placeholder)
    if require_logic:
        _check_has_logic(tree, errors)

    return len(errors) == 0, errors


def _check_has_logic(tree: ast.AST, errors: list[str]):
    """Check that TOP-LEVEL function bodies contain actual logic, not just pass.

    Only checks top-level functions and methods in non-stub classes.
    Skips Protocol/ABC method stubs and abstract methods (legitimate ... bodies).
    """
    # Collect class names that are Protocols or ABCs (stub bodies are expected)
    stub_classes: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_name = ""
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr
                if base_name in ("Protocol", "ABC", "ABCMeta"):
                    stub_classes.add(node.name)

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _check_function_body(node, errors)
        elif isinstance(node, ast.ClassDef) and node.name not in stub_classes:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Skip abstractmethods
                    is_abstract = any(
                        (isinstance(d, ast.Name) and d.id == "abstractmethod")
                        or (isinstance(d, ast.Attribute) and d.attr == "abstractmethod")
                        for d in item.decorator_list
                    )
                    if not is_abstract:
                        _check_function_body(item, errors)


def _check_function_body(node: ast.FunctionDef | ast.AsyncFunctionDef, errors: list[str]):
    """Check a single function has real logic."""
    body = node.body
    non_doc = [n for n in body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    if len(non_doc) == 0:
        errors.append(f"Function '{node.name}' is placeholder (empty body)")
        return
    if len(non_doc) == 1 and isinstance(non_doc[0], ast.Pass):
        errors.append(f"Function '{node.name}' is placeholder (just pass)")
        return


def _validate_code_schema(content: str, src_root: Path) -> tuple[bool, list[str]]:
    errors = []
    try:
        schema = json.loads(content) if isinstance(content, str) else content
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]

    target = schema.get("target_file", "")
    is_new = schema.get("is_new_file", False)
    functions = schema.get("functions", [])
    classes = schema.get("classes", [])

    if not functions and not classes:
        errors.append("code_schema must specify at least one function or class")

    if not target:
        errors.append("code_schema must specify target_file")

    if not is_new and target:
        # Check file exists
        full_path = Path(target)
        if not full_path.exists():
            # Try relative to src_root parent (project root)
            project_root = src_root.parent if src_root.name == "src" else src_root
            full_path = project_root / target
            if not full_path.exists():
                errors.append(f"target_file does not exist (is_new_file=false): {target}")

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Blame analysis
# ---------------------------------------------------------------------------


def blame_pytest(output: str) -> str:
    """Classify pytest failure → blamed artifact name."""
    # SyntaxError — check which file
    if "SyntaxError" in output:
        if "tests/generated/" in output or "test_" in output:
            return "test_code"
        return "source_code"

    # NameError in test file → test_code
    if "NameError" in output:
        return "test_code"

    # Everything else → source_code
    # ImportError, AssertionError, TypeError, AttributeError, etc.
    return "source_code"


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------


def aggregate_tokens(iterations: list[dict]) -> TokenUsage:
    total = TokenUsage()
    for it in iterations:
        t = it["tokens"]
        total.input_tokens += t.input_tokens
        total.cached_input_tokens += t.cached_input_tokens
        total.output_tokens += t.output_tokens
    return total


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _classify_execution_failure(message: str, *, returncode: int | None = None) -> tuple[str, str, dict]:
    lower = (message or "").lower()
    reason = "openclaw_execution_failed"
    failure_kind = "failure_launch"

    if "401" in lower or "unauthorized" in lower or "missing bearer" in lower or "authentication" in lower:
        reason = "openclaw_execution_auth_failed"
        failure_kind = "failure_env"
    elif "quota exceeded" in lower or "billing" in lower or "rate limit" in lower:
        reason = "openclaw_execution_provider_blocked"
        failure_kind = "failure_env"
    elif "timed out" in lower or "timeout" in lower:
        reason = "openclaw_execution_timed_out"
        failure_kind = "failure_launch"
    elif returncode and returncode != 0:
        reason = "openclaw_execution_nonzero_exit"

    return failure_kind, reason, {
        "runner_backend": "openclaw",
        "message": message[:1000],
        "returncode": returncode,
        "observed_at": _utcnow(),
    }


def _get_git_output(cwd: Path, args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=15, check=False)
    return result.stdout


def snapshot_candidate_artifacts(
    cwd: Path,
    artifact_dir: Path,
    *,
    issue_id: str,
    run_id: str,
    mode: str,
    iteration: int,
    execution_status: ExecutionStatus | None = None,
) -> dict | None:
    changed_files = [line.strip() for line in _get_git_output(cwd, ["status", "--short", "--untracked-files=all"]).splitlines() if line.strip()]
    if not changed_files:
        return None

    artifact_dir.mkdir(parents=True, exist_ok=True)
    files_dir = artifact_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "issue_id": issue_id,
        "run_id": run_id,
        "mode": mode,
        "iteration": iteration,
        "captured_at": _utcnow(),
        "runner_session_id": execution_status.session_id if execution_status else None,
        "execution": execution_status.to_dict() if execution_status else None,
        "changed_files": [],
    }

    for rel in changed_files:
        parts = rel.split(maxsplit=1)
        path_text = parts[1].strip() if len(parts) > 1 else rel.strip()
        if not path_text:
            continue
        manifest["changed_files"].append(path_text)
        src = cwd / path_text
        if src.is_file():
            dest = files_dir / path_text
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(src.read_text())

    diff = _get_git_output(cwd, ["diff", "--binary", "HEAD"])
    tracked = _get_git_output(cwd, ["ls-files", "--others", "--exclude-standard"])
    (artifact_dir / "worktree.diff").write_text(diff)
    (artifact_dir / "untracked_files.txt").write_text(tracked)
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _throughput_summary(result: dict) -> dict:
    candidate_artifacts = result.get("candidate_artifacts") or {}
    workflow_learning = result.get("workflow_learning") or {}
    useful = bool(
        candidate_artifacts.get("present")
        or workflow_learning.get("present")
        or result.get("blocked_diagnosis_changed_next_action")
    )
    return {
        "useful": useful,
        "counts_as_useful_throughput": useful,
        "reason": (
            "candidate_artifact_preserved"
            if candidate_artifacts.get("present")
            else "workflow_learning_recorded"
            if workflow_learning.get("present")
            else "validated_blocked_state"
            if result.get("blocked_diagnosis_changed_next_action")
            else "no_preserved_artifact_or_learning"
        ),
    }


def _qa_artifact_blame(reject) -> str:
    text = f"{getattr(reject, 'reviewer', '')} {getattr(reject, 'concern', '')}".lower()
    if any(token in text for token in ("test", "pytest", "assert", "fixture")):
        return "test_code"
    return "source_code"


def apply_qa_verdict(bank: ArtifactBank, verdict: QAVerdict) -> list[str]:
    ejected = []
    for reject in verdict.hard_rejects:
        artifact = _qa_artifact_blame(reject)
        bank.eject(artifact, f"QA {reject.reviewer}: {reject.concern}")
        ejected.append(artifact)
    return ejected


def run_qa_review(issue: IssueSpec, bank: ArtifactBank, cwd: Path, qa_panel: QAPanel) -> tuple[QAVerdict | None, ExecutionStatus]:
    schema_file = cwd / ".qa_output_schema.json"
    schema_file.write_text(json.dumps(QA_OUTPUT_SCHEMA, indent=2))
    prompt = qa_panel.build_prompt(issue, bank.get_all_accepted())
    exec_status = run_codex(prompt, cwd, output_schema_file=schema_file)
    if not exec_status.ok:
        return None, exec_status
    try:
        verdict = qa_panel.parse_verdict(exec_status.output, tokens_used=exec_status.tokens.total)
    except Exception as exc:
        failed = ExecutionStatus(
            ok=False,
            output=f"qa verdict parse failed: {exc}",
            tokens=exec_status.tokens,
            session_id=exec_status.session_id,
            failure_kind="failure_parse",
            blocked_reason="qa_verdict_parse_failed",
            auth_diagnosis={"raw_output": exec_status.output[:1000]},
        )
        return None, failed
    return verdict, exec_status


# ---------------------------------------------------------------------------
# Schema mode: parse response
# ---------------------------------------------------------------------------

REQUIRED_ARTIFACTS = ["code_schema", "test_code", "source_code"]


def parse_schema_response(raw: str) -> dict | None:
    """Extract artifacts from structured JSON response.

    The agent may emit multiple JSON objects (one per internal turn).
    We take the LAST complete JSON object that has all required fields.
    """
    text = raw.strip()

    # Try direct JSON parse
    try:
        data = json.loads(text)
        if _has_required_fields(data):
            return _normalize_schema_data(data)
    except json.JSONDecodeError:
        pass

    # Agent may emit multiple JSON objects — try each line or split on }{ boundaries
    # Find all JSON objects by looking for top-level { } pairs
    candidates = []
    # Split on newlines first — each turn may produce a separate JSON line
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("{"):
            try:
                data = json.loads(line)
                if _has_required_fields(data):
                    candidates.append(data)
            except json.JSONDecodeError:
                pass

    # Also try }{ boundary split for concatenated objects
    if not candidates and "}{" in text:
        parts = text.replace("}{", "}\n{").split("\n")
        for part in parts:
            part = part.strip()
            try:
                data = json.loads(part)
                if _has_required_fields(data):
                    candidates.append(data)
            except json.JSONDecodeError:
                pass

    # Take the last complete one (most likely the final answer)
    if candidates:
        return _normalize_schema_data(candidates[-1])

    # Try extracting from markdown fence
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if fence_match:
        try:
            data = json.loads(fence_match.group(1))
            if _has_required_fields(data):
                return _normalize_schema_data(data)
        except json.JSONDecodeError:
            pass

    return None


def _has_required_fields(data: dict) -> bool:
    return all(k in data for k in REQUIRED_ARTIFACTS)


def _normalize_schema_data(data: dict) -> dict:
    """Ensure code_schema is a JSON string for consistency."""
    result = {}
    for key in REQUIRED_ARTIFACTS:
        val = data[key]
        if key == "code_schema" and isinstance(val, dict):
            result[key] = json.dumps(val)
        else:
            result[key] = str(val)
    return result


def _coerce_code_schema(content: object) -> dict | None:
    """Parse code_schema from either dict or JSON string.

    Prevents TypeError when code_schema arrives as a dict (from
    parse_schema_response) but callers assume it's a JSON string.
    """
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


# ---------------------------------------------------------------------------
# Schema mode: prompt building
# ---------------------------------------------------------------------------


def _get_target_file_content(code_schema_dict: dict | None, worktree: Path | None) -> str | None:
    """Read the target file content when modifying an existing file.

    Returns the file content string, or None if not applicable.
    """
    if not code_schema_dict or not worktree:
        return None
    if code_schema_dict.get("is_new_file", True):
        return None
    target = code_schema_dict.get("target_file", "")
    if not target:
        return None
    target_path = worktree / target
    if target_path.exists():
        try:
            return target_path.read_text()
        except Exception:
            return None
    return None


def build_schema_prompt(
    issue: IssueSpec,
    codebase_ctx: str,
    bank: ArtifactBank,
    code_schema_dict: dict | None = None,
    worktree: Path | None = None,
) -> str:
    parts = [
        f"Implement issue #{issue.issue_id}: {issue.title}",
        "",
        f"Description:\n{issue.description}",
        "",
    ]

    if codebase_ctx:
        parts.extend([codebase_ctx, ""])

    # Include target file content when modifying an existing file.
    # This prevents the LLM from generating partial code fragments
    # with missing imports/definitions.
    target_content = _get_target_file_content(code_schema_dict, worktree)
    if target_content:
        target_file = code_schema_dict.get("target_file", "")
        parts.extend(
            [
                f"CURRENT CONTENT OF {target_file} (you are modifying this file):",
                "```python",
                target_content,
                "```",
                "",
                "Your source_code MUST be the complete updated file, not a fragment.",
                "Include ALL existing imports and definitions, plus your changes.",
                "",
            ]
        )

    parts.extend(
        [
            "Return your response as a single JSON object with these fields:",
            '  "code_schema": {"target_file": "...", "is_new_file": bool, '
            '"functions": [...], "classes": [...], "rationale": "..."}',
            '  "test_code": "...full pytest file content..."',
            '  "source_code": "...full implementation code to add..."',
            "",
            "Rules:",
            "- Do NOT write files directly — return everything in the JSON response",
            "- test_code must use unittest.mock, have real assert statements",
            "- source_code must contain actual logic, not placeholders",
            "- Import from real project modules only",
            "- If modifying an existing file, return the COMPLETE file with all imports",
            "",
        ]
    )

    # Add bank state
    feedback = bank.get_all_feedback()
    if feedback:
        parts.extend(["PRIOR ATTEMPT FEEDBACK:", feedback, ""])

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Direct mode: prompt building
# ---------------------------------------------------------------------------


def build_direct_prompt(issue: IssueSpec, codebase_ctx: str, feedback: str | None = None) -> str:
    parts = [
        f"Implement issue #{issue.issue_id}: {issue.title}",
        "",
        f"Description:\n{issue.description}",
        "",
        "INSTRUCTIONS:",
        "1. Write a pytest test file in tests/generated/ that covers the acceptance criteria",
        "2. Write the implementation code in the appropriate source file",
        "3. Make sure the tests pass when run with pytest",
        "4. Only modify/create files under src/ and tests/generated/",
        "",
    ]

    if codebase_ctx:
        parts.extend([codebase_ctx, ""])

    parts.extend(
        [
            "Rules:",
            "- Import from real project modules only, not invented paths",
            "- Keep the implementation minimal",
            "- Tests should use unittest.mock to avoid real API calls",
            "",
        ]
    )

    if feedback:
        parts.extend(["PRIOR ATTEMPT FEEDBACK:", feedback, ""])

    return "\n".join(parts)


def build_constrained_prompt(issue: IssueSpec, codebase_ctx: str, feedback: str | None = None) -> str:
    """Constraint-heavy prompt: explicit signatures, import whitelist, structured checklist."""
    parts = [
        f"Implement issue #{issue.issue_id}: {issue.title}",
        "",
        f"Description:\n{issue.description}",
        "",
    ]

    if codebase_ctx:
        parts.extend([codebase_ctx, ""])

    parts.extend(
        [
            "STRICT CONSTRAINTS (violating any of these will cause rejection):",
            "",
            "1. FILE STRUCTURE:",
            f"   - Test file: tests/generated/test_issue_{issue.issue_id}.py",
            "   - Source file: modify existing file mentioned in the description, or create under src/accruvia/",
            "   - Do NOT create files outside src/ and tests/generated/",
            "",
            "2. IMPORT RULES:",
            "   - ONLY import from modules listed in AVAILABLE IMPORTS above",
            "   - ONLY import from stdlib and installed packages (pytest, unittest.mock)",
            "   - NEVER invent module paths — if it's not in AVAILABLE IMPORTS, don't import it",
            "   - Every name you use must be defined, imported, or from builtins",
            "",
            "3. TEST REQUIREMENTS:",
            "   - Every test function must start with 'test_'",
            "   - Every test must have at least one assert statement",
            "   - Use unittest.mock.patch/MagicMock for external dependencies",
            "   - Test the happy path AND at least one edge case",
            "",
            "4. IMPLEMENTATION REQUIREMENTS:",
            "   - Must contain actual logic (not pass/placeholder)",
            "   - Must be valid Python (parseable by ast.parse)",
            "   - Function/class names must match what the tests import",
            "",
            "5. BEFORE WRITING CODE, plan your approach:",
            "   - What function(s) will you create/modify?",
            "   - What are their signatures (args, return type)?",
            "   - What edge cases exist?",
            "   - What imports do you need?",
            "",
        ]
    )

    if feedback:
        parts.extend(["PRIOR ATTEMPT FEEDBACK:", feedback, ""])

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Direct mode: discover artifacts from disk
# ---------------------------------------------------------------------------


def get_changed_files(cwd: Path) -> list[str]:
    """List files changed or added in worktree.

    Uses --no-exclude-standard for untracked files because tests/generated/
    is in .gitignore but we need to discover test files written there.
    """
    changed = (
        subprocess.run(
            ["git", "diff", "--name-only"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=10,
        )
        .stdout.strip()
        .split("\n")
    )
    # Don't use --exclude-standard — tests/generated/ is gitignored but we need it
    untracked = (
        subprocess.run(
            ["git", "ls-files", "--others", "--exclude", "__pycache__", "--exclude", "*.pyc"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=10,
        )
        .stdout.strip()
        .split("\n")
    )
    return [f for f in changed + untracked if f.strip()]


def discover_artifacts(cwd: Path) -> dict[str, str]:
    """Scan worktree for changed files and map to artifact types."""
    files = get_changed_files(cwd)
    artifacts = {}

    for f in files:
        fp = cwd / f
        if not fp.exists() or not fp.is_file():
            continue

        content = fp.read_text()
        if "tests/generated/" in f and f.endswith(".py"):
            artifacts["test_code"] = content
            artifacts["_test_file"] = f
        elif f.startswith("src/") and f.endswith(".py"):
            artifacts["source_code"] = content
            artifacts["_source_file"] = f

    return artifacts


# ---------------------------------------------------------------------------
# Worktree + context
# ---------------------------------------------------------------------------


def setup_worktree(issue_id: str, base_dir: Path, run_id: str) -> Path:
    wt_dir = base_dir / ".worktrees" / f"ab-{issue_id}-{run_id}"
    branch = f"ab/{issue_id}-{run_id}"

    if wt_dir.exists():
        subprocess.run(["git", "checkout", "--", "."], cwd=str(wt_dir), capture_output=True, timeout=10)
        subprocess.run(["git", "clean", "-fd"], cwd=str(wt_dir), capture_output=True, timeout=10)
        return wt_dir

    subprocess.run(["git", "branch", "-D", branch], cwd=str(base_dir), capture_output=True)
    subprocess.run(
        ["git", "worktree", "add", str(wt_dir), "-b", branch, "main"],
        cwd=str(base_dir),
        capture_output=True,
        check=True,
        timeout=30,
    )
    return wt_dir


def get_codebase_context(src_root: Path, title: str, description: str) -> str:
    try:
        from accruvia.orchestration.codebase_index import CodebaseIndex

        index = CodebaseIndex(src_root)
        index.index_all()
        ctx = index.get_context_for_issue(title, description)
        if ctx[0]:
            return f"AVAILABLE IMPORTS (use ONLY these):\n{ctx[0]}\n\n{ctx[1]}"
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Agent invocation
# ---------------------------------------------------------------------------

AGENT_TIMEOUT = 300
OPENCLAW_RUNNER_STATE = ".openclaw-runner.json"
RUNNER_LOCK_PATH = Path("memory/runner.lock")
ORCHESTRATOR_STATE_PATH = Path("memory/orchestrator.json")


@dataclass
class OpenClawRunnerState:
    session_id: str | None = None
    backend: str = "openclaw"
    agent_id: str = "main"
    worktree: str = ""
    updated_at: str = ""


def _runner_state_path(cwd: Path) -> Path:
    return cwd / OPENCLAW_RUNNER_STATE


def _runner_lock_path(cwd: Path) -> Path:
    return cwd / RUNNER_LOCK_PATH


def _orchestrator_state_path(cwd: Path) -> Path:
    return cwd / ORCHESTRATOR_STATE_PATH


def _coerce_openclaw_runner_state(payload: dict | None, cwd: Path) -> OpenClawRunnerState | None:
    if not isinstance(payload, dict):
        return None
    return OpenClawRunnerState(
        session_id=payload.get("session_id"),
        backend=str(payload.get("backend") or "openclaw"),
        agent_id=str(payload.get("agent_id") or "main"),
        worktree=str(payload.get("worktree") or cwd),
        updated_at=str(payload.get("updated_at") or ""),
    )


def _read_json_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json_file(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _runner_state_payload(cwd: Path, state: OpenClawRunnerState) -> dict:
    return {
        "session_id": state.session_id,
        "backend": state.backend or "openclaw",
        "agent_id": state.agent_id or "main",
        "worktree": state.worktree or str(cwd),
        "updated_at": state.updated_at,
    }


def _normalize_openclaw_runner_state(cwd: Path, state: OpenClawRunnerState) -> OpenClawRunnerState:
    return OpenClawRunnerState(
        session_id=state.session_id,
        backend=state.backend or "openclaw",
        agent_id=state.agent_id or "main",
        worktree=str(state.worktree or cwd),
        updated_at=state.updated_at or _utcnow(),
    )


def load_openclaw_runner_state(cwd: Path) -> OpenClawRunnerState | None:
    return _coerce_openclaw_runner_state(_read_json_file(_runner_state_path(cwd)), cwd)


def load_runner_lock_state(cwd: Path) -> OpenClawRunnerState | None:
    return _coerce_openclaw_runner_state(_read_json_file(_runner_lock_path(cwd)), cwd)


def load_orchestrator_active_runner(cwd: Path) -> OpenClawRunnerState | None:
    payload = _read_json_file(_orchestrator_state_path(cwd)) or {}
    return _coerce_openclaw_runner_state(payload.get("active_runner"), cwd)


def save_openclaw_runner_state(cwd: Path, state: OpenClawRunnerState) -> None:
    normalized = _normalize_openclaw_runner_state(cwd, state)
    _write_json_file(_runner_state_path(cwd), _runner_state_payload(cwd, normalized))


def save_runner_lock_state(cwd: Path, state: OpenClawRunnerState) -> None:
    normalized = _normalize_openclaw_runner_state(cwd, state)
    _write_json_file(_runner_lock_path(cwd), _runner_state_payload(cwd, normalized))


def save_orchestrator_active_runner(cwd: Path, state: OpenClawRunnerState | None) -> None:
    path = _orchestrator_state_path(cwd)
    payload = _read_json_file(path) or {}
    if state is None:
        payload["active_runner"] = None
    else:
        normalized = _normalize_openclaw_runner_state(cwd, state)
        payload["active_runner"] = _runner_state_payload(cwd, normalized)
    _write_json_file(path, payload)


def persist_openclaw_runner_identity(cwd: Path, state: OpenClawRunnerState) -> OpenClawRunnerState:
    normalized = _normalize_openclaw_runner_state(cwd, state)
    save_openclaw_runner_state(cwd, normalized)
    save_runner_lock_state(cwd, normalized)
    save_orchestrator_active_runner(cwd, normalized)
    return normalized


def clear_openclaw_runner_identity(cwd: Path, session_id: str | None = None) -> None:
    local_state = load_openclaw_runner_state(cwd)
    lock_state = load_runner_lock_state(cwd)
    active_state = load_orchestrator_active_runner(cwd)
    matches_session = session_id is None or any(
        state and state.session_id == session_id for state in (local_state, lock_state, active_state)
    )
    if not matches_session:
        return
    for path in (_runner_state_path(cwd), _runner_lock_path(cwd)):
        if path.exists():
            path.unlink()
    save_orchestrator_active_runner(cwd, None)


def reconcile_openclaw_runner_identity(cwd: Path) -> OpenClawRunnerState | None:
    local_state = load_openclaw_runner_state(cwd)
    lock_state = load_runner_lock_state(cwd)
    active_state = load_orchestrator_active_runner(cwd)

    resolved: OpenClawRunnerState | None = None
    if lock_state is not None:
        resolved = lock_state
    elif local_state is not None:
        resolved = local_state
    elif active_state is not None:
        resolved = active_state

    if resolved is None:
        return None

    # memory/runner.lock is the orchestration-visible source of truth. If it exists,
    # conservatively repair any disagreement by copying it back to helper/orchestrator state.
    if lock_state is not None:
        resolved = lock_state

    return persist_openclaw_runner_identity(cwd, resolved)


def _build_openclaw_message(prompt: str, cwd: Path, output_schema_file: Path | None = None) -> str:
    parts = [
        "You are the Accruvia A/B runner worker.",
        f"Repository worktree: {cwd}",
        "Operate only inside that worktree.",
        "Use OpenClaw-native tools/subagents as needed. Do not shell out to the local codex CLI.",
        "When you need Python, use /workspace/accruvia/.venv/bin/python explicitly (or /usr/bin/python3 if the repo venv is unavailable). Never call bare `python`.",
    ]

    if output_schema_file is not None:
        schema_text = output_schema_file.read_text()
        parts.extend(
            [
                "Do NOT write files for this turn.",
                "Return exactly one JSON object matching this schema, with no markdown fences or extra commentary:",
                schema_text,
            ]
        )
    else:
        parts.extend(
            [
                "Edit files directly in the worktree using file tools.",
                "Do not touch files outside the worktree.",
                "Reply briefly when done.",
            ]
        )

    parts.extend(["", "TASK:", prompt])
    return "\n".join(parts)


def _extract_openclaw_result(stdout: str) -> tuple[bool, str, TokenUsage, str | None]:
    tokens = TokenUsage()
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return False, f"openclaw output was not valid JSON: {exc}", tokens, None

    result = payload.get("result") or {}
    outputs = result.get("payloads") or []
    text_parts = []
    for item in outputs:
        if isinstance(item, dict):
            text = item.get("text")
            if text:
                text_parts.append(str(text))

    meta = result.get("meta") or {}
    agent_meta = meta.get("agentMeta") or {}
    usage = agent_meta.get("lastCallUsage") or agent_meta.get("usage") or {}
    tokens.input_tokens = int(usage.get("input") or 0)
    tokens.cached_input_tokens = int(usage.get("cacheRead") or 0)
    tokens.output_tokens = int(usage.get("output") or 0)
    session_id = agent_meta.get("sessionId")

    status = payload.get("status")
    ok = status == "ok"
    output = "\n".join(text_parts)
    if not ok and not output:
        output = str(payload)
    return ok, output, tokens, session_id


def run_openclaw(prompt: str, cwd: Path, output_schema_file: Path | None = None) -> ExecutionStatus:
    """Run the worker via OpenClaw agent delegation instead of local codex CLI."""
    state = reconcile_openclaw_runner_identity(cwd)
    agent_id = os.getenv("ACCRUVIA_RUNNER_AGENT", (state.agent_id if state else "main"))
    cmd = ["openclaw", "agent", "--json", "--timeout", str(AGENT_TIMEOUT)]
    if state and state.session_id:
        cmd.extend(["--session-id", state.session_id])
    else:
        cmd.extend(["--agent", agent_id])
    cmd.extend(["--message", _build_openclaw_message(prompt, cwd, output_schema_file=output_schema_file)])

    tokens = TokenUsage()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=AGENT_TIMEOUT + 30, cwd=str(cwd))
    except subprocess.TimeoutExpired:
        clear_openclaw_runner_identity(cwd, session_id=state.session_id if state else None)
        failure_kind, blocked_reason, auth_diagnosis = _classify_execution_failure(
            f"openclaw agent timed out after {AGENT_TIMEOUT}s"
        )
        return ExecutionStatus(False, f"openclaw agent timed out after {AGENT_TIMEOUT}s", tokens, failure_kind=failure_kind, blocked_reason=blocked_reason, auth_diagnosis=auth_diagnosis)
    except Exception as e:
        failure_kind, blocked_reason, auth_diagnosis = _classify_execution_failure(str(e))
        return ExecutionStatus(False, str(e), tokens, failure_kind=failure_kind, blocked_reason=blocked_reason, auth_diagnosis=auth_diagnosis)

    ok, output, tokens, session_id = _extract_openclaw_result(result.stdout)
    if result.returncode != 0:
        detail = output or result.stderr[:500]
        message = f"openclaw exit {result.returncode}: {detail}"
        failure_kind, blocked_reason, auth_diagnosis = _classify_execution_failure(message, returncode=result.returncode)
        return ExecutionStatus(False, message, tokens, session_id=session_id, failure_kind=failure_kind, blocked_reason=blocked_reason, auth_diagnosis=auth_diagnosis)

    resolved_session_id = session_id or (state.session_id if state else None)
    if resolved_session_id:
        persist_openclaw_runner_identity(
            cwd,
            OpenClawRunnerState(session_id=resolved_session_id, backend="openclaw", agent_id=agent_id),
        )

    if not ok:
        failure_kind, blocked_reason, auth_diagnosis = _classify_execution_failure(output)
        return ExecutionStatus(False, output, tokens, session_id=resolved_session_id, failure_kind=failure_kind, blocked_reason=blocked_reason, auth_diagnosis=auth_diagnosis)

    return ExecutionStatus(True, output, tokens, session_id=resolved_session_id, auth_diagnosis={
        "runner_backend": "openclaw",
        "provider": "openai-codex",
        "observed_at": _utcnow(),
        "status": "ok",
    })


def run_codex(prompt: str, cwd: Path, output_schema_file: Path | None = None) -> ExecutionStatus:
    """Compatibility shim: runner launches now go through OpenClaw agent delegation."""
    return run_openclaw(prompt, cwd, output_schema_file=output_schema_file)


# ---------------------------------------------------------------------------
# pytest
# ---------------------------------------------------------------------------


def run_pytest(cwd: Path) -> tuple[bool, str]:
    test_dir = cwd / "tests" / "generated"
    if not test_dir.exists() or not list(test_dir.glob("test_*.py")):
        return False, "No test files found in tests/generated/"
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_dir), "-v", "--tb=short", "-x", "-p", "no:xdist"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(cwd),
            env={**os.environ, "PYTHONPATH": str(cwd / "src")},
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "pytest timed out after 60s"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# File operations
# ---------------------------------------------------------------------------


def revert_files(cwd: Path, files: list[str]):
    """Revert specific files in worktree."""
    for f in files:
        fp = cwd / f
        if fp.exists():
            # Check if tracked or untracked
            result = subprocess.run(
                ["git", "ls-files", f],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.stdout.strip():
                subprocess.run(["git", "checkout", "--", f], cwd=str(cwd), capture_output=True, timeout=5)
            else:
                fp.unlink()


def write_artifacts_to_disk(
    cwd: Path,
    bank: ArtifactBank,
    code_schema: dict | None,
    full_file_mode: bool = False,
):
    """Write accepted artifacts from bank to worktree files.

    Args:
        full_file_mode: If True, source_code replaces the target file entirely
            (used when the prompt included target file content and asked for a
            complete updated file). If False, source_code is appended to the
            existing file for existing-file modifications.
    """
    test_code = bank.get("test_code")
    if test_code:
        test_dir = cwd / "tests" / "generated"
        test_dir.mkdir(parents=True, exist_ok=True)
        test_file = test_dir / f"test_issue_{code_schema.get('target_file', 'output').split('/')[-1]}"
        if not test_file.name.startswith("test_"):
            test_file = test_dir / "test_generated.py"
        test_file.write_text(test_code)

    source_code = bank.get("source_code")
    if source_code and code_schema:
        target = code_schema.get("target_file", "")
        is_new = code_schema.get("is_new_file", False)
        if target:
            target_path = cwd / target
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if is_new or full_file_mode or not target_path.exists():
                target_path.write_text(source_code)
            else:
                existing = target_path.read_text()
                target_path.write_text(existing.rstrip() + "\n\n\n" + source_code)


# ---------------------------------------------------------------------------
# Output schema for codex
# ---------------------------------------------------------------------------

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "code_schema": {
            "type": "object",
            "properties": {
                "target_file": {"type": "string"},
                "is_new_file": {"type": "boolean"},
                "functions": {"type": "array", "items": {"type": "string"}},
                "classes": {"type": "array", "items": {"type": "string"}},
                "rationale": {"type": "string"},
            },
            "required": ["target_file", "is_new_file", "functions", "classes", "rationale"],
            "additionalProperties": False,
        },
        "test_code": {"type": "string"},
        "source_code": {"type": "string"},
    },
    "required": ["code_schema", "test_code", "source_code"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------


def save_ab_result(result: dict, base_dir: Path) -> Path:
    """Save A/B test result to versioned directory."""
    mode = result["mode"]
    issue_id = result["issue_id"]
    run_id = result["run_id"]

    out_dir = base_dir / "data" / "ab_tests" / issue_id / f"{mode}-{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokens = result["total_tokens"]
    saved = {
        "mode": mode,
        "issue_id": issue_id,
        "run_id": run_id,
        "success": result["success"],
        "total_tokens": tokens.to_dict(),
        "total_time_s": result["total_time_s"],
        "iterations": result["iterations"],
        "execution": result.get("execution"),
        "candidate_artifacts": result.get("candidate_artifacts", {"present": False}),
        "promotion_eligible": bool(result.get("candidate_artifacts", {}).get("present")),
        "qa": result.get("qa", {"enabled": False, "reviews": [], "tokens_total": 0}),
        "workflow_learning": result.get("workflow_learning", {"present": False}),
        "throughput": result.get("throughput", {"useful": False}),
        "per_iteration": [
            {
                "tokens": it["tokens"].to_dict(),
                "time_s": it.get("time_s", 0),
                "validation": it.get("validation", {}),
                "blame": it.get("blame"),
                "execution": it.get("execution"),
            }
            for it in result["per_iteration"]
        ],
    }
    (out_dir / "result.json").write_text(json.dumps(saved, indent=2))
    return out_dir


def _make_log_dir(base_dir: Path, issue_id: str, mode: str, run_id: str) -> Path:
    """Create the output directory early so we can write progress logs."""
    out_dir = base_dir / "data" / "ab_tests" / issue_id / f"{mode}-{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _log(msg: str, log_file: Path | None = None) -> None:
    """Print to stdout and append to progress log file."""
    print(msg, flush=True)
    if log_file is not None:
        with log_file.open("a") as f:
            f.write(msg + "\n")


def build_run_result(
    *,
    mode: str,
    issue: IssueSpec,
    run_id: str,
    success: bool,
    iterations: list[dict],
    started_at: float,
    execution_status: ExecutionStatus | None = None,
    candidate_manifest: dict | None = None,
    qa_reviews: list[QAVerdict] | None = None,
) -> dict:
    workflow_learning_present = bool(
        execution_status and (not execution_status.ok) and execution_status.blocked_reason
    )
    qa_reviews = qa_reviews or []
    result = {
        "mode": mode,
        "issue_id": issue.issue_id,
        "run_id": run_id,
        "success": success,
        "total_tokens": aggregate_tokens(iterations),
        "total_time_s": time.time() - started_at,
        "iterations": len(iterations),
        "per_iteration": iterations,
        "execution": execution_status.to_dict() if execution_status else None,
        "candidate_artifacts": {
            "present": candidate_manifest is not None,
            "manifest": candidate_manifest,
        },
        "promotion_eligible": bool(candidate_manifest),
        "qa": {
            "enabled": bool(qa_reviews),
            "reviews": [verdict.to_dict() for verdict in qa_reviews],
            "tokens_total": sum(verdict.tokens_used for verdict in qa_reviews),
        },
        "workflow_learning": {
            "present": workflow_learning_present,
            "failure_kind": execution_status.failure_kind if execution_status else None,
            "blocked_reason": execution_status.blocked_reason if execution_status else None,
            "auth_diagnosis": execution_status.auth_diagnosis if execution_status else None,
        },
        "blocked_diagnosis_changed_next_action": bool(workflow_learning_present),
    }
    result["throughput"] = _throughput_summary(result)
    return result


# ---------------------------------------------------------------------------
# Main run loops
# ---------------------------------------------------------------------------


def run_schema_mode(
    issue: IssueSpec,
    base_dir: Path,
    run_id: str,
    qa_panel: QAPanel | None = None,
    max_qa_rejections: int = 2,
) -> dict:
    """Run in schema mode: agent returns JSON, we validate per-element."""
    t0 = time.time()
    log_dir = _make_log_dir(base_dir, issue.issue_id, "schema", run_id)
    log_file = log_dir / "progress.log"
    wt = setup_worktree(issue.issue_id, base_dir, f"schema-{run_id}")
    ctx = get_codebase_context(wt / "src", issue.title, issue.description)

    def log(msg: str) -> None:
        _log(msg, log_file)

    # Write output schema to temp file
    schema_file = wt / ".output_schema.json"
    schema_file.write_text(json.dumps(OUTPUT_SCHEMA, indent=2))

    bank = ArtifactBank()
    iterations = []
    code_schema_dict = None
    last_execution_status: ExecutionStatus | None = None
    candidate_manifest = None
    qa_reviews: list[QAVerdict] = []
    qa_rejections = 0

    log(f"\n{'=' * 60}")
    log(f"SCHEMA MODE — Issue #{issue.issue_id}")
    log(f"{'=' * 60}")

    iteration = 0
    while iteration < MAX_ITERATIONS:
        iteration += 1
        it_t0 = time.time()
        log(f"\n  [Iteration {iteration}] Calling agent...")

        prompt = build_schema_prompt(issue, ctx, bank, code_schema_dict=code_schema_dict, worktree=wt)
        exec_status = run_codex(prompt, wt, output_schema_file=schema_file)
        last_execution_status = exec_status
        ok, output, tokens = exec_status.ok, exec_status.output, exec_status.tokens

        it_record = {"tokens": tokens, "time_s": 0, "validation": {}, "blame": None, "execution": exec_status.to_dict()}

        if not ok:
            log(f"  [Iteration {iteration}] Agent FAILED: {output[:200]}")
            it_record["time_s"] = time.time() - it_t0
            iterations.append(it_record)
            continue

        # Parse structured response
        artifacts = parse_schema_response(output)
        if not artifacts:
            log(f"  [Iteration {iteration}] Could not parse JSON response")
            log(f"  [Iteration {iteration}] Raw output preview: {output[:300]}")
            malformed_msg = (
                "Your response was not valid JSON. You MUST return a SINGLE JSON object "
                "with all three fields: code_schema, test_code, source_code. "
                "Do NOT return multiple JSON objects. Return ONE complete JSON object. "
                f"What you returned (first 500 chars): {output[:500]}"
            )
            bank.reject("code_schema", [malformed_msg])
            bank.reject("test_code", [malformed_msg])
            bank.reject("source_code", [malformed_msg])
            it_record["time_s"] = time.time() - it_t0
            iterations.append(it_record)
            continue

        # Resolve code_schema early so allowlist derivation handles both dict
        # and JSON-string artifacts without double parsing.
        if code_schema_dict is None and "code_schema" in artifacts:
            code_schema_dict = _coerce_code_schema(artifacts.get("code_schema"))

        # Build import allowlist from code_schema target_file so test_code
        # can import a module being created by source_code in this same response.
        import_allowlist = None
        if code_schema_dict:
            target = code_schema_dict.get("target_file", "")
            if target.startswith("src/") and target.endswith(".py"):
                mod_path = target[4:].replace("/", ".").removesuffix(".py")
                import_allowlist = {mod_path}

        # Validate each artifact
        for name in REQUIRED_ARTIFACTS:
            if bank.is_accepted(name):
                continue
            content = artifacts.get(name, "")
            valid, errors = validate_artifact(
                name,
                content,
                wt / "src",
                import_allowlist=import_allowlist,
            )
            it_record["validation"][name] = {"valid": valid, "errors": errors}

            if valid:
                bank.accept(name, content)
                log(f"  [Iteration {iteration}] {name}: ACCEPTED")
                if name == "code_schema":
                    parsed_schema = _coerce_code_schema(content)
                    if parsed_schema is not None:
                        code_schema_dict = parsed_schema
            else:
                bank.reject(name, errors)
                log(f"  [Iteration {iteration}] {name}: REJECTED — {'; '.join(errors)}")

        it_record["time_s"] = time.time() - it_t0
        iterations.append(it_record)

        log(
            f"  [Iteration {iteration}] tokens: {tokens.input_tokens}in/"
            f"{tokens.output_tokens}out (cached: {tokens.cached_input_tokens})"
        )

        if not bank.all_accepted(REQUIRED_ARTIFACTS):
            continue

        if qa_panel is not None:
            verdict, qa_exec_status = run_qa_review(issue, bank, wt, qa_panel)
            if verdict is None:
                log(f"  [QA] Skipping QA gate due to execution failure: {qa_exec_status.output[:200]}")
            else:
                qa_reviews.append(verdict)
                log(f"  [QA] approved={verdict.approved} hard_rejects={len(verdict.hard_rejects)} suggestions={len(verdict.suggestions)}")
                if not verdict.approved and qa_rejections < max_qa_rejections:
                    qa_rejections += 1
                    ejected = apply_qa_verdict(bank, verdict)
                    log(f"  [QA] Rejected artifacts: {', '.join(ejected) if ejected else 'none'}")
                    continue

        # All accepted — write to disk and verify
        log("\n  [VERIFY] Writing artifacts to disk...")
        sent_file_context = _get_target_file_content(code_schema_dict, wt) is not None
        write_artifacts_to_disk(wt, bank, code_schema_dict, full_file_mode=sent_file_context)

        log("  [VERIFY] Running pytest...")
        passed, test_output = run_pytest(wt)

        if passed:
            log("  [VERIFY] PASSED")
            return build_run_result(
                mode="schema",
                issue=issue,
                run_id=run_id,
                success=True,
                iterations=iterations,
                started_at=t0,
                execution_status=last_execution_status,
                candidate_manifest=candidate_manifest,
                qa_reviews=qa_reviews,
            )

        # Blame analysis
        blamed = blame_pytest(test_output)
        log(f"  [VERIFY] FAILED — blame: {blamed}")
        bank.eject(blamed, f"pytest failure:\n{test_output[-1500:]}")

        # Revert files for clean re-write
        subprocess.run(["git", "checkout", "--", "."], cwd=str(wt), capture_output=True, timeout=10)
        subprocess.run(["git", "clean", "-fd", "tests/generated/"], cwd=str(wt), capture_output=True, timeout=10)

    log(f"\n[MAX ITERATIONS] Giving up after {iteration} iterations")
    return build_run_result(
        mode="schema",
        issue=issue,
        run_id=run_id,
        success=False,
        iterations=iterations,
        started_at=t0,
        execution_status=last_execution_status,
        candidate_manifest=candidate_manifest,
        qa_reviews=qa_reviews,
    )


def run_direct_mode(
    issue: IssueSpec,
    base_dir: Path,
    run_id: str,
    prompt_strategy: str = "baseline",
    qa_panel: QAPanel | None = None,
    max_qa_rejections: int = 2,
) -> dict:
    """Run in direct mode: agent writes files, we discover + validate.

    Args:
        prompt_strategy: "baseline" (standard prompt) or "constrained" (constraint-heavy prompt)
    """
    mode_label = prompt_strategy if prompt_strategy != "direct" else "baseline"
    prompt_builder = build_constrained_prompt if mode_label == "constrained" else build_direct_prompt

    t0 = time.time()
    log_dir = _make_log_dir(base_dir, issue.issue_id, mode_label, run_id)
    log_file = log_dir / "progress.log"
    wt = setup_worktree(issue.issue_id, base_dir, f"{mode_label}-{run_id}")
    ctx = get_codebase_context(wt / "src", issue.title, issue.description)

    def log(msg: str) -> None:
        _log(msg, log_file)

    bank = ArtifactBank()
    iterations = []
    feedback = None
    last_execution_status: ExecutionStatus | None = None
    candidate_manifest = None
    qa_reviews: list[QAVerdict] = []
    qa_rejections = 0

    log(f"\n{'=' * 60}")
    log(f"{mode_label.upper()} MODE — Issue #{issue.issue_id}")
    log(f"{'=' * 60}")

    iteration = 0
    while iteration < MAX_ITERATIONS:
        iteration += 1
        it_t0 = time.time()
        log(f"\n  [Iteration {iteration}] Calling agent...")

        prompt = prompt_builder(issue, ctx, feedback)
        exec_status = run_codex(prompt, wt)
        last_execution_status = exec_status
        ok, output, tokens = exec_status.ok, exec_status.output, exec_status.tokens

        it_record = {"tokens": tokens, "time_s": 0, "validation": {}, "blame": None, "execution": exec_status.to_dict()}

        if not ok:
            log(f"  [Iteration {iteration}] Agent FAILED: {output[:200]}")
            feedback = f"Agent call failed: {output[:500]}"
            it_record["time_s"] = time.time() - it_t0
            iterations.append(it_record)
            continue

        # Discover artifacts from disk
        artifacts = discover_artifacts(wt)
        candidate_manifest = snapshot_candidate_artifacts(
            wt,
            log_dir / "candidate_artifacts" / f"iteration-{iteration:02d}",
            issue_id=issue.issue_id,
            run_id=run_id,
            mode=mode_label,
            iteration=iteration,
            execution_status=exec_status,
        ) or candidate_manifest
        if not artifacts:
            log(f"  [Iteration {iteration}] No artifacts found on disk")
            feedback = "You didn't create or modify any files. Write test and implementation files."
            it_record["time_s"] = time.time() - it_t0
            iterations.append(it_record)
            continue

        # Check for missing artifacts
        feedback_parts = []
        all_valid = True
        for name in ["test_code", "source_code"]:
            if name not in artifacts:
                all_valid = False
                msg = f"Missing {name}: "
                if name == "test_code":
                    msg += "You MUST create a test file in tests/generated/test_*.py"
                else:
                    msg += "You MUST modify or create a source file in src/"
                feedback_parts.append(msg)
                log(f"  [Iteration {iteration}] {name}: MISSING")
                continue
            content = artifacts[name]
            valid, errors = validate_artifact(name, content, wt / "src")
            it_record["validation"][name] = {"valid": valid, "errors": errors}

            if valid:
                bank.accept(name, content)
                log(f"  [Iteration {iteration}] {name}: ACCEPTED")
                feedback_parts.append(f"{name}: ACCEPTED (locked — do not change)")
            else:
                bank.reject(name, errors)
                all_valid = False
                log(f"  [Iteration {iteration}] {name}: REJECTED — {'; '.join(errors)}")
                feedback_parts.append(f"{name}: REJECTED — {'; '.join(errors)}")
                # Revert rejected file
                actual_key = "_test_file" if name == "test_code" else "_source_file"
                if actual_key in artifacts:
                    revert_files(wt, [artifacts[actual_key]])

        it_record["time_s"] = time.time() - it_t0
        iterations.append(it_record)

        log(
            f"  [Iteration {iteration}] tokens: {tokens.input_tokens}in/"
            f"{tokens.output_tokens}out (cached: {tokens.cached_input_tokens})"
        )

        if not all_valid or "test_code" not in artifacts or "source_code" not in artifacts:
            feedback = "\n".join(feedback_parts)
            continue

        if qa_panel is not None:
            verdict, qa_exec_status = run_qa_review(issue, bank, wt, qa_panel)
            if verdict is None:
                log(f"  [QA] Skipping QA gate due to execution failure: {qa_exec_status.output[:200]}")
            else:
                qa_reviews.append(verdict)
                log(f"  [QA] approved={verdict.approved} hard_rejects={len(verdict.hard_rejects)} suggestions={len(verdict.suggestions)}")
                if not verdict.approved and qa_rejections < max_qa_rejections:
                    qa_rejections += 1
                    ejected = apply_qa_verdict(bank, verdict)
                    feedback = "\n".join(
                        feedback_parts
                        + [f"QA rejected artifacts: {', '.join(ejected) if ejected else 'none'}"]
                        + [bank.get_rejection_feedback(name) for name in ejected]
                    )
                    log(f"  [QA] Rejected artifacts: {', '.join(ejected) if ejected else 'none'}")
                    continue

        # All validated — run pytest
        log("\n  [VERIFY] Running pytest...")
        passed, test_output = run_pytest(wt)

        if passed:
            log("  [VERIFY] PASSED")
            return build_run_result(
                mode=mode_label,
                issue=issue,
                run_id=run_id,
                success=True,
                iterations=iterations,
                started_at=t0,
                execution_status=last_execution_status,
                candidate_manifest=candidate_manifest,
                qa_reviews=qa_reviews,
            )

        # Blame analysis
        blamed = blame_pytest(test_output)
        log(f"  [VERIFY] FAILED — blame: {blamed}")
        it_record["blame"] = blamed
        bank.eject(blamed, f"pytest failure:\n{test_output[-1500:]}")

        # Revert blamed file
        actual_key = "_test_file" if blamed == "test_code" else "_source_file"
        if actual_key in artifacts:
            revert_files(wt, [artifacts[actual_key]])

        feedback_parts = []
        for name in ["test_code", "source_code"]:
            if bank.is_accepted(name):
                feedback_parts.append(f"{name}: ACCEPTED (locked)")
            else:
                fb = bank.get_rejection_feedback(name)
                feedback_parts.append(f"{name}: REJECTED — {fb}")
        feedback = "\n".join(feedback_parts)

    log(f"\n[MAX ITERATIONS] Giving up after {iteration} iterations")
    return build_run_result(
        mode=mode_label,
        issue=issue,
        run_id=run_id,
        success=False,
        iterations=iterations,
        started_at=t0,
        execution_status=last_execution_status,
        candidate_manifest=candidate_manifest,
        qa_reviews=qa_reviews,
    )


# ---------------------------------------------------------------------------
# Issue definitions
# ---------------------------------------------------------------------------

ISSUE_437 = IssueSpec(
    issue_id="437",
    title="Startup validation: verify model max_tokens against provider limits",
    description="""At startup, validate that every model in MODEL_REGISTRY has a max_tokens
value that doesn't exceed the provider's actual limit. If litellm reports a lower
max_output_tokens than we configured, log a warning and cap to the provider limit.

Acceptance criteria:
- On startup, iterate MODEL_REGISTRY entries
- For each model, query litellm.get_model_info() for max_output_tokens
- If our configured max_tokens > provider limit, log warning and cap
- Add a validate_model_limits() function to llm_provider_types.py
- Unit tests covering: normal case, capped case, missing model info""",
)

ISSUE_438 = IssueSpec(
    issue_id="438",
    title="Consolidate model token limits: stages should use ModelConfig.max_tokens",
    description="""_get_model_max_output_tokens() in stages.py queries litellm at runtime and
maintains its own _max_output_cache. Since validate_model_limits() now caps
ModelConfig.max_tokens at startup, stages should just read from the registry.

The current code in src/accruvia/orchestration/ga_pipeline/stages.py:

    _max_output_cache: dict[str, int] = {}

    def _get_model_max_output_tokens(model_id: str) -> int:
        if model_id in _max_output_cache:
            return _max_output_cache[model_id]
        from accruvia.services.llm_provider_types import MODEL_REGISTRY, PROVIDER_MAX_TOKENS
        config = MODEL_REGISTRY.get(model_id)
        if not config:
            _max_output_cache[model_id] = 16384
            return 16384
        try:
            import litellm
            info = litellm.get_model_info(config.litellm_id)
            result = info.get("max_output_tokens", 16384)
        except Exception:
            result = 16384
        provider_cap = PROVIDER_MAX_TOKENS.get(config.provider)
        if provider_cap:
            result = min(result, provider_cap)
        _max_output_cache[model_id] = result
        return result

Replace this with a simple registry lookup:
- Read config.max_tokens from MODEL_REGISTRY (already capped at startup)
- Fall back to 16384 if model not in registry
- Remove _max_output_cache dict (registry IS the cache)
- Remove the litellm import from this function
- Remove the PROVIDER_MAX_TOKENS import from this function

Acceptance criteria:
- _get_model_max_output_tokens() returns ModelConfig.max_tokens from registry
- _max_output_cache module-level dict is removed
- No litellm import in _get_model_max_output_tokens
- No PROVIDER_MAX_TOKENS import in _get_model_max_output_tokens
- Default to 16384 for unknown models
- Unit tests: known model returns config.max_tokens, unknown model returns 16384""",
)

ISSUE_439 = IssueSpec(
    issue_id="439",
    title="API cost estimator for A/B test token data",
    description="""Add a cost estimation module that calculates what each A/B test run
would cost if the same token usage happened via API calls instead of CLI.

Create src/accruvia/telemetry/cost_model.py with:

1. PRICE_TABLE: dict mapping provider to per-token prices:
   - anthropic: input=$3.00/M, cached_input=$0.30/M, output=$15.00/M
   - openai: input=$2.50/M, cached_input=$1.25/M, output=$10.00/M
   - google: input=$1.25/M, cached_input=$0.315/M, output=$5.00/M
   - groq: input=$0.59/M, cached_input=$0.59/M, output=$0.79/M

2. estimate_cost(input_tokens, cached_tokens, output_tokens, provider) -> dict:
   Returns {"input_cost": float, "cached_cost": float, "output_cost": float,
   "total_cost": float, "savings_from_caching": float}
   where savings_from_caching = (cached_tokens * input_price - cached_tokens * cached_price)

3. format_cost_comparison(runs: list[dict]) -> str:
   Takes a list of run results (each with token breakdown and provider),
   returns a formatted string comparing costs across runs.
   Show per-run cost and highlight which mode is cheaper on API.

Acceptance criteria:
- PRICE_TABLE has entries for anthropic, openai, google, groq
- estimate_cost returns correct costs for each token type
- savings_from_caching shows how much caching saved vs full-price input
- format_cost_comparison produces readable output for 2+ runs
- Default provider is 'anthropic' when not specified
- Unit tests for all three functions""",
)

ISSUE_440 = IssueSpec(
    issue_id="440",
    title="Log full prompts, responses, and artifacts per A/B run",
    description="""The A/B runner saves result.json with token totals but not the actual
prompts, responses, or generated code. Without these, we can't debug why
a mode succeeded or failed, or compare solution quality across runs.

Create src/accruvia/telemetry/run_logger.py with:

1. RunLogger class:
   - __init__(self, out_dir: Path) — creates the output directory
   - log_iteration(self, iteration: int, prompt: str, raw_response: str,
     tokens: dict, validation: dict, artifacts: dict) — appends to iterations.jsonl
   - log_result(self, result: dict) — writes final result.json
   - log_diff(self, diff: str) — writes diff.patch
   - log_artifacts(self, artifacts: dict[str, str]) — writes each artifact
     to artifacts/{name}.py

2. Each iteration log entry in iterations.jsonl contains:
   - iteration number
   - full prompt sent
   - raw response received (truncated to 50KB if larger)
   - token usage
   - validation results per artifact
   - timestamp

Acceptance criteria:
- RunLogger creates out_dir on init
- log_iteration appends JSONL (one JSON object per line)
- log_artifacts writes files to artifacts/ subdirectory
- log_diff writes diff.patch
- Large responses are truncated to 50KB with a marker
- Unit tests for all methods""",
)

ISSUE_441 = IssueSpec(
    issue_id="441",
    title="Cost-aware model selection for pipeline stages",
    description="""Wire cost_model.py into live model routing so the pipeline picks
cheaper models when confidence is high, saving API spend on routine stages.

Changes required across multiple files:

1. src/accruvia/services/llm_provider_types.py:
   - Add cost_per_input_token and cost_per_output_token fields to ModelConfig
   - Populate from PRICE_TABLE values (per-million divided by 1M)
   - Models without pricing data default to the anthropic rate

2. src/accruvia/orchestration/ga_pipeline/orchestrator.py:
   - Add CostSelector class with select_model(stage, candidates, min_confidence=0.7):
     * Filters candidates to those with success_rate >= min_confidence
     * Among qualified candidates, picks the cheapest by cost_per_output_token
     * Falls back to GA selection if no candidate meets confidence threshold
   - Wire CostSelector into _choose_model() as an optional mode
   - Add enable_cost_selection flag to PipelineConfig (default False)

3. src/accruvia/telemetry/cost_model.py:
   - Add get_model_cost(model_id) -> dict returning per-token costs
   - Lookup from ModelConfig fields added in step 1

4. src/accruvia/orchestration/ga_pipeline/stages.py:
   - Each stage's execute() logs which selection method was used (GA vs cost)
   - Add selection_method field to stage result metadata

5. Tests:
   - CostSelector picks cheapest model above confidence threshold
   - CostSelector falls back to GA when no model meets threshold
   - get_model_cost returns correct rates for known models
   - Integration: PipelineConfig.enable_cost_selection toggles behavior

Acceptance criteria:
- CostSelector ranks models by cost when confidence is sufficient
- Falls back to GA exploration when confidence is low
- Pipeline stages log which selection method chose the model
- Feature is off by default (enable_cost_selection=False)
- Unit tests for CostSelector, get_model_cost, and config toggle""",
)

ISSUE_442 = IssueSpec(
    issue_id="442",
    title="Blame-driven feedback loop for GA settlement retraction",
    description="""When VERIFY fails, classify_failure() identifies which upstream stage
is to blame, but this blame is only logged — never acted upon. Close the
loop: retract the blamed stage's GA settlement and downgrade the model
that produced the faulty output.

Changes required across multiple files:

1. src/accruvia/orchestration/ga_pipeline/blame.py:
   - Add retraction_target(failure_classification) -> tuple[PipelineStage, str]
     that maps a blame classification to (stage, model_id) to retract
   - Enhance classify_failure() to return structured BlameResult with
     stage, model_id, confidence, and reason fields

2. src/accruvia/orchestration/ga_pipeline/settlement.py:
   - Add retract_settlement(stage, model_id, reason) method:
     * Finds the most recent win for that model on that stage
     * Reverses the fitness boost (subtract the original delta)
     * Logs the retraction with reason and timestamp
   - Add RetractionRecord dataclass for audit trail

3. src/accruvia/orchestration/ga_pipeline/ga_population.py:
   - Add downgrade_persona(model_id, penalty) method to ModelSelectionPopulation
   - penalty reduces the persona's fitness score (clamped to 0.0 minimum)
   - Persona is NOT removed — just deprioritized in next tournament selection

4. src/accruvia/orchestration/ga_pipeline/orchestrator.py:
   - In _handle_verify_failure(), after classify_failure():
     * Call retract_settlement() on the blamed stage
     * Call downgrade_persona() on the blamed model
     * Re-queue the blamed stage with fresh model selection
   - Add enable_blame_retraction flag to PipelineConfig (default False)

5. src/accruvia/orchestration/ga_pipeline/telemetry.py:
   - Log retraction events: stage, model, reason, fitness_before, fitness_after

6. Tests:
   - retract_settlement reverses a previous fitness boost
   - downgrade_persona reduces fitness but doesn't go below 0
   - _handle_verify_failure triggers retraction when blame confidence > 0.8
   - No retraction when blame confidence is low
   - RetractionRecord captures audit trail correctly

Acceptance criteria:
- BlameResult is a structured dataclass, not just a string
- retract_settlement undoes a specific prior settlement
- downgrade_persona penalizes but doesn't eliminate a model
- Orchestrator wires blame -> retraction -> re-queue
- Feature is off by default (enable_blame_retraction=False)
- Unit tests for all new methods""",
)

ISSUE_443 = IssueSpec(
    issue_id="443",
    title="Integrate trust registry into GA pipeline model selection",
    description="""The TrustGraduation system and GA populations are currently disconnected.
Wire trust scores into _choose_model() so stages prefer models with
proven track records when enough data exists.

Changes required across multiple files:

1. src/accruvia/orchestration/ga_pipeline/orchestrator.py:
   - Modify _choose_model() to query TrustGraduation before GA:
     * If a model has graduated (trust_level >= 'proven') for this stage,
       use it directly without GA tournament
     * If multiple models are graduated, pick the one with highest
       success_rate
     * Fall back to GA exploration if no graduated models exist
   - Add trust_weight parameter to PipelineConfig (0.0-1.0, default 0.0)
     controlling how much trust influences selection vs pure GA

2. src/accruvia/orchestration/ga_pipeline/trust_graduation.py:
   - Add get_graduated_models(stage: PipelineStage) -> list[TrustRecord]
   - Add query_trust(stage, model_id) -> TrustRecord | None
   - TrustRecord includes: model_id, trust_level, success_rate,
     total_attempts, last_success_at

3. src/accruvia/orchestration/ga_pipeline/stages.py:
   - Each stage's execute() records trust metadata in its result:
     selection_source ('trust' | 'ga' | 'cost'), model_id, trust_level
   - After successful execution, call trust_graduation.record_success()

4. src/accruvia/orchestration/ga_pipeline/models.py:
   - Add TrustRecord dataclass
   - Add selection_source field to stage result models

5. Tests:
   - _choose_model uses graduated model when trust_weight > 0
   - _choose_model falls back to GA when no graduated models
   - get_graduated_models returns only models above threshold
   - Stage results include selection_source metadata
   - trust_weight=0.0 disables trust-based selection entirely

Acceptance criteria:
- Trust-based selection is opt-in via trust_weight config
- Graduated models bypass GA tournament when trust_weight > 0
- Stage results record how the model was selected
- Falls back gracefully to GA when trust data is insufficient
- Unit tests for trust queries and selection logic""",
)

ISSUE_444 = IssueSpec(
    issue_id="444",
    title="Iterative code-fix loop instead of full regeneration on failure",
    description="""When CODE_WRITE fails validation (syntax errors, bad imports, undefined
names), the pipeline currently retries from scratch — regenerating ALL
code. This wastes tokens on large issues where only a small part is
wrong. Implement targeted fix-and-retry.

Changes required across multiple files:

1. src/accruvia/orchestration/ga_pipeline/stages.py (CodeWriteStage):
   - After validation failure, instead of full retry:
     a. Save the failed code to a staging buffer
     b. Collect specific errors from ImportValidator and NameChecker
     c. Build a fix prompt: "Here is the code you generated and the
        errors found. Fix ONLY the errors, preserve everything else."
     d. Send fix prompt to LLM (may use a different/cheaper model)
     e. Validate the fixed code
     f. Allow up to 3 fix attempts before falling back to full regen
   - Add FixAttempt dataclass: attempt_num, errors_in, code_out,
     errors_remaining, model_used, tokens_used
   - Track fix attempts separately from full regeneration attempts

2. src/accruvia/orchestration/import_validator.py:
   - Add validate_with_details(code, filename) -> ValidationResult
   - ValidationResult includes: valid (bool), errors (list of
     ImportError with line_number, module_name, error_type)
   - Current validate() becomes a thin wrapper over validate_with_details()

3. src/accruvia/orchestration/name_checker.py:
   - Add check_with_details(code) -> NameCheckResult
   - NameCheckResult includes: valid (bool), errors (list of
     UndefinedName with line_number, name, context)
   - Current check() becomes a thin wrapper over check_with_details()

4. src/accruvia/orchestration/ga_pipeline/models.py:
   - Add FixAttempt dataclass
   - Add ValidationResult and NameCheckResult dataclasses
   - Add max_fix_attempts field to PipelineConfig (default 3)

5. src/accruvia/orchestration/ga_pipeline/telemetry.py:
   - Log fix attempts: iteration, attempt_num, errors_before,
     errors_after, tokens_used, model_used
   - Distinguish fix-tokens from regen-tokens in reporting

6. Tests:
   - Fix prompt includes the specific errors and original code
   - Fix attempt succeeds when LLM corrects the error
   - Falls back to full regen after max_fix_attempts exhausted
   - validate_with_details returns structured error info
   - check_with_details returns structured error info
   - Fix tokens tracked separately from regen tokens
   - FixAttempt dataclass captures all metadata

Acceptance criteria:
- CODE_WRITE tries targeted fixes before full regeneration
- Fix prompts include specific error messages and line numbers
- Up to max_fix_attempts fixes before falling back
- ImportValidator and NameChecker provide structured error details
- Fix vs regen token usage tracked separately
- Unit tests for fix loop, structured validation, and telemetry""",
)

ISSUE_445 = IssueSpec(
    issue_id="445",
    title="Task fingerprinting for issue complexity stratification",
    description="""Extract complexity features from issues before screening so A/B results
can be stratified by issue size. Small vs large issues have different
cost profiles — we need the data to prove it.

Changes required across multiple files:

1. src/accruvia/orchestration/ga_pipeline/stages.py:
   - Add FeatureExtractionStage (or hook into ScreeningStage.execute()):
     * Parse issue description for: file count hints, import references,
       class/function mentions, estimated lines of change
     * Assign complexity_bucket: 'small' (1-2 files, <100 LOC),
       'medium' (3-5 files, 100-300 LOC), 'large' (6+ files, 300+ LOC)
   - Output TaskFingerprint attached to PipelineIssue

2. src/accruvia/orchestration/ga_pipeline/models.py:
   - Add TaskFingerprint dataclass: file_count_estimate (int),
     loc_estimate (int), import_count (int), complexity_bucket (str),
     cross_cutting (bool), extracted_at (datetime)
   - Add fingerprint field to PipelineIssue (Optional[TaskFingerprint])

3. src/accruvia/orchestration/ga_pipeline/trust_graduation.py:
   - Refine TaskFingerprint calculation using extracted features
   - Use fingerprint in trust lookups: "this model is trusted for
     small issues but not large ones"

4. src/accruvia/orchestration/ga_pipeline/telemetry.py:
   - Record fingerprint in pipeline_runs table
   - Add fingerprint columns: file_count, loc_estimate, complexity_bucket

5. scripts/ab_report.py:
   - Group results by complexity_bucket when fingerprint data exists
   - Show cost comparison per bucket: "small issues: schema saves X%,
     large issues: schema saves Y%"

6. Tests:
   - Feature extraction parses file count from issue description
   - complexity_bucket assigned correctly for each size range
   - TaskFingerprint attached to PipelineIssue after extraction
   - ab_report groups by bucket when data available
   - Missing fingerprint data handled gracefully (no crash)

Acceptance criteria:
- TaskFingerprint captures complexity signals from issue text
- complexity_bucket categorizes issues into small/medium/large
- Fingerprint stored in telemetry for post-hoc analysis
- ab_report can stratify by complexity when data exists
- Unit tests for extraction, bucketing, and report grouping""",
)

ISSUE_446 = IssueSpec(
    issue_id="446",
    title="AB runner: add max iteration cap and failure return path",
    description="""Both run_schema_mode() and run_direct_mode() have `while True` loops
with no exit condition. On hard issues the runner loops forever, burning
tokens and producing garbage cost data. This blocks all larger issues.

Fix in scripts/ab_runner.py:

1. Add MAX_ITERATIONS constant (default 10) at module level.

2. In run_schema_mode() (line ~779 `while True`):
   - Change to `while iteration < MAX_ITERATIONS`
   - After the loop, return a failure result dict:
     {"mode": "schema", "issue_id": ..., "run_id": ..., "success": False,
      "total_tokens": aggregate_tokens(iterations),
      "total_time_s": time.time() - t0,
      "iterations": iteration, "per_iteration": iterations}
   - Print "[MAX ITERATIONS] Giving up after {iteration} iterations"

3. In run_direct_mode() (line ~905 `while True`):
   - Same change: `while iteration < MAX_ITERATIONS`
   - Same failure return dict with mode="direct"
   - Same max iteration message

4. Tests:
   - Schema mode returns success=False after MAX_ITERATIONS
   - Direct mode returns success=False after MAX_ITERATIONS
   - Failure result includes correct token aggregation
   - save_ab_result handles success=False results

Acceptance criteria:
- Both loops exit after MAX_ITERATIONS
- Failure results are saved with success=False
- Token totals are correct even on failure
- ab_report.py handles failed runs in its stats""",
)

ISSUE_447 = IssueSpec(
    issue_id="447",
    title="AB runner: fix code_schema double-parse crash risk",
    description="""In run_schema_mode(), code_schema_dict is populated in two places with
inconsistent type handling:

Line ~818: `cs = code_schema_dict or json.loads(artifacts.get("code_schema", "{}"))`
  - artifacts["code_schema"] may already be a dict from parse_schema_response()
  - json.loads(dict) would raise TypeError

Line ~844: `code_schema_dict = json.loads(content) if isinstance(content, str) else content`
  - This correctly handles both types but runs AFTER the line 818 path

Fix in scripts/ab_runner.py:

1. In the import allowlist block (~line 816-825):
   - Use the same isinstance guard: parse only if content is a string
   - Or restructure so code_schema_dict is set before the allowlist block

2. Tests:
   - Schema mode handles code_schema as dict without crash
   - Schema mode handles code_schema as JSON string without crash
   - Import allowlist is correctly derived in both cases

Acceptance criteria:
- No TypeError when code_schema arrives as a dict
- Import allowlist works regardless of artifact type
- Unit test covers both code paths""",
)

ISSUE_448 = IssueSpec(
    issue_id="448",
    title="Move AB runner and telemetry into accruvia_client package",
    description="""The AB runner (scripts/ab_runner.py), telemetry modules
(src/accruvia/telemetry/cost_model.py, run_logger.py), and report script
(scripts/ab_report.py) ARE the client product — they resolve issues, validate
code, track costs, and report results. But they live in scripts/ and a generic
telemetry package instead of accruvia_client/ where they belong.

This is a pure move — no logic changes, no refactoring. Fix imports, verify
tests pass.

Steps:

1. Move scripts/ab_runner.py → src/accruvia_client/runner.py
   - Update all imports (sys.path hacks → proper package imports)
   - Add CLI entry point in src/accruvia_client/__main__.py or keep a
     thin scripts/ab_runner.py that imports and calls main()

2. Move scripts/ab_report.py → src/accruvia_client/report.py
   - Same import cleanup

3. Move src/accruvia/telemetry/cost_model.py → src/accruvia_client/telemetry/cost_model.py
   - Update imports in runner.py and report.py

4. Move src/accruvia/telemetry/run_logger.py → src/accruvia_client/telemetry/run_logger.py
   - Update imports in runner.py

5. Update test imports:
   - tests/test_ab_runner.py → import from accruvia_client.runner
   - tests/test_cost_model.py → import from accruvia_client.telemetry.cost_model
   - tests/test_run_logger.py → import from accruvia_client.telemetry.run_logger

6. Keep thin wrapper scripts in scripts/ for CLI convenience:
   - scripts/ab_runner.py: 3 lines — import and call main()
   - scripts/ab_report.py: 3 lines — import and call main()

Acceptance criteria:
- All existing tests pass with updated imports
- ab_runner CLI still works: PYTHONPATH=src python scripts/ab_runner.py --issue 446 --mode schema
- No logic changes — pure file relocation + import fixup
- accruvia_client/ is the home for all client-side functionality""",
)

ISSUE_449 = IssueSpec(
    issue_id="449",
    title="Refactor runner monolith into client modules",
    description="""After issue #448 moves ab_runner.py into accruvia_client/runner.py,
it's still a ~1500-line monolith. Split it into focused modules that reflect
the product's actual architecture.

Depends on: #448 (move first, refactor second)

Target module structure under src/accruvia_client/:

1. resolver.py — The core issue resolution engine:
   - run_schema_mode() and run_direct_mode()
   - IssueSpec dataclass
   - MAX_ITERATIONS constant
   - build_schema_prompt(), build_direct_prompt()
   - parse_schema_response(), discover_artifacts()

2. validator.py — Artifact validation pipeline:
   - validate_artifact() (ast.parse + ImportValidator + NameChecker)
   - _check_has_logic(), _coerce_code_schema()
   - ArtifactBank (accept/reject/eject state machine)
   - blame_pytest()

3. reporter.py — Reporting and cost analysis:
   - Moved from report.py
   - format_cost_comparison integration
   - Per-run and per-issue aggregation

4. telemetry/ — Already moved in #448:
   - cost_model.py (API cost estimation)
   - run_logger.py (iteration/artifact logging)

5. issues.py — Issue registry:
   - ISSUES dict
   - All IssueSpec definitions (437-449+)
   - Separated from runner logic so issues can be added without
     touching the resolver

6. runner.py — Thin orchestration layer:
   - CLI argument parsing
   - Worktree setup/teardown
   - Calls resolver with config
   - Saves results via run_logger

Each module should be independently testable. Existing tests split
accordingly into test_resolver.py, test_validator.py, etc.

Acceptance criteria:
- No module exceeds 400 lines
- Each module has a single responsibility
- All existing tests pass (relocated to match new modules)
- CLI entry point unchanged
- No logic changes — pure structural refactor""",
)

ISSUE_456 = IssueSpec(
    issue_id="456",
    title="QA review panel: single-call multi-persona code review before VERIFY",
    description="""Add an automated QA gate that reviews accepted artifacts before pytest
runs. A single LLM call evaluates the code from multiple stakeholder
perspectives, catching issues that deterministic validation misses.

Depends on: #448 (move to accruvia_client first)

The QA panel runs identically for both schema and direct modes — it sees
only the accepted artifacts, not how they were generated. This preserves
AB test fairness while adding quality signal.

Architecture:

1. src/accruvia_client/qa_panel.py — QAPanel class:

   - __init__(self, model: str = "claude-haiku-4.5") — cheap model for reviews
   - review(self, issue: IssueSpec, artifacts: dict[str, str]) -> QAVerdict
   - Single LLM call with a system prompt containing ALL reviewer personas:

     System prompt structure:
     ```
     You are a QA review panel. Evaluate this code from these perspectives:

     ## Security Reviewer
     Check for: injection risks, hardcoded secrets, unsafe deserialization,
     unvalidated input at system boundaries, OWASP top 10.

     ## Efficiency Reviewer
     Check for: O(n²) in hot paths, unnecessary allocations, redundant
     iterations, missing early returns, unbounded growth.

     ## Factorability Reviewer
     Check for: functions >50 lines, classes with >1 responsibility,
     copy-paste patterns, missing abstractions that would reduce duplication.

     ## Product Owner
     Check for: does the code actually solve the issue as described?
     Are acceptance criteria met? Any scope creep or missing requirements?

     ## UX Reviewer (API surface)
     Check for: confusing parameter names, inconsistent return types,
     missing defaults, surprising behavior, poor error messages.
     ```

   - Response schema (JSON):
     ```json
     {
       "approved": bool,
       "hard_rejects": [{"reviewer": str, "concern": str, "line": int|null}],
       "suggestions": [{"reviewer": str, "suggestion": str}],
       "summary": str
     }
     ```

   - QAVerdict dataclass:
     approved (bool), hard_rejects (list), suggestions (list), summary (str),
     tokens_used (int), model_used (str)

2. Integration into runner (both modes equally):

   Insert point: after bank.all_accepted() but before write_artifacts_to_disk()

   ```python
   if qa_panel is not None:
       verdict = qa_panel.review(issue, bank.get_all_accepted())
       if not verdict.approved:
           for reject in verdict.hard_rejects:
               # Route rejection to the blamed artifact
               blamed = "source_code"  # default
               bank.eject(blamed, f"QA {reject['reviewer']}: {reject['concern']}")
           continue  # retry with QA feedback
       # Log suggestions even on approval
       log(f"  [QA] Approved with {len(verdict.suggestions)} suggestions")
   ```

   Both modes hit this identical code path — QA doesn't know or care
   whether artifacts came from schema parsing or file discovery.

3. Configuration:
   - --qa flag on CLI to enable (off by default, doesn't affect existing runs)
   - qa_model parameter for model selection (default: cheapest available)
   - max_qa_rejections: int = 2 — don't let QA loop forever

4. Telemetry:
   - QA tokens tracked separately from generation tokens
   - QA verdicts logged in iterations.jsonl via RunLogger
   - ab_report shows QA pass/fail rate per mode

5. Tests:
   - QAPanel.review returns QAVerdict with correct structure
   - Approved verdict doesn't block VERIFY
   - Hard reject ejects artifact and triggers retry
   - max_qa_rejections prevents infinite QA loop
   - QA tokens are not counted in generation token totals
   - Both modes produce identical QA inputs for same artifacts
   - QA panel uses cheap model (not the generation model)

Acceptance criteria:
- Single LLM call per QA review (not N separate calls)
- All 5 reviewer perspectives in one system prompt
- Identical QA path for both schema and direct modes
- QA is opt-in (--qa flag), default off
- QA tokens tracked separately from generation tokens
- Hard rejects feed back through existing retry mechanism
- Suggestions logged but don't block acceptance
- Unit tests for QAPanel, verdict handling, and telemetry""",
)

ISSUES = {
    "437": ISSUE_437,
    "438": ISSUE_438,
    "439": ISSUE_439,
    "440": ISSUE_440,
    "441": ISSUE_441,
    "442": ISSUE_442,
    "443": ISSUE_443,
    "444": ISSUE_444,
    "445": ISSUE_445,
    "446": ISSUE_446,
    "447": ISSUE_447,
    "448": ISSUE_448,
    "449": ISSUE_449,
    "450": ISSUE_456,
    "456": ISSUE_456,
}


def load_issue_from_gitlab(issue_id: str, base_dir: Path) -> IssueSpec | None:
    """Fetch issue details from GitLab when the local registry is stale.

    This keeps the runner usable for newly created backlog items without
    requiring a code edit every time the issue list changes.
    """
    try:
        result = subprocess.run(
            ["glab", "issue", "view", issue_id, "-F", "json"],
            cwd=str(base_dir),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except Exception:
        return None

    if result.returncode != 0 or not result.stdout.strip():
        return None

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None

    title = payload.get("title")
    description = payload.get("description") or ""
    if not title:
        return None

    labels = []
    raw_labels = payload.get("labels") or []
    for label in raw_labels:
        if isinstance(label, dict):
            name = label.get("name")
            if name:
                labels.append(name)
        elif isinstance(label, str):
            labels.append(label)

    return IssueSpec(
        issue_id=str(payload.get("iid") or issue_id),
        title=title,
        description=description,
        labels=labels or None,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="A/B test runner for CLI issue resolution")
    parser.add_argument("--mode", required=True, choices=["baseline", "constrained", "direct", "schema"])
    parser.add_argument("--issue", default="438", help="Issue ID")
    parser.add_argument("--force", action="store_true", help="Run even if mode is retired (for re-testing)")
    parser.add_argument("--qa", action="store_true", help="Enable QA review panel before VERIFY")
    parser.add_argument("--qa-model", default="claude-haiku-4.5", help="QA review model hint")
    parser.add_argument("--max-qa-rejections", type=int, default=2, help="Max QA rejection retries before continuing")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent.parent
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    from routellect.decisions import is_retired

    if is_retired(args.mode, base_dir) and not args.force:
        print(f"Mode '{args.mode}' has been retired. See data/ab_decisions.json for details.")
        print("Use --force to run anyway (for re-testing with fixes).")
        sys.exit(1)

    issue = ISSUES.get(args.issue)
    if not issue:
        issue = load_issue_from_gitlab(args.issue, base_dir)
    if not issue:
        print(
            f"Unknown issue: {args.issue}. Available locally: {', '.join(ISSUES.keys())}. "
            "GitLab lookup also failed."
        )
        sys.exit(1)

    qa_panel = QAPanel(model=args.qa_model) if args.qa else None

    if args.mode == "schema":
        result = run_schema_mode(
            issue,
            base_dir,
            run_id,
            qa_panel=qa_panel,
            max_qa_rejections=args.max_qa_rejections,
        )
    else:
        # "direct" is legacy alias for "baseline"
        strategy = args.mode if args.mode in ("baseline", "constrained") else "baseline"
        result = run_direct_mode(
            issue,
            base_dir,
            run_id,
            prompt_strategy=strategy,
            qa_panel=qa_panel,
            max_qa_rejections=args.max_qa_rejections,
        )

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"RESULT — {result['mode']} mode")
    print(f"{'=' * 60}")
    print(f"Success:    {result['success']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Time:       {result['total_time_s']:.1f}s")
    t = result["total_tokens"]
    print(
        f"Tokens:     {t.total:,} total ({t.net_input:,} net + "
        f"{t.output_tokens:,} out + {t.cached_input_tokens:,} cached)"
    )

    out_dir = save_ab_result(result, base_dir)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
