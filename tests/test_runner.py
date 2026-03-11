"""Tests for A/B runner — derived from approved mermaid specs.

Tests cover every atomic unit in both the schema and direct mode
workflows, as documented in docs/cli_runner_workflows.md.

Shared units:
- validate_artifact: ast.parse + ImportValidator + NameChecker per element
- blame_pytest: classify pytest output → blamed artifact name
- token aggregation across iterations
- result saving format

Schema mode units:
- parse_schema_response: extract artifacts from structured JSON
- build_schema_prompt: includes accepted bank + rejection feedback
- accept/reject bank management

Direct mode units:
- discover_artifacts: scan git changes → artifact dict
- revert_files: revert only specific files
- build_direct_prompt: includes feedback from prior failures
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Shared: validate_artifact
# ---------------------------------------------------------------------------


class TestValidateArtifact:
    """From mermaid: Local Validation (cheap, no tokens) subgraph.

    test_code checks: ast.parse, ImportValidator, NameChecker, has asserts
    source_code checks: ast.parse, ImportValidator, NameChecker, has logic
    code_schema checks: target file/dir exists, is_new_file correct, has names
    """

    def test_valid_test_code_passes(self):
        from routellect.runner import validate_artifact

        code = """
import pytest
from unittest.mock import patch

def test_example():
    result = 1 + 1
    assert result == 2
"""
        valid, errors = validate_artifact("test_code", code, src_root=Path("src"))
        assert valid, f"Expected valid but got errors: {errors}"

    def test_test_code_syntax_error_fails(self):
        from routellect.runner import validate_artifact

        code = "def test_bad(:\n    pass"
        valid, errors = validate_artifact("test_code", code, src_root=Path("src"))
        assert not valid
        assert any("syntax" in e.lower() or "parse" in e.lower() for e in errors)

    def test_test_code_no_asserts_fails(self):
        from routellect.runner import validate_artifact

        code = """
import pytest

def test_no_assertions():
    x = 1 + 1
"""
        valid, errors = validate_artifact("test_code", code, src_root=Path("src"))
        assert not valid
        assert any("assert" in e.lower() for e in errors)

    def test_test_code_hallucinated_import_fails(self):
        from routellect.runner import validate_artifact

        code = """
from pipeline.fake_module import FakeClass

def test_fake():
    assert FakeClass() is not None
"""
        valid, errors = validate_artifact("test_code", code, src_root=Path("src"))
        assert not valid
        assert any("import" in e.lower() for e in errors)

    def test_valid_source_code_passes(self):
        from routellect.runner import validate_artifact

        code = """
import logging

logger = logging.getLogger(__name__)

def validate_model_limits():
    logger.info("validating")
    return True
"""
        valid, errors = validate_artifact("source_code", code, src_root=Path("src"))
        assert valid, f"Expected valid but got errors: {errors}"

    def test_source_code_placeholder_fails(self):
        from routellect.runner import validate_artifact

        code = """
def validate_model_limits():
    pass
"""
        valid, errors = validate_artifact("source_code", code, src_root=Path("src"))
        assert not valid
        assert any("logic" in e.lower() or "placeholder" in e.lower() for e in errors)

    def test_source_code_syntax_error_fails(self):
        from routellect.runner import validate_artifact

        code = "def broken(:\n    return 1"
        valid, errors = validate_artifact("source_code", code, src_root=Path("src"))
        assert not valid

    def test_code_schema_valid(self):
        from routellect.runner import validate_artifact

        schema = json.dumps(
            {
                "target_file": "src/routellect/protocols.py",
                "is_new_file": False,
                "functions": [],
                "classes": ["AccruviaServiceProtocol"],
                "rationale": "This module holds the public routing contracts",
            }
        )
        valid, errors = validate_artifact("code_schema", schema, src_root=Path("src"))
        assert valid, f"Expected valid but got errors: {errors}"

    def test_code_schema_nonexistent_file_not_new_fails(self):
        from routellect.runner import validate_artifact

        schema = json.dumps(
            {
                "target_file": "src/accruvia/services/does_not_exist.py",
                "is_new_file": False,
                "functions": ["foo"],
                "classes": [],
            }
        )
        valid, errors = validate_artifact("code_schema", schema, src_root=Path("src"))
        assert not valid
        assert any("exist" in e.lower() for e in errors)

    def test_code_schema_no_functions_or_classes_fails(self):
        from routellect.runner import validate_artifact

        schema = json.dumps(
            {
                "target_file": "src/accruvia/services/llm_provider_types.py",
                "is_new_file": False,
                "functions": [],
                "classes": [],
            }
        )
        valid, errors = validate_artifact("code_schema", schema, src_root=Path("src"))
        assert not valid
        assert any("function" in e.lower() or "class" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# Shared: blame_pytest
# ---------------------------------------------------------------------------


class TestBlamePytest:
    """From mermaid: Blame Analysis subgraph.

    ImportError → source_code
    AssertionError → source_code
    NameError in test → test_code
    SyntaxError → check which file
    TypeError → source_code
    """

    def test_import_error_blames_source_code(self):
        from routellect.runner import blame_pytest

        output = (
            "FAILED tests/generated/test_issue_437.py::test_normal - "
            "ImportError: cannot import name 'validate_model_limits' "
            "from 'accruvia.services.llm_provider_types'"
        )
        assert blame_pytest(output) == "source_code"

    def test_assertion_error_blames_source_code(self):
        from routellect.runner import blame_pytest

        output = """FAILED tests/generated/test_issue_437.py::test_capped - AssertionError: assert 5000 == 3000"""
        assert blame_pytest(output) == "source_code"

    def test_name_error_in_test_blames_test_code(self):
        from routellect.runner import blame_pytest

        output = (
            "FAILED tests/generated/test_issue_437.py::test_normal - "
            "NameError: name 'validate_model_limits' is not defined"
        )
        assert blame_pytest(output) == "test_code"

    def test_syntax_error_in_test_file_blames_test_code(self):
        from routellect.runner import blame_pytest

        output = """SyntaxError: invalid syntax
  File "tests/generated/test_issue_437.py", line 10"""
        assert blame_pytest(output) == "test_code"

    def test_syntax_error_in_src_file_blames_source_code(self):
        from routellect.runner import blame_pytest

        output = """SyntaxError: invalid syntax
  File "src/accruvia/services/llm_provider_types.py", line 300"""
        assert blame_pytest(output) == "source_code"

    def test_type_error_blames_source_code(self):
        from routellect.runner import blame_pytest

        output = (
            "FAILED tests/generated/test_issue_437.py::test_normal - "
            "TypeError: validate_model_limits() takes 0 positional "
            "arguments but 1 was given"
        )
        assert blame_pytest(output) == "source_code"

    def test_unknown_error_defaults_to_source_code(self):
        from routellect.runner import blame_pytest

        output = """FAILED tests/generated/test_issue_437.py::test_normal - RuntimeError: something weird"""
        assert blame_pytest(output) == "source_code"


# ---------------------------------------------------------------------------
# Shared: token aggregation
# ---------------------------------------------------------------------------


class TestTokenAggregation:
    """From mermaid: Logging subgraph — cumulative token total."""

    def test_aggregates_across_iterations(self):
        from routellect.runner import TokenUsage, aggregate_tokens

        iterations = [
            {"tokens": TokenUsage(input_tokens=1000, cached_input_tokens=800, output_tokens=100)},
            {"tokens": TokenUsage(input_tokens=2000, cached_input_tokens=1500, output_tokens=200)},
        ]
        total = aggregate_tokens(iterations)
        assert total.input_tokens == 3000
        assert total.cached_input_tokens == 2300
        assert total.output_tokens == 300
        assert total.net_input == 700
        assert total.total == 3300


# ---------------------------------------------------------------------------
# Shared: accept/reject bank
# ---------------------------------------------------------------------------


class TestArtifactBank:
    """From mermaid: Accept/Reject subgraph + Blame eject."""

    def test_accept_locks_artifact(self):
        from routellect.runner import ArtifactBank

        bank = ArtifactBank()
        bank.accept("test_code", "def test_x():\n    assert True")
        assert bank.is_accepted("test_code")
        assert bank.get("test_code") == "def test_x():\n    assert True"

    def test_reject_records_error(self):
        from routellect.runner import ArtifactBank

        bank = ArtifactBank()
        bank.reject("source_code", ["SyntaxError on line 5"])
        assert not bank.is_accepted("source_code")
        assert "SyntaxError" in bank.get_rejection_feedback("source_code")

    def test_eject_removes_from_bank(self):
        from routellect.runner import ArtifactBank

        bank = ArtifactBank()
        bank.accept("test_code", "code here")
        bank.eject("test_code", "NameError in test")
        assert not bank.is_accepted("test_code")
        assert "NameError" in bank.get_rejection_feedback("test_code")

    def test_all_accepted(self):
        from routellect.runner import ArtifactBank

        bank = ArtifactBank()
        required = ["code_schema", "test_code", "source_code"]
        assert not bank.all_accepted(required)
        bank.accept("code_schema", "{}")
        bank.accept("test_code", "test")
        assert not bank.all_accepted(required)
        bank.accept("source_code", "code")
        assert bank.all_accepted(required)

    def test_pending_returns_unaccepted(self):
        from routellect.runner import ArtifactBank

        bank = ArtifactBank()
        required = ["code_schema", "test_code", "source_code"]
        bank.accept("code_schema", "{}")
        pending = bank.pending(required)
        assert "test_code" in pending
        assert "source_code" in pending
        assert "code_schema" not in pending


# ---------------------------------------------------------------------------
# Schema mode: parse_schema_response
# ---------------------------------------------------------------------------


class TestParseSchemaResponse:
    """From mermaid: Parse structured JSON response node."""

    def test_parses_valid_json(self):
        from routellect.runner import parse_schema_response

        raw = json.dumps(
            {
                "code_schema": {"target_file": "src/foo.py", "is_new_file": True, "functions": ["bar"], "classes": []},
                "test_code": "def test_bar():\n    assert True",
                "source_code": "def bar():\n    return 1",
            }
        )
        result = parse_schema_response(raw)
        assert "code_schema" in result
        assert "test_code" in result
        assert "source_code" in result

    def test_extracts_json_from_markdown_fence(self):
        from routellect.runner import parse_schema_response

        raw = """Here is the response:
```json
{"code_schema": {}, "test_code": "test", "source_code": "code"}
```
"""
        result = parse_schema_response(raw)
        assert result is not None
        assert "test_code" in result

    def test_returns_none_on_garbage(self):
        from routellect.runner import parse_schema_response

        result = parse_schema_response("this is not json at all")
        assert result is None

    def test_returns_none_on_missing_fields(self):
        from routellect.runner import parse_schema_response

        raw = json.dumps({"test_code": "test"})  # missing source_code
        result = parse_schema_response(raw)
        assert result is None

    def test_handles_multiple_json_objects_takes_last(self):
        from routellect.runner import parse_schema_response

        partial = json.dumps({"code_schema": {}, "test_code": "", "source_code": ""})
        complete = json.dumps(
            {
                "code_schema": {"target_file": "src/foo.py"},
                "test_code": "def test_it(): assert True",
                "source_code": "def foo(): return 1",
            }
        )
        raw = partial + "\n" + complete
        result = parse_schema_response(raw)
        assert result is not None
        assert "src/foo.py" in result["code_schema"]

    def test_handles_concatenated_json_objects(self):
        from routellect.runner import parse_schema_response

        obj1 = '{"code_schema":{},"test_code":"","source_code":""}'
        obj2 = '{"code_schema":{"target_file":"src/bar.py"},"test_code":"test","source_code":"code"}'
        raw = obj1 + obj2  # no newline between
        result = parse_schema_response(raw)
        assert result is not None
        assert "src/bar.py" in result["code_schema"]


# ---------------------------------------------------------------------------
# Schema mode: build_schema_prompt
# ---------------------------------------------------------------------------


class TestBuildSchemaPrompt:
    """From mermaid: Agent Request subgraph — prompt includes accepted + rejected."""

    def test_includes_issue_and_context(self):
        from routellect.runner import ArtifactBank, IssueSpec, build_schema_prompt

        issue = IssueSpec(issue_id="437", title="Test", description="Do the thing")
        prompt = build_schema_prompt(issue, "IMPORTS HERE", ArtifactBank())
        assert "437" in prompt
        assert "Do the thing" in prompt
        assert "IMPORTS HERE" in prompt

    def test_includes_accepted_artifacts(self):
        from routellect.runner import ArtifactBank, IssueSpec, build_schema_prompt

        issue = IssueSpec(issue_id="1", title="T", description="D")
        bank = ArtifactBank()
        bank.accept("code_schema", '{"target_file": "src/foo.py"}')
        prompt = build_schema_prompt(issue, "", bank)
        assert "ACCEPTED" in prompt or "accepted" in prompt
        assert "src/foo.py" in prompt

    def test_includes_rejection_feedback(self):
        from routellect.runner import ArtifactBank, IssueSpec, build_schema_prompt

        issue = IssueSpec(issue_id="1", title="T", description="D")
        bank = ArtifactBank()
        bank.reject("test_code", ["SyntaxError on line 5"])
        prompt = build_schema_prompt(issue, "", bank)
        assert "REJECTED" in prompt or "rejected" in prompt
        assert "SyntaxError" in prompt

    def test_includes_target_file_content_when_modifying(self, tmp_path):
        from routellect.runner import ArtifactBank, IssueSpec, build_schema_prompt

        # Create a fake target file in the worktree
        target = tmp_path / "src" / "accruvia" / "services" / "thing.py"
        target.parent.mkdir(parents=True)
        target.write_text("import logging\n\ndef existing_func():\n    return 42\n")

        issue = IssueSpec(issue_id="441", title="Fix thing", description="Update thing.py")
        code_schema = {
            "target_file": "src/accruvia/services/thing.py",
            "is_new_file": False,
            "functions": ["existing_func"],
        }
        prompt = build_schema_prompt(issue, "", ArtifactBank(), code_schema_dict=code_schema, worktree=tmp_path)
        assert "CURRENT CONTENT OF src/accruvia/services/thing.py" in prompt
        assert "import logging" in prompt
        assert "def existing_func" in prompt
        assert "complete updated file" in prompt.lower()

    def test_no_target_content_for_new_file(self, tmp_path):
        from routellect.runner import ArtifactBank, IssueSpec, build_schema_prompt

        issue = IssueSpec(issue_id="440", title="New module", description="Create new")
        code_schema = {
            "target_file": "src/accruvia/services/brand_new.py",
            "is_new_file": True,
            "functions": ["new_func"],
        }
        prompt = build_schema_prompt(issue, "", ArtifactBank(), code_schema_dict=code_schema, worktree=tmp_path)
        assert "CURRENT CONTENT" not in prompt

    def test_no_target_content_without_code_schema(self):
        from routellect.runner import ArtifactBank, IssueSpec, build_schema_prompt

        issue = IssueSpec(issue_id="1", title="T", description="D")
        prompt = build_schema_prompt(issue, "", ArtifactBank())
        assert "CURRENT CONTENT" not in prompt


# ---------------------------------------------------------------------------
# Schema mode: _get_target_file_content
# ---------------------------------------------------------------------------


class TestGetTargetFileContent:
    """Tests for _get_target_file_content — reads existing file for schema context."""

    def test_returns_content_for_existing_file(self, tmp_path):
        from routellect.runner import _get_target_file_content

        target = tmp_path / "src" / "mod.py"
        target.parent.mkdir(parents=True)
        target.write_text("x = 1\n")
        schema = {"target_file": "src/mod.py", "is_new_file": False}
        assert _get_target_file_content(schema, tmp_path) == "x = 1\n"

    def test_returns_none_for_new_file(self, tmp_path):
        from routellect.runner import _get_target_file_content

        schema = {"target_file": "src/new.py", "is_new_file": True}
        assert _get_target_file_content(schema, tmp_path) is None

    def test_returns_none_when_no_schema(self, tmp_path):
        from routellect.runner import _get_target_file_content

        assert _get_target_file_content(None, tmp_path) is None

    def test_returns_none_when_no_worktree(self):
        from routellect.runner import _get_target_file_content

        schema = {"target_file": "src/mod.py", "is_new_file": False}
        assert _get_target_file_content(schema, None) is None

    def test_returns_none_when_file_missing(self, tmp_path):
        from routellect.runner import _get_target_file_content

        schema = {"target_file": "src/nonexistent.py", "is_new_file": False}
        assert _get_target_file_content(schema, tmp_path) is None

    def test_returns_none_when_is_new_file_defaults_true(self, tmp_path):
        from routellect.runner import _get_target_file_content

        # No is_new_file key — defaults to True, so no content
        schema = {"target_file": "src/mod.py"}
        assert _get_target_file_content(schema, tmp_path) is None


# ---------------------------------------------------------------------------
# Schema mode: write_artifacts_to_disk
# ---------------------------------------------------------------------------


class TestWriteArtifactsToDisk:
    """Tests for write_artifacts_to_disk — append vs replace behavior."""

    def test_append_mode_existing_file(self, tmp_path):
        from routellect.runner import ArtifactBank, write_artifacts_to_disk

        target = tmp_path / "src" / "mod.py"
        target.parent.mkdir(parents=True)
        target.write_text("import os\n\ndef existing():\n    pass\n")

        bank = ArtifactBank()
        bank.accept("source_code", "def new_func():\n    return 1\n")
        schema = {"target_file": "src/mod.py", "is_new_file": False}

        write_artifacts_to_disk(tmp_path, bank, schema, full_file_mode=False)

        content = target.read_text()
        assert "import os" in content
        assert "def existing" in content
        assert "def new_func" in content

    def test_full_file_mode_replaces_existing(self, tmp_path):
        from routellect.runner import ArtifactBank, write_artifacts_to_disk

        target = tmp_path / "src" / "mod.py"
        target.parent.mkdir(parents=True)
        target.write_text("import os\n\ndef existing():\n    pass\n")

        bank = ArtifactBank()
        bank.accept(
            "source_code",
            "import os\nimport logging\n\ndef existing():\n    pass\n\ndef new_func():\n    return 1\n",
        )
        schema = {"target_file": "src/mod.py", "is_new_file": False}

        write_artifacts_to_disk(tmp_path, bank, schema, full_file_mode=True)

        content = target.read_text()
        assert "import logging" in content
        assert "def new_func" in content
        # Should NOT have duplicate existing content
        assert content.count("def existing") == 1

    def test_new_file_always_writes_directly(self, tmp_path):
        from routellect.runner import ArtifactBank, write_artifacts_to_disk

        bank = ArtifactBank()
        bank.accept("source_code", "def brand_new():\n    return 42\n")
        schema = {"target_file": "src/new_mod.py", "is_new_file": True}

        write_artifacts_to_disk(tmp_path, bank, schema, full_file_mode=False)

        target = tmp_path / "src" / "new_mod.py"
        assert target.exists()
        assert "def brand_new" in target.read_text()


# ---------------------------------------------------------------------------
# Direct mode: discover_artifacts
# ---------------------------------------------------------------------------


class TestDiscoverArtifacts:
    """From mermaid: Discover Artifacts subgraph — scan git changes."""

    def test_finds_test_and_src_files(self, tmp_path):
        from routellect.runner import discover_artifacts

        # Create a fake worktree with test and src files
        test_dir = tmp_path / "tests" / "generated"
        test_dir.mkdir(parents=True)
        (test_dir / "test_issue_437.py").write_text("def test_x():\n    assert True")

        src_dir = tmp_path / "src" / "accruvia" / "services"
        src_dir.mkdir(parents=True)
        (src_dir / "new_module.py").write_text("def foo(): return 1")

        # Mock git to return these files
        with patch(
            "routellect.runner.get_changed_files",
            return_value=[
                "tests/generated/test_issue_437.py",
                "src/accruvia/services/new_module.py",
            ],
        ):
            artifacts = discover_artifacts(tmp_path)

        assert "test_code" in artifacts
        assert "assert True" in artifacts["test_code"]
        assert "source_code" in artifacts
        assert "def foo" in artifacts["source_code"]

    def test_no_test_file_returns_no_test_code(self, tmp_path):
        from routellect.runner import discover_artifacts

        src_dir = tmp_path / "src" / "accruvia"
        src_dir.mkdir(parents=True)
        (src_dir / "mod.py").write_text("x = 1")

        with patch(
            "routellect.runner.get_changed_files",
            return_value=[
                "src/accruvia/mod.py",
            ],
        ):
            artifacts = discover_artifacts(tmp_path)

        assert "test_code" not in artifacts
        assert "source_code" in artifacts


# ---------------------------------------------------------------------------
# Direct mode: build_direct_prompt
# ---------------------------------------------------------------------------


class TestBuildDirectPrompt:
    """From mermaid: Agent Request subgraph — no schema, includes feedback."""

    def test_includes_issue_and_context(self):
        from routellect.runner import IssueSpec, build_direct_prompt

        issue = IssueSpec(issue_id="437", title="Test", description="Do the thing")
        prompt = build_direct_prompt(issue, "IMPORTS HERE", feedback=None)
        assert "437" in prompt
        assert "Do the thing" in prompt
        assert "IMPORTS HERE" in prompt

    def test_includes_feedback_when_provided(self):
        from routellect.runner import IssueSpec, build_direct_prompt

        issue = IssueSpec(issue_id="1", title="T", description="D")
        prompt = build_direct_prompt(
            issue, "", feedback="test_code REJECTED: SyntaxError\nsource_code ACCEPTED (locked)"
        )
        assert "SyntaxError" in prompt
        assert "ACCEPTED" in prompt or "locked" in prompt


# ---------------------------------------------------------------------------
# Shared: result saving
# ---------------------------------------------------------------------------


class TestResultSaving:
    """From mermaid: Results & Artifacts subgraph."""

    def test_saves_required_files(self, tmp_path):
        from routellect.runner import TokenUsage, save_ab_result

        result = {
            "mode": "schema",
            "issue_id": "437",
            "run_id": "20260307-120000",
            "success": True,
            "total_tokens": TokenUsage(input_tokens=1000, cached_input_tokens=800, output_tokens=100),
            "total_time_s": 185.0,
            "iterations": 1,
            "per_iteration": [],
        }
        out_dir = save_ab_result(result, tmp_path)

        assert (out_dir / "result.json").exists()
        saved = json.loads((out_dir / "result.json").read_text())
        assert saved["mode"] == "schema"
        assert saved["success"] is True
        assert saved["total_tokens"]["input"] == 1000
        assert saved["total_tokens"]["net"] == 200

    def test_saves_to_correct_path(self, tmp_path):
        from routellect.runner import TokenUsage, save_ab_result

        result = {
            "mode": "direct",
            "issue_id": "437",
            "run_id": "test-run",
            "success": False,
            "total_tokens": TokenUsage(),
            "total_time_s": 0,
            "iterations": 0,
            "per_iteration": [],
        }
        out_dir = save_ab_result(result, tmp_path)
        assert "ab_tests" in str(out_dir)
        assert "437" in str(out_dir)
        assert "direct-test-run" in str(out_dir)


# ---------------------------------------------------------------------------
# Prompt strategy: constrained vs baseline
# ---------------------------------------------------------------------------


class TestPromptStrategy:
    def test_baseline_prompt_has_rules(self):
        from routellect.runner import IssueSpec, build_direct_prompt

        issue = IssueSpec(issue_id="1", title="Test", description="Desc")
        prompt = build_direct_prompt(issue, "")
        assert "Import from real project modules" in prompt
        assert "STRICT CONSTRAINTS" not in prompt

    def test_constrained_prompt_has_constraints(self):
        from routellect.runner import IssueSpec, build_constrained_prompt

        issue = IssueSpec(issue_id="42", title="Test", description="Desc")
        prompt = build_constrained_prompt(issue, "")
        assert "STRICT CONSTRAINTS" in prompt
        assert "test_issue_42.py" in prompt
        assert "NEVER invent module paths" in prompt

    def test_constrained_includes_feedback(self):
        from routellect.runner import IssueSpec, build_constrained_prompt

        issue = IssueSpec(issue_id="1", title="Test", description="Desc")
        prompt = build_constrained_prompt(issue, "", feedback="fix imports")
        assert "PRIOR ATTEMPT FEEDBACK" in prompt
        assert "fix imports" in prompt


# ---------------------------------------------------------------------------
# _coerce_code_schema: handles dict and JSON string without crash
# ---------------------------------------------------------------------------

from routellect.runner import _coerce_code_schema  # noqa: E402


class TestCoerceCodeSchema:
    def test_returns_dict_as_is(self):
        d = {"target_file": "src/foo.py", "is_new_file": True}
        assert _coerce_code_schema(d) == d

    def test_parses_json_string(self):
        s = json.dumps({"target_file": "src/bar.py"})
        result = _coerce_code_schema(s)
        assert result == {"target_file": "src/bar.py"}

    def test_returns_none_for_invalid_json(self):
        assert _coerce_code_schema("not json{") is None

    def test_returns_none_for_non_dict_json(self):
        assert _coerce_code_schema(json.dumps([1, 2, 3])) is None

    def test_returns_none_for_non_string_non_dict(self):
        assert _coerce_code_schema(42) is None
        assert _coerce_code_schema(None) is None


# ---------------------------------------------------------------------------
# OpenClaw runner delegation
# ---------------------------------------------------------------------------


class TestOpenClawRunnerDelegation:
    def test_extract_openclaw_result_parses_payload_and_usage(self):
        from routellect.runner import _extract_openclaw_result

        raw = json.dumps(
            {
                "status": "ok",
                "result": {
                    "payloads": [{"text": "done", "mediaUrl": None}],
                    "meta": {
                        "agentMeta": {
                            "sessionId": "sess-123",
                            "lastCallUsage": {"input": 11, "output": 7, "cacheRead": 5},
                        }
                    },
                },
            }
        )

        ok, output, tokens, session_id = _extract_openclaw_result(raw)
        assert ok is True
        assert output == "done"
        assert tokens.input_tokens == 11
        assert tokens.output_tokens == 7
        assert tokens.cached_input_tokens == 5
        assert session_id == "sess-123"

    def test_save_and_load_openclaw_runner_state(self, tmp_path):
        from routellect.runner import (
            OpenClawRunnerState,
            load_openclaw_runner_state,
            load_orchestrator_active_runner,
            load_runner_lock_state,
            persist_openclaw_runner_identity,
        )

        state = OpenClawRunnerState(session_id="sess-9", agent_id="main")
        persist_openclaw_runner_identity(tmp_path, state)

        loaded = load_openclaw_runner_state(tmp_path)
        locked = load_runner_lock_state(tmp_path)
        active = load_orchestrator_active_runner(tmp_path)
        assert loaded is not None
        assert locked is not None
        assert active is not None
        assert loaded.session_id == "sess-9"
        assert locked.session_id == "sess-9"
        assert active.session_id == "sess-9"
        assert loaded.agent_id == "main"
        assert loaded.backend == "openclaw"
        assert loaded.worktree == str(tmp_path)

    def test_run_codex_shim_uses_openclaw_session_id_when_present(self, tmp_path):
        from routellect.runner import OpenClawRunnerState, persist_openclaw_runner_identity, run_codex

        persist_openclaw_runner_identity(tmp_path, OpenClawRunnerState(session_id="sess-55", agent_id="main"))

        payload = json.dumps(
            {
                "status": "ok",
                "result": {
                    "payloads": [{"text": "worker reply"}],
                    "meta": {
                        "agentMeta": {
                            "sessionId": "sess-55",
                            "lastCallUsage": {"input": 3, "output": 2, "cacheRead": 1},
                        }
                    },
                },
            }
        )

        with patch("routellect.runner.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = payload
            mock_run.return_value.stderr = ""

            status = run_codex("do work", tmp_path)

        cmd = mock_run.call_args.args[0]
        assert "openclaw" in cmd[0]
        assert "--session-id" in cmd
        assert "sess-55" in cmd
        assert status.ok is True
        assert status.output == "worker reply"
        assert status.tokens.total == 5

    def test_reconcile_prefers_runner_lock_and_repairs_local_helper(self, tmp_path):
        from routellect.runner import (
            OpenClawRunnerState,
            load_openclaw_runner_state,
            load_orchestrator_active_runner,
            reconcile_openclaw_runner_identity,
            save_openclaw_runner_state,
            save_orchestrator_active_runner,
            save_runner_lock_state,
        )

        save_openclaw_runner_state(tmp_path, OpenClawRunnerState(session_id="sess-local", agent_id="main"))
        save_runner_lock_state(tmp_path, OpenClawRunnerState(session_id="sess-lock", agent_id="main"))
        save_orchestrator_active_runner(tmp_path, OpenClawRunnerState(session_id="sess-old", agent_id="main"))

        resolved = reconcile_openclaw_runner_identity(tmp_path)
        local = load_openclaw_runner_state(tmp_path)
        active = load_orchestrator_active_runner(tmp_path)

        assert resolved is not None
        assert resolved.session_id == "sess-lock"
        assert local is not None and local.session_id == "sess-lock"
        assert active is not None and active.session_id == "sess-lock"

    def test_reconcile_bootstraps_runner_lock_from_local_helper_when_missing(self, tmp_path):
        from routellect.runner import (
            OpenClawRunnerState,
            load_runner_lock_state,
            reconcile_openclaw_runner_identity,
            save_openclaw_runner_state,
        )

        save_openclaw_runner_state(tmp_path, OpenClawRunnerState(session_id="sess-local", agent_id="main"))
        resolved = reconcile_openclaw_runner_identity(tmp_path)
        locked = load_runner_lock_state(tmp_path)

        assert resolved is not None
        assert resolved.session_id == "sess-local"
        assert locked is not None and locked.session_id == "sess-local"

    def test_run_codex_persists_session_id_to_all_identity_records(self, tmp_path):
        from routellect.runner import (
            load_openclaw_runner_state,
            load_orchestrator_active_runner,
            load_runner_lock_state,
            run_codex,
        )

        payload = json.dumps(
            {
                "status": "ok",
                "result": {
                    "payloads": [{"text": "worker reply"}],
                    "meta": {
                        "agentMeta": {
                            "sessionId": "sess-77",
                            "lastCallUsage": {"input": 3, "output": 2, "cacheRead": 1},
                        }
                    },
                },
            }
        )

        with patch("routellect.runner.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = payload
            mock_run.return_value.stderr = ""

            status = run_codex("do work", tmp_path)

        assert status.ok is True
        assert status.output == "worker reply"
        assert status.tokens.total == 5
        assert load_openclaw_runner_state(tmp_path).session_id == "sess-77"
        assert load_runner_lock_state(tmp_path).session_id == "sess-77"
        assert load_orchestrator_active_runner(tmp_path).session_id == "sess-77"

    def test_build_openclaw_message_includes_schema_and_worktree(self, tmp_path):
        from routellect.runner import _build_openclaw_message

        schema_file = tmp_path / "schema.json"
        schema_file.write_text('{"type":"object"}')
        message = _build_openclaw_message("solve it", tmp_path, output_schema_file=schema_file)

        assert str(tmp_path) in message
        assert 'Do NOT write files for this turn.' in message
        assert '{"type":"object"}' in message
        assert 'solve it' in message


class TestExecutionFailureClassification:
    def test_auth_failures_are_classified_as_workflow_env_failures(self):
        from routellect.runner import _classify_execution_failure

        kind, reason, diagnosis = _classify_execution_failure(
            "openclaw exit 1: 401 Unauthorized: Missing bearer or basic authentication in header",
            returncode=1,
        )

        assert kind == "failure_env"
        assert reason == "openclaw_execution_auth_failed"
        assert "401" in diagnosis["message"]


class TestCandidateArtifactPreservation:
    def test_snapshot_candidate_artifacts_preserves_changed_files_and_diff(self, tmp_path):
        from routellect.runner import ExecutionStatus, TokenUsage, snapshot_candidate_artifacts

        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)
        tracked = tmp_path / "src" / "mod.py"
        tracked.parent.mkdir(parents=True)
        tracked.write_text("x = 1\n")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)

        tracked.write_text("x = 2\n")
        untracked = tmp_path / "tests" / "generated" / "test_issue_demo.py"
        untracked.parent.mkdir(parents=True)
        untracked.write_text("def test_demo():\n    assert True\n")

        manifest = snapshot_candidate_artifacts(
            tmp_path,
            tmp_path / "artifacts",
            issue_id="456",
            run_id="run-1",
            mode="baseline",
            iteration=1,
            execution_status=ExecutionStatus(True, "ok", TokenUsage(), session_id="sess-1"),
        )

        assert manifest is not None
        assert "src/mod.py" in manifest["changed_files"]
        assert "tests/generated/test_issue_demo.py" in manifest["changed_files"]
        assert (tmp_path / "artifacts" / "manifest.json").exists()
        assert (tmp_path / "artifacts" / "worktree.diff").exists()
        assert (tmp_path / "artifacts" / "files" / "src" / "mod.py").read_text() == "x = 2\n"


class TestIssueRegistry:
    def test_qa_panel_issue_is_registered_under_456(self):
        from routellect.runner import ISSUES

        assert "456" in ISSUES
        assert ISSUES["456"].issue_id == "456"
        assert "QA review panel" in ISSUES["456"].title


class TestQAPanel:
    def test_build_prompt_contains_all_reviewers_and_artifacts(self):
        from routellect.qa_panel import QAPanel
        from routellect.runner import IssueSpec

        panel = QAPanel(model="cheap-model")
        issue = IssueSpec(issue_id="456", title="QA panel", description="Review accepted artifacts")
        prompt = panel.build_prompt(
            issue,
            {
                "code_schema": '{"target_file":"src/demo.py"}',
                "test_code": "def test_demo():\n    assert True\n",
                "source_code": "def demo():\n    return True\n",
            },
        )

        assert "Security Reviewer" in prompt
        assert "Efficiency Reviewer" in prompt
        assert "Factorability Reviewer" in prompt
        assert "Product Owner" in prompt
        assert "UX Reviewer" in prompt
        assert "## code_schema" in prompt
        assert "## test_code" in prompt
        assert "## source_code" in prompt

    def test_parse_verdict_returns_structured_dataclass(self):
        from routellect.qa_panel import QAPanel

        panel = QAPanel()
        verdict = panel.parse_verdict(
            json.dumps(
                {
                    "approved": False,
                    "hard_rejects": [{"reviewer": "Security Reviewer", "concern": "Unsanitized input", "line": 12}],
                    "suggestions": [{"reviewer": "UX Reviewer", "suggestion": "Clarify error message"}],
                    "summary": "Needs work",
                }
            ),
            tokens_used=17,
        )

        assert verdict.approved is False
        assert verdict.hard_rejects[0].reviewer == "Security Reviewer"
        assert verdict.suggestions[0].suggestion == "Clarify error message"
        assert verdict.tokens_used == 17


class TestQAVerdictHandling:
    def test_apply_qa_verdict_ejects_blamed_artifacts(self):
        from routellect.qa_panel import QAHardReject, QAVerdict
        from routellect.runner import ArtifactBank, apply_qa_verdict

        bank = ArtifactBank()
        bank.accept("test_code", "def test_ok():\n    assert True\n")
        bank.accept("source_code", "def f():\n    return 1\n")

        verdict = QAVerdict(
            approved=False,
            hard_rejects=[
                QAHardReject(reviewer="Security Reviewer", concern="Unsafe deserialization", line=9),
                QAHardReject(reviewer="Test Reviewer", concern="Test misses assertion edge case", line=3),
            ],
            summary="reject",
        )

        ejected = apply_qa_verdict(bank, verdict)
        assert "source_code" in ejected
        assert "test_code" in ejected
        assert bank.is_accepted("source_code") is False
        assert bank.is_accepted("test_code") is False


class TestRunResultMetadata:
    def test_build_run_result_requires_candidate_artifacts_for_promotion(self):
        from routellect.runner import ExecutionStatus, IssueSpec, TokenUsage, build_run_result

        issue = IssueSpec(issue_id="456", title="T", description="D")
        result = build_run_result(
            mode="baseline",
            issue=issue,
            run_id="run-1",
            success=True,
            iterations=[{"tokens": TokenUsage(input_tokens=1, output_tokens=1)}],
            started_at=0.0,
            execution_status=ExecutionStatus(True, "ok", TokenUsage()),
            candidate_manifest=None,
        )

        assert result["promotion_eligible"] is False
        assert result["candidate_artifacts"]["present"] is False
        assert result["throughput"]["useful"] is False

    def test_build_run_result_counts_blocked_auth_diagnosis_as_useful_learning(self):
        from routellect.runner import ExecutionStatus, IssueSpec, TokenUsage, build_run_result

        issue = IssueSpec(issue_id="456", title="T", description="D")
        result = build_run_result(
            mode="baseline",
            issue=issue,
            run_id="run-1",
            success=False,
            iterations=[{"tokens": TokenUsage()}],
            started_at=time.time(),
            execution_status=ExecutionStatus(
                False,
                "401 Unauthorized",
                TokenUsage(),
                failure_kind="failure_env",
                blocked_reason="openclaw_execution_auth_failed",
                auth_diagnosis={"provider": "openai-codex"},
            ),
            candidate_manifest=None,
        )

        assert result["workflow_learning"]["present"] is True
        assert result["throughput"]["useful"] is True
        assert result["workflow_learning"]["blocked_reason"] == "openclaw_execution_auth_failed"

    def test_build_run_result_tracks_qa_tokens_separately_from_generation(self):
        from routellect.qa_panel import QAVerdict
        from routellect.runner import ExecutionStatus, IssueSpec, TokenUsage, build_run_result

        issue = IssueSpec(issue_id="456", title="T", description="D")
        result = build_run_result(
            mode="baseline",
            issue=issue,
            run_id="run-1",
            success=True,
            iterations=[{"tokens": TokenUsage(input_tokens=5, output_tokens=7)}],
            started_at=time.time(),
            execution_status=ExecutionStatus(True, "ok", TokenUsage()),
            candidate_manifest=None,
            qa_reviews=[QAVerdict(approved=True, summary="ok", tokens_used=13)],
        )

        assert result["total_tokens"].total == 12
        assert result["qa"]["enabled"] is True
        assert result["qa"]["tokens_total"] == 13
