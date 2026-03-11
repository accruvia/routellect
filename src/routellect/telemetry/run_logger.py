"""Run-level telemetry logger for A/B runner outputs."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

MAX_RESPONSE_BYTES = 50 * 1024
TRUNCATION_MARKER = "\n[TRUNCATED: raw_response exceeded 50KB]"


class RunLogger:
    """Persist prompts, responses, validation, and generated artifacts per run."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def log_iteration(
        self,
        iteration: int,
        prompt: str,
        raw_response: str,
        tokens: dict,
        validation: dict,
        artifacts: dict,
    ) -> None:
        entry = {
            "iteration": iteration,
            "prompt": prompt,
            "raw_response": self._truncate_response(raw_response),
            "tokens": dict(tokens),
            "validation": dict(validation),
            "artifacts": dict(artifacts),
            "timestamp": dt.datetime.now(dt.UTC).isoformat(),
        }

        iterations_path = self.out_dir / "iterations.jsonl"
        with iterations_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def log_result(self, result: dict) -> None:
        result_path = self.out_dir / "result.json"
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    def log_diff(self, diff: str) -> None:
        diff_path = self.out_dir / "diff.patch"
        diff_path.write_text(diff, encoding="utf-8")

    def log_artifacts(self, artifacts: dict[str, str]) -> None:
        artifacts_dir = self.out_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        for name, content in artifacts.items():
            artifact_path = artifacts_dir / f"{name}.py"
            artifact_path.write_text(content, encoding="utf-8")

    @staticmethod
    def _truncate_response(raw_response: str) -> str:
        payload = raw_response.encode("utf-8")
        if len(payload) <= MAX_RESPONSE_BYTES:
            return raw_response

        trimmed = payload[:MAX_RESPONSE_BYTES].decode("utf-8", errors="ignore")
        return f"{trimmed}{TRUNCATION_MARKER}"
