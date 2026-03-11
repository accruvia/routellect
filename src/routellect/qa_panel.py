import json
from dataclasses import dataclass, field


QA_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "approved": {"type": "boolean"},
        "hard_rejects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "reviewer": {"type": "string"},
                    "concern": {"type": "string"},
                    "line": {"type": ["integer", "null"]},
                },
                "required": ["reviewer", "concern", "line"],
                "additionalProperties": False,
            },
        },
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "reviewer": {"type": "string"},
                    "suggestion": {"type": "string"},
                },
                "required": ["reviewer", "suggestion"],
                "additionalProperties": False,
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["approved", "hard_rejects", "suggestions", "summary"],
    "additionalProperties": False,
}


@dataclass
class QAHardReject:
    reviewer: str
    concern: str
    line: int | None = None


@dataclass
class QASuggestion:
    reviewer: str
    suggestion: str


@dataclass
class QAVerdict:
    approved: bool
    hard_rejects: list[QAHardReject] = field(default_factory=list)
    suggestions: list[QASuggestion] = field(default_factory=list)
    summary: str = ""
    tokens_used: int = 0
    model_used: str = "claude-haiku-4.5"

    def to_dict(self) -> dict:
        return {
            "approved": self.approved,
            "hard_rejects": [
                {"reviewer": item.reviewer, "concern": item.concern, "line": item.line}
                for item in self.hard_rejects
            ],
            "suggestions": [
                {"reviewer": item.reviewer, "suggestion": item.suggestion}
                for item in self.suggestions
            ],
            "summary": self.summary,
            "tokens_used": self.tokens_used,
            "model_used": self.model_used,
        }


class QAPanel:
    def __init__(self, model: str = "claude-haiku-4.5"):
        self.model = model

    def build_prompt(self, issue, artifacts: dict[str, str]) -> str:
        artifact_blocks = []
        for name in ("code_schema", "test_code", "source_code"):
            content = artifacts.get(name)
            if content:
                artifact_blocks.append(f"## {name}\n```\n{content}\n```")

        return "\n\n".join(
            [
                f"You are a QA review panel. Use model hint: {self.model}.",
                "Evaluate this code from these perspectives:",
                "## Security Reviewer\nCheck for: injection risks, hardcoded secrets, unsafe deserialization, unvalidated input at system boundaries, OWASP-style risks.",
                "## Efficiency Reviewer\nCheck for: quadratic hot paths, unnecessary allocations, redundant iterations, missing early returns, unbounded growth.",
                "## Factorability Reviewer\nCheck for: functions that are too large, mixed responsibilities, copy-paste patterns, and missing abstractions.",
                "## Product Owner\nCheck whether the implementation actually solves the issue and meets the acceptance criteria without scope creep.",
                "## UX Reviewer (API surface)\nCheck for confusing names, inconsistent return types, surprising behavior, and poor error messages.",
                f"Issue #{issue.issue_id}: {issue.title}\n\n{issue.description}",
                "Review only the accepted artifacts below. Return a single JSON object matching the requested schema.",
                *artifact_blocks,
            ]
        )

    def parse_verdict(self, raw: str, *, tokens_used: int = 0) -> QAVerdict:
        payload = json.loads(raw)
        hard_rejects = [QAHardReject(**item) for item in payload.get("hard_rejects", [])]
        suggestions = [QASuggestion(**item) for item in payload.get("suggestions", [])]
        return QAVerdict(
            approved=bool(payload.get("approved")),
            hard_rejects=hard_rejects,
            suggestions=suggestions,
            summary=str(payload.get("summary") or ""),
            tokens_used=tokens_used,
            model_used=self.model,
        )
