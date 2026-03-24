"""Session-level batch grading via a cheap LLM call.

Buffers conversation exchanges per session.  When a trigger fires (session
idle, buffer full, or strong signal detected), the batch is sent to a small
grading model (haiku by default) to rate each assistant response.  Grades
are persisted to the local SQLite DB and fed back to the selector.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from routellect.protocols import RoutingDecision, RoutingOutcome
from routellect.proxy._grades_db import (
    GradeRecord,
    RoutingRecord,
    ensure_session,
    log_routing,
    save_grades,
)

logger = logging.getLogger("routellect.proxy")

# Default grading model — cheapest option that can follow structured output.
DEFAULT_GRADER_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_GRADER_PROVIDER = "anthropic"
DEFAULT_BATCH_SIZE = 10
DEFAULT_IDLE_SECONDS = 120  # Grade after 2 min of silence.

_GRADING_PROMPT = """\
You are a conversation quality grader.  Below is a sequence of user/assistant
exchanges from an LLM-powered coding session.  For each assistant response,
rate whether the user was satisfied.

Signals of a GOOD response (grade: pass):
- User continues building on the response
- User says "thanks", "perfect", "great", moves to a new topic
- User asks a follow-up that shows the response was useful

Signals of a BAD response (grade: fail):
- User says "no", "wrong", "that's not what I asked"
- User re-sends a very similar message (retry)
- User expresses frustration ("stop", "ugh", profanity)
- User explicitly corrects the assistant

Signals of a MIXED response (grade: mixed):
- User partially accepts but corrects part of it
- User says "close but..." or "almost"

Return a JSON array with one object per assistant response:
[
  {"index": 0, "grade": "pass"|"mixed"|"fail", "confidence": 0.0-1.0, "reason": "<5 words>"},
  ...
]

Return ONLY the JSON array, no other text.

---

CONVERSATION:
"""


@dataclass
class _Exchange:
    """One user→assistant round-trip."""

    message_index: int
    user_message: str
    assistant_response: str
    model_used: str
    provider: str
    decision: RoutingDecision
    latency_ms: int
    input_tokens: int
    output_tokens: int


@dataclass
class SessionBuffer:
    """Accumulates exchanges for a single session."""

    session_id: str
    exchanges: list[_Exchange] = field(default_factory=list)
    last_activity: float = field(default_factory=time.monotonic)
    graded: bool = False

    @property
    def size(self) -> int:
        return len(self.exchanges)


class Grader:
    """Manages session buffers and triggers batch grading."""

    def __init__(
        self,
        credentials: dict[str, str],
        selector: Any = None,
        grader_model: str = DEFAULT_GRADER_MODEL,
        grader_provider: str = DEFAULT_GRADER_PROVIDER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        idle_seconds: float = DEFAULT_IDLE_SECONDS,
    ) -> None:
        self.credentials = credentials
        self.selector = selector
        self.grader_model = grader_model
        self.grader_provider = grader_provider
        self.batch_size = batch_size
        self.idle_seconds = idle_seconds
        self._sessions: dict[str, SessionBuffer] = {}

    def get_or_create_session(self, session_id: str | None = None) -> SessionBuffer:
        if session_id is None:
            session_id = uuid.uuid4().hex[:12]
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionBuffer(session_id=session_id)
            ensure_session(session_id)
        return self._sessions[session_id]

    def record_exchange(
        self,
        session_id: str,
        message_index: int,
        user_message: str,
        assistant_response: str,
        decision: RoutingDecision,
        latency_ms: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Buffer an exchange and log routing to DB."""
        buf = self.get_or_create_session(session_id)
        exchange = _Exchange(
            message_index=message_index,
            user_message=user_message,
            assistant_response=assistant_response,
            model_used=decision.model_id,
            provider=decision.backend,
            decision=decision,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        buf.exchanges.append(exchange)
        buf.last_activity = time.monotonic()

        # Log to DB immediately
        log_routing(RoutingRecord(
            session_id=session_id,
            message_index=message_index,
            model_used=decision.model_id,
            provider=decision.backend,
            is_exploration=decision.is_exploration,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ))

    def should_grade(self, session_id: str) -> bool:
        """Check if a session buffer should be graded now."""
        buf = self._sessions.get(session_id)
        if buf is None or buf.graded or buf.size == 0:
            return False
        if buf.size >= self.batch_size:
            return True
        if (time.monotonic() - buf.last_activity) >= self.idle_seconds:
            return True
        return False

    def check_idle_sessions(self) -> list[str]:
        """Return session IDs that are idle and ready for grading."""
        return [
            sid for sid in self._sessions
            if self.should_grade(sid)
        ]

    def _build_grading_messages(self, buf: SessionBuffer) -> list[dict[str, str]]:
        """Build the conversation text for the grading prompt."""
        lines: list[str] = []
        for ex in buf.exchanges:
            lines.append(f"[Exchange {ex.message_index}, model: {ex.model_used}]")
            # Truncate messages to avoid blowing up the grading context
            user_msg = ex.user_message[:2000]
            asst_msg = ex.assistant_response[:2000]
            lines.append(f"USER: {user_msg}")
            lines.append(f"ASSISTANT: {asst_msg}")
            lines.append("")

        return [{"role": "user", "content": _GRADING_PROMPT + "\n".join(lines)}]

    def _parse_grades(self, raw: str, buf: SessionBuffer) -> list[GradeRecord]:
        """Parse the grader's JSON response into GradeRecords."""
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Grader returned invalid JSON for session %s", buf.session_id)
            return []

        if not isinstance(items, list):
            return []

        # Build a lookup of exchange index -> exchange
        exchange_map = {ex.message_index: ex for ex in buf.exchanges}
        records: list[GradeRecord] = []

        for item in items:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            grade = item.get("grade", "mixed")
            confidence = float(item.get("confidence", 0.5))
            reason = str(item.get("reason", ""))[:100]

            if grade not in ("pass", "mixed", "fail"):
                grade = "mixed"

            ex = exchange_map.get(idx)
            if ex is None:
                continue

            records.append(GradeRecord(
                session_id=buf.session_id,
                message_index=idx,
                model_used=ex.model_used,
                provider=ex.provider,
                grade=grade,
                confidence=confidence,
                reason=reason,
                is_exploration=ex.decision.is_exploration,
            ))

        return records

    async def grade_session(self, session_id: str) -> list[GradeRecord]:
        """Send a session buffer to the grading model and persist results."""
        buf = self._sessions.get(session_id)
        if buf is None or buf.size == 0:
            return []

        from routellect.proxy._translation import forward_completion

        messages = self._build_grading_messages(buf)
        start = time.monotonic()

        try:
            response = await forward_completion(
                provider=self.grader_provider,
                model_id=self.grader_model,
                messages=messages,
                stream=False,
                credentials=self.credentials,
                max_tokens=1000,
                temperature=0,
            )
        except Exception as exc:
            logger.error("Grading call failed for session %s: %s", session_id, exc)
            return []

        elapsed_ms = int((time.monotonic() - start) * 1000)
        data = response.model_dump() if hasattr(response, "model_dump") else response
        usage = data.get("usage", {})
        raw_content = ""
        choices = data.get("choices", [])
        if choices:
            raw_content = choices[0].get("message", {}).get("content", "")

        # Estimate grading cost (haiku rates)
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        grading_cost = (input_tokens / 1_000_000) * 0.80 + (output_tokens / 1_000_000) * 4.00

        records = self._parse_grades(raw_content, buf)
        avg_confidence = (
            sum(r.confidence for r in records) / len(records) if records else 0.0
        )

        if records:
            save_grades(
                grades=records,
                session_id=session_id,
                batch_size=buf.size,
                grading_cost_usd=grading_cost,
                grader_model=self.grader_model,
                avg_confidence=avg_confidence,
            )

            # Feed back to selector
            if self.selector:
                for rec in records:
                    ex_match = next(
                        (ex for ex in buf.exchanges if ex.message_index == rec.message_index),
                        None,
                    )
                    if ex_match:
                        self.selector.record_outcome(
                            ex_match.decision,
                            RoutingOutcome(
                                success=rec.grade == "pass",
                                latency_ms=ex_match.latency_ms,
                                input_tokens=ex_match.input_tokens,
                                output_tokens=ex_match.output_tokens,
                                cost=grading_cost / max(len(records), 1),
                                qa_result=rec.grade,
                                extra={"grader_confidence": rec.confidence, "reason": rec.reason},
                            ),
                        )

        buf.graded = True
        logger.info(
            "Graded session %s: %d exchanges, %d grades, cost=$%.4f, avg_confidence=%.2f",
            session_id,
            buf.size,
            len(records),
            grading_cost,
            avg_confidence,
        )

        return records

    def flush_session(self, session_id: str) -> None:
        """Remove a session buffer from memory (after grading)."""
        self._sessions.pop(session_id, None)
