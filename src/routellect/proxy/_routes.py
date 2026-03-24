"""Proxy route handlers — provider-agnostic universal routing.

Every inbound request (OpenAI, Anthropic, Google) follows the same flow:
    1. Detect format from URL
    2. Normalize to OpenAI messages format
    3. Model selector picks provider/model
    4. forward_completion() via litellm
    5. Translate response back to original inbound format
    6. Record outcome for grading
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

from routellect.protocols import ModelSelectorProtocol, RoutingDecision, RoutingOutcome
from routellect.proxy._circuit_breaker import CircuitBreaker
from routellect.proxy._formats import (
    InboundFormat,
    NormalizedRequest,
    StreamState,
    build_task_fingerprint,
    error_response,
    extract_assistant_text,
    extract_usage,
    extract_user_message,
    normalize_to_openai,
    stream_epilogue,
    translate_response,
    translate_stream_chunk,
)
from routellect.proxy._grader import Grader
from routellect.proxy._middleware import scrub_keys
from routellect.proxy._provider_registry import build_model_universe
from routellect.proxy._translation import forward_completion

logger = logging.getLogger("routellect.proxy")

# Patterns that indicate a provider is down (not transient).
_PROVIDER_DOWN_PATTERNS = [
    "credit balance is too low",
    "billing",
    "payment required",
    "quota exceeded",
    "rate limit",
    "authentication_error",
    "invalid_api_key",
    "invalid bearer token",
    "account deactivated",
]


def _is_provider_down(error_text: str) -> bool:
    lower = error_text.lower()
    return any(p in lower for p in _PROVIDER_DOWN_PATTERNS)


class _ProviderDownError(Exception):
    pass


class ProxyRoutes:
    """Universal proxy — accepts any format, routes to any provider."""

    def __init__(
        self,
        selector: ModelSelectorProtocol,
        credentials: dict[str, str],
        grader: Grader | None = None,
    ) -> None:
        self.selector = selector
        self.credentials = credentials
        self._models = build_model_universe(credentials)
        self.selector.set_model_universe(self._models)
        self.grader = grader or Grader(credentials=credentials, selector=selector)
        self.circuit_breaker = CircuitBreaker()
        self._session_msg_counters: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Thin endpoint handlers — one line each
    # ------------------------------------------------------------------

    async def chat_completions(self, request: Request) -> Any:
        """POST /v1/chat/completions — OpenAI format."""
        return await self._route_completion(request, InboundFormat.OPENAI)

    async def anthropic_messages(self, request: Request) -> Any:
        """POST /v1/messages — Anthropic Messages format."""
        return await self._route_completion(request, InboundFormat.ANTHROPIC)

    async def google_generate(self, request: Request) -> Any:
        """POST /v1beta/models/{model}:generateContent — Google format."""
        return await self._route_completion(request, InboundFormat.GOOGLE)

    # ------------------------------------------------------------------
    # Universal routing handler
    # ------------------------------------------------------------------

    async def _route_completion(self, request: Request, fmt: InboundFormat) -> Any:
        """Accept any format, route through selector, return in original format."""
        body = await request.json()

        # For Google, the model may be in the URL path
        if fmt == InboundFormat.GOOGLE:
            model_path = request.path_params.get("model_path", "")
            if model_path and "model" not in body:
                body["model"] = model_path

        normalized = normalize_to_openai(body, fmt)
        fingerprint = build_task_fingerprint(normalized)
        session_id = self._get_session_id(request, body)
        user_message = extract_user_message(body, fmt)

        # Background grading of idle sessions
        asyncio.create_task(self._maybe_grade_idle_sessions())

        # Circuit breaker retry loop — same for all formats
        down = self.circuit_breaker.get_down_providers()
        exclude = down if down else None

        for _attempt in range(3):
            decision: RoutingDecision = self.selector.select_model(
                fingerprint,
                constraints={"exclude_backends": exclude} if exclude else None,
            )
            start = time.monotonic()

            try:
                response = await forward_completion(
                    provider=decision.backend,
                    model_id=decision.model_id,
                    messages=normalized.messages,
                    stream=normalized.stream,
                    credentials=self.credentials,
                    **normalized.params,
                )
            except Exception as exc:
                error_msg = str(exc)
                if _is_provider_down(error_msg):
                    elapsed_ms = int((time.monotonic() - start) * 1000)
                    self.selector.record_outcome(
                        decision,
                        RoutingOutcome(success=False, latency_ms=elapsed_ms, failure_kind="provider_down"),
                    )
                    self.circuit_breaker.record_failure(decision.backend)
                    exclude = self.circuit_breaker.get_down_providers()
                    logger.warning("Provider %s is down (%s), failing over...", decision.backend, scrub_keys(error_msg[:200]))
                    continue
                # Non-retryable error
                elapsed_ms = int((time.monotonic() - start) * 1000)
                self.selector.record_outcome(
                    decision,
                    RoutingOutcome(success=False, latency_ms=elapsed_ms, failure_kind="provider_error"),
                )
                return JSONResponse(
                    error_response(fmt, scrub_keys(error_msg)),
                    status_code=502,
                )

            self.circuit_breaker.record_success(decision.backend)

            if normalized.stream:
                return self._stream_response(
                    response, fmt, decision, start, session_id,
                    user_message, normalized.original_model,
                )
            else:
                return self._sync_response(
                    response, fmt, decision, start, session_id,
                    user_message, normalized.original_model,
                )

        # All providers failed
        return JSONResponse(
            error_response(fmt, "All providers failed. Check API keys and billing."),
            status_code=502,
        )

    # ------------------------------------------------------------------
    # Response handlers
    # ------------------------------------------------------------------

    def _sync_response(
        self,
        response: Any,
        fmt: InboundFormat,
        decision: RoutingDecision,
        start: float,
        session_id: str,
        user_message: str,
        original_model: str,
    ) -> JSONResponse:
        """Handle a non-streaming response in any format."""
        data = response.model_dump() if hasattr(response, "model_dump") else response
        input_tokens, output_tokens = extract_usage(data)
        assistant_text = extract_assistant_text(data)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        self._record_outcome(
            decision, elapsed_ms, session_id, user_message,
            assistant_text, input_tokens, output_tokens,
        )

        translated = translate_response(data, fmt, original_model)
        headers = self._response_headers(decision, session_id)
        return JSONResponse(translated, headers=headers)

    def _stream_response(
        self,
        response: Any,
        fmt: InboundFormat,
        decision: RoutingDecision,
        start: float,
        session_id: str,
        user_message: str,
        original_model: str,
    ) -> StreamingResponse:
        """Handle a streaming response in any format."""
        state = StreamState(original_model=original_model)

        async def event_generator():
            try:
                async for chunk in response:
                    chunk_data = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
                    lines = translate_stream_chunk(chunk_data, fmt, state)
                    for line in lines:
                        yield line
            finally:
                # Epilogue
                for line in stream_epilogue(fmt, state):
                    yield line

                # Record outcome
                elapsed_ms = int((time.monotonic() - start) * 1000)
                assistant_text = "".join(state.collected_content)[:2000]
                self._record_outcome(
                    decision, elapsed_ms, session_id, user_message,
                    assistant_text, state.input_tokens, state.output_tokens,
                )

        headers = self._response_headers(decision, session_id)
        headers["content-type"] = "text/event-stream"
        return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)

    # ------------------------------------------------------------------
    # Outcome recording (one place, not six)
    # ------------------------------------------------------------------

    def _record_outcome(
        self,
        decision: RoutingDecision,
        elapsed_ms: int,
        session_id: str,
        user_message: str,
        assistant_text: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Record routing outcome and grading exchange."""
        self.selector.record_outcome(
            decision,
            RoutingOutcome(
                success=True,
                latency_ms=elapsed_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
        )
        msg_idx = self._session_msg_counters.get(session_id, 0)
        self._session_msg_counters[session_id] = msg_idx + 1
        self.grader.record_exchange(
            session_id=session_id,
            message_index=msg_idx,
            user_message=user_message,
            assistant_response=assistant_text[:2000],
            decision=decision,
            latency_ms=elapsed_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_session_id(self, request: Request, body: dict) -> str:
        sid = request.headers.get("x-routellect-session-id")
        if not sid:
            sid = body.get("session_id")
        if not sid:
            client_host = request.client.host if request.client else "unknown"
            sid = f"auto-{client_host}"
        return str(sid)

    def _response_headers(self, decision: RoutingDecision, session_id: str) -> dict[str, str]:
        return {
            "x-routellect-model": decision.model_id,
            "x-routellect-routed": "true",
            "x-routellect-exploration": str(decision.is_exploration).lower(),
            "x-routellect-session-id": session_id,
        }

    async def _maybe_grade_idle_sessions(self) -> None:
        idle = self.grader.check_idle_sessions()
        for sid in idle:
            try:
                await self.grader.grade_session(sid)
                self.grader.flush_session(sid)
            except Exception as exc:
                logger.warning("Background grading failed for %s: %s", sid, exc)

    # ------------------------------------------------------------------
    # Non-routing endpoints (dashboard, health, models, controls)
    # ------------------------------------------------------------------

    async def list_models(self, request: Request) -> JSONResponse:
        """GET /v1/models — list available models in OpenAI format."""
        model_list = [
            {"id": m.model_id, "object": "model", "created": 0, "owned_by": m.provider}
            for m in self._models
        ]
        return JSONResponse({"object": "list", "data": model_list})

    async def health(self, request: Request) -> JSONResponse:
        """GET /health — proxy status."""
        providers: dict[str, int] = {}
        for m in self._models:
            providers.setdefault(m.provider, 0)
            providers[m.provider] += 1
        return JSONResponse({
            "status": "ok",
            "providers": providers,
            "total_models": len(self._models),
        })

    async def api_provider_reenable(self, request: Request) -> JSONResponse:
        """POST /api/provider/reenable — manually re-enable a downed provider."""
        body = await request.json()
        provider = body.get("provider", "")
        if not provider:
            return JSONResponse({"error": "provider is required"}, status_code=400)
        self.circuit_breaker.re_enable(provider)
        return JSONResponse({"provider": provider, "status": "re-enabled"})

    async def api_selector_toggle(self, request: Request) -> JSONResponse:
        """POST /api/selector/toggle — toggle demotion lock."""
        self.selector.locked = not self.selector.locked
        return JSONResponse({
            "locked": self.selector.locked,
            "current_tier": getattr(self.selector, "current_tier", 1),
        })

    async def api_stats(self, request: Request) -> JSONResponse:
        """GET /api/stats — model performance data for the dashboard."""
        from routellect.proxy._grades_db import _get_db, query_model_stats, query_recent_grades
        from routellect.proxy._provider_registry import TIER_LABELS, get_model_tier

        grade_stats = query_model_stats()
        grade_map: dict[str, dict] = {}
        for row in grade_stats:
            model = row["model_used"]
            if model not in grade_map:
                grade_map[model] = {"model": model, "provider": row["provider"], "pass": 0, "fail": 0, "mixed": 0, "total_confidence": 0, "grade_count": 0}
            g = grade_map[model]
            g[row["grade"]] = row["count"]
            g["total_confidence"] += row["avg_confidence"] * row["count"]
            g["grade_count"] += row["count"]

        conn = _get_db()
        try:
            call_rows = conn.execute(
                "SELECT model_used, provider, COUNT(*) as calls, AVG(latency_ms) as avg_latency "
                "FROM routing_log GROUP BY model_used, provider"
            ).fetchall()
        finally:
            conn.close()

        call_map = {r["model_used"]: {"calls": r["calls"], "avg_latency": int(r["avg_latency"] or 0)} for r in call_rows}

        models = []
        all_model_ids = set(grade_map.keys()) | set(call_map.keys())
        for model_id in sorted(all_model_ids):
            gm = grade_map.get(model_id, {})
            cm = call_map.get(model_id, {"calls": 0, "avg_latency": 0})
            provider = gm.get("provider", "")
            if not provider:
                for m in self._models:
                    if m.model_id == model_id:
                        provider = m.provider
                        break
            p = gm.get("pass", 0)
            f = gm.get("fail", 0)
            total_graded = p + f + gm.get("mixed", 0)
            pass_rate = round(p / max(p + f, 1) * 100)
            avg_conf = round(gm.get("total_confidence", 0) / max(gm.get("grade_count", 1), 1), 2)
            tier = get_model_tier(provider, model_id)
            models.append({
                "model": model_id, "provider": provider, "tier": tier,
                "calls": cm["calls"], "pass": p, "fail": f, "mixed": gm.get("mixed", 0),
                "passRate": pass_rate, "avgLatency": cm["avg_latency"], "avgConfidence": avg_conf,
            })
        models.sort(key=lambda m: (m["tier"], -m["calls"]))

        selector = self.selector
        sel_state = {
            "current_tier": getattr(selector, "current_tier", 1),
            "current_tier_label": TIER_LABELS.get(getattr(selector, "current_tier", 1), "unknown"),
            "trial_tier": getattr(selector, "trial_tier", None),
            "locked": getattr(selector, "locked", False),
        }

        recent = query_recent_grades(limit=20)

        ungraded = []
        conn2 = _get_db()
        try:
            ungraded_rows = conn2.execute(
                """SELECT r.session_id, r.model_used, r.latency_ms, r.input_tokens,
                          r.output_tokens, r.timestamp
                   FROM routing_log r
                   LEFT JOIN grades g ON r.session_id = g.session_id
                                     AND r.message_index = g.message_index
                   WHERE g.id IS NULL
                   ORDER BY r.timestamp DESC LIMIT 30"""
            ).fetchall()
            ungraded = [dict(r) for r in ungraded_rows]
        except Exception:
            pass
        finally:
            conn2.close()

        return JSONResponse({
            "models": models,
            "recent_grades": recent,
            "selector": sel_state,
            "failed_backends": self.circuit_breaker.get_down_providers(),
            "provider_status": self.circuit_breaker.get_status(),
            "ungraded_queue": ungraded,
        })
