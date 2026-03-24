"""Proxy route handlers: /v1/chat/completions, /v1/models, /health."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

from routellect.protocols import ModelSelectorProtocol, RoutingDecision, RoutingOutcome
from routellect.proxy._grader import Grader
from routellect.proxy._middleware import scrub_keys
from routellect.proxy._provider_registry import build_model_universe
from routellect.proxy._translation import forward_completion

logger = logging.getLogger("routellect.proxy")


def _build_task_fingerprint(body: dict[str, Any]) -> dict[str, Any]:
    """Extract non-sensitive features from the request for routing decisions.

    Never includes message content — only structural metadata.
    """
    messages = body.get("messages", [])
    return {
        "message_count": len(messages),
        "has_system_prompt": any(m.get("role") == "system" for m in messages),
        "has_tools": bool(body.get("tools")),
        "requested_model": body.get("model", ""),
        "stream": body.get("stream", False),
        "max_tokens": body.get("max_tokens"),
    }


class ProxyRoutes:
    """Encapsulates route handlers and shared state."""

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
        self._session_msg_counters: dict[str, int] = {}

    def _get_session_id(self, request: Request, body: dict) -> str:
        """Extract or generate a session ID for grading correlation."""
        # Check header, then body, then generate
        sid = request.headers.get("x-routellect-session-id")
        if not sid:
            sid = body.get("session_id")
        if not sid:
            # Use a per-connection session derived from client info
            client_host = request.client.host if request.client else "unknown"
            sid = f"auto-{client_host}"
        return str(sid)

    def _get_last_user_message(self, body: dict) -> str:
        """Extract the last user message content (for grading context)."""
        messages = body.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content[:2000]
                if isinstance(content, list):
                    # Multi-part content
                    parts = [p.get("text", "") for p in content if isinstance(p, dict)]
                    return " ".join(parts)[:2000]
        return ""

    async def _maybe_grade_idle_sessions(self) -> None:
        """Check for idle sessions and grade them in the background."""
        idle = self.grader.check_idle_sessions()
        for sid in idle:
            try:
                await self.grader.grade_session(sid)
                self.grader.flush_session(sid)
            except Exception as exc:
                logger.warning("Background grading failed for %s: %s", sid, exc)

    async def chat_completions(self, request: Request) -> Any:
        """POST /v1/chat/completions — core proxy endpoint."""
        body = await request.json()
        fingerprint = _build_task_fingerprint(body)
        stream = body.get("stream", False)
        session_id = self._get_session_id(request, body)

        # Check for idle sessions to grade (non-blocking)
        asyncio.create_task(self._maybe_grade_idle_sessions())

        # Let the selector decide
        decision: RoutingDecision = self.selector.select_model(fingerprint)

        # Forward to real provider
        start = time.monotonic()
        fwd_kwargs: dict[str, Any] = {}
        for key in ("temperature", "max_tokens", "top_p", "stop", "tools", "tool_choice"):
            if key in body:
                fwd_kwargs[key] = body[key]

        try:
            response = await forward_completion(
                provider=decision.backend,
                model_id=decision.model_id,
                messages=body.get("messages", []),
                stream=stream,
                credentials=self.credentials,
                **fwd_kwargs,
            )
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            self.selector.record_outcome(
                decision,
                RoutingOutcome(success=False, latency_ms=elapsed_ms, failure_kind="provider_error"),
            )
            message = scrub_keys(str(exc))
            logger.error("Provider error (%s/%s): %s", decision.backend, decision.model_id, message)
            return JSONResponse(
                {"error": {"message": message, "type": "provider_error"}},
                status_code=502,
            )

        user_message = self._get_last_user_message(body)

        if stream:
            return await self._handle_stream(response, decision, start, session_id, user_message)
        else:
            return self._handle_sync(response, decision, start, session_id, user_message)

    async def _handle_stream(
        self,
        response: Any,
        decision: RoutingDecision,
        start: float,
        session_id: str = "",
        user_message: str = "",
    ) -> StreamingResponse:
        """Proxy an SSE streaming response."""
        total_chunks = 0
        collected_content: list[str] = []

        async def event_generator():
            nonlocal total_chunks
            try:
                async for chunk in response:
                    total_chunks += 1
                    data = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
                    # Collect content for grading
                    for choice in data.get("choices", []):
                        delta = choice.get("delta", {})
                        if delta.get("content"):
                            collected_content.append(delta["content"])
                    yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                elapsed_ms = int((time.monotonic() - start) * 1000)
                self.selector.record_outcome(
                    decision,
                    RoutingOutcome(success=True, latency_ms=elapsed_ms),
                )
                # Record exchange for grading
                msg_idx = self._session_msg_counters.get(session_id, 0)
                self._session_msg_counters[session_id] = msg_idx + 1
                self.grader.record_exchange(
                    session_id=session_id,
                    message_index=msg_idx,
                    user_message=user_message,
                    assistant_response="".join(collected_content)[:2000],
                    decision=decision,
                    latency_ms=elapsed_ms,
                )

        headers = {
            "x-routellect-model": decision.model_id,
            "x-routellect-routed": "true",
            "x-routellect-exploration": str(decision.is_exploration).lower(),
            "x-routellect-session-id": session_id,
        }
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers=headers,
        )

    def _handle_sync(
        self,
        response: Any,
        decision: RoutingDecision,
        start: float,
        session_id: str = "",
        user_message: str = "",
    ) -> JSONResponse:
        """Handle a non-streaming response."""
        elapsed_ms = int((time.monotonic() - start) * 1000)

        data = response.model_dump() if hasattr(response, "model_dump") else response
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        self.selector.record_outcome(
            decision,
            RoutingOutcome(
                success=True,
                latency_ms=elapsed_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
        )

        # Record exchange for grading
        assistant_content = ""
        choices = data.get("choices", [])
        if choices:
            msg = choices[0].get("message") or {}
            assistant_content = msg.get("content") or ""

        msg_idx = self._session_msg_counters.get(session_id, 0)
        self._session_msg_counters[session_id] = msg_idx + 1
        self.grader.record_exchange(
            session_id=session_id,
            message_index=msg_idx,
            user_message=user_message,
            assistant_response=str(assistant_content)[:2000],
            decision=decision,
            latency_ms=elapsed_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        headers = {
            "x-routellect-model": decision.model_id,
            "x-routellect-routed": "true",
            "x-routellect-exploration": str(decision.is_exploration).lower(),
            "x-routellect-session-id": session_id,
        }
        return JSONResponse(data, headers=headers)

    async def anthropic_messages(self, request: Request) -> Any:
        """POST /v1/messages — Anthropic Messages API compatible endpoint.

        Accepts Anthropic-format requests, makes a routing decision, and
        forwards to the real provider.  For Anthropic→Anthropic routing,
        the request is passed through with the real API key.  For cross-
        provider routing, litellm handles format translation.
        """
        body = await request.json()
        messages = body.get("messages", [])
        system = body.get("system", "")
        stream = body.get("stream", False)
        session_id = self._get_session_id(request, body)

        # Build fingerprint from Anthropic-format request
        fingerprint = {
            "message_count": len(messages),
            "has_system_prompt": bool(system),
            "has_tools": bool(body.get("tools")),
            "requested_model": body.get("model", ""),
            "stream": stream,
            "max_tokens": body.get("max_tokens"),
        }

        asyncio.create_task(self._maybe_grade_idle_sessions())

        decision: RoutingDecision = self.selector.select_model(fingerprint)
        user_message = self._get_last_user_message(body)
        start = time.monotonic()

        # If routing to an Anthropic model, pass through directly via httpx
        # to preserve the native Anthropic request format exactly.
        if decision.backend == "anthropic":
            return await self._anthropic_passthrough(
                body, decision, start, stream, session_id, user_message,
            )

        # Cross-provider: convert Anthropic format to OpenAI via litellm
        openai_messages = []
        if system:
            sys_text = system if isinstance(system, str) else json.dumps(system)
            openai_messages.append({"role": "system", "content": sys_text})
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                # Anthropic content blocks → extract text
                parts = [b.get("text", "") for b in content if b.get("type") == "text"]
                content = "\n".join(parts)
            openai_messages.append({"role": msg.get("role", "user"), "content": content})

        fwd_kwargs: dict[str, Any] = {}
        for key in ("temperature", "max_tokens", "top_p", "stop"):
            if key in body:
                fwd_kwargs[key] = body[key]

        try:
            response = await forward_completion(
                provider=decision.backend,
                model_id=decision.model_id,
                messages=openai_messages,
                stream=False,
                credentials=self.credentials,
                **fwd_kwargs,
            )
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            self.selector.record_outcome(
                decision,
                RoutingOutcome(success=False, latency_ms=elapsed_ms, failure_kind="provider_error"),
            )
            message = scrub_keys(str(exc))
            return JSONResponse(
                {"type": "error", "error": {"type": "api_error", "message": message}},
                status_code=502,
            )

        elapsed_ms = int((time.monotonic() - start) * 1000)
        data = response.model_dump() if hasattr(response, "model_dump") else response
        usage = data.get("usage", {})

        # Convert OpenAI response back to Anthropic format
        content_text = ""
        choices = data.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            content_text = msg.get("content") or ""

        anthropic_response = {
            "id": data.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
            "type": "message",
            "role": "assistant",
            "model": decision.model_id,
            "content": [{"type": "text", "text": content_text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
        }

        self.selector.record_outcome(
            decision,
            RoutingOutcome(
                success=True,
                latency_ms=elapsed_ms,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            ),
        )

        msg_idx = self._session_msg_counters.get(session_id, 0)
        self._session_msg_counters[session_id] = msg_idx + 1
        self.grader.record_exchange(
            session_id=session_id,
            message_index=msg_idx,
            user_message=user_message,
            assistant_response=content_text[:2000],
            decision=decision,
            latency_ms=elapsed_ms,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )

        headers = {
            "x-routellect-model": decision.model_id,
            "x-routellect-routed": "true",
        }
        return JSONResponse(anthropic_response, headers=headers)

    async def _anthropic_passthrough(
        self,
        body: dict[str, Any],
        decision: RoutingDecision,
        start: float,
        stream: bool,
        session_id: str,
        user_message: str,
    ) -> Any:
        """Forward an Anthropic-format request directly to Anthropic's API.

        Preserves the exact request format — no translation needed.
        Just swaps the model and injects the real API key.
        """
        import httpx

        body["model"] = decision.model_id
        api_key = self.credentials.get("anthropic", "")

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        if stream:
            return await self._anthropic_stream(body, headers, decision, start, session_id, user_message)

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    json=body,
                    headers=headers,
                )
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            self.selector.record_outcome(
                decision,
                RoutingOutcome(success=False, latency_ms=elapsed_ms, failure_kind="provider_error"),
            )
            message = scrub_keys(str(exc))
            return JSONResponse(
                {"type": "error", "error": {"type": "api_error", "message": message}},
                status_code=502,
            )

        elapsed_ms = int((time.monotonic() - start) * 1000)
        data = resp.json()

        if resp.status_code >= 400:
            self.selector.record_outcome(
                decision,
                RoutingOutcome(success=False, latency_ms=elapsed_ms, failure_kind="provider_error"),
            )
            return JSONResponse(data, status_code=resp.status_code)

        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        self.selector.record_outcome(
            decision,
            RoutingOutcome(
                success=True,
                latency_ms=elapsed_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
        )

        # Extract assistant text for grading
        content_blocks = data.get("content", [])
        assistant_text = " ".join(
            b.get("text", "") for b in content_blocks if b.get("type") == "text"
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

        resp_headers = {
            "x-routellect-model": decision.model_id,
            "x-routellect-routed": "true",
        }
        return JSONResponse(data, headers=resp_headers)

    async def _anthropic_stream(
        self,
        body: dict[str, Any],
        headers: dict[str, str],
        decision: RoutingDecision,
        start: float,
        session_id: str,
        user_message: str,
    ) -> StreamingResponse:
        """Stream an Anthropic-format response."""
        import httpx

        collected_content: list[str] = []

        async def event_generator():
            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    async with client.stream(
                        "POST",
                        "https://api.anthropic.com/v1/messages",
                        json=body,
                        headers=headers,
                    ) as resp:
                        async for line in resp.aiter_lines():
                            if line.startswith("data: "):
                                try:
                                    chunk = json.loads(line[6:])
                                    if chunk.get("type") == "content_block_delta":
                                        delta_text = chunk.get("delta", {}).get("text", "")
                                        if delta_text:
                                            collected_content.append(delta_text)
                                except json.JSONDecodeError:
                                    pass
                            yield line + "\n"
            finally:
                elapsed_ms = int((time.monotonic() - start) * 1000)
                self.selector.record_outcome(
                    decision,
                    RoutingOutcome(success=True, latency_ms=elapsed_ms),
                )
                msg_idx = self._session_msg_counters.get(session_id, 0)
                self._session_msg_counters[session_id] = msg_idx + 1
                self.grader.record_exchange(
                    session_id=session_id,
                    message_index=msg_idx,
                    user_message=user_message,
                    assistant_response="".join(collected_content)[:2000],
                    decision=decision,
                    latency_ms=elapsed_ms,
                )

        resp_headers = {
            "x-routellect-model": decision.model_id,
            "x-routellect-routed": "true",
            "content-type": "text/event-stream",
        }
        return StreamingResponse(event_generator(), media_type="text/event-stream", headers=resp_headers)

    async def list_models(self, request: Request) -> JSONResponse:
        """GET /v1/models — list available models in OpenAI format."""
        model_list = []
        for m in self._models:
            model_list.append({
                "id": m.model_id,
                "object": "model",
                "created": 0,
                "owned_by": m.provider,
            })
        return JSONResponse({"object": "list", "data": model_list})

    async def health(self, request: Request) -> JSONResponse:
        """GET /health — proxy status."""
        providers = {}
        for m in self._models:
            providers.setdefault(m.provider, 0)
            providers[m.provider] += 1

        return JSONResponse({
            "status": "ok",
            "providers": providers,
            "total_models": len(self._models),
        })
