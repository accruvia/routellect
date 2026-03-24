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
from routellect.proxy._circuit_breaker import CircuitBreaker
from routellect.proxy._grader import Grader
from routellect.proxy._middleware import scrub_keys
from routellect.proxy._provider_registry import build_model_universe
from routellect.proxy._translation import forward_completion

logger = logging.getLogger("routellect.proxy")

# Patterns that indicate a provider is down (not a transient error).
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


class _ProviderDownError(Exception):
    """Raised when a provider rejects a request due to billing, auth, or quota."""
    pass


def _is_provider_down(error_text: str) -> bool:
    """Check if an error indicates the provider is down (not transient)."""
    lower = error_text.lower()
    return any(p in lower for p in _PROVIDER_DOWN_PATTERNS)


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
        self.circuit_breaker = CircuitBreaker()
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

        user_message = self._get_last_user_message(body)

        # Use circuit breaker to skip known-down providers.
        down = self.circuit_breaker.get_down_providers()
        exclude = down if down else None

        for _attempt in range(3):
            decision: RoutingDecision = self.selector.select_model(
                fingerprint,
                constraints={"exclude_backends": exclude} if exclude else None,
            )

            start = time.monotonic()

            try:
                if decision.backend == "anthropic":
                    result = await self._anthropic_passthrough(
                        body, decision, start, stream, session_id, user_message,
                    )
                else:
                    result = await self._cross_provider_request(
                        body, decision, start, stream, session_id, user_message,
                        messages, system, fingerprint,
                    )
                self.circuit_breaker.record_success(decision.backend)
                return result
            except _ProviderDownError as exc:
                elapsed_ms = int((time.monotonic() - start) * 1000)
                self.selector.record_outcome(
                    decision,
                    RoutingOutcome(success=False, latency_ms=elapsed_ms, failure_kind="provider_down"),
                )
                self.circuit_breaker.record_failure(decision.backend)
                exclude = self.circuit_breaker.get_down_providers()
                logger.warning(
                    "Provider %s is down (%s), failing over...",
                    decision.backend, exc,
                )
                continue

        return JSONResponse(
            {"type": "error", "error": {"type": "api_error", "message": "All providers failed. Check API keys and billing."}},
            status_code=502,
        )

    async def _cross_provider_request(
        self,
        body: dict[str, Any],
        decision: RoutingDecision,
        start: float,
        stream: bool,
        session_id: str,
        user_message: str,
        messages: list,
        system: Any,
        fingerprint: dict,
    ) -> Any:
        """Route an Anthropic-format request to a non-Anthropic provider via litellm."""
        openai_messages = []
        if system:
            sys_text = system if isinstance(system, str) else json.dumps(system)
            openai_messages.append({"role": "system", "content": sys_text})
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
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
                stream=stream,
                credentials=self.credentials,
                **fwd_kwargs,
            )
        except Exception as exc:
            error_msg = str(exc)
            if _is_provider_down(error_msg):
                raise _ProviderDownError(f"{decision.backend}: {error_msg[:200]}")
            elapsed_ms = int((time.monotonic() - start) * 1000)
            self.selector.record_outcome(
                decision,
                RoutingOutcome(success=False, latency_ms=elapsed_ms, failure_kind="provider_error"),
            )
            message = scrub_keys(error_msg)
            return JSONResponse(
                {"type": "error", "error": {"type": "api_error", "message": message}},
                status_code=502,
            )

        if stream:
            return await self._cross_provider_stream(
                response, decision, start, session_id, user_message,
                requested_model=body.get("model", decision.model_id),
            )

        elapsed_ms = int((time.monotonic() - start) * 1000)
        data = response.model_dump() if hasattr(response, "model_dump") else response
        usage = data.get("usage", {})

        content_text = ""
        choices = data.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            content_text = msg.get("content") or ""

        requested_model = body.get("model", decision.model_id)
        anthropic_response = {
            "id": data.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
            "type": "message",
            "role": "assistant",
            "model": requested_model,
            "content": [{"type": "text", "text": content_text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
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

    async def _cross_provider_stream(
        self,
        response: Any,
        decision: RoutingDecision,
        start: float,
        session_id: str,
        user_message: str,
        requested_model: str = "",
    ) -> StreamingResponse:
        """Stream a cross-provider response, translating OpenAI SSE to Anthropic SSE."""
        collected_content: list[str] = []
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        input_tokens = 0
        output_tokens = 0
        # Report the model the client asked for, not the one we actually used
        report_model = requested_model or decision.model_id

        async def event_generator():
            nonlocal input_tokens, output_tokens
            # Emit Anthropic message_start — match Anthropic's exact format
            msg_start = {
                'type': 'message_start',
                'message': {
                    'id': msg_id,
                    'type': 'message',
                    'role': 'assistant',
                    'model': report_model,
                    'content': [],
                    'stop_reason': None,
                    'stop_sequence': None,
                    'usage': {
                        'input_tokens': 0,
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'output_tokens': 0,
                    },
                },
            }
            yield f"event: message_start\ndata: {json.dumps(msg_start)}\n\n"
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

            try:
                async for chunk in response:
                    data = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
                    choices = data.get("choices") or []
                    if choices:
                        delta = (choices[0].get("delta") or {})
                        text = delta.get("content") or ""
                        if text:
                            collected_content.append(text)
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text}})}\n\n"
                    usage = data.get("usage") or {}
                    if usage.get("prompt_tokens"):
                        input_tokens = usage["prompt_tokens"]
                    if usage.get("completion_tokens"):
                        output_tokens = usage["completion_tokens"]
            finally:
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}})}\n\n"
                yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

                elapsed_ms = int((time.monotonic() - start) * 1000)
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
                    assistant_response="".join(collected_content)[:2000],
                    decision=decision,
                    latency_ms=elapsed_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

        resp_headers = {
            "x-routellect-model": decision.model_id,
            "x-routellect-routed": "true",
            "content-type": "text/event-stream",
        }
        return StreamingResponse(event_generator(), media_type="text/event-stream", headers=resp_headers)

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
            error_msg = str(exc)
            if _is_provider_down(error_msg):
                raise _ProviderDownError(f"{decision.backend}: {error_msg[:200]}")
            elapsed_ms = int((time.monotonic() - start) * 1000)
            self.selector.record_outcome(
                decision,
                RoutingOutcome(success=False, latency_ms=elapsed_ms, failure_kind="provider_error"),
            )
            message = scrub_keys(error_msg)
            return JSONResponse(
                {"type": "error", "error": {"type": "api_error", "message": message}},
                status_code=502,
            )

        elapsed_ms = int((time.monotonic() - start) * 1000)
        data = resp.json()

        if resp.status_code >= 400:
            error_text = json.dumps(data)
            if _is_provider_down(error_text):
                raise _ProviderDownError(f"{decision.backend}: {error_text[:200]}")
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

        # If circuit breaker knows this provider is down, fail fast.
        if not self.circuit_breaker.is_available(decision.backend):
            raise _ProviderDownError(f"{decision.backend}: circuit breaker open")

        # Open the connection and check status BEFORE returning StreamingResponse.
        client = httpx.AsyncClient(timeout=120)
        req = client.build_request("POST", "https://api.anthropic.com/v1/messages",
                                    json=body, headers=headers)
        resp = await client.send(req, stream=True)

        if resp.status_code >= 400:
            error_body = await resp.aread()
            await resp.aclose()
            await client.aclose()
            error_text = error_body.decode(errors="replace")
            if _is_provider_down(error_text):
                raise _ProviderDownError(f"{decision.backend}: {error_text[:200]}")
            # Non-provider-down error — return it as JSON
            return JSONResponse(json.loads(error_text), status_code=resp.status_code)

        collected_content: list[str] = []
        stream_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

        async def event_generator():
            try:
                async for line in resp.aiter_lines():
                            if line.startswith("data: "):
                                try:
                                    chunk = json.loads(line[6:])
                                    if chunk.get("type") == "content_block_delta":
                                        delta_text = (chunk.get("delta") or {}).get("text", "")
                                        if delta_text:
                                            collected_content.append(delta_text)
                                    elif chunk.get("type") == "message_start":
                                        msg_usage = (chunk.get("message") or {}).get("usage") or {}
                                        stream_usage["input_tokens"] = msg_usage.get("input_tokens", 0)
                                    elif chunk.get("type") == "message_delta":
                                        delta_usage = chunk.get("usage") or {}
                                        stream_usage["output_tokens"] = delta_usage.get("output_tokens", 0)
                                except json.JSONDecodeError:
                                    pass
                            yield line + "\n"
            finally:
                elapsed_ms = int((time.monotonic() - start) * 1000)
                self.selector.record_outcome(
                    decision,
                    RoutingOutcome(
                        success=True,
                        latency_ms=elapsed_ms,
                        input_tokens=stream_usage["input_tokens"],
                        output_tokens=stream_usage["output_tokens"],
                    ),
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
                    input_tokens=stream_usage["input_tokens"],
                    output_tokens=stream_usage["output_tokens"],
                )
                await resp.aclose()
                await client.aclose()

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
        from routellect.proxy._grades_db import query_model_stats, query_recent_grades
        from routellect.proxy._provider_registry import get_model_tier, TIER_LABELS

        # Per-model stats from grades
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

        # Per-model call counts and latency from routing_log
        import sqlite3
        from routellect.proxy._grades_db import _get_db
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
                "model": model_id,
                "provider": provider,
                "tier": tier,
                "calls": cm["calls"],
                "pass": p,
                "fail": f,
                "mixed": gm.get("mixed", 0),
                "passRate": pass_rate,
                "avgLatency": cm["avg_latency"],
                "avgConfidence": avg_conf,
            })

        models.sort(key=lambda m: (m["tier"], -m["calls"]))

        # Selector state
        selector = self.selector
        sel_state = {
            "current_tier": getattr(selector, "current_tier", 1),
            "current_tier_label": TIER_LABELS.get(getattr(selector, "current_tier", 1), "unknown"),
            "trial_tier": getattr(selector, "trial_tier", None),
            "locked": getattr(selector, "locked", False),
        }

        recent = query_recent_grades(limit=20)

        # Ungraded queue: recent routing_log entries that have no matching grade
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
                   ORDER BY r.timestamp DESC
                   LIMIT 30"""
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
