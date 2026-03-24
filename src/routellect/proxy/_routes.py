"""Proxy route handlers: /v1/chat/completions, /v1/models, /health."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

from routellect.protocols import ModelSelectorProtocol, RoutingDecision, RoutingOutcome
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
    ) -> None:
        self.selector = selector
        self.credentials = credentials
        self._models = build_model_universe(credentials)
        self.selector.set_model_universe(self._models)

    async def chat_completions(self, request: Request) -> Any:
        """POST /v1/chat/completions — core proxy endpoint."""
        body = await request.json()
        fingerprint = _build_task_fingerprint(body)
        stream = body.get("stream", False)

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

        if stream:
            return await self._handle_stream(response, decision, start)
        else:
            return self._handle_sync(response, decision, start)

    async def _handle_stream(
        self,
        response: Any,
        decision: RoutingDecision,
        start: float,
    ) -> StreamingResponse:
        """Proxy an SSE streaming response."""
        total_chunks = 0

        async def event_generator():
            nonlocal total_chunks
            try:
                async for chunk in response:
                    total_chunks += 1
                    data = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
                    yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                elapsed_ms = int((time.monotonic() - start) * 1000)
                self.selector.record_outcome(
                    decision,
                    RoutingOutcome(success=True, latency_ms=elapsed_ms),
                )

        headers = {
            "x-routellect-model": decision.model_id,
            "x-routellect-routed": "true",
            "x-routellect-exploration": str(decision.is_exploration).lower(),
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
    ) -> JSONResponse:
        """Handle a non-streaming response."""
        elapsed_ms = int((time.monotonic() - start) * 1000)

        data = response.model_dump() if hasattr(response, "model_dump") else response
        usage = data.get("usage", {})
        self.selector.record_outcome(
            decision,
            RoutingOutcome(
                success=True,
                latency_ms=elapsed_ms,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            ),
        )

        headers = {
            "x-routellect-model": decision.model_id,
            "x-routellect-routed": "true",
            "x-routellect-exploration": str(decision.is_exploration).lower(),
        }
        return JSONResponse(data, headers=headers)

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
