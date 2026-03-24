"""Format detection, normalization, and response translation.

Pure functions — no I/O, no state, no side effects.  Every inbound request
is normalized to OpenAI chat-completion format for internal routing.  Every
outbound response (always OpenAI format from litellm) is translated back
to the original inbound format before returning to the client.

Supported formats:
    OPENAI    — /v1/chat/completions
    ANTHROPIC — /v1/messages
    GOOGLE    — /v1beta/models/{model}:generateContent
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class InboundFormat(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


def detect_format(path: str) -> InboundFormat:
    """Determine the inbound format from the request URL path."""
    if "/v1/messages" in path:
        return InboundFormat.ANTHROPIC
    if "generateContent" in path or "streamGenerateContent" in path:
        return InboundFormat.GOOGLE
    return InboundFormat.OPENAI


# ---------------------------------------------------------------------------
# Normalized request
# ---------------------------------------------------------------------------

@dataclass
class NormalizedRequest:
    """Provider-neutral representation of an LLM request."""

    messages: list[dict[str, Any]]
    model: str
    stream: bool
    params: dict[str, Any]
    original_model: str  # Echo back to client


# ---------------------------------------------------------------------------
# Inbound normalization  (any format → OpenAI messages)
# ---------------------------------------------------------------------------

def normalize_to_openai(body: dict[str, Any], fmt: InboundFormat) -> NormalizedRequest:
    """Convert any inbound request format to OpenAI chat-completion format."""
    if fmt == InboundFormat.OPENAI:
        return _normalize_openai(body)
    if fmt == InboundFormat.ANTHROPIC:
        return _normalize_anthropic(body)
    if fmt == InboundFormat.GOOGLE:
        return _normalize_google(body)
    raise ValueError(f"Unknown format: {fmt}")


def _normalize_openai(body: dict[str, Any]) -> NormalizedRequest:
    messages = body.get("messages", [])
    model = body.get("model", "")
    stream = body.get("stream", False)
    params: dict[str, Any] = {}
    for key in ("temperature", "max_tokens", "top_p", "stop", "tools", "tool_choice"):
        if key in body:
            params[key] = body[key]
    return NormalizedRequest(
        messages=messages,
        model=model,
        stream=stream,
        params=params,
        original_model=model,
    )


def _normalize_anthropic(body: dict[str, Any]) -> NormalizedRequest:
    messages: list[dict[str, Any]] = []
    system = body.get("system", "")
    if system:
        sys_text = system if isinstance(system, str) else json.dumps(system)
        messages.append({"role": "system", "content": sys_text})

    for msg in body.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
            content = "\n".join(parts)
        messages.append({"role": msg.get("role", "user"), "content": content})

    model = body.get("model", "")
    stream = body.get("stream", False)
    params: dict[str, Any] = {}
    for key in ("temperature", "max_tokens", "top_p", "stop"):
        if key in body:
            params[key] = body[key]

    return NormalizedRequest(
        messages=messages,
        model=model,
        stream=stream,
        params=params,
        original_model=model,
    )


def _normalize_google(body: dict[str, Any]) -> NormalizedRequest:
    messages: list[dict[str, Any]] = []

    # System instruction
    sys_inst = body.get("system_instruction") or body.get("systemInstruction")
    if sys_inst:
        parts = sys_inst.get("parts", [])
        sys_text = " ".join(p.get("text", "") for p in parts if isinstance(p, dict))
        if sys_text:
            messages.append({"role": "system", "content": sys_text})

    # Conversation contents
    for turn in body.get("contents", []):
        role = turn.get("role", "user")
        if role == "model":
            role = "assistant"
        parts = turn.get("parts", [])
        text = " ".join(p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p)
        messages.append({"role": role, "content": text})

    # Model from body or empty (will be overridden by selector)
    model = body.get("model", "")
    stream = body.get("stream", False)

    gen_config = body.get("generationConfig") or {}
    params: dict[str, Any] = {}
    if "temperature" in gen_config:
        params["temperature"] = gen_config["temperature"]
    if "maxOutputTokens" in gen_config:
        params["max_tokens"] = gen_config["maxOutputTokens"]
    if "topP" in gen_config:
        params["top_p"] = gen_config["topP"]
    if "stopSequences" in gen_config:
        params["stop"] = gen_config["stopSequences"]

    return NormalizedRequest(
        messages=messages,
        model=model,
        stream=stream,
        params=params,
        original_model=model,
    )


# ---------------------------------------------------------------------------
# Task fingerprint (works on normalized request)
# ---------------------------------------------------------------------------

def build_task_fingerprint(req: NormalizedRequest) -> dict[str, Any]:
    """Extract non-sensitive structural metadata for routing decisions."""
    return {
        "message_count": len(req.messages),
        "has_system_prompt": any(m.get("role") == "system" for m in req.messages),
        "has_tools": "tools" in req.params,
        "requested_model": req.model,
        "stream": req.stream,
        "max_tokens": req.params.get("max_tokens"),
    }


# ---------------------------------------------------------------------------
# User message extraction (for grading)
# ---------------------------------------------------------------------------

def extract_user_message(body: dict[str, Any], fmt: InboundFormat) -> str:
    """Get the last user message content for grading context."""
    if fmt == InboundFormat.GOOGLE:
        for turn in reversed(body.get("contents", [])):
            if turn.get("role", "user") == "user":
                parts = turn.get("parts", [])
                text = " ".join(p.get("text", "") for p in parts if isinstance(p, dict))
                return text[:2000]
        return ""

    # OpenAI and Anthropic both use messages[].role/content
    messages = body.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content[:2000]
            if isinstance(content, list):
                parts = [p.get("text", "") for p in content if isinstance(p, dict)]
                return " ".join(parts)[:2000]
    return ""


# ---------------------------------------------------------------------------
# Response translation  (OpenAI → original format)
# ---------------------------------------------------------------------------

def _sanitize_response(data: dict[str, Any]) -> dict[str, Any]:
    """Strip provider-specific fields and thinking/reasoning content.

    The caller should never see implementation details like thinking tokens,
    provider_specific_fields, or vertex_ai metadata.
    """
    # Remove top-level provider noise
    for key in list(data.keys()):
        if key.startswith("vertex_ai") or key in ("provider_specific_fields", "citations"):
            del data[key]

    # Clean up choices
    for choice in data.get("choices") or []:
        msg = choice.get("message") or {}
        delta = choice.get("delta") or {}

        # Remove provider fields from message/delta
        for obj in (msg, delta):
            for key in list(obj.keys()):
                if key in ("provider_specific_fields", "function_call", "audio"):
                    del obj[key]

        # Strip thinking/reasoning from usage details
        usage = data.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        reasoning_tokens = details.get("reasoning_tokens", 0)
        text_tokens = details.get("text_tokens", 0)

        # If model used reasoning tokens and produced no text, content may be None
        if msg.get("content") is None and reasoning_tokens > 0:
            msg["content"] = ""

    return data


def translate_response(
    data: dict[str, Any],
    fmt: InboundFormat,
    original_model: str,
) -> dict[str, Any]:
    """Translate an OpenAI-format response back to the inbound format."""
    data = _sanitize_response(data)
    if fmt == InboundFormat.OPENAI:
        return data
    if fmt == InboundFormat.ANTHROPIC:
        return _response_to_anthropic(data, original_model)
    if fmt == InboundFormat.GOOGLE:
        return _response_to_google(data, original_model)
    return data


def _response_to_anthropic(data: dict[str, Any], original_model: str) -> dict[str, Any]:
    usage = data.get("usage") or {}
    content_text = ""
    choices = data.get("choices") or []
    if choices:
        msg = choices[0].get("message") or {}
        content_text = msg.get("content") or ""

    return {
        "id": data.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
        "type": "message",
        "role": "assistant",
        "model": original_model,
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


def _response_to_google(data: dict[str, Any], original_model: str) -> dict[str, Any]:
    usage = data.get("usage") or {}
    content_text = ""
    choices = data.get("choices") or []
    if choices:
        msg = choices[0].get("message") or {}
        content_text = msg.get("content") or ""

    return {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": content_text}],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": usage.get("prompt_tokens", 0),
            "candidatesTokenCount": usage.get("completion_tokens", 0),
            "totalTokenCount": usage.get("total_tokens", 0),
        },
        "modelVersion": original_model,
    }


# ---------------------------------------------------------------------------
# Error response translation
# ---------------------------------------------------------------------------

def error_response(fmt: InboundFormat, message: str, status_code: int = 502) -> dict[str, Any]:
    """Build a format-appropriate error response body."""
    if fmt == InboundFormat.ANTHROPIC:
        return {"type": "error", "error": {"type": "api_error", "message": message}}
    if fmt == InboundFormat.GOOGLE:
        return {"error": {"code": status_code, "message": message, "status": "UNAVAILABLE"}}
    return {"error": {"message": message, "type": "proxy_error"}}


# ---------------------------------------------------------------------------
# Streaming translation  (OpenAI SSE → original format SSE)
# ---------------------------------------------------------------------------

@dataclass
class StreamState:
    """Mutable state accumulated during streaming."""

    started: bool = False
    msg_id: str = ""
    original_model: str = ""
    collected_content: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self):
        if not self.msg_id:
            self.msg_id = f"msg_{uuid.uuid4().hex[:24]}"


def stream_prologue(fmt: InboundFormat, state: StreamState) -> list[str]:
    """Lines to emit before the first content chunk."""
    if fmt == InboundFormat.ANTHROPIC:
        msg_start = {
            "type": "message_start",
            "message": {
                "id": state.msg_id,
                "type": "message",
                "role": "assistant",
                "model": state.original_model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 0,
                },
            },
        }
        block_start = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }
        return [
            f"event: message_start\ndata: {json.dumps(msg_start)}\n\n",
            f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n",
        ]
    # OpenAI and Google: no prologue needed
    return []


def translate_stream_chunk(
    chunk_data: dict[str, Any],
    fmt: InboundFormat,
    state: StreamState,
) -> list[str]:
    """Convert one OpenAI SSE chunk to the inbound format's SSE lines."""
    # Sanitize chunk — strip thinking tokens and provider fields
    chunk_data = _sanitize_response(chunk_data)

    lines: list[str] = []

    if not state.started:
        lines.extend(stream_prologue(fmt, state))
        state.started = True

    # Extract content delta and usage from OpenAI chunk
    choices = chunk_data.get("choices") or []
    text = ""
    if choices:
        delta = (choices[0].get("delta") or {})
        text = delta.get("content") or ""
    usage = chunk_data.get("usage") or {}
    if usage.get("prompt_tokens"):
        state.input_tokens = usage["prompt_tokens"]
    if usage.get("completion_tokens"):
        state.output_tokens = usage["completion_tokens"]

    if text:
        state.collected_content.append(text)

    if fmt == InboundFormat.OPENAI:
        lines.append(f"data: {json.dumps(chunk_data)}\n\n")

    elif fmt == InboundFormat.ANTHROPIC:
        if text:
            delta_event = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            }
            lines.append(f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n")

    elif fmt == InboundFormat.GOOGLE:
        if text:
            chunk_resp = {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": text}],
                        },
                    }
                ],
            }
            lines.append(f"data: {json.dumps(chunk_resp)}\r\n\r\n")

    return lines


def stream_epilogue(fmt: InboundFormat, state: StreamState) -> list[str]:
    """SSE lines to emit after the last content chunk."""
    if fmt == InboundFormat.OPENAI:
        return ["data: [DONE]\n\n"]

    if fmt == InboundFormat.ANTHROPIC:
        block_stop = {"type": "content_block_stop", "index": 0}
        msg_delta = {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": state.output_tokens},
        }
        msg_stop = {"type": "message_stop"}
        return [
            f"event: content_block_stop\ndata: {json.dumps(block_stop)}\n\n",
            f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n",
            f"event: message_stop\ndata: {json.dumps(msg_stop)}\n\n",
        ]

    if fmt == InboundFormat.GOOGLE:
        # Final SSE chunk with usage metadata — no content (already streamed)
        final = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": state.input_tokens,
                "candidatesTokenCount": state.output_tokens,
                "totalTokenCount": state.input_tokens + state.output_tokens,
            },
        }
        return [f"data: {json.dumps(final)}\r\n\r\n"]

    return []


# ---------------------------------------------------------------------------
# Helpers for extracting assistant text from responses
# ---------------------------------------------------------------------------

def extract_assistant_text(data: dict[str, Any]) -> str:
    """Extract assistant text from an OpenAI-format response."""
    choices = data.get("choices") or []
    if choices:
        msg = choices[0].get("message") or {}
        return str(msg.get("content") or "")
    return ""


def extract_usage(data: dict[str, Any]) -> tuple[int, int]:
    """Extract (input_tokens, output_tokens) from an OpenAI-format response."""
    usage = data.get("usage") or {}
    return usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
