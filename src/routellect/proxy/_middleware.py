"""Starlette middleware for auth, key scrubbing, and logging control."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger("routellect.proxy")

# Regex patterns that look like API keys.  Used to scrub error messages.
_KEY_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),      # OpenAI
    re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),   # Anthropic
    re.compile(r"AIza[A-Za-z0-9_-]{30,}"),       # Google
    re.compile(r"gsk_[A-Za-z0-9_-]{20,}"),       # Groq
]


def scrub_keys(text: str) -> str:
    """Replace anything that looks like an API key with '***'."""
    for pattern in _KEY_PATTERNS:
        text = pattern.sub("***", text)
    return text


class AuthMiddleware(BaseHTTPMiddleware):
    """Enforce Bearer token auth when ROUTELLECT_PROXY_TOKEN is set."""

    def __init__(self, app: Any, auth_token: str | None = None) -> None:
        super().__init__(app)
        self.auth_token = auth_token

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        # /health is always public
        if request.url.path == "/health":
            return await call_next(request)

        if self.auth_token:
            auth_header = request.headers.get("authorization", "")
            if auth_header != f"Bearer {self.auth_token}":
                return JSONResponse(
                    {"error": {"message": "Unauthorized", "type": "auth_error"}},
                    status_code=401,
                )

        return await call_next(request)


class KeyScrubMiddleware(BaseHTTPMiddleware):
    """Catch unhandled exceptions and scrub API keys from error responses."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        try:
            return await call_next(request)
        except Exception as exc:
            message = scrub_keys(str(exc))
            logger.error("Unhandled error: %s", message)
            return JSONResponse(
                {"error": {"message": message, "type": "proxy_error"}},
                status_code=500,
            )


class RequestLogMiddleware(BaseHTTPMiddleware):
    """Log request method, path, status, and latency.  Never log bodies."""

    def __init__(self, app: Any, log_bodies: bool = False) -> None:
        super().__init__(app)
        self.log_bodies = log_bodies

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        start = time.monotonic()
        response = await call_next(request)
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "%s %s %d %.0fms",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response
