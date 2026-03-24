"""Starlette ASGI application assembly."""

from __future__ import annotations

from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.routing import Route

from routellect.proxy._config import ProxyConfig
from routellect.proxy._middleware import AuthMiddleware, KeyScrubMiddleware, RequestLogMiddleware
from routellect.proxy._routes import ProxyRoutes
from routellect.proxy._selector import GraduatedDemotionSelector

_STATIC_DIR = Path(__file__).parent / "static"


async def _dashboard(request: Request) -> HTMLResponse:
    html = (_STATIC_DIR / "dashboard.html").read_text()
    return HTMLResponse(html)


def create_app(
    config: ProxyConfig | None = None,
    credentials: dict[str, str] | None = None,
) -> Starlette:
    """Build the Starlette ASGI application.

    Args:
        config: Proxy configuration.  Defaults to ``ProxyConfig.from_env()``.
        credentials: Provider credentials.  If None, loads from encrypted store.

    Returns:
        Configured Starlette app ready to serve.
    """
    if config is None:
        config = ProxyConfig.from_env()

    if credentials is None:
        from routellect.proxy._credentials import load_credentials

        credentials = load_credentials()

    selector = config.selector or GraduatedDemotionSelector()
    routes = ProxyRoutes(selector=selector, credentials=credentials)

    app = Starlette(
        routes=[
            Route("/", _dashboard, methods=["GET"]),
            Route("/dashboard", _dashboard, methods=["GET"]),
            Route("/api/stats", routes.api_stats, methods=["GET"]),
            Route("/api/selector/toggle", routes.api_selector_toggle, methods=["POST"]),
            Route("/api/provider/reenable", routes.api_provider_reenable, methods=["POST"]),
            Route("/v1/chat/completions", routes.chat_completions, methods=["POST"]),
            Route("/v1/messages", routes.anthropic_messages, methods=["POST"]),
            Route("/v1beta/models/{model_path:path}", routes.google_generate, methods=["POST"]),
            Route("/v1/models", routes.list_models, methods=["GET"]),
            Route("/health", routes.health, methods=["GET"]),
        ],
    )

    # Middleware is applied in reverse order (outermost first).
    app.add_middleware(RequestLogMiddleware, log_bodies=config.log_bodies)
    app.add_middleware(KeyScrubMiddleware)
    app.add_middleware(AuthMiddleware, auth_token=config.auth_token)

    return app
