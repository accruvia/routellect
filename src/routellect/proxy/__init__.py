"""Routellect LLM proxy — OpenAI-compatible model routing proxy.

Quick start::

    pip install routellect[proxy]
    python -m routellect.proxy
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from routellect.proxy._config import ProxyConfig

__all__ = ["ProxyConfig", "create_app", "serve", "main"]

logger = logging.getLogger("routellect.proxy")


def create_app(
    config: ProxyConfig | None = None,
    credentials: dict[str, str] | None = None,
) -> Any:
    """Create the Starlette ASGI application.

    See :func:`routellect.proxy._app.create_app` for details.
    """
    from routellect.proxy._app import create_app as _create_app

    return _create_app(config=config, credentials=credentials)


def serve(
    *,
    host: str | None = None,
    port: int | None = None,
    auth_token: str | None = None,
    config: ProxyConfig | None = None,
) -> None:
    """Start the proxy server (blocking).

    Args:
        host: Override bind address.
        port: Override bind port.
        auth_token: Override proxy auth token.
        config: Full config override.  If given, host/port/auth_token are ignored.
    """
    import uvicorn

    if config is None:
        config = ProxyConfig.from_env()
    if host is not None:
        config.host = host
    if port is not None:
        config.port = port
    if auth_token is not None:
        config.auth_token = auth_token

    if config.host != "127.0.0.1" and config.host != "localhost":
        logger.warning(
            "Proxy binding to %s — this exposes the proxy to the network. "
            "Consider using 127.0.0.1 (default) for local-only access.",
            config.host,
        )

    app = create_app(config=config)

    logger.info("Listening on http://%s:%d", config.host, config.port)
    sys.stderr.write(f"\n  \u2714 Proxy running on http://{config.host}:{config.port}\n")
    sys.stderr.write(f"\n  Add this to your app's environment:\n")
    sys.stderr.write(f"    OPENAI_BASE_URL=http://{config.host}:{config.port}/v1\n\n")
    sys.stderr.write(f"  That's it. All LLM calls will route through routellect.\n\n")

    uvicorn.run(app, host=config.host, port=config.port, log_level="warning")


def main() -> None:
    """Entry point for ``python -m routellect.proxy`` and ``routellect-proxy``."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="routellect-proxy",
        description="Routellect LLM proxy — transparent model routing",
    )
    parser.add_argument("--setup", action="store_true", help="Re-run the credential setup wizard")
    parser.add_argument("--host", type=str, default=None, help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Bind port (default: 11411)")
    args = parser.parse_args()

    from routellect.proxy._credentials import has_credentials

    if args.setup or not has_credentials():
        from routellect.proxy._setup import run_setup

        run_setup(force=args.setup)

    if not has_credentials():
        sys.stderr.write("No credentials configured. Exiting.\n")
        raise SystemExit(1)

    # Show provider summary on normal startup
    from routellect.proxy._credentials import load_credentials
    from routellect.proxy._provider_registry import build_model_universe

    creds = load_credentials()
    models = build_model_universe(creds)
    providers_summary = {}
    for m in models:
        providers_summary.setdefault(m.provider, 0)
        providers_summary[m.provider] += 1

    sys.stderr.write("\n  routellect proxy\n")
    sys.stderr.write("  " + "\u2500" * 16 + "\n")
    parts = [f"{p} ({n} models)" for p, n in providers_summary.items()]
    sys.stderr.write(f"  Providers: {', '.join(parts)}\n")

    serve(host=args.host, port=args.port)
