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
    parser.add_argument("--grades", action="store_true", help="Show recent grades and exit")
    parser.add_argument(
        "--export",
        type=str,
        nargs="?",
        const="routellect-export.zip",
        metavar="FILE",
        help="Export all grading data as a ZIP file (default: routellect-export.zip)",
    )
    args = parser.parse_args()

    # --grades: dump recent grades and exit
    if args.grades:
        _show_grades()
        raise SystemExit(0)

    # --export: write ZIP and exit
    if args.export:
        _run_export(args.export)
        raise SystemExit(0)

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


def _show_grades() -> None:
    """Print recent grades to stdout."""
    from routellect.proxy._grades_db import query_model_stats, query_recent_grades

    stats = query_model_stats()
    if stats:
        sys.stdout.write("\n  Model Performance Summary\n")
        sys.stdout.write("  " + "\u2500" * 60 + "\n")
        sys.stdout.write(f"  {'Model':<30} {'Grade':<8} {'Count':<8} {'Avg Conf':<10}\n")
        sys.stdout.write("  " + "\u2500" * 60 + "\n")
        for row in stats:
            sys.stdout.write(
                f"  {row['model_used']:<30} {row['grade']:<8} {row['count']:<8} "
                f"{row['avg_confidence']:.2f}\n"
            )
    else:
        sys.stdout.write("\n  No grades recorded yet.\n")

    recent = query_recent_grades(limit=20)
    if recent:
        sys.stdout.write(f"\n  Recent Grades (last {len(recent)})\n")
        sys.stdout.write("  " + "\u2500" * 70 + "\n")
        for g in recent:
            expl = " [exploration]" if g.get("is_exploration") else ""
            sys.stdout.write(
                f"  {g['graded_at'][:19]}  {g['model_used']:<25} "
                f"{g['grade']:<6} {g['confidence']:.1f}  {g['reason']}{expl}\n"
            )
    sys.stdout.write("\n")


def _run_export(output_file: str) -> None:
    """Export grading data to a ZIP file."""
    from pathlib import Path

    from routellect.proxy._grades_db import export_zip

    out_path = Path(output_file)
    export_zip(out_path)
    size_kb = out_path.stat().st_size / 1024
    sys.stderr.write(f"\n  Exported to {out_path.resolve()} ({size_kb:.1f} KB)\n")
    sys.stderr.write(f"  Contains: sessions.csv, grades.csv, routing_log.csv, model_summary.csv\n\n")
