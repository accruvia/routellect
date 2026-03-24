"""Interactive first-run setup wizard.

Prompts for provider API keys, verifies each with a lightweight API call,
then saves to the encrypted credential store.
"""

from __future__ import annotations

import getpass
import sys
from typing import TextIO

import httpx

from routellect.proxy._credentials import PROVIDER_ENV_VARS, has_credentials, save_credentials

# Lightweight verification endpoints per provider.
_VERIFY_CONFIG: dict[str, dict] = {
    "openai": {
        "url": "https://api.openai.com/v1/models",
        "method": "GET",
        "headers_fn": lambda key: {"Authorization": f"Bearer {key}"},
    },
    "anthropic": {
        "url": "https://api.anthropic.com/v1/messages",
        "method": "POST",
        "headers_fn": lambda key: {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        "body": {"model": "claude-haiku-4-5-20251001", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]},
    },
    "google": {
        "url": "https://generativelanguage.googleapis.com/v1beta/models",
        "method": "GET",
        "headers_fn": lambda key: {},
        "params_fn": lambda key: {"key": key},
    },
    "groq": {
        "url": "https://api.groq.com/openai/v1/models",
        "method": "GET",
        "headers_fn": lambda key: {"Authorization": f"Bearer {key}"},
    },
}


def _verify_key(provider: str, key: str) -> tuple[bool, str]:
    """Verify an API key with a lightweight call. Returns (ok, message)."""
    cfg = _VERIFY_CONFIG.get(provider)
    if cfg is None:
        return True, "unverified (no check available)"

    headers = cfg["headers_fn"](key)
    params = cfg.get("params_fn", lambda _: {})(key)
    try:
        if cfg["method"] == "GET":
            resp = httpx.get(cfg["url"], headers=headers, params=params, timeout=15)
        else:
            resp = httpx.post(cfg["url"], headers=headers, json=cfg.get("body"), params=params, timeout=15)

        if resp.status_code < 400 or (provider == "anthropic" and resp.status_code == 400):
            # Anthropic returns 400 for our minimal payload but that means auth succeeded
            return True, "verified"
        if resp.status_code in (401, 403):
            return False, "authentication failed — check your key"
        return False, f"unexpected status {resp.status_code}"
    except httpx.TimeoutException:
        return False, "request timed out"
    except httpx.HTTPError as exc:
        return False, f"connection error: {exc}"


def _prompt_key(provider: str, env_vars: list[str], out: TextIO = sys.stderr) -> str | None:
    """Prompt for a single provider key with masked input."""
    label = provider.capitalize()
    hint = env_vars[0] if env_vars else ""
    out.write(f"\n  {label} API key ({hint}): ")
    out.flush()

    try:
        key = getpass.getpass(prompt="")
    except (EOFError, KeyboardInterrupt):
        out.write("\n")
        return None

    return key.strip() or None


def run_setup(force: bool = False, out: TextIO = sys.stderr) -> dict[str, str]:
    """Run the interactive setup wizard.

    Args:
        force: Run even if credentials already exist.
        out: Output stream for prompts and status messages.

    Returns:
        Dict of provider -> api_key for all configured providers.
    """
    if has_credentials() and not force:
        out.write("  Credentials already configured. Use --setup to reconfigure.\n")
        from routellect.proxy._credentials import load_credentials

        return load_credentials()

    out.write("\n  routellect proxy — first-time setup\n")
    out.write("  " + "\u2500" * 36 + "\n")
    out.write("\n  Paste your API keys below (Enter to skip a provider):\n")

    credentials: dict[str, str] = {}

    for provider, env_vars in PROVIDER_ENV_VARS.items():
        key = _prompt_key(provider, env_vars, out=out)
        if key is None:
            out.write(f"    (skipped)\n")
            continue

        ok, msg = _verify_key(provider, key)
        if ok:
            credentials[provider] = key
            out.write(f"    \u2714 {msg}\n")
        else:
            out.write(f"    \u2718 {msg}\n")
            out.write(f"    Try again? (Enter to skip): ")
            out.flush()
            try:
                retry_key = getpass.getpass(prompt="")
            except (EOFError, KeyboardInterrupt):
                out.write("\n")
                continue
            retry_key = retry_key.strip()
            if retry_key:
                ok2, msg2 = _verify_key(provider, retry_key)
                if ok2:
                    credentials[provider] = retry_key
                    out.write(f"    \u2714 {msg2}\n")
                else:
                    out.write(f"    \u2718 {msg2} — skipping {provider}\n")

    if not credentials:
        out.write("\n  No API keys provided. At least one provider is required.\n")
        out.write("  Run again with: python -m routellect.proxy --setup\n\n")
        raise SystemExit(1)

    path = save_credentials(credentials)
    providers = ", ".join(credentials.keys())
    out.write(f"\n  Keys encrypted and saved to {path}\n")
    out.write(f"  Providers configured: {providers}\n")

    return credentials
