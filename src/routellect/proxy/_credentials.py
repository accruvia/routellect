"""Encrypted credential storage for provider API keys.

Keys are stored in ~/.routellect/credentials using Fernet symmetric encryption.
The encryption key is derived from machine-specific identifiers via PBKDF2 so
that the credential file is not portable across machines and cannot be read by
a simple ``cat``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

_ROUTELLECT_DIR = Path.home() / ".routellect"
_CREDENTIALS_PATH = _ROUTELLECT_DIR / "credentials"
_FALLBACK_KEY_PATH = _ROUTELLECT_DIR / ".key"
_SALT = b"routellect-credential-store-v1"

# Well-known provider env vars and their canonical names.
PROVIDER_ENV_VARS: dict[str, list[str]] = {
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    "groq": ["GROQ_API_KEY"],
}


def _get_machine_id() -> bytes:
    """Return a machine-specific identifier, or generate a fallback key."""
    # Linux: /etc/machine-id
    machine_id_path = Path("/etc/machine-id")
    if machine_id_path.exists():
        return machine_id_path.read_text().strip().encode()

    # macOS: IOPlatformUUID via ioreg
    try:
        import subprocess

        result = subprocess.run(
            ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            if "IOPlatformUUID" in line:
                uuid = line.split('"')[-2]
                return uuid.encode()
    except (FileNotFoundError, subprocess.TimeoutExpired, IndexError):
        pass

    # Fallback: generate a persistent random key
    _ROUTELLECT_DIR.mkdir(parents=True, exist_ok=True)
    if _FALLBACK_KEY_PATH.exists():
        return _FALLBACK_KEY_PATH.read_bytes()
    key = os.urandom(32)
    _FALLBACK_KEY_PATH.write_bytes(key)
    _FALLBACK_KEY_PATH.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return key


def _derive_fernet_key() -> bytes:
    """Derive a Fernet key from machine ID + user ID."""
    machine_id = _get_machine_id()
    user_id = str(os.getuid()).encode()
    raw = hashlib.pbkdf2_hmac("sha256", machine_id + user_id, _SALT, 480_000)
    return base64.urlsafe_b64encode(raw)


def _get_fernet():
    """Return a Fernet instance using the derived key."""
    from cryptography.fernet import Fernet

    return Fernet(_derive_fernet_key())


def _ensure_dir() -> None:
    _ROUTELLECT_DIR.mkdir(parents=True, exist_ok=True)


def save_credentials(credentials: dict[str, str]) -> Path:
    """Encrypt and save provider credentials to disk.

    Args:
        credentials: Mapping of provider name to API key.

    Returns:
        Path to the credentials file.
    """
    _ensure_dir()
    fernet = _get_fernet()
    payload = json.dumps(credentials).encode()
    encrypted = fernet.encrypt(payload)
    _CREDENTIALS_PATH.write_bytes(encrypted)
    _CREDENTIALS_PATH.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return _CREDENTIALS_PATH


def load_credentials() -> dict[str, str]:
    """Load and decrypt provider credentials from disk.

    Returns:
        Mapping of provider name to API key, or empty dict if no file exists.
    """
    if not _CREDENTIALS_PATH.exists():
        return {}
    fernet = _get_fernet()
    encrypted = _CREDENTIALS_PATH.read_bytes()
    try:
        payload = fernet.decrypt(encrypted)
    except Exception:
        return {}
    try:
        data: Any = json.loads(payload)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def has_credentials() -> bool:
    """Return True if an encrypted credentials file exists."""
    return _CREDENTIALS_PATH.exists()


def delete_credentials() -> None:
    """Remove the credentials file if it exists."""
    if _CREDENTIALS_PATH.exists():
        _CREDENTIALS_PATH.unlink()
