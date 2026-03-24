"""Tests for encrypted credential storage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


class TestCredentialStore:
    def test_save_and_load_round_trip(self, tmp_path: Path):
        cred_path = tmp_path / "credentials"
        with (
            patch("routellect.proxy._credentials._ROUTELLECT_DIR", tmp_path),
            patch("routellect.proxy._credentials._CREDENTIALS_PATH", cred_path),
            patch("routellect.proxy._credentials._FALLBACK_KEY_PATH", tmp_path / ".key"),
        ):
            from routellect.proxy._credentials import load_credentials, save_credentials

            original = {"openai": "sk-test-123", "anthropic": "sk-ant-test-456"}
            save_credentials(original)
            loaded = load_credentials()
            assert loaded == original

    def test_load_missing_file_returns_empty(self, tmp_path: Path):
        cred_path = tmp_path / "credentials"
        with (
            patch("routellect.proxy._credentials._ROUTELLECT_DIR", tmp_path),
            patch("routellect.proxy._credentials._CREDENTIALS_PATH", cred_path),
            patch("routellect.proxy._credentials._FALLBACK_KEY_PATH", tmp_path / ".key"),
        ):
            from routellect.proxy._credentials import load_credentials

            assert load_credentials() == {}

    def test_has_credentials(self, tmp_path: Path):
        cred_path = tmp_path / "credentials"
        with (
            patch("routellect.proxy._credentials._ROUTELLECT_DIR", tmp_path),
            patch("routellect.proxy._credentials._CREDENTIALS_PATH", cred_path),
            patch("routellect.proxy._credentials._FALLBACK_KEY_PATH", tmp_path / ".key"),
        ):
            from routellect.proxy._credentials import has_credentials, save_credentials

            assert not has_credentials()
            save_credentials({"openai": "sk-test"})
            assert has_credentials()

    def test_delete_credentials(self, tmp_path: Path):
        cred_path = tmp_path / "credentials"
        with (
            patch("routellect.proxy._credentials._ROUTELLECT_DIR", tmp_path),
            patch("routellect.proxy._credentials._CREDENTIALS_PATH", cred_path),
            patch("routellect.proxy._credentials._FALLBACK_KEY_PATH", tmp_path / ".key"),
        ):
            from routellect.proxy._credentials import (
                delete_credentials,
                has_credentials,
                save_credentials,
            )

            save_credentials({"openai": "sk-test"})
            assert has_credentials()
            delete_credentials()
            assert not has_credentials()

    def test_file_permissions(self, tmp_path: Path):
        cred_path = tmp_path / "credentials"
        with (
            patch("routellect.proxy._credentials._ROUTELLECT_DIR", tmp_path),
            patch("routellect.proxy._credentials._CREDENTIALS_PATH", cred_path),
            patch("routellect.proxy._credentials._FALLBACK_KEY_PATH", tmp_path / ".key"),
        ):
            from routellect.proxy._credentials import save_credentials

            save_credentials({"openai": "sk-test"})
            mode = cred_path.stat().st_mode & 0o777
            assert mode == 0o600

    def test_corrupted_file_returns_empty(self, tmp_path: Path):
        cred_path = tmp_path / "credentials"
        with (
            patch("routellect.proxy._credentials._ROUTELLECT_DIR", tmp_path),
            patch("routellect.proxy._credentials._CREDENTIALS_PATH", cred_path),
            patch("routellect.proxy._credentials._FALLBACK_KEY_PATH", tmp_path / ".key"),
        ):
            from routellect.proxy._credentials import load_credentials

            cred_path.write_bytes(b"not encrypted data")
            assert load_credentials() == {}
