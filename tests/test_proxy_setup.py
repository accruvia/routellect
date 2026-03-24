"""Tests for the first-run setup wizard."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import pytest


class TestSetupWizard:
    def test_run_setup_saves_credentials(self, tmp_path: Path):
        cred_path = tmp_path / "credentials"
        patches = {
            "routellect.proxy._credentials._ROUTELLECT_DIR": tmp_path,
            "routellect.proxy._credentials._CREDENTIALS_PATH": cred_path,
            "routellect.proxy._credentials._FALLBACK_KEY_PATH": tmp_path / ".key",
            "routellect.proxy._setup.has_credentials": lambda: False,
        }
        # Simulate user entering one key and skipping others
        inputs = iter(["sk-test-openai-key-1234567890", "", "", ""])

        with (
            patch.multiple("routellect.proxy._credentials", **{
                "_ROUTELLECT_DIR": tmp_path,
                "_CREDENTIALS_PATH": cred_path,
                "_FALLBACK_KEY_PATH": tmp_path / ".key",
            }),
            patch("routellect.proxy._setup.has_credentials", return_value=False),
            patch("routellect.proxy._setup.getpass.getpass", side_effect=inputs),
            patch("routellect.proxy._setup._verify_key", return_value=(True, "verified")),
        ):
            from routellect.proxy._setup import run_setup

            out = io.StringIO()
            creds = run_setup(force=True, out=out)

        assert "openai" in creds
        assert creds["openai"] == "sk-test-openai-key-1234567890"
        output = out.getvalue()
        assert "verified" in output

    def test_run_setup_exits_on_no_keys(self, tmp_path: Path):
        cred_path = tmp_path / "credentials"
        # All empty inputs = skip all providers
        inputs = iter(["", "", "", ""])

        with (
            patch.multiple("routellect.proxy._credentials", **{
                "_ROUTELLECT_DIR": tmp_path,
                "_CREDENTIALS_PATH": cred_path,
                "_FALLBACK_KEY_PATH": tmp_path / ".key",
            }),
            patch("routellect.proxy._setup.has_credentials", return_value=False),
            patch("routellect.proxy._setup.getpass.getpass", side_effect=inputs),
        ):
            from routellect.proxy._setup import run_setup

            out = io.StringIO()
            with pytest.raises(SystemExit):
                run_setup(force=True, out=out)

    def test_verify_key_called_per_provider(self, tmp_path: Path):
        cred_path = tmp_path / "credentials"
        call_log = []

        def fake_verify(provider, key):
            call_log.append(provider)
            return True, "verified"

        inputs = iter(["key1", "key2", "", ""])

        with (
            patch.multiple("routellect.proxy._credentials", **{
                "_ROUTELLECT_DIR": tmp_path,
                "_CREDENTIALS_PATH": cred_path,
                "_FALLBACK_KEY_PATH": tmp_path / ".key",
            }),
            patch("routellect.proxy._setup.has_credentials", return_value=False),
            patch("routellect.proxy._setup.getpass.getpass", side_effect=inputs),
            patch("routellect.proxy._setup._verify_key", side_effect=fake_verify),
        ):
            from routellect.proxy._setup import run_setup

            out = io.StringIO()
            run_setup(force=True, out=out)

        assert "openai" in call_log
        assert "anthropic" in call_log
