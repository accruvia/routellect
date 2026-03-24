"""Tests for proxy configuration loading."""

from __future__ import annotations

from unittest.mock import patch

from routellect.proxy._config import ProxyConfig


class TestProxyConfig:
    def test_defaults(self):
        config = ProxyConfig()
        assert config.host == "127.0.0.1"
        assert config.port == 11411
        assert config.auth_token is None
        assert config.log_bodies is False

    def test_from_env(self):
        env = {
            "ROUTELLECT_PROXY_HOST": "0.0.0.0",
            "ROUTELLECT_PROXY_PORT": "9100",
            "ROUTELLECT_PROXY_TOKEN": "my-secret",
            "ROUTELLECT_LOG_BODIES": "true",
        }
        with patch.dict("os.environ", env, clear=False):
            config = ProxyConfig.from_env()
        assert config.host == "0.0.0.0"
        assert config.port == 9100
        assert config.auth_token == "my-secret"
        assert config.log_bodies is True

    def test_from_env_defaults(self):
        with patch.dict("os.environ", {}, clear=True):
            config = ProxyConfig.from_env()
        assert config.host == "127.0.0.1"
        assert config.port == 11411
        assert config.auth_token is None
        assert config.log_bodies is False
