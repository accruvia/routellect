"""Tests for proxy route handlers."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest

from routellect.protocols import ModelCapability, RoutingDecision, RoutingOutcome


class FakeSelector:
    """Test double for ModelSelectorProtocol."""

    def __init__(self):
        self.universe = []
        self.decisions = []
        self.outcomes = []

    def set_model_universe(self, models):
        self.universe = models

    def select_model(self, task_fingerprint, constraints=None):
        decision = RoutingDecision(
            model_id="gpt-4o",
            backend="openai",
            confidence=0.8,
            reasoning="test",
        )
        self.decisions.append(decision)
        return decision

    def record_outcome(self, decision, outcome):
        self.outcomes.append((decision, outcome))


@dataclass
class FakeLitellmResponse:
    """Mimics a litellm ModelResponse for non-streaming."""

    def model_dump(self):
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }


class TestProxyRoutes:
    @pytest.fixture
    def app(self):
        from routellect.proxy._app import create_app
        from routellect.proxy._config import ProxyConfig

        selector = FakeSelector()
        config = ProxyConfig(selector=selector)
        credentials = {"openai": "sk-test"}
        return create_app(config=config, credentials=credentials), selector

    def test_health_endpoint(self, app):
        from starlette.testclient import TestClient

        application, _ = app
        client = TestClient(application)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["total_models"] > 0

    def test_list_models(self, app):
        from starlette.testclient import TestClient

        application, _ = app
        client = TestClient(application)
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0
        assert all(m["object"] == "model" for m in data["data"])

    @patch("routellect.proxy._routes.forward_completion", new_callable=AsyncMock)
    def test_chat_completions_non_streaming(self, mock_fwd, app):
        from starlette.testclient import TestClient

        mock_fwd.return_value = FakeLitellmResponse()
        application, selector = app
        client = TestClient(application)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Hello!"
        assert resp.headers.get("x-routellect-model") == "gpt-4o"
        assert resp.headers.get("x-routellect-routed") == "true"

        # Selector recorded outcome
        assert len(selector.outcomes) == 1
        _, outcome = selector.outcomes[0]
        assert outcome.success is True
        assert outcome.input_tokens == 10
        assert outcome.output_tokens == 5

    @patch("routellect.proxy._routes.forward_completion", new_callable=AsyncMock)
    def test_chat_completions_provider_error(self, mock_fwd, app):
        from starlette.testclient import TestClient

        mock_fwd.side_effect = RuntimeError("connection refused")
        application, selector = app
        client = TestClient(application)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert resp.status_code == 502
        assert "connection refused" in resp.json()["error"]["message"]

        # Outcome recorded as failure
        assert len(selector.outcomes) == 1
        _, outcome = selector.outcomes[0]
        assert outcome.success is False

    def test_auth_required_when_token_set(self):
        from starlette.testclient import TestClient

        from routellect.proxy._app import create_app
        from routellect.proxy._config import ProxyConfig

        config = ProxyConfig(auth_token="secret-123", selector=FakeSelector())
        application = create_app(config=config, credentials={"openai": "sk-test"})
        client = TestClient(application)

        # No auth header
        resp = client.get("/v1/models")
        assert resp.status_code == 401

        # Wrong token
        resp = client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 401

        # Correct token
        resp = client.get("/v1/models", headers={"Authorization": "Bearer secret-123"})
        assert resp.status_code == 200

        # /health is always public
        resp = client.get("/health")
        assert resp.status_code == 200


class TestKeyScrubbing:
    def test_scrub_openai_key(self):
        from routellect.proxy._middleware import scrub_keys

        text = "Error with key sk-1234567890abcdefghijklmnop"
        assert "sk-1234567890" not in scrub_keys(text)
        assert "***" in scrub_keys(text)

    def test_scrub_anthropic_key(self):
        from routellect.proxy._middleware import scrub_keys

        text = "Auth failed: sk-ant-abcdef1234567890abcdefgh"
        assert "sk-ant-" not in scrub_keys(text)

    def test_no_scrub_when_clean(self):
        from routellect.proxy._middleware import scrub_keys

        text = "Normal error message"
        assert scrub_keys(text) == text


class TestTaskFingerprint:
    def test_fingerprint_does_not_include_content(self):
        from routellect.proxy._routes import _build_task_fingerprint

        body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "SECRET SYSTEM PROMPT"},
                {"role": "user", "content": "SECRET USER MESSAGE"},
            ],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            "stream": True,
        }
        fp = _build_task_fingerprint(body)
        fp_str = str(fp)
        assert "SECRET" not in fp_str
        assert fp["message_count"] == 2
        assert fp["has_system_prompt"] is True
        assert fp["has_tools"] is True
        assert fp["stream"] is True
