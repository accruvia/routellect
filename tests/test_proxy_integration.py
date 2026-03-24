"""Integration tests for the provider-agnostic proxy.

Tests all 3 inbound formats through the universal _route_completion handler
with mocked litellm to verify format normalization, routing, response
translation, circuit breaker, and grading work end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest

from routellect.protocols import RoutingDecision


class FakeSelector:
    def __init__(self):
        self.universe = []
        self.decisions = []
        self.outcomes = []
        self.locked = False
        self.current_tier = 1
        self.trial_tier = None

    def set_model_universe(self, models):
        self.universe = models

    def select_model(self, task_fingerprint, constraints=None):
        decision = RoutingDecision(
            model_id="gemini-2.5-pro",
            backend="google",
            confidence=0.8,
            reasoning="test",
        )
        self.decisions.append(decision)
        return decision

    def record_outcome(self, decision, outcome):
        self.outcomes.append((decision, outcome))


@dataclass
class FakeResponse:
    def model_dump(self):
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "gemini-2.5-pro",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "pong"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }


@pytest.fixture
def app():
    from routellect.proxy._app import create_app
    from routellect.proxy._config import ProxyConfig

    selector = FakeSelector()
    config = ProxyConfig(selector=selector)
    credentials = {"google": "test-key", "groq": "test-key"}
    return create_app(config=config, credentials=credentials), selector


class TestOpenAIFormat:
    @patch("routellect.proxy._routes.forward_completion", new_callable=AsyncMock)
    def test_non_streaming(self, mock_fwd, app):
        from starlette.testclient import TestClient

        mock_fwd.return_value = FakeResponse()
        application, selector = app
        client = TestClient(application)

        resp = client.post("/v1/chat/completions", json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "ping"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        assert data["choices"][0]["message"]["content"] == "pong"
        assert resp.headers.get("x-routellect-model") == "gemini-2.5-pro"
        assert resp.headers.get("x-routellect-routed") == "true"

        # Outcome recorded
        assert len(selector.outcomes) == 1
        _, outcome = selector.outcomes[0]
        assert outcome.success is True
        assert outcome.input_tokens == 10
        assert outcome.output_tokens == 5

    @patch("routellect.proxy._routes.forward_completion", new_callable=AsyncMock)
    def test_provider_error(self, mock_fwd, app):
        from starlette.testclient import TestClient

        mock_fwd.side_effect = RuntimeError("connection refused")
        application, _ = app
        client = TestClient(application)

        resp = client.post("/v1/chat/completions", json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "ping"}],
        })
        assert resp.status_code == 502
        assert "error" in resp.json()
        assert resp.json()["error"]["type"] == "proxy_error"


class TestAnthropicFormat:
    @patch("routellect.proxy._routes.forward_completion", new_callable=AsyncMock)
    def test_non_streaming(self, mock_fwd, app):
        from starlette.testclient import TestClient

        mock_fwd.return_value = FakeResponse()
        application, selector = app
        client = TestClient(application)

        resp = client.post("/v1/messages", json={
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 100,
        })
        assert resp.status_code == 200
        data = resp.json()
        # Response should be in Anthropic format
        assert data["type"] == "message"
        assert data["content"][0]["type"] == "text"
        assert data["content"][0]["text"] == "pong"
        assert data["model"] == "claude-sonnet-4-6"  # Original model echoed
        assert data["usage"]["input_tokens"] == 10
        assert data["usage"]["output_tokens"] == 5
        assert data["usage"]["cache_creation_input_tokens"] == 0
        assert data["stop_reason"] == "end_turn"

    @patch("routellect.proxy._routes.forward_completion", new_callable=AsyncMock)
    def test_with_system_prompt(self, mock_fwd, app):
        from starlette.testclient import TestClient

        mock_fwd.return_value = FakeResponse()
        application, _ = app
        client = TestClient(application)

        resp = client.post("/v1/messages", json={
            "model": "claude-sonnet-4-6",
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 100,
        })
        assert resp.status_code == 200

        # Verify the system prompt was passed to litellm
        call_kwargs = mock_fwd.call_args
        messages = call_kwargs.kwargs.get("messages", call_kwargs[1].get("messages", []))
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."

    @patch("routellect.proxy._routes.forward_completion", new_callable=AsyncMock)
    def test_content_blocks_normalized(self, mock_fwd, app):
        from starlette.testclient import TestClient

        mock_fwd.return_value = FakeResponse()
        application, _ = app
        client = TestClient(application)

        resp = client.post("/v1/messages", json={
            "model": "claude-sonnet-4-6",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": "World"},
                ],
            }],
            "max_tokens": 100,
        })
        assert resp.status_code == 200

        call_kwargs = mock_fwd.call_args
        messages = call_kwargs.kwargs.get("messages", call_kwargs[1].get("messages", []))
        assert messages[0]["content"] == "Hello\nWorld"

    @patch("routellect.proxy._routes.forward_completion", new_callable=AsyncMock)
    def test_provider_error_anthropic_format(self, mock_fwd, app):
        from starlette.testclient import TestClient

        mock_fwd.side_effect = RuntimeError("connection refused")
        application, _ = app
        client = TestClient(application)

        resp = client.post("/v1/messages", json={
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 100,
        })
        assert resp.status_code == 502
        data = resp.json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "api_error"


class TestGoogleFormat:
    @patch("routellect.proxy._routes.forward_completion", new_callable=AsyncMock)
    def test_non_streaming(self, mock_fwd, app):
        from starlette.testclient import TestClient

        mock_fwd.return_value = FakeResponse()
        application, selector = app
        client = TestClient(application)

        resp = client.post("/v1beta/models/gemini-2.5-pro:generateContent", json={
            "contents": [{"role": "user", "parts": [{"text": "ping"}]}],
            "generationConfig": {"maxOutputTokens": 100},
        })
        assert resp.status_code == 200
        data = resp.json()
        # Response should be in Google format
        assert "candidates" in data
        assert data["candidates"][0]["content"]["role"] == "model"
        assert data["candidates"][0]["content"]["parts"][0]["text"] == "pong"
        assert data["candidates"][0]["finishReason"] == "STOP"
        assert data["usageMetadata"]["promptTokenCount"] == 10
        assert data["usageMetadata"]["candidatesTokenCount"] == 5

    @patch("routellect.proxy._routes.forward_completion", new_callable=AsyncMock)
    def test_system_instruction(self, mock_fwd, app):
        from starlette.testclient import TestClient

        mock_fwd.return_value = FakeResponse()
        application, _ = app
        client = TestClient(application)

        resp = client.post("/v1beta/models/gemini-2.5-pro:generateContent", json={
            "system_instruction": {"parts": [{"text": "Be concise."}]},
            "contents": [{"role": "user", "parts": [{"text": "ping"}]}],
        })
        assert resp.status_code == 200

        call_kwargs = mock_fwd.call_args
        messages = call_kwargs.kwargs.get("messages", call_kwargs[1].get("messages", []))
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be concise."

    @patch("routellect.proxy._routes.forward_completion", new_callable=AsyncMock)
    def test_model_role_mapping(self, mock_fwd, app):
        from starlette.testclient import TestClient

        mock_fwd.return_value = FakeResponse()
        application, _ = app
        client = TestClient(application)

        resp = client.post("/v1beta/models/gemini-2.5-pro:generateContent", json={
            "contents": [
                {"role": "user", "parts": [{"text": "hi"}]},
                {"role": "model", "parts": [{"text": "hello"}]},
                {"role": "user", "parts": [{"text": "bye"}]},
            ],
        })
        assert resp.status_code == 200

        call_kwargs = mock_fwd.call_args
        messages = call_kwargs.kwargs.get("messages", call_kwargs[1].get("messages", []))
        assert messages[1]["role"] == "assistant"

    @patch("routellect.proxy._routes.forward_completion", new_callable=AsyncMock)
    def test_generation_config_mapping(self, mock_fwd, app):
        from starlette.testclient import TestClient

        mock_fwd.return_value = FakeResponse()
        application, _ = app
        client = TestClient(application)

        resp = client.post("/v1beta/models/gemini-2.5-pro:generateContent", json={
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "generationConfig": {
                "temperature": 0.5,
                "maxOutputTokens": 200,
                "topP": 0.9,
                "stopSequences": ["END"],
            },
        })
        assert resp.status_code == 200

        call_kwargs = mock_fwd.call_args
        assert call_kwargs.kwargs.get("temperature") == 0.5 or call_kwargs[1].get("temperature") == 0.5

    @patch("routellect.proxy._routes.forward_completion", new_callable=AsyncMock)
    def test_provider_error_google_format(self, mock_fwd, app):
        from starlette.testclient import TestClient

        mock_fwd.side_effect = RuntimeError("connection refused")
        application, _ = app
        client = TestClient(application)

        resp = client.post("/v1beta/models/gemini-2.5-pro:generateContent", json={
            "contents": [{"role": "user", "parts": [{"text": "ping"}]}],
        })
        assert resp.status_code == 502
        data = resp.json()
        assert data["error"]["code"] == 502
        assert data["error"]["status"] == "UNAVAILABLE"


class TestCircuitBreakerAllFormats:
    """Circuit breaker should work identically for all formats."""

    @patch("routellect.proxy._routes.forward_completion", new_callable=AsyncMock)
    def test_openai_billing_failover(self, mock_fwd, app):
        from starlette.testclient import TestClient

        mock_fwd.side_effect = [
            RuntimeError("credit balance is too low"),
            FakeResponse(),
        ]
        application, selector = app
        # Need 2 decisions for the retry
        selector.select_model = lambda fp, constraints=None: RoutingDecision(
            model_id="gemini-2.5-pro", backend="google", confidence=0.8, reasoning="fallback"
        )
        client = TestClient(application)

        resp = client.post("/v1/chat/completions", json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "ping"}],
        })
        assert resp.status_code == 200

    @patch("routellect.proxy._routes.forward_completion", new_callable=AsyncMock)
    def test_anthropic_billing_failover(self, mock_fwd, app):
        from starlette.testclient import TestClient

        mock_fwd.side_effect = [
            RuntimeError("credit balance is too low"),
            FakeResponse(),
        ]
        application, selector = app
        selector.select_model = lambda fp, constraints=None: RoutingDecision(
            model_id="gemini-2.5-pro", backend="google", confidence=0.8, reasoning="fallback"
        )
        client = TestClient(application)

        resp = client.post("/v1/messages", json={
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 100,
        })
        assert resp.status_code == 200
        assert resp.json()["type"] == "message"

    @patch("routellect.proxy._routes.forward_completion", new_callable=AsyncMock)
    def test_google_billing_failover(self, mock_fwd, app):
        from starlette.testclient import TestClient

        mock_fwd.side_effect = [
            RuntimeError("quota exceeded"),
            FakeResponse(),
        ]
        application, selector = app
        selector.select_model = lambda fp, constraints=None: RoutingDecision(
            model_id="gemini-2.5-pro", backend="google", confidence=0.8, reasoning="fallback"
        )
        client = TestClient(application)

        resp = client.post("/v1beta/models/gemini-2.5-pro:generateContent", json={
            "contents": [{"role": "user", "parts": [{"text": "ping"}]}],
        })
        assert resp.status_code == 200
        assert "candidates" in resp.json()


class TestDashboard:
    def test_health(self, app):
        from starlette.testclient import TestClient

        application, _ = app
        client = TestClient(application)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_dashboard_html(self, app):
        from starlette.testclient import TestClient

        application, _ = app
        client = TestClient(application)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "routellect" in resp.text

    def test_stats_api(self, app):
        from starlette.testclient import TestClient

        application, _ = app
        client = TestClient(application)
        resp = client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert "selector" in data
        assert "ungraded_queue" in data

    def test_models_list(self, app):
        from starlette.testclient import TestClient

        application, _ = app
        client = TestClient(application)
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        assert resp.json()["object"] == "list"


class TestAuthMiddleware:
    def test_auth_required(self):
        from starlette.testclient import TestClient

        from routellect.proxy._app import create_app
        from routellect.proxy._config import ProxyConfig

        config = ProxyConfig(auth_token="secret-123", selector=FakeSelector())
        application = create_app(config=config, credentials={"google": "test"})
        client = TestClient(application)

        assert client.get("/v1/models").status_code == 401
        assert client.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.get("/v1/models", headers={"Authorization": "Bearer secret-123"}).status_code == 200
        assert client.get("/health").status_code == 200  # Always public
