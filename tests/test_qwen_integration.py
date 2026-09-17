import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from omlx_proxy import app, load_config
import omlx_proxy


class TestQwenBackendIntegration:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up ACTIVE_CONFIG with datacenter profile."""
        omlx_proxy.ACTIVE_CONFIG = load_config("datacenter")

    def test_route_messages_detects_qwen_backend(self):
        """resolve_route() should return Qwen URL for haiku in datacenter profile."""
        from omlx_proxy import resolve_route

        target, model = resolve_route("claude-haiku-4")

        assert "vllm-service.example.com" in target
        assert model == "Qwen/Qwen3.6-35B-A3B"

    def test_route_messages_sends_to_qwen_endpoint(self):
        """route_messages() should call /v1/chat/completions for Qwen backend."""
        client = TestClient(app)

        with patch("httpx.AsyncClient.post") as mock_post:
            # Mock Qwen response
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_response.content = json.dumps({
                "id": "chatcmpl-123",
                "choices": [{
                    "message": {"role": "assistant", "content": "Hello"},
                    "finish_reason": "stop"
                }],
                "model": "Qwen3-VL-30B",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}
            }).encode()
            mock_post.return_value = mock_response

            response = client.post(
                "/v1/messages",
                json={
                    "model": "claude-haiku-4",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "system": "You are helpful"
                }
            )

            # Check that /v1/chat/completions was called (not /v1/messages)
            call_args = mock_post.call_args
            assert "/v1/chat/completions" in call_args[0][0]

    def test_route_messages_translates_request_for_qwen(self):
        """Request should be translated to OpenAI format for Qwen."""
        client = TestClient(app)

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_response.content = json.dumps({
                "id": "chatcmpl-123",
                "choices": [{"message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
                "model": "Qwen3-VL-30B",
                "usage": {"prompt_tokens": 5, "completion_tokens": 2}
            }).encode()
            mock_post.return_value = mock_response

            response = client.post(
                "/v1/messages",
                json={
                    "model": "claude-haiku-4",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "system": "You are helpful"
                }
            )

            # Check that request body was translated
            call_args = mock_post.call_args
            sent_body = json.loads(call_args[1]["content"])

            # System should be in messages array, not top-level
            assert sent_body["messages"][0]["role"] == "system"
            assert sent_body["messages"][0]["content"] == "You are helpful"
            # Model should be Qwen model
            assert sent_body["model"] == "Qwen/Qwen3.6-35B-A3B"

    def test_route_messages_translates_response_from_qwen(self):
        """Response should be translated back to Anthropic format."""
        client = TestClient(app)

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_response.content = json.dumps({
                "id": "chatcmpl-123",
                "choices": [{
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop"
                }],
                "model": "Qwen3-VL-30B",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}
            }).encode()
            mock_post.return_value = mock_response

            response = client.post(
                "/v1/messages",
                json={
                    "model": "claude-haiku-4",
                    "messages": [{"role": "user", "content": "Hi"}]
                }
            )

            # Response should be in Anthropic format
            data = response.json()
            assert data["type"] == "message"
            assert data["role"] == "assistant"
            assert data["content"][0]["type"] == "text"
            assert data["content"][0]["text"] == "Hello!"
            assert data["stop_reason"] == "end_turn"
            assert data["model"] == "claude-haiku-4"  # Original model name

    def test_anthropic_backend_unchanged(self):
        """Anthropic backend should bypass translation (regression test)."""
        config = load_config("local")
        anthropic_backend = config.backends["anthropic"]

        # Should not have type="openai"
        assert anthropic_backend.type != "openai"

    def test_datacenter_profile_routing(self):
        """Datacenter profile should route Haiku to Qwen."""
        config = load_config("datacenter")
        assert config.routes["haiku"].backend == "qwen_vl"
        assert config.backends["qwen_vl"].type == "openai"

    def test_hybrid_profile_routing(self):
        """Hybrid profile should route Sonnet to Qwen."""
        config = load_config("hybrid")
        assert config.routes["sonnet"].backend == "qwen_vl"
        assert config.backends["qwen_vl"].type == "openai"
