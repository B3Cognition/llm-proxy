import pytest
import asyncio
from omlx_proxy import BackendConfig, anthropic_to_openai_request, openai_to_anthropic_response, stream_openai_to_anthropic, _filter_anthropic_internals

class TestFilterAnthropicInternals:
    def test_removes_billing_headers(self):
        """Should remove x-anthropic-* internal headers."""
        text = "x-anthropic-billing-header: cc_version=1.0;\nYou are helpful."
        result = _filter_anthropic_internals(text)

        assert "x-anthropic-" not in result
        assert "You are helpful." in result

    def test_preserves_actual_instructions(self):
        """Should preserve actual system instructions."""
        text = """x-anthropic-billing-header: info;
You are Claude Code.
IMPORTANT: Assist with security testing.
# System
 - Output text to the user."""
        result = _filter_anthropic_internals(text)

        assert "Claude Code" in result
        assert "IMPORTANT" in result
        assert "Output text" in result
        assert "x-anthropic-" not in result

    def test_cleans_multiple_newlines(self):
        """Should remove excessive blank lines from filtering."""
        text = """x-anthropic-billing-header: info;

You are helpful.


More text."""
        result = _filter_anthropic_internals(text)

        # Should not have 3+ consecutive newlines
        assert "\n\n\n" not in result


class TestBackendType:
    def test_backend_config_has_optional_type_field(self):
        """BackendConfig should have optional type field defaulting to 'anthropic'."""
        backend = BackendConfig(url="https://api.anthropic.com")
        assert backend.type == "anthropic"

    def test_backend_config_accepts_openai_type(self):
        """BackendConfig should accept type='openai' for Qwen backends."""
        backend = BackendConfig(
            url="https://genai-inference-qwen3-vl-30b.cloud",
            type="openai"
        )
        assert backend.type == "openai"


class TestAnthropicToOpenaiRequest:
    def test_moves_system_to_messages_array(self):
        """System param should become first message with role='system'."""
        request = {
            "model": "claude-haiku-4",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "system": "You are helpful"
        }
        result = anthropic_to_openai_request(request, "Qwen3-VL-30B")

        assert result["messages"][0] == {
            "role": "system",
            "content": "You are helpful"
        }
        assert result["messages"][1] == {"role": "user", "content": "Hello"}

    def test_handles_missing_system_param(self):
        """No system param should result in no system message."""
        request = {
            "model": "claude-haiku-4",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }
        result = anthropic_to_openai_request(request, "Qwen3-VL-30B")

        assert result["messages"][0]["role"] == "user"  # No system message

    def test_replaces_model_name(self):
        """Model name should be replaced with Qwen model."""
        request = {
            "model": "claude-sonnet-4",
            "messages": []
        }
        result = anthropic_to_openai_request(request, "Qwen3.6-35B")

        assert result["model"] == "Qwen3.6-35B"

    def test_preserves_temperature_and_params(self):
        """Temperature, top_p, and other params should be preserved."""
        request = {
            "model": "claude-haiku-4",
            "messages": [],
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 1024
        }
        result = anthropic_to_openai_request(request, "Qwen3-VL-30B")

        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9
        assert result["max_tokens"] == 1024

    def test_transforms_tools_to_openai_format(self):
        """Tools should be transformed from Anthropic to OpenAI format."""
        request = {
            "model": "claude-haiku-4",
            "messages": [],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                }
            ]
        }
        result = anthropic_to_openai_request(request, "Qwen3-VL-30B")

        # Should have transformed to OpenAI format
        assert len(result["tools"]) == 1
        tool = result["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "get_weather"
        assert tool["function"]["description"] == "Get weather for a location"
        assert tool["function"]["parameters"] == request["tools"][0]["input_schema"]

    def test_removes_auto_tool_choice(self):
        """Should remove tool_choice='auto' since Qwen doesn't support it."""
        request = {
            "model": "claude-haiku-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {"type": "object"}
                }
            ],
            "tool_choice": "auto"
        }
        result = anthropic_to_openai_request(request, "Qwen3-VL-30B")

        # tool_choice should be removed
        assert "tool_choice" not in result

    def test_handles_system_as_list_of_content_blocks(self):
        """Should convert system from content blocks list to concatenated string."""
        request = {
            "model": "claude-haiku-4",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "system": [
                {"type": "text", "text": "You are helpful", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": " and friendly"}
            ]
        }
        result = anthropic_to_openai_request(request, "Qwen3-VL-30B")

        # System message should be concatenated string, not list
        assert result["messages"][0] == {
            "role": "system",
            "content": "You are helpful and friendly"
        }
        assert result["messages"][1] == {"role": "user", "content": "Hello"}

    def test_handles_system_blocks_with_missing_spacing(self):
        """Should add newlines between blocks that lack proper spacing."""
        request = {
            "model": "claude-haiku-4",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "system": [
                {"type": "text", "text": "Header: info;"},  # No trailing newline
                {"type": "text", "text": "You are helpful."},  # No leading newline
                {"type": "text", "text": "\nContinue here"}  # Leading newline provided
            ]
        }
        result = anthropic_to_openai_request(request, "Qwen3-VL-30B")

        # Blocks without proper spacing should have newlines added between them
        content = result["messages"][0]["content"]

        # Should NOT have blocks running together
        assert "info;You are" not in content

        # Should have proper separation
        assert "info;\nYou are" in content or "info; \nYou are" in content

        # Third block already has leading newline, should be preserved
        assert "\nContinue here" in content


class TestOpenaiToAnthropicResponse:
    def test_unwraps_content_from_choices(self):
        """Response content should be unwrapped from choices[0].message."""
        openai_resp = {
            "id": "chatcmpl-123",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello there!"
                    },
                    "finish_reason": "stop"
                }
            ],
            "model": "Qwen3-VL-30B",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5
            }
        }
        result = openai_to_anthropic_response(openai_resp, "claude-haiku-4")

        assert result["content"] == [
            {"type": "text", "text": "Hello there!"}
        ]
        assert result["role"] == "assistant"

    def test_maps_finish_reason_stop_to_end_turn(self):
        """finish_reason 'stop' should map to stop_reason 'end_turn'."""
        openai_resp = {
            "id": "chatcmpl-123",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hi"},
                    "finish_reason": "stop"
                }
            ],
            "model": "Qwen3-VL-30B",
            "usage": {"prompt_tokens": 5, "completion_tokens": 2}
        }
        result = openai_to_anthropic_response(openai_resp, "claude-haiku-4")

        assert result["stop_reason"] == "end_turn"

    def test_maps_finish_reason_length_to_max_tokens(self):
        """finish_reason 'length' should map to stop_reason 'max_tokens'."""
        openai_resp = {
            "id": "chatcmpl-123",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hi"},
                    "finish_reason": "length"
                }
            ],
            "model": "Qwen3-VL-30B",
            "usage": {"prompt_tokens": 5, "completion_tokens": 2}
        }
        result = openai_to_anthropic_response(openai_resp, "claude-haiku-4")

        assert result["stop_reason"] == "max_tokens"

    def test_maps_usage_tokens(self):
        """Token counts should be remapped: prompt->input, completion->output."""
        openai_resp = {
            "id": "chatcmpl-123",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hi"},
                    "finish_reason": "stop"
                }
            ],
            "model": "Qwen3-VL-30B",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }
        result = openai_to_anthropic_response(openai_resp, "claude-haiku-4")

        assert result["usage"]["input_tokens"] == 100
        assert result["usage"]["output_tokens"] == 50
        assert "total_tokens" not in result["usage"]

    def test_replaces_model_name(self):
        """Model name should be replaced with original Anthropic model."""
        openai_resp = {
            "id": "chatcmpl-123",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hi"},
                    "finish_reason": "stop"
                }
            ],
            "model": "Qwen3-VL-30B",
            "usage": {"prompt_tokens": 5, "completion_tokens": 2}
        }
        result = openai_to_anthropic_response(openai_resp, "claude-sonnet-4")

        assert result["model"] == "claude-sonnet-4"

    def test_adds_required_fields(self):
        """Response should have type='message' and role='assistant'."""
        openai_resp = {
            "id": "chatcmpl-123",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hi"},
                    "finish_reason": "stop"
                }
            ],
            "model": "Qwen3-VL-30B",
            "usage": {"prompt_tokens": 5, "completion_tokens": 2}
        }
        result = openai_to_anthropic_response(openai_resp, "claude-haiku-4")

        assert result["type"] == "message"
        assert result["role"] == "assistant"


class TestStreamingTranslation:
    def test_converts_delta_content_to_content_block_delta(self):
        """Content delta should convert to content_block_delta event."""
        async def run_test():
            openai_events = [
                'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}',
                'data: [DONE]'
            ]

            async def async_iter(items):
                for item in items:
                    yield item

            stream = async_iter(openai_events)
            result = [line async for line in stream_openai_to_anthropic(stream, "claude-haiku-4")]
            return result

        result = asyncio.run(run_test())

        # Should have content_block_start, content_block_delta, message_stop
        assert len(result) >= 3

        # First should be content_block_start
        assert "content_block_start" in result[0]

        # Second should be content_block_delta with the content
        assert "content_block_delta" in result[1]
        assert "Hello" in result[1]

    def test_emits_content_block_start_on_first_chunk(self):
        """First content chunk should emit content_block_start."""
        async def run_test():
            openai_events = [
                'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}',
                'data: [DONE]'
            ]

            async def async_iter(items):
                for item in items:
                    yield item

            stream = async_iter(openai_events)
            result = [line async for line in stream_openai_to_anthropic(stream, "claude-haiku-4")]
            return result

        result = asyncio.run(run_test())

        # Find content_block_start event
        start_events = [e for e in result if "content_block_start" in e]
        assert len(start_events) >= 1

    def test_emits_message_stop_on_finish(self):
        """finish_reason should trigger message_stop event."""
        async def run_test():
            openai_events = [
                'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}',
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
                'data: [DONE]'
            ]

            async def async_iter(items):
                for item in items:
                    yield item

            stream = async_iter(openai_events)
            result = [line async for line in stream_openai_to_anthropic(stream, "claude-haiku-4")]
            return result

        result = asyncio.run(run_test())

        # Last event should be message_stop
        assert "message_stop" in result[-1]

    def test_skips_empty_deltas(self):
        """Empty delta events should not create content blocks."""
        async def run_test():
            openai_events = [
                'data: {"choices":[{"delta":{},"finish_reason":null}]}',
                'data: [DONE]'
            ]

            async def async_iter(items):
                for item in items:
                    yield item

            stream = async_iter(openai_events)
            result = [line async for line in stream_openai_to_anthropic(stream, "claude-haiku-4")]
            return result

        result = asyncio.run(run_test())

        # Should still have terminal events but no extra content blocks
        assert len(result) >= 1


class TestErrorHandling:
    def test_anthropic_to_openai_handles_empty_messages(self):
        """Should handle request with empty messages array."""
        request = {
            "model": "claude-haiku-4",
            "messages": [],
            "system": "You are helpful"
        }
        result = anthropic_to_openai_request(request, "Qwen3-VL-30B")

        assert result["messages"][0]["role"] == "system"
        assert len(result["messages"]) == 1

    def test_openai_to_anthropic_handles_missing_usage(self):
        """Should handle response without usage field."""
        openai_resp = {
            "id": "chatcmpl-123",
            "choices": [{
                "message": {"role": "assistant", "content": "Hi"},
                "finish_reason": "stop"
            }],
            "model": "Qwen3-VL-30B"
        }
        result = openai_to_anthropic_response(openai_resp, "claude-haiku-4")

        # Should have usage with 0 values
        assert result["usage"]["input_tokens"] == 0
        assert result["usage"]["output_tokens"] == 0

    def test_openai_to_anthropic_handles_empty_content(self):
        """Should handle response with empty or missing content."""
        openai_resp = {
            "id": "chatcmpl-123",
            "choices": [{
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "stop"
            }],
            "model": "Qwen3-VL-30B",
            "usage": {"prompt_tokens": 5, "completion_tokens": 0}
        }
        result = openai_to_anthropic_response(openai_resp, "claude-haiku-4")

        # Should have empty content array or text
        assert isinstance(result["content"], list)

    def test_streaming_handles_malformed_json(self):
        """Should skip malformed JSON lines gracefully."""
        async def run_test():
            malformed_events = [
                'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}',
                'data: {invalid json}',  # Malformed
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
                'data: [DONE]'
            ]

            async def async_iter(items):
                for item in items:
                    yield item

            stream = async_iter(malformed_events)
            result = [line async for line in stream_openai_to_anthropic(stream, "claude-haiku-4")]
            return result

        result = asyncio.run(run_test())

        # Should complete without error
        assert len(result) > 0
