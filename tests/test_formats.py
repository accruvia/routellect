"""Tests for format detection, normalization, and translation."""

from __future__ import annotations

from routellect.proxy._formats import (
    InboundFormat,
    StreamState,
    _google_tools_to_openai,
    _google_tool_mode_to_openai,
    _strip_thinking_text,
    build_task_fingerprint,
    detect_format,
    error_response,
    extract_assistant_text,
    extract_usage,
    extract_user_message,
    normalize_to_openai,
    stream_epilogue,
    stream_prologue,
    translate_response,
    translate_stream_chunk,
)


class TestDetectFormat:
    def test_openai(self):
        assert detect_format("/v1/chat/completions") == InboundFormat.OPENAI

    def test_anthropic(self):
        assert detect_format("/v1/messages") == InboundFormat.ANTHROPIC

    def test_google_generate(self):
        assert detect_format("/v1beta/models/gemini-2.5-pro:generateContent") == InboundFormat.GOOGLE

    def test_google_stream(self):
        assert detect_format("/v1beta/models/gemini-2.5-flash:streamGenerateContent") == InboundFormat.GOOGLE

    def test_unknown_defaults_to_openai(self):
        assert detect_format("/unknown/path") == InboundFormat.OPENAI


class TestNormalizeOpenAI:
    def test_passthrough(self):
        body = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 100,
        }
        req = normalize_to_openai(body, InboundFormat.OPENAI)
        assert req.messages == [{"role": "user", "content": "hi"}]
        assert req.model == "gpt-4o"
        assert req.stream is True
        assert req.params["temperature"] == 0.7
        assert req.params["max_tokens"] == 100
        assert req.original_model == "gpt-4o"


class TestNormalizeAnthropic:
    def test_basic(self):
        body = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        }
        req = normalize_to_openai(body, InboundFormat.ANTHROPIC)
        assert req.messages == [{"role": "user", "content": "hi"}]
        assert req.model == "claude-sonnet-4-6"

    def test_system_prompt(self):
        body = {
            "model": "claude-sonnet-4-6",
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        }
        req = normalize_to_openai(body, InboundFormat.ANTHROPIC)
        assert req.messages[0] == {"role": "system", "content": "You are helpful."}
        assert req.messages[1] == {"role": "user", "content": "hi"}

    def test_content_blocks(self):
        body = {
            "model": "claude-sonnet-4-6",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "text", "text": "World"},
                    ],
                }
            ],
            "max_tokens": 100,
        }
        req = normalize_to_openai(body, InboundFormat.ANTHROPIC)
        assert req.messages[0]["content"] == "Hello\nWorld"


class TestNormalizeGoogle:
    def test_basic(self):
        body = {
            "contents": [
                {"role": "user", "parts": [{"text": "hi"}]},
            ],
        }
        req = normalize_to_openai(body, InboundFormat.GOOGLE)
        assert req.messages == [{"role": "user", "content": "hi"}]

    def test_system_instruction(self):
        body = {
            "system_instruction": {"parts": [{"text": "Be concise."}]},
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        }
        req = normalize_to_openai(body, InboundFormat.GOOGLE)
        assert req.messages[0] == {"role": "system", "content": "Be concise."}
        assert req.messages[1] == {"role": "user", "content": "hi"}

    def test_model_role_mapping(self):
        body = {
            "contents": [
                {"role": "user", "parts": [{"text": "hi"}]},
                {"role": "model", "parts": [{"text": "hello"}]},
                {"role": "user", "parts": [{"text": "bye"}]},
            ],
        }
        req = normalize_to_openai(body, InboundFormat.GOOGLE)
        assert req.messages[1]["role"] == "assistant"

    def test_generation_config(self):
        body = {
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "generationConfig": {
                "temperature": 0.5,
                "maxOutputTokens": 200,
                "topP": 0.9,
                "stopSequences": ["END"],
            },
        }
        req = normalize_to_openai(body, InboundFormat.GOOGLE)
        assert req.params["temperature"] == 0.5
        assert req.params["max_tokens"] == 200
        assert req.params["top_p"] == 0.9
        assert req.params["stop"] == ["END"]


class TestFingerprint:
    def test_basic(self):
        body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ],
            "tools": [{"type": "function"}],
            "max_tokens": 100,
        }
        req = normalize_to_openai(body, InboundFormat.OPENAI)
        fp = build_task_fingerprint(req)
        assert fp["message_count"] == 2
        assert fp["has_system_prompt"] is True
        assert fp["has_tools"] is True
        assert fp["max_tokens"] == 100

    def test_no_content_in_fingerprint(self):
        body = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "SECRET DATA"}],
        }
        req = normalize_to_openai(body, InboundFormat.OPENAI)
        fp = build_task_fingerprint(req)
        assert "SECRET" not in str(fp)


class TestTranslateResponse:
    OPENAI_RESPONSE = {
        "id": "chatcmpl-123",
        "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    def test_openai_passthrough(self):
        result = translate_response(self.OPENAI_RESPONSE, InboundFormat.OPENAI, "gpt-4o")
        assert result == self.OPENAI_RESPONSE

    def test_to_anthropic(self):
        result = translate_response(self.OPENAI_RESPONSE, InboundFormat.ANTHROPIC, "claude-sonnet-4-6")
        assert result["type"] == "message"
        assert result["model"] == "claude-sonnet-4-6"
        assert result["content"] == [{"type": "text", "text": "Hello!"}]
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5
        assert result["usage"]["cache_creation_input_tokens"] == 0
        assert result["stop_reason"] == "end_turn"

    def test_to_google(self):
        result = translate_response(self.OPENAI_RESPONSE, InboundFormat.GOOGLE, "gemini-2.5-pro")
        assert result["candidates"][0]["content"]["role"] == "model"
        assert result["candidates"][0]["content"]["parts"] == [{"text": "Hello!"}]
        assert result["candidates"][0]["finishReason"] == "STOP"
        assert result["usageMetadata"]["promptTokenCount"] == 10
        assert result["usageMetadata"]["candidatesTokenCount"] == 5

    def test_none_content_handling(self):
        data = {"choices": [{"message": None}], "usage": {}}
        result = translate_response(data, InboundFormat.ANTHROPIC, "test")
        assert result["content"] == [{"type": "text", "text": ""}]


class TestErrorResponse:
    def test_openai_format(self):
        err = error_response(InboundFormat.OPENAI, "bad request")
        assert err["error"]["message"] == "bad request"
        assert err["error"]["type"] == "proxy_error"

    def test_anthropic_format(self):
        err = error_response(InboundFormat.ANTHROPIC, "bad request")
        assert err["type"] == "error"
        assert err["error"]["type"] == "api_error"

    def test_google_format(self):
        err = error_response(InboundFormat.GOOGLE, "bad request", 502)
        assert err["error"]["code"] == 502
        assert err["error"]["status"] == "UNAVAILABLE"


class TestExtractUserMessage:
    def test_openai(self):
        body = {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]}
        assert extract_user_message(body, InboundFormat.OPENAI) == "hello"

    def test_anthropic(self):
        body = {"messages": [{"role": "user", "content": "hello"}]}
        assert extract_user_message(body, InboundFormat.ANTHROPIC) == "hello"

    def test_google(self):
        body = {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]}
        assert extract_user_message(body, InboundFormat.GOOGLE) == "hello"

    def test_empty(self):
        assert extract_user_message({}, InboundFormat.OPENAI) == ""


class TestStreamTranslation:
    def test_openai_passthrough(self):
        state = StreamState(original_model="gpt-4o")
        chunk = {"choices": [{"delta": {"content": "Hi"}}]}
        lines = translate_stream_chunk(chunk, InboundFormat.OPENAI, state)
        assert any("Hi" in l for l in lines)

    def test_anthropic_with_prologue(self):
        state = StreamState(original_model="claude-sonnet-4-6")
        chunk = {"choices": [{"delta": {"content": "Hi"}}]}
        lines = translate_stream_chunk(chunk, InboundFormat.ANTHROPIC, state)
        # Should have prologue + content delta
        assert any("message_start" in l for l in lines)
        assert any("content_block_delta" in l for l in lines)
        assert state.started is True

    def test_google_streaming(self):
        state = StreamState(original_model="gemini-2.5-pro")
        chunk = {"choices": [{"delta": {"content": "Hi"}}]}
        lines = translate_stream_chunk(chunk, InboundFormat.GOOGLE, state)
        assert any('"text": "Hi"' in l for l in lines)

    def test_epilogue_openai(self):
        state = StreamState(original_model="gpt-4o", output_tokens=50)
        lines = stream_epilogue(InboundFormat.OPENAI, state)
        assert lines == ["data: [DONE]\n\n"]

    def test_epilogue_anthropic(self):
        state = StreamState(original_model="claude-sonnet-4-6", output_tokens=50)
        lines = stream_epilogue(InboundFormat.ANTHROPIC, state)
        assert any("content_block_stop" in l for l in lines)
        assert any("message_delta" in l for l in lines)
        assert any("message_stop" in l for l in lines)

    def test_epilogue_google(self):
        state = StreamState(original_model="gemini-2.5-pro", input_tokens=10, output_tokens=50)
        lines = stream_epilogue(InboundFormat.GOOGLE, state)
        assert any("usageMetadata" in l for l in lines)

    def test_token_collection(self):
        state = StreamState(original_model="gpt-4o")
        chunk = {"choices": [{"delta": {"content": "Hi"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        translate_stream_chunk(chunk, InboundFormat.OPENAI, state)
        assert state.input_tokens == 10
        assert state.output_tokens == 5
        assert state.collected_content == ["Hi"]


class TestExtractHelpers:
    def test_extract_assistant_text(self):
        data = {"choices": [{"message": {"content": "Hello!"}}]}
        assert extract_assistant_text(data) == "Hello!"

    def test_extract_assistant_text_none(self):
        data = {"choices": [{"message": None}]}
        assert extract_assistant_text(data) == ""

    def test_extract_usage(self):
        data = {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        assert extract_usage(data) == (10, 5)

    def test_extract_usage_empty(self):
        assert extract_usage({}) == (0, 0)


class TestGoogleToolNormalization:
    """Test Google functionDeclarations → OpenAI tools conversion."""

    def test_basic_function_declarations(self):
        google_tools = [
            {
                "functionDeclarations": [
                    {
                        "name": "get_weather",
                        "description": "Get weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    }
                ]
            }
        ]
        result = _google_tools_to_openai(google_tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get weather for a city"
        assert result[0]["function"]["parameters"]["type"] == "object"

    def test_snake_case_key(self):
        google_tools = [
            {
                "function_declarations": [
                    {"name": "do_thing", "description": "Does a thing", "parameters": {}},
                ]
            }
        ]
        result = _google_tools_to_openai(google_tools)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "do_thing"

    def test_multiple_declarations(self):
        google_tools = [
            {
                "functionDeclarations": [
                    {"name": "tool_a", "description": "A"},
                    {"name": "tool_b", "description": "B"},
                ]
            }
        ]
        result = _google_tools_to_openai(google_tools)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "tool_a"
        assert result[1]["function"]["name"] == "tool_b"

    def test_empty_tools(self):
        assert _google_tools_to_openai([]) == []

    def test_tool_mode_mapping(self):
        assert _google_tool_mode_to_openai("AUTO") == "auto"
        assert _google_tool_mode_to_openai("ANY") == "required"
        assert _google_tool_mode_to_openai("NONE") == "none"
        assert _google_tool_mode_to_openai("auto") == "auto"

    def test_normalize_google_with_tools(self):
        body = {
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "tools": [
                {
                    "functionDeclarations": [
                        {"name": "read", "description": "Read a file", "parameters": {}},
                    ]
                }
            ],
            "toolConfig": {
                "functionCallingConfig": {"mode": "AUTO"},
            },
        }
        req = normalize_to_openai(body, InboundFormat.GOOGLE)
        assert len(req.params["tools"]) == 1
        assert req.params["tools"][0]["function"]["name"] == "read"
        assert req.params["tool_choice"] == "auto"


class TestGoogleToolResponseTranslation:
    """Test OpenAI tool_calls → Google functionCall response translation."""

    def test_response_with_tool_calls(self):
        data = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Paris"}',
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = translate_response(data, InboundFormat.GOOGLE, "gemini-2.5-pro")
        parts = result["candidates"][0]["content"]["parts"]
        assert any("functionCall" in p for p in parts)
        fc_part = [p for p in parts if "functionCall" in p][0]
        assert fc_part["functionCall"]["name"] == "get_weather"
        assert fc_part["functionCall"]["args"] == {"city": "Paris"}
        assert result["candidates"][0]["finishReason"] == "TOOL_CALLS"

    def test_response_text_only(self):
        data = {
            "choices": [{"message": {"content": "Hello!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = translate_response(data, InboundFormat.GOOGLE, "gemini-2.5-pro")
        assert result["candidates"][0]["content"]["parts"] == [{"text": "Hello!"}]
        assert result["candidates"][0]["finishReason"] == "STOP"


class TestThinkingTextStripping:
    """Test that thinking/reasoning blocks are stripped from content."""

    def test_strip_thinking_tags(self):
        content = "<thinking>Let me reason about this...</thinking>The answer is 42."
        assert _strip_thinking_text(content) == "The answer is 42."

    def test_strip_thought_tags(self):
        content = "<thought>Internal reasoning here</thought>Hello!"
        assert _strip_thinking_text(content) == "Hello!"

    def test_strip_reasoning_tags(self):
        content = "<reasoning>Step 1... Step 2...</reasoning>Result: yes"
        assert _strip_thinking_text(content) == "Result: yes"

    def test_strip_internal_monologue(self):
        content = "<internal_monologue>Hmm, should I call a tool?</internal_monologue>I'll help you."
        assert _strip_thinking_text(content) == "I'll help you."

    def test_multiline_thinking(self):
        content = "<thinking>\nLet me think step by step.\n1. First\n2. Second\n</thinking>\nThe answer is Paris."
        assert _strip_thinking_text(content) == "The answer is Paris."

    def test_no_thinking_passthrough(self):
        content = "Just a normal response."
        assert _strip_thinking_text(content) == "Just a normal response."

    def test_empty_string(self):
        assert _strip_thinking_text("") == ""

    def test_sanitize_response_strips_thinking(self):
        data = {
            "choices": [
                {
                    "message": {
                        "content": "<thinking>Let me plan...</thinking>Here's your answer.",
                    }
                }
            ],
        }
        from routellect.proxy._formats import _sanitize_response
        result = _sanitize_response(data)
        assert result["choices"][0]["message"]["content"] == "Here's your answer."

    def test_case_insensitive(self):
        content = "<THINKING>Uppercase thinking</THINKING>Answer."
        assert _strip_thinking_text(content) == "Answer."
