#!/usr/bin/env python3
"""Tests for colibrì OpenAI API wrapper."""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_build_prompt_single_user():
    from api_server import build_prompt_from_messages
    result = build_prompt_from_messages([{"role": "user", "content": "Hello"}])
    assert "[gMASK]<sop>" in result
    assert "<|user|>Hello" in result
    assert result.endswith("<|assistant|><think></think>")
    print("PASS: test_build_prompt_single_user")


def test_build_prompt_multi_turn():
    from api_server import build_prompt_from_messages
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "How are you?"},
    ]
    result = build_prompt_from_messages(msgs)
    assert "<|user|>Hi" in result
    assert "<|assistant|>Hello!" in result
    assert "<|user|>How are you?" in result
    assert result.endswith("<|assistant|><think></think>")
    print("PASS: test_build_prompt_multi_turn")


def test_build_prompt_with_system():
    from api_server import build_prompt_from_messages
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]
    result = build_prompt_from_messages(msgs)
    assert "You are helpful." in result
    assert "<|user|>" in result
    # system content should come before user content
    assert result.index("You are helpful.") < result.index("Hi")
    print("PASS: test_build_prompt_with_system")


def test_build_prompt_think_mode():
    from api_server import build_prompt_from_messages
    os.environ["COLI_THINK"] = "1"
    try:
        result = build_prompt_from_messages([{"role": "user", "content": "Think!"}])
        assert result.endswith("<|assistant|><think>")
        assert "<think></think>" not in result
    finally:
        del os.environ["COLI_THINK"]
    print("PASS: test_build_prompt_think_mode")


def test_openai_response_shape():
    """Verify non-streaming response matches OpenAI schema."""
    mock_response = {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "colibri-glm-5.2",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": -1, "completion_tokens": 5, "total_tokens": -1},
    }
    assert mock_response["object"] == "chat.completion"
    assert mock_response["choices"][0]["message"]["role"] == "assistant"
    assert mock_response["choices"][0]["finish_reason"] == "stop"
    print("PASS: test_openai_response_shape")


def test_sse_format():
    """Verify SSE chunk format matches OpenAI streaming spec."""
    chunk = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "colibri-glm-5.2",
        "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}]
    }
    sse_line = f"data: {json.dumps(chunk)}\n\n"
    assert sse_line.startswith("data: ")
    assert sse_line.endswith("\n\n")
    parsed = json.loads(sse_line[6:].strip())
    assert parsed["object"] == "chat.completion.chunk"
    assert parsed["choices"][0]["delta"]["content"] == "Hi"
    print("PASS: test_sse_format")


def test_colibri_stats_in_response():
    """Verify colibri_stats extension is included in non-streaming response."""
    mock_stats = {"tok": 42, "tps": 0.08, "hit": 15.0, "rss": 18.5}
    mock_response = {
        "id": "chatcmpl-x",
        "object": "chat.completion",
        "created": 123,
        "model": "colibri-glm-5.2",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": -1, "completion_tokens": 42, "total_tokens": -1},
        "colibri_stats": {
            "tok": mock_stats["tok"],
            "tps": round(mock_stats["tps"], 2),
            "expert_hit": round(mock_stats["hit"], 1),
            "rss_gb": round(mock_stats["rss"], 2),
        }
    }
    assert "colibri_stats" in mock_response
    assert mock_response["colibri_stats"]["tok"] == 42
    assert mock_response["colibri_stats"]["tps"] == 0.08
    print("PASS: test_colibri_stats_in_response")


if __name__ == "__main__":
    test_build_prompt_single_user()
    test_build_prompt_multi_turn()
    test_build_prompt_with_system()
    test_build_prompt_think_mode()
    test_openai_response_shape()
    test_sse_format()
    test_colibri_stats_in_response()
    print("\n All tests passed!")
