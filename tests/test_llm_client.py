import json

from riskmodel_checker.llm_client import OpenAICompatibleLLMClient


class _StreamingResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def __iter__(self):
        events = [
            'data: {"choices":[{"delta":{"reasoning_content":"hidden thinking"}}]}\n',
            'data: {"choices":[{"delta":{"content":"第一段"}}]}\n',
            'data: {"choices":[{"delta":{"content":"第二段"}}]}\n',
            "data: [DONE]\n",
        ]
        return iter(event.encode("utf-8") for event in events)


def test_openai_compatible_client_uses_streaming_reasoning_and_thinking(monkeypatch):
    captured = {}
    chunks = []

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["authorization"] = request.get_header("Authorization")
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _StreamingResponse()

    monkeypatch.setattr("riskmodel_checker.llm_client.urlopen", fake_urlopen)

    content = OpenAICompatibleLLMClient(
        {
            "api_base_url": "https://api.deepseek.com",
            "model_name": "deepseek-v4-pro",
            "api_key": "secret",
            "timeout_seconds": 45,
        }
    ).complete(
        system_prompt="You are a helpful assistant",
        user_prompt="Hello",
        response_format={"type": "json_object"},
        on_delta=chunks.append,
    )

    assert content == "第一段第二段"
    assert chunks == ["第一段", "第二段"]
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["timeout"] == 45
    assert captured["authorization"] == "Bearer secret"
    assert captured["payload"]["model"] == "deepseek-v4-pro"
    assert captured["payload"]["stream"] is True
    assert captured["payload"]["reasoning_effort"] == "high"
    assert captured["payload"]["thinking"] == {"type": "enabled"}
    assert captured["payload"]["response_format"] == {"type": "json_object"}


def test_client_rejects_non_http_base_url():
    import pytest

    from riskmodel_checker.llm_client import LLMClientError

    with pytest.raises(LLMClientError):
        OpenAICompatibleLLMClient(
            {
                "api_base_url": "file:///etc",
                "model_name": "m",
                "api_key": "secret",
            }
        ).complete(system_prompt="s", user_prompt="u")


def test_client_honors_reasoning_effort_from_profile(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _StreamingResponse()

    monkeypatch.setattr("riskmodel_checker.llm_client.urlopen", fake_urlopen)

    OpenAICompatibleLLMClient(
        {
            "api_base_url": "https://api.deepseek.com",
            "model_name": "deepseek-v4-pro",
            "api_key": "secret",
            "reasoning_effort": "low",
        }
    ).complete(system_prompt="s", user_prompt="u")

    assert captured["payload"]["reasoning_effort"] == "low"
