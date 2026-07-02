import json
from http.client import RemoteDisconnected
from urllib.error import HTTPError, URLError

import pytest

from marvis.llm_client import LLMClientError, OpenAICompatibleLLMClient


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


class _JsonResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def __iter__(self):
        return iter([
            b'{"choices":[{"message":{"content":"plain json"}}]}',
        ])


class _InterruptedStreamingResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def __iter__(self):
        raise RemoteDisconnected("stream dropped")


def test_openai_compatible_client_defaults_to_portable_stream_payload(monkeypatch):
    captured = {}
    chunks = []

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["authorization"] = request.get_header("Authorization")
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _StreamingResponse()

    monkeypatch.setattr("marvis.llm_client.urlopen", fake_urlopen)

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
    assert "reasoning_effort" not in captured["payload"]
    assert "thinking" not in captured["payload"]
    assert captured["payload"]["response_format"] == {"type": "json_object"}


def test_client_rejects_non_http_base_url():
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

    monkeypatch.setattr("marvis.llm_client.urlopen", fake_urlopen)

    OpenAICompatibleLLMClient(
        {
            "api_base_url": "https://api.deepseek.com",
            "model_name": "deepseek-v4-pro",
            "api_key": "secret",
            "enable_thinking": True,
            "reasoning_effort": "low",
        }
    ).complete(system_prompt="s", user_prompt="u")

    assert captured["payload"]["reasoning_effort"] == "low"
    assert captured["payload"]["thinking"] == {"type": "enabled"}


def test_client_sends_extra_request_fields_when_configured(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _StreamingResponse()

    monkeypatch.setattr("marvis.llm_client.urlopen", fake_urlopen)

    OpenAICompatibleLLMClient(
        {
            "api_base_url": "https://api.example.com/v1",
            "model_name": "compatible-model",
            "api_key": "secret",
            "extra_request_fields": {"top_p": 0.8},
        }
    ).complete(system_prompt="s", user_prompt="u")

    assert captured["payload"]["top_p"] == 0.8
    assert "thinking" not in captured["payload"]


def test_client_can_request_non_streaming_json_response(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _JsonResponse()

    monkeypatch.setattr("marvis.llm_client.urlopen", fake_urlopen)

    content = OpenAICompatibleLLMClient(
        {
            "api_base_url": "https://api.example.com/v1",
            "model_name": "compatible-model",
            "api_key": "secret",
        }
    ).complete(system_prompt="s", user_prompt="u", stream=False)

    assert content == "plain json"
    assert captured["payload"]["stream"] is False


def test_client_wraps_stream_interruptions(monkeypatch):
    def fake_urlopen(request, timeout):
        return _InterruptedStreamingResponse()

    monkeypatch.setattr("marvis.llm_client.urlopen", fake_urlopen)

    with pytest.raises(LLMClientError, match="LLM stream interrupted"):
        OpenAICompatibleLLMClient(
            {
                "api_base_url": "https://api.example.com/v1",
                "model_name": "compatible-model",
                "api_key": "secret",
            }
        ).complete(system_prompt="s", user_prompt="u")


def test_client_sends_json_schema_when_profile_supports_it(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _JsonResponse()

    monkeypatch.setattr("marvis.llm_client.urlopen", fake_urlopen)

    schema = {"name": "decision", "schema": {"type": "object"}, "strict": True}
    OpenAICompatibleLLMClient(
        {
            "api_base_url": "https://api.example.com/v1",
            "model_name": "m",
            "api_key": "secret",
            "structured_output": "json_schema",
        }
    ).complete(
        system_prompt="s",
        user_prompt="u",
        response_format={"type": "json_object"},
        json_schema=schema,
        stream=False,
    )

    assert captured["payload"]["response_format"] == {
        "type": "json_schema",
        "json_schema": schema,
    }


def test_client_falls_back_to_json_object_when_schema_unsupported(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _JsonResponse()

    monkeypatch.setattr("marvis.llm_client.urlopen", fake_urlopen)

    schema = {"name": "decision", "schema": {"type": "object"}, "strict": True}
    OpenAICompatibleLLMClient(
        {
            "api_base_url": "https://api.example.com/v1",
            "model_name": "m",
            "api_key": "secret",
            # structured_output defaults to json_object -> schema ignored
        }
    ).complete(
        system_prompt="s",
        user_prompt="u",
        response_format={"type": "json_object"},
        json_schema=schema,
        stream=False,
    )

    assert captured["payload"]["response_format"] == {"type": "json_object"}


class _JsonResponseWithUsage:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def __iter__(self):
        return iter([
            b'{"choices":[{"message":{"content":"ok"}}],'
            b'"usage":{"prompt_tokens":11,"completion_tokens":7}}',
        ])


def test_complete_invokes_on_call_recorded_with_usage(monkeypatch):
    def fake_urlopen(request, timeout):
        return _JsonResponseWithUsage()

    monkeypatch.setattr("marvis.llm_client.urlopen", fake_urlopen)

    records = []
    OpenAICompatibleLLMClient(
        {
            "model_id": "m1",
            "api_base_url": "https://api.example.com/v1",
            "model_name": "m",
            "api_key": "secret",
        }
    ).complete(
        system_prompt="sys",
        user_prompt="user",
        stream=False,
        caller="gate",
        on_call_recorded=records.append,
    )

    assert len(records) == 1
    record = records[0]
    assert record["caller"] == "gate"
    assert record["model_id"] == "m1"
    assert record["prompt_tokens"] == 11
    assert record["completion_tokens"] == 7
    assert record["ok"] is True
    assert record["error_kind"] is None
    assert record["streamed"] is False
    assert record["prompt_chars"] == len("sys") + len("user")
    assert record["latency_ms"] >= 0


def test_streaming_requests_include_usage_stream_option(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _StreamingResponse()

    monkeypatch.setattr("marvis.llm_client.urlopen", fake_urlopen)

    OpenAICompatibleLLMClient(
        {
            "api_base_url": "https://api.example.com/v1",
            "model_name": "m",
            "api_key": "secret",
        }
    ).complete(system_prompt="s", user_prompt="u")

    assert captured["payload"]["stream_options"] == {"include_usage": True}


class _Http4xxError(HTTPError):
    def __init__(self):
        super().__init__(
            url="https://api.example.com/v1/chat/completions",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=None,
        )

    def read(self):
        return b""


def test_transient_failure_is_retried_once_then_succeeds(monkeypatch):
    monkeypatch.setattr("marvis.llm_client.time.sleep", lambda _s: None)
    calls = {"n": 0}
    records = []

    def fake_urlopen(request, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise URLError("connection reset")
        return _JsonResponse()

    monkeypatch.setattr("marvis.llm_client.urlopen", fake_urlopen)

    content = OpenAICompatibleLLMClient(
        {
            "api_base_url": "https://api.example.com/v1",
            "model_name": "m",
            "api_key": "secret",
        }
    ).complete(
        system_prompt="s",
        user_prompt="u",
        stream=False,
        on_call_recorded=records.append,
    )

    assert content == "plain json"
    assert calls["n"] == 2
    assert records[0]["ok"] is True
    assert records[0]["retry_count"] == 1


def test_http_4xx_is_not_retried(monkeypatch):
    monkeypatch.setattr("marvis.llm_client.time.sleep", lambda _s: None)
    calls = {"n": 0}
    records = []

    def fake_urlopen(request, timeout):
        calls["n"] += 1
        raise _Http4xxError()

    monkeypatch.setattr("marvis.llm_client.urlopen", fake_urlopen)

    with pytest.raises(LLMClientError, match="LLM HTTP 400"):
        OpenAICompatibleLLMClient(
            {
                "api_base_url": "https://api.example.com/v1",
                "model_name": "m",
                "api_key": "secret",
            }
        ).complete(
            system_prompt="s",
            user_prompt="u",
            stream=False,
            on_call_recorded=records.append,
        )

    assert calls["n"] == 1
    assert records[0]["ok"] is False
    assert records[0]["error_kind"] == "http_4xx"
    assert records[0]["retry_count"] == 0


class _InterruptedAfterDeltaResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def __iter__(self):
        yield 'data: {"choices":[{"delta":{"content":"partial"}}]}\n'.encode("utf-8")
        raise RemoteDisconnected("stream dropped after first delta")


def test_interruption_after_on_delta_is_not_retried(monkeypatch):
    monkeypatch.setattr("marvis.llm_client.time.sleep", lambda _s: None)
    calls = {"n": 0}
    deltas = []

    def fake_urlopen(request, timeout):
        calls["n"] += 1
        return _InterruptedAfterDeltaResponse()

    monkeypatch.setattr("marvis.llm_client.urlopen", fake_urlopen)

    with pytest.raises(LLMClientError, match="LLM stream interrupted"):
        OpenAICompatibleLLMClient(
            {
                "api_base_url": "https://api.example.com/v1",
                "model_name": "m",
                "api_key": "secret",
            }
        ).complete(system_prompt="s", user_prompt="u", on_delta=deltas.append)

    assert calls["n"] == 1
    assert deltas == ["partial"]
