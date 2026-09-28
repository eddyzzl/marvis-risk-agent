"""The JSON-only provider must see the same declared output contract."""
import json

import pytest

from marvis.llm_client import LLMClientError, LLMClientErrorKind, OpenAICompatibleLLMClient


SCHEMA = {
    "name": "bounded_route",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"intent": {"enum": ["current_workflow", "none"]}, "reason": {"type": "string"}},
        "required": ["intent", "reason"],
        "additionalProperties": False,
    },
}


class Reply:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def __iter__(self):
        yield json.dumps({"choices": [{"message": {"content": '{"intent":"none","reason":"unclear"}'}}]}).encode()


def profile(**updates):
    return {"api_base_url": "https://example.test/v1", "api_key": "private-key", "model_name": "m", **updates}


@pytest.mark.parametrize("format_arg", [None, {"type": "json_object"}])
def test_json_object_provider_receives_exact_schema_in_system_message(monkeypatch, format_arg):
    sent = []
    monkeypatch.setattr("marvis.llm_client.urlopen", lambda request, timeout: (sent.append(json.loads(request.data)) or Reply()))
    records = []
    result = OpenAICompatibleLLMClient(profile()).complete(
        system_prompt="Classify only; never execute.", user_prompt="Continue?",
        json_schema=SCHEMA, response_format=format_arg, stream=False, on_call_recorded=records.append,
    )
    body = sent[0]
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][0]["content"].startswith("Classify only; never execute.\n\n")
    embedded = body["messages"][0]["content"].split("\n")[-1]
    assert json.loads(embedded) == SCHEMA
    assert body["messages"][1] == {"role": "user", "content": "Continue?"}
    assert records[0]["prompt_chars"] == sum(len(m["content"]) for m in body["messages"])
    assert json.loads(result)["intent"] == "none"


def test_native_schema_remains_provider_field_without_duplicate_system_prompt(monkeypatch):
    sent = []
    monkeypatch.setattr("marvis.llm_client.urlopen", lambda request, timeout: (sent.append(json.loads(request.data)) or Reply()))
    OpenAICompatibleLLMClient(profile(structured_output="json_schema")).complete(
        system_prompt="s", user_prompt="u", json_schema=SCHEMA, stream=False,
    )
    assert sent[0]["messages"] == [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    assert sent[0]["response_format"] == {"type": "json_schema", "json_schema": SCHEMA}


@pytest.mark.parametrize("format_arg", [None, {"type": "json_object"}])
def test_plain_text_profile_keeps_explicit_transport_choice(monkeypatch, format_arg):
    sent = []
    def provider(request, timeout):
        body = json.loads(request.data)
        sent.append(body)
        if format_arg is None:
            assert "response_format" not in body, "text-only provider rejects this field"
        return Reply()

    monkeypatch.setattr("marvis.llm_client.urlopen", provider)
    OpenAICompatibleLLMClient(profile(structured_output="none")).complete(
        system_prompt="s", user_prompt="u", json_schema=SCHEMA,
        response_format=format_arg, stream=False,
    )
    assert json.loads(sent[0]["messages"][0]["content"].split("\n")[-1]) == SCHEMA
    if format_arg is not None:
        assert sent[0]["response_format"] == format_arg


@pytest.mark.parametrize("structured_output", ["json_schema", "json_object"])
def test_schema_counts_towards_context_before_any_provider_request(monkeypatch, structured_output):
    calls = []
    monkeypatch.setattr("marvis.llm_client.urlopen", lambda *a, **k: calls.append(1))
    schema = {**SCHEMA, "schema": {**SCHEMA["schema"], "description": "sensitive-value" * 200}}
    with pytest.raises(LLMClientError) as error:
        OpenAICompatibleLLMClient(profile(structured_output=structured_output, context_window=100)).complete(
            system_prompt="s", user_prompt="u", json_schema=schema, max_tokens=10, stream=False,
        )
    assert error.value.error_kind is LLMClientErrorKind.CONTEXT_LENGTH_EXCEEDED
    assert "sensitive-value" not in str(error.value)
    assert calls == []


@pytest.mark.parametrize("bad_value", [float("nan"), object()])
def test_invalid_schema_is_rejected_without_echo_or_request(monkeypatch, bad_value):
    calls = []
    monkeypatch.setattr("marvis.llm_client.urlopen", lambda *a, **k: calls.append(1))
    schema = {**SCHEMA, "sensitive-contract-key": bad_value}
    with pytest.raises(LLMClientError) as error:
        OpenAICompatibleLLMClient(profile()).complete(system_prompt="s", user_prompt="u", json_schema=schema, stream=False)
    assert "sensitive-contract-key" not in str(error.value)
    assert calls == []
