from __future__ import annotations

from dataclasses import replace
import json

import pytest

from marvis.agent.semantic_authorization import review_semantic_authorization
from marvis.agent.semantic_diagnostics import SCHEMA, sanitize_diagnostics
from marvis.agent.semantic_intent import (
    INTENT_CURRENT_WORKFLOW,
    INTENT_NONE,
    route_semantic_intent,
)

SECRET = "raw-private-response-sentinel"
INSTRUCTION = "材料已经齐备，请继续当前流程。"


class Client:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def intent_reply(**changes):
    data = dict(
        intent=INTENT_CURRENT_WORKFLOW,
        evidence_quote=INSTRUCTION,
        reason=SECRET,
        confidence="high",
        is_question=False,
        is_conditional=False,
        requests_change=False,
        withholds_action=False,
    )
    data.update(changes)
    return json.dumps(data, ensure_ascii=False)


def route(client):
    return route_semantic_intent(
        client,
        task_type="vintage",
        instruction=INSTRUCTION,
        context={},
        allowed_intents=(INTENT_CURRENT_WORKFLOW, INTENT_NONE),
    )


def auth_reply(**changes):
    data = dict(
        verdict="authorize",
        evidence_quote=INSTRUCTION,
        reason=SECRET,
        confidence="high",
        is_question=False,
        is_conditional=False,
        requests_change=False,
        withholds_authorization=False,
    )
    data.update(changes)
    return json.dumps(data, ensure_ascii=False)


def review(client):
    return review_semantic_authorization(
        client,
        gate_context="当前计划",
        instruction=INSTRUCTION,
        proposed_params={"private": SECRET},
    )


@pytest.mark.parametrize(
    ("raw", "code", "first"),
    [
        ("```json\n" + SECRET + "\n```", "non_json", "code_fence"),
        ('{"private": "' + SECRET + '",}', "non_json", "object"),
        ('{"private":1,"private":2}', "duplicate_key", "object"),
        ('["' + SECRET + '"]', "non_object", "array"),
        ("", "empty_response", "empty"),
        (None, "non_text", "non_text"),
    ],
)
def test_intent_failure_shape_is_content_free_and_does_not_add_calls(raw, code, first):
    client = Client(raw)
    result = route(client)
    diag = result.as_metadata()["semantic_diagnostics"]
    assert not result.accepted
    assert diag["failure_code"] == "semantic_intent_router_" + code
    assert len(client.calls) == 1
    attempt = diag["passes"][0]["attempts"][0]
    assert attempt["first_token"] == first
    assert attempt["parse_valid"] is False
    assert attempt["code_fence"] == (first == "code_fence")
    if code == "non_json":
        assert type(attempt["json_error_position"]) is int
        assert attempt["json_error_line"] >= 1
        assert attempt["json_error_column"] >= 1
    assert diag["passes"][0]["repair_status"] == "not_attempted"
    assert SECRET not in json.dumps(diag)
    assert INSTRUCTION not in json.dumps(diag, ensure_ascii=False)


def test_missing_reason_repair_records_both_passes_without_promoting_medium_confidence():
    first = json.loads(intent_reply(confidence="medium"))
    first.pop("reason")
    second = json.loads(intent_reply())
    second.pop("reason")
    client = Client(
        json.dumps(first),
        intent_reply(confidence="medium"),
        json.dumps(second),
        intent_reply(),
    )
    result = route(client)
    assert not result.accepted
    diag = result.diagnostics
    assert diag["failure_code"] == "semantic_intent_unsafe_decision"
    assert [p["repair_status"] for p in diag["passes"]] == ["succeeded", "succeeded"]
    for passed in diag["passes"]:
        assert [a["failure_code"] for a in passed["attempts"]] == [
            "missing_fields",
            None,
        ]
        assert [a["parse_valid"] for a in passed["attempts"]] == [False, True]
    assert diag["passes"][0]["attempts"][-1]["decision"]["confidence"] == "medium"
    assert diag["passes"][1]["attempts"][-1]["decision"]["confidence"] == "high"
    assert len(client.calls) == 4
    assert all(c["max_tokens"] == 2048 for c in client.calls)
    assert SECRET not in json.dumps(diag)


def test_semantic_drift_and_disagreement_are_distinct_from_parse_failure():
    missing = json.loads(intent_reply(confidence="medium"))
    missing.pop("reason")
    client = Client(json.dumps(missing), intent_reply())
    result = route(client)
    assert result.failure_code == "semantic_intent_router_repair_semantic_drift"
    assert result.diagnostics["passes"][0]["repair_status"] == "failed"
    assert result.diagnostics["passes"][0]["attempts"][-1]["parse_valid"] is True
    assert len(client.calls) == 2
    client = Client(intent_reply(), intent_reply(intent=INTENT_NONE))
    result = route(client)
    assert result.failure_code == "semantic_intent_pass_disagreement"
    assert all(p["attempts"][0]["parse_valid"] for p in result.diagnostics["passes"])
    assert len(client.calls) == 2


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        ("```json\n" + SECRET + "\n```", "non_json"),
        ('{"verdict":"authorize","verdict":"reject"}', "duplicate_key"),
        ('{"verdict":"authorize"}', "missing_fields"),
        (auth_reply(extra=SECRET), "extra_fields"),
        (auth_reply(confidence=SECRET), "invalid_field_type"),
        (auth_reply(confidence="medium"), "unauthorized"),
        (auth_reply(evidence_quote=SECRET), "unauthorized"),
        (auth_reply(is_conditional=True), "unauthorized"),
        (RuntimeError(SECRET), "request_failed"),
    ],
)
def test_authorization_failure_preserves_single_call_and_reports_safe_facts(raw, code):
    client = Client(raw)
    result = review(client)
    assert not result.authorized
    assert result.diagnostics["failure_code"] == code
    assert result.diagnostics["accepted"] is False
    assert len(client.calls) == 1
    assert client.calls[0]["max_tokens"] == 1024
    assert SECRET not in json.dumps(result.diagnostics)
    assert INSTRUCTION not in json.dumps(result.diagnostics, ensure_ascii=False)
    if code == "unauthorized":
        attempt = result.diagnostics["passes"][0]["attempts"][0]
        assert attempt["parse_valid"] is True
        if "is_conditional" in raw and json.loads(raw)["is_conditional"]:
            assert attempt["decision"]["is_conditional"] is True


def test_whitelist_drops_arbitrary_strings_fields_positions_and_deep_content():
    hostile = dict(
        stage=SECRET,
        failure_code=SECRET,
        raw=SECRET,
        reason=SECRET,
        first_token=SECRET,
        json_error_category=SECRET,
        accepted="true",
        json_error_position=True,
        json_error_line=-1,
        json_error_column=10**20,
        decision=dict(
            intent=SECRET,
            verdict=SECRET,
            confidence=SECRET,
            evidence_quote=SECRET,
            params={"key": SECRET},
            is_question=True,
        ),
    )
    payload = dict(
        schema_version=SCHEMA,
        **hostile,
        passes=[
            dict(
                hostile,
                stage="authorization",
                attempts=[dict(hostile, attempts=[hostile])] * 3,
            )
        ]
        * 3,
    )
    safe = sanitize_diagnostics(payload)
    assert safe == {
        "schema_version": SCHEMA,
        "decision": {"is_question": True},
        "passes": [
            {
                "stage": "authorization",
                "decision": {"is_question": True},
                "attempts": [{"decision": {"is_question": True}}] * 2,
            }
        ]
        * 2,
    }
    assert SECRET not in json.dumps(safe)
    assert sanitize_diagnostics({"schema_version": "unknown", "raw": SECRET}) is None


def test_optional_diagnostics_preserve_legacy_metadata_and_result_equality():
    result = route(Client(intent_reply(), intent_reply()))
    legacy = replace(result, diagnostics=None)
    assert result == legacy
    metadata = result.as_metadata()
    metadata.pop("semantic_diagnostics")
    assert legacy.as_metadata() == metadata
    assert (
        replace(result, diagnostics={"schema_version": "unknown"}).as_metadata()
        == metadata
    )
    auth = review(Client(auth_reply()))
    assert auth == replace(auth, diagnostics=None)
