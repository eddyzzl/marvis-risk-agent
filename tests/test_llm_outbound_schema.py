"""Protocol evidence for installed schema provenance, without provider calls."""

from dataclasses import FrozenInstanceError
import json

import pytest

from marvis.agent.semantic_intent import ALL_SEMANTIC_INTENTS, _schema
from marvis.agent.strategy_request_compiler.core import STRATEGY_REQUEST_JSON_SCHEMA
from marvis.agent.strategy_request_compiler.sample_design import SAMPLE_DESIGN_V2_CORRECTION_JSON_SCHEMA
from marvis.llm_client import LLMClientError, LLMClientErrorKind
from marvis.llm_outbound import (
    ApplicationSchema,
    OutboundContextError,
    application_schema,
    outbound_prompt,
    safe_schema,
)
from tests.test_llm_outbound import Wire


def _send(wire, schema):
    wire.client.complete(system_prompt="schema test", user_prompt=outbound_prompt({}),
                         json_schema=schema, stream=False)


def _contract_on_wire(wire):
    body = wire.requests[-1]
    if body.get("response_format", {}).get("type") == "json_schema":
        return body["response_format"]["json_schema"]
    return json.loads(body["messages"][0]["content"].split("\n")[-1])


@pytest.mark.parametrize("output_mode", ["json_schema", "json_object", "none"])
def test_all_installed_intent_enums_reach_exact_protocol_contract(monkeypatch, output_mode):
    wire = Wire(monkeypatch)
    wire.client.profile["structured_output"] = output_mode
    allowed = sorted(ALL_SEMANTIC_INTENTS)
    _send(wire, application_schema("semantic_intent", allowed_intents=allowed))
    assert _contract_on_wire(wire) == _schema(allowed)
    assert len(_contract_on_wire(wire)["schema"]["properties"]["intent"]["enum"]) == 15


@pytest.mark.parametrize("contract,expected", [
    ("strategy_request", STRATEGY_REQUEST_JSON_SCHEMA),
    ("sample_design_v2_correction", SAMPLE_DESIGN_V2_CORRECTION_JSON_SCHEMA),
])
def test_installed_strategy_unions_and_nested_consts_are_preserved(monkeypatch, contract, expected):
    wire = Wire(monkeypatch)
    _send(wire, application_schema(contract))
    assert _contract_on_wire(wire) == expected


def test_dynamic_feature_enums_are_family_scoped(monkeypatch):
    wire = Wire(monkeypatch)
    _send(wire, application_schema("feature_meaning_direction", features=["income", "utilization"]))
    fields = _contract_on_wire(wire)["schema"]["properties"]["directions"]["items"]["properties"]
    assert fields["feature"]["enum"] == ["income", "utilization"]
    assert fields["direction"]["enum"] == ["negative", "positive", "u_shape", "uncertain"]


def _registered_reference_fixture(monkeypatch, schema):
    # Test-only installed fixture exercises schema vocabulary absent from the
    # current business constants. There is no public arbitrary registration API.
    monkeypatch.setattr("marvis.llm_outbound._installed_application_schema", lambda contract, bindings: schema)
    return application_schema("test_installed_reference_fixture")


@pytest.mark.parametrize("output_mode", ["json_schema", "json_object", "none"])
def test_installed_local_refs_defs_and_conditional_constraints_reach_wire(monkeypatch, output_mode):
    wire = Wire(monkeypatch)
    wire.client.profile["structured_output"] = output_mode
    schema = {"name": "conditional_fixture", "strict": False, "schema": {
        "type": "object", "$defs": {"direction": {"enum": ["higher_is_riskier", "lower_is_riskier"]}},
        "properties": {"direction": {"$ref": "#/$defs/direction"}, "mode": {"const": "frozen"},
                       "description": {"type": "string"}},
        "if": {"required": ["direction"]}, "then": {"required": ["mode"]},
        "else": {"not": {"required": ["mode"]}}, "additionalProperties": False,
    }}
    _send(wire, _registered_reference_fixture(monkeypatch, schema))
    assert _contract_on_wire(wire) == schema


def test_schema_capability_is_immutable_and_does_not_retain_caller_mutations(monkeypatch):
    source = {"name": "frozen_fixture", "schema": {"type": "string", "enum": ["installed"]}}
    schema = _registered_reference_fixture(monkeypatch, source)
    source["schema"]["enum"].append("PRIVATE_MUTATION")
    assert safe_schema(schema)["schema"]["enum"] == ["installed"]
    detached = schema["schema"]
    detached["enum"].append("PRIVATE_DETACHED_MUTATION")
    assert safe_schema(schema)["schema"]["enum"] == ["installed"]
    with pytest.raises(FrozenInstanceError):
        schema.contract = "other"
    with pytest.raises(OutboundContextError):
        ApplicationSchema("forged", '{"const":"PRIVATE"}')


@pytest.mark.parametrize("bindings", [
    {"allowed_intents": ["PRIVATE_CUSTOM_INTENT"]},
    {"allowed_intents": ["none", "none"]},
    {"allowed_intents": ["none"], "schema": {"const": "PRIVATE_BODY"}},
])
def test_caller_cannot_override_installed_intent_contract(monkeypatch, bindings):
    wire = Wire(monkeypatch)
    with pytest.raises(OutboundContextError):
        application_schema("semantic_intent", **bindings)
    assert wire.requests == []


def test_arbitrary_dictionary_cannot_be_blessed_as_installed_schema(monkeypatch):
    wire = Wire(monkeypatch)
    with pytest.raises(OutboundContextError):
        application_schema({"name": "private", "schema": {"const": "PRIVATE_BODY"}})
    assert wire.requests == []


def test_unowned_annotations_are_removed_without_forwarding_private_text(monkeypatch):
    wire = Wire(monkeypatch)
    schema = {"name": "caller_safe_subset", "description": "PRIVATE_BODY", "schema": {
        "type": "object", "properties": {"choice": {"type": "string", "enum": ["confirm", "halt"],
        "description": "PRIVATE_BODY", "examples": ["PRIVATE_BODY"], "default": "PRIVATE_BODY"}},
        "required": ["choice"], "additionalProperties": False,
    }}
    _send(wire, schema)
    encoded = json.dumps(wire.requests)
    assert "PRIVATE_BODY" not in encoded
    assert _contract_on_wire(wire)["schema"]["properties"]["choice"] == {
        "type": "string", "enum": ["confirm", "halt"],
    }


@pytest.mark.parametrize("schema", [
    {"name": "caller", "schema": {"const": "PRIVATE_BODY"}},
    {"name": "caller", "schema": {"$ref": "https://private.invalid/schema"}},
    {"name": "caller", "schema": {"$defs": {"secret": {"enum": ["PRIVATE_BODY"]}}}},
    {"name": "caller", "schema": {"type": "PRIVATE_BODY"}},
    {"name": "caller", "schema": {"properties": {"x": []}}},
])
def test_invalid_or_unowned_extended_schema_has_zero_requests(monkeypatch, schema):
    wire = Wire(monkeypatch)
    with pytest.raises(LLMClientError) as caught:
        _send(wire, schema)
    assert caught.value.error_kind == LLMClientErrorKind.OUTBOUND_POLICY
    assert "PRIVATE_BODY" not in str(caught.value)
    assert wire.requests == []


@pytest.mark.parametrize("reference", ["https://private.invalid/schema", "#/$defs/missing"])
def test_even_installed_schema_rejects_external_or_unbound_refs(monkeypatch, reference):
    wire = Wire(monkeypatch)
    source = {"name": "reference_fixture", "schema": {"$ref": reference}}
    with pytest.raises(OutboundContextError):
        _registered_reference_fixture(monkeypatch, source)
    assert wire.requests == []


def test_twenty_one_safe_schema_branches_survive_without_truncation(monkeypatch):
    wire = Wire(monkeypatch)
    schema = {"name": "branches", "schema": {"anyOf": [{"type": "string"} for _ in range(21)]}}
    _send(wire, schema)
    assert _contract_on_wire(wire) == schema


@pytest.mark.parametrize("kind", ["depth", "branches", "fields", "nodes", "bytes", "nonfinite", "cycle"])
def test_schema_limits_reject_before_any_protocol_request(monkeypatch, kind):
    wire = Wire(monkeypatch)
    if kind == "depth":
        inner = {"type": "string"}
        for _ in range(40):
            inner = {"type": "object", "properties": {"child": inner}}
    elif kind == "branches":
        inner = {"anyOf": [{"type": "string"} for _ in range(257)]}
    elif kind == "fields":
        inner = {"type": "object", "properties": {f"field_{index}": {} for index in range(10001)}}
    elif kind == "nodes":
        inner = {"properties": {f"field_{index}": {"enum": ["confirm"] * 1000} for index in range(101)}}
    elif kind == "bytes":
        inner = {"type": "string", "description": "x" * 1000001}
    elif kind == "cycle":
        inner = {}
        inner["properties"] = {"child": inner}
    else:
        inner = {"type": "number", "minimum": float("nan")}
    with pytest.raises(LLMClientError) as caught:
        _send(wire, {"name": "excessive", "schema": inner})
    assert caught.value.error_kind == LLMClientErrorKind.OUTBOUND_POLICY
    assert wire.requests == []
