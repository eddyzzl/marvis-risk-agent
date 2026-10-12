"""Installed split grammar reaches the actual model protocol with no raw data."""
import json

import jsonschema
import pandas as pd
import pytest

from marvis.agent.adjust_specs import adjust_param_error, normalize_adjust_params
from marvis.agent.instruction_router import route_instruction
from marvis.agent.semantic_authorization import review_semantic_authorization
from marvis.llm_outbound import gate_instruction_prompt
from marvis.modeling_split_schema import split_configuration_schema
from marvis.packs.modeling.prepare import _make_split
from tests.test_llm_outbound import Wire


def test_exact_whole_population_aliases_have_real_60_20_20_native_membership():
    normalized = normalize_adjust_params({"split_config": {
        "method": "random", "train_ratio": .6, "test_ratio": .2, "oot_ratio": .2,
    }})
    assert normalized == {"split_config": {"test_size": .25, "oot_size": .2, "random_oot": True}}
    assert adjust_param_error(normalized) is None
    source = pd.DataFrame({"x": range(100)})
    output = _make_split(source, normalized["split_config"], seed=7)
    assert output["split"].value_counts().to_dict() == {"train": 60, "test": 20, "oot": 20}
    assert output["x"].tolist() == source["x"].tolist()
    canonical = _make_split(source, {"random_oot": True, "test_size": .2, "oot_size": .2}, seed=7)
    assert canonical["split"].value_counts().to_dict() == {"train": 64, "test": 16, "oot": 20}


@pytest.mark.parametrize("config", [
    {"method": "random", "oot_by_time": "date"},
    {"method": "time", "date_col": "date", "random_oot": True},
    {"method": "time", "date_col": "date", "random_oot": "false"},
    {"method": "random", "random_oot": False},
    {"oot_by_time": "date_a", "date_col": "date_b"},
    {"test_size": .2, "test_ratio": "0.2"},
    {"method": "existing", "date_col": "date"},
    {"method": []},
])
def test_conflicting_or_untyped_aliases_are_not_removed_or_executed(config):
    params = normalize_adjust_params({"split_config": config})
    assert params["split_config"] == config
    assert adjust_param_error(params)


def test_unsupported_partition_change_is_retained_and_refused():
    params = normalize_adjust_params({"split_col": "cohort_b", "split_config": {"method": "existing"}})
    assert params == {"split_col": "cohort_b", "split_config": {"method": "existing"}}
    assert "不支持调整 split_col" in adjust_param_error(params)


def test_installed_split_schema_rejects_bool_unknown_and_conflicting_controls():
    schema = split_configuration_schema()
    for valid in ({}, {"method": "existing"}, {"oot_by_time": "date", "random_oot": False}, {"random_oot": True, "test_size": .25}):
        jsonschema.validate(valid, schema)
    for invalid in ({"oot_by_time": True}, {"method": "existing", "test_size": .3},
                    {"oot_by_time": "date", "random_oot": True}, {"unknown": 1},
                    {"rules": [{"when": [{"col": "x", "op": "eq", "val": [1]}], "assign": "test"}]}):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, schema)


def test_installed_grammar_and_existing_proposal_reach_router_and_independent_review_wire(monkeypatch):
    wire = Wire(monkeypatch)
    instruction = "使用现有 split 的 train/test/oot，不重新划分。"
    route_reply = {"action": "adjust", "params": {"split_config": {"method": "existing"}},
                   "constraint": "", "reason": "保留原分区", "confidence": "high", "explicit_authorization": False}
    wire.replies = [route_reply, route_reply, {
        "verdict": "authorize", "evidence_quote": instruction, "reason": "原指令明确使用现有分区",
        "confidence": "high", "is_question": False, "is_conditional": False,
        "requests_change": False, "withholds_authorization": False,
    }]
    specs = [{"name": "split_config", "type": "object", "current": {"method": "existing"},
              "schema": {"description": "PRIVATE_SCHEMA_TEXT", "properties": {"PRIVATE_FIELD": {"examples": ["RAW_RECORD"]}}}}]
    route = route_instruction(wire.client, gate_context={"kind": "confirmation"}, instruction=instruction, param_schema=specs)
    assert route["params"] == {"split_config": {"method": "existing"}}
    review_semantic_authorization(wire.client, gate_context={"kind": "confirmation"}, instruction=instruction,
                                  proposed_params=route["params"])
    assert len(wire.requests) == 3
    for request in wire.requests[:2]:
        schema = request["response_format"]["json_schema"]["schema"]["properties"]["params"]["properties"]["split_config"]
        assert schema == split_configuration_schema()
        context = json.loads(request["messages"][1]["content"])
        control = context["param_schema"][0]
        assert control["schema"] == schema
        assert control["current"] == {"method": "existing"}
        assert control["semantics"]["test_size_basis"] == "non_oot_remainder"
    final = json.loads(wire.requests[-1]["messages"][1]["content"])
    assert final["proposed_params"]["split_config"] == {"method": "existing"}
    assert all(value not in json.dumps(wire.requests) for value in ("PRIVATE_SCHEMA_TEXT", "PRIVATE_FIELD", "RAW_RECORD"))


@pytest.mark.parametrize("current", [{"oot_by_time": True}, {"oot_by_time": "date", "random_oot": True}, {"method": "unknown"}])
def test_invalid_current_split_context_is_zero_request_clarification(monkeypatch, current):
    wire = Wire(monkeypatch)
    route = route_instruction(wire.client, gate_context={}, instruction="按当前分区继续",
                              param_schema=[{"name": "split_config", "type": "object", "current": current}])
    assert route["action"] == "clarify"
    assert wire.requests == []


def test_unknown_uninstalled_schema_annotations_cannot_override_fixed_control():
    payload = json.loads(gate_instruction_prompt({}, "使用现有分区", param_schema=[{
        "name": "split_config", "type": "object", "current": {"method": "existing"},
        "schema": {"type": "string", "enum": ["private arbitrary value"]},
    }]))
    assert payload["param_schema"][0]["schema"] == split_configuration_schema()
