"""Exact installed Tool contracts on the protocol body, without providers."""

from dataclasses import replace
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from marvis.db import PluginRepository, init_db
from marvis.llm_client import estimate_tokens
from marvis.llm_outbound import OutboundContextError, planning_prompt, safe_schema
from marvis.llm_outbound_catalog import catalog_schema_contracts, restore_planning_catalog
from marvis.orchestrator.planner import (
    PLAN_SYS,
    PLANNER_CONTEXT_SAFETY_MARGIN_TOKENS,
    PLANNER_MIN_OUTPUT_TOKENS,
    PlannerConstraints,
    _fit_planner_request,
    build_plan_prompt,
    compact_catalog_for_prompt,
)
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef, parse_manifest
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from tests.test_llm_outbound import Wire


def _registry(tmp_path):
    path = tmp_path / "catalog.sqlite"
    init_db(path)
    plugins = PluginRegistry(PluginRepository(path))
    return plugins, ToolRegistry(plugins)


def _dynamic_manifest(schema=None):
    schema = schema or {
        "type": "object", "description": "PRIVATE_SCHEMA_DESCRIPTION",
        "$defs": {"mode": {"enum": ["frozen", "adaptive"], "examples": ["PRIVATE_EXAMPLE"]}},
        "properties": {
            "mode": {"$ref": "#/$defs/mode"},
            "threshold": {"type": "number", "minimum": .1, "maximum": .8},
            "description": {"type": "string", "maxLength": 20},
        },
        "if": {"properties": {"mode": {"const": "frozen"}}, "required": ["mode"]},
        "then": {"required": ["threshold"]}, "else": {"not": {"required": ["threshold"]}},
        "additionalProperties": False,
    }
    return parse_manifest({
        "name": "approved_dynamic", "version": "0.1.0", "module": "approved_dynamic.tools",
        "checksum": "a" * 64, "permissions": [], "tools": [{
            "name": "bounded_tool", "summary": "PRIVATE_TOOL_SUMMARY",
            "input_schema": schema,
            "output_schema": {"type": "object", "properties": {"accepted": {"const": True}},
                              "required": ["accepted"], "additionalProperties": False},
            "determinism": "deterministic", "timeout_seconds": 10,
            "failure_policy": "fail", "entrypoint": "tool_bounded", "side_effects": [],
        }],
    }, builtin=False)


def _send(wire, prompt):
    wire.client.complete(system_prompt=PLAN_SYS, user_prompt=prompt,
                         stream=False, json_schema={"name": "plan", "schema": {"type": "object"}})
    return json.loads(wire.requests[-1]["messages"][1]["content"])


@pytest.mark.parametrize("output_mode", ["json_schema", "json_object", "none"])
def test_packaged_catalog_preserves_full_input_output_refs_consts_and_enums(tmp_path, monkeypatch, output_mode):
    plugins, registry = _registry(tmp_path)
    load_builtin_packs(plugins, Path(__file__).parents[1] / "marvis" / "packs")
    names = {"train_model_with_evidence_v2", "materialize_model_score_evidence_v2"}
    catalog = [item for item in registry.catalog_for_planner()
               if item["plugin"] == "modeling" and item["tool"] in names]
    assert len(catalog) == 2
    wire = Wire(monkeypatch)
    wire.client.profile["structured_output"] = output_mode
    payload = _send(wire, build_plan_prompt("核验模型证据后评估策略", catalog, {}, {}, None))
    by_name = {item["tool"]: item for item in payload["available_tools"]}
    for entry in catalog:
        tool = registry.resolve(ToolRef(entry["plugin"], entry["tool"], entry["version"]))
        assert by_name[entry["tool"]]["input_schema"] == tool.input_schema
        assert by_name[entry["tool"]]["output_schema"] == tool.output_schema
    train = by_name["train_model_with_evidence_v2"]
    assert train["input_schema"]["properties"]["recipe"]["enum"] == ["lgb", "xgb", "catboost", "lr", "scorecard", "mlp"]
    assert train["output_schema"]["properties"]["governance"]["properties"]["not_deployed"]["const"] is True
    score = by_name["materialize_model_score_evidence_v2"]
    assert score["input_schema"]["properties"]["training_evidence_ref"]["$ref"] == "#/$defs/training_evidence_ref"
    assert "training_evidence_ref" in score["input_schema"]["$defs"]


def test_audited_dynamic_installation_preserves_constraints_but_withholds_annotations(tmp_path, monkeypatch):
    plugins, registry = _registry(tmp_path)
    manifest = _dynamic_manifest()
    plugins.register(manifest)
    catalog = registry.catalog_for_planner()
    assert set(catalog_schema_contracts(catalog[0])) == {"input_schema", "output_schema"}
    wire = Wire(monkeypatch)
    payload = _send(wire, build_plan_prompt("使用冻结模式，阈值为0.4", catalog, {}, {}, None))
    schema = payload["available_tools"][0]["input_schema"]
    assert schema["$defs"]["mode"]["enum"] == ["frozen", "adaptive"]
    assert schema["if"]["properties"]["mode"]["const"] == "frozen"
    assert schema["then"] == {"required": ["threshold"]}
    assert schema["else"] == {"not": {"required": ["threshold"]}}
    assert schema["properties"]["description"] == {"type": "string", "maxLength": 20}
    assert "PRIVATE_" not in json.dumps(wire.requests)
    # The runtime still validates the same full installed constraint.
    runtime_schema = registry.resolve(ToolRef(manifest.name, "bounded_tool")).input_schema
    assert Draft202012Validator(runtime_schema).is_valid({"mode": "frozen", "threshold": .4})
    assert not Draft202012Validator(runtime_schema).is_valid({"mode": "frozen"})
    assert not Draft202012Validator(runtime_schema).is_valid({"mode": "adaptive", "threshold": .4})


def test_plain_bytes_and_unproven_builtin_flag_do_not_receive_catalog_authority(tmp_path, monkeypatch):
    plugins, registry = _registry(tmp_path)
    manifest = _dynamic_manifest()
    plugins.register(replace(manifest, builtin=True, checksum=""))
    item = registry.catalog_for_planner()[0]
    assert catalog_schema_contracts(item) == {}
    wire = Wire(monkeypatch)
    payload = _send(wire, build_plan_prompt("分析工具输入", [item], {}, {}, None))
    assert "input_schema" not in payload["available_tools"][0]
    assert "PRIVATE_" not in json.dumps(wire.requests)
    plugins.set_enabled(manifest.name, False)
    assert registry.catalog_for_planner() == []


def test_repository_row_without_installation_audit_has_no_schema_authority(tmp_path):
    plugins, registry = _registry(tmp_path)
    manifest = _dynamic_manifest()
    plugins._repo.upsert_plugin(manifest, enabled=True)
    plugins.load_from_db()
    assert catalog_schema_contracts(registry.catalog_for_planner()[0]) == {}


def test_catalog_contract_snapshots_ignore_later_dictionary_mutation_and_reject_identity_change(tmp_path):
    plugins, registry = _registry(tmp_path)
    plugins.register(_dynamic_manifest())
    catalog = registry.catalog_for_planner()
    catalog[0]["input_schema"]["$defs"]["mode"]["enum"].append("PRIVATE_MUTATION")
    compact = compact_catalog_for_prompt(catalog)
    assert compact[0]["input_schema"]["$defs"]["mode"]["enum"] == ["frozen", "adaptive"]
    assert catalog_schema_contracts(dict(catalog[0])) == {}
    catalog[0]["version"] = "9.0.0"
    with pytest.raises(OutboundContextError):
        compact_catalog_for_prompt(catalog)


def test_budget_replay_keeps_full_contract_or_explicitly_fails_without_truncating_schema(tmp_path, monkeypatch):
    plugins, registry = _registry(tmp_path)
    plugins.register(_dynamic_manifest())
    ref = ToolRef("approved_dynamic", "bounded_tool")
    prompt = build_plan_prompt("如果核验通过再使用冻结模式，阈值0.4", registry.catalog_for_planner(), {}, {}, None,
                               constraints=PlannerConstraints(required_tool_refs=(ref,)))
    original = json.loads(prompt)
    wire = Wire(monkeypatch)
    wire.client.profile["context_window"] = (
        estimate_tokens(PLAN_SYS) + estimate_tokens(prompt)
        + PLANNER_CONTEXT_SAFETY_MARGIN_TOKENS + PLANNER_MIN_OUTPUT_TOKENS - 5
    )
    wire.client.profile["max_output_tokens"] = PLANNER_MIN_OUTPUT_TOKENS
    fitted, cap, truncated = _fit_planner_request(wire.client, system_prompt=PLAN_SYS, user_prompt=prompt)
    assert truncated and cap >= PLANNER_MIN_OUTPUT_TOKENS
    payload = _send(wire, fitted)
    assert payload["goal"] == original["goal"]
    for field in ("input_schema", "output_schema"):
        assert payload["available_tools"][0][field] == original["available_tools"][0][field]
    altered = json.loads(prompt)
    altered["available_tools"][0]["input_schema"]["then"] = {}
    with pytest.raises(OutboundContextError):
        restore_planning_catalog(altered, prompt)


def test_json_round_trip_cannot_reauthorize_unowned_full_catalog_schemas(tmp_path):
    plugins, registry = _registry(tmp_path)
    plugins.register(_dynamic_manifest())
    prompt = build_plan_prompt("保留冻结约束", registry.catalog_for_planner(), {}, {}, None)
    unowned = planning_prompt(json.loads(prompt))
    assert "input_schema" not in json.loads(unowned)["available_tools"][0]
    restored = restore_planning_catalog(json.loads(prompt), prompt)
    approved = planning_prompt(restored)
    assert json.loads(approved)["available_tools"][0]["input_schema"] == safe_schema(
        catalog_schema_contracts(registry.catalog_for_planner()[0])["input_schema"]
    )


@pytest.mark.parametrize("schema", [
    {"$ref": "https://private.invalid/contract"},
    {"$ref": "#/$defs/missing"},
    {"anyOf": [{"type": "string"}] * 257},
    {"type": "object", "x-internal-source": "TEST_PRIVATE_INTERNAL_REPORT_BODY"},
    {"type": "object", "properties": {"mode": {"enum": ["OPENAI_API_KEY=TEST_ONLY_VALUE"]}}},
    {"type": "object", "properties": {"mode": {"const": "DB_PASSWORD=TEST_ONLY_VALUE"}}},
    {"type": "string", "format": "TEST_PRIVATE_INTERNAL_REPORT_BODY"},
    {"type": "string", "$schema": "https://private.invalid/contract"},
    {"type": "object", "properties": {"value": {"$ref": "#/$defs/scoped"}}, "$defs": {
        "leaf": {"type": "number"}, "scoped": {"$id": "https://private.invalid/subresource", "$ref": "#/$defs/leaf"},
    }},
])
def test_invalid_installed_tool_schema_refuses_before_protocol_request(tmp_path, monkeypatch, schema):
    plugins, registry = _registry(tmp_path)
    plugins.register(_dynamic_manifest(schema))
    wire = Wire(monkeypatch)
    with pytest.raises(OutboundContextError):
        registry.catalog_for_planner()
    assert wire.requests == []
