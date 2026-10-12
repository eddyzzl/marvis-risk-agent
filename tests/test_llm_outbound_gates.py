import json
from pathlib import Path

import pandas as pd
import pytest

from marvis.agent.auto_drive import decide_gate
from marvis.agent.modeling_setup import _sample_weight_diagnostics
from marvis.llm_outbound import OutboundContextError, gate_instruction_prompt
from tests.test_llm_outbound import Wire

PRIVATE = "PRIVATE_SAMPLE_OR_REPORT_915"


def test_unknown_extension_risk_labels_stay_local_while_human_guard_still_applies(monkeypatch):
    wire = Wire(monkeypatch)
    wire.reply = {"action": "confirm", "reason": "继续", "confidence": .9}
    decision = decide_gate(wire.client, gate={"metadata": {"gate_envelope": {
        "kind": "gate", "risk_flags": ["requires_human_" + PRIVATE],
        "allowed_actions": ["confirm", "halt"],
    }}})
    assert decision["action"] == "halt"
    assert PRIVATE not in json.dumps(wire.requests)
    assert wire.last_context()["gate"]["unrecognized_risk_flag_count"] == 1


def test_rule_based_current_split_is_not_silently_presented_as_an_empty_config():
    with pytest.raises(OutboundContextError, match="切分含规则条件"):
        gate_instruction_prompt({}, "按当前配置继续", param_schema=[{
            "name": "split_config", "type": "object",
            "current": {"rules": [{"column": "segment", "value": PRIVATE, "split": "train"}]},
        }])


def test_native_screen_shape_keeps_risk_metrics_but_not_subgroup_values(monkeypatch):
    from types import SimpleNamespace
    from marvis.agent.gate_payloads import build_screen_payload

    screen = build_screen_payload(dep=SimpleNamespace(id="screen-step", title=PRIVATE, inputs={"leakage_ks": .4}), output={
        "selected": ["x"], "ranked": [["x", .2]], "leakage": [["target_copy", None, PRIVATE]],
        "suspected": [["old_score", .31, PRIVATE]], "unusable": [["all_null", PRIVATE]],
        "psi_watch": [["x", .36, PRIVATE]], "scores": {"x": {"ks": .2, "missing_rate": .1,
            "deterministic_outcome_value": .99876543, "note": PRIVATE}}, "n_screened": 4,
    })
    wire = Wire(monkeypatch)
    wire.reply = {"action": "halt", "reason": "需复核风险"}
    decide_gate(wire.client, gate={"metadata": {"screen": screen}})
    assert PRIVATE not in json.dumps(wire.requests)
    assert ".99876543" not in json.dumps(wire.requests)
    sent = wire.last_context()["metadata"]["screen"]
    assert sent["leakage"] == [{"feature": "target_copy", "ks": None}]
    assert sent["suspected"] == [{"feature": "old_score", "ks": .31}]
    assert sent["unusable"] == [{"feature": "all_null"}]
    assert sent["psi_watch"] == [{"feature": "x", "psi_split": .36}]
    assert sent["thresholds"]["leakage_ks"] == .4
    assert sent["feature_metrics"] == [{"feature": "x", "ks": .2, "missing_rate": .1}]


def test_gate_adapter_native_risk_evidence_matches_local_checklist_and_reaches_wire(monkeypatch):
    from types import SimpleNamespace
    from marvis.agent.gate_adapters import render_gate_dependencies
    from marvis.plugins.manifest import ToolRef

    deps = [SimpleNamespace(id="tune", title=PRIVATE, output_ref=None, inputs={}, tool_ref=ToolRef("modeling", "tune_hyperparameters")),
            SimpleNamespace(id="train", title=PRIVATE, output_ref=None, inputs={}, tool_ref=ToolRef("modeling", "train_models"))]
    outputs = {"tune": {"trials": [{"train_ks": .5, "test_ks": .35}]},
               "train": {"failed": [{"recipe": "catboost", "error": PRIVATE}]}}
    gate = SimpleNamespace(depends_on=["tune", "train"], tool_ref=ToolRef("modeling", "select_experiment"))
    rendered = render_gate_dependencies(SimpleNamespace(steps=deps), gate, outputs.get,
                                       render_output=lambda *_: ("", []))
    assert any("过拟合迹象" in text for text in rendered.red_flags)
    assert any("1 个候选算法失败" in text for text in rendered.red_flags)
    wire = Wire(monkeypatch)
    wire.reply = {"action": "halt", "reason": "需复核候选风险"}
    decide_gate(wire.client, gate={"metadata": {"red_flags": rendered.red_flags,
                                              "red_flag_evidence": rendered.red_flag_evidence}})
    assert PRIVATE not in json.dumps(wire.requests)
    assert wire.last_context()["metadata"]["red_flag_evidence"] == rendered.red_flag_evidence
    assert rendered.red_flag_evidence == [{"code": "train_test_ks_gap", "ks_gap": .5 - .35, "threshold": .1},
                                         {"code": "failed_candidates", "count": 1, "recipes": ["catboost"]}]


def test_native_weight_diagnostics_and_control_defaults_reach_gate_wire(monkeypatch):
    class Backend:
        def sample_rows(self, path, count, seed):
            return pd.DataFrame({"target": [0, 0, 1, 1], "weight": [1, 1, 2, 2],
                                 "bad_weight": [1, None, 0, 2], "record": [PRIVATE] * 4})

    diagnostics = _sample_weight_diagnostics(Backend(), Path("unused"), target_col="target", split_col=None)
    wire = Wire(monkeypatch)
    wire.reply = {"action": "confirm", "reason": "仍需关注标签相关性", "confidence": .9}
    gate = {"metadata": {"modeling_setup": {
        "target_type": "binary", "recipes": ["lgb", "lr"], "eligible_algorithms": ["lgb", "lr"],
        "n_trials": 12, "feature_count": 9, "sample_weight_col": "",
        "sample_weight_candidates": ["weight", "bad_weight"], "sample_weight_diagnostics": diagnostics,
        "split_config": {"oot_by_time": "applyDate", "group_cols": ["applicant_id"], "test_size": .25},
        "split_summary": {"split_counts": {"train": 80, "test": 10, "oot": 10}, "total_rows": 100},
    }, "tables": [{"rows": [[PRIVATE]]}]}}
    decide_gate(wire.client, gate=gate)
    payload = wire.last_context()
    assert PRIVATE not in json.dumps(wire.requests)
    setup = payload["metadata"]["modeling_setup"]
    valid, invalid = setup["sample_weight_diagnostics"]
    assert valid["valid"] is True and valid["leakage_risk"] == "high"
    assert valid["target_correlation"] == 1.
    assert valid["rows_sampled"] == 4 and valid["non_missing"] == 4
    assert invalid["valid"] is False and invalid["reason_code"] == "missing_or_non_numeric"
    assert invalid["missing_rate"] == .25
    assert setup["split_config"] == gate["metadata"]["modeling_setup"]["split_config"]
    assert setup["sample_weight_col"] == ""
    controls = {item["id"]: item for item in payload["gate"]["controls"]}
    assert controls["n_trials"]["default"] == 12
    assert controls["recipes"]["default"] == ["lgb", "lr"]
    assert controls["recipes"]["schema"]["enum"] == ["lgb", "lr"]
    assert controls["sample_weight_col"]["default"] == ""
    assert controls["sample_weight_col"]["schema"]["enum"] == ["", "weight", "bad_weight"]


def test_split_control_context_preserves_random_vs_time_and_bounds_without_prose():
    specs = [{"name": "split_config", "type": "object", "current": {
        "random_oot": False, "oot_by_time": "applyDate", "oot_size": .2, "test_size": .25,
        "group_cols": ["applicant_id"], "notes": PRIVATE,
    }}, {"name": "recipes", "type": "array", "current": ["lr"], "bounds": {"enum": ["lr", "lgb"]}}]
    prompt = gate_instruction_prompt({}, "按当前配置继续", param_schema=specs)
    controls = json.loads(prompt)["param_schema"]
    assert PRIVATE not in prompt
    assert controls[0]["current"] == {key: value for key, value in specs[0]["current"].items() if key != "notes"}
    assert controls[1]["bounds"] == {"enum": ["lr", "lgb"]}


def test_gate_retry_preserves_the_complete_control_context_without_rejected_text(monkeypatch):
    wire = Wire(monkeypatch)
    wire.replies = [PRIVATE, {"action": "confirm", "reason": "核对通过", "confidence": .9}]
    assert decide_gate(wire.client, gate={"metadata": {"modeling_setup": {
        "target_type": "binary", "n_trials": 3, "recipes": ["lr"],
    }}})["action"] == "confirm"
    assert len(wire.requests) == 2
    first = json.loads(wire.requests[0]["messages"][1]["content"])
    second = wire.last_context()
    assert "无法解析" in second.pop("review_instruction")
    assert first == second
    assert PRIVATE not in json.dumps(wire.requests)
