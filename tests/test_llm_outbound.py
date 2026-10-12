import json

import pytest

from marvis.agent.auto_drive import decide_gate
from marvis.llm_client import LLMClientError, LLMClientErrorKind, OpenAICompatibleLLMClient
from marvis.llm_outbound import outbound_prompt
from marvis.llm_prompts import PLAN_SYS
from marvis.orchestrator.planner import _fit_planner_request, build_plan_prompt
from marvis.orchestrator.reviewer import Reviewer
from marvis.packs.modeling.report_tools import _draft_report_narratives
from tests.test_orch_reviewer import _plan, _step


PRIVATE = ["RAW_ROW_PERSON_739", "RAW_SAMPLE_VALUE_814", "RAW_MODEL_BODY_925",
           "RAW_FILE_CONTENT_362", "RAW_ERROR_TEXT_651"]


class Wire:
    def __init__(self, monkeypatch, *, text_reply=False):
        self.requests = []
        self.text_reply = text_reply
        self.reply = {}
        self.replies = []
        monkeypatch.setattr("marvis.llm_client.urlopen", self.open)
        self.client = OpenAICompatibleLLMClient({"api_base_url": "https://synthetic.invalid/v1",
            "api_key": "test-only", "model_name": "synthetic", "structured_output": "json_schema"})

    def open(self, request, **kwargs):
        self.requests.append(json.loads(request.data))
        reply = self.replies.pop(0) if self.replies else self.reply
        content = reply if self.text_reply else json.dumps(reply)
        payload = json.dumps({"choices": [{"message": {"content": content}}]}).encode()

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def __iter__(self):
                return iter([payload])

        return Response()

    def last_context(self):
        encoded = json.dumps(self.requests, ensure_ascii=False)
        for secret in PRIVATE:
            assert secret not in encoded
        return json.loads(self.requests[-1]["messages"][1]["content"])


def _tainted():
    return {"rows": [[PRIVATE[0]]], "columns": [{"name": "x", "dtype": "float64",
        "sample_values": [PRIVATE[1]]}], "model_bytes": PRIVATE[2], "file_content": PRIVATE[3],
        "error": PRIVATE[4], "unknown_business_container": {"name": PRIVATE[0]},
        "row_count": 160, "metrics": {"train_ks": .42, "test_auc": .71}}


@pytest.mark.parametrize("kind", ["critique", "summary"])
def test_reviewer_repair_preserves_frozen_evidence_without_echoing_rejected_text(monkeypatch, kind):
    wire = Wire(monkeypatch)
    wire.replies = [" ".join(PRIVATE), {"passed": True, "reasons": []} if kind == "critique"
                    else {"summary": "仅解释当前指标"}]
    reviewer = Reviewer(lambda: wire.client)
    if kind == "critique":
        assert reviewer.llm_critique(_step([]), _tainted(), "验证").passed
    else:
        reviewer.final_review(_plan(_step([])), {"step-1": _tainted()}, "验证")
    assert len(wire.requests) == 2
    first = json.loads(wire.requests[0]["messages"][1]["content"])
    second = wire.last_context()
    repair = second.pop("response_repair")
    assert repair["kind"] == kind
    assert "same schema" in repair["instruction"]
    assert second == first


def test_planner_wire_keeps_structure_and_metrics_but_not_samples_or_errors(monkeypatch):
    wire = Wire(monkeypatch)
    prompt = build_plan_prompt("比较当前模型", [{"plugin": "modeling", "tool": "train_model",
        "summary": PRIVATE[3], "input_schema": {"type": "object", "properties": {
            "dataset_id": {"type": "string", "description": PRIVATE[1]}}}}], {}, _tainted(), "不要改变训练数据")
    prompt, cap, _ = _fit_planner_request(wire.client, system_prompt=PLAN_SYS.text, user_prompt=prompt)
    assert not any(secret in prompt for secret in PRIVATE)
    wire.client.complete(system_prompt=PLAN_SYS.text, user_prompt=prompt, max_tokens=cap, stream=False)
    context = wire.last_context()
    assert context["task_context"]["row_count"] == 160
    assert context["task_context"]["columns"] == [{"dtype": "float64", "name": "x"}]
    assert context["available_tools"][0]["tool"] == "train_model"
    assert context["task_context"]["metrics"]["test_auc"] == .71


def test_gate_wire_omits_display_tables_and_preserves_safe_decision_context(monkeypatch):
    wire = Wire(monkeypatch)
    wire.reply = {"action": "confirm", "reason": "结构检查通过", "confidence": .9}
    gate = {"content": PRIVATE[3], "metadata": {
        "tables": [{"title": PRIVATE[0], "columns": ["x"], "rows": [[PRIVATE[1]]]}],
        "modeling_setup": {"target_type": "binary", "feature_count": 2,
            "split_summary": {"split_counts": {"train": 120, "test": 40}}},
        "output": _tainted(), "memory_anchor": [PRIVATE[2]], "error": PRIVATE[4]}}
    assert decide_gate(wire.client, gate=gate)["action"] == "confirm"
    context = wire.last_context()
    assert context["metadata"]["modeling_setup"]["feature_count"] == 2
    assert context["gate"]["allowed_actions"] == ["confirm", "adjust", "replan", "clarify", "halt"]


def test_report_and_reviewer_wire_withhold_raw_material(monkeypatch):
    wire = Wire(monkeypatch)
    wire.reply = {key: "仅使用聚合证据" for key in ("sample", "vintage", "model", "stress")}
    from pathlib import Path
    from types import SimpleNamespace
    import pandas as pd
    from marvis.packs.modeling.report_compute import BusinessColumns, compute_sample_analysis
    from marvis.packs.modeling.report_tools import _dataset_split_rows, _stability_rows
    class Backend:
        def read_frame(self, path, columns):
            return pd.DataFrame({"loan_month": ["2026-01", "2026-01", "2026-02", "2026-02"],
                                 "target": [0, 1, 1, 1]})[columns]
    metrics = SimpleNamespace(train_rmse=None, train_ks=.61, train_auc=.81,
        test_ks=.41, test_auc=.71, oot_ks=.21, oot_auc=.61,
        psi_test_vs_train=.15, psi_oot_vs_train=.32, overfit_flag=True)
    summary = {"sample_analysis": compute_sample_analysis(Backend(), Path("/unused"),
        loan_month_col="loan_month", target_col="target", business=BusinessColumns(), mob_cols=()),
        "dataset_split": _dataset_split_rows(metrics), "stability": _stability_rows(metrics),
        "project_meta": _tainted()}
    result = _draft_report_narratives(summary, llm_factory=lambda: wire.client)
    assert result["sample"] == "仅使用聚合证据"
    report = wire.last_context()["structured_summary"]
    assert report["sample_analysis"][0] == {"month": "2026-01", "row_count": 2,
        "bad_count": 1, "good_count": 1, "bad_rate": .5}
    assert [item["split"] for item in report["dataset_split"]] == ["train", "test", "oot"]
    assert report["stability"][0] == {"metric": "psi_test_vs_train", "value": .15}
    wire.reply = {"passed": True, "reasons": []}
    reviewer = Reviewer(lambda: wire.client)
    assert reviewer.llm_critique(_step([]), _tainted(), "modeling " + PRIVATE[0]).passed
    assert wire.last_context()["output_summary"]["metrics"]["test_auc"] == .71
    wire.reply = {"summary": "结构审阅完成"}
    reviewer.final_review(_plan(_step([])), {"step-1": _tainted()}, "validation " + PRIVATE[0])
    wire.last_context()


def test_unknown_free_text_and_schema_examples_never_enter_transport(monkeypatch):
    wire = Wire(monkeypatch)
    wire.client.complete(system_prompt=" ".join(PRIVATE), user_prompt=outbound_prompt({}),
        json_schema={"name": "safe_result", "schema": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["confirm", PRIVATE[0]],
                       "description": PRIVATE[1], "default": PRIVATE[2], "examples": [PRIVATE[3]]}},
            "description": PRIVATE[4]}}, stream=False)
    wire.last_context()
    assert wire.requests[-1]["response_format"]["json_schema"]["schema"]["properties"]["action"]["enum"] == ["confirm"]


@pytest.mark.parametrize("field", ["messages", "tools", "metadata", "response_format", "model"])
def test_extra_fields_cannot_override_guarded_body(monkeypatch, field):
    wire = Wire(monkeypatch)
    wire.client.profile["extra_request_fields"] = {field: PRIVATE[0]}
    with pytest.raises(LLMClientError) as failure:
        wire.client.complete(system_prompt="s", user_prompt=outbound_prompt({}), stream=False)
    assert failure.value.error_kind == LLMClientErrorKind.OUTBOUND_POLICY
    assert not wire.requests
    assert PRIVATE[0] not in str(failure.value)


def test_projection_is_idempotent_and_unknown_nested_values_are_dropped():
    once = outbound_prompt({"task_context": _tainted(), "features": ["x"], "seed": 19})
    assert outbound_prompt(once) == once
    assert not any(secret in once for secret in PRIVATE)


def test_arbitrary_prose_json_cannot_be_mistaken_for_structural_context(monkeypatch):
    from marvis.agent.instruction_router import route_instruction
    wire = Wire(monkeypatch)
    text = '请分析客户记录：{"name":"RAW_ROW_PERSON_739","id":"RAW_SAMPLE_VALUE_814","score":612}'
    result = route_instruction(wire.client, gate_context="gate", instruction=text)
    assert result["action"] == "clarify"
    assert not wire.requests
    with pytest.raises(LLMClientError):
        wire.client.complete(system_prompt="s", user_prompt=text)
    assert not wire.requests


def test_authorization_retains_opposite_control_instructions(monkeypatch):
    from marvis.agent.semantic_authorization import review_semantic_authorization
    wire = Wire(monkeypatch)
    for instruction in ("立即按当前方案继续", "禁止继续"):
        review_semantic_authorization(wire.client, gate_context={"kind": "confirmation"},
            instruction=instruction, proposed_params={})
    contexts = [json.loads(request["messages"][1]["content"]) for request in wire.requests]
    assert [context["instruction"] for context in contexts] == ["立即按当前方案继续", "禁止继续"]
    schema = wire.requests[-1]["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["verdict"]["enum"] == ["authorize", "reject", "ambiguous"]


def test_bare_customer_record_cannot_masquerade_as_reviewer_structure(monkeypatch):
    wire = Wire(monkeypatch)
    wire.reply = {"passed": True, "reasons": []}
    record = {"name": PRIVATE[0], "id": PRIVATE[1], "score": 612,
              "status": PRIVATE[2], "type": PRIVATE[3], "mode": PRIVATE[4]}
    Reviewer(lambda: wire.client).llm_critique(_step([]), record, "validation")
    assert wire.last_context()["output_summary"] == {}
    assert "612" not in wire.requests[-1]["messages"][1]["content"]
    safe = json.loads(outbound_prompt({"steps": [{"id": "step-1", "name": PRIVATE[0]}],
        "columns": [{"name": "x", "dtype": "float64", "id": PRIVATE[1], "score": 612}]}))
    assert safe["steps"] == [{"id": "step-1"}]
    assert safe["columns"] == [{"name": "x", "dtype": "float64"}]


def test_report_wire_preserves_native_vintage_bands_stress_and_calibration(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from marvis.data.backend import DataBackend
    from marvis.packs.modeling.report_compute import (
        BusinessColumns, compute_vintage_report, stress_low_pricing, resolve_report_sections,
    )
    from marvis.packs.modeling.report_tools import _score_band_rows, _report_structured_summary
    from marvis.packs.modeling.scoring import _artifact_calibration_rows, CALIBRATION_PARAMS_KEY
    from tests.test_modeling_report import _business_frame

    frame = _business_frame()
    frame["split"] = ["train", "train", "test", "test"]
    frame["person"] = PRIVATE[0]
    path = tmp_path / "aggregate-input.parquet"
    frame.to_parquet(path, index=False)
    backend = DataBackend(tmp_path)
    vintage = compute_vintage_report(backend, path, loan_month_col="loan_month",
        mob_observe_cols=("mob1", "mob2", "mob3"), amount_col="amount")
    stress = stress_low_pricing(backend, path, score_col="score", target_col="y", interest_rate_col="rate",
        low_pricing_threshold=None, ratios=(.25, .5))
    bands = _score_band_rows(SimpleNamespace(backend=backend), path, score_col="score", target_col="y",
        config=SimpleNamespace(split_col="split", split_values={"train": "train", "test": "test"}), bin_count=2)
    calibration = _artifact_calibration_rows(SimpleNamespace(params={CALIBRATION_PARAMS_KEY: {
        "fit_split": "train", "eval_split": "test", "evaluated_on": "test", "method": "isotonic",
        "brier_raw": .18, "brier_calibrated": .16, "sample_count": 4, "positive_count": 2,
        "model_bytes": PRIVATE[2],
    }}))
    summary = _report_structured_summary(vintage=vintage, stress_low_pricing=stress,
        score_bands=bands, calibration=calibration, section_status=resolve_report_sections(BusinessColumns(), None),
        scorecard_table=[{"feature": "x", "points": 617, "bin_label": PRIVATE[1]}],
        project_meta={"file_content": PRIVATE[3]}, records=frame.to_dict("records"),
        stress_product_removal={"baseline": {"ks": .61, "sample_count": 4}, PRIVATE[0]: {
            "ks_after": .34, "ks_delta": -.27, "psi_vs_baseline": .12, "error": PRIVATE[4]}},
        feature_importance=[{"feature": "x", "importance": .3, "厂商名称": PRIVATE[0]}])
    wire = Wire(monkeypatch)
    _draft_report_narratives(summary, llm_factory=lambda: wire.client)
    sent = wire.last_context()["structured_summary"]
    assert sent["vintage"]["points"][0]["sample_count"] == vintage["points"][0]["sample_count"]
    assert sent["vintage"]["points"][0]["bad_rate"] == vintage["points"][0]["bad_rate"]
    assert sent["stress_low_pricing"]["scenarios"][0] == {"ratio": .25, **stress["by_ratio"]["0.25"]}
    assert sent["score_bands"][0]["cum_direction"] == "higher_is_riskier"
    assert sent["score_bands"][0]["lift"] == bands[0]["lift"]
    assert sent["score_bands"][0]["bin_edges_source"] == "train"
    assert sent["calibration"][0]["brier_calibrated"] == .16
    assert sent["calibration"][0]["evaluated_on"] == "test"
    assert sent["section_status"][0] == {"section": "sample_analysis", "available": False}
    assert sent["stress_product_removal"]["groups"] == [{"ks_after": .34, "ks_delta": -.27, "psi_vs_baseline": .12}]
    assert sent["feature_importance"] == [{"feature": "x", "importance": .3}]
    assert "scorecard_table" not in sent


def test_report_output_field_schema_survives_and_rejected_enum_is_not_widened(monkeypatch):
    wire = Wire(monkeypatch)
    schema = {"name": "validation_report", "schema": {"type": "object", "properties": {
        "TEXT:final_validation_conclusion": {"type": "string"},
        "untrusted_choice": {"type": "string", "enum": [PRIVATE[0]]},
    }, "required": ["TEXT:final_validation_conclusion"]}}
    wire.client.complete(system_prompt="s", user_prompt=outbound_prompt({}), json_schema=schema, stream=False)
    wire.last_context()
    sent = wire.requests[-1]["response_format"]["json_schema"]["schema"]
    assert sent["required"] == ["TEXT:final_validation_conclusion"]
    assert "TEXT:final_validation_conclusion" in sent["properties"]
    assert sent["properties"]["untrusted_choice"]["enum"] == []


def test_gate_selection_repair_preserves_full_control_and_bound_ids_without_private_names(monkeypatch):
    from marvis.agent.instruction_router import route_instruction
    wire = Wire(monkeypatch)
    wire.replies = [
        {"action": "replan", "params": {}, "constraint": PRIVATE[0], "reason": PRIVATE[4],
         "confidence": "medium", "explicit_authorization": False},
        {"action": "confirm", "params": {"selected_experiment_id": "experiment_lr"}, "constraint": "",
         "reason": "采用推荐模型", "confidence": "high", "explicit_authorization": True},
    ]
    instruction = "采用平台推荐的逻辑回归作为最终模型。"
    result = route_instruction(wire.client, gate_context="选择实验", instruction=instruction,
        param_schema=[{"name": "selected_experiment_id", "type": "string",
                       "enum": ["experiment_lr", "experiment_xgb"], "current": "experiment_lr",
                       "candidates": [{"experiment_id": "experiment_lr", "recipe": "lr_multiclass",
                                       "display_name": PRIVATE[0], "recommended": True}]}])
    assert result["action"] == "confirm"
    assert result["params"] == {"selected_experiment_id": "experiment_lr"}
    sent = wire.last_context()
    assert sent["instruction"] == instruction
    assert sent["param_schema"][0]["candidates"][0]["display_name"] == "逻辑回归（多分类）"
    assert "reason" not in sent["first_pass"]
    schema = wire.requests[-1]["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["params"]["properties"]["selected_experiment_id"]["enum"] == [
        "experiment_lr", "experiment_xgb"]


@pytest.mark.parametrize("instruction", [
    "立即继续，客户 RAW_ROW_PERSON_739 分数 612",
    "n_trials 调到 3；账户 1234567890",
    '继续 {"name":"RAW_ROW_PERSON_739","score":612}',
    "不要继续；文件正文：RAW_FILE_CONTENT_362",
])
def test_control_projection_never_drops_unknown_tail_to_create_authorization(monkeypatch, instruction):
    from marvis.agent.instruction_router import route_instruction
    wire = Wire(monkeypatch)
    result = route_instruction(wire.client, gate_context="选择实验", instruction=instruction)
    assert result["action"] == "clarify"
    assert result["explicit_authorization"] is False
    assert wire.requests == []


def test_large_gate_keeps_complete_typed_structure_within_actual_model_budget(monkeypatch):
    from marvis.llm_client import estimate_tokens
    wire = Wire(monkeypatch)
    wire.client.profile.update(context_window=8192, max_output_tokens=512)
    wire.reply = {"action": "confirm", "reason": "已检查结构", "confidence": .9}
    columns = [{"name": f"feature_{i}_" + "x" * 65, "dtype": "float64"} for i in range(150)]
    decide_gate(wire.client, gate={"content": PRIVATE[0], "metadata": {"columns": columns}})
    context = wire.last_context()
    assert context["metadata"]["columns"] == columns
    assert estimate_tokens(wire.requests[-1]["messages"][1]["content"]) > 8192 * .25


@pytest.mark.parametrize("kwargs,profile", [
    ({"temperature": {"file_contents": PRIVATE[0]}}, {}),
    ({}, {"enable_thinking": True, "thinking_style": "openai_reasoning", "reasoning_effort": PRIVATE[1]}),
])
def test_scalar_provider_knobs_cannot_carry_unapproved_content(monkeypatch, kwargs, profile):
    wire = Wire(monkeypatch)
    wire.client.profile.update(profile)
    with pytest.raises(LLMClientError) as failure:
        wire.client.complete(system_prompt="s", user_prompt=outbound_prompt({}), **kwargs)
    assert failure.value.error_kind == LLMClientErrorKind.OUTBOUND_POLICY
    assert wire.requests == []


def test_full_declared_feature_universe_is_kept_and_unbound_replan_controls_refuse():
    from marvis.llm_outbound import OutboundContextError, planning_prompt
    features = [f"x{i}" for i in range(1120)]
    context = json.loads(planning_prompt({"task_context": {"features": features}}))
    assert context["task_context"]["features"] == features
    with pytest.raises(OutboundContextError, match="工具未完整保留"):
        planning_prompt({
            "task_context": {"features": features},
        "remaining_steps": [{"id": "report", "inputs": {
            "metrics": "$ref:analysis.output.metrics", "file": "$ref:analysis.output.report_path",
            "raw": PRIVATE[0]}}, {"id": "analysis", "inputs": {"metrics": ["iv", "ks", "auc", "coverage"]}}],
        })


def test_reviewer_keeps_adverse_step_after_first_200_outputs(monkeypatch):
    from marvis.business_acceptance import evaluate_business_acceptance

    wire = Wire(monkeypatch)
    wire.reply = {"summary": "最后一个输出仍需关注"}
    outputs = {f"step-{i}": {"metrics": {"test_ks": .4}, "file_content": PRIVATE[3]} for i in range(300)}
    outputs["step-299"]["metrics"]["test_ks"] = .01
    Reviewer(lambda: wire.client)._llm_summarize("验证", _plan(_step([])), outputs,
                                               evaluate_business_acceptance(None, None))
    actual = wire.last_context()["outputs"]
    assert len(actual) == 300
    assert actual[-1] == {"step_id": "step-299", "output": {"metrics": {"test_ks": .01}}}


def test_structural_catalog_keeps_tail_tool_and_oversized_outputs_fail_before_wire(monkeypatch):
    from marvis.llm_outbound import OutboundContextError

    wire = Wire(monkeypatch)
    catalog = [{"plugin": "synthetic", "tool": f"tool_{i}"} for i in range(250)]
    prompt = outbound_prompt({"available_tools": catalog})
    wire.client.complete(system_prompt="", user_prompt=prompt, stream=False)
    assert wire.last_context()["available_tools"] == catalog
    with pytest.raises(OutboundContextError, match="未截断"):
        outbound_prompt({"outputs": {f"step-{i}": {"metrics": {"test_ks": .4}} for i in range(10001)}})
    assert len(wire.requests) == 1
