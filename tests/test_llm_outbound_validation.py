"""Privacy projection must preserve native deterministic validation evidence."""
from copy import deepcopy
from dataclasses import asdict
import json

import pytest

from marvis.llm_outbound import OutboundContextError
from marvis.llm_outbound_validation import validation_metrics, validation_metrics_prompt
from marvis.validation.lift_ranking import assess_lift_ranking
from marvis.validation.overfitting import overfitting_check_from_validation_results
from marvis.validation.results import (
    BasicInfoResult, BinRow, EffectivenessResult, FeatureImportanceRow, MonthlyKsRow,
    MonthlyPsiRow, MonthlyRow, OverallRow, PsiStabilityRow, SplitRow,
    StressBaseline, StressCategoryResult, StressTestResult, ValidationResults,
)
from tests.test_llm_outbound import Wire


PRIVATE = "PRIVATE_BUSINESS_LITERAL_739"


def _native_result():
    aligned = [BinRow(1, 0.1, 0.4, 80, 4, .05, .8, .2, .25, .3),
               BinRow(2, 0.4, 0.9, 20, 16, .8, 1., 1., 4., .0)]
    independent = [BinRow(1, .1, .2, 50, 9, .18, .5, .45, .9, .05),
                   BinRow(2, .2, .9, 50, 11, .22, 1., 1., 1.1, .0)]
    result = ValidationResults(
        model_name=PRIVATE, model_version=PRIVATE, algorithm="logistic", target_type="binary",
        basic_info=BasicInfoResult(
            sample_period=("2025-01-01", "2026-03-31"),
            split_summary=[SplitRow("train", 100, 20, .2, "2025-01-01", "2025-12-31")],
            monthly_distribution=[MonthlyRow("202601", 100, 20, .2)],
            hyperparameters={"C": .25, "solver": "lbfgs", "coef_": [0.398765432], "secret": PRIVATE},
            feature_importance=[FeatureImportanceRow(1, "feature_a", .8, PRIVATE)],
        ),
        effectiveness=EffectivenessResult(
            overall=[OverallRow("train", .41, 0., 100, .2, 20, .78, .25, 4.),
                     OverallRow("test", .37, .06, 100, .2, 20, .73, .9, 1.1),
                     OverallRow("oot", .28, .36, 100, .2, 20, .68, .9, 1.1)],
            bin_tables={"oot": aligned}, independent_quantile_bin_tables={"oot": independent},
            monthly_ks=[MonthlyKsRow("202601", .31, 100, 20, .2, .72, .9, 1.1)],
            monthly_psi=[MonthlyPsiRow("2026-03", .36, .1, .2, .29, "2026-01", True)],
            psi_stability_table=[PsiStabilityRow(PRIVATE, 80, .8, 50, .5, .14)],
        ),
        stress_test=StressTestResult(
            StressBaseline(.28, 100, aligned),
            [StressCategoryResult(PRIVATE, ["feature_a"], .17, -.11, .21, aligned, PRIVATE)],
        ),
    )
    payload = asdict(result)
    payload["effectiveness"]["lift_ranking_assessment"] = assess_lift_ranking(payload["effectiveness"])
    payload["overfitting_check"] = overfitting_check_from_validation_results(payload)
    payload["reproducibility"] = {"rows": [{"score_code_model": .987654321, "row_index": PRIVATE}]}
    payload["report_draft"] = {"TEXT:model_scope": PRIVATE}
    return payload


def test_native_metrics_reach_wire_unchanged_without_private_material(monkeypatch):
    raw = _native_result()
    before = deepcopy(raw)
    wire = Wire(monkeypatch)
    wire.client.complete(system_prompt="", user_prompt=validation_metrics_prompt(raw), stream=False)
    body = json.dumps(wire.requests)
    assert PRIVATE not in body
    assert "0.987654321" not in body and "0.398765432" not in body
    metrics = wire.last_context()["evidence"]["validation_results"]
    assert raw == before
    assert metrics["effectiveness"]["overall"] == raw["effectiveness"]["overall"]
    assert metrics["effectiveness"]["monthly_ks"] == raw["effectiveness"]["monthly_ks"]
    assert metrics["effectiveness"]["monthly_psi"] == raw["effectiveness"]["monthly_psi"]
    assert metrics["effectiveness"]["lift_ranking_assessment"] == raw["effectiveness"]["lift_ranking_assessment"]
    assert metrics["overfitting_check"] == raw["overfitting_check"]
    assert metrics["overfitting_check"]["status"] == "fail"
    assert metrics["basic_info"]["hyperparameters"] == {"C": .25, "solver": "lbfgs"}
    assert metrics["basic_info"]["monthly_distribution"] == raw["basic_info"]["monthly_distribution"]
    assert metrics["stress_test"]["per_category"] == [{
        "category_index": 1, "dropped_feature_count": 1, "status": "completed",
        "ks_after": .17, "ks_delta": -.11, "psi_vs_baseline": .21,
    }]
    assert metrics["effectiveness"]["psi_stability_table"] == [{
        "bin_index": 1, "expected_count": 80, "expected_pct": .8,
        "actual_count": 50, "actual_pct": .5, "psi": .14,
    }]


def test_native_ranking_keeps_distinct_sources_and_missing_evidence():
    metrics = validation_metrics(_native_result())
    rows = metrics["effectiveness"]["lift_ranking_assessment"]["splits"]
    assert rows[0]["train_aligned"] is None
    oot = rows[-1]
    assert oot["train_aligned"]["source"] == "train_aligned"
    assert oot["train_aligned"]["head_group_strength"] == "strong"
    assert oot["independent_quantile"]["source"] == "independent_quantile"
    assert oot["independent_quantile"]["head_group_strength"] == "weak"
    assert "head_lift_near_one" in oot["independent_quantile"]["notes"]


def test_more_than_200_months_does_not_hide_latest_adverse_period():
    rows = [{"month": f"{2000 + index // 12:04d}-{index % 12 + 1:02d}", "psi_vs_train": .01}
            for index in range(210)]
    rows[-1]["psi_vs_train"] = .41
    raw = {"effectiveness": {"monthly_psi": rows}}
    assert validation_metrics(raw)["effectiveness"]["monthly_psi"] == rows


def test_excessive_metric_history_fails_before_any_http_request(monkeypatch):
    wire = Wire(monkeypatch)
    with pytest.raises(OutboundContextError, match="未截断"):
        wire.client.complete(system_prompt="", user_prompt=validation_metrics_prompt({
            "effectiveness": {"monthly_psi": [{"month": "202601", "psi_vs_train": .1}] * 10001},
        }), stream=False)
    assert wire.requests == []


@pytest.mark.parametrize("month", [PRIVATE, "2026-13", "202602-30", "2026-02-30", "0000-01"])
def test_invalid_period_or_string_metric_is_not_forwarded(month):
    raw = {"effectiveness": {"monthly_ks": [{"month": month, "ks": PRIVATE}]}}
    assert validation_metrics(raw)["effectiveness"]["monthly_ks"] == []


def test_modeling_report_preserves_native_compact_months():
    from marvis.llm_outbound_reports import modeling_report_prompt

    payload = json.loads(modeling_report_prompt({"sample_analysis": [
        {"放款月": "202601", "放款笔数": 100, "逾期": 20, "逾期率": .2},
    ], "vintage": {"points": [{"cohort": "202601", "mob": 3, "sample_count": 100, "bad_rate": .2}]}}))
    summary = payload["structured_summary"]
    assert summary["sample_analysis"][0]["month"] == "202601"
    assert summary["vintage"]["points"][0]["cohort"] == "202601"


def test_modeling_report_keeps_late_feature_and_stress_risks_beyond_200_rows():
    from marvis.llm_outbound_reports import modeling_report_prompt

    univariate = [{"feature": f"x_{index}", "split": split, "ks": .2, "psi_vs_train": .01}
                  for index in range(100) for split in ("train", "test", "oot")]
    univariate[-1]["psi_vs_train"] = .42
    groups = {f"private_group_{index}": {"ks_after": .3, "psi_vs_baseline": .02} for index in range(210)}
    groups["private_group_209"]["psi_vs_baseline"] = .51
    payload = json.loads(modeling_report_prompt({"univariate": univariate, "stress_product_removal": groups}))
    result = payload["structured_summary"]
    assert result["univariate"] == univariate
    assert len(result["stress_product_removal"]["groups"]) == 210
    assert result["stress_product_removal"]["groups"][-1]["psi_vs_baseline"] == .51
    assert "private_group" not in json.dumps(payload)


def test_fixed_labels_do_not_become_a_string_exfiltration_channel():
    raw = _native_result()
    raw["algorithm"] = PRIVATE
    raw["overfitting_check"]["status"] = PRIVATE
    raw["stress_test"]["per_category"][0]["status"] = PRIVATE
    ranking = raw["effectiveness"]["lift_ranking_assessment"]
    ranking["guidance"] = PRIVATE
    ranking["splits"][-1]["train_aligned"]["notes"].append(PRIVATE)
    assert PRIVATE not in validation_metrics_prompt(raw)


def test_v2_metrics_service_native_evidence_reaches_each_topic_without_legacy_truncation(monkeypatch):
    from dataclasses import replace
    from marvis.agent.service import summarize_stage
    from tests.test_agent_service import _task

    raw = _native_result()
    periods = [{"month": f"{2000 + i // 12:04d}-{i % 12 + 1:02d}", "psi_vs_train": .01}
               for i in range(210)]
    periods[-1]["psi_vs_train"] = .49
    raw["effectiveness"]["monthly_psi"] = periods
    before = deepcopy(raw)
    wire = Wire(monkeypatch)
    wire.reply = {"summary": "已测 OOT KS 为 0.28，PSI 为 0.36；压力测试剔除后 KS 为 0.17，需关注风险。"}
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    deltas = []
    content, metadata = summarize_stage(
        task=replace(_task(), validation_workflow_version=2), stage="metrics",
        evidence={"validation_results": raw, "scan": {"text": PRIVATE},
                  "report_draft": {"text_values": {"TEXT:model_scope": PRIVATE}}},
        memory_context={"summary": PRIVATE, "references": [{"id": PRIVATE}]},
        model_profile={}, fallback="本地回退", on_delta=deltas.append,
    )
    assert len(wire.requests) == 6
    assert metadata["fallback"] is False
    assert metadata["history_comparison_available"] is False
    assert "memory_refs" not in metadata and "memory_used" not in metadata
    assert deltas == [content]
    assert raw == before
    assert PRIVATE not in json.dumps(wire.requests)
    contexts = {json.loads(request["messages"][1]["content"])["narrative_topic"]:
                json.loads(request["messages"][1]["content"]) for request in wire.requests}
    for topic in ("overall", "stability", "recommendation"):
        metrics = contexts[topic]["evidence"]["validation_results"]
        assert metrics["effectiveness"]["monthly_psi"] == periods
        assert metrics["effectiveness"]["overall"] == raw["effectiveness"]["overall"]
    ranking = contexts["ranking"]["evidence"]["validation_results"]["effectiveness"]
    assert ranking["lift_ranking_assessment"] == raw["effectiveness"]["lift_ranking_assessment"]
    assert contexts["performance"]["evidence"]["validation_results"]["overfitting_check"] == raw["overfitting_check"]
    assert contexts["pressure"]["evidence"]["validation_results"]["stress_test"]["per_category"][0]["ks_delta"] == -.11


def test_v2_metrics_repair_keeps_native_evidence_and_never_echoes_rejected_text(monkeypatch):
    from marvis.agent.validation_narrative import generate_v2_metrics_summary

    wire = Wire(monkeypatch)
    wire.replies = [PRIVATE, {"summary": "首个主题完整解读"}]
    wire.reply = {"summary": "其余主题完整解读"}
    content = generate_v2_metrics_summary(wire.client, {"evidence": {"validation_results": _native_result()}})
    assert "首个主题完整解读" in content
    assert len(wire.requests) == 7
    first, second = [json.loads(request["messages"][1]["content"]) for request in wire.requests[:2]]
    assert second.pop("response_repair")["reason"] == "json_text_contract"
    assert first == second
    assert PRIVATE not in json.dumps(wire.requests)


def _complete_metrics_summary():
    return (
        "总体判断\n本次未见可比历史模型，已测结果存在风险。\n\n"
        "效果表现\nOOT KS 为 0.28，按 train 分箱与独立分箱的结果应分别解释。\n\n"
        "稳定性表现\nOOT PSI 为 0.36，需关注漂移。\n\n"
        "压力测试风险\n剔除类别 1 后 KS 为 0.17，未测分层不能推断。\n\n"
        "建议\n结合平台明细复核并持续监测。"
    )


@pytest.mark.parametrize("structured_output", ["json_schema", "json_object", "none"])
def test_v1_metrics_native_wire_and_repair_emit_one_complete_summary(monkeypatch, structured_output):
    from marvis.agent.service import summarize_stage
    from tests.test_agent_service import _task

    raw = _native_result()
    raw["effectiveness"].pop("lift_ranking_assessment")
    periods = [{"month": f"{2000 + i // 12:04d}-{i % 12 + 1:02d}", "psi_vs_train": .01}
               for i in range(215)]
    periods[-1]["psi_vs_train"] = .49
    raw["effectiveness"]["monthly_psi"] = periods
    before = deepcopy(raw)
    wire = Wire(monkeypatch, text_reply=True)
    wire.client.profile["structured_output"] = structured_output
    wire.replies = [PRIVATE, _complete_metrics_summary()]
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    deltas = []
    content, metadata = summarize_stage(
        task=_task(), stage="metrics", evidence={"validation_results": raw, "report_draft": {"body": PRIVATE}},
        memory_context={"memories": [{"summary": PRIVATE}]}, model_profile={}, fallback="本地回退",
        on_delta=deltas.append,
    )
    assert content == _complete_metrics_summary() and deltas == [content]
    assert metadata["summary_repaired"] is True and metadata["fallback"] is False
    assert metadata["history_comparison_available"] is False and "memory_references" not in metadata
    assert len(wire.requests) == 2 and all(not request["stream"] for request in wire.requests)
    first, second = [json.loads(request["messages"][1]["content"]) for request in wire.requests]
    assert second.pop("repair_request")["missing_sections"] == ["总体判断", "效果表现", "稳定性表现", "压力测试风险", "建议"]
    assert first == second
    metrics = first["evidence"]["validation_results"]
    assert metrics["effectiveness"]["monthly_psi"] == periods
    assert metrics["overfitting_check"] == raw["overfitting_check"]
    assert metrics["effectiveness"]["lift_ranking_assessment"] == assess_lift_ranking(raw["effectiveness"])
    assert "bin_tables" not in metrics["effectiveness"]
    assert raw == before and PRIVATE not in json.dumps(wire.requests)


@pytest.mark.parametrize("workflow_version", [1, 2])
def test_persisted_native_metrics_keep_ranking_without_precomputed_assessment(tmp_path, monkeypatch, workflow_version):
    from types import SimpleNamespace
    from dataclasses import replace
    from marvis.agent.service import summarize_stage
    from marvis.agent.validation_evidence import agent_evidence_from_settings
    from tests.test_agent_service import _task

    raw = _native_result()
    raw["effectiveness"].pop("lift_ranking_assessment")
    raw.pop("overfitting_check")
    raw["stress_test"]["unclassified_features"] = [PRIVATE]
    directory = tmp_path / "task-1" / "outputs"
    directory.mkdir(parents=True)
    (directory / "validation_results.json").write_text(json.dumps(raw), encoding="utf-8")
    evidence = agent_evidence_from_settings(SimpleNamespace(tasks_dir=tmp_path), "task-1")
    assert "lift_ranking_assessment" not in evidence["validation_results"]["effectiveness"]
    wire = Wire(monkeypatch, text_reply=workflow_version == 1)
    wire.reply = _complete_metrics_summary() if workflow_version == 1 else {"summary": "已依据平台聚合指标解释。"}
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    _, metadata = summarize_stage(task=replace(_task(), validation_workflow_version=workflow_version), stage="metrics",
                                  evidence=evidence, model_profile={}, fallback="本地回退")
    assert metadata["fallback"] is False
    contexts = [json.loads(request["messages"][1]["content"]) for request in wire.requests]
    ranking = next(context for context in contexts if context.get("narrative_topic", "ranking") == "ranking")
    measured = ranking["evidence"]["validation_results"]["effectiveness"]["lift_ranking_assessment"]
    assert measured == assess_lift_ranking(evidence["validation_results"]["effectiveness"])
    assert measured["splits"][-1]["train_aligned"]["source"] == "train_aligned"
    assert measured["splits"][-1]["independent_quantile"]["source"] == "independent_quantile"
    pressure = next(context for context in contexts if context.get("narrative_topic", "pressure") == "pressure")
    assert pressure["evidence"]["validation_results"]["stress_test"]["unclassified_feature_count"] == 1
    assert PRIVATE not in json.dumps(wire.requests)


def test_v1_metrics_failed_repair_persists_only_the_final_fallback(tmp_path, monkeypatch):
    from dataclasses import replace
    from marvis.agent.service import summarize_stage
    from marvis.agent.validation_messages import add_and_stream_agent_message
    from marvis.db import TaskRepository, init_db
    from marvis.domain import TaskCreate

    repo = TaskRepository(tmp_path / "messages.sqlite")
    init_db(repo.db_path)
    task = repo.create_task(TaskCreate(model_name="fixture", model_version="v1", validator="test", source_dir=str(tmp_path)))
    task = replace(task, validation_workflow_version=1)
    wire = Wire(monkeypatch, text_reply=True)
    wire.reply = "仅返回不完整的说明。"
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    message = add_and_stream_agent_message(
        repo, task.id, stage="metrics", model_profile={}, raise_if_cancelled=lambda _: None,
        producer=lambda callback: summarize_stage(task=task, stage="metrics", evidence={"validation_results": _native_result()},
                                                 model_profile={}, fallback="本地回退", on_delta=callback),
    )
    messages = repo.list_agent_messages(task.id)
    assert len(messages) == 1 and messages[0]["id"] == message["id"]
    assert messages[0]["content"].count("总体判断") == 1
    assert messages[0]["content"].count(wire.reply) == 1
    assert messages[0]["metadata"]["streaming"] is False
    assert messages[0]["metadata"]["summary_repair_failed"] is True
    assert messages[0]["metadata"]["fallback"] is True
    assert len(wire.requests) == 2


def _report_replies():
    from marvis.agent.validation_narrative import _REPORT_SECTIONS
    return [{keys[0]: f"{topic}：本次 OOT KS 为 0.28，压力剔除后 KS 为 0.17，需关注已测风险。"}
            for topic, keys in _REPORT_SECTIONS]


def _v1_report_reply():
    return {
        "TEXT:pressure_test_summary": "类别 1 剔除后 KS 为 0.17，其他风险分层证据不足。",
        "TEXT:pressure_impact_recommendation": "应关注该类别的缺失与分布变化。",
        "TEXT:final_validation_conclusion": "分数一致性未通过，效果与稳定性仍需复核。本次未见可比历史模型。",
        "TEXT:model_training_description": "当前模型采用逻辑回归，C 为 0.25。",
    }


@pytest.mark.parametrize("structured_output", ["json_schema", "json_object", "none"])
def test_initial_v1_word_report_uses_measured_context_and_local_business_definitions(monkeypatch, structured_output):
    from dataclasses import replace
    from marvis.agent.service import generate_word_conclusions
    from marvis.validation_report_copy import narrative_report_values
    from tests.test_agent_service import _task

    raw = _native_result()
    raw["effectiveness"].pop("lift_ranking_assessment")
    periods = [{"month": f"{2000 + i // 12:04d}-{i % 12 + 1:02d}", "psi_vs_train": .01}
               for i in range(215)]
    periods[-1]["psi_vs_train"] = .49
    raw["effectiveness"]["monthly_psi"] = periods
    evidence = {
        "validation_results": raw,
        "scan": {"status": "scanned", "checks": [
            {"id": key, "status": "success", "message": PRIVATE} for key in
            ("material_notebook", "material_sample", "material_model_pmml", "material_data_dictionary", "notebook_contract")
        ]},
        "reproducibility": {"sample_size": 2, "seed": 42, "summary": {
            "status": "fail", "match_count": 1, "mismatch_count": 1, "max_abs_diff": 0.,
        }, "rows": [{"row_id": PRIVATE, "score_code_model": .987654321, "score_submitted_pmml": None}]},
        "notebook_steps": {"steps": [{"id": PRIVATE, "status": "succeeded", "cell_count": 2}],
                           "cells": [{"source": PRIVATE}]},
        "contract": {"algorithm": "logistic", "target_col": "target", "score_decimal_places": 6,
                     "feature_columns": ["feature_a"], "source": PRIVATE},
        "visible_stage_summaries": [{"content": PRIVATE}], "report_fields": {"text_values": {"body": PRIVATE}},
    }
    before = deepcopy(evidence)
    task = replace(_task(), model_name=PRIVATE)
    wire = Wire(monkeypatch)
    wire.client.profile["structured_output"] = structured_output
    wire.reply = {**_v1_report_reply(), "TEXT:model_scope": "不应覆盖本地定义", "TEXT:bad_sample_definition": "假设坏样本定义"}
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    values, metadata = generate_word_conclusions(task=task, evidence=evidence,
                                                 memory_context={"summary": PRIVATE}, model_profile={})
    assert metadata == {"fallback": False, "history_comparison_available": False}
    assert all(values[key] == value for key, value in _v1_report_reply().items())
    seeds = narrative_report_values(PRIVATE)
    assert values["TEXT:model_scope"] == seeds["TEXT:model_scope"]
    assert values["TEXT:bad_sample_definition"] == seeds["TEXT:bad_sample_definition"]
    assert len(wire.requests) == 1 and wire.requests[0]["stream"] is False
    context = wire.last_context()
    assert context["report_output_status_available"] is False
    measured = context["evidence"]
    assert measured["scan_interpretation"]["required_materials_complete"] is True
    assert measured["reproducibility"]["summary"] == evidence["reproducibility"]["summary"]
    assert measured["reproducibility"]["missing_submitted_score_count"] == 1
    assert measured["contract"]["score_decimal_places"] == 6
    assert measured["notebook_steps"]["steps"] == [{"step_index": 1, "status": "succeeded", "cell_count": 2}]
    assert measured["validation_results"]["effectiveness"]["monthly_psi"] == periods
    assert measured["validation_results"]["effectiveness"]["lift_ranking_assessment"] == assess_lift_ranking(raw["effectiveness"])
    assert "V2 PMML 打分工作流" not in wire.requests[0]["messages"][0]["content"]
    assert evidence == before and PRIVATE not in json.dumps(wire.requests)
    assert "0.987654321" not in json.dumps(wire.requests)


@pytest.mark.parametrize("bad_value", [123, False, {"content": "不应被字符串化"}, ["不应被字符串化"], None])
def test_v1_report_rejects_nontext_required_output_without_confirmable_values(monkeypatch, bad_value):
    from marvis.agent.service import generate_word_conclusions
    from tests.test_agent_service import _task

    wire = Wire(monkeypatch)
    wire.reply = {**_v1_report_reply(), "TEXT:final_validation_conclusion": bad_value}
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    values, metadata = generate_word_conclusions(task=_task(), evidence={"validation_results": _native_result()}, model_profile={})
    assert values == {} and metadata["fallback"] is True and metadata["confirmable"] is False
    assert metadata["history_comparison_available"] is False
    assert "fields must contain text" in metadata["llm_error"]


@pytest.mark.parametrize("revision_kind", ["user_instruction", "report_draft", "saved_report_narrative"])
def test_v1_unapproved_instruction_or_unversioned_report_context_cannot_send_raw_local_text(monkeypatch, revision_kind):
    from marvis.agent.service import generate_word_conclusions
    from tests.test_agent_service import _task

    wire = Wire(monkeypatch)
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    evidence = {"validation_results": _native_result()}
    instruction = PRIVATE if revision_kind == "user_instruction" else None
    if revision_kind != "user_instruction":
        evidence[revision_kind] = {"text_values": {"TEXT:model_scope": PRIVATE}}
    values, metadata = generate_word_conclusions(task=_task(), evidence=evidence, user_instruction=instruction, model_profile={})
    assert values == {} and metadata["confirmable"] is False and metadata["fallback"] is True
    assert wire.requests == []


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("saved", [
    {"report_revision": 1, "text_values": {}},
    {"report_revision": 0, "text_values": {}, "unknown_context": PRIVATE},
    {"report_revision": False, "text_values": {}},
])
def test_invalid_saved_revision_provenance_is_rejected_before_request(monkeypatch, version, saved):
    from dataclasses import replace
    from marvis.agent.service import generate_word_conclusions
    from tests.test_agent_service import _task

    wire = Wire(monkeypatch)
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    _, metadata = generate_word_conclusions(
        task=replace(_task(), validation_workflow_version=version),
        evidence={"validation_results": _native_result(), "saved_report_narrative": saved},
        model_profile={},
    )
    assert metadata["confirmable"] is False and wire.requests == []


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("seeded", [False, True])
def test_native_uninitialized_saved_envelope_generates_first_draft(monkeypatch, version, seeded):
    from dataclasses import replace
    from marvis.agent.service import generate_word_conclusions, saved_report_narrative_context
    from marvis.validation_report_copy import merge_seed_report_values
    from tests.test_agent_service import _task

    wire = Wire(monkeypatch)
    wire.reply = _v1_report_reply()
    if version == 2:
        wire.replies = _report_replies()
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    task = replace(_task(), validation_workflow_version=version)
    initial_values = merge_seed_report_values({}, model_name=task.model_name,
        model_version=task.model_version, validator=task.validator, algorithm=task.algorithm) if seeded else {}
    values, metadata = generate_word_conclusions(
        task=task,
        evidence={"validation_results": _native_result(),
                  "saved_report_narrative": saved_report_narrative_context(initial_values, 0)},
        model_profile={},
    )
    assert values and metadata["fallback"] is False
    assert len(wire.requests) == (1 if version == 1 else 7)


def test_fresh_v2_report_service_keeps_native_tail_and_local_definitions(monkeypatch):
    from dataclasses import replace
    from marvis.agent.service import generate_word_conclusions
    from marvis.validation_report_copy import narrative_report_values
    from tests.test_agent_service import _task

    raw = _native_result()
    rows = [{"month": f"{2000 + i // 12:04d}-{i % 12 + 1:02d}", "psi_vs_train": .01} for i in range(210)]
    rows[-1]["psi_vs_train"] = .49
    raw["effectiveness"]["monthly_psi"] = rows
    original = deepcopy(raw)
    wire = Wire(monkeypatch)
    wire.replies = _report_replies()
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    task = replace(_task(), validation_workflow_version=2, model_name=PRIVATE)
    values, meta = generate_word_conclusions(
        task=task, evidence={"validation_results": raw, "scan": {"content": PRIVATE},
                             "report_fields": {"private": PRIVATE}},
        memory_context={"summary": PRIVATE}, model_profile={},
    )
    assert meta == {"fallback": False, "history_comparison_available": False}
    assert len(wire.requests) == 7 and raw == original
    assert PRIVATE not in json.dumps(wire.requests)
    assert values["TEXT:model_scope"] == narrative_report_values(PRIVATE)["TEXT:model_scope"]
    contexts = {json.loads(r["messages"][1]["content"])["narrative_topic"]:
                json.loads(r["messages"][1]["content"]) for r in wire.requests}
    for topic in ("stability", "overall", "recommendation"):
        assert contexts[topic]["evidence"]["validation_results"]["effectiveness"]["monthly_psi"] == rows
    assert contexts["training"]["evidence"]["validation_results"]["basic_info"]["hyperparameters"] == {
        "C": .25, "solver": "lbfgs"}
    assert contexts["training"]["requested_fields"] == ["TEXT:model_training_description"]
    assert contexts["ranking"]["evidence"]["validation_results"]["effectiveness"]["lift_ranking_assessment"] == raw["effectiveness"]["lift_ranking_assessment"]
    assert len(values["TEXT:final_validation_conclusion"].split("\n\n")) == 4


def test_fresh_report_invalid_last_topic_does_not_publish_partial_values(monkeypatch):
    from dataclasses import replace
    from marvis.agent.service import generate_word_conclusions
    from tests.test_agent_service import _task

    wire = Wire(monkeypatch)
    wire.replies = _report_replies()[:-1] + [PRIVATE, PRIVATE]
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    values, meta = generate_word_conclusions(
        task=replace(_task(), validation_workflow_version=2),
        evidence={"validation_results": _native_result()}, model_profile={},
    )
    assert values == {} and meta["fallback"] is True and meta["confirmable"] is False
    assert len(wire.requests) == 8
    assert PRIVATE not in json.dumps(wire.requests)
    first, retry = [json.loads(r["messages"][1]["content"]) for r in wire.requests[-2:]]
    assert retry.pop("response_repair")["reason"] == "json_text_contract"
    assert first == retry


def test_fresh_report_cannot_overwrite_business_definitions_via_optional_output(monkeypatch):
    from dataclasses import replace
    from marvis.agent.service import generate_word_conclusions
    from marvis.validation_report_copy import narrative_report_values
    from tests.test_agent_service import _task

    wire = Wire(monkeypatch)
    replies = _report_replies()
    wire.replies = replies[:-1] + [{**replies[-1], "TEXT:model_scope": "未经要求的新业务范围"}, replies[-1]]
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    task = replace(_task(), validation_workflow_version=2)
    values, meta = generate_word_conclusions(task=task, evidence={"validation_results": _native_result()}, model_profile={})
    assert meta["fallback"] is False
    assert values["TEXT:model_scope"] == narrative_report_values(task.model_name)["TEXT:model_scope"]
    assert len(wire.requests) == 8
    assert "未经要求的新业务范围" not in json.dumps(wire.requests, ensure_ascii=False)


def test_pending_revision_migration_never_sends_private_draft_or_mutates_local_context(monkeypatch):
    from dataclasses import replace
    from marvis.agent.service import generate_word_conclusions
    from tests.test_agent_service import _task

    wire = Wire(monkeypatch)
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    evidence = {"validation_results": _native_result(), "report_draft": {"text_values": {
        "TEXT:model_scope": PRIVATE, "TEXT:sample_audience": "",
    }}}
    before = deepcopy(evidence)
    values, meta = generate_word_conclusions(task=replace(_task(), validation_workflow_version=2),
                                           evidence=evidence, user_instruction=PRIVATE, model_profile={})
    assert values == {} and meta["confirmable"] is False
    assert "未发送" in meta["llm_error"]
    assert wire.requests == [] and evidence == before


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_v2_scoring_service_preserves_native_counts_without_private_material(monkeypatch, failed, nested):
    from marvis.agent.service import summarize_stage
    from marvis.validation.results import pmml_scoring_result_from_dict, pmml_scoring_result_to_dict
    from tests.test_agent_pmml_scoring_evidence import _scoring_payload, _task

    raw = _scoring_payload(
        engine="pypmml-pmml4s-batch", engine_version=PRIVATE, output_field=PRIVATE,
        score_artifact_path=PRIVATE, success_count=999_997 if failed else 1_000_000,
        failure_count=3 if failed else 0, null_count=1 if failed else 0,
        non_finite_count=2 if failed else 0, status="failed" if failed else "pass",
        missing_inputs=[PRIVATE] if failed else [], bounded_errors=[PRIVATE] if failed else [],
    )
    raw = pmml_scoring_result_to_dict(pmml_scoring_result_from_dict(raw))
    evidence = {"validation_results": {"pmml_scoring": raw}} if nested else {"pmml_scoring": raw}
    evidence.update(notebook_steps={"source": PRIVATE}, contract={"score_fn": PRIVATE},
                    reproducibility={"rows": [{"score_code_model": .987654321}]})
    before = deepcopy(evidence)
    wire = Wire(monkeypatch, text_reply=True)
    wire.reply = "当前评分结果已按平台计数汇总。"
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    deltas = []
    text, meta = summarize_stage(task=_task(validation_workflow_version=2), stage="reproducibility",
                                 evidence=evidence, memory_context={"summary": PRIVATE},
                                 model_profile={}, fallback="本地回退", on_delta=deltas.append)
    assert meta == {"fallback": False, "history_comparison_available": False}
    assert deltas == [text]
    assert evidence == before
    assert len(wire.requests) == 1 and wire.requests[0]["stream"] is False
    assert PRIVATE not in json.dumps(wire.requests) and "0.987654321" not in json.dumps(wire.requests)
    context = wire.last_context()
    assert set(context["evidence"]) == {"pmml_scoring"}
    scoring = context["evidence"]["pmml_scoring"]
    for key in ("input_row_count", "success_count", "failure_count", "null_count", "non_finite_count",
                "elapsed_seconds", "rows_per_second", "chunk_size", "required_input_count", "status", "engine"):
        assert scoring[key] == raw[key]
    assert scoring["missing_input_count"] == scoring["error_count"] == int(failed)


def test_v2_scoring_guard_runs_before_any_response_is_emitted(monkeypatch):
    from marvis.agent.service import summarize_stage
    from tests.test_agent_pmml_scoring_evidence import _scoring_payload, _task

    wire = Wire(monkeypatch, text_reply=True)
    wire.reply = "PMML打分全部成功，未发现可复现性问题，建议补做代码模型与 PMML 分数一致性验证。"
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    deltas = []
    text, meta = summarize_stage(task=_task(validation_workflow_version=2), stage="reproducibility",
                                 evidence={"pmml_scoring": _scoring_payload()}, model_profile={},
                                 fallback="评分覆盖已确认，后续将分析模型效果。", on_delta=deltas.append)
    assert text == "评分覆盖已确认，后续将分析模型效果。"
    assert meta["guarded_stage_summary"] is True
    assert deltas == [text]
    assert wire.requests[0]["stream"] is False


def test_scoring_stage_does_not_replace_primary_failure_with_nested_success(monkeypatch):
    from marvis.agent.service import summarize_stage
    from tests.test_agent_pmml_scoring_evidence import _scoring_payload, _task

    failed = _scoring_payload(status="failed", success_count=999_999, failure_count=1, null_count=1)
    wire = Wire(monkeypatch, text_reply=True)
    wire.reply = "本次评分存在失败。"
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    summarize_stage(task=_task(validation_workflow_version=2), stage="reproducibility",
                    evidence={"pmml_scoring": failed, "validation_results": {"pmml_scoring": _scoring_payload()}},
                    model_profile={}, fallback="本地回退")
    scoring = wire.last_context()["evidence"]["pmml_scoring"]
    assert scoring["status"] == "failed" and scoring["failure_count"] == 1


def test_scoring_context_skips_unknown_metadata_but_does_not_invent_missing_counts():
    from marvis.llm_outbound_validation import pmml_scoring_prompt
    from tests.test_agent_pmml_scoring_evidence import _scoring_payload

    context = json.loads(pmml_scoring_prompt({"pmml_scoring": {"private": PRIVATE},
                                             "validation_results": {"pmml_scoring": _scoring_payload()}}))
    assert context["evidence"]["pmml_scoring"]["input_row_count"] == 1_000_000
    missing = json.loads(pmml_scoring_prompt({"pmml_scoring": {"private": PRIVATE}}))
    assert missing["evidence"]["pmml_scoring"] == {}


@pytest.mark.parametrize("case", ["pass", "review", "fail", "missing_score"])
def test_v1_native_score_comparison_reaches_wire_without_individual_scores(monkeypatch, case):
    from marvis.agent.service import summarize_stage
    from marvis.validation.reproducibility import run_reproducibility
    from tests.test_agent_pmml_scoring_evidence import _task
    from tests.validation.test_reproducibility import _FakeScorer, _config
    import pandas as pd

    count = 200
    private_score = .876543219
    sample = pd.DataFrame({"x1": [1.] * count, "x2": [2.] * count,
                           "sample_score": [private_score] * count, "y": [0] * count,
                           "split": ["train"] * count, "apply_month": ["202601"] * count})
    submitted = dict.fromkeys(range(count), private_score)
    if case != "pass":
        for index in range(5):
            submitted[index] = None if case == "missing_score" else private_score + (.00007 if case == "review" else .1)
    result = run_reproducibility(sample=sample, config=_config(random_sample_size=count),
                                code_scores=pd.Series([private_score] * count),
                                submitted_pmml_scorer=_FakeScorer(submitted))
    raw = asdict(result)
    assert raw["summary"]["status"] == ("fail" if case == "missing_score" else case)
    before = deepcopy(raw)
    wire = Wire(monkeypatch, text_reply=True)
    wire.reply = "分数一致性结论以平台判定为准。"
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    deltas = []
    text, meta = summarize_stage(task=_task(validation_workflow_version=1), stage="reproducibility",
                                 evidence={"reproducibility": raw, "contract": {
                                     "algorithm": "logistic", "target_col": "y", "score_col": "pred",
                                     "score_decimal_places": 6, "feature_columns": ["x1", "x2"],
                                     "score_fn": PRIVATE}}, model_profile={}, fallback="本地回退",
                                 memory_context={"summary": PRIVATE}, on_delta=deltas.append)
    assert meta == {"fallback": False, "history_comparison_available": False}
    assert deltas == [text] and raw == before
    context = wire.last_context()
    assert context["validation_workflow_version"] == 1
    assert context["evidence"]["reproducibility"] == {"sample_size": result.sample_size,
                                                         "seed": result.seed, "summary": raw["summary"],
                                                         "missing_submitted_score_count": 5 if case == "missing_score" else 0}
    if case == "missing_score":
        assert context["evidence"]["reproducibility"]["summary"]["max_abs_diff"] == 0
    assert context["evidence"]["contract"]["score_decimal_places"] == 6
    assert context["evidence"]["contract"]["feature_count"] == 2
    serialized = json.dumps(wire.requests)
    assert PRIVATE not in serialized and str(private_score) not in serialized
    assert '"rows"' not in serialized and wire.requests[0]["stream"] is False


def test_v1_native_notebook_failure_preserves_position_and_type_without_source(monkeypatch):
    import nbformat
    from marvis.agent.service import summarize_stage
    from marvis.notebook_steps import notebook_step_plan
    from marvis.notebooks import _build_step_events, _record_cell_error, _record_cell_start
    from tests.test_agent_pmml_scoring_evidence import _task

    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell("# " + PRIVATE),
                                              nbformat.v4.new_code_cell(f"raise FileNotFoundError('{PRIVATE}')")])
    plan = notebook_step_plan(notebook)
    events = {}
    _record_cell_start(events, cell=notebook.cells[1], cell_index=1)
    _record_cell_error(events, cell=notebook.cells[1], cell_index=1,
                       execute_reply={"content": {"ename": "FileNotFoundError", "evalue": PRIVATE,
                                                   "traceback": [PRIVATE]}})
    native = _build_step_events(plan, events)
    before = deepcopy(native)
    wire = Wire(monkeypatch, text_reply=True)
    wire.reply = "Notebook 的 cell 索引 1 执行失败，可查看本地诊断。"
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    _, meta = summarize_stage(task=_task(validation_workflow_version=1), stage="reproducibility",
                             evidence={"notebook_steps": native}, model_profile={}, fallback="本地回退")
    assert meta["fallback"] is False and native == before
    context = wire.last_context()["evidence"]
    assert context["reproducibility"] == {}
    assert context["notebook_steps"]["steps"][0]["status"] == "failed"
    assert context["notebook_steps"]["steps"][0]["cell_count"] == 1
    assert context["notebook_steps"]["failed_cells"] == [{"cell_index": 1, "step_index": 1,
                                                           "status": "failed", "exception_name": "FileNotFoundError"}]
    assert PRIVATE not in json.dumps(wire.requests)


def test_v1_notebook_tail_failure_is_not_truncated_and_oversize_fails_before_wire(monkeypatch):
    from marvis.agent.service import summarize_stage
    from tests.test_agent_pmml_scoring_evidence import _task

    steps = [{"id": f"step-{i}", "status": "succeeded", "cell_count": 1} for i in range(300)]
    steps[-1]["status"] = "failed"
    cells = [{"cell_index": 299, "step_id": "step-299", "status": "failed", "exception_name": PRIVATE}]
    wire = Wire(monkeypatch, text_reply=True)
    wire.reply = "末尾步骤执行失败。"
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    task = _task(validation_workflow_version=1)
    summarize_stage(task=task, stage="reproducibility", evidence={"notebook_steps": {"steps": steps, "cells": cells}},
                    model_profile={}, fallback="本地回退")
    notebook = wire.last_context()["evidence"]["notebook_steps"]
    assert len(notebook["steps"]) == 300 and notebook["steps"][-1]["status"] == "failed"
    assert notebook["failed_cells"] == [{"cell_index": 299, "step_index": 300, "status": "failed"}]
    text, meta = summarize_stage(task=task, stage="reproducibility", model_profile={}, fallback="本地回退",
                                 evidence={"notebook_steps": {"steps": steps * 34}})
    assert text == "本地回退" and meta["fallback"] is True
    assert len(wire.requests) == 1


@pytest.mark.parametrize("version", [1, 2])
def test_validation_start_uses_fixed_intent_without_task_private_metadata(monkeypatch, version):
    from dataclasses import replace
    from marvis.agent.service import compose_agent_start_message
    from tests.test_agent_pmml_scoring_evidence import _task

    wire = Wire(monkeypatch, text_reply=True)
    wire.reply = "将开始本次模型验证，先检查材料完备性。"
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    task = replace(_task(validation_workflow_version=version), model_name=PRIVATE,
                   model_version=PRIVATE, validator=PRIVATE, source_dir=PRIVATE)
    text, meta = compose_agent_start_message(task=task, model_profile={})
    assert text == wire.reply and meta == {"fallback": False}
    assert len(wire.requests) == 1 and PRIVATE not in json.dumps(wire.requests)
    context = wire.last_context()
    assert context["stage"] == "agent_start" and context["next_tool"] == "scan_materials"
    assert context["validation_workflow_version"] == version
