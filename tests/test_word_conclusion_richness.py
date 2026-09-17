from __future__ import annotations

import json
from dataclasses import replace

from marvis.agent.service import (
    _stage_instructions,
    fallback_word_conclusions,
    generate_word_conclusions,
)
from marvis.domain import TaskRecord, TaskStatus
from marvis.llm_client import LLMClientError
from marvis.validation.results import BinRow, OverallRow, validation_results_to_dict
from tests.output.test_excel import _make_results


def _task(**overrides) -> TaskRecord:
    payload = dict(
        id="task-rich",
        model_name="自营渠道甲T卡",
        model_version="v1",
        validator="qa",
        source_dir="/tmp/materials",
        algorithm="lgb",
        run_mode="agent",
        target_col="y",
        score_col="pred",
        split_col="split",
        time_col="apply_month",
        feature_columns=[],
        notebook_path=None,
        sample_path=None,
        pmml_path=None,
        dictionary_path=None,
        report_values_revision=0,
        status=TaskStatus.WRITING_ARTIFACTS,
        status_message="metrics generated",
        created_at="2026-05-31T00:00:00",
        updated_at="2026-05-31T00:00:00",
        validation_workflow_version=2,
    )
    payload.update(overrides)
    return TaskRecord(**payload)


def _evidence(
    *,
    oot_ks: float,
    oot_auc: float,
    oot_psi: float,
    train_ks: float,
    test_ks: float,
    overfit_status: str,
    tail_lift: float,
) -> dict:
    results = _make_results()
    overall: list[OverallRow] = []
    for row in results.effectiveness.overall:
        if row.split == "train":
            overall.append(replace(row, ks=train_ks))
        elif row.split == "test":
            overall.append(replace(row, ks=test_ks))
        else:
            overall.append(
                replace(
                    row,
                    ks=oot_ks,
                    auc=oot_auc,
                    psi_vs_train=oot_psi,
                    tail_lift_5pct=tail_lift,
                )
            )
    independent_oot = BinRow(1, 0.70, 0.82, 10, 6, 0.60, 0.10, 0.60, tail_lift, 0.12)
    results = replace(
        results,
        effectiveness=replace(
            results.effectiveness,
            overall=overall,
            independent_quantile_bin_tables={"oot": [independent_oot]},
        ),
    )
    payload = validation_results_to_dict(results)
    payload["overfitting_check"] = {
        "metric": "ks",
        "status": overfit_status,
        "train_ks": train_ks,
        "test_ks": test_ks,
        "oot_ks": oot_ks,
        "train_test_relative_diff": abs(train_ks - test_ks) / train_ks,
        "train_oot_abs_diff": abs(train_ks - oot_ks),
        "train_test_status": overfit_status,
        "train_oot_status": overfit_status,
    }
    return {"validation_results": payload}


def test_fallback_final_conclusion_is_model_specific_not_boilerplate():
    strong = fallback_word_conclusions(
        task=_task(model_name="自营通用T卡"),
        evidence=_evidence(
            oot_ks=0.4120,
            oot_auc=0.7810,
            oot_psi=0.0210,
            train_ks=0.4300,
            test_ks=0.4180,
            overfit_status="pass",
            tail_lift=2.80,
        ),
    )
    weak = fallback_word_conclusions(
        task=_task(model_name="渠道甲T卡"),
        evidence=_evidence(
            oot_ks=0.1810,
            oot_auc=0.6020,
            oot_psi=0.3100,
            train_ks=0.4100,
            test_ks=0.2500,
            overfit_status="fail",
            tail_lift=1.05,
        ),
    )
    strong_text = strong["TEXT:final_validation_conclusion"]
    weak_text = weak["TEXT:final_validation_conclusion"]
    assert strong_text != weak_text
    for text in (strong_text, weak_text):
        assert "KS" in text
        assert "AUC" in text
        assert "PSI" in text
        assert "过拟合" in text
        assert "分箱" in text
        assert "压力" in text
    assert "0.4120" in strong_text
    assert "0.7810" in strong_text
    assert "0.0210" in strong_text
    assert "0.1810" in weak_text
    assert "0.3100" in weak_text
    assert "过拟合风险" in weak_text or "存在过拟合" in weak_text
    assert "!!过拟合检查未通过" in weak_text
    assert "lift" in weak_text.lower() or "分箱" in weak_text


def test_word_conclusion_prompt_requires_model_specific_overall_judgment():
    instructions = _stage_instructions(
        "word_conclusion_draft",
        validation_workflow_version=2,
    )
    assert "本模型" in instructions
    assert "不得套用" in instructions or "不得使用同一套套话" in instructions
    assert "KS" in instructions
    assert "AUC" in instructions
    assert "PSI" in instructions
    assert "过拟合" in instructions
    assert "分箱" in instructions
    assert "历史同类模型" in instructions or "跨任务记忆" in instructions
    assert "lift_ranking_assessment" in instructions or "单调" in instructions
    assert "本次未见可比历史模型" in instructions
    assert "!!" in instructions


def test_metrics_stage_instructions_require_lift_ranking_highlight_and_memory_compare():
    instructions = _stage_instructions("metrics", validation_workflow_version=2)
    assert "lift_ranking_assessment" in instructions
    assert "单调" in instructions
    assert "0.99" in instructions
    assert "!!关键短语!!" in instructions
    assert "本次未见可比历史模型" in instructions
    assert "稳定性表现和压力测试风险" in instructions


def test_generate_word_conclusions_attaches_memory_refs_on_llm_failure(monkeypatch):
    class FailingClient:
        def complete(self, **_kwargs):
            raise LLMClientError("offline")

    monkeypatch.setattr(
        "marvis.agent.service._client",
        lambda _profile: FailingClient(),
    )
    memory_context = {
        "scope": "cross_task_agent_memory",
        "memories": [
            {
                "id": "mem-hist-1",
                "memory_type": "model_experience",
                "summary": "上一版自营T卡 OOT KS 0.36。",
                "source_task_id": "task-hist",
                "confidence": "high",
            }
        ],
    }

    values, metadata = generate_word_conclusions(
        task=_task(),
        evidence=_evidence(
            oot_ks=0.30,
            oot_auc=0.70,
            oot_psi=0.04,
            train_ks=0.32,
            test_ks=0.31,
            overfit_status="pass",
            tail_lift=2.0,
        ),
        memory_context=memory_context,
        model_profile={"api_base_url": "http://llm", "model_name": "m", "api_key": "k"},
    )

    assert values == {}
    references = metadata["memory_references"]
    assert references[0]["id"] == "mem-hist-1"
    assert references[0]["memory_type"] == "model_experience"
    assert references[0]["source_task_id"] == "task-hist"
    assert references[0]["confidence"] == "high"
    assert references[0]["use_reason"] == "word_conclusion_draft"


def test_word_conclusion_prompt_injects_memory_and_forbids_metric_rewrite(monkeypatch):
    captured = {}

    class CapturingClient:
        def complete(self, **kwargs):
            captured.update(kwargs)
            return json.dumps(
                {
                    "TEXT:pressure_test_summary": "压力测试摘要。",
                    "TEXT:pressure_impact_recommendation": "压力测试建议。",
                    "TEXT:final_validation_conclusion": "本模型 OOT KS 0.3000，稳定性可接受。PMML部署可用。",
                },
                ensure_ascii=False,
            )

    monkeypatch.setattr(
        "marvis.agent.service._client",
        lambda _profile: CapturingClient(),
    )
    memory_context = {
        "scope": "cross_task_agent_memory",
        "memories": [
            {
                "id": "mem-hist-2",
                "memory_type": "model_experience",
                "summary": "历史同类模型 OOT KS 0.28。",
                "source_task_id": "task-old",
                "confidence": "medium",
            }
        ],
    }

    _values, metadata = generate_word_conclusions(
        task=_task(),
        evidence=_evidence(
            oot_ks=0.30,
            oot_auc=0.70,
            oot_psi=0.04,
            train_ks=0.32,
            test_ks=0.31,
            overfit_status="pass",
            tail_lift=2.0,
        ),
        memory_context=memory_context,
        model_profile={"api_base_url": "http://llm", "model_name": "m", "api_key": "k"},
    )

    prompt = json.loads(captured["user_prompt"])
    assert prompt["cross_task_memory"]["memories"][0]["id"] == "mem-hist-2"
    assert "不能改变" in prompt["cross_task_memory"]["usage_rules"]
    assert metadata["memory_references"][0]["id"] == "mem-hist-2"
    assert "不得编造或改写 KS、AUC、PSI" in captured["system_prompt"]
