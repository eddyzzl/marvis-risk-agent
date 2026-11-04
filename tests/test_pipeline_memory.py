from types import SimpleNamespace

import pytest

import marvis.pipeline as pipeline_module
from marvis.agent_memory.extractors import extract_model_experience
from marvis.db import TaskRepository
from marvis.domain import TaskStatus
from marvis.pipeline import run_metrics_stage, run_pmml_scoring_stage
from marvis.pipeline_memory import _memory_channel, _memory_model_experience_payload


def test_model_experience_payload_fills_missing_model_version_and_keeps_zero_psi():
    task = SimpleNamespace(
        id="task-1",
        model_name="A卡模型",
        model_version="",
    )
    results = {
        "model_name": "A卡模型",
        "model_version": "",
        "effectiveness": {
            "overall": [
                {
                    "split": "train",
                    "ks": 0.31,
                    "auc": 0.72,
                    "psi_vs_train": 0.0,
                }
            ],
            "monthly_ks": [{"month": "202601", "ks": 0.31}],
        },
        "basic_info": {
            "feature_importance": [{"feature": "x1", "category": "征信"}],
        },
    }

    payload = _memory_model_experience_payload(task=task, results=results)
    candidate = extract_model_experience(payload)

    assert payload["model_version"] == "未标注"
    assert payload["metrics"]["psi"] == 0.0
    assert candidate is not None
    assert candidate.payload["model_version"] == "未标注"
    assert candidate.payload["psi"] == 0.0
    assert candidate.payload["evaluation_split"] == "train"
    assert candidate.payload["metric_units"] == {"ks": "fraction", "auc": "fraction", "psi": "index"}


def test_model_experience_payload_infers_channel_and_lifts_from_results():
    task = SimpleNamespace(
        id="task-t",
        model_name="自营通用T卡多头MOB6",
        model_version="v1",
    )
    results = {
        "model_name": "自营通用T卡多头MOB6",
        "model_version": "v1",
        "effectiveness": {
            "overall": [
                {
                    "split": "train",
                    "ks": 0.40,
                    "auc": 0.78,
                    "psi_vs_train": 0.0,
                },
                {
                    "split": "oot",
                    "ks": 0.31,
                    "auc": 0.72,
                    "psi_vs_train": 0.04,
                    "head_lift_5pct": 0.20,
                    "tail_lift_5pct": 2.10,
                },
            ],
            "monthly_ks": [{"month": "202511", "ks": 0.31}],
        },
        "basic_info": {
            "feature_importance": [{"feature": "x1", "category": "征信"}],
        },
    }

    payload = _memory_model_experience_payload(task=task, results=results)
    candidate = extract_model_experience(payload)

    assert payload["channel"] == "自营"
    assert payload["overfitting_status"] == "fail"
    assert payload["head_lift_5pct"] == 0.20
    assert candidate is not None
    assert "过拟合检查未通过" in candidate.summary
    assert candidate.payload["evaluation_split"] == "oot"
    assert candidate.payload["metric_units"]["head_lift_5pct"] == "ratio"


def test_model_experience_does_not_label_an_unknown_split_as_oot():
    task = SimpleNamespace(id="source", model_name="A卡", model_version="v1")
    payload = _memory_model_experience_payload(task=task, results={
        "effectiveness": {"overall": [{"split": "historical_private_label", "ks": .2, "auc": .6, "psi_vs_train": .1}]},
    })
    assert "evaluation_split" not in payload
    assert payload["metrics"]["ks"] == .2


def _record_memory_capture(monkeypatch, repo):
    calls = []

    def record(*, repo, task_id, outputs_dir, hook_dispatcher):
        calls.append((task_id, repo.get_task(task_id).status, (outputs_dir / "validation_results.json").is_file()))

    monkeypatch.setattr(pipeline_module, "_capture_agent_memory_for_metrics_success", record)
    return calls


def test_v2_metrics_stage_captures_agent_memory_after_success(
    ready_validation_task, pipeline_settings, monkeypatch
):
    repo = TaskRepository(pipeline_settings.db_path)
    calls = _record_memory_capture(monkeypatch, repo)
    run_pmml_scoring_stage(task_id=ready_validation_task.id, settings=pipeline_settings)
    run_metrics_stage(task_id=ready_validation_task.id, settings=pipeline_settings)
    # Memory is captured once, only after results are durable and the status advanced.
    assert calls == [(ready_validation_task.id, TaskStatus.WRITING_ARTIFACTS, True)]


def test_v2_metrics_stage_failure_does_not_capture_agent_memory(
    ready_validation_task, pipeline_settings, monkeypatch
):
    repo = TaskRepository(pipeline_settings.db_path)
    calls = _record_memory_capture(monkeypatch, repo)
    run_pmml_scoring_stage(task_id=ready_validation_task.id, settings=pipeline_settings)

    def fail(**_kwargs):
        raise ValueError("metrics kernel failed")

    monkeypatch.setattr(pipeline_module, "compute_platform_validation_results", fail)
    with pytest.raises(Exception):
        run_metrics_stage(task_id=ready_validation_task.id, settings=pipeline_settings)
    assert calls == []


def test_memory_channel_prefers_explicit_input_and_does_not_invent_cohorts():
    task = SimpleNamespace(model_name="自营渠道甲T卡MOB6")
    assert _memory_channel(task, {}) == "渠道甲"
    assert _memory_channel(task, {"channel": "用户指定渠道"}) == "用户指定渠道"
    assert _memory_channel(task, {"model_name": "普通模型"}) == "未标注"
