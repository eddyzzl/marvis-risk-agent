import inspect
from types import SimpleNamespace

from marvis.agent_memory.extractors import extract_model_experience
from marvis.pipeline import _execute_v2_metrics_stage
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


def test_v2_metrics_stage_captures_agent_memory_after_success():
    source = inspect.getsource(_execute_v2_metrics_stage)
    assert "_capture_agent_memory_for_metrics_success" in source


def test_memory_channel_prefers_explicit_input_and_does_not_invent_cohorts():
    task = SimpleNamespace(model_name="自营渠道甲T卡MOB6")
    assert _memory_channel(task, {}) == "渠道甲"
    assert _memory_channel(task, {"channel": "用户指定渠道"}) == "用户指定渠道"
    assert _memory_channel(task, {"model_name": "普通模型"}) == "未标注"
