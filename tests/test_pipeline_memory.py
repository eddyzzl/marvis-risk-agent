from types import SimpleNamespace

from marvis.agent_memory.extractors import extract_model_experience
from marvis.pipeline_memory import _memory_model_experience_payload


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
