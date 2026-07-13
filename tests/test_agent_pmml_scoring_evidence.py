from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from marvis.agent.service import _stage_prompt, fallback_word_conclusions
from marvis.agent.validation_evidence import agent_evidence_from_settings
from marvis.app import create_app
from marvis.domain import TaskRecord, TaskStatus


def _scoring_payload(**overrides) -> dict:
    payload = {
        "schema_version": "marvis.pmml_scoring.v1",
        "cache_key": "a" * 64,
        "pmml_sha256": "b" * 64,
        "sample_sha256": "c" * 64,
        "engine": "jpmml-batch",
        "engine_version": "1.0",
        "output_field": "probability_1",
        "input_row_count": 1_000_000,
        "success_count": 1_000_000,
        "failure_count": 0,
        "null_count": 0,
        "non_finite_count": 0,
        "elapsed_seconds": 20.0,
        "rows_per_second": 50_000.0,
        "chunk_size": 10_000,
        "required_input_count": 2,
        "missing_inputs": [],
        "score_artifact_path": "pmml_scores.parquet",
        "score_artifact_sha256": "d" * 64,
        "status": "pass",
        "bounded_errors": [],
    }
    payload.update(overrides)
    return payload


def _task(*, validation_workflow_version: int) -> TaskRecord:
    return TaskRecord(
        id="task-pmml",
        model_name="A卡",
        model_version="V1",
        validator="qa",
        source_dir="/tmp",
        algorithm="xgb",
        run_mode="agent",
        target_col="y",
        score_col="pred",
        split_col="split",
        time_col="apply_month",
        feature_columns=["x1", "x2"],
        notebook_path=None,
        sample_path=None,
        pmml_path=None,
        dictionary_path=None,
        report_values_revision=0,
        status=TaskStatus.EXECUTED,
        status_message="PMML scoring complete",
        created_at="2026-07-13T00:00:00Z",
        updated_at="2026-07-13T00:00:00Z",
        validation_workflow_version=validation_workflow_version,
    )


def test_agent_evidence_strictly_loads_pmml_scoring_result(tmp_path: Path):
    task_id = "task-pmml"
    output_dir = tmp_path / "tasks" / task_id / "outputs"
    output_dir.mkdir(parents=True)
    result_path = output_dir / "pmml_scoring_result.json"
    result_path.write_text(json.dumps(_scoring_payload()), encoding="utf-8")
    settings = SimpleNamespace(tasks_dir=tmp_path / "tasks")

    evidence = agent_evidence_from_settings(settings, task_id)

    assert evidence["pmml_scoring"] == _scoring_payload()

    invalid = _scoring_payload(untrusted_extra="must not reach agent")
    result_path.write_text(json.dumps(invalid), encoding="utf-8")
    assert agent_evidence_from_settings(settings, task_id)["pmml_scoring"] == {}


def test_agent_evidence_migrates_legacy_v2_lift_order(tmp_path: Path):
    task_id = "task-pmml"
    output_dir = tmp_path / "tasks" / task_id / "outputs"
    output_dir.mkdir(parents=True)
    (output_dir / "validation_results.json").write_text(
        json.dumps(
            {
                "schema_version": "marvis.validation_results.v2",
                "effectiveness": {
                    "overall": [
                        {
                            "split": "oot",
                            "head_lift_5pct": 2.4,
                            "tail_lift_5pct": 0.3,
                        }
                    ],
                    "monthly_ks": [
                        {
                            "month": "202607",
                            "head_lift_5pct": 2.2,
                            "tail_lift_5pct": 0.4,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    settings = SimpleNamespace(tasks_dir=tmp_path / "tasks")

    validation_results = agent_evidence_from_settings(settings, task_id)[
        "validation_results"
    ]

    assert validation_results["lift_order"] == "good_to_bad"
    assert validation_results["effectiveness"]["overall"][0][
        "head_lift_5pct"
    ] == pytest.approx(0.3)
    assert validation_results["effectiveness"]["overall"][0][
        "tail_lift_5pct"
    ] == pytest.approx(2.4)


def test_v2_pmml_scoring_prompt_uses_only_scoring_coverage_and_performance():
    prompt = json.loads(
        _stage_prompt(
            task=_task(validation_workflow_version=2),
            stage="reproducibility",
            evidence={
                "pmml_scoring": _scoring_payload(),
                "notebook_steps": {"steps": [{"status": "passed"}]},
                "contract": {"score_fn": "model.predict_proba"},
                "reproducibility": {"summary": {"status": "pass"}},
                "validation_results": {
                    "pmml_scoring": _scoring_payload(),
                    "effectiveness": {"overall": [{"split": "oot", "ks": 0.3}]},
                },
            },
        )
    )

    assert set(prompt["evidence"]) == {"pmml_scoring"}
    scoring = prompt["evidence"]["pmml_scoring"]
    assert scoring["input_row_count"] == 1_000_000
    assert scoring["failure_count"] == 0
    assert scoring["rows_per_second"] == 50_000.0
    assert "cache_key" not in scoring
    assert "只使用 evidence.pmml_scoring" in prompt["instructions"]
    assert "不得声称执行过 Notebook" in prompt["instructions"]
    assert "不得建议当前或后续补做" in prompt["instructions"]
    assert "代码模型与 PMML 分数一致性" in prompt["instructions"]
    assert "AUC" in prompt["instructions"]


def test_v2_word_prompt_and_fallback_use_pmml_scoring_not_reproducibility():
    task = _task(validation_workflow_version=2)
    prompt = json.loads(
        _stage_prompt(
            task=task,
            stage="word_conclusion_draft",
            evidence={
                "scan": {"checks": []},
                "pmml_scoring": _scoring_payload(),
                "reproducibility": {"summary": {"status": "pass"}},
                "validation_results": {
                    "pmml_scoring": _scoring_payload(),
                    "reproducibility": {"summary": {"status": "pass"}},
                    "effectiveness": {"overall": [{"split": "oot", "ks": 0.3}]},
                    "stress_test": {"status": "completed", "per_category": []},
                },
                "visible_stage_summaries": [],
            },
        )
    )

    assert "pmml_scoring" in prompt["evidence"]
    assert "reproducibility" not in prompt["evidence"]
    assert "reproducibility" not in prompt["evidence"]["validation_results"]
    assert "PMML 打分测试" in prompt["instructions"]
    assert "不得声称执行过 Notebook" in prompt["instructions"]

    fallback = fallback_word_conclusions(task=task)
    combined = " ".join(fallback.values())
    assert "PMML部署可用" in combined
    assert "PMML 打分测试" not in fallback["TEXT:final_validation_conclusion"]
    assert "模型压力测试" in combined
    assert "Notebook 可复现性" not in combined
    assert "分数一致性" not in combined

    legacy = " ".join(
        fallback_word_conclusions(task=_task(validation_workflow_version=1)).values()
    )
    assert "Notebook 可复现性" in legacy


def test_task_evidence_endpoint_exposes_strict_pmml_scoring(tmp_path: Path):
    client = TestClient(create_app(tmp_path))
    response = client.post(
        "/api/tasks",
        json={
            "model_name": "A卡",
            "validator": "qa",
            "source_dir": str(tmp_path),
            "run_mode": "agent",
        },
    )
    task_id = response.json()["id"]
    output_dir = tmp_path / "tasks" / task_id / "outputs"
    output_dir.mkdir(parents=True)
    (output_dir / "pmml_scoring_result.json").write_text(
        json.dumps(_scoring_payload(untrusted_extra="reject file")),
        encoding="utf-8",
    )
    (output_dir / "validation_results.json").write_text(
        json.dumps({"pmml_scoring": _scoring_payload()}),
        encoding="utf-8",
    )

    evidence = client.get(f"/api/tasks/{task_id}/evidence").json()

    assert evidence["pmml_scoring"] == _scoring_payload()
    assert "untrusted_extra" not in evidence["pmml_scoring"]
