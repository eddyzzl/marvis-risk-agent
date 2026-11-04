"""Behaviour of the evidence-only diagnostics Tools wired into the strategy pack."""

import pandas as pd
import pytest

from marvis.plugins.manifest import ToolRef
from marvis.packs.strategy.fairness_evidence import NEUTRAL_PHRASING_DRAFT_NOTICE
from tests.test_strategy_pack import _runtime


def _register(registry, tmp_path, task_id, name, frame):
    path = tmp_path / f"{name}.parquet"
    frame.to_parquet(path, index=False)
    return registry.register_existing(path, task_id=task_id, role="strategy_sample")


def test_time_leakage_tool_flags_features_observed_inside_the_label_window(tmp_path):
    runner, _, registry, task = _runtime(tmp_path)
    frame = pd.DataFrame({
        "score": [0.1, 0.2, 0.3, 0.4],
        "score_time": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-03-15", "2026-03-20"]),
    })
    dataset = _register(registry, tmp_path, task.id, "leak", frame)
    result = runner.invoke(
        ToolRef("strategy", "diagnose_time_leakage"),
        {
            "dataset_id": dataset.id,
            "feature_cols": ["score"],
            "observation_time_col": "score_time",
            "label_window_start": "2026-03-01",
            "label_window_end": "2026-03-31",
        },
        task_id=task.id,
    ).output
    assert result["evidence_only"] is True
    [feature] = result["features"]
    assert feature["feature"] == "score"
    assert result["red_flags"], "rows observed inside the label window must be flagged"


def test_selection_bias_tool_reports_both_populations_without_a_verdict(tmp_path):
    runner, _, registry, task = _runtime(tmp_path)
    approval = _register(registry, tmp_path, task.id, "approval",
                         pd.DataFrame({"income": [5.0, 6.0, 7.0], "bad": [0, 0, 1]}))
    risk = _register(registry, tmp_path, task.id, "risk",
                     pd.DataFrame({"income": [1.0, 2.0, 5.0, 6.0, 7.0], "bad": [1, 1, 0, 0, 1]}))
    result = runner.invoke(
        ToolRef("strategy", "diagnose_selection_bias"),
        {"approval_dataset_id": approval.id, "risk_dataset_id": risk.id,
         "columns": ["income"], "target": "bad"},
        task_id=task.id,
    ).output
    assert result["approval"]["count"] == 3
    assert result["risk"]["count"] == 5
    assert result["approval"]["bad_rate"] == pytest.approx(1 / 3)
    assert result["risk"]["bad_rate"] == pytest.approx(3 / 5)
    assert "biased" not in str(result).lower()


def test_segment_disparity_tool_counts_rates_per_segment(tmp_path):
    runner, _, registry, task = _runtime(tmp_path)
    frame = pd.DataFrame({
        "segment": ["A", "A", "B", "B"],
        "action": ["approval", "reject", "approval", "approval"],
        "reason": [None, "R1", None, None],
        "bad": [0, 1, 1, 0],
    })
    dataset = _register(registry, tmp_path, task.id, "applied", frame)
    result = runner.invoke(
        ToolRef("strategy", "segment_disparity_evidence"),
        {"dataset_id": dataset.id, "segment_col": "segment", "action_col": "action",
         "reason_code_col": "reason", "target_col": "bad"},
        task_id=task.id,
    ).output
    by_segment = {row["segment"]: row for row in result["segments"]}
    assert by_segment["A"]["approval_rate"] == pytest.approx(0.5)
    assert by_segment["B"]["approval_rate"] == pytest.approx(1.0)
    assert result["overall"]["count"] == 4
    assert result["phrasing_draft_notice"] == NEUTRAL_PHRASING_DRAFT_NOTICE
    assert "unfair" not in str(result["segments"]).lower()


def test_diagnostics_tools_refuse_datasets_owned_by_another_task(tmp_path):
    runner, _, registry, task = _runtime(tmp_path)
    dataset = _register(registry, tmp_path, "other-task", "foreign",
                        pd.DataFrame({"segment": ["A"], "action": ["approval"], "bad": [0]}))
    result = runner.invoke(
        ToolRef("strategy", "segment_disparity_evidence"),
        {"dataset_id": dataset.id, "segment_col": "segment",
         "action_col": "action", "target_col": "bad"},
        task_id=task.id,
    )
    assert result.ok is False
    assert result.output is None
    assert "dataset not found" in str(result.error)
