"""S3 Commit 3: 稳定性趋势工具 tests.

Covers: INV-1 内核一致性——趋势 score PSI 与 monitor_run 同款 bin_distribution+
compute_psi 逐值相等；feature CSI 同款均匀期望；缺基准 typed error (missing_baseline)
经 subprocess runner；month_gap 红旗；以及一个训练→逐月打分→趋势的端到端 slow 测试。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, PluginRepository, TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.packs.analysis.trend import (
    feature_csi_trend,
    level_for,
    score_stability_trend,
)
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner
from marvis.settings import build_settings
from marvis.validation.binning import bin_distribution, compute_psi


def _runtime(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    plugin_repo = PluginRepository(settings.db_path)
    plugin_registry = PluginRegistry(plugin_repo)
    packs_root = Path(__file__).parents[1] / "marvis" / "packs"
    load_builtin_packs(plugin_registry, packs_root)
    runner = ToolRunner(
        ToolRegistry(plugin_registry),
        plugin_repo,
        python_executable=sys.executable,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )
    data_repo = DatasetRepository(settings.db_path)
    backend = DataBackend(settings.datasets_dir)
    registry = DatasetRegistry(data_repo, backend, settings.datasets_dir)
    task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="趋势样例",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "source"),
            algorithm="lr",
            run_mode="agent",
        )
    )
    return runner, registry, task


def _baseline_from_scores(train_scores: np.ndarray, bin_count: int = 10) -> dict:
    from marvis.validation.binning import equal_frequency_bin_edges

    edges = equal_frequency_bin_edges(train_scores, bin_count)
    return {
        "score_edges": [float(v) for v in edges],
        "score_distribution": {
            "train": {"bin_proportions": [float(v) for v in bin_distribution(train_scores, edges)]}
        },
        "feature_distributions": {},
    }


def test_score_stability_trend_matches_monitor_kernel():
    """INV-1: 趋势的每月 PSI 必须与 monitor_run 同款 bin_distribution+compute_psi 逐值相等。"""
    rng = np.random.default_rng(7)
    train_scores = rng.uniform(0, 1, size=500)
    baseline = _baseline_from_scores(train_scores)
    edges = np.asarray(baseline["score_edges"], dtype=float)
    expected = np.asarray(baseline["score_distribution"]["train"]["bin_proportions"], dtype=float)

    # two months, one stable one drifted
    frames = [
        ("2025-01", pd.DataFrame({"model_score": rng.uniform(0, 1, size=300)})),
        ("2025-02", pd.DataFrame({"model_score": rng.uniform(0.3, 1.0, size=300)})),
    ]
    result = score_stability_trend(baseline, frames, score_col="model_score")

    for month, frame in frames:
        point = next(p for p in result.trend if p.month == month)
        scores = frame["model_score"].to_numpy(dtype=float)
        manual = float(compute_psi(expected, bin_distribution(scores, edges)))
        assert point.metric == pytest.approx(manual)
    # the drifted month should read a higher PSI than the stable one
    m1 = next(p for p in result.trend if p.month == "2025-01").metric
    m2 = next(p for p in result.trend if p.month == "2025-02").metric
    assert m2 > m1


def test_feature_csi_trend_matches_uniform_expected_kernel():
    rng = np.random.default_rng(11)
    quantile_edges = list(np.quantile(rng.uniform(0, 1, size=500), np.linspace(0, 1, 11)))
    baseline = {
        "score_edges": [0.0, 1.0],
        "score_distribution": {"train": {"bin_proportions": [1.0]}},
        "feature_distributions": {"x1": {"quantile_edges": quantile_edges}},
    }
    frame = pd.DataFrame({"x1": rng.uniform(0, 1, size=400), "model_score": rng.uniform(0, 1, size=400)})
    result = feature_csi_trend(baseline, [("2025-01", frame)])
    point = result.trend[0]
    # manual CSI: bin_distribution(values, quantile_edges) vs uniform 1/bin_count
    edges = np.asarray(quantile_edges, dtype=float)
    actual = bin_distribution(frame["x1"].to_numpy(dtype=float), edges)
    bin_count = edges.size - 1
    manual = float(compute_psi(np.full(bin_count, 1.0 / bin_count), actual))
    assert point.metric == pytest.approx(manual)


def test_level_for_bands():
    assert level_for(0.05) == "green"
    assert level_for(0.15) == "amber"
    assert level_for(0.30) == "red"
    assert level_for(None) == "n/a"


def test_month_gap_red_flag():
    baseline = {
        "score_edges": [0.0, 0.5, 1.0],
        "score_distribution": {"train": {"bin_proportions": [0.5, 0.5]}},
        "feature_distributions": {},
    }
    frames = [
        ("2025-01", pd.DataFrame({"model_score": [0.1, 0.6, 0.2, 0.8]})),
        ("2025-03", pd.DataFrame({"model_score": [0.1, 0.6, 0.2, 0.8]})),  # skips 2025-02
    ]
    result = score_stability_trend(baseline, frames, score_col="model_score")
    assert "month_gap" in {flag["kind"] for flag in result.red_flags}


@pytest.mark.slow
def test_score_stability_trend_missing_baseline_typed_error(tmp_path):
    """An experiment with no artifact/baseline surfaces missing_baseline typed error."""
    runner, _registry, task = _runtime(tmp_path)
    from marvis.packs.modeling.experiment import ExperimentStore
    from marvis.packs.modeling.contracts import TrainConfig

    settings = build_settings(tmp_path / "workspace")
    store = ExperimentStore(settings.db_path)
    experiment_id = store.create(
        task.id,
        "lr",
        TrainConfig(
            dataset_id="ds",
            features=("x1", "x2"),
            target_col="y",
            split_col="split",
            split_values={"train": "train", "test": "test", "oot": "oot"},
            params={},
            seed=23,
            early_stopping_rounds=None,
        ),
    )  # created, never trained -> no artifact_id
    result = runner.invoke(
        ToolRef("analysis", "score_stability_trend"),
        {"experiment_id": experiment_id, "dataset_ids": []},
        task_id=task.id,
    )
    assert result.ok is False
    assert result.error_kind == "missing_baseline"


@pytest.mark.slow
def test_score_stability_trend_end_to_end_with_trained_experiment(tmp_path):
    """Train a real LR experiment (captures a baseline), score two monthly frames,
    then run the trend tool over them -- exercises the read:experiment path."""
    runner, registry, task = _runtime(tmp_path)

    def _frame(rows, seed):
        rng = np.random.default_rng(seed)
        x1 = rng.uniform(0, 1, size=rows)
        x2 = rng.uniform(0, 1, size=rows)
        y = ((x1 + x2) > 1.0).astype(int)
        return pd.DataFrame({"x1": x1, "x2": x2, "y": y})

    train_frame = _frame(400, 1)
    train_frame["split"] = ["train"] * 240 + ["test"] * 100 + ["oot"] * 60
    train_path = tmp_path / "train.parquet"
    train_frame.to_parquet(train_path, index=False)
    dataset = registry.register_existing(train_path, task_id=task.id, role="modeling_sample")

    trained = runner.invoke(
        ToolRef("modeling", "train_model"),
        {
            "dataset_id": dataset.id,
            "recipe": "lr",
            "features": ["x1", "x2"],
            "target_col": "y",
            "split_col": "split",
            "split_values": {"train": "train", "test": "test", "oot": "oot"},
            "seed": 23,
        },
        task_id=task.id,
    )
    assert trained.ok is True, trained.error
    experiment_id = trained.output["experiment_id"]

    # two monthly unscored frames -> score_dataset each
    scored_ids = []
    for month_seed, month in ((5, "2025-01"), (6, "2025-02")):
        month_frame = _frame(150, month_seed)
        month_path = tmp_path / f"month_{month}.parquet"
        month_frame.to_parquet(month_path, index=False)
        month_ds = registry.register_existing(month_path, task_id=task.id, role="performance")
        scored = runner.invoke(
            ToolRef("modeling", "score_dataset"),
            {"experiment_id": experiment_id, "dataset_id": month_ds.id},
            task_id=task.id,
        )
        assert scored.ok is True, scored.error
        scored_ids.append(scored.output["result_dataset_id"])

    trend = runner.invoke(
        ToolRef("analysis", "score_stability_trend"),
        {
            "experiment_id": experiment_id,
            "dataset_ids": scored_ids,
            "months": ["2025-01", "2025-02"],
            "score_col": "model_score",
        },
        task_id=task.id,
    )
    assert trend.ok is True, trend.error
    assert trend.output["metric_name"] == "psi"
    assert [p["month"] for p in trend.output["trend"]] == ["2025-01", "2025-02"]
    for point in trend.output["trend"]:
        assert point["level"] in {"green", "amber", "red", "n/a"}
        assert point["metric"] is not None
