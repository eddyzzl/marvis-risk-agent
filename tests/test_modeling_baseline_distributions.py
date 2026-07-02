"""S1b regression tests: training-time baseline distribution snapshots
(marvis/packs/modeling/tools.py::_compute_baseline_distributions).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marvis.db import ModelingRepository
from marvis.plugins.manifest import ToolRef

from test_modeling_pack import _runtime


def _linearly_separable_frame(rows: int = 240) -> pd.DataFrame:
    x1 = [((i * 37) % 101) / 100 for i in range(rows)]
    x2 = [((i * 17) % 89) / 100 for i in range(rows)]
    y = [1 if (x1[i] + x2[i]) > 1.0 else 0 for i in range(rows)]
    return pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "y": y,
        "split": ["train"] * 140 + ["test"] * 60 + ["oot"] * 40,
    })


def test_train_model_persists_baseline_distributions_with_correct_equal_frequency_bins(tmp_path):
    """(a) Training persists a baseline snapshot whose train-split bin proportions
    are exactly equal-frequency (each of the 10 bins holds ~1/10 of the train rows
    by construction) and whose feature quantile edges match a hand-computed
    numpy.quantile call on the same train rows."""
    runner, _pr, registry, _backend, settings, task = _runtime(tmp_path)
    frame = _linearly_separable_frame()
    path = tmp_path / "modeling_sample.parquet"
    frame.to_parquet(path, index=False)
    dataset = registry.register_existing(path, task_id=task.id, role="modeling_sample")

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

    artifact = ModelingRepository(settings.db_path).get_model_artifact(trained.output["artifact_id"])
    assert artifact is not None
    baseline = artifact.baseline_distributions
    assert baseline is not None
    assert baseline["bin_count"] == 10
    assert baseline["score_direction"] == "higher_is_riskier"

    train_bins = baseline["score_distribution"]["train"]["bin_proportions"]
    assert len(train_bins) == 10
    # Equal-frequency by construction: each bin holds exactly 1/10 of the 140
    # train rows (140 divides evenly by 10, so there's no rounding slack).
    assert train_bins == pytest.approx([0.1] * 10, abs=1e-9)
    assert baseline["score_distribution"]["train"]["sample_count"] == 140
    assert "test" in baseline["score_distribution"]
    assert "oot" in baseline["score_distribution"]
    assert sum(baseline["score_distribution"]["test"]["bin_proportions"]) == pytest.approx(1.0)

    # Hand-computed feature quantile edges must match numpy.quantile on the same
    # train-split raw x1/x2 values (11 equal-frequency quantile edges for 10 bins).
    train_frame = frame[frame["split"] == "train"]
    for feature in ("x1", "x2"):
        expected_edges = np.quantile(
            train_frame[feature].to_numpy(dtype=float), np.linspace(0.0, 1.0, 11)
        )
        actual_edges = baseline["feature_distributions"][feature]["quantile_edges"]
        assert actual_edges == pytest.approx(list(expected_edges), abs=1e-9)
        assert baseline["feature_distributions"][feature]["sample_count"] == 140
        assert baseline["feature_distributions"][feature]["missing_rate"] == 0.0


def test_train_model_baseline_score_edges_are_finite_open_ended(tmp_path):
    """Baseline score_edges follow the platform's equal_frequency_bin_edges
    convention: the outer edges are -inf/inf so any future score, however
    extreme, still lands in a bin instead of falling outside the range."""
    runner, _pr, registry, _backend, settings, task = _runtime(tmp_path)
    frame = _linearly_separable_frame()
    path = tmp_path / "modeling_sample.parquet"
    frame.to_parquet(path, index=False)
    dataset = registry.register_existing(path, task_id=task.id, role="modeling_sample")

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
    artifact = ModelingRepository(settings.db_path).get_model_artifact(trained.output["artifact_id"])
    edges = artifact.baseline_distributions["score_edges"]
    assert edges[0] == float("-inf")
    assert edges[-1] == float("inf")
    assert len(edges) == 11


