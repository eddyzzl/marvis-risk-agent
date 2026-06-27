import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, PluginRepository, init_db
from marvis.feature.metrics import feature_metrics
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner
from marvis.settings import build_settings


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
    return runner, registry, data_repo, backend


def _register_sample(registry, tmp_path):
    frame = pd.DataFrame({
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "x2": [2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 1000.0],
        "cat": ["a", "b", "a", "b", "a", "b", "a", np.nan],
        "missing": [1.0, np.nan, 3.0, np.nan, 5.0, 6.0, np.nan, 8.0],
        "split": ["base", "base", "base", "base", "cmp", "cmp", "cmp", "cmp"],
        "y": [0, 0, 0, 1, 0, 1, 1, 1],
    })
    path = tmp_path / "sample.csv"
    frame.to_csv(path, index=False)
    return registry.register_from_upload("task-feature", path, role="sample")


def test_feature_pack_tools_round_trip_via_runner(tmp_path):
    runner, registry, repo, backend = _runtime(tmp_path)
    dataset = _register_sample(registry, tmp_path)

    metrics = runner.invoke(
        ToolRef("feature", "compute_feature_metrics"),
        {"dataset_id": dataset.id, "features": ["x1", "x2"], "target_col": "y", "bins": 3},
        task_id="task-feature",
    )
    binned = runner.invoke(
        ToolRef("feature", "bin_feature"),
        {"dataset_id": dataset.id, "feature": "x1", "target_col": "y", "method": "equal_frequency", "max_bins": 3},
        task_id="task-feature",
    )
    psi = runner.invoke(
        ToolRef("feature", "compute_psi"),
        {
            "dataset_id": dataset.id,
            "feature": "x1",
            "base_filter": {"column": "split", "op": "eq", "value": "base"},
            "compare_filter": {"column": "split", "op": "eq", "value": "cmp"},
            "bins": 2,
        },
        task_id="task-feature",
    )
    correlation = runner.invoke(
        ToolRef("feature", "correlation_analysis"),
        {"dataset_id": dataset.id, "features": ["x1", "x2"], "threshold": 0.8},
        task_id="task-feature",
    )
    woe = runner.invoke(
        ToolRef("feature", "woe_encode"),
        {"dataset_id": dataset.id, "features": ["x1"], "target_col": "y", "max_bins": 2},
        task_id="task-feature",
    )
    onehot = runner.invoke(
        ToolRef("feature", "onehot_encode"),
        {"dataset_id": dataset.id, "columns": ["cat"], "max_categories": 3},
        task_id="task-feature",
    )
    normalized = runner.invoke(
        ToolRef("feature", "normalize"),
        {"dataset_id": dataset.id, "columns": ["amount"], "method": "minmax"},
        task_id="task-feature",
    )
    imputed = runner.invoke(
        ToolRef("feature", "impute_missing"),
        {"dataset_id": dataset.id, "columns": ["missing"], "strategy": "median"},
        task_id="task-feature",
    )
    capped = runner.invoke(
        ToolRef("feature", "cap_outliers"),
        {"dataset_id": dataset.id, "columns": ["amount"], "method": "quantile", "lower_q": 0.0, "upper_q": 0.75},
        task_id="task-feature",
    )
    crossed = runner.invoke(
        ToolRef("feature", "cross_features"),
        {
            "dataset_id": dataset.id,
            "recipe": [
                {"kind": "ratio", "num": "x1", "den": "x2"},
                {"kind": "agg", "group": "cat", "value": "amount", "aggs": ["mean"]},
            ],
        },
        task_id="task-feature",
    )

    for result in [metrics, binned, psi, correlation, woe, onehot, normalized, imputed, capped, crossed]:
        assert result.ok is True, result.error

    assert 0.0 <= metrics.output["metrics"][0]["ks"] <= 1.0
    assert 0.0 <= metrics.output["metrics"][0]["auc"] <= 1.0
    assert binned.output["total_iv"] >= 0.0
    assert psi.output["psi"] >= 0.0
    assert correlation.output["collinear_pairs"]

    _assert_registered_frame(repo, registry, backend, woe.output["result_dataset_id"], ["x1_woe"])
    _assert_registered_frame(repo, registry, backend, onehot.output["result_dataset_id"], ["cat_a", "cat_b"])
    _assert_registered_frame(repo, registry, backend, normalized.output["result_dataset_id"], ["amount"])
    _assert_registered_frame(repo, registry, backend, imputed.output["result_dataset_id"], ["missing"])
    _assert_registered_frame(repo, registry, backend, capped.output["result_dataset_id"], ["amount"])
    _assert_registered_frame(
        repo,
        registry,
        backend,
        crossed.output["result_dataset_id"],
        ["x1_ratio_x2", "amount_by_cat_mean"],
    )

    assert woe.output["woe_maps"]["x1"]["woe_by_bin"]
    assert normalized.output["scaler_params"]["amount"]["min"] == 10.0
    assert imputed.output["fill_values"]["missing"] == 5.0
    assert capped.output["bounds"]["amount"]["upper"] == 62.5
    assert crossed.output["new_columns"] == ["x1_ratio_x2", "amount_by_cat_mean"]


def test_feature_metrics_tool_drops_unlabeled_target_rows_when_confirmed(tmp_path):
    runner, registry, _repo, _backend = _runtime(tmp_path)
    frame = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "y": [0.0, 0.0, 1.0, np.nan, 1.0, np.nan, 0.0, 1.0],
    })
    path = tmp_path / "unlabeled.csv"
    frame.to_csv(path, index=False)
    dataset = registry.register_from_upload("task-feature", path, role="sample")

    # Without confirmation the gate stops and reports the NaN labels (never coerced to 0).
    blocked = runner.invoke(
        ToolRef("feature", "compute_feature_metrics"),
        {"dataset_id": dataset.id, "features": ["x"], "target_col": "y", "bins": 3},
        task_id="task-feature",
    )
    assert blocked.ok is False
    assert blocked.error_kind == "nan_label_not_confirmed"
    assert blocked.error_detail["n_nan"] == 2

    # Confirmed: the unlabeled rows are dropped (not coerced), metrics match the labeled subset.
    result = runner.invoke(
        ToolRef("feature", "compute_feature_metrics"),
        {
            "dataset_id": dataset.id,
            "features": ["x"],
            "target_col": "y",
            "bins": 3,
            "drop_nan_labels": True,
        },
        task_id="task-feature",
    )

    expected = feature_metrics(
        frame["x"].to_numpy(dtype=float),
        frame["y"].to_numpy(dtype=float),
        feature="x",
        bins=3,
    )
    assert result.ok is True, result.error
    assert result.output["nan_labels_dropped"] == 2
    actual = result.output["metrics"][0]
    assert actual["iv"] == pytest.approx(expected.iv)
    assert actual["ks"] == pytest.approx(expected.ks)
    assert actual["auc"] == pytest.approx(expected.auc)


def test_feature_metrics_computes_collinear_only_when_vif_selected(tmp_path):
    """Optional metrics are computed only when selected (spec §2: 选了才算).

    No selection → base per-feature metrics only, no collinear section. Selecting
    VIF adds the collinear / VIF block alongside the base metrics.
    """
    runner, registry, _repo, _backend = _runtime(tmp_path)
    dataset = _register_sample(registry, tmp_path)

    base = runner.invoke(
        ToolRef("feature", "compute_feature_metrics"),
        {"dataset_id": dataset.id, "features": ["x1", "x2"], "target_col": "y", "bins": 3},
        task_id="task-feature",
    )
    assert base.ok is True, base.error
    assert "collinear" not in base.output

    with_vif = runner.invoke(
        ToolRef("feature", "compute_feature_metrics"),
        {
            "dataset_id": dataset.id,
            "features": ["x1", "x2"],
            "target_col": "y",
            "bins": 3,
            "metrics": ["vif"],
        },
        task_id="task-feature",
    )
    assert with_vif.ok is True, with_vif.error
    assert with_vif.output["metrics"]  # base per-feature metrics still present
    collinear = with_vif.output["collinear"]
    assert set(collinear["vif"]) == {"x1", "x2"}
    assert collinear["collinear_pairs"]  # x1/x2 are strongly correlated in the fixture


def test_head_tail_lift_is_risk_direction_aware_and_deterministic():
    """head/tail lift slices by RISK direction (corr sign), not raw feature magnitude,
    so a feature and its negation report the same high-risk head. Tiny N → None."""
    from marvis.feature.metrics import head_tail_lift

    n = 200
    feature = np.arange(n, dtype=float)
    target = (feature >= 180).astype(float)  # top 20 rows bad → base rate 0.10

    out = head_tail_lift(feature, target)
    assert out["lift_head_10"] == pytest.approx(10.0)  # highest-risk 10% all bad: 1.0/0.10
    assert out["lift_tail_10"] == pytest.approx(0.0)    # lowest-risk 10% all good
    assert out["lift_head_5"] == pytest.approx(10.0)

    # Negating the feature flips the correlation sign; the high-risk end must still be
    # tracked (head stays the bad end), proving risk-direction awareness.
    flipped = head_tail_lift(-feature, target)
    assert flipped["lift_head_10"] == pytest.approx(10.0)
    assert flipped["lift_tail_10"] == pytest.approx(0.0)

    # Deterministic: identical inputs → identical output.
    assert head_tail_lift(feature, target) == out

    # Too few labelled rows → None (not a misleading 1-row slice).
    tiny = head_tail_lift(np.arange(5.0), np.array([0.0, 1.0, 0.0, 1.0, 0.0]))
    assert all(value is None for value in tiny.values())


def test_feature_metrics_adds_head_tail_lift_only_when_selected(tmp_path):
    """The head/tail lift keys ride inside each per-feature metrics dict, present ONLY
    when the metric was selected (spec §2: 选了才算)."""
    runner, registry, _repo, _backend = _runtime(tmp_path)
    dataset = _register_sample(registry, tmp_path)

    base = runner.invoke(
        ToolRef("feature", "compute_feature_metrics"),
        {"dataset_id": dataset.id, "features": ["x1", "x2"], "target_col": "y", "bins": 3},
        task_id="task-feature",
    )
    assert base.ok is True, base.error
    assert "lift_head_5" not in base.output["metrics"][0]

    selected = runner.invoke(
        ToolRef("feature", "compute_feature_metrics"),
        {
            "dataset_id": dataset.id,
            "features": ["x1", "x2"],
            "target_col": "y",
            "bins": 3,
            "metrics": ["head_tail_lift"],
        },
        task_id="task-feature",
    )
    assert selected.ok is True, selected.error
    for key in ("lift_head_5", "lift_head_10", "lift_tail_5", "lift_tail_10"):
        assert key in selected.output["metrics"][0]  # gating wired the keys into the row


def test_feature_importance_is_deterministic_and_ranks_signal():
    """Model-based gain importance is bit-reproducible (pinned seed + single-thread +
    deterministic LGB) and ranks a real signal above noise; single-class → None."""
    from marvis.feature.importance import feature_importance

    rng = np.random.RandomState(0)
    n = 600
    signal = rng.normal(size=n)
    noise = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(1.5 * signal)))
    y = (rng.uniform(size=n) < p).astype(float)
    frame = pd.DataFrame({"signal": signal, "noise": noise, "y": y})

    first = feature_importance(frame, ["signal", "noise"], "y")
    second = feature_importance(frame, ["signal", "noise"], "y")
    assert first == second  # bit-identical across runs (determinism invariant)
    assert first["signal"] > first["noise"]  # the predictive feature ranks higher
    assert sum(first.values()) == pytest.approx(1.0)  # normalised to fraction of gain

    degenerate = feature_importance(
        pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [0.0, 0.0, 0.0, 0.0]}), ["x"], "y"
    )
    assert degenerate["x"] is None  # single-class target → cannot train → None


def test_feature_metrics_adds_importance_only_when_selected(tmp_path):
    """Importance rides inside each per-feature metrics dict, present ONLY when the
    metric was selected (spec §2: 选了才算) — no model trains otherwise."""
    runner, registry, _repo, _backend = _runtime(tmp_path)
    dataset = _register_sample(registry, tmp_path)

    base = runner.invoke(
        ToolRef("feature", "compute_feature_metrics"),
        {"dataset_id": dataset.id, "features": ["x1", "x2"], "target_col": "y", "bins": 3},
        task_id="task-feature",
    )
    assert base.ok is True, base.error
    assert "importance" not in base.output["metrics"][0]

    selected = runner.invoke(
        ToolRef("feature", "compute_feature_metrics"),
        {
            "dataset_id": dataset.id,
            "features": ["x1", "x2"],
            "target_col": "y",
            "bins": 3,
            "metrics": ["importance"],
        },
        task_id="task-feature",
    )
    assert selected.ok is True, selected.error
    assert "importance" in selected.output["metrics"][0]  # gating wired the key in


def test_feature_pack_screen_features_via_runner(tmp_path):
    """The feature pack exposes leakage-aware screening (form B §4), reusing the shared
    screen_features so FEATURE and MODELING screen identically (spec §0/§7)."""
    runner, registry, _repo, _backend = _runtime(tmp_path)
    dataset = _register_sample(registry, tmp_path)

    result = runner.invoke(
        ToolRef("feature", "screen_features"),
        {"dataset_id": dataset.id, "features": ["x1", "x2", "amount", "missing"], "target_col": "y"},
        task_id="task-feature",
    )

    assert result.ok is True, result.error
    # the screen yields a selected feature set + the leakage/unusable buckets (form B)
    for key in ("selected", "ranked", "leakage", "suspected", "unusable", "scores"):
        assert key in result.output
    assert isinstance(result.output["selected"], list)
    assert result.output["n_screened"] >= 0


def test_screen_features_continuous_skips_leakage_and_keeps_all(tmp_path):
    """A non-binary (continuous) target skips the binary-only leakage KS screen: every
    candidate is selected (ks None), no leakage/suspected flags, and a skip note is set."""
    runner, registry, _repo, _backend = _runtime(tmp_path)
    dataset = _register_sample(registry, tmp_path)

    result = runner.invoke(
        ToolRef("feature", "screen_features"),
        {
            "dataset_id": dataset.id,
            "features": ["x1", "x2", "missing"],
            "target_col": "amount",
            "target_type": "continuous",
        },
        task_id="task-feature",
    )

    assert result.ok is True, result.error
    assert set(result.output["selected"]) == {"x1", "x2", "missing"}
    assert result.output["leakage"] == []
    assert result.output["suspected"] == []
    assert result.output["note"] == "非二分类目标：跳过泄漏KS筛选，保留全部候选特征"
    # Continuous screen reports missing_rate/unique_count only — KS is None (not computed).
    assert result.output["scores"]["x1"]["ks"] is None
    assert "missing_rate" in result.output["scores"]["x1"]


def test_screen_features_binary_default_runs_leakage_screen_unchanged(tmp_path):
    """Without target_type (or target_type='binary') the leakage screen runs as before:
    no skip note, KS computed per feature."""
    runner, registry, _repo, _backend = _runtime(tmp_path)
    dataset = _register_sample(registry, tmp_path)

    default = runner.invoke(
        ToolRef("feature", "screen_features"),
        {"dataset_id": dataset.id, "features": ["x1", "x2"], "target_col": "y"},
        task_id="task-feature",
    )
    binary = runner.invoke(
        ToolRef("feature", "screen_features"),
        {"dataset_id": dataset.id, "features": ["x1", "x2"], "target_col": "y", "target_type": "binary"},
        task_id="task-feature",
    )

    assert default.ok is True, default.error
    assert binary.ok is True, binary.error
    assert "note" not in default.output  # binary path carries no skip note
    assert default.output["selected"] == binary.output["selected"]
    # KS is computed for at least one scored feature on the binary path.
    assert any(score.get("ks") is not None for score in default.output["scores"].values())


def _assert_registered_frame(repo, registry, backend, dataset_id: str, expected_columns: list[str]) -> None:
    dataset = repo.get_dataset(dataset_id)
    assert dataset is not None
    frame = backend.read_frame(registry.resolve_path(dataset.id))
    for column in expected_columns:
        assert column in frame.columns
