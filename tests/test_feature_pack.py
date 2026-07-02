import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, PluginRepository, init_db
from marvis.feature.metrics import feature_metrics
from marvis.packs.feature import tools as feature_tools
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


def test_feature_register_frame_rolls_back_parquet_when_registration_fails(tmp_path):
    class FailingRegistry:
        def register_existing(self, *_args, **_kwargs):
            raise RuntimeError("db unavailable")

    runtime = SimpleNamespace(
        datasets_root=tmp_path / "datasets",
        registry=FailingRegistry(),
    )
    source_dataset = SimpleNamespace(id="source-dataset")
    ctx = SimpleNamespace(task_id="task-feature", seed=13)

    with pytest.raises(RuntimeError, match="db unavailable"):
        feature_tools._register_frame(
            runtime,
            pd.DataFrame({"x": [1, 2, 3]}),
            source_dataset,
            ctx,
            "rollback",
        )

    feature_dir = runtime.datasets_root / ctx.task_id / "feature"
    if feature_dir.exists():
        assert not list(feature_dir.rglob("*.parquet"))


def test_bin_feature_can_enforce_monotonic_bad_rates(tmp_path):
    runner, registry, _repo, _backend = _runtime(tmp_path)
    frame = pd.DataFrame({
        "x": list(range(1, 13)),
        "y": [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
    })
    path = tmp_path / "non_monotonic.csv"
    frame.to_csv(path, index=False)
    dataset = registry.register_from_upload("task-feature", path, role="sample")

    result = runner.invoke(
        ToolRef("feature", "bin_feature"),
        {
            "dataset_id": dataset.id,
            "feature": "x",
            "target_col": "y",
            "method": "manual",
            "breakpoints": [2.5, 4.5, 6.5],
            "enforce_monotonic": True,
            "monotonic_direction": "auto",
        },
        task_id="task-feature",
    )

    assert result.ok is True, result.error
    assert result.output["monotonic_enforced"] is True
    assert result.output["monotonic_before"] is False
    assert result.output["monotonic"] is True
    assert result.output["monotonic_direction"] == "increasing"
    assert result.output["edges"] == [float("-inf"), 2.5, 6.5, float("inf")]
    assert result.output["total_iv_before_monotonic"] > result.output["total_iv"]


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
        {
            "dataset_id": dataset.id,
            "features": ["x1"],
            "target_col": "y",
            "max_bins": 2,
            "allow_full_fit": True,
        },
        task_id="task-feature",
    )
    onehot = runner.invoke(
        ToolRef("feature", "onehot_encode"),
        {"dataset_id": dataset.id, "columns": ["cat"], "max_categories": 3},
        task_id="task-feature",
    )
    normalized = runner.invoke(
        ToolRef("feature", "normalize"),
        {"dataset_id": dataset.id, "columns": ["amount"], "method": "minmax", "allow_full_fit": True},
        task_id="task-feature",
    )
    imputed = runner.invoke(
        ToolRef("feature", "impute_missing"),
        {"dataset_id": dataset.id, "columns": ["missing"], "strategy": "median", "allow_full_fit": True},
        task_id="task-feature",
    )
    capped = runner.invoke(
        ToolRef("feature", "cap_outliers"),
        {
            "dataset_id": dataset.id,
            "columns": ["amount"],
            "method": "quantile",
            "lower_q": 0.0,
            "upper_q": 0.75,
            "allow_full_fit": True,
        },
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


def test_woe_encode_fits_on_non_holdout_rows_and_applies_to_all_rows(tmp_path):
    runner, registry, _repo, backend = _runtime(tmp_path)
    frame = pd.DataFrame({
        "x": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        "y": [0, 0, 1, 1, 1, 1, 0, 0],
        "split": ["train", "test", "train", "test", "oot", "oot", "oot", "oot"],
    })
    path = tmp_path / "woe_holdout.csv"
    frame.to_csv(path, index=False)
    dataset = registry.register_from_upload("task-feature", path, role="sample")

    result = runner.invoke(
        ToolRef("feature", "woe_encode"),
        {
            "dataset_id": dataset.id,
            "features": ["x"],
            "target_col": "y",
            "method": "manual",
            "breakpoints": [0.5],
            "split_col": "split",
            "holdout_values": ["oot"],
        },
        task_id="task-feature",
    )

    assert result.ok is True, result.error
    woe_map = result.output["woe_maps"]["x"]
    assert woe_map["woe_by_bin"][0] > 1.0
    assert woe_map["woe_by_bin"][1] < -1.0
    encoded = backend.read_frame(registry.resolve_path(result.output["result_dataset_id"]))
    assert encoded.shape[0] == frame.shape[0]
    assert encoded.loc[0, "x_woe"] == pytest.approx(woe_map["woe_by_bin"][0])
    assert encoded.loc[4, "x_woe"] == pytest.approx(woe_map["woe_by_bin"][0])


def test_woe_encode_without_split_raises_typed_error_unless_allow_full_fit(tmp_path):
    """PREP-1: a fit-class tool with no split_col to exclude holdout rows must stop with
    a typed error (mirrors the NaN-label gate), not silently pool-fit on everything."""
    runner, registry, _repo, _backend = _runtime(tmp_path)
    dataset = _register_sample(registry, tmp_path)

    blocked = runner.invoke(
        ToolRef("feature", "woe_encode"),
        {"dataset_id": dataset.id, "features": ["x1"], "target_col": "y", "max_bins": 2},
        task_id="task-feature",
    )
    assert blocked.ok is False
    assert blocked.error_kind == "fit_requires_split"
    assert blocked.error_detail["tool"] == "woe_encode"

    confirmed = runner.invoke(
        ToolRef("feature", "woe_encode"),
        {
            "dataset_id": dataset.id,
            "features": ["x1"],
            "target_col": "y",
            "max_bins": 2,
            "allow_full_fit": True,
        },
        task_id="task-feature",
    )
    assert confirmed.ok is True, confirmed.error
    assert confirmed.output["fit_split"] == "full"
    assert confirmed.output["fit_rows"] == 8


@pytest.mark.parametrize(
    "tool_name,extra_inputs",
    [
        ("normalize", {"columns": ["amount"], "method": "minmax"}),
        ("impute_missing", {"columns": ["missing"], "strategy": "median"}),
        ("cap_outliers", {"columns": ["amount"], "method": "quantile", "lower_q": 0.0, "upper_q": 0.75}),
    ],
)
def test_stat_transform_tools_without_split_raise_typed_error_unless_allow_full_fit(
    tmp_path, tool_name, extra_inputs
):
    """PREP-1: impute/normalize/cap must also stop without a split_col (or allow_full_fit)."""
    runner, registry, _repo, _backend = _runtime(tmp_path)
    dataset = _register_sample(registry, tmp_path)

    blocked = runner.invoke(
        ToolRef("feature", tool_name),
        {"dataset_id": dataset.id, **extra_inputs},
        task_id="task-feature",
    )
    assert blocked.ok is False
    assert blocked.error_kind == "fit_requires_split"
    assert blocked.error_detail["tool"] == tool_name

    confirmed = runner.invoke(
        ToolRef("feature", tool_name),
        {"dataset_id": dataset.id, "allow_full_fit": True, **extra_inputs},
        task_id="task-feature",
    )
    assert confirmed.ok is True, confirmed.error
    assert confirmed.output["fit_split"] == "full"
    assert confirmed.output["fit_rows"] == 8


def test_woe_encode_default_holdout_excludes_test_and_oot(tmp_path):
    """PREP-1: the default holdout changed from ("oot",) to ("test", "oot") — WOE fit
    must never see test-split labels even without an explicit holdout_values override."""
    runner, registry, _repo, backend = _runtime(tmp_path)
    frame = pd.DataFrame({
        "x": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        # test rows are flipped relative to train — if test leaked into the fit,
        # the WOE direction for bin 0 would collapse toward 0 instead of staying > 1.
        "y": [0, 0, 1, 1, 1, 1, 0, 0],
        "split": ["train", "train", "train", "train", "test", "test", "oot", "oot"],
    })
    path = tmp_path / "woe_default_holdout.csv"
    frame.to_csv(path, index=False)
    dataset = registry.register_from_upload("task-feature", path, role="sample")

    result = runner.invoke(
        ToolRef("feature", "woe_encode"),
        {
            "dataset_id": dataset.id,
            "features": ["x"],
            "target_col": "y",
            "method": "manual",
            "breakpoints": [0.5],
            "split_col": "split",
        },
        task_id="task-feature",
    )

    assert result.ok is True, result.error
    assert result.output["fit_split"] == "train"
    assert result.output["fit_rows"] == 4
    woe_map = result.output["woe_maps"]["x"]
    # Hand-computed train-only WOE (train rows only: x=0→y=[0,0], x=1→y=[1,1]).
    # bin0 (x<0.5): all good (y=0) -> woe = ln((1+0.5)/(0+0.5)) - ln((4+0.5)/(4+0.5))
    good_bin0, bad_bin0 = 2, 0
    good_bin1, bad_bin1 = 0, 2
    total_good, total_bad = 2, 2
    expected_woe0 = np.log((good_bin0 + 0.5) / (total_good + 0.5)) - np.log((bad_bin0 + 0.5) / (total_bad + 0.5))
    expected_woe1 = np.log((good_bin1 + 0.5) / (total_good + 0.5)) - np.log((bad_bin1 + 0.5) / (total_bad + 0.5))
    assert woe_map["woe_by_bin"][0] == pytest.approx(expected_woe0)
    assert woe_map["woe_by_bin"][1] == pytest.approx(expected_woe1)


def test_impute_normalize_cap_fit_only_on_train_rows_with_skewed_holdout(tmp_path):
    """PREP-1: with split_col set, fill values / scaler params / capping bounds must be
    computed from train rows only — verified against a hand-computed train-only stat,
    on a fixture where train and test/oot distributions are deliberately skewed."""
    runner, registry, _repo, backend = _runtime(tmp_path)
    frame = pd.DataFrame({
        # train: 1..6 (median 3.5); test/oot: wildly different values that would drag
        # the pooled median/mean/bounds if they leaked into the fit.
        "amount": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 500.0, 600.0],
        "missing": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0, np.nan, 900.0],
        "split": ["train"] * 6 + ["test", "oot"],
    })
    path = tmp_path / "skewed_holdout.csv"
    frame.to_csv(path, index=False)
    dataset = registry.register_from_upload("task-feature", path, role="sample")

    normalized = runner.invoke(
        ToolRef("feature", "normalize"),
        {
            "dataset_id": dataset.id,
            "columns": ["amount"],
            "method": "minmax",
            "split_col": "split",
        },
        task_id="task-feature",
    )
    imputed = runner.invoke(
        ToolRef("feature", "impute_missing"),
        {
            "dataset_id": dataset.id,
            "columns": ["missing"],
            "strategy": "median",
            "split_col": "split",
        },
        task_id="task-feature",
    )
    capped = runner.invoke(
        ToolRef("feature", "cap_outliers"),
        {
            "dataset_id": dataset.id,
            "columns": ["amount"],
            "method": "quantile",
            "lower_q": 0.0,
            "upper_q": 1.0,
            "split_col": "split",
        },
        task_id="task-feature",
    )

    for result in (normalized, imputed, capped):
        assert result.ok is True, result.error
        assert result.output["fit_split"] == "train"
        assert result.output["fit_rows"] == 6

    # Train-only hand-computed stats (rows 0-5 only: amount=[1..6], missing=[1,NaN,3,4,5,6]).
    assert normalized.output["scaler_params"]["amount"]["min"] == 1.0
    assert normalized.output["scaler_params"]["amount"]["max"] == 6.0
    assert imputed.output["fill_values"]["missing"] == 4.0  # median of [1,3,4,5,6]
    assert capped.output["bounds"]["amount"]["lower"] == 1.0
    assert capped.output["bounds"]["amount"]["upper"] == 6.0

    # Transform still applies to every row (holdout rows are transformed, not dropped).
    normalized_frame = backend.read_frame(registry.resolve_path(normalized.output["result_dataset_id"]))
    assert normalized_frame.shape[0] == frame.shape[0]
    assert normalized_frame["amount"].iloc[-1] > 1.0  # holdout outlier scales past the [0,1] train range

    capped_frame = backend.read_frame(registry.resolve_path(capped.output["result_dataset_id"]))
    assert capped_frame["amount"].iloc[-1] == 6.0  # holdout outlier clipped to the train-fit upper bound
    assert capped_frame["amount"].iloc[-2] == 6.0


def test_feature_transform_tools_return_feature_error_for_missing_columns(tmp_path):
    runner, registry, _repo, _backend = _runtime(tmp_path)
    dataset = _register_sample(registry, tmp_path)

    result = runner.invoke(
        ToolRef("feature", "normalize"),
        {"dataset_id": dataset.id, "columns": ["missing_column"], "method": "minmax"},
        task_id="task-feature",
    )

    assert result.ok is False
    assert result.error_kind == "execution"
    assert "missing columns: missing_column" in result.error


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


def test_screen_features_reports_excluded_categorical_when_features_inferred(tmp_path):
    """PREP-3/FS-3: when `features` is omitted, candidate inference silently used to drop
    string columns; screen_features must now surface them as excluded_categorical instead."""
    runner, registry, _repo, _backend = _runtime(tmp_path)
    dataset = _register_sample(registry, tmp_path)

    inferred = runner.invoke(
        ToolRef("feature", "screen_features"),
        {"dataset_id": dataset.id, "features": [], "target_col": "y"},
        task_id="task-feature",
    )
    assert inferred.ok is True, inferred.error
    excluded = {item["column"]: item["cardinality"] for item in inferred.output["excluded_categorical"]}
    assert "cat" in excluded
    assert excluded["cat"] == 2  # "a"/"b" (NaN not counted)
    assert "cat" not in inferred.output["selected"]

    explicit = runner.invoke(
        ToolRef("feature", "screen_features"),
        {"dataset_id": dataset.id, "features": ["x1", "x2"], "target_col": "y"},
        task_id="task-feature",
    )
    assert explicit.ok is True, explicit.error
    # An explicit feature list is the caller's own choice — nothing to surface.
    assert explicit.output["excluded_categorical"] == []


def test_woe_encode_categorical_matches_hand_computed_woe_and_is_train_only(tmp_path):
    runner, registry, _repo, backend = _runtime(tmp_path)
    frame = pd.DataFrame({
        "chan": ["A"] * 10 + ["B"] * 10 + ["A", "B"],
        "y": [1] * 5 + [0] * 5 + [1] * 2 + [0] * 8 + [1, 0],
        "split": ["train"] * 20 + ["oot"] * 2,
    })
    path = tmp_path / "cat_woe.csv"
    frame.to_csv(path, index=False)
    dataset = registry.register_from_upload("task-feature", path, role="sample")

    blocked = runner.invoke(
        ToolRef("feature", "woe_encode_categorical"),
        {"dataset_id": dataset.id, "features": ["chan"], "target_col": "y", "min_count": 5},
        task_id="task-feature",
    )
    assert blocked.ok is False
    assert blocked.error_kind == "fit_requires_split"
    assert blocked.error_detail["tool"] == "woe_encode_categorical"

    result = runner.invoke(
        ToolRef("feature", "woe_encode_categorical"),
        {
            "dataset_id": dataset.id,
            "features": ["chan"],
            "target_col": "y",
            "min_count": 5,
            "split_col": "split",
            "holdout_values": ["oot"],
        },
        task_id="task-feature",
    )

    assert result.ok is True, result.error
    assert result.output["fit_split"] == "train"
    assert result.output["fit_rows"] == 20  # excludes the 2 oot rows
    woe_map = result.output["woe_maps"]["chan"]
    by_category = {item["category"]: item for item in woe_map["categories"]}
    total_bad, total_good, n_groups = 7, 13, 2
    expected_a = np.log(((5 + 0.5) / (total_good + 0.5 * n_groups)) / ((5 + 0.5) / (total_bad + 0.5 * n_groups)))
    assert by_category["A"]["woe"] == pytest.approx(expected_a)

    encoded = backend.read_frame(registry.resolve_path(result.output["result_dataset_id"]))
    assert encoded.shape[0] == frame.shape[0]  # oot rows still get encoded, just not fit on
    assert encoded.loc[0, "chan_woe"] == pytest.approx(by_category["A"]["woe"])
    assert encoded.loc[20, "chan_woe"] == pytest.approx(by_category["A"]["woe"])  # oot row, category A


def test_woe_encode_categorical_merges_rare_and_falls_back_for_unseen(tmp_path):
    runner, registry, _repo, backend = _runtime(tmp_path)
    frame = pd.DataFrame({
        "chan": ["A"] * 10 + ["B"] * 10 + ["C", "D", "E", "F", "G"],
        "y": [1] * 5 + [0] * 5 + [1] * 2 + [0] * 8 + [1, 0, 1, 0, 1],
    })
    path = tmp_path / "cat_woe_rare.csv"
    frame.to_csv(path, index=False)
    dataset = registry.register_from_upload("task-feature", path, role="sample")

    result = runner.invoke(
        ToolRef("feature", "woe_encode_categorical"),
        {
            "dataset_id": dataset.id,
            "features": ["chan"],
            "target_col": "y",
            "min_count": 5,
            "allow_full_fit": True,
        },
        task_id="task-feature",
    )

    assert result.ok is True, result.error
    woe_map = result.output["woe_maps"]["chan"]
    assert set(woe_map["rare_categories"]) == {"C", "D", "E", "F", "G"}
    by_category = {item["category"]: item for item in woe_map["categories"]}
    assert "C" not in by_category

    encoded = backend.read_frame(registry.resolve_path(result.output["result_dataset_id"]))
    rare_woe = by_category["__rare__"]["woe"]
    assert encoded.loc[frame["chan"] == "C", "chan_woe"].iloc[0] == pytest.approx(rare_woe)


def test_woe_encode_categorical_without_split_raises_typed_error_unless_allow_full_fit(tmp_path):
    runner, registry, _repo, _backend = _runtime(tmp_path)
    dataset = _register_sample(registry, tmp_path)

    blocked = runner.invoke(
        ToolRef("feature", "woe_encode_categorical"),
        {"dataset_id": dataset.id, "features": ["cat"], "target_col": "y"},
        task_id="task-feature",
    )
    assert blocked.ok is False
    assert blocked.error_kind == "fit_requires_split"
    assert blocked.error_detail["tool"] == "woe_encode_categorical"

    confirmed = runner.invoke(
        ToolRef("feature", "woe_encode_categorical"),
        {"dataset_id": dataset.id, "features": ["cat"], "target_col": "y", "allow_full_fit": True},
        task_id="task-feature",
    )
    assert confirmed.ok is True, confirmed.error
    assert confirmed.output["fit_split"] == "full"
    assert confirmed.output["fit_rows"] == 8


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
    assert result.output["note"] == "非二分类目标：跳过泄漏KS筛选，已剔除常量/高缺失列"
    # Continuous screen reports missing_rate/unique_count only — KS is None (not computed).
    assert result.output["scores"]["x1"]["ks"] is None
    assert "missing_rate" in result.output["scores"]["x1"]


def test_screen_features_continuous_drops_constant_and_all_missing(tmp_path):
    """The non-binary screen still drops unusable columns (mirroring the binary path): a
    constant column and an all-NaN column land in `unusable`, not `selected`; top_k truncates."""
    runner, registry, _repo, _backend = _runtime(tmp_path)
    frame = pd.DataFrame({
        "good1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "good2": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        "const": [7.0, 7.0, 7.0, 7.0, 7.0, 7.0],
        "allnan": [np.nan] * 6,
        "amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })
    path = tmp_path / "screen_unusable.csv"
    frame.to_csv(path, index=False)
    dataset = registry.register_from_upload("task-feature", path, role="sample")

    result = runner.invoke(
        ToolRef("feature", "screen_features"),
        {
            "dataset_id": dataset.id,
            "features": ["good1", "good2", "const", "allnan"],
            "target_col": "amount",
            "target_type": "continuous",
        },
        task_id="task-feature",
    )
    assert result.ok is True, result.error
    assert set(result.output["selected"]) == {"good1", "good2"}
    reasons = {row[0]: row[1] for row in result.output["unusable"]}
    assert reasons == {"const": "constant", "allnan": "high_missing"}
    # all candidates are still scored, and the stats stay JSON-safe (no NaN/inf)
    assert result.output["scores"]["allnan"]["missing_rate"] == 1.0
    assert result.output["scores"]["const"]["unique_count"] == 1

    # top_k truncates only the proposed selected set; ranked remains the full clean
    # review surface so the user can re-add clean features outside top_k.
    capped = runner.invoke(
        ToolRef("feature", "screen_features"),
        {
            "dataset_id": dataset.id,
            "features": ["good1", "good2", "const", "allnan"],
            "target_col": "amount",
            "target_type": "continuous",
            "top_k": 1,
        },
        task_id="task-feature",
    )
    assert capped.ok is True, capped.error
    assert len(capped.output["selected"]) == 1
    assert [row[0] for row in capped.output["ranked"]] == ["good1", "good2"]


def test_screen_features_continuous_ignores_oot_for_usability_stats(tmp_path):
    """Non-binary eligibility must match binary screening's dev-mask contract: OOT
    rows do not decide whether a train/test-usable feature is missing or constant."""
    runner, registry, _repo, _backend = _runtime(tmp_path)
    frame = pd.DataFrame({
        "good_dev_missing_oot": [1.0, 2.0, 3.0, 4.0, np.nan, np.nan],
        "good_dev_constant_oot_varies": [9.0, 8.0, 7.0, 6.0, 100.0, 200.0],
        "bad_dev_constant": [5.0, 5.0, 5.0, 5.0, 6.0, 7.0],
        "amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "split": ["train", "train", "test", "test", "oot", "oot"],
    })
    path = tmp_path / "screen_holdout.csv"
    frame.to_csv(path, index=False)
    dataset = registry.register_from_upload("task-feature", path, role="sample")

    result = runner.invoke(
        ToolRef("feature", "screen_features"),
        {
            "dataset_id": dataset.id,
            "features": ["good_dev_missing_oot", "good_dev_constant_oot_varies", "bad_dev_constant"],
            "target_col": "amount",
            "split_col": "split",
            "target_type": "continuous",
        },
        task_id="task-feature",
    )

    assert result.ok is True, result.error
    assert set(result.output["selected"]) == {"good_dev_missing_oot", "good_dev_constant_oot_varies"}
    assert {row[0]: row[1] for row in result.output["unusable"]} == {"bad_dev_constant": "constant"}
    assert result.output["scores"]["good_dev_missing_oot"]["missing_rate"] == 0.0


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


def test_impute_cap_normalize_onehot_persist_preprocessing_chain_sidecar(tmp_path):
    """PREP-2: impute/cap/normalize/onehot must write a lineage sidecar next to the
    derived dataset's parquet so a model trained downstream can replay the exact
    fitted params at scoring time — not just echo them in the tool's JSON response."""
    from marvis.feature.preprocessing import read_preprocessing_chain

    runner, registry, _repo, _backend = _runtime(tmp_path)
    dataset = _register_sample(registry, tmp_path)

    imputed = runner.invoke(
        ToolRef("feature", "impute_missing"),
        {
            "dataset_id": dataset.id,
            "columns": ["missing"],
            "strategy": "median",
            "allow_full_fit": True,
        },
        task_id="task-feature",
    )
    assert imputed.ok is True, imputed.error
    imputed_chain = read_preprocessing_chain(registry.resolve_path(imputed.output["result_dataset_id"]))
    assert imputed_chain == [
        {"kind": "impute", "columns": ["missing"], "params": imputed.output["fill_values"]}
    ]

    capped = runner.invoke(
        ToolRef("feature", "cap_outliers"),
        {
            "dataset_id": imputed.output["result_dataset_id"],
            "columns": ["amount"],
            "method": "quantile",
            "lower_q": 0.0,
            "upper_q": 1.0,
            "allow_full_fit": True,
        },
        task_id="task-feature",
    )
    assert capped.ok is True, capped.error
    capped_chain = read_preprocessing_chain(registry.resolve_path(capped.output["result_dataset_id"]))
    # Chain accumulates: impute step from the source dataset + this new cap step.
    assert capped_chain == [
        {"kind": "impute", "columns": ["missing"], "params": imputed.output["fill_values"]},
        {"kind": "cap", "columns": ["amount"], "params": capped.output["bounds"]},
    ]

    normalized = runner.invoke(
        ToolRef("feature", "normalize"),
        {
            "dataset_id": capped.output["result_dataset_id"],
            "columns": ["amount"],
            "method": "minmax",
            "allow_full_fit": True,
        },
        task_id="task-feature",
    )
    assert normalized.ok is True, normalized.error
    normalized_chain = read_preprocessing_chain(registry.resolve_path(normalized.output["result_dataset_id"]))
    assert [step["kind"] for step in normalized_chain] == ["impute", "cap", "normalize"]

    onehot = runner.invoke(
        ToolRef("feature", "onehot_encode"),
        {"dataset_id": dataset.id, "columns": ["cat"]},
        task_id="task-feature",
    )
    assert onehot.ok is True, onehot.error
    onehot_chain = read_preprocessing_chain(registry.resolve_path(onehot.output["result_dataset_id"]))
    assert onehot_chain == [{"kind": "onehot", "columns": ["cat"], "params": onehot.output["mapping"]}]


def test_woe_encode_and_cross_features_do_not_append_preprocessing_steps(tmp_path):
    """WOE replay is handled separately (scorecard/woe_encode already replay their own
    WOE maps) and cross_features derives new columns rather than transforming existing
    ones in place — neither should be recorded in preprocessing_steps (PREP-2 scope)."""
    from marvis.feature.preprocessing import read_preprocessing_chain

    runner, registry, _repo, _backend = _runtime(tmp_path)
    dataset = _register_sample(registry, tmp_path)

    woe = runner.invoke(
        ToolRef("feature", "woe_encode"),
        {
            "dataset_id": dataset.id,
            "features": ["x1"],
            "target_col": "y",
            "allow_full_fit": True,
        },
        task_id="task-feature",
    )
    assert woe.ok is True, woe.error
    assert read_preprocessing_chain(registry.resolve_path(woe.output["result_dataset_id"])) == []

    crossed = runner.invoke(
        ToolRef("feature", "cross_features"),
        {"dataset_id": dataset.id, "recipe": [{"kind": "ratio", "num": "x1", "den": "x2"}]},
        task_id="task-feature",
    )
    assert crossed.ok is True, crossed.error
    assert read_preprocessing_chain(registry.resolve_path(crossed.output["result_dataset_id"])) == []


def test_screen_features_reports_suspected_categorical_numeric_codes(tmp_path):
    """PREP-5: a numeric column that looks like a nominal code (low cardinality,
    all-integer, code-like name) must be surfaced as suspected_categorical without
    changing the selected/candidate set."""
    runner, registry, _repo, _backend = _runtime(tmp_path)
    frame = pd.DataFrame({
        "region_code": [1, 2, 3, 1, 2, 3, 1, 2] * 4,
        "amount": np.linspace(10.0, 100.0, 32),
        "y": [0, 1] * 16,
    })
    path = tmp_path / "suspected_categorical.csv"
    frame.to_csv(path, index=False)
    dataset = registry.register_from_upload("task-feature", path, role="sample")

    result = runner.invoke(
        ToolRef("feature", "screen_features"),
        {"dataset_id": dataset.id, "features": [], "target_col": "y"},
        task_id="task-feature",
    )

    assert result.ok is True, result.error
    suspected = {item["column"]: item["cardinality"] for item in result.output["suspected_categorical"]}
    assert suspected == {"region_code": 3}
    # Informational only: region_code is still modeled as numeric (not dropped/changed).
    assert "region_code" in result.output["selected"]


def test_derive_date_features_tool_creates_derived_dataset(tmp_path):
    """PREP-7: derive_date_features is opt-in (not part of any default template)
    and must produce deterministic datediff/month/tenure columns registered as a
    new derived dataset, same pattern as cross_features."""
    runner, registry, _repo, backend = _runtime(tmp_path)
    frame = pd.DataFrame({
        "apply_date": ["2024-01-15", "2024-03-01", "2024-06-10"],
        "open_date": ["2023-01-15", "2023-01-01", "2023-06-01"],
        "y": [0, 1, 0],
    })
    path = tmp_path / "dates.csv"
    frame.to_csv(path, index=False)
    dataset = registry.register_from_upload("task-feature", path, role="sample")

    result = runner.invoke(
        ToolRef("feature", "derive_date_features"),
        {
            "dataset_id": dataset.id,
            "recipe": [
                {"kind": "datediff", "col": "apply_date", "anchor": "open_date", "unit": "days"},
                {"kind": "month", "col": "apply_date"},
                {"kind": "tenure_months", "col": "apply_date", "anchor": "open_date"},
            ],
        },
        task_id="task-feature",
    )

    assert result.ok is True, result.error
    assert result.output["new_columns"] == [
        "apply_date__days_since_open_date",
        "apply_date__month",
        "apply_date__months_on_book",
    ]
    derived_frame = backend.read_frame(registry.resolve_path(result.output["result_dataset_id"]))
    assert derived_frame["apply_date__days_since_open_date"].tolist() == [365.0, 425.0, 375.0]
    assert derived_frame["apply_date__month"].tolist() == [1.0, 3.0, 6.0]


def test_screen_features_reports_sentinel_columns_notice(tmp_path):
    """PREP-4: screen_features must surface a suspected sentinel/special-value column
    (isolated extreme peak, e.g. -999 "no hit") so the user can pass sentinel_values to
    impute/cap/normalize/bin_feature/woe_encode before fitting."""
    runner, registry, _repo, _backend = _runtime(tmp_path)
    rng = np.concatenate([np.linspace(1.0, 100.0, 190), np.full(10, -999.0)])
    frame = pd.DataFrame({
        "score": rng,
        "y": ([0, 1] * 100)[:200],
    })
    path = tmp_path / "sentinel.csv"
    frame.to_csv(path, index=False)
    dataset = registry.register_from_upload("task-feature", path, role="sample")

    result = runner.invoke(
        ToolRef("feature", "screen_features"),
        {"dataset_id": dataset.id, "features": ["score"], "target_col": "y"},
        task_id="task-feature",
    )

    assert result.ok is True, result.error
    assert "score" in result.output["sentinel_columns"]
    assert result.output["sentinel_columns"]["score"][0][0] == -999.0
    assert "score" in result.output["sentinel_notice"]
    assert "sentinel_values" in result.output["sentinel_notice"]


def test_impute_cap_normalize_bin_woe_accept_sentinel_values(tmp_path):
    """PREP-4: sentinel_values on impute/cap/normalize/bin_feature/woe_encode must be
    treated as missing before fitting/binning, not as real observations."""
    runner, registry, _repo, backend = _runtime(tmp_path)
    frame = pd.DataFrame({
        "amount": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -999.0, -999.0],
        "y": [0, 0, 0, 1, 0, 1, 1, 1],
    })
    path = tmp_path / "sentinel_tools.csv"
    frame.to_csv(path, index=False)
    dataset = registry.register_from_upload("task-feature", path, role="sample")

    imputed = runner.invoke(
        ToolRef("feature", "impute_missing"),
        {
            "dataset_id": dataset.id,
            "columns": ["amount"],
            "strategy": "median",
            "sentinel_values": [-999.0],
            "allow_full_fit": True,
        },
        task_id="task-feature",
    )
    assert imputed.ok is True, imputed.error
    assert imputed.output["fill_values"]["amount"] == 3.5  # median of [1..6], -999 excluded
    imputed_frame = backend.read_frame(registry.resolve_path(imputed.output["result_dataset_id"]))
    assert imputed_frame["amount"].iloc[-1] == 3.5  # sentinel rows filled with the train-only median

    capped = runner.invoke(
        ToolRef("feature", "cap_outliers"),
        {
            "dataset_id": dataset.id,
            "columns": ["amount"],
            "method": "quantile",
            "lower_q": 0.0,
            "upper_q": 1.0,
            "sentinel_values": [-999.0],
            "allow_full_fit": True,
        },
        task_id="task-feature",
    )
    assert capped.ok is True, capped.error
    assert capped.output["bounds"]["amount"]["lower"] == 1.0
    assert capped.output["bounds"]["amount"]["upper"] == 6.0  # -999 excluded from the IQR fit

    normalized = runner.invoke(
        ToolRef("feature", "normalize"),
        {
            "dataset_id": dataset.id,
            "columns": ["amount"],
            "method": "minmax",
            "sentinel_values": [-999.0],
            "allow_full_fit": True,
        },
        task_id="task-feature",
    )
    assert normalized.ok is True, normalized.error
    assert normalized.output["scaler_params"]["amount"]["min"] == 1.0
    assert normalized.output["scaler_params"]["amount"]["max"] == 6.0

    binned = runner.invoke(
        ToolRef("feature", "bin_feature"),
        {
            "dataset_id": dataset.id,
            "feature": "amount",
            "target_col": "y",
            "method": "equal_width",
            "max_bins": 2,
            "sentinel_values": [-999.0],
        },
        task_id="task-feature",
    )
    assert binned.ok is True, binned.error
    # Sentinel rows land in the NA bin rather than skewing the equal-width edges.
    assert binned.output["na_bin"] is not None
    assert binned.output["na_bin"]["count"] == 2

    woe = runner.invoke(
        ToolRef("feature", "woe_encode"),
        {
            "dataset_id": dataset.id,
            "features": ["amount"],
            "target_col": "y",
            "sentinel_values": [-999.0],
            "allow_full_fit": True,
        },
        task_id="task-feature",
    )
    assert woe.ok is True, woe.error
    assert woe.output["woe_maps"]["amount"]["na_woe"] is not None
