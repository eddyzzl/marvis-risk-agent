import sys
from pathlib import Path

import numpy as np
import pandas as pd

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, PluginRepository, init_db
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


def _assert_registered_frame(repo, registry, backend, dataset_id: str, expected_columns: list[str]) -> None:
    dataset = repo.get_dataset(dataset_id)
    assert dataset is not None
    frame = backend.read_frame(registry.resolve_path(dataset.id))
    for column in expected_columns:
        assert column in frame.columns
