"""Regression tests for the V2 label-NaN confirmation gate (INV-1 / INV-2).

A NaN target must NEVER be silently coerced to a class. These tests assert the
shared helpers in ``marvis.data.labels`` either raise ``NanLabelNotConfirmedError``
(default) or drop the offending rows (opt-in), never count NaN as a 0/1 label.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from marvis.data.errors import NanLabelNotConfirmedError
from marvis.data.labels import (
    nan_label_mask,
    require_labels_confirmed,
    resolve_labeled_frame,
    resolve_modeling_splits,
)


def _frame(targets: list[float], feature: list[float] | None = None) -> pd.DataFrame:
    n = len(targets)
    return pd.DataFrame(
        {
            "x": feature if feature is not None else list(range(n)),
            "y": targets,
        }
    )


def test_nan_label_mask_flags_only_non_finite() -> None:
    frame = _frame([1.0, 0.0, np.nan, 1.0])
    mask = nan_label_mask(frame, "y")
    assert mask.tolist() == [False, False, True, False]


def test_nan_label_mask_parses_numeric_strings() -> None:
    frame = pd.DataFrame({"y": ["1", "0", "1"]})
    assert nan_label_mask(frame, "y").tolist() == [False, False, False]


def test_nan_label_mask_raises_on_non_numeric_target() -> None:
    frame = pd.DataFrame({"y": ["good", "bad"]})
    with pytest.raises((ValueError, TypeError)):
        nan_label_mask(frame, "y")


def test_resolve_labeled_frame_clean_is_passthrough() -> None:
    frame = _frame([1.0, 0.0, 1.0])
    out, dropped = resolve_labeled_frame(frame, "y", drop_nan_labels=False)
    assert dropped == 0
    assert len(out) == 3


def test_resolve_labeled_frame_raises_by_default_on_nan() -> None:
    frame = _frame([1.0, 0.0, np.nan, 1.0])
    with pytest.raises(NanLabelNotConfirmedError) as exc:
        resolve_labeled_frame(frame, "y", drop_nan_labels=False)
    err = exc.value
    assert err.target_col == "y"
    assert err.n_total == 4
    assert err.n_nan == 1
    detail = err.to_detail()
    assert detail["kind"] == "nan_label_not_confirmed"
    assert detail["n_nan"] == 1
    assert detail["n_total"] == 4


def test_resolve_labeled_frame_drops_when_confirmed() -> None:
    frame = _frame([1.0, 0.0, np.nan, 1.0])
    out, dropped = resolve_labeled_frame(frame, "y", drop_nan_labels=True)
    assert dropped == 1
    # The NaN row is gone (not coerced to 0): no fabricated "good" label.
    assert out["y"].tolist() == [1.0, 0.0, 1.0]
    assert len(out) == 3


def test_resolve_labeled_frame_regression_not_coerced_to_zero() -> None:
    # If NaN were coerced to 0, bad-rate denominator/numerator would be wrong.
    frame = _frame([1.0, np.nan, 1.0])
    out, _ = resolve_labeled_frame(frame, "y", drop_nan_labels=True)
    y = out["y"].to_numpy(dtype=float)
    assert np.array_equal(y, np.array([1.0, 1.0]))
    # The coerced-to-zero array would have been [1, 0, 1]; assert it is not.
    assert y.sum() == 2.0


def test_require_labels_confirmed_is_check_only() -> None:
    # Check-only must NOT modify/filter the frame (edges depend on the full feature column).
    frame = _frame([1.0, 0.0, np.nan, 1.0])
    with pytest.raises(NanLabelNotConfirmedError):
        require_labels_confirmed(frame, "y", drop_nan_labels=False)
    n = require_labels_confirmed(frame, "y", drop_nan_labels=True)
    assert n == 1
    # The caller's frame is untouched; the core drops the NaN target itself.
    assert len(frame) == 4
    assert require_labels_confirmed(_frame([1.0, 0.0, 1.0]), "y", drop_nan_labels=False) == 0


# --- modeling per-split resolution -------------------------------------------------


def _split_frame() -> pd.DataFrame:
    rows = []
    for split, ys in (("train", [1, 0, 1, 0]), ("test", [1, 0]), ("oot", [1, 0])):
        for y in ys:
            rows.append({"x": 1.0, "split": split, "y": float(y)})
    return pd.DataFrame(rows)


def _split(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    return frame[frame["split"] == name]


def test_modeling_splits_clean_passthrough() -> None:
    frame = _split_frame()
    train, test, oot, oot_has_labels, audit = resolve_modeling_splits(
        _split(frame, "train"), _split(frame, "test"), _split(frame, "oot"),
        target_col="y", drop_nan_labels=False,
    )
    assert oot_has_labels is True
    assert audit["total_dropped"] == 0
    assert len(train) == 4 and len(test) == 2 and len(oot) == 2


def test_modeling_splits_train_nan_raises_by_default() -> None:
    frame = _split_frame()
    train = _split(frame, "train").copy()
    train.iloc[0, train.columns.get_loc("y")] = np.nan
    with pytest.raises(NanLabelNotConfirmedError) as exc:
        resolve_modeling_splits(
            train, _split(frame, "test"), _split(frame, "oot"),
            target_col="y", drop_nan_labels=False,
        )
    assert exc.value.by_split["train"]["n_nan"] == 1


def test_modeling_splits_train_nan_dropped_when_confirmed() -> None:
    frame = _split_frame()
    train = _split(frame, "train").copy()
    train.iloc[0, train.columns.get_loc("y")] = np.nan
    train_c, test_c, oot_c, oot_has_labels, audit = resolve_modeling_splits(
        train, _split(frame, "test"), _split(frame, "oot"),
        target_col="y", drop_nan_labels=True,
    )
    assert len(train_c) == 3
    assert audit["total_dropped"] == 1
    assert not train_c["y"].isna().any()


def test_modeling_splits_oot_all_nan_is_scoring_only() -> None:
    frame = _split_frame()
    oot = _split(frame, "oot").copy()
    oot["y"] = np.nan
    train_c, test_c, oot_c, oot_has_labels, audit = resolve_modeling_splits(
        _split(frame, "train"), _split(frame, "test"), oot,
        target_col="y", drop_nan_labels=False,  # no confirmation needed for scoring-only OOT
    )
    assert oot_has_labels is False
    assert len(oot_c) == 2  # rows kept for scoring
    assert audit["total_dropped"] == 0


def test_modeling_splits_oot_partial_nan_gates() -> None:
    frame = _split_frame()
    oot = _split(frame, "oot").copy()
    oot.iloc[0, oot.columns.get_loc("y")] = np.nan  # 1 of 2 -> partial
    with pytest.raises(NanLabelNotConfirmedError) as exc:
        resolve_modeling_splits(
            _split(frame, "train"), _split(frame, "test"), oot,
            target_col="y", drop_nan_labels=False,
        )
    assert exc.value.scope == "oot"


def test_modeling_splits_oot_none_ok() -> None:
    frame = _split_frame()
    train_c, test_c, oot_c, oot_has_labels, audit = resolve_modeling_splits(
        _split(frame, "train"), _split(frame, "test"), None,
        target_col="y", drop_nan_labels=False,
    )
    assert oot_c is None
    assert oot_has_labels is False


# --- subprocess structured error_detail channel ------------------------------------

_ADHOC_NAN_TOOL = '''
from marvis.data.errors import NanLabelNotConfirmedError


def run(inputs, ctx):
    raise NanLabelNotConfirmedError(
        target_col="y", n_total=10, n_nan=3, scope="dataset",
    )
'''


def test_nan_label_error_detail_survives_subprocess(tmp_path) -> None:
    import sys

    from marvis.db import PluginRepository, init_db
    from marvis.plugins.registry import PluginRegistry, ToolRegistry
    from marvis.plugins.runner import ToolRunner

    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PluginRepository(db_path)
    runner = ToolRunner(
        ToolRegistry(PluginRegistry(repo)),
        repo,
        python_executable=sys.executable,
        datasets_root=tmp_path / "datasets",
        workspace=tmp_path / "workspace",
    )
    module = tmp_path / "nan_tool.py"
    module.write_text(_ADHOC_NAN_TOOL, encoding="utf-8")

    result = runner.invoke_adhoc(
        module=module,
        entrypoint="run",
        inputs={},
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        timeout_seconds=30,
        task_id="task-nan",
    )

    assert result.ok is False
    assert result.error_kind == "nan_label_not_confirmed"
    assert result.error_detail is not None
    assert result.error_detail["target_col"] == "y"
    assert result.error_detail["n_nan"] == 3
    assert result.error_detail["n_total"] == 10


# --- feature pack end-to-end via the tool runner -----------------------------------


def _feature_runtime(tmp_path):
    import sys

    from marvis.data.backend import DataBackend
    from marvis.data.registry import DatasetRegistry
    from marvis.db import DatasetRepository, PluginRepository, init_db
    from marvis.plugins.loader import load_builtin_packs
    from marvis.plugins.registry import PluginRegistry, ToolRegistry
    from marvis.plugins.runner import ToolRunner
    from marvis.settings import build_settings

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
    return runner, registry


def _register_nan_sample(registry, tmp_path):
    frame = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            # One missing label; everything else is a clean 0/1.
            "y": [0, 1, 0, 1, np.nan, 1, 0, 1],
        }
    )
    path = tmp_path / "sample.csv"
    frame.to_csv(path, index=False)
    return registry.register_from_upload("task-feature", path, role="sample")


def test_feature_metrics_gate_raises_without_confirmation(tmp_path):
    from marvis.plugins.manifest import ToolRef

    runner, registry = _feature_runtime(tmp_path)
    dataset = _register_nan_sample(registry, tmp_path)
    result = runner.invoke(
        ToolRef("feature", "compute_feature_metrics"),
        {"dataset_id": dataset.id, "features": ["x1"], "target_col": "y", "bins": 3},
        task_id="task-feature",
    )
    assert result.ok is False
    assert result.error_kind == "nan_label_not_confirmed"
    assert result.error_detail["n_nan"] == 1
    assert result.error_detail["n_total"] == 8


def test_feature_metrics_gate_drops_when_confirmed(tmp_path):
    from marvis.plugins.manifest import ToolRef

    runner, registry = _feature_runtime(tmp_path)
    dataset = _register_nan_sample(registry, tmp_path)
    result = runner.invoke(
        ToolRef("feature", "compute_feature_metrics"),
        {
            "dataset_id": dataset.id,
            "features": ["x1"],
            "target_col": "y",
            "bins": 3,
            "drop_nan_labels": True,
        },
        task_id="task-feature",
    )
    assert result.ok is True, result.error
    assert result.output["nan_labels_dropped"] == 1
