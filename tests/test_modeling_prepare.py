import pandas as pd
import pytest

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, init_db
from marvis.packs.modeling.prepare import ModelingError, _make_split, prepare_modeling_frame


def _runtime(tmp_path):
    db_path = tmp_path / "app.sqlite"
    datasets_root = tmp_path / "datasets"
    init_db(db_path)
    repo = DatasetRepository(db_path)
    backend = DataBackend(datasets_root)
    registry = DatasetRegistry(repo, backend, datasets_root)
    return backend, registry


def _register_frame(tmp_path, frame: pd.DataFrame):
    backend, registry = _runtime(tmp_path)
    source_path = tmp_path / "source.parquet"
    frame.to_parquet(source_path, index=False)
    dataset = registry.register_existing(
        source_path,
        task_id="task-1",
        role="sample",
        seed=0,
    )
    return backend, registry, dataset


def test_prepare_modeling_frame_uses_existing_split_and_selected_columns(tmp_path):
    frame = pd.DataFrame({
        "x1": [1, 2, 3, 4],
        "x2": [10, 20, 30, 40],
        "unused": [99, 99, 99, 99],
        "y": [0, 1, 0, 1],
        "split": ["train", "train", "test", "oot"],
    })
    backend, registry, dataset = _register_frame(tmp_path, frame)

    result = prepare_modeling_frame(
        registry,
        backend,
        dataset.id,
        target_col="y",
        feature_cols=["x1", "x2"],
        split_col="split",
        split_config=None,
        seed=3,
    )
    out = backend.read_frame(registry.resolve_path(result.id))

    assert result.role == "derived"
    assert list(out.columns) == ["x1", "x2", "y", "split"]
    assert out["split"].tolist() == ["train", "train", "test", "oot"]


def test_prepare_modeling_frame_auto_split_is_seed_reproducible(tmp_path):
    frame = pd.DataFrame({
        "row_id": list(range(100)),
        "x": [i % 7 for i in range(100)],
        "y": [i % 2 for i in range(100)],
    })
    backend, registry, dataset = _register_frame(tmp_path, frame)

    first = prepare_modeling_frame(
        registry,
        backend,
        dataset.id,
        target_col="y",
        feature_cols=["row_id", "x"],
        split_col=None,
        split_config={"test_size": 0.25},
        seed=11,
    )
    second = prepare_modeling_frame(
        registry,
        backend,
        dataset.id,
        target_col="y",
        feature_cols=["row_id", "x"],
        split_col=None,
        split_config={"test_size": 0.25},
        seed=11,
    )
    first_frame = backend.read_frame(registry.resolve_path(first.id)).sort_values("row_id")
    second_frame = backend.read_frame(registry.resolve_path(second.id)).sort_values("row_id")

    assert first_frame["split"].value_counts().to_dict() == {"train": 75, "test": 25}
    assert first_frame["split"].tolist() == second_frame["split"].tolist()


def test_prepare_modeling_frame_splits_oot_by_time_before_random_test(tmp_path):
    frame = pd.DataFrame({
        "row_id": list(range(100)),
        "month": list(range(100)),
        "x": [i % 5 for i in range(100)],
        "y": [i % 2 for i in range(100)],
    })
    backend, registry, dataset = _register_frame(tmp_path, frame)

    result = prepare_modeling_frame(
        registry,
        backend,
        dataset.id,
        target_col="y",
        feature_cols=["row_id", "month", "x"],
        split_col=None,
        split_config={"test_size": 0.25, "oot_size": 0.2, "oot_by_time": "month"},
        seed=5,
    )
    out = backend.read_frame(registry.resolve_path(result.id))

    assert out["split"].value_counts().to_dict() == {"train": 60, "oot": 20, "test": 20}
    assert out[out["split"] == "oot"]["month"].min() >= 80
    assert not (out[out["split"] == "test"]["month"] >= 80).any()


def test_make_split_grouped_keeps_each_group_on_one_side():
    """Anti-leakage (spec §2): with group_cols, every group's near-duplicate rows
    land entirely in one split set — never straddling train/test."""
    rows = []
    for group in range(20):
        for offset in range(5):  # 5 near-duplicate rows per identity+date group
            rows.append({"grp": group, "x": offset, "y": (group + offset) % 2})
    frame = pd.DataFrame(rows)

    out = _make_split(frame, {"test_size": 0.3, "group_cols": ["grp"]}, seed=7)

    # every group is wholly in a single split set
    assert (out.groupby("grp")["split"].nunique() == 1).all()
    assert set(out["split"].unique()) == {"train", "test"}
    # reproducible
    again = _make_split(frame, {"test_size": 0.3, "group_cols": ["grp"]}, seed=7)
    assert out["split"].tolist() == again["split"].tolist()


def test_make_split_blocks_empty_split():
    """A rule/ratio that would leave an expected set empty is blocked, not shipped."""
    frame = pd.DataFrame({"grp": [0, 0, 0], "x": [1, 2, 3], "y": [0, 1, 0]})
    # single group + test_size 1.0 → the whole (ungroupable) group goes to test, train empty
    with pytest.raises(ModelingError, match="为空"):
        _make_split(frame, {"test_size": 1.0, "group_cols": ["grp"]}, seed=1)


def test_prepare_modeling_frame_rejects_missing_columns(tmp_path):
    frame = pd.DataFrame({"x": [1, 2], "y": [0, 1]})
    backend, registry, dataset = _register_frame(tmp_path, frame)

    with pytest.raises(ModelingError, match="missing columns: missing_feature"):
        prepare_modeling_frame(
            registry,
            backend,
            dataset.id,
            target_col="y",
            feature_cols=["x", "missing_feature"],
            split_col=None,
            split_config=None,
        )

    with pytest.raises(ModelingError, match="missing columns: missing_target"):
        prepare_modeling_frame(
            registry,
            backend,
            dataset.id,
            target_col="missing_target",
            feature_cols=["x"],
            split_col=None,
            split_config=None,
        )
