from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_TEST_SIZE = 0.30
DEFAULT_OOT_SIZE = 0.20
SPLIT_COLUMN = "split"


class ModelingError(ValueError):
    pass


def prepare_modeling_frame(
    registry,
    backend,
    dataset_id: str,
    *,
    target_col: str,
    feature_cols: list[str],
    split_col: str | None,
    split_config: dict | None,
    seed: int = 0,
):
    dataset = registry.get(dataset_id)
    dataset_path = registry.resolve_path(dataset.id)
    split_config = dict(split_config or {})
    requested = _requested_columns(feature_cols, target_col, split_col, split_config)
    _assert_columns_exist(dataset, requested)

    frame = backend.read_frame(dataset_path, columns=requested)
    if split_col:
        prepared = frame[_unique([*feature_cols, target_col, split_col])].copy()
    else:
        prepared = _make_split(frame, split_config, seed=seed)
        prepared = prepared[_unique([*feature_cols, target_col, SPLIT_COLUMN])].copy()

    out_path = _output_path(registry, dataset)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_parquet(out_path, index=False)
    return registry.register_existing(
        out_path,
        task_id=dataset.task_id,
        role="derived",
        anchor_target=dataset.id,
        seed=seed,
    )


def _make_split(df: pd.DataFrame, split_config: dict[str, Any] | None, seed: int) -> pd.DataFrame:
    config = dict(split_config or {})
    out = df.copy()
    out[SPLIT_COLUMN] = "train"

    remaining_index = out.index
    time_col = config.get("oot_by_time")
    if time_col:
        time_col = str(time_col)
        if time_col not in out.columns:
            raise ModelingError(f"missing columns: {time_col}")
        cutoff = out[time_col].quantile(1 - float(config.get("oot_size", DEFAULT_OOT_SIZE)))
        oot_mask = out[time_col] >= cutoff
        out.loc[oot_mask, SPLIT_COLUMN] = "oot"
        remaining_index = out.index[~oot_mask]

    test_size = float(config.get("test_size", DEFAULT_TEST_SIZE))
    test_count = int(len(remaining_index) * test_size)
    if test_count > 0:
        rng = np.random.RandomState(int(seed))
        test_index = rng.choice(remaining_index.to_numpy(), size=test_count, replace=False)
        out.loc[test_index, SPLIT_COLUMN] = "test"
    return out


def _requested_columns(
    feature_cols: list[str],
    target_col: str,
    split_col: str | None,
    split_config: dict,
) -> list[str]:
    columns = [*feature_cols, target_col]
    if split_col:
        columns.append(split_col)
    time_col = split_config.get("oot_by_time")
    if time_col:
        columns.append(str(time_col))
    return _unique(columns)


def _assert_columns_exist(dataset, columns: list[str]) -> None:
    existing = {profile.name for profile in dataset.columns}
    missing = [column for column in columns if column not in existing]
    if missing:
        raise ModelingError(f"missing columns: {', '.join(missing)}")


def _output_path(registry, dataset) -> Path:
    root = _registry_root(registry, dataset)
    return root / dataset.task_id / "modeling" / f"{dataset.id}_modeling_{uuid.uuid4().hex[:8]}.parquet"


def _registry_root(registry, dataset) -> Path:
    root = registry.resolve_path(dataset.id)
    for _part in Path(dataset.source_path).parts:
        root = root.parent
    return root


def _unique(values: list[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


__all__ = [
    "DEFAULT_OOT_SIZE",
    "DEFAULT_TEST_SIZE",
    "ModelingError",
    "SPLIT_COLUMN",
    "prepare_modeling_frame",
]
