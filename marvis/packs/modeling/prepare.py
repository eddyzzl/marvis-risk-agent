from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from marvis.packs.modeling.errors import ModelingError


DEFAULT_TEST_SIZE = 0.30
DEFAULT_OOT_SIZE = 0.20
SPLIT_COLUMN = "split"


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
    # Anti-leakage grouping is best-effort: keep only group columns that exist in the
    # dataset (the JOIN identity key is often dropped before modeling). Missing group
    # columns fall back to per-row assignment rather than erroring.
    existing_columns = {profile.name for profile in dataset.columns}
    group_cols = split_config.get("group_cols")
    if group_cols:
        group_cols = group_cols if isinstance(group_cols, list) else [group_cols]
        split_config["group_cols"] = [str(col) for col in group_cols if str(col) in existing_columns]
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
    """Deterministically build a train/test/oot split column.

    OOT is carved out first (by a time cutoff, the credit-risk standard; or by an
    explicit random_oot request). Test is then drawn from the non-OOT remainder.
    Both random draws are GROUP-AWARE when ``group_cols`` is given: whole groups
    (e.g. same person + same day) are assigned to one side so near-duplicate samples
    never straddle train/test — the anti-leakage guardrail (spec §2). Fixed seed →
    reproducible. Empties that were expected are blocked with a clear error.
    """
    config = dict(split_config or {})
    out = df.copy()
    out[SPLIT_COLUMN] = "train"
    rng = np.random.RandomState(int(seed))
    groups = _group_ids(out, _valid_group_cols(out, config.get("group_cols")))

    # 1) OOT
    time_col = config.get("oot_by_time")
    oot_size = float(config.get("oot_size", DEFAULT_OOT_SIZE))
    expect_oot = False
    if time_col:
        time_col = str(time_col)
        if time_col not in out.columns:
            raise ModelingError(f"missing columns: {time_col}")
        cutoff = out[time_col].quantile(1 - oot_size)
        out.loc[out[time_col] >= cutoff, SPLIT_COLUMN] = "oot"
        expect_oot = True
    elif config.get("random_oot") and oot_size > 0:
        # OOT means out-of-time; only fabricate a random OOT when explicitly asked.
        train_mask = (out[SPLIT_COLUMN] == "train").to_numpy()
        chosen = _sample_groups_by_rows(groups, train_mask, int(round(train_mask.sum() * oot_size)), rng)
        out.loc[np.isin(groups, list(chosen)), SPLIT_COLUMN] = "oot"
        expect_oot = True

    # 2) test from the non-OOT remainder (test_size is a fraction of that remainder,
    #    preserving the original contract), whole groups together.
    test_size = float(config.get("test_size", DEFAULT_TEST_SIZE))
    if test_size > 0:
        train_mask = (out[SPLIT_COLUMN] == "train").to_numpy()
        chosen = _sample_groups_by_rows(groups, train_mask, int(train_mask.sum() * test_size), rng)
        out.loc[np.isin(groups, list(chosen)), SPLIT_COLUMN] = "test"

    _guard_non_empty(out, expect_test=test_size > 0, expect_oot=expect_oot)
    return out


def _valid_group_cols(out: pd.DataFrame, group_cols) -> list[str]:
    if not group_cols:
        return []
    cols = group_cols if isinstance(group_cols, list) else [group_cols]
    return [str(col) for col in cols if str(col) in out.columns]


def _group_ids(out: pd.DataFrame, group_cols: list[str]) -> np.ndarray:
    """A group id per row. With group_cols, rows sharing those values form one group
    (so they never split across sets); without, each row is its own group."""
    if group_cols:
        return out.groupby(group_cols, sort=False).ngroup().to_numpy()
    return np.arange(len(out))


def _sample_groups_by_rows(groups: np.ndarray, mask: np.ndarray, target_rows: int, rng) -> set:
    """Pick whole groups (restricted to ``mask``) until ~``target_rows`` rows are
    covered. Deterministic given the seeded ``rng``."""
    idx = np.where(mask)[0]
    if len(idx) == 0 or target_rows <= 0:
        return set()
    masked_groups = groups[idx]
    unique_groups = np.unique(masked_groups)
    rng.shuffle(unique_groups)
    chosen: set = set()
    covered = 0
    for group in unique_groups:
        if covered >= target_rows:
            break
        chosen.add(int(group))
        covered += int((masked_groups == group).sum())
    return chosen


def _guard_non_empty(out: pd.DataFrame, *, expect_test: bool, expect_oot: bool) -> None:
    counts = out[SPLIT_COLUMN].value_counts().to_dict()
    empty = []
    if counts.get("train", 0) == 0:
        empty.append("train")
    if expect_test and counts.get("test", 0) == 0:
        empty.append("test")
    if expect_oot and counts.get("oot", 0) == 0:
        empty.append("oot")
    if empty:
        raise ModelingError(
            f"切分后这些集合为空:{empty}(当前 {counts});请调整切分规则或比例。"
        )


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
    for group_col in split_config.get("group_cols") or []:
        columns.append(str(group_col))
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
