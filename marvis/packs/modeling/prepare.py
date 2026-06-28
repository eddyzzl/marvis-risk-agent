from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from marvis.feature.candidates import candidate_numeric_features
from marvis.packs.modeling.errors import ModelingError


DEFAULT_TEST_SIZE = 0.30
DEFAULT_OOT_SIZE = 0.20
SPLIT_COLUMN = "split"

VALID_SPLIT_ASSIGNMENTS = ("train", "test", "oot")
_RULE_OPS = ("eq", "ne", "in", "lt", "le", "gt", "ge")


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
    if not feature_cols:
        feature_cols = candidate_numeric_features(
            backend,
            dataset_path,
            target_col=target_col,
            split_col=split_col,
        )
    if not feature_cols:
        raise ModelingError("未找到可用候选特征列;请检查拼接结果或指定特征列。")
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

    # 0) Rule set (optional). Any-condition rules (e.g. channel A → train, channel B
    #    before a cutoff → test, channel C → oot) are applied first, in order: the first
    #    matching rule wins and that row is FROZEN — excluded from later rules and from
    #    every random draw below. Pure condition logic, no RNG, so it is deterministic.
    rule_mask = _apply_rules(out, config.get("rules"))
    free = ~rule_mask  # rows still eligible for the legacy random/time split

    # 1) OOT — restricted to rule-free rows so a rule assignment is never overwritten.
    time_col = config.get("oot_by_time")
    oot_size = float(config.get("oot_size", DEFAULT_OOT_SIZE))
    expect_oot = bool(rule_mask.any() and (out.loc[rule_mask, SPLIT_COLUMN] == "oot").any())
    if time_col:
        time_col = str(time_col)
        if time_col not in out.columns:
            raise ModelingError(f"missing columns: {time_col}")
        free_time = out.loc[free, time_col]
        if not free_time.empty:
            cutoff = free_time.quantile(1 - oot_size)
            out.loc[free & (out[time_col] >= cutoff), SPLIT_COLUMN] = "oot"
            expect_oot = True
    elif config.get("random_oot") and oot_size > 0:
        # OOT means out-of-time; only fabricate a random OOT when explicitly asked.
        train_mask = ((out[SPLIT_COLUMN] == "train") & free).to_numpy()
        chosen = _sample_groups_by_rows(groups, train_mask, int(round(train_mask.sum() * oot_size)), rng)
        out.loc[np.isin(groups, list(chosen)) & free, SPLIT_COLUMN] = "oot"
        expect_oot = True

    # 2) test from the non-OOT, rule-free remainder (test_size is a fraction of that
    #    remainder, preserving the original contract), whole groups together.
    test_size = float(config.get("test_size", DEFAULT_TEST_SIZE))
    if test_size > 0:
        train_mask = ((out[SPLIT_COLUMN] == "train") & free).to_numpy()
        chosen = _sample_groups_by_rows(groups, train_mask, int(train_mask.sum() * test_size), rng)
        out.loc[np.isin(groups, list(chosen)) & free, SPLIT_COLUMN] = "test"

    expect_test = bool(test_size > 0 or (rule_mask.any() and (out.loc[rule_mask, SPLIT_COLUMN] == "test").any()))
    _guard_non_empty(out, expect_test=expect_test, expect_oot=expect_oot)
    return out


def _apply_rules(out: pd.DataFrame, rules) -> np.ndarray:
    """Apply an ordered rule set, returning a boolean mask of rule-assigned rows.

    Each rule is ``{"when": [{"col", "op", "val"}, ...], "assign": "train"|"test"|"oot"}``
    where the ``when`` conditions are AND-combined. Rules apply in order and the first
    match wins — a row claimed by an earlier rule is never re-evaluated. Returns an
    all-False mask when no rules are given (legacy behaviour, untouched).
    """
    assigned = np.zeros(len(out), dtype=bool)
    if not rules:
        return assigned
    if not isinstance(rules, list):
        raise ModelingError("split rules must be a list of rule objects")
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise ModelingError(f"split rule #{index} must be an object")
        assign = rule.get("assign")
        if assign not in VALID_SPLIT_ASSIGNMENTS:
            raise ModelingError(
                f"split rule #{index} has invalid assign {assign!r}; "
                f"expected one of {VALID_SPLIT_ASSIGNMENTS}"
            )
        match = _rule_condition_mask(out, rule.get("when"), index)
        fresh = match & ~assigned
        if fresh.any():
            out.loc[fresh, SPLIT_COLUMN] = assign
            assigned |= fresh
    return assigned


def _rule_condition_mask(out: pd.DataFrame, conditions, rule_index: int) -> np.ndarray:
    """AND-combine the conditions of one rule into a boolean row mask."""
    if not conditions or not isinstance(conditions, list):
        raise ModelingError(f"split rule #{rule_index} must have a non-empty 'when' list")
    mask = np.ones(len(out), dtype=bool)
    for condition in conditions:
        if not isinstance(condition, dict):
            raise ModelingError(f"split rule #{rule_index} has a non-object condition")
        col = condition.get("col")
        op = condition.get("op")
        val = condition.get("val")
        if col is None or str(col) not in out.columns:
            raise ModelingError(f"missing columns: {col}")
        if op not in _RULE_OPS:
            raise ModelingError(
                f"split rule #{rule_index} uses unknown op {op!r}; expected one of {_RULE_OPS}"
            )
        mask &= _condition_values(out[str(col)], op, val)
    return mask


def _condition_values(series: pd.Series, op: str, val) -> np.ndarray:
    if op == "eq":
        return (series == val).to_numpy()
    if op == "ne":
        return (series != val).to_numpy()
    if op == "in":
        if not isinstance(val, (list, tuple, set)):
            raise ModelingError("op 'in' requires a list value")
        return series.isin(list(val)).to_numpy()
    if op == "lt":
        return (series < val).to_numpy()
    if op == "le":
        return (series <= val).to_numpy()
    if op == "gt":
        return (series > val).to_numpy()
    if op == "ge":
        return (series >= val).to_numpy()
    raise ModelingError(f"unknown op {op!r}")


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
    for rule in split_config.get("rules") or []:
        if isinstance(rule, dict):
            for condition in rule.get("when") or []:
                if isinstance(condition, dict) and condition.get("col") is not None:
                    columns.append(str(condition["col"]))
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
