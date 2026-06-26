"""Deterministic sample setup detection shared by the feature/modeling drivers.

Reads only a row sample (for dtypes/binary checks) plus the key columns in full
(for counts/bad-rate) — never the whole frame — and proposes the target column,
the train/test/oot split column + values, and the numeric candidate features
(ids / time / weight columns excluded). Extracted from the original conversational
modeling prototype so both feature_analysis and modeling can reuse it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Columns that are never modeling features regardless of dtype (ids / time / weights).
_META_TOKENS = re.compile(
    r"(^|_)(id|uid|uuid|idcard|cust|user|order|loan|apply|cert|phone|mobile|name|"
    r"date|time|month|day|dt|ts|created|updated|weight|sample_weight)(_|$)",
    re.IGNORECASE,
)
# Preferred binary-target name tokens, most-specific first.
_TARGET_PRIORITY = (
    "long_y", "fission_y", "y", "label", "target",
    "is_bad", "bad_flag", "flag_bad", "bad", "default", "dpd", "fpd",
)
# Recognised split-membership values (lower-cased).
_SPLIT_TRAIN = {"train", "training", "dev", "develop", "development", "build"}
_SPLIT_TEST = {"test", "testing", "valid", "validation", "val", "holdout"}
_SPLIT_OOT = {"oot", "ootest", "out_of_time", "oos", "time_oot"}


@dataclass
class SetupProposal:
    target_col: str
    split_col: Optional[str]
    split_values: dict[str, str]
    candidates: list[str]
    counts: dict[str, int]
    bad_rate: Optional[float]
    notes: list[str]


def _is_binary(series) -> bool:
    vals = set(series.dropna().unique().tolist())
    return vals.issubset({0, 1, 0.0, 1.0, True, False}) and len(vals) == 2


def detect_setup(
    backend,
    path: Path,
    *,
    configured_target: str = "",
    configured_split: str = "",
    sample_rows: int = 4000,
) -> SetupProposal:
    """Propose target column, split column/values and numeric candidate features."""
    columns = backend.column_names(path)
    # Random sample (NOT a head slice) — samples are often ordered by split, so a
    # head read would miss whole splits and skew dtype/binary detection.
    probe = backend.sample_rows(path, sample_rows, seed=0)
    notes: list[str] = []

    # -- target ---------------------------------------------------------------
    target = ""
    if configured_target and configured_target in probe.columns and _is_binary(probe[configured_target]):
        target = configured_target
    if not target:
        binary_cols = [c for c in probe.columns if _is_binary(probe[c])]
        ranked = sorted(
            binary_cols,
            key=lambda c: next((i for i, tok in enumerate(_TARGET_PRIORITY) if tok in c.lower()), len(_TARGET_PRIORITY)),
        )
        prioritised = [c for c in ranked if any(tok in c.lower() for tok in _TARGET_PRIORITY)]
        target = (prioritised or ranked or [""])[0]
    if not target:
        notes.append("未能自动识别 0/1 目标列，请直接告诉我目标列名。")

    # -- split ----------------------------------------------------------------
    split_col = ""
    split_values: dict[str, str] = {}
    by_name = [configured_split] if configured_split in columns else []
    by_name += [c for c in columns if _looks_like_split_name(c)]
    obj_cols = [c for c in probe.columns if probe[c].dtype == object and probe[c].nunique(dropna=True) <= 8]
    for cand in dict.fromkeys(c for c in (by_name + obj_cols) if c):
        col = backend.read_frame(path, columns=[cand])[cand]
        mapping = _classify_split_values(col)
        if "train" in mapping and ("test" in mapping or "oot" in mapping):
            split_col, split_values = cand, mapping
            break
    if not split_col:
        notes.append("未能自动识别 train/test/oot 切分列；可指定切分列，或我按时间字段为你切分。")

    # -- candidate features (numeric, minus target/split/meta) ----------------
    numeric = [c for c in probe.select_dtypes("number").columns if c not in {target, split_col}]
    candidates = [c for c in numeric if not _META_TOKENS.search(c)]

    # -- counts / bad-rate (read only key columns in full) --------------------
    counts: dict[str, int] = {}
    bad_rate: Optional[float] = None
    key_cols = [c for c in {target, split_col} if c]
    if key_cols:
        keys = backend.read_frame(path, columns=key_cols)
        if target and target in keys:
            bad_rate = float(keys[target].mean())
        if split_col and split_col in keys:
            counts = {
                role: int((keys[split_col] == val).sum())
                for role, val in split_values.items()
            }
    return SetupProposal(target, split_col or None, split_values, candidates, counts, bad_rate, notes)


def _looks_like_split_name(name: str) -> bool:
    low = name.lower()
    return any(tok in low for tok in ("split", "flag", "set", "fold", "sample_type", "model_flag", "new_flag"))


def _classify_split_values(series) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw in series.dropna().unique().tolist():
        low = str(raw).strip().lower()
        if low in _SPLIT_TRAIN and "train" not in mapping:
            mapping["train"] = raw
        elif low in _SPLIT_TEST and "test" not in mapping:
            mapping["test"] = raw
        elif low in _SPLIT_OOT and "oot" not in mapping:
            mapping["oot"] = raw
    return mapping


__all__ = ["detect_setup", "SetupProposal"]
