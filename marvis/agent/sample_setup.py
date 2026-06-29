"""Deterministic sample setup detection shared by the feature/modeling drivers.

Reads only a row sample (for dtypes/binary checks) plus the key columns in full
(for counts/bad-rate) — never the whole frame — and proposes the target column,
the train/test/oot split column + values, and the numeric candidate features
(ids / time / weight columns excluded). Extracted from the original conversational
modeling prototype so both feature_analysis and modeling can reuse it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from marvis.feature.candidates import META_TOKENS, candidate_numeric_features
# Preferred binary-target name tokens, most-specific first.
_TARGET_PRIORITY = (
    "long_y", "fission_y", "y", "label", "target",
    "is_bad", "bad_flag", "flag_bad", "bad", "default", "dpd", "fpd",
)
# Recognised split-membership values (lower-cased).
_SPLIT_TRAIN = {"train", "training", "dev", "develop", "development", "build"}
_SPLIT_TEST = {"test", "testing", "valid", "validation", "val", "holdout"}
_SPLIT_OOT = {"oot", "ootest", "out_of_time", "oos", "time_oot"}
# Preferred continuous-target name tokens, most-specific first (case-insensitive).
_CONTINUOUS_TARGET_TOKENS = (
    "income", "amount", "amt", "balance", "limit",
    "loan_amount", "gmv", "revenue", "price", "salary",
)
# Preferred multiclass-target name tokens (case-insensitive).
_MULTICLASS_TARGET_TOKENS = (
    "risk_grade", "grade", "rating", "class", "level", "等级", "评级", "类别",
)
# A multiclass target must have between this many distinct classes (inclusive).
_MULTICLASS_MIN_CLASSES = 3
_MULTICLASS_MAX_CLASSES = 20


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
    target_type: str = "binary",
) -> SetupProposal:
    """Propose target column, split column/values and numeric candidate features.

    ``target_type`` defaults to ``"binary"`` (the existing 0/1 detection — the
    feature_analysis flow never passes it, so its behaviour is unchanged). When
    ``"continuous"`` the target column is resolved as a numeric column (for a
    regression task) and ``bad_rate`` is left ``None``.
    """
    columns = backend.column_names(path)
    # Random sample (NOT a head slice) — samples are often ordered by split, so a
    # head read would miss whole splits and skew dtype/binary detection.
    probe = backend.sample_rows(path, sample_rows, seed=0)
    notes: list[str] = []
    continuous = target_type == "continuous"
    multiclass = target_type == "multiclass"

    # -- target ---------------------------------------------------------------
    target = ""
    if continuous:
        target = _detect_continuous_target(probe, configured_target)
        if not target:
            notes.append("回归任务请指定连续型目标列。")
    elif multiclass:
        target = _detect_multiclass_target(probe, configured_target)
        if not target:
            notes.append("多分类任务请指定 3-20 类的目标列。")
    else:
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
    candidates = candidate_numeric_features(
        backend,
        path,
        target_col=target,
        split_col=split_col,
        sample_rows=sample_rows,
    )

    # -- counts / bad-rate (read only key columns in full) --------------------
    counts: dict[str, int] = {}
    bad_rate: Optional[float] = None
    key_cols = [c for c in {target, split_col} if c]
    if key_cols:
        keys = backend.read_frame(path, columns=key_cols)
        # bad_rate is a binary-only notion (mean of a 0/1 label); regression and
        # multiclass targets have no bad_rate, so leave it None for those tasks.
        if target and target in keys and not continuous and not multiclass:
            bad_rate = float(keys[target].mean())
        if split_col and split_col in keys:
            counts = {
                role: int((keys[split_col] == val).sum())
                for role, val in split_values.items()
            }
    return SetupProposal(target, split_col or None, split_values, candidates, counts, bad_rate, notes)


def _detect_continuous_target(probe, configured_target: str) -> str:
    """Resolve the continuous (regression) target column from a row sample.

    Prefer ``configured_target`` when it is present and numeric; otherwise pick the
    first numeric column whose name matches a known continuous-target token (income,
    amount, …). Returns "" when no numeric candidate is found (caller adds a note)."""
    numeric_cols = list(probe.select_dtypes("number").columns)
    if (
        configured_target
        and configured_target in probe.columns
        and configured_target in numeric_cols
    ):
        return configured_target
    for token in _CONTINUOUS_TARGET_TOKENS:
        for col in numeric_cols:
            name = str(col)
            if _looks_like_split_name(name) or META_TOKENS.search(name):
                continue
            if token in name.lower():
                return name
    return ""


def _detect_multiclass_target(probe, configured_target: str) -> str:
    """Resolve the multiclass (3-20 class) target column from a row sample.

    Prefer ``configured_target`` when it has a distinct-class count in [3, 20]. Else
    pick the first column whose name matches a known grade/rating token and whose
    distinct-class count is in [3, 20]. Returns "" when nothing qualifies (caller adds
    a note). The split column is never a candidate."""
    if (
        configured_target
        and configured_target in probe.columns
        and _class_count_in_range(probe[configured_target])
    ):
        return configured_target
    for col in probe.columns:
        low = str(col).lower()
        name = str(col)
        if not any(tok in low or tok in name for tok in _MULTICLASS_TARGET_TOKENS):
            continue
        if _looks_like_split_name(name):
            continue
        if _class_count_in_range(probe[col]):
            return name
    return ""


def _class_count_in_range(series) -> bool:
    distinct = int(series.dropna().nunique())
    return _MULTICLASS_MIN_CLASSES <= distinct <= _MULTICLASS_MAX_CLASSES


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
