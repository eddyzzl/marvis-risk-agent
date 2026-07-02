from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


# Columns that should not be treated as model/feature-analysis candidates even
# when they are numeric. Keep this shared between setup and runtime tools so a
# JOIN-composed workflow infers the same feature family as the single-table path.
META_TOKENS = re.compile(
    r"(^|_)(id|uid|uuid|idcard|cust|user|order|loan|apply|cert|phone|mobile|name|"
    r"date|time|month|day|dt|ts|created|updated|weight|sample_weight)(_|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExcludedCategorical:
    """A non-numeric (string/object/category) column dropped from candidates,
    with its cardinality so the caller can surface *why* (PREP-3): distinguish a
    near-constant column from a high-cardinality one that needs WOE/target
    encoding rather than a silent drop."""

    column: str
    cardinality: int


def candidate_numeric_features(
    backend,
    path: Path,
    *,
    target_col: str,
    split_col: str | None = None,
    sample_rows: int = 4000,
) -> list[str]:
    """Infer numeric candidate features from the actual dataset schema/sample."""
    target = str(target_col)
    split = str(split_col) if split_col else ""
    probe = backend.sample_rows(Path(path), int(sample_rows), seed=0)
    excluded = {target, split}
    return [
        str(column)
        for column in probe.select_dtypes("number").columns
        if str(column) not in excluded and not META_TOKENS.search(str(column))
    ]


def excluded_categorical_columns(
    backend,
    path: Path,
    *,
    target_col: str,
    split_col: str | None = None,
    sample_rows: int = 4000,
) -> list[ExcludedCategorical]:
    """Companion to :func:`candidate_numeric_features` (PREP-3/FS-3): the string/
    object/category columns that function silently drops from candidates, each
    with its sampled cardinality. Applies the same target/split/META_TOKENS
    exclusions so the two functions never disagree on what counts as "in scope".
    Used to surface an explicit "N categorical columns not modeled" notice
    instead of a silent drop."""
    target = str(target_col)
    split = str(split_col) if split_col else ""
    probe = backend.sample_rows(Path(path), int(sample_rows), seed=0)
    excluded = {target, split}
    numeric_columns = set(probe.select_dtypes("number").columns.map(str))
    out: list[ExcludedCategorical] = []
    for column in probe.columns:
        name = str(column)
        if name in excluded or name in numeric_columns or META_TOKENS.search(name):
            continue
        cardinality = int(probe[column].nunique(dropna=True))
        out.append(ExcludedCategorical(column=name, cardinality=cardinality))
    return out


__all__ = [
    "META_TOKENS",
    "ExcludedCategorical",
    "candidate_numeric_features",
    "excluded_categorical_columns",
]
