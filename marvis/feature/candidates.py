from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# Only structurally proven technical fields are hard-excluded. Broad business
# tokens such as ``loan``/``apply``/``mobile``/``cust`` are intentionally absent:
# ``loan_amount`` and ``cust_income`` are ordinary predictors, not metadata.
META_TOKENS = re.compile(
    r"^(?:id|uid|uuid|idcard|id_card|id_card_no|cert_no|phone|mobile|name|"
    r"date|time|month|day|dt|ts|created_at|updated_at|weight|sample_weight)$"
    r"|(?:^|_)(?:id|uid|uuid)$"
    r"|(?:^|_)(?:date|time|month|day|dt|ts)$"
    r"|(?:^|_)(?:sample_)?weight$",
    re.IGNORECASE,
)

# Real-world join keys are often compact legacy names with no underscore, e.g.
# ``applydt``/``usedate``. The token regex above deliberately uses boundaries to
# avoid excluding ordinary words; this narrower suffix rule catches only clear
# date/time technical fields.
COMPACT_DATE_META = re.compile(
    r"^(?:apply|use|event|create|update|start|end|biz)?(?:date|dt|time|timestamp|ts)$",
    re.IGNORECASE,
)


def is_meta_column(name: str) -> bool:
    value = str(name)
    return bool(META_TOKENS.search(value) or COMPACT_DATE_META.search(value))

# Name hints for integer-coded nominal columns (PREP-5): zip/postal/industry/region
# codes carry no ordinal distance semantics even though they parse as numbers, so
# equal-width/equal-frequency binning would group unrelated codes together. These
# tokens are checked *in addition to* the low-cardinality + all-integer signal below
# -- neither alone is a reliable enough trigger.
_CODE_NAME_TOKENS = re.compile(
    r"(^|_)(code|zip|postal|postcode|zipcode|type|category|industry|region|"
    r"area|district|county|province|city|channel|source|level|grade|rank|"
    r"编码|代码|类型|行业|区号|邮编|地区|渠道|等级)(_|$)",
    re.IGNORECASE,
)

# Below this many distinct values, an integer column with a code-like name is very
# likely a nominal code rather than a continuous measure.
SUSPECTED_CATEGORICAL_MAX_CARDINALITY = 20


@dataclass(frozen=True)
class SuspectedCategorical:
    """A numeric column that is likely a nominal code (PREP-5), e.g. a zip/industry
    code: low cardinality, all-integer values, and a code-like column name. Purely
    informational -- candidate inference still treats it as numeric (no automatic
    behavior change); this is surfaced so the user can route it through categorical
    WOE / rare-category grouping instead of continuous binning if they choose to."""

    column: str
    cardinality: int


@dataclass(frozen=True)
class ExcludedCategorical:
    """A non-numeric (string/object/category) column dropped from candidates,
    with its cardinality so the caller can surface *why* (PREP-3): distinguish a
    near-constant column from a high-cardinality one that needs WOE/target
    encoding rather than a silent drop."""

    column: str
    cardinality: int


@dataclass(frozen=True)
class ExcludedNumeric:
    """A numeric technical column excluded from automatic candidates."""

    column: str
    reason: str


def candidate_numeric_features(
    backend,
    path: Path,
    *,
    target_col: str,
    split_col: str | None = None,
    sample_rows: int = 4000,
    include_columns: tuple[str, ...] | list[str] = (),
) -> list[str]:
    """Infer numeric candidate features from the actual dataset schema/sample."""
    target = str(target_col)
    split = str(split_col) if split_col else ""
    probe = backend.sample_rows(Path(path), int(sample_rows), seed=0)
    excluded = {target, split}
    explicit = {str(item) for item in include_columns}
    return [
        str(column)
        for column in probe.select_dtypes("number").columns
        if str(column) not in excluded
        and (str(column) in explicit or not is_meta_column(str(column)))
    ]


def excluded_numeric_columns(
    backend,
    path: Path,
    *,
    target_col: str,
    split_col: str | None = None,
    sample_rows: int = 4000,
    include_columns: tuple[str, ...] | list[str] = (),
) -> list[ExcludedNumeric]:
    """List every automatically excluded numeric column and its reason."""

    target = str(target_col)
    split = str(split_col) if split_col else ""
    explicit = {str(item) for item in include_columns}
    probe = backend.sample_rows(Path(path), int(sample_rows), seed=0)
    out: list[ExcludedNumeric] = []
    for column in probe.select_dtypes("number").columns:
        name = str(column)
        if name in {target, split} or name in explicit or not is_meta_column(name):
            continue
        out.append(ExcludedNumeric(name, _meta_exclusion_reason(name)))
    return out


def _meta_exclusion_reason(name: str) -> str:
    if COMPACT_DATE_META.fullmatch(str(name)):
        return "结构化日期/时间字段"
    if re.search(r"(?:^|_)(?:date|time|month|day|dt|ts)$", str(name), re.IGNORECASE):
        return "结构化日期/时间字段"
    if re.search(r"(?:^|_)(?:sample_)?weight$", str(name), re.IGNORECASE):
        return "样本权重字段"
    return "结构化标识字段"


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
        if name in excluded or name in numeric_columns or is_meta_column(name):
            continue
        cardinality = int(probe[column].nunique(dropna=True))
        out.append(ExcludedCategorical(column=name, cardinality=cardinality))
    return out


def suspected_categorical_columns(
    backend,
    path: Path,
    *,
    target_col: str,
    split_col: str | None = None,
    sample_rows: int = 4000,
    max_cardinality: int = SUSPECTED_CATEGORICAL_MAX_CARDINALITY,
) -> list[SuspectedCategorical]:
    """Numeric candidate columns that look like nominal codes rather than continuous
    measures (PREP-5), e.g. a 4-digit industry code or a zip code: low cardinality,
    every finite value is integer-valued, and the column name carries no ordinal-
    distance semantics (matches a code-like token such as ``code``/``zip``/``type``).
    Purely informational -- does not change what :func:`candidate_numeric_features`
    returns; callers surface this as a screen-gate hint so the user can route the
    column through categorical WOE / rare-category grouping instead of continuous
    binning if they choose to."""
    target = str(target_col)
    split = str(split_col) if split_col else ""
    probe = backend.sample_rows(Path(path), int(sample_rows), seed=0)
    excluded = {target, split}
    out: list[SuspectedCategorical] = []
    for column in probe.select_dtypes("number").columns:
        name = str(column)
        if name in excluded or is_meta_column(name) or not _CODE_NAME_TOKENS.search(name):
            continue
        series = pd.to_numeric(probe[column], errors="coerce").dropna()
        if series.empty:
            continue
        cardinality = int(series.nunique())
        if cardinality > max_cardinality:
            continue
        values = series.to_numpy(dtype=float)
        if not np.all(np.equal(np.mod(values, 1), 0)):
            continue
        out.append(SuspectedCategorical(column=name, cardinality=cardinality))
    return out


__all__ = [
    "META_TOKENS",
    "COMPACT_DATE_META",
    "SUSPECTED_CATEGORICAL_MAX_CARDINALITY",
    "ExcludedCategorical",
    "ExcludedNumeric",
    "SuspectedCategorical",
    "candidate_numeric_features",
    "is_meta_column",
    "excluded_categorical_columns",
    "excluded_numeric_columns",
    "suspected_categorical_columns",
]
