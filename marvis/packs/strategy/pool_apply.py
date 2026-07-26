"""Pure deterministic execution of one authenticated Strategy Pool snapshot."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Any

import pandas as pd
from pandas.testing import assert_frame_equal

from marvis.packs.strategy.apply_projection import (
    DEFAULT_POOL_APPLY_PREFIX,
    deterministic_rule_counts,
    deterministic_string_counts,
    resolve_apply_output_columns,
    serialize_strategy_decisions,
)
from marvis.packs.strategy.dsl import strategy_spec_hash
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.evaluator import evaluate_strategy_frame
from marvis.packs.strategy.pool import (
    compile_strategy_pool,
    validate_strategy_pool,
)

MAX_POOL_APPLY_ROWS = 1_000_000
MAX_POOL_APPLY_SOURCE_COLUMNS = 500


@dataclass(frozen=True)
class StrategyPoolApplyResult:
    """Detached Pool assignment result and deterministic evidence summaries."""

    derived_frame: pd.DataFrame = field(compare=False, repr=False)
    output_columns: Mapping[str, str]
    source_row_count: int
    action_counts: Mapping[str, int]
    rule_counts: Mapping[str, int]
    entry_counts: Mapping[str, int]
    default_count: int
    strategy_spec_hash: str
    design_hash: str
    assignment_hash: str


def apply_strategy_pool(
    original_frame: pd.DataFrame,
    execution_frame: pd.DataFrame,
    pool: Mapping[str, Any],
    *,
    compiled_design: Mapping[str, Any] | None = None,
    output_prefix: str = DEFAULT_POOL_APPLY_PREFIX,
) -> StrategyPoolApplyResult:
    """Apply Pool first-match semantics and project assignments onto source rows."""

    _require_aligned_frames(original_frame, execution_frame)
    current_pool = validate_strategy_pool(pool)
    canonical_design = compile_strategy_pool(current_pool)
    if compiled_design is not None and compiled_design != canonical_design:
        raise StrategyError("compiled design does not match Strategy Pool snapshot")
    design = canonical_design
    spec = design["strategy_spec"]
    evaluation = evaluate_strategy_frame(execution_frame, spec)
    rule_to_entry = {
        str(entry["rule_id"]): str(entry["entry_id"])
        for entry in current_pool["entries"]
    }
    if len(rule_to_entry) != len(current_pool["entries"]):
        raise StrategyError("Strategy Pool rule to entry mapping is not one-to-one")
    matched_entries = pd.Series(
        [
            None if value is None else rule_to_entry.get(str(value))
            for value in evaluation.matched_rule_id.tolist()
        ],
        index=evaluation.matched_rule_id.index,
        dtype="object",
        name="matched_entry_id",
    )
    if bool(
        (
            evaluation.matched_rule_id.notna()
            & matched_entries.isna()
        ).any()
    ):
        raise StrategyError("compiled Strategy rule has no matching Pool entry")

    output = resolve_apply_output_columns(
        original_frame.columns,
        output_prefix=output_prefix,
        default_prefix=DEFAULT_POOL_APPLY_PREFIX,
        include_entry_id=True,
    )
    columns = output.as_dict()
    serialized = serialize_strategy_decisions(
        evaluation.decisions,
        strategy_type=current_pool["strategy_type"],
    )
    derived = original_frame.copy(deep=True)
    derived[columns["action"]] = evaluation.action_type
    derived[columns["value"]] = serialized.values
    derived[columns["value_type"]] = serialized.value_types
    derived[columns["rule_id"]] = evaluation.matched_rule_id
    derived[columns["entry_id"]] = matched_entries
    derived[columns["reason_code"]] = evaluation.reason_code

    assignments = [
        {
            "rule_id": rule_id,
            "entry_id": entry_id,
        }
        for rule_id, entry_id in zip(
            evaluation.matched_rule_id.tolist(),
            matched_entries.tolist(),
            strict=True,
        )
    ]
    return StrategyPoolApplyResult(
        derived_frame=derived,
        output_columns=MappingProxyType(columns),
        source_row_count=len(original_frame),
        action_counts=MappingProxyType(
            deterministic_string_counts(evaluation.action_type)
        ),
        rule_counts=MappingProxyType(
            deterministic_rule_counts(evaluation.matched_rule_id)
        ),
        entry_counts=MappingProxyType(deterministic_rule_counts(matched_entries)),
        default_count=int(evaluation.matched_rule_id.isna().sum()),
        strategy_spec_hash=strategy_spec_hash(spec),
        design_hash=str(design["design_hash"]),
        assignment_hash=_canonical_sha256(assignments),
    )


def _require_aligned_frames(
    original_frame: pd.DataFrame,
    execution_frame: pd.DataFrame,
) -> None:
    if not isinstance(original_frame, pd.DataFrame) or not isinstance(
        execution_frame,
        pd.DataFrame,
    ):
        raise StrategyError("Strategy Pool apply requires DataFrame inputs")
    if len(original_frame) > MAX_POOL_APPLY_ROWS:
        raise StrategyError("Strategy Pool apply row budget exceeded")
    if len(original_frame.columns) > MAX_POOL_APPLY_SOURCE_COLUMNS:
        raise StrategyError(
            "Strategy Pool apply source column budget exceeded"
        )
    expected = pd.RangeIndex(start=0, stop=len(original_frame), step=1)
    if (
        not isinstance(original_frame.index, pd.RangeIndex)
        or not original_frame.index.equals(expected)
        or not isinstance(execution_frame.index, pd.RangeIndex)
        or not execution_frame.index.equals(expected)
    ):
        raise StrategyError(
            "Strategy Pool apply requires exact zero-based row order"
        )
    missing = [
        column
        for column in original_frame.columns
        if column not in execution_frame.columns
    ]
    if missing:
        raise StrategyError(
            "Strategy Pool execution frame is missing source columns"
        )
    try:
        assert_frame_equal(
            execution_frame.loc[:, original_frame.columns],
            original_frame,
            check_dtype=True,
            check_exact=True,
        )
    except AssertionError as exc:
        raise StrategyError(
            "Strategy Pool execution frame changed source values, types, or order"
        ) from exc


def _canonical_sha256(value: object) -> str:
    try:
        canonical = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise StrategyError(
            "Strategy Pool assignments must be canonical JSON"
        ) from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "MAX_POOL_APPLY_ROWS",
    "MAX_POOL_APPLY_SOURCE_COLUMNS",
    "StrategyPoolApplyResult",
    "apply_strategy_pool",
]
