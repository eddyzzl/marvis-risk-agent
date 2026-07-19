from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import pandas as pd

from marvis.packs.strategy.dsl import (
    STRATEGY_DSL_SCHEMA_VERSION,
    StrategySpec,
    parse_strategy_spec,
    strategy_spec_hash,
)
from marvis.packs.strategy.economics import (
    NumericInput,
    limit_metrics,
    pricing_metrics,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.evaluator import evaluate_strategy_frame
from marvis.packs.strategy.profit import ProfitParams


STRATEGY_BACKTEST_SCHEMA_VERSION = "strategy.backtest.v2"
_SUPPORTED_STRATEGY_TYPES = frozenset(
    {"approval", "reject", "limit", "pricing", "segmentation"}
)
_ACTION_ORDER = ("approve", "reject", "review")
_ACTION_TYPE_TO_DECISION = {
    "approval": "approve",
    "reject": "reject",
    "review": "review",
}


def _json_value(value: Any, *, path: str) -> Any:
    """Return a detached, deterministic JSON-native value."""

    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise StrategyError(f"{path} must contain only finite JSON numbers")
        return normalized
    if isinstance(value, Mapping):
        keys = list(value)
        if any(not isinstance(key, str) for key in keys):
            raise StrategyError(f"{path} keys must be strings")
        normalized: dict[str, Any] = {}
        for key in sorted(keys):
            normalized[key] = _json_value(value[key], path=f"{path}.{key}")
        return normalized
    if isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    ):
        return [
            _json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise StrategyError(f"{path} must be JSON serializable")


def _row_tuple(value: object, *, name: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(
        value, str | bytes | bytearray
    ):
        raise StrategyError(f"{name} must be a list")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise StrategyError(f"{name}[{index}] must be an object")
        normalized = _json_value(row, path=f"{name}[{index}]")
        rows.append(normalized)
    return tuple(rows)


@dataclass(frozen=True)
class ApprovalProfitInputs:
    """Explicit columns and parameters for approved-population profit evidence."""

    params: ProfitParams
    ead_col: str
    pd_col: str

    def __post_init__(self) -> None:
        if not isinstance(self.params, ProfitParams):
            raise StrategyError("approval profit params must be ProfitParams")
        for name, value in (("ead_col", self.ead_col), ("pd_col", self.pd_col)):
            if not isinstance(value, str) or not value.strip():
                raise StrategyError(
                    f"approval profit {name} must be a non-empty string"
                )
            object.__setattr__(self, name, value.strip())


@dataclass(frozen=True)
class StrategyBacktestResult:
    """Versioned, repository-safe envelope for typed strategy backtests."""

    strategy_id: str
    strategy_type: str
    population_count: int
    labeled_count: int
    label_coverage: float
    metrics: Mapping[str, Any]
    breakdown: tuple[Mapping[str, Any], ...]
    transitions: tuple[Mapping[str, Any], ...]
    economics: Mapping[str, Any]
    warnings: tuple[str, ...]
    normalized_input: Mapping[str, Any]
    schema_version: str = STRATEGY_BACKTEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STRATEGY_BACKTEST_SCHEMA_VERSION:
            raise StrategyError(
                f"unsupported backtest schema_version: {self.schema_version}"
            )
        if not isinstance(self.strategy_id, str) or not self.strategy_id.strip():
            raise StrategyError("strategy_id must be a non-empty string")
        if self.strategy_type not in _SUPPORTED_STRATEGY_TYPES:
            raise StrategyError(
                f"typed backtest does not support strategy type {self.strategy_type}"
            )
        _assert_count(self.population_count, name="population_count")
        _assert_count(self.labeled_count, name="labeled_count")
        if self.labeled_count > self.population_count:
            raise StrategyError("labeled_count cannot exceed population_count")
        if not isinstance(self.label_coverage, Real) or isinstance(
            self.label_coverage, bool
        ):
            raise StrategyError("label_coverage must be a number")
        coverage = float(self.label_coverage)
        if not math.isfinite(coverage) or not 0 <= coverage <= 1:
            raise StrategyError("label_coverage must be between 0 and 1")
        expected_coverage = _ratio(self.labeled_count, self.population_count)
        if not math.isclose(coverage, expected_coverage, rel_tol=0.0, abs_tol=1e-12):
            raise StrategyError("label_coverage does not match the row counts")
        if not isinstance(self.metrics, Mapping):
            raise StrategyError("metrics must be an object")
        if not isinstance(self.economics, Mapping):
            raise StrategyError("economics must be an object")
        if not isinstance(self.normalized_input, Mapping):
            raise StrategyError("normalized_input must be an object")
        if not isinstance(self.warnings, Sequence) or isinstance(
            self.warnings, str | bytes | bytearray
        ):
            raise StrategyError("warnings must be a list")
        warnings = tuple(self.warnings)
        if any(not isinstance(warning, str) for warning in warnings):
            raise StrategyError("warnings must contain strings")

        object.__setattr__(self, "strategy_id", self.strategy_id.strip())
        object.__setattr__(self, "label_coverage", coverage)
        object.__setattr__(self, "metrics", _json_value(self.metrics, path="metrics"))
        object.__setattr__(
            self,
            "breakdown",
            _row_tuple(self.breakdown, name="breakdown"),
        )
        object.__setattr__(
            self,
            "transitions",
            _row_tuple(self.transitions, name="transitions"),
        )
        object.__setattr__(
            self,
            "economics",
            _json_value(self.economics, path="economics"),
        )
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(
            self,
            "normalized_input",
            _json_value(self.normalized_input, path="normalized_input"),
        )
        _validate_backtest_semantics(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "strategy_id": self.strategy_id,
            "strategy_type": self.strategy_type,
            "population_count": self.population_count,
            "labeled_count": self.labeled_count,
            "label_coverage": self.label_coverage,
            "metrics": _json_value(self.metrics, path="metrics"),
            "breakdown": [
                _json_value(row, path="breakdown") for row in self.breakdown
            ],
            "transitions": [
                _json_value(row, path="transitions") for row in self.transitions
            ],
            "economics": _json_value(self.economics, path="economics"),
            "warnings": list(self.warnings),
            "normalized_input": _json_value(
                self.normalized_input, path="normalized_input"
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StrategyBacktestResult:
        if not isinstance(payload, Mapping):
            raise StrategyError("backtest result must be an object")
        required = {
            "schema_version",
            "strategy_id",
            "strategy_type",
            "population_count",
            "labeled_count",
            "label_coverage",
            "metrics",
            "breakdown",
            "transitions",
            "economics",
            "warnings",
            "normalized_input",
        }
        missing = sorted(required - set(payload))
        unexpected = sorted(set(payload) - required)
        if missing:
            raise StrategyError(
                f"backtest result is missing fields: {', '.join(missing)}"
            )
        if unexpected:
            raise StrategyError(
                f"backtest result has unsupported fields: {', '.join(unexpected)}"
            )
        return cls(
            schema_version=payload["schema_version"],
            strategy_id=payload["strategy_id"],
            strategy_type=payload["strategy_type"],
            population_count=payload["population_count"],
            labeled_count=payload["labeled_count"],
            label_coverage=payload["label_coverage"],
            metrics=payload["metrics"],
            breakdown=_row_tuple(payload["breakdown"], name="breakdown"),
            transitions=_row_tuple(payload["transitions"], name="transitions"),
            economics=payload["economics"],
            warnings=payload["warnings"],
            normalized_input=payload["normalized_input"],
        )


_ACTION_METRIC_KEYS = {
    "overall_bad_count",
    "overall_bad_rate",
    *(
        f"{action}_{suffix}"
        for action in _ACTION_ORDER
        for suffix in ("count", "rate", "labeled_count", "bad_count", "bad_rate")
    ),
}
_ACTION_BREAKDOWN_KEYS = {
    "action",
    "count",
    "rate",
    "labeled_count",
    "bad_count",
    "bad_rate",
}
_ACTION_TRANSITION_KEYS = {
    "from_action",
    "to_action",
    "count",
    "rate",
    "population_share",
    "labeled_count",
    "bad_count",
    "bad_rate",
}
_LAYER_KEYS = {
    "count",
    "share",
    "labeled_count",
    "bad_count",
    "bad_rate",
}
_PRICING_ECONOMICS_KEYS = {
    "total_ead",
    "ead_weighted_rate",
    "revenue",
    "expected_loss",
    "funding_cost",
    "operating_cost",
    "profit",
    "roa",
    "baseline_profit",
    "profit_delta_vs_baseline",
    "by_row",
}
_ECONOMIC_ROW_KEYS = {
    "position",
    "revenue",
    "expected_loss",
    "funding_cost",
    "operating_cost",
    "profit",
    "roa",
    "profit_delta_vs_baseline",
}
_NORMALIZED_INPUT_KEYS = {
    "strategy_schema_version",
    "strategy_effect_hash",
    "baseline_effect_hash",
    "target_col",
    "target_encoding",
    "missing_label_policy",
    "population_rate_denominator",
    "bad_rate_denominator",
    "economics_input_kinds",
    "economics_input_evidence",
    "approval_profit_input",
}
_LIMIT_ECONOMICS_INPUT_KEYS = {"pd", "lgd", "utilization"}
_PRICING_ECONOMICS_INPUT_KEYS = {
    "ead",
    "pd",
    "lgd",
    "funding_rate",
    "term_months",
    "operating_cost_per_loan",
}


def _validate_backtest_semantics(result: StrategyBacktestResult) -> None:
    _validate_normalized_input(result)
    if result.strategy_type in {"approval", "reject"}:
        _validate_action_result(result)
    elif result.strategy_type == "limit":
        _validate_limit_result(result)
    elif result.strategy_type == "pricing":
        _validate_pricing_result(result)
    else:
        _validate_segmentation_result(result)


def _validate_normalized_input(result: StrategyBacktestResult) -> None:
    normalized = result.normalized_input
    _require_exact_keys(normalized, _NORMALIZED_INPUT_KEYS, name="normalized_input")
    if normalized["strategy_schema_version"] != STRATEGY_DSL_SCHEMA_VERSION:
        raise StrategyError("normalized_input.strategy_schema_version is unsupported")
    _require_sha256(
        normalized["strategy_effect_hash"],
        name="normalized_input.strategy_effect_hash",
    )
    baseline_hash = normalized["baseline_effect_hash"]
    if baseline_hash is not None:
        _require_sha256(
            baseline_hash,
            name="normalized_input.baseline_effect_hash",
        )
    target_col = normalized["target_col"]
    if not isinstance(target_col, str) or not target_col.strip():
        raise StrategyError("normalized_input.target_col must be a non-empty string")
    if normalized["target_encoding"] != {"bad": 1, "good": 0}:
        raise StrategyError("normalized_input.target_encoding is unsupported")
    expected_literals = {
        "missing_label_policy": "exclude_from_label_metrics",
        "population_rate_denominator": "population_count",
        "bad_rate_denominator": "labeled_group_count",
    }
    for key, expected in expected_literals.items():
        if normalized[key] != expected:
            raise StrategyError(f"normalized_input.{key} is unsupported")

    kinds = normalized["economics_input_kinds"]
    evidence = normalized["economics_input_evidence"]
    if not isinstance(kinds, Mapping):
        raise StrategyError("normalized_input.economics_input_kinds must be an object")
    if not isinstance(evidence, Mapping):
        raise StrategyError(
            "normalized_input.economics_input_evidence must be an object"
        )
    if result.strategy_type == "limit":
        allowed_economics_keys = _LIMIT_ECONOMICS_INPUT_KEYS
    elif result.strategy_type == "pricing":
        allowed_economics_keys = _PRICING_ECONOMICS_INPUT_KEYS
    else:
        allowed_economics_keys = set()
    actual_economics_keys = set(kinds)
    if actual_economics_keys not in (set(), allowed_economics_keys):
        raise StrategyError(
            "normalized_input economics inputs must be empty or a complete bundle"
        )
    _require_exact_keys(
        evidence,
        actual_economics_keys,
        name="normalized_input.economics_input_evidence",
    )
    for key in sorted(actual_economics_keys):
        kind = kinds[key]
        if kind not in {"scalar", "series"}:
            raise StrategyError(
                f"normalized_input.economics_input_kinds.{key} is unsupported"
            )
        item = evidence[key]
        if not isinstance(item, Mapping):
            raise StrategyError(
                f"normalized_input.economics_input_evidence.{key} must be an object"
            )
        if kind == "scalar":
            _require_exact_keys(
                item,
                {"kind", "value"},
                name=f"normalized_input.economics_input_evidence.{key}",
            )
            if item["kind"] != "scalar":
                raise StrategyError("economics scalar evidence kind is inconsistent")
            _number(
                item,
                "value",
                name=f"normalized_input.economics_input_evidence.{key}",
            )
        else:
            _require_exact_keys(
                item,
                {"kind", "name", "row_count", "content_hash"},
                name=f"normalized_input.economics_input_evidence.{key}",
            )
            if item["kind"] != "series":
                raise StrategyError("economics series evidence kind is inconsistent")
            series_name = item["name"]
            if series_name is not None and not isinstance(series_name, str):
                raise StrategyError("economics series evidence name must be string or null")
            _require_equal(
                _integer(
                    item,
                    "row_count",
                    name=f"normalized_input.economics_input_evidence.{key}",
                ),
                result.population_count,
                name=(
                    f"normalized_input.economics_input_evidence.{key}.row_count"
                ),
            )
            _require_sha256(
                item["content_hash"],
                name=(
                    f"normalized_input.economics_input_evidence.{key}.content_hash"
                ),
            )

    has_calculated_economics = bool(actual_economics_keys)
    if result.strategy_type in {"limit", "pricing"}:
        if has_calculated_economics != bool(result.economics):
            raise StrategyError(
                "normalized_input economics evidence does not match economics output"
            )
    approval_input = normalized["approval_profit_input"]
    if approval_input is not None:
        if result.strategy_type not in {"approval", "reject"}:
            raise StrategyError(
                "normalized_input.approval_profit_input requires approval/reject"
            )
        _validate_approval_profit_input(approval_input)
        if set(result.economics) == {"expected_profit", "profit_note"}:
            raise StrategyError(
                "approval profit input requires calculated economics evidence"
            )
    elif result.strategy_type in {"approval", "reject"} and set(
        result.economics
    ) - {"expected_profit", "profit_note"}:
        raise StrategyError(
            "calculated approval economics requires approval profit input"
        )

    baseline_present = baseline_hash is not None
    if result.strategy_type in {"approval", "reject", "segmentation"}:
        transition_baseline = bool(result.transitions)
    elif result.strategy_type == "limit":
        transition_baseline = result.metrics.get("up_count") is not None
    else:
        transition_baseline = result.metrics.get("repriced_up_count") is not None
    if baseline_present != transition_baseline:
        raise StrategyError(
            "normalized_input.baseline_effect_hash does not match baseline evidence"
        )


def _validate_approval_profit_input(value: object) -> None:
    if not isinstance(value, Mapping):
        raise StrategyError("normalized_input.approval_profit_input must be an object")
    _require_exact_keys(
        value,
        {"ead_col", "pd_col", "params"},
        name="normalized_input.approval_profit_input",
    )
    for key in ("ead_col", "pd_col"):
        column = value[key]
        if not isinstance(column, str) or not column.strip():
            raise StrategyError(
                f"normalized_input.approval_profit_input.{key} must be non-empty"
            )
    params = value["params"]
    if not isinstance(params, Mapping):
        raise StrategyError(
            "normalized_input.approval_profit_input.params must be an object"
        )
    _require_exact_keys(
        params,
        {
            "annual_rate",
            "funding_rate",
            "lgd",
            "operating_cost_per_loan",
            "term_months",
        },
        name="normalized_input.approval_profit_input.params",
    )
    for key in ("annual_rate", "funding_rate", "lgd"):
        _number(
            params,
            key,
            minimum=0.0,
            maximum=1.0,
            name="normalized_input.approval_profit_input.params",
        )
    _number(
        params,
        "operating_cost_per_loan",
        minimum=0.0,
        name="normalized_input.approval_profit_input.params",
    )
    _integer(
        params,
        "term_months",
        minimum=1,
        name="normalized_input.approval_profit_input.params",
    )


def _validate_action_result(result: StrategyBacktestResult) -> None:
    expected_keys = set(_ACTION_METRIC_KEYS)
    if result.strategy_type == "reject":
        expected_keys.update({"bad_capture_rate", "good_reject_rate"})
    _require_exact_keys(result.metrics, expected_keys, name="metrics")
    metrics = result.metrics

    overall_bad_count = _integer(metrics, "overall_bad_count")
    if overall_bad_count > result.labeled_count:
        raise StrategyError("metrics.overall_bad_count exceeds labeled_count")
    _require_rate(
        metrics,
        "overall_bad_rate",
        numerator=overall_bad_count,
        denominator=result.labeled_count,
        empty_value=None,
    )

    counts: dict[str, int] = {}
    labeled_counts: dict[str, int] = {}
    bad_counts: dict[str, int] = {}
    for action in _ACTION_ORDER:
        count = _integer(metrics, f"{action}_count")
        labeled_count = _integer(metrics, f"{action}_labeled_count")
        bad_count = _integer(metrics, f"{action}_bad_count")
        if labeled_count > count:
            raise StrategyError(
                f"metrics.{action}_labeled_count exceeds {action}_count"
            )
        if bad_count > labeled_count:
            raise StrategyError(
                f"metrics.{action}_bad_count exceeds {action}_labeled_count"
            )
        _require_rate(
            metrics,
            f"{action}_rate",
            numerator=count,
            denominator=result.population_count,
            empty_value=0.0,
        )
        _require_rate(
            metrics,
            f"{action}_bad_rate",
            numerator=bad_count,
            denominator=labeled_count,
            empty_value=None,
        )
        counts[action] = count
        labeled_counts[action] = labeled_count
        bad_counts[action] = bad_count

    if sum(counts.values()) != result.population_count:
        raise StrategyError("action counts must sum to population_count")
    if sum(labeled_counts.values()) != result.labeled_count:
        raise StrategyError("action labeled counts must sum to labeled_count")
    if sum(bad_counts.values()) != overall_bad_count:
        raise StrategyError("action bad counts must sum to overall_bad_count")

    if result.strategy_type == "reject":
        _require_rate(
            metrics,
            "bad_capture_rate",
            numerator=bad_counts["reject"],
            denominator=overall_bad_count,
            empty_value=None,
        )
        total_good = result.labeled_count - overall_bad_count
        rejected_good = labeled_counts["reject"] - bad_counts["reject"]
        _require_rate(
            metrics,
            "good_reject_rate",
            numerator=rejected_good,
            denominator=total_good,
            empty_value=None,
        )

    if len(result.breakdown) != len(_ACTION_ORDER):
        raise StrategyError("approval/reject breakdown must contain three action rows")
    for position, (action, row) in enumerate(
        zip(_ACTION_ORDER, result.breakdown, strict=True)
    ):
        _require_exact_keys(
            row,
            _ACTION_BREAKDOWN_KEYS,
            name=f"breakdown[{position}]",
        )
        if row["action"] != action:
            raise StrategyError("approval/reject breakdown action order is invalid")
        _require_equal(row["count"], counts[action], name=f"breakdown[{position}].count")
        _require_equal(
            row["labeled_count"],
            labeled_counts[action],
            name=f"breakdown[{position}].labeled_count",
        )
        _require_equal(
            row["bad_count"],
            bad_counts[action],
            name=f"breakdown[{position}].bad_count",
        )
        _require_equal(
            row["rate"],
            metrics[f"{action}_rate"],
            name=f"breakdown[{position}].rate",
        )
        _require_equal(
            row["bad_rate"],
            metrics[f"{action}_bad_rate"],
            name=f"breakdown[{position}].bad_rate",
        )

    _validate_action_transitions(result)
    _validate_approval_economics(result.economics, approve_count=counts["approve"])


def _validate_action_transitions(result: StrategyBacktestResult) -> None:
    if not result.transitions:
        return
    if len(result.transitions) != len(_ACTION_ORDER) ** 2:
        raise StrategyError("approval/reject transitions must be a complete 3x3 matrix")
    expected_pairs = [
        (old_action, new_action)
        for old_action in _ACTION_ORDER
        for new_action in _ACTION_ORDER
    ]
    actual_pairs: list[tuple[object, object]] = []
    counts_by_from = {action: 0 for action in _ACTION_ORDER}
    counts_by_to = {action: 0 for action in _ACTION_ORDER}
    labeled_counts_by_to = {action: 0 for action in _ACTION_ORDER}
    bad_counts_by_to = {action: 0 for action in _ACTION_ORDER}
    count_total = labeled_total = bad_total = 0
    for position, row in enumerate(result.transitions):
        _require_exact_keys(
            row,
            _ACTION_TRANSITION_KEYS,
            name=f"transitions[{position}]",
        )
        old_action = row["from_action"]
        new_action = row["to_action"]
        if old_action not in _ACTION_ORDER or new_action not in _ACTION_ORDER:
            raise StrategyError(
                "approval/reject transition references an unknown action"
            )
        actual_pairs.append((old_action, new_action))
        count = _integer(row, "count", name=f"transitions[{position}]")
        labeled_count = _integer(
            row,
            "labeled_count",
            name=f"transitions[{position}]",
        )
        bad_count = _integer(row, "bad_count", name=f"transitions[{position}]")
        if labeled_count > count or bad_count > labeled_count:
            raise StrategyError("transition label counts exceed their population")
        counts_by_from[old_action] += count
        counts_by_to[new_action] += count
        labeled_counts_by_to[new_action] += labeled_count
        bad_counts_by_to[new_action] += bad_count
        count_total += count
        labeled_total += labeled_count
        bad_total += bad_count
        _require_rate(
            row,
            "population_share",
            numerator=count,
            denominator=result.population_count,
            empty_value=0.0,
            name=f"transitions[{position}]",
        )
        _require_rate(
            row,
            "bad_rate",
            numerator=bad_count,
            denominator=labeled_count,
            empty_value=None,
            name=f"transitions[{position}]",
        )
    if actual_pairs != expected_pairs:
        raise StrategyError("approval/reject transition order is invalid")
    if count_total != result.population_count:
        raise StrategyError("transition counts must sum to population_count")
    if labeled_total != result.labeled_count:
        raise StrategyError("transition labeled counts must sum to labeled_count")
    if bad_total != result.metrics["overall_bad_count"]:
        raise StrategyError("transition bad counts must sum to overall_bad_count")
    for action in _ACTION_ORDER:
        _require_equal(
            counts_by_to[action],
            result.metrics[f"{action}_count"],
            name=f"transitions.to_action.{action}.count",
        )
        _require_equal(
            labeled_counts_by_to[action],
            result.metrics[f"{action}_labeled_count"],
            name=f"transitions.to_action.{action}.labeled_count",
        )
        _require_equal(
            bad_counts_by_to[action],
            result.metrics[f"{action}_bad_count"],
            name=f"transitions.to_action.{action}.bad_count",
        )
    for position, row in enumerate(result.transitions):
        count = int(row["count"])
        old_action = str(row["from_action"])
        _require_rate(
            row,
            "rate",
            numerator=count,
            denominator=counts_by_from[old_action],
            empty_value=None,
            name=f"transitions[{position}]",
        )


def _validate_approval_economics(
    economics: Mapping[str, Any], *, approve_count: int
) -> None:
    if not economics:
        return
    compatibility_keys = {"expected_profit", "profit_note"}
    if set(economics) == compatibility_keys:
        if economics["expected_profit"] is not None:
            raise StrategyError(
                "economics.expected_profit cannot be supplied by the caller; "
                "use ApprovalProfitInputs"
            )
        note = economics["profit_note"]
        if not isinstance(note, str) or not note.strip():
            raise StrategyError(
                "economics.profit_note must explain why expected_profit is unavailable"
            )
        return
    expected_keys = set(_PRICING_ECONOMICS_KEYS) | compatibility_keys
    _require_exact_keys(economics, expected_keys, name="economics")
    _validate_pricing_economics(
        economics,
        expected_rows=approve_count,
        baseline_present=False,
        name="economics",
    )
    expected_profit = _optional_number(economics, "expected_profit", name="economics")
    if expected_profit is None:
        raise StrategyError("calculated economics.expected_profit cannot be null")
    _require_equal(expected_profit, economics["profit"], name="economics.expected_profit")
    if economics["profit_note"] is not None:
        raise StrategyError("calculated economics.profit_note must be null")


def _validate_limit_result(result: StrategyBacktestResult) -> None:
    metric_keys = {
        "count",
        "total_limit",
        "mean_limit",
        "min_limit",
        "max_limit",
        "up_count",
        "down_count",
        "unchanged_count",
        "total_limit_delta",
    }
    _require_exact_keys(result.metrics, metric_keys, name="metrics")
    metrics = result.metrics
    count = _integer(metrics, "count")
    if count != result.population_count:
        raise StrategyError("metrics.count must equal population_count")
    total_limit = _number(metrics, "total_limit", minimum=0.0)
    mean_limit = _optional_number(metrics, "mean_limit", minimum=0.0)
    min_limit = _optional_number(metrics, "min_limit", minimum=0.0)
    max_limit = _optional_number(metrics, "max_limit", minimum=0.0)
    if result.population_count == 0:
        if any(value is not None for value in (mean_limit, min_limit, max_limit)):
            raise StrategyError("empty limit population statistics must be null")
        _require_equal(total_limit, 0.0, name="metrics.total_limit")
    else:
        if any(value is None for value in (mean_limit, min_limit, max_limit)):
            raise StrategyError("non-empty limit population statistics cannot be null")
        assert mean_limit is not None and min_limit is not None and max_limit is not None
        if not min_limit <= mean_limit <= max_limit:
            raise StrategyError("limit min/mean/max ordering is invalid")
        _require_equal(
            mean_limit,
            total_limit / result.population_count,
            name="metrics.mean_limit",
        )
    baseline_present = _validate_change_metrics(
        metrics,
        labels=("up", "down", "unchanged"),
        population_count=result.population_count,
        delta_key="total_limit_delta",
    )
    _validate_numeric_layers(
        result,
        value_key="assigned_limit",
        maximum=None,
        expected_total=total_limit,
    )
    _validate_change_transitions(
        result,
        labels=("up", "down", "unchanged"),
        baseline_present=baseline_present,
    )
    _require_exact_keys(
        result.economics,
        set() if not result.economics else {"expected_ead", "expected_loss"},
        name="economics",
    )
    if result.economics:
        expected_ead = _number(result.economics, "expected_ead", minimum=0.0)
        expected_loss = _number(result.economics, "expected_loss", minimum=0.0)
        if expected_loss > expected_ead and not math.isclose(
            expected_loss, expected_ead, rel_tol=0.0, abs_tol=1e-12
        ):
            raise StrategyError("economics.expected_loss exceeds expected_ead")


def _validate_pricing_result(result: StrategyBacktestResult) -> None:
    metric_keys = {
        "count",
        "mean_rate",
        "repriced_up_count",
        "repriced_down_count",
        "unchanged_count",
    }
    _require_exact_keys(result.metrics, metric_keys, name="metrics")
    count = _integer(result.metrics, "count")
    if count != result.population_count:
        raise StrategyError("metrics.count must equal population_count")
    mean_rate = _optional_number(
        result.metrics,
        "mean_rate",
        minimum=0.0,
        maximum=1.0,
    )
    if result.population_count == 0 and mean_rate is not None:
        raise StrategyError("empty pricing population mean_rate must be null")
    if result.population_count and mean_rate is None:
        raise StrategyError("non-empty pricing population mean_rate cannot be null")
    baseline_present = _validate_change_metrics(
        result.metrics,
        labels=("repriced_up", "repriced_down", "unchanged"),
        population_count=result.population_count,
    )
    _validate_numeric_layers(
        result,
        value_key="assigned_rate",
        maximum=1.0,
        expected_total=(
            None
            if mean_rate is None
            else mean_rate * result.population_count
        ),
    )
    _validate_change_transitions(
        result,
        labels=("repriced_up", "repriced_down", "unchanged"),
        baseline_present=baseline_present,
    )
    if not result.economics:
        return
    _require_exact_keys(result.economics, _PRICING_ECONOMICS_KEYS, name="economics")
    _validate_pricing_economics(
        result.economics,
        expected_rows=result.population_count,
        baseline_present=baseline_present,
        name="economics",
    )


def _validate_change_metrics(
    metrics: Mapping[str, Any],
    *,
    labels: tuple[str, str, str],
    population_count: int,
    delta_key: str | None = None,
) -> bool:
    fields = [f"{label}_count" for label in labels]
    values = [metrics[field] for field in fields]
    baseline_present = any(value is not None for value in values)
    if baseline_present and any(value is None for value in values):
        raise StrategyError("baseline change counts must be all present or all null")
    if baseline_present:
        counts = [_integer(metrics, field) for field in fields]
        if sum(counts) != population_count:
            raise StrategyError("baseline change counts must sum to population_count")
    if delta_key is not None:
        delta = metrics[delta_key]
        if baseline_present:
            _number(metrics, delta_key)
        elif delta is not None:
            raise StrategyError(f"metrics.{delta_key} must be null without baseline")
    return baseline_present


def _validate_numeric_layers(
    result: StrategyBacktestResult,
    *,
    value_key: str,
    maximum: float | None,
    expected_total: float | None,
) -> None:
    if result.population_count == 0:
        if result.breakdown:
            raise StrategyError("empty economic strategy breakdown must be empty")
        return
    if not result.breakdown:
        raise StrategyError("non-empty economic strategy breakdown cannot be empty")
    values: list[float] = []
    total_count = total_labeled = 0
    weighted_total = 0.0
    for position, row in enumerate(result.breakdown):
        _require_exact_keys(
            row,
            set(_LAYER_KEYS) | {value_key},
            name=f"breakdown[{position}]",
        )
        value = _number(
            row,
            value_key,
            minimum=0.0,
            maximum=maximum,
            name=f"breakdown[{position}]",
        )
        count = _integer(row, "count", minimum=1, name=f"breakdown[{position}]")
        labeled_count = _integer(
            row,
            "labeled_count",
            name=f"breakdown[{position}]",
        )
        bad_count = _integer(row, "bad_count", name=f"breakdown[{position}]")
        if labeled_count > count or bad_count > labeled_count:
            raise StrategyError("breakdown label counts exceed their population")
        _require_rate(
            row,
            "share",
            numerator=count,
            denominator=result.population_count,
            empty_value=0.0,
            name=f"breakdown[{position}]",
        )
        _require_rate(
            row,
            "bad_rate",
            numerator=bad_count,
            denominator=labeled_count,
            empty_value=None,
            name=f"breakdown[{position}]",
        )
        values.append(value)
        total_count += count
        total_labeled += labeled_count
        weighted_total += value * count
    if values != sorted(set(values)):
        raise StrategyError(f"{value_key} breakdown values must be unique and sorted")
    if total_count != result.population_count:
        raise StrategyError("breakdown counts must sum to population_count")
    if total_labeled != result.labeled_count:
        raise StrategyError("breakdown labeled counts must sum to labeled_count")
    if expected_total is not None:
        _require_equal(weighted_total, expected_total, name=f"breakdown.{value_key} total")


def _validate_change_transitions(
    result: StrategyBacktestResult,
    *,
    labels: tuple[str, str, str],
    baseline_present: bool,
) -> None:
    if not baseline_present:
        if result.transitions:
            raise StrategyError("change transitions require baseline metrics")
        return
    if len(result.transitions) != len(labels):
        raise StrategyError("change transitions must contain three direction rows")
    for position, (label, row) in enumerate(
        zip(labels, result.transitions, strict=True)
    ):
        _require_exact_keys(
            row,
            {"direction", "count", "rate"},
            name=f"transitions[{position}]",
        )
        if row["direction"] != label:
            raise StrategyError("change transition order is invalid")
        count = _integer(row, "count", name=f"transitions[{position}]")
        _require_equal(
            count,
            result.metrics[f"{label}_count"],
            name=f"transitions[{position}].count",
        )
        _require_rate(
            row,
            "rate",
            numerator=count,
            denominator=result.population_count,
            empty_value=0.0,
            name=f"transitions[{position}]",
        )


def _validate_pricing_economics(
    economics: Mapping[str, Any],
    *,
    expected_rows: int,
    baseline_present: bool,
    name: str,
) -> None:
    total_ead = _number(economics, "total_ead", minimum=0.0, name=name)
    weighted_rate = _optional_number(
        economics,
        "ead_weighted_rate",
        minimum=0.0,
        maximum=1.0,
        name=name,
    )
    if (total_ead == 0.0) != (weighted_rate is None):
        raise StrategyError(f"{name}.ead_weighted_rate nullability is inconsistent")
    revenue = _number(economics, "revenue", minimum=0.0, name=name)
    expected_loss = _number(economics, "expected_loss", minimum=0.0, name=name)
    funding_cost = _number(economics, "funding_cost", minimum=0.0, name=name)
    operating_cost = _number(
        economics,
        "operating_cost",
        minimum=0.0,
        name=name,
    )
    profit = _number(economics, "profit", name=name)
    _require_equal(
        profit,
        revenue - expected_loss - funding_cost - operating_cost,
        name=f"{name}.profit",
    )
    roa = _optional_number(economics, "roa", name=name)
    if total_ead == 0.0:
        if roa is not None:
            raise StrategyError(f"{name}.roa must be null when total_ead is zero")
    else:
        _require_equal(roa, profit / total_ead, name=f"{name}.roa")

    baseline_profit = _optional_number(economics, "baseline_profit", name=name)
    profit_delta = _optional_number(
        economics,
        "profit_delta_vs_baseline",
        name=name,
    )
    if baseline_present:
        if baseline_profit is None or profit_delta is None:
            raise StrategyError(f"{name} baseline profit evidence cannot be null")
        _require_equal(
            profit_delta,
            profit - baseline_profit,
            name=f"{name}.profit_delta_vs_baseline",
        )
    elif baseline_profit is not None or profit_delta is not None:
        raise StrategyError(f"{name} baseline profit evidence requires a baseline")

    rows = economics["by_row"]
    if not isinstance(rows, Sequence) or isinstance(rows, str | bytes | bytearray):
        raise StrategyError(f"{name}.by_row must be a list")
    if len(rows) != expected_rows:
        raise StrategyError(f"{name}.by_row count does not match the population")
    sums = {
        "revenue": 0.0,
        "expected_loss": 0.0,
        "funding_cost": 0.0,
        "operating_cost": 0.0,
        "profit": 0.0,
        "profit_delta_vs_baseline": 0.0,
    }
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise StrategyError(f"{name}.by_row[{position}] must be an object")
        _require_exact_keys(row, _ECONOMIC_ROW_KEYS, name=f"{name}.by_row[{position}]")
        _require_equal(
            _integer(row, "position", name=f"{name}.by_row[{position}]"),
            position,
            name=f"{name}.by_row[{position}].position",
        )
        row_revenue = _number(
            row, "revenue", minimum=0.0, name=f"{name}.by_row[{position}]"
        )
        row_loss = _number(
            row, "expected_loss", minimum=0.0, name=f"{name}.by_row[{position}]"
        )
        row_funding = _number(
            row, "funding_cost", minimum=0.0, name=f"{name}.by_row[{position}]"
        )
        row_operating = _number(
            row, "operating_cost", minimum=0.0, name=f"{name}.by_row[{position}]"
        )
        row_profit = _number(row, "profit", name=f"{name}.by_row[{position}]")
        _require_equal(
            row_profit,
            row_revenue - row_loss - row_funding - row_operating,
            name=f"{name}.by_row[{position}].profit",
        )
        _optional_number(row, "roa", name=f"{name}.by_row[{position}]")
        row_delta = _optional_number(
            row,
            "profit_delta_vs_baseline",
            name=f"{name}.by_row[{position}]",
        )
        if baseline_present and row_delta is None:
            raise StrategyError(f"{name}.by_row baseline delta cannot be null")
        if not baseline_present and row_delta is not None:
            raise StrategyError(f"{name}.by_row delta requires a baseline")
        for key, value in (
            ("revenue", row_revenue),
            ("expected_loss", row_loss),
            ("funding_cost", row_funding),
            ("operating_cost", row_operating),
            ("profit", row_profit),
        ):
            sums[key] += value
        if row_delta is not None:
            sums["profit_delta_vs_baseline"] += row_delta
    for key in ("revenue", "expected_loss", "funding_cost", "operating_cost", "profit"):
        _require_equal(sums[key], economics[key], name=f"{name}.{key}")
    if baseline_present:
        _require_equal(
            sums["profit_delta_vs_baseline"],
            economics["profit_delta_vs_baseline"],
            name=f"{name}.profit_delta_vs_baseline",
        )


def _validate_segmentation_result(result: StrategyBacktestResult) -> None:
    _require_exact_keys(
        result.metrics,
        {"segment_count", "overall_bad_count", "overall_bad_rate"},
        name="metrics",
    )
    segment_count = _integer(result.metrics, "segment_count")
    overall_bad_count = _integer(result.metrics, "overall_bad_count")
    if overall_bad_count > result.labeled_count:
        raise StrategyError("metrics.overall_bad_count exceeds labeled_count")
    _require_rate(
        result.metrics,
        "overall_bad_rate",
        numerator=overall_bad_count,
        denominator=result.labeled_count,
        empty_value=None,
    )
    if segment_count != len(result.breakdown):
        raise StrategyError("metrics.segment_count does not match breakdown")
    if result.population_count and not result.breakdown:
        raise StrategyError("non-empty segmentation breakdown cannot be empty")
    tokens: list[str] = []
    counts_by_token: dict[str, int] = {}
    total_count = total_labeled = total_bad = 0
    for position, row in enumerate(result.breakdown):
        _require_exact_keys(
            row,
            {"segment", "count", "share", "labeled_count", "bad_count", "bad_rate", "lift"},
            name=f"breakdown[{position}]",
        )
        token = _segment_token(row["segment"])
        count = _integer(row, "count", minimum=1, name=f"breakdown[{position}]")
        labeled_count = _integer(
            row,
            "labeled_count",
            name=f"breakdown[{position}]",
        )
        bad_count = _integer(row, "bad_count", name=f"breakdown[{position}]")
        if labeled_count > count or bad_count > labeled_count:
            raise StrategyError("segment label counts exceed their population")
        _require_rate(
            row,
            "share",
            numerator=count,
            denominator=result.population_count,
            empty_value=0.0,
            name=f"breakdown[{position}]",
        )
        bad_rate = _require_rate(
            row,
            "bad_rate",
            numerator=bad_count,
            denominator=labeled_count,
            empty_value=None,
            name=f"breakdown[{position}]",
        )
        overall_bad_rate = result.metrics["overall_bad_rate"]
        expected_lift = (
            None
            if bad_rate is None or overall_bad_rate in {None, 0.0}
            else float(bad_rate / overall_bad_rate)
        )
        _require_equal(row["lift"], expected_lift, name=f"breakdown[{position}].lift")
        tokens.append(token)
        counts_by_token[token] = count
        total_count += count
        total_labeled += labeled_count
        total_bad += bad_count
    if tokens != sorted(set(tokens)):
        raise StrategyError("segment breakdown values must be unique and sorted")
    if total_count != result.population_count:
        raise StrategyError("segment counts must sum to population_count")
    if total_labeled != result.labeled_count:
        raise StrategyError("segment labeled counts must sum to labeled_count")
    if total_bad != overall_bad_count:
        raise StrategyError("segment bad counts must sum to overall_bad_count")
    if result.economics:
        raise StrategyError("segmentation economics must be empty")
    _validate_segment_transitions(result, candidate_counts=counts_by_token)


def _validate_segment_transitions(
    result: StrategyBacktestResult,
    *,
    candidate_counts: Mapping[str, int],
) -> None:
    if not result.transitions:
        return
    candidate_tokens = list(candidate_counts)
    if not candidate_tokens:
        raise StrategyError("segment transitions require candidate segments")
    actual_pairs: list[tuple[str, str]] = []
    from_counts: dict[str, int] = {}
    to_counts = {token: 0 for token in candidate_tokens}
    for position, row in enumerate(result.transitions):
        _require_exact_keys(
            row,
            {"from_segment", "to_segment", "count", "rate", "population_share"},
            name=f"transitions[{position}]",
        )
        from_token = _segment_token(row["from_segment"])
        to_token = _segment_token(row["to_segment"])
        if to_token not in candidate_counts:
            raise StrategyError("segment transition references an unknown candidate segment")
        count = _integer(row, "count", name=f"transitions[{position}]")
        from_counts[from_token] = from_counts.get(from_token, 0) + count
        to_counts[to_token] += count
        actual_pairs.append((from_token, to_token))
        _require_rate(
            row,
            "population_share",
            numerator=count,
            denominator=result.population_count,
            empty_value=0.0,
            name=f"transitions[{position}]",
        )
    from_tokens = sorted(from_counts)
    expected_pairs = [
        (from_token, to_token)
        for from_token in from_tokens
        for to_token in candidate_tokens
    ]
    if actual_pairs != expected_pairs:
        raise StrategyError("segment transitions must be a complete stable matrix")
    if sum(from_counts.values()) != result.population_count:
        raise StrategyError("segment transition counts must sum to population_count")
    if to_counts != dict(candidate_counts):
        raise StrategyError("segment transition candidate counts do not match breakdown")
    for position, row in enumerate(result.transitions):
        count = int(row["count"])
        from_token = _segment_token(row["from_segment"])
        _require_rate(
            row,
            "rate",
            numerator=count,
            denominator=from_counts[from_token],
            empty_value=None,
            name=f"transitions[{position}]",
        )


def _require_exact_keys(
    payload: Mapping[str, Any], expected: set[str], *, name: str
) -> None:
    actual = set(payload)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        raise StrategyError(f"{name} is missing fields: {', '.join(missing)}")
    if unexpected:
        raise StrategyError(f"{name} has unsupported fields: {', '.join(unexpected)}")


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StrategyError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _integer(
    payload: Mapping[str, Any],
    key: str,
    *,
    minimum: int = 0,
    name: str = "metrics",
) -> int:
    value = payload[key]
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise StrategyError(f"{name}.{key} must be an integer >= {minimum}")
    return value


def _number(
    payload: Mapping[str, Any],
    key: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    name: str = "metrics",
) -> float:
    value = payload[key]
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise StrategyError(f"{name}.{key} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise StrategyError(f"{name}.{key} must be a finite number")
    if minimum is not None and number < minimum:
        raise StrategyError(f"{name}.{key} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise StrategyError(f"{name}.{key} must be <= {maximum}")
    return number


def _optional_number(
    payload: Mapping[str, Any],
    key: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    name: str = "metrics",
) -> float | None:
    if payload[key] is None:
        return None
    return _number(payload, key, minimum=minimum, maximum=maximum, name=name)


def _require_rate(
    payload: Mapping[str, Any],
    key: str,
    *,
    numerator: int,
    denominator: int,
    empty_value: float | None,
    name: str = "metrics",
) -> float | None:
    value = _optional_number(
        payload,
        key,
        minimum=0.0,
        maximum=1.0,
        name=name,
    )
    expected = empty_value if denominator == 0 else float(numerator / denominator)
    _require_equal(value, expected, name=f"{name}.{key}")
    return value


def _require_equal(actual: object, expected: object, *, name: str) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise StrategyError(f"{name} is inconsistent with its denominator")
        return
    if (
        isinstance(actual, int | float)
        and not isinstance(actual, bool)
        and isinstance(expected, int | float)
        and not isinstance(expected, bool)
    ):
        if math.isclose(
            float(actual),
            float(expected),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            return
    elif actual == expected:
        return
    raise StrategyError(f"{name} is inconsistent with the canonical result")


def run_typed_backtest(
    frame: pd.DataFrame,
    spec: StrategySpec | Mapping[str, Any],
    *,
    target_col: str,
    strategy_id: str | None = None,
    baseline: StrategySpec | Mapping[str, Any] | None = None,
    economics: Mapping[str, Any] | None = None,
    economics_inputs: Mapping[str, NumericInput] | None = None,
    approval_profit_inputs: ApprovalProfitInputs | None = None,
) -> StrategyBacktestResult:
    """Run a deterministic typed backtest without persistence or tool side effects.

    Population action/segment rates always use all rows. Bad-rate and capture
    metrics use only rows carrying a valid binary target; missing targets remain in
    the population and are never silently converted to good outcomes.
    """

    if not isinstance(frame, pd.DataFrame):
        raise StrategyError("strategy backtest rows must be a DataFrame")
    if not isinstance(target_col, str) or not target_col.strip():
        raise StrategyError("target_col must be a non-empty string")
    parsed = parse_strategy_spec(spec)
    if parsed.strategy_type not in _SUPPORTED_STRATEGY_TYPES:
        raise StrategyError(
            f"typed backtest does not support strategy type {parsed.strategy_type}"
        )
    parsed_baseline = parse_strategy_spec(baseline) if baseline is not None else None
    if (
        parsed_baseline is not None
        and parsed_baseline.strategy_type != parsed.strategy_type
    ):
        raise StrategyError(
            "baseline strategy type must match candidate strategy type"
        )
    if economics is not None and (
        economics_inputs is not None or approval_profit_inputs is not None
    ):
        raise StrategyError(
            "economics cannot be combined with calculated economics inputs"
        )
    if economics_inputs is not None and parsed.strategy_type not in {"limit", "pricing"}:
        raise StrategyError(
            "economics_inputs is only supported for limit and pricing strategies"
        )
    if economics is not None and parsed.strategy_type in {"limit", "pricing"}:
        raise StrategyError(
            "precomputed economics is not accepted for limit/pricing; "
            "use economics_inputs"
        )
    if approval_profit_inputs is not None and parsed.strategy_type not in {
        "approval",
        "reject",
    }:
        raise StrategyError(
            "approval_profit_inputs is only supported for approval/reject strategies"
        )
    target = _normalized_target(frame, target_col)
    evaluation = evaluate_strategy_frame(frame, parsed)
    baseline_evaluation = (
        None
        if parsed_baseline is None
        else evaluate_strategy_frame(frame, parsed_baseline)
    )
    population_count = len(frame)
    labeled_count = int(target.notna().sum())
    effect_hash = strategy_spec_hash(parsed)
    resolved_strategy_id = strategy_id or f"strategy-{effect_hash[:16]}"

    baseline_hash = (
        None if parsed_baseline is None else strategy_spec_hash(parsed_baseline)
    )
    transitions: tuple[dict[str, Any], ...] = ()
    computed_economics: Mapping[str, Any] = economics or {}
    normalized_economics_inputs: dict[str, NumericInput] = {}
    missing_economics_warning: str | None = None

    if parsed.strategy_type in {"approval", "reject"}:
        actions = _normalized_actions(evaluation.action_type)
        metrics, breakdown = _decision_metrics(
            actions,
            target,
            population_count=population_count,
            include_reject_quality=parsed.strategy_type == "reject",
        )
        if baseline_evaluation is not None:
            baseline_actions = _normalized_actions(baseline_evaluation.action_type)
            transitions = _action_transitions(
                baseline_actions,
                actions,
                target,
                population_count=population_count,
            )
        if approval_profit_inputs is not None:
            computed_economics = _approval_profit_economics(
                frame,
                target,
                actions,
                approval_profit_inputs,
            )
    elif parsed.strategy_type == "segmentation":
        segments = _normalized_segments(evaluation.action_values)
        metrics, breakdown = _segmentation_metrics(
            segments,
            target,
            population_count=population_count,
        )
        if baseline_evaluation is not None:
            baseline_segments = _normalized_segments(baseline_evaluation.action_values)
            transitions = _segment_transitions(
                baseline_segments,
                segments,
                population_count=population_count,
            )
    else:
        normalized_economics_inputs = _normalized_economic_inputs(
            frame,
            economics_inputs,
            strategy_type=parsed.strategy_type,
        )
        baseline_decisions = (
            None
            if baseline_evaluation is None
            else baseline_evaluation.action_values.reset_index(drop=True)
        )
        assigned = evaluation.action_values.reset_index(drop=True)
        if parsed.strategy_type == "limit":
            calculated = limit_metrics(
                assigned,
                target,
                baseline_decisions,
                **normalized_economics_inputs,
            )
            metrics, breakdown, computed_economics = _limit_result(calculated)
            transitions = _change_transitions(
                metrics,
                population_count=population_count,
                labels=("up", "down", "unchanged"),
            )
        else:
            calculated = pricing_metrics(
                assigned,
                target,
                baseline_decisions,
                **normalized_economics_inputs,
            )
            metrics, breakdown, computed_economics = _pricing_result(calculated)
            transitions = _change_transitions(
                metrics,
                population_count=population_count,
                labels=("repriced_up", "repriced_down", "unchanged"),
            )
        if not normalized_economics_inputs:
            missing_economics_warning = (
                f"{parsed.strategy_type} economics inputs were not supplied"
            )

    missing_labels = population_count - labeled_count
    warnings: list[str] = []
    if population_count == 0:
        warnings.append("population is empty")
    if missing_labels:
        warnings.append(f"{missing_labels} population rows have no target label")
    if population_count and labeled_count == 0:
        warnings.append("no labeled rows; bad-rate metrics are undefined")
    if missing_economics_warning is not None:
        warnings.append(missing_economics_warning)

    normalized_input = {
        "strategy_schema_version": parsed.schema_version,
        "strategy_effect_hash": effect_hash,
        "baseline_effect_hash": baseline_hash,
        "target_col": target_col,
        "target_encoding": {"good": 0, "bad": 1},
        "missing_label_policy": "exclude_from_label_metrics",
        "population_rate_denominator": "population_count",
        "bad_rate_denominator": "labeled_group_count",
        "economics_input_kinds": _economic_input_kinds(
            normalized_economics_inputs
        ),
        "economics_input_evidence": _economic_input_evidence(
            normalized_economics_inputs
        ),
        "approval_profit_input": _approval_profit_input_summary(
            approval_profit_inputs
        ),
    }
    return StrategyBacktestResult(
        strategy_id=resolved_strategy_id,
        strategy_type=parsed.strategy_type,
        population_count=population_count,
        labeled_count=labeled_count,
        label_coverage=_ratio(labeled_count, population_count),
        metrics=metrics,
        breakdown=breakdown,
        transitions=transitions,
        economics=computed_economics,
        warnings=tuple(warnings),
        normalized_input=normalized_input,
    )


def _assert_count(value: object, *, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise StrategyError(f"{name} must be a non-negative integer")


def _normalized_target(frame: pd.DataFrame, target_col: str) -> pd.Series:
    if target_col not in frame.columns:
        raise StrategyError(f"missing columns: {target_col}")
    raw = frame[target_col]
    if not isinstance(raw, pd.Series):
        raise StrategyError(f"target_col must identify one column: {target_col}")
    raw = raw.reset_index(drop=True)
    missing = raw.isna()
    numeric = pd.to_numeric(raw, errors="coerce")
    invalid = (~missing) & numeric.isna()
    if bool(invalid.any()) or bool((~numeric.loc[~missing].isin([0, 1])).any()):
        raise StrategyError("target must contain only 0, 1, or missing")
    return numeric.astype(float)


def _normalized_actions(action_types: pd.Series) -> pd.Series:
    actions = action_types.reset_index(drop=True).map(_ACTION_TYPE_TO_DECISION)
    if bool(actions.isna().any()):
        raise StrategyError(
            "approval/reject strategy produced an unsupported decision action"
        )
    return actions.astype("object")


def _decision_metrics(
    actions: pd.Series,
    target: pd.Series,
    *,
    population_count: int,
    include_reject_quality: bool,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    labeled = target.notna()
    overall_bad_count = int(target.loc[labeled].eq(1).sum())
    overall_bad_rate = _optional_ratio(overall_bad_count, int(labeled.sum()))
    metrics: dict[str, Any] = {
        "overall_bad_count": overall_bad_count,
        "overall_bad_rate": overall_bad_rate,
    }
    breakdown: list[dict[str, Any]] = []
    for action in _ACTION_ORDER:
        action_mask = actions.eq(action)
        action_labeled = action_mask & labeled
        count = int(action_mask.sum())
        action_labeled_count = int(action_labeled.sum())
        bad_count = int(target.loc[action_labeled].eq(1).sum())
        bad_rate = _optional_ratio(bad_count, action_labeled_count)
        row = {
            "action": action,
            "count": count,
            "rate": _ratio(count, population_count),
            "labeled_count": action_labeled_count,
            "bad_count": bad_count,
            "bad_rate": bad_rate,
        }
        breakdown.append(row)
        metrics.update(
            {
                f"{action}_count": count,
                f"{action}_rate": row["rate"],
                f"{action}_labeled_count": action_labeled_count,
                f"{action}_bad_count": bad_count,
                f"{action}_bad_rate": bad_rate,
            }
        )
    if include_reject_quality:
        rejected_labeled = actions.eq("reject") & labeled
        rejected_bad = int(target.loc[rejected_labeled].eq(1).sum())
        rejected_good = int(target.loc[rejected_labeled].eq(0).sum())
        total_bad = overall_bad_count
        total_good = int(target.loc[labeled].eq(0).sum())
        metrics["bad_capture_rate"] = _optional_ratio(rejected_bad, total_bad)
        metrics["good_reject_rate"] = _optional_ratio(rejected_good, total_good)
    return metrics, tuple(breakdown)


def _segment_token(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise StrategyError("segmentation decisions must be scalar segment ids")
    if isinstance(value, float) and not math.isfinite(value):
        raise StrategyError("segmentation decisions must be finite segment ids")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _normalized_segments(decisions: pd.Series) -> pd.DataFrame:
    values = decisions.reset_index(drop=True).astype("object")
    tokens = values.map(_segment_token)
    return pd.DataFrame({"value": values, "token": tokens})


def _segment_values(segments: pd.DataFrame) -> list[tuple[str, Any]]:
    values: dict[str, Any] = {}
    for token, value in zip(segments["token"], segments["value"], strict=True):
        values.setdefault(token, value)
    return sorted(values.items(), key=lambda item: item[0])


def _segmentation_metrics(
    segments: pd.DataFrame,
    target: pd.Series,
    *,
    population_count: int,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    labeled = target.notna()
    overall_bad_count = int(target.loc[labeled].eq(1).sum())
    overall_bad_rate = _optional_ratio(overall_bad_count, int(labeled.sum()))
    breakdown: list[dict[str, Any]] = []
    for token, value in _segment_values(segments):
        segment_mask = segments["token"].eq(token)
        segment_labeled = segment_mask & labeled
        count = int(segment_mask.sum())
        labeled_count = int(segment_labeled.sum())
        bad_count = int(target.loc[segment_labeled].eq(1).sum())
        bad_rate = _optional_ratio(bad_count, labeled_count)
        lift = (
            None
            if bad_rate is None or overall_bad_rate in {None, 0.0}
            else float(bad_rate / overall_bad_rate)
        )
        breakdown.append(
            {
                "segment": value,
                "count": count,
                "share": _ratio(count, population_count),
                "labeled_count": labeled_count,
                "bad_count": bad_count,
                "bad_rate": bad_rate,
                "lift": lift,
            }
        )
    metrics = {
        "segment_count": len(breakdown),
        "overall_bad_count": overall_bad_count,
        "overall_bad_rate": overall_bad_rate,
    }
    return metrics, tuple(breakdown)


def _approval_profit_economics(
    frame: pd.DataFrame,
    target: pd.Series,
    actions: pd.Series,
    inputs: ApprovalProfitInputs,
) -> dict[str, Any]:
    approved = actions.eq("approve")
    columns: dict[str, pd.Series] = {}
    for name, column in (("ead", inputs.ead_col), ("pd", inputs.pd_col)):
        if column not in frame.columns:
            raise StrategyError(f"missing columns: {column}")
        values = frame[column]
        if not isinstance(values, pd.Series):
            raise StrategyError(f"approval profit {name} must identify one column")
        columns[name] = values.reset_index(drop=True).loc[approved].reset_index(drop=True)

    params = inputs.params
    approved_target = target.loc[approved].reset_index(drop=True)
    assigned_rate = pd.Series(
        [params.annual_rate] * int(approved.sum()),
        dtype=float,
    )
    calculated = pricing_metrics(
        assigned_rate,
        approved_target,
        ead=columns["ead"],
        pd=columns["pd"],
        lgd=params.lgd,
        funding_rate=params.funding_rate,
        term_months=params.term_months,
        operating_cost_per_loan=params.operating_cost_per_loan,
    )
    economics = dict(calculated["economics"] or {})
    economics["expected_profit"] = economics["profit"]
    economics["profit_note"] = None
    return economics


def _approval_profit_input_summary(
    inputs: ApprovalProfitInputs | None,
) -> dict[str, Any] | None:
    if inputs is None:
        return None
    params = inputs.params
    return {
        "ead_col": inputs.ead_col,
        "pd_col": inputs.pd_col,
        "params": {
            "annual_rate": params.annual_rate,
            "funding_rate": params.funding_rate,
            "lgd": params.lgd,
            "operating_cost_per_loan": params.operating_cost_per_loan,
            "term_months": params.term_months,
        },
    }


def _normalized_economic_inputs(
    frame: pd.DataFrame,
    inputs: Mapping[str, NumericInput] | None,
    *,
    strategy_type: str,
) -> dict[str, NumericInput]:
    if inputs is None:
        return {}
    if not isinstance(inputs, Mapping):
        raise StrategyError("economics_inputs must be an object")
    keys = list(inputs)
    if any(not isinstance(key, str) for key in keys):
        raise StrategyError("economics_inputs keys must be strings")
    allowed = (
        {"pd", "lgd", "utilization"}
        if strategy_type == "limit"
        else {
            "ead",
            "pd",
            "lgd",
            "funding_rate",
            "term_months",
            "operating_cost_per_loan",
        }
    )
    unsupported = sorted(set(keys) - allowed)
    if unsupported:
        raise StrategyError(
            f"unsupported {strategy_type} economics inputs: {', '.join(unsupported)}"
        )
    normalized: dict[str, NumericInput] = {}
    for key in sorted(keys):
        value = inputs[key]
        if value is None:
            raise StrategyError(f"economics input {key} cannot be null")
        if isinstance(value, pd.Series):
            if not value.index.equals(frame.index):
                raise StrategyError(
                    f"economics input {key} index must exactly match backtest rows"
                )
            normalized[key] = value.reset_index(drop=True)
        else:
            normalized[key] = value
    return normalized


def _economic_input_kinds(inputs: Mapping[str, NumericInput]) -> dict[str, str]:
    return {
        key: "series" if isinstance(value, pd.Series) else "scalar"
        for key, value in sorted(inputs.items())
    }


def _economic_input_evidence(
    inputs: Mapping[str, NumericInput],
) -> dict[str, dict[str, Any]]:
    """Return deterministic input identity without persisting row-level values."""

    evidence: dict[str, dict[str, Any]] = {}
    for key, value in sorted(inputs.items()):
        if isinstance(value, pd.Series):
            canonical_values = [
                _json_value(item, path=f"economics_inputs.{key}[{position}]")
                for position, item in enumerate(value.tolist())
            ]
            canonical = json.dumps(
                canonical_values,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            evidence[key] = {
                "kind": "series",
                "name": None if value.name is None else str(value.name),
                "row_count": len(value),
                "content_hash": hashlib.sha256(canonical).hexdigest(),
            }
        else:
            evidence[key] = {
                "kind": "scalar",
                "value": _json_value(value, path=f"economics_inputs.{key}"),
            }
    return evidence


def _limit_result(
    calculated: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], Mapping[str, Any]]:
    baseline = calculated["baseline"]
    metrics = {
        "count": calculated["count"],
        "total_limit": calculated["total_limit"],
        "mean_limit": calculated["mean_limit"],
        "min_limit": calculated["min_limit"],
        "max_limit": calculated["max_limit"],
        "up_count": None if baseline is None else baseline["up_count"],
        "down_count": None if baseline is None else baseline["down_count"],
        "unchanged_count": (
            None if baseline is None else baseline["unchanged_count"]
        ),
        "total_limit_delta": (
            None if baseline is None else baseline["total_limit_delta"]
        ),
    }
    economics = calculated["economics"]
    return metrics, tuple(calculated["by_limit"]), economics or {}


def _pricing_result(
    calculated: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], Mapping[str, Any]]:
    baseline = calculated["baseline"]
    metrics = {
        "count": calculated["count"],
        "mean_rate": calculated["mean_rate"],
        "repriced_up_count": (
            None if baseline is None else baseline["repriced_up_count"]
        ),
        "repriced_down_count": (
            None if baseline is None else baseline["repriced_down_count"]
        ),
        "unchanged_count": (
            None if baseline is None else baseline["unchanged_count"]
        ),
    }
    economics = calculated["economics"]
    return metrics, tuple(calculated["risk_tiers"]), economics or {}


def _change_transitions(
    metrics: Mapping[str, Any],
    *,
    population_count: int,
    labels: tuple[str, str, str],
) -> tuple[dict[str, Any], ...]:
    counts = [metrics[f"{label}_count"] for label in labels]
    if any(count is None for count in counts):
        return ()
    return tuple(
        {
            "direction": label,
            "count": int(count),
            "rate": _ratio(int(count), population_count),
        }
        for label, count in zip(labels, counts, strict=True)
    )


def _action_transitions(
    baseline: pd.Series,
    candidate: pd.Series,
    target: pd.Series,
    *,
    population_count: int,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    labeled = target.notna()
    for old_action in _ACTION_ORDER:
        old_mask = baseline.eq(old_action)
        old_count = int(old_mask.sum())
        for new_action in _ACTION_ORDER:
            transition_mask = old_mask & candidate.eq(new_action)
            transition_labeled = transition_mask & labeled
            count = int(transition_mask.sum())
            labeled_count = int(transition_labeled.sum())
            bad_count = int(target.loc[transition_labeled].eq(1).sum())
            rows.append(
                {
                    "from_action": old_action,
                    "to_action": new_action,
                    "count": count,
                    "rate": _optional_ratio(count, old_count),
                    "population_share": _ratio(count, population_count),
                    "labeled_count": labeled_count,
                    "bad_count": bad_count,
                    "bad_rate": _optional_ratio(bad_count, labeled_count),
                }
            )
    return tuple(rows)


def _segment_transitions(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    population_count: int,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for old_token, old_value in _segment_values(baseline):
        old_mask = baseline["token"].eq(old_token)
        old_count = int(old_mask.sum())
        for new_token, new_value in _segment_values(candidate):
            count = int((old_mask & candidate["token"].eq(new_token)).sum())
            rows.append(
                {
                    "from_segment": old_value,
                    "to_segment": new_value,
                    "count": count,
                    "rate": _optional_ratio(count, old_count),
                    "population_share": _ratio(count, population_count),
                }
            )
    return tuple(rows)


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def _optional_ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


__all__ = [
    "ApprovalProfitInputs",
    "STRATEGY_BACKTEST_SCHEMA_VERSION",
    "StrategyBacktestResult",
    "run_typed_backtest",
]
