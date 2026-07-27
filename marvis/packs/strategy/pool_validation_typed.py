"""Typed independent validation evidence for non-decision Strategy Pools.

Approval and reject Pools retain the byte-stable V1 replay contract. Limit,
pricing, and segmentation Pools use this V2 contract so their native outputs
remain numeric/segment evidence instead of being coerced into approval actions.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
import hmac
import json
import math
from numbers import Integral, Real
from typing import Any

import pandas as pd

from marvis.packs.strategy.dsl import strategy_spec_hash
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pool import (
    compile_strategy_pool,
    validate_strategy_pool,
)
from marvis.packs.strategy.typed_backtest import (
    StrategyBacktestResult,
    run_typed_backtest,
)


STRATEGY_POOL_TYPED_VALIDATION_SCHEMA_VERSION = (
    "strategy.pool-validation-evidence.v2"
)
STRATEGY_POOL_TYPED_VALIDATION_PRODUCER_VERSION = (
    "marvis.strategy.pool-validation/2"
)

_TYPED_STRATEGY_TYPES = frozenset({"limit", "pricing", "segmentation"})
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "evidence_id",
        "identity",
        "source_bindings",
        "partition",
        "population",
        "comparison_mode",
        "lifecycle",
        "population_metrics",
        "risk_summary",
        "typed_backtest",
        "monthly",
        "conservation",
        "red_flags",
        "content_hash",
    }
)
_RISK_SUMMARY_FIELDS = frozenset(
    {"overall_bad_count", "overall_bad_rate"}
)
_POPULATION_FIELDS = frozenset(
    {
        "population_count",
        "labelled_count",
        "unlabelled_count",
        "label_coverage",
    }
)
_MONTHLY_FIELDS = frozenset({"status", "column", "periods"})
_MONTHLY_PERIOD_FIELDS = frozenset(
    {"value", "risk_summary", "typed_backtest"}
)
_CONSERVATION_FIELDS = frozenset(
    {
        "typed_population_equals_membership_count",
        "typed_breakdown_equals_population",
        "monthly_rolls_to_overall",
        "selected_partition_equals_membership_count",
        "risk_partition_excludes_development",
    }
)
_RED_FLAG_FIELDS = frozenset({"code", "level", "message"})


def build_typed_strategy_pool_validation_evidence(
    *,
    pool: Mapping[str, Any],
    frame: pd.DataFrame,
    pool_artifact_ref: Mapping[str, Any],
    sample_design_v2_ref: Mapping[str, Any],
    dataset_binding: Mapping[str, Any],
    legacy_development_ref: Mapping[str, Any],
    partition: str,
    population: str,
    comparison_mode: str,
    target_col: str,
    target_bad_value: int,
    month_col: str | None = None,
    loan_amount_col: str | None = None,
    overdue_amount_col: str | None = None,
    development_rows_excluded: bool,
) -> dict[str, Any]:
    """Build aggregate-only typed evidence on one exact independent partition."""

    from marvis.packs.strategy.pool_validation import (
        _canonical_json,
        _dataset_binding,
        _expected_lifecycle,
        _field_bindings,
        _pool_artifact_ref,
        _pool_sample_binding,
        _risk_development_ref,
        _sample_design_v2_ref,
        _sha256,
        _target_bad_value,
        _target_binding,
    )

    current_pool = validate_strategy_pool(pool)
    strategy_type = current_pool["strategy_type"]
    if strategy_type not in _TYPED_STRATEGY_TYPES:
        raise StrategyError(
            "typed Strategy Pool validation requires limit, pricing, or "
            "segmentation"
        )
    if partition not in {"validation", "oot"}:
        raise StrategyError(
            "Strategy Pool validation partition must be validation or oot"
        )
    if population != "risk":
        raise StrategyError("Strategy Pool validation population must be risk")
    if comparison_mode != "absolute":
        raise StrategyError(
            "Strategy Pool validation comparison_mode must be absolute"
        )
    if development_rows_excluded is not True:
        raise StrategyError(
            "Strategy Pool validation must exclude risk/development rows"
        )
    if not isinstance(frame, pd.DataFrame):
        raise StrategyError("Strategy Pool validation rows must be a DataFrame")
    if frame.empty:
        raise StrategyError(f"Strategy Pool {partition} partition is empty")

    pool_artifact = _pool_artifact_ref(pool_artifact_ref)
    sample_v2 = _sample_design_v2_ref(
        sample_design_v2_ref,
        partition=partition,
    )
    if len(frame) != sample_v2["partition_count"]:
        raise StrategyError(
            "Strategy Pool validation selected rows do not match membership count"
        )
    dataset = _dataset_binding(dataset_binding)
    if dataset["task_id"] != current_pool["task_id"]:
        raise StrategyError(
            "Strategy Pool validation dataset belongs to another task"
        )
    development_ref = _risk_development_ref(legacy_development_ref)
    sample_binding = _pool_sample_binding(
        current_pool,
        task_id=current_pool["task_id"],
    )
    for field in (
        "dataset_id",
        "dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
    ):
        if sample_binding[field] != dataset[field]:
            raise StrategyError(
                f"Strategy Pool development lineage {field} does not match "
                "the V2 dataset binding"
            )

    target = _target_binding(
        {
            "column": target_col,
            "good_value": 1 - _target_bad_value(target_bad_value),
            "bad_value": target_bad_value,
            "missing_policy": "retain_population_exclude_risk_denominator",
        }
    )
    fields = _field_bindings(
        {
            "month_col": month_col,
            "loan_amount_col": loan_amount_col,
            "overdue_amount_col": overdue_amount_col,
        },
        target_col=target["column"],
    )
    compiled = compile_strategy_pool(current_pool)
    typed_result = run_typed_backtest(
        frame,
        compiled["strategy_spec"],
        target_col=target["column"],
        target_bad_value=target["bad_value"],
    ).to_dict()
    monthly = _build_monthly_evidence(
        frame,
        spec=compiled["strategy_spec"],
        strategy_type=strategy_type,
        target_col=target["column"],
        target_bad_value=target["bad_value"],
        month_col=fields["month_col"],
    )
    population_metrics = _population_metrics(typed_result)
    body = {
        "schema_version": STRATEGY_POOL_TYPED_VALIDATION_SCHEMA_VERSION,
        "producer_version": (
            STRATEGY_POOL_TYPED_VALIDATION_PRODUCER_VERSION
        ),
        "identity": {
            "pool_id": current_pool["pool_id"],
            "task_id": current_pool["task_id"],
            "strategy_type": strategy_type,
            "revision": current_pool["revision"],
            "revision_id": current_pool["revision_id"],
            "snapshot_hash": current_pool["snapshot_hash"],
            "design_hash": compiled["design_hash"],
            "strategy_spec_hash": strategy_spec_hash(
                compiled["strategy_spec"]
            ),
        },
        "source_bindings": {
            "pool_artifact": pool_artifact,
            "sample_design_v2": sample_v2,
            "dataset": dataset,
            "development_lineage": {
                "legacy_development_ref": development_ref,
                "sample_binding": sample_binding,
            },
            "target": target,
            "fields": fields,
        },
        "partition": partition,
        "population": "risk",
        "comparison_mode": "absolute",
        "lifecycle": _expected_lifecycle(partition),
        "population_metrics": population_metrics,
        "risk_summary": _risk_summary(typed_result),
        "typed_backtest": typed_result,
        "monthly": monthly,
        "conservation": {
            "typed_population_equals_membership_count": True,
            "typed_breakdown_equals_population": True,
            "monthly_rolls_to_overall": True,
            "selected_partition_equals_membership_count": True,
            "risk_partition_excludes_development": True,
        },
        "red_flags": _warning_flags(typed_result["warnings"]),
    }
    evidence_id = "strategy-pool-validation-" + _sha256(
        _canonical_json(body)
    )[:24]
    document = {**body, "evidence_id": evidence_id}
    document["content_hash"] = _sha256(_canonical_json(document))
    return validate_typed_strategy_pool_validation_evidence(document)


def validate_typed_strategy_pool_validation_evidence(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate typed metrics, exact lineage, hashes, and row conservation."""

    from marvis.packs.strategy.pool_validation import (
        _canonical_json,
        _dataset_binding,
        _development_lineage,
        _exact_fields,
        _expected_lifecycle,
        _field_bindings,
        _hash,
        _identity,
        _json_object,
        _pool_artifact_ref,
        _sample_design_v2_ref,
        _sha256,
        _target_binding,
        _text,
    )

    obj = _json_object(payload, "typed Strategy Pool validation evidence")
    _exact_fields(
        obj,
        _TOP_LEVEL_FIELDS,
        "typed Strategy Pool validation evidence",
    )
    if obj["schema_version"] != STRATEGY_POOL_TYPED_VALIDATION_SCHEMA_VERSION:
        raise StrategyError(
            "typed Strategy Pool validation schema_version is invalid"
        )
    if (
        obj["producer_version"]
        != STRATEGY_POOL_TYPED_VALIDATION_PRODUCER_VERSION
    ):
        raise StrategyError(
            "typed Strategy Pool validation producer_version is invalid"
        )
    evidence_id = _text(obj["evidence_id"], "evidence_id")
    if not evidence_id.startswith("strategy-pool-validation-"):
        raise StrategyError(
            "typed Strategy Pool validation evidence_id is invalid"
        )
    content_hash = _hash(obj["content_hash"], "content_hash")
    without_hash = {
        key: value for key, value in obj.items() if key != "content_hash"
    }
    if not hmac.compare_digest(
        content_hash,
        _sha256(_canonical_json(without_hash)),
    ):
        raise StrategyError(
            "typed Strategy Pool validation content_hash does not match content"
        )
    body = {
        key: value
        for key, value in without_hash.items()
        if key != "evidence_id"
    }
    expected_id = "strategy-pool-validation-" + _sha256(
        _canonical_json(body)
    )[:24]
    if not hmac.compare_digest(evidence_id, expected_id):
        raise StrategyError(
            "typed Strategy Pool validation evidence_id does not match content"
        )

    partition = obj["partition"]
    if partition not in {"validation", "oot"}:
        raise StrategyError(
            "typed Strategy Pool validation partition is invalid"
        )
    if obj["population"] != "risk" or obj["comparison_mode"] != "absolute":
        raise StrategyError(
            "typed Strategy Pool validation controls changed"
        )
    if obj["lifecycle"] != _expected_lifecycle(partition):
        raise StrategyError(
            "typed Strategy Pool validation lifecycle changed"
        )

    identity = _identity(obj["identity"])
    strategy_type = identity["strategy_type"]
    if strategy_type not in _TYPED_STRATEGY_TYPES:
        raise StrategyError(
            "typed Strategy Pool validation strategy_type is unsupported"
        )
    sources = _json_object(
        obj["source_bindings"],
        "typed Strategy Pool validation source_bindings",
    )
    _exact_fields(
        sources,
        {
            "pool_artifact",
            "sample_design_v2",
            "dataset",
            "development_lineage",
            "target",
            "fields",
        },
        "typed Strategy Pool validation source_bindings",
    )
    _pool_artifact_ref(sources["pool_artifact"])
    sample_v2 = _sample_design_v2_ref(
        sources["sample_design_v2"],
        partition=partition,
    )
    dataset = _dataset_binding(sources["dataset"])
    development = _development_lineage(sources["development_lineage"])
    target = _target_binding(sources["target"])
    fields = _field_bindings(
        sources["fields"],
        target_col=target["column"],
    )
    if identity["task_id"] != dataset["task_id"]:
        raise StrategyError(
            "typed Strategy Pool validation task and dataset disagree"
        )
    for field in (
        "dataset_id",
        "dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
    ):
        if development["sample_binding"][field] != dataset[field]:
            raise StrategyError(
                "typed Strategy Pool validation development and dataset "
                f"disagree on {field}"
            )

    result = StrategyBacktestResult.from_dict(obj["typed_backtest"])
    _require_typed_result_matches_sources(
        result,
        identity=identity,
        target=target,
    )
    population = _validate_population_metrics(
        obj["population_metrics"],
        result=result,
    )
    if obj["risk_summary"] != _risk_summary(result.to_dict()):
        raise StrategyError(
            "typed Strategy Pool validation risk summary changed"
        )
    risk_summary = _json_object(
        obj["risk_summary"],
        "typed Strategy Pool validation risk_summary",
    )
    _exact_fields(
        risk_summary,
        _RISK_SUMMARY_FIELDS,
        "typed Strategy Pool validation risk_summary",
    )
    if population["population_count"] != sample_v2["partition_count"]:
        raise StrategyError(
            "typed Strategy Pool validation population does not match "
            "V2 membership"
        )
    _validate_monthly(
        obj["monthly"],
        strategy_type=strategy_type,
        target=target,
        month_col=fields["month_col"],
        overall=result,
    )
    conservation = _json_object(
        obj["conservation"],
        "typed Strategy Pool validation conservation",
    )
    _exact_fields(
        conservation,
        _CONSERVATION_FIELDS,
        "typed Strategy Pool validation conservation",
    )
    if not all(value is True for value in conservation.values()):
        raise StrategyError(
            "typed Strategy Pool validation conservation checks must pass"
        )
    red_flags = obj["red_flags"]
    if not isinstance(red_flags, list):
        raise StrategyError(
            "typed Strategy Pool validation red_flags must be a list"
        )
    expected_flags = _warning_flags(list(result.warnings))
    if red_flags != expected_flags:
        raise StrategyError(
            "typed Strategy Pool validation red_flags changed"
        )
    for position, flag in enumerate(red_flags):
        normalized = _json_object(
            flag,
            f"typed Strategy Pool validation red_flags[{position}]",
        )
        _exact_fields(
            normalized,
            _RED_FLAG_FIELDS,
            f"typed Strategy Pool validation red_flags[{position}]",
        )
    return obj


def _build_monthly_evidence(
    frame: pd.DataFrame,
    *,
    spec: Mapping[str, Any],
    strategy_type: str,
    target_col: str,
    target_bad_value: int,
    month_col: str | None,
) -> dict[str, Any]:
    if month_col is None:
        return {"status": "unavailable", "column": None, "periods": []}
    if month_col not in frame.columns:
        raise StrategyError(
            "typed Strategy Pool validation month column is missing"
        )
    grouped: dict[str, tuple[Any, list[int]]] = {}
    for position, raw_value in enumerate(frame[month_col].tolist()):
        value = _json_scalar_or_null(raw_value, name=month_col)
        token = _canonical_json_value(value)
        if token not in grouped:
            grouped[token] = (value, [])
        grouped[token][1].append(position)
    periods = []
    for token in sorted(grouped):
        value, positions = grouped[token]
        selected = frame.iloc[positions].reset_index(drop=True)
        typed = run_typed_backtest(
            selected,
            spec,
            target_col=target_col,
            target_bad_value=target_bad_value,
        ).to_dict()
        if typed["strategy_type"] != strategy_type:
            raise StrategyError(
                "typed Strategy Pool monthly strategy type changed"
            )
        periods.append(
            {
                "value": value,
                "risk_summary": _risk_summary(typed),
                "typed_backtest": typed,
            }
        )
    return {"status": "available", "column": month_col, "periods": periods}


def _validate_monthly(
    value: object,
    *,
    strategy_type: str,
    target: Mapping[str, Any],
    month_col: str | None,
    overall: StrategyBacktestResult,
) -> None:
    from marvis.packs.strategy.pool_validation import (
        _exact_fields,
        _json_object,
    )

    monthly = _json_object(value, "typed Strategy Pool validation monthly")
    _exact_fields(
        monthly,
        _MONTHLY_FIELDS,
        "typed Strategy Pool validation monthly",
    )
    expected_status = "unavailable" if month_col is None else "available"
    if monthly["status"] != expected_status or monthly["column"] != month_col:
        raise StrategyError(
            "typed Strategy Pool validation monthly binding changed"
        )
    periods = monthly["periods"]
    if not isinstance(periods, list):
        raise StrategyError(
            "typed Strategy Pool validation monthly periods must be a list"
        )
    if month_col is None and periods:
        raise StrategyError(
            "typed Strategy Pool validation unavailable monthly must be empty"
        )
    if month_col is None:
        return
    tokens: list[str] = []
    population_count = labelled_count = 0
    for position, raw_period in enumerate(periods):
        period = _json_object(
            raw_period,
            f"typed Strategy Pool validation monthly.periods[{position}]",
        )
        _exact_fields(
            period,
            _MONTHLY_PERIOD_FIELDS,
            f"typed Strategy Pool validation monthly.periods[{position}]",
        )
        period_value = _json_scalar_or_null(
            period["value"],
            name=f"monthly.periods[{position}].value",
        )
        tokens.append(_canonical_json_value(period_value))
        result = StrategyBacktestResult.from_dict(period["typed_backtest"])
        if result.strategy_type != strategy_type:
            raise StrategyError(
                "typed Strategy Pool validation monthly strategy type changed"
            )
        risk_summary = _json_object(
            period["risk_summary"],
            (
                "typed Strategy Pool validation "
                f"monthly.periods[{position}].risk_summary"
            ),
        )
        _exact_fields(
            risk_summary,
            _RISK_SUMMARY_FIELDS,
            (
                "typed Strategy Pool validation "
                f"monthly.periods[{position}].risk_summary"
            ),
        )
        if risk_summary != _risk_summary(result.to_dict()):
            raise StrategyError(
                "typed Strategy Pool validation monthly risk summary changed"
            )
        _require_typed_result_target(result, target=target)
        if (
            result.normalized_input["strategy_effect_hash"]
            != overall.normalized_input["strategy_effect_hash"]
            or result.strategy_id != overall.strategy_id
        ):
            raise StrategyError(
                "typed Strategy Pool validation monthly strategy identity changed"
            )
        population_count += result.population_count
        labelled_count += result.labeled_count
    if tokens != sorted(set(tokens)):
        raise StrategyError(
            "typed Strategy Pool validation monthly values must be unique "
            "and sorted"
        )
    if (
        population_count != overall.population_count
        or labelled_count != overall.labeled_count
    ):
        raise StrategyError(
            "typed Strategy Pool validation monthly rows do not roll up"
        )


def _require_typed_result_matches_sources(
    result: StrategyBacktestResult,
    *,
    identity: Mapping[str, Any],
    target: Mapping[str, Any],
) -> None:
    if result.strategy_type != identity["strategy_type"]:
        raise StrategyError(
            "typed Strategy Pool validation result type changed"
        )
    if (
        result.normalized_input["strategy_effect_hash"]
        != identity["strategy_spec_hash"]
    ):
        raise StrategyError(
            "typed Strategy Pool validation strategy hash changed"
        )
    expected_strategy_id = (
        "strategy-" + identity["strategy_spec_hash"][:16]
    )
    if result.strategy_id != expected_strategy_id:
        raise StrategyError(
            "typed Strategy Pool validation strategy_id changed"
        )
    _require_typed_result_target(result, target=target)


def _require_typed_result_target(
    result: StrategyBacktestResult,
    *,
    target: Mapping[str, Any],
) -> None:
    normalized = result.normalized_input
    if (
        normalized["target_col"] != target["column"]
        or normalized["target_encoding"]
        != {"good": target["good_value"], "bad": target["bad_value"]}
    ):
        raise StrategyError(
            "typed Strategy Pool validation target binding changed"
        )


def _population_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    population_count = int(result["population_count"])
    labelled_count = int(result["labeled_count"])
    return {
        "population_count": population_count,
        "labelled_count": labelled_count,
        "unlabelled_count": population_count - labelled_count,
        "label_coverage": result["label_coverage"],
    }


def _risk_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    labelled_count = int(result["labeled_count"])
    bad_count = sum(
        int(row["bad_count"])
        for row in result["breakdown"]
    )
    return {
        "overall_bad_count": bad_count,
        "overall_bad_rate": (
            None if labelled_count == 0 else bad_count / labelled_count
        ),
    }


def _validate_population_metrics(
    value: object,
    *,
    result: StrategyBacktestResult,
) -> dict[str, Any]:
    from marvis.packs.strategy.pool_validation import (
        _exact_fields,
        _json_object,
    )

    population = _json_object(
        value,
        "typed Strategy Pool validation population_metrics",
    )
    _exact_fields(
        population,
        _POPULATION_FIELDS,
        "typed Strategy Pool validation population_metrics",
    )
    expected = _population_metrics(result.to_dict())
    if population != expected:
        raise StrategyError(
            "typed Strategy Pool validation population metrics changed"
        )
    return population


def _warning_flags(warnings: list[str]) -> list[dict[str, str]]:
    return [
        {
            "code": "typed_backtest_warning",
            "level": "amber",
            "message": warning,
        }
        for warning in warnings
    ]


def _json_scalar_or_null(value: object, *, name: str) -> Any:
    if value is None or _is_missing_scalar(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise StrategyError(f"{name} must be finite")
        return number
    if isinstance(value, str):
        if "\x00" in value:
            raise StrategyError(f"{name} contains a null byte")
        return value
    if isinstance(value, datetime | date | pd.Timestamp):
        return pd.Timestamp(value).isoformat()
    raise StrategyError(f"{name} must be a JSON scalar or null")


def _is_missing_scalar(value: object) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    try:
        return bool(missing)
    except (TypeError, ValueError):
        return False


def _canonical_json_value(value: object) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise StrategyError("typed validation value is not canonical JSON") from exc


__all__ = [
    "STRATEGY_POOL_TYPED_VALIDATION_PRODUCER_VERSION",
    "STRATEGY_POOL_TYPED_VALIDATION_SCHEMA_VERSION",
    "build_typed_strategy_pool_validation_evidence",
    "validate_typed_strategy_pool_validation_evidence",
]
