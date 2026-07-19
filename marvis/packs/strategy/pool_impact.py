"""Deterministic first-match impact evidence for one Strategy Pool snapshot.

The module is deliberately persistence-free.  It compiles the supplied Pool,
evaluates its canonical Strategy DSL once, and projects count, risk, amount,
waterfall, and optional monthly/baseline evidence into a self-authenticating
JSON document.  Tool boundaries own task/dataset lineage and artifact writes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import hmac
import json
import math
import re
from typing import Any

import numpy as np
import pandas as pd

from marvis.packs.strategy.dsl import parse_strategy_spec, strategy_spec_hash
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.evaluator import (
    evaluate_expression_frame,
    evaluate_strategy_frame,
)
from marvis.packs.strategy.pool import compile_strategy_pool, validate_strategy_pool
from marvis.validation.time_periods import month_key_series


STRATEGY_POOL_IMPACT_SCHEMA_VERSION = "strategy.impact-assessment.v1"
STRATEGY_POOL_IMPACT_PRODUCER_VERSION = "marvis.strategy.pool-impact/1"
MAX_IMPACT_ROWS = 2_000_000
MAX_IMPACT_RULES = 200
MAX_IMPACT_WORK = 50_000_000

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ASSESSMENT_ID_RE = re.compile(r"^strategy-impact-assessment-[0-9a-f]{24}$")
_SAMPLE_BINDING_FIELDS = frozenset(
    {
        "task_id",
        "dataset_id",
        "dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "sample_context_hash",
    }
)
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "assessment_id",
        "identity",
        "bindings",
        "lifecycle",
        "population",
        "overall",
        "waterfall",
        "default_unmatched",
        "monthly",
        "baseline",
        "conservation",
        "red_flags",
        "content_hash",
    }
)
_ACTION_NAMES = {"approval": "approve", "reject": "reject", "review": "review"}
_ACTION_ORDER = ("approve", "reject", "review")


def build_strategy_pool_impact_assessment(
    *,
    pool: Mapping[str, Any],
    frame: pd.DataFrame,
    sample_binding: Mapping[str, Any],
    target_col: str,
    month_col: str | None = None,
    loan_amount_col: str | None = None,
    overdue_amount_col: str | None = None,
    comparison_mode: str = "absolute",
    baseline_spec: Mapping[str, Any] | None = None,
    baseline_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build immutable impact evidence without persistence or lifecycle effects."""

    current_pool = validate_strategy_pool(pool)
    if current_pool["strategy_type"] not in {"approval", "reject"}:
        raise StrategyError(
            "Pool impact v1 supports approval/reject only; other typed impact "
            "semantics remain explicit V2 work"
        )
    selected = compile_strategy_pool(current_pool)
    if not current_pool["entries"]:
        raise StrategyError("cannot measure an empty Strategy Pool")
    normalized_sample = _sample_binding(sample_binding)
    if normalized_sample["task_id"] != current_pool["task_id"]:
        raise StrategyError("sample binding belongs to another task")
    expected_evidence = {
        key: normalized_sample[key]
        for key in _SAMPLE_BINDING_FIELDS
        if key != "task_id"
    }
    for entry in current_pool["entries"]:
        if entry["source"]["evidence_identity"] != expected_evidence:
            raise StrategyError("Pool entries do not match the supplied sample binding")

    working, columns = _working_frame(
        frame,
        target_col=target_col,
        month_col=month_col,
        loan_amount_col=loan_amount_col,
        overdue_amount_col=overdue_amount_col,
        rule_count=len(current_pool["entries"]),
    )
    target = _target_series(working, columns["target_col"])
    amounts = _amount_series(
        working,
        loan_amount_col=columns["loan_amount_col"],
        overdue_amount_col=columns["overdue_amount_col"],
    )
    periods = _period_series(working, columns["month_col"])

    spec = parse_strategy_spec(selected["strategy_spec"])
    evaluation = evaluate_strategy_frame(working, spec)
    actions = _action_series(evaluation.action_type)
    matched_rule_ids = evaluation.matched_rule_id.reset_index(drop=True)
    all_mask = pd.Series(True, index=working.index, dtype=bool)
    overall_effect = _effect_slice(all_mask, target=target, amounts=amounts)
    overall_actions = _action_summary(actions, all_mask, target=target)

    waterfall: list[dict[str, Any]] = []
    claimed = pd.Series(False, index=working.index, dtype=bool)
    incremental_masks: dict[str, pd.Series] = {}
    for position, (entry, rule) in enumerate(
        zip(current_pool["entries"], spec.rules, strict=True), start=1
    ):
        standalone = evaluate_expression_frame(working, rule.condition).reset_index(
            drop=True
        )
        incremental = matched_rule_ids.eq(rule.rule_id)
        shadowed = standalone & claimed
        if not (standalone == (incremental | shadowed)).all():
            raise StrategyError(
                f"waterfall conservation failed for rule_id {rule.rule_id}"
            )
        if bool((incremental & shadowed).any()):
            raise StrategyError(f"waterfall masks overlap for rule_id {rule.rule_id}")
        claimed |= standalone
        remaining = ~claimed
        incremental_masks[rule.rule_id] = incremental
        waterfall.append(
            {
                "position": position,
                "entry_id": entry["entry_id"],
                "rule_id": rule.rule_id,
                "source_ref": {
                    "artifact_id": entry["source"]["artifact_id"],
                    "artifact_content_hash": entry["source"][
                        "artifact_content_hash"
                    ],
                    "asset_id": entry["source"]["asset_id"],
                    "asset_hash": entry["source"]["asset_hash"],
                    "fragment_id": entry["source"]["fragment_id"],
                },
                "action": rule.action.to_dict(),
                "standalone": _effect_slice(
                    standalone, target=target, amounts=amounts
                ),
                "incremental": _effect_slice(
                    incremental, target=target, amounts=amounts
                ),
                "shadowed": _effect_slice(shadowed, target=target, amounts=amounts),
                "remaining_after": _effect_slice(
                    remaining, target=target, amounts=amounts
                ),
            }
        )

    unmatched = matched_rule_ids.isna()
    default_unmatched = {
        "action": spec.default_action.to_dict(),
        "effect": _effect_slice(unmatched, target=target, amounts=amounts),
    }
    _require_population_conservation(
        len(working),
        waterfall=waterfall,
        unmatched_count=int(unmatched.sum()),
    )

    baseline = _baseline_evidence(
        mode=comparison_mode,
        baseline_spec=baseline_spec,
        baseline_binding=baseline_binding,
        strategy_type=current_pool["strategy_type"],
        frame=working,
        target=target,
        periods=periods,
        current_actions=actions,
    )
    monthly = _monthly_evidence(
        periods=periods,
        target=target,
        amounts=amounts,
        actions=actions,
        incremental_masks=incremental_masks,
        baseline_actions=baseline.pop("_actions", None),
    )
    _require_monthly_rollup(
        monthly,
        overall_effect=overall_effect,
        overall_actions=overall_actions,
        waterfall=waterfall,
    )
    if baseline["status"] == "available":
        baseline["monthly"] = monthly.get("baseline", {"status": "unavailable"})
    monthly.pop("baseline", None)

    red_flags = _red_flags(
        population_count=len(working),
        labelled_count=int(target.notna().sum()),
        month_col=columns["month_col"],
        amounts=amounts,
        waterfall=waterfall,
    )
    body = {
        "schema_version": STRATEGY_POOL_IMPACT_SCHEMA_VERSION,
        "producer_version": STRATEGY_POOL_IMPACT_PRODUCER_VERSION,
        "identity": {
            "pool_id": current_pool["pool_id"],
            "task_id": current_pool["task_id"],
            "strategy_type": current_pool["strategy_type"],
            "revision": current_pool["revision"],
            "revision_id": current_pool["revision_id"],
            "snapshot_hash": current_pool["snapshot_hash"],
            "design_hash": selected["design_hash"],
            "strategy_spec_hash": strategy_spec_hash(spec),
        },
        "bindings": {
            "sample": normalized_sample,
            **columns,
            "comparison_mode": comparison_mode,
        },
        "lifecycle": {
            "candidate_stage": "development",
            "observation_stage": "backtested",
            "validation_status": "unvalidated",
            "creates_strategy": False,
            "adopted": False,
            "deployed": False,
        },
        "population": {
            "population_count": len(working),
            "labelled_count": int(target.notna().sum()),
            "unlabelled_count": int(target.isna().sum()),
            "label_coverage": _ratio(int(target.notna().sum()), len(working)),
        },
        "overall": {
            "effect": overall_effect,
            "actions": overall_actions,
        },
        "waterfall": waterfall,
        "default_unmatched": default_unmatched,
        "monthly": monthly,
        "baseline": baseline,
        "conservation": {
            "standalone_equals_incremental_plus_shadowed": True,
            "incremental_plus_default_equals_population": True,
            "monthly_rolls_to_overall": monthly["status"] != "available" or True,
        },
        "red_flags": red_flags,
    }
    assessment_id = "strategy-impact-assessment-" + _sha256(_canonical_json(body))[:24]
    document = {**body, "assessment_id": assessment_id}
    document["content_hash"] = _sha256(_canonical_json(document))
    return validate_strategy_pool_impact_assessment(document)


def validate_strategy_pool_impact_assessment(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact top-level contract and its self-authenticating hashes."""

    if not isinstance(payload, Mapping):
        raise StrategyError("Strategy Pool impact assessment must be an object")
    if set(payload) != _TOP_LEVEL_FIELDS:
        unexpected = sorted(set(payload) - _TOP_LEVEL_FIELDS)
        missing = sorted(_TOP_LEVEL_FIELDS - set(payload))
        detail = []
        if unexpected:
            detail.append("unsupported fields: " + ", ".join(unexpected))
        if missing:
            detail.append("missing fields: " + ", ".join(missing))
        raise StrategyError("impact assessment " + "; ".join(detail))
    normalized = _json_value(dict(payload), name="impact assessment")
    if normalized["schema_version"] != STRATEGY_POOL_IMPACT_SCHEMA_VERSION:
        raise StrategyError("impact assessment schema_version is invalid")
    if normalized["producer_version"] != STRATEGY_POOL_IMPACT_PRODUCER_VERSION:
        raise StrategyError("impact assessment producer_version is invalid")
    assessment_id = normalized["assessment_id"]
    if not isinstance(assessment_id, str) or _ASSESSMENT_ID_RE.fullmatch(
        assessment_id
    ) is None:
        raise StrategyError("impact assessment_id is invalid")
    content_hash = normalized["content_hash"]
    if not isinstance(content_hash, str) or _HASH_RE.fullmatch(content_hash) is None:
        raise StrategyError("impact content_hash is invalid")
    without_hash = {key: value for key, value in normalized.items() if key != "content_hash"}
    if not hmac.compare_digest(content_hash, _sha256(_canonical_json(without_hash))):
        raise StrategyError("impact content_hash does not match the document")
    without_identity = {
        key: value
        for key, value in without_hash.items()
        if key != "assessment_id"
    }
    expected_id = "strategy-impact-assessment-" + _sha256(
        _canonical_json(without_identity)
    )[:24]
    if not hmac.compare_digest(assessment_id, expected_id):
        raise StrategyError("impact assessment_id does not match the document")
    lifecycle = normalized.get("lifecycle")
    expected_lifecycle = {
        "candidate_stage": "development",
        "observation_stage": "backtested",
        "validation_status": "unvalidated",
        "creates_strategy": False,
        "adopted": False,
        "deployed": False,
    }
    if lifecycle != expected_lifecycle:
        raise StrategyError("impact lifecycle must remain development/unvalidated")
    conservation = normalized.get("conservation")
    if not isinstance(conservation, dict) or not all(
        value is True for value in conservation.values()
    ):
        raise StrategyError("impact conservation checks must all pass")
    return normalized


def canonical_strategy_pool_impact_json(payload: Mapping[str, Any]) -> str:
    """Return byte-stable canonical JSON for one validated assessment."""

    return _canonical_json(validate_strategy_pool_impact_assessment(payload))


def _working_frame(
    frame: pd.DataFrame,
    *,
    target_col: str,
    month_col: str | None,
    loan_amount_col: str | None,
    overdue_amount_col: str | None,
    rule_count: int,
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    if not isinstance(frame, pd.DataFrame):
        raise StrategyError("Pool impact rows must be a DataFrame")
    if frame.empty:
        raise StrategyError("Pool impact source dataset must not be empty")
    if len(frame) > MAX_IMPACT_ROWS or rule_count > MAX_IMPACT_RULES:
        raise StrategyError("Pool impact exceeds the row or rule budget")
    if len(frame) * rule_count > MAX_IMPACT_WORK:
        raise StrategyError("Pool impact exceeds the rows-by-rules work budget")
    if frame.columns.duplicated().any():
        raise StrategyError("Pool impact source dataset has duplicate columns")
    columns = {
        "target_col": _column(target_col, "target_col"),
        "month_col": _optional_column(month_col, "month_col"),
        "loan_amount_col": _optional_column(loan_amount_col, "loan_amount_col"),
        "overdue_amount_col": _optional_column(
            overdue_amount_col, "overdue_amount_col"
        ),
    }
    selected = [value for value in columns.values() if value is not None]
    if len(selected) != len(set(selected)):
        raise StrategyError("Pool impact column bindings must be distinct")
    missing = sorted(set(selected) - set(frame.columns))
    if missing:
        raise StrategyError("Pool impact source is missing columns: " + ", ".join(missing))
    return frame.reset_index(drop=True), columns


def _sample_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _SAMPLE_BINDING_FIELDS:
        raise StrategyError("sample_binding must contain the exact governed fields")
    result = {
        "task_id": _text(value["task_id"], "sample task_id"),
        "dataset_id": _text(value["dataset_id"], "sample dataset_id"),
        "dataset_content_hash": _hash(
            value["dataset_content_hash"], "sample dataset_content_hash"
        ),
        "workspace_revision": _non_negative_int(
            value["workspace_revision"], "sample workspace_revision"
        ),
        "workspace_generation": _non_negative_int(
            value["workspace_generation"], "sample workspace_generation"
        ),
        "semantic_mapping_hash": _hash(
            value["semantic_mapping_hash"], "sample semantic_mapping_hash"
        ),
        "sample_context_hash": _hash(
            value["sample_context_hash"], "sample sample_context_hash"
        ),
    }
    return result


def _target_series(frame: pd.DataFrame, column: str) -> pd.Series:
    raw = frame[column]
    missing = raw.isna()
    numeric = pd.to_numeric(raw, errors="coerce")
    invalid = (~missing) & (~np.isfinite(numeric))
    if bool(invalid.any()) or bool((~numeric.loc[~missing].isin([0, 1])).any()):
        raise StrategyError("target must contain only 0, 1, or missing")
    return numeric.astype(float).reset_index(drop=True)


def _amount_series(
    frame: pd.DataFrame,
    *,
    loan_amount_col: str | None,
    overdue_amount_col: str | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, column in (
        ("loan_amount", loan_amount_col),
        ("overdue_amount", overdue_amount_col),
    ):
        if column is None:
            result[key] = {"status": "unavailable", "column": None, "values": None}
            continue
        raw = frame[column]
        numeric = pd.to_numeric(raw, errors="coerce")
        invalid = raw.notna() & (~np.isfinite(numeric))
        if bool(invalid.any()) or bool((numeric.dropna() < 0).any()):
            raise StrategyError(
                f"{column} must contain non-negative finite numbers or missing"
            )
        result[key] = {
            "status": "available",
            "column": column,
            "values": numeric.astype(float).reset_index(drop=True),
        }
    return result


def _period_series(frame: pd.DataFrame, column: str | None) -> pd.Series | None:
    if column is None:
        return None
    try:
        return month_key_series(frame[column], column_name=column).reset_index(drop=True)
    except ValueError as exc:
        raise StrategyError(str(exc)) from exc


def _action_series(values: pd.Series) -> pd.Series:
    normalized = values.reset_index(drop=True).map(_ACTION_NAMES)
    if bool(normalized.isna().any()):
        raise StrategyError("approval/reject Pool produced an unsupported action")
    return normalized.astype("object")


def _effect_slice(
    mask: pd.Series,
    *,
    target: pd.Series,
    amounts: Mapping[str, Any],
) -> dict[str, Any]:
    selected = mask.reset_index(drop=True).astype(bool)
    population_count = int(selected.sum())
    labelled = selected & target.notna()
    labelled_count = int(labelled.sum())
    bad_count = int(target.loc[labelled].eq(1).sum())
    return {
        "population_count": population_count,
        "population_share": _ratio(population_count, len(selected)),
        "labelled_count": labelled_count,
        "label_coverage": _ratio(labelled_count, population_count),
        "bad_count": bad_count,
        "bad_rate": _ratio(bad_count, labelled_count),
        "amounts": _amount_observations(selected, amounts=amounts),
    }


def _amount_observations(
    mask: pd.Series, *, amounts: Mapping[str, Any]
) -> dict[str, Any]:
    population_count = int(mask.sum())
    output: dict[str, Any] = {}
    for key in ("loan_amount", "overdue_amount"):
        item = amounts[key]
        values = item["values"]
        if values is None:
            output[key] = {
                "status": "unavailable",
                "column": None,
                "coverage_count": None,
                "coverage_rate": None,
                "sum": None,
            }
            continue
        covered = mask & values.notna()
        count = int(covered.sum())
        output[key] = {
            "status": "available",
            "column": item["column"],
            "coverage_count": count,
            "coverage_rate": _ratio(count, population_count),
            "sum": float(values.loc[covered].sum()),
        }
    loan = amounts["loan_amount"]["values"]
    overdue = amounts["overdue_amount"]["values"]
    if loan is None or overdue is None:
        output["paired"] = {
            "status": "unavailable",
            "coverage_count": None,
            "coverage_rate": None,
            "loan_amount_sum": None,
            "overdue_amount_sum": None,
            "overdue_rate": None,
        }
    else:
        paired = mask & loan.notna() & overdue.notna()
        count = int(paired.sum())
        loan_sum = float(loan.loc[paired].sum())
        overdue_sum = float(overdue.loc[paired].sum())
        output["paired"] = {
            "status": "available",
            "coverage_count": count,
            "coverage_rate": _ratio(count, population_count),
            "loan_amount_sum": loan_sum,
            "overdue_amount_sum": overdue_sum,
            "overdue_rate": _ratio(overdue_sum, loan_sum),
        }
    return output


def _action_summary(
    actions: pd.Series,
    mask: pd.Series,
    *,
    target: pd.Series,
) -> dict[str, Any]:
    population_count = int(mask.sum())
    rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    for action in _ACTION_ORDER:
        selected = mask & actions.eq(action)
        labelled = selected & target.notna()
        count = int(selected.sum())
        labelled_count = int(labelled.sum())
        bad_count = int(target.loc[labelled].eq(1).sum())
        row = {
            "action": action,
            "count": count,
            "rate": _ratio(count, population_count),
            "labelled_count": labelled_count,
            "bad_count": bad_count,
            "bad_rate": _ratio(bad_count, labelled_count),
        }
        rows.append(row)
        for key in ("count", "rate", "labelled_count", "bad_count", "bad_rate"):
            metrics[f"{action}_{key}"] = row[key]
    labelled = mask & target.notna()
    metrics["overall_bad_count"] = int(target.loc[labelled].eq(1).sum())
    metrics["overall_bad_rate"] = _ratio(
        metrics["overall_bad_count"], int(labelled.sum())
    )
    rejected_labelled = mask & actions.eq("reject") & target.notna()
    rejected_bad = int(target.loc[rejected_labelled].eq(1).sum())
    rejected_good = int(target.loc[rejected_labelled].eq(0).sum())
    total_good = int(target.loc[labelled].eq(0).sum())
    metrics["bad_capture_rate"] = _ratio(
        rejected_bad, metrics["overall_bad_count"]
    )
    metrics["good_reject_rate"] = _ratio(rejected_good, total_good)
    return {"metrics": metrics, "breakdown": rows}


def _baseline_evidence(
    *,
    mode: str,
    baseline_spec: Mapping[str, Any] | None,
    baseline_binding: Mapping[str, Any] | None,
    strategy_type: str,
    frame: pd.DataFrame,
    target: pd.Series,
    periods: pd.Series | None,
    current_actions: pd.Series,
) -> dict[str, Any]:
    if mode not in {"absolute", "vs_baseline"}:
        raise StrategyError("comparison_mode must be absolute or vs_baseline")
    if mode == "absolute":
        if baseline_spec is not None or baseline_binding is not None:
            raise StrategyError("absolute comparison must not provide a baseline")
        return {"status": "not_requested", "binding": None, "overall": None}
    if baseline_spec is None or baseline_binding is None:
        raise StrategyError("vs_baseline requires baseline_spec and baseline_binding")
    if not isinstance(baseline_binding, Mapping) or set(baseline_binding) != {
        "strategy_id",
        "strategy_type",
        "spec_hash",
    }:
        raise StrategyError("baseline_binding must contain exact strategy fields")
    parsed = parse_strategy_spec(baseline_spec)
    binding = {
        "strategy_id": _text(baseline_binding["strategy_id"], "baseline strategy_id"),
        "strategy_type": _text(
            baseline_binding["strategy_type"], "baseline strategy_type"
        ),
        "spec_hash": _hash(baseline_binding["spec_hash"], "baseline spec_hash"),
    }
    if parsed.strategy_type != strategy_type or binding["strategy_type"] != strategy_type:
        raise StrategyError("baseline strategy type must match the Pool")
    computed_hash = strategy_spec_hash(parsed)
    if not hmac.compare_digest(binding["spec_hash"], computed_hash):
        raise StrategyError("baseline spec_hash does not match baseline_spec")
    baseline_eval = evaluate_strategy_frame(frame, parsed)
    baseline_actions = _action_series(baseline_eval.action_type)
    all_mask = pd.Series(True, index=frame.index, dtype=bool)
    current_summary = _action_summary(current_actions, all_mask, target=target)
    baseline_summary = _action_summary(baseline_actions, all_mask, target=target)
    return {
        "status": "available",
        "binding": binding,
        "overall": {
            "current": current_summary,
            "baseline": baseline_summary,
            "metric_deltas": _metric_deltas(
                current_summary["metrics"], baseline_summary["metrics"]
            ),
        },
        "_actions": baseline_actions,
    }


def _monthly_evidence(
    *,
    periods: pd.Series | None,
    target: pd.Series,
    amounts: Mapping[str, Any],
    actions: pd.Series,
    incremental_masks: Mapping[str, pd.Series],
    baseline_actions: pd.Series | None,
) -> dict[str, Any]:
    if periods is None:
        return {
            "status": "unavailable",
            "reason": "month_column_not_provided",
            "periods": [],
        }
    rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    for period in sorted(str(value) for value in periods.unique()):
        mask = periods.eq(period)
        action_summary = _action_summary(actions, mask, target=target)
        row = {
            "period": period,
            "effect": _effect_slice(mask, target=target, amounts=amounts),
            "actions": action_summary,
            "rule_incremental": [
                {
                    "rule_id": rule_id,
                    "effect": _effect_slice(
                        mask & incremental, target=target, amounts=amounts
                    ),
                }
                for rule_id, incremental in incremental_masks.items()
            ],
        }
        rows.append(row)
        if baseline_actions is not None:
            baseline_summary = _action_summary(baseline_actions, mask, target=target)
            baseline_rows.append(
                {
                    "period": period,
                    "current": action_summary,
                    "baseline": baseline_summary,
                    "metric_deltas": _metric_deltas(
                        action_summary["metrics"], baseline_summary["metrics"]
                    ),
                }
            )
    result: dict[str, Any] = {
        "status": "available",
        "reason": None,
        "periods": rows,
    }
    if baseline_actions is not None:
        result["baseline"] = {"status": "available", "periods": baseline_rows}
    return result


def _require_population_conservation(
    population_count: int,
    *,
    waterfall: Sequence[Mapping[str, Any]],
    unmatched_count: int,
) -> None:
    incremental = sum(int(row["incremental"]["population_count"]) for row in waterfall)
    if incremental + unmatched_count != population_count:
        raise StrategyError("incremental Pool hits plus default do not cover population")
    for row in waterfall:
        if int(row["standalone"]["population_count"]) != int(
            row["incremental"]["population_count"]
        ) + int(row["shadowed"]["population_count"]):
            raise StrategyError("waterfall standalone count is inconsistent")


def _require_monthly_rollup(
    monthly: Mapping[str, Any],
    *,
    overall_effect: Mapping[str, Any],
    overall_actions: Mapping[str, Any],
    waterfall: Sequence[Mapping[str, Any]],
) -> None:
    if monthly["status"] != "available":
        return
    rows = monthly["periods"]
    if sum(row["effect"]["population_count"] for row in rows) != overall_effect[
        "population_count"
    ]:
        raise StrategyError("monthly population does not roll to overall")
    if sum(row["effect"]["labelled_count"] for row in rows) != overall_effect[
        "labelled_count"
    ] or sum(row["effect"]["bad_count"] for row in rows) != overall_effect["bad_count"]:
        raise StrategyError("monthly label counts do not roll to overall")
    for key, expected in overall_actions["metrics"].items():
        if key.endswith("_count"):
            actual = sum(row["actions"]["metrics"][key] for row in rows)
            if actual != expected:
                raise StrategyError(f"monthly {key} does not roll to overall")
    for position, rule in enumerate(waterfall):
        actual = sum(
            row["rule_incremental"][position]["effect"]["population_count"]
            for row in rows
        )
        if actual != rule["incremental"]["population_count"]:
            raise StrategyError("monthly rule hits do not roll to overall")
    for amount_key in ("loan_amount", "overdue_amount"):
        expected = overall_effect["amounts"][amount_key]["sum"]
        if expected is None:
            continue
        actual = sum(row["effect"]["amounts"][amount_key]["sum"] for row in rows)
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9):
            raise StrategyError(f"monthly {amount_key} does not roll to overall")


def _metric_deltas(
    current: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, float | int | None]:
    if set(current) != set(baseline):
        raise StrategyError("baseline metric keys differ from current metrics")
    deltas: dict[str, float | int | None] = {}
    for key in current:
        left = current[key]
        right = baseline[key]
        if left is None or right is None:
            deltas[key] = None
        elif isinstance(left, int) and isinstance(right, int):
            deltas[key] = left - right
        else:
            deltas[key] = float(left) - float(right)
    return deltas


def _red_flags(
    *,
    population_count: int,
    labelled_count: int,
    month_col: str | None,
    amounts: Mapping[str, Any],
    waterfall: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []
    if labelled_count < population_count:
        flags.append(
            {
                "code": "incomplete_label_coverage",
                "level": "amber",
                "message": (
                    f"{population_count - labelled_count} population rows are excluded "
                    "from risk denominators"
                ),
            }
        )
    if month_col is None:
        flags.append(
            {
                "code": "monthly_unavailable",
                "level": "info",
                "message": "month column was not provided; monthly evidence is unavailable",
            }
        )
    for key in ("loan_amount", "overdue_amount"):
        if amounts[key]["status"] == "unavailable":
            flags.append(
                {
                    "code": f"{key}_unavailable",
                    "level": "info",
                    "message": f"{key} column was not provided",
                }
            )
    for row in waterfall:
        if row["incremental"]["population_count"] == 0:
            flags.append(
                {
                    "code": "rule_fully_shadowed",
                    "level": "amber",
                    "message": f"rule {row['rule_id']} has zero first-match reach",
                }
            )
    return flags


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _column(value: object, name: str) -> str:
    return _text(value, name)


def _optional_column(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _column(value, name)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise StrategyError(f"{name} must be a non-empty string")
    return value.strip()


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise StrategyError(f"{name} must be a lowercase SHA-256 hash")
    return value


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StrategyError(f"{name} must be a non-negative integer")
    return value


def _json_value(value: Any, *, name: str) -> Any:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError, OverflowError) as exc:
        raise StrategyError(f"{name} must be canonical JSON data") from exc
    if decoded != value:
        raise StrategyError(f"{name} contains non-canonical JSON values")
    return decoded


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: str | bytes) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "MAX_IMPACT_ROWS",
    "MAX_IMPACT_RULES",
    "MAX_IMPACT_WORK",
    "STRATEGY_POOL_IMPACT_PRODUCER_VERSION",
    "STRATEGY_POOL_IMPACT_SCHEMA_VERSION",
    "build_strategy_pool_impact_assessment",
    "canonical_strategy_pool_impact_json",
    "validate_strategy_pool_impact_assessment",
]
