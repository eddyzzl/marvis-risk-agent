"""Deterministic first-match impact evidence for one Strategy Pool snapshot.

The module is deliberately persistence-free.  It compiles the supplied Pool,
evaluates its canonical Strategy DSL once, and projects count, risk, amount,
waterfall, and optional monthly/baseline evidence into a canonical,
content-addressed JSON document. Tool boundaries own task/dataset lineage and
artifact writes. Embedded hashes detect drift against a trusted expected hash;
they are not signatures and do not independently prove source-row semantics
after a caller deliberately reauthors and rehashes a document.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
import hashlib
import hmac
import json
import math
from numbers import Real
import re
from typing import Any

import numpy as np
import pandas as pd

from marvis.packs.strategy.dsl import (
    StrategyAction,
    parse_strategy_spec,
    strategy_spec_hash,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.evaluator import (
    evaluate_expression_frame,
    evaluate_strategy_frame,
)
from marvis.packs.strategy.pool import compile_strategy_pool, validate_strategy_pool
from marvis.validation.time_periods import month_key_series


STRATEGY_POOL_IMPACT_SCHEMA_VERSION = "strategy.impact-assessment.v2"
STRATEGY_POOL_IMPACT_PRODUCER_VERSION = "marvis.strategy.pool-impact/2"
MAX_IMPACT_ROWS = 2_000_000
MAX_IMPACT_RULES = 200
MAX_IMPACT_WORK = 50_000_000
MAX_IMPACT_MONTHS = 240
MAX_IMPACT_MONTHLY_WORK = 50_000_000
MAX_IMPACT_EXPRESSION_NODES = 10_000
MAX_IMPACT_EXPRESSION_DEPTH = 64

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ASSESSMENT_ID_RE = re.compile(r"^strategy-impact-assessment-[0-9a-f]{24}$")
_SAMPLE_DESIGN_ID_RE = re.compile(r"^strategy-sample-design-[0-9a-f]{24}$")
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
_SAMPLE_DESIGN_REF_FIELDS = frozenset(
    {
        "artifact_id",
        "artifact_content_hash",
        "sample_design_id",
        "sample_design_content_hash",
        "partition",
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
    sample_design_ref: Mapping[str, Any],
    target_col: str,
    target_bad_value: int,
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
            "Pool impact supports approval/reject only; other typed impact "
            "semantics remain explicit V2 work"
        )
    selected = compile_strategy_pool(current_pool)
    if not current_pool["entries"]:
        raise StrategyError("cannot measure an empty Strategy Pool")
    normalized_sample = _sample_binding(sample_binding)
    normalized_sample_design_ref = _sample_design_ref(sample_design_ref)
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
    target = _target_series(
        working,
        columns["target_col"],
        target_bad_value=target_bad_value,
    )
    amounts = _amount_series(
        working,
        loan_amount_col=columns["loan_amount_col"],
        overdue_amount_col=columns["overdue_amount_col"],
    )
    periods = _period_series(working, columns["month_col"])
    _require_monthly_budget(
        row_count=len(working),
        rule_count=len(current_pool["entries"]),
        periods=periods,
    )

    _preflight_strategy_expression_limits(
        selected["strategy_spec"], name="current Pool"
    )
    spec = parse_strategy_spec(selected["strategy_spec"])
    current_expression_nodes, current_expression_cost = _strategy_expression_budget(
        spec.rules,
        name="current Pool",
    )
    _require_evaluation_work_budget(
        row_count=len(working),
        current_expression_cost=current_expression_cost,
        baseline_expression_cost=0,
    )
    evaluation = evaluate_strategy_frame(working, spec)
    actions = _action_series(evaluation.action_type)
    matched_rule_ids = evaluation.matched_rule_id.reset_index(drop=True)
    all_mask = pd.Series(True, index=working.index, dtype=bool)
    overall_effect = _effect_slice(all_mask, target=target, amounts=amounts)
    overall_actions = _action_summary(
        actions,
        all_mask,
        target=target,
        amounts=amounts,
    )

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
                    "sample_design_ref": normalized_sample_design_ref,
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
        amounts=amounts,
        current_actions=actions,
        current_rule_count=len(current_pool["entries"]),
        current_expression_nodes=current_expression_nodes,
        current_expression_cost=current_expression_cost,
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
            "sample_design_ref": normalized_sample_design_ref,
            **columns,
            "target_bad_value": target_bad_value,
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
            # Reaching document construction means _require_monthly_rollup
            # either proved every supplied period rolls up or no month was bound.
            "monthly_rolls_to_overall": True,
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
    """Validate canonical shape, hashes, derived fields, and conservation.

    Source authenticity belongs to the Tool/TaskArtifact boundary: callers loading
    persisted evidence must first compare its bytes with the registry's trusted
    content hash. Without the original frame/spec or an external signature, a
    standalone aggregate validator cannot prove that coherently reauthored and
    rehashed results came from the bound source rows.
    """

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
    _validate_impact_document(normalized)
    return normalized


def canonical_strategy_pool_impact_json(payload: Mapping[str, Any]) -> str:
    """Return byte-stable canonical JSON for one validated assessment."""

    return _canonical_json(validate_strategy_pool_impact_assessment(payload))


def _validate_impact_document(document: Mapping[str, Any]) -> None:
    """Validate nested artifact structure and the main conservation relations."""

    identity = _exact_object(
        document["identity"],
        {
            "pool_id",
            "task_id",
            "strategy_type",
            "revision",
            "revision_id",
            "snapshot_hash",
            "design_hash",
            "strategy_spec_hash",
        },
        "impact identity",
    )
    for field in ("pool_id", "task_id", "revision_id"):
        _text(identity[field], f"impact identity {field}")
    if identity["strategy_type"] not in {"approval", "reject"}:
        raise StrategyError("impact identity strategy_type is invalid")
    _positive_int(identity["revision"], "impact identity revision")
    for field in ("snapshot_hash", "design_hash", "strategy_spec_hash"):
        _hash(identity[field], f"impact identity {field}")

    bindings = _exact_object(
        document["bindings"],
        {
            "sample",
            "sample_design_ref",
            "target_col",
            "target_bad_value",
            "month_col",
            "loan_amount_col",
            "overdue_amount_col",
            "comparison_mode",
        },
        "impact bindings",
    )
    sample = _sample_binding(bindings["sample"])
    sample_design_ref = _sample_design_ref(bindings["sample_design_ref"])
    if sample["task_id"] != identity["task_id"]:
        raise StrategyError("impact sample task does not match identity")
    target_col = _text(bindings["target_col"], "impact target_col")
    _target_bad_value(bindings["target_bad_value"])
    bound_columns = [target_col]
    for field in ("month_col", "loan_amount_col", "overdue_amount_col"):
        value = bindings[field]
        if value is not None:
            bound_columns.append(_text(value, f"impact {field}"))
    if len(bound_columns) != len(set(bound_columns)):
        raise StrategyError("impact column bindings must be distinct")
    if bindings["comparison_mode"] not in {"absolute", "vs_baseline"}:
        raise StrategyError("impact comparison_mode is invalid")

    population = _validate_population(document["population"])
    population_count = population["population_count"]
    overall = _exact_object(
        document["overall"], {"effect", "actions"}, "impact overall"
    )
    overall_effect = _validate_effect_slice(
        overall["effect"], population_count, "impact overall effect"
    )
    amount_bindings = {
        "loan_amount": bindings["loan_amount_col"],
        "overdue_amount": bindings["overdue_amount_col"],
    }
    _require_effect_amount_bindings(
        overall_effect, amount_bindings, "impact overall effect"
    )
    if (
        overall_effect["population_count"] != population_count
        or overall_effect["labelled_count"] != population["labelled_count"]
    ):
        raise StrategyError("impact overall effect does not match population")
    overall_actions = _validate_action_summary(
        overall["actions"],
        population_count,
        "impact overall actions",
        amount_bindings=amount_bindings,
        total_effect=overall_effect,
    )
    action_rows = overall_actions["breakdown"]
    if (
        sum(row["labelled_count"] for row in action_rows)
        != overall_effect["labelled_count"]
        or sum(row["bad_count"] for row in action_rows)
        != overall_effect["bad_count"]
    ):
        raise StrategyError("impact overall action risk does not match population")

    waterfall = document["waterfall"]
    if not isinstance(waterfall, list) or not waterfall:
        raise StrategyError("impact waterfall must be a non-empty list")
    validated_waterfall: list[dict[str, Any]] = []
    waterfall_actions: list[str] = []
    seen_rule_ids: set[str] = set()
    seen_entry_ids: set[str] = set()
    for expected_position, raw in enumerate(waterfall, start=1):
        row = _exact_object(
            raw,
            {
                "position",
                "entry_id",
                "rule_id",
                "source_ref",
                "action",
                "standalone",
                "incremental",
                "shadowed",
                "remaining_after",
            },
            "impact waterfall row",
        )
        if row["position"] != expected_position:
            raise StrategyError("impact waterfall positions must be consecutive")
        entry_id = _text(row["entry_id"], "impact waterfall entry_id")
        rule_id = _text(row["rule_id"], "impact waterfall rule_id")
        if entry_id in seen_entry_ids or rule_id in seen_rule_ids:
            raise StrategyError("impact waterfall ids must be unique")
        seen_entry_ids.add(entry_id)
        seen_rule_ids.add(rule_id)
        source = _exact_object(
            row["source_ref"],
            {
                "artifact_id",
                "artifact_content_hash",
                "asset_id",
                "asset_hash",
                "fragment_id",
                "sample_design_ref",
            },
            "impact waterfall source_ref",
        )
        for field in ("artifact_id", "asset_id", "fragment_id"):
            _text(source[field], f"impact waterfall {field}")
        for field in ("artifact_content_hash", "asset_hash"):
            _hash(source[field], f"impact waterfall {field}")
        if _sample_design_ref(source["sample_design_ref"]) != sample_design_ref:
            raise StrategyError(
                "impact waterfall source sample design does not match assessment"
            )
        action = _validate_action(row["action"], "impact waterfall action")
        waterfall_actions.append(_ACTION_NAMES[action.type])
        normalized_row = dict(row)
        for field in ("standalone", "incremental", "shadowed", "remaining_after"):
            normalized_row[field] = _validate_effect_slice(
                row[field], population_count, f"impact waterfall {field}"
            )
            _require_effect_amount_bindings(
                normalized_row[field],
                amount_bindings,
                f"impact waterfall {field}",
            )
        validated_waterfall.append(normalized_row)

    default_unmatched = _exact_object(
        document["default_unmatched"],
        {"action", "effect"},
        "impact default_unmatched",
    )
    default_action = _validate_action(
        default_unmatched["action"], "impact default action"
    )
    default_effect = _validate_effect_slice(
        default_unmatched["effect"], population_count, "impact default effect"
    )
    _require_effect_amount_bindings(
        default_effect, amount_bindings, "impact default effect"
    )
    previous_remaining = overall_effect
    for row in validated_waterfall:
        consumed_before = _residual_effect(
            overall_effect,
            [previous_remaining],
            population_total=population_count,
            name=f"waterfall rule {row['rule_id']} consumed-before effect",
        )
        _require_disjoint_effects_within(
            consumed_before,
            [row["shadowed"]],
            name=f"waterfall rule {row['rule_id']} shadowed",
        )
        _require_effect_rollup(
            [row["incremental"], row["shadowed"]],
            row["standalone"],
            name=f"waterfall rule {row['rule_id']} standalone partition",
        )
        _require_effect_rollup(
            [row["incremental"], row["remaining_after"]],
            previous_remaining,
            name=f"waterfall rule {row['rule_id']} remaining partition",
        )
        previous_remaining = row["remaining_after"]
    if previous_remaining != default_effect:
        raise StrategyError("impact default effect does not match final remaining rows")
    _require_population_conservation(
        population_count,
        waterfall=validated_waterfall,
        unmatched_count=default_effect["population_count"],
    )
    _require_action_effects_match_summary(
        overall_actions,
        action_effects=[
            *zip(
                waterfall_actions,
                (row["incremental"] for row in validated_waterfall),
                strict=True,
            ),
            (_ACTION_NAMES[default_action.type], default_effect),
        ],
    )

    monthly = _validate_monthly(
        document["monthly"],
        population_count=population_count,
        rule_ids=tuple(row["rule_id"] for row in validated_waterfall),
        rule_actions=tuple(waterfall_actions),
        default_action=_ACTION_NAMES[default_action.type],
        amount_bindings=amount_bindings,
    )
    if (bindings["month_col"] is None) != (monthly["status"] == "unavailable"):
        raise StrategyError("impact month binding contradicts monthly evidence status")
    _require_monthly_rollup(
        monthly,
        overall_effect=overall_effect,
        overall_actions=overall_actions,
        waterfall=validated_waterfall,
    )
    _validate_baseline(
        document["baseline"],
        mode=bindings["comparison_mode"],
        strategy_type=identity["strategy_type"],
        overall_actions=overall_actions,
        overall_effect=overall_effect,
        amount_bindings=amount_bindings,
        monthly=monthly,
    )

    expected_flags = _red_flags(
        population_count=population_count,
        labelled_count=population["labelled_count"],
        month_col=bindings["month_col"],
        amounts=overall_effect["amounts"],
        waterfall=validated_waterfall,
    )
    if document["red_flags"] != expected_flags:
        raise StrategyError("impact red_flags do not match derived evidence")

    conservation = _exact_object(
        document["conservation"],
        {
            "standalone_equals_incremental_plus_shadowed",
            "incremental_plus_default_equals_population",
            "monthly_rolls_to_overall",
        },
        "impact conservation",
    )
    if any(value is not True for value in conservation.values()):
        raise StrategyError("impact conservation checks must all pass")


def _validate_population(value: Any) -> dict[str, Any]:
    population = _exact_object(
        value,
        {
            "population_count",
            "labelled_count",
            "unlabelled_count",
            "label_coverage",
        },
        "impact population",
    )
    total = _positive_int(population["population_count"], "population_count")
    labelled = _count(population["labelled_count"], "labelled_count")
    unlabelled = _count(population["unlabelled_count"], "unlabelled_count")
    if labelled + unlabelled != total:
        raise StrategyError("impact label counts do not match population")
    _require_same_number(
        population["label_coverage"],
        _ratio(labelled, total),
        "impact label_coverage",
    )
    return dict(population)


def _validate_effect_slice(
    value: Any, population_total: int, name: str
) -> dict[str, Any]:
    effect = _exact_object(
        value,
        {
            "population_count",
            "population_share",
            "labelled_count",
            "label_coverage",
            "bad_count",
            "bad_rate",
            "amounts",
        },
        name,
    )
    count = _count(effect["population_count"], f"{name} population_count")
    labelled = _count(effect["labelled_count"], f"{name} labelled_count")
    bad = _count(effect["bad_count"], f"{name} bad_count")
    if count > population_total or labelled > count or bad > labelled:
        raise StrategyError(f"{name} counts are inconsistent")
    _require_same_number(
        effect["population_share"],
        _ratio(count, population_total),
        f"{name} population_share",
    )
    _require_same_number(
        effect["label_coverage"],
        _ratio(labelled, count),
        f"{name} label_coverage",
    )
    _require_same_number(
        effect["bad_rate"], _ratio(bad, labelled), f"{name} bad_rate"
    )
    _validate_amounts(effect["amounts"], count, f"{name} amounts")
    return dict(effect)


def _validate_amounts(value: Any, count: int, name: str) -> None:
    amounts = _exact_object(
        value, {"loan_amount", "overdue_amount", "paired"}, name
    )
    singles: dict[str, Mapping[str, Any]] = {}
    for field in ("loan_amount", "overdue_amount"):
        item = _exact_object(
            amounts[field],
            {"status", "column", "coverage_count", "coverage_rate", "sum"},
            f"{name} {field}",
        )
        singles[field] = item
        if item["status"] == "unavailable":
            if any(item[key] is not None for key in item if key != "status"):
                raise StrategyError(f"{name} {field} unavailable values must be null")
            continue
        if item["status"] != "available":
            raise StrategyError(f"{name} {field} status is invalid")
        _text(item["column"], f"{name} {field} column")
        coverage = _count(item["coverage_count"], f"{name} {field} coverage")
        if coverage > count:
            raise StrategyError(f"{name} {field} coverage exceeds population")
        _require_same_number(
            item["coverage_rate"],
            _ratio(coverage, count),
            f"{name} {field} coverage_rate",
        )
        amount_sum = _non_negative_number(item["sum"], f"{name} {field} sum")
        if coverage == 0 and amount_sum != 0:
            raise StrategyError(f"{name} {field} sum requires covered rows")

    paired = _exact_object(
        amounts["paired"],
        {
            "status",
            "coverage_count",
            "coverage_rate",
            "loan_amount_sum",
            "overdue_amount_sum",
            "overdue_rate",
        },
        f"{name} paired",
    )
    if paired["status"] == "unavailable":
        if any(paired[key] is not None for key in paired if key != "status"):
            raise StrategyError(f"{name} paired unavailable values must be null")
        if all(item["status"] == "available" for item in singles.values()):
            raise StrategyError(f"{name} paired status contradicts amount bindings")
        return
    if paired["status"] != "available" or any(
        item["status"] != "available" for item in singles.values()
    ):
        raise StrategyError(f"{name} paired status is invalid")
    coverage = _count(paired["coverage_count"], f"{name} paired coverage")
    if coverage > count or any(
        coverage > int(item["coverage_count"]) for item in singles.values()
    ):
        raise StrategyError(f"{name} paired coverage is inconsistent")
    _require_same_number(
        paired["coverage_rate"],
        _ratio(coverage, count),
        f"{name} paired coverage_rate",
    )
    loan_sum = _non_negative_number(
        paired["loan_amount_sum"], f"{name} paired loan_amount_sum"
    )
    overdue_sum = _non_negative_number(
        paired["overdue_amount_sum"], f"{name} paired overdue_amount_sum"
    )
    if coverage == 0 and (loan_sum != 0 or overdue_sum != 0):
        raise StrategyError(f"{name} paired sums require covered rows")
    if loan_sum > float(singles["loan_amount"]["sum"]) or overdue_sum > float(
        singles["overdue_amount"]["sum"]
    ):
        raise StrategyError(f"{name} paired sums exceed single-column totals")
    _require_same_number(
        paired["overdue_rate"],
        _ratio(overdue_sum, loan_sum),
        f"{name} paired overdue_rate",
    )


def _validate_action_summary(
    value: Any,
    count: int,
    name: str,
    *,
    amount_bindings: Mapping[str, Any],
    total_effect: Mapping[str, Any],
) -> dict[str, Any]:
    summary = _exact_object(value, {"metrics", "breakdown"}, name)
    breakdown = summary["breakdown"]
    if not isinstance(breakdown, list) or len(breakdown) != len(_ACTION_ORDER):
        raise StrategyError(f"{name} breakdown must contain all actions")
    rows: list[dict[str, Any]] = []
    for expected_action, raw in zip(_ACTION_ORDER, breakdown, strict=True):
        row = _exact_object(
            raw,
            {
                "action",
                "count",
                "rate",
                "labelled_count",
                "bad_count",
                "bad_rate",
                "amounts",
            },
            f"{name} breakdown row",
        )
        if row["action"] != expected_action:
            raise StrategyError(f"{name} action order is invalid")
        action_count = _count(row["count"], f"{name} {expected_action} count")
        labelled = _count(
            row["labelled_count"], f"{name} {expected_action} labelled_count"
        )
        bad = _count(row["bad_count"], f"{name} {expected_action} bad_count")
        if labelled > action_count or bad > labelled:
            raise StrategyError(f"{name} {expected_action} counts are inconsistent")
        _require_same_number(
            row["rate"], _ratio(action_count, count), f"{name} {expected_action} rate"
        )
        _require_same_number(
            row["bad_rate"],
            _ratio(bad, labelled),
            f"{name} {expected_action} bad_rate",
        )
        _validate_amounts(
            row["amounts"], action_count, f"{name} {expected_action} amounts"
        )
        _require_effect_amount_bindings(
            {"amounts": row["amounts"]},
            amount_bindings,
            f"{name} {expected_action}",
        )
        rows.append(dict(row))
    if sum(row["count"] for row in rows) != count:
        raise StrategyError(f"{name} action counts do not cover the population")
    labelled_total = sum(row["labelled_count"] for row in rows)
    bad_total = sum(row["bad_count"] for row in rows)
    reject_row = rows[1]
    good_total = labelled_total - bad_total
    expected_metrics: dict[str, Any] = {}
    for row in rows:
        for field in ("count", "rate", "labelled_count", "bad_count", "bad_rate"):
            expected_metrics[f"{row['action']}_{field}"] = row[field]
    expected_metrics.update(
        {
            "overall_bad_count": bad_total,
            "overall_bad_rate": _ratio(bad_total, labelled_total),
            "bad_capture_rate": _ratio(reject_row["bad_count"], bad_total),
            "good_reject_rate": _ratio(
                reject_row["labelled_count"] - reject_row["bad_count"], good_total
            ),
        }
    )
    metrics = summary["metrics"]
    if not isinstance(metrics, dict) or set(metrics) != set(expected_metrics):
        raise StrategyError(f"{name} metrics are incomplete")
    for field, expected in expected_metrics.items():
        _require_same_number(metrics[field], expected, f"{name} metric {field}")
    _require_amount_rollup(
        [row["amounts"] for row in rows],
        total_effect["amounts"],
        name=f"{name} action amounts",
    )
    return dict(summary)


def _validate_monthly(
    value: Any,
    *,
    population_count: int,
    rule_ids: tuple[str, ...],
    rule_actions: tuple[str, ...],
    default_action: str,
    amount_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    monthly = _exact_object(value, {"status", "reason", "periods"}, "impact monthly")
    if monthly["status"] == "unavailable":
        if (
            monthly["reason"] != "month_column_not_provided"
            or monthly["periods"] != []
        ):
            raise StrategyError("unavailable impact monthly evidence is invalid")
        return dict(monthly)
    if monthly["status"] != "available" or monthly["reason"] is not None:
        raise StrategyError("impact monthly status is invalid")
    periods = monthly["periods"]
    if not isinstance(periods, list) or not periods:
        raise StrategyError("available impact monthly evidence needs periods")
    seen: list[str] = []
    for raw in periods:
        row = _exact_object(
            raw,
            {"period", "effect", "actions", "rule_incremental"},
            "impact monthly period",
        )
        period = _text(row["period"], "impact monthly period key")
        if re.fullmatch(r"\d{4}(?:0[1-9]|1[0-2])", period) is None:
            raise StrategyError("impact monthly period must use canonical YYYYMM")
        if period in seen:
            raise StrategyError("impact monthly period keys must be unique")
        seen.append(period)
        effect = _validate_effect_slice(
            row["effect"], population_count, f"impact monthly {period} effect"
        )
        _require_effect_amount_bindings(
            effect, amount_bindings, f"impact monthly {period} effect"
        )
        actions = _validate_action_summary(
            row["actions"],
            effect["population_count"],
            f"impact monthly {period} actions",
            amount_bindings=amount_bindings,
            total_effect=effect,
        )
        incremental = row["rule_incremental"]
        if not isinstance(incremental, list) or len(incremental) != len(rule_ids):
            raise StrategyError("impact monthly rule_incremental is incomplete")
        rule_effects: list[dict[str, Any]] = []
        for expected_rule_id, raw_rule in zip(rule_ids, incremental, strict=True):
            rule = _exact_object(
                raw_rule,
                {"rule_id", "effect"},
                "impact monthly rule_incremental row",
            )
            if rule["rule_id"] != expected_rule_id:
                raise StrategyError("impact monthly rule order is invalid")
            rule_effect = _validate_effect_slice(
                rule["effect"],
                population_count,
                f"impact monthly {period} rule effect",
            )
            _require_effect_amount_bindings(
                rule_effect,
                amount_bindings,
                f"impact monthly {period} rule effect",
            )
            rule_effects.append(rule_effect)
        _require_disjoint_effects_within(
            effect,
            rule_effects,
            name=f"impact monthly {period} rule increments",
        )
        default_effect = _residual_effect(
            effect,
            rule_effects,
            population_total=population_count,
            name=f"impact monthly {period} default residual",
        )
        _require_action_effects_match_summary(
            actions,
            action_effects=[
                *zip(rule_actions, rule_effects, strict=True),
                (default_action, default_effect),
            ],
        )
    if seen != sorted(seen):
        raise StrategyError("impact monthly periods must be sorted")
    return dict(monthly)


def _validate_baseline(
    value: Any,
    *,
    mode: str,
    strategy_type: str,
    overall_actions: Mapping[str, Any],
    overall_effect: Mapping[str, Any],
    amount_bindings: Mapping[str, Any],
    monthly: Mapping[str, Any],
) -> None:
    baseline = value
    if mode == "absolute":
        expected = {"status": "not_requested", "binding": None, "overall": None}
        if baseline != expected:
            raise StrategyError("absolute impact baseline must be not_requested")
        return
    baseline = _exact_object(
        baseline, {"status", "binding", "overall", "monthly"}, "impact baseline"
    )
    if baseline["status"] != "available":
        raise StrategyError("requested impact baseline must be available")
    binding = _exact_object(
        baseline["binding"],
        {"strategy_id", "strategy_type", "spec_hash"},
        "impact baseline binding",
    )
    _text(binding["strategy_id"], "impact baseline strategy_id")
    if binding["strategy_type"] != strategy_type:
        raise StrategyError("impact baseline strategy type is inconsistent")
    _hash(binding["spec_hash"], "impact baseline spec_hash")
    overall = _exact_object(
        baseline["overall"],
        {"current", "baseline", "metric_deltas", "amount_deltas"},
        "impact baseline overall",
    )
    current = _validate_action_summary(
        overall["current"],
        sum(row["count"] for row in overall_actions["breakdown"]),
        "impact baseline current",
        amount_bindings=amount_bindings,
        total_effect=overall_effect,
    )
    if current != overall_actions:
        raise StrategyError("impact baseline current does not match overall actions")
    baseline_summary = _validate_action_summary(
        overall["baseline"],
        sum(row["count"] for row in overall_actions["breakdown"]),
        "impact baseline comparison",
        amount_bindings=amount_bindings,
        total_effect=overall_effect,
    )
    _require_same_target_totals(
        current,
        baseline_summary,
        "impact baseline overall",
    )
    _validate_metric_deltas(
        overall["metric_deltas"],
        current["metrics"],
        baseline_summary["metrics"],
        "impact baseline overall deltas",
    )
    _validate_action_amount_deltas(
        overall["amount_deltas"],
        current,
        baseline_summary,
        "impact baseline overall amount deltas",
    )

    baseline_monthly = baseline["monthly"]
    if monthly["status"] == "unavailable":
        if baseline_monthly != {"status": "unavailable"}:
            raise StrategyError("impact baseline monthly status is inconsistent")
        return
    baseline_monthly = _exact_object(
        baseline_monthly, {"status", "periods"}, "impact baseline monthly"
    )
    if baseline_monthly["status"] != "available" or not isinstance(
        baseline_monthly["periods"], list
    ):
        raise StrategyError("impact baseline monthly evidence is invalid")
    if len(baseline_monthly["periods"]) != len(monthly["periods"]):
        raise StrategyError("impact baseline monthly periods are incomplete")
    period_comparisons: list[dict[str, Any]] = []
    for current_period, raw in zip(
        monthly["periods"], baseline_monthly["periods"], strict=True
    ):
        row = _exact_object(
            raw,
            {
                "period",
                "current",
                "baseline",
                "metric_deltas",
                "amount_deltas",
            },
            "impact baseline monthly period",
        )
        if row["period"] != current_period["period"]:
            raise StrategyError("impact baseline monthly period is inconsistent")
        period_count = current_period["effect"]["population_count"]
        current_summary = _validate_action_summary(
            row["current"],
            period_count,
            "impact baseline monthly current",
            amount_bindings=amount_bindings,
            total_effect=current_period["effect"],
        )
        if current_summary != current_period["actions"]:
            raise StrategyError("impact baseline monthly current evidence drifted")
        comparison = _validate_action_summary(
            row["baseline"],
            period_count,
            "impact baseline monthly comparison",
            amount_bindings=amount_bindings,
            total_effect=current_period["effect"],
        )
        _require_same_target_totals(
            current_summary,
            comparison,
            "impact baseline monthly",
        )
        period_comparisons.append(comparison)
        _validate_metric_deltas(
            row["metric_deltas"],
            current_summary["metrics"],
            comparison["metrics"],
            "impact baseline monthly deltas",
        )
        _validate_action_amount_deltas(
            row["amount_deltas"],
            current_summary,
            comparison,
            "impact baseline monthly amount deltas",
        )
    for field, expected in baseline_summary["metrics"].items():
        if field.endswith("_count") and sum(
            summary["metrics"][field] for summary in period_comparisons
        ) != expected:
            raise StrategyError(
                f"impact baseline monthly {field} does not roll to overall"
            )
    for position, action in enumerate(_ACTION_ORDER):
        _require_amount_rollup(
            [
                summary["breakdown"][position]["amounts"]
                for summary in period_comparisons
            ],
            baseline_summary["breakdown"][position]["amounts"],
            name=f"impact baseline monthly {action} action amounts",
        )


def _require_same_target_totals(
    current: Mapping[str, Any],
    baseline: Mapping[str, Any],
    name: str,
) -> None:
    for field in ("labelled_count", "bad_count"):
        current_total = sum(int(row[field]) for row in current["breakdown"])
        baseline_total = sum(int(row[field]) for row in baseline["breakdown"])
        if current_total != baseline_total:
            raise StrategyError(f"{name} {field} differs on the same sample")


def _validate_metric_deltas(
    value: Any,
    current: Mapping[str, Any],
    baseline: Mapping[str, Any],
    name: str,
) -> None:
    expected = _metric_deltas(current, baseline)
    if not isinstance(value, dict) or set(value) != set(expected):
        raise StrategyError(f"{name} are incomplete")
    for field, expected_value in expected.items():
        _require_same_number(value[field], expected_value, f"{name} {field}")


def _require_effect_amount_bindings(
    effect: Mapping[str, Any],
    bindings: Mapping[str, Any],
    name: str,
) -> None:
    amounts = effect["amounts"]
    for field in ("loan_amount", "overdue_amount"):
        expected_column = bindings[field]
        item = amounts[field]
        if expected_column is None:
            if item["status"] != "unavailable" or item["column"] is not None:
                raise StrategyError(f"{name} {field} contradicts its column binding")
        elif item["status"] != "available" or item["column"] != expected_column:
            raise StrategyError(f"{name} {field} contradicts its column binding")
    expected_paired = all(bindings[field] is not None for field in bindings)
    if (amounts["paired"]["status"] == "available") is not expected_paired:
        raise StrategyError(f"{name} paired status contradicts amount bindings")


def _require_disjoint_effects_within(
    total: Mapping[str, Any],
    parts: Sequence[Mapping[str, Any]],
    *,
    name: str,
) -> None:
    for field in ("population_count", "labelled_count", "bad_count"):
        if sum(int(part[field]) for part in parts) > int(total[field]):
            raise StrategyError(f"{name} {field} exceeds the period total")
    for amount_key in ("loan_amount", "overdue_amount", "paired"):
        expected = total["amounts"][amount_key]
        if expected["status"] != "available":
            continue
        observations = [part["amounts"][amount_key] for part in parts]
        if sum(int(item["coverage_count"]) for item in observations) > int(
            expected["coverage_count"]
        ):
            raise StrategyError(f"{name} {amount_key} coverage exceeds period total")
        sum_fields = (
            ("loan_amount_sum", "overdue_amount_sum")
            if amount_key == "paired"
            else ("sum",)
        )
        for field in sum_fields:
            if sum(float(item[field]) for item in observations) > float(
                expected[field]
            ) + 1e-9:
                raise StrategyError(
                    f"{name} {amount_key} {field} exceeds period total"
                )


def _residual_effect(
    total: Mapping[str, Any],
    parts: Sequence[Mapping[str, Any]],
    *,
    population_total: int,
    name: str,
) -> dict[str, Any]:
    counts = {
        field: int(total[field]) - sum(int(part[field]) for part in parts)
        for field in ("population_count", "labelled_count", "bad_count")
    }
    if any(value < 0 for value in counts.values()):
        raise StrategyError(f"{name} counts are negative")
    result = {
        **counts,
        "population_share": _ratio(counts["population_count"], population_total),
        "label_coverage": _ratio(
            counts["labelled_count"], counts["population_count"]
        ),
        "bad_rate": _ratio(counts["bad_count"], counts["labelled_count"]),
        "amounts": _residual_amounts(
            total["amounts"],
            [part["amounts"] for part in parts],
            population_count=counts["population_count"],
            name=name,
        ),
    }
    return _validate_effect_slice(result, population_total, name)


def _residual_amounts(
    total: Mapping[str, Any],
    parts: Sequence[Mapping[str, Any]],
    *,
    population_count: int,
    name: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for amount_key in ("loan_amount", "overdue_amount"):
        expected = total[amount_key]
        if expected["status"] == "unavailable":
            result[amount_key] = dict(expected)
            continue
        observations = [part[amount_key] for part in parts]
        coverage = int(expected["coverage_count"]) - sum(
            int(item["coverage_count"]) for item in observations
        )
        amount_sum = _non_negative_difference(
            float(expected["sum"]),
            sum(float(item["sum"]) for item in observations),
            name=f"{name} {amount_key} sum",
        )
        result[amount_key] = {
            "status": "available",
            "column": expected["column"],
            "coverage_count": coverage,
            "coverage_rate": _ratio(coverage, population_count),
            "sum": amount_sum,
        }
    expected_paired = total["paired"]
    if expected_paired["status"] == "unavailable":
        result["paired"] = dict(expected_paired)
        return result
    paired_parts = [part["paired"] for part in parts]
    coverage = int(expected_paired["coverage_count"]) - sum(
        int(item["coverage_count"]) for item in paired_parts
    )
    loan_sum = _non_negative_difference(
        float(expected_paired["loan_amount_sum"]),
        sum(float(item["loan_amount_sum"]) for item in paired_parts),
        name=f"{name} paired loan_amount_sum",
    )
    overdue_sum = _non_negative_difference(
        float(expected_paired["overdue_amount_sum"]),
        sum(float(item["overdue_amount_sum"]) for item in paired_parts),
        name=f"{name} paired overdue_amount_sum",
    )
    result["paired"] = {
        "status": "available",
        "coverage_count": coverage,
        "coverage_rate": _ratio(coverage, population_count),
        "loan_amount_sum": loan_sum,
        "overdue_amount_sum": overdue_sum,
        "overdue_rate": _ratio(overdue_sum, loan_sum),
    }
    return result


def _non_negative_difference(left: float, right: float, *, name: str) -> float:
    result = left - right
    if result < -1e-9:
        raise StrategyError(f"{name} is negative")
    return 0.0 if abs(result) <= 1e-9 else result


def _require_action_effects_match_summary(
    summary: Mapping[str, Any],
    *,
    action_effects: Sequence[tuple[str, Mapping[str, Any]]],
) -> None:
    totals = {
        action: {"count": 0, "labelled_count": 0, "bad_count": 0}
        for action in _ACTION_ORDER
    }
    for action, effect in action_effects:
        if action not in totals:
            raise StrategyError("impact action effect type is invalid")
        totals[action]["count"] += int(effect["population_count"])
        totals[action]["labelled_count"] += int(effect["labelled_count"])
        totals[action]["bad_count"] += int(effect["bad_count"])
    for row in summary["breakdown"]:
        expected = totals[row["action"]]
        if any(int(row[field]) != expected[field] for field in expected):
            raise StrategyError("impact action summary does not match Pool effects")
        _require_amount_rollup(
            [
                effect["amounts"]
                for action, effect in action_effects
                if action == row["action"]
            ],
            row["amounts"],
            name=f"impact {row['action']} action amounts",
        )


def _validate_action(value: Any, name: str) -> StrategyAction:
    if not isinstance(value, Mapping):
        raise StrategyError(f"{name} must be an object")
    action = StrategyAction.from_dict(value)
    if action.type not in {"approval", "reject", "review"}:
        raise StrategyError(f"{name} type is invalid")
    if action.to_dict() != value:
        raise StrategyError(f"{name} is not canonical")
    return action


def _exact_object(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise StrategyError(f"{name} fields are invalid")
    return value


def _count(value: Any, name: str) -> int:
    return _non_negative_int(value, name)


def _positive_int(value: Any, name: str) -> int:
    result = _non_negative_int(value, name)
    if result == 0:
        raise StrategyError(f"{name} must be a positive integer")
    return result


def _non_negative_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise StrategyError(f"{name} must be a non-negative finite number")
    return float(value)


def _require_same_number(actual: Any, expected: Any, name: str) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise StrategyError(f"{name} is inconsistent")
        return
    if (
        isinstance(actual, bool)
        or not isinstance(actual, int | float)
        or not math.isfinite(float(actual))
        or not math.isclose(
            float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12
        )
    ):
        raise StrategyError(f"{name} is inconsistent")


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
    _require_work_budget(
        row_count=len(frame),
        rule_count=rule_count,
        name="Pool impact",
    )
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


def _require_work_budget(*, row_count: int, rule_count: int, name: str) -> None:
    if row_count > MAX_IMPACT_ROWS or rule_count > MAX_IMPACT_RULES:
        raise StrategyError(f"{name} exceeds the row or rule budget")
    if row_count * rule_count > MAX_IMPACT_WORK:
        raise StrategyError(f"{name} exceeds the rows-by-rules work budget")


def _preflight_strategy_expression_limits(
    spec: object,
    *,
    name: str,
) -> None:
    """Bound raw expression shape before recursive Strategy DSL parsing."""

    if not isinstance(spec, Mapping):
        return
    rules = spec.get("rules")
    if not isinstance(rules, Sequence) or isinstance(
        rules, str | bytes | bytearray
    ):
        return
    node_count = 0
    for rule in rules:
        if not isinstance(rule, Mapping) or not isinstance(
            rule.get("condition"), Mapping
        ):
            continue
        stack: list[tuple[Mapping[str, Any], int]] = [(rule["condition"], 1)]
        while stack:
            expression, depth = stack.pop()
            node_count += 1
            if depth > MAX_IMPACT_EXPRESSION_DEPTH:
                raise StrategyError(f"{name} exceeds the expression depth budget")
            if node_count > MAX_IMPACT_EXPRESSION_NODES:
                raise StrategyError(f"{name} exceeds the expression node budget")
            op = expression.get("op")
            if op in {"and", "or", "n_of_k"}:
                args = expression.get("args")
                if isinstance(args, Sequence) and not isinstance(
                    args, str | bytes | bytearray
                ):
                    stack.extend(
                        (item, depth + 1)
                        for item in args
                        if isinstance(item, Mapping)
                    )
            elif op == "not" and isinstance(expression.get("arg"), Mapping):
                stack.append((expression["arg"], depth + 1))
            elif op == "compare" and expression.get("operator") in {
                "in",
                "not_in",
            }:
                values = expression.get("value")
                if isinstance(values, Sequence) and not isinstance(
                    values, str | bytes | bytearray
                ):
                    node_count += len(values)
                    if node_count > MAX_IMPACT_EXPRESSION_NODES:
                        raise StrategyError(
                            f"{name} exceeds the expression node budget"
                        )


def _strategy_expression_budget(
    rules: Sequence[Any],
    *,
    name: str,
) -> tuple[int, int]:
    node_count = 0
    evaluation_cost = 0
    for rule in rules:
        stack: list[tuple[Mapping[str, Any], int]] = [(rule.condition, 1)]
        while stack:
            expression, depth = stack.pop()
            node_count += 1
            evaluation_cost += 1
            if depth > MAX_IMPACT_EXPRESSION_DEPTH:
                raise StrategyError(f"{name} exceeds the expression depth budget")
            op = expression["op"]
            if op in {"and", "or", "n_of_k"}:
                stack.extend(
                    (item, depth + 1) for item in expression["args"]
                )
            elif op == "not":
                stack.append((expression["arg"], depth + 1))
            elif op == "compare" and expression["operator"] in {
                "in",
                "not_in",
            }:
                membership_cost = len(expression["value"])
                node_count += membership_cost
                evaluation_cost += membership_cost
            if node_count > MAX_IMPACT_EXPRESSION_NODES:
                raise StrategyError(f"{name} exceeds the expression node budget")
    return node_count, evaluation_cost


def _require_evaluation_work_budget(
    *,
    row_count: int,
    current_expression_cost: int,
    baseline_expression_cost: int,
) -> None:
    # Current rules execute once for first-match assignment and once for each
    # standalone waterfall slice. Baseline rules execute once.
    cost = row_count * (
        2 * current_expression_cost + baseline_expression_cost
    )
    if cost > MAX_IMPACT_WORK:
        raise StrategyError("Pool impact exceeds the expression evaluation work budget")


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


def _sample_design_ref(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _SAMPLE_DESIGN_REF_FIELDS:
        raise StrategyError(
            "sample design reference must contain the exact governed fields"
        )
    sample_design_id = _text(
        value["sample_design_id"], "sample design sample_design_id"
    )
    if _SAMPLE_DESIGN_ID_RE.fullmatch(sample_design_id) is None:
        raise StrategyError("sample design sample_design_id is invalid")
    if value["partition"] != "development":
        raise StrategyError("sample design partition must be development")
    return {
        "artifact_id": _hash(value["artifact_id"], "sample design artifact_id"),
        "artifact_content_hash": _hash(
            value["artifact_content_hash"],
            "sample design artifact_content_hash",
        ),
        "sample_design_id": sample_design_id,
        "sample_design_content_hash": _hash(
            value["sample_design_content_hash"],
            "sample design sample_design_content_hash",
        ),
        "partition": "development",
    }


def _target_bad_value(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in {0, 1}:
        raise StrategyError("target_bad_value must be integer 0 or 1")
    return value


def _target_series(
    frame: pd.DataFrame,
    column: str,
    *,
    target_bad_value: int,
) -> pd.Series:
    bad_value = _target_bad_value(target_bad_value)
    raw = frame[column]
    missing = raw.isna()
    numeric = pd.to_numeric(raw, errors="coerce")
    invalid = (~missing) & (np.iscomplex(numeric) | (~np.isfinite(numeric)))
    if bool(invalid.any()) or bool((~numeric.loc[~missing].isin([0, 1])).any()):
        raise StrategyError("target must contain only 0, 1, or missing")
    normalized = numeric.astype(float)
    if bad_value == 0:
        normalized = normalized.map(
            lambda value: value if pd.isna(value) else 1.0 - value
        )
    return normalized.reset_index(drop=True)


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
        unsupported = raw.notna() & raw.map(
            lambda value: not _is_supported_amount_scalar(value)
        )
        if bool(unsupported.any()):
            raise StrategyError(
                f"{column} must contain non-negative finite real numbers or missing"
            )
        numeric = pd.to_numeric(raw, errors="coerce")
        invalid = raw.notna() & (
            np.iscomplex(numeric) | (~np.isfinite(numeric))
        )
        if bool(invalid.any()) or bool((numeric.dropna() < 0).any()):
            raise StrategyError(
                f"{column} must contain non-negative finite real numbers or missing"
            )
        result[key] = {
            "status": "available",
            "column": column,
            "values": numeric.astype(float).reset_index(drop=True),
        }
    return result


def _is_supported_amount_scalar(value: object) -> bool:
    if isinstance(value, bool | np.bool_ | complex | np.complexfloating):
        return False
    return isinstance(value, str | Real | Decimal)


def _period_series(frame: pd.DataFrame, column: str | None) -> pd.Series | None:
    if column is None:
        return None
    try:
        return month_key_series(frame[column], column_name=column).reset_index(drop=True)
    except ValueError as exc:
        raise StrategyError(str(exc)) from exc


def _require_monthly_budget(
    *,
    row_count: int,
    rule_count: int,
    periods: pd.Series | None,
) -> None:
    if periods is None:
        return
    period_count = int(periods.nunique(dropna=False))
    if period_count > MAX_IMPACT_MONTHS:
        raise StrategyError("Pool impact exceeds the monthly period budget")
    if row_count * max(1, rule_count) * max(1, period_count) > MAX_IMPACT_MONTHLY_WORK:
        raise StrategyError("Pool impact exceeds the monthly work budget")


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
    amounts: Mapping[str, Any],
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
            "amounts": _amount_observations(selected, amounts=amounts),
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
    amounts: Mapping[str, Any],
    current_actions: pd.Series,
    current_rule_count: int,
    current_expression_nodes: int,
    current_expression_cost: int,
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
    raw_rules = baseline_spec.get("rules") if isinstance(baseline_spec, Mapping) else None
    if isinstance(raw_rules, list):
        _require_work_budget(
            row_count=len(frame),
            rule_count=current_rule_count + len(raw_rules),
            name="Pool impact comparison",
        )
    _preflight_strategy_expression_limits(baseline_spec, name="baseline strategy")
    parsed = parse_strategy_spec(baseline_spec)
    baseline_expression_nodes, baseline_expression_cost = _strategy_expression_budget(
        parsed.rules,
        name="baseline strategy",
    )
    if current_expression_nodes + baseline_expression_nodes > MAX_IMPACT_EXPRESSION_NODES:
        raise StrategyError("Pool impact comparison exceeds the expression node budget")
    _require_evaluation_work_budget(
        row_count=len(frame),
        current_expression_cost=current_expression_cost,
        baseline_expression_cost=baseline_expression_cost,
    )
    comparison_rule_count = current_rule_count + len(parsed.rules)
    _require_work_budget(
        row_count=len(frame),
        rule_count=comparison_rule_count,
        name="Pool impact comparison",
    )
    _require_monthly_budget(
        row_count=len(frame),
        rule_count=comparison_rule_count,
        periods=periods,
    )
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
    current_summary = _action_summary(
        current_actions,
        all_mask,
        target=target,
        amounts=amounts,
    )
    baseline_summary = _action_summary(
        baseline_actions,
        all_mask,
        target=target,
        amounts=amounts,
    )
    return {
        "status": "available",
        "binding": binding,
        "overall": {
            "current": current_summary,
            "baseline": baseline_summary,
            "metric_deltas": _metric_deltas(
                current_summary["metrics"], baseline_summary["metrics"]
            ),
            "amount_deltas": _action_amount_deltas(
                current_summary,
                baseline_summary,
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
        action_summary = _action_summary(
            actions,
            mask,
            target=target,
            amounts=amounts,
        )
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
            baseline_summary = _action_summary(
                baseline_actions,
                mask,
                target=target,
                amounts=amounts,
            )
            baseline_rows.append(
                {
                    "period": period,
                    "current": action_summary,
                    "baseline": baseline_summary,
                    "metric_deltas": _metric_deltas(
                        action_summary["metrics"], baseline_summary["metrics"]
                    ),
                    "amount_deltas": _action_amount_deltas(
                        action_summary,
                        baseline_summary,
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
    _require_effect_rollup(
        [row["effect"] for row in rows],
        overall_effect,
        name="monthly overall effect",
    )
    for key, expected in overall_actions["metrics"].items():
        if key.endswith("_count"):
            actual = sum(row["actions"]["metrics"][key] for row in rows)
            if actual != expected:
                raise StrategyError(f"monthly {key} does not roll to overall")
    for position, action in enumerate(_ACTION_ORDER):
        _require_amount_rollup(
            [row["actions"]["breakdown"][position]["amounts"] for row in rows],
            overall_actions["breakdown"][position]["amounts"],
            name=f"monthly {action} action amounts",
        )
    for position, rule in enumerate(waterfall):
        _require_effect_rollup(
            [row["rule_incremental"][position]["effect"] for row in rows],
            rule["incremental"],
            name=f"monthly rule {rule['rule_id']} incremental effect",
        )


def _require_effect_rollup(
    parts: Sequence[Mapping[str, Any]],
    overall: Mapping[str, Any],
    *,
    name: str,
) -> None:
    for field in ("population_count", "labelled_count", "bad_count"):
        if sum(int(part[field]) for part in parts) != int(overall[field]):
            raise StrategyError(f"{name} {field} does not roll to overall")
    _require_amount_rollup(
        [part["amounts"] for part in parts],
        overall["amounts"],
        name=name,
    )


def _require_amount_rollup(
    parts: Sequence[Mapping[str, Any]],
    overall: Mapping[str, Any],
    *,
    name: str,
) -> None:
    for amount_key in ("loan_amount", "overdue_amount", "paired"):
        expected = overall[amount_key]
        observations = [part[amount_key] for part in parts]
        if any(item["status"] != expected["status"] for item in observations):
            raise StrategyError(f"{name} {amount_key} status does not roll up")
        if expected["status"] != "available":
            continue
        if amount_key != "paired" and any(
            item["column"] != expected["column"] for item in observations
        ):
            raise StrategyError(f"{name} {amount_key} column changed by period")
        if sum(int(item["coverage_count"]) for item in observations) != int(
            expected["coverage_count"]
        ):
            raise StrategyError(f"{name} {amount_key} coverage does not roll up")
        sum_fields = (
            ("loan_amount_sum", "overdue_amount_sum")
            if amount_key == "paired"
            else ("sum",)
        )
        for field in sum_fields:
            actual = sum(float(item[field]) for item in observations)
            if not math.isclose(
                actual, float(expected[field]), rel_tol=1e-12, abs_tol=1e-9
            ):
                raise StrategyError(
                    f"{name} {amount_key} {field} does not roll to overall"
                )


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


def _action_amount_deltas(
    current: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, dict[str, dict[str, Any]]]:
    current_rows = {row["action"]: row for row in current["breakdown"]}
    baseline_rows = {row["action"]: row for row in baseline["breakdown"]}
    if set(current_rows) != set(_ACTION_ORDER) or set(baseline_rows) != set(
        _ACTION_ORDER
    ):
        raise StrategyError("baseline action amount rows are incomplete")
    return {
        action: _amount_deltas(
            current_rows[action]["amounts"], baseline_rows[action]["amounts"]
        )
        for action in _ACTION_ORDER
    }


def _amount_deltas(
    current: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for amount_key in ("loan_amount", "overdue_amount", "paired"):
        left = current[amount_key]
        right = baseline[amount_key]
        if left["status"] != right["status"]:
            raise StrategyError("baseline amount availability differs on the same sample")
        fields = (
            (
                "coverage_count",
                "coverage_rate",
                "loan_amount_sum",
                "overdue_amount_sum",
                "overdue_rate",
            )
            if amount_key == "paired"
            else ("coverage_count", "coverage_rate", "sum")
        )
        values: dict[str, Any] = {"status": left["status"]}
        for field in fields:
            left_value = left[field]
            right_value = right[field]
            if left_value is None or right_value is None:
                values[field] = None
            elif isinstance(left_value, int) and isinstance(right_value, int):
                values[field] = left_value - right_value
            else:
                values[field] = float(left_value) - float(right_value)
        result[amount_key] = values
    return result


def _validate_action_amount_deltas(
    value: Any,
    current: Mapping[str, Any],
    baseline: Mapping[str, Any],
    name: str,
) -> None:
    expected = _action_amount_deltas(current, baseline)
    rows = _exact_object(value, set(_ACTION_ORDER), name)
    for action, expected_amounts in expected.items():
        amounts = _exact_object(
            rows[action], {"loan_amount", "overdue_amount", "paired"}, f"{name} {action}"
        )
        for amount_key, expected_item in expected_amounts.items():
            item = _exact_object(
                amounts[amount_key],
                set(expected_item),
                f"{name} {action} {amount_key}",
            )
            if item["status"] != expected_item["status"]:
                raise StrategyError(
                    f"{name} {action} {amount_key} status is inconsistent"
                )
            for field, expected_value in expected_item.items():
                if field == "status":
                    continue
                _require_same_number(
                    item[field],
                    expected_value,
                    f"{name} {action} {amount_key} {field}",
                )


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
    "MAX_IMPACT_MONTHS",
    "MAX_IMPACT_MONTHLY_WORK",
    "MAX_IMPACT_EXPRESSION_NODES",
    "MAX_IMPACT_EXPRESSION_DEPTH",
    "STRATEGY_POOL_IMPACT_PRODUCER_VERSION",
    "STRATEGY_POOL_IMPACT_SCHEMA_VERSION",
    "build_strategy_pool_impact_assessment",
    "canonical_strategy_pool_impact_json",
    "validate_strategy_pool_impact_assessment",
]
