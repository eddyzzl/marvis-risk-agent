"""Deterministic refinement of univariate evidence into a governed candidate.

This module deliberately stops before adoption.  It replays the signed parent
evidence against the exact labelled rows, applies only explicitly requested bin
merges, and emits a self-authenticating development asset containing a Strategy
DSL expression fragment.  It does not choose an approval/reject action.
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

from marvis.feature.iv import _smoothed_woe_iv
from marvis.feature.metrics import feature_auc, feature_ks
from marvis.packs.strategy.candidate_evidence import validate_candidate_evidence
from marvis.packs.strategy.dsl import canonicalize_expression
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.evaluator import evaluate_expression


CANDIDATE_ASSET_SCHEMA_VERSION = "strategy.candidate-asset.v1"
PRODUCER_VERSION = "strategy.candidate-asset/1"

_DIMENSIONS = ("count", "loan_amount", "overdue_amount")
_METRIC_STATUSES = frozenset(
    {"observed", "unavailable", "insufficient_data", "not_applicable"}
)
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ASSET_ID_RE = re.compile(r"^candidate-asset-[0-9a-f]{32}$")
_BIN_ID_RE = re.compile(r"^candidate-bin-[0-9a-f]{32}$")
_RULE_ID_RE = re.compile(r"^candidate-rule-[0-9a-f]{32}$")
_EFFECT_ID_RE = re.compile(r"^candidate-effect-[0-9a-f]{32}$")

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "asset_id",
        "asset_type",
        "effect_stage",
        "validation_status",
        "parent",
        "feature",
        "method",
        "refinement",
        "selection",
        "selection_reason",
        "rule",
        "effect",
        "metrics",
        "producer_version",
        "asset_hash",
    }
)
_PARENT_FIELDS = frozenset({"candidate_id", "evidence_hash", "source_evidence"})
_SOURCE_FIELDS = frozenset({"artifact_id", "kind", "content_hash"})
_REFINEMENT_FIELDS = frozenset(
    {
        "source_bin_count",
        "edited_bin_count",
        "merge_groups",
        "smoothing",
        "metrics",
        "bins",
    }
)
_METHOD_METRIC_FIELDS = frozenset(
    {"iv", "ks", "auc", "risk_direction", "missing_rate", "amount_metrics"}
)
_BIN_FIELDS = frozenset(
    {
        "index",
        "bin_id",
        "source_bin_ids",
        "kind",
        "condition",
        "count",
        "share",
        "good",
        "bad",
        "bad_rate",
        "woe",
        "iv_contribution",
        "lift",
        "cumulative_ks",
        "amount_metrics",
    }
)
_RULE_FIELDS = frozenset({"rule_id", "condition", "selected_bin_ids", "source_bin_ids"})
_EFFECT_FIELDS = frozenset(
    {
        "effect_id",
        "selected_count",
        "selected_share",
        "good",
        "bad",
        "bad_rate",
        "lift",
        "amount_metrics",
    }
)
_METRIC_FIELDS = frozenset({"metric_name", "dimension", "status", "value"})
_AMOUNT_FIELDS = frozenset({"loan_amount", "overdue_amount", "overdue_rate"})


class CandidateAssetError(StrategyError):
    """Candidate refinement or asset validation failed closed."""


def refine_univariate_candidate(
    candidate_evidence: Mapping[str, Any],
    frame: pd.DataFrame,
    *,
    source_evidence: Mapping[str, Any],
    feature: str,
    method: str,
    merge_groups: Sequence[Sequence[str]],
    selection: Mapping[str, Any],
    selection_reason: str | None = None,
) -> dict[str, Any]:
    """Refine one available parent method and return Candidate Asset v1.

    ``frame`` must be the exact, already-labelled row set used by the parent
    analysis.  Every parent bin condition is evaluated row by row through the
    canonical Strategy DSL evaluator before any edit is accepted.
    """

    parent_evidence = validate_candidate_evidence(candidate_evidence)
    source = _normalize_source_evidence(source_evidence)
    feature_name = _text(feature, "feature")
    method_name = _text(method, "method")
    reason = _optional_text(selection_reason, "selection_reason")
    if not isinstance(frame, pd.DataFrame):
        raise CandidateAssetError("frame must be a pandas DataFrame")
    if not frame.columns.is_unique:
        raise CandidateAssetError("frame columns must be unique")

    analysis = parent_evidence["analysis"]
    feature_result, method_result = _parent_method(
        analysis,
        feature=feature_name,
        method=method_name,
    )
    target_name, target, loan_values, overdue_values = _labelled_arrays(
        frame,
        analysis=analysis,
        feature=feature_name,
    )
    source_bins, source_masks = _replay_parent_bins(
        frame,
        feature=feature_name,
        target_name=target_name,
        target=target,
        method_result=method_result,
    )
    normalized_merges = _normalize_merge_groups(merge_groups, source_bins)
    edited_groups = _edited_groups(source_bins, normalized_merges)
    smoothing = _smoothing(analysis)
    bins, method_metrics = _recompute_bins(
        edited_groups,
        source_masks=source_masks,
        target=target,
        loan_values=loan_values,
        overdue_values=overdue_values,
        feature=feature_name,
        method=method_name,
        smoothing=smoothing,
    )
    normalized_selection, selected_bins = _resolve_selection(selection, bins)
    if sum(int(row["count"]) for row in selected_bins) == 0:
        raise CandidateAssetError("selection must include at least one observed row")
    selected_mask = np.zeros(len(frame), dtype=bool)
    for bin_row in selected_bins:
        for source_bin_id in bin_row["source_bin_ids"]:
            selected_mask |= source_masks[source_bin_id]

    selected_conditions = [bin_row["condition"] for bin_row in selected_bins]
    rule_condition = canonicalize_expression({"op": "or", "args": selected_conditions})
    rule_without_id = {
        "condition": rule_condition,
        "selected_bin_ids": [row["bin_id"] for row in selected_bins],
        "source_bin_ids": [
            source_id for row in selected_bins for source_id in row["source_bin_ids"]
        ],
    }
    rule = {
        "rule_id": _stable_id(
            "candidate-rule",
            {
                "parent_evidence_hash": parent_evidence["evidence_hash"],
                "feature": feature_name,
                "method": method_name,
                **rule_without_id,
            },
        ),
        **rule_without_id,
    }

    effect_body = _effect(
        selected_mask,
        target=target,
        loan_values=loan_values,
        overdue_values=overdue_values,
    )
    effect = {
        "effect_id": _stable_id(
            "candidate-effect",
            {
                "parent_evidence_hash": parent_evidence["evidence_hash"],
                "rule_id": rule["rule_id"],
                **effect_body,
            },
        ),
        **effect_body,
    }
    metrics = _candidate_metrics(
        bins=bins,
        method_metrics=method_metrics,
        effect=effect,
    )
    body = {
        "schema_version": CANDIDATE_ASSET_SCHEMA_VERSION,
        "asset_type": "univariate_refinement",
        "effect_stage": "development",
        "validation_status": "unvalidated",
        "parent": {
            "candidate_id": parent_evidence["candidate_id"],
            "evidence_hash": parent_evidence["evidence_hash"],
            "source_evidence": source,
        },
        "feature": feature_name,
        "method": method_name,
        "refinement": {
            "source_bin_count": len(source_bins),
            "edited_bin_count": len(bins),
            "merge_groups": normalized_merges,
            "smoothing": smoothing,
            "metrics": method_metrics,
            "bins": bins,
        },
        "selection": normalized_selection,
        "selection_reason": reason,
        "rule": rule,
        "effect": effect,
        "metrics": metrics,
        "producer_version": PRODUCER_VERSION,
    }
    normalized_body = _normalize_asset_body(body)
    asset_id = _stable_id("candidate-asset", normalized_body)
    without_hash = {**normalized_body, "asset_id": asset_id}
    asset_hash = _sha256(_canonical_json(without_hash))
    return validate_candidate_asset({**without_hash, "asset_hash": asset_hash})


def validate_candidate_refinement_source_controls(
    candidate_evidence: Mapping[str, Any],
    *,
    feature: str,
    method: str,
    merge_groups: Sequence[Sequence[str]],
    selection: Mapping[str, Any],
) -> None:
    """Validate user pointers against one exact parent candidate without data I/O."""

    evidence = validate_candidate_evidence(candidate_evidence)
    feature_name = _text(feature, "feature")
    method_name = _text(method, "method")
    _feature_result, method_result = _parent_method(
        evidence["analysis"],
        feature=feature_name,
        method=method_name,
    )
    source_bins = method_result["bins"]
    normalized_merges = _normalize_merge_groups(merge_groups, source_bins)
    preview_bins = []
    for index, group in enumerate(_edited_groups(source_bins, normalized_merges)):
        count = sum(_integer(item["count"], "parent bin count", 0) for item in group)
        bad = sum(_integer(item["bad"], "parent bin bad", 0) for item in group)
        preview_bins.append(
            {
                "bin_id": f"preflight:{index}",
                "source_bin_ids": [item["id"] for item in group],
                "bad_rate": None if count == 0 else float(bad / count),
            }
        )
    _resolve_selection(selection, preview_bins)


def validate_candidate_asset(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an exact Candidate Asset v1 and return a detached canonical copy."""

    if not isinstance(payload, Mapping):
        raise CandidateAssetError("candidate asset must be an object")
    _exact_fields(payload, _TOP_LEVEL_FIELDS, "candidate asset")
    asset_id = _text(payload["asset_id"], "asset_id")
    if not _ASSET_ID_RE.fullmatch(asset_id):
        raise CandidateAssetError("asset_id has an invalid format")
    asset_hash = _hash(payload["asset_hash"], "asset_hash")
    body = {
        key: payload[key] for key in payload if key not in {"asset_id", "asset_hash"}
    }
    normalized_body = _normalize_asset_body(body)
    expected_id = _stable_id("candidate-asset", normalized_body)
    if not hmac.compare_digest(asset_id, expected_id):
        raise CandidateAssetError("asset_id does not match canonical asset identity")
    normalized_without_hash = {**normalized_body, "asset_id": asset_id}
    expected_hash = _sha256(_canonical_json(normalized_without_hash))
    if not hmac.compare_digest(asset_hash, expected_hash):
        raise CandidateAssetError("asset_hash does not match canonical candidate asset")
    return {**normalized_without_hash, "asset_hash": asset_hash}


def canonical_candidate_asset_json(payload: Mapping[str, Any]) -> str:
    """Return the sole canonical serialization for a verified asset."""

    return _canonical_json(validate_candidate_asset(payload))


def _parent_method(
    analysis: object,
    *,
    feature: str,
    method: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not isinstance(analysis, Mapping):
        raise CandidateAssetError("parent analysis must be an object")
    features = analysis.get("features")
    if not _sequence(features):
        raise CandidateAssetError("parent analysis.features must be a non-empty array")
    matches = [
        item
        for item in features
        if isinstance(item, Mapping) and item.get("feature") == feature
    ]
    if len(matches) != 1:
        raise CandidateAssetError("feature must identify exactly one parent analysis")
    feature_result = matches[0]
    methods = feature_result.get("methods")
    if not _sequence(methods):
        raise CandidateAssetError("parent feature.methods must be a non-empty array")
    method_matches = [
        item
        for item in methods
        if isinstance(item, Mapping) and item.get("method") == method
    ]
    if len(method_matches) != 1:
        raise CandidateAssetError("method must identify exactly one parent analysis")
    method_result = method_matches[0]
    if method_result.get("status") != "available":
        raise CandidateAssetError("only an available parent method can be refined")
    bins = method_result.get("bins")
    if not _sequence(bins):
        raise CandidateAssetError("available parent method must contain bins")
    return feature_result, method_result


def _labelled_arrays(
    frame: pd.DataFrame,
    *,
    analysis: Mapping[str, Any],
    feature: str,
) -> tuple[str, np.ndarray, np.ndarray | None, np.ndarray | None]:
    target_name = _text(analysis.get("target"), "parent analysis.target")
    row_count = _integer(analysis.get("row_count"), "parent analysis.row_count", 1)
    if len(frame) != row_count:
        raise CandidateAssetError(
            "frame row count does not match the parent labelled analysis"
        )
    if feature not in frame.columns or target_name not in frame.columns:
        raise CandidateAssetError("frame must contain the parent feature and target")
    target_numeric = pd.to_numeric(frame[target_name], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.all(np.isfinite(target_numeric)) or not np.all(
        np.isin(target_numeric, [0, 1])
    ):
        raise CandidateAssetError("frame target must contain only labelled binary 0/1")
    target = target_numeric.astype(int)
    if np.unique(target).size != 2:
        raise CandidateAssetError("frame target must contain both good and bad classes")
    parameters = analysis.get("parameters")
    if not isinstance(parameters, Mapping):
        raise CandidateAssetError("parent analysis.parameters must be an object")
    loan_name = parameters.get("loan_amount")
    overdue_name = parameters.get("overdue_amount")
    loan = _amount_values(frame, loan_name, "loan_amount")
    overdue = _amount_values(frame, overdue_name, "overdue_amount")
    return target_name, target, loan, overdue


def _amount_values(
    frame: pd.DataFrame,
    column: object,
    label: str,
) -> np.ndarray | None:
    if column is None:
        return None
    name = _text(column, f"parent analysis.parameters.{label}")
    if name not in frame.columns:
        raise CandidateAssetError(f"frame is missing configured {label} column")
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
    invalid = frame[name].notna().to_numpy() & ~np.isfinite(values)
    if np.any(invalid) or np.any(values[np.isfinite(values)] < 0):
        raise CandidateAssetError(
            f"{label} must contain non-negative finite numbers or null"
        )
    return values


def _replay_parent_bins(
    frame: pd.DataFrame,
    *,
    feature: str,
    target_name: str,
    target: np.ndarray,
    method_result: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    raw_bins = method_result["bins"]
    assert _sequence(raw_bins)
    source_bins: list[dict[str, Any]] = []
    for raw in raw_bins:
        if not isinstance(raw, Mapping):
            raise CandidateAssetError("parent bins must contain objects")
        index = _integer(raw.get("index"), "parent bin.index", 0)
        source_id = _text(raw.get("id"), "parent bin.id")
        kind = _text(raw.get("kind"), "parent bin.kind")
        if kind not in {"numeric_interval", "category", "sentinel", "missing"}:
            raise CandidateAssetError(f"unsupported parent bin kind: {kind}")
        condition_raw = raw.get("condition")
        if not isinstance(condition_raw, Mapping):
            raise CandidateAssetError("parent bin.condition must be an object")
        condition = canonicalize_expression(condition_raw)
        if _canonical_json(condition) != _canonical_json(condition_raw):
            raise CandidateAssetError(
                "parent bin.condition must be canonical Strategy DSL"
            )
        if _expression_fields(condition) != {feature}:
            raise CandidateAssetError(
                "parent bin.condition must reference only its analyzed feature"
            )
        source_bins.append(
            {
                "index": index,
                "id": source_id,
                "kind": kind,
                "condition": condition,
                "expected_count": _integer(raw.get("count"), "parent bin.count", 0),
                "expected_good": _integer(raw.get("good"), "parent bin.good", 0),
                "expected_bad": _integer(raw.get("bad"), "parent bin.bad", 0),
                "expected_share": _number(raw.get("share"), "parent bin.share", 0, 1),
            }
        )
    source_bins.sort(key=lambda item: item["index"])
    if [item["index"] for item in source_bins] != list(range(len(source_bins))):
        raise CandidateAssetError("parent bin indices must be contiguous from zero")
    source_ids = [item["id"] for item in source_bins]
    if len(set(source_ids)) != len(source_ids):
        raise CandidateAssetError("parent bin ids must be unique")

    masks: dict[str, np.ndarray] = {}
    membership = np.zeros(len(frame), dtype=int)
    feature_values = frame[feature].array
    for source_bin in source_bins:
        condition = source_bin["condition"]
        mask = np.fromiter(
            (
                evaluate_expression({feature: value}, condition)
                for value in feature_values
            ),
            dtype=bool,
            count=len(frame),
        )
        count = int(np.sum(mask))
        good = int(np.sum(target[mask] == 0))
        bad = int(np.sum(target[mask] == 1))
        if count != source_bin["expected_count"]:
            raise CandidateAssetError(
                f"parent bin {source_bin['id']} count does not replay on exact rows"
            )
        if good != source_bin["expected_good"] or bad != source_bin["expected_bad"]:
            raise CandidateAssetError(
                f"parent bin {source_bin['id']} label partition does not replay"
            )
        if not math.isclose(
            count / len(frame),
            source_bin["expected_share"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise CandidateAssetError(
                f"parent bin {source_bin['id']} share does not replay"
            )
        masks[source_bin["id"]] = mask
        membership += mask.astype(int)
    if not np.all(membership == 1):
        raise CandidateAssetError(
            "parent bin conditions must form an exact one-bin-per-row partition"
        )
    del target_name  # target is deliberately not available to DSL bin conditions.
    return source_bins, masks


def _normalize_merge_groups(
    merge_groups: object,
    source_bins: Sequence[Mapping[str, Any]],
) -> list[list[str]]:
    if not _sequence(merge_groups, allow_empty=True):
        raise CandidateAssetError("merge_groups must be an array")
    source_order = {item["id"]: index for index, item in enumerate(source_bins)}
    source_kind = {item["id"]: item["kind"] for item in source_bins}
    normalized: list[list[str]] = []
    claimed: set[str] = set()
    for position, raw_group in enumerate(merge_groups):
        if not _sequence(raw_group):
            raise CandidateAssetError(
                f"merge_groups[{position}] must contain at least two source bin ids"
            )
        group = [_text(item, f"merge_groups[{position}] item") for item in raw_group]
        if len(group) < 2:
            raise CandidateAssetError(
                f"merge_groups[{position}] must contain at least two source bin ids"
            )
        if len(set(group)) != len(group):
            raise CandidateAssetError("a merge group cannot repeat a source bin id")
        unknown = sorted(set(group) - set(source_order))
        if unknown:
            raise CandidateAssetError(
                "merge_groups contains unknown source bin ids: " + ", ".join(unknown)
            )
        overlap = sorted(set(group) & claimed)
        if overlap:
            raise CandidateAssetError(
                "a source bin cannot belong to multiple merge groups: "
                + ", ".join(overlap)
            )
        group.sort(key=source_order.__getitem__)
        kinds = {source_kind[source_id] for source_id in group}
        if "missing" in kinds:
            raise CandidateAssetError("a missing bin cannot be merged with another bin")
        ordinary = kinds & {"numeric_interval", "category"}
        special = kinds & {"sentinel"}
        if ordinary and special:
            raise CandidateAssetError(
                "missing or sentinel bins cannot be merged with ordinary bins"
            )
        if len(ordinary) > 1:
            raise CandidateAssetError("numeric and categorical bins cannot be merged")
        if ordinary == {"numeric_interval"}:
            positions = [source_order[source_id] for source_id in group]
            if positions != list(range(positions[0], positions[-1] + 1)):
                raise CandidateAssetError(
                    "numeric ordinary bins can only be merged when adjacent"
                )
        claimed.update(group)
        normalized.append(group)
    normalized.sort(key=lambda group: source_order[group[0]])
    return normalized


def _edited_groups(
    source_bins: Sequence[Mapping[str, Any]],
    merge_groups: Sequence[Sequence[str]],
) -> list[list[Mapping[str, Any]]]:
    by_id = {item["id"]: item for item in source_bins}
    group_by_id = {
        source_id: tuple(group) for group in merge_groups for source_id in group
    }
    emitted: set[tuple[str, ...]] = set()
    result: list[list[Mapping[str, Any]]] = []
    for source_bin in source_bins:
        source_id = source_bin["id"]
        group = group_by_id.get(source_id, (source_id,))
        if group in emitted:
            continue
        emitted.add(group)
        result.append([by_id[item] for item in group])
    return result


def _recompute_bins(
    edited_groups: Sequence[Sequence[Mapping[str, Any]]],
    *,
    source_masks: Mapping[str, np.ndarray],
    target: np.ndarray,
    loan_values: np.ndarray | None,
    overdue_values: np.ndarray | None,
    feature: str,
    method: str,
    smoothing: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    total_bad = int(np.sum(target == 1))
    total_good = int(np.sum(target == 0))
    base_bad_rate = total_bad / len(target)
    group_count = len(edited_groups)
    rows: list[dict[str, Any]] = []
    score = np.zeros(len(target), dtype=float)
    for index, group in enumerate(edited_groups):
        source_ids = [str(item["id"]) for item in group]
        mask = np.zeros(len(target), dtype=bool)
        for source_id in source_ids:
            mask |= source_masks[source_id]
        count = int(np.sum(mask))
        bad = int(np.sum(target[mask] == 1))
        good = count - bad
        bad_rate = bad / count if count else None
        woe, contribution = _smoothed_woe_iv(
            bad,
            good,
            total_bad,
            total_good,
            group_count,
            smoothing=smoothing,
        )
        if bad_rate is not None:
            score[mask] = bad_rate
        condition = (
            group[0]["condition"]
            if len(group) == 1
            else canonicalize_expression(
                {"op": "or", "args": [item["condition"] for item in group]}
            )
        )
        kind = str(group[0]["kind"])
        rows.append(
            {
                "index": index,
                "bin_id": _stable_id(
                    "candidate-bin",
                    {
                        "feature": feature,
                        "method": method,
                        "source_bin_ids": source_ids,
                    },
                ),
                "source_bin_ids": source_ids,
                "kind": kind,
                "condition": condition,
                "count": count,
                "share": count / len(target),
                "good": good,
                "bad": bad,
                "bad_rate": bad_rate,
                "woe": woe,
                "iv_contribution": contribution,
                "lift": bad_rate / base_bad_rate if bad_rate is not None else None,
                "cumulative_ks": 0.0,
                "amount_metrics": _amount_metrics(mask, loan_values, overdue_values),
            }
        )
    _assign_cumulative_ks(rows, total_bad=total_bad, total_good=total_good)
    ordinary_rates = [
        float(row["bad_rate"])
        for row in rows
        if row["kind"] == "numeric_interval" and row["bad_rate"] is not None
    ]
    metrics = {
        "iv": float(sum(row["iv_contribution"] for row in rows)),
        "ks": feature_ks(score, target),
        "auc": feature_auc(score, target),
        "risk_direction": _risk_direction(ordinary_rates, method=method),
        "missing_rate": next(
            (row["share"] for row in rows if row["kind"] == "missing"), 0.0
        ),
        "amount_metrics": _amount_metrics(
            np.ones(len(target), dtype=bool), loan_values, overdue_values
        ),
    }
    return rows, metrics


def _resolve_selection(
    selection: object,
    bins: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    if not isinstance(selection, Mapping):
        raise CandidateAssetError("selection must be an object")
    keys = set(selection)
    if keys == {"source_bin_ids"}:
        values = selection["source_bin_ids"]
        if not _sequence(values):
            raise CandidateAssetError("selection.source_bin_ids must be non-empty")
        requested = [_text(item, "selection.source_bin_ids item") for item in values]
        if len(set(requested)) != len(requested):
            raise CandidateAssetError("selection.source_bin_ids must be unique")
        source_order = [
            source_id for row in bins for source_id in row["source_bin_ids"]
        ]
        unknown = sorted(set(requested) - set(source_order))
        if unknown:
            raise CandidateAssetError(
                "selection contains unknown source bin ids: " + ", ".join(unknown)
            )
        requested_set = set(requested)
        selected = []
        for row in bins:
            members = set(row["source_bin_ids"])
            overlap = members & requested_set
            if overlap and overlap != members:
                raise CandidateAssetError(
                    "selection cannot partially select an explicitly merged bin"
                )
            if overlap:
                selected.append(row)
        if not selected:
            raise CandidateAssetError("selection must resolve at least one bin")
        resolved_source_ids = [
            source_id for source_id in source_order if source_id in requested_set
        ]
        return (
            {
                "mode": "source_bin_ids",
                "source_bin_ids": resolved_source_ids,
                "selected_bin_ids": [row["bin_id"] for row in selected],
            },
            selected,
        )
    if keys == {"risk_threshold"}:
        threshold = selection["risk_threshold"]
        if not isinstance(threshold, Mapping):
            raise CandidateAssetError("selection.risk_threshold must be an object")
        _exact_fields(
            threshold,
            frozenset({"operator", "value"}),
            "selection.risk_threshold",
        )
        operator = _text(threshold["operator"], "risk_threshold.operator")
        if operator not in {">=", ">", "<=", "<"}:
            raise CandidateAssetError("risk_threshold.operator is unsupported")
        value = _number(threshold["value"], "risk_threshold.value", 0, 1)
        selected = [
            row
            for row in bins
            if row["bad_rate"] is not None
            and _threshold_match(float(row["bad_rate"]), operator, value)
        ]
        if not selected:
            raise CandidateAssetError("risk_threshold must resolve at least one bin")
        return (
            {
                "mode": "risk_threshold",
                "risk_threshold": {
                    "metric": "bad_rate",
                    "operator": operator,
                    "value": value,
                },
                "source_bin_ids": [
                    source_id for row in selected for source_id in row["source_bin_ids"]
                ],
                "selected_bin_ids": [row["bin_id"] for row in selected],
            },
            selected,
        )
    raise CandidateAssetError(
        "selection requires exactly one of source_bin_ids or risk_threshold"
    )


def _threshold_match(value: float, operator: str, threshold: float) -> bool:
    if operator == ">=":
        return value >= threshold
    if operator == ">":
        return value > threshold
    if operator == "<=":
        return value <= threshold
    return value < threshold


def _effect(
    mask: np.ndarray,
    *,
    target: np.ndarray,
    loan_values: np.ndarray | None,
    overdue_values: np.ndarray | None,
) -> dict[str, Any]:
    count = int(np.sum(mask))
    bad = int(np.sum(target[mask] == 1))
    good = count - bad
    bad_rate = bad / count
    base_bad_rate = float(np.mean(target == 1))
    return {
        "selected_count": count,
        "selected_share": count / len(target),
        "good": good,
        "bad": bad,
        "bad_rate": bad_rate,
        "lift": bad_rate / base_bad_rate,
        "amount_metrics": _amount_metrics(mask, loan_values, overdue_values),
    }


def _candidate_metrics(
    *,
    bins: Sequence[Mapping[str, Any]],
    method_metrics: Mapping[str, Any],
    effect: Mapping[str, Any],
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for metric_name in ("iv", "ks", "auc"):
        observations.extend(
            _metric_triplet(
                f"refinement.{metric_name}",
                count=("observed", method_metrics[metric_name]),
                loan_amount=("unavailable", None),
                overdue_amount=("unavailable", None),
            )
        )
    total_amounts = method_metrics["amount_metrics"]
    for row in bins:
        prefix = f"refinement.bin.{row['bin_id']}"
        observations.extend(
            _metric_triplet(
                f"{prefix}.hit_rate",
                count=("observed", row["share"]),
                loan_amount=_amount_share(
                    row["amount_metrics"]["loan_amount"],
                    total_amounts["loan_amount"],
                ),
                overdue_amount=_amount_share(
                    row["amount_metrics"]["overdue_amount"],
                    total_amounts["overdue_amount"],
                ),
            )
        )
    observations.extend(
        _metric_triplet(
            "rule.hit_rate",
            count=("observed", effect["selected_share"]),
            loan_amount=_amount_share(
                effect["amount_metrics"]["loan_amount"],
                total_amounts["loan_amount"],
            ),
            overdue_amount=_amount_share(
                effect["amount_metrics"]["overdue_amount"],
                total_amounts["overdue_amount"],
            ),
        )
    )
    observations.extend(
        _metric_triplet(
            "rule.bad_rate",
            count=("observed", effect["bad_rate"]),
            loan_amount=("unavailable", None),
            overdue_amount=("unavailable", None),
        )
    )
    rate = effect["amount_metrics"]["overdue_rate"]
    observations.extend(
        _metric_triplet(
            "rule.overdue_rate",
            count=("not_applicable", None),
            loan_amount=(
                ("observed", rate["value"])
                if rate["status"] == "available"
                else (
                    "not_applicable"
                    if rate["status"] == "not_applicable"
                    else "unavailable",
                    None,
                )
            ),
            overdue_amount=("not_applicable", None),
        )
    )
    observations.sort(
        key=lambda item: (item["metric_name"], _DIMENSIONS.index(item["dimension"]))
    )
    return observations


def _metric_triplet(
    metric_name: str,
    *,
    count: tuple[str, int | float | None],
    loan_amount: tuple[str, int | float | None],
    overdue_amount: tuple[str, int | float | None],
) -> list[dict[str, Any]]:
    return [
        {
            "metric_name": metric_name,
            "dimension": "count",
            "status": count[0],
            "value": count[1],
        },
        {
            "metric_name": metric_name,
            "dimension": "loan_amount",
            "status": loan_amount[0],
            "value": loan_amount[1],
        },
        {
            "metric_name": metric_name,
            "dimension": "overdue_amount",
            "status": overdue_amount[0],
            "value": overdue_amount[1],
        },
    ]


def _amount_share(
    selected: Mapping[str, Any], total: Mapping[str, Any]
) -> tuple[str, float | None]:
    if selected.get("status") != "available" or total.get("status") != "available":
        return ("unavailable", None)
    denominator = float(total["sum"])
    if denominator == 0:
        return ("not_applicable", None)
    return ("observed", float(selected["sum"]) / denominator)


def _amount_metrics(
    mask: np.ndarray,
    loan_values: np.ndarray | None,
    overdue_values: np.ndarray | None,
) -> dict[str, Any]:
    loan = _amount_measure(mask, loan_values, "loan_amount")
    overdue = _amount_measure(mask, overdue_values, "overdue_amount")
    if loan_values is None or overdue_values is None:
        rate = {"status": "unavailable", "reason": "amount_column_not_configured"}
    else:
        paired = mask & np.isfinite(loan_values) & np.isfinite(overdue_values)
        if not np.any(mask):
            rate = {"status": "not_applicable", "reason": "empty_bin"}
        elif not np.any(paired):
            rate = {"status": "unavailable", "reason": "no_paired_amounts"}
        else:
            denominator = float(np.sum(loan_values[paired]))
            if denominator == 0:
                rate = {"status": "not_applicable", "reason": "zero_loan_amount"}
            else:
                rate = {
                    "status": "available",
                    "value": float(np.sum(overdue_values[paired]) / denominator),
                    "paired_count": int(np.sum(paired)),
                }
    return {"loan_amount": loan, "overdue_amount": overdue, "overdue_rate": rate}


def _amount_measure(
    mask: np.ndarray, values: np.ndarray | None, name: str
) -> dict[str, Any]:
    if values is None:
        return {"status": "unavailable", "reason": f"{name}_not_configured"}
    covered = mask & np.isfinite(values)
    selected_count = int(np.sum(mask))
    if selected_count == 0:
        return {
            "status": "available",
            "sum": 0.0,
            "covered_count": 0,
            "coverage_rate": 1.0,
        }
    if not np.any(covered):
        return {
            "status": "unavailable",
            "reason": "no_covered_rows",
            "coverage_rate": 0.0,
        }
    return {
        "status": "available",
        "sum": float(np.sum(values[covered])),
        "covered_count": int(np.sum(covered)),
        "coverage_rate": float(np.sum(covered) / selected_count),
    }


def _assign_cumulative_ks(
    rows: list[dict[str, Any]], *, total_bad: int, total_good: int
) -> None:
    by_rate: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        if row["count"] and row["bad_rate"] is not None:
            by_rate.setdefault(float(row["bad_rate"]), []).append(row)
    cumulative_bad = 0
    cumulative_good = 0
    for rate in sorted(by_rate):
        tied = by_rate[rate]
        cumulative_bad += sum(int(row["bad"]) for row in tied)
        cumulative_good += sum(int(row["good"]) for row in tied)
        ks = abs(cumulative_bad / total_bad - cumulative_good / total_good)
        for row in tied:
            row["cumulative_ks"] = ks


def _risk_direction(rates: Sequence[float], *, method: str) -> str:
    if method == "categorical":
        return "unordered"
    if len(rates) <= 1 or all(value == rates[0] for value in rates):
        return "flat"
    if all(
        left <= right for left, right in zip(rates, rates[1:], strict=False)
    ):
        return "increasing"
    if all(
        left >= right for left, right in zip(rates, rates[1:], strict=False)
    ):
        return "decreasing"
    return "non_monotonic"


def _smoothing(analysis: Mapping[str, Any]) -> float:
    parameters = analysis.get("parameters")
    if not isinstance(parameters, Mapping):
        raise CandidateAssetError("parent analysis.parameters must be an object")
    return _number(parameters.get("smoothing"), "parent smoothing", 1e-12, None)


def _normalize_asset_body(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected = _TOP_LEVEL_FIELDS - {"asset_id", "asset_hash"}
    _exact_fields(payload, expected, "candidate asset body")
    if payload["schema_version"] != CANDIDATE_ASSET_SCHEMA_VERSION:
        raise CandidateAssetError(
            f"schema_version must be {CANDIDATE_ASSET_SCHEMA_VERSION}"
        )
    if payload["asset_type"] != "univariate_refinement":
        raise CandidateAssetError("asset_type must be univariate_refinement")
    if payload["effect_stage"] != "development":
        raise CandidateAssetError("effect_stage must be development")
    if payload["validation_status"] != "unvalidated":
        raise CandidateAssetError("validation_status must remain unvalidated")
    parent = _normalize_parent(payload["parent"])
    feature = _text(payload["feature"], "feature")
    method = _text(payload["method"], "method")
    refinement = _normalize_refinement(
        payload["refinement"], feature=feature, method=method
    )
    selection = _normalize_asset_selection(payload["selection"], refinement["bins"])
    reason = _optional_text(payload["selection_reason"], "selection_reason")
    rule = _normalize_rule(
        payload["rule"],
        parent_hash=parent["evidence_hash"],
        feature=feature,
        method=method,
        bins=refinement["bins"],
        selection=selection,
    )
    effect = _normalize_effect(
        payload["effect"], parent_hash=parent["evidence_hash"], rule_id=rule["rule_id"]
    )
    metrics = _normalize_metrics(payload["metrics"])
    _assert_effect_consistency(
        effect,
        bins=refinement["bins"],
        selected_bin_ids=selection["selected_bin_ids"],
    )
    expected_metrics = _candidate_metrics(
        bins=refinement["bins"],
        method_metrics=refinement["metrics"],
        effect=effect,
    )
    if metrics != expected_metrics:
        raise CandidateAssetError(
            "metrics must match the canonical refinement and selected effect"
        )
    producer = _text(payload["producer_version"], "producer_version")
    if producer != PRODUCER_VERSION:
        raise CandidateAssetError(f"producer_version must be {PRODUCER_VERSION}")
    return {
        "schema_version": CANDIDATE_ASSET_SCHEMA_VERSION,
        "asset_type": "univariate_refinement",
        "effect_stage": "development",
        "validation_status": "unvalidated",
        "parent": parent,
        "feature": feature,
        "method": method,
        "refinement": refinement,
        "selection": selection,
        "selection_reason": reason,
        "rule": rule,
        "effect": effect,
        "metrics": metrics,
        "producer_version": PRODUCER_VERSION,
    }


def _normalize_parent(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateAssetError("parent must be an object")
    _exact_fields(value, _PARENT_FIELDS, "parent")
    candidate_id = _text(value["candidate_id"], "parent.candidate_id")
    if not re.fullmatch(r"candidate-[0-9a-f]{32}", candidate_id):
        raise CandidateAssetError("parent.candidate_id has an invalid format")
    return {
        "candidate_id": candidate_id,
        "evidence_hash": _hash(value["evidence_hash"], "parent.evidence_hash"),
        "source_evidence": _normalize_source_evidence(value["source_evidence"]),
    }


def _normalize_source_evidence(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise CandidateAssetError("source_evidence must be an object")
    _exact_fields(value, _SOURCE_FIELDS, "source_evidence")
    return {
        "artifact_id": _text(value["artifact_id"], "source_evidence.artifact_id"),
        "kind": _text(value["kind"], "source_evidence.kind"),
        "content_hash": _hash(value["content_hash"], "source_evidence.content_hash"),
    }


def _normalize_refinement(
    value: object, *, feature: str, method: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateAssetError("refinement must be an object")
    _exact_fields(value, _REFINEMENT_FIELDS, "refinement")
    source_count = _integer(value["source_bin_count"], "source_bin_count", 1)
    edited_count = _integer(value["edited_bin_count"], "edited_bin_count", 1)
    smoothing = _number(value["smoothing"], "refinement.smoothing", 1e-12, None)
    bins_raw = value["bins"]
    if not _sequence(bins_raw):
        raise CandidateAssetError("refinement.bins must be a non-empty array")
    bins = [
        _normalize_bin(item, index=index, feature=feature, method=method)
        for index, item in enumerate(bins_raw)
    ]
    all_source_ids = [source_id for row in bins for source_id in row["source_bin_ids"]]
    if len(set(all_source_ids)) != len(all_source_ids):
        raise CandidateAssetError(
            "refinement bins must partition unique source bin ids"
        )
    if source_count != len(all_source_ids) or edited_count != len(bins):
        raise CandidateAssetError("refinement bin counts are inconsistent")
    merge_groups_raw = value["merge_groups"]
    if not _sequence(merge_groups_raw, allow_empty=True):
        raise CandidateAssetError("refinement.merge_groups must be an array")
    merge_groups = []
    for item in merge_groups_raw:
        if not _sequence(item) or len(item) < 2:
            raise CandidateAssetError("refinement merge groups must contain two ids")
        merge_groups.append([_text(source_id, "merge source id") for source_id in item])
    expected_groups = [
        row["source_bin_ids"] for row in bins if len(row["source_bin_ids"]) > 1
    ]
    if merge_groups != expected_groups:
        raise CandidateAssetError("refinement.merge_groups does not match edited bins")
    metrics = _normalize_method_metrics(value["metrics"])
    _assert_refinement_consistency(
        bins,
        metrics=metrics,
        smoothing=smoothing,
        method=method,
    )
    return {
        "source_bin_count": source_count,
        "edited_bin_count": edited_count,
        "merge_groups": merge_groups,
        "smoothing": smoothing,
        "metrics": metrics,
        "bins": bins,
    }


def _assert_refinement_consistency(
    bins: Sequence[Mapping[str, Any]],
    *,
    metrics: Mapping[str, Any],
    smoothing: float,
    method: str,
) -> None:
    total_count = sum(int(row["count"]) for row in bins)
    total_good = sum(int(row["good"]) for row in bins)
    total_bad = sum(int(row["bad"]) for row in bins)
    if total_count <= 0 or total_good <= 0 or total_bad <= 0:
        raise CandidateAssetError("refinement bins must cover both target classes")
    base_bad_rate = total_bad / total_count
    group_count = len(bins)
    total_iv = 0.0
    for row in bins:
        expected_share = row["count"] / total_count
        if not _close(row["share"], expected_share):
            raise CandidateAssetError("bin.share is inconsistent with total count")
        expected_woe, expected_iv = _smoothed_woe_iv(
            int(row["bad"]),
            int(row["good"]),
            total_bad,
            total_good,
            group_count,
            smoothing=smoothing,
        )
        if not _close(row["woe"], expected_woe) or not _close(
            row["iv_contribution"], expected_iv
        ):
            raise CandidateAssetError("bin WOE/IV must be recomputed after merging")
        total_iv += expected_iv
        expected_lift = (
            None if row["bad_rate"] is None else float(row["bad_rate"]) / base_bad_rate
        )
        if not _optional_close(row["lift"], expected_lift):
            raise CandidateAssetError("bin.lift is inconsistent")

    expected_cumulative = _aggregate_cumulative_ks(bins, total_bad, total_good)
    for row, expected in zip(bins, expected_cumulative, strict=True):
        if not _close(row["cumulative_ks"], expected):
            raise CandidateAssetError("bin.cumulative_ks is inconsistent")
    expected_ks = max(expected_cumulative, default=0.0)
    expected_auc = _aggregate_auc(bins, total_bad=total_bad, total_good=total_good)
    if not _close(metrics["iv"], total_iv):
        raise CandidateAssetError(
            "refinement IV must equal recomputed bin contributions"
        )
    if not _close(metrics["ks"], expected_ks):
        raise CandidateAssetError("refinement KS must be recomputed from edited bins")
    if not _close(metrics["auc"], expected_auc):
        raise CandidateAssetError("refinement AUC must be recomputed from edited bins")
    ordinary_rates = [
        float(row["bad_rate"])
        for row in bins
        if row["kind"] == "numeric_interval" and row["bad_rate"] is not None
    ]
    if metrics["risk_direction"] != _risk_direction(ordinary_rates, method=method):
        raise CandidateAssetError("refinement risk_direction is inconsistent")
    missing_rate = next(
        (float(row["share"]) for row in bins if row["kind"] == "missing"), 0.0
    )
    if not _close(metrics["missing_rate"], missing_rate):
        raise CandidateAssetError("refinement missing_rate is inconsistent")
    _assert_aggregate_amount_metrics(
        metrics["amount_metrics"], bins=bins, selected_count=total_count
    )


def _aggregate_cumulative_ks(
    bins: Sequence[Mapping[str, Any]], total_bad: int, total_good: int
) -> list[float]:
    by_rate: dict[float, list[int]] = {}
    for index, row in enumerate(bins):
        if row["count"] and row["bad_rate"] is not None:
            by_rate.setdefault(float(row["bad_rate"]), []).append(index)
    result = [0.0] * len(bins)
    cumulative_bad = 0
    cumulative_good = 0
    for rate in sorted(by_rate):
        indices = by_rate[rate]
        cumulative_bad += sum(int(bins[index]["bad"]) for index in indices)
        cumulative_good += sum(int(bins[index]["good"]) for index in indices)
        value = abs(cumulative_bad / total_bad - cumulative_good / total_good)
        for index in indices:
            result[index] = value
    return result


def _aggregate_auc(
    bins: Sequence[Mapping[str, Any]], *, total_bad: int, total_good: int
) -> float:
    by_rate: dict[float, tuple[int, int]] = {}
    for row in bins:
        if not row["count"] or row["bad_rate"] is None:
            continue
        rate = float(row["bad_rate"])
        bad, good = by_rate.get(rate, (0, 0))
        by_rate[rate] = (bad + int(row["bad"]), good + int(row["good"]))
    cumulative_good = 0
    numerator = 0.0
    for rate in sorted(by_rate):
        bad, good = by_rate[rate]
        numerator += bad * cumulative_good + 0.5 * bad * good
        cumulative_good += good
    return numerator / (total_bad * total_good)


def _assert_effect_consistency(
    effect: Mapping[str, Any],
    *,
    bins: Sequence[Mapping[str, Any]],
    selected_bin_ids: Sequence[str],
) -> None:
    selected = [row for row in bins if row["bin_id"] in set(selected_bin_ids)]
    total_count = sum(int(row["count"]) for row in bins)
    total_bad = sum(int(row["bad"]) for row in bins)
    count = sum(int(row["count"]) for row in selected)
    good = sum(int(row["good"]) for row in selected)
    bad = sum(int(row["bad"]) for row in selected)
    bad_rate = bad / count
    base_bad_rate = total_bad / total_count
    expected = {
        "selected_count": count,
        "selected_share": count / total_count,
        "good": good,
        "bad": bad,
        "bad_rate": bad_rate,
        "lift": bad_rate / base_bad_rate,
    }
    for name in ("selected_count", "good", "bad"):
        if effect[name] != expected[name]:
            raise CandidateAssetError(f"effect.{name} does not match selected bins")
    for name in ("selected_share", "bad_rate", "lift"):
        if not _close(effect[name], expected[name]):
            raise CandidateAssetError(f"effect.{name} does not match selected bins")
    _assert_aggregate_amount_metrics(
        effect["amount_metrics"], bins=selected, selected_count=count
    )


def _assert_aggregate_amount_metrics(
    value: Mapping[str, Any],
    *,
    bins: Sequence[Mapping[str, Any]],
    selected_count: int,
) -> None:
    for dimension in ("loan_amount", "overdue_amount"):
        measures = [row["amount_metrics"][dimension] for row in bins]
        expected = _aggregate_amount_measure(
            measures,
            selected_count=selected_count,
            dimension=dimension,
        )
        if not _amount_measure_matches(value[dimension], expected):
            raise CandidateAssetError(
                f"aggregate {dimension} metrics do not match edited bins"
            )


def _aggregate_amount_measure(
    measures: Sequence[Mapping[str, Any]],
    *,
    selected_count: int,
    dimension: str,
) -> dict[str, Any]:
    available = [item for item in measures if item["status"] == "available"]
    configured = any(
        item["status"] == "available"
        or item.get("reason") != f"{dimension}_not_configured"
        for item in measures
    )
    if not configured:
        return {"status": "unavailable", "reason": f"{dimension}_not_configured"}
    covered_count = sum(int(item["covered_count"]) for item in available)
    if covered_count == 0:
        return {
            "status": "unavailable",
            "reason": "no_covered_rows",
            "coverage_rate": 0.0,
        }
    return {
        "status": "available",
        "sum": float(sum(float(item["sum"]) for item in available)),
        "covered_count": covered_count,
        "coverage_rate": covered_count / selected_count,
    }


def _amount_measure_matches(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    if set(actual) != set(expected):
        return False
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if key in {"sum", "coverage_rate"}:
            if not _close(actual_value, expected_value):
                return False
        elif actual_value != expected_value:
            return False
    return True


def _close(left: object, right: object) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)


def _optional_close(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return _close(left, right)


def _normalize_method_metrics(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateAssetError("refinement.metrics must be an object")
    _exact_fields(value, _METHOD_METRIC_FIELDS, "refinement.metrics")
    direction = _text(value["risk_direction"], "risk_direction")
    if direction not in {
        "increasing",
        "decreasing",
        "flat",
        "non_monotonic",
        "unordered",
    }:
        raise CandidateAssetError("risk_direction is unsupported")
    return {
        "iv": _number(value["iv"], "refinement.metrics.iv", 0, None),
        "ks": _number(value["ks"], "refinement.metrics.ks", 0, 1),
        "auc": _number(value["auc"], "refinement.metrics.auc", 0, 1),
        "risk_direction": direction,
        "missing_rate": _number(value["missing_rate"], "missing_rate", 0, 1),
        "amount_metrics": _normalize_amount_metrics(value["amount_metrics"]),
    }


def _normalize_bin(
    value: object, *, index: int, feature: str, method: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateAssetError("refinement bins must contain objects")
    _exact_fields(value, _BIN_FIELDS, f"refinement.bins[{index}]")
    actual_index = _integer(value["index"], "bin.index", 0)
    if actual_index != index:
        raise CandidateAssetError("refinement bin indices must be contiguous")
    source_ids_raw = value["source_bin_ids"]
    if not _sequence(source_ids_raw):
        raise CandidateAssetError("bin.source_bin_ids must be non-empty")
    source_ids = [_text(item, "bin source id") for item in source_ids_raw]
    if len(set(source_ids)) != len(source_ids):
        raise CandidateAssetError("bin.source_bin_ids must be unique")
    bin_id = _text(value["bin_id"], "bin.bin_id")
    if not _BIN_ID_RE.fullmatch(bin_id) or bin_id != _stable_id(
        "candidate-bin",
        {"feature": feature, "method": method, "source_bin_ids": source_ids},
    ):
        raise CandidateAssetError("bin_id does not match its canonical source identity")
    kind = _text(value["kind"], "bin.kind")
    if kind not in {"numeric_interval", "category", "sentinel", "missing"}:
        raise CandidateAssetError("bin.kind is unsupported")
    _assert_source_ids_match_kind(source_ids, kind=kind)
    condition_raw = value["condition"]
    if not isinstance(condition_raw, Mapping):
        raise CandidateAssetError("bin.condition must be an object")
    condition = canonicalize_expression(condition_raw)
    if _canonical_json(condition) != _canonical_json(condition_raw):
        raise CandidateAssetError("bin.condition must be canonical Strategy DSL")
    if _expression_fields(condition) != {feature}:
        raise CandidateAssetError("bin.condition must reference only its feature")
    if method == "categorical" and kind in {"category", "sentinel"}:
        _assert_strict_category_condition(condition)
    count = _integer(value["count"], "bin.count", 0)
    good = _integer(value["good"], "bin.good", 0)
    bad = _integer(value["bad"], "bin.bad", 0)
    if count != good + bad:
        raise CandidateAssetError("bin count must equal good plus bad")
    bad_rate = _optional_number(value["bad_rate"], "bin.bad_rate", 0, 1)
    if (count == 0) != (bad_rate is None):
        raise CandidateAssetError(
            "bin.bad_rate nullability must match empty-bin status"
        )
    if count and not math.isclose(bad_rate, bad / count, rel_tol=1e-12, abs_tol=1e-12):
        raise CandidateAssetError("bin.bad_rate is inconsistent")
    return {
        "index": index,
        "bin_id": bin_id,
        "source_bin_ids": source_ids,
        "kind": kind,
        "condition": condition,
        "count": count,
        "share": _number(value["share"], "bin.share", 0, 1),
        "good": good,
        "bad": bad,
        "bad_rate": bad_rate,
        "woe": _number(value["woe"], "bin.woe", None, None),
        "iv_contribution": _number(
            value["iv_contribution"], "bin.iv_contribution", 0, None
        ),
        "lift": _optional_number(value["lift"], "bin.lift", 0, None),
        "cumulative_ks": _number(value["cumulative_ks"], "bin.cumulative_ks", 0, 1),
        "amount_metrics": _normalize_amount_metrics(value["amount_metrics"]),
    }


def _assert_source_ids_match_kind(source_ids: Sequence[str], *, kind: str) -> None:
    expected_prefix = {
        "numeric_interval": "regular:",
        "category": "category:",
        "sentinel": "sentinel:",
    }.get(kind)
    if kind == "missing":
        if list(source_ids) != ["missing"]:
            raise CandidateAssetError("a missing edited bin must remain singleton")
        return
    assert expected_prefix is not None
    positions = []
    for source_id in source_ids:
        if not source_id.startswith(expected_prefix):
            raise CandidateAssetError("bin source ids do not match their bin kind")
        suffix = source_id.removeprefix(expected_prefix)
        if not suffix.isdigit() or str(int(suffix)) != suffix:
            raise CandidateAssetError("bin source id has an invalid canonical index")
        positions.append(int(suffix))
    if positions != sorted(positions):
        raise CandidateAssetError("bin source ids must use fixed source order")
    if kind == "numeric_interval" and positions != list(
        range(positions[0], positions[-1] + 1)
    ):
        raise CandidateAssetError(
            "numeric edited bins must contain adjacent source bins"
        )


def _assert_strict_category_condition(expression: Mapping[str, Any]) -> None:
    op = expression["op"]
    if op == "compare":
        if (
            expression["operator"] not in {"==", "in"}
            or expression.get("coercion") != "strict"
        ):
            raise CandidateAssetError(
                "categorical bins must preserve strict scalar equality"
            )
        return
    if op == "or":
        for argument in expression["args"]:
            _assert_strict_category_condition(argument)
        return
    raise CandidateAssetError("categorical bins must use strict equality OR fragments")


def _normalize_asset_selection(
    value: object, bins: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateAssetError("selection must be an object")
    mode = value.get("mode")
    expected = (
        frozenset({"mode", "source_bin_ids", "selected_bin_ids"})
        if mode == "source_bin_ids"
        else frozenset({"mode", "risk_threshold", "source_bin_ids", "selected_bin_ids"})
    )
    _exact_fields(value, expected, "selection")
    source_ids = [
        _text(item, "selection source id")
        for item in _required_sequence(
            value["source_bin_ids"], "selection.source_bin_ids"
        )
    ]
    selected_ids = [
        _text(item, "selection bin id")
        for item in _required_sequence(
            value["selected_bin_ids"], "selection.selected_bin_ids"
        )
    ]
    by_id = {row["bin_id"]: row for row in bins}
    if any(item not in by_id for item in selected_ids) or len(set(selected_ids)) != len(
        selected_ids
    ):
        raise CandidateAssetError("selection.selected_bin_ids are invalid")
    ordered_selected = [row for row in bins if row["bin_id"] in set(selected_ids)]
    expected_source = [
        source_id for row in ordered_selected for source_id in row["source_bin_ids"]
    ]
    if (
        selected_ids != [row["bin_id"] for row in ordered_selected]
        or source_ids != expected_source
    ):
        raise CandidateAssetError("selection ids must use fixed source order")
    if mode == "source_bin_ids":
        return {
            "mode": mode,
            "source_bin_ids": source_ids,
            "selected_bin_ids": selected_ids,
        }
    if mode != "risk_threshold":
        raise CandidateAssetError("selection.mode is unsupported")
    threshold = value["risk_threshold"]
    if not isinstance(threshold, Mapping):
        raise CandidateAssetError("selection.risk_threshold must be an object")
    _exact_fields(
        threshold,
        frozenset({"metric", "operator", "value"}),
        "selection.risk_threshold",
    )
    if threshold["metric"] != "bad_rate":
        raise CandidateAssetError("risk_threshold.metric must be bad_rate")
    operator = _text(threshold["operator"], "risk_threshold.operator")
    if operator not in {">=", ">", "<=", "<"}:
        raise CandidateAssetError("risk_threshold.operator is unsupported")
    threshold_value = _number(threshold["value"], "risk_threshold.value", 0, 1)
    actually_selected = [
        row
        for row in bins
        if row["bad_rate"] is not None
        and _threshold_match(float(row["bad_rate"]), operator, threshold_value)
    ]
    if [row["bin_id"] for row in actually_selected] != selected_ids:
        raise CandidateAssetError("risk_threshold selection does not match bin risk")
    return {
        "mode": mode,
        "risk_threshold": {
            "metric": "bad_rate",
            "operator": operator,
            "value": threshold_value,
        },
        "source_bin_ids": source_ids,
        "selected_bin_ids": selected_ids,
    }


def _normalize_rule(
    value: object,
    *,
    parent_hash: str,
    feature: str,
    method: str,
    bins: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateAssetError("rule must be an object")
    _exact_fields(value, _RULE_FIELDS, "rule")
    selected_ids = list(selection["selected_bin_ids"])
    by_id = {row["bin_id"]: row for row in bins}
    expected_condition = canonicalize_expression(
        {"op": "or", "args": [by_id[item]["condition"] for item in selected_ids]}
    )
    condition_raw = value["condition"]
    if not isinstance(condition_raw, Mapping):
        raise CandidateAssetError("rule.condition must be an object")
    condition = canonicalize_expression(condition_raw)
    source_ids = list(selection["source_bin_ids"])
    if (
        value["selected_bin_ids"] != selected_ids
        or value["source_bin_ids"] != source_ids
    ):
        raise CandidateAssetError("rule ids must match selection")
    if _canonical_json(condition) != _canonical_json(expected_condition):
        raise CandidateAssetError(
            "rule.condition must be the fixed-order selected-bin OR"
        )
    without_id = {
        "condition": condition,
        "selected_bin_ids": selected_ids,
        "source_bin_ids": source_ids,
    }
    rule_id = _text(value["rule_id"], "rule.rule_id")
    expected_id = _stable_id(
        "candidate-rule",
        {
            "parent_evidence_hash": parent_hash,
            "feature": feature,
            "method": method,
            **without_id,
        },
    )
    if not _RULE_ID_RE.fullmatch(rule_id) or rule_id != expected_id:
        raise CandidateAssetError("rule_id does not match canonical rule identity")
    return {"rule_id": rule_id, **without_id}


def _normalize_effect(
    value: object, *, parent_hash: str, rule_id: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateAssetError("effect must be an object")
    _exact_fields(value, _EFFECT_FIELDS, "effect")
    count = _integer(value["selected_count"], "effect.selected_count", 1)
    good = _integer(value["good"], "effect.good", 0)
    bad = _integer(value["bad"], "effect.bad", 0)
    if count != good + bad:
        raise CandidateAssetError("effect count must equal good plus bad")
    bad_rate = _number(value["bad_rate"], "effect.bad_rate", 0, 1)
    if not math.isclose(bad_rate, bad / count, rel_tol=1e-12, abs_tol=1e-12):
        raise CandidateAssetError("effect.bad_rate is inconsistent")
    body = {
        "selected_count": count,
        "selected_share": _number(
            value["selected_share"], "effect.selected_share", 0, 1
        ),
        "good": good,
        "bad": bad,
        "bad_rate": bad_rate,
        "lift": _number(value["lift"], "effect.lift", 0, None),
        "amount_metrics": _normalize_amount_metrics(value["amount_metrics"]),
    }
    effect_id = _text(value["effect_id"], "effect.effect_id")
    expected_id = _stable_id(
        "candidate-effect",
        {"parent_evidence_hash": parent_hash, "rule_id": rule_id, **body},
    )
    if not _EFFECT_ID_RE.fullmatch(effect_id) or effect_id != expected_id:
        raise CandidateAssetError("effect_id does not match canonical effect identity")
    return {"effect_id": effect_id, **body}


def _normalize_metrics(value: object) -> list[dict[str, Any]]:
    if not _sequence(value):
        raise CandidateAssetError("metrics must be a non-empty array")
    normalized = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise CandidateAssetError("metrics must contain objects")
        _exact_fields(item, _METRIC_FIELDS, f"metrics[{index}]")
        name = _text(item["metric_name"], "metric_name")
        dimension = _text(item["dimension"], "metric dimension")
        status = _text(item["status"], "metric status")
        if dimension not in _DIMENSIONS or status not in _METRIC_STATUSES:
            raise CandidateAssetError("metric dimension or status is unsupported")
        metric_value = item["value"]
        if status == "observed":
            metric_value = _number(metric_value, "metric value", None, None)
        elif metric_value is not None:
            raise CandidateAssetError("non-observed metric value must be null")
        normalized.append(
            {
                "metric_name": name,
                "dimension": dimension,
                "status": status,
                "value": metric_value,
            }
        )
    expected = sorted(
        normalized,
        key=lambda item: (item["metric_name"], _DIMENSIONS.index(item["dimension"])),
    )
    if normalized != expected:
        raise CandidateAssetError("metrics must use canonical name and dimension order")
    identities = [(item["metric_name"], item["dimension"]) for item in normalized]
    if len(set(identities)) != len(identities):
        raise CandidateAssetError("metrics contain duplicate identities")
    grouped: dict[str, set[str]] = {}
    for item in normalized:
        grouped.setdefault(item["metric_name"], set()).add(item["dimension"])
    if any(dimensions != set(_DIMENSIONS) for dimensions in grouped.values()):
        raise CandidateAssetError("every metric must explicitly cover all dimensions")
    return normalized


def _normalize_amount_metrics(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateAssetError("amount_metrics must be an object")
    _exact_fields(value, _AMOUNT_FIELDS, "amount_metrics")
    return {
        "loan_amount": _normalize_amount_measure(value["loan_amount"]),
        "overdue_amount": _normalize_amount_measure(value["overdue_amount"]),
        "overdue_rate": _normalize_overdue_rate(value["overdue_rate"]),
    }


def _normalize_amount_measure(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateAssetError("amount measure must be an object")
    status = value.get("status")
    if status == "available":
        _exact_fields(
            value,
            frozenset({"status", "sum", "covered_count", "coverage_rate"}),
            "amount measure",
        )
        return {
            "status": "available",
            "sum": _number(value["sum"], "amount sum", 0, None),
            "covered_count": _integer(value["covered_count"], "covered_count", 0),
            "coverage_rate": _number(value["coverage_rate"], "coverage_rate", 0, 1),
        }
    if status == "unavailable":
        allowed = (
            frozenset({"status", "reason", "coverage_rate"})
            if "coverage_rate" in value
            else frozenset({"status", "reason"})
        )
        _exact_fields(value, allowed, "amount measure")
        result = {
            "status": "unavailable",
            "reason": _text(value["reason"], "amount reason"),
        }
        if "coverage_rate" in value:
            result["coverage_rate"] = _number(
                value["coverage_rate"], "coverage_rate", 0, 1
            )
        return result
    raise CandidateAssetError("amount measure status is unsupported")


def _normalize_overdue_rate(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateAssetError("overdue_rate must be an object")
    status = value.get("status")
    if status == "available":
        _exact_fields(
            value, frozenset({"status", "value", "paired_count"}), "overdue_rate"
        )
        return {
            "status": "available",
            "value": _number(value["value"], "overdue_rate.value", 0, None),
            "paired_count": _integer(value["paired_count"], "paired_count", 1),
        }
    if status in {"unavailable", "not_applicable"}:
        _exact_fields(value, frozenset({"status", "reason"}), "overdue_rate")
        return {
            "status": status,
            "reason": _text(value["reason"], "overdue_rate.reason"),
        }
    raise CandidateAssetError("overdue_rate status is unsupported")


def _expression_fields(expression: Mapping[str, Any]) -> set[str]:
    op = expression["op"]
    if op in {"compare", "between", "is_null", "is_not_null"}:
        return {str(expression["field"])}
    if op in {"and", "or", "n_of_k"}:
        return set().union(*(_expression_fields(item) for item in expression["args"]))
    if op == "not":
        return _expression_fields(expression["arg"])
    raise CandidateAssetError(f"unsupported expression op: {op}")


def _stable_id(prefix: str, value: Mapping[str, Any]) -> str:
    return f"{prefix}-{_sha256(_canonical_json(value))[:32]}"


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise CandidateAssetError(
            "candidate asset must be finite canonical JSON"
        ) from exc


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash(value: object, name: str) -> str:
    text = _text(value, name)
    if not _HASH_RE.fullmatch(text):
        raise CandidateAssetError(f"{name} must be a lowercase SHA-256 hash")
    return text


def _exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], name: str
) -> None:
    keys = set(value)
    unknown = sorted(keys - expected)
    missing = sorted(expected - keys)
    if unknown:
        raise CandidateAssetError(f"{name} has unknown fields: {', '.join(unknown)}")
    if missing:
        raise CandidateAssetError(f"{name} is missing fields: {', '.join(missing)}")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CandidateAssetError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _integer(value: object, name: str, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise CandidateAssetError(f"{name} must be an integer >= {minimum}")
    return value


def _number(
    value: object,
    name: str,
    minimum: float | None,
    maximum: float | None,
) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise CandidateAssetError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise CandidateAssetError(f"{name} must be a finite number")
    if minimum is not None and result < minimum:
        raise CandidateAssetError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise CandidateAssetError(f"{name} must be <= {maximum}")
    return result


def _optional_number(
    value: object,
    name: str,
    minimum: float | None,
    maximum: float | None,
) -> float | None:
    if value is None:
        return None
    return _number(value, name, minimum, maximum)


def _sequence(value: object, *, allow_empty: bool = False) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str | bytes | bytearray)
        and (allow_empty or bool(value))
    )


def _required_sequence(value: object, name: str) -> Sequence[Any]:
    if not _sequence(value):
        raise CandidateAssetError(f"{name} must be a non-empty array")
    assert isinstance(value, Sequence)
    return value


__all__ = [
    "CANDIDATE_ASSET_SCHEMA_VERSION",
    "PRODUCER_VERSION",
    "CandidateAssetError",
    "canonical_candidate_asset_json",
    "refine_univariate_candidate",
    "validate_candidate_asset",
    "validate_candidate_refinement_source_controls",
]
