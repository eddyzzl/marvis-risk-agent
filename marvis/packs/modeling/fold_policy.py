"""Bind fold selection to native source, policy and full candidate identities."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from marvis.data.preprocessing_evidence import load_preprocessing_state
from marvis.feature.transform import normalize_sentinel_value
from marvis.packs.modeling.errors import ModelingError
from marvis.packs.modeling.fold_selection import FoldSelectionSession
from marvis.packs.modeling.selection_evidence import (
    _selection_source_mapping,
    load_selection_evidence,
    selection_evidence_for_training,
)
from marvis.packs.modeling.special_value_tools import special_value_decision_fingerprint


VERSION = "marvis.fold_selection_plan.v1"
_SCREEN_KEYS = frozenset({"leakage_ks", "max_missing_rate", "top_k", "batch_size", "drop_nan_labels"})
_SELECT_KEYS = frozenset({"iv_min", "corr_max", "vif_max", "top_k", "seed", "drop_nan_labels",
    "space", "scorecard_max_bins", "enforce_monotonic", "monotonic_direction_request",
    "sign_check", "batch_size", "multivariate_sample_rows"})


def plan_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def build_fold_selection_plan(registry, task_id, inputs):
    """Authenticate a requested plan before cache use and again in the worker.

    Earlier outer selection receipts supply parameter provenance, not the
    candidate subset or a claim of independent upstream decisions.
    """
    requested = inputs.get("fold_selection")
    if requested is None:
        return None
    if not isinstance(requested, dict) or set(requested) != {"source_dataset_id", "candidates"}:
        raise ModelingError("fold_selection requires source_dataset_id and complete candidates")
    source = registry.get(requested["source_dataset_id"])
    current = registry.get(inputs["dataset_id"])
    if source.task_id != task_id or current.task_id != task_id:
        raise ModelingError("fold selection source belongs to another task")
    mapped = _selection_source_mapping(registry, current.id, source.id, ())
    if (mapped is None or mapped[0] is None or source.row_count != current.row_count
            or not np.array_equal(mapped[0], np.arange(source.row_count))):
        raise ModelingError("fold source and training data require an authenticated identity row map")
    candidates = requested["candidates"]
    if (not isinstance(candidates, list) or not candidates
            or any(not isinstance(value, str) or not value for value in candidates)
            or len(set(candidates)) != len(candidates)):
        raise ModelingError("fold selection requires a complete ordered candidate list")
    references = inputs.get("selection_evidence_refs") or []
    # Verify every reference, including lineage and target, before reading its
    # effective parameters. An unrelated extra reference cannot be ignored.
    exposure = selection_evidence_for_training(registry, task_id, current.id,
        candidates, inputs["target_col"], references)
    receipts = [load_selection_evidence(registry, task_id, ref) for ref in references]
    screens = [item for item in receipts if item["tool"] in {
        "modeling.screen_features", "modeling.screen_features_non_binary"}]
    refinements = [item for item in receipts if item["tool"] == "modeling.select_features"]
    if len(screens) != 1 or len(refinements) != 1:
        raise ModelingError("fold selection requires exactly one native screen and refinement receipt")
    screen, refinement = screens[0], refinements[0]
    if (screen["dataset_id"] != source.id or screen["candidates"] != candidates
            or refinement["dataset_id"] != current.id):
        raise ModelingError("fold candidates or source differ from their native selection receipts")
    screen_parameters = {key: value for key, value in screen["parameters"].items() if key in _SCREEN_KEYS}
    selection_parameters = {key: value for key, value in refinement["parameters"].items() if key in _SELECT_KEYS}
    if screen_parameters.get("top_k") is None:
        screen_parameters["top_k"] = 200
    # Label confirmation is one explicit runtime policy, not a stale setting
    # copied from an earlier exploratory selection.
    for parameters in (screen_parameters, selection_parameters):
        parameters["drop_nan_labels"] = bool(inputs.get("drop_nan_labels"))
    governance = inputs.get("special_value_governance") or {}
    policies = {}
    for column, evidence in governance.items():
        if column not in candidates:
            continue
        if (not isinstance(evidence, dict) or evidence.get("column") != column
                or evidence.get("source_dataset_id") != source.id
                or evidence.get("source_dataset_content_hash") != source.content_hash
                or evidence.get("resolved_dataset_id") != current.id
                or evidence.get("decision_fingerprint") != special_value_decision_fingerprint(evidence)):
            raise ModelingError("fold special-value governance differs from its native source")
        policies[column] = {"action": evidence["action"],
            "values": [normalize_sentinel_value(value) for value in evidence["detected_values"]],
            "confirmed": evidence.get("confirmed") is True, "reason": str(evidence.get("reason") or "")}
    source_state = load_preprocessing_state(registry, source.id)
    current_state = load_preprocessing_state(registry, current.id)
    if current_state.steps[:len(source_state.steps)] != source_state.steps:
        raise ModelingError("training preprocessing does not extend the fold source")
    additions = current_state.steps[len(source_state.steps):]
    masks = {column: policy["values"] for column, policy in policies.items() if policy["action"] == "mask"}
    expected = [{"kind": "sentinel", "columns": sorted(masks), "params": masks}] if masks else []
    if additions != expected:
        raise ModelingError("fold source may differ from training data only by its exact frozen masks")
    plan = {"schema_version": VERSION, "source_dataset_id": source.id,
        "source_content_hash": source.content_hash, "training_dataset_id": current.id,
        "training_content_hash": current.content_hash, "row_count": source.row_count,
        "candidates": list(candidates), "target_col": inputs["target_col"],
        "split_col": inputs["split_col"], "split_values": dict(inputs["split_values"]),
        "sample_weight_col": str(inputs.get("sample_weight_col") or ""),
        "drop_nan_labels": bool(inputs.get("drop_nan_labels")),
        "screen_parameters": screen_parameters, "selection_parameters": selection_parameters,
        "special_value_policy": policies, "references": references,
        "ancestry_artifacts": mapped[1], "outer_selection_exposure": exposure,
        "round_aggregation": "ceil_median_positive_fold_best_iterations",
        "upstream_candidate_selection": "unknown", "historical_availability": "unknown"}
    plan["sha256"] = plan_digest(plan)
    return plan


def selection_session(registry, plan, *, target_type, group_columns=()):
    """Create a provider only from a reauthenticated effective plan."""
    return FoldSelectionSession(registry, plan["source_dataset_id"],
        candidates=plan["candidates"], target_col=plan["target_col"], target_type=target_type,
        sample_weight_col=plan["sample_weight_col"] or None,
        control_columns=[plan["split_col"], *group_columns],
        screen_parameters=plan["screen_parameters"], selection_parameters=plan["selection_parameters"],
        special_value_policy=plan["special_value_policy"])
