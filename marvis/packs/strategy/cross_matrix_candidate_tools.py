"""Governed Tool adapter for immutable two-dimensional Cross Matrix candidates.

The univariate CandidateEvidence artifact remains the authority for dataset,
sample, axis bins, and amount-column semantics.  This boundary reads that exact
sample once, replays every parent bin through the canonical vectorized Strategy
DSL evaluator, measures the complete Cartesian matrix, and atomically publishes
the kernel-owned asset.  It does not select cells or admit/adopt/deploy rules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import hmac
import json
import math
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd

from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.labels import resolve_labeled_frame
from marvis.output.strategy_candidate_report import (
    canonical_strategy_candidate_report_json,
    strategy_candidate_report_from_json,
)
from marvis.packs.strategy import candidate_asset_tools
from marvis.packs.strategy.candidate_evidence import validate_candidate_evidence
from marvis.packs.strategy.candidate_fragment import (
    sample_context_hash_from_candidate_evidence,
)
from marvis.packs.strategy.cross_matrix_candidate import (
    build_cross_matrix_candidate_asset,
    canonical_cross_matrix_candidate_asset_json,
    parse_cross_matrix_candidate_asset_json,
    rebuild_cross_matrix_candidate_asset,
    validate_cross_matrix_candidate_asset,
)
from marvis.packs.strategy.dsl import canonicalize_expression
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.evaluator import evaluate_expression_frame


TOOL_SCHEMA_VERSION = "strategy.build-cross-matrix-candidate-tool.v1"
ASSET_ARTIFACT_KIND = "strategy_cross_matrix_candidate_json"
ASSET_ARTIFACT_SCHEMA_VERSION = "strategy.cross-matrix-candidate-artifact.v1"
ORIGIN_TOOL = "strategy.build_cross_matrix_candidate"

# The analysis workflow currently supports at most 20 bins per axis.  Keeping
# the Cross Matrix budget at the complete 20 x 20 product makes the bound a
# platform policy, never an LLM/user-controlled argument.
CROSS_MATRIX_MAX_CELLS = 400

_INPUT_FIELDS = frozenset(
    {
        "source_artifact_id",
        "expected_artifact_content_hash",
        "expected_candidate_id",
        "expected_evidence_hash",
        "x_feature",
        "x_method",
        "y_feature",
        "y_method",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ASSET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ASSET_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "asset_schema_version",
        "asset_type",
        "asset_id",
        "asset_hash",
        "parent_candidate_id",
        "parent_evidence_hash",
        "candidate_id",
        "evidence_hash",
        "source_artifact_id",
        "source_artifact_content_hash",
        "task_id",
        "dataset_id",
        "dataset_content_hash",
        "registry_metadata_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "sample_context_hash",
        "target_col",
        "labeled_row_count",
        "row_axis",
        "column_axis",
        "cell_count",
        "candidate_stage",
        "observation_stage",
        "validation_status",
        "budget",
        "truncated",
    }
)


def run_build_cross_matrix_candidate(inputs, ctx, runtime) -> dict[str, Any]:
    """Build and persist one complete development-stage Cross Matrix asset."""

    normalized = _validate_inputs(inputs)
    if normalized["x_feature"] == normalized["y_feature"]:
        raise StrategyError("Cross Matrix axis features must be distinct")
    task_id = _text(ctx.task_id, "task_id")
    source = candidate_asset_tools._load_source_artifact(
        runtime,
        task_id=task_id,
        artifact_id=normalized["source_artifact_id"],
        expected_content_hash=normalized["expected_artifact_content_hash"],
        expected_candidate_id=normalized["expected_candidate_id"],
        expected_evidence_hash=normalized["expected_evidence_hash"],
    )
    evidence = _load_exact_parent_evidence(
        source,
        task_id=task_id,
        expected_candidate_id=normalized["expected_candidate_id"],
        expected_evidence_hash=normalized["expected_evidence_hash"],
    )
    dataset = candidate_asset_tools._load_dataset_binding(
        runtime,
        evidence=evidence,
        source=source,
    )
    row_axis = _resolve_axis(
        evidence,
        dataset=dataset,
        feature=normalized["x_feature"],
        method=normalized["x_method"],
        label="x",
    )
    column_axis = _resolve_axis(
        evidence,
        dataset=dataset,
        feature=normalized["y_feature"],
        method=normalized["y_method"],
        label="y",
    )
    cell_count = len(row_axis["bins"]) * len(column_axis["bins"])
    if cell_count > CROSS_MATRIX_MAX_CELLS:
        raise StrategyError(
            "Cross Matrix exceeds the platform cell budget "
            f"({cell_count} > {CROSS_MATRIX_MAX_CELLS})"
        )

    projection = _resolve_projection(
        evidence,
        dataset=dataset,
        row_feature=row_axis["feature"],
        column_feature=column_axis["feature"],
    )
    # This is deliberately the sole dataset-frame read in this Tool invocation.
    frame = runtime.backend.read_frame(dataset.path, columns=projection["columns"])
    population_count = len(frame)
    if population_count != dataset.row_count:
        raise StrategyError("Cross Matrix source dataset row count changed")
    candidate_asset_tools._require_dataset_unchanged(runtime, dataset)
    labeled = _resolve_exact_labeled_sample(
        frame,
        evidence=evidence,
        target_col=projection["target_col"],
        drop_nan_labels=projection["drop_nan_labels"],
        expected_dropped=projection["expected_dropped"],
    )

    candidate_asset_tools._require_source_unchanged(runtime, source)
    candidate_asset_tools._require_dataset_unchanged(runtime, dataset)
    target = _binary_target(labeled, projection["target_col"])
    row_index = _replay_axis(
        labeled,
        target=target,
        axis=row_axis,
        axis_label="row",
    )
    column_index = _replay_axis(
        labeled,
        target=target,
        axis=column_axis,
        axis_label="column",
    )
    measurement = _measure_matrix(
        labeled,
        evidence=evidence,
        target=target,
        row_axis=row_axis,
        column_axis=column_axis,
        row_index=row_index,
        column_index=column_index,
        loan_amount_col=projection["loan_amount_col"],
        overdue_amount_col=projection["overdue_amount_col"],
    )
    sample_identity = _sample_identity(evidence)

    candidate_asset_tools._require_source_unchanged(runtime, source)
    candidate_asset_tools._require_dataset_unchanged(runtime, dataset)
    asset = build_cross_matrix_candidate_asset(
        evidence,
        row_axis={"feature": row_axis["feature"], "method": row_axis["method"]},
        column_axis={
            "feature": column_axis["feature"],
            "method": column_axis["method"],
        },
        sample_identity=sample_identity,
        measurement=measurement,
        budget=CROSS_MATRIX_MAX_CELLS,
    )
    asset = validate_cross_matrix_candidate_asset(asset)
    if rebuild_cross_matrix_candidate_asset(asset, evidence) != asset:
        raise StrategyError("Cross Matrix asset does not rebuild against exact evidence")
    candidate_asset_tools._require_source_unchanged(runtime, source)
    candidate_asset_tools._require_dataset_unchanged(runtime, dataset)

    canonical = canonical_cross_matrix_candidate_asset_json(asset)
    content = canonical.encode("utf-8")
    if parse_cross_matrix_candidate_asset_json(content) != asset:
        raise StrategyError("Cross Matrix canonical asset JSON is not stable")
    lifecycle = _asset_lifecycle(asset)
    artifact = _persist_asset(
        runtime,
        task_id=task_id,
        source=source,
        dataset=dataset,
        evidence=evidence,
        asset=asset,
        row_axis=row_axis,
        column_axis=column_axis,
        cell_count=cell_count,
        content=content,
    )
    return {
        "schema_version": TOOL_SCHEMA_VERSION,
        "asset_id": asset["asset_id"],
        "asset_hash": asset["asset_hash"],
        "parent_candidate_id": evidence["candidate_id"],
        "parent_evidence_hash": evidence["evidence_hash"],
        "candidate_id": asset["candidate_evidence"]["candidate_id"],
        "evidence_hash": asset["candidate_evidence"]["evidence_hash"],
        "dataset_id": evidence["identity"]["dataset_id"],
        "target_col": projection["target_col"],
        "population_count": population_count,
        "labeled_count": len(labeled),
        "drop_nan_labels": projection["drop_nan_labels"],
        "nan_labels_dropped": projection["expected_dropped"],
        "row_axis": {
            "feature": row_axis["feature"],
            "method": row_axis["method"],
            "bin_count": len(row_axis["bins"]),
        },
        "column_axis": {
            "feature": column_axis["feature"],
            "method": column_axis["method"],
            "bin_count": len(column_axis["bins"]),
        },
        "cell_count": cell_count,
        "candidate_stage": lifecycle["candidate_stage"],
        "observation_stage": lifecycle["observation_stage"],
        "validation_status": lifecycle["validation_status"],
        "cross_matrix_candidate": asset,
        "artifacts": [artifact],
        "not_selected": True,
        "not_admitted": True,
        "not_applied": True,
        "not_adopted": True,
        "not_deployed": True,
    }


def _validate_inputs(inputs: object) -> dict[str, str]:
    if not isinstance(inputs, Mapping):
        raise StrategyError("build_cross_matrix_candidate inputs must be an object")
    if any(not isinstance(key, str) for key in inputs):
        raise StrategyError("build_cross_matrix_candidate input keys must be strings")
    missing = sorted(_INPUT_FIELDS - set(inputs))
    unsupported = sorted(set(inputs) - _INPUT_FIELDS)
    if missing or unsupported:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unsupported:
            details.append("unsupported: " + ", ".join(unsupported))
        raise StrategyError(
            "invalid build_cross_matrix_candidate inputs (" + "; ".join(details) + ")"
        )
    return {
        "source_artifact_id": _text(inputs["source_artifact_id"], "source_artifact_id"),
        "expected_artifact_content_hash": _sha256_text(
            inputs["expected_artifact_content_hash"],
            "expected_artifact_content_hash",
        ),
        "expected_candidate_id": _text(
            inputs["expected_candidate_id"], "expected_candidate_id"
        ),
        "expected_evidence_hash": _sha256_text(
            inputs["expected_evidence_hash"], "expected_evidence_hash"
        ),
        "x_feature": _text(inputs["x_feature"], "x_feature"),
        "x_method": _text(inputs["x_method"], "x_method"),
        "y_feature": _text(inputs["y_feature"], "y_feature"),
        "y_method": _text(inputs["y_method"], "y_method"),
    }


def _load_exact_parent_evidence(
    source,
    *,
    task_id: str,
    expected_candidate_id: str,
    expected_evidence_hash: str,
) -> dict[str, Any]:
    try:
        raw = source.path.read_bytes()
    except OSError as exc:
        raise StrategyError("source candidate artifact could not be read") from exc
    if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), source.content_hash):
        raise StrategyError("source candidate artifact content hash drifted")
    try:
        report = strategy_candidate_report_from_json(raw)
        evidence = validate_candidate_evidence(report["candidate_evidence"])
        canonical = canonical_strategy_candidate_report_json(
            evidence,
            report["univariate_analysis"],
        )
    except (KeyError, TypeError, ValueError, StrategyError) as exc:
        raise StrategyError("source candidate report failed strict validation") from exc
    if canonical != raw:
        raise StrategyError("source candidate report is not canonical JSON")
    candidate_asset_tools._require_report_binding(
        evidence,
        source=source,
        task_id=task_id,
        expected_candidate_id=expected_candidate_id,
        expected_evidence_hash=expected_evidence_hash,
    )
    return evidence


def _resolve_axis(
    evidence: Mapping[str, Any],
    *,
    dataset,
    feature: str,
    method: str,
    label: str,
) -> dict[str, Any]:
    features = evidence["analysis"].get("features")
    if not _sequence(features):
        raise StrategyError("candidate analysis features are invalid")
    feature_matches = [
        item
        for item in features
        if isinstance(item, Mapping) and item.get("feature") == feature
    ]
    if len(feature_matches) != 1:
        raise StrategyError(f"Cross Matrix {label} feature not found: {feature}")
    methods = feature_matches[0].get("methods")
    if not _sequence(methods):
        raise StrategyError(f"Cross Matrix {label} feature methods are invalid")
    method_matches = [
        item
        for item in methods
        if isinstance(item, Mapping) and item.get("method") == method
    ]
    if len(method_matches) != 1 or method_matches[0].get("status") != "available":
        raise StrategyError(
            f"available Cross Matrix {label} method not found: {feature}/{method}"
        )
    if feature not in set(dataset.columns):
        raise StrategyError(f"Cross Matrix {label} feature column not found: {feature}")
    raw_bins = method_matches[0].get("bins")
    if not _sequence(raw_bins):
        raise StrategyError(f"Cross Matrix {label} method must contain bins")
    bins = sorted(
        (_normalize_parent_bin(item, feature=feature, axis_label=label) for item in raw_bins),
        key=lambda item: item["index"],
    )
    if [item["index"] for item in bins] != list(range(len(bins))):
        raise StrategyError(f"Cross Matrix {label} bin indices must be contiguous")
    ids = [item["id"] for item in bins]
    if len(set(ids)) != len(ids):
        raise StrategyError(f"Cross Matrix {label} bin ids must be unique")
    return {"feature": feature, "method": method, "bins": bins}


def _normalize_parent_bin(
    value: object, *, feature: str, axis_label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StrategyError(f"Cross Matrix {axis_label} bins must contain objects")
    index = _non_negative_int(value.get("index"), f"{axis_label} bin.index")
    source_bin_id = _text(value.get("id"), f"{axis_label} bin.id")
    condition_raw = value.get("condition")
    if not isinstance(condition_raw, Mapping):
        raise StrategyError(f"Cross Matrix {axis_label} bin.condition must be an object")
    try:
        condition = canonicalize_expression(condition_raw)
    except StrategyError as exc:
        raise StrategyError(
            f"Cross Matrix {axis_label} bin condition is invalid"
        ) from exc
    if _canonical_json(condition) != _canonical_json(condition_raw):
        raise StrategyError(
            f"Cross Matrix {axis_label} bin condition must be canonical Strategy DSL"
        )
    if _expression_fields(condition) != {feature}:
        raise StrategyError(
            f"Cross Matrix {axis_label} bin condition must reference only {feature}"
        )
    return {
        "index": index,
        "id": source_bin_id,
        "condition": condition,
        "count": _non_negative_int(value.get("count"), f"{axis_label} bin.count"),
        "good": _non_negative_int(value.get("good"), f"{axis_label} bin.good"),
        "bad": _non_negative_int(value.get("bad"), f"{axis_label} bin.bad"),
        "share": _share(value.get("share"), f"{axis_label} bin.share"),
    }


def _resolve_projection(
    evidence: Mapping[str, Any],
    *,
    dataset,
    row_feature: str,
    column_feature: str,
) -> dict[str, Any]:
    analysis = evidence["analysis"]
    target_col = _text(analysis.get("target"), "candidate target")
    parameters = evidence["generation"]["parameters"]
    if parameters.get("target_col") != target_col:
        raise StrategyError("candidate target binding is inconsistent")
    available = set(dataset.columns)
    if target_col not in available:
        raise StrategyError(f"candidate target column not found: {target_col}")
    drop_nan_labels = parameters.get("drop_nan_labels")
    if not isinstance(drop_nan_labels, bool):
        raise StrategyError("candidate drop_nan_labels evidence is invalid")
    expected_dropped = _non_negative_int(
        parameters.get("nan_labels_dropped"),
        "candidate nan_labels_dropped",
    )
    analysis_parameters = analysis.get("parameters")
    if not isinstance(analysis_parameters, Mapping):
        raise StrategyError("candidate analysis parameters must be an object")
    amounts: dict[str, str | None] = {}
    for name, evidence_name in (
        ("loan_amount_col", "loan_amount"),
        ("overdue_amount_col", "overdue_amount"),
    ):
        column = analysis_parameters.get(evidence_name)
        if parameters.get(name) != column:
            raise StrategyError(f"candidate {name} binding is inconsistent")
        if column is not None and (
            not isinstance(column, str) or not column or column not in available
        ):
            raise StrategyError(f"candidate {name} evidence is invalid")
        amounts[name] = column
    columns = []
    for column in (
        row_feature,
        column_feature,
        target_col,
        amounts["loan_amount_col"],
        amounts["overdue_amount_col"],
    ):
        if column is not None and column not in columns:
            columns.append(column)
    return {
        "target_col": target_col,
        "columns": columns,
        "drop_nan_labels": drop_nan_labels,
        "expected_dropped": expected_dropped,
        **amounts,
    }


def _resolve_exact_labeled_sample(
    frame: pd.DataFrame,
    *,
    evidence: Mapping[str, Any],
    target_col: str,
    drop_nan_labels: bool,
    expected_dropped: int,
) -> pd.DataFrame:
    labeled, dropped = resolve_labeled_frame(
        frame,
        target_col,
        drop_nan_labels=drop_nan_labels,
        scope="Cross Matrix source dataset",
    )
    labeled = labeled.reset_index(drop=True)
    if dropped != expected_dropped:
        raise StrategyError("Cross Matrix NaN-label evidence does not match the dataset")
    expected_rows = _non_negative_int(
        evidence["analysis"].get("row_count"), "candidate analysis row_count"
    )
    if len(labeled) != expected_rows:
        raise StrategyError("Cross Matrix row-count evidence does not match the dataset")
    return labeled


def _binary_target(frame: pd.DataFrame, target_col: str) -> np.ndarray:
    try:
        values = pd.to_numeric(frame[target_col], errors="raise").to_numpy(dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise StrategyError("Cross Matrix target must contain binary 0/1 values") from exc
    if not np.all(np.isfinite(values)) or not np.all(np.isin(values, [0.0, 1.0])):
        raise StrategyError("Cross Matrix target must contain binary 0/1 values")
    return values.astype(np.int64)


def _replay_axis(
    frame: pd.DataFrame,
    *,
    target: np.ndarray,
    axis: Mapping[str, Any],
    axis_label: str,
) -> np.ndarray:
    assignment = np.full(len(frame), -1, dtype=np.int64)
    membership = np.zeros(len(frame), dtype=np.int64)
    for bin_row in axis["bins"]:
        mask = evaluate_expression_frame(frame, bin_row["condition"]).to_numpy(
            dtype=bool,
            copy=False,
        )
        count = int(mask.sum())
        bad = int(target[mask].sum())
        good = count - bad
        if count != bin_row["count"]:
            raise StrategyError(
                f"Cross Matrix {axis_label} parent bin {bin_row['id']} count did not replay"
            )
        if good != bin_row["good"] or bad != bin_row["bad"]:
            raise StrategyError(
                f"Cross Matrix {axis_label} parent bin {bin_row['id']} labels did not replay"
            )
        expected_share = count / len(frame) if len(frame) else 0.0
        if not math.isclose(
            expected_share,
            bin_row["share"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise StrategyError(
                f"Cross Matrix {axis_label} parent bin {bin_row['id']} share did not replay"
            )
        assignment[mask] = bin_row["index"]
        membership += mask.astype(np.int64)
    if not np.all(membership == 1) or np.any(assignment < 0):
        raise StrategyError(
            f"Cross Matrix {axis_label} parent bins must form a one-hot partition"
        )
    return assignment


def _measure_matrix(
    frame: pd.DataFrame,
    *,
    evidence: Mapping[str, Any],
    target: np.ndarray,
    row_axis: Mapping[str, Any],
    column_axis: Mapping[str, Any],
    row_index: np.ndarray,
    column_index: np.ndarray,
    loan_amount_col: str | None,
    overdue_amount_col: str | None,
) -> dict[str, Any]:
    column_count = len(column_axis["bins"])
    cell_count = len(row_axis["bins"]) * column_count
    flat_index = row_index * column_count + column_index
    counts = np.bincount(flat_index, minlength=cell_count).astype(np.int64)
    bads = np.bincount(
        flat_index,
        weights=target.astype(float),
        minlength=cell_count,
    ).astype(np.int64)
    goods = counts - bads
    loan = _amount_array(frame, loan_amount_col, "loan_amount")
    overdue = _amount_array(frame, overdue_amount_col, "overdue_amount")
    loan_observations = _aggregate_amount(flat_index, loan, cell_count=cell_count)
    overdue_observations = _aggregate_amount(
        flat_index, overdue, cell_count=cell_count
    )
    paired_observations = _aggregate_paired(
        flat_index,
        loan,
        overdue,
        cell_count=cell_count,
    )
    cells = []
    for row_bin in row_axis["bins"]:
        for column_bin in column_axis["bins"]:
            index = row_bin["index"] * column_count + column_bin["index"]
            cells.append(
                {
                    "row_source_bin_id": row_bin["id"],
                    "column_source_bin_id": column_bin["id"],
                    "count": int(counts[index]),
                    "good": int(goods[index]),
                    "bad": int(bads[index]),
                    "amounts": {
                        "loan_amount": loan_observations[index],
                        "overdue_amount": overdue_observations[index],
                        "paired": paired_observations[index],
                    },
                }
            )
    population_count = len(frame)
    if int(counts.sum()) != population_count or int(bads.sum()) != int(target.sum()):
        raise StrategyError("Cross Matrix primary measurements do not conserve rows")
    return {
        "schema_version": "strategy.cross-matrix-measurement.v1",
        "sample_context_hash": sample_context_hash_from_candidate_evidence(evidence),
        "population_count": population_count,
        "good": int(population_count - target.sum()),
        "bad": int(target.sum()),
        "cells": cells,
    }


def _amount_array(
    frame: pd.DataFrame, column: str | None, label: str
) -> np.ndarray | None:
    if column is None:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    invalid = frame[column].notna().to_numpy() & ~np.isfinite(values)
    if np.any(invalid) or np.any(values[np.isfinite(values)] < 0):
        raise StrategyError(
            f"Cross Matrix {label} must contain non-negative finite values or null"
        )
    return values


def _aggregate_amount(
    flat_index: np.ndarray,
    values: np.ndarray | None,
    *,
    cell_count: int,
) -> list[dict[str, Any]]:
    if values is None:
        return [
            {"status": "unavailable", "covered_count": None, "value": None}
            for _ in range(cell_count)
        ]
    covered = np.isfinite(values)
    covered_counts = np.bincount(
        flat_index,
        weights=covered.astype(float),
        minlength=cell_count,
    ).astype(np.int64)
    totals = np.bincount(
        flat_index,
        weights=np.where(covered, values, 0.0),
        minlength=cell_count,
    )
    if not np.all(np.isfinite(totals)):
        raise StrategyError("Cross Matrix amount aggregation overflowed")
    return [
        {
            "status": "available",
            "covered_count": int(covered_counts[index]),
            "value": float(totals[index]),
        }
        for index in range(cell_count)
    ]


def _aggregate_paired(
    flat_index: np.ndarray,
    loan: np.ndarray | None,
    overdue: np.ndarray | None,
    *,
    cell_count: int,
) -> list[dict[str, Any]]:
    if loan is None or overdue is None:
        return [
            {
                "status": "unavailable",
                "covered_count": None,
                "loan_value": None,
                "overdue_value": None,
            }
            for _ in range(cell_count)
        ]
    covered = np.isfinite(loan) & np.isfinite(overdue)
    covered_counts = np.bincount(
        flat_index,
        weights=covered.astype(float),
        minlength=cell_count,
    ).astype(np.int64)
    loan_totals = np.bincount(
        flat_index,
        weights=np.where(covered, loan, 0.0),
        minlength=cell_count,
    )
    overdue_totals = np.bincount(
        flat_index,
        weights=np.where(covered, overdue, 0.0),
        minlength=cell_count,
    )
    if not np.all(np.isfinite(loan_totals)) or not np.all(
        np.isfinite(overdue_totals)
    ):
        raise StrategyError("Cross Matrix paired amount aggregation overflowed")
    return [
        {
            "status": "available",
            "covered_count": int(covered_counts[index]),
            "loan_value": float(loan_totals[index]),
            "overdue_value": float(overdue_totals[index]),
        }
        for index in range(cell_count)
    ]


def _sample_identity(evidence: Mapping[str, Any]) -> dict[str, Any]:
    identity = evidence["identity"]
    return {
        "task_id": identity["task_id"],
        "dataset_id": identity["dataset_id"],
        "dataset_content_hash": identity["dataset_content_hash"],
        "workspace_revision": identity["workspace_revision"],
        "workspace_generation": identity["workspace_generation"],
        "semantic_mapping_hash": identity["semantic_mapping_hash"],
        "sample_context_hash": sample_context_hash_from_candidate_evidence(evidence),
        "target_col": evidence["analysis"]["target"],
        "row_count": evidence["analysis"]["row_count"],
    }


def _asset_lifecycle(asset: Mapping[str, Any]) -> dict[str, str]:
    lifecycle = asset.get("lifecycle")
    if not isinstance(lifecycle, Mapping):
        raise StrategyError("Cross Matrix asset lifecycle must be an object")
    expected = frozenset(
        {"candidate_stage", "observation_stage", "validation_status"}
    )
    if set(lifecycle) != expected:
        raise StrategyError("Cross Matrix asset lifecycle fields are invalid")
    normalized = {
        "candidate_stage": _text(
            lifecycle["candidate_stage"], "Cross Matrix candidate_stage"
        ),
        "observation_stage": _text(
            lifecycle["observation_stage"], "Cross Matrix observation_stage"
        ),
        "validation_status": _text(
            lifecycle["validation_status"], "Cross Matrix validation_status"
        ),
    }
    if normalized != {
        "candidate_stage": "development",
        "observation_stage": "backtested",
        "validation_status": "unvalidated",
    }:
        raise StrategyError("Cross Matrix asset lifecycle cannot claim deployment")
    return normalized


def _persist_asset(
    runtime,
    *,
    task_id: str,
    source,
    dataset,
    evidence: Mapping[str, Any],
    asset: Mapping[str, Any],
    row_axis: Mapping[str, Any],
    column_axis: Mapping[str, Any],
    cell_count: int,
    content: bytes,
) -> dict[str, Any]:
    asset_id = _text(asset.get("asset_id"), "Cross Matrix asset_id")
    if _SAFE_ASSET_ID_RE.fullmatch(asset_id) is None:
        raise StrategyError("Cross Matrix asset_id is not safe for persistence")
    asset_hash = _sha256_text(asset.get("asset_hash"), "Cross Matrix asset_hash")
    content_hash = hashlib.sha256(content).hexdigest()
    tasks_root = Path(runtime.settings.tasks_dir)
    out_dir = tasks_root / task_id / "strategy_cross_matrix_candidates"
    candidate_asset_tools._require_output_directory_boundary(out_dir, root=tasks_root)
    filename = f"{asset_id}_{content_hash[:12]}.json"
    identity = evidence["identity"]
    lifecycle = _asset_lifecycle(asset)
    provenance = {
        "schema_version": ASSET_ARTIFACT_SCHEMA_VERSION,
        "producer_version": asset["producer_version"],
        "asset_schema_version": asset["schema_version"],
        "asset_type": asset["asset_type"],
        "asset_id": asset_id,
        "asset_hash": asset_hash,
        "parent_candidate_id": evidence["candidate_id"],
        "parent_evidence_hash": evidence["evidence_hash"],
        "candidate_id": asset["candidate_evidence"]["candidate_id"],
        "evidence_hash": asset["candidate_evidence"]["evidence_hash"],
        "source_artifact_id": source.artifact_id,
        "source_artifact_content_hash": source.content_hash,
        "task_id": identity["task_id"],
        "dataset_id": dataset.dataset_id,
        "dataset_content_hash": dataset.content_hash,
        "registry_metadata_hash": dataset.registry_metadata_hash,
        "workspace_revision": identity["workspace_revision"],
        "workspace_generation": identity["workspace_generation"],
        "semantic_mapping_hash": identity["semantic_mapping_hash"],
        "sample_context_hash": asset["sample_identity"]["sample_context_hash"],
        "target_col": asset["sample_identity"]["target_col"],
        "labeled_row_count": asset["sample_identity"]["row_count"],
        "row_axis": {
            "feature": row_axis["feature"],
            "method": row_axis["method"],
        },
        "column_axis": {
            "feature": column_axis["feature"],
            "method": column_axis["method"],
        },
        "cell_count": cell_count,
        "candidate_stage": lifecycle["candidate_stage"],
        "observation_stage": lifecycle["observation_stage"],
        "validation_status": lifecycle["validation_status"],
        "budget": asset["budget"]["limit"],
        "truncated": False,
    }
    if set(provenance) != _ASSET_PROVENANCE_FIELDS:
        raise StrategyError("Cross Matrix artifact provenance fields are invalid")
    uow = ArtifactUnitOfWork()
    staged = uow.stage_file(out_dir, filename)
    db_committed = False
    rollback_attempted_under_lock = False
    try:
        staged.path.write_bytes(content)
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                candidate_asset_tools._require_source_on_connection(conn, source)
                candidate_asset_tools._require_dataset_on_connection(conn, dataset)
                candidate_asset_tools._require_regular_artifact_path(
                    source.path,
                    root=tasks_root,
                )
                candidate_asset_tools._require_file_content_hash(
                    source.path,
                    source.content_hash,
                    "source candidate artifact content hash drifted",
                )
                candidate_asset_tools._require_file_content_hash(
                    dataset.path,
                    dataset.content_hash,
                    "candidate source dataset content hash drifted",
                )
                if rebuild_cross_matrix_candidate_asset(asset, evidence) != asset:
                    raise StrategyError(
                        "Cross Matrix asset changed before artifact registration"
                    )
                uow.promote_all()
                candidate_asset_tools._require_file_content_hash(
                    staged.final_path,
                    content_hash,
                    "Cross Matrix asset changed before artifact registration",
                )
                record = runtime.task_artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=ASSET_ARTIFACT_KIND,
                    path=str(staged.final_path),
                    content_hash=content_hash,
                    origin_tool=ORIGIN_TOOL,
                    provenance=provenance,
                )
                conn.commit()
                db_committed = True
            except Exception:
                rollback_attempted_under_lock = True
                uow.rollback()
                raise
        uow.commit()
    except Exception:
        if not db_committed and not rollback_attempted_under_lock:
            uow.rollback()
        raise
    return {
        "artifact_id": str(record["id"]),
        "kind": ASSET_ARTIFACT_KIND,
        "format": "json",
        "filename": staged.final_path.name,
        "content_hash": content_hash,
        "download_url": (
            f"/api/tasks/{quote(task_id, safe='')}"
            f"/task-artifacts/{quote(str(record['id']), safe='')}/download"
        ),
    }


def _expression_fields(expression: Mapping[str, Any]) -> set[str]:
    fields: set[str] = set()
    stack: list[Mapping[str, Any]] = [expression]
    while stack:
        node = stack.pop()
        field = node.get("field")
        if isinstance(field, str):
            fields.add(field)
        arg = node.get("arg")
        if isinstance(arg, Mapping):
            stack.append(arg)
        args = node.get("args")
        if _sequence(args, allow_empty=True):
            stack.extend(item for item in args if isinstance(item, Mapping))
    return fields


def _sequence(value: object, *, allow_empty: bool = False) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and (allow_empty or bool(value))
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise StrategyError(f"{name} must be non-empty text")
    return value.strip()


def _sha256_text(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise StrategyError(f"{name} must be a lowercase SHA-256 hash")
    return value


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StrategyError(f"{name} must be a non-negative integer")
    return value


def _share(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StrategyError(f"{name} must be a finite number in [0, 1]")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0 or normalized > 1:
        raise StrategyError(f"{name} must be a finite number in [0, 1]")
    return normalized


__all__ = [
    "ASSET_ARTIFACT_KIND",
    "CROSS_MATRIX_MAX_CELLS",
    "TOOL_SCHEMA_VERSION",
    "run_build_cross_matrix_candidate",
]
