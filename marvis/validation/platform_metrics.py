"""Deterministic validation results assembled from in-memory, verified inputs.

File loading, artifact publication and task/repository lifecycle belong to
``validation_services`` and the pipeline, never this module.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from marvis.model_algorithms import normalize_algorithm
from marvis.validation.config import ValidationConfig
from marvis.validation.effectiveness import run_effectiveness
from marvis.validation.feature_categories import (
    FeatureCategoryConflict, FeatureCategoryResolution, resolve_feature_categories,
)
from marvis.validation.input_contracts import FeatureMetadataResolution, ValidationInputContract
from marvis.validation.results import (
    FeatureImportanceRow, PmmlScoringResult, StressTestResult, ValidationResults,
)
from marvis.validation.sample_stats import run_basic_info_from_metadata


def validate_metrics_contract(
    contract: ValidationInputContract, metadata_resolution: FeatureMetadataResolution,
) -> None:
    if contract.status != "ready":
        raise ValueError("validation input contract is not ready for metrics")
    if metadata_resolution != contract.require_feature_metadata():
        raise ValueError("feature metadata does not match validation input contract")


def compute_platform_validation_results(
    *,
    model_name: str,
    model_version: str,
    contract: ValidationInputContract,
    sample_scored: pd.DataFrame,
    config: ValidationConfig,
    scoring_result: PmmlScoringResult,
    metadata_resolution: FeatureMetadataResolution,
    stress_test: StressTestResult,
    cancellation_check: Callable[[], None] | None = None,
) -> ValidationResults:
    validate_metrics_contract(contract, metadata_resolution)
    _check_cancelled(cancellation_check)
    basic_info = run_basic_info_from_metadata(
        sample=sample_scored,
        config=config,
        model_params=contract.require_model_params(),
        feature_metadata=metadata_resolution.rows,
        cancellation_check=cancellation_check,
    )
    effectiveness = run_effectiveness(
        sample=sample_scored,
        config=config,
        cancellation_check=cancellation_check,
    )
    _check_cancelled(cancellation_check)
    return ValidationResults(
        model_name=model_name,
        model_version=model_version,
        algorithm=normalize_algorithm(contract.require_algorithm()),
        target_type="binary",
        schema_version="marvis.validation_results.v2",
        pmml_scoring=scoring_result,
        basic_info=basic_info,
        effectiveness=effectiveness,
        stress_test=stress_test,
    )


def stress_category_resolution_for_metrics(
    *,
    feature_importance: list[FeatureImportanceRow],
    fallback_model_features: list[str] | None = None,
    dictionary: pd.DataFrame,
    feature_col: str,
    category_col: str,
    stress_scores_payload: dict[str, Any] | None,
) -> FeatureCategoryResolution:
    model_features = [(row.feature, row.category) for row in feature_importance]
    if not model_features:
        dictionary_features = {
            str(value).strip()
            for value in dictionary[feature_col].tolist()
            if pd.notna(value) and str(value).strip()
        }
        model_features = [
            (feature, "")
            for feature in (fallback_model_features or [])
            if feature in dictionary_features
        ]
    expected = resolve_feature_categories(
        model_features=model_features,
        dictionary=dictionary,
        feature_col=feature_col,
        category_col=category_col,
    )
    _raise_category_conflict(expected)
    return validate_stress_category_artifact(expected, stress_scores_payload)


def validate_stress_category_artifact(
    expected: FeatureCategoryResolution,
    stress_scores_payload: dict[str, Any] | None,
) -> FeatureCategoryResolution:
    """Validate already loaded scenario metadata against the resolved categories."""
    if stress_scores_payload is None:
        return expected

    payload = stress_scores_payload
    if payload.get("schema_version") != "marvis.validation_stress_scores.v2":
        raise ValueError(
            "stress scenario artifact category mapping does not match model metadata"
        )
    actual = _category_resolution_from_stress_payload(payload)
    _raise_category_conflict(actual)
    if (
        actual.per_category != expected.per_category
        or actual.unclassified_features != expected.unclassified_features
        or actual.source_counts != expected.source_counts
    ):
        raise ValueError(
            "stress scenario artifact category mapping does not match model metadata"
        )
    return actual


def _category_resolution_from_stress_payload(
    payload: dict[str, Any],
) -> FeatureCategoryResolution:
    raw_categories = payload.get("feature_categories") or {}
    per_category = {
        str(category): [str(feature) for feature in features]
        for category, features in raw_categories.items()
    }
    conflicts = [
        FeatureCategoryConflict(
            feature=str(row.get("feature") or ""),
            categories=tuple(str(value) for value in row.get("categories") or []),
            source=str(row.get("source") or ""),
        )
        for row in payload.get("conflicts") or []
    ]
    return FeatureCategoryResolution(
        per_category=per_category,
        unclassified_features=[
            str(feature) for feature in payload.get("unclassified_features") or []
        ],
        conflicts=conflicts,
        source_counts={
            str(source): int(count)
            for source, count in (payload.get("source_counts") or {}).items()
        },
    )


def _raise_category_conflict(resolution: FeatureCategoryResolution) -> None:
    if not resolution.conflicts:
        return
    conflict = resolution.conflicts[0]
    raise ValueError(
        f"stress category conflict for {conflict.feature}: "
        + ", ".join(conflict.categories)
    )


def _check_cancelled(callback: Callable[[], None] | None) -> None:
    if callback is not None:
        callback()
