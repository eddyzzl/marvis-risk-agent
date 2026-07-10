from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from marvis.validation.binning import (
    bin_distribution,
    bin_table,
    compute_ks,
    compute_psi,
    equal_frequency_bin_edges,
)
from marvis.validation.checks import finite_score_series
from marvis.validation.config import ValidationConfig
from marvis.validation.results import (
    StressBaseline,
    StressCategoryResult,
    StressTestResult,
)
from marvis.validation.scorer import Scorer


STRESS_MISSING_VALUE = -9999


def load_feature_categories(
    dictionary: pd.DataFrame,
    *,
    feature_col: str,
    category_col: str,
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for _, row in dictionary.iterrows():
        category = str(row[category_col])
        feature = str(row[feature_col])
        grouped.setdefault(category, []).append(feature)
    return grouped


def _model_features(config: ValidationConfig, feature_importance: list) -> list[str]:
    if config.feature_columns:
        return list(config.feature_columns)
    return [str(row.feature) for row in feature_importance]


def _filter_feature_categories(
    feature_categories: dict[str, list[str]],
    *,
    model_features: list[str],
) -> dict[str, list[str]]:
    if not model_features:
        return feature_categories
    allowed = set(model_features)
    filtered: dict[str, list[str]] = {}
    for category, features in feature_categories.items():
        in_model = [feature for feature in features if feature in allowed]
        if in_model:
            filtered[category] = in_model
    return filtered


def run_stress_test(
    *,
    oot_sample: pd.DataFrame,
    config: ValidationConfig,
    feature_categories: dict[str, list[str]],
    input_scorer: Scorer,
    cancellation_check: Callable[[], None] | None = None,
    unclassified_features: list[str] | None = None,
    category_source_counts: dict[str, int] | None = None,
) -> StressTestResult:
    if oot_sample.empty:
        raise ValueError("OOT sample is required for stress test")
    _raise_if_cancelled(cancellation_check)
    baseline_raw_scores = input_scorer.score(oot_sample.copy())
    if len(baseline_raw_scores) != len(oot_sample):
        raise ValueError(
            f"stress baseline scorer returned {len(baseline_raw_scores)} scores "
            f"for {len(oot_sample)} rows"
        )
    baseline_scores = finite_score_series(
        baseline_raw_scores,
        index=oot_sample.index,
        label="stress baseline scorer",
    ).to_numpy(dtype=float)
    _raise_if_cancelled(cancellation_check)
    baseline_labels = oot_sample[config.target_col].to_numpy(dtype=int)
    edges = equal_frequency_bin_edges(baseline_scores, config.bin_count)

    baseline_df = oot_sample.copy()
    baseline_df["__baseline_score__"] = baseline_scores
    baseline_bin_rows = bin_table(
        baseline_df, edges, score_col="__baseline_score__", target_col=config.target_col
    )
    baseline = StressBaseline(
        ks=float(compute_ks(baseline_scores, baseline_labels)),
        sample_count=int(len(oot_sample)),
        bin_table=baseline_bin_rows,
    )
    baseline_distribution = bin_distribution(baseline_scores, edges)

    per_category: list[StressCategoryResult] = []
    for category, features in feature_categories.items():
        _raise_if_cancelled(cancellation_check)
        in_model = [f for f in features if f in oot_sample.columns]
        if not in_model:
            per_category.append(StressCategoryResult(
                category=category,
                dropped_features=[],
                ks_after=None, ks_delta=None, psi_vs_baseline=None,
                bin_table=[],
                error="no features from this category are present in the sample",
                status="skipped",
            ))
            continue
        try:
            modified = oot_sample.copy()
            for feature in in_model:
                modified[feature] = STRESS_MISSING_VALUE
            raw_scores_after = input_scorer.score(modified)
            if len(raw_scores_after) != len(modified):
                raise ValueError(
                    f"stress scorer for {category} returned {len(raw_scores_after)} "
                    f"scores for {len(modified)} rows"
                )
            scores_after = finite_score_series(
                raw_scores_after,
                index=modified.index,
                label=f"stress scorer for {category}",
            ).to_numpy(dtype=float)
            _raise_if_cancelled(cancellation_check)
            ks_after = float(compute_ks(scores_after, baseline_labels))
            psi = float(compute_psi(baseline_distribution, bin_distribution(scores_after, edges)))
            after_df = oot_sample.copy()
            after_df["__after_score__"] = scores_after
            after_bin_rows = bin_table(
                after_df, edges, score_col="__after_score__", target_col=config.target_col
            )
            per_category.append(StressCategoryResult(
                category=category,
                dropped_features=in_model,
                ks_after=ks_after,
                ks_delta=ks_after - baseline.ks,
                psi_vs_baseline=psi,
                bin_table=after_bin_rows,
                error=None,
                status="completed",
            ))
            _raise_if_cancelled(cancellation_check)
        except Exception as exc:
            per_category.append(StressCategoryResult(
                category=category,
                dropped_features=in_model,
                ks_after=None, ks_delta=None, psi_vs_baseline=None,
                bin_table=[],
                error=f"{type(exc).__name__}: {exc}",
                status="error",
            ))

    unresolved = list(unclassified_features or [])
    status = _stress_test_status(per_category)
    if unresolved and not per_category:
        status = "failed"
    elif unresolved and status == "completed":
        status = "partial"
    return StressTestResult(
        baseline=baseline,
        per_category=per_category,
        status=status,
        unclassified_features=unresolved,
        category_source_counts=dict(category_source_counts or {}),
    )


def _raise_if_cancelled(cancellation_check: Callable[[], None] | None) -> None:
    if cancellation_check is not None:
        cancellation_check()


def _stress_test_status(per_category: list[StressCategoryResult]) -> str:
    if not per_category:
        return "skipped"
    statuses = {row.status for row in per_category}
    if statuses == {"completed"}:
        return "completed"
    if statuses == {"skipped"}:
        return "skipped"
    if statuses == {"error"}:
        return "failed"
    return "partial"
