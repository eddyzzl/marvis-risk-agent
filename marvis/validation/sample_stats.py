import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pandas as pd

from marvis.validation.config import ValidationConfig
from marvis.validation.checks import binary_target_series, validate_required_splits
from marvis.validation.input_contracts import FeatureMetadataRow, JsonValue
from marvis.validation.results import (
    BasicInfoResult,
    FeatureImportanceRow,
    MonthlyRow,
    SplitRow,
)
from marvis.validation.time_periods import date_key_series, month_key_series


def run_basic_info(
    *,
    sample: pd.DataFrame,
    config: ValidationConfig,
    model_meta_path: Path,
) -> BasicInfoResult:
    split_summary, monthly_distribution, sample_period = _sample_distribution_rows(
        sample=sample,
        config=config,
    )

    meta = json.loads(model_meta_path.read_text(encoding="utf-8"))
    hyperparameters = dict(meta.get("hyperparameters", {}))
    raw_importance = meta.get("feature_importance", [])
    feature_importance = [
        FeatureImportanceRow(
            rank=index + 1,
            feature=str(entry["feature"]),
            importance=float(entry["importance"]),
            category=str(entry.get("category") or entry.get("类别") or ""),
        )
        for index, entry in enumerate(raw_importance)
    ]

    return BasicInfoResult(
        sample_period=sample_period,
        split_summary=split_summary,
        monthly_distribution=monthly_distribution,
        hyperparameters=hyperparameters,
        feature_importance=feature_importance,
    )


def run_basic_info_from_metadata(
    *,
    sample: pd.DataFrame,
    config: ValidationConfig,
    model_params: Mapping[str, JsonValue],
    feature_metadata: Sequence[FeatureMetadataRow],
    cancellation_check: Callable[[], None] | None = None,
) -> BasicInfoResult:
    """Compute the legacy basic-info payload from normalized PMML metadata."""

    _check_cancelled(cancellation_check)
    split_summary, monthly_distribution, sample_period = _sample_distribution_rows(
        sample=sample,
        config=config,
        cancellation_check=cancellation_check,
    )
    _check_cancelled(cancellation_check)
    ranked = sorted(
        (row for row in feature_metadata if row.in_pmml),
        key=lambda row: (-row.importance, row.feature),
    )
    return BasicInfoResult(
        sample_period=sample_period,
        split_summary=split_summary,
        monthly_distribution=monthly_distribution,
        hyperparameters=dict(model_params),
        feature_importance=[
            FeatureImportanceRow(
                rank=index,
                feature=row.feature,
                importance=row.importance,
                category=row.category,
            )
            for index, row in enumerate(ranked, start=1)
        ],
    )


def _sample_distribution_rows(
    *,
    sample: pd.DataFrame,
    config: ValidationConfig,
    cancellation_check: Callable[[], None] | None = None,
) -> tuple[list[SplitRow], list[MonthlyRow], tuple[str, str]]:
    target = binary_target_series(sample, config.target_col)
    validate_required_splits(
        sample,
        split_col=config.split_col,
        split_values=config.split_values,
    )
    date_keys = date_key_series(sample[config.time_col], column_name=config.time_col)
    month_keys = month_key_series(sample[config.time_col], column_name=config.time_col)
    sample_period = _period_from_keys(date_keys)

    split_summary: list[SplitRow] = []
    for split_name in ("train", "test", "oot"):
        _check_cancelled(cancellation_check)
        value = config.split_values[split_name]
        rows = sample[sample[config.split_col] == value]
        bad = int(target.loc[rows.index].sum())
        count = int(len(rows))
        split_dates = date_keys.loc[rows.index] if count else pd.Series(dtype=str)
        split_summary.append(SplitRow(
            split=split_name,
            sample_count=count,
            bad_count=bad,
            bad_rate=(bad / count) if count else 0.0,
            period_start=str(split_dates.min()) if count else "",
            period_end=str(split_dates.max()) if count else "",
        ))

    monthly_distribution: list[MonthlyRow] = []
    for month, group in sample.groupby(month_keys, sort=True):
        _check_cancelled(cancellation_check)
        count = int(len(group))
        bad = int(target.loc[group.index].sum())
        monthly_distribution.append(MonthlyRow(
            month=str(month),
            sample_count=count,
            bad_count=bad,
            bad_rate=(bad / count) if count else 0.0,
        ))

    return split_summary, monthly_distribution, sample_period


def _check_cancelled(callback: Callable[[], None] | None) -> None:
    if callback is not None:
        callback()


def _period_from_keys(date_keys: pd.Series) -> tuple[str, str]:
    if date_keys.empty:
        return ("", "")
    return (str(date_keys.min()), str(date_keys.max()))
