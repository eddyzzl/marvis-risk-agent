from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from marvis.data.contracts import ColumnProfile, Dataset


QUALITY_SAMPLE_N = 50_000
MISSING_BLOCK_THRESHOLD = 0.95
MISSING_WARN_THRESHOLD = 0.50
HIGH_CARDINALITY_THRESHOLD = 1000
LEAKAGE_CORRELATION_THRESHOLD = 0.95
MIN_MODELING_ROWS = 1000
BAD_RATE_LOW_WARN = 0.005
BAD_RATE_HIGH_WARN = 0.50
ACCEPT_ONLY_WARNING = (
    "样本疑似仅含已批准客群；建议使用 reject_inference 工具或敏感性分析校正接受偏差，"
    "否则效果指标只代表已批准人群"
)

_DECISION_COLUMNS = {
    "approve_flag",
    "approval_flag",
    "approved",
    "decision",
    "approval_status",
    "reject_flag",
    "declined",
    "审批结果",
    "是否通过",
    "申请结果",
}


@dataclass(frozen=True)
class QualityIssue:
    column: str
    kind: str
    detail: str
    severity: str


def check_data_quality(
    backend,
    dataset: Dataset,
    dataset_path: Path,
    *,
    target_col: str | None = None,
) -> list[QualityIssue]:
    issues = _profile_quality_issues(dataset.columns)
    sample = backend.sample_rows(dataset_path, QUALITY_SAMPLE_N, seed=0)
    issues.extend(_detect_duplicate_columns(sample, dataset.columns, target_col=target_col))
    if target_col and target_col in sample.columns:
        issues.extend(_detect_leakage(sample, dataset.columns, target_col))
    return issues


def modeling_readiness(
    backend,
    dataset: Dataset,
    dataset_path: Path,
    *,
    target_col: str,
    split_col: str | None,
) -> dict:
    blockers: list[str] = []
    warnings: list[str] = []
    sample = backend.sample_rows(dataset_path, QUALITY_SAMPLE_N, seed=0)

    bad_rate = None
    if target_col not in sample.columns:
        blockers.append(f"missing target column: {target_col}")
    else:
        target = sample[target_col].dropna()
        values = set(target.unique().tolist())
        if values - {0, 1, False, True}:
            blockers.append("target must be binary 0/1")
        numeric_target = pd.to_numeric(sample[target_col], errors="coerce")
        bad_rate = float(numeric_target.mean()) if numeric_target.notna().any() else None
        if bad_rate is not None and (bad_rate < BAD_RATE_LOW_WARN or bad_rate > BAD_RATE_HIGH_WARN):
            warnings.append(f"imbalanced bad_rate {bad_rate:.2%}")

    if dataset.row_count < MIN_MODELING_ROWS:
        blockers.append("too few samples (<1000)")

    blockers.extend(_split_blockers(sample, split_col))
    quality = check_data_quality(backend, dataset, dataset_path, target_col=target_col)
    blockers.extend(f"{issue.column}: {issue.detail}" for issue in quality if issue.severity == "block")
    warnings.extend(f"{issue.column}: {issue.detail}" for issue in quality if issue.severity == "warn")
    if _looks_accept_only(sample):
        warnings.append(ACCEPT_ONLY_WARNING)

    return {
        "ready": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "stats": {
            "rows": dataset.row_count,
            "bad_rate": None if bad_rate is None else round(bad_rate, 4),
        },
    }


def _profile_quality_issues(profiles: tuple[ColumnProfile, ...]) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    for profile in profiles:
        if profile.null_rate > MISSING_BLOCK_THRESHOLD:
            issues.append(QualityIssue(profile.name, "missing", f"null {profile.null_rate:.0%}", "block"))
        elif profile.null_rate > MISSING_WARN_THRESHOLD:
            issues.append(QualityIssue(profile.name, "missing", f"null {profile.null_rate:.0%}", "warn"))
        if profile.cardinality <= 1 and not _is_decision_column(profile.name):
            issues.append(QualityIssue(profile.name, "constant", "single value", "block"))
        if (
            profile.semantic_role == "categorical"
            and profile.cardinality > HIGH_CARDINALITY_THRESHOLD
        ):
            issues.append(
                QualityIssue(
                    profile.name,
                    "high_cardinality",
                    f"cardinality {profile.cardinality}",
                    "warn",
                )
            )
    return issues


def _detect_duplicate_columns(
    frame: pd.DataFrame,
    profiles: tuple[ColumnProfile, ...],
    *,
    target_col: str | None,
) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    columns = [profile.name for profile in profiles if profile.name in frame.columns and profile.name != target_col]
    for index, column in enumerate(columns):
        left = frame[column].reset_index(drop=True)
        for previous in columns[:index]:
            if left.equals(frame[previous].reset_index(drop=True)):
                issues.append(
                    QualityIssue(
                        column,
                        "duplicate_col",
                        f"duplicates {previous}",
                        "block",
                    )
                )
                break
    return issues


def _detect_leakage(
    frame: pd.DataFrame,
    profiles: tuple[ColumnProfile, ...],
    target_col: str,
) -> list[QualityIssue]:
    target = pd.to_numeric(frame[target_col], errors="coerce")
    if target.nunique(dropna=True) <= 1:
        return []
    issues: list[QualityIssue] = []
    for profile in profiles:
        column = profile.name
        if column == target_col or column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.nunique(dropna=True) <= 1:
            continue
        corr = values.corr(target)
        if pd.notna(corr) and abs(float(corr)) > LEAKAGE_CORRELATION_THRESHOLD:
            issues.append(
                QualityIssue(
                    column,
                    "leakage_suspect",
                    f"abs corr {abs(float(corr)):.4f} with target {target_col}",
                    "block",
                )
            )
    return issues


def _split_blockers(frame: pd.DataFrame, split_col: str | None) -> list[str]:
    if not split_col:
        return []
    if split_col not in frame.columns:
        return [f"missing split column: {split_col}"]
    present = {str(value).strip().lower() for value in frame[split_col].dropna().unique()}
    blockers = []
    if "train" not in present:
        blockers.append("missing train split")
    if "test" not in present:
        blockers.append("missing test split")
    return blockers


def _looks_accept_only(frame: pd.DataFrame) -> bool:
    decision_cols = [
        column for column in frame.columns
        if _is_decision_column(str(column))
    ]
    if not decision_cols:
        return True
    for column in decision_cols:
        if frame[column].dropna().nunique() > 1:
            return False
    return True


def _is_decision_column(column: str) -> bool:
    text = str(column).strip()
    return text.lower() in _DECISION_COLUMNS or text in _DECISION_COLUMNS


__all__ = [
    "ACCEPT_ONLY_WARNING",
    "QualityIssue",
    "check_data_quality",
    "modeling_readiness",
]
