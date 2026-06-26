from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

from marvis.feature.metrics import feature_ks
from marvis.validation.binning import (
    assign_bins,
    bin_distribution,
    bin_table,
    compute_psi,
    equal_frequency_bin_edges,
)
from marvis.validation.vintage import compute_vintage_curve, vintage_curve_wide


@dataclass(frozen=True)
class BusinessColumns:
    loan_month_col: str | None = None
    interest_rate_col: str | None = None
    loan_amount_col: str | None = None
    term_col: str | None = None
    drawdown_amount_col: str | None = None
    credit_limit_col: str | None = None
    mob_observe_cols: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReportSectionStatus:
    section: str
    available: bool
    reason: str | None = None


def resolve_report_sections(
    business: BusinessColumns | None,
    dictionary_id: str | None,
) -> list[ReportSectionStatus]:
    business = business or BusinessColumns()
    requirements = {
        "sample_analysis": ["loan_month_col"],
        "vintage": ["loan_month_col", "mob_observe_cols"],
        "amount_bin": ["loan_amount_col"],
        "low_pricing": ["interest_rate_col"],
        "product_list": [],
    }
    statuses = []
    for section, columns in requirements.items():
        missing = [column for column in columns if not _has_business_column(business, column)]
        if section == "product_list" and not dictionary_id:
            missing.append("feature_dictionary")
        statuses.append(
            ReportSectionStatus(
                section=section,
                available=not missing,
                reason=None if not missing else f"缺少业务列/字典: {', '.join(missing)}",
            )
        )
    return statuses


def compute_sample_analysis(
    backend,
    dataset_path: Path,
    *,
    loan_month_col: str,
    target_col: str,
    business: BusinessColumns,
    mob_cols: tuple[str, ...],
) -> list[dict]:
    columns = _unique([
        loan_month_col,
        target_col,
        business.interest_rate_col,
        business.loan_amount_col,
        business.term_col,
        business.drawdown_amount_col,
        *mob_cols,
    ])
    frame = backend.read_frame(dataset_path, columns=columns)
    rows = []
    for month, group in frame.groupby(loan_month_col, sort=True):
        # NaN labels are excluded (never counted as "未逾期"); rate is over labeled loans.
        target = pd.to_numeric(group[target_col], errors="coerce")
        labeled_count = int(target.notna().sum())
        bad = int((target == 1).sum())
        good = int((target == 0).sum())
        row = {
            "放款月": str(month),
            "放款笔数": int(len(group)),
            "未逾期": good,
            "逾期": bad,
            "逾期率": _ratio(float(bad), float(labeled_count)),
        }
        _add_mean(row, "平均利率", group, business.interest_rate_col)
        _add_mean(row, "平均放款金额", group, business.loan_amount_col)
        _add_mean(row, "平均期数", group, business.term_col)
        _add_mean(row, "平均支用金额", group, business.drawdown_amount_col)
        for mob_label in ("3", "6"):
            col = _mob_col(mob_cols, mob_label)
            if col and col in group.columns:
                row[f"Mob{mob_label}逾期率"] = float(
                    pd.to_numeric(group[col], errors="coerce").fillna(0).mean()
                )
        rows.append(row)
    return rows


def compute_vintage_report(
    backend,
    dataset_path: Path,
    *,
    loan_month_col: str,
    mob_observe_cols: tuple[str, ...],
    amount_col: str | None,
) -> dict:
    columns = _unique([loan_month_col, amount_col, *mob_observe_cols])
    frame = backend.read_frame(dataset_path, columns=columns)
    long_rows = []
    for mob_index, column in enumerate(mob_observe_cols, start=1):
        mob = _mob_number(column, fallback=mob_index)
        for _, row in frame[[loan_month_col, column, *( [amount_col] if amount_col else [] )]].iterrows():
            item = {
                "cohort": row[loan_month_col],
                "mob": mob,
                "target": row[column],
            }
            if amount_col:
                item["amount"] = row[amount_col]
            long_rows.append(item)
    long_frame = pd.DataFrame(long_rows)
    points = compute_vintage_curve(
        long_frame,
        cohort_col="cohort",
        mob_col="mob",
        target_col="target",
        balance_col="amount" if amount_col else None,
    )
    counts, amounts = _vintage_cohort_business_columns(points)
    return {
        "cohorts": sorted({point.cohort for point in points}),
        "headers": _vintage_headers(points, mob_observe_cols),
        "counts": counts,
        "amounts": amounts,
        "curves": vintage_curve_wide(points, metric="cum_bad_rate"),
        "points": [asdict(point) for point in points],
    }


def compute_amount_bin_table(
    backend,
    dataset_path: Path,
    *,
    score_col: str,
    target_col: str,
    edges,
    business: BusinessColumns,
    filters: dict[str, object] | None = None,
) -> list[dict]:
    columns = _unique([
        score_col,
        target_col,
        *(filters or {}).keys(),
        business.interest_rate_col,
        business.term_col,
        business.credit_limit_col,
        business.loan_amount_col,
        business.drawdown_amount_col,
    ])
    frame = backend.read_frame(dataset_path, columns=columns)
    for column, value in (filters or {}).items():
        frame = frame[frame[str(column)] == value]
    edges = np.asarray(edges, dtype=float)
    bins = pd.Series(assign_bins(frame[score_col].to_numpy(dtype=float), edges), index=frame.index)
    valid = (bins > 0) & np.isfinite(pd.to_numeric(frame[target_col], errors="coerce"))
    base_rows = bin_table(frame, edges, score_col=score_col, target_col=target_col)
    output = []
    cumulative_amount = 0.0
    cumulative_bad_amount = 0.0
    overall_amount_bad_rate = _amount_bad_rate(frame, target_col, business.loan_amount_col)
    for row in base_rows:
        group = frame[valid & (bins == row.bin_index)]
        payload = asdict(row)
        _add_mean(payload, "平均利率", group, business.interest_rate_col)
        _add_mean(payload, "平均期数", group, business.term_col)
        _add_mean(payload, "平均授信金额", group, business.credit_limit_col)
        _add_mean(payload, "平均放款金额", group, business.loan_amount_col)
        if (
            business.drawdown_amount_col
            and business.drawdown_amount_col in group.columns
            and business.credit_limit_col
            and business.credit_limit_col in group.columns
        ):
            drawdown = pd.to_numeric(group[business.drawdown_amount_col], errors="coerce").fillna(0)
            limit = pd.to_numeric(group[business.credit_limit_col], errors="coerce").fillna(0)
            payload["额度使用率"] = _ratio(float(drawdown.sum()), float(limit.sum()))
        if business.loan_amount_col and business.loan_amount_col in group.columns:
            amount = pd.to_numeric(group[business.loan_amount_col], errors="coerce").fillna(0)
            # NaN labels are excluded from both numerator and denominator (never counted as good).
            target = pd.to_numeric(group[target_col], errors="coerce")
            labeled = target.notna()
            amount_sum = float(amount[labeled].sum())
            bad_amount = float(amount[target == 1].sum())
            amount_bad_rate = _ratio(bad_amount, amount_sum)
            cumulative_amount += amount_sum
            cumulative_bad_amount += bad_amount
            payload["金额逾期率"] = amount_bad_rate
            payload["累计金额逾期率"] = _ratio(cumulative_bad_amount, cumulative_amount)
            payload["金额lift"] = _ratio(amount_bad_rate, overall_amount_bad_rate)
        output.append(payload)
    return output


def stress_low_pricing(
    backend,
    dataset_path: Path,
    *,
    score_col: str,
    target_col: str,
    interest_rate_col: str,
    low_pricing_threshold: float | None,
    ratios: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9),
) -> dict:
    frame = backend.read_frame(dataset_path, columns=[score_col, target_col, interest_rate_col])
    threshold = (
        float(low_pricing_threshold)
        if low_pricing_threshold is not None
        else float(pd.to_numeric(frame[interest_rate_col], errors="coerce").median())
    )
    edges = equal_frequency_bin_edges(frame[score_col].to_numpy(dtype=float), 10)
    base_dist = bin_distribution(frame[score_col].to_numpy(dtype=float), edges)
    baseline_ks = feature_ks(
        frame[score_col].to_numpy(dtype=float),
        frame[target_col].to_numpy(dtype=float),
    )
    baseline_low_pricing_ratio = float((frame[interest_rate_col] <= threshold).mean())
    by_ratio = {}
    bins_by_ratio = {}
    ks_by_ratio = {}
    psi_by_ratio = {}
    for ratio in ratios:
        sampled = _resample_low_pricing(frame, interest_rate_col, threshold, float(ratio))
        scores = sampled[score_col].to_numpy(dtype=float)
        target = sampled[target_col].to_numpy(dtype=float)
        ratio_key = str(float(ratio)).rstrip("0").rstrip(".")
        sampled_distribution = bin_distribution(scores, edges)
        ks = feature_ks(scores, target)
        psi = compute_psi(base_dist, sampled_distribution)
        bins_by_ratio[ratio_key] = [float(value) for value in np.cumsum(sampled_distribution)]
        ks_by_ratio[ratio_key] = ks
        psi_by_ratio[ratio_key] = psi
        by_ratio[ratio_key] = {
            "ks": ks,
            "psi": psi,
            "sample_count": int(len(sampled)),
        }
    return {
        "threshold": threshold,
        "bins_by_ratio": bins_by_ratio,
        "ks_by_ratio": ks_by_ratio,
        "psi_by_ratio": psi_by_ratio,
        "by_ratio": by_ratio,
        "conclusion_data": _low_pricing_conclusion_data(
            threshold=threshold,
            baseline_low_pricing_ratio=baseline_low_pricing_ratio,
            baseline_ks=baseline_ks,
            ks_by_ratio=ks_by_ratio,
            psi_by_ratio=psi_by_ratio,
        ),
    }


def build_feature_dictionary(backend, dict_dataset_id, registry) -> dict:
    dataset = registry.get(str(dict_dataset_id))
    frame = backend.read_frame(registry.resolve_path(dataset.id))
    feature_col = _first_existing(frame, ("特征名", "feature", "feature_name"))
    if not feature_col:
        return {}
    return {
        str(row[feature_col]): {
            "含义": _row_value(row, ("含义", "meaning", "description")),
            "产品名称": _row_value(row, ("产品名称", "产品", "product")),
            "厂商名称": _row_value(row, ("厂商名称", "厂商", "vendor")),
        }
        for _, row in frame.iterrows()
    }


def _amount_bad_rate(frame: pd.DataFrame, target_col: str, amount_col: str | None) -> float:
    if not amount_col or amount_col not in frame.columns:
        return 0.0
    amount = pd.to_numeric(frame[amount_col], errors="coerce").fillna(0)
    # NaN labels are excluded (never counted as good) from both numerator and denominator.
    target = pd.to_numeric(frame[target_col], errors="coerce")
    labeled = target.notna()
    return _ratio(float(amount[target == 1].sum()), float(amount[labeled].sum()))


def _vintage_cohort_business_columns(points) -> tuple[dict[str, int], dict[str, dict[str, float]]]:
    counts: dict[str, int] = {}
    amounts: dict[str, dict[str, float]] = {}
    first_mobs: dict[str, int] = {}
    for point in points:
        current_mob = first_mobs.get(point.cohort)
        if current_mob is not None and point.mob >= current_mob:
            continue
        first_mobs[point.cohort] = int(point.mob)
        counts[point.cohort] = int(point.sample_count)
        if point.balance_sum is not None:
            total = float(point.balance_sum)
            average = _ratio(total, float(point.sample_count))
            amounts[point.cohort] = {"total": total, "average": average}
    return counts, amounts


def _vintage_headers(points, mob_observe_cols: tuple[str, ...]) -> list[str]:
    by_mob = {
        _mob_number(column, fallback=index): str(column)
        for index, column in enumerate(mob_observe_cols, start=1)
    }
    return [by_mob.get(mob, f"mob{mob}") for mob in sorted({point.mob for point in points})]


def _has_business_column(business: BusinessColumns, field: str) -> bool:
    value = getattr(business, field)
    if isinstance(value, tuple):
        return bool(value)
    return bool(value)


def _unique(values) -> list[str]:
    out = []
    for value in values:
        if value and str(value) not in out:
            out.append(str(value))
    return out


def _add_mean(payload: dict, key: str, frame: pd.DataFrame, column: str | None) -> None:
    if column and column in frame.columns:
        payload[key] = float(pd.to_numeric(frame[column], errors="coerce").mean())


def _mob_col(columns: tuple[str, ...], mob_label: str) -> str | None:
    pattern = re.compile(rf"(?:^|[^0-9]){re.escape(mob_label)}(?:[^0-9]|$)")
    for column in columns:
        lower = column.lower()
        if lower == f"mob{mob_label}" or pattern.search(lower):
            return column
    return None


def _mob_number(column: str, *, fallback: int) -> int:
    match = re.search(r"(\d+)", str(column))
    return int(match.group(1)) if match else fallback


def _ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def _resample_low_pricing(
    frame: pd.DataFrame,
    interest_rate_col: str,
    threshold: float,
    ratio: float,
) -> pd.DataFrame:
    total = len(frame)
    low = frame[frame[interest_rate_col] <= threshold]
    high = frame[frame[interest_rate_col] > threshold]
    low_count = max(0, min(total, int(round(total * ratio))))
    high_count = total - low_count
    return pd.concat([
        _cycle_take(low, low_count),
        _cycle_take(high, high_count),
    ], ignore_index=True)


def _cycle_take(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    if count <= 0:
        return frame.iloc[0:0].copy()
    if frame.empty:
        return frame.copy()
    repeats = int(np.ceil(count / len(frame)))
    return pd.concat([frame] * repeats, ignore_index=True).iloc[:count].copy()


def _low_pricing_conclusion_data(
    *,
    threshold: float,
    baseline_low_pricing_ratio: float,
    baseline_ks: float,
    ks_by_ratio: dict[str, float],
    psi_by_ratio: dict[str, float],
) -> dict:
    max_psi_ratio = max(psi_by_ratio, key=lambda ratio: psi_by_ratio[ratio]) if psi_by_ratio else ""
    min_ks_ratio = min(ks_by_ratio, key=lambda ratio: ks_by_ratio[ratio]) if ks_by_ratio else ""
    min_ks = ks_by_ratio[min_ks_ratio] if min_ks_ratio else None
    return {
        "threshold": threshold,
        "baseline_low_pricing_ratio": baseline_low_pricing_ratio,
        "baseline_ks": baseline_ks,
        "max_psi_ratio": max_psi_ratio,
        "max_psi": psi_by_ratio[max_psi_ratio] if max_psi_ratio else None,
        "min_ks_ratio": min_ks_ratio,
        "min_ks": min_ks,
        "max_ks_drop": None if min_ks is None else baseline_ks - min_ks,
    }


def _first_existing(frame: pd.DataFrame, columns: tuple[str, ...]) -> str | None:
    for column in columns:
        if column in frame.columns:
            return column
    return None


def _row_value(row, columns: tuple[str, ...]) -> str:
    for column in columns:
        if column in row and pd.notna(row[column]):
            return str(row[column])
    return ""


__all__ = [
    "BusinessColumns",
    "ReportSectionStatus",
    "build_feature_dictionary",
    "compute_amount_bin_table",
    "compute_sample_analysis",
    "compute_vintage_report",
    "resolve_report_sections",
    "stress_low_pricing",
]
