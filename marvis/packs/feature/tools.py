from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.data_dictionary import first_data_dictionary_id, load_business_names
from marvis.data.labels import require_labels_confirmed
from marvis.feature.candidates import (
    candidate_numeric_features,
    excluded_categorical_columns,
    suspected_categorical_columns,
)
from marvis.feature.binning import (
    chimerge_edges,
    equal_frequency_edges,
    equal_width_edges,
    manual_edges,
    monotonic_direction,
    monotonic_edges,
    tree_edges,
)
from marvis.feature.correlation import correlation_report
from marvis.feature.derive import derive_batch, derive_date_features
from marvis.feature.encode import apply_categorical_woe, categorical_woe_encode, onehot_encode, woe_encode
from marvis.feature.errors import FeatureError, FitRequiresSplitError
from marvis.feature.iv import compute_woe_iv, woe_result_from_binning
from marvis.feature.metrics import (
    feature_psi,
    head_tail_lift,
    selected_feature_metrics,
)
from marvis.feature.preprocessing import (
    read_preprocessing_chain,
    sidecar_path,
    write_preprocessing_chain,
)
from marvis.feature.transform import (
    apply_scaler,
    cap_outliers,
    impute_missing,
    mask_sentinel_values,
    minmax_normalize,
    zscore_standardize,
)
from marvis.plugins.sdk import PackRuntime
from marvis.settings import build_settings


def tool_compute_feature_metrics(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    selected = _selected_metric_names(inputs)
    target_col = str(inputs.get("target_col") or "").strip()
    time_col = str(inputs.get("time_col") or "").strip()
    split_col = str(inputs.get("split_col") or "").strip()
    dataset = _task_dataset(runtime, ctx, inputs["dataset_id"])
    if not time_col and selected & {
        "psi_month_first",
        "psi_month_last",
        "psi_month_previous",
    }:
        time_col = _auto_time_column(dataset)
    if not split_col and "psi_split" in selected:
        split_col = _auto_split_column(runtime, dataset)
    features = _resolve_feature_cols(
        runtime,
        str(inputs["dataset_id"]),
        inputs.get("features") or [],
        target_col=target_col,
        split_col=split_col or None,
    )
    available = set(runtime.backend.column_names(runtime.registry.resolve_path(dataset.id)))
    dependency_columns = [
        column
        for column in (target_col, time_col, split_col)
        if column and column in available
    ]
    dataset, frame = _read_frame(
        runtime,
        ctx,
        str(inputs["dataset_id"]),
        _unique([*features, *dependency_columns]),
    )
    target_dependent = bool(
        selected
        & {
            "iv",
            "ks",
            "auc",
            "lift",
            "head_tail_lift",
            "importance",
            "meaning_consistency",
        }
    )
    target_available = bool(target_col and target_col in frame.columns)
    nan_labels_dropped = (
        require_labels_confirmed(
            frame,
            target_col,
            drop_nan_labels=bool(inputs.get("drop_nan_labels")),
        )
        if target_dependent and target_available
        else 0
    )
    compare_frame = None
    if "psi" in selected and inputs.get("compare_dataset_id"):
        compare_dataset = _task_dataset(
            runtime,
            ctx,
            inputs["compare_dataset_id"],
        )
        compare_available = set(
            runtime.backend.column_names(runtime.registry.resolve_path(compare_dataset.id))
        )
        compare_features = [feature for feature in features if feature in compare_available]
        if compare_features:
            _compare_dataset, compare_frame = _read_frame(
                runtime,
                ctx,
                str(inputs["compare_dataset_id"]),
                compare_features,
            )
    target_values = _target_values(frame, target_col) if target_available else None
    metrics: list[dict] = []
    for feature in features:
        compare_values = (
            compare_frame[str(feature)].to_numpy(dtype=float)
            if compare_frame is not None and str(feature) in compare_frame.columns
            else None
        )
        row = selected_feature_metrics(
            frame[str(feature)].to_numpy(dtype=float),
            target_values,
            feature=str(feature),
            selected=selected,
            bins=int(inputs.get("bins") or 10),
            compare_values=compare_values,
        )
        metrics.append(row)
    result = {
        "dataset_id": dataset.id,
        "metrics": metrics,
        "collinear": None,
        "nan_labels_dropped": nan_labels_dropped,
        "selected_metrics": sorted(selected),
    }
    # Every optional family is guarded by the selected set.  In particular the
    # four PSI views do not read their dependency columns unless selected.
    if "vif" in selected:
        report = correlation_report(
            frame,
            features,
            method="pearson",
            threshold=float(inputs.get("corr_threshold", 0.8)),
        )
        result["collinear"] = _jsonable(report)
    if "head_tail_lift" in selected:
        # Merge the risk-direction-aware head/tail lift into each per-feature row so it
        # rides the existing metrics echo (no new output key / $ref needed).
        for index, feature in enumerate(features):
            if target_values is None:
                _set_metric_na(
                    metrics[index],
                    ("lift_head_5", "lift_head_10", "lift_tail_5", "lift_tail_10"),
                    code="missing_dependency",
                    dependency="target_col",
                    message="未提供可用目标列；头尾 Lift 需要 0/1 目标列。",
                )
            else:
                valid_rows = int(
                    metrics[index].get("valid_count")
                    or np.isfinite(frame[str(feature)].to_numpy(dtype=float)).sum()
                )
                metrics[index].update(
                    head_tail_lift(
                        frame[str(feature)].to_numpy(dtype=float),
                        target_values,
                        min_rows=1,
                    )
                )
                metrics[index]["lift_reason"] = (
                    f"有效标注样本仅 {valid_rows} 行，5%/10% 极端分位切片很小，结果仅供方向参考。"
                    if valid_rows < 20
                    else ""
                )
    if "importance" in selected:
        # Multivariate gain importance: train ONE capped, seed-pinned model over all
        # features and merge each feature's share into its row (lazy lightgbm import).
        if target_values is None:
            for row in metrics:
                _set_metric_na(
                    row,
                    ("importance",),
                    code="missing_dependency",
                    dependency="target_col",
                    message="未提供可用目标列；特征重要性需要 0/1 目标列。",
                )
        else:
            from marvis.feature.importance import feature_importance

            feature_names = list(features)
            importance = feature_importance(frame, feature_names, target_col)
            for index, feature in enumerate(feature_names):
                metrics[index]["importance"] = importance.get(feature)
    _attach_selected_psi_views(
        metrics,
        frame,
        features=features,
        selected=selected,
        time_col=time_col,
        split_col=split_col,
        bins=int(inputs.get("bins") or 10),
    )
    if "meaning_consistency" in selected:
        _attach_meaning_consistency(
            metrics,
            frame,
            features=features,
            target_values=target_values,
            runtime=runtime,
            ctx=ctx,
            decisions=inputs.get("meaning_directions"),
        )
    for row in metrics:
        recommendation, reason = _feature_recommendation(row)
        row["recommendation"] = recommendation
        row["recommendation_reason"] = reason
        row["recommendation_state"] = _RECOMMENDATION_STATES[recommendation]
        evidence = _recommendation_evidence(row)
        row["recommendation_evidence"] = evidence
        row["recommendation_confidence"] = _recommendation_confidence(
            recommendation,
            evidence,
        )
    return result


_METRIC_ALIASES = {
    "iv": "iv",
    "ks": "ks",
    "auc": "auc",
    "coverage": "coverage",
    "覆盖率": "coverage",
    "lift": "lift",
    "psi": "psi",
    "vif": "vif",
    "collinear": "vif",
    "共线": "vif",
    "head_tail_lift": "head_tail_lift",
    "headtail_lift": "head_tail_lift",
    "头尾lift": "head_tail_lift",
    "importance": "importance",
    "feature_importance": "importance",
    "重要性": "importance",
    "psi_month_first": "psi_month_first",
    "psi_month_last": "psi_month_last",
    "psi_month_previous": "psi_month_previous",
    "psi_split": "psi_split",
    "meaning_consistency": "meaning_consistency",
    "semantic_consistency": "meaning_consistency",
    "含义一致性": "meaning_consistency",
}
_LEGACY_METRICS = frozenset({"iv", "ks", "auc", "coverage", "lift", "psi"})
_PSI_VIEWS = ("psi_month_first", "psi_month_last", "psi_month_previous", "psi_split")
_RECOMMENDATION_STATES = {
    "推荐": "recommended",
    "候选": "candidate",
    "待评估": "unevaluated",
    "谨慎": "caution",
    "暂不推荐": "not_recommended",
    "不推荐": "not_recommended",
}


def _selected_metric_names(inputs: dict) -> set[str]:
    # Backward compatibility for direct pack callers that predate the checkbox
    # field: an absent ``metrics`` key retains the historical all-base result.
    # An explicit [] really means "calculate none".
    raw = inputs.get("metrics") if "metrics" in inputs else list(_LEGACY_METRICS)
    selected: set[str] = set()
    for item in raw or []:
        normalized = str(item).strip().lower()
        if normalized in _METRIC_ALIASES:
            selected.add(_METRIC_ALIASES[normalized])
    return selected


def _auto_time_column(dataset) -> str:
    profiles = list(getattr(dataset, "columns", None) or [])
    for profile in profiles:
        if str(getattr(profile, "semantic_role", "")) == "date":
            return str(getattr(profile, "name", "") or "")
    for profile in profiles:
        name = str(getattr(profile, "name", "") or "")
        normalized = name.lower().replace("-", "_")
        if normalized in {"month", "apply_month", "data_month", "date", "dt", "apply_date"}:
            return name
    return ""


def _auto_split_column(runtime: "_Runtime", dataset) -> str:
    path = runtime.registry.resolve_path(dataset.id)
    columns = runtime.backend.column_names(path)
    candidates = [
        name
        for name in columns
        if str(name).strip().lower().replace("-", "_")
        in {
            "split",
            "data_split",
            "dataset_split",
            "sample_split",
            "sample_type",
            "dataset_type",
            "data_type",
        }
    ]
    if not candidates:
        return ""
    probe = runtime.backend.sample_rows(path, 1000, seed=0)
    for name in candidates:
        roles = {
            value
            for value in _split_role_labels(probe[name])
            if value is not None
        }
        if "train" in roles and roles & {"test", "oot"}:
            return str(name)
    return ""


def _set_metric_na(
    row: dict,
    keys,
    *,
    code: str,
    dependency: str,
    message: str,
) -> None:
    reason = {
        "code": code,
        "metric_dependency": dependency,
        "message": message,
    }
    for key in keys:
        row[key] = None
        row[f"{key}_reason"] = dict(reason)


def _attach_selected_psi_views(
    metrics: list[dict],
    frame: pd.DataFrame,
    *,
    features: list[str],
    selected: set[str],
    time_col: str,
    split_col: str,
    bins: int,
) -> None:
    for view in _PSI_VIEWS:
        if view not in selected:
            continue
        dependency = "split_col" if view == "psi_split" else "time_col"
        column = split_col if view == "psi_split" else time_col
        if not column or column not in frame.columns:
            for row in metrics:
                _set_metric_na(
                    row,
                    (view,),
                    code="missing_dependency",
                    dependency=dependency,
                    message=f"未提供可用{dependency}；{view} 无法计算。",
                )
            continue
        labels = (
            _split_role_labels(frame[column])
            if view == "psi_split"
            else _month_labels(frame[column])
        )
        pairs = _psi_comparison_pairs(labels, view)
        if not pairs:
            for row in metrics:
                _set_metric_na(
                    row,
                    (view,),
                    code="insufficient_groups",
                    dependency=dependency,
                    message=f"{column} 未形成该 PSI 视角所需的至少两个有效分组。",
                )
            continue
        for row, feature in zip(metrics, features):
            series, reason = _feature_psi_series(
                frame[str(feature)].to_numpy(dtype=float),
                labels,
                pairs,
                bins=bins,
            )
            if reason is not None:
                row[view] = None
                row[f"{view}_reason"] = reason
                continue
            row[f"{view}_series"] = series
            row[view] = max((float(item["psi"]) for item in series), default=None)


def _feature_psi_series(
    values: np.ndarray,
    labels: np.ndarray,
    pairs: list[tuple[str, str]],
    *,
    bins: int,
) -> tuple[list[dict], dict | None]:
    rows: list[dict] = []
    for base_label, compare_label in pairs:
        base = np.asarray(values, dtype=float)[labels == base_label]
        compare = np.asarray(values, dtype=float)[labels == compare_label]
        finite_base = base[np.isfinite(base)]
        finite_compare = compare[np.isfinite(compare)]
        if finite_base.size < 2 or finite_compare.size < 2:
            continue
        edges = equal_frequency_edges(finite_base, bins)
        rows.append({
            "base": str(base_label),
            "compare": str(compare_label),
            "psi": float(feature_psi(base, compare, edges)),
        })
    if rows:
        return rows, None
    return [], {
        "code": "insufficient_feature_values",
        "message": "各对比分组中的有效特征值不足，无法形成 PSI 分布。",
    }


def _month_labels(series: pd.Series) -> np.ndarray:
    text = series.astype("string").str.strip()
    digits = text.str.replace(r"\D+", "", regex=True)
    digit_month = digits.where(digits.str.len().isin([6, 8])).str.slice(0, 6)
    parsed = pd.to_datetime(series, errors="coerce")
    parsed_month = parsed.dt.strftime("%Y-%m")
    labels = digit_month.fillna(parsed_month).fillna(text)
    labels = labels.where(labels.notna() & (labels != ""), other=pd.NA)
    return labels.astype(object).to_numpy()


def _split_role_labels(series: pd.Series) -> np.ndarray:
    train = {"train", "training", "dev", "develop", "development", "build"}
    test = {"test", "testing", "valid", "validation", "val", "holdout"}
    oot = {"oot", "ootest", "out_of_time", "oos", "time_oot"}

    def role(value):
        normalized = str(value).strip().lower()
        if normalized in train:
            return "train"
        if normalized in test:
            return "test"
        if normalized in oot:
            return "oot"
        return None

    return np.asarray([role(value) for value in series], dtype=object)


def _psi_comparison_pairs(labels: np.ndarray, view: str) -> list[tuple[str, str]]:
    groups = sorted({str(value) for value in labels if value is not None and not pd.isna(value)})
    if len(groups) < 2:
        return []
    if view == "psi_split":
        if "train" not in groups:
            return []
        return [("train", role) for role in ("test", "oot") if role in groups]
    if view == "psi_month_first":
        return [(groups[0], item) for item in groups[1:]]
    if view == "psi_month_last":
        return [(groups[-1], item) for item in groups[:-1]]
    return list(zip(groups[:-1], groups[1:]))


def _attach_meaning_consistency(
    metrics: list[dict],
    frame: pd.DataFrame,
    *,
    features: list[str],
    target_values: np.ndarray | None,
    runtime: "_Runtime",
    ctx,
    decisions,
) -> None:
    run_mode = _task_run_mode(runtime, ctx)
    dictionary_id = first_data_dictionary_id(runtime.registry.list_for_task(ctx.task_id))
    meanings = load_business_names(runtime.backend, runtime.registry, dictionary_id)
    if run_mode != "agent":
        for row in metrics:
            _set_metric_na(
                row,
                ("meaning_consistency",),
                code="agent_mode_required",
                dependency="run_mode",
                message="含义正负向一致性仅在 Agent 模式计算。",
            )
        return
    if not meanings:
        for row in metrics:
            _set_metric_na(
                row,
                ("meaning_consistency",),
                code="data_dictionary_required",
                dependency="data_dictionary",
                message="未提供可解析的数据字典，无法判断业务含义方向。",
            )
        return
    if target_values is None:
        for row in metrics:
            _set_metric_na(
                row,
                ("meaning_consistency",),
                code="missing_dependency",
                dependency="target_col",
                message="未提供可用目标列，无法核对实际方向。",
            )
        return
    from marvis.feature.correlation import safe_correlation

    direction_decisions = decisions if isinstance(decisions, dict) else {}

    for row, feature in zip(metrics, features):
        meaning = str(meanings.get(str(feature)) or "").strip()
        if not meaning:
            _set_metric_na(
                row,
                ("meaning_consistency",),
                code="dictionary_entry_missing",
                dependency="data_dictionary",
                message=f"数据字典中没有 {feature} 的业务含义。",
            )
            continue
        correlation = float(
            safe_correlation(
                frame[str(feature)].to_numpy(dtype=float),
                np.asarray(target_values, dtype=float),
            )
        )
        decision = (
            direction_decisions.get(str(feature))
            if isinstance(direction_decisions.get(str(feature)), dict)
            else {}
        )
        expected = str(decision.get("expected_direction") or "uncertain")
        if expected not in {"positive", "negative", "u_shape", "uncertain"}:
            expected = "uncertain"
        u_shape_evidence = None
        u_shape_reason = ""
        if expected == "u_shape":
            actual, u_shape_evidence, u_shape_reason = _actual_u_shape_direction(
                frame[str(feature)].to_numpy(dtype=float),
                np.asarray(target_values, dtype=float),
                feature=str(feature),
            )
        else:
            actual = (
                "positive"
                if correlation > 0.03
                else "negative"
                if correlation < -0.03
                else "uncertain"
            )
        if expected == "uncertain" or actual == "uncertain":
            consistency = "需人工看"
        else:
            consistency = "一致" if expected == actual else "不一致"
        rationale = str(decision.get("rationale") or "").strip()
        if expected == "uncertain" and not rationale:
            rationale = "没有可用的受约束 LLM 语义判向，按保守策略不作方向判断。"
        consistency_reason = (
            u_shape_reason
            if expected == "u_shape" and u_shape_reason
            else (
                f"业务预期={expected}，实际方向={actual}。"
                if expected != "uncertain"
                else rationale
            )
        )
        row.update({
            "business_meaning": meaning,
            "expected_direction": expected,
            "actual_direction": actual,
            "direction_correlation": correlation,
            "meaning_consistency": consistency,
            "meaning_consistency_reason": consistency_reason,
            "meaning_judgement_source": str(
                decision.get("judgement_source") or "no_llm_fallback"
            ),
            "meaning_direction_confidence": str(decision.get("confidence") or "low"),
            "meaning_direction_rationale": rationale,
            "meaning_direction_model": str(decision.get("model") or ""),
            "meaning_direction_prompt": {
                "name": str(decision.get("prompt_name") or "feature_meaning_direction"),
                "version": int(decision.get("prompt_version") or 1),
            },
            "u_shape_evidence": u_shape_evidence,
        })


def _task_run_mode(runtime: "_Runtime", ctx) -> str:
    try:
        from marvis.db import TaskRepository

        return str(TaskRepository(runtime.settings.db_path).get_task(ctx.task_id).run_mode or "")
    except (KeyError, AttributeError):
        return ""


def _actual_u_shape_direction(
    values: np.ndarray,
    target: np.ndarray,
    *,
    feature: str,
) -> tuple[str, dict | None, str]:
    """Test a semantic U-shape with the governed equal-frequency bin kernel."""

    from marvis.feature.bin_analysis import feature_bin_analysis

    try:
        analysis = feature_bin_analysis(
            values,
            target,
            feature=feature,
            requested_bins=7,
        )
    except Exception:
        return "uncertain", None, "分箱数据不足，无法稳定判断 U 型走势。"
    rows = [
        row
        for row in analysis.get("rows") or []
        if row.get("interval") != "缺失值"
    ]
    counts = [int(row.get("count") or 0) for row in rows]
    rates = [float(row.get("bad_rate") or 0.0) for row in rows]
    if len(rates) < 3:
        return "uncertain", {
            "bin_bad_rates": rates,
            "bin_counts": counts,
        }, "有效分箱少于 3 个，无法判断 U 型走势。"
    total = sum(counts)
    min_support = max(3, int(np.ceil(total * 0.02)))
    if any(count < min_support for count in counts):
        return "uncertain", {
            "bin_bad_rates": rates,
            "bin_counts": counts,
        }, f"存在样本量低于 {min_support} 的分箱，U 型判断不稳定。"
    trough = int(np.argmin(np.asarray(rates, dtype=float)))
    amplitude = min(rates[0] - rates[trough], rates[-1] - rates[trough])
    left_diff = np.diff(np.asarray(rates[:trough + 1], dtype=float))
    right_diff = np.diff(np.asarray(rates[trough:], dtype=float))
    left_ok = bool(left_diff.size) and float(np.mean(left_diff <= 0.01)) >= 0.67
    right_ok = bool(right_diff.size) and float(np.mean(right_diff >= -0.01)) >= 0.67
    is_u_shape = (
        0 < trough < len(rates) - 1
        and amplitude >= 0.03
        and left_ok
        and right_ok
    )
    evidence = {
        "bin_bad_rates": rates,
        "bin_counts": counts,
        "trough_bin": trough + 1,
        "minimum_end_gap": float(amplitude),
        "requested_bins": 7,
        "actual_bins": int(analysis.get("actual_bins") or len(rates)),
    }
    if not is_u_shape:
        return "uncertain", evidence, "分箱坏率未形成两端高、中间低的稳定 U 型走势。"
    return "u_shape", evidence, "分箱坏率呈两端高、中间低的 U 型走势。"


def _feature_recommendation(metric: dict) -> tuple[str, str]:
    """Deterministic, evidence-backed feature suggestion for the Agent surface.

    This is deliberately conservative: it never calls an LLM to invent a score,
    and it distinguishes predictive promise from missing stability evidence.
    """
    valid_count = int(metric.get("valid_count") or 0)
    if metric.get("meaning_consistency") == "不一致":
        return "不推荐", "业务预期方向与实际数据方向不一致，需先核对口径或数据质量。"
    if "coverage" in metric:
        missing_rate = float(metric.get("missing_rate") or 0.0)
        mode_rate = float(metric.get("mode_rate") or 0.0)
        unique_count = int(metric.get("unique_count") or 0)
        if valid_count == 0:
            return "不推荐", "没有有效数值，无法用于分析。"
        if missing_rate >= 0.80:
            return "不推荐", f"缺失率 {missing_rate:.1%} 过高。"
        if unique_count <= 1 or mode_rate >= 0.95:
            return "不推荐", f"单一值率 {mode_rate:.1%}，变量几乎没有区分信息。"

    signal_checks: list[bool] = []
    if metric.get("iv") is not None:
        signal_checks.append(float(metric["iv"]) >= 0.10)
    if metric.get("ks") is not None:
        signal_checks.append(float(metric["ks"]) >= 0.10)
    if metric.get("auc") is not None:
        signal_checks.append(float(metric["auc"]) >= 0.60)
    if not signal_checks:
        return "待评估", "本次未选择 IV、KS 或 AUC，无法判断单变量区分力。"
    has_signal = any(signal_checks)
    if not has_signal:
        return "暂不推荐", "本次所选区分力指标未显示出足够信号。"

    psi_values = [
        float(metric[key])
        for key in ("psi", *_PSI_VIEWS)
        if metric.get(key) is not None
    ]
    psi = max(psi_values) if psi_values else None
    sample_note = "样本量偏小，建议扩大样本复核；" if valid_count < 100 else ""
    if psi is None:
        return "候选", f"{sample_note}有区分力；本次没有可用 PSI 结果，需另行验证稳定性。"
    if psi > 0.25:
        return "不推荐", f"最大 PSI={psi:.3f}，稳定性风险较高。"
    if psi > 0.10:
        return "谨慎", f"{sample_note}有区分力，但最大 PSI={psi:.3f} 需要关注。"
    return "推荐", f"{sample_note}区分力与已提供的稳定性结果均通过基础规则。"


def _recommendation_evidence(metric: dict) -> list[dict]:
    evidence: list[dict] = []
    for name in (
        "iv",
        "ks",
        "auc",
        "missing_rate",
        "mode_rate",
        "zero_rate",
        "unique_count",
        "psi",
        *_PSI_VIEWS,
        "meaning_consistency",
    ):
        if name not in metric or metric.get(name) is None:
            continue
        value = metric.get(name)
        if isinstance(value, (str, bool, int, float)):
            evidence.append({"metric": name, "value": value})
    return evidence


def _recommendation_confidence(
    recommendation: str,
    evidence: list[dict],
) -> str:
    if recommendation == "待评估":
        return "none"
    names = {str(item.get("metric") or "") for item in evidence}
    signal = names & {"iv", "ks", "auc"}
    stability = names & {"psi", *_PSI_VIEWS}
    if signal and stability:
        return "high"
    if signal or recommendation in {"不推荐", "暂不推荐"}:
        return "medium"
    return "low"


def tool_screen_features(inputs: dict, ctx) -> dict:
    """Leakage-aware feature screening (spec form B §4 backend; shared screen with
    MODELING via marvis.feature.screen). Flags hard leakage (KS>=leakage_ks), model-output
    names, and unusable (constant/sparse) columns, and ranks the rest — yielding a selected
    feature set for the downstream model.

    For a non-binary target (``target_type != "binary"``, e.g. a regression task) the
    leakage KS screen is skipped: ``feature_ks`` is a binary-only statistic and would
    miscompute or crash on a continuous target. In that case candidates are ranked by a
    target-type-appropriate association score after the same semantic and usability gates."""
    target_type = str(inputs.get("target_type", "binary"))
    if target_type != "binary":
        return _screen_features_non_binary(inputs, ctx)

    from marvis.feature.screen import (
        DEFAULT_SCREEN_BATCH_SIZE,
        screen_features,
        sentinel_screen_notice,
    )

    runtime = _runtime(ctx)
    dataset = _task_dataset(runtime, ctx, inputs["dataset_id"])
    split_col = inputs.get("split_col")
    requested_features = inputs.get("features") or []
    features = _resolve_feature_cols(
        runtime,
        dataset.id,
        requested_features,
        target_col=str(inputs["target_col"]),
        split_col=str(split_col) if split_col else None,
    )
    excluded_categorical = _excluded_categorical_for_screen(
        runtime,
        dataset.id,
        requested_features,
        target_col=str(inputs["target_col"]),
        split_col=str(split_col) if split_col else None,
    )
    suspected_categorical = _suspected_categorical_for_screen(
        runtime,
        dataset.id,
        target_col=str(inputs["target_col"]),
        split_col=str(split_col) if split_col else None,
    )
    holdout = inputs.get("holdout_values")
    top_k = inputs.get("top_k")
    result = screen_features(
        runtime.backend,
        runtime.registry.resolve_path(dataset.id),
        features=features,
        target_col=str(inputs["target_col"]),
        split_col=str(split_col) if split_col else None,
        sample_weight_col=str(inputs["sample_weight_col"]) if inputs.get("sample_weight_col") else None,
        holdout_values=tuple(str(value) for value in holdout) if holdout else ("oot",),
        leakage_ks=float(inputs.get("leakage_ks", 0.40)),
        max_missing_rate=float(inputs.get("max_missing_rate", 0.95)),
        top_k=int(top_k) if top_k is not None else None,
        batch_size=int(inputs.get("batch_size", DEFAULT_SCREEN_BATCH_SIZE)),
        max_ks_decay=float(inputs["max_ks_decay"]) if inputs.get("max_ks_decay") is not None else None,
        drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    payload = {
        "selected": list(result.selected),
        "ranked": [[feature, ks] for feature, ks in result.ranked],
        "leakage": [[feature, ks, reason] for feature, ks, reason in result.leakage],
        "suspected": [[feature, ks, reason] for feature, ks, reason in result.suspected],
        "unusable": [[feature, reason] for feature, reason in result.unusable],
        "scores": _jsonable(result.scores),
        "n_screened": result.n_screened,
        "nan_labels_dropped": result.nan_labels_dropped,
        "excluded_categorical": excluded_categorical,
    }
    if suspected_categorical:
        payload["suspected_categorical"] = suspected_categorical
    if result.split_shift:
        payload["split_shift"] = [[feature, delta, reason] for feature, delta, reason in result.split_shift]
    if result.leakage_watch:
        payload["leakage_watch"] = [[feature, ks, reason] for feature, ks, reason in result.leakage_watch]
    if result.ks_decay_watch:
        payload["ks_decay_watch"] = [[feature, decay, reason] for feature, decay, reason in result.ks_decay_watch]
    if result.sentinel_columns:
        payload["sentinel_columns"] = _jsonable(result.sentinel_columns)
        payload["sentinel_notice"] = sentinel_screen_notice(result.sentinel_columns)
    return payload


def _excluded_categorical_for_screen(
    runtime: "_Runtime",
    dataset_id: str,
    requested_features: list,
    *,
    target_col: str,
    split_col: str | None,
) -> list[dict]:
    """String/object columns silently dropped by candidate inference (PREP-3/FS-3).

    Only meaningful when ``features`` was NOT explicitly provided — an explicit
    feature list is the caller's own choice, not an inference the platform made
    on their behalf, so there is nothing to surface."""
    if [str(item) for item in requested_features if str(item).strip()]:
        return []
    dataset = runtime.registry.get(str(dataset_id))
    excluded = excluded_categorical_columns(
        runtime.backend,
        runtime.registry.resolve_path(dataset.id),
        target_col=target_col,
        split_col=split_col,
    )
    return [{"column": item.column, "cardinality": item.cardinality} for item in excluded]


def _suspected_categorical_for_screen(
    runtime: "_Runtime",
    dataset_id: str,
    *,
    target_col: str,
    split_col: str | None,
) -> list[dict]:
    """Numeric columns that look like nominal codes rather than continuous measures
    (PREP-5), e.g. a zip/industry code — surfaced as a screen-gate hint, always (even
    with an explicit feature list) since these columns keep being modeled as continuous
    numeric today; nothing about candidate inference or the selected set changes."""
    dataset = runtime.registry.get(str(dataset_id))
    suspected = suspected_categorical_columns(
        runtime.backend,
        runtime.registry.resolve_path(dataset.id),
        target_col=target_col,
        split_col=split_col,
    )
    return [{"column": item.column, "cardinality": item.cardinality} for item in suspected]


def _screen_features_non_binary(inputs: dict, ctx) -> dict:
    """Screen path for a non-binary (continuous/multiclass) target: the binary-only leakage
    KS screen is skipped, but unusable columns are still dropped into ``unusable`` — mirroring
    the binary screen — namely constant (unique_count<=1) or mostly-missing
    (missing_rate>=max_missing_rate) columns; the rest are kept as selected (ks=None)."""
    from marvis.feature.screen import (
        DEFAULT_SCREEN_BATCH_SIZE,
        screen_features_non_binary,
    )

    runtime = _runtime(ctx)
    dataset = _task_dataset(runtime, ctx, inputs["dataset_id"])
    holdout = inputs.get("holdout_values")
    features = _resolve_feature_cols(
        runtime,
        dataset.id,
        inputs.get("features") or [],
        target_col=str(inputs["target_col"]),
        split_col=str(inputs["split_col"]) if inputs.get("split_col") else None,
    )
    result = screen_features_non_binary(
        runtime.backend,
        runtime.registry.resolve_path(dataset.id),
        features=features,
        target_col=str(inputs["target_col"]),
        target_type=str(inputs.get("target_type") or "continuous"),
        split_col=str(inputs["split_col"]) if inputs.get("split_col") else None,
        sample_weight_col=str(inputs["sample_weight_col"]) if inputs.get("sample_weight_col") else None,
        holdout_values=tuple(str(value) for value in holdout) if holdout else ("oot",),
        max_missing_rate=float(inputs.get("max_missing_rate", 0.95)),
        top_k=int(inputs["top_k"]) if inputs.get("top_k") is not None else None,
        batch_size=int(inputs.get("batch_size", DEFAULT_SCREEN_BATCH_SIZE)),
        drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    return {
        "selected": list(result.selected),
        "ranked": [[feature, ks] for feature, ks in result.ranked],
        "leakage": [
            [feature, ks, reason] for feature, ks, reason in result.leakage
        ],
        "suspected": [],
        "unusable": [[feature, reason] for feature, reason in result.unusable],
        "scores": _jsonable(result.scores),
        "n_screened": result.n_screened,
        "nan_labels_dropped": result.nan_labels_dropped,
        "note": "非二分类目标：跳过统计型泄漏KS筛选；语义/时序泄漏与控制列仍硬剔除",
    }


def tool_generate_feature_report(inputs: dict, ctx) -> dict:
    """Write the per-feature metrics into a downloadable Excel report (FEATURE form A)."""
    from marvis.output.feature_report import render_feature_report

    metrics = [item for item in (inputs.get("metrics") or []) if isinstance(item, dict)]
    collinear = inputs.get("collinear") if isinstance(inputs.get("collinear"), dict) else None
    binning = [item for item in (inputs.get("binning") or []) if isinstance(item, dict)]
    settings = build_settings(ctx.workspace)
    out_path = Path(settings.tasks_dir) / ctx.task_id / "outputs" / "feature_report.xlsx"
    render_feature_report(metrics, out_path, collinear=collinear, binning=binning)
    # Echo metrics (+ optional collinear) so the driver renders the wide table, the VIF
    # section, and the report link together.
    out = {
        "report_path": str(out_path),
        "feature_count": len(metrics),
        "metrics": metrics,
        "binning": binning,
    }
    if collinear is not None:
        out["collinear"] = collinear
    return out


def tool_analyze_feature_bins(inputs: dict, ctx) -> dict:
    """Compute optional get_eval_table-style details for user-selected features."""
    from marvis.feature.bin_analysis import feature_bin_analysis

    requested = [str(item).strip() for item in (inputs.get("features") or []) if str(item).strip()]
    requested = list(dict.fromkeys(requested))
    requested_bins = int(inputs.get("bins") or 10)
    if requested_bins < 3 or requested_bins > 20:
        raise FeatureError("分箱数量必须是 3 到 20 之间的整数")
    runtime = _runtime(ctx)
    dataset = _task_dataset(runtime, ctx, inputs["dataset_id"])
    if not requested:
        return {
            "dataset_id": dataset.id,
            "selected_features": [],
            "requested_bins": requested_bins,
            "binning": [],
            "nan_labels_dropped": 0,
        }
    _dataset, frame = _read_frame(
        runtime,
        ctx,
        dataset.id,
        _unique([*requested, str(inputs["target_col"])]),
    )
    nan_labels_dropped = require_labels_confirmed(
        frame,
        str(inputs["target_col"]),
        drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    target = _target_values(frame, str(inputs["target_col"]))
    binning = [
        feature_bin_analysis(
            frame[feature].to_numpy(dtype=float),
            target,
            feature=feature,
            requested_bins=requested_bins,
        )
        for feature in requested
    ]
    return {
        "dataset_id": dataset.id,
        "selected_features": requested,
        "requested_bins": requested_bins,
        "binning": _jsonable(binning),
        "nan_labels_dropped": nan_labels_dropped,
    }


def tool_bin_feature(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    _dataset, frame = _read_frame(
        runtime,
        ctx,
        str(inputs["dataset_id"]),
        [str(inputs["feature"]), str(inputs["target_col"])],
    )
    nan_labels_dropped = require_labels_confirmed(
        frame,
        str(inputs["target_col"]),
        drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    feature = str(inputs["feature"])
    target_col = str(inputs["target_col"])
    target = _target_values(frame, target_col)
    sentinel_values = inputs.get("sentinel_values")
    if sentinel_values:
        frame = frame.copy()
        frame[feature] = mask_sentinel_values(frame[feature], [float(v) for v in sentinel_values])
    values = frame[feature].to_numpy(dtype=float)
    edges = _edges_for(frame, inputs, ctx)
    before = None
    resolved_direction = None
    if bool(inputs.get("enforce_monotonic")):
        before = compute_woe_iv(values, target, edges, feature=feature)
        resolved_direction = monotonic_direction(
            values,
            target,
            edges,
            direction=str(inputs.get("monotonic_direction") or "auto"),
        )
        edges = monotonic_edges(values, target, edges, direction=resolved_direction)
    result = compute_woe_iv(
        values,
        target,
        edges,
        feature=feature,
    )
    payload = _jsonable(result)
    payload["bins"] = [_jsonable(bin_row) for bin_row in result.bins]
    payload["na_bin"] = _jsonable(result.na_bin) if result.na_bin else None
    payload["nan_labels_dropped"] = nan_labels_dropped
    if before is not None:
        payload["monotonic_enforced"] = True
        payload["monotonic_direction"] = resolved_direction
        payload["monotonic_before"] = bool(before.monotonic)
        payload["total_iv_before_monotonic"] = before.total_iv
        payload["edges_before_monotonic"] = [float(value) for value in before.edges]
    return payload


def tool_compute_psi(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    feature = str(inputs["feature"])
    columns = _unique([feature, *_filter_columns(inputs.get("base_filter")), *_filter_columns(inputs.get("compare_filter"))])
    dataset, frame = _read_frame(
        runtime,
        ctx,
        str(inputs["dataset_id"]),
        columns,
    )
    base = _apply_filter(frame, inputs.get("base_filter"))[feature].to_numpy(dtype=float)
    compare = _apply_filter(frame, inputs.get("compare_filter"))[feature].to_numpy(dtype=float)
    edges = equal_frequency_edges(base, int(inputs.get("bins") or 10))
    psi = feature_psi(base, compare, edges)
    return {
        "dataset_id": dataset.id,
        "feature": feature,
        "psi": float(psi),
        "edges": _jsonable(edges),
        "bin_distributions": {
            "base": _jsonable(_bin_distribution(base, edges)),
            "compare": _jsonable(_bin_distribution(compare, edges)),
        },
    }


def tool_correlation_analysis(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset, frame = _read_frame(
        runtime,
        ctx,
        str(inputs["dataset_id"]),
        [str(item) for item in inputs["features"]],
    )
    report = correlation_report(
        frame,
        [str(item) for item in inputs["features"]],
        method=str(inputs.get("method") or "pearson"),
        threshold=float(inputs.get("threshold", 0.8)),
    )
    payload = _jsonable(report)
    payload["dataset_id"] = dataset.id
    return payload


def tool_woe_encode(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    features = [str(item) for item in inputs["features"]]
    target_col = str(inputs["target_col"])
    dataset, frame = _read_frame(runtime, ctx, str(inputs["dataset_id"]))
    required_columns = [*features, target_col]
    if inputs.get("split_col"):
        required_columns.append(str(inputs["split_col"]))
    _assert_columns(frame, required_columns)
    out = frame.copy()
    sentinel_values = _sentinel_values_for(inputs, features)
    for feature, column_sentinels in sentinel_values.items():
        out[feature] = mask_sentinel_values(out[feature], column_sentinels)
    fit_frame, fit_split = _woe_fit_frame(out, inputs, dataset.id)
    nan_labels_dropped = require_labels_confirmed(
        fit_frame,
        target_col,
        drop_nan_labels=bool(inputs.get("drop_nan_labels")),
        scope="woe fit",
    )
    woe_maps = {}
    new_columns = []
    for feature in features:
        edges = _edges_for(fit_frame, {**inputs, "feature": feature}, ctx)
        binning = compute_woe_iv(
            fit_frame[feature].to_numpy(dtype=float),
            _target_values(fit_frame, target_col),
            edges,
            feature=feature,
        )
        woe = woe_result_from_binning(binning)
        encoded = woe_encode(out, feature, woe)
        out[encoded.name] = encoded
        new_columns.append(encoded.name)
        woe_maps[feature] = _jsonable(woe)
    sentinel_step = _sentinel_step(sentinel_values)
    result = _register_frame(
        runtime,
        out,
        dataset,
        ctx,
        "woe",
        preprocessing_steps=[
            *([sentinel_step] if sentinel_step else []),
            {"kind": "woe", "columns": features, "params": _jsonable(woe_maps)},
        ],
    )
    return {
        "result_dataset_id": result.id,
        "new_columns": new_columns,
        "woe_maps": woe_maps,
        "nan_labels_dropped": nan_labels_dropped,
        "fit_rows": int(len(fit_frame)),
        "fit_split": fit_split,
    }


def _woe_fit_frame(
    frame: pd.DataFrame, inputs: dict, dataset_id: str, *, tool: str = "woe_encode"
) -> tuple[pd.DataFrame, str]:
    """Rows used to fit the WOE mapping — excludes holdout (default test+OOT) so the
    mapping never peeks at evaluation labels (PREP-1). No ``split_col`` means the caller
    cannot express train-only fitting; that's a typed-error stop unless the caller
    explicitly confirms a full-pool fit via ``allow_full_fit``."""
    split_col = inputs.get("split_col")
    if not split_col:
        if bool(inputs.get("allow_full_fit")):
            return frame, "full"
        raise FitRequiresSplitError(tool=tool, dataset_id=dataset_id)
    holdout_values = tuple(str(value) for value in (inputs.get("holdout_values") or ("test", "oot")))
    mask = ~frame[str(split_col)].astype(str).isin(holdout_values)
    fit_frame = frame.loc[mask]
    if fit_frame.empty:
        raise FeatureError("WOE fit frame is empty after excluding holdout rows")
    return fit_frame, "train"


def tool_woe_encode_categorical(inputs: dict, ctx) -> dict:
    """Category -> WOE encode string/object columns (PREP-3/FS-3) — the categorical
    analogue of ``woe_encode``. Same train-only fitting contract: fits on the non-
    holdout rows (default excludes test+OOT) and raises ``FitRequiresSplitError``
    unless ``split_col`` is given or the caller passes ``allow_full_fit=true``."""
    runtime = _runtime(ctx)
    features = [str(item) for item in inputs["features"]]
    target_col = str(inputs["target_col"])
    dataset, frame = _read_frame(runtime, ctx, str(inputs["dataset_id"]))
    required_columns = [*features, target_col]
    if inputs.get("split_col"):
        required_columns.append(str(inputs["split_col"]))
    _assert_columns(frame, required_columns)
    out = frame.copy()
    fit_frame, fit_split = _woe_fit_frame(out, inputs, dataset.id, tool="woe_encode_categorical")
    nan_labels_dropped = require_labels_confirmed(
        fit_frame,
        target_col,
        drop_nan_labels=bool(inputs.get("drop_nan_labels")),
        scope="categorical woe fit",
    )
    min_count = inputs.get("min_count")
    smoothing = float(inputs.get("smoothing", 0.5))
    woe_maps = {}
    new_columns = []
    for feature in features:
        woe = categorical_woe_encode(
            fit_frame[feature],
            _target_values(fit_frame, target_col),
            feature=feature,
            min_count=int(min_count) if min_count is not None else None,
            smoothing=smoothing,
        )
        encoded = apply_categorical_woe(out, feature, woe)
        out[encoded.name] = encoded
        new_columns.append(encoded.name)
        woe_maps[feature] = _jsonable(woe)
    result = _register_frame(
        runtime,
        out,
        dataset,
        ctx,
        "catwoe",
        preprocessing_step={"kind": "categorical_woe", "columns": features, "params": _jsonable(woe_maps)},
    )
    return {
        "result_dataset_id": result.id,
        "new_columns": new_columns,
        "woe_maps": woe_maps,
        "nan_labels_dropped": nan_labels_dropped,
        "fit_rows": int(len(fit_frame)),
        "fit_split": fit_split,
    }


def tool_onehot_encode(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset, frame = _read_frame(runtime, ctx, str(inputs["dataset_id"]))
    columns = [str(item) for item in inputs["columns"]]
    _assert_columns(frame, columns)
    encoded, mapping = onehot_encode(
        frame,
        columns,
        max_categories=int(inputs.get("max_categories") or 50),
    )
    result = _register_frame(
        runtime,
        encoded,
        dataset,
        ctx,
        "onehot",
        preprocessing_step={"kind": "onehot", "columns": columns, "params": _jsonable(mapping)},
    )
    return {"result_dataset_id": result.id, "mapping": _jsonable(mapping)}


def tool_normalize(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset, frame = _read_frame(runtime, ctx, str(inputs["dataset_id"]))
    method = str(inputs["method"])
    out = frame.copy()
    params = {}
    columns = [str(column) for column in inputs["columns"]]
    _assert_columns(out, columns)
    fit_mask, fit_split = _stat_fit_mask(out, inputs, "normalize", dataset.id)
    sentinel_values = _sentinel_values_for(inputs, columns)
    for col in columns:
        column_sentinels = sentinel_values.get(col)
        fit_series = mask_sentinel_values(out.loc[fit_mask, col], column_sentinels)
        fit_values = fit_series.to_numpy(dtype=float)
        full_series = mask_sentinel_values(out[col], column_sentinels)
        full_values = full_series.to_numpy(dtype=float)
        if method == "minmax":
            _fit_values, column_params = minmax_normalize(
                fit_values,
                feature_range=tuple(inputs.get("feature_range") or (0, 1)),
            )
            values = apply_scaler(full_values, column_params, kind="minmax")
        elif method == "zscore":
            _fit_values, column_params = zscore_standardize(fit_values)
            values = apply_scaler(full_values, column_params, kind="zscore")
        else:
            raise FeatureError("method must be minmax or zscore")
        out[col] = values
        params[col] = column_params
    sentinel_step = _sentinel_step(sentinel_values)
    result = _register_frame(
        runtime,
        out,
        dataset,
        ctx,
        "normalize",
        preprocessing_steps=[
            *([sentinel_step] if sentinel_step else []),
            {"kind": "normalize", "columns": columns, "params": _jsonable(params)},
        ],
    )
    return {
        "result_dataset_id": result.id,
        "scaler_params": _jsonable(params),
        "fit_rows": int(fit_mask.sum()),
        "fit_split": fit_split,
    }


def tool_impute_missing(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset, frame = _read_frame(runtime, ctx, str(inputs["dataset_id"]))
    out = frame.copy()
    fill_values = {}
    indicators = {}
    columns = [str(column) for column in inputs["columns"]]
    _assert_columns(out, columns)
    fit_mask, fit_split = _stat_fit_mask(out, inputs, "impute_missing", dataset.id)
    sentinel_values = _sentinel_values_for(inputs, columns)
    add_indicators = bool(inputs.get("add_indicators"))
    indicator_columns: list[str] = []
    for column in columns:
        column_sentinels = sentinel_values.get(column)
        _filled_fit, value = impute_missing(
            out.loc[fit_mask, column],
            strategy=str(inputs["strategy"]),
            fill_value=inputs.get("fill_value"),
            sentinel_values=column_sentinels,
        )
        masked = mask_sentinel_values(out[column], column_sentinels)
        if add_indicators and masked.isna().any():
            # PREP-8: preserve the "missing" signal that plain imputation erases —
            # a col__was_missing 0/1 column, guarded against colliding with an
            # existing column name.
            indicator_name = _unique_column_name(f"{column}__was_missing", out.columns)
            out[indicator_name] = masked.isna().astype(int)
            indicators[column] = indicator_name
            indicator_columns.append(indicator_name)
        out[column] = masked.fillna(value)
        fill_values[column] = value
    preprocessing_steps = []
    sentinel_step = _sentinel_step(sentinel_values)
    if sentinel_step:
        # A3: mask raw sentinels to NaN first so the missing_indicator/impute steps
        # below see NaN, reproducing the fit-time indicator and train-only fill.
        preprocessing_steps.append(sentinel_step)
    if indicators:
        # Ordered before "impute" so replay computes the pre-fill NaN mask first.
        preprocessing_steps.append(
            {"kind": "missing_indicator", "columns": list(indicators), "params": _jsonable(indicators)}
        )
    preprocessing_steps.append({"kind": "impute", "columns": columns, "params": _jsonable(fill_values)})
    result = _register_frame(
        runtime,
        out,
        dataset,
        ctx,
        "impute",
        preprocessing_steps=preprocessing_steps,
    )
    return {
        "result_dataset_id": result.id,
        "fill_values": _jsonable(fill_values),
        "fit_rows": int(fit_mask.sum()),
        "fit_split": fit_split,
        "indicator_columns": indicator_columns,
    }


def tool_cap_outliers(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset, frame = _read_frame(runtime, ctx, str(inputs["dataset_id"]))
    out = frame.copy()
    bounds = {}
    columns = [str(column) for column in inputs["columns"]]
    _assert_columns(out, columns)
    fit_mask, fit_split = _stat_fit_mask(out, inputs, "cap_outliers", dataset.id)
    sentinel_values = _sentinel_values_for(inputs, columns)
    for column in columns:
        column_sentinels = sentinel_values.get(column)
        fit_values = out.loc[fit_mask, column].to_numpy(dtype=float)
        _capped_fit, params = cap_outliers(
            fit_values,
            method=str(inputs.get("method") or "iqr"),
            lower_q=float(inputs.get("lower_q", 0.01)),
            upper_q=float(inputs.get("upper_q", 0.99)),
            sentinel_values=column_sentinels,
        )
        all_values = mask_sentinel_values(out[column], column_sentinels).to_numpy(dtype=float)
        mask = np.isfinite(all_values)
        clipped = all_values.copy()
        lower = params["lower"]
        upper = params["upper"]
        if np.isfinite(lower) and np.isfinite(upper):
            clipped[mask] = np.clip(clipped[mask], lower, upper)
        out[column] = clipped
        bounds[column] = params
    sentinel_step = _sentinel_step(sentinel_values)
    result = _register_frame(
        runtime,
        out,
        dataset,
        ctx,
        "cap",
        preprocessing_steps=[
            *([sentinel_step] if sentinel_step else []),
            {"kind": "cap", "columns": columns, "params": _jsonable(bounds)},
        ],
    )
    return {
        "result_dataset_id": result.id,
        "bounds": _jsonable(bounds),
        "fit_rows": int(fit_mask.sum()),
        "fit_split": fit_split,
    }


def _sentinel_values_for(inputs: dict, columns: list[str]) -> dict[str, list[float]]:
    """Resolve the ``sentinel_values`` input (PREP-4) into a per-column map.

    Accepts either a flat list (applied to every column in ``columns``) or a
    ``{column: [values, ...]}`` mapping (applied per-column only). Columns with
    no sentinel values configured are omitted from the result."""
    raw = inputs.get("sentinel_values")
    if not raw:
        return {}
    if isinstance(raw, dict):
        return {
            str(column): [float(v) for v in values]
            for column, values in raw.items()
            if str(column) in columns and values
        }
    flat = [float(v) for v in raw]
    return {column: flat for column in columns} if flat else {}


def _sentinel_step(sentinel_values: dict[str, list[float]]) -> dict[str, Any] | None:
    """A3: build a ``sentinel`` preprocessing step carrying the per-column sentinel
    values a tool masked before fitting, or ``None`` when no sentinels were used.

    Emitted *first* in the tool's chain (ahead of the paired impute/cap/normalize/
    woe step) so scoring-time replay masks raw sentinels to NaN before any downstream
    transform runs -- without it, a raw ``-999`` at serve time is treated as a genuine
    value (train/serve skew)."""
    if not sentinel_values:
        return None
    return {
        "kind": "sentinel",
        "columns": list(sentinel_values),
        "params": _jsonable(sentinel_values),
    }


def _unique_column_name(candidate: str, existing) -> str:
    """A column name guaranteed not to collide with ``existing`` (PREP-8): appends
    an incrementing numeric suffix (``_2``, ``_3``, ...) until it is unique."""
    existing_set = set(str(column) for column in existing)
    if candidate not in existing_set:
        return candidate
    suffix = 2
    while f"{candidate}_{suffix}" in existing_set:
        suffix += 1
    return f"{candidate}_{suffix}"


def _stat_fit_mask(frame: pd.DataFrame, inputs: dict, tool: str, dataset_id: str) -> tuple[np.ndarray, str]:
    """Rows used to fit statistical transforms (impute/normalize/cap) — excludes holdout
    (default test+OOT) so fill values / scaler params / capping bounds never absorb
    evaluation-set distribution (PREP-1). No ``split_col`` means the caller cannot
    express train-only fitting; that's a typed-error stop unless the caller explicitly
    confirms a full-pool fit via ``allow_full_fit``."""
    split_col = inputs.get("split_col")
    if not split_col:
        if bool(inputs.get("allow_full_fit")):
            return np.ones(len(frame), dtype=bool), "full"
        raise FitRequiresSplitError(tool=tool, dataset_id=dataset_id)
    _assert_columns(frame, [str(split_col)])
    holdout_values = tuple(str(value) for value in (inputs.get("holdout_values") or ("test", "oot")))
    mask = (~frame[str(split_col)].astype(str).isin(holdout_values)).to_numpy()
    if not mask.any():
        raise FeatureError(f"{tool} fit frame is empty after excluding holdout rows")
    return mask, "train"


def tool_cross_features(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset, frame = _read_frame(runtime, ctx, str(inputs["dataset_id"]))
    derived, new_columns = derive_batch(frame, list(inputs["recipe"]), dataset_id=dataset.id)
    result = _register_frame(runtime, derived, dataset, ctx, "cross")
    return {"result_dataset_id": result.id, "new_columns": new_columns}


def tool_derive_date_features(inputs: dict, ctx) -> dict:
    """Derive datediff/month/tenure-months numeric columns from date-role columns
    (PREP-7). Opt-in: never runs as part of any default template, so a caller must
    explicitly invoke it (with a date column identified e.g. via profiling/schema
    inference) to pull date information into the modeling frame."""
    runtime = _runtime(ctx)
    dataset, frame = _read_frame(runtime, ctx, str(inputs["dataset_id"]))
    derived, new_columns = derive_date_features(frame, list(inputs["recipe"]))
    result = _register_frame(runtime, derived, dataset, ctx, "datefeat")
    return {"result_dataset_id": result.id, "new_columns": new_columns}


class _Runtime(PackRuntime):
    pass


def _runtime(ctx) -> _Runtime:
    return _Runtime(ctx)


def _task_dataset(runtime: _Runtime, ctx, dataset_id):
    """Resolve a dataset only when it belongs to the invoking task."""

    dataset = runtime.registry.get(str(dataset_id))
    if str(dataset.task_id) != str(ctx.task_id):
        raise FeatureError("dataset does not belong to the active task")
    return dataset


def _resolve_feature_cols(
    runtime: _Runtime,
    dataset_id: str,
    features,
    *,
    target_col: str,
    split_col: str | None = None,
) -> list[str]:
    provided = _flatten_feature_cols(features)
    if provided:
        return provided
    dataset = runtime.registry.get(str(dataset_id))
    inferred = candidate_numeric_features(
        runtime.backend,
        runtime.registry.resolve_path(dataset.id),
        target_col=str(target_col),
        split_col=split_col,
    )
    if not inferred:
        raise FeatureError("未找到可用候选特征列;请检查拼接结果或指定特征列。")
    return inferred


def _flatten_feature_cols(features) -> list[str]:
    """Flatten a features input that may be a union of lists (FS-5): a workflow can pass
    ``features=[<base cols>, <$ref new_columns>]`` which resolves to nested lists; screen
    them together as one de-duplicated flat set (input order preserved)."""
    flat: list[str] = []
    seen: set[str] = set()
    for item in (features or []):
        candidates = item if isinstance(item, (list, tuple)) else [item]
        for candidate in candidates:
            name = str(candidate).strip()
            if name and name not in seen:
                seen.add(name)
                flat.append(name)
    return flat


def _read_frame(
    runtime: _Runtime,
    ctx,
    dataset_id: str,
    columns: list[str] | None = None,
):
    dataset = _task_dataset(runtime, ctx, dataset_id)
    frame = runtime.backend.read_frame(runtime.registry.resolve_path(dataset.id), columns=columns)
    return dataset, frame


def _register_frame(
    runtime: _Runtime,
    frame: pd.DataFrame,
    source_dataset,
    ctx,
    suffix: str,
    *,
    preprocessing_step: dict[str, Any] | None = None,
    preprocessing_steps: list[dict[str, Any]] | None = None,
):
    out_path = runtime.datasets_root / ctx.task_id / "feature" / f"{source_dataset.id}_{suffix}.parquet"
    uow = ArtifactUnitOfWork()
    artifact = uow.stage_file(out_path.parent, out_path.name)
    try:
        frame.to_parquet(artifact.path, index=False)
        steps_to_append = list(preprocessing_steps or [])
        if preprocessing_step is not None:
            steps_to_append.append(preprocessing_step)
        if steps_to_append:
            # PREP-2/PREP-8: persist the accumulated preprocessing chain (source
            # dataset's chain + these new steps, in order) as a sidecar next to the
            # derived parquet, so a model trained downstream can replay every fit
            # param at scoring time instead of only seeing it in this tool's JSON
            # response. Staged via the same unit of work as the parquet so both
            # promote/commit atomically.
            source_path = None
            try:
                source_path = runtime.registry.resolve_path(source_dataset.id)
            except KeyError:
                source_path = None
            chain = read_preprocessing_chain(source_path) if source_path else []
            for step in steps_to_append:
                chain = [
                    *chain,
                    {
                        "kind": str(step["kind"]),
                        "columns": [str(c) for c in step["columns"]],
                        "params": step["params"],
                    },
                ]
            sidecar_name = sidecar_path(Path(out_path.name)).name
            sidecar_artifact = uow.stage_file(out_path.parent, sidecar_name)
            write_preprocessing_chain(sidecar_artifact.path, chain)
        register_kwargs = {
            "task_id": ctx.task_id,
            "role": "derived",
            "anchor_target": source_dataset.id,
            "seed": int(ctx.seed or 0),
        }
        register_on_connection = getattr(runtime.registry, "register_existing_on_connection", None)
        transaction = getattr(runtime.registry, "transaction", None)
        if callable(register_on_connection) and callable(transaction):
            return uow.finalize_with_connection(
                transaction,
                lambda conn: register_on_connection(conn, artifact.final_path, **register_kwargs),
            )
        return uow.finalize(
            lambda: runtime.registry.register_existing(artifact.final_path, **register_kwargs)
        )
    except Exception:
        uow.rollback()
        raise


def _edges_for(frame: pd.DataFrame, inputs: dict, ctx) -> np.ndarray:
    feature = str(inputs["feature"])
    target_col = str(inputs.get("target_col") or "")
    values = frame[feature].to_numpy(dtype=float)
    method = str(inputs.get("method") or "equal_frequency")
    max_bins = int(inputs.get("max_bins") or inputs.get("bins") or 10)
    # PREP-9: minimum bin share (default 5%) merges small bins so WOE stays stable
    # across time periods; only meaningful for the frequency-driven methods below
    # (equal_width/manual/tree already control bin size some other way).
    min_bin_pct = float(inputs.get("min_bin_pct", 0.05))
    if method in {"equal_frequency", "quantile"}:
        return equal_frequency_edges(values, max_bins, min_bin_pct=min_bin_pct)
    if method == "equal_width":
        return equal_width_edges(values, max_bins)
    if method == "manual":
        return manual_edges([float(item) for item in inputs.get("breakpoints") or []])
    if method == "chimerge":
        return chimerge_edges(
            values, _target_values(frame, target_col), max_bins=max_bins, min_bin_pct=min_bin_pct
        )
    if method == "tree":
        return tree_edges(values, _target_values(frame, target_col), max_bins=max_bins, seed=int(ctx.seed or 0))
    raise FeatureError("method must be equal_frequency, equal_width, manual, chimerge, or tree")


def _target_values(frame: pd.DataFrame, target_col: str) -> np.ndarray:
    if not target_col or target_col not in frame.columns:
        raise FeatureError(f"missing target column: {target_col}")
    return pd.to_numeric(frame[target_col], errors="coerce").to_numpy(dtype=float)


def _apply_filter(frame: pd.DataFrame, spec: Any) -> pd.DataFrame:
    if not spec:
        return frame
    if not isinstance(spec, dict):
        raise FeatureError("filter must be an object")
    if "column" not in spec:
        if len(spec) == 1:
            column, value = next(iter(spec.items()))
            return frame[frame[str(column)] == value]
        raise FeatureError("filter requires column, op, and value")
    column = str(spec["column"])
    op = str(spec.get("op") or "eq")
    value = spec.get("value")
    _assert_columns(frame, [column])
    series = frame[column]
    if op == "eq":
        mask = series == value
    elif op == "ne":
        mask = series != value
    elif op == "lt":
        mask = series < value
    elif op == "lte":
        mask = series <= value
    elif op == "gt":
        mask = series > value
    elif op == "gte":
        mask = series >= value
    elif op == "in":
        mask = series.isin(value or [])
    else:
        raise FeatureError("filter op must be eq, ne, lt, lte, gt, gte, or in")
    return frame[mask]


def _filter_columns(spec: Any) -> list[str]:
    if not isinstance(spec, dict) or not spec:
        return []
    if "column" in spec:
        return [str(spec["column"])]
    return [str(key) for key in spec]


def _bin_distribution(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    from marvis.feature.binning import assign_bins

    assigned = assign_bins(values, edges)
    valid = assigned >= 0
    if not np.any(valid):
        return np.zeros(len(edges) - 1, dtype=float)
    counts = np.bincount(assigned[valid], minlength=len(edges) - 1).astype(float)
    return counts / counts.sum()


def _assert_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise FeatureError(f"missing columns: {', '.join(missing)}")


def _unique(values: list[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value
