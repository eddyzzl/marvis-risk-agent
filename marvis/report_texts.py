from __future__ import annotations

from marvis.model_algorithms import model_training_description
from marvis.validation.results import (
    ConsistencyStatus,
    OverallRow,
    SplitRow,
    ValidationResults,
)


COMPUTED_REPORT_TEXT_KEYS = frozenset({
    "TEXT:sample_period",
    "TEXT:sample_start_month",
    "TEXT:sample_end_month",
    "TEXT:data_source_summary",
    "TEXT:dataset_split_summary",
    "TEXT:train_test_period",
    "TEXT:train_test_ratio",
    "TEXT:oot_period",
    "TEXT:train_count",
    "TEXT:test_count",
    "TEXT:oot_count",
    "TEXT:train_bad_rate",
    "TEXT:test_bad_rate",
    "TEXT:oot_bad_rate",
    "TEXT:oot_ks",
    "TEXT:oot_psi",
    "TEXT:reproducibility_summary",
    "TEXT:stress_test_summary",
    "TEXT:pressure_test_summary",
})

AGENT_CONFIRMED_REPORT_TEXT_KEYS = frozenset({
    "TEXT:pressure_impact_recommendation",
    "TEXT:final_validation_conclusion",
})


def report_text_values_from_results(
    results: ValidationResults,
    *,
    report_values: dict[str, str] | None = None,
    manual_values: dict[str, str] | None = None,
) -> dict[str, str]:
    overall_by_split = {row.split: row for row in results.effectiveness.overall}
    split_summary = {row.split: row for row in results.basic_info.split_summary}
    sample_period = _period_text(
        results.basic_info.sample_period[0],
        results.basic_info.sample_period[1],
    )
    version_suffix = f"{results.model_version}版" if results.model_version else ""

    stress_summary = _stress_text(results)
    values: dict[str, str] = {
        "TEXT:report_title": f"{results.model_name}模型{version_suffix}验证文档",
        "TEXT:model_name": results.model_name,
        "TEXT:model_version": results.model_version,
        "TEXT:algorithm": results.algorithm,
        "TEXT:model_training_description": model_training_description(results.algorithm),
        "TEXT:sample_period": sample_period,
        "TEXT:sample_start_month": results.basic_info.sample_period[0],
        "TEXT:sample_end_month": results.basic_info.sample_period[1],
        "TEXT:data_source_summary": f"建模样本覆盖 {sample_period}，平台基于样本文件自动识别样本周期和分布。",
        "TEXT:dataset_split_summary": _dataset_split_text(split_summary),
        "TEXT:train_test_period": _train_test_period_text(split_summary),
        "TEXT:train_test_ratio": _train_test_ratio_text(split_summary),
        "TEXT:oot_period": _split_period_text(split_summary.get("oot")),
        "TEXT:train_count": _count(split_summary.get("train")),
        "TEXT:test_count": _count(split_summary.get("test")),
        "TEXT:oot_count": _count(split_summary.get("oot")),
        "TEXT:train_bad_rate": _percent(split_summary.get("train")),
        "TEXT:test_bad_rate": _percent(split_summary.get("test")),
        "TEXT:oot_bad_rate": _percent(split_summary.get("oot")),
        "TEXT:oot_ks": _decimal4(overall_by_split.get("oot")),
        "TEXT:oot_psi": _decimal4_psi(overall_by_split.get("oot")),
        "TEXT:reproducibility_summary": _reproducibility_text(results),
        "TEXT:stress_test_summary": stress_summary,
        "TEXT:pressure_test_summary": stress_summary,
    }
    return merge_report_text_values(
        values,
        report_values=report_values,
        manual_values=manual_values,
    )


def computed_report_text_values_from_payload(payload: dict) -> dict[str, str]:
    basic_info = payload.get("basic_info", {})
    values: dict[str, str] = {}
    sample_period = basic_info.get("sample_period")
    if isinstance(sample_period, (list, tuple)) and len(sample_period) >= 2:
        start = str(sample_period[0])
        end = str(sample_period[1])
        period = _period_text(start, end)
        values.update({
            "TEXT:sample_period": period,
            "TEXT:sample_start_month": start,
            "TEXT:sample_end_month": end,
            "TEXT:data_source_summary": f"建模样本覆盖 {period}，平台基于样本文件自动识别样本周期和分布。",
        })

    split_summary = _split_rows_from_payload(basic_info.get("split_summary", []))
    if split_summary:
        values.update({
            "TEXT:dataset_split_summary": _dataset_split_text(split_summary),
            "TEXT:train_test_period": _train_test_period_text(split_summary),
            "TEXT:train_test_ratio": _train_test_ratio_text(split_summary),
            "TEXT:oot_period": _split_period_text(split_summary.get("oot")),
            "TEXT:train_count": _count(split_summary.get("train")),
            "TEXT:test_count": _count(split_summary.get("test")),
            "TEXT:oot_count": _count(split_summary.get("oot")),
            "TEXT:train_bad_rate": _percent(split_summary.get("train")),
            "TEXT:test_bad_rate": _percent(split_summary.get("test")),
            "TEXT:oot_bad_rate": _percent(split_summary.get("oot")),
        })

    for row in payload.get("effectiveness", {}).get("overall", []):
        if row.get("split") != "oot":
            continue
        oot = _overall_row_from_payload(row)
        values["TEXT:oot_ks"] = _decimal4(oot)
        values["TEXT:oot_psi"] = _decimal4_psi(oot)
        break
    return values


def merge_report_text_values(
    generated_values: dict[str, str],
    *,
    report_values: dict[str, str] | None = None,
    manual_values: dict[str, str] | None = None,
) -> dict[str, str]:
    values = dict(generated_values)
    for candidate_values, allowed_computed_keys in (
        (report_values, AGENT_CONFIRMED_REPORT_TEXT_KEYS),
        (manual_values, frozenset()),
    ):
        if not candidate_values:
            continue
        values.update({
            key: value
            for key, value in _with_text_prefix(candidate_values).items()
            if (
                (key not in COMPUTED_REPORT_TEXT_KEYS or key in allowed_computed_keys)
                and (key not in AGENT_CONFIRMED_REPORT_TEXT_KEYS or key in allowed_computed_keys)
            )
        })
    return _apply_report_text_aliases(values)


def _apply_report_text_aliases(values: dict[str, str]) -> dict[str, str]:
    values = dict(values)
    recommendation = values.get("TEXT:pressure_recommendation_summary")
    if recommendation is not None:
        values["TEXT:pressure_impact_recommendation"] = recommendation
    return values


def _with_text_prefix(values: dict[str, str]) -> dict[str, str]:
    return {
        key if key.startswith("TEXT:") else f"TEXT:{key}": str(value)
        for key, value in values.items()
        if value is not None
    }


def _split_rows_from_payload(rows) -> dict[str, SplitRow]:
    split_rows: dict[str, SplitRow] = {}
    if not isinstance(rows, list):
        return split_rows
    for row in rows:
        if not isinstance(row, dict) or not row.get("split"):
            continue
        split_rows[str(row["split"])] = SplitRow(
            split=str(row["split"]),
            sample_count=int(row.get("sample_count") or 0),
            bad_count=int(row.get("bad_count") or 0),
            bad_rate=float(row.get("bad_rate") or 0.0),
            period_start=str(row.get("period_start") or ""),
            period_end=str(row.get("period_end") or ""),
        )
    return split_rows


def _overall_row_from_payload(row: dict) -> OverallRow:
    return OverallRow(
        split=str(row.get("split") or ""),
        ks=float(row.get("ks") or 0.0),
        psi_vs_train=float(row.get("psi_vs_train") or 0.0),
        sample_count=int(row.get("sample_count") or 0),
        bad_rate=float(row.get("bad_rate") or 0.0),
        bad_count=int(row.get("bad_count") or 0),
        auc=float(row.get("auc") or 0.0),
        head_lift_5pct=_optional_float(row.get("head_lift_5pct")),
        tail_lift_5pct=_optional_float(row.get("tail_lift_5pct")),
    )


def _count(row) -> str:
    return str(row.sample_count) if row else "暂无样本数据，待复核"


def _percent(row) -> str:
    return f"{row.bad_rate:.2%}" if row else "暂无样本数据，待复核"


def _period_text(start: str, end: str) -> str:
    if not start and not end:
        return "暂无样本周期，待复核"
    if not start:
        return str(end)
    if not end:
        return str(start)
    return str(start) if start == end else f"{start}-{end}"


def _split_period_text(row: SplitRow | None) -> str:
    if not row or row.sample_count == 0:
        return "暂无样本周期，待复核"
    return _period_text(row.period_start, row.period_end)


def _train_test_period_text(split_summary: dict[str, SplitRow]) -> str:
    return _combined_split_period_text(
        split_summary.get("train"),
        split_summary.get("test"),
    )


def _train_test_ratio_text(split_summary: dict[str, SplitRow]) -> str:
    train_count = split_summary.get("train").sample_count if split_summary.get("train") else 0
    test_count = split_summary.get("test").sample_count if split_summary.get("test") else 0
    total = train_count + test_count
    if total == 0:
        return "暂无训练/测试样本，待复核"
    return f"{train_count / total:.2%}:{test_count / total:.2%}"


def _combined_split_period_text(*rows: SplitRow | None) -> str:
    starts = [
        row.period_start
        for row in rows
        if row and row.sample_count and row.period_start
    ]
    ends = [
        row.period_end
        for row in rows
        if row and row.sample_count and row.period_end
    ]
    if not starts and not ends:
        return "暂无样本周期，待复核"
    start = min(starts) if starts else ""
    end = max(ends) if ends else ""
    return _period_text(start, end)


def _dataset_split_text(split_summary: dict[str, SplitRow]) -> str:
    train = split_summary.get("train")
    test = split_summary.get("test")
    oot = split_summary.get("oot")
    return (
        f"训练样本 {train.sample_count if train else 0} 条，"
        f"测试样本 {test.sample_count if test else 0} 条，"
        f"OOT 样本 {oot.sample_count if oot else 0} 条；"
        f"训练/测试周期为 {_train_test_period_text(split_summary)}，"
        f"OOT 周期为 {_split_period_text(oot)}。"
    )


def _decimal4(row) -> str:
    return f"{row.ks:.4f}" if row else "暂无OOT模型效果数据，待复核"


def _decimal4_psi(row) -> str:
    return f"{row.psi_vs_train:.4f}" if row else "暂无OOT稳定性数据，待复核"


def _reproducibility_text(results: ValidationResults) -> str:
    summary = results.reproducibility.summary
    status_word = {
        ConsistencyStatus.PASS: "通过",
        ConsistencyStatus.REVIEW: "需复核",
        ConsistencyStatus.FAIL: "不通过",
    }[summary.status]
    return (
        f"对 {results.reproducibility.sample_size} 行抽样进行三方分数对比，"
        f"对齐 {summary.match_count} 行，差异 {summary.mismatch_count} 行，"
        f"最大绝对差 {summary.max_abs_diff:.6f}，可复现性验证 {status_word}。"
    )


def _stress_text(results: ValidationResults) -> str:
    items = []
    status_prefix = {
        "partial": "压力测试部分完成，需关注异常类别：",
        "failed": "压力测试未完成，需先修复异常：",
        "skipped": "压力测试未执行有效类别：",
    }.get(results.stress_test.status, "")
    if results.stress_test.unclassified_features:
        features = results.stress_test.unclassified_features
        items.append(
            f"未分类特征 {len(features)} 个：{_feature_name_preview(features)}"
        )
    for item in results.stress_test.per_category:
        if item.status == "skipped":
            items.append(f"{item.category}：未找到可用于压力测试的入模特征")
            continue
        if item.error or item.status == "error":
            items.append(f"{item.category}：{item.error}")
            continue
        delta = item.ks_delta if item.ks_delta is not None else 0.0
        psi = item.psi_vs_baseline if item.psi_vs_baseline is not None else 0.0
        items.append(
            f"{item.category}（置 -9999 {len(item.dropped_features)} 个特征）："
            f"KS 变化 {delta:+.4f}，PSI {psi:.4f}"
        )
    text = "；".join(items) or "无压力测试结果"
    return f"{status_prefix}{text}" if status_prefix else text


def _feature_name_preview(features: list[str], *, limit: int = 20) -> str:
    visible = features[:limit]
    text = "、".join(visible)
    if len(features) > limit:
        return f"{text} 等 {len(features)} 个"
    return text


def _optional_float(value) -> float | None:
    return None if value is None else float(value)
