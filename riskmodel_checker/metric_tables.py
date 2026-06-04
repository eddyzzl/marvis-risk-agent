from __future__ import annotations

from typing import Any

LAYOUT_BY_KEY = {
    "IMAGE:overall_model_effect": "kpi_cards",
    "IMAGE:loan_month_effect": "trend_table",
}

PSI_THRESHOLDS = [0.02, 0.10]

COLUMN_SPEC_BY_HEADER = {
    # split / time
    "数据集":        {"kind": "split-badge"},
    "月份":          {"kind": "text"},
    "时间范围":      {"kind": "period"},
    # counts / share
    "样本量":        {"kind": "databar", "color": "primary"},
    "样本占比":      {"kind": "text"},
    "样本总数":      {"kind": "databar", "color": "primary"},
    "坏样本量":      {"kind": "databar", "color": "neutral"},
    "逾期数量":      {"kind": "databar", "color": "neutral"},
    "累计占比":      {"kind": "text"},
    # rates / risk
    "逾期率":        {"kind": "percent-heat"},
    "累计逾期率":    {"kind": "percent-heat"},
    # discrimination
    "KS":            {"kind": "databar-primary"},
    "ks":            {"kind": "text"},
    "AUC":           {"kind": "databar", "color": "accent"},
    "5%头部lift":    {"kind": "databar", "color": "accent"},
    "5%尾部lift":    {"kind": "databar", "color": "accent"},
    "单组lift":      {"kind": "databar", "color": "accent"},
    "累计lift":      {"kind": "databar-primary"},
    # stability
    "PSI":            {"kind": "psi", "thresholds": PSI_THRESHOLDS},
    "PSI(首月基准)":  {"kind": "psi", "thresholds": PSI_THRESHOLDS},
    "PSI(尾月基准)":  {"kind": "psi", "thresholds": PSI_THRESHOLDS},
    "PSI(环比)":      {"kind": "psi", "thresholds": PSI_THRESHOLDS},
    "PSI vs baseline":{"kind": "psi", "thresholds": PSI_THRESHOLDS},
    # stress
    "类别":           {"kind": "text"},
    "KS_baseline":    {"kind": "text"},
    "KS_after":       {"kind": "databar", "color": "accent"},
    "KS_delta":       {"kind": "psi", "thresholds": [0.01, 0.03]},
    # feature
    "排名":           {"kind": "text"},
    "特征":           {"kind": "text"},
    "重要性":         {"kind": "databar-primary"},
}


def _column_specs_for(headers: list[str]) -> list[dict[str, Any]]:
    return [COLUMN_SPEC_BY_HEADER.get(header, {"kind": "text"}) for header in headers]


SECTION_THEME = {
    "样本情况": "cool-blue",
    "整体效果&稳定性": "warm-orange",
    "分月效果&稳定性": "deep-purple",
    "分箱排序性": "heatmap",
    "特征重要性": "cool-blue",
    "压力测试": "warning-red",
    "ROC&KS 曲线": "deep-purple",
}


def metric_table_sections_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not payload:
        return []

    basic_info = _as_dict(payload.get("basic_info"))
    effectiveness = _as_dict(payload.get("effectiveness"))
    stress_test = _as_dict(payload.get("stress_test"))
    roc_ks_curves = _as_dict(effectiveness.get("roc_ks_curves"))
    split_summary = _as_list(basic_info.get("split_summary"))
    monthly_distribution = _as_list(basic_info.get("monthly_distribution"))
    overall = _as_list(effectiveness.get("overall"))
    bin_tables = _as_dict(effectiveness.get("bin_tables"))
    feature_importance = _as_list(basic_info.get("feature_importance"))
    per_category = _as_list(stress_test.get("per_category"))

    sections = [
        {
            "title": "样本情况",
            "tables": [
                _table(
                    "IMAGE:sample_overall_distribution",
                    "样本总体分布",
                    ["数据集", "时间范围", "样本量", "样本占比", "坏样本量", "逾期率"],
                    _sample_overall_rows(split_summary),
                ),
                _table(
                    "IMAGE:sample_month_distribution",
                    "样本逐月分布",
                    ["月份", "样本量", "样本占比", "坏样本量", "逾期率"],
                    _sample_month_rows(monthly_distribution),
                ),
            ],
        },
        {
            "title": "整体效果&稳定性",
            "tables": [
                _table(
                    "IMAGE:overall_model_effect",
                    "整体效果&稳定性",
                    [
                        "数据集",
                        "时间范围",
                        "样本量",
                        "逾期率",
                        "坏样本量",
                        "KS",
                        "AUC",
                        "5%头部lift",
                        "5%尾部lift",
                        "PSI",
                    ],
                    _overall_model_effect_rows(overall, split_summary),
                )
            ],
        },
        {
            "title": "分月效果&稳定性",
            "tables": [
                _table(
                    "IMAGE:loan_month_effect",
                    "分月效果&稳定性",
                    [
                        "月份",
                        "样本量",
                        "逾期率",
                        "坏样本量",
                        "KS",
                        "AUC",
                        "5%头部lift",
                        "5%尾部lift",
                        "PSI(首月基准)",
                        "PSI(尾月基准)",
                        "PSI(环比)",
                    ],
                    _monthly_effect_rows(
                        _as_list(effectiveness.get("monthly_ks")),
                        _as_list(effectiveness.get("monthly_psi")),
                    ),
                )
            ],
        },
        {
            "title": "分箱排序性",
            "tables": [
                _table(
                    f"IMAGE:ranking_table_{split}",
                    title,
                    [
                        title,
                        "样本总数",
                        "累计占比",
                        "逾期数量",
                        "逾期率",
                        "累计逾期率",
                        "单组lift",
                        "累计lift",
                        "ks",
                    ],
                    _ranking_rows(_as_list(bin_tables.get(split))),
                )
                for split, title in (("train", "Train"), ("test", "Test"), ("oot", "OOT"))
            ],
        },
        {
            "title": "特征重要性",
            "tables": [
                _table(
                    "IMAGE:top20_feature_ranking",
                    "Top20 特征重要性",
                    ["排名", "特征", "类别", "重要性"],
                    [
                        [
                            _value(row.get("rank")),
                            _value(row.get("feature")),
                            _value(row.get("category") or row.get("类别")),
                            _decimal(row.get("importance"), digits=4),
                        ]
                        for row in feature_importance[:20]
                        if isinstance(row, dict)
                    ],
                )
            ],
        },
        {
            "title": "压力测试",
            "tables": [
                _table(
                    "IMAGE:pressure_ks_table",
                    "压力测试",
                    ["类别", "KS_baseline", "KS_after", "KS_delta", "PSI vs baseline"],
                    _pressure_test_rows(_as_dict(stress_test.get("baseline")), per_category),
                ),
            ],
        },
    ]
    if any(isinstance(roc_ks_curves.get(split), dict) for split in ("train", "test", "oot")):
        sections.append(_roc_ks_section(roc_ks_curves))
    return [_tag_section(section) for section in sections]


def _table(key: str, title: str, headers: list[str], rows: list[list[Any]]) -> dict[str, Any]:
    return {
        "key": key,
        "title": title,
        "headers": headers,
        "rows": rows,
        "layout": LAYOUT_BY_KEY.get(key, "table"),
        "column_specs": _column_specs_for(headers),
    }


def _tag_section(section: dict) -> dict:
    return {**section, "section_theme": SECTION_THEME.get(section["title"], "cool-blue")}


def _roc_ks_section(roc_ks_curves: dict[str, Any]) -> dict[str, Any]:
    curves: dict[str, dict[str, Any]] = {}
    for split in ("train", "test", "oot"):
        raw = roc_ks_curves.get(split)
        if not isinstance(raw, dict):
            continue
        curves[split] = {
            "fpr": [float(v) for v in raw.get("fpr") or []],
            "tpr": [float(v) for v in raw.get("tpr") or []],
            "ks_curve": [float(v) for v in raw.get("ks_curve") or []],
            "ks": _scalar(raw.get("ks")),
            "population_at_ks": _scalar(raw.get("population_at_ks")),
        }
    return {
        "title": "ROC&KS 曲线",
        "tables": [
            {
                "key": "ROC_KS_CURVES",
                "title": "ROC&KS 曲线",
                "layout": "roc_ks_curve",
                "headers": [],
                "rows": [],
                "column_specs": [],
                "curves": curves,
            }
        ],
    }


def _sample_overall_rows(rows: list[Any]) -> list[list[Any]]:
    total_count = sum(_number(row, "sample_count") for row in rows if isinstance(row, dict))
    return [
        [
            _value(row.get("split")),
            _period_text(row.get("period_start"), row.get("period_end"), default="-"),
            _integer(row.get("sample_count")),
            _percent(_ratio(_number(row, "sample_count"), total_count)),
            _integer(row.get("bad_count")),
            _percent(row.get("bad_rate")),
        ]
        for row in rows
        if isinstance(row, dict)
    ]


def _sample_month_rows(rows: list[Any]) -> list[list[Any]]:
    total_count = sum(_number(row, "sample_count") for row in rows if isinstance(row, dict))
    return [
        [
            _value(row.get("month")),
            _integer(row.get("sample_count")),
            _percent(_ratio(_number(row, "sample_count"), total_count)),
            _integer(row.get("bad_count")),
            _percent(row.get("bad_rate")),
        ]
        for row in rows
        if isinstance(row, dict)
    ]


def _overall_model_effect_rows(rows: list[Any], split_summary: list[Any]) -> list[list[Any]]:
    split_by_name = {
        str(row.get("split")): row
        for row in split_summary
        if isinstance(row, dict)
    }
    formatted_rows: list[list[Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        split = str(row.get("split") or "")
        split_row = split_by_name.get(split) or {}
        formatted_rows.append([
            _value(split),
            _period_text(split_row.get("period_start"), split_row.get("period_end"), default="-"),
            _integer(row.get("sample_count")),
            _percent(row.get("bad_rate")),
            _integer(row.get("bad_count")),
            _percent_point(row.get("ks")),
            _percent_point(row.get("auc")),
            _decimal(row.get("head_lift_5pct"), digits=2),
            _decimal(row.get("tail_lift_5pct"), digits=2),
            "BASE" if split == "train" else _decimal(row.get("psi_vs_train"), digits=3),
        ])
    return formatted_rows


def _monthly_effect_rows(monthly_ks: list[Any], monthly_psi: list[Any]) -> list[list[Any]]:
    by_month: dict[str, dict[str, Any]] = {}
    for row in monthly_ks:
        if isinstance(row, dict):
            by_month.setdefault(str(row.get("month") or ""), {}).update(row)
    for row in monthly_psi:
        if isinstance(row, dict):
            by_month.setdefault(str(row.get("month") or ""), {}).update(row)

    months = sorted(month for month in by_month if month)
    first_month = months[0] if months else ""
    last_month = months[-1] if months else ""
    return [
        [
            month,
            _integer(data.get("sample_count")),
            _percent(data.get("bad_rate")),
            _integer(data.get("bad_count")),
            _percent_point(data.get("ks")),
            _percent_point(data.get("auc")),
            _decimal(data.get("head_lift_5pct"), digits=2),
            _decimal(data.get("tail_lift_5pct"), digits=2),
            "BASE" if month == first_month else _decimal(data.get("psi_first_month"), digits=3),
            "BASE" if month == last_month else _decimal(data.get("psi_last_month"), digits=3),
            "-" if month == first_month else _decimal(data.get("psi_mom"), digits=3),
        ]
        for month, data in ((month, by_month[month]) for month in months)
    ]


def _ranking_rows(rows: list[Any]) -> list[list[Any]]:
    valid_rows = [row for row in rows if isinstance(row, dict)]
    total = sum(_number(row, "sample_count") for row in valid_rows)
    total_bad = sum(_number(row, "bad_count") for row in valid_rows)
    overall_bad_rate = _ratio(total_bad, total)
    cumulative_count = 0.0
    cumulative_bad = 0.0
    formatted_rows: list[list[Any]] = []
    for row in valid_rows:
        cumulative_count += _number(row, "sample_count")
        cumulative_bad += _number(row, "bad_count")
        cumulative_bad_rate = _ratio(cumulative_bad, cumulative_count)
        formatted_rows.append([
            _score_interval(row.get("score_lower"), row.get("score_upper")),
            _integer(row.get("sample_count")),
            _percent(_ratio(cumulative_count, total)),
            _integer(row.get("bad_count")),
            _percent(row.get("bad_rate")),
            _percent(cumulative_bad_rate),
            _decimal(row.get("lift"), digits=2),
            _decimal(_ratio(cumulative_bad_rate, overall_bad_rate), digits=2),
            _decimal(row.get("ks"), digits=4),
        ])
    return formatted_rows


def _pressure_test_rows(baseline: dict[str, Any], rows: list[Any]) -> list[list[Any]]:
    baseline_ks = baseline.get("ks")
    return [
        [
            _value(row.get("category")),
            _decimal(baseline_ks, digits=4),
            _decimal(row.get("ks_after"), digits=4),
            _decimal(row.get("ks_delta"), digits=4),
            _decimal(row.get("psi_vs_baseline"), digits=4),
        ]
        for row in rows
        if isinstance(row, dict)
    ]


def _score_interval(lower: Any, upper: Any) -> str:
    return f"[{_compact_number(lower)},{_compact_number(upper)}]"


def _compact_number(value: Any) -> str:
    numeric = _to_float(value)
    if numeric is None:
        return "-"
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.3f}".rstrip("0").rstrip(".")


def _period_text(start: Any, end: Any, *, default: str) -> str:
    start_text = str(start or "")
    end_text = str(end or "")
    if not start_text and not end_text:
        return default
    if not start_text:
        return end_text
    if not end_text:
        return start_text
    return start_text if start_text == end_text else f"{start_text}-{end_text}"


def _integer(value: Any) -> str:
    numeric = _to_float(value)
    return "-" if numeric is None else f"{numeric:,.0f}"


def _percent(value: Any) -> str:
    numeric = _to_float(value)
    return "-" if numeric is None else f"{numeric:.2%}"


def _percent_point(value: Any) -> str:
    numeric = _to_float(value)
    return "-" if numeric is None else f"{numeric * 100:.1f}"


def _decimal(value: Any, *, digits: int) -> str:
    numeric = _to_float(value)
    return "-" if numeric is None else f"{numeric:.{digits}f}"


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _number(row: dict[str, Any], key: str) -> float:
    return _to_float(row.get(key)) or 0.0


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _scalar(value: Any, default: float = 0.0) -> float:
    parsed = _to_float(value)
    return parsed if parsed is not None else default


def _value(value: Any) -> str:
    return "-" if value is None or value == "" else str(value)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
