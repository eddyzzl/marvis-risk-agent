"""External narrative DTOs for the deterministic modeling report producers.

Local reports retain their full tables. These adapters expose measured group
statistics and fixed structural labels, never report prose, model internals,
category values, sample records, or free-text errors.
"""
from __future__ import annotations

import re
from datetime import date

from marvis.llm_outbound import OutboundContextError, _DROP, _finite, _identifier, _mint_prompt


_MONTH = re.compile(r"(\d{4})(?:-(\d{2})(?:-(\d{2}))?|(\d{2}))")
_SPLITS = {"train", "test", "oot", "valid"}
_STATUS = {"ok", "passed", "failed", "skipped", "unavailable", "missing", "partial"}


def _rows(value):
    if not isinstance(value, (list, tuple)):
        return []
    if len(value) > 10000:
        raise OutboundContextError("指标列表过大，未截断风险证据或发送请求。")
    return [item for item in value if isinstance(item, dict)]


def _items(value):
    if not isinstance(value, dict):
        return ()
    if len(value) > 10000:
        raise OutboundContextError("指标分组过多，未截断风险证据或发送请求。")
    return value.items()


def _numeric(row, fields):
    return {key: row[key] for key in fields if key in row and (row[key] is None or _finite(row[key]))}


def _labels(row, fields):
    return {key: row[key] for key, allowed in fields.items()
            if isinstance(row.get(key), str) and row[key] in allowed}


def _period(value):
    if not isinstance(value, str) or not (match := _MONTH.fullmatch(value)):
        return None
    try:
        date(int(match[1]), int(match[2] or match[4]), int(match[3] or 1))
    except ValueError:
        return None
    return value


def _months(row, fields):
    return {key: value for key in fields if (value := _period(row.get(key))) is not None}


def _feature(row):
    value = _identifier(row.get("feature"))
    return {"feature": value} if value is not _DROP else {}


def _samples(value):
    mapping = {
        "放款笔数": "row_count", "未逾期": "good_count", "逾期": "bad_count", "逾期率": "bad_rate",
        "平均利率": "mean_interest_rate", "平均放款金额": "mean_loan_amount",
        "平均期数": "mean_term", "平均支用金额": "mean_drawdown_amount",
        "Mob3逾期率": "mob3_bad_rate", "Mob6逾期率": "mob6_bad_rate",
    }
    return [{"month": row["放款月"], **{mapping[key]: val for key, val in _numeric(row, mapping).items()}}
            for row in _rows(value) if _months(row, ["放款月"])]


def _vintage(value):
    if not isinstance(value, dict):
        return {}
    return {"points": [
        {**_months(row, ["cohort"]),
         **_numeric(row, ("mob", "sample_count", "bad_count", "bad_rate", "cum_bad_rate", "balance_sum")),
         **_labels(row, {"denominator": {"count", "balance"}})}
        for row in _rows(value.get("points")) if _months(row, ["cohort"])
    ]}


def _stress(value):
    if not isinstance(value, dict):
        return {}
    fields = ("sample_count", "ks", "dropped_feature_count", "ks_after", "ks_delta", "psi_vs_baseline")
    out = {"groups": []}
    for key, row in _items(value):
        if not isinstance(row, dict):
            continue
        item = {**_numeric(row, fields), **_labels(row, {"status": _STATUS})}
        if key == "baseline":
            out["baseline"] = item
        else:
            # Product/category labels may name a private business or supplier.
            # Only their measured risk effects are required for an overview.
            out["groups"].append(item)
    return out


def _low_pricing(value):
    if not isinstance(value, dict):
        return {}
    data = value.get("conclusion_data")
    summary = _numeric(data if isinstance(data, dict) else {}, (
        "threshold", "baseline_low_pricing_ratio", "baseline_ks", "max_psi", "min_ks", "max_ks_drop",
    ))
    rows = []
    by_ratio = value.get("by_ratio")
    for ratio, row in _items(by_ratio):
        if not isinstance(ratio, str) or not re.fullmatch(r"(?:0(?:\.\d{1,8})?|1(?:\.0{1,8})?)", ratio):
            continue
        if isinstance(row, dict):
            rows.append({"ratio": float(ratio), **_numeric(row, ("ks", "psi", "sample_count"))})
    return {"summary": summary, "scenarios": rows}


def modeling_report_prompt(summary):
    clean = {
        "sample_analysis": _samples(summary.get("sample_analysis")),
        "dataset_split": [
            {**_labels(row, {"split": _SPLITS}), **_months(row, ("window_start", "window_end")),
             **_numeric(row, ("sample_count", "bad_rate", "ks", "auc", "rmse", "mae", "r2"))}
            for row in _rows(summary.get("dataset_split")) if row.get("split") in _SPLITS
        ],
        "vintage": _vintage(summary.get("vintage")),
        "feature_importance": [
            {**_feature(row), **_numeric(row, ("importance", "importance_pct", "cumulative_importance_pct"))}
            for row in _rows(summary.get("feature_importance")) if _feature(row)
        ],
        "stress_product_removal": _stress(summary.get("stress_product_removal")),
        "stress_low_pricing": _low_pricing(summary.get("stress_low_pricing")),
    }
    metrics = {"psi_test_vs_train", "psi_oot_vs_train", "overfit_flag", "rmse_test_minus_train", "rmse_oot_minus_train"}
    clean["stability"] = [{"metric": row["metric"], "value": row["value"]}
        for row in _rows(summary.get("stability")) if row.get("metric") in metrics
        and (row.get("value") is None or type(row.get("value")) is bool or _finite(row.get("value")))]
    band_fields = ("bin", "score_lower", "score_upper", "sample_count", "split_sample_count", "scored_count",
        "unscored_count", "score_coverage", "labeled_count", "bad_count", "bad_rate", "avg_score",
        "cum_count_pct", "cum_bad_rate", "cum_bad_capture", "cum_reject_rate", "cum_pass_rate", "lift", "ks_contribution")
    clean["score_bands"] = [{**_numeric(row, band_fields), **_labels(row, {
        "split": _SPLITS, "bin_edges_source": _SPLITS, "cum_direction": {"higher_is_better", "higher_is_riskier"},
    })} for row in _rows(summary.get("score_bands"))]
    clean["oot_bin_table"] = [_numeric(row, (
        "bin_index", "score_lower", "score_upper", "sample_count", "bad_count", "bad_rate", "cum_sample_pct",
        "cum_bad_pct", "lift", "ks", "平均利率", "平均期数", "平均授信金额", "平均放款金额", "额度使用率",
        "金额逾期率", "累计金额逾期率", "金额lift",
    )) for row in _rows(summary.get("oot_bin_table"))]
    clean["univariate"] = [{**_feature(row), **_labels(row, {"split": _SPLITS}),
        **_numeric(row, ("iv", "ks", "auc", "sample_count", "coverage", "missing_rate", "unique_count", "psi_vs_train"))}
        for row in _rows(summary.get("univariate")) if _feature(row)]
    clean["calibration"] = [{**_numeric(row, ("sample_count", "positive_count", "brier_raw", "brier_calibrated",
        "ece_raw", "ece_calibrated", "bin", "prob_lower", "prob_upper", "avg_predicted_pd", "observed_bad_rate",
        "calibration_gap", "abs_gap")), **_labels(row, {
            "score_type": {"summary", "raw", "calibrated"}, "method": {"isotonic", "platt", "sigmoid"},
            "split": _SPLITS, "fit_split": _SPLITS, "eval_split": _SPLITS,
            "evaluated_on": {"train", "test", "oot", "fit_sample(in-sample)"},
        })} for row in _rows(summary.get("calibration"))]
    clean["section_status"] = [{**_labels(row, {"section": {
        "sample_analysis", "vintage", "feature_importance", "score_bands", "calibration", "univariate",
        "oot_bin_table", "stress_product_removal", "low_pricing", "feature_dictionary", "amount_bin", "product_list",
    }, "status": _STATUS}), **({"available": row["available"]} if type(row.get("available")) is bool else {})}
        for row in _rows(summary.get("section_status"))]
    return _mint_prompt({"structured_summary": clean, "instruction": (
        "Explain only these platform-computed aggregates. Missing evidence is unknown. "
        "Product names, report prose, individual records and model internals are withheld. "
        "Explain group effects without inventing group names. Return sample, vintage, model and stress text."
    )})
