# tests/test_metric_tables.py
from __future__ import annotations

import pytest

from riskmodel_checker.metric_tables import metric_table_sections_from_payload


@pytest.fixture
def payload() -> dict:
    return {
        "basic_info": {
            "split_summary": [
                {"split": "train", "sample_count": 80, "bad_count": 8, "bad_rate": 0.10,
                 "period_start": "20250101", "period_end": "20250228"},
                {"split": "test",  "sample_count": 20, "bad_count": 2, "bad_rate": 0.10,
                 "period_start": "20250101", "period_end": "20250228"},
                {"split": "oot",   "sample_count": 30, "bad_count": 3, "bad_rate": 0.10,
                 "period_start": "20250301", "period_end": "20250331"},
            ],
            "monthly_distribution": [
                {"month": "202501", "sample_count": 80, "bad_count": 8, "bad_rate": 0.10},
                {"month": "202503", "sample_count": 30, "bad_count": 3, "bad_rate": 0.10},
            ],
            "feature_importance": [
                {"rank": 1, "feature": "income",   "category": "征信", "importance": 0.42},
                {"rank": 2, "feature": "behavior", "category": "行为", "importance": 0.31},
            ],
        },
        "effectiveness": {
            "overall": [
                {"split": "train", "sample_count": 80, "bad_count": 8, "bad_rate": 0.10,
                 "ks": 0.3215, "auc": 0.7345, "head_lift_5pct": 2.1, "tail_lift_5pct": 0.4, "psi_vs_train": 0.0},
                {"split": "test",  "sample_count": 20, "bad_count": 2, "bad_rate": 0.10,
                 "ks": 0.3308, "auc": 0.7401, "head_lift_5pct": 2.0, "tail_lift_5pct": 0.5, "psi_vs_train": 0.0008},
                {"split": "oot",   "sample_count": 30, "bad_count": 3, "bad_rate": 0.10,
                 "ks": 0.3050, "auc": 0.7210, "head_lift_5pct": 1.9, "tail_lift_5pct": 0.4, "psi_vs_train": 0.0490},
            ],
            "monthly_ks": [
                {"month": "202501", "sample_count": 80, "bad_count": 8, "bad_rate": 0.10,
                 "ks": 0.3215, "auc": 0.7345, "head_lift_5pct": 2.1, "tail_lift_5pct": 0.4},
                {"month": "202503", "sample_count": 30, "bad_count": 3, "bad_rate": 0.10,
                 "ks": 0.3050, "auc": 0.7210, "head_lift_5pct": 1.9, "tail_lift_5pct": 0.4},
            ],
            "monthly_psi": [
                {"month": "202501", "psi_first_month": 0.0,   "psi_last_month": 0.012, "psi_mom": None},
                {"month": "202503", "psi_first_month": 0.012, "psi_last_month": 0.0,   "psi_mom": 0.008},
            ],
            "bin_tables": {
                "train": [
                    {"bin_index": 1, "score_lower": 0.0, "score_upper": 0.5,
                     "sample_count": 80, "bad_count": 8, "bad_rate": 0.10,
                     "cum_sample_pct": 1.0, "cum_bad_pct": 1.0, "lift": 1.0, "ks": 0.0},
                ],
                "test": [],
                "oot":  [],
            },
            "roc_ks_curves": {
                "train": {"fpr": [0.0, 0.5, 1.0], "tpr": [0.0, 0.7, 1.0],
                          "ks_curve": [0.0, 0.2, 0.0], "ks": 0.20, "population_at_ks": 0.5},
                "test":  {"fpr": [0.0, 0.5, 1.0], "tpr": [0.0, 0.6, 1.0],
                          "ks_curve": [0.0, 0.1, 0.0], "ks": 0.10, "population_at_ks": 0.5},
                "oot":   {"fpr": [0.0, 0.4, 1.0], "tpr": [0.0, 0.5, 1.0],
                          "ks_curve": [0.0, 0.1, 0.0], "ks": 0.10, "population_at_ks": 0.4},
            },
        },
        "stress_test": {
            "baseline": {"ks": 0.3215},
            "per_category": [
                {"category": "京东", "ks_after": 0.281, "ks_delta": -0.0404, "psi_vs_baseline": 0.1234},
            ],
        },
    }


def test_sections_basic_titles_preserved(payload):
    sections = metric_table_sections_from_payload(payload)
    titles = [section["title"] for section in sections]
    assert titles[:6] == [
        "样本情况",
        "整体效果&稳定性",
        "分月效果&稳定性",
        "分箱排序性",
        "特征重要性",
        "压力测试",
    ]


def test_each_section_has_expected_theme(payload):
    sections = metric_table_sections_from_payload(payload)
    actual = {section["title"]: section.get("section_theme") for section in sections}
    assert actual == {
        "样本情况": "cool-blue",
        "整体效果&稳定性": "warm-orange",
        "分月效果&稳定性": "deep-purple",
        "分箱排序性": "heatmap",
        "特征重要性": "cool-blue",
        "压力测试": "warning-red",
        "ROC&KS 曲线": "deep-purple",
    }


def test_each_table_has_expected_layout(payload):
    sections = metric_table_sections_from_payload(payload)
    by_key = {table["key"]: table.get("layout") for section in sections for table in section["tables"]}
    assert by_key == {
        "IMAGE:sample_overall_distribution": "table",
        "IMAGE:sample_month_distribution": "table",
        "IMAGE:overall_model_effect": "kpi_cards",
        "IMAGE:loan_month_effect": "trend_table",
        "IMAGE:ranking_table_train": "table",
        "IMAGE:ranking_table_test": "table",
        "IMAGE:ranking_table_oot": "table",
        "IMAGE:top20_feature_ranking": "table",
        "IMAGE:pressure_ks_table": "table",
        "ROC_KS_CURVES": "roc_ks_curve",
    }


def test_roc_ks_section_is_appended_with_curves(payload):
    sections = metric_table_sections_from_payload(payload)
    titles = [section["title"] for section in sections]
    assert titles[-1] == "ROC&KS 曲线"

    roc_section = sections[-1]
    assert roc_section["section_theme"] == "deep-purple"
    assert len(roc_section["tables"]) == 1

    curve_table = roc_section["tables"][0]
    assert curve_table["layout"] == "roc_ks_curve"
    assert curve_table["key"] == "ROC_KS_CURVES"
    assert set(curve_table["curves"].keys()) == {"train", "test", "oot"}

    train_curve = curve_table["curves"]["train"]
    assert train_curve["fpr"] == [0.0, 0.5, 1.0]
    assert train_curve["tpr"] == [0.0, 0.7, 1.0]
    assert train_curve["ks_curve"] == [0.0, 0.2, 0.0]
    assert train_curve["ks"] == pytest.approx(0.20)
    assert train_curve["population_at_ks"] == pytest.approx(0.5)


def test_roc_ks_section_omitted_when_no_curves(payload):
    payload["effectiveness"]["roc_ks_curves"] = {}
    sections = metric_table_sections_from_payload(payload)
    assert "ROC&KS 曲线" not in [section["title"] for section in sections]


def test_roc_ks_section_present_with_only_train_curve(payload):
    payload["effectiveness"]["roc_ks_curves"] = {
        "train": {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "ks_curve": [0.0, 0.0], "ks": 0.15, "population_at_ks": 0.4},
    }
    sections = metric_table_sections_from_payload(payload)
    assert sections[-1]["title"] == "ROC&KS 曲线"
    curves = sections[-1]["tables"][0]["curves"]
    assert set(curves.keys()) == {"train"}


def test_column_specs_match_headers_length_and_kind(payload):
    sections = metric_table_sections_from_payload(payload)
    by_key = {t["key"]: t for s in sections for t in s["tables"]}

    overall = by_key["IMAGE:overall_model_effect"]
    assert len(overall["column_specs"]) == len(overall["headers"])
    kinds_overall = [spec["kind"] for spec in overall["column_specs"]]
    assert kinds_overall == [
        "split-badge", "period", "databar", "percent-heat", "databar",
        "databar-primary", "databar", "databar", "databar", "psi",
    ]

    sample = by_key["IMAGE:sample_overall_distribution"]
    assert [spec["kind"] for spec in sample["column_specs"]] == [
        "split-badge", "period", "databar", "text", "databar", "percent-heat",
    ]

    bin_train = by_key["IMAGE:ranking_table_train"]
    assert [spec["kind"] for spec in bin_train["column_specs"]] == [
        "text", "databar", "text", "databar", "percent-heat",
        "percent-heat", "databar", "databar-primary", "text",
    ]

    psi_spec = overall["column_specs"][-1]
    assert psi_spec["kind"] == "psi"
    assert psi_spec["thresholds"] == [0.02, 0.10]
