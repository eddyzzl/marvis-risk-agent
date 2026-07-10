from marvis.validation.results import (
    ValidationResults,
    ReproducibilityResult,
    ConsistencySummary,
    BasicInfoResult,
    EffectivenessResult,
    StressTestResult,
    StressBaseline,
    ScoreCompareRow,
    SplitRow,
    MonthlyRow,
    FeatureImportanceRow,
    OverallRow,
    BinRow,
    MonthlyKsRow,
    MonthlyPsiRow,
    StressCategoryResult,
    ConsistencyStatus,
    validation_results_from_dict,
)


def test_consistency_status_values():
    assert {s.value for s in ConsistencyStatus} == {"pass", "review", "fail"}


def test_validation_results_assembles():
    repro = ReproducibilityResult(
        sample_size=2,
        seed=42,
        rows=[
            ScoreCompareRow(row_index=0, score_trained_pmml=0.1, score_input_pmml=0.1, score_sample_col=0.1),
            ScoreCompareRow(row_index=1, score_trained_pmml=0.2, score_input_pmml=0.2, score_sample_col=0.2),
        ],
        summary=ConsistencySummary(match_count=2, mismatch_count=0, max_abs_diff=0.0, status=ConsistencyStatus.PASS),
    )
    basic = BasicInfoResult(
        sample_period=("2025-03", "2025-08"),
        split_summary=[SplitRow(split="train", sample_count=100, bad_count=10, bad_rate=0.1)],
        monthly_distribution=[MonthlyRow(month="202503", sample_count=50, bad_count=5, bad_rate=0.1)],
        hyperparameters={"lr": 0.05},
        feature_importance=[FeatureImportanceRow(rank=1, feature="x1", importance=0.42)],
    )
    eff = EffectivenessResult(
        overall=[OverallRow(split="train", ks=0.3, psi_vs_train=0.0, sample_count=100, bad_rate=0.1)],
        bin_tables={
            "train": [BinRow(bin_index=1, score_lower=0.0, score_upper=0.1, sample_count=10,
                              bad_count=1, bad_rate=0.1, cum_sample_pct=0.1, cum_bad_pct=0.1,
                              lift=1.0, ks=0.0)]
        },
        monthly_ks=[MonthlyKsRow(month="202503", ks=0.25, sample_count=50)],
        monthly_psi=[MonthlyPsiRow(month="202503", psi_vs_train=0.05)],
    )
    stress = StressTestResult(
        baseline=StressBaseline(ks=0.3, sample_count=200, bin_table=[]),
        per_category=[
            StressCategoryResult(
                category="征信",
                dropped_features=["x1", "x2"],
                ks_after=0.28,
                ks_delta=-0.02,
                psi_vs_baseline=0.03,
                bin_table=[],
                error=None,
            )
        ],
    )
    results = ValidationResults(
        model_name="A卡",
        model_version="v1",
        algorithm="lgb",
        target_type="binary",
        reproducibility=repro,
        basic_info=basic,
        effectiveness=eff,
        stress_test=stress,
    )
    assert results.algorithm == "lgb"
    assert results.reproducibility.summary.status is ConsistencyStatus.PASS


def test_validation_results_from_dict_ignores_pipeline_task_identity():
    payload = {
        "task_id": "task-orchestration-only",
        "model_name": "A卡",
        "model_version": "v1",
        "algorithm": "lgb",
        "target_type": "binary",
        "reproducibility": {
            "sample_size": 1,
            "seed": 42,
            "rows": [
                {
                    "row_index": 0,
                    "score_code_model": 0.1,
                    "score_submitted_pmml": 0.1,
                    "abs_diff": 0.0,
                    "matched": True,
                }
            ],
            "summary": {
                "match_count": 1,
                "mismatch_count": 0,
                "max_abs_diff": 0.0,
                "status": "pass",
            },
        },
        "basic_info": {
            "sample_period": ["20250101", "20250131"],
            "split_summary": [],
            "monthly_distribution": [],
            "hyperparameters": {},
            "feature_importance": [],
        },
        "effectiveness": {
            "overall": [],
            "bin_tables": {},
            "monthly_ks": [],
            "monthly_psi": [],
        },
        "stress_test": {
            "baseline": {"ks": 0.0, "sample_count": 0, "bin_table": []},
            "per_category": [],
        },
    }

    results = validation_results_from_dict(payload)

    assert not hasattr(results, "task_id")
    assert results.model_name == "A卡"
    assert results.reproducibility.summary.status is ConsistencyStatus.PASS
    assert results.stress_test.unclassified_features == []
    assert results.stress_test.category_source_counts == {}


def test_validation_results_from_dict_preserves_stress_category_coverage():
    payload = {
        "stress_test": {
            "status": "partial",
            "baseline": {"ks": 0.3, "sample_count": 10, "bin_table": []},
            "per_category": [],
            "unclassified_features": ["BH_A044_C0580"],
            "category_source_counts": {
                "notebook": 2,
                "dictionary": 1,
                "unresolved": 1,
            },
        }
    }

    results = validation_results_from_dict(payload)

    assert results.stress_test.status == "partial"
    assert results.stress_test.unclassified_features == ["BH_A044_C0580"]
    assert results.stress_test.category_source_counts == {
        "notebook": 2,
        "dictionary": 1,
        "unresolved": 1,
    }


def test_validation_results_from_dict_preserves_feature_importance_category():
    payload = {
        "model_name": "A卡",
        "model_version": "v1",
        "algorithm": "lgb",
        "target_type": "binary",
        "reproducibility": {
            "sample_size": 0,
            "seed": 42,
            "rows": [],
            "summary": {
                "match_count": 0,
                "mismatch_count": 0,
                "max_abs_diff": 0.0,
                "status": "pass",
            },
        },
        "basic_info": {
            "sample_period": ["20250101", "20250131"],
            "split_summary": [],
            "monthly_distribution": [],
            "hyperparameters": {},
            "feature_importance": [
                {"rank": 1, "feature": "x1", "category": "征信", "importance": 0.8},
                {"rank": 2, "feature": "x2", "类别": "行为", "importance": 0.2},
            ],
        },
        "effectiveness": {
            "overall": [],
            "bin_tables": {},
            "monthly_ks": [],
            "monthly_psi": [],
        },
        "stress_test": {
            "baseline": {"ks": 0.0, "sample_count": 0, "bin_table": []},
            "per_category": [],
        },
    }

    results = validation_results_from_dict(payload)

    assert [row.category for row in results.basic_info.feature_importance] == ["征信", "行为"]


def test_validation_results_from_dict_preserves_monthly_psi_reference_month():
    payload = {
        "model_name": "A卡",
        "model_version": "v1",
        "algorithm": "lgb",
        "target_type": "binary",
        "reproducibility": {
            "sample_size": 0,
            "seed": 42,
            "rows": [],
            "summary": {
                "match_count": 0,
                "mismatch_count": 0,
                "max_abs_diff": 0.0,
                "status": "pass",
            },
        },
        "basic_info": {
            "sample_period": ["20250101", "20250131"],
            "split_summary": [],
            "monthly_distribution": [],
            "hyperparameters": {},
            "feature_importance": [],
        },
        "effectiveness": {
            "overall": [],
            "bin_tables": {},
            "monthly_ks": [],
            "monthly_psi": [
                {
                    "month": "202503",
                    "psi_vs_train": 0.01,
                    "psi_mom": 0.02,
                    "psi_mom_reference_month": "202501",
                    "psi_mom_has_calendar_gap": True,
                }
            ],
        },
        "stress_test": {
            "baseline": {"ks": 0.0, "sample_count": 0, "bin_table": []},
            "per_category": [],
        },
    }

    results = validation_results_from_dict(payload)

    row = results.effectiveness.monthly_psi[0]
    assert row.psi_mom_reference_month == "202501"
    assert row.psi_mom_has_calendar_gap is True
