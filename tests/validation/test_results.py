from dataclasses import replace
import json
from pathlib import Path

import pytest

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
    MAX_PMML_SCORING_ERROR_CHARS,
    MAX_PMML_SCORING_ERRORS,
    PmmlScoringResult,
    pmml_scoring_result_from_dict,
    pmml_scoring_result_to_dict,
    validate_pmml_scoring_result_fields,
    validation_results_from_dict,
    validation_results_to_dict,
)
from tests.output.test_excel import _make_pmml_results, _make_results


LEGACY_FIXTURE = Path("tests/fixtures/legacy_validation_results.json")


def _valid_pmml_scoring_result(**overrides):
    values = {
        "schema_version": "marvis.pmml_scoring.v1",
        "cache_key": "c" * 64,
        "pmml_sha256": "a" * 64,
        "sample_sha256": "b" * 64,
        "engine": "pypmml-pmml4s-batch",
        "engine_version": "1.5.5",
        "output_field": "probability_1",
        "input_row_count": 3,
        "success_count": 3,
        "failure_count": 0,
        "null_count": 0,
        "non_finite_count": 0,
        "elapsed_seconds": 0.1,
        "rows_per_second": 30.0,
        "chunk_size": 2,
        "required_input_count": 2,
        "missing_inputs": [],
        "score_artifact_path": "pmml_scores.parquet",
        "score_artifact_sha256": "d" * 64,
        "status": "pass",
        "bounded_errors": [],
    }
    values.update(overrides)
    return PmmlScoringResult(**values)


def test_new_results_round_trip_uses_pmml_scoring_not_reproducibility():
    results = _make_pmml_results()

    payload = validation_results_to_dict(results)

    assert payload["schema_version"] == "marvis.validation_results.v2"
    assert payload["lift_order"] == "good_to_bad"
    assert payload["pmml_scoring"]["status"] == "pass"
    assert "reproducibility" not in payload
    assert validation_results_from_dict(payload) == results


def test_legacy_v2_lift_fields_are_migrated_to_good_head_bad_tail():
    payload = validation_results_to_dict(_make_pmml_results())
    payload.pop("lift_order")
    for section in ("overall", "monthly_ks"):
        for row in payload["effectiveness"][section]:
            row["head_lift_5pct"], row["tail_lift_5pct"] = (
                row["tail_lift_5pct"],
                row["head_lift_5pct"],
            )
    bad_to_good_bins = [
        {
            "bin_index": 1,
            "score_lower": 0.0,
            "score_upper": 0.5,
            "sample_count": 5,
            "bad_count": 5,
            "bad_rate": 1.0,
            "cum_sample_pct": 0.5,
            "cum_bad_pct": 1.0,
            "lift": 2.0,
            "ks": 1.0,
        },
        {
            "bin_index": 2,
            "score_lower": 0.5,
            "score_upper": 1.0,
            "sample_count": 5,
            "bad_count": 0,
            "bad_rate": 0.0,
            "cum_sample_pct": 1.0,
            "cum_bad_pct": 1.0,
            "lift": 0.0,
            "ks": 0.0,
        },
    ]
    payload["effectiveness"]["bin_tables"]["oot"] = bad_to_good_bins
    payload["stress_test"]["baseline"]["bin_table"] = bad_to_good_bins
    payload["stress_test"]["per_category"][0]["bin_table"] = bad_to_good_bins
    legacy_head = payload["effectiveness"]["overall"][0]["head_lift_5pct"]

    restored = validation_results_from_dict(payload)
    current = validation_results_to_dict(restored)

    assert legacy_head == pytest.approx(2.0)
    assert payload["effectiveness"]["overall"][0]["head_lift_5pct"] == legacy_head
    assert restored.effectiveness.overall[0].head_lift_5pct == pytest.approx(0.2)
    assert restored.effectiveness.overall[0].tail_lift_5pct == pytest.approx(2.0)
    assert restored.effectiveness.monthly_ks[0].head_lift_5pct == pytest.approx(0.2)
    assert restored.effectiveness.monthly_ks[0].tail_lift_5pct == pytest.approx(2.0)
    for rows in [
        restored.effectiveness.bin_tables["oot"],
        restored.stress_test.baseline.bin_table,
        restored.stress_test.per_category[0].bin_table,
    ]:
        assert rows[0].lift == pytest.approx(0.0)
        assert rows[-1].lift == pytest.approx(2.0)
        assert rows[-1].cum_sample_pct == pytest.approx(1.0)
        assert rows[-1].cum_bad_pct == pytest.approx(1.0)
    assert current["lift_order"] == "good_to_bad"


def test_unmarked_v1_bad_head_is_migrated_but_good_head_is_preserved():
    bad_head_payload = validation_results_to_dict(_make_results())
    assert bad_head_payload["lift_order"] == "good_to_bad"
    bad_head_payload.pop("lift_order")
    for section in ("overall", "monthly_ks"):
        for row in bad_head_payload["effectiveness"][section]:
            row["head_lift_5pct"], row["tail_lift_5pct"] = (
                row["tail_lift_5pct"],
                row["head_lift_5pct"],
            )

    migrated = validation_results_from_dict(bad_head_payload)
    good_head_payload = json.loads(LEGACY_FIXTURE.read_text(encoding="utf-8"))
    preserved = validation_results_from_dict(good_head_payload)

    assert migrated.effectiveness.overall[0].head_lift_5pct == pytest.approx(0.2)
    assert migrated.effectiveness.overall[0].tail_lift_5pct == pytest.approx(2.0)
    assert preserved.effectiveness.overall[0].head_lift_5pct == pytest.approx(0.2)
    assert preserved.effectiveness.overall[0].tail_lift_5pct == pytest.approx(2.0)


def test_pmml_results_require_explicit_v2_schema_version():
    payload = validation_results_to_dict(_make_pmml_results())
    payload.pop("schema_version")

    with pytest.raises(ValueError, match="legacy payload must contain only"):
        validation_results_from_dict(payload)


def test_explicit_null_schema_version_is_rejected_not_treated_as_legacy():
    payload = validation_results_to_dict(_make_results())
    payload["schema_version"] = None

    with pytest.raises(ValueError, match="schema_version.*expected string"):
        validation_results_from_dict(payload)


def test_legacy_results_fixture_remains_readable_without_schema_version():
    payload = json.loads(LEGACY_FIXTURE.read_text(encoding="utf-8"))

    results = validation_results_from_dict(payload)

    assert results.schema_version == "marvis.validation_results.v1"
    assert results.reproducibility is not None
    assert results.pmml_scoring is None
    current = validation_results_to_dict(results)
    assert current["schema_version"] == "marvis.validation_results.v1"
    assert current["lift_order"] == "good_to_bad"


def test_legacy_missing_stress_status_preserves_empty_and_error_inference():
    empty_payload = validation_results_to_dict(_make_results())
    empty_payload.pop("schema_version")
    empty_payload["stress_test"].pop("status")
    empty_payload["stress_test"]["per_category"] = []

    empty = validation_results_from_dict(empty_payload)
    assert empty.stress_test.status == "skipped"

    failed_payload = validation_results_to_dict(_make_results())
    failed_payload.pop("schema_version")
    failed_payload["stress_test"].pop("status")
    category = failed_payload["stress_test"]["per_category"][0]
    category.pop("status")
    category["error"] = "legacy scoring failed"

    failed = validation_results_from_dict(failed_payload)
    assert failed.stress_test.status == "failed"
    assert failed.stress_test.per_category[0].status == "error"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update({"unexpected": "field"}),
        lambda payload: payload.update({"model_name": 123}),
        lambda payload: payload["basic_info"].pop("sample_period"),
        lambda payload: payload["basic_info"]["split_summary"][0].update(
            {"sample_count": "100"}
        ),
        lambda payload: payload["basic_info"]["split_summary"][0].update(
            {"bad_rate": float("nan")}
        ),
    ],
)
def test_validation_results_rejects_unknown_missing_and_wrongly_typed_fields(mutate):
    payload = validation_results_to_dict(_make_results())
    mutate(payload)

    with pytest.raises(ValueError, match="invalid validation results"):
        validation_results_from_dict(payload)


def test_validation_results_rejects_both_canonical_score_sections():
    payload = validation_results_to_dict(_make_results())
    payload["pmml_scoring"] = pmml_scoring_result_to_dict(
        _valid_pmml_scoring_result()
    )

    with pytest.raises(ValueError, match="v1 validation results cannot contain pmml_scoring"):
        validation_results_from_dict(payload)


def test_serializer_rejects_non_json_hyperparameter_values():
    results = _make_results()
    results.basic_info.hyperparameters["bad"] = object()

    with pytest.raises(ValueError, match="expected JSON value"):
        validation_results_to_dict(results)


def test_pmml_scoring_result_round_trip_is_exact_and_detached():
    result = _valid_pmml_scoring_result()

    payload = pmml_scoring_result_to_dict(result)
    restored = pmml_scoring_result_from_dict(payload)

    assert restored == result
    assert payload["missing_inputs"] is not result.missing_inputs
    assert payload["bounded_errors"] is not result.bounded_errors


@pytest.mark.parametrize("key", ["status", "bounded_errors"])
def test_pmml_scoring_result_from_dict_rejects_missing_fields(key):
    payload = pmml_scoring_result_to_dict(_valid_pmml_scoring_result())
    payload.pop(key)

    with pytest.raises(ValueError, match="missing"):
        pmml_scoring_result_from_dict(payload)


def test_pmml_scoring_result_from_dict_rejects_unknown_fields():
    payload = pmml_scoring_result_to_dict(_valid_pmml_scoring_result())
    payload["task_id"] = "orchestration-only"

    with pytest.raises(ValueError, match="unknown"):
        pmml_scoring_result_from_dict(payload)


def test_pmml_scoring_result_from_dict_rejects_non_mapping_payload():
    with pytest.raises(ValueError, match="must be an object"):
        pmml_scoring_result_from_dict([])  # type: ignore[arg-type]


def test_pmml_scoring_result_rejects_unknown_schema_version():
    result = _valid_pmml_scoring_result(schema_version="marvis.pmml_scoring.v2")

    with pytest.raises(ValueError, match="unsupported PMML scoring schema"):
        validate_pmml_scoring_result_fields(result)


@pytest.mark.parametrize(
    "field_name",
    [
        "input_row_count",
        "success_count",
        "failure_count",
        "null_count",
        "non_finite_count",
        "chunk_size",
        "required_input_count",
    ],
)
def test_pmml_scoring_result_rejects_bool_disguised_as_integer(field_name):
    result = replace(_valid_pmml_scoring_result(), **{field_name: True})

    with pytest.raises(ValueError, match="integer"):
        validate_pmml_scoring_result_fields(result)


@pytest.mark.parametrize("field_name", ["elapsed_seconds", "rows_per_second"])
@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -0.1])
def test_pmml_scoring_result_rejects_invalid_numeric_evidence(field_name, value):
    result = replace(_valid_pmml_scoring_result(), **{field_name: value})

    with pytest.raises(ValueError, match="PMML scoring number"):
        validate_pmml_scoring_result_fields(result)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"input_row_count": 0, "success_count": 0}, "must be positive"),
        ({"chunk_size": 0}, "must be positive"),
        ({"success_count": 2}, "do not add to input"),
        (
            {"success_count": 2, "failure_count": 1},
            "detail counts are inconsistent",
        ),
    ],
)
def test_pmml_scoring_result_rejects_invalid_count_invariants(overrides, message):
    result = replace(_valid_pmml_scoring_result(), **overrides)

    with pytest.raises(ValueError, match=message):
        validate_pmml_scoring_result_fields(result)


@pytest.mark.parametrize(
    "overrides",
    [
        {"success_count": 2, "failure_count": 1, "null_count": 1},
        {"missing_inputs": ["x1"]},
        {"bounded_errors": ["PMML output failed"]},
    ],
)
def test_pmml_scoring_result_rejects_passing_evidence_with_errors(overrides):
    result = replace(_valid_pmml_scoring_result(), **overrides)

    with pytest.raises(ValueError, match="passing PMML scoring evidence"):
        validate_pmml_scoring_result_fields(result)


def test_pmml_scoring_result_rejects_failed_status_without_failure_evidence():
    result = _valid_pmml_scoring_result(status="failed")

    with pytest.raises(ValueError, match="failed PMML scoring evidence is empty"):
        validate_pmml_scoring_result_fields(result)


@pytest.mark.parametrize(
    "overrides",
    [
        {"success_count": 2, "failure_count": 1, "null_count": 1},
        {"missing_inputs": ["x1"]},
        {"bounded_errors": ["PMML engine failed"]},
    ],
)
def test_pmml_scoring_result_accepts_failed_status_with_failure_evidence(overrides):
    result = replace(_valid_pmml_scoring_result(), status="failed", **overrides)

    assert validate_pmml_scoring_result_fields(result) is result


@pytest.mark.parametrize(
    "field_name",
    [
        "cache_key",
        "pmml_sha256",
        "sample_sha256",
        "score_artifact_sha256",
    ],
)
@pytest.mark.parametrize("value", ["a" * 63, "A" * 64, "g" * 64, 1])
def test_pmml_scoring_result_rejects_invalid_hashes(field_name, value):
    result = replace(_valid_pmml_scoring_result(), **{field_name: value})

    with pytest.raises(ValueError, match="SHA-256"):
        validate_pmml_scoring_result_fields(result)


@pytest.mark.parametrize(
    "field_name",
    ["engine", "engine_version", "output_field", "score_artifact_path"],
)
@pytest.mark.parametrize("value", ["", "   ", None])
def test_pmml_scoring_result_rejects_empty_identity_fields(field_name, value):
    result = replace(_valid_pmml_scoring_result(), **{field_name: value})

    with pytest.raises(ValueError, match="identity field"):
        validate_pmml_scoring_result_fields(result)


def test_pmml_scoring_result_rejects_too_many_bounded_errors():
    result = _valid_pmml_scoring_result(
        status="failed",
        bounded_errors=["error"] * (MAX_PMML_SCORING_ERRORS + 1),
    )

    with pytest.raises(ValueError, match="too many"):
        validate_pmml_scoring_result_fields(result)


def test_pmml_scoring_result_rejects_oversized_bounded_error():
    result = _valid_pmml_scoring_result(
        status="failed",
        bounded_errors=["x" * (MAX_PMML_SCORING_ERROR_CHARS + 1)],
    )

    with pytest.raises(ValueError, match="too long"):
        validate_pmml_scoring_result_fields(result)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("missing_inputs", ("x1",)),
        ("missing_inputs", [""]),
        ("bounded_errors", ("error",)),
        ("bounded_errors", [""]),
    ],
)
def test_pmml_scoring_result_rejects_invalid_string_lists(field_name, value):
    result = replace(
        _valid_pmml_scoring_result(),
        status="failed",
        **{field_name: value},
    )

    with pytest.raises(ValueError, match=field_name):
        validate_pmml_scoring_result_fields(result)


def test_pmml_scoring_result_to_dict_revalidates_direct_instances():
    result = _valid_pmml_scoring_result(rows_per_second=float("nan"))

    with pytest.raises(ValueError, match="PMML scoring number"):
        pmml_scoring_result_to_dict(result)


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
        schema_version="marvis.validation_results.v1",
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
    payload = validation_results_to_dict(_make_results())
    payload["stress_test"] = {
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
