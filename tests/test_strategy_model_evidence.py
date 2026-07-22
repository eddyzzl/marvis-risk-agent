from __future__ import annotations

from copy import deepcopy
import hashlib

import pytest

from marvis.packs.strategy.model_evidence import (
    METRIC_SCHEMA_TABLE,
    OBSERVATION_STATUSES,
    STRATEGY_MODEL_EVIDENCE_BUNDLE_SCHEMA_VERSION,
    StrategyModelEvidenceError,
    build_artifact_ref,
    build_evidence_source_ref,
    build_model_comparison_evidence,
    build_model_comparison_metric,
    build_model_evidence_ref,
    build_model_observation,
    build_model_selection,
    build_score_bin,
    build_single_model_evidence,
    build_strategy_model_evidence_bundle,
    build_univariate_bin_ref,
    build_univariate_evidence,
    build_univariate_observation,
    canonical_strategy_model_evidence_bundle_json,
    sample_partition_refs_from_strategy_sample_design_v2,
    strategy_model_evidence_bundle_from_json,
    validate_model_comparison_evidence,
    validate_model_observation,
    validate_single_model_evidence,
    validate_strategy_model_evidence_bundle,
    validate_univariate_evidence,
)
from tests.test_strategy_sample_design_v2 import (
    _bundle as build_sample_design_v2_fixture,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _artifact(label: str, *, kind: str) -> dict[str, str]:
    return build_artifact_ref(
        kind=kind,
        ref_id=label,
        content_hash=_hash(label),
    )


def _source(sample_design, label: str, population: str, partition: str):
    return build_evidence_source_ref(
        sample_design_bundle=sample_design,
        population=population,
        partition=partition,
        kind="tool_output",
        ref_id=label,
        content_hash=_hash(label),
    )


def _univariate_observation(
    sample_design,
    *,
    metric_key="ks",
    status="present",
    value=0.31,
    numerator=None,
    denominator=None,
    sample_count=2,
    unit="ratio",
    population="risk",
    partition="development",
    bin_id=None,
    period=None,
    reason=None,
    source_label="univariate-run",
):
    return build_univariate_observation(
        sample_design_bundle=sample_design,
        population=population,
        partition=partition,
        metric_key=metric_key,
        status=status,
        value=value,
        numerator=numerator,
        denominator=denominator,
        sample_count=sample_count,
        unit=unit,
        source_ref=_source(sample_design, source_label, population, partition),
        feature="age",
        bin_id=bin_id,
        period=period,
        reason=reason,
    )


def _univariate(sample_design):
    analysis_ref = _source(sample_design, "univariate-run", "risk", "development")
    bins = [
        build_univariate_bin_ref(
            sample_design_bundle=sample_design,
            population="risk",
            partition="development",
            ordinal=0,
            bin_id="age-low",
            kind="interval",
            lower_bound=None,
            upper_bound=30,
            lower_inclusive=False,
            upper_inclusive=False,
            definition_ref=analysis_ref,
        ),
        build_univariate_bin_ref(
            sample_design_bundle=sample_design,
            population="risk",
            partition="development",
            ordinal=1,
            bin_id="age-high",
            kind="interval",
            lower_bound=30,
            upper_bound=None,
            lower_inclusive=True,
            upper_inclusive=False,
            definition_ref=analysis_ref,
        ),
        build_univariate_bin_ref(
            sample_design_bundle=sample_design,
            population="risk",
            partition="development",
            ordinal=2,
            bin_id="missing",
            kind="missing",
            definition_ref=analysis_ref,
        ),
    ]
    observations = [
        _univariate_observation(
            sample_design,
            metric_key="bin_woe",
            value=-0.42,
            unit="number",
            bin_id="age-low",
        ),
        _univariate_observation(
            sample_design,
            metric_key="bin_iv",
            value=0.08,
            unit="number",
            bin_id="age-low",
        ),
        _univariate_observation(
            sample_design,
            metric_key="iv",
            value=0.17,
            unit="number",
        ),
        _univariate_observation(sample_design),
        _univariate_observation(
            sample_design,
            metric_key="lift",
            value=1.8,
            unit="multiple",
            bin_id="age-low",
        ),
        _univariate_observation(
            sample_design,
            metric_key="missing_rate",
            value=0.5,
            numerator=1,
            denominator=2,
            unit="ratio",
            bin_id=None,
        ),
        _univariate_observation(
            sample_design,
            metric_key="sentinel_rate",
            status="unavailable",
            value=None,
            numerator=None,
            denominator=None,
            sample_count=None,
            unit="ratio",
            reason="No sentinel definition was configured.",
        ),
        _univariate_observation(
            sample_design,
            metric_key="monthly_psi",
            value=0.06,
            unit="number",
            period="2026-06",
        ),
    ]
    return build_univariate_evidence(
        sample_design_bundle=sample_design,
        population="risk",
        partition="development",
        analysis_ref=analysis_ref,
        feature="age",
        bins=bins,
        missing_treatment="separate_bin",
        sentinel_treatment="not_configured",
        observations=observations,
    )


def _score_bins(sample_design, model_ref, score_ref, *, gap=False):
    source = _source(sample_design, "score-bins", "risk", "development")
    return [
        build_score_bin(
            sample_design_bundle=sample_design,
            ordinal=0,
            bin_id="score-low",
            lower_bound=None,
            upper_bound=0.5,
            lower_inclusive=False,
            upper_inclusive=False,
            definition_ref=source,
            model_ref=model_ref,
            score_ref=score_ref,
        ),
        build_score_bin(
            sample_design_bundle=sample_design,
            ordinal=1,
            bin_id="score-high",
            lower_bound=0.6 if gap else 0.5,
            upper_bound=None,
            lower_inclusive=True,
            upper_inclusive=False,
            definition_ref=source,
            model_ref=model_ref,
            score_ref=score_ref,
        ),
    ]


def _model_observation(
    sample_design,
    model_ref,
    score_ref,
    *,
    metric_key="auc",
    status="present",
    value=0.73,
    numerator=None,
    denominator=None,
    sample_count=2,
    unit="ratio",
    population="risk",
    partition="development",
    bin_id=None,
    period=None,
    reason=None,
    source_label=None,
):
    label = source_label or f"{model_ref['ref_id']}-{population}-{partition}-{metric_key}"
    return build_model_observation(
        sample_design_bundle=sample_design,
        population=population,
        partition=partition,
        metric_key=metric_key,
        status=status,
        value=value,
        numerator=numerator,
        denominator=denominator,
        sample_count=sample_count,
        unit=unit,
        source_ref=_source(sample_design, label, population, partition),
        model_ref=model_ref,
        score_ref=score_ref,
        bin_id=bin_id,
        period=period,
        reason=reason,
    )


def _model(sample_design, label: str, *, auc=0.73, validation_ks=0.38):
    model_ref = _artifact(f"{label}-model", kind="model_artifact")
    score_ref = _artifact(f"{label}-score", kind="score_artifact")
    observations = [
        _model_observation(
            sample_design,
            model_ref,
            score_ref,
            value=auc,
        ),
        _model_observation(
            sample_design,
            model_ref,
            score_ref,
            metric_key="ks",
            value=validation_ks,
            population="risk",
            partition="validation",
        ),
        _model_observation(
            sample_design,
            model_ref,
            score_ref,
            metric_key="lift_head_10",
            status="unavailable",
            value=None,
            numerator=None,
            denominator=None,
            sample_count=None,
            unit="multiple",
            population="risk",
            partition="oot",
            reason="OOT lift was not produced by the scoring artifact.",
        ),
        _model_observation(
            sample_design,
            model_ref,
            score_ref,
            metric_key="calibration_observed_rate",
            value=0.5,
            numerator=1,
            denominator=2,
            population="risk",
            partition="validation",
            bin_id="score-low",
        ),
        _model_observation(
            sample_design,
            model_ref,
            score_ref,
            metric_key="score_bin_share",
            value=0.5,
            numerator=1,
            denominator=2,
            population="approval",
            partition="development",
            bin_id="score-low",
        ),
        _model_observation(
            sample_design,
            model_ref,
            score_ref,
            metric_key="monthly_psi",
            value=0.08,
            unit="number",
            population="approval",
            partition="oot",
            period="2026-06",
        ),
    ]
    return build_single_model_evidence(
        sample_design_bundle=sample_design,
        training_source_ref=_source(
            sample_design, f"{label}-training", "risk", "development"
        ),
        model_ref=model_ref,
        score_ref=score_ref,
        features=["income", "age"],
        score_bins=_score_bins(sample_design, model_ref, score_ref),
        observations=observations,
    )


def _comparison(sample_design, first_model, second_model):
    first_ref = build_model_evidence_ref(
        first_model, sample_design_bundle=sample_design
    )
    second_ref = build_model_evidence_ref(
        second_model, sample_design_bundle=sample_design
    )
    metric = build_model_comparison_metric(
        sample_design_bundle=sample_design,
        population="risk",
        partition="validation",
        metric_key="ks",
        status="present",
        unit="ratio",
        source_ref=_source(
            sample_design, "model-comparison-ks", "risk", "validation"
        ),
        model_values=[
            {"model_evidence_ref": first_ref, "value": 0.38},
            {"model_evidence_ref": second_ref, "value": 0.35},
        ],
        delta=0.03,
    )
    return build_model_comparison_evidence(
        sample_design_bundle=sample_design,
        population="risk",
        partition="validation",
        comparison_ref=_artifact("comparison-run", kind="model_comparison"),
        model_evidence_refs=[first_ref, second_ref],
        metrics=[metric],
        selection=build_model_selection(
            status="selected",
            selected_model_evidence_ref=first_ref,
            metric_key="ks",
            direction="higher_is_better",
        ),
    )


def _bundle():
    sample_design = build_sample_design_v2_fixture()
    univariate = _univariate(sample_design)
    first_model = _model(sample_design, "first", validation_ks=0.38)
    second_model = _model(
        sample_design,
        "second",
        auc=0.71,
        validation_ks=0.35,
    )
    comparison = _comparison(sample_design, first_model, second_model)
    evidence = build_strategy_model_evidence_bundle(
        sample_design_bundle=sample_design,
        univariate_evidence=[univariate],
        model_evidence=[first_model, second_model],
        comparison_evidence=[comparison],
    )
    return sample_design, evidence


def test_bundle_uses_real_sample_design_and_derives_all_six_exact_refs():
    sample_design, evidence = _bundle()
    design = sample_design["sample_design"]
    canonical = canonical_strategy_model_evidence_bundle_json(
        evidence, sample_design_bundle=sample_design
    )

    assert evidence["schema_version"] == (
        STRATEGY_MODEL_EVIDENCE_BUNDLE_SCHEMA_VERSION
    )
    assert evidence["sample_refs"] == (
        sample_partition_refs_from_strategy_sample_design_v2(sample_design)
    )
    assert len(evidence["sample_refs"]) == 6
    assert {
        (item["population"], item["partition"])
        for item in evidence["sample_refs"]
    } == {
        (population, partition)
        for population in ("approval", "risk")
        for partition in ("development", "validation", "oot")
    }
    assert all(
        item["sample_design_ref"]["sample_design_id"]
        == design["sample_design_id"]
        and item["dataset_ref"] == design["identity"]["dataset_ref"]
        and item["workspace_ref"] == design["identity"]["workspace_ref"]
        for item in evidence["sample_refs"]
    )
    assert strategy_model_evidence_bundle_from_json(
        canonical, sample_design_bundle=sample_design
    ) == evidence
    assert validate_strategy_model_evidence_bundle(
        evidence, sample_design_bundle=sample_design
    ) == evidence


def test_source_refs_carry_exact_sample_design_membership_dataset_and_workspace():
    sample_design = build_sample_design_v2_fixture()
    source = _source(sample_design, "metrics", "approval", "oot")
    expected = next(
        item
        for item in sample_partition_refs_from_strategy_sample_design_v2(
            sample_design
        )
        if (item["population"], item["partition"]) == ("approval", "oot")
    )

    assert {key: source[key] for key in expected} == expected


def test_validator_requires_the_exact_live_sample_design_bundle():
    sample_design, evidence = _bundle()
    other_design = build_sample_design_v2_fixture(maturity_status="not_matured")

    with pytest.raises(StrategyModelEvidenceError, match="sample_design_binding"):
        validate_strategy_model_evidence_bundle(
            evidence,
            sample_design_bundle=other_design,
        )
    tampered_design = deepcopy(sample_design)
    tampered_design["sample_design"]["identity"]["workspace_ref"]["revision"] += 1
    with pytest.raises(StrategyModelEvidenceError, match="strict StrategySampleDesign"):
        validate_strategy_model_evidence_bundle(
            evidence,
            sample_design_bundle=tampered_design,
        )


def test_public_individual_validators_preserve_exact_contracts():
    sample_design, evidence = _bundle()
    univariate = evidence["univariate_evidence"][0]
    model = evidence["model_evidence"][0]
    comparison = evidence["comparison_evidence"][0]

    assert validate_univariate_evidence(
        univariate, sample_design_bundle=sample_design
    ) == univariate
    assert validate_single_model_evidence(
        model, sample_design_bundle=sample_design
    ) == model
    assert validate_model_observation(
        model["observations"][0], sample_design_bundle=sample_design
    ) == model["observations"][0]
    assert validate_model_comparison_evidence(
        comparison, sample_design_bundle=sample_design
    ) == comparison


def test_metric_schema_table_is_typed_for_units_ranges_dimensions_and_maturity():
    auc = METRIC_SCHEMA_TABLE["model"]["auc"]
    psi = METRIC_SCHEMA_TABLE["model"]["monthly_psi"]
    count = METRIC_SCHEMA_TABLE["model"]["score_bin_count"]

    assert auc["units"] == ("ratio",)
    assert (auc["minimum"], auc["maximum"]) == (0, 1)
    assert auc["populations"] == ("risk",)
    assert auc["maturity_sensitive"] is True
    assert auc["requires_binary_classes"] is True
    assert count["requires_binary_classes"] is False
    assert psi["period"] == "required"
    assert psi["direction"] == "lower_is_better"
    assert count["integer"] is True
    assert count["bin_id"] == "required"
    assert METRIC_SCHEMA_TABLE["model"]["calibration_gap"]["minimum"] == -1
    assert "calibration_abs_gap" not in METRIC_SCHEMA_TABLE["comparison"]
    assert OBSERVATION_STATUSES == frozenset(
        {"present", "unavailable", "not_matured", "not_applicable"}
    )


@pytest.mark.parametrize(
    ("metric_key", "value", "unit", "population", "bin_id", "period", "match"),
    [
        ("auc", 1.01, "ratio", "risk", None, None, "maximum"),
        ("ks", -0.01, "ratio", "risk", None, None, "minimum"),
        ("lift_head_10", -1, "multiple", "risk", None, None, "minimum"),
        ("calibration_gap", 1.1, "ratio", "risk", "score-low", None, "maximum"),
        ("score_bin_share", 1.1, "ratio", "approval", "score-low", None, "maximum"),
        ("score_psi", -0.1, "number", "approval", None, None, "minimum"),
        ("score_bin_count", 1.5, "count", "approval", "score-low", None, "integer"),
        ("auc", 0.7, "number", "risk", None, None, "unit"),
        ("monthly_psi", 0.1, "number", "approval", None, "2026-13", "YYYY-MM"),
        ("monthly_psi", 0.1, "number", "approval", None, None, "period"),
        ("auc", 0.7, "ratio", "risk", "score-low", None, "forbidden"),
        ("score_bin_share", 0.5, "ratio", "approval", None, None, "bin_id"),
    ],
)
def test_model_metric_schema_rejects_invalid_numbers_units_and_dimensions(
    metric_key, value, unit, population, bin_id, period, match
):
    sample_design = build_sample_design_v2_fixture()
    model_ref = _artifact("model", kind="model_artifact")
    score_ref = _artifact("score", kind="score_artifact")

    with pytest.raises(StrategyModelEvidenceError, match=match):
        _model_observation(
            sample_design,
            model_ref,
            score_ref,
            metric_key=metric_key,
            value=value,
            unit=unit,
            population=population,
            bin_id=bin_id,
            period=period,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True])
def test_observations_reject_nonfinite_numbers_and_boolean(value):
    sample_design = build_sample_design_v2_fixture()
    model_ref = _artifact("model", kind="model_artifact")
    score_ref = _artifact("score", kind="score_artifact")
    with pytest.raises(StrategyModelEvidenceError, match="finite number"):
        _model_observation(
            sample_design,
            model_ref,
            score_ref,
            value=value,
        )


@pytest.mark.parametrize(
    "status", ["unavailable", "not_matured", "not_applicable"]
)
def test_non_present_status_requires_null_value_operands_and_reason(status):
    sample_design = build_sample_design_v2_fixture(
        maturity_status="not_matured" if status == "not_matured" else "confirmed_matured"
    )
    model_ref = _artifact("model", kind="model_artifact")
    score_ref = _artifact("score", kind="score_artifact")
    kwargs = {
        "metric_key": "auc",
        "status": status,
        "value": None,
        "numerator": None,
        "denominator": None,
        "sample_count": None,
        "unit": "ratio",
        "population": "approval" if status == "not_applicable" else "risk",
        "partition": "oot" if status == "not_matured" else "development",
        "reason": "The observation is intentionally absent.",
    }
    observation = _model_observation(
        sample_design,
        model_ref,
        score_ref,
        **kwargs,
    )
    assert all(observation[field] is None for field in ("value", "numerator", "denominator", "sample_count"))

    with pytest.raises(StrategyModelEvidenceError, match="must be null"):
        _model_observation(
            sample_design,
            model_ref,
            score_ref,
            **{**kwargs, "value": 0},
        )
    with pytest.raises(StrategyModelEvidenceError, match="reason"):
        _model_observation(
            sample_design,
            model_ref,
            score_ref,
            **{**kwargs, "reason": None},
        )


def test_not_matured_is_limited_to_outcome_metrics_and_matches_sample_truth():
    sample_design = build_sample_design_v2_fixture(maturity_status="not_matured")
    model_ref = _artifact("model", kind="model_artifact")
    score_ref = _artifact("score", kind="score_artifact")
    common = {
        "status": "not_matured",
        "value": None,
        "numerator": None,
        "denominator": None,
        "sample_count": None,
        "reason": "Not matured.",
    }
    with pytest.raises(StrategyModelEvidenceError, match="not_matured"):
        _model_observation(
            sample_design,
            model_ref,
            score_ref,
            metric_key="score_psi",
            unit="number",
            population="approval",
            partition="oot",
            **common,
        )

    matured_design = build_sample_design_v2_fixture()
    with pytest.raises(StrategyModelEvidenceError, match="not_matured"):
        _model_observation(
            matured_design,
            model_ref,
            score_ref,
            metric_key="auc",
            unit="ratio",
            population="risk",
            partition="oot",
            **common,
        )
    development = _model_observation(
        sample_design,
        model_ref,
        score_ref,
        metric_key="auc",
        unit="ratio",
        population="risk",
        partition="development",
        **common,
    )
    assert development["status"] == "not_matured"


@pytest.mark.parametrize("maturity_status", ["not_matured", "unknown", "unavailable"])
def test_present_outcome_metrics_require_confirmed_maturity(maturity_status):
    sample_design = build_sample_design_v2_fixture(
        maturity_status=maturity_status
    )
    model_ref = _artifact("model", kind="model_artifact")
    score_ref = _artifact("score", kind="score_artifact")

    with pytest.raises(StrategyModelEvidenceError, match="confirmed_matured"):
        _model_observation(
            sample_design,
            model_ref,
            score_ref,
            metric_key="auc",
            value=0.7,
            unit="ratio",
            population="risk",
            partition="oot",
        )


def test_observation_sample_count_cannot_exceed_bound_partition_rows():
    sample_design = build_sample_design_v2_fixture()
    model_ref = _artifact("model", kind="model_artifact")
    score_ref = _artifact("score", kind="score_artifact")
    with pytest.raises(StrategyModelEvidenceError, match="sample_count exceeds"):
        _model_observation(
            sample_design,
            model_ref,
            score_ref,
            metric_key="score_bin_share",
            value=0.5,
            numerator=1,
            denominator=2,
            sample_count=3,
            population="approval",
            partition="oot",
        bin_id="score-low",
    )


def test_present_metric_rejects_empty_partition_and_single_class_discrimination():
    empty_oot = build_sample_design_v2_fixture(empty_oot=True)
    model_ref = _artifact("model", kind="model_artifact")
    score_ref = _artifact("score", kind="score_artifact")
    with pytest.raises(StrategyModelEvidenceError, match="non-empty"):
        _model_observation(
            empty_oot,
            model_ref,
            score_ref,
            metric_key="score_psi",
            value=0.1,
            unit="number",
            population="approval",
            partition="oot",
        )

    single_class = build_sample_design_v2_fixture(
        single_class_validation=True
    )
    with pytest.raises(StrategyModelEvidenceError, match="both good and bad"):
        _model_observation(
            single_class,
            model_ref,
            score_ref,
            metric_key="auc",
            value=0.7,
            unit="ratio",
            population="risk",
            partition="validation",
        )


def test_count_metric_cannot_exceed_evaluated_or_bound_sample_count():
    sample_design = build_sample_design_v2_fixture()
    model_ref = _artifact("model", kind="model_artifact")
    score_ref = _artifact("score", kind="score_artifact")
    with pytest.raises(StrategyModelEvidenceError, match="count metric value exceeds"):
        _model_observation(
            sample_design,
            model_ref,
            score_ref,
            metric_key="score_bin_count",
            value=3,
            sample_count=2,
            unit="count",
            population="approval",
            partition="oot",
            bin_id="score-low",
        )


def test_outcome_metrics_reject_approval_but_score_distribution_accepts_all_populations():
    sample_design = build_sample_design_v2_fixture()
    model_ref = _artifact("model", kind="model_artifact")
    score_ref = _artifact("score", kind="score_artifact")
    with pytest.raises(StrategyModelEvidenceError, match="not applicable"):
        _model_observation(
            sample_design,
            model_ref,
            score_ref,
            metric_key="auc",
            population="approval",
        )
    observation = _model_observation(
        sample_design,
        model_ref,
        score_ref,
        metric_key="score_bin_share",
        value=0.5,
        numerator=1,
        denominator=2,
        population="approval",
        partition="oot",
        bin_id="score-low",
    )
    assert (observation["sample_ref"]["population"], observation["sample_ref"]["partition"]) == ("approval", "oot")


def test_source_partition_cannot_masquerade_as_oot_or_cross_workspace():
    sample_design = build_sample_design_v2_fixture()
    model_ref = _artifact("model", kind="model_artifact")
    score_ref = _artifact("score", kind="score_artifact")
    development_source = _source(
        sample_design, "development-metrics", "risk", "development"
    )
    with pytest.raises(StrategyModelEvidenceError, match="does not match exact sample"):
        build_model_observation(
            sample_design_bundle=sample_design,
            population="risk",
            partition="oot",
            metric_key="auc",
            status="present",
            value=0.7,
            numerator=None,
            denominator=None,
            sample_count=1,
            unit="ratio",
            source_ref=development_source,
            model_ref=model_ref,
            score_ref=score_ref,
        )

    forged_workspace = deepcopy(
        _source(sample_design, "oot-metrics", "risk", "oot")
    )
    forged_workspace["workspace_ref"]["revision"] += 1
    with pytest.raises(StrategyModelEvidenceError, match="not derived"):
        build_model_observation(
            sample_design_bundle=sample_design,
            population="risk",
            partition="oot",
            metric_key="auc",
            status="present",
            value=0.7,
            numerator=None,
            denominator=None,
            sample_count=1,
            unit="ratio",
            source_ref=forged_workspace,
            model_ref=model_ref,
            score_ref=score_ref,
        )


def test_model_observation_cannot_be_reused_for_another_model_or_score():
    sample_design = build_sample_design_v2_fixture()
    first = _model(sample_design, "first")
    second_model_ref = _artifact("second-model", kind="model_artifact")
    second_score_ref = _artifact("second-score", kind="score_artifact")

    with pytest.raises(StrategyModelEvidenceError, match="cannot be reused"):
        build_single_model_evidence(
            sample_design_bundle=sample_design,
            training_source_ref=_source(
                sample_design, "second-training", "risk", "development"
            ),
            model_ref=second_model_ref,
            score_ref=second_score_ref,
            features=["age"],
            score_bins=_score_bins(
                sample_design,
                second_model_ref,
                second_score_ref,
            ),
            observations=first["observations"],
        )


def test_model_training_is_fixed_to_risk_development():
    sample_design = build_sample_design_v2_fixture()
    model = deepcopy(_model(sample_design, "first"))
    approval_development = next(
        item
        for item in sample_partition_refs_from_strategy_sample_design_v2(
            sample_design
        )
        if (item["population"], item["partition"])
        == ("approval", "development")
    )
    model["training_sample_ref"] = approval_development

    with pytest.raises(StrategyModelEvidenceError, match="risk/development|content"):
        validate_single_model_evidence(model, sample_design_bundle=sample_design)


@pytest.mark.parametrize(
    "maturity_status", ["not_matured", "unknown", "unavailable"]
)
def test_model_training_requires_confirmed_matured_risk_sample(maturity_status):
    sample_design = build_sample_design_v2_fixture(
        maturity_status=maturity_status
    )
    model_ref = _artifact("model", kind="model_artifact")
    score_ref = _artifact("score", kind="score_artifact")

    with pytest.raises(StrategyModelEvidenceError, match="confirmed_matured"):
        build_single_model_evidence(
            sample_design_bundle=sample_design,
            training_source_ref=_source(
                sample_design, "training", "risk", "development"
            ),
            model_ref=model_ref,
            score_ref=score_ref,
            features=["age"],
            score_bins=_score_bins(sample_design, model_ref, score_ref),
            observations=[],
        )


@pytest.mark.parametrize(
    "fixture_options",
    [
        {"empty_development": True},
        {"single_class_development": True},
    ],
)
def test_model_training_requires_both_good_and_bad_development_rows(
    fixture_options,
):
    sample_design = build_sample_design_v2_fixture(**fixture_options)
    model_ref = _artifact("model", kind="model_artifact")
    score_ref = _artifact("score", kind="score_artifact")

    with pytest.raises(StrategyModelEvidenceError, match="good and bad"):
        build_single_model_evidence(
            sample_design_bundle=sample_design,
            training_source_ref=_source(
                sample_design, "training", "risk", "development"
            ),
            model_ref=model_ref,
            score_ref=score_ref,
            features=["age"],
            score_bins=_score_bins(sample_design, model_ref, score_ref),
            observations=[],
        )


def test_applicable_metric_cannot_be_hidden_as_not_applicable():
    sample_design = build_sample_design_v2_fixture()
    model_ref = _artifact("model", kind="model_artifact")
    score_ref = _artifact("score", kind="score_artifact")

    with pytest.raises(StrategyModelEvidenceError, match="use unavailable"):
        _model_observation(
            sample_design,
            model_ref,
            score_ref,
            metric_key="auc",
            status="not_applicable",
            value=None,
            numerator=None,
            denominator=None,
            sample_count=None,
            unit="ratio",
            population="risk",
            partition="development",
            reason="Pretend the applicable metric does not apply.",
        )


def test_score_bins_require_explicit_contiguous_canonical_boundaries_and_model_refs():
    sample_design = build_sample_design_v2_fixture()
    model_ref = _artifact("model", kind="model_artifact")
    score_ref = _artifact("score", kind="score_artifact")
    with pytest.raises(StrategyModelEvidenceError, match="ordered and contiguous"):
        build_single_model_evidence(
            sample_design_bundle=sample_design,
            training_source_ref=_source(
                sample_design, "training", "risk", "development"
            ),
            model_ref=model_ref,
            score_ref=score_ref,
            features=["age"],
            score_bins=_score_bins(
                sample_design,
                model_ref,
                score_ref,
                gap=True,
            ),
            observations=[],
        )

    wrong_model_bins = _score_bins(sample_design, model_ref, score_ref)
    wrong_model_bins[0]["model_ref"] = _artifact(
        "other-model", kind="model_artifact"
    )
    with pytest.raises(StrategyModelEvidenceError, match="model/score refs|content"):
        build_single_model_evidence(
            sample_design_bundle=sample_design,
            training_source_ref=_source(
                sample_design, "training", "risk", "development"
            ),
            model_ref=model_ref,
            score_ref=score_ref,
            features=["age"],
            score_bins=wrong_model_bins,
            observations=[],
        )


def test_univariate_bins_require_explicit_unique_consecutive_ordinals():
    sample_design = build_sample_design_v2_fixture()
    evidence = _univariate(sample_design)
    bins = deepcopy(evidence["bins"])
    bins[1]["ordinal"] = 2

    with pytest.raises(StrategyModelEvidenceError, match="ordinals"):
        build_univariate_evidence(
            sample_design_bundle=sample_design,
            population="risk",
            partition="development",
            analysis_ref=evidence["analysis_ref"],
            feature="age",
            bins=bins,
            missing_treatment="separate_bin",
            sentinel_treatment="not_configured",
            observations=evidence["observations"],
        )


def test_duplicate_semantic_observations_are_rejected_even_with_different_values():
    sample_design = build_sample_design_v2_fixture()
    first = _univariate_observation(sample_design, value=0.31)
    second = _univariate_observation(
        sample_design,
        value=0.32,
        source_label="other-run",
    )
    evidence = _univariate(sample_design)

    with pytest.raises(StrategyModelEvidenceError, match="duplicates"):
        build_univariate_evidence(
            sample_design_bundle=sample_design,
            population="risk",
            partition="development",
            analysis_ref=evidence["analysis_ref"],
            feature="age",
            bins=evidence["bins"],
            missing_treatment="separate_bin",
            sentinel_treatment="not_configured",
            observations=[first, second],
        )


def test_comparison_requires_typed_complete_values_delta_and_evaluation_sample():
    sample_design = build_sample_design_v2_fixture()
    first = _model(sample_design, "first")
    second = _model(sample_design, "second")
    refs = [
        build_model_evidence_ref(first, sample_design_bundle=sample_design),
        build_model_evidence_ref(second, sample_design_bundle=sample_design),
    ]
    source = _source(sample_design, "comparison", "risk", "validation")

    with pytest.raises(StrategyModelEvidenceError, match="max minus min"):
        build_model_comparison_metric(
            sample_design_bundle=sample_design,
            population="risk",
            partition="validation",
            metric_key="ks",
            status="present",
            unit="ratio",
            source_ref=source,
            model_values=[
                {"model_evidence_ref": refs[0], "value": 0.38},
                {"model_evidence_ref": refs[1], "value": 0.35},
            ],
            delta=0.02,
        )

    incomplete = build_model_comparison_metric(
        sample_design_bundle=sample_design,
        population="risk",
        partition="validation",
        metric_key="ks",
        status="present",
        unit="ratio",
        source_ref=source,
        model_values=[
            {"model_evidence_ref": refs[0], "value": 0.38},
            {
                "model_evidence_ref": {
                    "evidence_id": "strategy-model-evidence-" + "0" * 24,
                    "content_hash": "0" * 64,
                },
                "value": 0.35,
            },
        ],
        delta=0.03,
    )
    with pytest.raises(StrategyModelEvidenceError, match="cover every compared model"):
        build_model_comparison_evidence(
            sample_design_bundle=sample_design,
            population="risk",
            partition="validation",
            comparison_ref=_artifact("comparison", kind="model_comparison"),
            model_evidence_refs=refs,
            metrics=[incomplete],
            selection=build_model_selection(
                status="no_selection",
                reason="Evidence is incomplete.",
            ),
        )


def test_bundle_reconciles_comparison_values_to_exact_model_observations():
    sample_design = build_sample_design_v2_fixture()
    first = _model(sample_design, "first", validation_ks=0.38)
    second = _model(sample_design, "second", validation_ks=0.35)
    first_ref = build_model_evidence_ref(
        first, sample_design_bundle=sample_design
    )
    second_ref = build_model_evidence_ref(
        second, sample_design_bundle=sample_design
    )
    forged_metric = build_model_comparison_metric(
        sample_design_bundle=sample_design,
        population="risk",
        partition="validation",
        metric_key="ks",
        status="present",
        unit="ratio",
        source_ref=_source(
            sample_design, "forged-comparison", "risk", "validation"
        ),
        model_values=[
            {"model_evidence_ref": first_ref, "value": 0.10},
            {"model_evidence_ref": second_ref, "value": 0.90},
        ],
        delta=0.80,
    )
    forged_comparison = build_model_comparison_evidence(
        sample_design_bundle=sample_design,
        population="risk",
        partition="validation",
        comparison_ref=_artifact(
            "forged-comparison", kind="model_comparison"
        ),
        model_evidence_refs=[first_ref, second_ref],
        metrics=[forged_metric],
        selection=build_model_selection(
            status="selected",
            selected_model_evidence_ref=second_ref,
            metric_key="ks",
            direction="higher_is_better",
        ),
    )

    with pytest.raises(
        StrategyModelEvidenceError, match="does not match.*model observation"
    ):
        build_strategy_model_evidence_bundle(
            sample_design_bundle=sample_design,
            model_evidence=[first, second],
            comparison_evidence=[forged_comparison],
        )

def test_comparison_selection_must_follow_typed_metric_direction_and_best_value():
    sample_design = build_sample_design_v2_fixture()
    first = _model(sample_design, "first", validation_ks=0.38)
    second = _model(sample_design, "second", validation_ks=0.35)
    first_ref = build_model_evidence_ref(first, sample_design_bundle=sample_design)
    second_ref = build_model_evidence_ref(second, sample_design_bundle=sample_design)
    metric = build_model_comparison_metric(
        sample_design_bundle=sample_design,
        population="risk",
        partition="validation",
        metric_key="ks",
        status="present",
        unit="ratio",
        source_ref=_source(sample_design, "comparison", "risk", "validation"),
        model_values=[
            {"model_evidence_ref": first_ref, "value": 0.38},
            {"model_evidence_ref": second_ref, "value": 0.35},
        ],
        delta=0.03,
    )
    with pytest.raises(StrategyModelEvidenceError, match="not best"):
        build_model_comparison_evidence(
            sample_design_bundle=sample_design,
            population="risk",
            partition="validation",
            comparison_ref=_artifact("comparison", kind="model_comparison"),
            model_evidence_refs=[first_ref, second_ref],
            metrics=[metric],
            selection=build_model_selection(
                status="selected",
                selected_model_evidence_ref=second_ref,
                metric_key="ks",
                direction="higher_is_better",
            ),
        )
    with pytest.raises(StrategyModelEvidenceError, match="direction"):
        build_model_comparison_evidence(
            sample_design_bundle=sample_design,
            population="risk",
            partition="validation",
            comparison_ref=_artifact("comparison", kind="model_comparison"),
            model_evidence_refs=[first_ref, second_ref],
            metrics=[metric],
            selection=build_model_selection(
                status="selected",
                selected_model_evidence_ref=first_ref,
                metric_key="ks",
                direction="lower_is_better",
            ),
        )


def test_non_present_comparison_metric_requires_null_values_delta_and_reason():
    sample_design = build_sample_design_v2_fixture(maturity_status="not_matured")
    metric = build_model_comparison_metric(
        sample_design_bundle=sample_design,
        population="risk",
        partition="oot",
        metric_key="auc",
        status="not_matured",
        unit="ratio",
        source_ref=_source(sample_design, "maturity", "risk", "oot"),
        model_values=None,
        delta=None,
        reason="OOT outcomes have not matured.",
    )
    assert metric["model_values"] is None and metric["delta"] is None

    with pytest.raises(StrategyModelEvidenceError, match="must be null"):
        build_model_comparison_metric(
            sample_design_bundle=sample_design,
            population="risk",
            partition="oot",
            metric_key="auc",
            status="not_matured",
            unit="ratio",
            source_ref=_source(sample_design, "maturity", "risk", "oot"),
            model_values=[],
            delta=None,
            reason="OOT outcomes have not matured.",
        )


def test_unknown_fields_tampering_duplicate_json_keys_and_bad_root_fail_closed():
    sample_design, evidence = _bundle()
    with pytest.raises(StrategyModelEvidenceError, match="unknown"):
        validate_strategy_model_evidence_bundle(
            {**evidence, "invented": True},
            sample_design_bundle=sample_design,
        )

    tampered = deepcopy(evidence)
    tampered["model_evidence"][0]["features"] = ["invented"]
    with pytest.raises(StrategyModelEvidenceError, match="content"):
        validate_strategy_model_evidence_bundle(
            tampered,
            sample_design_bundle=sample_design,
        )

    with pytest.raises(StrategyModelEvidenceError, match="duplicate key"):
        strategy_model_evidence_bundle_from_json(
            '{"schema_version":1,"schema_version":2}',
            sample_design_bundle=sample_design,
        )
    with pytest.raises(StrategyModelEvidenceError, match="must contain an object"):
        strategy_model_evidence_bundle_from_json(
            "[]",
            sample_design_bundle=sample_design,
        )


def test_content_addressed_ids_use_sample_design_style_24_hex_suffixes():
    _, evidence = _bundle()
    assert evidence["bundle_id"].startswith("strategy-model-evidence-bundle-")
    assert len(evidence["bundle_id"].rsplit("-", 1)[1]) == 24
    assert all(
        len(item["evidence_id"].rsplit("-", 1)[1]) == 24
        for item in [
            *evidence["univariate_evidence"],
            *evidence["model_evidence"],
        ]
    )
    assert all(len(item["content_hash"]) == 64 for item in evidence["model_evidence"])
