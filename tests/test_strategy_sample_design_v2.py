from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

import marvis.packs.strategy.sample_design_v2 as sample_design_v2_module
from marvis.packs.strategy.sample_design_v2 import (
    DIAGNOSTIC_STATUSES,
    METRIC_OBSERVATION_V2_STATUSES,
    SAMPLE_RELATIONSHIPS,
    STRATEGY_METRIC_DEFINITION_V2_SCHEMA_VERSION,
    STRATEGY_METRIC_OBSERVATION_V2_SCHEMA_VERSION,
    StrategySampleDesignV2Error,
    build_historical_score_v2,
    build_metric_definitions_v2,
    build_metric_observation_v2,
    build_sample_design_policy_v2,
    build_sample_population_v2,
    build_strategy_sample_design_v2,
    build_strategy_sample_design_v2_bundle,
    build_target_selector_v2,
    canonical_strategy_sample_design_v2_bundle_json,
    strategy_sample_design_v2_bundle_from_json,
    validate_metric_observation_v2,
    validate_strategy_sample_design_v2_bundle,
)
from marvis.packs.strategy.sample_membership import (
    decode_sample_membership,
    encode_sample_membership,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _source_ref(label: str, *, kind: str = "tool_output") -> dict[str, str]:
    return {"kind": kind, "ref_id": label, "content_hash": _hash(label)}


def _legacy_ref(*, partition: str = "development") -> dict[str, str]:
    return {
        "artifact_id": _hash("legacy-artifact-id"),
        "artifact_content_hash": _hash("legacy-artifact"),
        "sample_design_id": "strategy-sample-design-legacy",
        "sample_design_content_hash": _hash("legacy-design"),
        "partition": partition,
    }


def _decoded_membership(
    *,
    risk_outside_approval: bool = False,
    empty_development: bool = False,
    empty_oot: bool = False,
):
    approval_development = np.array(
        [False, False, False, False, False, False, False, False]
        if empty_development
        else [True, True, True, False, False, False, False, False]
    )
    risk_development = np.array(
        [False, False, False, False, False, False, False, False]
        if empty_development
        else [True, True, False, False, False, False, False, False]
    )
    if risk_outside_approval:
        risk_development[7] = True
    masks = {
        "approval/development": approval_development,
        "approval/validation": np.array(
            [False, False, False, True, False, True, False, False]
        ),
        "approval/oot": np.array(
            [False, False, False, False, False, False, False, False]
            if empty_oot
            else [False, False, False, False, True, False, True, False]
        ),
        "risk/development": risk_development,
        "risk/validation": np.array(
            [False, False, False, True, False, True, False, False]
        ),
        "risk/oot": np.array(
            [False, False, False, False, False, False, False, False]
            if empty_oot
            else [False, False, False, False, True, False, True, False]
        ),
    }
    return decode_sample_membership(
        encode_sample_membership(
            task_id="task-v2",
            dataset_id="dataset-v2",
            dataset_content_hash=_hash("dataset-v2"),
            masks=masks,
        )
    )


def _policy():
    return build_sample_design_policy_v2(
        minimum_partition_count=1,
        minimum_bad_count=1,
        minimum_label_coverage=0.8,
        minimum_historical_score_coverage=0.8,
        maximum_group_coverage_gap=0.1,
    )


def _maturity(status: str, total: int) -> dict[str, object]:
    if status == "confirmed_matured":
        return {
            "status": status,
            "performance_window_days": 90,
            "cutoff_date": "2026-06-30",
            "eligible_count": total,
            "labeled_count": total,
            "source_refs": [_source_ref("maturity")],
            "reason": None,
        }
    if status == "not_matured":
        eligible = min(total, 2)
        return {
            "status": status,
            "performance_window_days": 90,
            "cutoff_date": "2026-06-30",
            "eligible_count": eligible,
            "labeled_count": eligible,
            "source_refs": [_source_ref("maturity")],
            "reason": "Performance window has not elapsed for every risk row.",
        }
    return {
        "status": status,
        "performance_window_days": None,
        "cutoff_date": None,
        "eligible_count": None,
        "labeled_count": None,
        "source_refs": [_source_ref(f"maturity-{status}")],
        "reason": f"Maturity is {status}.",
    }


def _components(decoded, *, maturity_status: str = "confirmed_matured"):
    predicate_ref = _source_ref("predicate", kind="predicate_ast")
    approval = build_sample_population_v2(
        role="approval",
        membership_header=decoded["header"],
        inclusion_predicate_ref=predicate_ref,
        exclusion_predicate_ref=None,
        source_refs=[predicate_ref],
    )
    risk = build_sample_population_v2(
        role="risk",
        membership_header=decoded["header"],
        inclusion_predicate_ref=predicate_ref,
        exclusion_predicate_ref=None,
        maturity_evidence=_maturity(
            maturity_status, decoded["header"]["counts"]["risk"]["total"]
        ),
        source_refs=[predicate_ref, _source_ref("maturity")],
    )
    target = build_target_selector_v2(
        status="resolved",
        column="target",
        good_value=0,
        bad_value=1,
        drop_missing=True,
        source_refs=[_source_ref("target")],
    )
    historical = build_historical_score_v2(
        status="available",
        column="legacy_score",
        direction="higher_is_riskier",
        source_refs=[_source_ref("historical-score")],
    )
    return approval, risk, target, historical


def _design_kwargs(*, maturity_status: str = "confirmed_matured") -> dict:
    return {
        "workspace_revision": 7,
        "workspace_generation": 3,
        "semantic_mapping_hash": _hash("semantic-mapping"),
        "field_bindings": {
            "entity_field": "customer_id",
            "time_field": "apply_date",
            "group_field": "channel",
            "month_field": "apply_month",
            "weight_field": None,
            "loan_amount_field": "loan_amount",
            "overdue_amount_field": "overdue_amount",
        },
        "scope": (
            "strategy_development"
            if maturity_status == "confirmed_matured"
            else "exploration_only"
        ),
        "performance_window": {"status": "provided", "days": 90},
        "observation_window": {
            "status": "provided",
            "start": "2026-01-01",
            "end": "2026-06-30",
        },
        "split_definition": {
            "status": "available",
            "method": "precomputed_masks",
            "column": None,
            "development_values": [],
            "validation_values": [],
            "oot_values": [],
            "source_refs": [_source_ref("split")],
        },
        "legacy_development_ref": _legacy_ref(),
    }


def _diagnostic_statistics(decoded) -> dict:
    risk_total = decoded["header"]["counts"]["risk"]["total"]
    return {
        "entity_overlap": {
            "availability": "available",
            "overlap_count": 0,
            "compared_count": 7,
            "source_refs": [_source_ref("entity-overlap")],
        },
        "temporal_oot": {
            "availability": "available",
            "ordered": True,
            "source_refs": [_source_ref("temporal-oot")],
        },
        "historical_score_coverage": {
            "availability": "available",
            "covered_count": risk_total,
            "eligible_count": risk_total,
            "source_refs": [_source_ref("score-coverage")],
        },
        "group_coverage_gap": {
            "availability": "available",
            "maximum_gap": 0.05,
            "group_count": 2,
            "source_refs": [_source_ref("group-gap")],
        },
        "sufficiency": {
            "availability": "available",
            "bad_count": min(
                1,
                decoded["header"]["counts"]["risk"]["development"],
            ),
            "source_refs": [_source_ref("sufficiency")],
        },
    }


def _build_design(
    decoded,
    *,
    relationship: str,
    maturity_status: str,
):
    approval, risk, target, historical = _components(
        decoded, maturity_status=maturity_status
    )
    design = build_strategy_sample_design_v2(
        task_id="task-v2",
        membership_header=decoded["header"],
        relationship=relationship,
        target_selector=target,
        approval_population=approval,
        risk_population=risk,
        historical_score=historical,
        policy=_policy(),
        source_refs=[_source_ref("design-b"), _source_ref("design-a")],
        **_design_kwargs(maturity_status=maturity_status),
    )
    return design, approval, risk, target, historical


def _required_sources(decoded, design, label: str) -> list[dict[str, str]]:
    return [
        {
            "kind": "dataset",
            "ref_id": decoded["header"]["dataset_ref"]["dataset_id"],
            "content_hash": decoded["header"]["dataset_ref"]["content_hash"],
        },
        {
            "kind": "sample_membership",
            "ref_id": decoded["header"]["membership_id"],
            "content_hash": decoded["header"]["content_hash"],
        },
        {
            "kind": "sample_design",
            "ref_id": design["sample_design_id"],
            "content_hash": design["content_hash"],
        },
        _source_ref(label),
    ]


def _metric_observations(
    decoded,
    design,
    *,
    maturity_status: str,
    single_class_development: bool = False,
    single_class_validation: bool = False,
):
    definitions = {
        item["metric_key"]: item for item in build_metric_definitions_v2()
    }
    design_ref = {
        "sample_design_id": design["sample_design_id"],
        "content_hash": design["content_hash"],
    }
    observations = []
    for role in ("approval", "risk"):
        partition_counts = decoded["header"]["counts"][role]
        labels: dict[str, int] = {}
        if role == "risk" and maturity_status == "not_matured":
            remaining = min(partition_counts["total"], 2)
            for partition in ("development", "validation", "oot"):
                labels[partition] = min(partition_counts[partition], remaining)
                remaining -= labels[partition]
        else:
            labels = {
                partition: partition_counts[partition]
                for partition in ("development", "validation", "oot")
            }
        labels["overall"] = sum(labels.values())
        bads = {
            partition: int(labels[partition] > 0)
            for partition in ("development", "validation", "oot")
        }
        if role == "risk" and single_class_development:
            bads["development"] = labels["development"]
        if role == "risk" and single_class_validation:
            bads["validation"] = labels["validation"]
        bads["overall"] = sum(bads.values())
        for partition in ("overall", "development", "validation", "oot"):
            population = (
                partition_counts["total"]
                if partition == "overall"
                else partition_counts[partition]
            )
            labeled = labels[partition]
            bad = bads[partition]
            sources = _required_sources(
                decoded, design, f"{role}-{partition}-statistics"
            )
            values = {
                "population_count": ("present", population, population, population),
            }
            if role == "approval":
                values.update(
                    {
                        metric_key: ("not_applicable", None, None, None)
                        for metric_key in (
                            "labeled_count",
                            "label_coverage",
                            "bad_count",
                            "bad_rate",
                        )
                    }
                )
            else:
                values.update(
                    {
                        "labeled_count": ("present", labeled, labeled, population),
                        "label_coverage": (
                            ("present", labeled / population, labeled, population)
                            if population
                            else ("insufficient_data", None, None, None)
                        ),
                    }
                )
            if role == "approval":
                bad_status = "not_applicable"
            elif maturity_status == "not_matured":
                bad_status = "not_matured"
            elif maturity_status in {"unknown", "unavailable"}:
                bad_status = "unavailable"
            else:
                bad_status = "present"
            if bad_status == "present":
                values["bad_count"] = ("present", bad, bad, labeled)
                values["bad_rate"] = (
                    ("present", bad / labeled, bad, labeled)
                    if labeled
                    else ("insufficient_data", None, None, None)
                )
            else:
                values["bad_count"] = (bad_status, None, None, None)
                values["bad_rate"] = (bad_status, None, None, None)
            for metric_key, (status, value, numerator, denominator) in values.items():
                observations.append(
                    build_metric_observation_v2(
                        sample_design_ref=design_ref,
                        metric_definition=definitions[metric_key],
                        population=role,
                        partition=partition,
                        status=status,
                        value=value,
                        numerator=numerator,
                        denominator=denominator,
                        sample_count=population,
                        source_refs=sources,
                    )
                )
    return observations


def _bundle(
    *,
    relationship: str = "nested_same_cohort",
    outside: bool = False,
    maturity_status: str = "confirmed_matured",
    empty_development: bool = False,
    empty_oot: bool = False,
    single_class_development: bool = False,
    single_class_validation: bool = False,
):
    decoded = _decoded_membership(
        risk_outside_approval=outside,
        empty_development=empty_development,
        empty_oot=empty_oot,
    )
    design, approval, risk, target, historical = _build_design(
        decoded,
        relationship=relationship,
        maturity_status=maturity_status,
    )
    return build_strategy_sample_design_v2_bundle(
        task_id="task-v2",
        membership_header=decoded["header"],
        membership_masks=decoded["masks"],
        relationship=relationship,
        target_selector=target,
        approval_population=approval,
        risk_population=risk,
        historical_score=historical,
        policy=_policy(),
        diagnostic_statistics=_diagnostic_statistics(decoded),
        metric_observations=_metric_observations(
            decoded,
            design,
            maturity_status=maturity_status,
            single_class_development=single_class_development,
            single_class_validation=single_class_validation,
        ),
        source_refs=[_source_ref("design-a"), _source_ref("design-b")],
        **_design_kwargs(maturity_status=maturity_status),
    )


def test_v2_bundle_binds_governed_identity_semantics_and_complete_metrics():
    bundle = _bundle()

    assert SAMPLE_RELATIONSHIPS == frozenset(
        {"nested_same_cohort", "parallel_time_cohorts"}
    )
    assert DIAGNOSTIC_STATUSES == frozenset(
        {"pass", "warn", "fail", "unavailable", "not_applicable"}
    )
    assert {"not_matured", "insufficient_data"} <= METRIC_OBSERVATION_V2_STATUSES
    assert [item["role"] for item in bundle["populations"]] == [
        "approval",
        "risk",
    ]
    design = bundle["sample_design"]
    assert design["identity"] == {
        "task_id": "task-v2",
        "dataset_ref": {
            **bundle["membership"]["dataset_ref"],
            "role": "active",
        },
        "workspace_ref": {
            "revision": 7,
            "generation": 3,
            "semantic_mapping_hash": _hash("semantic-mapping"),
        },
    }
    assert design["sample_semantics"]["scope"] == "strategy_development"
    assert design["compatibility"] == {
        "legacy_development_ref": _legacy_ref(),
        "maps_to": "risk/development",
    }
    assert {
        item["metric_key"] for item in bundle["metric_definitions"]
    } == {
        "population_count",
        "labeled_count",
        "label_coverage",
        "bad_count",
        "bad_rate",
    }
    assert all(
        item["schema_version"] == STRATEGY_METRIC_DEFINITION_V2_SCHEMA_VERSION
        for item in bundle["metric_definitions"]
    )
    assert len(bundle["metric_observations"]) == 2 * 4 * 5
    assert all(item["source_refs"] for item in bundle["metric_observations"])
    metric_key_by_id = {
        item["metric_definition_id"]: item["metric_key"]
        for item in bundle["metric_definitions"]
    }
    assert {
        item["status"]
        for item in bundle["metric_observations"]
        if item["population"] == "approval"
        and metric_key_by_id[
            item["metric_definition_ref"]["metric_definition_id"]
        ]
        != "population_count"
    } == {"not_applicable"}
    assert all(item["status"] == "pass" for item in bundle["diagnostics"])

    canonical = canonical_strategy_sample_design_v2_bundle_json(bundle)
    assert strategy_sample_design_v2_bundle_from_json(canonical) == bundle
    assert validate_strategy_sample_design_v2_bundle(bundle) == bundle


def test_legacy_anchored_v2_bundle_keeps_golden_identity_and_canonical_bytes():
    bundle = _bundle()
    canonical = canonical_strategy_sample_design_v2_bundle_json(bundle)

    assert bundle["bundle_id"] == (
        "strategy-sample-design-bundle-b49814f01db078f4878ba6cb"
    )
    assert bundle["content_hash"] == (
        "d816c009074a462a4229cc749d304b953d5695375d2892b49d0850ba0613fa63"
    )
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == (
        "6ad1b796d308f71fd0a043bdf75558a3498cbcaf291138170c83177080492898"
    )


def test_v2_bundle_accepts_native_active_dataset_without_legacy_compatibility():
    decoded = _decoded_membership()
    approval, risk, target, historical = _components(decoded)
    kwargs = _design_kwargs()
    kwargs.pop("legacy_development_ref")
    design = build_strategy_sample_design_v2(
        task_id="task-v2",
        membership_header=decoded["header"],
        relationship="nested_same_cohort",
        target_selector=target,
        approval_population=approval,
        risk_population=risk,
        historical_score=historical,
        policy=_policy(),
        source_mode="native_active_dataset",
        source_refs=[_source_ref("design")],
        **kwargs,
    )

    assert design["compatibility"] == {
        "source_mode": "native_active_dataset",
        "development_partition": "risk/development",
    }


@pytest.mark.parametrize(
    ("maturity_status", "expected_diagnostic", "expected_metric_status"),
    [
        ("not_matured", "fail", "not_matured"),
        ("unknown", "unavailable", "unavailable"),
        ("unavailable", "unavailable", "unavailable"),
    ],
)
def test_nonempty_risk_preserves_non_matured_or_unknown_statuses(
    maturity_status: str,
    expected_diagnostic: str,
    expected_metric_status: str,
):
    bundle = _bundle(maturity_status=maturity_status)
    statuses = {item["code"]: item["status"] for item in bundle["diagnostics"]}
    assert bundle["populations"][1]["total_count"] > 0
    assert bundle["populations"][1]["maturity_evidence"]["status"] == maturity_status
    assert statuses["maturity"] == expected_diagnostic
    assert statuses["label_coverage"] == expected_diagnostic
    assert {
        item["status"]
        for item in bundle["metric_observations"]
        if item["population"] == "risk"
        and next(
            definition["metric_key"]
            for definition in bundle["metric_definitions"]
            if definition["metric_definition_id"]
            == item["metric_definition_ref"]["metric_definition_id"]
        )
        in {"bad_count", "bad_rate"}
    } == {expected_metric_status}


def test_nested_relationship_rejects_each_partition_outside_membership():
    with pytest.raises(StrategySampleDesignV2Error, match="subset.*development"):
        _bundle(outside=True)
    parallel = _bundle(relationship="parallel_time_cohorts", outside=True)
    statuses = {item["code"]: item["status"] for item in parallel["diagnostics"]}
    assert statuses["risk_outside_approval"] == "not_applicable"


@pytest.mark.parametrize(
    ("good_value", "bad_value"),
    [(True, 1), (0, False), (1, 1.0), (0.0, 0), (2, 0), ("0", 1)],
)
def test_target_selector_requires_complementary_numeric_zero_one(
    good_value, bad_value
):
    with pytest.raises(StrategySampleDesignV2Error, match="0 or 1|complementary"):
        build_target_selector_v2(
            status="resolved",
            column="target",
            good_value=good_value,
            bad_value=bad_value,
            drop_missing=True,
            source_refs=[_source_ref("target")],
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda ref: {**ref, "partition": "validation"},
        lambda ref: {key: value for key, value in ref.items() if key != "artifact_id"},
        lambda ref: {**ref, "unknown": "value"},
        lambda ref: {**ref, "artifact_content_hash": "not-a-hash"},
    ],
)
def test_legacy_compatibility_requires_exact_v1_development_ref(mutation):
    decoded = _decoded_membership()
    approval, risk, target, historical = _components(decoded)
    kwargs = _design_kwargs()
    kwargs["legacy_development_ref"] = mutation(_legacy_ref())
    with pytest.raises(StrategySampleDesignV2Error, match="sample_design_ref"):
        build_strategy_sample_design_v2(
            task_id="task-v2",
            membership_header=decoded["header"],
            relationship="nested_same_cohort",
            target_selector=target,
            approval_population=approval,
            risk_population=risk,
            historical_score=historical,
            policy=_policy(),
            source_refs=[_source_ref("design")],
            **kwargs,
        )


def test_metric_observation_binds_definition_and_rejects_tamper():
    bundle = _bundle()
    definition = next(
        item for item in bundle["metric_definitions"] if item["metric_key"] == "bad_rate"
    )
    observation = next(
        item
        for item in bundle["metric_observations"]
        if item["population"] == "risk"
        and item["partition"] == "development"
        and item["metric_definition_ref"]["metric_definition_id"]
        == definition["metric_definition_id"]
    )
    assert observation["schema_version"] == STRATEGY_METRIC_OBSERVATION_V2_SCHEMA_VERSION
    assert (
        validate_metric_observation_v2(
            observation, metric_definitions=bundle["metric_definitions"]
        )
        == observation
    )
    tampered = {**observation, "value": 0.75}
    with pytest.raises(
        StrategySampleDesignV2Error, match="inconsistent|does not match"
    ):
        validate_metric_observation_v2(
            tampered, metric_definitions=bundle["metric_definitions"]
        )


def test_bundle_rejects_unknown_fields_and_address_tampering():
    bundle = _bundle()
    unknown = {**bundle, "invented": True}
    with pytest.raises(StrategySampleDesignV2Error, match="unknown: invented"):
        validate_strategy_sample_design_v2_bundle(unknown)

    tampered = deepcopy(bundle)
    tampered["sample_design"]["identity"]["workspace_ref"]["revision"] = 99
    with pytest.raises(StrategySampleDesignV2Error, match="does not match content"):
        validate_strategy_sample_design_v2_bundle(tampered)


@pytest.mark.parametrize(
    ("metric_key", "field", "replacement", "message"),
    [
        ("population_count", "value", 999, "population_count"),
        ("labeled_count", "value", 999, "labeled_count"),
        ("bad_count", "value", 999, "bad_count"),
        ("bad_rate", "numerator", 999, "bad_rate|ratio metric"),
    ],
)
def test_observation_conservation_rejects_readdressed_invalid_counts(
    metric_key: str, field: str, replacement: int, message: str
):
    bundle = _bundle()
    definition_by_id = {
        item["metric_definition_id"]: item["metric_key"]
        for item in bundle["metric_definitions"]
    }
    index = next(
        index
        for index, item in enumerate(bundle["metric_observations"])
        if item["population"] == "risk"
        and item["partition"] == "development"
        and definition_by_id[item["metric_definition_ref"]["metric_definition_id"]]
        == metric_key
    )
    original = bundle["metric_observations"][index]
    body = {
        key: value
        for key, value in original.items()
        if key not in {"observation_id", "content_hash"}
    }
    body[field] = replacement
    if metric_key == "bad_count" and field == "value":
        body["numerator"] = replacement
    rebuilt = sample_design_v2_module._address_object(body, "observation_id")
    invalid = deepcopy(bundle)
    invalid["metric_observations"][index] = rebuilt
    bundle_body = {
        key: value
        for key, value in invalid.items()
        if key not in {"bundle_id", "content_hash"}
    }
    invalid = sample_design_v2_module._address_object(bundle_body, "bundle_id")
    with pytest.raises(StrategySampleDesignV2Error, match=message):
        validate_strategy_sample_design_v2_bundle(invalid)


def test_bundle_rejects_empty_or_incomplete_observations_and_metric_definitions():
    bundle = _bundle()
    for field, message in (
        ("metric_observations", "must not be empty"),
        ("metric_definitions", "must not be empty"),
    ):
        body = {
            key: ([] if key == field else value)
            for key, value in bundle.items()
            if key not in {"bundle_id", "content_hash"}
        }
        invalid = sample_design_v2_module._address_object(body, "bundle_id")
        with pytest.raises(StrategySampleDesignV2Error, match=message):
            validate_strategy_sample_design_v2_bundle(invalid)


def test_design_source_refs_are_canonical_before_addressing():
    decoded = _decoded_membership()
    approval, risk, target, historical = _components(decoded)
    common = {
        "task_id": "task-v2",
        "membership_header": decoded["header"],
        "relationship": "nested_same_cohort",
        "target_selector": target,
        "approval_population": approval,
        "risk_population": risk,
        "historical_score": historical,
        "policy": _policy(),
        **_design_kwargs(),
    }
    refs = [_source_ref("b"), _source_ref("a")]
    first = build_strategy_sample_design_v2(source_refs=refs, **common)
    second = build_strategy_sample_design_v2(
        source_refs=list(reversed(refs)), **common
    )
    assert first == second


def test_json_loader_rejects_duplicate_nonfinite_and_budget(monkeypatch):
    bundle = _bundle()
    canonical = canonical_strategy_sample_design_v2_bundle_json(bundle)
    with pytest.raises(StrategySampleDesignV2Error, match="duplicate key"):
        strategy_sample_design_v2_bundle_from_json(
            canonical.replace("{", '{"schema_version":"duplicate",', 1)
        )
    nonfinite = canonical.replace('"workspace_ref":{"generation":3', '"workspace_ref":{"generation":NaN')
    with pytest.raises(StrategySampleDesignV2Error, match="non-finite|generation"):
        strategy_sample_design_v2_bundle_from_json(nonfinite)
    monkeypatch.setattr(sample_design_v2_module, "MAX_SAMPLE_DESIGN_V2_JSON_BYTES", 1)
    with pytest.raises(StrategySampleDesignV2Error, match="byte budget"):
        strategy_sample_design_v2_bundle_from_json(canonical)


def test_diagnostic_statistics_cannot_exceed_bound_membership():
    decoded = _decoded_membership()
    design, approval, risk, target, historical = _build_design(
        decoded,
        relationship="nested_same_cohort",
        maturity_status="confirmed_matured",
    )
    stats = _diagnostic_statistics(decoded)
    stats["sufficiency"]["bad_count"] = 999
    with pytest.raises(StrategySampleDesignV2Error, match="bad_count exceeds"):
        build_strategy_sample_design_v2_bundle(
            task_id="task-v2",
            membership_header=decoded["header"],
            membership_masks=decoded["masks"],
            relationship="nested_same_cohort",
            target_selector=target,
            approval_population=approval,
            risk_population=risk,
            historical_score=historical,
            policy=_policy(),
            diagnostic_statistics=stats,
            metric_observations=_metric_observations(
                decoded, design, maturity_status="confirmed_matured"
            ),
            source_refs=[_source_ref("design-a"), _source_ref("design-b")],
            **_design_kwargs(),
        )


def test_fixed_metric_definition_rejects_unknown_and_nonfinite_observation():
    definitions = build_metric_definitions_v2()
    unknown = {**definitions[0], "future": True}
    with pytest.raises(StrategySampleDesignV2Error, match="unknown: future"):
        sample_design_v2_module.validate_metric_definition_v2(unknown)

    bundle = _bundle()
    observation = bundle["metric_observations"][0]
    body = {
        key: value
        for key, value in observation.items()
        if key not in {"observation_id", "content_hash"}
    }
    body["value"] = float("inf")
    with pytest.raises(StrategySampleDesignV2Error, match="non-finite|finite"):
        sample_design_v2_module._address_object(body, "observation_id")


def test_bundle_json_is_canonical_json_not_python_nan():
    bundle = _bundle()
    raw = canonical_strategy_sample_design_v2_bundle_json(bundle)
    assert json.loads(raw)["bundle_id"] == bundle["bundle_id"]
    assert "NaN" not in raw and "Infinity" not in raw
