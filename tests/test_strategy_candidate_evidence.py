from __future__ import annotations

from copy import deepcopy
import json

import pytest

from marvis.packs.strategy.candidate_evidence import (
    CANDIDATE_EVIDENCE_SCHEMA_VERSION,
    CandidateEvidenceError,
    MetricObservation,
    build_candidate_evidence,
    candidate_evidence_from_json,
    candidate_evidence_hash,
    candidate_evidence_to_json,
    canonical_candidate_evidence_json,
    validate_candidate_evidence,
)


HASH_A = "a" * 64
HASH_B = "b" * 64


def _build(**overrides):
    values = {
        "task_id": "task-1",
        "dataset_id": "dataset-1",
        "dataset_content_hash": HASH_A,
        "workspace_revision": 3,
        "workspace_generation": 2,
        "semantic_mapping_hash": HASH_B,
        "generation_parameters": {"features": ["age"], "bins": 5},
        "seed": 20260719,
        "budget": 100,
        "truncated": False,
        "analysis": {"feature": "age", "cut": 30, "operator": "<"},
        "metrics": [
            MetricObservation("hit_rate", "count", "observed", 0.12),
            MetricObservation("hit_rate", "loan_amount", "observed", 0.15),
            MetricObservation("hit_rate", "overdue_amount", "unavailable", None),
        ],
        "source_refs": ["analysis:run-1", "dataset:dataset-1"],
        "red_flags": ["overdue_amount_unavailable"],
        "producer_version": "strategy.univariate-candidate/1",
    }
    values.update(overrides)
    return build_candidate_evidence(**values)


def test_builds_exact_deterministic_self_authenticating_contract():
    first = _build()
    second = _build(
        metrics=list(reversed(_build()["metrics"])),
        source_refs=list(reversed(_build()["source_refs"])),
    )

    assert first == second
    assert first["schema_version"] == CANDIDATE_EVIDENCE_SCHEMA_VERSION
    assert first["candidate_id"].startswith("candidate-")
    assert len(first["evidence_hash"]) == 64
    assert candidate_evidence_hash(first) == first["evidence_hash"]
    assert json.loads(canonical_candidate_evidence_json(first)) == first


def test_json_roundtrip_is_canonical_and_detached():
    payload = _build()
    raw = candidate_evidence_to_json(payload)
    restored = candidate_evidence_from_json(raw)

    assert restored == payload
    assert raw == candidate_evidence_to_json(restored)
    restored["analysis"]["cut"] = 99
    assert payload["analysis"]["cut"] == 30


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("identity", "task_id"), "task-2"),
        (("identity", "dataset_id"), "dataset-2"),
        (("identity", "dataset_content_hash"), "c" * 64),
        (("identity", "workspace_revision"), 4),
        (("identity", "workspace_generation"), 3),
        (("identity", "semantic_mapping_hash"), "d" * 64),
        (("generation", "seed"), 5),
        (("generation", "budget"), 99),
        (("analysis", "cut"), 29),
        (("metrics", 0, "value"), 0.13),
        (("producer_version",), "strategy.univariate-candidate/2"),
    ],
)
def test_any_identity_or_evidence_mutation_invalidates_contract(path, replacement):
    payload = deepcopy(_build())
    target = payload
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = replacement

    with pytest.raises(CandidateEvidenceError):
        validate_candidate_evidence(payload)


def test_rejects_forged_candidate_id_or_evidence_hash():
    forged_id = deepcopy(_build())
    forged_id["candidate_id"] = "candidate-" + "0" * 32
    with pytest.raises(CandidateEvidenceError, match="candidate_id"):
        validate_candidate_evidence(forged_id)

    forged_hash = deepcopy(_build())
    forged_hash["evidence_hash"] = "0" * 64
    with pytest.raises(CandidateEvidenceError, match="evidence_hash"):
        validate_candidate_evidence(forged_hash)


@pytest.mark.parametrize(
    "field", ["candidate_id", "analysis", "metrics", "producer_version"]
)
def test_rejects_missing_top_level_fields(field):
    payload = deepcopy(_build())
    del payload[field]
    with pytest.raises(CandidateEvidenceError, match="missing"):
        validate_candidate_evidence(payload)


@pytest.mark.parametrize(
    "container", [(), ("identity",), ("generation",), ("metrics", 0)]
)
def test_rejects_unknown_fields_at_every_contract_level(container):
    payload = deepcopy(_build())
    target = payload
    for part in container:
        target = target[part]
    target["unexpected"] = True
    with pytest.raises(CandidateEvidenceError, match="unknown"):
        validate_candidate_evidence(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate_type", "tree"),
        ("effect_stage", "validation"),
        ("validation_status", "validated"),
    ],
)
def test_rejects_validation_or_other_scope_claims(field, value):
    payload = deepcopy(_build())
    payload[field] = value
    with pytest.raises(CandidateEvidenceError):
        validate_candidate_evidence(payload)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"workspace_revision": True}, "workspace_revision"),
        ({"workspace_generation": False}, "workspace_generation"),
        ({"seed": True}, "seed"),
        ({"budget": False}, "budget"),
        ({"truncated": 1}, "truncated"),
    ],
)
def test_rejects_bool_as_integer_or_integer_as_boolean(override, match):
    with pytest.raises(CandidateEvidenceError, match=match):
        _build(**override)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf"), True])
def test_rejects_nonfinite_and_boolean_metric_values(bad_value):
    with pytest.raises(CandidateEvidenceError, match="finite number"):
        _build(metrics=[MetricObservation("hit_rate", "count", "observed", bad_value)])


def test_rejects_nonfinite_nested_analysis_or_parameters():
    with pytest.raises(CandidateEvidenceError, match="finite canonical JSON"):
        _build(analysis={"score": float("nan")})
    with pytest.raises(CandidateEvidenceError, match="finite canonical JSON"):
        _build(generation_parameters={"cut": float("inf")})


def test_rejects_duplicate_metric_identity_and_bad_status_value_pair():
    duplicate = [
        MetricObservation("hit_rate", "count", "observed", 0.1),
        MetricObservation("hit_rate", "count", "observed", 0.2),
    ]
    with pytest.raises(CandidateEvidenceError, match="duplicate metric identities"):
        _build(metrics=[*duplicate, *_complete_other_dimensions("hit_rate")])

    with pytest.raises(CandidateEvidenceError, match="must be null"):
        _build(
            metrics=[
                MetricObservation("hit_rate", "count", "observed", 0.1),
                MetricObservation("hit_rate", "loan_amount", "observed", 0.1),
                MetricObservation("hit_rate", "overdue_amount", "unavailable", 0.1),
            ]
        )


def _complete_other_dimensions(metric_name):
    return [
        MetricObservation(metric_name, "loan_amount", "unavailable", None),
        MetricObservation(metric_name, "overdue_amount", "unavailable", None),
    ]


def test_requires_every_metric_dimension_with_explicit_missing_status():
    with pytest.raises(CandidateEvidenceError, match="explicitly cover"):
        _build(metrics=[MetricObservation("hit_rate", "count", "observed", 0.1)])


@pytest.mark.parametrize("container", ["analysis", "generation_parameters"])
def test_rejects_nested_validation_claims(container):
    with pytest.raises(CandidateEvidenceError, match="validation or adoption claims"):
        _build(**{container: {"nested": {"validation_status": "validated"}}})


@pytest.mark.parametrize("hash_value", ["A" * 64, "a" * 63, "not-a-hash"])
def test_rejects_noncanonical_identity_hashes(hash_value):
    with pytest.raises(CandidateEvidenceError, match="SHA-256"):
        _build(dataset_content_hash=hash_value)


def test_rejects_duplicate_or_empty_sources_and_bad_json_root():
    with pytest.raises(CandidateEvidenceError, match="must not be empty"):
        _build(source_refs=[])
    with pytest.raises(CandidateEvidenceError, match="duplicates"):
        _build(source_refs=["dataset:1", "dataset:1"])
    with pytest.raises(CandidateEvidenceError, match="must contain an object"):
        candidate_evidence_from_json("[]")
    with pytest.raises(CandidateEvidenceError, match="duplicate key"):
        candidate_evidence_from_json('{"schema_version":1,"schema_version":2}')
