from __future__ import annotations

from copy import deepcopy
import json
import re

import pandas as pd
import pytest

from marvis.feature.iv import _smoothed_woe_iv
from marvis.feature.univariate import analyze_univariate
from marvis.packs.strategy.candidate_evidence import build_candidate_evidence
from marvis.packs.strategy.cross_matrix_candidate import (
    CROSS_MATRIX_CANDIDATE_ASSET_SCHEMA_VERSION,
    CROSS_MATRIX_CANDIDATE_ASSET_V2_SCHEMA_VERSION,
    CROSS_MATRIX_MEASUREMENT_SCHEMA_VERSION,
    CrossMatrixCandidateAssetError,
    build_cross_matrix_candidate_asset,
    canonical_cross_matrix_candidate_asset_json,
    parse_cross_matrix_candidate_asset_json,
    rebuild_cross_matrix_candidate_asset,
    validate_cross_matrix_candidate_asset,
)
from marvis.packs.strategy.candidate_fragment import (
    sample_context_hash_from_candidate_evidence,
)
from marvis.packs.strategy.evaluator import evaluate_expression_frame


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
ID_RE = re.compile(r"^[a-z-]+-[0-9a-f]{32}$")


def _frame() -> pd.DataFrame:
    # Correlated features deliberately leave multiple Cartesian cells empty.
    return pd.DataFrame(
        {
            "age": [20, 21, 22, 30, 31, 32, 40, 41, 42, 50, 51, 52],
            "score": [300, 310, 320, 400, 410, 420, 500, 510, 520, 600, 610, 620],
            "bad": [0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1],
            "loan": [100.0, 120, 140, 200, 220, 240, 300, 320, 340, 400, 420, 440],
            "overdue": [0.0, 0, 10, 0, 20, 30, 0, 0, 40, 50, 60, 70],
        }
    )


def _analysis(
    frame: pd.DataFrame | None = None,
    *,
    method: str = "equal_width",
    manual_breakpoints: dict[str, list[float]] | None = None,
) -> dict:
    source = _frame() if frame is None else frame
    return analyze_univariate(
        source,
        features=["age", "score"],
        target="bad",
        methods=[method],
        manual_breakpoints=manual_breakpoints,
        bin_count=4,
        min_bin_pct=0,
        loan_amount="loan",
        overdue_amount="overdue",
    )


def _evidence(
    analysis: dict | None = None,
    *,
    dataset_hash: str = HASH_A,
    producer_version: str | None = None,
    generation_analysis_schema_version: str | None = None,
) -> dict:
    resolved_analysis = _analysis() if analysis is None else analysis
    method = resolved_analysis["features"][0]["methods"][0]["method"]
    generation_parameters = {
        "analysis_schema_version": (
            generation_analysis_schema_version
            or resolved_analysis["schema_version"]
        ),
        "features": ["age", "score"],
        "methods": [method],
        "sample_design_ref": {
            "artifact_id": HASH_A,
            "artifact_content_hash": HASH_B,
            "sample_design_id": "strategy-sample-design-cross-1",
            "sample_design_content_hash": HASH_C,
            "partition": "development",
        },
    }
    if method == "manual":
        generation_parameters["manual_breakpoints"] = resolved_analysis["parameters"][
            "manual_breakpoints"
        ]
    return build_candidate_evidence(
        task_id="task-cross-1",
        dataset_id="dataset-cross-1",
        dataset_content_hash=dataset_hash,
        workspace_revision=3,
        workspace_generation=2,
        semantic_mapping_hash=HASH_B,
        generation_parameters=generation_parameters,
        seed=0,
        budget=100_000,
        truncated=False,
        analysis=resolved_analysis,
        metrics=[
            {"metric_name": "univariate.iv", "dimension": "count", "status": "observed", "value": 0.2},
            {"metric_name": "univariate.iv", "dimension": "loan_amount", "status": "unavailable", "value": None},
            {"metric_name": "univariate.iv", "dimension": "overdue_amount", "status": "unavailable", "value": None},
        ],
        source_refs=["dataset:dataset-cross-1"],
        producer_version=producer_version
        or (
            "strategy.univariate-candidate/2"
            if method == "manual"
            else "strategy.univariate-candidate/1"
        ),
    )


def _method(evidence: dict, feature: str, method: str = "equal_width") -> dict:
    feature_row = next(row for row in evidence["analysis"]["features"] if row["feature"] == feature)
    return next(row for row in feature_row["methods"] if row["method"] == method)


def _sample_identity(evidence: dict) -> dict:
    return {
        **evidence["identity"],
        "sample_context_hash": sample_context_hash_from_candidate_evidence(evidence),
        "target_col": evidence["analysis"]["target"],
        "row_count": evidence["analysis"]["row_count"],
    }


def _available_amount(series: pd.Series) -> dict:
    covered = series.notna()
    return {
        "status": "available",
        "covered_count": int(covered.sum()),
        "value": float(series[covered].sum()),
    }


def _measurement(
    evidence: dict,
    frame: pd.DataFrame | None = None,
    *,
    row_method: str = "equal_width",
    column_method: str = "equal_width",
) -> dict:
    source = _frame() if frame is None else frame
    row_bins = _method(evidence, "age", row_method)["bins"]
    column_bins = _method(evidence, "score", column_method)["bins"]
    cells = []
    for row_bin in row_bins:
        row_mask = evaluate_expression_frame(source, row_bin["condition"])
        for column_bin in column_bins:
            mask = row_mask & evaluate_expression_frame(source, column_bin["condition"])
            selected = source.loc[mask]
            paired_mask = selected["loan"].notna() & selected["overdue"].notna()
            paired = selected.loc[paired_mask]
            cells.append(
                {
                    "row_source_bin_id": row_bin["id"],
                    "column_source_bin_id": column_bin["id"],
                    "count": len(selected),
                    "good": int((selected["bad"] == 0).sum()),
                    "bad": int((selected["bad"] == 1).sum()),
                    "amounts": {
                        "loan_amount": _available_amount(selected["loan"]),
                        "overdue_amount": _available_amount(selected["overdue"]),
                        "paired": {
                            "status": "available",
                            "covered_count": len(paired),
                            "loan_value": float(paired["loan"].sum()),
                            "overdue_value": float(paired["overdue"].sum()),
                        },
                    },
                }
            )
    return {
        "schema_version": CROSS_MATRIX_MEASUREMENT_SCHEMA_VERSION,
        "sample_context_hash": sample_context_hash_from_candidate_evidence(evidence),
        "population_count": len(source),
        "good": int((source["bad"] == 0).sum()),
        "bad": int((source["bad"] == 1).sum()),
        "cells": cells,
    }


def _build(
    *,
    evidence: dict | None = None,
    measurement: dict | None = None,
    budget: int = 100,
    row_method: str = "equal_width",
    column_method: str = "equal_width",
) -> dict:
    parent = _evidence() if evidence is None else evidence
    measured = (
        _measurement(
            parent,
            row_method=row_method,
            column_method=column_method,
        )
        if measurement is None
        else measurement
    )
    return build_cross_matrix_candidate_asset(
        parent,
        row_axis={"feature": "age", "method": row_method},
        column_axis={"feature": "score", "method": column_method},
        sample_identity=_sample_identity(parent),
        measurement=measured,
        budget=budget,
    )


def _manual_evidence() -> dict:
    return _evidence(
        _analysis(
            method="manual",
            manual_breakpoints={
                "age": [30.0, 40.0, 50.0],
                "score": [400.0, 500.0, 600.0],
            },
        )
    )


def test_manual_axes_are_v2_and_freeze_cutpoints_with_exact_parent_lineage() -> None:
    evidence = _manual_evidence()
    asset = _build(
        evidence=evidence,
        row_method="manual",
        column_method="manual",
    )

    assert asset["schema_version"] == CROSS_MATRIX_CANDIDATE_ASSET_V2_SCHEMA_VERSION
    assert asset["producer_version"] == "strategy.cross-matrix-candidate-asset/2"
    assert asset["parent"]["analysis_schema_version"] == (
        "univariate-analysis-result.v2"
    )
    assert asset["parent"]["evidence_hash"] == evidence["evidence_hash"]
    assert asset["axes"][0]["manual_breakpoints"] == [30.0, 40.0, 50.0]
    assert asset["axes"][1]["manual_breakpoints"] == [400.0, 500.0, 600.0]
    assert all(
        axis["parent_evidence_hash"] == evidence["evidence_hash"]
        for axis in asset["axes"]
    )
    assert rebuild_cross_matrix_candidate_asset(asset, evidence) == asset

    tampered = deepcopy(asset)
    tampered["axes"][0]["parent_evidence_hash"] = HASH_C
    with pytest.raises(
        CrossMatrixCandidateAssetError,
        match="axis parent evidence hash",
    ):
        validate_cross_matrix_candidate_asset(tampered)


@pytest.mark.parametrize(
    ("producer_version", "generation_schema_version"),
    [
        ("strategy.univariate-candidate/1", "univariate-analysis-result.v2"),
        ("strategy.univariate-candidate/2", "univariate-analysis-result.v1"),
    ],
)
def test_manual_axis_rejects_self_authenticated_parent_version_drift(
    producer_version: str,
    generation_schema_version: str,
) -> None:
    evidence = _evidence(
        _analysis(
            method="manual",
            manual_breakpoints={
                "age": [30.0, 40.0, 50.0],
                "score": [400.0, 500.0, 600.0],
            },
        ),
        producer_version=producer_version,
        generation_analysis_schema_version=generation_schema_version,
    )

    with pytest.raises(
        CrossMatrixCandidateAssetError,
        match="analysis schema and producer versions do not match",
    ):
        _build(
            evidence=evidence,
            row_method="manual",
            column_method="manual",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "manual_breakpoints"),
        ("unsorted", "strictly increasing"),
        ("bin_drift", "manual_breakpoints do not match"),
    ],
)
def test_manual_axis_rejects_self_authenticated_parent_cutpoint_drift(
    mutation: str,
    message: str,
) -> None:
    analysis = _analysis(
        method="manual",
        manual_breakpoints={
            "age": [30.0, 40.0, 50.0],
            "score": [400.0, 500.0, 600.0],
        },
    )
    age_method = next(
        row
        for row in analysis["features"][0]["methods"]
        if row["method"] == "manual"
    )
    if mutation == "missing":
        del age_method["manual_breakpoints"]
    elif mutation == "unsorted":
        age_method["manual_breakpoints"] = [40.0, 30.0, 50.0]
    else:
        age_method["manual_breakpoints"] = [25.0, 40.0, 50.0]
        analysis["parameters"]["manual_breakpoints"]["age"] = [
            25.0,
            40.0,
            50.0,
        ]
    evidence = _evidence(analysis)

    with pytest.raises(CrossMatrixCandidateAssetError, match=message):
        _build(
            evidence=evidence,
            row_method="manual",
            column_method="manual",
        )


def test_builds_stable_complete_self_authenticating_cartesian_asset() -> None:
    evidence = _evidence()
    measurement = _measurement(evidence)
    first = _build(evidence=evidence, measurement=measurement)
    reordered = deepcopy(measurement)
    reordered["cells"].reverse()
    second = _build(evidence=evidence, measurement=reordered)

    assert first == second
    assert first["schema_version"] == CROSS_MATRIX_CANDIDATE_ASSET_SCHEMA_VERSION
    assert first["lifecycle"] == {
        "candidate_stage": "development",
        "observation_stage": "backtested",
        "validation_status": "unvalidated",
    }
    assert first["parent"]["candidate_id"] == evidence["candidate_id"]
    assert first["parent"]["evidence_hash"] == evidence["evidence_hash"]
    assert first["sample_identity"] == _sample_identity(evidence)
    assert first["budget"] == {
        "unit": "matrix_cells",
        "limit": 100,
        "required": 16,
        "truncated": False,
    }
    assert first["matrix"]["cell_count"] == 16
    assert len(first["matrix"]["cells"]) == 16
    assert sum(cell["effect"]["count"] for cell in first["matrix"]["cells"]) == 12
    assert all(ID_RE.fullmatch(first[field]) for field in ("asset_id",))
    assert ID_RE.fullmatch(first["candidate_evidence"]["candidate_id"])
    assert all(ID_RE.fullmatch(axis["axis_id"]) for axis in first["axes"])
    assert all(ID_RE.fullmatch(item["bin_id"]) for axis in first["axes"] for item in axis["bins"])
    assert all(ID_RE.fullmatch(cell["cell_id"]) for cell in first["matrix"]["cells"])
    assert all(cell["rule"]["condition"]["op"] == "and" for cell in first["matrix"]["cells"])
    assert json.loads(canonical_cross_matrix_candidate_asset_json(first)) == first
    assert validate_cross_matrix_candidate_asset(first) == first
    assert rebuild_cross_matrix_candidate_asset(first, evidence) == first


def test_empty_cells_remain_explicit_without_zero_division_claims() -> None:
    asset = _build()
    empty = next(cell for cell in asset["matrix"]["cells"] if cell["effect"]["count"] == 0)
    effect = empty["effect"]
    assert effect["good"] == effect["bad"] == 0
    assert effect["share"] == 0
    assert effect["bad_rate"] is None
    assert effect["lift"] is None
    assert isinstance(effect["woe"], float)
    assert isinstance(effect["iv_contribution"], float)
    assert effect["amount_metrics"]["loan_amount"]["coverage_rate"] is None
    assert effect["amount_metrics"]["overdue_rate"]["status"] == "not_applicable"
    assert effect["amount_metrics"]["overdue_rate"]["reason"] == "no_observations"

    expected_woe, expected_iv = _smoothed_woe_iv(
        effect["bad"],
        effect["good"],
        asset["summary"]["bad"],
        asset["summary"]["good"],
        asset["matrix"]["cell_count"],
        smoothing=asset["parent"]["smoothing"],
    )
    assert effect["woe"] == expected_woe
    assert effect["iv_contribution"] == expected_iv


def test_budget_is_fail_closed_and_never_truncates_cells() -> None:
    with pytest.raises(CrossMatrixCandidateAssetError, match="insufficient for 16 complete cells"):
        _build(budget=15)


@pytest.mark.parametrize("mutation, message", [
    (lambda value: value["cells"].pop(), "complete Cartesian"),
    (lambda value: value["cells"].append(deepcopy(value["cells"][0])), "pairs must be unique"),
    (lambda value: value["cells"][0].__setitem__("count", value["cells"][0]["count"] + 1), r"good \+ bad"),
    (lambda value: value.__setitem__("action", "reject"), "non-exact fields"),
])
def test_measurement_contract_rejects_missing_duplicate_inconsistent_and_extra_fields(mutation, message: str) -> None:
    evidence = _evidence()
    measured = _measurement(evidence)
    mutation(measured)
    with pytest.raises(CrossMatrixCandidateAssetError, match=message):
        _build(evidence=evidence, measurement=measured)


def test_row_column_marginals_and_amount_primary_values_must_reconcile() -> None:
    evidence = _evidence()
    measured = _measurement(evidence)
    populated = next(cell for cell in measured["cells"] if cell["count"] > 0)
    populated["amounts"]["loan_amount"]["value"] += 1
    with pytest.raises(CrossMatrixCandidateAssetError, match="loan_amount primary facts"):
        _build(evidence=evidence, measurement=measured)

    measured = _measurement(evidence)
    left, right = [cell for cell in measured["cells"] if cell["count"] > 0][:2]
    left["good"] -= 1
    left["bad"] += 1
    right["good"] += 1
    right["bad"] -= 1
    with pytest.raises(CrossMatrixCandidateAssetError, match="marginal changed"):
        _build(evidence=evidence, measurement=measured)


def test_parent_sample_axis_binding_is_exact_and_features_must_differ() -> None:
    evidence = _evidence()
    sample = _sample_identity(evidence)
    sample["dataset_content_hash"] = "d" * 64
    with pytest.raises(CrossMatrixCandidateAssetError, match="exactly match parent identity"):
        build_cross_matrix_candidate_asset(
            evidence,
            row_axis={"feature": "age", "method": "equal_width"},
            column_axis={"feature": "score", "method": "equal_width"},
            sample_identity=sample,
            measurement=_measurement(evidence),
            budget=100,
        )

    with pytest.raises(CrossMatrixCandidateAssetError, match="different features"):
        build_cross_matrix_candidate_asset(
            evidence,
            row_axis={"feature": "age", "method": "equal_width"},
            column_axis={"feature": "age", "method": "equal_width"},
            sample_identity=_sample_identity(evidence),
            measurement=_measurement(evidence),
            budget=100,
        )

    with pytest.raises(CrossMatrixCandidateAssetError, match="uniquely available"):
        build_cross_matrix_candidate_asset(
            evidence,
            row_axis={"feature": "age", "method": "tree"},
            column_axis={"feature": "score", "method": "equal_width"},
            sample_identity=_sample_identity(evidence),
            measurement=_measurement(evidence),
            budget=100,
        )

    sample = _sample_identity(evidence)
    sample["sample_context_hash"] = HASH_C
    measured = _measurement(evidence)
    measured["sample_context_hash"] = HASH_C
    with pytest.raises(CrossMatrixCandidateAssetError, match="exact parent CandidateEvidence"):
        build_cross_matrix_candidate_asset(
            evidence,
            row_axis={"feature": "age", "method": "equal_width"},
            column_axis={"feature": "score", "method": "equal_width"},
            sample_identity=sample,
            measurement=measured,
            budget=100,
        )


def test_parent_axis_condition_must_be_canonical_and_reference_only_its_feature() -> None:
    analysis = _analysis()
    analysis["features"][0]["methods"][0]["bins"][0]["condition"] = {
        "op": "compare", "field": "score", "operator": "<", "value": 1,
        "missing": "no_match",
    }
    evidence = _evidence(analysis)
    with pytest.raises(CrossMatrixCandidateAssetError, match="reference only 'age'"):
        _build(evidence=evidence, measurement=_measurement(_evidence()))


def test_validation_rejects_derived_metric_lifecycle_and_reserved_claim_tampering() -> None:
    asset = _build()
    tampered = deepcopy(asset)
    tampered["matrix"]["cells"][0]["effect"]["share"] = 0.99
    with pytest.raises(CrossMatrixCandidateAssetError, match="not deterministic"):
        validate_cross_matrix_candidate_asset(tampered)

    tampered = deepcopy(asset)
    tampered["lifecycle"]["validation_status"] = "validated"
    with pytest.raises(CrossMatrixCandidateAssetError, match="development/backtested/unvalidated"):
        validate_cross_matrix_candidate_asset(tampered)

    tampered = deepcopy(asset)
    tampered["adopted"] = True
    with pytest.raises(CrossMatrixCandidateAssetError, match="non-exact fields"):
        validate_cross_matrix_candidate_asset(tampered)

    tampered = deepcopy(asset)
    paired = tampered["axes"][0]["bins"][0]["amount_evidence"]["paired"]
    paired.update(
        {"status": "available", "covered_count": 0, "value": 0.0, "reason": None}
    )
    with pytest.raises(CrossMatrixCandidateAssetError, match="covered_count"):
        validate_cross_matrix_candidate_asset(tampered)

    canonical = canonical_cross_matrix_candidate_asset_json(asset).casefold()
    assert '"action"' not in canonical
    assert '"pool"' not in canonical
    assert '"adopted"' not in canonical
    assert '"deployed"' not in canonical
    assert '"validated"' not in canonical


def test_parse_rejects_duplicate_keys_and_rebuild_rejects_different_parent() -> None:
    asset = _build()
    with pytest.raises(CrossMatrixCandidateAssetError, match="duplicate key: asset_id"):
        parse_cross_matrix_candidate_asset_json('{"asset_id":"one","asset_id":"two"}')

    other_parent = _evidence(dataset_hash="d" * 64)
    with pytest.raises(CrossMatrixCandidateAssetError, match="exact parent"):
        rebuild_cross_matrix_candidate_asset(asset, other_parent)

    confused = deepcopy(asset)
    confused["parent"]["candidate_id"] = asset["asset_id"]
    with pytest.raises(CrossMatrixCandidateAssetError, match="invalid format"):
        validate_cross_matrix_candidate_asset(confused)


def test_configured_but_all_null_amounts_reconcile_as_zero_covered_not_unconfigured() -> None:
    frame = _frame()
    frame["loan"] = float("nan")
    frame["overdue"] = float("nan")
    analysis = _analysis(frame)
    evidence = _evidence(analysis)
    asset = _build(evidence=evidence, measurement=_measurement(evidence, frame))

    assert asset["summary"]["amount_metrics"]["loan_amount"] == {
        "status": "available",
        "covered_count": 0,
        "coverage_rate": 0.0,
        "value": 0.0,
        "reason": None,
    }
    assert asset["summary"]["amount_metrics"]["overdue_rate"] == {
        "status": "not_applicable",
        "covered_count": 0,
        "coverage_rate": 0.0,
        "value": None,
        "reason": "no_paired_observations",
    }


def test_unconfigured_optional_amounts_build_with_explicit_unavailable_status() -> None:
    frame = _frame()
    analysis = analyze_univariate(
        frame,
        features=["age", "score"],
        target="bad",
        methods=["equal_width"],
        bin_count=4,
        min_bin_pct=0,
    )
    evidence = _evidence(analysis)
    measured = _measurement(evidence, frame)
    unavailable = {
        "loan_amount": {
            "status": "unavailable", "covered_count": None, "value": None
        },
        "overdue_amount": {
            "status": "unavailable", "covered_count": None, "value": None
        },
        "paired": {
            "status": "unavailable", "covered_count": None,
            "loan_value": None, "overdue_value": None,
        },
    }
    for cell in measured["cells"]:
        cell["amounts"] = deepcopy(unavailable)

    asset = _build(evidence=evidence, measurement=measured)
    assert asset["summary"]["amount_metrics"]["loan_amount"] == {
        "status": "unavailable",
        "covered_count": None,
        "coverage_rate": None,
        "value": None,
        "reason": "column_unavailable",
    }
    assert asset["summary"]["amount_metrics"]["overdue_rate"]["status"] == (
        "unavailable"
    )


def test_zero_covered_primary_amounts_must_have_zero_values() -> None:
    evidence = _evidence()
    measured = _measurement(evidence)
    empty = next(cell for cell in measured["cells"] if cell["count"] == 0)
    empty["amounts"]["loan_amount"]["value"] = 1.0
    with pytest.raises(CrossMatrixCandidateAssetError, match="must be zero"):
        _build(evidence=evidence, measurement=measured)

    measured = _measurement(evidence)
    empty = next(cell for cell in measured["cells"] if cell["count"] == 0)
    empty["amounts"]["paired"]["overdue_value"] = 1.0
    with pytest.raises(CrossMatrixCandidateAssetError, match="must be zero"):
        _build(evidence=evidence, measurement=measured)
