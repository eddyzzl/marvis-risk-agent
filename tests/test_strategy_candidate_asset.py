from __future__ import annotations

from copy import deepcopy
import json

import pandas as pd
import pytest

from marvis.feature.univariate import analyze_univariate
from marvis.packs.strategy.candidate_asset import (
    CANDIDATE_ASSET_SCHEMA_VERSION,
    CandidateAssetError,
    canonical_candidate_asset_json,
    refine_univariate_candidate,
    validate_candidate_asset,
)
from marvis.packs.strategy.candidate_evidence import (
    MetricObservation,
    build_candidate_evidence,
)
from marvis.packs.strategy.evaluator import evaluate_expression


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _source() -> dict[str, str]:
    return {
        "artifact_id": "artifact-univariate-json",
        "kind": "strategy_candidate_json",
        "content_hash": HASH_C,
    }


def _evidence(analysis: dict) -> dict:
    return build_candidate_evidence(
        task_id="task-1",
        dataset_id="dataset-1",
        dataset_content_hash=HASH_A,
        workspace_revision=3,
        workspace_generation=2,
        semantic_mapping_hash=HASH_B,
        generation_parameters={"features": ["x"], "bin_count": 4},
        seed=0,
        budget=100_000,
        truncated=False,
        analysis=analysis,
        metrics=[
            MetricObservation("parent.iv", "count", "observed", 0.2),
            MetricObservation("parent.iv", "loan_amount", "unavailable", None),
            MetricObservation("parent.iv", "overdue_amount", "unavailable", None),
        ],
        source_refs=["dataset:dataset-1", "analysis:univariate-1"],
        producer_version="strategy.univariate-candidate/1",
    )


def _numeric_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": list(range(1, 13)),
            "bad": [0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1],
            "loan": [100.0] * 12,
            "overdue": [0, 0, 0, 20, 0, 30, 0, 40, 50, 60, 70, 80],
        }
    )


def _numeric_parent(frame: pd.DataFrame) -> dict:
    analysis = analyze_univariate(
        frame,
        features=["x"],
        target="bad",
        methods=["equal_width"],
        bin_count=4,
        loan_amount="loan",
        overdue_amount="overdue",
    )
    return _evidence(analysis)


def _method(evidence: dict, feature: str = "x", method: str = "equal_width") -> dict:
    feature_row = next(
        row for row in evidence["analysis"]["features"] if row["feature"] == feature
    )
    return next(row for row in feature_row["methods"] if row["method"] == method)


def test_numeric_refinement_replays_parent_and_recomputes_stable_asset() -> None:
    frame = _numeric_frame()
    evidence = _numeric_parent(frame)
    source_bins = _method(evidence)["bins"]
    middle = [source_bins[1]["id"], source_bins[2]["id"]]

    first = refine_univariate_candidate(
        evidence,
        frame,
        source_evidence=_source(),
        feature="x",
        method="equal_width",
        merge_groups=[list(reversed(middle))],
        selection={"source_bin_ids": list(reversed(middle))},
        selection_reason="  中间风险段人工复核  ",
    )
    repeated = refine_univariate_candidate(
        evidence,
        frame,
        source_evidence=_source(),
        feature="x",
        method="equal_width",
        merge_groups=[middle],
        selection={"source_bin_ids": middle},
        selection_reason="中间风险段人工复核",
    )

    assert first == repeated
    assert first["schema_version"] == CANDIDATE_ASSET_SCHEMA_VERSION
    assert first["effect_stage"] == "development"
    assert first["validation_status"] == "unvalidated"
    assert first["parent"] == {
        "candidate_id": evidence["candidate_id"],
        "evidence_hash": evidence["evidence_hash"],
        "source_evidence": _source(),
    }
    assert first["selection_reason"] == "中间风险段人工复核"
    assert first["refinement"]["edited_bin_count"] == len(source_bins) - 1
    assert first["refinement"]["merge_groups"] == [middle]
    assert first["refinement"]["metrics"]["iv"] == pytest.approx(
        sum(row["iv_contribution"] for row in first["refinement"]["bins"])
    )
    merged = next(
        row for row in first["refinement"]["bins"] if row["source_bin_ids"] == middle
    )
    assert merged["condition"]["op"] == "or"
    assert merged["condition"]["args"] == [
        source_bins[1]["condition"],
        source_bins[2]["condition"],
    ]
    assert first["rule"]["condition"] == {
        "op": "or",
        "args": [merged["condition"]],
    }
    assert first["effect"]["selected_count"] == (
        source_bins[1]["count"] + source_bins[2]["count"]
    )
    assert json.loads(canonical_candidate_asset_json(first)) == first

    dimensions: dict[str, set[str]] = {}
    for observation in first["metrics"]:
        dimensions.setdefault(observation["metric_name"], set()).add(
            observation["dimension"]
        )
    assert all(
        value == {"count", "loan_amount", "overdue_amount"}
        for value in dimensions.values()
    )


def test_numeric_ordinary_bins_only_merge_when_adjacent() -> None:
    frame = _numeric_frame()
    evidence = _numeric_parent(frame)
    bins = _method(evidence)["bins"]

    with pytest.raises(CandidateAssetError, match="adjacent"):
        refine_univariate_candidate(
            evidence,
            frame,
            source_evidence=_source(),
            feature="x",
            method="equal_width",
            merge_groups=[[bins[0]["id"], bins[2]["id"]]],
            selection={"source_bin_ids": [bins[0]["id"]]},
        )


def test_category_merge_preserves_strict_scalar_types_and_fixed_source_order() -> None:
    frame = pd.DataFrame(
        {
            "segment": [1, "1", "A", 1, "1", "A", 2, "2"],
            "bad": [0, 1, 0, 1, 1, 0, 1, 0],
        }
    )
    analysis = analyze_univariate(
        frame,
        features=["segment"],
        target="bad",
        methods=["categorical"],
        feature_types={"segment": "categorical"},
        bin_count=3,
    )
    evidence = _evidence(analysis)
    bins = _method(evidence, "segment", "categorical")["bins"]
    int_bin = next(row for row in bins if row.get("value") == 1)
    str_bin = next(row for row in bins if row.get("value") == "1")
    merge = [int_bin["id"], str_bin["id"]]

    asset = refine_univariate_candidate(
        evidence,
        frame,
        source_evidence=_source(),
        feature="segment",
        method="categorical",
        merge_groups=[list(reversed(merge))],
        selection={"source_bin_ids": list(reversed(merge))},
    )

    merged = next(
        row for row in asset["refinement"]["bins"] if len(row["source_bin_ids"]) == 2
    )
    assert all(arg.get("coercion") == "strict" for arg in merged["condition"]["args"])
    assert evaluate_expression({"segment": 1}, merged["condition"])
    assert evaluate_expression({"segment": "1"}, merged["condition"])
    assert not evaluate_expression({"segment": True}, merged["condition"])


def test_missing_and_sentinel_cannot_merge_with_ordinary_bins() -> None:
    frame = pd.DataFrame(
        {
            "x": [-999, 1, 2, 3, 4, 5, None, 6],
            "bad": [1, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    evidence = _evidence(
        analyze_univariate(
            frame,
            features=["x"],
            target="bad",
            methods=["equal_width"],
            bin_count=3,
            sentinel_values={"x": [-999]},
        )
    )
    bins = _method(evidence)["bins"]
    ordinary = next(row for row in bins if row["kind"] == "numeric_interval")
    sentinel = next(row for row in bins if row["kind"] == "sentinel")
    missing = next(row for row in bins if row["kind"] == "missing")

    for special in (sentinel, missing):
        with pytest.raises(CandidateAssetError, match="cannot be merged"):
            refine_univariate_candidate(
                evidence,
                frame,
                source_evidence=_source(),
                feature="x",
                method="equal_width",
                merge_groups=[[ordinary["id"], special["id"]]],
                selection={"source_bin_ids": [ordinary["id"], special["id"]]},
            )


def test_parent_conditions_must_replay_exact_counts_and_partition() -> None:
    frame = _numeric_frame()
    evidence = _numeric_parent(frame)
    analysis = deepcopy(evidence["analysis"])
    analysis["features"][0]["methods"][0]["bins"][0]["count"] += 1
    forged_parent = _evidence(analysis)
    first_id = _method(forged_parent)["bins"][0]["id"]

    with pytest.raises(CandidateAssetError, match="count does not replay"):
        refine_univariate_candidate(
            forged_parent,
            frame,
            source_evidence=_source(),
            feature="x",
            method="equal_width",
            merge_groups=[],
            selection={"source_bin_ids": [first_id]},
        )


def test_risk_threshold_selection_is_resolved_from_recomputed_bad_rates() -> None:
    frame = _numeric_frame()
    evidence = _numeric_parent(frame)

    asset = refine_univariate_candidate(
        evidence,
        frame,
        source_evidence=_source(),
        feature="x",
        method="equal_width",
        merge_groups=[],
        selection={"risk_threshold": {"operator": ">=", "value": 0.5}},
    )

    selected = set(asset["selection"]["selected_bin_ids"])
    assert selected == {
        row["bin_id"]
        for row in asset["refinement"]["bins"]
        if row["bad_rate"] is not None and row["bad_rate"] >= 0.5
    }
    assert asset["selection"]["risk_threshold"] == {
        "metric": "bad_rate",
        "operator": ">=",
        "value": 0.5,
    }


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update({"unknown": True}), "unknown"),
        (lambda value: value.update({"validation_status": "validated"}), "unvalidated"),
        (lambda value: value.update({"effect_stage": "validation"}), "development"),
        (lambda value: value.update({"asset_hash": "0" * 64}), "asset_hash"),
        (
            lambda value: value["rule"].update(
                {"rule_id": "candidate-rule-" + "0" * 32}
            ),
            "rule_id",
        ),
    ],
)
def test_strict_validator_rejects_unknown_fields_and_forged_claims(
    mutation, match: str
) -> None:
    frame = _numeric_frame()
    evidence = _numeric_parent(frame)
    source_bin_id = _method(evidence)["bins"][0]["id"]
    asset = refine_univariate_candidate(
        evidence,
        frame,
        source_evidence=_source(),
        feature="x",
        method="equal_width",
        merge_groups=[],
        selection={"source_bin_ids": [source_bin_id]},
    )
    forged = deepcopy(asset)
    mutation(forged)

    with pytest.raises(CandidateAssetError, match=match):
        validate_candidate_asset(forged)


def test_validator_cross_checks_recomputed_bins_effect_and_metric_triplets() -> None:
    frame = _numeric_frame()
    evidence = _numeric_parent(frame)
    source_bin_id = _method(evidence)["bins"][0]["id"]
    asset = refine_univariate_candidate(
        evidence,
        frame,
        source_evidence=_source(),
        feature="x",
        method="equal_width",
        merge_groups=[],
        selection={"source_bin_ids": [source_bin_id]},
    )

    forged_bin = deepcopy(asset)
    forged_bin["refinement"]["bins"][0]["iv_contribution"] += 0.01
    with pytest.raises(CandidateAssetError, match="WOE/IV"):
        validate_candidate_asset(forged_bin)

    forged_effect = deepcopy(asset)
    forged_effect["effect"]["selected_share"] += 0.01
    with pytest.raises(CandidateAssetError, match="effect_id"):
        validate_candidate_asset(forged_effect)

    forged_metric = deepcopy(asset)
    observation = next(
        item
        for item in forged_metric["metrics"]
        if item["metric_name"] == "rule.bad_rate" and item["dimension"] == "count"
    )
    observation["value"] += 0.01
    with pytest.raises(CandidateAssetError, match="canonical refinement"):
        validate_candidate_asset(forged_metric)
