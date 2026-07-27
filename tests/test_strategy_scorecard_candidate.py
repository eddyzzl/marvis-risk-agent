from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

import numpy as np
import pytest

from marvis.packs.strategy.scorecard_candidate import (
    SCORECARD_BAND_ASSET_ARTIFACT_KIND,
    SCORECARD_BAND_ASSET_ARTIFACT_SCHEMA_VERSION,
    SCORECARD_BAND_ASSET_ORIGIN_TOOL,
    SCORECARD_BAND_ASSET_SCHEMA_VERSION,
    SCORECARD_CUTOFF_SELECTION_ARTIFACT_KIND,
    SCORECARD_CUTOFF_SELECTION_ARTIFACT_SCHEMA_VERSION,
    SCORECARD_CUTOFF_SELECTION_ORIGIN_TOOL,
    ScorecardCandidateError,
    build_scorecard_band_asset,
    build_scorecard_cutoff_selection,
    canonical_scorecard_band_asset_json,
    canonical_scorecard_cutoff_selection_json,
    scorecard_cutoff_selection_to_verified_candidate_fragment,
    validate_scorecard_cutoff_selection,
    validate_scorecard_band_asset,
)
from marvis.packs.strategy.candidate_fragment import (
    validate_verified_candidate_fragment,
)
from marvis.packs.strategy.pool_requirement_resolver import (
    model_score_virtual_field,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _sample_design_ref() -> dict[str, str]:
    return {
        "membership_artifact_id": _hash("membership-artifact"),
        "expected_membership_artifact_content_hash": _hash("membership-file"),
        "bundle_artifact_id": _hash("bundle-artifact"),
        "expected_bundle_artifact_content_hash": _hash("bundle-file"),
        "expected_bundle_id": "strategy-sample-design-bundle-a",
        "expected_sample_design_id": "strategy-sample-design-a",
        "expected_sample_design_content_hash": _hash("sample-design"),
    }


def _identity() -> dict[str, object]:
    return {
        "task_id": "task-scorecard",
        "dataset_id": "dataset-scorecard",
        "dataset_content_hash": _hash("dataset"),
        "workspace_revision": 7,
        "workspace_generation": 3,
        "semantic_mapping_hash": _hash("semantics"),
        "sample_context_hash": _hash("sample-context"),
    }


def _evidence_ref(prefix: str) -> dict[str, str]:
    return {
        "artifact_id": _hash(f"{prefix}-artifact"),
        "artifact_content_hash": _hash(f"{prefix}-artifact-content"),
        "evidence_id": f"{prefix}-evidence-a",
        "evidence_content_hash": _hash(f"{prefix}-evidence-content"),
    }


def _score_vector_ref() -> dict[str, str]:
    return {
        "artifact_id": _hash("score-vector-artifact"),
        "artifact_content_hash": _hash("score-vector-file"),
    }


def _scale() -> dict[str, float | int]:
    factor = 50.0 / math.log(2.0)
    return {
        "base_score": 600,
        "pdo": 50,
        "base_odds": 50.0,
        "factor": factor,
        "offset": 600.0 - factor * math.log(50.0),
    }


def _scorecard_table() -> list[dict[str, object]]:
    scale = _scale()
    return [
        {
            "feature": "__base__",
            "bin_index": -999,
            "bin_label": "base_points",
            "lower": None,
            "upper": None,
            "count": None,
            "bad_count": None,
            "good_count": None,
            "bad_rate": None,
            "woe": None,
            "iv_contribution": None,
            "coefficient": None,
            "monotonic_direction": None,
            "points": 320.0,
            **scale,
        },
        {
            "feature": "income",
            "bin_index": 0,
            "bin_label": "[-inf, 10)",
            "lower": None,
            "upper": 10.0,
            "count": 3,
            "bad_count": 2,
            "good_count": 1,
            "bad_rate": 2.0 / 3.0,
            "woe": 0.4,
            "iv_contribution": 0.08,
            "coefficient": 0.5,
            "monotonic_direction": "increasing",
            "points": -14.0,
        },
        {
            "feature": "income",
            "bin_index": 1,
            "bin_label": "[10, inf)",
            "lower": 10.0,
            "upper": None,
            "count": 3,
            "bad_count": 0,
            "good_count": 3,
            "bad_rate": 0.0,
            "woe": -0.4,
            "iv_contribution": 0.08,
            "coefficient": 0.5,
            "monotonic_direction": "increasing",
            "points": 14.0,
        },
    ]


def _score_bins() -> list[dict[str, object]]:
    return [
        {
            "ordinal": 0,
            "bin_id": "score-bin-00",
            "lower_bound": None,
            "upper_bound": 0.3,
            "lower_inclusive": False,
            "upper_inclusive": False,
        },
        {
            "ordinal": 1,
            "bin_id": "score-bin-01",
            "lower_bound": 0.3,
            "upper_bound": 0.7,
            "lower_inclusive": True,
            "upper_inclusive": False,
        },
        {
            "ordinal": 2,
            "bin_id": "score-bin-02",
            "lower_bound": 0.7,
            "upper_bound": None,
            "lower_inclusive": True,
            "upper_inclusive": False,
        },
    ]


def _build(**overrides: object) -> dict:
    inputs: dict[str, object] = {
        "identity": _identity(),
        "sample_design_ref": _sample_design_ref(),
        "training_evidence_ref": _evidence_ref("training"),
        "score_evidence_ref": _evidence_ref("score"),
        "score_vector_ref": _score_vector_ref(),
        "score_product": "raw_native_uncalibrated_bad_probability",
        "score_direction": "higher_is_riskier",
        "points_direction": "higher_is_better",
        "scorecard_scale": _scale(),
        "scorecard_table": _scorecard_table(),
        "raw_pd": np.asarray([0.1, 0.2, 0.4, 0.6, 0.8, 0.9]),
        "risk_development_mask": np.ones(6, dtype=np.bool_),
        "labels": np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, np.nan]),
        "score_bins": _score_bins(),
    }
    inputs.update(overrides)
    return build_scorecard_band_asset(**inputs)


def test_build_scorecard_band_asset_recomputes_bins_cutoffs_and_metrics() -> None:
    asset = _build()

    assert asset["schema_version"] == SCORECARD_BAND_ASSET_SCHEMA_VERSION
    assert asset["performance"] == {"auc": 1.0, "ks": 1.0}
    assert [
        (
            item["count"],
            item["labeled_count"],
            item["bad_count"],
            item["bad_rate"],
            item["average_pd"],
        )
        for item in asset["bands"]
    ] == [
        (2, 2, 0, 0.0, pytest.approx(0.15)),
        (2, 2, 1, 0.5, pytest.approx(0.5)),
        (2, 1, 1, 1.0, pytest.approx(0.85)),
    ]
    assert [item["execution_pd"] for item in asset["cutoffs"]] == [0.3, 0.7]
    assert all(item["mask_equivalence"] is True for item in asset["cutoffs"])
    assert asset["cutoffs"][0]["lower_risk"] == {
        "count": 2,
        "labeled_count": 2,
        "bad_count": 0,
        "bad_rate": 0.0,
    }
    assert asset["cutoffs"][0]["higher_risk"] == {
        "count": 4,
        "labeled_count": 3,
        "bad_count": 2,
        "bad_rate": pytest.approx(2.0 / 3.0),
    }
    assert math.isfinite(asset["cutoffs"][0]["display_points"])
    assert asset["governance"]["best_cutoff_recommended"] is False
    assert validate_scorecard_band_asset(asset) == asset
    assert json.loads(canonical_scorecard_band_asset_json(asset)) == asset


def _asset_binding(asset: dict) -> dict[str, object]:
    canonical = canonical_scorecard_band_asset_json(asset).encode("utf-8")
    return {
        "artifact_id": _hash("band-artifact"),
        "task_id": asset["identity"]["task_id"],
        "kind": SCORECARD_BAND_ASSET_ARTIFACT_KIND,
        "artifact_schema_version": SCORECARD_BAND_ASSET_ARTIFACT_SCHEMA_VERSION,
        "content_hash": hashlib.sha256(canonical).hexdigest(),
        "origin_tool": SCORECARD_BAND_ASSET_ORIGIN_TOOL,
        "canonical_bytes": canonical,
    }


def _selection_binding(selection: dict) -> dict[str, object]:
    canonical = canonical_scorecard_cutoff_selection_json(selection).encode("utf-8")
    return {
        "artifact_id": _hash("selection-artifact"),
        "task_id": "task-scorecard",
        "kind": SCORECARD_CUTOFF_SELECTION_ARTIFACT_KIND,
        "artifact_schema_version": (
            SCORECARD_CUTOFF_SELECTION_ARTIFACT_SCHEMA_VERSION
        ),
        "content_hash": hashlib.sha256(canonical).hexdigest(),
        "origin_tool": SCORECARD_CUTOFF_SELECTION_ORIGIN_TOOL,
        "canonical_bytes": canonical,
    }


def test_pointer_only_cutoff_selection_replays_one_typed_pool_fragment() -> None:
    asset = _build()
    asset_binding = _asset_binding(asset)
    cutoff = asset["cutoffs"][0]

    selection = build_scorecard_cutoff_selection(
        asset,
        source_artifact_binding=asset_binding,
        cutoff_id=cutoff["cutoff_id"],
        selection_reason="风险上限方案",
    )

    assert set(selection) == {
        "schema_version",
        "producer_version",
        "source_asset_ref",
        "cutoff_id",
        "selection_reason",
        "selection_id",
        "selection_hash",
    }
    assert set(selection["source_asset_ref"]) == {
        "artifact_id",
        "task_id",
        "kind",
        "artifact_schema_version",
        "artifact_content_hash",
        "origin_tool",
        "asset_schema_version",
        "asset_type",
        "asset_id",
        "asset_hash",
    }
    assert not ({"execution_pd", "display_points", "metrics", "condition"} & set(selection))
    assert validate_scorecard_cutoff_selection(selection) == selection
    assert json.loads(canonical_scorecard_cutoff_selection_json(selection)) == selection

    fragment = scorecard_cutoff_selection_to_verified_candidate_fragment(
        selection,
        asset,
        selection_artifact_binding=_selection_binding(selection),
        source_artifact_binding=asset_binding,
    )
    virtual_field = model_score_virtual_field(
        asset["source_refs"]["score_vector"]["artifact_id"]
    )
    assert validate_verified_candidate_fragment(fragment) == fragment
    assert fragment["artifact"] == {
        "artifact_id": _hash("selection-artifact"),
        "artifact_kind": SCORECARD_CUTOFF_SELECTION_ARTIFACT_KIND,
        "artifact_schema_version": (
            SCORECARD_CUTOFF_SELECTION_ARTIFACT_SCHEMA_VERSION
        ),
        "artifact_content_hash": _selection_binding(selection)["content_hash"],
        "origin_tool": SCORECARD_CUTOFF_SELECTION_ORIGIN_TOOL,
    }
    assert fragment["asset"] == {
        "schema_version": SCORECARD_BAND_ASSET_SCHEMA_VERSION,
        "asset_id": asset["asset_id"],
        "asset_hash": asset["asset_hash"],
        "asset_type": asset["asset_type"],
    }
    assert fragment["fragment"]["condition"] == {
        "op": "compare",
        "field": virtual_field,
        "operator": ">=",
        "value": cutoff["execution_pd"],
        "missing": "no_match",
    }
    assert fragment["fragment"]["requirements"] == [
        {
            "type": "model_score_vector.v1",
            "virtual_field": virtual_field,
            "score_product": "raw_native_uncalibrated_bad_probability",
            "score_evidence_artifact_id": asset["source_refs"]["score_evidence"][
                "artifact_id"
            ],
            "score_evidence_artifact_content_hash": asset["source_refs"][
                "score_evidence"
            ]["artifact_content_hash"],
            "score_vector_artifact_id": asset["source_refs"]["score_vector"][
                "artifact_id"
            ],
            "score_vector_artifact_content_hash": asset["source_refs"][
                "score_vector"
            ]["artifact_content_hash"],
        }
    ]
    assert fragment["evidence"]["evidence_id"] == asset["asset_id"]
    assert fragment["evidence"]["evidence_hash"] == asset["asset_hash"]


def test_build_rejects_duplicate_scorecard_table_bin_identity() -> None:
    table = _scorecard_table()
    table.append(deepcopy(table[1]))

    with pytest.raises(ScorecardCandidateError, match="scorecard_table|duplicate"):
        _build(scorecard_table=table)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"score_product": "calibrated_bad_probability"},
            "score_product|raw native",
        ),
        ({"score_direction": "higher_is_better"}, "score_direction"),
        ({"points_direction": "higher_is_riskier"}, "points_direction"),
        (
            {"raw_pd": np.asarray([0.1, 0.2, np.nan, 0.6, 0.8, 0.9])},
            "raw_pd",
        ),
        (
            {"labels": np.asarray([0.0, 0.0, 2.0, 1.0, 1.0, np.nan])},
            "labels",
        ),
        (
            {"risk_development_mask": np.ones(6, dtype=np.int64)},
            "boolean",
        ),
        ({"score_bins": _score_bins()[:1]}, "between 2 and 20"),
        (
            {
                "score_bins": [
                    _score_bins()[0],
                    {
                        **_score_bins()[1],
                        "lower_bound": 0.31,
                    },
                    _score_bins()[2],
                ]
            },
            "contiguous",
        ),
    ],
)
def test_build_fails_closed_on_invalid_score_sample_and_boundaries(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ScorecardCandidateError, match=message):
        _build(**overrides)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda asset: asset.update({"unknown": True}),
        lambda asset: asset["performance"].update({"auc": 0.5}),
        lambda asset: asset["governance"].update(
            {"best_cutoff_recommended": True}
        ),
        lambda asset: asset["bands"][0].update({"count": 3}),
        lambda asset: asset["cutoffs"][0].update({"mask_equivalence": False}),
    ],
)
def test_asset_validation_rejects_unknown_forged_or_tampered_content(
    mutation,
) -> None:
    asset = deepcopy(_build())
    mutation(asset)

    with pytest.raises(ScorecardCandidateError):
        validate_scorecard_band_asset(asset)


def test_selection_and_replay_reject_unknown_cutoff_and_counterfeit_bytes() -> None:
    asset = _build()
    asset_binding = _asset_binding(asset)
    with pytest.raises(ScorecardCandidateError, match="cutoff_id"):
        build_scorecard_cutoff_selection(
            asset,
            source_artifact_binding=asset_binding,
            cutoff_id="scorecard-cutoff-" + "0" * 32,
        )

    selection = build_scorecard_cutoff_selection(
        asset,
        source_artifact_binding=asset_binding,
        cutoff_id=asset["cutoffs"][0]["cutoff_id"],
    )
    forged = {**selection, "metrics": {}}
    with pytest.raises(ScorecardCandidateError, match="fields"):
        validate_scorecard_cutoff_selection(forged)

    counterfeit_selection_binding = {
        **_selection_binding(selection),
        "canonical_bytes": (
            _selection_binding(selection)["canonical_bytes"] + b" "
        ),
    }
    with pytest.raises(ScorecardCandidateError, match="canonical bytes"):
        scorecard_cutoff_selection_to_verified_candidate_fragment(
            selection,
            asset,
            selection_artifact_binding=counterfeit_selection_binding,
            source_artifact_binding=asset_binding,
        )

    counterfeit_source_binding = {
        **asset_binding,
        "canonical_bytes": asset_binding["canonical_bytes"] + b" ",
    }
    with pytest.raises(ScorecardCandidateError, match="canonical bytes"):
        scorecard_cutoff_selection_to_verified_candidate_fragment(
            selection,
            asset,
            selection_artifact_binding=_selection_binding(selection),
            source_artifact_binding=counterfeit_source_binding,
        )


def test_build_and_selection_are_stable_but_reason_is_audited() -> None:
    first = _build()
    second = _build()
    assert first["asset_id"] == second["asset_id"]
    assert first["asset_hash"] == second["asset_hash"]

    binding = _asset_binding(first)
    cutoff_id = first["cutoffs"][0]["cutoff_id"]
    without_reason = build_scorecard_cutoff_selection(
        first,
        source_artifact_binding=binding,
        cutoff_id=cutoff_id,
    )
    with_reason = build_scorecard_cutoff_selection(
        first,
        source_artifact_binding=binding,
        cutoff_id=cutoff_id,
        selection_reason="人工确认",
    )
    assert without_reason["selection_reason"] is None
    assert without_reason["selection_id"] != with_reason["selection_id"]
    assert without_reason["selection_hash"] != with_reason["selection_hash"]
