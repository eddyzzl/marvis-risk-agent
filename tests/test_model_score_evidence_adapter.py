from __future__ import annotations

from copy import deepcopy
import hashlib

import numpy as np
import pytest

from marvis.packs.modeling.score_evidence import (
    build_single_model_score_evidence,
)
from marvis.packs.strategy.model_evidence import build_artifact_ref
from marvis.packs.strategy.model_score_evidence_adapter import (
    ModelScoreEvidenceComparisonError,
    build_model_score_comparison,
)
from tests.test_model_score_evidence_kernel import _frame
from tests.test_strategy_sample_design_v2 import (
    _bundle,
    _decoded_membership,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _artifact(label: str, kind: str) -> dict[str, str]:
    return build_artifact_ref(
        kind=kind,
        ref_id=label,
        content_hash=_hash(label),
    )


def _model(bundle, label: str, scores, *, frame=None) -> dict:
    return build_single_model_score_evidence(
        sample_design_bundle=bundle,
        membership_masks=_decoded_membership()["masks"],
        frame=_frame() if frame is None else frame,
        scores=np.asarray(scores, dtype=np.float64),
        training_evidence_ref=_artifact(
            f"{label}-training",
            "modeling_training_evidence_json",
        ),
        model_ref=_artifact(f"{label}-model", "modeling_model_binary"),
        score_ref=_artifact(f"{label}-score", "model_score_vector_parquet"),
        features=["x"],
    )


def test_comparison_derives_shared_values_and_never_selects_a_model() -> None:
    bundle = _bundle()
    first = _model(
        bundle,
        "first",
        [0.1, 0.9, 0.2, 0.15, 0.25, 0.3, 0.35, 0.4],
    )
    second = _model(
        bundle,
        "second",
        [0.2, 0.8, 0.3, 0.4, 0.1, 0.7, 0.9, 0.5],
    )

    comparison = build_model_score_comparison(
        sample_design_bundle=bundle,
        model_evidence=[first, second],
        population="risk",
        partition="development",
    )

    assert comparison["selection"] == {
        "status": "no_selection",
        "selected_model_evidence_ref": None,
        "metric_key": None,
        "period": None,
        "direction": None,
        "reason": "comparison_evidence_does_not_authorize_selection",
    }
    values = {
        item["metric_key"]: {
            value["model_evidence_ref"]["evidence_id"]: value["value"]
            for value in item["model_values"]
        }
        for item in comparison["metrics"]
        if item["period"] is None
    }
    assert values["auc"][first["evidence_id"]] == 1.0
    assert values["auc"][second["evidence_id"]] == 1.0
    assert set(values) >= {"auc", "ks", "lift_head_5", "score_psi"}
    assert (
        build_model_score_comparison(
            sample_design_bundle=bundle,
            model_evidence=[second, first],
            population="risk",
            partition="development",
        )
        == comparison
    )


def test_comparison_keeps_only_metrics_present_on_every_model() -> None:
    bundle = _bundle()
    first = _model(
        bundle,
        "first",
        [0.1, 0.9, 0.2, 0.15, 0.25, 0.3, 0.35, 0.4],
    )
    single_class_frame = _frame()
    single_class_frame.loc[[4, 6], "target"] = 1
    second = _model(
        bundle,
        "second",
        [0.2, 0.8, 0.3, 0.4, 0.1, 0.7, 0.9, 0.5],
        frame=single_class_frame,
    )

    comparison = build_model_score_comparison(
        sample_design_bundle=bundle,
        model_evidence=[first, second],
        population="risk",
        partition="oot",
    )

    coordinates = {
        (item["metric_key"], item["period"]) for item in comparison["metrics"]
    }
    assert ("score_psi", None) in coordinates
    assert ("auc", None) not in coordinates
    assert all(item["status"] == "present" for item in comparison["metrics"])


def test_comparison_rejects_unvalidated_or_cross_sample_model_evidence() -> None:
    bundle = _bundle()
    first = _model(
        bundle,
        "first",
        [0.1, 0.9, 0.2, 0.15, 0.25, 0.3, 0.35, 0.4],
    )
    second = _model(
        bundle,
        "second",
        [0.2, 0.8, 0.3, 0.4, 0.1, 0.7, 0.9, 0.5],
    )
    forged = deepcopy(second)
    forged["observations"][0]["value"] = 0.123
    with pytest.raises(ModelScoreEvidenceComparisonError):
        build_model_score_comparison(
            sample_design_bundle=bundle,
            model_evidence=[first, forged],
            population="risk",
            partition="development",
        )

    other_bundle = _bundle(empty_oot=True)
    other = build_single_model_score_evidence(
        sample_design_bundle=other_bundle,
        membership_masks=_decoded_membership(empty_oot=True)["masks"],
        frame=_frame(),
        scores=np.asarray(
            [0.2, 0.8, 0.3, 0.4, 0.1, 0.7, 0.9, 0.5],
            dtype=np.float64,
        ),
        training_evidence_ref=_artifact(
            "other-training",
            "modeling_training_evidence_json",
        ),
        model_ref=_artifact("other-model", "modeling_model_binary"),
        score_ref=_artifact("other-score", "model_score_vector_parquet"),
        features=["x"],
    )
    with pytest.raises(ModelScoreEvidenceComparisonError):
        build_model_score_comparison(
            sample_design_bundle=bundle,
            model_evidence=[first, other],
            population="risk",
            partition="development",
        )


def test_comparison_requires_two_distinct_models() -> None:
    bundle = _bundle()
    first = _model(
        bundle,
        "first",
        [0.1, 0.9, 0.2, 0.15, 0.25, 0.3, 0.35, 0.4],
    )
    with pytest.raises(ModelScoreEvidenceComparisonError, match="at least two"):
        build_model_score_comparison(
            sample_design_bundle=bundle,
            model_evidence=[first],
            population="risk",
            partition="development",
        )


def test_comparison_ref_rejects_unknown_caller_fields() -> None:
    bundle = _bundle()
    first = _model(
        bundle,
        "first",
        [0.1, 0.9, 0.2, 0.15, 0.25, 0.3, 0.35, 0.4],
    )
    second = _model(
        bundle,
        "second",
        [0.2, 0.8, 0.3, 0.4, 0.1, 0.7, 0.9, 0.5],
    )
    with pytest.raises(ModelScoreEvidenceComparisonError, match="artifact reference"):
        build_model_score_comparison(
            sample_design_bundle=bundle,
            model_evidence=[first, second],
            population="risk",
            partition="development",
            comparison_ref={
                **_artifact("comparison", "model_score_comparison_adapter"),
                "value": 1.0,
            },
        )
