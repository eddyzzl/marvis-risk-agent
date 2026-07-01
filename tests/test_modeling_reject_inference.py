import pandas as pd
import pytest

import marvis.packs.modeling as modeling
from marvis.packs.modeling.reject_inference import (
    SAMPLE_WEIGHT_COL,
    SOURCE_COL,
    reject_inference,
)


def test_reject_inference_parceling_assigns_labels_and_weights_by_score():
    frame = pd.DataFrame({
        "score": [0.1, 0.2, 0.9, 0.8],
        "bad": [0, 1, None, None],
        "decision": ["approved", "approved", "rejected", "rejected"],
    })

    result = reject_inference(
        frame,
        target_col="bad",
        decision_col="decision",
        score_col="score",
        reject_bad_rate=0.5,
        reject_weight=0.7,
    )

    rejected = result.frame[result.frame[SOURCE_COL] == "rejected_inferred"].sort_values("score", ascending=False)
    assert result.target_col == "__reject_inference_target__"
    assert result.sample_weight_col == SAMPLE_WEIGHT_COL
    assert result.diagnostics["method"] == "parceling"
    assert result.diagnostics["reject_bad_rate_assumption"] == 0.5
    assert rejected[result.target_col].tolist() == [1, 0]
    assert rejected[SAMPLE_WEIGHT_COL].tolist() == [0.7, 0.7]


def test_reject_inference_fuzzy_augmentation_splits_reject_weight():
    frame = pd.DataFrame({
        "bad": [0, 1, None],
        "approved": [1, 1, 0],
    })

    result = reject_inference(
        frame,
        target_col="bad",
        decision_col="approved",
        method="fuzzy_augmentation",
        reject_bad_rate=0.25,
        reject_weight=2.0,
    )

    inferred = result.frame[result.frame[SOURCE_COL].str.startswith("rejected_inferred")]
    assert inferred[result.target_col].tolist() == [1, 0]
    assert inferred[SAMPLE_WEIGHT_COL].tolist() == pytest.approx([0.5, 1.5])
    assert result.diagnostics["output_rows"] == 4


def test_reject_inference_fuzzy_augmentation_omits_zero_weight_side():
    frame = pd.DataFrame({
        "bad": [0, 1, None],
        "approved": [1, 1, 0],
    })

    result = reject_inference(
        frame,
        target_col="bad",
        decision_col="approved",
        method="fuzzy_augmentation",
        reject_bad_rate=1.0,
        reject_weight=2.0,
    )

    inferred = result.frame[result.frame[SOURCE_COL].str.startswith("rejected_inferred")]
    assert inferred[result.target_col].tolist() == [1]
    assert inferred[SAMPLE_WEIGHT_COL].tolist() == pytest.approx([2.0])
    assert result.diagnostics["output_rows"] == 3


def test_reject_inference_is_registered_as_modeling_capability():
    assert "reject_inference" in modeling.__all__
