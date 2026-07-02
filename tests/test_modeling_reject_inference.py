import pandas as pd
import pytest

import marvis.packs.modeling as modeling
from marvis.data.errors import ScoreDirectionConflictError
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


def _direction_frame(n: int = 40, *, positive_corr: bool) -> pd.DataFrame:
    """40 approved (observed-label) rows + 10 rejected rows.

    positive_corr=True: score positively correlates with bad -> implied direction
    higher_is_riskier (matches the reject_inference default). positive_corr=False:
    negatively correlated -> implied higher_is_better.
    """
    approved_scores = list(range(n))
    if positive_corr:
        approved_bad = [1 if i >= n // 2 else 0 for i in range(n)]
    else:
        approved_bad = [1 if i < n // 2 else 0 for i in range(n)]
    rejected_scores = list(range(n, n + 10))
    return pd.DataFrame({
        "score": approved_scores + rejected_scores,
        "bad": approved_bad + [None] * 10,
        "decision": ["approved"] * n + ["rejected"] * 10,
    })


def test_reject_inference_score_direction_default_matches_legacy_behavior():
    frame = pd.DataFrame({
        "score": [0.1, 0.2, 0.9, 0.8],
        "bad": [0, 1, None, None],
        "decision": ["approved", "approved", "rejected", "rejected"],
    })

    default_result = reject_inference(
        frame, target_col="bad", decision_col="decision", score_col="score",
        reject_bad_rate=0.5, reject_weight=0.7,
    )
    explicit_result = reject_inference(
        frame, target_col="bad", decision_col="decision", score_col="score",
        reject_bad_rate=0.5, reject_weight=0.7, score_direction="higher_is_riskier",
    )

    default_rejected = default_result.frame[default_result.frame[SOURCE_COL] == "rejected_inferred"]
    explicit_rejected = explicit_result.frame[explicit_result.frame[SOURCE_COL] == "rejected_inferred"]
    assert default_rejected[default_result.target_col].tolist() == explicit_rejected[explicit_result.target_col].tolist()


def test_reject_inference_score_direction_higher_is_better_flips_risk_order():
    frame = pd.DataFrame({
        "score": [0.1, 0.2, 0.9, 0.8],
        "bad": [0, 1, None, None],
        "decision": ["approved", "approved", "rejected", "rejected"],
    })

    result = reject_inference(
        frame, target_col="bad", decision_col="decision", score_col="score",
        reject_bad_rate=0.5, reject_weight=0.7, score_direction="higher_is_better",
    )

    rejected = result.frame[result.frame[SOURCE_COL] == "rejected_inferred"].sort_values("score")
    # higher_is_better -> low score is high risk, so the lowest-scored rejected row (0.8) is bad.
    assert rejected[result.target_col].tolist() == [1, 0]


def test_reject_inference_raises_on_direction_conflict():
    frame = _direction_frame(positive_corr=True)

    with pytest.raises(ScoreDirectionConflictError):
        reject_inference(
            frame, target_col="bad", decision_col="decision", score_col="score",
            score_direction="higher_is_better",
        )


def test_reject_inference_confirm_direction_conflict_bypasses_gate():
    frame = _direction_frame(positive_corr=True)

    result = reject_inference(
        frame, target_col="bad", decision_col="decision", score_col="score",
        score_direction="higher_is_better", confirm_direction_conflict=True,
    )

    assert result.diagnostics["direction_diagnostics"]["status"] == "conflict"


def test_reject_inference_does_not_raise_when_declared_direction_matches_data():
    frame = _direction_frame(positive_corr=False)

    result = reject_inference(
        frame, target_col="bad", decision_col="decision", score_col="score",
        score_direction="higher_is_better",
    )

    assert result.diagnostics["direction_diagnostics"]["status"] == "consistent"


def test_reject_inference_skips_direction_check_without_score_col():
    frame = pd.DataFrame({
        "bad": [0, 1, None],
        "approved": [1, 1, 0],
    })

    result = reject_inference(
        frame, target_col="bad", decision_col="approved", method="fuzzy_augmentation",
        reject_bad_rate=0.25, reject_weight=2.0,
    )

    assert "direction_diagnostics" not in result.diagnostics
