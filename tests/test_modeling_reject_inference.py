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


def _fuzzy_score_binned_frame() -> pd.DataFrame:
    """20 accepted rows (2 per equal-frequency bin, 10 bins) + 4 rejected rows.

    Bin 1 (scores 1,2): labels [0,1] -> empirical bad rate 0.5.
    Bin 2 (scores 3,4): labels [1,1] -> empirical bad rate 1.0.
    Bins 3..10 (scores 5..20): all good -> empirical bad rate 0.0.
    Overall accepted_bad_rate = 3/20 = 0.15.
    Rejected rows 1.5, 1.8 land in bin 1; 3.2, 3.9 land in bin 2.
    """
    accepted_scores = list(range(1, 21))
    accepted_labels = [0] * 20
    accepted_labels[0:2] = [0, 1]
    accepted_labels[2:4] = [1, 1]
    rejected_scores = [1.5, 1.8, 3.2, 3.9]
    return pd.DataFrame({
        "score": accepted_scores + rejected_scores,
        "bad": accepted_labels + [None] * 4,
        "decision": ["approved"] * 20 + ["rejected"] * 4,
    })


def test_reject_inference_fuzzy_augmentation_uses_per_record_binned_scores():
    # DOM-12: with a score column, fuzzy augmentation maps each rejected row to its
    # own equal-frequency bin's empirical bad rate instead of one global bad_rate
    # for every reject. reject_bad_rate=0.15 matches accepted_bad_rate exactly, so
    # the anchor scale factor is 1.0 and bin rates pass through unchanged.
    frame = _fuzzy_score_binned_frame()

    result = reject_inference(
        frame,
        target_col="bad",
        decision_col="decision",
        method="fuzzy_augmentation",
        score_col="score",
        reject_bad_rate=0.15,
        reject_weight=1.0,
    )

    inferred = result.frame[result.frame[SOURCE_COL].str.startswith("rejected_inferred")]
    by_score = {
        (round(float(row["score"]), 1), row[SOURCE_COL]): row[SAMPLE_WEIGHT_COL]
        for _, row in inferred.iterrows()
    }
    # Bin 1 (score 1.5, 1.8): p=0.5 -> bad/good weights both 1.0*0.5 = 0.5 (hand-computed).
    assert by_score[(1.5, "rejected_inferred_bad")] == pytest.approx(0.5)
    assert by_score[(1.5, "rejected_inferred_good")] == pytest.approx(0.5)
    assert by_score[(1.8, "rejected_inferred_bad")] == pytest.approx(0.5)
    assert by_score[(1.8, "rejected_inferred_good")] == pytest.approx(0.5)
    # Bin 2 (score 3.2, 3.9): p=1.0 -> bad weight 1.0*1.0=1.0, good weight 0 (omitted).
    assert by_score[(3.2, "rejected_inferred_bad")] == pytest.approx(1.0)
    assert by_score[(3.9, "rejected_inferred_bad")] == pytest.approx(1.0)
    assert (3.2, "rejected_inferred_good") not in by_score
    assert (3.9, "rejected_inferred_good") not in by_score
    assert result.diagnostics["reject_bad_rate_assumption"] == pytest.approx(0.15)


def test_reject_inference_fuzzy_augmentation_anchors_binned_rates_to_override():
    # DOM-12: when reject_bad_rate overrides the raw accepted rate, per-bin rates
    # are scaled proportionally so the population still anchors to the override
    # (scale = reject_bad_rate / accepted_bad_rate = 0.30 / 0.15 = 2.0), then
    # clipped to [0, 1].
    frame = _fuzzy_score_binned_frame()

    result = reject_inference(
        frame,
        target_col="bad",
        decision_col="decision",
        method="fuzzy_augmentation",
        score_col="score",
        reject_bad_rate=0.30,
        reject_weight=1.0,
    )

    inferred = result.frame[result.frame[SOURCE_COL].str.startswith("rejected_inferred")]
    by_score = {
        (round(float(row["score"]), 1), row[SOURCE_COL]): row[SAMPLE_WEIGHT_COL]
        for _, row in inferred.iterrows()
    }
    # Bin 1: raw p=0.5, scaled by 2.0 -> 1.0 (clipped at 1.0) -> bad weight 1.0, good omitted.
    assert by_score[(1.5, "rejected_inferred_bad")] == pytest.approx(1.0)
    assert (1.5, "rejected_inferred_good") not in by_score
    # Bin 2: raw p=1.0, scaled by 2.0 -> clipped to 1.0 -> bad weight 1.0, good omitted.
    assert by_score[(3.2, "rejected_inferred_bad")] == pytest.approx(1.0)
    assert (3.2, "rejected_inferred_good") not in by_score


def test_reject_inference_fuzzy_augmentation_falls_back_without_score_col():
    # DOM-12: no score_col -> unchanged global-bad-rate behavior (existing contract).
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


def test_reject_inference_fuzzy_augmentation_binned_weights_deterministic_repeat():
    # DOM-12 / INV-1: same input -> same output, run twice.
    frame = _fuzzy_score_binned_frame()
    kwargs = dict(
        target_col="bad", decision_col="decision", method="fuzzy_augmentation",
        score_col="score", reject_bad_rate=0.15, reject_weight=1.0,
    )
    first = reject_inference(frame, **kwargs)
    second = reject_inference(frame, **kwargs)
    pd.testing.assert_frame_equal(
        first.frame.sort_values("score").reset_index(drop=True),
        second.frame.sort_values("score").reset_index(drop=True),
    )


def test_reject_inference_fuzzy_augmentation_binned_weights_direction_reversal_unaffected():
    # DOM-12: per-bin empirical bad rate is computed from raw score value + observed
    # label, independent of the declared score_direction (only parceling's risk
    # ordering consumes direction). Flipping the declared direction must not change
    # the per-record fuzzy weights.
    frame = _fuzzy_score_binned_frame()
    higher_riskier = reject_inference(
        frame, target_col="bad", decision_col="decision", method="fuzzy_augmentation",
        score_col="score", reject_bad_rate=0.15, reject_weight=1.0,
        score_direction="higher_is_riskier",
    )
    higher_better = reject_inference(
        frame, target_col="bad", decision_col="decision", method="fuzzy_augmentation",
        score_col="score", reject_bad_rate=0.15, reject_weight=1.0,
        score_direction="higher_is_better", confirm_direction_conflict=True,
    )
    a = higher_riskier.frame.sort_values(["score", SOURCE_COL]).reset_index(drop=True)
    b = higher_better.frame.sort_values(["score", SOURCE_COL]).reset_index(drop=True)
    pd.testing.assert_series_equal(a[SAMPLE_WEIGHT_COL], b[SAMPLE_WEIGHT_COL])


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
