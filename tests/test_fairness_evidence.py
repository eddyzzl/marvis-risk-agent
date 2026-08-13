"""Tests for the evidence-only fairness segment module."""

import pandas as pd
import pytest

from marvis.packs.strategy.fairness_evidence import (
    MISSING_LABEL,
    NEUTRAL_PHRASING_DRAFT_NOTICE,
    NEUTRAL_PHRASING_TEMPLATES,
    OVERALL_SEGMENT,
    FairnessEvidenceError,
    SegmentDisparityEvidence,
    segment_disparity_evidence,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "strategy_pool_action": [
                "approval",
                "approval",
                "reject",
                "reject",
                "approval",
                "reject",
                "reject",
                "review",
            ],
            "strategy_pool_reason_code": [
                None,
                None,
                "R1",
                "R1",
                None,
                "R2",
                "R2",
                None,
            ],
            "bad": [0, 0, 1, 0, 1, 1, 0, 0],
        }
    )


_OUTCOME_COLUMNS = {
    "action": "strategy_pool_action",
    "reason_code": "strategy_pool_reason_code",
}

_FORBIDDEN_VERDICT_TERMS = (
    "不公平",
    "公平",
    "歧视",
    "偏见",
    "bias",
    "fair",
    "unfair",
    "verdict",
)


def test_segment_disparity_evidence_golden_numbers():
    result = segment_disparity_evidence(
        _frame(), "segment", _OUTCOME_COLUMNS, "bad"
    )

    assert isinstance(result, SegmentDisparityEvidence)
    assert result.segment_column == "segment"
    assert result.action_column == "strategy_pool_action"
    assert result.reason_code_column == "strategy_pool_reason_code"
    assert result.target_column == "bad"

    overall = result.overall
    assert overall.segment == OVERALL_SEGMENT
    assert overall.count == 8
    assert overall.approved_count == 3
    assert overall.approval_rate == pytest.approx(3 / 8)
    assert overall.bad_count == 3
    assert overall.labelled_count == 8
    assert overall.bad_rate == pytest.approx(3 / 8)
    assert dict(overall.reason_code_distribution) == {"R1": 2, "R2": 2}

    assert tuple(row.segment for row in result.segments) == ("A", "B")
    by_segment = {row.segment: row for row in result.segments}

    a = by_segment["A"]
    assert a.count == 4
    assert a.approved_count == 2
    assert a.approval_rate == pytest.approx(0.5)
    assert a.bad_count == 1
    assert a.labelled_count == 4
    assert a.bad_rate == pytest.approx(0.25)
    assert dict(a.reason_code_distribution) == {"R1": 2}

    b = by_segment["B"]
    assert b.count == 4
    assert b.approved_count == 1
    assert b.approval_rate == pytest.approx(0.25)
    assert b.bad_count == 2
    assert b.labelled_count == 4
    assert b.bad_rate == pytest.approx(0.5)
    assert dict(b.reason_code_distribution) == {"R2": 2}


def test_reason_code_distribution_counts_rejected_rows_only():
    # The review row (reason_code None) and the approval rows must never leak
    # into the rejection-reason distribution.
    result = segment_disparity_evidence(
        _frame(), "segment", _OUTCOME_COLUMNS, "bad"
    )
    assert dict(result.overall.reason_code_distribution) == {"R1": 2, "R2": 2}


def test_missing_segment_column_fails_closed():
    frame = _frame().drop(columns=["segment"])
    with pytest.raises(FairnessEvidenceError):
        segment_disparity_evidence(frame, "segment", _OUTCOME_COLUMNS, "bad")


def test_missing_action_column_fails_closed():
    frame = _frame().drop(columns=["strategy_pool_action"])
    with pytest.raises(FairnessEvidenceError):
        segment_disparity_evidence(frame, "segment", _OUTCOME_COLUMNS, "bad")


def test_missing_target_column_fails_closed():
    frame = _frame().drop(columns=["bad"])
    with pytest.raises(FairnessEvidenceError):
        segment_disparity_evidence(frame, "segment", _OUTCOME_COLUMNS, "bad")


def test_outcome_columns_without_action_key_fails_closed():
    with pytest.raises(FairnessEvidenceError):
        segment_disparity_evidence(
            _frame(),
            "segment",
            {"reason_code": "strategy_pool_reason_code"},
            "bad",
        )


def test_requires_dataframe_input():
    with pytest.raises(FairnessEvidenceError):
        segment_disparity_evidence(
            [{"segment": "A"}], "segment", _OUTCOME_COLUMNS, "bad"
        )


def test_reason_code_column_is_optional():
    result = segment_disparity_evidence(
        _frame(), "segment", {"action": "strategy_pool_action"}, "bad"
    )
    assert result.reason_code_column is None
    assert result.overall.reason_code_distribution is None
    assert all(
        row.reason_code_distribution is None for row in result.segments
    )


def test_reason_code_distribution_includes_missing_label():
    frame = _frame()
    frame.loc[3, "strategy_pool_reason_code"] = None
    result = segment_disparity_evidence(
        frame, "segment", _OUTCOME_COLUMNS, "bad"
    )
    a = next(row for row in result.segments if row.segment == "A")
    assert dict(a.reason_code_distribution) == {MISSING_LABEL: 1, "R1": 1}


def test_missing_segment_value_is_labelled_deterministically():
    frame = _frame()
    frame.loc[0, "segment"] = None
    result = segment_disparity_evidence(
        frame, "segment", _OUTCOME_COLUMNS, "bad"
    )
    labels = {row.segment for row in result.segments}
    assert MISSING_LABEL in labels


def test_bad_rate_none_when_no_labelled_rows():
    frame = pd.DataFrame(
        {
            "segment": ["A", "A"],
            "strategy_pool_action": ["approval", "reject"],
            "strategy_pool_reason_code": [None, "R1"],
            "bad": [None, None],
        }
    )
    result = segment_disparity_evidence(
        frame, "segment", _OUTCOME_COLUMNS, "bad"
    )
    assert result.overall.labelled_count == 0
    assert result.overall.bad_count == 0
    assert result.overall.bad_rate is None


def test_output_contains_no_verdict_wording():
    result = segment_disparity_evidence(
        _frame(), "segment", _OUTCOME_COLUMNS, "bad"
    )
    text = repr(result).lower()
    for term in _FORBIDDEN_VERDICT_TERMS:
        assert term.lower() not in text


def test_neutral_phrasing_templates_are_draft_marked_and_neutral():
    assert NEUTRAL_PHRASING_TEMPLATES
    assert "草稿" in NEUTRAL_PHRASING_DRAFT_NOTICE
    assert "合规" in NEUTRAL_PHRASING_DRAFT_NOTICE
    for template in NEUTRAL_PHRASING_TEMPLATES:
        lowered = template.lower()
        for term in _FORBIDDEN_VERDICT_TERMS:
            assert term.lower() not in lowered


def test_output_is_deterministic():
    frame = _frame()
    first = segment_disparity_evidence(
        frame, "segment", _OUTCOME_COLUMNS, "bad"
    )
    second = segment_disparity_evidence(
        frame, "segment", _OUTCOME_COLUMNS, "bad"
    )
    assert repr(first) == repr(second)
