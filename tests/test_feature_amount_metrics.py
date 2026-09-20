import numpy as np

from marvis.feature.amount_metrics import amount_metrics


def test_amount_rate_uses_paired_rows_and_retains_separate_coverage():
    result = amount_metrics(
        np.array([True, True, True, False]),
        np.array([100, np.nan, 300, 900]),
        np.array([10, 60, np.nan, 90]),
    )
    assert result["loan_amount"] == {
        "status": "available", "sum": 400.0,
        "covered_count": 2, "coverage_rate": 2 / 3,
    }
    assert result["overdue_amount"]["sum"] == 70.0
    assert result["overdue_rate"] == {
        "status": "available", "value": 0.1, "paired_count": 1,
    }


def test_amount_missing_and_empty_bins_keep_distinct_meanings():
    selected = np.array([True])
    absent = amount_metrics(selected, None, None)
    assert absent["loan_amount"] == {"status": "unavailable", "reason": "loan_amount_not_configured"}
    assert absent["overdue_rate"] == {"status": "unavailable", "reason": "amount_column_not_configured"}
    unpaired = amount_metrics(selected, np.array([np.nan]), np.array([3.0]))
    assert unpaired["loan_amount"] == {
        "status": "unavailable", "reason": "no_covered_rows", "coverage_rate": 0.0,
    }
    assert unpaired["overdue_rate"] == {"status": "unavailable", "reason": "no_paired_amounts"}
    empty = amount_metrics(np.array([False]), np.array([0.0]), np.array([3.0]))
    assert empty["loan_amount"] == {
        "status": "available", "sum": 0.0, "covered_count": 0, "coverage_rate": 1.0,
    }
    assert empty["overdue_rate"] == {"status": "not_applicable", "reason": "empty_bin"}
    zero = amount_metrics(selected, np.array([0.0]), np.array([3.0]))
    assert zero["overdue_rate"] == {"status": "not_applicable", "reason": "zero_loan_amount"}
