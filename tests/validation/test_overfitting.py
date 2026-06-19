import pytest

from marvis.validation.overfitting import (
    OVERFIT_TRAIN_OOT_ABS,
    OVERFIT_TRAIN_TEST_REL,
    overfitting_check,
    overfitting_check_from_validation_results,
)


def test_overfitting_check_returns_modeling_gaps_and_flag():
    gap_tt, gap_to, flag = overfitting_check(0.40, 0.30, 0.32)

    assert gap_tt == pytest.approx(0.25)
    assert gap_to == pytest.approx(0.08)
    assert flag is True


def test_overfitting_check_passes_at_modeling_threshold_boundary():
    gap_tt, gap_to, flag = overfitting_check(
        0.50,
        0.50 * (1 - OVERFIT_TRAIN_TEST_REL),
        0.50 - OVERFIT_TRAIN_OOT_ABS,
    )

    assert gap_tt == pytest.approx(OVERFIT_TRAIN_TEST_REL)
    assert gap_to == pytest.approx(OVERFIT_TRAIN_OOT_ABS)
    assert flag is False


def test_overfitting_check_handles_zero_train_ks_and_missing_oot():
    gap_tt, gap_to, flag = overfitting_check(0.0, 0.20, None)

    assert gap_tt == 0.0
    assert gap_to is None
    assert flag is False


def test_overfitting_check_flags_train_test_relative_gap_and_train_oot_abs_gap():
    result = overfitting_check_from_validation_results({
        "effectiveness": {
            "overall": [
                {"split": "train", "ks": 0.40},
                {"split": "test", "ks": 0.30},
                {"split": "oot", "ks": 0.32},
            ]
        }
    })

    assert result["status"] == "fail"
    assert result["train_test_relative_diff"] == pytest.approx(0.25)
    assert result["train_test_status"] == "fail"
    assert result["train_oot_abs_diff"] == pytest.approx(0.08)
    assert result["train_oot_status"] == "fail"


def test_overfitting_check_reports_not_available_when_required_splits_missing():
    result = overfitting_check_from_validation_results({
        "effectiveness": {"overall": [{"split": "train", "ks": 0.40}]}
    })

    assert result["status"] == "not_available"
    assert result["test_ks"] is None
    assert result["oot_ks"] is None


def test_overfitting_check_passes_when_gaps_within_thresholds():
    result = overfitting_check_from_validation_results({
        "effectiveness": {
            "overall": [
                {"split": "train", "ks": 0.50},
                {"split": "test", "ks": 0.46},   # relative diff = 0.08 < 0.10
                {"split": "oot", "ks": 0.47},    # abs diff = 0.03 < 0.05
            ]
        }
    })

    assert result["status"] == "pass"
    assert result["train_test_status"] == "pass"
    assert result["train_oot_status"] == "pass"


def test_overfitting_check_passes_at_threshold_boundary():
    # value == threshold must pass (the check uses strict > for fail)
    result = overfitting_check_from_validation_results({
        "effectiveness": {
            "overall": [
                {"split": "train", "ks": 0.50},
                {"split": "test", "ks": 0.45},   # relative diff = 0.10 exactly
                {"split": "oot", "ks": 0.45},    # abs diff = 0.05 exactly
            ]
        }
    })

    assert result["train_test_relative_diff"] == pytest.approx(0.10)
    assert result["train_test_status"] == "pass"
    assert result["train_oot_abs_diff"] == pytest.approx(0.05)
    assert result["train_oot_status"] == "pass"
    assert result["status"] == "pass"
