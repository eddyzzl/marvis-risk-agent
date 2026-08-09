import pytest

from marvis.output.styles import (
    BRAND_HEADER_FILL,
    BRAND_HEADER_FONT_COLOR,
    CJK_FONT_CANDIDATES,
    FONT_NAME,
    FONT_SIZE_PT,
    STRESS_HIGH_FILL,
    STRESS_LOW_FILL,
    STRESS_MEDIUM_FILL,
    ks_drop_ratio,
    stress_ks_risk,
    stress_psi_risk,
    stress_risk_cell_color,
    status_cell_color,
    ks_delta_cell_color,
    worst_stress_risk,
)
from marvis.validation.results import ConsistencyStatus


def test_brand_tokens_are_hex_strings():
    assert BRAND_HEADER_FILL == "C00000"
    assert BRAND_HEADER_FONT_COLOR == "FFFFFF"
    assert FONT_NAME == CJK_FONT_CANDIDATES[0]
    assert "Microsoft YaHei" in CJK_FONT_CANDIDATES
    assert len(CJK_FONT_CANDIDATES) > 1
    assert FONT_SIZE_PT == 8


def test_status_cell_color_maps_three_states():
    assert status_cell_color(ConsistencyStatus.PASS).startswith("C6")  # green-ish
    assert status_cell_color(ConsistencyStatus.REVIEW).startswith("FF")  # yellow-ish
    assert status_cell_color(ConsistencyStatus.FAIL).startswith("F4")  # red-ish


def test_ks_delta_cell_color_thresholds():
    assert ks_delta_cell_color(0.0) is None
    assert ks_delta_cell_color(-0.005) is None
    assert ks_delta_cell_color(-0.02).startswith("FF")  # yellow at >=0.01
    assert ks_delta_cell_color(-0.05).startswith("F4")  # red at >=0.03


def test_stress_ks_drop_ratio_and_one_sided_risk_thresholds():
    assert ks_drop_ratio(0.25, 0.225) == pytest.approx(0.1)
    assert ks_drop_ratio(0.25, 0.20) == pytest.approx(0.2)
    assert ks_drop_ratio(0.25, 0.30) == pytest.approx(-0.2)
    assert ks_drop_ratio(0.0, 0.0) is None

    assert stress_ks_risk(0.25, 0.251) == "low"
    assert stress_ks_risk(0.25, 0.225) == "medium"
    assert stress_ks_risk(0.25, 0.20) == "high"
    assert stress_ks_risk(0.25, 0.30) == "low"
    assert stress_ks_risk(0.0, 0.0) is None


def test_stress_psi_and_overall_risk_thresholds_use_fixed_rgb():
    assert stress_psi_risk(0.099999) == "low"
    assert stress_psi_risk(0.10) == "medium"
    assert stress_psi_risk(0.249999) == "medium"
    assert stress_psi_risk(0.25) == "high"
    assert worst_stress_risk("low", "high", "medium") == "high"

    assert stress_risk_cell_color("low") == STRESS_LOW_FILL == "C6EFCE"
    assert stress_risk_cell_color("medium") == STRESS_MEDIUM_FILL == "FFEB9C"
    assert stress_risk_cell_color("high") == STRESS_HIGH_FILL == "FFC7CE"
