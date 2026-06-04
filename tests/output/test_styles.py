from riskmodel_checker.output.styles import (
    BRAND_HEADER_FILL,
    BRAND_HEADER_FONT_COLOR,
    CJK_FONT_CANDIDATES,
    FONT_NAME,
    FONT_SIZE_PT,
    status_cell_color,
    ks_delta_cell_color,
)
from riskmodel_checker.validation.results import ConsistencyStatus


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
