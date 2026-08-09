from __future__ import annotations

import pytest

from marvis.output.xlsx_safety import looks_like_excel_formula, safe_xlsx_text


@pytest.mark.parametrize(
    "value",
    [
        "=1+1",
        "  +SUM(A1:A2)",
        "\t-HYPERLINK('https://example.invalid')",
        "\u200b@cmd",
        "\x00=hidden",
    ],
)
def test_formula_detection_ignores_leading_space_and_control_characters(
    value: str,
) -> None:
    assert looks_like_excel_formula(value) is True
    assert safe_xlsx_text(value).startswith("'")


@pytest.mark.parametrize("value", ["ordinary", "1+1", "邮箱@example.com", ""])
def test_formula_detection_keeps_plain_text(value: str) -> None:
    assert looks_like_excel_formula(value) is False
