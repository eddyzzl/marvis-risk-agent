"""Portfolio report labels are literal text on the actual XLSX carrier."""
import pytest
from openpyxl import load_workbook

from marvis.output.portfolio_report import PortfolioReportPayload, render_portfolio_report


@pytest.mark.parametrize("text", ["=1+1", "#N/A"])
def test_uploaded_labels_are_literal_in_values_and_dynamic_headers(tmp_path, text):
    path = render_portfolio_report(PortfolioReportPayload(
        project_meta={text: text},
        migration={"heat_table": [{"from": text, text: 0.125}]},
        segment={"segments": [{"segment": text, "count": 2, "net_profit": -0.25}]},
        red_flags=[{"message": text}],
    ), tmp_path / "report.xlsx")
    book = load_workbook(path, read_only=True, data_only=False)
    try:
        for sheet, address in (("组合概览", "A2"), ("组合概览", "B2"), ("桶迁徙", "B1"),
                               ("桶迁徙", "A2"), ("细分画像", "A2"), ("数据质量红旗", "A2")):
            cell = book[sheet][address]
            assert cell.value == text and cell.data_type == "s"
        assert book["桶迁徙"]["B2"].value == 0.125 and book["桶迁徙"]["B2"].data_type == "n"
        assert book["细分画像"]["C2"].value == -0.25 and book["细分画像"]["C2"].data_type == "n"
    finally:
        book.close()
