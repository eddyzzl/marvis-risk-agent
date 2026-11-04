"""The development-refit model report shows business Chinese, not producer codes.

The structured rows keep ``pre_refit_oot`` and friends (tests, archives and the
delivery card key off them); only the workbook swaps them for labels.
"""
from types import SimpleNamespace

from openpyxl import load_workbook

from marvis.output.model_report import ModelReportPayload, render_model_report
from marvis.packs.modeling.development_refit_reporting import (
    dataset_split_rows,
    diagnostic_rows,
    report_display_labels,
)

_SCOPE = {
    "experiment_id": "experiment_refit",
    "artifact_id": "artifact_refit",
    "parent_metrics": {"train_ks": 0.5, "train_auc": 0.8, "test_ks": 0.42, "test_auc": 0.77,
                       "oot_ks": 0.4, "oot_auc": 0.75},
    "proof": {
        "parent": {"experiment_id": "experiment_parent", "artifact_id": "artifact_parent"},
        "memberships": {"development_fit": {"rows": 900}, "original_train": {"rows": 700},
                        "original_test": {"rows": 200}},
    },
}
_METADATA = {"final_fit_scope": "完整 development train+test；原 test 已参与拟合",
             "historical_availability": "unknown", "label_maturity": "unknown"}


def _rows():
    split = dataset_split_rows(_SCOPE, SimpleNamespace(train_ks=0.5, train_auc=0.8))
    bands = diagnostic_rows([{"split": "test", "bin": 1, "sample_count": 20}], _SCOPE)
    return split, bands


def test_every_code_the_refit_producer_writes_has_a_label():
    split, bands = _rows()
    labels = report_display_labels()
    values = labels["values"]
    assert {row["split"] for row in split} <= set(values["split"])
    assert {row["purpose"] for row in split} <= set(values["purpose"])
    assert {row["evaluation_purpose"] for row in bands} <= set(values["evaluation_purpose"])
    for key in ("historical_availability", "label_maturity"):
        assert {row[key] for row in bands} <= set(values[key])
    # structured rows keep their codes
    assert [row["split"] for row in split][:2] == ["development_train_plus_test", "pre_refit_train"]


def test_refit_workbook_shows_labels_and_ordinary_workbook_keeps_its_cells(tmp_path):
    split, bands = _rows()
    refit = render_model_report(ModelReportPayload(
        project_meta=dict(_METADATA), dataset_split=split, stability=[], sample_analysis=None,
        vintage=None, feature_importance=[], score_bands=bands,
        display_labels=report_display_labels(),
    ), tmp_path / "refit.xlsx")
    workbook = load_workbook(refit)
    summary = {
        (workbook["汇总"].cell(row=row, column=1).value, workbook["汇总"].cell(row=row, column=2).value)
        for row in range(1, workbook["汇总"].max_row + 1)
    }
    assert ("split", "所选实验 OOT（重训前留出）") in summary
    assert ("用途", "重训前的 OOT 留出证据，重训后未重新评估") in summary
    assert ("最终拟合范围", _METADATA["final_fit_scope"]) in summary
    assert ("标签成熟度", "未知") in summary
    band_cells = {cell.value for row in workbook["评分分段"].iter_rows() for cell in row}
    assert {"评估用途", "已参与拟合，仅作诊断", "历史数据可得性", "未知"} <= band_cells
    every = {cell.value for sheet in workbook for row in sheet.iter_rows() for cell in row}
    assert not {"pre_refit_oot", "parent_heldout_oot_before_refit", "consumed_in_fit_diagnostic",
                "evaluation_purpose", "final_fit_scope", "unknown"} & every

    ordinary = render_model_report(ModelReportPayload(
        project_meta={"项目名称": "建模报告"}, dataset_split=[{"split": "train", "ks": 0.4}],
        stability=[], sample_analysis=None, vintage=None, feature_importance=[],
    ), tmp_path / "ordinary.xlsx")
    sheet = load_workbook(ordinary)["汇总"]
    cells = {(sheet.cell(row=row, column=1).value, sheet.cell(row=row, column=2).value)
             for row in range(1, sheet.max_row + 1)}
    assert ("split", "train") in cells and ("ks", 0.4) in cells


def test_pre_refit_oot_row_only_when_the_parent_measured_oot():
    split, _bands = _rows()
    oot = next(row for row in split if row["split"] == "pre_refit_oot")
    # every split row carries the same columns
    assert len({tuple(row) for row in split}) == 1
    assert oot["ks"] == 0.4 and oot["sample_count"] is None

    no_oot = {**_SCOPE, "parent_metrics": {k: v for k, v in _SCOPE["parent_metrics"].items()
                                           if not k.startswith("oot_")}}
    rows = dataset_split_rows(no_oot, SimpleNamespace(train_ks=0.5, train_auc=0.8))
    assert "pre_refit_oot" not in {row["split"] for row in rows}
