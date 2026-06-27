"""Reduced model-development report for NON-binary targets (regression / multiclass).

The full credit-risk report (``model_report.py``) is binary-specific end to end —
bad-rate, Vintage, OOT decile bins, low-pricing/product-removal stress tests all
assume a 0/1 label and a binary score. Those concepts do not apply to a regression
or multiclass model, so for a non-binary target we write a compact workbook (a
summary + the relevant metrics) instead, letting the conversational flow finish with
a downloadable artifact rather than crashing on the binary-only computations.
"""

from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font

# (display label, ModelMetrics field stem) per target type. The stem is prefixed with
# the split (train_/test_/oot_) to read the matching ModelMetrics attribute.
_METRIC_SPECS: dict[str, list[tuple[str, str]]] = {
    "continuous": [("RMSE", "rmse"), ("MAE", "mae"), ("R²", "r2")],
    "multiclass": [("macro-AUC", "macro_auc"), ("logloss", "logloss"), ("准确率", "accuracy")],
}
_TARGET_LABEL = {"continuous": "回归(连续型)", "multiclass": "多分类"}


def render_minimal_model_report(experiment, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    config = experiment.config
    metrics = experiment.metrics
    target_type = getattr(config, "target_type", "binary")

    workbook = Workbook()
    summary = workbook.active
    summary.title = "汇总"
    summary["A1"] = "模型开发报告(精简)"
    summary["A1"].font = Font(bold=True, size=14)
    summary_rows = [
        ("目标列", config.target_col),
        ("目标类型", _TARGET_LABEL.get(target_type, target_type)),
        ("算法", experiment.recipe_id),
        ("特征数", len(config.features)),
        ("说明", "非二分类任务:不含坏率/Vintage/OOT分箱/压力测试等二分类信贷专用分析。"),
    ]
    for offset, (key, value) in enumerate(summary_rows, start=3):
        summary[f"A{offset}"] = key
        summary[f"B{offset}"] = value

    sheet = workbook.create_sheet("模型指标")
    for col, header in zip("ABCD", ("指标", "train", "test", "oot")):
        cell = sheet[f"{col}1"]
        cell.value = header
        cell.font = Font(bold=True)
    specs = _METRIC_SPECS.get(target_type, [("KS", "ks"), ("AUC", "auc")])
    for row, (label, stem) in enumerate(specs, start=2):
        sheet[f"A{row}"] = label
        for col, split in zip("BCD", ("train", "test", "oot")):
            value = getattr(metrics, f"{split}_{stem}", None) if metrics is not None else None
            sheet[f"{col}{row}"] = "n/a" if value is None else round(float(value), 6)

    workbook.save(out_path)
    return out_path


__all__ = ["render_minimal_model_report"]
