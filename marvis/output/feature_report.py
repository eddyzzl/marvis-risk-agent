"""Standalone feature-analysis Excel report (FEATURE phase, form A).

Writes the per-feature metrics computed by ``compute_feature_metrics`` into a
downloadable workbook, mirroring the model-report download pipeline. Missing
metrics render as "n/a" rather than blank, so a sheet is never silently empty.
"""

from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font

from marvis.artifacts import TransactionalArtifactStore


# (metric key, column header). Order is the sheet column order.
_COLUMNS: list[tuple[str, str]] = [
    ("feature", "特征"),
    ("iv", "IV"),
    ("ks", "KS"),
    ("auc", "AUC"),
    ("psi", "PSI"),
    ("missing_rate", "缺失率"),
    ("lift_top_bin", "头部lift"),
]

# Optional columns — each appended only when that metric was selected (i.e. the
# per-feature rows actually carry the key).
_HEAD_TAIL_COLUMNS: list[tuple[str, str]] = [
    ("lift_head_5", "头部lift5%"),
    ("lift_head_10", "头部lift10%"),
    ("lift_tail_5", "尾部lift5%"),
    ("lift_tail_10", "尾部lift10%"),
]
_IMPORTANCE_COLUMN: tuple[str, str] = ("importance", "重要性")


def render_feature_report(metrics: list[dict], out_path: Path, *, collinear: dict | None = None) -> Path:
    out_path = Path(out_path)
    artifact = TransactionalArtifactStore(out_path.parent).stage(out_path.name)
    rows = [item for item in (metrics or []) if isinstance(item, dict)]
    columns = list(_COLUMNS)
    if any("lift_head_5" in item for item in rows):
        columns += _HEAD_TAIL_COLUMNS
    if any("importance" in item for item in rows):
        columns += [_IMPORTANCE_COLUMN]
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "特征指标"
    sheet.append([label for _key, label in columns])
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    for item in rows:
        sheet.append([_cell(item.get(key)) for key, _label in columns])
    # Optional collinear / VIF sheet — written only when the VIF metric was selected.
    if isinstance(collinear, dict):
        _append_collinear_sheet(workbook, collinear)
    try:
        workbook.save(artifact.path)
        artifact.promote()
        artifact.commit()
    except Exception:
        artifact.rollback()
        raise
    return artifact.final_path


def _append_collinear_sheet(workbook: Workbook, collinear: dict) -> None:
    sheet = workbook.create_sheet("共线性(VIF)")
    sheet.append(["特征", "VIF"])
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    for feat, value in (collinear.get("vif") or {}).items():
        sheet.append([str(feat), _cell(value)])
    pairs = [p for p in (collinear.get("collinear_pairs") or []) if isinstance(p, (list, tuple)) and len(p) >= 3]
    if pairs:
        sheet.append([])
        header = sheet.max_row + 1
        sheet.append(["特征A", "特征B", "相关系数"])
        for cell in sheet[header]:
            cell.font = Font(bold=True)
        for pair in pairs:
            sheet.append([str(pair[0]), str(pair[1]), _cell(pair[2])])


def _cell(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return round(value, 6)
    return value


__all__ = ["render_feature_report"]
