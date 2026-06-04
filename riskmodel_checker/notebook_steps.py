from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

import nbformat


_HEADING_RE = re.compile(r"^\s{0,3}(#{1,3})\s+(.+?)\s*$")
_SYSTEM_STEP_TITLES = {
    "head": ("system-head", "平台初始化"),
    "tail": ("system-tail", "平台契约检查"),
    "repro-pmml": ("system-repro-pmml", "PMML 打分"),
    "repro-compare": ("system-repro-compare", "分数一致性对比"),
    "metrics-prepare": ("system-metrics-prepare", "指标数据准备"),
    "metrics-score": ("system-metrics-score", "RMC_SCORE_FN 全量打分"),
    "metrics-basic": ("system-metrics-basic", "样本与变量概览"),
    "metrics-ks": ("system-metrics-ks", "KS 计算"),
    "metrics-psi": ("system-metrics-psi", "PSI 计算"),
    "metrics-binning": ("system-metrics-binning", "分箱计算"),
    "metrics-effectiveness": ("system-metrics-effectiveness", "KS / PSI / 分箱计算"),
    "metrics-stress": ("system-metrics-stress", "压力测试"),
    "metrics-output": ("system-metrics-output", "写入指标产物"),
}


@dataclass(frozen=True)
class NotebookStep:
    id: str
    title: str
    cell_indexes: list[int] = field(default_factory=list)
    source_previews: list[str] = field(default_factory=list)
    system: bool = False


@dataclass(frozen=True)
class NotebookStepPlan:
    steps: list[NotebookStep]
    cell_to_step: dict[int, str]


def notebook_step_plan(notebook: Any) -> NotebookStepPlan:
    steps_by_id: dict[str, NotebookStep] = {}
    cell_to_step: dict[int, str] = {}
    current_step_id = "notebook-init"
    current_title = "Notebook 初始化"

    for cell_index, cell in enumerate(notebook.cells):
        system_kind = cell.get("metadata", {}).get("riskmodel_checker")
        if system_kind in _SYSTEM_STEP_TITLES and cell.cell_type == "code":
            step_id, title = _SYSTEM_STEP_TITLES[system_kind]
            _append_cell(
                steps_by_id,
                cell_to_step,
                step_id=step_id,
                title=title,
                cell_index=cell_index,
                source=cell.source,
                system=True,
            )
            continue

        if cell.cell_type == "markdown":
            heading = _first_heading(str(cell.source))
            if heading:
                current_title = heading
                current_step_id = f"step-{cell_index + 1}"
            continue

        if cell.cell_type != "code":
            continue
        _append_cell(
            steps_by_id,
            cell_to_step,
            step_id=current_step_id,
            title=current_title,
            cell_index=cell_index,
            source=cell.source,
            system=False,
        )

    return NotebookStepPlan(steps=list(steps_by_id.values()), cell_to_step=cell_to_step)


def notebook_step_preview(notebook_path: Path) -> list[dict]:
    notebook = nbformat.read(notebook_path, as_version=4)
    plan = notebook_step_plan(notebook)
    return [
        {
            "id": step.id,
            "step_order": order,
            "title": step.title,
            "status": "pending",
            "cell_count": len(step.cell_indexes),
            "cell_indexes": step.cell_indexes,
            "source_previews": step.source_previews,
            "system": step.system,
        }
        for order, step in enumerate(plan.steps, start=1)
    ]


def _first_heading(source: str) -> str | None:
    for line in source.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            return match.group(2).strip()
    return None


def _append_cell(
    steps_by_id: dict[str, NotebookStep],
    cell_to_step: dict[int, str],
    *,
    step_id: str,
    title: str,
    cell_index: int,
    source: str,
    system: bool,
) -> None:
    existing = steps_by_id.get(step_id)
    preview = _source_preview(source)
    if existing is None:
        steps_by_id[step_id] = NotebookStep(
            id=step_id,
            title=title,
            cell_indexes=[cell_index],
            source_previews=[preview],
            system=system,
        )
    elif system and existing.system:
        for existing_cell_index in existing.cell_indexes:
            cell_to_step.pop(existing_cell_index, None)
        steps_by_id[step_id] = NotebookStep(
            id=existing.id,
            title=existing.title,
            cell_indexes=[cell_index],
            source_previews=[preview],
            system=existing.system,
        )
    else:
        steps_by_id[step_id] = NotebookStep(
            id=existing.id,
            title=existing.title,
            cell_indexes=[*existing.cell_indexes, cell_index],
            source_previews=[*existing.source_previews, preview],
            system=existing.system,
        )
    cell_to_step[cell_index] = step_id


def _source_preview(source: str, limit: int = 120) -> str:
    first_line = next((line.strip() for line in source.splitlines() if line.strip()), "")
    return first_line[:limit]
