"""Setup (slot-filling) for the vintage risk-analysis task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from marvis.agent.sample_setup import detect_setup
from marvis.domain import FileRole
from marvis.files import scan_source_dir

_DATA_ROLES = frozenset({FileRole.SAMPLE.value, "sample", "strategy_sample"})
_COHORT_HINTS = (
    "cohort",
    "vintage",
    "origination_month",
    "orig_month",
    "apply_month",
    "application_month",
    "month",
)
_MOB_HINTS = (
    "mob",
    "month_on_book",
    "months_on_book",
    "loan_age",
    "age_month",
    "observe_mob",
)


class VintageSetupError(ValueError):
    """Raised when a vintage task cannot infer the required columns."""


@dataclass
class VintageProposal:
    dataset_id: str
    dataset_name: str
    cohort_col: str
    mob_col: str
    bad_col: str
    mob_max: int = 12
    ref_mob: int = 6
    template_id: str = "vintage_analysis"

    def template_slots(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "cohort_col": self.cohort_col,
            "mob_col": self.mob_col,
            "bad_col": self.bad_col,
            "mob_max": self.mob_max,
            "ref_mob": self.ref_mob,
        }


def build_vintage_proposal(
    registry,
    backend,
    task_id: str,
    source_dir,
    *,
    target_col: str | None = None,
    time_col: str | None = None,
) -> VintageProposal:
    dataset = _resolve_dataset(registry, task_id, source_dir)
    path = registry.resolve_path(dataset.id)
    columns = backend.column_names(path)
    cohort_col = _resolve_named_col(columns, time_col, _COHORT_HINTS)
    if not cohort_col:
        raise VintageSetupError("未能识别 cohort/放款月份列;请在创建任务时用 time_col 指定。")
    mob_col = _resolve_named_col(columns, None, _MOB_HINTS)
    if not mob_col:
        raise VintageSetupError("未能识别 MOB(月龄)列;请确认数据包含 mob/month_on_book 等字段。")
    bad_col = _resolve_bad_col(backend, path, columns, target_col)
    return VintageProposal(
        dataset_id=dataset.id,
        dataset_name=_dataset_name(dataset),
        cohort_col=cohort_col,
        mob_col=mob_col,
        bad_col=bad_col,
    )


def _resolve_dataset(registry, task_id: str, source_dir):
    datasets = [d for d in registry.list_for_task(task_id) if d.role in _DATA_ROLES]
    if not datasets and source_dir is not None:
        for artifact in scan_source_dir(Path(source_dir)):
            if artifact.role == FileRole.SAMPLE:
                registry.register_from_upload(task_id, Path(artifact.path), role="sample")
        datasets = [d for d in registry.list_for_task(task_id) if d.role in _DATA_ROLES]
    if not datasets:
        raise VintageSetupError(f"风险分析未找到数据文件:{source_dir}")
    return sorted(
        datasets,
        key=lambda d: (not bool(getattr(d, "has_target", False)), -int(getattr(d, "row_count", 0) or 0)),
    )[0]


def _resolve_bad_col(backend, path: Path, columns: list[str], requested: str | None) -> str:
    requested = str(requested or "").strip()
    if requested and requested in columns:
        return requested
    setup = detect_setup(backend, path)
    if setup.target_col:
        return setup.target_col
    raise VintageSetupError("未能识别 0/1 坏账标签列;请在创建任务时指定 target_col。")


def _resolve_named_col(columns: list[str], requested: str | None, hints: tuple[str, ...]) -> str:
    requested = str(requested or "").strip()
    if requested and requested in columns:
        return requested
    lowered = {column.lower(): column for column in columns}
    for hint in hints:
        if hint in lowered:
            return lowered[hint]
    for column in columns:
        low = column.lower()
        if any(hint in low for hint in hints):
            return column
    return ""


def _dataset_name(dataset) -> str:
    source = getattr(dataset, "source_path", None)
    return Path(source).name if source else str(getattr(dataset, "id", ""))


__all__ = ["VintageProposal", "VintageSetupError", "build_vintage_proposal"]
