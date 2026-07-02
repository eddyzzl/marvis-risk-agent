"""Setup (slot-filling) for the data_join task.

The driver needs the join template's slots filled before it can build a plan:
which registered dataset is the *anchor* (the sample table whose rows are kept
1:1) and which are *feature* tables to left-join. This module discovers the
task's data files, registers any not yet registered, and proposes roles:

    anchor  = the dataset that carries a target/label (the sample), else the
              largest by row count;
    features = every other data dataset.

The proposal is deterministic and conservative; the C2 diagnostics gate in the
plan (and, later, a manual C1 role-assignment control) lets the user correct it
before any join executes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from marvis.domain import FileRole
from marvis.files import scan_source_dir

# Dataset roles that represent join-able data tables (not dictionaries/notebooks).
_DATA_ROLES = frozenset({FileRole.SAMPLE.value, "sample", "feature"})
# Cap column names carried in the C1 proposal (target dropdown) to keep the
# stored message metadata small for very wide sample tables.
_MAX_PROPOSAL_COLUMNS = 300


class JoinSetupError(ValueError):
    """Raised when the task does not have enough data files to join."""


@dataclass
class JoinFileInfo:
    dataset_id: str
    name: str
    row_count: int
    n_cols: int
    has_target: bool
    candidate_target: str | None
    proposed_role: str  # "anchor" | "feature"
    columns: list[str] = field(default_factory=list)


@dataclass
class JoinProposal:
    """C1 proposal: every data file with a proposed anchor/feature role + target.

    ``skip`` is True when there is ≤1 data table (nothing to join — the single
    table is already the result, but the sample + target are still confirmed)."""

    files: list[JoinFileInfo]
    anchor_id: str | None
    feature_ids: list[str]
    target_col: str | None
    skip: bool


def build_join_proposal(registry, task_id: str, source_dir) -> JoinProposal:
    """Discover/register the task's data files and propose C1 roles + target.

    Unlike :func:`discover_join_inputs` this never raises on a single file — it
    returns ``skip=True`` so the driver can confirm the sample + target and then
    skip the join stage."""
    datasets = _data_datasets(registry, task_id)
    if len(datasets) < 2 and source_dir is not None:
        for artifact in scan_source_dir(Path(source_dir)):
            if artifact.role == FileRole.SAMPLE:
                _register_once(registry, task_id, Path(artifact.path))
        datasets = _data_datasets(registry, task_id)
    ranked = propose_roles(datasets)
    files = [
        JoinFileInfo(
            dataset_id=d.id,
            name=_dataset_name(d),
            row_count=int(getattr(d, "row_count", 0) or 0),
            n_cols=len(_column_names(d)),
            has_target=bool(getattr(d, "has_target", False)),
            candidate_target=getattr(d, "target_col", None),
            proposed_role="anchor" if index == 0 else "feature",
            columns=_column_names(d)[:_MAX_PROPOSAL_COLUMNS],
        )
        for index, d in enumerate(ranked)
    ]
    anchor = ranked[0] if ranked else None
    return JoinProposal(
        files=files,
        anchor_id=anchor.id if anchor else None,
        feature_ids=[d.id for d in ranked[1:]],
        target_col=getattr(anchor, "target_col", None) if anchor else None,
        skip=len(ranked) < 2,
    )


def _dataset_name(dataset) -> str:
    source = getattr(dataset, "source_path", None)
    return Path(source).name if source else str(getattr(dataset, "id", ""))


def _column_names(dataset) -> list[str]:
    out = []
    for column in getattr(dataset, "columns", None) or []:
        name = getattr(column, "name", None)
        if name is None and isinstance(column, dict):
            name = column.get("name")
        if name:
            out.append(str(name))
    return out


def discover_join_inputs(registry, task_id: str, source_dir) -> tuple[str, list[str]]:
    """Return (anchor_dataset_id, [feature_dataset_id, ...]) for the task.

    Registers data files found under ``source_dir`` on first use. Raises
    :class:`JoinSetupError` if fewer than two data tables are available.
    """
    datasets = _data_datasets(registry, task_id)
    if len(datasets) < 2 and source_dir is not None:
        for artifact in scan_source_dir(Path(source_dir)):
            if artifact.role == FileRole.SAMPLE:
                _register_once(registry, task_id, Path(artifact.path))
        datasets = _data_datasets(registry, task_id)
    if len(datasets) < 2:
        raise JoinSetupError(
            "数据拼接至少需要 2 个数据文件（1 个锚样本 + ≥1 个特征表），"
            f"当前只发现 {len(datasets)} 个:{source_dir}"
        )
    anchor, *features = propose_roles(datasets)
    return anchor.id, [d.id for d in features]


def propose_roles(datasets):
    """Order datasets anchor-first: prefer one carrying a target, then most rows."""
    return sorted(
        datasets,
        key=lambda d: (not bool(getattr(d, "has_target", False)), -int(getattr(d, "row_count", 0) or 0)),
    )


def _data_datasets(registry, task_id: str):
    return [d for d in registry.list_for_task(task_id) if d.role in _DATA_ROLES]


def _register_once(registry, task_id: str, path: Path) -> None:
    for existing in registry.list_for_task(task_id):
        source = getattr(existing, "source_path", None)
        if source and Path(source) == path:
            return
    registry.register_from_upload(task_id, path, role="feature")


__all__ = [
    "discover_join_inputs",
    "propose_roles",
    "build_join_proposal",
    "JoinProposal",
    "JoinFileInfo",
    "JoinSetupError",
]
