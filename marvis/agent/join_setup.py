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

from marvis.agent.data_setup import reconcile_source_data_tables
from marvis.data.data_dictionary import resolve_data_dictionary_id
from marvis.domain import FileRole

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
    target_candidates: list[str] = field(default_factory=list)


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
    ingest_notices: list[dict] = field(default_factory=list)


def build_join_proposal(registry, task_id: str, source_dir) -> JoinProposal:
    """Discover/register the task's data files and propose C1 roles + target.

    Unlike :func:`discover_join_inputs` this never raises on a single file — it
    returns ``skip=True`` so the driver can confirm the sample + target and then
    skip the join stage."""
    datasets = reconcile_source_data_tables(
        registry,
        task_id,
        source_dir,
        accepted_roles=_DATA_ROLES,
        registered_role="feature",
    )
    # GAP-4: register a data-dictionary material as a dataset (if present) the
    # same way the modeling setup flow does, so a dictionary uploaded for a
    # data_join task is available to downstream consumers too. Best-effort —
    # never blocks the C1 proposal when no dictionary file exists.
    resolve_data_dictionary_id(registry, task_id, source_dir)
    ranked = propose_roles(datasets)
    files: list[JoinFileInfo] = []
    for index, dataset in enumerate(ranked):
        target_candidates = _strong_target_candidates(dataset)
        files.append(JoinFileInfo(
            dataset_id=dataset.id,
            name=_dataset_name(dataset),
            row_count=int(getattr(dataset, "row_count", 0) or 0),
            n_cols=len(_column_names(dataset)),
            has_target=bool(target_candidates) or bool(getattr(dataset, "has_target", False)),
            candidate_target=_unique_target_candidate(dataset, target_candidates),
            proposed_role="anchor" if index == 0 else "feature",
            columns=_proposal_columns(dataset, target_candidates),
            target_candidates=target_candidates,
        ))
    anchor = ranked[0] if ranked else None
    anchor_targets = _strong_target_candidates(anchor) if anchor else []
    return JoinProposal(
        files=files,
        anchor_id=anchor.id if anchor else None,
        feature_ids=[d.id for d in ranked[1:]],
        target_col=(
            _unique_target_candidate(anchor, anchor_targets)
            if anchor is not None
            else None
        ),
        skip=len(ranked) < 2,
        ingest_notices=_consume_ingest_notices(registry, task_id),
    )


def _consume_ingest_notices(registry, task_id: str) -> list[dict]:
    consume = getattr(registry, "consume_ingest_notices", None)
    return list(consume(task_id)) if callable(consume) else []


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
    datasets = reconcile_source_data_tables(
        registry,
        task_id,
        source_dir,
        accepted_roles=_DATA_ROLES,
        registered_role="feature",
    )
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
        key=lambda d: (
            not (bool(_strong_target_candidates(d)) or bool(getattr(d, "has_target", False))),
            -int(getattr(d, "row_count", 0) or 0),
        ),
    )


def _strong_target_candidates(dataset) -> list[str]:
    """Return binary target-like profiles in source-column order.

    The registry intentionally stores at most one ``target_col``.  C1 needs the
    full candidate set so two competing bad definitions remain an explicit
    user decision.
    """

    candidates: list[str] = []
    for profile in getattr(dataset, "columns", None) or []:
        semantic_role = getattr(profile, "semantic_role", None)
        cardinality = getattr(profile, "cardinality", None)
        samples = getattr(profile, "sample_values", None) or ()
        if semantic_role != "target" or cardinality != 2:
            continue
        if samples and any(_binary_candidate_value(value) is None for value in samples):
            continue
        name = getattr(profile, "name", None)
        if name:
            candidates.append(str(name))
    return candidates


def _binary_candidate_value(value) -> int | None:
    if isinstance(value, bool):
        return int(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number in {0, 1} else None


def _unique_target_candidate(dataset, candidates: list[str]) -> str | None:
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return None
    # Compatibility for persisted/legacy datasets that predate column-profile
    # target metadata.  New profiled datasets always take the validated path.
    return getattr(dataset, "target_col", None) if getattr(dataset, "has_target", False) else None


def _proposal_columns(dataset, target_candidates: list[str]) -> list[str]:
    """Keep C1 metadata bounded without dropping tail-positioned labels."""

    columns = _column_names(dataset)
    if len(columns) <= _MAX_PROPOSAL_COLUMNS:
        return columns
    targets = list(dict.fromkeys(target_candidates))
    if len(targets) >= _MAX_PROPOSAL_COLUMNS:
        return targets[:_MAX_PROPOSAL_COLUMNS]
    target_set = set(targets)
    ordinary = [name for name in columns if name not in target_set]
    # Put the governed choices first so a native select/search opens on labels,
    # not after hundreds of unrelated feature names.
    return targets + ordinary[: _MAX_PROPOSAL_COLUMNS - len(targets)]


def _data_datasets(registry, task_id: str):
    return [d for d in registry.list_for_task(task_id) if d.role in _DATA_ROLES]


__all__ = [
    "discover_join_inputs",
    "propose_roles",
    "build_join_proposal",
    "JoinProposal",
    "JoinFileInfo",
    "JoinSetupError",
]
