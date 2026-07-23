"""Shared source-folder reconciliation for JOIN-composed workflows."""

from __future__ import annotations

from pathlib import Path

from marvis.files import scan_data_workflow_dir, sha256_file


def reconcile_source_data_tables(
    registry,
    task_id: str,
    source_dir,
    *,
    accepted_roles,
    registered_role: str,
):
    """Register missing source tables and return the task's joinable datasets.

    The task can already contain a partial registration from an earlier start.
    Reconciliation is multiplicity-aware, so ``vars.csv`` plus ``vars.xlsx``
    remains two inputs even though both normalize to ``vars_<id>.parquet``.
    """

    datasets = [
        dataset
        for dataset in registry.list_for_task(task_id)
        if dataset.role in accepted_roles
    ]
    if source_dir is None:
        return datasets

    artifacts = scan_data_workflow_dir(Path(source_dir))
    represented = [
        identity
        for dataset in datasets
        if (identity := _registered_source_identity(registry, dataset)) is not None
    ]
    missing = []
    for artifact in artifacts:
        identity = _artifact_source_identity(artifact)
        match_index = next(
            (
                index
                for index, existing in enumerate(represented)
                if _same_source_identity(existing, identity)
            ),
            None,
        )
        if match_index is None:
            missing.append(artifact)
        else:
            represented.pop(match_index)

    for artifact in missing:
        registry.register_from_upload(
            task_id,
            Path(artifact.path),
            role=registered_role,
        )
    return [
        dataset
        for dataset in registry.list_for_task(task_id)
        if dataset.role in accepted_roles
    ]


def _registered_source_identity(registry, dataset) -> dict | None:
    lookup = getattr(registry, "source_identity", None)
    if not callable(lookup):
        return None
    try:
        identity = lookup(str(dataset.id))
    except (KeyError, OSError, ValueError):
        return None
    return dict(identity) if isinstance(identity, dict) else None


def _artifact_source_identity(artifact) -> dict[str, str]:
    path = Path(artifact.path).resolve()
    return {
        "resolved_path": str(path),
        "original_name": path.name,
        "suffix": path.suffix.lower(),
        "sha256": str(getattr(artifact, "sha256", None) or sha256_file(path)),
    }


def _same_source_identity(left: dict, right: dict) -> bool:
    same_hash = (
        bool(str(left.get("sha256") or ""))
        and str(left.get("sha256") or "") == str(right.get("sha256") or "")
    )
    if not same_hash:
        return False
    same_path = (
        str(left.get("resolved_path") or "")
        == str(right.get("resolved_path") or "")
    )
    # Relocated task folders can still reconcile when the exact original name,
    # suffix and bytes agree.  Names remain part of identity, so same-content
    # aliases are not collapsed into one upload.
    same_name_and_suffix = all(
        str(left.get(key) or "") == str(right.get(key) or "")
        for key in ("original_name", "suffix")
    )
    return same_path or same_name_and_suffix


__all__ = ["reconcile_source_data_tables"]
