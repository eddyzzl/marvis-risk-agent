"""Task-scoped persistence for the V2 data and semantics workspace."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import hmac
import json
from pathlib import Path
import re
import sqlite3

from marvis.data.workspace import (
    DATA_WORKSPACE_SCHEMA_VERSION,
    DataWorkspaceDraft,
    DataWorkspaceSnapshot,
    data_semantic_mapping_from_dict,
    data_semantic_mapping_to_dict,
    data_workspace_draft_to_dict,
)
from marvis.db_schema import connect
from marvis.repositories.audit import _write_audit_row


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class DataWorkspaceRevisionConflict(RuntimeError):
    """The caller attempted to replace a stale workspace revision."""


class DataWorkspaceDataError(ValueError):
    """A supplied or persisted workspace violates the canonical contract."""


class DataWorkspaceDatasetNotFound(KeyError):
    """The requested active dataset does not exist."""


class DataWorkspaceDatasetMismatch(DataWorkspaceDataError):
    """The dataset is not task-owned or its registered content hash differs."""


class DataWorkspaceResetRequired(DataWorkspaceDataError):
    """A dataset change tried to retain choices from a previous generation."""


class DataWorkspaceRepository:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def get_or_default(self, task_id: str) -> DataWorkspaceSnapshot:
        normalized_task_id = _canonical_text(task_id, field_name="task_id")
        with connect(self.db_path) as conn:
            task = _task_row(conn, normalized_task_id)
            row = conn.execute(
                "SELECT * FROM data_workspaces WHERE task_id = ?",
                (normalized_task_id,),
            ).fetchone()
            if row is None:
                return _default_snapshot(task)
            snapshot = _snapshot_from_row(row)
            _validate_snapshot_against_dataset(conn, snapshot)
            return snapshot

    def save(
        self,
        task_id: str,
        draft: DataWorkspaceDraft,
        expected_revision: int,
        audit: dict | None = None,
    ) -> DataWorkspaceSnapshot:
        normalized_task_id = _canonical_text(task_id, field_name="task_id")
        expected = _non_negative_int(
            expected_revision,
            field_name="expected_revision",
        )
        if not isinstance(draft, DataWorkspaceDraft):
            raise DataWorkspaceDataError("draft must be a DataWorkspaceDraft")
        if audit is not None and not isinstance(audit, dict):
            raise DataWorkspaceDataError("audit must be an object")

        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            task = _task_row(conn, normalized_task_id)
            row = conn.execute(
                "SELECT * FROM data_workspaces WHERE task_id = ?",
                (normalized_task_id,),
            ).fetchone()
            if row is None:
                current = _default_snapshot(task)
            else:
                current = _snapshot_from_row(row)
                _validate_snapshot_against_dataset(conn, current)

            if current.revision != expected:
                raise DataWorkspaceRevisionConflict(
                    "stale data workspace revision: "
                    f"expected {expected}, found {current.revision}"
                )

            dataset_columns = _dataset_columns_for_draft(
                conn,
                task_id=normalized_task_id,
                draft=draft,
            )
            dataset_changed = (
                current.active_dataset_id != draft.active_dataset_id
                or current.active_dataset_content_hash
                != draft.active_dataset_content_hash
            )
            if dataset_changed and not _is_reset_payload(draft):
                raise DataWorkspaceResetRequired(
                    "active dataset change requires reset payload"
                )
            _validate_column_references(draft, dataset_columns)

            if _matches_draft(current, draft):
                return current

            new_revision = current.revision + 1
            new_generation = current.analysis_generation + int(dataset_changed)
            updated_at = _now()
            mapping_json = _canonical_json(
                data_semantic_mapping_to_dict(draft.semantic_mapping)
            )
            if row is None:
                conn.execute(
                    """
                    INSERT INTO data_workspaces(
                        task_id, schema_version, revision, active_dataset_id,
                        active_dataset_content_hash, analysis_generation, page,
                        selected_field, semantic_mapping_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        normalized_task_id,
                        DATA_WORKSPACE_SCHEMA_VERSION,
                        new_revision,
                        draft.active_dataset_id,
                        draft.active_dataset_content_hash,
                        new_generation,
                        draft.page,
                        draft.selected_field,
                        mapping_json,
                        updated_at,
                    ),
                )
            else:
                cursor = conn.execute(
                    """
                    UPDATE data_workspaces
                       SET schema_version = ?, revision = ?,
                           active_dataset_id = ?,
                           active_dataset_content_hash = ?,
                           analysis_generation = ?, page = ?,
                           selected_field = ?, semantic_mapping_json = ?,
                           updated_at = ?
                     WHERE task_id = ? AND revision = ?
                    """,
                    (
                        DATA_WORKSPACE_SCHEMA_VERSION,
                        new_revision,
                        draft.active_dataset_id,
                        draft.active_dataset_content_hash,
                        new_generation,
                        draft.page,
                        draft.selected_field,
                        mapping_json,
                        updated_at,
                        normalized_task_id,
                        current.revision,
                    ),
                )
                if cursor.rowcount != 1:
                    raise DataWorkspaceRevisionConflict(
                        "data workspace revision changed while saving"
                    )

            snapshot = DataWorkspaceSnapshot(
                task_id=normalized_task_id,
                updated_at=updated_at,
                revision=new_revision,
                active_dataset_id=draft.active_dataset_id,
                active_dataset_content_hash=draft.active_dataset_content_hash,
                analysis_generation=new_generation,
                page=draft.page,
                selected_field=draft.selected_field,
                semantic_mapping=draft.semantic_mapping,
            )
            _write_workspace_audit(
                conn,
                task_id=normalized_task_id,
                snapshot=snapshot,
                draft=draft,
                audit=audit,
            )
            return snapshot

    def activate_derived(
        self,
        task_id: str,
        *,
        expected_revision: int,
        source_dataset_id: str,
        source_dataset_content_hash: str,
        result_dataset_id: str,
        result_dataset_content_hash: str,
        semantic_mapping,
        page: str = "overview",
        selected_field: str | None = None,
        audit: dict | None = None,
    ) -> DataWorkspaceSnapshot:
        """Atomically replace the active dataset with a verified derived dataset.

        Ordinary ``save`` deliberately requires a full semantic reset whenever
        the physical dataset changes.  A governed transform is the sole safe
        exception: its caller supplies an explicitly migrated semantic mapping,
        and this method binds that migration to the exact source workspace
        revision and content hash before activating the result.
        """

        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            return self.activate_derived_on_connection(
                conn,
                task_id,
                expected_revision=expected_revision,
                source_dataset_id=source_dataset_id,
                source_dataset_content_hash=source_dataset_content_hash,
                result_dataset_id=result_dataset_id,
                result_dataset_content_hash=result_dataset_content_hash,
                semantic_mapping=semantic_mapping,
                page=page,
                selected_field=selected_field,
                audit=audit,
            )

    def activate_derived_on_connection(
        self,
        conn: sqlite3.Connection,
        task_id: str,
        *,
        expected_revision: int,
        source_dataset_id: str,
        source_dataset_content_hash: str,
        result_dataset_id: str,
        result_dataset_content_hash: str,
        semantic_mapping,
        page: str = "overview",
        selected_field: str | None = None,
        audit: dict | None = None,
    ) -> DataWorkspaceSnapshot:
        """Connection-scoped variant for a dataset/run/artifact unit of work."""

        normalized_task_id = _canonical_text(task_id, field_name="task_id")
        expected = _non_negative_int(
            expected_revision,
            field_name="expected_revision",
        )
        source_id = _canonical_text(
            source_dataset_id,
            field_name="source_dataset_id",
        )
        source_hash = _canonical_hash(
            source_dataset_content_hash,
            field_name="source_dataset_content_hash",
        )
        result_id = _canonical_text(
            result_dataset_id,
            field_name="result_dataset_id",
        )
        result_hash = _canonical_hash(
            result_dataset_content_hash,
            field_name="result_dataset_content_hash",
        )
        if audit is not None and not isinstance(audit, dict):
            raise DataWorkspaceDataError("audit must be an object")
        if audit is not None:
            unexpected = set(audit) - {"actor", "detail"}
            if unexpected:
                raise DataWorkspaceDataError(
                    "audit may only customize actor and detail"
                )
            if not isinstance(audit.get("detail", {}), dict):
                raise DataWorkspaceDataError("audit detail must be an object")
        try:
            draft = DataWorkspaceDraft(
                active_dataset_id=result_id,
                active_dataset_content_hash=result_hash,
                page=page,
                selected_field=selected_field,
                semantic_mapping=semantic_mapping,
            )
        except (TypeError, ValueError) as exc:
            raise DataWorkspaceDataError(str(exc)) from exc

        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")
        _task_row(conn, normalized_task_id)
        row = conn.execute(
            "SELECT * FROM data_workspaces WHERE task_id = ?",
            (normalized_task_id,),
        ).fetchone()
        if row is None:
            raise DataWorkspaceRevisionConflict(
                "derived activation requires a persisted source data workspace"
            )
        current = _snapshot_from_row(row)
        _validate_snapshot_against_dataset(conn, current)
        if current.revision != expected:
            raise DataWorkspaceRevisionConflict(
                "stale data workspace revision: "
                f"expected {expected}, found {current.revision}"
            )
        if current.active_dataset_id != source_id or not (
            isinstance(current.active_dataset_content_hash, str)
            and hmac.compare_digest(current.active_dataset_content_hash, source_hash)
        ):
            raise DataWorkspaceRevisionConflict(
                "data workspace source dataset changed before derived activation"
            )
        if source_id == result_id:
            raise DataWorkspaceDataError(
                "derived result dataset must differ from its source dataset"
            )

        result_columns = _dataset_columns_for_draft(
            conn,
            task_id=normalized_task_id,
            draft=draft,
        )
        _validate_column_references(draft, result_columns)

        new_revision = current.revision + 1
        new_generation = current.analysis_generation + 1
        updated_at = _now()
        mapping_json = _canonical_json(
            data_semantic_mapping_to_dict(draft.semantic_mapping)
        )
        cursor = conn.execute(
            """
            UPDATE data_workspaces
               SET schema_version = ?, revision = ?,
                   active_dataset_id = ?, active_dataset_content_hash = ?,
                   analysis_generation = ?, page = ?, selected_field = ?,
                   semantic_mapping_json = ?, updated_at = ?
             WHERE task_id = ?
               AND revision = ?
               AND active_dataset_id = ?
               AND active_dataset_content_hash = ?
            """,
            (
                DATA_WORKSPACE_SCHEMA_VERSION,
                new_revision,
                result_id,
                result_hash,
                new_generation,
                draft.page,
                draft.selected_field,
                mapping_json,
                updated_at,
                normalized_task_id,
                current.revision,
                source_id,
                source_hash,
            ),
        )
        if cursor.rowcount != 1:
            raise DataWorkspaceRevisionConflict(
                "data workspace changed while activating derived dataset"
            )

        snapshot = DataWorkspaceSnapshot(
            task_id=normalized_task_id,
            updated_at=updated_at,
            revision=new_revision,
            active_dataset_id=result_id,
            active_dataset_content_hash=result_hash,
            analysis_generation=new_generation,
            page=draft.page,
            selected_field=draft.selected_field,
            semantic_mapping=draft.semantic_mapping,
        )
        audit_detail = {"source_dataset_id": source_id}
        if audit is not None:
            audit_detail.update(audit.get("detail", {}))
        derived_audit = {
            "actor": (audit or {}).get("actor", "system"),
            "detail": audit_detail,
        }
        _write_workspace_audit(
            conn,
            task_id=normalized_task_id,
            snapshot=snapshot,
            draft=draft,
            audit=derived_audit,
            kind="data.workspace.derived.activate",
        )
        return snapshot


def _task_row(conn: sqlite3.Connection, task_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT id, created_at FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"Task not found: {task_id}")
    return row


def _default_snapshot(task: sqlite3.Row) -> DataWorkspaceSnapshot:
    task_id = str(task["id"])
    try:
        return DataWorkspaceSnapshot(
            task_id=task_id,
            updated_at=task["created_at"],
        )
    except (IndexError, TypeError, ValueError) as exc:
        raise DataWorkspaceDataError(
            f"corrupt task timestamp for data workspace: {task_id}"
        ) from exc


def _snapshot_from_row(row: sqlite3.Row) -> DataWorkspaceSnapshot:
    task_id = str(row["task_id"])
    try:
        raw_mapping = row["semantic_mapping_json"]
        if not isinstance(raw_mapping, str):
            raise ValueError("semantic_mapping_json must be text")
        mapping_payload = json.loads(raw_mapping)
        mapping = data_semantic_mapping_from_dict(mapping_payload)
        if raw_mapping != _canonical_json(data_semantic_mapping_to_dict(mapping)):
            raise ValueError("semantic_mapping_json is not canonical")
        return DataWorkspaceSnapshot(
            schema_version=row["schema_version"],
            task_id=task_id,
            revision=row["revision"],
            active_dataset_id=row["active_dataset_id"],
            active_dataset_content_hash=row["active_dataset_content_hash"],
            analysis_generation=row["analysis_generation"],
            page=row["page"],
            selected_field=row["selected_field"],
            semantic_mapping=mapping,
            updated_at=row["updated_at"],
        )
    except (json.JSONDecodeError, IndexError, KeyError, TypeError, ValueError) as exc:
        raise DataWorkspaceDataError(
            f"corrupt data workspace record for task {task_id}"
        ) from exc


def _validate_snapshot_against_dataset(
    conn: sqlite3.Connection,
    snapshot: DataWorkspaceSnapshot,
) -> None:
    draft = DataWorkspaceDraft(
        active_dataset_id=snapshot.active_dataset_id,
        active_dataset_content_hash=snapshot.active_dataset_content_hash,
        page=snapshot.page,
        selected_field=snapshot.selected_field,
        semantic_mapping=snapshot.semantic_mapping,
    )
    columns = _dataset_columns_for_draft(
        conn,
        task_id=snapshot.task_id,
        draft=draft,
    )
    _validate_column_references(draft, columns)


def _dataset_columns_for_draft(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    draft: DataWorkspaceDraft,
) -> frozenset[str] | None:
    dataset_id = draft.active_dataset_id
    content_hash = draft.active_dataset_content_hash
    if dataset_id is None:
        return None
    row = conn.execute(
        "SELECT id, task_id, content_hash, columns_json FROM datasets WHERE id = ?",
        (dataset_id,),
    ).fetchone()
    if row is None:
        raise DataWorkspaceDatasetNotFound(f"dataset not found: {dataset_id}")
    owner_task_id = str(row["task_id"])
    if owner_task_id != task_id:
        raise DataWorkspaceDatasetMismatch(
            f"dataset {dataset_id} belongs to task {owner_task_id}, not {task_id}"
        )
    registered_hash = row["content_hash"]
    if (
        not isinstance(registered_hash, str)
        or _SHA256_RE.fullmatch(registered_hash) is None
    ):
        raise DataWorkspaceDatasetMismatch(
            f"dataset {dataset_id} has no verified content_hash"
        )
    if content_hash is None or not hmac.compare_digest(registered_hash, content_hash):
        raise DataWorkspaceDatasetMismatch(
            "active dataset content hash does not match registered content_hash"
        )
    return _dataset_columns(row["columns_json"], dataset_id=dataset_id)


def _dataset_columns(raw: object, *, dataset_id: str) -> frozenset[str]:
    try:
        if not isinstance(raw, str):
            raise TypeError("columns_json must be text")
        payload = json.loads(raw)
        if not isinstance(payload, list):
            raise TypeError("columns_json must be an array")
        columns: list[str] = []
        for item in payload:
            if not isinstance(item, dict):
                raise TypeError("column profile must be an object")
            columns.append(_canonical_text(item.get("name"), field_name="column name"))
        if len(columns) != len(set(columns)):
            raise ValueError("dataset contains duplicate column names")
        return frozenset(columns)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise DataWorkspaceDataError(
            f"dataset {dataset_id} has corrupt columns_json"
        ) from exc


def _validate_column_references(
    draft: DataWorkspaceDraft,
    columns: frozenset[str] | None,
) -> None:
    mapping = draft.semantic_mapping
    references = set(mapping.field_roles) | set(mapping.business_names)
    if mapping.target_col is not None:
        references.add(mapping.target_col)
    if draft.selected_field is not None:
        references.add(draft.selected_field)
    if columns is None:
        if references:
            raise DataWorkspaceDataError(
                "field selections require an active dataset"
            )
        return
    unknown = sorted(references - columns)
    if unknown:
        raise DataWorkspaceDataError(
            "workspace references unknown dataset column(s): " + ", ".join(unknown)
        )


def _is_reset_payload(draft: DataWorkspaceDraft) -> bool:
    mapping = draft.semantic_mapping
    return (
        draft.page == "overview"
        and draft.selected_field is None
        and mapping.target_col is None
        and not mapping.field_roles
        and not mapping.business_names
    )


def _matches_draft(
    snapshot: DataWorkspaceSnapshot,
    draft: DataWorkspaceDraft,
) -> bool:
    return (
        snapshot.active_dataset_id == draft.active_dataset_id
        and snapshot.active_dataset_content_hash
        == draft.active_dataset_content_hash
        and snapshot.page == draft.page
        and snapshot.selected_field == draft.selected_field
        and snapshot.semantic_mapping == draft.semantic_mapping
    )


def _write_workspace_audit(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    snapshot: DataWorkspaceSnapshot,
    draft: DataWorkspaceDraft,
    audit: dict | None,
    kind: str = "data.workspace.update",
) -> None:
    draft_json = _canonical_json(data_workspace_draft_to_dict(draft))
    inputs_hash = hashlib.sha256(draft_json.encode("utf-8")).hexdigest()
    actor = "system"
    extra_detail: dict = {}
    if audit is not None:
        unexpected = set(audit) - {"actor", "detail"}
        if unexpected:
            raise DataWorkspaceDataError(
                "audit may only customize actor and detail"
            )
        if "actor" in audit:
            actor = _canonical_text(audit["actor"], field_name="audit actor")
        supplied_detail = audit.get("detail", {})
        if not isinstance(supplied_detail, dict):
            raise DataWorkspaceDataError("audit detail must be an object")
        extra_detail = dict(supplied_detail)
    detail = {
        **extra_detail,
        "task_id": task_id,
        "revision": snapshot.revision,
        "analysis_generation": snapshot.analysis_generation,
        "active_dataset_id": snapshot.active_dataset_id,
        "inputs_hash": inputs_hash,
    }
    _write_audit_row(
        conn,
        kind=kind,
        target_ref=task_id,
        actor=actor,
        inputs_hash=inputs_hash,
        outcome="succeeded",
        detail=detail,
    )


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise DataWorkspaceDataError("workspace payload is not canonical JSON") from exc


def _canonical_text(value: object, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or value == ""
        or value != value.strip()
        or "\x00" in value
    ):
        raise DataWorkspaceDataError(
            f"{field_name} must be canonical non-empty text"
        )
    return value


def _canonical_hash(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DataWorkspaceDataError(
            f"{field_name} must be a 64-character lowercase SHA-256 hex"
        )
    return value


def _non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DataWorkspaceDataError(
            f"{field_name} must be a non-negative integer"
        )
    return value


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "DataWorkspaceDataError",
    "DataWorkspaceDatasetMismatch",
    "DataWorkspaceDatasetNotFound",
    "DataWorkspaceRepository",
    "DataWorkspaceResetRequired",
    "DataWorkspaceRevisionConflict",
]
