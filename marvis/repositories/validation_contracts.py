from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from typing import Any

from marvis.db_schema import connect
from marvis.files import sha256_file
from marvis.repositories.audit import _write_audit_row
from marvis.repositories.tasks import _now
from marvis.validation.input_contracts import (
    FeatureMetadataResolution,
    SampleSchema,
    ValidationInputConfirmation,
    ValidationInputContract,
    input_contract_from_dict,
    input_contract_to_dict,
    validation_confirmation_to_dict,
)

_MATERIAL_TASK_FIELDS = (
    ("notebook", "notebook_path"),
    ("sample", "sample_path"),
    ("pmml", "pmml_path"),
    ("dictionary", "dictionary_path"),
)
_CANDIDATE_KEYS = frozenset({"candidates", "transformations", "conflicts"})


@dataclass(frozen=True)
class ValidationInputContractRecord:
    task_id: str
    revision: int
    status: str
    contract: ValidationInputContract
    created_at: str
    updated_at: str

    def to_api_payload(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "revision": self.revision,
            "status": self.status,
            "needs_confirmation": self.status == "pending_confirmation",
            "read_only": True,
            "contract": input_contract_to_dict(self.contract),
        }


class ValidationContractRevisionConflict(RuntimeError):
    pass


class ValidationContractDataError(RuntimeError):
    pass


class ValidationContractMaterialMismatch(ValueError):
    pass


class ValidationContractRepository:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def get(self, task_id: str) -> ValidationInputContractRecord | None:
        return _read_contract_record(self.db_path, task_id)

    def replace_candidates(
        self, task_id: str, contract: ValidationInputContract
    ) -> ValidationInputContractRecord:
        return _replace_contract_candidates(self.db_path, task_id, contract)

    def confirm(
        self,
        task_id: str,
        confirmation: ValidationInputConfirmation,
        *,
        expected_revision: int,
        resolved_sample_schema: SampleSchema | None = None,
        resolved_feature_metadata: FeatureMetadataResolution | None = None,
    ) -> ValidationInputContractRecord:
        return _confirm_contract(
            self.db_path,
            task_id,
            confirmation,
            expected_revision=expected_revision,
            resolved_sample_schema=resolved_sample_schema,
            resolved_feature_metadata=resolved_feature_metadata,
        )


def _read_contract_record(
    db_path: Path, task_id: str
) -> ValidationInputContractRecord | None:
    with connect(db_path) as conn:
        conn.execute("BEGIN")
        row = conn.execute(
            "SELECT * FROM validation_input_contracts WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_contract_record(row)


def _replace_contract_candidates(
    db_path: Path,
    task_id: str,
    contract: ValidationInputContract,
) -> ValidationInputContractRecord:
    validated = _validate_contract(contract)
    candidate_status = (
        "blocked" if validated.status == "blocked" else "pending_confirmation"
    )
    candidate = replace(validated, status=candidate_status, confirmed={})
    encoded = _encode_contract_columns(candidate)
    now = _now()
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        task_row = conn.execute(
            "SELECT id FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if task_row is None:
            raise KeyError(f"Task not found: {task_id}")
        previous = conn.execute(
            "SELECT revision, created_at FROM validation_input_contracts WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        if previous is None:
            revision = 1
            created_at = now
            conn.execute(
                """
                INSERT INTO validation_input_contracts(
                    task_id, schema_version, revision, status, candidate_json,
                    confirmed_json, material_hashes_json, sample_schema_json,
                    pmml_manifest_json, metadata_resolution_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    candidate.schema_version,
                    revision,
                    candidate.status,
                    encoded["candidate_json"],
                    encoded["confirmed_json"],
                    encoded["material_hashes_json"],
                    encoded["sample_schema_json"],
                    encoded["pmml_manifest_json"],
                    encoded["metadata_resolution_json"],
                    created_at,
                    now,
                ),
            )
        else:
            revision = int(previous["revision"]) + 1
            created_at = str(previous["created_at"])
            conn.execute(
                """
                UPDATE validation_input_contracts
                   SET schema_version = ?, revision = ?, status = ?,
                       candidate_json = ?, confirmed_json = ?,
                       material_hashes_json = ?, sample_schema_json = ?,
                       pmml_manifest_json = ?, metadata_resolution_json = ?,
                       updated_at = ?
                 WHERE task_id = ?
                """,
                (
                    candidate.schema_version,
                    revision,
                    candidate.status,
                    encoded["candidate_json"],
                    encoded["confirmed_json"],
                    encoded["material_hashes_json"],
                    encoded["sample_schema_json"],
                    encoded["pmml_manifest_json"],
                    encoded["metadata_resolution_json"],
                    now,
                    task_id,
                ),
            )
        return ValidationInputContractRecord(
            task_id=task_id,
            revision=revision,
            status=candidate.status,
            contract=candidate,
            created_at=created_at,
            updated_at=now,
        )


def _confirm_contract(
    db_path: Path,
    task_id: str,
    confirmation: ValidationInputConfirmation,
    *,
    expected_revision: int,
    resolved_sample_schema: SampleSchema | None,
    resolved_feature_metadata: FeatureMetadataResolution | None,
) -> ValidationInputContractRecord:
    if isinstance(expected_revision, bool) or not isinstance(expected_revision, int):
        raise ValueError("expected revision must be an integer")
    confirmation_payload = validation_confirmation_to_dict(confirmation)
    confirmation_payload.pop("transformations")
    transformations = tuple(confirmation.transformations)
    now = _now()
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        task_row = conn.execute(
            """
            SELECT id, source_dir, notebook_path, sample_path, pmml_path,
                   dictionary_path
              FROM tasks
             WHERE id = ?
            """,
            (task_id,),
        ).fetchone()
        if task_row is None:
            raise KeyError(f"Task not found: {task_id}")
        row = conn.execute(
            "SELECT * FROM validation_input_contracts WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Validation input contract not found: {task_id}")
        current = _row_to_contract_record(row)
        if current.revision != expected_revision:
            raise ValidationContractRevisionConflict(
                f"validation input contract revision conflict: expected "
                f"{expected_revision}, found {current.revision}"
            )
        if current.contract.status == "blocked":
            raise ValueError("blocked validation input contract cannot be confirmed")

        selected_paths = resolve_selected_material_paths(task_row)
        observed_hashes = {
            role: sha256_file(path) for role, path in selected_paths.items()
        }
        mismatches = sorted(
            role
            for role, observed in observed_hashes.items()
            if current.contract.material_hashes.get(role) != observed
        )
        if mismatches:
            raise ValidationContractMaterialMismatch(
                "selected validation materials changed; rescan before confirmation: "
                + ", ".join(mismatches)
            )

        sample_schema = resolved_sample_schema or current.contract.sample_schema
        feature_metadata = (
            resolved_feature_metadata or current.contract.feature_metadata
        )
        if sample_schema is None:
            raise ValueError("validation input contract has no sample schema")
        if feature_metadata is None:
            raise ValueError(
                "validation input contract has no resolved feature metadata"
            )
        manifest = current.contract.require_pmml_manifest()
        if current.contract.conflicts:
            raise ValueError("validation input contract has unresolved conflicts")
        confirmation_payload["algorithm"] = manifest.algorithm
        ready = _validate_contract(
            replace(
                current.contract,
                status="ready",
                confirmed=confirmation_payload,
                transformations=transformations,
                sample_schema=sample_schema,
                feature_metadata=feature_metadata,
            )
        )
        encoded = _encode_contract_columns(ready)
        new_revision = current.revision + 1
        cursor = conn.execute(
            """
            UPDATE validation_input_contracts
               SET schema_version = ?, revision = ?, status = ?,
                   candidate_json = ?, confirmed_json = ?,
                   material_hashes_json = ?, sample_schema_json = ?,
                   pmml_manifest_json = ?, metadata_resolution_json = ?,
                   updated_at = ?
             WHERE task_id = ? AND revision = ?
            """,
            (
                ready.schema_version,
                new_revision,
                ready.status,
                encoded["candidate_json"],
                encoded["confirmed_json"],
                encoded["material_hashes_json"],
                encoded["sample_schema_json"],
                encoded["pmml_manifest_json"],
                encoded["metadata_resolution_json"],
                now,
                task_id,
                current.revision,
            ),
        )
        if cursor.rowcount == 0:
            raise ValidationContractRevisionConflict(
                "validation input contract revision changed during confirmation"
            )
        task_cursor = conn.execute(
            """
            UPDATE tasks
               SET target_col = ?, split_col = ?, time_col = ?, algorithm = ?,
                   updated_at = ?
             WHERE id = ?
            """,
            (
                confirmation.target_col,
                confirmation.split_col,
                confirmation.time_col,
                ready.require_algorithm(),
                now,
                task_id,
            ),
        )
        if task_cursor.rowcount == 0:
            raise KeyError(f"Task not found: {task_id}")
        _write_audit_row(
            conn,
            kind="validation.input_contract.confirm",
            target_ref=task_id,
            outcome="succeeded",
            inputs_hash=sha256(
                _canonical_json(observed_hashes).encode("utf-8")
            ).hexdigest(),
            detail={
                "task_id": task_id,
                "revision": new_revision,
                "schema_version": ready.schema_version,
            },
        )
        return ValidationInputContractRecord(
            task_id=task_id,
            revision=new_revision,
            status=ready.status,
            contract=ready,
            created_at=current.created_at,
            updated_at=now,
        )


def resolve_selected_material_paths(task_row: sqlite3.Row) -> dict[str, Path]:
    raw_source_dir = Path(str(task_row["source_dir"])).expanduser()
    try:
        source_dir = raw_source_dir.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"task source_dir does not exist: {raw_source_dir}") from exc
    if not source_dir.is_dir():
        raise ValueError(f"task source_dir is not a directory: {source_dir}")

    resolved: dict[str, Path] = {}
    for role, field in _MATERIAL_TASK_FIELDS:
        selected = task_row[field]
        if selected is None or not str(selected).strip():
            raise ValueError(f"selected {role} path is required")
        raw_path = Path(str(selected)).expanduser()
        candidate = raw_path if raw_path.is_absolute() else source_dir / raw_path
        try:
            path = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ValueError(f"selected {role} does not exist: {selected}") from exc
        try:
            path.relative_to(source_dir)
        except ValueError as exc:
            raise ValueError(
                f"selected {role} must be inside source_dir: {selected}"
            ) from exc
        if not path.is_file():
            raise ValueError(f"selected {role} is not a file: {selected}")
        resolved[role] = path
    return resolved


def _validate_contract(contract: ValidationInputContract) -> ValidationInputContract:
    return input_contract_from_dict(input_contract_to_dict(contract))


def _encode_contract_columns(contract: ValidationInputContract) -> dict[str, str]:
    payload = input_contract_to_dict(contract)
    candidate_payload = {
        "candidates": payload["candidates"],
        "transformations": payload["transformations"],
        "conflicts": payload["conflicts"],
    }
    return {
        "candidate_json": _canonical_json(candidate_payload),
        "confirmed_json": _canonical_json(payload["confirmed"]),
        "material_hashes_json": _canonical_json(payload["material_hashes"]),
        "sample_schema_json": _canonical_json(payload["sample_schema"]),
        "pmml_manifest_json": _canonical_json(payload["pmml_manifest"]),
        "metadata_resolution_json": _canonical_json(payload["feature_metadata"]),
    }


def _row_to_contract_record(row: sqlite3.Row) -> ValidationInputContractRecord:
    task_id = str(row["task_id"])
    try:
        candidate = _load_json(row["candidate_json"])
        if not isinstance(candidate, dict) or set(candidate) != _CANDIDATE_KEYS:
            raise ValueError("candidate JSON has invalid keys")
        payload = {
            "schema_version": row["schema_version"],
            "material_hashes": _load_json(row["material_hashes_json"]),
            "status": row["status"],
            "candidates": candidate["candidates"],
            "sample_schema": _load_json(row["sample_schema_json"]),
            "pmml_manifest": _load_json(row["pmml_manifest_json"]),
            "feature_metadata": _load_json(row["metadata_resolution_json"]),
            "confirmed": _load_json(row["confirmed_json"]),
            "transformations": candidate["transformations"],
            "conflicts": candidate["conflicts"],
        }
        contract = input_contract_from_dict(payload)
        revision = row["revision"]
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise ValueError("contract revision must be positive")
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValidationContractDataError(
            f"corrupt validation input contract record for task {task_id}"
        ) from exc
    return ValidationInputContractRecord(
        task_id=task_id,
        revision=revision,
        status=contract.status,
        contract=contract,
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _load_json(raw: object) -> Any:
    if not isinstance(raw, str):
        raise ValueError("stored JSON must be text")
    return json.loads(raw)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
