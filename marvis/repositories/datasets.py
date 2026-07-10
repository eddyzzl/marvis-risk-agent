import json
import sqlite3
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from marvis.data.contracts import (
    ColumnFingerprint,
    ColumnProfile,
    Dataset,
    JoinDiagnostics,
    JoinPlan,
    JoinSpec,
    KeyDtypeDivergence,
    KeyPair,
)
from marvis.db_schema import connect
from marvis.state_machine import ConflictError


def _now() -> str:
    return datetime.now(UTC).isoformat()


class DatasetRepository:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def transaction(self):
        return connect(self.db_path)

    def create_dataset(self, dataset: Dataset) -> None:
        with connect(self.db_path) as conn:
            _insert_dataset_row(conn, dataset)

    def create_dataset_on_connection(self, conn: sqlite3.Connection, dataset: Dataset) -> None:
        _insert_dataset_row(conn, dataset)

    def create_dataset_with_audit(self, dataset: Dataset, *, audit: dict) -> None:
        with connect(self.db_path) as conn:
            _insert_dataset_row(conn, dataset)
            _write_audit_row(conn, **audit)

    def create_dataset_with_audit_on_connection(
        self,
        conn: sqlite3.Connection,
        dataset: Dataset,
        *,
        audit: dict,
    ) -> None:
        _insert_dataset_row(conn, dataset)
        _write_audit_row(conn, **audit)

    def get_dataset(self, dataset_id: str) -> Dataset | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT id, task_id, role, source_path, format, sheet, row_count,
                       columns_json, has_target, target_col, created_at, content_hash
                  FROM datasets
                 WHERE id = ?
                """,
                (dataset_id,),
            ).fetchone()
        return None if row is None else _dataset_from_row(row)

    def list_datasets(self, task_id: str) -> list[Dataset]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, task_id, role, source_path, format, sheet, row_count,
                       columns_json, has_target, target_col, created_at, content_hash
                  FROM datasets
                 WHERE task_id = ?
                 ORDER BY created_at, id
                """,
                (task_id,),
            ).fetchall()
        return [_dataset_from_row(row) for row in rows]

    def find_dataset_by_content_hash(self, content_hash: str) -> Dataset | None:
        """GAP-7: look up an already-registered dataset with identical file
        content, regardless of which task registered it, so a new upload can
        reuse the existing parquet + profiling instead of duplicating both."""
        with connect(self.db_path) as conn:
            return self.find_dataset_by_content_hash_on_connection(conn, content_hash)

    def find_dataset_by_content_hash_on_connection(
        self,
        conn: sqlite3.Connection,
        content_hash: str,
    ) -> Dataset | None:
        row = conn.execute(
            """
            SELECT id, task_id, role, source_path, format, sheet, row_count,
                   columns_json, has_target, target_col, created_at, content_hash
              FROM datasets
             WHERE content_hash = ?
             ORDER BY created_at, id
             LIMIT 1
            """,
            (content_hash,),
        ).fetchone()
        return None if row is None else _dataset_from_row(row)

    def set_dataset_role(self, dataset_id: str, role: str) -> None:
        with connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE datasets SET role = ? WHERE id = ?",
                (role, dataset_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(dataset_id)

    def create_join_plan(self, plan: JoinPlan) -> None:
        with connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO joins(
                    id, task_id, anchor_dataset_id, joins_json, status,
                    result_dataset_id, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plan.id,
                    plan.task_id,
                    plan.anchor_dataset_id,
                    _dump_json_any([_join_spec_to_dict(spec) for spec in plan.joins]),
                    plan.status,
                    plan.result_dataset_id,
                    _now(),
                ),
            )

    def load_join_plan(self, plan_id: str) -> JoinPlan:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT id, task_id, anchor_dataset_id, joins_json, status,
                       result_dataset_id
                  FROM joins
                 WHERE id = ?
                """,
                (plan_id,),
            ).fetchone()
        if row is None:
            raise KeyError(plan_id)
        return _join_plan_from_row(row)

    def update_join_spec(self, plan_id: str, spec: JoinSpec) -> None:
        with connect(self.db_path) as conn:
            _update_join_spec_row(conn, plan_id, spec)

    def update_join_spec_with_audit(
        self,
        plan_id: str,
        spec: JoinSpec,
        *,
        audit: dict,
    ) -> None:
        with connect(self.db_path) as conn:
            _update_join_spec_row(conn, plan_id, spec)
            _write_audit_row(conn, **audit)

    def set_join_plan_executed(self, plan_id: str, result_dataset_id: str) -> None:
        with connect(self.db_path) as conn:
            _set_join_plan_executed_row(conn, plan_id, result_dataset_id)

    def set_join_plan_executed_with_audit(
        self,
        plan_id: str,
        result_dataset_id: str,
        *,
        audit: dict,
    ) -> None:
        with connect(self.db_path) as conn:
            _set_join_plan_executed_row(conn, plan_id, result_dataset_id)
            _write_audit_row(conn, **audit)

    def record_join_result_with_audit(
        self,
        plan_id: str,
        dataset: Dataset,
        *,
        audit: dict,
    ) -> None:
        with connect(self.db_path) as conn:
            self.record_join_result_with_audit_on_connection(
                conn,
                plan_id,
                dataset,
                audit=audit,
            )

    def record_join_result_with_audit_on_connection(
        self,
        conn: sqlite3.Connection,
        plan_id: str,
        dataset: Dataset,
        *,
        audit: dict,
    ) -> None:
        _insert_dataset_row(conn, dataset)
        _set_join_plan_executed_row(conn, plan_id, dataset.id)
        _write_audit_row(conn, **audit)

    def write_audit(self, **kwargs) -> None:
        with connect(self.db_path) as conn:
            _write_audit_row(conn, **kwargs)

    def write_audit_on_connection(self, conn: sqlite3.Connection, **kwargs) -> None:
        _write_audit_row(conn, **kwargs)


def _dataset_insert_values(dataset: Dataset) -> tuple:
    return (
        dataset.id,
        dataset.task_id,
        dataset.role,
        dataset.source_path,
        dataset.format,
        dataset.sheet,
        dataset.row_count,
        _dump_json_any([_column_profile_to_dict(column) for column in dataset.columns]),
        1 if dataset.has_target else 0,
        dataset.target_col,
        dataset.created_at,
        dataset.content_hash,
    )


def _insert_dataset_row(conn: sqlite3.Connection, dataset: Dataset) -> None:
    conn.execute(
        """
        INSERT INTO datasets(
            id, task_id, role, source_path, format, sheet, row_count,
            columns_json, has_target, target_col, created_at, content_hash
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        _dataset_insert_values(dataset),
    )


def _dataset_from_row(row: sqlite3.Row) -> Dataset:
    return Dataset(
        id=str(row["id"]),
        task_id=str(row["task_id"]),
        role=str(row["role"]),
        source_path=str(row["source_path"]),
        format=str(row["format"]),
        sheet=row["sheet"],
        row_count=int(row["row_count"]),
        columns=tuple(
            _column_profile_from_dict(item)
            for item in _load_json_array(row["columns_json"])
        ),
        has_target=bool(row["has_target"]),
        target_col=row["target_col"],
        created_at=str(row["created_at"]),
        content_hash=(
            _optional_str(row["content_hash"]) if "content_hash" in row.keys() else None
        ),
    )


def _update_join_spec_row(conn: sqlite3.Connection, plan_id: str, spec: JoinSpec) -> None:
    row = conn.execute(
        "SELECT joins_json FROM joins WHERE id = ?",
        (plan_id,),
    ).fetchone()
    if row is None:
        raise KeyError(plan_id)
    original_json = str(row["joins_json"])
    replaced = False
    joins = []
    for item in (
        _join_spec_from_dict(payload) for payload in _load_json_array(original_json)
    ):
        if item.feature_dataset_id == spec.feature_dataset_id:
            joins.append(spec)
            replaced = True
        else:
            joins.append(item)
    if not replaced:
        raise KeyError(spec.feature_dataset_id)
    cursor = conn.execute(
        """
        UPDATE joins
           SET joins_json = ?
         WHERE id = ?
           AND joins_json = ?
        """,
        (
            _dump_json_any([_join_spec_to_dict(item) for item in joins]),
            plan_id,
            original_json,
        ),
    )
    if cursor.rowcount == 0:
        raise ConflictError(f"join plan {plan_id} changed while updating spec")


def _set_join_plan_executed_row(
    conn: sqlite3.Connection,
    plan_id: str,
    result_dataset_id: str,
) -> None:
    cursor = conn.execute(
        """
        UPDATE joins
           SET status = 'executed',
               result_dataset_id = ?
         WHERE id = ?
           AND status = 'draft'
        """,
        (result_dataset_id, plan_id),
    )
    if cursor.rowcount != 0:
        return
    row = conn.execute("SELECT status FROM joins WHERE id = ?", (plan_id,)).fetchone()
    if row is None:
        raise KeyError(plan_id)
    raise ConflictError(f"join plan {plan_id} is already {row['status']}; cannot execute again")


def _column_profile_to_dict(profile: ColumnProfile) -> dict:
    return asdict(profile)


def _column_profile_from_dict(payload: dict) -> ColumnProfile:
    fingerprint_payload = dict(payload["fingerprint"])
    return ColumnProfile(
        name=str(payload["name"]),
        dtype=str(payload["dtype"]),
        semantic_role=str(payload["semantic_role"]),
        fingerprint=ColumnFingerprint(
            value_kind=str(fingerprint_payload["value_kind"]),
            length_mode=_optional_int(fingerprint_payload.get("length_mode")),
            regex_pattern=_optional_str(fingerprint_payload.get("regex_pattern")),
            is_hashed=bool(fingerprint_payload["is_hashed"]),
            hash_type=_optional_str(fingerprint_payload.get("hash_type")),
            hex_case=_optional_str(fingerprint_payload.get("hex_case")),
            date_format=_optional_str(fingerprint_payload.get("date_format")),
        ),
        null_rate=float(payload["null_rate"]),
        cardinality=int(payload["cardinality"]),
        sample_values=tuple(payload.get("sample_values") or ()),
    )


def _join_spec_to_dict(spec: JoinSpec) -> dict:
    return {
        "feature_dataset_id": spec.feature_dataset_id,
        "key_pairs": [asdict(pair) for pair in spec.key_pairs],
        "diagnostics": asdict(spec.diagnostics),
        "dedup_strategy": spec.dedup_strategy,
        "confirmed": spec.confirmed,
    }


def _join_plan_from_row(row: sqlite3.Row) -> JoinPlan:
    return JoinPlan(
        id=str(row["id"]),
        task_id=str(row["task_id"]),
        anchor_dataset_id=str(row["anchor_dataset_id"]),
        joins=[
            _join_spec_from_dict(item)
            for item in _load_json_array(row["joins_json"])
        ],
        status=str(row["status"]),
        result_dataset_id=_optional_str(row["result_dataset_id"]),
    )


def _join_spec_from_dict(payload: dict) -> JoinSpec:
    return JoinSpec(
        feature_dataset_id=str(payload["feature_dataset_id"]),
        key_pairs=[
            KeyPair(
                anchor_col=str(item["anchor_col"]),
                feature_col=str(item["feature_col"]),
                match_method=str(item["match_method"]),
                transform_side=str(item["transform_side"]),
                match_rate=float(item["match_rate"]),
                resolved_by=str(item["resolved_by"]),
                # T1-B8: dtype provenance survives the plan round-trip (defaulted for legacy).
                anchor_dtype=str(item.get("anchor_dtype", "")),
                feature_dtype=str(item.get("feature_dtype", "")),
                dtype_divergent=bool(item.get("dtype_divergent", False)),
            )
            for item in payload.get("key_pairs") or []
        ],
        diagnostics=_diagnostics_from_dict(dict(payload["diagnostics"])),
        dedup_strategy=_optional_str(payload.get("dedup_strategy")),
        confirmed=bool(payload.get("confirmed", False)),
    )


def _diagnostics_from_dict(payload: dict) -> JoinDiagnostics:
    # T1-B8: re-hydrate key_dtype_divergences into typed KeyDtypeDivergence objects so the
    # forced-confirmation gate can read `.level` after a plan reload (the other nested fields
    # -- conflict_report / key_alternatives -- are consumed as dicts downstream and left as-is).
    data = dict(payload)
    divergences = data.get("key_dtype_divergences") or ()
    data["key_dtype_divergences"] = tuple(
        item if isinstance(item, KeyDtypeDivergence) else KeyDtypeDivergence(
            anchor_col=str(item.get("anchor_col", "")),
            feature_col=str(item.get("feature_col", "")),
            anchor_dtype=str(item.get("anchor_dtype", "")),
            feature_dtype=str(item.get("feature_dtype", "")),
            level=str(item.get("level", "warn")),
        )
        for item in divergences
    )
    return JoinDiagnostics(**data)


def _write_audit_row(
    conn: sqlite3.Connection,
    *,
    kind: str,
    target_ref: str,
    actor: str = "system",
    inputs_hash: str | None = None,
    outcome: str | None = None,
    detail: dict | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO audit(
            id, kind, actor, target_ref, inputs_hash, outcome,
            detail_json, at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            uuid.uuid4().hex,
            kind,
            actor,
            target_ref,
            inputs_hash,
            outcome,
            json.dumps(detail or {}, ensure_ascii=False, separators=(",", ":")),
            _now(),
        ),
    )


def _optional_str(value) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_int(value) -> int | None:
    return None if value is None else int(value)


def _dump_json_any(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _load_json_array(raw: str | None) -> list:
    if not raw:
        return []
    value = json.loads(raw)
    return value if isinstance(value, list) else []
