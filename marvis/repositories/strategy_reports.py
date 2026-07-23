"""Immutable, task-owned StrategyReportBundle V2 publication ledger.

The pure report contract proves internal consistency.  This repository proves
publication lineage: every revision is based on the current task/strategy head
and is bound to three immutable task-artifact rows produced by the governed
report tool.  Canonical report bytes and artifact provenance are revalidated at
every read and write boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
from typing import Any

from marvis.db_schema import connect
from marvis.packs.strategy.report_bundle import (
    STRATEGY_REPORT_BUNDLE_SCHEMA_VERSION,
    STRATEGY_REPORT_DATA_CLASSIFICATION,
    STRATEGY_REPORT_PRODUCER_VERSION,
    canonical_strategy_report_bundle_json,
    validate_strategy_report_bundle,
)


STRATEGY_REPORT_HEAD_SCHEMA_VERSION = "strategy.report-head.v2"
STRATEGY_REPORT_OUTPUT_ARTIFACT_SCHEMA_VERSION = (
    "strategy.report-output-artifact.v2"
)
STRATEGY_REPORT_ORIGIN_TOOL = "strategy.build_report_bundle_v2"
ABSENT_STRATEGY_REPORT_REVISION = 0

STRATEGY_REPORT_OUTPUT_KINDS = {
    "json": "strategy_report_bundle_json",
    "markdown": "strategy_report_markdown",
    "xlsx": "strategy_report_xlsx",
}

_OUTPUT_FORMATS = ("json", "markdown", "xlsx")
_DRAFT_SCOPE = "task-draft"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "report_id",
        "report_revision",
        "bundle_content_sha256",
        "format",
        "data_classification",
    }
)


class StrategyReportConflictError(RuntimeError):
    """The caller's expected report head is stale or collided."""


class StrategyReportDataError(ValueError):
    """A supplied or persisted report publication violates its contract."""


class StrategyReportNotFoundError(KeyError):
    """The owning task, strategy, report, or artifact does not exist."""


def build_strategy_report_output_artifact_provenance(
    bundle: Mapping[str, Any],
    *,
    output_format: str,
) -> dict[str, Any]:
    """Build the exact pointer-only provenance required for one output file."""

    report = _validate_bundle(bundle)
    if output_format not in STRATEGY_REPORT_OUTPUT_KINDS:
        raise StrategyReportDataError(
            f"output_format must be one of {', '.join(_OUTPUT_FORMATS)}"
        )
    return {
        "schema_version": STRATEGY_REPORT_OUTPUT_ARTIFACT_SCHEMA_VERSION,
        "report_id": report["report_id"],
        "report_revision": report["report_revision"],
        "bundle_content_sha256": report["content_sha256"],
        "format": output_format,
        "data_classification": STRATEGY_REPORT_DATA_CLASSIFICATION,
    }


class StrategyReportRepository:
    """Persist one append-only report revision chain per task/strategy scope."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def transaction(self):
        """Return a configured connection for a caller-owned unit of work."""

        return connect(self.db_path)

    def get_head(
        self,
        *,
        task_id: str,
        strategy_id: str | None,
    ) -> dict[str, Any]:
        with connect(self.db_path) as conn:
            return self.get_head_on_connection(
                conn,
                task_id=task_id,
                strategy_id=strategy_id,
            )

    @staticmethod
    def get_head_on_connection(
        conn: sqlite3.Connection,
        *,
        task_id: str,
        strategy_id: str | None,
    ) -> dict[str, Any]:
        task = _required_text(task_id, field="task_id")
        identity = _optional_text(strategy_id, field="strategy_id")
        scope = _strategy_scope(identity)
        row = _select_head(conn, task_id=task, strategy_scope=scope)
        if row is None:
            return {
                "task_id": task,
                "strategy_id": identity,
                "strategy_scope": scope,
                "current_revision": ABSENT_STRATEGY_REPORT_REVISION,
                "current_report_id": None,
                "current_content_hash": None,
            }
        return _head_from_row(
            row,
            task_id=task,
            strategy_id=identity,
            strategy_scope=scope,
        )

    def get_current(
        self,
        *,
        task_id: str,
        strategy_id: str | None,
    ) -> dict[str, Any] | None:
        with connect(self.db_path) as conn:
            return self.get_current_on_connection(
                conn,
                task_id=task_id,
                strategy_id=strategy_id,
            )

    @staticmethod
    def get_current_on_connection(
        conn: sqlite3.Connection,
        *,
        task_id: str,
        strategy_id: str | None,
    ) -> dict[str, Any] | None:
        head = StrategyReportRepository.get_head_on_connection(
            conn,
            task_id=task_id,
            strategy_id=strategy_id,
        )
        if head["current_revision"] == ABSENT_STRATEGY_REPORT_REVISION:
            return None
        row = conn.execute(
            """
            SELECT * FROM strategy_report_revisions
             WHERE report_id = ? AND task_id = ? AND strategy_scope = ?
            """,
            (
                head["current_report_id"],
                head["task_id"],
                head["strategy_scope"],
            ),
        ).fetchone()
        if row is None:
            raise StrategyReportDataError(
                "strategy report head references a missing revision"
            )
        record = _record_from_row(conn, row)
        if (
            record["bundle"]["report_revision"] != head["current_revision"]
            or record["bundle"]["report_id"] != head["current_report_id"]
            or not hmac.compare_digest(
                record["bundle"]["content_sha256"],
                head["current_content_hash"],
            )
        ):
            raise StrategyReportDataError(
                "strategy report head does not match its current revision"
            )
        return record

    def get_revision(
        self,
        *,
        task_id: str,
        strategy_id: str | None,
        report_revision: int,
    ) -> dict[str, Any] | None:
        task = _required_text(task_id, field="task_id")
        identity = _optional_text(strategy_id, field="strategy_id")
        revision = _positive_int(report_revision, field="report_revision")
        scope = _strategy_scope(identity)
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT * FROM strategy_report_revisions
                 WHERE task_id = ? AND strategy_scope = ?
                   AND report_revision = ?
                """,
                (task, scope, revision),
            ).fetchone()
            return None if row is None else _record_from_row(conn, row)

    def get_by_id(
        self,
        *,
        task_id: str,
        report_id: str,
    ) -> dict[str, Any] | None:
        task = _required_text(task_id, field="task_id")
        identity = _required_text(report_id, field="report_id")
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT * FROM strategy_report_revisions
                 WHERE task_id = ? AND report_id = ?
                """,
                (task, identity),
            ).fetchone()
            return None if row is None else _record_from_row(conn, row)

    def publish(
        self,
        *,
        bundle: Mapping[str, Any],
        artifacts: Mapping[str, Mapping[str, Any]],
        expected_revision: int,
        expected_report_id: str | None,
        expected_content_hash: str | None,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            return self.publish_on_connection(
                conn,
                bundle=bundle,
                artifacts=artifacts,
                expected_revision=expected_revision,
                expected_report_id=expected_report_id,
                expected_content_hash=expected_content_hash,
                created_at=created_at,
            )

    @staticmethod
    def publish_on_connection(
        conn: sqlite3.Connection,
        *,
        bundle: Mapping[str, Any],
        artifacts: Mapping[str, Mapping[str, Any]],
        expected_revision: int,
        expected_report_id: str | None,
        expected_content_hash: str | None,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        """Publish a report within the caller's artifact/audit transaction."""

        report = _validate_bundle(bundle)
        expected, expected_id, expected_hash = _expected_head(
            expected_revision,
            expected_report_id,
            expected_content_hash,
        )
        timestamp = _optional_timestamp(created_at)
        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")

        task = conn.execute(
            "SELECT id, task_type FROM tasks WHERE id = ?",
            (report["task_id"],),
        ).fetchone()
        if task is None:
            raise StrategyReportNotFoundError(
                f"task not found: {report['task_id']}"
            )
        if str(task["task_type"]) != "strategy":
            raise StrategyReportDataError(
                "StrategyReportBundle requires a strategy task"
            )
        _require_strategy_identity_on_connection(conn, report)

        scope = _strategy_scope(report["strategy_id"])
        output_records = _verify_output_artifacts_on_connection(
            conn,
            report=report,
            artifacts=artifacts,
        )
        head_row = _select_head(
            conn,
            task_id=report["task_id"],
            strategy_scope=scope,
        )
        if head_row is None:
            if expected != 0 or expected_id is not None or expected_hash is not None:
                raise StrategyReportConflictError(
                    "initial report write requires the absent head triple"
                )
            conn.execute(
                """
                INSERT INTO strategy_report_heads(
                    task_id, strategy_scope, strategy_id, schema_version,
                    current_revision, current_report_id, current_content_hash,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, 0, NULL, NULL, ?, ?)
                """,
                (
                    report["task_id"],
                    scope,
                    report["strategy_id"],
                    STRATEGY_REPORT_HEAD_SCHEMA_VERSION,
                    timestamp,
                    timestamp,
                ),
            )
            current_revision = 0
            current_report_id = None
            current_content_hash = None
        else:
            head = _head_from_row(
                head_row,
                task_id=report["task_id"],
                strategy_id=report["strategy_id"],
                strategy_scope=scope,
            )
            current_revision = head["current_revision"]
            current_report_id = head["current_report_id"]
            current_content_hash = head["current_content_hash"]

        existing = conn.execute(
            "SELECT * FROM strategy_report_revisions WHERE report_id = ?",
            (report["report_id"],),
        ).fetchone()
        if existing is not None:
            persisted = _record_from_row(conn, existing)
            if (
                persisted["bundle"] != report
                or persisted["artifacts"] != output_records
            ):
                raise StrategyReportDataError(
                    "stable strategy report identity collided"
                )
            original_parent_hash = None
            if report["previous_report_id"] is not None:
                parent = conn.execute(
                    """
                    SELECT bundle_content_hash
                      FROM strategy_report_revisions
                     WHERE report_id = ? AND task_id = ? AND strategy_scope = ?
                    """,
                    (
                        report["previous_report_id"],
                        report["task_id"],
                        scope,
                    ),
                ).fetchone()
                if parent is None:
                    raise StrategyReportDataError(
                        "persisted strategy report parent is missing"
                    )
                original_parent_hash = _sha256(
                    parent["bundle_content_hash"],
                    field="parent.bundle_content_hash",
                )
            if (
                expected != report["report_revision"] - 1
                or expected_id != report["previous_report_id"]
                or expected_hash != original_parent_hash
            ):
                raise StrategyReportConflictError(
                    "exact report retry must use its original parent head triple"
                )
            if (
                current_revision == report["report_revision"]
                and current_report_id == report["report_id"]
                and current_content_hash == report["content_sha256"]
            ):
                return persisted
            raise StrategyReportConflictError(
                "exact report operation is no longer the current head"
            )

        if (
            current_revision != expected
            or current_report_id != expected_id
            or current_content_hash != expected_hash
        ):
            raise StrategyReportConflictError(
                "stale strategy report head: expected "
                f"({expected}, {expected_id!r}, {expected_hash!r}), found "
                f"({current_revision}, {current_report_id!r}, "
                f"{current_content_hash!r})"
            )
        if report["report_revision"] != expected + 1:
            raise StrategyReportDataError(
                "new report revision must equal expected_revision + 1"
            )
        if report["previous_report_id"] != expected_id:
            raise StrategyReportDataError(
                "previous_report_id must match the expected report head"
            )

        canonical = canonical_strategy_report_bundle_json(report)
        conn.execute(
            """
            INSERT INTO strategy_report_revisions(
                report_id, schema_version, producer_version, task_id,
                strategy_scope, strategy_id, strategy_version,
                report_revision, previous_report_id, report_json,
                bundle_content_hash,
                json_artifact_id, json_artifact_hash,
                markdown_artifact_id, markdown_artifact_hash,
                xlsx_artifact_id, xlsx_artifact_hash, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report["report_id"],
                report["schema_version"],
                report["producer_version"],
                report["task_id"],
                scope,
                report["strategy_id"],
                report["strategy_version"],
                report["report_revision"],
                report["previous_report_id"],
                canonical,
                report["content_sha256"],
                output_records["json"]["id"],
                output_records["json"]["content_hash"],
                output_records["markdown"]["id"],
                output_records["markdown"]["content_hash"],
                output_records["xlsx"]["id"],
                output_records["xlsx"]["content_hash"],
                timestamp,
            ),
        )
        cursor = conn.execute(
            """
            UPDATE strategy_report_heads
               SET current_revision = ?, current_report_id = ?,
                   current_content_hash = ?, updated_at = ?
             WHERE task_id = ? AND strategy_scope = ?
               AND current_revision = ?
               AND current_report_id IS ?
               AND current_content_hash IS ?
            """,
            (
                report["report_revision"],
                report["report_id"],
                report["content_sha256"],
                timestamp,
                report["task_id"],
                scope,
                expected,
                expected_id,
                expected_hash,
            ),
        )
        if cursor.rowcount != 1:
            raise StrategyReportConflictError(
                "strategy report head changed while publishing"
            )
        return {
            "bundle": report,
            "artifacts": output_records,
            "created_at": timestamp,
        }


def _verify_output_artifacts_on_connection(
    conn: sqlite3.Connection,
    *,
    report: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(_OUTPUT_FORMATS):
        raise StrategyReportDataError(
            "artifacts must contain exactly json, markdown, and xlsx"
        )
    normalized: dict[str, dict[str, Any]] = {}
    for output_format in _OUTPUT_FORMATS:
        supplied = artifacts[output_format]
        if not isinstance(supplied, Mapping):
            raise StrategyReportDataError(
                f"artifacts.{output_format} must be an object"
            )
        artifact_id = _required_text(
            supplied.get("id"),
            field=f"artifacts.{output_format}.id",
        )
        row = conn.execute(
            "SELECT * FROM task_artifacts WHERE id = ?",
            (artifact_id,),
        ).fetchone()
        if row is None:
            raise StrategyReportNotFoundError(
                f"report output artifact not found: {artifact_id}"
            )
        persisted = _artifact_from_row(row)
        for field in (
            "id",
            "task_id",
            "kind",
            "path",
            "content_hash",
            "origin_tool",
            "provenance",
            "created_at",
        ):
            if field in supplied and supplied[field] != persisted[field]:
                raise StrategyReportDataError(
                    f"artifacts.{output_format}.{field} does not match registry"
                )
        expected_provenance = build_strategy_report_output_artifact_provenance(
            report,
            output_format=output_format,
        )
        if persisted["task_id"] != report["task_id"]:
            raise StrategyReportDataError(
                f"artifacts.{output_format} belongs to another task"
            )
        if persisted["kind"] != STRATEGY_REPORT_OUTPUT_KINDS[output_format]:
            raise StrategyReportDataError(
                f"artifacts.{output_format} kind is invalid"
            )
        if persisted["origin_tool"] != STRATEGY_REPORT_ORIGIN_TOOL:
            raise StrategyReportDataError(
                f"artifacts.{output_format} origin_tool is invalid"
            )
        if persisted["provenance"] != expected_provenance:
            raise StrategyReportDataError(
                f"artifacts.{output_format} provenance is invalid"
            )
        _require_canonical_output_path(
            Path(persisted["path"]),
            report=report,
            output_format=output_format,
        )
        normalized[output_format] = persisted

    try:
        from marvis.output.strategy_report_bundle import (
            render_strategy_report_bundle,
        )

        expected_outputs = render_strategy_report_bundle(report)
    except Exception as exc:
        raise StrategyReportDataError(
            "strategy report outputs could not be reproduced"
        ) from exc
    for output_format in _OUTPUT_FORMATS:
        expected = expected_outputs[output_format]
        persisted = normalized[output_format]
        expected_hash = hashlib.sha256(expected).hexdigest()
        if not hmac.compare_digest(
            persisted["content_hash"],
            expected_hash,
        ):
            raise StrategyReportDataError(
                f"{output_format} artifact hash does not match rendered bytes"
            )
        _read_exact_output_file(
            Path(persisted["path"]),
            expected=expected,
            output_format=output_format,
        )
    return normalized


def _require_canonical_output_path(
    path: Path,
    *,
    report: Mapping[str, Any],
    output_format: str,
) -> None:
    suffix = "md" if output_format == "markdown" else output_format
    if (
        not path.is_absolute()
        or path.name != f"report.{suffix}"
        or path.parent.name != report["report_id"]
        or path.parent.parent.name != "strategy_reports"
        or path.parent.parent.parent.name != report["task_id"]
    ):
        raise StrategyReportDataError(
            f"{output_format} report artifact path is not canonical"
        )


def _read_exact_output_file(
    path: Path,
    *,
    expected: bytes,
    output_format: str,
) -> None:
    descriptor = -1
    chunks: list[bytes] = []
    try:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != len(expected):
            raise StrategyReportDataError(
                f"{output_format} report artifact file is invalid"
            )
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise StrategyReportDataError(
                f"{output_format} report artifact changed while being read"
            )
        live_path = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(live_path.st_mode)
            or (live_path.st_dev, live_path.st_ino)
            != (after.st_dev, after.st_ino)
        ):
            raise StrategyReportDataError(
                f"{output_format} report artifact path changed while being read"
            )
    except OSError as exc:
        raise StrategyReportDataError(
            f"{output_format} report artifact file is unavailable"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if b"".join(chunks) != expected:
        raise StrategyReportDataError(
            f"{output_format} report artifact bytes do not match the report"
        )


def _record_from_row(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
) -> dict[str, Any]:
    raw = row["report_json"]
    if not isinstance(raw, str):
        raise StrategyReportDataError("persisted report_json is invalid")
    try:
        parsed = json.loads(raw)
        report = validate_strategy_report_bundle(parsed)
        canonical = canonical_strategy_report_bundle_json(report)
    except (TypeError, ValueError) as exc:
        raise StrategyReportDataError("persisted report_json is invalid") from exc
    if canonical != raw:
        raise StrategyReportDataError("persisted report_json is not canonical")
    scope = _strategy_scope(report["strategy_id"])
    comparisons = {
        "report_id": report["report_id"],
        "schema_version": STRATEGY_REPORT_BUNDLE_SCHEMA_VERSION,
        "producer_version": STRATEGY_REPORT_PRODUCER_VERSION,
        "task_id": report["task_id"],
        "strategy_scope": scope,
        "strategy_id": report["strategy_id"],
        "strategy_version": report["strategy_version"],
        "report_revision": report["report_revision"],
        "previous_report_id": report["previous_report_id"],
        "bundle_content_hash": report["content_sha256"],
    }
    for field, expected in comparisons.items():
        if row[field] != expected:
            raise StrategyReportDataError(
                f"persisted strategy report {field} does not match report_json"
            )
    artifact_ids = {
        "json": row["json_artifact_id"],
        "markdown": row["markdown_artifact_id"],
        "xlsx": row["xlsx_artifact_id"],
    }
    supplied: dict[str, dict[str, Any]] = {}
    for output_format, artifact_id in artifact_ids.items():
        artifact_row = conn.execute(
            "SELECT * FROM task_artifacts WHERE id = ?",
            (artifact_id,),
        ).fetchone()
        if artifact_row is None:
            raise StrategyReportDataError(
                f"persisted {output_format} report artifact is missing"
            )
        supplied[output_format] = _artifact_from_row(artifact_row)
        expected_hash = row[f"{output_format}_artifact_hash"]
        if supplied[output_format]["content_hash"] != expected_hash:
            raise StrategyReportDataError(
                f"persisted {output_format} report artifact hash drifted"
            )
    artifacts = _verify_output_artifacts_on_connection(
        conn,
        report=report,
        artifacts=supplied,
    )
    return {
        "bundle": report,
        "artifacts": artifacts,
        "created_at": _required_text(row["created_at"], field="created_at"),
    }


def _artifact_from_row(row: sqlite3.Row) -> dict[str, Any]:
    try:
        provenance = json.loads(str(row["provenance_json"]))
    except (TypeError, ValueError) as exc:
        raise StrategyReportDataError(
            "persisted report artifact provenance is invalid"
        ) from exc
    if (
        not isinstance(provenance, dict)
        or set(provenance) != _ARTIFACT_PROVENANCE_FIELDS
    ):
        raise StrategyReportDataError(
            "persisted report artifact provenance is invalid"
        )
    return {
        "id": _required_text(row["id"], field="artifact.id"),
        "task_id": _required_text(row["task_id"], field="artifact.task_id"),
        "kind": _required_text(row["kind"], field="artifact.kind"),
        "path": _required_text(row["path"], field="artifact.path"),
        "content_hash": _sha256(row["content_hash"], field="artifact.content_hash"),
        "origin_tool": _required_text(
            row["origin_tool"],
            field="artifact.origin_tool",
        ),
        "provenance": provenance,
        "created_at": _required_text(
            row["created_at"],
            field="artifact.created_at",
        ),
    }


def _validate_bundle(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        report = validate_strategy_report_bundle(value)
    except (TypeError, ValueError) as exc:
        raise StrategyReportDataError(
            f"invalid StrategyReportBundle: {exc}"
        ) from exc
    return report


def _require_strategy_identity_on_connection(
    conn: sqlite3.Connection,
    report: Mapping[str, Any],
) -> None:
    strategy_id = report["strategy_id"]
    if strategy_id is None:
        return
    row = conn.execute(
        "SELECT task_id, version, strategy_type FROM strategies WHERE id = ?",
        (strategy_id,),
    ).fetchone()
    if row is None:
        raise StrategyReportNotFoundError(f"strategy not found: {strategy_id}")
    if str(row["task_id"]) != report["task_id"]:
        raise StrategyReportDataError("strategy belongs to another task")
    if str(row["version"]) != report["strategy_version"]:
        raise StrategyReportDataError(
            "strategy_version does not match the persisted strategy"
        )
    if str(row["strategy_type"]) != report["strategy_type"]:
        raise StrategyReportDataError(
            "strategy_type does not match the persisted strategy"
        )


def _head_from_row(
    row: sqlite3.Row,
    *,
    task_id: str,
    strategy_id: str | None,
    strategy_scope: str,
) -> dict[str, Any]:
    if (
        row["task_id"] != task_id
        or row["strategy_scope"] != strategy_scope
        or row["strategy_id"] != strategy_id
        or row["schema_version"] != STRATEGY_REPORT_HEAD_SCHEMA_VERSION
    ):
        raise StrategyReportDataError("persisted strategy report head is invalid")
    revision = row["current_revision"]
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
        raise StrategyReportDataError(
            "persisted strategy report head revision is invalid"
        )
    report_id = _optional_text(
        row["current_report_id"],
        field="head.current_report_id",
    )
    content_hash = (
        None
        if row["current_content_hash"] is None
        else _sha256(
            row["current_content_hash"],
            field="head.current_content_hash",
        )
    )
    if revision == 0 and (report_id is not None or content_hash is not None):
        raise StrategyReportDataError(
            "absent strategy report head triple is incomplete"
        )
    if revision > 0 and (report_id is None or content_hash is None):
        raise StrategyReportDataError(
            "current strategy report head triple is incomplete"
        )
    return {
        "task_id": task_id,
        "strategy_id": strategy_id,
        "strategy_scope": strategy_scope,
        "current_revision": revision,
        "current_report_id": report_id,
        "current_content_hash": content_hash,
    }


def _expected_head(
    revision: object,
    report_id: object,
    content_hash: object,
) -> tuple[int, str | None, str | None]:
    number = _non_negative_int(revision, field="expected_revision")
    if number == 0:
        if report_id is not None or content_hash is not None:
            raise StrategyReportDataError(
                "absent expected head requires null report_id and content_hash"
            )
        return number, None, None
    return (
        number,
        _required_text(report_id, field="expected_report_id"),
        _sha256(content_hash, field="expected_content_hash"),
    )


def _select_head(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    strategy_scope: str,
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT * FROM strategy_report_heads
         WHERE task_id = ? AND strategy_scope = ?
        """,
        (task_id, strategy_scope),
    ).fetchone()


def _strategy_scope(strategy_id: str | None) -> str:
    return _DRAFT_SCOPE if strategy_id is None else f"strategy:{strategy_id}"


def _optional_timestamp(value: object) -> str:
    return (
        datetime.now(UTC).isoformat()
        if value is None
        else _required_text(value, field="created_at")
    )


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StrategyReportDataError(f"{field} must be non-empty text")
    normalized = value.strip()
    if "\x00" in normalized:
        raise StrategyReportDataError(f"{field} must not contain NUL bytes")
    return normalized


def _optional_text(value: object, *, field: str) -> str | None:
    return None if value is None else _required_text(value, field=field)


def _sha256(value: object, *, field: str) -> str:
    normalized = _required_text(value, field=field)
    if _SHA256_RE.fullmatch(normalized) is None:
        raise StrategyReportDataError(
            f"{field} must be a lowercase SHA-256 digest"
        )
    return normalized


def _non_negative_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise StrategyReportDataError(
            f"{field} must be a non-negative integer"
        )
    return value


def _positive_int(value: object, *, field: str) -> int:
    number = _non_negative_int(value, field=field)
    if number == 0:
        raise StrategyReportDataError(f"{field} must be positive")
    return number


__all__ = [
    "ABSENT_STRATEGY_REPORT_REVISION",
    "STRATEGY_REPORT_HEAD_SCHEMA_VERSION",
    "STRATEGY_REPORT_ORIGIN_TOOL",
    "STRATEGY_REPORT_OUTPUT_ARTIFACT_SCHEMA_VERSION",
    "STRATEGY_REPORT_OUTPUT_KINDS",
    "StrategyReportConflictError",
    "StrategyReportDataError",
    "StrategyReportNotFoundError",
    "StrategyReportRepository",
    "build_strategy_report_output_artifact_provenance",
]
