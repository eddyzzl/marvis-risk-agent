"""Governed deterministic materialization for ``StrategyProjectContext``.

The boundary intentionally accepts only optimistic-CAS controls and bounded
user-authored text.  Every platform identity, metric, lifecycle value and
evidence pointer is discovered from task-owned storage and revalidated while a
SQLite writer lock is held.  External reports remain opaque byte evidence: this
module snapshots and hashes them, but never parses numbers out of them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import hmac
import json
import os
from pathlib import Path, PurePath
import re
import stat
from typing import Any
from urllib.parse import quote

from marvis.artifacts import ArtifactUnitOfWork
from marvis.db_schema import connect
from marvis.files import sha256_file
from marvis.packs.strategy.dsl import (
    canonical_strategy_json,
    parse_strategy_spec,
    strategy_spec_hash,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pool_impact import (
    STRATEGY_POOL_IMPACT_PRODUCER_VERSION,
    canonical_strategy_pool_impact_json,
    validate_strategy_pool_impact_assessment,
)
from marvis.packs.strategy.pool_impact_tools import (
    POOL_IMPACT_ARTIFACT_KIND,
    POOL_IMPACT_ARTIFACT_SCHEMA_VERSION,
    POOL_IMPACT_ORIGIN_TOOL,
)
from marvis.packs.strategy.project_context import (
    MAX_PROJECT_CONTEXT_JSON_BYTES,
    build_context_field,
    build_current_project_snapshot,
    build_effect_observation_ref,
    build_historical_strategy_review,
    build_missing_information_record,
    build_red_flag,
    build_report_field,
    build_source_ref,
    build_strategy_project_context_revision,
    build_strategy_project_context_state,
    canonical_strategy_project_context_revision_json,
    diff_strategy_rules,
    strategy_project_context_revision_from_json,
)
from marvis.packs.strategy.sample_design import (
    STRATEGY_SAMPLE_DESIGN_PRODUCER_VERSION,
    canonical_strategy_sample_design_bundle_json,
    strategy_sample_design_bundle_from_json,
)
from marvis.packs.strategy.sample_design_tools import (
    SAMPLE_DESIGN_ARTIFACT_KIND,
    SAMPLE_DESIGN_ARTIFACT_SCHEMA_VERSION,
    SAMPLE_DESIGN_ORIGIN_TOOL,
)
from marvis.packs.strategy.typed_backtest import (
    STRATEGY_BACKTEST_SCHEMA_VERSION,
    StrategyBacktestResult,
)
from marvis.repositories.audit import _write_audit_row
from marvis.repositories.strategy_monitoring import (
    StrategyMonitoringRepository,
)
from marvis.repositories.strategy_project_context import (
    StrategyProjectContextConflictError,
    StrategyProjectContextRepository,
)
from marvis.repositories.task_artifacts import TaskArtifactRepository


PROJECT_CONTEXT_TOOL_SCHEMA_VERSION = "strategy.materialize-project-context-tool.v1"
PROJECT_CONTEXT_ARTIFACT_KIND = "strategy_project_context_json"
PROJECT_CONTEXT_EXTERNAL_ARTIFACT_KIND = "strategy_history_external_source"
PROJECT_CONTEXT_ORIGIN_TOOL = "strategy.materialize_project_context"
PROJECT_CONTEXT_ARTIFACT_SCHEMA_VERSION = "strategy.project-context-artifact.v1"
PROJECT_CONTEXT_EXTERNAL_SCHEMA_VERSION = "strategy.history-external-source.v1"

MAX_SCOPE_CHARS = 4_000
MAX_BUSINESS_CONTEXT_FIELDS = 50
MAX_BUSINESS_CONTEXT_KEY_CHARS = 200
MAX_BUSINESS_CONTEXT_VALUE_CHARS = 4_000
MAX_EXPLICIT_UNAVAILABLE_FIELDS = 100
MAX_EXTERNAL_REPORTS = 20
MAX_EXTERNAL_REPORT_BYTES = 64 * 1024 * 1024
MAX_EXTERNAL_REPORT_TOTAL_BYTES = 256 * 1024 * 1024
MAX_EXTERNAL_FILENAME_CHARS = 500
MAX_EXTERNAL_SOURCE_ENTRIES = 20_000
_EXTERNAL_SUFFIXES = frozenset({".csv", ".docx", ".json", ".md", ".pdf", ".xlsx"})
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_ID_RE = re.compile(r"^strategy-project-context-revision-[0-9a-f]{24}$")
_FIELD_PATH_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*){0,7}$")
_INPUT_FIELDS = frozenset(
    {
        "expected_revision",
        "expected_revision_id",
        "expected_state_hash",
        "user_message_ref",
        "as_of",
        "scope",
        "business_context",
        "explicit_unavailable",
        "external_report_filenames",
    }
)
_REQUIRED_INPUT_FIELDS = _INPUT_FIELDS - {
    "expected_revision_id",
    "expected_state_hash",
    "scope",
}
_OUTPUT_FIELDS = frozenset(
    {
        "schema_version",
        "created",
        "revision",
        "context_artifact",
        "external_artifacts",
        "missing_information_records",
        "warnings",
    }
)
_OUTPUT_ARTIFACT_FIELDS = frozenset(
    {"artifact_id", "kind", "format", "filename", "content_hash", "download_url"}
)
_SAMPLE_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "format",
        "task_id",
        "bundle_id",
        "bundle_content_hash",
        "sample_design_id",
        "sample_design_content_hash",
        "dataset_id",
        "dataset_content_hash",
        "dataset_source_path",
        "registry_metadata_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "target_col",
        "request",
        "request_hash",
    }
)
_POOL_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "task_id",
        "assessment_id",
        "assessment_content_hash",
        "pool_id",
        "pool_revision",
        "pool_revision_id",
        "pool_snapshot_hash",
        "design_hash",
        "strategy_spec_hash",
        "dataset_id",
        "dataset_content_hash",
        "registry_metadata_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "target_col",
        "sample_design_ref",
        "month_col",
        "loan_amount_col",
        "overdue_amount_col",
        "source_target_bad_value",
        "normalized_target_bad_value",
        "sample_partition",
        "comparison_mode",
        "baseline_strategy_id",
        "baseline_spec_hash",
    }
)
_KNOWN_MISSING = {
    "current.status_fields.volume": (
        "No governed current-population evidence is available.",
        "impact",
        "请提供当前项目样本或先完成本次样本设计。",
    ),
    "current.status_fields.approval": (
        "No governed approval-impact evidence is available.",
        "impact",
        "请提供当前通过率材料，或先完成策略影响测算/回测。",
    ),
    "current.status_fields.risk": (
        "No governed current-risk evidence is available.",
        "strategy",
        "请提供当前风险表现材料，或先完成成熟样本设计/回测。",
    ),
    "current.status_fields.economics": (
        "No governed economics evidence is available.",
        "report_optional",
        "如有收益、损失、资金成本等口径，请提供；暂时没有也可以说明。",
    ),
    "current.maturity_summary": (
        "No governed sample-maturity evidence is available.",
        "validation",
        "请确认表现窗、观察窗以及样本是否已成熟。",
    ),
    "historical_strategy_reviews": (
        "No governed historical strategy or external report is available.",
        "report_optional",
        "如有历史策略版本或评审报告，请提供；暂时没有也可以说明。",
    ),
}


@dataclass(frozen=True)
class _ExternalSnapshot:
    relative_name: str
    suffix: str
    content_hash: str
    content_size: int


def run_materialize_project_context(inputs, ctx, runtime) -> dict[str, Any]:
    """Discover, validate and atomically publish one project-context revision."""

    request = _validate_inputs(inputs)
    task_id = _text(getattr(ctx, "task_id", None), "task_id")
    db_path = Path(runtime.settings.db_path)
    tasks_root = Path(runtime.settings.tasks_dir)
    repositories = _repositories(runtime, db_path=db_path)

    # Read once without a writer lock so a mutation in the planning/computation
    # window is observable.  The same evidence is read again under BEGIN
    # IMMEDIATE and compared byte-for-byte by its canonical fingerprint.
    with connect(db_path) as initial_conn:
        initial = _discover_evidence(
            initial_conn,
            task_id=task_id,
            request=request,
            runtime=runtime,
            tasks_root=tasks_root,
        )
    initial_external = _snapshot_external_reports(
        initial["task"]["source_dir"], request["external_report_filenames"]
    )

    uow = ArtifactUnitOfWork()
    promoted = False
    committed = False
    requested_external_records: list[dict[str, Any]] = []
    context_record: dict[str, Any] | None = None
    persisted: dict[str, Any]
    created = False
    try:
        with repositories["contexts"].transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            locked = _discover_evidence(
                conn,
                task_id=task_id,
                request=request,
                runtime=runtime,
                tasks_root=tasks_root,
            )
            locked_external = _snapshot_external_reports(
                locked["task"]["source_dir"], request["external_report_filenames"]
            )
            if initial["fingerprint"] != locked["fingerprint"]:
                raise StrategyError(
                    "strategy project context evidence changed before registration"
                )
            if _external_identity(initial_external) != _external_identity(
                locked_external
            ):
                raise StrategyError(
                    "external strategy report changed before registration"
                )
            _require_external_total_with_registered(locked, locked_external)
            # Snapshots contain metadata only, but drop the redundant first
            # pass before staging so the lifetime remains explicit.
            initial_external = []

            current = StrategyProjectContextRepository.get_current_on_connection(
                conn, task_id
            )
            state = _build_state(
                task_id=task_id,
                request=request,
                evidence=locked,
                external=locked_external,
                tasks_root=tasks_root,
                current=current,
            )
            candidate = build_strategy_project_context_revision(
                state=state,
                revision=request["expected_revision"] + 1,
                parent_revision_id=request["expected_revision_id"],
                parent_state_hash=request["expected_state_hash"],
                operation_kind="materialize_project_context",
            )
            mode = _classify_write(
                current=current,
                candidate=candidate,
                request=request,
            )

            if mode in {"replay", "unchanged"}:
                assert current is not None
                persisted = _load_revision_artifact_on_connection(
                    conn,
                    tasks_root=tasks_root,
                    task_id=task_id,
                    revision=current,
                )
                context_record = _context_artifact_record_on_connection(
                    conn,
                    tasks_root=tasks_root,
                    task_id=task_id,
                    revision_id=current["revision_id"],
                )
                requested_external_records = _external_records_on_connection(
                    conn,
                    tasks_root=tasks_root,
                    task_id=task_id,
                    snapshots=locked_external,
                )
            else:
                staged_external_count = _stage_external_sources(
                    uow=uow,
                    tasks_root=tasks_root,
                    task_id=task_id,
                    source_dir=locked["task"]["source_dir"],
                    snapshots=locked_external,
                )
                canonical = canonical_strategy_project_context_revision_json(
                    candidate
                ).encode("utf-8")
                context_hash = hashlib.sha256(canonical).hexdigest()
                context_path = _context_artifact_path(
                    tasks_root, task_id=task_id, revision_id=candidate["revision_id"]
                )
                staged_context = _stage_new_file(
                    uow,
                    root=context_path.parent,
                    final_path=context_path,
                    data=canonical,
                    expected_hash=context_hash,
                    tasks_root=tasks_root,
                )
                if staged_context is not None or staged_external_count:
                    uow.promote_all()
                    promoted = True
                _verify_regular_file(
                    context_path,
                    root=tasks_root,
                    expected_hash=context_hash,
                    expected_bytes=canonical,
                    max_bytes=MAX_PROJECT_CONTEXT_JSON_BYTES,
                )
                # External files share the same UoW and have now been promoted;
                # register them before the context row as required by its refs.
                requested_external_records = _register_external_sources(
                    conn,
                    task_artifacts=repositories["artifacts"],
                    tasks_root=tasks_root,
                    task_id=task_id,
                    snapshots=locked_external,
                )
                provenance = {
                    "schema_version": PROJECT_CONTEXT_ARTIFACT_SCHEMA_VERSION,
                    "task_id": task_id,
                    "revision_id": candidate["revision_id"],
                    "revision": candidate["revision"],
                    "revision_content_hash": candidate["content_hash"],
                    "state_hash": candidate["state_hash"],
                    "operation_hash": candidate["operation_hash"],
                    "format": "json",
                }
                context_record = repositories["artifacts"].register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=PROJECT_CONTEXT_ARTIFACT_KIND,
                    path=str(context_path),
                    content_hash=context_hash,
                    origin_tool=PROJECT_CONTEXT_ORIGIN_TOOL,
                    provenance=provenance,
                )
                persisted = StrategyProjectContextRepository.refresh_on_connection(
                    conn,
                    revision=candidate,
                    expected_revision=request["expected_revision"],
                    expected_revision_id=request["expected_revision_id"],
                    expected_state_hash=request["expected_state_hash"],
                )
                if persisted["revision_id"] != candidate["revision_id"]:
                    raise StrategyError(
                        "project context head unexpectedly resolved another revision"
                    )
                _write_audit_row(
                    conn,
                    kind="strategy.project_context.materialize",
                    target_ref=task_id,
                    inputs_hash=_request_hash(request),
                    outcome="succeeded",
                    detail={
                        "task_id": task_id,
                        "revision": persisted["revision"],
                        "revision_id": persisted["revision_id"],
                        "state_hash": persisted["state_hash"],
                        "artifact_id": context_record["id"],
                        "external_artifact_ids": [
                            item["id"] for item in requested_external_records
                        ],
                        "external_report_filenames": request[
                            "external_report_filenames"
                        ],
                    },
                )
                # Close the filesystem TOCTOU window before committing pointers.
                final_evidence = _discover_evidence(
                    conn,
                    task_id=task_id,
                    request=request,
                    runtime=runtime,
                    tasks_root=tasks_root,
                    include_context_artifacts=False,
                )
                final_external = _snapshot_external_reports(
                    locked["task"]["source_dir"],
                    request["external_report_filenames"],
                )
                requested_hashes = {item.content_hash for item in locked_external}
                if _evidence_fingerprint_without_external_hashes(
                    locked, requested_hashes
                ) != _evidence_fingerprint_without_external_hashes(
                    final_evidence, requested_hashes
                ):
                    raise StrategyError(
                        "strategy project context evidence changed during registration"
                    )
                if _external_identity(locked_external) != _external_identity(
                    final_external
                ):
                    raise StrategyError(
                        "external strategy report changed during registration"
                    )
                created = True
            conn.commit()
            committed = True
        if promoted:
            uow.commit()
        else:
            uow.rollback()
    except Exception as exc:
        if not committed:
            uow.rollback()
        if isinstance(exc, StrategyError):
            raise
        if isinstance(exc, StrategyProjectContextConflictError):
            raise StrategyError(str(exc)) from exc
        raise StrategyError(
            f"could not materialize strategy project context: {exc}"
        ) from exc

    assert context_record is not None
    output = {
        "schema_version": PROJECT_CONTEXT_TOOL_SCHEMA_VERSION,
        "created": created,
        "revision": persisted,
        "context_artifact": _artifact_output(
            context_record, task_id=task_id, format_name="json"
        ),
        "external_artifacts": [
            _artifact_output(
                record, task_id=task_id, format_name=Path(record["path"]).suffix[1:]
            )
            for record in requested_external_records
        ],
        "missing_information_records": persisted["state"][
            "missing_information_records"
        ],
        "warnings": [flag["message"] for flag in persisted["state"]["red_flags"]],
    }
    return validate_materialize_project_context_tool_output(
        output,
        expected_context_artifact=_artifact_output(
            context_record, task_id=task_id, format_name="json"
        ),
        expected_external_artifacts=[
            _artifact_output(
                record,
                task_id=task_id,
                format_name=Path(record["path"]).suffix[1:],
            )
            for record in requested_external_records
        ],
    )


def load_strategy_project_context_revision_for_audit(
    runtime,
    *,
    task_id: str,
    revision_id: str,
) -> dict[str, Any]:
    """Load immutable historical evidence without requiring live refs to survive."""

    normalized_task = _text(task_id, "task_id")
    normalized_revision = _text(revision_id, "revision_id")
    repository = StrategyProjectContextRepository(runtime.settings.db_path)
    revision = repository.get_revision_by_id(normalized_task, normalized_revision)
    if revision is None:
        raise StrategyError("strategy project context revision not found")
    with connect(runtime.settings.db_path) as conn:
        return _load_revision_artifact_on_connection(
            conn,
            tasks_root=Path(runtime.settings.tasks_dir),
            task_id=normalized_task,
            revision=revision,
        )


def load_current_strategy_project_context(
    runtime,
    *,
    task_id: str,
) -> dict[str, Any] | None:
    """Load current context, verifying its artifact and every still-live source."""

    normalized_task = _text(task_id, "task_id")
    repository = StrategyProjectContextRepository(runtime.settings.db_path)
    with repository.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        current = StrategyProjectContextRepository.get_current_on_connection(
            conn, normalized_task
        )
        if current is None:
            return None
        loaded = _load_revision_artifact_on_connection(
            conn,
            tasks_root=Path(runtime.settings.tasks_dir),
            task_id=normalized_task,
            revision=current,
        )
        _verify_live_refs(
            conn,
            runtime=runtime,
            tasks_root=Path(runtime.settings.tasks_dir),
            task_id=normalized_task,
            source_refs=loaded["state"]["source_refs"],
        )
        return loaded


def validate_materialize_project_context_tool_output(
    value: object,
    *,
    expected_context_artifact: Mapping[str, Any] | None = None,
    expected_external_artifacts: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _OUTPUT_FIELDS:
        raise StrategyError("materialize_project_context output shape is invalid")
    if value["schema_version"] != PROJECT_CONTEXT_TOOL_SCHEMA_VERSION:
        raise StrategyError(
            "materialize_project_context output schema_version is invalid"
        )
    if not isinstance(value["created"], bool):
        raise StrategyError(
            "materialize_project_context output created must be boolean"
        )
    revision = strategy_project_context_revision_from_json(
        canonical_strategy_project_context_revision_json(value["revision"])
    )
    task_id = revision["state"]["task_id"]
    context_artifact = _validate_artifact_output(
        value["context_artifact"], task_id=task_id
    )
    if context_artifact["kind"] != PROJECT_CONTEXT_ARTIFACT_KIND:
        raise StrategyError("materialize_project_context context artifact kind drifted")
    canonical_revision = canonical_strategy_project_context_revision_json(
        revision
    ).encode("utf-8")
    expected_context_hash = hashlib.sha256(canonical_revision).hexdigest()
    if (
        context_artifact["format"] != "json"
        or context_artifact["filename"] != f"{revision['revision_id']}.json"
        or not hmac.compare_digest(
            context_artifact["content_hash"], expected_context_hash
        )
    ):
        raise StrategyError(
            "materialize_project_context context artifact is not bound to revision"
        )
    if expected_context_artifact is not None and context_artifact != dict(
        expected_context_artifact
    ):
        raise StrategyError(
            "materialize_project_context context artifact registration drifted"
        )
    external = value["external_artifacts"]
    if not isinstance(external, list):
        raise StrategyError(
            "materialize_project_context external_artifacts must be a list"
        )
    external_normalized = [
        _validate_artifact_output(item, task_id=task_id) for item in external
    ]
    if any(
        item["kind"] != PROJECT_CONTEXT_EXTERNAL_ARTIFACT_KIND
        for item in external_normalized
    ):
        raise StrategyError(
            "materialize_project_context external artifact kind drifted"
        )
    if len({item["artifact_id"] for item in external_normalized}) != len(
        external_normalized
    ):
        raise StrategyError(
            "materialize_project_context external artifacts contain duplicates"
        )
    source_refs = revision["state"]["source_refs"]
    source_identities = {
        (item["kind"], item["ref_id"], item["content_hash"]) for item in source_refs
    }
    for item in external_normalized:
        suffix = Path(item["filename"]).suffix.lower()
        if (
            suffix not in _EXTERNAL_SUFFIXES
            or item["format"] != suffix[1:]
            or item["filename"] != f"{item['content_hash']}{suffix}"
            or (
                "task_artifact",
                item["artifact_id"],
                item["content_hash"],
            )
            not in source_identities
            or (
                "external_report",
                item["content_hash"],
                item["content_hash"],
            )
            not in source_identities
        ):
            raise StrategyError(
                "materialize_project_context external artifact is not bound to revision sources"
            )
    if expected_external_artifacts is not None and external_normalized != [
        dict(item) for item in expected_external_artifacts
    ]:
        raise StrategyError(
            "materialize_project_context external artifact registration drifted"
        )
    missing = value["missing_information_records"]
    if missing != revision["state"]["missing_information_records"]:
        raise StrategyError("materialize_project_context missing information drifted")
    warnings = value["warnings"]
    if not isinstance(warnings, list) or any(
        not isinstance(item, str) for item in warnings
    ):
        raise StrategyError("materialize_project_context warnings must contain strings")
    return {
        "schema_version": PROJECT_CONTEXT_TOOL_SCHEMA_VERSION,
        "created": value["created"],
        "revision": revision,
        "context_artifact": context_artifact,
        "external_artifacts": external_normalized,
        "missing_information_records": missing,
        "warnings": list(warnings),
    }


def _validate_inputs(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise StrategyError("materialize_project_context inputs must be an object")
    missing = sorted(_REQUIRED_INPUT_FIELDS - set(value))
    unexpected = sorted(set(value) - _INPUT_FIELDS)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unsupported: " + ", ".join(unexpected))
        raise StrategyError(
            "invalid materialize_project_context inputs (" + "; ".join(details) + ")"
        )
    revision = _non_negative_int(value["expected_revision"], "expected_revision")
    revision_id = _optional_text(
        value.get("expected_revision_id"), "expected_revision_id"
    )
    if revision_id is not None and _REVISION_ID_RE.fullmatch(revision_id) is None:
        raise StrategyError("expected_revision_id is invalid")
    state_hash = _optional_hash(value.get("expected_state_hash"), "expected_state_hash")
    if revision == 0:
        if revision_id is not None or state_hash is not None:
            raise StrategyError("absent project context CAS requires null id and hash")
    elif revision_id is None or state_hash is None:
        raise StrategyError(
            "existing project context CAS requires revision id and state hash"
        )
    message = value["user_message_ref"]
    if not isinstance(message, Mapping) or set(message) != {
        "message_id",
        "content_hash",
    }:
        raise StrategyError("user_message_ref must contain message_id and content_hash")
    as_of = _iso_date(value["as_of"], "as_of")
    scope = _optional_bounded_text(value.get("scope"), "scope", MAX_SCOPE_CHARS)
    business = value["business_context"]
    if not isinstance(business, Mapping) or any(
        not isinstance(key, str) for key in business
    ):
        raise StrategyError("business_context must be an object")
    if len(business) > MAX_BUSINESS_CONTEXT_FIELDS:
        raise StrategyError("business_context has too many fields")
    normalized_business: dict[str, str | None] = {}
    for key in sorted(business):
        normalized_key = _bounded_text(
            key, "business_context field path", MAX_BUSINESS_CONTEXT_KEY_CHARS
        )
        if _FIELD_PATH_RE.fullmatch(normalized_key) is None:
            raise StrategyError("business_context field path is invalid")
        if normalized_key in normalized_business:
            raise StrategyError(
                "business_context contains duplicate normalized field paths"
            )
        normalized_business[normalized_key] = _optional_bounded_text(
            business[key],
            f"business_context.{normalized_key}",
            MAX_BUSINESS_CONTEXT_VALUE_CHARS,
        )
    explicit = _bounded_text_list(
        value["explicit_unavailable"],
        field="explicit_unavailable",
        maximum=MAX_EXPLICIT_UNAVAILABLE_FIELDS,
        item_maximum=MAX_BUSINESS_CONTEXT_KEY_CHARS,
    )
    if any(_FIELD_PATH_RE.fullmatch(item) is None for item in explicit):
        raise StrategyError("explicit_unavailable field path is invalid")
    unsupported_unavailable = sorted(
        set(explicit) - (set(_KNOWN_MISSING) | set(normalized_business))
    )
    if unsupported_unavailable:
        raise StrategyError(
            "explicit_unavailable contains unsupported field paths: "
            + ", ".join(unsupported_unavailable)
        )
    external = _bounded_text_list(
        value["external_report_filenames"],
        field="external_report_filenames",
        maximum=MAX_EXTERNAL_REPORTS,
        item_maximum=MAX_EXTERNAL_FILENAME_CHARS,
    )
    return {
        "expected_revision": revision,
        "expected_revision_id": revision_id,
        "expected_state_hash": state_hash,
        "user_message_ref": {
            "message_id": _text(message["message_id"], "user_message_ref.message_id"),
            "content_hash": _hash(
                message["content_hash"], "user_message_ref.content_hash"
            ),
        },
        "as_of": as_of,
        "scope": scope,
        "business_context": normalized_business,
        "explicit_unavailable": explicit,
        "external_report_filenames": external,
    }


def _repositories(runtime, *, db_path: Path) -> dict[str, Any]:
    return {
        "contexts": getattr(
            runtime, "project_contexts", StrategyProjectContextRepository(db_path)
        ),
        "artifacts": getattr(
            runtime, "task_artifacts", TaskArtifactRepository(db_path)
        ),
    }


def _discover_evidence(
    conn,
    *,
    task_id: str,
    request: Mapping[str, Any],
    runtime,
    tasks_root: Path,
    include_context_artifacts: bool = False,
) -> dict[str, Any]:
    task = conn.execute(
        "SELECT id, task_type, source_dir FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    if task is None:
        raise StrategyError(f"task not found: {task_id}")
    if str(task["task_type"]) != "strategy":
        raise StrategyError("StrategyProjectContext requires a strategy task")
    message = conn.execute(
        "SELECT id, task_id, role, content, created_at FROM agent_messages WHERE id = ?",
        (request["user_message_ref"]["message_id"],),
    ).fetchone()
    if (
        message is None
        or str(message["task_id"]) != task_id
        or str(message["role"]) != "user"
    ):
        raise StrategyError(
            "user_message_ref must identify a persisted task-owned user message"
        )
    message_hash = hashlib.sha256(str(message["content"]).encode("utf-8")).hexdigest()
    if not hmac.compare_digest(
        message_hash, request["user_message_ref"]["content_hash"]
    ):
        raise StrategyError("user_message_ref content hash changed")
    _validate_external_report_message_bindings(
        source_dir=str(task["source_dir"]),
        filenames=request["external_report_filenames"],
        persisted_message=str(message["content"]),
    )
    message_ref = build_source_ref(
        kind="agent_message", ref_id=str(message["id"]), content_hash=message_hash
    )

    datasets, dataset_by_id = _discover_datasets(
        conn, task_id=task_id, datasets_root=Path(runtime.settings.datasets_dir)
    )
    workspace = _discover_workspace(conn, task_id=task_id, dataset_by_id=dataset_by_id)
    artifacts = _discover_task_artifacts(
        conn,
        task_id=task_id,
        tasks_root=tasks_root,
        dataset_by_id=dataset_by_id,
        include_context_artifacts=include_context_artifacts,
    )
    strategies = _discover_strategies(
        conn, task_id=task_id, dataset_by_id=dataset_by_id
    )
    monitoring = _discover_monitoring(
        conn,
        task_id=task_id,
        strategies=strategies,
        dataset_by_id=dataset_by_id,
    )
    descriptor = {
        "task": {
            "id": task_id,
            "task_type": "strategy",
            "source_dir": str(task["source_dir"]),
        },
        "message": {
            "id": str(message["id"]),
            "content_hash": message_hash,
            "created_at": str(message["created_at"]),
        },
        "datasets": datasets,
        "workspace": workspace,
        "artifacts": artifacts,
        "strategies": strategies,
        "monitoring": monitoring,
    }
    return {
        **descriptor,
        "message_ref": message_ref,
        "fingerprint": _sha256_json(descriptor),
    }


def _discover_datasets(
    conn, *, task_id: str, datasets_root: Path
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = conn.execute(
        "SELECT id, task_id, source_path, content_hash, role, created_at FROM datasets WHERE task_id = ? ORDER BY created_at, id",
        (task_id,),
    ).fetchall()
    items: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        content_hash = _hash(row["content_hash"], "dataset.content_hash")
        relative = Path(_text(row["source_path"], "dataset.source_path"))
        if relative.is_absolute() or ".." in relative.parts:
            raise StrategyError("task dataset source path is unsafe")
        path = datasets_root / relative
        _verify_regular_file(path, root=datasets_root, expected_hash=content_hash)
        item = {
            "id": str(row["id"]),
            "task_id": str(row["task_id"]),
            "content_hash": content_hash,
            "source_path": relative.as_posix(),
            "role": str(row["role"]),
            "created_at": str(row["created_at"]),
            "ref": build_source_ref(
                kind="dataset", ref_id=str(row["id"]), content_hash=content_hash
            ),
        }
        items.append(item)
        by_id[item["id"]] = item
    return items, by_id


def _discover_workspace(
    conn, *, task_id: str, dataset_by_id: Mapping[str, Any]
) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM data_workspaces WHERE task_id = ?", (task_id,)
    ).fetchone()
    if row is None:
        return None
    active_id = (
        None if row["active_dataset_id"] is None else str(row["active_dataset_id"])
    )
    active_hash = (
        None
        if row["active_dataset_content_hash"] is None
        else _hash(
            row["active_dataset_content_hash"], "workspace.active_dataset_content_hash"
        )
    )
    if active_id is not None:
        dataset = dataset_by_id.get(active_id)
        if dataset is None or dataset["content_hash"] != active_hash:
            raise StrategyError(
                "DataWorkspace active dataset does not belong to the task"
            )
    try:
        semantic = json.loads(str(row["semantic_mapping_json"]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StrategyError("DataWorkspace semantic mapping is invalid") from exc
    body = {
        "schema_version": str(row["schema_version"]),
        "task_id": task_id,
        "revision": int(row["revision"]),
        "active_dataset_id": active_id,
        "active_dataset_content_hash": active_hash,
        "analysis_generation": int(row["analysis_generation"]),
        "semantic_mapping": semantic,
    }
    content_hash = _sha256_json(body)
    return {
        **body,
        "content_hash": content_hash,
        "ref": build_source_ref(
            kind="workspace",
            ref_id=f"{task_id}:r{body['revision']}",
            content_hash=content_hash,
        ),
    }


def _discover_task_artifacts(
    conn,
    *,
    task_id: str,
    tasks_root: Path,
    dataset_by_id: Mapping[str, Any],
    include_context_artifacts: bool,
) -> dict[str, list[dict[str, Any]]]:
    kinds = [
        SAMPLE_DESIGN_ARTIFACT_KIND,
        POOL_IMPACT_ARTIFACT_KIND,
        PROJECT_CONTEXT_EXTERNAL_ARTIFACT_KIND,
    ]
    if include_context_artifacts:
        kinds.append(PROJECT_CONTEXT_ARTIFACT_KIND)
    placeholders = ",".join("?" for _ in kinds)
    rows = conn.execute(
        f"SELECT * FROM task_artifacts WHERE task_id = ? AND kind IN ({placeholders}) ORDER BY created_at, id",
        (task_id, *kinds),
    ).fetchall()
    result = {"sample_designs": [], "pool_impacts": [], "external_reports": []}
    external_total_size = 0
    for row in rows:
        kind = str(row["kind"])
        if kind == PROJECT_CONTEXT_ARTIFACT_KIND:
            continue
        record = _artifact_row(row, task_id=task_id, tasks_root=tasks_root)
        raw: bytes | None = None
        external_size: int | None = None
        if kind == PROJECT_CONTEXT_EXTERNAL_ARTIFACT_KIND:
            external_size, external_hash = _stream_regular_nofollow(
                Path(record["path"]), max_bytes=MAX_EXTERNAL_REPORT_BYTES
            )
            if not hmac.compare_digest(external_hash, record["content_hash"]):
                raise StrategyError(
                    "task artifact bytes do not match its registry hash"
                )
            external_total_size += external_size
            if external_total_size > MAX_EXTERNAL_REPORT_TOTAL_BYTES:
                raise StrategyError(
                    "registered external reports exceed total byte limit"
                )
        else:
            raw = _read_verified_registered_artifact(record, tasks_root=tasks_root)
        if kind == SAMPLE_DESIGN_ARTIFACT_KIND:
            assert raw is not None
            if record["origin_tool"] != SAMPLE_DESIGN_ORIGIN_TOOL:
                raise StrategyError("strategy sample-design artifact origin changed")
            bundle = strategy_sample_design_bundle_from_json(raw)
            canonical = canonical_strategy_sample_design_bundle_json(bundle).encode(
                "utf-8"
            )
            if raw != canonical:
                raise StrategyError(
                    "strategy sample-design artifact bytes are not canonical"
                )
            provenance = record["provenance"]
            design = bundle["sample_design"]
            dataset_ref = design["identity"]["dataset_ref"]
            if (
                set(provenance) != _SAMPLE_PROVENANCE_FIELDS
                or Path(record["path"])
                != Path(tasks_root).absolute()
                / task_id
                / "strategy_sample_designs"
                / f"{design['sample_design_id']}.json"
                or provenance.get("schema_version")
                != SAMPLE_DESIGN_ARTIFACT_SCHEMA_VERSION
                or provenance.get("producer_version")
                != STRATEGY_SAMPLE_DESIGN_PRODUCER_VERSION
                or provenance.get("format") != "json"
                or provenance.get("task_id") != task_id
                or provenance.get("sample_design_id") != design["sample_design_id"]
                or provenance.get("sample_design_content_hash")
                != design["content_hash"]
                or provenance.get("bundle_id") != bundle["bundle_id"]
                or provenance.get("bundle_content_hash") != bundle["content_hash"]
                or provenance.get("dataset_id") != dataset_ref["dataset_id"]
                or provenance.get("dataset_content_hash") != dataset_ref["content_hash"]
                or provenance.get("request_hash")
                != _sha256_json(provenance.get("request"))
            ):
                raise StrategyError(
                    "strategy sample-design artifact provenance changed"
                )
            dataset = dataset_by_id.get(dataset_ref["dataset_id"])
            if (
                dataset is None
                or dataset["content_hash"] != dataset_ref["content_hash"]
            ):
                raise StrategyError("strategy sample-design dataset binding changed")
            result["sample_designs"].append({**record, "bundle": bundle})
        elif kind == POOL_IMPACT_ARTIFACT_KIND:
            assert raw is not None
            if record["origin_tool"] != POOL_IMPACT_ORIGIN_TOOL:
                raise StrategyError("Pool impact artifact origin changed")
            try:
                payload = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise StrategyError("Pool impact artifact JSON is invalid") from exc
            assessment = validate_strategy_pool_impact_assessment(payload)
            canonical = canonical_strategy_pool_impact_json(assessment).encode("utf-8")
            if raw != canonical:
                raise StrategyError("Pool impact artifact bytes are not canonical")
            provenance = record["provenance"]
            identity = assessment["identity"]
            if (
                set(provenance) != _POOL_PROVENANCE_FIELDS
                or Path(record["path"])
                != Path(tasks_root).absolute()
                / task_id
                / "strategy_pool_impacts"
                / f"{assessment['assessment_id']}.json"
                or provenance.get("schema_version")
                != POOL_IMPACT_ARTIFACT_SCHEMA_VERSION
                or provenance.get("producer_version")
                != STRATEGY_POOL_IMPACT_PRODUCER_VERSION
                or provenance.get("task_id") != task_id
                or provenance.get("assessment_id") != assessment["assessment_id"]
                or provenance.get("assessment_content_hash")
                != assessment["content_hash"]
                or provenance.get("pool_id") != identity["pool_id"]
                or provenance.get("pool_revision") != identity["revision"]
                or provenance.get("pool_snapshot_hash") != identity["snapshot_hash"]
                or provenance.get("design_hash") != identity["design_hash"]
                or provenance.get("strategy_spec_hash")
                != identity["strategy_spec_hash"]
                or provenance.get("dataset_id")
                != assessment["bindings"]["sample"]["dataset_id"]
                or provenance.get("dataset_content_hash")
                != assessment["bindings"]["sample"]["dataset_content_hash"]
                or provenance.get("comparison_mode")
                != assessment["bindings"]["comparison_mode"]
            ):
                raise StrategyError("Pool impact artifact provenance changed")
            dataset_id = assessment["bindings"]["sample"]["dataset_id"]
            dataset_hash = assessment["bindings"]["sample"]["dataset_content_hash"]
            dataset = dataset_by_id.get(dataset_id)
            if dataset is None or dataset["content_hash"] != dataset_hash:
                raise StrategyError("Pool impact dataset binding changed")
            result["pool_impacts"].append({**record, "assessment": assessment})
        else:
            provenance = record["provenance"]
            if set(provenance) != {
                "schema_version",
                "task_id",
                "content_hash",
                "content_size",
                "suffix",
            }:
                raise StrategyError("external strategy report provenance is invalid")
            if (
                record["origin_tool"] != PROJECT_CONTEXT_ORIGIN_TOOL
                or provenance["schema_version"]
                != PROJECT_CONTEXT_EXTERNAL_SCHEMA_VERSION
                or provenance["task_id"] != task_id
                or provenance["content_hash"] != record["content_hash"]
                or provenance["content_size"] != external_size
                or provenance["suffix"] != Path(record["path"]).suffix.lower()
                or Path(record["path"])
                != Path(tasks_root).absolute()
                / task_id
                / "strategy_project_context_sources"
                / f"{record['content_hash']}{provenance['suffix']}"
            ):
                raise StrategyError("external strategy report provenance changed")
            result["external_reports"].append(record)
    return result


def _discover_strategies(
    conn, *, task_id: str, dataset_by_id: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM strategies WHERE task_id = ? ORDER BY version, created_at, id",
        (task_id,),
    ).fetchall()
    all_evidence = {
        str(row["id"]): _strategy_evidence(row)
        for row in conn.execute("SELECT * FROM strategies").fetchall()
    }
    for item in all_evidence.values():
        parent_id = item["parent_strategy_id"]
        item["parent"] = None if parent_id is None else all_evidence.get(parent_id)
        if parent_id is not None and item["parent"] is None:
            raise StrategyError("strategy parent lineage references a missing strategy")
    result: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        item = all_evidence[str(row["id"])]
        parent = item["parent"]
        if parent is not None:
            if (
                parent["strategy_type"] != item["strategy_type"]
                or parent["version"] >= item["version"]
            ):
                raise StrategyError("strategy parent lineage is inconsistent")
        by_id[item["id"]] = item
        result.append(item)
    for item in result:
        chain: set[str] = set()
        current = item
        while current["parent"] is not None:
            if current["id"] in chain:
                raise StrategyError("strategy parent lineage contains a cycle")
            chain.add(current["id"])
            current = current["parent"]
    backtest_rows = conn.execute(
        "SELECT b.* FROM backtests b JOIN strategies s ON s.id = b.strategy_id WHERE s.task_id = ? ORDER BY b.created_at, b.id",
        (task_id,),
    ).fetchall()
    for row in backtest_rows:
        strategy = by_id.get(str(row["strategy_id"]))
        if strategy is None:
            raise StrategyError("backtest does not belong to a task strategy")
        try:
            payload = json.loads(str(row["result_json"]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise StrategyError("persisted backtest JSON is invalid") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != STRATEGY_BACKTEST_SCHEMA_VERSION
        ):
            # Legacy backtests are deliberately not promoted into V2 project context.
            continue
        result_obj = StrategyBacktestResult.from_dict(payload)
        normalized = result_obj.to_dict()
        if (
            result_obj.strategy_id != strategy["id"]
            or result_obj.strategy_type != strategy["strategy_type"]
        ):
            raise StrategyError("typed backtest strategy binding changed")
        if (
            result_obj.normalized_input["strategy_effect_hash"]
            != strategy["effect_hash"]
        ):
            raise StrategyError("typed backtest strategy effect hash changed")
        dataset = dataset_by_id.get(str(row["dataset_id"]))
        if dataset is None:
            raise StrategyError(
                "typed backtest dataset does not belong to the strategy task"
            )
        content_hash = _sha256_json(normalized)
        strategy["backtests"].append(
            {
                "id": str(row["id"]),
                "dataset_id": str(row["dataset_id"]),
                "created_at": str(row["created_at"]),
                "content_hash": content_hash,
                "result": normalized,
                "ref": build_source_ref(
                    kind="backtest", ref_id=str(row["id"]), content_hash=content_hash
                ),
            }
        )
    return result


def _strategy_evidence(row) -> dict[str, Any]:
    dsl_raw = row["dsl_json"]
    if dsl_raw is None:
        raise StrategyError("strategy is missing canonical DSL evidence")
    try:
        dsl = json.loads(str(dsl_raw))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StrategyError("strategy DSL JSON is invalid") from exc
    spec = parse_strategy_spec(dsl)
    content_hash = hashlib.sha256(
        canonical_strategy_json(spec).encode("utf-8")
    ).hexdigest()
    effect_hash = strategy_spec_hash(spec)
    stored_hash = row["dsl_content_hash"]
    if stored_hash is None or not hmac.compare_digest(
        _hash(stored_hash, "strategy.dsl_content_hash"), content_hash
    ):
        raise StrategyError("strategy DSL content hash changed")
    if spec.strategy_type != str(row["strategy_type"]):
        raise StrategyError("strategy DSL type does not match strategy row")
    return {
        "id": str(row["id"]),
        "task_id": str(row["task_id"]),
        "strategy_type": str(row["strategy_type"]),
        "version": int(row["version"]),
        "status": str(row["status"]),
        "asset_status": str(row["asset_status"]),
        "adopted_at": None if row["adopted_at"] is None else str(row["adopted_at"]),
        "adoption_reason": None
        if row["adoption_reason"] is None
        else str(row["adoption_reason"]),
        "parent_strategy_id": None
        if row["parent_strategy_id"] is None
        else str(row["parent_strategy_id"]),
        "created_at": str(row["created_at"]),
        "description": str(row["description"]),
        "content_hash": content_hash,
        "effect_hash": effect_hash,
        "rules": spec.to_dict()["rules"],
        "ref": build_source_ref(
            kind="strategy", ref_id=str(row["id"]), content_hash=content_hash
        ),
        "backtests": [],
    }


def _discover_monitoring(
    conn,
    *,
    task_id: str,
    strategies: Sequence[Mapping[str, Any]],
    dataset_by_id: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    strategy_ids = {item["id"] for item in strategies}
    if not strategy_ids:
        return {"plans": [], "runs": []}
    repository = StrategyMonitoringRepository(
        Path(conn.execute("PRAGMA database_list").fetchone()[2])
    )
    plans: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for strategy_id in sorted(strategy_ids):
        for record in repository.list_plans(strategy_id):
            plans.append(
                {
                    "id": record.id,
                    "strategy_id": record.strategy_id,
                    "revision": record.revision,
                    "content_hash": record.payload_hash,
                    "created_at": record.created_at,
                    "ref": build_source_ref(
                        kind="monitoring_plan",
                        ref_id=record.id,
                        content_hash=record.payload_hash,
                    ),
                }
            )
        for record in repository.list_runs(strategy_id):
            dataset = dataset_by_id.get(record.dataset_id)
            if (
                dataset is None
                or dataset["content_hash"] != record.dataset_content_hash
            ):
                raise StrategyError(
                    "monitoring run dataset does not belong to the strategy task"
                )
            runs.append(
                {
                    "id": record.id,
                    "strategy_id": record.strategy_id,
                    "monitoring_plan_id": record.monitoring_plan_id,
                    "dataset_id": record.dataset_id,
                    "dataset_content_hash": record.dataset_content_hash,
                    "content_hash": record.result_hash,
                    "overall_level": record.overall_level,
                    "result": record.result,
                    "created_at": record.created_at,
                    "ref": build_source_ref(
                        kind="monitoring_run",
                        ref_id=record.id,
                        content_hash=record.result_hash,
                    ),
                }
            )
    return {"plans": plans, "runs": runs}


def _sample_design_proves_cutoff(artifact: Mapping[str, Any], *, as_of: str) -> bool:
    if not _timestamp_on_or_before_as_of(
        artifact["created_at"], as_of=as_of, field="sample_design.created_at"
    ):
        return False
    observation_window = artifact["bundle"]["sample_design"]["observation_window"]
    return (
        observation_window["status"] == "provided"
        and observation_window["end"] <= as_of
    )


def _strategies_at_cutoff(
    strategies: Sequence[Mapping[str, Any]], *, as_of: str
) -> list[dict[str, Any]]:
    def lineage_is_available(item: Mapping[str, Any], seen: set[str]) -> bool:
        if item["id"] in seen:
            raise StrategyError("strategy parent lineage contains a cycle")
        if not _timestamp_on_or_before_as_of(
            item["created_at"], as_of=as_of, field="strategy.created_at"
        ):
            return False
        parent = item["parent"]
        if parent is None:
            return True
        return lineage_is_available(parent, {*seen, item["id"]})

    result = []
    for item in strategies:
        if not lineage_is_available(item, set()):
            continue
        normalized = dict(item)
        normalized["backtests"] = [
            backtest
            for backtest in item["backtests"]
            if _timestamp_on_or_before_as_of(
                backtest["created_at"],
                as_of=as_of,
                field="backtest.created_at",
            )
        ]
        result.append(normalized)
    return result


def _build_state(
    *,
    task_id: str,
    request: Mapping[str, Any],
    evidence: Mapping[str, Any],
    external: Sequence[_ExternalSnapshot],
    tasks_root: Path,
    current: Mapping[str, Any] | None,
) -> dict[str, Any]:
    message_ref = evidence["message_ref"]
    explicit = set(request["explicit_unavailable"])
    as_of = request["as_of"]
    datasets = [
        item
        for item in evidence["datasets"]
        if _timestamp_on_or_before_as_of(
            item["created_at"], as_of=as_of, field="dataset.created_at"
        )
    ]
    sample_designs = [
        item
        for item in evidence["artifacts"]["sample_designs"]
        if _sample_design_proves_cutoff(item, as_of=as_of)
    ]
    eligible_sample_artifact_ids = {item["id"] for item in sample_designs}
    pool_impacts = [
        item
        for item in evidence["artifacts"]["pool_impacts"]
        if _timestamp_on_or_before_as_of(
            item["created_at"], as_of=as_of, field="pool_impact.created_at"
        )
        and item["assessment"]["bindings"]["sample_design_ref"]["artifact_id"]
        in eligible_sample_artifact_ids
    ]
    sample = sample_designs[-1] if sample_designs else None
    pool = pool_impacts[-1] if pool_impacts else None
    strategies = _strategies_at_cutoff(evidence["strategies"], as_of=as_of)
    monitoring = {
        category: [
            item
            for item in evidence["monitoring"][category]
            if item["strategy_id"] in {strategy["id"] for strategy in strategies}
            and _timestamp_on_or_before_as_of(
                item["created_at"],
                as_of=as_of,
                field=f"monitoring_{category[:-1]}.created_at",
            )
        ]
        for category in ("plans", "runs")
    }
    state_evidence = {
        **evidence,
        "datasets": datasets,
        "strategies": strategies,
        "monitoring": monitoring,
        "artifacts": {
            **evidence["artifacts"],
            "sample_designs": sample_designs,
            "pool_impacts": pool_impacts,
        },
    }
    champion_candidates = [
        item
        for item in strategies
        if item["asset_status"] == "adopted_local"
        and item["status"] == "adopted"
        and item["adopted_at"] is not None
        and _timestamp_on_or_before_as_of(
            item["adopted_at"], as_of=as_of, field="strategy.adopted_at"
        )
    ]
    champion = champion_candidates[-1] if champion_candidates else None
    latest_backtest = None
    if champion is not None and champion["backtests"]:
        latest_backtest = champion["backtests"][-1]
    elif strategies:
        all_backtests = [
            item for strategy in strategies for item in strategy["backtests"]
        ]
        latest_backtest = (
            max(all_backtests, key=lambda item: (item["created_at"], item["id"]))
            if all_backtests
            else None
        )
    approval_backtests = [
        item
        for strategy in strategies
        for item in strategy["backtests"]
        if item["result"]["strategy_type"] in {"approval", "reject"}
    ]
    latest_approval_backtest = (
        max(
            approval_backtests,
            key=lambda item: (item["created_at"], item["id"]),
        )
        if approval_backtests
        else None
    )

    metric_def_refs: list[dict[str, str]] = []
    metric_obs_refs: list[dict[str, str]] = []
    segment_obs_refs: list[dict[str, str]] = []
    volume_field = _absent_field("current.status_fields.volume", explicit=explicit)
    risk_field = _absent_field("current.status_fields.risk", explicit=explicit)
    maturity_field = _absent_field("current.maturity_summary", explicit=explicit)
    if sample is not None:
        bundle = sample["bundle"]
        definitions = {
            item["metric_definition_id"]: item for item in bundle["metric_definitions"]
        }
        observations = bundle["metric_observations"]
        metric_def_refs = [
            build_source_ref(
                kind="metric_definition",
                ref_id=item["metric_definition_id"],
                content_hash=item["content_hash"],
            )
            for item in bundle["metric_definitions"]
        ]
        metric_obs_refs = [
            build_source_ref(
                kind="metric_observation",
                ref_id=item["observation_id"],
                content_hash=item["content_hash"],
            )
            for item in observations
        ]
        segment_obs_refs = [
            ref
            for ref, item in zip(metric_obs_refs, observations, strict=True)
            if item["dimension"]["kind"] == "split"
        ]
        overall = [
            item
            for item in observations
            if item["dimension"] == {"kind": "overall", "value": "overall"}
        ]
        volume_field = _sample_metric_field(
            overall, definitions, families={"volume"}, as_of=request["as_of"]
        )
        risk_field = _sample_metric_field(
            overall, definitions, families={"risk"}, as_of=request["as_of"]
        )
        maturity = bundle["sample_design"]["maturity"]
        maturity_availability = (
            "not_matured" if maturity == "not_matured" else "present"
        )
        maturity_field = build_report_field(
            value=None if maturity_availability != "present" else maturity,
            availability=maturity_availability,
            origin="tool_output",
            source_refs=[
                build_source_ref(
                    kind="sample_design",
                    ref_id=bundle["sample_design"]["sample_design_id"],
                    content_hash=bundle["sample_design"]["content_hash"],
                )
            ],
            as_of=request["as_of"],
            blocking="validation",
        )
    if latest_backtest is not None:
        if volume_field["availability"] != "present":
            volume_field = build_report_field(
                value={
                    "population_count": latest_backtest["result"]["population_count"],
                    "labeled_count": latest_backtest["result"]["labeled_count"],
                    "label_coverage": latest_backtest["result"]["label_coverage"],
                },
                availability="present",
                origin="tool_output",
                source_refs=[latest_backtest["ref"]],
                as_of=request["as_of"],
            )
        risk_metrics = {
            key: value
            for key, value in latest_backtest["result"]["metrics"].items()
            if any(token in key for token in ("bad", "loss", "risk"))
        }
        if risk_field["availability"] != "present" and risk_metrics:
            risk_field = build_report_field(
                value=risk_metrics,
                availability="present",
                origin="tool_output",
                source_refs=[latest_backtest["ref"]],
                as_of=request["as_of"],
                blocking="strategy",
            )
    latest_monitoring_run = monitoring["runs"][-1] if monitoring["runs"] else None
    if risk_field["availability"] != "present" and latest_monitoring_run is not None:
        risk_field = build_report_field(
            value={
                "overall_level": latest_monitoring_run["overall_level"],
                "checks": latest_monitoring_run["result"]["checks"],
            },
            availability="present",
            origin="tool_output",
            source_refs=[latest_monitoring_run["ref"]],
            as_of=request["as_of"],
            blocking="strategy",
        )

    approval_field = _absent_field("current.status_fields.approval", explicit=explicit)
    if pool is not None:
        assessment = pool["assessment"]
        ref = build_source_ref(
            kind="pool_impact",
            ref_id=assessment["assessment_id"],
            content_hash=assessment["content_hash"],
        )
        approval_field = build_report_field(
            value={
                "population": assessment["population"],
                "overall": assessment["overall"],
                "comparison_mode": assessment["bindings"]["comparison_mode"],
            },
            availability="present",
            origin="tool_output",
            source_refs=[ref],
            as_of=request["as_of"],
            blocking="impact",
        )
    elif latest_approval_backtest is not None:
        metrics = latest_approval_backtest["result"]["metrics"]
        approval_field = build_report_field(
            value={
                "strategy_type": latest_approval_backtest["result"]["strategy_type"],
                "population_count": latest_approval_backtest["result"][
                    "population_count"
                ],
                "approve_count": metrics["approve_count"],
                "approve_rate": metrics["approve_rate"],
                "reject_count": metrics["reject_count"],
                "reject_rate": metrics["reject_rate"],
                "review_count": metrics["review_count"],
                "review_rate": metrics["review_rate"],
            },
            availability="present",
            origin="tool_output",
            source_refs=[latest_approval_backtest["ref"]],
            as_of=request["as_of"],
            blocking="impact",
        )
    economics_field = _absent_field(
        "current.status_fields.economics", explicit=explicit
    )
    if latest_backtest is not None and latest_backtest["result"]["economics"]:
        economics_field = build_report_field(
            value=dict(latest_backtest["result"]["economics"]),
            availability="present",
            origin="tool_output",
            source_refs=[latest_backtest["ref"]],
            as_of=request["as_of"],
            blocking="impact",
        )

    if request["scope"] is not None:
        scope_field = build_report_field(
            value=request["scope"],
            availability="present",
            origin="user",
            source_refs=[message_ref],
            as_of=request["as_of"],
        )
    elif current is not None:
        # Project-context requests are patches, not full replacements.  Keep
        # the exact prior evidence pointer and original as-of when this turn
        # does not mention scope.
        scope_field = current["state"]["current_project_snapshot"]["scope"]
    else:
        scope_field = build_report_field(
            value=None,
            availability="unavailable",
            origin="user",
            source_refs=[message_ref],
            as_of=request["as_of"],
        )
    context_by_path = {}
    if current is not None:
        context_by_path = {
            item["field_path"]: item
            for item in current["state"]["current_project_snapshot"][
                "user_context_fields"
            ]
        }
    for field_path, value in request["business_context"].items():
        availability = (
            "unavailable" if value is None or field_path in explicit else "present"
        )
        context_by_path[field_path] = build_context_field(
            field_path=field_path,
            field=build_report_field(
                value=value if availability == "present" else None,
                availability=availability,
                origin="user",
                source_refs=[message_ref],
                as_of=request["as_of"],
                note=(
                    "user-provided/unverified; not deterministic metric evidence"
                    if availability == "present"
                    else "user reported unavailable; no zero imputation"
                ),
            ),
        )
    for field_path in explicit:
        if field_path not in request["business_context"]:
            context_by_path[field_path] = build_context_field(
                field_path=field_path,
                field=build_report_field(
                    value=None,
                    availability="unavailable",
                    origin="user",
                    source_refs=[message_ref],
                    as_of=request["as_of"],
                    note="user reported unavailable; no zero imputation",
                ),
            )
    context_fields = [context_by_path[key] for key in sorted(context_by_path)]
    user_resolution_fields = {
        field_path: item["field"]
        for field_path, item in context_by_path.items()
        if field_path in _KNOWN_MISSING
    }
    status_field_values = {
        "current.status_fields.volume": volume_field,
        "current.status_fields.approval": approval_field,
        "current.status_fields.risk": risk_field,
        "current.status_fields.economics": economics_field,
    }
    governed_available_paths = {
        field_path
        for field_path, field in status_field_values.items()
        if field["availability"] in {"present", "not_applicable", "not_matured"}
    }
    for field_path, field in list(status_field_values.items()):
        user_field = user_resolution_fields.get(field_path)
        if (
            field["availability"] == "unavailable"
            and user_field is not None
            and user_field["availability"] == "present"
        ):
            missing_blocking = _KNOWN_MISSING[field_path][1]
            status_field_values[field_path] = build_report_field(
                value=user_field["value"],
                availability="present",
                origin="user",
                source_refs=user_field["source_refs"],
                as_of=user_field["as_of"],
                blocking=(
                    "none"
                    if missing_blocking == "report_optional"
                    else missing_blocking
                ),
                note="user-provided/unverified; not deterministic metric evidence",
            )

    flags = []
    if len(champion_candidates) > 1:
        flags.append(
            build_red_flag(
                code="multiple_local_champions",
                level="red",
                message="任务存在多个本地 adopted strategy；当前快照仅展示最新一条。",
                source_refs=[item["ref"] for item in champion_candidates],
            )
        )
    snapshot = build_current_project_snapshot(
        task_id=task_id,
        as_of=request["as_of"],
        scope=scope_field,
        dataset_refs=[item["ref"] for item in datasets],
        workspace_ref=None
        if evidence["workspace"] is None
        or (
            evidence["workspace"]["active_dataset_id"] is not None
            and evidence["workspace"]["active_dataset_id"]
            not in {item["id"] for item in datasets}
        )
        else evidence["workspace"]["ref"],
        champion_strategy_ref=None if champion is None else champion["ref"],
        status_fields={
            "volume": status_field_values["current.status_fields.volume"],
            "approval": status_field_values["current.status_fields.approval"],
            "risk": status_field_values["current.status_fields.risk"],
            "economics": status_field_values["current.status_fields.economics"],
        },
        metric_definition_refs=metric_def_refs,
        metric_observation_refs=metric_obs_refs,
        monthly_observation_refs=[],
        segment_observation_refs=segment_obs_refs,
        maturity_summary=maturity_field,
        user_context_fields=context_fields,
        red_flags=flags,
        tool_run_refs=[],
    )

    histories = [
        _strategy_history(
            task_id, item, evidence=state_evidence, as_of=request["as_of"]
        )
        for item in strategies
    ]
    external_refs = _external_refs(evidence, external)
    if external_refs:
        histories.append(
            build_historical_strategy_review(
                task_id=task_id,
                strategy_ref=None,
                version=None,
                effective_period=_unavailable_report_field(origin="uploaded_file"),
                asset_status=_unavailable_report_field(origin="uploaded_file"),
                scope=_unavailable_report_field(origin="uploaded_file"),
                traffic_allocation=_unavailable_report_field(origin="uploaded_file"),
                change_set=diff_strategy_rules([], []),
                observation_refs_by_effect_stage={
                    stage: []
                    for stage in (
                        "estimated",
                        "backtested",
                        "oot_validated",
                        "post_launch_observed",
                    )
                },
                external_source_refs=external_refs,
                decision_context_fields=[],
                availability="present",
                red_flags=[
                    build_red_flag(
                        code="external_report_opaque",
                        level="info",
                        message="外部历史材料仅作为不透明原始证据保留，未从中抽取或推断指标。",
                        source_refs=external_refs,
                    )
                ],
                tool_run_refs=[],
            )
        )

    if snapshot["maturity_summary"]["availability"] in {"present", "not_matured"}:
        governed_available_paths.add("current.maturity_summary")
    if histories:
        governed_available_paths.add("historical_strategy_reviews")
    missing_records = _missing_records(
        task_id=task_id,
        request=request,
        message=evidence["message"],
        message_ref=message_ref,
        governed_available_paths=governed_available_paths,
        user_resolution_fields=user_resolution_fields,
        current=current,
        evidence=state_evidence,
        external_refs=external_refs,
    )
    extra_sources = [message_ref]
    for category in ("sample_designs", "pool_impacts", "external_reports"):
        extra_sources.extend(
            item["artifact_ref"] for item in state_evidence["artifacts"][category]
        )
    for snapshot_source in external:
        external_path = _external_artifact_path(
            tasks_root,
            task_id=task_id,
            snapshot=snapshot_source,
        )
        extra_sources.append(
            build_source_ref(
                kind="task_artifact",
                ref_id=_stable_task_artifact_id(
                    task_id=task_id,
                    kind=PROJECT_CONTEXT_EXTERNAL_ARTIFACT_KIND,
                    path=str(external_path),
                ),
                content_hash=snapshot_source.content_hash,
            )
        )
    extra_sources.extend(
        item["parent"]["ref"] for item in strategies if item["parent"] is not None
    )
    return build_strategy_project_context_state(
        task_id=task_id,
        as_of=request["as_of"],
        current_project_snapshot=snapshot,
        historical_strategy_reviews=histories,
        missing_information_records=missing_records,
        source_refs=extra_sources,
        red_flags=flags,
    )


def _strategy_history(
    task_id: str,
    strategy: Mapping[str, Any],
    *,
    evidence: Mapping[str, Any],
    as_of: str,
) -> dict[str, Any]:
    parent_rules = [] if strategy["parent"] is None else strategy["parent"]["rules"]
    adoption_visible = strategy["adopted_at"] is None or _timestamp_on_or_before_as_of(
        strategy["adopted_at"], as_of=as_of, field="strategy.adopted_at"
    )
    backtested = [
        build_effect_observation_ref(observation_ref=item["ref"])
        for item in strategy["backtests"]
    ]
    plans = [
        item["ref"]
        for item in evidence["monitoring"]["plans"]
        if item["strategy_id"] == strategy["id"]
    ]
    runs = [
        item["ref"]
        for item in evidence["monitoring"]["runs"]
        if item["strategy_id"] == strategy["id"]
    ]
    # A local monitoring run proves a governed observation exists, not that a
    # production deployment/environment/effective period exists.  Therefore it
    # stays a tool ref and is never mislabeled post_launch_observed.
    return build_historical_strategy_review(
        task_id=task_id,
        strategy_ref=strategy["ref"],
        version=strategy["version"],
        effective_period=build_report_field(
            value={"start": str(strategy["created_at"])[:10], "end": None},
            availability="present",
            origin="repository",
            source_refs=[strategy["ref"]],
            as_of=as_of,
        ),
        asset_status=(
            build_report_field(
                value=strategy["asset_status"],
                availability="present",
                origin="repository",
                source_refs=[strategy["ref"]],
                as_of=as_of,
            )
            if adoption_visible
            else _unavailable_report_field(origin="repository")
        ),
        scope=build_report_field(
            value={
                "strategy_type": strategy["strategy_type"],
                "description": strategy["description"],
            },
            availability="present",
            origin="repository",
            source_refs=[strategy["ref"]],
            as_of=as_of,
        ),
        traffic_allocation=_unavailable_report_field(origin="repository"),
        change_set=diff_strategy_rules(parent_rules, strategy["rules"]),
        observation_refs_by_effect_stage={
            "estimated": [],
            "backtested": backtested,
            "oot_validated": [],
            "post_launch_observed": [],
        },
        external_source_refs=[],
        decision_context_fields=(
            []
            if strategy["adoption_reason"] is None or not adoption_visible
            else [
                build_context_field(
                    field_path="adoption_reason",
                    field=build_report_field(
                        value=strategy["adoption_reason"],
                        availability="present",
                        origin="repository",
                        source_refs=[strategy["ref"]],
                        as_of=as_of,
                    ),
                )
            ]
        ),
        availability="present",
        red_flags=[],
        tool_run_refs=[*(item["ref"] for item in strategy["backtests"]), *plans, *runs],
    )


def _sample_metric_field(
    observations, definitions, *, families: set[str], as_of: str
) -> dict[str, Any]:
    selected = []
    refs = []
    statuses = []
    for observation in observations:
        definition = definitions[
            observation["metric_definition_ref"]["metric_definition_id"]
        ]
        if definition["metric_family"] not in families:
            continue
        statuses.append(observation["status"])
        refs.append(
            build_source_ref(
                kind="metric_observation",
                ref_id=observation["observation_id"],
                content_hash=observation["content_hash"],
            )
        )
        if observation["status"] == "present":
            selected.append(
                {
                    "metric_key": definition["metric_key"],
                    "value": observation["value"],
                    "unit": definition["unit"],
                }
            )
    if selected:
        return build_report_field(
            value=selected,
            availability="present",
            origin="tool_output",
            source_refs=refs,
            as_of=as_of,
        )
    availability = "not_matured" if "not_matured" in statuses else "unavailable"
    return build_report_field(
        value=None,
        availability=availability,
        origin="tool_output",
        source_refs=refs,
        as_of=as_of,
    )


def _missing_records(
    *,
    task_id: str,
    request: Mapping[str, Any],
    message: Mapping[str, Any],
    message_ref: Mapping[str, Any],
    governed_available_paths: set[str],
    user_resolution_fields: Mapping[str, Mapping[str, Any]],
    current: Mapping[str, Any] | None,
    evidence: Mapping[str, Any],
    external_refs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    previous = {}
    if current is not None:
        previous = {
            (item["field_path"], item["dependency_hash"]): item
            for item in current["state"]["missing_information_records"]
        }
    records = []
    dependency_sources = [
        *(item["ref"] for item in evidence["datasets"]),
        *(item["ref"] for item in evidence["strategies"]),
        *external_refs,
    ]
    for field_path, (reason, blocking, question) in sorted(_KNOWN_MISSING.items()):
        if field_path in governed_available_paths:
            continue
        dependency_hash = _sha256_json(
            {
                "task_id": task_id,
                "field_path": field_path,
                "sources": dependency_sources,
            }
        )
        prior = previous.get((field_path, dependency_hash))
        resolution = user_resolution_fields.get(field_path)
        if resolution is not None and resolution["availability"] == "present":
            status = "provided"
        elif resolution is not None and resolution["availability"] == "unavailable":
            status = "unavailable"
        else:
            status = "pending"
        if status == "pending":
            if prior is not None and prior["asked_count"] > 0:
                asked_count = prior["asked_count"]
                asked_at = prior["asked_at"]
            else:
                asked_count = 1
                asked_at = str(message["created_at"])
            answered_at = None
            answer_ref = None
        else:
            asked_count = 0 if prior is None else prior["asked_count"]
            asked_at = None if prior is None else prior["asked_at"]
            assert resolution is not None
            answer_ref = resolution["source_refs"][0]
            if (
                prior is not None
                and prior["status"] == status
                and prior["answer_source_ref"] == answer_ref
            ):
                answered_at = prior["answered_at"]
            else:
                answered_at = str(message["created_at"])
        records.append(
            build_missing_information_record(
                task_id=task_id,
                field_path=field_path,
                reason=reason,
                blocking=blocking,
                question=question,
                status=status,
                asked_count=asked_count,
                asked_at=asked_at,
                answered_at=answered_at,
                answer_source_ref=answer_ref,
                dependency_hash=dependency_hash,
            )
        )
    return records


def _absent_field(field_path: str, *, explicit: set[str]) -> dict[str, Any]:
    missing_blocking = _KNOWN_MISSING[field_path][1]
    report_blocking = (
        "none" if missing_blocking == "report_optional" else missing_blocking
    )
    return build_report_field(
        value=None,
        availability="unavailable",
        origin="repository",
        source_refs=[],
        blocking=report_blocking,
    )


def _unavailable_report_field(*, origin: str) -> dict[str, Any]:
    return build_report_field(
        value=None, availability="unavailable", origin=origin, source_refs=[]
    )


def _external_refs(
    evidence: Mapping[str, Any], snapshots: Sequence[_ExternalSnapshot]
) -> list[dict[str, str]]:
    refs = [
        build_source_ref(
            kind="external_report",
            ref_id=item["content_hash"],
            content_hash=item["content_hash"],
        )
        for item in evidence["artifacts"]["external_reports"]
    ]
    refs.extend(
        build_source_ref(
            kind="external_report",
            ref_id=item.content_hash,
            content_hash=item.content_hash,
        )
        for item in snapshots
    )
    return _dedupe_refs(refs)


def _classify_write(
    *,
    current: Mapping[str, Any] | None,
    candidate: Mapping[str, Any],
    request: Mapping[str, Any],
) -> str:
    if current is not None and current["revision_id"] == candidate["revision_id"]:
        return "replay"
    actual = (
        (0, None, None)
        if current is None
        else (current["revision"], current["revision_id"], current["state_hash"])
    )
    expected = (
        request["expected_revision"],
        request["expected_revision_id"],
        request["expected_state_hash"],
    )
    if actual != expected:
        raise StrategyProjectContextConflictError(
            "project context head changed; refresh and retry"
        )
    if current is not None and current["state_hash"] == candidate["state_hash"]:
        return "unchanged"
    return "new"


def _stage_external_sources(*, uow, tasks_root, task_id, source_dir, snapshots) -> int:
    staged_count = 0
    for snapshot in snapshots:
        path = _external_artifact_path(tasks_root, task_id=task_id, snapshot=snapshot)
        staged = _stage_external_file(
            uow=uow,
            tasks_root=tasks_root,
            source_root=Path(source_dir).absolute(),
            final_path=path,
            snapshot=snapshot,
        )
        staged_count += int(staged is not None)
    return staged_count


def _register_external_sources(
    conn, *, task_artifacts, tasks_root, task_id, snapshots
) -> list[dict[str, Any]]:
    records = []
    for snapshot in snapshots:
        path = _external_artifact_path(tasks_root, task_id=task_id, snapshot=snapshot)
        _verify_regular_file(
            path,
            root=tasks_root,
            expected_hash=snapshot.content_hash,
            max_bytes=MAX_EXTERNAL_REPORT_BYTES,
        )
        provenance = {
            "schema_version": PROJECT_CONTEXT_EXTERNAL_SCHEMA_VERSION,
            "task_id": task_id,
            "content_hash": snapshot.content_hash,
            "content_size": snapshot.content_size,
            "suffix": snapshot.suffix,
        }
        records.append(
            task_artifacts.register_on_connection(
                conn,
                task_id=task_id,
                kind=PROJECT_CONTEXT_EXTERNAL_ARTIFACT_KIND,
                path=str(path),
                content_hash=snapshot.content_hash,
                origin_tool=PROJECT_CONTEXT_ORIGIN_TOOL,
                provenance=provenance,
            )
        )
    return records


def _stage_external_file(
    *,
    uow,
    tasks_root: Path,
    source_root: Path,
    final_path: Path,
    snapshot: _ExternalSnapshot,
):
    _prepare_output_directory(final_path.parent, tasks_root=tasks_root)
    if final_path.exists() or final_path.is_symlink():
        _verify_regular_file(
            final_path,
            root=tasks_root,
            expected_hash=snapshot.content_hash,
            max_bytes=MAX_EXTERNAL_REPORT_BYTES,
        )
        return None
    source_path = source_root.joinpath(*PurePath(snapshot.relative_name).parts)
    _require_no_symlink_path(source_path, root=source_root)
    staged = uow.stage_file(final_path.parent, final_path.name)
    with staged.path.open("wb") as destination:
        size, digest = _stream_regular_nofollow(
            source_path,
            max_bytes=MAX_EXTERNAL_REPORT_BYTES,
            consume=destination.write,
        )
    if size != snapshot.content_size or not hmac.compare_digest(
        digest, snapshot.content_hash
    ):
        raise StrategyError("external strategy report changed while being staged")
    return staged


def _stage_new_file(
    uow,
    *,
    root: Path,
    final_path: Path,
    data: bytes,
    expected_hash: str,
    tasks_root: Path,
):
    _prepare_output_directory(root, tasks_root=tasks_root)
    if final_path.exists() or final_path.is_symlink():
        _verify_regular_file(
            final_path,
            root=tasks_root,
            expected_hash=expected_hash,
            expected_bytes=data,
            max_bytes=max(MAX_EXTERNAL_REPORT_BYTES, MAX_PROJECT_CONTEXT_JSON_BYTES),
        )
        return None
    staged = uow.stage_file(root, final_path.name)
    staged.path.write_bytes(data)
    if sha256_file(staged.path) != expected_hash:
        raise StrategyError("staged strategy project context artifact hash changed")
    return staged


def _context_artifact_path(tasks_root: Path, *, task_id: str, revision_id: str) -> Path:
    return (
        Path(tasks_root).absolute()
        / task_id
        / "strategy_project_contexts"
        / f"{revision_id}.json"
    )


def _external_artifact_path(
    tasks_root: Path, *, task_id: str, snapshot: _ExternalSnapshot
) -> Path:
    return (
        Path(tasks_root).absolute()
        / task_id
        / "strategy_project_context_sources"
        / f"{snapshot.content_hash}{snapshot.suffix}"
    )


def _prepare_output_directory(path: Path, *, tasks_root: Path) -> None:
    root = Path(tasks_root).absolute()
    if root.is_symlink():
        raise StrategyError("task artifact root must not be a symlink")
    path.mkdir(parents=True, exist_ok=True)
    _require_no_symlink_path(path, root=root)
    if not path.is_dir():
        raise StrategyError("strategy project context output path is not a directory")


def _snapshot_external_reports(
    source_dir: str, filenames: Sequence[str]
) -> list[_ExternalSnapshot]:
    source_root = Path(_text(source_dir, "task.source_dir")).absolute()
    if not source_root.exists() or not source_root.is_dir() or source_root.is_symlink():
        if filenames:
            raise StrategyError("task source_dir is unavailable for external reports")
        return []
    snapshots = []
    total_bytes = 0
    for filename in filenames:
        relative, suffix = _external_relative_path(filename)
        path = source_root.joinpath(*relative.parts)
        _require_no_symlink_path(path, root=source_root)
        content_size, digest = _stream_regular_nofollow(
            path, max_bytes=MAX_EXTERNAL_REPORT_BYTES
        )
        total_bytes += content_size
        if total_bytes > MAX_EXTERNAL_REPORT_TOTAL_BYTES:
            raise StrategyError("external reports exceed total byte limit")
        snapshots.append(
            _ExternalSnapshot(
                relative_name=relative.as_posix(),
                suffix=suffix,
                content_hash=digest,
                content_size=content_size,
            )
        )
    return snapshots


def _external_relative_path(filename: str) -> tuple[PurePath, str]:
    relative = PurePath(filename)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise StrategyError("external report filename must be a safe relative path")
    suffix = Path(filename).suffix.lower()
    if suffix == ".xls":
        raise StrategyError(
            "legacy .xls external reports are unsupported; save as .xlsx"
        )
    if suffix not in _EXTERNAL_SUFFIXES:
        raise StrategyError("external report suffix is unsupported")
    return relative, suffix


def _validate_external_report_message_bindings(
    *, source_dir: str, filenames: Sequence[str], persisted_message: str
) -> None:
    if not filenames:
        return
    source_root = Path(_text(source_dir, "task.source_dir")).absolute()
    basename_only: set[str] = set()
    for filename in filenames:
        relative, _ = _external_relative_path(filename)
        relative_name = relative.as_posix()
        if relative_name in persisted_message:
            continue
        basename = relative.name
        if basename not in persisted_message:
            raise StrategyError(
                "external report filename is not grounded in the bound user message"
            )
        basename_only.add(basename)
    if not basename_only:
        return
    if not source_root.exists() or not source_root.is_dir() or source_root.is_symlink():
        raise StrategyError("task source_dir is unavailable for external reports")
    counts = dict.fromkeys(basename_only, 0)
    entry_count = 0
    for directory, child_dirs, child_files in os.walk(source_root, followlinks=False):
        directory_path = Path(directory)
        child_dirs[:] = [
            name for name in child_dirs if not (directory_path / name).is_symlink()
        ]
        entry_count += len(child_dirs) + len(child_files)
        if entry_count > MAX_EXTERNAL_SOURCE_ENTRIES:
            raise StrategyError(
                "task source_dir is too large to prove external report basename uniqueness"
            )
        for basename in counts:
            counts[basename] += child_files.count(basename)
    if any(count != 1 for count in counts.values()):
        raise StrategyError(
            "external report basename must identify exactly one source file"
        )


def _stream_regular_nofollow(
    path: Path,
    *,
    max_bytes: int,
    consume=None,
) -> tuple[int, str]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise StrategyError(f"external report is unavailable: {path.name}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise StrategyError("external report must be a regular file")
        if before.st_size > max_bytes:
            raise StrategyError("external report exceeds byte limit")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise StrategyError("external report exceeds byte limit")
            digest.update(chunk)
            if consume is not None:
                consume(chunk)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise StrategyError("external report changed while being read")
        return total, digest.hexdigest()
    finally:
        os.close(fd)


def _read_regular_nofollow(path: Path, *, max_bytes: int) -> tuple[bytes, str]:
    chunks: list[bytes] = []
    _, digest = _stream_regular_nofollow(
        path,
        max_bytes=max_bytes,
        consume=chunks.append,
    )
    return b"".join(chunks), digest


def _artifact_row(row, *, task_id: str, tasks_root: Path) -> dict[str, Any]:
    if str(row["task_id"]) != task_id:
        raise StrategyError("task artifact belongs to another task")
    try:
        provenance = json.loads(str(row["provenance_json"]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StrategyError("task artifact provenance is invalid") from exc
    if not isinstance(provenance, dict) or _canonical_json(provenance) != str(
        row["provenance_json"]
    ):
        raise StrategyError("task artifact provenance is not canonical")
    path = Path(_text(row["path"], "task_artifact.path"))
    _require_no_symlink_path(path, root=Path(tasks_root).absolute())
    content_hash = _hash(row["content_hash"], "task_artifact.content_hash")
    expected_artifact_id = _stable_task_artifact_id(
        task_id=task_id,
        kind=str(row["kind"]),
        path=str(path),
    )
    if str(row["id"]) != expected_artifact_id:
        raise StrategyError("task artifact stable identity changed")
    return {
        "id": str(row["id"]),
        "task_id": task_id,
        "kind": str(row["kind"]),
        "path": str(path),
        "content_hash": content_hash,
        "origin_tool": str(row["origin_tool"]),
        "provenance": provenance,
        "created_at": str(row["created_at"]),
        "artifact_ref": build_source_ref(
            kind="task_artifact", ref_id=str(row["id"]), content_hash=content_hash
        ),
    }


def _read_verified_registered_artifact(
    record: Mapping[str, Any], *, tasks_root: Path
) -> bytes:
    data, digest = _read_regular_nofollow(
        Path(record["path"]), max_bytes=MAX_PROJECT_CONTEXT_JSON_BYTES
    )
    if not hmac.compare_digest(digest, record["content_hash"]):
        raise StrategyError("task artifact bytes do not match its registry hash")
    return data


def _verify_regular_file(
    path: Path,
    *,
    root: Path,
    expected_hash: str,
    expected_bytes: bytes | None = None,
    max_bytes: int = MAX_EXTERNAL_REPORT_BYTES,
) -> None:
    _require_no_symlink_path(path, root=Path(root).absolute())
    if expected_bytes is None:
        _, digest = _stream_regular_nofollow(path, max_bytes=max_bytes)
        data = None
    else:
        data, digest = _read_regular_nofollow(path, max_bytes=max_bytes)
    if not hmac.compare_digest(digest, expected_hash):
        raise StrategyError("artifact bytes do not match the expected content hash")
    if expected_bytes is not None and data != expected_bytes:
        raise StrategyError("artifact bytes do not match canonical evidence")


def _require_no_symlink_path(path: Path, *, root: Path) -> None:
    root = root.absolute()
    candidate = path.absolute()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise StrategyError("artifact path escaped its governed root") from exc
    current = root
    if current.exists() and current.is_symlink():
        raise StrategyError("governed artifact root must not be a symlink")
    for part in candidate.relative_to(root).parts:
        current = current / part
        if current.exists() or current.is_symlink():
            if current.is_symlink():
                raise StrategyError("governed artifact path must not contain symlinks")
    try:
        candidate.resolve(strict=False).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise StrategyError("artifact path escaped its governed root") from exc


def _load_revision_artifact_on_connection(
    conn, *, tasks_root: Path, task_id: str, revision: Mapping[str, Any]
) -> dict[str, Any]:
    record = _context_artifact_record_on_connection(
        conn,
        tasks_root=tasks_root,
        task_id=task_id,
        revision_id=revision["revision_id"],
    )
    expected_path = _context_artifact_path(
        tasks_root, task_id=task_id, revision_id=revision["revision_id"]
    )
    if (
        Path(record["path"]) != expected_path
        or record["origin_tool"] != PROJECT_CONTEXT_ORIGIN_TOOL
    ):
        raise StrategyError(
            "strategy project context artifact registry binding changed"
        )
    raw = _read_verified_registered_artifact(record, tasks_root=tasks_root)
    loaded = strategy_project_context_revision_from_json(raw)
    canonical = canonical_strategy_project_context_revision_json(loaded).encode("utf-8")
    if raw != canonical or loaded != revision:
        raise StrategyError(
            "strategy project context artifact does not match its revision"
        )
    provenance = record["provenance"]
    if set(provenance) != {
        "schema_version",
        "task_id",
        "revision_id",
        "revision",
        "revision_content_hash",
        "state_hash",
        "operation_hash",
        "format",
    }:
        raise StrategyError("strategy project context artifact provenance is invalid")
    if provenance != {
        "schema_version": PROJECT_CONTEXT_ARTIFACT_SCHEMA_VERSION,
        "task_id": task_id,
        "revision_id": loaded["revision_id"],
        "revision": loaded["revision"],
        "revision_content_hash": loaded["content_hash"],
        "state_hash": loaded["state_hash"],
        "operation_hash": loaded["operation_hash"],
        "format": "json",
    }:
        raise StrategyError("strategy project context artifact provenance changed")
    return loaded


def _context_artifact_record_on_connection(
    conn,
    *,
    tasks_root: Path,
    task_id: str,
    revision_id: str,
) -> dict[str, Any]:
    expected_path = _context_artifact_path(
        tasks_root,
        task_id=task_id,
        revision_id=revision_id,
    )
    row = conn.execute(
        "SELECT * FROM task_artifacts WHERE task_id = ? AND kind = ? AND path = ?",
        (task_id, PROJECT_CONTEXT_ARTIFACT_KIND, str(expected_path)),
    ).fetchone()
    if row is None:
        raise StrategyError("strategy project context artifact is not registered")
    return _artifact_row(row, task_id=task_id, tasks_root=tasks_root)


def _external_records_on_connection(
    conn,
    *,
    tasks_root: Path,
    task_id: str,
    snapshots: Sequence[_ExternalSnapshot],
) -> list[dict[str, Any]]:
    records = []
    for snapshot in snapshots:
        expected_path = _external_artifact_path(
            tasks_root,
            task_id=task_id,
            snapshot=snapshot,
        )
        row = conn.execute(
            "SELECT * FROM task_artifacts WHERE task_id = ? AND kind = ? AND path = ?",
            (task_id, PROJECT_CONTEXT_EXTERNAL_ARTIFACT_KIND, str(expected_path)),
        ).fetchone()
        if row is None:
            raise StrategyError("external report artifact is not registered for replay")
        records.append(_artifact_row(row, task_id=task_id, tasks_root=tasks_root))
    return records


def _verify_live_refs(
    conn,
    *,
    runtime,
    tasks_root: Path,
    task_id: str,
    source_refs: Sequence[Mapping[str, Any]],
) -> None:
    for ref in source_refs:
        kind, ref_id, expected = ref["kind"], ref["ref_id"], ref["content_hash"]
        actual = None
        if kind == "agent_message":
            row = conn.execute(
                "SELECT task_id, content FROM agent_messages WHERE id = ?", (ref_id,)
            ).fetchone()
            if row is not None and str(row["task_id"]) == task_id:
                actual = hashlib.sha256(str(row["content"]).encode("utf-8")).hexdigest()
        elif kind == "dataset":
            row = conn.execute(
                "SELECT task_id, source_path, content_hash FROM datasets WHERE id = ?",
                (ref_id,),
            ).fetchone()
            if row is not None and str(row["task_id"]) == task_id:
                actual = _hash(row["content_hash"], "dataset.content_hash")
                _verify_regular_file(
                    Path(runtime.settings.datasets_dir) / str(row["source_path"]),
                    root=Path(runtime.settings.datasets_dir),
                    expected_hash=actual,
                )
        elif kind == "workspace":
            row = conn.execute(
                "SELECT * FROM data_workspaces WHERE task_id = ?", (task_id,)
            ).fetchone()
            if row is not None:
                datasets, by_id = _discover_datasets(
                    conn,
                    task_id=task_id,
                    datasets_root=Path(runtime.settings.datasets_dir),
                )
                workspace = _discover_workspace(
                    conn, task_id=task_id, dataset_by_id=by_id
                )
                if workspace is not None and workspace["ref"]["ref_id"] == ref_id:
                    actual = workspace["content_hash"]
        elif kind == "strategy":
            row = conn.execute(
                "SELECT * FROM strategies WHERE id = ?", (ref_id,)
            ).fetchone()
            # A task-owned strategy version may legitimately point to a parent
            # from a governed red-monitoring handoff task.  The materializer
            # alone discovers that lineage; current loading verifies the exact
            # parent bytes/hash without incorrectly requiring same-task ownership.
            if row is not None:
                actual = _strategy_evidence(row)["content_hash"]
        elif kind == "backtest":
            row = conn.execute(
                "SELECT result_json FROM backtests WHERE id = ?", (ref_id,)
            ).fetchone()
            if row is not None:
                payload = StrategyBacktestResult.from_dict(
                    json.loads(str(row["result_json"]))
                ).to_dict()
                actual = _sha256_json(payload)
        elif kind == "monitoring_plan":
            record = StrategyMonitoringRepository(runtime.settings.db_path).get_plan(
                ref_id
            )
            actual = None if record is None else record.payload_hash
        elif kind == "monitoring_run":
            record = StrategyMonitoringRepository(runtime.settings.db_path).get_run(
                ref_id
            )
            actual = None if record is None else record.result_hash
        elif kind in {
            "task_artifact",
            "external_report",
            "sample_design",
            "pool_impact",
            "metric_definition",
            "metric_observation",
        }:
            actual = _resolve_artifact_derived_ref(
                conn,
                tasks_root=tasks_root,
                task_id=task_id,
                kind=kind,
                ref_id=ref_id,
                expected=expected,
            )
        else:
            raise StrategyError(f"unsupported live project-context source kind: {kind}")
        if actual is None or not hmac.compare_digest(actual, expected):
            raise StrategyError(
                f"current strategy project context source changed: {kind}:{ref_id}"
            )


def _resolve_artifact_derived_ref(
    conn, *, tasks_root: Path, task_id: str, kind: str, ref_id: str, expected: str
) -> str | None:
    rows = conn.execute(
        "SELECT * FROM task_artifacts WHERE task_id = ? AND kind IN (?, ?, ?)",
        (
            task_id,
            SAMPLE_DESIGN_ARTIFACT_KIND,
            POOL_IMPACT_ARTIFACT_KIND,
            PROJECT_CONTEXT_EXTERNAL_ARTIFACT_KIND,
        ),
    ).fetchall()
    for row in rows:
        record = _artifact_row(row, task_id=task_id, tasks_root=tasks_root)
        raw: bytes | None = None
        if record["kind"] == PROJECT_CONTEXT_EXTERNAL_ARTIFACT_KIND:
            _verify_regular_file(
                Path(record["path"]),
                root=tasks_root,
                expected_hash=record["content_hash"],
                max_bytes=MAX_EXTERNAL_REPORT_BYTES,
            )
        else:
            raw = _read_verified_registered_artifact(record, tasks_root=tasks_root)
        if kind == "task_artifact" and record["id"] == ref_id:
            return record["content_hash"]
        if (
            record["kind"] == PROJECT_CONTEXT_EXTERNAL_ARTIFACT_KIND
            and kind == "external_report"
            and record["content_hash"] == ref_id
        ):
            return record["content_hash"]
        if record["kind"] == SAMPLE_DESIGN_ARTIFACT_KIND:
            assert raw is not None
            bundle = strategy_sample_design_bundle_from_json(raw)
            design = bundle["sample_design"]
            if kind == "sample_design" and design["sample_design_id"] == ref_id:
                return design["content_hash"]
            for item in bundle["metric_definitions"]:
                if (
                    kind == "metric_definition"
                    and item["metric_definition_id"] == ref_id
                ):
                    return item["content_hash"]
            for item in bundle["metric_observations"]:
                if kind == "metric_observation" and item["observation_id"] == ref_id:
                    return item["content_hash"]
        if record["kind"] == POOL_IMPACT_ARTIFACT_KIND:
            assert raw is not None
            assessment = validate_strategy_pool_impact_assessment(json.loads(raw))
            if kind == "pool_impact" and assessment["assessment_id"] == ref_id:
                return assessment["content_hash"]
    return None


def _artifact_output(
    record: Mapping[str, Any], *, task_id: str, format_name: str
) -> dict[str, str]:
    artifact_id = _text(record["id"], "artifact_id")
    return {
        "artifact_id": artifact_id,
        "kind": str(record["kind"]),
        "format": format_name,
        "filename": Path(str(record["path"])).name,
        "content_hash": _hash(record["content_hash"], "artifact.content_hash"),
        "download_url": f"/api/tasks/{quote(task_id, safe='')}/task-artifacts/{quote(artifact_id, safe='')}/download",
    }


def _validate_artifact_output(value: object, *, task_id: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _OUTPUT_ARTIFACT_FIELDS:
        raise StrategyError("materialize_project_context artifact output is invalid")
    normalized = {
        key: (
            _hash(value[key], key)
            if key in {"artifact_id", "content_hash"}
            else _text(value[key], key)
        )
        for key in sorted(_OUTPUT_ARTIFACT_FIELDS)
    }
    filename = normalized["filename"]
    if Path(filename).name != filename or PurePath(filename).is_absolute():
        raise StrategyError("materialize_project_context artifact filename is invalid")
    expected_url = (
        f"/api/tasks/{quote(task_id, safe='')}/task-artifacts/"
        f"{quote(normalized['artifact_id'], safe='')}/download"
    )
    if normalized["download_url"] != expected_url:
        raise StrategyError(
            "materialize_project_context artifact download URL is invalid"
        )
    return normalized


def _external_identity(
    value: Sequence[_ExternalSnapshot],
) -> list[tuple[str, str, int]]:
    return [
        (item.relative_name, item.content_hash, item.content_size) for item in value
    ]


def _require_external_total_with_registered(
    evidence: Mapping[str, Any], snapshots: Sequence[_ExternalSnapshot]
) -> None:
    registered = {
        (item["content_hash"], item["provenance"]["suffix"]): int(
            item["provenance"]["content_size"]
        )
        for item in evidence["artifacts"]["external_reports"]
    }
    combined = dict(registered)
    for snapshot in snapshots:
        combined[(snapshot.content_hash, snapshot.suffix)] = snapshot.content_size
    if sum(combined.values()) > MAX_EXTERNAL_REPORT_TOTAL_BYTES:
        raise StrategyError("external reports exceed total byte limit")


def _evidence_fingerprint_without_external_hashes(
    evidence: Mapping[str, Any], content_hashes: set[str]
) -> str:
    descriptor = {
        key: value
        for key, value in evidence.items()
        if key not in {"fingerprint", "message_ref"}
    }
    artifacts = dict(descriptor["artifacts"])
    artifacts["external_reports"] = [
        item
        for item in artifacts["external_reports"]
        if item["content_hash"] not in content_hashes
    ]
    descriptor["artifacts"] = artifacts
    return _sha256_json(descriptor)


def _request_hash(request: Mapping[str, Any]) -> str:
    return _sha256_json(request)


def _stable_task_artifact_id(*, task_id: str, kind: str, path: str) -> str:
    identity = json.dumps(
        [task_id, kind, path],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(
        f"marvis.task_artifact.v1:{identity}".encode("utf-8")
    ).hexdigest()


def _dedupe_refs(refs: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    result = {}
    for ref in refs:
        key = (ref["kind"], ref["ref_id"])
        prior = result.get(key)
        if prior is not None and prior["content_hash"] != ref["content_hash"]:
            raise StrategyError("source reference identity drift")
        result[key] = dict(ref)
    return [result[key] for key in sorted(result)]


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise StrategyError("project context evidence must be canonical JSON") from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise StrategyError(f"{field} must be a non-empty string")
    return value.strip()


def _bounded_text(value: object, field: str, maximum: int) -> str:
    result = _text(value, field)
    if len(result) > maximum:
        raise StrategyError(f"{field} exceeds character limit")
    return result


def _optional_text(value: object, field: str) -> str | None:
    return None if value is None else _text(value, field)


def _optional_bounded_text(value: object, field: str, maximum: int) -> str | None:
    return None if value is None else _bounded_text(value, field, maximum)


def _hash(value: object, field: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value.lower()) is None:
        raise StrategyError(f"{field} must be a SHA-256 digest")
    return value.lower()


def _optional_hash(value: object, field: str) -> str | None:
    return None if value is None else _hash(value, field)


def _non_negative_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise StrategyError(f"{field} must be a non-negative integer")
    return value


def _iso_date(value: object, field: str) -> str:
    text = _text(value, field)
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise StrategyError(f"{field} must be YYYY-MM-DD") from exc
    if parsed.isoformat() != text:
        raise StrategyError(f"{field} must be YYYY-MM-DD")
    return text


def _timestamp_on_or_before_as_of(value: object, *, as_of: str, field: str) -> bool:
    timestamp = _text(value, field)
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise StrategyError(f"{field} must be an ISO timestamp") from exc
    return parsed.date() <= date.fromisoformat(as_of)


def _bounded_text_list(
    value: object, *, field: str, maximum: int, item_maximum: int
) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise StrategyError(f"{field} must be a list")
    if len(value) > maximum:
        raise StrategyError(f"{field} has too many items")
    result = [_bounded_text(item, f"{field} item", item_maximum) for item in value]
    if len(result) != len(set(result)):
        raise StrategyError(f"{field} contains duplicate items")
    return result


__all__ = [
    "PROJECT_CONTEXT_ARTIFACT_KIND",
    "PROJECT_CONTEXT_EXTERNAL_ARTIFACT_KIND",
    "PROJECT_CONTEXT_ORIGIN_TOOL",
    "PROJECT_CONTEXT_TOOL_SCHEMA_VERSION",
    "load_current_strategy_project_context",
    "load_strategy_project_context_revision_for_audit",
    "run_materialize_project_context",
    "validate_materialize_project_context_tool_output",
]
