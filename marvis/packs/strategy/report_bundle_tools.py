"""Governed build/publish boundary for ``StrategyReportBundle`` V2.

Callers may select only immutable evidence references and report controls.  The
Tool reloads every metric-bearing source from its governed repository/artifact
boundary, projects those authenticated bindings through the pure report
adapter, renders all four deterministic formats, and publishes files,
TaskArtifact rows, the report revision, and one audit row in a shared
filesystem/database unit of work.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from marvis.artifacts import ArtifactUnitOfWork
from marvis.artifacts.transactional import ArtifactTransactionError
from marvis.output.strategy_report_bundle import render_strategy_report_bundle
from marvis.packs.modeling.errors import ModelingError
from marvis.packs.modeling.evidence_tools import (
    ModelingTrainingEvidenceArtifactBinding,
    build_training_evidence_ref,
    load_modeling_training_evidence_artifacts,
    require_modeling_training_evidence_artifact_binding_on_connection,
)
from marvis.packs.modeling.score_evidence_tools import (
    ModelScoreEvidenceArtifactBinding,
    load_model_score_evidence_artifacts,
    require_model_score_evidence_artifact_binding_on_connection,
)
from marvis.packs.strategy.candidate_stability_tools import (
    StrategyCandidateStabilityArtifactBinding,
    load_candidate_stability_artifact,
    require_candidate_stability_artifact_binding_on_connection,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.impact_cube_binding import (
    StrategyImpactCubeArtifactBinding,
    load_strategy_impact_cube_artifact,
    require_strategy_impact_cube_artifact_binding_on_connection,
)
from marvis.packs.strategy.model_evidence_tools import (
    StrategyModelEvidenceV2ArtifactBinding,
    load_strategy_model_evidence_v2_artifact,
    require_strategy_model_evidence_v2_artifact_binding_on_connection,
)
from marvis.packs.strategy.pool_impact_tools import (
    StrategyPoolImpactArtifactBinding,
    load_strategy_pool_impact_artifact,
    require_strategy_pool_impact_artifact_binding_on_connection,
)
from marvis.packs.strategy.pool_stability_tools import (
    StrategyPoolStabilityArtifactBinding,
    load_strategy_pool_stability_artifact,
    require_strategy_pool_stability_artifact_binding_on_connection,
)
from marvis.packs.strategy.pool_tools import (
    StrategyCandidatePoolArtifactBinding,
    load_current_strategy_candidate_pool_artifact,
    require_strategy_candidate_pool_artifact_binding_on_connection,
)
from marvis.packs.strategy.pool_validation_tools import (
    StrategyPoolValidationArtifactBinding,
    load_strategy_pool_validation_artifacts,
    require_strategy_pool_validation_artifact_binding_on_connection,
    validate_strategy_pool_validation_artifact_refs,
)
from marvis.packs.strategy.project_context import build_report_field, build_source_ref
from marvis.packs.strategy.project_context_tools import (
    StrategyProjectContextArtifactBinding,
    load_current_strategy_project_context_artifact,
    require_strategy_project_context_artifact_binding_on_connection,
)
from marvis.packs.strategy.report_bundle import (
    REPORT_STATUSES,
    build_strategy_report_bundle,
    validate_strategy_report_bundle,
)
from marvis.packs.strategy.report_bundle_adapters import (
    build_strategy_report_bundle_source_inputs,
)
from marvis.packs.strategy.sample_design_v2_tools import (
    StrategySampleDesignV2ArtifactBinding,
    load_any_strategy_sample_design_v2_artifacts,
    require_any_strategy_sample_design_v2_artifact_binding_on_connection,
    resolve_strategy_sample_design_v2_source_mode,
)
from marvis.packs.strategy.voting_candidate_search_tools import (
    VotingCandidateSearchArtifactBinding,
    load_voting_candidate_search_artifact,
    require_voting_candidate_search_artifact_binding_on_connection,
)
from marvis.repositories.strategy_reports import (
    STRATEGY_REPORT_ORIGIN_TOOL,
    STRATEGY_REPORT_OUTPUT_KINDS,
    StrategyReportConflictError,
    StrategyReportDataError,
    StrategyReportNotFoundError,
    StrategyReportRepository,
    build_strategy_report_output_artifact_provenance,
)
from marvis.repositories.task_artifacts import (
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
)

if TYPE_CHECKING:
    from marvis.packs.strategy.sample_design_v2_native_tools import (
        StrategySampleDesignV2NativeArtifactBinding,
    )


BUILD_STRATEGY_REPORT_BUNDLE_V2_TOOL_SCHEMA_VERSION = (
    "strategy.build-report-bundle-v2-tool.v6"
)
BUILD_STRATEGY_REPORT_BUNDLE_V2_AUDIT_KIND = (
    "strategy.report_bundle.published"
)

_OUTPUT_FORMATS = ("json", "markdown", "xlsx", "docx")
_INPUT_FIELDS = frozenset(
    {
        "title",
        "status",
        "report_revision",
        "previous_report_id",
        "previous_report_content_hash",
        "generated_at",
        "project_context_ref",
        "sample_design_ref",
        "candidate_pool_ref",
        "pool_validation_refs",
        "candidate_stability_ref",
        "pool_stability_ref",
        "voting_candidate_search_ref",
        "pool_impact_ref",
        "impact_cube_ref",
        "strategy_identity",
        "model_evidence_ref",
        "training_evidence_ref",
        "score_evidence_ref",
    }
)
_REQUIRED_INPUT_FIELDS = frozenset(
    {
        "title",
        "status",
        "report_revision",
        "generated_at",
        "project_context_ref",
        "sample_design_ref",
        "candidate_pool_ref",
    }
)
_OPTIONAL_INPUT_FIELDS = _INPUT_FIELDS - _REQUIRED_INPUT_FIELDS
_PROJECT_CONTEXT_REF_FIELDS = frozenset(
    {
        "artifact_id",
        "expected_artifact_content_hash",
        "expected_revision",
        "expected_revision_id",
        "expected_state_hash",
    }
)
_SAMPLE_REF_FIELDS = frozenset(
    {
        "membership_artifact_id",
        "expected_membership_artifact_content_hash",
        "bundle_artifact_id",
        "expected_bundle_artifact_content_hash",
        "expected_bundle_id",
        "expected_sample_design_id",
        "expected_sample_design_content_hash",
    }
)
_POOL_REF_FIELDS = frozenset(
    {
        "strategy_type",
        "expected_pool_revision",
        "expected_pool_snapshot_hash",
        "expected_artifact_id",
        "expected_artifact_content_hash",
    }
)
_CANDIDATE_STABILITY_REF_FIELDS = frozenset(
    {
        "artifact_id",
        "expected_artifact_content_hash",
        "expected_stability_id",
        "expected_stability_content_hash",
    }
)
_POOL_STABILITY_REF_FIELDS = frozenset(
    {
        "artifact_id",
        "expected_artifact_content_hash",
        "expected_stability_id",
        "expected_stability_content_hash",
    }
)
_VOTING_CANDIDATE_SEARCH_REF_FIELDS = frozenset(
    {
        "artifact_id",
        "expected_artifact_content_hash",
        "expected_search_id",
        "expected_search_content_hash",
    }
)
_IMPACT_REF_FIELDS = frozenset(
    {
        "artifact_id",
        "expected_artifact_content_hash",
        "expected_assessment_id",
        "expected_assessment_content_hash",
    }
)
_IMPACT_CUBE_REF_FIELDS = frozenset(
    {
        "artifact_id",
        "expected_artifact_content_hash",
        "expected_cube_id",
        "expected_cube_content_hash",
    }
)
_STRATEGY_IDENTITY_FIELDS = frozenset(
    {"strategy_id", "strategy_version", "strategy_type"}
)
_MODEL_REF_FIELDS = frozenset(
    {
        "artifact_id",
        "expected_artifact_content_hash",
        "expected_bundle_id",
        "expected_bundle_content_hash",
    }
)
_TRAINING_REF_FIELDS = frozenset(
    {
        "sample_design_ref",
        "model_binary_artifact_id",
        "expected_model_binary_artifact_content_hash",
        "evidence_artifact_id",
        "expected_evidence_artifact_content_hash",
        "expected_experiment_id",
        "expected_model_artifact_id",
        "expected_evidence_id",
        "expected_evidence_content_hash",
    }
)
_SCORE_REF_FIELDS = frozenset(
    {
        "evidence_artifact_id",
        "expected_evidence_artifact_content_hash",
        "score_vector_artifact_id",
        "expected_score_vector_artifact_content_hash",
    }
)
_OUTPUT_FIELDS = frozenset(
    {
        "schema_version",
        "report_id",
        "report_revision",
        "content_hash",
        "status",
        "strategy_id",
        "strategy_version",
        "strategy_type",
        "bundle",
        "artifacts",
        "warnings",
        "not_created_strategy",
        "not_adopted",
        "not_deployed",
    }
)
_OUTPUT_ARTIFACT_FIELDS = frozenset(
    {
        "artifact_id",
        "kind",
        "format",
        "filename",
        "content_hash",
        "download_url",
    }
)
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_REPORT_ID_RE = re.compile(r"^strategy-report-[0-9a-f]{24}$")
_CANDIDATE_STABILITY_ID_RE = re.compile(
    r"^candidate-stability-[0-9a-f]{24}$"
)
_POOL_STABILITY_ID_RE = re.compile(
    r"^strategy-pool-stability-[0-9a-f]{24}$"
)
_VOTING_CANDIDATE_SEARCH_ID_RE = re.compile(
    r"^voting-search-[0-9a-f]{32}$"
)
_IMPACT_CUBE_ID_RE = re.compile(
    r"^strategy-impact-cube-[0-9a-f]{24}$"
)
_STRATEGY_TYPES = frozenset(
    {"approval", "reject", "limit", "pricing", "segmentation"}
)
_MAX_INPUT_BYTES = 1024 * 1024

_BOUNDARY_ERRORS = (
    ArtifactTransactionError,
    ModelingError,
    StrategyReportConflictError,
    StrategyReportDataError,
    StrategyReportNotFoundError,
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
)


@dataclass(frozen=True)
class _ReportSources:
    project_context: StrategyProjectContextArtifactBinding
    sample_design: (
        StrategySampleDesignV2ArtifactBinding
        | StrategySampleDesignV2NativeArtifactBinding
    )
    candidate_pool: StrategyCandidatePoolArtifactBinding
    pool_validations: tuple[StrategyPoolValidationArtifactBinding, ...]
    candidate_stability: StrategyCandidateStabilityArtifactBinding | None
    pool_stability: StrategyPoolStabilityArtifactBinding | None
    voting_candidate_search: VotingCandidateSearchArtifactBinding | None
    pool_impact: StrategyPoolImpactArtifactBinding | None
    impact_cube: StrategyImpactCubeArtifactBinding | None
    model_evidence: StrategyModelEvidenceV2ArtifactBinding | None
    training_evidence: ModelingTrainingEvidenceArtifactBinding | None
    score_evidence: ModelScoreEvidenceArtifactBinding | None


def run_build_strategy_report_bundle_v2(inputs, ctx, runtime) -> dict[str, Any]:
    """Build and atomically publish one exact report revision."""

    try:
        request = _validate_inputs(inputs)
        task_id = _text(ctx.task_id, "task_id")
        sources = _load_sources(
            runtime,
            task_id=task_id,
            request=request,
        )
        request_hash = _request_hash(request)
        source_inputs = build_strategy_report_bundle_source_inputs(
            project_context=sources.project_context,
            sample_design=sources.sample_design,
            candidate_pool=sources.candidate_pool,
            pool_validations=sources.pool_validations,
            candidate_stability=sources.candidate_stability,
            pool_stability=sources.pool_stability,
            voting_candidate_search=sources.voting_candidate_search,
            pool_impact=sources.pool_impact,
            impact_cube=sources.impact_cube,
            model_evidence=sources.model_evidence,
            training_evidence=sources.training_evidence,
            score_evidence=sources.score_evidence,
        )
        title_source = build_source_ref(
            kind="tool_input",
            ref_id=f"strategy-report-request-{request_hash[:24]}",
            content_hash=request_hash,
        )
        title = build_report_field(
            value=request["title"],
            availability="present",
            origin="user",
            source_refs=[title_source],
            as_of=sources.project_context.revision["state"]["as_of"],
        )
        strategy_identity = request["strategy_identity"]
        strategy_type = (
            sources.candidate_pool.strategy_type
            if strategy_identity is None
            else strategy_identity["strategy_type"]
        )
        bundle = build_strategy_report_bundle(
            task_id=task_id,
            report_revision=request["report_revision"],
            strategy_id=(
                None
                if strategy_identity is None
                else strategy_identity["strategy_id"]
            ),
            strategy_version=(
                None
                if strategy_identity is None
                else strategy_identity["strategy_version"]
            ),
            strategy_type=strategy_type,
            title=title,
            status=request["status"],
            generated_at=request["generated_at"],
            previous_report_id=request["previous_report_id"],
            **source_inputs,
        )
        rendered = render_strategy_report_bundle(bundle)
        if set(rendered) != set(_OUTPUT_FORMATS) or not all(
            isinstance(rendered[item], bytes) for item in _OUTPUT_FORMATS
        ):
            raise StrategyError(
                "strategy report renderer returned an invalid output set"
            )
        return _publish_report(
            runtime,
            task_id=task_id,
            request=request,
            request_hash=request_hash,
            sources=sources,
            bundle=bundle,
            rendered=rendered,
        )
    except StrategyError:
        raise
    except _BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc


def validate_build_strategy_report_bundle_v2_tool_output(
    value: object,
) -> dict[str, Any]:
    """Validate the complete cached Tool envelope without trusting flat fields."""

    if not isinstance(value, Mapping) or set(value) != _OUTPUT_FIELDS:
        raise StrategyError("build_report_bundle_v2 output envelope is invalid")
    normalized = _canonical_object(value, "build_report_bundle_v2 output")
    if (
        normalized["schema_version"]
        != BUILD_STRATEGY_REPORT_BUNDLE_V2_TOOL_SCHEMA_VERSION
    ):
        raise StrategyError(
            "build_report_bundle_v2 output schema_version is invalid"
        )
    try:
        bundle = validate_strategy_report_bundle(normalized["bundle"])
        rendered = render_strategy_report_bundle(bundle)
    except StrategyError:
        raise
    except Exception as exc:
        raise StrategyError(
            "build_report_bundle_v2 output bundle could not be rendered"
        ) from exc
    expected_scalars = {
        "report_id": bundle["report_id"],
        "report_revision": bundle["report_revision"],
        "content_hash": bundle["content_sha256"],
        "status": bundle["status"],
        "strategy_id": bundle["strategy_id"],
        "strategy_version": bundle["strategy_version"],
        "strategy_type": bundle["strategy_type"],
        "warnings": _report_warnings(bundle),
        "not_created_strategy": True,
        "not_adopted": True,
        "not_deployed": True,
    }
    for field, expected in expected_scalars.items():
        if normalized[field] != expected:
            raise StrategyError(
                f"build_report_bundle_v2 output {field} drifted"
            )

    artifacts = normalized["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != len(_OUTPUT_FORMATS):
        raise StrategyError(
            "build_report_bundle_v2 output needs four canonical artifacts"
        )
    for output_format, artifact in zip(_OUTPUT_FORMATS, artifacts, strict=True):
        if not isinstance(artifact, dict) or set(artifact) != _OUTPUT_ARTIFACT_FIELDS:
            raise StrategyError(
                "build_report_bundle_v2 output artifact is invalid"
            )
        artifact_id = _hash(
            artifact["artifact_id"],
            f"artifacts.{output_format}.artifact_id",
        )
        expected_hash = hashlib.sha256(rendered[output_format]).hexdigest()
        suffix = "md" if output_format == "markdown" else output_format
        expected = {
            "kind": STRATEGY_REPORT_OUTPUT_KINDS[output_format],
            "format": output_format,
            "filename": f"report.{suffix}",
            "content_hash": expected_hash,
            "download_url": (
                f"/api/tasks/{quote(bundle['task_id'], safe='')}"
                f"/task-artifacts/{quote(artifact_id, safe='')}/download"
                f"?expected_content_hash={quote(expected_hash, safe='')}"
            ),
        }
        for field, expected_value in expected.items():
            if artifact[field] != expected_value:
                raise StrategyError(
                    "build_report_bundle_v2 output artifact "
                    f"{output_format}.{field} drifted"
                )
    normalized["bundle"] = bundle
    return normalized


def _load_sources(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
) -> _ReportSources:
    project_context = load_current_strategy_project_context_artifact(
        runtime,
        task_id=task_id,
    )
    if project_context is None:
        raise StrategyError("current strategy project context not found")
    expected_project_context = {
        "artifact_id": project_context.artifact_id,
        "expected_artifact_content_hash": (
            project_context.artifact_content_hash
        ),
        "expected_revision": project_context.revision["revision"],
        "expected_revision_id": project_context.revision["revision_id"],
        "expected_state_hash": project_context.revision["state_hash"],
    }
    if expected_project_context != request["project_context_ref"]:
        raise StrategyError(
            "current strategy project context no longer matches the exact "
            "planned revision"
        )
    sample_design = load_any_strategy_sample_design_v2_artifacts(
        runtime,
        task_id=task_id,
        **request["sample_design_ref"],
    )
    resolve_strategy_sample_design_v2_source_mode(
        sample_design.bundle["sample_design"],
        capability="physical_v2",
        consumer="strategy_report_bundle",
    )
    candidate_pool = load_current_strategy_candidate_pool_artifact(
        runtime,
        task_id=task_id,
        **request["candidate_pool_ref"],
    )
    pool_validations = load_strategy_pool_validation_artifacts(
        runtime,
        task_id=task_id,
        refs=request["pool_validation_refs"],
        candidate_pool=candidate_pool,
        sample_design=sample_design,
    )
    candidate_stability = (
        None
        if request["candidate_stability_ref"] is None
        else load_candidate_stability_artifact(
            runtime,
            task_id=task_id,
            **request["candidate_stability_ref"],
        )
    )
    voting_candidate_search = (
        None
        if request["voting_candidate_search_ref"] is None
        else load_voting_candidate_search_artifact(
            runtime,
            task_id=task_id,
            **request["voting_candidate_search_ref"],
        )
    )
    impact_cube = (
        None
        if request["impact_cube_ref"] is None
        else load_strategy_impact_cube_artifact(
            runtime,
            task_id=task_id,
            **request["impact_cube_ref"],
        )
    )
    pool_stability = (
        None
        if request["pool_stability_ref"] is None
        else load_strategy_pool_stability_artifact(
            runtime,
            task_id=task_id,
            **request["pool_stability_ref"],
        )
    )
    pool_impact = (
        None
        if impact_cube is not None
        else load_strategy_pool_impact_artifact(
            runtime,
            task_id=task_id,
            **request["pool_impact_ref"],
        )
    )
    if pool_impact is not None and (
        pool_impact.pool.artifact_id != candidate_pool.artifact_id
        or pool_impact.pool.artifact_content_hash
        != candidate_pool.artifact_content_hash
        or pool_impact.pool.pool != candidate_pool.pool
    ):
        raise StrategyError(
            "Pool impact does not bind the exact current Strategy Pool"
        )

    modeling_runtime = _modeling_runtime(runtime)
    model_evidence = (
        None
        if request["model_evidence_ref"] is None
        else load_strategy_model_evidence_v2_artifact(
            modeling_runtime,
            task_id=task_id,
            sample_design_ref=request["sample_design_ref"],
            **request["model_evidence_ref"],
        )
    )
    training_evidence = (
        None
        if request["training_evidence_ref"] is None
        else load_modeling_training_evidence_artifacts(
            modeling_runtime,
            task_id=task_id,
            **request["training_evidence_ref"],
        )
    )
    score_evidence = (
        None
        if request["score_evidence_ref"] is None
        else load_model_score_evidence_artifacts(
            modeling_runtime,
            task_id=task_id,
            **request["score_evidence_ref"],
        )
    )
    if score_evidence is not None:
        score_training_ref = build_training_evidence_ref(score_evidence.training)
        if training_evidence is None:
            training_evidence = score_evidence.training
        elif build_training_evidence_ref(training_evidence) != score_training_ref:
            raise StrategyError(
                "score evidence and training evidence refs do not match"
            )
    return _ReportSources(
        project_context=project_context,
        sample_design=sample_design,
        candidate_pool=candidate_pool,
        pool_validations=pool_validations,
        candidate_stability=candidate_stability,
        pool_stability=pool_stability,
        voting_candidate_search=voting_candidate_search,
        pool_impact=pool_impact,
        impact_cube=impact_cube,
        model_evidence=model_evidence,
        training_evidence=training_evidence,
        score_evidence=score_evidence,
    )


def _modeling_runtime(runtime):
    if hasattr(runtime, "experiments") and hasattr(runtime, "modeling_repo"):
        return runtime
    from marvis.db import ModelingRepository
    from marvis.packs.modeling.experiment import ExperimentStore

    values = dict(vars(runtime))
    values.setdefault("experiments", ExperimentStore(runtime.settings.db_path))
    values.setdefault(
        "modeling_repo",
        ModelingRepository(runtime.settings.db_path),
    )
    return SimpleNamespace(**values)


def _publish_report(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
    request_hash: str,
    sources: _ReportSources,
    bundle: Mapping[str, Any],
    rendered: Mapping[str, bytes],
) -> dict[str, Any]:
    tasks_root = Path(runtime.settings.tasks_dir).absolute()
    output_dir = _prepare_output_directory(
        tasks_root,
        task_id=task_id,
        report_id=str(bundle["report_id"]),
    )
    paths = {
        output_format: output_dir
        / f"report.{'md' if output_format == 'markdown' else output_format}"
        for output_format in _OUTPUT_FORMATS
    }
    hashes = {
        output_format: hashlib.sha256(rendered[output_format]).hexdigest()
        for output_format in _OUTPUT_FORMATS
    }
    provenances = {
        output_format: build_strategy_report_output_artifact_provenance(
            bundle,
            output_format=output_format,
        )
        for output_format in _OUTPUT_FORMATS
    }
    uow = ArtifactUnitOfWork()
    try:
        staged = {
            output_format: uow.stage_file(
                output_dir,
                paths[output_format].name,
            )
            for output_format in _OUTPUT_FORMATS
        }
        for output_format in _OUTPUT_FORMATS:
            _write_private_bytes(
                staged[output_format].path,
                rendered[output_format],
            )
    except Exception:
        uow.rollback()
        raise

    report_repository = StrategyReportRepository(runtime.settings.db_path)
    reused = False
    db_committed = False
    rollback_attempted_under_lock = False
    result: dict[str, Any] | None = None
    try:
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                _revalidate_sources(conn, sources)
                reused = _prepare_outputs_under_lock(
                    conn,
                    uow=uow,
                    task_id=task_id,
                    tasks_root=tasks_root,
                    paths=paths,
                    hashes=hashes,
                    provenances=provenances,
                    rendered=rendered,
                )
                records = {}
                for output_format in _OUTPUT_FORMATS:
                    records[output_format] = (
                        runtime.task_artifacts.register_on_connection(
                            conn,
                            task_id=task_id,
                            kind=STRATEGY_REPORT_OUTPUT_KINDS[output_format],
                            path=str(paths[output_format]),
                            content_hash=hashes[output_format],
                            origin_tool=STRATEGY_REPORT_ORIGIN_TOOL,
                            provenance=provenances[output_format],
                            created_at=request["generated_at"],
                        )
                    )
                publication = report_repository.publish_on_connection(
                    conn,
                    bundle=bundle,
                    artifacts=records,
                    expected_revision=request["report_revision"] - 1,
                    expected_report_id=request["previous_report_id"],
                    expected_content_hash=request[
                        "previous_report_content_hash"
                    ],
                    created_at=request["generated_at"],
                )
                _write_or_require_audit(
                    conn,
                    runtime=runtime,
                    request_hash=request_hash,
                    sources=sources,
                    bundle=bundle,
                    records=records,
                )
                result = validate_build_strategy_report_bundle_v2_tool_output(
                    _tool_output(publication)
                )
                _revalidate_sources(conn, sources)
                for output_format in _OUTPUT_FORMATS:
                    _read_exact_regular_file(
                        paths[output_format],
                        root=tasks_root,
                        expected=rendered[output_format],
                        expected_hash=hashes[output_format],
                        label=f"{output_format} report output",
                    )
                conn.commit()
                db_committed = True
            except Exception:
                rollback_attempted_under_lock = True
                uow.rollback()
                raise
        if not reused:
            uow.commit()
    except Exception:
        if not db_committed and not rollback_attempted_under_lock:
            uow.rollback()
        raise
    if result is None:
        raise StrategyError("strategy report publication returned no result")
    return result


def _prepare_outputs_under_lock(
    conn,
    *,
    uow: ArtifactUnitOfWork,
    task_id: str,
    tasks_root: Path,
    paths: Mapping[str, Path],
    hashes: Mapping[str, str],
    provenances: Mapping[str, Mapping[str, Any]],
    rendered: Mapping[str, bytes],
) -> bool:
    rows = {}
    for output_format in _OUTPUT_FORMATS:
        rows[output_format] = conn.execute(
            """
            SELECT * FROM task_artifacts
             WHERE task_id = ? AND kind = ? AND path = ?
            """,
            (
                task_id,
                STRATEGY_REPORT_OUTPUT_KINDS[output_format],
                str(paths[output_format]),
            ),
        ).fetchone()
    present = [item for item, row in rows.items() if row is not None]
    if present and len(present) != len(_OUTPUT_FORMATS):
        raise StrategyError(
            "strategy report output registry set is incomplete"
        )
    if present:
        for output_format in _OUTPUT_FORMATS:
            _require_existing_output_row(
                rows[output_format],
                task_id=task_id,
                kind=STRATEGY_REPORT_OUTPUT_KINDS[output_format],
                path=paths[output_format],
                content_hash=hashes[output_format],
                provenance=provenances[output_format],
            )
            _read_exact_regular_file(
                paths[output_format],
                root=tasks_root,
                expected=rendered[output_format],
                expected_hash=hashes[output_format],
                label=f"{output_format} report output",
            )
        uow.rollback()
        return True

    for output_format in _OUTPUT_FORMATS:
        if paths[output_format].exists() or paths[output_format].is_symlink():
            raise StrategyError(
                f"{output_format} report output path exists without a registry row"
            )
    _require_safe_directory(output_dir=next(iter(paths.values())).parent, root=tasks_root)
    uow.promote_all()
    for output_format in _OUTPUT_FORMATS:
        _read_exact_regular_file(
            paths[output_format],
            root=tasks_root,
            expected=rendered[output_format],
            expected_hash=hashes[output_format],
            label=f"{output_format} report output",
        )
    return False


def _revalidate_sources(conn, sources: _ReportSources) -> None:
    require_strategy_project_context_artifact_binding_on_connection(
        conn,
        sources.project_context,
    )
    require_any_strategy_sample_design_v2_artifact_binding_on_connection(
        conn,
        sources.sample_design,
    )
    require_strategy_candidate_pool_artifact_binding_on_connection(
        conn,
        sources.candidate_pool,
    )
    for validation in sources.pool_validations:
        require_strategy_pool_validation_artifact_binding_on_connection(
            conn,
            validation,
        )
    if sources.candidate_stability is not None:
        require_candidate_stability_artifact_binding_on_connection(
            conn,
            sources.candidate_stability,
        )
    if sources.pool_stability is not None:
        require_strategy_pool_stability_artifact_binding_on_connection(
            conn,
            sources.pool_stability,
        )
    if sources.voting_candidate_search is not None:
        require_voting_candidate_search_artifact_binding_on_connection(
            conn,
            sources.voting_candidate_search,
        )
    if sources.impact_cube is not None:
        require_strategy_impact_cube_artifact_binding_on_connection(
            conn,
            sources.impact_cube,
        )
    elif sources.pool_impact is not None:
        require_strategy_pool_impact_artifact_binding_on_connection(
            conn,
            sources.pool_impact,
        )
    else:
        raise StrategyError(
            "strategy report impact evidence disappeared"
        )
    if sources.model_evidence is not None:
        require_strategy_model_evidence_v2_artifact_binding_on_connection(
            conn,
            sources.model_evidence,
        )
    if sources.training_evidence is not None:
        require_modeling_training_evidence_artifact_binding_on_connection(
            conn,
            sources.training_evidence,
        )
    if sources.score_evidence is not None:
        require_model_score_evidence_artifact_binding_on_connection(
            conn,
            sources.score_evidence,
        )


def _write_or_require_audit(
    conn,
    *,
    runtime,
    request_hash: str,
    sources: _ReportSources,
    bundle: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
) -> None:
    detail = {
        "task_id": bundle["task_id"],
        "report_id": bundle["report_id"],
        "report_revision": bundle["report_revision"],
        "report_content_hash": bundle["content_sha256"],
        "strategy_id": bundle["strategy_id"],
        "strategy_version": bundle["strategy_version"],
        "strategy_type": bundle["strategy_type"],
        "source_artifacts": _audit_source_artifacts(sources),
        "output_artifacts": {
            output_format: {
                "artifact_id": records[output_format]["id"],
                "content_hash": records[output_format]["content_hash"],
                "kind": records[output_format]["kind"],
            }
            for output_format in _OUTPUT_FORMATS
        },
        "data_classification": bundle["data_classification"],
        "not_created_strategy": True,
        "not_adopted": True,
        "not_deployed": True,
    }
    rows = conn.execute(
        """
        SELECT actor, inputs_hash, outcome, detail_json
          FROM audit
         WHERE kind = ? AND target_ref = ?
         ORDER BY at, id
        """,
        (
            BUILD_STRATEGY_REPORT_BUNDLE_V2_AUDIT_KIND,
            bundle["report_id"],
        ),
    ).fetchall()
    if len(rows) > 1:
        raise StrategyError("strategy report publication audit is duplicated")
    if rows:
        row = rows[0]
        try:
            persisted_detail = json.loads(str(row["detail_json"]))
        except json.JSONDecodeError as exc:
            raise StrategyError(
                "strategy report publication audit is invalid"
            ) from exc
        if (
            str(row["actor"]) != "system"
            or str(row["inputs_hash"]) != request_hash
            or str(row["outcome"]) != "succeeded"
            or persisted_detail != detail
        ):
            raise StrategyError(
                "strategy report publication audit binding changed"
            )
        return
    runtime.repo.write_audit_on_connection(
        conn,
        kind=BUILD_STRATEGY_REPORT_BUNDLE_V2_AUDIT_KIND,
        target_ref=bundle["report_id"],
        inputs_hash=request_hash,
        outcome="succeeded",
        detail=detail,
    )


def _audit_source_artifacts(sources: _ReportSources) -> dict[str, Any]:
    result: dict[str, Any] = {
        "project_context": {
            "artifact_id": sources.project_context.artifact_id,
            "content_hash": sources.project_context.artifact_content_hash,
        },
        "sample_design": {
            "membership_artifact_id": sources.sample_design.membership_artifact_id,
            "membership_content_hash": (
                sources.sample_design.membership_artifact_content_hash
            ),
            "bundle_artifact_id": sources.sample_design.bundle_artifact_id,
            "bundle_content_hash": (
                sources.sample_design.bundle_artifact_content_hash
            ),
        },
        "candidate_pool": {
            "artifact_id": sources.candidate_pool.artifact_id,
            "content_hash": sources.candidate_pool.artifact_content_hash,
        },
        "pool_validations": {
            binding.evidence["partition"]: {
                "artifact_id": binding.artifact_id,
                "content_hash": binding.artifact_content_hash,
                "evidence_id": binding.evidence["evidence_id"],
                "evidence_content_hash": binding.evidence["content_hash"],
                "validation_status": binding.evidence["lifecycle"][
                    "validation_status"
                ],
                "not_adopted": True,
                "not_deployed": True,
            }
            for binding in sources.pool_validations
        },
        "candidate_stability": None,
        "pool_stability": None,
        "voting_candidate_search": None,
        "pool_impact": None,
        "impact_cube": None,
        "model_evidence": None,
        "training_evidence": None,
        "score_evidence": None,
    }
    if sources.impact_cube is not None:
        result["impact_cube"] = {
            "artifact_id": sources.impact_cube.artifact_id,
            "content_hash": sources.impact_cube.artifact_content_hash,
            "producer_run_ref": {
                "kind": "tool_run",
                "ref_id": sources.impact_cube.artifact_provenance[
                    "producer_run"
                ]["run_id"],
                "content_hash": sources.impact_cube.artifact_provenance[
                    "producer_run"
                ]["content_hash"],
            },
        }
    elif sources.pool_impact is not None:
        result["pool_impact"] = {
            "artifact_id": sources.pool_impact.artifact_id,
            "content_hash": sources.pool_impact.artifact_content_hash,
        }
    if sources.candidate_stability is not None:
        result["candidate_stability"] = {
            "artifact_id": sources.candidate_stability.artifact_id,
            "content_hash": (
                sources.candidate_stability.artifact_content_hash
            ),
            "stability_id": sources.candidate_stability.stability[
                "stability_id"
            ],
            "stability_content_hash": sources.candidate_stability.stability[
                "content_hash"
            ],
        }
    if sources.pool_stability is not None:
        producer_run = sources.pool_stability.artifact_provenance[
            "producer_run"
        ]
        result["pool_stability"] = {
            "artifact_id": sources.pool_stability.artifact_id,
            "content_hash": sources.pool_stability.artifact_content_hash,
            "stability_id": sources.pool_stability.stability["stability_id"],
            "stability_content_hash": sources.pool_stability.stability[
                "content_hash"
            ],
            "producer_run_ref": {
                "kind": "tool_run",
                "ref_id": producer_run["run_id"],
                "content_hash": producer_run["content_hash"],
            },
        }
    if sources.voting_candidate_search is not None:
        result["voting_candidate_search"] = {
            "artifact_id": sources.voting_candidate_search.artifact_id,
            "content_hash": (
                sources.voting_candidate_search.artifact_content_hash
            ),
            "search_id": sources.voting_candidate_search.result["search_id"],
            "search_content_hash": (
                sources.voting_candidate_search.result["content_hash"]
            ),
        }
    if sources.model_evidence is not None:
        result["model_evidence"] = {
            "artifact_id": sources.model_evidence.artifact_id,
            "content_hash": sources.model_evidence.artifact_content_hash,
        }
    if sources.training_evidence is not None:
        result["training_evidence"] = {
            "artifact_id": str(
                sources.training_evidence.evidence_record["id"]
            ),
            "content_hash": str(
                sources.training_evidence.evidence_record["content_hash"]
            ),
        }
    if sources.score_evidence is not None:
        result["score_evidence"] = {
            "artifact_id": str(sources.score_evidence.evidence_record["id"]),
            "content_hash": str(
                sources.score_evidence.evidence_record["content_hash"]
            ),
            "vector_artifact_id": str(
                sources.score_evidence.vector_record["id"]
            ),
            "vector_content_hash": str(
                sources.score_evidence.vector_record["content_hash"]
            ),
        }
    return result


def _tool_output(publication: Mapping[str, Any]) -> dict[str, Any]:
    bundle = validate_strategy_report_bundle(publication["bundle"])
    records = publication["artifacts"]
    artifacts = []
    for output_format in _OUTPUT_FORMATS:
        record = records[output_format]
        artifact_id = str(record["id"])
        artifacts.append(
            {
                "artifact_id": artifact_id,
                "kind": STRATEGY_REPORT_OUTPUT_KINDS[output_format],
                "format": output_format,
                "filename": Path(str(record["path"])).name,
                "content_hash": str(record["content_hash"]),
                "download_url": (
                    f"/api/tasks/{quote(bundle['task_id'], safe='')}"
                    f"/task-artifacts/{quote(artifact_id, safe='')}/download"
                    "?expected_content_hash="
                    f"{quote(str(record['content_hash']), safe='')}"
                ),
            }
        )
    return {
        "schema_version": BUILD_STRATEGY_REPORT_BUNDLE_V2_TOOL_SCHEMA_VERSION,
        "report_id": bundle["report_id"],
        "report_revision": bundle["report_revision"],
        "content_hash": bundle["content_sha256"],
        "status": bundle["status"],
        "strategy_id": bundle["strategy_id"],
        "strategy_version": bundle["strategy_version"],
        "strategy_type": bundle["strategy_type"],
        "bundle": bundle,
        "artifacts": artifacts,
        "warnings": _report_warnings(bundle),
        "not_created_strategy": True,
        "not_adopted": True,
        "not_deployed": True,
    }


def _report_warnings(bundle: Mapping[str, Any]) -> list[str]:
    warnings = []
    seen = set()
    for section in bundle["sections"]:
        for flag in section["red_flags"]:
            if flag["level"] not in {"amber", "red"}:
                continue
            message = str(flag["message"])
            if message not in seen:
                warnings.append(message)
                seen.add(message)
    return warnings


def _validate_inputs(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StrategyError("build_report_bundle_v2 inputs must be an object")
    fields = set(value)
    missing = sorted(_REQUIRED_INPUT_FIELDS - fields)
    unexpected = sorted(fields - _INPUT_FIELDS)
    if missing or unexpected:
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if unexpected:
            detail.append("unsupported " + ", ".join(unexpected))
        raise StrategyError(
            "build_report_bundle_v2 input fields are invalid ("
            + "; ".join(detail)
            + ")"
        )
    candidate = dict(value)
    for field in _OPTIONAL_INPUT_FIELDS:
        candidate.setdefault(
            field,
            [] if field == "pool_validation_refs" else None,
        )
    request = _canonical_object(candidate, "build_report_bundle_v2 inputs")
    if len(_canonical_json(request).encode("utf-8")) > _MAX_INPUT_BYTES:
        raise StrategyError("build_report_bundle_v2 inputs exceed byte budget")

    request["title"] = _text(request["title"], "title")
    request["status"] = _enum(request["status"], REPORT_STATUSES, "status")
    revision = _positive_int(request["report_revision"], "report_revision")
    request["report_revision"] = revision
    request["generated_at"] = _timestamp(
        request["generated_at"],
        "generated_at",
    )
    previous_id = _optional_report_id(
        request["previous_report_id"],
        "previous_report_id",
    )
    previous_hash = _optional_hash(
        request["previous_report_content_hash"],
        "previous_report_content_hash",
    )
    if revision == 1:
        if previous_id is not None or previous_hash is not None:
            raise StrategyError(
                "first report revision requires an absent previous head"
            )
    elif previous_id is None or previous_hash is None:
        raise StrategyError(
            "later report revision requires the exact previous report id and hash"
        )
    request["previous_report_id"] = previous_id
    request["previous_report_content_hash"] = previous_hash
    request["project_context_ref"] = _project_context_ref(
        request["project_context_ref"]
    )
    request["sample_design_ref"] = _sample_ref(request["sample_design_ref"])
    request["candidate_pool_ref"] = _pool_ref(request["candidate_pool_ref"])
    request["pool_validation_refs"] = list(
        validate_strategy_pool_validation_artifact_refs(
            request["pool_validation_refs"]
        )
    )
    request["candidate_stability_ref"] = _optional_candidate_stability_ref(
        request["candidate_stability_ref"]
    )
    request["pool_stability_ref"] = _optional_pool_stability_ref(
        request["pool_stability_ref"]
    )
    request["voting_candidate_search_ref"] = (
        _optional_voting_candidate_search_ref(
            request["voting_candidate_search_ref"]
        )
    )
    request["impact_cube_ref"] = _optional_impact_cube_ref(
        request["impact_cube_ref"]
    )
    request["pool_impact_ref"] = _optional_impact_ref(
        request["pool_impact_ref"]
    )
    if (
        request["impact_cube_ref"] is None
        and request["pool_impact_ref"] is None
    ):
        raise StrategyError(
            "build_report_bundle_v2 requires impact_cube_ref or "
            "pool_impact_ref"
        )
    if request["impact_cube_ref"] is not None:
        request["pool_impact_ref"] = None
    request["strategy_identity"] = _strategy_identity(
        request["strategy_identity"]
    )
    request["model_evidence_ref"] = _optional_model_ref(
        request["model_evidence_ref"]
    )
    request["training_evidence_ref"] = _optional_training_ref(
        request["training_evidence_ref"]
    )
    request["score_evidence_ref"] = _optional_score_ref(
        request["score_evidence_ref"]
    )
    if (
        request["strategy_identity"] is not None
        and request["strategy_identity"]["strategy_type"]
        != request["candidate_pool_ref"]["strategy_type"]
    ):
        raise StrategyError(
            "strategy identity type must match the current Candidate Pool"
        )
    training = request["training_evidence_ref"]
    if (
        training is not None
        and training["sample_design_ref"] != request["sample_design_ref"]
    ):
        raise StrategyError(
            "training evidence must reference the selected sample-design V2"
        )
    return request


def _project_context_ref(value: object) -> dict[str, Any]:
    obj = _exact_object(
        value,
        _PROJECT_CONTEXT_REF_FIELDS,
        "project_context_ref",
    )
    revision_id = _text(
        obj["expected_revision_id"],
        "project_context_ref.expected_revision_id",
    )
    if re.fullmatch(
        r"strategy-project-context-revision-[0-9a-f]{24}",
        revision_id,
    ) is None:
        raise StrategyError(
            "project_context_ref.expected_revision_id is not canonical"
        )
    return {
        "artifact_id": _hash(
            obj["artifact_id"],
            "project_context_ref.artifact_id",
        ),
        "expected_artifact_content_hash": _hash(
            obj["expected_artifact_content_hash"],
            "project_context_ref.expected_artifact_content_hash",
        ),
        "expected_revision": _positive_int(
            obj["expected_revision"],
            "project_context_ref.expected_revision",
        ),
        "expected_revision_id": revision_id,
        "expected_state_hash": _hash(
            obj["expected_state_hash"],
            "project_context_ref.expected_state_hash",
        ),
    }


def _sample_ref(value: object) -> dict[str, Any]:
    obj = _exact_object(value, _SAMPLE_REF_FIELDS, "sample_design_ref")
    return {
        "membership_artifact_id": _hash(
            obj["membership_artifact_id"],
            "sample_design_ref.membership_artifact_id",
        ),
        "expected_membership_artifact_content_hash": _hash(
            obj["expected_membership_artifact_content_hash"],
            "sample_design_ref.expected_membership_artifact_content_hash",
        ),
        "bundle_artifact_id": _hash(
            obj["bundle_artifact_id"],
            "sample_design_ref.bundle_artifact_id",
        ),
        "expected_bundle_artifact_content_hash": _hash(
            obj["expected_bundle_artifact_content_hash"],
            "sample_design_ref.expected_bundle_artifact_content_hash",
        ),
        "expected_bundle_id": _text(
            obj["expected_bundle_id"],
            "sample_design_ref.expected_bundle_id",
        ),
        "expected_sample_design_id": _text(
            obj["expected_sample_design_id"],
            "sample_design_ref.expected_sample_design_id",
        ),
        "expected_sample_design_content_hash": _hash(
            obj["expected_sample_design_content_hash"],
            "sample_design_ref.expected_sample_design_content_hash",
        ),
    }


def _pool_ref(value: object) -> dict[str, Any]:
    obj = _exact_object(value, _POOL_REF_FIELDS, "candidate_pool_ref")
    return {
        "strategy_type": _enum(
            obj["strategy_type"],
            _STRATEGY_TYPES,
            "candidate_pool_ref.strategy_type",
        ),
        "expected_pool_revision": _positive_int(
            obj["expected_pool_revision"],
            "candidate_pool_ref.expected_pool_revision",
        ),
        "expected_pool_snapshot_hash": _hash(
            obj["expected_pool_snapshot_hash"],
            "candidate_pool_ref.expected_pool_snapshot_hash",
        ),
        "expected_artifact_id": _hash(
            obj["expected_artifact_id"],
            "candidate_pool_ref.expected_artifact_id",
        ),
        "expected_artifact_content_hash": _hash(
            obj["expected_artifact_content_hash"],
            "candidate_pool_ref.expected_artifact_content_hash",
        ),
    }


def _optional_candidate_stability_ref(
    value: object,
) -> dict[str, Any] | None:
    if value is None:
        return None
    obj = _exact_object(
        value,
        _CANDIDATE_STABILITY_REF_FIELDS,
        "candidate_stability_ref",
    )
    stability_id = _text(
        obj["expected_stability_id"],
        "candidate_stability_ref.expected_stability_id",
    )
    if _CANDIDATE_STABILITY_ID_RE.fullmatch(stability_id) is None:
        raise StrategyError(
            "candidate_stability_ref.expected_stability_id is not canonical"
        )
    return {
        "artifact_id": _hash(
            obj["artifact_id"],
            "candidate_stability_ref.artifact_id",
        ),
        "expected_artifact_content_hash": _hash(
            obj["expected_artifact_content_hash"],
            "candidate_stability_ref.expected_artifact_content_hash",
        ),
        "expected_stability_id": stability_id,
        "expected_stability_content_hash": _hash(
            obj["expected_stability_content_hash"],
            "candidate_stability_ref.expected_stability_content_hash",
        ),
    }


def _optional_pool_stability_ref(
    value: object,
) -> dict[str, Any] | None:
    if value is None:
        return None
    obj = _exact_object(
        value,
        _POOL_STABILITY_REF_FIELDS,
        "pool_stability_ref",
    )
    stability_id = _text(
        obj["expected_stability_id"],
        "pool_stability_ref.expected_stability_id",
    )
    if _POOL_STABILITY_ID_RE.fullmatch(stability_id) is None:
        raise StrategyError(
            "pool_stability_ref.expected_stability_id is not canonical"
        )
    return {
        "artifact_id": _hash(
            obj["artifact_id"],
            "pool_stability_ref.artifact_id",
        ),
        "expected_artifact_content_hash": _hash(
            obj["expected_artifact_content_hash"],
            "pool_stability_ref.expected_artifact_content_hash",
        ),
        "expected_stability_id": stability_id,
        "expected_stability_content_hash": _hash(
            obj["expected_stability_content_hash"],
            "pool_stability_ref.expected_stability_content_hash",
        ),
    }


def _optional_voting_candidate_search_ref(
    value: object,
) -> dict[str, Any] | None:
    if value is None:
        return None
    obj = _exact_object(
        value,
        _VOTING_CANDIDATE_SEARCH_REF_FIELDS,
        "voting_candidate_search_ref",
    )
    search_id = _text(
        obj["expected_search_id"],
        "voting_candidate_search_ref.expected_search_id",
    )
    if _VOTING_CANDIDATE_SEARCH_ID_RE.fullmatch(search_id) is None:
        raise StrategyError(
            "voting_candidate_search_ref.expected_search_id is not canonical"
        )
    return {
        "artifact_id": _hash(
            obj["artifact_id"],
            "voting_candidate_search_ref.artifact_id",
        ),
        "expected_artifact_content_hash": _hash(
            obj["expected_artifact_content_hash"],
            "voting_candidate_search_ref.expected_artifact_content_hash",
        ),
        "expected_search_id": search_id,
        "expected_search_content_hash": _hash(
            obj["expected_search_content_hash"],
            "voting_candidate_search_ref.expected_search_content_hash",
        ),
    }


def _optional_impact_ref(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    obj = _exact_object(value, _IMPACT_REF_FIELDS, "pool_impact_ref")
    return {
        "artifact_id": _hash(
            obj["artifact_id"],
            "pool_impact_ref.artifact_id",
        ),
        "expected_artifact_content_hash": _hash(
            obj["expected_artifact_content_hash"],
            "pool_impact_ref.expected_artifact_content_hash",
        ),
        "expected_assessment_id": _text(
            obj["expected_assessment_id"],
            "pool_impact_ref.expected_assessment_id",
        ),
        "expected_assessment_content_hash": _hash(
            obj["expected_assessment_content_hash"],
            "pool_impact_ref.expected_assessment_content_hash",
        ),
    }


def _optional_impact_cube_ref(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    obj = _exact_object(
        value,
        _IMPACT_CUBE_REF_FIELDS,
        "impact_cube_ref",
    )
    cube_id = _text(
        obj["expected_cube_id"],
        "impact_cube_ref.expected_cube_id",
    )
    if _IMPACT_CUBE_ID_RE.fullmatch(cube_id) is None:
        raise StrategyError(
            "impact_cube_ref.expected_cube_id is not canonical"
        )
    return {
        "artifact_id": _hash(
            obj["artifact_id"],
            "impact_cube_ref.artifact_id",
        ),
        "expected_artifact_content_hash": _hash(
            obj["expected_artifact_content_hash"],
            "impact_cube_ref.expected_artifact_content_hash",
        ),
        "expected_cube_id": cube_id,
        "expected_cube_content_hash": _hash(
            obj["expected_cube_content_hash"],
            "impact_cube_ref.expected_cube_content_hash",
        ),
    }


def _strategy_identity(value: object) -> dict[str, str] | None:
    if value is None:
        return None
    obj = _exact_object(
        value,
        _STRATEGY_IDENTITY_FIELDS,
        "strategy_identity",
    )
    return {
        "strategy_id": _text(
            obj["strategy_id"],
            "strategy_identity.strategy_id",
        ),
        "strategy_version": _text(
            obj["strategy_version"],
            "strategy_identity.strategy_version",
        ),
        "strategy_type": _enum(
            obj["strategy_type"],
            _STRATEGY_TYPES,
            "strategy_identity.strategy_type",
        ),
    }


def _optional_model_ref(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    obj = _exact_object(value, _MODEL_REF_FIELDS, "model_evidence_ref")
    return {
        "artifact_id": _hash(
            obj["artifact_id"],
            "model_evidence_ref.artifact_id",
        ),
        "expected_artifact_content_hash": _hash(
            obj["expected_artifact_content_hash"],
            "model_evidence_ref.expected_artifact_content_hash",
        ),
        "expected_bundle_id": _text(
            obj["expected_bundle_id"],
            "model_evidence_ref.expected_bundle_id",
        ),
        "expected_bundle_content_hash": _hash(
            obj["expected_bundle_content_hash"],
            "model_evidence_ref.expected_bundle_content_hash",
        ),
    }


def _optional_training_ref(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    obj = _exact_object(
        value,
        _TRAINING_REF_FIELDS,
        "training_evidence_ref",
    )
    result = {"sample_design_ref": _sample_ref(obj["sample_design_ref"])}
    for field in (
        "model_binary_artifact_id",
        "expected_model_binary_artifact_content_hash",
        "evidence_artifact_id",
        "expected_evidence_artifact_content_hash",
    ):
        result[field] = _hash(obj[field], f"training_evidence_ref.{field}")
    for field in (
        "expected_experiment_id",
        "expected_model_artifact_id",
        "expected_evidence_id",
    ):
        result[field] = _text(obj[field], f"training_evidence_ref.{field}")
    result["expected_evidence_content_hash"] = _hash(
        obj["expected_evidence_content_hash"],
        "training_evidence_ref.expected_evidence_content_hash",
    )
    return result


def _optional_score_ref(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    obj = _exact_object(value, _SCORE_REF_FIELDS, "score_evidence_ref")
    return {
        field: _hash(obj[field], f"score_evidence_ref.{field}")
        for field in _SCORE_REF_FIELDS
    }


def _prepare_output_directory(
    tasks_root: Path,
    *,
    task_id: str,
    report_id: str,
) -> Path:
    if "/" in task_id or task_id in {".", ".."}:
        raise StrategyError("task_id is not a safe artifact path segment")
    if _REPORT_ID_RE.fullmatch(report_id) is None:
        raise StrategyError("report_id is not canonical")
    root = tasks_root.absolute()
    root.mkdir(parents=True, exist_ok=True)
    _require_directory(root, "task artifact root")
    task_dir = root / task_id
    reports_dir = task_dir / "strategy_reports"
    output_dir = reports_dir / report_id
    for path, label in (
        (task_dir, "task artifact directory"),
        (reports_dir, "strategy report root"),
        (output_dir, "strategy report output directory"),
    ):
        if path.exists() or path.is_symlink():
            _require_directory(path, label)
        else:
            try:
                path.mkdir(mode=0o700)
            except FileExistsError:
                # An identical writer may create the canonical directory
                # before either publication acquires BEGIN IMMEDIATE.
                pass
            _require_directory(path, label)
    _require_safe_directory(output_dir=output_dir, root=root)
    return output_dir


def _require_safe_directory(*, output_dir: Path, root: Path) -> None:
    try:
        output_dir.relative_to(root)
    except ValueError as exc:
        raise StrategyError(
            "strategy report output directory escaped the task root"
        ) from exc
    current = output_dir
    while True:
        _require_directory(current, "strategy report path")
        if current == root:
            break
        current = current.parent


def _require_directory(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise StrategyError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise StrategyError(f"{label} must be a regular directory, not a symlink")


def _write_private_bytes(path: Path, payload: bytes) -> None:
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_TRUNC
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        os.fchmod(descriptor, 0o600)
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("short artifact write")
            written += count
    except OSError as exc:
        raise StrategyError("strategy report output could not be staged") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_exact_regular_file(
    path: Path,
    *,
    root: Path,
    expected: bytes,
    expected_hash: str,
    label: str,
) -> bytes:
    _require_safe_directory(output_dir=path.parent, root=root)
    descriptor = -1
    chunks: list[bytes] = []
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != len(expected):
            raise StrategyError(f"{label} is not an exact regular file")
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
            raise StrategyError(f"{label} changed while being read")
        live = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(live.st_mode)
            or (live.st_dev, live.st_ino) != (after.st_dev, after.st_ino)
        ):
            raise StrategyError(
                f"{label} path is not a stable regular file"
            )
    except StrategyError:
        raise
    except OSError as exc:
        raise StrategyError(
            f"{label} is unavailable or is a symlink"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    raw = b"".join(chunks)
    if raw != expected or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(),
        expected_hash,
    ):
        raise StrategyError(f"{label} bytes or hash changed")
    _require_safe_directory(output_dir=path.parent, root=root)
    return raw


def _require_existing_output_row(
    row: sqlite3.Row,
    *,
    task_id: str,
    kind: str,
    path: Path,
    content_hash: str,
    provenance: Mapping[str, Any],
) -> None:
    try:
        persisted_provenance = json.loads(str(row["provenance_json"]))
    except (json.JSONDecodeError, TypeError, KeyError) as exc:
        raise StrategyError(
            "strategy report output registry provenance is invalid"
        ) from exc
    expected = {
        "task_id": task_id,
        "kind": kind,
        "path": str(path),
        "content_hash": content_hash,
        "origin_tool": STRATEGY_REPORT_ORIGIN_TOOL,
    }
    if any(str(row[field]) != value for field, value in expected.items()):
        raise StrategyError(
            "strategy report output registry binding changed"
        )
    if persisted_provenance != dict(provenance):
        raise StrategyError(
            "strategy report output registry provenance changed"
        )


def _request_hash(request: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(request).encode("utf-8")).hexdigest()


def _exact_object(
    value: object,
    fields: frozenset[str],
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise StrategyError(f"{name} fields are invalid")
    return dict(value)


def _canonical_object(value: object, name: str) -> dict[str, Any]:
    try:
        raw = _canonical_json(value)
        normalized = json.loads(raw)
    except (TypeError, ValueError, RecursionError, MemoryError) as exc:
        raise StrategyError(f"{name} must be finite canonical JSON") from exc
    if not isinstance(normalized, dict):
        raise StrategyError(f"{name} must be an object")
    return normalized


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise StrategyError(f"{name} must be non-empty text")
    return value.strip()


def _enum(value: object, allowed: frozenset[str], name: str) -> str:
    result = _text(value, name)
    if result not in allowed:
        raise StrategyError(
            f"{name} must be one of {', '.join(sorted(allowed))}"
        )
    return result


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise StrategyError(f"{name} must be a lowercase SHA-256 hash")
    return value


def _optional_hash(value: object, name: str) -> str | None:
    return None if value is None else _hash(value, name)


def _optional_report_id(value: object, name: str) -> str | None:
    if value is None:
        return None
    result = _text(value, name)
    if _REPORT_ID_RE.fullmatch(result) is None:
        raise StrategyError(f"{name} is not a canonical strategy report id")
    return result


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise StrategyError(f"{name} must be a positive integer")
    return value


def _timestamp(value: object, name: str) -> str:
    observed = _text(value, name)
    try:
        parsed = datetime.fromisoformat(observed.replace("Z", "+00:00"))
    except ValueError as exc:
        raise StrategyError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise StrategyError(f"{name} must include a timezone")
    return observed


__all__ = [
    "BUILD_STRATEGY_REPORT_BUNDLE_V2_AUDIT_KIND",
    "BUILD_STRATEGY_REPORT_BUNDLE_V2_TOOL_SCHEMA_VERSION",
    "load_strategy_impact_cube_artifact",
    "run_build_strategy_report_bundle_v2",
    "validate_build_strategy_report_bundle_v2_tool_output",
]
