"""Governed task-artifact boundary for Strategy Pool impact evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.errors import (
    DataLayerError,
    DatasetContentDriftError,
    NanLabelNotConfirmedError,
)
from marvis.data.labels import require_labels_confirmed
from marvis.data.workspace import (
    data_semantic_mapping_from_dict,
    data_semantic_mapping_hash,
)
from marvis.files import sha256_file
from marvis.packs.strategy.automatic_tree_sample_design import (
    sample_design_ref_from_automatic_tree_source_refs,
)
from marvis.packs.strategy.dsl import strategy_spec_hash
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pool import compile_strategy_pool, validate_strategy_pool
from marvis.packs.strategy.pool_impact import (
    STRATEGY_POOL_IMPACT_PRODUCER_VERSION,
    build_strategy_pool_impact_assessment,
    canonical_strategy_pool_impact_json,
    validate_strategy_pool_impact_assessment,
)
from marvis.packs.strategy.sample_design_binding import (
    StrategySampleDesignExecutionBinding,
    StrategySampleDesignRef,
    bind_strategy_development_frame,
    load_strategy_sample_design_execution_binding,
    revalidate_strategy_sample_design_execution_binding,
)
from marvis.packs.strategy.sample_design_tools import (
    SAMPLE_DESIGN_ARTIFACT_KIND,
    SAMPLE_DESIGN_ORIGIN_TOOL,
)
from marvis.packs.strategy.voting_candidate import (
    VOTING_CANDIDATE_ASSET_SCHEMA_VERSION,
)
from marvis.repositories.data_workspace import (
    DataWorkspaceDataError,
    DataWorkspaceDatasetNotFound,
    DataWorkspaceRepository,
)
from marvis.repositories.strategy import (
    _strategy_from_row,
    _strategy_spec_hash_from_row,
)
from marvis.repositories.strategy_pool import (
    StrategyCandidatePoolConflictError,
    StrategyCandidatePoolDataError,
    StrategyCandidatePoolNotFoundError,
    StrategyCandidatePoolRepository,
)
from marvis.repositories.task_artifacts import (
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
)

if TYPE_CHECKING:
    from marvis.packs.strategy.pool_tools import (
        StrategyCandidatePoolArtifactBinding,
        StrategyPoolDevelopmentExecutionBinding,
    )


POOL_IMPACT_TOOL_SCHEMA_VERSION = "strategy.measure-pool-impact-tool.v2"
POOL_IMPACT_ARTIFACT_KIND = "strategy_pool_impact_json"
POOL_IMPACT_ARTIFACT_SCHEMA_VERSION = "strategy.pool-impact-artifact.v2"
POOL_IMPACT_ORIGIN_TOOL = "strategy.measure_pool_impact"

_INPUT_FIELDS = frozenset(
    {
        "strategy_type",
        "expected_pool_revision",
        "expected_pool_snapshot_hash",
        "dataset_id",
        "expected_dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "target_col",
        "comparison_mode",
        "baseline_strategy_id",
        "month_col",
        "loan_amount_col",
        "overdue_amount_col",
        "drop_nan_labels",
        "sample_design_ref",
    }
)
_OPTIONAL_FIELDS = frozenset(
    {
        "baseline_strategy_id",
        "month_col",
        "loan_amount_col",
        "overdue_amount_col",
        "drop_nan_labels",
    }
)
_REQUIRED_FIELDS = _INPUT_FIELDS - _OPTIONAL_FIELDS
_OUTPUT_FIELDS = frozenset(
    {
        "schema_version",
        "assessment_id",
        "content_hash",
        "pool_id",
        "revision",
        "snapshot_hash",
        "design_hash",
        "strategy_type",
        "comparison_mode",
        "population_count",
        "labeled_count",
        "nan_labels_excluded",
        "monthly_status",
        "assessment",
        "warnings",
        "artifacts",
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
_TASK_ARTIFACT_ROW_FIELDS = (
    "id",
    "task_id",
    "kind",
    "path",
    "content_hash",
    "origin_tool",
    "provenance_json",
    "created_at",
)
_TASK_ARTIFACT_RECORD_FIELDS = frozenset(
    {
        "id",
        "task_id",
        "kind",
        "path",
        "content_hash",
        "origin_tool",
        "provenance",
        "created_at",
    }
)
_MAX_POOL_IMPACT_ARTIFACT_BYTES = 64 * 1024 * 1024
_POOL_IMPACT_PROVENANCE_FIELDS = frozenset(
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
_BOUNDARY_ERRORS = (
    DataLayerError,
    DataWorkspaceDataError,
    DataWorkspaceDatasetNotFound,
    DatasetContentDriftError,
    StrategyCandidatePoolConflictError,
    StrategyCandidatePoolDataError,
    StrategyCandidatePoolNotFoundError,
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
)


@dataclass(frozen=True)
class _DatasetBinding:
    dataset: Any
    path: Path
    source_path: str
    dataset_id: str
    content_hash: str
    registry_metadata_hash: str
    row_count: int
    columns: tuple[str, ...]
    workspace_revision: int
    workspace_generation: int
    semantic_mapping_hash: str
    target_col: str


@dataclass(frozen=True)
class _BaselineBinding:
    strategy_id: str
    strategy_type: str
    spec: dict[str, Any]
    spec_hash: str


@dataclass(frozen=True)
class StrategyPoolImpactArtifactBinding:
    """Authenticated development-backtest impact and all exact source inputs."""

    task_id: str
    artifact_id: str
    artifact_path: Path
    artifact_content_hash: str
    artifact_provenance: dict[str, Any]
    artifact_provenance_json: str
    assessment: dict[str, Any]
    request: dict[str, Any]
    pool: StrategyCandidatePoolArtifactBinding
    dataset: _DatasetBinding
    sample_design: StrategySampleDesignExecutionBinding
    baseline: _BaselineBinding | None
    stage: str
    validation_status: str
    tasks_root: Path
    db_path: Path
    development: StrategyPoolDevelopmentExecutionBinding | None = None


def run_measure_pool_impact(inputs, ctx, runtime) -> dict[str, Any]:
    """Measure and atomically publish one exact current Pool assessment."""

    try:
        request = _validate_inputs(inputs)
        task_id = _text(ctx.task_id, "task_id")
        repository = StrategyCandidatePoolRepository(runtime.settings.db_path)
        from marvis.packs.strategy import pool_tools

        pool_binding = pool_tools.load_current_strategy_candidate_pool_artifact(
            runtime,
            task_id=task_id,
            strategy_type=request["strategy_type"],
            expected_pool_revision=request["expected_pool_revision"],
            expected_pool_snapshot_hash=request["expected_pool_snapshot_hash"],
        )
        development = pool_tools.bind_strategy_pool_development_execution(
            runtime,
            pool_binding,
        )
        pool = pool_binding.pool
        if not pool["entries"]:
            raise StrategyError("cannot measure an empty Strategy Pool")
        selected = pool_binding.compiled_design
        if selected["requirements"]:
            raise StrategyError(
                "Pool impact cannot execute unresolved candidate requirements"
            )
        sample = _pool_development_sample_binding(development)
        dataset = _load_dataset_binding(
            runtime,
            request=request,
            task_id=task_id,
            sample=sample,
        )
        sample_design = development.sample_design
        requested_sample_ref = StrategySampleDesignRef.from_value(
            request["sample_design_ref"]
        )
        if requested_sample_ref != sample_design.reference:
            load_strategy_sample_design_execution_binding(
                runtime,
                task_id=task_id,
                sample_design_ref=request["sample_design_ref"],
                dataset_id=dataset.dataset_id,
                dataset_content_hash=dataset.content_hash,
                workspace_revision=dataset.workspace_revision,
                workspace_generation=dataset.workspace_generation,
                semantic_mapping_hash=dataset.semantic_mapping_hash,
                target_col=dataset.target_col,
                drop_nan_labels=request["drop_nan_labels"],
                month_col=request.get("month_col"),
                loan_amount_col=request.get("loan_amount_col"),
                overdue_amount_col=request.get("overdue_amount_col"),
            )
            raise StrategyError(
                "Strategy Pool candidate sample-design reference does not match "
                "the requested development sample"
            )
        request = _resolve_sample_design_optional_bindings(request, sample_design)
        _require_pool_development_request(
            development,
            request=request,
            task_id=task_id,
        )
        baseline = _load_baseline(
            runtime,
            request=request,
            task_id=task_id,
        )
        frame = _read_frame(
            runtime,
            dataset=dataset,
            sample_design=sample_design,
            strategy_spec=selected["strategy_spec"],
            baseline_spec=None if baseline is None else baseline.spec,
            request=request,
        )
        frame = bind_strategy_development_frame(frame, binding=sample_design)
        nan_labels_excluded = require_labels_confirmed(
            frame,
            dataset.target_col,
            drop_nan_labels=request["drop_nan_labels"],
            scope="Strategy Pool impact source dataset",
        )
        assessment = build_strategy_pool_impact_assessment(
            pool=pool,
            frame=frame,
            sample_binding=sample,
            sample_design_ref=sample_design.to_ref_dict(),
            target_col=dataset.target_col,
            target_bad_value=1,
            month_col=request.get("month_col"),
            loan_amount_col=request.get("loan_amount_col"),
            overdue_amount_col=request.get("overdue_amount_col"),
            comparison_mode=request["comparison_mode"],
            baseline_spec=None if baseline is None else baseline.spec,
            baseline_binding=(
                None
                if baseline is None
                else {
                    "strategy_id": baseline.strategy_id,
                    "strategy_type": baseline.strategy_type,
                    "spec_hash": baseline.spec_hash,
                }
            ),
        )
        if sha256_file(dataset.path) != dataset.content_hash:
            raise StrategyError(
                "source dataset changed while Pool impact was being measured"
            )
        revalidated_sample_design = (
            revalidate_strategy_sample_design_execution_binding(
                runtime,
                sample_design,
            )
        )
        if revalidated_sample_design != sample_design:
            raise StrategyError(
                "strategy sample-design changed while Pool impact was measured"
            )
        return _persist_assessment(
            runtime,
            repository=repository,
            request=request,
            task_id=task_id,
            pool=pool,
            development=development,
            dataset=dataset,
            sample_design=sample_design,
            baseline=baseline,
            assessment=assessment,
            nan_labels_excluded=nan_labels_excluded,
        )
    except StrategyError:
        raise
    except NanLabelNotConfirmedError:
        # Preserve the structured confirmation contract consumed by the
        # Agent/Runner instead of flattening it to an ordinary StrategyError.
        raise
    except _BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc


def validate_measure_pool_impact_tool_output(value: object) -> dict[str, Any]:
    """Fail closed when cached Tool output drifts from its canonical assessment.

    This validates the self-contained Tool envelope and its declared artifact-byte
    hash. Authenticating persisted bytes still belongs to TaskArtifact registry
    lookup, whose trusted ``content_hash`` is outside this cached output.
    """

    if not isinstance(value, Mapping) or set(value) != _OUTPUT_FIELDS:
        raise StrategyError("measure_pool_impact output envelope is invalid")
    try:
        normalized = json.loads(
            json.dumps(
                dict(value),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError, RecursionError, json.JSONDecodeError) as exc:
        raise StrategyError("measure_pool_impact output must be canonical JSON") from exc

    try:
        assessment = validate_strategy_pool_impact_assessment(
            normalized["assessment"]
        )
    except RecursionError as exc:
        raise StrategyError(
            "measure_pool_impact output must be canonical JSON"
        ) from exc
    identity = assessment["identity"]
    population = assessment["population"]
    expected = {
        "schema_version": POOL_IMPACT_TOOL_SCHEMA_VERSION,
        "assessment_id": assessment["assessment_id"],
        "content_hash": assessment["content_hash"],
        "pool_id": identity["pool_id"],
        "revision": identity["revision"],
        "snapshot_hash": identity["snapshot_hash"],
        "design_hash": identity["design_hash"],
        "strategy_type": identity["strategy_type"],
        "comparison_mode": assessment["bindings"]["comparison_mode"],
        "population_count": population["population_count"],
        "labeled_count": population["labelled_count"],
        "monthly_status": assessment["monthly"]["status"],
    }
    for field, expected_value in expected.items():
        if normalized[field] != expected_value:
            raise StrategyError(f"measure_pool_impact output {field} drifted")

    excluded = _non_negative_int(
        normalized["nan_labels_excluded"], "nan_labels_excluded"
    )
    if excluded != population["unlabelled_count"]:
        raise StrategyError(
            "measure_pool_impact output nan_labels_excluded drifted"
        )
    expected_warnings = [
        str(flag["message"])
        for flag in assessment["red_flags"]
        if flag.get("level") in {"amber", "red"}
    ]
    if normalized["warnings"] != expected_warnings:
        raise StrategyError("measure_pool_impact output warnings drifted")
    for field in ("not_created_strategy", "not_adopted", "not_deployed"):
        if normalized[field] is not True:
            raise StrategyError(f"measure_pool_impact output {field} must be true")

    artifacts = normalized["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise StrategyError("measure_pool_impact output needs one canonical artifact")
    artifact = artifacts[0]
    if not isinstance(artifact, dict) or set(artifact) != _OUTPUT_ARTIFACT_FIELDS:
        raise StrategyError("measure_pool_impact output artifact is invalid")
    artifact_id = _text(artifact["artifact_id"], "artifact_id")
    expected_artifact_hash = hashlib.sha256(
        canonical_strategy_pool_impact_json(assessment).encode("utf-8")
    ).hexdigest()
    expected_download_url = (
        f"/api/tasks/{quote(identity['task_id'], safe='')}"
        f"/task-artifacts/{quote(artifact_id, safe='')}/download"
    )
    artifact_expected = {
        "kind": POOL_IMPACT_ARTIFACT_KIND,
        "format": "json",
        "filename": f"{assessment['assessment_id']}.json",
        "content_hash": expected_artifact_hash,
        "download_url": expected_download_url,
    }
    if artifact["artifact_id"] != artifact_id:
        raise StrategyError("measure_pool_impact artifact_id is not canonical")
    for field, expected_value in artifact_expected.items():
        if artifact[field] != expected_value:
            raise StrategyError(f"measure_pool_impact artifact {field} drifted")
    normalized["assessment"] = assessment
    return normalized


def load_strategy_pool_impact_artifact(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_artifact_content_hash: str,
    expected_assessment_id: str | None = None,
    expected_assessment_content_hash: str | None = None,
) -> StrategyPoolImpactArtifactBinding:
    """Load one authenticated legacy PoolImpact as development backtest only."""

    return _load_strategy_pool_impact_artifact(
        runtime,
        task_id=task_id,
        artifact_id=artifact_id,
        expected_artifact_content_hash=expected_artifact_content_hash,
        expected_assessment_id=expected_assessment_id,
        expected_assessment_content_hash=expected_assessment_content_hash,
        require_current_sources=True,
    )


def load_historical_strategy_pool_impact_artifact(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_artifact_content_hash: str,
    expected_assessment_id: str | None = None,
    expected_assessment_content_hash: str | None = None,
) -> StrategyPoolImpactArtifactBinding:
    """Authenticate immutable PoolImpact without requiring source heads."""

    return _load_strategy_pool_impact_artifact(
        runtime,
        task_id=task_id,
        artifact_id=artifact_id,
        expected_artifact_content_hash=expected_artifact_content_hash,
        expected_assessment_id=expected_assessment_id,
        expected_assessment_content_hash=expected_assessment_content_hash,
        require_current_sources=False,
    )


def _load_strategy_pool_impact_artifact(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_artifact_content_hash: str,
    expected_assessment_id: str | None,
    expected_assessment_content_hash: str | None,
    require_current_sources: bool,
) -> StrategyPoolImpactArtifactBinding:
    try:
        from marvis.packs.strategy import pool_tools

        normalized_task = _text(task_id, "task_id")
        normalized_artifact_id = _hash(artifact_id, "artifact_id")
        artifact_hash = _hash(
            expected_artifact_content_hash,
            "expected_artifact_content_hash",
        )
        assessment_id = (
            None
            if expected_assessment_id is None
            else _text(expected_assessment_id, "expected_assessment_id")
        )
        assessment_content_hash = (
            None
            if expected_assessment_content_hash is None
            else _hash(
                expected_assessment_content_hash,
                "expected_assessment_content_hash",
            )
        )
        tasks_root = Path(runtime.settings.tasks_dir).absolute()
        db_path = Path(runtime.settings.db_path).absolute()
        record = _load_impact_artifact_record(
            runtime,
            task_id=normalized_task,
            artifact_id=normalized_artifact_id,
            expected_content_hash=artifact_hash,
        )
        provenance = _validate_impact_provenance(record["provenance"])
        path = Path(str(record["path"]))
        raw = _read_impact_artifact(
            path,
            root=tasks_root,
            expected_content_hash=artifact_hash,
        )
        assessment = _impact_assessment_from_bytes(raw)
        canonical = canonical_strategy_pool_impact_json(assessment).encode("utf-8")
        if raw != canonical:
            raise StrategyError("Pool impact artifact bytes are not canonical")
        if assessment_id is not None and assessment["assessment_id"] != assessment_id:
            raise StrategyError("Pool impact assessment id changed")
        if assessment_content_hash is not None and not hmac.compare_digest(
            assessment["content_hash"],
            assessment_content_hash,
        ):
            raise StrategyError("Pool impact assessment content hash changed")
        expected_path = (
            tasks_root
            / normalized_task
            / "strategy_pool_impacts"
            / f"{assessment['assessment_id']}.json"
        )
        if path != expected_path:
            raise StrategyError("Pool impact artifact path is not canonical")

        identity = assessment["identity"]
        if require_current_sources:
            pool = pool_tools.load_current_strategy_candidate_pool_artifact(
                runtime,
                task_id=normalized_task,
                strategy_type=identity["strategy_type"],
                expected_pool_revision=identity["revision"],
                expected_pool_snapshot_hash=identity["snapshot_hash"],
            )
            development = pool_tools.bind_strategy_pool_development_execution(
                runtime,
                pool,
            )
        else:
            pool = _load_historical_impact_pool(
                runtime,
                task_id=normalized_task,
                identity=identity,
                provenance=provenance,
            )
            development = (
                pool_tools.bind_strategy_pool_revision_development_execution(
                    runtime,
                    pool,
                )
            )
        designed_drop_nan = development.sample_design.drop_nan_labels
        request = _impact_request_from_provenance(
            provenance,
            strategy_type=identity["strategy_type"],
            drop_nan_labels=designed_drop_nan,
        )
        _require_pool_development_request(
            development,
            request=request,
            task_id=normalized_task,
        )
        sample = _pool_development_sample_binding(development)
        dataset = _load_dataset_binding(
            runtime,
            request=request,
            task_id=normalized_task,
            sample=sample,
            require_current_workspace=require_current_sources,
        )
        sample_design = development.sample_design
        request = _resolve_sample_design_optional_bindings(request, sample_design)
        _require_pool_development_request(
            development,
            request=request,
            task_id=normalized_task,
        )
        baseline = _load_baseline(
            runtime,
            request=request,
            task_id=normalized_task,
        )
        binding = StrategyPoolImpactArtifactBinding(
            task_id=normalized_task,
            artifact_id=normalized_artifact_id,
            artifact_path=path,
            artifact_content_hash=artifact_hash,
            artifact_provenance=provenance,
            artifact_provenance_json=_canonical_json(provenance),
            assessment=assessment,
            request=request,
            pool=pool,
            dataset=dataset,
            sample_design=sample_design,
            baseline=baseline,
            stage="development_backtest",
            validation_status="unvalidated",
            tasks_root=tasks_root,
            db_path=db_path,
            development=development,
        )
        _require_impact_binding_relationships(binding)
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            _require_strategy_pool_impact_artifact_binding_on_connection(
                conn,
                binding,
                require_current_sources=require_current_sources,
            )
            conn.commit()
        return binding
    except StrategyError:
        raise
    except _BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc


def require_strategy_pool_impact_artifact_binding_on_connection(
    conn,
    binding: StrategyPoolImpactArtifactBinding,
) -> None:
    """Re-authenticate PoolImpact while a downstream writer owns the lock."""

    _require_strategy_pool_impact_artifact_binding_on_connection(
        conn,
        binding,
        require_current_sources=True,
    )


def require_historical_strategy_pool_impact_artifact_binding_on_connection(
    conn,
    binding: StrategyPoolImpactArtifactBinding,
) -> None:
    """Re-authenticate immutable PoolImpact without requiring source heads."""

    _require_strategy_pool_impact_artifact_binding_on_connection(
        conn,
        binding,
        require_current_sources=False,
    )


def _require_strategy_pool_impact_artifact_binding_on_connection(
    conn,
    binding: StrategyPoolImpactArtifactBinding,
    *,
    require_current_sources: bool,
) -> None:
    from marvis.packs.strategy import pool_tools

    if not isinstance(binding, StrategyPoolImpactArtifactBinding):
        raise StrategyError("Pool impact artifact binding is invalid")
    _require_binding_connection(
        conn,
        db_path=binding.db_path,
        name="Pool impact",
    )
    if binding.tasks_root != binding.tasks_root.absolute():
        raise StrategyError("Pool impact task root changed")
    _require_impact_binding_relationships(binding)
    if binding.development is None:
        raise StrategyError("Pool impact development binding is missing")
    if require_current_sources:
        pool_tools.require_strategy_pool_development_execution_binding_on_connection(
            conn,
            binding.development,
        )
    else:
        pool_tools.require_strategy_pool_revision_development_execution_binding_on_connection(
            conn,
            binding.development,
        )
    _require_dataset_and_workspace_on_connection(
        conn,
        request=binding.request,
        task_id=binding.task_id,
        dataset=binding.dataset,
        require_current_workspace=require_current_sources,
    )
    _require_baseline_on_connection(
        conn,
        request=binding.request,
        task_id=binding.task_id,
        baseline=binding.baseline,
    )
    _require_impact_artifact_on_connection(conn, binding)
    canonical = canonical_strategy_pool_impact_json(binding.assessment).encode("utf-8")
    _verify_file(
        binding.artifact_path,
        root=binding.tasks_root,
        canonical=canonical,
        content_hash=binding.artifact_content_hash,
    )


def _load_historical_impact_pool(
    runtime,
    *,
    task_id: str,
    identity: Mapping[str, Any],
    provenance: Mapping[str, Any],
):
    from marvis.packs.strategy import pool_tools

    strategy_type = _text(identity["strategy_type"], "assessment strategy_type")
    revision_id = _text(provenance["pool_revision_id"], "pool revision id")
    if (
        identity["task_id"] != task_id
        or identity["revision_id"] != revision_id
        or provenance["task_id"] != task_id
        or provenance["pool_id"] != identity["pool_id"]
        or provenance["pool_revision"] != identity["revision"]
        or provenance["pool_snapshot_hash"] != identity["snapshot_hash"]
        or provenance["design_hash"] != identity["design_hash"]
        or provenance["strategy_spec_hash"] != identity["strategy_spec_hash"]
    ):
        raise StrategyError(
            "Pool impact historical Pool provenance identity changed"
        )
    repository = StrategyCandidatePoolRepository(runtime.settings.db_path)
    historical = repository.get_revision_by_id(
        task_id,
        strategy_type,
        revision_id,
    )
    if historical is None:
        raise StrategyError("Pool impact historical Pool revision not found")
    pool = validate_strategy_pool(historical)
    compiled = compile_strategy_pool(pool)
    if (
        pool["pool_id"] != identity["pool_id"]
        or pool["task_id"] != task_id
        or pool["strategy_type"] != strategy_type
        or pool["revision"] != identity["revision"]
        or pool["revision_id"] != revision_id
        or pool["snapshot_hash"] != identity["snapshot_hash"]
        or compiled["design_hash"] != identity["design_hash"]
        or strategy_spec_hash(compiled["strategy_spec"])
        != identity["strategy_spec_hash"]
    ):
        raise StrategyError("Pool impact historical Pool revision changed")
    artifact = pool_tools._normalize_source_record(
        pool_tools._load_pool_artifact(
            runtime,
            task_id=task_id,
            snapshot=pool,
        )
    )
    return pool_tools.load_strategy_candidate_pool_revision_artifact(
        runtime,
        task_id=task_id,
        strategy_type=strategy_type,
        revision_id=revision_id,
        artifact_id=artifact.artifact_id,
        expected_artifact_content_hash=artifact.content_hash,
    )


def _load_impact_artifact_record(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_content_hash: str,
) -> dict[str, Any]:
    record = runtime.task_artifacts.get_for_task(task_id, artifact_id)
    if record is None:
        raise StrategyError("Pool impact artifact not found")
    if not isinstance(record, Mapping) or set(record) != _TASK_ARTIFACT_RECORD_FIELDS:
        raise StrategyError("Pool impact artifact registry row is invalid")
    if (
        record["id"] != artifact_id
        or record["task_id"] != task_id
        or record["kind"] != POOL_IMPACT_ARTIFACT_KIND
        or record["origin_tool"] != POOL_IMPACT_ORIGIN_TOOL
        or not hmac.compare_digest(
            str(record["content_hash"]),
            expected_content_hash,
        )
    ):
        raise StrategyError("Pool impact artifact registry binding changed")
    return dict(record)


def _validate_impact_provenance(value: object) -> dict[str, Any]:
    provenance = _json_object(value, "Pool impact artifact provenance")
    if set(provenance) != _POOL_IMPACT_PROVENANCE_FIELDS:
        raise StrategyError("Pool impact artifact provenance fields are invalid")
    expected_text = {
        "schema_version": POOL_IMPACT_ARTIFACT_SCHEMA_VERSION,
        "producer_version": STRATEGY_POOL_IMPACT_PRODUCER_VERSION,
        "sample_partition": "development",
    }
    for field, expected in expected_text.items():
        if provenance[field] != expected:
            raise StrategyError(f"Pool impact artifact provenance {field} is invalid")
    for field in (
        "task_id",
        "assessment_id",
        "pool_id",
        "pool_revision_id",
        "dataset_id",
        "target_col",
    ):
        if provenance[field] != _text(provenance[field], f"provenance.{field}"):
            raise StrategyError(f"Pool impact artifact provenance {field} is invalid")
    for field in (
        "assessment_content_hash",
        "pool_snapshot_hash",
        "design_hash",
        "strategy_spec_hash",
        "dataset_content_hash",
        "registry_metadata_hash",
        "semantic_mapping_hash",
    ):
        _hash(provenance[field], f"provenance.{field}")
    _positive_int(provenance["pool_revision"], "provenance.pool_revision")
    _non_negative_int(
        provenance["workspace_revision"],
        "provenance.workspace_revision",
    )
    _non_negative_int(
        provenance["workspace_generation"],
        "provenance.workspace_generation",
    )
    sample_ref = StrategySampleDesignRef.from_value(
        provenance["sample_design_ref"]
    )
    provenance["sample_design_ref"] = sample_ref.to_ref_dict()
    for field in ("month_col", "loan_amount_col", "overdue_amount_col"):
        value = provenance[field]
        if value is not None and value != _text(value, f"provenance.{field}"):
            raise StrategyError(f"Pool impact artifact provenance {field} is invalid")
    source_bad = provenance["source_target_bad_value"]
    normalized_bad = provenance["normalized_target_bad_value"]
    if (
        isinstance(source_bad, bool)
        or not isinstance(source_bad, int)
        or source_bad not in {0, 1}
        or isinstance(normalized_bad, bool)
        or not isinstance(normalized_bad, int)
        or normalized_bad != 1
    ):
        raise StrategyError("Pool impact artifact target polarity is invalid")
    comparison_mode = provenance["comparison_mode"]
    if comparison_mode not in {"absolute", "vs_baseline"}:
        raise StrategyError("Pool impact artifact comparison_mode is invalid")
    baseline_id = provenance["baseline_strategy_id"]
    baseline_hash = provenance["baseline_spec_hash"]
    if comparison_mode == "absolute":
        if baseline_id is not None or baseline_hash is not None:
            raise StrategyError("absolute Pool impact provenance forbids baseline")
    else:
        if baseline_id != _text(baseline_id, "provenance.baseline_strategy_id"):
            raise StrategyError("Pool impact baseline strategy id is invalid")
        _hash(baseline_hash, "provenance.baseline_spec_hash")
    return provenance


def _impact_request_from_provenance(
    provenance: Mapping[str, Any],
    *,
    strategy_type: str,
    drop_nan_labels: bool,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "strategy_type": strategy_type,
        "expected_pool_revision": provenance["pool_revision"],
        "expected_pool_snapshot_hash": provenance["pool_snapshot_hash"],
        "dataset_id": provenance["dataset_id"],
        "expected_dataset_content_hash": provenance["dataset_content_hash"],
        "workspace_revision": provenance["workspace_revision"],
        "workspace_generation": provenance["workspace_generation"],
        "semantic_mapping_hash": provenance["semantic_mapping_hash"],
        "target_col": provenance["target_col"],
        "sample_design_ref": provenance["sample_design_ref"],
        "comparison_mode": provenance["comparison_mode"],
        "drop_nan_labels": drop_nan_labels,
    }
    for field in ("month_col", "loan_amount_col", "overdue_amount_col"):
        value = provenance[field]
        if value is not None:
            request[field] = value
    if provenance["baseline_strategy_id"] is not None:
        request["baseline_strategy_id"] = provenance["baseline_strategy_id"]
    return _validate_inputs(request)


def _require_impact_binding_relationships(
    binding: StrategyPoolImpactArtifactBinding,
) -> None:
    from marvis.packs.strategy import pool_tools

    task_id = _text(binding.task_id, "Pool impact binding.task_id")
    _hash(binding.artifact_id, "Pool impact binding.artifact_id")
    artifact_hash = _hash(
        binding.artifact_content_hash,
        "Pool impact binding.artifact_content_hash",
    )
    if not isinstance(
        binding.pool,
        pool_tools.StrategyCandidatePoolArtifactBinding,
    ):
        raise StrategyError("Pool impact Pool binding is invalid")
    development = binding.development
    if not isinstance(
        development,
        pool_tools.StrategyPoolDevelopmentExecutionBinding,
    ):
        raise StrategyError("Pool impact development binding is invalid")
    if not isinstance(binding.dataset, _DatasetBinding) or not isinstance(
        binding.sample_design,
        StrategySampleDesignExecutionBinding,
    ):
        raise StrategyError("Pool impact dataset/sample binding is invalid")
    if binding.baseline is not None and not isinstance(
        binding.baseline,
        _BaselineBinding,
    ):
        raise StrategyError("Pool impact baseline binding is invalid")
    pool_tools._require_regular_dataset_path(
        binding.dataset.path,
        root=binding.pool.datasets_root,
    )
    try:
        expected_dataset_path = (
            binding.pool.datasets_root / binding.dataset.source_path
        ).resolve(strict=True)
    except OSError as exc:
        raise StrategyError("Pool impact dataset binding changed") from exc
    if binding.dataset.path != expected_dataset_path:
        raise StrategyError("Pool impact dataset path changed")
    registered_dataset = binding.dataset.dataset
    if (
        binding.pool.task_id != task_id
        or development.task_id != task_id
        or development.pool != binding.pool
        or development.sample_design != binding.sample_design
        or binding.sample_design.task_id != task_id
        or getattr(registered_dataset, "task_id", None) != task_id
    ):
        raise StrategyError("Pool impact source belongs to another task")
    if binding.stage != "development_backtest":
        raise StrategyError("Pool impact stage must remain development_backtest")
    if binding.validation_status != "unvalidated":
        raise StrategyError("Pool impact validation status must remain unvalidated")

    provenance = _validate_impact_provenance(binding.artifact_provenance)
    provenance_json = _canonical_json(provenance)
    if (
        provenance != binding.artifact_provenance
        or provenance_json != binding.artifact_provenance_json
    ):
        raise StrategyError("Pool impact artifact provenance binding changed")
    expected_request = _impact_request_from_provenance(
        provenance,
        strategy_type=binding.pool.strategy_type,
        drop_nan_labels=binding.sample_design.drop_nan_labels,
    )
    expected_request = _resolve_sample_design_optional_bindings(
        expected_request,
        binding.sample_design,
    )
    if expected_request != binding.request:
        raise StrategyError("Pool impact exact input binding changed")

    assessment = validate_strategy_pool_impact_assessment(binding.assessment)
    if assessment != binding.assessment:
        raise StrategyError("Pool impact canonical assessment changed")
    canonical = canonical_strategy_pool_impact_json(assessment).encode("utf-8")
    if not hmac.compare_digest(
        hashlib.sha256(canonical).hexdigest(),
        artifact_hash,
    ):
        raise StrategyError("Pool impact artifact hash changed")
    expected_path = (
        binding.tasks_root
        / task_id
        / "strategy_pool_impacts"
        / f"{assessment['assessment_id']}.json"
    )
    if binding.artifact_path != expected_path:
        raise StrategyError("Pool impact artifact path changed")

    pool = validate_strategy_pool(binding.pool.pool)
    compiled = compile_strategy_pool(pool)
    if compiled != binding.pool.compiled_design:
        raise StrategyError("Pool impact compiled Pool design changed")
    if compiled["requirements"]:
        raise StrategyError("Pool impact cannot bind unresolved candidate requirements")
    identity = assessment["identity"]
    identity_expected = {
        "pool_id": pool["pool_id"],
        "task_id": task_id,
        "strategy_type": pool["strategy_type"],
        "revision": pool["revision"],
        "revision_id": pool["revision_id"],
        "snapshot_hash": pool["snapshot_hash"],
        "design_hash": compiled["design_hash"],
        "strategy_spec_hash": strategy_spec_hash(compiled["strategy_spec"]),
    }
    if identity != identity_expected:
        raise StrategyError("Pool impact Pool identity changed")

    dataset = binding.dataset
    public_dataset = development.dataset
    if (
        getattr(registered_dataset, "id", None) != dataset.dataset_id
        or getattr(registered_dataset, "source_path", None) != dataset.source_path
        or getattr(registered_dataset, "content_hash", None) != dataset.content_hash
        or getattr(registered_dataset, "row_count", None) != dataset.row_count
        or tuple(
            str(column.name)
            for column in getattr(registered_dataset, "columns", ())
        )
        != dataset.columns
        or public_dataset.task_id != task_id
        or public_dataset.dataset_id != dataset.dataset_id
        or public_dataset.source_path != dataset.source_path
        or public_dataset.path != dataset.path
        or public_dataset.content_hash != dataset.content_hash
        or public_dataset.registry_metadata_hash != dataset.registry_metadata_hash
        or public_dataset.row_count != dataset.row_count
        or public_dataset.columns != dataset.columns
        or development.target_col != dataset.target_col
    ):
        raise StrategyError("Pool impact dataset binding changed")
    dataset_expected = {
        "dataset_id": dataset.dataset_id,
        "dataset_content_hash": dataset.content_hash,
        "registry_metadata_hash": dataset.registry_metadata_hash,
        "workspace_revision": dataset.workspace_revision,
        "workspace_generation": dataset.workspace_generation,
        "semantic_mapping_hash": dataset.semantic_mapping_hash,
        "target_col": dataset.target_col,
    }
    if any(provenance[field] != value for field, value in dataset_expected.items()):
        raise StrategyError("Pool impact dataset/workspace provenance changed")
    development_identity = development.evidence_identity
    if (
        development_identity.get("dataset_id") != dataset.dataset_id
        or development_identity.get("dataset_content_hash") != dataset.content_hash
        or development_identity.get("workspace_revision")
        != dataset.workspace_revision
        or development_identity.get("workspace_generation")
        != dataset.workspace_generation
        or development_identity.get("semantic_mapping_hash")
        != dataset.semantic_mapping_hash
    ):
        raise StrategyError("Pool impact development evidence identity changed")
    sample_binding = _pool_development_sample_binding(development)
    assessment_bindings = assessment["bindings"]
    if (
        assessment_bindings["sample"] != sample_binding
        or assessment_bindings["sample_design_ref"]
        != binding.sample_design.to_ref_dict()
        or assessment_bindings["target_col"] != dataset.target_col
        or assessment_bindings["target_bad_value"] != 1
        or assessment_bindings["comparison_mode"] != binding.request["comparison_mode"]
    ):
        raise StrategyError("Pool impact assessment source binding changed")
    optional_columns = {
        "month_col": binding.sample_design.month_col,
        "loan_amount_col": binding.sample_design.loan_amount_col,
        "overdue_amount_col": binding.sample_design.overdue_amount_col,
    }
    if any(
        provenance[field] != value
        or assessment_bindings[field] != value
        or binding.request[field] != value
        for field, value in optional_columns.items()
    ):
        raise StrategyError("Pool impact optional column binding changed")
    if (
        provenance["task_id"] != task_id
        or provenance["assessment_id"] != assessment["assessment_id"]
        or provenance["assessment_content_hash"] != assessment["content_hash"]
        or provenance["pool_id"] != pool["pool_id"]
        or provenance["pool_revision"] != pool["revision"]
        or provenance["pool_revision_id"] != pool["revision_id"]
        or provenance["pool_snapshot_hash"] != pool["snapshot_hash"]
        or provenance["design_hash"] != compiled["design_hash"]
        or provenance["strategy_spec_hash"] != identity["strategy_spec_hash"]
        or provenance["sample_design_ref"] != binding.sample_design.to_ref_dict()
        or provenance["source_target_bad_value"]
        != binding.sample_design.target_bad_value
        or provenance["normalized_target_bad_value"] != 1
    ):
        raise StrategyError("Pool impact artifact provenance source binding changed")
    _require_pool_development_request(
        development,
        request=binding.request,
        task_id=task_id,
    )
    _require_waterfall_pool_binding(assessment, pool=pool, compiled=compiled)
    _require_assessment_baseline_binding(
        assessment,
        request=binding.request,
        baseline=binding.baseline,
        provenance=provenance,
    )


def _require_waterfall_pool_binding(
    assessment: Mapping[str, Any],
    *,
    pool: Mapping[str, Any],
    compiled: Mapping[str, Any],
) -> None:
    rules = compiled["strategy_spec"]["rules"]
    if len(assessment["waterfall"]) != len(pool["entries"]) or len(rules) != len(
        pool["entries"]
    ):
        raise StrategyError("Pool impact waterfall no longer matches the Pool")
    sample_ref = assessment["bindings"]["sample_design_ref"]
    for row, entry, rule in zip(
        assessment["waterfall"],
        pool["entries"],
        rules,
        strict=True,
    ):
        source = entry["source"]
        expected_source = {
            "artifact_id": source["artifact_id"],
            "artifact_content_hash": source["artifact_content_hash"],
            "asset_id": source["asset_id"],
            "asset_hash": source["asset_hash"],
            "fragment_id": source["fragment_id"],
            "sample_design_ref": sample_ref,
        }
        if (
            row["position"] != entry["position"] + 1
            or row["entry_id"] != entry["entry_id"]
            or row["rule_id"] != entry["rule_id"]
            or row["source_ref"] != expected_source
            or row["action"] != entry["action"]
            or rule["rule_id"] != entry["rule_id"]
            or rule["action"] != entry["action"]
        ):
            raise StrategyError("Pool impact waterfall source binding changed")


def _require_assessment_baseline_binding(
    assessment: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    baseline: _BaselineBinding | None,
    provenance: Mapping[str, Any],
) -> None:
    if baseline is None:
        if (
            request["comparison_mode"] != "absolute"
            or provenance["baseline_strategy_id"] is not None
            or provenance["baseline_spec_hash"] is not None
        ):
            raise StrategyError("Pool impact baseline binding changed")
        return
    expected = {
        "strategy_id": baseline.strategy_id,
        "strategy_type": baseline.strategy_type,
        "spec_hash": baseline.spec_hash,
    }
    if (
        request.get("baseline_strategy_id") != baseline.strategy_id
        or provenance["baseline_strategy_id"] != baseline.strategy_id
        or provenance["baseline_spec_hash"] != baseline.spec_hash
        or assessment["baseline"]["binding"] != expected
    ):
        raise StrategyError("Pool impact baseline binding changed")


def _require_impact_artifact_on_connection(
    conn,
    binding: StrategyPoolImpactArtifactBinding,
) -> None:
    row = conn.execute(
        """
        SELECT id, task_id, kind, path, content_hash, origin_tool,
               provenance_json
          FROM task_artifacts
         WHERE task_id = ? AND id = ?
        """,
        (binding.task_id, binding.artifact_id),
    ).fetchone()
    if row is None:
        raise StrategyError("Pool impact artifact is no longer registered")
    if (
        str(row["id"]) != binding.artifact_id
        or str(row["task_id"]) != binding.task_id
        or str(row["kind"]) != POOL_IMPACT_ARTIFACT_KIND
        or str(row["path"]) != str(binding.artifact_path)
        or not hmac.compare_digest(
            str(row["content_hash"]),
            binding.artifact_content_hash,
        )
        or str(row["origin_tool"]) != POOL_IMPACT_ORIGIN_TOOL
        or str(row["provenance_json"]) != binding.artifact_provenance_json
    ):
        raise StrategyError("Pool impact artifact registry binding changed")


def _read_impact_artifact(
    path: Path,
    *,
    root: Path,
    expected_content_hash: str,
) -> bytes:
    return _read_regular_nofollow(
        path,
        root=root,
        expected_content_hash=expected_content_hash,
    )


def _impact_assessment_from_bytes(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
        return validate_strategy_pool_impact_assessment(value)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        RecursionError,
    ) as exc:
        raise StrategyError("Pool impact artifact JSON is invalid") from exc


def _validate_inputs(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StrategyError("measure_pool_impact inputs must be an object")
    unexpected = sorted(set(value) - _INPUT_FIELDS)
    if unexpected:
        raise StrategyError(
            "unsupported measure_pool_impact inputs: " + ", ".join(unexpected)
        )
    missing = sorted(_REQUIRED_FIELDS - set(value))
    if missing:
        raise StrategyError("missing measure_pool_impact inputs: " + ", ".join(missing))
    request: dict[str, Any] = {
        "strategy_type": _text(value["strategy_type"], "strategy_type"),
        "expected_pool_revision": _positive_int(
            value["expected_pool_revision"], "expected_pool_revision"
        ),
        "expected_pool_snapshot_hash": _hash(
            value["expected_pool_snapshot_hash"], "expected_pool_snapshot_hash"
        ),
        "dataset_id": _text(value["dataset_id"], "dataset_id"),
        "expected_dataset_content_hash": _hash(
            value["expected_dataset_content_hash"],
            "expected_dataset_content_hash",
        ),
        "workspace_revision": _non_negative_int(
            value["workspace_revision"], "workspace_revision"
        ),
        "workspace_generation": _non_negative_int(
            value["workspace_generation"], "workspace_generation"
        ),
        "semantic_mapping_hash": _hash(
            value["semantic_mapping_hash"], "semantic_mapping_hash"
        ),
        "target_col": _text(value["target_col"], "target_col"),
        "sample_design_ref": StrategySampleDesignRef.from_value(
            value["sample_design_ref"]
        ).to_ref_dict(),
        "comparison_mode": _text(value["comparison_mode"], "comparison_mode"),
    }
    if request["strategy_type"] not in {"approval", "reject"}:
        raise StrategyError("Pool impact supports approval/reject only")
    if request["comparison_mode"] not in {"absolute", "vs_baseline"}:
        raise StrategyError("comparison_mode must be absolute or vs_baseline")
    for field in (
        "baseline_strategy_id",
        "month_col",
        "loan_amount_col",
        "overdue_amount_col",
    ):
        raw = value.get(field)
        if raw not in (None, ""):
            request[field] = _text(raw, field)
    drop_nan_labels = value.get("drop_nan_labels", False)
    if not isinstance(drop_nan_labels, bool):
        raise StrategyError("drop_nan_labels must be boolean")
    request["drop_nan_labels"] = drop_nan_labels
    baseline_id = request.get("baseline_strategy_id")
    if request["comparison_mode"] == "vs_baseline" and baseline_id is None:
        raise StrategyError("vs_baseline requires baseline_strategy_id")
    if request["comparison_mode"] == "absolute" and baseline_id is not None:
        raise StrategyError("absolute comparison forbids baseline_strategy_id")
    optional_columns = [
        request[field]
        for field in ("month_col", "loan_amount_col", "overdue_amount_col")
        if field in request
    ]
    if request["target_col"] in optional_columns or len(optional_columns) != len(
        set(optional_columns)
    ):
        raise StrategyError("Pool impact column bindings must be distinct")
    return request


def _require_pool_cas(pool: Mapping[str, Any], request: Mapping[str, Any]) -> None:
    if pool["strategy_type"] != request["strategy_type"]:
        raise StrategyError("Strategy Pool type changed")
    if pool["revision"] != request["expected_pool_revision"] or not hmac.compare_digest(
        pool["snapshot_hash"], request["expected_pool_snapshot_hash"]
    ):
        raise StrategyError("stale strategy candidate pool revision or snapshot hash")


def _require_pool_development_request(
    development,
    *,
    request: Mapping[str, Any],
    task_id: str,
) -> None:
    """Match caller assertions to the Pool boundary's public evidence projection."""

    from marvis.packs.strategy import pool_tools

    if not isinstance(
        development,
        pool_tools.StrategyPoolDevelopmentExecutionBinding,
    ):
        raise StrategyError("Strategy Pool development binding is invalid")
    sample = development.sample_design
    identity = development.evidence_identity
    expected_identity = {
        "dataset_id": request["dataset_id"],
        "dataset_content_hash": request["expected_dataset_content_hash"],
        "workspace_revision": request["workspace_revision"],
        "workspace_generation": request["workspace_generation"],
        "semantic_mapping_hash": request["semantic_mapping_hash"],
    }
    if (
        development.task_id != task_id
        or development.pool.task_id != task_id
        or development.pool.strategy_type != request["strategy_type"]
    ):
        raise StrategyError("Pool development task or strategy binding changed")
    for field, expected in expected_identity.items():
        if identity.get(field) != expected:
            raise StrategyError(f"Pool sample {field} does not match the request")
    if (
        development.dataset.dataset_id != request["dataset_id"]
        or not hmac.compare_digest(
            development.dataset.content_hash,
            request["expected_dataset_content_hash"],
        )
    ):
        raise StrategyError("Pool development dataset binding changed")
    if development.target_col != request["target_col"]:
        raise StrategyError(
            "strategy sample-design target_col does not match Pool impact binding"
        )
    if sample.reference != StrategySampleDesignRef.from_value(
        request["sample_design_ref"]
    ):
        raise StrategyError(
            "Strategy Pool candidate sample-design reference does not match "
            "the requested development sample"
        )
    if sample.drop_nan_labels is not request["drop_nan_labels"]:
        raise StrategyError(
            "strategy sample-design drop_nan_labels does not match Pool impact binding"
        )
    optional_columns = {
        "month_col": sample.month_col,
        "loan_amount_col": sample.loan_amount_col,
        "overdue_amount_col": sample.overdue_amount_col,
    }
    for field, expected in optional_columns.items():
        if field in request and request[field] != expected:
            raise StrategyError(
                f"strategy sample-design {field} does not match Pool impact binding"
            )


def _pool_development_sample_binding(development) -> dict[str, Any]:
    from marvis.packs.strategy import pool_tools

    if not isinstance(
        development,
        pool_tools.StrategyPoolDevelopmentExecutionBinding,
    ):
        raise StrategyError("Strategy Pool development binding is invalid")
    return {
        "task_id": development.task_id,
        **dict(development.evidence_identity),
    }


def _pool_sample_binding(
    pool: Mapping[str, Any], *, task_id: str
) -> dict[str, Any]:
    identities = [entry["source"]["evidence_identity"] for entry in pool["entries"]]
    if not identities or any(identity != identities[0] for identity in identities[1:]):
        raise StrategyError("Strategy Pool entries do not share one sample identity")
    return {"task_id": task_id, **dict(identities[0])}


def _require_pool_measurement_target(
    lineages,
    *,
    expected_target_col: str,
) -> None:
    if not lineages:
        raise StrategyError("Strategy Pool has no candidate lineages")
    targets = [_lineage_target_col(lineage) for lineage in lineages]
    if any(target != targets[0] for target in targets[1:]):
        raise StrategyError(
            "Strategy Pool candidates do not share one measurement target"
        )
    if targets[0] != expected_target_col:
        raise StrategyError(
            "Strategy Pool candidate target does not match the confirmed workspace target"
        )


def _require_pool_sample_design_ref(
    lineages,
    *,
    expected: StrategySampleDesignRef,
) -> None:
    if not lineages:
        raise StrategyError("Strategy Pool has no candidate lineages")
    for lineage in lineages:
        actual = _lineage_sample_design_ref(lineage)
        if actual != expected:
            raise StrategyError(
                "Strategy Pool candidate sample-design reference does not match "
                "the requested development sample"
            )


def _lineage_sample_design_ref(lineage) -> StrategySampleDesignRef:
    candidate = getattr(lineage, "candidate", None)
    if candidate is not None:
        asset = getattr(candidate, "asset", None)
        if (
            not isinstance(asset, Mapping)
            or asset.get("schema_version")
            != VOTING_CANDIDATE_ASSET_SCHEMA_VERSION
        ):
            raise StrategyError(
                "legacy Voting candidate is not bound to a governed sample "
                "design; regenerate it before impact measurement"
            )
        actual = StrategySampleDesignRef.from_value(
            asset.get("sample_design_ref")
        )
        provenance = getattr(candidate, "provenance", None)
        if not isinstance(provenance, Mapping):
            raise StrategyError("Voting candidate sample-design provenance is invalid")
        if StrategySampleDesignRef.from_value(
            provenance.get("sample_design_ref")
        ) != actual:
            raise StrategyError(
                "Voting candidate sample-design asset and provenance disagree"
            )
        parents = getattr(lineage, "parent_lineages", ())
        if not parents:
            raise StrategyError("Voting candidate parent lineage is incomplete")
        for parent in parents:
            if _lineage_sample_design_ref(parent) != actual:
                raise StrategyError(
                    "Voting candidate sample-design reference does not match "
                    "all selected parent lineages"
                )
        return actual

    evidence = getattr(lineage, "evidence", None)
    if isinstance(evidence, Mapping):
        try:
            value = evidence["generation"]["parameters"]["sample_design_ref"]
        except (KeyError, TypeError) as exc:
            raise StrategyError(
                "Strategy Pool candidate is not bound to a governed sample design; "
                "regenerate the candidate from StrategySampleDesign development"
            ) from exc
        return StrategySampleDesignRef.from_value(value)

    tree = getattr(lineage, "tree", None)
    asset = getattr(tree, "asset", None)
    if isinstance(asset, Mapping):
        try:
            value = sample_design_ref_from_automatic_tree_source_refs(
                asset["source_refs"]
            )
        except (KeyError, TypeError, StrategyError) as exc:
            raise StrategyError(
                "automatic-tree Strategy Pool candidate is not bound to exactly "
                "one governed sample design; regenerate it from "
                "StrategySampleDesign development"
            ) from exc
        return StrategySampleDesignRef.from_value(value)

    raise StrategyError(
        "Strategy Pool candidate type is not yet bound to a governed sample "
        "design; regenerate it with a sample-design-aware candidate Tool"
    )


def _resolve_sample_design_optional_bindings(
    request: Mapping[str, Any],
    binding: StrategySampleDesignExecutionBinding,
) -> dict[str, Any]:
    """Resolve optional measurement columns through the sample-design authority.

    Missing fields inherit the designed columns.  A caller-provided non-empty
    value remains a fail-closed equality assertion against that design.
    """

    expected = {
        "month_col": binding.month_col,
        "loan_amount_col": binding.loan_amount_col,
        "overdue_amount_col": binding.overdue_amount_col,
    }
    resolved = dict(request)
    for field, designed in expected.items():
        requested = request.get(field)
        if requested is not None and requested != designed:
            raise StrategyError(
                f"strategy sample-design {field} does not match Pool impact binding"
            )
        resolved[field] = designed
    return resolved


def _lineage_target_col(lineage) -> str:
    candidate = getattr(lineage, "candidate", None)
    if candidate is not None:
        try:
            target = _text(
                candidate.asset["measurement_context"]["target_col"],
                "Voting candidate target_col",
            )
        except (AttributeError, KeyError, TypeError) as exc:
            raise StrategyError("Voting candidate target binding is invalid") from exc
        parent_targets = [
            _lineage_target_col(parent)
            for parent in getattr(lineage, "parent_lineages", ())
        ]
        if not parent_targets or any(item != target for item in parent_targets):
            raise StrategyError(
                "Voting candidate target does not match its parent Pool targets"
            )
        return target
    evidence = getattr(lineage, "evidence", None)
    if isinstance(evidence, Mapping):
        try:
            target = _text(evidence["analysis"]["target"], "candidate target_col")
            generated_target = _text(
                evidence["generation"]["parameters"]["target_col"],
                "candidate generation target_col",
            )
        except (KeyError, TypeError) as exc:
            raise StrategyError("candidate target binding is invalid") from exc
        if target != generated_target:
            raise StrategyError("candidate target binding is inconsistent")
        return target
    tree = getattr(lineage, "tree", None)
    if tree is not None:
        try:
            return _text(
                tree.asset["tree_result"]["training"]["target_col"],
                "automatic-tree target_col",
            )
        except (AttributeError, KeyError, TypeError) as exc:
            raise StrategyError(
                "automatic-tree target binding is invalid"
            ) from exc
    raise StrategyError("unsupported Strategy Pool candidate target binding")


def _load_dataset_binding(
    runtime,
    *,
    request: Mapping[str, Any],
    task_id: str,
    sample: Mapping[str, Any],
    require_current_workspace: bool = True,
) -> _DatasetBinding:
    comparisons = {
        "dataset_id": request["dataset_id"],
        "dataset_content_hash": request["expected_dataset_content_hash"],
        "workspace_revision": request["workspace_revision"],
        "workspace_generation": request["workspace_generation"],
        "semantic_mapping_hash": request["semantic_mapping_hash"],
    }
    for field, actual in comparisons.items():
        if sample[field] != actual:
            raise StrategyError(f"Pool sample {field} does not match the request")
    workspace_revision = request["workspace_revision"]
    workspace_generation = request["workspace_generation"]
    semantic_hash = request["semantic_mapping_hash"]
    if require_current_workspace:
        workspace = DataWorkspaceRepository(
            runtime.settings.db_path
        ).get_or_default(task_id)
        semantic_hash = data_semantic_mapping_hash(workspace.semantic_mapping)
        if (
            workspace.active_dataset_id != request["dataset_id"]
            or workspace.active_dataset_content_hash
            != request["expected_dataset_content_hash"]
            or workspace.revision != request["workspace_revision"]
            or workspace.analysis_generation != request["workspace_generation"]
            or not hmac.compare_digest(
                semantic_hash,
                request["semantic_mapping_hash"],
            )
            or workspace.semantic_mapping.target_col != request["target_col"]
        ):
            raise StrategyError("DataWorkspace binding changed before Pool impact")
        workspace_revision = workspace.revision
        workspace_generation = workspace.analysis_generation
    try:
        dataset = runtime.registry.get(request["dataset_id"])
        path = Path(runtime.registry.resolve_verified_path(request["dataset_id"]))
    except (DatasetContentDriftError, KeyError, OSError, TypeError, ValueError) as exc:
        raise StrategyError("Pool impact source dataset is unavailable or drifted") from exc
    if dataset.task_id != task_id:
        raise StrategyError("Pool impact source dataset belongs to another task")
    content_hash = str(dataset.content_hash or "")
    if not hmac.compare_digest(content_hash, request["expected_dataset_content_hash"]):
        raise StrategyError("Pool impact dataset content hash changed")
    if sha256_file(path) != content_hash:
        raise StrategyError("Pool impact dataset bytes changed")
    columns = tuple(str(column.name) for column in dataset.columns)
    requested_columns = {
        request["target_col"],
        *(request[field] for field in ("month_col", "loan_amount_col", "overdue_amount_col") if field in request),
    }
    missing = sorted(requested_columns - set(columns))
    if missing:
        raise StrategyError("Pool impact dataset is missing columns: " + ", ".join(missing))

    from marvis.packs.strategy.candidate_asset_tools import (
        _registry_metadata_hash_on_connection,
    )

    with runtime.task_artifacts.transaction() as conn:
        registry_hash = _registry_metadata_hash_on_connection(
            conn,
            task_id=task_id,
            dataset_id=dataset.id,
            expected_content_hash=content_hash,
        )
    return _DatasetBinding(
        dataset=dataset,
        path=path,
        source_path=str(dataset.source_path),
        dataset_id=dataset.id,
        content_hash=content_hash,
        registry_metadata_hash=registry_hash,
        row_count=int(dataset.row_count),
        columns=columns,
        workspace_revision=workspace_revision,
        workspace_generation=workspace_generation,
        semantic_mapping_hash=semantic_hash,
        target_col=request["target_col"],
    )


def _load_baseline(
    runtime,
    *,
    request: Mapping[str, Any],
    task_id: str,
) -> _BaselineBinding | None:
    if request["comparison_mode"] == "absolute":
        return None
    strategy_id = request["baseline_strategy_id"]
    strategy = runtime.strategies.get_strategy(strategy_id)
    meta = runtime.strategies.get_strategy_meta(strategy_id)
    spec_hash = runtime.strategies.get_strategy_spec_hash(strategy_id)
    if (
        strategy is None
        or meta is None
        or strategy.spec is None
        or not isinstance(spec_hash, str)
        or meta.get("task_id") != task_id
    ):
        raise StrategyError("baseline strategy is not owned by the current task")
    if (
        strategy.strategy_type != request["strategy_type"]
        or meta.get("strategy_type") != request["strategy_type"]
    ):
        raise StrategyError("baseline strategy type must match the Pool")
    calculated = strategy_spec_hash(strategy.spec)
    if not hmac.compare_digest(spec_hash, calculated):
        raise StrategyError("baseline strategy spec hash is inconsistent")
    return _BaselineBinding(
        strategy_id=strategy.id,
        strategy_type=strategy.strategy_type,
        spec=strategy.spec.to_dict(),
        spec_hash=spec_hash,
    )


def _read_frame(
    runtime,
    *,
    dataset: _DatasetBinding,
    sample_design: StrategySampleDesignExecutionBinding,
    strategy_spec: Mapping[str, Any],
    baseline_spec: Mapping[str, Any] | None,
    request: Mapping[str, Any],
):
    fields = _expression_fields(strategy_spec)
    if baseline_spec is not None:
        fields.update(_expression_fields(baseline_spec))
    fields.add(dataset.target_col)
    if sample_design.split_column is not None:
        fields.add(sample_design.split_column)
    fields.update(
        request[field]
        for field in ("month_col", "loan_amount_col", "overdue_amount_col")
        if request.get(field) is not None
    )
    unknown = sorted(fields - set(dataset.columns))
    if unknown:
        raise StrategyError("Pool rules reference missing columns: " + ", ".join(unknown))
    frame = runtime.backend.read_frame(dataset.path, columns=sorted(fields))
    if len(frame) != dataset.row_count:
        raise StrategyError("Pool impact dataset row count changed")
    if sha256_file(dataset.path) != dataset.content_hash:
        raise StrategyError("Pool impact dataset bytes changed before evaluation")
    return frame


def _expression_fields(value: Any) -> set[str]:
    fields: set[str] = set()
    if isinstance(value, Mapping):
        field = value.get("field")
        if isinstance(field, str):
            fields.add(field)
        for item in value.values():
            fields.update(_expression_fields(item))
    elif isinstance(value, list | tuple):
        for item in value:
            fields.update(_expression_fields(item))
    return fields


def _persist_assessment(
    runtime,
    *,
    repository: StrategyCandidatePoolRepository,
    request: Mapping[str, Any],
    task_id: str,
    pool: Mapping[str, Any],
    development,
    dataset: _DatasetBinding,
    sample_design: StrategySampleDesignExecutionBinding,
    baseline: _BaselineBinding | None,
    assessment: Mapping[str, Any],
    nan_labels_excluded: int,
) -> dict[str, Any]:
    from marvis.packs.strategy import pool_tools

    canonical = canonical_strategy_pool_impact_json(assessment).encode("utf-8")
    artifact_content_hash = hashlib.sha256(canonical).hexdigest()
    identity = assessment["identity"]
    assessment_id = assessment["assessment_id"]
    out_dir = _prepare_output_directory(
        runtime.settings.tasks_dir,
        task_id=task_id,
    )
    final_path = out_dir / f"{assessment_id}.json"
    provenance = {
        "schema_version": POOL_IMPACT_ARTIFACT_SCHEMA_VERSION,
        "producer_version": STRATEGY_POOL_IMPACT_PRODUCER_VERSION,
        "task_id": task_id,
        "assessment_id": assessment_id,
        "assessment_content_hash": assessment["content_hash"],
        "pool_id": identity["pool_id"],
        "pool_revision": identity["revision"],
        "pool_revision_id": identity["revision_id"],
        "pool_snapshot_hash": identity["snapshot_hash"],
        "design_hash": identity["design_hash"],
        "strategy_spec_hash": identity["strategy_spec_hash"],
        "dataset_id": dataset.dataset_id,
        "dataset_content_hash": dataset.content_hash,
        "registry_metadata_hash": dataset.registry_metadata_hash,
        "workspace_revision": dataset.workspace_revision,
        "workspace_generation": dataset.workspace_generation,
        "semantic_mapping_hash": dataset.semantic_mapping_hash,
        "target_col": dataset.target_col,
        "sample_design_ref": sample_design.to_ref_dict(),
        "month_col": sample_design.month_col,
        "loan_amount_col": sample_design.loan_amount_col,
        "overdue_amount_col": sample_design.overdue_amount_col,
        "source_target_bad_value": sample_design.target_bad_value,
        "normalized_target_bad_value": 1,
        "sample_partition": sample_design.reference.partition,
        "comparison_mode": request["comparison_mode"],
        "baseline_strategy_id": None if baseline is None else baseline.strategy_id,
        "baseline_spec_hash": None if baseline is None else baseline.spec_hash,
    }
    uow = ArtifactUnitOfWork()
    staged = uow.stage_file(out_dir, final_path.name)
    try:
        staged.path.write_bytes(canonical)
    except OSError as exc:
        uow.rollback()
        raise StrategyError("Pool impact artifact could not be staged") from exc
    db_committed = False
    rollback_under_lock = False
    reused = False
    record: Mapping[str, Any]
    try:
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                locked = repository.get_current_on_connection(
                    conn, task_id, request["strategy_type"]
                )
                if locked is None:
                    raise StrategyError("strategy candidate pool not found")
                locked = validate_strategy_pool(locked)
                _require_pool_cas(locked, request)
                if locked != pool:
                    raise StrategyError("Strategy Pool changed before impact registration")
                pool_tools.require_strategy_pool_development_execution_binding_on_connection(
                    conn,
                    development,
                )
                _require_dataset_and_workspace_on_connection(
                    conn,
                    request=request,
                    task_id=task_id,
                    dataset=dataset,
                )
                _require_baseline_on_connection(
                    conn,
                    request=request,
                    task_id=task_id,
                    baseline=baseline,
                )
                row = conn.execute(
                    """
                    SELECT id, task_id, kind, path, content_hash, origin_tool,
                           provenance_json, created_at
                      FROM task_artifacts
                     WHERE task_id = ? AND kind = ? AND path = ?
                    """,
                    (task_id, POOL_IMPACT_ARTIFACT_KIND, str(final_path)),
                ).fetchone()
                if row is not None:
                    _require_existing_artifact(
                        row,
                        task_id=task_id,
                        final_path=final_path,
                        canonical=canonical,
                        content_hash=artifact_content_hash,
                        provenance=provenance,
                    )
                    uow.rollback()
                    reused = True
                else:
                    if final_path.exists() or final_path.is_symlink():
                        raise StrategyError(
                            "Pool impact artifact path exists without a registry row"
                        )
                    uow.promote_all()
                    _verify_file(
                        final_path,
                        root=Path(runtime.settings.tasks_dir),
                        canonical=canonical,
                        content_hash=artifact_content_hash,
                    )
                record = runtime.task_artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=POOL_IMPACT_ARTIFACT_KIND,
                    path=str(final_path),
                    content_hash=artifact_content_hash,
                    origin_tool=POOL_IMPACT_ORIGIN_TOOL,
                    provenance=provenance,
                )
                conn.commit()
                db_committed = True
            except Exception:
                rollback_under_lock = True
                uow.rollback()
                raise
        if not reused:
            uow.commit()
    except Exception:
        if not db_committed and not rollback_under_lock:
            uow.rollback()
        raise
    return _tool_output(
        assessment,
        record=record,
        task_id=task_id,
        nan_labels_excluded=nan_labels_excluded,
    )


def _require_dataset_and_workspace_on_connection(
    conn,
    *,
    request: Mapping[str, Any],
    task_id: str,
    dataset: _DatasetBinding,
    require_current_workspace: bool = True,
) -> None:
    from marvis.packs.strategy.candidate_asset_tools import (
        _registry_metadata_hash_on_connection,
        _require_file_content_hash,
    )

    registry_hash = _registry_metadata_hash_on_connection(
        conn,
        task_id=task_id,
        dataset_id=dataset.dataset_id,
        expected_content_hash=dataset.content_hash,
    )
    if not hmac.compare_digest(registry_hash, dataset.registry_metadata_hash):
        raise StrategyError("dataset registry metadata changed before registration")
    row = conn.execute(
        "SELECT source_path FROM datasets WHERE task_id = ? AND id = ?",
        (task_id, dataset.dataset_id),
    ).fetchone()
    if row is None or str(row["source_path"]) != dataset.source_path:
        raise StrategyError("dataset registry path changed before registration")
    _require_file_content_hash(
        dataset.path,
        dataset.content_hash,
        "Pool impact dataset bytes changed before registration",
    )
    if not require_current_workspace:
        return
    row = conn.execute(
        """
        SELECT revision, active_dataset_id, active_dataset_content_hash,
               analysis_generation, semantic_mapping_json
          FROM data_workspaces WHERE task_id = ?
        """,
        (task_id,),
    ).fetchone()
    if row is None:
        raise StrategyError("DataWorkspace disappeared before registration")
    try:
        raw_mapping = str(row["semantic_mapping_json"])
        mapping = data_semantic_mapping_from_dict(json.loads(raw_mapping))
        canonical_mapping = json.dumps(
            {
                "target_col": mapping.target_col,
                "field_roles": dict(mapping.field_roles),
                "business_names": dict(mapping.business_names),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StrategyError("DataWorkspace semantic mapping is invalid") from exc
    if raw_mapping != canonical_mapping:
        raise StrategyError("DataWorkspace semantic mapping is not canonical")
    if (
        int(row["revision"]) != request["workspace_revision"]
        or str(row["active_dataset_id"]) != dataset.dataset_id
        or str(row["active_dataset_content_hash"]) != dataset.content_hash
        or int(row["analysis_generation"]) != request["workspace_generation"]
        or not hmac.compare_digest(
            data_semantic_mapping_hash(mapping), request["semantic_mapping_hash"]
        )
        or mapping.target_col != request["target_col"]
    ):
        raise StrategyError("DataWorkspace changed before impact registration")


def _require_sample_design_on_connection(
    conn,
    *,
    task_id: str,
    binding: StrategySampleDesignExecutionBinding,
) -> None:
    row = conn.execute(
        """
        SELECT task_id, kind, path, content_hash, origin_tool, provenance_json
          FROM task_artifacts
         WHERE id = ?
        """,
        (binding.reference.artifact_id,),
    ).fetchone()
    expected_provenance = json.dumps(
        binding.artifact.provenance,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if (
        row is None
        or str(row["task_id"]) != task_id
        or str(row["kind"]) != SAMPLE_DESIGN_ARTIFACT_KIND
        or str(row["path"]) != str(binding.artifact.path)
        or not hmac.compare_digest(
            str(row["content_hash"]), binding.reference.artifact_content_hash
        )
        or str(row["origin_tool"]) != SAMPLE_DESIGN_ORIGIN_TOOL
        or str(row["provenance_json"]) != expected_provenance
    ):
        raise StrategyError(
            "strategy sample-design artifact changed before impact registration"
        )
    if sha256_file(binding.artifact.path) != binding.reference.artifact_content_hash:
        raise StrategyError(
            "strategy sample-design artifact bytes changed before impact registration"
        )


def _require_baseline_on_connection(
    conn,
    *,
    request: Mapping[str, Any],
    task_id: str,
    baseline: _BaselineBinding | None,
) -> None:
    if baseline is None:
        if request["comparison_mode"] != "absolute":
            raise StrategyError("baseline disappeared before registration")
        return
    row = conn.execute(
        """
        SELECT id, task_id, strategy_type, rules_json, score_col,
               default_decision_json, description, created_at,
               dsl_json, dsl_schema_version, dsl_content_hash
          FROM strategies WHERE id = ?
        """,
        (baseline.strategy_id,),
    ).fetchone()
    if row is None or str(row["task_id"]) != task_id:
        raise StrategyError("baseline strategy changed before registration")
    strategy = _strategy_from_row(row)
    spec_hash = _strategy_spec_hash_from_row(row)
    if (
        strategy.strategy_type != request["strategy_type"]
        or strategy.spec is None
        or strategy.spec.to_dict() != baseline.spec
        or not hmac.compare_digest(spec_hash, baseline.spec_hash)
    ):
        raise StrategyError("baseline strategy changed before registration")


def _prepare_output_directory(tasks_dir: Path | str, *, task_id: str) -> Path:
    root = Path(tasks_dir).absolute()
    if root.is_symlink():
        raise StrategyError("task artifact root must not be a symlink")
    task_dir = root / task_id
    if task_dir.exists() and task_dir.is_symlink():
        raise StrategyError("task artifact directory must not be a symlink")
    task_dir.mkdir(parents=True, exist_ok=True)
    try:
        if task_dir.resolve(strict=True).parent != root.resolve(strict=True):
            raise StrategyError("Pool impact artifact directory escaped task storage")
    except OSError as exc:
        raise StrategyError("Pool impact artifact directory is unavailable") from exc
    out_dir = task_dir / "strategy_pool_impacts"
    if out_dir.exists() and (out_dir.is_symlink() or not out_dir.is_dir()):
        raise StrategyError("Pool impact artifact path must be a regular directory")
    out_dir.mkdir(exist_ok=True)
    if out_dir.is_symlink() or out_dir.resolve(strict=True).parent != task_dir.resolve(
        strict=True
    ):
        raise StrategyError("Pool impact artifact directory escaped task storage")
    return out_dir


def _require_existing_artifact(
    row,
    *,
    task_id: str,
    final_path: Path,
    canonical: bytes,
    content_hash: str,
    provenance: Mapping[str, Any],
) -> None:
    record = {field: row[field] for field in _TASK_ARTIFACT_ROW_FIELDS}
    expected = {
        "task_id": task_id,
        "kind": POOL_IMPACT_ARTIFACT_KIND,
        "path": str(final_path),
        "content_hash": content_hash,
        "origin_tool": POOL_IMPACT_ORIGIN_TOOL,
    }
    if any(str(record[field]) != value for field, value in expected.items()):
        raise StrategyError("existing Pool impact artifact registry row changed")
    expected_provenance = json.dumps(
        dict(provenance),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if str(record["provenance_json"]) != expected_provenance:
        raise StrategyError("existing Pool impact artifact provenance changed")
    _verify_file(
        final_path,
        root=final_path.parents[2],
        canonical=canonical,
        content_hash=content_hash,
    )


def _verify_file(
    path: Path,
    *,
    root: Path,
    canonical: bytes,
    content_hash: str,
) -> None:
    raw = _read_regular_nofollow(
        path,
        root=root,
        expected_content_hash=content_hash,
    )
    if raw != canonical:
        raise StrategyError("Pool impact artifact bytes changed")


def _read_regular_nofollow(
    path: Path,
    *,
    root: Path,
    expected_content_hash: str,
) -> bytes:
    """Read one bounded regular file without following or racing a path swap."""

    if not path.is_absolute() or path.is_symlink():
        raise StrategyError("Pool impact artifact must be a regular file")
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise StrategyError("Pool impact artifact escaped task storage") from exc

    descriptor = -1
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    total = 0
    try:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise StrategyError("Pool impact artifact must be a regular file")
        if (
            before.st_size < 0
            or before.st_size > _MAX_POOL_IMPACT_ARTIFACT_BYTES
        ):
            raise StrategyError("Pool impact artifact exceeds byte budget")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_POOL_IMPACT_ARTIFACT_BYTES:
                raise StrategyError("Pool impact artifact exceeds byte budget")
            digest.update(chunk)
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
            raise StrategyError("Pool impact artifact changed while being read")
        live_path = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(live_path.st_mode)
            or (live_path.st_dev, live_path.st_ino)
            != (after.st_dev, after.st_ino)
        ):
            raise StrategyError(
                "Pool impact artifact path changed while being read"
            )
    except OSError as exc:
        raise StrategyError("Pool impact artifact is unavailable") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    raw = b"".join(chunks)
    if (
        len(raw) != before.st_size
        or not hmac.compare_digest(digest.hexdigest(), expected_content_hash)
    ):
        raise StrategyError("Pool impact artifact bytes changed")
    return raw


def _tool_output(
    assessment: Mapping[str, Any],
    *,
    record: Mapping[str, Any],
    task_id: str,
    nan_labels_excluded: int,
) -> dict[str, Any]:
    identity = assessment["identity"]
    path = Path(str(record["path"]))
    artifact_id = str(record["id"])
    warnings = [
        str(flag["message"])
        for flag in assessment["red_flags"]
        if flag.get("level") in {"amber", "red"}
    ]
    return {
        "schema_version": POOL_IMPACT_TOOL_SCHEMA_VERSION,
        "assessment_id": assessment["assessment_id"],
        "content_hash": assessment["content_hash"],
        "pool_id": identity["pool_id"],
        "revision": identity["revision"],
        "snapshot_hash": identity["snapshot_hash"],
        "design_hash": identity["design_hash"],
        "strategy_type": identity["strategy_type"],
        "comparison_mode": assessment["bindings"]["comparison_mode"],
        "population_count": assessment["population"]["population_count"],
        "labeled_count": assessment["population"]["labelled_count"],
        "nan_labels_excluded": nan_labels_excluded,
        "monthly_status": assessment["monthly"]["status"],
        "assessment": dict(assessment),
        "warnings": warnings,
        "artifacts": [
            {
                "artifact_id": artifact_id,
                "kind": POOL_IMPACT_ARTIFACT_KIND,
                "format": "json",
                "filename": path.name,
                "content_hash": str(record["content_hash"]),
                "download_url": (
                    f"/api/tasks/{quote(task_id, safe='')}"
                    f"/task-artifacts/{quote(artifact_id, safe='')}/download"
                ),
            }
        ],
        "not_created_strategy": True,
        "not_adopted": True,
        "not_deployed": True,
    }


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise StrategyError(f"{name} must be a non-empty string")
    return value.strip()


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise StrategyError(f"{name} must be a lowercase SHA-256 hash")
    return value


def _positive_int(value: object, name: str) -> int:
    normalized = _non_negative_int(value, name)
    if normalized < 1:
        raise StrategyError(f"{name} must be at least 1")
    return normalized


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StrategyError(f"{name} must be a non-negative integer")
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_object(value: object, name: str) -> dict[str, Any]:
    try:
        normalized = json.loads(_canonical_json(value))
    except (TypeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise StrategyError(f"{name} must be a finite JSON object") from exc
    if not isinstance(normalized, dict):
        raise StrategyError(f"{name} must be an object")
    return normalized


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrategyError(f"JSON contains duplicate key: {key}")
        result[key] = value
    return result


def _require_binding_connection(conn, *, db_path: Path, name: str) -> None:
    if not conn.in_transaction:
        raise StrategyError(f"{name} binding requires a caller-owned transaction")
    database = conn.execute(
        "SELECT file FROM pragma_database_list WHERE name = 'main'"
    ).fetchone()
    if (
        database is None
        or not str(database["file"])
        or Path(str(database["file"])).absolute() != db_path
    ):
        raise StrategyError(f"{name} binding database changed")


__all__ = [
    "POOL_IMPACT_ARTIFACT_KIND",
    "POOL_IMPACT_ARTIFACT_SCHEMA_VERSION",
    "POOL_IMPACT_ORIGIN_TOOL",
    "POOL_IMPACT_TOOL_SCHEMA_VERSION",
    "StrategyPoolImpactArtifactBinding",
    "load_historical_strategy_pool_impact_artifact",
    "load_strategy_pool_impact_artifact",
    "require_historical_strategy_pool_impact_artifact_binding_on_connection",
    "require_strategy_pool_impact_artifact_binding_on_connection",
    "run_measure_pool_impact",
    "validate_measure_pool_impact_tool_output",
]
