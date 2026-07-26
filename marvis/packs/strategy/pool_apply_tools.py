"""Governed full-population writeback for the current Strategy Pool.

The current Pool remains an unadopted draft.  This Tool authenticates its
complete candidate lineage, resolves any model-score requirements in memory,
applies deterministic first-match semantics to the governed source universe,
and atomically registers a non-active derived Parquet plus immutable evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import hmac
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote

import pandas as pd
import pyarrow

from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.workspace import data_semantic_mapping_hash
from marvis.files import sha256_file
from marvis.packs.strategy.apply_projection import (
    DEFAULT_POOL_APPLY_PREFIX,
    resolve_apply_output_columns,
)
from marvis.packs.strategy.dsl import strategy_spec_hash
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pool_apply import (
    MAX_POOL_APPLY_ROWS,
    MAX_POOL_APPLY_SOURCE_COLUMNS,
    StrategyPoolApplyResult,
    apply_strategy_pool,
)
from marvis.packs.strategy.pool_requirement_resolver import (
    ResolvedPoolRequirements,
    hydrate_requirement_fields,
    normalize_pool_requirements,
    project_pool_entry_requirements,
    require_resolved_pool_requirements_on_connection,
    resolve_pool_requirements,
)
from marvis.packs.strategy.pool_tools import (
    StrategyPoolDevelopmentExecutionBinding,
    bind_strategy_pool_development_execution,
    load_current_strategy_candidate_pool_artifact,
    require_strategy_pool_development_execution_binding_on_connection,
)
from marvis.repositories.task_artifacts import stable_task_artifact_id


TOOL_SCHEMA_VERSION = "strategy.apply-strategy-pool-tool.v1"
EVIDENCE_SCHEMA_VERSION = "strategy.pool-apply-evidence.v1"
EVIDENCE_PROVENANCE_SCHEMA_VERSION = (
    "strategy.pool-apply-evidence-artifact.v1"
)
PRODUCER_VERSION = "1"
EVIDENCE_ARTIFACT_KIND = "strategy_pool_apply_evidence_json"
ORIGIN_TOOL = "strategy.apply_strategy_pool"
RESULT_DATASET_ROLE = "strategy.pool.applied"
DEFAULT_OUTPUT_PREFIX = DEFAULT_POOL_APPLY_PREFIX

_INPUT_FIELDS = frozenset(
    {
        "strategy_type",
        "expected_pool_revision",
        "expected_pool_snapshot_hash",
        "output_prefix",
    }
)
_REQUIRED_INPUT_FIELDS = _INPUT_FIELDS - {"output_prefix"}
_OUTPUT_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "input_hash",
        "cached",
        "activated",
        "adopted",
        "deployed",
        "source",
        "result",
        "columns",
        "action_counts",
        "rule_counts",
        "entry_counts",
        "default_count",
        "requirements",
        "workspace",
        "evidence",
    }
)
_EVIDENCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "task_id",
        "run_id",
        "input_hash",
        "activated",
        "adopted",
        "deployed",
        "source",
        "result",
        "columns",
        "action_counts",
        "rule_counts",
        "entry_counts",
        "default_count",
        "requirements",
        "workspace",
    }
)
_SOURCE_FIELDS = frozenset(
    {
        "pool_id",
        "revision",
        "revision_id",
        "snapshot_hash",
        "pool_artifact_id",
        "pool_artifact_content_hash",
        "design_hash",
        "strategy_spec_hash",
        "dataset_id",
        "dataset_content_hash",
        "row_count",
        "sample_design_ref",
    }
)
_RESULT_FIELDS = frozenset(
    {"dataset_id", "dataset_content_hash", "row_count", "result_hash"}
)
_COLUMN_FIELDS = frozenset(
    {"action", "value", "value_type", "rule_id", "entry_id", "reason_code"}
)
_REQUIREMENT_FIELDS = frozenset({"requirements_hash", "virtual_fields"})
_WORKSPACE_FIELDS = frozenset(
    {
        "source_revision",
        "source_analysis_generation",
        "source_semantic_mapping_hash",
        "active_dataset_id",
        "result_revision",
        "result_analysis_generation",
    }
)
_EVIDENCE_OUTPUT_FIELDS = frozenset(
    {"artifact_id", "content_hash", "download_url"}
)
_SAMPLE_REF_FIELDS = frozenset(
    {
        "artifact_id",
        "artifact_content_hash",
        "sample_design_id",
        "sample_design_content_hash",
        "partition",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "task_id",
        "run_id",
        "input_hash",
        "input_identity",
        "result_dataset_id",
        "result_dataset_content_hash",
        "result_hash",
    }
)
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID_RE = re.compile(r"^strategy-pool-apply-[0-9a-f]{32}$")
_DATASET_ID_RE = re.compile(r"^ds_[0-9a-f]{32}$")
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_WRITER_CONTRACT = {
    "schema_version": "strategy.pool-apply-parquet-writer.v1",
    "engine": "pyarrow",
    "engine_version": str(pyarrow.__version__),
    "pandas_version": str(pd.__version__),
    "compression": "zstd",
    "index": False,
}


def run_apply_strategy_pool(inputs: object, ctx, runtime) -> dict[str, Any]:
    """Apply the exact current Pool to its governed source population."""

    normalized = _validate_inputs(inputs)
    task_id = _required_text(ctx.task_id, "task_id")
    pool = load_current_strategy_candidate_pool_artifact(
        runtime,
        task_id=task_id,
        strategy_type=normalized["strategy_type"],
        expected_pool_revision=normalized["expected_pool_revision"],
        expected_pool_snapshot_hash=normalized[
            "expected_pool_snapshot_hash"
        ],
    )
    development = bind_strategy_pool_development_execution(runtime, pool)
    resolved = _resolve_requirements(
        runtime,
        task_id=task_id,
        development=development,
    )
    workspace = _require_source_workspace(runtime, development)
    output_columns = resolve_apply_output_columns(
        development.dataset.columns,
        output_prefix=normalized["output_prefix"],
        default_prefix=DEFAULT_OUTPUT_PREFIX,
        include_entry_id=True,
    ).as_dict()
    identity = _input_identity(
        task_id=task_id,
        development=development,
        resolved=resolved,
        output_columns=output_columns,
    )
    input_hash = _canonical_sha256(identity)
    run_id = f"strategy-pool-apply-{input_hash[:32]}"
    result_dir = _prepare_output_directory(
        runtime.datasets_root,
        task_id=task_id,
        child="strategy_pool_applies",
    )
    evidence_dir = _prepare_output_directory(
        runtime.settings.tasks_dir,
        task_id=task_id,
        child="strategy_pool_applies",
    )
    result_path = result_dir / f"{run_id}.parquet"
    evidence_path = evidence_dir / f"{run_id}.evidence.json"

    cached = _try_cached_run(
        runtime,
        task_id=task_id,
        development=development,
        resolved=resolved,
        identity=identity,
        input_hash=input_hash,
        run_id=run_id,
        result_path=result_path,
        evidence_path=evidence_path,
    )
    if cached is not None:
        return cached

    source_path = _require_source_file(development)
    original = _read_source_frame(
        runtime,
        development=development,
        source_path=source_path,
    )
    execution = hydrate_requirement_fields(original, resolved=resolved)
    kernel = apply_strategy_pool(
        original,
        execution,
        development.pool.pool,
        compiled_design=development.pool.compiled_design,
        output_prefix=normalized["output_prefix"],
    )
    if dict(kernel.output_columns) != output_columns:
        raise StrategyError("Strategy Pool apply output columns changed")

    uow = ArtifactUnitOfWork()
    commit_state = {"db_committed": False}
    staged_result = uow.stage_file(result_dir, result_path.name)
    try:
        kernel.derived_frame.to_parquet(
            staged_result.path,
            engine=_WRITER_CONTRACT["engine"],
            compression=_WRITER_CONTRACT["compression"],
            index=False,
        )
        staged_result_hash = sha256_file(staged_result.path)
        _require_staged_result(
            runtime,
            staged_result.path,
            kernel=kernel,
            source_columns=development.dataset.columns,
        )
        output = _commit_computed_run(
            runtime,
            task_id=task_id,
            development=development,
            resolved=resolved,
            identity=identity,
            input_hash=input_hash,
            run_id=run_id,
            result_path=result_path,
            evidence_path=evidence_path,
            staged_result=staged_result,
            staged_result_hash=staged_result_hash,
            kernel=kernel,
            source_workspace=workspace,
            uow=uow,
            commit_state=commit_state,
            seed=int(ctx.seed or 0),
        )
    except Exception:
        if not commit_state["db_committed"]:
            uow.rollback()
        raise
    return output


def validate_apply_strategy_pool_tool_output(
    payload: object,
) -> dict[str, Any]:
    """Validate and detach the public Tool output used by Agent renderers."""

    value = _object(payload, "Strategy Pool apply output")
    _exact_fields(value, _OUTPUT_FIELDS, "Strategy Pool apply output")
    if value["schema_version"] != TOOL_SCHEMA_VERSION:
        raise StrategyError(
            f"Strategy Pool apply schema_version must be {TOOL_SCHEMA_VERSION}"
        )
    run_id = _run_id(value["run_id"])
    input_hash = _hash(value["input_hash"], "input_hash")
    cached = _boolean(value["cached"], "cached")
    flags = _inactive_flags(value)
    source = _source(value["source"])
    result = _result(value["result"])
    columns = _columns(value["columns"])
    action_counts = _counts(value["action_counts"], "action_counts")
    rule_counts = _counts(value["rule_counts"], "rule_counts")
    entry_counts = _counts(value["entry_counts"], "entry_counts")
    default_count = _non_negative_int(value["default_count"], "default_count")
    requirements = _requirements(value["requirements"])
    workspace = _workspace(value["workspace"])
    evidence = _evidence_output(value["evidence"])
    _require_cross_field_consistency(
        run_id=run_id,
        input_hash=input_hash,
        source=source,
        result=result,
        columns=columns,
        action_counts=action_counts,
        rule_counts=rule_counts,
        entry_counts=entry_counts,
        default_count=default_count,
        workspace=workspace,
    )
    normalized = {
        "schema_version": TOOL_SCHEMA_VERSION,
        "run_id": run_id,
        "input_hash": input_hash,
        "cached": cached,
        **flags,
        "source": source,
        "result": result,
        "columns": columns,
        "action_counts": action_counts,
        "rule_counts": rule_counts,
        "entry_counts": entry_counts,
        "default_count": default_count,
        "requirements": requirements,
        "workspace": workspace,
        "evidence": evidence,
    }
    return _json_clone(normalized)


def _validate_inputs(value: object) -> dict[str, Any]:
    inputs = _object(value, "Strategy Pool apply inputs")
    _exact_fields(
        inputs,
        _INPUT_FIELDS,
        "Strategy Pool apply inputs",
        required=_REQUIRED_INPUT_FIELDS,
    )
    strategy_type = _required_text(inputs["strategy_type"], "strategy_type")
    revision = _positive_int(
        inputs["expected_pool_revision"],
        "expected_pool_revision",
    )
    snapshot_hash = _hash(
        inputs["expected_pool_snapshot_hash"],
        "expected_pool_snapshot_hash",
    )
    prefix = inputs.get("output_prefix", DEFAULT_OUTPUT_PREFIX)
    if not isinstance(prefix, str):
        raise StrategyError("output_prefix must be a string")
    return {
        "strategy_type": strategy_type,
        "expected_pool_revision": revision,
        "expected_pool_snapshot_hash": snapshot_hash,
        "output_prefix": prefix,
    }


def _resolve_requirements(
    runtime,
    *,
    task_id: str,
    development: StrategyPoolDevelopmentExecutionBinding,
) -> ResolvedPoolRequirements:
    compiled = normalize_pool_requirements(
        development.pool.compiled_design.get("requirements")
    )
    projected = project_pool_entry_requirements(
        development.pool.pool["entries"]
    )
    if compiled != projected:
        raise StrategyError(
            "Strategy Pool compiled requirements changed from Pool entries"
        )
    requirements_hash = _canonical_sha256(list(compiled))
    if not compiled:
        return ResolvedPoolRequirements(
            task_id=task_id,
            requirements_hash=requirements_hash,
            requirements=(),
            field_bindings=(),
        )
    if development.sample_design_v2 is None:
        raise StrategyError(
            "Strategy Pool model-score requirements require SampleDesign V2"
        )
    resolved = resolve_pool_requirements(
        runtime,
        task_id=task_id,
        compiled_design=development.pool.compiled_design,
        sample_design=development.sample_design_v2,
    )
    if (
        resolved.requirements != compiled
        or not hmac.compare_digest(
            resolved.requirements_hash,
            requirements_hash,
        )
    ):
        raise StrategyError("Strategy Pool requirement bindings changed")
    return resolved


def _require_source_workspace(
    runtime,
    development: StrategyPoolDevelopmentExecutionBinding,
):
    workspace = runtime.data_workspaces.get_or_default(development.task_id)
    sample = development.sample_design
    if (
        workspace.active_dataset_id != development.dataset.dataset_id
        or workspace.active_dataset_content_hash
        != development.dataset.content_hash
        or workspace.revision != sample.workspace_revision
        or workspace.analysis_generation != sample.workspace_generation
        or not hmac.compare_digest(
            data_semantic_mapping_hash(workspace.semantic_mapping),
            sample.semantic_mapping_hash,
        )
    ):
        raise StrategyError(
            "Strategy Pool source data workspace changed from candidate lineage"
        )
    return workspace


def _input_identity(
    *,
    task_id: str,
    development: StrategyPoolDevelopmentExecutionBinding,
    resolved: ResolvedPoolRequirements,
    output_columns: Mapping[str, str],
) -> dict[str, Any]:
    pool = development.pool
    snapshot = pool.pool
    spec_hash = strategy_spec_hash(pool.compiled_design["strategy_spec"])
    return {
        "schema_version": "strategy.pool-apply-input.v1",
        "producer_version": PRODUCER_VERSION,
        "task_id": task_id,
        "pool": {
            "pool_id": snapshot["pool_id"],
            "strategy_type": snapshot["strategy_type"],
            "revision": snapshot["revision"],
            "revision_id": snapshot["revision_id"],
            "snapshot_hash": snapshot["snapshot_hash"],
            "artifact_id": pool.artifact_id,
            "artifact_content_hash": pool.artifact_content_hash,
            "design_hash": pool.compiled_design["design_hash"],
            "strategy_spec_hash": spec_hash,
        },
        "dataset": {
            "dataset_id": development.dataset.dataset_id,
            "dataset_content_hash": development.dataset.content_hash,
            "registry_metadata_hash": (
                development.dataset.registry_metadata_hash
            ),
            "row_count": development.dataset.row_count,
            "columns": list(development.dataset.columns),
        },
        "sample_design_ref": development.sample_design.to_ref_dict(),
        "workspace": {
            "revision": development.sample_design.workspace_revision,
            "analysis_generation": (
                development.sample_design.workspace_generation
            ),
            "semantic_mapping_hash": (
                development.sample_design.semantic_mapping_hash
            ),
        },
        "requirements": {
            "requirements_hash": resolved.requirements_hash,
            "virtual_fields": list(resolved.virtual_fields),
        },
        "output_columns": dict(output_columns),
        "writer_contract": dict(_WRITER_CONTRACT),
    }


def _try_cached_run(
    runtime,
    *,
    task_id: str,
    development: StrategyPoolDevelopmentExecutionBinding,
    resolved: ResolvedPoolRequirements,
    identity: Mapping[str, Any],
    input_hash: str,
    run_id: str,
    result_path: Path,
    evidence_path: Path,
) -> dict[str, Any] | None:
    with runtime.task_artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        _reauthorize_on_connection(
            conn,
            development=development,
            resolved=resolved,
        )
        _require_source_file(development)
        row = _select_evidence_row(
            conn,
            task_id=task_id,
            evidence_path=evidence_path,
        )
        if row is None:
            if (
                result_path.exists()
                or result_path.is_symlink()
                or evidence_path.exists()
                or evidence_path.is_symlink()
            ):
                raise StrategyError(
                    "Strategy Pool apply output path exists without governed evidence"
                )
            conn.commit()
            return None
        output = _load_cached_output(
            conn,
            runtime,
            row=row,
            task_id=task_id,
            identity=identity,
            input_hash=input_hash,
            run_id=run_id,
            result_path=result_path,
            evidence_path=evidence_path,
            cached=True,
        )
        _reauthorize_on_connection(
            conn,
            development=development,
            resolved=resolved,
        )
        conn.commit()
        return output


def _read_source_frame(
    runtime,
    *,
    development: StrategyPoolDevelopmentExecutionBinding,
    source_path: Path,
) -> pd.DataFrame:
    dataset = development.dataset
    if dataset.row_count > MAX_POOL_APPLY_ROWS:
        raise StrategyError(
            f"Strategy Pool apply supports at most {MAX_POOL_APPLY_ROWS} rows"
        )
    if len(dataset.columns) > MAX_POOL_APPLY_SOURCE_COLUMNS:
        raise StrategyError(
            "Strategy Pool apply supports at most "
            f"{MAX_POOL_APPLY_SOURCE_COLUMNS} source columns"
        )
    before = sha256_file(source_path)
    if not hmac.compare_digest(before, dataset.content_hash):
        raise StrategyError("Strategy Pool source dataset content hash changed")
    frame = runtime.backend.read_frame(source_path).reset_index(drop=True)
    after = sha256_file(source_path)
    if not hmac.compare_digest(before, after):
        raise StrategyError(
            "Strategy Pool source dataset changed while it was being read"
        )
    if (
        len(frame) != dataset.row_count
        or tuple(str(column) for column in frame.columns) != dataset.columns
    ):
        raise StrategyError("Strategy Pool source dataset shape changed")
    return frame


def _require_staged_result(
    runtime,
    path: Path,
    *,
    kernel: StrategyPoolApplyResult,
    source_columns: Sequence[str],
) -> None:
    expected_columns = [
        *source_columns,
        *dict(kernel.output_columns).values(),
    ]
    if runtime.backend.row_count(path) != kernel.source_row_count:
        raise StrategyError("Strategy Pool apply result row count changed")
    if runtime.backend.column_names(path) != expected_columns:
        raise StrategyError("Strategy Pool apply result columns changed")


def _commit_computed_run(
    runtime,
    *,
    task_id: str,
    development: StrategyPoolDevelopmentExecutionBinding,
    resolved: ResolvedPoolRequirements,
    identity: Mapping[str, Any],
    input_hash: str,
    run_id: str,
    result_path: Path,
    evidence_path: Path,
    staged_result,
    staged_result_hash: str,
    kernel: StrategyPoolApplyResult,
    source_workspace,
    uow: ArtifactUnitOfWork,
    commit_state: dict[str, bool],
    seed: int,
) -> dict[str, Any]:
    with runtime.task_artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        _reauthorize_on_connection(
            conn,
            development=development,
            resolved=resolved,
        )
        _require_source_file(development)
        row = _select_evidence_row(
            conn,
            task_id=task_id,
            evidence_path=evidence_path,
        )
        if row is not None:
            output = _load_cached_output(
                conn,
                runtime,
                row=row,
                task_id=task_id,
                identity=identity,
                input_hash=input_hash,
                run_id=run_id,
                result_path=result_path,
                evidence_path=evidence_path,
                cached=True,
            )
            if (
                output["result"]["result_hash"] != kernel.assignment_hash
                or output["result"]["dataset_content_hash"]
                != staged_result_hash
            ):
                raise StrategyError(
                    "concurrent Strategy Pool apply result differs from replay"
                )
            uow.rollback()
            conn.commit()
            commit_state["db_committed"] = True
            return output
        if (
            result_path.exists()
            or result_path.is_symlink()
            or evidence_path.exists()
            or evidence_path.is_symlink()
        ):
            raise StrategyError(
                "Strategy Pool apply output path exists without governed evidence"
            )

        staged_result.promote()
        if not hmac.compare_digest(
            sha256_file(result_path),
            staged_result_hash,
        ):
            raise StrategyError("Strategy Pool apply result changed during promotion")
        registered = runtime.registry.register_existing_on_connection(
            conn,
            result_path,
            task_id=task_id,
            role=RESULT_DATASET_ROLE,
            anchor_target=development.dataset.dataset_id,
            target_col_override=development.target_col,
            seed=seed,
        )
        if (
            registered.task_id != task_id
            or registered.role != RESULT_DATASET_ROLE
            or registered.row_count != kernel.source_row_count
            or not hmac.compare_digest(
                str(registered.content_hash),
                staged_result_hash,
            )
        ):
            raise StrategyError(
                "registered Strategy Pool apply dataset changed"
            )
        source = _source_projection(development)
        result = {
            "dataset_id": registered.id,
            "dataset_content_hash": staged_result_hash,
            "row_count": kernel.source_row_count,
            "result_hash": kernel.assignment_hash,
        }
        workspace = _workspace_projection(
            development,
            source_workspace=source_workspace,
        )
        evidence_document = _evidence_document(
            task_id=task_id,
            run_id=run_id,
            input_hash=input_hash,
            source=source,
            result=result,
            columns=dict(kernel.output_columns),
            kernel=kernel,
            resolved=resolved,
            workspace=workspace,
        )
        evidence_bytes = _canonical_json(evidence_document).encode("utf-8")
        evidence_hash = hashlib.sha256(evidence_bytes).hexdigest()
        staged_evidence = uow.stage_file(
            evidence_path.parent,
            evidence_path.name,
        )
        staged_evidence.path.write_bytes(evidence_bytes)
        staged_evidence.promote()
        _require_exact_file(
            evidence_path,
            root=Path(runtime.settings.tasks_dir),
            expected=evidence_bytes,
            expected_hash=evidence_hash,
            label="Strategy Pool apply evidence",
        )
        provenance = _evidence_provenance(
            evidence_document,
            identity=identity,
        )
        record = runtime.task_artifacts.register_on_connection(
            conn,
            task_id=task_id,
            kind=EVIDENCE_ARTIFACT_KIND,
            path=str(evidence_path),
            content_hash=evidence_hash,
            origin_tool=ORIGIN_TOOL,
            provenance=provenance,
        )
        runtime.repo.write_audit_on_connection(
            conn,
            kind="strategy.pool.apply",
            target_ref=registered.id,
            inputs_hash=input_hash,
            outcome="succeeded",
            detail={
                "task_id": task_id,
                "run_id": run_id,
                "input_hash": input_hash,
                "source_dataset_id": development.dataset.dataset_id,
                "result_dataset_id": registered.id,
                "pool_id": development.pool.pool["pool_id"],
                "pool_revision": development.pool.pool["revision"],
                "pool_snapshot_hash": development.pool.pool["snapshot_hash"],
                "population_count": kernel.source_row_count,
                "action_counts": dict(kernel.action_counts),
                "rule_counts": dict(kernel.rule_counts),
                "entry_counts": dict(kernel.entry_counts),
                "default_count": kernel.default_count,
                "output_columns": dict(kernel.output_columns),
                "activated": False,
                "adopted": False,
                "deployed": False,
                "evidence_artifact_id": record["id"],
            },
        )
        _reauthorize_on_connection(
            conn,
            development=development,
            resolved=resolved,
        )
        _require_source_file(development)
        conn.commit()
        commit_state["db_committed"] = True
    uow.commit()
    return _tool_output(
        evidence_document,
        record=record,
        task_id=task_id,
        cached=False,
    )


def _evidence_document(
    *,
    task_id: str,
    run_id: str,
    input_hash: str,
    source: Mapping[str, Any],
    result: Mapping[str, Any],
    columns: Mapping[str, str],
    kernel: StrategyPoolApplyResult,
    resolved: ResolvedPoolRequirements,
    workspace: Mapping[str, Any],
) -> dict[str, Any]:
    document = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "producer_version": PRODUCER_VERSION,
        "task_id": task_id,
        "run_id": run_id,
        "input_hash": input_hash,
        "activated": False,
        "adopted": False,
        "deployed": False,
        "source": dict(source),
        "result": dict(result),
        "columns": dict(columns),
        "action_counts": dict(kernel.action_counts),
        "rule_counts": dict(kernel.rule_counts),
        "entry_counts": dict(kernel.entry_counts),
        "default_count": kernel.default_count,
        "requirements": {
            "requirements_hash": resolved.requirements_hash,
            "virtual_fields": list(resolved.virtual_fields),
        },
        "workspace": dict(workspace),
    }
    return _validate_evidence_document(document, expected_task_id=task_id)


def _source_projection(
    development: StrategyPoolDevelopmentExecutionBinding,
) -> dict[str, Any]:
    pool = development.pool
    snapshot = pool.pool
    return {
        "pool_id": snapshot["pool_id"],
        "revision": snapshot["revision"],
        "revision_id": snapshot["revision_id"],
        "snapshot_hash": snapshot["snapshot_hash"],
        "pool_artifact_id": pool.artifact_id,
        "pool_artifact_content_hash": pool.artifact_content_hash,
        "design_hash": pool.compiled_design["design_hash"],
        "strategy_spec_hash": strategy_spec_hash(
            pool.compiled_design["strategy_spec"]
        ),
        "dataset_id": development.dataset.dataset_id,
        "dataset_content_hash": development.dataset.content_hash,
        "row_count": development.dataset.row_count,
        "sample_design_ref": development.sample_design.to_ref_dict(),
    }


def _workspace_projection(
    development: StrategyPoolDevelopmentExecutionBinding,
    *,
    source_workspace,
) -> dict[str, Any]:
    sample = development.sample_design
    if (
        source_workspace.revision != sample.workspace_revision
        or source_workspace.analysis_generation != sample.workspace_generation
    ):
        raise StrategyError("Strategy Pool source workspace changed before commit")
    return {
        "source_revision": sample.workspace_revision,
        "source_analysis_generation": sample.workspace_generation,
        "source_semantic_mapping_hash": sample.semantic_mapping_hash,
        "active_dataset_id": development.dataset.dataset_id,
        "result_revision": None,
        "result_analysis_generation": None,
    }


def _load_cached_output(
    conn,
    runtime,
    *,
    row,
    task_id: str,
    identity: Mapping[str, Any],
    input_hash: str,
    run_id: str,
    result_path: Path,
    evidence_path: Path,
    cached: bool,
) -> dict[str, Any]:
    record = _task_artifact_record(row)
    expected_artifact_id = stable_task_artifact_id(
        task_id=task_id,
        kind=EVIDENCE_ARTIFACT_KIND,
        path=str(evidence_path),
    )
    if (
        record["id"] != expected_artifact_id
        or record["task_id"] != task_id
        or record["kind"] != EVIDENCE_ARTIFACT_KIND
        or record["path"] != str(evidence_path)
        or record["origin_tool"] != ORIGIN_TOOL
    ):
        raise StrategyError(
            "cached Strategy Pool apply evidence registry binding changed"
        )
    evidence_bytes = _read_exact_file(
        evidence_path,
        root=Path(runtime.settings.tasks_dir),
        expected_hash=record["content_hash"],
        label="cached Strategy Pool apply evidence",
    )
    try:
        raw = evidence_bytes.decode("utf-8")
        parsed = json.loads(raw, object_pairs_hook=_object_without_duplicate_keys)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as exc:
        raise StrategyError(
            "cached Strategy Pool apply evidence is invalid"
        ) from exc
    document = _validate_evidence_document(
        parsed,
        expected_task_id=task_id,
    )
    if _canonical_json(document) != raw:
        raise StrategyError(
            "cached Strategy Pool apply evidence is not canonical"
        )
    if (
        document["run_id"] != run_id
        or not hmac.compare_digest(document["input_hash"], input_hash)
        or not hmac.compare_digest(_canonical_sha256(identity), input_hash)
    ):
        raise StrategyError(
            "cached Strategy Pool apply input identity changed"
        )
    _require_document_matches_identity(document, identity=identity)
    provenance = _evidence_provenance(document, identity=identity)
    if record["provenance"] != provenance:
        raise StrategyError(
            "cached Strategy Pool apply evidence provenance changed"
        )
    _require_cached_result_dataset(
        conn,
        runtime,
        task_id=task_id,
        result=document["result"],
        expected_path=result_path,
    )
    return _tool_output(
        document,
        record=record,
        task_id=task_id,
        cached=cached,
    )


def _validate_evidence_document(
    payload: object,
    *,
    expected_task_id: str,
) -> dict[str, Any]:
    value = _object(payload, "Strategy Pool apply evidence")
    _exact_fields(value, _EVIDENCE_FIELDS, "Strategy Pool apply evidence")
    if value["schema_version"] != EVIDENCE_SCHEMA_VERSION:
        raise StrategyError(
            "Strategy Pool apply evidence schema_version changed"
        )
    if value["producer_version"] != PRODUCER_VERSION:
        raise StrategyError(
            "Strategy Pool apply evidence producer_version changed"
        )
    task_id = _required_text(value["task_id"], "evidence.task_id")
    if task_id != expected_task_id:
        raise StrategyError(
            "Strategy Pool apply evidence belongs to another task"
        )
    run_id = _run_id(value["run_id"])
    input_hash = _hash(value["input_hash"], "evidence.input_hash")
    flags = _inactive_flags(value)
    source = _source(value["source"])
    result = _result(value["result"])
    columns = _columns(value["columns"])
    action_counts = _counts(value["action_counts"], "action_counts")
    rule_counts = _counts(value["rule_counts"], "rule_counts")
    entry_counts = _counts(value["entry_counts"], "entry_counts")
    default_count = _non_negative_int(value["default_count"], "default_count")
    requirements = _requirements(value["requirements"])
    workspace = _workspace(value["workspace"])
    _require_cross_field_consistency(
        run_id=run_id,
        input_hash=input_hash,
        source=source,
        result=result,
        columns=columns,
        action_counts=action_counts,
        rule_counts=rule_counts,
        entry_counts=entry_counts,
        default_count=default_count,
        workspace=workspace,
    )
    return _json_clone(
        {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "producer_version": PRODUCER_VERSION,
            "task_id": task_id,
            "run_id": run_id,
            "input_hash": input_hash,
            **flags,
            "source": source,
            "result": result,
            "columns": columns,
            "action_counts": action_counts,
            "rule_counts": rule_counts,
            "entry_counts": entry_counts,
            "default_count": default_count,
            "requirements": requirements,
            "workspace": workspace,
        }
    )


def _tool_output(
    document: Mapping[str, Any],
    *,
    record: Mapping[str, Any],
    task_id: str,
    cached: bool,
) -> dict[str, Any]:
    artifact_id = str(record["id"])
    output = {
        "schema_version": TOOL_SCHEMA_VERSION,
        "run_id": document["run_id"],
        "input_hash": document["input_hash"],
        "cached": cached,
        "activated": False,
        "adopted": False,
        "deployed": False,
        "source": document["source"],
        "result": document["result"],
        "columns": document["columns"],
        "action_counts": document["action_counts"],
        "rule_counts": document["rule_counts"],
        "entry_counts": document["entry_counts"],
        "default_count": document["default_count"],
        "requirements": document["requirements"],
        "workspace": document["workspace"],
        "evidence": {
            "artifact_id": artifact_id,
            "content_hash": str(record["content_hash"]),
            "download_url": (
                f"/api/tasks/{quote(task_id, safe='')}"
                f"/task-artifacts/{quote(artifact_id, safe='')}/download"
            ),
        },
    }
    return validate_apply_strategy_pool_tool_output(output)


def _evidence_provenance(
    document: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    result = document["result"]
    value = {
        "schema_version": EVIDENCE_PROVENANCE_SCHEMA_VERSION,
        "producer_version": PRODUCER_VERSION,
        "task_id": document["task_id"],
        "run_id": document["run_id"],
        "input_hash": document["input_hash"],
        "input_identity": _json_clone(identity),
        "result_dataset_id": result["dataset_id"],
        "result_dataset_content_hash": result["dataset_content_hash"],
        "result_hash": result["result_hash"],
    }
    _exact_fields(
        value,
        _PROVENANCE_FIELDS,
        "Strategy Pool apply evidence provenance",
    )
    return _json_clone(value)


def _require_document_matches_identity(
    document: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
) -> None:
    pool = identity["pool"]
    dataset = identity["dataset"]
    workspace = identity["workspace"]
    expected_source = {
        "pool_id": pool["pool_id"],
        "revision": pool["revision"],
        "revision_id": pool["revision_id"],
        "snapshot_hash": pool["snapshot_hash"],
        "pool_artifact_id": pool["artifact_id"],
        "pool_artifact_content_hash": pool["artifact_content_hash"],
        "design_hash": pool["design_hash"],
        "strategy_spec_hash": pool["strategy_spec_hash"],
        "dataset_id": dataset["dataset_id"],
        "dataset_content_hash": dataset["dataset_content_hash"],
        "row_count": dataset["row_count"],
        "sample_design_ref": identity["sample_design_ref"],
    }
    expected_workspace = {
        "source_revision": workspace["revision"],
        "source_analysis_generation": workspace["analysis_generation"],
        "source_semantic_mapping_hash": workspace["semantic_mapping_hash"],
        "active_dataset_id": dataset["dataset_id"],
        "result_revision": None,
        "result_analysis_generation": None,
    }
    if (
        document["source"] != expected_source
        or document["columns"] != identity["output_columns"]
        or document["requirements"] != identity["requirements"]
        or document["workspace"] != expected_workspace
    ):
        raise StrategyError(
            "cached Strategy Pool apply evidence differs from live input identity"
        )


def _reauthorize_on_connection(
    conn,
    *,
    development: StrategyPoolDevelopmentExecutionBinding,
    resolved: ResolvedPoolRequirements,
) -> None:
    require_strategy_pool_development_execution_binding_on_connection(
        conn,
        development,
    )
    require_resolved_pool_requirements_on_connection(conn, resolved)


def _require_source_file(
    development: StrategyPoolDevelopmentExecutionBinding,
) -> Path:
    path = Path(development.dataset.path)
    try:
        resolved_root = development.pool.datasets_root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(resolved_root)
        if path.is_symlink() or not resolved.is_file():
            raise OSError("source dataset is not a regular file")
        actual = sha256_file(resolved)
    except (OSError, ValueError) as exc:
        raise StrategyError(
            "Strategy Pool source dataset is missing or escaped storage"
        ) from exc
    if not hmac.compare_digest(actual, development.dataset.content_hash):
        raise StrategyError("Strategy Pool source dataset content hash drifted")
    return resolved


def _require_cached_result_dataset(
    conn,
    runtime,
    *,
    task_id: str,
    result: Mapping[str, Any],
    expected_path: Path,
) -> None:
    row = conn.execute(
        """
        SELECT id, task_id, role, source_path, format, row_count, content_hash
          FROM datasets
         WHERE id = ?
        """,
        (result["dataset_id"],),
    ).fetchone()
    expected_relative = (
        expected_path.resolve(strict=False)
        .relative_to(Path(runtime.datasets_root).resolve(strict=False))
        .as_posix()
    )
    if row is None or (
        str(row["id"]) != result["dataset_id"]
        or str(row["task_id"]) != task_id
        or str(row["role"]) != RESULT_DATASET_ROLE
        or str(row["source_path"]) != expected_relative
        or str(row["format"]) != "parquet"
        or int(row["row_count"]) != result["row_count"]
        or str(row["content_hash"]) != result["dataset_content_hash"]
    ):
        raise StrategyError(
            "cached Strategy Pool apply result dataset binding changed"
        )
    _require_exact_file(
        expected_path,
        root=Path(runtime.datasets_root),
        expected=None,
        expected_hash=result["dataset_content_hash"],
        label="cached Strategy Pool apply result",
    )


def _select_evidence_row(conn, *, task_id: str, evidence_path: Path):
    return conn.execute(
        """
        SELECT id, task_id, kind, path, content_hash, origin_tool,
               provenance_json, created_at
          FROM task_artifacts
         WHERE task_id = ? AND kind = ? AND path = ?
        """,
        (task_id, EVIDENCE_ARTIFACT_KIND, str(evidence_path)),
    ).fetchone()


def _task_artifact_record(row) -> dict[str, Any]:
    try:
        provenance = json.loads(
            str(row["provenance_json"]),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise StrategyError(
            "Strategy Pool apply evidence provenance is invalid"
        ) from exc
    if not isinstance(provenance, dict):
        raise StrategyError(
            "Strategy Pool apply evidence provenance must be an object"
        )
    return {
        "id": str(row["id"]),
        "task_id": str(row["task_id"]),
        "kind": str(row["kind"]),
        "path": str(row["path"]),
        "content_hash": _hash(
            row["content_hash"],
            "evidence registry content_hash",
        ),
        "origin_tool": str(row["origin_tool"]),
        "provenance": provenance,
        "created_at": str(row["created_at"]),
    }


def _prepare_output_directory(
    root: Path,
    *,
    task_id: str,
    child: str,
) -> Path:
    root_path = Path(root).absolute()
    try:
        root_path.mkdir(parents=True, exist_ok=True)
        resolved_root = root_path.resolve(strict=True)
        task_dir = root_path / task_id
        task_dir.resolve(strict=False).relative_to(resolved_root)
        if task_dir.exists() and (
            task_dir.is_symlink() or not task_dir.is_dir()
        ):
            raise OSError("task output path is not a regular directory")
        task_dir.mkdir(exist_ok=True)
        out_dir = task_dir / child
        if out_dir.exists() and (
            out_dir.is_symlink() or not out_dir.is_dir()
        ):
            raise OSError("output path is not a regular directory")
        out_dir.mkdir(exist_ok=True)
        if (
            out_dir.is_symlink()
            or out_dir.resolve(strict=True).parent
            != task_dir.resolve(strict=True)
        ):
            raise OSError("output directory escaped task storage")
    except (OSError, ValueError) as exc:
        raise StrategyError(
            "Strategy Pool apply output directory is unsafe"
        ) from exc
    return out_dir


def _read_exact_file(
    path: Path,
    *,
    root: Path,
    expected_hash: str,
    label: str,
) -> bytes:
    try:
        resolved_root = Path(root).resolve(strict=True)
        resolved = Path(path).resolve(strict=True)
        resolved.relative_to(resolved_root)
        if Path(path).is_symlink() or not resolved.is_file():
            raise OSError(f"{label} is not a regular file")
        before = resolved.stat()
        raw = resolved.read_bytes()
        after = resolved.stat()
    except (OSError, ValueError) as exc:
        raise StrategyError(f"{label} is missing or escaped storage") from exc
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise StrategyError(f"{label} changed while it was read")
    actual_hash = hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(actual_hash, expected_hash):
        raise StrategyError(f"{label} content hash changed")
    return raw


def _require_exact_file(
    path: Path,
    *,
    root: Path,
    expected: bytes | None,
    expected_hash: str,
    label: str,
) -> None:
    raw = _read_exact_file(
        path,
        root=root,
        expected_hash=expected_hash,
        label=label,
    )
    if expected is not None and raw != expected:
        raise StrategyError(f"{label} bytes changed")


def _source(value: object) -> dict[str, Any]:
    source = _object(value, "source")
    _exact_fields(source, _SOURCE_FIELDS, "source")
    sample_ref = _object(source["sample_design_ref"], "sample_design_ref")
    _exact_fields(sample_ref, _SAMPLE_REF_FIELDS, "sample_design_ref")
    normalized_ref = {
        "artifact_id": _hash(
            sample_ref["artifact_id"],
            "sample_design_ref.artifact_id",
        ),
        "artifact_content_hash": _hash(
            sample_ref["artifact_content_hash"],
            "sample_design_ref.artifact_content_hash",
        ),
        "sample_design_id": _required_text(
            sample_ref["sample_design_id"],
            "sample_design_ref.sample_design_id",
        ),
        "sample_design_content_hash": _hash(
            sample_ref["sample_design_content_hash"],
            "sample_design_ref.sample_design_content_hash",
        ),
        "partition": _required_text(
            sample_ref["partition"],
            "sample_design_ref.partition",
        ),
    }
    if normalized_ref["partition"] != "development":
        raise StrategyError("sample_design_ref.partition must be development")
    return {
        "pool_id": _required_text(source["pool_id"], "source.pool_id"),
        "revision": _positive_int(source["revision"], "source.revision"),
        "revision_id": _required_text(
            source["revision_id"],
            "source.revision_id",
        ),
        "snapshot_hash": _hash(
            source["snapshot_hash"],
            "source.snapshot_hash",
        ),
        "pool_artifact_id": _hash(
            source["pool_artifact_id"],
            "source.pool_artifact_id",
        ),
        "pool_artifact_content_hash": _hash(
            source["pool_artifact_content_hash"],
            "source.pool_artifact_content_hash",
        ),
        "design_hash": _hash(source["design_hash"], "source.design_hash"),
        "strategy_spec_hash": _hash(
            source["strategy_spec_hash"],
            "source.strategy_spec_hash",
        ),
        "dataset_id": _dataset_id(
            source["dataset_id"],
            "source.dataset_id",
        ),
        "dataset_content_hash": _hash(
            source["dataset_content_hash"],
            "source.dataset_content_hash",
        ),
        "row_count": _positive_int(
            source["row_count"],
            "source.row_count",
        ),
        "sample_design_ref": normalized_ref,
    }


def _result(value: object) -> dict[str, Any]:
    result = _object(value, "result")
    _exact_fields(result, _RESULT_FIELDS, "result")
    return {
        "dataset_id": _dataset_id(
            result["dataset_id"],
            "result.dataset_id",
        ),
        "dataset_content_hash": _hash(
            result["dataset_content_hash"],
            "result.dataset_content_hash",
        ),
        "row_count": _positive_int(
            result["row_count"],
            "result.row_count",
        ),
        "result_hash": _hash(
            result["result_hash"],
            "result.result_hash",
        ),
    }


def _columns(value: object) -> dict[str, str]:
    columns = _object(value, "columns")
    _exact_fields(columns, _COLUMN_FIELDS, "columns")
    normalized = {
        field: _required_text(columns[field], f"columns.{field}")
        for field in (
            "action",
            "value",
            "value_type",
            "rule_id",
            "entry_id",
            "reason_code",
        )
    }
    for field, column in normalized.items():
        if _SAFE_IDENTIFIER_RE.fullmatch(column) is None:
            raise StrategyError(f"columns.{field} must be a safe identifier")
    if len({value.casefold() for value in normalized.values()}) != len(
        normalized
    ):
        raise StrategyError("Strategy Pool apply columns must be unique")
    return normalized


def _requirements(value: object) -> dict[str, Any]:
    requirements = _object(value, "requirements")
    _exact_fields(requirements, _REQUIREMENT_FIELDS, "requirements")
    raw_fields = requirements["virtual_fields"]
    if (
        isinstance(raw_fields, str | bytes | bytearray)
        or not isinstance(raw_fields, Sequence)
    ):
        raise StrategyError("requirements.virtual_fields must be an array")
    fields = [
        _required_text(item, "requirements.virtual_fields item")
        for item in raw_fields
    ]
    if len(fields) != len(set(fields)):
        raise StrategyError(
            "requirements.virtual_fields must not contain duplicates"
        )
    return {
        "requirements_hash": _hash(
            requirements["requirements_hash"],
            "requirements.requirements_hash",
        ),
        "virtual_fields": fields,
    }


def _workspace(value: object) -> dict[str, Any]:
    workspace = _object(value, "workspace")
    _exact_fields(workspace, _WORKSPACE_FIELDS, "workspace")
    if (
        workspace["result_revision"] is not None
        or workspace["result_analysis_generation"] is not None
    ):
        raise StrategyError(
            "Strategy Pool apply result must not activate the data workspace"
        )
    return {
        "source_revision": _positive_int(
            workspace["source_revision"],
            "workspace.source_revision",
        ),
        "source_analysis_generation": _positive_int(
            workspace["source_analysis_generation"],
            "workspace.source_analysis_generation",
            allow_zero=True,
        ),
        "source_semantic_mapping_hash": _hash(
            workspace["source_semantic_mapping_hash"],
            "workspace.source_semantic_mapping_hash",
        ),
        "active_dataset_id": _dataset_id(
            workspace["active_dataset_id"],
            "workspace.active_dataset_id",
        ),
        "result_revision": None,
        "result_analysis_generation": None,
    }


def _evidence_output(value: object) -> dict[str, str]:
    evidence = _object(value, "evidence")
    _exact_fields(evidence, _EVIDENCE_OUTPUT_FIELDS, "evidence")
    artifact_id = _hash(evidence["artifact_id"], "evidence.artifact_id")
    content_hash = _hash(evidence["content_hash"], "evidence.content_hash")
    download_url = _required_text(
        evidence["download_url"],
        "evidence.download_url",
    )
    if (
        not download_url.startswith("/api/tasks/")
        or "/task-artifacts/" not in download_url
        or not download_url.endswith("/download")
    ):
        raise StrategyError("evidence.download_url is invalid")
    return {
        "artifact_id": artifact_id,
        "content_hash": content_hash,
        "download_url": download_url,
    }


def _counts(value: object, name: str) -> dict[str, int]:
    counts = _object(value, name)
    normalized: dict[str, int] = {}
    for key, count in counts.items():
        text = _required_text(key, f"{name} key")
        normalized[text] = _non_negative_int(count, f"{name}.{text}")
    if list(normalized) != sorted(normalized):
        raise StrategyError(f"{name} keys must be sorted")
    return normalized


def _inactive_flags(value: Mapping[str, Any]) -> dict[str, bool]:
    flags = {
        name: _boolean(value[name], name)
        for name in ("activated", "adopted", "deployed")
    }
    if any(flags.values()):
        raise StrategyError(
            "Strategy Pool apply cannot activate, adopt, or deploy"
        )
    return flags


def _require_cross_field_consistency(
    *,
    run_id: str,
    input_hash: str,
    source: Mapping[str, Any],
    result: Mapping[str, Any],
    columns: Mapping[str, str],
    action_counts: Mapping[str, int],
    rule_counts: Mapping[str, int],
    entry_counts: Mapping[str, int],
    default_count: int,
    workspace: Mapping[str, Any],
) -> None:
    del columns
    if run_id != f"strategy-pool-apply-{input_hash[:32]}":
        raise StrategyError("run_id does not match input_hash")
    row_count = source["row_count"]
    if (
        result["row_count"] != row_count
        or sum(action_counts.values()) != row_count
        or sum(rule_counts.values()) + default_count != row_count
        or sum(entry_counts.values()) + default_count != row_count
    ):
        raise StrategyError(
            "Strategy Pool apply counts do not conserve the source population"
        )
    if source["dataset_id"] == result["dataset_id"]:
        raise StrategyError(
            "Strategy Pool apply result must be a new derived dataset"
        )
    if workspace["active_dataset_id"] != source["dataset_id"]:
        raise StrategyError(
            "Strategy Pool apply must leave the source dataset active"
        )


def _exact_fields(
    value: Mapping[str, Any],
    allowed: frozenset[str],
    name: str,
    *,
    required: frozenset[str] | None = None,
) -> None:
    keys = set(value)
    required_fields = allowed if required is None else required
    missing = sorted(required_fields - keys)
    unsupported = sorted(keys - allowed)
    if missing or unsupported:
        details = []
        if missing:
            details.append("missing fields: " + ", ".join(missing))
        if unsupported:
            details.append("unsupported fields: " + ", ".join(unsupported))
        raise StrategyError(f"{name} has " + "; ".join(details))


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise StrategyError(f"{name} must be an object")
    return dict(value)


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise StrategyError(f"{name} must be a non-empty string")
    if value != value.strip():
        raise StrategyError(f"{name} must not contain surrounding whitespace")
    return value


def _hash(value: object, name: str) -> str:
    text = _required_text(value, name)
    if _HASH_RE.fullmatch(text) is None:
        raise StrategyError(f"{name} must be a lowercase SHA-256")
    return text


def _run_id(value: object) -> str:
    text = _required_text(value, "run_id")
    if _RUN_ID_RE.fullmatch(text) is None:
        raise StrategyError("run_id is invalid")
    return text


def _dataset_id(value: object, name: str) -> str:
    text = _required_text(value, name)
    if _DATASET_ID_RE.fullmatch(text) is None:
        raise StrategyError(f"{name} is invalid")
    return text


def _positive_int(
    value: object,
    name: str,
    *,
    allow_zero: bool = False,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StrategyError(f"{name} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise StrategyError(f"{name} must be {qualifier}")
    return value


def _non_negative_int(value: object, name: str) -> int:
    return _positive_int(value, name, allow_zero=True)


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise StrategyError(f"{name} must be a boolean")
    return value


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
        raise StrategyError(
            "Strategy Pool apply evidence must be canonical JSON"
        ) from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _json_clone(value: object):
    return json.loads(_canonical_json(value))


def _object_without_duplicate_keys(pairs):
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


__all__ = [
    "DEFAULT_OUTPUT_PREFIX",
    "EVIDENCE_ARTIFACT_KIND",
    "EVIDENCE_SCHEMA_VERSION",
    "ORIGIN_TOOL",
    "RESULT_DATASET_ROLE",
    "TOOL_SCHEMA_VERSION",
    "run_apply_strategy_pool",
    "validate_apply_strategy_pool_tool_output",
]
