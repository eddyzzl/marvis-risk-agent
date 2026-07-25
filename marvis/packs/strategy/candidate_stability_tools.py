"""Governed Tool boundary for monthly Strategy candidate stability evidence."""

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
import tempfile
from typing import Any
from urllib.parse import quote

import pandas as pd

from marvis.artifacts import ArtifactUnitOfWork
from marvis.domain import STRATEGY_TYPES
from marvis.files import sha256_file
from marvis.packs.strategy.candidate_fragment import (
    sample_context_hash_from_candidate_evidence,
)
from marvis.packs.strategy.candidate_stability import (
    CANDIDATE_STABILITY_MAX_ROWS,
    CANDIDATE_STABILITY_PRODUCER_VERSION,
    build_candidate_stability_artifact,
    candidate_stability_artifact_content_hash,
    canonical_candidate_stability_artifact_json,
    validate_candidate_stability_artifact,
)
from marvis.packs.strategy.candidate_asset_tools import (
    ASSET_ARTIFACT_KIND,
    ORIGIN_TOOL as ASSET_ORIGIN_TOOL,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.evaluator import (
    evaluate_expression_frame,
    evaluate_strategy_frame,
)
from marvis.packs.strategy.pool_tools import (
    StrategyCandidatePoolArtifactBinding,
    VerifiedUnivariateCandidateLineageBinding,
    load_current_strategy_candidate_pool_artifact,
    load_verified_univariate_candidate_lineage,
    require_strategy_candidate_pool_artifact_binding_on_connection,
    require_verified_univariate_candidate_lineage_on_connection,
)
from marvis.packs.strategy.sample_design_binding import (
    StrategySampleDesignExecutionBinding,
    bind_strategy_development_frame,
    load_strategy_sample_design_execution_binding,
    require_strategy_sample_design_execution_binding_on_connection,
    revalidate_strategy_sample_design_execution_binding,
)
from marvis.repositories.task_artifacts import (
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
)


TOOL_SCHEMA_VERSION = "strategy.measure-candidate-monthly-stability-tool.v1"
ARTIFACT_KIND = "strategy_candidate_monthly_stability_json"
ARTIFACT_SCHEMA_VERSION = "strategy.candidate-monthly-stability-artifact.v1"
ORIGIN_TOOL = "strategy.measure_candidate_monthly_stability"

_ASSET_INPUT_FIELDS = frozenset(
    {
        "source_kind",
        "source_artifact_id",
        "expected_artifact_content_hash",
        "expected_asset_id",
        "expected_asset_hash",
    }
)
_POOL_INPUT_FIELDS = frozenset(
    {
        "source_kind",
        "strategy_type",
        "expected_pool_revision",
        "expected_pool_snapshot_hash",
        "entry_id",
    }
)
_ASSET_POINTER_FIELDS = frozenset({"source_kind", "asset_id"})
_POOL_POINTER_FIELDS = frozenset({"source_kind", "strategy_type", "entry_id"})
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
_BOUNDARY_ERRORS = (
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
)


@dataclass(frozen=True)
class _ExecutionBinding:
    task_id: str
    source_kind: str
    basis: str
    candidate: VerifiedUnivariateCandidateLineageBinding
    pool: StrategyCandidatePoolArtifactBinding | None
    entry: dict[str, Any] | None
    sample_design: StrategySampleDesignExecutionBinding
    identity: dict[str, Any]
    source_ref: dict[str, Any]
    condition: dict[str, Any] | None
    strategy_spec: dict[str, Any] | None

    @property
    def dataset(self):
        return self.candidate.dataset


def resolve_candidate_monthly_stability_inputs(
    runtime,
    *,
    task_id: str,
    user_pointer: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve a human-friendly pointer to exact fail-closed Tool inputs.

    This is the single read-only preflight seam used by Agent planning.  It
    verifies the candidate/Pool lineage and the governed sample design, and
    refuses to construct a plan when the sample has no bound month field.
    """

    normalized_task = _text(task_id, "task_id")
    if not isinstance(user_pointer, Mapping):
        raise StrategyError("candidate stability pointer must be an object")
    source_kind = user_pointer.get("source_kind")
    if source_kind == "univariate_asset":
        _require_exact_fields(
            user_pointer,
            _ASSET_POINTER_FIELDS,
            "univariate candidate stability pointer",
        )
        asset_id = _text(user_pointer["asset_id"], "asset_id")
        with runtime.task_artifacts.transaction() as conn:
            rows = conn.execute(
                """
                SELECT id, content_hash, provenance_json
                  FROM task_artifacts
                 WHERE task_id = ?
                   AND kind = ?
                   AND origin_tool = ?
                   AND json_valid(provenance_json)
                   AND json_type(provenance_json, '$.asset_id') = 'text'
                   AND json_extract(provenance_json, '$.asset_id') = ?
                 ORDER BY created_at DESC, id DESC
                 LIMIT 2
                """,
                (
                    normalized_task,
                    ASSET_ARTIFACT_KIND,
                    ASSET_ORIGIN_TOOL,
                    asset_id,
                ),
            ).fetchall()
        if not rows:
            raise StrategyError(f"univariate candidate asset not found: {asset_id}")
        if len(rows) != 1:
            raise StrategyError(
                f"univariate candidate asset is ambiguous: {asset_id}"
            )
        try:
            provenance = json.loads(str(rows[0]["provenance_json"]))
        except (TypeError, json.JSONDecodeError) as exc:
            raise StrategyError("candidate asset provenance is invalid") from exc
        asset_hash = _hash(
            provenance.get("asset_hash"),
            "candidate asset provenance.asset_hash",
        )
        resolved = {
            "source_kind": "univariate_asset",
            "source_artifact_id": _hash(rows[0]["id"], "source_artifact_id"),
            "expected_artifact_content_hash": _hash(
                rows[0]["content_hash"],
                "expected_artifact_content_hash",
            ),
            "expected_asset_id": asset_id,
            "expected_asset_hash": asset_hash,
        }
    elif source_kind == "pool_entry":
        _require_exact_fields(
            user_pointer,
            _POOL_POINTER_FIELDS,
            "Pool-entry candidate stability pointer",
        )
        strategy_type = _text(user_pointer["strategy_type"], "strategy_type")
        entry_id = _text(user_pointer["entry_id"], "entry_id")
        pool = load_current_strategy_candidate_pool_artifact(
            runtime,
            task_id=normalized_task,
            strategy_type=strategy_type,
        )
        _select_univariate_pool_entry(runtime, pool, entry_id=entry_id)
        resolved = {
            "source_kind": "pool_entry",
            "strategy_type": strategy_type,
            "expected_pool_revision": pool.pool["revision"],
            "expected_pool_snapshot_hash": pool.pool["snapshot_hash"],
            "entry_id": entry_id,
        }
    else:
        raise StrategyError(
            "candidate stability pointer source_kind must be "
            "univariate_asset or pool_entry"
        )
    # Construction itself authenticates the sample design and month binding.
    _load_execution_binding(runtime, task_id=normalized_task, request=resolved)
    return resolved


def run_measure_candidate_monthly_stability(inputs, ctx, runtime) -> dict[str, Any]:
    """Measure and atomically publish monthly candidate hit-distribution PSI."""

    try:
        request = _validate_inputs(inputs)
        task_id = _text(ctx.task_id, "task_id")
        binding = _load_execution_binding(
            runtime,
            task_id=task_id,
            request=request,
        )
        frame = _read_development_frame(runtime, binding=binding)
        hit_mask = _evaluate_hit_mask(frame, binding=binding)
        artifact = build_candidate_stability_artifact(
            frame=frame,
            month_col=binding.sample_design.month_col,
            target_col=binding.sample_design.target_col,
            hit_mask=hit_mask,
            basis=binding.basis,
            identity=binding.identity,
            source_ref=binding.source_ref,
            sample_design_ref=binding.sample_design.to_ref_dict(),
        )
        artifact = validate_candidate_stability_artifact(artifact)
        _revalidate_before_registration(runtime, binding=binding)
        return _persist_artifact(
            runtime,
            task_id=task_id,
            binding=binding,
            artifact=artifact,
        )
    except StrategyError:
        raise
    except _BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc


def _validate_inputs(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StrategyError(
            "measure_candidate_monthly_stability inputs must be an object"
        )
    source_kind = value.get("source_kind")
    if source_kind == "univariate_asset":
        _require_exact_fields(
            value,
            _ASSET_INPUT_FIELDS,
            "univariate candidate stability inputs",
        )
        return {
            "source_kind": source_kind,
            "source_artifact_id": _hash(
                value["source_artifact_id"],
                "source_artifact_id",
            ),
            "expected_artifact_content_hash": _hash(
                value["expected_artifact_content_hash"],
                "expected_artifact_content_hash",
            ),
            "expected_asset_id": _text(
                value["expected_asset_id"],
                "expected_asset_id",
            ),
            "expected_asset_hash": _hash(
                value["expected_asset_hash"],
                "expected_asset_hash",
            ),
        }
    if source_kind == "pool_entry":
        _require_exact_fields(
            value,
            _POOL_INPUT_FIELDS,
            "Pool-entry candidate stability inputs",
        )
        revision = _positive_int(
            value["expected_pool_revision"],
            "expected_pool_revision",
        )
        strategy_type = _text(value["strategy_type"], "strategy_type")
        if strategy_type not in STRATEGY_TYPES:
            raise StrategyError("candidate stability strategy_type is invalid")
        return {
            "source_kind": source_kind,
            "strategy_type": strategy_type,
            "expected_pool_revision": revision,
            "expected_pool_snapshot_hash": _hash(
                value["expected_pool_snapshot_hash"],
                "expected_pool_snapshot_hash",
            ),
            "entry_id": _text(value["entry_id"], "entry_id"),
        }
    raise StrategyError(
        "measure_candidate_monthly_stability source_kind must be "
        "univariate_asset or pool_entry"
    )


def _load_execution_binding(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
) -> _ExecutionBinding:
    pool: StrategyCandidatePoolArtifactBinding | None = None
    entry: dict[str, Any] | None = None
    if request["source_kind"] == "univariate_asset":
        candidate = load_verified_univariate_candidate_lineage(
            runtime,
            task_id=task_id,
            artifact_id=request["source_artifact_id"],
            expected_content_hash=request["expected_artifact_content_hash"],
            expected_asset_id=request["expected_asset_id"],
            expected_asset_hash=request["expected_asset_hash"],
        )
        basis = "asset_rule_hit"
        condition = dict(candidate.asset["rule"]["condition"])
        strategy_spec = None
    else:
        pool = load_current_strategy_candidate_pool_artifact(
            runtime,
            task_id=task_id,
            strategy_type=request["strategy_type"],
            expected_pool_revision=request["expected_pool_revision"],
            expected_pool_snapshot_hash=request["expected_pool_snapshot_hash"],
        )
        entry, candidate = _select_univariate_pool_entry(
            runtime,
            pool,
            entry_id=request["entry_id"],
        )
        basis = "pool_entry_incremental_first_match"
        condition = None
        strategy_spec = dict(pool.compiled_design["strategy_spec"])

    evidence = candidate.evidence
    evidence_identity = evidence["identity"]
    sample_context_hash = sample_context_hash_from_candidate_evidence(evidence)
    identity = {
        "task_id": task_id,
        "dataset_id": evidence_identity["dataset_id"],
        "dataset_content_hash": evidence_identity["dataset_content_hash"],
        "workspace_revision": evidence_identity["workspace_revision"],
        "workspace_generation": evidence_identity["workspace_generation"],
        "semantic_mapping_hash": evidence_identity["semantic_mapping_hash"],
        "sample_context_hash": sample_context_hash,
    }
    if pool is not None:
        source_identity = entry["source"]["evidence_identity"]
        if source_identity != {
            key: identity[key]
            for key in (
                "dataset_id",
                "dataset_content_hash",
                "workspace_revision",
                "workspace_generation",
                "semantic_mapping_hash",
                "sample_context_hash",
            )
        }:
            raise StrategyError(
                "Pool entry evidence identity changed from its candidate lineage"
            )

    sample = _load_sample_design_binding(
        runtime,
        task_id=task_id,
        candidate=candidate,
    )
    if not sample.month_col:
        raise StrategyError(
            "candidate monthly stability requires a month field in the "
            "governed StrategySampleDesign"
        )
    if pool is None:
        source_ref = {
            "artifact_id": candidate.lineage.asset_record.artifact_id,
            "artifact_content_hash": candidate.lineage.asset_record.content_hash,
            "source_kind": "univariate_asset",
            "asset_id": candidate.asset["asset_id"],
            "asset_hash": candidate.asset["asset_hash"],
            "rule_id": candidate.asset["rule"]["rule_id"],
        }
    else:
        source_ref = {
            "artifact_id": pool.artifact_id,
            "artifact_content_hash": pool.artifact_content_hash,
            "source_kind": "pool_entry",
            "pool_id": pool.pool["pool_id"],
            "snapshot_hash": pool.pool["snapshot_hash"],
            "rule_id": entry["rule_id"],
            "entry_id": entry["entry_id"],
            "revision": pool.pool["revision"],
            "revision_id": pool.pool["revision_id"],
        }
    return _ExecutionBinding(
        task_id=task_id,
        source_kind=request["source_kind"],
        basis=basis,
        candidate=candidate,
        pool=pool,
        entry=entry,
        sample_design=sample,
        identity=identity,
        source_ref=source_ref,
        condition=condition,
        strategy_spec=strategy_spec,
    )


def _select_univariate_pool_entry(
    runtime,
    pool: StrategyCandidatePoolArtifactBinding,
    *,
    entry_id: str,
) -> tuple[dict[str, Any], VerifiedUnivariateCandidateLineageBinding]:
    matches = [
        (index, entry)
        for index, entry in enumerate(pool.pool["entries"])
        if entry["entry_id"] == entry_id
    ]
    if len(matches) != 1:
        raise StrategyError(f"unknown or ambiguous Strategy Pool entry: {entry_id}")
    index, entry = matches[0]
    if entry["source"]["asset_type"] != "univariate_refinement":
        raise StrategyError(
            "candidate stability Pool entry currently supports "
            "univariate_refinement only"
        )
    lineage = pool.lineages[index]
    # The public loader replays the exact concrete artifact independently of
    # the Pool adapter, then both projections must agree byte-for-byte.
    candidate = load_verified_univariate_candidate_lineage(
        runtime,
        task_id=pool.task_id,
        artifact_id=entry["source"]["artifact_id"],
        expected_content_hash=entry["source"]["artifact_content_hash"],
        expected_asset_id=entry["source"]["asset_id"],
        expected_asset_hash=entry["source"]["asset_hash"],
    )
    if candidate.lineage != lineage or candidate.source_binding != entry["source"]:
        raise StrategyError("Strategy Pool entry candidate lineage changed")
    return dict(entry), candidate


def _load_sample_design_binding(
    runtime,
    *,
    task_id: str,
    candidate: VerifiedUnivariateCandidateLineageBinding,
) -> StrategySampleDesignExecutionBinding:
    evidence = candidate.evidence
    generation = evidence["generation"]["parameters"]
    identity = evidence["identity"]
    target_col = evidence["analysis"]["target"]
    drop_nan_labels = generation.get("drop_nan_labels")
    if not isinstance(drop_nan_labels, bool):
        raise StrategyError("candidate drop_nan_labels binding is invalid")
    return load_strategy_sample_design_execution_binding(
        runtime,
        task_id=task_id,
        sample_design_ref=generation.get("sample_design_ref"),
        dataset_id=identity["dataset_id"],
        dataset_content_hash=identity["dataset_content_hash"],
        workspace_revision=identity["workspace_revision"],
        workspace_generation=identity["workspace_generation"],
        semantic_mapping_hash=identity["semantic_mapping_hash"],
        target_col=target_col,
        drop_nan_labels=drop_nan_labels,
    )


def _read_development_frame(runtime, *, binding: _ExecutionBinding) -> pd.DataFrame:
    dataset = binding.dataset
    if (
        dataset.row_count > CANDIDATE_STABILITY_MAX_ROWS
        or binding.sample_design.active_population_count
        > CANDIDATE_STABILITY_MAX_ROWS
        or binding.sample_design.development_population_count
        > CANDIDATE_STABILITY_MAX_ROWS
    ):
        raise StrategyError(
            "candidate stability exceeds the 1,000,000-row read budget"
        )
    fields = _expression_fields(
        binding.condition if binding.condition is not None else binding.strategy_spec
    )
    fields.add(binding.sample_design.target_col)
    fields.add(binding.sample_design.month_col)
    if binding.sample_design.split_column is not None:
        fields.add(binding.sample_design.split_column)
    unknown = sorted(fields - set(dataset.columns))
    if unknown:
        raise StrategyError(
            "candidate stability source references missing columns: "
            + ", ".join(unknown)
        )
    frame = _read_authenticated_parquet_snapshot(
        dataset.path,
        root=Path(runtime.settings.datasets_dir).absolute(),
        expected_content_hash=dataset.content_hash,
        columns=sorted(fields),
    )
    if len(frame) != dataset.row_count:
        raise StrategyError("candidate stability dataset row count changed")
    return bind_strategy_development_frame(
        frame,
        binding=binding.sample_design,
    )


def _evaluate_hit_mask(
    frame: pd.DataFrame,
    *,
    binding: _ExecutionBinding,
) -> pd.Series:
    if binding.condition is not None:
        hit_mask = evaluate_expression_frame(frame, binding.condition)
    else:
        if binding.strategy_spec is None or binding.entry is None:
            raise StrategyError("candidate stability Pool binding is incomplete")
        evaluation = evaluate_strategy_frame(frame, binding.strategy_spec)
        hit_mask = evaluation.matched_rule_id.eq(binding.entry["rule_id"])
    return pd.Series(
        hit_mask.to_numpy(dtype=bool, copy=False),
        index=frame.index,
        dtype=bool,
        name="candidate_hit",
    )


def _revalidate_before_registration(runtime, *, binding: _ExecutionBinding) -> None:
    _require_dataset_bytes(binding.dataset)
    if (
        revalidate_strategy_sample_design_execution_binding(
            runtime,
            binding.sample_design,
        )
        != binding.sample_design
    ):
        raise StrategyError(
            "candidate stability sample design changed during measurement"
        )
    reloaded_candidate = load_verified_univariate_candidate_lineage(
        runtime,
        task_id=binding.task_id,
        artifact_id=binding.candidate.lineage.asset_record.artifact_id,
        expected_content_hash=binding.candidate.lineage.asset_record.content_hash,
        expected_asset_id=binding.candidate.asset["asset_id"],
        expected_asset_hash=binding.candidate.asset["asset_hash"],
    )
    if reloaded_candidate != binding.candidate:
        raise StrategyError(
            "candidate stability source changed during measurement"
        )
    if binding.pool is not None:
        reloaded_pool = load_current_strategy_candidate_pool_artifact(
            runtime,
            task_id=binding.task_id,
            strategy_type=binding.pool.strategy_type,
            expected_pool_revision=binding.pool.pool["revision"],
            expected_pool_snapshot_hash=binding.pool.pool["snapshot_hash"],
            expected_artifact_id=binding.pool.artifact_id,
            expected_artifact_content_hash=binding.pool.artifact_content_hash,
        )
        if reloaded_pool != binding.pool:
            raise StrategyError(
                "Strategy Pool changed during candidate stability measurement"
            )


def _persist_artifact(
    runtime,
    *,
    task_id: str,
    binding: _ExecutionBinding,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = validate_candidate_stability_artifact(artifact)
    canonical = canonical_candidate_stability_artifact_json(normalized).encode("utf-8")
    artifact_hash = hashlib.sha256(canonical).hexdigest()
    if not hmac.compare_digest(
        candidate_stability_artifact_content_hash(normalized),
        normalized["content_hash"],
    ):
        raise StrategyError("candidate stability content hash is inconsistent")
    stability_id = _safe_id(normalized["stability_id"], "stability_id")
    out_dir = _prepare_output_directory(
        runtime.settings.tasks_dir,
        task_id=task_id,
    )
    final_path = out_dir / f"{stability_id}.json"
    provenance = _artifact_provenance(
        task_id=task_id,
        binding=binding,
        artifact=normalized,
    )
    uow = ArtifactUnitOfWork()
    staged = uow.stage_file(out_dir, final_path.name)
    try:
        staged.path.write_bytes(canonical)
    except OSError as exc:
        uow.rollback()
        raise StrategyError(
            "candidate stability artifact could not be staged"
        ) from exc
    db_committed = False
    rollback_under_lock = False
    reused = False
    record: Mapping[str, Any]
    try:
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                require_verified_univariate_candidate_lineage_on_connection(
                    conn,
                    binding.candidate,
                )
                require_strategy_sample_design_execution_binding_on_connection(
                    conn,
                    binding.sample_design,
                )
                if binding.pool is not None:
                    require_strategy_candidate_pool_artifact_binding_on_connection(
                        conn,
                        binding.pool,
                    )
                _require_dataset_bytes(binding.dataset)
                row = conn.execute(
                    """
                    SELECT id, task_id, kind, path, content_hash, origin_tool,
                           provenance_json
                      FROM task_artifacts
                     WHERE task_id = ? AND kind = ? AND path = ?
                    """,
                    (task_id, ARTIFACT_KIND, str(final_path)),
                ).fetchone()
                if row is not None:
                    _require_existing_artifact(
                        row,
                        task_id=task_id,
                        path=final_path,
                        content=canonical,
                        content_hash=artifact_hash,
                        provenance=provenance,
                    )
                    uow.rollback()
                    reused = True
                else:
                    if final_path.exists() or final_path.is_symlink():
                        raise StrategyError(
                            "candidate stability path exists without a registry row"
                        )
                    uow.promote_all()
                    _verify_artifact_file(
                        final_path,
                        root=Path(runtime.settings.tasks_dir),
                        content=canonical,
                        content_hash=artifact_hash,
                    )
                record = runtime.task_artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=ARTIFACT_KIND,
                    path=str(final_path),
                    content_hash=artifact_hash,
                    origin_tool=ORIGIN_TOOL,
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
        normalized,
        task_id=task_id,
        record=record,
        path=final_path,
    )


def _artifact_provenance(
    *,
    task_id: str,
    binding: _ExecutionBinding,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    sample = binding.sample_design
    source = artifact["source_ref"]
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "producer_version": CANDIDATE_STABILITY_PRODUCER_VERSION,
        "task_id": task_id,
        "stability_id": artifact["stability_id"],
        "stability_content_hash": artifact["content_hash"],
        "basis": binding.basis,
        "source_kind": source["source_kind"],
        "source_artifact_id": source["artifact_id"],
        "source_artifact_content_hash": source["artifact_content_hash"],
        "source_id": (
            source["asset_id"]
            if binding.source_kind == "univariate_asset"
            else source["pool_id"]
        ),
        "source_hash": (
            source["asset_hash"]
            if binding.source_kind == "univariate_asset"
            else source["snapshot_hash"]
        ),
        "rule_id": source["rule_id"],
        "entry_id": source.get("entry_id"),
        "pool_id": source.get("pool_id"),
        "pool_revision": source.get("revision"),
        "pool_revision_id": source.get("revision_id"),
        "dataset_id": sample.dataset_id,
        "dataset_content_hash": sample.dataset_content_hash,
        "workspace_revision": sample.workspace_revision,
        "workspace_generation": sample.workspace_generation,
        "semantic_mapping_hash": sample.semantic_mapping_hash,
        "target_col": sample.target_col,
        "month_col": sample.month_col,
        "sample_design_ref": sample.to_ref_dict(),
        "sample_context_hash": binding.identity["sample_context_hash"],
        "sample_partition": sample.reference.partition,
    }


def _tool_output(
    artifact: Mapping[str, Any],
    *,
    task_id: str,
    record: Mapping[str, Any],
    path: Path,
) -> dict[str, Any]:
    summary = artifact["summary"]
    lifecycle = artifact["lifecycle"]
    red_flags = artifact["red_flags"]
    return {
        "schema_version": TOOL_SCHEMA_VERSION,
        "stability_id": artifact["stability_id"],
        "content_hash": artifact["content_hash"],
        "basis": artifact["basis"],
        "source_kind": artifact["source_ref"]["source_kind"],
        "month_col": artifact["bindings"]["month_col"],
        "population_count": summary["population_count"],
        "month_count": summary["month_count"],
        "max_psi": summary["max_psi"],
        "stability": dict(artifact),
        "warnings": [
            (
                f"month {flag['month']} has {flag['observed_rows']} rows, "
                f"below minimum {flag['minimum_rows']}"
            )
            for flag in red_flags
        ],
        "artifacts": [
            {
                "artifact_id": str(record["id"]),
                "kind": ARTIFACT_KIND,
                "format": "json",
                "filename": path.name,
                "content_hash": str(record["content_hash"]),
                "download_url": (
                    f"/api/tasks/{quote(task_id, safe='')}"
                    f"/task-artifacts/{quote(str(record['id']), safe='')}/download"
                ),
            }
        ],
        "not_created_strategy": lifecycle["not_created_strategy"],
        "not_adopted": lifecycle["not_adopted"],
        "not_deployed": lifecycle["not_deployed"],
    }


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
            raise StrategyError(
                "candidate stability artifact directory escaped task storage"
            )
    except OSError as exc:
        raise StrategyError(
            "candidate stability artifact directory is unavailable"
        ) from exc
    out_dir = task_dir / "strategy_candidate_stability"
    if out_dir.exists() and (out_dir.is_symlink() or not out_dir.is_dir()):
        raise StrategyError(
            "candidate stability artifact path must be a regular directory"
        )
    out_dir.mkdir(exist_ok=True)
    if (
        out_dir.is_symlink()
        or out_dir.resolve(strict=True).parent != task_dir.resolve(strict=True)
    ):
        raise StrategyError(
            "candidate stability artifact directory escaped task storage"
        )
    return out_dir


def _require_existing_artifact(
    row,
    *,
    task_id: str,
    path: Path,
    content: bytes,
    content_hash: str,
    provenance: Mapping[str, Any],
) -> None:
    expected = {
        "task_id": task_id,
        "kind": ARTIFACT_KIND,
        "path": str(path),
        "content_hash": content_hash,
        "origin_tool": ORIGIN_TOOL,
        "provenance_json": _canonical_json(provenance),
    }
    for field, expected_value in expected.items():
        if str(row[field]) != expected_value:
            raise StrategyError(
                "existing candidate stability artifact registry row changed"
            )
    _verify_artifact_file(
        path,
        root=path.parents[2],
        content=content,
        content_hash=content_hash,
    )


def _verify_artifact_file(
    path: Path,
    *,
    root: Path,
    content: bytes,
    content_hash: str,
) -> None:
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
    ):
        raise StrategyError("candidate stability artifact must be a regular file")
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
        raw = path.read_bytes()
    except (OSError, ValueError) as exc:
        raise StrategyError(
            "candidate stability artifact is unavailable"
        ) from exc
    if (
        raw != content
        or not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), content_hash)
    ):
        raise StrategyError("candidate stability artifact bytes changed")


def _require_dataset_bytes(dataset) -> None:
    try:
        live_hash = sha256_file(dataset.path)
    except OSError as exc:
        raise StrategyError(
            "candidate stability source dataset is unavailable"
        ) from exc
    if not hmac.compare_digest(live_hash, dataset.content_hash):
        raise StrategyError("candidate stability source dataset bytes changed")


def _read_authenticated_parquet_snapshot(
    path: Path,
    *,
    root: Path,
    expected_content_hash: str,
    columns: list[str],
) -> pd.DataFrame:
    """Read only bytes copied from one authenticated, retained source fd."""

    _require_dataset_path(path, root=root)
    source_fd = -1
    snapshot = None
    try:
        before = os.lstat(path)
        if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
            raise StrategyError(
                "candidate stability dataset must be a regular file"
            )
        flags = (
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        source_fd = os.open(path, flags)
        opened = os.fstat(source_fd)
        after_open = os.lstat(path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_ISLNK(after_open.st_mode)
            or _file_identity(before) != _file_identity(opened)
            or _file_identity(opened) != _file_identity(after_open)
            or _stable_file_stat(before) != _stable_file_stat(opened)
            or _stable_file_stat(opened) != _stable_file_stat(after_open)
        ):
            raise StrategyError(
                "candidate stability dataset changed while opening"
            )

        snapshot = tempfile.TemporaryFile(mode="w+b", dir=root)
        digest = hashlib.sha256()
        copied = 0
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            copied += len(chunk)
            snapshot.write(chunk)
        snapshot.flush()
        source_after_copy = os.fstat(source_fd)
        if (
            _stable_file_stat(source_after_copy)
            != _stable_file_stat(opened)
            or copied != int(opened.st_size)
            or not hmac.compare_digest(
                digest.hexdigest(),
                expected_content_hash,
            )
        ):
            raise StrategyError(
                "candidate stability dataset bytes changed before replay"
            )

        snapshot_stat = os.fstat(snapshot.fileno())
        if int(snapshot_stat.st_size) != copied:
            raise StrategyError(
                "candidate stability private dataset snapshot is incomplete"
            )
        snapshot.seek(0)
        frame = pd.read_parquet(snapshot, columns=columns)
        snapshot_after_read = os.fstat(snapshot.fileno())
        current = os.lstat(path)
        if (
            _stable_file_stat(snapshot_after_read)
            != _stable_file_stat(snapshot_stat)
            or _stable_file_stat(os.fstat(source_fd))
            != _stable_file_stat(opened)
            or stat.S_ISLNK(current.st_mode)
            or _stable_file_stat(current) != _stable_file_stat(opened)
        ):
            raise StrategyError(
                "candidate stability dataset changed during replay"
            )
        return frame
    except StrategyError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise StrategyError(
            "candidate stability dataset could not be read"
        ) from exc
    finally:
        if snapshot is not None:
            snapshot.close()
        if source_fd >= 0:
            os.close(source_fd)


def _file_identity(value: os.stat_result) -> tuple[int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(stat.S_IFMT(value.st_mode)),
    )


def _stable_file_stat(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(stat.S_IFMT(value.st_mode)),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _require_dataset_path(path: Path, *, root: Path) -> None:
    resolved_root = root.absolute()
    if (
        not path.is_absolute()
        or resolved_root.is_symlink()
        or path.is_symlink()
        or not path.is_file()
    ):
        raise StrategyError(
            "candidate stability dataset must be a regular file"
        )
    current = path.parent
    while current != resolved_root:
        if current.is_symlink():
            raise StrategyError(
                "candidate stability dataset path traverses a symlink"
            )
        if current == current.parent:
            break
        current = current.parent
    try:
        path.resolve(strict=True).relative_to(
            resolved_root.resolve(strict=True)
        )
    except (OSError, ValueError) as exc:
        raise StrategyError(
            "candidate stability dataset escaped dataset storage"
        ) from exc


def _expression_fields(value: object) -> set[str]:
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


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    if any(not isinstance(key, str) for key in value):
        raise StrategyError(f"{name} keys must be strings")
    if set(value) != expected:
        missing = sorted(expected - set(value))
        unexpected = sorted(set(value) - expected)
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unsupported: " + ", ".join(unexpected))
        raise StrategyError(f"{name} has invalid fields ({'; '.join(details)})")


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise StrategyError("candidate stability value is not canonical JSON") from exc


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise StrategyError(f"{name} must be non-empty text")
    return value.strip()


def _hash(value: object, name: str) -> str:
    normalized = _text(value, name)
    if _HASH_RE.fullmatch(normalized) is None:
        raise StrategyError(f"{name} must be a lowercase SHA-256 hash")
    return normalized


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise StrategyError(f"{name} must be a positive integer")
    return value


def _safe_id(value: object, name: str) -> str:
    normalized = _text(value, name)
    if _SAFE_ID_RE.fullmatch(normalized) is None:
        raise StrategyError(f"{name} is not safe for an artifact filename")
    return normalized


__all__ = [
    "ARTIFACT_KIND",
    "ARTIFACT_SCHEMA_VERSION",
    "ORIGIN_TOOL",
    "TOOL_SCHEMA_VERSION",
    "resolve_candidate_monthly_stability_inputs",
    "run_measure_candidate_monthly_stability",
]
