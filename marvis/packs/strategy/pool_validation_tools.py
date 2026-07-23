"""Governed Tool boundary for Strategy Pool validation/OOT replay evidence."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd

from marvis.artifacts import ArtifactUnitOfWork
from marvis.artifacts.transactional import ArtifactTransactionError
from marvis.files import sha256_file
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pool_tools import (
    StrategyCandidatePoolArtifactBinding,
    load_current_strategy_candidate_pool_artifact,
    require_strategy_candidate_pool_artifact_binding_on_connection,
)
from marvis.packs.strategy.pool_validation import (
    STRATEGY_POOL_VALIDATION_PRODUCER_VERSION,
    build_strategy_pool_validation_evidence,
    canonical_strategy_pool_validation_json,
    validate_strategy_pool_validation_evidence,
)
from marvis.packs.strategy.sample_design_binding import StrategySampleDesignRef
from marvis.packs.strategy.sample_design_v2_tools import (
    StrategySampleDesignV2ArtifactBinding,
    load_strategy_sample_design_v2_artifacts,
    require_strategy_sample_design_v2_artifact_binding_on_connection,
)
from marvis.repositories.task_artifacts import (
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
)


POOL_VALIDATION_TOOL_SCHEMA_VERSION = (
    "strategy.measure-pool-validation-tool.v1"
)
POOL_VALIDATION_ARTIFACT_KIND = "strategy_pool_validation_json"
POOL_VALIDATION_ARTIFACT_SCHEMA_VERSION = (
    "strategy.pool-validation-artifact.v1"
)
POOL_VALIDATION_ORIGIN_TOOL = "strategy.measure_strategy_pool_validation"

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_POOL_ID_RE = re.compile(r"^strategy-pool-[0-9a-f]{32}$")
_POOL_REVISION_ID_RE = re.compile(
    r"^strategy-pool-revision-[0-9a-f]{32}$"
)
_INPUT_FIELDS = frozenset(
    {
        "strategy_type",
        "pool_ref",
        "sample_design_ref",
        "partition",
        "population",
        "comparison_mode",
    }
)
_POOL_REF_FIELDS = frozenset(
    {
        "artifact_id",
        "expected_artifact_content_hash",
        "expected_pool_id",
        "expected_revision",
        "expected_revision_id",
        "expected_snapshot_hash",
    }
)
_SAMPLE_DESIGN_REF_FIELDS = frozenset(
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
_OUTPUT_FIELDS = frozenset(
    {
        "schema_version",
        "evidence_id",
        "content_hash",
        "pool_id",
        "pool_revision",
        "pool_snapshot_hash",
        "partition",
        "population",
        "comparison_mode",
        "lifecycle_stage",
        "validation_status",
        "population_count",
        "labeled_count",
        "unlabeled_count",
        "evidence",
        "warnings",
        "artifact",
        "not_mutated_pool",
        "not_created_strategy",
        "not_adopted",
        "not_promoted",
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
_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "task_id",
        "evidence_id",
        "evidence_content_hash",
        "pool_ref",
        "sample_design_ref",
        "dataset_binding",
        "target_binding",
        "field_bindings",
        "partition",
        "population",
        "comparison_mode",
        "lifecycle_stage",
        "validation_status",
    }
)
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
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_BOUNDARY_ERRORS = (
    ArtifactTransactionError,
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
)


def run_measure_strategy_pool_validation(inputs, ctx, runtime) -> dict[str, Any]:
    """Replay one exact current Pool on V2 risk validation/OOT membership."""

    try:
        request = _validate_inputs(inputs)
        task_id = _text(ctx.task_id, "task_id")
        pool = _load_pool_binding(
            runtime,
            task_id=task_id,
            request=request,
        )
        sample = _load_sample_design_binding(
            runtime,
            task_id=task_id,
            request=request,
        )
        semantics = _require_independent_sample_contract(
            pool=pool,
            sample=sample,
            partition=request["partition"],
        )
        _require_bindings_under_lock(runtime, pool=pool, sample=sample)
        selected = _read_selected_partition(
            runtime,
            pool=pool,
            sample=sample,
            partition=request["partition"],
            target_col=semantics["target_col"],
            month_col=semantics["month_col"],
            loan_amount_col=semantics["loan_amount_col"],
            overdue_amount_col=semantics["overdue_amount_col"],
        )
        evidence = build_strategy_pool_validation_evidence(
            pool=pool.pool,
            frame=selected,
            pool_artifact_ref={
                "artifact_id": pool.artifact_id,
                "artifact_content_hash": pool.artifact_content_hash,
            },
            sample_design_v2_ref=_sample_design_evidence_ref(
                sample,
                partition=request["partition"],
            ),
            dataset_binding=_dataset_evidence_binding(sample),
            legacy_development_ref=semantics["legacy_development_ref"],
            partition=request["partition"],
            population="risk",
            comparison_mode="absolute",
            target_col=semantics["target_col"],
            target_bad_value=semantics["target_bad_value"],
            month_col=semantics["month_col"],
            loan_amount_col=semantics["loan_amount_col"],
            overdue_amount_col=semantics["overdue_amount_col"],
            development_rows_excluded=True,
        )
        if (
            sha256_file(sample.source_binding.dataset_path)
            != sample.source_binding.dataset_content_hash
        ):
            raise StrategyError(
                "Strategy Pool validation dataset changed during computation"
            )
        return _persist_evidence(
            runtime,
            task_id=task_id,
            request=request,
            pool=pool,
            sample=sample,
            evidence=evidence,
        )
    except StrategyError:
        raise
    except _BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc


def validate_measure_strategy_pool_validation_tool_output(
    value: object,
) -> dict[str, Any]:
    """Reconstruct every display scalar from canonical embedded evidence."""

    obj = _json_object(value, "measure_strategy_pool_validation output")
    _exact_fields(
        obj,
        _OUTPUT_FIELDS,
        "measure_strategy_pool_validation output",
    )
    evidence = validate_strategy_pool_validation_evidence(obj["evidence"])
    identity = evidence["identity"]
    population = evidence["population_metrics"]
    expected = {
        "schema_version": POOL_VALIDATION_TOOL_SCHEMA_VERSION,
        "evidence_id": evidence["evidence_id"],
        "content_hash": evidence["content_hash"],
        "pool_id": identity["pool_id"],
        "pool_revision": identity["revision"],
        "pool_snapshot_hash": identity["snapshot_hash"],
        "partition": evidence["partition"],
        "population": evidence["population"],
        "comparison_mode": evidence["comparison_mode"],
        "lifecycle_stage": evidence["lifecycle"]["stage"],
        "validation_status": evidence["lifecycle"]["validation_status"],
        "population_count": population["population_count"],
        "labeled_count": population["labelled_count"],
        "unlabeled_count": population["unlabelled_count"],
    }
    for field, expected_value in expected.items():
        if obj[field] != expected_value:
            raise StrategyError(
                f"measure_strategy_pool_validation output {field} drifted"
            )
    warnings = [
        str(flag["message"])
        for flag in evidence["red_flags"]
        if flag.get("level") in {"amber", "red"}
    ]
    if obj["warnings"] != warnings:
        raise StrategyError(
            "measure_strategy_pool_validation output warnings drifted"
        )
    for field in (
        "not_mutated_pool",
        "not_created_strategy",
        "not_adopted",
        "not_promoted",
        "not_deployed",
    ):
        if obj[field] is not True:
            raise StrategyError(
                f"measure_strategy_pool_validation output {field} must be true"
            )

    artifact = _json_object(
        obj["artifact"],
        "measure_strategy_pool_validation artifact",
    )
    _exact_fields(
        artifact,
        _OUTPUT_ARTIFACT_FIELDS,
        "measure_strategy_pool_validation artifact",
    )
    artifact_id = _hash(artifact["artifact_id"], "artifact_id")
    canonical = canonical_strategy_pool_validation_json(evidence).encode(
        "utf-8"
    )
    expected_artifact = {
        "artifact_id": artifact_id,
        "kind": POOL_VALIDATION_ARTIFACT_KIND,
        "format": "json",
        "filename": f"{evidence['evidence_id']}.json",
        "content_hash": hashlib.sha256(canonical).hexdigest(),
        "download_url": (
            f"/api/tasks/{quote(identity['task_id'], safe='')}"
            f"/task-artifacts/{quote(artifact_id, safe='')}/download"
        ),
    }
    if artifact != expected_artifact:
        raise StrategyError(
            "measure_strategy_pool_validation artifact binding drifted"
        )
    obj["evidence"] = evidence
    return obj


def _validate_inputs(value: object) -> dict[str, Any]:
    obj = _json_object(value, "measure_strategy_pool_validation inputs")
    _exact_fields(
        obj,
        _INPUT_FIELDS,
        "measure_strategy_pool_validation inputs",
    )
    strategy_type = _text(obj["strategy_type"], "strategy_type")
    if strategy_type not in {"approval", "reject"}:
        raise StrategyError(
            "Strategy Pool validation supports approval/reject only"
        )
    partition = _text(obj["partition"], "partition")
    if partition not in {"validation", "oot"}:
        raise StrategyError("partition must be validation or oot")
    if obj["population"] != "risk":
        raise StrategyError("population must be risk")
    if obj["comparison_mode"] != "absolute":
        raise StrategyError("comparison_mode must be absolute")
    return {
        "strategy_type": strategy_type,
        "pool_ref": _validate_pool_ref(obj["pool_ref"]),
        "sample_design_ref": _validate_sample_design_ref(
            obj["sample_design_ref"]
        ),
        "partition": partition,
        "population": "risk",
        "comparison_mode": "absolute",
    }


def _validate_pool_ref(value: object) -> dict[str, Any]:
    obj = _json_object(value, "pool_ref")
    _exact_fields(obj, _POOL_REF_FIELDS, "pool_ref")
    pool_id = _text(obj["expected_pool_id"], "pool_ref.expected_pool_id")
    if _POOL_ID_RE.fullmatch(pool_id) is None:
        raise StrategyError("pool_ref.expected_pool_id is invalid")
    revision_id = _text(
        obj["expected_revision_id"],
        "pool_ref.expected_revision_id",
    )
    if _POOL_REVISION_ID_RE.fullmatch(revision_id) is None:
        raise StrategyError("pool_ref.expected_revision_id is invalid")
    return {
        "artifact_id": _hash(
            obj["artifact_id"],
            "pool_ref.artifact_id",
        ),
        "expected_artifact_content_hash": _hash(
            obj["expected_artifact_content_hash"],
            "pool_ref.expected_artifact_content_hash",
        ),
        "expected_pool_id": pool_id,
        "expected_revision": _positive_int(
            obj["expected_revision"],
            "pool_ref.expected_revision",
        ),
        "expected_revision_id": revision_id,
        "expected_snapshot_hash": _hash(
            obj["expected_snapshot_hash"],
            "pool_ref.expected_snapshot_hash",
        ),
    }


def _validate_sample_design_ref(value: object) -> dict[str, str]:
    obj = _json_object(value, "sample_design_ref")
    _exact_fields(
        obj,
        _SAMPLE_DESIGN_REF_FIELDS,
        "sample_design_ref",
    )
    result: dict[str, str] = {}
    for field in (
        "membership_artifact_id",
        "expected_membership_artifact_content_hash",
        "bundle_artifact_id",
        "expected_bundle_artifact_content_hash",
        "expected_sample_design_content_hash",
    ):
        result[field] = _hash(obj[field], f"sample_design_ref.{field}")
    for field in ("expected_bundle_id", "expected_sample_design_id"):
        result[field] = _text(obj[field], f"sample_design_ref.{field}")
    return result


def _load_pool_binding(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
) -> StrategyCandidatePoolArtifactBinding:
    ref = request["pool_ref"]
    binding = load_current_strategy_candidate_pool_artifact(
        runtime,
        task_id=task_id,
        strategy_type=request["strategy_type"],
        expected_pool_revision=ref["expected_revision"],
        expected_pool_snapshot_hash=ref["expected_snapshot_hash"],
        expected_artifact_id=ref["artifact_id"],
        expected_artifact_content_hash=ref[
            "expected_artifact_content_hash"
        ],
    )
    if (
        binding.pool["pool_id"] != ref["expected_pool_id"]
        or binding.pool["revision_id"] != ref["expected_revision_id"]
    ):
        raise StrategyError("current Strategy Pool identity changed")
    if not binding.pool["entries"]:
        raise StrategyError("cannot validate an empty Strategy Pool")
    if binding.compiled_design["requirements"]:
        raise StrategyError(
            "Strategy Pool validation cannot execute unresolved requirements"
        )
    return binding


def _load_sample_design_binding(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
) -> StrategySampleDesignV2ArtifactBinding:
    ref = request["sample_design_ref"]
    return load_strategy_sample_design_v2_artifacts(
        runtime,
        task_id=task_id,
        membership_artifact_id=ref["membership_artifact_id"],
        expected_membership_artifact_content_hash=ref[
            "expected_membership_artifact_content_hash"
        ],
        bundle_artifact_id=ref["bundle_artifact_id"],
        expected_bundle_artifact_content_hash=ref[
            "expected_bundle_artifact_content_hash"
        ],
        expected_bundle_id=ref["expected_bundle_id"],
        expected_sample_design_id=ref["expected_sample_design_id"],
        expected_sample_design_content_hash=ref[
            "expected_sample_design_content_hash"
        ],
    )


def _require_independent_sample_contract(
    *,
    pool: StrategyCandidatePoolArtifactBinding,
    sample: StrategySampleDesignV2ArtifactBinding,
    partition: str,
) -> dict[str, Any]:
    from marvis.packs.strategy import pool_impact_tools

    design = sample.bundle["sample_design"]
    target = design["target_selector"]
    if target["status"] != "resolved":
        raise StrategyError(
            "Strategy Pool validation requires a resolved V2 target selector"
        )
    if design["sample_semantics"]["scope"] != "strategy_development":
        raise StrategyError(
            "Strategy Pool validation requires governed strategy scope"
        )
    risk = next(
        item
        for item in sample.bundle["populations"]
        if item["role"] == "risk"
    )
    if risk["maturity_evidence"]["status"] != "confirmed_matured":
        raise StrategyError(
            "Strategy Pool validation requires confirmed_matured risk outcomes"
        )
    partition_count = sample.membership["header"]["counts"]["risk"][
        partition
    ]
    if partition_count == 0:
        raise StrategyError(f"Strategy Pool {partition} partition is empty")

    legacy_ref = StrategySampleDesignRef.from_value(
        design["compatibility"]["legacy_development_ref"]
    )
    if legacy_ref != sample.source_binding.legacy.reference:
        raise StrategyError(
            "StrategySampleDesign V2 legacy development mapping changed"
        )
    for lineage in pool.lineages:
        if pool_impact_tools._lineage_sample_design_ref(
            lineage
        ) != legacy_ref:
            raise StrategyError(
                "Strategy Pool source lineage does not match the exact "
                "StrategySampleDesign V2 legacy development ref"
            )
        if pool_impact_tools._lineage_target_col(lineage) != target["column"]:
            raise StrategyError(
                "Strategy Pool source target does not match "
                "StrategySampleDesign V2"
            )
    if (
        sample.source_binding.legacy.target_col != target["column"]
        or sample.source_binding.legacy.target_bad_value
        != target["bad_value"]
    ):
        raise StrategyError(
            "StrategySampleDesign V2 target polarity changed from legacy lineage"
        )
    fields = design["sample_semantics"]["field_bindings"]
    return {
        "legacy_development_ref": legacy_ref.to_ref_dict(),
        "target_col": target["column"],
        "target_bad_value": target["bad_value"],
        "month_col": fields["month_field"],
        "loan_amount_col": fields["loan_amount_field"],
        "overdue_amount_col": fields["overdue_amount_field"],
    }


def _require_bindings_under_lock(
    runtime,
    *,
    pool: StrategyCandidatePoolArtifactBinding,
    sample: StrategySampleDesignV2ArtifactBinding,
) -> None:
    with runtime.task_artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        require_strategy_candidate_pool_artifact_binding_on_connection(
            conn,
            pool,
        )
        require_strategy_sample_design_v2_artifact_binding_on_connection(
            conn,
            sample,
        )
        conn.commit()


def _require_bindings_on_connection(
    conn,
    *,
    pool: StrategyCandidatePoolArtifactBinding,
    sample: StrategySampleDesignV2ArtifactBinding,
) -> None:
    require_strategy_candidate_pool_artifact_binding_on_connection(conn, pool)
    require_strategy_sample_design_v2_artifact_binding_on_connection(
        conn,
        sample,
    )


def _read_selected_partition(
    runtime,
    *,
    pool: StrategyCandidatePoolArtifactBinding,
    sample: StrategySampleDesignV2ArtifactBinding,
    partition: str,
    target_col: str,
    month_col: str | None,
    loan_amount_col: str | None,
    overdue_amount_col: str | None,
) -> pd.DataFrame:
    path = sample.source_binding.dataset_path
    _require_dataset_path(
        path,
        root=Path(runtime.settings.datasets_dir).absolute(),
    )
    fields = _expression_fields(pool.compiled_design["strategy_spec"])
    fields.add(target_col)
    fields.update(
        value
        for value in (month_col, loan_amount_col, overdue_amount_col)
        if value is not None
    )
    unknown = sorted(fields - set(sample.source_binding.columns))
    if unknown:
        raise StrategyError(
            "Strategy Pool rules reference missing V2 dataset columns: "
            + ", ".join(unknown)
        )
    if (
        sha256_file(path)
        != sample.source_binding.dataset_content_hash
    ):
        raise StrategyError(
            "Strategy Pool validation dataset bytes changed before replay"
        )
    frame = runtime.backend.read_frame(path, columns=sorted(fields))
    if not isinstance(frame, pd.DataFrame) or len(frame) != (
        sample.source_binding.row_count
    ):
        raise StrategyError(
            "Strategy Pool validation analysis universe row count changed"
        )
    mask = np.asarray(
        sample.membership["masks"][f"risk/{partition}"],
        dtype=np.bool_,
    )
    development = np.asarray(
        sample.membership["masks"]["risk/development"],
        dtype=np.bool_,
    )
    if len(mask) != len(frame) or len(development) != len(frame):
        raise StrategyError(
            "StrategySampleDesign V2 membership row order changed"
        )
    if bool(np.any(mask & development)):
        raise StrategyError(
            "Strategy Pool validation membership overlaps development rows"
        )
    expected_count = sample.membership["header"]["counts"]["risk"][
        partition
    ]
    if int(np.count_nonzero(mask)) != expected_count:
        raise StrategyError(
            "Strategy Pool validation membership count changed"
        )
    if expected_count == 0:
        raise StrategyError(f"Strategy Pool {partition} partition is empty")
    selected = frame.loc[
        pd.Series(mask, index=frame.index, dtype=bool)
    ].reset_index(drop=True)
    if len(selected) != expected_count:
        raise StrategyError(
            "Strategy Pool validation selected partition count changed"
        )
    if sha256_file(path) != sample.source_binding.dataset_content_hash:
        raise StrategyError(
            "Strategy Pool validation dataset bytes changed during replay"
        )
    return selected


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


def _sample_design_evidence_ref(
    sample: StrategySampleDesignV2ArtifactBinding,
    *,
    partition: str,
) -> dict[str, Any]:
    header = sample.membership["header"]
    bundle = sample.bundle
    design = bundle["sample_design"]
    return {
        "membership_artifact_id": sample.membership_artifact_id,
        "membership_artifact_content_hash": (
            sample.membership_artifact_content_hash
        ),
        "membership_id": header["membership_id"],
        "membership_content_hash": header["content_hash"],
        "bundle_artifact_id": sample.bundle_artifact_id,
        "bundle_artifact_content_hash": (
            sample.bundle_artifact_content_hash
        ),
        "bundle_id": bundle["bundle_id"],
        "bundle_content_hash": bundle["content_hash"],
        "sample_design_id": design["sample_design_id"],
        "sample_design_content_hash": design["content_hash"],
        "partition_key": f"risk/{partition}",
        "partition_count": header["counts"]["risk"][partition],
        "analysis_universe_row_count": header["row_count"],
    }


def _dataset_evidence_binding(
    sample: StrategySampleDesignV2ArtifactBinding,
) -> dict[str, Any]:
    source = sample.source_binding
    return {
        "task_id": source.task_id,
        "dataset_id": source.dataset_id,
        "dataset_content_hash": source.dataset_content_hash,
        "dataset_source_path": source.dataset_source_path,
        "dataset_registry_metadata_hash": (
            source.dataset_registry_metadata_hash
        ),
        "workspace_revision": source.workspace_revision,
        "workspace_generation": source.workspace_generation,
        "semantic_mapping_hash": source.semantic_mapping_hash,
    }


def _persist_evidence(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
    pool: StrategyCandidatePoolArtifactBinding,
    sample: StrategySampleDesignV2ArtifactBinding,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    canonical = canonical_strategy_pool_validation_json(evidence).encode(
        "utf-8"
    )
    artifact_hash = hashlib.sha256(canonical).hexdigest()
    out_dir = _prepare_output_directory(
        runtime.settings.tasks_dir,
        task_id=task_id,
    )
    final_path = out_dir / f"{evidence['evidence_id']}.json"
    sources = evidence["source_bindings"]
    provenance = {
        "schema_version": POOL_VALIDATION_ARTIFACT_SCHEMA_VERSION,
        "producer_version": STRATEGY_POOL_VALIDATION_PRODUCER_VERSION,
        "task_id": task_id,
        "evidence_id": evidence["evidence_id"],
        "evidence_content_hash": evidence["content_hash"],
        "pool_ref": {
            **dict(request["pool_ref"]),
            "pool_id": evidence["identity"]["pool_id"],
            "revision_id": evidence["identity"]["revision_id"],
        },
        "sample_design_ref": dict(request["sample_design_ref"]),
        "dataset_binding": dict(sources["dataset"]),
        "target_binding": dict(sources["target"]),
        "field_bindings": dict(sources["fields"]),
        "partition": evidence["partition"],
        "population": "risk",
        "comparison_mode": "absolute",
        "lifecycle_stage": evidence["lifecycle"]["stage"],
        "validation_status": "independent_evidence",
    }
    _validate_provenance(provenance)
    uow = ArtifactUnitOfWork()
    staged = uow.stage_file(out_dir, final_path.name)
    try:
        staged.path.write_bytes(canonical)
    except OSError as exc:
        uow.rollback()
        raise StrategyError(
            "Strategy Pool validation artifact could not be staged"
        ) from exc

    db_committed = False
    rollback_under_lock = False
    reused = False
    try:
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                _require_bindings_on_connection(
                    conn,
                    pool=pool,
                    sample=sample,
                )
                row = _select_artifact_row(
                    conn,
                    task_id=task_id,
                    path=final_path,
                )
                if row is not None:
                    _require_existing_artifact(
                        row,
                        task_id=task_id,
                        path=final_path,
                        canonical=canonical,
                        content_hash=artifact_hash,
                        provenance=provenance,
                    )
                    uow.rollback()
                    reused = True
                else:
                    if final_path.exists() or final_path.is_symlink():
                        raise StrategyError(
                            "Strategy Pool validation artifact path exists "
                            "without a registry row"
                        )
                    uow.promote_all()
                    _require_exact_file(
                        final_path,
                        root=Path(runtime.settings.tasks_dir).absolute(),
                        canonical=canonical,
                        content_hash=artifact_hash,
                    )
                _require_bindings_on_connection(
                    conn,
                    pool=pool,
                    sample=sample,
                )
                record = runtime.task_artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=POOL_VALIDATION_ARTIFACT_KIND,
                    path=str(final_path),
                    content_hash=artifact_hash,
                    origin_tool=POOL_VALIDATION_ORIGIN_TOOL,
                    provenance=provenance,
                )
                _require_bindings_on_connection(
                    conn,
                    pool=pool,
                    sample=sample,
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
    return validate_measure_strategy_pool_validation_tool_output(
        _tool_output(
            evidence=evidence,
            record=record,
            task_id=task_id,
        )
    )


def _tool_output(
    *,
    evidence: Mapping[str, Any],
    record: Mapping[str, Any],
    task_id: str,
) -> dict[str, Any]:
    identity = evidence["identity"]
    population = evidence["population_metrics"]
    artifact_id = str(record["id"])
    return {
        "schema_version": POOL_VALIDATION_TOOL_SCHEMA_VERSION,
        "evidence_id": evidence["evidence_id"],
        "content_hash": evidence["content_hash"],
        "pool_id": identity["pool_id"],
        "pool_revision": identity["revision"],
        "pool_snapshot_hash": identity["snapshot_hash"],
        "partition": evidence["partition"],
        "population": "risk",
        "comparison_mode": "absolute",
        "lifecycle_stage": evidence["lifecycle"]["stage"],
        "validation_status": "independent_evidence",
        "population_count": population["population_count"],
        "labeled_count": population["labelled_count"],
        "unlabeled_count": population["unlabelled_count"],
        "evidence": dict(evidence),
        "warnings": [
            str(flag["message"])
            for flag in evidence["red_flags"]
            if flag.get("level") in {"amber", "red"}
        ],
        "artifact": {
            "artifact_id": artifact_id,
            "kind": POOL_VALIDATION_ARTIFACT_KIND,
            "format": "json",
            "filename": Path(str(record["path"])).name,
            "content_hash": str(record["content_hash"]),
            "download_url": (
                f"/api/tasks/{quote(task_id, safe='')}"
                f"/task-artifacts/{quote(artifact_id, safe='')}/download"
            ),
        },
        "not_mutated_pool": True,
        "not_created_strategy": True,
        "not_adopted": True,
        "not_promoted": True,
        "not_deployed": True,
    }


def _validate_provenance(value: object) -> dict[str, Any]:
    obj = _json_object(value, "Strategy Pool validation provenance")
    _exact_fields(
        obj,
        _PROVENANCE_FIELDS,
        "Strategy Pool validation provenance",
    )
    if obj["schema_version"] != POOL_VALIDATION_ARTIFACT_SCHEMA_VERSION:
        raise StrategyError(
            "Strategy Pool validation provenance schema_version is invalid"
        )
    if obj["producer_version"] != STRATEGY_POOL_VALIDATION_PRODUCER_VERSION:
        raise StrategyError(
            "Strategy Pool validation provenance producer_version is invalid"
        )
    for field in ("task_id", "evidence_id", "partition"):
        _text(obj[field], f"validation provenance.{field}")
    _hash(
        obj["evidence_content_hash"],
        "validation provenance.evidence_content_hash",
    )
    if obj["population"] != "risk" or obj["comparison_mode"] != "absolute":
        raise StrategyError(
            "Strategy Pool validation provenance controls changed"
        )
    if (
        obj["lifecycle_stage"] != obj["partition"]
        or obj["validation_status"] != "independent_evidence"
    ):
        raise StrategyError(
            "Strategy Pool validation provenance lifecycle changed"
        )
    return obj


def _prepare_output_directory(
    tasks_dir: Path | str,
    *,
    task_id: str,
) -> Path:
    if Path(task_id).name != task_id or task_id in {".", ".."}:
        raise StrategyError("task_id cannot escape task storage")
    root = Path(tasks_dir).absolute()
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise StrategyError(
            "task artifact root must be a regular directory"
        )
    root.mkdir(parents=True, exist_ok=True)
    task_dir = root / task_id
    if task_dir.exists() and (
        task_dir.is_symlink() or not task_dir.is_dir()
    ):
        raise StrategyError(
            "task artifact directory must be a regular directory"
        )
    task_dir.mkdir(exist_ok=True)
    if (
        task_dir.is_symlink()
        or task_dir.resolve(strict=True).parent
        != root.resolve(strict=True)
    ):
        raise StrategyError(
            "Strategy Pool validation task directory escaped storage"
        )
    out_dir = task_dir / "strategy_pool_validations"
    if out_dir.exists() and (
        out_dir.is_symlink() or not out_dir.is_dir()
    ):
        raise StrategyError(
            "Strategy Pool validation output path must be a regular directory"
        )
    out_dir.mkdir(exist_ok=True)
    if (
        out_dir.is_symlink()
        or out_dir.resolve(strict=True).parent
        != task_dir.resolve(strict=True)
    ):
        raise StrategyError(
            "Strategy Pool validation output directory escaped storage"
        )
    return out_dir


def _select_artifact_row(conn, *, task_id: str, path: Path):
    return conn.execute(
        """
        SELECT id, task_id, kind, path, content_hash, origin_tool,
               provenance_json, created_at
          FROM task_artifacts
         WHERE task_id = ? AND kind = ? AND path = ?
        """,
        (task_id, POOL_VALIDATION_ARTIFACT_KIND, str(path)),
    ).fetchone()


def _require_existing_artifact(
    row,
    *,
    task_id: str,
    path: Path,
    canonical: bytes,
    content_hash: str,
    provenance: Mapping[str, Any],
) -> None:
    record = {field: row[field] for field in _TASK_ARTIFACT_ROW_FIELDS}
    expected = {
        "task_id": task_id,
        "kind": POOL_VALIDATION_ARTIFACT_KIND,
        "path": str(path),
        "content_hash": content_hash,
        "origin_tool": POOL_VALIDATION_ORIGIN_TOOL,
    }
    if any(str(record[field]) != value for field, value in expected.items()):
        raise StrategyError(
            "existing Strategy Pool validation artifact registry row changed"
        )
    provenance_json = json.dumps(
        dict(provenance),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if str(record["provenance_json"]) != provenance_json:
        raise StrategyError(
            "existing Strategy Pool validation provenance changed"
        )
    _require_exact_file(
        path,
        root=path.parents[2],
        canonical=canonical,
        content_hash=content_hash,
    )


def _require_exact_file(
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
        raise StrategyError(
            "Strategy Pool validation artifact bytes changed"
        )


def _read_regular_nofollow(
    path: Path,
    *,
    root: Path,
    expected_content_hash: str,
) -> bytes:
    if not path.is_absolute() or path.is_symlink():
        raise StrategyError(
            "Strategy Pool validation artifact must be a regular file"
        )
    current = path.parent
    resolved_root = root.absolute()
    while current != resolved_root:
        if current.is_symlink():
            raise StrategyError(
                "Strategy Pool validation artifact path traverses a symlink"
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
            "Strategy Pool validation artifact escaped task storage"
        ) from exc

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
            raise StrategyError(
                "Strategy Pool validation artifact must be a regular file"
            )
        if before.st_size < 0 or before.st_size > _MAX_ARTIFACT_BYTES:
            raise StrategyError(
                "Strategy Pool validation artifact exceeds byte budget"
            )
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_ARTIFACT_BYTES:
                raise StrategyError(
                    "Strategy Pool validation artifact exceeds byte budget"
                )
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
            raise StrategyError(
                "Strategy Pool validation artifact changed while read"
            )
        live = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(live.st_mode)
            or (live.st_dev, live.st_ino)
            != (after.st_dev, after.st_ino)
        ):
            raise StrategyError(
                "Strategy Pool validation artifact path changed while read"
            )
    except OSError as exc:
        raise StrategyError(
            "Strategy Pool validation artifact is unavailable"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    raw = b"".join(chunks)
    if (
        len(raw) != before.st_size
        or not hmac.compare_digest(
            digest.hexdigest(),
            expected_content_hash,
        )
    ):
        raise StrategyError(
            "Strategy Pool validation artifact bytes changed"
        )
    return raw


def _require_dataset_path(path: Path, *, root: Path) -> None:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise StrategyError(
            "Strategy Pool validation dataset must be a regular file"
        )
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise StrategyError(
            "Strategy Pool validation dataset escaped dataset storage"
        ) from exc


def _json_object(value: object, name: str) -> dict[str, Any]:
    try:
        normalized = json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise StrategyError(f"{name} must be canonical JSON") from exc
    if not isinstance(normalized, dict):
        raise StrategyError(f"{name} must be an object")
    if normalized != value:
        raise StrategyError(f"{name} contains non-canonical JSON values")
    return normalized


def _exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    if set(value) != expected:
        unexpected = sorted(set(value) - expected)
        missing = sorted(expected - set(value))
        details: list[str] = []
        if unexpected:
            details.append("unsupported fields: " + ", ".join(unexpected))
        if missing:
            details.append("missing fields: " + ", ".join(missing))
        raise StrategyError(f"{name} has invalid fields ({'; '.join(details)})")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise StrategyError(f"{name} must be a non-empty string")
    return value.strip()


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise StrategyError(f"{name} must be a lowercase SHA-256 hash")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise StrategyError(f"{name} must be a positive integer")
    return value


__all__ = [
    "POOL_VALIDATION_ARTIFACT_KIND",
    "POOL_VALIDATION_ARTIFACT_SCHEMA_VERSION",
    "POOL_VALIDATION_ORIGIN_TOOL",
    "POOL_VALIDATION_TOOL_SCHEMA_VERSION",
    "run_measure_strategy_pool_validation",
    "validate_measure_strategy_pool_validation_tool_output",
]
