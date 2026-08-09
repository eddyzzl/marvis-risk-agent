"""Governed publication of same-sample model-score comparison evidence.

This boundary accepts only task-owned score-evidence artifact pairs produced by
the modeling pack.  It re-authenticates every pair and the exact sample design,
delegates metric reconciliation to :mod:`model_score_evidence_adapter`, and
publishes an immutable comparison document.  The document deliberately carries
``no_selection``; this Tool does not rank, select, adopt, or deploy a model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
from pathlib import Path
import re
import stat
from typing import Any
from urllib.parse import quote

from marvis.artifacts import ArtifactUnitOfWork
from marvis.artifacts.transactional import ArtifactTransactionError
from marvis.packs.modeling.errors import ModelingError
from marvis.packs.modeling.score_evidence import (
    MODEL_SCORE_EVIDENCE_ARTIFACT_KIND,
    MODEL_SCORE_VECTOR_ARTIFACT_KIND,
)
from marvis.packs.modeling.score_evidence_tools import (
    MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_ORIGIN_TOOL,
    ModelScoreEvidenceArtifactBinding,
    load_model_score_evidence_artifacts,
    require_model_score_evidence_artifact_binding_on_connection,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.model_evidence import (
    MAX_MODEL_EVIDENCE,
    MAX_MODEL_EVIDENCE_JSON_BYTES,
    StrategyModelEvidenceError,
)
from marvis.packs.strategy.model_score_evidence_adapter import (
    ModelScoreEvidenceComparisonError,
    build_model_score_comparison,
)
from marvis.packs.strategy.sample_design_v2 import (
    PARTITION_NAMES,
    POPULATION_ROLES,
)
from marvis.packs.strategy.sample_design_v2_native_tools import (
    SAMPLE_DESIGN_V2_NATIVE_MEMBERSHIP_ARTIFACT_KIND,
    SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL,
)
from marvis.packs.strategy.sample_design_v2_tools import (
    SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
    SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND,
    SAMPLE_DESIGN_V2_ORIGIN_TOOL,
    load_any_strategy_sample_design_v2_artifacts,
    require_any_strategy_sample_design_v2_artifact_binding_on_connection,
)
from marvis.repositories.task_artifacts import (
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
)


MODEL_SCORE_COMPARISON_V2_TOOL_SCHEMA_VERSION = (
    "strategy.materialize-model-score-comparison-v2-tool.v1"
)
MODEL_SCORE_COMPARISON_V2_ARTIFACT_SCHEMA_VERSION = (
    "strategy.model-score-comparison-v2-artifact.v1"
)
MODEL_SCORE_COMPARISON_V2_ARTIFACT_KIND = (
    "strategy_model_score_comparison_v2_json"
)
MODEL_SCORE_COMPARISON_V2_ORIGIN_TOOL = (
    "strategy.materialize_model_score_comparison_v2"
)
MODEL_SCORE_COMPARISON_V2_AUDIT_KIND = (
    "strategy.model_score_comparison.published"
)
MODEL_SCORE_COMPARISON_V2_PRODUCER_VERSION = (
    "marvis.strategy.model-score-comparison/1"
)

_MAX_INPUT_JSON_BYTES = 1024 * 1024
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_INPUT_FIELDS = frozenset(
    {
        "sample_design_ref",
        "model_score_evidence_refs",
        "population",
        "partition",
        "expected_registry_token",
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
_SCORE_REF_FIELDS = frozenset(
    {
        "evidence_artifact_id",
        "expected_evidence_artifact_content_hash",
        "score_vector_artifact_id",
        "expected_score_vector_artifact_content_hash",
    }
)
_DOCUMENT_FIELDS = frozenset(
    {
        "schema_version",
        "task_id",
        "sample_design_ref",
        "sample_design_bundle_id",
        "sample_design_id",
        "sample_design_content_hash",
        "sample_design_bundle_content_hash",
        "model_score_evidence_refs",
        "population",
        "partition",
        "comparison",
        "governance",
        "content_hash",
    }
)
_GOVERNANCE_FIELDS = frozenset(
    {
        "selection_status",
        "winner_selected",
        "not_adopted",
        "not_deployed",
    }
)
_OUTPUT_FIELDS = frozenset(
    {
        "schema_version",
        "comparison_id",
        "comparison_content_hash",
        "sample_design_id",
        "sample_design_content_hash",
        "population",
        "partition",
        "model_score_evidence_refs",
        "comparison",
        "artifact",
        "governance",
        "content_hash",
    }
)
_OUTPUT_ARTIFACT_FIELDS = frozenset(
    {"artifact_id", "kind", "filename", "content_hash", "download_url"}
)
_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "format",
        "task_id",
        "request_hash",
        "artifact_content_hash",
        "document_content_hash",
        "comparison_id",
        "comparison_content_hash",
        "sample_design_ref",
        "sample_design_bundle_id",
        "sample_design_id",
        "sample_design_content_hash",
        "sample_design_bundle_content_hash",
        "model_score_evidence_refs",
        "model_evidence_refs",
        "population",
        "partition",
        "governance",
    }
)
_BOUNDARY_ERRORS = (
    ArtifactTransactionError,
    ModelingError,
    ModelScoreEvidenceComparisonError,
    StrategyModelEvidenceError,
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
)


@dataclass(frozen=True)
class ModelScoreComparisonV2ArtifactBinding:
    """One authenticated comparison artifact and all live dependencies."""

    task_id: str
    record: dict[str, Any]
    path: Path
    document: dict[str, Any]
    sample_binding: Any
    score_bindings: tuple[ModelScoreEvidenceArtifactBinding, ...]


def model_score_comparison_registry_snapshot_token(
    artifacts: Sequence[Mapping[str, Any]],
) -> str:
    """Hash the complete ordered registry of eligible sample/score sources."""

    if isinstance(artifacts, (str, bytes, bytearray)) or not isinstance(
        artifacts, Sequence
    ):
        raise StrategyError("model-score comparison registry snapshot must be a list")
    relevant: list[dict[str, Any]] = []
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise StrategyError(
                "model-score comparison registry snapshot must contain objects"
            )
        kind = artifact.get("kind")
        origin = artifact.get("origin_tool")
        is_sample = (
            kind
            in {
                SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND,
                SAMPLE_DESIGN_V2_NATIVE_MEMBERSHIP_ARTIFACT_KIND,
                SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
            }
            and origin
            in {SAMPLE_DESIGN_V2_ORIGIN_TOOL, SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL}
        )
        is_score = (
            kind
            in {MODEL_SCORE_VECTOR_ARTIFACT_KIND, MODEL_SCORE_EVIDENCE_ARTIFACT_KIND}
            and origin == MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_ORIGIN_TOOL
        )
        if is_sample or is_score:
            relevant.append(
                {
                    "id": artifact.get("id"),
                    "kind": kind,
                    "content_hash": artifact.get("content_hash"),
                    "origin_tool": origin,
                    "provenance": artifact.get("provenance"),
                    "created_at": artifact.get("created_at"),
                }
            )
    relevant.sort(key=lambda item: (str(item["created_at"]), str(item["id"])))
    try:
        return _sha256(_canonical_json(relevant).encode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise StrategyError(
            "model-score comparison registry snapshot is not canonical JSON"
        ) from exc


def run_materialize_model_score_comparison_v2(
    inputs: object,
    ctx,
    runtime,
) -> dict[str, Any]:
    """Publish one deterministic, non-selecting comparison artifact."""

    try:
        normalized = _validate_inputs(inputs)
        expected_registry_token = normalized.pop("expected_registry_token")
        request = normalized
        task_id = _text(ctx.task_id, "task_id")
        sample_binding = _load_sample(runtime, task_id=task_id, request=request)
        score_bindings = _load_scores(
            runtime,
            task_id=task_id,
            requests=request["model_score_evidence_refs"],
            sample_binding=sample_binding,
        )
        comparison = _build_comparison(
            request=request,
            sample_binding=sample_binding,
            score_bindings=score_bindings,
        )
        binding = _persist_comparison(
            runtime,
            task_id=task_id,
            request=request,
            expected_registry_token=expected_registry_token,
            sample_binding=sample_binding,
            score_bindings=score_bindings,
            comparison=comparison,
        )
        output = _tool_output(binding)
        return validate_materialize_model_score_comparison_v2_tool_output(
            output,
            runtime=runtime,
            task_id=task_id,
        )
    except StrategyError:
        raise
    except _BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc


def validate_materialize_model_score_comparison_v2_tool_output(
    value: object,
    *,
    runtime,
    task_id: str,
) -> dict[str, Any]:
    """Rebuild a cached Tool output from current authenticated artifact bytes."""

    obj = _object(value, "model-score comparison output")
    _exact_fields(obj, _OUTPUT_FIELDS, "model-score comparison output")
    if obj["schema_version"] != MODEL_SCORE_COMPARISON_V2_TOOL_SCHEMA_VERSION:
        raise StrategyError("model-score comparison output schema_version is invalid")
    artifact = _object(obj["artifact"], "model-score comparison output artifact")
    _exact_fields(
        artifact,
        _OUTPUT_ARTIFACT_FIELDS,
        "model-score comparison output artifact",
    )
    artifact_id = _hash(artifact["artifact_id"], "output.artifact.artifact_id")
    artifact_hash = _hash(artifact["content_hash"], "output.artifact.content_hash")
    binding = load_model_score_comparison_v2_artifact(
        runtime,
        task_id=task_id,
        artifact_id=artifact_id,
        expected_artifact_content_hash=artifact_hash,
    )
    expected = _tool_output(binding)
    if obj != expected:
        raise StrategyError(
            "model-score comparison output drifted from authenticated artifact"
        )
    return dict(obj)


def load_model_score_comparison_v2_artifact(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_artifact_content_hash: str,
) -> ModelScoreComparisonV2ArtifactBinding:
    """Load and re-authenticate one persisted comparison and its dependencies."""

    try:
        normalized_task = _text(task_id, "task_id")
        normalized_artifact_id = _hash(artifact_id, "artifact_id")
        artifact_hash = _hash(
            expected_artifact_content_hash,
            "expected_artifact_content_hash",
        )
        record = _registered_record(
            runtime,
            task_id=normalized_task,
            artifact_id=normalized_artifact_id,
            expected_content_hash=artifact_hash,
        )
        raw = _read_verified(
            Path(str(record["path"])),
            root=Path(runtime.settings.tasks_dir),
            expected_hash=artifact_hash,
        )
        decoded = _decode_json(raw, "model-score comparison artifact")
        semantic_request = _document_request(decoded)
        sample_binding = _load_sample(
            runtime,
            task_id=normalized_task,
            request=semantic_request,
        )
        score_bindings = _load_scores(
            runtime,
            task_id=normalized_task,
            requests=semantic_request["model_score_evidence_refs"],
            sample_binding=sample_binding,
        )
        comparison = _build_comparison(
            request=semantic_request,
            sample_binding=sample_binding,
            score_bindings=score_bindings,
        )
        expected_document = _artifact_document(
            task_id=normalized_task,
            request=semantic_request,
            sample_binding=sample_binding,
            comparison=comparison,
        )
        if decoded != expected_document:
            raise StrategyError(
                "model-score comparison artifact drifted from authenticated sources"
            )
        expected_path = _artifact_path(
            runtime.settings.tasks_dir,
            task_id=normalized_task,
            comparison_id=comparison["comparison_id"],
        )
        if Path(str(record["path"])) != expected_path:
            raise StrategyError("model-score comparison artifact path is not canonical")
        expected_provenance = _artifact_provenance(
            task_id=normalized_task,
            request=semantic_request,
            sample_binding=sample_binding,
            comparison=comparison,
            document=expected_document,
            artifact_content_hash=artifact_hash,
        )
        if _validate_provenance(record["provenance"]) != expected_provenance:
            raise StrategyError("model-score comparison artifact provenance changed")
        binding = ModelScoreComparisonV2ArtifactBinding(
            task_id=normalized_task,
            record=dict(record),
            path=expected_path,
            document=expected_document,
            sample_binding=sample_binding,
            score_bindings=tuple(score_bindings),
        )
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            _require_binding_on_connection(conn, binding)
            conn.commit()
        return binding
    except StrategyError:
        raise
    except _BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc


def _validate_inputs(value: object) -> dict[str, Any]:
    obj = _object(value, "materialize_model_score_comparison_v2 inputs")
    _exact_fields(
        obj,
        _INPUT_FIELDS,
        "materialize_model_score_comparison_v2 inputs",
    )
    request = _semantic_request(
        sample_design_ref=obj["sample_design_ref"],
        model_score_evidence_refs=obj["model_score_evidence_refs"],
        population=obj["population"],
        partition=obj["partition"],
    )
    request["expected_registry_token"] = _hash(
        obj["expected_registry_token"],
        "expected_registry_token",
    )
    _require_json_budget(request, "materialize_model_score_comparison_v2 inputs")
    return request


def _semantic_request(
    *,
    sample_design_ref: object,
    model_score_evidence_refs: object,
    population: object,
    partition: object,
) -> dict[str, Any]:
    sample = _object(sample_design_ref, "sample_design_ref")
    _exact_fields(sample, _SAMPLE_REF_FIELDS, "sample_design_ref")
    normalized_sample = {
        field: (
            _text(sample[field], f"sample_design_ref.{field}")
            if field in {"expected_bundle_id", "expected_sample_design_id"}
            else _hash(sample[field], f"sample_design_ref.{field}")
        )
        for field in sorted(_SAMPLE_REF_FIELDS)
    }
    raw_refs = _array(
        model_score_evidence_refs,
        "model_score_evidence_refs",
    )
    if not 2 <= len(raw_refs) <= MAX_MODEL_EVIDENCE:
        raise StrategyError(
            "model_score_evidence_refs requires between 2 and "
            f"{MAX_MODEL_EVIDENCE} items"
        )
    refs: list[dict[str, str]] = []
    for index, raw in enumerate(raw_refs):
        item = _object(raw, f"model_score_evidence_refs[{index}]")
        _exact_fields(
            item,
            _SCORE_REF_FIELDS,
            f"model_score_evidence_refs[{index}]",
        )
        refs.append(
            {
                field: _hash(item[field], f"model_score_evidence_refs[{index}].{field}")
                for field in sorted(_SCORE_REF_FIELDS)
            }
        )
    if len({item["evidence_artifact_id"] for item in refs}) != len(refs):
        raise StrategyError(
            "model_score_evidence_refs contains duplicate evidence artifacts"
        )
    if len({item["score_vector_artifact_id"] for item in refs}) != len(refs):
        raise StrategyError(
            "model_score_evidence_refs contains duplicate score vectors"
        )
    refs.sort(key=lambda item: item["evidence_artifact_id"])
    normalized_population = _enum(
        population,
        set(POPULATION_ROLES),
        "population",
    )
    normalized_partition = _enum(
        partition,
        {"overall", *PARTITION_NAMES},
        "partition",
    )
    request = {
        "sample_design_ref": normalized_sample,
        "model_score_evidence_refs": refs,
        "population": normalized_population,
        "partition": normalized_partition,
    }
    _require_json_budget(request, "model-score comparison request")
    return request


def _document_request(value: object) -> dict[str, Any]:
    obj = _object(value, "model-score comparison artifact")
    _exact_fields(obj, _DOCUMENT_FIELDS, "model-score comparison artifact")
    if obj["schema_version"] != MODEL_SCORE_COMPARISON_V2_ARTIFACT_SCHEMA_VERSION:
        raise StrategyError("model-score comparison artifact schema_version is invalid")
    request = _semantic_request(
        sample_design_ref=obj["sample_design_ref"],
        model_score_evidence_refs=obj["model_score_evidence_refs"],
        population=obj["population"],
        partition=obj["partition"],
    )
    _hash(obj["content_hash"], "artifact.content_hash")
    _governance(obj["governance"])
    return request


def _load_sample(runtime, *, task_id: str, request: Mapping[str, Any]):
    return load_any_strategy_sample_design_v2_artifacts(
        runtime,
        task_id=task_id,
        **request["sample_design_ref"],
    )


def _load_scores(
    runtime,
    *,
    task_id: str,
    requests: Sequence[Mapping[str, str]],
    sample_binding,
) -> tuple[ModelScoreEvidenceArtifactBinding, ...]:
    result: list[ModelScoreEvidenceArtifactBinding] = []
    for request in requests:
        binding = load_model_score_evidence_artifacts(
            runtime,
            task_id=task_id,
            evidence_artifact_id=request["evidence_artifact_id"],
            expected_evidence_artifact_content_hash=request[
                "expected_evidence_artifact_content_hash"
            ],
            score_vector_artifact_id=request["score_vector_artifact_id"],
            expected_score_vector_artifact_content_hash=request[
                "expected_score_vector_artifact_content_hash"
            ],
        )
        if (
            binding.task_id != task_id
            or binding.training.task_id != task_id
            or binding.training.sample.task_id != task_id
        ):
            raise StrategyError("model score evidence belongs to another task")
        if binding.training.sample.bundle != sample_binding.bundle:
            raise StrategyError(
                "model score evidence does not use the requested sample design"
            )
        result.append(binding)
    return tuple(result)


def _build_comparison(
    *,
    request: Mapping[str, Any],
    sample_binding,
    score_bindings: Sequence[ModelScoreEvidenceArtifactBinding],
) -> dict[str, Any]:
    comparison = build_model_score_comparison(
        sample_design_bundle=sample_binding.bundle,
        model_evidence=[
            binding.envelope["single_model_evidence"] for binding in score_bindings
        ],
        population=request["population"],
        partition=request["partition"],
    )
    if comparison["selection"]["status"] != "no_selection":
        raise StrategyError("model-score comparison adapter selected a winner")
    return comparison


def _persist_comparison(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
    expected_registry_token: str,
    sample_binding,
    score_bindings: Sequence[ModelScoreEvidenceArtifactBinding],
    comparison: Mapping[str, Any],
) -> ModelScoreComparisonV2ArtifactBinding:
    document = _artifact_document(
        task_id=task_id,
        request=request,
        sample_binding=sample_binding,
        comparison=comparison,
    )
    canonical = _canonical_json(document).encode("utf-8")
    artifact_content_hash = _sha256(canonical)
    path = _artifact_path(
        runtime.settings.tasks_dir,
        task_id=task_id,
        comparison_id=comparison["comparison_id"],
    )
    out_dir = path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance = _artifact_provenance(
        task_id=task_id,
        request=request,
        sample_binding=sample_binding,
        comparison=comparison,
        document=document,
        artifact_content_hash=artifact_content_hash,
    )
    uow = ArtifactUnitOfWork()
    staged = uow.stage_file(out_dir, path.name)
    try:
        staged.path.write_bytes(canonical)
    except OSError as exc:
        uow.rollback()
        raise StrategyError(
            "model-score comparison artifact could not be staged"
        ) from exc
    db_committed = False
    rollback_attempted_under_lock = False
    try:
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                _require_registry_snapshot_on_connection(
                    conn,
                    task_id=task_id,
                    expected_token=expected_registry_token,
                )
                require_any_strategy_sample_design_v2_artifact_binding_on_connection(
                    conn,
                    sample_binding,
                )
                for binding in score_bindings:
                    require_model_score_evidence_artifact_binding_on_connection(
                        conn,
                        binding,
                    )
                rebuilt = _build_comparison(
                    request=request,
                    sample_binding=sample_binding,
                    score_bindings=score_bindings,
                )
                if rebuilt != comparison:
                    raise StrategyError(
                        "model-score comparison changed before persistence"
                    )
                _publish_staged_file(
                    conn,
                    staged=staged,
                    path=path,
                    canonical=canonical,
                    task_id=task_id,
                )
                record = runtime.task_artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=MODEL_SCORE_COMPARISON_V2_ARTIFACT_KIND,
                    path=str(path),
                    content_hash=artifact_content_hash,
                    origin_tool=MODEL_SCORE_COMPARISON_V2_ORIGIN_TOOL,
                    provenance=provenance,
                )
                runtime.repo.write_audit_on_connection(
                    conn,
                    kind=MODEL_SCORE_COMPARISON_V2_AUDIT_KIND,
                    target_ref=comparison["comparison_id"],
                    inputs_hash=_request_hash(request),
                    outcome="succeeded",
                    detail={
                        "task_id": task_id,
                        "comparison_artifact_id": record["id"],
                        "comparison_id": comparison["comparison_id"],
                        "model_count": len(score_bindings),
                        "population": request["population"],
                        "partition": request["partition"],
                        **_governance_value(),
                    },
                )
                pending = ModelScoreComparisonV2ArtifactBinding(
                    task_id=task_id,
                    record=dict(record),
                    path=path,
                    document=document,
                    sample_binding=sample_binding,
                    score_bindings=tuple(score_bindings),
                )
                _require_binding_on_connection(conn, pending)
                conn.commit()
                db_committed = True
            except Exception:
                rollback_attempted_under_lock = True
                uow.rollback()
                raise
        uow.commit()
    except Exception:
        if not db_committed and not rollback_attempted_under_lock:
            uow.rollback()
        raise
    return ModelScoreComparisonV2ArtifactBinding(
        task_id=task_id,
        record=dict(record),
        path=path,
        document=document,
        sample_binding=sample_binding,
        score_bindings=tuple(score_bindings),
    )


def _publish_staged_file(
    conn,
    *,
    staged,
    path: Path,
    canonical: bytes,
    task_id: str,
) -> None:
    row = conn.execute(
        """
        SELECT * FROM task_artifacts
        WHERE task_id = ? AND kind = ? AND path = ?
        """,
        (task_id, MODEL_SCORE_COMPARISON_V2_ARTIFACT_KIND, str(path)),
    ).fetchone()
    if row is not None or path.exists() or path.is_symlink():
        _require_exact_file(path, canonical=canonical)
        staged.rollback()
        return
    staged.promote()
    _require_exact_file(path, canonical=canonical)


def _require_binding_on_connection(
    conn,
    binding: ModelScoreComparisonV2ArtifactBinding,
) -> None:
    if not isinstance(binding, ModelScoreComparisonV2ArtifactBinding):
        raise StrategyError("model-score comparison binding is invalid")
    if not getattr(conn, "in_transaction", False):
        raise StrategyError(
            "model-score comparison revalidation requires an active transaction"
        )
    require_any_strategy_sample_design_v2_artifact_binding_on_connection(
        conn,
        binding.sample_binding,
    )
    for score_binding in binding.score_bindings:
        require_model_score_evidence_artifact_binding_on_connection(
            conn,
            score_binding,
        )
    row = conn.execute(
        "SELECT * FROM task_artifacts WHERE id = ? AND task_id = ?",
        (binding.record["id"], binding.task_id),
    ).fetchone()
    if row is None:
        raise StrategyError("model-score comparison TaskArtifact disappeared")
    if (
        row["kind"] != MODEL_SCORE_COMPARISON_V2_ARTIFACT_KIND
        or row["path"] != str(binding.path)
        or row["content_hash"] != binding.record["content_hash"]
        or row["origin_tool"] != MODEL_SCORE_COMPARISON_V2_ORIGIN_TOOL
        or json.loads(row["provenance_json"]) != binding.record["provenance"]
    ):
        raise StrategyError("model-score comparison TaskArtifact binding changed")
    canonical = _canonical_json(binding.document).encode("utf-8")
    _require_exact_file(binding.path, canonical=canonical)


def _require_registry_snapshot_on_connection(
    conn,
    *,
    task_id: str,
    expected_token: str,
) -> None:
    rows = conn.execute(
        """
        SELECT id, kind, content_hash, origin_tool, provenance_json, created_at
        FROM task_artifacts
        WHERE task_id = ?
        ORDER BY created_at, id
        """,
        (task_id,),
    ).fetchall()
    try:
        artifacts = [
            {
                "id": row["id"],
                "kind": row["kind"],
                "content_hash": row["content_hash"],
                "origin_tool": row["origin_tool"],
                "provenance": json.loads(row["provenance_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise StrategyError(
            "model-score comparison registry snapshot is invalid"
        ) from exc
    current = model_score_comparison_registry_snapshot_token(artifacts)
    if not hmac.compare_digest(current, expected_token):
        raise StrategyError(
            "model-score comparison registry snapshot changed before persistence"
        )


def _artifact_document(
    *,
    task_id: str,
    request: Mapping[str, Any],
    sample_binding,
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    design = sample_binding.bundle["sample_design"]
    body = {
        "schema_version": MODEL_SCORE_COMPARISON_V2_ARTIFACT_SCHEMA_VERSION,
        "task_id": task_id,
        "sample_design_ref": dict(request["sample_design_ref"]),
        "sample_design_bundle_id": sample_binding.bundle["bundle_id"],
        "sample_design_id": design["sample_design_id"],
        "sample_design_content_hash": design["content_hash"],
        "sample_design_bundle_content_hash": sample_binding.bundle["content_hash"],
        "model_score_evidence_refs": [
            dict(item) for item in request["model_score_evidence_refs"]
        ],
        "population": request["population"],
        "partition": request["partition"],
        "comparison": dict(comparison),
        "governance": _governance_value(),
    }
    return {
        **body,
        "content_hash": _sha256(_canonical_json(body).encode("utf-8")),
    }


def _artifact_provenance(
    *,
    task_id: str,
    request: Mapping[str, Any],
    sample_binding,
    comparison: Mapping[str, Any],
    document: Mapping[str, Any],
    artifact_content_hash: str,
) -> dict[str, Any]:
    design = sample_binding.bundle["sample_design"]
    return {
        "schema_version": MODEL_SCORE_COMPARISON_V2_ARTIFACT_SCHEMA_VERSION,
        "producer_version": MODEL_SCORE_COMPARISON_V2_PRODUCER_VERSION,
        "format": "json",
        "task_id": task_id,
        "request_hash": _request_hash(request),
        "artifact_content_hash": artifact_content_hash,
        "document_content_hash": document["content_hash"],
        "comparison_id": comparison["comparison_id"],
        "comparison_content_hash": comparison["content_hash"],
        "sample_design_ref": dict(request["sample_design_ref"]),
        "sample_design_bundle_id": sample_binding.bundle["bundle_id"],
        "sample_design_id": design["sample_design_id"],
        "sample_design_content_hash": design["content_hash"],
        "sample_design_bundle_content_hash": sample_binding.bundle["content_hash"],
        "model_score_evidence_refs": [
            dict(item) for item in request["model_score_evidence_refs"]
        ],
        "model_evidence_refs": [
            dict(item) for item in comparison["model_evidence_refs"]
        ],
        "population": request["population"],
        "partition": request["partition"],
        "governance": _governance_value(),
    }


def _validate_provenance(value: object) -> dict[str, Any]:
    obj = _object(value, "model-score comparison provenance")
    _exact_fields(obj, _PROVENANCE_FIELDS, "model-score comparison provenance")
    if (
        obj["schema_version"] != MODEL_SCORE_COMPARISON_V2_ARTIFACT_SCHEMA_VERSION
        or obj["producer_version"] != MODEL_SCORE_COMPARISON_V2_PRODUCER_VERSION
        or obj["format"] != "json"
    ):
        raise StrategyError("model-score comparison provenance contract is invalid")
    for field in (
        "request_hash",
        "artifact_content_hash",
        "document_content_hash",
        "comparison_content_hash",
        "sample_design_content_hash",
        "sample_design_bundle_content_hash",
    ):
        _hash(obj[field], f"provenance.{field}")
    for field in (
        "task_id",
        "comparison_id",
        "sample_design_bundle_id",
        "sample_design_id",
    ):
        _text(obj[field], f"provenance.{field}")
    _governance(obj["governance"])
    return dict(obj)


def _tool_output(binding: ModelScoreComparisonV2ArtifactBinding) -> dict[str, Any]:
    document = binding.document
    comparison = document["comparison"]
    artifact_id = str(binding.record["id"])
    artifact_hash = str(binding.record["content_hash"])
    body = {
        "schema_version": MODEL_SCORE_COMPARISON_V2_TOOL_SCHEMA_VERSION,
        "comparison_id": comparison["comparison_id"],
        "comparison_content_hash": comparison["content_hash"],
        "sample_design_id": document["sample_design_id"],
        "sample_design_content_hash": document["sample_design_content_hash"],
        "population": document["population"],
        "partition": document["partition"],
        "model_score_evidence_refs": [
            dict(item) for item in document["model_score_evidence_refs"]
        ],
        "comparison": dict(comparison),
        "artifact": {
            "artifact_id": artifact_id,
            "kind": MODEL_SCORE_COMPARISON_V2_ARTIFACT_KIND,
            "filename": binding.path.name,
            "content_hash": artifact_hash,
            "download_url": (
                f"/api/tasks/{quote(binding.task_id, safe='')}"
                f"/task-artifacts/{quote(artifact_id, safe='')}/download"
                f"?expected_content_hash={artifact_hash}"
            ),
        },
        "governance": _governance_value(),
    }
    return {
        **body,
        "content_hash": _sha256(_canonical_json(body).encode("utf-8")),
    }


def _registered_record(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_content_hash: str,
) -> dict[str, Any]:
    record = runtime.task_artifacts.get_for_task(task_id, artifact_id)
    if not isinstance(record, Mapping):
        raise StrategyError("model-score comparison artifact was not found")
    if (
        record.get("id") != artifact_id
        or record.get("task_id") != task_id
        or record.get("kind") != MODEL_SCORE_COMPARISON_V2_ARTIFACT_KIND
        or record.get("origin_tool") != MODEL_SCORE_COMPARISON_V2_ORIGIN_TOOL
        or not hmac.compare_digest(
            str(record.get("content_hash")),
            expected_content_hash,
        )
    ):
        raise StrategyError("model-score comparison artifact registry binding changed")
    return dict(record)


def _artifact_path(
    tasks_dir: Path,
    *,
    task_id: str,
    comparison_id: str,
) -> Path:
    task = _text(task_id, "task_id")
    comparison = _text(comparison_id, "comparison_id")
    filename = f"{comparison}.json"
    if Path(filename).name != filename:
        raise StrategyError("model-score comparison id is not path safe")
    root = Path(tasks_dir).absolute()
    path = root / task / "strategy_model_score_comparisons" / filename
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise StrategyError("model-score comparison path escaped task root") from exc
    return path


def _read_verified(
    path: Path,
    *,
    root: Path,
    expected_hash: str,
) -> bytes:
    absolute_root = Path(root).absolute()
    absolute_path = Path(path).absolute()
    try:
        absolute_path.relative_to(absolute_root)
    except ValueError as exc:
        raise StrategyError("model-score comparison artifact escaped task root") from exc
    if absolute_path.is_symlink():
        raise StrategyError("model-score comparison artifact must not be a symlink")
    try:
        info = absolute_path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise StrategyError(
                "model-score comparison artifact must be a regular file"
            )
        if info.st_size > MAX_MODEL_EVIDENCE_JSON_BYTES:
            raise StrategyError("model-score comparison artifact exceeds byte budget")
        raw = absolute_path.read_bytes()
    except OSError as exc:
        raise StrategyError("model-score comparison artifact is unavailable") from exc
    if not hmac.compare_digest(_sha256(raw), expected_hash):
        raise StrategyError("model-score comparison artifact hash changed")
    return raw


def _require_exact_file(path: Path, *, canonical: bytes) -> None:
    expected_hash = _sha256(canonical)
    raw = _read_verified(
        path,
        root=path.parents[2],
        expected_hash=expected_hash,
    )
    if raw != canonical:
        raise StrategyError("model-score comparison artifact bytes changed")


def _decode_json(raw: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StrategyError(f"{name} is not valid UTF-8 JSON") from exc
    return _object(value, name)


def _governance_value() -> dict[str, Any]:
    return {
        "selection_status": "no_selection",
        "winner_selected": False,
        "not_adopted": True,
        "not_deployed": True,
    }


def _governance(value: object) -> dict[str, Any]:
    obj = _object(value, "model-score comparison governance")
    _exact_fields(obj, _GOVERNANCE_FIELDS, "model-score comparison governance")
    expected = _governance_value()
    if obj != expected:
        raise StrategyError("model-score comparison governance flags are invalid")
    return dict(obj)


def _request_hash(request: Mapping[str, Any]) -> str:
    return _sha256(_canonical_json(request).encode("utf-8"))


def _require_json_budget(value: object, name: str) -> None:
    try:
        size = len(_canonical_json(value).encode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise StrategyError(f"{name} is not canonical JSON") from exc
    if size > _MAX_INPUT_JSON_BYTES:
        raise StrategyError(f"{name} exceeds byte budget")


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StrategyError(f"{name} must be an object")
    return dict(value)


def _array(value: object, name: str) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise StrategyError(f"{name} must be an array")
    return list(value)


def _exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if extra:
            detail.append("unsupported " + ", ".join(extra))
        raise StrategyError(f"{name} fields are invalid: {'; '.join(detail)}")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise StrategyError(f"{name} must be non-empty text")
    return value


def _hash(value: object, name: str) -> str:
    text = _text(value, name)
    if _HASH_RE.fullmatch(text) is None:
        raise StrategyError(f"{name} must be a lowercase sha256")
    return text


def _enum(value: object, allowed: set[str], name: str) -> str:
    text = _text(value, name)
    if text not in allowed:
        raise StrategyError(f"{name} is invalid")
    return text


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


__all__ = [
    "MODEL_SCORE_COMPARISON_V2_ARTIFACT_KIND",
    "MODEL_SCORE_COMPARISON_V2_ARTIFACT_SCHEMA_VERSION",
    "MODEL_SCORE_COMPARISON_V2_ORIGIN_TOOL",
    "MODEL_SCORE_COMPARISON_V2_TOOL_SCHEMA_VERSION",
    "ModelScoreComparisonV2ArtifactBinding",
    "load_model_score_comparison_v2_artifact",
    "model_score_comparison_registry_snapshot_token",
    "run_materialize_model_score_comparison_v2",
    "validate_materialize_model_score_comparison_v2_tool_output",
]
