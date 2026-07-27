"""Authenticated model-score evidence publication for Strategy V2.

The Tool accepts only a complete ``ModelingTrainingEvidence`` reference.  It
scores every row of that evidence's bound active dataset exactly once, using an
independently opened and hashed task-owned copy of the registered model binary.
The immutable Parquet vector, canonical JSON envelope, TaskArtifact rows, and
audit row are then published in one caller-owned filesystem/database unit of
work.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Any
from urllib.parse import quote

from filelock import FileLock, Timeout as FileLockTimeout
import numpy as np
import pandas as pd

from marvis.artifacts import ArtifactUnitOfWork
from marvis.artifacts.model_score_vector import (
    MAX_MODEL_SCORE_VECTOR_ROWS,
    ModelScoreVector,
    ModelScoreVectorError,
    validate_model_score_vector,
    write_model_score_vector,
)
from marvis.artifacts.transactional import ArtifactTransactionError
from marvis.data.errors import DataLayerError
from marvis.files import sha256_file
from marvis.packs.modeling._runtime import _runtime
from marvis.packs.modeling.errors import ModelingError
from marvis.packs.modeling.evidence import (
    MODELING_TRAINING_EVIDENCE_ARTIFACT_KIND,
    MODEL_BINARY_REF_KIND,
    RAW_SCORE_PRODUCT,
)
from marvis.packs.modeling.evidence_tools import (
    ModelingTrainingEvidenceArtifactBinding,
    build_training_evidence_ref,
    load_historical_modeling_training_evidence_artifacts,
    load_modeling_training_evidence_artifacts,
    require_historical_modeling_training_evidence_artifact_binding_on_connection,
    require_modeling_training_evidence_artifact_binding_on_connection,
)
from marvis.packs.modeling.score_evidence import (
    MAX_MODEL_SCORE_EVIDENCE_JSON_BYTES,
    MODEL_SCORE_EVIDENCE_ARTIFACT_KIND,
    MODEL_SCORE_EVIDENCE_PRODUCER_VERSION,
    MODEL_SCORE_INPUT_SPACE,
    MODEL_SCORE_VECTOR_ARTIFACT_KIND,
    ModelScoreEvidenceError,
    build_model_score_evidence_envelope,
    build_single_model_score_evidence,
    canonical_model_score_evidence_json,
    model_score_evidence_from_json,
)
from marvis.packs.modeling.scoring import _ModelArtifactScorer
from marvis.repositories.task_artifacts import (
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
)


MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_TOOL_SCHEMA_VERSION = (
    "modeling.materialize-model-score-evidence-v2-tool.v1"
)
MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_ORIGIN_TOOL = (
    "modeling.materialize_model_score_evidence_v2"
)
MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_AUDIT_KIND = (
    "modeling.model_score_evidence.published"
)
MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_HOUSEKEEPING_WARNING_AUDIT_KIND = (
    "modeling.model_score_evidence.housekeeping_warning"
)

_MODEL_SCORE_TASK_LOCK_TIMEOUT_SECONDS = 0
_MAX_INPUT_JSON_BYTES = 1024 * 1024
_MAX_MODEL_BINARY_BYTES = 8 * 1024 * 1024 * 1024
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_INPUT_FIELDS = frozenset({"training_evidence_ref"})
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
_OUTPUT_FIELDS = frozenset(
    {
        "schema_version",
        "evidence_id",
        "evidence_content_hash",
        "single_model_evidence_id",
        "single_model_evidence_content_hash",
        "input_space",
        "training_evidence_ref",
        "artifacts",
        "resource_budgets",
        "governance",
        "content_hash",
    }
)
_OUTPUT_ARTIFACTS_FIELDS = frozenset({"score_vector", "score_evidence"})
_OUTPUT_ARTIFACT_FIELDS = frozenset(
    {"artifact_id", "kind", "filename", "content_hash", "download_url"}
)
_GOVERNANCE_FIELDS = frozenset(
    {"not_compared", "not_selected", "not_adopted", "not_deployed"}
)
_VECTOR_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "format",
        "artifact_role",
        "task_id",
        "request_hash",
        "training_evidence_ref",
        "training_evidence_id",
        "training_evidence_content_hash",
        "training_evidence_artifact_id",
        "training_evidence_artifact_content_hash",
        "model_artifact_id",
        "model_binary_artifact_id",
        "model_binary_artifact_content_hash",
        "sample_design_id",
        "sample_design_content_hash",
        "sample_membership_id",
        "sample_membership_content_hash",
        "dataset_id",
        "dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "input_space",
        "score_product",
        "load_calibration",
        "replay_preprocessing",
        "rows_scored_exactly_once",
        "row_count",
        "row_ordinal",
        "score_dtype",
        "score_min",
        "score_max",
        "vector_content_hash",
    }
)
_EVIDENCE_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "format",
        "artifact_role",
        "task_id",
        "request_hash",
        "training_evidence_ref",
        "training_evidence_id",
        "training_evidence_content_hash",
        "training_evidence_artifact_id",
        "training_evidence_artifact_content_hash",
        "model_artifact_id",
        "model_binary_artifact_id",
        "model_binary_artifact_content_hash",
        "sample_design_id",
        "sample_design_content_hash",
        "sample_membership_id",
        "sample_membership_content_hash",
        "dataset_id",
        "dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "input_space",
        "score_product",
        "score_vector_artifact_id",
        "score_vector_artifact_content_hash",
        "score_evidence_id",
        "score_evidence_content_hash",
        "score_evidence_artifact_content_hash",
        "single_model_evidence_id",
        "single_model_evidence_content_hash",
        "resource_budgets",
        "governance",
    }
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
_BOUNDARY_ERRORS = (
    ArtifactTransactionError,
    DataLayerError,
    ModelScoreEvidenceError,
    ModelScoreVectorError,
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
)


@dataclass(frozen=True)
class ModelScoreEvidenceArtifactBinding:
    """Fully re-authenticated score-vector/evidence artifact pair."""

    task_id: str
    training: ModelingTrainingEvidenceArtifactBinding
    vector_record: dict[str, Any]
    evidence_record: dict[str, Any]
    vector: ModelScoreVector
    envelope: dict[str, Any]
    vector_path: Path
    evidence_path: Path


def tool_materialize_model_score_evidence_v2(
    inputs: dict,
    ctx,
) -> dict[str, Any]:
    """Plugin entrypoint for governed model-score evidence."""

    return run_materialize_model_score_evidence_v2(inputs, ctx, _runtime(ctx))


def run_materialize_model_score_evidence_v2(
    inputs: object,
    ctx,
    runtime,
) -> dict[str, Any]:
    """Score the exact active training dataset and publish immutable evidence."""

    request = _validate_inputs(inputs)
    task_id = _text(ctx.task_id, "task_id")
    lock = FileLock(
        str(_model_score_task_lock_path(runtime.settings.tasks_dir, task_id=task_id))
    )
    try:
        lock.acquire(timeout=_MODEL_SCORE_TASK_LOCK_TIMEOUT_SECONDS)
    except FileLockTimeout as exc:
        raise ModelingError(
            "governed model score evidence is already running for this task"
        ) from exc

    private_dir: Path | None = None
    try:
        training = _load_training(runtime, task_id=task_id, request=request)
        canonical_ref = build_training_evidence_ref(training)
        if canonical_ref != request["training_evidence_ref"]:
            raise ModelingError(
                "training_evidence_ref drifted from authenticated training evidence"
            )
        paths = _publication_paths(
            runtime.settings.tasks_dir,
            task_id=task_id,
            request_hash=_request_hash(request),
            create=True,
        )
        replay = _load_existing_publication(
            runtime,
            task_id=task_id,
            training_ref=canonical_ref,
            vector_path=paths["vector"],
            evidence_path=paths["evidence"],
        )
        if replay is not None:
            return _tool_output(replay)

        frame = _active_training_frame(runtime, training=training)
        _require_bound_scoring_contract(training, frame=frame)
        private_dir = _create_private_model_dir(
            runtime.settings.tasks_dir,
            task_id=task_id,
        )
        private_artifact = _copy_verified_model_binary(
            training,
            private_dir=private_dir,
        )
        scores = _score_private_model(
            private_artifact,
            private_dir=private_dir,
            frame=frame,
        )
        binding = _publish_score_evidence(
            runtime,
            task_id=task_id,
            request=request,
            training=training,
            frame=frame,
            scores=scores,
            vector_path=paths["vector"],
            evidence_path=paths["evidence"],
        )
        return _tool_output(binding)
    except _BOUNDARY_ERRORS as exc:
        raise ModelingError(str(exc)) from exc
    finally:
        if private_dir is not None:
            _cleanup_private_model_dir(private_dir)
        lock.release()


def validate_materialize_model_score_evidence_v2_tool_output(
    value: object,
    *,
    runtime,
    task_id: str,
) -> dict[str, Any]:
    """Rebuild the Tool output from current authenticated artifact bytes."""

    obj = _object(value, "materialize_model_score_evidence_v2 output")
    _exact_fields(
        obj,
        _OUTPUT_FIELDS,
        "materialize_model_score_evidence_v2 output",
    )
    if obj["schema_version"] != MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_TOOL_SCHEMA_VERSION:
        raise ModelingError("model-score-evidence output schema_version is invalid")
    if obj["input_space"] != MODEL_SCORE_INPUT_SPACE:
        raise ModelingError("model-score-evidence output input_space is invalid")
    training_ref = _training_ref(obj["training_evidence_ref"])
    artifacts = _object(obj["artifacts"], "model-score-evidence output artifacts")
    _exact_fields(
        artifacts,
        _OUTPUT_ARTIFACTS_FIELDS,
        "model-score-evidence output artifacts",
    )
    vector_output = _output_artifact(
        artifacts["score_vector"],
        name="score_vector output artifact",
        expected_kind=MODEL_SCORE_VECTOR_ARTIFACT_KIND,
        task_id=task_id,
    )
    evidence_output = _output_artifact(
        artifacts["score_evidence"],
        name="score_evidence output artifact",
        expected_kind=MODEL_SCORE_EVIDENCE_ARTIFACT_KIND,
        task_id=task_id,
    )
    binding = load_model_score_evidence_artifacts(
        runtime,
        task_id=task_id,
        evidence_artifact_id=evidence_output["artifact_id"],
        expected_evidence_artifact_content_hash=evidence_output["content_hash"],
        score_vector_artifact_id=vector_output["artifact_id"],
        expected_score_vector_artifact_content_hash=vector_output["content_hash"],
    )
    expected = _tool_output(binding)
    if training_ref != binding.envelope["training_evidence_ref"] or obj != expected:
        raise ModelingError("model-score-evidence output drifted from live artifacts")
    return dict(obj)


def load_model_score_evidence_artifacts(
    runtime,
    *,
    task_id: str,
    evidence_artifact_id: str,
    expected_evidence_artifact_content_hash: str,
    score_vector_artifact_id: str | None = None,
    expected_score_vector_artifact_content_hash: str | None = None,
) -> ModelScoreEvidenceArtifactBinding:
    """Load one score evidence pair, normalizing typed boundary failures."""

    try:
        return _load_model_score_evidence_artifacts(
            runtime,
            task_id=task_id,
            evidence_artifact_id=evidence_artifact_id,
            expected_evidence_artifact_content_hash=(
                expected_evidence_artifact_content_hash
            ),
            score_vector_artifact_id=score_vector_artifact_id,
            expected_score_vector_artifact_content_hash=(
                expected_score_vector_artifact_content_hash
            ),
            require_current_training=True,
        )
    except _BOUNDARY_ERRORS as exc:
        raise ModelingError(str(exc)) from exc


def load_historical_model_score_evidence_artifacts(
    runtime,
    *,
    task_id: str,
    evidence_artifact_id: str,
    expected_evidence_artifact_content_hash: str,
    score_vector_artifact_id: str | None = None,
    expected_score_vector_artifact_content_hash: str | None = None,
) -> ModelScoreEvidenceArtifactBinding:
    """Load immutable score evidence without requiring its sample to be head."""

    try:
        return _load_model_score_evidence_artifacts(
            runtime,
            task_id=task_id,
            evidence_artifact_id=evidence_artifact_id,
            expected_evidence_artifact_content_hash=(
                expected_evidence_artifact_content_hash
            ),
            score_vector_artifact_id=score_vector_artifact_id,
            expected_score_vector_artifact_content_hash=(
                expected_score_vector_artifact_content_hash
            ),
            require_current_training=False,
        )
    except _BOUNDARY_ERRORS as exc:
        raise ModelingError(str(exc)) from exc


def _load_model_score_evidence_artifacts(
    runtime,
    *,
    task_id: str,
    evidence_artifact_id: str,
    expected_evidence_artifact_content_hash: str,
    score_vector_artifact_id: str | None = None,
    expected_score_vector_artifact_content_hash: str | None = None,
    require_current_training: bool,
) -> ModelScoreEvidenceArtifactBinding:
    """Load and rebuild one score evidence pair from live governed sources."""

    normalized_task = _text(task_id, "task_id")
    evidence_record = _registered_record(
        runtime,
        task_id=normalized_task,
        artifact_id=_hash(evidence_artifact_id, "evidence_artifact_id"),
        kind=MODEL_SCORE_EVIDENCE_ARTIFACT_KIND,
        expected_content_hash=_hash(
            expected_evidence_artifact_content_hash,
            "expected_evidence_artifact_content_hash",
        ),
    )
    evidence_path = Path(str(evidence_record["path"]))
    evidence_raw = _read_regular_file(
        evidence_path,
        root=Path(runtime.settings.tasks_dir),
        expected_hash=str(evidence_record["content_hash"]),
        maximum_bytes=MAX_MODEL_SCORE_EVIDENCE_JSON_BYTES,
    )
    cached = _decode_json_object(evidence_raw, "model score evidence JSON")
    training_ref = _training_ref(cached.get("training_evidence_ref"))
    score_ref = _artifact_ref(
        cached.get("score_vector_ref"),
        "score_vector_ref",
        expected_kind=MODEL_SCORE_VECTOR_ARTIFACT_KIND,
    )
    requested_vector_id = (
        score_ref["ref_id"]
        if score_vector_artifact_id is None
        else _hash(score_vector_artifact_id, "score_vector_artifact_id")
    )
    requested_vector_hash = (
        score_ref["content_hash"]
        if expected_score_vector_artifact_content_hash is None
        else _hash(
            expected_score_vector_artifact_content_hash,
            "expected_score_vector_artifact_content_hash",
        )
    )
    if (
        requested_vector_id != score_ref["ref_id"]
        or requested_vector_hash != score_ref["content_hash"]
    ):
        raise ModelingError(
            "score vector output reference does not match evidence JSON"
        )
    if (score_vector_artifact_id is None) != (
        expected_score_vector_artifact_content_hash is None
    ):
        raise ModelingError(
            "score vector artifact id and expected hash must be supplied together"
        )
    training_loader = (
        load_modeling_training_evidence_artifacts
        if require_current_training
        else load_historical_modeling_training_evidence_artifacts
    )
    training = training_loader(
        runtime,
        task_id=normalized_task,
        **training_ref,
    )
    canonical_ref = build_training_evidence_ref(training)
    if canonical_ref != training_ref:
        raise ModelingError("model score evidence training reference changed")
    request = {"training_evidence_ref": canonical_ref}
    paths = _publication_paths(
        runtime.settings.tasks_dir,
        task_id=normalized_task,
        request_hash=_request_hash(request),
        create=False,
    )
    _require_registered_path(
        evidence_record,
        paths["evidence"],
        name="model score evidence",
    )
    vector_record = _registered_record(
        runtime,
        task_id=normalized_task,
        artifact_id=requested_vector_id,
        kind=MODEL_SCORE_VECTOR_ARTIFACT_KIND,
        expected_content_hash=requested_vector_hash,
    )
    _require_registered_path(
        vector_record,
        paths["vector"],
        name="model score vector",
    )
    vector = validate_model_score_vector(
        paths["vector"],
        expected_content_hash=str(vector_record["content_hash"]),
        expected_row_count=training.sample.source_binding.row_count,
    )
    frame = _active_training_frame(runtime, training=training)
    _require_bound_scoring_contract(training, frame=frame)
    expected = _rebuild_envelope(
        task_id=normalized_task,
        training_ref=canonical_ref,
        training=training,
        frame=frame,
        vector=vector,
        vector_record=vector_record,
    )
    envelope = model_score_evidence_from_json(
        evidence_raw,
        sample_design_bundle=training.sample.bundle,
        training_evidence=training.evidence,
        expected_training_evidence_ref=canonical_ref,
        score_vector=vector,
    )
    if envelope != expected:
        raise ModelingError(
            "model score evidence JSON drifted from live dataset and vector"
        )
    _require_provenance(
        vector_record["provenance"],
        expected=_vector_provenance(
            task_id=normalized_task,
            request=request,
            training=training,
            vector=vector,
        ),
        fields=_VECTOR_PROVENANCE_FIELDS,
        name="model score vector provenance",
    )
    _require_provenance(
        evidence_record["provenance"],
        expected=_evidence_provenance(
            task_id=normalized_task,
            request=request,
            training=training,
            vector_record=vector_record,
            envelope=envelope,
            evidence_file_hash=str(evidence_record["content_hash"]),
        ),
        fields=_EVIDENCE_PROVENANCE_FIELDS,
        name="model score evidence provenance",
    )
    return ModelScoreEvidenceArtifactBinding(
        task_id=normalized_task,
        training=training,
        vector_record=vector_record,
        evidence_record=evidence_record,
        vector=vector,
        envelope=envelope,
        vector_path=paths["vector"],
        evidence_path=paths["evidence"],
    )


def require_model_score_evidence_artifact_binding_on_connection(
    conn,
    binding: ModelScoreEvidenceArtifactBinding,
) -> None:
    """Re-authenticate a loaded score-evidence pair under a caller write lock."""

    try:
        _require_model_score_evidence_artifact_binding_on_connection(
            conn,
            binding,
            require_current_training=True,
        )
    except _BOUNDARY_ERRORS as exc:
        raise ModelingError(str(exc)) from exc


def require_historical_model_score_evidence_artifact_binding_on_connection(
    conn,
    binding: ModelScoreEvidenceArtifactBinding,
) -> None:
    """Re-authenticate score evidence without requiring its sample to be head."""

    try:
        _require_model_score_evidence_artifact_binding_on_connection(
            conn,
            binding,
            require_current_training=False,
        )
    except _BOUNDARY_ERRORS as exc:
        raise ModelingError(str(exc)) from exc


def _require_model_score_evidence_artifact_binding_on_connection(
    conn,
    binding: ModelScoreEvidenceArtifactBinding,
    *,
    require_current_training: bool,
) -> None:
    if not isinstance(binding, ModelScoreEvidenceArtifactBinding):
        raise ModelingError("model-score-evidence artifact binding is invalid")
    if not getattr(conn, "in_transaction", False):
        raise ModelingError(
            "model-score-evidence revalidation requires an active transaction"
        )
    task_id = _text(binding.task_id, "task_id")
    if binding.training.task_id != task_id:
        raise ModelingError("model-score-evidence training task binding changed")
    if require_current_training:
        require_modeling_training_evidence_artifact_binding_on_connection(
            conn,
            binding.training,
        )
    else:
        require_historical_modeling_training_evidence_artifact_binding_on_connection(
            conn,
            binding.training,
        )
    canonical_ref = build_training_evidence_ref(binding.training)
    request = {"training_evidence_ref": canonical_ref}
    if (
        binding.envelope.get("task_id") != task_id
        or binding.envelope.get("training_evidence_ref") != canonical_ref
    ):
        raise ModelingError("model-score-evidence envelope training binding changed")
    try:
        tasks_root = binding.vector_path.parents[2]
        training_tasks_root = binding.training.model_binary_path.parents[2]
    except IndexError as exc:
        raise ModelingError(
            "model-score-evidence artifact path is not task-owned"
        ) from exc
    if tasks_root != training_tasks_root:
        raise ModelingError(
            "model-score-evidence artifacts escaped the training task root"
        )
    paths = _publication_paths(
        tasks_root,
        task_id=task_id,
        request_hash=_request_hash(request),
        create=False,
    )
    if (
        binding.vector_path != paths["vector"]
        or binding.evidence_path != paths["evidence"]
    ):
        raise ModelingError("model-score-evidence canonical publication path changed")
    _require_binding_record_identity(
        binding.vector_record,
        task_id=task_id,
        expected_kind=MODEL_SCORE_VECTOR_ARTIFACT_KIND,
        expected_path=binding.vector_path,
        name="model score vector TaskArtifact",
    )
    _require_binding_record_identity(
        binding.evidence_record,
        task_id=task_id,
        expected_kind=MODEL_SCORE_EVIDENCE_ARTIFACT_KIND,
        expected_path=binding.evidence_path,
        name="model score evidence TaskArtifact",
    )
    _require_task_artifact_row_on_connection(
        conn,
        task_id=task_id,
        record=binding.vector_record,
        name="model score vector TaskArtifact",
    )
    _require_task_artifact_row_on_connection(
        conn,
        task_id=task_id,
        record=binding.evidence_record,
        name="model score evidence TaskArtifact",
    )
    vector = validate_model_score_vector(
        binding.vector_path,
        expected_content_hash=str(binding.vector_record["content_hash"]),
        expected_row_count=binding.training.sample.source_binding.row_count,
    )
    if vector != binding.vector:
        raise ModelingError("model score vector changed after authenticated loading")
    evidence_raw = _read_regular_file(
        binding.evidence_path,
        root=tasks_root,
        expected_hash=str(binding.evidence_record["content_hash"]),
        maximum_bytes=MAX_MODEL_SCORE_EVIDENCE_JSON_BYTES,
    )
    envelope = model_score_evidence_from_json(
        evidence_raw,
        sample_design_bundle=binding.training.sample.bundle,
        training_evidence=binding.training.evidence,
        expected_training_evidence_ref=canonical_ref,
        score_vector=vector,
    )
    if envelope != binding.envelope:
        raise ModelingError("model score evidence changed after authenticated loading")
    _require_provenance(
        binding.vector_record["provenance"],
        expected=_vector_provenance(
            task_id=task_id,
            request=request,
            training=binding.training,
            vector=vector,
        ),
        fields=_VECTOR_PROVENANCE_FIELDS,
        name="model score vector provenance",
    )
    _require_provenance(
        binding.evidence_record["provenance"],
        expected=_evidence_provenance(
            task_id=task_id,
            request=request,
            training=binding.training,
            vector_record=binding.vector_record,
            envelope=envelope,
            evidence_file_hash=str(binding.evidence_record["content_hash"]),
        ),
        fields=_EVIDENCE_PROVENANCE_FIELDS,
        name="model score evidence provenance",
    )


def _publish_score_evidence(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
    training: ModelingTrainingEvidenceArtifactBinding,
    frame: pd.DataFrame,
    scores: np.ndarray,
    vector_path: Path,
    evidence_path: Path,
) -> ModelScoreEvidenceArtifactBinding:
    out_dir = vector_path.parent
    uow = ArtifactUnitOfWork()
    rollback_attempted_under_lock = False
    vector_record: dict[str, Any] | None = None
    evidence_record: dict[str, Any] | None = None
    vector: ModelScoreVector | None = None
    envelope: dict[str, Any] | None = None
    try:
        vector_stage = uow.stage_file(out_dir, vector_path.name)
        staged_vector = write_model_score_vector(vector_stage.path, scores)
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                require_modeling_training_evidence_artifact_binding_on_connection(
                    conn,
                    training,
                )
                _require_dataset_frame_still_bound(training, frame=frame)
                uow.promote_all()
                vector = validate_model_score_vector(
                    vector_path,
                    expected_content_hash=staged_vector.content_hash,
                    expected_row_count=len(frame),
                )
                vector_record = runtime.task_artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=MODEL_SCORE_VECTOR_ARTIFACT_KIND,
                    path=str(vector_path),
                    content_hash=vector.content_hash,
                    origin_tool=MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_ORIGIN_TOOL,
                    provenance=_vector_provenance(
                        task_id=task_id,
                        request=request,
                        training=training,
                        vector=vector,
                    ),
                )
                envelope = _rebuild_envelope(
                    task_id=task_id,
                    training_ref=request["training_evidence_ref"],
                    training=training,
                    frame=frame,
                    vector=vector,
                    vector_record=vector_record,
                )
                evidence_raw = canonical_model_score_evidence_json(
                    envelope,
                    sample_design_bundle=training.sample.bundle,
                    training_evidence=training.evidence,
                    expected_training_evidence_ref=request["training_evidence_ref"],
                    score_vector=vector,
                ).encode("utf-8")
                evidence_file_hash = hashlib.sha256(evidence_raw).hexdigest()
                evidence_stage = uow.stage_file(out_dir, evidence_path.name)
                _write_private_bytes(evidence_stage.path, evidence_raw)
                uow.promote_all()
                _read_regular_file(
                    evidence_path,
                    root=Path(runtime.settings.tasks_dir),
                    expected_hash=evidence_file_hash,
                    maximum_bytes=MAX_MODEL_SCORE_EVIDENCE_JSON_BYTES,
                )
                evidence_record = runtime.task_artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=MODEL_SCORE_EVIDENCE_ARTIFACT_KIND,
                    path=str(evidence_path),
                    content_hash=evidence_file_hash,
                    origin_tool=MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_ORIGIN_TOOL,
                    provenance=_evidence_provenance(
                        task_id=task_id,
                        request=request,
                        training=training,
                        vector_record=vector_record,
                        envelope=envelope,
                        evidence_file_hash=evidence_file_hash,
                    ),
                )
                runtime.repo.write_audit_on_connection(
                    conn,
                    kind=MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_AUDIT_KIND,
                    target_ref=envelope["evidence_id"],
                    inputs_hash=_request_hash(request),
                    outcome="succeeded",
                    detail={
                        "task_id": task_id,
                        "experiment_id": training.experiment.id,
                        "model_artifact_id": training.model_artifact.id,
                        "training_evidence_artifact_id": training.evidence_record["id"],
                        "score_vector_artifact_id": vector_record["id"],
                        "score_evidence_artifact_id": evidence_record["id"],
                        "score_evidence_id": envelope["evidence_id"],
                        "single_model_evidence_id": envelope["single_model_evidence"][
                            "evidence_id"
                        ],
                        **_governance(),
                    },
                )
                _require_dataset_frame_still_bound(training, frame=frame)
                pending_binding = ModelScoreEvidenceArtifactBinding(
                    task_id=task_id,
                    training=training,
                    vector_record=vector_record,
                    evidence_record=evidence_record,
                    vector=vector,
                    envelope=envelope,
                    vector_path=vector_path,
                    evidence_path=evidence_path,
                )
                require_model_score_evidence_artifact_binding_on_connection(
                    conn,
                    pending_binding,
                )
                conn.commit()
            except Exception:
                rollback_attempted_under_lock = True
                uow.rollback()
                raise
    except Exception:
        if not rollback_attempted_under_lock:
            uow.rollback()
        raise

    try:
        uow.commit()
    except Exception as exc:
        _record_housekeeping_warning_best_effort(
            runtime,
            task_id=task_id,
            request=request,
            envelope=envelope,
            error=exc,
        )
    assert vector_record is not None
    assert evidence_record is not None
    assert vector is not None
    assert envelope is not None
    return ModelScoreEvidenceArtifactBinding(
        task_id=task_id,
        training=training,
        vector_record=vector_record,
        evidence_record=evidence_record,
        vector=vector,
        envelope=envelope,
        vector_path=vector_path,
        evidence_path=evidence_path,
    )


def _rebuild_envelope(
    *,
    task_id: str,
    training_ref: Mapping[str, Any],
    training: ModelingTrainingEvidenceArtifactBinding,
    frame: pd.DataFrame,
    vector: ModelScoreVector,
    vector_record: Mapping[str, Any],
) -> dict[str, Any]:
    training_artifact_ref = {
        "kind": MODELING_TRAINING_EVIDENCE_ARTIFACT_KIND,
        "ref_id": str(training.evidence_record["id"]),
        "content_hash": str(training.evidence_record["content_hash"]),
    }
    model_ref = {
        "kind": MODEL_BINARY_REF_KIND,
        "ref_id": str(training.model_binary_record["id"]),
        "content_hash": str(training.model_binary_record["content_hash"]),
    }
    score_ref = {
        "kind": MODEL_SCORE_VECTOR_ARTIFACT_KIND,
        "ref_id": str(vector_record["id"]),
        "content_hash": str(vector_record["content_hash"]),
    }
    single = build_single_model_score_evidence(
        sample_design_bundle=training.sample.bundle,
        membership_masks=training.sample.membership["masks"],
        frame=frame,
        scores=vector.scores,
        training_evidence_ref=training_artifact_ref,
        model_ref=model_ref,
        score_ref=score_ref,
        features=training.evidence["training_contract"]["features"],
    )
    return build_model_score_evidence_envelope(
        task_id=task_id,
        training_evidence_ref=training_ref,
        training_evidence=training.evidence,
        sample_design_bundle=training.sample.bundle,
        model_ref=model_ref,
        score_ref=score_ref,
        score_vector=vector,
        single_model_evidence=single,
    )


def _load_training(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
) -> ModelingTrainingEvidenceArtifactBinding:
    return load_modeling_training_evidence_artifacts(
        runtime,
        task_id=task_id,
        **request["training_evidence_ref"],
    )


def _active_training_frame(
    runtime,
    *,
    training: ModelingTrainingEvidenceArtifactBinding,
) -> pd.DataFrame:
    source = training.sample.source_binding
    path = source.dataset_path
    if _dataset_content_hash(path) != source.dataset_content_hash:
        raise ModelingError("bound active training dataset bytes changed")
    frame = runtime.backend.read_frame(path)
    if not isinstance(frame, pd.DataFrame):
        raise ModelingError("bound active training dataset is not a DataFrame")
    if len(frame) != source.row_count:
        raise ModelingError("bound active training dataset row count changed")
    if len(frame) <= 0 or len(frame) > MAX_MODEL_SCORE_VECTOR_ROWS:
        raise ModelingError("bound active training dataset exceeds score row budget")
    if _dataset_content_hash(path) != source.dataset_content_hash:
        raise ModelingError("bound active training dataset bytes changed while reading")
    return frame


def _require_bound_scoring_contract(
    training: ModelingTrainingEvidenceArtifactBinding,
    *,
    frame: pd.DataFrame,
) -> None:
    source = training.sample.source_binding
    evidence = training.evidence
    sample_binding = evidence["sample_design_binding"]
    dataset_ref = sample_binding["dataset_ref"]
    config = evidence["training_contract"]["train_config"]
    features = list(evidence["training_contract"]["features"])
    scoring = evidence["model_artifact"]["scoring_metadata"]
    if (
        dataset_ref
        != {
            "dataset_id": source.dataset_id,
            "content_hash": source.dataset_content_hash,
            "role": "active",
        }
        or config["dataset_id"] != source.dataset_id
    ):
        raise ModelingError(
            "training evidence does not prove the bound active training dataset"
        )
    if (
        features != list(config["features"])
        or features != list(training.model_artifact.feature_list)
        or features != list(scoring["feature_list"])
    ):
        raise ModelingError("training evidence feature binding changed before scoring")
    missing = sorted(set(features) - {str(column) for column in frame.columns})
    if missing:
        raise ModelingError(
            "bound active training dataset is missing model features: "
            + ", ".join(missing)
        )
    if (
        scoring["score_direction"] != "higher_is_riskier"
        or scoring["score_product"] != RAW_SCORE_PRODUCT
        or scoring["calibration_status"] != "not_applied"
        or training.model_artifact.score_direction != "higher_is_riskier"
    ):
        raise ModelingError(
            "training evidence does not prove an uncalibrated higher-risk score"
        )


def _require_dataset_frame_still_bound(
    training: ModelingTrainingEvidenceArtifactBinding,
    *,
    frame: pd.DataFrame,
) -> None:
    source = training.sample.source_binding
    if len(frame) != source.row_count:
        raise ModelingError("scored active dataset row count changed")
    if _dataset_content_hash(source.dataset_path) != source.dataset_content_hash:
        raise ModelingError("bound active training dataset bytes changed before write")


def _dataset_content_hash(path: Path) -> str:
    try:
        return sha256_file(path)
    except OSError as exc:
        raise ModelingError("bound active training dataset is unavailable") from exc


def _score_private_model(
    artifact,
    *,
    private_dir: Path,
    frame: pd.DataFrame,
) -> np.ndarray:
    scorer = _ModelArtifactScorer(
        artifact,
        base_dir=private_dir,
        load_calibration=False,
        replay_preprocessing=False,
    )
    scores = np.asarray(
        scorer.score(frame, use_calibration=False),
        dtype=np.float64,
    )
    if scores.ndim != 1 or len(scores) != len(frame):
        raise ModelingError(
            "model scorer did not return exactly one score for every active row"
        )
    if not np.all(np.isfinite(scores)):
        raise ModelingError("model scorer returned non-finite scores")
    if np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ModelingError("model scorer returned values outside [0, 1]")
    return np.ascontiguousarray(scores, dtype=np.float64)


def _copy_verified_model_binary(
    training: ModelingTrainingEvidenceArtifactBinding,
    *,
    private_dir: Path,
):
    source = training.model_binary_path
    expected_hash = str(training.model_binary_record["content_hash"])
    name = Path(training.model_artifact.model_path).name
    if (
        not name
        or Path(training.model_artifact.model_path).is_absolute()
        or Path(training.model_artifact.model_path).name
        != training.model_artifact.model_path
    ):
        raise ModelingError("registered model binary filename is unsafe")
    destination = private_dir / name
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    source_fd = -1
    destination_fd = -1
    digest = hashlib.sha256()
    try:
        source_fd = os.open(source, flags)
        before = os.fstat(source_fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_MODEL_BINARY_BYTES
        ):
            raise ModelingError("registered model binary file is invalid")
        destination_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        copied = 0
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            copied += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                if written <= 0:
                    raise OSError("short write while copying model binary")
                view = view[written:]
        os.fsync(destination_fd)
        after = os.fstat(source_fd)
        if copied != before.st_size or (
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_mode,
        ) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_mode,
        ):
            raise ModelingError("registered model binary changed while copying")
        copied_hash = digest.hexdigest()
        if not hmac.compare_digest(copied_hash, expected_hash):
            raise ModelingError("registered model binary hash changed before scoring")
    except ModelingError:
        raise
    except OSError as exc:
        raise ModelingError(
            "registered model binary could not be copied privately"
        ) from exc
    finally:
        if destination_fd >= 0:
            os.close(destination_fd)
        if source_fd >= 0:
            os.close(source_fd)
    try:
        os.chmod(destination, 0o400)
    except OSError as exc:
        raise ModelingError("private model copy could not be protected") from exc
    _require_regular_file_hash(
        destination,
        root=private_dir,
        expected_hash=expected_hash,
        maximum_bytes=_MAX_MODEL_BINARY_BYTES,
    )
    copied_artifact = replace(training.model_artifact, model_path=name)
    return copied_artifact


def _create_private_model_dir(tasks_root: Path, *, task_id: str) -> Path:
    task_dir = _model_score_task_lock_path(tasks_root, task_id=task_id).parent
    parent = task_dir / ".model_score_evidence_v2.staging"
    if parent.is_symlink():
        raise ModelingError("model score evidence staging parent is unsafe")
    created: Path | None = None
    try:
        parent.mkdir(parents=False, exist_ok=True, mode=0o700)
        if parent.is_symlink() or not parent.is_dir():
            raise ModelingError("model score evidence staging parent is unavailable")
        created = Path(tempfile.mkdtemp(prefix="run.", dir=parent))
        os.chmod(created, 0o700)
        created.resolve(strict=True).relative_to(task_dir.resolve(strict=True))
    except ModelingError:
        if created is not None:
            _cleanup_private_model_dir(created)
        raise
    except (OSError, ValueError) as exc:
        if created is not None:
            _cleanup_private_model_dir(created)
        raise ModelingError(
            "private model scoring directory could not be created"
        ) from exc
    return created


def _cleanup_private_model_dir(private_dir: Path) -> None:
    stage = Path(private_dir)
    parent = stage.parent
    try:
        if stage.is_symlink() or not stage.name.startswith("run."):
            return
        stage.resolve(strict=False).relative_to(parent.resolve(strict=True))
        shutil.rmtree(stage)
    except (OSError, ValueError):
        return
    try:
        parent.rmdir()
    except OSError:
        pass


def _load_existing_publication(
    runtime,
    *,
    task_id: str,
    training_ref: Mapping[str, Any],
    vector_path: Path,
    evidence_path: Path,
) -> ModelScoreEvidenceArtifactBinding | None:
    records = runtime.task_artifacts.list_for_task(task_id)
    vector_records = [
        item
        for item in records
        if item["kind"] == MODEL_SCORE_VECTOR_ARTIFACT_KIND
        and Path(str(item["path"])) == vector_path
    ]
    evidence_records = [
        item
        for item in records
        if item["kind"] == MODEL_SCORE_EVIDENCE_ARTIFACT_KIND
        and Path(str(item["path"])) == evidence_path
    ]
    path_collisions = [
        item
        for item in records
        if Path(str(item["path"])) in {vector_path, evidence_path}
        and item not in vector_records
        and item not in evidence_records
    ]
    if path_collisions or len(vector_records) > 1 or len(evidence_records) > 1:
        raise ModelingError(
            "model score evidence publication path has conflicting registry rows"
        )
    if not vector_records and not evidence_records:
        if (
            vector_path.exists()
            or vector_path.is_symlink()
            or evidence_path.exists()
            or evidence_path.is_symlink()
        ):
            raise ModelingError(
                "unregistered model score evidence files already occupy publication path"
            )
        return None
    if len(vector_records) != 1 or len(evidence_records) != 1:
        raise ModelingError(
            "model score evidence publication is an incomplete artifact pair"
        )
    binding = load_model_score_evidence_artifacts(
        runtime,
        task_id=task_id,
        evidence_artifact_id=str(evidence_records[0]["id"]),
        expected_evidence_artifact_content_hash=str(
            evidence_records[0]["content_hash"]
        ),
        score_vector_artifact_id=str(vector_records[0]["id"]),
        expected_score_vector_artifact_content_hash=str(
            vector_records[0]["content_hash"]
        ),
    )
    if binding.envelope["training_evidence_ref"] != training_ref:
        raise ModelingError(
            "existing score evidence does not match requested training evidence"
        )
    return binding


def _publication_paths(
    tasks_root: Path,
    *,
    task_id: str,
    request_hash: str,
    create: bool,
) -> dict[str, Path]:
    task_dir = _model_score_task_lock_path(tasks_root, task_id=task_id).parent
    out_dir = task_dir / "model_score_evidence"
    if out_dir.is_symlink():
        raise ModelingError("model score evidence output directory is unsafe")
    if create:
        try:
            out_dir.mkdir(parents=False, exist_ok=True)
        except OSError as exc:
            raise ModelingError(
                "model score evidence output directory is unavailable"
            ) from exc
    if out_dir.is_symlink() or not out_dir.is_dir():
        raise ModelingError("model score evidence output directory is unavailable")
    try:
        out_dir.resolve(strict=True).relative_to(task_dir.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ModelingError(
            "model score evidence output directory escaped task"
        ) from exc
    token = _hash(request_hash, "request_hash")
    return {
        "vector": out_dir / f"{token}.scores.parquet",
        "evidence": out_dir / f"{token}.model_score_evidence.json",
    }


def _model_score_task_lock_path(tasks_root: Path, *, task_id: str) -> Path:
    root = Path(tasks_root)
    task_dir = root / _text(task_id, "task_id")
    if root.is_symlink() or task_dir.is_symlink():
        raise ModelingError("model score evidence task lock path is unsafe")
    try:
        resolved_root = root.resolve(strict=True)
        resolved_task = task_dir.resolve(strict=True)
        resolved_task.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise ModelingError(
            "model score evidence task lock path is unavailable or escaped"
        ) from exc
    if not resolved_task.is_dir():
        raise ModelingError("model score evidence task directory is unavailable")
    lock_path = resolved_task / ".materialize_model_score_evidence_v2.lock"
    if lock_path.is_symlink():
        raise ModelingError("model score evidence task lock file is unsafe")
    return lock_path


def _validate_inputs(value: object) -> dict[str, Any]:
    obj = _object(value, "materialize_model_score_evidence_v2 inputs")
    _exact_fields(
        obj,
        _INPUT_FIELDS,
        "materialize_model_score_evidence_v2 inputs",
    )
    _preflight_json(obj, "materialize_model_score_evidence_v2 inputs")
    return {"training_evidence_ref": _training_ref(obj["training_evidence_ref"])}


def _training_ref(value: object) -> dict[str, Any]:
    obj = _object(value, "training_evidence_ref")
    _exact_fields(obj, _TRAINING_REF_FIELDS, "training_evidence_ref")
    sample_obj = _object(
        obj["sample_design_ref"],
        "training_evidence_ref.sample_design_ref",
    )
    _exact_fields(
        sample_obj,
        _SAMPLE_REF_FIELDS,
        "training_evidence_ref.sample_design_ref",
    )
    sample = {
        field: (
            _text(sample_obj[field], f"sample_design_ref.{field}")
            if field in {"expected_bundle_id", "expected_sample_design_id"}
            else _hash(sample_obj[field], f"sample_design_ref.{field}")
        )
        for field in sorted(_SAMPLE_REF_FIELDS)
    }
    result: dict[str, Any] = {"sample_design_ref": sample}
    for field in (
        "model_binary_artifact_id",
        "expected_model_binary_artifact_content_hash",
        "evidence_artifact_id",
        "expected_evidence_artifact_content_hash",
        "expected_evidence_content_hash",
    ):
        result[field] = _hash(obj[field], f"training_evidence_ref.{field}")
    for field in (
        "expected_experiment_id",
        "expected_model_artifact_id",
        "expected_evidence_id",
    ):
        result[field] = _text(obj[field], f"training_evidence_ref.{field}")
    return result


def _artifact_ref(
    value: object,
    name: str,
    *,
    expected_kind: str,
) -> dict[str, str]:
    obj = _object(value, name)
    _exact_fields(obj, frozenset({"kind", "ref_id", "content_hash"}), name)
    result = {
        "kind": _text(obj["kind"], f"{name}.kind"),
        "ref_id": _hash(obj["ref_id"], f"{name}.ref_id"),
        "content_hash": _hash(obj["content_hash"], f"{name}.content_hash"),
    }
    if result["kind"] != expected_kind:
        raise ModelingError(f"{name}.kind is invalid")
    return result


def _tool_output(
    binding: ModelScoreEvidenceArtifactBinding,
) -> dict[str, Any]:
    envelope = binding.envelope
    single = envelope["single_model_evidence"]
    body = {
        "schema_version": MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_TOOL_SCHEMA_VERSION,
        "evidence_id": envelope["evidence_id"],
        "evidence_content_hash": envelope["content_hash"],
        "single_model_evidence_id": single["evidence_id"],
        "single_model_evidence_content_hash": single["content_hash"],
        "input_space": MODEL_SCORE_INPUT_SPACE,
        "training_evidence_ref": envelope["training_evidence_ref"],
        "artifacts": {
            "score_vector": _artifact_output(
                task_id=binding.task_id,
                record=binding.vector_record,
            ),
            "score_evidence": _artifact_output(
                task_id=binding.task_id,
                record=binding.evidence_record,
            ),
        },
        "resource_budgets": envelope["resource_budgets"],
        "governance": _governance(),
    }
    return {
        **body,
        "content_hash": hashlib.sha256(
            _canonical_json(body).encode("utf-8")
        ).hexdigest(),
    }


def _artifact_output(
    *,
    task_id: str,
    record: Mapping[str, Any],
) -> dict[str, str]:
    artifact_id = str(record["id"])
    content_hash = str(record["content_hash"])
    return {
        "artifact_id": artifact_id,
        "kind": str(record["kind"]),
        "filename": Path(str(record["path"])).name,
        "content_hash": content_hash,
        "download_url": (
            f"/api/tasks/{quote(task_id, safe='')}"
            f"/task-artifacts/{quote(artifact_id, safe='')}/download"
            f"?expected_content_hash={content_hash}"
        ),
    }


def _output_artifact(
    value: object,
    *,
    name: str,
    expected_kind: str,
    task_id: str,
) -> dict[str, str]:
    obj = _object(value, name)
    _exact_fields(obj, _OUTPUT_ARTIFACT_FIELDS, name)
    artifact_id = _hash(obj["artifact_id"], f"{name}.artifact_id")
    content_hash = _hash(obj["content_hash"], f"{name}.content_hash")
    kind = _text(obj["kind"], f"{name}.kind")
    filename = _text(obj["filename"], f"{name}.filename")
    if kind != expected_kind or Path(filename).name != filename:
        raise ModelingError(f"{name} kind or filename is invalid")
    expected_url = (
        f"/api/tasks/{quote(task_id, safe='')}"
        f"/task-artifacts/{quote(artifact_id, safe='')}/download"
        f"?expected_content_hash={content_hash}"
    )
    if obj["download_url"] != expected_url:
        raise ModelingError(f"{name} download_url drifted")
    return {
        "artifact_id": artifact_id,
        "kind": kind,
        "filename": filename,
        "content_hash": content_hash,
        "download_url": expected_url,
    }


def _vector_provenance(
    *,
    task_id: str,
    request: Mapping[str, Any],
    training: ModelingTrainingEvidenceArtifactBinding,
    vector: ModelScoreVector,
) -> dict[str, Any]:
    common = _lineage_provenance(
        task_id=task_id,
        request=request,
        training=training,
    )
    return {
        "schema_version": MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_TOOL_SCHEMA_VERSION,
        "producer_version": MODEL_SCORE_EVIDENCE_PRODUCER_VERSION,
        "format": "parquet",
        "artifact_role": "model_score_vector",
        **common,
        "input_space": MODEL_SCORE_INPUT_SPACE,
        "score_product": RAW_SCORE_PRODUCT,
        "load_calibration": False,
        "replay_preprocessing": False,
        "rows_scored_exactly_once": True,
        "row_count": vector.row_count,
        "row_ordinal": {
            "start": 0,
            "stop": vector.row_count,
            "step": 1,
        },
        "score_dtype": "float64",
        "score_min": vector.score_min,
        "score_max": vector.score_max,
        "vector_content_hash": vector.content_hash,
    }


def _evidence_provenance(
    *,
    task_id: str,
    request: Mapping[str, Any],
    training: ModelingTrainingEvidenceArtifactBinding,
    vector_record: Mapping[str, Any],
    envelope: Mapping[str, Any],
    evidence_file_hash: str,
) -> dict[str, Any]:
    common = _lineage_provenance(
        task_id=task_id,
        request=request,
        training=training,
    )
    single = envelope["single_model_evidence"]
    return {
        "schema_version": MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_TOOL_SCHEMA_VERSION,
        "producer_version": MODEL_SCORE_EVIDENCE_PRODUCER_VERSION,
        "format": "json",
        "artifact_role": "model_score_evidence",
        **common,
        "input_space": MODEL_SCORE_INPUT_SPACE,
        "score_product": RAW_SCORE_PRODUCT,
        "score_vector_artifact_id": str(vector_record["id"]),
        "score_vector_artifact_content_hash": str(vector_record["content_hash"]),
        "score_evidence_id": str(envelope["evidence_id"]),
        "score_evidence_content_hash": str(envelope["content_hash"]),
        "score_evidence_artifact_content_hash": evidence_file_hash,
        "single_model_evidence_id": str(single["evidence_id"]),
        "single_model_evidence_content_hash": str(single["content_hash"]),
        "resource_budgets": envelope["resource_budgets"],
        "governance": _governance(),
    }


def _lineage_provenance(
    *,
    task_id: str,
    request: Mapping[str, Any],
    training: ModelingTrainingEvidenceArtifactBinding,
) -> dict[str, Any]:
    sample_binding = training.evidence["sample_design_binding"]
    return {
        "task_id": task_id,
        "request_hash": _request_hash(request),
        "training_evidence_ref": request["training_evidence_ref"],
        "training_evidence_id": training.evidence["evidence_id"],
        "training_evidence_content_hash": training.evidence["content_hash"],
        "training_evidence_artifact_id": training.evidence_record["id"],
        "training_evidence_artifact_content_hash": training.evidence_record[
            "content_hash"
        ],
        "model_artifact_id": training.model_artifact.id,
        "model_binary_artifact_id": training.model_binary_record["id"],
        "model_binary_artifact_content_hash": training.model_binary_record[
            "content_hash"
        ],
        "sample_design_id": sample_binding["sample_design_ref"]["sample_design_id"],
        "sample_design_content_hash": sample_binding["sample_design_ref"][
            "content_hash"
        ],
        "sample_membership_id": sample_binding["membership_ref"]["membership_id"],
        "sample_membership_content_hash": sample_binding["membership_ref"][
            "content_hash"
        ],
        "dataset_id": sample_binding["dataset_ref"]["dataset_id"],
        "dataset_content_hash": sample_binding["dataset_ref"]["content_hash"],
        "workspace_revision": sample_binding["workspace_ref"]["revision"],
        "workspace_generation": sample_binding["workspace_ref"]["generation"],
    }


def _registered_record(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    kind: str,
    expected_content_hash: str,
) -> dict[str, Any]:
    record = runtime.task_artifacts.get_for_task(task_id, artifact_id)
    if record is None:
        raise ModelingError("model score evidence TaskArtifact was not found in task")
    if record["kind"] != kind:
        raise ModelingError("model score evidence TaskArtifact kind changed")
    if not hmac.compare_digest(
        str(record["content_hash"]),
        expected_content_hash,
    ):
        raise ModelingError("model score evidence TaskArtifact hash changed")
    if record["origin_tool"] != MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_ORIGIN_TOOL:
        raise ModelingError("model score evidence TaskArtifact origin changed")
    return record


def _require_binding_record_identity(
    value: object,
    *,
    task_id: str,
    expected_kind: str,
    expected_path: Path,
    name: str,
) -> None:
    record = _object(value, name)
    _exact_fields(record, _TASK_ARTIFACT_RECORD_FIELDS, name)
    if (
        _hash(record["id"], f"{name}.id") != record["id"]
        or _text(record["task_id"], f"{name}.task_id") != task_id
        or _text(record["kind"], f"{name}.kind") != expected_kind
        or Path(_text(record["path"], f"{name}.path")) != expected_path
        or _hash(record["content_hash"], f"{name}.content_hash")
        != record["content_hash"]
        or _text(record["origin_tool"], f"{name}.origin_tool")
        != MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_ORIGIN_TOOL
    ):
        raise ModelingError(f"{name} identity changed")
    _text(record["created_at"], f"{name}.created_at")
    _object(record["provenance"], f"{name}.provenance")


def _require_task_artifact_row_on_connection(
    conn,
    *,
    task_id: str,
    record: Mapping[str, Any],
    name: str,
) -> None:
    row = conn.execute(
        """
        SELECT id, task_id, kind, path, content_hash, origin_tool,
               provenance_json, created_at
          FROM task_artifacts
         WHERE task_id = ? AND id = ?
        """,
        (task_id, str(record["id"])),
    ).fetchone()
    if row is None:
        raise ModelingError(f"{name} disappeared before commit")
    try:
        provenance = json.loads(
            str(row["provenance_json"]),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (ModelingError, json.JSONDecodeError, TypeError) as exc:
        raise ModelingError(f"{name} provenance became invalid") from exc
    live = {
        "id": str(row["id"]),
        "task_id": str(row["task_id"]),
        "kind": str(row["kind"]),
        "path": str(row["path"]),
        "content_hash": str(row["content_hash"]),
        "origin_tool": str(row["origin_tool"]),
        "provenance": provenance,
        "created_at": str(row["created_at"]),
    }
    if live != dict(record):
        raise ModelingError(f"{name} changed before commit")


def _require_registered_path(
    record: Mapping[str, Any],
    expected: Path,
    *,
    name: str,
) -> None:
    if Path(str(record["path"])) != expected:
        raise ModelingError(f"{name} TaskArtifact path changed")


def _require_provenance(
    value: object,
    *,
    expected: Mapping[str, Any],
    fields: frozenset[str],
    name: str,
) -> None:
    obj = _object(value, name)
    _exact_fields(obj, fields, name)
    if obj != expected:
        raise ModelingError(f"{name} drifted from live evidence")


def _read_regular_file(
    path: Path,
    *,
    root: Path,
    expected_hash: str,
    maximum_bytes: int,
) -> bytes:
    before = _regular_file_stat(path, root=root, maximum_bytes=maximum_bytes)
    descriptor = -1
    digest = hashlib.sha256()
    parts: list[bytes] = []
    try:
        descriptor = _open_regular_file_descriptor(path)
        if _descriptor_stat(descriptor, maximum_bytes=maximum_bytes) != before:
            raise ModelingError("registered artifact changed before reading")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            parts.append(chunk)
        if _descriptor_stat(descriptor, maximum_bytes=maximum_bytes) != before:
            raise ModelingError("registered artifact changed while reading")
    except ModelingError:
        raise
    except OSError as exc:
        raise ModelingError("registered artifact could not be read") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    raw = b"".join(parts)
    after = _regular_file_stat(path, root=root, maximum_bytes=maximum_bytes)
    if before != after or len(raw) != before[1]:
        raise ModelingError("registered artifact changed while reading")
    if not hmac.compare_digest(digest.hexdigest(), expected_hash):
        raise ModelingError("registered artifact content hash changed")
    return raw


def _require_regular_file_hash(
    path: Path,
    *,
    root: Path,
    expected_hash: str,
    maximum_bytes: int,
) -> str:
    before = _regular_file_stat(path, root=root, maximum_bytes=maximum_bytes)
    descriptor = -1
    digest = hashlib.sha256()
    try:
        descriptor = _open_regular_file_descriptor(path)
        if _descriptor_stat(descriptor, maximum_bytes=maximum_bytes) != before:
            raise ModelingError("registered artifact changed before hashing")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        if _descriptor_stat(descriptor, maximum_bytes=maximum_bytes) != before:
            raise ModelingError("registered artifact changed while hashing")
    except ModelingError:
        raise
    except OSError as exc:
        raise ModelingError("registered artifact could not be hashed") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if _regular_file_stat(path, root=root, maximum_bytes=maximum_bytes) != before:
        raise ModelingError("registered artifact changed while hashing")
    observed = digest.hexdigest()
    if not hmac.compare_digest(observed, expected_hash):
        raise ModelingError("registered artifact content hash changed")
    return observed


def _regular_file_stat(
    path: Path,
    *,
    root: Path,
    maximum_bytes: int,
) -> tuple[int, int, int, int]:
    if path.is_symlink():
        raise ModelingError("registered artifact must not be a symlink")
    try:
        resolved_root = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(resolved_root)
        observed = os.lstat(path)
    except (OSError, ValueError) as exc:
        raise ModelingError(
            "registered artifact path is unavailable or escaped"
        ) from exc
    if not stat.S_ISREG(observed.st_mode):
        raise ModelingError("registered artifact must be a regular file")
    if observed.st_size <= 0 or observed.st_size > maximum_bytes:
        raise ModelingError("registered artifact file size is invalid")
    return (
        int(observed.st_ino),
        int(observed.st_size),
        int(observed.st_mtime_ns),
        int(observed.st_mode),
    )


def _open_regular_file_descriptor(path: Path) -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(path, flags)


def _descriptor_stat(
    descriptor: int,
    *,
    maximum_bytes: int,
) -> tuple[int, int, int, int]:
    observed = os.fstat(descriptor)
    if not stat.S_ISREG(observed.st_mode):
        raise ModelingError("registered artifact must be a regular file")
    if observed.st_size <= 0 or observed.st_size > maximum_bytes:
        raise ModelingError("registered artifact file size is invalid")
    return (
        int(observed.st_ino),
        int(observed.st_size),
        int(observed.st_mtime_ns),
        int(observed.st_mode),
    )


def _write_private_bytes(path: Path, raw: bytes) -> None:
    try:
        with path.open("wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise ModelingError("model score evidence JSON could not be staged") from exc


def _record_housekeeping_warning_best_effort(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
    envelope: Mapping[str, Any] | None,
    error: Exception,
) -> None:
    try:
        runtime.repo.write_audit(
            kind=(MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_HOUSEKEEPING_WARNING_AUDIT_KIND),
            target_ref=(
                request["training_evidence_ref"]["expected_evidence_id"]
                if envelope is None
                else str(envelope["evidence_id"])
            ),
            inputs_hash=_request_hash(request),
            outcome="warning",
            detail={
                "task_id": task_id,
                "score_evidence_id": (
                    None if envelope is None else envelope["evidence_id"]
                ),
                "warning": ("artifact backup cleanup failed after database commit"),
                "error_type": type(error).__name__,
                "error": str(error)[:500],
                "publication_committed": True,
            },
        )
    except Exception:
        pass


def _governance() -> dict[str, bool]:
    return {
        "not_compared": True,
        "not_selected": True,
        "not_adopted": True,
        "not_deployed": True,
    }


def _decode_json_object(raw: bytes, name: str) -> dict[str, Any]:
    if len(raw) > MAX_MODEL_SCORE_EVIDENCE_JSON_BYTES:
        raise ModelingError(f"{name} exceeds byte budget")
    try:
        decoded = json.loads(
            raw,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except ModelingError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ModelingError(f"{name} is not valid bounded JSON") from exc
    return dict(_object(decoded, name))


def _object_without_duplicate_keys(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ModelingError("model score evidence JSON contains duplicate keys")
        result[key] = value
    return result


def _preflight_json(value: object, name: str) -> None:
    try:
        raw = _canonical_json(value).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ModelingError(f"{name} must contain finite JSON values") from exc
    if len(raw) > _MAX_INPUT_JSON_BYTES:
        raise ModelingError(f"{name} exceeds byte budget")


def _request_hash(request: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(request).encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _object(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ModelingError(f"{name} must be an object with string keys")
    return value


def _exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unknown:
            details.append("unknown=" + ",".join(unknown))
        raise ModelingError(f"{name} fields are invalid ({'; '.join(details)})")


def _text(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise ModelingError(f"{name} must be canonical non-empty text")
    return value


def _hash(value: object, name: str) -> str:
    text = _text(value, name)
    if _HASH_RE.fullmatch(text) is None:
        raise ModelingError(f"{name} must be lowercase SHA-256")
    return text


__all__ = [
    "MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_AUDIT_KIND",
    "MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_HOUSEKEEPING_WARNING_AUDIT_KIND",
    "MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_ORIGIN_TOOL",
    "MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_TOOL_SCHEMA_VERSION",
    "MODEL_SCORE_EVIDENCE_ARTIFACT_KIND",
    "MODEL_SCORE_VECTOR_ARTIFACT_KIND",
    "ModelScoreEvidenceArtifactBinding",
    "load_historical_model_score_evidence_artifacts",
    "load_model_score_evidence_artifacts",
    "require_historical_model_score_evidence_artifact_binding_on_connection",
    "require_model_score_evidence_artifact_binding_on_connection",
    "run_materialize_model_score_evidence_v2",
    "tool_materialize_model_score_evidence_v2",
    "validate_materialize_model_score_evidence_v2_tool_output",
]
