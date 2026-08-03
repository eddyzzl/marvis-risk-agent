"""Governed training Tool for Strategy V2 model evidence.

This module is intentionally separate from ``train_model``/``train_models``.
The historical tools keep their schemas and lifecycle unchanged; this vertical
adds the stricter trust boundary needed by the Strategy Workbench:

* the active dataset, target and missing-label policy come only from a verified
  StrategySampleDesign V2 artifact pair;
* a collision-safe private split is materialized directly from decoded
  membership, and its selector masks are proved row-for-row against it;
* only the governed risk-union rows reach the recipe;
* recipes run in a task-local private directory, then their model/meta outputs,
  immutable registrations, canonical evidence and audits share one caller-owned
  ``ArtifactUnitOfWork`` plus ``BEGIN IMMEDIATE`` transaction.

Once the database commit succeeds, artifact-backup housekeeping is best effort:
its failure is audited as a warning and can never turn a durable successful
publication into an ordinary Tool failure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
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

from marvis.artifacts import (
    ArtifactUnitOfWork,
    StagedArtifact,
    TransactionalArtifactStore,
)
from marvis.files import sha256_file
from marvis.packs.modeling._common import BINARY_MODELING_RECIPES
from marvis.packs.modeling._runtime import _artifact_base_dir, _runtime
from marvis.packs.modeling.contracts import Experiment, ModelArtifact, TrainConfig
from marvis.packs.modeling.evidence import (
    GOVERNED_SPLIT_MATERIALIZATION_SOURCE,
    GOVERNED_SPLIT_MATERIALIZATION_VALUES,
    MAX_TRAINING_EVIDENCE_JSON_BYTES,
    MODELING_TRAINING_EVIDENCE_ARTIFACT_KIND,
    MODEL_BINARY_REF_KIND,
    NATIVE_SAMPLE_MEMBERSHIP_ARTIFACT_KIND,
    RAW_SCORE_PRODUCT,
    SAMPLE_DESIGN_BUNDLE_ARTIFACT_KIND,
    SAMPLE_MEMBERSHIP_ARTIFACT_KIND,
    ModelingTrainingEvidenceError,
    build_model_binary_artifact_ref,
    build_modeling_target_contract,
    build_modeling_training_evidence,
    build_task_artifact_ref,
    build_training_split_mask_hashes,
    canonical_modeling_training_evidence_json,
    modeling_scoring_metadata_from_artifact,
    modeling_training_evidence_from_json,
)
from marvis.packs.modeling.errors import ModelingError
from marvis.packs.modeling.train_tools import (
    _preprocessing_chain_traceable,
    _preprocessing_steps_for_training,
    _train_recipe,
)
from marvis.packs.modeling.training_dataset import TrainingDataset
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.sample_design_v2_tools import (
    load_any_strategy_sample_design_v2_artifacts,
    load_historical_any_strategy_sample_design_v2_artifacts,
    require_any_strategy_sample_design_v2_artifact_binding_on_connection,
    require_historical_any_strategy_sample_design_v2_artifact_binding_on_connection,
)
from marvis.repositories.task_artifacts import (
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
)


TRAIN_MODEL_WITH_EVIDENCE_V2_TOOL_SCHEMA_VERSION = (
    "modeling.train-model-with-evidence-v2-tool.v1.2"
)
TRAINING_EVIDENCE_ARTIFACT_SCHEMA_VERSION = (
    "modeling.training-evidence-artifact.v1.2"
)
TRAIN_MODEL_WITH_EVIDENCE_V2_ORIGIN_TOOL = (
    "modeling.train_model_with_evidence_v2"
)
TRAIN_MODEL_WITH_EVIDENCE_V2_AUDIT_KIND = (
    "modeling.training_evidence.published"
)
TRAIN_MODEL_WITH_EVIDENCE_V2_HOUSEKEEPING_WARNING_AUDIT_KIND = (
    "modeling.training_evidence.housekeeping_warning"
)

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_INPUT_JSON_BYTES = 1024 * 1024
_MAX_MODEL_BINARY_BYTES = 8 * 1024 * 1024 * 1024
_MAX_RECIPE_META_BYTES = 64 * 1024 * 1024
_TRAINING_TASK_LOCK_TIMEOUT_SECONDS = 0
_INTERNAL_SPLIT_SOURCE = GOVERNED_SPLIT_MATERIALIZATION_SOURCE
_INTERNAL_SPLIT_COLUMN_BASE = "__marvis_governed_split_v2__"
_INTERNAL_SPLIT_COLUMN_RE = re.compile(
    r"^__marvis_governed_split_v2(?:_[1-9][0-9]*)?__$"
)
_INTERNAL_SPLIT_VALUES = dict(GOVERNED_SPLIT_MATERIALIZATION_VALUES)
_INPUT_FIELDS = frozenset(
    {
        "sample_design_ref",
        "recipe",
        "features",
        "params",
        "seed",
        "early_stopping_rounds",
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
_SPLIT_NAMES = ("train", "test", "oot")
_OUTPUT_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "model_artifact_id",
        "evidence_id",
        "evidence_content_hash",
        "score_product",
        "sample_design_ref",
        "artifacts",
        "governance",
    }
)
_OUTPUT_ARTIFACTS_FIELDS = frozenset({"model_binary", "training_evidence"})
_OUTPUT_ARTIFACT_FIELDS = frozenset(
    {"artifact_id", "kind", "filename", "content_hash", "download_url"}
)
_GOVERNANCE_FIELDS = frozenset(
    {"not_selected", "not_calibrated", "not_adopted", "not_deployed"}
)
_MODEL_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "format",
        "artifact_role",
        "task_id",
        "experiment_id",
        "model_artifact_id",
        "algorithm",
        "model_path",
        "model_binary_artifact_content_hash",
        "scoring_metadata_hash",
        "train_config_hash",
        "metrics_snapshot_content_hash",
        "target_encoding_content_hash",
        "dataset_id",
        "dataset_content_hash",
        "sample_design_id",
        "sample_design_content_hash",
        "sample_membership_id",
        "sample_membership_content_hash",
        "sample_membership_artifact_id",
        "sample_membership_artifact_content_hash",
        "sample_bundle_artifact_id",
        "sample_bundle_artifact_content_hash",
        "split_source",
        "internal_split_column",
        "internal_split_values",
        "internal_split_contract_hash",
    }
)
_EVIDENCE_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "format",
        "artifact_role",
        "task_id",
        "experiment_id",
        "model_artifact_id",
        "model_binary_artifact_id",
        "model_binary_artifact_content_hash",
        "scoring_metadata_hash",
        "train_config_hash",
        "metrics_snapshot_content_hash",
        "target_encoding_content_hash",
        "evidence_id",
        "evidence_content_hash",
        "evidence_artifact_content_hash",
        "dataset_id",
        "dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "sample_design_id",
        "sample_design_content_hash",
        "sample_membership_id",
        "sample_membership_content_hash",
        "sample_membership_artifact_id",
        "sample_membership_artifact_content_hash",
        "sample_bundle_artifact_id",
        "sample_bundle_artifact_content_hash",
        "split_source",
        "internal_split_column",
        "internal_split_values",
        "internal_split_contract_hash",
        "split_proof_content_hash",
        "governance",
    }
)

_BOUNDARY_ERRORS = (
    ModelingTrainingEvidenceError,
    StrategyError,
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
)


@dataclass(frozen=True)
class ModelingTrainingEvidenceArtifactBinding:
    """Authenticated model-binary/training-evidence artifact pair."""

    task_id: str
    sample: Any
    experiment: Experiment
    model_artifact: ModelArtifact
    model_binary_record: dict[str, Any]
    evidence_record: dict[str, Any]
    evidence: dict[str, Any]
    model_binary_path: Path
    evidence_path: Path


def build_training_evidence_ref(
    binding: ModelingTrainingEvidenceArtifactBinding,
) -> dict[str, Any]:
    """Build the complete authenticated input ref expected by downstream Tools."""

    if not isinstance(binding, ModelingTrainingEvidenceArtifactBinding):
        raise ModelingError("training-evidence artifact binding is invalid")
    sample = binding.sample
    design = sample.bundle["sample_design"]
    reference = {
        "sample_design_ref": {
            "membership_artifact_id": sample.membership_artifact_id,
            "expected_membership_artifact_content_hash": (
                sample.membership_artifact_content_hash
            ),
            "bundle_artifact_id": sample.bundle_artifact_id,
            "expected_bundle_artifact_content_hash": (
                sample.bundle_artifact_content_hash
            ),
            "expected_bundle_id": sample.bundle["bundle_id"],
            "expected_sample_design_id": design["sample_design_id"],
            "expected_sample_design_content_hash": design["content_hash"],
        },
        "model_binary_artifact_id": str(binding.model_binary_record["id"]),
        "expected_model_binary_artifact_content_hash": str(
            binding.model_binary_record["content_hash"]
        ),
        "evidence_artifact_id": str(binding.evidence_record["id"]),
        "expected_evidence_artifact_content_hash": str(
            binding.evidence_record["content_hash"]
        ),
        "expected_experiment_id": binding.experiment.id,
        "expected_model_artifact_id": binding.model_artifact.id,
        "expected_evidence_id": str(binding.evidence["evidence_id"]),
        "expected_evidence_content_hash": str(
            binding.evidence["content_hash"]
        ),
    }
    # Reuse every normal input boundary before exposing the reference.
    reference["sample_design_ref"] = _sample_ref(
        reference["sample_design_ref"]
    )
    for name in (
        "model_binary_artifact_id",
        "expected_model_binary_artifact_content_hash",
        "evidence_artifact_id",
        "expected_evidence_artifact_content_hash",
        "expected_evidence_content_hash",
    ):
        reference[name] = _hash(reference[name], name)
    for name in (
        "expected_experiment_id",
        "expected_model_artifact_id",
        "expected_evidence_id",
    ):
        reference[name] = _text(reference[name], name)
    return reference


@dataclass
class _DeferredLatestStagedArtifact:
    """Publish the shared latest pointer only after the database commits.

    Artifact-specific model metadata is immutable and promoted before the DB
    transaction commits.  ``model_meta.json`` is only a legacy convenience
    pointer and legacy writers do not share the V2 task lock.  Deferring this
    one item's promotion to UoW commit means a failed V2 transaction never
    reads, deletes, or restores that shared pointer, eliminating a rollback
    race that could otherwise overwrite a concurrent legacy publication.
    """

    staged: StagedArtifact

    @property
    def path(self) -> Path:
        return self.staged.path

    def promote(self) -> Path:
        # ArtifactUnitOfWork.promote_all() runs before the DB commit.  Keep the
        # shared, non-authoritative latest pointer staged until commit().
        return self.staged.path

    def commit(self) -> Path:
        self.staged.promote()
        return self.staged.commit()

    def rollback(self) -> None:
        self.staged.rollback()


def tool_train_model_with_evidence_v2(inputs: dict, ctx) -> dict[str, Any]:
    """Plugin entrypoint for governed single-model Strategy V2 training."""

    return run_train_model_with_evidence_v2(inputs, ctx, _runtime(ctx))


def run_train_model_with_evidence_v2(
    inputs: object,
    ctx,
    runtime,
) -> dict[str, Any]:
    """Train one native binary model and publish authenticated evidence."""

    request = _validate_inputs(inputs)
    task_id = _text(ctx.task_id, "task_id")
    lock = FileLock(
        str(_training_task_lock_path(runtime.settings.tasks_dir, task_id=task_id))
    )
    try:
        lock.acquire(timeout=_TRAINING_TASK_LOCK_TIMEOUT_SECONDS)
    except FileLockTimeout as exc:
        raise ModelingError(
            "governed model training is already running for this task"
        ) from exc
    training_stage_dir: Path | None = None
    try:
        sample = _load_sample(runtime, task_id=task_id, request=request)
        frame, membership_masks, risk_mask = _training_frame_and_masks(
            runtime,
            sample=sample,
            request=request,
        )
        internal_split = _build_internal_split_contract(frame)
        config = _training_config(
            runtime,
            sample=sample,
            request=request,
            internal_split=internal_split,
        )
        _validate_governed_training_weights(
            frame,
            membership_masks=membership_masks,
            config=config,
        )
        risk_frame = frame.loc[risk_mask].copy()
        if risk_frame.empty:
            raise ModelingError("governed risk training population is empty")
        _materialize_private_governed_split(
            risk_frame,
            membership_masks=membership_masks,
            risk_membership_mask=risk_mask,
            internal_split=internal_split,
        )
        selector_masks = _private_split_selector_masks(
            risk_frame,
            risk_membership_mask=risk_mask,
            internal_split=internal_split,
        )
        split_hashes = build_training_split_mask_hashes(
            sample_design_bundle=sample.bundle,
            selector_masks=selector_masks,
            membership_masks=membership_masks,
            risk_membership_mask=risk_mask,
        )
        target_contract = _governed_target_contract(sample)
        _encode_private_model_target(
            risk_frame,
            target_contract=target_contract,
        )
        _require_binary_train_test_support(
            risk_frame,
            config=config,
        )
        single_class_oot = _has_labeled_single_class_oot(
            risk_frame,
            config=config,
        )
        artifact_dir = _artifact_base_dir(runtime.settings, task_id)
        _require_artifact_directory_boundary(
            runtime.settings.tasks_dir,
            task_id=task_id,
            artifact_dir=artifact_dir,
            allow_missing=True,
        )
        training_stage_dir = _create_training_stage_dir(
            runtime.settings.tasks_dir,
            task_id=task_id,
        )
        training_backend = TrainingDataset(
            path=sample.source_binding.dataset_path,
            frame=risk_frame,
        ).backend_adapter(runtime.backend)
        result = _train_recipe(
            request["recipe"],
            training_backend,
            sample.source_binding.dataset_path,
            config,
            out_dir=training_stage_dir,
        )
        if single_class_oot:
            result = _without_single_class_oot_label_metrics(result)
        _, model_hash = _require_staged_recipe_model_binary(
            training_stage_dir,
            artifact=result.artifact,
        )
        experiment = runtime.experiments.prepare(task_id, request["recipe"], config)
        output = _persist_training_evidence(
            runtime,
            task_id=task_id,
            request=request,
            sample=sample,
            experiment=experiment,
            result=result,
            split_hashes=split_hashes,
            training_stage_dir=training_stage_dir,
            model_hash=model_hash,
        )
        # A fresh live load is part of the Tool boundary.  JSON Schema alone is
        # deliberately insufficient for artifact ids supplied by cached output.
        validate_train_model_with_evidence_v2_tool_output(
            output,
            runtime=runtime,
            task_id=task_id,
        )
        return output
    except _BOUNDARY_ERRORS as exc:
        raise ModelingError(str(exc)) from exc
    finally:
        if training_stage_dir is not None:
            _cleanup_training_stage_dir(training_stage_dir)
        lock.release()


def validate_train_model_with_evidence_v2_tool_output(
    value: object,
    *,
    runtime,
    task_id: str,
) -> dict[str, Any]:
    """Validate a Tool envelope through the live immutable registries.

    There is intentionally no pure cached-envelope validator: a self-consistent
    JSON object cannot prove a TaskArtifact id exists in the owning task.
    """

    obj = _object(value, "train_model_with_evidence_v2 output")
    _exact_fields(obj, _OUTPUT_FIELDS, "train_model_with_evidence_v2 output")
    if obj["schema_version"] != TRAIN_MODEL_WITH_EVIDENCE_V2_TOOL_SCHEMA_VERSION:
        raise ModelingError("training-evidence output schema_version is invalid")
    if obj["score_product"] != RAW_SCORE_PRODUCT:
        raise ModelingError("training-evidence output score_product is invalid")
    governance = _governance(obj["governance"])
    sample_ref = _sample_ref(obj["sample_design_ref"])
    artifacts = _object(obj["artifacts"], "training-evidence output artifacts")
    _exact_fields(artifacts, _OUTPUT_ARTIFACTS_FIELDS, "training-evidence output artifacts")
    model_output = _output_artifact(
        artifacts["model_binary"],
        name="model_binary output artifact",
        expected_kind=MODEL_BINARY_REF_KIND,
        task_id=task_id,
    )
    evidence_output = _output_artifact(
        artifacts["training_evidence"],
        name="training_evidence output artifact",
        expected_kind=MODELING_TRAINING_EVIDENCE_ARTIFACT_KIND,
        task_id=task_id,
    )
    binding = load_modeling_training_evidence_artifacts(
        runtime,
        task_id=task_id,
        sample_design_ref=sample_ref,
        model_binary_artifact_id=model_output["artifact_id"],
        expected_model_binary_artifact_content_hash=model_output["content_hash"],
        evidence_artifact_id=evidence_output["artifact_id"],
        expected_evidence_artifact_content_hash=evidence_output["content_hash"],
        expected_experiment_id=_text(obj["experiment_id"], "experiment_id"),
        expected_model_artifact_id=_text(
            obj["model_artifact_id"], "model_artifact_id"
        ),
        expected_evidence_id=_text(obj["evidence_id"], "evidence_id"),
        expected_evidence_content_hash=_hash(
            obj["evidence_content_hash"], "evidence_content_hash"
        ),
    )
    expected = _tool_output(
        task_id=task_id,
        sample_ref=sample_ref,
        experiment=binding.experiment,
        model_artifact=binding.model_artifact,
        evidence=binding.evidence,
        model_record=binding.model_binary_record,
        evidence_record=binding.evidence_record,
    )
    if obj != expected or governance != expected["governance"]:
        raise ModelingError("training-evidence output drifted from live artifacts")
    return obj


def load_modeling_training_evidence_artifacts(
    runtime,
    *,
    task_id: str,
    sample_design_ref: Mapping[str, Any],
    model_binary_artifact_id: str,
    expected_model_binary_artifact_content_hash: str,
    evidence_artifact_id: str,
    expected_evidence_artifact_content_hash: str,
    expected_experiment_id: str,
    expected_model_artifact_id: str,
    expected_evidence_id: str,
    expected_evidence_content_hash: str,
) -> ModelingTrainingEvidenceArtifactBinding:
    """Load and fully re-authenticate one published training evidence pair."""

    return _load_modeling_training_evidence_artifacts(
        runtime,
        task_id=task_id,
        sample_design_ref=sample_design_ref,
        model_binary_artifact_id=model_binary_artifact_id,
        expected_model_binary_artifact_content_hash=(
            expected_model_binary_artifact_content_hash
        ),
        evidence_artifact_id=evidence_artifact_id,
        expected_evidence_artifact_content_hash=(
            expected_evidence_artifact_content_hash
        ),
        expected_experiment_id=expected_experiment_id,
        expected_model_artifact_id=expected_model_artifact_id,
        expected_evidence_id=expected_evidence_id,
        expected_evidence_content_hash=expected_evidence_content_hash,
        require_current_sample=True,
    )


def load_historical_modeling_training_evidence_artifacts(
    runtime,
    *,
    task_id: str,
    sample_design_ref: Mapping[str, Any],
    model_binary_artifact_id: str,
    expected_model_binary_artifact_content_hash: str,
    evidence_artifact_id: str,
    expected_evidence_artifact_content_hash: str,
    expected_experiment_id: str,
    expected_model_artifact_id: str,
    expected_evidence_id: str,
    expected_evidence_content_hash: str,
) -> ModelingTrainingEvidenceArtifactBinding:
    """Load one immutable training-evidence pair without requiring sample head."""

    return _load_modeling_training_evidence_artifacts(
        runtime,
        task_id=task_id,
        sample_design_ref=sample_design_ref,
        model_binary_artifact_id=model_binary_artifact_id,
        expected_model_binary_artifact_content_hash=(
            expected_model_binary_artifact_content_hash
        ),
        evidence_artifact_id=evidence_artifact_id,
        expected_evidence_artifact_content_hash=(
            expected_evidence_artifact_content_hash
        ),
        expected_experiment_id=expected_experiment_id,
        expected_model_artifact_id=expected_model_artifact_id,
        expected_evidence_id=expected_evidence_id,
        expected_evidence_content_hash=expected_evidence_content_hash,
        require_current_sample=False,
    )


def _load_modeling_training_evidence_artifacts(
    runtime,
    *,
    task_id: str,
    sample_design_ref: Mapping[str, Any],
    model_binary_artifact_id: str,
    expected_model_binary_artifact_content_hash: str,
    evidence_artifact_id: str,
    expected_evidence_artifact_content_hash: str,
    expected_experiment_id: str,
    expected_model_artifact_id: str,
    expected_evidence_id: str,
    expected_evidence_content_hash: str,
    require_current_sample: bool,
) -> ModelingTrainingEvidenceArtifactBinding:
    normalized_task = _text(task_id, "task_id")
    sample_request = {"sample_design_ref": _sample_ref(sample_design_ref)}
    sample = _load_sample(
        runtime,
        task_id=normalized_task,
        request=sample_request,
        require_current=require_current_sample,
    )
    model_record = _registered_record(
        runtime,
        task_id=normalized_task,
        artifact_id=_hash(model_binary_artifact_id, "model_binary_artifact_id"),
        kind=MODEL_BINARY_REF_KIND,
        expected_content_hash=_hash(
            expected_model_binary_artifact_content_hash,
            "expected_model_binary_artifact_content_hash",
        ),
    )
    evidence_record = _registered_record(
        runtime,
        task_id=normalized_task,
        artifact_id=_hash(evidence_artifact_id, "evidence_artifact_id"),
        kind=MODELING_TRAINING_EVIDENCE_ARTIFACT_KIND,
        expected_content_hash=_hash(
            expected_evidence_artifact_content_hash,
            "expected_evidence_artifact_content_hash",
        ),
    )
    experiment_id = _text(expected_experiment_id, "expected_experiment_id")
    model_artifact_id = _text(
        expected_model_artifact_id, "expected_model_artifact_id"
    )
    evidence_id = _text(expected_evidence_id, "expected_evidence_id")
    evidence_content_hash = _hash(
        expected_evidence_content_hash, "expected_evidence_content_hash"
    )
    try:
        experiment = runtime.experiments.get(experiment_id)
    except KeyError as exc:
        raise ModelingError(
            "training-evidence experiment was not found in this task"
        ) from exc
    if experiment.task_id != normalized_task or experiment.status != "trained":
        raise ModelingError("training-evidence experiment is not trained in this task")
    if experiment.artifact_id != model_artifact_id:
        raise ModelingError("training-evidence experiment model binding changed")
    model_artifact = runtime.modeling_repo.get_model_artifact(model_artifact_id)
    if model_artifact is None or model_artifact.experiment_id != experiment_id:
        raise ModelingError("training-evidence model artifact binding changed")
    artifact_dir = _artifact_base_dir(runtime.settings, normalized_task)
    model_path, model_hash = _require_model_binary(
        runtime.settings.tasks_dir,
        task_id=normalized_task,
        artifact_dir=artifact_dir,
        artifact=model_artifact,
    )
    if Path(str(model_record["path"])) != model_path or not hmac.compare_digest(
        str(model_record["content_hash"]), model_hash
    ):
        raise ModelingError("registered model binary no longer matches live model bytes")
    evidence_path = artifact_dir / f"{evidence_id}.training_evidence.json"
    _require_registered_path(evidence_record, evidence_path)
    evidence_raw = _read_regular_file(
        evidence_path,
        root=Path(runtime.settings.tasks_dir),
        expected_hash=str(evidence_record["content_hash"]),
        maximum_bytes=MAX_TRAINING_EVIDENCE_JSON_BYTES,
    )
    evidence = modeling_training_evidence_from_json(
        evidence_raw,
        sample_design_bundle=sample.bundle,
    )
    if evidence["evidence_id"] != evidence_id or not hmac.compare_digest(
        evidence["content_hash"], evidence_content_hash
    ):
        raise ModelingError("training-evidence identity changed")
    if evidence["experiment"]["experiment_id"] != experiment_id:
        raise ModelingError("training-evidence experiment reference changed")
    if evidence["model_artifact"]["artifact_id"] != model_artifact_id:
        raise ModelingError("training-evidence model reference changed")
    if evidence["model_artifact"]["model_binary_ref"] != {
        "artifact_id": model_record["id"],
        "kind": MODEL_BINARY_REF_KIND,
        "content_hash": model_record["content_hash"],
        "model_artifact_id": model_artifact_id,
    }:
        raise ModelingError("training-evidence model-binary reference changed")
    _require_live_training_snapshot_binding(
        experiment=experiment,
        model_artifact=model_artifact,
        evidence=evidence,
    )
    expected_model_provenance = _model_provenance(
        task_id=normalized_task,
        sample=sample,
        experiment=experiment,
        model_artifact=model_artifact,
        model_hash=model_hash,
    )
    expected_evidence_provenance = _evidence_provenance(
        task_id=normalized_task,
        sample=sample,
        experiment=experiment,
        model_artifact=model_artifact,
        model_record=model_record,
        evidence=evidence,
        evidence_file_hash=str(evidence_record["content_hash"]),
    )
    _require_provenance(
        model_record["provenance"],
        expected=expected_model_provenance,
        fields=_MODEL_PROVENANCE_FIELDS,
        name="model binary provenance",
    )
    _require_provenance(
        evidence_record["provenance"],
        expected=expected_evidence_provenance,
        fields=_EVIDENCE_PROVENANCE_FIELDS,
        name="training evidence provenance",
    )
    return ModelingTrainingEvidenceArtifactBinding(
        task_id=normalized_task,
        sample=sample,
        experiment=experiment,
        model_artifact=model_artifact,
        model_binary_record=model_record,
        evidence_record=evidence_record,
        evidence=evidence,
        model_binary_path=model_path,
        evidence_path=evidence_path,
    )


def require_modeling_training_evidence_artifact_binding_on_connection(
    conn,
    binding: ModelingTrainingEvidenceArtifactBinding,
) -> None:
    """Re-authenticate training evidence while a downstream writer holds a lock."""

    _require_modeling_training_evidence_artifact_binding_on_connection(
        conn,
        binding,
        require_current_sample=True,
    )


def require_historical_modeling_training_evidence_artifact_binding_on_connection(
    conn,
    binding: ModelingTrainingEvidenceArtifactBinding,
) -> None:
    """Re-authenticate immutable training evidence without requiring sample head."""

    _require_modeling_training_evidence_artifact_binding_on_connection(
        conn,
        binding,
        require_current_sample=False,
    )


def _require_modeling_training_evidence_artifact_binding_on_connection(
    conn,
    binding: ModelingTrainingEvidenceArtifactBinding,
    *,
    require_current_sample: bool,
) -> None:
    if not isinstance(binding, ModelingTrainingEvidenceArtifactBinding):
        raise ModelingError("training-evidence artifact binding is invalid")
    try:
        if require_current_sample:
            require_any_strategy_sample_design_v2_artifact_binding_on_connection(
                conn,
                binding.sample,
            )
        else:
            require_historical_any_strategy_sample_design_v2_artifact_binding_on_connection(
                conn,
                binding.sample,
            )
    except StrategyError as exc:
        raise ModelingError(str(exc)) from exc

    _require_task_artifact_row_on_connection(
        conn,
        task_id=binding.task_id,
        record=binding.model_binary_record,
        name="model binary TaskArtifact",
    )
    _require_task_artifact_row_on_connection(
        conn,
        task_id=binding.task_id,
        record=binding.evidence_record,
        name="training evidence TaskArtifact",
    )
    _require_experiment_row_on_connection(conn, binding=binding)
    _require_model_artifact_row_on_connection(conn, binding=binding)

    tasks_root = binding.model_binary_path.parents[2]
    artifact_dir = binding.model_binary_path.parent
    model_path, model_hash = _require_model_binary(
        tasks_root,
        task_id=binding.task_id,
        artifact_dir=artifact_dir,
        artifact=binding.model_artifact,
    )
    if model_path != binding.model_binary_path or not hmac.compare_digest(
        model_hash,
        str(binding.model_binary_record["content_hash"]),
    ):
        raise ModelingError("model binary file changed before write")

    expected_evidence_path = artifact_dir / (
        f"{binding.evidence['evidence_id']}.training_evidence.json"
    )
    if binding.evidence_path != expected_evidence_path:
        raise ModelingError("training evidence path changed before write")
    evidence_raw = _read_regular_file(
        binding.evidence_path,
        root=tasks_root,
        expected_hash=str(binding.evidence_record["content_hash"]),
        maximum_bytes=MAX_TRAINING_EVIDENCE_JSON_BYTES,
    )
    live_evidence = modeling_training_evidence_from_json(
        evidence_raw,
        sample_design_bundle=binding.sample.bundle,
    )
    if live_evidence != binding.evidence:
        raise ModelingError("training evidence content changed before write")
    _require_live_training_snapshot_binding(
        experiment=binding.experiment,
        model_artifact=binding.model_artifact,
        evidence=live_evidence,
    )
    _require_provenance(
        binding.model_binary_record["provenance"],
        expected=_model_provenance(
            task_id=binding.task_id,
            sample=binding.sample,
            experiment=binding.experiment,
            model_artifact=binding.model_artifact,
            model_hash=model_hash,
        ),
        fields=_MODEL_PROVENANCE_FIELDS,
        name="model binary provenance",
    )
    _require_provenance(
        binding.evidence_record["provenance"],
        expected=_evidence_provenance(
            task_id=binding.task_id,
            sample=binding.sample,
            experiment=binding.experiment,
            model_artifact=binding.model_artifact,
            model_record=binding.model_binary_record,
            evidence=live_evidence,
            evidence_file_hash=str(binding.evidence_record["content_hash"]),
        ),
        fields=_EVIDENCE_PROVENANCE_FIELDS,
        name="training evidence provenance",
    )


def _persist_training_evidence(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
    sample: Any,
    experiment: Experiment,
    result,
    split_hashes: Mapping[str, Any],
    training_stage_dir: Path,
    model_hash: str,
) -> dict[str, Any]:
    artifact_dir = _artifact_base_dir(runtime.settings, task_id)
    model_path = artifact_dir / result.artifact.model_path
    sample_ref = request["sample_design_ref"]
    uow = ArtifactUnitOfWork()
    rollback_attempted_under_lock = False
    model_record: dict[str, Any] | None = None
    evidence_record: dict[str, Any] | None = None
    evidence: dict[str, Any] | None = None
    try:
        staged_hash = _stage_recipe_outputs(
            uow,
            training_stage_dir=training_stage_dir,
            artifact_dir=artifact_dir,
            artifact=result.artifact,
        )
        if not hmac.compare_digest(staged_hash, model_hash):
            raise ModelingError("staged model binary changed before publication")
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                require_any_strategy_sample_design_v2_artifact_binding_on_connection(
                    conn,
                    sample,
                )
                uow.promote_all()
                locked_model_path, locked_model_hash = _require_model_binary(
                    runtime.settings.tasks_dir,
                    task_id=task_id,
                    artifact_dir=artifact_dir,
                    artifact=result.artifact,
                )
                if locked_model_path != model_path or not hmac.compare_digest(
                    locked_model_hash, model_hash
                ):
                    raise ModelingError("trained model binary changed before registration")
                runtime.experiments.create_on_connection(conn, experiment)
                persisted_artifact = runtime.experiments.attach_result_on_connection(
                    conn,
                    experiment.id,
                    result,
                )
                trained_experiment = replace(
                    experiment,
                    metrics=result.metrics,
                    artifact_id=persisted_artifact.id,
                    status="trained",
                )
                model_record = runtime.task_artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=MODEL_BINARY_REF_KIND,
                    path=str(model_path),
                    content_hash=model_hash,
                    origin_tool=TRAIN_MODEL_WITH_EVIDENCE_V2_ORIGIN_TOOL,
                    provenance=_model_provenance(
                        task_id=task_id,
                        sample=sample,
                        experiment=trained_experiment,
                        model_artifact=persisted_artifact,
                        model_hash=model_hash,
                    ),
                )
                evidence = build_modeling_training_evidence(
                    experiment=trained_experiment,
                    model_artifact=persisted_artifact,
                    sample_design_bundle=sample.bundle,
                    membership_artifact_ref=build_task_artifact_ref(
                        artifact_id=sample.membership_artifact_id,
                        kind=(
                            NATIVE_SAMPLE_MEMBERSHIP_ARTIFACT_KIND
                            if sample.bundle["sample_design"]["compatibility"].get(
                                "source_mode"
                            )
                            == "native_active_dataset"
                            else SAMPLE_MEMBERSHIP_ARTIFACT_KIND
                        ),
                        content_hash=sample.membership_artifact_content_hash,
                    ),
                    sample_design_bundle_artifact_ref=build_task_artifact_ref(
                        artifact_id=sample.bundle_artifact_id,
                        kind=SAMPLE_DESIGN_BUNDLE_ARTIFACT_KIND,
                        content_hash=sample.bundle_artifact_content_hash,
                    ),
                    model_binary_artifact_ref=build_model_binary_artifact_ref(
                        artifact_id=str(model_record["id"]),
                        model_artifact_id=persisted_artifact.id,
                        content_hash=model_hash,
                    ),
                    training_split_mask_hashes=split_hashes,
                    nan_labels_dropped=result.nan_labels_dropped,
                )
                _require_live_training_snapshot_binding(
                    experiment=trained_experiment,
                    model_artifact=persisted_artifact,
                    evidence=evidence,
                )
                evidence_raw = canonical_modeling_training_evidence_json(
                    evidence,
                    sample_design_bundle=sample.bundle,
                ).encode("utf-8")
                evidence_file_hash = hashlib.sha256(evidence_raw).hexdigest()
                evidence_path = artifact_dir / (
                    f"{evidence['evidence_id']}.training_evidence.json"
                )
                staged = uow.stage_file(artifact_dir, evidence_path.name)
                staged.path.write_bytes(evidence_raw)
                uow.promote_all()
                _read_regular_file(
                    evidence_path,
                    root=Path(runtime.settings.tasks_dir),
                    expected_hash=evidence_file_hash,
                    maximum_bytes=MAX_TRAINING_EVIDENCE_JSON_BYTES,
                )
                evidence_record = runtime.task_artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=MODELING_TRAINING_EVIDENCE_ARTIFACT_KIND,
                    path=str(evidence_path),
                    content_hash=evidence_file_hash,
                    origin_tool=TRAIN_MODEL_WITH_EVIDENCE_V2_ORIGIN_TOOL,
                    provenance=_evidence_provenance(
                        task_id=task_id,
                        sample=sample,
                        experiment=trained_experiment,
                        model_artifact=persisted_artifact,
                        model_record=model_record,
                        evidence=evidence,
                        evidence_file_hash=evidence_file_hash,
                    ),
                )
                runtime.repo.write_audit_on_connection(
                    conn,
                    kind=TRAIN_MODEL_WITH_EVIDENCE_V2_AUDIT_KIND,
                    target_ref=evidence["evidence_id"],
                    inputs_hash=_request_hash(request),
                    outcome="succeeded",
                    detail={
                        "task_id": task_id,
                        "experiment_id": trained_experiment.id,
                        "model_artifact_id": persisted_artifact.id,
                        "model_binary_artifact_id": model_record["id"],
                        "training_evidence_artifact_id": evidence_record["id"],
                        "sample_design_id": sample.bundle["sample_design"][
                            "sample_design_id"
                        ],
                        "not_selected": True,
                        "not_calibrated": True,
                        "not_adopted": True,
                        "not_deployed": True,
                    },
                )
                # Recheck both source and newly-published bytes immediately
                # before the only database commit in this Tool.
                require_any_strategy_sample_design_v2_artifact_binding_on_connection(
                    conn,
                    sample,
                )
                _require_file_content_hash(
                    model_path,
                    root=Path(runtime.settings.tasks_dir),
                    expected_hash=model_hash,
                    maximum_bytes=_MAX_MODEL_BINARY_BYTES,
                )
                _read_regular_file(
                    evidence_path,
                    root=Path(runtime.settings.tasks_dir),
                    expected_hash=evidence_file_hash,
                    maximum_bytes=MAX_TRAINING_EVIDENCE_JSON_BYTES,
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
        # Files and rows are already durable. Removing UoW backups is
        # housekeeping only and must not reverse the Tool's success semantics.
        uow.commit()
    except Exception as exc:
        _record_housekeeping_warning_best_effort(
            runtime,
            task_id=task_id,
            request=request,
            experiment_id=experiment.id,
            evidence_id=(
                None if evidence is None else str(evidence["evidence_id"])
            ),
            error=exc,
        )
    assert model_record is not None
    assert evidence_record is not None
    assert evidence is not None
    persisted_experiment = runtime.experiments.get(experiment.id)
    persisted_model = runtime.modeling_repo.get_model_artifact(result.artifact.id)
    if persisted_model is None:
        raise ModelingError("trained model artifact disappeared after commit")
    return _tool_output(
        task_id=task_id,
        sample_ref=sample_ref,
        experiment=persisted_experiment,
        model_artifact=persisted_model,
        evidence=evidence,
        model_record=model_record,
        evidence_record=evidence_record,
    )


def _training_task_lock_path(tasks_root: Path, *, task_id: str) -> Path:
    root = Path(tasks_root)
    task_dir = root / task_id
    if root.is_symlink() or task_dir.is_symlink():
        raise ModelingError("training task lock path is unsafe")
    try:
        resolved_root = root.resolve(strict=True)
        resolved_task = task_dir.resolve(strict=True)
        resolved_task.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise ModelingError("training task lock path is unavailable or escaped") from exc
    if not resolved_task.is_dir():
        raise ModelingError("training task directory is unavailable")
    lock_path = resolved_task / ".train_model_with_evidence_v2.lock"
    if lock_path.is_symlink():
        raise ModelingError("training task lock file is unsafe")
    return lock_path


def _create_training_stage_dir(tasks_root: Path, *, task_id: str) -> Path:
    task_dir = _training_task_lock_path(tasks_root, task_id=task_id).parent
    parent = task_dir / ".train_model_with_evidence_v2.staging"
    if parent.is_symlink():
        raise ModelingError("training staging parent is unsafe")
    parent.mkdir(parents=False, exist_ok=True)
    if not parent.is_dir():
        raise ModelingError("training staging parent is unavailable")
    created: Path | None = None
    try:
        created = Path(tempfile.mkdtemp(prefix="run.", dir=parent))
        created.resolve(strict=True).relative_to(task_dir.resolve(strict=True))
    except (OSError, ValueError) as exc:
        if created is not None:
            _cleanup_training_stage_dir(created)
        raise ModelingError("training staging directory could not be created") from exc
    return created


def _cleanup_training_stage_dir(stage_dir: Path) -> None:
    stage = Path(stage_dir)
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


def _stage_recipe_outputs(
    uow: ArtifactUnitOfWork,
    *,
    training_stage_dir: Path,
    artifact_dir: Path,
    artifact: ModelArtifact,
) -> str:
    stage = Path(training_stage_dir)
    artifact_meta_name = f"{artifact.id}.model_meta.json"
    names = [artifact.model_path, artifact_meta_name, "model_meta.json"]
    if artifact.pmml_path:
        names.append(artifact.pmml_path)
    if len(names) != len(set(names)):
        raise ModelingError("recipe output filenames must be distinct")
    for name in names:
        relative = Path(str(name))
        if relative.is_absolute() or relative.name != str(name):
            raise ModelingError("recipe output must be one staging-local filename")

    artifact_meta = _read_regular_file(
        stage / artifact_meta_name,
        root=stage,
        expected_hash=None,
        maximum_bytes=_MAX_RECIPE_META_BYTES,
    )
    latest_meta = _read_regular_file(
        stage / "model_meta.json",
        root=stage,
        expected_hash=None,
        maximum_bytes=_MAX_RECIPE_META_BYTES,
    )
    if artifact_meta != latest_meta:
        raise ModelingError("recipe latest metadata does not match its artifact")

    model_hash = ""
    for name in names:
        source = stage / str(name)
        maximum_bytes = (
            _MAX_MODEL_BINARY_BYTES
            if str(name) == artifact.model_path
            else _MAX_RECIPE_META_BYTES
        )
        observed_hash = _require_file_content_hash(
            source,
            root=stage,
            expected_hash=None,
            maximum_bytes=maximum_bytes,
        )
        if str(name) == "model_meta.json":
            raw_staged = TransactionalArtifactStore(artifact_dir).stage(str(name))
            staged = _DeferredLatestStagedArtifact(staged=raw_staged)
            uow.track(staged)
        else:
            staged = uow.stage_file(artifact_dir, str(name))
        shutil.copyfile(source, staged.path)
        copied_hash = _require_file_content_hash(
            staged.path,
            root=artifact_dir,
            expected_hash=observed_hash,
            maximum_bytes=maximum_bytes,
        )
        if str(name) == artifact.model_path:
            model_hash = copied_hash
    if not model_hash:
        raise ModelingError("recipe model binary was not staged")
    return model_hash


def _record_housekeeping_warning_best_effort(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
    experiment_id: str,
    evidence_id: str | None,
    error: Exception,
) -> None:
    try:
        runtime.repo.write_audit(
            kind=TRAIN_MODEL_WITH_EVIDENCE_V2_HOUSEKEEPING_WARNING_AUDIT_KIND,
            target_ref=evidence_id or experiment_id,
            inputs_hash=_request_hash(request),
            outcome="warning",
            detail={
                "task_id": task_id,
                "experiment_id": experiment_id,
                "evidence_id": evidence_id,
                "warning": "artifact backup cleanup failed after database commit",
                "error_type": type(error).__name__,
                "error": str(error)[:500],
                "publication_committed": True,
            },
        )
    except Exception:
        # Warning persistence is deliberately best effort: the publication is
        # already committed and must still return its authenticated output.
        pass


def _training_frame_and_masks(
    runtime,
    *,
    sample: Any,
    request: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    source = sample.source_binding
    path = source.dataset_path
    frame = runtime.backend.read_frame(path)
    if not isinstance(frame, pd.DataFrame) or len(frame) != source.row_count:
        raise ModelingError("governed training dataset row count changed")
    if sha256_file(path) != source.dataset_content_hash:
        raise ModelingError("governed training dataset bytes changed")
    required = {
        *request["features"],
        sample.bundle["sample_design"]["target_selector"]["column"],
    }
    weight = request["params"].get("sample_weight_col")
    if weight is not None:
        required.add(str(weight))
    missing = sorted(required - {str(column) for column in frame.columns})
    if missing:
        raise ModelingError(
            "governed training dataset is missing columns: " + ", ".join(missing)
        )
    masks = sample.membership["masks"]
    membership = {
        "train": np.ascontiguousarray(masks["risk/development"], dtype=np.bool_),
        "test": np.ascontiguousarray(masks["risk/validation"], dtype=np.bool_),
        "oot": np.ascontiguousarray(masks["risk/oot"], dtype=np.bool_),
    }
    risk = np.logical_or.reduce(
        [membership["train"], membership["test"], membership["oot"]]
    )
    risk = np.ascontiguousarray(risk, dtype=np.bool_)
    return frame, membership, risk


def _build_internal_split_contract(frame: pd.DataFrame) -> dict[str, Any]:
    """Choose one deterministic source-column-safe private split contract."""

    source_columns = {str(column) for column in frame.columns}
    candidate = _INTERNAL_SPLIT_COLUMN_BASE
    attempt = 0
    while candidate in source_columns:
        attempt += 1
        candidate = f"__marvis_governed_split_v2_{attempt}__"
    body = {
        "source": _INTERNAL_SPLIT_SOURCE,
        "column": candidate,
        "values": dict(_INTERNAL_SPLIT_VALUES),
    }
    return body


def _materialize_private_governed_split(
    risk_frame: pd.DataFrame,
    *,
    membership_masks: Mapping[str, np.ndarray],
    risk_membership_mask: np.ndarray,
    internal_split: Mapping[str, Any],
) -> None:
    """Project authenticated full-frame membership into the private risk frame."""

    column = str(internal_split["column"])
    if column in {str(value) for value in risk_frame.columns}:
        raise ModelingError("internal governed split column collided before training")
    risk = np.asarray(risk_membership_mask, dtype=np.bool_)
    if int(np.count_nonzero(risk)) != len(risk_frame):
        raise ModelingError("private risk frame no longer matches governed membership")
    assigned = np.zeros(len(risk_frame), dtype=np.bool_)
    values = np.empty(len(risk_frame), dtype=object)
    for name in _SPLIT_NAMES:
        full_mask = np.asarray(membership_masks[name], dtype=np.bool_)
        if full_mask.shape != risk.shape:
            raise ModelingError("governed membership mask length changed")
        private_mask = np.ascontiguousarray(full_mask[risk], dtype=np.bool_)
        if np.any(np.logical_and(assigned, private_mask)):
            raise ModelingError("governed membership masks overlap")
        values[private_mask] = str(internal_split["values"][name])
        assigned = np.logical_or(assigned, private_mask)
    if not np.all(assigned):
        raise ModelingError("private governed split does not cover risk membership")
    risk_frame[column] = values


def _private_split_selector_masks(
    risk_frame: pd.DataFrame,
    *,
    risk_membership_mask: np.ndarray,
    internal_split: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Re-evaluate the actual recipe split column into full-source selectors."""

    risk = np.asarray(risk_membership_mask, dtype=np.bool_)
    if int(np.count_nonzero(risk)) != len(risk_frame):
        raise ModelingError("private risk frame no longer matches governed membership")
    column = str(internal_split["column"])
    if column not in risk_frame.columns:
        raise ModelingError("private governed split column is unavailable")
    series = risk_frame[column]
    selector: dict[str, np.ndarray] = {}
    for name in _SPLIT_NAMES:
        private_mask = (
            series.eq(str(internal_split["values"][name]))
            .fillna(False)
            .to_numpy(dtype=np.bool_)
        )
        full_mask = np.zeros(risk.shape, dtype=np.bool_)
        full_mask[risk] = private_mask
        selector[name] = np.ascontiguousarray(full_mask, dtype=np.bool_)
    return selector


def _training_config(
    runtime,
    *,
    sample: Any,
    request: Mapping[str, Any],
    internal_split: Mapping[str, Any],
) -> TrainConfig:
    design = sample.bundle["sample_design"]
    target = design["target_selector"]
    if target["status"] != "resolved":
        raise ModelingError(
            "governed training requires a resolved binary target"
        )
    _governed_target_contract(sample)
    target_col = target["column"]
    if target_col in request["features"]:
        raise ModelingError("target column must not leak into features")
    split_col = _text(internal_split["column"], "internal split column")
    split_values = _object(internal_split["values"], "internal split values")
    _exact_fields(split_values, frozenset(_SPLIT_NAMES), "internal split values")
    if not _INTERNAL_SPLIT_COLUMN_RE.fullmatch(split_col):
        raise ModelingError("internal governed split column is invalid")
    if split_values != _INTERNAL_SPLIT_VALUES:
        raise ModelingError("internal governed split values are invalid")
    params = dict(request["params"])
    weight_col = params.get("sample_weight_col")
    governed_weight_col = design["sample_semantics"]["field_bindings"][
        "weight_field"
    ]
    if weight_col is not None and weight_col != governed_weight_col:
        raise ModelingError(
            "params.sample_weight_col must equal the governed "
            "SampleDesign V2 weight_field"
        )
    if weight_col is not None and weight_col in {
        target_col,
        split_col,
        *request["features"],
    }:
        raise ModelingError(
            "sample weight column must not be target, split, or a feature"
        )
    steps = _preprocessing_steps_for_training(
        runtime,
        sample.source_binding.dataset_id,
    )
    if steps:
        params["preprocessing_steps"] = steps
    elif not _preprocessing_chain_traceable(
        runtime,
        sample.source_binding.dataset_id,
    ):
        params["preprocessing_chain_traceable"] = False
    return TrainConfig(
        dataset_id=sample.source_binding.dataset_id,
        features=tuple(request["features"]),
        target_col=target_col,
        split_col=split_col,
        split_values=dict(split_values),
        params=params,
        seed=request["seed"],
        early_stopping_rounds=request["early_stopping_rounds"],
        recipe_id=request["recipe"],
        scenario_id=None,
        target_type="binary",
        eval_metric="ks_auc",
        drop_nan_labels=target["drop_missing"],
    )


def _governed_target_contract(
    sample: Any,
) -> dict[str, Any]:
    target = sample.bundle["sample_design"]["target_selector"]
    return build_modeling_target_contract(
        column=target["column"],
        good_value=target["good_value"],
        bad_value=target["bad_value"],
        drop_missing=target["drop_missing"],
    )


def _encode_private_model_target(
    frame: pd.DataFrame,
    *,
    target_contract: Mapping[str, Any],
) -> None:
    """Encode only the private risk frame; the governed source remains raw."""

    column = str(target_contract["column"])
    raw = frame[column]
    missing = raw.isna().to_numpy(dtype=np.bool_)
    bool_values = raw.map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).to_numpy(dtype=np.bool_)
    numeric = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=float)
    invalid = (
        (~missing & bool_values)
        | (~missing & ~np.isfinite(numeric))
        | (
            ~missing
            & (numeric != float(target_contract["good_value"]))
            & (numeric != float(target_contract["bad_value"]))
        )
    )
    if np.any(invalid):
        raise ModelingError(
            "governed target contains non-numeric, non-finite, boolean, or "
            "out-of-contract values"
        )
    encoded = np.full(len(frame), np.nan, dtype=float)
    encoded[
        ~missing & (numeric == float(target_contract["good_value"]))
    ] = float(target_contract["encoded_good_value"])
    encoded[
        ~missing & (numeric == float(target_contract["bad_value"]))
    ] = float(target_contract["encoded_bad_value"])
    frame[column] = encoded


def _has_labeled_single_class_oot(
    frame: pd.DataFrame,
    *,
    config: TrainConfig,
) -> bool:
    oot_mask = (
        frame[config.split_col]
        .eq(config.split_values["oot"])
        .fillna(False)
        .to_numpy(dtype=np.bool_)
    )
    target = pd.to_numeric(
        frame.loc[oot_mask, config.target_col],
        errors="coerce",
    ).to_numpy(dtype=float)
    classes = np.unique(target[np.isfinite(target)])
    return classes.size == 1


def _require_binary_train_test_support(
    frame: pd.DataFrame,
    *,
    config: TrainConfig,
) -> None:
    for split_name in ("train", "test"):
        mask = (
            frame[config.split_col]
            .eq(config.split_values[split_name])
            .fillna(False)
            .to_numpy(dtype=np.bool_)
        )
        target = pd.to_numeric(
            frame.loc[mask, config.target_col],
            errors="coerce",
        ).to_numpy(dtype=float)
        classes = set(target[np.isfinite(target)].tolist())
        if classes != {0.0, 1.0}:
            raise ModelingError(
                f"governed {split_name} split requires both good and bad labels"
            )


def _without_single_class_oot_label_metrics(result):
    metrics = result.metrics
    normalized = replace(
        metrics,
        oot_ks=None,
        oot_auc=None,
        weighted_oot_ks=None,
        weighted_oot_auc=None,
        overfit_train_oot_gap=None,
        overfit_flag=metrics.overfit_train_test_gap > 0.10,
        oot_ks_ci_low=None,
        oot_ks_ci_high=None,
        oot_ks_ci_std=None,
        oot_lift_head_5=None,
        oot_lift_tail_5=None,
        oot_lift_head_10=None,
        oot_lift_tail_10=None,
    )
    return replace(result, metrics=normalized)


def _validate_governed_training_weights(
    frame: pd.DataFrame,
    *,
    membership_masks: Mapping[str, np.ndarray],
    config: TrainConfig,
) -> None:
    raw_weight_col = config.params.get("sample_weight_col")
    if raw_weight_col is None:
        return
    weight_col = str(raw_weight_col)
    for split_name in _SPLIT_NAMES:
        mask = np.asarray(membership_masks[split_name], dtype=np.bool_)
        values = frame.loc[mask, weight_col]
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        invalid = (
            values.isna().to_numpy(dtype=np.bool_)
            | ~np.isfinite(numeric)
            | (numeric <= 0)
        )
        if np.any(invalid):
            raise ModelingError(
                "governed sample weights must be non-null, finite, and strictly "
                f"positive in risk/{split_name}"
            )


def _validate_inputs(value: object) -> dict[str, Any]:
    obj = _object(value, "train_model_with_evidence_v2 inputs")
    _preflight_json(obj, "train_model_with_evidence_v2 inputs")
    _exact_fields(obj, _INPUT_FIELDS, "train_model_with_evidence_v2 inputs")
    recipe = _text(obj["recipe"], "recipe")
    if recipe not in BINARY_MODELING_RECIPES:
        raise ModelingError(
            "recipe must be one authoritative binary native-score recipe"
        )
    features = _text_array(obj["features"], "features", required=True)
    if len(features) != len(set(features)):
        raise ModelingError("features must not contain duplicates")
    params = _json_object(obj["params"], "params")
    _reject_caller_owned_platform_params(params)
    weight_aliases = [
        key
        for key in ("sample_weight_col", "sample_weight_column", "weight_col")
        if key in params
    ]
    if len(weight_aliases) > 1:
        raise ModelingError("params may define only sample_weight_col")
    if weight_aliases and weight_aliases[0] != "sample_weight_col":
        raise ModelingError("params must use canonical sample_weight_col")
    if "sample_weight_col" in params:
        params["sample_weight_col"] = _text(
            params["sample_weight_col"], "params.sample_weight_col"
        )
    seed = _non_negative_int(obj["seed"], "seed")
    if seed > 2**32 - 1:
        raise ModelingError("seed exceeds uint32")
    early = obj["early_stopping_rounds"]
    early_stopping = None if early is None else _positive_int(
        early, "early_stopping_rounds"
    )
    return {
        "sample_design_ref": _sample_ref(obj["sample_design_ref"]),
        "recipe": recipe,
        "features": features,
        "params": params,
        "seed": seed,
        "early_stopping_rounds": early_stopping,
    }


def _sample_ref(value: object) -> dict[str, str]:
    obj = _object(value, "sample_design_ref")
    _exact_fields(obj, _SAMPLE_REF_FIELDS, "sample_design_ref")
    return {
        "membership_artifact_id": _hash(
            obj["membership_artifact_id"], "membership_artifact_id"
        ),
        "expected_membership_artifact_content_hash": _hash(
            obj["expected_membership_artifact_content_hash"],
            "expected_membership_artifact_content_hash",
        ),
        "bundle_artifact_id": _hash(
            obj["bundle_artifact_id"], "bundle_artifact_id"
        ),
        "expected_bundle_artifact_content_hash": _hash(
            obj["expected_bundle_artifact_content_hash"],
            "expected_bundle_artifact_content_hash",
        ),
        "expected_bundle_id": _text(obj["expected_bundle_id"], "expected_bundle_id"),
        "expected_sample_design_id": _text(
            obj["expected_sample_design_id"], "expected_sample_design_id"
        ),
        "expected_sample_design_content_hash": _hash(
            obj["expected_sample_design_content_hash"],
            "expected_sample_design_content_hash",
        ),
    }


def _load_sample(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
    require_current: bool = True,
):
    ref = request["sample_design_ref"]
    try:
        loader = (
            load_any_strategy_sample_design_v2_artifacts
            if require_current
            else load_historical_any_strategy_sample_design_v2_artifacts
        )
        return loader(
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
    except StrategyError as exc:
        raise ModelingError(str(exc)) from exc


def _reject_caller_owned_platform_params(value: object, path: str = "params") -> None:
    calibration_fragments = ("calibrat", "isotonic", "platt")
    platform_owned = {
        "preprocessing_steps",
        "preprocessing_chain_traceable",
        "refit_on_train_plus_test",
        "split_col",
        "split_values",
    }
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            lowered = key.lower()
            if key in platform_owned:
                raise ModelingError(f"{path}.{key} is platform-owned")
            if any(fragment in lowered for fragment in calibration_fragments):
                raise ModelingError(f"{path}.{key} requests unsupported calibration")
            _reject_caller_owned_platform_params(child, f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_caller_owned_platform_params(child, f"{path}[{index}]")
        return
    if isinstance(value, str) and value.lower() in {
        "calibrated",
        "isotonic",
        "platt",
        "platt_scaling",
    }:
        raise ModelingError(f"{path} requests unsupported calibration")


def _live_training_snapshot_hashes(
    *,
    experiment: Experiment,
    model_artifact: ModelArtifact,
) -> dict[str, str]:
    """Canonical hashes of every live scoring/training value frozen in evidence."""

    if experiment.metrics is None:
        raise ModelingError("training-evidence experiment metrics are unavailable")
    scoring_metadata = modeling_scoring_metadata_from_artifact(model_artifact)
    return {
        "scoring_metadata_hash": _canonical_snapshot_hash(
            scoring_metadata,
            name="live model scoring metadata",
        ),
        "train_config_hash": _canonical_snapshot_hash(
            asdict(experiment.config),
            name="live TrainConfig",
        ),
        "metrics_snapshot_content_hash": _canonical_snapshot_hash(
            asdict(experiment.metrics),
            name="live model metrics",
        ),
    }


def _require_live_training_snapshot_binding(
    *,
    experiment: Experiment,
    model_artifact: ModelArtifact,
    evidence: Mapping[str, Any],
) -> dict[str, str]:
    try:
        live = _live_training_snapshot_hashes(
            experiment=experiment,
            model_artifact=model_artifact,
        )
    except ModelingTrainingEvidenceError as exc:
        raise ModelingError(
            "live scoring_metadata_hash drifted from immutable training "
            "evidence because the current scoring metadata is invalid"
        ) from exc
    frozen = {
        "scoring_metadata_hash": evidence["model_artifact"][
            "scoring_metadata_hash"
        ],
        "train_config_hash": evidence["training_contract"]["train_config_hash"],
        "metrics_snapshot_content_hash": evidence["metrics_snapshot"][
            "content_hash"
        ],
    }
    for name, live_hash in live.items():
        frozen_hash = str(frozen[name])
        if not hmac.compare_digest(live_hash, frozen_hash):
            raise ModelingError(
                f"live {name} drifted from immutable training evidence"
            )
    return live


def _canonical_snapshot_hash(value: object, *, name: str) -> str:
    try:
        payload = _canonical_json(value).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ModelingError(f"{name} is not canonical JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def _internal_split_provenance(config: TrainConfig) -> dict[str, Any]:
    """Authenticate the platform-owned split frozen in live TrainConfig."""

    column = _text(config.split_col, "live internal split column")
    values = dict(config.split_values)
    if not _INTERNAL_SPLIT_COLUMN_RE.fullmatch(column):
        raise ModelingError("live TrainConfig does not use a governed internal split")
    if values != _INTERNAL_SPLIT_VALUES:
        raise ModelingError("live TrainConfig governed split values changed")
    body = {
        "source": _INTERNAL_SPLIT_SOURCE,
        "column": column,
        "values": values,
    }
    return {
        "split_source": _INTERNAL_SPLIT_SOURCE,
        "internal_split_column": column,
        "internal_split_values": values,
        "internal_split_contract_hash": hashlib.sha256(
            _canonical_json(body).encode("utf-8")
        ).hexdigest(),
    }


def _model_provenance(
    *,
    task_id: str,
    sample: Any,
    experiment: Experiment,
    model_artifact: ModelArtifact,
    model_hash: str,
) -> dict[str, Any]:
    design = sample.bundle["sample_design"]
    membership = sample.membership["header"]
    snapshot_hashes = _live_training_snapshot_hashes(
        experiment=experiment,
        model_artifact=model_artifact,
    )
    target_encoding_hash = _governed_target_contract(sample)[
        "encoding_content_hash"
    ]
    return {
        "schema_version": TRAINING_EVIDENCE_ARTIFACT_SCHEMA_VERSION,
        "format": "binary",
        "artifact_role": "model_binary",
        "task_id": task_id,
        "experiment_id": experiment.id,
        "model_artifact_id": model_artifact.id,
        "algorithm": model_artifact.algorithm,
        "model_path": model_artifact.model_path,
        "model_binary_artifact_content_hash": model_hash,
        **snapshot_hashes,
        "target_encoding_content_hash": target_encoding_hash,
        "dataset_id": sample.source_binding.dataset_id,
        "dataset_content_hash": sample.source_binding.dataset_content_hash,
        "sample_design_id": design["sample_design_id"],
        "sample_design_content_hash": design["content_hash"],
        "sample_membership_id": membership["membership_id"],
        "sample_membership_content_hash": membership["content_hash"],
        "sample_membership_artifact_id": sample.membership_artifact_id,
        "sample_membership_artifact_content_hash": (
            sample.membership_artifact_content_hash
        ),
        "sample_bundle_artifact_id": sample.bundle_artifact_id,
        "sample_bundle_artifact_content_hash": sample.bundle_artifact_content_hash,
        **_internal_split_provenance(experiment.config),
    }


def _evidence_provenance(
    *,
    task_id: str,
    sample: Any,
    experiment: Experiment,
    model_artifact: ModelArtifact,
    model_record: Mapping[str, Any],
    evidence: Mapping[str, Any],
    evidence_file_hash: str,
) -> dict[str, Any]:
    design = sample.bundle["sample_design"]
    identity = design["identity"]
    membership = sample.membership["header"]
    workspace = identity["workspace_ref"]
    snapshot_hashes = _require_live_training_snapshot_binding(
        experiment=experiment,
        model_artifact=model_artifact,
        evidence=evidence,
    )
    target_encoding_hash = _governed_target_contract(sample)[
        "encoding_content_hash"
    ]
    if not hmac.compare_digest(
        target_encoding_hash,
        str(
            evidence["training_contract"]["target"][
                "encoding_content_hash"
            ]
        ),
    ):
        raise ModelingError(
            "live target encoding drifted from immutable training evidence"
        )
    return {
        "schema_version": TRAINING_EVIDENCE_ARTIFACT_SCHEMA_VERSION,
        "format": "json",
        "artifact_role": "training_evidence",
        "task_id": task_id,
        "experiment_id": experiment.id,
        "model_artifact_id": model_artifact.id,
        "model_binary_artifact_id": str(model_record["id"]),
        "model_binary_artifact_content_hash": str(model_record["content_hash"]),
        **snapshot_hashes,
        "evidence_id": evidence["evidence_id"],
        "evidence_content_hash": evidence["content_hash"],
        "evidence_artifact_content_hash": evidence_file_hash,
        "target_encoding_content_hash": target_encoding_hash,
        "dataset_id": identity["dataset_ref"]["dataset_id"],
        "dataset_content_hash": identity["dataset_ref"]["content_hash"],
        "workspace_revision": workspace["revision"],
        "workspace_generation": workspace["generation"],
        "semantic_mapping_hash": workspace["semantic_mapping_hash"],
        "sample_design_id": design["sample_design_id"],
        "sample_design_content_hash": design["content_hash"],
        "sample_membership_id": membership["membership_id"],
        "sample_membership_content_hash": membership["content_hash"],
        "sample_membership_artifact_id": sample.membership_artifact_id,
        "sample_membership_artifact_content_hash": (
            sample.membership_artifact_content_hash
        ),
        "sample_bundle_artifact_id": sample.bundle_artifact_id,
        "sample_bundle_artifact_content_hash": sample.bundle_artifact_content_hash,
        **_internal_split_provenance(experiment.config),
        "split_proof_content_hash": evidence["training_contract"][
            "split_proof"
        ]["content_hash"],
        "governance": _governance_flags(),
    }


def _tool_output(
    *,
    task_id: str,
    sample_ref: Mapping[str, Any],
    experiment: Experiment,
    model_artifact: ModelArtifact,
    evidence: Mapping[str, Any],
    model_record: Mapping[str, Any],
    evidence_record: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": TRAIN_MODEL_WITH_EVIDENCE_V2_TOOL_SCHEMA_VERSION,
        "experiment_id": experiment.id,
        "model_artifact_id": model_artifact.id,
        "evidence_id": evidence["evidence_id"],
        "evidence_content_hash": evidence["content_hash"],
        "score_product": RAW_SCORE_PRODUCT,
        "sample_design_ref": dict(sample_ref),
        "artifacts": {
            "model_binary": _artifact_output(task_id=task_id, record=model_record),
            "training_evidence": _artifact_output(
                task_id=task_id,
                record=evidence_record,
            ),
        },
        "governance": _governance_flags(),
    }


def _artifact_output(*, task_id: str, record: Mapping[str, Any]) -> dict[str, str]:
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
        raise ModelingError("training-evidence TaskArtifact was not found in task")
    if record["kind"] != kind:
        raise ModelingError("training-evidence TaskArtifact kind changed")
    if not hmac.compare_digest(record["content_hash"], expected_content_hash):
        raise ModelingError("training-evidence TaskArtifact hash changed")
    if record["origin_tool"] != TRAIN_MODEL_WITH_EVIDENCE_V2_ORIGIN_TOOL:
        raise ModelingError("training-evidence TaskArtifact origin changed")
    return record


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
        raise ModelingError(f"{name} disappeared before write")
    expected_scalars = {
        "id": str(record["id"]),
        "task_id": task_id,
        "kind": str(record["kind"]),
        "path": str(record["path"]),
        "content_hash": str(record["content_hash"]),
        "origin_tool": str(record["origin_tool"]),
        "created_at": str(record["created_at"]),
    }
    if any(str(row[key]) != value for key, value in expected_scalars.items()):
        raise ModelingError(f"{name} row changed before write")
    provenance = _load_database_json(
        row["provenance_json"],
        name=f"{name} provenance",
    )
    if provenance != record["provenance"]:
        raise ModelingError(f"{name} provenance changed before write")


def _require_experiment_row_on_connection(
    conn,
    *,
    binding: ModelingTrainingEvidenceArtifactBinding,
) -> None:
    row = conn.execute(
        """
        SELECT id, task_id, recipe_id, config_json, metrics_json,
               artifact_id, status, created_at
          FROM experiments
         WHERE id = ?
        """,
        (binding.experiment.id,),
    ).fetchone()
    if row is None:
        raise ModelingError("training experiment disappeared before write")
    expected_scalars = {
        "id": binding.experiment.id,
        "task_id": binding.task_id,
        "recipe_id": binding.experiment.recipe_id,
        "artifact_id": binding.model_artifact.id,
        "status": "trained",
        "created_at": binding.experiment.created_at,
    }
    if any(str(row[key]) != value for key, value in expected_scalars.items()):
        raise ModelingError("training experiment row changed before write")
    expected_config = json.loads(
        _canonical_json(asdict(binding.experiment.config))
    )
    expected_metrics = json.loads(
        _canonical_json(asdict(binding.experiment.metrics))
    )
    if _load_database_json(
        row["config_json"],
        name="training experiment config",
    ) != expected_config:
        raise ModelingError("training experiment config drifted before write")
    if _load_database_json(
        row["metrics_json"],
        name="training experiment metrics",
    ) != expected_metrics:
        raise ModelingError("training experiment metrics drifted before write")


def _require_model_artifact_row_on_connection(
    conn,
    *,
    binding: ModelingTrainingEvidenceArtifactBinding,
) -> None:
    row = conn.execute(
        """
        SELECT id, experiment_id, algorithm, model_path, pmml_path,
               feature_list_json, feature_importance_json, params_json,
               woe_maps_json, scorecard_table_json, created_at,
               score_direction, points_direction,
               baseline_distributions_json
          FROM model_artifacts
         WHERE id = ?
        """,
        (binding.model_artifact.id,),
    ).fetchone()
    if row is None:
        raise ModelingError("model artifact row disappeared before write")
    artifact = binding.model_artifact
    expected_scalars = {
        "id": artifact.id,
        "experiment_id": binding.experiment.id,
        "algorithm": artifact.algorithm,
        "model_path": artifact.model_path,
        "pmml_path": artifact.pmml_path,
        "created_at": artifact.created_at,
        "score_direction": artifact.score_direction,
        "points_direction": artifact.points_direction,
    }
    for key, expected in expected_scalars.items():
        observed = None if row[key] is None else str(row[key])
        if observed != expected:
            raise ModelingError("model artifact row changed before write")
    expected_json = {
        "feature_list_json": list(artifact.feature_list),
        "feature_importance_json": [
            [feature, importance]
            for feature, importance in artifact.feature_importance
        ],
        "params_json": artifact.params,
        "woe_maps_json": artifact.woe_maps,
        "scorecard_table_json": list(artifact.scorecard_table),
        "baseline_distributions_json": artifact.baseline_distributions,
    }
    for column, expected in expected_json.items():
        observed = (
            None
            if row[column] is None
            else _load_database_json(
                row[column],
                name=f"model artifact {column}",
            )
        )
        if observed != expected:
            raise ModelingError("model artifact metadata drifted before write")


def _load_database_json(value: object, *, name: str) -> Any:
    if not isinstance(value, str) or len(value.encode("utf-8")) > _MAX_RECIPE_META_BYTES:
        raise ModelingError(f"{name} is invalid")

    def without_duplicates(pairs):
        result: dict[str, Any] = {}
        for key, child in pairs:
            if key in result:
                raise ModelingError(f"{name} contains duplicate keys")
            result[key] = child
        return result

    try:
        return json.loads(value, object_pairs_hook=without_duplicates)
    except ModelingError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ModelingError(f"{name} is invalid") from exc


def _require_staged_recipe_model_binary(
    training_stage_dir: Path,
    *,
    artifact: ModelArtifact,
) -> tuple[Path, str]:
    relative = Path(artifact.model_path)
    if relative.is_absolute() or relative.name != artifact.model_path:
        raise ModelingError("trained model path must be one staging-local filename")
    path = Path(training_stage_dir) / relative
    observed_hash = _require_file_content_hash(
        path,
        root=Path(training_stage_dir),
        expected_hash=None,
        maximum_bytes=_MAX_MODEL_BINARY_BYTES,
    )
    return path, observed_hash


def _require_model_binary(
    tasks_root: Path,
    *,
    task_id: str,
    artifact_dir: Path,
    artifact: ModelArtifact,
) -> tuple[Path, str]:
    _require_artifact_directory_boundary(
        tasks_root,
        task_id=task_id,
        artifact_dir=artifact_dir,
        allow_missing=False,
    )
    relative = Path(artifact.model_path)
    if relative.is_absolute() or relative.name != artifact.model_path:
        raise ModelingError("trained model path must be one task-local filename")
    path = artifact_dir / relative
    observed_hash = _require_file_content_hash(
        path,
        root=Path(tasks_root),
        expected_hash=None,
        maximum_bytes=_MAX_MODEL_BINARY_BYTES,
    )
    return path, observed_hash


def _require_artifact_directory_boundary(
    tasks_root: Path,
    *,
    task_id: str,
    artifact_dir: Path,
    allow_missing: bool,
) -> None:
    root = Path(tasks_root)
    task_dir = root / task_id
    expected = task_dir / "modeling_artifacts"
    if artifact_dir != expected:
        raise ModelingError("modeling artifact directory is not canonical")
    for path, label in ((root, "tasks root"), (task_dir, "task directory")):
        if path.is_symlink() or not path.is_dir():
            raise ModelingError(f"{label} is unavailable or unsafe")
    if artifact_dir.exists() or artifact_dir.is_symlink():
        if artifact_dir.is_symlink() or not artifact_dir.is_dir():
            raise ModelingError("modeling artifact directory is unsafe")
    elif not allow_missing:
        raise ModelingError("modeling artifact directory is missing")
    try:
        artifact_dir.resolve(strict=False).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ModelingError("modeling artifact directory escaped tasks root") from exc


def _read_regular_file(
    path: Path,
    *,
    root: Path,
    expected_hash: str | None,
    maximum_bytes: int,
) -> bytes:
    before = _regular_file_stat(path, root=root, maximum_bytes=maximum_bytes)
    observed = sha256_file(path)
    if expected_hash is not None and not hmac.compare_digest(observed, expected_hash):
        raise ModelingError("registered artifact bytes changed")
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise ModelingError("registered artifact could not be read") from exc
    after = _regular_file_stat(path, root=root, maximum_bytes=maximum_bytes)
    if before != after or len(raw) != before[1]:
        raise ModelingError("registered artifact changed while reading")
    if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), observed):
        raise ModelingError("registered artifact changed while reading")
    return raw


def _require_file_content_hash(
    path: Path,
    *,
    root: Path,
    expected_hash: str | None,
    maximum_bytes: int,
) -> str:
    before = _regular_file_stat(path, root=root, maximum_bytes=maximum_bytes)
    try:
        observed = sha256_file(Path(path))
    except OSError as exc:
        raise ModelingError("registered artifact could not be hashed") from exc
    after = _regular_file_stat(path, root=root, maximum_bytes=maximum_bytes)
    if before != after:
        raise ModelingError("registered artifact changed while hashing")
    if expected_hash is not None and not hmac.compare_digest(observed, expected_hash):
        raise ModelingError("registered artifact bytes changed")
    return observed


def _regular_file_stat(
    path: Path,
    *,
    root: Path,
    maximum_bytes: int,
) -> tuple[int, int, int, int]:
    path = Path(path)
    if path.is_symlink():
        raise ModelingError("registered artifact must not be a symlink")
    try:
        resolved_root = Path(root).resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(resolved_root)
        observed = os.lstat(path)
    except (OSError, ValueError) as exc:
        raise ModelingError("registered artifact path is unavailable or escaped") from exc
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


def _require_registered_path(record: Mapping[str, Any], expected: Path) -> None:
    if Path(str(record["path"])) != expected:
        raise ModelingError("training-evidence TaskArtifact path changed")


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


def _governance_flags() -> dict[str, bool]:
    return {
        "not_selected": True,
        "not_calibrated": True,
        "not_adopted": True,
        "not_deployed": True,
    }


def _governance(value: object) -> dict[str, bool]:
    obj = _object(value, "governance")
    _exact_fields(obj, _GOVERNANCE_FIELDS, "governance")
    if obj != _governance_flags():
        raise ModelingError("training-evidence governance flags must remain false-state")
    return dict(obj)


def _request_hash(request: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(request).encode("utf-8")).hexdigest()


def _preflight_json(value: object, name: str) -> None:
    try:
        raw = _canonical_json(value).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ModelingError(f"{name} must contain finite JSON values") from exc
    if len(raw) > _MAX_INPUT_JSON_BYTES:
        raise ModelingError(f"{name} exceeds byte budget")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_object(value: object, name: str) -> dict[str, Any]:
    obj = _object(value, name)
    try:
        raw = _canonical_json(obj)
        normalized = json.loads(raw)
    except (TypeError, ValueError, OverflowError, json.JSONDecodeError) as exc:
        raise ModelingError(f"{name} must contain finite JSON values") from exc
    if not isinstance(normalized, dict):
        raise ModelingError(f"{name} must be an object")
    return normalized


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelingError(f"{name} must be an object")
    return dict(value)


def _exact_fields(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected)
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unknown:
            details.append("unknown=" + ",".join(unknown))
        raise ModelingError(f"{name} fields are invalid ({'; '.join(details)})")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ModelingError(f"{name} must be a non-empty string")
    return value.strip()


def _text_array(value: object, name: str, *, required: bool) -> list[str]:
    if not isinstance(value, list):
        raise ModelingError(f"{name} must be an array")
    result = [_text(item, f"{name}[{index}]") for index, item in enumerate(value)]
    if required and not result:
        raise ModelingError(f"{name} must not be empty")
    return result


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ModelingError(f"{name} must be a lowercase SHA-256 hash")
    return value


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelingError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: object, name: str) -> int:
    result = _non_negative_int(value, name)
    if result == 0:
        raise ModelingError(f"{name} must be positive")
    return result


__all__ = [
    "TRAINING_EVIDENCE_ARTIFACT_SCHEMA_VERSION",
    "TRAIN_MODEL_WITH_EVIDENCE_V2_AUDIT_KIND",
    "TRAIN_MODEL_WITH_EVIDENCE_V2_ORIGIN_TOOL",
    "TRAIN_MODEL_WITH_EVIDENCE_V2_TOOL_SCHEMA_VERSION",
    "ModelingTrainingEvidenceArtifactBinding",
    "build_training_evidence_ref",
    "load_historical_modeling_training_evidence_artifacts",
    "load_modeling_training_evidence_artifacts",
    "require_historical_modeling_training_evidence_artifact_binding_on_connection",
    "require_modeling_training_evidence_artifact_binding_on_connection",
    "run_train_model_with_evidence_v2",
    "tool_train_model_with_evidence_v2",
    "validate_train_model_with_evidence_v2_tool_output",
]
