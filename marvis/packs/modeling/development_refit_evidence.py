"""Authenticate a new fit scope without changing its parent's fold receipt."""

from dataclasses import asdict
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from marvis.data.backend import DataBackend
from marvis.data.authenticated_snapshot import authenticated_file_snapshot
from marvis.data.labels import resolve_labeled_frame
from marvis.data.preprocessing_evidence import load_preprocessing_state
from marvis.data.registry import DatasetRegistry
from marvis.packs.modeling.errors import ModelingError
from marvis.repositories.datasets import DatasetRepository
from marvis.repositories.modeling import ModelingRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository

KIND = "modeling_development_refit"
ORIGIN = "modeling.native_development_refit.v1"
MAX_BYTES = 32 * 1024**2


def digest(value):
    from marvis.packs.modeling.producer_receipts import canonical

    return hashlib.sha256(canonical(value).encode()).hexdigest()


def receipt_id(artifact):
    return (artifact.params.get("fitted_input_time_evidence") or {}).get(
        "development_refit_receipt_id"
    )


def members(mask):
    raw = np.packbits(mask, bitorder="little").tobytes()
    return {
        "rows": int(mask.sum()),
        "membership": base64.b64encode(raw).decode(),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _mask(value, count):
    try:
        raw = base64.b64decode(value["membership"], validate=True)
        mask = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")
        if (
            len(raw) != (count + 7) // 8
            or mask[count:].any()
            or hashlib.sha256(raw).hexdigest() != value["sha256"]
            or int(mask[:count].sum()) != value["rows"]
        ):
            raise ValueError("invalid membership")
        return mask[:count].astype(bool)
    except (ValueError, KeyError, TypeError) as exc:
        raise ModelingError("development_refit_membership_invalid") from exc


def parent_binding(experiment, artifact, native_record):
    from marvis.packs.modeling.producer_receipts import metadata_hash

    return {
        "experiment_id": experiment.id,
        "artifact_id": artifact.id,
        "config_sha256": digest(asdict(experiment.config)),
        "metrics_sha256": digest(asdict(experiment.metrics)),
        "model_metadata_sha256": metadata_hash(artifact),
        "native_receipt_id": native_record["id"],
        "native_receipt_hash": native_record["content_hash"],
    }


def _authenticate_memberships(
    registry, source, config, artifact, train, test, development
):
    import pyarrow.parquet as pq
    from marvis.packs.modeling.fold_selection import _ProjectedBackend

    if source.row_count > 1_000_000:
        raise ModelingError("development_refit_membership_verification_limit")
    path = registry.resolve_verified_path(source.id)
    with authenticated_file_snapshot(
        path, root=registry.datasets_root, expected_sha256=source.content_hash
    ) as snapshot:
        parquet = pq.ParquetFile(snapshot)
        if parquet.metadata.num_rows != source.row_count:
            raise ModelingError("development_refit_membership_source_rows_changed")
        controls = _ProjectedBackend(
            parquet, np.arange(source.row_count), [config.split_col], {}
        ).read_frame(None)
        original_train = (
            controls[config.split_col]
            .eq(config.split_values["train"])
            .fillna(False)
            .to_numpy(dtype=bool)
        )
        original_test = (
            controls[config.split_col]
            .eq(config.split_values["test"])
            .fillna(False)
            .to_numpy(dtype=bool)
        )
        positions = np.flatnonzero(original_train | original_test)
        if not len(positions):
            raise ModelingError("development_refit_membership_changed")
        labels = _ProjectedBackend(
            parquet, positions, [config.target_col], {}
        ).read_frame(None)
        labels, _ = resolve_labeled_frame(
            labels, config.target_col, drop_nan_labels=config.drop_nan_labels
        )
    eligible = np.zeros(source.row_count, dtype=bool)
    eligible[labels.index] = True
    if not (
        np.array_equal(original_train & eligible, train)
        and np.array_equal(original_test & eligible, test)
        and np.array_equal(eligible, development)
    ):
        raise ModelingError("development_refit_source_membership_changed")
    temporal = artifact.params.get("fitted_input_time_evidence") or {}
    fitted = members(development)
    empty = members(np.zeros(source.row_count, dtype=bool))
    if (
        temporal != config.params.get("fitted_input_time_evidence")
        or temporal.get("dataset_id") != source.id
        or temporal.get("dataset_content_hash") != source.content_hash
        or temporal.get("row_count") != source.row_count
        or temporal.get("fit_rows") != fitted["rows"]
        or temporal.get("fit_membership") != fitted["membership"]
        or temporal.get("fit_membership_sha256") != fitted["sha256"]
        or temporal.get("evaluation_rows") != 0
        or temporal.get("evaluation_membership") != empty["membership"]
        or temporal.get("evaluation_membership_sha256") != empty["sha256"]
        or temporal.get("evaluation_roles") != []
        or temporal.get("assurance") != "unknown"
        or temporal.get("label_input_assurance") != "unknown"
    ):
        raise ModelingError("development_refit_fit_scope_evidence_changed")


def validate_development_refit(db_path, experiment, artifact, directory):
    """Verify every refit claim before native delivery, monitoring or freeze."""
    identity = receipt_id(artifact)
    configured = (experiment.config.params.get("fitted_input_time_evidence") or {}).get(
        "development_refit_receipt_id"
    )
    if identity is None and configured is None:
        return None
    if identity != configured or not isinstance(identity, str):
        raise ModelingError("development_refit_reference_mismatch")
    repo = TaskArtifactRepository(db_path)
    record = repo.get_for_task(experiment.task_id, identity)
    if (
        not record
        or record["kind"] != KIND
        or record["origin_tool"] != ORIGIN
        or record["provenance"].get("schema_version") != ORIGIN
    ):
        raise ModelingError("development_refit_receipt_unknown")
    prefix = f"tasks/{experiment.task_id}/modeling_artifacts/"
    if not record["path"].startswith(prefix):
        raise ModelingError("development_refit_receipt_path_invalid")
    name = record["path"][len(prefix) :]
    path = Path(directory) / name
    if (
        Path(name).name != name
        or path.resolve(strict=True) != path.absolute()
        or path.stat().st_size > MAX_BYTES
    ):
        raise ModelingError("development_refit_receipt_path_invalid")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["content_hash"]:
        raise ModelingError("development_refit_receipt_changed")
    payload = json.loads(raw)
    from marvis.packs.modeling.producer_receipts import load_receipt, metadata_hash

    if (
        payload.get("schema_version") != ORIGIN
        or payload["experiment_id"] != experiment.id
        or payload["artifact_id"] != artifact.id
        or payload["model_metadata_sha256"] != metadata_hash(artifact)
        or payload["config_sha256"] != digest(asdict(experiment.config))
        or payload["metrics_sha256"] != digest(asdict(experiment.metrics))
        or payload.get("baseline_sha256") != digest(artifact.baseline_distributions)
        or payload.get("baseline_scope") != "development_fit"
    ):
        raise ModelingError("development_refit_model_binding_changed")
    modeling = ModelingRepository(db_path)
    parent = modeling.get_experiment(payload["parent"]["experiment_id"])
    parent_artifact = modeling.get_model_artifact(payload["parent"]["artifact_id"])
    if (
        parent is None
        or parent_artifact is None
        or parent.task_id != experiment.task_id
        or parent.artifact_id != parent_artifact.id
        or parent_artifact.experiment_id != parent.id
        or receipt_id(parent_artifact) is not None
    ):
        raise ModelingError("development_refit_parent_binding_changed")
    parent_record, _ = load_receipt(db_path, parent, parent_artifact, directory)
    if parent_binding(parent, parent_artifact, parent_record) != payload["parent"]:
        raise ModelingError("development_refit_parent_evidence_changed")
    root = Path(db_path).parent / "datasets"
    registry = DatasetRegistry(DatasetRepository(db_path), DataBackend(root), root)
    from marvis.packs.modeling.fold_tuning_evidence import validate_fold_experiment

    fold = validate_fold_experiment(
        registry, experiment.task_id, parent, parent_artifact
    )
    if (
        fold is None
        or parent.config.params.get("fold_tuning_evidence_ref")
        != payload["parent_fold_ref"]
    ):
        raise ModelingError("development_refit_parent_fold_changed")
    source = registry.get(experiment.config.dataset_id)
    if (
        source.task_id != experiment.task_id
        or {
            "dataset_id": source.id,
            "content_hash": source.content_hash,
            "row_count": source.row_count,
        }
        != payload["source"]
    ):
        raise ModelingError("development_refit_source_changed")
    registry.resolve_verified_path(source.id)
    state = load_preprocessing_state(registry, source.id)
    if {
        "artifact_id": state.artifact_id,
        "content_hash": state.content_hash,
        "steps": state.steps,
        "assurance": state.assurance,
    } != payload["preprocessing"]:
        raise ModelingError("development_refit_preprocessing_changed")
    count = source.row_count
    bindings = payload["memberships"]
    train, test, development, excluded = (
        _mask(bindings[key], count)
        for key in ("original_train", "original_test", "development_fit", "excluded")
    )
    if (
        not train.any()
        or not test.any()
        or (train & test).any()
        or not np.array_equal(development, train | test)
        or not np.array_equal(excluded, ~development)
    ):
        raise ModelingError("development_refit_membership_changed")
    baseline = artifact.baseline_distributions or {}
    if (baseline.get("baseline_scope") != "development_fit"
            or baseline.get("source_dataset_id") != source.id
            or baseline.get("source_dataset_content_hash") != source.content_hash
            or baseline.get("fit_membership_sha256") != members(development)["sha256"]
            or set(baseline.get("score_distribution", {})) != {"train"}
            or baseline["score_distribution"]["train"].get("sample_count") != int(development.sum())
            or baseline["score_distribution"]["train"].get("population_scope") != "development_fit"
            or baseline.get("oot_evaluation") != "not_performed"
            or baseline.get("label_maturity_assurance") != "unknown"):
        raise ModelingError("development_refit_baseline_scope_changed")
    _authenticate_memberships(
        registry, source, experiment.config, artifact, train, test, development
    )
    if (
        list(artifact.feature_list) != payload["features"]
        or tuple(experiment.config.features) != parent.config.features
        or artifact.algorithm != "lr"
        or experiment.recipe_id != "lr"
        or experiment.config.early_stopping_rounds is not None
        or experiment.config.params.get("fold_tuning_evidence_ref") is not None
    ):
        raise ModelingError("development_refit_profile_changed")
    from marvis.packs.modeling.recipes.common import model_params, sample_weight_col

    if (
        model_params(artifact.params) != payload["learner_params"]
        or payload["learner_params"] != model_params(parent_artifact.params)
        or model_params(experiment.config.params) != model_params(parent.config.params)
        or experiment.config.seed != parent.config.seed
        or experiment.config.target_col != parent.config.target_col
        or experiment.config.split_col != parent.config.split_col
        or experiment.config.split_values != parent.config.split_values
        or experiment.config.drop_nan_labels != parent.config.drop_nan_labels
        or sample_weight_col(experiment.config) != sample_weight_col(parent.config)
        or artifact.params.get("refit_on_train_plus_test") is not True
    ):
        raise ModelingError("development_refit_frozen_parameters_changed")
    final_metrics = asdict(experiment.metrics)
    if any(
        value is not None
        for key, value in final_metrics.items()
        if key.startswith(
            ("test_", "oot_", "weighted_test_", "weighted_oot_", "psi_", "overfit_")
        )
    ):
        raise ModelingError("development_refit_independent_evaluation_not_established")
    return payload
