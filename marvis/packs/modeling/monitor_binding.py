"""Read-only identity binding for user-confirmed model monitoring runs.

This freezes the selected model, baseline, and new-period sample. It does not
establish production approval or historical point-in-time availability.
"""
from dataclasses import asdict
from contextlib import contextmanager
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

from marvis.data.backend import DataBackend
from marvis.data.dataset_identity import dataset_identity_equal
from marvis.data.errors import DataLayerError
from marvis.data.registry import DatasetRegistry
from marvis.packs.modeling._runtime import _artifact_base_dir
from marvis.packs.modeling.errors import ModelingError
from marvis.packs.modeling.producer_receipts import canonical, load_receipt
from marvis.repositories.audit import _list_audit_rows
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.datasets import DatasetRepository
from marvis.repositories.modeling import ModelingRepository


def monitoring_binding_hash(value: dict) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _registry(settings):
    return DatasetRegistry(
        DatasetRepository(settings.db_path), DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )


def capture_monitoring_binding(
    settings, task_id: str, experiment_id: str, dataset_id: str,
    expected_content_hash: str,
) -> dict:
    """Capture selected native-model and sample identities without changing them."""
    try:
        repo = ModelingRepository(settings.db_path)
        experiment = repo.get_experiment(experiment_id)
        if (
            experiment is None or experiment.task_id != task_id
            or experiment.status != "selected" or not experiment.artifact_id
        ):
            raise ModelingError("monitoring_selected_task_experiment_required")
        artifact = repo.get_model_artifact(experiment.artifact_id)
        if artifact is None or artifact.experiment_id != experiment.id:
            raise ModelingError("monitoring_model_attachment_changed")
        if not artifact.baseline_distributions:
            raise ModelingError("monitoring_baseline_required")
        if experiment.config.target_type != "binary":
            raise ModelingError("monitoring_binary_baseline_required")
        receipt, snapshots = load_receipt(
            settings.db_path, experiment, artifact,
            _artifact_base_dir(settings, task_id),
        )
        # load_receipt authenticates every dependency, including ensemble members
        # and calibration files. Never deserialize a model to establish identity.
        del snapshots
        registry = _registry(settings)
        dataset = registry.get(dataset_id)
        if dataset.task_id != task_id or dataset.content_hash != expected_content_hash:
            raise ModelingError("monitoring_source_identity_changed")
        registry.authenticated_parquet_column_names(dataset.id)
        if not dataset_identity_equal(dataset, registry.get(dataset.id)):
            raise ModelingError("monitoring_source_identity_changed")
        if monitoring_binding_hash(asdict(experiment)) != monitoring_binding_hash(
            asdict(repo.get_experiment(experiment.id))
        ) or (
            monitoring_binding_hash(asdict(artifact))
            != monitoring_binding_hash(asdict(repo.get_model_artifact(artifact.id)))
        ):
            raise ModelingError("monitoring_model_identity_changed")
        return {
            "schema_version": "model-monitoring-binding.v1",
            "task_id": task_id,
            "experiment_id": experiment.id,
            "experiment_status": experiment.status,
            "experiment_sha256": monitoring_binding_hash(asdict(experiment)),
            "artifact_id": artifact.id,
            "artifact_sha256": monitoring_binding_hash(asdict(artifact)),
            "baseline_sha256": monitoring_binding_hash(artifact.baseline_distributions),
            "producer_receipt_id": receipt["id"],
            "files": receipt["provenance"]["files"],
            "dataset": {
                "id": dataset.id, "task_id": dataset.task_id,
                "content_hash": dataset.content_hash,
                "identity_sha256": monitoring_binding_hash(asdict(dataset)),
            },
        }
    except ModelingError:
        raise
    except (KeyError, ValueError, TypeError, OSError, DataLayerError) as exc:
        raise ModelingError("monitoring_binding_authentication_failed") from exc


def validate_monitoring_binding(
    settings, task_id: str, binding: dict, *, scored_dataset_id: str | None = None,
    score_col: str | None = None,
) -> None:
    """Reject changed input/model/workspace or an unrelated scored child."""
    try:
        if not isinstance(binding, dict):
            raise ModelingError("monitoring_binding_invalid")
        expected = {k: v for k, v in binding.items() if k != "workspace_binding"}
        source = expected["dataset"]
        current = capture_monitoring_binding(
            settings, task_id, expected["experiment_id"], source["id"],
            source["content_hash"],
        )
        if canonical(current) != canonical(expected):
            raise ModelingError("monitoring_binding_changed")
        if "workspace_binding" in binding:
            workspace = DataWorkspaceRepository(settings.db_path).get_or_default(task_id)
            actual_workspace = {
                name: getattr(workspace, name) for name in (
                    "revision", "analysis_generation", "active_dataset_id",
                    "active_dataset_content_hash",
                )
            }
            if canonical(actual_workspace) != canonical(binding["workspace_binding"]) or (
                workspace.active_dataset_id != source["id"]
                or workspace.active_dataset_content_hash != source["content_hash"]
            ):
                raise ModelingError("monitoring_workspace_changed")
        if scored_dataset_id is not None:
            registry = _registry(settings)
            scored = registry.get(scored_dataset_id)
            if scored.task_id != task_id or scored.role != "modeling.scored":
                raise ModelingError("monitoring_scored_source_mismatch")
            receipts = _list_audit_rows(
                settings.db_path, kind="modeling.dataset.scored",
                target_ref=scored.id,
            )
            matching = [r for r in receipts if r["outcome"] == "succeeded" and all(
                r["detail"].get(key) == value for key, value in {
                    "source_dataset_id": source["id"],
                    "source_content_hash": source["content_hash"],
                    "scored_content_hash": scored.content_hash,
                    "experiment_id": expected["experiment_id"],
                    "artifact_id": expected["artifact_id"],
                    "monitoring_binding_hash": monitoring_binding_hash(binding),
                }.items()
            )]
            if len(matching) != 1:
                raise ModelingError("monitoring_scored_source_mismatch")
            if score_col is not None and matching[0]["detail"].get("score_col") != score_col:
                raise ModelingError("monitoring_scored_column_mismatch")
            registry.authenticated_parquet_column_names(scored.id)
    except ModelingError:
        raise
    except (KeyError, ValueError, TypeError, OSError, DataLayerError) as exc:
        raise ModelingError("monitoring_binding_invalid") from exc


@contextmanager
def monitoring_model_directory(settings, task_id: str, binding: dict | None, base_dir):
    """Deserialize only the authenticated bytes retained for this bound run.

    Hashing before and after a read alone permits transient file replacement.
    Ensemble members load lazily while scoring, so retain the whole private
    dependency directory until all predictions and points have been produced.
    """
    if binding is None:
        yield base_dir
        return
    validate_monitoring_binding(settings, task_id, binding)
    repo = ModelingRepository(settings.db_path)
    experiment = repo.get_experiment(binding["experiment_id"])
    artifact = repo.get_model_artifact(binding["artifact_id"])
    if experiment is None or artifact is None:
        raise ModelingError("monitoring_model_identity_changed")
    receipt, snapshots = load_receipt(settings.db_path, experiment, artifact, base_dir)
    if (
        monitoring_binding_hash(asdict(artifact)) != binding["artifact_sha256"]
        or canonical(receipt["provenance"]["files"]) != canonical(binding["files"])
    ):
        raise ModelingError("monitoring_model_identity_changed")
    with TemporaryDirectory(prefix="marvis-monitor-model-") as directory:
        root = Path(directory)
        for name, data in snapshots.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        yield root
        validate_monitoring_binding(settings, task_id, binding)
