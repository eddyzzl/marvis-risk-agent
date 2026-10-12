"""Project registered native model bytes and an explicit ordered input schema."""
import base64

from marvis.packs.modeling.producer_receipts import load_receipt, metadata_hash
from marvis.reference_decision.contracts import digest
from marvis.repositories.modeling import ModelingRepository

from .contracts import AdapterError


def native_upload(settings, task_id, request):
    repo = ModelingRepository(settings.db_path)
    artifact = repo.get_model_artifact(request.model_artifact_id)
    experiment = None if artifact is None else repo.get_experiment(artifact.experiment_id)
    if artifact is None or experiment is None or experiment.task_id != task_id or experiment.artifact_id != artifact.id:
        raise AdapterError("registered_native_model_task_binding_required", 422)
    schema = [f.model_dump() for f in request.raw_schema]
    if [f["name"] for f in schema] != list(artifact.feature_list) or len({f["name"] for f in schema}) != len(schema):
        raise AdapterError("registered_model_schema_order_mismatch", 422)
    directory = settings.tasks_dir / task_id / "modeling_artifacts"
    try:
        record, snapshots = load_receipt(settings.db_path, experiment, artifact, directory)
    except (ValueError, KeyError, OSError) as exc:
        raise AdapterError("registered_model_native_closure_unavailable", 422) from exc
    if sum(len(data) for data in snapshots.values()) > 40_000_000:
        raise AdapterError("registered_model_transfer_budget_exceeded", 422)
    files = [{"name": f["path"], "sha256": f["sha256"], "size": f["size"]} for f in record["provenance"]["files"]]
    manifest = {"artifact_id": artifact.id, "algorithm": artifact.algorithm, "model_metadata_hash": metadata_hash(artifact), "schema_hash": digest(schema), "raw_schema": schema, "files": files, "storage_only": True}
    return {"manifest": manifest, "bundle_hash": digest(manifest), "files": {name: base64.b64encode(data).decode("ascii") for name, data in snapshots.items()}}


def verify_registry_readback(payload, original):
    expected = {"bundle_hash": original["bundle_hash"], "manifest": original["manifest"], "files_verified": original["manifest"]["files"], "storage_only": True}
    for key, value in expected.items():
        if payload.get(key) != value:
            raise AdapterError("registry_readback_content_or_schema_mismatch")
    if payload.get("state") not in {"uploaded", "promoted", "revoked"}:
        raise AdapterError("registry_readback_state_invalid")
    return payload
