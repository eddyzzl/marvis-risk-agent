"""Native model closure proofs, captured at production and committed with the model.

Missing old receipts stay unknown. Reading an existing pickle is never a way to
manufacture producer evidence. The shared task artifact registry is the ledger.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from marvis.packs.modeling.errors import ModelingError
from marvis.repositories.task_artifacts import TaskArtifactRepository

KIND = "modeling_native_model_closure"
ORIGIN = "modeling.native_model_producer.v1"
_capture = ContextVar("native_model_writes", default=None)


def canonical(value):
    # This serialization is hashed, never interpreted as JSON evidence. Python's
    # distinct Infinity token avoids a collision with user-provided tagged data.
    def normalize(obj):
        if isinstance(obj, dict):
            return {str(k): normalize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [normalize(v) for v in obj]
        if hasattr(obj, "item"):
            return normalize(obj.item())
        return obj

    return json.dumps(
        normalize(value), sort_keys=True, separators=(",", ":"), allow_nan=True
    )


def metadata_hash(artifact):
    data = asdict(artifact)
    # Publication/export annotations do not alter the scorer's inputs.
    for key in (
        "experiment_id",
        "pmml_path",
        "feature_importance",
        "baseline_distributions",
    ):
        data.pop(key, None)
    return hashlib.sha256(canonical(data).encode()).hexdigest()


@contextmanager
def capture_native_writes():
    writes = {"files": {}, "members": []}
    token = _capture.set(writes)
    try:
        yield writes
    finally:
        _capture.reset(token)


def capture_file(name, path):
    writes = _capture.get()
    if writes is not None:
        data = Path(path).read_bytes()
        writes["files"][name] = {
            "path": name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
        }


def capture_members(members):
    writes = _capture.get()
    if writes is not None:
        writes["members"] = [dict(member) for member in members]


def _read(directory, name):
    root = Path(directory).resolve()
    path = root / name
    if (
        Path(name).is_absolute()
        or ".." in Path(name).parts
        or path.resolve() != path.absolute()
        or not path.is_file()
    ):
        raise ModelingError("native_model_source_path_invalid")
    if path.stat().st_size > 512_000_000:
        raise ModelingError("native_model_source_size_limit")
    return path.read_bytes()


def verify_files(directory, files):
    snapshots = {}
    for item in files:
        data = _read(directory, item["path"])
        if (
            len(data) != item["size"]
            or hashlib.sha256(data).hexdigest() != item["sha256"]
        ):
            raise ModelingError("native_model_source_integrity_failed")
        snapshots[item["path"]] = data
    return snapshots


def finish_capture(writes, artifact, directory):
    members = writes["members"] if artifact.algorithm == "ensemble" else []
    names = [artifact.model_path, *(m["model_path"] for m in members)]
    if not all(name in writes["files"] for name in names):
        # Custom/legacy recipes remain usable, but cannot enter a trusted package.
        return None
    files = [writes["files"][name] for name in sorted(set(names))]
    verify_files(directory, files)
    return {
        "directory": str(directory),
        "files": files,
        "members": members,
        "metadata_hash": metadata_hash(artifact),
        "anchor": artifact.model_path,
    }


def register_on_connection(conn, db_path, artifact, receipt):
    if receipt is None:
        return None
    if receipt["metadata_hash"] != metadata_hash(artifact):
        raise ModelingError("native_model_metadata_changed_before_publication")
    verify_files(receipt["directory"], receipt["files"])
    experiment = conn.execute(
        "SELECT task_id FROM experiments WHERE id=?", (artifact.experiment_id,)
    ).fetchone()
    anchor = next(
        item for item in receipt["files"] if item["path"] == receipt["anchor"]
    )
    return TaskArtifactRepository(db_path).register_on_connection(
        conn,
        task_id=experiment["task_id"],
        kind=KIND,
        path=f"modeling_artifacts/{receipt['anchor']}",
        content_hash=anchor["sha256"],
        origin_tool=ORIGIN,
        provenance={
            "schema_version": ORIGIN,
            "artifact_id": artifact.id,
            "experiment_id": artifact.experiment_id,
            "scoring_metadata_hash": receipt["metadata_hash"],
            "files": receipt["files"],
            "members": receipt["members"],
        },
    )


def load_receipt(db_path, experiment, artifact, directory, *, required=True):
    records = [
        r
        for r in TaskArtifactRepository(db_path).list_for_task(experiment.task_id)
        if r["kind"] == KIND and r["provenance"].get("artifact_id") == artifact.id
    ]
    if not records:
        if required:
            raise ModelingError(
                "native_model_producer_evidence_unknown_retrain_required"
            )
        return None
    matching = [
        r
        for r in records
        if r["origin_tool"] == ORIGIN
        and r["provenance"].get("schema_version") == ORIGIN
        and r["provenance"].get("experiment_id") == experiment.id
        and r["provenance"].get("scoring_metadata_hash") == metadata_hash(artifact)
    ]
    if len(matching) != 1:
        raise ModelingError("native_model_metadata_authentication_failed")
    record = matching[0]
    proof = record["provenance"]
    members = proof["members"]
    expected = {artifact.model_path, *(m["model_path"] for m in members)}
    calibration = artifact.params.get("calibration") or {}
    if calibration.get("path"):
        expected.add(calibration["path"])
    if set(f["path"] for f in proof["files"]) != expected:
        raise ModelingError("native_model_dependency_closure_mismatch")
    if artifact.algorithm == "ensemble" and [
        m["artifact_id"] for m in members
    ] != artifact.params.get("ensemble_member_artifact_ids"):
        raise ModelingError("native_model_dependency_closure_mismatch")
    snapshots = verify_files(directory, proof["files"])
    return record, snapshots
