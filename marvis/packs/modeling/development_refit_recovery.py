"""Reconcile only a native refit producer's own interrupted publication."""

import base64
import hashlib
import json
from pathlib import Path
import re

from marvis.files import write_text_atomic
from marvis.packs.modeling.errors import ModelingError

_MODEL = re.compile(r"artifact_[0-9a-f]{32}")


def journal_path(directory, identity):
    return Path(directory) / f".development-refit-{identity}.pending.json"


def prepare_journal(directory, identity, ref_id, artifact, staged_items):
    latest = Path(directory) / "model_meta.json"
    old = latest.read_bytes() if latest.exists() else None
    items = [
        {
            "final": item.final_path.name,
            "stage": item.stage_path.relative_to(directory).as_posix(),
            "backup": item.backup_path.name,
            "sha256": hashlib.sha256(item.path.read_bytes()).hexdigest(),
            "previous_sha256": hashlib.sha256(item.final_path.read_bytes()).hexdigest()
            if item.final_path.exists()
            else None,
        }
        for item in staged_items
    ]
    body = {
        "schema_version": "development-refit-publication.v1",
        "identity": identity,
        "receipt_id": ref_id,
        "artifact_id": artifact.id,
        "items": items,
        "latest_before": None if old is None else base64.b64encode(old).decode(),
    }
    if len(json.dumps(body)) > 2 * 1024**2:
        raise ModelingError("development_refit_publication_journal_limit")
    path = journal_path(directory, identity)
    write_text_atomic(path, json.dumps(body, sort_keys=True))
    path.chmod(0o600)
    return path


def reconcile_journal(runtime, directory, identity, ref_id, task_id):
    path = journal_path(directory, identity)
    if not path.exists():
        return
    if path.is_symlink() or path.stat().st_size > 2 * 1024**2:
        raise ModelingError("development_refit_publication_journal_invalid")
    body = json.loads(path.read_bytes())
    artifact = body.get("artifact_id")
    if (
        body.get("schema_version") != "development-refit-publication.v1"
        or body.get("identity") != identity
        or body.get("receipt_id") != ref_id
        or not isinstance(artifact, str)
        or not _MODEL.fullmatch(artifact)
    ):
        raise ModelingError("development_refit_publication_journal_invalid")
    allowed = {
        f"{artifact}.joblib",
        f"{artifact}.model_meta.json",
        "model_meta.json",
        f"development_refit_{identity}.json",
    }
    items = body.get("items")
    if not isinstance(items, list) or {item.get("final") for item in items} != allowed:
        raise ModelingError("development_refit_publication_journal_invalid")
    for item in items:
        stage = Path(item["stage"])
        backup = Path(item["backup"])
        if (
            stage.parent != Path(".staging")
            or backup.name != str(backup)
            or not backup.name.startswith(f".{item['final']}.")
            or not backup.name.endswith(".bak")
            or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
        ):
            raise ModelingError("development_refit_publication_journal_invalid")
        for relative in (item["final"], item["stage"], item["backup"]):
            candidate = Path(directory) / relative
            if (
                candidate.is_symlink()
                or candidate.resolve(strict=False) != candidate.absolute()
            ):
                raise ModelingError("development_refit_publication_journal_invalid")
    committed = runtime.task_artifacts.get_for_task(task_id, ref_id)
    if committed is not None and committed["provenance"].get("artifact_id") != artifact:
        raise ModelingError("development_refit_publication_journal_binding_changed")
    if (
        committed is None
        and runtime.modeling_repo.get_model_artifact(artifact) is not None
    ):
        raise ModelingError("development_refit_partial_database_publication")
    for item in items:
        final = Path(directory) / item["final"]
        stage = Path(directory) / item["stage"]
        backup = Path(directory) / item["backup"]
        if committed is None and final.exists():
            current = hashlib.sha256(final.read_bytes()).hexdigest()
            if current == item["sha256"]:
                if backup.exists():
                    if (
                        hashlib.sha256(backup.read_bytes()).hexdigest()
                        != item["previous_sha256"]
                    ):
                        raise ModelingError("development_refit_original_backup_changed")
                    backup.replace(final)
                elif final.name == "model_meta.json":
                    old = body["latest_before"]
                    if old is None:
                        final.unlink()
                    else:
                        write_text_atomic(
                            final, base64.b64decode(old, validate=True).decode("utf-8")
                        )
                elif item["previous_sha256"] is None:
                    final.unlink()
                else:
                    raise ModelingError("development_refit_original_backup_missing")
            elif current != item["previous_sha256"] and final.name != "model_meta.json":
                raise ModelingError("development_refit_uncommitted_file_changed")
        stage.unlink(missing_ok=True)
        backup.unlink(missing_ok=True)
    path.unlink()
