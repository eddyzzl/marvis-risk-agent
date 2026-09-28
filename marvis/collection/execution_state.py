"""Authenticated local queue facts and original governed producer verification."""

from pathlib import Path
import hashlib
import hmac
import json

from marvis.collection.batches import decode_batch
from marvis.collection.ledger import CollectionEvidenceError
from marvis.db_schema import connect
from marvis.decision_twin._canonical import canonical_json, content_hash
from marvis.settings import build_settings

OPERATIONS = {
    "queue_batch": (("proposed",), "queued"),
    "execute_reference": (("queued",), "completed"),
    "cancel_batch": (("proposed", "queued"), "cancelled"),
}


def signing_secret(db_path):
    secret = (
        build_settings(Path(db_path).parent)
        .plugin_admin_token_path.read_text()
        .strip()
        .encode()
    )
    if not secret:
        raise CollectionEvidenceError("collection_authentication_unavailable")
    return secret


def sign(secret, body):
    return hmac.new(
        secret,
        ("collection.reference-execution.v1:" + canonical_json(body)).encode(),
        hashlib.sha256,
    ).hexdigest()


def authenticated(row, secret):
    if row is None:
        raise CollectionEvidenceError("collection_execution_evidence_missing")
    body = json.loads(row["payload_json"])
    if not hmac.compare_digest(row["signature"], sign(secret, body)):
        raise CollectionEvidenceError("collection_execution_integrity_failed")
    return body


def ensure_execution_schema(db_path):
    with connect(db_path) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS collection_queue_items(
            id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
            batch_id TEXT NOT NULL, case_id TEXT NOT NULL, policy_id TEXT NOT NULL,
            queue_id TEXT NOT NULL, subject_namespace TEXT NOT NULL, subject_token TEXT NOT NULL,
            payload_json TEXT NOT NULL, signature TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS collection_queue_scope ON collection_queue_items(policy_id,queue_id);
        CREATE INDEX IF NOT EXISTS collection_queue_subject ON collection_queue_items(subject_namespace,subject_token);
        CREATE TABLE IF NOT EXISTS collection_action_finals(
            action_id TEXT PRIMARY KEY REFERENCES collection_queue_items(id) ON DELETE CASCADE,
            task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
            payload_json TEXT NOT NULL, signature TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS collection_native_effects(
            effect_id TEXT PRIMARY KEY, invocation_id TEXT NOT NULL UNIQUE,
            task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
            batch_id TEXT NOT NULL, payload_json TEXT NOT NULL, signature TEXT NOT NULL);
        CREATE TRIGGER IF NOT EXISTS collection_reference_task_retention
        BEFORE DELETE ON tasks
        WHEN EXISTS(SELECT 1 FROM collection_queue_items WHERE task_id=OLD.id)
        BEGIN SELECT RAISE(ABORT,'collection_reference_evidence_retained'); END;
        """)
        for table in (
            "collection_queue_items",
            "collection_action_finals",
            "collection_native_effects",
        ):
            conn.execute(
                f"CREATE TRIGGER IF NOT EXISTS {table}_no_update BEFORE UPDATE ON {table} BEGIN SELECT RAISE(ABORT,'collection execution evidence is immutable'); END"
            )
            conn.execute(
                f"CREATE TRIGGER IF NOT EXISTS {table}_no_delete BEFORE DELETE ON {table} WHEN EXISTS(SELECT 1 FROM tasks WHERE id=OLD.task_id) BEGIN SELECT RAISE(ABORT,'collection execution evidence is immutable'); END"
            )


def batch_on_connection(conn, task_id, batch_id, secret):
    batch = decode_batch(
        conn.execute(
            "SELECT * FROM collection_batches WHERE task_id=? AND batch_id=?",
            (task_id, batch_id),
        ).fetchone(),
        secret,
    )
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='collection_native_effects'"
    ).fetchone()
    rows = (
        conn.execute(
            "SELECT * FROM collection_native_effects WHERE task_id=? AND batch_id=?",
            (task_id, batch_id),
        ).fetchall()
        if exists
        else []
    )
    if len(rows) > 2:
        raise CollectionEvidenceError("collection_batch_head_integrity_failed")
    chain = [authenticated(row, secret) for row in rows]
    chain.sort(key=lambda item: item["facts"]["target"]["version"])
    status, revision = "proposed", 1
    for item in chain:
        facts, output = item["facts"], item["output"]
        allowed, result_status = OPERATIONS.get(facts["operation"], ((), None))
        target = target_for_batch(
            {**batch, "status": status, "revision": revision}, result_status
        )
        if (
            status not in allowed
            or facts["target"] != target
            or facts["batch_status_at_commit"] != result_status
            or facts["batch_revision_at_commit"] != revision + 1
            or output["status"] != result_status
            or output["revision"] != revision + 1
            or output["receipt_hash"] != content_hash(facts)
        ):
            raise CollectionEvidenceError("collection_batch_head_integrity_failed")
        status, revision = result_status, revision + 1
    if batch["status"] != status or batch["revision"] != revision:
        raise CollectionEvidenceError("collection_batch_head_integrity_failed")
    return batch


def effect_target(db_path, task_id, inputs, target_policy):
    secret = signing_secret(db_path)
    with connect(db_path) as conn:
        batch = batch_on_connection(conn, task_id, inputs.get("batch_id"), secret)
    if (
        inputs.get("request_hash") != batch["request_hash"]
        or inputs.get("preview_hash") != batch["preview_hash"]
    ):
        raise CollectionEvidenceError("collection_reviewed_proposal_mismatch")
    if batch["status"] not in target_policy.expected_statuses:
        raise CollectionEvidenceError("collection_batch_status_changed")
    if (
        tuple(target_policy.expected_statuses),
        target_policy.result_status,
    ) not in OPERATIONS.values():
        raise CollectionEvidenceError("collection_effect_transition_unsupported")
    return target_for_batch(batch, target_policy.result_status)


def target_for_batch(batch, result_status):
    return {
        "kind": "collection_batch",
        "id": batch["batch_id"],
        "task_id": batch["task_id"],
        "expected_status": batch["status"],
        "result_status": result_status,
        "version": batch["revision"],
        "request_hash": batch["request_hash"],
        "preview_hash": batch["preview_hash"],
        "actor_id": batch["actor_id"],
        "execution_mode": "local_reference",
    }


def queue_item(row, secret):
    body = authenticated(row, secret)
    for field in (
        "id",
        "task_id",
        "batch_id",
        "case_id",
        "policy_id",
        "queue_id",
        "subject_namespace",
        "subject_token",
    ):
        if body[field] != row[field]:
            raise CollectionEvidenceError("collection_queue_binding_failed")
    return body


def final_for(conn, item, secret):
    row = conn.execute(
        "SELECT * FROM collection_action_finals WHERE action_id=?", (item["id"],)
    ).fetchone()
    if row is None:
        return None
    body = authenticated(row, secret)
    if (
        body["action_id"] != item["id"]
        or body["task_id"] != item["task_id"]
        or body["state"] not in {"completed", "cancelled"}
    ):
        raise CollectionEvidenceError("collection_action_final_binding_failed")
    return body


def verify_output(conn, db_path, *, binding, invocation_id, output):
    secret = signing_secret(db_path)
    row = conn.execute(
        "SELECT * FROM collection_native_effects WHERE invocation_id=?",
        (invocation_id,),
    ).fetchone()
    native = authenticated(row, secret)
    facts = native["facts"]
    if (
        native["output"] != output
        or binding.tool_ref.split("@")[0] != "collection." + facts["operation"]
        or len(facts["action_ids"]) != len(set(facts["action_ids"]))
        or row["task_id"] != binding.task_id
        or row["batch_id"] != binding.effect_target["id"]
        or facts["bindings"]["effect_execution_id"] != row["effect_id"]
        or facts["bindings"]["invocation_id"] != invocation_id
        or facts["target"] != binding.effect_target
        or output["receipt_hash"] != content_hash(facts)
    ):
        raise CollectionEvidenceError("collection_producer_binding_failed")
    for action_id in facts["action_ids"]:
        item = queue_item(
            conn.execute(
                "SELECT * FROM collection_queue_items WHERE id=?", (action_id,)
            ).fetchone(),
            secret,
        )
        if item["task_id"] != binding.task_id or item["batch_id"] != row["batch_id"]:
            raise CollectionEvidenceError("collection_producer_action_binding_failed")
        if facts["operation"] == "queue_batch":
            if item["effect_id"] != row["effect_id"]:
                raise CollectionEvidenceError(
                    "collection_producer_action_origin_failed"
                )
        else:
            final = final_for(conn, item, secret)
            if final is None or final["effect_id"] != row["effect_id"]:
                raise CollectionEvidenceError("collection_producer_final_missing")
    artifact = conn.execute(
        "SELECT * FROM task_artifacts WHERE task_id=? AND id=?",
        (binding.task_id, output["artifact_id"]),
    ).fetchone()
    settings = build_settings(Path(db_path).parent)
    expected_path = (
        settings.tasks_dir
        / binding.task_id
        / "collection"
        / f"execution-{content_hash(facts['bindings'])}.json"
    )
    if (
        artifact is None
        or artifact["kind"] != "collection_reference_execution"
        or artifact["origin_tool"] != "collection." + facts["operation"]
        or artifact["path"] != str(expected_path.relative_to(settings.workspace))
        or artifact["content_hash"] != content_hash(facts)
        or json.loads(artifact["provenance_json"])
        != {
            "effect_id": row["effect_id"],
            "invocation_id": invocation_id,
            "batch_id": row["batch_id"],
        }
    ):
        raise CollectionEvidenceError("collection_execution_artifact_binding_failed")
    if (
        not expected_path.is_file()
        or expected_path.resolve() != expected_path.absolute()
        or expected_path.read_bytes() != canonical_json(facts).encode()
    ):
        raise CollectionEvidenceError("collection_execution_artifact_integrity_failed")


def target_unchanged(conn, db_path, binding):
    batch = batch_on_connection(
        conn, binding.task_id, binding.effect_target["id"], signing_secret(db_path)
    )
    return (
        target_for_batch(batch, binding.effect_target["result_status"])
        == binding.effect_target
    )
