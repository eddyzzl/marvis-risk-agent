"""Local native basis/grant checks before an explicit source protocol query."""
import hashlib
import hmac
import json
from pathlib import Path
import time

from marvis.db_schema import connect
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.risk_context.source_contracts import ProviderQuery, ProviderResult, SourceGrant, assess, instant, iso
from marvis.risk_context.source_repository import SourceRepository, require_actor

from .contracts import AdapterError


def declared_basis(settings, task_id, artifact_id):
    record = TaskArtifactRepository(settings.db_path).get_for_task(task_id, artifact_id)
    if record is None or record["kind"] != "source_authorization_basis" or record["origin_tool"] != "risk_context.source_query.v1":
        raise AdapterError("native_source_authorization_basis_required", 422)
    path = Path(record["path"])
    if not path.is_absolute():
        path = settings.workspace / path
    if not path.is_file() or path.resolve() != path.absolute() or not path.is_relative_to(settings.tasks_dir / task_id):
        raise AdapterError("native_source_basis_path_invalid")
    data = path.read_bytes()
    if len(data) > 65536 or hashlib.sha256(data).hexdigest() != record["content_hash"]:
        raise AdapterError("native_source_basis_integrity_failed")
    value = json.loads(data)
    repo = SourceRepository(settings)
    if value["body"]["task_id"] != task_id or not hmac.compare_digest(value["signature"], repo.sign(value["body"])):
        raise AdapterError("native_source_basis_authentication_failed")
    return record["content_hash"]


def authorize_grant(settings, alias, task_id, actor, body):
    grant = SourceGrant.model_validate(body["grant"])
    if grant.profile_id != alias or grant.task_id != task_id or grant.grantee_id != actor:
        raise AdapterError("configured_source_grant_scope_mismatch", 403)
    if not instant(grant.starts_at) <= time.time() < instant(grant.expires_at):
        raise AdapterError("configured_source_grant_not_current", 403)
    if declared_basis(settings, task_id, grant.basis_artifact_id) != body["basis_hash"]:
        raise AdapterError("configured_source_grant_basis_changed", 403)
    with connect(settings.db_path) as conn:
        require_actor(conn, actor, {"maker"})
    return grant


def query_payload(grant, operation_id, expires_in_seconds):
    query = ProviderQuery(request_id=hashlib.sha256(operation_id.encode()).hexdigest(), subject_namespace=grant.subject_namespace, subject_token=grant.subject_token, purpose=grant.purpose, expires_at=iso(min(time.time() + expires_in_seconds, instant(grant.expires_at))))
    return {"query": query.model_dump()}


def source_result(payload, query):
    result = ProviderResult.model_validate(payload["response"])
    from marvis.reference_decision.contracts import digest
    if result.request_id != query["request_id"] or result.request_hash != digest(query):
        raise AdapterError("source_response_request_binding_mismatch")
    if result.envelope is not None and (result.envelope.subject_namespace != query["subject_namespace"] or result.envelope.subject_token != query["subject_token"]):
        raise AdapterError("source_response_subject_binding_mismatch")
    assessment = assess(result.envelope, instant(payload["recorded_at"])) if result.envelope else {"status": "unknown", "missing_reasons": ["provider_no_record"], "next_action": "required_review"}
    return {"provider_status": result.status, "assessment": assessment, "source_truth": "provider_claims_not_independently_attested", "automated_clearance": False}
