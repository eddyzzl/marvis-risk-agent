"""Explicit configured effects with durable identity and observed remote readback."""

from marvis.db_schema import connect
from marvis.production_governance.evidence import ActivationVerificationContext, VerifiedActivationEvidence
from marvis.production_governance.errors import GovernanceConflict
from marvis.production_governance.repository import ProductionGovernanceRepository
from marvis.reference_decision.contracts import digest
from marvis.risk_context.source_repository import require_actor

from .contracts import AdapterError, EndpointConfig, RemoteMutation
from .deployment import approved_promotion, install_payload
from .journal import Journal
from .registry import native_upload, verify_registry_readback
from .source import authorize_grant, declared_basis, query_payload, source_result
from .transport import Transport


class InstitutionAdapters:
    def __init__(self, settings, endpoints=()):
        self.settings = settings
        self.configs = {e.alias: EndpointConfig.model_validate(e.model_dump()) for e in endpoints}
        if len(self.configs) != len(endpoints) or len({e.environment for e in endpoints}) != len(endpoints):
            raise AdapterError("configured_alias_or_environment_duplicate", 503)
        self.journal = Journal(settings.workspace / "institution_adapters")
        self.governance = ProductionGovernanceRepository(settings.db_path)

    def config(self, alias):
        if alias not in self.configs:
            raise AdapterError("institution_adapter_unconfigured", 503, next_action="请由本地管理员配置机构协议与凭据；当前没有目标系统")
        return self.configs[alias]

    def capabilities(self):
        return {"protocol": "marvis.institution-adapter.v1", "status": "configured" if self.configs else "unconfigured", "endpoints": [{"alias": e.alias, "instance_id": e.instance_id, "environment": e.environment, "mode": e.mode} for e in self.configs.values()], "credentials_local_only": True, "registry_execution": "opaque_native_storage_only", "deployment_execution": "rule_only", "institution_production_acceptance": "not_established"}

    def role(self, task_id, actor, roles):
        with connect(self.settings.db_path) as conn:
            if conn.execute("SELECT 1 FROM tasks WHERE id=?", (task_id,)).fetchone() is None:
                raise AdapterError("adapter_task_not_found", 404)
            require_actor(conn, actor, roles)

    def _owned(self, alias, identity, task_id, actor):
        self.config(alias)
        row = self.journal.read(alias, identity)
        self.role(task_id, actor, {"maker", "checker", "admin"})
        with connect(self.settings.db_path) as conn:
            role = require_actor(conn, actor, {"maker", "checker", "admin"})
        if row["task_id"] != task_id or (row["actor_id"] != actor and role != "admin"):
            raise AdapterError("adapter_operation_scope_forbidden", 403)
        original = self.journal.decode(row)
        if original["payload"]["endpoint_config_hash"] != digest(self.config(alias).model_dump(mode="json")):
            raise AdapterError("configured_endpoint_binding_changed")
        if original["kind"] == "source_query":
            body = self.journal.read_grant(alias, original["payload"]["business_request"]["grant_id"])
            authorize_grant(self.settings, alias, task_id, row["actor_id"], body)
        return row, original

    def _perform(self, alias, task_id, actor, identity, kind, payload):
        config = self.config(alias)
        request = RemoteMutation(operation_id=identity, kind=kind, payload={**payload, "endpoint_config_hash": digest(config.model_dump(mode="json"))})
        if not self.journal.claim(alias, task_id, actor, request):
            return self.read(alias, task_id, actor, identity)
        try:
            transport = Transport(config)
            transport.request("GET", resource="capabilities")
            receipt = transport.request("POST", body=request.model_dump())
            self.journal.receipt(alias, identity, receipt)
        except Exception:
            self.journal.uncertain(alias, identity)
            raise
        return self.read(alias, task_id, actor, identity, finish=True)

    def _replay(self, alias, task_id, actor, request, kind):
        try:
            row = self.journal.read(alias, request.operation_id)
        except AdapterError as exc:
            if exc.code == "operation_not_found":
                return None
            raise
        original = self.journal.decode(row)
        if original["kind"] != kind or original["payload"]["business_request"] != request.model_dump():
            raise AdapterError("operation_identity_conflict")
        return self.read(alias, task_id, actor, request.operation_id)

    def read(self, alias, task_id, actor, identity, *, finish=False):
        row, original = self._owned(alias, identity, task_id, actor)
        receipt = self.journal.result(row)
        if receipt["state"] != "completed":
            raise AdapterError("remote_operation_rejected", 422)
        result = receipt["result"]
        pending = False
        if original["kind"] == "registry_upload":
            verify_registry_readback(result, original["payload"])
        if original["kind"] in {"registry_promote", "registry_revoke"}:
            expected = {"bundle_hash": original["payload"]["bundle_hash"], "schema_hash": original["payload"]["schema_hash"], "state": "promoted" if original["kind"] == "registry_promote" else "revoked", "serving": False}
            if result != expected:
                raise AdapterError("registry_transition_receipt_mismatch")
        if original["kind"] == "deployment_install":
            if (result.get("manifest") != original["payload"]["manifest"] or result.get("bundle_hash") != original["payload"]["bundle_hash"]
                    or result.get("install_ref") != identity or result.get("health_status") != "healthy"):
                raise AdapterError("deployment_install_receipt_mismatch")
        if original["kind"] == "source_query":
            result = source_result(result, original["payload"]["query"])
        if original["kind"] == "deployment_activate":
            completion = self.journal.completion(row)
            if completion is None and finish:
                completion = self._finish_activation(alias, row, original, receipt)
            pending = completion is None
            result = {**result, "local_governance": completion}
        if original["kind"] == "deployment_rollback":
            completion = self.journal.completion(row)
            if completion is None and finish:
                completion = self._finish_rollback(alias, row, original, receipt)
            pending = completion is None
            result = {**result, "local_governance": completion}
        return {"operation_id": identity, "kind": original["kind"], "status": "remote_committed_local_pending" if pending else "completed", "request_hash": row["request_hash"], "result": result, "assurance": self.config(alias).mode, "next_action": "显式恢复以完成本地治理；不重新发送远端操作" if pending else "none", "institution_production_acceptance": "not_established"}

    def recover(self, alias, task_id, actor, identity):
        row, original = self._owned(alias, identity, task_id, actor)
        # Always GET, including claimed/crashed/rejected/unknown requests. There
        # is no hidden re-send even when the remote server currently says 404.
        receipt = Transport(self.config(alias)).request("GET", identity)
        self.journal.receipt(alias, identity, receipt)
        return self.read(alias, task_id, actor, identity, finish=True)

    def upload(self, alias, task_id, actor, request):
        self.role(task_id, actor, {"maker"})
        payload = native_upload(self.settings, task_id, request)
        replay = self._replay(alias, task_id, actor, request, "registry_upload")
        if replay is not None:
            return replay
        return self._perform(alias, task_id, actor, request.operation_id, "registry_upload", {**payload, "business_request": request.model_dump()})

    def registry_readback(self, alias, task_id, actor, identity):
        row, original = self._owned(alias, identity, task_id, actor)
        if original["kind"] != "registry_upload":
            raise AdapterError("registry_upload_operation_required", 422)
        self.journal.result(row)
        observed = Transport(self.config(alias)).request("GET", original["payload"]["bundle_hash"], resource="registry")
        return verify_registry_readback(observed.payload, original["payload"])

    def registry_transition(self, alias, task_id, actor, request, kind):
        self.role(task_id, actor, {"admin"})
        replay = self._replay(alias, task_id, actor, request, kind)
        if replay is not None:
            return replay
        observed = self.registry_readback(alias, task_id, actor, request.upload_operation_id)
        promotion = approved_promotion(self.settings, self.config(alias), request.promotion_id)
        target = {"bundle_hash": observed["bundle_hash"], "model_artifact_id": observed["manifest"]["artifact_id"], "schema_hash": observed["manifest"]["schema_hash"]}
        if promotion["task_id"] != task_id or promotion["strategy_spec"]["metadata"].get("institution_registry") != target:
            raise AdapterError("approved_registry_exact_binding_required", 422)
        return self._perform(alias, task_id, actor, request.operation_id, kind, {**target, "context": promotion["context"], "business_request": request.model_dump()})

    def install(self, alias, task_id, actor, request):
        self.role(task_id, actor, {"admin"})
        replay = self._replay(alias, task_id, actor, request, "deployment_install")
        if replay is not None:
            return replay
        payload = install_payload(self.settings, self.config(alias), task_id, request)
        return self._perform(alias, task_id, actor, request.operation_id, "deployment_install", {**payload, "business_request": request.model_dump()})

    def deployment_readback(self, alias, task_id, actor, identity):
        row, original = self._owned(alias, identity, task_id, actor)
        if original["kind"] != "deployment_install":
            raise AdapterError("deployment_install_operation_required", 422)
        self.journal.result(row)
        observed = Transport(self.config(alias)).request("GET", identity, resource="deployment")
        if observed.payload.get("bundle_hash") != original["payload"]["bundle_hash"] or observed.payload.get("manifest") != original["payload"]["manifest"] or observed.payload.get("health_status") != "healthy":
            raise AdapterError("deployment_observed_health_or_content_mismatch")
        return observed.payload

    def activate(self, alias, task_id, actor, request):
        self.role(task_id, actor, {"admin"})
        replay = self._replay(alias, task_id, actor, request, "deployment_activate")
        if replay is not None:
            return replay
        _, installed = self._owned(alias, request.install_operation_id, task_id, actor)
        observed = self.deployment_readback(alias, task_id, actor, request.install_operation_id)
        promotion = approved_promotion(self.settings, self.config(alias), installed["payload"]["manifest"]["context"]["promotion_request_id"])
        if promotion["context"] != installed["payload"]["manifest"]["context"]:
            raise AdapterError("installed_approval_binding_changed")
        head = self.governance.get_environment(self.config(alias).environment)
        current = head["active" if promotion["context"]["deployment_slot"] == "production" else "shadow"]
        if (None if current is None else current["id"]) != promotion["expected_current_deployment_id"]:
            raise AdapterError("local_deployment_head_changed")
        payload = {"install_operation_id": request.install_operation_id, "bundle_hash": observed["bundle_hash"], "context": promotion["context"], "expected_current_ref": None if current is None else current["external_deployment_ref"], "business_request": request.model_dump()}
        return self._perform(alias, task_id, actor, request.operation_id, "deployment_activate", payload)

    def _finish_activation(self, alias, row, original, receipt):
        payload, result = original["payload"], receipt["result"]
        context = ActivationVerificationContext(**payload["context"])
        if result.get("active_ref") != row["id"] or result.get("install_ref") != payload["install_operation_id"] or result.get("bundle_hash") != payload["bundle_hash"] or result.get("health_status") != "healthy":
            raise AdapterError("remote_activation_binding_mismatch")
        evidence = VerifiedActivationEvidence.issue(verifier_id="institution-adapter.v1", receipt_id=row["id"], context=context, external_deployment_ref=row["id"], health_evidence_ref=result["health_evidence_hash"], health_status="healthy", verified_at=result["verified_at"])
        existing = self.governance.get_promotion_request(context.promotion_request_id)
        if existing["status"] == "promoted":
            with connect(self.settings.db_path) as conn:
                found = conn.execute("SELECT * FROM production_deployments WHERE id=?", (existing["promoted_deployment_id"],)).fetchone()
            if found is None or found["external_deployment_ref"] != row["id"] or found["activation_evidence_id"] != evidence.content_hash or found["manifest_hash"] != context.manifest_hash:
                raise AdapterError("local_activation_replay_binding_mismatch")
            completion = {"deployment_id": found["id"], "status": "activated"}
        else:
            try:
                created = self.governance.activate_promotion_request(request_id=context.promotion_request_id, environment=context.environment, actor_principal_id=row["actor_id"], evidence=evidence, reason="configured adapter observed remote activation")
            except GovernanceConflict as exc:
                raise AdapterError("remote_committed_local_activation_pending", next_action="远端已提交，本地治理尚未完成；显式恢复并核查两个头，不得重新发送激活") from exc
            completion = {"deployment_id": created["id"], "status": "activated"}
        self.journal.finalize(alias, row["id"], completion)
        return completion

    def rollback(self, alias, task_id, actor, request):
        self.role(task_id, actor, {"admin"})
        replay = self._replay(alias, task_id, actor, request, "deployment_rollback")
        if replay is not None:
            return replay
        row, active = self._owned(alias, request.deployment_operation_id, task_id, actor)
        if active["kind"] != "deployment_activate":
            raise AdapterError("activated_deployment_operation_required", 422)
        completion = self.journal.completion(row)
        if completion is None:
            raise AdapterError("local_activation_completion_required")
        context = active["payload"]["context"]
        head = self.governance.get_environment(context["environment"])
        current = head["active" if context["deployment_slot"] == "production" else "shadow"]
        if current is None or current["id"] != completion["deployment_id"] or current["external_deployment_ref"] != request.expected_current_ref:
            raise AdapterError("rollback_current_head_changed")
        with connect(self.settings.db_path) as conn:
            predecessor = conn.execute("SELECT * FROM production_deployments WHERE id=?", (current["predecessor_deployment_id"],)).fetchone()
        if predecessor is None:
            raise AdapterError("rollback_native_predecessor_required", 422)
        payload = {"context": context, "local_deployment_id": current["id"], "expected_current_ref": request.expected_current_ref, "restore_ref": None if predecessor is None else predecessor["external_deployment_ref"], "business_request": request.model_dump()}
        return self._perform(alias, task_id, actor, request.operation_id, "deployment_rollback", payload)

    def _finish_rollback(self, alias, row, original, receipt):
        payload = original["payload"]
        result = receipt["result"]
        if result.get("active_ref") != payload["restore_ref"] or result.get("rolled_back_ref") != payload["expected_current_ref"]:
            raise AdapterError("remote_rollback_binding_mismatch")
        head = self.governance.get_environment(payload["context"]["environment"])
        current = head["active" if payload["context"]["deployment_slot"] == "production" else "shadow"]
        if current is None or current["external_deployment_ref"] != payload["restore_ref"]:
            try:
                self.governance.rollback_deployment(environment=payload["context"]["environment"], deployment_id=payload["local_deployment_id"], expected_active_deployment_id=payload["local_deployment_id"], actor_principal_id=row["actor_id"], deployment_slot=payload["context"]["deployment_slot"], reason="configured adapter observed remote rollback")
            except GovernanceConflict as exc:
                raise AdapterError("remote_committed_local_rollback_pending") from exc
        completion = {"status": "rolled_back", "restored_ref": payload["restore_ref"]}
        self.journal.finalize(alias, row["id"], completion)
        return completion

    def grant_source(self, alias, task_id, actor, grant):
        self.config(alias)
        self.role(task_id, actor, {"admin"})
        if grant.profile_id != alias or grant.task_id != task_id:
            raise AdapterError("configured_source_grant_scope_mismatch", 403)
        body = {"grant": grant.model_dump(), "basis_hash": declared_basis(self.settings, task_id, grant.basis_artifact_id)}
        authorize_grant(self.settings, alias, task_id, grant.grantee_id, body)
        self.journal.grant(alias, grant.grant_id, body)
        return {"grant_id": grant.grant_id, "status": "declared", "authorization": "local_admin_declaration_not_external_attestation"}

    def query_source(self, alias, task_id, actor, request):
        self.role(task_id, actor, {"maker"})
        grant = authorize_grant(self.settings, alias, task_id, actor, self.journal.read_grant(alias, request.grant_id))
        try:
            row = self.journal.read(alias, request.operation_id)
        except AdapterError as exc:
            if exc.code != "operation_not_found":
                raise
            row = None
        if row is not None:
            original = self.journal.decode(row)
            if original["kind"] != "source_query" or original["payload"]["business_request"] != request.model_dump():
                raise AdapterError("operation_identity_conflict")
            return self.read(alias, task_id, actor, request.operation_id)
        return self._perform(alias, task_id, actor, request.operation_id, "source_query", {**query_payload(grant, request.operation_id, request.expires_in_seconds), "business_request": request.model_dump()})
