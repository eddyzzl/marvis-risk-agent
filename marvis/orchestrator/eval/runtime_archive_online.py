"""Independent local online decisions, deployment bindings and operations ledger."""
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from marvis.reference_decision.contracts import DecisionRequest

from .runtime_archive_domain import equal
from .runtime_archive_reader import _Unsupported
from .runtime_contracts import digest
from .runtime_extensions_archive import ExtensionArchive, FrozenExtensionBinding, Hash, Identity, declared_requests, measured_values
from .runtime_online_reference import audit_chain_reference, calendar_reference, rule_decision_reference, schedule_contract_reference
from .runtime_native_identity import declaration_matches


class FrozenOnlineDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    request: DecisionRequest
    slot: str = Field(pattern=r"^(production|shadow)$")
    record_sha256: Hash


class FrozenOperationsSchedule(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schedule_id: Identity
    revision: int = Field(ge=1)
    records_sha256: Hash


class FrozenOnlineBinding(FrozenExtensionBinding):
    package_hash: Hash
    package_record_sha256: Hash
    deployment_id: Identity
    deployment_records_sha256: Hash
    production_audit_head_sha256: Hash
    decisions: list[FrozenOnlineDecision] = Field(default_factory=list, max_length=100)
    schedules: list[FrozenOperationsSchedule] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def declared(self):
        if self.case.family != "online_decision" or not (self.decisions or self.schedules):
            raise ValueError("declared online/operations inputs required")
        for item in self.decisions:
            if item.request.expected_package_hash != self.package_hash or not any(declaration_matches(self.case, kind, item.request, slot=item.slot) for kind in ("submit_reference_decision", "replay_reference_decision")):
                raise ValueError("frozen decision must match a declared public business request")
        if len({(item.slot, item.request.request_id) for item in self.decisions}) != len(self.decisions):
            raise ValueError("freeze each durable decision once, including replay")
        return self


def _deployments(ctx, binding, manifest):
    deployment = ctx.conn.execute("SELECT * FROM production_deployments WHERE id=?", (binding.deployment_id,)).fetchone()
    if deployment is None:
        raise ValueError("native deployment absent")
    promotion = ctx.conn.execute("SELECT * FROM production_promotion_requests WHERE id=?", (deployment["promotion_request_id"],)).fetchone()
    approvals = ctx.conn.execute("SELECT * FROM production_promotion_approvals WHERE request_id=? ORDER BY stage", (deployment["promotion_request_id"],)).fetchall()
    installation = ctx.conn.execute("SELECT * FROM reference_installations WHERE promotion_id=?", (deployment["promotion_request_id"],)).fetchone()
    deployed = ctx.conn.execute("SELECT * FROM production_deployment_manifests WHERE id=?", (deployment["manifest_hash"],)).fetchone()
    head = ctx.conn.execute("SELECT * FROM production_environment_heads WHERE environment='local-reference'").fetchone()
    if any(row is None for row in (promotion, installation, deployed, head)):
        raise ValueError("native deployment governance chain incomplete")
    records = {"deployment": dict(deployment), "promotion": dict(promotion), "approvals": [dict(row) for row in approvals], "installation": dict(installation), "manifest": dict(deployed), "head": dict(head)}
    equal(digest(records), binding.deployment_records_sha256)
    deployed_body = json.loads(deployed["canonical_json"])
    equal(digest(deployed_body), deployment["manifest_hash"])
    if (deployed_body.get("decision_package_hash") != binding.package_hash
            or installation["package_hash"] != binding.package_hash
            or promotion["status"] != "promoted" or promotion["promoted_deployment_id"] != binding.deployment_id
            or deployment["environment"] != "local-reference" or deployment["manifest_hash"] != promotion["manifest_hash"]
            or deployment["strategy_content_hash"] != manifest["strategy"]["strategy_content_hash"]):
        raise ValueError("package/promotion/deployment identity differs")
    roles = {row["stage"]: row for row in approvals}
    if set(roles) != {"checker", "admin"} or len({promotion["maker_principal_id"], roles["checker"]["principal_id"], roles["admin"]["principal_id"]}) != 3:
        raise ValueError("independent maker/checker/admin identities required")
    for stage, row in roles.items():
        principal = ctx.conn.execute("SELECT * FROM production_principals WHERE local_principal_id=?", (row["principal_id"],)).fetchone()
        if row["role"] != stage or principal is None or principal["role"] != stage:
            raise ValueError("native deployment role differs")
    strategy = ctx.conn.execute("SELECT * FROM strategies WHERE id=?", (deployment["strategy_id"],)).fetchone()
    if strategy is None or strategy["task_id"] != binding.task_id or strategy["version"] != deployment["strategy_version"]:
        raise ValueError("online package native strategy identity differs")
    equal(json.loads(strategy["dsl_json"]), manifest["strategy_spec"])
    equal(digest(manifest["strategy_spec"]), deployment["strategy_content_hash"])
    pointer = "active_deployment_id" if deployment["deployment_slot"] == "production" else "shadow_deployment_id"
    if head[pointer] != binding.deployment_id or deployment["status"] not in {"active", "shadow"}:
        raise _Unsupported("historical or rolled-back deployment requires an explicitly frozen per-request head history")
    rows = [dict(row) for row in ctx.conn.execute("SELECT * FROM production_governance_events ORDER BY sequence").fetchmany(10001)]
    if len(rows) > 10000:
        raise _Unsupported("production audit row budget")
    return audit_chain_reference(rows, binding.production_audit_head_sha256)


def _operations(ctx, item):
    record = ctx.conn.execute("SELECT * FROM operations_schedules WHERE schedule_id=? AND revision=?", (item.schedule_id, item.revision)).fetchone()
    if record is None:
        raise ValueError("operations schedule missing")
    tables = {}
    for table in ("operations_periods", "operations_runs", "operations_outcomes", "operations_notification_outbox"):
        rows = ctx.conn.execute(f"SELECT * FROM {table} WHERE schedule_id=? ORDER BY period_key", (item.schedule_id,)).fetchmany(10001)  # nosec B608 -- table is from the fixed four-table operations tuple; schedule id is bound
        if len(rows) > 10000:
            raise _Unsupported("operations ledger row budget")
        tables[table] = sorted([dict(row) for row in rows], key=digest)
    tables["schedule"] = dict(record)
    equal(digest(tables), item.records_sha256)
    contract = json.loads(record["contract_json"])
    equal(digest(contract), record["contract_hash"])
    declared = declared_requests(ctx.business_case, "publish_operations_schedule")
    matches = [a for a in declared if a.request.schedule.schedule_id == item.schedule_id and a.request.schedule.revision == item.revision]
    if len(matches) != 1:
        raise ValueError("operations schedule must be publicly declared")
    # The public endpoint binds this local principal itself. The role is the
    # declared actor in the already pinned three-party deployment chain.
    actor = matches[0].actor
    if actor == "maker":
        principal = ctx.conn.execute("SELECT maker_principal_id FROM production_promotion_requests WHERE promoted_deployment_id=?", (ctx.binding.deployment_id,)).fetchone()
        publisher = None if principal is None else principal["maker_principal_id"]
    elif actor == "admin":
        principal = ctx.conn.execute("SELECT a.principal_id FROM production_promotion_approvals a JOIN production_deployments d ON a.request_id=d.promotion_request_id WHERE d.id=? AND a.stage='admin'", (ctx.binding.deployment_id,)).fetchone()
        publisher = None if principal is None else principal["principal_id"]
    else:
        raise ValueError("operations publisher role differs")
    value = matches[0].request.schedule.model_dump(mode="json", exclude_none=True)
    expected_contract = schedule_contract_reference(value, execution_actor_id=publisher)
    equal(contract, expected_contract)
    for period in tables["operations_periods"]:
        if period["schedule_revision"] != item.revision:
            raise _Unsupported("mixed schedule revisions require separately frozen contracts")
        calendar_reference(contract, period)
        runs = [row for row in tables["operations_runs"] if row["period_key"] == period["period_key"]]
        if sorted(row["attempt_number"] for row in runs) != list(range(1, period["attempt_count"] + 1)):
            raise ValueError("operations attempt sequence differs")
        outcomes = [row for row in tables["operations_outcomes"] if row["period_key"] == period["period_key"]]
        if period["state"] == "succeeded":
            if len(outcomes) != 1 or outcomes[0]["outcome_id"] != period["outcome_id"] or outcomes[0]["run_id"] not in {row["run_id"] for row in runs}:
                raise ValueError("operations successful outcome closure differs")
        for row in outcomes:
            equal(digest(json.loads(row["outcome_json"])), row["outcome_hash"])
    for row in tables["operations_notification_outbox"]:
        equal(digest(json.loads(row["payload_json"])), row["payload_hash"])
        if row["outcome_id"] not in {outcome["outcome_id"] for outcome in tables["operations_outcomes"]}:
            raise ValueError("notification outcome identity differs")
    return {"period_count": len(tables["operations_periods"]), "run_count": len(tables["operations_runs"]), "outcome_count": len(tables["operations_outcomes"])}


def revalidate_online_archive(archive: Path, *, expected_manifest_sha256: str, frozen_binding: FrozenOnlineBinding):
    binding = FrozenOnlineBinding.model_validate(frozen_binding.model_dump())
    ctx = ExtensionArchive(archive, expected_manifest_sha256, binding, "online", ("deployment_governance", "independent_decisions", "operations_ledger"))
    with ctx:
        ctx.open()
        ctx.current = "native_records"
        row = ctx.row("SELECT * FROM reference_decision_packages WHERE id=?", (binding.package_hash,), binding.package_record_sha256)
        manifest = json.loads(row["canonical_json"])
        equal(digest(manifest), binding.package_hash)
        ctx.mark("native_records", scope="externally pinned signed record bytes; HMAC authority unavailable")
        ctx.current = "deployment_governance"
        chain_count = _deployments(ctx, binding, manifest)
        ctx.mark("deployment_governance", audit_event_count=chain_count)
        ctx.current = "independent_decisions"
        outcomes = []
        for item in binding.decisions:
            if not any(a.request == item.request and a.slot == item.slot for kind in ("submit_reference_decision", "replay_reference_decision") for a in declared_requests(ctx.business_case, kind)):
                raise ValueError("frozen decision differs from authenticated original bindings")
            request = item.request.model_dump(exclude={"event_evidence"} if item.request.event_evidence is None else set())
            row = ctx.row("SELECT * FROM reference_decisions WHERE environment=? AND request_id=?", ("local-reference:" + item.slot, item.request.request_id), item.record_sha256)
            if row["status"] != "completed" or row["input_hash"] != digest(request) or row["package_hash"] != binding.package_hash:
                raise ValueError("durable online request binding differs")
            response = json.loads(row["response_json"])
            if response["status"] != "decided":
                raise _Unsupported("fallback is explicitly unavailable scoring, not a successful independently evaluated decision")
            expected = rule_decision_reference(manifest, request)
            measured_values(response, expected)
            equal(response["decision_id"], digest({"scope": "local-reference:" + item.slot, "request_id": item.request.request_id}))
            equal(response["output_hash"], digest({key: response[key] for key in ("action", "matched_rule_id", "score", "score_product", "package_hash", "versions", "status")}))
            outcomes.append({"request_id": item.request.request_id, "action_type": expected["action"]["type"], "matched_rule_id": expected["matched_rule_id"]})
        ctx.mark("independent_decisions", count=len(outcomes))
        ctx.current = "operations_ledger"
        schedules = [_operations(ctx, item) for item in binding.schedules]
        ctx.mark("operations_ledger", count=len(schedules))
        ctx.result["values"] = {"decisions": outcomes, "schedules": schedules}
        ctx.result["unsupported"].update(model_package_scoring="safe independently implemented model/preprocessing carrier required", installation_probe_timing="not independently rerun", operations_monitoring_measurements="calendar/outcome closure only; use independent monitoring domain evidence for measurements")
    return ctx.result
