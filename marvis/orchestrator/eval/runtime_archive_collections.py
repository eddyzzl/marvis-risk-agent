"""Independent declared cashflows and committed local collection effects."""
import json
import hashlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from marvis.collection.contracts import CashflowEvent, CollectionCase, InstallmentSchedule, ReconciliationRequest

from .runtime_archive_domain import equal
from .runtime_archive_reader import _Unsupported
from .runtime_collections_reference import reconciliation_reference
from .runtime_contracts import digest
from .runtime_extensions_archive import ExtensionArchive, FrozenExtensionBinding, Hash, Identity, declared_requests, measured_values
from .runtime_native_identity import declaration_matches


class FrozenReconciliation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    request: ReconciliationRequest
    receipt_hash: Hash
    artifact_id: Identity
    ledger_records_sha256: Hash


class FrozenCollectionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    batch_id: Identity
    records_sha256: Hash


class FrozenCollectionsBinding(FrozenExtensionBinding):
    reconciliations: list[FrozenReconciliation] = Field(default_factory=list, max_length=100)
    batches: list[FrozenCollectionBatch] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def declared(self):
        if self.case.family != "collections" or not (self.reconciliations or self.batches):
            raise ValueError("declared collection inputs required")
        if any(not declaration_matches(self.case, "reconcile_collection_cashflows", item.request) for item in self.reconciliations):
            raise ValueError("reconciliation must match its declared public request")
        if len({item.receipt_hash for item in self.reconciliations}) != len(self.reconciliations):
            raise ValueError("duplicate frozen reconciliation")
        return self


def _sources(ctx, bodies):
    refs = set()
    for body in bodies:
        for key, hash_key in (("source_artifact_id", "source_artifact_hash"), ("terms_artifact_id", "terms_artifact_hash"), ("policy_artifact_id", "policy_artifact_hash"), ("basis_artifact_id", "basis_artifact_hash")):
            if key in body:
                refs.add((body[key], body[hash_key]))
    for artifact_id, sha in sorted(refs):
        ctx.artifact(artifact_id, sha)
    return len(refs)


def _reconcile(ctx, item):
    if not any(a.request == item.request for a in declared_requests(ctx.business_case, "reconcile_collection_cashflows")):
        raise ValueError("frozen reconciliation differs from authenticated original bindings")
    rows = ctx.conn.execute("SELECT * FROM collection_evidence WHERE task_id=? AND case_id=? ORDER BY kind,identity", (ctx.binding.task_id, item.request.case_id)).fetchmany(10003)
    if len(rows) > 10002:
        raise _Unsupported("collection ledger row budget")
    equal(digest([dict(row) for row in rows]), item.ledger_records_sha256)
    decoded = {}
    for row in rows:
        body = json.loads(row["body_json"])
        if row["body_hash"] != digest(body) or body["case_id"] != item.request.case_id:
            raise ValueError("collection ledger record binding differs")
        decoded[(row["kind"], row["identity"])] = body
    case = CollectionCase.model_validate(decoded[("case", item.request.case_id)]).model_dump()
    declared_cases = declared_requests(ctx.business_case, "create_collection_case")
    if not any(a.request.model_dump() == case for a in declared_cases):
        raise ValueError("native case differs from predeclared input")
    flows = [CashflowEvent.model_validate(body).model_dump() for (kind, _), body in decoded.items() if kind == "flow"]
    declared_flows = [e.model_dump() for a in declared_requests(ctx.business_case, "import_collection_cashflows") for e in a.request.events if e.case_id == item.request.case_id]
    equal(sorted(flows, key=digest), sorted({digest(row): row for row in declared_flows}.values(), key=digest))
    request = item.request.model_dump()
    schedule = None
    if request["schedule_id"]:
        schedule = InstallmentSchedule.model_validate(decoded[("schedule", request["schedule_id"])]).model_dump()
        if not any(a.request.model_dump() == schedule for a in declared_requests(ctx.business_case, "create_collection_schedule")):
            raise ValueError("native installment differs from predeclared input")
    _sources(ctx, [case, *flows, *request["coverage"], *([schedule] if schedule else []), *([request["maturity"]] if request["maturity"] else [])])
    expected = reconciliation_reference(case, flows, request, schedule)
    actual = decoded[("reconciliation", item.receipt_hash)]
    measured_values(actual, expected)
    equal(digest(actual), item.receipt_hash)
    artifact = ctx.artifact(item.artifact_id, item.receipt_hash, kind="collection_cashflow_reconciliation", origin="collection.reconcile.v1")
    equal(artifact, actual)
    return {"receipt_hash": item.receipt_hash, "status": expected["status"], "visible_event_count": expected["visible_event_count"], "amounts": expected["amounts"], "observed_subtotals": expected["observed_subtotals"], "reasons": expected["reasons"]}


def _batch_effects(ctx, item):
    batch = ctx.conn.execute("SELECT * FROM collection_batches WHERE task_id=? AND batch_id=?", (ctx.binding.task_id, item.batch_id)).fetchone()
    if batch is None:
        raise ValueError("collection batch missing")
    records = {"batch": dict(batch)}
    for table in ("collection_queue_items", "collection_native_effects"):
        rows = ctx.conn.execute(f"SELECT * FROM {table} WHERE task_id=? AND batch_id=?", (ctx.binding.task_id, item.batch_id)).fetchmany(10001)  # nosec B608 -- table iterates only collection_queue_items/collection_native_effects; task and batch ids are bound
        if len(rows) > 10000:
            raise _Unsupported("collection effect row budget")
        records[table] = sorted([dict(row) for row in rows], key=digest)
    finals = ctx.conn.execute("SELECT f.* FROM collection_action_finals f JOIN collection_queue_items q ON q.id=f.action_id WHERE q.task_id=? AND q.batch_id=? ORDER BY f.action_id", (ctx.binding.task_id, item.batch_id)).fetchmany(10001)
    records["collection_action_finals"] = sorted([dict(row) for row in finals], key=digest)
    equal(digest(records), item.records_sha256)
    request, preview = json.loads(batch["request_json"]), json.loads(batch["preview_json"])
    equal(digest(request), batch["request_hash"])
    equal(digest({key: value for key, value in preview.items() if key != "preview_hash"}), batch["preview_hash"])
    if not any(a.request.model_dump() == request for a in declared_requests(ctx.business_case, "prepare_collection_batch")):
        raise _Unsupported("batch effect proof requires the complete declared native batch DTO; business-proposal compiler is a separate proof scope")
    _sources(ctx, [request["policy"], *request["cases"], *request["histories"]])
    from .runtime_collections_reference import batch_preview_reference
    measured_values(preview, batch_preview_reference(request))
    queues = {row["id"]: json.loads(row["payload_json"]) for row in records["collection_queue_items"]}
    finals = {row["action_id"]: json.loads(row["payload_json"]) for row in records["collection_action_finals"]}
    native = [json.loads(row["payload_json"]) for row in records["collection_native_effects"]]
    if len(native) > 2:
        raise ValueError("too many collection batch transitions")
    native.sort(key=lambda row: row["facts"]["target"]["version"])
    state, revision = "proposed", 1
    transitions = {"queue_batch": ({"proposed"}, "queued"), "execute_reference": ({"queued"}, "completed"), "cancel_batch": ({"proposed", "queued"}, "cancelled")}
    for effect in native:
        facts, output = effect["facts"], effect["output"]
        operation = facts["operation"]
        _governed_effect_closure(ctx, facts, output)
        permitted, target = transitions[operation]
        if state not in permitted or facts["target"]["version"] != revision or facts["target"]["expected_status"] != state:
            raise ValueError("collection effect state transition differs")
        for key in ("request_hash", "preview_hash"):
            equal(facts["target"][key], batch[key])
        if facts["batch_status_at_commit"] != target or facts["batch_revision_at_commit"] != revision + 1 or output["status"] != target or output["revision"] != revision + 1 or output["receipt_hash"] != digest(facts):
            raise ValueError("collection committed transition differs")
        identities = facts["action_ids"]
        if len(set(identities)) != len(identities) or set(identities) != set(queues):
            raise ValueError("collection effect action membership differs")
        if operation != "queue_batch" and set(finals) != set(queues):
            raise ValueError("collection final action membership differs")
        if operation == "queue_batch":
            expected_cost = sum(row["estimated_cost_minor"] or 0 for row in queues.values())
            equal(facts["estimated_cost_minor"], expected_cost)
        for identity in identities:
            queue = queues[identity]
            if (identity != digest([ctx.binding.task_id, item.batch_id, queue["case_id"]]) or queue["task_id"] != ctx.binding.task_id or queue["batch_id"] != item.batch_id):
                raise ValueError("collection queue stable identity differs")
            if operation != "queue_batch":
                final = finals[identity]
                if final["state"] != ("completed" if operation == "execute_reference" else "cancelled") or final["customer_contacted"] is not False or final["external_action_executed"] is not False or final["actual_cost_minor"] is not None:
                    raise ValueError("local collection action must not claim customer contact or measured cost")
        artifact = ctx.artifact(output["artifact_id"], digest(facts), kind="collection_reference_execution", origin="collection." + operation)
        equal(artifact, facts)
        state, revision = target, revision + 1
    if state != batch["status"] or revision != batch["revision"]:
        raise ValueError("collection durable batch head differs")
    return {"batch_id": item.batch_id, "status": state, "revision": revision, "action_count": len(queues), "customer_contacted": False, "actual_cost_minor": None}


def _governed_effect_closure(ctx, facts, output):
    """Read actual approval/run/plan/effect facts without producer validators."""
    bindings = facts["bindings"]
    effect = ctx.conn.execute("SELECT * FROM effect_executions WHERE id=?", (bindings["effect_execution_id"],)).fetchone()
    approval = ctx.conn.execute("SELECT * FROM approval_records WHERE id=?", (bindings["approval_id"],)).fetchone()
    decision = ctx.conn.execute("SELECT * FROM decision_records WHERE id=?", (bindings["decision_id"],)).fetchone()
    run = ctx.conn.execute("SELECT * FROM plan_step_runs WHERE id=?", (bindings["invocation_id"],)).fetchone()
    plan = ctx.conn.execute("SELECT * FROM plans WHERE id=?", (bindings["plan_id"],)).fetchone()
    step = ctx.conn.execute("SELECT * FROM plan_steps WHERE id=?", (bindings["step_id"],)).fetchone()
    if any(row is None for row in (effect, approval, decision, run, plan, step)):
        raise ValueError("collection native governed producer closure incomplete")
    tool = "collection." + facts["operation"]
    if (effect["status"] != "committed" or approval["status"] != "consumed" or decision["decision"] != "approve"
            or approval["principal_id"] != decision["principal_id"] or effect["approval_id"] != approval["id"]
            or effect["invocation_id"] != run["id"] or run["status"] != "succeeded" or run["invocation_id"] != run["id"]
            or run["tool_ref"] != tool or run["plan_id"] != plan["id"] or plan["task_id"] != ctx.binding.task_id
            or step["plan_id"] != plan["id"] or step["status"] != "done" or not step["needs_confirmation"] or not step["confirmed"]
            or bindings["reservation_id"] != effect["reservation_id"] or bindings["reservation_id"] != approval["reservation_id"]
            or effect["released_at"] is not None):
        raise ValueError("collection original human/worker/effect state differs")
    inputs = {key: facts["target"][key] for key in ("request_hash", "preview_hash")}
    inputs["batch_id"] = output["batch_id"]
    equal(json.loads(run["input_json"]), inputs)
    for key in ("task_id", "plan_id", "plan_revision", "step_id", "tool_ref", "manifest_hash", "policy_hash", "input_hash", "evidence_hash", "effect_target_json"):
        equal(approval[key], decision[key])
    equal(json.loads(approval["effect_target_json"]), facts["target"])
    typed = ["dict", [[["str", key], ["str", value]] for key, value in sorted(inputs.items())]]
    equal(approval["input_hash"], digest(["marvis-governance-canonical.v1", typed]))
    declaration = json.loads(run["invocation_contract_json"])
    invocation_hash = "sha256:" + hashlib.sha256(json.dumps(declaration, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    if (approval["tool_ref"].split("@")[0] != tool or bindings["tool_ref"] != approval["tool_ref"]
            or bindings["invocation_contract_hash"] != effect["invocation_contract_hash"]
            or bindings["invocation_contract_hash"] != invocation_hash):
        raise ValueError("collection native code/input invocation binding differs")


def revalidate_collections_archive(archive: Path, *, expected_manifest_sha256: str, frozen_binding: FrozenCollectionsBinding):
    binding = FrozenCollectionsBinding.model_validate(frozen_binding.model_dump())
    ctx = ExtensionArchive(archive, expected_manifest_sha256, binding, "collections", ("source_artifacts", "independent_reconciliation", "local_effect_closure"))
    with ctx:
        ctx.open()
        ctx.current = "independent_reconciliation"
        reports = [_reconcile(ctx, item) for item in binding.reconciliations]
        ctx.mark("source_artifacts", scope="registered task artifact hashes; declared source truth unverified")
        ctx.mark("independent_reconciliation", count=len(reports))
        ctx.current = "local_effect_closure"
        batches = [_batch_effects(ctx, item) for item in binding.batches]
        ctx.mark("local_effect_closure", count=len(batches))
        ctx.mark("native_records", scope="externally pinned signed ledger bytes and actual governed producer closure; HMAC authority unavailable")
        ctx.result["values"] = {"reconciliations": reports, "batches": batches}
        ctx.result["unsupported"].update(causal_incremental_recovery="retrospective declared cashflows cannot identify treatment effects", legal_contact_authority="business policy declarations are not legal/contact authorization", live_cross_task_capacity_and_concurrency="requires separately held precommit resource snapshot and runtime race adjudication")
    return ctx.result
