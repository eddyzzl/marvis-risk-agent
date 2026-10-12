"""Independent original CAS, event time and local durable duplicate checks.

No business producer evaluator, signing key or network callback is used. Missing
original before/fault/authority data remains explicitly unavailable.
"""
from contextlib import closing
import json
import hashlib
from pathlib import Path
import sqlite3

from .runtime_archive_connector import _event
from .runtime_archive_domain import equal
from .runtime_archive_reader import _Unsupported, _file, _json
from .runtime_collections_reference import at
from .runtime_contracts import digest
from .runtime_extensions_archive import ExtensionArchive
from .runtime_native_json import decode_native
from .runtime_safety_contracts import FrozenSafetyBinding
from .runtime_safety_reference import governance_input_reference
from .runtime_fault_reference import archived_target, native_completed_output, native_json_sha, original_progress_to_completed, original_target, resolved_original_inputs

_PLAN_FIELDS = ("id", "task_id", "goal", "source", "template_id", "steps", "autonomy_level", "status", "created_at", "updated_at", "novel_mode", "tier", "replan_count", "loop_events", "success_criteria")
_STEP_FIELDS = ("id", "plan_id", "index", "title", "tool_ref", "inputs", "depends_on", "post_checks", "needs_confirmation", "policy", "decision_point", "sub_agent_scope", "granted_tools", "status", "sub_agent_id", "output_ref", "review_verdicts", "error", "phase")


def _private(ctx):
    return decode_native(_json(ctx.archive, "private.json"))


def _action_output(ctx, index):
    action = ctx.binding.case.actions[index - 1]
    matches = [item for item in _private(ctx).get("extension_outputs", ()) if item.get("case_action_index") == index]
    if len(matches) != 1 or matches[0].get("action_sha256") != digest(action.model_dump()):
        raise _Unsupported("original indexed request/response capture unavailable")
    return matches[0]


def _http(ctx, output, *, statuses):
    bindings = _json(ctx.archive, "bindings.json")
    events = bindings["execution_before_scoring"].get("http_events", [])
    ordinal = output.get("http_ordinal")
    matches = [event for event in events if event.get("ordinal") == ordinal]
    if type(ordinal) is not int or ordinal < 1 or len(matches) != 1 or matches[0].get("status_code") not in statuses:
        raise _Unsupported("actual original HTTP response fact unavailable")
    kind = output.get("kind")
    if kind is None:
        if output.get("fault_kind") == "discard_success_response_once":
            stage = "reference_discard_response_fault"
        else:
            scope = "human_extension_" if output.get("actor") is not None else "human_"
            stage = scope + ("approval" if output["request"]["decision"] == "approve" else "rejection")
        method = "POST"
    else:
        stage = "human_extension_" + kind
        method = "GET" if kind in {"read_risk_source_evidence", "read_risk_event_evidence", "read_collection_effects"} else "POST"
    if matches[0].get("stage") != stage or matches[0].get("method") != method:
        raise ValueError("original HTTP method/action stage differs")
    return matches[0]


def _before(ctx):
    pin = ctx.binding.before_manifest_sha256
    if pin is None:
        raise _Unsupported("original before snapshot was not externally frozen")
    if not (ctx.archive / "safety/before/manifest.json").is_file() or not (ctx.archive / "safety/before/marvis.sqlite").is_file():
        raise _Unsupported("original before snapshot files were not retained")
    raw = _file(ctx.archive, "safety/before/manifest.json").read_bytes()
    equal(digest(raw), pin)
    manifest = json.loads(raw)
    equal(ctx.manifest.get("safety_before"), {"manifest_sha256": pin, "scope": "original_future_runtime_before_only"})
    database = _file(ctx.archive, "safety/before/marvis.sqlite", maximum=256 * 1024**2)
    equal(manifest["files"], {"marvis.sqlite": {"sha256": digest(database.read_bytes()), "size_bytes": database.stat().st_size}})
    if (manifest.get("schema") != "marvis.runtime-before-originals.v1" or manifest.get("case_sha256") != digest(ctx.binding.case.model_dump())
            or manifest.get("phase") != "after_application_startup_before_first_case_http_action" or manifest.get("first_case_http_ordinal") != 0
            or manifest.get("database_capture") != "sqlite_committed_logical_snapshot"):
        raise ValueError("original before capture phase/case differs")
    for key in ("capture_started_elapsed_ns", "capture_finished_elapsed_ns"):
        if type(manifest.get(key)) is not int or manifest[key] < 0:
            raise ValueError("original before capture timing differs")
    if manifest["capture_started_elapsed_ns"] > manifest["capture_finished_elapsed_ns"]:
        raise ValueError("original before capture order differs")
    events = _json(ctx.archive, "bindings.json")["execution_before_scoring"].get("http_events", [])
    if not events or any(type(event.get("started_elapsed_ns")) is not int or type(event.get("finished_elapsed_ns")) is not int for event in events):
        raise _Unsupported("original pre/post HTTP monotonic timeline unavailable")
    if manifest["capture_finished_elapsed_ns"] > min(event["started_elapsed_ns"] for event in events) or any(event["started_elapsed_ns"] > event["finished_elapsed_ns"] for event in events):
        raise ValueError("original before snapshot is not before all HTTP actions")
    conn = ctx.stack.enter_context(closing(sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True)))
    conn.row_factory = sqlite3.Row
    return conn


def _principal(ctx, identity, when, role=None):
    row = ctx.conn.execute("SELECT * FROM local_principals WHERE id=?", (identity,)).fetchone()
    if row is None or at(row["created_at"]) > when or at(row["expires_at"]) <= when or (row["revoked_at"] and at(row["revoked_at"]) <= when):
        raise ValueError("native principal unavailable at decision time")
    if role:
        production = ctx.conn.execute("SELECT * FROM production_principals WHERE local_principal_id=?", (identity,)).fetchone()
        if production is None or production["role"] != role:
            raise ValueError("native human role differs")


def _institution(ctx, decision, when):
    settings = ctx.conn.execute("SELECT * FROM institution_settings WHERE singleton=1").fetchone()
    if settings is None:
        raise _Unsupported("native responsibility configuration unavailable")
    if settings["mode"] == "trusted_local":
        return "trusted_local_reference; institution_authority_not_established"
    actor = ctx.conn.execute("SELECT * FROM institution_subjects WHERE id=?", (decision["principal_id"],)).fetchone()
    owners = [ctx.conn.execute("SELECT * FROM institution_resource_owners WHERE kind=? AND resource_id=?", (kind, identity)).fetchone() for kind, identity in (("plan", decision["plan_id"]), ("task", ctx.binding.task_id))]
    if actor is None or not any(owners) or actor["actor_kind"] != "human" or actor["created_at"] > when.timestamp() or (actor["revoked_at"] is not None and actor["revoked_at"] <= when.timestamp()):
        raise ValueError("native institution human/responsibility missing")
    if not any(owner and owner["owner_id"] == actor["id"] for owner in owners):
        grants = [ctx.conn.execute("SELECT * FROM institution_resource_grants WHERE kind=? AND resource_id=? AND subject_id=?", (kind, identity, actor["id"])).fetchone() for kind, identity in (("plan", decision["plan_id"]), ("task", ctx.binding.task_id))]
        if not any(grant and "approve" in json.loads(grant["actions_json"]) and (grant["revoked_at"] is None or grant["revoked_at"] > when.timestamp()) and (grant["expires_at"] is None or grant["expires_at"] > when.timestamp()) for grant in grants):
            raise ValueError("native institution exact plan grant missing")
        # Mutable grant rows have no original event-time historical receipt.
        raise _Unsupported("institutional mutable grant event-time original authority unavailable")
    executor = ctx.conn.execute("SELECT * FROM institution_plan_actors WHERE plan_id=?", (decision["plan_id"],)).fetchone()
    if executor is None or executor["task_id"] != ctx.binding.task_id:
        raise ValueError("native plan execution responsibility missing")
    raise _Unsupported("institution signature/session authority is not exported with archive")


def _original_cas(before, step_id, request):
    steps = [row for row in before["steps"] if row["id"] == step_id]
    if len(steps) != 1:
        raise ValueError("native original governed step differs")
    step = steps[0]
    canonical_plan = {key: before[key] for key in _PLAN_FIELDS}
    canonical_plan["steps"] = [{key: row[key] for key in _STEP_FIELDS} for row in before["steps"]]
    canonical_step = {key: step[key] for key in _STEP_FIELDS}
    for key, expected in (("expected_plan_fingerprint", digest(canonical_plan)), ("expected_step_fingerprint", digest({**canonical_step, "confirmed": False})), ("expected_plan_status", before["status"]), ("expected_plan_revision", before["replan_count"])):
        equal(request[key], expected)
    return step


def _governed(ctx, item):
    action = ctx.binding.case.actions[item.action_index - 1]
    private = _private(ctx)
    captures = [row for key in ("extension_decisions", "human_decisions") for row in private.get(key, ()) if row.get("case_action_index") == item.action_index]
    if len(captures) != 1 or captures[0].get("action_sha256") != digest(action.model_dump()):
        raise _Unsupported("original human decision/CAS capture unavailable")
    capture = captures[0]
    _http(ctx, capture, statuses={202})
    decision = ctx.row("SELECT * FROM decision_records WHERE id=? AND task_id=?", (item.decision_id, ctx.binding.task_id), item.decision_record_sha256)
    before, after = capture.get("plan_before"), capture.get("plan_after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise _Unsupported("original decision before/after plans unavailable")
    if before["id"] != item.plan_id or after["id"] != item.plan_id or before["task_id"] != ctx.binding.task_id:
        raise ValueError("native original plan identity differs")
    request = capture["request"]
    # The public snapshot commits the original one-shot confirmed flag through
    # its complete fingerprint. The required preimage is false at this gate;
    # a true flag changes the hash and is rejected, never inferred from post.
    step = _original_cas(before, item.step_id, request)
    expected_decision = "approve" if action.kind == "approve_step" else "reject"
    plugin = ctx.conn.execute("SELECT * FROM plugins WHERE name=?", (action.tool.split(".", 1)[0],)).fetchone()
    if plugin is None or not plugin["builtin"]:
        raise _Unsupported("safety reader only supports exact installed native builtin tools")
    manifest = json.loads(plugin["manifest_json"])
    manifest_commitment = plugin["checksum"] or digest(manifest)
    equal(decision["manifest_hash"], manifest_commitment if manifest_commitment.startswith("sha256:") else "sha256:" + manifest_commitment)
    equal(decision["tool_ref"], action.tool + "@" + plugin["version"])
    tools = [tool for tool in manifest["tools"] if tool["name"] == action.tool.split(".", 1)[1]]
    if len(tools) != 1:
        raise ValueError("native installed policy missing")
    installed = tools[0]["policy"]
    stricter = {**installed, "human_decision_gate": "required"}
    if digest(step["policy"]) not in {digest(installed), digest(stricter)}:
        raise _Unsupported("safety proof supports exact native policy or its explicit additional human gate only")
    equal(decision["policy_hash"], "sha256:" + digest(step["policy"]))
    dependencies = []
    for identity in sorted(step["depends_on"]):
        members = [row for row in before["steps"] if row["id"] == identity]
        if len(members) != 1 or not members[0].get("output_ref"):
            raise _Unsupported("original dependency output-ref unavailable")
        reference = members[0]["output_ref"]
        prefix = "metrics:" + identity + ":v"
        if not reference.startswith(prefix) or not reference[len(prefix):].isdigit():
            raise ValueError("original dependency version differs")
        rows = ctx.conn.execute("SELECT evidence_json FROM plan_step_output_versions WHERE step_id=? AND version=?", (identity, int(reference[len(prefix):]))).fetchmany(2)
        if len(rows) != 1:
            raise ValueError("original dependency evidence missing")
        dependencies.append({"step_id": identity, "output_ref": reference, "evidence_hash": governance_input_reference(json.loads(rows[0]["evidence_json"]))})
    equal(decision["evidence_hash"], governance_input_reference(dependencies))
    resolved_inputs = resolved_original_inputs(ctx, before, step)
    if (decision["plan_id"] != item.plan_id or decision["step_id"] != item.step_id or decision["decision"] != expected_decision
            or decision["reason"] != action.content or request["reason"] != action.content or request["decision"] != expected_decision
            or not decision["tool_ref"].startswith(action.tool + "@") or decision["plan_revision"] != before["replan_count"]
            or step["status"] != "awaiting_confirm" or not step["needs_confirmation"]
            or decision["input_hash"] != governance_input_reference(resolved_inputs)):
        raise ValueError("native exact human action/CAS/input commitment differs")
    response = capture["response"]
    if response["decision_id"] != item.decision_id or response["principal_id"] != decision["principal_id"]:
        raise ValueError("native decision response identity differs")
    actor = capture["actor"]
    if actor is not None and getattr(ctx, "native_actors", {}).get(actor) != decision["principal_id"]:
        raise ValueError("native original actor/session responsibility differs")
    when = at(decision["created_at"])
    _principal(ctx, decision["principal_id"], when, actor)
    responsibility = _institution(ctx, decision, when)
    if action.tool == "risk_context.replay_events":
        intent = ctx.conn.execute("SELECT * FROM event_requests WHERE task_id=? AND request_id=?", (ctx.binding.task_id, step["inputs"]["request_id"])).fetchone()
        if intent is None:
            raise ValueError("native event exact proposal absent")
        body = json.loads(intent["payload"])
        if body["actor_id"] != decision["principal_id"] or digest(body) != step["inputs"]["proposal_hash"]:
            raise ValueError("native event proposal/grantee differs")
        grant = ctx.conn.execute("SELECT * FROM event_grants WHERE task_id=? AND id=?", (ctx.binding.task_id, body["request"]["grant_id"])).fetchone()
        if grant is None:
            raise ValueError("native event access grant absent")
        access = json.loads(grant["payload"])["grant"]
        if (access["grantee_id"] != decision["principal_id"] or access["source_id"] != body["request"]["contract"]["source_id"]
                or access["source_contract_hash"] != body["request"]["contract"]["source_contract_hash"] or "write" not in access["permissions"]
                or at(access["starts_at"]) > when or at(access["expires_at"]) <= when or (grant["revoked_at"] and at(grant["revoked_at"]) <= when)):
            raise ValueError("native exact event permission/time differs")
    elif action.tool.startswith("collection."):
        owner = ctx.conn.execute("SELECT * FROM collection_batches WHERE task_id=? AND batch_id=?", (ctx.binding.task_id, step["inputs"]["batch_id"])).fetchone()
        if owner is None or ctx.native_actors.get("maker") != owner["actor_id"] or actor != "checker":
            raise ValueError("native maker/checker collection responsibility differs")
    elif actor is None and action.tool in {"data_ops.infer_schema", "data_ops.profile_dataset", "data_ops.execute_join", "feature.analyze_feature_bins", "modeling.select_experiment", "strategy.create_strategy", "strategy.backtest_strategy"}:
        if step["policy"].get("effect_authorization") != "none":
            raise _Unsupported("native core effect authorization requires a dedicated exact business effect archive")
    else:
        raise _Unsupported("governed safety exact installed tool scope unsupported")
    native_step = ctx.conn.execute("SELECT * FROM plan_steps WHERE id=? AND plan_id=?", (item.step_id, item.plan_id)).fetchone()
    native_plan = ctx.conn.execute("SELECT * FROM plans WHERE id=? AND task_id=?", (item.plan_id, ctx.binding.task_id)).fetchone()
    after_steps = [row for row in after["steps"] if row["id"] == item.step_id]
    if native_plan is None or len(after_steps) != 1 or native_step is None:
        raise ValueError("original captured after plan/native state differs")
    # GET plans precedes GET job in the original polling protocol. The last
    # step can already have committed its immutable output while its plan is
    # still running or completing its review; job completion commits done. This
    # single transition supplies no completion proof by itself: the original
    # step/output version, successful run and full native receipt close below.
    plan_progress = (expected_decision == "approve" and after["status"] in {"running", "review"}
                     and native_plan["status"] == "done" and after_steps[0]["status"] == "done"
                     and native_step["status"] == "done" and bool(native_step["output_ref"]))
    if ((after["status"] != native_plan["status"] and not plan_progress)
            or after.get("task_id") != ctx.binding.task_id
            or after["replan_count"] != native_plan["replan_count"]
            or after_steps[0]["status"] != native_step["status"]
            or after_steps[0].get("output_ref") != native_step["output_ref"]):
        raise ValueError("original captured after plan/native state differs")
    runs = ctx.conn.execute("SELECT * FROM plan_step_runs WHERE step_id=?", (item.step_id,)).fetchmany(101)
    if len(runs) > 100 or native_step is None:
        raise ValueError("native governed run closure differs")
    if expected_decision == "reject":
        if runs or native_step["status"] not in {"skipped", "cancelled", "error"} or native_step["error"] != "human_rejected:" + item.decision_id:
            raise ValueError("native rejected action executed or closure differs")
    else:
        if len(runs) != 1 or runs[0]["status"] != "succeeded" or native_step["status"] != "done":
            raise _Unsupported("approved action requires one actual successful original invocation")
        if governance_input_reference(json.loads(runs[0]["input_json"])) != decision["input_hash"]:
            raise ValueError("native approved invocation input differs")
        if decision["tool_ref"] != action.tool + "@" + runs[0]["tool_version"]:
            raise ValueError("native approved installed tool/version/manifest differs")
        receipt_commitment = plugin["checksum"] or hashlib.sha256(json.dumps(manifest, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()
        equal(runs[0]["manifest_hash"], receipt_commitment if receipt_commitment.startswith("sha256:") else "sha256:" + receipt_commitment)
        native_completed_output(ctx, item.plan_id, item.step_id, private)
        if action.tool.startswith("collection."):
            approvals = ctx.conn.execute("SELECT * FROM approval_records WHERE decision_id=?", (item.decision_id,)).fetchmany(2)
            if len(approvals) != 1:
                raise ValueError("native exact approval absent")
            approval = approvals[0]
            effect = ctx.conn.execute("SELECT * FROM effect_executions WHERE approval_id=?", (approval["id"],)).fetchmany(2)
            if len(effect) != 1 or approval["status"] != "consumed" or effect[0]["status"] != "committed" or effect[0]["invocation_id"] != runs[0]["id"] or effect[0]["reservation_id"] != approval["reservation_id"]:
                raise ValueError("native one-shot approval/effect closure differs")
            for field in ("principal_id", "task_id", "plan_id", "plan_revision", "step_id", "tool_ref", "manifest_hash", "policy_hash", "input_hash", "evidence_hash", "effect_target_json"):
                equal(approval[field], decision[field])
            native = ctx.conn.execute("SELECT * FROM collection_native_effects WHERE effect_id=? AND task_id=?", (effect[0]["id"], ctx.binding.task_id)).fetchone()
            if native is None:
                raise ValueError("native committed collection target receipt missing")
            native_body = json.loads(native["payload_json"])
            facts = native_body["facts"]
            equal(facts["target"], json.loads(decision["effect_target_json"]))
            if facts["operation"] != action.tool.split(".", 1)[1] or facts["bindings"]["decision_id"] != decision["id"]:
                raise ValueError("native committed exact collection operation differs")
            from .runtime_archive_collections import _governed_effect_closure
            _governed_effect_closure(ctx, facts, native_body["output"])
    return {"action_index": item.action_index, "decision": expected_decision, "tool": action.tool, "responsibility_scope": responsibility,
            "proof_scope": "exact_native_authorization_and_completed_output; not_external_business_effect" if actor is None else "exact_native_role_authorization_and_supported_effect"}


def _duplicate(ctx, before, item):
    outputs = [_action_output(ctx, index) for index in item.action_indices]
    for output in outputs:
        _http(ctx, output, statuses={200})
        if output["http_status"] != 200:
            raise ValueError("duplicate request did not return original successful response")
    if item.kind == "reference_decision":
        requests = [output["request"]["request"] for output in outputs]
        if any(output["request"]["slot"] != item.slot for output in outputs):
            raise ValueError("duplicate deployment slot differs")
        equal(requests, [requests[0]] * len(requests))
        key = ("local-reference:" + item.slot, item.request_id)
        row = ctx.row("SELECT * FROM reference_decisions WHERE environment=? AND request_id=?", key, item.record_sha256)
        request = requests[0]
        if request.get("event_evidence") is None:
            request = {key: value for key, value in request.items() if key != "event_evidence"}
        if row["input_hash"] != digest(request) or row["status"] != "completed" or row["package_hash"] != request["expected_package_hash"] or request["request_id"] != item.request_id:
            raise ValueError("duplicate complete native input/package differs")
        old = before.execute("SELECT * FROM reference_decisions WHERE environment=? AND request_id=?", key).fetchone()
        if old is not None and dict(old) != row:
            raise ValueError("duplicate existing durable original changed")
        response = json.loads(row["response_json"])
        for output in outputs:
            equal(output["response"], response)
        return {"request_id": item.request_id, "original_action_count": len(outputs), "durable_completed_rows": 1, "scope": "local_read_only_reference_decision; external_effects_not_claimed"}
    row = ctx.row("SELECT * FROM source_requests WHERE task_id=? AND id=?", (ctx.binding.task_id, item.request_id), item.record_sha256)
    contract = json.loads(row["contract"])
    if row["state"] != "completed" or contract["profile"]["mode"] != "historical_file" or contract["query"].get("historical") is None or row["contract_hash"] != digest(contract):
        raise _Unsupported("duplicate source proof only supports native registered historical queries")
    if any(output["request"]["request_id"] != item.request_id or ctx.native_actors.get(output["actor"]) != row["actor_id"] for output in outputs):
        raise ValueError("duplicate native source request differs")
    attempts = [dict(r) for r in ctx.conn.execute("SELECT * FROM source_attempts WHERE task_id=? AND request_id=? ORDER BY id", (ctx.binding.task_id, item.request_id)).fetchmany(101)]
    equal(digest(attempts), item.attempts_sha256)
    old_attempts = before.execute("SELECT COUNT(*) FROM source_attempts WHERE task_id=? AND request_id=?", (ctx.binding.task_id, item.request_id)).fetchone()[0]
    if len(attempts) > 100 or len(attempts) - old_attempts != 1:
        raise ValueError("duplicate historical source has more than one original attempt")
    if before.execute("SELECT 1 FROM source_requests WHERE task_id=? AND id=?", (ctx.binding.task_id, item.request_id)).fetchone() is not None:
        raise _Unsupported("historical duplicate query before-existing request needs original per-attempt state")
    for output in outputs:
        equal(output["response"], outputs[0]["response"])
    return {"request_id": item.request_id, "original_action_count": len(outputs), "new_native_attempts": 1, "scope": "local_historical_read; network_provider_calls_not_claimed"}


def _revalidate(archive, sha, binding, family, check, work):
    binding = FrozenSafetyBinding.model_validate(binding.model_dump())
    ctx = ExtensionArchive(archive, sha, binding, family, (check,))
    with ctx:
        try:
            ctx.open()
            ctx.current = check
            values = work(ctx)
            ctx.result["values"] = values
            ctx.mark(check, scope="independent_original_local_native_consistency")
            ctx.mark("native_records", scope="actual_native_rows_and_original_request_response_closure")
        except _Unsupported as exc:
            ctx.result["unsupported"][ctx.current] = str(exc)
            raise
    for item in ctx.result["checks"].values():
        if item["status"] == "unsupported":
            item["status"] = "unavailable"
    ctx.result["status"] = "verified" if ctx.result["supported_checks_verified"] else "mismatch" if any(c["status"] == "mismatch" for c in ctx.result["checks"].values()) else "unavailable"
    return ctx.result


def revalidate_authorization_archive(archive: Path, *, expected_manifest_sha256: str, frozen_binding: FrozenSafetyBinding):
    def work(ctx):
        if not ctx.binding.governed_actions:
            raise _Unsupported("original governed action bindings unavailable")
        return {"governed_actions": [_governed(ctx, item) for item in ctx.binding.governed_actions]}
    return _revalidate(archive, expected_manifest_sha256, frozen_binding, "authorization", "exact_original_authorization", work)


def revalidate_point_in_time_archive(archive: Path, *, expected_manifest_sha256: str, frozen_binding: FrozenSafetyBinding):
    def work(ctx):
        if not ctx.binding.event_times:
            raise _Unsupported("original event time bindings unavailable; other model/strategy time scopes are unsupported")
        results = []
        for item in ctx.binding.event_times:
            output = _action_output(ctx, item.action_index)
            _http(ctx, output, statuses={201})
            equal(output["request"]["request"], item.event.request.model_dump())
            result = _event(ctx, item.event)
            result["time_scope"] = "native_ingestion_cutoff_version_window; publisher_availability_and_label_maturity_not_established"
            results.append(result)
        return {"events": results}
    return _revalidate(archive, expected_manifest_sha256, frozen_binding, "point-in-time", "native_ingestion_cutoff_and_versions", work)


def revalidate_duplicate_effects_archive(archive: Path, *, expected_manifest_sha256: str, frozen_binding: FrozenSafetyBinding):
    def work(ctx):
        if not ctx.binding.duplicate_operations:
            raise _Unsupported("original duplicate operation bindings unavailable")
        before = _before(ctx)
        return {"operations": [_duplicate(ctx, before, item) for item in ctx.binding.duplicate_operations]}
    return _revalidate(archive, expected_manifest_sha256, frozen_binding, "duplicates", "original_before_after_local_durable_operations", work)


def revalidate_boundary_archive(archive: Path, *, expected_manifest_sha256: str, frozen_binding: FrozenSafetyBinding):
    def work(ctx):
        if not ctx.binding.boundary_action_indices:
            raise _Unsupported("original boundary actions unavailable")
        _before(ctx)
        refusals = []
        for index in ctx.binding.boundary_action_indices:
            action = ctx.binding.case.actions[index - 1]
            if getattr(getattr(action, "fault", None), "kind", None) == "stale_step_cas_once":
                fault, before, after, event = _fault_original(ctx, index, "stale_step_cas_once")
                expected = {**fault["original_request"], "expected_step_fingerprint": "0" * 64}
                equal(fault["dispatched_request"], expected)
                if fault["original_request"]["expected_step_fingerprint"] == "0" * 64:
                    raise ValueError("declared stale snapshot was already zero")
                equal(before["target"], after["target"])
                equal(before["tables"], after["tables"])
                if any(before["tables"].values()):
                    raise _Unsupported("stale proof requires a fresh unexecuted exact native target")
                # The final archive may contain a later *declared* approval.
                final = archived_target(ctx, fault["plan_id"], fault["step_id"])
                for key in ("id", "plan_id", "task_id", "tool_plugin", "tool_name", "input_sha256", "policy_sha256"):
                    equal(final["target"][key], before["target"][key])
                if final["tables"]["decision_records"]:
                    originals = [row for key in ("human_decisions", "extension_decisions") for row in _private(ctx).get(key, ()) if row.get("case_action_index", 0) > index and row.get("response", {}).get("step_id") == fault["step_id"]]
                    equal(sorted(row["response"]["decision_id"] for row in originals), sorted(row["id"] for row in final["tables"]["decision_records"]))
                else:
                    equal(final, {"target": after["target"], "tables": after["tables"]})
                refusals.append({"action_index": index, "http_status": event["status_code"], "native_target_counter_delta": {key: 0 for key in before["tables"]},
                                 "scope": "single_actual_stale_CAS_refusal_and_zero_native_target_effect_at_boundary; not_external_institution_effect"})
                continue
            output = _action_output(ctx, index)
            event = _http(ctx, output, statuses={400, 401, 403, 404, 409, 422})
            if output["response"] != {"status": "http_rejected", "http_status": event["status_code"]}:
                raise ValueError("native boundary refusal fact differs")
            refusals.append({"action_index": index, "http_status": event["status_code"]})
            ctx.result["values"] = {"observed_http_refusals": refusals}
            raise _Unsupported("observed refusal alone cannot establish no external effect; exact original target/fault carrier required")
        return {"boundaries": refusals, "fault_scope": "future_declared_local_client_reference_only"}
    return _revalidate(archive, expected_manifest_sha256, frozen_binding, "boundary", "original_boundary_target_effect", work)


def revalidate_recovery_archive(archive: Path, *, expected_manifest_sha256: str, frozen_binding: FrozenSafetyBinding):
    def work(ctx):
        _before(ctx)
        if not ctx.binding.recovery_action_indices:
            raise _Unsupported("original declared fault/readback action binding unavailable")
        recovered = []
        for index in ctx.binding.recovery_action_indices:
            action = ctx.binding.case.actions[index - 1]
            source = action.fault_source_action
            fault, before, after, event = _fault_original(ctx, source, "discard_success_response_once")
            equal(fault["dispatched_request"], fault["original_request"])
            equal(fault["visible_response"], "discarded_by_declared_client_reference_fault")
            recovery = fault.get("recovery")
            if not isinstance(recovery, dict):
                raise _Unsupported("actual fixed native recovery readback originals unavailable")
            equal(recovery["action_index"], index)
            equal(recovery["action_sha256"], digest(action.model_dump()))
            equal(recovery["transport"], "existing_fixed_get_plan_readback_only")
            pre = original_target(recovery.get("before_readback"), task=ctx.binding.task_id, plan=fault["plan_id"], step=fault["step_id"])
            post = original_target(recovery.get("after_readback"), task=ctx.binding.task_id, plan=fault["plan_id"], step=fault["step_id"])
            events = _json(ctx.archive, "bindings.json")["execution_before_scoring"]["http_events"]
            ordinals = recovery.get("http_ordinals")
            if (not isinstance(ordinals, list) or not ordinals or any(type(n) is not int for n in ordinals)
                    or ordinals != list(range(ordinals[0], ordinals[-1] + 1)) or ordinals[0] <= fault["http_ordinal"]):
                raise ValueError("original recovery readback ordinal sequence differs")
            requests = [row for row in events if row.get("ordinal") in ordinals]
            if len(requests) != len(ordinals) or any(row.get("method") != "GET" or row.get("stage") not in {"read_plans", "read_job"} or row.get("status_code") != 200 for row in requests):
                raise ValueError("recovery repeated a write or lacks actual successful fixed readback")
            if after["finished_elapsed_ns"] > pre["started_elapsed_ns"] or pre["finished_elapsed_ns"] > requests[0]["started_elapsed_ns"] or requests[-1]["finished_elapsed_ns"] > post["started_elapsed_ns"]:
                raise ValueError("original fault/recovery native timeline differs")
            if any(before["tables"].values()):
                raise _Unsupported("recovery counter proof requires a fresh original unexecuted native target")
            final = archived_target(ctx, fault["plan_id"], fault["step_id"])
            equal(final, {"target": post["target"], "tables": post["tables"]})
            for snapshot in (after, pre):
                for key in ("id", "plan_id", "task_id", "tool_plugin", "tool_name", "input_sha256", "policy_sha256"):
                    equal(snapshot["target"][key], post["target"][key])
                original_progress_to_completed(snapshot, post)
            table = post["tables"]
            if len(table["decision_records"]) != 1 or len(table["plan_step_runs"]) != 1 or len(table["plan_step_output_versions"]) != 1 or len(table["approval_records"]) > 1 or len(table["effect_executions"]) > 1:
                raise ValueError("native recovery executed more than one exact target operation")
            declared = [item for item in ctx.binding.governed_actions if item.action_index == source]
            if len(declared) != 1:
                raise _Unsupported("original recovered decision record must be independently frozen")
            governed = _governed(ctx, declared[0])
            equal(fault["received_response"]["decision_id"], declared[0].decision_id)
            output = native_completed_output(ctx, fault["plan_id"], fault["step_id"], _private(ctx))
            recovered.append({"action_index": index, "fault_action_index": source, "governed_action": governed, **output,
                              "native_successful_invocations": 1, "recovery_writes": 0,
                              "scope": "actual_client_response_discard_then_fixed_GET_readback; not_institution_network_failure"})
        return {"recoveries": recovered}
    return _revalidate(archive, expected_manifest_sha256, frozen_binding, "recovery", "original_fault_recovery", work)


def _fault_original(ctx, index, kind):
    action = ctx.binding.case.actions[index - 1]
    if action.kind != "approve_step" or getattr(action.fault, "kind", None) != kind:
        raise _Unsupported("original fixed reference fault declaration unavailable")
    rows = [row for row in _private(ctx).get("runtime_faults", ()) if row.get("action_index") == index]
    if len(rows) != 1:
        raise _Unsupported("original actually injected fault carrier unavailable")
    fault = rows[0]
    equal(fault["schema"], "marvis.runtime-reference-fault.v1")
    equal(fault["action_sha256"], digest(action.model_dump()))
    equal((fault["task_id"], fault["tool"], fault["kind"]), (ctx.binding.task_id, action.tool, kind))
    equal(fault["scope"], "local_client_reference_fault; not_institution_network_fault_proof")
    plan = fault.get("plan_before")
    if not isinstance(plan, dict):
        raise _Unsupported("original fault plan/CAS preimage unavailable")
    equal((plan["task_id"], plan["id"]), (ctx.binding.task_id, fault["plan_id"]))
    body = fault["original_request"]
    equal(sorted(body), sorted(("decision", "reason", "expected_plan_fingerprint", "expected_step_fingerprint", "expected_plan_status", "expected_plan_revision")))
    equal((body["decision"], body["reason"]), ("approve", action.content))
    step = _original_cas(plan, fault["step_id"], body)
    equal(step["tool_ref"]["plugin"] + "." + step["tool_ref"]["tool"], action.tool)
    before = original_target(fault.get("before"), task=ctx.binding.task_id, plan=fault["plan_id"], step=fault["step_id"])
    after = original_target(fault.get("after_response"), task=ctx.binding.task_id, plan=fault["plan_id"], step=fault["step_id"])
    target = before["target"]
    equal((target["status"], target["confirmed"], target["needs_confirmation"], target["output_ref"]), ("awaiting_confirm", 0, 1, None))
    equal((target["input_sha256"], target["policy_sha256"]), (native_json_sha(step["inputs"]), native_json_sha(step["policy"])))
    equal((target["plan_status"], target["replan_count"]), (plan["status"], plan["replan_count"]))
    events = _json(ctx.archive, "bindings.json")["execution_before_scoring"]["http_events"]
    matches = [row for row in events if row.get("ordinal") == fault.get("http_ordinal")]
    if len(matches) != 1:
        raise _Unsupported("original injected fault HTTP fact unavailable")
    event = matches[0]
    expected = 409 if kind == "stale_step_cas_once" else 202
    label = "reference_stale_cas_fault" if kind == "stale_step_cas_once" else "reference_discard_response_fault"
    equal((event["method"], event["stage"], event["status_code"], fault["http_status"]), ("POST", label, expected, expected))
    if before["finished_elapsed_ns"] > event["started_elapsed_ns"] or event["finished_elapsed_ns"] > after["started_elapsed_ns"]:
        raise ValueError("fault native pre/post originals do not surround actual HTTP")
    return fault, before, after, event
