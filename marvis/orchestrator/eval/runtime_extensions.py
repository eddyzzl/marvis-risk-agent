"""Fixed public extension journeys with separate server sessions for each role."""

from copy import copy
from contextlib import contextmanager
from http.cookiejar import CookieJar
from pathlib import Path
from urllib.parse import quote
import json

from pydantic import TypeAdapter

from .runtime_extension_inputs import ExtensionBusinessInput


# Exact semantic pairs: an arbitrary earlier `id` is never a promotion, a
# principal, or authorization basis merely because its JSON type is a string.
_REFERENCES = {
    ("request_production_promotion", "id"): {"promotion_id", "request.promotion_id"},
    ("activate_production_promotion", "id"): {"deployment_id", "request.expected_active_deployment_id"},
    ("install_decision_package", "activation_evidence.receipt_id"): {"request.receipt_id"},
    ("build_decision_package", "package_hash"): {"request.decision_package_hash", "request.package_hash", "request.expected_package_hash"},
    ("claim_extension_role", "id"): {"request.grantee_id"},
    ("declare_source_authorization_basis", "id"): {"request.basis_artifact_id"},
    ("register_risk_event_source", "source_id"): {"request.source_id", "request.dataset.source_id", "request.contract.source_id"},
    ("register_risk_event_source", "contract_hash"): {"request.source_contract_hash", "request.dataset.source_contract_hash", "request.contract.source_contract_hash"},
    ("import_risk_event_dataset", "ingested_at"): {"request.contract.knowledge_cutoff"},
    ("grant_risk_source", "grant_id"): {"request.grant_id", "grant_id"},
    ("grant_risk_event_source", "grant_id"): {"request.grant_id", "grant_id"},
    ("prepare_risk_source_query", "request_id"): {"request_id"},
    ("prepare_risk_event_features", "request_id"): {"request_id"},
    ("prepare_collection_batch", "batch_id"): {"batch_id"},
    ("publish_operations_schedule", "contract.schedule_id"): {"schedule_id"},
}


def _action_hash(action):
    from .runtime_contracts import digest
    return digest(action.model_dump())


for _field, _suffix in (("source_artifact_id", "id"), ("source_artifact_hash", "hash")):
    _REFERENCES[("declare_collection_material", _field)] = {
        f"request.source_artifact_{_suffix}", f"request.terms_artifact_{_suffix}",
        f"request.policy.basis_artifact_{_suffix}",
        *(f"request.{part}[].source_artifact_{_suffix}" for part in ("cases", "histories", "events", "coverage")),
    }


def _assign(target, parts, actual):
    key = parts[0]
    if key.endswith("[]"):
        items = target[key[:-2]]
        if not isinstance(items, list) or not items:
            raise ValueError("extension_identity_target_invalid")
        for item in items:
            _assign(item, parts[1:], actual)
    elif key not in target:
        raise ValueError("extension_identity_target_invalid")
    elif len(parts) == 1:
        target[key] = actual
    else:
        if not isinstance(target[key], dict):
            raise ValueError("extension_identity_target_invalid")
        _assign(target[key], parts[1:], actual)


def _cookies(original):
    result = CookieJar()
    for cookie in getattr(original, "jar", original):
        result.set_cookie(copy(cookie))
    return result


@contextmanager
def actor_session(journey, actor):
    sessions = getattr(journey, "extension_sessions", None)
    if sessions is None or actor not in sessions:
        raise ValueError("extension_actor_not_claimed")
    original = _cookies(journey.client.cookies)
    journey.client.cookies = _cookies(sessions[actor])
    try:
        yield
    finally:
        sessions[actor] = _cookies(journey.client.cookies)
        journey.client.cookies = original


def bound_input(action, task_id, previous, *, materials=(), native_outputs=None):
    """Resolve only fixed identities from the same journey's earlier responses."""
    value = action.extension_request.model_dump()
    for binding in action.extension_bindings:
        if binding.source_field == "resolved_policy_hash":
            if (value["kind"] != "prepare_collection_batch" or binding.target not in {
                    "request.strategy.default_action.value.policy_hash", "request.strategy.rules[].action.value.policy_hash"}):
                raise ValueError("extension_declared_policy_hash_reference_invalid")
            from marvis.collection.actions import CollectionPolicy
            actual = CollectionPolicy.model_validate(value["request"]["policy"]).content_hash
        elif binding.source_material is not None:
            targets = {"dataset_id": {"request.dataset.dataset_id", "request.historical.dataset_id"},
                       "dataset_content_hash": {"request.dataset.expected_content_hash", "request.historical.expected_content_hash"}}
            if binding.source_material > len(materials) or binding.target not in targets.get(binding.source_field, set()):
                raise ValueError("extension_material_reference_invalid")
            material = materials[binding.source_material - 1]
            if len(material["datasets"]) != 1:
                raise ValueError("extension_material_source_not_unique")
            actual = material["datasets"][0][{"dataset_id": "id", "dataset_content_hash": "content_hash"}[binding.source_field]]
        elif binding.source_tool is not None:
            if (binding.target, binding.source_field) not in {("request.strategy_id", "strategy_id"), ("request.strategy_version", "version")}:
                raise ValueError("extension_native_reference_invalid")
            actual = (native_outputs or {})[binding.source_tool][binding.source_field]
        elif binding.source_action is None:
            if binding.target != "request.task_id" or binding.source_field != "task_id":
                raise ValueError("extension_task_reference_invalid")
            actual = task_id
        else:
            if binding.source_action > len(previous):
                raise ValueError("extension_forward_reference_forbidden")
            original = previous[binding.source_action - 1]
            if binding.target not in _REFERENCES.get((original["kind"], binding.source_field), set()):
                raise ValueError("extension_identity_reference_invalid")
            if original["kind"] == "claim_extension_role" and original.get("actor") != "maker":
                raise ValueError("extension_grantee_must_be_claimed_maker")
            actual = original["response"]
            for part in binding.source_field.split("."):
                actual = actual[part]
        if not isinstance(actual, (str, int)) or isinstance(actual, bool):
            raise ValueError("extension_identity_response_invalid")
        _assign(value, binding.target.split("."), actual)
    # Validate through the same installed DTO after binding. No injected keys,
    # arbitrary route or authority material is permitted by the input schema.
    return TypeAdapter(ExtensionBusinessInput).validate_json(json.dumps(value, allow_nan=False))


def _native_outputs(journey, action):
    tools = {binding.source_tool for binding in action.extension_bindings if binding.source_tool is not None}
    outputs = {}
    sources = {}
    if tools:
        plans = journey.plans()
        for tool in tools:
            from .runtime_runner import _tool_name
            matches = [step for plan in plans if plan["task_id"] == journey.task_id
                       for step in plan["steps"] if step["status"] == "done"
                       and _tool_name(step["tool_ref"]) == tool and step.get("output_ref")]
            if len(matches) != 1:
                raise ValueError("extension_native_source_not_unique")
            step = matches[0]
            outputs[tool] = journey.json_request("GET", f"/api/step-outputs/{quote(step['id'], safe='')}", label="read_extension_native_identity")
            fields = {binding.source_field for binding in action.extension_bindings if binding.source_tool == tool}
            if not fields.issubset(outputs[tool]):
                from .runtime_runner import RuntimeJourneyError
                raise RuntimeJourneyError("extension_native_identity_field_not_returned_requires_explicit_declared_version")
            from .runtime_contracts import digest
            sources[tool] = {"step_id": step["id"], "plan_id": step["plan_id"], "output_ref": step["output_ref"],
                             "output_sha256": digest(outputs[tool]), "http_ordinal": journey.count}
    journey.current_extension_native_sources = sources
    return outputs


def _route(item, task_id):
    def esc(value):
        return quote(str(value), safe="")
    task = esc(task_id)
    fixed = {
        "claim_extension_role": "/api/production-governance/principals/claim",
        "build_decision_package": "/api/reference-decision/packages",
        "request_production_promotion": "/api/production-governance/promotion-requests",
        "install_decision_package": "/api/reference-decision/installations",
        "publish_operations_schedule": "/api/operations/schedules",
        "tick_operations": "/api/operations/tick",
        "register_historical_source_profile": "/api/risk-sources/profiles",
        "grant_risk_source": "/api/risk-sources/grants",
        "register_risk_event_source": "/api/risk-events/sources",
        "grant_risk_event_source": "/api/risk-events/grants",
    }
    if item.kind in fixed:
        return "POST", fixed[item.kind]
    if item.kind in {"submit_reference_decision", "replay_reference_decision"}:
        return "POST", f"/api/reference-decision/decisions?slot={item.slot}"
    if item.kind == "review_production_promotion":
        return "POST", f"/api/production-governance/promotion-requests/{esc(item.promotion_id)}/approvals"
    if item.kind == "activate_production_promotion":
        # Installed reference packages have one fixed native environment.
        return "POST", f"/api/production-governance/environments/local-reference/promotion-requests/{esc(item.promotion_id)}/activate"
    if item.kind == "rollback_production_deployment":
        return "POST", f"/api/production-governance/environments/{esc(item.environment)}/deployments/{esc(item.deployment_id)}/rollback"
    if item.kind == "recheck_operations_period":
        return "POST", f"/api/operations/schedules/{esc(item.schedule_id)}/rechecks"
    task_suffix = {
        "declare_source_authorization_basis": "risk-sources/authorization-bases",
        "prepare_risk_source_query": "risk-sources/requests",
        "import_risk_event_dataset": "risk-events/imports",
        "prepare_risk_event_features": "risk-events/requests",
        "declare_collection_material": "collection/materials",
        "create_collection_case": "collection/cases",
        "create_collection_schedule": "collection/schedules",
        "import_collection_cashflows": "collection/cashflows",
        "reconcile_collection_cashflows": "collection/reconciliations",
        "propose_collection_batch": "collection/batch-proposals",
        "prepare_collection_batch": "collection/batches",
    }
    if item.kind in task_suffix:
        return "POST", f"/api/tasks/{task}/{task_suffix[item.kind]}"
    if item.kind in {"execute_risk_source_query", "read_risk_source_evidence"}:
        execute = item.kind == "execute_risk_source_query"
        return ("POST" if execute else "GET"), f"/api/tasks/{task}/risk-sources/requests/{esc(item.request_id)}/{'execute' if execute else 'evidence'}"
    if item.kind == "read_risk_event_evidence":
        return "GET", f"/api/tasks/{task}/risk-events/requests/{esc(item.request_id)}/evidence?grant_id={esc(item.grant_id)}"
    if item.kind == "read_collection_effects":
        return "GET", f"/api/tasks/{task}/collection/batches/{esc(item.batch_id)}"
    raise ValueError("extension_action_not_installed")


def submit_extension(journey, action, workspace: Path):
    previous = getattr(journey, "extension_outputs", [])
    item = bound_input(action, journey.task_id, previous, materials=getattr(journey, "uploaded_materials", ()), native_outputs=_native_outputs(journey, action))
    method, path = _route(item, journey.task_id)
    sessions = getattr(journey, "extension_sessions", None)
    if sessions is None:
        sessions = {"maker": _cookies(journey.client.cookies), "checker": CookieJar(), "admin": CookieJar()}
        journey.extension_sessions = sessions
    original = _cookies(journey.client.cookies)
    journey.client.cookies = _cookies(sessions[item.actor])
    journey.record_human_action(action)
    try:
        kwargs = {}
        if method == "POST":
            body = getattr(item, "request", None)
            kwargs["json"] = body.model_dump(mode="json") if body is not None else {}
            if item.kind == "register_historical_source_profile":
                kwargs["json"] = {"profile": kwargs["json"]}
        if item.kind == "claim_extension_role":
            token_path = Path(workspace) / "plugin_admin_token"
            if token_path.is_symlink() or not token_path.is_file():
                raise ValueError("extension_local_admin_unavailable")
            kwargs["headers"] = {"x-marvis-governance-admin": token_path.read_text().strip()}
        reply = journey.request(method, path, label=f"human_extension_{item.kind}", **kwargs)
        if reply.status_code >= 400:
            # Preserve a real public boundary refusal without copying error
            # prose. Transport/server failures remain runtime errors.
            if reply.status_code not in {400, 401, 403, 404, 409, 422}:
                from .runtime_runner import RuntimeJourneyError
                raise RuntimeJourneyError(f"human_extension_{item.kind}_http_{reply.status_code}")
            response = {"status": "http_rejected", "http_status": reply.status_code}
        else:
            response = reply.json()
        previous.append({"kind": item.kind, "actor": item.actor, "request": item.model_dump(), "response": response,
                         "http_status": reply.status_code, "http_ordinal": journey.count,
                         "native_identity_sources": journey.current_extension_native_sources,
                         "case_action_index": getattr(journey, "current_action_index", None),
                         "action_sha256": _action_hash(action)})
        journey.extension_outputs = previous
    finally:
        sessions[item.actor] = _cookies(journey.client.cookies)
        journey.client.cookies = original


def start_extension_workflow(journey, action):
    """An explicit human start validates only the exact earlier native proposal."""
    previous = getattr(journey, "extension_outputs", ())
    if action.extension_source_action > len(previous):
        raise ValueError("extension_forward_reference_forbidden")
    source = previous[action.extension_source_action - 1]
    if source.get("http_status", 200) >= 400:
        raise ValueError("extension_workflow_requires_accepted_proposal")
    if action.tool == "risk_context.replay_events":
        if source["kind"] != "prepare_risk_event_features":
            raise ValueError("extension_workflow_proposal_kind_mismatch")
        proposal = source["response"]
        slots = {"event_request_id": proposal["request_id"], "event_proposal_hash": proposal["proposal_hash"], "event_contract": proposal["contract"]}
        goal, template, decision_actor = "事件窗口特征回放", "event_feature_replay", "maker"
        expected = {"request_id": slots["event_request_id"], "proposal_hash": slots["event_proposal_hash"], "contract": slots["event_contract"]}
    else:
        if source["kind"] != "prepare_collection_batch":
            raise ValueError("extension_workflow_proposal_kind_mismatch")
        proposal = source["response"]
        slots = {"collection_" + key: proposal[key] for key in ("batch_id", "request_hash", "preview_hash")}
        goals = {"collection.queue_batch": ("催收参考排队", "collection_queue_batch"),
                 "collection.execute_reference": ("催收参考执行", "collection_execute_reference"),
                 "collection.cancel_batch": ("撤销催收参考排队", "collection_cancel_batch")}
        goal, template = goals[action.tool]
        decision_actor = "checker"
        expected = {key: proposal[key] for key in ("batch_id", "request_hash", "preview_hash")}
    with actor_session(journey, "maker"):
        journey.record_human_action(action)
        plan = journey.json_request("POST", f"/api/tasks/{journey.task_id}/plans", label="human_extension_workflow_prepare", json={"goal": goal, "slots": slots})["plan"]
        from .runtime_runner import _tool_name
        if (plan["task_id"] != journey.task_id or plan["status"] != "validated" or plan["template_id"] != template
                or len(plan["steps"]) != 1 or _tool_name(plan["steps"][0]["tool_ref"]) != action.tool
                or plan["steps"][0]["inputs"] != expected):
            raise ValueError("extension_native_plan_binding_mismatch")
        task = journey.json_request("GET", f"/api/tasks/{journey.task_id}", label="read_extension_task_entry")
        if task["run_mode"] == "agent":
            journey.json_request("POST", f"/api/tasks/{journey.task_id}/agent/messages", label="human_extension_agent_plan_start",
                                 json={"content": action.content, "ui_action": "start_plan", "expected_plan_id": plan["id"],
                                       **plan["confirmation_snapshot"], "acceptance_mode": "manual_review"})
        else:
            confirmed = journey.json_request("POST", f"/api/plans/{plan['id']}/confirm", label="human_extension_plan_confirm", json=plan["confirmation_snapshot"])["plan"]
            journey.json_request("POST", f"/api/plans/{plan['id']}/run", label="human_extension_plan_start", json={"expected_plan_fingerprint": confirmed["confirmation_snapshot"]["expected_plan_fingerprint"]})
        journey.wait_idle()
    gates = getattr(journey, "extension_plan_gates", {})
    gates[plan["id"]] = {"tool": action.tool, "actor": decision_actor, "proposal_action": action.extension_source_action}
    journey.extension_plan_gates = gates


def extension_step_decision(journey, action):
    """Use the installed role/session and exact CAS snapshot; no fake done flag."""
    gates = getattr(journey, "extension_plan_gates", {})
    candidates = [identity for identity, item in gates.items() if item["tool"] == action.tool]
    if not candidates:
        return False
    wanted = "failed" if action.kind == "retry_step" else "awaiting_confirm"
    matches = []
    for identity in candidates:
        with actor_session(journey, "maker"):
            plan = journey.json_request("GET", f"/api/plans/{identity}", label="read_extension_plan_gate")["plan"]
        matches.extend((plan, step) for step in plan["steps"] if step["status"] == wanted)
    if len(matches) != 1:
        raise ValueError("extension_current_gate_not_unique")
    plan, step = matches[0]
    actor = "maker" if action.kind == "retry_step" else gates[plan["id"]]["actor"]
    with actor_session(journey, actor):
        journey.record_human_action(action)
        route = f"/api/plans/{plan['id']}/steps/{step['id']}"
        if action.kind == "retry_step":
            journey.json_request("POST", route + "/retry", label="human_extension_retry", json={})
        else:
            body = {"decision": "approve" if action.kind == "approve_step" else "reject", "reason": action.content, **step["confirmation_snapshot"]}
            if action.fault is not None:
                from .runtime_fault_capture import dispatch_fault
                workspace = getattr(journey, "workspace", None)
                if workspace is None:
                    raise ValueError("reference_fault_native_workspace_unavailable")
                dispatch_fault(journey, action, workspace, plan, step, body, actor=actor)
                return True
            response = journey.json_request("POST", route + "/decisions", label="human_extension_approval" if action.kind == "approve_step" else "human_extension_rejection", json=body)
            decisions = getattr(journey, "extension_decisions", [])
            decisions.append({"tool": action.tool, "actor": actor, "request": body, "response": response,
                              "http_ordinal": journey.count, "proposal_action": gates[plan["id"]]["proposal_action"],
                              "case_action_index": getattr(journey, "current_action_index", None), "plan_before": plan,
                              "action_sha256": _action_hash(action)})
            journey.extension_decisions = decisions
            if action.kind == "approve_step":
                journey.approval = (route + "/decisions", body)
                journey.extension_approval_actor = actor
    with actor_session(journey, "maker"):
        after = journey.wait_idle(changed_step=step["id"] if action.kind != "retry_step" else None)
        if action.kind != "retry_step":
            matches = [item for item in after if item["id"] == plan["id"]]
            if len(matches) != 1:
                raise ValueError("extension_native_plan_after_unavailable")
            decisions[-1]["plan_after"] = matches[0]
    return True
