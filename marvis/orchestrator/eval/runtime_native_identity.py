"""Finite original-action identity resolution from retained native carriers.

This never rewrites a case or trusts response IDs by themselves. Replacement
values are accepted only from the corresponding original action's native row.
It establishes local consistency, not the omitted native signing authority.
"""
from copy import deepcopy
import json
from types import SimpleNamespace

from pydantic import TypeAdapter

from .runtime_archive_domain import equal
from .runtime_archive_reader import _Unsupported, _json
from .runtime_contracts import digest
from .runtime_extension_inputs import ExtensionBusinessInput


def _replace(value, target, replacement):
    parts = target.split(".")
    def walk(node, index):
        field = parts[index]
        if field.endswith("[]"):
            members = node[field[:-2]]
            if not isinstance(members, list) or not members:
                raise ValueError("native identity target has no declared members")
            for member in members:
                walk(member, index + 1)
        elif field not in node:
            raise ValueError("native identity target absent")
        elif index == len(parts) - 1:
            node[field] = replacement
        else:
            walk(node[field], index + 1)
    walk(value, 0)


def declaration_matches(case, kind, request, *, slot=None):
    """Structural admission only; every replaced identity is later revalidated."""
    actual_request = request.model_dump() if hasattr(request, "model_dump") else request
    for action in case.actions:
        item = getattr(action, "extension_request", None) or action
        if getattr(item, "kind", None) != kind or (slot is not None and item.slot != slot):
            continue
        original = item.model_dump()
        actual = {**deepcopy(original), "request": deepcopy(actual_request)}
        for binding in getattr(action, "extension_bindings", ()):
            if binding.target.startswith("request."):
                marker = {"fixed_original_binding": binding.target}
                _replace(original, binding.target, marker)
                _replace(actual, binding.target, marker)
        if digest(original) == digest(actual):
            return True
    return False


# Every semantic pair is repeated here independently of the producer adapter.
_PAIRS = {
    ("claim_extension_role", "id"): {"request.grantee_id"},
    ("declare_source_authorization_basis", "id"): {"request.basis_artifact_id"},
    ("register_risk_event_source", "source_id"): {"request.source_id", "request.dataset.source_id", "request.contract.source_id"},
    ("register_risk_event_source", "contract_hash"): {"request.source_contract_hash", "request.dataset.source_contract_hash", "request.contract.source_contract_hash"},
    ("grant_risk_source", "grant_id"): {"request.grant_id", "grant_id"},
    ("grant_risk_event_source", "grant_id"): {"request.grant_id", "grant_id"},
    ("import_risk_event_dataset", "ingested_at"): {"request.contract.knowledge_cutoff"},
    ("prepare_risk_source_query", "request_id"): {"request_id"},
    ("prepare_risk_event_features", "request_id"): {"request_id"},
    ("prepare_collection_batch", "batch_id"): {"batch_id"},
    ("build_decision_package", "package_hash"): {"request.decision_package_hash", "request.package_hash", "request.expected_package_hash"},
    ("request_production_promotion", "id"): {"promotion_id", "request.promotion_id"},
    ("install_decision_package", "activation_evidence.receipt_id"): {"request.receipt_id"},
    ("activate_production_promotion", "id"): {"deployment_id", "request.expected_active_deployment_id"},
    ("publish_operations_schedule", "contract.schedule_id"): {"schedule_id"},
}
for field, suffix in (("source_artifact_id", "id"), ("source_artifact_hash", "hash")):
    _PAIRS[("declare_collection_material", field)] = {
        f"request.source_artifact_{suffix}", f"request.terms_artifact_{suffix}", f"request.policy.basis_artifact_{suffix}",
        *(f"request.{part}[].source_artifact_{suffix}" for part in ("cases", "histories", "events", "coverage")),
    }


def _one(ctx, table, where, parameters):
    # Table/where are solely fixed literals below, never public input paths.
    rows = ctx.conn.execute(f"SELECT * FROM {table} WHERE {where}", parameters).fetchmany(2)  # nosec B608 -- table/WHERE are source literals at all installed _one callers; only variable table is source_grants/event_grants; external values are bound
    if len(rows) != 1:
        raise ValueError("native identity source is absent or ambiguous")
    return dict(rows[0])


def _response_native(ctx, item, output, actors):
    """Return only identities authenticated against the corresponding row."""
    kind, request, response = item["kind"], item.get("request"), output["response"]
    if output.get("http_status") not in {200, 201, 202} or response.get("status") == "http_rejected":
        raise _Unsupported("rejected response has no usable native identity")
    if kind == "claim_extension_role":
        row = _one(ctx, "production_principals", "local_principal_id=?", (response["id"],))
        local = _one(ctx, "local_principals", "id=?", (response["id"],))
        if row["role"] != request["role"] or row["display_name"] != request["display_name"] or local["kind"] != "local_session":
            raise ValueError("native role claim differs")
        actors[item["actor"]] = response["id"]
        return {"id": response["id"]}
    actor = actors.get(item["actor"])
    if actor is None:
        raise _Unsupported("original role claim is required for native identity")
    if kind in {"declare_source_authorization_basis", "declare_collection_material"}:
        identity = response["id"] if kind == "declare_source_authorization_basis" else response["source_artifact_id"]
        row = _one(ctx, "task_artifacts", "task_id=? AND id=?", (ctx.binding.task_id, identity))
        origin = "risk_context.source_query.v1" if kind == "declare_source_authorization_basis" else "collection.declare_material"
        body = ctx.artifact(identity, row["content_hash"], origin=origin)
        if kind == "declare_source_authorization_basis":
            equal(body["body"]["basis"], request)
            if body["body"]["task_id"] != ctx.binding.task_id or body["body"]["actor_id"] != actor:
                raise ValueError("native basis actor/task differs")
            return {"id": identity}
        if body.get("schema_version") != "collection.declaration.v1" or body.get("source_truth_verified") is not False:
            raise ValueError("native collection material assurance differs")
        for key, value in request.items():
            equal(body[key], value)
        owner = _one(ctx, "collection_ownership", "task_id=? AND kind='material' AND identity=?", (ctx.binding.task_id, request["material_id"]))
        owner_body = json.loads(owner["payload_json"])
        if owner_body["actor_id"] != actor or owner_body["task_id"] != ctx.binding.task_id or owner_body["identity"] != request["material_id"] or response["source_artifact_hash"] != row["content_hash"]:
            raise ValueError("native collection material identity differs")
        return {"source_artifact_id": identity, "source_artifact_hash": row["content_hash"]}
    if kind == "register_risk_event_source":
        row = _one(ctx, "event_sources", "id=? AND task_id=?", (response["source_id"], ctx.binding.task_id))
        equal(json.loads(row["payload"]), request)
        if row["created_by"] != actor or row["content_hash"] != digest(request) or response["contract_hash"] != row["content_hash"]:
            raise ValueError("native event source identity differs")
        return {"source_id": row["id"], "contract_hash": row["content_hash"]}
    if kind in {"grant_risk_source", "grant_risk_event_source"}:
        table = "source_grants" if kind == "grant_risk_source" else "event_grants"
        row = _one(ctx, table, "id=? AND task_id=?", (response["grant_id"], ctx.binding.task_id))
        payload = json.loads(row["payload"])
        equal(payload if kind == "grant_risk_source" else payload["grant"], request)
        if row["created_by"] != actor:
            raise ValueError("native source grant actor differs")
        return {"grant_id": row["id"]}
    if kind == "import_risk_event_dataset":
        sha = digest(request["dataset"])
        row = _one(ctx, "event_imports", "source_id=? AND contract_hash=?", (request["dataset"]["source_id"], sha))
        payload = json.loads(row["payload"])
        equal(payload["import_contract"], request["dataset"])
        grant = _one(ctx, "event_grants", "task_id=? AND id=?", (ctx.binding.task_id, request["grant_id"]))
        if json.loads(grant["payload"])["grant"]["grantee_id"] != actor or response["ingested_at"] != payload["ingested_at"]:
            raise ValueError("native event import lineage differs")
        return {"ingested_at": payload["ingested_at"]}
    if kind == "prepare_risk_event_features":
        row = _one(ctx, "event_requests", "task_id=? AND request_id=?", (ctx.binding.task_id, request["request_id"]))
        payload = json.loads(row["payload"])
        equal(payload["request"], request)
        if payload["actor_id"] != actor or response["request_id"] != request["request_id"] or response["proposal_hash"] != digest(payload):
            raise ValueError("native event proposal differs")
        return {"request_id": request["request_id"]}
    if kind == "prepare_risk_source_query":
        row = _one(ctx, "source_requests", "task_id=? AND id=?", (ctx.binding.task_id, request["request_id"]))
        payload = json.loads(row["contract"])
        equal(payload["query"], request)
        if row["actor_id"] != actor or row["contract_hash"] != digest(payload) or response["request_id"] != request["request_id"]:
            raise ValueError("native source proposal differs")
        return {"request_id": request["request_id"]}
    if kind == "prepare_collection_batch":
        row = _one(ctx, "collection_batches", "task_id=? AND batch_id=?", (ctx.binding.task_id, request["batch_id"]))
        equal(json.loads(row["request_json"]), request)
        if row["request_hash"] != digest(request) or response["batch_id"] != row["batch_id"]:
            raise ValueError("native collection proposal differs")
        return {"batch_id": row["batch_id"]}
    if kind == "build_decision_package":
        row = _one(ctx, "reference_decision_packages", "id=?", (response["package_hash"],))
        body = json.loads(row["canonical_json"])
        if row["created_by"] != actor or digest(body) != row["id"]:
            raise ValueError("native package actor/bytes differs")
        if body["strategy"]["id"] != request["strategy_id"] or body["strategy"]["version"] != request["strategy_version"]:
            raise ValueError("native package strategy differs")
        for key in ("package_kind", "decision_node", "raw_schema"):
            equal(body["configuration"][key], request[key])
        return {"package_hash": row["id"]}
    if kind == "request_production_promotion":
        row = _one(ctx, "production_promotion_requests", "id=?", (response["id"],))
        for key in ("environment", "deployment_slot", "strategy_id", "strategy_version", "reason"):
            equal(row[key], request[key])
        manifest = _one(ctx, "production_deployment_manifests", "id=?", (row["manifest_hash"],))
        equal(json.loads(manifest["canonical_json"])["decision_package_hash"], request["decision_package_hash"])
        if row["maker_principal_id"] != actor:
            raise ValueError("native promotion actor differs")
        return {"id": row["id"]}
    if kind == "install_decision_package":
        row = _one(ctx, "reference_installations", "promotion_id=?", (request["promotion_id"],))
        body = json.loads(row["receipt_json"])
        equal(body["activation_evidence"], response["activation_evidence"])
        return {"activation_evidence.receipt_id": body["activation_evidence"]["receipt_id"]}
    if kind == "activate_production_promotion":
        row = _one(ctx, "production_deployments", "id=?", (response["id"],))
        if row["promotion_request_id"] != item["promotion_id"]:
            raise ValueError("native activation promotion differs")
        return {"id": row["id"]}
    if kind == "publish_operations_schedule":
        contract = response["contract"]
        row = _one(ctx, "operations_schedules", "schedule_id=? AND revision=?", (contract["schedule_id"], contract["revision"]))
        equal(json.loads(row["contract_json"]), contract)
        if row["contract_hash"] != digest(contract) or contract["execution_actor_id"] != actor:
            raise ValueError("native operations publisher differs")
        return {"contract.schedule_id": contract["schedule_id"]}
    raise _Unsupported("native identity carrier is unsupported")


def _material(ctx, ordinal, field):
    if not 1 <= ordinal <= len(ctx.binding.case.materials):
        raise ValueError("native material index differs")
    material = ctx.binding.case.materials[ordinal - 1]
    directory = ctx.archive / "workspace/datasets" / ctx.binding.task_id / ".source-identities"
    matches = []
    if directory.is_dir():
        paths = list(directory.glob("*.json"))
        if len(paths) > 1000:
            raise _Unsupported("native material identity budget")
        for path in paths:
            body = _json(ctx.archive, path.relative_to(ctx.archive))
            if body.get("sha256") == material.sha256:
                matches.append(path.stem)
    if len(matches) != 1:
        raise _Unsupported("original material must have one authenticated native upload identity")
    row = _one(ctx, "datasets", "id=? AND task_id=?", (matches[0], ctx.binding.task_id))
    from .runtime_archive_connector import _registered_source
    _registered_source(ctx, row["id"], row["content_hash"])
    return row["id"] if field == "dataset_id" else row["content_hash"]


def _strategy_output(ctx, output, binding):
    from marvis.orchestrator.evidence import payload_hash
    source = output.get("native_identity_sources", {}).get(binding.source_tool)
    if not isinstance(source, dict) or set(source) != {"step_id", "plan_id", "output_ref", "output_sha256", "http_ordinal"}:
        raise _Unsupported("native strategy output requires original per-action output-ref capture")
    if type(source["http_ordinal"]) is not int or source["http_ordinal"] >= output["http_ordinal"]:
        raise ValueError("native strategy identity was not read before this action")
    step = _one(ctx, "plan_steps", "id=? AND plan_id=?", (source["step_id"], source["plan_id"]))
    plan = _one(ctx, "plans", "id=? AND task_id=?", (source["plan_id"], ctx.binding.task_id))
    tool = step["tool_plugin"] + "." + step["tool_name"]
    prefix = "metrics:" + step["id"] + ":v"
    if tool != binding.source_tool or not source["output_ref"].startswith(prefix):
        raise ValueError("native strategy output identity differs")
    suffix = source["output_ref"][len(prefix):]
    if not suffix.isdigit() or int(suffix) < 1:
        raise ValueError("native strategy output version differs")
    version = _one(ctx, "plan_step_output_versions", "step_id=? AND version=?", (step["id"], int(suffix)))
    body, evidence = json.loads(version["output_json"]), json.loads(version["evidence_json"])
    run = _one(ctx, "plan_step_runs", "id=?", (evidence.get("step_run_id"),))
    if (digest(body) != source["output_sha256"] or run["status"] != "succeeded" or run["tool_ref"] != tool
            or run["plan_id"] != plan["id"] or run["step_id"] != step["id"] or run["output_ref"] != source["output_ref"]
            or run["invocation_id"] != run["id"] or run["output_hash"] != payload_hash(body)
            or evidence.get("producer_invocation_id") != run["id"] or evidence.get("raw_output_hash") != run["raw_output_hash"]
            or evidence.get("manifest_hash") != run["manifest_hash"] or evidence.get("input_hash") != payload_hash(json.loads(run["input_json"]))
            or evidence.get("task_id") != ctx.binding.task_id or evidence.get("output_ref") != source["output_ref"]):
        raise ValueError("native strategy producer closure differs")
    strategy = _one(ctx, "strategies", "id=? AND task_id=?", (body["strategy_id"], ctx.binding.task_id))
    if "version" in body and strategy["version"] != body["version"]:
        raise ValueError("native strategy version differs")
    if "strategy_spec" in body:
        equal(json.loads(strategy["dsl_json"]), body["strategy_spec"])
    if binding.source_field not in body:
        raise _Unsupported("original native output did not return a version; an explicit declared version is required")
    return body[binding.source_field]


def resolve_native_case(ctx):
    """Keep original case immutable; return an authenticated semantic projection."""
    original = ctx.binding.case
    actions = [a for a in original.actions if getattr(a, "extension_request", None) is not None]
    if not actions:
        return original
    from .runtime_native_json import decode_native
    private = decode_native(_json(ctx.archive, "private.json"))
    outputs = private.get("extension_outputs")
    if not isinstance(outputs, list) or len(outputs) != len(actions):
        raise _Unsupported("complete original action responses are required for identity resolution")
    previous, resolved, actors, native_cache = [], [], {}, {}
    for index, action in enumerate(original.actions, 1):
        if getattr(action, "extension_request", None) is None:
            continue
        ordinal = len(previous)
        output = outputs[ordinal]
        if output.get("case_action_index") != index or output.get("action_sha256") != digest(action.model_dump()):
            raise _Unsupported("original action ordinal/hash capture is missing or differs")
        value = action.extension_request.model_dump()
        for binding in action.extension_bindings:
            if binding.source_field == "task_id":
                if binding.target != "request.task_id":
                    raise ValueError("native task semantic target differs")
                actual = ctx.binding.task_id
            elif binding.source_field == "resolved_policy_hash":
                if value["kind"] != "prepare_collection_batch" or binding.target not in {"request.strategy.default_action.value.policy_hash", "request.strategy.rules[].action.value.policy_hash"}:
                    raise ValueError("native policy semantic target differs")
                actual = digest(value["request"]["policy"])
            elif binding.source_material is not None:
                targets = {"dataset_id": {"request.dataset.dataset_id", "request.historical.dataset_id"}, "dataset_content_hash": {"request.dataset.expected_content_hash", "request.historical.expected_content_hash"}}
                if binding.target not in targets.get(binding.source_field, ()):
                    raise ValueError("native material semantic target differs")
                actual = _material(ctx, binding.source_material, binding.source_field)
            elif binding.source_tool is not None:
                if (binding.target, binding.source_field) not in {("request.strategy_id", "strategy_id"), ("request.strategy_version", "version")}:
                    raise ValueError("native strategy semantic target differs")
                actual = _strategy_output(ctx, output, binding)
            else:
                source = binding.source_action
                if source is None or not 1 <= source <= len(previous):
                    raise ValueError("native identity forward reference")
                prior = previous[source - 1]
                if binding.target not in _PAIRS.get((prior["kind"], binding.source_field), ()):
                    raise ValueError("native identity semantic pair differs")
                if prior["kind"] == "claim_extension_role" and prior["actor"] != "maker":
                    raise ValueError("native grantee requires original maker role")
                if source not in native_cache:
                    native_cache[source] = _response_native(ctx, resolved[source - 1].model_dump(), prior, actors)
                actual = native_cache[source][binding.source_field]
            if type(actual) not in {str, int}:
                raise ValueError("native identity scalar differs")
            _replace(value, binding.target, actual)
        equal(output["request"], value)
        if output["kind"] != value["kind"] or output["actor"] != value["actor"]:
            raise ValueError("original extension response kind/actor differs")
        item = TypeAdapter(ExtensionBusinessInput).validate_json(json.dumps(value, allow_nan=False))
        resolved.append(item)
        previous.append(output)
        if item.kind == "claim_extension_role":
            native_cache[ordinal + 1] = _response_native(ctx, value, output, actors)
    ctx.native_actors = actors
    return SimpleNamespace(id=original.id, task=original.task, materials=original.materials, actions=resolved, family=original.family)
