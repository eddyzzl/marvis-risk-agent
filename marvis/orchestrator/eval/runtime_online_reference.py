"""Independent online rule decisions and calendar/ledger consistency."""
from datetime import timedelta
import math

from .runtime_archive_reader import _Unsupported
from .runtime_collections_reference import at
from .runtime_contracts import digest
from .runtime_strategy_reference import expression_reference


def rule_decision_reference(manifest, request):
    config = manifest["configuration"]
    if config.get("package_kind") != "rule_only" or manifest.get("model") is not None:
        raise _Unsupported("model scoring requires an independently supported native model/preprocessing carrier; archived pickle is never executed")
    if request.get("event_evidence") is not None or config.get("event_binding") is not None:
        raise _Unsupported("online event injection needs the separately authenticated event receipt and recipe reader")
    fields, inputs = config["raw_schema"], request["features"]
    if (len(inputs) > 500 or set(inputs) != {field["name"] for field in fields}
            or len(fields) != len({field["name"] for field in fields})
            or request["decision_node"] != config["decision_node"]):
        raise ValueError("online schema/node mismatch")
    for field in fields:
        value = inputs[field["name"]]
        if value is None:
            if not field.get("nullable", False):
                raise ValueError("online required value missing")
        elif field["type"] == "number":
            if type(value) not in {int, float} or not math.isfinite(value):
                raise ValueError("online finite number required")
        elif field["type"] == "integer":
            if type(value) is not int or not -(2**63) <= value < 2**63:
                raise ValueError("online bounded integer required")
        elif field["type"] == "string":
            if type(value) is not str or len(value) > 4096:
                raise ValueError("online bounded string required")
        elif field["type"] == "boolean":
            if type(value) is not bool:
                raise ValueError("online boolean required")
        else:
            raise _Unsupported("unknown installed online field type")
    spec = manifest["strategy_spec"]
    if spec.get("match_policy", "first_match") != "first_match" or len(spec["rules"]) > 4096:
        raise _Unsupported("online first-match rule budget")
    selected, matched = spec["default_action"], None
    for rule in sorted(spec["rules"], key=lambda row: (row["priority"], row["rule_id"])):
        if expression_reference(inputs, rule["condition"]):
            selected, matched = rule["action"], rule["rule_id"]
            break
    return {"action": selected, "matched_rule_id": matched, "score": None, "score_product": None}


def calendar_reference(contract, period):
    calendar = contract["calendar"]
    interval, index = calendar["interval_seconds"], period["period_index"]
    if type(interval) is not int or interval <= 0 or type(index) is not int or index < 0:
        raise ValueError("invalid fixed interval")
    start = at(calendar["anchor_at"]) + timedelta(seconds=index * interval)
    end = start + timedelta(seconds=interval)
    key = calendar["schema_version"] + ":" + start.isoformat().replace("+00:00", "Z") + "/" + end.isoformat().replace("+00:00", "Z")
    if (period["period_key"] != key or at(period["period_starts_at"]) != start
            or at(period["period_ends_at"]) != end or start < at(contract["active_from"])
            or (contract.get("active_until") and start >= at(contract["active_until"]))):
        raise ValueError("independent operations calendar differs")
    return key


def schedule_contract_reference(declared, *, execution_actor_id):
    """Project an exact public schedule plus its native authenticated publisher."""
    if not execution_actor_id:
        raise ValueError("operations publisher identity unavailable")
    clean = {key: value for key, value in declared.items() if value is not None}
    if set(clean["calendar"]) != {"anchor_at", "interval_seconds"}:
        raise _Unsupported("fixed interval calendar only")
    expected = {**clean, "schema_version": "operations.schedule.v1",
                "execution_actor_id": execution_actor_id,
                "calendar": {**clean["calendar"], "schema_version": "operations.calendar.fixed_interval.v1"},
                "retry_policy": {**clean["retry_policy"], "schema_version": "operations.retry.v1"}}
    for owner, key in ((expected, "active_from"), (expected, "active_until"), (expected["calendar"], "anchor_at")):
        if key in owner:
            owner[key] = at(owner[key]).isoformat().replace("+00:00", "Z")
    return expected


def audit_chain_reference(rows, expected_head):
    previous, sequence = "0" * 64, 0
    for row in rows:
        import json
        sequence += 1
        body = {key: row[key] for key in ("id", "sequence", "event_type", "actor_principal_id", "actor_role", "target_type", "target_id", "environment", "at", "previous_event_hash")}
        body["payload"] = json.loads(row["payload_json"])
        if row["sequence"] != sequence or row["previous_event_hash"] != previous or digest(body) != row["event_hash"]:
            raise ValueError("independent production audit chain differs")
        previous = row["event_hash"]
    if previous != expected_head:
        raise ValueError("externally held production audit head differs")
    return sequence
