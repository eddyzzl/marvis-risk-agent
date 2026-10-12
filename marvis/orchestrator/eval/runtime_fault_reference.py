"""Independent fixed native counter/output reference for future fault originals.

This reader never imports the capture implementation or business executors.
Archived observations establish local native consistency, not session authority
or the absence of an effect in an external institution.
"""
import hashlib
import json
import re

from .runtime_archive_domain import equal
from .runtime_archive_reader import _Unsupported
from .runtime_archive_reader import _json
from .runtime_contracts import digest

_FIELDS = {
    "decision_records": ("id", "plan_id", "step_id", "principal_id", "decision", "manifest_hash", "policy_hash", "input_hash", "evidence_hash", "created_at"),
    "approval_records": ("id", "decision_id", "plan_id", "step_id", "principal_id", "status", "reservation_id", "issued_at", "expires_at", "consumed_at", "revoked_at"),
    "plan_step_runs": ("id", "plan_id", "step_id", "attempt", "tool_ref", "status", "invocation_id", "output_ref", "output_hash", "raw_output_hash", "manifest_hash", "tool_version"),
    "plan_step_output_versions": ("step_id", "version", "created_at", "output_sha256", "evidence_sha256"),
    "effect_executions": ("id", "approval_id", "invocation_id", "reservation_id", "status", "released_at"),
}
_TARGET = ("id", "plan_id", "task_id", "plan_status", "replan_count", "tool_plugin", "tool_name", "status", "confirmed", "needs_confirmation", "output_ref", "input_sha256", "policy_sha256")


def native_json_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def resolved_original_inputs(ctx, before, step):
    """Resolve only the original declared dependency versions, never current heads."""
    outputs = {}
    for identity in step["depends_on"]:
        members = [row for row in before["steps"] if row["id"] == identity]
        if len(members) != 1:
            raise ValueError("original input dependency differs")
        reference = members[0].get("output_ref") or ""
        match = re.fullmatch(r"metrics:" + re.escape(identity) + r":v([1-9][0-9]*)", reference)
        if match is None:
            raise _Unsupported("original reference dependency version unavailable")
        saved = ctx.conn.execute("SELECT output_json FROM plan_step_output_versions WHERE step_id=? AND version=?", (identity, int(match[1]))).fetchone()
        if saved is None:
            raise ValueError("original input dependency output missing")
        outputs[identity] = json.loads(saved["output_json"])
    remaining = 100000

    def visit(value, depth=0):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 64:
            raise ValueError("original input reference budget")
        if isinstance(value, dict):
            return {key: visit(child, depth + 1) for key, child in value.items()}
        if isinstance(value, list):
            return [visit(child, depth + 1) for child in value]
        if not isinstance(value, str) or not value.startswith("$ref:"):
            return value
        match = re.fullmatch(r"\$ref:([^\s]+?)\.output(?:\.(.+))?", value)
        if match is None or match[1] not in outputs:
            raise _Unsupported("original reference outside exact declared dependency scope")
        result = outputs[match[1]]
        for key in match[2].split(".") if match[2] else ():
            if isinstance(result, dict) and key in result:
                result = result[key]
            elif isinstance(result, list) and key.isdigit() and int(key) < len(result):
                result = result[int(key)]
            else:
                raise ValueError("original reference field unavailable")
        return result

    return visit(step["inputs"])


def original_target(value, *, task, plan, step):
    if not isinstance(value, dict):
        raise _Unsupported("original fixed native target snapshot unavailable")
    equal(sorted(value), sorted(("schema", "target", "tables", "started_elapsed_ns", "finished_elapsed_ns", "capture", "authority")))
    equal(value["schema"], "marvis.runtime-target-originals.v1")
    equal(value["capture"], "fixed_native_fields_from_one_committed_read_transaction")
    equal(value["authority"], "not_established")
    equal(sorted(value["target"]), sorted(_TARGET))
    equal((value["target"]["task_id"], value["target"]["plan_id"], value["target"]["id"]), (task, plan, step))
    times = (value["started_elapsed_ns"], value["finished_elapsed_ns"])
    if any(type(t) is not int or t < 0 for t in times) or times[0] > times[1]:
        raise ValueError("native fault target original timing differs")
    equal(sorted(value["tables"]), sorted(_FIELDS))
    for name, fields in _FIELDS.items():
        rows = value["tables"][name]
        if not isinstance(rows, list) or len(rows) > 100 or any(not isinstance(row, dict) or set(row) != set(fields) for row in rows):
            raise ValueError("native fault fixed original rows differ")
        equal(rows, sorted(rows, key=digest))
        if len({digest(row) for row in rows}) != len(rows):
            raise ValueError("native fault original row repeated")
    return value


def archived_target(ctx, plan, step):
    row = ctx.conn.execute("SELECT s.*,p.task_id,p.status AS plan_status,p.replan_count FROM plan_steps s JOIN plans p ON p.id=s.plan_id WHERE s.id=? AND p.id=? AND p.task_id=?", (step, plan, ctx.binding.task_id)).fetchone()
    if row is None:
        raise ValueError("native exact archived fault target missing")
    target = {key: row[key] for key in _TARGET if not key.endswith("_sha256")}
    target.update(input_sha256=native_json_sha(json.loads(row["inputs_json"])), policy_sha256=native_json_sha(json.loads(row["policy_json"])))
    tables = {}
    for table, fields in _FIELDS.items():
        if table == "effect_executions":
            rows = ctx.conn.execute("SELECT e.* FROM effect_executions e JOIN approval_records a ON a.id=e.approval_id WHERE a.step_id=?", (step,)).fetchmany(101)
        else:
            rows = ctx.conn.execute(f"SELECT * FROM {table} WHERE step_id=?", (step,)).fetchmany(101)  # nosec B608 -- table iterates only source-owned _FIELDS literal keys; step id is bound
        if len(rows) > 100:
            raise ValueError("native archived target original budget")
        values = []
        for item in rows:
            value = {key: item[key] for key in fields if key not in {"output_sha256", "evidence_sha256"}}
            if table == "plan_step_output_versions":
                value.update(output_sha256=native_json_sha(json.loads(item["output_json"])), evidence_sha256=native_json_sha(json.loads(item["evidence_json"])))
            values.append(value)
        tables[table] = sorted(values, key=digest)
    return {"target": target, "tables": tables}


def original_progress_to_completed(snapshot, completed):
    """Allow only the native one-invocation success transitions of fixed rows."""
    mutable = {
        "decision_records": (),
        "approval_records": ("status", "reservation_id", "consumed_at"),
        "plan_step_runs": ("status", "invocation_id", "output_ref", "output_hash", "raw_output_hash", "manifest_hash", "tool_version"),
        "plan_step_output_versions": (),
        "effect_executions": ("status", "released_at"),
    }
    statuses = {"approval_records": ("issued", "reserved", "consumed"),
                "plan_step_runs": ("running", "succeeded"),
                "effect_executions": ("prepared", "dispatched", "committed")}
    for table, rows in snapshot["tables"].items():
        key = "version" if table == "plan_step_output_versions" else "id"
        for row in rows:
            targets = [target for target in completed["tables"][table] if target[key] == row[key]]
            if len(targets) != 1:
                raise ValueError("original native recovery row lost or replaced")
            target = targets[0]
            for field, value in row.items():
                if field not in mutable[table]:
                    equal(value, target[field])
                elif field == "status":
                    allowed = statuses[table]
                    if value not in allowed or target[field] not in allowed or allowed.index(value) > allowed.index(target[field]):
                        raise ValueError("native recovery status reversed or unsupported")
                elif value not in (None, ""):
                    equal(value, target[field])


def native_completed_output(ctx, plan_id, step_id, private):
    step = ctx.conn.execute("SELECT * FROM plan_steps WHERE id=? AND plan_id=?", (step_id, plan_id)).fetchone()
    match = re.fullmatch(r"metrics:" + re.escape(step_id) + r":v([1-9][0-9]*)", step["output_ref"] or "") if step else None
    if step is None or step["status"] != "done" or match is None:
        raise _Unsupported("actual native completed output unavailable")
    saved = ctx.conn.execute("SELECT * FROM plan_step_output_versions WHERE step_id=? AND version=?", (step_id, int(match[1]))).fetchone()
    if saved is None:
        raise ValueError("native original output version absent")
    output, evidence = json.loads(saved["output_json"]), json.loads(saved["evidence_json"])
    run = ctx.conn.execute("SELECT * FROM plan_step_runs WHERE id=? AND step_id=? AND plan_id=?", (evidence.get("step_run_id"), step_id, plan_id)).fetchone()
    if run is None or run["status"] != "succeeded" or run["id"] != run["invocation_id"]:
        raise ValueError("native original successful invocation absent")
    tool = step["tool_plugin"] + "." + step["tool_name"]
    output_hash = "sha256:" + native_json_sha(output)
    try:
        observed_hash = digest(output)
    except ValueError:
        observed_hash = output_hash.removeprefix("sha256:")
    equal(native_json_sha(output), native_json_sha(private.get("outputs", {}).get(step_id)))
    equal(run["tool_ref"], tool)
    equal(run["output_ref"], step["output_ref"])
    equal(run["output_hash"], output_hash)
    values = {"schema_version": "evidence.v1", "task_id": ctx.binding.task_id, "plan_id": plan_id,
              "step_id": step_id, "tool_name": tool, "producer_invocation_id": run["id"],
              "step_run_id": run["id"], "output_ref": step["output_ref"], "output_hash": output_hash,
              "input_hash": "sha256:" + native_json_sha(json.loads(run["input_json"])),
              "manifest_hash": run["manifest_hash"], "tool_version": run["tool_version"], "raw_output_hash": run["raw_output_hash"]}
    for key, value in values.items():
        equal(evidence.get(key), value)
    recorded = [row for row in _json(ctx.archive, "bindings.json")["execution_before_scoring"]["execution"].get("steps", ()) if row.get("id") == step_id]
    if len(recorded) != 1 or not recorded[0].get("binding_verified"):
        raise _Unsupported("original authenticated native output observation unavailable")
    for key, value in (("tool", tool), ("output_ref", step["output_ref"]), ("output_sha256", observed_hash), ("producer_invocation_id", run["id"])):
        equal(recorded[0].get(key), value)
    return {"output_ref": step["output_ref"], "output_sha256": observed_hash, "invocation_id": run["id"]}
