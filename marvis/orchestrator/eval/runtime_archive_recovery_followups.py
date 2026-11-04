"""Read exact post-recovery modeling actions while keeping target rows immutable."""
import json

from .runtime_archive_domain import equal
from .runtime_archive_reader import _Unsupported, _json
from .runtime_contracts import digest
from .runtime_collections_reference import at
from .runtime_fault_reference import archived_target, native_completed_output, original_target

_PAIRS = (("select_recommended_experiment", "modeling.select_experiment"),
          ("approve_step", "modeling.generate_model_reports"), ("approve_step", "modeling.post_training_action"))


def final_target(ctx, fault, post):
    from .runtime_archive_safety import _original_cas, _private, _governed
    from .runtime_safety_contracts import FrozenGovernedAction
    final = archived_target(ctx, fault["plan_id"], fault["step_id"])
    if final == {k: post[k] for k in ("target", "tables")}:
        return final
    if ctx.binding.case.family != "modeling" or fault["tool"] != "modeling.configure_tuning":
        raise ValueError("recovered target changed")
    rows = fault.get("declared_followups")
    if not isinstance(rows, list) or len(rows) != 3:
        raise _Unsupported("exact declared modeling continuation originals unavailable")
    index = fault["recovery"]["action_index"]
    later = ctx.binding.case.actions[index:]
    if [(a.kind, a.tool) for a in later] != list(_PAIRS) or len(rows) != len(later):
        raise _Unsupported("only fixed native model selection/report/delivery continuation is supported")
    private = _private(ctx)
    events = _json(ctx.archive, "bindings.json")["execution_before_scoring"]["http_events"]
    target_run = ctx.conn.execute("SELECT finished_at FROM plan_step_runs WHERE plan_id=? AND step_id=? AND status='succeeded'",
                                  (fault["plan_id"], fault["step_id"])).fetchmany(2)
    if len(target_run) != 1:
        raise ValueError("native recovered target completion timestamp unavailable")
    previous_native_finish = at(target_run[0]["finished_at"])
    previous = post
    last_ordinal = fault["recovery"]["http_ordinals"][-1]
    for offset, (row, action) in enumerate(zip(rows, later, strict=True), 1):
        action_index = index + offset
        equal((row["action_index"], row["action_sha256"]), (action_index, digest(action.model_dump())))
        pre, after = [original_target(row.get(key), task=ctx.binding.task_id, plan=fault["plan_id"], step=fault["step_id"])
                      for key in ("before", "after")]
        equal({k: pre[k] for k in ("target", "tables")}, {k: previous[k] for k in ("target", "tables")})
        equal(after["tables"], post["tables"])
        equal({k: v for k, v in after["target"].items() if k != "plan_status"},
              {k: v for k, v in post["target"].items() if k != "plan_status"})
        if after["target"]["plan_status"] not in {"running", "review", "awaiting_confirm", "done"}:
            raise ValueError("native continuation plan status differs")
        ordinals = row["http_ordinals"]
        if not ordinals or ordinals != list(range(last_ordinal + 1, ordinals[-1] + 1)):
            raise ValueError("native continuation request sequence differs")
        requests = [e for e in events if e["ordinal"] in ordinals]
        if len(requests) != len(ordinals):
            raise ValueError("native continuation requests absent")
        posts = [e for e in requests if e["method"] != "GET"]
        label = "human_recommended_selection" if offset == 1 else "human_approval"
        if len(posts) != 1 or (posts[0]["method"], posts[0]["stage"], posts[0]["status_code"]) != ("POST", label, 202):
            raise ValueError("native continuation requires exactly its declared successful write")
        if any(e["status_code"] != 200 for e in requests if e["method"] == "GET"):
            raise ValueError("native continuation read failed")
        captures = [c for c in private.get("model_selections" if offset == 1 else "human_decisions", [])
                    if c.get("case_action_index") == action_index]
        if len(captures) != 1:
            raise _Unsupported("native continuation CAS preimage unavailable")
        capture = captures[0]
        equal((capture["action_sha256"], capture["http_ordinal"]), (digest(action.model_dump()), posts[0]["ordinal"]))
        plan, body = capture["plan_before"], capture["request"]
        equal((plan["id"], plan["task_id"], plan["replan_count"]),
              (fault["plan_id"], ctx.binding.task_id, post["target"]["replan_count"]))
        matches = [s for s in plan["steps"] if s["tool_ref"]["plugin"] + "." + s["tool_ref"]["tool"] == action.tool]
        if len(matches) != 1 or matches[0]["id"] == fault["step_id"]:
            raise ValueError("native continuation target differs")
        step = _original_cas(plan, matches[0]["id"], body)
        equal(step["status"], "awaiting_confirm")
        decisions = ctx.conn.execute("SELECT * FROM decision_records WHERE plan_id=? AND step_id=?", (plan["id"], step["id"])).fetchmany(2)
        if len(decisions) != 1 or decisions[0]["decision"] != "approve" or decisions[0]["reason"] != ("确认" if offset == 1 else action.content):
            raise ValueError("native continuation decision differs")
        if offset == 1:
            equal((body["content"], body["ui_action"], body["expected_plan_id"], body["expected_step_id"]),
                  (action.content, "confirm_gate", plan["id"], step["id"]))
            gate = capture["gate"]
            # The app updates its current gate message in place. Bind the
            # retained GET preimage to the actual later native user action;
            # requiring the final DB to still contain the old gate is invalid.
            reads = [e for e in requests if e["stage"] == "read_selection_gate"]
            if len(reads) != 1 or reads[0]["finished_elapsed_ns"] > posts[0]["started_elapsed_ns"]:
                raise ValueError("native selection display read original differs")
            messages = [dict(message) for message in ctx.conn.execute(
                "SELECT content,metadata_json,created_at FROM agent_messages WHERE task_id=? AND role='user'", (ctx.binding.task_id,))]
            native_ui = [message for message in messages if json.loads(message["metadata_json"]).get("ui_action") == "confirm_gate"
                         and json.loads(message["metadata_json"]).get("expected_plan_id") == plan["id"]
                         and json.loads(message["metadata_json"]).get("expected_step_id") == step["id"]]
            if len(native_ui) != 1 or native_ui[0]["content"] != action.content:
                raise ValueError("native selection normalized user action differs")
            equal((capture["response"]["task_id"], capture["response"]["status"]), (ctx.binding.task_id, "ok"))
            equal(gate["confirmation_snapshot"], step["confirmation_snapshot"])
            selected = body["adjust_params"]["selected_experiment_id"]
            equal(selected, gate["model_delivery"]["recommended_experiment_id"])
            equal([c["id"] for c in gate["model_delivery"]["candidates"] if c.get("recommended") is True], [selected])
            inputs = json.loads(ctx.conn.execute("SELECT inputs_json FROM plan_steps WHERE id=?", (step["id"],)).fetchone()[0])
            equal(inputs["selected_experiment_id"], selected)
            equal(body["adjust_params"], {"selected_experiment_id": selected,
                                          "refit_on_train_plus_test": action.refit_on_train_plus_test})
            equal(inputs.get("refit_on_train_plus_test"), action.refit_on_train_plus_test)
            trains = ctx.conn.execute("SELECT s.id,o.output_json FROM plan_steps s JOIN plan_step_output_versions o ON o.step_id=s.id "
                "WHERE s.plan_id=? AND s.tool_plugin='modeling' AND s.tool_name='train_models'", (plan["id"],)).fetchmany(2)
            if len(trains) != 1:
                raise ValueError("native selection candidate origin differs")
            train = json.loads(trains[0]["output_json"])
            equal(train["best_experiment_id"], selected)
            equal(sorted(c["id"] for c in gate["model_delivery"]["candidates"]), sorted(train["experiment_ids"]))
            native_completed_output(ctx, plan["id"], trains[0]["id"], private)
        else:
            equal((body["decision"], body["reason"], capture["response"]["decision_id"]),
                  ("approve", action.content, decisions[0]["id"]))
        runs = ctx.conn.execute("SELECT * FROM plan_step_runs WHERE plan_id=? AND step_id=?", (plan["id"], step["id"])).fetchmany(2)
        if len(runs) != 1:
            raise ValueError("native continuation invocation timestamp unavailable")
        native_run = runs[0]
        if not (previous_native_finish <= at(decisions[0]["created_at"]) <= at(native_run["started_at"])
                <= at(native_run["dispatch_started_at"]) <= at(native_run["finished_at"])):
            raise ValueError("native continuation decision/invocation chronology differs")
        previous_native_finish = at(native_run["finished_at"])
        if offset == 1:
            # confirm_gate persists the displayed user message after the
            # accepted native action; it is not a pre-dispatch receipt.
            if at(native_ui[0]["created_at"]) < at(decisions[0]["created_at"]):
                raise ValueError("native selection user action chronology differs")
            previous_native_finish = max(previous_native_finish, at(native_ui[0]["created_at"]))
        _governed(ctx, FrozenGovernedAction(action_index=action_index, plan_id=plan["id"], step_id=step["id"],
            decision_id=decisions[0]["id"], decision_record_sha256=digest(dict(decisions[0]))), continuation=True)
        native_completed_output(ctx, plan["id"], step["id"], private)
        if not previous["finished_elapsed_ns"] <= pre["started_elapsed_ns"] <= pre["finished_elapsed_ns"]:
            raise ValueError("native continuation snapshot timeline differs")
        when = pre["finished_elapsed_ns"]
        for event in requests:
            if not when <= event["started_elapsed_ns"] <= event["finished_elapsed_ns"] <= after["started_elapsed_ns"]:
                raise ValueError("native continuation HTTP timeline differs")
            when = event["finished_elapsed_ns"]
        previous, last_ordinal = after, ordinals[-1]
    if any(e["method"] != "GET" for e in events if e["ordinal"] > last_ordinal):
        raise ValueError("another write followed fixed native continuation")
    equal(final, {k: previous[k] for k in ("target", "tables")})
    return final
