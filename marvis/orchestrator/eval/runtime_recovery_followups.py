"""Record only declared native modeling followups without altering recovered targets."""
from .runtime_contracts import digest
from .runtime_fault_capture import capture_target_original

PAIRS = {("select_recommended_experiment", "modeling.select_experiment"),
         ("approve_step", "modeling.generate_model_reports"), ("approve_step", "modeling.post_training_action")}


def begin_followup(journey, action, workspace):
    if journey.case.family != "modeling" or (action.kind, action.tool) not in PAIRS:
        return []
    captures = []
    for fault in getattr(journey, "runtime_faults", []):
        recovery = fault.get("recovery")
        if (fault["kind"] != "discard_success_response_once" or fault["tool"] != "modeling.configure_tuning"
                or not recovery or recovery["action_index"] >= journey.current_action_index):
            continue
        captures.append((fault, {"action_index": journey.current_action_index, "action_sha256": digest(action.model_dump()),
            "ordinal_before": journey.count, "before": capture_target_original(workspace, journey.task_id,
                fault["plan_id"], fault["step_id"], origin_ns=journey.origin_ns)}))
    return captures


def finish_followup(journey, captures, workspace):
    for fault, record in captures:
        record["http_ordinals"] = list(range(record.pop("ordinal_before") + 1, journey.count + 1))
        record["after"] = capture_target_original(workspace, journey.task_id, fault["plan_id"], fault["step_id"], origin_ns=journey.origin_ns)
        fault.setdefault("declared_followups", []).append(record)
