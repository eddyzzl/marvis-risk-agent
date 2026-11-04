"""Case-declared current-version confirmation following a stale report refusal."""
from .runtime_contracts import digest
from .runtime_validation_fault import _capture


def recover_stale_report(journey, action, fault):
    from .runtime_validation_adapter import confirm_current_report
    if (action.kind != "recover_validation_report" or fault["kind"] != "stale_report_revision_once"
            or action.fault_source_action != fault["action_index"] or action.fault_source_action >= journey.current_action_index):
        raise ValueError("stale_report_recovery_original_unavailable")
    before = _capture(journey)
    begin = journey.count
    # This action is an explicit frozen human confirmation. The normal adapter
    # reads the current draft/CAS, submits it once, waits and downloads reports.
    journey.capture_stale_confirmation = {}
    try:
        confirm_current_report(journey, action)
        captured = journey.capture_stale_confirmation
    finally:
        del journey.capture_stale_confirmation
    fault["recovery"] = {"action_index": journey.current_action_index,
        "action_sha256": digest(action.model_dump()), "before_confirmation": before,
        "confirmation": captured, "http_ordinals": list(range(begin + 1, journey.count + 1)),
        "after_readback": _capture(journey), "downloads": list(journey.validation_downloads)}
