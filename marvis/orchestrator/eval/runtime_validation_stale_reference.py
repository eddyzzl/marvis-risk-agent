"""Validate stale zero effects before reading the explicit current confirmation."""
from .runtime_archive_domain import equal
from .runtime_archive_reader import _Unsupported, _json
from .runtime_contracts import digest


def confirmation_original(ctx, fault, original, action, index):
    from .runtime_validation_boundary import _boundary_values
    equal((fault["schema"], fault["task_id"], fault["kind"], fault["action_sha256"]),
          ("marvis.validation-report-boundary.v1", ctx.binding.task_id, "stale_report_revision_once", digest(original.model_dump())))
    events = _json(ctx.archive, "bindings.json")["execution_before_scoring"]["http_events"]
    # Only the refusal interval is zero effect. Later explicit approval is
    # separately bound below; the full boundary reader rightly disallows it.
    _boundary_values(fault, [e for e in events if e["ordinal"] <= fault["http_ordinal"]])
    recovery = fault.get("recovery")
    if not isinstance(recovery, dict):
        raise _Unsupported("current confirmation recovery original unavailable")
    equal((recovery["action_index"], recovery["action_sha256"]), (index, digest(action.model_dump())))
    capture = recovery["confirmation"]
    request, response = capture["request"], capture["response"]
    equal(request, fault["original_request"])
    pre, after = recovery["before_confirmation"], capture["after_response"]
    equal(pre["state"], fault["after_response"]["state"])
    ordinals = recovery["http_ordinals"]
    if not ordinals or ordinals != list(range(fault["http_ordinal"] + 1, ordinals[-1] + 1)):
        raise ValueError("current confirmation request sequence differs")
    requests = [e for e in events if e["ordinal"] in ordinals]
    if len(requests) != len(ordinals):
        raise ValueError("current confirmation requests absent")
    posts = [e for e in events if e["ordinal"] > fault["http_ordinal"] and e["method"] != "GET"]
    if len(posts) != 1 or posts[0]["ordinal"] != capture["http_ordinal"]:
        raise ValueError("stale recovery requires exactly one fresh confirmation")
    posted = posts[0]
    equal((posted["method"], posted["stage"], posted["status_code"]), ("POST", "human_validation_report_confirmation", 202))
    if not (fault["after_response"]["finished_elapsed_ns"] <= pre["started_elapsed_ns"] <= pre["finished_elapsed_ns"]
            <= requests[0]["started_elapsed_ns"] <= posted["started_elapsed_ns"] <= posted["finished_elapsed_ns"]
            <= after["started_elapsed_ns"] <= after["finished_elapsed_ns"]):
        raise ValueError("current confirmation original timeline differs")
    reads = [e for e in requests if e["ordinal"] < posted["ordinal"]]
    if len(reads) != 1 or (reads[0]["method"], reads[0]["stage"], reads[0]["status_code"]) != ("GET", "read_validation_report_draft", 200):
        raise ValueError("fresh current draft must be read once before confirmation")
    if reads[0]["finished_elapsed_ns"] > posted["started_elapsed_ns"]:
        raise ValueError("current draft read occurs after approval")
    # A view of actual confirmation originals for the shared native report
    # verifier. No stored fault is rewritten and no request is fabricated.
    return {"request": request, "received_response": response, "http_ordinal": posted["ordinal"], "http_status": 202,
            "before": pre, "after_response": after,
            "recovery": {**recovery, "http_ordinals": [n for n in ordinals if n > posted["ordinal"]]}}
