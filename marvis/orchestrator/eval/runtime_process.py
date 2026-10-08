"""Original process observations; partial measurements never imply a full SLO.

Offsets use the parent's monotonic clock from case start. Provider transport
time includes the gateway's request/response handling, not model-only compute.
Human records describe scripted dispatch, not a person's decision or wait time.
"""
from __future__ import annotations

import json

from marvis.runtime_observations import SCOPES, QUEUE_TERMINALS, REVISION_SCOPES, REVISION_KINDS, WAIT_SCOPES, WAIT_ENDS
from .runtime_contracts import digest


def read_backend_events(path):
    """Bound the journal and reject unexpected content before public archival."""
    try:
        if path.is_symlink() or path.stat().st_size > 4_000_000:
            raise ValueError("invalid_journal")
        events = [json.loads(line) for line in path.read_text().splitlines()]
        _validate_backend_events(events)
        return events
    except (OSError, ValueError, TypeError, KeyError):
        return [{"event": "unavailable"}]


def _validate_backend_events(events):
    if events == [{"event": "unavailable"}]:
        return
    if not isinstance(events, list) or len(events) > 20001:
        raise ValueError("invalid_backend_events")
    headers = [index for index, row in enumerate(events) if isinstance(row, dict) and row.get("event") in {"queue_terminals_enabled", "plan_revisions_enabled", "confirmation_waits_enabled", "confirmation_plan_states_enabled"}]
    if headers not in ([], [0]):
        raise ValueError("invalid_backend_header")
    revision_enabled = bool(headers) and events[0]["event"] in {"plan_revisions_enabled", "confirmation_waits_enabled", "confirmation_plan_states_enabled"}
    waits_enabled = bool(headers) and events[0]["event"] in {"confirmation_waits_enabled", "confirmation_plan_states_enabled"}
    previous = 0
    for row in events:
        if not isinstance(row, dict):
            raise ValueError("invalid_backend_event")
        kind = row.get("event")
        if kind in QUEUE_TERMINALS and not headers:
            raise ValueError("missing_backend_header")
        if kind == "closed":
            valid = set(row) == {"event", "elapsed_ns", "valid"} and type(row["valid"]) is bool
        elif kind in {"queue_terminals_enabled", "plan_revisions_enabled", "confirmation_waits_enabled", "confirmation_plan_states_enabled"}:
            valid = set(row) == {"event", "elapsed_ns"}
        elif kind == "plan_revision":
            identity = row.get("identity")
            valid = (revision_enabled and set(row) == {"event", "elapsed_ns", "identity", "revision", "kind"}
                     and isinstance(identity, str) and len(identity) == 64
                     and all(c in "0123456789abcdef" for c in identity)
                     and row["kind"] in REVISION_KINDS and type(row["revision"]) is int and row["revision"] >= 0)
        elif row.get("scope") in WAIT_SCOPES:
            identity = row.get("identity")
            hashes = (row.get("task_sha256"), row.get("binding_sha256"))
            valid = (waits_enabled and set(row) == {"event", "elapsed_ns", "scope", "identity", "task_sha256", "binding_sha256"}
                     and isinstance(identity, str) and len(identity) == 32 and all(c in "0123456789abcdef" for c in identity)
                     and all(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value) for value in hashes)
                     and kind in {"waiting"} | WAIT_ENDS)
        else:
            identity = row.get("identity")
            scope = row.get("scope")
            valid = (set(row) == {"event", "elapsed_ns", "scope", "identity"}
                     and scope in (SCOPES | REVISION_SCOPES if revision_enabled else SCOPES) and isinstance(identity, str)
                     and len(identity) == (64 if scope == "queue" else 32)
                     and all(c in "0123456789abcdef" for c in identity)
                     and kind in ({"queued", "running"} | QUEUE_TERMINALS if scope == "queue" else {"started", "returned", "raised"}))
        timestamp = row.get("elapsed_ns")
        if not valid or type(timestamp) is not int or timestamp < previous:
            raise ValueError("invalid_backend_event")
        previous = timestamp


def backend_timing(events):
    _validate_backend_events(events)
    sealed = bool(events) and events[-1].get("event") == "closed" and events[-1]["valid"]
    sealed &= sum(row.get("event") == "closed" for row in events) == 1
    result = {"journal_complete": bool(sealed), "scopes": {}}
    revision_enabled = bool(events) and events[0].get("event") in {"plan_revisions_enabled", "confirmation_waits_enabled", "confirmation_plan_states_enabled"}
    waits_enabled = bool(events) and events[0].get("event") in {"confirmation_waits_enabled", "confirmation_plan_states_enabled"}
    scopes = SCOPES | (REVISION_SCOPES if revision_enabled else set()) | (WAIT_SCOPES if waits_enabled else set())
    for scope in sorted(scopes):
        starts, ends, invalid = {}, {}, False
        for row in events:
            if row.get("scope") != scope:
                continue
            target = starts if row["event"] in {"started", "queued", "waiting"} else ends
            invalid |= row["identity"] in target
            target[row["identity"]] = row
        intervals = []
        for identity, start in starts.items():
            end = ends.get(identity)
            if end and scope in WAIT_SCOPES and any(start[key] != end[key] for key in ("task_sha256", "binding_sha256")):
                invalid = True
                continue
            if end and start["elapsed_ns"] <= end["elapsed_ns"]:
                intervals.append({"identity": identity, "start_ns": start["elapsed_ns"],
                                  "end_ns": end["elapsed_ns"], "outcome": end["event"]})
                if scope in WAIT_SCOPES:
                    intervals[-1].update({key: start[key] for key in ("task_sha256", "binding_sha256")})
        intervals.sort(key=lambda row: (row["start_ns"], row["end_ns"], row["identity"]))
        busy, cursor = 0, 0
        for row in intervals:
            busy += max(0, row["end_ns"] - max(cursor, row["start_ns"]))
            cursor = max(cursor, row["end_ns"])
        denominator = len(set(starts) | set(ends))
        result["scopes"][scope] = {
            "complete": bool(sealed and not invalid and len(intervals) == denominator),
            "observed_denominator": denominator, "measured_intervals": len(intervals),
            "unknown_intervals": denominator - len(intervals), "invalid_events": invalid,
            "known_summed_duration_ns": sum(row["end_ns"] - row["start_ns"] for row in intervals),
            "known_busy_duration_ns": busy, "intervals": intervals,
        }
        if scope in WAIT_SCOPES:
            censored = sum(row["outcome"] == "censored" for row in intervals)
            result["scopes"][scope].update(
                right_censored_intervals=censored,
                complete=bool(result["scopes"][scope]["complete"] and censored == 0),
                duration_scope="pending_external_confirmation_not_verified_human_thinking",
            )
    return result


def material_selection_identity(case):
    return digest({"kind": "validation_material_selection", "case_sha256": digest(case.model_dump())})


def llm_transport_timing(events):
    starts, finishes, invalid = {}, {}, False
    seals = []
    for event in events:
        kind = event.get("event")
        if kind == "measurement_closed":
            seals.append(event.get("elapsed_ns"))
        if kind not in {"started", "finished"}:
            invalid |= kind == "incomplete_receipt"
            continue
        identity = event.get("attempt_id")
        target = starts if kind == "started" else finishes
        if not isinstance(identity, str) or not identity or identity in target:
            invalid = True
            continue
        target[identity] = event.get("elapsed_ns")
    intervals = []
    for identity, start in starts.items():
        end = finishes.get(identity)
        if type(start) is int and type(end) is int and 0 <= start <= end:
            intervals.append({"attempt_id": identity, "start_ns": start, "end_ns": end})
    intervals.sort(key=lambda row: (row["start_ns"], row["end_ns"], row["attempt_id"]))
    busy, cursor = 0, 0
    for row in intervals:
        busy += max(0, row["end_ns"] - max(cursor, row["start_ns"]))
        cursor = max(cursor, row["end_ns"])
    sealed = (len(seals) == 1 and type(seals[0]) is int and seals[0] >= 0
              and all(row["end_ns"] <= seals[0] for row in intervals))
    complete = sealed and not invalid and set(starts) == set(finishes) and len(intervals) == len(starts)
    return {
        "scope": "metered_provider_transport_not_model_compute",
        "complete": complete,
        "attempt_denominator": len(set(starts) | set(finishes)),
        "measured_attempts": len(intervals),
        "unknown_attempts": len(set(starts) | set(finishes)) - len(intervals),
        "invalid_events": invalid,
        "known_summed_duration_ns": sum(row["end_ns"] - row["start_ns"] for row in intervals),
        "known_busy_duration_ns": busy,
        "intervals": intervals,
    }


def process_observation(events, human_actions, backend_events=None, *, plan_states=None):
    result = {
        "schema": "marvis.runtime-process-observation.v1",
        "clock": "parent_monotonic_ns_from_case_start",
        "llm_transport": llm_transport_timing(events),
        "human_actions": human_actions,
        "human_action_scope": "scripted_dispatch_attempts_not_human_wait_or_success",
        "unmeasured": ["human_wait", "tool", "queue", "replans"],
        "formal_timing_complete": False,
    }
    if backend_events is not None:
        result.update(
            schema="marvis.runtime-process-observation.v2",
            backend=backend_timing(backend_events),
            backend_scope="enclosing_call_wall_time_and_postcommit_queue_transitions",
            overlapping_scopes=True,
            unmeasured=["human_wait", "uninstrumented_tools", "queue_terminal_without_running", "replans"],
        )
        if any(row.get("event") in {"queue_terminals_enabled", "plan_revisions_enabled", "confirmation_waits_enabled", "confirmation_plan_states_enabled"} for row in backend_events):
            result["schema"] = "marvis.runtime-process-observation.v3"
            result["unmeasured"] = ["human_wait", "uninstrumented_tools", "replans"]
        if backend_events and backend_events[0].get("event") in {"plan_revisions_enabled", "confirmation_waits_enabled", "confirmation_plan_states_enabled"}:
            from .runtime_revisions import revision_observation
            result["schema"] = "marvis.runtime-process-observation.v4"
            result["revisions"] = revision_observation(backend_events, plan_states, journal_complete=result["backend"]["journal_complete"])
            result["unmeasured"] = ["human_wait", "uninstrumented_tools"]
        if backend_events and backend_events[0].get("event") in {"confirmation_waits_enabled", "confirmation_plan_states_enabled"}:
            result["schema"] = "marvis.runtime-process-observation.v5"
            result["confirmation_wait_scope"] = "native_confirmable_state_after_execution_until_work_resumes_or_state_changes"
            result["confirmation_wait_entry_coverage"] = ["task_job_completion", "report_draft_publication", "report_draft_save"]
            result["unmeasured"].append("confirmation_states_without_instrumented_entry")
            if backend_events[0].get("event") == "confirmation_plan_states_enabled":
                result["schema"] = "marvis.runtime-process-observation.v6"
                result["confirmation_wait_entry_coverage"].extend([
                    "committed_plan_and_step_state_changes", "committed_governance_decisions",
                ])
    return result


def validate_process_observation(observed, events, *, interventions, duration_ms, backend_events=None, plan_states=None):
    """Recompute only new carriers; callers preserve older unmeasured runs."""
    actions = observed.get("human_actions") if isinstance(observed, dict) else None
    if (not isinstance(actions, list) or type(interventions) is not int
            or not 0 <= interventions <= 10000 or len(actions) != interventions
            or type(duration_ms) is not int or duration_ms < 0):
        raise ValueError("invalid_original_process_observation")
    cursor = 0
    for ordinal, action in enumerate(actions, 1):
        if (not isinstance(action, dict) or set(action) != {"ordinal", "action_sha256", "kind", "phase", "elapsed_ns"}
                or type(action["ordinal"]) is not int or action["ordinal"] != ordinal
                or not isinstance(action["action_sha256"], str) or len(action["action_sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in action["action_sha256"])
                or not isinstance(action["kind"], str) or not 0 < len(action["kind"]) <= 80
                or action["phase"] not in {"dispatch", "dataset_selection", "label_declaration", "material_upload", "monitoring_proposal"}
                or type(action["elapsed_ns"]) is not int
                or not cursor <= action["elapsed_ns"] < (duration_ms + 1) * 1_000_000):
            raise ValueError("invalid_original_human_action")
        cursor = action["elapsed_ns"]
    if digest(observed) != digest(process_observation(events, actions, backend_events, plan_states=plan_states)):
        raise ValueError("original_process_observation_mismatch")
    if backend_events is not None:
        for row in backend_events:
            if row.get("elapsed_ns", 0) >= (duration_ms + 1) * 1_000_000:
                raise ValueError("original_backend_time_exceeds_case")
    for interval in observed["llm_transport"]["intervals"]:
        if interval["end_ns"] >= (duration_ms + 1) * 1_000_000:
            raise ValueError("original_transport_time_exceeds_case")
