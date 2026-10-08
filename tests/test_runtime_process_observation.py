"""Timing arithmetic and source consistency, without invented formal SLOs."""
from copy import deepcopy
import json
import time

import pytest

from marvis.orchestrator.eval.runtime_process import (
    llm_transport_timing, process_observation, validate_process_observation,
)
from marvis.orchestrator.eval.runtime_runner import AttemptObserver


def events():
    return [
        {"event": "started", "attempt_id": "a", "elapsed_ns": 10},
        {"event": "started", "attempt_id": "b", "elapsed_ns": 20},
        {"event": "finished", "attempt_id": "b", "elapsed_ns": 40},
        {"event": "finished", "attempt_id": "a", "elapsed_ns": 30},
        {"event": "measurement_closed", "elapsed_ns": 50},
    ]


def test_concurrent_calls_keep_individual_time_and_union_without_double_counting():
    measured = llm_transport_timing(events())
    assert measured["complete"] is True
    assert measured["known_summed_duration_ns"] == 40
    assert measured["known_busy_duration_ns"] == 30
    assert measured["attempt_denominator"] == measured["measured_attempts"] == 2


@pytest.mark.parametrize("change", ["unfinished", "legacy", "duplicate", "boolean", "backwards", "unsealed", "orphan", "early_seal", "duplicate_seal"])
def test_incomplete_or_invalid_clocks_cannot_claim_complete_measurement(change):
    source = events()
    if change == "unfinished":
        source.pop(3)
    elif change == "legacy":
        for row in source:
            row.pop("elapsed_ns")
    elif change == "duplicate":
        source.append(source[0].copy())
    elif change == "boolean":
        source[0]["elapsed_ns"] = False
    elif change == "backwards":
        source[3]["elapsed_ns"] = 1
    elif change == "unsealed":
        source.pop()
    elif change == "orphan":
        source.pop(0)
    elif change == "early_seal":
        source[-1]["elapsed_ns"] = 25
    else:
        source.append(source[-1].copy())
    assert llm_transport_timing(source)["complete"] is False


def test_observer_records_parent_clock_and_does_not_accept_supplied_timestamps(tmp_path):
    path = tmp_path / "attempts.jsonl"
    observer = AttemptObserver(path, max_attempts=2, deadline=time.monotonic() + 10)
    identity = observer.before_attempt({"logical_call_id": "one", "elapsed_ns": -100})
    observer.after_attempt(identity, {"elapsed_ns": -200})
    observer.seal()
    source = [json.loads(line) for line in path.read_text().splitlines()]
    assert 0 <= source[0]["elapsed_ns"] <= source[1]["elapsed_ns"] <= source[2]["elapsed_ns"]
    assert llm_transport_timing(source)["complete"] is True


@pytest.mark.parametrize("change", [None, "claim_complete", "changed_time", "lost_action", "future_action"])
def test_original_process_summary_is_recomputed_and_partial_stays_partial(change):
    source = events()
    actions = [{"ordinal": 1, "action_sha256": "a" * 64, "kind": "message", "phase": "dispatch", "elapsed_ns": 60}]
    observed = process_observation(source, actions)
    if change is None:
        validate_process_observation(observed, source, interventions=1, duration_ms=1)
        assert observed["formal_timing_complete"] is False
        assert set(observed["unmeasured"]) == {"human_wait", "tool", "queue", "replans"}
        return
    changed = deepcopy(observed)
    if change == "claim_complete":
        changed["formal_timing_complete"] = True
    elif change == "changed_time":
        changed["llm_transport"]["known_busy_duration_ns"] += 1
    elif change == "lost_action":
        changed["human_actions"].clear()
    else:
        changed["human_actions"][0]["elapsed_ns"] = 10_000_000
    with pytest.raises(ValueError):
        validate_process_observation(changed, source, interventions=1, duration_ms=1)
