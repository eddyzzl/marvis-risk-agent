"""Build the formal gate's process observation from one original execution.

Inputs are the archived execution record only: the parent's LLM transport
intervals, the backend journal (tool/queue/confirmation-wait scopes, plan
revisions) and the side-car ``process_timing_events`` (subprocess, step tool,
operator wait). Any incomplete source fails closed; missing spans are never
inferred. ``overhead`` is the residual not covered by any measured scope and
the coverage companion lists what that residual may still contain.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from marvis.process_timing import validate_process_timing
from marvis.runtime_observations import WAIT_SCOPES

from .runtime_process import backend_timing, llm_transport_timing

SCHEMA = "marvis.formal-process-observation.v1"
COVERAGE_SCHEMA = "marvis.formal-process-coverage.v1"
# Highest first. Waits exclude other work; transport is the narrowest span;
# step/tool scopes enclose nested transport; queue intervals enclose running.
PRIORITY = ("human_wait", "llm", "tool", "queue")
TOOL_SCOPES = ("plugin", "adhoc", "validation_scan", "validation_notebook",
               "validation_pmml", "validation_metrics", "validation_report")
REVISION_SCHEMAS = {"marvis.runtime-process-observation.v4", "marvis.runtime-process-observation.v5",
                    "marvis.runtime-process-observation.v6"}
SIDECAR_PHASE = {"human_wait": "human_wait", "subprocess": "tool", "tool": "tool"}


class TimingCollectionError(ValueError):
    """Fixed reason codes; never carries record content."""


def _require(condition, reason):
    if not condition:
        raise TimingCollectionError(reason)


def _sidecar_intervals(rows):
    try:
        validate_process_timing(rows)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise TimingCollectionError("process_timing_sidecar_unavailable") from exc
    _require(rows[-1].get("event") == "closed" and rows[-1]["valid"] is True, "process_timing_sidecar_unsealed")
    starts, intervals = {}, []
    for row in rows[1:-1]:
        if row["event"] == "start":
            _require(row["identity"] not in starts, "process_timing_sidecar_duplicate_span")
            starts[row["identity"]] = row
        else:
            start = starts.pop(row["identity"], None)
            _require(start is not None and start["phase"] == row["phase"], "process_timing_sidecar_unpaired_span")
            intervals.append((start["elapsed_ns"], row["elapsed_ns"], SIDECAR_PHASE[row["phase"]], row["phase"]))
    _require(not starts, "process_timing_sidecar_open_span")
    return intervals


def _measured_intervals(execution):
    # A crashed epoch is censored; it cannot be replaced by the later service
    # journal or silently omitted from the recovery latency denominator.
    _require(not execution.get("process_epochs"), "multi_epoch_process_timing_incomplete")
    process = execution.get("process_observation")
    _require(isinstance(process, dict), "process_observation_unavailable")
    _require(process.get("schema") in REVISION_SCHEMAS, "replan_observation_unavailable")
    revisions = process.get("revisions") or {}
    _require(revisions.get("complete") is True
             and type(revisions.get("known_counts", {}).get("structural_replan")) is int,
             "replan_observation_unavailable")
    transport = llm_transport_timing(execution.get("llm_events") or [])
    _require(transport["complete"] and transport == process.get("llm_transport"), "llm_transport_incomplete")
    events = execution.get("backend_events")
    _require(isinstance(events, list) and events and events != [{"event": "unavailable"}], "backend_journal_unavailable")
    try:
        backend = backend_timing(events)
    except (ValueError, TypeError, KeyError) as exc:
        raise TimingCollectionError("backend_journal_unavailable") from exc
    _require(backend["journal_complete"], "backend_journal_incomplete")
    intervals = [(row["start_ns"], row["end_ns"], "llm", "llm_transport") for row in transport["intervals"]]
    for scope, phase in [*((s, "tool") for s in TOOL_SCOPES), ("queue", "queue"),
                         *((s, "human_wait") for s in sorted(WAIT_SCOPES))]:
        observed = backend["scopes"].get(scope)
        if observed is None:
            continue
        _require(observed["complete"], "human_wait_censored_or_incomplete" if phase == "human_wait" else f"{scope}_incomplete")
        intervals.extend((row["start_ns"], row["end_ns"], phase, scope) for row in observed["intervals"])
    sidecar = execution.get("process_timing_events")
    _require(isinstance(sidecar, list), "process_timing_sidecar_unavailable")
    intervals.extend(_sidecar_intervals(sidecar))
    return process, intervals


def segments_from_intervals(intervals, duration_ms):
    """Partition [0, duration_ms] into disjoint phases by fixed priority."""
    _require(type(duration_ms) is int and duration_ms > 0, "original_duration_unavailable")
    limit_ns = (duration_ms + 1) * 1_000_000
    spans = []
    for start_ns, end_ns, phase, _source in intervals:
        _require(type(start_ns) is int and type(end_ns) is int and 0 <= start_ns <= end_ns < limit_ns,
                 "measured_time_exceeds_case")
        start, end = min(round(start_ns / 1e6), duration_ms), min(round(end_ns / 1e6), duration_ms)
        if end > start:
            spans.append((start, end, PRIORITY.index(phase)))
    changes = {}
    for start, end, rank in spans:
        changes.setdefault(start, [0] * len(PRIORITY))[rank] += 1
        changes.setdefault(end, [0] * len(PRIORITY))[rank] -= 1
    points = sorted({0, duration_ms, *changes})
    active, segments = [0] * len(PRIORITY), []
    for left, right in zip(points, points[1:]):
        for rank, delta in enumerate(changes.get(left, ())):
            active[rank] += delta
        phase = next((PRIORITY[rank] for rank, count in enumerate(active) if count > 0), "overhead")
        if segments and segments[-1]["phase"] == phase:
            segments[-1]["end_ms"] = right
        else:
            segments.append({"phase": phase, "start_ms": left, "end_ms": right})
    _require(0 < len(segments) <= 10000, "too_many_process_segments")
    return segments


def build_formal_observation(execution, *, attempt_id, case_id, run_id, load_mode, model, environment):
    """Return ``(observation, coverage)`` in the exact formal-gate format."""
    _require(isinstance(execution, dict) and execution.get("case_id") == case_id, "case_identity_mismatch")
    duration = execution.get("duration_ms")
    process, intervals = _measured_intervals(execution)
    # Always validate every original interval before applying a frozen window.
    segments = segments_from_intervals(intervals, duration)
    original_duration = duration
    window = None
    if execution.get("load_originals") is not None:
        from .runtime_warmup import measurement_window
        window = measurement_window(execution, load_mode=load_mode, frozen_load_sha256=environment["load_sha256"])
        start_ns, end_ns = window["start_ns"], window["end_ns"]
        duration = math.ceil((end_ns - start_ns) / 1_000_000)
        cropped = [(max(start, start_ns) - start_ns, min(end, end_ns) - start_ns, phase, source)
                   for start, end, phase, source in intervals if end > start_ns and start < end_ns]
        segments = segments_from_intervals(cropped, duration)
    humans = [action["action_sha256"] for action in process.get("human_actions") or []]
    _require(type(execution.get("human_interventions")) is int and len(humans) == execution["human_interventions"],
             "human_action_trace_incomplete")
    observation = {
        "schema": SCHEMA,
        "attempt_id": attempt_id,
        "run_id": run_id,
        "case_id": case_id,
        "model": model,
        "environment": environment,
        "load_mode": load_mode,
        "segments": segments,
        "human_action_sha256s": humans,
        "replans": process["revisions"]["known_counts"]["structural_replan"],
    }
    if window is not None:
        observation.update(schema="marvis.formal-process-observation.v2", measurement_window=window)
    totals = dict.fromkeys((*PRIORITY, "overhead"), 0)
    for row in segments:
        totals[row["phase"]] += row["end_ms"] - row["start_ms"]
    coverage = {
        "schema": COVERAGE_SCHEMA,
        "attempt_id": attempt_id,
        "duration_ms": duration,
        "original_duration_ms": original_duration,
        "measurement_scope": window["scope"] if window else "whole_original_case",
        "phase_totals_ms": totals,
        "sources": sorted({source for *_, source in intervals}),
        "priority": [*PRIORITY, "overhead"],
        "residual_overhead_may_include": [
            "confirmation_states_without_instrumented_entry",
            "clarification_reply_wait_without_operator_span",
            "in_process_work_outside_plan_steps_and_validation_scopes",
        ],
        "human_wait_scope": "platform_confirmation_waits_and_operator_spans_not_verified_human_thinking",
        "acceptance_claim": "not_established_by_collection",
    }
    return observation, coverage


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write a formal process observation for one attempt.")
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--load-mode", required=True, choices=("cold_start", "steady", "concurrent", "recovery"))
    parser.add_argument("--model-json", type=Path, required=True, help="frozen benchmark model block")
    parser.add_argument("--environment-json", type=Path, required=True, help="frozen benchmark environment block")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    execution = json.loads(args.execution.read_text(encoding="utf-8"))
    try:
        observation, coverage = build_formal_observation(
            execution, attempt_id=args.attempt_id, case_id=execution.get("case_id"), run_id=args.run_id,
            load_mode=args.load_mode, model=json.loads(args.model_json.read_text(encoding="utf-8")),
            environment=json.loads(args.environment_json.read_text(encoding="utf-8")),
        )
    except TimingCollectionError as exc:
        print(json.dumps({"status": "unavailable", "reason": str(exc)}))
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    for suffix, value in (("process-observation", observation), ("coverage", coverage)):
        path = args.out / f"{args.attempt_id}.{suffix}.json"
        with open(path, "x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
    print(json.dumps({"status": "written", "attempt_id": args.attempt_id}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
