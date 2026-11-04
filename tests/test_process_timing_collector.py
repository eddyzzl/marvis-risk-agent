from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from marvis.orchestrator.eval.formal_acceptance import _telemetry
from marvis.orchestrator.eval.process_timing_collector import (
    TimingCollectionError,
    build_formal_observation,
    segments_from_intervals,
)
from marvis.orchestrator.eval.runtime_contracts import digest
from marvis.orchestrator.eval.runtime_process import process_observation
from marvis.process_timing import (
    observe_process_timing,
    read_process_timing,
    timed,
    timed_span,
)

MS = 1_000_000
MODEL = {"profile": "p", "model_name": "m", "source": "real_model"}
ENV = {"hardware_sha256": "a" * 64}


def _sidecar(spans, closed_ns=100 * MS, valid=True):
    rows = [{"event": "opened", "schema": "marvis.process-timing-journal.v1", "elapsed_ns": 0}]
    events = []
    for n, (phase, start, end) in enumerate(spans):
        identity = f"{n:032x}"
        events.append((start, {"event": "start", "phase": phase, "label": "stub_tool", "identity": identity, "elapsed_ns": start}))
        events.append((end, {"event": "end", "phase": phase, "label": "stub_tool", "identity": identity, "elapsed_ns": end}))
    rows += [row for _, row in sorted(events, key=lambda item: item[0])]
    rows.append({"event": "closed", "valid": valid, "elapsed_ns": closed_ns})
    return rows


def _execution(*, llm=((10, 20),), waits=(), sidecar=None, duration_ms=100, humans=0):
    llm_events = []
    for n, (start, end) in enumerate(llm):
        llm_events += [{"event": "started", "attempt_id": f"a{n}", "elapsed_ns": start * MS},
                       {"event": "finished", "attempt_id": f"a{n}", "elapsed_ns": end * MS}]
    llm_events.append({"event": "measurement_closed", "elapsed_ns": duration_ms * MS})
    backend = [{"event": "confirmation_plan_states_enabled", "elapsed_ns": 0}]
    for n, (start, end, outcome) in enumerate(waits):
        common = {"scope": "plan_confirmation_wait", "identity": f"{n:032x}", "task_sha256": "b" * 64, "binding_sha256": "c" * 64}
        backend += [{"event": "waiting", "elapsed_ns": start * MS, **common}, {"event": outcome, "elapsed_ns": end * MS, **common}]
    backend.append({"event": "closed", "elapsed_ns": duration_ms * MS, "valid": True})
    actions = [{"ordinal": i + 1, "action_sha256": f"{i:064x}", "kind": "dispatch", "phase": "dispatch", "elapsed_ns": 0}
               for i in range(humans)]
    return {
        "case_id": "c1", "duration_ms": duration_ms, "llm_events": llm_events, "backend_events": backend,
        "human_interventions": humans,
        "process_observation": process_observation(llm_events, actions, backend, plan_states=[]),
        "process_timing_events": _sidecar([]) if sidecar is None else sidecar,
    }


def _build(execution):
    return build_formal_observation(execution, attempt_id="c1-r0", case_id="c1", run_id="run", load_mode="cold_start",
                                    model=MODEL, environment=ENV)


def test_collector_output_passes_installed_formal_telemetry_reader(tmp_path):
    execution = _execution(llm=((10, 20), (60, 70)), waits=((30, 40, "resumed"),),
                           sidecar=_sidecar([("subprocess", 15 * MS, 50 * MS)]), humans=1)
    observation, coverage = _build(execution)
    raw = json.dumps(observation).encode()
    (tmp_path / "obs.json").write_bytes(raw)
    reference = {"path": "obs.json", "sha256": digest(raw)}
    facts, reason = _telemetry(
        tmp_path.resolve(), {"artifact_hashes": {"obs.json": reference["sha256"]}}, reference,
        SimpleNamespace(model=SimpleNamespace(model_dump=lambda: MODEL), environment=SimpleNamespace(model_dump=lambda: ENV)),
        SimpleNamespace(id="c1-r0", case_id="c1", load_mode="cold_start"),
        {"manifest": {"run_id": "run"}, "executions": {"c1": execution}},
        SimpleNamespace(family="feature", actions=[]),
    )
    assert reason is None
    parts = dict(facts["timing_components"])
    assert parts == {"human_wait": 0.01, "llm": 0.02, "tool": 0.02, "queue": 0.0, "overhead": 0.05}
    assert facts["wall_seconds"] == 0.1 and facts["replans"] == 0
    assert coverage["phase_totals_ms"]["tool"] == 20


def test_priority_and_merge():
    segments = segments_from_intervals(
        [(0, 50 * MS, "tool", "s"), (10 * MS, 20 * MS, "llm", "t"), (40 * MS, 60 * MS, "queue", "q"),
         (45 * MS, 48 * MS, "human_wait", "w")], 80)
    assert segments == [
        {"phase": "tool", "start_ms": 0, "end_ms": 10}, {"phase": "llm", "start_ms": 10, "end_ms": 20},
        {"phase": "tool", "start_ms": 20, "end_ms": 45}, {"phase": "human_wait", "start_ms": 45, "end_ms": 48},
        {"phase": "tool", "start_ms": 48, "end_ms": 50}, {"phase": "queue", "start_ms": 50, "end_ms": 60},
        {"phase": "overhead", "start_ms": 60, "end_ms": 80},
    ]


@pytest.mark.parametrize("mutate, reason", [
    (lambda e: e.update(process_timing_events=[{"event": "unavailable"}]), "process_timing_sidecar_unavailable"),
    (lambda e: e.pop("process_timing_events"), "process_timing_sidecar_unavailable"),
    (lambda e: e.update(process_timing_events=_sidecar([], valid=False)), "process_timing_sidecar_unsealed"),
    (lambda e: e["llm_events"].pop(), "llm_transport_incomplete"),
    (lambda e: e.update(backend_events=[{"event": "unavailable"}]), "backend_journal_unavailable"),
    (lambda e: e["process_observation"].update(schema="marvis.runtime-process-observation.v2"), "replan_observation_unavailable"),
    (lambda e: e.update(human_interventions=3), "human_action_trace_incomplete"),
])
def test_incomplete_sources_fail_closed(mutate, reason):
    execution = _execution()
    mutate(execution)
    with pytest.raises(TimingCollectionError, match=reason):
        _build(execution)


def test_censored_human_wait_fails_closed():
    with pytest.raises(TimingCollectionError, match="human_wait_censored_or_incomplete"):
        _build(_execution(waits=((30, 40, "censored"),)))


def test_measured_time_beyond_case_fails_closed():
    with pytest.raises(TimingCollectionError, match="measured_time_exceeds_case"):
        _build(_execution(sidecar=_sidecar([("tool", 10 * MS, 150 * MS)], closed_ns=150 * MS)))


def test_sidecar_journal_is_opt_in_and_sealed(tmp_path):
    @timed("tool", "stub_tool")
    def work():
        return 7

    assert work() == 7  # inactive: no journal, no error
    path = tmp_path / "timing.jsonl"
    with observe_process_timing(path, time.monotonic_ns()):
        assert work() == 7
        with pytest.raises(RuntimeError), timed_span("subprocess", "stub_process"):
            raise RuntimeError("boom")
    rows = read_process_timing(path)
    assert [r["event"] for r in rows] == ["opened", "start", "end", "start", "raised", "closed"]
    assert rows[-1]["valid"] is True
    with pytest.raises(ValueError):
        timed_span("llm", "stub_tool").__enter__()


def test_unclosed_span_invalidates_journal(tmp_path):
    path = tmp_path / "timing.jsonl"
    with observe_process_timing(path, time.monotonic_ns()) as journal:
        journal.emit("start", "tool", "stub_tool", "0" * 32)
    assert read_process_timing(path)[-1]["valid"] is False
