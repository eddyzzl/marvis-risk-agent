from __future__ import annotations

from dataclasses import dataclass
import sqlite3

from marvis.agent_memory.capture import save_memory_candidate
from marvis.agent_memory.models import MemoryCandidate
from marvis.memory_policy import MemoryPolicySettings, save_memory_policy


@dataclass
class _Entry:
    id: str = "memory-1"
    memory_type: str = "field_convention"
    status: str = "active"


class _RecordingStore:
    def __init__(self, events, *, entry=None):
        self.events = events
        self.entry = entry or _Entry()

    def create(self, candidate, *, task_id=None):
        self.events.append(("store.create", candidate.memory_type, task_id))
        return self.entry


class _RecordingDispatcher:
    def __init__(self, events, *, fail_event=None):
        self.events = events
        self.fail_event = fail_event

    def dispatch(self, event, payload, *, task_id):
        self.events.append((event, payload, task_id))
        if event == self.fail_event:
            raise RuntimeError("synthetic hook failure")
        return []


def _candidate() -> MemoryCandidate:
    return MemoryCandidate(
        memory_type="field_convention",
        summary="sensitive summary must not enter hooks",
        payload={"target_col": "bad_flag", "raw": "must-not-enter-hooks"},
        source_task_id="task-source",
        source_message_id="message-secret",
        confidence="high",
        reason="operator supplied",
    )


def test_memory_save_dispatches_sanitized_before_and_after_hooks_in_order():
    events = []
    entry = save_memory_candidate(
        _RecordingStore(events),
        _candidate(),
        task_id="task-1",
        hook_dispatcher=_RecordingDispatcher(events),
    )

    assert entry.id == "memory-1"
    assert [event[0] for event in events] == [
        "memory.before_save",
        "store.create",
        "memory.after_save",
    ]
    before_payload = events[0][1]
    assert before_payload == {
        "task_id": "task-1",
        "memory_type": "field_convention",
        "confidence": "high",
    }
    after_payload = events[2][1]
    assert after_payload == {
        "task_id": "task-1",
        "memory_id": "memory-1",
        "memory_type": "field_convention",
        "status": "active",
    }


def test_before_hook_failure_does_not_block_save_and_rejected_entry_has_no_after_hook():
    events = []
    entry = _Entry(status="rejected")

    saved = save_memory_candidate(
        _RecordingStore(events, entry=entry),
        _candidate(),
        task_id="task-1",
        hook_dispatcher=_RecordingDispatcher(
            events,
            fail_event="memory.before_save",
        ),
    )

    assert saved.status == "rejected"
    assert [event[0] for event in events] == [
        "memory.before_save",
        "store.create",
    ]


def test_pipeline_failure_capture_uses_the_governed_hook_boundary(
    tmp_path,
    monkeypatch,
):
    import marvis.pipeline as pipeline

    save_memory_policy(
        tmp_path,
        MemoryPolicySettings(reference_cross_task=True, auto_distill=True),
    )
    events = []
    store = _RecordingStore(events)
    monkeypatch.setattr(pipeline, "AgentMemoryStore", lambda *_args, **_kwargs: store)
    monkeypatch.setattr(
        pipeline,
        "extract_validation_pitfall",
        lambda _payload: [_candidate()],
    )
    monkeypatch.setattr(pipeline, "extract_task_experience", lambda _payload: None)

    pipeline._capture_agent_memory_for_failure(
        repo=type("Repo", (), {"db_path": tmp_path / "marvis.sqlite"})(),
        task_id="task-1",
        failure_kind="metrics",
        message="boom",
        hook_dispatcher=_RecordingDispatcher(events),
    )

    assert [event[0] for event in events] == [
        "memory.before_save",
        "store.create",
        "memory.after_save",
    ]


def test_pipeline_failure_capture_does_not_mask_stage_failure_when_memory_is_locked(
    tmp_path,
    monkeypatch,
):
    import marvis.pipeline as pipeline

    save_memory_policy(
        tmp_path,
        MemoryPolicySettings(reference_cross_task=True, auto_distill=True),
    )

    class _LockedStore:
        def list_entries(self, **_kwargs):
            return [_Entry()]

        def record_negative_feedback(self, *_args, **_kwargs):
            raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(
        pipeline,
        "AgentMemoryStore",
        lambda *_args, **_kwargs: _LockedStore(),
    )
    monkeypatch.setattr(pipeline, "extract_validation_pitfall", lambda _payload: [])
    monkeypatch.setattr(pipeline, "extract_task_experience", lambda _payload: None)

    # This call runs from a stage's exception handler.  Its best-effort
    # downgrade must not replace the original notebook/metrics/report error.
    pipeline._capture_agent_memory_for_failure(
        repo=type("Repo", (), {"db_path": tmp_path / "marvis.sqlite"})(),
        task_id="task-1",
        failure_kind="metrics",
        message="original metrics failure",
    )
