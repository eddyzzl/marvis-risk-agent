from pathlib import Path
from types import SimpleNamespace

import marvis.api as api
from marvis.db import init_db
from marvis.memory_policy import (
    MemoryPolicySettings,
    _memory_policy_path,
    load_memory_policy,
    save_memory_policy,
)


def test_load_returns_defaults_when_file_absent(tmp_path: Path):
    settings = load_memory_policy(tmp_path)

    assert settings == MemoryPolicySettings()
    assert settings.reference_cross_task is True
    assert settings.auto_distill is True
    # Create-on-read safe: loading must not write a settings file.
    assert not _memory_policy_path(tmp_path).exists()


def test_save_then_load_round_trips_both_flags(tmp_path: Path):
    settings = MemoryPolicySettings(reference_cross_task=False, auto_distill=False)

    saved = save_memory_policy(tmp_path, settings)

    assert saved == settings
    assert load_memory_policy(tmp_path) == settings

    mixed = MemoryPolicySettings(reference_cross_task=True, auto_distill=False)
    save_memory_policy(tmp_path, mixed)
    assert load_memory_policy(tmp_path) == mixed


def test_load_is_robust_to_corrupt_json(tmp_path: Path):
    path = _memory_policy_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")

    assert load_memory_policy(tmp_path) == MemoryPolicySettings()


def test_load_is_robust_to_empty_file(tmp_path: Path):
    path = _memory_policy_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")

    assert load_memory_policy(tmp_path) == MemoryPolicySettings()


def test_load_ignores_non_object_json(tmp_path: Path):
    path = _memory_policy_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[1, 2, 3]", encoding="utf-8")

    assert load_memory_policy(tmp_path) == MemoryPolicySettings()


def test_load_falls_back_to_defaults_for_non_bool_values(tmp_path: Path):
    path = _memory_policy_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"reference_cross_task": "nope", "auto_distill": 0}',
        encoding="utf-8",
    )

    # Garbage non-bool values must not silently flip a flag off; fall back to
    # the safe default (on) so behavior matches a fresh workspace.
    assert load_memory_policy(tmp_path) == MemoryPolicySettings()


def test_context_gate_returns_none_when_reference_cross_task_off(tmp_path: Path):
    # The gate short-circuits on the policy before touching the store, so a fake
    # store carrying only db_path (workspace/marvis.sqlite) is sufficient.
    save_memory_policy(
        tmp_path,
        MemoryPolicySettings(reference_cross_task=False, auto_distill=True),
    )
    fake_store = SimpleNamespace(db_path=tmp_path / "marvis.sqlite")
    fake_task = SimpleNamespace(model_name="m", id="task-1")

    result = api._agent_memory_context_from_store(
        fake_store,
        fake_task,
        stage="metrics",
    )

    assert result is None


def test_capture_gate_skips_when_auto_distill_off(tmp_path: Path, monkeypatch):
    save_memory_policy(
        tmp_path,
        MemoryPolicySettings(reference_cross_task=True, auto_distill=False),
    )

    called = {"extract": False}

    def _fail_extract(*_args, **_kwargs):
        called["extract"] = True
        raise AssertionError("auto_distill OFF must not capture")

    monkeypatch.setattr(api, "extract_user_preference", _fail_extract)
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(settings=SimpleNamespace(workspace=tmp_path))
        )
    )

    api._capture_user_preference_memory(request, "task-1", {"content": "x", "id": "1"})

    assert called["extract"] is False


def test_capture_consults_policy_when_auto_distill_on(tmp_path: Path, monkeypatch):
    save_memory_policy(
        tmp_path,
        MemoryPolicySettings(reference_cross_task=True, auto_distill=True),
    )

    called = {"extract": False}

    def _extract(*_args, **_kwargs):
        called["extract"] = True
        return None  # no candidate -> no store write

    monkeypatch.setattr(api, "extract_user_preference", _extract)
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(settings=SimpleNamespace(workspace=tmp_path))
        )
    )

    api._capture_user_preference_memory(request, "task-1", {"content": "x", "id": "1"})

    assert called["extract"] is True


def test_capture_dispatches_memory_after_save_when_hook_dispatcher_present(tmp_path: Path):
    init_db(tmp_path / "marvis.sqlite")
    save_memory_policy(
        tmp_path,
        MemoryPolicySettings(reference_cross_task=True, auto_distill=True),
    )
    dispatched = []
    fake_dispatcher = SimpleNamespace(
        dispatch=lambda event, payload, *, task_id: dispatched.append((event, payload, task_id))
    )
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                settings=SimpleNamespace(workspace=tmp_path, db_path=tmp_path / "marvis.sqlite"),
                hook_dispatcher=fake_dispatcher,
            )
        )
    )

    api._capture_user_preference_memory(
        request,
        "task-1",
        {"content": "请记住：优先用KS指标对比。", "id": "msg-1"},
    )

    assert len(dispatched) == 1
    event, payload, task_id = dispatched[0]
    assert event == "memory.after_save"
    assert payload["task_id"] == "task-1"
    assert payload["memory_type"] == "user_preference"
    assert task_id == "task-1"


def test_capture_does_not_dispatch_when_no_candidate_extracted(tmp_path: Path, monkeypatch):
    save_memory_policy(
        tmp_path,
        MemoryPolicySettings(reference_cross_task=True, auto_distill=True),
    )
    dispatched = []
    fake_dispatcher = SimpleNamespace(
        dispatch=lambda event, payload, *, task_id: dispatched.append((event, payload, task_id))
    )
    monkeypatch.setattr(api, "extract_user_preference", lambda *_a, **_k: None)
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                settings=SimpleNamespace(workspace=tmp_path, db_path=tmp_path / "marvis.sqlite"),
                hook_dispatcher=fake_dispatcher,
            )
        )
    )

    api._capture_user_preference_memory(request, "task-1", {"content": "x", "id": "1"})

    assert dispatched == []


def test_pipeline_metrics_success_capture_skipped_when_auto_distill_off(
    tmp_path: Path,
):
    # The pipeline is a SECOND automatic-capture surface; auto_distill OFF must
    # also disable it. The gate returns before touching the repo at all.
    import marvis.pipeline as pipeline

    save_memory_policy(
        tmp_path,
        MemoryPolicySettings(reference_cross_task=True, auto_distill=False),
    )
    touched = {"repo": False}

    def _get_task(*_args, **_kwargs):
        touched["repo"] = True
        return SimpleNamespace(model_name="m", id="task-1")

    fake_repo = SimpleNamespace(db_path=tmp_path / "marvis.sqlite", get_task=_get_task)

    pipeline._capture_agent_memory_for_metrics_success(
        repo=fake_repo, task_id="task-1", outputs_dir=tmp_path
    )

    assert touched["repo"] is False


def test_pipeline_failure_capture_skipped_when_auto_distill_off(
    tmp_path: Path, monkeypatch
):
    import marvis.pipeline as pipeline

    save_memory_policy(
        tmp_path,
        MemoryPolicySettings(reference_cross_task=True, auto_distill=False),
    )
    called = {"extract": False}

    def _mark(*_args, **_kwargs):
        called["extract"] = True
        return []

    monkeypatch.setattr(
        pipeline, "AgentMemoryStore", lambda *a, **k: SimpleNamespace(create=lambda *a, **k: None)
    )
    monkeypatch.setattr(pipeline, "extract_validation_pitfall", _mark)
    monkeypatch.setattr(pipeline, "extract_task_experience", lambda *a, **k: None)
    fake_repo = SimpleNamespace(db_path=tmp_path / "marvis.sqlite")

    pipeline._capture_agent_memory_for_failure(
        repo=fake_repo, task_id="task-1", failure_kind="metrics", message="boom"
    )

    assert called["extract"] is False


def test_pipeline_failure_capture_runs_when_auto_distill_on(
    tmp_path: Path, monkeypatch
):
    import marvis.pipeline as pipeline

    save_memory_policy(
        tmp_path,
        MemoryPolicySettings(reference_cross_task=True, auto_distill=True),
    )
    called = {"extract": False}

    def _mark(*_args, **_kwargs):
        called["extract"] = True
        return []

    monkeypatch.setattr(
        pipeline, "AgentMemoryStore", lambda *a, **k: SimpleNamespace(create=lambda *a, **k: None)
    )
    monkeypatch.setattr(pipeline, "extract_validation_pitfall", _mark)
    monkeypatch.setattr(pipeline, "extract_task_experience", lambda *a, **k: None)
    fake_repo = SimpleNamespace(db_path=tmp_path / "marvis.sqlite")

    pipeline._capture_agent_memory_for_failure(
        repo=fake_repo, task_id="task-1", failure_kind="metrics", message="boom"
    )

    assert called["extract"] is True


def test_pipeline_failure_capture_downgrades_prior_task_memory(tmp_path: Path):
    # MEM-7 negative feedback loop, exercised end to end (real store, no
    # monkeypatching): an active field_convention memory captured earlier for
    # this task must have its confidence stepped down and a negative_feedback
    # audit event recorded once the task reaches its FAILED terminal state --
    # a stale prior tied to a failed run is worse than no prior at all.
    import marvis.pipeline as pipeline
    from marvis.agent_memory.models import MemoryCandidate
    from marvis.agent_memory.store import AgentMemoryStore
    from marvis.db import init_db

    db_path = tmp_path / "marvis.sqlite"
    init_db(db_path)
    save_memory_policy(
        tmp_path,
        MemoryPolicySettings(reference_cross_task=True, auto_distill=True),
    )
    store = AgentMemoryStore(db_path)
    prior = store.create(
        MemoryCandidate(
            memory_type="field_convention",
            summary="字段口径：目标字段=bad_flag",
            payload={"target_col": "bad_flag"},
            source_task_id="task-1",
            confidence="high",
        ),
        task_id="task-1",
    )
    fake_repo = SimpleNamespace(db_path=db_path)

    pipeline._capture_agent_memory_for_failure(
        repo=fake_repo, task_id="task-1", failure_kind="metrics", message="boom"
    )

    downgraded = store.get_entry(prior.id, audit=False)
    assert downgraded.confidence == "medium"
    events = store.list_events(prior.id)
    negative_events = [event for event in events if event["event_type"] == "negative_feedback"]
    assert len(negative_events) == 1
    assert negative_events[0]["task_id"] == "task-1"
    assert negative_events[0]["details"]["reason"] == "task_failed:metrics"

    # The failure-record candidates this same call creates (task_experience /
    # validation_pitfall describing *this* failure) must NOT be immediately
    # self-downgraded -- the downgrade pass runs before they are created.
    failure_records = [
        entry
        for entry in store.list_entries(source_task_id="task-1", limit=50)
        if entry.id != prior.id
    ]
    assert failure_records
    task_experience_entries = [e for e in failure_records if e.memory_type == "task_experience"]
    assert task_experience_entries
    assert all(e.confidence == "medium" for e in task_experience_entries)
    assert all(
        not any(ev["event_type"] == "negative_feedback" for ev in store.list_events(e.id))
        for e in task_experience_entries
    )
