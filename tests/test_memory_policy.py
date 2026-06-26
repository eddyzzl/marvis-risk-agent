from pathlib import Path
from types import SimpleNamespace

import marvis.api as api
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
