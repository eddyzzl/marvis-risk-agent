import asyncio
import io
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from marvis.db import PluginRepository, init_db
from marvis.governance.errors import ApprovalBindingError, ApprovalStateError
from marvis.job_cancellation import JobCancellationToken
from marvis.plugins.contracts import PROTOCOL_VERSION, WORKER_RESULT_SENTINEL
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import (
    EffectTargetPolicy,
    GovernancePolicy,
    PluginManifest,
    ToolRef,
    ToolSpec,
)
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import (
    _WORKER_ENV_ALLOWLIST,
    _parse_worker_result,
    _run_worker,
    ToolContext,
    ToolRunner,
    WorkerTimeoutExpired,
)


def _runner(tmp_path):
    runner, _repo = _runtime(tmp_path)
    return runner


def test_plugin_worker_result_sentinel_survives_native_stdout_after_protocol():
    payload = {"ok": True, "worker_protocol_version": PROTOCOL_VERSION, "output": {}}
    stdout = "\n".join(
        [
            "native library startup",
            WORKER_RESULT_SENTINEL + json.dumps(payload),
            "native library shutdown",
        ]
    )

    assert _parse_worker_result(stdout) == payload


def _runtime(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PluginRepository(db_path)
    registry = PluginRegistry(repo)
    packs_root = Path(__file__).parents[1] / "marvis" / "packs"
    load_builtin_packs(registry, packs_root)
    runner = ToolRunner(
        ToolRegistry(registry),
        repo,
        python_executable=sys.executable,
        datasets_root=tmp_path / "datasets",
        workspace=tmp_path / "workspace",
    )
    return runner, repo


def test_train_models_dispatch_uses_manifest_timeout_for_wall_cpu_and_audit(
    tmp_path,
    monkeypatch,
):
    """A heavy modeling batch must never fall back to the generic 600s ceiling."""

    runner, repo = _runtime(tmp_path)
    observed = {}
    monkeypatch.setattr("marvis.plugins.runner._available_cpu_count", lambda: 1)

    def fake_run_worker(_python, job, *, timeout, **_kwargs):
        observed["timeout"] = timeout
        observed["cpu_limit_seconds"] = job["cpu_limit_seconds"]
        output = {
            "experiments": [
                {"experiment_id": "exp-1", "recipe": "lgb", "metrics": {}}
            ],
            "experiment_ids": ["exp-1"],
            "best_experiment_id": "exp-1",
            "best_recipe": "lgb",
            "target_type": "binary",
            "selection_metric": "test_ks(overfit-penalized)",
            "eval_metric": "ks_auc",
            "failed": [],
        }
        payload = {
            "ok": True,
            "worker_protocol_version": PROTOCOL_VERSION,
            "output": output,
        }
        return subprocess.CompletedProcess(
            ["worker"],
            0,
            stdout=WORKER_RESULT_SENTINEL + json.dumps(payload),
            stderr="",
        )

    monkeypatch.setattr("marvis.plugins.runner._run_worker", fake_run_worker)
    result = runner.invoke(
        ToolRef("modeling", "train_models"),
        {
            "dataset_id": "dataset-1",
            "recipes": ["lgb", "xgb", "catboost"],
            "features": ["x1"],
            "target_col": "y",
            "split_col": "split",
            "split_values": {"train": "train", "test": "test", "oot": "oot"},
            "seed": 23,
        },
        task_id="task-1",
    )

    expected = runner._tools.resolve(ToolRef("modeling", "train_models")).timeout_seconds
    assert result.ok is True, result.error
    assert expected > 600
    assert observed["timeout"] == expected
    assert observed["cpu_limit_seconds"] == expected + 2
    started = repo.list_audit(kind="tool.invoke.started")[-1]
    assert started["detail"]["timeout_seconds"] == expected


def test_tool_runner_invokes_sample_echo_in_subprocess(tmp_path):
    runner = _runner(tmp_path)

    result = runner.invoke(ToolRef("_sample", "echo"), {"message": "hi"}, task_id="task-1")

    assert result.ok is True
    assert result.output == {"echoed": "hi"}
    assert result.error is None
    assert result.duration_ms >= 0


def test_progress_path_setup_failure_does_not_fail_tool(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    monkeypatch.setattr(
        "marvis.plugins.runner._new_progress_path",
        lambda _workspace: (_ for _ in ()).throw(OSError("read only")),
    )

    result = runner.invoke(
        ToolRef("_sample", "echo"),
        {"message": "hi"},
        task_id="task-1",
        progress_callback=lambda _payload: None,
    )

    assert result.ok is True
    assert result.output == {"echoed": "hi"}


def test_tool_runner_fails_closed_before_worker_for_protected_tool_without_context(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PluginRepository(db_path)
    policy = GovernancePolicy(
        human_decision_gate="required",
        effect_authorization="required",
        effect_target=EffectTargetPolicy(
            kind="strategy",
            id_input="strategy_id",
            expected_statuses=("draft",),
            result_status="adopted",
        ),
    )
    manifest = PluginManifest(
        name="protected",
        version="1.0.0",
        display_name="Protected",
        description="Protected effect",
        module="protected.tools",
        python_requires="",
        tools=(
            ToolSpec(
                name="adopt",
                summary="Adopt",
                input_schema={
                    "type": "object",
                    "properties": {"strategy_id": {"type": "string"}},
                    "required": ["strategy_id"],
                    "additionalProperties": False,
                },
                output_schema={"type": "object"},
                determinism="deterministic",
                timeout_seconds=10,
                failure_policy="fail",
                side_effects=("write:strategy",),
                entrypoint="run",
                policy=policy,
            ),
        ),
        permissions=("write:strategy",),
        builtin=True,
    )

    class FakeTools:
        def resolve_with_manifest(self, ref):
            return manifest, manifest.tools[0]

    monkeypatch.setattr(
        "marvis.plugins.runner.subprocess.Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("protected worker must not start")
        ),
    )
    runner = ToolRunner(
        FakeTools(),
        repo,
        python_executable=sys.executable,
        datasets_root=tmp_path / "datasets",
        workspace=tmp_path / "workspace",
    )

    result = runner.invoke(
        ToolRef("protected", "adopt"),
        {"strategy_id": "strategy-1"},
        task_id="task-1",
    )

    assert result.ok is False
    assert result.error_kind == "authorization"


def _protected_runner_runtime(
    tmp_path,
    *,
    governance=None,
    binding_resolver=None,
    policy=None,
):
    db_path = tmp_path / "protected.sqlite"
    init_db(db_path)
    repo = PluginRepository(db_path)
    policy = policy or GovernancePolicy(
        human_decision_gate="required",
        effect_authorization="required",
        effect_target=EffectTargetPolicy(
            kind="strategy",
            id_input="strategy_id",
            expected_statuses=("draft",),
            result_status="adopted",
        ),
    )
    side_effects = (
        ("write:strategy",)
        if policy.effect_authorization == "required"
        else ()
    )
    manifest = PluginManifest(
        name="protected",
        version="1.0.0",
        display_name="Protected",
        description="Protected effect",
        module="protected.tools",
        python_requires="",
        tools=(
            ToolSpec(
                name="adopt",
                summary="Adopt",
                input_schema={
                    "type": "object",
                    "properties": {"strategy_id": {"type": "string"}},
                    "required": ["strategy_id"],
                    "additionalProperties": False,
                },
                output_schema={
                    "type": "object",
                    "properties": {"status": {"type": "string"}},
                    "required": ["status"],
                    "additionalProperties": False,
                },
                determinism="deterministic",
                timeout_seconds=10,
                failure_policy="retry",
                side_effects=side_effects,
                entrypoint="run",
                policy=policy,
            ),
        ),
        permissions=side_effects,
        builtin=True,
    )

    class FakeTools:
        def resolve_with_manifest(self, ref):
            return manifest, manifest.tools[0]

    return ToolRunner(
        FakeTools(),
        repo,
        python_executable=sys.executable,
        datasets_root=tmp_path / "datasets",
        workspace=tmp_path / "workspace",
        governance=governance,
        binding_resolver=binding_resolver,
    ), repo


def _execution_context(*, effect=True):
    return SimpleNamespace(
        plan_id="plan-1",
        plan_revision=3,
        step_id="step-adopt",
        decision_id="decision-1",
        approval_id="approval-1" if effect else None,
        runtime_generation="runtime-1",
        human_decision_required=True,
        effect_authorization_required=effect,
    )


def _worker_success(output=None):
    payload = {
        "ok": True,
        "output": output or {"status": "adopted"},
        "worker_protocol_version": PROTOCOL_VERSION,
    }
    return subprocess.CompletedProcess(
        args=[sys.executable],
        returncode=0,
        stdout=WORKER_RESULT_SENTINEL + json.dumps(payload),
        stderr="",
    )


def test_human_only_manifest_policy_fails_closed_without_decision_context(
    tmp_path,
    monkeypatch,
):
    runner, _repo = _protected_runner_runtime(
        tmp_path,
        policy=GovernancePolicy(human_decision_gate="required"),
    )
    monkeypatch.setattr(
        "marvis.plugins.runner._run_worker",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("human-governed worker must not start")
        ),
    )

    result = runner.invoke(
        ToolRef("protected", "adopt"),
        {"strategy_id": "strategy-1"},
        task_id="task-1",
    )

    assert result.ok is False
    assert result.error_kind == "authorization"


def test_human_only_manifest_policy_verifies_live_decision_before_worker(
    tmp_path,
    monkeypatch,
):
    events = []
    binding = object()
    context = _execution_context(effect=False)

    class Governance:
        def verify_decision(self, observed_context, live_binding):
            events.append(("verify", observed_context, live_binding))
            return SimpleNamespace(id="decision-1")

        def reserve_effect(self, *_args, **_kwargs):
            raise AssertionError("human-only steps must not reserve an effect")

    def resolve_binding(**kwargs):
        events.append(("binding", kwargs))
        return binding

    runner, _repo = _protected_runner_runtime(
        tmp_path,
        governance=Governance(),
        binding_resolver=resolve_binding,
        policy=GovernancePolicy(human_decision_gate="required"),
    )

    def run_worker(_python, job, **kwargs):
        events.append(("worker", job, kwargs))
        return _worker_success()

    monkeypatch.setattr("marvis.plugins.runner._run_worker", run_worker)

    result = runner.invoke(
        ToolRef("protected", "adopt"),
        {"strategy_id": "strategy-1"},
        task_id="task-1",
        execution_context=context,
    )

    assert result.ok is True
    assert [event[0] for event in events] == ["binding", "verify", "worker"]
    assert events[1] == ("verify", context, binding)


def test_context_elevated_effect_policy_reserves_dispatches_and_consumes(
    tmp_path,
    monkeypatch,
):
    events = []
    binding = object()
    context = _execution_context(effect=True)

    class Governance:
        def reserve_effect(self, observed_context, live_binding):
            events.append(("reserve", observed_context, live_binding))
            return SimpleNamespace(id="effect-1", reservation_id="reservation-1")

        def mark_effect_dispatched(self, effect_id, *, reservation_id):
            events.append(("dispatched", effect_id, reservation_id))

        def mark_effect_committed(self, effect_id, *, reservation_id, result_hash):
            events.append(("committed", effect_id, reservation_id, result_hash))

        def mark_effect_uncertain(self, *args, **kwargs):
            events.append(("uncertain", args, kwargs))

        def release_prepared_effect(self, *args, **kwargs):
            events.append(("released", args, kwargs))

    def resolve_binding(**kwargs):
        events.append(("binding", kwargs))
        return binding

    runner, _repo = _protected_runner_runtime(
        tmp_path,
        governance=Governance(),
        binding_resolver=resolve_binding,
        policy=GovernancePolicy(),
    )

    def run_worker(_python, job, **kwargs):
        events.append(("worker", job, kwargs))
        assert job["effect_execution_id"] == "effect-1"
        assert job["runtime_generation"] == "runtime-1"
        return _worker_success()

    monkeypatch.setattr("marvis.plugins.runner._run_worker", run_worker)

    result = runner.invoke(
        ToolRef("protected", "adopt"),
        {"strategy_id": "strategy-1"},
        task_id="task-1",
        execution_context=context,
    )

    assert result.ok is True
    assert [event[0] for event in events] == [
        "binding",
        "reserve",
        "dispatched",
        "worker",
        "committed",
    ]


def test_protected_tool_requires_live_binding_resolver_before_worker(tmp_path, monkeypatch):
    governance = SimpleNamespace()
    runner, _repo = _protected_runner_runtime(tmp_path, governance=governance)
    monkeypatch.setattr(
        "marvis.plugins.runner._run_worker",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("protected worker must not start")
        ),
    )

    result = runner.invoke(
        ToolRef("protected", "adopt"),
        {"strategy_id": "strategy-1"},
        task_id="task-1",
        execution_context=_execution_context(),
    )

    assert result.ok is False
    assert result.error_kind == "authorization"


def test_protected_tool_rejects_stale_binding_before_worker(tmp_path, monkeypatch):
    class Governance:
        def reserve_effect(self, context, binding):
            raise ApprovalBindingError("input hash changed")

    runner, _repo = _protected_runner_runtime(
        tmp_path,
        governance=Governance(),
        binding_resolver=lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "marvis.plugins.runner._run_worker",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("worker must not start for stale authorization")
        ),
    )

    result = runner.invoke(
        ToolRef("protected", "adopt"),
        {"strategy_id": "strategy-1"},
        task_id="task-1",
        execution_context=_execution_context(),
    )

    assert result.ok is False
    assert result.error_kind == "authorization"
    assert result.error_detail == {"authorization_phase": "reserve"}


def test_protected_tool_reserves_dispatches_and_commits_one_shot_effect(
    tmp_path,
    monkeypatch,
):
    events = []
    binding = object()

    class Governance:
        def reserve_effect(self, context, live_binding):
            events.append(("reserve", context, live_binding))
            return SimpleNamespace(id="effect-1", reservation_id="reservation-1")

        def mark_effect_dispatched(self, effect_id, *, reservation_id):
            events.append(("dispatched", effect_id, reservation_id))

        def mark_effect_committed(self, effect_id, *, reservation_id, result_hash):
            events.append(("committed", effect_id, reservation_id, result_hash))

        def mark_effect_uncertain(self, *args, **kwargs):
            events.append(("uncertain", args, kwargs))

        def release_prepared_effect(self, *args, **kwargs):
            events.append(("released", args, kwargs))

    def resolve_binding(**kwargs):
        events.append(("binding", kwargs))
        return binding

    runner, _repo = _protected_runner_runtime(
        tmp_path,
        governance=Governance(),
        binding_resolver=resolve_binding,
    )

    def run_worker(_python, job, **kwargs):
        events.append(("worker", job, kwargs))
        assert job["inputs"] == {"strategy_id": "strategy-1"}
        assert "approval_id" not in job["inputs"]
        assert "effect_execution_id" not in job["inputs"]
        assert job["effect_execution_id"] == "effect-1"
        assert job["runtime_generation"] == "runtime-1"
        return _worker_success()

    monkeypatch.setattr("marvis.plugins.runner._run_worker", run_worker)

    result = runner.invoke(
        ToolRef("protected", "adopt"),
        {"strategy_id": "strategy-1"},
        task_id="task-1",
        execution_context=_execution_context(),
    )

    assert result.ok is True
    assert [event[0] for event in events] == [
        "binding",
        "reserve",
        "dispatched",
        "worker",
        "committed",
    ]
    assert events[-1][3].startswith("sha256:")


def test_runner_preserves_domain_owned_committed_effect_receipt(tmp_path, monkeypatch):
    events = []

    class Governance:
        def reserve_effect(self, context, binding):
            return SimpleNamespace(id="effect-1", reservation_id="reservation-1")

        def mark_effect_dispatched(self, effect_id, *, reservation_id):
            events.append("dispatched")

        def get_effect_execution(self, effect_id):
            return SimpleNamespace(state="committed")

        def mark_effect_committed(self, effect_id, *, reservation_id, result_hash):
            events.append(("committed", result_hash))

        def mark_effect_uncertain(self, *args, **kwargs):
            raise AssertionError("committed effect must not be downgraded")

    runner, _repo = _protected_runner_runtime(
        tmp_path,
        governance=Governance(),
        binding_resolver=lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "marvis.plugins.runner._run_worker",
        lambda *args, **kwargs: _worker_success(),
    )

    result = runner.invoke(
        ToolRef("protected", "adopt"),
        {"strategy_id": "strategy-1"},
        task_id="task-1",
        execution_context=_execution_context(),
    )

    assert result.ok is True
    assert events == ["dispatched", ("committed", None)]


def test_post_commit_worker_failure_never_downgrades_or_reissues_effect(
    tmp_path,
    monkeypatch,
):
    events = []

    class Governance:
        def reserve_effect(self, context, binding):
            return SimpleNamespace(id="effect-1", reservation_id="reservation-1")

        def mark_effect_dispatched(self, effect_id, *, reservation_id):
            events.append("dispatched")

        def get_effect_execution(self, effect_id):
            return SimpleNamespace(state="committed")

        def mark_effect_uncertain(self, *args, **kwargs):
            raise AssertionError("committed effect must not be downgraded")

        def release_prepared_effect(self, *args, **kwargs):
            raise AssertionError("committed effect must not be released")

    runner, _repo = _protected_runner_runtime(
        tmp_path,
        governance=Governance(),
        binding_resolver=lambda **kwargs: object(),
    )
    failed_payload = {
        "ok": False,
        "error_kind": "execution",
        "error": "artifact write failed after domain commit",
        "worker_protocol_version": PROTOCOL_VERSION,
    }
    monkeypatch.setattr(
        "marvis.plugins.runner._run_worker",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=[sys.executable],
            returncode=1,
            stdout=WORKER_RESULT_SENTINEL + json.dumps(failed_payload),
            stderr="",
        ),
    )

    result = runner.invoke(
        ToolRef("protected", "adopt"),
        {"strategy_id": "strategy-1"},
        task_id="task-1",
        execution_context=_execution_context(),
    )

    assert result.ok is False
    assert result.error_kind == "execution"
    assert "artifact write failed" in result.error
    assert events == ["dispatched"]


def test_protected_tool_releases_only_pre_dispatch_reservation_failure(
    tmp_path,
    monkeypatch,
):
    events = []

    class Governance:
        def reserve_effect(self, context, binding):
            return SimpleNamespace(id="effect-1", reservation_id="reservation-1")

        def mark_effect_dispatched(self, effect_id, *, reservation_id):
            raise ApprovalStateError("ledger unavailable before worker spawn")

        def release_prepared_effect(self, effect_id, *, reservation_id, reason):
            events.append((effect_id, reservation_id, reason))

    runner, _repo = _protected_runner_runtime(
        tmp_path,
        governance=Governance(),
        binding_resolver=lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "marvis.plugins.runner._run_worker",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("worker must not start when dispatch checkpoint fails")
        ),
    )

    result = runner.invoke(
        ToolRef("protected", "adopt"),
        {"strategy_id": "strategy-1"},
        task_id="task-1",
        execution_context=_execution_context(),
    )

    assert result.ok is False
    assert result.error_kind == "authorization"
    assert len(events) == 1
    assert events[0][0:2] == ("effect-1", "reservation-1")


def test_protected_tool_marks_post_dispatch_timeout_uncertain_without_release(
    tmp_path,
    monkeypatch,
):
    events = []

    class Governance:
        def reserve_effect(self, context, binding):
            return SimpleNamespace(id="effect-1", reservation_id="reservation-1")

        def mark_effect_dispatched(self, effect_id, *, reservation_id):
            events.append("dispatched")

        def mark_effect_uncertain(self, effect_id, *, reservation_id, reason):
            events.append(("uncertain", effect_id, reservation_id, reason))

        def release_prepared_effect(self, *args, **kwargs):
            events.append("released")

    runner, _repo = _protected_runner_runtime(
        tmp_path,
        governance=Governance(),
        binding_resolver=lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "marvis.plugins.runner._run_worker",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired("worker", 10)
        ),
    )

    result = runner.invoke(
        ToolRef("protected", "adopt"),
        {"strategy_id": "strategy-1"},
        task_id="task-1",
        execution_context=_execution_context(),
    )

    assert result.ok is False
    assert result.error_kind == "timeout"
    assert events[0] == "dispatched"
    assert events[1][0] == "uncertain"
    assert "released" not in events


def test_protected_tool_marks_worker_launch_failure_uncertain(tmp_path, monkeypatch):
    events = []

    class Governance:
        def reserve_effect(self, context, binding):
            return SimpleNamespace(id="effect-1", reservation_id="reservation-1")

        def mark_effect_dispatched(self, effect_id, *, reservation_id):
            events.append("dispatched")

        def mark_effect_uncertain(self, effect_id, *, reservation_id, reason):
            events.append(("uncertain", reason))

    runner, _repo = _protected_runner_runtime(
        tmp_path,
        governance=Governance(),
        binding_resolver=lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "marvis.plugins.runner._run_worker",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("spawn failed")),
    )

    result = runner.invoke(
        ToolRef("protected", "adopt"),
        {"strategy_id": "strategy-1"},
        task_id="task-1",
        execution_context=_execution_context(),
    )

    assert result.ok is False
    assert result.error_kind == "execution"
    assert [event[0] if isinstance(event, tuple) else event for event in events] == [
        "dispatched",
        "uncertain",
    ]


def test_tool_context_load_dataset_path_rejects_parent_escape(tmp_path):
    ctx = ToolContext(
        task_id="task-1",
        seed=None,
        datasets_root=tmp_path / "datasets",
        workspace=tmp_path / "workspace",
    )

    assert ctx.load_dataset_path("task-1/sample.parquet") == tmp_path / "datasets" / "task-1" / "sample.parquet"
    with pytest.raises(PermissionError):
        ctx.load_dataset_path("../outside.parquet")


def test_tool_context_reports_progress_atomically_and_never_raises(tmp_path):
    workspace = tmp_path / "workspace"
    progress_path = workspace / ".runtime" / "progress" / "run.json"
    ctx = ToolContext(
        task_id="task-1",
        seed=None,
        datasets_root=tmp_path / "datasets",
        workspace=workspace,
        progress_path=progress_path,
    )

    assert ctx.report_progress({"kind": "model_tuning", "trial": 1}) is True
    assert json.loads(progress_path.read_text(encoding="utf-8"))["trial"] == 1
    assert not progress_path.with_name("run.json.tmp").exists()
    assert ctx.report_progress({"not_json": object()}) is False

    escaped = ToolContext(
        task_id="task-1",
        seed=None,
        datasets_root=tmp_path / "datasets",
        workspace=workspace,
        progress_path=tmp_path / "outside.json",
    )
    assert escaped.report_progress({"kind": "model_tuning"}) is False


def test_worker_tool_context_receives_only_opaque_effect_execution_metadata(
    tmp_path,
    monkeypatch,
):
    from marvis.plugins import subprocess_worker

    captured = {}

    def tool(inputs, ctx):
        captured["inputs"] = inputs
        captured["effect_execution_id"] = ctx.effect_execution_id
        captured["runtime_generation"] = ctx.runtime_generation
        return {"ok": True}

    monkeypatch.setattr(subprocess_worker, "_install_network_guard", lambda _effects: None)
    monkeypatch.setattr(subprocess_worker, "_install_process_guard", lambda _effects: None)
    monkeypatch.setattr(subprocess_worker, "_should_install_file_guard", lambda _job: False)
    monkeypatch.setattr(
        subprocess_worker,
        "_load_module",
        lambda _job: SimpleNamespace(run=tool),
    )

    result = subprocess_worker._run_tool(
        {
            "module": "protected.tools",
            "entrypoint": "run",
            "inputs": {"strategy_id": "strategy-1"},
            "task_id": "task-1",
            "seed": None,
            "datasets_root": str(tmp_path / "datasets"),
            "workspace": str(tmp_path / "workspace"),
            "plugin_paths": [],
            "side_effects": ["write:strategy"],
            "builtin": True,
            "effect_execution_id": "effect-1",
            "runtime_generation": "runtime-1",
        }
    )

    assert result == {"ok": True}
    assert captured == {
        "inputs": {"strategy_id": "strategy-1"},
        "effect_execution_id": "effect-1",
        "runtime_generation": "runtime-1",
    }


def test_builtin_worker_context_bridges_progress_path(tmp_path, monkeypatch):
    from marvis.plugins import subprocess_worker

    progress_path = tmp_path / "workspace" / ".runtime" / "progress" / "run.json"

    def tool(_inputs, ctx):
        assert ctx.report_progress({
            "kind": "model_tuning",
            "algorithm": "lgb",
            "trial": 2,
        }) is True
        return {"ok": True}

    monkeypatch.setattr(subprocess_worker, "_install_network_guard", lambda _effects: None)
    monkeypatch.setattr(subprocess_worker, "_should_install_file_guard", lambda _job: False)
    monkeypatch.setattr(
        subprocess_worker,
        "_load_module",
        lambda _job: SimpleNamespace(run=tool),
    )

    result = subprocess_worker._run_tool({
        "module": "modeling.train_tools",
        "entrypoint": "run",
        "inputs": {},
        "task_id": "task-1",
        "seed": None,
        "datasets_root": str(tmp_path / "datasets"),
        "workspace": str(tmp_path / "workspace"),
        "plugin_paths": [],
        "side_effects": [],
        "builtin": True,
        "progress_path": str(progress_path),
    })

    assert result == {"ok": True}
    assert json.loads(progress_path.read_text(encoding="utf-8"))["trial"] == 2


def test_worker_protocol_promotes_nested_resource_limits(monkeypatch):
    from marvis.plugins import subprocess_worker

    nested_limits = {
        "cpu_limit_seconds": 123,
        "cpu_limit_exceeded": True,
        "termination_signal": "SIGXCPU",
    }

    class NestedResourceError(Exception):
        def to_detail(self):
            return {
                "kind": "resource_limit",
                "subkind": "isolated_recipe_cpu_limit",
                "resource_limits": nested_limits,
            }

    emitted = []
    raw_job = json.dumps({"protocol_version": PROTOCOL_VERSION}).encode("utf-8")
    monkeypatch.setattr(
        subprocess_worker.sys,
        "stdin",
        SimpleNamespace(buffer=io.BytesIO(raw_job)),
    )
    monkeypatch.setattr(
        subprocess_worker,
        "_apply_resource_limits",
        lambda *_args, **_kwargs: {"outer": True},
    )
    monkeypatch.setattr(
        subprocess_worker,
        "_run_tool",
        lambda _job: (_ for _ in ()).throw(NestedResourceError("limit")),
    )
    monkeypatch.setattr(subprocess_worker, "_emit", emitted.append)
    monkeypatch.setattr(
        subprocess_worker,
        "_hard_exit",
        lambda code: (_ for _ in ()).throw(SystemExit(code)),
    )

    with pytest.raises(SystemExit) as raised:
        subprocess_worker.worker_main()

    assert raised.value.code == 1
    assert emitted[0]["error_kind"] == "resource_limit"
    assert emitted[0]["error_detail"]["subkind"] == "isolated_recipe_cpu_limit"
    assert emitted[0]["resource_limits"] == nested_limits


def test_tool_runner_starts_worker_with_explicit_utf8_encoding(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    calls = []
    monkeypatch.setattr("marvis.plugins.runner._available_cpu_count", lambda: 4)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("PYTHONPATH", "/tmp/shadow")
    monkeypatch.setenv("MARVIS_PROBE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("PATH", "/usr/bin")

    class FakeProcess:
        pid = 123
        returncode = 0

        def __init__(self, args):
            self.args = args

        def communicate(self, input=None, timeout=None):
            calls.append({"input": input, "timeout": timeout})
            return json.dumps(
                {"ok": True, "output": {"echoed": "你好"}, "worker_protocol_version": PROTOCOL_VERSION},
                ensure_ascii=False,
            ), ""

        def poll(self):
            return self.returncode

    def fake_popen(args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return FakeProcess(args)

    monkeypatch.setattr("marvis.plugins.runner.subprocess.Popen", fake_popen)

    result = runner.invoke(ToolRef("_sample", "echo"), {"message": "你好"}, task_id="task-1")

    assert result.ok is True
    assert result.output == {"echoed": "你好"}
    assert calls[0]["kwargs"]["encoding"] == "utf-8"
    assert calls[0]["kwargs"]["text"] is True
    assert calls[0]["kwargs"]["start_new_session"] is True
    assert calls[0]["kwargs"]["env"]["PATH"] == "/usr/bin"
    assert calls[0]["kwargs"]["env"]["MARVIS_PROBE_URL"] == "http://127.0.0.1:9"
    assert calls[0]["kwargs"]["env"]["PYTHONIOENCODING"] == "utf-8"
    assert "OPENAI_API_KEY" not in calls[0]["kwargs"]["env"]
    assert "PYTHONPATH" not in calls[0]["kwargs"]["env"]
    job = json.loads(calls[1]["input"])
    assert job["protocol_version"] == PROTOCOL_VERSION
    # RLIMIT_CPU counts aggregate CPU time across native-library threads,
    # whereas the manifest timeout is wall-clock time.  Give a worker enough
    # aggregate CPU budget to use every logical core for the full wall budget.
    assert job["cpu_limit_seconds"] == 48
    assert job["file_size_limit_mb"] == 2048
    assert job["side_effects"] == []


def test_tool_runner_classifies_sigxcpu_as_resource_limit(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    monkeypatch.setattr("marvis.plugins.runner._available_cpu_count", lambda: 4)
    monkeypatch.setattr(
        "marvis.plugins.runner._run_worker",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=[sys.executable],
            returncode=-signal.SIGXCPU,
            stdout="",
            stderr="",
        ),
    )

    result = runner.invoke(
        ToolRef("_sample", "echo"),
        {"message": "hi"},
        task_id="task-1",
    )

    assert result.ok is False
    assert result.error_kind == "resource_limit"
    assert "SIGXCPU" in result.error
    assert "CPU" in result.error
    assert result.resource_limits == {
        "cpu_limit_seconds": 48,
        "cpu_limit_exceeded": True,
        "termination_signal": "SIGXCPU",
        "worker_returncode": -signal.SIGXCPU,
    }


def test_tool_runner_records_invocation_audit(tmp_path):
    runner, repo = _runtime(tmp_path)

    result = runner.invoke(ToolRef("_sample", "echo"), {"message": "hi"}, task_id="task-1")

    started = repo.list_audit(kind="tool.invoke.started")
    audits = repo.list_audit(kind="tool.invoke")
    assert result.ok is True
    assert len(started) == 1
    assert started[0]["target_ref"] == "_sample.echo"
    assert started[0]["outcome"] == "started"
    assert len(audits) == 1
    assert audits[0]["target_ref"] == "_sample.echo"
    assert audits[0]["outcome"] == "succeeded"
    assert audits[0]["inputs_hash"]
    assert result.resource_limits is not None
    assert result.resource_limits["memory_limit_mb"] == 2048
    assert "memory_limit_applied" in result.resource_limits
    assert audits[0]["detail"]["resource_limits"] == result.resource_limits


def test_tool_runner_does_not_start_worker_when_started_audit_fails(tmp_path, monkeypatch):
    runner, repo = _runtime(tmp_path)
    original_write_audit = repo.write_audit

    def fail_started_audit(**kwargs):
        if kwargs.get("kind") == "tool.invoke.started":
            raise RuntimeError("audit down")
        return original_write_audit(**kwargs)

    monkeypatch.setattr(repo, "write_audit", fail_started_audit)
    monkeypatch.setattr(
        "marvis.plugins.runner.subprocess.Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("worker should not start")),
    )

    result = runner.invoke(ToolRef("_sample", "echo"), {"message": "hi"}, task_id="task-1")

    assert result.ok is False
    assert result.error_kind == "audit"
    assert result.error_detail["audit_phase"] == "start"
    assert repo.list_audit(kind="tool.invoke") == []


def test_tool_runner_returns_audit_failure_when_finish_audit_fails_after_checkpoint(
    tmp_path,
    monkeypatch,
):
    runner, repo = _runtime(tmp_path)
    original_write_audit = repo.write_audit

    def fail_finish_audit(**kwargs):
        if kwargs.get("kind") == "tool.invoke":
            raise RuntimeError("audit down")
        return original_write_audit(**kwargs)

    monkeypatch.setattr(repo, "write_audit", fail_finish_audit)

    result = runner.invoke(ToolRef("_sample", "echo"), {"message": "hi"}, task_id="task-1")

    assert result.ok is False
    assert result.error_kind == "audit"
    assert result.error_detail["audit_phase"] == "finish"
    assert result.error_detail["result_ok"] is True
    started = repo.list_audit(kind="tool.invoke.started")
    assert len(started) == 1
    assert started[0]["target_ref"] == "_sample.echo"
    assert repo.list_audit(kind="tool.invoke") == []


def test_worker_resource_limits_apply_memory_cpu_and_file_size(monkeypatch):
    from marvis.plugins import subprocess_worker

    class FakeResource:
        RLIM_INFINITY = -1
        RLIMIT_DATA = 1
        RLIMIT_AS = 2
        RLIMIT_CPU = 3
        RLIMIT_FSIZE = 4

        def __init__(self):
            self.calls = []

        def getrlimit(self, _kind):
            return (self.RLIM_INFINITY, self.RLIM_INFINITY)

        def setrlimit(self, kind, limits):
            self.calls.append((kind, limits))

    fake = FakeResource()
    monkeypatch.setitem(sys.modules, "resource", fake)

    meta = subprocess_worker._apply_resource_limits(
        128,
        cpu_seconds=12,
        file_size_mb=256,
    )

    # Memory is monitor-enforced (RSS polling in the runner), never rlimited:
    # RLIMIT_AS/RLIMIT_DATA bound virtual address space, which the JVM and
    # OpenBLAS reserve in multi-GB quantities — setting them broke PMML export
    # and scipy imports on Linux. The configured ceiling is still reported.
    assert meta["memory_limit_mb"] == 128
    assert meta["memory_limit_applied"] is False
    assert meta["cpu_limit_applied"] is True
    assert meta["file_size_limit_applied"] is True
    assert meta["degraded"] is False
    assert (fake.RLIMIT_CPU, (12, 12)) in fake.calls
    assert (fake.RLIMIT_FSIZE, (256 * 1024 * 1024, 256 * 1024 * 1024)) in fake.calls
    limited_kinds = {kind for kind, _limits in fake.calls}
    assert fake.RLIMIT_DATA not in limited_kinds
    assert fake.RLIMIT_AS not in limited_kinds


def test_worker_resource_limit_degradation_is_per_limit(monkeypatch):
    from marvis.plugins import subprocess_worker

    class FakeResource:
        RLIM_INFINITY = -1
        RLIMIT_DATA = 1
        RLIMIT_AS = 2
        RLIMIT_CPU = 3
        RLIMIT_FSIZE = 4

        def getrlimit(self, _kind):
            return (self.RLIM_INFINITY, self.RLIM_INFINITY)

        def setrlimit(self, kind, _limits):
            if kind == self.RLIMIT_CPU:
                raise ValueError("unsupported cpu limit")

    monkeypatch.setitem(sys.modules, "resource", FakeResource())

    meta = subprocess_worker._apply_resource_limits(
        128,
        cpu_seconds=12,
        file_size_mb=256,
    )

    assert meta["cpu_limit_applied"] is False
    assert meta["file_size_limit_applied"] is True  # FSIZE unaffected by CPU failure.
    assert meta["degraded"] is True
    assert "cpu" in meta["error"]


def test_tool_runner_derives_seed_for_stochastic_tools(tmp_path):
    runner = _runner(tmp_path)

    first = runner.invoke(ToolRef("_sample", "random"), {"key": "a"}, task_id="task-1")
    second = runner.invoke(ToolRef("_sample", "random"), {"key": "a"}, task_id="task-1")

    assert first.ok is True
    assert second.ok is True
    assert first.output == second.output
    assert isinstance(first.output["seed"], int)


def test_tool_runner_uses_input_seed_for_stochastic_tools(tmp_path):
    runner = _runner(tmp_path)

    first = runner.invoke(ToolRef("_sample", "random"), {"key": "a", "seed": 123}, task_id="task-1")
    second = runner.invoke(ToolRef("_sample", "random"), {"key": "a", "seed": 123}, task_id="task-2")
    different = runner.invoke(ToolRef("_sample", "random"), {"key": "a", "seed": 124}, task_id="task-1")

    assert first.ok is True
    assert second.ok is True
    assert different.ok is True
    assert first.output == second.output
    assert first.output["seed"] == 123
    assert different.output["seed"] == 124
    assert different.output["value"] != first.output["value"]


def test_tool_runner_returns_schema_error_before_worker(tmp_path):
    runner = _runner(tmp_path)

    result = runner.invoke(ToolRef("_sample", "echo"), {}, task_id="task-1")

    assert result.ok is False
    assert result.error_kind == "schema"
    assert "message" in result.error


def test_tool_runner_blocks_tool_when_side_effect_exceeds_permissions(tmp_path, monkeypatch):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PluginRepository(db_path)
    manifest = PluginManifest(
        name="unsafe",
        version="0.1.0",
        display_name="Unsafe",
        description="Bypassed manifest parser",
        module="unsafe.tools",
        python_requires="",
        tools=(
            ToolSpec(
                name="write_data",
                summary="Writes data",
                input_schema={"type": "object", "additionalProperties": False},
                output_schema={"type": "object", "additionalProperties": False},
                determinism="deterministic",
                timeout_seconds=10,
                failure_policy="fail",
                side_effects=("write:dataset",),
                entrypoint="run",
            ),
        ),
        permissions=("read:dataset",),
    )

    class FakeTools:
        def resolve_with_manifest(self, ref):
            assert ref == ToolRef("unsafe", "write_data")
            return manifest, manifest.tools[0]

    monkeypatch.setattr(
        "marvis.plugins.runner.subprocess.Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("worker should not start")),
    )
    runner = ToolRunner(
        FakeTools(),
        repo,
        python_executable=sys.executable,
        datasets_root=tmp_path / "datasets",
        workspace=tmp_path / "workspace",
    )

    result = runner.invoke(ToolRef("unsafe", "write_data"), {}, task_id="task-1")

    assert result.ok is False
    assert result.error_kind == "permission"
    assert "write:dataset" in result.error
    audits = repo.list_audit(kind="tool.invoke")
    assert audits[0]["target_ref"] == "unsafe.write_data"
    assert audits[0]["outcome"] == "failed"


def test_tool_runner_validates_output_paths_for_registered_tool(tmp_path, monkeypatch):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PluginRepository(db_path)
    manifest = PluginManifest(
        name="pathpack",
        version="0.1.0",
        display_name="Path Pack",
        description="Path output",
        module="pathpack.tools",
        python_requires="",
        tools=(
            ToolSpec(
                name="emit_path",
                summary="Emits a path",
                input_schema={"type": "object", "additionalProperties": False},
                output_schema={
                    "type": "object",
                    "properties": {"report_path": {"type": "string"}},
                    "required": ["report_path"],
                    "additionalProperties": False,
                },
                determinism="deterministic",
                timeout_seconds=10,
                failure_policy="fail",
                side_effects=(),
                entrypoint="run",
            ),
        ),
        permissions=(),
    )

    class FakeTools:
        def resolve_with_manifest(self, ref):
            assert ref == ToolRef("pathpack", "emit_path")
            return manifest, manifest.tools[0]

    class FakeProcess:
        pid = 123
        returncode = 0

        def communicate(self, input=None, timeout=None):
            return (
                json.dumps({
                    "ok": True,
                    "output": {"report_path": str(tmp_path / "outside" / "report.xlsx")},
                    "worker_protocol_version": PROTOCOL_VERSION,
                }),
                "",
            )

        def poll(self):
            return self.returncode

    monkeypatch.setattr(
        "marvis.plugins.runner.subprocess.Popen",
        lambda *args, **kwargs: FakeProcess(),
    )
    runner = ToolRunner(
        FakeTools(),
        repo,
        python_executable=sys.executable,
        datasets_root=tmp_path / "datasets",
        workspace=tmp_path / "workspace",
    )

    result = runner.invoke(ToolRef("pathpack", "emit_path"), {}, task_id="task-1")

    assert result.ok is False
    assert result.error_kind == "permission"
    assert "escapes allowed roots" in result.error
    assert repo.list_audit(kind="tool.invoke")[0]["outcome"] == "failed"


def test_tool_runner_converts_tool_exception_to_execution_error(tmp_path):
    runner = _runner(tmp_path)

    result = runner.invoke(ToolRef("_sample", "fail"), {}, task_id="task-1")

    assert result.ok is False
    assert result.error_kind == "execution"
    assert "sample failure" in result.error
    assert "RuntimeError" in result.stderr_tail


def test_worker_execution_failure_uses_nonzero_exit_code(tmp_path):
    module_path = tmp_path / "failing_tool.py"
    module_path.write_text(
        "def run(inputs, ctx):\n"
        "    raise RuntimeError('boom')\n",
        encoding="utf-8",
    )
    job = {
        "protocol_version": PROTOCOL_VERSION,
        "module_path": str(module_path),
        "entrypoint": "run",
        "inputs": {},
        "task_id": "task-1",
        "datasets_root": str(tmp_path / "datasets"),
        "workspace": str(tmp_path / "workspace"),
    }

    completed = subprocess.run(
        [sys.executable, "-m", "marvis.plugins.subprocess_worker"],
        input=json.dumps(job),
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=10,
        check=False,
    )

    payload = _parse_worker_result(completed.stdout)
    assert payload is not None
    assert completed.returncode != 0
    assert payload["ok"] is False
    assert payload["error_kind"] == "execution"


def test_tool_runner_redacts_stdout_and_stderr_tails(tmp_path):
    runner = _runner(tmp_path)
    module_path = tmp_path / "noisy_tool.py"
    module_path.write_text(
        "import sys\n"
        "def run(inputs, ctx):\n"
        "    print('mobile=13800138000 email=raw@example.com')\n"
        "    print('bank=6222000000001234', file=sys.stderr)\n"
        "    return {'ok': True}\n",
        encoding="utf-8",
    )

    result = runner.invoke_adhoc(
        module=module_path,
        entrypoint="run",
        inputs={},
        input_schema={"type": "object", "additionalProperties": False},
        output_schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
        timeout_seconds=10,
        task_id="task-1",
        mode="draft",
    )

    assert result.ok is True, result.error
    combined = f"{result.stdout_tail}\n{result.stderr_tail}"
    assert "13800138000" not in combined
    assert "raw@example.com" not in combined
    assert "6222000000001234" not in combined
    assert "138******00" in combined
    assert "[REDACTED_EMAIL]" in combined
    assert "6222********1234" in combined


def test_invoke_adhoc_cannot_bypass_registered_strategy_governance(
    tmp_path,
    monkeypatch,
):
    runner = _runner(tmp_path)
    strategy_module = (
        Path(__file__).parents[1] / "marvis" / "packs" / "strategy" / "tools.py"
    )
    monkeypatch.setattr(
        "marvis.plugins.runner.subprocess.Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("registered plugin worker must not start through invoke_adhoc")
        ),
    )

    result = runner.invoke_adhoc(
        module=strategy_module,
        entrypoint="tool_adopt_strategy",
        inputs={
            "strategy_id": "strategy-1",
            "backtest_id": "backtest-1",
            "adoption_reason": "bypass attempt",
        },
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        timeout_seconds=10,
        task_id="task-1",
        mode="draft",
    )

    assert result.ok is False
    assert result.error_kind == "authorization"
    assert "manifest-aware" in result.error


def test_invoke_adhoc_cannot_bypass_registered_module_through_hardlink(
    tmp_path,
    monkeypatch,
):
    runner = _runner(tmp_path)
    strategy_module = (
        Path(__file__).parents[1] / "marvis" / "packs" / "strategy" / "tools.py"
    )
    alias_module = tmp_path / "strategy_tools_alias.py"
    alias_module.hardlink_to(strategy_module)
    monkeypatch.setattr(
        "marvis.plugins.runner.subprocess.Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("registered plugin alias must not start an adhoc worker")
        ),
    )

    result = runner.invoke_adhoc(
        module=alias_module,
        entrypoint="tool_adopt_strategy",
        inputs={
            "strategy_id": "strategy-1",
            "backtest_id": "backtest-1",
            "adoption_reason": "alias bypass attempt",
        },
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        timeout_seconds=10,
        task_id="task-1",
        mode="draft",
    )

    assert result.ok is False
    assert result.error_kind == "authorization"
    assert "manifest-aware" in result.error


def test_tool_runner_denies_network_for_adhoc_without_network_side_effect(tmp_path):
    runner = _runner(tmp_path)
    module_path = tmp_path / "network_tool.py"
    module_path.write_text(
        "import socket\n"
        "def run(inputs, ctx):\n"
        "    socket.create_connection(('203.0.113.1', 80), timeout=0.1)\n"
        "    return {'ok': True}\n",
        encoding="utf-8",
    )

    result = runner.invoke_adhoc(
        module=module_path,
        entrypoint="run",
        inputs={},
        input_schema={"type": "object", "additionalProperties": False},
        output_schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
        timeout_seconds=10,
        task_id="task-1",
        mode="draft",
    )

    assert result.ok is False
    assert result.error_kind == "execution"
    assert "network access requires network:optional or llm" in result.error


def test_tool_runner_denies_adhoc_file_read_outside_allowed_roots_at_runtime(tmp_path):
    runner = _runner(tmp_path)
    module_dir = tmp_path / "drafts"
    module_dir.mkdir()
    secret_path = tmp_path / "outside-secret.txt"
    secret_path.write_text("raw-secret", encoding="utf-8")
    module_path = module_dir / "reader_tool.py"
    module_path.write_text(
        "def run(inputs, ctx):\n"
        "    with open(inputs['path'], encoding='utf-8') as handle:\n"
        "        return {'content': handle.read()}\n",
        encoding="utf-8",
    )

    result = runner.invoke_adhoc(
        module=module_path,
        entrypoint="run",
        inputs={"path": str(secret_path)},
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"content": {"type": "string"}},
            "required": ["content"],
            "additionalProperties": False,
        },
        timeout_seconds=10,
        task_id="task-1",
        mode="draft",
    )

    assert result.ok is False
    assert result.error_kind == "execution"
    assert "file read access denied" in result.error
    assert "outside-secret.txt" in result.error


def test_tool_runner_allows_loopback_for_local_kernels_without_network_side_effect(tmp_path):
    runner = _runner(tmp_path)
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    def serve_once():
        conn, _addr = server.accept()
        conn.close()
        server.close()

    thread = threading.Thread(target=serve_once, daemon=True)
    thread.start()
    module_path = tmp_path / "loopback_tool.py"
    module_path.write_text(
        "import socket\n"
        "def run(inputs, ctx):\n"
        "    conn = socket.create_connection(('127.0.0.1', inputs['port']), timeout=1.0)\n"
        "    conn.close()\n"
        "    return {'ok': True}\n",
        encoding="utf-8",
    )

    result = runner.invoke_adhoc(
        module=module_path,
        entrypoint="run",
        inputs={"port": port},
        input_schema={
            "type": "object",
            "properties": {"port": {"type": "integer"}},
            "required": ["port"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
        timeout_seconds=10,
        task_id="task-1",
        mode="draft",
    )
    thread.join(timeout=2)

    assert result.ok is True, result.error


def test_tool_runner_denies_adhoc_process_spawn_without_side_effect(tmp_path):
    runner = _runner(tmp_path)
    module_path = tmp_path / "process_tool.py"
    module_path.write_text(
        "import subprocess\n"
        "import sys\n"
        "def run(inputs, ctx):\n"
        "    subprocess.run([sys.executable, '-c', 'print(1)'], check=True)\n"
        "    return {'ok': True}\n",
        encoding="utf-8",
    )

    result = runner.invoke_adhoc(
        module=module_path,
        entrypoint="run",
        inputs={},
        input_schema={"type": "object", "additionalProperties": False},
        output_schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
        timeout_seconds=10,
        task_id="task-1",
        mode="draft",
    )

    assert result.ok is False
    assert result.error_kind == "execution"
    assert "process spawn access requires process:spawn" in result.error


def test_tool_runner_denies_adhoc_os_symlink_without_write_side_effect(tmp_path):
    if not hasattr(os, "symlink"):
        pytest.skip("os.symlink is not available on this platform")
    runner = _runner(tmp_path)
    module_path = tmp_path / "symlink_tool.py"
    module_path.write_text(
        "import os\n"
        "def run(inputs, ctx):\n"
        "    os.symlink(inputs['source'], inputs['target'])\n"
        "    return {'ok': True}\n",
        encoding="utf-8",
    )
    target = tmp_path / "workspace" / "tasks" / "task-1" / "blocked-link"

    result = runner.invoke_adhoc(
        module=module_path,
        entrypoint="run",
        inputs={"source": sys.executable, "target": str(target)},
        input_schema={
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "target": {"type": "string"},
            },
            "required": ["source", "target"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
        timeout_seconds=10,
        task_id="task-1",
        mode="draft",
    )

    assert result.ok is False
    assert result.error_kind == "execution"
    assert "file write access denied" in result.error
    assert not target.exists()
    assert not target.is_symlink()


def test_tool_runner_allows_external_plugin_process_spawn_with_side_effect(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PluginRepository(db_path)
    plugin_root = tmp_path / "plugins"
    package = plugin_root / "processpack"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "tools.py").write_text(
        "import subprocess\n"
        "import sys\n"
        "def run(inputs, ctx):\n"
        "    out = subprocess.check_output([sys.executable, '-c', 'print(\"child-ok\")'], text=True)\n"
        "    return {'child_output': out.strip()}\n",
        encoding="utf-8",
    )
    manifest = PluginManifest(
        name="processpack",
        version="0.1.0",
        display_name="Process Pack",
        description="Spawns a child process",
        module="processpack.tools",
        python_requires="",
        tools=(
            ToolSpec(
                name="spawn",
                summary="Spawn a child process",
                input_schema={"type": "object", "additionalProperties": False},
                output_schema={
                    "type": "object",
                    "properties": {"child_output": {"type": "string"}},
                    "required": ["child_output"],
                    "additionalProperties": False,
                },
                determinism="deterministic",
                timeout_seconds=10,
                failure_policy="fail",
                side_effects=("process:spawn",),
                entrypoint="run",
            ),
        ),
        permissions=("process:spawn",),
        builtin=False,
    )

    class FakeTools:
        def resolve_with_manifest(self, ref):
            assert ref == ToolRef("processpack", "spawn")
            return manifest, manifest.tools[0]

    runner = ToolRunner(
        FakeTools(),
        repo,
        python_executable=sys.executable,
        datasets_root=tmp_path / "datasets",
        workspace=tmp_path / "workspace",
        plugin_paths=[plugin_root],
    )

    result = runner.invoke(ToolRef("processpack", "spawn"), {}, task_id="task-1")

    assert result.ok is True, result.error
    assert result.output == {"child_output": "child-ok"}


def test_tool_runner_denies_external_plugin_file_write_without_write_side_effect(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PluginRepository(db_path)
    plugin_root = tmp_path / "plugins"
    package = plugin_root / "writerpack"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "tools.py").write_text(
        "from pathlib import Path\n"
        "def run(inputs, ctx):\n"
        "    target = Path(inputs['path'])\n"
        "    target.parent.mkdir(parents=True, exist_ok=True)\n"
        "    target.write_text('unsafe', encoding='utf-8')\n"
        "    return {'ok': True}\n",
        encoding="utf-8",
    )
    manifest = PluginManifest(
        name="writerpack",
        version="0.1.0",
        display_name="Writer Pack",
        description="Writes a file",
        module="writerpack.tools",
        python_requires="",
        tools=(
            ToolSpec(
                name="write_file",
                summary="Writes a file",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
                output_schema={
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": False,
                },
                determinism="deterministic",
                timeout_seconds=10,
                failure_policy="fail",
                side_effects=(),
                entrypoint="run",
            ),
        ),
        permissions=(),
        builtin=False,
    )

    class FakeTools:
        def resolve_with_manifest(self, ref):
            assert ref == ToolRef("writerpack", "write_file")
            return manifest, manifest.tools[0]

    runner = ToolRunner(
        FakeTools(),
        repo,
        python_executable=sys.executable,
        datasets_root=tmp_path / "datasets",
        workspace=tmp_path / "workspace",
        plugin_paths=[plugin_root],
    )
    target = tmp_path / "workspace" / "tasks" / "task-1" / "blocked.txt"

    result = runner.invoke(ToolRef("writerpack", "write_file"), {"path": str(target)}, task_id="task-1")

    assert result.ok is False
    assert result.error_kind == "execution"
    assert "file write access denied" in result.error
    assert not target.exists()


def test_tool_runner_allows_external_plugin_file_write_with_write_side_effect(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PluginRepository(db_path)
    plugin_root = tmp_path / "plugins"
    package = plugin_root / "writerpack_allowed"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "tools.py").write_text(
        "from pathlib import Path\n"
        "def run(inputs, ctx):\n"
        "    target = Path(inputs['path'])\n"
        "    target.parent.mkdir(parents=True, exist_ok=True)\n"
        "    target.write_text('allowed', encoding='utf-8')\n"
        "    return {'ok': True}\n",
        encoding="utf-8",
    )
    manifest = PluginManifest(
        name="writerpack_allowed",
        version="0.1.0",
        display_name="Writer Pack Allowed",
        description="Writes a file",
        module="writerpack_allowed.tools",
        python_requires="",
        tools=(
            ToolSpec(
                name="write_file",
                summary="Writes a file",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
                output_schema={
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": False,
                },
                determinism="deterministic",
                timeout_seconds=10,
                failure_policy="fail",
                side_effects=("write:artifact",),
                entrypoint="run",
            ),
        ),
        permissions=("write:artifact",),
        builtin=False,
    )

    class FakeTools:
        def resolve_with_manifest(self, ref):
            assert ref == ToolRef("writerpack_allowed", "write_file")
            return manifest, manifest.tools[0]

    runner = ToolRunner(
        FakeTools(),
        repo,
        python_executable=sys.executable,
        datasets_root=tmp_path / "datasets",
        workspace=tmp_path / "workspace",
        plugin_paths=[plugin_root],
    )
    target = tmp_path / "workspace" / "tasks" / "task-1" / "allowed.txt"

    result = runner.invoke(ToolRef("writerpack_allowed", "write_file"), {"path": str(target)}, task_id="task-1")

    assert result.ok is True, result.error
    assert target.read_text(encoding="utf-8") == "allowed"


def test_tool_runner_allows_output_paths_under_workspace(tmp_path):
    runner = _runner(tmp_path)
    report_path = tmp_path / "workspace" / "tasks" / "task-1" / "outputs" / "report.xlsx"
    module_path = tmp_path / "path_tool.py"
    module_path.write_text(
        "def run(inputs, ctx):\n"
        "    return {'report_path': inputs['report_path']}\n",
        encoding="utf-8",
    )

    result = runner.invoke_adhoc(
        module=module_path,
        entrypoint="run",
        inputs={"report_path": str(report_path)},
        input_schema={
            "type": "object",
            "properties": {"report_path": {"type": "string"}},
            "required": ["report_path"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"report_path": {"type": "string"}},
            "required": ["report_path"],
            "additionalProperties": False,
        },
        timeout_seconds=10,
        task_id="task-1",
        mode="draft",
    )

    assert result.ok is True, result.error
    assert result.output == {"report_path": str(report_path)}


def test_tool_runner_rejects_output_paths_outside_allowed_roots(tmp_path):
    runner = _runner(tmp_path)
    module_path = tmp_path / "path_tool.py"
    module_path.write_text(
        "def run(inputs, ctx):\n"
        "    return {'report_path': inputs['report_path']}\n",
        encoding="utf-8",
    )

    result = runner.invoke_adhoc(
        module=module_path,
        entrypoint="run",
        inputs={"report_path": str(tmp_path / "outside" / "report.xlsx")},
        input_schema={
            "type": "object",
            "properties": {"report_path": {"type": "string"}},
            "required": ["report_path"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"report_path": {"type": "string"}},
            "required": ["report_path"],
            "additionalProperties": False,
        },
        timeout_seconds=10,
        task_id="task-1",
        mode="draft",
    )

    assert result.ok is False
    assert result.error_kind == "permission"
    assert "escapes allowed roots" in result.error


def test_tool_runner_rejects_unsafe_relative_output_paths(tmp_path):
    runner = _runner(tmp_path)
    module_path = tmp_path / "path_tool.py"
    module_path.write_text(
        "def run(inputs, ctx):\n"
        "    return {'artifacts': [{'path': '../secret.txt'}]}\n",
        encoding="utf-8",
    )

    result = runner.invoke_adhoc(
        module=module_path,
        entrypoint="run",
        inputs={},
        input_schema={"type": "object", "additionalProperties": False},
        output_schema={
            "type": "object",
            "properties": {
                "artifacts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            "required": ["artifacts"],
            "additionalProperties": False,
        },
        timeout_seconds=10,
        task_id="task-1",
        mode="draft",
    )

    assert result.ok is False
    assert result.error_kind == "permission"
    assert "unsafe relative path" in result.error


def test_tool_runner_validates_artifact_refs_in_output(tmp_path):
    runner = _runner(tmp_path)
    module_path = tmp_path / "artifact_ref_tool.py"
    module_path.write_text(
        "def run(inputs, ctx):\n"
        "    return {'sample_ref': inputs['sample_ref'], 'download_url': '/api/downloads/../x'}\n",
        encoding="utf-8",
    )
    schema = {
        "type": "object",
        "properties": {"sample_ref": {"type": "string"}},
        "required": ["sample_ref"],
        "additionalProperties": False,
    }
    output_schema = {
        "type": "object",
        "properties": {
            "sample_ref": {"type": "string"},
            "download_url": {"type": "string"},
        },
        "required": ["sample_ref", "download_url"],
        "additionalProperties": False,
    }

    allowed = runner.invoke_adhoc(
        module=module_path,
        entrypoint="run",
        inputs={"sample_ref": "artifact:tasks/task-1/outputs/sample.parquet"},
        input_schema=schema,
        output_schema=output_schema,
        timeout_seconds=10,
        task_id="task-1",
        mode="draft",
    )
    escaped_relative = runner.invoke_adhoc(
        module=module_path,
        entrypoint="run",
        inputs={"sample_ref": "artifact:../secret.parquet"},
        input_schema=schema,
        output_schema=output_schema,
        timeout_seconds=10,
        task_id="task-1",
        mode="draft",
    )
    escaped_absolute = runner.invoke_adhoc(
        module=module_path,
        entrypoint="run",
        inputs={"sample_ref": "artifact:/etc/passwd"},
        input_schema=schema,
        output_schema=output_schema,
        timeout_seconds=10,
        task_id="task-1",
        mode="draft",
    )

    assert allowed.ok is True, allowed.error
    assert escaped_relative.ok is False
    assert escaped_relative.error_kind == "permission"
    assert "unsafe relative path" in escaped_relative.error
    assert escaped_absolute.ok is False
    assert escaped_absolute.error_kind == "permission"
    assert "must be relative" in escaped_absolute.error


def test_tool_runner_rejects_output_schema_mismatch(tmp_path):
    runner = _runner(tmp_path)

    result = runner.invoke(ToolRef("_sample", "bad_output"), {}, task_id="task-1")

    assert result.ok is False
    assert result.error_kind == "schema"
    assert "echoed" in result.error


def test_tool_runner_kills_timed_out_worker(tmp_path):
    runner = _runner(tmp_path)

    result = runner.invoke(
        ToolRef("_sample", "sleep"),
        {"seconds": 2},
        task_id="task-1",
        progress_callback=lambda _payload: None,
    )

    assert result.ok is False
    assert result.error_kind == "timeout"
    assert "timed out" in result.error
    progress_dir = tmp_path / "workspace" / ".runtime" / "progress"
    assert list(progress_dir.glob("*")) == []


def test_tool_runner_kills_worker_process_group_on_timeout(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    killed = []

    class FakeProcess:
        pid = 4321
        returncode = None

        def __init__(self, args):
            self.args = args
            self.communicate_calls = 0

        def communicate(self, input=None, timeout=None):
            self.communicate_calls += 1
            if timeout is not None:
                raise subprocess.TimeoutExpired(self.args, timeout, output="out", stderr="err")
            self.returncode = -9
            return "out", "err"

        def poll(self):
            return self.returncode

    def fake_popen(args, **_kwargs):
        return FakeProcess(args)

    monkeypatch.setattr("marvis.plugins.runner.subprocess.Popen", fake_popen)
    monkeypatch.setattr("marvis.plugins.runner.os.getpgid", lambda pid: pid)
    monkeypatch.setattr("marvis.plugins.runner.os.killpg", lambda pgid, sig: killed.append((pgid, sig)))

    result = runner.invoke(ToolRef("_sample", "sleep"), {"seconds": 2}, task_id="task-1")

    assert result.ok is False
    assert result.error_kind == "timeout"
    assert killed == [(4321, signal.SIGKILL)]


def test_run_worker_delivers_progress_before_exit_and_callback_errors_are_isolated(
    tmp_path,
    monkeypatch,
):
    progress_path = tmp_path / "progress.json"
    callback_seen = threading.Event()
    events = []

    class FakeProcess:
        pid = 4321
        returncode = 0

        def __init__(self, args):
            self.args = args

        def communicate(self, input=None, timeout=None):
            progress_path.write_text(
                json.dumps({
                    "kind": "model_tuning",
                    "algorithm": "xgb",
                    "trial": 1,
                    "trial_total": 2,
                }),
                encoding="utf-8",
            )
            assert callback_seen.wait(2), "watcher did not publish while worker was running"
            progress_path.write_text(
                json.dumps({
                    "kind": "model_tuning",
                    "algorithm": "xgb",
                    "trial": 2,
                    "trial_total": 2,
                }),
                encoding="utf-8",
            )
            return "out", "err"

        def poll(self):
            return self.returncode

    class FakeMonitor:
        memory_limit_exceeded = False

        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def callback(payload):
        events.append(payload)
        callback_seen.set()
        raise RuntimeError("display failed")

    monkeypatch.setattr(
        "marvis.plugins.runner.subprocess.Popen",
        lambda args, **_kwargs: FakeProcess(args),
    )
    monkeypatch.setattr("marvis.plugins.runner.ProcessTreeResourceMonitor", FakeMonitor)

    completed = _run_worker(
        sys.executable,
        {"protocol_version": PROTOCOL_VERSION},
        timeout=10,
        progress_path=progress_path,
        progress_callback=callback,
    )

    assert completed.returncode == 0
    assert [event["trial"] for event in events] == [1, 2]
    assert not progress_path.exists()
    assert not progress_path.with_name("progress.json.tmp").exists()


def test_run_worker_cancellation_terminates_and_reaps_worker_tree(monkeypatch):
    """A cooperative job cancellation must wake a blocked communicate()."""

    token = JobCancellationToken(job_id="driver-job-1")
    worker_started = threading.Event()
    worker_killed = threading.Event()
    processes = []

    class FakeProcess:
        pid = 4321
        returncode = None

        def __init__(self, args):
            self.args = args
            self.communicate_calls = 0

        def communicate(self, input=None, timeout=None):
            self.communicate_calls += 1
            worker_started.set()
            assert worker_killed.wait(2), "cancellation did not terminate worker"
            self.returncode = -signal.SIGTERM
            return "", ""

        def poll(self):
            return self.returncode

    class FakeMonitor:
        memory_limit_exceeded = False

        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def snapshot(self):
            return {}

    def fake_popen(args, **_kwargs):
        process = FakeProcess(args)
        processes.append(process)
        return process

    def fake_terminate(process, *, owns_process_group):
        assert owns_process_group is True
        worker_killed.set()
        process.returncode = -signal.SIGTERM

    monkeypatch.setattr("marvis.plugins.runner.subprocess.Popen", fake_popen)
    monkeypatch.setattr("marvis.plugins.runner.ProcessTreeResourceMonitor", FakeMonitor)
    monkeypatch.setattr("marvis.plugins.runner._terminate_worker_process", fake_terminate)
    monkeypatch.setattr(
        "marvis.plugins.runner._bounded_reap_worker",
        lambda process, **_kwargs: ("", ""),
    )

    canceller = threading.Thread(
        target=lambda: (worker_started.wait(2), token.cancel()),
        daemon=True,
    )
    canceller.start()
    with pytest.raises(Exception, match="job cancelled"):
        _run_worker(
            sys.executable,
            {"protocol_version": PROTOCOL_VERSION},
            timeout=30,
            cancellation_check=token.raise_if_cancelled,
        )
    canceller.join(timeout=2)

    assert worker_killed.is_set()
    assert processes[0].communicate_calls >= 1


def test_run_worker_completed_protocol_wins_over_late_cancellation(monkeypatch):
    """A stop after result emission belongs to the next plan-step boundary."""

    token = JobCancellationToken(job_id="driver-job-late-cancel")
    cancellation_seen = threading.Event()
    payload = {
        "ok": True,
        "worker_protocol_version": PROTOCOL_VERSION,
        "output": {"artifact_id": "artifact-complete"},
    }

    class FakeProcess:
        pid = 4321
        returncode = 0

        def __init__(self, args):
            self.args = args

        def communicate(self, input=None, timeout=None):
            token.cancel()
            assert cancellation_seen.wait(2), "watcher did not observe cancellation"
            return WORKER_RESULT_SENTINEL + json.dumps(payload), ""

        def poll(self):
            return self.returncode

    class FakeMonitor:
        memory_limit_exceeded = False

        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_terminate(_process, *, owns_process_group):
        assert owns_process_group is True
        cancellation_seen.set()

    monkeypatch.setattr(
        "marvis.plugins.runner.subprocess.Popen",
        lambda args, **_kwargs: FakeProcess(args),
    )
    monkeypatch.setattr("marvis.plugins.runner.ProcessTreeResourceMonitor", FakeMonitor)
    monkeypatch.setattr(
        "marvis.plugins.runner._terminate_worker_process",
        fake_terminate,
    )

    completed = _run_worker(
        sys.executable,
        {"protocol_version": PROTOCOL_VERSION},
        timeout=30,
        cancellation_check=token.raise_if_cancelled,
    )

    assert _parse_worker_result(completed.stdout) == payload


def test_tool_runner_returns_audited_cancelled_result(tmp_path, monkeypatch):
    token = JobCancellationToken(job_id="driver-job-1")
    token.cancel()
    runner, repo = _runtime(tmp_path)

    result = runner.invoke(
        ToolRef("_sample", "echo"),
        {"message": "hi"},
        task_id="task-1",
        cancellation_check=token.raise_if_cancelled,
    )

    assert result.ok is False
    assert result.error_kind == "cancelled"
    assert result.error_detail["kind"] == "user_cancelled"
    audits = repo.list_audit(kind="tool.invoke")
    assert audits[-1]["outcome"] == "failed"
    assert audits[-1]["detail"]["error_kind"] == "cancelled"


@pytest.mark.parametrize("interruption_type", [KeyboardInterrupt, asyncio.CancelledError])
def test_run_worker_kills_and_reaps_process_tree_on_interruption(
    tmp_path,
    monkeypatch,
    interruption_type,
):
    interruption = interruption_type()
    killed = []
    processes = []
    progress_path = tmp_path / "progress.json"

    class FakeProcess:
        pid = 4321
        returncode = None

        def __init__(self, args):
            self.args = args
            self.communicate_calls = 0

        def communicate(self, input=None, timeout=None):
            self.communicate_calls += 1
            if self.communicate_calls == 1:
                progress_path.write_text(
                    json.dumps({"kind": "model_tuning", "trial": 1}),
                    encoding="utf-8",
                )
                raise interruption
            return "", ""

        def poll(self):
            return self.returncode

    class FakeMonitor:
        memory_limit_exceeded = False

        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_popen(args, **_kwargs):
        process = FakeProcess(args)
        processes.append(process)
        return process

    def fake_kill_worker_tree(process):
        killed.append(process)
        process.returncode = -signal.SIGKILL

    monkeypatch.setattr("marvis.plugins.runner.subprocess.Popen", fake_popen)
    monkeypatch.setattr("marvis.plugins.runner.ProcessTreeResourceMonitor", FakeMonitor)
    monkeypatch.setattr("marvis.plugins.runner._kill_worker_tree", fake_kill_worker_tree)

    with pytest.raises(interruption_type) as raised:
        _run_worker(
            sys.executable,
            {"protocol_version": PROTOCOL_VERSION},
            timeout=10,
            progress_path=progress_path,
            progress_callback=lambda _payload: None,
        )

    assert raised.value is interruption
    assert killed == processes
    assert processes[0].communicate_calls == 2
    assert not progress_path.exists()
    assert not progress_path.with_name("progress.json.tmp").exists()


def test_run_worker_nested_mode_kills_child_without_killing_parent_group(monkeypatch):
    """Nested recipe workers share the parent group for host-level cleanup.

    Local interruption must therefore kill only the child process; using
    ``killpg`` here would kill the aggregate worker that is responsible for
    checkpointing the completed recipes.
    """

    process_holder = []
    popen_kwargs = {}
    terminated = []

    class FakeProcess:
        pid = 4321
        returncode = None

        def __init__(self, args):
            self.args = args
            self.communicate_calls = 0
            self.kill_calls = 0

        def communicate(self, input=None, timeout=None):
            self.communicate_calls += 1
            if self.communicate_calls == 1:
                raise KeyboardInterrupt()
            return "", ""

        def poll(self):
            return self.returncode

        def kill(self):
            self.kill_calls += 1
            self.returncode = -signal.SIGKILL

    class FakeMonitor:
        memory_limit_exceeded = False

        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_popen(args, **kwargs):
        popen_kwargs.update(kwargs)
        process = FakeProcess(args)
        process_holder.append(process)
        return process

    monkeypatch.setattr("marvis.plugins.runner.subprocess.Popen", fake_popen)
    monkeypatch.setattr("marvis.plugins.runner.ProcessTreeResourceMonitor", FakeMonitor)
    monkeypatch.setattr(
        "marvis.plugins.runner._kill_worker_tree",
        lambda _process: (_ for _ in ()).throw(AssertionError("must not kill parent group")),
    )
    monkeypatch.setattr(
        "marvis.plugins.runner.terminate_process_tree_by_pid",
        lambda pid, **_kwargs: terminated.append(pid) or True,
    )

    with pytest.raises(KeyboardInterrupt):
        _run_worker(
            sys.executable,
            {"protocol_version": PROTOCOL_VERSION},
            timeout=10,
            start_new_session=False,
        )

    assert popen_kwargs["start_new_session"] is False
    assert terminated == [4321]
    assert process_holder[0].kill_calls == 0
    assert process_holder[0].communicate_calls == 2


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-tree regression")
def test_nested_termination_kills_grandchild_holding_worker_pipes_with_bounded_reap():
    from marvis.plugins import runner as runner_module

    grandchild_code = "import time; time.sleep(60)"
    parent_code = (
        "import subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c',sys.argv[1]]); "
        "print(child.pid, flush=True); time.sleep(60)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", parent_code, grandchild_code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        start_new_session=False,
    )
    assert process.stdout is not None
    grandchild_pid = int(process.stdout.readline().strip())
    started = time.monotonic()
    try:
        runner_module._terminate_worker_process(
            process,
            owns_process_group=False,
        )
        runner_module._bounded_reap_worker(process, timeout_seconds=2.0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        try:
            grandchild = psutil.Process(grandchild_pid)
            if grandchild.is_running() and grandchild.status() != psutil.STATUS_ZOMBIE:
                grandchild.kill()
        except psutil.NoSuchProcess:
            pass

    assert time.monotonic() - started < 5.0
    assert process.poll() is not None
    try:
        grandchild = psutil.Process(grandchild_pid)
        assert not grandchild.is_running() or grandchild.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        pass


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group regression")
def test_run_worker_reaps_group_when_crashed_leader_left_live_grandchild(tmp_path):
    """A native-crashed worker leader must not make group cleanup a no-op.

    The real worker dies by signal after spawning a child that inherits its
    stdout/stderr pipes.  ``communicate`` therefore reaches the host timeout
    with ``poll() != None`` for the leader while the descendant is still live —
    the exact shape that previously skipped ``killpg`` and orphaned learners.
    """

    module_path = tmp_path / "crash_with_grandchild.py"
    pid_path = tmp_path / "grandchild.pid"
    module_path.write_text(
        "\n".join(
            [
                "import os",
                "from pathlib import Path",
                "import signal",
                "import subprocess",
                "import sys",
                "",
                "def run(inputs, _ctx):",
                "    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])",
                "    Path(inputs['pid_path']).write_text(str(child.pid), encoding='utf-8')",
                "    os.kill(os.getpid(), signal.SIGKILL)",
            ]
        ),
        encoding="utf-8",
    )
    job = {
        "protocol_version": PROTOCOL_VERSION,
        "module": "crash_with_grandchild",
        "entrypoint": "run",
        "inputs": {"pid_path": str(pid_path)},
        "task_id": "task-crash",
        "seed": 1,
        "datasets_root": str(tmp_path / "datasets"),
        "workspace": str(tmp_path),
        "memory_limit_mb": 256,
        "cpu_limit_seconds": 60,
        "file_size_limit_mb": 16,
        "plugin_paths": [str(tmp_path)],
        "side_effects": ["process:spawn"],
        "builtin": True,
    }

    grandchild_pid = None
    try:
        with pytest.raises(WorkerTimeoutExpired):
            _run_worker(sys.executable, job, timeout=1, start_new_session=True)
        assert pid_path.exists(), "native-crash fixture never spawned its descendant"
        grandchild_pid = int(pid_path.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and psutil.pid_exists(grandchild_pid):
            try:
                child = psutil.Process(grandchild_pid)
                if child.status() == psutil.STATUS_ZOMBIE:
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(0.02)
        if psutil.pid_exists(grandchild_pid):
            child = psutil.Process(grandchild_pid)
            assert child.status() == psutil.STATUS_ZOMBIE
    finally:
        if grandchild_pid is not None:
            try:
                child = psutil.Process(grandchild_pid)
                if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                    child.kill()
            except psutil.NoSuchProcess:
                pass


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group regression")
def test_run_worker_reaps_nested_recipe_immediately_after_aggregate_signal(tmp_path):
    """Signal cleanup must run on EOF, without waiting for the host timeout.

    The aggregate owns the host-facing stdout pipe, while its nested recipe has
    independent ``PIPE`` streams just like ``run_tuning_recipe_isolated``.  Once
    the aggregate dies the host sees EOF immediately even though recipe and
    grandchild remain live in its process group.
    """

    module_path = tmp_path / "crash_aggregate_with_recipe.py"
    recipe_pid_path = tmp_path / "recipe.pid"
    grandchild_pid_path = tmp_path / "recipe-grandchild.pid"
    module_path.write_text(
        "\n".join(
            [
                "import os",
                "from pathlib import Path",
                "import signal",
                "import subprocess",
                "import sys",
                "import time",
                "",
                "def run(inputs, _ctx):",
                "    recipe_code = (",
                "        \"import subprocess,sys,time; from pathlib import Path; \"",
                "        \"child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); \"",
                "        \"Path(sys.argv[1]).write_text(str(child.pid),encoding='utf-8'); time.sleep(60)\"",
                "    )",
                "    recipe = subprocess.Popen(",
                "        [sys.executable, '-c', ''.join(recipe_code), inputs['grandchild_pid_path']],",
                "        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,",
                "    )",
                "    Path(inputs['recipe_pid_path']).write_text(str(recipe.pid), encoding='utf-8')",
                "    deadline = time.monotonic() + 5",
                "    while not Path(inputs['grandchild_pid_path']).exists():",
                "        if time.monotonic() >= deadline: raise RuntimeError('grandchild did not start')",
                "        time.sleep(0.01)",
                "    os.kill(os.getpid(), signal.SIGKILL)",
            ]
        ),
        encoding="utf-8",
    )
    job = {
        "protocol_version": PROTOCOL_VERSION,
        "module": "crash_aggregate_with_recipe",
        "entrypoint": "run",
        "inputs": {
            "recipe_pid_path": str(recipe_pid_path),
            "grandchild_pid_path": str(grandchild_pid_path),
        },
        "task_id": "task-aggregate-crash",
        "seed": 1,
        "datasets_root": str(tmp_path / "datasets"),
        "workspace": str(tmp_path),
        "memory_limit_mb": 256,
        "cpu_limit_seconds": 60,
        "file_size_limit_mb": 16,
        "plugin_paths": [str(tmp_path)],
        "side_effects": ["process:spawn"],
        "builtin": True,
    }

    descendants = []
    started = time.monotonic()
    try:
        completed = _run_worker(sys.executable, job, timeout=30, start_new_session=True)
        assert completed.returncode == -signal.SIGKILL
        assert time.monotonic() - started < 5.0
        descendants = [
            int(recipe_pid_path.read_text(encoding="utf-8")),
            int(grandchild_pid_path.read_text(encoding="utf-8")),
        ]
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            live = []
            for pid in descendants:
                try:
                    process = psutil.Process(pid)
                    if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                        live.append(pid)
                except psutil.NoSuchProcess:
                    pass
            if not live:
                break
            time.sleep(0.02)
        for pid in descendants:
            try:
                process = psutil.Process(pid)
                assert not process.is_running() or process.status() == psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                pass
    finally:
        for pid in descendants:
            try:
                process = psutil.Process(pid)
                if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    process.kill()
            except psutil.NoSuchProcess:
                pass


def test_cached_worker_group_cleanup_refuses_host_process_group(monkeypatch):
    from marvis.plugins import runner as runner_module

    killed = []

    class FakeProcess:
        pid = 4321
        returncode = -signal.SIGKILL
        _marvis_process_group_id = os.getpgrp()

        def poll(self):
            return self.returncode

    monkeypatch.setattr(runner_module.os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))

    runner_module._kill_worker_tree(FakeProcess())

    assert killed == []


def test_nested_worker_windows_fallback_targets_only_child_tree(monkeypatch):
    from marvis.plugins import runner as runner_module

    taskkill = []

    class FakeProcess:
        pid = 4321

        @staticmethod
        def poll():
            return None

        @staticmethod
        def kill():
            raise AssertionError("Windows tree fallback must use taskkill /T")

    monkeypatch.setattr(runner_module.os, "name", "nt")
    monkeypatch.setattr(
        runner_module,
        "terminate_process_tree_by_pid",
        lambda _pid, **_kwargs: False,
    )
    monkeypatch.setattr(
        runner_module,
        "_taskkill_worker_tree",
        lambda pid: taskkill.append(pid),
    )

    runner_module._terminate_worker_process(
        FakeProcess(),
        owns_process_group=False,
    )

    assert taskkill == [4321]


def test_bounded_reap_closes_inherited_pipes_after_second_timeout():
    from marvis.plugins import runner as runner_module

    streams = [SimpleNamespace(close_calls=0) for _ in range(3)]
    for stream in streams:
        stream.close = lambda stream=stream: setattr(
            stream,
            "close_calls",
            stream.close_calls + 1,
        )

    class FakeProcess:
        stdin, stdout, stderr = streams

        def __init__(self):
            self.kill_calls = 0
            self.wait_calls = 0

        def communicate(self, timeout=None):
            raise subprocess.TimeoutExpired(
                ["worker"],
                timeout,
                output="partial-out",
                stderr="partial-err",
            )

        def kill(self):
            self.kill_calls += 1

        def wait(self, timeout=None):
            self.wait_calls += 1
            return -signal.SIGKILL

    process = FakeProcess()
    stdout, stderr = runner_module._bounded_reap_worker(
        process,
        timeout_seconds=0.05,
    )

    assert (stdout, stderr) == ("partial-out", "partial-err")
    assert process.kill_calls == 1
    assert process.wait_calls == 1
    assert [stream.close_calls for stream in streams] == [1, 1, 1]


def test_nested_sigxcpu_maps_to_typed_resource_limit():
    from marvis.plugins import runner as runner_module

    if not hasattr(signal, "SIGXCPU"):
        pytest.skip("SIGXCPU is POSIX-only")
    completed = subprocess.CompletedProcess(
        ["worker"],
        -int(signal.SIGXCPU),
        stdout="",
        stderr="",
    )

    error = runner_module._nested_worker_signal_resource_error(
        completed,
        cpu_limit_seconds=123,
    )

    assert isinstance(error, runner_module.WorkerResourceLimitExceeded)
    assert error.resource_usage["termination_signal"] == "SIGXCPU"
    assert error.error_detail["kind"] == "worker_cpu_limit"


def test_stale_progress_sweep_is_bounded_to_host_owned_uuid_files(tmp_path):
    from marvis.plugins import runner as runner_module

    progress = tmp_path / ".runtime" / "progress"
    progress.mkdir(parents=True)
    old_json = progress / ("a" * 32 + ".json")
    old_journal = progress / ("b" * 32 + ".jsonl")
    recent = progress / ("c" * 32 + ".json")
    unrelated = progress / "user-notes.json"
    for path in (old_json, old_journal, recent, unrelated):
        path.write_text("{}", encoding="utf-8")
    old_time = time.time() - runner_module.PROGRESS_STALE_AFTER_SECONDS - 10
    os.utime(old_json, (old_time, old_time))
    os.utime(old_journal, (old_time, old_time))
    os.utime(unrelated, (old_time, old_time))

    runner_module._sweep_stale_progress_files(tmp_path)

    assert not old_json.exists()
    assert not old_journal.exists()
    assert recent.exists()
    assert unrelated.exists()


def test_progress_sanitizer_preserves_checkpoint_boolean_metadata():
    from marvis.plugins import runner as runner_module

    payload = runner_module._sanitize_progress_payload(
        {
            "kind": "model_tuning",
            "stage": "checkpoint_saved",
            "cache_hit": False,
            "checkpoint_saved": True,
        }
    )

    assert payload["kind"] == "model_tuning"
    assert payload["stage"] == "checkpoint_saved"
    assert payload["cache_hit"] is False
    assert payload["checkpoint_saved"] is True


def test_tool_runner_invokes_adhoc_module_in_subprocess_and_audits_draft(tmp_path):
    runner, repo = _runtime(tmp_path)
    module_path = tmp_path / "draft_calc.py"
    module_path.write_text(
        "def calc_margin(inputs, ctx):\n"
        "    return {'margin': inputs['revenue'] - inputs['cost'], 'task_id': ctx.task_id}\n",
        encoding="utf-8",
    )

    result = runner.invoke_adhoc(
        module=module_path,
        entrypoint="calc_margin",
        inputs={"revenue": 10, "cost": 3},
        input_schema={
            "type": "object",
            "properties": {"revenue": {"type": "number"}, "cost": {"type": "number"}},
            "required": ["revenue", "cost"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"margin": {"type": "number"}, "task_id": {"type": "string"}},
            "required": ["margin", "task_id"],
            "additionalProperties": False,
        },
        timeout_seconds=10,
        task_id="task-1",
        mode="draft",
    )

    assert result.ok is True, result.error
    assert result.output == {"margin": 7, "task_id": "task-1"}
    audits = repo.list_audit(kind="draft.invoke")
    assert len(audits) == 1
    assert audits[0]["target_ref"] == "draft.calc_margin"
    assert audits[0]["outcome"] == "succeeded"


def test_tool_runner_adhoc_validates_input_and_output_schema(tmp_path):
    runner = _runner(tmp_path)
    module_path = tmp_path / "draft_bad.py"
    module_path.write_text(
        "def bad_output(inputs, ctx):\n"
        "    return {'wrong': True}\n",
        encoding="utf-8",
    )
    input_schema = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
        "additionalProperties": False,
    }
    output_schema = {
        "type": "object",
        "properties": {"echoed": {"type": "string"}},
        "required": ["echoed"],
        "additionalProperties": False,
    }

    missing = runner.invoke_adhoc(
        module=module_path,
        entrypoint="bad_output",
        inputs={},
        input_schema=input_schema,
        output_schema=output_schema,
        timeout_seconds=10,
        task_id="task-1",
        mode="draft",
    )
    mismatch = runner.invoke_adhoc(
        module=module_path,
        entrypoint="bad_output",
        inputs={"message": "hi"},
        input_schema=input_schema,
        output_schema=output_schema,
        timeout_seconds=10,
        task_id="task-1",
        mode="draft",
    )

    assert missing.ok is False
    assert missing.error_kind == "schema"
    assert mismatch.ok is False
    assert mismatch.error_kind == "schema"
    assert "echoed" in mismatch.error


def test_worker_protocol_version_matches_host_runs_normally(tmp_path):
    # ARCH-5: happy path -- host and worker agree on protocol_version, tool
    # executes normally, and the result carries the worker's reported version.
    runner, repo = _runtime(tmp_path)

    result = runner.invoke(ToolRef("_sample", "echo"), {"message": "hi"}, task_id="task-1")

    assert result.ok is True
    assert result.error_kind is None
    audits = repo.list_audit(kind="tool.invoke")
    assert audits[-1]["outcome"] == "succeeded"


def test_worker_rejects_job_with_mismatched_protocol_version(tmp_path):
    # ARCH-5: the worker subprocess validates protocol_version itself before
    # doing any real work, independent of the host. Simulate an old/new host
    # sending a version the worker doesn't recognize.
    job = {
        "protocol_version": PROTOCOL_VERSION + 999,
        "module": "marvis.packs.sample.tools",
        "entrypoint": "echo",
        "inputs": {"message": "hi"},
        "task_id": "task-1",
        "datasets_root": str(tmp_path / "datasets"),
        "workspace": str(tmp_path / "workspace"),
        "side_effects": [],
        "builtin": True,
    }

    completed = subprocess.run(
        [sys.executable, "-m", "marvis.plugins.subprocess_worker"],
        input=json.dumps(job),
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=10,
        check=False,
    )

    payload = _parse_worker_result(completed.stdout)
    assert payload is not None
    assert completed.returncode != 0
    assert payload["ok"] is False
    assert payload["error_kind"] == "protocol_version_mismatch"
    assert payload["error_detail"]["kind"] == "protocol_version_mismatch"
    assert payload["error_detail"]["host_protocol_version"] == PROTOCOL_VERSION + 999
    assert payload["error_detail"]["worker_protocol_version"] == PROTOCOL_VERSION
    assert payload["worker_protocol_version"] == PROTOCOL_VERSION
    # No side effects, guards, or module loading should have run.
    assert "output" not in payload


def test_tool_runner_surfaces_typed_error_and_audit_on_worker_version_mismatch(tmp_path, monkeypatch):
    # ARCH-5: end-to-end through ToolRunner.invoke with the real subprocess
    # worker, host-side pinned to a stale protocol version via monkeypatch so
    # the worker (unpatched, current code) rejects the job. Verifies the typed
    # error_kind and that the failure is recorded in the audit log (INV-8).
    runner, repo = _runtime(tmp_path)
    monkeypatch.setattr("marvis.plugins.runner.PROTOCOL_VERSION", PROTOCOL_VERSION + 999)

    result = runner.invoke(ToolRef("_sample", "echo"), {"message": "hi"}, task_id="task-1")

    assert result.ok is False
    assert result.error_kind == "protocol_version_mismatch"
    assert "协议版本不匹配" in result.error
    assert result.error_detail["kind"] == "protocol_version_mismatch"

    audits = repo.list_audit(kind="tool.invoke")
    assert audits[-1]["outcome"] == "failed"
    assert audits[-1]["detail"]["error_kind"] == "protocol_version_mismatch"


def test_tool_runner_flags_worker_missing_version_as_mismatch(tmp_path, monkeypatch):
    # ARCH-5: host-side defense in depth -- an old worker binary predating the
    # handshake would silently ignore the unknown protocol_version job field
    # and report ok=true with no worker_protocol_version at all. The host must
    # not treat that as success.
    runner = _runner(tmp_path)

    class FakeProcess:
        pid = 123
        returncode = 0

        def communicate(self, input=None, timeout=None):
            return json.dumps({"ok": True, "output": {"echoed": "hi"}}), ""

        def poll(self):
            return self.returncode

    monkeypatch.setattr("marvis.plugins.runner.subprocess.Popen", lambda *a, **k: FakeProcess())

    result = runner.invoke(ToolRef("_sample", "echo"), {"message": "hi"}, task_id="task-1")

    assert result.ok is False
    assert result.error_kind == "protocol_version_mismatch"
    assert "未上报版本号" in result.error


def test_worker_entrypoint_import_stays_dependency_free():
    """PERF-5: the worker entrypoint must not reverse-import the DB/registry/
    modeling dependency chain at module load time. Cold start should only pay
    for the stdlib plus the lightweight ToolContext dataclass; heavy pack
    modules are imported on demand inside _run_tool/_load_module."""
    probe_lines = [
        "import sys, json",
        "before = set(sys.modules)",
        "import marvis.plugins.subprocess_worker",
        "after = set(sys.modules)",
        "new_modules = after - before",
        "print(json.dumps(sorted(new_modules)))",
    ]
    probe = "\n".join(probe_lines)
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    new_modules = json.loads(completed.stdout.strip().splitlines()[-1])
    heavy_prefixes = ("marvis.db", "sklearn", "lightgbm", "scipy", "pandas", "marvis.repositories.modeling")
    leaked = [name for name in new_modules if any(name == p or name.startswith(p + ".") for p in heavy_prefixes)]
    assert leaked == [], f"worker entrypoint import pulled in heavy modules: {leaked}"


def test_all_pack_runtimes_share_the_packruntime_base():
    """ARCH-8: strategy/data_ops/feature/modeling/analysis each used to hand-roll an
    identical five-object ``_Runtime`` (settings/datasets_root/repo/backend/registry)
    wired straight from ``ToolContext``. They now all derive from the shared
    ``marvis.plugins.sdk.PackRuntime`` base so the five-object construction can't
    re-fork pack by pack; this test pins that invariant structurally."""
    from marvis.plugins.sdk import PackRuntime

    import marvis.packs.analysis.tools as analysis_tools
    import marvis.packs.data_ops.tools as data_ops_tools
    import marvis.packs.feature.tools as feature_tools
    import marvis.packs.modeling._runtime as modeling_runtime
    import marvis.packs.strategy.tools as strategy_tools

    pack_runtimes = {
        "strategy": strategy_tools._Runtime,
        "data_ops": data_ops_tools._Runtime,
        "feature": feature_tools._Runtime,
        "modeling": modeling_runtime._Runtime,
        "analysis": analysis_tools._Runtime,
    }
    for pack_name, runtime_cls in pack_runtimes.items():
        assert issubclass(runtime_cls, PackRuntime), (
            f"{pack_name} pack's _Runtime must subclass PackRuntime (got MRO {runtime_cls.__mro__})"
        )
        assert runtime_cls is not PackRuntime, f"{pack_name} pack must define its own _Runtime subclass"

    # modeling/tools.py re-exports the same _Runtime object as a facade (ARCH-2 split).
    import marvis.packs.modeling.tools as modeling_tools

    assert modeling_tools._Runtime is modeling_runtime._Runtime


def test_tool_runner_kills_worker_when_rss_exceeds_soft_limit(tmp_path):
    runner, repo = _runtime(tmp_path)
    runner._rss_memory_limit_mb = 64

    result = runner.invoke(
        ToolRef("_sample", "memory_hog"),
        {"megabytes": 200, "hold_seconds": 5},
        task_id="task-1",
    )

    assert result.ok is False
    assert result.error_kind == "resource_limit"
    assert result.resource_limits is not None
    assert result.resource_limits["memory_limit_mb"] == 64
    assert result.resource_limits["peak_rss_mb"] > 64
    assert result.resource_limits["memory_limit_exceeded"] is True

    audits = repo.list_audit(kind="tool.invoke")
    assert audits[-1]["outcome"] == "failed"
    assert audits[-1]["detail"]["error_kind"] == "resource_limit"
    assert audits[-1]["detail"]["resource_limits"]["memory_limit_exceeded"] is True


def test_tool_runner_records_resource_usage_when_under_rss_limit(tmp_path):
    runner, repo = _runtime(tmp_path)
    runner._rss_memory_limit_mb = 4096

    result = runner.invoke(ToolRef("_sample", "echo"), {"message": "hi"}, task_id="task-1")

    assert result.ok is True
    assert result.error_kind is None
    audits = repo.list_audit(kind="tool.invoke")
    assert audits[-1]["outcome"] == "succeeded"
    assert audits[-1]["detail"]["resource_limits"]["memory_limit_applied"] in (True, False)


@pytest.mark.slow
def test_tool_runner_kills_process_tree_with_no_surviving_pids_on_rss_limit(tmp_path):
    """TST-4a: real OOM. The worker allocates real memory past a lowered soft
    RSS ceiling; assert the worker PID itself is gone from the OS process
    table (via psutil), not just that ToolResult reports failure. Complements
    the existing REL-3 test_tool_runner_kills_worker_when_rss_exceeds_soft_limit,
    which checks the ToolResult/audit fields but never confirms the OS-level
    process actually disappeared."""
    runner, repo = _runtime(tmp_path)
    runner._rss_memory_limit_mb = 200

    audits_before = len(repo.list_audit(kind="tool.invoke"))

    result = runner.invoke(
        ToolRef("_sample", "memory_hog"),
        {"megabytes": 600, "hold_seconds": 10},
        task_id="task-1",
    )

    assert result.ok is False
    assert result.error_kind == "resource_limit"
    assert result.resource_limits["memory_limit_mb"] == 200
    assert result.resource_limits["peak_rss_mb"] > 200
    assert result.resource_limits["memory_limit_exceeded"] is True
    worker_pid = result.resource_limits["pid"]
    assert worker_pid is not None

    audits = repo.list_audit(kind="tool.invoke")
    assert len(audits) == audits_before + 1
    assert audits[-1]["detail"]["resource_limits"]["peak_rss_mb"] > 200

    # INV-6: the worker PID does not survive the kill. Poll briefly -- SIGKILL
    # delivery and psutil's view of the process table are not perfectly
    # synchronous.
    deadline = time.monotonic() + 5.0
    alive = True
    while time.monotonic() < deadline:
        alive = psutil.pid_exists(worker_pid)
        if not alive:
            break
        time.sleep(0.1)
    assert not alive, f"worker pid {worker_pid} survived RSS kill"


@pytest.mark.slow
def test_tool_runner_kills_grandchild_process_on_timeout(tmp_path):
    """TST-4b: real kill, process-tree semantics. The worker tool itself
    spawns a grandchild subprocess (without start_new_session, so it stays in
    the worker's process group) and sleeps past the tool timeout. The parent
    ToolRunner kill path (_kill_worker_tree -> os.killpg) must reap the whole
    group, not just the direct child -- assert the grandchild PID is also
    gone, proving this isn't a single-process kill."""
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PluginRepository(db_path)
    plugin_root = tmp_path / "plugins"
    package = plugin_root / "grandchildpack"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "tools.py").write_text(
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "def run(inputs, ctx):\n"
        "    grandchild_script = (\n"
        "        'import os, time, pathlib; '\n"
        "        'pathlib.Path(' + repr(inputs['pid_file']) + ').write_text(str(os.getpid())); '\n"
        "        'time.sleep(60)'\n"
        "    )\n"
        "    grandchild_out = open(inputs['grandchild_log'], 'wb')\n"
        "    subprocess.Popen(\n"
        "        [sys.executable, '-c', grandchild_script],\n"
        "        stdout=grandchild_out,\n"
        "        stderr=grandchild_out,\n"
        "    )\n"
        "    time.sleep(60)\n"
        "    return {'ok': True}\n",
        encoding="utf-8",
    )
    manifest = PluginManifest(
        name="grandchildpack",
        version="0.1.0",
        display_name="Grandchild Pack",
        description="Spawns a grandchild process and hangs past timeout",
        module="grandchildpack.tools",
        python_requires="",
        tools=(
            ToolSpec(
                name="spawn_and_hang",
                summary="Spawn a grandchild then hang",
                input_schema={
                    "type": "object",
                    "properties": {
                        "pid_file": {"type": "string"},
                        "grandchild_log": {"type": "string"},
                    },
                    "required": ["pid_file", "grandchild_log"],
                    "additionalProperties": False,
                },
                output_schema={
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": False,
                },
                determinism="deterministic",
                timeout_seconds=1,
                failure_policy="fail",
                side_effects=("process:spawn", "write:workspace"),
                entrypoint="run",
            ),
        ),
        permissions=("process:spawn", "write:workspace"),
        builtin=False,
    )

    class FakeTools:
        def resolve_with_manifest(self, ref):
            assert ref == ToolRef("grandchildpack", "spawn_and_hang")
            return manifest, manifest.tools[0]

    runner = ToolRunner(
        FakeTools(),
        repo,
        python_executable=sys.executable,
        datasets_root=tmp_path / "datasets",
        workspace=tmp_path / "workspace",
        plugin_paths=[plugin_root],
    )
    pid_file = tmp_path / "workspace" / "tasks" / "task-1" / "grandchild.pid"
    grandchild_log = tmp_path / "workspace" / "tasks" / "task-1" / "grandchild.log"
    pid_file.parent.mkdir(parents=True, exist_ok=True)

    result = runner.invoke(
        ToolRef("grandchildpack", "spawn_and_hang"),
        {"pid_file": str(pid_file), "grandchild_log": str(grandchild_log)},
        task_id="task-1",
    )

    assert result.ok is False
    assert result.error_kind == "timeout"

    # Wait for the grandchild to have written its PID (it does so before
    # sleeping); if it never got a chance to run at all there is nothing to
    # verify, which would defeat the purpose of the test.
    deadline = time.monotonic() + 5.0
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert pid_file.exists(), "grandchild never started; test setup is broken"
    grandchild_pid = int(pid_file.read_text().strip())

    # INV-6: process-tree kill semantics -- the grandchild must also be dead,
    # not just the immediate worker child. Poll briefly for SIGKILL delivery.
    deadline = time.monotonic() + 5.0
    alive = True
    while time.monotonic() < deadline:
        alive = psutil.pid_exists(grandchild_pid)
        if not alive:
            break
        time.sleep(0.1)
    assert not alive, f"grandchild pid {grandchild_pid} survived process-tree kill"


def test_tool_runner_worker_subprocess_only_sees_allowlisted_env_vars(tmp_path, monkeypatch):
    """TST-4c: real isolation. Round-trip through a real worker subprocess
    (no Popen mocking) that echoes back its own os.environ key set, proving
    _WORKER_ENV_ALLOWLIST filtering actually takes effect inside the child --
    not just that the parent's Popen call was passed a filtered dict."""
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PluginRepository(db_path)
    plugin_root = tmp_path / "plugins"
    package = plugin_root / "envpack"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "tools.py").write_text(
        "import os\n"
        "def run(inputs, ctx):\n"
        "    return {'env_keys': sorted(os.environ.keys())}\n",
        encoding="utf-8",
    )
    manifest = PluginManifest(
        name="envpack",
        version="0.1.0",
        display_name="Env Pack",
        description="Echoes visible environment variable keys",
        module="envpack.tools",
        python_requires="",
        tools=(
            ToolSpec(
                name="echo_env",
                summary="Echo visible env var keys",
                input_schema={"type": "object", "additionalProperties": False},
                output_schema={
                    "type": "object",
                    "properties": {
                        "env_keys": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["env_keys"],
                    "additionalProperties": False,
                },
                determinism="deterministic",
                timeout_seconds=10,
                failure_policy="fail",
                side_effects=(),
                entrypoint="run",
            ),
        ),
        permissions=(),
        builtin=False,
    )

    class FakeTools:
        def resolve_with_manifest(self, ref):
            assert ref == ToolRef("envpack", "echo_env")
            return manifest, manifest.tools[0]

    monkeypatch.setenv("MARVIS_TEST_SECRET", "should-not-leak")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("PATH", os.environ.get("PATH", "/usr/bin"))

    runner = ToolRunner(
        FakeTools(),
        repo,
        python_executable=sys.executable,
        datasets_root=tmp_path / "datasets",
        workspace=tmp_path / "workspace",
        plugin_paths=[plugin_root],
    )

    result = runner.invoke(ToolRef("envpack", "echo_env"), {}, task_id="task-1")

    assert result.ok is True, result.error
    visible = set(result.output["env_keys"])
    assert "MARVIS_TEST_SECRET" not in visible
    assert "OPENAI_API_KEY" not in visible
    assert "PATH" in visible
    assert visible <= (_WORKER_ENV_ALLOWLIST | {"PYTHONUNBUFFERED"})
