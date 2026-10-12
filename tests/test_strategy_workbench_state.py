"""Actual local-session HTTP, immutable versions and checkpointed native searches."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import threading
import uuid

from fastapi.testclient import TestClient
import pytest

from marvis.app import create_app
from marvis.db_schema import connect
from marvis.domain import TaskCreate
from marvis.orchestrator.contracts import Plan, PlanStatus, PlanStep, StepStatus
from marvis.plugins.manifest import ToolRef
from marvis.repositories.plans import PlanRepository
from marvis.repositories.tasks import TaskRepository
from marvis.state_machine import ConflictError
from marvis.strategy_workbench.contracts import ControlRequest, WorkbenchError
from marvis.strategy_workbench.control import (
    NativeSearchControl,
    SearchInterrupted,
    control_search,
)
from marvis.strategy_workbench.service import register_plan
from tests.test_production_governance_api import _claim_role


def _client(app, name="maker"):
    client = TestClient(app)
    principal = _claim_role(app, client, display_name=name, role="maker")
    return client, principal["id"]


def _empty(tmp_path):
    app = create_app(tmp_path / "workspace")
    task = TaskRepository(app.state.settings.db_path).create_task(
        TaskCreate(
            model_name="工作台合成验收",
            model_version="dev",
            validator="synthetic",
            source_dir=str(app.state.settings.workspace),
            task_type="strategy",
        )
    )
    client, actor = _client(app)
    return app, task.id, client, actor


def _draft():
    return {
        "schema_version": "strategy.workbench-draft.v1",
        "fields": [
            {
                "workflow": "strategy_project_context",
                "field": "project_context_scope",
                "kind": "narrative",
                "values": ["合成审批场景"],
            },
            {
                "workflow": "strategy_project_context",
                "field": "project_context_business_context",
                "kind": "narrative",
                "values": ["project.channel=合成自营\nproject.product=合成复借"],
            },
            {
                "workflow": "interactive_tree_split_search",
                "field": "interactive_tree_search_max_thresholds",
                "kind": "scalar",
                "values": ["5"],
            },
        ],
        "open_workflows": ["strategy_project_context"],
        "scroll_top": 42,
    }


def _save(client, task, binding, draft=None):
    return client.put(
        f"/api/tasks/{task}/strategy-workbench/draft",
        json={
            "expected_revision": binding["head_revision"],
            "evidence_hash": binding["evidence_hash"],
            "draft": draft or _draft(),
        },
    )


def test_http_draft_actor_scope_cas_history_and_restart(tmp_path):
    app, task, client, actor = _empty(tmp_path)
    url = f"/api/tasks/{task}/strategy-workbench/draft"
    binding = client.get(url).json()
    assert binding["actor_id"] == actor and binding["head_revision"] == 0
    first = _save(client, task, binding)
    assert first.status_code == 200, first.text
    assert first.json()["head_revision"] == 1
    assert _save(client, task, binding).status_code == 409
    second_draft = _draft()
    second_draft["fields"][0]["values"] = ["修订审批场景"]
    second = _save(client, task, first.json(), second_draft)
    assert second.status_code == 200
    assert client.get(url + "?revision=1").json()["draft"] == _draft()
    other, other_actor = _client(app, "second")
    assert other_actor != actor
    assert other.get(url).json()["draft"] is None
    rebuilt = TestClient(create_app(app.state.settings))
    rebuilt.cookies.update(client.cookies)
    assert rebuilt.get(url).json()["draft"] == second_draft
    with connect(app.state.settings.db_path) as conn:
        with pytest.raises(Exception, match="immutable"):
            conn.execute(
                "UPDATE strategy_workbench_drafts SET payload='{}' WHERE task_id=?",
                (task,),
            )


def test_http_draft_concurrent_cas_exactly_one_winner(tmp_path):
    app, task, client, actor = _empty(tmp_path)
    binding = client.get(f"/api/tasks/{task}/strategy-workbench/draft").json()
    barrier = threading.Barrier(2)

    def save(name):
        peer = TestClient(app)
        peer.cookies.update(client.cookies)
        draft = _draft()
        draft["fields"][0]["values"] = [name]
        barrier.wait()
        return _save(peer, task, binding, draft).status_code

    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(save, "合成甲"), workers.submit(save, "合成乙")]
        statuses = [f.result() for f in futures]
    assert sorted(statuses) == [200, 409]
    with connect(app.state.settings.db_path) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM strategy_workbench_drafts WHERE actor_id=?",
                (actor,),
            ).fetchone()[0]
            == 1
        )


@pytest.mark.parametrize(
    "mutation", ["secret", "raw_rows", "fake_metric", "extra_actor"]
)
def test_http_draft_rejects_sensitive_or_nonform_payload(tmp_path, mutation):
    app, task, client, _ = _empty(tmp_path)
    binding = client.get(f"/api/tasks/{task}/strategy-workbench/draft").json()
    draft = _draft()
    if mutation == "secret":
        draft["fields"][0]["values"] = ["password=synthetic-secret-only"]
    if mutation == "raw_rows":
        draft["rows"] = [{"x": 1}]
    if mutation == "fake_metric":
        draft["fields"].append(
            {
                "workflow": "strategy_project_context",
                "field": "measured_ks",
                "kind": "scalar",
                "values": ["0.99"],
            }
        )
    if mutation == "extra_actor":
        draft["actor_id"] = "another-user"
    assert _save(client, task, binding, draft).status_code == 422
    assert binding["narrative_is_evidence"] is False


def _prepare(fixture, client, actor, inputs, tool="search_cross_threshold_rules"):
    settings = fixture["settings"]
    task = fixture["task"].id
    repo = PlanRepository(settings.db_path)
    plan_id, step_id = uuid.uuid4().hex, uuid.uuid4().hex
    step = PlanStep(
        step_id, plan_id, 0, "合成受控搜索", ToolRef("strategy", tool), inputs, [], []
    )
    plan = Plan(
        plan_id, task, "合成候选搜索", "template", None, [step], 1, PlanStatus.RUNNING
    )
    repo.create_plan(plan)
    register_plan(settings, plan_id, actor)
    step.status = StepStatus.RUNNING
    repo.update_step(step)
    runner = fixture["runner"]
    contract = runner.prepare_invocation(step.tool_ref)
    run_id = repo.start_step_run(
        plan_id=plan_id,
        step_id=step_id,
        tool_ref=step.tool_ref.label(),
        inputs=inputs,
        invocation_contract=contract,
    )
    repo.mark_step_run_dispatched(run_id)
    return repo, plan, step, run_id


def _cross(tmp_path):
    from tests.test_strategy_cross_matrix_candidate_tool import _setup
    from tests.test_strategy_cross_rule_search_tools import _inputs

    fixture = _setup(tmp_path, with_split=True)
    app = create_app(fixture["settings"])
    client, actor = _client(app)
    return fixture, app, client, actor, _inputs(fixture, max_trials=20)


def _execution(client, task, search):
    response = client.get(f"/api/tasks/{task}/strategy-workbench/executions")
    assert response.status_code == 200, response.text
    payload = response.json()
    return next(item for item in payload["items"] if item["id"] == search), payload[
        "evidence_hash"
    ]


def _action(client, task, search, action):
    row, evidence = _execution(client, task, search)
    return client.post(
        f"/api/tasks/{task}/strategy-workbench/executions/{search}/{action}",
        json={"expected_revision": row["revision"], "evidence_hash": evidence},
    )


def test_native_cross_pause_after_candidate_resume_skips_prefix_and_publishes_once(
    tmp_path, monkeypatch
):
    import marvis.packs.strategy.cross_rule_search_tools as tools

    fixture, app, client, actor, inputs = _cross(tmp_path)
    repo, plan, step, run_id = _prepare(fixture, client, actor, inputs)
    ctx = replace(fixture["ctx"], invocation_id=run_id)
    reached, proceed = threading.Event(), threading.Event()
    measured = []
    original = tools._measure_trial

    def measure(*args, **kwargs):
        value = original(*args, **kwargs)
        measured.append(value)
        if len(measured) == 2:
            reached.set()
            assert proceed.wait(10)
        return value

    monkeypatch.setattr(tools, "_measure_trial", measure)
    with ThreadPoolExecutor(max_workers=1) as workers:
        future = workers.submit(
            tools.run_search_cross_threshold_rules, inputs, ctx, fixture["runtime"]
        )
        assert reached.wait(10)
        foreign, _ = _client(app, "foreign")
        assert (
            foreign.post(
                f"/api/tasks/{fixture['task'].id}/strategy-workbench/executions/{step.id}/pause",
                json={"expected_revision": 0, "evidence_hash": "0" * 64},
            ).status_code
            == 404
        )
        assert _action(client, fixture["task"].id, step.id, "pause").status_code == 202
        proceed.set()
        with pytest.raises(SearchInterrupted):
            future.result()
    paused, evidence = _execution(client, fixture["task"].id, step.id)
    assert paused["state"] == "paused" and paused["completed"] == 2
    assert repo.step_run_retry_safe(run_id)
    with connect(fixture["settings"].db_path) as conn:
        checkpoint = conn.execute(
            "SELECT checkpoint_json FROM strategy_workbench_searches WHERE id=?",
            (step.id,),
        ).fetchone()[0]
        assert all(
            key not in checkpoint for key in ('"rows"', '"target"', '"hit_matrix"')
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM task_artifacts WHERE origin_tool='strategy.search_cross_threshold_rules'"
            ).fetchone()[0]
            == 0
        )
    # Same native plan retry: verified paused receipt permits safe reset only.
    repo.finish_step_run(
        run_id, status="failed", error_kind="search_paused", error="paused"
    )
    step.status = StepStatus.FAILED
    repo.update_step(step)
    repo.set_plan_status(plan.id, PlanStatus.FAILED)
    resumed = _action(client, fixture["task"].id, step.id, "resume")
    assert resumed.status_code == 202, resumed.text
    completed, _ = _execution(client, fixture["task"].id, step.id)
    assert (
        completed["state"] == "completed"
        and completed["completed"] == completed["total"] == 8
    )
    with connect(fixture["settings"].db_path) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM task_artifacts WHERE origin_tool='strategy.search_cross_threshold_rules'"
            ).fetchone()[0]
            == 1
        )
        runs = conn.execute(
            "SELECT attempt,status FROM plan_step_runs WHERE step_id=? ORDER BY attempt",
            (step.id,),
        ).fetchall()
        assert [tuple(row) for row in runs] == [(1, "failed"), (2, "succeeded")]
    # Fresh uncheckpointed native deterministic calculation has identical results.
    fresh = tools.run_search_cross_threshold_rules(
        inputs, fixture["ctx"], fixture["runtime"]
    )
    stored = json.loads(completed["result_json"])
    assert stored[0]["content_hash"] == fresh["artifacts"][0]["content_hash"]


def test_native_cancel_has_zero_publication_and_cannot_resume(tmp_path, monkeypatch):
    import marvis.packs.strategy.cross_rule_search_tools as tools

    fixture, app, client, actor, inputs = _cross(tmp_path)
    _, _, step, run_id = _prepare(fixture, client, actor, inputs)
    ctx = replace(fixture["ctx"], invocation_id=run_id)
    reached, proceed = threading.Event(), threading.Event()
    original = tools._measure_trial

    def measure(*args, **kwargs):
        value = original(*args, **kwargs)
        reached.set()
        assert proceed.wait(10)
        return value

    monkeypatch.setattr(tools, "_measure_trial", measure)
    with ThreadPoolExecutor(max_workers=1) as workers:
        future = workers.submit(
            tools.run_search_cross_threshold_rules, inputs, ctx, fixture["runtime"]
        )
        assert reached.wait(10)
        assert _action(client, fixture["task"].id, step.id, "cancel").status_code == 202
        proceed.set()
        with pytest.raises(SearchInterrupted):
            future.result()
    row, _ = _execution(client, fixture["task"].id, step.id)
    assert row["state"] == "cancelled" and row["completed"] == 1
    assert _action(client, fixture["task"].id, step.id, "resume").status_code == 409
    with connect(fixture["settings"].db_path) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM task_artifacts WHERE origin_tool='strategy.search_cross_threshold_rules'"
            ).fetchone()[0]
            == 0
        )


def test_native_control_rejects_caller_checkpoint_and_publication_pause(tmp_path):
    fixture, app, client, actor, inputs = _cross(tmp_path)
    _, _, step, run_id = _prepare(fixture, client, actor, inputs)
    control = NativeSearchControl(
        fixture["runtime"], replace(fixture["ctx"], invocation_id=run_id), inputs
    )
    control.bind_search("synthetic-native-identity")
    control.boundary([{"count": 8}], 1)
    control.publishing()
    assert _action(client, fixture["task"].id, step.id, "pause").status_code == 409
    row, evidence = _execution(client, fixture["task"].id, step.id)
    payload = {
        "expected_revision": row["revision"],
        "evidence_hash": evidence,
        "checkpoint": {"cursor": 0},
    }
    assert (
        client.post(
            f"/api/tasks/{fixture['task'].id}/strategy-workbench/executions/{step.id}/resume",
            json=payload,
        ).status_code
        == 422
    )


@pytest.mark.parametrize("action,kind", [("pause", "search_paused"), ("cancel", "search_cancelled")])
def test_native_halt_blocks_generic_retry_and_autonomous_replan(tmp_path, action, kind):
    from marvis.agent.plan_message_composer import PlanMessageComposer
    from marvis.orchestrator.executor import PlanExecutor
    from marvis.orchestrator.harness_state import HarnessState
    from tests.test_orch_executor import FakeAdaptivePlanner, FakeHooks

    fixture, _, client, actor, inputs = _cross(tmp_path)
    repo, plan, step, run_id = _prepare(fixture, client, actor, inputs)
    control = NativeSearchControl(
        fixture["runtime"], replace(fixture["ctx"], invocation_id=run_id), inputs
    )
    control.bind_search("synthetic-native-halt")
    control.boundary([], 2)
    assert _action(client, fixture["task"].id, step.id, action).status_code == 202
    with pytest.raises(SearchInterrupted):
        control.boundary([{"measured": 1}], 2)
    repo.finish_step_run(run_id, status="failed", error_kind=kind, error="native halt")
    step.status = StepStatus.FAILED
    step.error = "native halt"
    repo.update_step(step)
    with connect(repo.db_path) as conn:
        conn.execute("UPDATE plans SET tier='autonomous' WHERE id=?", (plan.id,))
    planner = FakeAdaptivePlanner()
    executor = PlanExecutor(
        repo, fixture["runner"], None, None, FakeHooks(), HarnessState(repo), planner=planner
    )
    result = executor.run(plan.id)
    assert result.status == PlanStatus.FAILED
    assert not planner.replan_calls
    with pytest.raises(ConflictError, match="native search requires"):
        repo.retry_failed_step(plan.id, step.id, preserve_target_confirmation=True)
    message = PlanMessageComposer(
        load_output=repo.load_step_output,
        latest_failed_step_run_error_kind=repo.latest_failed_step_run_error_kind,
    ).failed_message(repo.load_plan(plan.id), run_seq=1)
    assert message.metadata["failure_envelope"]["error_kind"] == kind
    assert message.metadata["failure_envelope"]["retryable"] is False
    assert message.metadata["error_diagnostic"]["retryable"] is False
    assert message.metadata["workbench_search"]["search_id"] == step.id
    assert "Agent 从当前失败步骤继续" not in message.content
    assert "工作台" in message.content if action == "pause" else "新的搜索" in message.content


def test_evidence_changed_restoration_clears_selection_and_rejects_old_save(tmp_path):
    from marvis.data.backend import DataBackend
    from marvis.data.registry import DatasetRegistry
    from marvis.repositories.datasets import DatasetRepository
    import pandas as pd

    app, task, client, _ = _empty(tmp_path)
    url = f"/api/tasks/{task}/strategy-workbench/draft"
    binding = client.get(url).json()
    draft = _draft()
    draft["fields"].extend([
        {
            "workflow": "interactive_tree_split_search",
            "field": "interactive_tree_search_mode",
            "kind": "select",
            "values": ["all_features"],
        },
        {
            "workflow": "univariate_candidate_analysis",
            "field": "methods",
            "kind": "checks",
            "values": ["equal_frequency", "tree"],
        },
        {
            "workflow": "strategy_pool_add_candidate",
            "field": "pool_add_strategy_type",
            "kind": "select",
            "values": ["approval"],
        },
        {
            "workflow": "interactive_tree_split_search",
            "field": "interactive_tree_search_source_id",
            "kind": "select",
            "values": ["x"],
        },
        {
            "workflow": "strategy_pool_add_candidate",
            "field": "pool_add_source_id",
            "kind": "select",
            "values": [""],
        },
    ])
    saved = _save(client, task, binding, draft)
    assert saved.status_code == 200, saved.text
    path = app.state.settings.workspace / "synthetic.parquet"
    pd.DataFrame({"x": [1.0, 2.0], "bad": [0, 1]}).to_parquet(path)
    registry = DatasetRegistry(
        DatasetRepository(app.state.settings.db_path),
        DataBackend(app.state.settings.datasets_dir),
        app.state.settings.datasets_dir,
    )
    registry.register_from_upload(task, path, role="sample")
    restored = client.get(url).json()
    assert restored["saved_evidence_hash"] != restored["evidence_hash"]
    assert restored["invalidated_selections"] == [
        {"workflow": "interactive_tree_split_search", "field": "interactive_tree_search_source_id"}
    ]
    fields = {f["field"]: f["values"] for f in restored["draft"]["fields"]}
    assert fields["interactive_tree_search_mode"] == ["all_features"]
    assert fields["methods"] == ["equal_frequency", "tree"]
    assert fields["pool_add_strategy_type"] == ["approval"]
    assert fields["interactive_tree_search_source_id"] == []
    assert fields["pool_add_source_id"] == [""]
    assert any(f["field"] == "pool_add_strategy_type" and "approval" in f["values"] for f in restored["static_choices"])
    assert restored["draft"]["fields"][0]["values"] == ["合成审批场景"]
    assert _save(client, task, saved.json(), draft).status_code == 409


@pytest.mark.parametrize("workflow,field,kind", [
    ("interactive_tree_split_search", "interactive_tree_search_mode", "select"),
    ("univariate_candidate_analysis", "methods", "checks"),
])
def test_source_installed_static_choices_reject_forged_values_without_a_version(tmp_path, workflow, field, kind):
    _, task, client, _ = _empty(tmp_path)
    url = f"/api/tasks/{task}/strategy-workbench/draft"
    binding = client.get(url).json()
    draft = _draft()
    draft["fields"].append({"workflow": workflow, "field": field, "kind": kind, "values": ["forged-option"]})
    rejected = _save(client, task, binding, draft)
    assert rejected.status_code == 422
    assert rejected.json()["detail"]["code"] == "workbench_draft_choice_invalid"
    assert client.get(url).json()["head_revision"] == 0


def test_native_resume_reuses_aggregate_prefix_without_remeasuring(
    tmp_path, monkeypatch
):
    import marvis.packs.strategy.cross_rule_search_tools as tools

    fixture, _, client, actor, inputs = _cross(tmp_path)
    repo, plan, step, run_id = _prepare(fixture, client, actor, inputs)
    original_measure = tools._measure_trial
    calls = []

    def measure(*args, **kwargs):
        calls.append(1)
        return original_measure(*args, **kwargs)

    original_boundary = NativeSearchControl.boundary

    def boundary(control, aggregates, total):
        if control.id == step.id and len(aggregates) == 2 and len(calls) == 2:
            assert (
                _action(client, fixture["task"].id, step.id, "pause").status_code == 202
            )
        return original_boundary(control, aggregates, total)

    monkeypatch.setattr(tools, "_measure_trial", measure)
    monkeypatch.setattr(NativeSearchControl, "boundary", boundary)
    with pytest.raises(SearchInterrupted):
        tools.run_search_cross_threshold_rules(
            inputs, replace(fixture["ctx"], invocation_id=run_id), fixture["runtime"]
        )
    assert len(calls) == 2
    row, evidence = _execution(client, fixture["task"].id, step.id)
    repo.finish_step_run(
        run_id, status="failed", error_kind="search_paused", error="paused"
    )
    step.status = StepStatus.FAILED
    repo.update_step(step)
    repo.set_plan_status(plan.id, PlanStatus.FAILED)
    control_search(
        fixture["settings"],
        fixture["task"].id,
        actor,
        step.id,
        "resume",
        ControlRequest(expected_revision=row["revision"], evidence_hash=evidence),
    )
    repo.retry_failed_step(plan.id, step.id)
    step.status = StepStatus.RUNNING
    repo.update_step(step)
    resumed_run = repo.start_step_run(
        plan_id=plan.id,
        step_id=step.id,
        tool_ref=step.tool_ref.label(),
        inputs=inputs,
        invocation_contract=fixture["runner"].prepare_invocation(step.tool_ref),
    )
    repo.mark_step_run_dispatched(resumed_run)
    monkeypatch.setattr(NativeSearchControl, "boundary", original_boundary)
    output = tools.run_search_cross_threshold_rules(
        inputs, replace(fixture["ctx"], invocation_id=resumed_run), fixture["runtime"]
    )
    assert output["evaluated"] == 8 and len(calls) == 8  # Exactly six new candidates.


def test_control_cas_pause_vs_cancel_and_auth_revocation(tmp_path):
    fixture, app, client, actor, inputs = _cross(tmp_path)
    _, _, step, run_id = _prepare(fixture, client, actor, inputs)
    control = NativeSearchControl(
        fixture["runtime"], replace(fixture["ctx"], invocation_id=run_id), inputs
    )
    control.bind_search("synthetic-measurement")
    control.boundary([], 8)
    row, evidence = _execution(client, fixture["task"].id, step.id)
    barrier = threading.Barrier(2)

    def action(action):
        peer = TestClient(app)
        peer.cookies.update(client.cookies)
        barrier.wait()
        return peer.post(
            f"/api/tasks/{fixture['task'].id}/strategy-workbench/executions/{step.id}/{action}",
            json={"expected_revision": row["revision"], "evidence_hash": evidence},
        ).status_code

    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(action, "pause"), workers.submit(action, "cancel")]
        assert sorted(f.result() for f in futures) == [202, 409]
    with connect(fixture["settings"].db_path) as conn:
        conn.execute(
            "UPDATE production_principals SET status='revoked' WHERE local_principal_id=?",
            (actor,),
        )
    assert (
        client.get(
            f"/api/tasks/{fixture['task'].id}/strategy-workbench/executions"
        ).status_code
        == 403
    )


def test_checkpoint_authentication_and_changed_inputs_cannot_resume(
    tmp_path, monkeypatch
):
    fixture, _, client, actor, inputs = _cross(tmp_path)
    repo, plan, step, run_id = _prepare(fixture, client, actor, inputs)
    native = NativeSearchControl(
        fixture["runtime"], replace(fixture["ctx"], invocation_id=run_id), inputs
    )
    native.bind_search("synthetic-owned-frame")
    native.boundary([], 8)
    assert _action(client, fixture["task"].id, step.id, "pause").status_code == 202
    with pytest.raises(SearchInterrupted):
        native.boundary([{"count": 8}], 8)
    row, evidence = _execution(client, fixture["task"].id, step.id)
    repo.finish_step_run(
        run_id, status="failed", error_kind="search_paused", error="paused"
    )
    step.status = StepStatus.FAILED
    repo.update_step(step)
    repo.set_plan_status(plan.id, PlanStatus.FAILED)
    control_search(
        fixture["settings"],
        fixture["task"].id,
        actor,
        step.id,
        "resume",
        ControlRequest(expected_revision=row["revision"], evidence_hash=evidence),
    )
    repo.retry_failed_step(plan.id, step.id)
    step.status = StepStatus.RUNNING
    repo.update_step(step)
    contract = fixture["runner"].prepare_invocation(step.tool_ref)
    resumed = repo.start_step_run(
        plan_id=plan.id,
        step_id=step.id,
        tool_ref=step.tool_ref.label(),
        inputs=inputs,
        invocation_contract=contract,
    )
    repo.mark_step_run_dispatched(resumed)
    # Same-UID database access is not an identity boundary. Even so, a changed
    # aggregate cannot be presented to the native resumed producer as genuine.
    with connect(fixture["settings"].db_path) as conn:
        conn.execute(
            "UPDATE strategy_workbench_searches SET checkpoint_json=? WHERE id=?",
            (
                json.dumps(
                    {
                        "search_id": step.id,
                        "input_hash": native.input_hash,
                        "total": 8,
                        "aggregates": [{"count": 0}],
                    }
                ),
                step.id,
            ),
        )
    with pytest.raises(WorkbenchError, match="authentication_failed"):
        NativeSearchControl(
            fixture["runtime"], replace(fixture["ctx"], invocation_id=resumed), inputs
        )


def test_resume_rejects_data_identity_change_before_reusing_cursor(tmp_path):
    fixture, _, client, actor, inputs = _cross(tmp_path)
    _, _, step, run_id = _prepare(fixture, client, actor, inputs)
    control = NativeSearchControl(
        fixture["runtime"], replace(fixture["ctx"], invocation_id=run_id), inputs
    )
    control.bind_search({"native-frame-hash": "original"})
    with pytest.raises(WorkbenchError, match="data_changed"):
        control.bind_search({"native-frame-hash": "changed"})


@pytest.mark.parametrize("family", ["voting", "cross_matrix", "interactive_tree"])
def test_all_supported_native_searches_checkpoint_then_resume_same_artifact(
    tmp_path, monkeypatch, family
):
    if family == "voting":
        from tests.test_strategy_voting_candidate_search_tools import _search_fixture
        from marvis.packs.strategy.voting_candidate_search_tools import (
            resolve_voting_candidate_search_inputs,
            run_search_voting_candidates,
        )

        fixture = _search_fixture(tmp_path)
        inputs = resolve_voting_candidate_search_inputs(
            fixture["runtime"],
            task_id=fixture["task"].id,
            user_controls=fixture["controls"],
        )
        native = run_search_voting_candidates
        tool = "search_voting_candidates"
    elif family == "cross_matrix":
        from tests.test_strategy_cross_matrix_candidate_tool import _setup
        from tests.test_strategy_cross_candidate_search_tools import _search_inputs
        from marvis.packs.strategy.cross_candidate_search_tools import (
            run_search_cross_matrix_candidates,
        )

        fixture = _setup(tmp_path, with_split=True)
        inputs = _search_inputs(fixture)
        native = run_search_cross_matrix_candidates
        tool = "search_cross_matrix_candidates"
    else:
        from tests.test_strategy_interactive_tree_tool import scenario
        from tests.test_strategy_interactive_tree_split_search_tool import _runtime
        from marvis.packs.strategy.interactive_tree_split_search_tools import (
            run_search_interactive_tree_split_candidates,
        )

        source = scenario.__wrapped__(tmp_path)
        fixture = {
            "settings": source.settings,
            "task": source.task,
            "ctx": source.ctx,
            "runtime": _runtime(source),
        }
        inputs = {
            "source_tree_id": source.source_asset["asset_id"],
            "node_id": source.source_asset["tree_result"]["tree"]["root_node_id"],
            "mode": "all_features",
            "max_thresholds_per_feature": 5,
            "max_row_evaluations": 2000,
        }
        native = run_search_interactive_tree_split_candidates
        tool = "search_interactive_tree_split_candidates"
    app = create_app(fixture["settings"])
    fixture["runner"] = app.state.tool_runner
    client, actor = _client(app)
    repo, plan, step, run_id = _prepare(fixture, client, actor, inputs, tool)
    old_boundary = NativeSearchControl.boundary
    stopped = False

    def boundary(control, aggregates, total):
        nonlocal stopped
        if control.id == step.id and len(aggregates) == 1 and not stopped:
            stopped = True
            assert (
                _action(client, fixture["task"].id, step.id, "pause").status_code == 202
            )
        return old_boundary(control, aggregates, total)

    monkeypatch.setattr(NativeSearchControl, "boundary", boundary)
    with pytest.raises(SearchInterrupted):
        native(
            inputs, replace(fixture["ctx"], invocation_id=run_id), fixture["runtime"]
        )
    paused, _ = _execution(client, fixture["task"].id, step.id)
    assert paused["state"] == "paused" and paused["completed"] == 1
    repo.finish_step_run(
        run_id, status="failed", error_kind="search_paused", error="paused"
    )
    step.status = StepStatus.FAILED
    repo.update_step(step)
    repo.set_plan_status(plan.id, PlanStatus.FAILED)
    response = _action(client, fixture["task"].id, step.id, "resume")
    assert response.status_code == 202, response.text
    completed, _ = _execution(client, fixture["task"].id, step.id)
    assert (
        completed["state"] == "completed"
        and completed["completed"] == completed["total"]
    )
    fresh = native(inputs, fixture["ctx"], fixture["runtime"])
    assert (
        json.loads(completed["result_json"])[0]["content_hash"]
        == fresh["artifacts"][0]["content_hash"]
    )
    with connect(fixture["settings"].db_path) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM task_artifacts WHERE origin_tool=?",
                ("strategy." + tool,),
            ).fetchone()[0]
            == 1
        )


def test_active_source_change_can_stop_but_cannot_resume_old_checkpoint(tmp_path):
    from marvis.data.backend import DataBackend
    from marvis.data.registry import DatasetRegistry
    from marvis.repositories.datasets import DatasetRepository
    import pandas as pd

    fixture, app, client, actor, inputs = _cross(tmp_path)
    repo, plan, step, run_id = _prepare(fixture, client, actor, inputs)
    control = NativeSearchControl(
        fixture["runtime"], replace(fixture["ctx"], invocation_id=run_id), inputs
    )
    control.bind_search("synthetic-first-frame")
    control.boundary([], 8)
    old, _ = _execution(client, fixture["task"].id, step.id)
    path = app.state.settings.workspace / "new-synthetic.parquet"
    pd.DataFrame({"x": [1.0, 2.0], "bad": [0, 1]}).to_parquet(path)
    registry = DatasetRegistry(
        DatasetRepository(app.state.settings.db_path),
        DataBackend(app.state.settings.datasets_dir),
        app.state.settings.datasets_dir,
    )
    registry.register_from_upload(fixture["task"].id, path, role="sample")
    payload = {
        "expected_revision": old["revision"],
        "evidence_hash": old["evidence_hash"],
    }
    url = f"/api/tasks/{fixture['task'].id}/strategy-workbench/executions/{step.id}"
    assert client.post(url + "/pause", json=payload).status_code == 202
    with pytest.raises(SearchInterrupted):
        control.boundary([{"count": 8}], 8)
    repo.finish_step_run(
        run_id, status="failed", error_kind="search_paused", error="paused"
    )
    step.status = StepStatus.FAILED
    repo.update_step(step)
    repo.set_plan_status(plan.id, PlanStatus.FAILED)
    row, _ = _execution(client, fixture["task"].id, step.id)
    payload["expected_revision"] = row["revision"]
    assert client.post(url + "/resume", json=payload).status_code == 409


@pytest.mark.parametrize("mode", ["physical", "business", "both"])
def test_field_display_preference_is_actor_bound_and_typed(tmp_path, mode):
    app, task, client, actor = _empty(tmp_path)
    binding = client.get(f"/api/tasks/{task}/strategy-workbench/draft").json()
    draft = _draft()
    draft["fields"].append(
        {
            "workflow": "strategy_workbench_preferences",
            "field": "field_display_mode",
            "kind": "select",
            "values": [mode],
        }
    )
    saved = _save(client, task, binding, draft)
    assert saved.status_code == 200, saved.text
    assert saved.json()["draft"]["fields"][-1]["values"] == [mode]
    invalid = _draft()
    invalid["fields"].append(
        {
            "workflow": "strategy_workbench_preferences",
            "field": "field_display_mode",
            "kind": "select",
            "values": ["guessed-language"],
        }
    )
    assert _save(client, task, saved.json(), invalid).status_code == 422


def test_changed_native_producer_rejects_resume_before_plan_reset(
    tmp_path, monkeypatch
):
    import marvis.strategy_workbench.control as controls

    fixture, _, client, actor, inputs = _cross(tmp_path)
    repo, plan, step, run_id = _prepare(fixture, client, actor, inputs)
    native = NativeSearchControl(
        fixture["runtime"], replace(fixture["ctx"], invocation_id=run_id), inputs
    )
    native.bind_search("synthetic-frame")
    native.boundary([], 8)
    assert _action(client, fixture["task"].id, step.id, "pause").status_code == 202
    with pytest.raises(SearchInterrupted):
        native.boundary([{"count": 8}], 8)
    repo.finish_step_run(
        run_id, status="failed", error_kind="search_paused", error="paused"
    )
    step.status = StepStatus.FAILED
    repo.update_step(step)
    repo.set_plan_status(plan.id, PlanStatus.FAILED)
    monkeypatch.setattr(
        controls, "producer_hash", lambda: "changed-native-implementation"
    )
    response = _action(client, fixture["task"].id, step.id, "resume")
    assert response.status_code == 409
    assert repo.load_plan(plan.id).status == PlanStatus.FAILED
    assert _execution(client, fixture["task"].id, step.id)[0]["state"] == "paused"


@pytest.mark.parametrize("tamper", ["actor_header", "aggregate"])
def test_corrupt_checkpoint_is_denied_before_original_plan_reset(tmp_path, tamper):
    fixture, _, client, actor, inputs = _cross(tmp_path)
    repo, plan, step, run_id = _prepare(fixture, client, actor, inputs)
    native = NativeSearchControl(
        fixture["runtime"], replace(fixture["ctx"], invocation_id=run_id), inputs
    )
    native.bind_search("synthetic-owned-frame")
    native.boundary([], 8)
    assert _action(client, fixture["task"].id, step.id, "pause").status_code == 202
    with pytest.raises(SearchInterrupted):
        native.boundary([{"count": 8}], 8)
    repo.finish_step_run(
        run_id, status="failed", error_kind="search_paused", error="paused"
    )
    step.status = StepStatus.FAILED
    repo.update_step(step)
    repo.set_plan_status(plan.id, PlanStatus.FAILED)
    with connect(fixture["settings"].db_path) as conn:
        row = conn.execute(
            "SELECT checkpoint_json FROM strategy_workbench_searches WHERE id=?",
            (step.id,),
        ).fetchone()
        damaged = json.loads(row[0])
        if tamper == "actor_header":
            damaged["actor_id"] = "different-owner"
        else:
            damaged["aggregates"][0]["count"] = 0
        conn.execute(
            "UPDATE strategy_workbench_searches SET checkpoint_json=? WHERE id=?",
            (json.dumps(damaged), step.id),
        )
    assert _action(client, fixture["task"].id, step.id, "resume").status_code == 409
    assert repo.load_plan(plan.id).status == PlanStatus.FAILED
    with connect(fixture["settings"].db_path) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM plan_step_runs WHERE step_id=?", (step.id,)
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT state FROM strategy_workbench_searches WHERE id=?", (step.id,)
            ).fetchone()[0]
            == "paused"
        )
