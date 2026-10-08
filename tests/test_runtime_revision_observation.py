from copy import deepcopy
from hashlib import sha256
import time

import pytest

from marvis.runtime_observations import observe_runtime
from marvis.orchestrator.eval.runtime_process import (
    read_backend_events, backend_timing, process_observation, validate_process_observation,
)


def _snapshots(repo, plan_id):
    plan = repo.load_plan(plan_id)
    return [{"id": plan.id, "replan_count": plan.replan_count}]


def test_native_replacement_and_explore_append_are_different_committed_revisions(tmp_path):
    from marvis.db import PlanRepository, init_db
    from test_orch_plans_db_adaptive import _plan, _step

    db = tmp_path / "app.sqlite"
    init_db(db)
    repo = PlanRepository(db)
    path = tmp_path / "events"
    with observe_runtime(path, time.monotonic_ns()):
        plan = _plan()
        plan.goal = "private-goal"
        repo.create_plan(plan)
        repo.replace_remaining_steps(plan.id, _plan(_step("replacement", 0)))
        repo.append_steps(plan.id, [_step("exploration", 1)])
    result = process_observation([], [], read_backend_events(path), plan_states=_snapshots(repo, plan.id))["revisions"]
    assert result["complete"] and result["plan_denominator"] == 1
    assert result["known_counts"] == {"structural_replan": 1, "explore_append": 1, "upstream_revision": 0}
    assert result["plans"][0]["initial_revision"] == 0 and result["plans"][0]["final_revision"] == 2
    assert "private-goal" not in path.read_text() and plan.id not in path.read_text()


def test_actual_upstream_modeling_revision_is_not_a_structural_replan(tmp_path):
    from test_workflow_feature_rollback import _persist_failed_plan, EXCLUSIONS

    path = tmp_path / "events"
    with observe_runtime(path, time.monotonic_ns()):
        repo, plan = _persist_failed_plan(tmp_path)
        revised = {**next(step for step in plan.steps if step.id == "screen").inputs, "features": ["safe_a", "safe_b"]}
        repo.rollback_failed_plan_from_step(
            plan.id, "screen", "tune", root_inputs=revised, excluded_features=EXCLUSIONS,
            expected_plan_revision=0, expected_root_output_ref="metrics:screen:v1",
        )
    result = process_observation([], [], read_backend_events(path), plan_states=_snapshots(repo, plan.id))["revisions"]
    assert result["complete"]
    assert result["known_counts"] == {"structural_replan": 0, "explore_append": 0, "upstream_revision": 1}


def test_failed_plan_creation_does_not_leave_a_committed_baseline(tmp_path):
    from marvis.db import PlanRepository, init_db
    from test_orch_plans_db_adaptive import _plan

    db = tmp_path / "app.sqlite"
    init_db(db)
    repo = PlanRepository(db)
    def reject(conn):
        raise RuntimeError("private-rejection")
    path = tmp_path / "events"
    with observe_runtime(path, time.monotonic_ns()):
        with pytest.raises(RuntimeError):
            repo.create_plan(_plan(), on_connection=reject)
    events = read_backend_events(path)
    assert not any(row["event"] == "plan_revision" for row in events)
    assert process_observation([], [], events, plan_states=[])["revisions"]["complete"]


def test_revision_commit_failure_does_not_emit_a_committed_replan(tmp_path):
    import sqlite3
    from marvis.db import PlanRepository, init_db, connect
    from test_orch_plans_db_adaptive import _plan, _step

    db = tmp_path / "app.sqlite"
    init_db(db)
    repo = PlanRepository(db)
    with connect(db) as conn:
        conn.execute("CREATE TABLE revision_commit_failure(plan_id TEXT REFERENCES plans(id) DEFERRABLE INITIALLY DEFERRED)")
        conn.execute("""CREATE TRIGGER fail_revision_commit AFTER UPDATE OF replan_count ON plans
                     WHEN NEW.replan_count > OLD.replan_count
                     BEGIN INSERT INTO revision_commit_failure VALUES ('missing-plan'); END""")
    path = tmp_path / "events"
    with observe_runtime(path, time.monotonic_ns()):
        repo.create_plan(_plan())
        with pytest.raises(sqlite3.IntegrityError):
            repo.replace_remaining_steps("plan-1", _plan(_step("replacement", 0)))
    events = read_backend_events(path)
    assert [row["kind"] for row in events if row["event"] == "plan_revision"] == ["created"]
    result = process_observation([], [], events, plan_states=_snapshots(repo, "plan-1"))["revisions"]
    assert result["complete"] and result["known_counts"]["structural_replan"] == 0


def test_planner_attempt_return_and_failure_do_not_imply_committed_replanning(tmp_path):
    from test_orch_planner import _planner, _template, _replanned_steps, FakeLLM
    from marvis.orchestrator.capability import resolve_tier
    from marvis.orchestrator.contracts import StepStatus
    from marvis.orchestrator.planner import ReplanError

    llm = FakeLLM([])
    planner = _planner(tmp_path, llm)
    plan = planner.from_template(_template(), {"message": "hello"}, task_id="task-1")
    done = plan.steps[0].id
    plan.steps[0].status = StepStatus.DONE
    plan.steps[0].output_ref = f"metrics:{done}"
    llm.responses = [_replanned_steps(ref_id=done)]
    tier = resolve_tier("balanced")
    path = tmp_path / "events"
    with observe_runtime(path, time.monotonic_ns()):
        # This tests the measurement boundary, not the still-open goal/constraint
        # preservation assertions in the separate Planner regression suite.
        planner.replan(plan, completed_summaries={done: {"echoed": "hello"}},
                       observation={"echoed": "hello"}, reason="decision_point", tier=tier)
        plan.replan_count = tier.max_replan_iterations
        with pytest.raises(ReplanError):
            planner.replan(plan, completed_summaries={}, observation={}, reason="decision_point", tier=tier)
    events = read_backend_events(path)
    attempts = backend_timing(events)["scopes"]["replan_attempt"]
    assert attempts["complete"] and attempts["measured_intervals"] == 2
    assert [row["outcome"] for row in attempts["intervals"]] == ["returned", "raised"]
    revisions = process_observation([], [], events, plan_states=[])["revisions"]
    assert revisions["complete"] and revisions["known_counts"]["structural_replan"] == 0


def _events():
    identity = sha256(b"plan-1").hexdigest()
    return [
        {"event": "plan_revisions_enabled", "elapsed_ns": 0},
        {"event": "plan_revision", "elapsed_ns": 10, "identity": identity, "kind": "created", "revision": 2},
        {"event": "plan_revision", "elapsed_ns": 20, "identity": identity, "kind": "structural_replan", "revision": 3},
        {"event": "closed", "elapsed_ns": 30, "valid": True},
    ]


@pytest.mark.parametrize("change", [None, "lost_baseline", "lost_revision", "duplicate", "gap", "extra_snapshot", "lost_snapshot", "unsealed"])
def test_revisions_require_contiguous_events_and_matching_native_snapshot(change):
    events = _events()
    plans = [{"id": "plan-1", "replan_count": 3}]
    if change == "lost_baseline":
        events.pop(1)
    elif change == "lost_revision":
        events.pop(2)
    elif change == "duplicate":
        events.insert(3, events[2].copy())
    elif change == "gap":
        events[2]["revision"] = 4
        plans[0]["replan_count"] = 4
    elif change == "extra_snapshot":
        plans.append({"id": "unobserved-plan", "replan_count": 0})
    elif change == "lost_snapshot":
        plans = None
    elif change == "unsealed":
        events.pop()
    result = process_observation([], [], events, plan_states=plans)["revisions"]
    assert result["complete"] is (change is None)
    if change is None:
        assert result["known_counts"]["structural_replan"] == 1
        assert result["plans"][0]["initial_revision"] == 2


def test_summary_and_final_plan_counter_tampering_are_rejected():
    events = _events()
    plans = [{"id": "plan-1", "replan_count": 3}]
    observed = process_observation([], [], events, plan_states=plans)
    validate_process_observation(observed, [], interventions=0, duration_ms=1, backend_events=events, plan_states=plans)
    changed = deepcopy(observed)
    changed["revisions"]["known_counts"]["structural_replan"] = 0
    with pytest.raises(ValueError, match="mismatch"):
        validate_process_observation(changed, [], interventions=0, duration_ms=1, backend_events=events, plan_states=plans)
    with pytest.raises(ValueError, match="mismatch"):
        validate_process_observation(observed, [], interventions=0, duration_ms=1, backend_events=events,
                                     plan_states=[{"id": "plan-1", "replan_count": 2}])


@pytest.mark.parametrize("change", ["raw_identity", "extra_prompt", "unknown_kind", "boolean", "negative", "legacy_header"])
def test_invalid_revision_content_is_not_accepted(change):
    events = _events()
    if change == "raw_identity":
        events[1]["identity"] = "private-plan-id"
    elif change == "extra_prompt":
        events[1]["prompt"] = "private instruction"
    elif change == "unknown_kind":
        events[1]["kind"] = "private-tool"
    elif change == "boolean":
        events[1]["revision"] = True
    elif change == "negative":
        events[1]["revision"] = -1
    else:
        events[0]["event"] = "queue_terminals_enabled"
    with pytest.raises(ValueError):
        backend_timing(events)


def test_legacy_backend_summary_does_not_gain_new_scopes_or_revision_claims():
    for header in (None, "queue_terminals_enabled"):
        events = ([{"event": header, "elapsed_ns": 0}] if header else []) + [{"event": "closed", "elapsed_ns": 1, "valid": True}]
        result = process_observation([], [], events)
        assert result["schema"].endswith("v3" if header else "v2")
        assert "revisions" not in result
        assert "replan_attempt" not in result["backend"]["scopes"]
