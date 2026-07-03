from types import SimpleNamespace

from marvis.orchestrator.context.budget import fit_to_budget
from marvis.orchestrator.context.ledger import build_progress_ledger
from marvis.orchestrator.context.observation import summarize_failure, summarize_output
from marvis.orchestrator.contracts import Plan, PlanStatus, PlanStep, StepStatus
from marvis.plugins.manifest import ToolRef


def test_summarize_output_keeps_scalars_and_shapes_large_arrays():
    tool = SimpleNamespace(output_schema={
        "type": "object",
        "properties": {
            "ks": {"type": "number"},
            "rows": {"type": "array"},
            "details": {"type": "object"},
        },
    })

    summary = summarize_output(
        {
            "ks": 0.42,
            "rows": [{"x": 1}, {"x": 2}, {"x": 3}],
            "details": {"a": 1, "b": 2},
        },
        tool,
        max_chars=500,
    )

    assert summary["ks"] == 0.42
    assert summary["rows"] == {"len": 3, "head": [{"x": 1}, {"x": 2}]}
    assert summary["details"] == {"type": "object", "keys": ["a", "b"]}


def test_summarize_output_handles_json_schema_union_types():
    tool = SimpleNamespace(output_schema={
        "type": "object",
        "properties": {
            "score_col": {"type": ["string", "null"]},
            "recommended": {"type": ["object", "null"]},
            "points": {"type": ["array", "null"]},
        },
    })

    summary = summarize_output(
        {
            "score_col": "score",
            "recommended": {"cutoff": 620, "bad_rate": 0.1},
            "points": [{"cutoff": 600}, {"cutoff": 620}, {"cutoff": 640}],
        },
        tool,
        max_chars=500,
    )

    assert summary["score_col"] == "score"
    assert summary["recommended"] == {"type": "object", "keys": ["bad_rate", "cutoff"]}
    assert summary["points"] == {"len": 3, "head": [{"cutoff": 600}, {"cutoff": 620}]}


def test_summarize_failure_is_bounded():
    summary = summarize_failure("x" * 100, "schema", max_chars=12)

    assert summary == {"error_kind": "schema", "error": "x" * 12}


def test_progress_ledger_anchors_goal_and_step_statuses():
    done = _step("step-1", "Profile", StepStatus.DONE)
    failed = _step("step-2", "Join", StepStatus.FAILED)
    failed.error = "fan-out"
    skipped = _step("step-3", "Optional", StepStatus.SKIPPED)
    plan = Plan(
        id="plan-1",
        task_id="task-1",
        goal="finish validation",
        source="generated",
        template_id=None,
        steps=[done, failed, skipped],
        autonomy_level=1,
        status=PlanStatus.RUNNING,
    )

    ledger = build_progress_ledger(plan, {"step-1": {"rows": 10}})

    assert "目标: finish validation" in ledger
    assert "[done] Profile" in ledger
    assert '"rows": 10' in ledger
    assert "[failed] Join: fan-out" in ledger
    assert "[skipped] Optional" in ledger


def test_fit_to_budget_keeps_high_priority_items_first():
    kept = fit_to_budget(
        [
            {"priority": 1, "payload": "low" * 20},
            {"priority": 3, "payload": "high"},
            {"priority": 2, "payload": "mid"},
        ],
        max_chars=80,
    )

    assert [item["priority"] for item in kept] == [3, 2]


def _step(step_id: str, title: str, status: StepStatus) -> PlanStep:
    return PlanStep(
        id=step_id,
        plan_id="plan-1",
        index=int(step_id.rsplit("-", 1)[1]),
        title=title,
        tool_ref=ToolRef("_sample", "echo"),
        inputs={},
        depends_on=[],
        post_checks=[],
        status=status,
    )
