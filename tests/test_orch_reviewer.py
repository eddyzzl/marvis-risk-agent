import json

from marvis.orchestrator.contracts import Plan, PlanStep, PostCheck, StepStatus
from marvis.orchestrator.reviewer import FinalReview, Reviewer
from marvis.plugins.manifest import ToolRef


class FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _step(post_checks: list[PostCheck]) -> PlanStep:
    return PlanStep(
        id="step-1",
        plan_id="plan-1",
        index=0,
        title="Metrics",
        tool_ref=ToolRef("_sample", "echo"),
        inputs={},
        depends_on=[],
        post_checks=post_checks,
    )


def _plan(*steps: PlanStep) -> Plan:
    return Plan(
        id="plan-1",
        task_id="task-1",
        goal="finish",
        source="template",
        template_id="test",
        steps=list(steps),
        autonomy_level=1,
    )


def test_reviewer_deterministic_check_passes_known_post_checks():
    step = _step([
        PostCheck("schema", {"schema": {"type": "object", "required": ["rows"]}}),
        PostCheck("range", {"field": "ks", "min": 0.0, "max": 1.0}),
        PostCheck("rowcount", {"field": "rows", "min": 1}),
        PostCheck("invariant", {"rule": "joined_rows<=anchor_rows"}),
        PostCheck("nonempty", {"field": "artifacts"}),
        PostCheck("match_rate", {"field": "match_rate", "min": 0.8}),
        PostCheck("one_of", {"field": "status", "values": ["ok", "review"]}),
    ])
    output = {
        "rows": 10,
        "ks": 0.42,
        "joined_rows": 9,
        "anchor_rows": 10,
        "artifacts": ["report.docx"],
        "match_rate": 0.9,
        "status": "ok",
    }

    verdict = Reviewer(lambda: FakeLLM("{}")).deterministic_check(step, output)

    assert verdict.reviewer == "deterministic"
    assert verdict.passed is True
    assert verdict.reasons == []


def test_reviewer_deterministic_check_blocks_invalid_metrics_and_join_invariants():
    step = _step([
        PostCheck("range", {"field": "ks", "min": 0.0, "max": 1.0}),
        PostCheck("invariant", {"rule": "joined_rows<=anchor_rows"}),
    ])

    verdict = Reviewer(lambda: FakeLLM("{}")).deterministic_check(
        step,
        {"ks": 1.2, "joined_rows": 11, "anchor_rows": 10},
    )

    assert verdict.passed is False
    assert any("ks=1.2 > 1.0" in reason for reason in verdict.reasons)
    assert any("joined_rows<=anchor_rows" in reason for reason in verdict.reasons)


def test_reviewer_deterministic_range_allows_declared_null_metric():
    step = _step([PostCheck("range", {"field": "psi", "min": 0.0, "allow_null": True})])

    verdict = Reviewer(lambda: FakeLLM("{}")).deterministic_check(step, {"psi": None})

    assert verdict.passed is True


def test_reviewer_deterministic_one_of_blocks_unexpected_status():
    step = _step([PostCheck("one_of", {"field": "status", "values": ["ok"]})])

    verdict = Reviewer(lambda: FakeLLM("{}")).deterministic_check(step, {"status": "failed"})

    assert verdict.passed is False
    assert "status=failed" in verdict.reasons[0]


def test_reviewer_llm_critique_returns_soft_verdict_only():
    llm = FakeLLM(json.dumps({"passed": False, "reasons": ["needs human review"]}))

    verdict = Reviewer(lambda: llm).llm_critique(_step([]), {"echoed": "hi"}, "finish")

    assert verdict.reviewer == "llm_critic"
    assert verdict.passed is False
    assert verdict.reasons == ["needs human review"]
    assert llm.calls


def test_reviewer_final_review_keeps_goal_met_deterministic_and_tracks_doubt():
    done = _step([])
    done.status = StepStatus.DONE
    skipped = _step([])
    skipped.id = "step-2"
    skipped.status = StepStatus.SKIPPED
    llm = FakeLLM(json.dumps({
        "summary": "Structurally complete.",
        "open_items": ["review business wording"],
        "goal_doubt": True,
    }))

    review = Reviewer(lambda: llm).final_review(
        _plan(done, skipped),
        {"step-1": {"ok": True}},
        "finish",
    )

    assert isinstance(review, FinalReview)
    assert review.goal_met is True
    assert review.goal_doubt is True
    assert review.open_items == ["review business wording"]


def test_reviewer_final_review_marks_incomplete_steps_as_open_items():
    pending = _step([])
    pending.status = StepStatus.PENDING

    review = Reviewer(lambda: FakeLLM("{}")).final_review(_plan(pending), {}, "finish")

    assert review.goal_met is False
    assert "Metrics" in review.open_items
