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


class SequencedLLM:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) <= len(self.responses):
            return self.responses[len(self.calls) - 1]
        return self.responses[-1]


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


def _plan(*steps: PlanStep, success_criteria=None) -> Plan:
    return Plan(
        id="plan-1",
        task_id="task-1",
        goal="finish",
        source="template",
        template_id="test",
        steps=list(steps),
        autonomy_level=1,
        success_criteria=list(success_criteria or []),
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


def test_reviewer_deterministic_check_supports_list_index_paths():
    step = _step([PostCheck("range", {"field": "metrics.0.ks", "min": 0.0, "max": 1.0})])

    verdict = Reviewer(lambda: FakeLLM("{}")).deterministic_check(
        step,
        {"metrics": [{"ks": 1.7}]},
    )

    assert verdict.passed is False
    assert any("metrics.0.ks=1.7 > 1.0" in reason for reason in verdict.reasons)


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


def test_reviewer_llm_critique_marks_unparseable_reply_as_soft_warning():
    verdict = Reviewer(lambda: FakeLLM("not json")).llm_critique(
        _step([]),
        {"echoed": "hi"},
        "finish",
    )

    assert verdict.reviewer == "llm_critic"
    assert verdict.passed is False
    assert verdict.reasons == ["llm critique returned non-json"]


def test_reviewer_llm_critique_retries_after_unparseable_reply():
    llm = SequencedLLM(["not json", json.dumps({"passed": True, "reasons": []})])

    verdict = Reviewer(lambda: llm).llm_critique(_step([]), {"echoed": "hi"}, "finish")

    assert verdict.passed is True
    assert verdict.reasons == []
    assert len(llm.calls) == 2
    assert "Previous reply was not parseable JSON" in llm.calls[1]["user_prompt"]


def test_reviewer_final_review_goal_doubt_blocks_goal_met():
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
    assert review.goal_met is False
    assert review.goal_doubt is True
    assert review.open_items == ["review business wording"]


def test_reviewer_final_review_llm_goal_met_false_blocks_goal_met():
    done = _step([])
    done.status = StepStatus.DONE
    llm = FakeLLM(json.dumps({
        "summary": "Outputs exist but do not satisfy the business goal.",
        "open_items": ["choose final production model"],
        "goal_doubt": False,
        "goal_met": False,
    }))

    review = Reviewer(lambda: llm).final_review(
        _plan(done),
        {"step-1": {"ok": True}},
        "finish",
    )

    assert review.goal_met is False
    assert review.goal_doubt is False
    assert review.llm_goal_met is False
    assert review.open_items == ["choose final production model"]


def test_reviewer_final_review_llm_goal_met_false_adds_default_open_item():
    done = _step([])
    done.status = StepStatus.DONE
    llm = FakeLLM(json.dumps({
        "summary": "Not done.",
        "open_items": [],
        "goal_doubt": False,
        "goal_met": False,
    }))

    review = Reviewer(lambda: llm).final_review(_plan(done), {"step-1": {"ok": True}}, "finish")

    assert review.goal_met is False
    assert review.open_items == ["LLM final review marked goal_met=false"]


def test_reviewer_final_review_retries_summary_after_unparseable_reply():
    done = _step([])
    done.status = StepStatus.DONE
    llm = SequencedLLM([
        "not json",
        json.dumps({"summary": "Retried summary.", "open_items": [], "goal_doubt": False}),
    ])

    review = Reviewer(lambda: llm).final_review(_plan(done), {"step-1": {"ok": True}}, "finish")

    assert review.summary == "Retried summary."
    assert review.goal_met is True
    assert len(llm.calls) == 2


def test_reviewer_final_review_marks_incomplete_steps_as_open_items():
    pending = _step([])
    pending.status = StepStatus.PENDING

    review = Reviewer(lambda: FakeLLM("{}")).final_review(_plan(pending), {}, "finish")

    assert review.goal_met is False
    assert "Metrics" in review.open_items


def test_reviewer_final_review_passes_when_success_criteria_are_met():
    done = _step([])
    done.status = StepStatus.DONE
    plan = _plan(
        done,
        success_criteria=[
            {
                "metric": "oot_ks",
                "min": 0.3331,
                "aggregate": "max",
                "label": "OOT KS",
                "target_type": "binary",
            }
        ],
    )

    review = Reviewer(lambda: FakeLLM("{}")).final_review(
        plan,
        {
            "step-1": {
                "target_type": "binary",
                "experiments": [{"metrics": {"oot_ks": 0.41}}],
            }
        },
        "finish",
    )

    assert review.goal_met is True
    assert review.open_items == []


def test_reviewer_final_review_fails_when_success_criteria_are_not_met():
    done = _step([])
    done.status = StepStatus.DONE
    plan = _plan(
        done,
        success_criteria=[
            {
                "metric": "oot_ks",
                "min": 0.3331,
                "aggregate": "max",
                "label": "OOT KS",
                "target_type": "binary",
            }
        ],
    )

    review = Reviewer(lambda: FakeLLM("{}")).final_review(
        plan,
        {"step-1": {"target_type": "binary", "metrics": {"oot_ks": 0.2}}},
        "finish",
    )

    assert review.goal_met is False
    assert "OOT KS=0.2 < 0.3331" in review.open_items
    assert "成功标准未达成" in review.summary


def test_reviewer_final_review_reports_invalid_success_thresholds():
    done = _step([])
    done.status = StepStatus.DONE
    plan = _plan(
        done,
        success_criteria=[
            {
                "metric": "oot_ks",
                "min": "not-a-number",
                "label": "OOT KS",
            }
        ],
    )

    review = Reviewer(lambda: FakeLLM("{}")).final_review(
        plan,
        {"step-1": {"metrics": {"oot_ks": 0.41}}},
        "finish",
    )

    assert review.goal_met is False
    assert "OOT KS invalid min threshold: 'not-a-number'" in review.open_items


def test_reviewer_final_review_skips_binary_success_criteria_for_continuous_targets():
    done = _step([])
    done.status = StepStatus.DONE
    plan = _plan(
        done,
        success_criteria=[
            {
                "metric": "oot_ks",
                "min": 0.3331,
                "label": "OOT KS",
                "target_type": "binary",
            }
        ],
    )

    review = Reviewer(lambda: FakeLLM("{}")).final_review(
        plan,
        {"step-1": {"target_type": "continuous", "metrics": {"oot_rmse": 1.4}}},
        "finish",
    )

    assert review.goal_met is True
    assert review.open_items == []
