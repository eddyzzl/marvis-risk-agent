import json

from marvis.llm_settings import LLMSettingsError
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


def test_reviewer_allows_legacy_screen_plan_to_render_empty_recommendations():
    step = PlanStep(
        id="screen-1",
        plan_id="plan-1",
        index=0,
        title="特征筛选",
        tool_ref=ToolRef("modeling", "screen_features"),
        inputs={},
        depends_on=[],
        post_checks=[PostCheck("nonempty", {"field": "selected"})],
    )

    verdict = Reviewer(lambda: FakeLLM("{}")).deterministic_check(
        step,
        {"selected": [], "ranked": [], "unusable": [["x1", "全为空"]]},
    )

    assert verdict.passed is True
    assert verdict.reasons == []
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


def test_reviewer_llm_critique_prompt_carries_real_metric_values():
    # AGT-3: the critic previously saw only {"type": "object", "keys": [...]} for
    # a step's metrics dict — no actual KS/AUC number ever reached the prompt.
    # After the metric-aware _summarize_output fix, the JSON sent to the LLM must
    # contain the real train/test/oot KS value, not just the field name.
    llm = FakeLLM(json.dumps({"passed": True, "reasons": []}))
    output = {
        "target_type": "binary",
        "metrics": {"train_ks": 0.52, "oot_ks": 0.41, "oot_auc": 0.77},
    }

    Reviewer(lambda: llm).llm_critique(_step([]), output, "finish")

    assert llm.calls
    prompt = llm.calls[0]["user_prompt"]
    assert "0.41" in prompt
    assert "0.52" in prompt
    assert "0.77" in prompt


def test_final_review_llm_opinion_is_explanation_only():
    done = _step([])
    done.status = StepStatus.DONE
    results = []
    for reply in (
        {"summary": "done", "goal_met": True},
        {"summary": "failed", "goal_met": False, "goal_doubt": True, "open_items": ["LLM concern"]},
        {},
    ):
        review = Reviewer(lambda: FakeLLM(json.dumps(reply))).final_review(_plan(done), {done.id: {"ok": True}}, "finish")
        assert review.execution_completed is True
        assert review.goal_met is False
        assert review.goal_doubt is False
        assert "LLM concern" not in review.open_items
        results.append(review.business_acceptance)
    assert results[0] == results[1] == results[2]
    assert results[0]["status"] == "not_configured"


def test_final_review_unavailable_llm_does_not_block_execution():
    done = _step([])
    done.status = StepStatus.DONE
    def unavailable():
        raise LLMSettingsError("no model")
    review = Reviewer(unavailable).final_review(_plan(done), {}, "finish")
    assert review.execution_completed is True
    assert review.explanation_status == "unavailable"
    assert review.business_acceptance["status"] == "not_configured"


def test_final_review_retries_narrative_without_changing_business_status():
    done = _step([])
    done.status = StepStatus.DONE
    llm = SequencedLLM(["not json", json.dumps({"summary": "Retried summary.", "goal_met": True})])
    review = Reviewer(lambda: llm).final_review(_plan(done), {}, "finish")
    assert review.summary == "Retried summary."
    assert review.execution_completed is True
    assert review.business_acceptance["status"] == "not_configured"
    assert len(llm.calls) == 2


def test_final_review_incomplete_step_remains_execution_failure():
    review = Reviewer(lambda: FakeLLM("{}")).final_review(_plan(_step([])), {}, "finish")
    assert review.execution_completed is False
    assert "Metrics" in review.open_items


def test_legacy_criteria_cannot_use_candidate_max_or_implicit_not_applicable():
    done = _step([])
    done.status = StepStatus.DONE
    for target, metric in (("binary", .9), ("binary", .2), ("continuous", None)):
        plan = _plan(done, success_criteria=[{"metric": "oot_ks", "min": .3331, "target_type": "binary", "aggregate": "max"}])
        review = Reviewer(lambda: FakeLLM("{}")).final_review(plan, {done.id: {"target_type": target, "experiments": [{"metrics": {"oot_ks": metric}}]}}, "finish")
        assert review.execution_completed is True
        assert review.goal_met is False
        assert review.business_acceptance["status"] == "insufficient_evidence"


def test_old_final_review_deserialization_keeps_compatibility_marker():
    review = FinalReview(**{"goal_met": True, "summary": "old", "open_items": []})
    assert review.execution_completed is None
    assert review.business_acceptance is None


def test_reviewer_final_review_prompt_carries_real_metric_values():
    done = _step([])
    done.status = StepStatus.DONE
    llm = FakeLLM(json.dumps({"summary": "ok", "open_items": [], "goal_doubt": False, "goal_met": True}))

    Reviewer(lambda: llm).final_review(
        _plan(done),
        {"step-1": {"target_type": "binary", "metrics": {"oot_ks": 0.4123, "oot_auc": 0.71}}},
        "finish",
    )

    assert llm.calls
    prompt = llm.calls[0]["user_prompt"]
    assert "0.4123" in prompt
    assert "0.71" in prompt


def test_reviewer_llm_critique_skips_cleanly_when_no_llm_configured():
    # AGT-6: manual mode (no LLM configured) previously rendered a deterministic
    # "failed" llm_critic verdict with a raw exception message on every step. It
    # must instead skip cleanly: passed=True, status="skipped", so it doesn't
    # drown real deterministic failures in critique noise.
    def factory():
        raise LLMSettingsError("请先在设置中配置至少一个启用的大模型")

    verdict = Reviewer(factory).llm_critique(_step([]), {"echoed": "hi"}, "finish")

    assert verdict.reviewer == "llm_critic"
    assert verdict.passed is True
    assert verdict.status == "skipped"
    assert verdict.reasons == ["skipped: no LLM configured"]


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
