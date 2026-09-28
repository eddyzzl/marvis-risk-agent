"""Bridge immutable execution evidence to the pure business verdict.

Only built-in adoption outputs identify the evaluated object. Candidate lists,
LLM summaries and request-supplied evidence never participate in this selection.
Business facts are reconstructed only from the adopted producer's exact sample,
training and independently declared context references.
"""

from marvis.business_acceptance import (
    OBJECTIVE_VERSION,
    BusinessEvidence,
    BusinessObjective,
    evaluate_business_acceptance,
    require_objective_covers_legacy,
)
from marvis.data.errors import DataLayerError
from marvis.orchestrator.contracts import StepStatus
from marvis.orchestrator.evidence import payload_hash


def review_business_acceptance(plan, repository):
    configured = [
        item
        for item in plan.success_criteria
        if item.get("schema_version") == OBJECTIVE_VERSION
    ]
    if not configured:
        result = evaluate_business_acceptance(None, None)
        if plan.success_criteria:
            result.update(
                status="insufficient_evidence",
                reasons=[
                    "旧成功标准缺少业务口径、时期与采用对象绑定；不能据候选最大值认定达标。"
                ],
            )
        return result
    try:
        if len(configured) != 1:
            raise ValueError("a plan requires one business objective")
        objective = BusinessObjective.from_dict(configured[0])
        require_objective_covers_legacy(objective, plan.success_criteria)
    except ValueError as exc:
        result = evaluate_business_acceptance(None, None)
        result.update(
            status="insufficient_evidence", reasons=[f"业务验收合同无效：{exc}"]
        )
        return result
    evidence = adopted_evidence(plan, repository, objective.target_kind)
    return evaluate_business_acceptance(objective, evidence)


def adopted_evidence(plan, repository, target_kind):
    if repository is None:
        return None
    producer = {
        "model": "modeling.select_experiment",
        "strategy": "strategy.adopt_strategy",
    }.get(target_kind)
    steps = [step for step in plan.steps if step.tool_ref.label() == producer]
    # An ambiguous adoption sequence needs an explicit binding, not a guessed
    # winner or an older successful candidate after a later failed adoption.
    if len(steps) != 1 or steps[0].status != StepStatus.DONE or not steps[0].output_ref:
        return None
    step = steps[0]
    try:
        binding = repository.load_step_presentation_binding(step.id, step.output_ref)
        output = binding["output"]
        if target_kind == "model":
            target_id = output.get("selected_experiment_id")
            target_version = output.get("artifact_id")
            metrics = output.get("metrics")
        else:
            target_id, target_version = output.get("strategy_id"), output.get("version")
            backtests = []
            for candidate in plan.steps:
                if (
                    candidate.tool_ref.label() != "strategy.backtest_strategy"
                    or candidate.status != StepStatus.DONE
                    or not candidate.output_ref
                ):
                    continue
                measured = repository.load_step_presentation_binding(
                    candidate.id, candidate.output_ref
                )["output"]
                if (
                    measured.get("backtest_id") == output.get("backtest_id")
                    and measured.get("strategy_id") == target_id
                ):
                    backtests.append(measured)
            if output.get("business_measurement") is not None:
                # Authenticated adoption below reloads its exact persisted backtest;
                # a prior-plan backtest is valid without inventing a local candidate.
                metrics = {}
            elif len(backtests) == 1:
                metrics = backtests[0].get("metrics") or backtests[0].get("risk") or {}
            else:
                return None
        if not target_id or target_version is None or not isinstance(metrics, dict):
            return None
        from marvis.business_context import authenticated_business_fields

        measured = authenticated_business_fields(
            repository, plan.task_id, output, target_kind
        )
        # Deliberately do not consume arbitrary measurement_context keys added
        # to an output by a plugin. Receipt authentication proves provenance,
        # not that a producer has measured every business claim.
        return BusinessEvidence(
            target_kind=target_kind,
            target_id=str(target_id),
            target_version=str(target_version),
            source_ref=step.output_ref,
            source_hash=payload_hash(output),
            **{"metrics": dict(metrics), **measured},
        )
    except (KeyError, TypeError, ValueError, OSError, DataLayerError):
        return None


def stored_business_review(repository, plan_id):
    """One stored projection used by HTTP, Agent and document renderers."""
    ref = repository.latest_plan_summary_ref(plan_id)
    if not ref:
        return {}
    summary = repository.load_plan_summary(ref)
    result = summary.get("business_acceptance")
    if not isinstance(result, dict):
        return {}  # Historical summaries remain readable without inventing a verdict.
    if result.get("execution_binding") != business_review_binding(
        repository.load_plan(plan_id)
    ):
        return {}  # A revised/reset plan must not display a previous run's verdict.
    return {
        "execution_completed": summary.get("execution_completed"),
        "business_acceptance": result,
        "explanation_status": summary.get("explanation_status", "unavailable"),
        "summary_ref": ref,
    }


def business_review_binding(plan):
    return payload_hash(
        {
            "plan_id": plan.id,
            "task_id": plan.task_id,
            "revision": plan.replan_count,
            "success_criteria": plan.success_criteria,
            "steps": [
                {
                    "id": step.id,
                    "tool": step.tool_ref.label(),
                    "inputs": step.inputs,
                    "status": step.status.value,
                    "output_ref": step.output_ref,
                }
                for step in plan.steps
            ],
        }
    )
