from copy import deepcopy
from dataclasses import replace
import json

import pytest

from marvis.business_acceptance import (
    BusinessCriterion,
    evaluate_business_acceptance,
    BusinessEvidence,
)
from marvis.business_context import (
    bind_business_context,
    load_business_context,
    model_business_measurement,
)
from marvis.db_schema import connect
from marvis.orchestrator.evidence import payload_hash
from marvis.packs.modeling.evidence_tools import (
    build_training_evidence_ref,
    require_historical_modeling_training_evidence_artifact_binding_on_connection,
)
from marvis.packs.modeling.select_tools import tool_select_experiment
from tests.test_business_acceptance import objective
from tests.test_modeling_training_evidence_tool import _native_fixture, _run, _binding


def declaration(partition="oot"):
    return {
        "business_line": "consumer_credit",
        "decision_node": "approval",
        "population": "existing_customers",
        "responsibility_source": "owner:independent-dataset-review",
        "partition": partition,
        "declared_label_origin": "observed",
    }


@pytest.fixture
def measured(tmp_path):
    fx = _native_fixture(tmp_path)
    context = bind_business_context(
        {"sample_design_ref": fx["sample_ref"], "declaration": declaration()},
        fx["ctx"],
        fx["runtime"],
    )
    trained = _run(fx)
    ref = build_training_evidence_ref(_binding(fx, trained))
    inputs = {
        "experiment_ids": [trained["experiment_id"]],
        "selected_experiment_id": trained["experiment_id"],
        "refit_on_train_plus_test": False,
        "training_evidence_ref": ref,
        "business_context_ref": context["business_context_ref"],
    }
    return fx, context, trained, ref, inputs


def test_real_model_selection_uses_only_exact_partition_and_authenticated_metrics(
    measured,
):
    fx, context, trained, ref, inputs = measured
    result = tool_select_experiment(inputs, fx["ctx"])
    value = result["business_measurement"]
    assert (
        value["target_id"]
        == result["selected_experiment_id"]
        == trained["experiment_id"]
    )
    assert (
        value["target_version"] == result["artifact_id"] == trained["model_artifact_id"]
    )
    assert set(value["metrics"]) == {"oot_ks", "oot_auc"}
    assert value["denominators"]["oot_ks"] == "risk/oot:labeled"
    assert value["period_start"] == "2026-03-01"
    assert value["period_end"] == "2026-03-06"
    assert value["labels_mature"] is True
    assert value["label_origin"] == "unknown"
    rebound = _binding(fx, trained, historical=True)
    assert rebound.experiment.status == "selected"
    with connect(fx["settings"].db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        require_historical_modeling_training_evidence_artifact_binding_on_connection(
            conn, rebound
        )
    assert (
        model_business_measurement(
            fx["runtime"],
            fx["task"].id,
            context_ref=context["business_context_ref"],
            training_ref=ref,
            experiment_id=result["selected_experiment_id"],
            artifact_id=result["artifact_id"],
        )
        == value
    )
    evidence = BusinessEvidence(
        target_kind="model",
        target_id=value["target_id"],
        target_version=value["target_version"],
        source_ref="step:selection",
        source_hash=payload_hash(result),
        **{
            key: value[key]
            for key in (
                "metrics",
                "business_line",
                "decision_node",
                "population",
                "currency",
                "period_start",
                "period_end",
                "metric_units",
                "denominators",
                "effect_stage",
                "labels_mature",
                "label_origin",
            )
        },
    )
    obj = objective(
        target_id=value["target_id"],
        target_version=value["target_version"],
        period_start="2026-03-01",
        period_end="2026-03-06",
        criteria=(BusinessCriterion("oot_ks", "ratio", "risk/oot:labeled", minimum=0),),
    )
    assert (
        evaluate_business_acceptance(obj, evidence)["status"] == "insufficient_evidence"
    )
    assert (
        evaluate_business_acceptance(
            replace(obj, require_mature_labels=False), evidence
        )["status"]
        == "passed"
    )


def test_selected_model_snapshot_drift_never_uses_cached_business_measurement(measured):
    fx, context, trained, ref, inputs = measured
    tool_select_experiment(inputs, fx["ctx"])
    with connect(fx["settings"].db_path) as conn:
        row = conn.execute(
            "SELECT metrics_json FROM experiments WHERE id=?",
            (trained["experiment_id"],),
        ).fetchone()
        metrics = json.loads(row["metrics_json"])
        metrics["oot_ks"] = 0.12345678
        conn.execute(
            "UPDATE experiments SET metrics_json=? WHERE id=?",
            (json.dumps(metrics), trained["experiment_id"]),
        )
    with pytest.raises(ValueError, match="metrics_snapshot"):
        model_business_measurement(
            fx["runtime"],
            fx["task"].id,
            context_ref=context["business_context_ref"],
            training_ref=ref,
            experiment_id=trained["experiment_id"],
            artifact_id=trained["model_artifact_id"],
        )


def test_independent_declaration_rejects_metrics_period_and_maturity(tmp_path):
    fx = _native_fixture(tmp_path)
    for field, value in [
        ("labels_mature", True),
        ("period_start", "2000-01-01"),
        ("metrics", {"oot_ks": 1}),
        ("label_origin", "observed"),
    ]:
        claimed = declaration()
        claimed[field] = value
        with pytest.raises(ValueError):
            bind_business_context(
                {"sample_design_ref": fx["sample_ref"], "declaration": claimed},
                fx["ctx"],
                fx["runtime"],
            )


def test_business_context_cannot_cross_tasks_or_change_sample_bytes(measured):
    fx, context, trained, ref, inputs = measured
    with pytest.raises(ValueError, match="task-owned"):
        load_business_context(
            fx["runtime"], "another-task", context["business_context_ref"]
        )
    wrong = deepcopy(inputs)
    wrong["training_evidence_ref"]["expected_model_artifact_id"] = "another-model"
    with pytest.raises(ValueError):
        tool_select_experiment(wrong, fx["ctx"])
    path = fx["runtime"].registry.resolve_path(fx["dataset"].id)
    path.write_bytes(path.read_bytes() + b"drift")
    with pytest.raises(ValueError):
        tool_select_experiment(inputs, fx["ctx"])


@pytest.mark.parametrize("drift_kind", ["metrics", "missing_dataset"])
def test_registered_tool_executor_human_context_and_actual_model_verdict(
    measured, drift_kind
):
    import sys
    from pathlib import Path
    from marvis.db import PluginRepository
    from marvis.governance.repository import GovernanceRepository
    from marvis.governance.service import GovernanceService
    from marvis.orchestrator.contracts import Plan, PlanStatus, PlanStep
    from marvis.orchestrator.reviewer import Reviewer
    from marvis.orchestrator.business_acceptance import stored_business_review
    from marvis.plugins.loader import load_builtin_packs
    from marvis.plugins.manifest import ToolRef
    from marvis.plugins.registry import PluginRegistry, ToolRegistry
    from marvis.plugins.runner import ToolRunner
    from marvis.repositories.plans import PlanRepository
    from marvis.repositories.strategy import StrategyRepository
    from tests.test_orch_executor import _executor, FakeLLM

    fx, context, trained, ref, inputs = measured
    settings = fx["settings"]
    plugins = PluginRepository(settings.db_path)
    registry = PluginRegistry(plugins)
    load_builtin_packs(registry, Path(__file__).parents[1] / "marvis" / "packs")
    tools = ToolRegistry(registry)
    plans = PlanRepository(settings.db_path)
    governance = GovernanceRepository(
        settings.db_path, runtime_generation="business-producer-test"
    )
    service = GovernanceService(
        plan_repo=plans,
        tool_registry=tools,
        strategy_repo=StrategyRepository(settings.db_path),
        governance_repo=governance,
    )
    runner = ToolRunner(
        tools,
        plugins,
        python_executable=sys.executable,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
        governance=governance,
        binding_resolver=service,
    )
    bind_ref = ToolRef("strategy", "bind_business_context")
    policy = tools.resolve(bind_ref).policy
    assert policy.human_decision_gate == "required"
    plan = Plan(
        id="business-plan",
        task_id=fx["task"].id,
        goal="synthetic software acceptance",
        source="generated",
        template_id=None,
        autonomy_level=1,
        status=PlanStatus.CONFIRMED,
        steps=[
            PlanStep(
                id="context",
                plan_id="business-plan",
                index=0,
                title="Confirm independent business context",
                tool_ref=bind_ref,
                inputs={
                    "sample_design_ref": fx["sample_ref"],
                    "declaration": declaration(),
                },
                policy=policy,
                needs_confirmation=True,
                depends_on=[],
                post_checks=[],
            ),
            PlanStep(
                id="select",
                plan_id="business-plan",
                index=1,
                title="Select evidence-bound model",
                tool_ref=ToolRef("modeling", "select_experiment"),
                depends_on=["context"],
                post_checks=[],
                inputs={
                    **inputs,
                    "business_context_ref": "$ref:context.output.business_context_ref",
                },
            ),
        ],
        success_criteria=[
            objective(
                target_id=trained["experiment_id"],
                target_version=trained["model_artifact_id"],
                period_start="2026-03-01",
                period_end="2026-03-06",
                require_mature_labels=False,
                criteria=(
                    BusinessCriterion("oot_ks", "ratio", "risk/oot:labeled", minimum=0),
                ),
            ).to_dict()
        ],
    )
    plans.create_plan(plan)
    reviewer = Reviewer(lambda: FakeLLM(), plan_repository=plans)
    executor = _executor(plans, runner, reviewer=reviewer, authorizer=service)
    paused = executor.run(plan.id)
    assert paused.status == PlanStatus.AWAITING_CONFIRM
    assert plans.load_plan(plan.id).steps[0].output_ref is None
    service.authorize_step(
        plan_id=plan.id,
        step_id="context",
        principal=governance.create_local_principal(),
        reason="Reviewed independent declaration against synthetic sample",
        expected_plan_revision=0,
    )
    result = executor.run(plan.id)
    assert result.status == PlanStatus.DONE, result
    review = stored_business_review(plans, plan.id)
    assert review["business_acceptance"]["status"] == "passed", review
    assert review["business_acceptance"]["evidence"]["label_origin"] == "unknown"
    assert review["business_acceptance"]["target"]["id"] == trained["experiment_id"]
    assert all(step.output_ref for step in plans.load_plan(plan.id).steps)
    # Cached output cannot hide changed metrics or unavailable source bytes.
    if drift_kind == "missing_dataset":
        fx["runtime"].registry.resolve_path(fx["dataset"].id).unlink()
    else:
        with connect(settings.db_path) as conn:
            raw = conn.execute(
                "SELECT metrics_json FROM experiments WHERE id=?",
                (trained["experiment_id"],),
            ).fetchone()[0]
            metrics = json.loads(raw)
            metrics["oot_ks"] = 0.12345678
            conn.execute(
                "UPDATE experiments SET metrics_json=? WHERE id=?",
                (json.dumps(metrics), trained["experiment_id"]),
            )
    from marvis.orchestrator.business_acceptance import review_business_acceptance

    assert (
        review_business_acceptance(plans.load_plan(plan.id), plans)["status"]
        == "insufficient_evidence"
    )


def test_strategy_adoption_binds_exact_native_development_context(tmp_path):
    from tests.test_strategy_sample_design_execution import _parallel_native_setup
    from marvis.packs.strategy import tools
    from marvis.business_context import authenticated_business_fields
    from marvis.repositories.plans import PlanRepository

    fx = _parallel_native_setup(tmp_path)
    ref = {
        "membership_artifact_id": fx["membership"]["id"],
        "expected_membership_artifact_content_hash": fx["membership"]["content_hash"],
        "bundle_artifact_id": fx["bundle"]["id"],
        "expected_bundle_artifact_content_hash": fx["bundle"]["content_hash"],
        "expected_bundle_id": fx["output"]["bundle_id"],
        "expected_sample_design_id": fx["output"]["sample_design_id"],
        "expected_sample_design_content_hash": fx["output"][
            "sample_design_content_hash"
        ],
    }
    bound = bind_business_context(
        {"sample_design_ref": ref, "declaration": declaration("development")},
        fx["ctx"],
        fx["runtime"],
    )
    strategy = tools.tool_build_strategy(
        {
            "strategy_type": "approval",
            "rules": [{"condition": "feature < 25", "decision": "reject"}],
            "score_col": "feature",
            "default_decision": "approve",
        },
        fx["ctx"],
    )
    backtest = tools.tool_backtest_strategy(
        {
            "dataset_id": fx["dataset"].id,
            "strategy_id": strategy["strategy_id"],
            "target_col": "bad",
            "sample_design_ref": fx["sample_ref"],
            "drop_nan_labels": True,
        },
        fx["ctx"],
    )
    output = tools.tool_adopt_strategy(
        {
            "strategy_id": strategy["strategy_id"],
            "backtest_id": backtest["backtest_id"],
            "adoption_reason": "Reviewed native development subset",
            "business_context_ref": bound["business_context_ref"],
        },
        fx["ctx"],
    )
    measured = output["business_measurement"]
    assert measured["effect_stage"] == "backtested"
    assert measured["target_id"] == strategy["strategy_id"]
    assert (
        measured["metrics"]["approved_bad_rate"]
        == output["adoption_evidence"]["metrics"]["approve_bad_rate"]
    )
    assert (
        authenticated_business_fields(
            PlanRepository(fx["settings"].db_path), fx["task"].id, output, "strategy"
        )["metrics"]
        == measured["metrics"]
    )
