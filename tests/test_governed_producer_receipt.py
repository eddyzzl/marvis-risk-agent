"""Producer receipts bind original governed invocations to atomic adoption output."""

from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from marvis.db import PluginRepository, connect
from marvis.governance.errors import ApprovalBindingError
from marvis.governance.repository import GovernanceRepository
from marvis.governance.service import GovernanceService
from marvis.orchestrator.contracts import Plan, PlanStatus, PlanStep, StepStatus
from marvis.packs.strategy.tools import tool_adopt_strategy
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner
from marvis.repositories.plans import PlanRepository
from marvis.repositories.strategy import StrategyRepository
from marvis.state_machine import ConflictError
from tests.test_strategy_typed_adoption import _backtest, _runtime_fixture


class HostCrash(BaseException):
    pass


def _case(tmp_path, *, run_input_overrides=None, legacy_contract=False):
    settings, _, task, _, dataset, strategy, strategies, tool_context = (
        _runtime_fixture(tmp_path, "approval")
    )
    backtest = _backtest(dataset.id, strategy.id, "approval", tool_context)
    plugins = PluginRepository(settings.db_path)
    registry = PluginRegistry(plugins)
    load_builtin_packs(registry, Path(__file__).parents[1] / "marvis" / "packs")
    tools = ToolRegistry(registry)
    plans = PlanRepository(settings.db_path)
    governance = GovernanceRepository(
        settings.db_path, runtime_generation="producer-runtime"
    )
    service = GovernanceService(
        plan_repo=plans,
        tool_registry=tools,
        strategy_repo=strategies,
        governance_repo=governance,
    )
    inputs = {
        "strategy_id": strategy.id,
        "backtest_id": backtest["backtest_id"],
        "adoption_reason": "Reviewed evidence and approved local adoption",
    }
    ref = ToolRef("strategy", "adopt_strategy")
    manifest, spec = tools.resolve_with_manifest(ref)
    plan = Plan(
        id="producer-plan",
        task_id=task.id,
        goal="adopt strategy",
        source="template",
        template_id="strategy_development",
        autonomy_level=1,
        status=PlanStatus.AWAITING_CONFIRM,
        steps=[
            PlanStep(
                id="producer-step",
                plan_id="producer-plan",
                index=0,
                title="adopt",
                tool_ref=ref,
                inputs=inputs,
                depends_on=[],
                post_checks=[],
                needs_confirmation=True,
                policy=spec.policy,
                status=StepStatus.AWAITING_CONFIRM,
            )
        ],
    )
    plans.create_plan(plan)
    principal = governance.create_local_principal()
    grant = service.authorize_step(
        plan_id=plan.id,
        step_id=plan.steps[0].id,
        principal=principal,
        reason=inputs["adoption_reason"],
        expected_plan_revision=0,
    )
    current = plans.load_plan(plan.id)
    plans.update_step(replace(current.steps[0], status=StepStatus.RUNNING))
    plans.set_plan_status(plan.id, PlanStatus.RUNNING)
    runner = ToolRunner(
        tools,
        plugins,
        python_executable=sys.executable,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
        governance=governance,
        binding_resolver=service,
    )
    contract = runner.prepare_invocation(ref)
    run_id = plans.start_step_run(
        plan_id=plan.id,
        step_id=plan.steps[0].id,
        tool_ref=ref.label(),
        inputs={**inputs, **(run_input_overrides or {})},
        invocation_contract=None if legacy_contract else contract,
    )
    binding = service.resolve_binding(
        task_id=task.id,
        ref=ref,
        inputs=inputs,
        execution_context=grant.context,
        manifest=manifest,
        tool=spec,
    )
    return SimpleNamespace(
        settings=settings,
        task=task,
        strategy=strategy,
        strategies=strategies,
        tool_context=tool_context,
        registry=registry,
        plans=plans,
        governance=governance,
        runner=runner,
        grant=grant,
        binding=binding,
        inputs=inputs,
        ref=ref,
        contract=contract,
        run_id=run_id,
    )


def _reserve(case):
    case.plans.mark_step_run_dispatched(case.run_id)
    effect = case.governance.reserve_effect(
        case.grant.context,
        case.binding,
        invocation_id=case.run_id,
        invocation_contract=case.contract,
    )
    case.governance.mark_effect_dispatched(
        effect.id, reservation_id=effect.reservation_id
    )
    case.effect = effect
    case.tool_context = replace(
        case.tool_context,
        effect_execution_id=effect.id,
        runtime_generation=effect.runtime_generation,
    )
    return effect


def _adopt(case):
    _reserve(case)
    return tool_adopt_strategy(case.inputs, case.tool_context)


def test_real_runner_commit_before_host_receipt_has_complete_original_output(
    tmp_path, monkeypatch
):
    case = _case(tmp_path)
    original = case.runner._finalize_effect_result

    def crash_after_worker(*args, **kwargs):
        result = original(*args, **kwargs)
        if result.ok:
            raise HostCrash("host died after worker committed adoption")
        return result

    monkeypatch.setattr(case.runner, "_finalize_effect_result", crash_after_worker)
    with pytest.raises(HostCrash):
        case.runner.invoke(
            case.ref,
            case.inputs,
            task_id=case.task.id,
            execution_context=case.grant.context,
            invocation_id=case.run_id,
            expected_invocation=case.contract,
            on_dispatch=lambda: case.plans.mark_step_run_dispatched(case.run_id),
        )
    restarted = GovernanceRepository(
        case.settings.db_path, runtime_generation="new-runtime"
    )
    restarted.reconcile_startup()
    outcome = restarted.verify_producer_outcome(case.run_id)
    assert outcome["outcome"] == "applied", outcome
    assert outcome["bindings"]["invocation_id"] == case.run_id
    assert outcome["bindings"]["runtime_generation"] == "producer-runtime"
    assert outcome["output"]["strategy_id"] == case.strategy.id
    assert {a["kind"] for a in outcome["output"]["artifacts"]} == {
        "decision_table_csv",
        "monitoring_plan_json",
    }
    assert len(case.strategies.list_strategy_artifacts(case.strategy.id)) == 2
    assert case.plans.list_step_runs("producer-step")[0]["output_ref"] is None
    assert case.plans.list_step_runs("producer-step")[0]["status"] == "running"


def test_complete_output_is_frozen_in_same_transaction_and_readback_is_read_only(
    tmp_path,
):
    case = _case(tmp_path)
    output = _adopt(case)
    before = case.governance.get_effect_execution(case.effect.id)
    with connect(case.settings.db_path) as conn:
        conn.execute("BEGIN")
        forbidden = {
            sqlite3.SQLITE_INSERT,
            sqlite3.SQLITE_UPDATE,
            sqlite3.SQLITE_DELETE,
        }
        conn.set_authorizer(
            lambda operation, *_args: (
                sqlite3.SQLITE_DENY if operation in forbidden else sqlite3.SQLITE_OK
            )
        )
        outcome = case.governance.verify_producer_outcome(case.run_id, conn=conn)
    assert outcome["outcome"] == "applied", outcome
    assert outcome["output"] == output
    assert before == case.governance.get_effect_execution(case.effect.id)
    assert (
        before.detail["producer_receipt"]["receipt_hash"]
        == before.result_hash
        == outcome["receipt_hash"]
    )
    assert before.invocation_id == case.run_id


@pytest.mark.parametrize("after_freeze", [False, True])
def test_freeze_failure_rolls_back_adoption_and_artifacts_then_persisted_fence_blocks_late_worker(
    tmp_path, monkeypatch, after_freeze
):
    case = _case(tmp_path)
    effect = _reserve(case)
    original = StrategyRepository.freeze_adoption_producer_receipt_on_connection

    def fail(self, *args, **kwargs):
        if after_freeze:
            original(self, *args, **kwargs)
        raise RuntimeError("producer commit checkpoint failed")

    monkeypatch.setattr(
        StrategyRepository, "freeze_adoption_producer_receipt_on_connection", fail
    )
    with pytest.raises(RuntimeError, match="producer commit checkpoint"):
        tool_adopt_strategy(case.inputs, case.tool_context)
    assert case.strategies.get_strategy_meta(case.strategy.id)["status"] == "draft"
    assert case.strategies.list_strategy_artifacts(case.strategy.id) == []
    assert (
        list(
            (case.settings.tasks_dir / case.task.id / "strategy").glob(
                "decision_table_*"
            )
        )
        == []
    )
    assert case.governance.get_effect_execution(effect.id).detail == {}
    assert case.governance.verify_producer_outcome(case.run_id)["outcome"] == "unknown"
    restarted = GovernanceRepository(
        case.settings.db_path, runtime_generation="new-runtime"
    )
    restarted.reconcile_startup()
    outcome = restarted.verify_producer_outcome(case.run_id)
    assert outcome["outcome"] == "not_applied_fenced", outcome
    assert "output" not in outcome
    monkeypatch.setattr(
        StrategyRepository, "freeze_adoption_producer_receipt_on_connection", original
    )
    with pytest.raises(ConflictError, match="dispatched"):
        tool_adopt_strategy(case.inputs, case.tool_context)


@pytest.mark.parametrize(
    "tamper", ["unknown_id", "declaration", "inputs", "older_attempt"]
)
def test_reservation_rejects_wrong_original_invocation_before_consuming_approval(
    tmp_path, tamper
):
    case = _case(
        tmp_path,
        run_input_overrides={"strategy_id": "other"} if tamper == "inputs" else None,
    )
    case.plans.mark_step_run_dispatched(case.run_id)
    invocation_id, contract = case.run_id, dict(case.contract)
    if tamper == "unknown_id":
        invocation_id = "invented-run"
    elif tamper == "declaration":
        contract["manifest_declaration_hash"] = "sha256:" + "a" * 64
    elif tamper == "older_attempt":
        case.plans.start_step_run(
            plan_id="producer-plan",
            step_id="producer-step",
            tool_ref=case.ref.label(),
            inputs=case.inputs,
            invocation_contract=case.contract,
        )
    with pytest.raises(ApprovalBindingError):
        case.governance.reserve_effect(
            case.grant.context,
            case.binding,
            invocation_id=invocation_id,
            invocation_contract=contract,
        )
    assert case.governance.get_approval(case.grant.approval.id).state.value == "issued"
    assert case.governance.list_effect_executions(case.grant.approval.id) == []


@pytest.mark.parametrize(
    "field", ["invocation_id", "invocation_contract_hash", "receipt", "result_hash"]
)
def test_database_rejects_rebinding_or_replacing_complete_producer_receipt(
    tmp_path, field
):
    case = _case(tmp_path)
    _adopt(case)
    effect = case.governance.get_effect_execution(case.effect.id)
    assignments = {
        "invocation_id": ("invocation_id = ?", "another-run"),
        "invocation_contract_hash": ("invocation_contract_hash = ?", "0" * 64),
        "receipt": (
            "detail_json = ?",
            json.dumps({"domain_receipt": effect.detail["domain_receipt"]}),
        ),
        "result_hash": ("result_hash = ?", "0" * 64),
    }
    assignment, value = assignments[field]
    with pytest.raises(sqlite3.IntegrityError), connect(case.settings.db_path) as conn:
        conn.execute(
            f"UPDATE effect_executions SET {assignment} WHERE id = ?",
            (value, effect.id),
        )
    assert case.governance.verify_producer_outcome(case.run_id)["outcome"] == "applied"


@pytest.mark.parametrize(
    "tamper",
    ["missing_artifact", "changed_artifact", "changed_strategy", "changed_run_input"],
)
def test_readback_rejects_stale_or_tampered_domain_evidence(tmp_path, tamper):
    case = _case(tmp_path)
    output = _adopt(case)
    artifact = Path(output["artifacts"][0]["path"])
    if tamper == "missing_artifact":
        artifact.unlink()
    elif tamper == "changed_artifact":
        artifact.write_text("forged output")
    elif tamper == "changed_strategy":
        with connect(case.settings.db_path) as conn:
            conn.execute(
                "UPDATE strategies SET description = 'different' WHERE id = ?",
                (case.strategy.id,),
            )
    else:
        with (
            pytest.raises(sqlite3.IntegrityError),
            connect(case.settings.db_path) as conn,
        ):
            conn.execute(
                "UPDATE plan_step_runs SET input_json = '{}' WHERE id = ?",
                (case.run_id,),
            )
        assert (
            case.governance.verify_producer_outcome(case.run_id)["outcome"] == "applied"
        )
        return
    outcome = case.governance.verify_producer_outcome(case.run_id)
    assert outcome["outcome"] == "unknown", outcome
    assert "output" not in outcome


def test_domain_receipt_without_complete_producer_output_remains_unknown(
    tmp_path, monkeypatch
):
    case = _case(tmp_path)
    monkeypatch.setattr(
        StrategyRepository,
        "freeze_adoption_producer_receipt_on_connection",
        lambda *args, **kwargs: None,
    )
    output = _adopt(case)
    assert output["status"] == "adopted"
    assert (
        "domain_receipt" in case.governance.get_effect_execution(case.effect.id).detail
    )
    outcome = case.governance.verify_producer_outcome(case.run_id)
    assert outcome["outcome"] == "unknown"
    assert outcome["reason"] == "complete_producer_receipt_missing"
    assert (
        outcome["invocation_contract_hash"]
        == case.governance.get_effect_execution(case.effect.id).invocation_contract_hash
    )


@pytest.mark.parametrize("forgery", ["binding", "target", "artifact"])
def test_self_consistent_receipt_hash_cannot_authorize_forged_original_facts(
    tmp_path, monkeypatch, forgery
):
    from marvis.governance.contracts import PRODUCER_RECEIPT_SCHEMA_VERSION
    from marvis.governance.repository import (
        _producer_binding_on_connection,
        canonical_payload_hash,
    )

    case = _case(tmp_path)
    # Simulate a predecessor producer that committed only a domain receipt.
    monkeypatch.setattr(
        StrategyRepository,
        "freeze_adoption_producer_receipt_on_connection",
        lambda *args, **kwargs: None,
    )
    output = _adopt(case)
    with connect(case.settings.db_path) as conn:
        effect = conn.execute(
            "SELECT * FROM effect_executions WHERE id = ?", (case.effect.id,)
        ).fetchone()
        approval = conn.execute(
            "SELECT * FROM approval_records WHERE id = ?", (effect["approval_id"],)
        ).fetchone()
        bindings = _producer_binding_on_connection(conn, effect, approval)
        if forgery == "binding":
            bindings["invocation_id"] = "invented-run"
        elif forgery == "target":
            output["strategy_id"] = "other-strategy"
        else:
            output["artifacts"][0]["content_hash"] = "a" * 64
        receipt = {
            "schema_version": PRODUCER_RECEIPT_SCHEMA_VERSION,
            "receipt_id": f"producer:{case.effect.id}",
            "bindings": bindings,
            "output": output,
            "output_hash": canonical_payload_hash(output),
        }
        receipt["receipt_hash"] = canonical_payload_hash(receipt)
        detail = json.loads(effect["detail_json"])
        detail["producer_receipt"] = receipt
        conn.execute(
            "UPDATE effect_executions SET detail_json = ?, result_hash = ? WHERE id = ?",
            (json.dumps(detail), receipt["receipt_hash"], case.effect.id),
        )
    outcome = case.governance.verify_producer_outcome(case.run_id)
    assert outcome["outcome"] == "unknown", outcome
    assert "output" not in outcome


def test_legacy_null_invocation_contract_never_receives_a_producer_proof(tmp_path):
    case = _case(tmp_path, legacy_contract=True)
    effect = case.governance.reserve_effect(case.grant.context, case.binding)
    case.governance.mark_effect_dispatched(
        effect.id, reservation_id=effect.reservation_id
    )
    context = replace(
        case.tool_context,
        effect_execution_id=effect.id,
        runtime_generation=effect.runtime_generation,
    )
    output = tool_adopt_strategy(case.inputs, context)
    assert output["status"] == "adopted"
    stored = case.governance.get_effect_execution(effect.id)
    assert stored.invocation_id is None
    assert stored.invocation_contract_hash is None
    assert "producer_receipt" not in stored.detail
    outcome = case.governance.verify_producer_outcome(case.run_id)
    assert outcome["outcome"] == "unknown"
    assert outcome["invocation_contract_hash"] is None
    assert "output" not in outcome
