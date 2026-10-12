"""One explicit governed response to authenticated scheduled red evidence."""

from datetime import UTC, datetime
import json
import uuid

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator
import math
from typing import Literal

from marvis.db_schema import connect
from marvis.operations.monitoring import read_monitoring_evidence
from marvis.operations.schedule_proposals import canonical, sha
from marvis.orchestrator.contracts import Plan, PlanStep, PlanStatus, StepStatus
from marvis.plugins.manifest import ToolRef
from marvis.repositories.strategy_monitoring import StrategyMonitoringRepository


class DispositionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    period_key: str = Field(min_length=1, max_length=500)
    expected_outcome_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    idempotency_key: str = Field(
        min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$"
    )
    expected_plan_id: str = Field(min_length=1, max_length=200)
    expected_plan_revision: StrictInt = Field(ge=1)
    expected_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    disposition: Literal["observe", "adjust_threshold", "new_version"]
    reason: str = Field(min_length=1, max_length=4000)
    threshold_patch: dict[str, dict[Literal["warn", "fail"], float]] | None = None

    @model_validator(mode="after")
    def patch_scope(self):
        if (self.disposition == "adjust_threshold") != (
            self.threshold_patch is not None
        ):
            raise ValueError("only threshold adjustment requires an explicit patch")
        if self.threshold_patch is not None:
            if not self.threshold_patch or len(self.threshold_patch) > 100:
                raise ValueError("bounded nonempty threshold patch required")
            for key, changes in self.threshold_patch.items():
                if (
                    not key.strip()
                    or len(key) > 200
                    or not changes
                    or any(not math.isfinite(value) for value in changes.values())
                ):
                    raise ValueError("finite nonempty named threshold changes required")
        return self


class DispositionConsumed(RuntimeError):
    pass


def red_context(runtime, schedule_id, payload):
    outcome = runtime.store.get_outcome(schedule_id, payload.period_key)
    if outcome is None:
        raise KeyError(payload.period_key)
    if (
        outcome.outcome_hash != payload.expected_outcome_hash
        or outcome.outcome.level != "red"
    ):
        raise ValueError("exact red scheduled outcome required")
    evidence = read_monitoring_evidence(runtime.settings.workspace, outcome.outcome)
    schedule = runtime.store.get_schedule(
        schedule_id, revision=evidence["schedule_revision"]
    )
    binding = schedule.contract.monitoring_binding
    if (
        binding is None
        or evidence["monitoring_ref"] != "strategy.run_strategy_monitoring"
        or evidence["schedule_id"] != schedule_id
        or evidence["run_id"] != outcome.run_id
        or evidence["period_key"] != payload.period_key
        or evidence["schedule_hash"] != schedule.contract_hash
    ):
        raise ValueError("scheduled strategy evidence binding differs")
    output = evidence["tool_output"]
    native = StrategyMonitoringRepository(runtime.settings.db_path).get_run(
        output["monitoring_run_id"]
    )
    if (
        native is None
        or native.strategy_id != binding.target_id
        or native.overall_level != "red"
        or output["monitoring_plan_id"] != payload.expected_plan_id
        or output["monitoring_plan_revision"] != payload.expected_plan_revision
        or output["monitoring_plan_hash"] != payload.expected_plan_hash
        or native.monitoring_plan_id != payload.expected_plan_id
        or native.dataset_id != evidence["window_dataset_id"]
        or native.dataset_content_hash != evidence["window_content_hash"]
        or output.get("monitoring_evidence", {}).get("result_hash")
        != native.result_hash
        or output.get("monitoring_evidence", {}).get("dataset_content_hash")
        != native.dataset_content_hash
    ):
        raise ValueError("native red monitoring receipt or reviewed plan differs")
    runtime.builtin_executor.registry.authenticate_dataset_binding(
        native.dataset_id,
        expected_task_id=binding.task_id,
        expected_content_hash=native.dataset_content_hash,
    )
    return outcome, binding, output


def apply_disposition(
    runtime,
    schedule_id,
    payload,
    *,
    principal,
    governance_service,
    plan_repo,
    plan_executor,
):
    outcome, binding, output = red_context(runtime, schedule_id, payload)
    from marvis.institution.policy import authorize

    authorize(
        runtime.settings.db_path,
        "task",
        binding.task_id,
        "approve",
        actor_id=principal.id,
    )
    inputs = {
        key: getattr(payload, key)
        for key in (
            "expected_plan_id",
            "expected_plan_revision",
            "expected_plan_hash",
            "disposition",
            "reason",
        )
    }
    inputs.update(
        strategy_id=binding.target_id, monitoring_run_id=output["monitoring_run_id"]
    )
    if payload.threshold_patch is not None:
        inputs["threshold_patch"] = payload.threshold_patch
    request_hash = sha(
        {
            "schedule_id": schedule_id,
            "outcome_id": outcome.outcome_id,
            "actor_id": principal.id,
            "inputs": inputs,
        }
    )
    plan_id = "opsdisposition_" + uuid.uuid4().hex
    request = {
        "task_id": binding.task_id,
        "schedule_id": schedule_id,
        "period_key": payload.period_key,
        "execution_plan_id": plan_id,
        "inputs": inputs,
        "native_result_hash": output["monitoring_evidence"]["result_hash"],
    }
    now = datetime.now(UTC).isoformat()
    with connect(runtime.settings.db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        old = conn.execute(
            "SELECT * FROM operations_disposition_claims WHERE idempotency_key=? OR outcome_id=?",
            (payload.idempotency_key, outcome.outcome_id),
        ).fetchone()
        if old:
            if (
                old["idempotency_key"] != payload.idempotency_key
                or old["request_hash"] != request_hash
                or old["outcome_hash"] != outcome.outcome_hash
            ):
                raise DispositionConsumed(
                    "outcome was consumed by another reviewed disposition"
                )
            if old["state"] == "completed":
                return {**json.loads(old["result_json"]), "replayed": True}
            raise DispositionConsumed(
                "disposition already consumed; reconcile native evidence before any new action"
            )
        conn.execute(
            "INSERT INTO operations_disposition_claims VALUES(?,?,?,?,?,?,?,NULL,NULL,?,NULL)",
            (
                payload.idempotency_key,
                outcome.outcome_id,
                principal.id,
                outcome.outcome_hash,
                request_hash,
                canonical(request),
                "claimed",
                now,
            ),
        )
    # This is a normal governed native plan. No parallel execution runtime or
    # fabricated approval capability is introduced for scheduled alerts.
    ref = ToolRef("strategy", "apply_monitoring_disposition")
    _manifest, tool = runtime.builtin_executor.runner._tools.resolve_with_manifest(ref)
    step = PlanStep(
        plan_id + "_step",
        plan_id,
        0,
        "处置本期红色监控",
        ref,
        inputs,
        [],
        [],
        needs_confirmation=True,
        policy=tool.policy,
        decision_point=True,
        status=StepStatus.AWAITING_CONFIRM,
    )
    plan = Plan(
        plan_id,
        binding.task_id,
        "人工确认本期监控处置",
        "builtin",
        "operations_disposition",
        [step],
        0,
        status=PlanStatus.AWAITING_CONFIRM,
    )
    try:
        plan_repo.create_plan(plan)
        governance_service.authorize_step(
            plan_id=plan_id,
            step_id=step.id,
            principal=principal,
            reason=payload.reason,
            expected_plan_revision=0,
        )
        completed = plan_executor.run(plan_id)
        if completed.status != PlanStatus.DONE:
            raise DispositionConsumed(
                "native disposition did not finish; consumed request requires evidence reconciliation"
            )
        result = plan_repo.load_step_output(step.id)
        value = {
            "state": "completed",
            "execution_plan_id": plan_id,
            "native_result": result,
            "replayed": False,
            "automatic_action_permitted": False,
        }
        _complete(runtime, payload.idempotency_key, value)
        return value
    except Exception:
        # A process may fail after an atomic native commit. Leave a durable
        # consumed claim rather than automatically invoking the Tool again.
        raise


def _complete(runtime, key, value):
    with connect(runtime.settings.db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        changed = conn.execute(
            "UPDATE operations_disposition_claims SET state='completed',result_json=?,result_hash=?,finished_at=? WHERE idempotency_key=? AND state='claimed'",
            (canonical(value), sha(value), datetime.now(UTC).isoformat(), key),
        )
        if changed.rowcount != 1:
            raise DispositionConsumed("disposition completion identity changed")


def reconcile_disposition(runtime, key, *, actor_id):
    with connect(runtime.settings.db_path) as conn:
        claim = conn.execute(
            "SELECT * FROM operations_disposition_claims WHERE idempotency_key=?",
            (key,),
        ).fetchone()
        if claim is None:
            raise KeyError(key)
        if claim["actor_id"] != actor_id:
            raise ValueError("original disposition actor required")
        request = json.loads(claim["request_json"])
        inputs = request["inputs"]
        from marvis.institution.policy import authorize_on_connection

        authorize_on_connection(
            conn, "task", request["task_id"], "approve", actor_id=actor_id
        )
        if claim["state"] == "completed":
            return {**json.loads(claim["result_json"]), "replayed": True}
    binding = _original_execution_binding(runtime, request)
    if binding is None:
        return _unresolved_disposition()
    with connect(runtime.settings.db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM audit WHERE kind='strategy.monitoring.disposition' AND target_ref=? AND outcome='succeeded'",
            (inputs["monitoring_run_id"],),
        ).fetchall()
        proofs = []
        for row in rows:
            detail = json.loads(row["detail_json"])
            if (
                detail.get("receipt_schema_version")
                == "strategy.monitoring-disposition.v1"
                and sha(detail) == row["inputs_hash"]
                and detail.get("task_id") == request["task_id"]
                and detail.get("strategy_id") == inputs["strategy_id"]
                and detail.get("source_monitoring_run_id")
                == inputs["monitoring_run_id"]
                and detail.get("disposition") == inputs["disposition"]
                and detail.get("reason") == inputs["reason"]
                and detail.get(
                    "old_monitoring_plan_id", detail.get("monitoring_plan_id")
                )
                == inputs["expected_plan_id"]
                and detail.get(
                    "old_monitoring_plan_revision",
                    detail.get("monitoring_plan_revision"),
                )
                == inputs["expected_plan_revision"]
                and detail.get(
                    "old_monitoring_plan_hash", detail.get("monitoring_plan_hash")
                )
                == inputs["expected_plan_hash"]
                and detail.get(
                    "source_monitoring_run_hash",
                    detail.get("monitoring_run_result_hash"),
                )
                == request["native_result_hash"]
                and (
                    inputs["disposition"] != "adjust_threshold"
                    or _patch_matches(
                        inputs["threshold_patch"], detail.get("threshold_patch")
                    )
                )
                and _receipt_matches_output(detail, binding["output"])
            ):
                proofs.append(detail)
        if len(proofs) != 1:
            return _unresolved_disposition()
        receipt = proofs[0]
    value = {
        "state": "completed",
        "execution_plan_id": request["execution_plan_id"],
        "native_receipt": receipt,
        "native_result": binding["output"],
        "native_output_ref": binding["output_ref"],
        "native_step_run_id": binding["evidence"]["step_run_id"],
        "reconciled": True,
        "replayed": False,
        "automatic_action_permitted": False,
    }
    _complete(runtime, key, value)
    return value


def _unresolved_disposition():
    return {
        "state": "consumed_unresolved",
        "native_effect": "unknown",
        "automatic_action_permitted": False,
        "next_action": "核对原请求的原生执行计划、结果绑定与审计；此请求不会重新执行",
    }


def _original_execution_binding(runtime, request):
    # A same-valued business audit can come from another approved Plan. Only
    # this claim's immutable invocation/output journal proves it completed.
    from marvis.orchestrator.errors import PlanNotFoundError
    from marvis.repositories.plans import PlanRepository
    from marvis.state_machine import ConflictError

    repo = PlanRepository(runtime.settings.db_path)
    try:
        plan = repo.load_plan(request["execution_plan_id"])
        if plan.task_id != request["task_id"] or len(plan.steps) != 1:
            return None
        step = plan.steps[0]
        if (
            step.id != plan.id + "_step"
            or step.tool_ref != ToolRef("strategy", "apply_monitoring_disposition")
            or step.inputs != request["inputs"]
            or not step.output_ref
        ):
            return None
        binding = repo.load_step_recovery_binding(step.id, step.output_ref)
        if (
            binding["plan_id"] != plan.id
            or binding["task_id"] != request["task_id"]
            or binding["inputs"] != request["inputs"]
        ):
            return None
        return binding
    except (PlanNotFoundError, KeyError, ValueError, ConflictError):
        # Missing/corrupt/partially committed result evidence does not grant a
        # repeat invocation, even when a business effect might already exist.
        return None


def _receipt_matches_output(receipt, output):
    common = {
        "strategy_id": "strategy_id",
        "source_monitoring_run_id": "source_monitoring_run_id",
        "disposition": "disposition",
        "status": "status",
        "reason": "reason",
    }
    if receipt.get("disposition") == "adjust_threshold":
        fields = {
            "new_monitoring_plan_id": "monitoring_plan_id",
            "new_monitoring_plan_revision": "monitoring_plan_revision",
            "new_monitoring_plan_hash": "monitoring_plan_hash",
            "new_monitoring_run_id": "resolved_monitoring_run_id",
        }
    else:
        fields = {
            "monitoring_plan_id": "monitoring_plan_id",
            "monitoring_plan_revision": "monitoring_plan_revision",
            "monitoring_plan_hash": "monitoring_plan_hash",
            "resolved_monitoring_run_id": "resolved_monitoring_run_id",
            "new_task_id": "new_task_id",
            "new_strategy_id": "new_strategy_id",
            "new_dataset_id": "new_dataset_id",
        }
    return all(
        receipt.get(receipt_key) == output.get(output_key)
        for receipt_key, output_key in {**common, **fields}.items()
    )


def _patch_matches(requested, applied):
    # Native threshold code may drop fields equal to their old value. Every
    # effective field must still be exactly the human-reviewed value.
    return bool(applied) and all(
        key in requested
        and all(
            field in requested[key] and value == requested[key][field]
            for field, value in changes.items()
        )
        for key, changes in applied.items()
    )
