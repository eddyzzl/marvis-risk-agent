"""Trusted outcome lookup followed by the existing completion protocol.

Neither HTTP callers nor an LLM supply an outcome. The only execution resumed
here is a frozen completion obligation; this module never invokes a Tool.
"""

from dataclasses import asdict
from datetime import UTC, datetime
import json
import uuid

from marvis.db_schema import connect
from marvis.governance.outcome_verifiers import (
    OutcomeProof,
    OutcomeVerifierRegistry,
    VerificationTarget,
)
from marvis.job_heartbeat import heartbeat_job
from marvis.orchestrator.completion import (
    complete_step,
    complete_workflow,
    pending_workflow_completion,
    prepare_workflow_completion,
    step_completion_identity,
    _workflow_event_id,
)
from marvis.orchestrator.contracts import PlanStatus, StepStatus, plan_fingerprint
from marvis.orchestrator.evidence import payload_hash
from marvis.orchestrator.reviewer import FinalReview
from marvis.plugins.invocation import valid_invocation_contract
from marvis.repositories.tasks import TaskRepository
from marvis.state_machine import ConflictError


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _now():
    return datetime.now(UTC).isoformat()


def _plan_binding(plan):
    return payload_hash(
        {
            "id": plan.id,
            "task_id": plan.task_id,
            "revision": plan.replan_count,
            "goal": plan.goal,
            "success_criteria": plan.success_criteria,
            "steps": [
                {
                    "id": step.id,
                    "tool": asdict(step.tool_ref),
                    "inputs": step.inputs,
                    "depends_on": step.depends_on,
                    "policy": step.policy.to_dict(),
                    "post_checks": [asdict(check) for check in step.post_checks],
                    "decision_point": step.decision_point,
                    "needs_confirmation": step.needs_confirmation,
                    "output_ref": step.output_ref,
                }
                for step in plan.steps
            ],
        }
    )


def register_governed_outcome_verifier(registry, governance):
    """An explicitly supported original producer, never a generic current-state lookup."""

    def verify(target, conn):
        binding = target.binding
        result = governance.verify_producer_outcome(binding["run_id"], conn=conn)
        if result.get("invocation_contract_hash") != payload_hash(
            binding["invocation_contract"]
        ):
            return OutcomeProof(
                "unknown", target.id, reason="原始调用声明与生产者凭据不匹配。"
            )
        output = result.get("output")
        outcome = result.get("outcome", "unknown")
        if outcome == "applied" and not isinstance(output, dict):
            return OutcomeProof(
                "unknown",
                target.id,
                reason="生产者未原子保存完整输出，不能恢复工具结果。",
            )
        return OutcomeProof(
            outcome,
            target.id,
            result.get("receipt_id", ""),
            result.get("receipt_hash", ""),
            {
                "applied": "已从原生产者凭据核对执行结果，保留原始输出。",
                "not_applied_fenced": "已确认原调用未生效；后续操作仍需正常审批。",
                "unknown": "原生产者凭据不足，执行结果仍待核对。",
            }[outcome],
            _json(output) if output is not None else None,
        )

    registry.register(
        "tool", "strategy.adopt_strategy", "strategy.atomic-producer.v1", verify
    )
    for producer in (
        "collection.queue_batch",
        "collection.execute_reference",
        "collection.cancel_batch",
    ):
        registry.register("tool", producer, "collection.atomic-producer.v1", verify)


class ExecutionReconciler:
    def __init__(self, executor, verifiers=None):
        self.executor = executor
        self.repo = executor._repo
        self.verifiers = verifiers or OutcomeVerifierRegistry()

    def _targets(self, plan):
        if plan.status not in {
            PlanStatus.FAILED,
            PlanStatus.RUNNING,
            PlanStatus.REVIEW,
            PlanStatus.CANCELLED,
        } and not (
            plan.status == PlanStatus.DONE
            and self.repo.workflow_completion_pending(plan.id)
        ):
            return []
        base = {
            "plan_id": plan.id,
            "task_id": plan.task_id,
            "plan_binding": _plan_binding(plan),
        }
        targets = []
        blocked = set(self.repo.unreconciled_step_ids([s.id for s in plan.steps]))
        with connect(self.repo.db_path) as conn:
            for step in plan.steps:
                if step.status not in {
                    StepStatus.FAILED,
                    StepStatus.CHECKING,
                    StepStatus.RUNNING,
                }:
                    continue
                if step.id not in blocked and step.status != StepStatus.CHECKING:
                    continue
                runs = self.repo.list_step_runs(step.id)
                run = runs[-1] if runs else None
                if runs:
                    run = max(runs, key=lambda item: item["attempt"])
                if not step.output_ref:
                    if run is not None:
                        targets.append(
                            VerificationTarget(
                                "tool",
                                run["tool_ref"],
                                _json(
                                    {
                                        **base,
                                        "step_id": step.id,
                                        "run_id": run["id"],
                                        "invocation_contract": run.get(
                                            "invocation_contract"
                                        ),
                                        "input_hash": payload_hash(run["input"]),
                                        "dispatch_started_at": run.get(
                                            "dispatch_started_at"
                                        ),
                                    }
                                ),
                            )
                        )
                    continue
                identity, _, evidence = step_completion_identity(self.repo, plan, step)
                if run is None or evidence.get("step_run_id") != run["id"]:
                    raise ConflictError("已保存输出不属于最近一次原始调用。")
                events = (
                    ["feature.computed"] if step.tool_ref.plugin == "feature" else []
                ) + ["step.completed"]
                event_ids = [
                    payload_hash({"event": name, **identity}) for name in events
                ]
                common = {
                    **base,
                    **identity,
                    "invocation_contract": run.get("invocation_contract")
                    if run
                    else None,
                    "evidence_hash": payload_hash(evidence),
                }
                targets.extend(self._hook_targets(conn, common, event_ids))
            workflow_event = conn.execute(
                "SELECT 1 FROM hook_events WHERE event_id=?",
                (_workflow_event_id(plan),),
            ).fetchone()
            all_outputs_complete = bool(plan.steps) and all(
                step.status in {StepStatus.DONE, StepStatus.SKIPPED}
                for step in plan.steps
            )
            if self.repo.workflow_completion_pending(plan.id) or (
                workflow_event and all_outputs_complete
            ):
                snapshot = pending_workflow_completion(self.repo, plan)
                summary_ref = self.repo.latest_plan_summary_ref(plan.id)
                summary = (
                    self.repo.load_plan_summary(summary_ref) if summary_ref else None
                )
                if snapshot is not None and not self.repo.workflow_completion_pending(
                    plan.id
                ):
                    return targets
                if snapshot is not None or summary is not None:
                    outputs = []
                    for step in plan.steps:
                        if step.status == StepStatus.DONE:
                            output_binding = self.repo.load_step_presentation_binding(
                                step.id, step.output_ref
                            )
                            evidence = output_binding["evidence"]
                            original_run = next(
                                (
                                    item
                                    for item in self.repo.list_step_runs(step.id)
                                    if item["id"] == evidence["step_run_id"]
                                ),
                                None,
                            )
                            outputs.append(
                                {
                                    "step_id": step.id,
                                    "output_ref": step.output_ref,
                                    "output_hash": evidence["output_hash"],
                                    "execution_id": evidence["step_run_id"],
                                    "invocation_contract": original_run.get(
                                        "invocation_contract"
                                    )
                                    if original_run
                                    else None,
                                }
                            )
                    targets.extend(
                        self._hook_targets(
                            conn,
                            {
                                **base,
                                "step_id": None,
                                "checkpoint_hash": payload_hash(snapshot),
                                "summary_ref": summary_ref,
                                "summary_hash": payload_hash(summary),
                                "outputs": outputs,
                            },
                            [_workflow_event_id(plan)],
                        )
                    )
        return targets

    def _hook_targets(self, conn, common, event_ids):
        result = []
        for event_id in event_ids:
            event = conn.execute(
                "SELECT * FROM hook_events WHERE event_id=?", (event_id,)
            ).fetchone()
            if not event:
                continue
            for target in json.loads(event["targets_json"]):
                if not target["required"]:
                    continue
                row = conn.execute(
                    "SELECT * FROM hook_deliveries WHERE event_id=? AND target_ref=?",
                    (event_id, target["ref"]),
                ).fetchone()
                if row and row["status"] == "succeeded":
                    continue
                bound = {
                    **common,
                    "event_id": event_id,
                    "target_ref": target["ref"],
                    "target": target,
                    "payload_hash": event["payload_hash"],
                    "payload": json.loads(event["payload_json"])
                    if event["payload_json"]
                    else None,
                    "generation": row["generation"] if row else 0,
                }
                result.append(
                    VerificationTarget(
                        "hook" if row else "completion", target["ref"], _json(bound)
                    )
                )
        if not result:
            result.append(
                VerificationTarget(
                    "completion",
                    "platform.completion",
                    _json({**common, "event_ids": event_ids}),
                )
            )
        return result

    def describe(self, plan):
        try:
            targets = self._targets(plan)
        except (ValueError, KeyError, ConflictError) as exc:
            return {
                "blocked": True,
                "targets": [],
                "reason": f"原执行绑定未通过校验：{exc}",
            }
        return {
            "blocked": bool(targets),
            "continuation": self._continuation(plan) if not targets else None,
            "targets": [
                {
                    "id": target.id,
                    "step_id": target.binding.get("step_id"),
                    "kind": target.kind,
                    "producer": target.producer,
                    "action": "resume_completion"
                    if plan.status == PlanStatus.CANCELLED and target.kind != "tool"
                    else "reconcile",
                    "supported": target.kind == "completion"
                    or self.verifiers.supports(target),
                    "reason": "可读取原生产者凭据核对。"
                    if target.kind == "completion" or self.verifiers.supports(target)
                    else "该动作尚未接入可信核对器；已有证据将继续保留。",
                }
                for target in targets
            ],
        }

    def _continuation(self, plan):
        """Expose remaining work without dispatching it as part of reconciliation."""
        if plan.status != PlanStatus.RUNNING or any(
            step.status not in {StepStatus.DONE, StepStatus.SKIPPED, StepStatus.PENDING}
            for step in plan.steps
        ):
            return None
        remaining = [step.id for step in plan.steps if step.status == StepStatus.PENDING]
        if not remaining or TaskRepository(self.repo.db_path).task_has_active_job(plan.task_id):
            return None
        if self.repo.unreconciled_step_ids([step.id for step in plan.steps]):
            return None
        with connect(self.repo.db_path) as conn:
            completed = conn.execute(
                "SELECT 1 FROM execution_reconciliation_completions WHERE plan_id=? LIMIT 1",
                (plan.id,),
            ).fetchone()
        if not completed:
            return None
        return {
            "expected_plan_fingerprint": plan_fingerprint(plan),
            "remaining_step_ids": remaining,
        }

    def reconcile(self, plan_id, target_id, *, resume_completion=False, actor_id=None):
        plan = self.repo.load_plan(plan_id)
        tasks = TaskRepository(self.repo.db_path)
        job_id = tasks.start_job(plan.task_id, "plan")
        if not tasks.mark_job_running(job_id):
            raise ConflictError("task already has an active job")
        try:
            with self.verifiers.reader(actor_id), heartbeat_job(tasks, job_id):
                result = self._reconcile(
                    plan_id, target_id, resume_completion=resume_completion
                )
            tasks.finish_job(job_id, status="succeeded")
            return result
        except BaseException as exc:
            tasks.finish_job(
                job_id,
                status="failed",
                error_name=type(exc).__name__,
                error_value=str(exc),
            )
            raise

    def _authorize_reads(self, plan_id, conn=None):
        if conn is None:
            with connect(self.repo.db_path) as snapshot:
                return self._authorize_reads(plan_id, snapshot)
        for run in conn.execute(
            "SELECT id,tool_ref FROM plan_step_runs WHERE plan_id=?", (plan_id,)
        ).fetchall():
            self.verifiers.authorize_read(run["tool_ref"], run["id"], conn)

    def authorize_proposed_plan_read(self, plan, *, actor_id=None):
        with self.verifiers.reader(actor_id), connect(self.repo.db_path) as conn:
            conn.execute("PRAGMA query_only=ON")
            for step in plan.steps:
                self.verifiers.authorize_inputs(
                    step.tool_ref.plugin + "." + step.tool_ref.tool,
                    plan.task_id, step.inputs, conn,
                )

    def authorize_plan_read(self, plan_id, *, actor_id=None):
        """Read-only authorization; no job, completion, or execution mutation."""
        with self.verifiers.reader(actor_id), connect(self.repo.db_path) as conn:
            conn.execute("PRAGMA query_only=ON")
            self._authorize_reads(plan_id, conn)
            for step in conn.execute(
                """SELECT s.*,p.task_id FROM plan_steps s JOIN plans p ON p.id=s.plan_id WHERE p.id=?""",
                (plan_id,),
            ).fetchall():
                self.verifiers.authorize_inputs(
                    step["tool_plugin"] + "." + step["tool_name"],
                    step["task_id"], json.loads(step["inputs_json"]), conn,
                )

    def authorize_task_read(self, task_id, *, actor_id=None):
        with connect(self.repo.db_path) as conn:
            plans = conn.execute(
                "SELECT id FROM plans WHERE task_id=?", (task_id,)
            ).fetchall()
        for plan in plans:
            self.authorize_plan_read(plan["id"], actor_id=actor_id)

    def _reconcile(self, plan_id, target_id, *, resume_completion=False):
        plan = self.repo.load_plan(plan_id)
        self._authorize_reads(plan_id)
        targets = self._targets(plan)
        target = next((item for item in targets if item.id == target_id), None)
        if target is None:
            return self._resume_resolved(
                plan, target_id, resume_completion=resume_completion
            )
        if resume_completion and (
            plan.status != PlanStatus.CANCELLED or target.kind == "tool"
        ):
            raise ConflictError("请先核对原工具结果，再明确恢复已取消计划的完成步骤。")
        if target.kind == "completion":
            verifier_id, proof = (
                "platform.undispatched-completion.v1",
                OutcomeProof(
                    "applied",
                    target.id,
                    target.id,
                    target.id,
                    "原工具结果已验证；仅恢复未派发或已结算的完成步骤。",
                ),
            )
        else:
            with connect(self.repo.db_path) as conn:
                verifier_id, proof = self.verifiers.verify(target, conn)
        resolution_id = self._apply(plan_id, target, verifier_id, proof)
        if proof.outcome != "unknown":
            self._finish_completion(
                plan_id,
                target,
                resolution_id,
                proof.outcome,
                resume_cancelled=resume_completion,
            )
        return {
            "outcome": proof.outcome,
            "reason": proof.reason,
            "resolution_id": resolution_id,
        }

    def _apply(self, plan_id, target, verifier_id, proof):
        binding = target.binding
        with connect(self.repo.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = self._targets(self.repo.load_plan(plan_id))
            if not any(item.id == target.id for item in current):
                raise ConflictError("执行绑定已变化，请刷新后重新核对。")
            if target.kind != "completion":
                current_verifier, current_proof = self.verifiers.verify(target, conn)
                if current_verifier != verifier_id or current_proof != proof:
                    raise ConflictError("原生产者凭据已变化，请重新核对。")
            previous = {}
            if target.kind == "hook":
                previous = dict(
                    conn.execute(
                        "SELECT * FROM hook_deliveries WHERE event_id=? AND target_ref=?",
                        (binding["event_id"], binding["target_ref"]),
                    ).fetchone()
                )
            elif target.kind == "tool":
                previous = dict(
                    conn.execute(
                        "SELECT * FROM plan_step_runs WHERE id=?", (binding["run_id"],)
                    ).fetchone()
                )
            proof_payload = asdict(proof)
            proof_hash = payload_hash(proof_payload)
            resolution_id = f"reconcile-{uuid.uuid4().hex}"
            conn.execute(
                "INSERT OR IGNORE INTO execution_reconciliations VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    resolution_id,
                    target.id,
                    target.id,
                    target.binding_json,
                    verifier_id,
                    proof.outcome,
                    _json(proof_payload),
                    proof_hash,
                    _json(previous),
                    _now(),
                ),
            )
            row = conn.execute(
                "SELECT id FROM execution_reconciliations WHERE target_id=? AND proof_hash=?",
                (target.id, proof_hash),
            ).fetchone()
            resolution_id = row["id"]
            if target.kind == "hook" and proof.outcome == "applied":
                conn.execute(
                    "UPDATE hook_deliveries SET status='succeeded',result_json=?,updated_at=? WHERE event_id=? AND target_ref=? AND generation=?",
                    (
                        _json(
                            {
                                "ok": True,
                                "output": None,
                                "error": None,
                                "error_kind": None,
                                "duration_ms": 0,
                            }
                        ),
                        _now(),
                        binding["event_id"],
                        binding["target_ref"],
                        binding["generation"],
                    ),
                )
            elif target.kind == "hook" and proof.outcome == "not_applied_fenced":
                conn.execute(
                    "UPDATE hook_deliveries SET retry_authorization_id=? WHERE event_id=? AND target_ref=? AND generation=?",
                    (
                        resolution_id,
                        binding["event_id"],
                        binding["target_ref"],
                        binding["generation"],
                    ),
                )
            elif target.kind == "tool" and proof.outcome == "applied":
                self._restore_tool(conn, binding, proof)
            elif target.kind == "tool" and proof.outcome == "not_applied_fenced":
                conn.execute(
                    "UPDATE plan_steps SET error=? WHERE id=?",
                    (
                        "可信凭据确认原调用未生效，可通过正常审批和重试继续。",
                        binding["step_id"],
                    ),
                )
        return resolution_id

    def _restore_tool(self, conn, binding, proof):
        contract = binding["invocation_contract"]
        if not valid_invocation_contract(contract) or not proof.output_json:
            raise ConflictError("原生产者凭据不包含完整调用和输出，不能恢复。")
        output = json.loads(proof.output_json)
        if not isinstance(output, dict):
            raise ConflictError("原生产者输出格式无效。")
        run_id, step_id = binding["run_id"], binding["step_id"]
        # The temporary running state never leaves this transaction. Reuse the
        # original output/evidence writer, not a second result representation.
        changed = conn.execute(
            "UPDATE plan_step_runs SET status='running' WHERE id=? AND output_ref IS NULL AND status IN ('failed','interrupted','running')",
            (run_id,),
        )
        if changed.rowcount != 1:
            raise ConflictError("原调用已经保存结果或状态已变化。")
        output_ref = self.repo.store_step_output(
            step_id,
            output,
            evidence={
                "step_run_id": run_id,
                "producer_invocation_id": run_id,
                "raw_output_hash": payload_hash(output),
                "canonical_binding_verified": True,
                "tool_version": contract["tool_version"],
                "manifest_hash": contract["manifest_hash"],
            },
            _connection=conn,
        )
        conn.execute(
            "UPDATE plan_step_runs SET status='succeeded',error=NULL,error_kind=NULL,finished_at=? WHERE id=?",
            (_now(), run_id),
        )
        conn.execute(
            "UPDATE plan_steps SET status='checking',output_ref=?,error=NULL WHERE id=?",
            (output_ref, step_id),
        )
        if (
            conn.execute(
                "SELECT status FROM plans WHERE id=?", (binding["plan_id"],)
            ).fetchone()[0]
            == "cancelled"
        ):
            conn.execute(
                "UPDATE plan_steps SET status='failed',error=? WHERE id=?",
                (
                    "原工具结果已核对，计划仍已取消；explicit reconciliation required before completion-only resume",
                    step_id,
                ),
            )

    def _resume_resolved(self, plan, target_id, *, resume_completion=False):
        with connect(self.repo.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM execution_reconciliations WHERE target_id=? AND outcome != 'unknown' ORDER BY created_at DESC LIMIT 1",
                (target_id,),
            ).fetchone()
        if not row or json.loads(row["target_json"])["plan_id"] != plan.id:
            raise ConflictError("核对目标已变化或不存在，请刷新。")
        binding = json.loads(row["target_json"])
        if binding["plan_binding"] != _plan_binding(plan) and not (
            binding.get("run_id") and self._restored_run_matches(plan, binding)
        ):
            raise ConflictError("计划已修订，旧核对凭据不能继续。")
        if binding.get("run_id"):
            runs = self.repo.list_step_runs(binding["step_id"])
            if (
                not runs
                or max(runs, key=lambda item: item["attempt"])["id"]
                != binding["run_id"]
            ):
                raise ConflictError("原调用已被后续尝试替代，请刷新。")
        # A recovered Tool adds its original output_ref; no other revision may
        # piggyback on the old resolution. Completion receipts make repeat calls
        # idempotent even after final state transitions.
        with connect(self.repo.db_path) as conn:
            done = conn.execute(
                "SELECT 1 FROM execution_reconciliation_completions WHERE resolution_id=?",
                (row["id"],),
            ).fetchone()
        if not done:
            target = VerificationTarget(
                "tool" if binding.get("run_id") else "hook", "", row["target_json"]
            )
            self._finish_completion(
                plan.id,
                target,
                row["id"],
                row["outcome"],
                resume_cancelled=resume_completion,
            )
        return {
            "outcome": row["outcome"],
            "reason": "已读取原核对记录。",
            "resolution_id": row["id"],
        }

    def _restored_run_matches(self, plan, binding):
        step = next((s for s in plan.steps if s.id == binding.get("step_id")), None)
        if step is None or not step.output_ref:
            return False
        original_ref = step.output_ref
        step.output_ref = None
        matches = _plan_binding(plan) == binding["plan_binding"]
        step.output_ref = original_ref
        evidence = self.repo.load_step_evidence(
            step.id, version=int(original_ref.rsplit(":v", 1)[1])
        )
        return matches and evidence.get("step_run_id") == binding["run_id"]

    def _finish_completion(
        self, plan_id, target, resolution_id, outcome, *, resume_cancelled=False
    ):
        binding = target.binding
        if target.kind == "tool" and outcome == "not_applied_fenced":
            return
        plan = self.repo.load_plan(plan_id)
        step_id = binding.get("step_id")
        if plan.status == PlanStatus.CANCELLED and not resume_cancelled:
            return
        for pending in self._targets(plan):
            if pending.kind != "hook" or pending.binding.get("step_id") != step_id:
                continue
            with connect(self.repo.db_path) as conn:
                delivery = conn.execute(
                    "SELECT retry_authorization_id FROM hook_deliveries WHERE event_id=? AND target_ref=?",
                    (pending.binding["event_id"], pending.binding["target_ref"]),
                ).fetchone()
            if not delivery or not delivery["retry_authorization_id"]:
                return
        with connect(self.repo.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._authorize_reads(plan_id, conn)
            current_status = conn.execute(
                "SELECT status FROM plans WHERE id=?", (plan_id,)
            ).fetchone()[0]
            if current_status == "cancelled" and not resume_cancelled:
                return
            if step_id:
                conn.execute(
                    "UPDATE plan_steps SET status='checking',error=NULL WHERE id=? AND status='failed' AND output_ref IS NOT NULL",
                    (step_id,),
                )
            conn.execute(
                "UPDATE plans SET status=? WHERE id=? AND status IN ('failed','cancelled')",
                ("running" if step_id else "review", plan_id),
            )
        plan = self.repo.load_plan(plan_id)
        try:
            if step_id:
                step = next(s for s in plan.steps if s.id == step_id)
                if step.status != StepStatus.DONE:
                    complete_step(
                        self.repo,
                        self.executor._reviewer,
                        self.executor._hooks,
                        plan,
                        step,
                        self.repo.load_step_output(step_id),
                    )
                if step.status == StepStatus.FAILED:
                    self.repo.set_plan_status(plan_id, PlanStatus.FAILED)
                    return
                if not all(
                    s.status in {StepStatus.DONE, StepStatus.SKIPPED}
                    for s in plan.steps
                ):
                    self._record_completion(resolution_id, plan, step_id)
                    return
            snapshot = pending_workflow_completion(self.repo, plan)
            if snapshot is None:
                summary_ref = binding.get("summary_ref")
                if summary_ref:
                    summary = self.repo.load_plan_summary(summary_ref)
                    if payload_hash(summary) != binding.get("summary_hash"):
                        raise ConflictError("原完成复核记录已变化。")
                    review = FinalReview(**summary)
                else:
                    outputs = {
                        s.id: self.repo.load_bound_step_output(s.id)
                        for s in plan.steps
                        if s.status == StepStatus.DONE
                    }
                    review = self.executor._reviewer.final_review(
                        plan, outputs, plan.goal
                    )
                    summary_ref = self.repo.store_plan_summary(plan.id, review)
                if review.execution_completed is None and review.goal_doubt:
                    if plan.status != PlanStatus.REVIEW:
                        self.repo.set_plan_status(plan.id, PlanStatus.REVIEW)
                    return
                snapshot = prepare_workflow_completion(
                    self.repo, plan, summary_ref, review, hooks=self.executor._hooks
                )
            if plan.status not in {PlanStatus.DONE, PlanStatus.FAILED}:
                complete_workflow(
                    self.repo,
                    self.executor._hooks,
                    self.executor._state,
                    plan,
                    snapshot,
                )
            self._record_completion(resolution_id, plan, step_id)
        except Exception as exc:
            if (
                step_id
                and self.repo.load_plan(plan_id)
                .steps[[s.id for s in plan.steps].index(step_id)]
                .status
                != StepStatus.DONE
            ):
                step.error = (
                    f"completion interrupted: {exc}; explicit reconciliation required"
                )
                step.status = StepStatus.FAILED
                self.repo.update_step(step)
            else:
                self.repo.append_loop_event(
                    plan_id, {"type": "hook_completion_failed", "reason": str(exc)}
                )
            current = self.repo.load_plan(plan_id)
            if current.status in {PlanStatus.RUNNING, PlanStatus.REVIEW}:
                self.repo.set_plan_status(plan_id, PlanStatus.FAILED)
            raise

    def _record_completion(self, resolution_id, plan, step_id):
        with connect(self.repo.db_path) as conn:
            failures = sum(
                event.type == "hook_completion_failed" for event in plan.loop_events
            )
            conn.execute(
                "INSERT OR IGNORE INTO execution_reconciliation_completions VALUES (?,?,?,?,?)",
                (resolution_id, plan.id, step_id, failures, _now()),
            )
