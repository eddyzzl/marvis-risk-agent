"""Execution adapter for gate-level adjust/replan actions."""

from __future__ import annotations

from collections.abc import Callable

from marvis.agent.adjust_specs import adjust_param_error
from marvis.agent.driver_turn import DriverMessage, DriverTurn
from marvis.agent.plan_utils import downstream_step_ids, find_step
from marvis.orchestrator.contracts import Plan, PlanStatus, PlanStep


class GateExecutionAdapter:
    """Apply structured gate actions while keeping PlanDriver focused on turn flow."""

    def __init__(
        self,
        plan_repo,
        executor,
        *,
        safe_output: Callable[[str], object],
        run_and_handle: Callable[..., DriverTurn],
        plan_overview_message: Callable[[Plan], DriverMessage],
    ):
        self._repo = plan_repo
        self._executor = executor
        self._safe_output = safe_output
        self._run_and_handle = run_and_handle
        self._plan_overview_message = plan_overview_message

    def apply_replan(self, plan: Plan, gate: PlanStep | None, instruction, run_seq) -> DriverTurn:
        """Regenerate remaining steps from a structural instruction and continue."""
        replan = getattr(self._executor, "replan_from_instruction", None)
        if replan is None or not replan(plan.id, instruction):
            return self._instruction_message(
                plan,
                gate,
                run_seq,
                "重规划未成功(重规划预算用尽或指令无法执行);可改为在节点处「调参重算」,"
                "或重新创建任务调整配置。",
            )
        revised = self._repo.load_plan(plan.id)
        if revised.status == PlanStatus.VALIDATED:
            return DriverTurn(revised.id, revised.status.value, [
                DriverMessage("chat", "已按指令重规划,请查看新计划。", {"plan_id": revised.id, "run_seq": run_seq}),
                self._plan_overview_message(revised),
            ])
        turn = self._run_and_handle(plan.id, run_seq=run_seq)
        turn.messages.insert(
            0,
            DriverMessage("chat", "已按指令重规划并继续执行。", {"plan_id": plan.id, "run_seq": run_seq}),
        )
        return turn

    def apply_adjust(self, plan: Plan, gate: PlanStep, params, run_seq) -> DriverTurn:
        """Apply declared parameter overrides, reset affected steps, and rerun."""
        deps = [step for step in (find_step(plan, dep_id) for dep_id in (gate.depends_on or [])) if step is not None]
        if not deps:
            return self._instruction_message(plan, gate, run_seq, "没找到可调整的上一步,请重新确认。")
        params = params or {}
        validation_error = adjust_param_error(params)
        if validation_error:
            return self._instruction_message(plan, gate, run_seq, validation_error)

        primary = None
        adjusted_ids: list[str] = []
        for dep in deps:
            overrides = {key: value for key, value in params.items() if key in (dep.inputs or {})}
            if "sample_weight_col" in overrides:
                if dep.tool_ref.tool != "choose_modeling_spec":
                    overrides.pop("sample_weight_col", None)
                else:
                    sample_weight_error = self._sample_weight_adjust_error(dep.id, overrides["sample_weight_col"])
                    if sample_weight_error:
                        return self._instruction_message(plan, gate, run_seq, sample_weight_error)
            if not overrides:
                continue
            self._repo.reset_step(dep.id, inputs={**(dep.inputs or {}), **overrides})
            adjusted_ids.append(dep.id)
            if primary is None:
                primary = dep
        if primary is None:
            available = sorted({str(key) for dep in deps for key in (dep.inputs or {}).keys()})
            hint = f"可调整参数: {', '.join(available)}。" if available else "当前节点没有声明可调整参数。"
            return self._instruction_message(
                plan,
                gate,
                run_seq,
                f"没有识别到可调整的参数,未重算。{hint}",
            )
        reset_ids = self._reset_downstream_steps(plan, adjusted_ids)
        if gate.id not in reset_ids:
            self._repo.reset_step(gate.id)
        turn = self._run_and_handle(plan.id, run_seq=run_seq)
        turn.messages.insert(
            0,
            DriverMessage(
                "chat",
                f"已按指令调整参数 {dict(params)} 并重算「{primary.title}」。",
                {"plan_id": plan.id, "step_id": primary.id, "run_seq": run_seq},
            ),
        )
        return turn

    def _reset_downstream_steps(self, plan: Plan, root_ids: list[str]) -> set[str]:
        downstream_ids = downstream_step_ids(plan, root_ids)
        reset_ids: set[str] = set()
        for step in sorted(
            (step for step in plan.steps if step.id in downstream_ids),
            key=lambda item: (item.index, item.id),
        ):
            self._repo.reset_step(step.id)
            reset_ids.add(step.id)
        return reset_ids

    def _sample_weight_adjust_error(self, step_id: str, value) -> str | None:
        selected = str(value or "").strip()
        if not selected:
            return None
        output = self._safe_output(step_id)
        if not isinstance(output, dict):
            return "缺少建模规格输出,无法调整样本权重列。"
        candidates = [str(col) for col in (output.get("sample_weight_candidates") or []) if str(col).strip()]
        current = str(output.get("sample_weight_col") or "").strip()
        allowed = set(candidates)
        if current:
            allowed.add(current)
        if selected not in allowed:
            display = "、".join(candidates) if candidates else "无"
            return f"样本权重列 `{selected}` 不在已检测候选列中,未重算。候选列:{display}。"
        return None

    def _instruction_message(self, plan: Plan, gate: PlanStep | None, run_seq, text) -> DriverTurn:
        return DriverTurn(
            plan.id,
            plan.status.value,
            [DriverMessage("gate", text, {"plan_id": plan.id, "step_id": gate.id if gate else None, "run_seq": run_seq})],
        )


__all__ = ["GateExecutionAdapter"]
