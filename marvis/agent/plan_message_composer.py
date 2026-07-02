"""Message composer for PlanDriver turns.

The driver owns state transitions; this module owns the assistant-facing
message payloads and metadata envelopes returned after those transitions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from marvis.agent.driver_turn import DriverMessage
from marvis.agent.gate_adapters import render_gate_dependencies
from marvis.agent.gate_payloads import build_model_delivery_payload
from marvis.agent.gates import build_failure_envelope, extract_gate_envelope
from marvis.agent.plan_utils import downstream_step_ids, find_step
from marvis.agent.renderers import render_tool_output
from marvis.orchestrator.contracts import Plan, PlanStep, StepStatus


class PlanMessageComposer:
    """Compose PlanDriver messages without mutating plan state."""

    def __init__(
        self,
        *,
        load_output: Callable[[str], Any],
        latest_failed_step_run_error_kind: Callable[[str], str | None] | None = None,
    ):
        self._load_output = load_output
        self._latest_failed_step_run_error_kind = latest_failed_step_run_error_kind

    def plan_overview_message(self, plan: Plan) -> DriverMessage:
        order: list[str] = []
        by_phase: dict[str, list[str]] = {}
        for step in plan.steps:
            phase = step.phase or "步骤"
            if phase not in by_phase:
                by_phase[phase] = []
                order.append(phase)
            by_phase[phase].append(step.title)
        lines = ["我已生成执行计划,会在每个关键节点停下与你确认:"]
        for phase in order:
            lines.append(f"**{phase}**:{' → '.join(by_phase[phase])}")
        lines.append("确认「开始」后按计划执行。")
        meta = {"plan_id": plan.id, "kind": "plan_overview"}
        meta["gate_envelope"] = extract_gate_envelope({"metadata": meta}).to_dict()
        return DriverMessage("plan_overview", "\n".join(lines), meta)

    def gate_message(self, plan: Plan, gate: PlanStep | None, *, run_seq) -> DriverMessage:
        rendered = render_gate_dependencies(plan, gate, self._safe_output)
        parts = rendered.parts
        if not parts:
            parts.append("上一步已完成。")
        # UX-2: the frontend now mounts the structured gate widgets (screening
        # table / dedup picker / modeling setup panel / C1 role form) directly
        # in the agent-mode chat timeline (app.js's agentMessageHtml), so the
        # gate copy tells the user both channels work — click the widget below
        # or describe the change in free text.
        parts.append("确认请回复「确认」继续;可直接操作下方控件，或用文字说明要调整的参数。")
        meta = {
            "plan_id": plan.id,
            "step_id": gate.id if gate else None,
            "run_seq": run_seq,
            "tables": rendered.tables,
            "kind": "gate",
        }
        if rendered.output_refs:
            meta["output_refs"] = rendered.output_refs
        if rendered.screen is not None:
            meta["screen"] = rendered.screen
        if rendered.dedup is not None:
            meta["dedup"] = rendered.dedup
        if rendered.modeling_setup is not None:
            meta["modeling_setup"] = rendered.modeling_setup
        if rendered.model_delivery is not None:
            meta["model_delivery"] = rendered.model_delivery
        if rendered.red_flags:
            # AGT-9: deterministic modeling red flags (computed in
            # gate_adapters.render_gate_dependencies straight from the tuning /
            # select-experiment dependency outputs) ride alongside the existing
            # screen/dedup metadata so auto_drive._extract_red_flags can surface
            # them in the 【平台红旗 checklist】 without re-parsing table strings.
            meta["red_flags"] = rendered.red_flags
        meta["gate_envelope"] = extract_gate_envelope({"metadata": meta}).to_dict()
        return DriverMessage("gate", "\n\n".join(parts), meta)

    def done_message(self, plan: Plan, *, run_seq) -> DriverMessage:
        terminal = max(
            (step for step in plan.steps if step.status == StepStatus.DONE and step.output_ref),
            key=lambda step: step.index,
            default=None,
        )
        parts = ["✅ 计划已全部完成。"]
        tables: list[dict] = []
        output = None
        if terminal is not None:
            output = self._safe_output(terminal.id)
            if output is not None:
                text, tables = render_tool_output(terminal.tool_ref.tool, output)
                if text:
                    parts.append(text)
        meta = {"plan_id": plan.id, "run_seq": run_seq, "tables": tables}
        if terminal is not None and output is not None:
            report_output, report_step = self._report_dependency_output(plan, terminal)
            delivery = build_model_delivery_payload(
                output,
                terminal,
                report_output=report_output,
                report_step=report_step,
            )
            if delivery is not None:
                meta["model_delivery"] = delivery
        return DriverMessage("done", "\n\n".join(parts), meta)

    def review_message(self, plan: Plan, *, run_seq) -> DriverMessage:
        return DriverMessage(
            "review",
            "计划已执行完,但结果需要你复核一下再定论。",
            {"plan_id": plan.id, "run_seq": run_seq},
        )

    def instruction_message(self, plan: Plan, gate: PlanStep | None, *, run_seq, text: str) -> DriverMessage:
        return DriverMessage(
            "gate",
            text,
            {"plan_id": plan.id, "step_id": gate.id if gate else None, "run_seq": run_seq},
        )

    def manual_adjust_placeholder_message(
        self,
        plan: Plan,
        gate: PlanStep | None,
        *,
        run_seq,
    ) -> DriverMessage:
        return self.instruction_message(
            plan,
            gate,
            run_seq=run_seq,
            text="收到。确认当前结果请回复「确认」继续。",
        )

    def failed_message(self, plan: Plan, *, run_seq) -> DriverMessage:
        failed = next((step for step in plan.steps if step.status == StepStatus.FAILED), None)
        detail = f"「{failed.title}」失败:{failed.error}" if failed and failed.error else "执行中断。"
        meta = {"plan_id": plan.id, "step_id": failed.id if failed else None, "run_seq": run_seq}
        reset_steps: tuple[str, ...] = ()
        error_kind = "execution"
        if failed is not None:
            downstream = downstream_step_ids(plan, [failed.id])
            reset_steps = tuple(
                step.id
                for step in sorted(plan.steps, key=lambda item: (item.index, item.id))
                if step.id == failed.id or step.id in downstream
            )
            if self._latest_failed_step_run_error_kind is not None:
                error_kind = self._latest_failed_step_run_error_kind(failed.id) or error_kind
        meta["failure_envelope"] = build_failure_envelope(
            plan_id=plan.id,
            step_id=failed.id if failed else None,
            run_seq=run_seq,
            message=detail,
            step_inputs=failed.inputs if failed else None,
            downstream_reset_steps=reset_steps,
            error_kind=error_kind,
            retryable=failed is not None,
        ).to_dict()
        return DriverMessage("error", f"❌ {detail}", meta)

    def _report_dependency_output(self, plan: Plan, step: PlanStep) -> tuple[dict | None, PlanStep | None]:
        for dep_id in step.depends_on or []:
            dep = find_step(plan, dep_id)
            if dep is None or dep.tool_ref.tool != "generate_model_report":
                continue
            output = self._safe_output(dep.id)
            return (output if isinstance(output, dict) else None), dep
        return None, None

    def _safe_output(self, step_id: str):
        try:
            return self._load_output(step_id)
        except KeyError:
            return None


__all__ = ["PlanMessageComposer"]
