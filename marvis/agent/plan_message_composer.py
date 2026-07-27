"""Message composer for PlanDriver turns.

The driver owns state transitions; this module owns the assistant-facing
message payloads and metadata envelopes returned after those transitions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from marvis.api_report_helpers import driver_report_download_metadata
from marvis.agent.driver_turn import DriverMessage
from marvis.agent.gate_adapters import render_gate_dependencies
from marvis.agent.gate_payloads import build_model_delivery_payload
from marvis.agent.gates import build_failure_envelope, extract_gate_envelope
from marvis.agent.gates.adapters import gate_editable_input_schema
from marvis.agent.plan_utils import downstream_step_ids, find_step
from marvis.agent.renderers import render_tool_output
from marvis.orchestrator.contracts import Plan, PlanStep, StepStatus
from marvis.plugins.manifest import governance_policy_hash


class PlanMessageComposer:
    """Compose PlanDriver messages without mutating plan state."""

    def __init__(
        self,
        *,
        load_output: Callable[[str], Any],
        load_task_artifact: Callable[[str, str], Any] | None = None,
        tasks_root: Path | str | None = None,
        db_path: Path | str | None = None,
        latest_failed_step_run_error_kind: Callable[[str], str | None] | None = None,
    ):
        self._load_output = load_output
        self._load_task_artifact = load_task_artifact
        self._tasks_root = (
            None if tasks_root is None else Path(tasks_root).absolute()
        )
        artifact_repository = getattr(
            load_task_artifact,
            "__self__",
            None,
        )
        inferred_db_path = getattr(artifact_repository, "db_path", None)
        effective_db_path = (
            db_path if db_path is not None else inferred_db_path
        )
        self._db_path = (
            None
            if effective_db_path is None
            else Path(effective_db_path).absolute()
        )
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
        lines = ["我已生成执行计划，会在每个关键节点停下与你确认:"]
        for phase in order:
            lines.append(f"**{phase}**:{' → '.join(by_phase[phase])}")
        lines.append("手动模式请点击「开始执行」；Agent 模式请回复「开始」或「继续」。")
        meta = {"plan_id": plan.id, "kind": "plan_overview"}
        meta["gate_envelope"] = extract_gate_envelope({"metadata": meta}).to_dict()
        return DriverMessage("plan_overview", "\n".join(lines), meta)

    def gate_message(self, plan: Plan, gate: PlanStep | None, *, run_seq) -> DriverMessage:
        rendered = render_gate_dependencies(plan, gate, self._safe_output)
        parts = rendered.parts
        if not parts:
            parts.append("上一步已完成。")
        if rendered.feature_binning is not None:
            parts.append(
                "单变量分析已完成。是否有特征需要进一步做分箱分析？"
                "可以多选特征并设置 3–20 箱，也可以跳过；确认后才会生成最终报告。"
            )
        if rendered.special_values is not None:
            columns = [
                str(item.get("column") or "")
                for item in rendered.special_values.get("columns") or []
                if isinstance(item, dict) and str(item.get("column") or "")
            ]
            parts.append(
                f"检测到 **{len(columns)}** 个已选特征包含疑似特殊值。"
                "这些值会改变缺失处理或特征集合，必须逐列选择「转为空值」「保留原值」"
                "或「删除特征」；保留时还必须说明理由。"
            )
        # The stored message is shared by both run modes. Manual-mode analysis
        # strips this chat-only line and exposes its button; Agent mode keeps it
        # and deliberately exposes no gate action buttons.
        if rendered.special_values is not None:
            parts.append(
                "Agent 模式请在一条消息中完整说明每列策略，例如："
                "「x1 转为空值；x2 删除；x3 保留，原因：业务约定值」。"
                "仅回复「确认」不会越过此人工决策节点。"
            )
        else:
            parts.append("Agent 模式请回复「继续」或「确认」；如需调整，直接用自然语言说明。")
        meta = {
            "plan_id": plan.id,
            "step_id": gate.id if gate else None,
            "run_seq": run_seq,
            "tables": rendered.tables,
            "kind": "gate",
        }
        # LT-2: the gate step's own source tool is the reliable signal for the
        # AUTO safety layer (production step_ids are opaque "{plan}-step-N"). Carry
        # it so gates/contracts.infer_gate_envelope can set risk_flags on forced
        # human-review gates (delivery / champion / dedup / strategy adopt) and
        # halt a bare AUTO confirm on them.
        if gate is not None and gate.tool_ref is not None:
            meta["gate_source_tool"] = gate.tool_ref.tool
        if gate is not None:
            # Phase 0B: AUTO and the frontend consume the same immutable policy
            # snapshot the validator accepted for this plan step.  Tool names and
            # free-form risk text remain useful context, but are no longer the
            # authority for whether a human must act.
            meta["human_decision_gate"] = gate.policy.human_decision_gate
            meta["effect_authorization"] = gate.policy.effect_authorization
            meta["policy_hash"] = governance_policy_hash(gate.policy)
        if rendered.output_refs:
            meta["output_refs"] = rendered.output_refs
        if rendered.screen is not None:
            meta["screen"] = rendered.screen
        if rendered.dedup is not None:
            meta["dedup"] = rendered.dedup
        if rendered.join_keys is not None:
            meta["join_keys"] = rendered.join_keys
        if rendered.modeling_setup is not None:
            meta["modeling_setup"] = rendered.modeling_setup
        if rendered.model_delivery is not None:
            meta["model_delivery"] = rendered.model_delivery
            report_output, report_step = self._report_dependency_output(plan, gate)
            self._attach_report_download_metadata(
                meta,
                plan=plan,
                report_step=report_step,
                report_output=report_output,
            )
        if rendered.feature_binning is not None:
            meta["feature_binning"] = rendered.feature_binning
        if rendered.special_values is not None:
            meta["special_values"] = rendered.special_values
        if rendered.red_flags:
            # AGT-9: deterministic modeling red flags (computed in
            # gate_adapters.render_gate_dependencies straight from the tuning /
            # select-experiment dependency outputs) ride alongside the existing
            # screen/dedup metadata so auto_drive._extract_red_flags can surface
            # them in the 【平台红旗 checklist】 without re-parsing table strings.
            meta["red_flags"] = rendered.red_flags
        if rendered.presentation_warnings:
            meta["presentation_warnings"] = rendered.presentation_warnings
        # LT-3 (A.3): the gate's reply adapter declares its adjustable parameters as
        # a JSON schema; surface it on the gate payload under editable_input_schema
        # (the SAME key the LT-4 retry form already consumes from failure_envelope)
        # so the frontend gets a real schema (enum/bounds/title) for the gate's
        # controls instead of only the type-inferred gate_envelope controls. Absent
        # (adapter-less gate or nothing adjustable) -> key omitted, zero change.
        editable_schema = gate_editable_input_schema(plan, gate, self._safe_output)
        if editable_schema:
            meta["editable_input_schema"] = editable_schema
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
                text, tables = render_tool_output(
                    terminal.tool_ref.tool,
                    output,
                    trusted_task_id=plan.task_id,
                    trusted_inputs=self._trusted_terminal_inputs(
                        plan,
                        terminal,
                    ),
                    trusted_artifacts=self._trusted_terminal_artifacts(
                        plan.task_id,
                        terminal,
                        output,
                    ),
                )
                if text:
                    parts.append(text)
        meta = {"plan_id": plan.id, "run_seq": run_seq, "tables": tables}
        result_dataset_id = self.latest_result_dataset_id(plan)
        if result_dataset_id:
            meta["result_dataset"] = {
                "dataset_id": result_dataset_id,
                "download_url": (
                    f"/api/tasks/{plan.task_id}/datasets/"
                    f"{result_dataset_id}/download"
                ),
            }
        report_details = self._latest_report_details(plan)
        if report_details is not None:
            report_step, report_output = report_details
            self._attach_report_download_metadata(
                meta,
                plan=plan,
                report_step=report_step,
                report_output=report_output,
            )
        if terminal is not None and output is not None:
            if terminal.tool_ref.tool == "generate_risk_analysis_report":
                # Keep a bounded, deterministic envelope for governed memory
                # capture. Raw rows never enter conversation metadata.
                allowed = (
                    "analysis_kind",
                    "product_scope",
                    "as_of_period",
                    "report_path",
                    "headline_metrics",
                    "key_points",
                    "red_flags",
                    "assumptions",
                    "source_row_count",
                    "row_count",
                    "column_map",
                )
                meta["risk_analysis_report"] = {
                    key: output[key] for key in allowed if key in output
                }
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

    def latest_result_dataset_id(self, plan: Plan) -> str | None:
        """Return the newest materialized dataset produced by a completed plan.

        Kept public so the message API can enrich historical completion messages
        that predate the ``result_dataset`` metadata contract without rewriting
        the persisted audit transcript.
        """
        for step in sorted(plan.steps, key=lambda item: item.index, reverse=True):
            if step.status != StepStatus.DONE or not step.output_ref:
                continue
            output = self._safe_output(step.id)
            if not isinstance(output, dict):
                continue
            dataset_id = str(output.get("result_dataset_id") or "").strip()
            if dataset_id:
                return dataset_id
        return None

    def latest_report(self, plan: Plan) -> tuple[str, str] | None:
        """Return the newest generated report so completion messages can expose
        the download exactly where the result is presented, independent of rail
        polling or viewport position."""
        details = self._latest_report_details(plan)
        if details is None:
            return None
        step, output = details
        return str(step.tool_ref.tool), str(output.get("report_path") or "")

    def _latest_report_details(self, plan: Plan) -> tuple[PlanStep, dict] | None:
        for step in sorted(plan.steps, key=lambda item: item.index, reverse=True):
            if step.status != StepStatus.DONE or not step.output_ref:
                continue
            output = self._safe_output(step.id)
            report_path = str(output.get("report_path") or "") if isinstance(output, dict) else ""
            if report_path:
                return step, output
        return None

    def _attach_report_download_metadata(
        self,
        meta: dict,
        *,
        plan: Plan,
        report_step: PlanStep | None,
        report_output: dict | None,
    ) -> None:
        if report_step is None or not isinstance(report_output, dict):
            return
        label = (
            "下载特征分析报告"
            if report_step.tool_ref.tool == "generate_feature_report"
            else "下载模型开发报告"
        )
        reports = driver_report_download_metadata(
            plan_id=plan.id,
            task_id=plan.task_id,
            step_id=report_step.id,
            output=report_output,
            default_label=label,
        )
        if not reports:
            return
        if isinstance(report_output.get("reports"), list):
            meta["report_downloads"] = reports
            meta["report_download"] = reports[0]
            return
        # Preserve the original single-report endpoint and payload contract.
        meta["report_download"] = {
            "label": label,
            "download_url": f"/api/tasks/{plan.task_id}/driver-report/download",
        }

    def review_message(self, plan: Plan, *, run_seq) -> DriverMessage:
        return DriverMessage(
            "review",
            "计划已执行完，但结果需要你复核一下再定论。",
            {"plan_id": plan.id, "run_seq": run_seq},
        )

    def cancelled_message(self, plan: Plan, *, run_seq) -> DriverMessage:
        interrupted = next(
            (step for step in plan.steps if step.status == StepStatus.FAILED),
            None,
        )
        detail = (
            f"“{interrupted.title}”已停止；"
            if interrupted is not None
            else "当前执行已停止；"
        )
        return DriverMessage(
            "chat",
            detail
            + "已完成步骤、中间结果、调参检查点和最后进度均已保留。"
            + "需要恢复时，请输入“继续当前步骤”。",
            {
                "plan_id": plan.id,
                "step_id": interrupted.id if interrupted is not None else None,
                "run_seq": run_seq,
                "intent": "execution_cancelled",
                "cancelled": True,
            },
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
        diagnostic = _step_failure_diagnostic(failed, detail, error_kind)
        meta["error"] = True
        meta["error_diagnostic"] = diagnostic
        return DriverMessage(
            "error",
            f"❌ {detail}\n\n已完成步骤和中间结果均已保留。是否由 Agent 从当前失败步骤继续处理？",
            meta,
        )

    def _report_dependency_output(self, plan: Plan, step: PlanStep) -> tuple[dict | None, PlanStep | None]:
        for dep_id in step.depends_on or []:
            dep = find_step(plan, dep_id)
            if dep is None or dep.tool_ref.tool not in {
                "generate_model_report",
                "generate_model_reports",
            }:
                continue
            output = self._safe_output(dep.id)
            return (output if isinstance(output, dict) else None), dep
        return None, None

    def _trusted_terminal_inputs(
        self,
        plan: Plan,
        step: PlanStep,
    ) -> Mapping[str, Any] | None:
        """Resolve only PoolStability's exact direct ImpactCube references."""

        if step.tool_ref.tool != "measure_strategy_pool_stability":
            return step.inputs
        if not isinstance(step.inputs, Mapping):
            return None
        raw = dict(step.inputs)
        if not any(
            isinstance(value, str) and value.startswith("$ref:")
            for value in raw.values()
        ):
            return raw
        if len(step.depends_on) != 1:
            return None
        dependency = find_step(plan, step.depends_on[0])
        if (
            dependency is None
            or dependency.status != StepStatus.DONE
            or dependency.tool_ref.tool != "measure_strategy_impact_cube"
            or not dependency.output_ref
        ):
            return None
        expected_paths = {
            "artifact_id": ("artifact", "artifact_id"),
            "expected_artifact_content_hash": (
                "artifact",
                "content_hash",
            ),
            "expected_cube_id": ("cube_id",),
            "expected_cube_content_hash": ("content_hash",),
        }
        if set(raw) != set(expected_paths):
            return None
        output = self._safe_output(dependency.id)
        if not isinstance(output, Mapping):
            return None
        resolved: dict[str, str] = {}
        for field, path in expected_paths.items():
            expected_ref = (
                f"$ref:{dependency.id}.output." + ".".join(path)
            )
            if raw[field] != expected_ref:
                return None
            value: object = output
            for component in path:
                if not isinstance(value, Mapping) or component not in value:
                    return None
                value = value[component]
            if not isinstance(value, str) or not value:
                return None
            resolved[field] = value
        return resolved

    def _safe_output(self, step_id: str):
        try:
            return self._load_output(step_id)
        except KeyError:
            return None

    def _trusted_delivery_artifacts(
        self,
        task_id: str,
        step: PlanStep,
        output: object,
    ) -> dict[str, dict] | None:
        if (
            step.tool_ref.tool != "export_strategy_delivery"
            or self._load_task_artifact is None
            or not isinstance(output, Mapping)
        ):
            return None
        artifacts = output.get("artifacts")
        names = ("python", "sql", "strategy_json", "equivalence_json")
        if not isinstance(artifacts, list) or len(artifacts) != len(names):
            return None
        trusted: dict[str, dict] = {}
        for name, artifact in zip(names, artifacts, strict=True):
            if not isinstance(artifact, Mapping):
                return None
            artifact_id = artifact.get("artifact_id")
            if not isinstance(artifact_id, str) or not artifact_id:
                return None
            try:
                record = self._load_task_artifact(task_id, artifact_id)
            except (KeyError, TypeError, ValueError):
                return None
            if not isinstance(record, Mapping):
                return None
            trusted[name] = dict(record)
        return trusted

    def _trusted_terminal_artifacts(
        self,
        task_id: str,
        step: PlanStep,
        output: object,
    ) -> dict[str, dict] | None:
        if step.tool_ref.tool == "measure_candidate_monthly_stability":
            return self._trusted_candidate_stability_artifact(
                task_id,
                output,
            )
        if step.tool_ref.tool == "measure_strategy_pool_validation":
            return self._trusted_pool_validation_artifact(
                task_id,
                output,
            )
        if step.tool_ref.tool == "measure_strategy_pool_stability":
            return self._trusted_pool_stability_artifact(
                task_id,
                output,
            )
        return self._trusted_delivery_artifacts(task_id, step, output)

    def _trusted_pool_stability_artifact(
        self,
        task_id: str,
        output: object,
    ) -> dict[str, dict] | None:
        if (
            self._load_task_artifact is None
            or self._tasks_root is None
            or self._db_path is None
            or not isinstance(output, Mapping)
        ):
            return None
        artifact = output.get("artifact")
        if not isinstance(artifact, Mapping):
            return None
        artifact_id = artifact.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            return None
        try:
            record = self._load_task_artifact(task_id, artifact_id)
        except (KeyError, TypeError, ValueError):
            return None
        if not isinstance(record, Mapping):
            return None
        return {
            "pool_stability": {
                "record": dict(record),
                "tasks_root": str(self._tasks_root),
                "db_path": str(self._db_path),
            }
        }

    def _trusted_pool_validation_artifact(
        self,
        task_id: str,
        output: object,
    ) -> dict[str, dict] | None:
        if (
            self._load_task_artifact is None
            or self._tasks_root is None
            or not isinstance(output, Mapping)
        ):
            return None
        artifact = output.get("artifact")
        if not isinstance(artifact, Mapping):
            return None
        artifact_id = artifact.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            return None
        try:
            record = self._load_task_artifact(task_id, artifact_id)
        except (KeyError, TypeError, ValueError):
            return None
        if not isinstance(record, Mapping):
            return None
        return {
            "pool_validation": {
                "record": dict(record),
                "tasks_root": str(self._tasks_root),
            }
        }

    def _trusted_candidate_stability_artifact(
        self,
        task_id: str,
        output: object,
    ) -> dict[str, dict] | None:
        if self._load_task_artifact is None or not isinstance(output, Mapping):
            return None
        artifacts = output.get("artifacts")
        if not isinstance(artifacts, list) or len(artifacts) != 1:
            return None
        artifact = artifacts[0]
        if not isinstance(artifact, Mapping):
            return None
        artifact_id = artifact.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            return None
        try:
            record = self._load_task_artifact(task_id, artifact_id)
        except (KeyError, TypeError, ValueError):
            return None
        if not isinstance(record, Mapping):
            return None
        return {"stability": dict(record)}


__all__ = ["PlanMessageComposer"]


def _step_failure_diagnostic(failed: PlanStep | None, detail: str, error_kind: str) -> dict:
    raw_error = str(failed.error or "") if failed is not None else ""
    lowered = raw_error.lower()
    if "'type' object is not subscriptable" in lowered:
        cause = (
            "工具进程使用的旧版 Python 在导入新式类型注解时发生兼容性错误；"
            "这不是数据内容或拼接规则本身的问题。"
        )
    else:
        cause = raw_error or "当前步骤返回了执行错误，后续依赖步骤已暂停。"
    step_title = failed.title if failed is not None else "当前步骤"
    diagnostic = {
        "schema_version": "workflow_error.v1",
        "workflow": "plan_driver",
        "code": "workflow_step_failed",
        "phase": "execution",
        "title": "步骤执行失败",
        "summary": f"「{step_title}」未完成；已完成步骤和中间结果均已保留。",
        "cause": cause,
        "location": step_title,
        "evidence": [{"label": "错误类型", "value": str(error_kind or "execution")}],
        "actions": [
            "由 Agent 从当前失败步骤重试，不会从头重跑已完成步骤。",
            "如输入参数需要调整，可在中间信息流的失败步骤卡片中编辑后重试。",
            "若重试仍失败，保留现有证据并让 Agent 重新规划后续步骤。",
        ],
        "agent_prompt": "是否由 Agent 从当前失败步骤重试？",
        "recovery_actions": [
            {"label": "由 Agent 重试当前步骤", "command": "重试当前步骤"},
        ],
        "technical_detail": detail,
        "retryable": failed is not None,
        "impact": "失败步骤之后的依赖步骤尚未执行。",
    }
    from marvis.agent.workflow_error_diagnostics import (  # noqa: PLC0415
        enrich_workflow_error_diagnostic,
    )

    return enrich_workflow_error_diagnostic(diagnostic)
