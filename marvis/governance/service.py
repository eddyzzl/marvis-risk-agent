from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from marvis.governance.contracts import (
    AuthorizationBinding,
    AuthorizationGrant,
    ExecutionContext,
    LocalPrincipal,
)
from marvis.governance.errors import ApprovalBindingError, ApprovalStateError
from marvis.governance.repository import GovernanceRepository, canonical_payload_hash
from marvis.orchestrator.contracts import Plan, PlanStep
from marvis.plugins.manifest import (
    GovernancePolicy,
    PluginManifest,
    ToolRef,
    ToolSpec,
    governance_policy_hash,
    manifest_to_dict,
    merge_governance_policies,
)
from marvis.strategy_adoption import normalize_adoption_reason


class GovernanceService:
    """Build and verify the live authorization surface for governed plan steps.

    Raw inputs and evidence stay in their owning repositories.  The governance
    ledger receives only canonical hashes plus the minimum target lifecycle
    snapshot needed to fence a protected side effect.
    """

    def __init__(
        self,
        *,
        plan_repo,
        tool_registry,
        strategy_repo,
        governance_repo: GovernanceRepository,
    ) -> None:
        self._plans = plan_repo
        self._tools = tool_registry
        self._strategies = strategy_repo
        self._governance = governance_repo

    def requires_human_decision(self, step: PlanStep) -> bool:
        _manifest, tool = self._tools.resolve_with_manifest(step.tool_ref)
        policy = self._effective_policy(step, tool)
        return policy.human_decision_gate == "required"

    def authorize_step(
        self,
        *,
        plan_id: str,
        step_id: str,
        principal: LocalPrincipal,
        reason: str,
        expected_plan_revision: int,
        input_updates: dict[str, Any] | None = None,
    ) -> AuthorizationGrant:
        plan, step = self._load_step(plan_id, step_id)
        self._assert_revision(plan, expected_plan_revision)
        manifest, tool = self._tools.resolve_with_manifest(step.tool_ref)
        policy = self._effective_policy(step, tool)
        if policy.human_decision_gate != "required":
            raise ApprovalStateError(
                f"step {step.id} does not require a governed human decision"
            )
        reviewed_updates = dict(input_updates or {})
        if step.tool_ref.label() == "strategy.adopt_strategy":
            reviewed_updates["adoption_reason"] = normalize_adoption_reason(
                reviewed_updates.get("adoption_reason") or reason
            )
        merged_inputs = {**dict(step.inputs), **reviewed_updates}
        resolved_inputs = self._resolve_inputs(merged_inputs)
        binding = self._binding(
            plan=plan,
            step=step,
            inputs=resolved_inputs,
            manifest=manifest,
            tool=tool,
            policy=policy,
        )
        return self._governance.authorize_step(
            binding,
            principal=principal,
            reason=reason,
            issue_effect_approval=policy.effect_authorization == "required",
            input_updates=reviewed_updates or None,
            expected_input_hash=canonical_payload_hash(merged_inputs),
        )

    def reject_step(
        self,
        *,
        plan_id: str,
        step_id: str,
        principal: LocalPrincipal,
        reason: str,
        expected_plan_revision: int,
    ) -> AuthorizationGrant:
        plan, step = self._load_step(plan_id, step_id)
        self._assert_revision(plan, expected_plan_revision)
        manifest, tool = self._tools.resolve_with_manifest(step.tool_ref)
        policy = self._effective_policy(step, tool)
        if policy.human_decision_gate != "required":
            raise ApprovalStateError(
                f"step {step.id} does not require a governed human decision"
            )
        binding = self._binding(
            plan=plan,
            step=step,
            inputs=self._resolve_inputs(step.inputs),
            manifest=manifest,
            tool=tool,
            policy=policy,
        )
        return self._governance.record_decision(
            binding,
            principal=principal,
            decision="reject",
            reason=reason,
        )

    def execution_context_for(
        self,
        *,
        plan: Plan,
        step: PlanStep,
        inputs: dict[str, Any],
    ) -> ExecutionContext | None:
        """Resolve proof for the exact live step binding used by PlanExecutor."""

        live_plan, live_step = self._load_step(plan.id, step.id)
        if live_plan.task_id != plan.task_id:
            raise ApprovalBindingError("authorization binding mismatch: task_id")
        if int(live_plan.replan_count) != int(plan.replan_count):
            raise ApprovalBindingError("authorization binding mismatch: plan_revision")
        if live_step.tool_ref.label() != step.tool_ref.label():
            raise ApprovalBindingError("authorization binding mismatch: tool_ref")
        manifest, tool = self._tools.resolve_with_manifest(live_step.tool_ref)
        policy = self._effective_policy(live_step, tool)
        if (
            policy.human_decision_gate != "required"
            and policy.effect_authorization != "required"
        ):
            raise ApprovalBindingError(
                f"plan step {live_step.id} does not require governed execution"
            )
        resolved_inputs = self._validated_live_inputs(live_step, inputs)
        binding = self._binding(
            plan=live_plan,
            step=live_step,
            inputs=resolved_inputs,
            manifest=manifest,
            tool=tool,
            policy=policy,
        )
        return self._governance.execution_context_for_binding(binding)

    def resolve_binding(
        self,
        *,
        task_id: str,
        ref: ToolRef,
        inputs: dict[str, Any],
        execution_context: ExecutionContext,
        manifest: PluginManifest,
        tool: ToolSpec,
    ) -> AuthorizationBinding:
        """Runner callback: rebuild every binding field from live platform state."""

        plan, step = self._load_step(
            str(execution_context.plan_id),
            str(execution_context.step_id),
        )
        if plan.task_id != str(task_id):
            raise ApprovalBindingError("authorization binding mismatch: task_id")
        if int(plan.replan_count) != int(execution_context.plan_revision):
            raise ApprovalBindingError("authorization binding mismatch: plan_revision")
        if step.tool_ref.label() != ref.label():
            raise ApprovalBindingError("authorization binding mismatch: tool_ref")
        resolved_manifest, resolved_tool = self._tools.resolve_with_manifest(step.tool_ref)
        if resolved_manifest != manifest or resolved_tool != tool:
            raise ApprovalBindingError("authorization binding mismatch: live manifest")
        policy = self._effective_policy(step, tool)
        human_required = policy.human_decision_gate == "required"
        effect_required = policy.effect_authorization == "required"
        if not human_required and not effect_required:
            raise ApprovalBindingError("step does not require governed execution")
        if bool(getattr(execution_context, "human_decision_required", False)) != (
            human_required or effect_required
        ):
            raise ApprovalBindingError(
                "authorization binding mismatch: human_decision_required"
            )
        if bool(
            getattr(execution_context, "effect_authorization_required", False)
        ) != effect_required:
            raise ApprovalBindingError(
                "authorization binding mismatch: effect_authorization_required"
            )
        resolved_inputs = self._validated_live_inputs(step, inputs)
        return self._binding(
            plan=plan,
            step=step,
            inputs=resolved_inputs,
            manifest=manifest,
            tool=tool,
            policy=policy,
        )

    def _binding(
        self,
        *,
        plan: Plan,
        step: PlanStep,
        inputs: dict[str, Any],
        manifest: PluginManifest,
        tool: ToolSpec,
        policy: GovernancePolicy,
    ) -> AuthorizationBinding:
        return AuthorizationBinding(
            task_id=plan.task_id,
            plan_id=plan.id,
            plan_revision=int(plan.replan_count),
            step_id=step.id,
            tool_ref=f"{step.tool_ref.label()}@{manifest.version}",
            manifest_hash=_manifest_hash(manifest),
            policy_hash=governance_policy_hash(policy),
            input_hash=_governance_payload_hash(inputs),
            evidence_hash=self._evidence_hash(step),
            effect_target=self._effect_target(plan, policy, inputs),
        )

    def _effective_policy(
        self,
        step: PlanStep,
        tool: ToolSpec,
    ) -> GovernancePolicy:
        effective = merge_governance_policies(tool.policy, step.policy)
        if effective != step.policy:
            raise ApprovalBindingError(
                f"plan step {step.id} policy is weaker than the live tool manifest"
            )
        if effective.human_decision_gate == "required" and not step.needs_confirmation:
            raise ApprovalBindingError(
                f"plan step {step.id} removed its mandatory confirmation gate"
            )
        return effective

    def _effect_target(
        self,
        plan: Plan,
        policy: GovernancePolicy,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        target_policy = policy.effect_target
        if policy.effect_authorization != "required":
            return {}
        if target_policy is None:
            raise ApprovalBindingError("effect target policy is missing")
        if target_policy.kind != "strategy":
            raise ApprovalBindingError(
                f"unsupported effect target kind: {target_policy.kind}"
            )
        target_id = str(inputs.get(target_policy.id_input) or "").strip()
        if not target_id:
            raise ApprovalBindingError(
                f"effect target input is missing: {target_policy.id_input}"
            )
        meta = self._strategies.get_strategy_meta(target_id)
        if meta is None:
            raise ApprovalBindingError(f"strategy target not found: {target_id}")
        target_task_id = str(meta.get("task_id") or "")
        if target_task_id != plan.task_id:
            raise ApprovalBindingError("strategy target belongs to another task")
        status = str(meta.get("status") or "")
        if status not in target_policy.expected_statuses:
            raise ApprovalBindingError(
                f"strategy target status is {status}, expected one of "
                f"{list(target_policy.expected_statuses)}"
            )
        strategy_type = str(meta.get("strategy_type") or "")
        champion_ids = sorted(
            str(item["id"])
            for item in self._strategies.list_meta_for_task(target_task_id)
            if str(item.get("strategy_type") or "") == strategy_type
            and str(item.get("status") or "") == "adopted"
        )
        strategy_spec_hash = self._strategies.get_strategy_spec_hash(target_id)
        if not strategy_spec_hash:
            raise ApprovalBindingError(
                f"strategy target definition not found: {target_id}"
            )
        return {
            "kind": "strategy",
            "id": target_id,
            "expected_status": status,
            "result_status": target_policy.result_status,
            "version": int(meta.get("version") or 0),
            "task_id": target_task_id,
            "strategy_type": strategy_type,
            "strategy_spec_hash": strategy_spec_hash,
            "current_champion_ids": champion_ids,
        }

    def _evidence_hash(self, step: PlanStep) -> str:
        dependencies: list[dict[str, Any]] = []
        for dependency_id in sorted(str(item) for item in step.depends_on):
            try:
                output_ref = self._plans.latest_step_output_ref(dependency_id)
                evidence = self._plans.load_step_evidence(dependency_id)
            except KeyError as exc:
                raise ApprovalBindingError(
                    f"dependency evidence is missing: {dependency_id}"
                ) from exc
            dependencies.append(
                {
                    "step_id": dependency_id,
                    "output_ref": output_ref,
                    "evidence_hash": _governance_payload_hash(evidence),
                }
            )
        return _governance_payload_hash(dependencies)

    def _resolve_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {
            str(key): self._resolve_value(value)
            for key, value in dict(inputs).items()
        }

    def _validated_live_inputs(
        self,
        step: PlanStep,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        resolved_inputs = self._resolve_inputs(step.inputs)
        if _governance_payload_hash(resolved_inputs) != _governance_payload_hash(inputs):
            raise ApprovalBindingError(
                "authorization binding mismatch: resolved inputs"
            )
        return resolved_inputs

    def _resolve_value(self, value: Any) -> Any:
        if isinstance(value, str) and value.startswith("$ref:"):
            step_id, field = _parse_ref(value)
            try:
                output = self._plans.load_step_output(step_id)
            except KeyError as exc:
                raise ApprovalBindingError(
                    f"referenced output is missing: {step_id}"
                ) from exc
            return _dig(output, field, ref=value) if field else output
        if isinstance(value, list):
            return [self._resolve_value(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self._resolve_value(item) for key, item in value.items()}
        return value

    def _load_step(self, plan_id: str, step_id: str) -> tuple[Plan, PlanStep]:
        plan = self._plans.load_plan(plan_id)
        step = next((item for item in plan.steps if item.id == step_id), None)
        if step is None:
            raise ApprovalBindingError(f"plan step not found: {step_id}")
        return plan, step

    @staticmethod
    def _assert_revision(plan: Plan, expected_plan_revision: int) -> None:
        if int(plan.replan_count) != int(expected_plan_revision):
            raise ApprovalBindingError(
                "plan revision changed; refresh the decision before approving"
            )


def _manifest_hash(manifest: PluginManifest) -> str:
    checksum = str(manifest.checksum or "").strip()
    if checksum:
        return checksum if checksum.startswith("sha256:") else f"sha256:{checksum}"
    encoded = json.dumps(
        manifest_to_dict(manifest),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _governance_payload_hash(payload: Any) -> str:
    """Hash JSON-like payloads with collision-safe non-finite float tags.

    Strategy evidence legitimately uses open ``+/-inf`` band endpoints. JSON
    cannot encode those values canonically, so every value is first represented
    as a recursively typed tree. A user string or object that resembles a float
    tag is therefore encoded as its own string/object type and cannot collide
    with the corresponding numeric value.
    """

    return canonical_payload_hash(
        ["marvis-governance-canonical.v1", _typed_canonical_value(payload)]
    )


def _typed_canonical_value(value: Any) -> list[Any]:
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", value]
    if isinstance(value, float):
        if math.isnan(value):
            return ["float", "nan"]
        if math.isinf(value):
            return ["float", "+inf" if value > 0 else "-inf"]
        return ["float", value]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, dict):
        entries = [
            [_typed_canonical_value(key), _typed_canonical_value(item)]
            for key, item in value.items()
        ]
        entries.sort(key=lambda item: _canonical_sort_key(item[0]))
        return ["dict", entries]
    if isinstance(value, (list, tuple)):
        return ["list", [_typed_canonical_value(item) for item in value]]
    raise TypeError(
        f"governance payload contains unsupported type: {type(value).__name__}"
    )


def _canonical_sort_key(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _parse_ref(value: str) -> tuple[str, str]:
    raw = value[len("$ref:") :]
    marker = ".output"
    if marker not in raw:
        raise ApprovalBindingError(f"invalid output reference: {value}")
    step_id, tail = raw.split(marker, 1)
    if not step_id:
        raise ApprovalBindingError(f"invalid output reference: {value}")
    if not tail:
        return step_id, ""
    if not tail.startswith(".") or tail == ".":
        raise ApprovalBindingError(f"invalid output reference: {value}")
    return step_id, tail[1:]


def _dig(value: Any, path: str, *, ref: str) -> Any:
    current = value
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        if isinstance(current, (list, tuple)) and part.isdigit():
            index = int(part)
            if 0 <= index < len(current):
                current = current[index]
                continue
        raise ApprovalBindingError(f"reference path is missing: {ref}")
    return current


__all__ = ["GovernanceService"]
