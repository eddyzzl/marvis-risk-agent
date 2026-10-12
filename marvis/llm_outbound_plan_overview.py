"""A live validated plan capability for AUTO's pre-execution review."""

from dataclasses import dataclass
import hashlib
import json
import re

from marvis.llm_outbound import (
    OutboundContextError,
    _DROP,
    _gate_parameters,
    _identifier,
    structural_summary,
)

_TOKEN = object()
MAX_OVERVIEW_STEPS = 2000


def _modeling_recipe_controls(inputs):
    """Keep native per-recipe budgets and field sets without free map text."""
    from marvis.agent.modeling_setup import supported_modeling_recipes

    recipes = set(supported_modeling_recipes())
    out = {}
    for name in ("n_trials_by_recipe", "features_by_recipe"):
        if name not in inputs:
            continue
        value = inputs[name]
        if isinstance(value, str) and value.startswith("$ref:"):
            projected = structural_summary({name: value})
            if projected.get(name) != value:
                raise OutboundContextError("计划算法控件引用不符合结构契约，未发送请求。")
            out[name] = value
            continue
        if not isinstance(value, dict) or any(key not in recipes for key in value):
            raise OutboundContextError("计划算法控件不属于已安装规格，未发送请求。")
        if name == "n_trials_by_recipe":
            if any(
                type(item) is not int or not 0 <= item <= 10**15
                for item in value.values()
            ):
                raise OutboundContextError("计划调参预算无效，未截断或发送请求。")
            out[name] = dict(value)
        else:
            from marvis.llm_outbound_instructions import column_identifiers

            out[name] = {
                recipe: column_identifiers(fields) for recipe, fields in value.items()
            }
    return out


def _tool_policy(policy):
    payload = policy.to_dict()
    target = payload.get("effect_target")
    if target is not None:
        strings = [
            target.get("kind"),
            target.get("id_input"),
            target.get("result_status"),
        ]
        statuses = target.get("expected_statuses")
        if not isinstance(statuses, list) or any(
            _identifier(value) is _DROP for value in [*strings, *statuses]
        ):
            raise OutboundContextError("执行工具治理标识不符合结构契约，未发送请求。")
    return payload


@dataclass(frozen=True)
class ApprovedPlanOverview:
    binding_json: str
    _token: object

    def payload(self):
        if self._token is not _TOKEN:
            raise OutboundContextError("执行计划概览来源无效，未发送请求。")
        return json.loads(self.binding_json)


def approved_plan_overview(repository, validator, *, task_id, gate_metadata):
    """Reload the actual owner-bound plan and match the displayed CAS snapshot.

    Neither caller dictionaries nor prose copied from a plan display establish
    this capability. Complete inputs remain bound by their digest and the plan
    fingerprint; only structural control values are exposed to the model.
    """
    from marvis.repositories.plans import PlanRepository
    from marvis.orchestrator.validator import PlanValidator
    from marvis.orchestrator.contracts import PlanStatus, plan_fingerprint
    from marvis.llm_outbound_instructions import column_identifiers

    if (
        not isinstance(repository, PlanRepository)
        or not isinstance(validator, PlanValidator)
        or _identifier(task_id, opaque=True) is _DROP
    ):
        raise OutboundContextError(
            "执行计划概览需要当前任务的已验证计划来源，未发送请求。"
        )
    if (
        not isinstance(gate_metadata, dict)
        or gate_metadata.get("kind") != "plan_overview"
    ):
        raise OutboundContextError("当前节点不是执行计划概览，未发送请求。")
    try:
        plan = repository.load_plan(gate_metadata.get("plan_id"))
    except (KeyError, TypeError, ValueError) as exc:
        raise OutboundContextError("当前任务执行计划不可用，未发送请求。") from exc
    snapshot = gate_metadata.get("confirmation_snapshot")
    expected = {
        "expected_plan_status": plan.status.value,
        "expected_plan_revision": plan.replan_count,
        "expected_plan_fingerprint": plan_fingerprint(plan),
    }
    if (
        plan.task_id != task_id
        or plan.status != PlanStatus.VALIDATED
        or snapshot != expected
    ):
        raise OutboundContextError("执行计划已变化或不属于当前任务，未发送请求。")
    if (
        not plan.steps
        or len(plan.steps) > MAX_OVERVIEW_STEPS
        or validator.validate_problems(plan)
    ):
        raise OutboundContextError(
            "执行计划未通过当前工具、参数或流程校验，未发送请求。"
        )
    steps = []
    for step in plan.steps:
        if _identifier(step.id, opaque=True) is _DROP:
            raise OutboundContextError("执行步骤身份无效，未发送请求。")
        manifest, tool = validator._tools.resolve_with_manifest(step.tool_ref)
        if any(
            _identifier(value) is _DROP for value in (manifest.name, tool.name)
        ) or not re.fullmatch(r"\d{1,4}\.\d{1,4}\.\d{1,4}", manifest.version):
            raise OutboundContextError("执行工具身份无效，未发送请求。")
        encoded = json.dumps(
            step.inputs,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        controls = _gate_parameters(
            [
                {
                    "name": name,
                    "type": (
                        tool.input_schema.get("properties", {}).get(name) or {}
                    ).get("type"),
                    "current": value,
                }
                for name, value in step.inputs.items()
            ]
        )
        projected_inputs = structural_summary(step.inputs)
        for key in (
            "target_col",
            "split_col",
            "time_col",
            "date_col",
            "group_col",
            "sample_weight_col",
            "oot_by_time",
        ):
            if key not in step.inputs:
                continue
            value = step.inputs[key]
            if value == "":
                projected_inputs[key] = ""
            elif (
                isinstance(value, str)
                and (value.startswith("$ref:") or (value.startswith("{") and value.endswith("}")))
            ):
                if projected_inputs.get(key) != value:
                    raise OutboundContextError(
                        "计划字段引用不符合结构契约，未发送请求。"
                    )
            else:
                projected_inputs[key] = column_identifiers([value])[0]
        for key in ("features", "feature_cols", "group_cols", "selection"):
            if isinstance(step.inputs.get(key), list):
                projected_inputs[key] = column_identifiers(step.inputs[key])
        if manifest.name == "modeling":
            projected_inputs.update(_modeling_recipe_controls(step.inputs))
        steps.append(
            {
                "id": step.id,
                "index": step.index,
                "tool": {
                    "plugin": manifest.name,
                    "tool": tool.name,
                    "version": manifest.version,
                },
                "depends_on": list(step.depends_on),
                "status": step.status.value,
                "needs_confirmation": step.needs_confirmation,
                "decision_point": step.decision_point,
                "policy": _tool_policy(tool.policy),
                "inputs": projected_inputs,
                "declared_controls": controls,
                "complete_inputs_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
            }
        )
    payload = {
        "purpose": "review_validated_plan_before_execution",
        "application_contract": (
            "当前节点尚未执行任何工具，因此不存在待审阅的计算结果。请审阅当前平台已校验的完整步骤顺序、"
            "依赖与结构控件，决定是否启动该计划；不要求补造结果。confirm 只启动当前已绑定计划，"
            "每个后续节点仍按其独立控件和人工决策/副作用授权策略执行，不授权选择候选或交付。"
            "未发送的文件内容、自由文本或私有控件不是模型可以补全的证据。"
        ),
        "plan_id": plan.id,
        "task_id": plan.task_id,
        "status": plan.status.value,
        "revision": plan.replan_count,
        "plan_fingerprint": expected["expected_plan_fingerprint"],
        "step_count": len(steps),
        "steps": steps,
    }
    return ApprovedPlanOverview(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False), _TOKEN
    )
