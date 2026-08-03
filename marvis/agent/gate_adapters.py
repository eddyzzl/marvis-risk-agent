"""Gate rendering adapters for PlanDriver.

This module bridges completed dependency outputs to the metadata/content needed
by an interactive gate. It deliberately has no repository or executor
dependency: callers provide a ``load_output(step_id)`` callback.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from marvis.agent.gate_payloads import (
    build_dedup_payload,
    build_feature_binning_payload,
    build_join_keys_payload,
    build_model_delivery_payload,
    build_modeling_setup_payload,
    build_screen_payload,
    build_special_value_payload,
)
from marvis.agent.modeling_red_flags import select_experiment_red_flags, tuning_setup_red_flags
from marvis.agent.renderers import render_tool_output
from marvis.orchestrator.contracts import Plan, PlanStep


@dataclass
class GateRenderResult:
    parts: list[str] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)
    output_refs: dict[str, str] = field(default_factory=dict)
    screen: dict | None = None
    dedup: dict | None = None
    join_keys: dict | None = None
    modeling_setup: dict | None = None
    model_delivery: dict | None = None
    feature_binning: dict | None = None
    special_values: dict | None = None
    monitoring_level: str | None = None
    # AGT-9: deterministic red flags for the tuning-config / select-experiment
    # gates, computed straight from those gates' dependency outputs (never from
    # the rendered table strings). Empty list when nothing tripped, or when
    # neither red-flag family's inputs are present at this gate.
    red_flags: list[str] = field(default_factory=list)
    presentation_warnings: list[dict] = field(default_factory=list)


def render_gate_dependencies(
    plan: Plan,
    gate: PlanStep | None,
    load_output: Callable[[str], Any],
) -> GateRenderResult:
    result = GateRenderResult()
    confirm_join_o: dict | None = None
    propose_join_o: dict | None = None
    split_o: dict | None = None
    split_step: PlanStep | None = None
    modeling_spec_o: dict | None = None
    modeling_spec_step: PlanStep | None = None
    select_features_o: dict | None = None
    model_delivery_o: dict | None = None
    model_delivery_step: PlanStep | None = None
    report_o: dict | None = None
    report_step: PlanStep | None = None
    tune_o: dict | None = None
    train_models_o: dict | None = None
    for dep_id in gate.depends_on if gate else []:
        dep = _find_step(plan, dep_id)
        if dep is None:
            continue
        if dep.output_ref:
            result.output_refs[dep.id] = dep.output_ref
        output = load_output(dep_id)
        if output is None:
            continue
        try:
            text, dep_tables = render_tool_output(
                dep.tool_ref.tool,
                output,
                presentation_state=_dependency_presentation_state(gate, dep),
            )
        except Exception as exc:  # presentation must never rewrite execution truth
            result.presentation_warnings.append({
                "step_id": dep.id,
                "step_title": dep.title,
                "error": f"{exc.__class__.__name__}: {exc}",
            })
            result.parts.append(
                f"「{dep.title}」结果已经生成，但结构化展示暂时失败；"
                "执行结果和后续进度已保留，可继续处理或稍后重新打开。"
            )
            continue
        if text:
            result.parts.append(text)
        result.tables.extend(dep_tables)
        if dep.tool_ref.tool == "screen_features":
            if (
                gate is not None
                and gate.tool_ref is not None
                and gate.tool_ref.tool == "resolve_special_values"
            ):
                result.special_values = build_special_value_payload(output, gate)
            else:
                result.screen = build_screen_payload(output, dep)
        elif dep.tool_ref.tool == "confirm_join":
            confirm_join_o = output
        elif dep.tool_ref.tool == "propose_join":
            propose_join_o = output
        elif dep.tool_ref.tool == "make_split" and isinstance(output, dict):
            split_o = output
            split_step = dep
        elif dep.tool_ref.tool == "choose_modeling_spec":
            modeling_spec_o = output if isinstance(output, dict) else None
            modeling_spec_step = dep
        elif dep.tool_ref.tool == "select_features" and isinstance(output, dict):
            select_features_o = output
        elif dep.tool_ref.tool in {"compare_experiments", "select_experiment", "post_training_action"}:
            model_delivery_o = output if isinstance(output, dict) else None
            model_delivery_step = dep
        elif dep.tool_ref.tool in {"generate_model_report", "generate_model_reports"}:
            report_o = output if isinstance(output, dict) else None
            report_step = dep
        elif dep.tool_ref.tool == "tune_hyperparameters" and isinstance(output, dict):
            tune_o = output
        elif dep.tool_ref.tool == "train_models" and isinstance(output, dict):
            train_models_o = output
        if (
            gate is not None
            and gate.tool_ref.tool == "apply_monitoring_disposition"
            and dep.tool_ref.tool == "run_strategy_monitoring"
            and isinstance(output, dict)
        ):
            monitoring_level = str(output.get("overall_level") or "").strip().lower()
            if monitoring_level in {"green", "amber", "red"}:
                result.monitoring_level = monitoring_level
        if (
            gate is not None
            and gate.tool_ref.tool == "analyze_feature_bins"
            and dep.tool_ref.tool == "compute_feature_metrics"
        ):
            result.feature_binning = build_feature_binning_payload(output, gate)
    if modeling_spec_o is not None and modeling_spec_step is not None:
        displayed_modeling_spec = _with_completed_tuning_budget(
            _with_pending_tuning_budget(modeling_spec_o, gate),
            tune_o,
        )
        effective_refined_output = (
            select_features_o
            if select_features_o is not None
            else _refined_features_from_model_delivery(model_delivery_o)
        )
        result.modeling_setup = build_modeling_setup_payload(
            displayed_modeling_spec,
            modeling_spec_step,
            split_output=split_o,
            split_step=split_step,
            refined_output=effective_refined_output,
        )
        _rewrite_modeling_spec_feature_count(
            result.tables,
            effective_refined_output,
        )
        _rewrite_pending_tuning_budget(
            result.tables,
            displayed_modeling_spec,
            gate,
            completed=tune_o is not None,
        )
    if model_delivery_o is not None and model_delivery_step is not None:
        result.model_delivery = build_model_delivery_payload(
            model_delivery_o,
            model_delivery_step,
            report_output=report_o,
            report_step=report_step,
        )
    result.dedup = build_dedup_payload(confirm_join_o, propose_join_o)
    result.join_keys = build_join_keys_payload(propose_join_o)
    result.red_flags = [
        *tuning_setup_red_flags(
            split_output=split_o,
            modeling_spec_output=_effective_modeling_spec(
                modeling_spec_o,
                select_features_o,
            ),
        ),
        *select_experiment_red_flags(tune_output=tune_o, train_models_output=train_models_o),
    ]
    return result


def _effective_modeling_spec(
    modeling_spec: dict | None,
    refined_output: dict | None,
) -> dict | None:
    if not isinstance(modeling_spec, dict):
        return modeling_spec
    selected = (
        refined_output.get("selected")
        if isinstance(refined_output, dict)
        and isinstance(refined_output.get("selected"), list)
        else None
    )
    if selected is None:
        return modeling_spec
    return {**modeling_spec, "feature_count": len(selected)}


def _rewrite_modeling_spec_feature_count(
    tables: list[dict],
    refined_output: dict | None,
) -> None:
    selected = (
        refined_output.get("selected")
        if isinstance(refined_output, dict)
        and isinstance(refined_output.get("selected"), list)
        else None
    )
    if selected is None:
        return
    for table in tables:
        if table.get("title") != "建模规格":
            continue
        rows = table.get("rows") if isinstance(table.get("rows"), list) else []
        for row in rows:
            if isinstance(row, list) and row and row[0] == "候选特征数":
                row[:] = ["精选后特征数", str(len(selected))]


def _with_pending_tuning_budget(output: dict, gate: PlanStep | None) -> dict:
    if gate is None or gate.tool_ref.tool != "configure_tuning":
        return output
    raw_budgets = (gate.inputs or {}).get("n_trials_by_recipe")
    if not isinstance(raw_budgets, dict) or not raw_budgets:
        return output
    budgets = {
        str(recipe): int(count)
        for recipe, count in raw_budgets.items()
        if str(recipe).strip()
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count > 0
    }
    if not budgets:
        return output
    recipes = [str(item) for item in (output.get("recipes") or []) if str(item)]
    if not recipes:
        recipes = list(budgets)
    primary = str(output.get("recipe") or recipes[0])
    scalar_budget = budgets.get(primary, next(iter(budgets.values())))
    budget_note = "、".join(f"{recipe}={budgets[recipe]}" for recipe in recipes if recipe in budgets)
    total = sum(budgets.get(recipe, 0) for recipe in recipes)
    return {
        **output,
        "recipe": primary,
        "recipes": recipes,
        "n_trials": scalar_budget,
        "n_trials_by_recipe": budgets,
        "reason": f"本次确认的调参预算为 {budget_note}，总计 {total} 轮。",
    }


def _with_completed_tuning_budget(output: dict, tune_output: dict | None) -> dict:
    if not isinstance(tune_output, dict):
        return output
    per_recipe = tune_output.get("per_recipe")
    if not isinstance(per_recipe, dict):
        return output
    budgets: dict[str, int] = {}
    for recipe, detail in per_recipe.items():
        if not isinstance(detail, dict):
            continue
        count = detail.get("n_trials")
        if isinstance(count, int) and not isinstance(count, bool) and count > 0:
            budgets[str(recipe)] = count
    if not budgets:
        return output
    recipes = [str(item) for item in (output.get("recipes") or []) if str(item)]
    if not recipes:
        recipes = list(budgets)
    primary = str(output.get("recipe") or recipes[0])
    scalar_budget = budgets.get(primary, next(iter(budgets.values())))
    total = sum(budgets.get(recipe, 0) for recipe in recipes)
    budget_note = "、".join(
        f"{recipe}={budgets[recipe]}" for recipe in recipes if recipe in budgets
    )
    return {
        **output,
        "recipe": primary,
        "recipes": recipes,
        "n_trials": scalar_budget,
        "n_trials_by_recipe": budgets,
        "reason": f"本轮实际完成的调参预算为 {budget_note}，总计 {total} 轮。",
    }


def _refined_features_from_model_delivery(output: dict | None) -> dict | None:
    if not isinstance(output, dict):
        return None
    for experiment in output.get("experiments") or []:
        if not isinstance(experiment, dict):
            continue
        raw_features = experiment.get("feature_list") or experiment.get("features")
        if not isinstance(raw_features, list) or not raw_features:
            continue
        selected = list(dict.fromkeys(str(item) for item in raw_features if str(item)))
        if selected:
            return {"selected": selected}
    return None


def _rewrite_pending_tuning_budget(
    tables: list[dict],
    displayed_output: dict,
    gate: PlanStep | None,
    *,
    completed: bool = False,
) -> None:
    if not completed and (gate is None or gate.tool_ref.tool != "configure_tuning"):
        return
    budgets = displayed_output.get("n_trials_by_recipe")
    recipes = [str(item) for item in (displayed_output.get("recipes") or []) if str(item)]
    if not isinstance(budgets, dict) or not budgets:
        return
    ordered = [(recipe, budgets[recipe]) for recipe in recipes if recipe in budgets]
    if not ordered:
        ordered = [(str(recipe), count) for recipe, count in budgets.items()]
    display = "、".join(f"{recipe}={count}" for recipe, count in ordered)
    display += f"（总计 {sum(int(count) for _, count in ordered)} 轮）"
    for table in tables:
        if table.get("title") != "建模规格":
            continue
        rows = table.get("rows") if isinstance(table.get("rows"), list) else []
        for row in rows:
            if isinstance(row, list) and row and row[0] == "调参轮数":
                row[:] = ["按算法调参预算", display]


def _find_step(plan: Plan, step_id: str) -> PlanStep | None:
    for step in plan.steps:
        if step.id == step_id:
            return step
    return None


def _dependency_presentation_state(gate: PlanStep | None, dependency: PlanStep) -> str | None:
    """Tell stateful renderers whether a completed output is preview or history.

    ``make_split`` is a proposal only at the feature-screen gate.  Once that gate
    has run, later gates are re-presenting the same already-adopted split; calling
    it a pre-screen preview contradicts the plan timeline.
    """
    if dependency.tool_ref.tool != "make_split" or gate is None:
        return None
    return "preview" if gate.tool_ref.tool == "screen_features" else "adopted"


__all__ = ["GateRenderResult", "render_gate_dependencies"]
