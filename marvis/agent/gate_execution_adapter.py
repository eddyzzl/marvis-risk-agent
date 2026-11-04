"""Execution adapter for gate-level adjust/replan actions."""

from __future__ import annotations

from collections.abc import Callable
import re

from marvis.agent.adjust_specs import (
    FEATURE_BINNING_ADJUST_PARAMS,
    adjust_param_error,
    normalize_adjust_params,
)
from marvis.agent.driver_turn import DriverMessage, DriverTurn
from marvis.agent.gate_payloads import screen_known_features
from marvis.agent.gates.adapters import monitoring_verdict_error
from marvis.agent.plan_utils import downstream_step_ids, find_step
from marvis.orchestrator.capability import resolve_tier
from marvis.orchestrator.contracts import (
    Plan,
    PlanStatus,
    PlanStep,
    plan_fingerprint,
    plan_step_confirmation_fingerprint,
)


ACTION_OUTCOME_METADATA_KEY = "action_outcome"
ACTION_OUTCOME_REJECTED = "rejected"

# The literal quick-reply command offered when feature metrics ran without
# stability: one click recomputes only the metrics step with monthly PSI.
FEATURE_STABILITY_COMMAND = "补算按月 PSI 并重算特征指标"
_PSI_VIEWS = ("psi_month_first", "psi_month_last", "psi_month_previous", "psi_split")
_LEGACY_FEATURE_METRICS = ("iv", "ks", "auc", "coverage", "lift", "psi")
_FEATURE_STABILITY_RE = re.compile(r"PSI|稳定性|稳定度", re.IGNORECASE)
_FEATURE_STABILITY_SPLIT_RE = re.compile(r"split|样本集|训练集|测试集|OOT", re.IGNORECASE)
_NAMED_COLUMN_RE = re.compile(r"按\s*([A-Za-z_][A-Za-z0-9_]*)")
_FEATURE_METRIC_LABELS = {
    "iv": "IV",
    "ks": "KS",
    "auc": "AUC",
    "coverage": "覆盖率",
    "lift": "Lift",
    "psi": "PSI（需对比数据集）",
    "vif": "共线性（VIF）",
    "head_tail_lift": "头尾 Lift",
    "importance": "重要性",
    "meaning_consistency": "含义一致性",
}
_PARAM_LABELS = {
    "features": "特征",
    "bins": "分箱数",
    "metrics": "指标",
    "leakage_ks": "泄漏 KS 阈值",
    "max_missing_rate": "缺失率上限",
    "iv_min": "IV 下限",
    "corr_max": "相关性上限",
    "n_trials": "调参次数",
    "num_boost_round": "迭代轮数",
    "seed": "随机种子",
    "recipes": "算法",
    "target_type": "目标类型",
    "sample_weight_col": "样本权重列",
    "split_config": "样本切分",
    "key_overrides": "拼接键",
    "band_edges": "分数带边界",
    "threshold_patch": "监控阈值",
    "disposition": "处置方式",
    "reason": "理由",
}


def describe_adjust_params(params: dict) -> str:
    """Business-Chinese summary of an adjustment (never a Python repr)."""

    parts: list[str] = []
    for key, value in params.items():
        label = _PARAM_LABELS.get(str(key), f"参数 {key}")
        if str(key) == "key_overrides" and isinstance(value, dict):
            # Keyed by feature-table dataset id; the reader sees the columns
            # (eval S6 read 「拼接键改为 ds_fe8e… mobile」), numbered when
            # several feature tables change at once.
            keys = [" + ".join(map(str, cols)) if isinstance(cols, (list, tuple)) else str(cols)
                    for cols in value.values()]
            value = keys if len(keys) == 1 else [f"第 {index} 张特征表 {text}" for index, text in enumerate(keys, 1)]
        parts.append(f"{label}改为 {_describe_value(value)}")
    return "；".join(parts)


def _describe_value(value) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (list, tuple)):
        return "、".join(_describe_value(item) for item in value) or "（空）"
    if isinstance(value, dict):
        return "，".join(f"{key} {_describe_value(item)}" for key, item in value.items()) or "（空）"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def requests_feature_stability(text: str | None) -> bool:
    return bool(_FEATURE_STABILITY_RE.search(str(text or "")))


def feature_metrics_without_stability(plan: Plan, gate: PlanStep | None) -> PlanStep | None:
    """The upstream feature-metrics step when it computed no PSI view."""

    if gate is None:
        return None
    upstream = GateExecutionAdapter._dependency_step_ids(plan, gate) | {gate.id}
    for step in sorted(plan.steps, key=lambda item: (item.index, item.id)):
        if step.id not in upstream or step.tool_ref.tool != "compute_feature_metrics":
            continue
        inputs = step.inputs or {}
        metrics = (
            list(inputs["metrics"])
            if isinstance(inputs.get("metrics"), list)
            else list(_LEGACY_FEATURE_METRICS)
        )
        if any(str(metric) in _PSI_VIEWS for metric in metrics):
            return None
        return step
    return None


def feature_stability_gap_explanation(step: PlanStep) -> str:
    inputs = step.inputs or {}
    metrics = (
        list(inputs["metrics"])
        if isinstance(inputs.get("metrics"), list)
        else list(_LEGACY_FEATURE_METRICS)
    )
    labels = [_FEATURE_METRIC_LABELS.get(str(metric), str(metric)) for metric in metrics]
    computed = "、".join(labels) if labels else "没有选任何指标"
    return (
        f"本次「{step.title}」只算了 {computed}，创建任务时没有选择稳定性（PSI），"
        "所以结果里没有 PSI。要补上的话，点「补算按月 PSI」：我会在当前计划里给全部特征"
        "加算按月 PSI（首月基准和逐月环比），只重算特征指标，其他设置不变；"
        "样本里没有可识别的月份列时，结果会注明无法计算。"
    )


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

    def needs_dedup_features(self, plan: Plan, gate: PlanStep | None) -> list[str]:
        """Feature ids a join confirmation dependency still needs dedup strategies for."""
        if gate is None:
            return []
        for dep_id in gate.depends_on or []:
            dep = find_step(plan, dep_id)
            if dep is None or dep.tool_ref.tool != "confirm_join":
                continue
            output = self._safe_output(dep.id)
            if not isinstance(output, dict):
                return []
            pending = output.get("needs_dedup") or []
            return [str(feature) for feature in pending]
        return []

    def apply_dedup_strategies(self, plan: Plan, gate: PlanStep | None, dedup_strategies) -> None:
        """Apply per-feature join dedup strategies and reset only the affected join gate."""
        if gate is None or not isinstance(dedup_strategies, dict) or not dedup_strategies:
            return
        clean = {str(key): str(value) for key, value in dedup_strategies.items() if str(value).strip()}
        if not clean:
            return
        replacement_inputs_by_step: dict[str, dict] = {}
        for dep_id in gate.depends_on or []:
            dep = find_step(plan, dep_id)
            if dep is None or dep.tool_ref.tool != "confirm_join":
                continue
            replacement_inputs_by_step[dep.id] = {
                **(dep.inputs or {}),
                "dedup_strategies": clean,
            }
        if replacement_inputs_by_step:
            reset_ids = self._ordered_adjustment_reset_ids(
                plan,
                root_ids=list(replacement_inputs_by_step),
                target_step_id=gate.id,
            )
            self._repo.apply_gate_adjustment(
                plan.id,
                target_step_id=gate.id,
                reset_step_ids=reset_ids,
                replacement_inputs_by_step=replacement_inputs_by_step,
                expected_plan_status=plan.status,
                expected_plan_revision=plan.replan_count,
                expected_plan_fingerprint=plan_fingerprint(plan),
                expected_target_step_fingerprint=(
                    plan_step_confirmation_fingerprint(gate)
                ),
            )

    def exclude_join_feature(
        self,
        plan: Plan,
        gate: PlanStep,
        feature_id: str,
        run_seq: int,
    ) -> DriverTurn:
        """Exclude one reviewed feature table and rerun the join diagnosis.

        This is the deterministic control behind the join conflict card's
        ``排除该特征表`` action.  It accepts only an id from the persisted
        ``propose_join.feature_ids`` input and never turns a multi-table join into
        an empty join.  The proposal input revision and every dependent reset use
        the same reviewed-snapshot transaction as other typed adjustments.
        """

        propose_steps = [
            step
            for step in plan.steps
            if step.id in self._dependency_step_ids(plan, gate)
            and step.tool_ref.tool == "propose_join"
        ]
        if len(propose_steps) != 1:
            return self._instruction_message(
                plan,
                gate,
                run_seq,
                "无法唯一定位当前拼接诊断，未排除特征表。请刷新后重试。",
            )
        propose = propose_steps[0]
        raw_feature_ids = (propose.inputs or {}).get("feature_ids")
        if not isinstance(raw_feature_ids, (list, tuple)):
            return self._instruction_message(
                plan,
                gate,
                run_seq,
                "当前拼接诊断没有可编辑的特征表清单，未执行排除。",
            )
        current_feature_ids = list(
            dict.fromkeys(
                str(item).strip()
                for item in raw_feature_ids
                if str(item).strip()
            )
        )
        selected_feature_id = str(feature_id or "").strip()
        if selected_feature_id not in current_feature_ids:
            return self._instruction_message(
                plan,
                gate,
                run_seq,
                "该特征表已不在当前拼接方案中，未重复排除。请刷新后重试。",
            )
        remaining_feature_ids = [
            item for item in current_feature_ids if item != selected_feature_id
        ]
        if not remaining_feature_ids:
            return self._instruction_message(
                plan,
                gate,
                run_seq,
                "拼接方案必须至少保留一张特征表，未执行排除。",
            )

        reset_ids = self._ordered_adjustment_reset_ids(
            plan,
            root_ids=[propose.id],
            target_step_id=gate.id,
        )
        self._repo.apply_gate_adjustment(
            plan.id,
            target_step_id=gate.id,
            reset_step_ids=reset_ids,
            replacement_inputs_by_step={
                propose.id: {
                    **(propose.inputs or {}),
                    "feature_ids": remaining_feature_ids,
                }
            },
            expected_plan_status=plan.status,
            expected_plan_revision=plan.replan_count,
            expected_plan_fingerprint=plan_fingerprint(plan),
            expected_target_step_fingerprint=plan_step_confirmation_fingerprint(gate),
        )
        turn = self._run_and_handle(plan.id, run_seq=run_seq)
        turn.messages.insert(
            0,
            DriverMessage(
                "chat",
                f"已排除特征表「{selected_feature_id}」并重新执行拼接诊断。",
                {
                    "plan_id": plan.id,
                    "step_id": propose.id,
                    "excluded_feature_id": selected_feature_id,
                    "run_seq": run_seq,
                },
            ),
        )
        return turn

    def screen_selection_input_updates(
        self,
        plan: Plan,
        gate: PlanStep | None,
        selection,
    ) -> dict[str, list[str]]:
        """Validate an edited screen selection for atomic gate confirmation.

        The current gate may sit behind a completed, deterministic
        ``resolve_special_values`` no-op.  The reviewed feature list therefore
        becomes a concrete gate input.  The completed ``screen_features`` Tool
        output and its immutable run receipt remain unchanged: a human review
        decision is not a second Tool execution and must not impersonate one.

        The caller merges this patch into the same transaction that confirms
        the gate, so a rejected/stale confirmation cannot leave revised inputs
        behind.
        """
        if gate is None:
            return {}
        selected = [str(feature) for feature in (selection or []) if str(feature).strip()]
        if not selected:
            return {}
        chosen_for_gate: list[str] = []
        for dep_id in gate.depends_on or []:
            dep = find_step(plan, dep_id)
            if dep is None or dep.tool_ref.tool != "screen_features":
                continue
            output = self._safe_output(dep_id)
            if not isinstance(output, dict):
                continue
            known = screen_known_features(output)
            chosen = [feature for feature in dict.fromkeys(selected) if not known or feature in known]
            if not chosen:
                continue
            chosen_for_gate = chosen

        if not chosen_for_gate or "features" not in (gate.inputs or {}):
            return {}
        for dep_id in gate.depends_on or []:
            dep = find_step(plan, dep_id)
            if dep is None or dep.tool_ref.tool != "resolve_special_values":
                continue
            output = self._safe_output(dep_id)
            resolved = output.get("selected") if isinstance(output, dict) else None
            if isinstance(resolved, list):
                allowed = {
                    str(feature).strip()
                    for feature in resolved
                    if str(feature).strip()
                }
                chosen_for_gate = [
                    feature
                    for feature in chosen_for_gate
                    if feature in allowed
                ]
            break
        if not chosen_for_gate:
            return {}
        return {"features": chosen_for_gate}

    def apply_replan(self, plan: Plan, gate: PlanStep | None, instruction, run_seq) -> DriverTurn:
        """Regenerate remaining steps from a structural instruction and continue."""
        replan = getattr(self._executor, "replan_from_instruction", None)
        if replan is None or not replan(plan.id, instruction):
            return self._instruction_message(
                plan,
                gate,
                run_seq,
                self._replan_failure_text(plan, gate),
            )
        revised = self._repo.load_plan(plan.id)
        if revised.status == PlanStatus.VALIDATED:
            return DriverTurn(revised.id, revised.status.value, [
                DriverMessage("chat", "已按指令重规划，请查看新计划。", {"plan_id": revised.id, "run_seq": run_seq}),
                self._plan_overview_message(revised),
            ])
        turn = self._run_and_handle(plan.id, run_seq=run_seq)
        turn.messages.insert(
            0,
            DriverMessage("chat", "已按指令重规划并继续执行。", {"plan_id": plan.id, "run_seq": run_seq}),
        )
        return turn

    @staticmethod
    def _replan_failure_text(plan: Plan, gate: PlanStep | None) -> str:
        """Say why the plan was kept, and name a next step that exists."""

        budget = resolve_tier(plan.tier).max_replan_iterations
        if budget <= 0:
            head = "当前能力档位不允许重规划，计划和已有结果保持不变。"
        elif int(plan.replan_count) >= budget:
            head = f"这个计划的重规划次数（{budget} 次）已经用完，计划和已有结果保持不变。"
        else:
            head = "我没能把这条要求改写成一份能通过平台校验的新计划，计划和已有结果保持不变。"
        where = f"当前仍停在「{gate.title}」。" if gate is not None else ""
        return (
            head
            + where
            + "可以直接说要把哪个参数改成什么值，我只重算受影响的步骤；"
            "需要换数据或换流程时，可以新建任务重新配置。"
        )

    def add_feature_stability_metrics(
        self,
        plan: Plan,
        gate: PlanStep | None,
        run_seq,
        *,
        instruction: str | None = None,
    ) -> DriverTurn | None:
        """Recompute feature metrics with monthly (and optionally split) PSI.

        Only the metrics selection changes; the feature set, target and every
        other reviewed input stay as they are.  Returns ``None`` when the plan
        has no feature-metrics step lacking PSI.
        """

        step = feature_metrics_without_stability(plan, gate)
        if step is None or gate is None:
            return None
        inputs = dict(step.inputs or {})
        metrics = (
            list(inputs["metrics"])
            if isinstance(inputs.get("metrics"), list)
            else list(_LEGACY_FEATURE_METRICS)
        )
        added = ["psi_month_first", "psi_month_previous"]
        text = str(instruction or "")
        if _FEATURE_STABILITY_SPLIT_RE.search(text):
            added.append("psi_split")
        inputs["metrics"] = [*metrics, *[metric for metric in added if metric not in metrics]]
        named = _NAMED_COLUMN_RE.search(text)
        if (
            named is not None
            and named.group(1).lower() not in {"psi", "split", "oot"}
            and not str(inputs.get("time_col") or "").strip()
        ):
            # A column the user names is passed through; the Tool reads only
            # columns that exist and marks the view unavailable otherwise.
            inputs["time_col"] = named.group(1)
        reset_ids = self._ordered_adjustment_reset_ids(
            plan,
            root_ids=[step.id],
            target_step_id=gate.id,
        )
        self._repo.apply_gate_adjustment(
            plan.id,
            target_step_id=gate.id,
            reset_step_ids=reset_ids,
            replacement_inputs_by_step={step.id: inputs},
            expected_plan_status=plan.status,
            expected_plan_revision=plan.replan_count,
            expected_plan_fingerprint=plan_fingerprint(plan),
            expected_target_step_fingerprint=plan_step_confirmation_fingerprint(gate),
        )
        turn = self._run_and_handle(plan.id, run_seq=run_seq)
        views = "首月基准、逐月环比" + ("、样本集" if "psi_split" in added else "")
        column = f"（时间列 {inputs['time_col']}）" if inputs.get("time_col") else ""
        feature_count = len(inputs.get("features") or [])
        scope = f"全部 {feature_count} 个特征" if feature_count else "全部特征"
        turn.messages.insert(
            0,
            DriverMessage(
                "chat",
                f"已在「{step.title}」里加上按月 PSI{column}（{views}），对{scope}重新计算；"
                "特征集合和其他设置不变。",
                {"plan_id": plan.id, "step_id": step.id, "run_seq": run_seq},
            ),
        )
        return turn

    def apply_adjust(self, plan: Plan, gate: PlanStep, params, run_seq) -> DriverTurn:
        """Apply declared parameter overrides, reset affected steps, and rerun.

        Candidates are the gate's dependencies plus the gate step itself. Most
        gates wrap a separate upstream computation (e.g. confirm_join gating
        propose_join), so the dependency is where params get overridden -- but
        some gates (e.g. STRATEGY_DEVELOPMENT's 设计分数带) are the computation
        being reviewed, with no separate confirm-wrapper step, so a param like
        band_edges only ever appears in the gate's own declared inputs. Checking
        the gate itself last (after its deps) keeps existing dependency-scoped
        adjust behavior unchanged for every template that already relies on it.
        """
        deps = [step for step in (find_step(plan, dep_id) for dep_id in (gate.depends_on or [])) if step is not None]
        candidates = [*deps, gate]
        params = normalize_adjust_params(params)
        if gate.tool_ref.tool == "apply_monitoring_disposition":
            monitoring_error = self._monitoring_adjust_error(plan, gate, params)
            if monitoring_error:
                return self._instruction_message(
                    plan,
                    gate,
                    run_seq,
                    monitoring_error,
                )
        validation_error = adjust_param_error(params)
        if validation_error:
            return self._instruction_message(plan, gate, run_seq, validation_error)

        primary = None
        replacement_inputs_by_step: dict[str, dict] = {}
        # The binning gate's features/bins are its own decision inputs (which
        # features to bin), never the upstream metrics feature set.
        gate_owned = (
            FEATURE_BINNING_ADJUST_PARAMS
            if gate.tool_ref.tool == "analyze_feature_bins"
            else frozenset()
        )
        for dep in candidates:
            overrides = {
                key: value
                for key, value in params.items()
                if key in (dep.inputs or {}) and (dep is gate or key not in gate_owned)
            }
            # Backward compatibility for plans created before the join-key picker
            # declared key_overrides in the builtin templates. Historical pending
            # gates can still be repaired in place instead of forcing a new task.
            if dep.tool_ref.tool == "propose_join" and "key_overrides" in params:
                overrides["key_overrides"] = params["key_overrides"]
            if "sample_weight_col" in overrides:
                if dep.tool_ref.tool != "choose_modeling_spec":
                    overrides.pop("sample_weight_col", None)
                else:
                    sample_weight_error = self._sample_weight_adjust_error(dep.id, overrides["sample_weight_col"])
                    if sample_weight_error:
                        return self._instruction_message(plan, gate, run_seq, sample_weight_error)
            if not overrides:
                continue
            replacement_inputs_by_step[dep.id] = {
                **(dep.inputs or {}),
                **overrides,
            }
            if primary is None:
                primary = dep
        if primary is None:
            available = sorted({str(key) for dep in deps for key in (dep.inputs or {}).keys()})
            hint = f"可调整参数: {', '.join(available)}。" if available else "当前节点没有声明可调整参数。"
            return self._instruction_message(
                plan,
                gate,
                run_seq,
                f"没有识别到可调整的参数，未重算。{hint}",
            )
        adjusted_ids = list(replacement_inputs_by_step)
        reset_ids = self._ordered_adjustment_reset_ids(
            plan,
            root_ids=adjusted_ids,
            target_step_id=gate.id,
        )
        self._repo.apply_gate_adjustment(
            plan.id,
            target_step_id=gate.id,
            reset_step_ids=reset_ids,
            replacement_inputs_by_step=replacement_inputs_by_step,
            expected_plan_status=plan.status,
            expected_plan_revision=plan.replan_count,
            expected_plan_fingerprint=plan_fingerprint(plan),
            expected_target_step_fingerprint=plan_step_confirmation_fingerprint(gate),
        )
        turn = self._run_and_handle(plan.id, run_seq=run_seq)
        turn.messages.insert(
            0,
            DriverMessage(
                "chat",
                f"已按你的要求调整（{describe_adjust_params(params)}），并重算「{primary.title}」。",
                {"plan_id": plan.id, "step_id": primary.id, "run_seq": run_seq},
            ),
        )
        return turn

    def is_noop_adjustment(
        self,
        plan: Plan,
        gate: PlanStep,
        params,
    ) -> bool:
        """Return whether every applicable normalized override already matches.

        A semantic router may correctly understand an instruction such as
        "keep every model at one trial and train them" as an ``adjust`` action.
        Rewriting the same value must not invalidate already-reviewed upstream
        evidence.  Keep this comparison beside ``apply_adjust`` so aliases and
        dependency scoping are identical for both the no-op and recompute paths.
        """

        normalized = normalize_adjust_params(params)
        if adjust_param_error(normalized):
            return False
        deps = [
            step
            for step in (
                find_step(plan, dep_id)
                for dep_id in (gate.depends_on or [])
            )
            if step is not None
        ]
        candidates = [*deps, gate]
        matched_any = False
        for candidate in candidates:
            current_inputs = candidate.inputs or {}
            for key, requested in normalized.items():
                if key not in current_inputs:
                    continue
                matched_any = True
                current = normalize_adjust_params(
                    {key: current_inputs[key]}
                ).get(key)
                if current != requested:
                    return False
        return matched_any

    def _monitoring_adjust_error(
        self,
        plan: Plan,
        gate: PlanStep,
        params: dict,
    ) -> str | None:
        verdict_error = monitoring_verdict_error(
            plan,
            gate,
            self._safe_output,
        )
        if verdict_error:
            return verdict_error
        allowed = {"disposition", "reason", "threshold_patch"}
        unexpected = sorted(set(params) - allowed)
        if unexpected:
            return (
                "监控处置只能调整 disposition、reason 和 threshold_patch；"
                "不可修改已冻结的 plan/run/strategy 证据。"
            )
        disposition = params.get("disposition")
        if disposition is not None and disposition not in {
            "observe",
            "adjust_threshold",
            "new_version",
        }:
            return "disposition 必须是 observe、adjust_threshold 或 new_version。"
        if "reason" in params and not str(params.get("reason") or "").strip():
            return "监控处置理由不能为空。"
        if "threshold_patch" not in params:
            return None
        patch = params.get("threshold_patch")
        if not isinstance(patch, dict) or not patch:
            return "threshold_patch 必须是非空对象。"

        checks: dict[str, dict] = {}
        for dep_id in gate.depends_on or []:
            dep = find_step(plan, dep_id)
            if dep is None or dep.tool_ref.tool != "run_strategy_monitoring":
                continue
            output = self._safe_output(dep.id)
            if not isinstance(output, dict):
                continue
            for check in output.get("checks") or []:
                if not isinstance(check, dict):
                    continue
                key = str(check.get("metric") or check.get("id") or "").strip()
                if key:
                    checks[key] = check

        for raw_check_id, changes in patch.items():
            check_id = str(raw_check_id)
            if checks and check_id not in checks:
                return f"threshold_patch 包含未知监控项 {check_id}。"
            if not isinstance(changes, dict) or not changes:
                return f"threshold_patch.{check_id} 必须是非空对象。"
            if not set(changes) <= {"warn", "fail"}:
                return "threshold_patch 只能修改 warn/fail。"
            for field, value in changes.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    return f"threshold_patch.{check_id}.{field} 必须是有限数字。"
                if value != value or value in {float("inf"), float("-inf")}:
                    return f"threshold_patch.{check_id}.{field} 必须是有限数字。"
            current = checks.get(check_id, {})
            direction = current.get("direction")
            warn = changes.get("warn", current.get("warn"))
            fail = changes.get("fail", current.get("fail"))
            if isinstance(warn, (int, float)) and isinstance(fail, (int, float)):
                if direction == "min" and float(warn) < float(fail):
                    return f"{check_id} 为 min 方向，必须满足 warn >= fail。"
                if direction == "max" and float(warn) > float(fail):
                    return f"{check_id} 为 max 方向，必须满足 warn <= fail。"
        return None

    @staticmethod
    def _ordered_adjustment_reset_ids(
        plan: Plan,
        *,
        root_ids: list[str],
        target_step_id: str,
    ) -> list[str]:
        affected_ids = (
            set(root_ids)
            | downstream_step_ids(plan, root_ids)
            | {target_step_id}
        )
        return [
            step.id
            for step in sorted(plan.steps, key=lambda item: (item.index, item.id))
            if step.id in affected_ids
        ]

    @staticmethod
    def _dependency_step_ids(plan: Plan, gate: PlanStep) -> set[str]:
        by_id = {step.id: step for step in plan.steps}
        dependency_ids = {
            str(step_id) for step_id in (gate.depends_on or []) if str(step_id)
        }
        pending = list(dependency_ids)
        while pending:
            step = by_id.get(pending.pop())
            if step is None:
                continue
            for parent_id in step.depends_on or []:
                normalized = str(parent_id)
                if normalized and normalized not in dependency_ids:
                    dependency_ids.add(normalized)
                    pending.append(normalized)
        return dependency_ids

    def _sample_weight_adjust_error(self, step_id: str, value) -> str | None:
        selected = str(value or "").strip()
        if not selected:
            return None
        output = self._safe_output(step_id)
        if not isinstance(output, dict):
            return "缺少建模规格输出，无法调整样本权重列。"
        candidates = [str(col) for col in (output.get("sample_weight_candidates") or []) if str(col).strip()]
        current = str(output.get("sample_weight_col") or "").strip()
        allowed = set(candidates)
        if current:
            allowed.add(current)
        if selected not in allowed:
            display = "、".join(candidates) if candidates else "无"
            return f"样本权重列 `{selected}` 不在已检测候选列中，未重算。候选列:{display}。"
        return None

    def _instruction_message(self, plan: Plan, gate: PlanStep | None, run_seq, text) -> DriverTurn:
        return DriverTurn(
            plan.id,
            plan.status.value,
            [
                DriverMessage(
                    "gate",
                    text,
                    {
                        "plan_id": plan.id,
                        "step_id": gate.id if gate else None,
                        "run_seq": run_seq,
                        ACTION_OUTCOME_METADATA_KEY: ACTION_OUTCOME_REJECTED,
                    },
                )
            ],
        )


__all__ = [
    "ACTION_OUTCOME_METADATA_KEY",
    "ACTION_OUTCOME_REJECTED",
    "FEATURE_STABILITY_COMMAND",
    "describe_adjust_params",
    "feature_metrics_without_stability",
    "feature_stability_gap_explanation",
    "requests_feature_stability",
    "GateExecutionAdapter",
]
