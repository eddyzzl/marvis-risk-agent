from __future__ import annotations

import re
from marvis.modeling_policy_signals import has_monotonic_policy, monotonic_policy_profile
from marvis.packs.modeling.errors import ModelingError

from marvis.packs.modeling._common import _cleanup_unattached_artifact, _finite_float_or_none, _format_number_token, _is_metric_key, _jsonable, _nonnegative_float_or_none, _positive_int_or_none, _score_first, _snapshot_latest_model_meta
from marvis.packs.modeling._runtime import _Runtime, _artifact, _artifact_base_dir, _runtime
from marvis.packs.modeling.delivery_tools import _artifact_capabilities
from marvis.packs.modeling.report_tools import _scorecard_table_rows
from marvis.packs.modeling.train_tools import _binary_selection_score_and_metric, _overfit_penalized_test_ks, _refit_champion_on_train_plus_test, _resolve_scenario_eval_metric


_POLICY_METRIC_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


_POLICY_METRIC_THRESHOLD_SHORTCUTS = {
    "min_oot_ks": ("oot_ks", "min"),
    "min_test_ks": ("test_ks", "min"),
    "min_oot_auc": ("oot_auc", "min"),
    "min_test_auc": ("test_auc", "min"),
    "min_oot_macro_auc": ("oot_macro_auc", "min"),
    "min_test_macro_auc": ("test_macro_auc", "min"),
    "max_oot_rmse": ("oot_rmse", "max"),
    "max_test_rmse": ("test_rmse", "max"),
    "max_oot_logloss": ("oot_logloss", "max"),
    "max_test_logloss": ("test_logloss", "max"),
}


#: SEL-7 default quality guardrails (warn-only, never block): a candidate whose
#: train-test KS gap exceeds this is flagged overfit_warning; the fail-level
#: threshold in DEFAULT_MONITORING_THRESHOLDS (overfit_train_test_gap) is 0.12,
#: this warn-level guardrail is intentionally a notch below it (0.10) so the
#: selection-time hint fires before the post-selection monitoring gate would.
DEFAULT_OVERFIT_GAP_WARN_THRESHOLD = 0.10


#: SEL-7: a candidate with fewer than this many input features is flagged
#: sanity_warning (too few features to plausibly be a robust credit model).
DEFAULT_MIN_FEATURE_COUNT_WARNING = 3


def tool_compare_experiments(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    experiment_ids = [str(item) for item in inputs["experiment_ids"]]
    compared = runtime.experiments.compare(experiment_ids)
    rows = [row for row in compared.get("experiments", []) if isinstance(row, dict)]
    _attach_capabilities_to_comparison_rows(runtime, rows)
    _attach_policy_profile_to_comparison_rows(runtime, rows)
    target_type = str(inputs.get("target_type") or "").strip()
    if not target_type and rows:
        first_id = str(rows[0].get("id") or "")
        if first_id:
            target_type = getattr(runtime.experiments.get(first_id).config, "target_type", "binary")
    eval_metric = _resolve_scenario_eval_metric(
        runtime, experiment_ids, str(inputs.get("eval_metric") or "").strip()
    )
    # SEL-5: surface the same "within sampling error" hint the gate text uses in
    # select_experiment, here against the metric-best row -- so a caller
    # comparing candidates (before ever calling select_experiment) sees the same
    # signal ahead of time.
    ks_ci_note = ""
    if rows and str(target_type or "binary") == "binary":
        best_row, _metric = _pick_best_comparison_row(
            rows, target_type=target_type or "binary", eval_metric=eval_metric
        )
        ks_ci_note = _ks_ci_overlap_note(best_row, rows, target_type=target_type or "binary")
    compared["ks_ci_note"] = ks_ci_note
    compared["eval_metric"] = eval_metric
    return _jsonable(compared)


def tool_select_experiment(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    experiment_ids = [str(item) for item in inputs.get("experiment_ids") or [] if str(item).strip()]
    if not experiment_ids:
        raise ModelingError("experiment_ids must not be empty")
    target_type = str(inputs.get("target_type") or "").strip()
    if not target_type:
        target_type = getattr(runtime.experiments.get(experiment_ids[0]).config, "target_type", "binary")
    eval_metric = _resolve_scenario_eval_metric(
        runtime, experiment_ids, str(inputs.get("eval_metric") or "").strip()
    )
    compared = runtime.experiments.compare(experiment_ids)
    rows = [row for row in compared.get("experiments", []) if isinstance(row, dict)]
    _attach_capabilities_to_comparison_rows(runtime, rows)
    _attach_policy_profile_to_comparison_rows(runtime, rows)
    selection_policy = _normalize_selection_policy(inputs.get("selection_policy"))
    selected_id = str(inputs.get("selected_experiment_id") or "").strip()
    if selected_id:
        selected = next((row for row in rows if row.get("id") == selected_id), None)
        if selected is None:
            raise ModelingError(f"selected_experiment_id is not in candidates: {selected_id}")
        selection_metric = str(inputs.get("selection_metric") or "manual")
        selection_reason = "用户指定实验。"
        policy_decision = _selection_policy_decision(selected, selection_policy, explicit=True)
        if policy_decision["status"] == "blocked":
            raise ModelingError(_selection_policy_block_message(selected_id, policy_decision))
    else:
        selected, selection_metric, policy_decision = _pick_best_comparison_row_with_policy(
            rows,
            target_type=target_type,
            policy=selection_policy,
            eval_metric=eval_metric,
        )
        selected_id = str(selected.get("id") or "")
        # DOM-6: name the scenario's evaluation basis in the gate text whenever
        # selection deviated from the platform default (ks_auc / overfit-penalized
        # test KS) -- e.g. "本场景按 test_lift_head_10 选优" for a marketing/recall
        # scenario, so the divergence from KS is explicit instead of silent.
        scenario_note = (
            f"本场景按 {eval_metric} 选优;实际选择指标 {selection_metric}。"
            if eval_metric and eval_metric != "ks_auc"
            else ""
        )
        if _selection_policy_requested(selection_policy) and policy_decision["status"] == "accepted":
            selection_reason = f"按 {selection_metric} 在满足交付/审批策略的候选中自动选择。"
            if policy_decision.get("selected_by_preference"):
                selection_reason = f"按 {selection_metric} 在评分卡优先候选中自动选择。"
        elif _selection_policy_requested(selection_policy) and policy_decision["status"] == "overridden":
            selection_reason = f"按 {selection_metric} 自动选择;未满足全部交付/审批策略,已按 override_reason 放行。"
        elif _delivery_ready(selected):
            selection_reason = f"按 {selection_metric} 在 PMML/验证移交可用候选中自动选择。"
        else:
            selection_reason = f"按 {selection_metric} 自动选择。"
        if scenario_note:
            selection_reason = f"{selection_reason} {scenario_note}"
    artifact_id = str(selected.get("artifact_id") or "")
    if not artifact_id:
        raise ModelingError(f"selected experiment has no artifact: {selected_id}")
    artifact = _artifact(runtime, artifact_id)
    experiment = runtime.experiments.get(selected_id)
    capabilities = _artifact_capabilities(
        artifact,
        base_dir=_artifact_base_dir(runtime.settings, experiment.task_id),
    )
    runtime.experiments.set_status(selected_id, "selected")
    pre_refit_metrics = {k: v for k, v in selected.items() if _is_metric_key(k) and v is not None}
    refit_requested = bool(inputs.get("refit_on_train_plus_test", True))
    refit_info = _apply_champion_refit(
        runtime,
        experiment=experiment,
        recipe=str(selected.get("recipe") or experiment.recipe_id),
        requested=refit_requested,
        pre_refit_metrics=pre_refit_metrics,
    ) if target_type == "binary" else {"applied": False, "requested": refit_requested, "reason": "非二分类任务暂不支持全量重训。"}
    final_artifact_id = refit_info.get("artifact_id") or artifact_id
    final_experiment_id = refit_info.get("experiment_id") or selected_id
    final_metrics = refit_info.get("metrics") or pre_refit_metrics
    ks_ci_note = _ks_ci_overlap_note(selected, rows, target_type=target_type)
    if ks_ci_note:
        selection_reason = f"{selection_reason} {ks_ci_note}"
    return {
        "selected_experiment_id": final_experiment_id,
        "artifact_id": final_artifact_id,
        "recipe": selected.get("recipe") or experiment.recipe_id,
        "target_type": target_type,
        "eval_metric": eval_metric,
        "selection_metric": selection_metric,
        "selection_reason": selection_reason,
        "metrics": final_metrics,
        "capabilities": capabilities,
        "policy_profile": selected.get("policy_profile") or {},
        "policy_decision": policy_decision,
        "scorecard_table": selected.get("scorecard_table") or [],
        "model_params": selected.get("model_params") or {},
        "experiments": rows,
        "refit": refit_info,
        "ks_ci_note": ks_ci_note,
    }


#: SEL-5: champion selection message appended when the runner-up's KS bootstrap
#: CI overlaps the champion's -- the observed gap may be within sampling noise.
_KS_CI_OVERLAP_NOTE_TEMPLATE = (
    "冠军与亚军({runner_recipe})的{metric_label} bootstrap 95% 置信区间存在重叠"
    "(冠军 [{champ_low:.4f}, {champ_high:.4f}] vs 亚军 [{runner_low:.4f}, {runner_high:.4f}]),"
    "差异在抽样误差内,不构成统计显著优势(选择规则本身未改变,仍按 {metric_label} 排序)。"
)


def _ks_ci_overlap_note(selected: dict, rows: list[dict], *, target_type: str) -> str:
    """SEL-5: when the champion and runner-up's KS bootstrap CIs overlap, surface
    a "statistically indistinguishable" hint in the selection output -- purely
    informational, the selection rule (max overfit-penalized test KS) is
    unchanged. Only evaluated for binary targets (KS is a binary-only
    statistic); compares test_ks CI first (the actual selection basis), falling
    back to oot_ks CI when test CI evidence is missing on either side."""
    if str(target_type or "binary") != "binary":
        return ""
    others = [row for row in rows if row.get("id") != selected.get("id")]
    if not others:
        return ""
    runner_up = max(others, key=_overfit_penalized_test_ks)
    for metric, label in (("test_ks", "test KS"), ("oot_ks", "OOT KS")):
        champ_low = selected.get(f"{metric}_ci_low")
        champ_high = selected.get(f"{metric}_ci_high")
        runner_low = runner_up.get(f"{metric}_ci_low")
        runner_high = runner_up.get(f"{metric}_ci_high")
        if not all(isinstance(v, (int, float)) for v in (champ_low, champ_high, runner_low, runner_high)):
            continue
        overlaps = float(champ_low) <= float(runner_high) and float(runner_low) <= float(champ_high)
        if not overlaps:
            return ""
        return _KS_CI_OVERLAP_NOTE_TEMPLATE.format(
            runner_recipe=runner_up.get("recipe") or runner_up.get("id") or "?",
            metric_label=label,
            champ_low=float(champ_low),
            champ_high=float(champ_high),
            runner_low=float(runner_low),
            runner_high=float(runner_high),
        )
    return ""


#: D14: metric keys computed on the refit's random 5% ``__refit_holdout__`` slice
#: that are optimistically biased and must never surface as headline "held-out"
#: results. The refit combines train+test into one training pool and carves a
#: deterministic random 5% back out only to satisfy split_modeling_frame's
#: non-empty-test contract (train_tools._refit_champion_on_train_plus_test); that
#: slice is in-distribution with the training data, so any test_*/weighted_test_*
#: KS/AUC/lift and the test-vs-train PSIs on it are near-meaningless. train_*,
#: oot_*, weighted_train/oot_*, psi_oot_vs_train and overfit_flag stay honest
#: (refit trained on train+test, OOT untouched) and are left intact.
def _is_refit_holdout_tainted_key(key: str) -> bool:
    return (
        key in ("psi_test_vs_train", "weighted_psi_test_vs_train")
        or key.startswith(("test_", "weighted_test_"))
    )


def _apply_champion_refit(
    runtime: "_Runtime",
    *,
    experiment,
    recipe: str,
    requested: bool,
    pre_refit_metrics: dict,
) -> dict:
    """select_experiment's post-selection refit step (TUNE-4, default enabled).

    Retrains the champion's frozen hyperparameters on train+test combined (test's
    information would otherwise be permanently wasted on the delivered artifact)
    and registers the refit as a new experiment/artifact so it is comparable and
    auditable like any other trained candidate. OOT is untouched by the refit;
    this reports before/after OOT metrics side by side so the caller can confirm
    the refit actually helped before relying on it. Returns
    ``{"applied": bool, "requested": bool, "reason": str, ...}``; on success also
    ``artifact_id``/``experiment_id`` (of the refit artifact) and ``metrics``
    (the refit artifact's own metrics, becoming the tool's headline ``metrics``).
    """
    if not requested:
        return {"applied": False, "requested": False, "reason": "未请求全量重训(refit_on_train_plus_test=false)。"}
    # LT-5: _refit_champion_on_train_plus_test (via _train_recipe -> save_model)
    # fully promotes the refit's model file/meta on disk -- including overwriting
    # the base_dir's "latest" model_meta.json pointer -- before this function ever
    # sees a result. The pre-refit snapshot must be taken BEFORE that call, or it
    # would capture the refit's own (already-overwritten) meta instead of the
    # true prior state, defeating the rollback below. Mirrors train_tools.py's
    # tool_train_model/tool_train_models cleanup.
    artifact_dir = _artifact_base_dir(runtime.settings, experiment.task_id)
    meta_snapshot = _snapshot_latest_model_meta(artifact_dir)
    try:
        refit = _refit_champion_on_train_plus_test(
            runtime, task_id=experiment.task_id, experiment=experiment, recipe=recipe,
        )
    except Exception as exc:  # noqa: BLE001 - refit is an enhancement, never a hard blocker
        return {"applied": False, "requested": True, "reason": f"全量重训失败,已保留原候选:{exc}"}
    if refit is None:
        return {"applied": False, "requested": True, "reason": "候选实验缺少可全量重训的切分信息,已保留原候选。"}
    refit_config, result = refit
    # attach_result's DB write is a separate, later transaction from the file
    # promotion above -- a failure here must not leave an on-disk artifact with
    # no DB row pointing at it.
    refit_experiment_id = runtime.experiments.create(experiment.task_id, recipe, refit_config)
    try:
        runtime.experiments.attach_result(refit_experiment_id, result)
    except Exception:
        _cleanup_unattached_artifact(result.artifact, artifact_dir, meta_snapshot)
        runtime.experiments.set_status(refit_experiment_id, "failed")
        raise
    refit_experiment = runtime.experiments.get(refit_experiment_id)
    refit_row = next(
        (row for row in runtime.experiments.compare([refit_experiment_id])["experiments"] if row.get("id") == refit_experiment_id),
        {},
    )
    # D14: the refit reports honest train_*/oot_* (trained on train+test, OOT
    # untouched) as headline metrics, but suppresses the random-5%-holdout test
    # family -- those are optimistically biased and would masquerade as a genuine
    # held-out test evaluation. The tainted values are kept as renamed internal
    # diagnostics (refit_holdout_*), never merged into the headline. The pre-refit
    # champion's honest held-out test_* stays available via metrics_before_refit.
    post_metrics = {
        k: v
        for k, v in refit_row.items()
        if _is_metric_key(k) and v is not None and not _is_refit_holdout_tainted_key(k)
    }
    refit_holdout_metrics = {
        f"refit_holdout_{k}": v
        for k, v in refit_row.items()
        if _is_metric_key(k) and v is not None and _is_refit_holdout_tainted_key(k)
    }
    return {
        "applied": True,
        "requested": True,
        "reason": (
            "已用冠军的定型超参在 train+test 全量重训,OOT 未参与训练;报告的 headline 指标为重训后模型的 "
            "OOT/train 结果。为满足非空 test 契约随机切出的 5% holdout 与训练同分布,其 test_ks/test_auc/psi "
            "已剔出 headline,仅作内部诊断(refit_holdout_*)记录。冠军在真实留出 test 上的指标见 metrics_before_refit。"
        ),
        "experiment_id": refit_experiment_id,
        "artifact_id": refit_experiment.artifact_id,
        "metrics": post_metrics,
        "metrics_before_refit": pre_refit_metrics,
        "metrics_after_refit": post_metrics,
        "refit_holdout_metrics": refit_holdout_metrics,
        "oot_ks_before_refit": pre_refit_metrics.get("oot_ks"),
        "oot_ks_after_refit": post_metrics.get("oot_ks"),
    }


def _pick_best_comparison_row_with_policy(
    rows: list[dict],
    *,
    target_type: str,
    policy: dict,
    eval_metric: str = "ks_auc",
) -> tuple[dict, str, dict]:
    policy = _normalize_selection_policy(policy)
    if not _selection_policy_requested(policy):
        selected, metric = _pick_best_comparison_row(rows, target_type=target_type, eval_metric=eval_metric)
        return selected, metric, _selection_policy_decision(selected, policy, explicit=False)

    compliant = [row for row in rows if not _selection_policy_violations(row, policy)]
    if compliant:
        candidates = compliant
    elif _selection_policy_has_hard_requirements(policy) and not policy.get("allow_policy_override"):
        raise ModelingError(_no_policy_candidate_message(rows, policy))
    else:
        candidates = rows

    selected_by_preference = False
    if policy.get("prefer_scorecard"):
        scorecard_candidates = [
            row for row in candidates
            if _row_policy_profile(row).get("scorecard")
        ]
        if scorecard_candidates:
            candidates = scorecard_candidates
            selected_by_preference = True

    selected, metric = _pick_best_comparison_row(candidates, target_type=target_type, eval_metric=eval_metric)
    decision = _selection_policy_decision(selected, policy, explicit=False)
    decision["evaluated_candidates"] = len(rows)
    decision["policy_candidate_count"] = len(compliant)
    decision["selected_by_preference"] = selected_by_preference
    if decision["status"] == "blocked":
        raise ModelingError(_selection_policy_block_message(str(selected.get("id") or ""), decision))
    return selected, metric, decision


def _pick_best_comparison_row(
    rows: list[dict], *, target_type: str, eval_metric: str = "ks_auc"
) -> tuple[dict, str]:
    """Pick the best comparison row. Binary maximizes the overfit-penalized test KS by
    default — OOT is reported but not used for selection, matching
    tune_hyperparameters (DOM-9). When ``eval_metric="response_lift"``
    (marketing/recall scenario templates, DOM-6) binary instead maximizes test
    top-decile lift.

    SEL-7: candidates excluded by the delivery-ready (PMML+handoff) pre-filter no
    longer vanish silently -- when the metric-best row overall is NOT
    delivery-ready-eligible (i.e. it would have won on the raw metric but is
    filtered out), every row this pre-filter drops is annotated in place with
    ``delivery_excluded=True`` / ``delivery_excluded_reason`` so the comparison
    table can display "excluded, would have scored higher" instead of the row
    just disappearing from consideration.
    """
    if not rows:
        raise ModelingError("experiment_ids must resolve to experiments")
    target_type = str(target_type or "binary")
    metric_key, minimize = _selection_metric_basis(target_type, eval_metric=eval_metric)
    delivery_ready = [row for row in rows if _delivery_ready(row)]
    if delivery_ready and len(delivery_ready) < len(rows):
        raw_best = max(rows, key=lambda row: _score_first(row, (metric_key,), minimize=minimize))
        if not _delivery_ready(raw_best):
            best_ready_score = (
                _score_first(delivery_ready[0], (metric_key,), minimize=minimize)
                if delivery_ready
                else float("-inf")
            )
            for row in delivery_ready[1:]:
                score = _score_first(row, (metric_key,), minimize=minimize)
                if score > best_ready_score:
                    best_ready_score = score
            for row in rows:
                if _delivery_ready(row):
                    continue
                row["delivery_excluded"] = True
                row_score = _score_first(row, (metric_key,), minimize=minimize)
                would_have_won = row_score > best_ready_score
                suffix = "该候选按此指标优于所有交付可用候选" if would_have_won else "即便不排除也非最优候选"
                row["delivery_excluded_reason"] = (
                    "不支持 PMML 导出和/或验证移交,已在默认交付可用性预过滤中排除,"
                    f"未参与冠军竞争({suffix})。"
                )
    if delivery_ready:
        rows = delivery_ready
    if target_type == "continuous":
        return max(rows, key=lambda row: _score_first(row, ("oot_rmse", "test_rmse"), minimize=True)), "oot_rmse"
    if target_type == "multiclass":
        auc_best = max(rows, key=lambda row: _score_first(row, ("oot_macro_auc", "test_macro_auc")))
        if _score_first(auc_best, ("oot_macro_auc", "test_macro_auc")) != float("-inf"):
            return auc_best, "oot_macro_auc"
        return max(rows, key=lambda row: _score_first(row, ("oot_logloss", "test_logloss"), minimize=True)), "oot_logloss"
    metric_score, selection_metric = _binary_selection_score_and_metric(eval_metric)
    return max(rows, key=metric_score), selection_metric


def _selection_metric_basis(target_type: str, *, eval_metric: str = "ks_auc") -> tuple[str, bool]:
    """The single metric key (and min/max direction) used to detect whether the
    delivery-ready pre-filter actually excluded a better-scoring candidate
    (SEL-7). Mirrors each target type's primary ranking metric in
    ``_pick_best_comparison_row`` -- binary's real selection key is the
    overfit-penalized test KS (not a plain column) by default, so it is
    approximated here by the raw test_ks column (DOM-6: test_lift_head_10 when
    the scenario's eval_metric is "response_lift"), which is what the pre-filter
    comparison cares about (relative ranking, not the exact champion score)."""
    target_type = str(target_type or "binary")
    if target_type == "continuous":
        return "oot_rmse", True
    if target_type == "multiclass":
        return "oot_macro_auc", False
    if str(eval_metric or "").strip() == "response_lift":
        return "test_lift_head_10", False
    return "test_ks", False


def _attach_capabilities_to_comparison_rows(runtime: _Runtime, rows: list[dict]) -> None:
    for row in rows:
        artifact_id = row.get("artifact_id")
        if not artifact_id:
            continue
        artifact = runtime.modeling_repo.get_model_artifact(str(artifact_id))
        if artifact is None:
            continue
        experiment = runtime.experiments.get(artifact.experiment_id)
        row["capabilities"] = _artifact_capabilities(
            artifact,
            base_dir=_artifact_base_dir(runtime.settings, experiment.task_id),
        )


def _attach_policy_profile_to_comparison_rows(runtime: _Runtime, rows: list[dict]) -> None:
    for row in rows:
        artifact_id = row.get("artifact_id")
        if not artifact_id:
            row["policy_profile"] = _row_policy_profile(row)
            continue
        artifact = runtime.modeling_repo.get_model_artifact(str(artifact_id))
        if artifact is None:
            row["policy_profile"] = _row_policy_profile(row)
            continue
        row["feature_count"] = len(artifact.feature_list)
        row["feature_list"] = list(artifact.feature_list)
        row["model_params"] = dict(artifact.params or {})
        row["scorecard_table"] = _scorecard_table_rows(artifact)
        row["policy_profile"] = _row_policy_profile(row)


def _delivery_ready(row: dict) -> bool:
    caps = row.get("capabilities") if isinstance(row.get("capabilities"), dict) else {}
    return bool(caps.get("pmml_supported") and caps.get("handoff_supported"))


def _normalize_selection_policy(raw) -> dict:
    source = raw if isinstance(raw, dict) else {}
    policy = {
        "require_pmml": _policy_bool(source.get("require_pmml")),
        "require_handoff": _policy_bool(source.get("require_handoff")),
        "require_scorecard": _policy_bool(source.get("require_scorecard")),
        "require_monotonicity": _policy_bool(source.get("require_monotonicity")),
        "prefer_scorecard": _policy_bool(source.get("prefer_scorecard")),
        "allow_policy_override": _policy_bool(source.get("allow_policy_override")),
        "override_reason": str(source.get("override_reason") or "").strip(),
        # SEL-7: default (non-opt-in) quality warnings -- overfit-gap and
        # feature-sanity checks that fire even when the caller requested no
        # other policy at all. Never block selection; disable_quality_warnings
        # is the explicit opt-out for a caller that wants the historical
        # PMML/handoff-only default behaviour back.
        "disable_quality_warnings": _policy_bool(source.get("disable_quality_warnings")),
    }
    max_feature_count = _positive_int_or_none(source.get("max_feature_count"))
    if max_feature_count is not None:
        policy["max_feature_count"] = max_feature_count
    max_oot_psi = _nonnegative_float_or_none(source.get("max_oot_psi"))
    if max_oot_psi is not None:
        policy["max_oot_psi"] = max_oot_psi
    overfit_gap_warn = _nonnegative_float_or_none(source.get("overfit_gap_warn_threshold"))
    policy["overfit_gap_warn_threshold"] = (
        overfit_gap_warn if overfit_gap_warn is not None else DEFAULT_OVERFIT_GAP_WARN_THRESHOLD
    )
    min_feature_count_warn = _positive_int_or_none(source.get("min_feature_count_warning"))
    policy["min_feature_count_warning"] = (
        min_feature_count_warn if min_feature_count_warn is not None else DEFAULT_MIN_FEATURE_COUNT_WARNING
    )
    metric_thresholds = _normalize_selection_policy_metric_thresholds(source)
    if metric_thresholds:
        policy["metric_thresholds"] = metric_thresholds
    return policy


def _policy_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _selection_policy_requested(policy: dict) -> bool:
    return any(
        bool(policy.get(key))
        for key in (
            "require_pmml",
            "require_handoff",
            "require_scorecard",
            "require_monotonicity",
            "prefer_scorecard",
        )
    ) or policy.get("max_feature_count") is not None or policy.get("max_oot_psi") is not None or bool(policy.get("metric_thresholds"))


def _selection_policy_has_hard_requirements(policy: dict) -> bool:
    return any(
        bool(policy.get(key))
        for key in (
            "require_pmml",
            "require_handoff",
            "require_scorecard",
            "require_monotonicity",
        )
    ) or policy.get("max_feature_count") is not None or policy.get("max_oot_psi") is not None or bool(policy.get("metric_thresholds"))


def _selection_policy_decision(row: dict, policy: dict, *, explicit: bool) -> dict:
    policy = _normalize_selection_policy(policy)
    profile = _row_policy_profile(row)
    requested = _selection_policy_requested(policy)
    violations = _selection_policy_violations(row, policy)
    missing_override_reason = bool(violations and policy.get("allow_policy_override") and not policy.get("override_reason"))
    if missing_override_reason:
        violations = [
            *violations,
            {
                "code": "override_reason_required",
                "message": "策略 override 必须填写 override_reason。",
            },
        ]
    if not requested:
        status = "not_requested"
    elif violations and policy.get("allow_policy_override") and not missing_override_reason:
        status = "overridden"
    elif violations:
        status = "blocked"
    else:
        status = "accepted"
    return {
        "status": status,
        "explicit_selection": bool(explicit),
        "selected_experiment_id": str(row.get("id") or ""),
        "policy": {
            key: value
            for key, value in policy.items()
            if value not in (None, "", False)
        },
        "profile": profile,
        "violations": violations,
        # SEL-7: non-blocking default quality guardrails -- never affect `status`.
        "warnings": _selection_policy_quality_warnings(row, policy),
        "override_reason": policy.get("override_reason") if status == "overridden" else "",
    }


def _selection_policy_quality_warnings(row: dict, policy: dict) -> list[dict]:
    """SEL-7 default quality guardrails: warn (never block) when a candidate's
    train-test KS gap looks like overfitting, or when it trained on
    suspiciously few features. Unlike ``_selection_policy_violations`` these
    fire by default (``disable_quality_warnings=True`` is the explicit opt-out)
    regardless of whether any other selection_policy field was requested --
    a caller who asked for nothing still gets these two sanity checks."""
    if policy.get("disable_quality_warnings"):
        return []
    warnings: list[dict] = []
    gap_threshold = policy.get("overfit_gap_warn_threshold", DEFAULT_OVERFIT_GAP_WARN_THRESHOLD)
    gap = _finite_float_or_none(row.get("overfit_train_test_gap"))
    if isinstance(gap_threshold, (int, float)) and gap is not None and gap > float(gap_threshold):
        warnings.append({
            "code": "overfit_warning",
            "message": (
                f"train-test KS 差距为 {_format_number_token(float(gap))},超过警戒阈值 "
                f"{_format_number_token(float(gap_threshold))},存在过拟合风险(未阻断选择)。"
            ),
        })
    min_feature_count = policy.get("min_feature_count_warning", DEFAULT_MIN_FEATURE_COUNT_WARNING)
    profile = _row_policy_profile(row)
    feature_count = profile.get("feature_count")
    if (
        isinstance(min_feature_count, int)
        and isinstance(feature_count, int)
        and feature_count < min_feature_count
    ):
        warnings.append({
            "code": "sanity_warning",
            "message": (
                f"入模特征数为 {feature_count},低于建议下限 {min_feature_count},"
                "模型稳健性存疑(未阻断选择)。"
            ),
        })
    return warnings


def _selection_policy_violations(row: dict, policy: dict) -> list[dict]:
    if not _selection_policy_requested(policy):
        return []
    profile = _row_policy_profile(row)
    violations: list[dict] = []
    if policy.get("require_pmml") and not profile.get("pmml_supported"):
        violations.append({
            "code": "require_pmml",
            "message": "要求最终模型支持 PMML 导出,但该候选不支持。",
        })
    if policy.get("require_handoff") and not profile.get("handoff_supported"):
        violations.append({
            "code": "require_handoff",
            "message": "要求最终模型支持验证移交,但该候选不支持。",
        })
    if policy.get("require_scorecard") and not profile.get("scorecard"):
        violations.append({
            "code": "require_scorecard",
            "message": "要求最终模型为评分卡,但该候选不是评分卡。",
        })
    if policy.get("require_monotonicity") and not profile.get("monotonicity_declared"):
        missing = profile.get("monotonicity_missing_features")
        if isinstance(missing, list) and missing:
            missing_text = ", ".join(str(item) for item in missing[:8])
            if len(missing) > 8:
                missing_text += ", ..."
            message = f"要求完整单调性证据,但以下特征缺少方向: {missing_text}。"
        else:
            message = "要求声明单调约束,但该候选缺少单调性证据。"
        violations.append({
            "code": "require_monotonicity",
            "message": message,
        })
    max_feature_count = policy.get("max_feature_count")
    feature_count = profile.get("feature_count")
    if isinstance(max_feature_count, int):
        if not isinstance(feature_count, int):
            violations.append({
                "code": "max_feature_count_missing",
                "message": f"要求特征数不超过 {max_feature_count},但该候选缺少特征数证据。",
            })
        elif feature_count > max_feature_count:
            violations.append({
                "code": "max_feature_count",
                "message": f"要求特征数不超过 {max_feature_count},但该候选有 {feature_count} 个特征。",
            })
    max_oot_psi = policy.get("max_oot_psi")
    oot_psi = profile.get("policy_psi_oot_vs_train")
    if isinstance(max_oot_psi, (int, float)):
        if not isinstance(oot_psi, (int, float)):
            violations.append({
                "code": "max_oot_psi_missing",
                "message": (
                    f"要求 OOT PSI 不超过 {_format_number_token(float(max_oot_psi))},"
                    "但该候选缺少 OOT PSI 证据。"
                ),
            })
        elif oot_psi > float(max_oot_psi):
            psi_source = str(profile.get("policy_psi_source") or "psi_oot_vs_train")
            psi_label = "加权 OOT PSI" if psi_source == "weighted_psi_oot_vs_train" else "OOT PSI"
            violations.append({
                "code": "max_oot_psi",
                "message": (
                    f"要求 OOT PSI 不超过 {_format_number_token(float(max_oot_psi))},"
                    f"但该候选{psi_label}为 {_format_number_token(float(oot_psi))}。"
                ),
            })
    violations.extend(_selection_policy_metric_threshold_violations(row, policy.get("metric_thresholds")))
    return violations


def _normalize_selection_policy_metric_thresholds(source: dict) -> dict[str, dict[str, float]]:
    thresholds: dict[str, dict[str, float]] = {}
    raw_thresholds = source.get("metric_thresholds")
    if isinstance(raw_thresholds, dict):
        for raw_metric, raw_spec in raw_thresholds.items():
            metric = _policy_metric_name(raw_metric)
            if not metric or not isinstance(raw_spec, dict):
                continue
            spec: dict[str, float] = {}
            minimum = _finite_float_or_none(raw_spec.get("min"))
            maximum = _finite_float_or_none(raw_spec.get("max"))
            if minimum is not None:
                spec["min"] = minimum
            if maximum is not None:
                spec["max"] = maximum
            if spec:
                thresholds[metric] = spec
    for key, (metric, direction) in _POLICY_METRIC_THRESHOLD_SHORTCUTS.items():
        value = _finite_float_or_none(source.get(key))
        if value is None:
            continue
        thresholds.setdefault(metric, {})[direction] = value
    return thresholds


def _selection_policy_metric_threshold_violations(row: dict, thresholds) -> list[dict]:
    if not isinstance(thresholds, dict) or not thresholds:
        return []
    violations: list[dict] = []
    for metric in sorted(thresholds):
        spec = thresholds.get(metric)
        if not isinstance(spec, dict):
            continue
        value = _finite_float_or_none(row.get(metric))
        if value is None:
            violations.append({
                "code": "metric_threshold_missing",
                "metric": metric,
                "message": f"要求指标 `{metric}` 满足阈值,但该候选缺少该指标证据。",
            })
            continue
        minimum = spec.get("min")
        maximum = spec.get("max")
        if isinstance(minimum, (int, float)) and value < float(minimum):
            violations.append({
                "code": "metric_min_threshold",
                "metric": metric,
                "message": (
                    f"要求 `{metric}` 不低于 {_format_number_token(float(minimum))},"
                    f"但该候选为 {_format_number_token(float(value))}。"
                ),
            })
        if isinstance(maximum, (int, float)) and value > float(maximum):
            violations.append({
                "code": "metric_max_threshold",
                "metric": metric,
                "message": (
                    f"要求 `{metric}` 不超过 {_format_number_token(float(maximum))},"
                    f"但该候选为 {_format_number_token(float(value))}。"
                ),
            })
    return violations


def _policy_metric_name(value) -> str:
    metric = str(value or "").strip()
    if not _POLICY_METRIC_NAME.fullmatch(metric):
        return ""
    return metric


def _selection_policy_block_message(experiment_id: str, decision: dict) -> str:
    reasons = "; ".join(
        f"{item.get('code')}: {item.get('message') or ''}".strip()
        for item in decision.get("violations", [])
        if isinstance(item, dict)
    )
    suffix = f" {reasons}" if reasons else ""
    return (
        f"selected_experiment_id violates selection_policy: {experiment_id}.{suffix} "
        "Set allow_policy_override=true with override_reason to keep this candidate."
    )


def _no_policy_candidate_message(rows: list[dict], policy: dict) -> str:
    details = []
    for row in rows[:5]:
        violations = _selection_policy_violations(row, policy)
        if not violations:
            continue
        details.append(
            f"{row.get('id') or '?'}: "
            + ", ".join(str(item.get("code") or "?") for item in violations if isinstance(item, dict))
        )
    suffix = f" Candidates: {'; '.join(details)}" if details else ""
    return (
        "no experiment satisfies selection_policy. "
        "Relax the policy, retrain a compliant candidate, or set allow_policy_override=true "
        f"with override_reason.{suffix}"
    )


def _row_policy_profile(row: dict) -> dict:
    item = row if isinstance(row, dict) else {}
    caps = item.get("capabilities") if isinstance(item.get("capabilities"), dict) else {}
    scorecard_rows = item.get("scorecard_table") if isinstance(item.get("scorecard_table"), list) else []
    recipe = str(item.get("recipe") or "")
    features = item.get("feature_list") if isinstance(item.get("feature_list"), list) else item.get("features")
    feature_count = item.get("feature_count")
    if not isinstance(feature_count, int) and isinstance(features, list):
        feature_count = len(features)
    monotonic = monotonic_policy_profile(item, scorecard_rows)
    profile = {
        "recipe": recipe,
        "scorecard": recipe == "scorecard" or bool(scorecard_rows),
        "scorecard_table_rows": len(scorecard_rows),
        "monotonicity_declared": bool(monotonic.get("monotonicity_declared")),
        "monotonicity_coverage": monotonic.get("monotonicity_coverage"),
        "monotonicity_missing_features": monotonic.get("monotonicity_missing_features") or [],
        "monotonicity_constrained_features": monotonic.get("monotonicity_constrained_features") or [],
        "pmml_supported": bool(caps.get("pmml_supported")),
        "handoff_supported": bool(caps.get("handoff_supported")),
        "native_model_supported": bool(caps.get("native_model_supported")),
        "feature_count": feature_count if isinstance(feature_count, int) else None,
    }
    psi_oot = _finite_float_or_none(item.get("psi_oot_vs_train"))
    weighted_psi_oot = _finite_float_or_none(item.get("weighted_psi_oot_vs_train"))
    if psi_oot is not None:
        profile["psi_oot_vs_train"] = psi_oot
    if weighted_psi_oot is not None:
        profile["weighted_psi_oot_vs_train"] = weighted_psi_oot
        profile["policy_psi_oot_vs_train"] = weighted_psi_oot
        profile["policy_psi_source"] = "weighted_psi_oot_vs_train"
    elif psi_oot is not None:
        profile["policy_psi_oot_vs_train"] = psi_oot
        profile["policy_psi_source"] = "psi_oot_vs_train"
    return profile


def _row_has_monotonic_policy(item: dict, scorecard_rows: list) -> bool:
    return has_monotonic_policy(item, scorecard_rows)
