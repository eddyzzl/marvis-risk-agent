"""Deterministic red-flag checklists for modeling gates (AGT-9).

The JOIN/screening gates already get a platform-computed 【平台红旗 checklist】
(``auto_drive._extract_red_flags``) so the LLM re-checks numbers instead of
inventing its own. That coverage stopped at JOIN — the two most consequential
modeling decision gates (which tuning-config funnel to accept, which trained
experiment to select) handed AUTO a bare table of KS/AUC numbers and left the
comparisons to the model itself, exactly the arithmetic a weak model is worst
at.

Both functions here read ONLY numbers a platform tool already computed
(INV-1: the LLM re-checks, it never computes) and return plain Chinese
sentences meant to sit in a gate message's ``metadata['red_flags']`` /
``decide_gate`` prompt, next to the existing JOIN/screen red flags.
"""

from __future__ import annotations

# 调参配置门 ("精选特征" gate, pauses before "配置调参" runs): flags derived from
# the split ("切分样本"/make_split) + modeling-spec ("选择建模规格"/
# choose_modeling_spec) outputs, both direct dependencies of that gate.
MIN_SAMPLE_SIZE = 5000
MAX_FEATURE_COUNT = 200

# 选实验门 ("选择实验" gate): flags derived from the tuning ("调参"/
# tune_hyperparameters) + training ("训练模型"/train_models) outputs, both
# direct dependencies of that gate.
MAX_TRAIN_TEST_KS_GAP = 0.10
MIN_CHAMPION_RUNNER_UP_KS_GAP = 0.005


def tuning_setup_red_flags(*, split_output: dict | None, modeling_spec_output: dict | None) -> list[str]:
    """Red flags for the tuning-config gate (samples too small, too many
    features, OOT missing) — the checks named in AGT-9's 调参门 list that this
    template's gate can actually see (make_split + choose_modeling_spec are
    both direct dependencies of that gate)."""
    flags: list[str] = []
    analysis = _dict(split_output).get("sample_analysis")
    split_counts = _split_counts(analysis)
    total_rows = _safe_int(_dict(analysis).get("total_rows"))
    if total_rows is None and split_counts:
        total_rows = sum(split_counts.values())
    if total_rows is not None and total_rows < MIN_SAMPLE_SIZE:
        flags.append(f"样本量偏小（共 {total_rows} 行 < {MIN_SAMPLE_SIZE}），指标抽样噪声可能较大。")
    if split_counts and split_counts.get("oot", 0) <= 0:
        flags.append("样本切分缺少 OOT（时间外）子集，稳定性结论需谨慎。")
    feature_count = _safe_int(_dict(modeling_spec_output).get("feature_count"))
    if feature_count is not None and feature_count > MAX_FEATURE_COUNT:
        flags.append(f"入模候选特征数偏多（{feature_count} > {MAX_FEATURE_COUNT}），过拟合与训练成本风险上升。")
    return flags


def select_experiment_red_flags(*, tune_output: dict | None, train_models_output: dict | None) -> list[str]:
    """Red flags for the select-experiment gate (train-test overfit gap,
    champion vs. runner-up margin too thin to be meaningful, weighted vs.
    unweighted champion disagreement, any failed candidate)."""
    flags: list[str] = []
    flags.extend(_overfit_gap_flags(tune_output))
    experiments = [
        item for item in _dict(train_models_output).get("experiments") or [] if isinstance(item, dict)
    ]
    flags.extend(_champion_runner_up_flags(experiments))
    flags.extend(_weighted_unweighted_mismatch_flags(experiments))
    failed = [item for item in _dict(train_models_output).get("failed") or [] if isinstance(item, dict)]
    if failed:
        recipes = ", ".join(str(item.get("recipe") or "?") for item in failed[:8])
        flags.append(f"训练阶段有 {len(failed)} 个候选算法失败（{recipes}），对比范围不完整。")
    return flags


def _overfit_gap_flags(tune_output: dict | None) -> list[str]:
    trials = [item for item in _dict(tune_output).get("trials") or [] if isinstance(item, dict)]
    worst_gap = None
    for trial in trials:
        train_ks = _finite(trial.get("train_ks"))
        test_ks = _finite(trial.get("test_ks"))
        if train_ks is None or test_ks is None:
            continue
        gap = train_ks - test_ks
        if worst_gap is None or gap > worst_gap:
            worst_gap = gap
    if worst_gap is not None and worst_gap > MAX_TRAIN_TEST_KS_GAP:
        return [
            f"调参 trial 中最大 train-test KS 差为 {worst_gap:.3f}（> {MAX_TRAIN_TEST_KS_GAP}），存在过拟合迹象。"
        ]
    return []


def _champion_runner_up_flags(experiments: list[dict]) -> list[str]:
    scores = sorted(
        (score for score in (_champion_score(item) for item in experiments) if score is not None),
        reverse=True,
    )
    if len(scores) < 2:
        return []
    gap = scores[0] - scores[1]
    if gap < MIN_CHAMPION_RUNNER_UP_KS_GAP:
        return [
            f"冠军与亚军的 test_ks 差距仅 {gap:.4f}（< {MIN_CHAMPION_RUNNER_UP_KS_GAP}），"
            "冠军选择可能落在噪声范围内，建议复核而非直接采信。"
        ]
    return []


def _weighted_unweighted_mismatch_flags(experiments: list[dict]) -> list[str]:
    weighted_best = _best_recipe(experiments, "weighted_test_ks")
    unweighted_best = _best_recipe(experiments, "test_ks")
    if weighted_best and unweighted_best and weighted_best != unweighted_best:
        return [
            f"按加权 test_ks 与未加权 test_ks 选出的冠军算法不一致"
            f"（加权={weighted_best}，未加权={unweighted_best}），请确认样本权重是否应纳入选型口径。"
        ]
    return []


def _champion_score(experiment: dict) -> float | None:
    metrics = experiment.get("metrics") if isinstance(experiment.get("metrics"), dict) else {}
    value = metrics.get("weighted_test_ks")
    if not isinstance(value, (int, float)):
        value = metrics.get("test_ks")
    return float(value) if isinstance(value, (int, float)) else None


def _best_recipe(experiments: list[dict], metric_key: str) -> str | None:
    best_recipe = None
    best_value = None
    for item in experiments:
        metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        value = metrics.get(metric_key)
        if not isinstance(value, (int, float)):
            continue
        if best_value is None or value > best_value:
            best_value = value
            best_recipe = str(item.get("recipe") or "") or None
    return best_recipe


def _split_counts(analysis) -> dict[str, int]:
    counts: dict[str, int] = {}
    raw = _dict(analysis).get("split_counts")
    for key, value in (raw.items() if isinstance(raw, dict) else ()):
        number = _safe_int(value)
        if number is not None:
            counts[str(key).lower()] = number
    return counts


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _safe_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _finite(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


__all__ = ["select_experiment_red_flags", "tuning_setup_red_flags"]
