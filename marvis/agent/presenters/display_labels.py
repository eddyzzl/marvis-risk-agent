"""Business-readable labels for platform codes shown in conversation cards.

Display only: tool outputs keep their machine codes (other code looks them
up); these helpers translate what the user reads.  Nothing here recomputes
or judges a metric.
"""

from __future__ import annotations

import math
import re
from pathlib import PurePath
from collections.abc import Mapping
from typing import Any


TARGET_TYPE_LABELS = {"binary": "二分类", "continuous": "连续值", "multiclass": "多分类"}

RECIPE_LABELS = {
    "lr": "逻辑回归",
    "lgb": "LightGBM",
    "xgb": "XGBoost",
    "catboost": "CatBoost",
    "scorecard": "评分卡",
    "mlp": "神经网络",
    "lr_regressor": "线性回归",
    "lgb_regressor": "LightGBM 回归",
    "xgb_regressor": "XGBoost 回归",
    "mlp_regressor": "神经网络回归",
    "lr_multiclass": "逻辑回归多分类",
    "lgb_multiclass": "LightGBM 多分类",
    "xgb_multiclass": "XGBoost 多分类",
    "mlp_multiclass": "神经网络多分类",
    "ensemble": "集成模型",
}

_SPLIT_LABELS = {"train": "训练集", "test": "测试集", "oot": "OOT", "cv": "交叉验证"}
_MEASURE_LABELS = {
    "ks": "KS",
    "auc": "AUC",
    "rmse": "RMSE",
    "mae": "MAE",
    "r2": "R²",
    "logloss": "Logloss",
    "accuracy": "准确率",
    "macro_auc": "Macro-AUC",
    "rows": "样本数",
    "count": "样本数",
    "bad_rate": "坏样本率",
    "lift_head_5": "头部 5% Lift",
    "lift_head_10": "头部 10% Lift",
    "lift_tail_5": "尾部 5% Lift",
    "lift_tail_10": "尾部 10% Lift",
    "ks_ci_low": "KS 置信下限",
    "ks_ci_high": "KS 置信上限",
    "ks_ci_std": "KS 置信区间标准差",
    "mean_ks": "KS 均值",
    "std_ks": "KS 标准差",
}
_METRIC_LABELS = {
    "ks": "KS",
    "auc": "AUC",
    "rmse": "RMSE",
    "mae": "MAE",
    "r2": "R²",
    "logloss": "Logloss",
    "accuracy": "准确率",
    "feature_count": "特征数",
    "n_features": "特征数",
    "psi_test_vs_train": "PSI（测试集 vs 训练集）",
    "psi_oot_vs_train": "PSI（OOT vs 训练集）",
    "overfit_flag": "过拟合标记",
    "overfit_train_test_gap": "训练/测试 KS 相对差距",
    "overfit_gap_tt": "过拟合差（训练-测试）",
    "overfit_gap_to": "过拟合差（训练-OOT）",
    "ks_ci_n_boot": "置信区间抽样次数",
    "score": "选优得分",
    "selection_score": "选优得分",
    "selection_metric": "选优指标",
    "deterministic": "结果可复现",
    "cv_fold_scores": "交叉验证各折得分",
    "cv_score_std": "交叉验证得分标准差",
    "cv_fold_test_ks": "交叉验证各折测试集 KS",
    "test_ks_std": "各折测试集 KS 标准差",
    "overfit_train_oot_gap": "训练/OOT KS 差",
    "oot_stability_gap": "OOT 稳定性差距",
    "lift_head_5": "头部 5% Lift",
    "lift_head_10": "头部 10% Lift",
    "lift_tail_5": "尾部 5% Lift",
    "lift_tail_10": "尾部 10% Lift",
    "search_stage": "搜索阶段",
}
_SEARCH_STAGE_LABELS = {"coarse": "粗搜", "fine": "细搜", "single": "单轮搜索"}
_COUNT_SUFFIXES = ("_count", "_rows", "_n_boot")


def report_role_suffix(output: Any, experiment_id: Any) -> str:
    """After a champion refit the reports cover the delivered model and the
    experiment it was refit from; both are the same algorithm, so the label
    says which is which (eval S4 showed two identical 「逻辑回归」 reports)."""

    scope = output.get("development_refit_scope") if isinstance(output, Mapping) else None
    if not isinstance(scope, Mapping):
        return ""
    current = str(experiment_id or "").strip()
    if current and current == str(scope.get("final_experiment_id") or "").strip():
        return "（交付模型）"
    if current and current == str(scope.get("parent_selection_experiment_id") or "").strip():
        return "（重新拟合前的所选实验）"
    return ""


_RECIPE_CODE_RE = None


def readable_recipe_text(text: Any) -> str:
    """Tool prose such as 「lr 使用有界随机搜索…按算法预算:lr=1」 with recipe codes
    as algorithm names (「逻辑回归 … 逻辑回归 1 轮」). Codes inside longer words stay."""

    global _RECIPE_CODE_RE
    if _RECIPE_CODE_RE is None:
        codes = "|".join(re.escape(code) for code in sorted(RECIPE_LABELS, key=len, reverse=True))
        _RECIPE_CODE_RE = re.compile(rf"`?(?<![A-Za-z0-9_])({codes})(?![A-Za-z0-9_])`?(?:=(\d+))?")

    def replace_code(match: re.Match) -> str:
        label = RECIPE_LABELS[match.group(1)]
        return f"{label} {match.group(2)} 轮" if match.group(2) else label

    return _RECIPE_CODE_RE.sub(replace_code, str(text or ""))


def recipe_label(code: Any) -> str:
    text = str(code or "").strip()
    return RECIPE_LABELS.get(text, text or "-")


def target_type_label(code: Any) -> str:
    text = str(code or "").strip()
    return TARGET_TYPE_LABELS.get(text, text or "-")


def metric_label(key: Any) -> str:
    """``oot_ks`` -> ``OOT KS``; unknown keys stay as written."""

    text = str(key or "")
    if text in _METRIC_LABELS:
        return _METRIC_LABELS[text]
    weighted = text.startswith("weighted_")
    body = text[len("weighted_"):] if weighted else text
    split, _, measure = body.partition("_")
    if split in _SPLIT_LABELS and measure in _MEASURE_LABELS:
        label = f"{_SPLIT_LABELS[split]} {_MEASURE_LABELS[measure]}"
        return f"加权{label}" if weighted else label
    return text


def is_count_metric(key: Any) -> bool:
    text = str(key or "")
    return text in {"feature_count", "n_features", "ks_ci_n_boot"} or text.endswith(_COUNT_SUFFIXES)


def metric_value_text(key: Any, value: Any) -> str:
    """Counts read as integers, booleans as 是/否, ratios to 4 decimals."""

    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if str(key or "") == "selection_metric":
        return selection_metric_label(value)
    if str(key or "") == "search_stage":
        return _SEARCH_STAGE_LABELS.get(str(value), str(value))
    if isinstance(value, (list, tuple)) and value and all(
        isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
    ):
        return "、".join(metric_value_text(key, item) for item in value)
    if isinstance(value, (int, float)) and is_count_metric(key):
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    if isinstance(value, float):
        return f"{value:.4f}" if math.isfinite(value) else "—"
    return str(value)


def money_text(value: Any) -> str:
    """Amounts read as money: thousands separators, two decimals."""

    if value is None or isinstance(value, bool):
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:,.2f}" if math.isfinite(number) else "n/a"


_SELECTION_METRIC_LABELS = {
    "manual": "用户指定",
    "test_ks(overfit-penalized)": "测试集 KS（扣除过拟合惩罚）",
    "test_rmse(overfit-penalized)": "测试集 RMSE（扣除过拟合惩罚）",
    "test_logloss(overfit-penalized)": "测试集 Logloss（扣除过拟合惩罚）",
    "cv_mean_ks_minus_std_penalty": "交叉验证 KS 均值减波动惩罚",
    "ks_auc": "KS 与 AUC",
}


def selection_metric_label(code: Any) -> str:
    text = str(code or "").strip()
    if not text:
        return "-"
    if text in _SELECTION_METRIC_LABELS:
        return _SELECTION_METRIC_LABELS[text]
    label = metric_label(text)
    return label if label != text else text


_METRIC_POLICY_LABELS = {
    "higher overfit-penalized test KS; OOT reported only, not used for selection": (
        "按测试集 KS（扣除过拟合惩罚）择优；OOT 只用于展示，不参与选择"
    ),
    "lower OOT RMSE, fallback lower test RMSE": "按 OOT RMSE 越低越好择优；无 OOT 时看测试集 RMSE",
    "higher OOT macro-AUC, fallback higher test macro-AUC then lower logloss": (
        "按 OOT Macro-AUC 越高越好择优；无 OOT 时看测试集 Macro-AUC，再看 Logloss"
    ),
}


def metric_policy_label(policy: Any) -> str:
    text = str(policy or "").strip()
    if not text:
        return "-"
    return _METRIC_POLICY_LABELS.get(text) or selection_metric_label(text)


_TARGET_FAMILY_MISMATCH = re.compile(r"^recipe target family does not match `?([a-z]+)`?$")


def disabled_algorithm_reason(reason: Any) -> str:
    text = str(reason or "").strip()
    match = _TARGET_FAMILY_MISMATCH.match(text)
    if match:
        return f"该算法不适用于{target_type_label(match.group(1))}目标"
    return text


_NUMBER = r"(-?\d+(?:\.\d+)?)"
_SCREEN_REASON_RULES: tuple[tuple[re.Pattern[str], Any], ...] = (
    (
        re.compile(rf"^univariate KS {_NUMBER} >= {_NUMBER} — suspected target leakage$"),
        lambda m: f"单变量 KS {m.group(1)} ≥ {m.group(2)}，疑似目标泄漏",
    ),
    (
        re.compile(r"^semantic target leakage: protected target column cannot be a model feature$"),
        lambda m: "这是目标列本身，不能作为入模特征",
    ),
    (
        re.compile(r"^semantic/temporal target leakage（语义/时序目标泄漏）：(.*)$"),
        lambda m: f"语义/时序目标泄漏：{m.group(1)}",
    ),
    (
        re.compile(
            r"^deterministic outcome subgroup leakage: value (.*?) implies target=(\S+) "
            r"in train fold A (\d+) rows and train fold B (\d+) rows"
        ),
        lambda m: (
            f"取值 {m.group(1)} 在训练集两个子样本中都 100% 对应目标={m.group(2)}"
            f"（{m.group(3)} 行 / {m.group(4)} 行），疑似目标构造别名或表现期后字段"
        ),
    ),
    (
        re.compile(r"^protected split column:"),
        lambda m: "样本切分控制列，不作为入模特征",
    ),
    (
        re.compile(r"^protected sample-weight column:"),
        lambda m: "样本权重列，不作为入模特征",
    ),
    (
        re.compile(r"^identity/row key:"),
        lambda m: "主键/流水号类标识列，按命名规则排除",
    ),
    (
        re.compile(rf"^missing rate {_NUMBER} >= {_NUMBER}$"),
        lambda m: f"缺失率 {m.group(1)} ≥ {m.group(2)}",
    ),
    (
        re.compile(r"^only (\d+) distinct non-null value\(s\)$"),
        lambda m: f"非空取值只有 {m.group(1)} 种",
    ),
    (re.compile(r"^high_missing$"), lambda m: "缺失率过高"),
    (re.compile(r"^constant$"), lambda m: "取值单一（常量）"),
    (
        re.compile(r"^name looks like a model output/score"),
        lambda m: "列名像模型输出/评分，请确认是否允许作为入模特征",
    ),
    (
        re.compile(rf"^leakage_watch: univariate KS {_NUMBER} in \[{_NUMBER}, {_NUMBER}\)"),
        lambda m: f"单变量 KS {m.group(1)} 偏强（{m.group(2)}–{m.group(3)}），请确认不是部分/条件泄漏",
    ),
    (
        re.compile(rf"^split_shift: \|ks_train {_NUMBER} - ks_test {_NUMBER}\| = {_NUMBER} > {_NUMBER}"),
        lambda m: (
            f"训练集与测试集 KS 相差 {m.group(3)}（{m.group(1)} → {m.group(2)}，超过 {m.group(4)}），"
            "请确认字段在时间上稳定"
        ),
    ),
    (
        re.compile(rf"^ks_decay {_NUMBER} < {_NUMBER} \(ks_train {_NUMBER} -> ks_test {_NUMBER}\)"),
        lambda m: f"KS 样本外衰减到 {m.group(1)}（低于 {m.group(2)}；{m.group(3)} → {m.group(4)}），请确认稳定性",
    ),
    (
        re.compile(rf"^psi_split {_NUMBER} >= {_NUMBER}"),
        lambda m: f"训练集与留出集 PSI {m.group(1)} ≥ {m.group(2)}，时间漂移偏大，请确认稳定性",
    ),
)


def screening_reason_label(reason: Any) -> str:
    """Chinese reading of a feature-screening reason; unknown text unchanged."""

    text = str(reason or "").strip()
    for pattern, render in _SCREEN_REASON_RULES:
        match = pattern.match(text)
        if match:
            return render(match)
    return text


_ARTIFACT_ID = re.compile(r"`?artifact_[0-9a-f]{8,}`?")
_RECOMMENDED_OTHER = re.compile(r"平台推荐为\s*`?experiment_[0-9a-f]{8,}`?")
_EXPERIMENT_IN_PROSE = re.compile(r"\s*`?experiment_[0-9a-f]{8,}`?\s*")
_ARTIFACT_IN_PROSE = re.compile(r"\s*`?artifact_[0-9a-f]{8,}`?\s*")


def scrub_internal_ids(text: Any, *, experiment: str = "", artifact: str = "") -> str:
    """Drop experiment/artifact ids from prose (``所选实验 `experiment_…` 的`` ->
    ``所选实验的``); the ids stay in the message metadata."""

    value = _RECOMMENDED_OTHER.sub("平台推荐的是另一个实验", str(text or ""))
    value = _EXPERIMENT_IN_PROSE.sub(experiment, value)
    return _ARTIFACT_IN_PROSE.sub(artifact, value)


def readable_selection_reason(text: Any) -> str:
    """Selection reason without experiment ids or raw metric codes."""

    value = scrub_internal_ids(text)
    for code, label in _SELECTION_METRIC_LABELS.items():
        if code != "manual":
            value = value.replace(code, label)
    return _METRIC_CODE.sub(lambda match: metric_label(match.group(0)), value)


_METRIC_CODE = re.compile(r"\b(?:weighted_)?(?:train|test|oot|cv)_[a-z0-9_]+\b")


def file_display_name(path: Any) -> str:
    """Only the file name of a stored path; internal ids become a plain noun."""

    text = str(path or "").strip()
    if not text:
        return ""
    name = PurePath(text.replace("\\", "/")).name
    return "" if _ARTIFACT_ID.fullmatch(PurePath(name).stem) else name


_DOCUMENT_KINDS = {".xlsx": "Excel 报告", ".xls": "Excel 报告", ".docx": "Word 报告", ".md": "Markdown 文档",
                   ".json": "JSON 数据", ".pmml": "PMML 文件", ".joblib": "模型文件", ".pkl": "模型文件",
                   ".csv": "CSV 文件", ".parquet": "数据文件"}


def document_kind_label(path: Any) -> str:
    """A stored file named by what it is (``Excel 报告``), not its hashed name."""

    text = str(path or "").strip()
    if not text:
        return ""
    return _DOCUMENT_KINDS.get(PurePath(text.replace("\\", "/")).suffix.lower(), "文件")


__all__ = [
    "RECIPE_LABELS",
    "TARGET_TYPE_LABELS",
    "disabled_algorithm_reason",
    "document_kind_label",
    "file_display_name",
    "is_count_metric",
    "metric_label",
    "metric_policy_label",
    "metric_value_text",
    "money_text",
    "readable_selection_reason",
    "recipe_label",
    "screening_reason_label",
    "scrub_internal_ids",
    "selection_metric_label",
    "target_type_label",
]
