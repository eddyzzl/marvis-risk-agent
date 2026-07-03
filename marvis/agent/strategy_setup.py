"""Setup (slot-filling) for the strategy task.

Strategy analysis starts from one scored sample: a binary target column plus a
score/probability column. This module discovers/registers the dataset and builds
a conservative default approval strategy candidate, then the PlanDriver pauses
before backtesting so the user can confirm or replan the rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from marvis.agent.sample_setup import detect_setup
from marvis.domain import FileRole
from marvis.files import scan_source_dir

_DATA_ROLES = frozenset({FileRole.SAMPLE.value, "sample", "strategy_sample"})
_SCORE_HINTS = (
    "score",
    "pred",
    "prediction",
    "prob",
    "probability",
    "pd",
    "risk_score",
    "model_score",
    "credit_score",
)


# S4: goal phrases that route a strategy task to the rule_strategy template
# (rule mining) instead of the default strategy_analysis. Kept in sync with
# RULE_STRATEGY.goal_patterns; the strategy_setup intent branch multi-recognizes
# these -- parallel to how strategy_development got its own goal_patterns (S2).
_RULE_STRATEGY_GOAL_PATTERNS = ("规则挖掘", "拒绝规则", "规则策略", "rule mining", "rule strategy")


def is_rule_strategy_goal(*texts: str | None) -> bool:
    haystack = " ".join(text.lower() for text in texts if text)
    return any(pattern.lower() in haystack for pattern in _RULE_STRATEGY_GOAL_PATTERNS)


class StrategySetupError(ValueError):
    """Raised when a strategy task cannot infer a scored binary sample."""


@dataclass
class StrategyProposal:
    dataset_id: str
    dataset_name: str
    target_col: str
    score_col: str
    strategy_type: str
    rules: list[dict]
    default_decision: str
    cutoff: float
    direction: str
    bad_rate: float | None
    notes: list[str]
    template_id: str = "strategy_analysis"

    def template_slots(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "target_col": self.target_col,
            "score_col": self.score_col,
            "strategy_type": self.strategy_type,
            "rules": self.rules,
            "default_decision": self.default_decision,
        }


@dataclass
class RuleStrategyProposal:
    """S4: setup proposal that routes to the rule_strategy template. Unlike the
    lightweight strategy_analysis proposal it carries no pre-built rules -- the
    template's mine_rules step discovers them -- only the dataset/target anchor
    and the rule-mining slots. adoption_reason is a placeholder the user reviews
    at the mandatory adopt gate (the whole point of that forced gate)."""

    dataset_id: str
    dataset_name: str
    target_col: str
    score_col: str | None
    bad_rate: float | None
    notes: list[str]
    template_id: str = "rule_strategy"

    def template_slots(self) -> dict:
        slots: dict = {
            "dataset_id": self.dataset_id,
            "target_col": self.target_col,
            "adoption_reason": "（待采纳时确认）",
        }
        if self.score_col:
            slots["score_col"] = self.score_col
        return slots


def build_strategy_proposal(
    registry,
    backend,
    task_id: str,
    source_dir,
    *,
    target_col: str | None = None,
    score_col: str | None = None,
) -> StrategyProposal:
    dataset = _resolve_dataset(registry, task_id, source_dir)
    path = registry.resolve_path(dataset.id)
    columns = backend.column_names(path)
    resolved_target = _resolve_target_col(backend, path, columns, target_col)
    resolved_score = _resolve_score_col(columns, score_col)
    if not resolved_score.isidentifier():
        raise StrategySetupError(
            f"策略条件暂只支持 Python 标识符列名；评分列 `{resolved_score}` 需先重命名后再回测。"
        )
    frame = backend.read_frame(path, columns=[resolved_target, resolved_score])
    profile = _score_profile(frame, target_col=resolved_target, score_col=resolved_score)
    rule = {
        "condition": profile["condition"],
        "decision": "reject",
    }
    return StrategyProposal(
        dataset_id=dataset.id,
        dataset_name=_dataset_name(dataset),
        target_col=resolved_target,
        score_col=resolved_score,
        strategy_type="approval",
        rules=[rule],
        default_decision="approve",
        cutoff=profile["cutoff"],
        direction=profile["direction"],
        bad_rate=profile["bad_rate"],
        notes=profile["notes"],
    )


def build_rule_strategy_proposal(
    registry,
    backend,
    task_id: str,
    source_dir,
    *,
    target_col: str | None = None,
    score_col: str | None = None,
) -> RuleStrategyProposal:
    """S4: resolve the dataset/target anchor for a rule-mining task. Score is
    optional here (rule mining works on arbitrary numeric features, not just a
    single score); when present it is passed through so build_strategy's rule
    direction self-check can fire on any score-band rules."""
    dataset = _resolve_dataset(registry, task_id, source_dir)
    path = registry.resolve_path(dataset.id)
    columns = backend.column_names(path)
    resolved_target = _resolve_target_col(backend, path, columns, target_col)
    resolved_score = _optional_score_col(columns, score_col)
    bad_rate = _target_bad_rate(backend, path, resolved_target)
    notes = ["将在数据上挖掘候选拒绝规则，选定规则集后回测并采纳。"]
    return RuleStrategyProposal(
        dataset_id=dataset.id,
        dataset_name=_dataset_name(dataset),
        target_col=resolved_target,
        score_col=resolved_score,
        bad_rate=bad_rate,
        notes=notes,
    )


def _optional_score_col(columns: list[str], requested: str | None) -> str | None:
    requested = str(requested or "").strip()
    if requested and requested in columns:
        return requested if requested.isidentifier() else None
    return None


def _target_bad_rate(backend, path, target_col: str) -> float | None:
    try:
        frame = backend.read_frame(path, columns=[target_col])
    except Exception:
        return None
    target = pd.to_numeric(frame[target_col], errors="coerce").dropna()
    if target.empty:
        return None
    return float((target == 1).mean())


def _resolve_dataset(registry, task_id: str, source_dir):
    datasets = [d for d in registry.list_for_task(task_id) if d.role in _DATA_ROLES]
    if not datasets and source_dir is not None:
        for artifact in scan_source_dir(Path(source_dir)):
            if artifact.role == FileRole.SAMPLE:
                registry.register_from_upload(task_id, Path(artifact.path), role="sample")
        datasets = [d for d in registry.list_for_task(task_id) if d.role in _DATA_ROLES]
    if not datasets:
        raise StrategySetupError(f"策略分析未找到数据文件:{source_dir}")
    return sorted(
        datasets,
        key=lambda d: (not bool(getattr(d, "has_target", False)), -int(getattr(d, "row_count", 0) or 0)),
    )[0]


def _resolve_target_col(backend, path: Path, columns: list[str], requested: str | None) -> str:
    requested = str(requested or "").strip()
    if requested and requested in columns:
        return requested
    setup = detect_setup(backend, path)
    if setup.target_col:
        return setup.target_col
    raise StrategySetupError("未能识别 0/1 目标列；请在创建任务时指定 target_col。")


def _resolve_score_col(columns: list[str], requested: str | None) -> str:
    requested = str(requested or "").strip()
    if requested and requested in columns:
        return requested
    lowered = {column.lower(): column for column in columns}
    for hint in _SCORE_HINTS:
        if hint in lowered:
            return lowered[hint]
    for column in columns:
        low = column.lower()
        if "score" in low or low in {"pred", "pd"} or "prob" in low:
            return column
    raise StrategySetupError("未能识别评分列；请在创建任务时指定 score_col。")


def _score_profile(frame: pd.DataFrame, *, target_col: str, score_col: str) -> dict:
    clean = frame[[target_col, score_col]].copy()
    clean[target_col] = pd.to_numeric(clean[target_col], errors="coerce")
    clean[score_col] = pd.to_numeric(clean[score_col], errors="coerce")
    clean = clean.dropna()
    if clean.empty:
        raise StrategySetupError("目标列/评分列没有可用于策略回测的有效数值。")
    target = clean[target_col].astype(int)
    scores = clean[score_col].astype(float)
    bad_rate = float((target == 1).mean())
    corr = scores.rank(method="average").corr(target)
    higher_score_riskier = bool(corr is not None and pd.notna(corr) and corr > 0)
    quantile = 0.80 if higher_score_riskier else 0.20
    cutoff = float(scores.quantile(quantile))
    cutoff_literal = _number_literal(cutoff)
    if higher_score_riskier:
        condition = f"{score_col} >= {cutoff_literal}"
        direction = "higher_score_riskier"
        notes = [f"评分越高坏样本率越高，默认拒绝评分最高约 20%（cutoff={cutoff_literal}）。"]
    else:
        condition = f"{score_col} < {cutoff_literal}"
        direction = "lower_score_riskier"
        notes = [f"评分越低坏样本率越高，默认拒绝评分最低约 20%（cutoff={cutoff_literal}）。"]
    return {
        "condition": condition,
        "cutoff": cutoff,
        "direction": direction,
        "bad_rate": bad_rate,
        "notes": notes,
    }


def _number_literal(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.6g}"


def _dataset_name(dataset) -> str:
    source = getattr(dataset, "source_path", None)
    return Path(source).name if source else str(getattr(dataset, "id", ""))


__all__ = [
    "RuleStrategyProposal",
    "StrategyProposal",
    "StrategySetupError",
    "build_rule_strategy_proposal",
    "build_strategy_proposal",
    "is_rule_strategy_goal",
]
