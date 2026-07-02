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


__all__ = ["StrategyProposal", "StrategySetupError", "build_strategy_proposal"]
