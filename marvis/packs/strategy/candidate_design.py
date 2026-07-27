"""Deterministic candidate design for non-approval strategy types.

The caller supplies only a bounded search space and business assumptions.  This
module owns every selected action, metric and generated Strategy DSL rule.  It is
deliberately pure: dataset ownership, source-hash freshness and persistence remain
tool-layer responsibilities.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from marvis.feature.binning import assign_bins, degraded_bin_diagnostic, equal_frequency_edges
from marvis.packs.strategy.dsl import parse_strategy_spec, strategy_spec_hash
from marvis.packs.strategy.economics import limit_metrics, pricing_metrics
from marvis.packs.strategy.errors import StrategyError


CANDIDATE_DESIGN_SCHEMA_VERSION = "strategy.candidate_design.v1"
CANDIDATE_DESIGN_EVIDENCE_VERSION = "strategy.candidate_design_evidence.v1"
CANDIDATE_POLICY_VERSION = "strategy.candidate_policy.v1"

_DEFAULT_BAND_COUNT = 5
_MIN_BIN_SHARE = 0.05
_MAX_BAND_COUNT = 20
_MAX_GRID_SIZE = 50
_SUPPORTED_TYPES = frozenset({"limit", "pricing", "segmentation"})
_METHOD_BY_TYPE = {
    "limit": "score_band_limit",
    "pricing": "score_band_pricing",
    "segmentation": "single_variable_segmentation",
}
_MISSING_POLICY_BY_TYPE = {
    "limit": "zero_limit",
    "pricing": "highest_risk_rate",
    "segmentation": "separate_segment",
}
_ECONOMIC_NAMES = {
    "limit": ("pd", "lgd", "utilization"),
    "pricing": (
        "ead",
        "pd",
        "lgd",
        "funding_rate",
        "term_months",
        "operating_cost_per_loan",
    ),
}
_RATIO_ECONOMICS = frozenset({"pd", "lgd", "utilization", "funding_rate"})
_FORBIDDEN_RESULT_FIELDS = frozenset(
    {
        "action",
        "actions",
        "default_action",
        "metrics",
        "recommendation",
        "recommended",
        "recommended_value",
        "rules",
        "selected_action",
        "strategy_spec",
    }
)


class CandidateDesignError(StrategyError):
    """Fail-closed candidate contract error with machine-readable clarification data."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "candidate_design_invalid",
        fields: Iterable[str] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.fields = tuple(dict.fromkeys(str(field) for field in fields))


@dataclass(frozen=True)
class CandidateDesignResult(Mapping[str, Any]):
    """Immutable, JSON-ready result returned by the deterministic design kernel."""

    strategy_spec: Mapping[str, Any]
    strategy_effect_hash: str
    design_evidence: Mapping[str, Any]
    economics_inputs: Mapping[str, Any] | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy_spec", _deep_freeze(self.strategy_spec))
        object.__setattr__(self, "design_evidence", _deep_freeze(self.design_evidence))
        if self.economics_inputs is not None:
            object.__setattr__(
                self,
                "economics_inputs",
                _deep_freeze(self.economics_inputs),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_spec": _deep_thaw(self.strategy_spec),
            "strategy_effect_hash": self.strategy_effect_hash,
            "design_evidence": _deep_thaw(self.design_evidence),
            "economics_inputs": (
                None
                if self.economics_inputs is None
                else _deep_thaw(self.economics_inputs)
            ),
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return 4


def normalize_candidate_design(
    strategy_type: str,
    candidate_design: object,
    *,
    allowed_columns: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Validate and canonicalize LLM-extracted candidate search parameters."""

    _require_supported_type(strategy_type)
    if not isinstance(candidate_design, Mapping):
        raise CandidateDesignError("candidate_design 必须是一个对象。")
    if any(not isinstance(key, str) for key in candidate_design):
        raise CandidateDesignError("candidate_design 的字段名必须是文本。")

    present_forbidden = sorted(set(candidate_design) & _FORBIDDEN_RESULT_FIELDS)
    if present_forbidden:
        raise CandidateDesignError(
            "candidate_design 只能描述候选输入，LLM 不得提交推荐值、指标、规则或动作："
            + "、".join(present_forbidden)
            + "。",
            code="candidate_result_field_forbidden",
            fields=present_forbidden,
        )

    type_fields = {
        "limit": {
            "schema_version",
            "method",
            "score_col",
            "n_bands",
            "limit_grid",
            "max_expected_loss_per_account",
            "missing_policy",
        },
        "pricing": {
            "schema_version",
            "method",
            "score_col",
            "n_bands",
            "rate_grid",
            "min_roa",
            "missing_policy",
        },
        "segmentation": {
            "schema_version",
            "method",
            "feature_col",
            "n_bands",
            "missing_policy",
        },
    }[strategy_type]
    unexpected = sorted(set(candidate_design) - type_fields)
    if unexpected:
        raise CandidateDesignError(
            f"{strategy_type} candidate_design 包含不支持的字段："
            + "、".join(unexpected)
            + "。",
            fields=unexpected,
        )

    schema_version = candidate_design.get(
        "schema_version", CANDIDATE_DESIGN_SCHEMA_VERSION
    )
    if schema_version != CANDIDATE_DESIGN_SCHEMA_VERSION:
        raise CandidateDesignError(
            "candidate_design schema_version 必须是 "
            f"{CANDIDATE_DESIGN_SCHEMA_VERSION}。"
        )
    method = candidate_design.get("method")
    expected_method = _METHOD_BY_TYPE[strategy_type]
    if method != expected_method:
        raise CandidateDesignError(
            f"{strategy_type} candidate_design.method 必须是 {expected_method}。",
            code="candidate_method_mismatch",
            fields=("method",),
        )

    column_field = "feature_col" if strategy_type == "segmentation" else "score_col"
    column = _required_text(
        candidate_design.get(column_field),
        name=f"candidate_design.{column_field}",
        code="candidate_design_incomplete",
    )
    whitelist = _column_whitelist(allowed_columns)
    if whitelist is not None and column not in whitelist:
        raise CandidateDesignError(
            f"candidate_design.{column_field} 使用了数据集中不存在的列「{column}」。",
            code="candidate_column_not_found",
            fields=(column_field,),
        )

    n_bands = candidate_design.get("n_bands", _DEFAULT_BAND_COUNT)
    if (
        isinstance(n_bands, bool)
        or not isinstance(n_bands, int)
        or not 2 <= n_bands <= _MAX_BAND_COUNT
    ):
        raise CandidateDesignError(
            f"candidate_design.n_bands 必须是 2 到 {_MAX_BAND_COUNT} 的整数。",
            fields=("n_bands",),
        )
    missing_policy = candidate_design.get(
        "missing_policy", _MISSING_POLICY_BY_TYPE[strategy_type]
    )
    expected_missing_policy = _MISSING_POLICY_BY_TYPE[strategy_type]
    if missing_policy != expected_missing_policy:
        raise CandidateDesignError(
            f"{strategy_type} candidate_design.missing_policy 必须是 "
            f"{expected_missing_policy}；缺失动作由版本化平台 policy 控制。",
            code="candidate_missing_policy_mismatch",
            fields=("missing_policy",),
        )

    normalized: dict[str, Any] = {
        "schema_version": CANDIDATE_DESIGN_SCHEMA_VERSION,
        "method": expected_method,
        column_field: column,
        "n_bands": n_bands,
        "missing_policy": expected_missing_policy,
    }
    if strategy_type == "limit":
        if "limit_grid" not in candidate_design:
            raise CandidateDesignError(
                "额度候选设计缺少 limit_grid。",
                code="candidate_design_incomplete",
                fields=("limit_grid",),
            )
        normalized["limit_grid"] = _numeric_grid(
            candidate_design["limit_grid"],
            name="candidate_design.limit_grid",
            minimum=0.0,
            exclusive_minimum=True,
        )
        if "max_expected_loss_per_account" not in candidate_design:
            raise CandidateDesignError(
                "额度候选设计缺少单户预期损失预算 max_expected_loss_per_account。",
                code="candidate_design_incomplete",
                fields=("max_expected_loss_per_account",),
            )
        normalized["max_expected_loss_per_account"] = _finite_number(
            candidate_design["max_expected_loss_per_account"],
            name="candidate_design.max_expected_loss_per_account",
            minimum=0.0,
        )
    elif strategy_type == "pricing":
        if "rate_grid" not in candidate_design:
            raise CandidateDesignError(
                "定价候选设计缺少 rate_grid。",
                code="candidate_design_incomplete",
                fields=("rate_grid",),
            )
        normalized["rate_grid"] = _numeric_grid(
            candidate_design["rate_grid"],
            name="candidate_design.rate_grid",
            minimum=0.0,
            maximum=1.0,
        )
        normalized["min_roa"] = _finite_number(
            candidate_design.get("min_roa", 0.0),
            name="candidate_design.min_roa",
            minimum=0.0,
            maximum=1.0,
        )
    return normalized


def normalize_candidate_economics_inputs(
    strategy_type: str,
    economics_inputs: object,
    *,
    allowed_columns: Iterable[str] | None = None,
) -> dict[str, Any] | None:
    """Normalize the economic bundle shared by candidate design and backtest."""

    _require_supported_type(strategy_type)
    if strategy_type == "segmentation":
        if economics_inputs not in (None, {}):
            raise CandidateDesignError(
                "分群候选设计不接受 economics_inputs。",
                code="candidate_economics_not_allowed",
                fields=("economics_inputs",),
            )
        return None
    if not isinstance(economics_inputs, Mapping):
        names = _ECONOMIC_NAMES[strategy_type]
        raise CandidateDesignError(
            f"{strategy_type} 候选设计缺少完整 economics_inputs。",
            code="candidate_economics_incomplete",
            fields=tuple(f"{name}_col/{name}_value" for name in names),
        )
    if any(not isinstance(key, str) for key in economics_inputs):
        raise CandidateDesignError("economics_inputs 的字段名必须是文本。")

    names = _ECONOMIC_NAMES[strategy_type]
    allowed = {key for name in names for key in (f"{name}_col", f"{name}_value")}
    unexpected = sorted(set(economics_inputs) - allowed)
    if unexpected:
        raise CandidateDesignError(
            f"{strategy_type} economics_inputs 包含不支持的字段："
            + "、".join(unexpected)
            + "。",
            fields=unexpected,
        )

    whitelist = _column_whitelist(allowed_columns)
    normalized: dict[str, Any] = {}
    missing: list[str] = []
    for name in names:
        column_key = f"{name}_col"
        value_key = f"{name}_value"
        has_column = column_key in economics_inputs
        has_value = value_key in economics_inputs
        if has_column == has_value:
            if has_column:
                raise CandidateDesignError(
                    f"经济参数 {name} 必须在 {column_key} 和 {value_key} 中二选一。",
                    code="candidate_economics_ambiguous",
                    fields=(column_key, value_key),
                )
            missing.append(f"{column_key}/{value_key}")
            continue
        if strategy_type == "pricing" and name in {"ead", "pd"} and not has_column:
            raise CandidateDesignError(
                "定价候选必须使用数据中的真实 EAD/PD 列，不能用固定值替代。",
                code="candidate_requires_observed_economics",
                fields=("ead_col", "pd_col"),
            )
        if has_column:
            column = _required_text(
                economics_inputs[column_key],
                name=f"economics_inputs.{column_key}",
            )
            if whitelist is not None and column not in whitelist:
                raise CandidateDesignError(
                    f"economics_inputs.{column_key} 使用了数据集中不存在的列「{column}」。",
                    code="candidate_column_not_found",
                    fields=(column_key,),
                )
            normalized[column_key] = column
            continue
        normalized[value_key] = _economic_value(name, economics_inputs[value_key])

    if missing:
        raise CandidateDesignError(
            f"{strategy_type} 候选设计经济口径不完整，缺少：" + "、".join(missing) + "。",
            code="candidate_economics_incomplete",
            fields=missing,
        )
    return normalized


def design_strategy_candidate(
    df: pd.DataFrame,
    *,
    strategy_type: str,
    target_col: str,
    candidate_design: object,
    economics_inputs: object = None,
    dataset_id: str | None = None,
    source_dataset_content_hash: str | None = None,
    candidate_policy_version: str = CANDIDATE_POLICY_VERSION,
) -> CandidateDesignResult:
    """Generate a canonical Strategy DSL candidate and deterministic evidence."""

    _require_supported_type(strategy_type)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise CandidateDesignError("候选设计需要非空 pandas DataFrame。")
    if candidate_policy_version != CANDIDATE_POLICY_VERSION:
        raise CandidateDesignError(
            f"candidate_policy_version 必须是 {CANDIDATE_POLICY_VERSION}。",
            code="candidate_policy_version_mismatch",
        )
    target_name = _required_text(target_col, name="target_col")
    if target_name not in df.columns:
        raise CandidateDesignError(
            f"target_col 使用了数据集中不存在的列「{target_name}」。",
            code="candidate_column_not_found",
            fields=("target_col",),
        )
    normalized_design = normalize_candidate_design(
        strategy_type,
        candidate_design,
        allowed_columns=df.columns,
    )
    normalized_economics = normalize_candidate_economics_inputs(
        strategy_type,
        economics_inputs,
        allowed_columns=df.columns,
    )
    normalized_dataset_id = (
        None if dataset_id is None else _required_text(dataset_id, name="dataset_id")
    )
    if source_dataset_content_hash is not None and not re.fullmatch(
        r"[0-9a-f]{64}", source_dataset_content_hash
    ):
        raise CandidateDesignError(
            "source_dataset_content_hash 必须是小写 SHA256。",
            code="candidate_source_hash_invalid",
        )

    target = _target_series(df[target_name])
    design_col = (
        normalized_design["feature_col"]
        if strategy_type == "segmentation"
        else normalized_design["score_col"]
    )
    values = _numeric_design_series(df[design_col], name=design_col)
    finite_values = values.dropna().to_numpy(dtype=float)
    if finite_values.size == 0:
        raise CandidateDesignError(f"候选设计列「{design_col}」没有可分箱的有限值。")
    edges = equal_frequency_edges(
        finite_values,
        normalized_design["n_bands"],
        min_bin_pct=_MIN_BIN_SHARE,
    )
    assigned = pd.Series(
        assign_bins(values.to_numpy(dtype=float), edges),
        index=df.index,
        dtype=int,
    )
    bands = _base_bands(df, target=target, assigned=assigned, edges=edges)
    design_hash = _design_input_hash(
        strategy_type=strategy_type,
        target_col=target_name,
        candidate_design=normalized_design,
        economics_inputs=normalized_economics,
        candidate_policy_version=candidate_policy_version,
        source_dataset_content_hash=source_dataset_content_hash,
    )

    if strategy_type == "limit":
        assert normalized_economics is not None
        selected_actions, bands = _design_limit_actions(
            df,
            target=target,
            bands=bands,
            assigned=assigned,
            design=normalized_design,
            economics_inputs=normalized_economics,
        )
        default_action = {"type": "limit", "value": 0.0}
        default_rationale = "缺失候选设计列时按版本化平台 policy 赋零额度。"
    elif strategy_type == "pricing":
        assert normalized_economics is not None
        selected_actions, bands, missing_rate = _design_pricing_actions(
            df,
            target=target,
            bands=bands,
            assigned=assigned,
            design=normalized_design,
            economics_inputs=normalized_economics,
        )
        default_action = {"type": "pricing", "value": missing_rate}
        default_rationale = "缺失候选设计列时按最高风险分箱的平台推荐利率处理。"
    else:
        selected_actions, bands = _design_segmentation_actions(bands)
        default_action = {"type": "segment", "value": "UNASSIGNED"}
        default_rationale = "缺失单变量值进入独立 UNASSIGNED 分群。"

    rules = [
        {
            "rule_id": f"{_rule_prefix(strategy_type)}-missing",
            "priority": 0,
            "condition": {"op": "is_null", "field": design_col},
            "action": {
                **default_action,
                "reason_code": "CANDIDATE_MISSING_POLICY",
            },
        }
    ]
    for index, (_band, action) in enumerate(
        zip(bands, selected_actions, strict=True)
    ):
        rules.append(
            {
                "rule_id": f"{_rule_prefix(strategy_type)}-band-{index + 1:02d}",
                "priority": (index + 1) * 10,
                "condition": _band_condition(
                    field=design_col,
                    edges=edges,
                    index=index,
                ),
                "action": {
                    **action,
                    "reason_code": f"CANDIDATE_BAND_{index + 1:02d}",
                },
            }
        )

    lineage: dict[str, Any] = {
        "source": "deterministic_candidate_design",
        "candidate_policy_version": candidate_policy_version,
        "candidate_design_input_hash": design_hash,
        "method": normalized_design["method"],
    }
    if normalized_dataset_id is not None:
        lineage["dataset_id"] = normalized_dataset_id
    if source_dataset_content_hash is not None:
        lineage["source_dataset_content_hash"] = source_dataset_content_hash
    parsed = parse_strategy_spec(
        {
            "strategy_type": strategy_type,
            "default_action": default_action,
            "rules": rules,
            "metadata": {
                "score_col": design_col,
                "description": "Deterministic non-approval strategy candidate",
                "lineage": lineage,
            },
        }
    )
    canonical_spec = parsed.to_dict()

    red_flags: list[dict[str, Any]] = []
    diagnostic = degraded_bin_diagnostic(
        edges,
        normalized_design["n_bands"],
        feature=design_col,
    )
    if diagnostic is not None:
        red_flags.append({"code": "degraded_binning", "level": "amber", **diagnostic})
    if strategy_type == "limit":
        objective = "maximize_limit_under_expected_loss_budget"
        assumptions = [
            "每个分箱选择满足单户预期损失预算的最高候选额度。",
            "额度设计显式使用 utilization 计算 EAD；当前不是额度与定价联合利润优化。",
        ]
    elif strategy_type == "pricing":
        objective = "maximize_static_expected_profit"
        assumptions = [
            "按真实逐行 EAD/PD 与给定经济口径计算每个候选利率的静态预期利润。",
            "当前未建模利率对申请、支用、提前还款或违约行为的价格弹性。",
        ]
        red_flags.append(
            {
                "code": "price_elasticity_not_modeled",
                "level": "amber",
                "message": (
                    "候选利率只在已批准 rate_grid 内最大化静态预期利润；"
                    "未建模价格弹性，不能解释为真实转化率或经营收益最优。"
                ),
            }
        )
    else:
        objective = "rank_equal_frequency_bands_by_observed_bad_rate"
        assumptions = [
            "使用固定等频分箱，并按当前样本的观测坏率稳定排序为 R1..Rn。",
            "分群标签是开发样本风险层级，不代表独立验证或已采纳结论。",
        ]
    evidence = {
        "schema_version": CANDIDATE_DESIGN_EVIDENCE_VERSION,
        "strategy_type": strategy_type,
        "method": normalized_design["method"],
        "objective": objective,
        "assumptions": assumptions,
        "candidate_policy_version": candidate_policy_version,
        "candidate_design_input_hash": design_hash,
        "dataset_id": normalized_dataset_id,
        "source_dataset_content_hash": source_dataset_content_hash,
        "design_column": design_col,
        "target_col": target_name,
        "requested_band_count": normalized_design["n_bands"],
        "actual_band_count": len(bands),
        "missing_count": int(values.isna().sum()),
        "missing_policy": normalized_design["missing_policy"],
        "default_action_rationale": default_rationale,
        "bands": bands,
        "red_flags": red_flags,
    }
    return CandidateDesignResult(
        strategy_spec=canonical_spec,
        strategy_effect_hash=strategy_spec_hash(parsed),
        design_evidence=evidence,
        economics_inputs=normalized_economics,
    )


def _design_limit_actions(
    df: pd.DataFrame,
    *,
    target: pd.Series,
    bands: list[dict[str, Any]],
    assigned: pd.Series,
    design: Mapping[str, Any],
    economics_inputs: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    resolved = _resolve_economics(df, economics_inputs, names=_ECONOMIC_NAMES["limit"])
    budget = float(design["max_expected_loss_per_account"])
    selected_actions: list[dict[str, Any]] = []
    for index, band in enumerate(bands):
        mask = assigned.eq(index)
        candidates: list[dict[str, Any]] = []
        for limit in design["limit_grid"]:
            metrics = limit_metrics(
                pd.Series(float(limit), index=df.index[mask], dtype=float),
                target.loc[mask],
                pd=_slice_numeric_input(resolved["pd"], mask),
                lgd=_slice_numeric_input(resolved["lgd"], mask),
                utilization=_slice_numeric_input(resolved["utilization"], mask),
            )
            economics = metrics["economics"]
            assert economics is not None
            mean_loss = float(economics["expected_loss"] / band["count"])
            candidates.append(
                {
                    "limit": float(limit),
                    "expected_ead": float(economics["expected_ead"]),
                    "expected_loss": float(economics["expected_loss"]),
                    "mean_expected_loss_per_account": mean_loss,
                    "feasible": bool(mean_loss <= budget + 1e-12),
                }
            )
        feasible = [candidate for candidate in candidates if candidate["feasible"]]
        if not feasible:
            raise CandidateDesignError(
                f"额度分箱 {band['band_id']} 在单户预期损失预算 {budget:g} 下无可行额度。",
                code="candidate_band_infeasible",
                fields=("limit_grid", "max_expected_loss_per_account"),
            )
        selected = max(
            feasible,
            key=lambda item: (item["limit"], -item["mean_expected_loss_per_account"]),
        )
        band["candidate_scores"] = candidates
        band["selected_action"] = {"type": "limit", "value": selected["limit"]}
        band["risk_estimate"] = _mean_numeric(_slice_numeric_input(resolved["pd"], mask))
        selected_actions.append({"type": "limit", "value": selected["limit"]})
    return selected_actions, bands


def _design_pricing_actions(
    df: pd.DataFrame,
    *,
    target: pd.Series,
    bands: list[dict[str, Any]],
    assigned: pd.Series,
    design: Mapping[str, Any],
    economics_inputs: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    resolved = _resolve_economics(df, economics_inputs, names=_ECONOMIC_NAMES["pricing"])
    min_roa = float(design["min_roa"])
    selected_actions: list[dict[str, Any]] = []
    for index, band in enumerate(bands):
        mask = assigned.eq(index)
        candidates: list[dict[str, Any]] = []
        for rate in design["rate_grid"]:
            metrics = pricing_metrics(
                pd.Series(float(rate), index=df.index[mask], dtype=float),
                target.loc[mask],
                ead=_slice_numeric_input(resolved["ead"], mask),
                pd=_slice_numeric_input(resolved["pd"], mask),
                lgd=_slice_numeric_input(resolved["lgd"], mask),
                funding_rate=_slice_numeric_input(resolved["funding_rate"], mask),
                term_months=_slice_numeric_input(resolved["term_months"], mask),
                operating_cost_per_loan=_slice_numeric_input(
                    resolved["operating_cost_per_loan"], mask
                ),
            )
            economics = metrics["economics"]
            assert economics is not None
            roa = economics["roa"]
            profit = float(economics["profit"])
            candidates.append(
                {
                    "rate": float(rate),
                    "total_ead": float(economics["total_ead"]),
                    "expected_loss": float(economics["expected_loss"]),
                    "expected_profit": profit,
                    "roa": None if roa is None else float(roa),
                    "feasible": bool(
                        profit >= -1e-12 and roa is not None and float(roa) >= min_roa - 1e-12
                    ),
                }
            )
        feasible = [candidate for candidate in candidates if candidate["feasible"]]
        if not feasible:
            raise CandidateDesignError(
                f"定价分箱 {band['band_id']} 在最小 ROA {min_roa:.2%} 下无可行利率。",
                code="candidate_band_infeasible",
                fields=("rate_grid", "min_roa", "economics_inputs"),
            )
        selected = max(
            feasible,
            key=lambda item: (item["expected_profit"], -item["rate"]),
        )
        band["candidate_scores"] = candidates
        band["selected_action"] = {"type": "pricing", "value": selected["rate"]}
        band["risk_estimate"] = _mean_numeric(_slice_numeric_input(resolved["pd"], mask))
        selected_actions.append({"type": "pricing", "value": selected["rate"]})
    highest_risk_index = max(
        range(len(bands)),
        key=lambda index: (bands[index]["risk_estimate"], index),
    )
    return (
        selected_actions,
        bands,
        float(selected_actions[highest_risk_index]["value"]),
    )


def _design_segmentation_actions(
    bands: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unlabeled = [band["band_id"] for band in bands if band["bad_rate"] is None]
    if unlabeled:
        raise CandidateDesignError(
            "分群候选每个有效分箱都必须有标签；缺少标签的分箱：" + "、".join(unlabeled) + "。",
            code="candidate_band_unlabeled",
            fields=("target_col",),
        )
    risk_order = sorted(
        range(len(bands)),
        key=lambda index: (float(bands[index]["bad_rate"]), index),
    )
    labels = {band_index: f"R{rank + 1}" for rank, band_index in enumerate(risk_order)}
    actions: list[dict[str, Any]] = []
    for index, band in enumerate(bands):
        action = {"type": "segment", "value": labels[index]}
        band["candidate_scores"] = []
        band["selected_action"] = action
        band["risk_estimate"] = band["bad_rate"]
        actions.append(action)
    return actions, bands


def _base_bands(
    df: pd.DataFrame,
    *,
    target: pd.Series,
    assigned: pd.Series,
    edges: np.ndarray,
) -> list[dict[str, Any]]:
    total = int(len(df))
    bands: list[dict[str, Any]] = []
    for index in range(len(edges) - 1):
        mask = assigned.eq(index)
        count = int(mask.sum())
        if count == 0:
            raise CandidateDesignError(
                f"确定性分箱 B{index + 1:02d} 为空，不能生成无证据动作。",
                code="candidate_empty_band",
            )
        labels = target.loc[mask]
        labeled_count = int(labels.notna().sum())
        bad_count = int(labels.eq(1.0).sum())
        bands.append(
            {
                "band_id": f"B{index + 1:02d}",
                "lower": _finite_bound(edges[index]),
                "upper": _finite_bound(edges[index + 1]),
                "include_lower": True,
                "include_upper": index == len(edges) - 2,
                "count": count,
                "population_share": float(count / total),
                "labeled_count": labeled_count,
                "bad_count": bad_count,
                "bad_rate": (
                    None if labeled_count == 0 else float(bad_count / labeled_count)
                ),
            }
        )
    return bands


def _band_condition(*, field: str, edges: np.ndarray, index: int) -> dict[str, Any]:
    if len(edges) == 2:
        return {"op": "is_not_null", "field": field}
    if index == 0:
        return {
            "op": "compare",
            "field": field,
            "operator": "<",
            "value": float(edges[1]),
            "missing": "no_match",
        }
    if index == len(edges) - 2:
        return {
            "op": "compare",
            "field": field,
            "operator": ">=",
            "value": float(edges[index]),
            "missing": "no_match",
        }
    return {
        "op": "between",
        "field": field,
        "lower": float(edges[index]),
        "upper": float(edges[index + 1]),
        "include_lower": True,
        "include_upper": False,
        "missing": "no_match",
    }


def _resolve_economics(
    df: pd.DataFrame,
    economics_inputs: Mapping[str, Any],
    *,
    names: Sequence[str],
) -> dict[str, pd.Series | float]:
    resolved: dict[str, pd.Series | float] = {}
    for name in names:
        column_key = f"{name}_col"
        if column_key in economics_inputs:
            resolved[name] = df[economics_inputs[column_key]]
        else:
            resolved[name] = float(economics_inputs[f"{name}_value"])
    return resolved


def _slice_numeric_input(
    value: pd.Series | float,
    mask: pd.Series,
) -> pd.Series | float:
    return value.loc[mask] if isinstance(value, pd.Series) else value


def _mean_numeric(value: pd.Series | float) -> float:
    if isinstance(value, pd.Series):
        numeric = pd.to_numeric(value, errors="raise").astype(float)
        return float(numeric.mean())
    return float(value)


def _target_series(value: pd.Series) -> pd.Series:
    try:
        numeric = pd.to_numeric(value, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise CandidateDesignError("target_col 必须只包含 0、1 或缺失值。") from exc
    finite = numeric.dropna()
    if not finite.map(math.isfinite).all() or not finite.isin([0.0, 1.0]).all():
        raise CandidateDesignError("target_col 必须只包含 0、1 或缺失值。")
    return numeric


def _numeric_design_series(value: pd.Series, *, name: str) -> pd.Series:
    try:
        numeric = pd.to_numeric(value, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise CandidateDesignError(f"候选设计列「{name}」必须是数值列。") from exc
    finite = numeric.dropna()
    if not finite.map(math.isfinite).all():
        raise CandidateDesignError(f"候选设计列「{name}」只能包含有限数值或缺失值。")
    return numeric


def _economic_value(name: str, value: object) -> float:
    minimum = 0.0
    maximum = 1.0 if name in _RATIO_ECONOMICS else None
    number = _finite_number(
        value,
        name=f"economics_inputs.{name}_value",
        minimum=minimum,
        maximum=maximum,
    )
    if name == "term_months" and number <= 0:
        raise CandidateDesignError("economics_inputs.term_months_value 必须大于 0。")
    return number


def _numeric_grid(
    value: object,
    *,
    name: str,
    minimum: float,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str | bytes | bytearray)
        or not 1 <= len(value) <= _MAX_GRID_SIZE
    ):
        raise CandidateDesignError(f"{name} 必须包含 1 到 {_MAX_GRID_SIZE} 个候选数值。")
    numbers = [
        _finite_number(
            item,
            name=name,
            minimum=minimum,
            maximum=maximum,
            exclusive_minimum=exclusive_minimum,
        )
        for item in value
    ]
    if len(set(numbers)) != len(numbers):
        raise CandidateDesignError(f"{name} 不能包含重复值。")
    return sorted(numbers)


def _finite_number(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CandidateDesignError(f"{name} 必须是有限数字。")
    number = float(value)
    if not math.isfinite(number):
        raise CandidateDesignError(f"{name} 必须是有限数字。")
    if minimum is not None and (
        number < minimum or (exclusive_minimum and number == minimum)
    ):
        relation = "大于" if exclusive_minimum else "大于等于"
        raise CandidateDesignError(f"{name} 必须{relation} {minimum:g}。")
    if maximum is not None and number > maximum:
        raise CandidateDesignError(f"{name} 必须小于等于 {maximum:g}。")
    return number


def _required_text(
    value: object,
    *,
    name: str,
    code: str = "candidate_design_invalid",
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CandidateDesignError(
            f"{name} 必须是非空文本。",
            code=code,
            fields=(name.rsplit(".", 1)[-1],),
        )
    return value.strip()


def _column_whitelist(values: Iterable[str] | None) -> frozenset[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        return frozenset({values})
    try:
        return frozenset(value for value in values if isinstance(value, str))
    except TypeError:
        return frozenset()


def _require_supported_type(strategy_type: object) -> None:
    if strategy_type == "collection":
        raise CandidateDesignError(
            "催收动作策略尚无已评审的动作、成本与回收口径，不能映射为分群或拒绝策略。",
            code="collection_strategy_unsupported",
            fields=("strategy_type",),
        )
    if not isinstance(strategy_type, str) or strategy_type not in _SUPPORTED_TYPES:
        raise CandidateDesignError(
            "确定性候选设计只支持 limit、pricing、segmentation。",
            code="candidate_strategy_type_unsupported",
            fields=("strategy_type",),
        )


def _design_input_hash(
    *,
    strategy_type: str,
    target_col: str,
    candidate_design: Mapping[str, Any],
    economics_inputs: Mapping[str, Any] | None,
    candidate_policy_version: str,
    source_dataset_content_hash: str | None,
) -> str:
    payload = {
        "strategy_type": strategy_type,
        "target_col": target_col,
        "candidate_design": dict(candidate_design),
        "economics_inputs": None if economics_inputs is None else dict(economics_inputs),
        "candidate_policy_version": candidate_policy_version,
        "source_dataset_content_hash": source_dataset_content_hash,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _finite_bound(value: float) -> float | None:
    return None if not math.isfinite(float(value)) else float(value)


def _rule_prefix(strategy_type: str) -> str:
    return "segment" if strategy_type == "segmentation" else strategy_type


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


__all__ = [
    "CANDIDATE_DESIGN_EVIDENCE_VERSION",
    "CANDIDATE_DESIGN_SCHEMA_VERSION",
    "CANDIDATE_POLICY_VERSION",
    "CandidateDesignError",
    "CandidateDesignResult",
    "design_strategy_candidate",
    "normalize_candidate_design",
    "normalize_candidate_economics_inputs",
]
