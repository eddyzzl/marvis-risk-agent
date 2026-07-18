"""Pure deterministic calculations for the risk-analysis deliverable.

The pack tool owns dataset access and artifact persistence.  This module accepts an
already-loaded frame plus an explicit canonical-to-source column map, validates every
business input, and returns a JSON-safe calculation payload.  The Excel renderer only
formats this payload; it never recomputes a financial metric.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import pandas as pd


ANALYSIS_KINDS = frozenset({"vtg_terminal", "profitability"})
WEIGHT_SUM_TOLERANCE = 1e-4
PROFIT_RATE_RECONCILIATION_TOLERANCE = 1e-4
PRODUCT_SCOPE_LIMIT = 8
PRODUCT_SCOPE_ITEM_MAX_CHARS = 80
AS_OF_PERIOD_MAX_CHARS = 40
SCENARIO_MAX_CHARS = 40

_VTG_REQUIRED = (
    "product",
    "cohort",
    "as_of_date",
    "amount_unit",
    "disbursement_amount",
    "mob14_bad_rate",
)
_VTG_CURVE_REQUIRED = ("mob", "mob_days", "day_count_basis")
_VTG_CURVE_BALANCE_FIELDS = ("mob_balance_rate", "mob_balance_amount")
_VTG_CURVE_DAY_TOLERANCE = 1e-6
_VTG_BALANCE_RECONCILIATION_REL_TOLERANCE = 1e-4
_VTG_BALANCE_RECONCILIATION_ABS_TOLERANCE = 0.01
_VTG_RECOVERY_RECONCILIATION_TOLERANCE = 1e-4
_VTG_ANNUALIZED_RECONCILIATION_TOLERANCE = 1e-4
_VTG_OPTIONAL_RATES = (
    "terminal_bad_rate",
    "long_term_recovery_rate",
    "auxiliary_terminal_bad_rate",
    "previous_mob14_bad_rate",
    "previous_terminal_bad_rate",
    "previous_annualized_bad_rate",
)
_VTG_OPTIONAL_NONNEGATIVE = (
    "avg_daily_balance",
    "previous_disbursement_amount",
    "previous_avg_daily_balance",
)
_VTG_OPTIONAL_GROUP_TEXT = (
    "scenario",
    "channel",
    "selection_rule",
)

_PROFIT_REQUIRED = (
    "product",
    "as_of_period",
    "asset_class",
    "weight",
    "weight_basis",
    "customer_rate",
)
_PROFIT_COST_FIELDS = (
    "interest_loss_rate",
    "revenue_share_rate",
    "risk_cost_rate",
    "acquisition_cost_rate",
    "data_cost_rate",
    "payment_cost_rate",
    "collection_cost_rate",
    "funding_cost_rate",
    "other_cost_rate",
    "tax_rate",
)
_PROFIT_ALWAYS_EXPLICIT_COST_FIELDS = (
    "acquisition_cost_rate",
    "payment_cost_rate",
    "collection_cost_rate",
    "funding_cost_rate",
    "other_cost_rate",
)
_PROFIT_REQUIRED = (*_PROFIT_REQUIRED, *_PROFIT_ALWAYS_EXPLICIT_COST_FIELDS)
_PROFIT_DERIVABLE_COST_FIELDS = (
    "risk_cost_rate",
    "interest_loss_rate",
    "revenue_share_rate",
    "data_cost_rate",
    "tax_rate",
)
_PROFIT_DRIVER_FIELDS = (
    "amount_unit",
    "terminal_vintage_rate",
    "risk_turnover",
    "loss_timing_factor",
    "profit_share_ratio",
    "per_application_cost",
    "credit_approval_rate",
    "draw_initiation_rate",
    "draw_approval_rate",
    "average_ticket",
    "data_annualization_factor",
    "tax_method",
    "tax_inclusive_divisor",
    "tax_combined_rate",
)

_COST_LABELS = {
    "interest_loss_rate": "息费损失",
    "revenue_share_rate": "分润成本",
    "risk_cost_rate": "风险成本",
    "acquisition_cost_rate": "获客成本",
    "data_cost_rate": "数据成本",
    "payment_cost_rate": "支付成本",
    "collection_cost_rate": "催收成本",
    "funding_cost_rate": "资金成本",
    "other_cost_rate": "其他成本",
    "tax_rate": "税费",
}


class RiskAnalysisError(ValueError):
    """Typed, user-readable invalid-input error for the pack boundary."""

    def __init__(
        self,
        message: str,
        *,
        analysis_kind: str | None = None,
        field: str | None = None,
        row_number: int | None = None,
    ) -> None:
        self.analysis_kind = analysis_kind
        self.field = field
        self.row_number = row_number
        super().__init__(message)

    def to_detail(self) -> dict[str, Any]:
        return {
            "kind": "risk_analysis_invalid",
            "analysis_kind": self.analysis_kind,
            "field": self.field,
            "row_number": self.row_number,
            "message": str(self),
        }


@dataclass(frozen=True)
class RiskAnalysisCalculation:
    analysis_kind: str
    column_map: dict[str, str]
    product_scope: list[str]
    as_of_period: str
    headline_metrics: dict[str, Any]
    key_points: list[str]
    red_flags: list[str]
    assumptions: list[str]
    source_row_count: int
    row_count: int
    detail_rows: list[dict[str, Any]]
    summary_rows: list[dict[str, Any]]
    formula_definitions: list[dict[str, str]]
    data_quality: list[dict[str, str]]


def calculate_risk_analysis(
    frame: pd.DataFrame,
    *,
    analysis_kind: str,
    column_map: dict[str, str],
) -> RiskAnalysisCalculation:
    kind = str(analysis_kind or "").strip()
    if kind == "vtg_terminal":
        return calculate_vtg_terminal(frame, column_map=column_map)
    if kind == "profitability":
        return calculate_profitability(frame, column_map=column_map)
    raise RiskAnalysisError(
        f"不支持的风险分析类型: {analysis_kind!r}",
        analysis_kind=kind or None,
    )


def calculate_vtg_terminal(
    frame: pd.DataFrame,
    *,
    column_map: dict[str, str],
) -> RiskAnalysisCalculation:
    kind = "vtg_terminal"
    data = _normalized_frame(frame, analysis_kind=kind)
    source_row_count = len(data)
    mapping = _normalize_column_map(data, column_map, analysis_kind=kind)
    _require_mappings(mapping, _VTG_REQUIRED, analysis_kind=kind)
    curve_derived_count = 0
    zero_disbursement_skipped_count = 0
    working_mapping = mapping
    if "turnover" not in mapping:
        _require_mappings(mapping, _VTG_CURVE_REQUIRED, analysis_kind=kind)
        if not any(field in mapping for field in _VTG_CURVE_BALANCE_FIELDS):
            raise RiskAnalysisError(
                "VTG 测算缺少 turnover 时，column_map 至少需要 "
                "mob_balance_rate 或 mob_balance_amount。",
                analysis_kind=kind,
                field="mob_balance_rate",
            )
        data, working_mapping, zero_disbursement_skipped_count = (
            _collapse_vtg_balance_curves(
                data,
                mapping=mapping,
                analysis_kind=kind,
            )
        )
        curve_derived_count = len(data)
    else:
        keep_positions: list[int] = []
        for position, row in data.iterrows():
            disbursement = _number(
                row,
                mapping,
                "disbursement_amount",
                kind,
                position + 1,
                required=True,
                minimum=0.0,
            )
            if disbursement == 0.0:
                zero_disbursement_skipped_count += 1
            else:
                keep_positions.append(position)
        data = data.loc[keep_positions].reset_index(drop=True)
    if data.empty:
        raise RiskAnalysisError(
            "VTG 材料中没有可测算的正放款金额行；零放款占位行不参与计算。",
            analysis_kind=kind,
            field="disbursement_amount",
        )
    has_explicit_mob_auxiliary_method = (
        "terminal_method" in mapping and "auxiliary_terminal_bad_rate" in mapping
    )
    if (
        "terminal_bad_rate" not in mapping
        and "long_term_recovery_rate" not in mapping
        and not has_explicit_mob_auxiliary_method
    ):
        raise RiskAnalysisError(
            "VTG 终值测算至少需要 terminal_bad_rate、long_term_recovery_rate，"
            "或 terminal_method=min_mob14_auxiliary 与辅助终值。",
            analysis_kind=kind,
            field="terminal_bad_rate",
        )

    detail_rows: list[dict[str, Any]] = []
    substitution_count = 0
    auxiliary_min_count = 0
    derived_balance_count = 0
    terminal_above_mob_count = 0
    annualized_over_one_count = 0
    zero_mob_count = 0
    partial_previous_count = 0
    declared_terminal_methods: list[str] = []
    direct_recovery_reconciled_count = 0
    direct_auxiliary_ignored_count = 0
    explicit_mob_auxiliary_min_count = 0
    derived_previous_turnover_count = 0
    derived_previous_annualized_count = 0

    for position, row in data.iterrows():
        row_number = position + 1
        product = _required_text(row, working_mapping, "product", kind, row_number)
        cohort = _required_text(row, working_mapping, "cohort", kind, row_number)
        as_of_date = _required_text(
            row, working_mapping, "as_of_date", kind, row_number
        )
        amount_unit = _required_text(
            row, working_mapping, "amount_unit", kind, row_number
        )
        scenario = _optional_text(row, working_mapping, "scenario")
        channel = _optional_text(row, working_mapping, "channel")
        selection_rule = _optional_text(row, working_mapping, "selection_rule")
        tenor_months = _number(
            row,
            working_mapping,
            "tenor_months",
            kind,
            row_number,
            minimum=0.0,
            minimum_inclusive=False,
        )
        day_count_basis = _number(
            row,
            working_mapping,
            "day_count_basis",
            kind,
            row_number,
            minimum=0.0,
            minimum_inclusive=False,
        )
        mob_balance_source = _optional_text(row, working_mapping, "mob_balance_source")
        disbursement = _number(
            row,
            working_mapping,
            "disbursement_amount",
            kind,
            row_number,
            required=True,
            minimum=0.0,
            minimum_inclusive=False,
        )
        mob14 = _number(
            row,
            working_mapping,
            "mob14_bad_rate",
            kind,
            row_number,
            required=True,
            rate=True,
        )
        turnover = _number(
            row,
            working_mapping,
            "turnover",
            kind,
            row_number,
            required=True,
            minimum=0.0,
            minimum_inclusive=False,
        )
        direct_terminal = _number(
            row, working_mapping, "terminal_bad_rate", kind, row_number, rate=True
        )
        recovery_input = _number(
            row,
            working_mapping,
            "long_term_recovery_rate",
            kind,
            row_number,
            rate=True,
        )
        auxiliary_terminal = _number(
            row,
            working_mapping,
            "auxiliary_terminal_bad_rate",
            kind,
            row_number,
            rate=True,
        )
        terminal_method = _optional_text(row, working_mapping, "terminal_method")
        if terminal_method is not None:
            declared_terminal_methods.append(terminal_method)
        avg_daily_balance = _number(
            row,
            working_mapping,
            "avg_daily_balance",
            kind,
            row_number,
            minimum=0.0,
        )
        previous_mob14 = _number(
            row,
            working_mapping,
            "previous_mob14_bad_rate",
            kind,
            row_number,
            rate=True,
        )
        previous_terminal = _number(
            row,
            working_mapping,
            "previous_terminal_bad_rate",
            kind,
            row_number,
            rate=True,
        )
        previous_turnover = _number(
            row,
            working_mapping,
            "previous_turnover",
            kind,
            row_number,
            minimum=0.0,
            minimum_inclusive=False,
        )
        previous_annualized = _number(
            row,
            working_mapping,
            "previous_annualized_bad_rate",
            kind,
            row_number,
            minimum=0.0,
        )
        previous_disbursement = _number(
            row,
            working_mapping,
            "previous_disbursement_amount",
            kind,
            row_number,
            minimum=0.0,
        )
        previous_avg_balance = _number(
            row,
            working_mapping,
            "previous_avg_daily_balance",
            kind,
            row_number,
            minimum=0.0,
        )
        previous_turnover_source = "direct" if previous_turnover is not None else None
        if previous_disbursement is not None and previous_avg_balance is not None:
            if previous_avg_balance <= 0.0:
                raise RiskAnalysisError(
                    f"第 {row_number} 行使用上期金额/日均余额推导周转次数时，"
                    "previous_avg_daily_balance 必须大于 0。",
                    analysis_kind=kind,
                    field="previous_avg_daily_balance",
                    row_number=row_number,
                )
            if previous_disbursement <= 0.0:
                raise RiskAnalysisError(
                    f"第 {row_number} 行使用上期金额/日均余额推导周转次数时，"
                    "previous_disbursement_amount 必须大于 0。",
                    analysis_kind=kind,
                    field="previous_disbursement_amount",
                    row_number=row_number,
                )
            if previous_turnover is None:
                previous_turnover = previous_disbursement / previous_avg_balance
                previous_turnover_source = "derived_from_amount_balance"
                derived_previous_turnover_count += 1
            else:
                implied_previous_avg_balance = previous_disbursement / previous_turnover
                if not math.isclose(
                    previous_avg_balance,
                    implied_previous_avg_balance,
                    rel_tol=_VTG_BALANCE_RECONCILIATION_REL_TOLERANCE,
                    abs_tol=_VTG_BALANCE_RECONCILIATION_ABS_TOLERANCE,
                ):
                    raise RiskAnalysisError(
                        f"第 {row_number} 行 previous_turnover={previous_turnover} "
                        "与 previous_disbursement_amount/previous_avg_daily_balance "
                        "口径不一致。",
                        analysis_kind=kind,
                        field="previous_turnover",
                        row_number=row_number,
                    )

        implied_avg_daily_balance = disbursement / turnover
        if avg_daily_balance is not None and not math.isclose(
            avg_daily_balance,
            implied_avg_daily_balance,
            rel_tol=_VTG_BALANCE_RECONCILIATION_REL_TOLERANCE,
            abs_tol=_VTG_BALANCE_RECONCILIATION_ABS_TOLERANCE,
        ):
            raise RiskAnalysisError(
                f"第 {row_number} 行 avg_daily_balance={avg_daily_balance} 与"
                f"放款金额/周转次数={implied_avg_daily_balance} 不一致；相对容差为 "
                f"{_VTG_BALANCE_RECONCILIATION_REL_TOLERANCE:g}。",
                analysis_kind=kind,
                field="avg_daily_balance",
                row_number=row_number,
            )

        if terminal_method == "min_mob14_auxiliary" and recovery_input is not None:
            raise RiskAnalysisError(
                f"第 {row_number} 行 terminal_method=min_mob14_auxiliary 与 "
                "long_term_recovery_rate 不能同时提供。",
                analysis_kind=kind,
                field="long_term_recovery_rate",
                row_number=row_number,
            )
        if terminal_method == "min_mob14_auxiliary" and direct_terminal is not None:
            raise RiskAnalysisError(
                f"第 {row_number} 行 terminal_method=min_mob14_auxiliary 与 "
                "terminal_bad_rate 不能同时提供。",
                analysis_kind=kind,
                field="terminal_bad_rate",
                row_number=row_number,
            )

        if direct_terminal is not None:
            terminal = direct_terminal
            terminal_source = "direct"
            if recovery_input is not None and mob14 > 0.0:
                direct_realized_recovery = (mob14 - direct_terminal) / mob14
                if not math.isclose(
                    recovery_input,
                    direct_realized_recovery,
                    rel_tol=0.0,
                    abs_tol=_VTG_RECOVERY_RECONCILIATION_TOLERANCE,
                ):
                    raise RiskAnalysisError(
                        f"第 {row_number} 行 long_term_recovery_rate={recovery_input} "
                        f"与直接终值对应的实现回收率={direct_realized_recovery} 不一致；"
                        f"绝对容差为 {_VTG_RECOVERY_RECONCILIATION_TOLERANCE:g}。",
                        analysis_kind=kind,
                        field="long_term_recovery_rate",
                        row_number=row_number,
                    )
                direct_recovery_reconciled_count += 1
            if auxiliary_terminal is not None:
                direct_auxiliary_ignored_count += 1
        else:
            if terminal_method == "min_mob14_auxiliary":
                if auxiliary_terminal is None:
                    raise RiskAnalysisError(
                        f"第 {row_number} 行 terminal_method=min_mob14_auxiliary "
                        "时 auxiliary_terminal_bad_rate 不能为空。",
                        analysis_kind=kind,
                        field="auxiliary_terminal_bad_rate",
                        row_number=row_number,
                    )
                terminal = min(mob14, auxiliary_terminal)
                terminal_source = "min(mob14,auxiliary)"
                explicit_mob_auxiliary_min_count += 1
            elif recovery_input is None:
                raise RiskAnalysisError(
                    f"第 {row_number} 行 terminal_bad_rate 缺失，且没有可用的 long_term_recovery_rate。",
                    analysis_kind=kind,
                    field="terminal_bad_rate",
                    row_number=row_number,
                )
            else:
                derived_terminal = mob14 * (1.0 - recovery_input)
                substitution_count += 1
                if auxiliary_terminal is not None:
                    if selection_rule != "min_auxiliary_recovery":
                        raise RiskAnalysisError(
                            f"第 {row_number} 行同时提供 long_term_recovery_rate 与 "
                            "auxiliary_terminal_bad_rate 时，必须显式设置 "
                            "selection_rule=min_auxiliary_recovery。",
                            analysis_kind=kind,
                            field="selection_rule",
                            row_number=row_number,
                        )
                    terminal = min(derived_terminal, auxiliary_terminal)
                    terminal_source = "min(auxiliary,recovery_derived)"
                    auxiliary_min_count += 1
                else:
                    terminal = derived_terminal
                    terminal_source = "recovery_derived"

        if avg_daily_balance is None:
            avg_daily_balance = implied_avg_daily_balance
            derived_balance_count += 1

        annualized = terminal * turnover
        observed_annualized = mob14 * turnover
        if mob14 == 0.0:
            realized_recovery = None
            zero_mob_count += 1
        else:
            realized_recovery = (mob14 - terminal) / mob14
        if terminal > mob14:
            terminal_above_mob_count += 1
        if annualized > 1.0 or observed_annualized > 1.0:
            annualized_over_one_count += 1

        previous_annualized_source = (
            "direct" if previous_annualized is not None else None
        )
        if previous_terminal is not None and previous_turnover is not None:
            implied_previous_annualized = previous_terminal * previous_turnover
            if previous_annualized is None:
                previous_annualized = implied_previous_annualized
                previous_annualized_source = "derived_from_terminal_turnover"
                derived_previous_annualized_count += 1
            elif not math.isclose(
                previous_annualized,
                implied_previous_annualized,
                rel_tol=0.0,
                abs_tol=_VTG_ANNUALIZED_RECONCILIATION_TOLERANCE,
            ):
                raise RiskAnalysisError(
                    f"第 {row_number} 行 previous_annualized_bad_rate="
                    f"{previous_annualized} 与 previous_terminal_bad_rate × "
                    f"previous_turnover={implied_previous_annualized} 不一致；"
                    f"绝对容差为 {_VTG_ANNUALIZED_RECONCILIATION_TOLERANCE:g}。",
                    analysis_kind=kind,
                    field="previous_annualized_bad_rate",
                    row_number=row_number,
                )
        if previous_annualized is not None and previous_annualized > 1.0:
            annualized_over_one_count += 1
        previous_fields_present = sum(
            value is not None
            for value in (
                previous_mob14,
                previous_terminal,
                previous_turnover,
                previous_annualized,
                previous_disbursement,
                previous_avg_balance,
            )
        )
        if previous_fields_present and previous_annualized is None:
            partial_previous_count += 1
        annualized_change = (
            annualized - previous_annualized
            if previous_annualized is not None
            else None
        )

        detail_rows.append(
            {
                "product": product,
                "cohort": cohort,
                "as_of_date": as_of_date,
                "amount_unit": amount_unit,
                "scenario": scenario,
                "channel": channel,
                "tenor_months": tenor_months,
                "selection_rule": selection_rule,
                "day_count_basis": day_count_basis,
                "mob_balance_source": mob_balance_source,
                "disbursement_amount": disbursement,
                "mob14_bad_rate": mob14,
                "terminal_bad_rate": terminal,
                "terminal_bad_rate_source": terminal_source,
                "terminal_method": terminal_method,
                "auxiliary_terminal_bad_rate": auxiliary_terminal,
                "turnover": turnover,
                "avg_daily_balance": avg_daily_balance,
                "observed_annualized_bad_rate": observed_annualized,
                "annualized_bad_rate": annualized,
                "long_term_recovery_rate": realized_recovery,
                "supplied_long_term_recovery_rate": recovery_input,
                "realized_long_term_recovery_rate": realized_recovery,
                "previous_mob14_bad_rate": previous_mob14,
                "previous_terminal_bad_rate": previous_terminal,
                "previous_turnover": previous_turnover,
                "previous_turnover_source": previous_turnover_source,
                "previous_annualized_bad_rate": previous_annualized,
                "previous_annualized_bad_rate_source": previous_annualized_source,
                "previous_disbursement_amount": previous_disbursement,
                "previous_avg_daily_balance": previous_avg_balance,
                "annualized_bad_rate_change": annualized_change,
            }
        )

    seen_vtg_slices: set[tuple[Any, ...]] = set()
    for row in detail_rows:
        slice_key = (
            row["product"],
            row["cohort"],
            row["as_of_date"],
            row.get("scenario") or "基准",
            row.get("channel") or "未提供",
            row.get("tenor_months"),
        )
        if slice_key in seen_vtg_slices:
            raise RiskAnalysisError(
                "VTG 汇总粒度必须是唯一的 product × cohort × as_of_date × "
                "scenario × channel × tenor_months；发现重复业务切片 "
                f"{slice_key!r}。不同终值方法是互斥口径，不能作为可加行混合汇总。",
                analysis_kind=kind,
                field="cohort",
            )
        seen_vtg_slices.add(slice_key)

    amount_units = _ordered_unique(row["amount_unit"] for row in detail_rows)
    if len(amount_units) != 1:
        raise RiskAnalysisError(
            "VTG 组合汇总要求 amount_unit 全表一致，实际为："
            + "、".join(amount_units)
            + "。",
            analysis_kind=kind,
            field="amount_unit",
        )
    scenario_scope = _ordered_unique(
        row.get("scenario") or "基准" for row in detail_rows
    )
    if len(scenario_scope) > 1:
        raise RiskAnalysisError(
            "VTG 单份组合报告不允许混合多个 scenario，以免替代场景重复计入组合；"
            "请按场景分别生成报告。",
            analysis_kind=kind,
            field="scenario",
        )
    as_of_date_scope = _ordered_unique(row["as_of_date"] for row in detail_rows)
    if len(as_of_date_scope) > 1:
        raise RiskAnalysisError(
            "VTG 单份组合报告不允许混合多个 as_of_date，以免重复快照计入组合；"
            "请按截面日期分别生成报告。",
            analysis_kind=kind,
            field="as_of_date",
        )
    summary_rows = _vtg_product_summaries(detail_rows)
    total_disbursement = sum(row["disbursement_amount"] for row in detail_rows)
    total_avg_balance = sum(row["avg_daily_balance"] for row in detail_rows)
    terminal_loss = sum(
        row["terminal_bad_rate"] * row["disbursement_amount"] for row in detail_rows
    )
    observed_loss = sum(
        row["mob14_bad_rate"] * row["disbursement_amount"] for row in detail_rows
    )
    weighted_terminal = _safe_ratio(terminal_loss, total_disbursement)
    weighted_mob14 = _safe_ratio(observed_loss, total_disbursement)
    portfolio_annualized = _safe_ratio(terminal_loss, total_avg_balance)
    portfolio_observed_annualized = _safe_ratio(observed_loss, total_avg_balance)
    portfolio_turnover = _safe_ratio(total_disbursement, total_avg_balance)

    highest = max(detail_rows, key=lambda item: item["annualized_bad_rate"])
    key_points = [
        (
            f"最高年化不良：产品 {highest['product']} / cohort {highest['cohort']} 为 "
            f"{_format_percent(highest['annualized_bad_rate'])}。"
        )
    ]
    if portfolio_annualized is not None:
        key_points.insert(
            0, f"组合加权年化不良率为 {_format_percent(portfolio_annualized)}。"
        )
    change_rows = [
        row for row in detail_rows if row["annualized_bad_rate_change"] is not None
    ]
    if change_rows:
        largest_change = max(
            change_rows, key=lambda item: abs(item["annualized_bad_rate_change"])
        )
        key_points.append(
            "年化不良变化最大：产品 "
            f"{largest_change['product']} / cohort {largest_change['cohort']}，"
            f"较上期 {_format_change(largest_change['annualized_bad_rate_change'])}。"
        )

    red_flags: list[str] = []
    if zero_disbursement_skipped_count:
        red_flags.append(
            f"{zero_disbursement_skipped_count} 行零放款金额未参与测算，"
            "已作为未发生/未来 cohort 占位行记入数据质量。"
        )
    if terminal_above_mob_count:
        red_flags.append(
            f"{terminal_above_mob_count} 行终值不良率高于 MOB14 不良率，请核查终值与观察口径。"
        )
    if substitution_count:
        red_flags.append(
            f"{substitution_count} 行缺少 terminal_bad_rate，已使用 MOB14 与长期回收率推导。"
        )
    if auxiliary_min_count:
        red_flags.append(
            f"{auxiliary_min_count} 行同时存在辅助终值和回收推导值，按两者最小值取终值。"
        )
    if direct_auxiliary_ignored_count:
        red_flags.append(
            f"{direct_auxiliary_ignored_count} 行同时提供直接终值和辅助终值；"
            "按直接终值优先，辅助终值仅保留用于追溯且未参与计算。"
        )
    if derived_balance_count:
        red_flags.append(
            f"{derived_balance_count} 行缺少日均余额，已按放款金额/周转次数替代。"
        )
    if annualized_over_one_count:
        red_flags.append(
            f"{annualized_over_one_count} 项当前、观察或上期年化不良率超过 100%，"
            "请核查周转次数和率口径。"
        )
    if zero_mob_count:
        red_flags.append(f"{zero_mob_count} 行 MOB14 不良率为 0，长期回收率无法计算。")
    if partial_previous_count:
        red_flags.append(
            f"{partial_previous_count} 行上期字段不完整，未计算年化不良变化。"
        )

    assumptions = [
        "terminal_bad_rate 缺失时，终值不良率 = MOB14 不良率 × (1 - 长期回收率)。",
        (
            "terminal_bad_rate 缺失且辅助终值、回收推导值同时存在时，"
            "仅在显式 selection_rule=min_auxiliary_recovery 下取两者最小值。"
        ),
        (
            "仅当 terminal_method=min_mob14_auxiliary 且直接终值、长期回收率均未提供时，"
            "终值按 min(MOB14 不良率, 辅助终值) 计算。"
        ),
        "年化不良率 = 终值不良率 × 周转次数；观察年化不良率 = MOB14 不良率 × 周转次数。",
        "日均余额缺失时，按放款金额 ÷ 周转次数推导。",
        (
            "汇总模式同时提供日均余额与周转次数时，要求日均余额与放款金额 ÷ 周转次数一致；"
            f"相对容差 {_VTG_BALANCE_RECONCILIATION_REL_TOLERANCE:g}，绝对容差 "
            f"{_VTG_BALANCE_RECONCILIATION_ABS_TOLERANCE:g}。"
        ),
        (
            "上期周转次数与上期金额/日均余额同时提供时沿用余额勾稽容差；"
            "上期年化不良率与上期终值/周转次数同时提供时按绝对容差 "
            f"{_VTG_ANNUALIZED_RECONCILIATION_TOLERANCE:g}（1 bp）勾稽。"
        ),
        "长期回收率 = (MOB14 不良率 - 终值不良率) ÷ MOB14 不良率。",
        "组合年化不良率按终值损失金额 ÷ 日均余额汇总计算。",
        (
            "mob14_bad_rate 必须已经是实际观察到 MOB14 的值，或由上游按明确方法投影至 "
            "MOB14；本工具不会用未成熟原始 MOB 值自行外推。"
        ),
        f"金额字段统一使用 amount_unit={amount_units[0]}。",
    ]
    if curve_derived_count:
        balance_basis = (
            "余额金额" if "mob_balance_amount" in mapping else "放款金额 × 余额率"
        )
        day_count_bases = _ordered_unique(
            f"{row['day_count_basis']:g}" for row in detail_rows
        )
        assumptions.append(
            f"原始 MOB 曲线按各组显式 day_count_basis（本次：{'、'.join(day_count_bases)}）计算："
            f"日均余额 = Σ({balance_basis} × MOB 天数) ÷ day_count_basis，"
            "周转次数 = 放款金额 ÷ 日均余额。mob_balance_rate/mob_balance_amount "
            "必须是各 MOB 区间的平均日余额率/金额，不得直接使用月末时点余额。"
        )
    if zero_disbursement_skipped_count:
        assumptions.append("零放款金额行是占位数据，不进入终值、周转和组合汇总。")
    if declared_terminal_methods:
        assumptions.append(
            "数据声明的 terminal_method 已保留在明细中用于口径追溯："
            + "、".join(_ordered_unique(declared_terminal_methods))
            + "；实际取值优先级仍以 terminal_bad_rate_source 为准。"
        )
    if direct_recovery_reconciled_count:
        assumptions.append(
            f"{direct_recovery_reconciled_count} 行同时提供直接终值与长期回收率；"
            "已保留 supplied_long_term_recovery_rate 与 "
            "realized_long_term_recovery_rate，并按绝对容差 "
            f"{_VTG_RECOVERY_RECONCILIATION_TOLERANCE:g}（1 bp）完成勾稽。"
        )
    if explicit_mob_auxiliary_min_count:
        assumptions.append(
            f"{explicit_mob_auxiliary_min_count} 行按显式 "
            "terminal_method=min_mob14_auxiliary 计算，未隐式套用该规则。"
        )
    if derived_previous_turnover_count:
        assumptions.append(
            f"{derived_previous_turnover_count} 行上期周转次数按上期放款金额 ÷ "
            "上期日均余额推导，来源记录为 derived_from_amount_balance。"
        )
    if derived_previous_annualized_count:
        assumptions.append(
            f"{derived_previous_annualized_count} 行上期年化不良率按上期终值不良率 × "
            "上期周转次数推导，来源记录为 derived_from_terminal_turnover。"
        )
    if not change_rows:
        assumptions.append("未提供完整的上期年化不良口径，未生成变化最大项。")
    if total_disbursement == 0.0 or total_avg_balance == 0.0:
        assumptions.append("组合金额分母为 0，对应加权指标留空。")

    headline_metrics = {
        "product_count": len({row["product"] for row in detail_rows}),
        "cohort_count": len({row["cohort"] for row in detail_rows}),
        "total_disbursement_amount": total_disbursement,
        "total_avg_daily_balance": total_avg_balance,
        "portfolio_turnover": portfolio_turnover,
        "weighted_mob14_bad_rate": weighted_mob14,
        "weighted_terminal_bad_rate": weighted_terminal,
        "observed_annualized_bad_rate": portfolio_observed_annualized,
        "annualized_bad_rate": portfolio_annualized,
        "highest_annualized_bad_rate": highest["annualized_bad_rate"],
        "highest_annualized_product": highest["product"],
        "highest_annualized_cohort": highest["cohort"],
    }
    all_products = _ordered_unique(row["product"] for row in detail_rows)
    product_scope = _product_scope(all_products)
    if len(all_products) > PRODUCT_SCOPE_LIMIT:
        assumptions.append(
            f"产品共 {len(all_products)} 个；DONE metadata 的 product_scope 仅保留前 "
            f"{PRODUCT_SCOPE_LIMIT} 个，报表明细未截断。"
        )
    as_of_period = _period_scope(row["as_of_date"] for row in detail_rows)
    formula_definitions = [
        {
            "metric": "terminal_bad_rate",
            "formula": (
                "direct terminal; otherwise min(mob14_bad_rate, auxiliary_terminal) "
                "when terminal_method=min_mob14_auxiliary; otherwise "
                "min(auxiliary_terminal, mob14_bad_rate * (1 - long_term_recovery_rate)) "
                "when both recovery substitutes exist"
            ),
            "note": (
                "直接终值优先；替代仅在直接终值缺失时启用；输入 terminal_method "
                "作为业务方法声明保留在明细中。"
            ),
        },
        {
            "metric": "annualized_bad_rate",
            "formula": "terminal_bad_rate * turnover",
            "note": "单行年化不良率。",
        },
        {
            "metric": "avg_daily_balance",
            "formula": (
                "sum(mob_balance_amount * mob_days) / day_count_basis, or "
                "disbursement_amount * sum(mob_balance_rate * mob_days) / day_count_basis; "
                "otherwise disbursement_amount / turnover when missing"
            ),
            "note": (
                "曲线余额必须是 MOB 区间平均日余额而非月末余额；汇总模式仅在日均余额"
                "缺失时按周转次数倒推。"
            ),
        },
        {
            "metric": "long_term_recovery_rate",
            "formula": "(mob14_bad_rate - terminal_bad_rate) / mob14_bad_rate",
            "note": (
                "MOB14 为 0 时留空；输入值另存为 supplied_long_term_recovery_rate，"
                "实现值另存为 realized_long_term_recovery_rate。"
            ),
        },
        {
            "metric": "previous_turnover",
            "formula": (
                "previous_disbursement_amount / previous_avg_daily_balance "
                "when previous_turnover is missing"
            ),
            "note": "上期日均余额必须大于 0；推导来源保留在明细中。",
        },
        {
            "metric": "previous_annualized_bad_rate",
            "formula": (
                "previous_terminal_bad_rate * previous_turnover "
                "when previous_annualized_bad_rate is missing"
            ),
            "note": "推导来源保留在明细中。",
        },
    ]
    data_quality = [
        {
            "check": "源数据行数",
            "status": "PASS",
            "detail": f"已校验 {source_row_count} 行源数据。",
        },
        {
            "check": "结果行数",
            "status": "PASS",
            "detail": f"已生成 {len(detail_rows)} 行产品/cohort 结果。",
        },
        {
            "check": "输入率范围",
            "status": "PASS",
            "detail": (
                "概率型输入率均在 [0, 1]；previous_annualized_bad_rate 按非负年化率校验，"
                "允许超过 100% 并生成风险提示。"
            ),
        },
        {"check": "金额范围", "status": "PASS", "detail": "所有金额均为非负数。"},
        {"check": "周转次数", "status": "PASS", "detail": "所有周转次数均大于 0。"},
        {
            "check": "缺失字段替代",
            "status": "WARN" if substitution_count or derived_balance_count else "PASS",
            "detail": f"终值替代 {substitution_count} 行；日均余额替代 {derived_balance_count} 行。",
        },
    ]
    if curve_derived_count:
        basis_scope = _ordered_unique(
            f"{row['day_count_basis']:g}" for row in detail_rows
        )
        data_quality.append(
            {
                "check": "MOB 余额曲线",
                "status": "PASS",
                "detail": (
                    f"已校验并按 day_count_basis={','.join(basis_scope)} 汇总 "
                    f"{curve_derived_count} 个产品/cohort 曲线。"
                ),
            }
        )
    if zero_disbursement_skipped_count:
        data_quality.append(
            {
                "check": "零放款占位行",
                "status": "WARN",
                "detail": (
                    f"源数据中 {zero_disbursement_skipped_count} 行放款金额为 0，"
                    "已跳过且未进入任何加权分母。"
                ),
            }
        )
    return RiskAnalysisCalculation(
        analysis_kind=kind,
        column_map=mapping,
        product_scope=product_scope,
        as_of_period=as_of_period,
        headline_metrics=headline_metrics,
        key_points=key_points,
        red_flags=_unique_strings(red_flags),
        assumptions=_unique_strings(assumptions),
        source_row_count=source_row_count,
        row_count=len(detail_rows),
        detail_rows=detail_rows,
        summary_rows=summary_rows,
        formula_definitions=formula_definitions,
        data_quality=data_quality,
    )


def calculate_profitability(
    frame: pd.DataFrame,
    *,
    column_map: dict[str, str],
) -> RiskAnalysisCalculation:
    kind = "profitability"
    data = _normalized_frame(frame, analysis_kind=kind)
    source_row_count = len(data)
    mapping = _normalize_column_map(data, column_map, analysis_kind=kind)
    _require_mappings(mapping, _PROFIT_REQUIRED, analysis_kind=kind)
    working_mapping = mapping
    customer_stage_group_count = 0
    if "customer_stage" in mapping:
        _require_mappings(mapping, ("transaction_weight",), analysis_kind=kind)
        data, working_mapping = _collapse_profitability_customer_stages(
            data,
            mapping=mapping,
            analysis_kind=kind,
        )
        customer_stage_group_count = len(data)

    detail_rows: list[dict[str, Any]] = []
    scenario_missing_count = 0
    derived_cost_counts = {field: 0 for field in _PROFIT_DERIVABLE_COST_FIELDS}
    reconciled_cost_counts = {field: 0 for field in _PROFIT_DERIVABLE_COST_FIELDS}
    supplied_driver_values: dict[str, list[Any]] = {
        field: [] for field in _PROFIT_DRIVER_FIELDS
    }

    for position, row in data.iterrows():
        row_number = position + 1
        product = _required_text(row, working_mapping, "product", kind, row_number)
        asset_class = _required_text(
            row, working_mapping, "asset_class", kind, row_number
        )
        as_of_period_value = _required_text(
            row, working_mapping, "as_of_period", kind, row_number
        )
        scenario = _optional_text(row, working_mapping, "scenario")
        if scenario is None:
            scenario_missing_count += 1
            scenario = "基准"
        weight = _number(
            row,
            working_mapping,
            "weight",
            kind,
            row_number,
            required=True,
            rate=True,
        )
        weight_basis = _required_text(
            row, working_mapping, "weight_basis", kind, row_number
        )
        if weight_basis != "average_balance":
            raise RiskAnalysisError(
                f"第 {row_number} 行 weight_basis={weight_basis!r}；"
                "收益测算仅支持 average_balance。",
                analysis_kind=kind,
                field="weight_basis",
                row_number=row_number,
            )
        customer_rate = _number(
            row,
            working_mapping,
            "customer_rate",
            kind,
            row_number,
            required=True,
            rate=True,
        )
        costs, cost_sources, drivers = _resolve_profitability_costs(
            row,
            mapping=working_mapping,
            analysis_kind=kind,
            row_number=row_number,
            customer_rate=customer_rate,
        )
        stage_data_source = _optional_text(
            row, working_mapping, "stage_data_cost_source"
        )
        customer_stage_provenance = _optional_text(
            row, working_mapping, "customer_stage_provenance"
        )
        customer_stage_count = _number(
            row,
            working_mapping,
            "customer_stage_count",
            kind,
            row_number,
            minimum=0.0,
            minimum_inclusive=False,
        )
        transaction_weight_sum = _number(
            row,
            working_mapping,
            "transaction_weight_sum",
            kind,
            row_number,
            minimum=0.0,
            minimum_inclusive=False,
        )
        if stage_data_source is not None:
            cost_sources["data_cost_rate"] = stage_data_source
            drivers["amount_unit"] = _optional_text(row, working_mapping, "amount_unit")
        for field in _PROFIT_DERIVABLE_COST_FIELDS:
            if cost_sources[field].startswith("derived_"):
                derived_cost_counts[field] += 1
            elif cost_sources[field].startswith("explicit_reconciled_"):
                reconciled_cost_counts[field] += 1
        for field, value in drivers.items():
            if value is not None:
                supplied_driver_values[field].append(value)

        fixed_income_yield = (
            customer_rate
            - costs["interest_loss_rate"]
            - costs["revenue_share_rate"]
            - costs["risk_cost_rate"]
            - costs["acquisition_cost_rate"]
        )
        total_cost_rate = sum(costs[field] for field in _PROFIT_COST_FIELDS)
        net_yield = customer_rate - total_cost_rate
        detail_rows.append(
            {
                "product": product,
                "asset_class": asset_class,
                "as_of_period": as_of_period_value,
                "scenario": scenario,
                "weight": weight,
                "weight_basis": weight_basis,
                "customer_rate": customer_rate,
                **{field: costs[field] for field in _PROFIT_COST_FIELDS},
                **{
                    f"{field}_source": cost_sources[field]
                    for field in _PROFIT_DERIVABLE_COST_FIELDS
                },
                **{field: drivers[field] for field in _PROFIT_DRIVER_FIELDS},
                "customer_stage_count": customer_stage_count,
                "transaction_weight_sum": transaction_weight_sum,
                "customer_stage_provenance": customer_stage_provenance,
                "fixed_income_yield": fixed_income_yield,
                "total_cost_rate": total_cost_rate,
                "net_yield": net_yield,
            }
        )

    seen_profit_slices: set[tuple[str, str, str, str]] = set()
    for row in detail_rows:
        slice_key = (
            row["product"],
            row["as_of_period"],
            row["scenario"],
            row["asset_class"],
        )
        if slice_key in seen_profit_slices:
            raise RiskAnalysisError(
                "收益测算粒度必须是唯一的 product × as_of_period × scenario × "
                f"asset_class；发现重复业务切片 {slice_key!r}。",
                analysis_kind=kind,
                field="asset_class",
            )
        seen_profit_slices.add(slice_key)

    summary_rows = _profit_product_summaries(detail_rows, analysis_kind=kind)
    scenario_spreads: list[dict[str, Any]] = []
    scenario_group_keys = list(
        dict.fromkeys((row["product"], row["as_of_period"]) for row in summary_rows)
    )
    for product, period in scenario_group_keys:
        scenario_rows = [
            row
            for row in summary_rows
            if row["product"] == product and row["as_of_period"] == period
        ]
        if len({row["scenario"] for row in scenario_rows}) < 2:
            continue
        high = max(scenario_rows, key=lambda item: item["net_yield"])
        low = min(scenario_rows, key=lambda item: item["net_yield"])
        spread = high["net_yield"] - low["net_yield"]
        if math.isclose(spread, 0.0, rel_tol=0.0, abs_tol=1e-12):
            continue
        scenario_spreads.append(
            {
                "product": product,
                "as_of_period": period,
                "spread": spread,
                "high_scenario": high["scenario"],
                "low_scenario": low["scenario"],
            }
        )
    largest_scenario_spread = (
        max(scenario_spreads, key=lambda item: item["spread"])
        if scenario_spreads
        else None
    )
    lowest = min(summary_rows, key=lambda item: item["net_yield"])
    highest = max(summary_rows, key=lambda item: item["net_yield"])
    max_cost_product: str | None = None
    max_cost_field: str | None = None
    max_cost_rate = -math.inf
    for summary in summary_rows:
        for field in _PROFIT_COST_FIELDS:
            value = summary[field]
            if value > max_cost_rate:
                max_cost_rate = value
                max_cost_field = field
                max_cost_product = summary["product"]
    max_cost_slice = max(
        summary_rows,
        key=lambda item: item[max_cost_field or "risk_cost_rate"],
    )

    key_points = [
        f"净收益率最低：{_profit_slice_identity(lowest)} 为 {_format_percent(lowest['net_yield'])}。",
        (
            f"最大成本项：{_profit_slice_identity(max_cost_slice)} 的 "
            f"{_COST_LABELS.get(max_cost_field or '', max_cost_field)} "
            f"为 {_format_percent(max_cost_rate)}。"
        ),
        f"净收益率最高：{_profit_slice_identity(highest)} 为 {_format_percent(highest['net_yield'])}。",
    ]
    if largest_scenario_spread is not None:
        key_points.append(
            "场景净收益率差异最大：产品 "
            f"{largest_scenario_spread['product']} / 期间 "
            f"{largest_scenario_spread['as_of_period']}，高场景 "
            f"{largest_scenario_spread['high_scenario']}、低场景 "
            f"{largest_scenario_spread['low_scenario']}，相差 "
            f"{largest_scenario_spread['spread'] * 100:.2f} 个百分点。"
        )
    red_flags = [
        f"{_profit_slice_identity(row)} 的加权净收益率为 {_format_percent(row['net_yield'])}，低于 0。"
        for row in summary_rows
        if row["net_yield"] < 0.0
    ]
    negative_fixed = [row for row in summary_rows if row["fixed_income_yield"] < 0.0]
    for row in negative_fixed:
        red_flags.append(
            f"{_profit_slice_identity(row)} 的类固收收益率为 "
            f"{_format_percent(row['fixed_income_yield'])}，低于 0。"
        )

    assumptions = [
        (
            "类固收收益率 = 对客利率 - 息费损失率 - 分润成本率 - 风险成本率 - 获客成本率；"
            "分润成本率必须已换算为资产收益率口径，不得直接传合同分润比例。"
        ),
        (
            "净收益率 = 对客利率 - 息费损失率 - 分润成本率 - 风险成本率 - 获客成本率 - "
            "数据成本率 - 支付成本率 - 催收成本率 - 资金成本率 - 其他成本率 - 税率。"
        ),
        (
            "每个产品、数据期间、场景组合内的资产类别权重和须在 "
            f"1±{WEIGHT_SUM_TOLERANCE:g} 内。"
        ),
        "各成本字段均为占资产余额的年化率，不能混入金额或合同分成比例。",
        "weight_basis 必须为 average_balance；产品/期间/场景权重按平均余额口径聚合。",
    ]
    if derived_cost_counts["risk_cost_rate"]:
        assumptions.append(
            "风险成本率 = terminal_vintage_rate × risk_turnover；本次提供的 "
            f"terminal_vintage_rate={_format_value_set(supplied_driver_values['terminal_vintage_rate'], percent=True)}，"
            f"risk_turnover={_format_value_set(supplied_driver_values['risk_turnover'])}。"
        )
    if derived_cost_counts["interest_loss_rate"]:
        assumptions.append(
            "息费损失率 = customer_rate × resolved_risk_cost_rate × loss_timing_factor；"
            "本次提供的 loss_timing_factor="
            f"{_format_value_set(supplied_driver_values['loss_timing_factor'])}。"
        )
    if derived_cost_counts["revenue_share_rate"]:
        assumptions.append(
            "分润成本率 = (customer_rate - resolved_interest_loss_rate) × "
            "profit_share_ratio；本次提供的 profit_share_ratio="
            f"{_format_value_set(supplied_driver_values['profit_share_ratio'], percent=True)}。"
        )
    if derived_cost_counts["data_cost_rate"]:
        data_amount_units = _ordered_unique(
            str(value) for value in supplied_driver_values["amount_unit"]
        )
        if customer_stage_group_count:
            stage_descriptions = _ordered_unique(
                row["customer_stage_provenance"]
                for row in detail_rows
                if row.get("customer_stage_provenance")
            )
            assumptions.append(
                "客户阶段数据成本先逐阶段按漏斗公式推导，再按 transaction_weight 加权；"
                f"已将 {source_row_count} 行阶段源数据折叠为 "
                f"{customer_stage_group_count} 行资产结果。阶段权重："
                + "；".join(stage_descriptions)
                + f"。金额单位={'、'.join(data_amount_units)}。"
            )
        else:
            assumptions.append(
                "数据成本率 = per_application_cost ÷ (credit_approval_rate × "
                "draw_initiation_rate × draw_approval_rate × average_ticket) × "
                "data_annualization_factor；本次提供的 data_annualization_factor="
                f"{_format_value_set(supplied_driver_values['data_annualization_factor'])}，"
                f"per_application_cost 与 average_ticket 的 amount_unit="
                f"{'、'.join(data_amount_units)}。"
            )
    if derived_cost_counts["tax_rate"]:
        assumptions.append(
            "tax_method=sample_net_revenue_vat_surcharge 时，税基严格按样例 D12 口径 = "
            "customer_rate - interest_loss_rate - revenue_share_rate - acquisition_cost_rate - "
            "data_cost_rate - payment_cost_rate - collection_cost_rate；"
            "税率 = 税基 ÷ tax_inclusive_divisor × tax_combined_rate。本次提供的 "
            f"tax_inclusive_divisor={_format_value_set(supplied_driver_values['tax_inclusive_divisor'])}，"
            f"tax_combined_rate={_format_value_set(supplied_driver_values['tax_combined_rate'], percent=True)}。"
        )
    if any(reconciled_cost_counts.values()):
        assumptions.append(
            "显式成本率与其推导驱动项同时提供时，已逐行勾稽；绝对容差 "
            f"{PROFIT_RATE_RECONCILIATION_TOLERANCE:g}（1 bp）。勾稽行数："
            + "；".join(
                f"{field} {count} 行"
                for field, count in reconciled_cost_counts.items()
                if count
            )
            + "。"
        )
    as_of_period = _period_scope(row["as_of_period"] for row in detail_rows)
    if scenario_missing_count:
        assumptions.append(
            f"收益测算场景列有 {scenario_missing_count} 行缺失，这些行归入“基准”场景。"
        )

    all_products = _ordered_unique(row["product"] for row in detail_rows)
    product_scope = _product_scope(all_products)
    if len(all_products) > PRODUCT_SCOPE_LIMIT:
        assumptions.append(
            f"产品共 {len(all_products)} 个；DONE metadata 的 product_scope 仅保留前 "
            f"{PRODUCT_SCOPE_LIMIT} 个，报表明细未截断。"
        )

    headline_metrics = {
        "product_count": len({row["product"] for row in summary_rows}),
        "analysis_slice_count": len(summary_rows),
        "negative_product_count": len(
            {row["product"] for row in summary_rows if row["net_yield"] < 0.0}
        ),
        "lowest_net_yield": lowest["net_yield"],
        "lowest_net_yield_product": lowest["product"],
        "lowest_net_yield_as_of_period": lowest["as_of_period"],
        "lowest_net_yield_scenario": lowest["scenario"],
        "highest_net_yield": highest["net_yield"],
        "highest_net_yield_product": highest["product"],
        "highest_net_yield_as_of_period": highest["as_of_period"],
        "highest_net_yield_scenario": highest["scenario"],
        "max_cost_rate": max_cost_rate,
        "max_cost_component": max_cost_field,
        "max_cost_product": max_cost_product,
        "max_cost_as_of_period": max_cost_slice["as_of_period"],
        "max_cost_scenario": max_cost_slice["scenario"],
    }
    if largest_scenario_spread is not None:
        headline_metrics.update(
            {
                "largest_scenario_net_yield_spread": largest_scenario_spread["spread"],
                "largest_scenario_net_yield_spread_product": _bounded_text(
                    largest_scenario_spread["product"],
                    PRODUCT_SCOPE_ITEM_MAX_CHARS,
                ),
                "largest_scenario_net_yield_spread_as_of_period": _bounded_text(
                    largest_scenario_spread["as_of_period"],
                    AS_OF_PERIOD_MAX_CHARS,
                ),
                "largest_scenario_net_yield_spread_high_scenario": _bounded_text(
                    largest_scenario_spread["high_scenario"], SCENARIO_MAX_CHARS
                ),
                "largest_scenario_net_yield_spread_low_scenario": _bounded_text(
                    largest_scenario_spread["low_scenario"], SCENARIO_MAX_CHARS
                ),
            }
        )
    formula_definitions = [
        {
            "metric": "risk_cost_rate",
            "formula": (
                "explicit risk_cost_rate; otherwise terminal_vintage_rate * "
                "risk_turnover"
            ),
            "note": "明细的 risk_cost_rate_source 标记显式或推导来源。",
        },
        {
            "metric": "interest_loss_rate",
            "formula": (
                "explicit interest_loss_rate; otherwise customer_rate * "
                "resolved_risk_cost_rate * loss_timing_factor"
            ),
            "note": "loss_timing_factor 无默认值，必须由输入提供。",
        },
        {
            "metric": "revenue_share_rate",
            "formula": (
                "explicit revenue_share_rate; otherwise (customer_rate - "
                "resolved_interest_loss_rate) * profit_share_ratio"
            ),
            "note": (
                "profit_share_ratio 是合同分成比例，与 revenue_share_rate 是同一成本的"
                "原始/年化表达；acquisition_cost_rate 只用于与该合同分润不同的独立获客成本。"
            ),
        },
        {
            "metric": "data_cost_rate",
            "formula": (
                "explicit data_cost_rate; otherwise per_application_cost / "
                "(credit_approval_rate * draw_initiation_rate * draw_approval_rate * "
                "average_ticket) * data_annualization_factor"
            ),
            "note": "漏斗率、客单价和年化因子必须显式提供且分母大于 0。",
        },
        {
            "metric": "tax_rate",
            "formula": (
                "explicit tax_rate; otherwise net_revenue_tax_base / "
                "tax_inclusive_divisor * tax_combined_rate when "
                "tax_method=sample_net_revenue_vat_surcharge"
            ),
            "note": "推导税基为负时拒绝计算，不隐式设置为 0。",
        },
        {
            "metric": "fixed_income_yield",
            "formula": "customer_rate - interest_loss_rate - revenue_share_rate - risk_cost_rate - acquisition_cost_rate",
            "note": "分润字段须为已换算成本率；该指标未扣资金和运营类成本。",
        },
        {
            "metric": "net_yield",
            "formula": "customer_rate - interest_loss_rate - revenue_share_rate - risk_cost_rate - acquisition_cost_rate - data_cost_rate - payment_cost_rate - collection_cost_rate - funding_cost_rate - other_cost_rate - tax_rate",
            "note": "最终净收益率。",
        },
        {
            "metric": "product_weighted_yield",
            "formula": "sum(asset_class_yield * weight) within product, as_of_period, and scenario",
            "note": (
                "weight_basis=average_balance；产品/期间/场景组合内权重和允许误差 "
                f"{WEIGHT_SUM_TOLERANCE:g}。"
            ),
        },
    ]
    data_quality = [
        {
            "check": "源数据行数",
            "status": "PASS",
            "detail": f"已校验 {source_row_count} 行源数据。",
        },
        {
            "check": "结果行数",
            "status": "PASS",
            "detail": f"已生成 {len(detail_rows)} 行收益测算结果。",
        },
        {
            "check": "输入率范围",
            "status": "PASS",
            "detail": "所有输入率和权重均在 [0, 1]。",
        },
        {
            "check": "产品/期间/场景权重",
            "status": "PASS",
            "detail": f"所有产品/期间/场景组合的权重和均在 1±{WEIGHT_SUM_TOLERANCE:g}。",
        },
        {
            "check": "成本字段完整性",
            "status": "PASS",
            "detail": (
                "所有净收益成本字段均已显式提供或由完整驱动项推导；"
                "显式 0 按零成本处理。"
            ),
        },
        {
            "check": "成本字段来源",
            "status": "PASS",
            "detail": "；".join(
                f"{field} 推导 {derived_cost_counts[field]} 行，"
                f"显式值勾稽 {reconciled_cost_counts[field]} 行"
                for field in _PROFIT_DERIVABLE_COST_FIELDS
            ),
        },
    ]
    return RiskAnalysisCalculation(
        analysis_kind=kind,
        column_map=mapping,
        product_scope=product_scope,
        as_of_period=as_of_period,
        headline_metrics=headline_metrics,
        key_points=key_points,
        red_flags=_unique_strings(red_flags),
        assumptions=_unique_strings(assumptions),
        source_row_count=source_row_count,
        row_count=len(detail_rows),
        detail_rows=detail_rows,
        summary_rows=summary_rows,
        formula_definitions=formula_definitions,
        data_quality=data_quality,
    )


def _collapse_profitability_customer_stages(
    data: pd.DataFrame,
    *,
    mapping: dict[str, str],
    analysis_kind: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    varying_fields = {
        "customer_stage",
        "transaction_weight",
        "per_application_cost",
        "credit_approval_rate",
        "draw_initiation_rate",
        "draw_approval_rate",
        "average_ticket",
        "data_annualization_factor",
        "data_cost_rate",
    }
    groups: dict[tuple[str, str, str, str], list[int]] = {}
    for position, row in data.iterrows():
        row_number = position + 1
        product = _required_text(row, mapping, "product", analysis_kind, row_number)
        asset_class = _required_text(
            row, mapping, "asset_class", analysis_kind, row_number
        )
        period = _optional_text(row, mapping, "as_of_period") or "未提供"
        scenario = _optional_text(row, mapping, "scenario") or "基准"
        groups.setdefault((product, period, scenario, asset_class), []).append(position)

    data_cost_column = _unused_column_name(data, "__stage_data_cost_rate")
    stage_count_column = _unused_column_name(data, "__customer_stage_count")
    weight_sum_column = _unused_column_name(data, "__transaction_weight_sum")
    provenance_column = _unused_column_name(data, "__customer_stage_provenance")
    source_column = _unused_column_name(data, "__stage_data_cost_source")
    collapsed_rows: list[pd.Series] = []

    for (product, period, scenario, asset_class), positions in groups.items():
        group = data.loc[positions]
        for field, source in mapping.items():
            if field in varying_fields:
                continue
            tokens = {_consistency_token(value) for value in group[source].tolist()}
            if len(tokens) != 1:
                raise RiskAnalysisError(
                    f"产品 {product} / 期间 {period} / 场景 {scenario} / 资产 "
                    f"{asset_class} 的 {field} 在客户阶段行之间必须一致。",
                    analysis_kind=analysis_kind,
                    field=field,
                    row_number=positions[0] + 1,
                )
        if "data_cost_rate" in mapping and any(
            not _is_missing(value) for value in group[mapping["data_cost_rate"]]
        ):
            raise RiskAnalysisError(
                f"产品 {product} / 期间 {period} / 场景 {scenario} / 资产 "
                f"{asset_class} 已提供 customer_stage 时不能同时提供 data_cost_rate。",
                analysis_kind=analysis_kind,
                field="data_cost_rate",
                row_number=positions[0] + 1,
            )

        weighted_data_cost = 0.0
        transaction_weight_sum = 0.0
        provenance: list[str] = []
        for position in positions:
            row = data.loc[position]
            row_number = position + 1
            stage = _required_text(
                row, mapping, "customer_stage", analysis_kind, row_number
            )
            transaction_weight = _number(
                row,
                mapping,
                "transaction_weight",
                analysis_kind,
                row_number,
                required=True,
                minimum=0.0,
                minimum_inclusive=False,
            )
            assert transaction_weight is not None
            stage_data_cost, _ = _derive_profitability_data_cost(
                row,
                mapping=mapping,
                analysis_kind=analysis_kind,
                row_number=row_number,
            )
            weighted_data_cost += stage_data_cost * transaction_weight
            transaction_weight_sum += transaction_weight
            provenance.append(
                f"{stage}(weight={transaction_weight:g},data_cost_rate={stage_data_cost:.6%})"
            )

        collapsed = data.loc[positions[0]].copy()
        collapsed[data_cost_column] = weighted_data_cost / transaction_weight_sum
        collapsed[stage_count_column] = len(positions)
        collapsed[weight_sum_column] = transaction_weight_sum
        collapsed[provenance_column] = "；".join(provenance)
        collapsed[source_column] = "derived_weighted_customer_stages"
        collapsed_rows.append(collapsed)

    working_mapping = dict(mapping)
    working_mapping["data_cost_rate"] = data_cost_column
    working_mapping["customer_stage_count"] = stage_count_column
    working_mapping["transaction_weight_sum"] = weight_sum_column
    working_mapping["customer_stage_provenance"] = provenance_column
    working_mapping["stage_data_cost_source"] = source_column
    return pd.DataFrame(collapsed_rows).reset_index(drop=True), working_mapping


def _derive_profitability_data_cost(
    row: pd.Series,
    *,
    mapping: dict[str, str],
    analysis_kind: str,
    row_number: int,
) -> tuple[float, dict[str, Any]]:
    amount_unit = _optional_text(row, mapping, "amount_unit")
    if amount_unit is None:
        raise RiskAnalysisError(
            f"第 {row_number} 行推导 data_cost_rate 时 amount_unit 不能为空。",
            analysis_kind=analysis_kind,
            field="amount_unit",
            row_number=row_number,
        )
    per_application_cost = _number(
        row,
        mapping,
        "per_application_cost",
        analysis_kind,
        row_number,
        required=True,
        minimum=0.0,
    )
    credit_approval_rate = _number(
        row,
        mapping,
        "credit_approval_rate",
        analysis_kind,
        row_number,
        required=True,
        rate=True,
        minimum=0.0,
        minimum_inclusive=False,
    )
    draw_initiation_rate = _number(
        row,
        mapping,
        "draw_initiation_rate",
        analysis_kind,
        row_number,
        required=True,
        rate=True,
        minimum=0.0,
        minimum_inclusive=False,
    )
    draw_approval_rate = _number(
        row,
        mapping,
        "draw_approval_rate",
        analysis_kind,
        row_number,
        required=True,
        rate=True,
        minimum=0.0,
        minimum_inclusive=False,
    )
    average_ticket = _number(
        row,
        mapping,
        "average_ticket",
        analysis_kind,
        row_number,
        required=True,
        minimum=0.0,
        minimum_inclusive=False,
    )
    annualization_factor = _number(
        row,
        mapping,
        "data_annualization_factor",
        analysis_kind,
        row_number,
        required=True,
        minimum=0.0,
        minimum_inclusive=False,
    )
    assert (
        per_application_cost is not None
        and credit_approval_rate is not None
        and draw_initiation_rate is not None
        and draw_approval_rate is not None
        and average_ticket is not None
        and annualization_factor is not None
    )
    funnel_denominator = (
        credit_approval_rate
        * draw_initiation_rate
        * draw_approval_rate
        * average_ticket
    )
    rate = _validated_derived_rate(
        per_application_cost / funnel_denominator * annualization_factor,
        field="data_cost_rate",
        analysis_kind=analysis_kind,
        row_number=row_number,
    )
    return rate, {
        "amount_unit": amount_unit,
        "per_application_cost": per_application_cost,
        "credit_approval_rate": credit_approval_rate,
        "draw_initiation_rate": draw_initiation_rate,
        "draw_approval_rate": draw_approval_rate,
        "average_ticket": average_ticket,
        "data_annualization_factor": annualization_factor,
    }


def _has_any_mapped_value(
    row: pd.Series,
    mapping: dict[str, str],
    fields: Iterable[str],
) -> bool:
    return any(
        field in mapping
        and not _is_missing(row[mapping[field]])
        and bool(str(row[mapping[field]]).strip())
        for field in fields
    )


def _reconcile_profit_rate(
    explicit: float,
    implied: float,
    *,
    field: str,
    analysis_kind: str,
    row_number: int,
) -> None:
    if math.isclose(
        explicit,
        implied,
        rel_tol=0.0,
        abs_tol=PROFIT_RATE_RECONCILIATION_TOLERANCE,
    ):
        return
    raise RiskAnalysisError(
        f"第 {row_number} 行 {field}={explicit} 与驱动项推导值={implied} "
        f"不一致；绝对容差为 {PROFIT_RATE_RECONCILIATION_TOLERANCE:g}。",
        analysis_kind=analysis_kind,
        field=field,
        row_number=row_number,
    )


def _resolve_profitability_costs(
    row: pd.Series,
    *,
    mapping: dict[str, str],
    analysis_kind: str,
    row_number: int,
    customer_rate: float,
) -> tuple[dict[str, float], dict[str, str], dict[str, Any]]:
    costs: dict[str, float] = {}
    sources: dict[str, str] = {}
    drivers: dict[str, Any] = {field: None for field in _PROFIT_DRIVER_FIELDS}

    for field in _PROFIT_ALWAYS_EXPLICIT_COST_FIELDS:
        value = _number(
            row,
            mapping,
            field,
            analysis_kind,
            row_number,
            required=True,
            rate=True,
        )
        assert value is not None
        costs[field] = value

    risk_cost = _number(
        row, mapping, "risk_cost_rate", analysis_kind, row_number, rate=True
    )
    terminal_vintage = _number(
        row, mapping, "terminal_vintage_rate", analysis_kind, row_number, rate=True
    )
    risk_turnover = _number(
        row,
        mapping,
        "risk_turnover",
        analysis_kind,
        row_number,
        minimum=0.0,
        minimum_inclusive=False,
    )
    if risk_cost is None:
        if terminal_vintage is None or risk_turnover is None:
            raise RiskAnalysisError(
                f"第 {row_number} 行 risk_cost_rate 缺失时，"
                "terminal_vintage_rate 与 risk_turnover 必须同时提供。",
                analysis_kind=analysis_kind,
                field="risk_cost_rate",
                row_number=row_number,
            )
        drivers["terminal_vintage_rate"] = terminal_vintage
        drivers["risk_turnover"] = risk_turnover
        risk_cost = _validated_derived_rate(
            terminal_vintage * risk_turnover,
            field="risk_cost_rate",
            analysis_kind=analysis_kind,
            row_number=row_number,
        )
        sources["risk_cost_rate"] = "derived_terminal_vintage_turnover"
    else:
        if terminal_vintage is not None or risk_turnover is not None:
            if terminal_vintage is None or risk_turnover is None:
                raise RiskAnalysisError(
                    f"第 {row_number} 行同时保留显式 risk_cost_rate 与推导驱动项时，"
                    "terminal_vintage_rate 与 risk_turnover 必须同时提供。",
                    analysis_kind=analysis_kind,
                    field="risk_cost_rate",
                    row_number=row_number,
                )
            implied_risk_cost = _validated_derived_rate(
                terminal_vintage * risk_turnover,
                field="risk_cost_rate",
                analysis_kind=analysis_kind,
                row_number=row_number,
            )
            _reconcile_profit_rate(
                risk_cost,
                implied_risk_cost,
                field="risk_cost_rate",
                analysis_kind=analysis_kind,
                row_number=row_number,
            )
            drivers["terminal_vintage_rate"] = terminal_vintage
            drivers["risk_turnover"] = risk_turnover
            sources["risk_cost_rate"] = "explicit_reconciled_terminal_vintage_turnover"
        else:
            sources["risk_cost_rate"] = "explicit"
    costs["risk_cost_rate"] = risk_cost

    interest_loss = _number(
        row, mapping, "interest_loss_rate", analysis_kind, row_number, rate=True
    )
    loss_timing_factor = _number(
        row, mapping, "loss_timing_factor", analysis_kind, row_number, rate=True
    )
    if interest_loss is None:
        if loss_timing_factor is None:
            raise RiskAnalysisError(
                f"第 {row_number} 行 interest_loss_rate 缺失时，"
                "loss_timing_factor 必须提供。",
                analysis_kind=analysis_kind,
                field="interest_loss_rate",
                row_number=row_number,
            )
        drivers["loss_timing_factor"] = loss_timing_factor
        interest_loss = _validated_derived_rate(
            customer_rate * risk_cost * loss_timing_factor,
            field="interest_loss_rate",
            analysis_kind=analysis_kind,
            row_number=row_number,
        )
        sources["interest_loss_rate"] = "derived_customer_risk_timing"
    else:
        if loss_timing_factor is not None:
            implied_interest_loss = _validated_derived_rate(
                customer_rate * risk_cost * loss_timing_factor,
                field="interest_loss_rate",
                analysis_kind=analysis_kind,
                row_number=row_number,
            )
            _reconcile_profit_rate(
                interest_loss,
                implied_interest_loss,
                field="interest_loss_rate",
                analysis_kind=analysis_kind,
                row_number=row_number,
            )
            drivers["loss_timing_factor"] = loss_timing_factor
            sources["interest_loss_rate"] = "explicit_reconciled_customer_risk_timing"
        else:
            sources["interest_loss_rate"] = "explicit"
    costs["interest_loss_rate"] = interest_loss

    revenue_share = _number(
        row, mapping, "revenue_share_rate", analysis_kind, row_number, rate=True
    )
    profit_share_ratio = _number(
        row, mapping, "profit_share_ratio", analysis_kind, row_number, rate=True
    )
    if revenue_share is None:
        if profit_share_ratio is None:
            raise RiskAnalysisError(
                f"第 {row_number} 行 revenue_share_rate 缺失时，"
                "profit_share_ratio 必须提供。",
                analysis_kind=analysis_kind,
                field="revenue_share_rate",
                row_number=row_number,
            )
        drivers["profit_share_ratio"] = profit_share_ratio
        revenue_share = _validated_derived_rate(
            (customer_rate - interest_loss) * profit_share_ratio,
            field="revenue_share_rate",
            analysis_kind=analysis_kind,
            row_number=row_number,
        )
        sources["revenue_share_rate"] = "derived_net_interest_profit_share"
    else:
        if profit_share_ratio is not None:
            implied_revenue_share = _validated_derived_rate(
                (customer_rate - interest_loss) * profit_share_ratio,
                field="revenue_share_rate",
                analysis_kind=analysis_kind,
                row_number=row_number,
            )
            _reconcile_profit_rate(
                revenue_share,
                implied_revenue_share,
                field="revenue_share_rate",
                analysis_kind=analysis_kind,
                row_number=row_number,
            )
            drivers["profit_share_ratio"] = profit_share_ratio
            sources["revenue_share_rate"] = (
                "explicit_reconciled_net_interest_profit_share"
            )
        else:
            sources["revenue_share_rate"] = "explicit"
    costs["revenue_share_rate"] = revenue_share

    data_cost = _number(
        row, mapping, "data_cost_rate", analysis_kind, row_number, rate=True
    )
    if data_cost is None:
        data_cost, data_drivers = _derive_profitability_data_cost(
            row,
            mapping=mapping,
            analysis_kind=analysis_kind,
            row_number=row_number,
        )
        drivers.update(data_drivers)
        sources["data_cost_rate"] = "derived_application_funnel"
    else:
        data_driver_fields = (
            "per_application_cost",
            "credit_approval_rate",
            "draw_initiation_rate",
            "draw_approval_rate",
            "average_ticket",
            "data_annualization_factor",
        )
        if "stage_data_cost_source" not in mapping and _has_any_mapped_value(
            row, mapping, data_driver_fields
        ):
            implied_data_cost, data_drivers = _derive_profitability_data_cost(
                row,
                mapping=mapping,
                analysis_kind=analysis_kind,
                row_number=row_number,
            )
            _reconcile_profit_rate(
                data_cost,
                implied_data_cost,
                field="data_cost_rate",
                analysis_kind=analysis_kind,
                row_number=row_number,
            )
            drivers.update(data_drivers)
            sources["data_cost_rate"] = "explicit_reconciled_application_funnel"
        else:
            sources["data_cost_rate"] = "explicit"
    costs["data_cost_rate"] = data_cost

    tax_rate = _number(row, mapping, "tax_rate", analysis_kind, row_number, rate=True)
    if tax_rate is None:
        tax_method = _optional_text(row, mapping, "tax_method")
        if tax_method != "sample_net_revenue_vat_surcharge":
            raise RiskAnalysisError(
                f"第 {row_number} 行 tax_rate 缺失时，tax_method 必须为 "
                "sample_net_revenue_vat_surcharge。",
                analysis_kind=analysis_kind,
                field="tax_method",
                row_number=row_number,
            )
        tax_inclusive_divisor = _number(
            row,
            mapping,
            "tax_inclusive_divisor",
            analysis_kind,
            row_number,
            required=True,
            minimum=0.0,
            minimum_inclusive=False,
        )
        tax_combined_rate = _number(
            row,
            mapping,
            "tax_combined_rate",
            analysis_kind,
            row_number,
            required=True,
            rate=True,
        )
        assert tax_inclusive_divisor is not None and tax_combined_rate is not None
        drivers.update(
            {
                "tax_method": tax_method,
                "tax_inclusive_divisor": tax_inclusive_divisor,
                "tax_combined_rate": tax_combined_rate,
            }
        )
        tax_base = (
            customer_rate
            - interest_loss
            - revenue_share
            - costs["acquisition_cost_rate"]
            - data_cost
            - costs["payment_cost_rate"]
            - costs["collection_cost_rate"]
        )
        if tax_base < 0.0:
            raise RiskAnalysisError(
                f"第 {row_number} 行 sample_net_revenue_vat_surcharge "
                f"推导税基={tax_base}，"
                "税基不能为负；请显式提供 tax_rate 或核查成本口径。",
                analysis_kind=analysis_kind,
                field="tax_rate",
                row_number=row_number,
            )
        tax_rate = _validated_derived_rate(
            tax_base / tax_inclusive_divisor * tax_combined_rate,
            field="tax_rate",
            analysis_kind=analysis_kind,
            row_number=row_number,
        )
        sources["tax_rate"] = "derived_sample_net_revenue_vat_surcharge"
    else:
        tax_driver_fields = (
            "tax_method",
            "tax_inclusive_divisor",
            "tax_combined_rate",
        )
        if _has_any_mapped_value(row, mapping, tax_driver_fields):
            tax_method = _optional_text(row, mapping, "tax_method")
            if tax_method != "sample_net_revenue_vat_surcharge":
                raise RiskAnalysisError(
                    f"第 {row_number} 行同时保留显式 tax_rate 与税费驱动项时，"
                    "tax_method 必须为 sample_net_revenue_vat_surcharge。",
                    analysis_kind=analysis_kind,
                    field="tax_method",
                    row_number=row_number,
                )
            tax_inclusive_divisor = _number(
                row,
                mapping,
                "tax_inclusive_divisor",
                analysis_kind,
                row_number,
                required=True,
                minimum=0.0,
                minimum_inclusive=False,
            )
            tax_combined_rate = _number(
                row,
                mapping,
                "tax_combined_rate",
                analysis_kind,
                row_number,
                required=True,
                rate=True,
            )
            assert tax_inclusive_divisor is not None and tax_combined_rate is not None
            tax_base = (
                customer_rate
                - interest_loss
                - revenue_share
                - costs["acquisition_cost_rate"]
                - data_cost
                - costs["payment_cost_rate"]
                - costs["collection_cost_rate"]
            )
            if tax_base < 0.0:
                raise RiskAnalysisError(
                    f"第 {row_number} 行 sample_net_revenue_vat_surcharge "
                    f"推导税基={tax_base}，税基不能为负。",
                    analysis_kind=analysis_kind,
                    field="tax_rate",
                    row_number=row_number,
                )
            implied_tax_rate = _validated_derived_rate(
                tax_base / tax_inclusive_divisor * tax_combined_rate,
                field="tax_rate",
                analysis_kind=analysis_kind,
                row_number=row_number,
            )
            _reconcile_profit_rate(
                tax_rate,
                implied_tax_rate,
                field="tax_rate",
                analysis_kind=analysis_kind,
                row_number=row_number,
            )
            drivers.update(
                {
                    "tax_method": tax_method,
                    "tax_inclusive_divisor": tax_inclusive_divisor,
                    "tax_combined_rate": tax_combined_rate,
                }
            )
            sources["tax_rate"] = "explicit_reconciled_sample_net_revenue_vat_surcharge"
        else:
            sources["tax_rate"] = "explicit"
    costs["tax_rate"] = tax_rate
    return costs, sources, drivers


def _collapse_vtg_balance_curves(
    data: pd.DataFrame,
    *,
    mapping: dict[str, str],
    analysis_kind: str,
) -> tuple[pd.DataFrame, dict[str, str], int]:
    """Collapse normalized long-form MOB balance curves to one row per cohort."""

    groups: dict[tuple[str, str, str, str, str, float | None, str], list[int]] = {}
    for position, row in data.iterrows():
        row_number = position + 1
        product = _required_text(row, mapping, "product", analysis_kind, row_number)
        cohort = _required_text(row, mapping, "cohort", analysis_kind, row_number)
        as_of_date = _required_text(
            row, mapping, "as_of_date", analysis_kind, row_number
        )
        scenario = _optional_text(row, mapping, "scenario") or "基准"
        channel = _optional_text(row, mapping, "channel") or "未提供"
        tenor_months = _number(
            row,
            mapping,
            "tenor_months",
            analysis_kind,
            row_number,
            minimum=0.0,
            minimum_inclusive=False,
        )
        selection_rule = _optional_text(row, mapping, "selection_rule") or "未提供"
        groups.setdefault(
            (
                product,
                cohort,
                as_of_date,
                scenario,
                channel,
                tenor_months,
                selection_rule,
            ),
            [],
        ).append(position)

    curve_fields = {
        "mob",
        "mob_days",
        "mob_balance_rate",
        "mob_balance_amount",
    }
    group_level_fields = (
        *_VTG_REQUIRED,
        *_VTG_OPTIONAL_RATES,
        *_VTG_OPTIONAL_NONNEGATIVE,
        *_VTG_OPTIONAL_GROUP_TEXT,
        "previous_turnover",
        "terminal_method",
        "tenor_months",
        "day_count_basis",
    )
    turnover_column = _unused_column_name(data, "__derived_turnover")
    avg_balance_column = _unused_column_name(data, "__derived_avg_daily_balance")
    balance_source_column = _unused_column_name(data, "__mob_balance_source")
    collapsed_rows: list[pd.Series] = []
    zero_disbursement_skipped_count = 0

    use_balance_amount = "mob_balance_amount" in mapping
    for (
        product,
        cohort,
        as_of_date,
        scenario,
        channel,
        tenor_months,
        selection_rule,
    ), positions in groups.items():
        identity = (
            f"产品 {product} / cohort {cohort} / 截至 {as_of_date} / 场景 {scenario} / "
            f"渠道 {channel} / 期限 {tenor_months or '未提供'} / 筛选 {selection_rule}"
        )
        group = data.loc[positions]
        for field in group_level_fields:
            if field in curve_fields or field not in mapping:
                continue
            source = mapping[field]
            tokens = {_consistency_token(value) for value in group[source].tolist()}
            if len(tokens) != 1:
                raise RiskAnalysisError(
                    f"{identity} 的 {field} 在 MOB 曲线行之间必须一致。",
                    analysis_kind=analysis_kind,
                    field=field,
                    row_number=positions[0] + 1,
                )

        first_position = positions[0]
        first_row = data.loc[first_position]
        disbursement = _number(
            first_row,
            mapping,
            "disbursement_amount",
            analysis_kind,
            first_position + 1,
            required=True,
            minimum=0.0,
        )
        day_count_basis = _number(
            first_row,
            mapping,
            "day_count_basis",
            analysis_kind,
            first_position + 1,
            required=True,
            minimum=0.0,
            minimum_inclusive=False,
        )
        assert disbursement is not None and day_count_basis is not None
        if disbursement == 0.0:
            zero_disbursement_skipped_count += len(positions)
            continue
        seen_mobs: set[tuple[str, Any]] = set()
        weighted_balance_days = 0.0
        total_mob_days = 0.0
        for position in positions:
            row = data.loc[position]
            row_number = position + 1
            mob_source = mapping["mob"]
            raw_mob = row[mob_source]
            if _is_missing(raw_mob) or not str(raw_mob).strip():
                raise RiskAnalysisError(
                    f"第 {row_number} 行 mob 不能为空。",
                    analysis_kind=analysis_kind,
                    field="mob",
                    row_number=row_number,
                )
            mob_token = _consistency_token(raw_mob)
            if mob_token in seen_mobs:
                raise RiskAnalysisError(
                    f"{identity} 的 mob 必须唯一，发现重复值 {raw_mob!r}。",
                    analysis_kind=analysis_kind,
                    field="mob",
                    row_number=row_number,
                )
            seen_mobs.add(mob_token)
            mob_days = _number(
                row,
                mapping,
                "mob_days",
                analysis_kind,
                row_number,
                required=True,
                minimum=0.0,
                minimum_inclusive=False,
            )
            total_mob_days += mob_days
            if use_balance_amount:
                balance_amount = _number(
                    row,
                    mapping,
                    "mob_balance_amount",
                    analysis_kind,
                    row_number,
                    required=True,
                    minimum=0.0,
                )
                assert balance_amount is not None
                if "mob_balance_rate" in mapping:
                    balance_rate = _number(
                        row,
                        mapping,
                        "mob_balance_rate",
                        analysis_kind,
                        row_number,
                        required=True,
                        rate=True,
                    )
                    assert balance_rate is not None
                    implied_balance_amount = disbursement * balance_rate
                    if not math.isclose(
                        balance_amount,
                        implied_balance_amount,
                        rel_tol=_VTG_BALANCE_RECONCILIATION_REL_TOLERANCE,
                        abs_tol=_VTG_BALANCE_RECONCILIATION_ABS_TOLERANCE,
                    ):
                        raise RiskAnalysisError(
                            f"第 {row_number} 行 mob_balance_amount={balance_amount} "
                            "与 disbursement_amount × mob_balance_rate="
                            f"{implied_balance_amount} 不一致。",
                            analysis_kind=analysis_kind,
                            field="mob_balance_amount",
                            row_number=row_number,
                        )
            else:
                balance_rate = _number(
                    row,
                    mapping,
                    "mob_balance_rate",
                    analysis_kind,
                    row_number,
                    required=True,
                    rate=True,
                )
                assert balance_rate is not None
                balance_amount = disbursement * balance_rate
            weighted_balance_days += balance_amount * mob_days

        if abs(total_mob_days - day_count_basis) > _VTG_CURVE_DAY_TOLERANCE:
            raise RiskAnalysisError(
                f"{identity} 的 mob_days 合计为 "
                f"{total_mob_days:.8f}，必须等于 day_count_basis={day_count_basis:g}。",
                analysis_kind=analysis_kind,
                field="mob_days",
                row_number=first_position + 1,
            )

        avg_daily_balance = weighted_balance_days / day_count_basis
        if avg_daily_balance <= 0.0:
            raise RiskAnalysisError(
                f"{identity} 的 MOB 余额曲线日均余额必须大于 0。",
                analysis_kind=analysis_kind,
                field=(
                    "mob_balance_amount" if use_balance_amount else "mob_balance_rate"
                ),
                row_number=first_position + 1,
            )
        supplied_avg_daily_balance = _number(
            first_row,
            mapping,
            "avg_daily_balance",
            analysis_kind,
            first_position + 1,
            minimum=0.0,
        )
        if supplied_avg_daily_balance is not None and not math.isclose(
            supplied_avg_daily_balance,
            avg_daily_balance,
            rel_tol=_VTG_BALANCE_RECONCILIATION_REL_TOLERANCE,
            abs_tol=_VTG_BALANCE_RECONCILIATION_ABS_TOLERANCE,
        ):
            raise RiskAnalysisError(
                f"{identity} 显式 avg_daily_balance={supplied_avg_daily_balance} 与 "
                f"MOB 曲线推导值={avg_daily_balance} 不一致。",
                analysis_kind=analysis_kind,
                field="avg_daily_balance",
                row_number=first_position + 1,
            )
        collapsed = first_row.copy()
        collapsed[turnover_column] = disbursement / avg_daily_balance
        collapsed[avg_balance_column] = avg_daily_balance
        collapsed[balance_source_column] = (
            "mob_balance_curve_reconciled_with_explicit_avg"
            if supplied_avg_daily_balance is not None
            else "mob_balance_amount_reconciled_with_rate"
            if use_balance_amount and "mob_balance_rate" in mapping
            else ("mob_balance_amount" if use_balance_amount else "mob_balance_rate")
        )
        collapsed_rows.append(collapsed)

    working_mapping = dict(mapping)
    working_mapping["turnover"] = turnover_column
    working_mapping["avg_daily_balance"] = avg_balance_column
    working_mapping["mob_balance_source"] = balance_source_column
    return (
        pd.DataFrame(collapsed_rows).reset_index(drop=True),
        working_mapping,
        zero_disbursement_skipped_count,
    )


def _vtg_product_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    slice_keys = list(
        dict.fromkeys(
            (
                row["product"],
                row["as_of_date"],
                row.get("scenario") or "基准",
                row.get("channel") or "未提供",
                row.get("tenor_months"),
                row.get("selection_rule") or "未提供",
            )
            for row in rows
        )
    )
    for (
        product,
        as_of_date,
        scenario,
        channel,
        tenor_months,
        selection_rule,
    ) in slice_keys:
        product_rows = [
            row
            for row in rows
            if (
                row["product"],
                row["as_of_date"],
                row.get("scenario") or "基准",
                row.get("channel") or "未提供",
                row.get("tenor_months"),
                row.get("selection_rule") or "未提供",
            )
            == (
                product,
                as_of_date,
                scenario,
                channel,
                tenor_months,
                selection_rule,
            )
        ]
        disbursement = sum(row["disbursement_amount"] for row in product_rows)
        avg_balance = sum(row["avg_daily_balance"] for row in product_rows)
        terminal_loss = sum(
            row["terminal_bad_rate"] * row["disbursement_amount"]
            for row in product_rows
        )
        observed_loss = sum(
            row["mob14_bad_rate"] * row["disbursement_amount"] for row in product_rows
        )
        result.append(
            {
                "product": product,
                "as_of_date": as_of_date,
                "scenario": scenario,
                "channel": channel,
                "tenor_months": tenor_months,
                "selection_rule": selection_rule,
                "amount_unit": product_rows[0]["amount_unit"],
                "row_count": len(product_rows),
                "disbursement_amount": disbursement,
                "avg_daily_balance": avg_balance,
                "turnover": _safe_ratio(disbursement, avg_balance),
                "mob14_bad_rate": _safe_ratio(observed_loss, disbursement),
                "terminal_bad_rate": _safe_ratio(terminal_loss, disbursement),
                "observed_annualized_bad_rate": _safe_ratio(observed_loss, avg_balance),
                "annualized_bad_rate": _safe_ratio(terminal_loss, avg_balance),
            }
        )
    return result


def _profit_product_summaries(
    rows: list[dict[str, Any]],
    *,
    analysis_kind: str,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    weighted_fields = (
        "customer_rate",
        *_PROFIT_COST_FIELDS,
        "fixed_income_yield",
        "total_cost_rate",
        "net_yield",
    )
    group_keys = list(
        dict.fromkeys(
            (
                row["product"],
                row.get("as_of_period") or "未提供",
                row.get("scenario") or "基准",
            )
            for row in rows
        )
    )
    for product, as_of_period, scenario in group_keys:
        product_rows = [
            row
            for row in rows
            if (
                row["product"],
                row.get("as_of_period") or "未提供",
                row.get("scenario") or "基准",
            )
            == (product, as_of_period, scenario)
        ]
        weight_sum = sum(row["weight"] for row in product_rows)
        if abs(weight_sum - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise RiskAnalysisError(
                f"产品 {product} / 期间 {as_of_period} / 场景 {scenario} 的 weight "
                f"合计为 {weight_sum:.8f}，必须约等于 1。",
                analysis_kind=analysis_kind,
                field="weight",
            )
        amount_units = {
            row["amount_unit"]
            for row in product_rows
            if row.get("amount_unit") is not None
        }
        if len(amount_units) > 1:
            raise RiskAnalysisError(
                f"产品 {product} / 期间 {as_of_period} / 场景 {scenario} 的 "
                "原始数据成本驱动 amount_unit 必须一致，实际为："
                + "、".join(sorted(amount_units))
                + "。",
                analysis_kind=analysis_kind,
                field="amount_unit",
            )
        summary: dict[str, Any] = {
            "product": product,
            "as_of_period": as_of_period,
            "scenario": scenario,
            "weight_basis": "average_balance",
            "amount_unit": next(iter(amount_units), None),
            "asset_class_count": len({row["asset_class"] for row in product_rows}),
            "weight_sum": weight_sum,
        }
        for field in weighted_fields:
            summary[field] = sum(row[field] * row["weight"] for row in product_rows)
        summaries.append(summary)
    return summaries


def _profit_slice_identity(row: dict[str, Any]) -> str:
    return (
        f"产品 {row['product']} / 期间 {row.get('as_of_period') or '未提供'} / "
        f"场景 {row.get('scenario') or '基准'}"
    )


def _normalized_frame(frame: pd.DataFrame, *, analysis_kind: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise RiskAnalysisError(
            "风险分析输入必须是表格数据。", analysis_kind=analysis_kind
        )
    if frame.empty:
        raise RiskAnalysisError("风险分析数据为空。", analysis_kind=analysis_kind)
    return frame.reset_index(drop=True)


def _normalize_column_map(
    frame: pd.DataFrame,
    column_map: dict[str, str],
    *,
    analysis_kind: str,
) -> dict[str, str]:
    if not isinstance(column_map, dict) or not column_map:
        raise RiskAnalysisError(
            "column_map 必须是非空对象。", analysis_kind=analysis_kind
        )
    normalized: dict[str, str] = {}
    for raw_canonical, raw_source in column_map.items():
        canonical = str(raw_canonical or "").strip()
        source = str(raw_source or "").strip()
        if not canonical or not source:
            raise RiskAnalysisError(
                "column_map 的 canonical 和 source 都不能为空。",
                analysis_kind=analysis_kind,
            )
        if source not in frame.columns:
            raise RiskAnalysisError(
                f"column_map 指向不存在的列: {canonical} -> {source}",
                analysis_kind=analysis_kind,
                field=canonical,
            )
        if list(frame.columns).count(source) != 1:
            raise RiskAnalysisError(
                f"源数据列名重复，无法安全映射: {source}",
                analysis_kind=analysis_kind,
                field=canonical,
            )
        normalized[canonical] = source
    return normalized


def _require_mappings(
    mapping: dict[str, str],
    required: Iterable[str],
    *,
    analysis_kind: str,
) -> None:
    missing = [field for field in required if field not in mapping]
    if missing:
        raise RiskAnalysisError(
            "column_map 缺少必需字段: " + ", ".join(missing),
            analysis_kind=analysis_kind,
            field=missing[0],
        )


def _required_text(
    row: pd.Series,
    mapping: dict[str, str],
    field: str,
    analysis_kind: str,
    row_number: int,
) -> str:
    source = mapping[field]
    value = row[source]
    if _is_missing(value) or not str(value).strip():
        raise RiskAnalysisError(
            f"第 {row_number} 行 {field} 不能为空。",
            analysis_kind=analysis_kind,
            field=field,
            row_number=row_number,
        )
    return str(value).strip()


def _optional_text(
    row: pd.Series,
    mapping: dict[str, str],
    field: str,
) -> str | None:
    source = mapping.get(field)
    if source is None:
        return None
    value = row[source]
    if _is_missing(value) or not str(value).strip():
        return None
    return str(value).strip()


def _number(
    row: pd.Series,
    mapping: dict[str, str],
    field: str,
    analysis_kind: str,
    row_number: int,
    *,
    required: bool = False,
    rate: bool = False,
    minimum: float | None = None,
    minimum_inclusive: bool = True,
) -> float | None:
    source = mapping.get(field)
    if source is None:
        if required:
            raise RiskAnalysisError(
                f"column_map 缺少必需字段: {field}",
                analysis_kind=analysis_kind,
                field=field,
                row_number=row_number,
            )
        return None
    value = row[source]
    if _is_missing(value):
        if required:
            raise RiskAnalysisError(
                f"第 {row_number} 行 {field} 不能为空。",
                analysis_kind=analysis_kind,
                field=field,
                row_number=row_number,
            )
        return None
    if isinstance(value, bool):
        raise RiskAnalysisError(
            f"第 {row_number} 行 {field} 必须是数值，不能是布尔值。",
            analysis_kind=analysis_kind,
            field=field,
            row_number=row_number,
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RiskAnalysisError(
            f"第 {row_number} 行 {field} 不是有效数值: {value!r}",
            analysis_kind=analysis_kind,
            field=field,
            row_number=row_number,
        ) from exc
    if not math.isfinite(number):
        raise RiskAnalysisError(
            f"第 {row_number} 行 {field} 必须是有限数值。",
            analysis_kind=analysis_kind,
            field=field,
            row_number=row_number,
        )
    if rate and not 0.0 <= number <= 1.0:
        raise RiskAnalysisError(
            f"第 {row_number} 行 {field}={number}，率必须在 [0, 1]。",
            analysis_kind=analysis_kind,
            field=field,
            row_number=row_number,
        )
    if minimum is not None:
        invalid = number < minimum if minimum_inclusive else number <= minimum
        if invalid:
            comparator = ">=" if minimum_inclusive else ">"
            raise RiskAnalysisError(
                f"第 {row_number} 行 {field}={number}，必须 {comparator} {minimum}。",
                analysis_kind=analysis_kind,
                field=field,
                row_number=row_number,
            )
    return number


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _consistency_token(value: Any) -> tuple[str, Any]:
    if _is_missing(value):
        return ("missing", None)
    if isinstance(value, bool):
        return ("bool", value)
    if not isinstance(value, str):
        try:
            number = float(value)
        except (TypeError, ValueError):
            pass
        else:
            if math.isfinite(number):
                return ("number", number)
    return ("text", str(value).strip())


def _unused_column_name(frame: pd.DataFrame, base: str) -> str:
    name = base
    suffix = 1
    while name in frame.columns:
        name = f"{base}_{suffix}"
        suffix += 1
    return name


def _validated_derived_rate(
    value: float,
    *,
    field: str,
    analysis_kind: str,
    row_number: int,
) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise RiskAnalysisError(
            f"第 {row_number} 行推导得到 {field}={number}，率必须在 [0, 1]。",
            analysis_kind=analysis_kind,
            field=field,
            row_number=row_number,
        )
    return number


def _format_value_set(values: Iterable[Any], *, percent: bool = False) -> str:
    unique: list[Any] = []
    seen: set[tuple[str, Any]] = set()
    for value in values:
        token = _consistency_token(value)
        if token in seen:
            continue
        seen.add(token)
        unique.append(value)
    rendered = [
        _format_percent(float(value)) if percent else f"{float(value):g}"
        for value in unique[:8]
    ]
    if len(unique) > 8:
        rendered.append(f"另 {len(unique) - 8} 个值")
    return "、".join(rendered) if rendered else "未提供"


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0.0 else None


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def _period_scope(values: Iterable[str]) -> str:
    periods = sorted(
        {
            str(value).strip()
            for value in values
            if value is not None and str(value).strip()
        }
    )
    if not periods:
        return "未提供"
    if len(periods) == 1:
        return _bounded_text(periods[0], AS_OF_PERIOD_MAX_CHARS)
    return _bounded_text(f"{periods[0]} ~ {periods[-1]}", AS_OF_PERIOD_MAX_CHARS)


def _product_scope(values: Iterable[str]) -> list[str]:
    return [
        _bounded_text(value, PRODUCT_SCOPE_ITEM_MAX_CHARS)
        for value in _ordered_unique(values)[:PRODUCT_SCOPE_LIMIT]
    ]


def _bounded_text(value: str, max_chars: int) -> str:
    text = str(value).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _unique_strings(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value).strip()))


def _format_percent(value: float) -> str:
    return f"{float(value):.2%}"


def _format_change(value: float) -> str:
    direction = "上升" if value >= 0.0 else "下降"
    return f"{direction} {abs(float(value)) * 100:.2f} 个百分点"


__all__ = [
    "ANALYSIS_KINDS",
    "PRODUCT_SCOPE_LIMIT",
    "WEIGHT_SUM_TOLERANCE",
    "RiskAnalysisCalculation",
    "RiskAnalysisError",
    "calculate_profitability",
    "calculate_risk_analysis",
    "calculate_vtg_terminal",
]
