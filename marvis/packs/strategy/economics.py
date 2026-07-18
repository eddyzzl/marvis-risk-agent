from __future__ import annotations

import math
from typing import Any, TypeAlias

import pandas as pandas

from marvis.packs.strategy.errors import StrategyError


NumericInput: TypeAlias = pandas.Series | int | float


def limit_metrics(
    assigned_limit: pandas.Series,
    target: pandas.Series,
    baseline: pandas.Series | None = None,
    *,
    pd: NumericInput | None = None,
    lgd: NumericInput | None = None,
    utilization: NumericInput | None = None,
) -> dict[str, Any]:
    """Summarize an assigned-limit strategy with an optional EL chain.

    ``target`` may contain missing labels.  Those rows remain in population counts
    and shares but are excluded from the bad-rate denominator.  The economic chain
    is deliberately all-or-nothing: PD, LGD and utilization must either all be
    supplied or all be omitted.
    """

    limits = _required_series(
        assigned_limit,
        name="assigned_limit",
        lower=0.0,
    )
    labels = _target_series(target, limits.index)
    baseline_limits = (
        None
        if baseline is None
        else _required_series(
            baseline,
            name="baseline",
            index=limits.index,
            lower=0.0,
        )
    )
    economic_inputs = _complete_bundle(
        "limit economics",
        {"pd": pd, "lgd": lgd, "utilization": utilization},
    )

    result: dict[str, Any] = {
        "count": int(len(limits)),
        "total_limit": _finite_sum(limits.tolist(), name="total_limit"),
        "mean_limit": _mean_or_none(limits),
        "min_limit": _min_or_none(limits),
        "max_limit": _max_or_none(limits),
        "by_limit": _risk_layers(limits, labels, value_key="assigned_limit"),
        "baseline": _limit_baseline(limits, baseline_limits),
        "economics": None,
    }
    if economic_inputs is not None:
        pd_values = _numeric_input(
            economic_inputs["pd"],
            name="pd",
            index=limits.index,
            lower=0.0,
            upper=1.0,
        )
        lgd_values = _numeric_input(
            economic_inputs["lgd"],
            name="lgd",
            index=limits.index,
            lower=0.0,
            upper=1.0,
        )
        utilization_values = _numeric_input(
            economic_inputs["utilization"],
            name="utilization",
            index=limits.index,
            lower=0.0,
            upper=1.0,
        )
        ead_rows = _products(
            limits,
            utilization_values,
            name="expected_ead",
        )
        loss_rows = _products(
            limits,
            utilization_values,
            pd_values,
            lgd_values,
            name="expected_loss",
        )
        result["economics"] = {
            "expected_ead": _finite_sum(ead_rows, name="expected_ead"),
            "expected_loss": _finite_sum(loss_rows, name="expected_loss"),
        }
    return result


def pricing_metrics(
    assigned_rate: pandas.Series,
    target: pandas.Series,
    baseline: pandas.Series | None = None,
    *,
    ead: NumericInput | None = None,
    pd: NumericInput | None = None,
    lgd: NumericInput | None = None,
    funding_rate: NumericInput | None = None,
    term_months: NumericInput | None = None,
    operating_cost_per_loan: NumericInput | None = None,
) -> dict[str, Any]:
    """Summarize risk-based prices and, when complete, row-level economics.

    Rates are annual decimal rates in ``[0, 1]``.  Economics are calculated per
    row before aggregation so heterogeneous exposure, risk, funding and term inputs
    cannot be accidentally replaced by averages.  Undefined ROA is ``None`` rather
    than a fabricated zero.
    """

    rates = _required_series(
        assigned_rate,
        name="assigned_rate",
        lower=0.0,
        upper=1.0,
    )
    labels = _target_series(target, rates.index)
    baseline_rates = (
        None
        if baseline is None
        else _required_series(
            baseline,
            name="baseline",
            index=rates.index,
            lower=0.0,
            upper=1.0,
        )
    )
    economic_inputs = _complete_bundle(
        "pricing economics",
        {
            "ead": ead,
            "pd": pd,
            "lgd": lgd,
            "funding_rate": funding_rate,
            "term_months": term_months,
            "operating_cost_per_loan": operating_cost_per_loan,
        },
    )

    result: dict[str, Any] = {
        "count": int(len(rates)),
        "mean_rate": _mean_or_none(rates),
        "risk_tiers": _risk_layers(rates, labels, value_key="assigned_rate"),
        "baseline": _pricing_baseline(rates, baseline_rates),
        "economics": None,
    }
    if economic_inputs is not None:
        ead_values = _numeric_input(
            economic_inputs["ead"],
            name="ead",
            index=rates.index,
            lower=0.0,
        )
        pd_values = _numeric_input(
            economic_inputs["pd"],
            name="pd",
            index=rates.index,
            lower=0.0,
            upper=1.0,
        )
        lgd_values = _numeric_input(
            economic_inputs["lgd"],
            name="lgd",
            index=rates.index,
            lower=0.0,
            upper=1.0,
        )
        funding_values = _numeric_input(
            economic_inputs["funding_rate"],
            name="funding_rate",
            index=rates.index,
            lower=0.0,
            upper=1.0,
        )
        term_values = _numeric_input(
            economic_inputs["term_months"],
            name="term_months",
            index=rates.index,
            lower=0.0,
            lower_inclusive=False,
        )
        operating_values = _numeric_input(
            economic_inputs["operating_cost_per_loan"],
            name="operating_cost_per_loan",
            index=rates.index,
            lower=0.0,
        )
        result["economics"] = _pricing_economics(
            rates=rates,
            baseline_rates=baseline_rates,
            ead=ead_values,
            pd_values=pd_values,
            lgd=lgd_values,
            funding_rate=funding_values,
            term_months=term_values,
            operating_cost=operating_values,
        )
    return result


def _pricing_economics(
    *,
    rates: pandas.Series,
    baseline_rates: pandas.Series | None,
    ead: pandas.Series,
    pd_values: pandas.Series,
    lgd: pandas.Series,
    funding_rate: pandas.Series,
    term_months: pandas.Series,
    operating_cost: pandas.Series,
) -> dict[str, Any]:
    rows: list[dict[str, float | int | None]] = []
    revenue_values: list[float] = []
    loss_values: list[float] = []
    funding_values: list[float] = []
    operating_values: list[float] = []
    profit_values: list[float] = []
    baseline_profit_values: list[float] = []

    for position in range(len(rates)):
        exposure = float(ead.iloc[position])
        term_factor = _checked_product(
            float(term_months.iloc[position]),
            1.0 / 12.0,
            name="term_factor",
        )
        revenue = _checked_product(
            exposure,
            float(rates.iloc[position]),
            term_factor,
            name="revenue",
        )
        expected_loss = _checked_product(
            exposure,
            float(pd_values.iloc[position]),
            float(lgd.iloc[position]),
            name="expected_loss",
        )
        funding_cost = _checked_product(
            exposure,
            float(funding_rate.iloc[position]),
            term_factor,
            name="funding_cost",
        )
        op_cost = float(operating_cost.iloc[position])
        profit = _checked_sum(
            [revenue, -expected_loss, -funding_cost, -op_cost],
            name="profit",
        )
        row: dict[str, float | int | None] = {
            "position": position,
            "revenue": revenue,
            "expected_loss": expected_loss,
            "funding_cost": funding_cost,
            "operating_cost": op_cost,
            "profit": profit,
            "roa": _ratio_or_none(profit, exposure),
            "profit_delta_vs_baseline": None,
        }
        if baseline_rates is not None:
            baseline_revenue = _checked_product(
                exposure,
                float(baseline_rates.iloc[position]),
                term_factor,
                name="baseline_revenue",
            )
            baseline_profit = _checked_sum(
                [baseline_revenue, -expected_loss, -funding_cost, -op_cost],
                name="baseline_profit",
            )
            baseline_profit_values.append(baseline_profit)
            row["profit_delta_vs_baseline"] = _checked_sum(
                [profit, -baseline_profit],
                name="profit_delta_vs_baseline",
            )
        rows.append(row)
        revenue_values.append(revenue)
        loss_values.append(expected_loss)
        funding_values.append(funding_cost)
        operating_values.append(op_cost)
        profit_values.append(profit)

    total_ead = _finite_sum(ead.tolist(), name="total_ead")
    total_profit = _finite_sum(profit_values, name="profit")
    ead_weighted_rate = _ratio_or_none(
        _finite_sum(
            _products(rates, ead, name="ead_weighted_rate_numerator"),
            name="ead_weighted_rate_numerator",
        ),
        total_ead,
    )
    baseline_profit = (
        None
        if baseline_rates is None
        else _finite_sum(baseline_profit_values, name="baseline_profit")
    )
    return {
        "total_ead": total_ead,
        "ead_weighted_rate": ead_weighted_rate,
        "revenue": _finite_sum(revenue_values, name="revenue"),
        "expected_loss": _finite_sum(loss_values, name="expected_loss"),
        "funding_cost": _finite_sum(funding_values, name="funding_cost"),
        "operating_cost": _finite_sum(operating_values, name="operating_cost"),
        "profit": total_profit,
        "roa": _ratio_or_none(total_profit, total_ead),
        "baseline_profit": baseline_profit,
        "profit_delta_vs_baseline": (
            None
            if baseline_profit is None
            else _checked_sum(
                [total_profit, -baseline_profit],
                name="profit_delta_vs_baseline",
            )
        ),
        "by_row": rows,
    }


def _risk_layers(
    values: pandas.Series,
    target: pandas.Series,
    *,
    value_key: str,
) -> list[dict[str, float | int | None]]:
    total = len(values)
    layers: list[dict[str, float | int | None]] = []
    for value in sorted(set(float(item) for item in values.tolist())):
        mask = values.eq(value)
        group_target = target.loc[mask]
        labeled_count = int(group_target.notna().sum())
        bad_count = int(group_target.eq(1.0).sum())
        count = int(mask.sum())
        layers.append(
            {
                value_key: value,
                "count": count,
                "share": float(count / total),
                "labeled_count": labeled_count,
                "bad_count": bad_count,
                "bad_rate": (
                    None
                    if labeled_count == 0
                    else float(bad_count / labeled_count)
                ),
            }
        )
    return layers


def _limit_baseline(
    assigned: pandas.Series,
    baseline: pandas.Series | None,
) -> dict[str, float | int] | None:
    if baseline is None:
        return None
    return {
        "up_count": int(assigned.gt(baseline).sum()),
        "down_count": int(assigned.lt(baseline).sum()),
        "unchanged_count": int(assigned.eq(baseline).sum()),
        "total_limit_delta": _checked_sum(
            [
                _finite_sum(assigned.tolist(), name="total_limit"),
                -_finite_sum(baseline.tolist(), name="baseline_total_limit"),
            ],
            name="total_limit_delta",
        ),
    }


def _pricing_baseline(
    assigned: pandas.Series,
    baseline: pandas.Series | None,
) -> dict[str, int] | None:
    if baseline is None:
        return None
    return {
        "repriced_up_count": int(assigned.gt(baseline).sum()),
        "repriced_down_count": int(assigned.lt(baseline).sum()),
        "unchanged_count": int(assigned.eq(baseline).sum()),
    }


def _complete_bundle(
    label: str,
    values: dict[str, NumericInput | None],
) -> dict[str, NumericInput] | None:
    supplied = [name for name, value in values.items() if value is not None]
    if not supplied:
        return None
    if len(supplied) != len(values):
        missing = [name for name, value in values.items() if value is None]
        raise StrategyError(
            f"{label} requires all inputs; missing: {', '.join(missing)}"
        )
    return {name: value for name, value in values.items() if value is not None}


def _required_series(
    value: pandas.Series,
    *,
    name: str,
    index: pandas.Index | None = None,
    lower: float | None = None,
    upper: float | None = None,
    lower_inclusive: bool = True,
) -> pandas.Series:
    if not isinstance(value, pandas.Series):
        raise StrategyError(f"{name} must be a pandas Series")
    if index is not None and not value.index.equals(index):
        raise StrategyError(f"{name} index must exactly match assigned values")
    return _coerce_numeric_series(
        value,
        name=name,
        lower=lower,
        upper=upper,
        lower_inclusive=lower_inclusive,
    )


def _numeric_input(
    value: NumericInput,
    *,
    name: str,
    index: pandas.Index,
    lower: float | None = None,
    upper: float | None = None,
    lower_inclusive: bool = True,
) -> pandas.Series:
    if isinstance(value, pandas.Series):
        return _required_series(
            value,
            name=name,
            index=index,
            lower=lower,
            upper=upper,
            lower_inclusive=lower_inclusive,
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StrategyError(f"{name} must be a numeric scalar or pandas Series")
    return _coerce_numeric_series(
        pandas.Series([value] * len(index), index=index, dtype=float),
        name=name,
        lower=lower,
        upper=upper,
        lower_inclusive=lower_inclusive,
    )


def _coerce_numeric_series(
    value: pandas.Series,
    *,
    name: str,
    lower: float | None,
    upper: float | None,
    lower_inclusive: bool,
) -> pandas.Series:
    try:
        numeric = pandas.to_numeric(value, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise StrategyError(f"{name} must contain numeric values") from exc
    if numeric.isna().any() or not numeric.map(math.isfinite).all():
        raise StrategyError(f"{name} must contain finite non-missing values")
    if lower is not None:
        invalid = numeric.lt(lower) if lower_inclusive else numeric.le(lower)
        if invalid.any():
            operator = ">=" if lower_inclusive else ">"
            raise StrategyError(f"{name} values must be {operator} {lower}")
    if upper is not None and numeric.gt(upper).any():
        raise StrategyError(f"{name} values must be <= {upper}")
    return numeric


def _target_series(value: pandas.Series, index: pandas.Index) -> pandas.Series:
    if not isinstance(value, pandas.Series):
        raise StrategyError("target must be a pandas Series")
    if not value.index.equals(index):
        raise StrategyError("target index must exactly match assigned values")
    try:
        numeric = pandas.to_numeric(value, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise StrategyError("target must contain binary values or missing labels") from exc
    finite = numeric.dropna()
    if not finite.map(math.isfinite).all() or not finite.isin([0.0, 1.0]).all():
        raise StrategyError("target must contain only 0, 1, or missing labels")
    return numeric


def _products(*values: pandas.Series, name: str) -> list[float]:
    return [
        _checked_product(
            *(float(value.iloc[position]) for value in values),
            name=name,
        )
        for position in range(len(values[0]))
    ]


def _checked_product(*values: float, name: str) -> float:
    try:
        result = math.prod(values)
    except OverflowError as exc:
        raise StrategyError(f"{name} calculation overflowed") from exc
    if not math.isfinite(result):
        raise StrategyError(f"{name} calculation produced a non-finite value")
    return 0.0 if result == 0.0 else float(result)


def _checked_sum(values: list[float], *, name: str) -> float:
    try:
        result = math.fsum(values)
    except OverflowError as exc:
        raise StrategyError(f"{name} calculation overflowed") from exc
    if not math.isfinite(result):
        raise StrategyError(f"{name} calculation produced a non-finite value")
    return 0.0 if result == 0.0 else float(result)


def _finite_sum(values: list[float], *, name: str) -> float:
    return _checked_sum([float(value) for value in values], name=name)


def _mean_or_none(values: pandas.Series) -> float | None:
    if values.empty:
        return None
    return _finite_sum(values.tolist(), name="mean") / len(values)


def _min_or_none(values: pandas.Series) -> float | None:
    return None if values.empty else float(values.min())


def _max_or_none(values: pandas.Series) -> float | None:
    return None if values.empty else float(values.max())


def _ratio_or_none(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    result = numerator / denominator
    if not math.isfinite(result):
        raise StrategyError("ratio calculation produced a non-finite value")
    return 0.0 if result == 0.0 else float(result)


__all__ = ["limit_metrics", "pricing_metrics"]
