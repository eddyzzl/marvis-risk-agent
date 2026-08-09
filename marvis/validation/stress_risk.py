from __future__ import annotations

import math


STRESS_LOW_FILL = "C6EFCE"
STRESS_MEDIUM_FILL = "FFEB9C"
STRESS_HIGH_FILL = "FFC7CE"
STRESS_KS_MEDIUM_THRESHOLD = 0.10
STRESS_KS_HIGH_THRESHOLD = 0.20
STRESS_PSI_MEDIUM_THRESHOLD = 0.10
STRESS_PSI_HIGH_THRESHOLD = 0.25

_STRESS_RISK_ORDER = {"low": 0, "medium": 1, "high": 2}
_STRESS_RISK_FILLS = {
    "low": STRESS_LOW_FILL,
    "medium": STRESS_MEDIUM_FILL,
    "high": STRESS_HIGH_FILL,
}


def ks_drop_ratio(baseline: float, after: float) -> float | None:
    """Return one-sided KS decay relative to the baseline KS.

    Positive values are deterioration and negative values are improvement.
    A zero/non-finite baseline has no meaningful relative denominator.
    """
    baseline_value = float(baseline)
    after_value = float(after)
    if (
        not math.isfinite(baseline_value)
        or not math.isfinite(after_value)
        or baseline_value <= 0
    ):
        return None
    return (baseline_value - after_value) / baseline_value


def stress_ks_risk(baseline: float, after: float) -> str | None:
    ratio = ks_drop_ratio(baseline, after)
    if ratio is None:
        return None
    if ratio + 1e-12 >= STRESS_KS_HIGH_THRESHOLD:
        return "high"
    if ratio + 1e-12 >= STRESS_KS_MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def stress_psi_risk(psi: float | None) -> str | None:
    if psi is None:
        return None
    value = float(psi)
    if not math.isfinite(value) or value < 0:
        return None
    if value + 1e-12 >= STRESS_PSI_HIGH_THRESHOLD:
        return "high"
    if value + 1e-12 >= STRESS_PSI_MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def worst_stress_risk(*risks: str | None) -> str | None:
    known = [risk for risk in risks if risk in _STRESS_RISK_ORDER]
    if not known:
        return None
    return max(known, key=_STRESS_RISK_ORDER.__getitem__)


def stress_risk_cell_color(risk: str | None) -> str | None:
    return _STRESS_RISK_FILLS.get(str(risk or ""))


def stress_risk_label(risk: str | None) -> str:
    return {
        "low": "低风险",
        "medium": "中风险",
        "high": "高风险",
    }.get(str(risk or ""), "无法评估")
