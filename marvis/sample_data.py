"""Built-in sample credit data generator (UX-9).

Generates a small, deterministic synthetic credit-scoring dataset so a new user
(or an internal demo) can try the modeling flow without preparing their own
material directory first. Two files are written:

  - ``样本表.csv``: 1500 rows with an ``apply_month`` time column, a binary ``y``
    label, and 6 numeric features with a real (if modest) relationship to ``y`` so
    the screen gate/training produce a non-degenerate KS instead of noise.
  - ``特征字典.csv``: a 特征名/含义 dictionary covering the 6 features + apply_month/y,
    so the demo also exercises GAP-4's business-meaning columns/tooltips end to end.

Fully deterministic (fixed seed, no wall-clock/random-uuid content) so repeated
"用示例数据试跑" clicks always produce the same data — a stable reproduction
vehicle for UI regressions, per the review's UX-9 guidance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_TABLE_NAME = "样本表.csv"
DICTIONARY_TABLE_NAME = "特征字典.csv"
# NOTE: must stay inside MODEL_ID_RE (marvis.api_task_helpers) — that pattern accepts
# \w / CJK / hyphen / space but not full-width brackets, so a "【示例】" prefix would
# make every demo task fail model_name validation at creation time.
DEMO_TASK_NAME_PREFIX = "示例-"

_SEED = 20260701
_N_ROWS = 1500
_MONTHS = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06"]

# Feature name -> (business meaning, mean, std) for the 6 generated numeric features.
_FEATURES: dict[str, tuple[str, float, float]] = {
    "credit_score": ("央行征信评分（越高越优质）", 620.0, 80.0),
    "debt_income_ratio": ("负债收入比", 0.35, 0.15),
    "monthly_income": ("月收入（元）", 8000.0, 3500.0),
    "loan_amount": ("申请贷款金额（元）", 20000.0, 9000.0),
    "history_overdue_count": ("历史逾期次数", 0.8, 1.2),
    "account_age_months": ("账户开户月数", 36.0, 20.0),
}
_DICTIONARY_ROWS: dict[str, str] = {
    "apply_month": "申请月份（时间切分列）",
    "y": "是否违约（1=坏客户，0=好客户）",
    **{name: meaning for name, (meaning, _mean, _std) in _FEATURES.items()},
}


def generate_sample_frame(*, n_rows: int = _N_ROWS, seed: int = _SEED) -> pd.DataFrame:
    """Deterministic synthetic credit sample: apply_month + y + 6 numeric features.

    The label is generated from a logistic function of a few features (mainly
    credit_score/debt_income_ratio/history_overdue_count) so screening/training on
    this data produces a real, non-trivial KS rather than pure noise."""
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    for name, (_meaning, mean, std) in _FEATURES.items():
        values = rng.normal(loc=mean, scale=std, size=n_rows)
        if name in {"loan_amount", "monthly_income", "account_age_months"}:
            values = np.clip(values, 500.0, None)
        if name == "history_overdue_count":
            values = np.clip(np.round(values), 0, None)
        if name == "credit_score":
            values = np.clip(values, 300.0, 850.0)
        if name == "debt_income_ratio":
            values = np.clip(values, 0.0, 1.5)
        data[name] = values

    # Standardize the risk-relevant drivers before combining so the logistic
    # relationship is stable regardless of each feature's raw scale.
    def _z(values: np.ndarray) -> np.ndarray:
        std = values.std()
        return (values - values.mean()) / std if std else values * 0.0

    # Intercept -2.2 targets a realistic ~19% bad rate (real credit portfolios
    # run far below 50/50) given the z-scored drivers average to ~0.
    logit = (
        -2.2
        - 1.1 * _z(data["credit_score"])
        + 0.9 * _z(data["debt_income_ratio"])
        + 0.7 * _z(data["history_overdue_count"])
        - 0.2 * _z(data["account_age_months"])
    )
    prob_bad = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n_rows) < prob_bad).astype(int)

    apply_month = rng.choice(_MONTHS, size=n_rows)

    frame = pd.DataFrame({"apply_month": apply_month, "y": y, **data})
    # Round to keep the CSV compact/readable; not load-bearing for detection.
    for name in _FEATURES:
        frame[name] = frame[name].round(2)
    return frame


def generate_dictionary_frame() -> pd.DataFrame:
    """The 特征名/含义 dictionary CSV covering every generated column (GAP-4 demo)."""
    return pd.DataFrame(
        {"特征名": list(_DICTIONARY_ROWS.keys()), "含义": list(_DICTIONARY_ROWS.values())}
    )


def write_sample_materials(target_dir: Path, *, seed: int = _SEED) -> Path:
    """Write the sample table + dictionary CSVs into ``target_dir`` (created if
    missing) and return the directory. Deterministic — safe to call repeatedly;
    each call overwrites with byte-identical content for a fixed seed."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    generate_sample_frame(seed=seed).to_csv(target_dir / SAMPLE_TABLE_NAME, index=False, encoding="utf-8-sig")
    generate_dictionary_frame().to_csv(target_dir / DICTIONARY_TABLE_NAME, index=False, encoding="utf-8-sig")
    return target_dir


__all__ = [
    "DEMO_TASK_NAME_PREFIX",
    "DICTIONARY_TABLE_NAME",
    "SAMPLE_TABLE_NAME",
    "generate_dictionary_frame",
    "generate_sample_frame",
    "write_sample_materials",
]
