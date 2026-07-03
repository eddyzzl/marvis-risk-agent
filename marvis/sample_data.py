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

# S3 组合分析套件：表现期快照 (performance snapshot) 合成数据。
PERFORMANCE_TABLE_NAME = "表现期快照.csv"
# 逾期桶状态，语义顺序由好到坏（charged_off 为吸收态/损失态）。工具侧要求
# 调用方显式给 states；这里的顺序即演示数据的规范顺序。
PERFORMANCE_STATES = ("current", "M1", "M2", "M3+", "charged_off")
# 好客户 (y=0) 的月度桶间转移矩阵：绝大多数留在 current，少量轻度逾期后自愈。
# 行=from 状态，列=to 状态，顺序同 PERFORMANCE_STATES；每行和为 1。
_PERF_MATRIX_GOOD = (
    (0.94, 0.05, 0.01, 0.00, 0.00),  # current
    (0.55, 0.30, 0.13, 0.02, 0.00),  # M1
    (0.20, 0.30, 0.30, 0.18, 0.02),  # M2
    (0.05, 0.10, 0.25, 0.45, 0.15),  # M3+
    (0.00, 0.00, 0.00, 0.00, 1.00),  # charged_off (吸收)
)
# 坏客户 (y=1) 的高恶化转移矩阵：更快向深逾期/核销迁移。
_PERF_MATRIX_BAD = (
    (0.70, 0.22, 0.06, 0.02, 0.00),  # current
    (0.20, 0.35, 0.30, 0.13, 0.02),  # M1
    (0.05, 0.15, 0.35, 0.35, 0.10),  # M2
    (0.00, 0.02, 0.10, 0.53, 0.35),  # M3+
    (0.00, 0.00, 0.00, 0.00, 1.00),  # charged_off (吸收)
)
_PERF_N_MONTHS = 12
_PERF_ID_COL = "loan_id"
_PERF_SNAPSHOT_COL = "snapshot_month"
_PERF_BUCKET_COL = "bucket"
_PERF_BALANCE_COL = "balance"

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


def generate_performance_frame(
    sample_df: pd.DataFrame,
    *,
    n_months: int = _PERF_N_MONTHS,
    seed: int = _SEED,
) -> pd.DataFrame:
    """Deterministic 表现期快照 long frame derived from ``sample_df`` (S3).

    For every loan (row of the sample), emits ``n_months`` monthly snapshots with:

    - ``loan_id``: stable per-loan id (row index, zero-padded);
    - ``snapshot_month``: consecutive YYYY-MM months starting from the loan's
      ``apply_month`` (the origination month), so snapshots are per-loan aligned;
    - ``bucket``: a delinquency bucket walked by a Markov chain — bad loans
      (``y==1``) use the high-deterioration matrix, good loans the benign one;
      ``charged_off`` is an absorbing state (once charged off, stays charged off);
    - ``balance``: linear amortization of an initial principal plus small
      deterministic per-loan noise, floored at 0 and forced to 0 once charged off.

    Fully deterministic: a per-loan RNG is seeded from ``seed`` + the loan index,
    so the same ``seed`` yields byte-identical output regardless of row order.
    """
    if n_months < 1:
        raise ValueError("n_months must be positive")
    if "y" not in sample_df.columns or "apply_month" not in sample_df.columns:
        raise ValueError("sample_df must have 'y' and 'apply_month' columns")

    states = PERFORMANCE_STATES
    charged_off_index = len(states) - 1
    good_matrix = np.asarray(_PERF_MATRIX_GOOD, dtype=float)
    bad_matrix = np.asarray(_PERF_MATRIX_BAD, dtype=float)

    # Stable per-loan initial principal (varies by loan_amount when present, else
    # a fixed base) so amortization curves aren't all identical.
    if "loan_amount" in sample_df.columns:
        principals = pd.to_numeric(sample_df["loan_amount"], errors="coerce").fillna(20000.0).to_numpy(dtype=float)
    else:
        principals = np.full(len(sample_df), 20000.0, dtype=float)

    labels = pd.to_numeric(sample_df["y"], errors="coerce").fillna(0).astype(int).to_numpy()
    origination = sample_df["apply_month"].astype(str).tolist()

    records: list[dict] = []
    for loan_index in range(len(sample_df)):
        rng = np.random.default_rng(seed + 1 + loan_index)
        matrix = bad_matrix if labels[loan_index] == 1 else good_matrix
        principal = max(float(principals[loan_index]), 500.0)
        start_month = _month_sequence(origination[loan_index], n_months)
        loan_id = f"L{loan_index:06d}"

        state_index = 0  # every loan starts current
        for month_pos in range(n_months):
            # amortization: linear paydown across n_months, + small noise
            remaining_fraction = max(0.0, 1.0 - month_pos / float(n_months))
            noise = float(rng.normal(0.0, 0.01))
            balance = max(0.0, principal * remaining_fraction * (1.0 + noise))
            if state_index == charged_off_index:
                balance = 0.0
            records.append(
                {
                    _PERF_ID_COL: loan_id,
                    _PERF_SNAPSHOT_COL: start_month[month_pos],
                    _PERF_BUCKET_COL: states[state_index],
                    _PERF_BALANCE_COL: round(balance, 2),
                }
            )
            # advance to next month's bucket (absorbing charged_off stays put)
            if state_index != charged_off_index:
                state_index = int(rng.choice(len(states), p=matrix[state_index]))

    return pd.DataFrame.from_records(
        records, columns=[_PERF_ID_COL, _PERF_SNAPSHOT_COL, _PERF_BUCKET_COL, _PERF_BALANCE_COL]
    )


def _month_sequence(start_month: str, n_months: int) -> list[str]:
    """``n_months`` consecutive YYYY-MM labels starting at ``start_month``.

    Falls back to a fixed 2025-01 anchor when ``start_month`` isn't a parseable
    YYYY-MM (keeps generation total; the sample always supplies YYYY-MM though)."""
    text = str(start_month).strip()
    try:
        if len(text) >= 7 and text[4] == "-":
            year, month = int(text[:4]), int(text[5:7])
        elif len(text) == 6 and text.isdigit():
            year, month = int(text[:4]), int(text[4:])
        else:
            year, month = 2025, 1
    except (ValueError, IndexError):
        year, month = 2025, 1
    out: list[str] = []
    for _ in range(n_months):
        out.append(f"{year:04d}-{month:02d}")
        month += 1
        if month > 12:
            month = 1
            year += 1
    return out


def write_sample_materials(
    target_dir: Path, *, seed: int = _SEED, include_performance: bool = False
) -> Path:
    """Write the sample table + dictionary CSVs into ``target_dir`` (created if
    missing) and return the directory. Deterministic — safe to call repeatedly;
    each call overwrites with byte-identical content for a fixed seed.

    S3: when ``include_performance`` is True, also writes the performance
    (表现期快照) snapshot CSV derived from the same sample, so the portfolio
    analysis flow has a one-click demo table. Left off by default so the
    first-run modeling entry keeps emitting exactly the two files it always has.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    sample = generate_sample_frame(seed=seed)
    sample.to_csv(target_dir / SAMPLE_TABLE_NAME, index=False, encoding="utf-8-sig")
    generate_dictionary_frame().to_csv(target_dir / DICTIONARY_TABLE_NAME, index=False, encoding="utf-8-sig")
    if include_performance:
        generate_performance_frame(sample, seed=seed).to_csv(
            target_dir / PERFORMANCE_TABLE_NAME, index=False, encoding="utf-8-sig"
        )
    return target_dir


__all__ = [
    "DEMO_TASK_NAME_PREFIX",
    "DICTIONARY_TABLE_NAME",
    "PERFORMANCE_STATES",
    "PERFORMANCE_TABLE_NAME",
    "SAMPLE_TABLE_NAME",
    "generate_dictionary_frame",
    "generate_performance_frame",
    "generate_sample_frame",
    "write_sample_materials",
]
