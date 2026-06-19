from __future__ import annotations

import math
from typing import Any

OVERFIT_TRAIN_TEST_REL = 0.10
OVERFIT_TRAIN_OOT_ABS = 0.05
OVERFITTING_TRAIN_TEST_RELATIVE_THRESHOLD = OVERFIT_TRAIN_TEST_REL
OVERFITTING_TRAIN_OOT_ABS_THRESHOLD = OVERFIT_TRAIN_OOT_ABS


def overfitting_check(train_ks: float, test_ks: float, oot_ks: float | None) -> tuple[float, float | None, bool]:
    train_test_gap = abs(train_ks - test_ks) / abs(train_ks) if abs(train_ks) > 1e-12 else 0.0
    train_oot_gap = abs(train_ks - oot_ks) if oot_ks is not None else None
    flag = train_test_gap > OVERFIT_TRAIN_TEST_REL or (
        train_oot_gap is not None and train_oot_gap > OVERFIT_TRAIN_OOT_ABS
    )
    return train_test_gap, train_oot_gap, flag


def overfitting_check_from_validation_results(validation_results: dict[str, Any]) -> dict[str, Any]:
    overall = (
        (validation_results.get("effectiveness") or {})
        .get("overall")
        or []
    )
    by_split: dict[str, float] = {}
    for row in overall:
        if not isinstance(row, dict):
            continue
        split = str(row.get("split") or "").strip().lower()
        ks = _optional_float(row.get("ks"))
        if split and ks is not None:
            by_split[split] = ks
    train_ks = by_split.get("train")
    test_ks = by_split.get("test")
    oot_ks = by_split.get("oot")
    if train_ks is None:
        train_test_relative_diff = None
        train_oot_abs_diff = None
    elif test_ks is None or abs(train_ks) <= 1e-12:
        train_test_relative_diff = None
        train_oot_abs_diff = abs(train_ks - oot_ks) if oot_ks is not None else None
    else:
        train_test_relative_diff, train_oot_abs_diff, _flag = overfitting_check(train_ks, test_ks, oot_ks)
    train_test_status = _threshold_status(
        train_test_relative_diff,
        OVERFITTING_TRAIN_TEST_RELATIVE_THRESHOLD,
    )
    train_oot_status = _threshold_status(
        train_oot_abs_diff,
        OVERFITTING_TRAIN_OOT_ABS_THRESHOLD,
    )
    statuses = {train_test_status, train_oot_status}
    status = "fail" if "fail" in statuses else "not_available" if "not_available" in statuses else "pass"
    return {
        "metric": "ks",
        "status": status,
        "train_ks": train_ks,
        "test_ks": test_ks,
        "oot_ks": oot_ks,
        "train_test_relative_diff": train_test_relative_diff,
        "train_test_threshold": OVERFITTING_TRAIN_TEST_RELATIVE_THRESHOLD,
        "train_test_status": train_test_status,
        "train_oot_abs_diff": train_oot_abs_diff,
        "train_oot_threshold": OVERFITTING_TRAIN_OOT_ABS_THRESHOLD,
        "train_oot_status": train_oot_status,
    }


def _threshold_status(value: float | None, threshold: float) -> str:
    if value is None:
        return "not_available"
    return "fail" if value > threshold else "pass"


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
