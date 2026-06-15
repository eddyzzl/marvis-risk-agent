from __future__ import annotations

import math
from typing import Any

OVERFITTING_TRAIN_TEST_RELATIVE_THRESHOLD = 0.10
OVERFITTING_TRAIN_OOT_ABS_THRESHOLD = 0.05


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
    train_test_relative_diff = (
        None
        if train_ks is None or test_ks is None or abs(train_ks) <= 1e-12
        else abs(train_ks - test_ks) / abs(train_ks)
    )
    train_oot_abs_diff = (
        None
        if train_ks is None or oot_ks is None
        else abs(train_ks - oot_ks)
    )
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
