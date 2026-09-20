"""Pure aggregation shared by stress execution and stored-result decoding."""

from collections.abc import Iterable


def stress_test_status(category_statuses: Iterable[str]) -> str:
    statuses = set(category_statuses)
    if not statuses or statuses == {"skipped"}:
        return "skipped"
    if statuses == {"completed"}:
        return "completed"
    if statuses == {"error"}:
        return "failed"
    return "partial"
