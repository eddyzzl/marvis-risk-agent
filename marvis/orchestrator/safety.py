from __future__ import annotations

import re

from marvis.orchestrator.contracts import PlanStep


METRIC_FIELDS = frozenset({
    "ks",
    "auc",
    "psi",
    "iv",
    "total_iv",
    "lift",
    "gini",
    "bad_rate",
    "approval_rate",
    "approved_bad_rate",
    "rejected_bad_rate",
    "expected_profit",
})
_METRIC_LITERAL_RE = re.compile(
    r"(?<![A-Za-z0-9_])("
    + "|".join(sorted(METRIC_FIELDS, key=len, reverse=True))
    + r")\s*[:=]\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",
    re.IGNORECASE,
)


def is_safety_step(step: PlanStep) -> bool:
    if step.tool_ref.tool == "execute_join" or is_draft_run_step(step):
        return True
    if step.tool_ref.plugin == "strategy" and step.tool_ref.tool == "backtest_strategy":
        return False
    return any(check.kind == "range" for check in step.post_checks)


def is_draft_run_step(step: PlanStep) -> bool:
    return step.tool_ref.plugin == "drafts" and step.tool_ref.tool == "run_draft"


def literal_metric_claims(value) -> set[str]:
    """Return metric result names asserted as literals in a nested value."""

    fields: set[str] = set()
    if _is_evidence_binding(value):
        return fields
    if isinstance(value, str):
        fields.update(
            match.group(1).lower()
            for match in _METRIC_LITERAL_RE.finditer(value)
        )
        return fields
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in METRIC_FIELDS and _is_numeric_literal(item):
                fields.add(normalized)
            fields.update(literal_metric_claims(item))
        return fields
    if isinstance(value, (list, tuple)):
        for item in value:
            fields.update(literal_metric_claims(item))
    return fields


def _is_numeric_literal(value) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if not isinstance(value, str) or _is_evidence_binding(value):
        return False
    try:
        float(value.strip())
    except ValueError:
        return False
    return True


def _is_evidence_binding(value) -> bool:
    if not isinstance(value, str):
        return False
    return value.startswith("$ref:") or (
        value.startswith("{slot:") and value.endswith("}")
    )


__all__ = [
    "METRIC_FIELDS",
    "is_draft_run_step",
    "is_safety_step",
    "literal_metric_claims",
]
