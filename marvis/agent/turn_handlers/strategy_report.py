"""Strategy report for governed Agent turns."""

from __future__ import annotations

from . import strategy_contracts as strategy_contracts_lane


def _raise_corrupt_report_optional(
    label: str,
    *,
    cause: Exception | None = None,
) -> None:
    error = strategy_contracts_lane._StrategyV2EvidenceSetupError(
        "strategy_report_bundle_v2_optional_evidence_invalid",
        f"最新 {label} artifact 未通过完整认证；平台不会回退到旧证据。",
    )
    if cause is None:
        raise error
    raise error from cause
