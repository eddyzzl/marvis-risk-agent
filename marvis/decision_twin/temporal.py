"""Pure temporal distribution evidence over native, record-matched replay output.

The reference window alone fits score bins. Windows are disjoint descriptive
populations, not proof of out-of-training validation or historical deployment.
"""

from dataclasses import dataclass
import math
import re
from typing import Any

import numpy as np

from marvis.decision_twin._canonical import content_hash, parse_datetime
from marvis.decision_twin.batch_contracts import HistoricalTemporalStability
from marvis.feature.binning import degraded_bin_diagnostic, equal_frequency_edges
from marvis.feature.metrics import compute_psi
from marvis.validation.binning import bin_distribution


_ACTIONS = ("approval", "reject", "review")
_HASH = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class TemporalWindowPopulation:
    window: dict
    positions: tuple[int, ...]
    members_hash: str

    def to_dict(self):
        return {
            **self.window,
            "sample_count": len(self.positions),
            "members_hash": self.members_hash,
        }


@dataclass(frozen=True)
class TemporalPopulation:
    contract_hash: str
    source_count: int
    source_members_hash: str
    reference: TemporalWindowPopulation
    comparisons: tuple[TemporalWindowPopulation, ...]
    excluded_count: int
    excluded_members_hash: str

    def to_dict(self):
        return {
            "schema_version": "decision_twin.temporal_population.v1",
            "contract_hash": self.contract_hash,
            "interval_convention": "[start,end)",
            "source_count": self.source_count,
            "source_members_hash": self.source_members_hash,
            "reference": self.reference.to_dict(),
            "comparisons": [window.to_dict() for window in self.comparisons],
            "excluded_count": self.excluded_count,
            "excluded_members_hash": self.excluded_members_hash,
        }


def _member_bindings(records):
    bindings, ids = [], set()
    for record in records:
        if any(
            not isinstance(record.get(key), str) or not _HASH.fullmatch(record[key])
            for key in ("record_id", "facts_hash")
        ):
            raise ValueError("temporal records require exact authenticated identities")
        if record["record_id"] in ids:
            raise ValueError("temporal record identities must be unique")
        parse_datetime(record.get("decision_at"), "decision_at")
        ids.add(record["record_id"])
        bindings.append(
            {key: record[key] for key in ("record_id", "decision_at", "facts_hash")}
        )
    return bindings


def bind_temporal_population(records, contract: HistoricalTemporalStability):
    # Revalidate at the pure boundary too, including model_copy/constructed callers.
    contract = HistoricalTemporalStability.model_validate(contract.model_dump())
    bindings = _member_bindings(records)
    times = [parse_datetime(record["decision_at"], "decision_at") for record in records]
    occupied = set()
    populations = []
    for window in [contract.reference_window, *contract.comparison_windows]:
        start, end = (
            parse_datetime(window.start, "start"),
            parse_datetime(window.end, "end"),
        )
        positions = tuple(i for i, at in enumerate(times) if start <= at < end)
        if occupied.intersection(positions):
            raise ValueError("temporal population windows overlap")
        occupied.update(positions)
        populations.append(
            TemporalWindowPopulation(
                window.model_dump(),
                positions,
                content_hash([bindings[i] for i in positions]),
            )
        )
    excluded = [bindings[i] for i in range(len(records)) if i not in occupied]
    return TemporalPopulation(
        contract_hash=content_hash(contract.model_dump()),
        source_count=len(records),
        source_members_hash=content_hash(bindings),
        reference=populations[0],
        comparisons=tuple(populations[1:]),
        excluded_count=len(excluded),
        excluded_members_hash=content_hash(excluded),
    )


def _actions(decisions, positions):
    values = [decisions[i]["action"]["type"] for i in positions]
    count = len(values)
    counts = [values.count(action) for action in _ACTIONS]
    return {
        "categories": list(_ACTIONS),
        "counts": counts,
        "proportions": None if not count else [n / count for n in counts],
        "approval_rate": None if not count else counts[0] / count,
    }


def _check(metric, value, threshold, *, reason=None, unit="psi"):
    passed = None if value is None else value <= threshold
    return {
        "metric": metric,
        "value": value,
        "operator": "<=",
        "threshold": threshold,
        "unit": unit,
        "passed": passed,
        "status": "unknown" if passed is None else "passed" if passed else "failed",
        "reason": reason,
    }


def _psi(expected, actual):
    expected, actual = (
        np.asarray(expected, dtype=float),
        np.asarray(actual, dtype=float),
    )
    if (
        not np.all(np.isfinite(expected))
        or not np.all(np.isfinite(actual))
        or expected.sum() <= 0
        or actual.sum() <= 0
    ):
        raise ValueError("temporal distributions require finite, nonempty support")
    value = compute_psi(expected, actual)
    if not math.isfinite(value):
        raise ValueError("temporal PSI is not finite")
    return value


def _reference_bins(scores, population, contract):
    reference_scores = scores[list(population.reference.positions)]
    evidence: dict[str, Any] = {
        "status": "unknown",
        "reason": "insufficient_reference_support",
        "requested_bin_count": contract.bin_count,
        "actual_bin_count": None,
        "edges": None,
        "proportions": None,
        "fit_sample_count": len(reference_scores),
        "fit_members_hash": population.reference.members_hash,
        "fit_scope": "reference_window_only",
    }
    if len(reference_scores) < contract.minimum_reference_rows:
        return None, evidence
    edges = equal_frequency_edges(reference_scores, contract.bin_count)
    evidence["actual_bin_count"] = len(edges) - 1
    if not np.all(np.isfinite(edges[1:-1])) or not np.all(np.diff(edges) > 0):
        evidence["reason"] = "invalid_reference_bins"
        return None, evidence
    evidence["edges"] = [
        "-inf" if value == -np.inf else "inf" if value == np.inf else float(value)
        for value in edges
    ]
    evidence["proportions"] = bin_distribution(reference_scores, edges).tolist()
    degraded = degraded_bin_diagnostic(edges, contract.bin_count)
    if degraded is not None:
        evidence.update(reason="degraded_reference_binning", diagnostic=degraded)
        return None, evidence
    evidence.update(status="measured", reason=None)
    return edges, evidence


def measure_temporal_stability(
    records,
    decisions,
    contract: HistoricalTemporalStability,
    *,
    population: TemporalPopulation,
    package_hash: str,
    score_product: str | None,
    package_kind: str = "model",
):
    actual_population = bind_temporal_population(records, contract)
    if actual_population != population:
        raise ValueError("temporal population binding drifted")
    rule_only = package_kind == "rule_only"
    if package_kind not in {"model", "rule_only"} or len(records) != len(decisions):
        raise ValueError("temporal replay score population mismatch")
    if (rule_only and score_product is not None) or (
        not rule_only
        and score_product not in {"raw_pd", "calibrated_pd", "scorecard_points"}
    ):
        raise ValueError("temporal replay score population mismatch")
    scores = []
    for record, decision in zip(records, decisions, strict=True):
        if any(
            decision.get(key) != record[key]
            for key in ("record_id", "decision_at", "facts_hash")
        ):
            raise ValueError("temporal replay record binding drifted")
        if (
            decision.get("package_hash") != package_hash
            or decision.get("score_product") != score_product
        ):
            raise ValueError("temporal replay package or score product drifted")
        score = decision.get("score")
        if rule_only and (
            "score" not in decision or "score_product" not in decision or score is not None
        ):
            raise ValueError("rule-only replay cannot contain a model score")
        if not rule_only and (type(score) not in (int, float) or not math.isfinite(score)):
            raise ValueError("temporal replay requires finite native scores")
        if not rule_only and score_product != "scorecard_points" and not 0 <= score <= 1:
            raise ValueError("temporal probability is outside its native range")
        if decision.get("action", {}).get("type") not in _ACTIONS:
            raise ValueError("temporal replay requires a supported native action")
        scores.append(score)
    if rule_only:
        # No score was produced, so no rows fitted score bins. Keep the actual
        # temporal membership and action distribution independently measurable.
        scores, edges = None, None
        bins = {
            "status": "unknown", "reason": "package_has_no_score",
            "requested_bin_count": contract.bin_count, "actual_bin_count": None,
            "edges": None, "proportions": None, "fit_sample_count": 0,
            "fit_members_hash": None, "fit_scope": "not_applicable",
        }
    else:
        scores = np.asarray(scores, dtype=float)
        edges, bins = _reference_bins(scores, population, contract)
    reference_actions = _actions(decisions, population.reference.positions)
    reference_supported = (
        len(population.reference.positions) >= contract.minimum_reference_rows
    )
    comparisons = []
    thresholds = contract.thresholds
    for window in population.comparisons:
        actions = _actions(decisions, window.positions)
        supported = len(window.positions) >= contract.minimum_comparison_rows
        support_reason = (
            "insufficient_reference_support"
            if not reference_supported
            else "insufficient_comparison_support"
            if not supported
            else None
        )
        score_reason = support_reason or bins["reason"]
        score_dist = None
        score_psi = None
        if score_reason is None:
            score_dist = bin_distribution(
                scores[list(window.positions)], edges
            ).tolist()
            score_psi = _psi(bins["proportions"], score_dist)
        action_psi = (
            None
            if support_reason
            else _psi(reference_actions["proportions"], actions["proportions"])
        )
        rate_delta = (
            None
            if support_reason
            else actions["approval_rate"] - reference_actions["approval_rate"]
        )
        comparisons.append(
            {
                "window": window.to_dict(),
                "actions": actions,
                "score_proportions": score_dist,
                "approval_rate_delta": rate_delta,
                "checks": [
                    _check(
                        "score_psi",
                        score_psi,
                        thresholds.max_score_psi,
                        reason=score_reason,
                    ),
                    _check(
                        "action_psi",
                        action_psi,
                        thresholds.max_action_psi,
                        reason=support_reason,
                    ),
                    _check(
                        "absolute_approval_rate_delta",
                        None if rate_delta is None else abs(rate_delta),
                        thresholds.max_absolute_approval_rate_delta,
                        reason=support_reason,
                        unit="rate",
                    ),
                ],
            }
        )
    checks = [check for window in comparisons for check in window["checks"]]
    measured = sum(check["value"] is not None for check in checks)
    summary = {}
    for name in ("score_psi", "action_psi", "absolute_approval_rate_delta"):
        values = [check["value"] for check in checks if check["metric"] == name]
        # A maximum over only the supported windows would conceal an unchecked period.
        summary["max_" + name] = (
            None if any(value is None for value in values) else max(values)
        )
    return {
        "schema_version": "decision_twin.temporal_stability.v1",
        "status": "unknown"
        if not measured
        else "measured"
        if measured == len(checks)
        else "partial",
        "verdict": "failed"
        if any(check["passed"] is False for check in checks)
        else "insufficient_evidence"
        if measured < len(checks)
        else "passed",
        "value": None if not measured else summary,
        "population": population.to_dict(),
        "policy": contract.model_dump(),
        "package_hash": package_hash,
        "score_product": score_product,
        "reference": {
            "window": population.reference.to_dict(),
            "actions": reference_actions,
            "score_bins": bins,
        },
        "comparisons": comparisons,
        "interpretation": "retrospective_temporal_distribution",
        "historical_package_availability": "not_established",
        "out_of_training_validation": "not_established",
        "causal_gain_verified": False,
        "automatic_action_permitted": False,
    }
