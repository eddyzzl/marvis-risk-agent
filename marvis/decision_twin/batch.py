"""Native package replay with matched populations and explicit unknown dimensions.

V2 is additive: it does not loosen the complete, observed-champion v1 contract.
Only native authenticated PackageStore snapshots can execute; there is no caller
supplied adapter, callback, Python module, or fabricated execution receipt.
"""

from datetime import UTC, datetime, timedelta
import hmac
import math

import pandas as pd

from marvis.decision_twin._canonical import iso_z
from marvis.decision_twin.batch_contracts import (
    HistoricalReconciliationRequest,
    HistoricalReplayRequest,
)
from marvis.decision_twin.batch_material import (
    BatchMaterial,
    OUTCOME_KIND,
    REPLAY_KIND,
    record_key,
    scalar,
    timestamp,
)
from marvis.packs.strategy.economics import pricing_metrics
from marvis.reference_decision.contracts import DecisionError
from marvis.reference_decision.evaluation import evaluate


def _unknown(reason):
    return {"status": "unknown", "value": None, "reason": reason}


def _economics(records, decisions, contract, product):
    if contract is None:
        return _unknown("economic_assumptions_not_declared")
    if product not in {"raw_pd", "calibrated_pd"}:
        return _unknown("score_is_not_a_probability")
    approved = [
        (record, decision)
        for record, decision in zip(records, decisions, strict=True)
        if decision["action"]["type"] == "approval"
    ]
    frame = pd.DataFrame(
        {
            "ead": [r["ead"] for r, _ in approved],
            "pd": [d["score"] for _, d in approved],
        },
        dtype=float,
    )
    measured = pricing_metrics(
        pd.Series(contract.annual_rate, index=frame.index, dtype=float),
        pd.Series(float("nan"), index=frame.index, dtype=float),
        ead=frame["ead"],
        pd=frame["pd"],
        lgd=contract.lgd,
        funding_rate=contract.funding_rate,
        term_months=contract.term_months,
        operating_cost_per_loan=contract.operating_cost_per_loan,
    )["economics"]
    by_row = measured.pop("by_row")
    for (record, decision), row in zip(approved, by_row, strict=True):
        decision["economic_projection"] = {
            "ead": record["ead"],
            "expected_loss": row["expected_loss"],
            "profit": row["profit"],
            "currency": contract.currency,
        }
    return {
        "status": "estimated",
        "interpretation": "fixed_condition_approved_loans",
        "value": {**measured, "ead": math.fsum(frame["ead"])},
        "currency": contract.currency,
        "assumptions": contract.model_dump(),
        "causal_gain_verified": False,
        "pd_calibration_status": "package_calibrated"
        if product == "calibrated_pd"
        else "not_established",
        "denominator": len(approved),
    }


def _fairness(records, decisions, contract):
    if contract is None:
        return _unknown("protected_groups_not_declared")
    groups = {}
    for record, decision in zip(records, decisions, strict=True):
        groups.setdefault(record["group"], []).append(
            decision["action"]["type"] == "approval"
        )
    if len(groups) < 2 or min(map(len, groups.values())) < contract.minimum_group_size:
        return {
            **_unknown("insufficient_protected_group_support"),
            "group_counts": {key: len(value) for key, value in groups.items()},
        }
    rates = [
        {
            "group": group,
            "count": len(values),
            "approvals": sum(values),
            "approval_rate": sum(values) / len(values),
        }
        for group, values in sorted(groups.items())
    ]
    return {
        "status": "measured",
        "value": max(r["approval_rate"] for r in rates)
        - min(r["approval_rate"] for r in rates),
        "unit": "rate",
        "groups": rates,
        "governance_ref": contract.governance_ref,
        "interpretation": "descriptive_approval_disparity_not_fairness_certification",
    }


def _constraints(contract, metrics):
    values = {
        "approval_rate": (metrics["approval_rate"], "rate"),
        "protected_group_disparity": (metrics["protected_groups"]["value"], "rate"),
        "operations_capacity": (
            metrics["operations_capacity"]["value"],
            None if contract.capacity is None else contract.capacity.unit,
        ),
    }
    economy = metrics["economics"]
    for key in ("ead", "expected_loss", "profit"):
        values[key] = (
            (economy["value"] or {}).get(key),
            None if contract.economics is None else contract.economics.currency,
        )
    checks = []
    for criterion in contract.constraints:
        value, unit = values[criterion.metric]
        if value is not None and criterion.unit != unit:
            raise DecisionError("historical_constraint_unit_mismatch")
        passed = (
            None
            if value is None
            else (
                value >= criterion.threshold
                if criterion.operator == ">="
                else value <= criterion.threshold
            )
        )
        checks.append(
            {
                **criterion.model_dump(),
                "value": value,
                "passed": passed,
                "status": "unknown"
                if passed is None
                else "passed"
                if passed
                else "failed",
            }
        )
    state = (
        "not_declared"
        if not checks
        else "failed"
        if any(c["passed"] is False for c in checks)
        else "insufficient_evidence"
        if any(c["passed"] is None for c in checks)
        else "passed"
    )
    return {"status": state, "checks": checks, "automatic_action_permitted": False}


def replay_batch(
    material: BatchMaterial, contract: HistoricalReplayRequest, proposal_hash: str
):
    binding, records, proposal = material.prepare(contract)
    if not hmac.compare_digest(proposal_hash, proposal["proposal_hash"]):
        raise DecisionError("historical_proposal_stale", 409)
    scenarios = []
    for scenario in contract.scenarios:
        decisions = []
        with material.packages.snapshot(scenario.package_hash) as (manifest, directory):
            for record in records:
                result = evaluate(manifest, directory, record["features"])
                # Runtime timings are not business values and are excluded from the
                # deterministic receipt identity. Exact model/rule outputs are retained.
                result.pop("timing_ms", None)
                decisions.append(
                    {
                        "record_id": record["record_id"],
                        "decision_at": record["decision_at"],
                        "facts_hash": record["facts_hash"],
                        "package_hash": scenario.package_hash,
                        **result,
                    }
                )
        approvals = sum(d["action"]["type"] == "approval" for d in decisions)
        metrics = {
            "count": len(records),
            "approval_count": approvals,
            "approval_rate": approvals / len(records),
            "review_count": sum(d["action"]["type"] == "review" for d in decisions),
            "economics": _economics(
                records,
                decisions,
                contract.economics,
                manifest["configuration"]["score_product"],
            ),
            "protected_groups": _fairness(records, decisions, contract.protected_group),
            "operations_capacity": _unknown("operations_assumptions_not_declared"),
            "stability": _unknown("independent_time_population_not_bound"),
        }
        if contract.capacity:
            metrics["operations_capacity"] = {
                "status": "estimated",
                "value": math.fsum(
                    contract.capacity.per_action[d["action"]["type"]] for d in decisions
                ),
                **contract.capacity.model_dump(),
            }
        scenarios.append(
            {
                **scenario.model_dump(),
                "population_hash": proposal["population_hash"],
                "metrics": metrics,
                "constraints": _constraints(contract, metrics),
                "decisions": decisions,
                "observed_execution": False,
                "receipt_origin": "marvis_native_package_replay",
            }
        )
    baseline = next(s for s in scenarios if s["kind"] == "baseline")
    for scenario in scenarios:
        scenario["comparison"] = {
            "approval_rate_delta": scenario["metrics"]["approval_rate"]
            - baseline["metrics"]["approval_rate"],
            "decision_change_count": sum(
                a["action"]["type"] != b["action"]["type"]
                for a, b in zip(
                    scenario["decisions"], baseline["decisions"], strict=True
                )
            ),
            "denominator": len(records),
            "population_hash": proposal["population_hash"],
            "causal_gain_verified": False,
        }
        baseline_economics = baseline["metrics"]["economics"]["value"]
        candidate_economics = scenario["metrics"]["economics"]["value"]
        scenario["comparison"]["estimated_economics_delta"] = (
            None
            if baseline_economics is None or candidate_economics is None
            else {
                key: candidate_economics[key] - baseline_economics[key]
                for key in ("ead", "expected_loss", "profit")
            }
        )
    observations = _unknown("historical_actions_not_imported")
    if contract.observed_actions:
        observations = {
            "status": "imported",
            "origin": "external_history_import",
            "source_ref": contract.observed_actions.source_ref,
            "marvis_execution_verified": False,
            "records": [
                {
                    k: r[k]
                    for k in (
                        "record_id",
                        "decision_at",
                        "observed_action",
                        "action_recorded_at",
                    )
                }
                for r in records
            ],
        }
        for scenario in scenarios:
            scenario["comparison"]["agreement_with_imported_history"] = sum(
                d["action"]["type"] == r["observed_action"]
                for d, r in zip(scenario["decisions"], records, strict=True)
            ) / len(records)
    payload = {
        "schema_version": "decision_twin.batch_receipt.v2",
        "contract": contract.model_dump(),
        "contract_hash": contract.contract_hash,
        "proposal_hash": proposal_hash,
        "source": {
            "dataset_id": binding.dataset_id,
            "content_hash": binding.content_hash,
            "population": contract.population,
            "population_count": len(records),
            "population_hash": proposal["population_hash"],
            "assurance": proposal["source_assurance"],
        },
        "scenarios": scenarios,
        "observed_actions": observations,
        "package_time_scope": "retrospective_policy_simulation",
        "historical_package_availability": "not_established",
        "causal_assessment": {
            "identifiability": "unidentified",
            "reason": "same_historical_features_do_not_establish_unobserved_treatment_outcomes",
            "causal_gain_verified": False,
        },
        "authority": "proposal_only",
        "automatic_action_permitted": False,
    }
    return material.publish(REPLAY_KIND, payload, binding=binding)


def reconcile_batch(
    material: BatchMaterial,
    contract: HistoricalReconciliationRequest,
    proposal_hash: str,
):
    if not hmac.compare_digest(contract.contract_hash, proposal_hash):
        raise DecisionError("historical_reconciliation_proposal_stale", 409)
    replay = material.load(contract.replay_artifact_id)["payload"]
    actions = replay["observed_actions"]
    if actions.get("origin") != "external_history_import":
        raise DecisionError("observed_historical_actions_required")
    binding, frame = material.dataset(
        contract.dataset_id, contract.expected_content_hash
    )
    columns = (
        contract.record_id_col,
        contract.observed_at_col,
        contract.actual_loss_col,
        contract.actual_profit_col,
    )
    if set(columns) - set(frame.columns):
        raise DecisionError("historical_outcome_columns_missing")
    approved = {
        r["record_id"]: r
        for r in actions["records"]
        if r["observed_action"] == "approval"
    }
    outcome_by_id = {}
    for _, row in frame.iterrows():
        key = record_key(row[contract.record_id_col])
        if key in outcome_by_id or key not in approved:
            raise DecisionError("historical_outcomes_must_match_observed_approvals")
        outcome_by_id[key] = row
    if set(outcome_by_id) != set(approved):
        raise DecisionError("historical_outcomes_must_match_observed_approvals")
    as_of = timestamp(contract.reconciled_at, "reconciled_at")
    if as_of > datetime.now(UTC):
        raise DecisionError("historical_reconciliation_cutoff_is_in_future")
    baseline = next(s for s in replay["scenarios"] if s["kind"] == "baseline")
    forecasts = {
        r["record_id"]: r.get("economic_projection") for r in baseline["decisions"]
    }
    records = []
    for key, action in approved.items():
        row = outcome_by_id[key]
        at = timestamp(row[contract.observed_at_col], "outcome_observed_at")
        mature_at = timestamp(action["decision_at"], "decision_at") + timedelta(
            days=contract.maturity_days
        )
        if at > as_of or at < mature_at or mature_at > as_of:
            raise DecisionError("historical_outcome_not_mature_at_cutoff")
        loss, profit = (
            scalar(row[contract.actual_loss_col]),
            scalar(row[contract.actual_profit_col]),
        )
        if (
            type(loss) not in (int, float)
            or type(profit) not in (int, float)
            or loss < 0
        ):
            raise DecisionError("invalid_historical_cashflow")
        projection = forecasts[key]
        if projection and projection["currency"] != contract.currency:
            raise DecisionError("historical_cashflow_currency_mismatch")
        records.append(
            {
                "record_id": key,
                "observed_at": iso_z(at),
                "mature_at": iso_z(mature_at),
                "actual_loss": loss,
                "actual_profit": profit,
                "projected_loss": None
                if projection is None
                else projection["expected_loss"],
                "projected_profit": None
                if projection is None
                else projection["profit"],
                "loss_delta": None
                if projection is None
                else loss - projection["expected_loss"],
                "profit_delta": None
                if projection is None
                else profit - projection["profit"],
            }
        )
    comparable = [r for r in records if r["loss_delta"] is not None]
    payload = {
        "schema_version": "decision_twin.batch_reconciliation.v2",
        "contract": contract.model_dump(),
        "contract_hash": contract.contract_hash,
        "replay_artifact_id": contract.replay_artifact_id,
        "origin": "external_history_import",
        "status": "reconciled"
        if len(comparable) == len(records) and records
        else "partial_comparability",
        "approved_denominator": len(approved),
        "comparable_denominator": len(comparable),
        "currency": contract.currency,
        "actual_loss": math.fsum(r["actual_loss"] for r in records),
        "actual_profit": math.fsum(r["actual_profit"] for r in records),
        "comparable_loss_delta": None
        if not comparable
        else math.fsum(r["loss_delta"] for r in comparable),
        "comparable_profit_delta": None
        if not comparable
        else math.fsum(r["profit_delta"] for r in comparable),
        "records": records,
        "marvis_execution_verified": False,
        "causal_gain_verified": False,
        "automatic_action_permitted": False,
    }
    return material.publish(OUTCOME_KIND, payload, binding=binding)
