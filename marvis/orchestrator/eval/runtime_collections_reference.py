"""Independent integer cashflow and installment reconciliation.

Only typed declarations are consumed. This proves their arithmetic, not their
source truth, causal recovery effectiveness, or customer-contact permission.
"""
from datetime import UTC, datetime, timedelta

from .runtime_contracts import digest


def at(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.utcoffset() is None:
        raise ValueError("timestamp timezone required")
    return result.astimezone(UTC)


def reconciliation_reference(case, flows, request, schedule=None):
    end, cutoff, opened = at(request["as_of"]), at(request["knowledge_cutoff"]), at(case["opened_at"])
    if end < opened or cutoff < end or len(flows) > 10000:
        raise ValueError("invalid reconciliation scope")
    keys = [(row["source_id"], row["event_id"]) for row in flows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate cashflow identity")
    reasons = set()
    coverage = {row["source_id"]: row for row in request["coverage"]}
    for identity in request["expected_sources"]:
        row = coverage.get(identity)
        if row is None:
            reasons.add("missing_source_coverage:" + identity)
        elif (row["status"] != "complete" or at(row["from_at"]) > opened
              or at(row["through_at"]) < end or at(row["available_at"]) > cutoff):
            reasons.add("incomplete_source_coverage:" + identity)
    visible = []
    for row in flows:
        if row["case_id"] != case["case_id"] or row["unit"] != case["unit"] or at(row["event_at"]) < opened:
            raise ValueError("cashflow case/unit differs")
        if at(row["event_at"]) <= end:
            if row["source_id"] not in request["expected_sources"]:
                reasons.add("undeclared_event_source:" + row["source_id"])
            if row["available_at"] is None:
                reasons.add("cashflow_availability_unknown")
            elif at(row["available_at"]) <= cutoff:
                visible.append(row)
    known = {(row["source_id"], row["event_id"]) for row in visible}
    for row in visible:
        ref = row.get("reverses")
        if ref and (ref["source_id"], ref["event_id"]) not in known:
            reasons.add("reversal_original_unavailable_at_cutoff")
    totals = {kind: sum(row["amount_minor"] for row in visible if row["kind"] == kind)
              for kind in ("payment", "payment_reversal", "cost", "cost_reversal")}
    pay, cost = totals["payment"] - totals["payment_reversal"], totals["cost"] - totals["cost_reversal"]
    amounts = {**{key + "_minor": value for key, value in totals.items()}, "net_payments_minor": pay,
               "net_cost_minor": cost, "net_recovery_minor": pay - cost,
               "remaining_balance_minor": max(case["opening_balance_minor"] - pay, 0),
               "overpayment_minor": max(pay - case["opening_balance_minor"], 0)}
    parts = []
    if request["schedule_id"]:
        if schedule is None or schedule["case_id"] != case["case_id"] or schedule["unit"] != case["unit"]:
            raise ValueError("installment case or unit differs")
        for part in schedule["installments"]:
            payments = [row for row in visible if row.get("installment") == {"schedule_id": request["schedule_id"], "installment_id": part["installment_id"]}]
            identities = {(row["source_id"], row["event_id"]) for row in payments}
            paid = sum(row["amount_minor"] for row in payments) - sum(row["amount_minor"] for row in visible if row.get("reverses") and (row["reverses"]["source_id"], row["reverses"]["event_id"]) in identities)
            parts.append({**part, "observed_paid_minor": paid,
                          "status": "unknown" if reasons else "paid" if paid >= part["amount_minor"] else "overdue" if at(part["due_at"]) <= end else "not_due",
                          "remaining_minor": None if reasons else max(part["amount_minor"] - paid, 0)})
    maturity = {"status": "not_declared", "mature_at": None, "mature_outcome_verified": False}
    if request["maturity"]:
        mature = opened + timedelta(days=request["maturity"]["observation_days"])
        maturity = {"status": "insufficient_evidence" if reasons else "mature_under_declared_policy" if end >= mature else "immature",
                    "mature_at": mature.isoformat().replace("+00:00", "Z"), "policy": request["maturity"],
                    "mature_outcome_verified": not reasons and end >= mature,
                    "scope": "declared_observation_window_not_independent_source_truth"}
    return {"schema_version": "collection.reconciliation.v1", "case_id": case["case_id"],
            "contract": request, "case_hash": digest(case), "members_hash": digest([digest(row) for row in visible]),
            "status": "insufficient_evidence" if reasons else "reconciled_under_declared_coverage",
            "reasons": sorted(reasons), "unit": case["unit"], "amounts": None if reasons else amounts,
            "observed_subtotals": amounts, "visible_event_count": len(visible), "stored_event_count": len(flows),
            "installments": parts, "unallocated_payment_count": sum(row["kind"] == "payment" and row.get("installment") is None for row in visible),
            "interpretation": "retrospective_source_declared_cashflows", "allocation_scope": "case_total_balance_without_principal_interest_split",
            "maturity": maturity, "source_truth_verified": False, "incremental_recovery_identified": False, "external_action_executed": False}


def batch_preview_reference(request):
    from collections import Counter
    from zoneinfo import ZoneInfo
    from .runtime_strategy_reference import expression_reference

    policy, spec = request["policy"], request["strategy"]
    end, cutoff = at(request["as_of"]), at(request["knowledge_cutoff"])
    queues = {row["queue_id"]: row for row in policy["queues"]}
    histories = {(row["subject_namespace"], row["subject_token"]): row for row in request["histories"]}
    rules = sorted(spec["rules"], key=lambda row: (row["priority"], row["rule_id"]))
    evaluated = []
    for case in request["cases"]:
        action, matched = spec["default_action"], None
        for rule in rules:
            if expression_reference(case["features"], rule["condition"]):
                action, matched = rule["action"], rule["rule_id"]
                break
        if action["type"] != "collection" or action["value"]["policy_hash"] != digest(policy):
            raise ValueError("collection action policy differs")
        evaluated.append((case, action, matched))
    evaluated.sort(key=lambda row: (-row[1]["value"]["priority"], row[0]["case_id"]))
    used, provisional, spent, results = Counter(), {}, 0, []
    for case, envelope, matched in evaluated:
        action = envelope["value"]
        subject = (case["subject_namespace"], case["subject_token"])
        history = histories.get(subject)
        state, reason = "held", "strategy_hold"
        if not at(policy["valid_from"]) <= end < at(policy["valid_until"]):
            reason = "policy_outside_validity"
        elif action["kind"] in {"contact", "review"}:
            queue = queues[action["queue_id"]]
            if action["kind"] == "contact":
                if action["channel"] not in queue["channels"]:
                    raise ValueError("collection channel differs")
                local = end.astimezone(ZoneInfo(policy["timezone"]))
                open_window = any(local.isoweekday() in row["weekdays"] and row["start"] <= local.strftime("%H:%M") < row["end"] for row in policy["contact_windows"])
                required = end - timedelta(seconds=max(policy["frequency_window_seconds"], policy["min_contact_interval_seconds"]))
                complete = history is not None and history["coverage"] == "complete" and at(history["from_at"]) <= required and at(history["through_at"]) >= end and at(history["available_at"]) <= cutoff
                attempts = []
                if complete:
                    for attempt in history["attempts"]:
                        if required <= at(attempt["attempted_at"]) <= end:
                            if attempt["available_at"] is None or at(attempt["available_at"]) > cutoff:
                                complete = False
                            elif attempt["state"] not in {"cancelled_before_dispatch", "failed_before_dispatch"}:
                                attempts.append(at(attempt["attempted_at"]))
                attempts += provisional.get(subject, [])
                if case["contact_permission"] == "prohibited":
                    reason = "contact_prohibited"
                elif case["contact_permission"] == "unknown":
                    state, reason = "insufficient_evidence", "contact_permission_unknown"
                elif not open_window:
                    reason = "outside_contact_window"
                elif not complete:
                    state, reason = "insufficient_evidence", "contact_history_incomplete"
                elif sum(end - timedelta(seconds=policy["frequency_window_seconds"]) < value <= end for value in attempts) >= policy["max_contacts_per_subject_window"]:
                    reason = "subject_frequency_limit"
                elif attempts and (end - max(attempts)).total_seconds() < policy["min_contact_interval_seconds"]:
                    reason = "subject_minimum_interval"
                else:
                    reason = None
            else:
                reason = None
            if reason is None:
                if used[action["queue_id"]] >= queue["max_batch_actions"]:
                    reason = "queue_batch_capacity"
                elif action["kind"] == "contact" and spent + action["estimated_cost_minor"] > policy["max_estimated_batch_cost_minor"]:
                    reason = "batch_estimated_cost_limit"
                else:
                    used[action["queue_id"]] += 1
                    state, reason = ("eligible_for_approval" if action["kind"] == "contact" else "manual_review"), "declared_constraints_satisfied"
                    if action["kind"] == "contact":
                        spent += action["estimated_cost_minor"]
                        provisional.setdefault(subject, []).append(end)
        results.append({"case_id": case["case_id"], "subject_namespace": case["subject_namespace"], "subject_token": case["subject_token"], "case_input_hash": digest(case), "history_hash": None if history is None else digest(history), "matched_rule_id": matched, "action": envelope, "status": state, "reason": reason})
    body = {"schema_version": "collection.allocation_preview.v1", "as_of": end.isoformat().replace("+00:00", "Z"), "knowledge_cutoff": cutoff.isoformat().replace("+00:00", "Z"), "knowledge_mode": "declared_at_decision" if cutoff == end else "retrospective_declared", "strategy_hash": digest({**spec, "metadata": {}}), "policy_hash": digest(policy), "case_members_hash": digest(sorted(digest(row) for row in request["cases"])), "history_members_hash": digest(sorted(digest(row) for row in request["histories"])), "unit": policy["unit"], "estimated_contact_cost_minor": spent, "queue_batch_allocations": dict(sorted(used.items())), "results": results, "execution_authorized": False, "live_capacity_reserved": False, "assurance": "business_and_source_declarations_not_independently_verified"}
    return {**body, "preview_hash": digest(body)}
