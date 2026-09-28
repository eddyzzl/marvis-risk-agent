"""Deterministic collection allocation preview using the common Strategy DSL.

This pure calculation never reserves live capacity or authorizes an effect.
Imported history and permission remain declarations until a trusted producer
binds them. The executor must recheck policy and history inside its reservation.
"""

from collections import Counter
from datetime import timedelta
from zoneinfo import ZoneInfo

from marvis.collection.actions import (
    CollectionAction,
    CollectionCaseInput,
    CollectionPolicy,
    ContactHistory,
)
from marvis.decision_twin._canonical import content_hash, iso_z, parse_datetime
from marvis.packs.strategy.dsl import parse_strategy_spec, strategy_spec_hash
from marvis.packs.strategy.evaluator import evaluate_strategy_row


def _subject(item):
    return item.subject_namespace, item.subject_token


def _policy_actions(spec, policy):
    queues = {queue.queue_id: queue for queue in policy.queues}
    for raw in (spec.default_action, *(rule.action for rule in spec.rules)):
        action = CollectionAction.model_validate(raw.value)
        if action.policy_hash != policy.content_hash:
            raise ValueError("collection_action_policy_hash_mismatch")
        if action.kind == "hold":
            continue
        queue = queues.get(action.queue_id)
        if queue is None:
            raise ValueError("collection_action_queue_unknown")
        if action.kind == "contact" and action.channel not in queue.channels:
            raise ValueError("collection_action_channel_not_allowed")
    return queues


def _window_open(policy, at):
    local = at.astimezone(ZoneInfo(policy.timezone))
    clock = local.strftime("%H:%M")
    return any(
        local.isoweekday() in window.weekdays and window.start <= clock < window.end
        for window in policy.contact_windows
    )


def _history_state(history, policy, at, cutoff):
    required_from = at - timedelta(
        seconds=max(
            policy.frequency_window_seconds,
            policy.min_contact_interval_seconds,
        )
    )
    if (
        history is None
        or history.coverage != "complete"
        or parse_datetime(history.from_at, "from_at") > required_from
        or parse_datetime(history.through_at, "through_at") < at
        or parse_datetime(history.available_at, "available_at") > cutoff
    ):
        return None
    active = []
    for attempt in history.attempts:
        when = parse_datetime(attempt.attempted_at, "attempted_at")
        if not required_from <= when <= at:
            continue
        if (
            attempt.available_at is None
            or parse_datetime(attempt.available_at, "available_at") > cutoff
        ):
            return None
        if attempt.state not in {"cancelled_before_dispatch", "failed_before_dispatch"}:
            active.append(when)
    return active


def _contact_constraint(case, policy, history_state, provisional, at):
    if case.contact_permission == "prohibited":
        return "held", "contact_prohibited"
    if case.contact_permission == "unknown":
        return "insufficient_evidence", "contact_permission_unknown"
    if not _window_open(policy, at):
        return "held", "outside_contact_window"
    if history_state is None:
        return "insufficient_evidence", "contact_history_incomplete"
    window_start = at - timedelta(seconds=policy.frequency_window_seconds)
    attempts = history_state + provisional
    if (
        sum(window_start < item <= at for item in attempts)
        >= policy.max_contacts_per_subject_window
    ):
        return "held", "subject_frequency_limit"
    # Exact min interval is allowed; a zero interval is an explicit declaration.
    if (
        attempts
        and (at - max(attempts)).total_seconds() < policy.min_contact_interval_seconds
    ):
        return "held", "subject_minimum_interval"
    return None


def plan_collection_actions(
    spec,
    policy: CollectionPolicy,
    cases: list[CollectionCaseInput],
    histories: list[ContactHistory],
    *,
    as_of: str,
    knowledge_cutoff: str,
):
    """Preview at most 10k cases, 10k subjects and 50k attempt records.

    Higher priority wins; ties use case_id. Frequency is shared across cases for
    the same explicit namespace/token and never across equal tokens in different
    namespaces. The money unit is policy-owned; totals are exact minor units.
    """
    spec = parse_strategy_spec(spec).to_dict()
    spec = parse_strategy_spec(spec)  # Detach and validate mutable nested values.
    if spec.strategy_type != "collection":
        raise ValueError("collection_strategy_required")
    policy = CollectionPolicy.model_validate(policy.model_dump())
    if not 0 < len(cases) <= 10000 or len(histories) > 10000:
        raise ValueError("collection_preview_budget_exceeded")
    if sum(len(history.attempts) for history in histories) > 50000:
        raise ValueError("collection_history_budget_exceeded")
    cases = [CollectionCaseInput.model_validate(case.model_dump()) for case in cases]
    histories = [ContactHistory.model_validate(item.model_dump()) for item in histories]
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("collection_case_identity_conflict")
    if len({_subject(history) for history in histories}) != len(histories):
        raise ValueError("collection_subject_history_conflict")
    known_case_subjects = {case.case_id: _subject(case) for case in cases}
    for history in histories:
        if any(
            attempt.case_id in known_case_subjects
            and known_case_subjects[attempt.case_id] != _subject(history)
            for attempt in history.attempts
        ):
            raise ValueError("collection_history_case_subject_conflict")
    if not {_subject(history) for history in histories} <= {
        _subject(case) for case in cases
    }:
        raise ValueError("collection_history_subject_not_in_batch")
    at = parse_datetime(as_of, "as_of")
    cutoff = parse_datetime(knowledge_cutoff, "knowledge_cutoff")
    if cutoff < at:
        raise ValueError("collection_knowledge_cutoff_precedes_decision")
    queues = _policy_actions(spec, policy)
    valid_policy = (
        parse_datetime(policy.valid_from, "valid_from")
        <= at
        < parse_datetime(policy.valid_until, "valid_until")
    )
    history_by_subject = {_subject(history): history for history in histories}
    history_states = {
        subject: _history_state(history, policy, at, cutoff)
        for subject, history in history_by_subject.items()
    }
    evaluated = [(case, evaluate_strategy_row(case.features, spec)) for case in cases]
    evaluated.sort(
        key=lambda item: (-item[1].action.value["priority"], item[0].case_id)
    )
    used_queues, provisional, spent = Counter(), {}, 0
    results = []
    for case, evaluation in evaluated:
        action = CollectionAction.model_validate(evaluation.action.value)
        subject = _subject(case)
        state, reason = "held", "strategy_hold"
        if not valid_policy:
            reason = "policy_outside_validity"
        elif action.kind in {"contact", "review"}:
            constraint = None
            if action.kind == "contact":
                constraint = _contact_constraint(
                    case,
                    policy,
                    history_states.get(subject),
                    provisional.get(subject, []),
                    at,
                )
            if constraint is not None:
                state, reason = constraint
            elif (
                used_queues[action.queue_id]
                >= queues[action.queue_id].max_batch_actions
            ):
                reason = "queue_batch_capacity"
            elif (
                action.kind == "contact"
                and spent + action.estimated_cost_minor
                > policy.max_estimated_batch_cost_minor
            ):
                reason = "batch_estimated_cost_limit"
            else:
                used_queues[action.queue_id] += 1
                state = (
                    "eligible_for_approval"
                    if action.kind == "contact"
                    else "manual_review"
                )
                reason = "declared_constraints_satisfied"
                if action.kind == "contact":
                    spent += action.estimated_cost_minor
                    provisional.setdefault(subject, []).append(at)
        history = history_by_subject.get(subject)
        results.append(
            {
                "case_id": case.case_id,
                "subject_namespace": case.subject_namespace,
                "subject_token": case.subject_token,
                "case_input_hash": case.content_hash,
                "history_hash": None if history is None else history.content_hash,
                "matched_rule_id": evaluation.matched_rule_id,
                "action": evaluation.action.to_dict(),
                "status": state,
                "reason": reason,
            }
        )
    payload = {
        "schema_version": "collection.allocation_preview.v1",
        "as_of": iso_z(at),
        "knowledge_cutoff": iso_z(cutoff),
        "knowledge_mode": "declared_at_decision"
        if cutoff == at
        else "retrospective_declared",
        "strategy_hash": strategy_spec_hash(spec),
        "policy_hash": policy.content_hash,
        "case_members_hash": content_hash(sorted(case.content_hash for case in cases)),
        "history_members_hash": content_hash(
            sorted(item.content_hash for item in histories)
        ),
        "unit": policy.unit.model_dump(),
        "estimated_contact_cost_minor": spent,
        "queue_batch_allocations": dict(sorted(used_queues.items())),
        "results": results,
        "execution_authorized": False,
        "live_capacity_reserved": False,
        "assurance": "business_and_source_declarations_not_independently_verified",
    }
    return {**payload, "preview_hash": content_hash(payload)}
