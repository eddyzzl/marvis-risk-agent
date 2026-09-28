from copy import deepcopy

import pandas as pd
import pytest
from pydantic import ValidationError

from marvis.collection.actions import (
    CollectionAction,
    CollectionCaseInput,
    CollectionPolicy,
    ContactHistory,
)
from marvis.collection.planning import plan_collection_actions
from marvis.packs.strategy.dsl import (
    StrategyAction,
    StrategySpec,
    canonical_strategy_json,
    strategy_spec_hash,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.evaluator import (
    evaluate_strategy_frame,
    evaluate_strategy_row,
)


AT = "2026-09-28T04:00:00Z"
TOKEN = "a" * 64


def policy(**overrides):
    return CollectionPolicy.model_validate(
        {
            "policy_id": "collections",
            "revision": "1",
            "unit": {
                "currency": "CNY",
                "minor_unit_exponent": 2,
                "definition_source": "test declared",
            },
            "valid_from": "2026-09-01T00:00:00Z",
            "valid_until": "2026-10-01T00:00:00Z",
            "timezone": "Asia/Shanghai",
            "contact_windows": [
                {"weekdays": [1, 2, 3, 4, 5, 6, 7], "start": "09:00", "end": "18:00"}
            ],
            "queues": [
                {"queue_id": "q", "channels": ["sms", "phone"], "max_batch_actions": 10}
            ],
            "frequency_window_seconds": 86400,
            "max_contacts_per_subject_window": 2,
            "min_contact_interval_seconds": 3600,
            "max_estimated_batch_cost_minor": 100,
            "basis_artifact_id": "policy-source",
            "basis_artifact_hash": "b" * 64,
            **overrides,
        }
    )


def action(p, **overrides):
    return {
        "type": "collection",
        "value": {
            "kind": "contact",
            "queue_id": "q",
            "channel": "sms",
            "priority": 100,
            "policy_hash": p.content_hash,
            "estimated_cost_minor": 20,
            **overrides,
        },
    }


def spec(p, **overrides):
    return StrategySpec.from_dict(
        {
            "schema_version": "strategy.dsl.v2",
            "strategy_type": "collection",
            "default_action": action(p),
            **overrides,
        }
    )


def case(case_id="c1", token=TOKEN, **overrides):
    return CollectionCaseInput.model_validate(
        {
            "case_id": case_id,
            "subject_namespace": "bank",
            "subject_token": token,
            "contact_permission": "allowed",
            "features": {"dpd": 30},
            "source_artifact_id": "case-source",
            "source_artifact_hash": "c" * 64,
            **overrides,
        }
    )


def history(token=TOKEN, **overrides):
    return ContactHistory.model_validate(
        {
            "subject_namespace": "bank",
            "subject_token": token,
            "from_at": "2026-09-20T00:00:00Z",
            "through_at": AT,
            "available_at": AT,
            "coverage": "complete",
            "attempts": [],
            "source_artifact_id": "history-source",
            "source_artifact_hash": "d" * 64,
            **overrides,
        }
    )


def attempt(when, state="completed", **overrides):
    return {
        "attempt_id": "a1",
        "case_id": "another-case",
        "attempted_at": when,
        "available_at": when,
        "state": state,
        **overrides,
    }


def preview(p=None, cases=None, histories=None, strategy=None, **times):
    p = p or policy()
    return plan_collection_actions(
        strategy or spec(p),
        p,
        [case()] if cases is None else cases,
        [history()] if histories is None else histories,
        **{"as_of": AT, "knowledge_cutoff": AT, **times},
    )


def test_collection_is_versioned_and_cannot_hide_in_v1_or_segmentation():
    p = policy()
    with pytest.raises(StrategyError, match="requires strategy.dsl.v2"):
        spec(p, schema_version="strategy.dsl.v1")
    with pytest.raises(StrategyError, match="not allowed for segmentation"):
        spec(p, strategy_type="segmentation")
    with pytest.raises(StrategyError, match="cannot replace"):
        StrategyAction.from_dict({**action(p), "output_value": "approved"})
    original = StrategySpec(
        strategy_type="approval", default_action=StrategyAction(type="reject")
    )
    assert original.schema_version == "strategy.dsl.v1"
    assert (
        canonical_strategy_json(original)
        == '{"default_action":{"reason_code":null,"stop":true,"type":"reject","value":"reject"},"match_policy":"first_match","metadata":{"lineage":{}},"rules":[],"schema_version":"strategy.dsl.v1","strategy_type":"approval"}'
    )


@pytest.mark.parametrize(
    "patch",
    [
        {"estimated_cost_minor": True},
        {"estimated_cost_minor": 1.2},
        {"estimated_cost_minor": -1},
        {"channel": "email"},
        {"queue_id": None},
        {"priority": 1.5},
        {"kind": "hold"},
        {"policy_hash": "arbitrary"},
        {"kind": "review"},
        {"extra": "provider-payload"},
    ],
)
def test_bad_or_ambiguous_actions_fail_before_execution(patch):
    with pytest.raises(StrategyError, match="invalid typed"):
        StrategyAction.from_dict(action(policy(), **patch))


def test_common_row_and_frame_kernels_emit_same_structured_actions():
    p = policy()
    s = spec(
        p,
        rules=[
            {
                "rule_id": "late",
                "priority": 0,
                "condition": {
                    "op": "compare",
                    "field": "dpd",
                    "operator": ">=",
                    "value": 90,
                },
                "action": action(
                    p, channel="phone", priority=200, estimated_cost_minor=50
                ),
            }
        ],
    )
    data = pd.DataFrame({"dpd": [30, 90, 120]}, index=[5, 8, 5])
    result = evaluate_strategy_frame(data, s)
    assert result.action_values.tolist() == [
        evaluate_strategy_row(row, s).action.value for row in data.to_dict("records")
    ]
    assert result.action_type.tolist() == ["collection"] * 3
    assert strategy_spec_hash(s) == strategy_spec_hash(
        StrategySpec.from_dict(s.to_dict())
    )


def test_preview_binds_evidence_without_authorizing_or_claiming_live_capacity():
    result = preview()
    assert result["results"][0]["status"] == "eligible_for_approval"
    assert result["estimated_contact_cost_minor"] == 20
    assert result["queue_batch_allocations"] == {"q": 1}
    assert result["execution_authorized"] is False
    assert result["live_capacity_reserved"] is False
    assert "not_independently_verified" in result["assurance"]
    assert preview()["preview_hash"] == result["preview_hash"]
    assert (
        preview(cases=[case(source_artifact_hash="f" * 64)])["preview_hash"]
        != result["preview_hash"]
    )


@pytest.mark.parametrize(
    "permission,status,reason",
    [
        ("prohibited", "held", "contact_prohibited"),
        ("unknown", "insufficient_evidence", "contact_permission_unknown"),
    ],
)
def test_contact_permission_fails_closed(permission, status, reason):
    result = preview(cases=[case(contact_permission=permission)])
    assert (result["results"][0]["status"], result["results"][0]["reason"]) == (
        status,
        reason,
    )
    assert result["estimated_contact_cost_minor"] == 0


@pytest.mark.parametrize(
    "histories",
    [
        [],
        [history(coverage="unknown")],
        [history(coverage="partial")],
        [history(from_at="2026-09-28T03:00:00Z")],
        [history(through_at="2026-09-28T03:59:59Z")],
        [history(available_at="2026-09-28T04:00:01Z")],
        [history(attempts=[attempt("2026-09-28T02:00:00Z", available_at=None)])],
        [
            history(
                attempts=[
                    attempt("2026-09-28T02:00:00Z", available_at="2026-09-29T00:00:00Z")
                ]
            )
        ],
    ],
)
def test_incomplete_or_late_history_is_unknown_not_zero(histories):
    row = preview(histories=histories)["results"][0]
    assert row["status"] == "insufficient_evidence"
    assert row["reason"] == "contact_history_incomplete"


@pytest.mark.parametrize(
    "state,held",
    [
        ("completed", True),
        ("dispatched", True),
        ("reserved", True),
        ("unknown_effect", True),
        ("cancelled_before_dispatch", False),
        ("failed_before_dispatch", False),
    ],
)
def test_unknown_effect_consumes_subject_frequency(state, held):
    p = policy(max_contacts_per_subject_window=1)
    result = preview(
        p, histories=[history(attempts=[attempt("2026-09-28T02:00:00Z", state)])]
    )
    assert result["results"][0]["status"] == (
        "held" if held else "eligible_for_approval"
    )


def test_frequency_left_boundary_is_open_and_min_interval_boundary_is_allowed():
    p = policy(max_contacts_per_subject_window=1)
    row = preview(p, histories=[history(attempts=[attempt("2026-09-27T04:00:00Z")])])[
        "results"
    ][0]
    assert row["status"] == "eligible_for_approval"
    row = preview(histories=[history(attempts=[attempt("2026-09-28T03:00:00Z")])])[
        "results"
    ][0]
    assert row["status"] == "eligible_for_approval"
    row = preview(histories=[history(attempts=[attempt("2026-09-28T03:00:01Z")])])[
        "results"
    ][0]
    assert row["reason"] == "subject_minimum_interval"


def test_same_subject_cross_case_provisional_reservations_and_namespace_separation():
    rows = preview(cases=[case("b"), case("a")])["results"]
    assert [(r["case_id"], r["status"]) for r in rows] == [
        ("a", "eligible_for_approval"),
        ("b", "held"),
    ]
    assert rows[1]["reason"] == "subject_minimum_interval"
    cases = [case("a"), case("b", subject_namespace="other-bank")]
    histories = [history(), history(subject_namespace="other-bank")]
    rows = preview(cases=cases, histories=histories)["results"]
    assert [row["status"] for row in rows] == ["eligible_for_approval"] * 2


def test_history_cannot_move_a_known_cases_attempt_to_a_different_subject():
    p = policy(max_contacts_per_subject_window=1, min_contact_interval_seconds=0)
    cases = [case("a"), case("b", "b" * 64)]
    histories = [
        history(attempts=[attempt("2026-09-28T03:30:00Z", case_id="b")]),
        history("b" * 64),
    ]
    with pytest.raises(ValueError, match="history_case_subject_conflict"):
        preview(p, cases, histories)


def test_priority_allocation_is_input_order_independent_and_budget_is_exact():
    p = policy(max_estimated_batch_cost_minor=30)
    s = spec(
        p,
        rules=[
            {
                "rule_id": "priority",
                "priority": 0,
                "condition": {
                    "op": "compare",
                    "field": "dpd",
                    "operator": ">",
                    "value": 60,
                },
                "action": action(p, priority=900),
            }
        ],
    )
    cases = [case("a"), case("z", "e" * 64, features={"dpd": 100})]
    histories = [history(), history("e" * 64)]
    result = preview(p, cases, histories, s)
    assert [row["case_id"] for row in result["results"]] == ["z", "a"]
    assert result["results"][1]["reason"] == "batch_estimated_cost_limit"
    assert result == preview(p, list(reversed(cases)), list(reversed(histories)), s)
    p = policy(queues=[{"queue_id": "q", "channels": ["sms"], "max_batch_actions": 1}])
    rows = preview(p, cases, histories)["results"]
    assert rows[1]["reason"] == "queue_batch_capacity"


def test_policy_hash_queue_and_channel_cannot_drift_even_on_unused_rules():
    p = policy()
    with pytest.raises(ValueError, match="policy_hash_mismatch"):
        preview(p, strategy=spec(policy(revision="2")))
    for overrides, code in [
        ({"queue_id": "missing"}, "queue_unknown"),
        ({"channel": "letter"}, "channel_not_allowed"),
    ]:
        s = spec(
            p,
            rules=[
                {
                    "rule_id": "never",
                    "priority": 0,
                    "condition": {"op": "is_null", "field": "dpd"},
                    "action": action(p, **overrides),
                }
            ],
        )
        with pytest.raises(ValueError, match=code):
            preview(p, strategy=s)


def test_expiry_and_timezone_window_boundaries():
    assert (
        preview(as_of="2026-09-28T10:00:00Z", knowledge_cutoff="2026-09-28T10:00:00Z")[
            "results"
        ][0]["reason"]
        == "outside_contact_window"
    )
    assert (
        preview(policy(valid_until=AT))["results"][0]["reason"]
        == "policy_outside_validity"
    )
    # US fall-back: both physical instants belong to the declared 01:00-02:00 window.
    p = policy(
        timezone="America/New_York",
        valid_until="2026-12-01T00:00:00Z",
        contact_windows=[{"weekdays": [7], "start": "01:00", "end": "02:00"}],
    )
    for when in ("2026-11-01T05:30:00Z", "2026-11-01T06:30:00Z"):
        result = preview(
            p,
            histories=[history(through_at=when, available_at=when)],
            as_of=when,
            knowledge_cutoff=when,
        )
        assert result["results"][0]["status"] == "eligible_for_approval"


def test_split_overnight_policy_can_include_the_final_minute_of_the_day():
    p = policy(
        contact_windows=[
            {"weekdays": [1], "start": "22:00", "end": "24:00"},
            {"weekdays": [2], "start": "00:00", "end": "01:00"},
        ]
    )
    for when in ("2026-09-28T15:59:59Z", "2026-09-28T16:00:00Z"):
        row = preview(
            p,
            histories=[history(through_at=when, available_at=when)],
            as_of=when,
            knowledge_cutoff=when,
        )["results"][0]
        assert row["status"] == "eligible_for_approval"


def test_review_and_hold_do_not_invent_contact_or_contact_cost():
    p = policy()
    review = CollectionAction(kind="review", policy_hash=p.content_hash, queue_id="q")
    hold = CollectionAction(kind="hold", policy_hash=p.content_hash)
    for value, status in ((review, "manual_review"), (hold, "held")):
        result = preview(
            p,
            histories=[],
            cases=[case(contact_permission="prohibited")],
            strategy=spec(
                p, default_action={"type": "collection", "value": value.model_dump()}
            ),
        )
        assert result["results"][0]["status"] == status
        assert result["estimated_contact_cost_minor"] == 0


def test_retrospective_knowledge_is_explicit_and_members_cannot_alias():
    result = preview(knowledge_cutoff="2026-09-29T00:00:00Z")
    assert result["knowledge_mode"] == "retrospective_declared"
    with pytest.raises(ValueError, match="precedes"):
        preview(knowledge_cutoff="2026-09-27T00:00:00Z")
    with pytest.raises(ValueError, match="case_identity_conflict"):
        preview(cases=[case(), case()])
    with pytest.raises(ValueError, match="subject_history_conflict"):
        preview(histories=[history(), history()])
    with pytest.raises(ValueError, match="not_in_batch"):
        preview(histories=[history("f" * 64)])
    with pytest.raises(ValueError, match="budget_exceeded"):
        preview(cases=[])


def test_invalid_policies_and_nested_mutation_are_revalidated():
    with pytest.raises(ValidationError, match="IANA"):
        policy(timezone="invented-zone")
    with pytest.raises(ValidationError, match="overnight"):
        policy(contact_windows=[{"weekdays": [1], "start": "18:00", "end": "09:00"}])
    p = policy()
    s = spec(p)
    changed = deepcopy(s)
    changed.default_action.value["estimated_cost_minor"] = False
    with pytest.raises(StrategyError, match="invalid typed"):
        preview(p, strategy=changed)
