from __future__ import annotations

import pytest

from marvis.packs.strategy.project_context import (
    StrategyProjectContextError,
    build_current_project_snapshot,
    build_effect_observation_ref,
    build_historical_strategy_review,
    build_missing_information_record,
    build_red_flag,
    build_report_field,
    build_source_ref,
    build_strategy_project_context_revision,
    build_strategy_project_context_state,
    canonical_strategy_project_context_revision_json,
    canonical_strategy_project_context_state_json,
    diff_strategy_rules,
    strategy_project_context_revision_from_json,
    strategy_project_context_state_from_json,
    strategy_project_context_state_hash,
    validate_current_project_snapshot,
    validate_historical_strategy_review,
    validate_missing_information_record,
    validate_report_field,
)


_HASH_A = "a" * 64
_HASH_B = "b" * 64


def _unavailable_field():
    return build_report_field(
        value=None,
        availability="unavailable",
        origin="repository",
        source_refs=[],
    )


def _minimal_snapshot():
    unavailable = _unavailable_field()
    return build_current_project_snapshot(
        task_id="task-1",
        as_of="2026-06-30",
        scope=unavailable,
        dataset_refs=[],
        workspace_ref=None,
        champion_strategy_ref=None,
        status_fields={
            "volume": unavailable,
            "approval": unavailable,
            "risk": unavailable,
            "economics": unavailable,
        },
        metric_definition_refs=[],
        metric_observation_refs=[],
        monthly_observation_refs=[],
        segment_observation_refs=[],
        maturity_summary=unavailable,
        user_context_fields=[],
        red_flags=[],
        tool_run_refs=[],
    )


def _minimal_history():
    strategy = build_source_ref(
        kind="strategy", ref_id="strategy-1", content_hash=_HASH_A
    )
    unavailable = _unavailable_field()
    return build_historical_strategy_review(
        task_id="task-1",
        strategy_ref=strategy,
        version=1,
        effective_period=unavailable,
        asset_status=build_report_field(
            value="draft",
            availability="present",
            origin="repository",
            source_refs=[strategy],
        ),
        scope=unavailable,
        traffic_allocation=unavailable,
        change_set=diff_strategy_rules([], []),
        observation_refs_by_effect_stage={
            "estimated": [],
            "backtested": [],
            "oot_validated": [],
            "post_launch_observed": [],
        },
        external_source_refs=[],
        decision_context_fields=[],
        availability="present",
        red_flags=[],
        tool_run_refs=[],
    )


def test_report_field_has_four_explicit_states_and_never_turns_missing_into_zero():
    source = build_source_ref(
        kind="metric_observation",
        ref_id="metric-observation-1",
        content_hash=_HASH_A,
    )

    present_zero = build_report_field(
        value=0,
        availability="present",
        origin="tool_output",
        source_refs=[source],
    )
    assert present_zero["value"] == 0
    assert present_zero["availability"] == "present"

    for availability in ("unavailable", "not_applicable", "not_matured"):
        field = build_report_field(
            value=None,
            availability=availability,
            origin="repository",
            source_refs=[source],
            note="typed absence",
        )
        assert field["value"] is None
        assert field["availability"] == availability

        with pytest.raises(StrategyProjectContextError, match="must have null value"):
            build_report_field(
                value=0,
                availability=availability,
                origin="repository",
                source_refs=[source],
            )

    with pytest.raises(StrategyProjectContextError, match="trusted source_ref"):
        build_report_field(
            value=1,
            availability="present",
            origin="tool_output",
            source_refs=[],
        )

    with pytest.raises(StrategyProjectContextError, match="availability"):
        build_report_field(
            value=None,
            availability="missing",
            origin="repository",
            source_refs=[],
        )

    drifted = {**present_zero, "invented": True}
    with pytest.raises(StrategyProjectContextError, match="unknown: invented"):
        validate_report_field(drifted)


def test_source_ref_identity_cannot_name_two_different_contents():
    first = build_source_ref(
        kind="task_artifact", ref_id="artifact-1", content_hash=_HASH_A
    )
    drifted = build_source_ref(
        kind="task_artifact", ref_id="artifact-1", content_hash=_HASH_B
    )

    with pytest.raises(StrategyProjectContextError, match="identity drift"):
        build_report_field(
            value="evidence",
            availability="present",
            origin="repository",
            source_refs=[first, drifted],
        )


def test_missing_information_identity_is_dependency_stable_across_resolution():
    pending = build_missing_information_record(
        task_id="task-1",
        field_path="current.status_fields.approval",
        reason="No governed approval evidence is available.",
        blocking="report_optional",
        question="Can you provide the historical approval report?",
        status="pending",
        asked_count=1,
        asked_at="2026-07-22T08:00:00+00:00",
        answered_at=None,
        answer_source_ref=None,
        dependency_hash=_HASH_A,
    )
    answer = build_source_ref(
        kind="agent_message",
        ref_id="message-1",
        content_hash=_HASH_B,
    )
    unavailable = build_missing_information_record(
        task_id="task-1",
        field_path="current.status_fields.approval",
        reason="No governed approval evidence is available.",
        blocking="report_optional",
        question="Can you provide the historical approval report?",
        status="unavailable",
        asked_count=1,
        asked_at="2026-07-22T08:00:00+00:00",
        answered_at="2026-07-22T08:05:00+00:00",
        answer_source_ref=answer,
        dependency_hash=_HASH_A,
    )

    assert pending["missing_information_id"] == unavailable["missing_information_id"]
    assert pending["content_hash"] != unavailable["content_hash"]

    with pytest.raises(StrategyProjectContextError, match="answer fields"):
        build_missing_information_record(
            task_id="task-1",
            field_path="current.status_fields.approval",
            reason="No governed approval evidence is available.",
            blocking="report_optional",
            question="Can you provide the historical approval report?",
            status="pending",
            asked_count=1,
            asked_at="2026-07-22T08:00:00+00:00",
            answered_at="2026-07-22T08:05:00+00:00",
            answer_source_ref=answer,
            dependency_hash=_HASH_A,
        )

    drifted = {**unavailable, "asked_count": 2}
    with pytest.raises(StrategyProjectContextError, match="at most once"):
        validate_missing_information_record(drifted)


def test_current_snapshot_is_content_addressed_with_exact_typed_statuses():
    dataset = build_source_ref(
        kind="dataset", ref_id="dataset-1", content_hash=_HASH_A
    )
    observation = build_source_ref(
        kind="metric_observation",
        ref_id="metric-observation-volume",
        content_hash=_HASH_B,
    )
    present_volume = build_report_field(
        value={"metric_observation_refs": [observation]},
        availability="present",
        origin="tool_output",
        source_refs=[observation],
        as_of="2026-06-30",
    )
    unavailable = build_report_field(
        value=None,
        availability="unavailable",
        origin="repository",
        source_refs=[],
    )
    not_matured = build_report_field(
        value=None,
        availability="not_matured",
        origin="tool_output",
        source_refs=[observation],
        note="Performance window has not matured.",
        blocking="validation",
    )
    snapshot = build_current_project_snapshot(
        task_id="task-1",
        as_of="2026-06-30",
        scope=build_report_field(
            value={"product": "cash-loan", "channel": "direct"},
            availability="present",
            origin="user",
            source_refs=[
                build_source_ref(
                    kind="agent_message",
                    ref_id="message-scope",
                    content_hash="c" * 64,
                )
            ],
        ),
        dataset_refs=[dataset],
        workspace_ref=None,
        champion_strategy_ref=None,
        status_fields={
            "volume": present_volume,
            "approval": unavailable,
            "risk": not_matured,
            "economics": build_report_field(
                value=None,
                availability="not_applicable",
                origin="repository",
                source_refs=[],
            ),
        },
        metric_definition_refs=[],
        metric_observation_refs=[observation],
        monthly_observation_refs=[],
        segment_observation_refs=[],
        maturity_summary=not_matured,
        user_context_fields=[],
        red_flags=[
            build_red_flag(
                code="risk_not_matured",
                level="amber",
                message="Risk observations are not yet mature.",
                source_refs=[observation],
            )
        ],
        tool_run_refs=[],
    )

    assert snapshot["status_fields"]["approval"]["value"] is None
    assert snapshot["status_fields"]["economics"]["value"] is None
    assert snapshot["snapshot_id"].startswith("current-project-snapshot-")
    assert len(snapshot["content_hash"]) == 64
    assert validate_current_project_snapshot(snapshot) == snapshot

    tampered = {
        **snapshot,
        "status_fields": {
            **snapshot["status_fields"],
            "approval": {**unavailable, "value": 0},
        },
    }
    with pytest.raises(StrategyProjectContextError, match="must have null value"):
        validate_current_project_snapshot(tampered)

    wrong_status_keys = {
        **snapshot,
        "status_fields": {
            key: value
            for key, value in snapshot["status_fields"].items()
            if key != "economics"
        },
    }
    with pytest.raises(StrategyProjectContextError, match="missing: economics"):
        validate_current_project_snapshot(wrong_status_keys)


def test_rule_diff_uses_stable_rule_id_and_canonical_rule_content():
    previous = [
        {
            "rule_id": "rule-b",
            "priority": 20,
            "condition": {"op": "lt", "field": "score", "value": 500},
            "action": {"type": "reject"},
        },
        {
            "rule_id": "rule-a",
            "priority": 10,
            "condition": {"op": "gte", "field": "score", "value": 700},
            "action": {"type": "approval"},
        },
    ]
    current = [
        {
            "rule_id": "rule-c",
            "priority": 30,
            "condition": {"op": "eq", "field": "segment", "value": "thin"},
            "action": {"type": "review"},
        },
        {
            "rule_id": "rule-a",
            "priority": 5,
            "condition": {"op": "gte", "field": "score", "value": 700},
            "action": {"type": "approval"},
        },
    ]

    change_set = diff_strategy_rules(previous, current)

    assert [item["rule_id"] for item in change_set["added_rule_refs"]] == [
        "rule-c"
    ]
    assert [item["rule_id"] for item in change_set["removed_rule_refs"]] == [
        "rule-b"
    ]
    assert [item["rule_id"] for item in change_set["modified_rule_refs"]] == [
        "rule-a"
    ]
    assert (
        change_set["modified_rule_refs"][0]["before_content_hash"]
        != change_set["modified_rule_refs"][0]["after_content_hash"]
    )
    assert change_set == diff_strategy_rules(list(reversed(previous)), current)

    with pytest.raises(StrategyProjectContextError, match="duplicate rule_id"):
        diff_strategy_rules(previous + [previous[0]], current)


def test_historical_review_keeps_effect_stages_distinct_and_requires_deployment_evidence():
    strategy = build_source_ref(
        kind="strategy", ref_id="strategy-1", content_hash=_HASH_A
    )
    backtest = build_source_ref(
        kind="backtest", ref_id="backtest-1", content_hash=_HASH_B
    )
    monitoring = build_source_ref(
        kind="monitoring_run", ref_id="monitor-1", content_hash="c" * 64
    )
    effect_stages = {
        "estimated": [],
        "backtested": [build_effect_observation_ref(observation_ref=backtest)],
        "oot_validated": [],
        "post_launch_observed": [],
    }
    unavailable = build_report_field(
        value=None,
        availability="unavailable",
        origin="repository",
        source_refs=[],
    )
    review = build_historical_strategy_review(
        task_id="task-1",
        strategy_ref=strategy,
        version=2,
        effective_period=unavailable,
        asset_status=build_report_field(
            value="adopted",
            availability="present",
            origin="repository",
            source_refs=[strategy],
        ),
        scope=unavailable,
        traffic_allocation=unavailable,
        change_set=diff_strategy_rules(
            [],
            [
                {
                    "rule_id": "rule-a",
                    "priority": 10,
                    "condition": {"op": "gte", "field": "score", "value": 700},
                    "action": {"type": "approval"},
                }
            ],
        ),
        observation_refs_by_effect_stage=effect_stages,
        external_source_refs=[],
        decision_context_fields=[],
        availability="present",
        red_flags=[],
        tool_run_refs=[],
    )

    assert review["observation_refs_by_effect_stage"]["backtested"][0][
        "observation_ref"
    ] == backtest
    assert review["observation_refs_by_effect_stage"]["post_launch_observed"] == []
    assert validate_historical_strategy_review(review) == review

    local_monitor_as_post_launch = {
        **effect_stages,
        "post_launch_observed": [
            build_effect_observation_ref(observation_ref=monitoring)
        ],
    }
    with pytest.raises(StrategyProjectContextError, match="deployment_ref"):
        build_historical_strategy_review(
            task_id="task-1",
            strategy_ref=strategy,
            version=2,
            effective_period=unavailable,
            asset_status=unavailable,
            scope=unavailable,
            traffic_allocation=unavailable,
            change_set=diff_strategy_rules([], []),
            observation_refs_by_effect_stage=local_monitor_as_post_launch,
            external_source_refs=[],
            decision_context_fields=[],
            availability="present",
            red_flags=[],
            tool_run_refs=[],
        )

    deployed = build_effect_observation_ref(
        observation_ref=monitoring,
        deployment_ref=build_source_ref(
            kind="deployment", ref_id="deployment-1", content_hash="d" * 64
        ),
        environment_ref=build_source_ref(
            kind="environment", ref_id="production-cn", content_hash="e" * 64
        ),
        effective_period={"start": "2026-06-01", "end": None},
    )
    deployed_review = build_historical_strategy_review(
        task_id="task-1",
        strategy_ref=strategy,
        version=2,
        effective_period=build_report_field(
            value={"start": "2026-06-01", "end": None},
            availability="present",
            origin="repository",
            source_refs=[strategy],
        ),
        asset_status=unavailable,
        scope=unavailable,
        traffic_allocation=unavailable,
        change_set=diff_strategy_rules([], []),
        observation_refs_by_effect_stage={
            **effect_stages,
            "post_launch_observed": [deployed],
        },
        external_source_refs=[],
        decision_context_fields=[],
        availability="present",
        red_flags=[],
        tool_run_refs=[],
    )
    assert deployed_review["observation_refs_by_effect_stage"][
        "post_launch_observed"
    ] == [deployed]


def test_state_and_revision_are_canonical_content_addressed_and_tamper_evident():
    snapshot = _minimal_snapshot()
    history = _minimal_history()
    state = build_strategy_project_context_state(
        task_id="task-1",
        as_of="2026-06-30",
        current_project_snapshot=snapshot,
        historical_strategy_reviews=[history],
        missing_information_records=[],
        source_refs=[],
        red_flags=[],
    )
    revision = build_strategy_project_context_revision(
        state=state,
        revision=1,
        parent_revision_id=None,
        parent_state_hash=None,
        operation_kind="materialize",
    )

    canonical = canonical_strategy_project_context_revision_json(revision)
    canonical_state = canonical_strategy_project_context_state_json(state)
    assert strategy_project_context_state_from_json(canonical_state) == state
    assert strategy_project_context_revision_from_json(canonical) == revision
    assert strategy_project_context_state_hash(state) == state["content_hash"]
    assert state["source_refs"] == [history["strategy_ref"]]
    assert revision["state_hash"] == state["content_hash"]
    assert revision["revision_id"].startswith("strategy-project-context-revision-")
    assert canonical == canonical_strategy_project_context_revision_json(
        strategy_project_context_revision_from_json(canonical.encode("utf-8"))
    )

    drifted_operation = {**revision, "operation_kind": "refresh"}
    with pytest.raises(StrategyProjectContextError, match="operation_hash"):
        canonical_strategy_project_context_revision_json(drifted_operation)

    drifted_state = {
        **state,
        "current_project_snapshot": {
            **snapshot,
            "as_of": "2026-06-29",
        },
    }
    with pytest.raises(StrategyProjectContextError, match="does not match content"):
        build_strategy_project_context_revision(
            state=drifted_state,
            revision=1,
            parent_revision_id=None,
            parent_state_hash=None,
            operation_kind="materialize",
        )

    with pytest.raises(StrategyProjectContextError, match="initial revision"):
        build_strategy_project_context_revision(
            state=state,
            revision=1,
            parent_revision_id="strategy-project-context-revision-" + "f" * 24,
            parent_state_hash="f" * 64,
            operation_kind="materialize",
        )

    duplicate_key = canonical.replace(
        '"revision":1,', '"revision":1,"revision":1,', 1
    )
    with pytest.raises(StrategyProjectContextError, match="duplicate key"):
        strategy_project_context_revision_from_json(duplicate_key)
