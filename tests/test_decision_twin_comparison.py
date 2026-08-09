from dataclasses import replace
from datetime import UTC, datetime
import hashlib

import pytest

from marvis.decision_twin import (
    BindingLineage,
    CounterfactualAssessment,
    DecisionLineage,
    DecisionTwinComparator,
    JointConstraintSet,
    PopulationDenominator,
    ProtectedGroupObservation,
    ReplayedDecision,
    ScenarioKind,
    ScenarioReplay,
    TrustedAdapterIdentity,
)
from marvis.production_governance.router import CreatePromotionRequest


DECISION_AT = datetime(2026, 1, 31, tzinfo=UTC)
MANIFEST_HASH = hashlib.sha256(b"manifest").hexdigest()


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _lineage(record_id: str) -> DecisionLineage:
    return DecisionLineage(
        record_id=record_id,
        manifest_hash=MANIFEST_HASH,
        adapter=TrustedAdapterIdentity("adapter", "v1", _sha("adapter")),
        bindings=tuple(
            BindingLineage(name, "v1", _sha(name))
            for name in (
                "data",
                "dictionary",
                "label",
                "preprocessing",
                "model",
                "strategy",
                "limit_pricing",
                "manual_override",
            )
        ),
        input_sha256=_sha(f"input:{record_id}"),
        output_sha256=_sha(f"output:{record_id}"),
        reason_codes=("policy_result",),
        manual_override_ref=None,
        appeal_ref=None,
        protected_group_source_sha256=_sha(f"group:{record_id}"),
    )


def _decision(
    record_id: str,
    *,
    approved: bool,
    group: str,
    currency: str = "CNY",
    operations_unit: str = "case_minutes",
    attribute: str = "sex",
) -> ReplayedDecision:
    return ReplayedDecision(
        record_id=record_id,
        decision_at=DECISION_AT,
        approved=approved,
        exposure=100.0 if approved else 0.0,
        ead=80.0 if approved else 0.0,
        projected_loss=5.0 if approved else 0.0,
        projected_profit=20.0 if approved else 0.0,
        currency=currency,
        stability=0.9,
        operations_capacity=1.0,
        operations_unit=operations_unit,
        protected_group=ProtectedGroupObservation(
            attribute=attribute,
            group=group,
            governance_ref="fairness-contract-7",
            source_sha256=_sha(f"group:{record_id}"),
        ),
        lineage=_lineage(record_id),
    )


def _scenario(
    name: str,
    kind: ScenarioKind,
    approvals: tuple[bool, bool, bool, bool],
    *,
    groups: tuple[str, str, str, str] = ("group_a", "group_a", "group_b", "group_b"),
    denominator: PopulationDenominator | None = None,
    counterfactual: CounterfactualAssessment | None = None,
) -> ScenarioReplay:
    return ScenarioReplay(
        name=name,
        kind=kind,
        decisions=tuple(
            _decision(f"application-{index}", approved=approved, group=group)
            for index, (approved, group) in enumerate(zip(approvals, groups), start=1)
        ),
        denominator=denominator
        or PopulationDenominator("eligible_applications", 4, "application"),
        observed_execution=kind is ScenarioKind.CHAMPION,
        counterfactual=counterfactual,
    )


def _constraints(**overrides: object) -> JointConstraintSet:
    values: dict[str, object] = {
        "currency": "CNY",
        "operations_unit": "case_minutes",
        "min_approval_rate": 0.4,
        "max_exposure": 400,
        "max_ead": 320,
        "max_projected_loss": 20,
        "min_projected_profit": 20,
        "min_stability": 0.8,
        "max_protected_group_disparity": 0.6,
        "max_operations_capacity": 4,
        "min_protected_group_size": 2,
    }
    values.update(overrides)
    return JointConstraintSet(**values)  # type: ignore[arg-type]


def _comparison():
    return DecisionTwinComparator(_constraints()).compare(
        champion=_scenario(
            "champion",
            ScenarioKind.CHAMPION,
            (True, False, True, False),
        ),
        challenger=_scenario(
            "challenger",
            ScenarioKind.CHALLENGER,
            (True, True, True, False),
        ),
        counterfactual=_scenario(
            "counterfactual",
            ScenarioKind.COUNTERFACTUAL,
            (True, False, False, False),
            counterfactual=CounterfactualAssessment(
                estimand="approval_rate_effect",
                identifiability="bounded",
                lower_bound=-0.1,
                upper_bound=0.2,
                unit="rate",
            ),
        ),
    )


def test_compares_three_scenarios_under_all_joint_constraints() -> None:
    comparison = _comparison()

    assert comparison.champion.metrics.approval_rate == 0.5
    assert comparison.challenger.metrics.exposure == 300.0
    assert comparison.challenger.metrics.ead == 240.0
    assert comparison.challenger.metrics.projected_loss == 15.0
    assert comparison.challenger.metrics.projected_profit == 60.0
    assert comparison.challenger.metrics.stability == 0.9
    assert comparison.challenger.metrics.protected_group_disparity == 0.5
    assert comparison.challenger.metrics.operations_capacity == 4.0
    assert tuple(result.metric for result in comparison.challenger.constraints) == (
        "approval_rate",
        "exposure",
        "ead",
        "projected_loss",
        "projected_profit",
        "stability",
        "protected_group_disparity",
        "operations_capacity",
    )
    assert comparison.challenger.jointly_satisfied is True
    assert comparison.counterfactual.jointly_satisfied is False
    assert comparison.counterfactual.counterfactual.identifiability == "bounded"


def test_counterfactual_outcomes_can_only_be_unidentified_or_bounded() -> None:
    with pytest.raises(ValueError, match="unidentified or bounded"):
        CounterfactualAssessment(
            estimand="bad_rate_effect",
            identifiability="identified",
            unit="rate",
        )
    with pytest.raises(ValueError, match="bounds"):
        CounterfactualAssessment(
            estimand="profit_effect",
            identifiability="bounded",
            unit="CNY",
        )
    with pytest.raises(ValueError, match="requires a counterfactual assessment"):
        _scenario(
            "counterfactual",
            ScenarioKind.COUNTERFACTUAL,
            (True, False, False, False),
        )


def test_comparison_fails_closed_on_denominator_or_unit_drift() -> None:
    with pytest.raises(ValueError, match="denominator count"):
        _scenario(
            "champion",
            ScenarioKind.CHAMPION,
            (True, False, True, False),
            denominator=PopulationDenominator(
                "eligible_applications", 3, "application"
            ),
        )
    bad_unit = _scenario(
        "challenger",
        ScenarioKind.CHALLENGER,
        (True, True, True, False),
        denominator=PopulationDenominator("approved_accounts", 4, "account"),
    )
    with pytest.raises(ValueError, match="denominator drift"):
        DecisionTwinComparator(_constraints()).compare(
            champion=_scenario(
                "champion",
                ScenarioKind.CHAMPION,
                (True, False, True, False),
            ),
            challenger=bad_unit,
            counterfactual=_scenario(
                "counterfactual",
                ScenarioKind.COUNTERFACTUAL,
                (True, False, False, False),
                counterfactual=CounterfactualAssessment(
                    "approval_rate_effect", "unidentified", unit="rate"
                ),
            ),
        )
    mixed_currency = list(
        _scenario(
            "challenger",
            ScenarioKind.CHALLENGER,
            (True, True, True, False),
        ).decisions
    )
    mixed_currency[0] = _decision(
        "application-1", approved=True, group="group_a", currency="USD"
    )
    with pytest.raises(ValueError, match="currency unit drift"):
        ScenarioReplay(
            "challenger",
            ScenarioKind.CHALLENGER,
            tuple(mixed_currency),
            PopulationDenominator("eligible_applications", 4, "application"),
        )


def test_comparison_requires_governed_and_stable_protected_groups() -> None:
    too_small = _scenario(
        "challenger",
        ScenarioKind.CHALLENGER,
        (True, True, True, False),
        groups=("group_a", "group_a", "group_a", "group_b"),
    )
    with pytest.raises(ValueError, match="protected group size"):
        DecisionTwinComparator(_constraints()).compare(
            champion=_scenario(
                "champion",
                ScenarioKind.CHAMPION,
                (True, False, True, False),
                groups=("group_a", "group_a", "group_a", "group_b"),
            ),
            challenger=too_small,
            counterfactual=_scenario(
                "counterfactual",
                ScenarioKind.COUNTERFACTUAL,
                (True, False, False, False),
                groups=("group_a", "group_a", "group_a", "group_b"),
                counterfactual=CounterfactualAssessment(
                    "approval_rate_effect", "unidentified", unit="rate"
                ),
            ),
        )
    drifted_groups = _scenario(
        "challenger",
        ScenarioKind.CHALLENGER,
        (True, True, True, False),
        groups=("group_b", "group_a", "group_b", "group_a"),
    )
    with pytest.raises(ValueError, match="protected group assignment drifted"):
        DecisionTwinComparator(_constraints()).compare(
            champion=_scenario(
                "champion",
                ScenarioKind.CHAMPION,
                (True, False, True, False),
            ),
            challenger=drifted_groups,
            counterfactual=_scenario(
                "counterfactual",
                ScenarioKind.COUNTERFACTUAL,
                (True, False, False, False),
                counterfactual=CounterfactualAssessment(
                    "approval_rate_effect", "unidentified", unit="rate"
                ),
            ),
        )


def test_only_emits_non_authoritative_evidence_for_fsp8() -> None:
    proposal = _comparison().proposal_for(
        scenario_name="challenger",
        strategy_id="strategy-7",
        strategy_version=12,
        evidence_ref="sha256://evidence",
        evidence_sha256=_sha("evidence"),
    )

    assert proposal.automatic_action_permitted is False
    assert proposal.requires_maker_checker is True
    assert proposal.fsp8_binding() == {
        "strategy_id": "strategy-7",
        "strategy_version": 12,
        "manifest_hash": MANIFEST_HASH,
        "evidence_ref": "sha256://evidence",
        "evidence_sha256": _sha("evidence"),
        "comparison_hash": proposal.comparison_hash,
    }
    request_fields = proposal.fsp8_request_fields(
        reason="joint constraints passed; request independent review"
    )
    assert set(request_fields) == {
        "strategy_id",
        "strategy_version",
        "reason",
    }
    assert request_fields["strategy_id"] == "strategy-7"
    assert request_fields["strategy_version"] == 12
    assert f"decision_twin_manifest_sha256={MANIFEST_HASH}" in request_fields["reason"]
    assert _sha("evidence") in request_fields["reason"]
    assert proposal.comparison_hash in request_fields["reason"]
    request = CreatePromotionRequest(
        environment="staging",
        deployment_slot="shadow",
        expires_in_seconds=900,
        **request_fields,
    )
    assert not hasattr(request, "manifest_hash")
    assert not hasattr(proposal, "promote")
    assert not hasattr(proposal, "deploy")


def test_scenario_cannot_bypass_complete_per_record_binding_lineage() -> None:
    valid = _decision("application-1", approved=True, group="group_a")

    with pytest.raises(ValueError, match="complete replay bindings"):
        replace(valid.lineage, bindings=valid.lineage.bindings[:-1])
