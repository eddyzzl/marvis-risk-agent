from datetime import UTC, datetime, timedelta
import hashlib

import pytest

from marvis.decision_twin import (
    BindingLineage,
    CounterfactualAssessment,
    DecisionLineage,
    MaturityPolicy,
    ObservedOutcome,
    OutcomeReconciler,
    PopulationDenominator,
    ProtectedGroupObservation,
    ReplayedDecision,
    ScenarioKind,
    ScenarioReplay,
    TrustedAdapterIdentity,
)


DECISION_AT = datetime(2026, 1, 1, tzinfo=UTC)
MANIFEST_HASH = hashlib.sha256(b"manifest").hexdigest()


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _decision(record_id: str, approved: bool) -> ReplayedDecision:
    lineage = DecisionLineage(
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
    return ReplayedDecision(
        record_id=record_id,
        decision_at=DECISION_AT,
        approved=approved,
        exposure=100 if approved else 0,
        ead=80 if approved else 0,
        projected_loss=8 if approved else 0,
        projected_profit=20 if approved else 0,
        currency="CNY",
        stability=0.9,
        operations_capacity=1,
        operations_unit="case_minutes",
        protected_group=ProtectedGroupObservation(
            "sex",
            "group_a" if record_id.endswith("1") else "group_b",
            "fairness-contract-7",
            _sha(f"group:{record_id}"),
        ),
        lineage=lineage,
    )


def _champion() -> ScenarioReplay:
    return ScenarioReplay(
        "champion",
        ScenarioKind.CHAMPION,
        (_decision("application-1", True), _decision("application-2", False)),
        PopulationDenominator("eligible_applications", 2, "application"),
        observed_execution=True,
    )


def _policy() -> MaturityPolicy:
    return MaturityPolicy(
        version="bad-90d.v1",
        window=timedelta(days=90),
        sha256=_sha("bad-90d.v1"),
    )


def _outcome(*, observed_at: datetime, currency: str = "CNY") -> ObservedOutcome:
    return ObservedOutcome(
        record_id="application-1",
        observed_at=observed_at,
        actual_loss=10,
        actual_profit=17,
        currency=currency,
        source_ref="ledger://loan-1/matured",
        source_sha256=_sha("ledger-loan-1-matured"),
    )


def test_reconciles_only_observed_approved_decisions_after_maturity() -> None:
    matured_at = DECISION_AT + timedelta(days=90)

    result = OutcomeReconciler(_policy()).reconcile(
        scenario=_champion(),
        outcomes=(_outcome(observed_at=matured_at),),
        reconciled_at=matured_at + timedelta(days=1),
    )

    assert result.manifest_hash == MANIFEST_HASH
    assert result.approved_denominator == 1
    assert result.projected_loss == 8.0
    assert result.actual_loss == 10.0
    assert result.loss_delta == 2.0
    assert result.projected_profit == 20.0
    assert result.actual_profit == 17.0
    assert result.profit_delta == -3.0
    assert result.records[0].matured_at == matured_at


def test_reconciliation_fails_before_the_full_maturity_window() -> None:
    too_early = DECISION_AT + timedelta(days=89, hours=23)

    with pytest.raises(ValueError, match="not mature"):
        OutcomeReconciler(_policy()).reconcile(
            scenario=_champion(),
            outcomes=(_outcome(observed_at=too_early),),
            reconciled_at=too_early,
        )


def test_reconciliation_rejects_future_outcomes_units_and_declined_records() -> None:
    matured_at = DECISION_AT + timedelta(days=90)
    reconciled_at = matured_at + timedelta(days=1)

    with pytest.raises(ValueError, match="future observed outcome"):
        OutcomeReconciler(_policy()).reconcile(
            scenario=_champion(),
            outcomes=(_outcome(observed_at=reconciled_at + timedelta(seconds=1)),),
            reconciled_at=reconciled_at,
        )
    with pytest.raises(ValueError, match="currency unit drift"):
        OutcomeReconciler(_policy()).reconcile(
            scenario=_champion(),
            outcomes=(_outcome(observed_at=matured_at, currency="USD"),),
            reconciled_at=reconciled_at,
        )
    declined = ObservedOutcome(
        record_id="application-2",
        observed_at=matured_at,
        actual_loss=0,
        actual_profit=0,
        currency="CNY",
        source_ref="ledger://declined",
        source_sha256=_sha("declined"),
    )
    with pytest.raises(ValueError, match="approved observed population"):
        OutcomeReconciler(_policy()).reconcile(
            scenario=_champion(),
            outcomes=(_outcome(observed_at=matured_at), declined),
            reconciled_at=reconciled_at,
        )


def test_counterfactual_scenarios_can_never_be_reconciled_as_observed() -> None:
    champion = _champion()
    counterfactual = ScenarioReplay(
        "counterfactual",
        ScenarioKind.COUNTERFACTUAL,
        champion.decisions,
        champion.denominator,
        counterfactual=CounterfactualAssessment(
            "loss_effect",
            "unidentified",
            unit="CNY",
        ),
    )
    matured_at = DECISION_AT + timedelta(days=90)

    with pytest.raises(ValueError, match="observed champion"):
        OutcomeReconciler(_policy()).reconcile(
            scenario=counterfactual,
            outcomes=(_outcome(observed_at=matured_at),),
            reconciled_at=matured_at,
        )
