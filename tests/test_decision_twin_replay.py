from datetime import UTC, datetime, timedelta
import hashlib

import pytest

from marvis.decision_twin import (
    AdapterDecision,
    ObservedField,
    ProtectedGroupObservation,
    ReplayEngine,
    ReplayFacts,
    ReplayManifest,
    ReplayRecord,
    TrustedAdapterIdentity,
    VersionedArtifact,
)


AS_OF = datetime(2026, 1, 31, tzinfo=UTC)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _manifest() -> ReplayManifest:
    bindings = {
        name: VersionedArtifact(f"{name}.v1", _sha(name), AS_OF)
        for name in ReplayManifest.REQUIRED_BINDINGS
    }
    return ReplayManifest(
        as_of=AS_OF,
        issued_at=AS_OF + timedelta(hours=1),
        **bindings,
    )


def _record(
    *, field_visible_at: datetime = AS_OF - timedelta(minutes=1)
) -> ReplayRecord:
    return ReplayRecord(
        facts=ReplayFacts(
            record_id="application-001",
            decision_at=AS_OF,
            source_sha256=_sha("application-001"),
            fields=(
                ObservedField(
                    name="income",
                    value=10_000.0,
                    visible_at=field_visible_at,
                    source_sha256=_sha("income"),
                ),
            ),
        ),
        protected_group=ProtectedGroupObservation(
            attribute="sex",
            group="group_a",
            governance_ref="fairness-contract-7",
            source_sha256=_sha("protected-group"),
        ),
    )


class _Adapter:
    identity = TrustedAdapterIdentity(
        adapter_id="local-credit-engine",
        version="2026.01",
        sha256=_sha("local-credit-engine:2026.01"),
    )

    def __init__(self) -> None:
        self.received_fields: tuple[str, ...] = ()

    def replay(
        self, *, manifest: ReplayManifest, facts: ReplayFacts
    ) -> AdapterDecision:
        self.received_fields = tuple(field.name for field in facts.fields)
        return AdapterDecision(
            manifest_hash=manifest.manifest_hash,
            strategy_approved=False,
            approved=True,
            exposure=8_000,
            ead=6_000,
            projected_loss=300,
            projected_profit=900,
            currency="CNY",
            stability=0.92,
            operations_capacity=12,
            operations_unit="case_minutes",
            reason_codes=("manual_income_review",),
            manual_override_applied=True,
            manual_override_ref="override-17",
            appeal_ref="appeal-4",
        )


def test_replay_uses_only_point_in_time_facts_and_emits_per_record_lineage() -> None:
    manifest = _manifest()
    adapter = _Adapter()

    decision = ReplayEngine(adapter).replay(manifest, _record())

    assert adapter.received_fields == ("income",)
    assert decision.approved is True
    assert decision.protected_group.group == "group_a"
    assert decision.lineage.manifest_hash == manifest.manifest_hash
    assert tuple(item.name for item in decision.lineage.bindings) == tuple(
        ReplayManifest.REQUIRED_BINDINGS
    )
    assert decision.lineage.manual_override_ref == "override-17"
    assert decision.lineage.appeal_ref == "appeal-4"
    assert len(decision.lineage.input_sha256) == 64
    assert len(decision.lineage.output_sha256) == 64


def test_replay_rejects_fields_that_were_not_visible_at_decision_time() -> None:
    adapter = _Adapter()

    with pytest.raises(ValueError, match="visible after decision_at"):
        ReplayEngine(adapter).replay(
            _manifest(),
            _record(field_visible_at=AS_OF + timedelta(microseconds=1)),
        )
    assert adapter.received_fields == ()


def test_replay_never_passes_the_protected_attribute_to_decision_adapter() -> None:
    record = _record()
    contaminated = ReplayRecord(
        facts=ReplayFacts(
            record_id=record.facts.record_id,
            decision_at=record.facts.decision_at,
            source_sha256=record.facts.source_sha256,
            fields=record.facts.fields
            + (
                ObservedField(
                    "sex",
                    "group_a",
                    AS_OF,
                    _sha("sex"),
                ),
            ),
        ),
        protected_group=record.protected_group,
    )

    with pytest.raises(ValueError, match="protected attribute"):
        ReplayEngine(_Adapter()).replay(_manifest(), contaminated)


def test_replay_rejects_unpinned_or_non_deterministic_adapters() -> None:
    with pytest.raises(ValueError, match="execution_mode"):
        TrustedAdapterIdentity(
            adapter_id="llm",
            version="v1",
            sha256=_sha("llm"),
            execution_mode="llm",
        )
    with pytest.raises(ValueError, match="SHA-256"):
        TrustedAdapterIdentity("adapter", "v1", "")


def test_replay_rejects_adapter_output_bound_to_another_manifest() -> None:
    class DriftedAdapter(_Adapter):
        def replay(
            self, *, manifest: ReplayManifest, facts: ReplayFacts
        ) -> AdapterDecision:
            output = super().replay(manifest=manifest, facts=facts)
            return AdapterDecision(
                **{**output.to_dict(), "manifest_hash": _sha("different-manifest")}
            )

    with pytest.raises(ValueError, match="manifest hash drifted"):
        ReplayEngine(DriftedAdapter()).replay(_manifest(), _record())


def test_replay_rejects_adapter_identity_substitution_during_execution() -> None:
    class IdentitySubstitutionAdapter(_Adapter):
        def replay(
            self, *, manifest: ReplayManifest, facts: ReplayFacts
        ) -> AdapterDecision:
            output = super().replay(manifest=manifest, facts=facts)
            self.identity = TrustedAdapterIdentity(
                adapter_id="substituted-credit-engine",
                version="2026.01",
                sha256=_sha("substituted-credit-engine:2026.01"),
            )
            return output

    with pytest.raises(ValueError, match="adapter identity drifted"):
        ReplayEngine(IdentitySubstitutionAdapter()).replay(_manifest(), _record())


def test_manual_override_must_be_explicitly_referenced_and_explainable() -> None:
    common = {
        "manifest_hash": _manifest().manifest_hash,
        "strategy_approved": False,
        "approved": True,
        "exposure": 8_000,
        "ead": 6_000,
        "projected_loss": 300,
        "projected_profit": 900,
        "currency": "CNY",
        "stability": 0.9,
        "operations_capacity": 12,
        "operations_unit": "case_minutes",
        "reason_codes": ("manual_income_review",),
    }

    with pytest.raises(ValueError, match="manual_override_ref"):
        AdapterDecision(**common, manual_override_applied=True)
    with pytest.raises(ValueError, match="requires manual_override_applied"):
        AdapterDecision(
            **common,
            manual_override_applied=False,
            manual_override_ref="override-17",
        )
    with pytest.raises(ValueError, match="decision changed"):
        AdapterDecision(**common, manual_override_applied=False)
