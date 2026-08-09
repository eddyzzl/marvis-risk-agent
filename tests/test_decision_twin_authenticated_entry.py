from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta
import hashlib

import pytest

from marvis.decision_twin import (
    AdapterDecision,
    AuthenticatedReplayEntryPoint,
    ContentAddressedAuditStore,
    ContentAddressedReplayRepository,
    ReplayFacts,
    ReplayArtifactNotFound,
    ReplayManifest,
    ReplayMaterialRejected,
    TrustedAdapterIdentity,
    TrustedAdapterRegistry,
    VersionedArtifact,
    trusted_adapter_deployment_sha256,
    trusted_adapter_implementation_sha256,
)


AS_OF = datetime(2026, 1, 31, tzinfo=UTC)
ADAPTER_BINARY = b"trusted-credit-adapter-binary-v1"
ADAPTER_PACKAGE = b"trusted-credit-adapter-package-v1"
HELPER_ATTACK_EXECUTED = False


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _trusted_decision_helper(
    manifest: ReplayManifest,
    facts: ReplayFacts,
) -> AdapterDecision:
    del facts
    return AdapterDecision(
        manifest_hash=manifest.manifest_hash,
        strategy_approved=True,
        approved=True,
        exposure=8_000,
        ead=6_000,
        projected_loss=300,
        projected_profit=900,
        currency="CNY",
        stability=0.92,
        operations_capacity=12,
        operations_unit="case_minutes",
        reason_codes=("policy_approved",),
    )


class _TrustedAdapter:
    identity = TrustedAdapterIdentity(
        adapter_id="local-credit-engine",
        version="2026.01",
        sha256=_sha_bytes(ADAPTER_BINARY),
    )

    replay_count = 0
    received_source_sha256 = ""
    received_field_source_sha256 = ""

    def __init__(self) -> None:
        type(self).replay_count = 0
        type(self).received_source_sha256 = ""
        type(self).received_field_source_sha256 = ""

    def replay(
        self, *, manifest: ReplayManifest, facts: ReplayFacts
    ) -> AdapterDecision:
        type(self).replay_count += 1
        type(self).received_source_sha256 = facts.source_sha256
        type(self).received_field_source_sha256 = facts.fields[0].source_sha256
        return _trusted_decision_helper(manifest, facts)


def _put_materials(
    store: ContentAddressedAuditStore,
) -> tuple[str, str, str]:
    bindings: dict[str, VersionedArtifact] = {}
    for name in ReplayManifest.REQUIRED_BINDINGS:
        if name == "data":
            continue
        receipt = store.put(
            "replay_binding",
            {
                "schema_version": "decision_twin.replay_binding.v1",
                "name": name,
                "version": f"{name}.v1",
                "visible_at": _iso(AS_OF),
            },
            idempotency_key=f"binding:{name}:v1",
        )
        bindings[name] = VersionedArtifact(
            version=f"{name}.v1",
            sha256=receipt.artifact_hash,
            visible_at=AS_OF,
        )

    field_receipt = store.put(
        "observed_field_source",
        {
            "schema_version": "decision_twin.observed_field_source.v1",
            "record_id": "application-001",
            "name": "income",
            "value": 10_000.0,
            "visible_at": _iso(AS_OF - timedelta(minutes=1)),
        },
        idempotency_key="field:application-001:income",
    )
    facts_receipt = store.put(
        "replay_facts",
        {
            "schema_version": "decision_twin.replay_facts.v1",
            "record_id": "application-001",
            "decision_at": _iso(AS_OF),
            "field_artifact_ids": [field_receipt.artifact_hash],
        },
        idempotency_key="facts:application-001",
    )
    protected_receipt = store.put(
        "protected_group_source",
        {
            "schema_version": "decision_twin.protected_group_source.v1",
            "record_id": "application-001",
            "attribute": "sex",
            "group": "group_a",
            "governance_ref": "fairness-contract-7",
            "visible_at": _iso(AS_OF - timedelta(minutes=1)),
        },
        idempotency_key="protected-group:application-001",
    )
    record_receipt = store.put(
        "replay_record",
        {
            "schema_version": "decision_twin.replay_record.v1",
            "facts_artifact_id": facts_receipt.artifact_hash,
            "protected_group_artifact_id": protected_receipt.artifact_hash,
        },
        idempotency_key="record:application-001",
    )
    data_receipt = store.put(
        "replay_data_binding",
        {
            "schema_version": "decision_twin.replay_data_binding.v1",
            "name": "data",
            "version": "data.v1",
            "visible_at": _iso(AS_OF),
            "record_artifact_id": record_receipt.artifact_hash,
            "facts_artifact_id": facts_receipt.artifact_hash,
        },
        idempotency_key="binding:data:v1:application-001",
    )
    bindings["data"] = VersionedArtifact(
        version="data.v1",
        sha256=data_receipt.artifact_hash,
        visible_at=AS_OF,
    )
    manifest = ReplayManifest(
        as_of=AS_OF,
        issued_at=AS_OF + timedelta(hours=1),
        **bindings,
    )
    manifest_receipt = store.put(
        "replay_manifest",
        {
            "schema_version": "decision_twin.replay_manifest_material.v1",
            "manifest": manifest.to_dict(),
        },
        idempotency_key="manifest:application-001",
    )

    binary_receipt = store.put(
        "trusted_adapter_binary",
        {
            "schema_version": "decision_twin.trusted_adapter_binary.v1",
            "encoding": "base64",
            "content": base64.b64encode(ADAPTER_BINARY).decode("ascii"),
        },
        idempotency_key="adapter-binary:local-credit-engine:2026.01",
    )
    package_receipt = store.put(
        "trusted_adapter_package",
        {
            "schema_version": "decision_twin.trusted_adapter_package.v1",
            "encoding": "base64",
            "content": base64.b64encode(ADAPTER_PACKAGE).decode("ascii"),
        },
        idempotency_key="adapter-package:local-credit-engine:2026.01",
    )
    adapter_receipt = store.put(
        "trusted_adapter_material",
        {
            "schema_version": "decision_twin.trusted_adapter_material.v2",
            "identity": _TrustedAdapter.identity.to_dict(),
            "binary_artifact_id": binary_receipt.artifact_hash,
            "binary_sha256": _sha_bytes(ADAPTER_BINARY),
            "package_artifact_id": package_receipt.artifact_hash,
            "package_sha256": _sha_bytes(ADAPTER_PACKAGE),
            "implementation_sha256": trusted_adapter_implementation_sha256(
                _TrustedAdapter
            ),
            "deployment_sha256": trusted_adapter_deployment_sha256(
                _TrustedAdapter.identity,
                binary_sha256=_sha_bytes(ADAPTER_BINARY),
                package_sha256=_sha_bytes(ADAPTER_PACKAGE),
                implementation_sha256=trusted_adapter_implementation_sha256(
                    _TrustedAdapter
                ),
            ),
        },
        idempotency_key="adapter-material:local-credit-engine:2026.01",
    )
    return (
        manifest_receipt.artifact_hash,
        record_receipt.artifact_hash,
        adapter_receipt.artifact_hash,
    )


def test_authenticated_entry_loads_all_replay_inputs_from_immutable_artifact_ids(
    tmp_path,
) -> None:
    store = ContentAddressedAuditStore(tmp_path / "decision-twin")
    manifest_id, record_id, adapter_id = _put_materials(store)
    entry = AuthenticatedReplayEntryPoint(
        ContentAddressedReplayRepository(store),
        TrustedAdapterRegistry(((_TrustedAdapter, ADAPTER_BINARY, ADAPTER_PACKAGE),)),
    )

    decision = entry.replay(
        manifest_artifact_id=manifest_id,
        record_artifact_id=record_id,
        adapter_artifact_id=adapter_id,
    )

    assert decision.record_id == "application-001"
    assert decision.approved is True
    record_material = store.get(record_id)
    facts_id = record_material["facts_artifact_id"]
    facts_material = store.get(facts_id)
    assert _TrustedAdapter.received_source_sha256 == facts_id
    assert (
        _TrustedAdapter.received_field_source_sha256
        == facts_material["field_artifact_ids"][0]
    )
    assert decision.lineage.adapter.sha256 == _sha_bytes(ADAPTER_BINARY)
    assert _TrustedAdapter.replay_count == 1


def test_authenticated_entry_rejects_record_from_another_manifest_data_binding(
    tmp_path,
) -> None:
    store = ContentAddressedAuditStore(tmp_path / "decision-twin")
    manifest_id, _, adapter_id = _put_materials(store)
    other_field = store.put(
        "observed_field_source",
        {
            "schema_version": "decision_twin.observed_field_source.v1",
            "record_id": "application-002",
            "name": "income",
            "value": 20_000.0,
            "visible_at": _iso(AS_OF - timedelta(minutes=1)),
        },
        idempotency_key="field:application-002:income",
    )
    other_facts = store.put(
        "replay_facts",
        {
            "schema_version": "decision_twin.replay_facts.v1",
            "record_id": "application-002",
            "decision_at": _iso(AS_OF),
            "field_artifact_ids": [other_field.artifact_hash],
        },
        idempotency_key="facts:application-002",
    )
    other_protected = store.put(
        "protected_group_source",
        {
            "schema_version": "decision_twin.protected_group_source.v1",
            "record_id": "application-002",
            "attribute": "sex",
            "group": "group_b",
            "governance_ref": "fairness-contract-7",
            "visible_at": _iso(AS_OF - timedelta(minutes=1)),
        },
        idempotency_key="protected-group:application-002",
    )
    other_record = store.put(
        "replay_record",
        {
            "schema_version": "decision_twin.replay_record.v1",
            "facts_artifact_id": other_facts.artifact_hash,
            "protected_group_artifact_id": other_protected.artifact_hash,
        },
        idempotency_key="record:application-002",
    )
    entry = AuthenticatedReplayEntryPoint(
        ContentAddressedReplayRepository(store),
        TrustedAdapterRegistry(((_TrustedAdapter, ADAPTER_BINARY, ADAPTER_PACKAGE),)),
    )

    with pytest.raises(ReplayMaterialRejected, match="manifest data binding"):
        entry.replay(
            manifest_artifact_id=manifest_id,
            record_artifact_id=other_record.artifact_hash,
            adapter_artifact_id=adapter_id,
        )
    assert _TrustedAdapter.replay_count == 0


def test_authenticated_entry_rejects_facts_not_pinned_by_manifest_data_binding(
    tmp_path,
) -> None:
    store = ContentAddressedAuditStore(tmp_path / "decision-twin")
    manifest_id, record_id, adapter_id = _put_materials(store)
    original_manifest = ReplayManifest.from_dict(store.get(manifest_id)["manifest"])
    alternate_field = store.put(
        "observed_field_source",
        {
            "schema_version": "decision_twin.observed_field_source.v1",
            "record_id": "application-001",
            "name": "income",
            "value": 99_999.0,
            "visible_at": _iso(AS_OF - timedelta(minutes=1)),
        },
        idempotency_key="field:application-001:income:alternate",
    )
    alternate_facts = store.put(
        "replay_facts",
        {
            "schema_version": "decision_twin.replay_facts.v1",
            "record_id": "application-001",
            "decision_at": _iso(AS_OF),
            "field_artifact_ids": [alternate_field.artifact_hash],
        },
        idempotency_key="facts:application-001:alternate",
    )
    mismatched_data = store.put(
        "replay_data_binding",
        {
            "schema_version": "decision_twin.replay_data_binding.v1",
            "name": "data",
            "version": "data.alternate",
            "visible_at": _iso(AS_OF),
            "record_artifact_id": record_id,
            "facts_artifact_id": alternate_facts.artifact_hash,
        },
        idempotency_key="binding:data:alternate",
    )
    bindings = dict(original_manifest.bindings)
    bindings["data"] = VersionedArtifact(
        version="data.alternate",
        sha256=mismatched_data.artifact_hash,
        visible_at=AS_OF,
    )
    mismatched_manifest = ReplayManifest(
        as_of=original_manifest.as_of,
        issued_at=original_manifest.issued_at,
        **bindings,
    )
    mismatched_manifest_receipt = store.put(
        "replay_manifest",
        {
            "schema_version": "decision_twin.replay_manifest_material.v1",
            "manifest": mismatched_manifest.to_dict(),
        },
        idempotency_key="manifest:application-001:alternate-facts",
    )
    entry = AuthenticatedReplayEntryPoint(
        ContentAddressedReplayRepository(store),
        TrustedAdapterRegistry(((_TrustedAdapter, ADAPTER_BINARY, ADAPTER_PACKAGE),)),
    )

    with pytest.raises(ReplayMaterialRejected, match="facts artifact"):
        entry.replay(
            manifest_artifact_id=mismatched_manifest_receipt.artifact_hash,
            record_artifact_id=record_id,
            adapter_artifact_id=adapter_id,
        )
    assert _TrustedAdapter.replay_count == 0


def test_authenticated_entry_allows_counterfactual_manifests_over_same_bound_record(
    tmp_path,
) -> None:
    store = ContentAddressedAuditStore(tmp_path / "decision-twin")
    baseline_manifest_id, record_id, adapter_id = _put_materials(store)
    baseline_manifest = ReplayManifest.from_dict(
        store.get(baseline_manifest_id)["manifest"]
    )
    candidate_strategy = store.put(
        "replay_binding",
        {
            "schema_version": "decision_twin.replay_binding.v1",
            "name": "strategy",
            "version": "strategy.v2",
            "visible_at": _iso(AS_OF),
        },
        idempotency_key="binding:strategy:v2",
    )
    candidate_bindings = dict(baseline_manifest.bindings)
    candidate_bindings["strategy"] = VersionedArtifact(
        version="strategy.v2",
        sha256=candidate_strategy.artifact_hash,
        visible_at=AS_OF,
    )
    candidate_manifest = ReplayManifest(
        as_of=baseline_manifest.as_of,
        issued_at=baseline_manifest.issued_at,
        **candidate_bindings,
    )
    candidate_manifest_receipt = store.put(
        "replay_manifest",
        {
            "schema_version": "decision_twin.replay_manifest_material.v1",
            "manifest": candidate_manifest.to_dict(),
        },
        idempotency_key="manifest:application-001:candidate-strategy",
    )
    entry = AuthenticatedReplayEntryPoint(
        ContentAddressedReplayRepository(store),
        TrustedAdapterRegistry(((_TrustedAdapter, ADAPTER_BINARY, ADAPTER_PACKAGE),)),
    )

    baseline = entry.replay(
        manifest_artifact_id=baseline_manifest_id,
        record_artifact_id=record_id,
        adapter_artifact_id=adapter_id,
    )
    candidate = entry.replay(
        manifest_artifact_id=candidate_manifest_receipt.artifact_hash,
        record_artifact_id=record_id,
        adapter_artifact_id=adapter_id,
    )

    assert baseline.record_id == candidate.record_id == "application-001"
    assert baseline.lineage.manifest_hash != candidate.lineage.manifest_hash
    assert baseline.lineage.bindings[0].sha256 == candidate.lineage.bindings[0].sha256
    assert _TrustedAdapter.replay_count == 2


def test_authenticated_entry_rejects_forged_fact_source_hash_before_replay(
    tmp_path,
) -> None:
    store = ContentAddressedAuditStore(tmp_path / "decision-twin")
    manifest_id, original_record_id, adapter_id = _put_materials(store)
    forged_facts = store.put(
        "replay_facts",
        {
            "schema_version": "decision_twin.replay_facts.v1",
            "record_id": "application-001",
            "decision_at": _iso(AS_OF),
            "field_artifact_ids": ["f" * 64],
        },
        idempotency_key="facts:application-001:forged",
    )
    forged_record = store.put(
        "replay_record",
        {
            "schema_version": "decision_twin.replay_record.v1",
            "facts_artifact_id": forged_facts.artifact_hash,
            "protected_group_artifact_id": store.get(original_record_id)[
                "protected_group_artifact_id"
            ],
        },
        idempotency_key="record:application-001:forged",
    )
    original_manifest = ReplayManifest.from_dict(store.get(manifest_id)["manifest"])
    forged_data = store.put(
        "replay_data_binding",
        {
            "schema_version": "decision_twin.replay_data_binding.v1",
            "name": "data",
            "version": "data.forged",
            "visible_at": _iso(AS_OF),
            "record_artifact_id": forged_record.artifact_hash,
            "facts_artifact_id": forged_facts.artifact_hash,
        },
        idempotency_key="binding:data:forged",
    )
    bindings = dict(original_manifest.bindings)
    bindings["data"] = VersionedArtifact(
        version="data.forged",
        sha256=forged_data.artifact_hash,
        visible_at=AS_OF,
    )
    forged_manifest = ReplayManifest(
        as_of=original_manifest.as_of,
        issued_at=original_manifest.issued_at,
        **bindings,
    )
    forged_manifest_receipt = store.put(
        "replay_manifest",
        {
            "schema_version": "decision_twin.replay_manifest_material.v1",
            "manifest": forged_manifest.to_dict(),
        },
        idempotency_key="manifest:application-001:forged",
    )
    entry = AuthenticatedReplayEntryPoint(
        ContentAddressedReplayRepository(store),
        TrustedAdapterRegistry(((_TrustedAdapter, ADAPTER_BINARY, ADAPTER_PACKAGE),)),
    )

    with pytest.raises(ReplayArtifactNotFound, match="observed_field_source"):
        entry.replay(
            manifest_artifact_id=forged_manifest_receipt.artifact_hash,
            record_artifact_id=forged_record.artifact_hash,
            adapter_artifact_id=adapter_id,
        )
    assert _TrustedAdapter.replay_count == 0


def test_trusted_registry_rejects_adapter_object_impersonating_trusted_identity() -> None:
    class _ImpersonatingAdapter(_TrustedAdapter):
        def replay(
            self, *, manifest: ReplayManifest, facts: ReplayFacts
        ) -> AdapterDecision:
            return AdapterDecision(
                manifest_hash=manifest.manifest_hash,
                strategy_approved=True,
                approved=False,
                exposure=0,
                ead=0,
                projected_loss=0,
                projected_profit=0,
                currency="CNY",
                stability=1,
                operations_capacity=0,
                operations_unit="case_minutes",
                reason_codes=("impersonated_adapter",),
            )

    with pytest.raises(ValueError, match="trusted adapter type"):
        TrustedAdapterRegistry(
            ((_ImpersonatingAdapter(), ADAPTER_BINARY, ADAPTER_PACKAGE),)
        )


def test_authenticated_entry_rejects_registered_adapter_implementation_replacement(
    tmp_path,
) -> None:
    store = ContentAddressedAuditStore(tmp_path / "decision-twin")
    manifest_id, record_id, adapter_id = _put_materials(store)
    entry = AuthenticatedReplayEntryPoint(
        ContentAddressedReplayRepository(store),
        TrustedAdapterRegistry(((_TrustedAdapter, ADAPTER_BINARY, ADAPTER_PACKAGE),)),
    )
    trusted_replay = _TrustedAdapter.replay

    def impersonated_replay(
        self, *, manifest: ReplayManifest, facts: ReplayFacts
    ) -> AdapterDecision:
        del self, facts
        return AdapterDecision(
            manifest_hash=manifest.manifest_hash,
            strategy_approved=True,
            approved=False,
            exposure=0,
            ead=0,
            projected_loss=0,
            projected_profit=0,
            currency="CNY",
            stability=1,
            operations_capacity=0,
            operations_unit="case_minutes",
            reason_codes=("replaced_implementation",),
        )

    _TrustedAdapter.replay = impersonated_replay  # type: ignore[method-assign]
    try:
        with pytest.raises(ReplayMaterialRejected, match="implementation drifted"):
            entry.replay(
                manifest_artifact_id=manifest_id,
                record_artifact_id=record_id,
                adapter_artifact_id=adapter_id,
            )
    finally:
        _TrustedAdapter.replay = trusted_replay  # type: ignore[method-assign]
    assert _TrustedAdapter.replay_count == 0


def test_authenticated_entry_rejects_in_place_adapter_code_mutation(
    tmp_path,
) -> None:
    store = ContentAddressedAuditStore(tmp_path / "decision-twin")
    manifest_id, record_id, adapter_id = _put_materials(store)
    entry = AuthenticatedReplayEntryPoint(
        ContentAddressedReplayRepository(store),
        TrustedAdapterRegistry(((_TrustedAdapter, ADAPTER_BINARY, ADAPTER_PACKAGE),)),
    )
    trusted_code = _TrustedAdapter.replay.__code__

    def impersonated_replay(
        self, *, manifest: ReplayManifest, facts: ReplayFacts
    ) -> AdapterDecision:
        del self, facts
        return AdapterDecision(
            manifest_hash=manifest.manifest_hash,
            strategy_approved=True,
            approved=False,
            exposure=0,
            ead=0,
            projected_loss=0,
            projected_profit=0,
            currency="CNY",
            stability=1,
            operations_capacity=0,
            operations_unit="case_minutes",
            reason_codes=("mutated_code",),
        )

    _TrustedAdapter.replay.__code__ = impersonated_replay.__code__
    try:
        with pytest.raises(ReplayMaterialRejected, match="implementation digest"):
            entry.replay(
                manifest_artifact_id=manifest_id,
                record_artifact_id=record_id,
                adapter_artifact_id=adapter_id,
            )
    finally:
        _TrustedAdapter.replay.__code__ = trusted_code
    assert _TrustedAdapter.replay_count == 0


def test_authenticated_entry_rejects_in_place_helper_code_mutation_before_execution(
    tmp_path,
) -> None:
    global HELPER_ATTACK_EXECUTED

    store = ContentAddressedAuditStore(tmp_path / "decision-twin")
    manifest_id, record_id, adapter_id = _put_materials(store)
    entry = AuthenticatedReplayEntryPoint(
        ContentAddressedReplayRepository(store),
        TrustedAdapterRegistry(((_TrustedAdapter, ADAPTER_BINARY, ADAPTER_PACKAGE),)),
    )
    trusted_code = _trusted_decision_helper.__code__

    def impersonated_helper(
        manifest: ReplayManifest,
        facts: ReplayFacts,
    ) -> AdapterDecision:
        global HELPER_ATTACK_EXECUTED
        del facts
        HELPER_ATTACK_EXECUTED = True
        return AdapterDecision(
            manifest_hash=manifest.manifest_hash,
            strategy_approved=True,
            approved=True,
            exposure=999_999,
            ead=999_999,
            projected_loss=999_999,
            projected_profit=-999_999,
            currency="CNY",
            stability=0,
            operations_capacity=0,
            operations_unit="case_minutes",
            reason_codes=("mutated_global_helper_executed",),
        )

    HELPER_ATTACK_EXECUTED = False
    _trusted_decision_helper.__code__ = impersonated_helper.__code__
    try:
        with pytest.raises(ReplayMaterialRejected, match="implementation digest"):
            entry.replay(
                manifest_artifact_id=manifest_id,
                record_artifact_id=record_id,
                adapter_artifact_id=adapter_id,
            )
    finally:
        _trusted_decision_helper.__code__ = trusted_code
    assert HELPER_ATTACK_EXECUTED is False
    assert _TrustedAdapter.replay_count == 0


def test_authenticated_entry_rejects_adapter_binary_or_package_substitution(
    tmp_path,
) -> None:
    store = ContentAddressedAuditStore(tmp_path / "decision-twin")
    manifest_id, record_id, _ = _put_materials(store)
    substituted_binary = b"substituted-credit-adapter-binary"
    substituted_package = b"substituted-credit-adapter-package"
    binary_receipt = store.put(
        "trusted_adapter_binary",
        {
            "schema_version": "decision_twin.trusted_adapter_binary.v1",
            "encoding": "base64",
            "content": base64.b64encode(substituted_binary).decode("ascii"),
        },
        idempotency_key="adapter-binary:substitution",
    )
    package_receipt = store.put(
        "trusted_adapter_package",
        {
            "schema_version": "decision_twin.trusted_adapter_package.v1",
            "encoding": "base64",
            "content": base64.b64encode(substituted_package).decode("ascii"),
        },
        idempotency_key="adapter-package:substitution",
    )
    substituted_material = store.put(
        "trusted_adapter_material",
        {
            "schema_version": "decision_twin.trusted_adapter_material.v2",
            "identity": {
                **_TrustedAdapter.identity.to_dict(),
                "sha256": _sha_bytes(substituted_binary),
            },
            "binary_artifact_id": binary_receipt.artifact_hash,
            "binary_sha256": _sha_bytes(substituted_binary),
            "package_artifact_id": package_receipt.artifact_hash,
            "package_sha256": _sha_bytes(substituted_package),
            "implementation_sha256": trusted_adapter_implementation_sha256(
                _TrustedAdapter
            ),
            "deployment_sha256": trusted_adapter_deployment_sha256(
                TrustedAdapterIdentity(
                    **{
                        **_TrustedAdapter.identity.to_dict(),
                        "sha256": _sha_bytes(substituted_binary),
                    }
                ),
                binary_sha256=_sha_bytes(substituted_binary),
                package_sha256=_sha_bytes(substituted_package),
                implementation_sha256=trusted_adapter_implementation_sha256(
                    _TrustedAdapter
                ),
            ),
        },
        idempotency_key="adapter-material:substitution",
    )
    entry = AuthenticatedReplayEntryPoint(
        ContentAddressedReplayRepository(store),
        TrustedAdapterRegistry(((_TrustedAdapter, ADAPTER_BINARY, ADAPTER_PACKAGE),)),
    )

    with pytest.raises(ReplayMaterialRejected, match="trusted allowlist"):
        entry.replay(
            manifest_artifact_id=manifest_id,
            record_artifact_id=record_id,
            adapter_artifact_id=substituted_material.artifact_hash,
        )
    assert _TrustedAdapter.replay_count == 0


def test_authenticated_entry_rejects_missing_top_level_artifact_before_replay(
    tmp_path,
) -> None:
    store = ContentAddressedAuditStore(tmp_path / "decision-twin")
    _, record_id, adapter_id = _put_materials(store)
    entry = AuthenticatedReplayEntryPoint(
        ContentAddressedReplayRepository(store),
        TrustedAdapterRegistry(((_TrustedAdapter, ADAPTER_BINARY, ADAPTER_PACKAGE),)),
    )

    with pytest.raises(ReplayArtifactNotFound, match="replay_manifest"):
        entry.replay(
            manifest_artifact_id="0" * 64,
            record_artifact_id=record_id,
            adapter_artifact_id=adapter_id,
        )
    assert _TrustedAdapter.replay_count == 0


def test_production_entry_rejects_caller_constructed_manifest_dataclass(
    tmp_path,
) -> None:
    store = ContentAddressedAuditStore(tmp_path / "decision-twin")
    manifest_id, record_id, adapter_id = _put_materials(store)
    caller_manifest = ReplayManifest.from_dict(store.get(manifest_id)["manifest"])
    entry = AuthenticatedReplayEntryPoint(
        ContentAddressedReplayRepository(store),
        TrustedAdapterRegistry(((_TrustedAdapter, ADAPTER_BINARY, ADAPTER_PACKAGE),)),
    )

    with pytest.raises(ReplayMaterialRejected, match="artifact id"):
        entry.replay(
            manifest_artifact_id=caller_manifest,  # type: ignore[arg-type]
            record_artifact_id=record_id,
            adapter_artifact_id=adapter_id,
        )
    assert _TrustedAdapter.replay_count == 0
