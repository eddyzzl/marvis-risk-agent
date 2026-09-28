"""Descriptive relations within one explicit source namespace, never identity inference."""

from marvis.risk_context.window_features import (
    coverage_reasons,
    feature_result,
    identity_key,
)


def relation_features(window):
    contract = window.snapshot.contract
    focus = identity_key(contract.focus_kind, contract.focus)
    output = {}
    for spec in contract.relation_features:
        required = {contract.focus_kind, spec.target_kind}
        if spec.via_kind is not None:
            required.add(spec.via_kind)
        reasons = [
            *window.uncertainty,
            *coverage_reasons(window, global_population=True),
        ]
        adjacency = {}
        for row in window.records:
            identities = {
                kind: identity_key(kind, getattr(row.claim, kind)) for kind in required
            }
            for kind, identity in identities.items():
                if identity is None:
                    reasons.append("relation_identity_missing:" + kind)
            keys = [key for key in identities.values() if key is not None]
            for key in keys:
                adjacency.setdefault(key, set()).update(
                    other for other in keys if other != key
                )
        neighbors = adjacency.get(focus, set())
        if spec.operation == "shared_neighbor_count":
            intermediates = [key for key in neighbors if key[0] == spec.via_kind]
            neighbors = {
                key
                for intermediate in intermediates
                for key in adjacency.get(intermediate, set())
            }
        targets = {
            key for key in neighbors if key[0] == spec.target_kind and key != focus
        }
        result = feature_result(
            window, window.records, value=len(targets), reasons=reasons, unit="count"
        )
        result.update(
            {
                "operation": spec.operation,
                "target_kind": spec.target_kind,
                "via_kind": spec.via_kind,
                "related_members_hash": content_hash_identities(
                    contract.source_id, targets
                ),
                "interpretation": "observed_source_relations_not_fraud_or_identity_proof",
            }
        )
        output[spec.name] = result
    return output


def content_hash_identities(source_id, identities):
    from marvis.risk_context.event_contracts import content_hash

    return content_hash(
        {
            "source_id": source_id,
            "identities": [list(identity) for identity in sorted(identities)],
        }
    )
