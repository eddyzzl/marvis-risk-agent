"""Independent source-claim assessment and bitemporal event/graph arithmetic."""
from datetime import timedelta

from .runtime_archive_reader import _Unsupported
from .runtime_collections_reference import at
from .runtime_contracts import digest


def source_assessment_reference(envelope, recorded_at):
    when, reasons = at(recorded_at), set()
    if at(envelope["available_at"]) > when:
        reasons.add("not_yet_available")
    if at(envelope["expires_at"]) <= when:
        reasons.add("expired")
    kyc, bureau = envelope["kyc"], envelope["bureau"]
    for field, value in kyc.items():
        if value in {"unknown", "no_record"}:
            reasons.add(f"kyc.{field}:{value}")
    if bureau["coverage"] != "complete":
        reasons.add("bureau:" + bureau["coverage"])
    totals = {}
    for account in bureau["accounts"]:
        values = totals.setdefault(account["currency"], {"balance_minor": 0, "past_due_minor": 0})
        for key in values:
            if account[key] is None:
                reasons.add("bureau." + key + ":missing")
                values[key] = None
            elif values[key] is not None:
                values[key] += account[key]
    complete = bureau["coverage"] == "complete"
    return {"status": "expired" if "expired" in reasons else "unknown" if "not_yet_available" in reasons else "available", "kyc_claims": kyc, "bureau_coverage": bureau["coverage"], "account_count": len(bureau["accounts"]) if complete else None, "inquiry_count": len(bureau["inquiries"]) if complete else None, "totals_by_currency": totals if complete else None, "missing_reasons": sorted(reasons), "finding_codes": sorted(f"kyc.{key}:{value}" for key, value in kyc.items() if value in {"mismatch", "invalid", "expired", "fail", "hit"}), "automated_clearance": False}


def _identity(kind, value):
    return None if value is None else (kind, value["namespace"], value["token"])


def _members(rows):
    return digest(sorted([{"source_id": row["source_id"], "event_id": row["claim"]["event_id"], "version": row["claim"]["version"], "content_hash": row["content_hash"]} for row in rows], key=lambda row: (row["source_id"], row["event_id"], row["version"])))


def event_features_reference(snapshot):
    source, contract = snapshot["source"], snapshot["contract"]
    end, cutoff = at(contract["decision_at"]), at(contract["knowledge_cutoff"])
    start = end - timedelta(seconds=contract["window_seconds"])
    types = set(contract["event_types"])

    def relevant(claim, kind):
        if kind == "events":
            return claim["event_type"] in types and start < at(claim["event_at"]) <= end and (contract["include_current_event"] or claim["event_id"] != contract["current_event_id"])
        return bool(set(claim["event_types"]) & types) and at(claim["start_exclusive"]) < end and at(claim["through_inclusive"]) > start

    def select(kind):
        groups, chosen, uncertainty = {}, [], set()
        key = "event_id" if kind == "events" else "coverage_id"
        for row in snapshot["claims"]:
            if row["kind"] == kind and at(row["ingested_at"]) <= cutoff:
                groups.setdefault(row["claim"][key], {})[row["claim"]["version"]] = row
        for versions in groups.values():
            if not any(relevant(row["claim"], kind) for row in versions.values()):
                continue
            known = [row for row in versions.values() if row["claim"]["available_at"] is not None and at(row["claim"]["available_at"]) <= end]
            latest = max(known, key=lambda row: row["claim"]["version"], default=None)
            number = latest["claim"]["version"] if latest else 0
            if number > 100000:
                raise _Unsupported("event lineage version budget")
            if any(row["claim"]["available_at"] is None and row["claim"]["version"] >= number for row in versions.values()):
                uncertainty.add("available_at_unknown")
                continue
            if latest is None:
                continue
            if set(range(1, number + 1)) - set(versions):
                uncertainty.add("version_lineage_incomplete")
                continue
            lineage = [versions[i]["claim"] for i in range(1, number + 1) if versions[i]["claim"]["available_at"] is not None]
            if any(at(a["available_at"]) > at(b["available_at"]) for a, b in zip(lineage, lineage[1:])):
                uncertainty.add("version_availability_conflict")
                continue
            if relevant(latest["claim"], kind):
                chosen.append(latest)
        return sorted(chosen, key=lambda row: (row["claim"][key], row["claim"]["version"])), uncertainty

    rows, uncertainty = select("events")
    coverage, coverage_uncertainty = select("coverage")
    if contract["include_current_event"] and contract["current_event_id"] is not None:
        current = next((row for row in rows if row["claim"]["event_id"] == contract["current_event_id"]), None)
        if current is None:
            uncertainty.add("current_event_not_available_in_window")
        elif current["claim"][contract["focus_kind"]] != contract["focus"]:
            uncertainty.add("current_event_focus_mismatch")

    def coverage_reasons(global_population):
        reasons, candidates = set(coverage_uncertainty), []
        for row in coverage:
            claim = row["claim"]
            if claim["subject"] is not None and (global_population or contract["focus_kind"] != "subject" or claim["subject"] != contract["focus"]):
                continue
            candidates.append(claim)
            if claim["state"] != "complete":
                reasons.add("publisher_coverage_" + claim["state"])
        for kind in types:
            edge = start
            intervals = sorted((at(row["start_exclusive"]), at(row["through_inclusive"])) for row in candidates if row["state"] == "complete" and kind in row["event_types"])
            for left, right in intervals:
                if left > edge:
                    break
                edge = max(edge, right)
            if edge < end:
                reasons.add("publisher_coverage_gap:" + kind)
        return reasons

    def measured(members, value, reasons, unit):
        return {"status": "unknown" if reasons else "measured", "value": None if reasons else value, "unit": unit, "next_action": "required_review" if reasons else "none", "missing_reasons": sorted(reasons), "observed_member_count": len(members), "members_hash": _members(members), "contract_hash": digest(contract), "source_id": contract["source_id"], "source_contract_hash": contract["source_contract_hash"], "coverage_hash": digest(coverage), "automated_clearance": False}

    features = {}
    focus = _identity(contract["focus_kind"], contract["focus"])
    members = [row for row in rows if _identity(contract["focus_kind"], row["claim"][contract["focus_kind"]]) == focus]
    common = uncertainty | coverage_reasons(False)
    if any(row["claim"][contract["focus_kind"]] is None for row in rows):
        common.add("focus_identity_missing")
    for spec in contract["window_features"]:
        reasons, unit = set(common), "count"
        if spec["operation"] == "count":
            value = len(members)
        elif spec["operation"] == "sum":
            unit = source["numeric_fields"][spec["field"]]
            values = [row["claim"]["values"].get(spec["field"]) for row in members]
            if None in values:
                reasons.add("numeric_value_missing:" + spec["field"])
            value = sum(item for item in values if item is not None)
        elif spec["operation"] == "distinct":
            keys = [_identity(spec["field"], row["claim"][spec["field"]]) for row in members]
            if None in keys:
                reasons.add("distinct_identity_missing:" + spec["field"])
            value = len(set(keys) - {None})
        else:
            raise _Unsupported("unknown event aggregate")
        features[spec["name"]] = measured(members, value, reasons, unit)
    for spec in contract["relation_features"]:
        kinds = {contract["focus_kind"], spec["target_kind"]} | ({spec["via_kind"]} if spec["via_kind"] else set())
        reasons, graph = uncertainty | coverage_reasons(True), {}
        for row in rows:
            nodes = []
            for kind in kinds:
                node = _identity(kind, row["claim"][kind])
                if node is None:
                    reasons.add("relation_identity_missing:" + kind)
                else:
                    nodes.append(node)
            for node in nodes:
                graph.setdefault(node, set()).update(other for other in nodes if other != node)
        near = graph.get(focus, set())
        if spec["operation"] == "shared_neighbor_count":
            near = {node for bridge in near if bridge[0] == spec["via_kind"] for node in graph.get(bridge, set())}
        targets = {node for node in near if node[0] == spec["target_kind"] and node != focus}
        features[spec["name"]] = {**measured(rows, len(targets), reasons, "count"), "operation": spec["operation"], "target_kind": spec["target_kind"], "via_kind": spec["via_kind"], "related_members_hash": digest({"source_id": contract["source_id"], "identities": [list(node) for node in sorted(targets)]}), "interpretation": "observed_source_relations_not_fraud_or_identity_proof"}
    missing = any(row["status"] == "unknown" for row in features.values())
    return {"schema_version": "risk-event.feature_result.v1", "contract": contract, "contract_hash": digest(contract), "snapshot_hash": digest(snapshot), "availability_mode": contract["availability_mode"], "source_assurance": "publisher_declared", "platform_ingestion_scope": "frozen_knowledge_cutoff", "historical_platform_visibility": "not_asserted" if contract["availability_mode"] == "retrospective_declared" else "ingested_by_declared_cutoff", "window_start_exclusive": start.isoformat(), "window_end_inclusive": end.isoformat(), "population_members_hash": _members(rows), "population_count": len(rows), "coverage_claims": coverage, "status": "unknown" if missing else "measured", "next_action": "required_review" if missing else "none", "features": features, "automated_clearance": False, "fraud_or_identity_proof": False}
