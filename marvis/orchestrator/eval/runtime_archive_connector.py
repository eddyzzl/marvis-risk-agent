"""Independent registered-history connector and event/relationship archive checks."""
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from marvis.risk_context.event_contracts import EventSnapshot
from marvis.risk_context.event_service import EventRequest
from marvis.risk_context.source_contracts import ProviderResult, SourceEnvelope, SourceQuery

from .runtime_archive_datasets import _dataset, _parquet, _source_input
from .runtime_archive_domain import equal
from .runtime_archive_reader import _Unsupported, _file, _json
from .runtime_collections_reference import at
from .runtime_connector_reference import event_features_reference, source_assessment_reference
from .runtime_contracts import digest
from .runtime_extensions_archive import ExtensionArchive, FrozenExtensionBinding, Hash, Identity, declared_requests, measured_values
from .runtime_native_identity import declaration_matches


class FrozenSourceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    request: SourceQuery
    record_sha256: Hash
    artifact_id: Identity
    artifact_sha256: Hash


class FrozenEventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    request: EventRequest
    intent_record_sha256: Hash
    decision_record_sha256: Hash
    artifact_id: Identity
    artifact_sha256: Hash


class FrozenConnectorBinding(FrozenExtensionBinding):
    source_requests: list[FrozenSourceRequest] = Field(default_factory=list, max_length=100)
    event_requests: list[FrozenEventRequest] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def declared(self):
        if self.case.family != "connector_antifraud" or not (self.source_requests or self.event_requests):
            raise ValueError("declared connector/event requests required")
        for kind, items in (("prepare_risk_source_query", self.source_requests), ("prepare_risk_event_features", self.event_requests)):
            if any(not declaration_matches(self.case, kind, item.request) for item in items):
                raise ValueError("frozen query must match declared business input")
        if any(item.request.historical is None for item in self.source_requests):
            raise ValueError("reference HTTP originals are excluded from current custody; do not substitute an unverified provider summary")
        return self


def _registered_source(ctx, dataset_id, sha):
    row, frame = _dataset(ctx.conn, ctx.archive, ctx.binding.task_id, dataset_id, sha)
    relative = Path("workspace/datasets") / ctx.binding.task_id / ".source-identities" / (dataset_id + ".json")
    if not (ctx.archive / relative).exists():
        # register_existing is a genuine product carrier without an upload
        # receipt. Only byte-identical, explicitly frozen original parquet is
        # admissible here; never invent an upload identity for derived data.
        matching = [(index, material) for index, material in enumerate(ctx.binding.case.materials) if material.sha256 == sha and Path(material.path).suffix == ".parquet"]
        if len(matching) != 1:
            raise _Unsupported("registered parquet requires a frozen byte-identical original material or authenticated upload identity")
        index, material = matching[0]
        original = _file(ctx.archive, f"inputs/{index:03d}.parquet")
        equal(digest(original.read_bytes()), material.sha256)
        import pandas as pd
        pd.testing.assert_frame_equal(_parquet(original), frame, check_dtype=False, check_exact=True)
        return frame
    identity = _json(ctx.archive, relative)
    matching = [index for index, material in enumerate(ctx.binding.case.materials) if material.sha256 == identity.get("sha256")]
    if len(matching) != 1:
        raise ValueError("registered source must have one exact frozen public material")
    _source_input(ctx.archive, ctx.manifest, ctx.binding, row, frame, ctx.archive.parent / "scratch", material_index=matching[0])
    return frame


def _authorization_basis(ctx, grant, expected_hash=None):
    row = ctx.conn.execute("SELECT * FROM task_artifacts WHERE id=? AND task_id=?", (grant["basis_artifact_id"], ctx.binding.task_id)).fetchone()
    if row is None or (expected_hash is not None and row["content_hash"] != expected_hash):
        raise ValueError("source authorization basis byte binding differs")
    envelope = ctx.artifact(grant["basis_artifact_id"], row["content_hash"], kind="source_authorization_basis", origin="risk_context.source_query.v1")
    body = envelope["body"]
    if (body["task_id"] != ctx.binding.task_id or body["assurance"] != "admin_declaration_not_external_attestation"
            or not any(a.request.model_dump() == body["basis"] for a in declared_requests(ctx.business_case, "declare_source_authorization_basis"))):
        raise ValueError("source authorization basis differs from declared local input")


def _source(ctx, item):
    if not any(a.request == item.request for a in declared_requests(ctx.business_case, "prepare_risk_source_query")):
        raise ValueError("frozen source request differs from authenticated original bindings")
    row = ctx.row("SELECT * FROM source_requests WHERE task_id=? AND id=?", (ctx.binding.task_id, item.request.request_id), item.record_sha256)
    frozen = json.loads(row["contract"])
    equal(frozen["query"], item.request.model_dump())
    equal(digest(frozen), row["contract_hash"])
    if row["state"] != "completed" or row["artifact_id"] != item.artifact_id or frozen["profile"]["mode"] != "historical_file":
        raise _Unsupported("completed registered historical connector receipt required")
    grant = ctx.conn.execute("SELECT * FROM source_grants WHERE id=? AND task_id=?", (row["grant_id"], ctx.binding.task_id)).fetchone()
    if grant is None or json.loads(grant["payload"]) != frozen["grant"] or frozen["grant"]["grantee_id"] != row["actor_id"] or frozen["task_id"] != ctx.binding.task_id:
        raise ValueError("connector frozen grant/task/principal differs")
    if not any(a.request.model_dump() == frozen["grant"] for a in declared_requests(ctx.business_case, "grant_risk_source")):
        raise ValueError("connector native grant differs from declared authorization request")
    if not any(a.request.model_dump() == frozen["profile"] for a in declared_requests(ctx.business_case, "register_historical_source_profile")):
        raise ValueError("connector native profile differs from declared profile")
    _authorization_basis(ctx, frozen["grant"])
    receipt = ctx.artifact(item.artifact_id, item.artifact_sha256, kind="risk_source_receipt", origin="risk_context.source_query.v1")
    body = receipt["body"]
    if (body["task_id"] != ctx.binding.task_id or body["request_id"] != item.request.request_id or body["contract_hash"] != row["contract_hash"]
            or body["external_authenticity"] != "not_independently_verified" or body["origin"] != "historical_import_unverified"):
        raise ValueError("connector receipt contract/source assurance differs")
    history = item.request.historical
    frame = _registered_source(ctx, history.dataset_id, history.expected_content_hash)
    if len(frame) > 10000:
        raise _Unsupported("historical connector row budget")
    selected = frame.loc[frame[history.row_id_column].astype(str) == history.row_id]
    if len(selected) != 1:
        raise ValueError("connector source membership not unique")
    raw = selected.iloc[0][history.response_column]
    if not isinstance(raw, str) or len(raw.encode()) > 2000000:
        raise ValueError("connector source record invalid")
    envelope = SourceEnvelope.model_validate_json(raw).model_dump()
    provider_query = frozen["provider_query"]
    before_provider = {key: value for key, value in frozen.items() if key != "provider_query"}
    if (provider_query["request_id"] != digest(before_provider)
            or envelope["subject_namespace"] != frozen["grant"]["subject_namespace"]
            or envelope["subject_token"] != frozen["grant"]["subject_token"]):
        raise ValueError("connector native source subject/request differs")
    normalized = ProviderResult.model_validate({"request_id": provider_query["request_id"], "request_hash": digest(provider_query), "status": "found", "envelope": envelope}).model_dump()
    equal(digest(normalized), body["response_hash"])
    equal(digest(raw.encode()), body["wire_response_hash"])
    expected_membership = {**history.model_dump(exclude={"row_id"}), "row_id_hash": digest(history.row_id)}
    equal(body["dataset_binding"], expected_membership)
    assessment = source_assessment_reference(envelope, body["recorded_at"])
    measured_values(body, {"recorded_status": assessment["status"], "missing_reasons": assessment["missing_reasons"], "finding_codes": assessment["finding_codes"], "source_expires_at": envelope["expires_at"]})
    return {"request_id": item.request.request_id, "status": assessment["status"], "account_count": assessment["account_count"], "inquiry_count": assessment["inquiry_count"], "totals_by_currency": assessment["totals_by_currency"], "missing_reasons": assessment["missing_reasons"], "finding_codes": assessment["finding_codes"], "automated_clearance": False}


def _event(ctx, item):
    if not any(a.request == item.request for a in declared_requests(ctx.business_case, "prepare_risk_event_features")):
        raise ValueError("frozen event request differs from authenticated original bindings")
    intent_row = ctx.row("SELECT * FROM event_requests WHERE task_id=? AND request_id=?", (ctx.binding.task_id, item.request.request_id), item.intent_record_sha256)
    intent = json.loads(intent_row["payload"])
    equal(intent["request"], item.request.model_dump())
    decision_id = "request:" + digest(intent)
    decision = ctx.row("SELECT * FROM event_decisions WHERE task_id=? AND decision_id=?", (ctx.binding.task_id, decision_id), item.decision_record_sha256)
    body = json.loads(decision["payload"])
    equal(digest(body), decision["receipt_hash"])
    if (body["task_id"] != ctx.binding.task_id or body["actor_id"] != intent["actor_id"] or body["grant_id"] != item.request.grant_id or body["decision_id"] != decision_id):
        raise ValueError("event intent/receipt identity differs")
    snapshot = EventSnapshot.model_validate(body["snapshot"]).model_dump()
    equal(snapshot["contract"], item.request.contract.model_dump())
    source_row = ctx.conn.execute("SELECT * FROM event_sources WHERE id=? AND task_id=?", (item.request.contract.source_id, ctx.binding.task_id)).fetchone()
    if source_row is None:
        raise ValueError("registered event source missing")
    equal(json.loads(source_row["payload"]), snapshot["source"])
    equal(source_row["content_hash"], digest(snapshot["source"]))
    if not any(a.request.model_dump() == snapshot["source"] for a in declared_requests(ctx.business_case, "register_risk_event_source")):
        raise ValueError("event native source differs from declared source contract")
    grant = ctx.conn.execute("SELECT * FROM event_grants WHERE id=? AND task_id=?", (item.request.grant_id, ctx.binding.task_id)).fetchone()
    if grant is None:
        raise ValueError("event source grant absent")
    grant_envelope = json.loads(grant["payload"])
    grant_body = grant_envelope["grant"]
    if (grant_body["grantee_id"] != intent["actor_id"] or grant_body["source_contract_hash"] != item.request.contract.source_contract_hash
            or not any(a.request.model_dump() == grant_body for a in declared_requests(ctx.business_case, "grant_risk_event_source"))):
        raise ValueError("event native grant differs from declared source authorization")
    _authorization_basis(ctx, grant_body, grant_envelope["basis_hash"])
    rows = ctx.conn.execute("SELECT * FROM event_claims WHERE source_id=? ORDER BY kind,claim_id,version", (item.request.contract.source_id,)).fetchmany(100001)
    if len(rows) > 100000:
        raise _Unsupported("event original claim budget")
    cutoff = at(item.request.contract.knowledge_cutoff)
    eligible = [row for row in rows if at(row["ingested_at"]) <= cutoff]
    equal([json.loads(row["payload"]) for row in eligible], snapshot["claims"])
    checked = {}
    for row in eligible:
        stored = json.loads(row["payload"])
        equal(row["content_hash"], digest(stored["claim"]))
        if row["version"] != stored["claim"]["version"] or row["ingested_at"] != stored["ingested_at"]:
            raise ValueError("event stored claim lineage/ingestion differs")
        key = (stored["dataset_id"], stored["import_contract_hash"])
        if key not in checked:
            original = ctx.conn.execute("SELECT * FROM event_imports WHERE source_id=? AND contract_hash=?", (stored["source_id"], stored["import_contract_hash"])).fetchone()
            if original is None:
                raise ValueError("event native import missing")
            imported = json.loads(original["payload"])
            contract = imported["import_contract"]
            equal(digest(contract), stored["import_contract_hash"])
            frame = _registered_source(ctx, stored["dataset_id"], stored["dataset_content_hash"])
            if contract["dataset_id"] != stored["dataset_id"] or contract["expected_content_hash"] != stored["dataset_content_hash"] or contract["kind"] != stored["kind"]:
                raise ValueError("event dataset import binding differs")
            checked[key] = {digest(json.loads(raw)) for raw in frame[contract["json_column"]].tolist()}
        if digest(stored["claim"]) not in checked[key]:
            raise ValueError("event claim not present in retained original dataset")
    actual = body["result"]
    expected = event_features_reference(snapshot)
    measured_values(actual, expected)
    artifact = ctx.artifact(item.artifact_id, item.artifact_sha256, kind="risk_event_features", origin="risk_context.replay_events.v1")
    equal(artifact["body"], {"intent": intent, "receipt": body})
    return {"request_id": item.request.request_id, "status": expected["status"], "population_count": expected["population_count"], "features": {key: {field: value[field] for field in ("status", "value", "unit", "observed_member_count", "missing_reasons")} for key, value in expected["features"].items()}, "automated_clearance": False, "fraud_or_identity_proof": False}


def revalidate_connector_archive(archive: Path, *, expected_manifest_sha256: str, frozen_binding: FrozenConnectorBinding):
    binding = FrozenConnectorBinding.model_validate(frozen_binding.model_dump())
    ctx = ExtensionArchive(archive, expected_manifest_sha256, binding, "connector", ("registered_originals", "independent_source_assessment", "independent_event_features"))
    with ctx:
        ctx.open()
        ctx.current = "independent_source_assessment"
        sources = [_source(ctx, item) for item in binding.source_requests]
        ctx.mark("independent_source_assessment", count=len(sources))
        ctx.current = "independent_event_features"
        events = [_event(ctx, item) for item in binding.event_requests]
        ctx.mark("independent_event_features", count=len(events))
        ctx.mark("registered_originals", scope="frozen upload, registered parquet, exact source cell and event claim membership")
        ctx.mark("native_records", scope="externally pinned native signed records; HMAC authority unavailable")
        ctx.result["values"] = {"sources": sources, "events": events}
        ctx.result["unsupported"].update(reference_http_originals="current custody excludes source_private wire/response originals; historical dataset path is independently replayable", legal_identity_or_fraud_truth="descriptive provider claims and observed source relations do not establish identity or fraud", publisher_availability_truth="timestamps are declarations; platform ingestion cutoff is independently checked")
    return ctx.result
