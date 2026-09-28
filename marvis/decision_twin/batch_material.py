"""Authenticated batch sources and platform-produced receipts, independent of v1."""

from datetime import UTC, datetime
import hashlib
import hmac
import math

import pandas as pd

from marvis import __version__
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.decision_twin._canonical import (
    canonical_json,
    content_hash,
    iso_z,
    parse_datetime,
)
from marvis.decision_twin.artifacts import ContentAddressedAuditStore
from marvis.decision_twin.batch_contracts import HistoricalReplayRequest
from marvis.decision_twin.temporal import bind_temporal_population
from marvis.reference_decision.contracts import DecisionError
from marvis.reference_decision.packages import PackageStore
from marvis.repositories.datasets import DatasetRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.repositories.tasks import TaskRepository


MAX_BATCH_ROWS = 10_000
REPLAY_KIND = "decision_twin_batch_replay"
OUTCOME_KIND = "decision_twin_batch_reconciliation"
PRODUCER = "decision_twin.historical_replay.v2"


def timestamp(value, field):
    if isinstance(value, (datetime, pd.Timestamp)):
        value = value.isoformat()
    try:
        return parse_datetime(value, field)
    except ValueError as exc:
        raise DecisionError(f"invalid_timestamp:{field}") from exc


def scalar(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if not isinstance(value, (str, int, float, bool)):
        raise DecisionError("unsupported_feature_value")
    if isinstance(value, float) and not math.isfinite(value):
        raise DecisionError("nonfinite_feature_value")
    return value


def record_key(value):
    normalized = scalar(value)
    if (
        normalized is None
        or isinstance(normalized, bool)
        or not str(normalized).strip()
    ):
        raise DecisionError("invalid_record_identity")
    # Opaque stable tokens allow matching without duplicating customer identifiers.
    return content_hash({"historical_record_id": normalized})


class BatchMaterial:
    def __init__(self, settings, task_id):
        TaskRepository(settings.db_path).get_task(task_id)
        self.settings, self.task_id = settings, task_id
        self.registry = DatasetRegistry(
            DatasetRepository(settings.db_path),
            DataBackend(settings.datasets_dir),
            settings.datasets_dir,
        )
        self.secret = settings.plugin_admin_token_path.read_text().strip().encode()
        if not self.secret:
            raise DecisionError("local_authentication_secret_unavailable")
        self.packages = PackageStore(settings, self.secret)
        self.artifacts = TaskArtifactRepository(settings.db_path)
        self.store = ContentAddressedAuditStore(
            settings.tasks_dir / task_id / "decision_twin"
        )

    def dataset(self, dataset_id, expected_hash):
        binding = self.registry.authenticate_dataset_binding(
            dataset_id,
            expected_task_id=self.task_id,
            expected_content_hash=expected_hash,
        )
        frame = self.registry.read_authenticated_binding_snapshot(binding)
        if not 0 < len(frame) <= MAX_BATCH_ROWS:
            raise DecisionError("historical_batch_row_budget_exceeded")
        return binding, frame

    def prepare(self, contract: HistoricalReplayRequest):
        binding, frame = self.dataset(
            contract.dataset_id, contract.expected_content_hash
        )
        columns = {contract.record_id_col, contract.decision_at_col}
        for feature in contract.features:
            columns.update(
                (feature.value_col, feature.available_at_col, feature.event_at_col)
            )
        if contract.economics:
            columns.update(
                (contract.economics.ead_col, contract.economics.available_at_col)
            )
        if contract.protected_group:
            columns.update(
                (
                    contract.protected_group.column,
                    contract.protected_group.available_at_col,
                )
            )
        if contract.observed_actions:
            columns.update(
                (
                    contract.observed_actions.action_col,
                    contract.observed_actions.recorded_at_col,
                )
            )
        if columns - set(frame.columns):
            raise DecisionError("historical_columns_missing")
        as_of = timestamp(contract.as_of, "as_of")
        if as_of > datetime.now(UTC):
            raise DecisionError("historical_as_of_is_in_future")
        records, keys = [], set()
        for position, (_, row) in enumerate(frame.iterrows()):
            key = record_key(row[contract.record_id_col])
            if key in keys:
                raise DecisionError("duplicate_historical_record_identity")
            keys.add(key)
            decision_at = timestamp(row[contract.decision_at_col], "decision_at")
            if decision_at > as_of:
                raise DecisionError("decision_after_historical_cutoff")
            values, clocks = {}, {}
            for feature in contract.features:
                event_at = timestamp(row[feature.event_at_col], "event_at")
                available_at = timestamp(row[feature.available_at_col], "available_at")
                if event_at > available_at or available_at > decision_at:
                    raise DecisionError("historical_feature_not_available_at_decision")
                values[feature.name] = scalar(row[feature.value_col])
                clocks[feature.name] = {
                    "event_at": iso_z(event_at),
                    "available_at": iso_z(available_at),
                }
            record = {
                "record_id": key,
                "row_position": position,
                "decision_at": iso_z(decision_at),
                "features": values,
                "field_times": clocks,
            }
            for name, extra in (
                ("economics", contract.economics),
                ("protected_group", contract.protected_group),
            ):
                if (
                    extra
                    and timestamp(row[extra.available_at_col], f"{name}_available_at")
                    > decision_at
                ):
                    raise DecisionError(f"{name}_not_available_at_decision")
            if contract.economics:
                amount = scalar(row[contract.economics.ead_col])
                if type(amount) not in (int, float) or amount < 0:
                    raise DecisionError("invalid_historical_ead")
                record["ead"] = amount
            if contract.protected_group:
                group = scalar(row[contract.protected_group.column])
                if group is None or not str(group).strip():
                    raise DecisionError("historical_group_missing")
                record["group"] = str(group)
            if contract.observed_actions:
                action = scalar(row[contract.observed_actions.action_col])
                at = timestamp(
                    row[contract.observed_actions.recorded_at_col], "action_recorded_at"
                )
                if (
                    action not in {"approval", "reject", "review"}
                    or not decision_at <= at <= as_of
                ):
                    raise DecisionError("invalid_external_historical_action")
                record["observed_action"] = action
                record["action_recorded_at"] = iso_z(at)
            record["facts_hash"] = content_hash(record)
            records.append(record)
        summaries = []
        fields = {f.name for f in contract.features}
        nodes = set()
        for scenario in contract.scenarios:
            manifest = self.packages.get(scenario.package_hash)
            configuration = manifest["configuration"]
            if fields != {f["name"] for f in configuration["raw_schema"]}:
                raise DecisionError("scenario_raw_schema_mismatch")
            # This batch version compares application approvals, not incompatible limit/rate actions.
            actions = [
                rule["action"] for rule in manifest["strategy_spec"]["rules"]
            ] + [manifest["strategy_spec"]["default_action"]]
            if any(
                action["type"] not in {"approval", "reject", "review"}
                for action in actions
            ):
                raise DecisionError("historical_approval_strategy_required")
            nodes.add(configuration["decision_node"])
            summaries.append(
                {
                    **scenario.model_dump(),
                    "decision_node": configuration["decision_node"],
                    "score_product": configuration["score_product"],
                    "strategy_id": configuration["strategy_id"],
                    "strategy_version": configuration["strategy_version"],
                }
            )
        if len(nodes) != 1:
            raise DecisionError("scenario_decision_nodes_do_not_match")
        population_hash = content_hash(
            [
                {k: r[k] for k in ("record_id", "decision_at", "facts_hash")}
                for r in records
            ]
        )
        proposal = {
            "schema_version": "decision_twin.batch_proposal.v2",
            "task_id": self.task_id,
            "contract": contract.model_dump(),
            "population_count": len(records),
            "population_hash": population_hash,
            "scenarios": summaries,
            "authority": "proposal_only",
            "source_assurance": "authenticated_import_with_declared_timestamps",
            "package_time_scope": "retrospective_policy_simulation",
            "causal_gain_verified": False,
        }
        if contract.temporal_stability is not None:
            proposal["temporal_population"] = bind_temporal_population(
                records, contract.temporal_stability
            ).to_dict()
        proposal["proposal_hash"] = content_hash(proposal)
        return binding, records, proposal

    def _signature(self, body):
        return hmac.new(
            self.secret,
            ("decision-twin-batch.v2:" + canonical_json(body)).encode(),
            hashlib.sha256,
        ).hexdigest()

    def publish(self, kind, payload, *, binding):
        body = {
            "producer": PRODUCER,
            "platform_version": __version__,
            "task_id": self.task_id,
            "kind": kind,
            "payload": payload,
        }
        envelope = {"body": body, "signature": self._signature(body)}
        # Publication into the generic registry is the visibility boundary. An interrupted
        # append may leave a recoverable audit entry, but no readable task artifact.
        with self.artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self.registry.verify_dataset_binding_on_connection(conn, binding)
            receipt = self.store.put(
                kind, envelope, idempotency_key=f"{kind}:{content_hash(payload)}"
            )
            record = self.artifacts.register_on_connection(
                conn,
                task_id=self.task_id,
                kind=kind,
                path=str(self.store.artifact_path(receipt.artifact_hash)),
                content_hash=receipt.artifact_hash,
                origin_tool=PRODUCER,
                provenance={
                    "contract_hash": payload["contract_hash"],
                    "artifact_id": receipt.artifact_hash,
                    "source_dataset_id": binding.dataset_id,
                    "source_content_hash": binding.content_hash,
                },
            )
        return {
            "artifact_id": receipt.artifact_hash,
            "registry_id": record["id"],
            "kind": kind,
            "payload": payload,
        }

    def load(self, artifact_id, *, kind=REPLAY_KIND):
        records = self.artifacts.list_for_task(self.task_id)
        if not any(
            r["kind"] == kind
            and r["origin_tool"] == PRODUCER
            and r["content_hash"] == artifact_id
            and r["provenance"].get("artifact_id") == artifact_id
            for r in records
        ):
            raise DecisionError("historical_artifact_not_found", 404)
        envelope = self.store.get_typed(artifact_id, expected_kind=kind)
        body = envelope.get("body")
        if not isinstance(body, dict) or not hmac.compare_digest(
            str(envelope.get("signature", "")), self._signature(body)
        ):
            raise DecisionError("historical_receipt_authentication_failed", 409)
        if (
            body.get("task_id") != self.task_id
            or body.get("kind") != kind
            or body.get("producer") != PRODUCER
            or body.get("platform_version") != __version__
        ):
            raise DecisionError("historical_receipt_binding_mismatch", 409)
        return {"artifact_id": artifact_id, "kind": kind, "payload": body["payload"]}
