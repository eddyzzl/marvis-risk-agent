"""Authenticated as-of materialization and two-parent evidence, independent of JOIN.

Legacy transform lineage accepts one transform parent only. This service records
both parents in the existing immutable task artifact registry, without fabricating
transform runs or upgrading ordinary JOINs to point-in-time verified data.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import uuid

import pandas as pd
from pydantic import Field

from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.asof_selection import AsOfSelection, select_asof_rows
from marvis.data.contracts import Dataset
from marvis.data.registry import DatasetRegistry
from marvis.data.time_contracts import (
    AsOfJoinSpec, DatasetTimeContract, DatasetTimeStatus, Sha256, _FrozenContract,
)
from marvis.decision_twin._canonical import canonical_json, content_hash
from marvis.files import sha256_file
from marvis.repositories.task_artifacts import TaskArtifactRepository

_KIND = "dataset_point_in_time"
_ORIGIN = "data.asof_join.v1"


class AsOfEvidence(_FrozenContract):
    decision_contract: DatasetTimeContract
    feature_contract: DatasetTimeContract
    spec: AsOfJoinSpec
    output_sha256: Sha256
    memberships: tuple[tuple[int, int | None], ...] = Field(max_length=200_000)
    membership_sha256: Sha256
    source_target_semantics: tuple[tuple[bool, str | None], tuple[bool, str | None]]


@dataclass(frozen=True)
class AsOfJoinResult:
    dataset: Dataset
    status: DatasetTimeStatus
    evidence_path: Path
    membership_sha256: str


class AsOfJoinEngine:
    def __init__(
        self, registry: DatasetRegistry, artifacts: TaskArtifactRepository,
        *, workspace_root: Path,
    ):
        self.registry = registry
        self.artifacts = artifacts
        self.workspace_root = Path(workspace_root).resolve(strict=True)
        self.root = registry.datasets_root.resolve(strict=True)
        self.root.relative_to(self.workspace_root)
        with registry.transaction() as conn:
            database = next(row for row in conn.execute("PRAGMA database_list") if row[1] == "main")
            if Path(database[2]).resolve() != artifacts.db_path.resolve():
                raise ValueError("dataset and artifact repositories must share one database")

    def _sources(self, task_id, dc, fc, spec):
        if dc.role != "decision" or fc.role != "feature_snapshot":
            raise ValueError("as-of materialization requires decision and feature_snapshot roles")
        datasets = [self.registry.get(contract.dataset_id) for contract in (dc, fc)]
        bindings = [self.registry.authenticate_dataset_binding(
            contract.dataset_id, expected_task_id=task_id, expected_content_hash=contract.content_hash,
        ) for contract in (dc, fc)]
        if any(binding.row_count > spec.max_source_rows for binding in bindings):
            raise ValueError("source row budget exceeded")
        if datasets[1].target_col in spec.feature_columns:
            raise ValueError("registered target cannot be selected as a feature")
        feature_columns = {fc.row_id, *fc.entity_columns, fc.version_column, *spec.feature_columns}
        feature_columns.update(column.column for column in fc.time_columns())
        feature_columns.update(pair[1] for pair in spec.partition_pairs)
        frames = [
            self.registry.read_authenticated_binding_snapshot(bindings[0]),
            self.registry.read_authenticated_binding_snapshot(bindings[1], columns=sorted(feature_columns)),
        ]
        selection = select_asof_rows(*frames, dc, fc, spec)
        for binding in bindings:
            self.registry.verify_dataset_binding(binding)
        return datasets, bindings, selection

    def execute(
        self, *, task_id: str, decision_contract: DatasetTimeContract,
        feature_contract: DatasetTimeContract, spec: AsOfJoinSpec,
    ) -> AsOfJoinResult:
        if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", task_id) is None:
            raise ValueError("task_id is not a safe dataset directory identifier")
        dc, fc = decision_contract, feature_contract
        datasets, bindings, selection = self._sources(task_id, dc, fc, spec)
        directory = self.root / task_id / ".pit" / uuid.uuid4().hex
        uow = ArtifactUnitOfWork()
        if directory.resolve() != directory:
            raise ValueError("as-of output directory must not traverse symlinks")
        try:
            output = uow.stage_file(directory, "matrix.parquet")
            proof = uow.stage_file(directory, "evidence.json")
            selection.frame.to_parquet(output.path, index=False)
            # Round-trip the actual carrier before it can become training data.
            _verify_matrix(selection.frame, pd.read_parquet(output.path))
            output_hash = sha256_file(output.path)
            evidence = AsOfEvidence(
                decision_contract=dc, feature_contract=fc, spec=spec,
                output_sha256=output_hash, memberships=selection.memberships,
                membership_sha256=selection.membership_sha256,
                source_target_semantics=tuple((ds.has_target, ds.target_col) for ds in datasets),
            )
            proof.path.write_text(canonical_json(evidence.model_dump(mode="json")), encoding="utf-8")
            evidence_hash = sha256_file(proof.path)

            def commit(conn):
                conn.execute("BEGIN IMMEDIATE")
                self._verify_sources_on_connection(conn, bindings, datasets)
                if sha256_file(output.final_path) != output_hash or sha256_file(proof.final_path) != evidence_hash:
                    raise ValueError("as-of artifacts changed before registration")
                dataset = self.registry.register_existing_on_connection(
                    conn, output.final_path, task_id=task_id, role="asof_sample",
                    target_col_override=datasets[0].target_col,
                )
                if dataset.content_hash != output_hash or dataset.row_count != len(selection.memberships):
                    raise ValueError("materialized matrix changed during registration")
                record = self.artifacts.register_on_connection(
                    conn, task_id=task_id, kind=_KIND,
                    path=proof.final_path.relative_to(self.workspace_root).as_posix(),
                    content_hash=evidence_hash, origin_tool=_ORIGIN,
                    provenance=self._provenance(dataset.id, evidence, selection),
                )
                self._verify_sources_on_connection(conn, bindings, datasets)
                if sha256_file(output.final_path) != output_hash or sha256_file(proof.final_path) != evidence_hash:
                    raise ValueError("as-of artifacts changed before commit")
                return AsOfJoinResult(
                    dataset, DatasetTimeStatus(assurance=selection.assurance, artifact_id=record["id"], reasons=selection.reasons),
                    proof.final_path, selection.membership_sha256,
                )

            return uow.finalize_with_connection(self.registry.transaction, commit)
        except Exception:
            uow.rollback()
            raise

    def _verify_sources_on_connection(self, conn, bindings, datasets):
        for binding, original in zip(bindings, datasets, strict=True):
            current = self.registry.verify_dataset_binding_on_connection(conn, binding)
            if (current.has_target, current.target_col) != (original.has_target, original.target_col):
                raise ValueError("source target semantics changed during as-of selection")

    @staticmethod
    def _provenance(dataset_id: str, evidence: AsOfEvidence, selection: AsOfSelection) -> dict:
        return {
            "schema_version": "dataset-asof-provenance.v1",
            "output_dataset_id": dataset_id,
            "assurance": selection.assurance,
            "reasons": list(selection.reasons),
            "evidence_sha256": content_hash(evidence.model_dump(mode="json")),
            "parents": [{
                "dataset_id": contract.dataset_id, "content_hash": contract.content_hash,
                "contract_sha256": contract.contract_sha256, "role": contract.role,
            } for contract in (evidence.decision_contract, evidence.feature_contract)],
        }

    def dataset_time_status(self, dataset_id: str) -> DatasetTimeStatus:
        """Revalidate all evidence before a downstream verified-data claim.

        A registered legacy dataset/ordinary JOIN remains unknown. Evidence drift
        raises instead of silently downgrading a previously verified materialization.
        This checks recorded temporal constraints, not external source authenticity,
        model split isolation, label maturity, or train-only transform fitting.
        """
        return self._dataset_time_evidence(dataset_id)[0]

    def _dataset_time_evidence(self, dataset_id: str) -> tuple[DatasetTimeStatus, AsOfEvidence | None]:
        dataset = self.registry.get(dataset_id)
        records = [record for record in self.artifacts.list_for_task(dataset.task_id)
                   if record["kind"] == _KIND and record["provenance"].get("output_dataset_id") == dataset_id]
        if not records:
            return DatasetTimeStatus(assurance="unknown", reasons=("no_point_in_time_evidence",)), None
        if len(records) != 1:
            raise ValueError("ambiguous point-in-time evidence")
        record = records[0]
        path = self.workspace_root / record["path"]
        resolved = path.resolve(strict=True)
        resolved.relative_to(self.root)
        if path != resolved or record["origin_tool"] != _ORIGIN or resolved.stat().st_size > 20_000_000:
            raise ValueError("invalid point-in-time evidence path/origin/size")
        raw = resolved.read_bytes()
        if hashlib.sha256(raw).hexdigest() != record["content_hash"]:
            raise ValueError("point-in-time evidence content changed")
        evidence = AsOfEvidence.model_validate_json(raw)
        if dataset.content_hash != evidence.output_sha256:
            raise ValueError("materialized dataset identity changed")
        datasets, bindings, selection = self._sources(
            dataset.task_id, evidence.decision_contract, evidence.feature_contract, evidence.spec,
        )
        if tuple((ds.has_target, ds.target_col) for ds in datasets) != evidence.source_target_semantics:
            raise ValueError("source target semantics changed since materialization")
        if (dataset.has_target, dataset.target_col) != evidence.source_target_semantics[0]:
            raise ValueError("materialized target semantics changed")
        if selection.memberships != evidence.memberships or selection.membership_sha256 != evidence.membership_sha256:
            raise ValueError("point-in-time row membership changed")
        if record["provenance"] != self._provenance(dataset_id, evidence, selection):
            raise ValueError("point-in-time provenance changed")
        output = self.registry.authenticate_dataset_binding(
            dataset_id, expected_task_id=dataset.task_id, expected_content_hash=evidence.output_sha256,
        )
        frame = self.registry.read_authenticated_binding_snapshot(output)
        _verify_matrix(selection.frame, frame)
        with self.registry.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._verify_sources_on_connection(conn, bindings, datasets)
            self.registry.verify_dataset_binding_on_connection(conn, output)
            if sha256_file(resolved) != record["content_hash"]:
                raise ValueError("point-in-time evidence changed during verification")
        return DatasetTimeStatus(assurance=selection.assurance, artifact_id=record["id"], reasons=selection.reasons), evidence

    def feature_time_status(self, dataset_id: str) -> dict:
        """Scope temporal assurance to the joined fields actually selected.

        Decision-table payload columns have no feature-availability contract.
        They must not acquire assurance merely by sharing an as-of output file.
        """
        status, evidence = self._dataset_time_evidence(dataset_id)
        if status.artifact_id is None:
            return {"artifact_id": None, "fields": {}}
        return {
            "artifact_id": status.artifact_id,
            "fields": {
                evidence.spec.feature_prefix + name: {
                    "assurance": status.assurance,
                    "reasons": list(status.reasons),
                }
                for name in evidence.spec.feature_columns
            },
        }


def _verify_matrix(expected: pd.DataFrame, actual: pd.DataFrame) -> None:
    try:
        pd.testing.assert_frame_equal(expected, actual, check_exact=True)
    except AssertionError:
        # Never expose row values in a failed integrity check.
        raise ValueError("materialized matrix differs from authenticated selection") from None
