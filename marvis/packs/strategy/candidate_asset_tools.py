"""Governed persistence boundary for refined univariate candidate assets.

The deterministic kernel owns every rule and measured effect.  This module only
resolves an immutable task-owned source report, projects its bound dataset,
revalidates lineage around calculation, and atomically registers the resulting
content-addressed asset.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import hmac
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote

from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.errors import DatasetContentDriftError
from marvis.data.labels import resolve_labeled_frame
from marvis.files import sha256_file
from marvis.output.strategy_candidate_report import (
    canonical_strategy_candidate_report_json,
    strategy_candidate_report_from_json,
)
from marvis.packs.strategy.candidate_asset import (
    canonical_candidate_asset_json,
    refine_univariate_candidate,
    validate_candidate_asset,
)
from marvis.packs.strategy.candidate_evidence import validate_candidate_evidence
from marvis.packs.strategy.errors import StrategyError


TOOL_SCHEMA_VERSION = "strategy.refine-univariate-candidate-tool.v1"
ASSET_ARTIFACT_KIND = "strategy_candidate_asset_json"
ASSET_ARTIFACT_SCHEMA_VERSION = "strategy.candidate-asset-artifact.v1"
ORIGIN_TOOL = "strategy.refine_univariate_candidate"
SOURCE_ARTIFACT_KIND = "strategy_candidate_json"
SOURCE_ORIGIN_TOOL = "strategy.analyze_univariate_candidates"
SOURCE_PROVENANCE_SCHEMA_VERSION = "strategy.univariate-candidate-artifact.v1"
SOURCE_PRODUCER_VERSION = "strategy.univariate-candidate/1"

_INPUT_FIELDS = frozenset(
    {
        "source_artifact_id",
        "expected_artifact_content_hash",
        "expected_candidate_id",
        "expected_evidence_hash",
        "feature",
        "method",
        "merge_groups",
        "selection",
        "selection_reason",
    }
)
_REQUIRED_INPUT_FIELDS = _INPUT_FIELDS - {"selection_reason"}
_SOURCE_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "candidate_id",
        "evidence_hash",
        "dataset_id",
        "dataset_content_hash",
        "registry_metadata_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "generation_parameters",
        "format",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ASSET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class _SourceArtifactBinding:
    artifact_id: str
    task_id: str
    kind: str
    path: Path
    content_hash: str
    origin_tool: str
    provenance: dict[str, Any]
    provenance_json: str


@dataclass(frozen=True)
class _DatasetBinding:
    dataset_id: str
    task_id: str
    source_path: str
    path: Path
    content_hash: str
    registry_metadata_hash: str
    columns: tuple[str, ...]
    row_count: int


def run_refine_univariate_candidate(inputs, ctx, runtime) -> dict[str, Any]:
    """Refine one source method into an immutable development-stage asset."""

    normalized_inputs = _validate_inputs(inputs)
    task_id = _required_text(ctx.task_id, "task_id")
    source = _load_source_artifact(
        runtime,
        task_id=task_id,
        artifact_id=normalized_inputs["source_artifact_id"],
        expected_content_hash=normalized_inputs["expected_artifact_content_hash"],
        expected_candidate_id=normalized_inputs["expected_candidate_id"],
        expected_evidence_hash=normalized_inputs["expected_evidence_hash"],
    )
    try:
        report_bytes = source.path.read_bytes()
    except OSError as exc:
        raise StrategyError("source candidate artifact could not be read") from exc
    if not hmac.compare_digest(_sha256(report_bytes), source.content_hash):
        raise StrategyError("source candidate artifact content hash drifted")
    try:
        report = strategy_candidate_report_from_json(report_bytes)
    except (TypeError, ValueError, StrategyError) as exc:
        raise StrategyError("source candidate report failed strict validation") from exc
    evidence = validate_candidate_evidence(report["candidate_evidence"])
    canonical_report = canonical_strategy_candidate_report_json(
        evidence,
        report["univariate_analysis"],
    )
    if canonical_report != report_bytes:
        raise StrategyError("source candidate report is not canonical JSON")
    _require_report_binding(
        evidence,
        source=source,
        task_id=task_id,
        expected_candidate_id=normalized_inputs["expected_candidate_id"],
        expected_evidence_hash=normalized_inputs["expected_evidence_hash"],
    )

    dataset = _load_dataset_binding(runtime, evidence=evidence, source=source)
    feature = normalized_inputs["feature"]
    method = normalized_inputs["method"]
    target_col, projected_columns, drop_nan_labels, expected_dropped = (
        _resolve_projection(
            evidence,
            dataset=dataset,
            feature=feature,
            method=method,
        )
    )
    frame = runtime.backend.read_frame(dataset.path, columns=projected_columns)
    _require_dataset_unchanged(runtime, dataset)
    frame, dropped = resolve_labeled_frame(
        frame,
        target_col,
        drop_nan_labels=drop_nan_labels,
        scope="strategy candidate source dataset",
    )
    if dropped != expected_dropped:
        raise StrategyError(
            "source candidate NaN-label evidence does not match the bound dataset"
        )
    expected_rows = evidence["analysis"]["row_count"]
    if len(frame) != expected_rows:
        raise StrategyError(
            "source candidate row-count evidence does not match the bound dataset"
        )

    # Recheck both immutable inputs immediately before calculation.  The same
    # checks run after calculation and once more under the SQLite writer lock.
    _require_source_unchanged(runtime, source)
    _require_dataset_unchanged(runtime, dataset)
    asset = refine_univariate_candidate(
        evidence,
        frame,
        source_evidence={
            "artifact_id": source.artifact_id,
            "kind": source.kind,
            "content_hash": source.content_hash,
        },
        feature=feature,
        method=method,
        merge_groups=normalized_inputs["merge_groups"],
        selection=normalized_inputs["selection"],
        selection_reason=normalized_inputs.get("selection_reason"),
    )
    normalized_asset = validate_candidate_asset(asset)
    _require_asset_binding(
        normalized_asset,
        evidence=evidence,
        source=source,
        feature=feature,
        method=method,
    )
    _require_source_unchanged(runtime, source)
    _require_dataset_unchanged(runtime, dataset)

    canonical_asset = canonical_candidate_asset_json(normalized_asset)
    if not isinstance(canonical_asset, str):
        raise StrategyError("candidate asset canonical JSON must be text")
    content = canonical_asset.encode("utf-8")
    try:
        roundtrip = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StrategyError("candidate asset canonical JSON is invalid") from exc
    if _canonical_json(validate_candidate_asset(roundtrip)) != canonical_asset:
        raise StrategyError("candidate asset canonical JSON is not stable")

    artifact = _write_candidate_asset(
        runtime,
        task_id=task_id,
        source=source,
        dataset=dataset,
        asset=normalized_asset,
        content=content,
    )
    effect = normalized_asset["effect"]
    if not isinstance(effect, Mapping):
        raise StrategyError("candidate asset effect must be an object")
    effect_id = _required_text(effect.get("effect_id"), "effect.effect_id")
    return {
        "schema_version": TOOL_SCHEMA_VERSION,
        "asset_id": normalized_asset["asset_id"],
        "asset_hash": normalized_asset["asset_hash"],
        "asset_type": normalized_asset["asset_type"],
        "effect_id": effect_id,
        "effect_stage": normalized_asset["effect_stage"],
        "validation_status": normalized_asset["validation_status"],
        "parent_candidate_id": evidence["candidate_id"],
        "parent_evidence_hash": evidence["evidence_hash"],
        "feature": normalized_asset["feature"],
        "method": normalized_asset["method"],
        "selection": normalized_asset["selection"],
        "rule": normalized_asset["rule"],
        "effect": normalized_asset["effect"],
        "metrics": normalized_asset["metrics"],
        "candidate_asset": normalized_asset,
        "artifacts": [artifact],
    }


def _validate_inputs(inputs: object) -> dict[str, Any]:
    if not isinstance(inputs, Mapping):
        raise StrategyError("refine_univariate_candidate inputs must be an object")
    if any(not isinstance(key, str) for key in inputs):
        raise StrategyError("refine_univariate_candidate input keys must be strings")
    missing = sorted(_REQUIRED_INPUT_FIELDS - set(inputs))
    unexpected = sorted(set(inputs) - _INPUT_FIELDS)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unsupported: " + ", ".join(unexpected))
        raise StrategyError(
            "invalid refine_univariate_candidate inputs (" + "; ".join(details) + ")"
        )
    return {
        "source_artifact_id": _required_text(
            inputs["source_artifact_id"], "source_artifact_id"
        ),
        "expected_artifact_content_hash": _required_sha256(
            inputs["expected_artifact_content_hash"],
            "expected_artifact_content_hash",
        ),
        "expected_candidate_id": _required_text(
            inputs["expected_candidate_id"], "expected_candidate_id"
        ),
        "expected_evidence_hash": _required_sha256(
            inputs["expected_evidence_hash"], "expected_evidence_hash"
        ),
        "feature": _required_text(inputs["feature"], "feature"),
        "method": _required_text(inputs["method"], "method"),
        "merge_groups": inputs["merge_groups"],
        "selection": inputs["selection"],
        **(
            {"selection_reason": inputs["selection_reason"]}
            if "selection_reason" in inputs
            else {}
        ),
    }


def _load_source_artifact(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_content_hash: str,
    expected_candidate_id: str,
    expected_evidence_hash: str,
) -> _SourceArtifactBinding:
    record = runtime.task_artifacts.get_for_task(task_id, artifact_id)
    if record is None:
        raise StrategyError(f"source candidate artifact not found: {artifact_id}")
    binding = _normalize_source_record(record)
    if binding.kind != SOURCE_ARTIFACT_KIND:
        raise StrategyError("source artifact must be strategy_candidate_json")
    if binding.origin_tool != SOURCE_ORIGIN_TOOL:
        raise StrategyError("source candidate artifact origin_tool is invalid")
    if not hmac.compare_digest(binding.content_hash, expected_content_hash):
        raise StrategyError("source candidate artifact content hash changed")
    provenance = binding.provenance
    _require_exact_fields(
        provenance,
        _SOURCE_PROVENANCE_FIELDS,
        "source candidate artifact provenance",
    )
    if (
        provenance["schema_version"] != SOURCE_PROVENANCE_SCHEMA_VERSION
        or provenance["producer_version"] != SOURCE_PRODUCER_VERSION
        or provenance["format"] != "json"
    ):
        raise StrategyError("source candidate artifact provenance contract is invalid")
    if provenance["candidate_id"] != expected_candidate_id:
        raise StrategyError("source candidate artifact candidate_id does not match")
    if not _matches_sha256(provenance["evidence_hash"], expected_evidence_hash):
        raise StrategyError("source candidate artifact evidence_hash does not match")
    expected_path = (
        Path(runtime.settings.tasks_dir)
        / task_id
        / "strategy_candidates"
        / f"{expected_candidate_id}_{expected_content_hash[:12]}.json"
    )
    if binding.path != expected_path:
        raise StrategyError("source candidate artifact path is not canonical")
    _require_regular_artifact_path(binding.path, root=Path(runtime.settings.tasks_dir))
    _require_file_content_hash(
        binding.path,
        binding.content_hash,
        "source candidate artifact content hash drifted",
    )
    return binding


def _normalize_source_record(record: Mapping[str, Any]) -> _SourceArtifactBinding:
    required = frozenset(
        {
            "id",
            "task_id",
            "kind",
            "path",
            "content_hash",
            "origin_tool",
            "provenance",
            "created_at",
        }
    )
    _require_exact_fields(record, required, "source candidate artifact record")
    provenance = record["provenance"]
    if not isinstance(provenance, Mapping):
        raise StrategyError("source candidate artifact provenance must be an object")
    normalized_provenance = _json_object(provenance, "source artifact provenance")
    return _SourceArtifactBinding(
        artifact_id=_required_text(record["id"], "source artifact id"),
        task_id=_required_text(record["task_id"], "source artifact task_id"),
        kind=_required_text(record["kind"], "source artifact kind"),
        path=Path(_required_text(record["path"], "source artifact path")),
        content_hash=_required_sha256(
            record["content_hash"], "source artifact content_hash"
        ),
        origin_tool=_required_text(
            record["origin_tool"], "source artifact origin_tool"
        ),
        provenance=normalized_provenance,
        provenance_json=_canonical_json(normalized_provenance),
    )


def _require_report_binding(
    evidence: Mapping[str, Any],
    *,
    source: _SourceArtifactBinding,
    task_id: str,
    expected_candidate_id: str,
    expected_evidence_hash: str,
) -> None:
    if source.task_id != task_id:
        raise StrategyError("source candidate artifact belongs to another task")
    if evidence["candidate_id"] != expected_candidate_id:
        raise StrategyError("source report candidate_id does not match the request")
    if not hmac.compare_digest(evidence["evidence_hash"], expected_evidence_hash):
        raise StrategyError("source report evidence_hash does not match the request")
    identity = evidence["identity"]
    if identity["task_id"] != task_id:
        raise StrategyError("source candidate evidence belongs to another task")
    provenance = source.provenance
    comparisons = {
        "candidate_id": evidence["candidate_id"],
        "evidence_hash": evidence["evidence_hash"],
        "dataset_id": identity["dataset_id"],
        "dataset_content_hash": identity["dataset_content_hash"],
        "workspace_revision": identity["workspace_revision"],
        "workspace_generation": identity["workspace_generation"],
        "semantic_mapping_hash": identity["semantic_mapping_hash"],
        "generation_parameters": evidence["generation"]["parameters"],
    }
    for field, expected in comparisons.items():
        if _canonical_json(provenance[field]) != _canonical_json(expected):
            raise StrategyError(
                f"source candidate artifact provenance {field} does not match evidence"
            )


def _require_asset_binding(
    asset: Mapping[str, Any],
    *,
    evidence: Mapping[str, Any],
    source: _SourceArtifactBinding,
    feature: str,
    method: str,
) -> None:
    expected_parent = {
        "candidate_id": evidence["candidate_id"],
        "evidence_hash": evidence["evidence_hash"],
        "source_evidence": {
            "artifact_id": source.artifact_id,
            "kind": source.kind,
            "content_hash": source.content_hash,
        },
    }
    if _canonical_json(asset.get("parent")) != _canonical_json(expected_parent):
        raise StrategyError("candidate asset parent binding does not match its source")
    if asset.get("feature") != feature or asset.get("method") != method:
        raise StrategyError("candidate asset feature or method binding changed")
    if (
        asset.get("asset_type") != "univariate_refinement"
        or asset.get("effect_stage") != "development"
        or asset.get("validation_status") != "unvalidated"
    ):
        raise StrategyError("candidate asset cannot claim adoption or validation")


def _load_dataset_binding(
    runtime,
    *,
    evidence: Mapping[str, Any],
    source: _SourceArtifactBinding,
) -> _DatasetBinding:
    identity = evidence["identity"]
    dataset_id = identity["dataset_id"]
    try:
        dataset = runtime.registry.get(dataset_id)
    except (KeyError, TypeError, ValueError) as exc:
        raise StrategyError(
            f"candidate source dataset not found: {dataset_id}"
        ) from exc
    if str(dataset.task_id) != identity["task_id"]:
        raise StrategyError("candidate source dataset belongs to another task")
    content_hash = str(dataset.content_hash or "")
    if not _matches_sha256(content_hash, identity["dataset_content_hash"]):
        raise StrategyError("candidate source dataset content hash changed")
    try:
        path = Path(runtime.registry.resolve_verified_path(dataset_id))
    except (
        DatasetContentDriftError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        raise StrategyError(
            "candidate source dataset failed hash verification"
        ) from exc
    _require_file_content_hash(
        path,
        content_hash,
        "candidate source dataset content hash drifted",
    )
    columns = tuple(str(profile.name) for profile in dataset.columns)
    with runtime.task_artifacts.transaction() as conn:
        metadata_hash = _registry_metadata_hash_on_connection(
            conn,
            task_id=identity["task_id"],
            dataset_id=dataset_id,
            expected_content_hash=content_hash,
        )
    if not _matches_sha256(source.provenance["registry_metadata_hash"], metadata_hash):
        raise StrategyError("candidate source dataset registry metadata changed")
    parameters = evidence["generation"]["parameters"]
    if not isinstance(parameters, Mapping):
        raise StrategyError("candidate generation parameters must be an object")
    if not _matches_sha256(parameters.get("registry_metadata_hash"), metadata_hash):
        raise StrategyError(
            "candidate generation registry metadata does not match the dataset"
        )
    return _DatasetBinding(
        dataset_id=dataset_id,
        task_id=identity["task_id"],
        source_path=str(dataset.source_path),
        path=path,
        content_hash=content_hash,
        registry_metadata_hash=metadata_hash,
        columns=columns,
        row_count=int(dataset.row_count),
    )


def _resolve_projection(
    evidence: Mapping[str, Any],
    *,
    dataset: _DatasetBinding,
    feature: str,
    method: str,
) -> tuple[str, list[str], bool, int]:
    analysis = evidence["analysis"]
    target_col = _required_text(analysis["target"], "candidate target")
    parameters = evidence["generation"]["parameters"]
    if parameters.get("target_col") != target_col:
        raise StrategyError("candidate target binding is inconsistent")
    available_columns = set(dataset.columns)
    if target_col not in available_columns:
        raise StrategyError(f"candidate target column not found: {target_col}")
    matching_features = [
        item for item in analysis["features"] if item.get("feature") == feature
    ]
    if len(matching_features) != 1:
        raise StrategyError(f"candidate feature not found: {feature}")
    matching_methods = [
        item for item in matching_features[0]["methods"] if item.get("method") == method
    ]
    if len(matching_methods) != 1 or matching_methods[0].get("status") != "available":
        raise StrategyError(f"available candidate method not found: {feature}/{method}")
    if feature not in available_columns:
        raise StrategyError(f"candidate feature column not found: {feature}")

    drop_nan_labels = parameters.get("drop_nan_labels")
    if not isinstance(drop_nan_labels, bool):
        raise StrategyError("candidate drop_nan_labels evidence is invalid")
    expected_dropped = parameters.get("nan_labels_dropped")
    if (
        isinstance(expected_dropped, bool)
        or not isinstance(expected_dropped, int)
        or expected_dropped < 0
    ):
        raise StrategyError("candidate nan_labels_dropped evidence is invalid")

    projected = [feature, target_col]
    analysis_parameters = analysis.get("parameters")
    if not isinstance(analysis_parameters, Mapping):
        raise StrategyError("candidate analysis parameters must be an object")
    for field, analysis_field in (
        ("loan_amount_col", "loan_amount"),
        ("overdue_amount_col", "overdue_amount"),
    ):
        column = analysis_parameters.get(analysis_field)
        if parameters.get(field) != column:
            raise StrategyError(f"candidate {field} binding is inconsistent")
        if column is None:
            continue
        if not isinstance(column, str) or not column or column not in available_columns:
            raise StrategyError(f"candidate {field} evidence is invalid")
        if column not in projected:
            projected.append(column)
    return target_col, projected, drop_nan_labels, expected_dropped


def _require_source_unchanged(runtime, source: _SourceArtifactBinding) -> None:
    record = runtime.task_artifacts.get_for_task(source.task_id, source.artifact_id)
    if record is None or _normalize_source_record(record) != source:
        raise StrategyError("source candidate artifact registry binding changed")
    _require_regular_artifact_path(source.path, root=Path(runtime.settings.tasks_dir))
    _require_file_content_hash(
        source.path,
        source.content_hash,
        "source candidate artifact content hash drifted",
    )


def _require_dataset_unchanged(runtime, dataset: _DatasetBinding) -> None:
    try:
        live = runtime.registry.get(dataset.dataset_id)
        live_path = Path(runtime.registry.resolve_verified_path(dataset.dataset_id))
    except (
        DatasetContentDriftError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        raise StrategyError("candidate source dataset binding changed") from exc
    live_columns = tuple(str(profile.name) for profile in live.columns)
    _require_file_content_hash(
        live_path,
        dataset.content_hash,
        "candidate source dataset binding changed",
    )
    if (
        str(live.task_id) != dataset.task_id
        or str(live.content_hash or "") != dataset.content_hash
        or int(live.row_count) != dataset.row_count
        or live_columns != dataset.columns
        or live_path != dataset.path
    ):
        raise StrategyError("candidate source dataset binding changed")
    with runtime.task_artifacts.transaction() as conn:
        metadata_hash = _registry_metadata_hash_on_connection(
            conn,
            task_id=dataset.task_id,
            dataset_id=dataset.dataset_id,
            expected_content_hash=dataset.content_hash,
        )
    if not hmac.compare_digest(metadata_hash, dataset.registry_metadata_hash):
        raise StrategyError("candidate source dataset registry metadata changed")


def _write_candidate_asset(
    runtime,
    *,
    task_id: str,
    source: _SourceArtifactBinding,
    dataset: _DatasetBinding,
    asset: Mapping[str, Any],
    content: bytes,
) -> dict[str, Any]:
    asset_id = _required_text(asset["asset_id"], "candidate asset_id")
    if _SAFE_ASSET_ID_RE.fullmatch(asset_id) is None:
        raise StrategyError("candidate asset_id is not safe for persistence")
    asset_hash = _required_sha256(asset["asset_hash"], "candidate asset_hash")
    content_hash = _sha256(content)
    tasks_root = Path(runtime.settings.tasks_dir)
    out_dir = tasks_root / task_id / "strategy_candidate_assets"
    _require_output_directory_boundary(out_dir, root=tasks_root)
    filename = f"{asset_id}_{content_hash[:12]}.json"
    provenance = {
        "schema_version": ASSET_ARTIFACT_SCHEMA_VERSION,
        "producer_version": _required_text(
            asset["producer_version"], "candidate asset producer_version"
        ),
        "asset_id": asset_id,
        "asset_hash": asset_hash,
        "candidate_id": source.provenance["candidate_id"],
        "evidence_hash": source.provenance["evidence_hash"],
        "source_artifact_id": source.artifact_id,
        "source_artifact_content_hash": source.content_hash,
        "dataset_id": dataset.dataset_id,
        "dataset_content_hash": dataset.content_hash,
        "feature": asset["feature"],
        "method": asset["method"],
    }
    uow = ArtifactUnitOfWork()
    staged = uow.stage_file(out_dir, filename)
    db_committed = False
    rollback_attempted_under_lock = False
    try:
        staged.path.write_bytes(content)
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                _require_source_on_connection(conn, source)
                _require_dataset_on_connection(conn, dataset)
                _require_regular_artifact_path(
                    source.path, root=Path(runtime.settings.tasks_dir)
                )
                _require_file_content_hash(
                    source.path,
                    source.content_hash,
                    "source candidate artifact content hash drifted",
                )
                _require_file_content_hash(
                    dataset.path,
                    dataset.content_hash,
                    "candidate source dataset content hash drifted",
                )
                uow.promote_all()
                _require_file_content_hash(
                    staged.final_path,
                    content_hash,
                    "candidate asset changed before artifact registration",
                )
                record = runtime.task_artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=ASSET_ARTIFACT_KIND,
                    path=str(staged.final_path),
                    content_hash=content_hash,
                    origin_tool=ORIGIN_TOOL,
                    provenance=provenance,
                )
                conn.commit()
                db_committed = True
            except Exception:
                rollback_attempted_under_lock = True
                uow.rollback()
                raise
        uow.commit()
    except Exception:
        if not db_committed and not rollback_attempted_under_lock:
            uow.rollback()
        raise
    return {
        "artifact_id": str(record["id"]),
        "kind": ASSET_ARTIFACT_KIND,
        "filename": staged.final_path.name,
        "content_hash": content_hash,
        "download_url": (
            f"/api/tasks/{quote(task_id, safe='')}"
            f"/task-artifacts/{quote(str(record['id']), safe='')}/download"
        ),
    }


def _require_source_on_connection(conn, source: _SourceArtifactBinding) -> None:
    row = conn.execute(
        "SELECT * FROM task_artifacts WHERE task_id = ? AND id = ?",
        (source.task_id, source.artifact_id),
    ).fetchone()
    if row is None:
        raise StrategyError("source candidate artifact disappeared")
    try:
        provenance = json.loads(str(row["provenance_json"]))
    except (TypeError, json.JSONDecodeError) as exc:
        raise StrategyError("source candidate artifact provenance is invalid") from exc
    if (
        str(row["kind"]) != source.kind
        or str(row["path"]) != str(source.path)
        or str(row["content_hash"]) != source.content_hash
        or str(row["origin_tool"]) != source.origin_tool
        or _canonical_json(provenance) != source.provenance_json
    ):
        raise StrategyError("source candidate artifact registry binding changed")


def _require_dataset_on_connection(conn, dataset: _DatasetBinding) -> None:
    metadata_hash = _registry_metadata_hash_on_connection(
        conn,
        task_id=dataset.task_id,
        dataset_id=dataset.dataset_id,
        expected_content_hash=dataset.content_hash,
    )
    if not hmac.compare_digest(metadata_hash, dataset.registry_metadata_hash):
        raise StrategyError("candidate source dataset registry metadata changed")
    row = conn.execute(
        "SELECT source_path FROM datasets WHERE task_id = ? AND id = ?",
        (dataset.task_id, dataset.dataset_id),
    ).fetchone()
    if row is None or str(row["source_path"]) != dataset.source_path:
        raise StrategyError("candidate source dataset registry path changed")


def _registry_metadata_hash_on_connection(
    conn,
    *,
    task_id: str,
    dataset_id: str,
    expected_content_hash: str,
) -> str:
    row = conn.execute(
        """
        SELECT task_id, role, row_count, columns_json, has_target, target_col,
               content_hash
          FROM datasets
         WHERE id = ?
        """,
        (dataset_id,),
    ).fetchone()
    if row is None or str(row["task_id"]) != task_id:
        raise StrategyError(f"candidate source dataset not found: {dataset_id}")
    if not _matches_sha256(row["content_hash"], expected_content_hash):
        raise StrategyError("candidate source dataset registered hash changed")
    columns_json = row["columns_json"]
    if not isinstance(columns_json, str):
        raise StrategyError("candidate source dataset schema is invalid")
    try:
        json.loads(columns_json)
    except json.JSONDecodeError as exc:
        raise StrategyError("candidate source dataset schema is invalid") from exc
    payload = {
        "role": str(row["role"]),
        "row_count": int(row["row_count"]),
        "columns_json": columns_json,
        "has_target": int(row["has_target"]),
        "target_col": row["target_col"],
    }
    return _sha256(_canonical_json(payload).encode("utf-8"))


def _require_regular_artifact_path(path: Path, *, root: Path) -> None:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise StrategyError("source candidate artifact path is not a regular file")
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise StrategyError(
            "source candidate artifact path escapes task storage"
        ) from exc
    current = path.parent
    while True:
        if current.is_symlink():
            raise StrategyError("source candidate artifact path uses a symlink")
        if current == root:
            break
        if current == current.parent:
            raise StrategyError("source candidate artifact path escapes task storage")
        current = current.parent


def _require_output_directory_boundary(path: Path, *, root: Path) -> None:
    if not path.is_absolute() or not root.is_absolute():
        raise StrategyError("candidate asset directory must use absolute task storage")
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise StrategyError("candidate asset directory escapes task storage") from exc
    current = path
    while True:
        if current.is_symlink():
            raise StrategyError("candidate asset directory must not use symlinks")
        if current.exists() and not current.is_dir():
            raise StrategyError("candidate asset directory must be a regular directory")
        if current == root:
            break
        if current == current.parent:
            raise StrategyError("candidate asset directory escapes task storage")
        current = current.parent


def _require_exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], field_name: str
) -> None:
    if any(not isinstance(key, str) for key in value):
        raise StrategyError(f"{field_name} keys must be strings")
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        details = []
        if missing:
            details.append("missing: " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown: " + ", ".join(sorted(unknown)))
        raise StrategyError(f"{field_name} fields are invalid ({'; '.join(details)})")


def _json_object(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    try:
        serialized = _canonical_json(value)
        normalized = json.loads(serialized)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StrategyError(f"{field_name} must be a finite JSON object") from exc
    if not isinstance(normalized, dict):
        raise StrategyError(f"{field_name} must be a JSON object")
    return normalized


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise StrategyError(f"{field_name} must be non-empty text")
    return value.strip()


def _required_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise StrategyError(f"{field_name} must be a lowercase SHA-256 hash")
    return value


def _matches_sha256(value: object, expected: str) -> bool:
    return (
        isinstance(value, str)
        and _SHA256_RE.fullmatch(value) is not None
        and hmac.compare_digest(value, expected)
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_file_content_hash(path: Path, expected: str, message: str) -> None:
    try:
        actual = sha256_file(path)
    except OSError as exc:
        raise StrategyError(message) from exc
    if not hmac.compare_digest(actual, expected):
        raise StrategyError(message)


__all__ = [
    "ASSET_ARTIFACT_KIND",
    "TOOL_SCHEMA_VERSION",
    "run_refine_univariate_candidate",
]
