"""Governed Tool boundary for immutable ``n_of_k`` Voting candidates.

The domain module owns the canonical Voting asset.  This boundary owns the
live Pool/dataset replay, deterministic row measurement, and one atomic
TaskArtifact publication.  Callers may choose only current Pool memberships
and ``n``; they cannot inject conditions, datasets, metrics, actions, adoption,
or deployment state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
from pathlib import Path
import re
import stat
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd

from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.errors import DataLayerError
from marvis.data.labels import resolve_labeled_frame
from marvis.db import ModelingRepository
from marvis.packs.modeling.experiment import ExperimentStore
from marvis.packs.strategy.candidate_evidence import MetricObservation
from marvis.packs.strategy.candidate_fragment import (
    sample_context_hash_from_candidate_evidence,
)
from marvis.packs.strategy.automatic_tree_sample_design import (
    sample_design_ref_from_automatic_tree_source_refs,
)
from marvis.packs.strategy.dsl import canonicalize_expression
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.evaluator import evaluate_expression_frame
from marvis.packs.strategy.pool import validate_strategy_pool
from marvis.packs.strategy.pool_requirement_resolver import (
    ResolvedPoolRequirements,
    hydrate_requirement_fields,
    project_pool_entry_requirements,
    require_resolved_pool_requirements_on_connection,
    resolve_pool_requirements,
)
from marvis.packs.strategy.sample_design_binding import (
    StrategySampleDesignExecutionBinding,
    StrategySampleDesignRef,
    bind_strategy_development_frame,
    load_strategy_sample_design_execution_binding,
    require_strategy_sample_design_execution_binding_on_connection,
    revalidate_strategy_sample_design_execution_binding,
)
from marvis.packs.strategy.sample_design_tools import (
    load_strategy_sample_design_artifact,
)
from marvis.packs.strategy.sample_design_v2_tools import (
    StrategySampleDesignV2ArtifactBinding,
)
from marvis.packs.strategy.voting_candidate import (
    VOTING_CANDIDATE_ASSET_TYPE,
    VOTING_CANDIDATE_ASSET_SCHEMA_VERSION,
    build_voting_candidate_asset,
    validate_voting_candidate_asset,
    verify_voting_candidate_asset_against_pool,
)
from marvis.packs.strategy.voting_candidate_fragment import (
    VOTING_CANDIDATE_ARTIFACT_KIND,
    VOTING_CANDIDATE_ARTIFACT_SCHEMA_VERSION,
    VOTING_CANDIDATE_ARTIFACT_SCHEMA_VERSION_V1,
    VOTING_CANDIDATE_ORIGIN_TOOL,
)
from marvis.repositories.strategy_pool import (
    StrategyCandidatePoolConflictError,
    StrategyCandidatePoolDataError,
    StrategyCandidatePoolNotFoundError,
    StrategyCandidatePoolRepository,
)
from marvis.repositories.task_artifacts import (
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
)


TOOL_SCHEMA_VERSION_V1 = "strategy.build-voting-candidate-tool.v1"
TOOL_SCHEMA_VERSION = "strategy.build-voting-candidate-tool.v2"
VOTING_MEASUREMENT_SCHEMA_VERSION_V1 = "strategy.voting-measurement.v1"
VOTING_MEASUREMENT_SCHEMA_VERSION = "strategy.voting-measurement.v2"

_INPUT_FIELDS = frozenset(
    {
        "strategy_type",
        "expected_pool_revision",
        "expected_pool_snapshot_hash",
        "selected_entry_ids",
        "n",
    }
)
_PROVENANCE_FIELDS_V1 = frozenset(
    {
        "schema_version",
        "producer_version",
        "task_id",
        "kind",
        "format",
        "asset_schema_version",
        "asset_id",
        "asset_hash",
        "asset_type",
        "rule_id",
        "rule_hash",
        "fragment_id",
        "fragment_hash",
        "effect_id",
        "effect_hash",
        "evidence_id",
        "evidence_hash",
        "pool_id",
        "strategy_type",
        "pool_revision",
        "pool_revision_id",
        "pool_snapshot_hash",
        "pool_artifact_id",
        "pool_artifact_content_hash",
        "dataset_id",
        "dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "sample_context_hash",
        "target_col",
        "n",
        "k",
        "measurement_hash",
    }
)
_PROVENANCE_FIELDS = _PROVENANCE_FIELDS_V1 | {"sample_design_ref"}
_DOCUMENT_FIELDS = frozenset({"schema_version", "asset", "measurement"})
_MEASUREMENT_FIELDS_V1 = frozenset(
    {
        "schema_version",
        "target_col",
        "drop_nan_labels",
        "nan_labels_dropped",
        "population_count",
        "labeled_count",
        "hit_distribution",
        "metric_observations",
        "measurement_hash",
    }
)
_MEASUREMENT_FIELDS = _MEASUREMENT_FIELDS_V1 | {"sample_design_ref"}
_DISTRIBUTION_FIELDS = frozenset(
    {"hit_count", "count", "share", "bad_count", "bad_rate", "lift"}
)
_TASK_ARTIFACT_ROW_FIELDS = frozenset(
    {
        "id",
        "task_id",
        "kind",
        "path",
        "content_hash",
        "origin_tool",
        "provenance_json",
        "created_at",
    }
)
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ASSET_ID_RE = re.compile(r"^candidate-asset-[0-9a-f]{32}$")
_BOUNDARY_ERRORS = (
    DataLayerError,
    StrategyCandidatePoolConflictError,
    StrategyCandidatePoolDataError,
    StrategyCandidatePoolNotFoundError,
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
)


@dataclass(frozen=True)
class VerifiedVotingCandidateArtifact:
    """One strict live Voting artifact plus its parent Pool binding."""

    artifact_id: str
    task_id: str
    kind: str
    path: Path
    content_hash: str
    origin_tool: str
    provenance: dict[str, Any]
    canonical_bytes: bytes
    document: dict[str, Any]
    asset: dict[str, Any]

    def artifact_binding(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "task_id": self.task_id,
            "kind": self.kind,
            "content_hash": self.content_hash,
            "origin_tool": self.origin_tool,
            "artifact_schema_version": self.provenance["schema_version"],
            "asset_id": self.asset["asset_id"],
            "asset_hash": self.asset["asset_hash"],
        }


@dataclass(frozen=True)
class _SampleBinding:
    dataset: Any
    path: Path
    dataset_id: str
    dataset_content_hash: str
    registry_metadata_hash: str
    columns: tuple[str, ...]
    row_count: int
    target_col: str
    drop_nan_labels: bool
    nan_labels_dropped: int
    labeled_row_count: int
    loan_amount_col: str | None
    overdue_amount_col: str | None
    sample_context_hash: str
    sample_design: StrategySampleDesignExecutionBinding
    sample_design_v2: StrategySampleDesignV2ArtifactBinding | None


def run_build_voting_candidate(inputs: object, ctx, runtime) -> dict[str, Any]:
    """Build and publish one candidate without changing the Pool or strategy."""

    try:
        request = _validate_inputs(inputs)
        task_id = _required_text(ctx.task_id, "task_id")
        repository = StrategyCandidatePoolRepository(runtime.settings.db_path)
        pool = repository.get_current(task_id, request["strategy_type"])
        if pool is None:
            raise StrategyError("strategy candidate pool not found")
        pool = validate_strategy_pool(pool)
        _require_expected_pool(pool, request)

        selected_entries = _selected_pool_entries(
            pool, request["selected_entry_ids"]
        )
        if any(
            entry["source"]["asset_type"] == VOTING_CANDIDATE_ASSET_TYPE
            or entry["source"]["artifact_kind"]
            == VOTING_CANDIDATE_ARTIFACT_KIND
            for entry in selected_entries
        ):
            raise StrategyError(
                "Voting candidates cannot select another Voting candidate"
            )

        # Local import deliberately avoids a future circular import when
        # pool_tools adds the explicit Voting artifact adapter.
        from marvis.packs.strategy import pool_tools

        cache = pool_tools._LineageCache.empty()
        selected_lineages = []
        for entry in selected_entries:
            source = entry["source"]
            lineage = pool_tools._load_candidate_lineage(
                runtime,
                task_id=task_id,
                artifact_id=source["artifact_id"],
                expected_content_hash=source["artifact_content_hash"],
                expected_asset_id=source["asset_id"],
                expected_asset_hash=source["asset_hash"],
                cache=cache,
            )
            if lineage.source_binding != source:
                raise StrategyError(
                    f"Pool source binding drifted for rule_id: {entry['rule_id']}"
                )
            selected_lineages.append(lineage)
        pool_artifact = pool_tools._load_pool_artifact(
            runtime,
            task_id=task_id,
            snapshot=pool,
        )
        sample = _recover_sample_binding(
            runtime,
            selected_lineages,
            selected_entries=selected_entries,
        )
        frame, resolved_requirements = _read_exact_sample_frame(
            runtime,
            sample=sample,
            entries=selected_entries,
        )
        measured = _measure_voting(
            frame,
            sample=sample,
            entries=selected_entries,
            n=request["n"],
        )
        asset = build_voting_candidate_asset(
            pool,
            selected_entry_ids=[entry["entry_id"] for entry in selected_entries],
            n=request["n"],
            target_col=sample.target_col,
            sample_design_ref=sample.sample_design.to_ref_dict(),
            effect=measured["effect"],
        )
        _require_voting_mask_equivalence(
            measured["voting_mask"],
            frame=measured["labeled_frame"],
            condition=asset["rule"]["condition"],
        )
        document = build_voting_candidate_artifact_document(
            asset,
            target_col=sample.target_col,
            drop_nan_labels=sample.drop_nan_labels,
            nan_labels_dropped=sample.nan_labels_dropped,
            population_count=sample.sample_design.development_population_count,
            labeled_count=sample.labeled_row_count,
            hit_distribution=measured["hit_distribution"],
            metric_observations=measured["metric_observations"],
        )
        return _persist_voting_candidate(
            runtime,
            task_id=task_id,
            repository=repository,
            request=request,
            pool=pool,
            pool_artifact=pool_artifact,
            lineages=selected_lineages,
            sample=sample,
            resolved_requirements=resolved_requirements,
            document=document,
        )
    except StrategyError:
        raise
    except _BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc


def require_voting_snapshot_marginal_reachability(
    runtime,
    *,
    entries: Sequence[Mapping[str, Any]],
    voting_candidates: Mapping[int, VerifiedVotingCandidateArtifact],
    anchor_lineage: Any,
    anchor_entry: Mapping[str, Any],
) -> dict[int, dict[str, int]]:
    """Replay one Pool snapshot once and prove every Voting rule is reachable."""

    sample = _recover_sample_binding(
        runtime,
        [anchor_lineage],
        selected_entries=[anchor_entry],
    )
    frame, _resolved_requirements = _read_exact_sample_frame(
        runtime,
        sample=sample,
        entries=entries,
    )
    labeled, dropped = resolve_labeled_frame(
        frame,
        sample.target_col,
        drop_nan_labels=sample.drop_nan_labels,
        scope="Voting Pool replay source dataset",
    )
    labeled = labeled.reset_index(drop=True)
    if dropped != sample.nan_labels_dropped or len(labeled) != sample.labeled_row_count:
        raise StrategyError("Voting Pool labelled sample changed from candidate")

    claimed = pd.Series(False, index=labeled.index, dtype=bool)
    results: dict[int, dict[str, int]] = {}
    for position, entry in enumerate(entries):
        mask = evaluate_expression_frame(
            labeled,
            entry["execution"]["condition"],
        )
        candidate = voting_candidates.get(position)
        if candidate is not None:
            standalone_count = int(mask.sum())
            expected_count = int(candidate.asset["effect"]["matched_count"])
            if standalone_count != expected_count:
                raise StrategyError(
                    "Voting Pool standalone hits changed from candidate effect"
                )
            marginal_count = int((mask & ~claimed).sum())
            if standalone_count == 0:
                raise StrategyError(
                    "Voting candidate is unreachable because it has no standalone "
                    "sample hits"
                )
            if marginal_count == 0:
                raise StrategyError(
                    "Voting candidate is unreachable because earlier first_match "
                    f"Pool rules shadow all {standalone_count} standalone sample hits"
                )
            results[position] = {
                "standalone_matched_count": standalone_count,
                "marginal_matched_count": marginal_count,
                "shadowed_matched_count": standalone_count - marginal_count,
                "earlier_rule_count": position,
            }
        claimed |= mask
    if set(results) != set(voting_candidates):
        raise StrategyError("Voting Pool replay candidate positions changed")
    return results


def build_voting_candidate_artifact_document(
    asset: Mapping[str, Any],
    *,
    target_col: str,
    drop_nan_labels: bool,
    nan_labels_dropped: int,
    population_count: int,
    labeled_count: int,
    hit_distribution: Sequence[Mapping[str, Any]],
    metric_observations: Sequence[Mapping[str, Any] | MetricObservation],
) -> dict[str, Any]:
    """Build the exact downloadable artifact wrapper for asset + diagnostics."""

    canonical_asset = validate_voting_candidate_asset(asset)
    is_current = (
        canonical_asset["schema_version"] == VOTING_CANDIDATE_ASSET_SCHEMA_VERSION
    )
    measurement_input = {
            "schema_version": (
                VOTING_MEASUREMENT_SCHEMA_VERSION
                if is_current
                else VOTING_MEASUREMENT_SCHEMA_VERSION_V1
            ),
            "target_col": target_col,
            "drop_nan_labels": drop_nan_labels,
            "nan_labels_dropped": nan_labels_dropped,
            "population_count": population_count,
            "labeled_count": labeled_count,
            "hit_distribution": list(hit_distribution),
            "metric_observations": [
                item.to_unchecked_dict()
                if isinstance(item, MetricObservation)
                else dict(item)
                for item in metric_observations
            ],
        }
    if is_current:
        measurement_input["sample_design_ref"] = canonical_asset["sample_design_ref"]
    measurement_body = _normalize_measurement_body(
        measurement_input,
        asset=canonical_asset,
    )
    measurement = {
        **measurement_body,
        "measurement_hash": _sha256(_canonical_json(measurement_body).encode()),
    }
    return validate_voting_candidate_artifact_document(
        {
            "schema_version": (
                VOTING_CANDIDATE_ARTIFACT_SCHEMA_VERSION
                if is_current
                else VOTING_CANDIDATE_ARTIFACT_SCHEMA_VERSION_V1
            ),
            "asset": canonical_asset,
            "measurement": measurement,
        }
    )


def validate_voting_candidate_artifact_document(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StrategyError("Voting candidate artifact must be an object")
    _exact_fields(value, _DOCUMENT_FIELDS, "Voting candidate artifact")
    artifact_schema_version = value["schema_version"]
    if artifact_schema_version not in {
        VOTING_CANDIDATE_ARTIFACT_SCHEMA_VERSION_V1,
        VOTING_CANDIDATE_ARTIFACT_SCHEMA_VERSION,
    }:
        raise StrategyError("Voting candidate artifact schema_version is invalid")
    asset = validate_voting_candidate_asset(value["asset"])
    expected_artifact_schema = (
        VOTING_CANDIDATE_ARTIFACT_SCHEMA_VERSION
        if asset["schema_version"] == VOTING_CANDIDATE_ASSET_SCHEMA_VERSION
        else VOTING_CANDIDATE_ARTIFACT_SCHEMA_VERSION_V1
    )
    if artifact_schema_version != expected_artifact_schema:
        raise StrategyError("Voting candidate artifact and asset schemas disagree")
    measurement_raw = value["measurement"]
    if not isinstance(measurement_raw, Mapping):
        raise StrategyError("Voting candidate measurement must be an object")
    expected_measurement_fields = (
        _MEASUREMENT_FIELDS
        if artifact_schema_version == VOTING_CANDIDATE_ARTIFACT_SCHEMA_VERSION
        else _MEASUREMENT_FIELDS_V1
    )
    _exact_fields(
        measurement_raw,
        expected_measurement_fields,
        "Voting measurement",
    )
    body = _normalize_measurement_body(
        {key: measurement_raw[key] for key in measurement_raw if key != "measurement_hash"},
        asset=asset,
    )
    supplied_hash = _required_hash(
        measurement_raw["measurement_hash"], "measurement_hash"
    )
    expected_hash = _sha256(_canonical_json(body).encode())
    if not hmac.compare_digest(supplied_hash, expected_hash):
        raise StrategyError("Voting candidate measurement_hash changed")
    return {
        "schema_version": artifact_schema_version,
        "asset": asset,
        "measurement": {**body, "measurement_hash": supplied_hash},
    }


def canonical_voting_candidate_artifact_json(value: Mapping[str, Any]) -> str:
    return _canonical_json(validate_voting_candidate_artifact_document(value))


def voting_candidate_artifact_provenance(
    document: Mapping[str, Any],
    *,
    task_id: str,
    pool_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the exact registry provenance for a canonical document."""

    normalized = validate_voting_candidate_artifact_document(document)
    asset = normalized["asset"]
    pool_ref = asset["pool_ref"]
    identity = asset["evidence_identity"]
    if pool_ref["task_id"] != task_id:
        raise StrategyError("Voting candidate belongs to another task")
    provenance = {
        "schema_version": normalized["schema_version"],
        "producer_version": (
            TOOL_SCHEMA_VERSION
            if normalized["schema_version"]
            == VOTING_CANDIDATE_ARTIFACT_SCHEMA_VERSION
            else TOOL_SCHEMA_VERSION_V1
        ),
        "task_id": task_id,
        "kind": VOTING_CANDIDATE_ARTIFACT_KIND,
        "format": "json",
        "asset_schema_version": asset["schema_version"],
        "asset_id": asset["asset_id"],
        "asset_hash": asset["asset_hash"],
        "asset_type": asset["asset_type"],
        "rule_id": asset["rule"]["rule_id"],
        "rule_hash": asset["rule"]["rule_hash"],
        "fragment_id": asset["fragment"]["fragment_id"],
        "fragment_hash": asset["fragment"]["fragment_hash"],
        "effect_id": asset["effect"]["effect_id"],
        "effect_hash": asset["effect"]["effect_hash"],
        "evidence_id": asset["candidate_evidence"]["candidate_id"],
        "evidence_hash": asset["candidate_evidence"]["evidence_hash"],
        "pool_id": pool_ref["pool_id"],
        "strategy_type": pool_ref["strategy_type"],
        "pool_revision": pool_ref["revision"],
        "pool_revision_id": pool_ref["revision_id"],
        "pool_snapshot_hash": pool_ref["snapshot_hash"],
        "pool_artifact_id": _required_text(pool_artifact.get("id"), "pool artifact id"),
        "pool_artifact_content_hash": _required_hash(
            pool_artifact.get("content_hash"), "pool artifact content hash"
        ),
        "dataset_id": identity["dataset_id"],
        "dataset_content_hash": identity["dataset_content_hash"],
        "workspace_revision": identity["workspace_revision"],
        "workspace_generation": identity["workspace_generation"],
        "semantic_mapping_hash": identity["semantic_mapping_hash"],
        "sample_context_hash": identity["sample_context_hash"],
        "target_col": asset["measurement_context"]["target_col"],
        "n": asset["voting"]["n"],
        "k": asset["voting"]["k"],
        "measurement_hash": normalized["measurement"]["measurement_hash"],
    }
    expected_fields = (
        _PROVENANCE_FIELDS
        if normalized["schema_version"] == VOTING_CANDIDATE_ARTIFACT_SCHEMA_VERSION
        else _PROVENANCE_FIELDS_V1
    )
    if normalized["schema_version"] == VOTING_CANDIDATE_ARTIFACT_SCHEMA_VERSION:
        provenance["sample_design_ref"] = asset["sample_design_ref"]
    _exact_fields(provenance, expected_fields, "Voting artifact provenance")
    return provenance


def canonical_voting_candidate_path(
    tasks_dir: Path | str,
    *,
    task_id: str,
    asset_id: str,
) -> Path:
    task = _safe_component(task_id, "task_id")
    asset = _required_text(asset_id, "asset_id")
    if _ASSET_ID_RE.fullmatch(asset) is None:
        raise StrategyError("Voting candidate asset_id is unsafe")
    return (
        Path(tasks_dir).absolute()
        / task
        / "strategy_voting_candidates"
        / f"{asset}.json"
    )


def load_verified_voting_candidate_artifact(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_content_hash: str,
    expected_asset_id: str,
    expected_asset_hash: str,
) -> VerifiedVotingCandidateArtifact:
    record = runtime.task_artifacts.get_for_task(task_id, artifact_id)
    if record is None:
        raise StrategyError(f"Voting candidate artifact not found: {artifact_id}")
    return _load_verified_voting_record(
        record,
        tasks_dir=runtime.settings.tasks_dir,
        task_id=task_id,
        artifact_id=artifact_id,
        expected_content_hash=expected_content_hash,
        expected_asset_id=expected_asset_id,
        expected_asset_hash=expected_asset_hash,
    )


def load_verified_voting_candidate_artifact_on_connection(
    conn,
    *,
    tasks_dir: Path | str,
    task_id: str,
    artifact_id: str,
    expected_content_hash: str,
    expected_asset_id: str,
    expected_asset_hash: str,
) -> VerifiedVotingCandidateArtifact:
    row = conn.execute(
        """
        SELECT id, task_id, kind, path, content_hash, origin_tool,
               provenance_json, created_at
          FROM task_artifacts
         WHERE task_id = ? AND id = ?
        """,
        (task_id, artifact_id),
    ).fetchone()
    if row is None:
        raise StrategyError(f"Voting candidate artifact not found: {artifact_id}")
    record = {field: row[field] for field in _TASK_ARTIFACT_ROW_FIELDS}
    return _load_verified_voting_record(
        record,
        tasks_dir=tasks_dir,
        task_id=task_id,
        artifact_id=artifact_id,
        expected_content_hash=expected_content_hash,
        expected_asset_id=expected_asset_id,
        expected_asset_hash=expected_asset_hash,
        raw_provenance=True,
    )


def _validate_inputs(inputs: object) -> dict[str, Any]:
    if not isinstance(inputs, Mapping):
        raise StrategyError("build_voting_candidate inputs must be an object")
    if any(not isinstance(key, str) for key in inputs):
        raise StrategyError("build_voting_candidate input keys must be strings")
    missing = sorted(_INPUT_FIELDS - set(inputs))
    unsupported = sorted(set(inputs) - _INPUT_FIELDS)
    if missing or unsupported:
        detail = []
        if missing:
            detail.append("missing: " + ", ".join(missing))
        if unsupported:
            detail.append("unsupported: " + ", ".join(unsupported))
        raise StrategyError(
            "invalid build_voting_candidate inputs (" + "; ".join(detail) + ")"
        )
    selected = _text_list(inputs["selected_entry_ids"], "selected_entry_ids")
    if len(selected) < 2 or len(selected) > 50:
        raise StrategyError("selected_entry_ids must contain between 2 and 50 entries")
    if len(set(selected)) != len(selected):
        raise StrategyError("selected_entry_ids must not contain duplicates")
    n = _positive_int(inputs["n"], "n")
    if n > len(selected):
        raise StrategyError("n must not exceed selected_entry_ids count")
    return {
        "strategy_type": _required_text(inputs["strategy_type"], "strategy_type"),
        "expected_pool_revision": _positive_int(
            inputs["expected_pool_revision"], "expected_pool_revision"
        ),
        "expected_pool_snapshot_hash": _required_hash(
            inputs["expected_pool_snapshot_hash"],
            "expected_pool_snapshot_hash",
        ),
        "selected_entry_ids": selected,
        "n": n,
    }


def _require_expected_pool(
    pool: Mapping[str, Any], request: Mapping[str, Any]
) -> None:
    if pool["strategy_type"] != request["strategy_type"]:
        raise StrategyError("strategy candidate pool type changed")
    if pool["revision"] != request["expected_pool_revision"] or not hmac.compare_digest(
        pool["snapshot_hash"], request["expected_pool_snapshot_hash"]
    ):
        raise StrategyError("stale strategy candidate pool revision or snapshot hash")


def _selected_pool_entries(
    pool: Mapping[str, Any], selected_entry_ids: Sequence[str]
) -> list[dict[str, Any]]:
    requested = set(selected_entry_ids)
    selected = [entry for entry in pool["entries"] if entry["entry_id"] in requested]
    found = {entry["entry_id"] for entry in selected}
    unknown = sorted(requested - found)
    if unknown:
        raise StrategyError(
            "selected_entry_ids contains unknown Pool entries: " + ", ".join(unknown)
        )
    if any(entry["enabled"] is not True for entry in selected):
        raise StrategyError("selected Pool entries must be enabled")
    # Pool validation already proves unique, contiguous positions.  Filtering
    # preserves that canonical order and deliberately ignores caller ordering.
    return selected


def _recover_sample_binding(
    runtime,
    lineages: Sequence[Any],
    *,
    selected_entries: Sequence[Mapping[str, Any]],
) -> _SampleBinding:
    if len(lineages) != len(selected_entries) or not lineages:
        raise StrategyError("Voting candidate lineage selection is incomplete")
    recovered: list[_SampleBinding] = []
    automatic_tree_samples: dict[tuple[str, str, str, str, str], _SampleBinding] = {}
    for lineage in lineages:
        if hasattr(lineage, "tree") and hasattr(lineage, "selection"):
            cache_key = (
                str(lineage.tree.artifact_id),
                str(lineage.tree.content_hash),
                str(lineage.dataset.dataset_id),
                str(lineage.dataset.content_hash),
                str(lineage.dataset.path),
            )
            sample = automatic_tree_samples.get(cache_key)
            if sample is None:
                sample = _sample_from_lineage(runtime, lineage)
                automatic_tree_samples[cache_key] = sample
            recovered.append(sample)
        else:
            recovered.append(_sample_from_lineage(runtime, lineage))
    first = recovered[0]
    identity_fields = (
        "dataset_id",
        "dataset_content_hash",
        "registry_metadata_hash",
        "columns",
        "row_count",
        "target_col",
        "drop_nan_labels",
        "nan_labels_dropped",
        "labeled_row_count",
        "loan_amount_col",
        "overdue_amount_col",
        "sample_context_hash",
        "sample_design",
    )
    for item in recovered[1:]:
        if any(getattr(item, field) != getattr(first, field) for field in identity_fields):
            raise StrategyError(
                "selected Pool entries do not share one exact measurement sample"
            )
        if item.path != first.path:
            raise StrategyError("selected Pool entries resolve different dataset paths")
        if _sample_design_v2_identity(
            item.sample_design_v2
        ) != _sample_design_v2_identity(first.sample_design_v2):
            raise StrategyError(
                "selected Pool entries do not share one exact SampleDesign V2"
            )
    for entry in selected_entries:
        identity = entry["source"]["evidence_identity"]
        comparisons = {
            "dataset_id": first.dataset_id,
            "dataset_content_hash": first.dataset_content_hash,
            "sample_context_hash": first.sample_context_hash,
        }
        for field, expected in comparisons.items():
            if identity[field] != expected:
                raise StrategyError(
                    f"selected Pool evidence identity {field} changed"
                )
    revalidated = revalidate_strategy_sample_design_execution_binding(
        runtime,
        first.sample_design,
    )
    if revalidated != first.sample_design:
        raise StrategyError("Voting sample-design binding changed")
    return first


def _sample_design_v2_identity(
    binding: StrategySampleDesignV2ArtifactBinding | None,
) -> tuple[object, ...] | None:
    if binding is None:
        return None
    design = binding.bundle["sample_design"]
    header = binding.membership["header"]
    source = binding.source_binding
    return (
        binding.task_id,
        binding.membership_artifact_id,
        binding.membership_artifact_content_hash,
        binding.bundle_artifact_id,
        binding.bundle_artifact_content_hash,
        binding.bundle["bundle_id"],
        design["sample_design_id"],
        design["content_hash"],
        header["membership_id"],
        header["content_hash"],
        header["payload_hash"],
        header["row_count"],
        source.dataset_id,
        source.dataset_content_hash,
        source.workspace_revision,
        source.workspace_generation,
        source.semantic_mapping_hash,
    )


def _sample_from_lineage(runtime, lineage: Any) -> _SampleBinding:
    # These concrete classes are intentionally detected structurally.  It
    # avoids importing pool_tools at module import time and therefore keeps the
    # future explicit Voting adapter free of an import cycle.
    if hasattr(lineage, "evidence") and hasattr(lineage, "asset_record"):
        return _sample_from_univariate_lineage(runtime, lineage)
    if (
        hasattr(lineage, "asset")
        and hasattr(lineage, "selection")
        and isinstance(
            getattr(lineage.asset, "sample_design", None),
            StrategySampleDesignV2ArtifactBinding,
        )
    ):
        return _sample_from_scorecard_lineage(lineage)
    if hasattr(lineage, "tree") and hasattr(lineage, "selection"):
        return _sample_from_automatic_tree_lineage(runtime, lineage)
    raise StrategyError("unsupported Voting candidate source lineage")


def _sample_from_univariate_lineage(runtime, lineage: Any) -> _SampleBinding:
    evidence = lineage.evidence
    dataset = lineage.dataset
    analysis = evidence["analysis"]
    parameters = evidence["generation"]["parameters"]
    identity = evidence["identity"]
    target_col = _required_text(analysis["target"], "candidate target_col")
    if parameters.get("target_col") != target_col:
        raise StrategyError("candidate target binding is inconsistent")
    drop_nan_labels = parameters.get("drop_nan_labels")
    if not isinstance(drop_nan_labels, bool):
        raise StrategyError("candidate drop_nan_labels binding is invalid")
    nan_labels_dropped = _non_negative_int(
        parameters.get("nan_labels_dropped"), "candidate nan_labels_dropped"
    )
    labeled_row_count = _positive_int(
        analysis["row_count"], "candidate labeled row_count"
    )
    sample_hash = sample_context_hash_from_candidate_evidence(evidence)
    source_hash = lineage.verified_fragment["evidence"]["identity"][
        "sample_context_hash"
    ]
    if not hmac.compare_digest(sample_hash, source_hash):
        raise StrategyError("candidate sample context hash changed")
    loan_col = _optional_column(parameters.get("loan_amount_col"), "loan_amount_col")
    overdue_col = _optional_column(
        parameters.get("overdue_amount_col"), "overdue_amount_col"
    )
    sample_design = load_strategy_sample_design_execution_binding(
        runtime,
        task_id=str(identity["task_id"]),
        sample_design_ref=parameters.get("sample_design_ref"),
        dataset_id=str(identity["dataset_id"]),
        dataset_content_hash=str(identity["dataset_content_hash"]),
        workspace_revision=int(identity["workspace_revision"]),
        workspace_generation=int(identity["workspace_generation"]),
        semantic_mapping_hash=str(identity["semantic_mapping_hash"]),
        target_col=target_col,
        drop_nan_labels=drop_nan_labels,
        loan_amount_col=loan_col,
        overdue_amount_col=overdue_col,
    )
    if (
        sample_design.loan_amount_col != loan_col
        or sample_design.overdue_amount_col != overdue_col
    ):
        raise StrategyError(
            "candidate amount bindings do not match governed sample design"
        )
    return _SampleBinding(
        dataset=dataset,
        path=Path(dataset.path),
        dataset_id=str(dataset.dataset_id),
        dataset_content_hash=str(dataset.content_hash),
        registry_metadata_hash=str(dataset.registry_metadata_hash),
        columns=tuple(dataset.columns),
        row_count=int(dataset.row_count),
        target_col=target_col,
        drop_nan_labels=drop_nan_labels,
        nan_labels_dropped=nan_labels_dropped,
        labeled_row_count=labeled_row_count,
        loan_amount_col=sample_design.loan_amount_col,
        overdue_amount_col=sample_design.overdue_amount_col,
        sample_context_hash=sample_hash,
        sample_design=sample_design,
        sample_design_v2=None,
    )


def _sample_from_scorecard_lineage(lineage: Any) -> _SampleBinding:
    band_binding = lineage.asset
    sample_design_v2 = band_binding.sample_design
    source = sample_design_v2.source_binding
    sample_design = source.legacy
    dataset = lineage.dataset
    asset = band_binding.asset
    identity = asset["identity"]
    vector = asset["score_vector"]
    design = sample_design_v2.bundle["sample_design"]
    target = design["target_selector"]
    if target.get("status") != "resolved":
        raise StrategyError("scorecard Voting target selector is unresolved")
    target_col = _required_text(
        target.get("column"),
        "scorecard Voting target_col",
    )
    if (
        target_col != sample_design.target_col
        or target.get("bad_value") != sample_design.target_bad_value
        or target.get("drop_missing") is not sample_design.drop_nan_labels
    ):
        raise StrategyError(
            "scorecard target semantics changed from governed sample design"
        )

    expected_sample_ref = {
        "membership_artifact_id": sample_design_v2.membership_artifact_id,
        "expected_membership_artifact_content_hash": (
            sample_design_v2.membership_artifact_content_hash
        ),
        "bundle_artifact_id": sample_design_v2.bundle_artifact_id,
        "expected_bundle_artifact_content_hash": (
            sample_design_v2.bundle_artifact_content_hash
        ),
        "expected_bundle_id": sample_design_v2.bundle["bundle_id"],
        "expected_sample_design_id": design["sample_design_id"],
        "expected_sample_design_content_hash": design["content_hash"],
    }
    if asset["sample_design_ref"] != expected_sample_ref:
        raise StrategyError(
            "scorecard sample-design V2 reference changed from source binding"
        )

    identity_comparisons = {
        "task_id": sample_design_v2.task_id,
        "dataset_id": source.dataset_id,
        "dataset_content_hash": source.dataset_content_hash,
        "workspace_revision": source.workspace_revision,
        "workspace_generation": source.workspace_generation,
        "semantic_mapping_hash": source.semantic_mapping_hash,
    }
    if any(identity[field] != expected for field, expected in identity_comparisons.items()):
        raise StrategyError(
            "scorecard candidate identity changed from governed sample design"
        )
    dataset_comparisons = {
        "task_id": source.task_id,
        "dataset_id": source.dataset_id,
        "content_hash": source.dataset_content_hash,
        "registry_metadata_hash": source.dataset_registry_metadata_hash,
        "columns": source.columns,
        "row_count": source.row_count,
    }
    if (
        any(
            getattr(dataset, field) != expected
            for field, expected in dataset_comparisons.items()
        )
        or Path(dataset.path) != source.dataset_path
        or dataset.source_path != source.dataset_source_path
    ):
        raise StrategyError(
            "scorecard candidate dataset changed from governed sample design"
        )

    development_count = _positive_int(
        vector["development_count"],
        "scorecard development_count",
    )
    labeled_count = _positive_int(
        vector["labeled_count"],
        "scorecard labeled_count",
    )
    if (
        vector["row_count"] != source.row_count
        or development_count != sample_design.development_population_count
        or labeled_count > development_count
        or development_count
        != int(
            np.count_nonzero(
                sample_design_v2.membership["masks"]["risk/development"]
            )
        )
    ):
        raise StrategyError(
            "scorecard candidate measurement sample changed"
        )
    missing = development_count - labeled_count
    if missing and not sample_design.drop_nan_labels:
        raise StrategyError(
            "scorecard candidate dropped labels without sample authorization"
        )
    source_hash = lineage.verified_fragment["evidence"]["identity"][
        "sample_context_hash"
    ]
    if not hmac.compare_digest(identity["sample_context_hash"], source_hash):
        raise StrategyError("scorecard candidate sample context hash changed")
    return _SampleBinding(
        dataset=dataset,
        path=Path(dataset.path),
        dataset_id=str(dataset.dataset_id),
        dataset_content_hash=str(dataset.content_hash),
        registry_metadata_hash=str(dataset.registry_metadata_hash),
        columns=tuple(dataset.columns),
        row_count=int(dataset.row_count),
        target_col=target_col,
        drop_nan_labels=sample_design.drop_nan_labels,
        nan_labels_dropped=missing,
        labeled_row_count=labeled_count,
        loan_amount_col=sample_design.loan_amount_col,
        overdue_amount_col=sample_design.overdue_amount_col,
        sample_context_hash=str(identity["sample_context_hash"]),
        sample_design=sample_design,
        sample_design_v2=sample_design_v2,
    )


def _sample_from_automatic_tree_lineage(runtime, lineage: Any) -> _SampleBinding:
    tree = lineage.tree.asset
    dataset = lineage.dataset
    training = tree["tree_result"]["training"]
    identity = tree["identity"]
    target_col = _required_text(training["target_col"], "automatic-tree target_col")
    loan_col = _optional_column(training["loan_amount_col"], "loan_amount_col")
    overdue_col = _optional_column(
        training["overdue_amount_col"], "overdue_amount_col"
    )
    sample_ref = sample_design_ref_from_automatic_tree_source_refs(
        tree["source_refs"]
    )
    provenance_ref = lineage.tree.provenance.get("sample_design_ref")
    if StrategySampleDesignRef.from_value(provenance_ref).to_ref_dict() != sample_ref:
        raise StrategyError(
            "automatic-tree sample-design asset and provenance bindings disagree"
        )
    weight = training["sample_weight"]
    weight_col = weight.get("column") if weight["status"] == "available" else None
    reference = StrategySampleDesignRef.from_value(sample_ref)
    sample_artifact = load_strategy_sample_design_artifact(
        runtime,
        task_id=str(identity["task_id"]),
        artifact_id=reference.artifact_id,
        expected_artifact_content_hash=reference.artifact_content_hash,
        expected_sample_design_id=reference.sample_design_id,
        expected_sample_design_content_hash=reference.sample_design_content_hash,
    )
    drop_nan_labels = bool(
        sample_artifact.bundle["sample_design"]["target_definition"][
            "drop_nan_labels"
        ]
    )
    sample_design = load_strategy_sample_design_execution_binding(
        runtime,
        task_id=str(identity["task_id"]),
        sample_design_ref=sample_ref,
        dataset_id=str(identity["dataset_id"]),
        dataset_content_hash=str(identity["dataset_content_hash"]),
        workspace_revision=int(identity["workspace_revision"]),
        workspace_generation=int(identity["workspace_generation"]),
        semantic_mapping_hash=str(identity["semantic_mapping_hash"]),
        target_col=target_col,
        drop_nan_labels=drop_nan_labels,
        weight_col=weight_col,
        loan_amount_col=loan_col,
        overdue_amount_col=overdue_col,
    )
    if lineage.dataset.dataset_id != identity["dataset_id"]:
        raise StrategyError("automatic-tree dataset identity changed")
    labeled = int(training["row_count"])
    missing = sample_design.development_population_count - labeled
    if missing < 0 or (missing and not sample_design.drop_nan_labels):
        raise StrategyError("automatic-tree labelled sample changed")
    return _SampleBinding(
        dataset=dataset,
        path=Path(dataset.path),
        dataset_id=str(dataset.dataset_id),
        dataset_content_hash=str(dataset.content_hash),
        registry_metadata_hash=str(dataset.registry_metadata_hash),
        columns=tuple(dataset.columns),
        row_count=int(dataset.row_count),
        target_col=target_col,
        drop_nan_labels=sample_design.drop_nan_labels,
        nan_labels_dropped=missing,
        labeled_row_count=labeled,
        loan_amount_col=loan_col,
        overdue_amount_col=overdue_col,
        sample_context_hash=str(identity["sample_context_hash"]),
        sample_design=sample_design,
        sample_design_v2=None,
    )


def _read_exact_sample_frame(
    runtime,
    *,
    sample: _SampleBinding,
    entries: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, ResolvedPoolRequirements | None]:
    requirements = project_pool_entry_requirements(entries)
    resolved: ResolvedPoolRequirements | None = None
    if requirements:
        if sample.sample_design_v2 is None:
            raise StrategyError(
                "Voting score requirements require one exact SampleDesign V2"
            )
        resolved = resolve_pool_requirements(
            _modeling_runtime(runtime),
            task_id=sample.sample_design_v2.task_id,
            compiled_design={"requirements": list(requirements)},
            sample_design=sample.sample_design_v2,
        )
    virtual_fields = set(() if resolved is None else resolved.virtual_fields)
    fields: set[str] = set()
    for entry in entries:
        fields.update(_expression_fields(entry["execution"]["condition"]))
    fields -= virtual_fields
    fields.add(sample.target_col)
    if sample.sample_design.split_column is not None:
        fields.add(sample.sample_design.split_column)
    for column in (sample.loan_amount_col, sample.overdue_amount_col):
        if column is not None:
            fields.add(column)
    unknown = sorted(fields - set(sample.columns))
    if unknown:
        raise StrategyError(
            "Voting source dataset is missing exact rule fields: " + ", ".join(unknown)
        )
    frame = runtime.backend.read_frame(sample.path, columns=sorted(fields))
    if len(frame) != sample.row_count:
        raise StrategyError("Voting source dataset row count changed")
    frame = frame.reset_index(drop=True)
    if resolved is not None:
        frame = hydrate_requirement_fields(frame, resolved=resolved)
    # Full pool lineage replay has already checked the registry row and bytes.
    # Do one more independent byte verification immediately around evaluation.
    _require_file_hash(
        sample.path,
        sample.dataset_content_hash,
        "Voting source dataset content hash changed",
    )
    return (
        bind_strategy_development_frame(
            frame,
            binding=sample.sample_design,
        ),
        resolved,
    )

def _modeling_runtime(runtime):
    """Add score-evidence repositories without mutating the pack runtime."""

    if hasattr(runtime, "experiments") and hasattr(runtime, "modeling_repo"):
        return runtime
    proxy = SimpleNamespace(**vars(runtime))
    proxy.experiments = ExperimentStore(runtime.settings.db_path)
    proxy.modeling_repo = ModelingRepository(runtime.settings.db_path)
    return proxy


def _measure_voting(
    frame: pd.DataFrame,
    *,
    sample: _SampleBinding,
    entries: Sequence[Mapping[str, Any]],
    n: int,
) -> dict[str, Any]:
    labeled, dropped = resolve_labeled_frame(
        frame,
        sample.target_col,
        drop_nan_labels=sample.drop_nan_labels,
        scope="Voting source dataset",
    )
    labeled = labeled.reset_index(drop=True)
    if dropped != sample.nan_labels_dropped or len(labeled) != sample.labeled_row_count:
        raise StrategyError("Voting labelled sample changed from source evidence")
    target_numeric = pd.to_numeric(labeled[sample.target_col], errors="raise").to_numpy(
        dtype=float
    )
    if not np.all(np.isfinite(target_numeric)) or not np.all(
        np.isin(target_numeric, [0.0, 1.0])
    ):
        raise StrategyError("Voting target must contain only binary 0/1 values")
    target = target_numeric.astype(np.int64)
    member_masks = [
        evaluate_expression_frame(labeled, entry["execution"]["condition"])
        .to_numpy(dtype=bool, copy=False)
        for entry in entries
    ]
    hit_count = np.zeros(len(labeled), dtype=np.int64)
    for member in member_masks:
        hit_count += member.astype(np.int64)
    voting_mask = hit_count >= n
    effect = _effect_from_mask(
        voting_mask,
        target=target,
        population_count=len(frame),
    )
    distribution = _hit_distribution(
        hit_count,
        target=target,
        k=len(entries),
    )
    amount_values = {
        "loan_amount": _amount_values(labeled, sample.loan_amount_col),
        "overdue_amount": _amount_values(labeled, sample.overdue_amount_col),
    }
    observations = _metric_observations(
        voting_mask,
        hit_count=hit_count,
        target=target,
        amount_values=amount_values,
        k=len(entries),
    )
    return {
        "effect": effect,
        "hit_distribution": distribution,
        "metric_observations": observations,
        "voting_mask": voting_mask,
        "labeled_frame": labeled,
    }


def _effect_from_mask(
    mask: np.ndarray,
    *,
    target: np.ndarray,
    population_count: int,
) -> dict[str, Any]:
    labeled_count = int(len(target))
    matched_count = int(mask.sum())
    unmatched = ~mask
    unmatched_count = int(unmatched.sum())
    matched_bad = int(target[mask].sum())
    unmatched_bad = int(target[unmatched].sum())
    total_bad = matched_bad + unmatched_bad
    matched_rate = _ratio(matched_count, labeled_count)
    matched_bad_rate = _ratio(matched_bad, matched_count)
    unmatched_bad_rate = _ratio(unmatched_bad, unmatched_count)
    bad_capture_rate = _ratio(matched_bad, total_bad)
    base_bad_rate = _ratio(total_bad, labeled_count)
    lift = (
        None
        if matched_bad_rate is None or base_bad_rate in {None, 0.0}
        else matched_bad_rate / base_bad_rate
    )
    return {
        "population_count": int(population_count),
        "labeled_count": labeled_count,
        "matched_count": matched_count,
        "matched_rate": matched_rate,
        "matched_bad_count": matched_bad,
        "matched_bad_rate": matched_bad_rate,
        "unmatched_count": unmatched_count,
        "unmatched_bad_count": unmatched_bad,
        "unmatched_bad_rate": unmatched_bad_rate,
        "bad_capture_rate": bad_capture_rate,
        "lift": lift,
    }


def _hit_distribution(
    hit_count: np.ndarray,
    *,
    target: np.ndarray,
    k: int,
) -> list[dict[str, Any]]:
    labeled_count = int(len(target))
    total_bad = int(target.sum())
    base_bad_rate = _ratio(total_bad, labeled_count)
    rows = []
    for value in range(k + 1):
        bucket = hit_count == value
        count = int(bucket.sum())
        bad_count = int(target[bucket].sum())
        bad_rate = _ratio(bad_count, count)
        lift = (
            None
            if bad_rate is None or base_bad_rate in {None, 0.0}
            else bad_rate / base_bad_rate
        )
        rows.append(
            {
                "hit_count": value,
                "count": count,
                "share": _ratio(count, labeled_count),
                "bad_count": bad_count,
                "bad_rate": bad_rate,
                "lift": lift,
            }
        )
    if sum(row["count"] for row in rows) != labeled_count:
        raise StrategyError("Voting hit distribution does not conserve rows")
    if sum(row["bad_count"] for row in rows) != total_bad:
        raise StrategyError("Voting hit distribution does not conserve bad rows")
    return rows


def _metric_observations(
    voting_mask: np.ndarray,
    *,
    hit_count: np.ndarray,
    target: np.ndarray,
    amount_values: Mapping[str, np.ndarray | None],
    k: int,
) -> list[dict[str, Any]]:
    observations: list[MetricObservation] = []
    bad_mask = target == 1
    for metric_name, selected, base in (
        ("voting.hit_share", voting_mask, np.ones(len(target), dtype=bool)),
        ("voting.bad_capture_rate", voting_mask & bad_mask, bad_mask),
    ):
        observations.extend(
            _share_observations(
                metric_name,
                selected=selected,
                base=base,
                amount_values=amount_values,
            )
        )
    observations.extend(
        _count_only_observations(
            "voting.matched_bad_rate",
            _ratio(int(target[voting_mask].sum()), int(voting_mask.sum())),
        )
    )
    for value in range(k + 1):
        bucket = hit_count == value
        observations.extend(
            _share_observations(
                f"voting.hit_count.{value}.share",
                selected=bucket,
                base=np.ones(len(target), dtype=bool),
                amount_values=amount_values,
            )
        )
        observations.extend(
            _count_only_observations(
                f"voting.hit_count.{value}.bad_rate",
                _ratio(int(target[bucket].sum()), int(bucket.sum())),
            )
        )
    return [observation.to_dict() for observation in observations]


def _share_observations(
    metric_name: str,
    *,
    selected: np.ndarray,
    base: np.ndarray,
    amount_values: Mapping[str, np.ndarray | None],
) -> list[MetricObservation]:
    result = [
        _observation(metric_name, "count", int(selected.sum()), int(base.sum()))
    ]
    for dimension in ("loan_amount", "overdue_amount"):
        values = amount_values[dimension]
        if values is None:
            result.append(
                MetricObservation(metric_name, dimension, "unavailable", None)
            )
            continue
        if not bool(base.any()):
            result.append(
                MetricObservation(metric_name, dimension, "not_applicable", None)
            )
            continue
        denominator_mask = base & np.isfinite(values)
        numerator_mask = selected & np.isfinite(values)
        if int(denominator_mask.sum()) != int(base.sum()):
            result.append(
                MetricObservation(metric_name, dimension, "insufficient_data", None)
            )
            continue
        denominator = float(values[denominator_mask].sum())
        numerator = float(values[numerator_mask].sum())
        result.append(
            _observation(metric_name, dimension, numerator, denominator)
        )
    return result


def _count_only_observations(
    metric_name: str, value: float | None
) -> list[MetricObservation]:
    status = "observed" if value is not None else "not_applicable"
    return [
        MetricObservation(metric_name, "count", status, value),
        MetricObservation(metric_name, "loan_amount", "not_applicable", None),
        MetricObservation(metric_name, "overdue_amount", "not_applicable", None),
    ]


def _observation(
    metric_name: str,
    dimension: str,
    numerator: int | float,
    denominator: int | float,
) -> MetricObservation:
    if denominator == 0:
        return MetricObservation(metric_name, dimension, "not_applicable", None)
    return MetricObservation(
        metric_name,
        dimension,
        "observed",
        float(numerator / denominator),
    )


def _amount_values(frame: pd.DataFrame, column: str | None) -> np.ndarray | None:
    if column is None:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    invalid = frame[column].notna().to_numpy() & ~np.isfinite(values)
    if np.any(invalid) or np.any(values[np.isfinite(values)] < 0):
        raise StrategyError(
            f"Voting amount column {column} must contain non-negative finite values or null"
        )
    return values


def _require_voting_mask_equivalence(
    expected: np.ndarray,
    *,
    frame: pd.DataFrame,
    condition: Mapping[str, Any],
) -> None:
    canonical = evaluate_expression_frame(frame, condition).to_numpy(
        dtype=bool, copy=False
    )
    if not np.array_equal(expected, canonical):
        raise StrategyError(
            "Voting member hit_count disagrees with canonical n_of_k evaluation"
        )


def _persist_voting_candidate(
    runtime,
    *,
    task_id: str,
    repository: StrategyCandidatePoolRepository,
    request: Mapping[str, Any],
    pool: Mapping[str, Any],
    pool_artifact: Mapping[str, Any],
    lineages: Sequence[Any],
    sample: _SampleBinding,
    resolved_requirements: ResolvedPoolRequirements | None,
    document: Mapping[str, Any],
) -> dict[str, Any]:
    from marvis.packs.strategy import pool_tools

    normalized = validate_voting_candidate_artifact_document(document)
    asset = normalized["asset"]
    canonical = canonical_voting_candidate_artifact_json(normalized).encode("utf-8")
    content_hash = _sha256(canonical)
    provenance = voting_candidate_artifact_provenance(
        normalized,
        task_id=task_id,
        pool_artifact=pool_artifact,
    )
    out_dir = _prepare_output_directory(
        runtime.settings.tasks_dir,
        task_id=task_id,
    )
    final_path = canonical_voting_candidate_path(
        runtime.settings.tasks_dir,
        task_id=task_id,
        asset_id=asset["asset_id"],
    )
    if final_path.parent != out_dir:
        raise StrategyError("Voting candidate output path drifted")
    uow = ArtifactUnitOfWork()
    try:
        staged = uow.stage_file(out_dir, final_path.name)
        staged.path.write_bytes(canonical)
    except OSError as exc:
        uow.rollback()
        raise StrategyError("Voting candidate artifact could not be staged") from exc
    db_committed = False
    rollback_attempted_under_lock = False
    record: Mapping[str, Any]
    try:
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                locked_pool = repository.get_current_on_connection(
                    conn,
                    task_id,
                    request["strategy_type"],
                )
                if locked_pool is None:
                    raise StrategyError("strategy candidate pool not found")
                locked_pool = validate_strategy_pool(locked_pool)
                _require_expected_pool(locked_pool, request)
                if locked_pool != pool:
                    raise StrategyError(
                        "strategy candidate pool changed before Voting registration"
                    )
                pool_binding = pool_tools._normalize_source_record(pool_artifact)
                pool_tools._require_parent_pool_artifact_on_connection(
                    conn,
                    pool_binding,
                    snapshot=pool,
                    tasks_root=Path(runtime.settings.tasks_dir),
                )
                cache = pool_tools._LineageCache.empty()
                for lineage in lineages:
                    pool_tools._require_lineage_on_connection(
                        conn,
                        lineage,
                        tasks_root=Path(runtime.settings.tasks_dir),
                        cache=cache,
                    )
                _require_file_hash(
                    sample.path,
                    sample.dataset_content_hash,
                    "Voting source dataset content hash changed before registration",
                )
                require_strategy_sample_design_execution_binding_on_connection(
                    conn,
                    sample.sample_design,
                )
                if resolved_requirements is not None:
                    require_resolved_pool_requirements_on_connection(
                        conn,
                        resolved_requirements,
                    )
                verify_voting_candidate_asset_against_pool(asset, locked_pool)
                _require_existing_artifact_consistent(
                    conn,
                    task_id=task_id,
                    final_path=final_path,
                    canonical=canonical,
                    content_hash=content_hash,
                    provenance=provenance,
                )
                uow.promote_all()
                _verify_artifact_file(
                    final_path,
                    root=Path(runtime.settings.tasks_dir).absolute(),
                    canonical=canonical,
                    content_hash=content_hash,
                )
                record = runtime.task_artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=VOTING_CANDIDATE_ARTIFACT_KIND,
                    path=str(final_path),
                    content_hash=content_hash,
                    origin_tool=VOTING_CANDIDATE_ORIGIN_TOOL,
                    provenance=provenance,
                )
                _require_registered_record(
                    record,
                    task_id=task_id,
                    final_path=final_path,
                    content_hash=content_hash,
                    provenance=provenance,
                )
                if resolved_requirements is not None:
                    require_resolved_pool_requirements_on_connection(
                        conn,
                        resolved_requirements,
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
    return _tool_output(normalized, record=record, task_id=task_id)


def _tool_output(
    document: Mapping[str, Any],
    *,
    record: Mapping[str, Any],
    task_id: str,
) -> dict[str, Any]:
    asset = document["asset"]
    measurement = document["measurement"]
    lifecycle = asset["lifecycle"]
    pool = asset["pool_ref"]
    return {
        "schema_version": TOOL_SCHEMA_VERSION,
        "asset_id": asset["asset_id"],
        "asset_hash": asset["asset_hash"],
        "candidate_id": asset["candidate_evidence"]["candidate_id"],
        "evidence_hash": asset["candidate_evidence"]["evidence_hash"],
        "rule_id": asset["rule"]["rule_id"],
        "rule_hash": asset["rule"]["rule_hash"],
        "fragment_id": asset["fragment"]["fragment_id"],
        "fragment_hash": asset["fragment"]["fragment_hash"],
        "effect_id": asset["effect"]["effect_id"],
        "effect_hash": asset["effect"]["effect_hash"],
        "pool_id": pool["pool_id"],
        "revision": pool["revision"],
        "snapshot_hash": pool["snapshot_hash"],
        "n": asset["voting"]["n"],
        "k": asset["voting"]["k"],
        "selected_entries": [
            {
                "pool_position": entry["pool_position"],
                "entry_id": entry["entry_id"],
                "rule_id": entry["rule_id"],
            }
            for entry in asset["selected_entries"]
        ],
        "dataset_id": asset["evidence_identity"]["dataset_id"],
        "sample_design_ref": asset["sample_design_ref"],
        "target_col": asset["measurement_context"]["target_col"],
        "drop_nan_labels": measurement["drop_nan_labels"],
        "nan_labels_dropped": measurement["nan_labels_dropped"],
        "population_count": measurement["population_count"],
        "labeled_count": measurement["labeled_count"],
        "candidate_stage": lifecycle["candidate_stage"],
        "observation_stage": lifecycle["observation_stage"],
        "validation_status": lifecycle["validation_status"],
        "effect": asset["effect"],
        "metrics": asset["metrics"],
        "hit_distribution": measurement["hit_distribution"],
        "metric_observations": measurement["metric_observations"],
        "not_admitted": True,
        "not_applied": True,
        "not_adopted": True,
        "not_deployed": True,
        "artifacts": [_artifact_output(record, task_id=task_id)],
    }


def _load_verified_voting_record(
    record: Mapping[str, Any],
    *,
    tasks_dir: Path | str,
    task_id: str,
    artifact_id: str,
    expected_content_hash: str,
    expected_asset_id: str,
    expected_asset_hash: str,
    raw_provenance: bool = False,
) -> VerifiedVotingCandidateArtifact:
    normalized_task = _required_text(task_id, "task_id")
    normalized_artifact = _required_text(artifact_id, "artifact_id")
    normalized_content_hash = _required_hash(
        expected_content_hash, "expected_content_hash"
    )
    normalized_asset_id = _required_text(expected_asset_id, "expected_asset_id")
    normalized_asset_hash = _required_hash(expected_asset_hash, "expected_asset_hash")
    expected_path = canonical_voting_candidate_path(
        tasks_dir,
        task_id=normalized_task,
        asset_id=normalized_asset_id,
    )
    expected = {
        "id": normalized_artifact,
        "task_id": normalized_task,
        "kind": VOTING_CANDIDATE_ARTIFACT_KIND,
        "path": str(expected_path),
        "content_hash": normalized_content_hash,
        "origin_tool": VOTING_CANDIDATE_ORIGIN_TOOL,
    }
    for field, expected_value in expected.items():
        actual = record.get(field)
        if actual != expected_value:
            raise StrategyError(f"Voting candidate registry {field} changed")
    provenance = (
        _strict_json_object(record.get("provenance_json"), "provenance_json")
        if raw_provenance
        else _json_object(record.get("provenance"), "registry provenance")
    )
    _verify_artifact_file(
        expected_path,
        root=Path(tasks_dir).absolute(),
        content_hash=normalized_content_hash,
    )
    raw = expected_path.read_bytes()
    document = _parse_artifact_document(raw)
    expected_provenance_fields = (
        _PROVENANCE_FIELDS
        if document["schema_version"] == VOTING_CANDIDATE_ARTIFACT_SCHEMA_VERSION
        else _PROVENANCE_FIELDS_V1
    )
    _exact_fields(
        provenance,
        expected_provenance_fields,
        "Voting artifact provenance",
    )
    canonical = canonical_voting_candidate_artifact_json(document).encode("utf-8")
    if not hmac.compare_digest(raw, canonical):
        raise StrategyError("Voting candidate artifact is not canonical JSON")
    asset = document["asset"]
    if asset["asset_id"] != normalized_asset_id or not hmac.compare_digest(
        asset["asset_hash"], normalized_asset_hash
    ):
        raise StrategyError("Voting candidate asset binding changed")
    minimal_pool_binding = {
        "id": provenance["pool_artifact_id"],
        "content_hash": provenance["pool_artifact_content_hash"],
    }
    expected_provenance = voting_candidate_artifact_provenance(
        document,
        task_id=normalized_task,
        pool_artifact=minimal_pool_binding,
    )
    if provenance != expected_provenance:
        raise StrategyError("Voting candidate artifact provenance changed")
    return VerifiedVotingCandidateArtifact(
        artifact_id=normalized_artifact,
        task_id=normalized_task,
        kind=VOTING_CANDIDATE_ARTIFACT_KIND,
        path=expected_path,
        content_hash=normalized_content_hash,
        origin_tool=VOTING_CANDIDATE_ORIGIN_TOOL,
        provenance=provenance,
        canonical_bytes=canonical,
        document=document,
        asset=asset,
    )


def _parse_artifact_document(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=_object_without_duplicate_keys)
        if not isinstance(value, dict):
            raise StrategyError("Voting candidate artifact JSON must contain an object")
        return validate_voting_candidate_artifact_document(value)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise StrategyError("Voting candidate artifact is invalid JSON") from exc


def _require_existing_artifact_consistent(
    conn,
    *,
    task_id: str,
    final_path: Path,
    canonical: bytes,
    content_hash: str,
    provenance: Mapping[str, Any],
) -> None:
    row = conn.execute(
        """
        SELECT id, task_id, kind, path, content_hash, origin_tool,
               provenance_json, created_at
          FROM task_artifacts
         WHERE task_id = ? AND kind = ? AND path = ?
        """,
        (task_id, VOTING_CANDIDATE_ARTIFACT_KIND, str(final_path)),
    ).fetchone()
    if row is None:
        if final_path.exists() or final_path.is_symlink():
            raise StrategyError("Voting candidate path exists without a registry row")
        return
    record = {field: row[field] for field in _TASK_ARTIFACT_ROW_FIELDS}
    _require_registered_record(
        record,
        task_id=task_id,
        final_path=final_path,
        content_hash=content_hash,
        provenance=provenance,
        raw_provenance=True,
    )
    _verify_artifact_file(
        final_path,
        root=final_path.parents[2],
        canonical=canonical,
        content_hash=content_hash,
    )


def _require_registered_record(
    record: Mapping[str, Any],
    *,
    task_id: str,
    final_path: Path,
    content_hash: str,
    provenance: Mapping[str, Any],
    raw_provenance: bool = False,
) -> None:
    expected = {
        "task_id": task_id,
        "kind": VOTING_CANDIDATE_ARTIFACT_KIND,
        "path": str(final_path),
        "content_hash": content_hash,
        "origin_tool": VOTING_CANDIDATE_ORIGIN_TOOL,
    }
    for field, expected_value in expected.items():
        if record.get(field) != expected_value:
            raise StrategyError(f"Voting candidate registry {field} changed")
    actual_provenance = (
        _strict_json_object(record.get("provenance_json"), "provenance_json")
        if raw_provenance
        else _json_object(record.get("provenance"), "registry provenance")
    )
    if actual_provenance != provenance:
        raise StrategyError("Voting candidate registry provenance changed")


def _artifact_output(record: Mapping[str, Any], *, task_id: str) -> dict[str, Any]:
    path = Path(_required_text(record.get("path"), "artifact path"))
    artifact_id = _required_text(record.get("id"), "artifact id")
    return {
        "artifact_id": artifact_id,
        "kind": VOTING_CANDIDATE_ARTIFACT_KIND,
        "format": "json",
        "filename": path.name,
        "content_hash": _required_hash(record.get("content_hash"), "content_hash"),
        "download_url": (
            f"/api/tasks/{quote(task_id, safe='')}"
            f"/task-artifacts/{quote(artifact_id, safe='')}/download"
        ),
    }


def _normalize_measurement_body(
    value: Mapping[str, Any],
    *,
    asset: Mapping[str, Any],
) -> dict[str, Any]:
    is_current = asset["schema_version"] == VOTING_CANDIDATE_ASSET_SCHEMA_VERSION
    fields = _MEASUREMENT_FIELDS if is_current else _MEASUREMENT_FIELDS_V1
    expected = fields - {"measurement_hash"}
    _exact_fields(value, expected, "Voting measurement body")
    expected_schema = (
        VOTING_MEASUREMENT_SCHEMA_VERSION
        if is_current
        else VOTING_MEASUREMENT_SCHEMA_VERSION_V1
    )
    if value["schema_version"] != expected_schema:
        raise StrategyError("Voting measurement schema_version is invalid")
    sample_design_ref = None
    if is_current:
        sample_design_ref = StrategySampleDesignRef.from_value(
            value["sample_design_ref"]
        ).to_ref_dict()
        if sample_design_ref != asset["sample_design_ref"]:
            raise StrategyError(
                "Voting measurement sample_design_ref does not match asset"
            )
    target_col = _required_text(value["target_col"], "measurement target_col")
    if target_col != asset["measurement_context"]["target_col"]:
        raise StrategyError("Voting measurement target_col does not match asset")
    drop_nan_labels = value["drop_nan_labels"]
    if not isinstance(drop_nan_labels, bool):
        raise StrategyError("Voting measurement drop_nan_labels must be boolean")
    nan_labels_dropped = _non_negative_int(
        value["nan_labels_dropped"], "measurement nan_labels_dropped"
    )
    if not drop_nan_labels and nan_labels_dropped != 0:
        raise StrategyError(
            "Voting measurement cannot drop labels when drop_nan_labels is false"
        )
    population_count = _positive_int(
        value["population_count"], "measurement population_count"
    )
    labeled_count = _non_negative_int(
        value["labeled_count"], "measurement labeled_count"
    )
    if labeled_count > population_count:
        raise StrategyError("Voting measurement labeled_count exceeds population_count")
    if population_count - labeled_count != nan_labels_dropped:
        raise StrategyError(
            "Voting measurement dropped-label count does not conserve population"
        )
    effect = asset["effect"]
    if (
        population_count != effect["population_count"]
        or labeled_count != effect["labeled_count"]
    ):
        raise StrategyError("Voting measurement population does not match asset effect")
    distribution = _normalize_distribution(
        value["hit_distribution"],
        k=asset["voting"]["k"],
        labeled_count=labeled_count,
        total_bad=effect["matched_bad_count"] + effect["unmatched_bad_count"],
    )
    _require_distribution_matches_effect(
        distribution,
        n=asset["voting"]["n"],
        effect=effect,
    )
    observations = _normalize_metric_observations(
        value["metric_observations"],
        effect=effect,
        distribution=distribution,
    )
    result = {
        "schema_version": expected_schema,
        "target_col": target_col,
        "drop_nan_labels": drop_nan_labels,
        "nan_labels_dropped": nan_labels_dropped,
        "population_count": population_count,
        "labeled_count": labeled_count,
        "hit_distribution": distribution,
        "metric_observations": observations,
    }
    if sample_design_ref is not None:
        result["sample_design_ref"] = sample_design_ref
    return result


def _normalize_distribution(
    value: object,
    *,
    k: int,
    labeled_count: int,
    total_bad: int,
) -> list[dict[str, Any]]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        raise StrategyError("Voting hit_distribution must be a list")
    if len(value) != k + 1:
        raise StrategyError("Voting hit_distribution must contain every 0..K bucket")
    base_bad_rate = _ratio(total_bad, labeled_count)
    result = []
    for expected_hit, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise StrategyError("Voting hit_distribution row must be an object")
        _exact_fields(item, _DISTRIBUTION_FIELDS, "Voting hit_distribution row")
        hit = _non_negative_int(item["hit_count"], "hit_count")
        if hit != expected_hit:
            raise StrategyError("Voting hit_distribution must be ordered from 0..K")
        count = _non_negative_int(item["count"], "distribution count")
        bad_count = _non_negative_int(item["bad_count"], "distribution bad_count")
        if bad_count > count:
            raise StrategyError("Voting distribution bad_count exceeds count")
        share = _derived_optional(
            item["share"], _ratio(count, labeled_count), "distribution share"
        )
        bad_rate = _derived_optional(
            item["bad_rate"], _ratio(bad_count, count), "distribution bad_rate"
        )
        expected_lift = (
            None
            if bad_rate is None or base_bad_rate in {None, 0.0}
            else bad_rate / base_bad_rate
        )
        lift = _derived_optional(item["lift"], expected_lift, "distribution lift")
        result.append(
            {
                "hit_count": hit,
                "count": count,
                "share": share,
                "bad_count": bad_count,
                "bad_rate": bad_rate,
                "lift": lift,
            }
        )
    if sum(row["count"] for row in result) != labeled_count:
        raise StrategyError("Voting hit_distribution count does not conserve rows")
    if sum(row["bad_count"] for row in result) != total_bad:
        raise StrategyError("Voting hit_distribution bad_count does not conserve bads")
    return result


def _require_distribution_matches_effect(
    distribution: Sequence[Mapping[str, Any]],
    *,
    n: int,
    effect: Mapping[str, Any],
) -> None:
    matched = [row for row in distribution if row["hit_count"] >= n]
    unmatched = [row for row in distribution if row["hit_count"] < n]
    expected = {
        "matched_count": sum(row["count"] for row in matched),
        "matched_bad_count": sum(row["bad_count"] for row in matched),
        "unmatched_count": sum(row["count"] for row in unmatched),
        "unmatched_bad_count": sum(row["bad_count"] for row in unmatched),
    }
    changed = [field for field, value in expected.items() if effect[field] != value]
    if changed:
        raise StrategyError(
            "Voting hit_distribution threshold aggregation does not match effect: "
            + ", ".join(changed)
        )


def _normalize_metric_observations(
    value: object,
    *,
    effect: Mapping[str, Any],
    distribution: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        raise StrategyError("Voting metric_observations must be a list")
    by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for item in value:
        if not isinstance(item, Mapping):
            raise StrategyError("Voting metric observation must be an object")
        try:
            normalized = MetricObservation(
                metric_name=item.get("metric_name"),
                dimension=item.get("dimension"),
                status=item.get("status"),
                value=item.get("value"),
            ).to_dict()
        except (TypeError, ValueError, StrategyError) as exc:
            raise StrategyError("Voting metric observation is invalid") from exc
        if set(item) != set(normalized):
            raise StrategyError("Voting metric observation has unsupported fields")
        identity = (normalized["metric_name"], normalized["dimension"])
        if identity in by_identity:
            raise StrategyError("Voting metric observation identity is duplicated")
        by_identity[identity] = normalized

    metric_names = [
        "voting.hit_share",
        "voting.bad_capture_rate",
        "voting.matched_bad_rate",
    ]
    for row in distribution:
        hit_count = row["hit_count"]
        metric_names.extend(
            (
                f"voting.hit_count.{hit_count}.share",
                f"voting.hit_count.{hit_count}.bad_rate",
            )
        )
    dimensions = ("count", "loan_amount", "overdue_amount")
    expected_identities = [
        (metric_name, dimension)
        for metric_name in metric_names
        for dimension in dimensions
    ]
    expected_set = set(expected_identities)
    actual_set = set(by_identity)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        unexpected = sorted(actual_set - expected_set)
        details = []
        if missing:
            details.append("missing: " + ", ".join(f"{a}/{b}" for a, b in missing))
        if unexpected:
            details.append(
                "unexpected: " + ", ".join(f"{a}/{b}" for a, b in unexpected)
            )
        raise StrategyError(
            "Voting metric_observations identities are incomplete ("
            + "; ".join(details)
            + ")"
        )

    count_values: dict[str, float | None] = {
        "voting.hit_share": effect["matched_rate"],
        "voting.bad_capture_rate": effect["bad_capture_rate"],
        "voting.matched_bad_rate": effect["matched_bad_rate"],
    }
    count_only = {"voting.matched_bad_rate"}
    share_metrics = {"voting.hit_share", "voting.bad_capture_rate"}
    for row in distribution:
        hit_count = row["hit_count"]
        share_name = f"voting.hit_count.{hit_count}.share"
        bad_rate_name = f"voting.hit_count.{hit_count}.bad_rate"
        count_values[share_name] = row["share"]
        count_values[bad_rate_name] = row["bad_rate"]
        share_metrics.add(share_name)
        count_only.add(bad_rate_name)

    for metric_name, expected_value in count_values.items():
        observation = by_identity[(metric_name, "count")]
        expected_status = "observed" if expected_value is not None else "not_applicable"
        if observation["status"] != expected_status or not _same_optional_number(
            observation["value"], expected_value
        ):
            raise StrategyError(
                f"Voting metric observation {metric_name}/count does not match "
                "deterministic measurement"
            )
    for metric_name in count_only:
        for dimension in ("loan_amount", "overdue_amount"):
            observation = by_identity[(metric_name, dimension)]
            if observation["status"] != "not_applicable" or observation["value"] is not None:
                raise StrategyError(
                    f"Voting metric observation {metric_name}/{dimension} "
                    "must be not_applicable"
                )
    for metric_name in share_metrics:
        for dimension in ("loan_amount", "overdue_amount"):
            observation = by_identity[(metric_name, dimension)]
            if observation["status"] == "observed" and not (
                0.0 <= float(observation["value"]) <= 1.0
            ):
                raise StrategyError(
                    f"Voting metric observation {metric_name}/{dimension} "
                    "must be a rate between 0 and 1"
                )
    for dimension in ("loan_amount", "overdue_amount"):
        unavailable = {
            metric_name
            for metric_name in share_metrics
            if by_identity[(metric_name, dimension)]["status"] == "unavailable"
        }
        if unavailable and unavailable != share_metrics:
            raise StrategyError(
                f"Voting metric observation {dimension} availability is inconsistent"
            )
    return [by_identity[identity] for identity in expected_identities]


def _same_optional_number(actual: object, expected: object) -> bool:
    if actual is None or expected is None:
        return actual is expected
    return math.isclose(
        float(actual),
        float(expected),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def _prepare_output_directory(tasks_dir: Path | str, *, task_id: str) -> Path:
    task = _safe_component(task_id, "task_id")
    root = Path(tasks_dir).absolute()
    output = root / task / "strategy_voting_candidates"
    try:
        if root.is_symlink():
            raise StrategyError("task artifact root must not be a symlink")
        root.mkdir(parents=True, exist_ok=True)
        root_resolved = root.resolve(strict=True)
        task_dir = root / task
        if task_dir.is_symlink():
            raise StrategyError("task artifact directory must not be a symlink")
        task_dir.mkdir(exist_ok=True)
        if task_dir.resolve(strict=True).parent != root_resolved:
            raise StrategyError("Voting task directory escaped task storage")
        if output.is_symlink():
            raise StrategyError("Voting output directory must not be a symlink")
        output.mkdir(exist_ok=True)
        if output.resolve(strict=True).parent != task_dir.resolve(strict=True):
            raise StrategyError("Voting output directory escaped task storage")
    except OSError as exc:
        raise StrategyError("Voting output directory is unavailable") from exc
    return output


def _verify_artifact_file(
    path: Path,
    *,
    root: Path,
    content_hash: str,
    canonical: bytes | None = None,
) -> None:
    _require_regular_path(path, root=root)
    before = path.lstat()
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise StrategyError("Voting candidate artifact could not be read") from exc
    _require_regular_path(path, root=root)
    after = path.lstat()
    if _stat_identity(before) != _stat_identity(after):
        raise StrategyError("Voting candidate artifact changed while read")
    if not hmac.compare_digest(_sha256(raw), content_hash):
        raise StrategyError("Voting candidate artifact content hash changed")
    if canonical is not None and not hmac.compare_digest(raw, canonical):
        raise StrategyError("Voting candidate artifact bytes changed")
    parsed = _parse_artifact_document(raw)
    if canonical_voting_candidate_artifact_json(parsed).encode("utf-8") != raw:
        raise StrategyError("Voting candidate artifact is not canonical JSON")


def _require_regular_path(path: Path, *, root: Path) -> None:
    absolute = path.absolute()
    declared_root = root.absolute()
    try:
        absolute.relative_to(declared_root)
    except ValueError as exc:
        raise StrategyError("Voting candidate artifact escaped task storage") from exc
    current = absolute
    while True:
        if current.is_symlink():
            raise StrategyError("Voting candidate artifact must not use symlinks")
        if current == declared_root:
            break
        if current == current.parent:
            raise StrategyError("Voting candidate artifact escaped task storage")
        current = current.parent
    try:
        mode = absolute.stat().st_mode
    except OSError as exc:
        raise StrategyError("Voting candidate artifact is unavailable") from exc
    if not stat.S_ISREG(mode):
        raise StrategyError("Voting candidate artifact must be a regular file")


def _require_file_hash(path: Path, expected: str, message: str) -> None:
    try:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise StrategyError(message) from exc
    if not hmac.compare_digest(actual, expected):
        raise StrategyError(message)


def _expression_fields(expression: Mapping[str, Any]) -> set[str]:
    canonical = canonicalize_expression(expression)
    op = canonical["op"]
    if op in {"compare", "between", "is_null", "is_not_null"}:
        return {canonical["field"]}
    if op in {"and", "or", "n_of_k"}:
        fields: set[str] = set()
        for argument in canonical["args"]:
            fields.update(_expression_fields(argument))
        return fields
    if op == "not":
        return _expression_fields(canonical["arg"])
    raise StrategyError(f"unsupported Voting expression op: {op}")


def _safe_component(value: object, field: str) -> str:
    text = _required_text(value, field)
    if Path(text).name != text or text in {".", ".."}:
        raise StrategyError(f"{field} is unsafe for artifact storage")
    return text


def _exact_fields(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    if not isinstance(value, Mapping):
        raise StrategyError(f"{name} must be an object")
    missing = sorted(expected - set(value))
    unsupported = sorted(set(value) - expected)
    if missing or unsupported:
        detail = []
        if missing:
            detail.append("missing: " + ", ".join(missing))
        if unsupported:
            detail.append("unsupported: " + ", ".join(unsupported))
        raise StrategyError(f"{name} fields are invalid ({'; '.join(detail)})")


def _required_text(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or "\x00" in value
    ):
        raise StrategyError(f"{field} must be non-empty canonical text")
    return value


def _optional_column(value: object, field: str) -> str | None:
    return None if value is None else _required_text(value, field)


def _required_hash(value: object, field: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise StrategyError(f"{field} must be a lowercase SHA-256 hash")
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise StrategyError(f"{field} must be a positive integer")
    return value


def _non_negative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StrategyError(f"{field} must be a non-negative integer")
    return value


def _text_list(value: object, field: str) -> list[str]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        raise StrategyError(f"{field} must be a list")
    return [_required_text(item, f"{field} item") for item in value]


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _derived_optional(value: object, expected: float | None, field: str) -> float | None:
    if expected is None:
        if value is not None:
            raise StrategyError(f"{field} must be null when denominator is zero")
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise StrategyError(f"{field} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized) or not math.isclose(
        normalized, expected, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise StrategyError(f"{field} does not match deterministic measurement")
    return normalized


def _json_object(value: object, field: str) -> dict[str, Any]:
    try:
        normalized = json.loads(_canonical_json(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StrategyError(f"{field} must be a finite JSON object") from exc
    if not isinstance(normalized, dict):
        raise StrategyError(f"{field} must be an object")
    return normalized


def _strict_json_object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise StrategyError(f"{field} must be canonical JSON text")
    try:
        parsed = json.loads(value, object_pairs_hook=_object_without_duplicate_keys)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise StrategyError(f"{field} is invalid JSON") from exc
    if not isinstance(parsed, dict) or _canonical_json(parsed) != value:
        raise StrategyError(f"{field} must be canonical JSON object text")
    return parsed


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrategyError(f"JSON contains duplicate key: {key}")
        result[key] = value
    return result


def _stat_identity(value) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


__all__ = [
    "TOOL_SCHEMA_VERSION",
    "VOTING_CANDIDATE_ARTIFACT_KIND",
    "VOTING_CANDIDATE_ARTIFACT_SCHEMA_VERSION",
    "VOTING_CANDIDATE_ORIGIN_TOOL",
    "VOTING_MEASUREMENT_SCHEMA_VERSION",
    "VerifiedVotingCandidateArtifact",
    "build_voting_candidate_artifact_document",
    "canonical_voting_candidate_artifact_json",
    "canonical_voting_candidate_path",
    "load_verified_voting_candidate_artifact",
    "load_verified_voting_candidate_artifact_on_connection",
    "require_voting_snapshot_marginal_reachability",
    "run_build_voting_candidate",
    "validate_voting_candidate_artifact_document",
    "voting_candidate_artifact_provenance",
]
