"""Governed Tool boundary for bounded automatic two-dimensional Cross search.

The search Tool authenticates one exact univariate CandidateEvidence artifact,
binds only its risk/development population, selects each explicitly named
feature's highest-ranked available method, replays every selected axis once,
and persists aggregate pair evidence.  A separate pointer Tool authenticates
``search_id + pair_id`` and asks the existing explicit Cross builder to
recompute the full asset under the same writer lock.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote

from marvis.artifacts import ArtifactUnitOfWork
from marvis.packs.strategy import candidate_asset_tools
from marvis.packs.strategy.candidate_fragment import (
    sample_context_hash_from_candidate_evidence,
)
from marvis.packs.strategy.cross_candidate_search import (
    CROSS_CANDIDATE_SEARCH_PRODUCER_VERSION,
    CROSS_CANDIDATE_SEARCH_REQUEST_SCHEMA_VERSION,
    MAX_ARTIFACT_BYTES,
    MAX_AXIS_BIN_ROW_EVALUATIONS,
    MAX_CELLS_PER_PAIR,
    MAX_DERIVED_CELLS,
    MAX_FEATURES,
    MAX_PAIR_ROW_EVALUATIONS,
    MAX_PAIRS,
    asset_fingerprint,
    canonical_cross_candidate_search_result_json,
    canonical_pair_prefix,
    parse_cross_candidate_search_result_json,
    search_cross_candidate_pairs,
    validate_cross_candidate_search_result,
)
from marvis.packs.strategy.cross_matrix_candidate import (
    build_cross_matrix_candidate_asset,
    rebuild_cross_matrix_candidate_asset,
    validate_cross_matrix_candidate_asset,
)
from marvis.packs.strategy.cross_matrix_candidate_tools import (
    CROSS_MATRIX_MAX_CELLS,
    _binary_target,
    _load_exact_parent_evidence,
    _measure_matrix,
    _replay_axis,
    _resolve_axis,
    _resolve_exact_labeled_sample,
    _resolve_projection,
    _sample_identity,
    run_build_cross_matrix_candidate_with_registration_guard,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.sample_design_execution import (
    StrategyRiskDevelopmentExecutionBinding,
    bind_strategy_risk_development_frame,
    load_historical_strategy_risk_development_execution_binding,
    require_historical_strategy_risk_development_execution_binding_on_connection,
    revalidate_historical_strategy_risk_development_execution_binding,
)


CROSS_CANDIDATE_SEARCH_TOOL_SCHEMA_VERSION = (
    "strategy.search-cross-matrix-candidates-tool.v1"
)
CROSS_CANDIDATE_SEARCH_SELECTION_TOOL_SCHEMA_VERSION = (
    "strategy.build-cross-matrix-candidate-from-search-tool.v1"
)
CROSS_CANDIDATE_SEARCH_ARTIFACT_KIND = "strategy_cross_candidate_search_json"
CROSS_CANDIDATE_SEARCH_ARTIFACT_SCHEMA_VERSION = (
    "strategy.cross-candidate-search-artifact.v1"
)
CROSS_CANDIDATE_SEARCH_ORIGIN_TOOL = "strategy.search_cross_matrix_candidates"

_SEARCH_INPUT_FIELDS = frozenset(
    {
        "source_artifact_id",
        "expected_artifact_content_hash",
        "expected_candidate_id",
        "expected_evidence_hash",
        "features",
        "max_pairs",
    }
)
_SELECTION_INPUT_FIELDS = frozenset({"search_id", "pair_id"})
_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "task_id",
        "search_id",
        "search_content_hash",
        "request_hash",
        "source_artifact_id",
        "source_artifact_content_hash",
        "candidate_id",
        "evidence_hash",
        "dataset_id",
        "dataset_content_hash",
        "registry_metadata_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "sample_design_ref",
        "sample_context_hash",
        "sample_partition",
        "target_col",
        "drop_nan_labels",
        "nan_labels_dropped",
        "labeled_count",
        "features",
        "max_pairs",
        "lifecycle",
    }
)
_LIFECYCLE = {
    "selected": False,
    "admitted": False,
    "applied": False,
    "adopted": False,
    "deployed": False,
}
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_ID_RE = re.compile(r"^candidate-[0-9a-f]{32}$")
_SEARCH_ID_RE = re.compile(r"^cross-search-[0-9a-f]{32}$")
_PAIR_ID_RE = re.compile(r"^cross-pair-[0-9a-f]{32}$")


@dataclass(frozen=True)
class CrossCandidateSearchArtifactBinding:
    """One authenticated aggregate search plus its immutable source chain."""

    task_id: str
    artifact_id: str
    artifact_path: Path
    artifact_content_hash: str
    artifact_provenance: dict[str, Any]
    artifact_provenance_json: str
    result: dict[str, Any]
    source: Any
    dataset: Any
    sample_binding: StrategyRiskDevelopmentExecutionBinding
    evidence: dict[str, Any]
    tasks_root: Path
    db_path: Path


def run_search_cross_matrix_candidates(inputs, ctx, runtime) -> dict[str, Any]:
    """Persist aggregate evidence for one bounded canonical feature-pair prefix."""

    request = _validate_search_inputs(inputs)
    task_id = _text(ctx.task_id, "task_id")
    source = candidate_asset_tools._load_source_artifact(
        runtime,
        task_id=task_id,
        artifact_id=request["source_artifact_id"],
        expected_content_hash=request["expected_artifact_content_hash"],
        expected_candidate_id=request["expected_candidate_id"],
        expected_evidence_hash=request["expected_evidence_hash"],
    )
    evidence = _load_exact_parent_evidence(
        source,
        task_id=task_id,
        expected_candidate_id=request["expected_candidate_id"],
        expected_evidence_hash=request["expected_evidence_hash"],
    )
    dataset = candidate_asset_tools._load_dataset_binding(
        runtime,
        evidence=evidence,
        source=source,
    )
    sample_binding = _load_sample_binding(
        runtime,
        task_id=task_id,
        evidence=evidence,
        dataset=dataset,
    )
    features, axes = _select_ranked_axes(
        evidence,
        dataset=dataset,
        requested_features=request["features"],
    )
    governed = sorted(
        set(request["features"]) & set(sample_binding.excluded_feature_columns)
    )
    if governed:
        raise StrategyError(
            "Cross search features cannot use target, sample partition, or "
            "population columns: "
            + ", ".join(governed)
        )
    pairs = canonical_pair_prefix(
        [item["feature"] for item in features],
        max_pairs=request["max_pairs"],
    )
    _require_planned_budget(
        row_count=int(evidence["analysis"]["row_count"]),
        features=features,
        pairs=pairs,
    )
    labeled, target, projection = _read_search_sample(
        runtime,
        evidence=evidence,
        source=source,
        dataset=dataset,
        sample_binding=sample_binding,
        features=features,
    )
    assignments = {
        item["feature"]: _replay_axis(
            labeled,
            target=target,
            axis=axes[item["feature"]],
            axis_label=f"search/{item['feature']}",
        )
        for item in features
    }
    trials: list[dict[str, Any]] = []
    first_asset: dict[str, Any] | None = None
    for x_feature, y_feature in pairs:
        row_axis = axes[x_feature]
        column_axis = axes[y_feature]
        measurement = _measure_matrix(
            labeled,
            evidence=evidence,
            target=target,
            row_axis=row_axis,
            column_axis=column_axis,
            row_index=assignments[x_feature],
            column_index=assignments[y_feature],
            loan_amount_col=projection["loan_amount_col"],
            overdue_amount_col=projection["overdue_amount_col"],
        )
        asset = build_cross_matrix_candidate_asset(
            evidence,
            row_axis={
                "feature": row_axis["feature"],
                "method": row_axis["method"],
            },
            column_axis={
                "feature": column_axis["feature"],
                "method": column_axis["method"],
            },
            sample_identity=_sample_identity(evidence),
            measurement=measurement,
            budget=CROSS_MATRIX_MAX_CELLS,
        )
        asset = validate_cross_matrix_candidate_asset(asset)
        if rebuild_cross_matrix_candidate_asset(asset, evidence) != asset:
            raise StrategyError(
                "Cross search pair asset does not rebuild against exact evidence"
            )
        if first_asset is None:
            first_asset = asset
        cell_counts = [
            int(item["effect"]["count"]) for item in asset["matrix"]["cells"]
        ]
        nonempty = [count for count in cell_counts if count > 0]
        trials.append(
            {
                "x_feature": x_feature,
                "y_feature": y_feature,
                "cross_total_iv": asset["summary"]["total_iv"],
                "cell_count": asset["matrix"]["cell_count"],
                "empty_cell_count": sum(1 for count in cell_counts if count == 0),
                "min_nonempty_cell_count": min(nonempty) if nonempty else 0,
                "asset_fingerprint": asset_fingerprint(asset),
            }
        )
    if first_asset is None:
        raise StrategyError("Cross search did not evaluate any feature pair")
    core_request = {
        "schema_version": CROSS_CANDIDATE_SEARCH_REQUEST_SCHEMA_VERSION,
        "source": {
            "candidate_id": evidence["candidate_id"],
            "evidence_hash": evidence["evidence_hash"],
            "sample_context_hash": sample_context_hash_from_candidate_evidence(
                evidence
            ),
        },
        "population": {
            "row_count": len(labeled),
            "good": int(len(labeled) - target.sum()),
            "bad": int(target.sum()),
        },
        "features": features,
        "pair_trials": trials,
        "max_pairs": request["max_pairs"],
    }
    result = search_cross_candidate_pairs(core_request)
    candidate_asset_tools._require_source_unchanged(runtime, source)
    candidate_asset_tools._require_dataset_unchanged(runtime, dataset)
    revalidate_historical_strategy_risk_development_execution_binding(
        runtime,
        sample_binding,
    )
    return _persist_search(
        runtime,
        task_id=task_id,
        source=source,
        dataset=dataset,
        sample_binding=sample_binding,
        evidence=evidence,
        result=result,
    )


def run_build_cross_matrix_candidate_from_search(
    inputs,
    ctx,
    runtime,
) -> dict[str, Any]:
    """Recompute and publish one exact authenticated search pair."""

    request = _validate_selection_inputs(inputs)
    task_id = _text(ctx.task_id, "task_id")
    binding, pair = resolve_cross_candidate_search_pair(
        runtime,
        task_id=task_id,
        search_id=request["search_id"],
        pair_id=request["pair_id"],
    )

    def require_search(conn) -> None:
        require_cross_candidate_search_artifact_binding_on_connection(
            conn,
            binding,
        )

    provenance = binding.artifact_provenance
    built = run_build_cross_matrix_candidate_with_registration_guard(
        {
            "source_artifact_id": provenance["source_artifact_id"],
            "expected_artifact_content_hash": provenance[
                "source_artifact_content_hash"
            ],
            "expected_candidate_id": provenance["candidate_id"],
            "expected_evidence_hash": provenance["evidence_hash"],
            "x_feature": pair["x_feature"],
            "x_method": pair["x_method"],
            "y_feature": pair["y_feature"],
            "y_method": pair["y_method"],
        },
        ctx,
        runtime,
        expected_asset_fingerprint=pair["asset_fingerprint"],
        registration_guard=require_search,
    )
    return {
        "schema_version": CROSS_CANDIDATE_SEARCH_SELECTION_TOOL_SCHEMA_VERSION,
        "source_search_selection": {
            "search_id": binding.result["search_id"],
            "pair_id": pair["pair_id"],
            "rank": pair["rank"],
            "x_feature": pair["x_feature"],
            "x_method": pair["x_method"],
            "y_feature": pair["y_feature"],
            "y_method": pair["y_method"],
            "eligible": pair["eligible"],
        },
        "cross_matrix_candidate": built,
        "not_selected": True,
        "not_admitted": built["not_admitted"],
        "not_applied": built["not_applied"],
        "not_adopted": built["not_adopted"],
        "not_deployed": built["not_deployed"],
    }


def resolve_cross_candidate_search_pair(
    runtime,
    *,
    task_id: str,
    search_id: str,
    pair_id: str,
) -> tuple[CrossCandidateSearchArtifactBinding, dict[str, Any]]:
    """Resolve one pair from exactly one authenticated task-owned search."""

    task = _text(task_id, "task_id")
    search = _search_id(search_id, "search_id")
    pair = _pair_id(pair_id, "pair_id")
    matches: list[CrossCandidateSearchArtifactBinding] = []
    for record in runtime.task_artifacts.list_for_task(task):
        provenance = (
            record.get("provenance") if isinstance(record, Mapping) else None
        )
        if (
            isinstance(record, Mapping)
            and record.get("kind") == CROSS_CANDIDATE_SEARCH_ARTIFACT_KIND
            and isinstance(provenance, Mapping)
            and provenance.get("search_id") == search
        ):
            binding = load_cross_candidate_search_artifact(
                runtime,
                task_id=task,
                artifact_id=record["id"],
                expected_artifact_content_hash=record["content_hash"],
            )
            if not hmac.compare_digest(binding.result["search_id"], search):
                raise StrategyError("Cross search artifact embedded identity changed")
            matches.append(binding)
    if not matches:
        raise StrategyError(
            "Cross search artifact not found; run the bounded search again"
        )
    if len(matches) != 1:
        raise StrategyError("Cross search identity is ambiguous")
    binding = matches[0]
    selected = next(
        (
            item
            for item in binding.result["pairs"]
            if hmac.compare_digest(item["pair_id"], pair)
        ),
        None,
    )
    if selected is None:
        raise StrategyError(
            "pair_id is not an authenticated evaluated Cross search pair"
        )
    return binding, dict(selected)


def load_cross_candidate_search_artifact(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_artifact_content_hash: str,
    expected_search_id: str | None = None,
    expected_search_content_hash: str | None = None,
) -> CrossCandidateSearchArtifactBinding:
    """Authenticate aggregate search evidence and its historical source chain."""

    task = _text(task_id, "task_id")
    artifact = _hash(artifact_id, "artifact_id")
    artifact_hash = _hash(
        expected_artifact_content_hash,
        "expected_artifact_content_hash",
    )
    if (expected_search_id is None) != (
        expected_search_content_hash is None
    ):
        raise StrategyError(
            "Cross search exact domain identity requires both search id and hash"
        )
    frozen_search_id = (
        None
        if expected_search_id is None
        else _search_id(expected_search_id, "expected_search_id")
    )
    frozen_search_hash = (
        None
        if expected_search_content_hash is None
        else _hash(
            expected_search_content_hash,
            "expected_search_content_hash",
        )
    )
    record = runtime.task_artifacts.get_for_task(task, artifact)
    if (
        not isinstance(record, Mapping)
        or record.get("id") != artifact
        or record.get("task_id") != task
        or record.get("kind") != CROSS_CANDIDATE_SEARCH_ARTIFACT_KIND
        or record.get("origin_tool") != CROSS_CANDIDATE_SEARCH_ORIGIN_TOOL
        or not hmac.compare_digest(str(record.get("content_hash")), artifact_hash)
    ):
        raise StrategyError("Cross search artifact registry binding changed")
    tasks_root = Path(runtime.settings.tasks_dir).absolute()
    path = Path(_text(record.get("path"), "Cross search artifact path"))
    candidate_asset_tools._require_regular_artifact_path(path, root=tasks_root)
    candidate_asset_tools._require_file_content_hash(
        path,
        artifact_hash,
        "Cross search artifact content hash changed",
    )
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise StrategyError("Cross search artifact could not be read") from exc
    if len(raw) > MAX_ARTIFACT_BYTES:
        raise StrategyError("Cross search artifact exceeds the byte budget")
    if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), artifact_hash):
        raise StrategyError("Cross search artifact content changed during read")
    result = parse_cross_candidate_search_result_json(raw)
    if (
        frozen_search_id is not None
        and (
            result["search_id"] != frozen_search_id
            or not hmac.compare_digest(
                result["content_hash"],
                frozen_search_hash,
            )
        )
    ):
        raise StrategyError("Cross search identity changed")
    canonical = canonical_cross_candidate_search_result_json(result).encode("utf-8")
    if raw != canonical:
        raise StrategyError("Cross search artifact bytes are not canonical")
    provenance = _validate_provenance(record.get("provenance"))
    if (
        provenance["task_id"] != task
        or provenance["search_id"] != result["search_id"]
        or not hmac.compare_digest(
            provenance["search_content_hash"],
            result["content_hash"],
        )
        or not hmac.compare_digest(
            provenance["request_hash"],
            result["request_hash"],
        )
    ):
        raise StrategyError("Cross search artifact provenance changed")
    expected_path = _expected_search_path(
        runtime.settings.tasks_dir,
        task_id=task,
        search_id=result["search_id"],
        artifact_content_hash=artifact_hash,
    )
    if path != expected_path:
        raise StrategyError("Cross search artifact path is not canonical")
    source = candidate_asset_tools._load_source_artifact(
        runtime,
        task_id=task,
        artifact_id=provenance["source_artifact_id"],
        expected_content_hash=provenance["source_artifact_content_hash"],
        expected_candidate_id=provenance["candidate_id"],
        expected_evidence_hash=provenance["evidence_hash"],
    )
    evidence = _load_exact_parent_evidence(
        source,
        task_id=task,
        expected_candidate_id=provenance["candidate_id"],
        expected_evidence_hash=provenance["evidence_hash"],
    )
    dataset = candidate_asset_tools._load_dataset_binding(
        runtime,
        evidence=evidence,
        source=source,
    )
    sample_binding = _load_sample_binding(
        runtime,
        task_id=task,
        evidence=evidence,
        dataset=dataset,
    )
    expected_provenance = _artifact_provenance(
        task_id=task,
        source=source,
        dataset=dataset,
        sample_binding=sample_binding,
        evidence=evidence,
        result=result,
        dropped=provenance["nan_labels_dropped"],
    )
    if provenance != expected_provenance:
        raise StrategyError("Cross search artifact provenance changed")
    binding = CrossCandidateSearchArtifactBinding(
        task_id=task,
        artifact_id=artifact,
        artifact_path=path,
        artifact_content_hash=artifact_hash,
        artifact_provenance=provenance,
        artifact_provenance_json=_canonical_json(provenance),
        result=result,
        source=source,
        dataset=dataset,
        sample_binding=sample_binding,
        evidence=evidence,
        tasks_root=tasks_root,
        db_path=Path(runtime.settings.db_path).absolute(),
    )
    with runtime.task_artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        require_cross_candidate_search_artifact_binding_on_connection(conn, binding)
        conn.commit()
    return binding


def require_cross_candidate_search_artifact_binding_on_connection(
    conn,
    binding: CrossCandidateSearchArtifactBinding,
) -> None:
    """Re-authenticate search, source, dataset, sample, file, and registry row."""

    if not isinstance(binding, CrossCandidateSearchArtifactBinding):
        raise StrategyError("Cross search artifact binding is invalid")
    if not conn.in_transaction:
        raise StrategyError(
            "Cross search artifact binding requires a caller-owned transaction"
        )
    database = conn.execute(
        "SELECT file FROM pragma_database_list WHERE name = 'main'"
    ).fetchone()
    if (
        database is None
        or not str(database["file"])
        or Path(str(database["file"])).absolute() != binding.db_path
    ):
        raise StrategyError("Cross search artifact database binding changed")
    if validate_cross_candidate_search_result(binding.result) != binding.result:
        raise StrategyError("Cross search artifact result binding changed")
    provenance = _validate_provenance(binding.artifact_provenance)
    if (
        provenance != binding.artifact_provenance
        or _canonical_json(provenance) != binding.artifact_provenance_json
    ):
        raise StrategyError("Cross search artifact provenance binding changed")
    candidate_asset_tools._require_source_on_connection(conn, binding.source)
    candidate_asset_tools._require_dataset_on_connection(conn, binding.dataset)
    require_historical_strategy_risk_development_execution_binding_on_connection(
        conn,
        binding.sample_binding,
    )
    candidate_asset_tools._require_regular_artifact_path(
        binding.source.path,
        root=binding.tasks_root,
    )
    candidate_asset_tools._require_file_content_hash(
        binding.source.path,
        binding.source.content_hash,
        "Cross search source artifact content changed",
    )
    candidate_asset_tools._require_file_content_hash(
        binding.dataset.path,
        binding.dataset.content_hash,
        "Cross search dataset content changed",
    )
    candidate_asset_tools._require_regular_artifact_path(
        binding.artifact_path,
        root=binding.tasks_root,
    )
    candidate_asset_tools._require_file_content_hash(
        binding.artifact_path,
        binding.artifact_content_hash,
        "Cross search artifact content changed",
    )
    row = conn.execute(
        """
        SELECT id, task_id, kind, path, content_hash, origin_tool,
               provenance_json
          FROM task_artifacts
         WHERE task_id = ? AND id = ?
        """,
        (binding.task_id, binding.artifact_id),
    ).fetchone()
    if row is None:
        raise StrategyError("Cross search artifact is no longer registered")
    if (
        str(row["id"]) != binding.artifact_id
        or str(row["task_id"]) != binding.task_id
        or str(row["kind"]) != CROSS_CANDIDATE_SEARCH_ARTIFACT_KIND
        or str(row["path"]) != str(binding.artifact_path)
        or not hmac.compare_digest(
            str(row["content_hash"]),
            binding.artifact_content_hash,
        )
        or str(row["origin_tool"]) != CROSS_CANDIDATE_SEARCH_ORIGIN_TOOL
        or str(row["provenance_json"]) != binding.artifact_provenance_json
    ):
        raise StrategyError("Cross search artifact registry binding changed")


def _load_sample_binding(
    runtime,
    *,
    task_id: str,
    evidence: Mapping[str, Any],
    dataset,
) -> StrategyRiskDevelopmentExecutionBinding:
    identity = evidence["identity"]
    generation = evidence["generation"]["parameters"]
    return load_historical_strategy_risk_development_execution_binding(
        runtime,
        task_id=task_id,
        sample_design_ref=generation.get("sample_design_ref"),
        dataset_id=dataset.dataset_id,
        dataset_content_hash=dataset.content_hash,
        workspace_revision=identity["workspace_revision"],
        workspace_generation=identity["workspace_generation"],
        semantic_mapping_hash=identity["semantic_mapping_hash"],
        target_col=evidence["analysis"]["target"],
        drop_nan_labels=generation.get("drop_nan_labels"),
        loan_amount_col=generation.get("loan_amount_col"),
        overdue_amount_col=generation.get("overdue_amount_col"),
    )


def _select_ranked_axes(
    evidence: Mapping[str, Any],
    *,
    dataset,
    requested_features: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rankings = evidence["analysis"].get("rankings")
    if not _array_like(rankings):
        raise StrategyError("Cross search parent rankings are invalid")
    selected: list[dict[str, Any]] = []
    axes: dict[str, dict[str, Any]] = {}
    for feature in sorted(requested_features):
        matches = [
            item
            for item in rankings
            if isinstance(item, Mapping) and item.get("feature") == feature
        ]
        if not matches:
            raise StrategyError(
                f"Cross search feature has no available parent ranking: {feature}"
            )
        ranking = matches[0]
        method = _text(ranking.get("method"), f"{feature} ranking method")
        axis = _resolve_axis(
            evidence,
            dataset=dataset,
            feature=feature,
            method=method,
            label=f"search/{feature}",
        )
        axis_iv = _finite(ranking.get("iv"), f"{feature} ranking iv", minimum=0.0)
        selected.append(
            {
                "feature": feature,
                "method": method,
                "axis_iv": axis_iv,
                "bin_count": len(axis["bins"]),
            }
        )
        axes[feature] = axis
    return selected, axes


def _require_planned_budget(
    *,
    row_count: int,
    features: Sequence[Mapping[str, Any]],
    pairs: Sequence[tuple[str, str]],
) -> None:
    bins = {item["feature"]: int(item["bin_count"]) for item in features}
    pair_rows = row_count * len(pairs)
    axis_bin_rows = row_count * sum(bins.values())
    cells = sum(bins[left] * bins[right] for left, right in pairs)
    for label, used, limit in (
        ("pair row evaluations", pair_rows, MAX_PAIR_ROW_EVALUATIONS),
        ("axis-bin row evaluations", axis_bin_rows, MAX_AXIS_BIN_ROW_EVALUATIONS),
        ("derived cells", cells, MAX_DERIVED_CELLS),
    ):
        if used > limit:
            raise StrategyError(
                f"Cross search {label} exceed hard budget ({used} > {limit})"
            )
    if any(bins[left] * bins[right] > MAX_CELLS_PER_PAIR for left, right in pairs):
        raise StrategyError("Cross search pair exceeds 400 cells")


def _read_search_sample(
    runtime,
    *,
    evidence: Mapping[str, Any],
    source,
    dataset,
    sample_binding: StrategyRiskDevelopmentExecutionBinding,
    features: Sequence[Mapping[str, Any]],
) -> tuple[Any, Any, dict[str, Any]]:
    first, second = features[:2]
    projection = _resolve_projection(
        evidence,
        dataset=dataset,
        row_feature=first["feature"],
        column_feature=second["feature"],
    )
    for item in features:
        if item["feature"] not in projection["columns"]:
            projection["columns"].append(item["feature"])
    for column in sample_binding.partition_columns:
        if column not in projection["columns"]:
            projection["columns"].append(column)
    frame = runtime.backend.read_frame(
        dataset.path,
        columns=projection["columns"],
    )
    population_count = len(frame)
    if population_count != dataset.row_count:
        raise StrategyError("Cross search source dataset row count changed")
    candidate_asset_tools._require_dataset_unchanged(runtime, dataset)
    frame = bind_strategy_risk_development_frame(
        frame,
        binding=sample_binding,
    )
    labeled = _resolve_exact_labeled_sample(
        frame,
        evidence=evidence,
        target_col=projection["target_col"],
        drop_nan_labels=projection["drop_nan_labels"],
        expected_dropped=projection["expected_dropped"],
    )
    candidate_asset_tools._require_source_unchanged(runtime, source)
    candidate_asset_tools._require_dataset_unchanged(runtime, dataset)
    revalidate_historical_strategy_risk_development_execution_binding(
        runtime,
        sample_binding,
    )
    target = _binary_target(labeled, projection["target_col"])
    return labeled, target, projection


def _persist_search(
    runtime,
    *,
    task_id: str,
    source,
    dataset,
    sample_binding: StrategyRiskDevelopmentExecutionBinding,
    evidence: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = validate_cross_candidate_search_result(result)
    canonical = canonical_cross_candidate_search_result_json(normalized).encode(
        "utf-8"
    )
    if len(canonical) > MAX_ARTIFACT_BYTES:
        raise StrategyError("Cross search artifact exceeds the byte budget")
    artifact_hash = hashlib.sha256(canonical).hexdigest()
    tasks_root = Path(runtime.settings.tasks_dir)
    out_dir = tasks_root / task_id / "strategy_cross_candidate_searches"
    candidate_asset_tools._require_output_directory_boundary(
        out_dir,
        root=tasks_root,
    )
    final_path = _expected_search_path(
        tasks_root,
        task_id=task_id,
        search_id=normalized["search_id"],
        artifact_content_hash=artifact_hash,
    )
    dropped = sample_binding.development_population_count - normalized["population"][
        "row_count"
    ]
    provenance = _artifact_provenance(
        task_id=task_id,
        source=source,
        dataset=dataset,
        sample_binding=sample_binding,
        evidence=evidence,
        result=normalized,
        dropped=dropped,
    )
    uow = ArtifactUnitOfWork()
    staged = uow.stage_file(out_dir, final_path.name)
    committed = False
    reused = False
    try:
        staged.path.write_bytes(canonical)
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                candidate_asset_tools._require_source_on_connection(conn, source)
                candidate_asset_tools._require_dataset_on_connection(conn, dataset)
                require_historical_strategy_risk_development_execution_binding_on_connection(
                    conn,
                    sample_binding,
                )
                row = conn.execute(
                    """
                    SELECT id, task_id, kind, path, content_hash, origin_tool,
                           provenance_json
                      FROM task_artifacts
                     WHERE task_id = ? AND kind = ? AND path = ?
                    """,
                    (
                        task_id,
                        CROSS_CANDIDATE_SEARCH_ARTIFACT_KIND,
                        str(final_path),
                    ),
                ).fetchone()
                if row is None:
                    if final_path.exists() or final_path.is_symlink():
                        raise StrategyError(
                            "Cross search path exists without a registry row"
                        )
                    uow.promote_all()
                else:
                    expected_provenance_json = _canonical_json(provenance)
                    if (
                        str(row["task_id"]) != task_id
                        or str(row["kind"])
                        != CROSS_CANDIDATE_SEARCH_ARTIFACT_KIND
                        or str(row["path"]) != str(final_path)
                        or not hmac.compare_digest(
                            str(row["content_hash"]),
                            artifact_hash,
                        )
                        or str(row["origin_tool"])
                        != CROSS_CANDIDATE_SEARCH_ORIGIN_TOOL
                        or str(row["provenance_json"])
                        != expected_provenance_json
                    ):
                        raise StrategyError(
                            "Cross search existing artifact binding changed"
                        )
                    candidate_asset_tools._require_file_content_hash(
                        final_path,
                        artifact_hash,
                        "Cross search existing artifact content changed",
                    )
                    uow.rollback()
                    reused = True
                candidate_asset_tools._require_file_content_hash(
                    final_path,
                    artifact_hash,
                    "Cross search artifact content changed before registration",
                )
                record = runtime.task_artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=CROSS_CANDIDATE_SEARCH_ARTIFACT_KIND,
                    path=str(final_path),
                    content_hash=artifact_hash,
                    origin_tool=CROSS_CANDIDATE_SEARCH_ORIGIN_TOOL,
                    provenance=provenance,
                )
                conn.commit()
                committed = True
            except Exception:
                uow.rollback()
                raise
        if not reused:
            uow.commit()
    except Exception:
        if not committed:
            uow.rollback()
        raise
    return {
        "schema_version": CROSS_CANDIDATE_SEARCH_TOOL_SCHEMA_VERSION,
        "search_id": normalized["search_id"],
        "request_hash": normalized["request_hash"],
        "content_hash": normalized["content_hash"],
        "source_artifact_id": source.artifact_id,
        "candidate_id": evidence["candidate_id"],
        "evidence_hash": evidence["evidence_hash"],
        "population_count": normalized["population"]["row_count"],
        "search_space": normalized["search_space"],
        "evaluated": normalized["evaluated"],
        "truncated": normalized["truncated"],
        "eligible": normalized["eligible"],
        "search_result": normalized,
        "artifacts": [
            {
                "artifact_id": str(record["id"]),
                "kind": CROSS_CANDIDATE_SEARCH_ARTIFACT_KIND,
                "format": "json",
                "filename": final_path.name,
                "content_hash": str(record["content_hash"]),
                "download_url": (
                    f"/api/tasks/{quote(task_id, safe='')}"
                    f"/task-artifacts/{quote(str(record['id']), safe='')}/download"
                ),
            }
        ],
        "not_selected": True,
        "not_admitted": True,
        "not_applied": True,
        "not_adopted": True,
        "not_deployed": True,
    }


def _artifact_provenance(
    *,
    task_id: str,
    source,
    dataset,
    sample_binding: StrategyRiskDevelopmentExecutionBinding,
    evidence: Mapping[str, Any],
    result: Mapping[str, Any],
    dropped: int,
) -> dict[str, Any]:
    expected_source = {
        "candidate_id": evidence["candidate_id"],
        "evidence_hash": evidence["evidence_hash"],
        "sample_context_hash": sample_context_hash_from_candidate_evidence(
            evidence
        ),
    }
    if result["source"] != expected_source:
        raise StrategyError("Cross search source identity changed")
    configured_features = result["configuration"]["features"]
    selected_features, selected_axes = _select_ranked_axes(
        evidence,
        dataset=dataset,
        requested_features=[item["feature"] for item in configured_features],
    )
    if selected_features != configured_features:
        raise StrategyError("Cross search parent-ranked feature methods changed")
    first_axis = selected_axes[selected_features[0]["feature"]]
    expected_population = {
        "row_count": sum(int(item["count"]) for item in first_axis["bins"]),
        "good": sum(int(item["good"]) for item in first_axis["bins"]),
        "bad": sum(int(item["bad"]) for item in first_axis["bins"]),
    }
    if result["population"] != expected_population:
        raise StrategyError("Cross search population changed from parent evidence")
    if (
        isinstance(dropped, bool)
        or not isinstance(dropped, int)
        or dropped < 0
        or result["population"]["row_count"] + dropped
        != sample_binding.development_population_count
    ):
        raise StrategyError("Cross search labelled population binding changed")
    value = {
        "schema_version": CROSS_CANDIDATE_SEARCH_ARTIFACT_SCHEMA_VERSION,
        "producer_version": CROSS_CANDIDATE_SEARCH_PRODUCER_VERSION,
        "task_id": task_id,
        "search_id": result["search_id"],
        "search_content_hash": result["content_hash"],
        "request_hash": result["request_hash"],
        "source_artifact_id": source.artifact_id,
        "source_artifact_content_hash": source.content_hash,
        "candidate_id": evidence["candidate_id"],
        "evidence_hash": evidence["evidence_hash"],
        "dataset_id": dataset.dataset_id,
        "dataset_content_hash": dataset.content_hash,
        "registry_metadata_hash": dataset.registry_metadata_hash,
        "workspace_revision": evidence["identity"]["workspace_revision"],
        "workspace_generation": evidence["identity"]["workspace_generation"],
        "semantic_mapping_hash": evidence["identity"]["semantic_mapping_hash"],
        "sample_design_ref": sample_binding.to_ref_dict(),
        "sample_context_hash": result["source"]["sample_context_hash"],
        "sample_partition": "risk/development",
        "target_col": sample_binding.target_col,
        "drop_nan_labels": sample_binding.drop_nan_labels,
        "nan_labels_dropped": dropped,
        "labeled_count": result["population"]["row_count"],
        "features": result["configuration"]["features"],
        "max_pairs": result["configuration"]["max_pairs"],
        "lifecycle": dict(_LIFECYCLE),
    }
    return _validate_provenance(value)


def _validate_provenance(value: object) -> dict[str, Any]:
    obj = _object(value, "Cross search provenance")
    _exact(obj, _PROVENANCE_FIELDS, "Cross search provenance")
    if (
        obj["schema_version"] != CROSS_CANDIDATE_SEARCH_ARTIFACT_SCHEMA_VERSION
        or obj["producer_version"] != CROSS_CANDIDATE_SEARCH_PRODUCER_VERSION
    ):
        raise StrategyError("Cross search provenance version is invalid")
    lifecycle = _object(obj["lifecycle"], "provenance.lifecycle")
    if dict(lifecycle) != _LIFECYCLE:
        raise StrategyError("Cross search provenance lifecycle changed")
    sample_ref = _object(obj["sample_design_ref"], "provenance.sample_design_ref")
    if set(sample_ref) != {
        "artifact_id",
        "artifact_content_hash",
        "sample_design_id",
        "sample_design_content_hash",
        "partition",
    }:
        raise StrategyError("Cross search sample_design_ref is invalid")
    features_raw = _array(obj["features"], "provenance.features")
    features: list[dict[str, Any]] = []
    for index, item in enumerate(features_raw):
        row = _object(item, f"provenance.features[{index}]")
        if set(row) != {"feature", "method", "axis_iv", "bin_count"}:
            raise StrategyError("Cross search provenance feature fields changed")
        features.append(
            {
                "feature": _text(row["feature"], "provenance feature"),
                "method": _text(row["method"], "provenance method"),
                "axis_iv": _finite(
                    row["axis_iv"],
                    "provenance axis_iv",
                    minimum=0.0,
                ),
                "bin_count": _integer(
                    row["bin_count"],
                    "provenance bin_count",
                    minimum=1,
                    maximum=20,
                ),
            }
        )
    result = {
        "schema_version": obj["schema_version"],
        "producer_version": obj["producer_version"],
        "task_id": _text(obj["task_id"], "provenance.task_id"),
        "search_id": _search_id(obj["search_id"], "provenance.search_id"),
        "search_content_hash": _hash(
            obj["search_content_hash"],
            "provenance.search_content_hash",
        ),
        "request_hash": _hash(obj["request_hash"], "provenance.request_hash"),
        "source_artifact_id": _hash(
            obj["source_artifact_id"],
            "provenance.source_artifact_id",
        ),
        "source_artifact_content_hash": _hash(
            obj["source_artifact_content_hash"],
            "provenance.source_artifact_content_hash",
        ),
        "candidate_id": _candidate_id(
            obj["candidate_id"],
            "provenance.candidate_id",
        ),
        "evidence_hash": _hash(
            obj["evidence_hash"],
            "provenance.evidence_hash",
        ),
        "dataset_id": _text(obj["dataset_id"], "provenance.dataset_id"),
        "dataset_content_hash": _hash(
            obj["dataset_content_hash"],
            "provenance.dataset_content_hash",
        ),
        "registry_metadata_hash": _hash(
            obj["registry_metadata_hash"],
            "provenance.registry_metadata_hash",
        ),
        "workspace_revision": _integer(
            obj["workspace_revision"],
            "provenance.workspace_revision",
            minimum=0,
        ),
        "workspace_generation": _integer(
            obj["workspace_generation"],
            "provenance.workspace_generation",
            minimum=0,
        ),
        "semantic_mapping_hash": _hash(
            obj["semantic_mapping_hash"],
            "provenance.semantic_mapping_hash",
        ),
        "sample_design_ref": {
            "artifact_id": _hash(
                sample_ref["artifact_id"],
                "provenance.sample_design_ref.artifact_id",
            ),
            "artifact_content_hash": _hash(
                sample_ref["artifact_content_hash"],
                "provenance.sample_design_ref.artifact_content_hash",
            ),
            "sample_design_id": _text(
                sample_ref["sample_design_id"],
                "provenance.sample_design_ref.sample_design_id",
            ),
            "sample_design_content_hash": _hash(
                sample_ref["sample_design_content_hash"],
                "provenance.sample_design_ref.sample_design_content_hash",
            ),
            "partition": _text(
                sample_ref["partition"],
                "provenance.sample_design_ref.partition",
            ),
        },
        "sample_context_hash": _hash(
            obj["sample_context_hash"],
            "provenance.sample_context_hash",
        ),
        "sample_partition": _text(
            obj["sample_partition"],
            "provenance.sample_partition",
        ),
        "target_col": _text(obj["target_col"], "provenance.target_col"),
        "drop_nan_labels": obj["drop_nan_labels"],
        "nan_labels_dropped": _integer(
            obj["nan_labels_dropped"],
            "provenance.nan_labels_dropped",
            minimum=0,
        ),
        "labeled_count": _integer(
            obj["labeled_count"],
            "provenance.labeled_count",
            minimum=1,
        ),
        "features": features,
        "max_pairs": _integer(
            obj["max_pairs"],
            "provenance.max_pairs",
            minimum=1,
            maximum=MAX_PAIRS,
        ),
        "lifecycle": dict(_LIFECYCLE),
    }
    if not isinstance(result["drop_nan_labels"], bool):
        raise StrategyError("Cross search provenance drop_nan_labels is invalid")
    if result["sample_partition"] != "risk/development":
        raise StrategyError("Cross search provenance partition changed")
    return result


def _validate_search_inputs(value: object) -> dict[str, Any]:
    obj = _object(value, "search_cross_matrix_candidates inputs")
    _exact(obj, _SEARCH_INPUT_FIELDS, "search_cross_matrix_candidates inputs")
    features = [
        _text(item, f"features[{index}]")
        for index, item in enumerate(_array(obj["features"], "features"))
    ]
    if not 2 <= len(features) <= MAX_FEATURES:
        raise StrategyError("Cross search features must contain 2..20 values")
    if len(set(features)) != len(features):
        raise StrategyError("Cross search features must be unique")
    return {
        "source_artifact_id": _hash(
            obj["source_artifact_id"],
            "source_artifact_id",
        ),
        "expected_artifact_content_hash": _hash(
            obj["expected_artifact_content_hash"],
            "expected_artifact_content_hash",
        ),
        "expected_candidate_id": _candidate_id(
            obj["expected_candidate_id"],
            "expected_candidate_id",
        ),
        "expected_evidence_hash": _hash(
            obj["expected_evidence_hash"],
            "expected_evidence_hash",
        ),
        "features": sorted(features),
        "max_pairs": _integer(
            obj["max_pairs"],
            "max_pairs",
            minimum=1,
            maximum=MAX_PAIRS,
        ),
    }


def _validate_selection_inputs(value: object) -> dict[str, str]:
    obj = _object(value, "build_cross_matrix_candidate_from_search inputs")
    _exact(
        obj,
        _SELECTION_INPUT_FIELDS,
        "build_cross_matrix_candidate_from_search inputs",
    )
    return {
        "search_id": _search_id(obj["search_id"], "search_id"),
        "pair_id": _pair_id(obj["pair_id"], "pair_id"),
    }


def _expected_search_path(
    tasks_dir: Path | str,
    *,
    task_id: str,
    search_id: str,
    artifact_content_hash: str,
) -> Path:
    return (
        Path(tasks_dir)
        / task_id
        / "strategy_cross_candidate_searches"
        / f"{search_id}_{artifact_content_hash[:12]}.json"
    )


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StrategyError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise StrategyError(f"{label} keys must be strings")
    return value


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(
        value, str | bytes | bytearray
    ):
        raise StrategyError(f"{label} must be an array")
    return list(value)


def _array_like(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    )


def _exact(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    missing = sorted(expected - set(value))
    unsupported = sorted(set(value) - expected)
    if missing or unsupported:
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unsupported:
            details.append("unsupported: " + ", ".join(unsupported))
        raise StrategyError(
            f"{label} has invalid fields (" + "; ".join(details) + ")"
        )


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise StrategyError(f"{label} must be non-empty text")
    return value


def _hash(value: object, label: str) -> str:
    text = _text(value, label)
    if _HASH_RE.fullmatch(text) is None:
        raise StrategyError(f"{label} must be lowercase sha256")
    return text


def _candidate_id(value: object, label: str) -> str:
    text = _text(value, label)
    if _CANDIDATE_ID_RE.fullmatch(text) is None:
        raise StrategyError(f"{label} is invalid")
    return text


def _search_id(value: object, label: str) -> str:
    text = _text(value, label)
    if _SEARCH_ID_RE.fullmatch(text) is None:
        raise StrategyError(f"{label} is invalid")
    return text


def _pair_id(value: object, label: str) -> str:
    text = _text(value, label)
    if _PAIR_ID_RE.fullmatch(text) is None:
        raise StrategyError(f"{label} is invalid")
    return text


def _integer(
    value: object,
    label: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise StrategyError(f"{label} must be an integer")
    normalized = int(value)
    if normalized < minimum or (
        maximum is not None and normalized > maximum
    ):
        if maximum is None:
            raise StrategyError(f"{label} must be at least {minimum}")
        raise StrategyError(f"{label} must be between {minimum} and {maximum}")
    return normalized


def _finite(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise StrategyError(f"{label} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise StrategyError(f"{label} must be a finite number")
    if minimum is not None and normalized < minimum:
        raise StrategyError(f"{label} must be at least {minimum}")
    return normalized


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise StrategyError("Cross search payload must be finite JSON") from exc


__all__ = [
    "CROSS_CANDIDATE_SEARCH_ARTIFACT_KIND",
    "CROSS_CANDIDATE_SEARCH_SELECTION_TOOL_SCHEMA_VERSION",
    "CROSS_CANDIDATE_SEARCH_TOOL_SCHEMA_VERSION",
    "CrossCandidateSearchArtifactBinding",
    "load_cross_candidate_search_artifact",
    "require_cross_candidate_search_artifact_binding_on_connection",
    "resolve_cross_candidate_search_pair",
    "run_build_cross_matrix_candidate_from_search",
    "run_search_cross_matrix_candidates",
]
