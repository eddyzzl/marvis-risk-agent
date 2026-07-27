"""Governed persistence boundary for interactive-tree split-candidate search."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
from numbers import Integral
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote

from marvis.artifacts import ArtifactUnitOfWork
from marvis.files import sha256_file
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.evaluator import evaluate_expression_frame
from marvis.packs.strategy.interactive_tree_revision import (
    interactive_tree_topology_evidence,
)
from marvis.packs.strategy.automatic_tree_sample_design import (
    sample_design_ref_from_automatic_tree_source_refs,
)
from marvis.packs.strategy.interactive_tree_split_search import (
    INTERACTIVE_TREE_SPLIT_SEARCH_PRODUCER_VERSION,
    MAX_ROW_EVALUATIONS,
    MAX_SEARCH_FEATURES,
    MAX_THRESHOLDS_PER_FEATURE,
    canonical_interactive_tree_split_search_json,
    search_interactive_tree_split_candidates,
    validate_interactive_tree_split_search,
)
from marvis.packs.strategy.interactive_tree_tools import (
    MAX_INTERACTIVE_TREE_REVISION_ANCESTRY_BYTES,
    _RevisionReadBudget,
    _ResolvedRevisionSource,
    _load_replay_context,
    _read_verified_regular_file,
    _require_binding_on_connection,
    _resolve_revision_source_on_connection,
    _safe_component,
)
from marvis.packs.strategy.sample_design_execution import (
    require_strategy_risk_development_execution_binding_on_connection,
)
from marvis.repositories.task_artifacts import stable_task_artifact_id


INTERACTIVE_TREE_SPLIT_SEARCH_TOOL_SCHEMA_VERSION = (
    "strategy.search-interactive-tree-split-candidates-tool.v1"
)
INTERACTIVE_TREE_SPLIT_SEARCH_ARTIFACT_KIND = (
    "strategy_interactive_tree_split_search_json"
)
INTERACTIVE_TREE_SPLIT_SEARCH_ARTIFACT_SCHEMA_VERSION = (
    "strategy.interactive-tree-split-search-artifact.v1"
)
INTERACTIVE_TREE_SPLIT_SEARCH_ORIGIN_TOOL = (
    "strategy.search_interactive_tree_split_candidates"
)
MAX_INTERACTIVE_TREE_SPLIT_SEARCH_BYTES = 8 * 1024 * 1024

_INPUT_FIELDS = frozenset(
    {
        "source_tree_id",
        "node_id",
        "mode",
        "features",
        "max_thresholds_per_feature",
        "max_row_evaluations",
    }
)
_REQUIRED_INPUT_FIELDS = frozenset(
    {
        "source_tree_id",
        "node_id",
        "mode",
        "max_thresholds_per_feature",
        "max_row_evaluations",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "task_id",
        "kind",
        "format",
        "search_id",
        "search_hash",
        "artifact_content_hash",
        "source_tree_id",
        "source_tree_hash",
        "base_asset_id",
        "base_asset_hash",
        "base_tree_result_hash",
        "dataset_id",
        "dataset_content_hash",
        "registry_metadata_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "sample_design_ref",
        "sample_context_hash",
        "sample_partition",
        "node_id",
        "node_kind",
        "node_condition_hash",
        "mode",
        "features",
        "min_leaf_count",
        "max_thresholds_per_feature",
        "max_row_evaluations",
        "lifecycle",
    }
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
_SEARCH_ID_RE = re.compile(
    r"^interactive-tree-split-search-[0-9a-f]{32}$"
)
_MODES = frozenset({"all_features", "selected_features"})


@dataclass(frozen=True)
class VerifiedInteractiveTreeSplitSearch:
    """One authenticated aggregate search and its exact tree source."""

    artifact_id: str
    task_id: str
    path: Path
    content_hash: str
    provenance: dict[str, Any]
    result: dict[str, Any]
    source: _ResolvedRevisionSource


@dataclass(frozen=True)
class _BuiltSearch:
    source: _ResolvedRevisionSource
    context: Any
    node: dict[str, Any]
    result: dict[str, Any]


def run_search_interactive_tree_split_candidates(
    inputs: object,
    ctx,
    runtime,
) -> dict[str, Any]:
    """Search one visible node without selecting a winner or editing the tree."""

    request = _validate_inputs(inputs)
    task_id = _text(ctx.task_id, "task_id")
    with runtime.task_artifacts.transaction() as conn:
        source = _resolve_revision_source_on_connection(
            conn,
            runtime=runtime,
            task_id=task_id,
            source_tree_id=request["source_tree_id"],
            revision_cache={},
            automatic_source_cache={},
            budget=_RevisionReadBudget(
                MAX_INTERACTIVE_TREE_REVISION_ANCESTRY_BYTES
            ),
        )
    built = _build_search(
        runtime,
        task_id=task_id,
        request=request,
        source=source,
    )
    artifact = _persist_search(
        runtime,
        task_id=task_id,
        request=request,
        built=built,
    )
    result = built.result
    return {
        "schema_version": INTERACTIVE_TREE_SPLIT_SEARCH_TOOL_SCHEMA_VERSION,
        "search_id": result["search_id"],
        "search_hash": result["search_hash"],
        "source_tree_id": request["source_tree_id"],
        "node_id": request["node_id"],
        "mode": request["mode"],
        "feature_count": result["budget"]["feature_count"],
        "evaluated_candidates": result["budget"]["evaluated_candidates"],
        "eligible_candidates": sum(
            1 for candidate in result["candidates"] if candidate["eligible"]
        ),
        "truncated": result["budget"]["truncated"],
        "search_result": result,
        "artifacts": [artifact],
        "winner_selected": False,
        "tree_modified": False,
    }


def canonical_interactive_tree_split_search_path(
    tasks_dir: Path | str,
    *,
    task_id: str,
    search_id: str,
) -> Path:
    normalized_task = _safe_component(task_id, "task_id")
    normalized_search = _text(search_id, "search_id")
    if _SEARCH_ID_RE.fullmatch(normalized_search) is None:
        raise StrategyError("interactive-tree split search_id has an invalid format")
    return (
        Path(tasks_dir).absolute()
        / normalized_task
        / "strategy_interactive_tree_split_searches"
        / f"{normalized_search}.json"
    )


def load_verified_interactive_tree_split_search(
    runtime,
    *,
    task_id: str,
    search_id: str,
) -> VerifiedInteractiveTreeSplitSearch:
    """Load and authenticate a search, its registry row, and its tree source."""

    normalized_task = _text(task_id, "task_id")
    normalized_search = _text(search_id, "search_id")
    path = canonical_interactive_tree_split_search_path(
        runtime.settings.tasks_dir,
        task_id=normalized_task,
        search_id=normalized_search,
    )
    with runtime.task_artifacts.transaction() as conn:
        row = _lookup_artifact_row(
            conn,
            task_id=normalized_task,
            path=path,
        )
        expected_artifact_id = stable_task_artifact_id(
            task_id=normalized_task,
            kind=INTERACTIVE_TREE_SPLIT_SEARCH_ARTIFACT_KIND,
            path=str(path),
        )
        if row["id"] != expected_artifact_id:
            raise StrategyError(
                "interactive-tree split search stable identity changed"
            )
        if row["origin_tool"] != INTERACTIVE_TREE_SPLIT_SEARCH_ORIGIN_TOOL:
            raise StrategyError(
                "interactive-tree split search origin_tool changed"
            )
        provenance = _strict_provenance(row)
        _require_exact_fields(
            provenance,
            _PROVENANCE_FIELDS,
            "interactive-tree split search provenance",
        )
        source = _resolve_revision_source_on_connection(
            conn,
            runtime=runtime,
            task_id=normalized_task,
            source_tree_id=_text(
                provenance["source_tree_id"],
                "provenance source_tree_id",
            ),
            revision_cache={},
            automatic_source_cache={},
            budget=_RevisionReadBudget(
                MAX_INTERACTIVE_TREE_REVISION_ANCESTRY_BYTES
            ),
        )
    registered_hash = _hash(
        row["content_hash"],
        "interactive-tree split search content_hash",
    )
    canonical = _read_verified_regular_file(
        path,
        root=Path(runtime.settings.tasks_dir).absolute(),
        expected_hash=registered_hash,
        max_bytes=MAX_INTERACTIVE_TREE_SPLIT_SEARCH_BYTES,
    )
    try:
        payload = json.loads(canonical.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StrategyError(
            "interactive-tree split search JSON is invalid"
        ) from exc
    result = validate_interactive_tree_split_search(payload)
    if canonical != canonical_interactive_tree_split_search_json(result).encode(
        "utf-8"
    ):
        raise StrategyError(
            "interactive-tree split search JSON is not canonical"
        )
    expected_provenance = _artifact_provenance(
        task_id=normalized_task,
        source=source,
        context=None,
        request=None,
        node=None,
        result=result,
        artifact_content_hash=registered_hash,
        stored=provenance,
    )
    if provenance != expected_provenance:
        raise StrategyError(
            "interactive-tree split search provenance changed"
        )
    if normalized_search != result["search_id"]:
        raise StrategyError("interactive-tree split search identity changed")
    return VerifiedInteractiveTreeSplitSearch(
        artifact_id=expected_artifact_id,
        task_id=normalized_task,
        path=path,
        content_hash=registered_hash,
        provenance=provenance,
        result=result,
        source=source,
    )


def _build_search(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
    source: _ResolvedRevisionSource,
) -> _BuiltSearch:
    context = _load_replay_context(
        runtime,
        task_id=task_id,
        source=source.automatic_source,
    )
    topology = _topology(source)
    node = next(
        (
            dict(item)
            for item in topology["nodes"]
            if item["node_id"] == request["node_id"]
        ),
        None,
    )
    if node is None or not node["is_visible"]:
        raise StrategyError(
            "interactive-tree split search node must be exact and visible"
        )
    training = source.automatic_source.asset["tree_result"]["training"]
    feature_order = list(training["feature_order"])
    features = (
        sorted(feature_order)
        if request["mode"] == "all_features"
        else list(request["features"])
    )
    unexpected = sorted(set(features) - set(feature_order))
    if unexpected:
        raise StrategyError(
            "interactive-tree split search features are not in the "
            "authenticated tree feature universe: "
            + ", ".join(unexpected)
        )
    try:
        mask = evaluate_expression_frame(
            context.labeled,
            node["condition"],
        ).to_numpy(dtype=bool)
    except (TypeError, ValueError, KeyError) as exc:
        raise StrategyError(
            "interactive-tree split search node condition cannot be replayed"
        ) from exc
    tree_result = source.automatic_source.asset["tree_result"]
    result = search_interactive_tree_split_candidates(
        context.labeled,
        node_mask=mask,
        node_id=request["node_id"],
        source_tree_id=request["source_tree_id"],
        features=features,
        target=context.target,
        weights=context.weights,
        medians=tree_result["preprocessing"]["medians"],
        directions=tree_result["directions"],
        min_leaf_count=training["cart"]["min_leaf_count"],
        max_thresholds_per_feature=request["max_thresholds_per_feature"],
        max_row_evaluations=request["max_row_evaluations"],
    )
    return _BuiltSearch(
        source=source,
        context=context,
        node=node,
        result=result,
    )


def _persist_search(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
    built: _BuiltSearch,
) -> dict[str, Any]:
    canonical = canonical_interactive_tree_split_search_json(
        built.result
    ).encode("utf-8")
    if len(canonical) > MAX_INTERACTIVE_TREE_SPLIT_SEARCH_BYTES:
        raise StrategyError(
            "interactive-tree split search exceeds the JSON byte budget"
        )
    content_hash = hashlib.sha256(canonical).hexdigest()
    final_path = canonical_interactive_tree_split_search_path(
        runtime.settings.tasks_dir,
        task_id=task_id,
        search_id=built.result["search_id"],
    )
    out_dir = final_path.parent
    provenance = _artifact_provenance(
        task_id=task_id,
        source=built.source,
        context=built.context,
        request=request,
        node=built.node,
        result=built.result,
        artifact_content_hash=content_hash,
    )
    uow = ArtifactUnitOfWork()
    staged = uow.stage_file(out_dir, final_path.name)
    staged.path.write_bytes(canonical)
    committed = False
    reused = False
    try:
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                locked_source = _resolve_revision_source_on_connection(
                    conn,
                    runtime=runtime,
                    task_id=task_id,
                    source_tree_id=request["source_tree_id"],
                    revision_cache={},
                    automatic_source_cache={},
                    budget=_RevisionReadBudget(
                        MAX_INTERACTIVE_TREE_REVISION_ANCESTRY_BYTES
                    ),
                )
                locked = _build_search(
                    runtime,
                    task_id=task_id,
                    request=request,
                    source=locked_source,
                )
                locked_canonical = (
                    canonical_interactive_tree_split_search_json(
                        locked.result
                    ).encode("utf-8")
                )
                locked_provenance = _artifact_provenance(
                    task_id=task_id,
                    source=locked.source,
                    context=locked.context,
                    request=request,
                    node=locked.node,
                    result=locked.result,
                    artifact_content_hash=content_hash,
                )
                if (
                    not hmac.compare_digest(locked_canonical, canonical)
                    or locked_provenance != provenance
                ):
                    raise StrategyError(
                        "interactive-tree split search evidence changed "
                        "before registration"
                    )
                _require_binding_on_connection(
                    conn,
                    task_id=task_id,
                    binding=locked.context.data_binding,
                )
                require_strategy_risk_development_execution_binding_on_connection(
                    conn,
                    locked.context.sample_design,
                )
                if not hmac.compare_digest(
                    sha256_file(locked.context.data_binding.path),
                    locked.context.data_binding.content_hash,
                ):
                    raise StrategyError(
                        "interactive-tree split search dataset changed "
                        "before registration"
                    )
                existing = _lookup_optional_artifact_row(
                    conn,
                    task_id=task_id,
                    path=final_path,
                )
                if existing is None:
                    if final_path.exists() or final_path.is_symlink():
                        raise StrategyError(
                            "interactive-tree split search path exists "
                            "without a registry row"
                        )
                    uow.promote_all()
                else:
                    _verify_existing(
                        existing,
                        task_id=task_id,
                        path=final_path,
                        content_hash=content_hash,
                        provenance=provenance,
                    )
                    uow.rollback()
                    reused = True
                _verify_file(
                    final_path,
                    content_hash=content_hash,
                    canonical=canonical,
                )
                record = runtime.task_artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=INTERACTIVE_TREE_SPLIT_SEARCH_ARTIFACT_KIND,
                    path=str(final_path),
                    content_hash=content_hash,
                    origin_tool=INTERACTIVE_TREE_SPLIT_SEARCH_ORIGIN_TOOL,
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
    artifact_id = _text(record["id"], "search artifact id")
    return {
        "artifact_id": artifact_id,
        "kind": INTERACTIVE_TREE_SPLIT_SEARCH_ARTIFACT_KIND,
        "format": "json",
        "filename": final_path.name,
        "content_hash": content_hash,
        "download_url": (
            f"/api/tasks/{quote(task_id, safe='')}"
            f"/task-artifacts/{quote(artifact_id, safe='')}/download"
        ),
    }


def _artifact_provenance(
    *,
    task_id: str,
    source: _ResolvedRevisionSource,
    context: Any | None,
    request: Mapping[str, Any] | None,
    node: Mapping[str, Any] | None,
    result: Mapping[str, Any],
    artifact_content_hash: str,
    stored: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    asset = source.automatic_source.asset
    identity = asset["identity"]
    source_tree_hash = (
        asset["asset_hash"]
        if source.parent_revision is None
        else source.parent_revision["revision_hash"]
    )
    if stored is None:
        if context is None or request is None or node is None:
            raise StrategyError(
                "interactive-tree split search provenance context is missing"
            )
        sample_ref = context.sample_design.to_ref_dict()
        node_kind = node["kind"]
        node_condition_hash = _sha256(node["condition"])
        mode = request["mode"]
    else:
        sample_ref = sample_design_ref_from_automatic_tree_source_refs(
            asset["source_refs"]
        )
        topology_node = next(
            (
                item
                for item in _topology(source)["nodes"]
                if item["node_id"] == result["source"]["node_id"]
            ),
            None,
        )
        if topology_node is None or not topology_node["is_visible"]:
            raise StrategyError(
                "interactive-tree split search node is no longer authentic"
            )
        node_kind = topology_node["kind"]
        node_condition_hash = _sha256(topology_node["condition"])
        mode = _text(stored["mode"], "provenance mode")
        if mode not in _MODES:
            raise StrategyError(
                "interactive-tree split search provenance mode changed"
            )
        feature_universe = set(
            asset["tree_result"]["training"]["feature_order"]
        )
        result_features = list(result["request"]["features"])
        if (
            not set(result_features).issubset(feature_universe)
            or (
                mode == "all_features"
                and result_features != sorted(feature_universe)
            )
        ):
            raise StrategyError(
                "interactive-tree split search feature universe changed"
            )
    provenance = {
        "schema_version": INTERACTIVE_TREE_SPLIT_SEARCH_ARTIFACT_SCHEMA_VERSION,
        "producer_version": INTERACTIVE_TREE_SPLIT_SEARCH_PRODUCER_VERSION,
        "task_id": task_id,
        "kind": INTERACTIVE_TREE_SPLIT_SEARCH_ARTIFACT_KIND,
        "format": "json",
        "search_id": result["search_id"],
        "search_hash": result["search_hash"],
        "artifact_content_hash": artifact_content_hash,
        "source_tree_id": result["source"]["source_tree_id"],
        "source_tree_hash": source_tree_hash,
        "base_asset_id": asset["asset_id"],
        "base_asset_hash": asset["asset_hash"],
        "base_tree_result_hash": asset["tree_result"]["result_hash"],
        "dataset_id": identity["dataset_id"],
        "dataset_content_hash": identity["dataset_content_hash"],
        "registry_metadata_hash": identity["registry_metadata_hash"],
        "workspace_revision": identity["workspace_revision"],
        "workspace_generation": identity["workspace_generation"],
        "semantic_mapping_hash": identity["semantic_mapping_hash"],
        "sample_design_ref": sample_ref,
        "sample_context_hash": identity["sample_context_hash"],
        "sample_partition": sample_ref["partition"],
        "node_id": result["source"]["node_id"],
        "node_kind": node_kind,
        "node_condition_hash": node_condition_hash,
        "mode": mode,
        "features": list(result["request"]["features"]),
        "min_leaf_count": result["request"]["min_leaf_count"],
        "max_thresholds_per_feature": result["request"][
            "max_thresholds_per_feature"
        ],
        "max_row_evaluations": result["budget"]["max_row_evaluations"],
        "lifecycle": result["lifecycle"],
    }
    _require_exact_fields(
        provenance,
        _PROVENANCE_FIELDS,
        "interactive-tree split search provenance",
    )
    return json.loads(_canonical_json(provenance))


def _topology(source: _ResolvedRevisionSource) -> dict[str, Any]:
    if source.parent_revision is None:
        return interactive_tree_topology_evidence(
            source.automatic_source.asset
        )
    ancestors = source.ancestor_revisions
    return interactive_tree_topology_evidence(
        source.automatic_source.asset,
        revision_payload=source.parent_revision,
        parent_revision=ancestors[0] if ancestors else None,
        ancestor_revisions=ancestors[1:],
    )


def _validate_inputs(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StrategyError(
            "interactive-tree split search inputs must be an object"
        )
    unexpected = sorted(set(value) - _INPUT_FIELDS)
    missing = sorted(_REQUIRED_INPUT_FIELDS - set(value))
    if unexpected:
        raise StrategyError(
            "interactive-tree split search inputs contain unsupported fields: "
            + ", ".join(unexpected)
        )
    if missing:
        raise StrategyError(
            "interactive-tree split search inputs are missing: "
            + ", ".join(missing)
        )
    mode = _text(value["mode"], "mode")
    if mode not in _MODES:
        raise StrategyError(
            "interactive-tree split search mode must be all_features "
            "or selected_features"
        )
    has_features = "features" in value
    if mode == "all_features" and has_features:
        raise StrategyError(
            "all_features split search must not provide features"
        )
    if mode == "selected_features" and not has_features:
        raise StrategyError(
            "selected_features split search requires features"
        )
    features = (
        None
        if not has_features
        else _features(value["features"])
    )
    return {
        "source_tree_id": _text(value["source_tree_id"], "source_tree_id"),
        "node_id": _text(value["node_id"], "node_id"),
        "mode": mode,
        "features": features,
        "max_thresholds_per_feature": _bounded_int(
            value["max_thresholds_per_feature"],
            "max_thresholds_per_feature",
            maximum=MAX_THRESHOLDS_PER_FEATURE,
        ),
        "max_row_evaluations": _bounded_int(
            value["max_row_evaluations"],
            "max_row_evaluations",
            maximum=MAX_ROW_EVALUATIONS,
        ),
    }


def _features(value: object) -> list[str]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value,
        Sequence,
    ):
        raise StrategyError(
            "interactive-tree split search features must be a list"
        )
    features = [_text(item, "feature") for item in value]
    if (
        not features
        or len(features) > MAX_SEARCH_FEATURES
        or len(features) != len(set(features))
    ):
        raise StrategyError(
            "interactive-tree split search features must be non-empty, "
            "unique, and within budget"
        )
    return sorted(features)


def _bounded_int(value: object, name: str, *, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) < 1
        or int(value) > maximum
    ):
        raise StrategyError(
            f"interactive-tree split search {name} is outside its hard budget"
        )
    return int(value)


def _lookup_artifact_row(
    conn,
    *,
    task_id: str,
    path: Path,
) -> dict[str, Any]:
    row = _lookup_optional_artifact_row(
        conn,
        task_id=task_id,
        path=path,
    )
    if row is None:
        raise StrategyError(
            "interactive-tree split search artifact was not found"
        )
    return row


def _lookup_optional_artifact_row(
    conn,
    *,
    task_id: str,
    path: Path,
) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT id, task_id, kind, path, content_hash, origin_tool,
               provenance_json, created_at
          FROM task_artifacts
         WHERE task_id = ? AND kind = ? AND path = ?
        """,
        (
            task_id,
            INTERACTIVE_TREE_SPLIT_SEARCH_ARTIFACT_KIND,
            str(path),
        ),
    ).fetchone()
    if row is None:
        return None
    return {field: row[field] for field in _TASK_ARTIFACT_ROW_FIELDS}


def _strict_provenance(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row.get("provenance_json")
    if not isinstance(raw, str):
        raise StrategyError(
            "interactive-tree split search provenance_json is invalid"
        )
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StrategyError(
            "interactive-tree split search provenance_json is invalid"
        ) from exc
    if (
        not isinstance(value, dict)
        or raw != _canonical_json(value)
    ):
        raise StrategyError(
            "interactive-tree split search provenance is not canonical"
        )
    return value


def _verify_existing(
    row: Mapping[str, Any],
    *,
    task_id: str,
    path: Path,
    content_hash: str,
    provenance: Mapping[str, Any],
) -> None:
    expected = {
        "id": stable_task_artifact_id(
            task_id=task_id,
            kind=INTERACTIVE_TREE_SPLIT_SEARCH_ARTIFACT_KIND,
            path=str(path),
        ),
        "task_id": task_id,
        "kind": INTERACTIVE_TREE_SPLIT_SEARCH_ARTIFACT_KIND,
        "path": str(path),
        "content_hash": content_hash,
        "origin_tool": INTERACTIVE_TREE_SPLIT_SEARCH_ORIGIN_TOOL,
        "provenance_json": _canonical_json(provenance),
    }
    for field, expected_value in expected.items():
        actual = row[field]
        if isinstance(expected_value, str) and isinstance(actual, str):
            matched = hmac.compare_digest(actual, expected_value)
        else:
            matched = actual == expected_value
        if not matched:
            raise StrategyError(
                "interactive-tree split search existing artifact binding changed"
            )


def _verify_file(
    path: Path,
    *,
    content_hash: str,
    canonical: bytes,
) -> None:
    observed = _read_verified_regular_file(
        path,
        root=path.parents[2],
        expected_hash=content_hash,
        max_bytes=MAX_INTERACTIVE_TREE_SPLIT_SEARCH_BYTES,
    )
    if not hmac.compare_digest(observed, canonical):
        raise StrategyError(
            "interactive-tree split search artifact content changed"
        )


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    if set(value) != expected:
        raise StrategyError(f"{name} fields changed")


def _hash(value: object, name: str) -> str:
    normalized = _text(value, name)
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise StrategyError(f"{name} is invalid")
    return normalized


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StrategyError(
            f"interactive-tree split search {name} is invalid"
        )
    return value.strip()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise StrategyError(
            "interactive-tree split search contains non-canonical JSON"
        ) from exc


__all__ = [
    "INTERACTIVE_TREE_SPLIT_SEARCH_ARTIFACT_KIND",
    "INTERACTIVE_TREE_SPLIT_SEARCH_ORIGIN_TOOL",
    "INTERACTIVE_TREE_SPLIT_SEARCH_TOOL_SCHEMA_VERSION",
    "VerifiedInteractiveTreeSplitSearch",
    "canonical_interactive_tree_split_search_path",
    "load_verified_interactive_tree_split_search",
    "run_search_interactive_tree_split_candidates",
]
