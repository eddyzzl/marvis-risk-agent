"""Governed Tool boundary for complete automatic rule-tree candidates.

The weighted-tree kernel owns every split, metric, and leaf.  This module owns
only task/dataset/workspace authorization, deterministic delivery equivalence,
and the single file/SQLite unit of work that publishes the six immutable
development-stage artifacts.  It never chooses an action or admits a leaf to a
Strategy Pool.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote

import duckdb
import pandas as pd

from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.errors import DatasetContentDriftError
from marvis.data.labels import resolve_labeled_frame
from marvis.data.workspace import (
    DataSemanticMapping,
    data_semantic_mapping_from_dict,
    data_semantic_mapping_hash,
)
from marvis.db_schema import connect
from marvis.feature.univariate import SCHEMA_VERSION as UNIVARIATE_SCHEMA_VERSION
from marvis.feature.weighted_rule_tree import (
    DEFAULT_WEIGHTED_RULE_TREE_SEED,
    WeightedRuleTreeBudgets,
    apply_weighted_rule_tree,
    build_weighted_rule_tree,
)
from marvis.files import sha256_file
from marvis.output.automatic_tree_report import (
    AUTOMATIC_TREE_REPORT_SCHEMA_VERSION,
    render_automatic_tree_report_xlsx,
)
from marvis.output.automatic_tree_visual import (
    AUTOMATIC_TREE_PNG_RENDERER_VERSION,
    AUTOMATIC_TREE_VISUAL_SCHEMA_VERSION,
    render_automatic_tree_png,
    render_automatic_tree_svg,
)
from marvis.packs.strategy.automatic_tree_asset import (
    build_automatic_tree_asset,
    canonical_automatic_tree_asset_json,
    validate_automatic_tree_asset,
)
from marvis.packs.strategy.automatic_tree_leaf_fragment import (
    AUTOMATIC_TREE_ASSET_ARTIFACT_KIND,
    AUTOMATIC_TREE_ASSET_ARTIFACT_SCHEMA_VERSION,
    AUTOMATIC_TREE_ASSET_ORIGIN_TOOL,
)
from marvis.packs.strategy.automatic_tree_sample_design import (
    sample_design_ref_from_automatic_tree_source_refs,
)
from marvis.packs.strategy.codegen import (
    generate_automatic_tree_duckdb_sql_source,
    generate_automatic_tree_python_source,
    validate_automatic_tree_duckdb_input_frame,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.sample_design_execution import (
    StrategyRiskDevelopmentExecutionBinding,
    StrategyRiskDevelopmentRef,
    bind_strategy_risk_development_frame,
    load_strategy_risk_development_execution_binding,
    require_strategy_risk_development_execution_binding_on_connection,
    revalidate_strategy_risk_development_execution_binding,
)
from marvis.repositories.data_workspace import DataWorkspaceRepository


TOOL_SCHEMA_VERSION = "strategy.automatic-tree-candidate-tool.v2"
ORIGIN_TOOL = AUTOMATIC_TREE_ASSET_ORIGIN_TOOL
SAMPLE_CONTEXT_SCHEMA_VERSION = "strategy.sample-context.v1"
EQUIVALENCE_SCHEMA_VERSION = "strategy.automatic-tree-equivalence.v1"
DELIVERY_PROVENANCE_SCHEMA_VERSION = "strategy.automatic-tree-delivery-artifact.v1"
MAX_EQUIVALENCE_SAMPLE_ROWS = 10_000

_PYTHON_CODEGEN_BOUNDARY = "strategy.automatic-tree-python-codegen/1"
_DUCKDB_SQL_CODEGEN_BOUNDARY = "strategy.automatic-tree-duckdb-sql-codegen/1"
_DELIVERY_SPECS = (
    ("json", AUTOMATIC_TREE_ASSET_ARTIFACT_KIND, "json"),
    ("python", "strategy_automatic_tree_python", "py"),
    ("duckdb_sql", "strategy_automatic_tree_duckdb_sql", "sql"),
    ("svg", "strategy_automatic_tree_svg", "svg"),
    ("png", "strategy_automatic_tree_png", "png"),
    ("xlsx", "strategy_automatic_tree_xlsx", "xlsx"),
)
_DELIVERY_BOUNDARIES = {
    "python": _PYTHON_CODEGEN_BOUNDARY,
    "duckdb_sql": _DUCKDB_SQL_CODEGEN_BOUNDARY,
    "svg": AUTOMATIC_TREE_VISUAL_SCHEMA_VERSION,
    "png": AUTOMATIC_TREE_PNG_RENDERER_VERSION,
    "xlsx": AUTOMATIC_TREE_REPORT_SCHEMA_VERSION,
}
_INPUT_FIELDS = frozenset(
    {
        "dataset_id",
        "expected_content_hash",
        "workspace_revision",
        "analysis_generation",
        "semantic_mapping_hash",
        "target_col",
        "sample_design_ref",
        "features",
        "drop_nan_labels",
        "sample_weight_col",
        "directions",
        "max_depth",
        "min_leaf_count",
        "min_weight_fraction_leaf",
        "seed",
        "loan_amount_col",
        "overdue_amount_col",
        "budgets",
    }
)
_REQUIRED_INPUT_FIELDS = frozenset(
    {
        "dataset_id",
        "expected_content_hash",
        "workspace_revision",
        "analysis_generation",
        "semantic_mapping_hash",
        "target_col",
        "sample_design_ref",
        "features",
    }
)
_CALLER_RESULT_FIELDS = frozenset(
    {
        "action",
        "actions",
        "adopt",
        "adoption",
        "default_action",
        "effect",
        "metrics",
        "pool",
        "recommendation",
        "requirements",
        "result",
        "rules",
        "selected_action",
        "strategy",
        "tree",
        "tree_result",
    }
)
_FORBIDDEN_FEATURE_ROLES = frozenset({"id", "phone", "idcard", "name", "ignore"})
_DIRECTION_VALUES = frozenset({"increasing", "decreasing", "unordered"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

FULL_TREE_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "task_id",
        "kind",
        "format",
        "asset_id",
        "asset_hash",
        "tree_result_hash",
        "dataset_id",
        "dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "registry_metadata_hash",
        "sample_context_hash",
        "sample_design_ref",
    }
)
DELIVERY_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "task_id",
        "kind",
        "format",
        "asset_id",
        "asset_hash",
        "tree_id",
        "tree_result_hash",
        "canonical_asset_content_hash",
        "delivery_boundary",
        "equivalence_schema_version",
        "equivalence_sample_hash",
        "equivalence_sample_count",
    }
)


@dataclass(frozen=True)
class _WorkspaceBinding:
    revision: int
    generation: int
    active_dataset_id: str
    active_dataset_content_hash: str
    semantic_mapping: DataSemanticMapping
    semantic_mapping_hash: str


@dataclass(frozen=True)
class _DatasetBinding:
    dataset: Any
    path: Path
    content_hash: str
    registry_metadata_hash: str
    columns: tuple[str, ...]
    row_count: int
    workspace: _WorkspaceBinding


@dataclass(frozen=True)
class _ArtifactSpec:
    artifact_format: str
    kind: str
    suffix: str
    content: bytes
    content_hash: str
    provenance: dict[str, Any]


def run_build_automatic_tree_candidate(inputs, ctx, runtime) -> dict[str, Any]:
    """Build and atomically publish one task-owned automatic-tree candidate."""

    normalized = _validate_inputs(inputs)
    task_id = _required_text(ctx.task_id, "task_id")
    binding = _load_binding(
        runtime,
        task_id=task_id,
        dataset_id=normalized["dataset_id"],
        expected_content_hash=normalized["expected_content_hash"],
        expected_revision=normalized["workspace_revision"],
        expected_generation=normalized["analysis_generation"],
        expected_semantic_hash=normalized["semantic_mapping_hash"],
    )
    sample_design = load_strategy_risk_development_execution_binding(
        runtime,
        task_id=task_id,
        sample_design_ref=normalized["sample_design_ref"],
        dataset_id=normalized["dataset_id"],
        dataset_content_hash=binding.content_hash,
        workspace_revision=binding.workspace.revision,
        workspace_generation=binding.workspace.generation,
        semantic_mapping_hash=binding.workspace.semantic_mapping_hash,
        target_col=normalized["target_col"],
        drop_nan_labels=normalized["drop_nan_labels"],
        weight_col=normalized["sample_weight_col"],
        loan_amount_col=normalized["loan_amount_col"],
        overdue_amount_col=normalized["overdue_amount_col"],
    )
    normalized = _resolve_optional_sample_design_bindings(normalized, sample_design)
    resolved = _resolve_columns(
        normalized,
        binding=binding,
        sample_design=sample_design,
    )
    frame = runtime.backend.read_frame(
        binding.path,
        columns=resolved["projected_columns"],
    )
    _require_binding_live(runtime, task_id=task_id, binding=binding)
    revalidate_strategy_risk_development_execution_binding(
        runtime,
        sample_design,
    )
    frame = bind_strategy_risk_development_frame(
        frame,
        binding=sample_design,
    )
    labeled_frame, nan_labels_dropped = resolve_labeled_frame(
        frame,
        normalized["target_col"],
        drop_nan_labels=normalized["drop_nan_labels"],
        scope="automatic-tree source dataset",
    )
    labeled_frame = labeled_frame.reset_index(drop=True)
    sample_context_hash = automatic_tree_sample_context_hash(
        task_id=task_id,
        binding=binding,
        target_col=normalized["target_col"],
        labeled_row_count=len(labeled_frame),
        drop_nan_labels=normalized["drop_nan_labels"],
        nan_labels_dropped=nan_labels_dropped,
        loan_amount_col=normalized["loan_amount_col"],
        overdue_amount_col=normalized["overdue_amount_col"],
        sample_design_ref=sample_design.to_ref_dict(),
    )

    _require_binding_live(runtime, task_id=task_id, binding=binding)
    tree_result = build_weighted_rule_tree(
        labeled_frame,
        feature_cols=normalized["features"],
        target_col=normalized["target_col"],
        sample_weight_col=normalized["sample_weight_col"],
        directions=normalized["directions"],
        max_depth=normalized["max_depth"],
        min_leaf_count=normalized["min_leaf_count"],
        min_weight_fraction_leaf=normalized["min_weight_fraction_leaf"],
        seed=normalized["seed"],
        loan_amount_col=normalized["loan_amount_col"],
        overdue_amount_col=normalized["overdue_amount_col"],
        budgets=normalized["budgets"],
    )
    _require_binding_live(runtime, task_id=task_id, binding=binding)
    asset = build_automatic_tree_asset(
        tree_result,
        task_id=task_id,
        dataset_id=normalized["dataset_id"],
        dataset_content_hash=binding.content_hash,
        workspace_revision=binding.workspace.revision,
        workspace_generation=binding.workspace.generation,
        semantic_mapping_hash=binding.workspace.semantic_mapping_hash,
        registry_metadata_hash=binding.registry_metadata_hash,
        sample_context_hash=sample_context_hash,
        source_refs=(
            f"dataset:{normalized['dataset_id']}@sha256:{binding.content_hash}",
            (
                f"data-workspace:{task_id}@revision:{binding.workspace.revision}"
                f":generation:{binding.workspace.generation}"
            ),
            f"registry-metadata:sha256:{binding.registry_metadata_hash}",
            sample_design.source_ref_token,
        ),
    )
    asset = validate_automatic_tree_asset(asset)
    _require_binding_live(runtime, task_id=task_id, binding=binding)
    full_feature_frame = labeled_frame.loc[:, normalized["features"]]
    validate_automatic_tree_duckdb_input_frame(
        full_feature_frame,
        asset,
        additional_feature_fields=normalized["features"],
    )
    sample_frame, sample_evidence = _equivalence_sample(
        labeled_frame,
        features=normalized["features"],
    )

    canonical_asset = canonical_automatic_tree_asset_json(asset).encode("utf-8")
    python_source = generate_automatic_tree_python_source(asset).encode("utf-8")
    sql_source = generate_automatic_tree_duckdb_sql_source(asset).encode("utf-8")
    svg = render_automatic_tree_svg(asset)
    png = render_automatic_tree_png(asset)
    xlsx = render_automatic_tree_report_xlsx(asset)
    delivery_bytes = {
        "json": canonical_asset,
        "python": python_source,
        "duckdb_sql": sql_source,
        "svg": svg,
        "png": png,
        "xlsx": xlsx,
    }
    _require_delivery_bytes(delivery_bytes)
    equivalence = _verify_delivery_equivalence(
        asset,
        sample_frame=sample_frame,
        sample_evidence=sample_evidence,
        python_source=python_source,
        sql_source=sql_source,
    )
    _require_binding_live(runtime, task_id=task_id, binding=binding)
    revalidate_strategy_risk_development_execution_binding(
        runtime,
        sample_design,
    )
    artifacts = _write_artifacts(
        runtime,
        task_id=task_id,
        binding=binding,
        sample_design=sample_design,
        asset=asset,
        delivery_bytes=delivery_bytes,
        equivalence=equivalence,
    )
    return _tool_output(
        asset,
        nan_labels_dropped=nan_labels_dropped,
        equivalence=equivalence,
        artifacts=artifacts,
    )


def _validate_inputs(inputs: object) -> dict[str, Any]:
    if not isinstance(inputs, Mapping):
        raise StrategyError("build_automatic_tree_candidate inputs must be an object")
    if any(not isinstance(key, str) for key in inputs):
        raise StrategyError("automatic-tree input keys must be strings")
    caller_results = sorted(set(inputs) & _CALLER_RESULT_FIELDS)
    if caller_results:
        raise StrategyError(
            "caller cannot supply automatic-tree results or actions: "
            + ", ".join(caller_results)
        )
    missing = sorted(_REQUIRED_INPUT_FIELDS - set(inputs))
    unexpected = sorted(set(inputs) - _INPUT_FIELDS)
    if missing or unexpected:
        detail = []
        if missing:
            detail.append("missing: " + ", ".join(missing))
        if unexpected:
            detail.append("unsupported: " + ", ".join(unexpected))
        raise StrategyError("invalid automatic-tree inputs (" + "; ".join(detail) + ")")

    features = _feature_list(inputs["features"])
    directions = _directions(inputs.get("directions"), features=features)
    budgets = _budgets(inputs.get("budgets"))
    return {
        "dataset_id": _required_text(inputs["dataset_id"], "dataset_id"),
        "expected_content_hash": _required_hash(
            inputs["expected_content_hash"], "expected_content_hash"
        ),
        "workspace_revision": _non_negative_int(
            inputs["workspace_revision"], "workspace_revision"
        ),
        "analysis_generation": _non_negative_int(
            inputs["analysis_generation"], "analysis_generation"
        ),
        "semantic_mapping_hash": _required_hash(
            inputs["semantic_mapping_hash"], "semantic_mapping_hash"
        ),
        "target_col": _required_text(inputs["target_col"], "target_col"),
        "sample_design_ref": StrategyRiskDevelopmentRef.from_value(
            inputs["sample_design_ref"]
        ).to_ref_dict(),
        "features": features,
        "drop_nan_labels": _optional_bool(
            inputs.get("drop_nan_labels"), default=False, field="drop_nan_labels"
        ),
        "sample_weight_col": _optional_column(
            inputs.get("sample_weight_col"), "sample_weight_col"
        ),
        "directions": directions,
        "max_depth": _positive_int(inputs.get("max_depth", 4), "max_depth"),
        "min_leaf_count": _positive_int(
            inputs.get("min_leaf_count", 200), "min_leaf_count"
        ),
        "min_weight_fraction_leaf": _fraction(
            inputs.get("min_weight_fraction_leaf", 0.0),
            "min_weight_fraction_leaf",
        ),
        "seed": _seed(inputs.get("seed", DEFAULT_WEIGHTED_RULE_TREE_SEED), "seed"),
        "loan_amount_col": _optional_column(
            inputs.get("loan_amount_col"), "loan_amount_col"
        ),
        "overdue_amount_col": _optional_column(
            inputs.get("overdue_amount_col"), "overdue_amount_col"
        ),
        "budgets": budgets,
    }


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StrategyError(f"{field} must be a non-empty string")
    normalized = value.strip()
    if "\x00" in normalized:
        raise StrategyError(f"{field} must not contain NUL bytes")
    return normalized


def _required_hash(value: object, field: str) -> str:
    normalized = _required_text(value, field).lower()
    if _SHA256_RE.fullmatch(normalized) is None:
        raise StrategyError(f"{field} must be a lowercase SHA-256 hash")
    return normalized


def _non_negative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise StrategyError(f"{field} must be a non-negative integer")
    return int(value)


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise StrategyError(f"{field} must be a positive integer")
    return int(value)


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise StrategyError(f"{field} must be an integer")
    return int(value)


def _seed(value: object, field: str) -> int:
    normalized = _integer(value, field)
    if not 0 <= normalized <= 4_294_967_295:
        raise StrategyError(f"{field} must be between 0 and 4294967295")
    return normalized


def _fraction(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 0.5
    ):
        raise StrategyError(f"{field} must be a finite number between 0 and 0.5")
    return float(value)


def _optional_bool(value: object, *, default: bool, field: str) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise StrategyError(f"{field} must be a boolean")
    return value


def _optional_column(value: object, field: str) -> str | None:
    if value is None or value == "":
        return None
    return _required_text(value, field)


def _feature_list(value: object) -> list[str]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, (list, tuple)
    ):
        raise StrategyError("features must be an ordered array")
    features = [_required_text(item, "features item") for item in value]
    if not features:
        raise StrategyError("features must contain at least one column")
    if len(features) != len(set(features)):
        raise StrategyError("features must not contain duplicates")
    return sorted(features)


def _directions(value: object, *, features: Sequence[str]) -> dict[str, str]:
    if value is None:
        supplied: Mapping[str, Any] = {}
    elif not isinstance(value, Mapping):
        raise StrategyError("directions must be an object")
    else:
        supplied = value
    if any(not isinstance(key, str) for key in supplied):
        raise StrategyError("direction feature names must be strings")
    unexpected = sorted(set(supplied) - set(features))
    if unexpected:
        raise StrategyError(
            "directions reference unselected features: " + ", ".join(unexpected)
        )
    normalized: dict[str, str] = {}
    for feature in features:
        direction = supplied.get(feature, "unordered")
        if not isinstance(direction, str) or direction not in _DIRECTION_VALUES:
            raise StrategyError(
                "direction must be increasing, decreasing, or unordered: " + feature
            )
        normalized[feature] = direction
    return normalized


def _budgets(value: object) -> WeightedRuleTreeBudgets:
    defaults = asdict(WeightedRuleTreeBudgets())
    if value is None:
        supplied: Mapping[str, Any] = {}
    elif not isinstance(value, Mapping):
        raise StrategyError("budgets must be an object")
    else:
        supplied = value
    if any(not isinstance(key, str) for key in supplied):
        raise StrategyError("budget names must be strings")
    unexpected = sorted(set(supplied) - set(defaults))
    if unexpected:
        raise StrategyError(
            "budgets contain unsupported fields: " + ", ".join(unexpected)
        )
    normalized = dict(defaults)
    for field, raw in supplied.items():
        normalized[field] = _positive_int(raw, f"budgets.{field}")
    try:
        return WeightedRuleTreeBudgets(**normalized)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise StrategyError(f"invalid automatic-tree budgets: {exc}") from exc


def _load_binding(
    runtime,
    *,
    task_id: str,
    dataset_id: str,
    expected_content_hash: str,
    expected_revision: int,
    expected_generation: int,
    expected_semantic_hash: str,
) -> _DatasetBinding:
    try:
        dataset = runtime.registry.get(dataset_id)
    except KeyError as exc:
        raise StrategyError(f"dataset not found: {dataset_id}") from exc
    if str(dataset.task_id) != task_id:
        raise StrategyError(f"dataset not found: {dataset_id}")
    registered_hash = str(dataset.content_hash or "")
    if not hmac.compare_digest(registered_hash, expected_content_hash):
        raise StrategyError(
            "automatic-tree data binding changed after user confirmation"
        )
    try:
        path = Path(runtime.registry.resolve_verified_path(dataset_id))
    except (DatasetContentDriftError, KeyError, OSError, ValueError) as exc:
        raise StrategyError(
            "automatic-tree source dataset failed immutable hash verification"
        ) from exc
    if not hmac.compare_digest(sha256_file(path), registered_hash):
        raise StrategyError("automatic-tree source dataset content hash is invalid")
    try:
        snapshot = DataWorkspaceRepository(runtime.settings.db_path).get_or_default(
            task_id
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise StrategyError("automatic-tree data workspace binding is invalid") from exc
    if (
        snapshot.active_dataset_id != dataset_id
        or snapshot.active_dataset_content_hash != registered_hash
    ):
        raise StrategyError(
            "automatic-tree dataset is not the current active task workspace dataset"
        )
    semantic_hash = data_semantic_mapping_hash(snapshot.semantic_mapping)
    if (
        snapshot.revision != expected_revision
        or snapshot.analysis_generation != expected_generation
        or not hmac.compare_digest(semantic_hash, expected_semantic_hash)
    ):
        raise StrategyError(
            "automatic-tree data binding changed after user confirmation"
        )
    with connect(runtime.settings.db_path) as conn:
        persisted = conn.execute(
            "SELECT 1 FROM data_workspaces WHERE task_id = ?", (task_id,)
        ).fetchone()
        if persisted is None:
            raise StrategyError("automatic-tree requires a persisted active workspace")
        registry_metadata_hash = _registry_metadata_hash_on_connection(
            conn,
            task_id=task_id,
            dataset_id=dataset_id,
            expected_content_hash=registered_hash,
        )
    columns = tuple(str(profile.name) for profile in dataset.columns)
    if len(columns) != len(set(columns)):
        raise StrategyError("automatic-tree dataset schema contains duplicate columns")
    if not columns or int(dataset.row_count) <= 0:
        raise StrategyError(
            "automatic-tree source dataset must contain rows and columns"
        )
    return _DatasetBinding(
        dataset=dataset,
        path=path,
        content_hash=registered_hash,
        registry_metadata_hash=registry_metadata_hash,
        columns=columns,
        row_count=int(dataset.row_count),
        workspace=_WorkspaceBinding(
            revision=snapshot.revision,
            generation=snapshot.analysis_generation,
            active_dataset_id=dataset_id,
            active_dataset_content_hash=registered_hash,
            semantic_mapping=snapshot.semantic_mapping,
            semantic_mapping_hash=semantic_hash,
        ),
    )


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
        raise StrategyError(f"dataset not found: {dataset_id}")
    registered_hash = row["content_hash"]
    if not isinstance(registered_hash, str) or not hmac.compare_digest(
        registered_hash, expected_content_hash
    ):
        raise StrategyError("automatic-tree registered dataset hash changed")
    columns_json = row["columns_json"]
    if not isinstance(columns_json, str):
        raise StrategyError("automatic-tree dataset schema is invalid")
    try:
        parsed_columns = json.loads(columns_json)
    except json.JSONDecodeError as exc:
        raise StrategyError("automatic-tree dataset schema is invalid") from exc
    if not isinstance(parsed_columns, list):
        raise StrategyError("automatic-tree dataset schema is invalid")
    payload = {
        "role": str(row["role"]),
        "row_count": int(row["row_count"]),
        "columns_json": columns_json,
        "has_target": int(row["has_target"]),
        "target_col": row["target_col"],
    }
    return _canonical_sha256(payload)


def _resolve_columns(
    normalized: Mapping[str, Any],
    *,
    binding: _DatasetBinding,
    sample_design: StrategyRiskDevelopmentExecutionBinding,
) -> dict[str, Any]:
    columns = list(binding.columns)
    available = set(columns)
    mapped_target = binding.workspace.semantic_mapping.target_col
    if mapped_target != normalized["target_col"]:
        raise StrategyError(
            "automatic-tree target_col must match the current workspace semantic target"
        )
    required = {
        normalized["target_col"],
        *normalized["features"],
        *(
            column
            for column in (
                normalized["sample_weight_col"],
                normalized["loan_amount_col"],
                normalized["overdue_amount_col"],
            )
            if column is not None
        ),
    }
    missing = sorted(required - available)
    if missing:
        raise StrategyError("unknown automatic-tree columns: " + ", ".join(missing))

    governed_feature_conflicts = sorted(
        set(normalized["features"])
        & set(sample_design.excluded_feature_columns)
    )
    if (
        sample_design.split_column is not None
        and sample_design.split_column in governed_feature_conflicts
    ):
        raise StrategyError(
            "sample-design split column cannot be an automatic-tree feature"
        )
    if governed_feature_conflicts:
        raise StrategyError(
            "sample-design governed target, partition, or population columns "
            "cannot be automatic-tree features: "
            + ", ".join(governed_feature_conflicts)
        )
    missing_partition_columns = sorted(
        set(sample_design.partition_columns) - available
    )
    if missing_partition_columns:
        raise StrategyError(
            "sample-design partition columns are missing from the dataset: "
            + ", ".join(missing_partition_columns)
        )

    roles = {
        str(column): str(role)
        for column, role in binding.workspace.semantic_mapping.field_roles.items()
    }
    inferred_sensitive: dict[str, str] = {}
    for profile in binding.dataset.columns:
        column = str(profile.name)
        role = str(profile.semantic_role or "")
        if role in _FORBIDDEN_FEATURE_ROLES:
            inferred_sensitive[column] = role
        if role:
            roles.setdefault(column, role)
    roles.update(inferred_sensitive)
    forbidden = sorted(
        feature
        for feature in normalized["features"]
        if roles.get(feature) in _FORBIDDEN_FEATURE_ROLES
    )
    if forbidden:
        raise StrategyError(
            "identifier, personal-data, or ignored fields cannot be automatic-tree "
            "features: " + ", ".join(forbidden)
        )

    target_col = normalized["target_col"]
    role_columns = [
        target_col,
        normalized["sample_weight_col"],
        normalized["loan_amount_col"],
        normalized["overdue_amount_col"],
    ]
    assigned = [column for column in role_columns if column is not None]
    duplicates = sorted({column for column in assigned if assigned.count(column) > 1})
    feature_conflicts = sorted(set(normalized["features"]) & set(assigned))
    if duplicates or feature_conflicts:
        conflicts = sorted(set(duplicates) | set(feature_conflicts))
        raise StrategyError(
            "target, features, weight, loan, and overdue roles must use distinct "
            "columns: " + ", ".join(conflicts)
        )
    projected = [*normalized["features"]]
    projected.extend(
        column
        for column in (
            normalized["sample_weight_col"],
            normalized["loan_amount_col"],
            normalized["overdue_amount_col"],
            target_col,
        )
        if column is not None and column not in projected
    )
    projected.extend(
        column
        for column in sample_design.partition_columns
        if column not in projected
    )
    return {"projected_columns": projected, "field_roles": roles}


def _resolve_optional_sample_design_bindings(
    normalized: Mapping[str, Any],
    sample_design: StrategyRiskDevelopmentExecutionBinding,
) -> dict[str, Any]:
    """Inherit omitted optional columns from the authenticated sample design.

    An explicit non-empty caller binding is still an assertion and therefore
    must match exactly.  ``None`` only means that the caller delegates the
    binding choice to the immutable sample-design artifact.
    """

    expected = {
        "sample_weight_col": sample_design.weight_col,
        "loan_amount_col": sample_design.loan_amount_col,
        "overdue_amount_col": sample_design.overdue_amount_col,
    }
    resolved = dict(normalized)
    for field, designed in expected.items():
        requested = normalized[field]
        if requested is not None and requested != designed:
            raise StrategyError(
                f"strategy sample-design {field} does not match automatic-tree binding"
            )
        resolved[field] = designed
    return resolved


def automatic_tree_sample_context_hash(
    *,
    task_id: str,
    binding: _DatasetBinding,
    target_col: str,
    labeled_row_count: int,
    drop_nan_labels: bool,
    nan_labels_dropped: int,
    loan_amount_col: str | None,
    overdue_amount_col: str | None,
    sample_design_ref: Mapping[str, Any],
) -> str:
    """Return the cross-candidate labelled-sample identity projection."""

    context = {
        "schema_version": SAMPLE_CONTEXT_SCHEMA_VERSION,
        "identity": {
            "task_id": task_id,
            "dataset_id": str(binding.dataset.id),
            "dataset_content_hash": binding.content_hash,
            "workspace_revision": binding.workspace.revision,
            "workspace_generation": binding.workspace.generation,
            "semantic_mapping_hash": binding.workspace.semantic_mapping_hash,
        },
        "analysis": {
            "schema_version": UNIVARIATE_SCHEMA_VERSION,
            "target": target_col,
            "target_definition": {"good": 0, "bad": 1},
            "row_count": int(labeled_row_count),
        },
        "sample_parameters": {
            "analysis_schema_version": UNIVARIATE_SCHEMA_VERSION,
            "target_col": target_col,
            "drop_nan_labels": bool(drop_nan_labels),
            "nan_labels_dropped": int(nan_labels_dropped),
            "loan_amount_col": loan_amount_col,
            "overdue_amount_col": overdue_amount_col,
            "registry_metadata_hash": binding.registry_metadata_hash,
            "sample_design_ref": StrategyRiskDevelopmentRef.from_value(
                sample_design_ref
            ).to_ref_dict(),
        },
    }
    return _canonical_sha256(context)


def _require_binding_live(runtime, *, task_id: str, binding: _DatasetBinding) -> None:
    with connect(runtime.settings.db_path) as conn:
        _require_binding_on_connection(conn, task_id=task_id, binding=binding)
    if not hmac.compare_digest(sha256_file(binding.path), binding.content_hash):
        raise StrategyError("automatic-tree source dataset changed during generation")


def _require_binding_on_connection(
    conn, *, task_id: str, binding: _DatasetBinding
) -> None:
    task = conn.execute("SELECT 1 FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if task is None:
        raise StrategyError(f"task not found: {task_id}")
    live_metadata_hash = _registry_metadata_hash_on_connection(
        conn,
        task_id=task_id,
        dataset_id=str(binding.dataset.id),
        expected_content_hash=binding.content_hash,
    )
    if not hmac.compare_digest(live_metadata_hash, binding.registry_metadata_hash):
        raise StrategyError("automatic-tree dataset metadata changed during generation")
    row = conn.execute(
        """
        SELECT revision, active_dataset_id, active_dataset_content_hash,
               analysis_generation, semantic_mapping_json
          FROM data_workspaces
         WHERE task_id = ?
        """,
        (task_id,),
    ).fetchone()
    if row is None:
        raise StrategyError(
            "automatic-tree data workspace disappeared during generation"
        )
    mapping_json = row["semantic_mapping_json"]
    if not isinstance(mapping_json, str):
        raise StrategyError("automatic-tree semantic mapping is invalid")
    try:
        mapping = data_semantic_mapping_from_dict(json.loads(mapping_json))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise StrategyError("automatic-tree semantic mapping is invalid") from exc
    live_semantic_hash = data_semantic_mapping_hash(mapping)
    expected = binding.workspace
    active_hash = row["active_dataset_content_hash"]
    if (
        int(row["revision"]) != expected.revision
        or int(row["analysis_generation"]) != expected.generation
        or row["active_dataset_id"] != expected.active_dataset_id
        or active_hash != expected.active_dataset_content_hash
        or not isinstance(active_hash, str)
        or not hmac.compare_digest(active_hash, binding.content_hash)
        or not hmac.compare_digest(live_semantic_hash, expected.semantic_mapping_hash)
    ):
        raise StrategyError("automatic-tree data workspace changed during generation")


def _equivalence_sample(
    frame: pd.DataFrame, *, features: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    row_count = int(len(frame))
    if row_count <= MAX_EQUIVALENCE_SAMPLE_ROWS:
        positions = list(range(row_count))
        selection_rule = "all_rows_in_source_order"
    else:
        denominator = MAX_EQUIVALENCE_SAMPLE_ROWS - 1
        positions = [
            (index * (row_count - 1)) // denominator
            for index in range(MAX_EQUIVALENCE_SAMPLE_ROWS)
        ]
        selection_rule = "evenly_spaced_source_positions_including_endpoints"
    sample = frame.iloc[positions].loc[:, list(features)].reset_index(drop=True).copy()
    sample_hash = _canonical_sha256(
        {
            "schema_version": "strategy.automatic-tree-equivalence-sample.v1",
            "source_row_count": row_count,
            "feature_order": list(features),
            "source_positions": positions,
            "rows": [
                [_sample_scalar(value) for value in row]
                for row in sample.itertuples(index=False, name=None)
            ],
        }
    )
    return sample, {
        "selection_rule": selection_rule,
        "source_row_count": row_count,
        "sample_count": len(sample),
        "sample_hash": sample_hash,
    }


def _sample_scalar(value: object) -> list[str]:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, Integral)) and bool(missing):
        return ["missing", ""]
    if isinstance(value, bool):
        return ["boolean", "true" if value else "false"]
    if isinstance(value, Integral):
        return ["integer", str(int(value))]
    if isinstance(value, Real):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise StrategyError("automatic-tree equivalence sample must be finite")
        return ["float64", numeric.hex()]
    if isinstance(value, str):
        return ["string", value]
    raise StrategyError(
        "automatic-tree equivalence sample contains an unsupported scalar type"
    )


def _verify_delivery_equivalence(
    asset: Mapping[str, Any],
    *,
    sample_frame: pd.DataFrame,
    sample_evidence: Mapping[str, Any],
    python_source: bytes,
    sql_source: bytes,
) -> dict[str, Any]:
    tree = asset["tree_result"]
    reference_leaf_ids = apply_weighted_rule_tree(sample_frame, tree).tolist()
    rule_by_leaf = {rule["leaf_id"]: rule["rule_id"] for rule in tree["rules"]}
    reference_rows = [
        {"leaf_id": leaf_id, "rule_id": rule_by_leaf[leaf_id]}
        for leaf_id in reference_leaf_ids
    ]

    namespace: dict[str, Any] = {"__name__": "__marvis_automatic_tree_equivalence__"}
    try:
        compiled = compile(
            python_source.decode("utf-8"),
            "<generated-automatic-tree.py>",
            "exec",
        )
        exec(compiled, namespace)  # noqa: S102 - trusted deterministic generator output
        python_apply_rows = namespace["apply_rows"]
        python_rows = python_apply_rows(sample_frame)
    except Exception as exc:
        raise StrategyError(
            f"generated automatic-tree Python failed equivalence execution: {exc}"
        ) from exc
    if python_rows != reference_rows:
        raise StrategyError(
            "generated automatic-tree Python does not match the kernel reference"
        )

    try:
        with duckdb.connect() as connection:
            validate_automatic_tree_duckdb_input_frame(sample_frame, asset)
            connection.register("input_rows", sample_frame)
            sql_frame = connection.sql(sql_source.decode("utf-8")).df()
    except Exception as exc:
        raise StrategyError(
            f"generated automatic-tree DuckDB SQL failed equivalence execution: {exc}"
        ) from exc
    if sql_frame.columns.tolist() != [
        "__marvis_row_ordinal",
        "leaf_id",
        "rule_id",
    ]:
        raise StrategyError(
            "generated automatic-tree DuckDB SQL returned invalid columns"
        )
    sql_rows = sql_frame.to_dict(orient="records")
    expected_sql_rows = [
        {
            "__marvis_row_ordinal": index,
            "leaf_id": row["leaf_id"],
            "rule_id": row["rule_id"],
        }
        for index, row in enumerate(reference_rows)
    ]
    normalized_sql_rows = [
        {
            "__marvis_row_ordinal": int(row["__marvis_row_ordinal"]),
            "leaf_id": str(row["leaf_id"]),
            "rule_id": str(row["rule_id"]),
        }
        for row in sql_rows
    ]
    if normalized_sql_rows != expected_sql_rows:
        raise StrategyError(
            "generated automatic-tree DuckDB SQL does not match the kernel reference"
        )

    reference_hash = _canonical_sha256(reference_rows)
    python_hash = _canonical_sha256(python_rows)
    sql_projection = [
        {"leaf_id": row["leaf_id"], "rule_id": row["rule_id"]}
        for row in normalized_sql_rows
    ]
    sql_hash = _canonical_sha256(sql_projection)
    return {
        "schema_version": EQUIVALENCE_SCHEMA_VERSION,
        "matched": True,
        "selection_rule": sample_evidence["selection_rule"],
        "source_row_count": int(sample_evidence["source_row_count"]),
        "sample_count": int(sample_evidence["sample_count"]),
        "sample_hash": str(sample_evidence["sample_hash"]),
        "reference_result_hash": reference_hash,
        "python_result_hash": python_hash,
        "duckdb_sql_result_hash": sql_hash,
    }


def _require_delivery_bytes(bundle: Mapping[str, Any]) -> None:
    expected = {artifact_format for artifact_format, _kind, _suffix in _DELIVERY_SPECS}
    if set(bundle) != expected or any(
        not isinstance(content, bytes) or not content for content in bundle.values()
    ):
        raise StrategyError(
            "automatic-tree delivery bundle must contain exactly six non-empty byte artifacts"
        )


def _write_artifacts(
    runtime,
    *,
    task_id: str,
    binding: _DatasetBinding,
    sample_design: StrategyRiskDevelopmentExecutionBinding,
    asset: Mapping[str, Any],
    delivery_bytes: Mapping[str, bytes],
    equivalence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    _require_delivery_bytes(delivery_bytes)
    asset_id = str(asset["asset_id"])
    asset_hash = str(asset["asset_hash"])
    tree = asset["tree_result"]
    tree_id = str(tree["tree"]["tree_id"])
    tree_result_hash = str(tree["result_hash"])
    canonical_asset_content_hash = hashlib.sha256(delivery_bytes["json"]).hexdigest()
    out_dir = _safe_task_artifact_directory(
        runtime.settings,
        task_id=task_id,
        child="strategy_automatic_trees",
    )

    specs: list[_ArtifactSpec] = []
    for artifact_format, kind, suffix in _DELIVERY_SPECS:
        content = delivery_bytes[artifact_format]
        content_hash = hashlib.sha256(content).hexdigest()
        if artifact_format == "json":
            provenance = {
                "schema_version": AUTOMATIC_TREE_ASSET_ARTIFACT_SCHEMA_VERSION,
                "producer_version": asset["producer_version"],
                "task_id": task_id,
                "kind": kind,
                "format": artifact_format,
                "asset_id": asset_id,
                "asset_hash": asset_hash,
                "tree_result_hash": tree_result_hash,
                "dataset_id": asset["identity"]["dataset_id"],
                "dataset_content_hash": asset["identity"]["dataset_content_hash"],
                "workspace_revision": asset["identity"]["workspace_revision"],
                "workspace_generation": asset["identity"]["workspace_generation"],
                "semantic_mapping_hash": asset["identity"]["semantic_mapping_hash"],
                "registry_metadata_hash": asset["identity"]["registry_metadata_hash"],
                "sample_context_hash": asset["identity"]["sample_context_hash"],
                "sample_design_ref": sample_design.to_ref_dict(),
            }
            if set(provenance) != FULL_TREE_PROVENANCE_FIELDS:
                raise StrategyError("automatic-tree JSON provenance fields drifted")
        else:
            provenance = {
                "schema_version": DELIVERY_PROVENANCE_SCHEMA_VERSION,
                "producer_version": TOOL_SCHEMA_VERSION,
                "task_id": task_id,
                "kind": kind,
                "format": artifact_format,
                "asset_id": asset_id,
                "asset_hash": asset_hash,
                "tree_id": tree_id,
                "tree_result_hash": tree_result_hash,
                "canonical_asset_content_hash": canonical_asset_content_hash,
                "delivery_boundary": _DELIVERY_BOUNDARIES[artifact_format],
                "equivalence_schema_version": equivalence["schema_version"],
                "equivalence_sample_hash": equivalence["sample_hash"],
                "equivalence_sample_count": equivalence["sample_count"],
            }
            if set(provenance) != DELIVERY_PROVENANCE_FIELDS:
                raise StrategyError("automatic-tree delivery provenance fields drifted")
        specs.append(
            _ArtifactSpec(
                artifact_format=artifact_format,
                kind=kind,
                suffix=suffix,
                content=content,
                content_hash=content_hash,
                provenance=provenance,
            )
        )

    uow = ArtifactUnitOfWork()
    staged_specs = []
    records: list[dict[str, Any]] = []
    db_committed = False
    rollback_attempted_under_lock = False
    try:
        for spec in specs:
            staged = uow.stage_file(out_dir, f"{asset_id}.{spec.suffix}")
            staged.path.write_bytes(spec.content)
            staged_specs.append((spec, staged))
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                _require_binding_on_connection(conn, task_id=task_id, binding=binding)
                require_strategy_risk_development_execution_binding_on_connection(
                    conn,
                    sample_design,
                )
                _require_artifact_directory_boundary(
                    out_dir,
                    root=Path(runtime.settings.tasks_dir),
                )
                if not hmac.compare_digest(
                    sha256_file(binding.path), binding.content_hash
                ):
                    raise StrategyError(
                        "automatic-tree source dataset changed before registration"
                    )
                uow.promote_all()
                for spec, staged in staged_specs:
                    if not hmac.compare_digest(
                        sha256_file(staged.final_path), spec.content_hash
                    ):
                        raise StrategyError(
                            "automatic-tree delivery changed before registration"
                        )
                    records.append(
                        runtime.task_artifacts.register_on_connection(
                            conn,
                            task_id=task_id,
                            kind=spec.kind,
                            path=str(staged.final_path),
                            content_hash=spec.content_hash,
                            origin_tool=ORIGIN_TOOL,
                            provenance=spec.provenance,
                        )
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

    return [
        {
            "artifact_id": str(record["id"]),
            "kind": spec.kind,
            "format": spec.artifact_format,
            "filename": staged.final_path.name,
            "content_hash": spec.content_hash,
            "download_url": (
                f"/api/tasks/{quote(task_id, safe='')}"
                f"/task-artifacts/{quote(str(record['id']), safe='')}/download"
            ),
        }
        for (spec, staged), record in zip(staged_specs, records, strict=True)
    ]


def _safe_task_artifact_directory(settings, *, task_id: str, child: str) -> Path:
    if Path(task_id).name != task_id or task_id in {".", ".."}:
        raise StrategyError("task id is unsafe for automatic-tree artifact storage")
    if Path(child).name != child or child in {".", ".."}:
        raise StrategyError("automatic-tree artifact directory name is unsafe")
    declared_root = Path(settings.tasks_dir).absolute()
    try:
        if declared_root.is_symlink():
            raise StrategyError("task artifact root must not be a symlink")
        declared_root.mkdir(parents=True, exist_ok=True)
        resolved_root = declared_root.resolve(strict=True)
        task_root = declared_root / task_id
        if task_root.is_symlink():
            raise StrategyError("task artifact directory must not be a symlink")
        task_root.mkdir(exist_ok=True)
        resolved_task = task_root.resolve(strict=True)
        if resolved_task.parent != resolved_root:
            raise StrategyError("task artifact directory escaped task storage")
        artifact_dir = task_root / child
        if artifact_dir.is_symlink():
            raise StrategyError(
                "automatic-tree artifact directory must not be a symlink"
            )
        artifact_dir.mkdir(exist_ok=True)
        if artifact_dir.resolve(strict=True).parent != resolved_task:
            raise StrategyError(
                "automatic-tree artifact directory escaped task storage"
            )
    except OSError as exc:
        raise StrategyError("automatic-tree artifact directory is unavailable") from exc
    return artifact_dir


def _require_artifact_directory_boundary(path: Path, *, root: Path) -> None:
    declared_root = root.absolute()
    if declared_root.is_symlink():
        raise StrategyError("task artifact root must not be a symlink")
    try:
        resolved_root = declared_root.resolve(strict=True)
        resolved_path = path.resolve(strict=True)
        resolved_path.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise StrategyError(
            "automatic-tree artifact directory escaped task storage"
        ) from exc
    current = path.absolute()
    while True:
        if current.is_symlink():
            raise StrategyError(
                "automatic-tree artifact directory must not use symlinks"
            )
        if current.exists() and not current.is_dir():
            raise StrategyError("automatic-tree artifact directory must be a directory")
        if current == declared_root:
            break
        if current == current.parent:
            raise StrategyError(
                "automatic-tree artifact directory escaped task storage"
            )
        current = current.parent
    if resolved_path == resolved_root:
        raise StrategyError("automatic-tree artifacts require a task subdirectory")


def _tool_output(
    asset: Mapping[str, Any],
    *,
    nan_labels_dropped: int,
    equivalence: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    tree_result = asset["tree_result"]
    tree = tree_result["tree"]
    identity = asset["identity"]
    lifecycle = asset["lifecycle"]
    sample_weight = tree_result["training"]["sample_weight"]
    primary_metric_basis = (
        "weighted" if sample_weight["status"] == "available" else "unweighted"
    )
    return {
        "schema_version": TOOL_SCHEMA_VERSION,
        "summary": {
            "asset_id": asset["asset_id"],
            "asset_hash": asset["asset_hash"],
            "candidate_id": asset["candidate_evidence"]["candidate_id"],
            "candidate_evidence_hash": asset["candidate_evidence"]["evidence_hash"],
            "tree_id": tree["tree_id"],
            "tree_result_hash": tree_result["result_hash"],
            "candidate_stage": lifecycle["candidate_stage"],
            "observation_stage": lifecycle["observation_stage"],
            "validation_status": lifecycle["validation_status"],
            "dataset_id": identity["dataset_id"],
            "dataset_content_hash": identity["dataset_content_hash"],
            "workspace_revision": identity["workspace_revision"],
            "workspace_generation": identity["workspace_generation"],
            "semantic_mapping_hash": identity["semantic_mapping_hash"],
            "registry_metadata_hash": identity["registry_metadata_hash"],
            "sample_context_hash": identity["sample_context_hash"],
            "sample_design_ref": _sample_design_ref_from_source_refs(
                asset["source_refs"]
            ),
            "target_col": tree_result["training"]["target_col"],
            "features": list(tree_result["training"]["feature_order"]),
            "nan_labels_dropped": int(nan_labels_dropped),
            "training_row_count": tree_result["training"]["row_count"],
            "node_count": tree["node_count"],
            "leaf_count": tree["leaf_count"],
        },
        "leaf_index": [
            {
                "leaf_id": fragment["leaf_id"],
                "fragment_id": fragment["fragment_id"],
                "fragment_hash": fragment["fragment_hash"],
                "rule_id": fragment["rule_id"],
                "effect_id": fragment["effect_id"],
                "condition": deepcopy(fragment["condition"]),
                "requirements": deepcopy(fragment["requirements"]),
                "metric_basis": {
                    "primary": primary_metric_basis,
                    "sample_weight": deepcopy(sample_weight),
                },
                "measurements": deepcopy(fragment["metrics"]),
            }
            for fragment in asset["fragments"]
        ],
        "report_info_gaps": _report_info_gaps(tree_result),
        "red_flags": [dict(flag) for flag in asset["diagnostics"]["red_flags"]],
        "equivalence": dict(equivalence),
        "artifacts": [dict(artifact) for artifact in artifacts],
    }


def _report_info_gaps(tree_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project only deterministic, nonblocking optional reporting gaps."""

    training = tree_result["training"]
    absent_contexts = (
        (
            training["sample_weight"]["status"] == "not_applicable",
            "sample_weight_not_provided",
            "sample_weight",
        ),
        (
            training["loan_amount_col"] is None,
            "loan_amount_not_provided",
            "loan_amount",
        ),
        (
            training["overdue_amount_col"] is None,
            "overdue_amount_not_provided",
            "overdue_amount",
        ),
    )
    return [
        {"code": code, "context": context, "blocking": False}
        for absent, code, context in absent_contexts
        if absent
    ]


def strategy_sample_design_ref_from_source_refs(
    source_refs: object,
) -> dict[str, str]:
    """Recover the one exact canonical sample-design reference from an asset."""

    return sample_design_ref_from_automatic_tree_source_refs(source_refs)


def _sample_design_ref_from_source_refs(source_refs: object) -> dict[str, str]:
    return strategy_sample_design_ref_from_source_refs(source_refs)


def _canonical_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise StrategyError(
            "automatic-tree evidence must be finite canonical JSON"
        ) from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "DELIVERY_PROVENANCE_FIELDS",
    "FULL_TREE_PROVENANCE_FIELDS",
    "TOOL_SCHEMA_VERSION",
    "automatic_tree_sample_context_hash",
    "run_build_automatic_tree_candidate",
    "strategy_sample_design_ref_from_source_refs",
]
