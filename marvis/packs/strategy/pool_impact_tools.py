"""Governed task-artifact boundary for Strategy Pool impact evidence."""

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
from marvis.data.errors import (
    DataLayerError,
    DatasetContentDriftError,
    NanLabelNotConfirmedError,
)
from marvis.data.labels import require_labels_confirmed
from marvis.data.workspace import (
    data_semantic_mapping_from_dict,
    data_semantic_mapping_hash,
)
from marvis.files import sha256_file
from marvis.packs.strategy.automatic_tree_sample_design import (
    sample_design_ref_from_automatic_tree_source_refs,
)
from marvis.packs.strategy.dsl import strategy_spec_hash
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pool import compile_strategy_pool, validate_strategy_pool
from marvis.packs.strategy.pool_impact import (
    STRATEGY_POOL_IMPACT_PRODUCER_VERSION,
    build_strategy_pool_impact_assessment,
    canonical_strategy_pool_impact_json,
    validate_strategy_pool_impact_assessment,
)
from marvis.packs.strategy.sample_design_binding import (
    StrategySampleDesignExecutionBinding,
    StrategySampleDesignRef,
    bind_strategy_development_frame,
    load_strategy_sample_design_execution_binding,
    revalidate_strategy_sample_design_execution_binding,
)
from marvis.packs.strategy.sample_design_tools import (
    SAMPLE_DESIGN_ARTIFACT_KIND,
    SAMPLE_DESIGN_ORIGIN_TOOL,
)
from marvis.packs.strategy.voting_candidate import (
    VOTING_CANDIDATE_ASSET_SCHEMA_VERSION,
)
from marvis.repositories.data_workspace import (
    DataWorkspaceDataError,
    DataWorkspaceDatasetNotFound,
    DataWorkspaceRepository,
)
from marvis.repositories.strategy import (
    _strategy_from_row,
    _strategy_spec_hash_from_row,
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


POOL_IMPACT_TOOL_SCHEMA_VERSION = "strategy.measure-pool-impact-tool.v2"
POOL_IMPACT_ARTIFACT_KIND = "strategy_pool_impact_json"
POOL_IMPACT_ARTIFACT_SCHEMA_VERSION = "strategy.pool-impact-artifact.v2"
POOL_IMPACT_ORIGIN_TOOL = "strategy.measure_pool_impact"

_INPUT_FIELDS = frozenset(
    {
        "strategy_type",
        "expected_pool_revision",
        "expected_pool_snapshot_hash",
        "dataset_id",
        "expected_dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "target_col",
        "comparison_mode",
        "baseline_strategy_id",
        "month_col",
        "loan_amount_col",
        "overdue_amount_col",
        "drop_nan_labels",
        "sample_design_ref",
    }
)
_OPTIONAL_FIELDS = frozenset(
    {
        "baseline_strategy_id",
        "month_col",
        "loan_amount_col",
        "overdue_amount_col",
        "drop_nan_labels",
    }
)
_REQUIRED_FIELDS = _INPUT_FIELDS - _OPTIONAL_FIELDS
_OUTPUT_FIELDS = frozenset(
    {
        "schema_version",
        "assessment_id",
        "content_hash",
        "pool_id",
        "revision",
        "snapshot_hash",
        "design_hash",
        "strategy_type",
        "comparison_mode",
        "population_count",
        "labeled_count",
        "nan_labels_excluded",
        "monthly_status",
        "assessment",
        "warnings",
        "artifacts",
        "not_created_strategy",
        "not_adopted",
        "not_deployed",
    }
)
_OUTPUT_ARTIFACT_FIELDS = frozenset(
    {
        "artifact_id",
        "kind",
        "format",
        "filename",
        "content_hash",
        "download_url",
    }
)
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_TASK_ARTIFACT_ROW_FIELDS = (
    "id",
    "task_id",
    "kind",
    "path",
    "content_hash",
    "origin_tool",
    "provenance_json",
    "created_at",
)
_BOUNDARY_ERRORS = (
    DataLayerError,
    DataWorkspaceDataError,
    DataWorkspaceDatasetNotFound,
    DatasetContentDriftError,
    StrategyCandidatePoolConflictError,
    StrategyCandidatePoolDataError,
    StrategyCandidatePoolNotFoundError,
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
)


@dataclass(frozen=True)
class _DatasetBinding:
    dataset: Any
    path: Path
    source_path: str
    dataset_id: str
    content_hash: str
    registry_metadata_hash: str
    row_count: int
    columns: tuple[str, ...]
    workspace_revision: int
    workspace_generation: int
    semantic_mapping_hash: str
    target_col: str


@dataclass(frozen=True)
class _BaselineBinding:
    strategy_id: str
    strategy_type: str
    spec: dict[str, Any]
    spec_hash: str


def run_measure_pool_impact(inputs, ctx, runtime) -> dict[str, Any]:
    """Measure and atomically publish one exact current Pool assessment."""

    try:
        request = _validate_inputs(inputs)
        task_id = _text(ctx.task_id, "task_id")
        repository = StrategyCandidatePoolRepository(runtime.settings.db_path)
        pool = repository.get_current(task_id, request["strategy_type"])
        if pool is None:
            raise StrategyError("strategy candidate pool not found")
        pool = validate_strategy_pool(pool)
        _require_pool_cas(pool, request)
        if not pool["entries"]:
            raise StrategyError("cannot measure an empty Strategy Pool")

        # Reuse the Pool's single candidate-adapter seam.  This verifies every
        # concrete artifact, parent artifact, dataset, and nested Voting lineage.
        from marvis.packs.strategy import pool_tools

        cache = pool_tools._LineageCache.empty()
        lineages = pool_tools._load_pool_lineages(
            runtime,
            task_id=task_id,
            pool=pool,
            cache=cache,
        )
        pool_artifact = pool_tools._normalize_source_record(
            pool_tools._load_pool_artifact(runtime, task_id=task_id, snapshot=pool)
        )
        selected = compile_strategy_pool(pool)
        if selected["requirements"]:
            raise StrategyError(
                "Pool impact cannot execute unresolved candidate requirements"
            )
        sample = _pool_sample_binding(pool, task_id=task_id)
        dataset = _load_dataset_binding(
            runtime,
            request=request,
            task_id=task_id,
            sample=sample,
        )
        sample_design = load_strategy_sample_design_execution_binding(
            runtime,
            task_id=task_id,
            sample_design_ref=request["sample_design_ref"],
            dataset_id=dataset.dataset_id,
            dataset_content_hash=dataset.content_hash,
            workspace_revision=dataset.workspace_revision,
            workspace_generation=dataset.workspace_generation,
            semantic_mapping_hash=dataset.semantic_mapping_hash,
            target_col=dataset.target_col,
            drop_nan_labels=request["drop_nan_labels"],
            month_col=request.get("month_col"),
            loan_amount_col=request.get("loan_amount_col"),
            overdue_amount_col=request.get("overdue_amount_col"),
        )
        request = _resolve_sample_design_optional_bindings(request, sample_design)
        _require_pool_sample_design_ref(
            lineages,
            expected=sample_design.reference,
        )
        _require_pool_measurement_target(
            lineages,
            expected_target_col=dataset.target_col,
        )
        baseline = _load_baseline(
            runtime,
            request=request,
            task_id=task_id,
        )
        frame = _read_frame(
            runtime,
            dataset=dataset,
            sample_design=sample_design,
            strategy_spec=selected["strategy_spec"],
            baseline_spec=None if baseline is None else baseline.spec,
            request=request,
        )
        frame = bind_strategy_development_frame(frame, binding=sample_design)
        nan_labels_excluded = require_labels_confirmed(
            frame,
            dataset.target_col,
            drop_nan_labels=request["drop_nan_labels"],
            scope="Strategy Pool impact source dataset",
        )
        assessment = build_strategy_pool_impact_assessment(
            pool=pool,
            frame=frame,
            sample_binding=sample,
            sample_design_ref=sample_design.to_ref_dict(),
            target_col=dataset.target_col,
            target_bad_value=1,
            month_col=request.get("month_col"),
            loan_amount_col=request.get("loan_amount_col"),
            overdue_amount_col=request.get("overdue_amount_col"),
            comparison_mode=request["comparison_mode"],
            baseline_spec=None if baseline is None else baseline.spec,
            baseline_binding=(
                None
                if baseline is None
                else {
                    "strategy_id": baseline.strategy_id,
                    "strategy_type": baseline.strategy_type,
                    "spec_hash": baseline.spec_hash,
                }
            ),
        )
        if sha256_file(dataset.path) != dataset.content_hash:
            raise StrategyError(
                "source dataset changed while Pool impact was being measured"
            )
        revalidated_sample_design = (
            revalidate_strategy_sample_design_execution_binding(
                runtime,
                sample_design,
            )
        )
        if revalidated_sample_design != sample_design:
            raise StrategyError(
                "strategy sample-design changed while Pool impact was measured"
            )
        return _persist_assessment(
            runtime,
            repository=repository,
            request=request,
            task_id=task_id,
            pool=pool,
            pool_artifact=pool_artifact,
            lineages=lineages,
            dataset=dataset,
            sample_design=sample_design,
            baseline=baseline,
            assessment=assessment,
            nan_labels_excluded=nan_labels_excluded,
        )
    except StrategyError:
        raise
    except NanLabelNotConfirmedError:
        # Preserve the structured confirmation contract consumed by the
        # Agent/Runner instead of flattening it to an ordinary StrategyError.
        raise
    except _BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc


def validate_measure_pool_impact_tool_output(value: object) -> dict[str, Any]:
    """Fail closed when cached Tool output drifts from its canonical assessment.

    This validates the self-contained Tool envelope and its declared artifact-byte
    hash. Authenticating persisted bytes still belongs to TaskArtifact registry
    lookup, whose trusted ``content_hash`` is outside this cached output.
    """

    if not isinstance(value, Mapping) or set(value) != _OUTPUT_FIELDS:
        raise StrategyError("measure_pool_impact output envelope is invalid")
    try:
        normalized = json.loads(
            json.dumps(
                dict(value),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError, RecursionError, json.JSONDecodeError) as exc:
        raise StrategyError("measure_pool_impact output must be canonical JSON") from exc

    try:
        assessment = validate_strategy_pool_impact_assessment(
            normalized["assessment"]
        )
    except RecursionError as exc:
        raise StrategyError(
            "measure_pool_impact output must be canonical JSON"
        ) from exc
    identity = assessment["identity"]
    population = assessment["population"]
    expected = {
        "schema_version": POOL_IMPACT_TOOL_SCHEMA_VERSION,
        "assessment_id": assessment["assessment_id"],
        "content_hash": assessment["content_hash"],
        "pool_id": identity["pool_id"],
        "revision": identity["revision"],
        "snapshot_hash": identity["snapshot_hash"],
        "design_hash": identity["design_hash"],
        "strategy_type": identity["strategy_type"],
        "comparison_mode": assessment["bindings"]["comparison_mode"],
        "population_count": population["population_count"],
        "labeled_count": population["labelled_count"],
        "monthly_status": assessment["monthly"]["status"],
    }
    for field, expected_value in expected.items():
        if normalized[field] != expected_value:
            raise StrategyError(f"measure_pool_impact output {field} drifted")

    excluded = _non_negative_int(
        normalized["nan_labels_excluded"], "nan_labels_excluded"
    )
    if excluded != population["unlabelled_count"]:
        raise StrategyError(
            "measure_pool_impact output nan_labels_excluded drifted"
        )
    expected_warnings = [
        str(flag["message"])
        for flag in assessment["red_flags"]
        if flag.get("level") in {"amber", "red"}
    ]
    if normalized["warnings"] != expected_warnings:
        raise StrategyError("measure_pool_impact output warnings drifted")
    for field in ("not_created_strategy", "not_adopted", "not_deployed"):
        if normalized[field] is not True:
            raise StrategyError(f"measure_pool_impact output {field} must be true")

    artifacts = normalized["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise StrategyError("measure_pool_impact output needs one canonical artifact")
    artifact = artifacts[0]
    if not isinstance(artifact, dict) or set(artifact) != _OUTPUT_ARTIFACT_FIELDS:
        raise StrategyError("measure_pool_impact output artifact is invalid")
    artifact_id = _text(artifact["artifact_id"], "artifact_id")
    expected_artifact_hash = hashlib.sha256(
        canonical_strategy_pool_impact_json(assessment).encode("utf-8")
    ).hexdigest()
    expected_download_url = (
        f"/api/tasks/{quote(identity['task_id'], safe='')}"
        f"/task-artifacts/{quote(artifact_id, safe='')}/download"
    )
    artifact_expected = {
        "kind": POOL_IMPACT_ARTIFACT_KIND,
        "format": "json",
        "filename": f"{assessment['assessment_id']}.json",
        "content_hash": expected_artifact_hash,
        "download_url": expected_download_url,
    }
    if artifact["artifact_id"] != artifact_id:
        raise StrategyError("measure_pool_impact artifact_id is not canonical")
    for field, expected_value in artifact_expected.items():
        if artifact[field] != expected_value:
            raise StrategyError(f"measure_pool_impact artifact {field} drifted")
    normalized["assessment"] = assessment
    return normalized


def _validate_inputs(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StrategyError("measure_pool_impact inputs must be an object")
    unexpected = sorted(set(value) - _INPUT_FIELDS)
    if unexpected:
        raise StrategyError(
            "unsupported measure_pool_impact inputs: " + ", ".join(unexpected)
        )
    missing = sorted(_REQUIRED_FIELDS - set(value))
    if missing:
        raise StrategyError("missing measure_pool_impact inputs: " + ", ".join(missing))
    request: dict[str, Any] = {
        "strategy_type": _text(value["strategy_type"], "strategy_type"),
        "expected_pool_revision": _positive_int(
            value["expected_pool_revision"], "expected_pool_revision"
        ),
        "expected_pool_snapshot_hash": _hash(
            value["expected_pool_snapshot_hash"], "expected_pool_snapshot_hash"
        ),
        "dataset_id": _text(value["dataset_id"], "dataset_id"),
        "expected_dataset_content_hash": _hash(
            value["expected_dataset_content_hash"],
            "expected_dataset_content_hash",
        ),
        "workspace_revision": _non_negative_int(
            value["workspace_revision"], "workspace_revision"
        ),
        "workspace_generation": _non_negative_int(
            value["workspace_generation"], "workspace_generation"
        ),
        "semantic_mapping_hash": _hash(
            value["semantic_mapping_hash"], "semantic_mapping_hash"
        ),
        "target_col": _text(value["target_col"], "target_col"),
        "sample_design_ref": StrategySampleDesignRef.from_value(
            value["sample_design_ref"]
        ).to_ref_dict(),
        "comparison_mode": _text(value["comparison_mode"], "comparison_mode"),
    }
    if request["strategy_type"] not in {"approval", "reject"}:
        raise StrategyError("Pool impact supports approval/reject only")
    if request["comparison_mode"] not in {"absolute", "vs_baseline"}:
        raise StrategyError("comparison_mode must be absolute or vs_baseline")
    for field in (
        "baseline_strategy_id",
        "month_col",
        "loan_amount_col",
        "overdue_amount_col",
    ):
        raw = value.get(field)
        if raw not in (None, ""):
            request[field] = _text(raw, field)
    drop_nan_labels = value.get("drop_nan_labels", False)
    if not isinstance(drop_nan_labels, bool):
        raise StrategyError("drop_nan_labels must be boolean")
    request["drop_nan_labels"] = drop_nan_labels
    baseline_id = request.get("baseline_strategy_id")
    if request["comparison_mode"] == "vs_baseline" and baseline_id is None:
        raise StrategyError("vs_baseline requires baseline_strategy_id")
    if request["comparison_mode"] == "absolute" and baseline_id is not None:
        raise StrategyError("absolute comparison forbids baseline_strategy_id")
    optional_columns = [
        request[field]
        for field in ("month_col", "loan_amount_col", "overdue_amount_col")
        if field in request
    ]
    if request["target_col"] in optional_columns or len(optional_columns) != len(
        set(optional_columns)
    ):
        raise StrategyError("Pool impact column bindings must be distinct")
    return request


def _require_pool_cas(pool: Mapping[str, Any], request: Mapping[str, Any]) -> None:
    if pool["strategy_type"] != request["strategy_type"]:
        raise StrategyError("Strategy Pool type changed")
    if pool["revision"] != request["expected_pool_revision"] or not hmac.compare_digest(
        pool["snapshot_hash"], request["expected_pool_snapshot_hash"]
    ):
        raise StrategyError("stale strategy candidate pool revision or snapshot hash")


def _pool_sample_binding(
    pool: Mapping[str, Any], *, task_id: str
) -> dict[str, Any]:
    identities = [entry["source"]["evidence_identity"] for entry in pool["entries"]]
    if not identities or any(identity != identities[0] for identity in identities[1:]):
        raise StrategyError("Strategy Pool entries do not share one sample identity")
    return {"task_id": task_id, **dict(identities[0])}


def _require_pool_measurement_target(
    lineages,
    *,
    expected_target_col: str,
) -> None:
    if not lineages:
        raise StrategyError("Strategy Pool has no candidate lineages")
    targets = [_lineage_target_col(lineage) for lineage in lineages]
    if any(target != targets[0] for target in targets[1:]):
        raise StrategyError(
            "Strategy Pool candidates do not share one measurement target"
        )
    if targets[0] != expected_target_col:
        raise StrategyError(
            "Strategy Pool candidate target does not match the confirmed workspace target"
        )


def _require_pool_sample_design_ref(
    lineages,
    *,
    expected: StrategySampleDesignRef,
) -> None:
    if not lineages:
        raise StrategyError("Strategy Pool has no candidate lineages")
    for lineage in lineages:
        actual = _lineage_sample_design_ref(lineage)
        if actual != expected:
            raise StrategyError(
                "Strategy Pool candidate sample-design reference does not match "
                "the requested development sample"
            )


def _lineage_sample_design_ref(lineage) -> StrategySampleDesignRef:
    candidate = getattr(lineage, "candidate", None)
    if candidate is not None:
        asset = getattr(candidate, "asset", None)
        if (
            not isinstance(asset, Mapping)
            or asset.get("schema_version")
            != VOTING_CANDIDATE_ASSET_SCHEMA_VERSION
        ):
            raise StrategyError(
                "legacy Voting candidate is not bound to a governed sample "
                "design; regenerate it before impact measurement"
            )
        actual = StrategySampleDesignRef.from_value(
            asset.get("sample_design_ref")
        )
        provenance = getattr(candidate, "provenance", None)
        if not isinstance(provenance, Mapping):
            raise StrategyError("Voting candidate sample-design provenance is invalid")
        if StrategySampleDesignRef.from_value(
            provenance.get("sample_design_ref")
        ) != actual:
            raise StrategyError(
                "Voting candidate sample-design asset and provenance disagree"
            )
        parents = getattr(lineage, "parent_lineages", ())
        if not parents:
            raise StrategyError("Voting candidate parent lineage is incomplete")
        for parent in parents:
            if _lineage_sample_design_ref(parent) != actual:
                raise StrategyError(
                    "Voting candidate sample-design reference does not match "
                    "all selected parent lineages"
                )
        return actual

    evidence = getattr(lineage, "evidence", None)
    if isinstance(evidence, Mapping):
        try:
            value = evidence["generation"]["parameters"]["sample_design_ref"]
        except (KeyError, TypeError) as exc:
            raise StrategyError(
                "Strategy Pool candidate is not bound to a governed sample design; "
                "regenerate the candidate from StrategySampleDesign development"
            ) from exc
        return StrategySampleDesignRef.from_value(value)

    tree = getattr(lineage, "tree", None)
    asset = getattr(tree, "asset", None)
    if isinstance(asset, Mapping):
        try:
            value = sample_design_ref_from_automatic_tree_source_refs(
                asset["source_refs"]
            )
        except (KeyError, TypeError, StrategyError) as exc:
            raise StrategyError(
                "automatic-tree Strategy Pool candidate is not bound to exactly "
                "one governed sample design; regenerate it from "
                "StrategySampleDesign development"
            ) from exc
        return StrategySampleDesignRef.from_value(value)

    raise StrategyError(
        "Strategy Pool candidate type is not yet bound to a governed sample "
        "design; regenerate it with a sample-design-aware candidate Tool"
    )


def _resolve_sample_design_optional_bindings(
    request: Mapping[str, Any],
    binding: StrategySampleDesignExecutionBinding,
) -> dict[str, Any]:
    """Resolve optional measurement columns through the sample-design authority.

    Missing fields inherit the designed columns.  A caller-provided non-empty
    value remains a fail-closed equality assertion against that design.
    """

    expected = {
        "month_col": binding.month_col,
        "loan_amount_col": binding.loan_amount_col,
        "overdue_amount_col": binding.overdue_amount_col,
    }
    resolved = dict(request)
    for field, designed in expected.items():
        requested = request.get(field)
        if requested is not None and requested != designed:
            raise StrategyError(
                f"strategy sample-design {field} does not match Pool impact binding"
            )
        resolved[field] = designed
    return resolved


def _lineage_target_col(lineage) -> str:
    candidate = getattr(lineage, "candidate", None)
    if candidate is not None:
        try:
            target = _text(
                candidate.asset["measurement_context"]["target_col"],
                "Voting candidate target_col",
            )
        except (AttributeError, KeyError, TypeError) as exc:
            raise StrategyError("Voting candidate target binding is invalid") from exc
        parent_targets = [
            _lineage_target_col(parent)
            for parent in getattr(lineage, "parent_lineages", ())
        ]
        if not parent_targets or any(item != target for item in parent_targets):
            raise StrategyError(
                "Voting candidate target does not match its parent Pool targets"
            )
        return target
    evidence = getattr(lineage, "evidence", None)
    if isinstance(evidence, Mapping):
        try:
            target = _text(evidence["analysis"]["target"], "candidate target_col")
            generated_target = _text(
                evidence["generation"]["parameters"]["target_col"],
                "candidate generation target_col",
            )
        except (KeyError, TypeError) as exc:
            raise StrategyError("candidate target binding is invalid") from exc
        if target != generated_target:
            raise StrategyError("candidate target binding is inconsistent")
        return target
    tree = getattr(lineage, "tree", None)
    if tree is not None:
        try:
            return _text(
                tree.asset["tree_result"]["training"]["target_col"],
                "automatic-tree target_col",
            )
        except (AttributeError, KeyError, TypeError) as exc:
            raise StrategyError(
                "automatic-tree target binding is invalid"
            ) from exc
    raise StrategyError("unsupported Strategy Pool candidate target binding")


def _load_dataset_binding(
    runtime,
    *,
    request: Mapping[str, Any],
    task_id: str,
    sample: Mapping[str, Any],
) -> _DatasetBinding:
    comparisons = {
        "dataset_id": request["dataset_id"],
        "dataset_content_hash": request["expected_dataset_content_hash"],
        "workspace_revision": request["workspace_revision"],
        "workspace_generation": request["workspace_generation"],
        "semantic_mapping_hash": request["semantic_mapping_hash"],
    }
    for field, actual in comparisons.items():
        if sample[field] != actual:
            raise StrategyError(f"Pool sample {field} does not match the request")
    workspace = DataWorkspaceRepository(runtime.settings.db_path).get_or_default(task_id)
    semantic_hash = data_semantic_mapping_hash(workspace.semantic_mapping)
    if (
        workspace.active_dataset_id != request["dataset_id"]
        or workspace.active_dataset_content_hash
        != request["expected_dataset_content_hash"]
        or workspace.revision != request["workspace_revision"]
        or workspace.analysis_generation != request["workspace_generation"]
        or not hmac.compare_digest(semantic_hash, request["semantic_mapping_hash"])
        or workspace.semantic_mapping.target_col != request["target_col"]
    ):
        raise StrategyError("DataWorkspace binding changed before Pool impact")
    try:
        dataset = runtime.registry.get(request["dataset_id"])
        path = Path(runtime.registry.resolve_verified_path(request["dataset_id"]))
    except (DatasetContentDriftError, KeyError, OSError, TypeError, ValueError) as exc:
        raise StrategyError("Pool impact source dataset is unavailable or drifted") from exc
    if dataset.task_id != task_id:
        raise StrategyError("Pool impact source dataset belongs to another task")
    content_hash = str(dataset.content_hash or "")
    if not hmac.compare_digest(content_hash, request["expected_dataset_content_hash"]):
        raise StrategyError("Pool impact dataset content hash changed")
    if sha256_file(path) != content_hash:
        raise StrategyError("Pool impact dataset bytes changed")
    columns = tuple(str(column.name) for column in dataset.columns)
    requested_columns = {
        request["target_col"],
        *(request[field] for field in ("month_col", "loan_amount_col", "overdue_amount_col") if field in request),
    }
    missing = sorted(requested_columns - set(columns))
    if missing:
        raise StrategyError("Pool impact dataset is missing columns: " + ", ".join(missing))

    from marvis.packs.strategy.candidate_asset_tools import (
        _registry_metadata_hash_on_connection,
    )

    with runtime.task_artifacts.transaction() as conn:
        registry_hash = _registry_metadata_hash_on_connection(
            conn,
            task_id=task_id,
            dataset_id=dataset.id,
            expected_content_hash=content_hash,
        )
    return _DatasetBinding(
        dataset=dataset,
        path=path,
        source_path=str(dataset.source_path),
        dataset_id=dataset.id,
        content_hash=content_hash,
        registry_metadata_hash=registry_hash,
        row_count=int(dataset.row_count),
        columns=columns,
        workspace_revision=workspace.revision,
        workspace_generation=workspace.analysis_generation,
        semantic_mapping_hash=semantic_hash,
        target_col=request["target_col"],
    )


def _load_baseline(
    runtime,
    *,
    request: Mapping[str, Any],
    task_id: str,
) -> _BaselineBinding | None:
    if request["comparison_mode"] == "absolute":
        return None
    strategy_id = request["baseline_strategy_id"]
    strategy = runtime.strategies.get_strategy(strategy_id)
    meta = runtime.strategies.get_strategy_meta(strategy_id)
    spec_hash = runtime.strategies.get_strategy_spec_hash(strategy_id)
    if (
        strategy is None
        or meta is None
        or strategy.spec is None
        or not isinstance(spec_hash, str)
        or meta.get("task_id") != task_id
    ):
        raise StrategyError("baseline strategy is not owned by the current task")
    if (
        strategy.strategy_type != request["strategy_type"]
        or meta.get("strategy_type") != request["strategy_type"]
    ):
        raise StrategyError("baseline strategy type must match the Pool")
    calculated = strategy_spec_hash(strategy.spec)
    if not hmac.compare_digest(spec_hash, calculated):
        raise StrategyError("baseline strategy spec hash is inconsistent")
    return _BaselineBinding(
        strategy_id=strategy.id,
        strategy_type=strategy.strategy_type,
        spec=strategy.spec.to_dict(),
        spec_hash=spec_hash,
    )


def _read_frame(
    runtime,
    *,
    dataset: _DatasetBinding,
    sample_design: StrategySampleDesignExecutionBinding,
    strategy_spec: Mapping[str, Any],
    baseline_spec: Mapping[str, Any] | None,
    request: Mapping[str, Any],
):
    fields = _expression_fields(strategy_spec)
    if baseline_spec is not None:
        fields.update(_expression_fields(baseline_spec))
    fields.add(dataset.target_col)
    if sample_design.split_column is not None:
        fields.add(sample_design.split_column)
    fields.update(
        request[field]
        for field in ("month_col", "loan_amount_col", "overdue_amount_col")
        if request.get(field) is not None
    )
    unknown = sorted(fields - set(dataset.columns))
    if unknown:
        raise StrategyError("Pool rules reference missing columns: " + ", ".join(unknown))
    frame = runtime.backend.read_frame(dataset.path, columns=sorted(fields))
    if len(frame) != dataset.row_count:
        raise StrategyError("Pool impact dataset row count changed")
    if sha256_file(dataset.path) != dataset.content_hash:
        raise StrategyError("Pool impact dataset bytes changed before evaluation")
    return frame


def _expression_fields(value: Any) -> set[str]:
    fields: set[str] = set()
    if isinstance(value, Mapping):
        field = value.get("field")
        if isinstance(field, str):
            fields.add(field)
        for item in value.values():
            fields.update(_expression_fields(item))
    elif isinstance(value, list | tuple):
        for item in value:
            fields.update(_expression_fields(item))
    return fields


def _persist_assessment(
    runtime,
    *,
    repository: StrategyCandidatePoolRepository,
    request: Mapping[str, Any],
    task_id: str,
    pool: Mapping[str, Any],
    pool_artifact,
    lineages,
    dataset: _DatasetBinding,
    sample_design: StrategySampleDesignExecutionBinding,
    baseline: _BaselineBinding | None,
    assessment: Mapping[str, Any],
    nan_labels_excluded: int,
) -> dict[str, Any]:
    from marvis.packs.strategy import pool_tools

    canonical = canonical_strategy_pool_impact_json(assessment).encode("utf-8")
    artifact_content_hash = hashlib.sha256(canonical).hexdigest()
    identity = assessment["identity"]
    assessment_id = assessment["assessment_id"]
    out_dir = _prepare_output_directory(
        runtime.settings.tasks_dir,
        task_id=task_id,
    )
    final_path = out_dir / f"{assessment_id}.json"
    provenance = {
        "schema_version": POOL_IMPACT_ARTIFACT_SCHEMA_VERSION,
        "producer_version": STRATEGY_POOL_IMPACT_PRODUCER_VERSION,
        "task_id": task_id,
        "assessment_id": assessment_id,
        "assessment_content_hash": assessment["content_hash"],
        "pool_id": identity["pool_id"],
        "pool_revision": identity["revision"],
        "pool_revision_id": identity["revision_id"],
        "pool_snapshot_hash": identity["snapshot_hash"],
        "design_hash": identity["design_hash"],
        "strategy_spec_hash": identity["strategy_spec_hash"],
        "dataset_id": dataset.dataset_id,
        "dataset_content_hash": dataset.content_hash,
        "registry_metadata_hash": dataset.registry_metadata_hash,
        "workspace_revision": dataset.workspace_revision,
        "workspace_generation": dataset.workspace_generation,
        "semantic_mapping_hash": dataset.semantic_mapping_hash,
        "target_col": dataset.target_col,
        "sample_design_ref": sample_design.to_ref_dict(),
        "month_col": sample_design.month_col,
        "loan_amount_col": sample_design.loan_amount_col,
        "overdue_amount_col": sample_design.overdue_amount_col,
        "source_target_bad_value": sample_design.target_bad_value,
        "normalized_target_bad_value": 1,
        "sample_partition": sample_design.reference.partition,
        "comparison_mode": request["comparison_mode"],
        "baseline_strategy_id": None if baseline is None else baseline.strategy_id,
        "baseline_spec_hash": None if baseline is None else baseline.spec_hash,
    }
    uow = ArtifactUnitOfWork()
    staged = uow.stage_file(out_dir, final_path.name)
    try:
        staged.path.write_bytes(canonical)
    except OSError as exc:
        uow.rollback()
        raise StrategyError("Pool impact artifact could not be staged") from exc
    db_committed = False
    rollback_under_lock = False
    reused = False
    record: Mapping[str, Any]
    try:
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                locked = repository.get_current_on_connection(
                    conn, task_id, request["strategy_type"]
                )
                if locked is None:
                    raise StrategyError("strategy candidate pool not found")
                locked = validate_strategy_pool(locked)
                _require_pool_cas(locked, request)
                if locked != pool:
                    raise StrategyError("Strategy Pool changed before impact registration")
                pool_tools._require_parent_pool_artifact_on_connection(
                    conn,
                    pool_artifact,
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
                _require_sample_design_on_connection(
                    conn,
                    task_id=task_id,
                    binding=sample_design,
                )
                _require_dataset_and_workspace_on_connection(
                    conn,
                    request=request,
                    task_id=task_id,
                    dataset=dataset,
                )
                _require_baseline_on_connection(
                    conn,
                    request=request,
                    task_id=task_id,
                    baseline=baseline,
                )
                row = conn.execute(
                    """
                    SELECT id, task_id, kind, path, content_hash, origin_tool,
                           provenance_json, created_at
                      FROM task_artifacts
                     WHERE task_id = ? AND kind = ? AND path = ?
                    """,
                    (task_id, POOL_IMPACT_ARTIFACT_KIND, str(final_path)),
                ).fetchone()
                if row is not None:
                    _require_existing_artifact(
                        row,
                        task_id=task_id,
                        final_path=final_path,
                        canonical=canonical,
                        content_hash=artifact_content_hash,
                        provenance=provenance,
                    )
                    uow.rollback()
                    reused = True
                else:
                    if final_path.exists() or final_path.is_symlink():
                        raise StrategyError(
                            "Pool impact artifact path exists without a registry row"
                        )
                    uow.promote_all()
                    _verify_file(
                        final_path,
                        root=Path(runtime.settings.tasks_dir),
                        canonical=canonical,
                        content_hash=artifact_content_hash,
                    )
                record = runtime.task_artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=POOL_IMPACT_ARTIFACT_KIND,
                    path=str(final_path),
                    content_hash=artifact_content_hash,
                    origin_tool=POOL_IMPACT_ORIGIN_TOOL,
                    provenance=provenance,
                )
                conn.commit()
                db_committed = True
            except Exception:
                rollback_under_lock = True
                uow.rollback()
                raise
        if not reused:
            uow.commit()
    except Exception:
        if not db_committed and not rollback_under_lock:
            uow.rollback()
        raise
    return _tool_output(
        assessment,
        record=record,
        task_id=task_id,
        nan_labels_excluded=nan_labels_excluded,
    )


def _require_dataset_and_workspace_on_connection(
    conn,
    *,
    request: Mapping[str, Any],
    task_id: str,
    dataset: _DatasetBinding,
) -> None:
    from marvis.packs.strategy.candidate_asset_tools import (
        _registry_metadata_hash_on_connection,
        _require_file_content_hash,
    )

    registry_hash = _registry_metadata_hash_on_connection(
        conn,
        task_id=task_id,
        dataset_id=dataset.dataset_id,
        expected_content_hash=dataset.content_hash,
    )
    if not hmac.compare_digest(registry_hash, dataset.registry_metadata_hash):
        raise StrategyError("dataset registry metadata changed before registration")
    row = conn.execute(
        "SELECT source_path FROM datasets WHERE task_id = ? AND id = ?",
        (task_id, dataset.dataset_id),
    ).fetchone()
    if row is None or str(row["source_path"]) != dataset.source_path:
        raise StrategyError("dataset registry path changed before registration")
    _require_file_content_hash(
        dataset.path,
        dataset.content_hash,
        "Pool impact dataset bytes changed before registration",
    )
    row = conn.execute(
        """
        SELECT revision, active_dataset_id, active_dataset_content_hash,
               analysis_generation, semantic_mapping_json
          FROM data_workspaces WHERE task_id = ?
        """,
        (task_id,),
    ).fetchone()
    if row is None:
        raise StrategyError("DataWorkspace disappeared before registration")
    try:
        raw_mapping = str(row["semantic_mapping_json"])
        mapping = data_semantic_mapping_from_dict(json.loads(raw_mapping))
        canonical_mapping = json.dumps(
            {
                "target_col": mapping.target_col,
                "field_roles": dict(mapping.field_roles),
                "business_names": dict(mapping.business_names),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StrategyError("DataWorkspace semantic mapping is invalid") from exc
    if raw_mapping != canonical_mapping:
        raise StrategyError("DataWorkspace semantic mapping is not canonical")
    if (
        int(row["revision"]) != request["workspace_revision"]
        or str(row["active_dataset_id"]) != dataset.dataset_id
        or str(row["active_dataset_content_hash"]) != dataset.content_hash
        or int(row["analysis_generation"]) != request["workspace_generation"]
        or not hmac.compare_digest(
            data_semantic_mapping_hash(mapping), request["semantic_mapping_hash"]
        )
        or mapping.target_col != request["target_col"]
    ):
        raise StrategyError("DataWorkspace changed before impact registration")


def _require_sample_design_on_connection(
    conn,
    *,
    task_id: str,
    binding: StrategySampleDesignExecutionBinding,
) -> None:
    row = conn.execute(
        """
        SELECT task_id, kind, path, content_hash, origin_tool, provenance_json
          FROM task_artifacts
         WHERE id = ?
        """,
        (binding.reference.artifact_id,),
    ).fetchone()
    expected_provenance = json.dumps(
        binding.artifact.provenance,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if (
        row is None
        or str(row["task_id"]) != task_id
        or str(row["kind"]) != SAMPLE_DESIGN_ARTIFACT_KIND
        or str(row["path"]) != str(binding.artifact.path)
        or not hmac.compare_digest(
            str(row["content_hash"]), binding.reference.artifact_content_hash
        )
        or str(row["origin_tool"]) != SAMPLE_DESIGN_ORIGIN_TOOL
        or str(row["provenance_json"]) != expected_provenance
    ):
        raise StrategyError(
            "strategy sample-design artifact changed before impact registration"
        )
    if sha256_file(binding.artifact.path) != binding.reference.artifact_content_hash:
        raise StrategyError(
            "strategy sample-design artifact bytes changed before impact registration"
        )


def _require_baseline_on_connection(
    conn,
    *,
    request: Mapping[str, Any],
    task_id: str,
    baseline: _BaselineBinding | None,
) -> None:
    if baseline is None:
        if request["comparison_mode"] != "absolute":
            raise StrategyError("baseline disappeared before registration")
        return
    row = conn.execute(
        """
        SELECT id, task_id, strategy_type, rules_json, score_col,
               default_decision_json, description, created_at,
               dsl_json, dsl_schema_version, dsl_content_hash
          FROM strategies WHERE id = ?
        """,
        (baseline.strategy_id,),
    ).fetchone()
    if row is None or str(row["task_id"]) != task_id:
        raise StrategyError("baseline strategy changed before registration")
    strategy = _strategy_from_row(row)
    spec_hash = _strategy_spec_hash_from_row(row)
    if (
        strategy.strategy_type != request["strategy_type"]
        or strategy.spec is None
        or strategy.spec.to_dict() != baseline.spec
        or not hmac.compare_digest(spec_hash, baseline.spec_hash)
    ):
        raise StrategyError("baseline strategy changed before registration")


def _prepare_output_directory(tasks_dir: Path | str, *, task_id: str) -> Path:
    root = Path(tasks_dir).absolute()
    if root.is_symlink():
        raise StrategyError("task artifact root must not be a symlink")
    task_dir = root / task_id
    if task_dir.exists() and task_dir.is_symlink():
        raise StrategyError("task artifact directory must not be a symlink")
    task_dir.mkdir(parents=True, exist_ok=True)
    try:
        if task_dir.resolve(strict=True).parent != root.resolve(strict=True):
            raise StrategyError("Pool impact artifact directory escaped task storage")
    except OSError as exc:
        raise StrategyError("Pool impact artifact directory is unavailable") from exc
    out_dir = task_dir / "strategy_pool_impacts"
    if out_dir.exists() and (out_dir.is_symlink() or not out_dir.is_dir()):
        raise StrategyError("Pool impact artifact path must be a regular directory")
    out_dir.mkdir(exist_ok=True)
    if out_dir.is_symlink() or out_dir.resolve(strict=True).parent != task_dir.resolve(
        strict=True
    ):
        raise StrategyError("Pool impact artifact directory escaped task storage")
    return out_dir


def _require_existing_artifact(
    row,
    *,
    task_id: str,
    final_path: Path,
    canonical: bytes,
    content_hash: str,
    provenance: Mapping[str, Any],
) -> None:
    record = {field: row[field] for field in _TASK_ARTIFACT_ROW_FIELDS}
    expected = {
        "task_id": task_id,
        "kind": POOL_IMPACT_ARTIFACT_KIND,
        "path": str(final_path),
        "content_hash": content_hash,
        "origin_tool": POOL_IMPACT_ORIGIN_TOOL,
    }
    if any(str(record[field]) != value for field, value in expected.items()):
        raise StrategyError("existing Pool impact artifact registry row changed")
    expected_provenance = json.dumps(
        dict(provenance),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if str(record["provenance_json"]) != expected_provenance:
        raise StrategyError("existing Pool impact artifact provenance changed")
    _verify_file(
        final_path,
        root=final_path.parents[2],
        canonical=canonical,
        content_hash=content_hash,
    )


def _verify_file(
    path: Path,
    *,
    root: Path,
    canonical: bytes,
    content_hash: str,
) -> None:
    if path.is_symlink() or not path.is_file():
        raise StrategyError("Pool impact artifact must be a regular file")
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise StrategyError("Pool impact artifact escaped task storage") from exc
    raw = path.read_bytes()
    if raw != canonical or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(), content_hash
    ):
        raise StrategyError("Pool impact artifact bytes changed")


def _tool_output(
    assessment: Mapping[str, Any],
    *,
    record: Mapping[str, Any],
    task_id: str,
    nan_labels_excluded: int,
) -> dict[str, Any]:
    identity = assessment["identity"]
    path = Path(str(record["path"]))
    artifact_id = str(record["id"])
    warnings = [
        str(flag["message"])
        for flag in assessment["red_flags"]
        if flag.get("level") in {"amber", "red"}
    ]
    return {
        "schema_version": POOL_IMPACT_TOOL_SCHEMA_VERSION,
        "assessment_id": assessment["assessment_id"],
        "content_hash": assessment["content_hash"],
        "pool_id": identity["pool_id"],
        "revision": identity["revision"],
        "snapshot_hash": identity["snapshot_hash"],
        "design_hash": identity["design_hash"],
        "strategy_type": identity["strategy_type"],
        "comparison_mode": assessment["bindings"]["comparison_mode"],
        "population_count": assessment["population"]["population_count"],
        "labeled_count": assessment["population"]["labelled_count"],
        "nan_labels_excluded": nan_labels_excluded,
        "monthly_status": assessment["monthly"]["status"],
        "assessment": dict(assessment),
        "warnings": warnings,
        "artifacts": [
            {
                "artifact_id": artifact_id,
                "kind": POOL_IMPACT_ARTIFACT_KIND,
                "format": "json",
                "filename": path.name,
                "content_hash": str(record["content_hash"]),
                "download_url": (
                    f"/api/tasks/{quote(task_id, safe='')}"
                    f"/task-artifacts/{quote(artifact_id, safe='')}/download"
                ),
            }
        ],
        "not_created_strategy": True,
        "not_adopted": True,
        "not_deployed": True,
    }


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise StrategyError(f"{name} must be a non-empty string")
    return value.strip()


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise StrategyError(f"{name} must be a lowercase SHA-256 hash")
    return value


def _positive_int(value: object, name: str) -> int:
    normalized = _non_negative_int(value, name)
    if normalized < 1:
        raise StrategyError(f"{name} must be at least 1")
    return normalized


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StrategyError(f"{name} must be a non-negative integer")
    return value


__all__ = [
    "POOL_IMPACT_ARTIFACT_KIND",
    "POOL_IMPACT_ARTIFACT_SCHEMA_VERSION",
    "POOL_IMPACT_ORIGIN_TOOL",
    "POOL_IMPACT_TOOL_SCHEMA_VERSION",
    "run_measure_pool_impact",
    "validate_measure_pool_impact_tool_output",
]
