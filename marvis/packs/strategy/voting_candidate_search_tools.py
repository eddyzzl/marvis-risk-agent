"""Governed Tool boundary for deterministic Voting candidate combination search.

The caller supplies only search controls.  This module recovers the exact current
Strategy Pool, development sample, dataset, target semantics, optional observation
columns, and score-vector requirements before it materializes a standalone hit
matrix.  Only aggregate search evidence is persisted; no row-level matrix, target,
weight, or amount vectors cross the Tool boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
from numbers import Real
import os
from pathlib import Path
import re
import stat
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote

import pandas as pd

from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.labels import resolve_labeled_frame
from marvis.db import ModelingRepository
from marvis.domain import STRATEGY_TYPES
from marvis.files import sha256_file
from marvis.packs.modeling.experiment import ExperimentStore
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.evaluator import evaluate_expression_frame
from marvis.packs.strategy.pool_requirement_resolver import (
    ResolvedPoolRequirements,
    hydrate_requirement_fields,
    pool_requirement_bindings_provenance,
    project_pool_entry_requirements,
    require_resolved_pool_requirements_on_connection,
    resolve_pool_requirements,
)
from marvis.packs.strategy.pool_tools import (
    StrategyCandidatePoolArtifactBinding,
    StrategyPoolDevelopmentExecutionBinding,
    bind_strategy_pool_development_execution,
    load_current_strategy_candidate_pool_artifact,
    require_strategy_pool_development_execution_binding_on_connection,
)
from marvis.packs.strategy.sample_design_binding import (
    bind_strategy_development_frame,
)
from marvis.packs.strategy.voting_candidate import (
    VOTING_CANDIDATE_ASSET_TYPE,
)
from marvis.packs.strategy.voting_candidate_search import (
    MAX_CANDIDATES,
    MAX_COMBINATIONS_BUDGET,
    MAX_EVALUATION_CELLS,
    MAX_JSON_BYTES,
    MAX_MATRIX_CELLS,
    MAX_RESULT_DISTRIBUTION_BINS,
    VOTING_CANDIDATE_SEARCH_PRODUCER_VERSION,
    VOTING_CANDIDATE_SEARCH_REQUEST_SCHEMA_VERSION,
    canonical_voting_candidate_search_result_json,
    parse_voting_candidate_search_result_json,
    search_voting_candidate_combinations,
    validate_voting_candidate_search_result,
)
from marvis.repositories.task_artifacts import (
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
)
from marvis.repositories.strategy_pool import (
    StrategyCandidatePoolConflictError,
    StrategyCandidatePoolDataError,
    StrategyCandidatePoolNotFoundError,
    StrategyCandidatePoolRepository,
)


VOTING_CANDIDATE_SEARCH_TOOL_SCHEMA_VERSION = (
    "strategy.search-voting-candidates-tool.v1"
)
VOTING_CANDIDATE_SEARCH_SELECTION_TOOL_SCHEMA_VERSION = (
    "strategy.build-voting-candidate-from-search-tool.v1"
)
VOTING_CANDIDATE_SEARCH_ARTIFACT_KIND = "strategy_voting_candidate_search_json"
VOTING_CANDIDATE_SEARCH_ARTIFACT_SCHEMA_VERSION = (
    "strategy.voting-candidate-search-artifact.v1"
)
VOTING_CANDIDATE_SEARCH_ORIGIN_TOOL = "strategy.search_voting_candidates"

_USER_CONTROL_FIELDS = frozenset(
    {
        "strategy_type",
        "member_count",
        "n",
        "objective",
        "constraints",
        "include_rule_ids",
        "exclude_rule_ids",
        "max_combinations",
    }
)
_RUN_INPUT_FIELDS = frozenset(
    {
        "strategy_type",
        "pool_ref",
        "member_count",
        "n",
        "objective",
        "constraints",
        "include_rule_ids",
        "exclude_rule_ids",
        "max_combinations",
    }
)
_SELECTION_INPUT_FIELDS = frozenset({"search_id", "combo_id", "strategy_type"})
_SELECTION_REQUIRED_INPUT_FIELDS = frozenset({"search_id", "combo_id"})
_POOL_REF_FIELDS = frozenset(
    {
        "artifact_id",
        "expected_artifact_content_hash",
        "expected_pool_id",
        "expected_revision",
        "expected_revision_id",
        "expected_snapshot_hash",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "task_id",
        "search_id",
        "search_content_hash",
        "request_hash",
        "pool_ref",
        "dataset_binding",
        "sample_design_ref",
        "sample_context_hash",
        "target_binding",
        "observation_bindings",
        "requirement_bindings",
        "excluded_unsupported_rule_ids",
        "lifecycle",
    }
)
_PROVENANCE_POOL_REF_FIELDS = frozenset(
    {
        "artifact_id",
        "artifact_content_hash",
        "pool_id",
        "strategy_type",
        "revision",
        "revision_id",
        "snapshot_hash",
    }
)
_DATASET_BINDING_FIELDS = frozenset(
    {
        "task_id",
        "dataset_id",
        "dataset_source_path",
        "dataset_content_hash",
        "dataset_registry_metadata_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
    }
)
_TARGET_BINDING_FIELDS = frozenset(
    {
        "column",
        "raw_bad_value",
        "normalized_bad_value",
        "drop_nan_labels",
        "nan_labels_dropped",
        "labeled_count",
        "sample_partition",
    }
)
_OBSERVATION_BINDING_FIELDS = frozenset({"weight_col", "amount_col"})
_LIFECYCLE = {
    "mutated_pool": False,
    "selected": False,
    "admitted": False,
    "applied": False,
    "adopted": False,
    "deployed": False,
}
_LIFECYCLE_FIELDS = frozenset(_LIFECYCLE)
_OUTPUT_FIELDS = frozenset(
    {
        "schema_version",
        "search_id",
        "request_hash",
        "content_hash",
        "pool_id",
        "pool_revision",
        "pool_snapshot_hash",
        "search_space",
        "evaluated",
        "truncated",
        "eligible",
        "excluded_unsupported_rule_ids",
        "search_result",
        "artifacts",
        "not_mutated_pool",
        "not_selected",
        "not_admitted",
        "not_applied",
        "not_adopted",
        "not_deployed",
    }
)
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SEARCH_ID_RE = re.compile(r"^voting-search-[0-9a-f]{32}$")
_COMBO_ID_RE = re.compile(r"^voting-combo-[0-9a-f]{32}$")
_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
_BOUNDARY_ERRORS = (
    StrategyCandidatePoolConflictError,
    StrategyCandidatePoolDataError,
    StrategyCandidatePoolNotFoundError,
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
)


@dataclass(frozen=True)
class VotingCandidateSearchArtifactBinding:
    """Authenticated aggregate search evidence for downstream combo selection."""

    task_id: str
    artifact_id: str
    artifact_path: Path
    artifact_content_hash: str
    artifact_provenance: dict[str, Any]
    artifact_provenance_json: str
    result: dict[str, Any]
    pool_development: StrategyPoolDevelopmentExecutionBinding
    resolved_requirements: ResolvedPoolRequirements | None
    tasks_root: Path
    db_path: Path


@dataclass(frozen=True)
class VotingCandidateSearchSelectionBinding:
    """One exact evaluated search combination mapped to its live Pool entries."""

    task_id: str
    search_id: str
    combo_id: str
    strategy_type: str
    artifact_binding: VotingCandidateSearchArtifactBinding
    pool_id: str
    pool_revision: int
    pool_snapshot_hash: str
    member_rule_ids: tuple[str, ...]
    selected_entry_ids: tuple[str, ...]
    n: int
    eligible: bool
    constraint_failures: tuple[dict[str, Any], ...]
    rank: int


def resolve_voting_candidate_search_inputs(
    runtime,
    *,
    task_id: str,
    user_controls: object,
) -> dict[str, Any]:
    """Recover platform-owned Pool identity from strict user-owned controls."""

    task = _text(task_id, "task_id")
    controls = _validate_user_controls(user_controls)
    pool = load_current_strategy_candidate_pool_artifact(
        runtime,
        task_id=task,
        strategy_type=controls["strategy_type"],
    )
    entries, _excluded = _searchable_entries(pool)
    _require_candidate_universe(
        controls,
        entries=entries,
    )
    return {
        **controls,
        "pool_ref": {
            "artifact_id": pool.artifact_id,
            "expected_artifact_content_hash": pool.artifact_content_hash,
            "expected_pool_id": pool.pool["pool_id"],
            "expected_revision": pool.pool["revision"],
            "expected_revision_id": pool.pool["revision_id"],
            "expected_snapshot_hash": pool.pool["snapshot_hash"],
        },
    }


def resolve_voting_candidate_search_selection(
    runtime,
    *,
    task_id: str,
    search_id: str,
    combo_id: str,
    strategy_type: str | None = None,
) -> VotingCandidateSearchSelectionBinding:
    """Authenticate one evaluated combo against its exact current Pool."""

    try:
        task = _text(task_id, "task_id")
        search = _search_id(search_id, "search_id")
        combo = _combo_id(combo_id, "combo_id")
        requested_type = (
            None
            if strategy_type is None
            else _strategy_type(strategy_type, "strategy_type")
        )
        pool_repository = StrategyCandidatePoolRepository(runtime.settings.db_path)
        matches: list[
            tuple[str, VotingCandidateSearchArtifactBinding]
        ] = []
        for candidate_type in (
            [requested_type] if requested_type is not None else sorted(STRATEGY_TYPES)
        ):
            if pool_repository.get_current(task, candidate_type) is None:
                continue
            pool = load_current_strategy_candidate_pool_artifact(
                runtime,
                task_id=task,
                strategy_type=candidate_type,
            )
            path = _expected_search_path(
                runtime.settings.tasks_dir,
                task_id=task,
                search_id=search,
                pool_snapshot_hash=pool.pool["snapshot_hash"],
            )
            record = runtime.task_artifacts.get_for_task_kind_path(
                task,
                VOTING_CANDIDATE_SEARCH_ARTIFACT_KIND,
                str(path),
            )
            if record is None:
                continue
            provenance = _validate_provenance(record.get("provenance"))
            artifact = load_voting_candidate_search_artifact(
                runtime,
                task_id=task,
                artifact_id=record["id"],
                expected_artifact_content_hash=record["content_hash"],
                expected_search_id=search,
                expected_search_content_hash=provenance["search_content_hash"],
            )
            matches.append((candidate_type, artifact))
        if not matches:
            raise StrategyError(
                "Voting search does not match any current Strategy Pool; "
                "run the search again"
            )
        if len(matches) != 1:
            raise StrategyError(
                "Voting search matches multiple current Strategy Pool types; "
                "strategy_type is required"
            )
        resolved_type, artifact = matches[0]
        pool = artifact.pool_development.pool
        selected = next(
            (
                item
                for item in artifact.result["combinations"]
                if hmac.compare_digest(item["combo_id"], combo)
            ),
            None,
        )
        if selected is None:
            raise StrategyError(
                "combo_id is not an authenticated evaluated Voting search "
                "combination; choose an evaluated combo or run the search again"
            )
        entries, _excluded = _searchable_entries(pool)
        entries_by_rule = {entry["rule_id"]: entry for entry in entries}
        if len(entries_by_rule) != len(entries):
            raise StrategyError("current Voting search Pool rule ids are not unique")
        member_rule_ids = tuple(selected["member_ids"])
        if any(rule_id not in entries_by_rule for rule_id in member_rule_ids):
            raise StrategyError("Voting search combination members changed from Pool")
        return VotingCandidateSearchSelectionBinding(
            task_id=task,
            search_id=search,
            combo_id=combo,
            strategy_type=resolved_type,
            artifact_binding=artifact,
            pool_id=pool.pool["pool_id"],
            pool_revision=pool.pool["revision"],
            pool_snapshot_hash=pool.pool["snapshot_hash"],
            member_rule_ids=member_rule_ids,
            selected_entry_ids=tuple(
                entries_by_rule[rule_id]["entry_id"] for rule_id in member_rule_ids
            ),
            n=selected["n"],
            eligible=selected["eligible"],
            constraint_failures=tuple(
                dict(item) for item in selected["constraint_failures"]
            ),
            rank=selected["rank"],
        )
    except StrategyError:
        raise
    except _BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc


def run_build_voting_candidate_from_search(
    inputs,
    ctx,
    runtime,
) -> dict[str, Any]:
    """Build one immutable Voting candidate from an authenticated search pointer."""

    request = _validate_selection_inputs(inputs)
    task_id = _text(ctx.task_id, "task_id")
    selection = resolve_voting_candidate_search_selection(
        runtime,
        task_id=task_id,
        search_id=request["search_id"],
        combo_id=request["combo_id"],
        strategy_type=request["strategy_type"],
    )
    # Local import prevents a module cycle: the explicit builder already owns
    # Pool/source replay and its compare-and-swap boundary.
    from marvis.packs.strategy.voting_candidate_tools import (
        _run_build_voting_candidate_with_registration_guard,
    )

    def require_search_binding_on_connection(conn) -> None:
        require_voting_candidate_search_artifact_binding_on_connection(
            conn,
            selection.artifact_binding,
        )

    output = _run_build_voting_candidate_with_registration_guard(
        {
            "strategy_type": selection.strategy_type,
            "expected_pool_revision": selection.pool_revision,
            "expected_pool_snapshot_hash": selection.pool_snapshot_hash,
            "selected_entry_ids": list(selection.selected_entry_ids),
            "n": selection.n,
        },
        ctx,
        runtime,
        registration_guard=require_search_binding_on_connection,
    )
    return {
        "schema_version": VOTING_CANDIDATE_SEARCH_SELECTION_TOOL_SCHEMA_VERSION,
        "source_search_selection": {
            "search_id": selection.search_id,
            "combo_id": selection.combo_id,
            "strategy_type": selection.strategy_type,
            "rank": selection.rank,
            "member_rule_ids": list(selection.member_rule_ids),
            "n": selection.n,
            "eligible": selection.eligible,
            "constraint_failures": [
                dict(item) for item in selection.constraint_failures
            ],
        },
        "voting_candidate": output,
        "not_mutated_pool": True,
        "not_admitted": output["not_admitted"],
        "not_applied": output["not_applied"],
        "not_adopted": output["not_adopted"],
        "not_deployed": output["not_deployed"],
    }


def run_search_voting_candidates(inputs, ctx, runtime) -> dict[str, Any]:
    """Search one exact current Pool without selecting or mutating anything."""

    try:
        request = _validate_run_inputs(inputs)
        task_id = _text(ctx.task_id, "task_id")
        pool = _load_pool(runtime, task_id=task_id, request=request)
        development = bind_strategy_pool_development_execution(runtime, pool)
        entries, excluded = _searchable_entries(pool)
        _require_search_budget(
            request,
            development=development,
            entries=entries,
        )
        requirements = project_pool_entry_requirements(entries)
        resolved = _resolve_requirements(
            runtime,
            development=development,
            requirements=requirements,
        )
        frame, dropped = _read_labeled_development_frame(
            runtime,
            development=development,
            resolved=resolved,
            entries=entries,
        )
        search_request = _materialize_search_request(
            request,
            development=development,
            entries=entries,
            frame=frame,
        )
        result = search_voting_candidate_combinations(search_request)
        result = validate_voting_candidate_search_result(result)
        return _persist_search(
            runtime,
            task_id=task_id,
            pool=pool,
            development=development,
            resolved=resolved,
            dropped=dropped,
            excluded_unsupported_rule_ids=excluded,
            result=result,
        )
    except StrategyError:
        raise
    except _BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc


def load_voting_candidate_search_artifact(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_artifact_content_hash: str,
    expected_search_id: str,
    expected_search_content_hash: str,
) -> VotingCandidateSearchArtifactBinding:
    """Load, authenticate, and reconnect aggregate search evidence to its Pool."""

    try:
        task = _text(task_id, "task_id")
        artifact = _hash(artifact_id, "artifact_id")
        artifact_hash = _hash(
            expected_artifact_content_hash,
            "expected_artifact_content_hash",
        )
        search_id = _search_id(expected_search_id, "expected_search_id")
        search_hash = _hash(
            expected_search_content_hash,
            "expected_search_content_hash",
        )
        record = runtime.task_artifacts.get_for_task(task, artifact)
        if record is None:
            raise StrategyError("Voting search artifact not found")
        provenance = _validate_provenance(record.get("provenance"))
        if (
            provenance["task_id"] != task
            or provenance["search_id"] != search_id
            or not hmac.compare_digest(
                provenance["search_content_hash"],
                search_hash,
            )
        ):
            raise StrategyError("Voting search artifact provenance changed")
        pool_ref = provenance["pool_ref"]
        path = _expected_search_path(
            runtime.settings.tasks_dir,
            task_id=task,
            search_id=search_id,
            pool_snapshot_hash=pool_ref["snapshot_hash"],
        )
        if Path(str(record["path"])) != path:
            raise StrategyError("Voting search artifact path is not canonical")
        raw = _read_exact_file(
            path,
            root=Path(runtime.settings.tasks_dir).absolute(),
            expected_content_hash=artifact_hash,
        )
        result = parse_voting_candidate_search_result_json(raw)
        canonical = canonical_voting_candidate_search_result_json(result).encode(
            "utf-8"
        )
        if raw != canonical:
            raise StrategyError("Voting search artifact bytes are not canonical")
        if result["search_id"] != search_id or not hmac.compare_digest(
            result["content_hash"], search_hash
        ):
            raise StrategyError("Voting search embedded identity changed")
        if not hmac.compare_digest(
            provenance["request_hash"],
            result["request_hash"],
        ):
            raise StrategyError("Voting search artifact provenance changed")
        pool = load_current_strategy_candidate_pool_artifact(
            runtime,
            task_id=task,
            strategy_type=pool_ref["strategy_type"],
            expected_pool_revision=pool_ref["revision"],
            expected_pool_snapshot_hash=pool_ref["snapshot_hash"],
            expected_artifact_id=pool_ref["artifact_id"],
            expected_artifact_content_hash=pool_ref["artifact_content_hash"],
        )
        development = bind_strategy_pool_development_execution(runtime, pool)
        entries, excluded = _searchable_entries(pool)
        requirements = project_pool_entry_requirements(entries)
        resolved = _resolve_requirements(
            runtime,
            development=development,
            requirements=requirements,
        )
        _require_result_pool_projection(result, pool=pool)
        expected = _artifact_provenance(
            task_id=task,
            pool=pool,
            development=development,
            resolved=resolved,
            dropped=provenance["target_binding"]["nan_labels_dropped"],
            excluded_unsupported_rule_ids=excluded,
            result=result,
        )
        if provenance != expected:
            raise StrategyError("Voting search artifact provenance changed")
        binding = VotingCandidateSearchArtifactBinding(
            task_id=task,
            artifact_id=artifact,
            artifact_path=path,
            artifact_content_hash=artifact_hash,
            artifact_provenance=provenance,
            artifact_provenance_json=_canonical_json(provenance),
            result=result,
            pool_development=development,
            resolved_requirements=resolved,
            tasks_root=Path(runtime.settings.tasks_dir).absolute(),
            db_path=Path(runtime.settings.db_path).absolute(),
        )
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            require_voting_candidate_search_artifact_binding_on_connection(
                conn,
                binding,
            )
            conn.commit()
        return binding
    except StrategyError:
        raise
    except _BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc


def require_voting_candidate_search_artifact_binding_on_connection(
    conn,
    binding: VotingCandidateSearchArtifactBinding,
) -> None:
    """Re-authenticate a search artifact and every governed source under lock."""

    if not isinstance(binding, VotingCandidateSearchArtifactBinding):
        raise StrategyError("Voting search artifact binding is invalid")
    if not conn.in_transaction:
        raise StrategyError(
            "Voting search artifact binding requires a caller-owned transaction"
        )
    database = conn.execute(
        "SELECT file FROM pragma_database_list WHERE name = 'main'"
    ).fetchone()
    if (
        database is None
        or not str(database["file"])
        or Path(str(database["file"])).absolute() != binding.db_path
    ):
        raise StrategyError("Voting search artifact database binding changed")
    result = validate_voting_candidate_search_result(binding.result)
    if result != binding.result:
        raise StrategyError("Voting search artifact payload changed")
    provenance = _validate_provenance(binding.artifact_provenance)
    if (
        provenance != binding.artifact_provenance
        or binding.artifact_provenance_json != _canonical_json(provenance)
        or provenance["task_id"] != binding.task_id
        or provenance["search_id"] != result["search_id"]
        or not hmac.compare_digest(
            provenance["search_content_hash"],
            result["content_hash"],
        )
    ):
        raise StrategyError("Voting search artifact provenance binding changed")
    require_strategy_pool_development_execution_binding_on_connection(
        conn,
        binding.pool_development,
    )
    if binding.resolved_requirements is not None:
        require_resolved_pool_requirements_on_connection(
            conn,
            binding.resolved_requirements,
        )
    _require_dataset_bytes(binding.pool_development)
    raw = _read_exact_file(
        binding.artifact_path,
        root=binding.tasks_root,
        expected_content_hash=binding.artifact_content_hash,
    )
    canonical = canonical_voting_candidate_search_result_json(result).encode("utf-8")
    if raw != canonical:
        raise StrategyError("Voting search artifact bytes changed")
    _require_artifact_row(
        conn,
        binding=binding,
    )


def _validate_user_controls(value: object) -> dict[str, Any]:
    obj = _json_object(value, "Voting search user controls")
    _exact_fields(obj, _USER_CONTROL_FIELDS, "Voting search user controls")
    return _normalize_controls(obj)


def _validate_run_inputs(value: object) -> dict[str, Any]:
    obj = _json_object(value, "search_voting_candidates inputs")
    _exact_fields(obj, _RUN_INPUT_FIELDS, "search_voting_candidates inputs")
    return {
        **_normalize_controls(obj),
        "pool_ref": _normalize_pool_ref(obj["pool_ref"]),
    }


def _validate_selection_inputs(value: object) -> dict[str, Any]:
    obj = _json_object(value, "build_voting_candidate_from_search inputs")
    missing = sorted(_SELECTION_REQUIRED_INPUT_FIELDS - set(obj))
    unsupported = sorted(set(obj) - _SELECTION_INPUT_FIELDS)
    if missing or unsupported:
        detail: list[str] = []
        if missing:
            detail.append("missing: " + ", ".join(missing))
        if unsupported:
            detail.append("unsupported: " + ", ".join(unsupported))
        raise StrategyError(
            "invalid build_voting_candidate_from_search inputs ("
            + "; ".join(detail)
            + ")"
        )
    return {
        "search_id": _search_id(obj["search_id"], "search_id"),
        "combo_id": _combo_id(obj["combo_id"], "combo_id"),
        "strategy_type": (
            None
            if "strategy_type" not in obj
            else _strategy_type(obj["strategy_type"], "strategy_type")
        ),
    }


def _normalize_controls(value: Mapping[str, Any]) -> dict[str, Any]:
    strategy_type = _text(value["strategy_type"], "strategy_type")
    if strategy_type not in STRATEGY_TYPES:
        raise StrategyError("Voting search strategy_type is invalid")
    objective = _json_object(value["objective"], "objective")
    _exact_fields(objective, frozenset({"metric", "direction"}), "objective")
    constraints_raw = _array(value["constraints"], "constraints")
    constraints: list[dict[str, Any]] = []
    for index, item in enumerate(constraints_raw):
        constraint = _json_object(item, f"constraints[{index}]")
        _exact_fields(
            constraint,
            frozenset({"metric", "operator", "value"}),
            f"constraints[{index}]",
        )
        constraints.append(
            {
                "metric": _text(
                    constraint["metric"],
                    f"constraints[{index}].metric",
                ),
                "operator": _text(
                    constraint["operator"],
                    f"constraints[{index}].operator",
                ),
                "value": constraint["value"],
            }
        )
    normalized_objective = {
        "metric": _text(objective["metric"], "objective.metric"),
        "direction": _text(
            objective["direction"],
            "objective.direction",
        ),
    }
    _require_minimum_hit_constraint(
        objective=normalized_objective,
        constraints=constraints,
    )
    return {
        "strategy_type": strategy_type,
        "member_count": _bounded_int(
            value["member_count"],
            "member_count",
            maximum=50,
        ),
        "n": _bounded_int(value["n"], "n", maximum=50),
        "objective": normalized_objective,
        "constraints": constraints,
        "include_rule_ids": _text_array(
            value["include_rule_ids"],
            "include_rule_ids",
        ),
        "exclude_rule_ids": _text_array(
            value["exclude_rule_ids"],
            "exclude_rule_ids",
        ),
        "max_combinations": _bounded_int(
            value["max_combinations"],
            "max_combinations",
            maximum=MAX_COMBINATIONS_BUDGET,
        ),
    }


def _require_minimum_hit_constraint(
    *,
    objective: Mapping[str, str],
    constraints: Sequence[Mapping[str, Any]],
) -> None:
    if objective["direction"] != "minimize":
        return
    coverage_by_rate = {
        "bad_rate": {"hit_count", "hit_share"},
        "lift": {"hit_count", "hit_share"},
        "weighted_bad_rate": {
            "weighted_hit_total",
            "weighted_hit_share",
        },
        "bad_amount_rate": {"hit_amount", "hit_amount_share"},
    }
    coverage_metrics = coverage_by_rate.get(objective["metric"])
    if coverage_metrics is None:
        return
    for constraint in constraints:
        value = constraint["value"]
        if (
            constraint["metric"] in coverage_metrics
            and constraint["operator"] == "gte"
            and isinstance(value, Real)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0.0
        ):
            return
    raise StrategyError(
        f"minimize {objective['metric']} requires a positive minimum hit "
        "constraint so an empty hit set cannot rank first"
    )


def _normalize_pool_ref(value: object) -> dict[str, Any]:
    obj = _json_object(value, "pool_ref")
    _exact_fields(obj, _POOL_REF_FIELDS, "pool_ref")
    return {
        "artifact_id": _hash(obj["artifact_id"], "pool_ref.artifact_id"),
        "expected_artifact_content_hash": _hash(
            obj["expected_artifact_content_hash"],
            "pool_ref.expected_artifact_content_hash",
        ),
        "expected_pool_id": _text(
            obj["expected_pool_id"],
            "pool_ref.expected_pool_id",
        ),
        "expected_revision": _positive_int(
            obj["expected_revision"],
            "pool_ref.expected_revision",
        ),
        "expected_revision_id": _text(
            obj["expected_revision_id"],
            "pool_ref.expected_revision_id",
        ),
        "expected_snapshot_hash": _hash(
            obj["expected_snapshot_hash"],
            "pool_ref.expected_snapshot_hash",
        ),
    }


def _load_pool(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
) -> StrategyCandidatePoolArtifactBinding:
    ref = request["pool_ref"]
    pool = load_current_strategy_candidate_pool_artifact(
        runtime,
        task_id=task_id,
        strategy_type=request["strategy_type"],
        expected_pool_revision=ref["expected_revision"],
        expected_pool_snapshot_hash=ref["expected_snapshot_hash"],
        expected_artifact_id=ref["artifact_id"],
        expected_artifact_content_hash=ref["expected_artifact_content_hash"],
    )
    if (
        pool.pool["pool_id"] != ref["expected_pool_id"]
        or pool.pool["revision_id"] != ref["expected_revision_id"]
    ):
        raise StrategyError("Voting search Pool identity changed")
    _searchable_entries(pool)
    return pool


def _searchable_entries(
    pool: StrategyCandidatePoolArtifactBinding,
) -> tuple[list[dict[str, Any]], list[str]]:
    enabled = [
        dict(entry) for entry in pool.pool["entries"] if entry["enabled"] is True
    ]
    excluded = sorted(
        entry["rule_id"]
        for entry in enabled
        if entry["source"]["asset_type"] == VOTING_CANDIDATE_ASSET_TYPE
    )
    entries = [
        entry
        for entry in enabled
        if entry["source"]["asset_type"] != VOTING_CANDIDATE_ASSET_TYPE
    ]
    if len(entries) < 2:
        raise StrategyError(
            "Voting search requires at least two enabled non-Voting Strategy "
            "Pool entries; unsupported existing Voting rules: "
            + (", ".join(excluded) if excluded else "none")
        )
    if len(entries) > MAX_CANDIDATES:
        raise StrategyError(f"Voting search Pool exceeds {MAX_CANDIDATES} candidates")
    return entries, excluded


def _require_candidate_universe(
    controls: Mapping[str, Any],
    *,
    entries: Sequence[Mapping[str, Any]],
) -> int:
    candidate_ids = [str(entry["rule_id"]) for entry in entries]
    candidate_set = set(candidate_ids)
    if len(candidate_set) != len(candidate_ids):
        raise StrategyError("Voting search Pool rule ids are not unique")
    member_count = controls["member_count"]
    n = controls["n"]
    if (
        isinstance(member_count, bool)
        or not isinstance(member_count, int)
        or member_count < 2
        or member_count > min(50, len(candidate_ids))
    ):
        raise StrategyError(
            "Voting search member_count must be between 2 and min(50, enabled "
            "non-Voting Pool candidate count)"
        )
    if isinstance(n, bool) or not isinstance(n, int) or n < 1 or n > member_count:
        raise StrategyError("Voting search n must be between 1 and member_count")
    include = controls["include_rule_ids"]
    exclude = controls["exclude_rule_ids"]
    include_set = set(include)
    exclude_set = set(exclude)
    unknown = sorted((include_set | exclude_set) - candidate_set)
    if unknown:
        raise StrategyError(
            "Voting search include/exclude contains unknown current non-Voting "
            "Pool rule ids: " + ", ".join(unknown)
        )
    overlap = sorted(include_set & exclude_set)
    if overlap:
        raise StrategyError(
            "Voting search include/exclude must be disjoint: " + ", ".join(overlap)
        )
    if len(include_set) > member_count:
        raise StrategyError("Voting search include count cannot exceed member_count")
    remaining = len(candidate_set - exclude_set)
    if member_count > remaining:
        raise StrategyError(
            "Voting search member_count exceeds candidates remaining after exclude"
        )
    optional_count = len(candidate_set - include_set - exclude_set)
    choose = member_count - len(include_set)
    return math.comb(optional_count, choose)


def _require_search_budget(
    controls: Mapping[str, Any],
    *,
    development: StrategyPoolDevelopmentExecutionBinding,
    entries: Sequence[Mapping[str, Any]],
) -> None:
    search_space = _require_candidate_universe(controls, entries=entries)
    population = development.sample_design.development_population_count
    candidate_count = len(entries)
    matrix_cells = candidate_count * population
    if matrix_cells > MAX_MATRIX_CELLS:
        raise StrategyError(
            f"Voting search hit matrix exceeds {MAX_MATRIX_CELLS} cells"
        )
    planned = min(search_space, controls["max_combinations"])
    evaluation_cells = planned * population * controls["member_count"]
    if evaluation_cells > MAX_EVALUATION_CELLS:
        raise StrategyError(
            f"Voting search plan exceeds {MAX_EVALUATION_CELLS} evaluation cells"
        )
    distribution_bins = planned * (controls["member_count"] + 1)
    if distribution_bins > MAX_RESULT_DISTRIBUTION_BINS:
        raise StrategyError(
            f"Voting search plan exceeds {MAX_RESULT_DISTRIBUTION_BINS} "
            "distribution bins"
        )


def _resolve_requirements(
    runtime,
    *,
    development: StrategyPoolDevelopmentExecutionBinding,
    requirements: Sequence[Mapping[str, Any]],
) -> ResolvedPoolRequirements | None:
    if not requirements:
        return None
    if development.sample_design_v2 is None:
        raise StrategyError(
            "Voting search score requirements require one exact StrategySampleDesign V2"
        )
    return resolve_pool_requirements(
        _modeling_runtime(runtime),
        task_id=development.task_id,
        compiled_design={"requirements": list(requirements)},
        sample_design=development.sample_design_v2,
    )


def _read_labeled_development_frame(
    runtime,
    *,
    development: StrategyPoolDevelopmentExecutionBinding,
    resolved: ResolvedPoolRequirements | None,
    entries: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, int]:
    dataset = development.dataset
    fields: set[str] = set()
    for entry in entries:
        fields.update(_expression_fields(entry["execution"]["condition"]))
    virtual_fields = set(() if resolved is None else resolved.virtual_fields)
    fields -= virtual_fields
    sample = development.sample_design
    fields.add(sample.target_col)
    for column in (
        sample.split_column,
        sample.weight_col,
        sample.loan_amount_col,
    ):
        if column is not None:
            fields.add(column)
    unknown = sorted(fields - set(dataset.columns))
    if unknown:
        raise StrategyError(
            "Voting search source references missing columns: " + ", ".join(unknown)
        )
    _require_dataset_bytes(development)
    frame = runtime.backend.read_frame(
        dataset.path,
        columns=sorted(fields),
    )
    if len(frame) != dataset.row_count:
        raise StrategyError("Voting search dataset row count changed")
    frame = frame.reset_index(drop=True)
    if resolved is not None:
        frame = hydrate_requirement_fields(frame, resolved=resolved)
    development_frame = bind_strategy_development_frame(
        frame,
        binding=sample,
    )
    labeled, dropped = resolve_labeled_frame(
        development_frame,
        sample.target_col,
        drop_nan_labels=sample.drop_nan_labels,
        scope="Voting candidate search development sample",
    )
    labeled = labeled.reset_index(drop=True)
    if not len(labeled):
        raise StrategyError(
            "Voting candidate search development sample has no labelled rows"
        )
    _require_dataset_bytes(development)
    return labeled, dropped


def _materialize_search_request(
    request: Mapping[str, Any],
    *,
    development: StrategyPoolDevelopmentExecutionBinding,
    entries: Sequence[Mapping[str, Any]],
    frame: pd.DataFrame,
) -> dict[str, Any]:
    sample = development.sample_design
    candidate_ids = [str(entry["rule_id"]) for entry in entries]
    hit_matrix = [
        evaluate_expression_frame(
            frame,
            entry["execution"]["condition"],
        ).tolist()
        for entry in entries
    ]
    target = [
        int(value)
        for value in pd.to_numeric(
            frame[sample.target_col],
            errors="raise",
        ).tolist()
    ]
    weights = (
        None
        if sample.weight_col is None
        else pd.to_numeric(
            frame[sample.weight_col],
            errors="raise",
        ).tolist()
    )
    amounts = (
        None
        if sample.loan_amount_col is None
        else pd.to_numeric(
            frame[sample.loan_amount_col],
            errors="raise",
        ).tolist()
    )
    return {
        "schema_version": VOTING_CANDIDATE_SEARCH_REQUEST_SCHEMA_VERSION,
        "candidate_ids": candidate_ids,
        "hit_matrix": hit_matrix,
        "target": target,
        "weights": weights,
        "amounts": amounts,
        "member_count": request["member_count"],
        "n": request["n"],
        "objective": request["objective"],
        "constraints": request["constraints"],
        "include": request["include_rule_ids"],
        "exclude": request["exclude_rule_ids"],
        "max_combinations": request["max_combinations"],
    }


def _persist_search(
    runtime,
    *,
    task_id: str,
    pool: StrategyCandidatePoolArtifactBinding,
    development: StrategyPoolDevelopmentExecutionBinding,
    resolved: ResolvedPoolRequirements | None,
    dropped: int,
    excluded_unsupported_rule_ids: Sequence[str],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = validate_voting_candidate_search_result(result)
    canonical = canonical_voting_candidate_search_result_json(normalized).encode(
        "utf-8"
    )
    if len(canonical) > MAX_JSON_BYTES:
        raise StrategyError("Voting search artifact exceeds the JSON byte budget")
    artifact_hash = hashlib.sha256(canonical).hexdigest()
    out_dir = _prepare_output_directory(
        runtime.settings.tasks_dir,
        task_id=task_id,
    )
    final_path = _expected_search_path(
        runtime.settings.tasks_dir,
        task_id=task_id,
        search_id=normalized["search_id"],
        pool_snapshot_hash=pool.pool["snapshot_hash"],
    )
    provenance = _artifact_provenance(
        task_id=task_id,
        pool=pool,
        development=development,
        resolved=resolved,
        dropped=dropped,
        excluded_unsupported_rule_ids=excluded_unsupported_rule_ids,
        result=normalized,
    )
    staged_uow = ArtifactUnitOfWork()
    staged = staged_uow.stage_file(out_dir, final_path.name)
    try:
        staged.path.write_bytes(canonical)
    except OSError as exc:
        staged_uow.rollback()
        raise StrategyError("Voting search artifact could not be staged") from exc

    reused = False
    committed = False
    record: Mapping[str, Any]
    try:
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                _require_live_bindings(
                    conn,
                    development=development,
                    resolved=resolved,
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
                        VOTING_CANDIDATE_SEARCH_ARTIFACT_KIND,
                        str(final_path),
                    ),
                ).fetchone()
                if row is not None:
                    _require_existing_artifact(
                        row,
                        task_id=task_id,
                        path=final_path,
                        canonical=canonical,
                        content_hash=artifact_hash,
                        provenance=provenance,
                        root=Path(runtime.settings.tasks_dir).absolute(),
                    )
                    staged_uow.rollback()
                    reused = True
                else:
                    if final_path.exists() or final_path.is_symlink():
                        raise StrategyError(
                            "Voting search path exists without a registry row"
                        )
                    staged_uow.promote_all()
                    _require_exact_file(
                        final_path,
                        root=Path(runtime.settings.tasks_dir).absolute(),
                        canonical=canonical,
                        content_hash=artifact_hash,
                    )
                _require_live_bindings(
                    conn,
                    development=development,
                    resolved=resolved,
                )
                record = runtime.task_artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=VOTING_CANDIDATE_SEARCH_ARTIFACT_KIND,
                    path=str(final_path),
                    content_hash=artifact_hash,
                    origin_tool=VOTING_CANDIDATE_SEARCH_ORIGIN_TOOL,
                    provenance=provenance,
                )
                conn.commit()
                committed = True
            except Exception:
                staged_uow.rollback()
                raise
        if not reused:
            staged_uow.commit()
    except Exception:
        if not committed:
            staged_uow.rollback()
        raise
    return _tool_output(
        task_id=task_id,
        pool=pool,
        result=normalized,
        excluded_unsupported_rule_ids=excluded_unsupported_rule_ids,
        record=record,
        path=final_path,
    )


def _artifact_provenance(
    *,
    task_id: str,
    pool: StrategyCandidatePoolArtifactBinding,
    development: StrategyPoolDevelopmentExecutionBinding,
    resolved: ResolvedPoolRequirements | None,
    dropped: int,
    excluded_unsupported_rule_ids: Sequence[str],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    sample = development.sample_design
    dataset = development.dataset
    _require_result_pool_projection(result, pool=pool)
    _entries, actual_excluded = _searchable_entries(pool)
    normalized_excluded = sorted(
        _text_array(
            excluded_unsupported_rule_ids,
            "excluded_unsupported_rule_ids",
        )
    )
    if normalized_excluded != actual_excluded:
        raise StrategyError(
            "Voting search unsupported candidate exclusion changed from Pool"
        )
    if (
        isinstance(dropped, bool)
        or not isinstance(dropped, int)
        or dropped < 0
        or (
            int(result["population"]["row_count"]) + dropped
            != sample.development_population_count
        )
        or (dropped > 0 and not sample.drop_nan_labels)
    ):
        raise StrategyError("Voting search labelled population binding changed")
    population = result["population"]
    if bool(population["weight"]["available"]) is not (
        sample.weight_col is not None
    ) or bool(population["amount"]["available"]) is not (
        sample.loan_amount_col is not None
    ):
        raise StrategyError("Voting search observation availability changed")
    value = {
        "schema_version": VOTING_CANDIDATE_SEARCH_ARTIFACT_SCHEMA_VERSION,
        "producer_version": VOTING_CANDIDATE_SEARCH_PRODUCER_VERSION,
        "task_id": task_id,
        "search_id": result["search_id"],
        "search_content_hash": result["content_hash"],
        "request_hash": result["request_hash"],
        "pool_ref": {
            "artifact_id": pool.artifact_id,
            "artifact_content_hash": pool.artifact_content_hash,
            "pool_id": pool.pool["pool_id"],
            "strategy_type": pool.pool["strategy_type"],
            "revision": pool.pool["revision"],
            "revision_id": pool.pool["revision_id"],
            "snapshot_hash": pool.pool["snapshot_hash"],
        },
        "dataset_binding": {
            "task_id": dataset.task_id,
            "dataset_id": dataset.dataset_id,
            "dataset_source_path": dataset.source_path,
            "dataset_content_hash": dataset.content_hash,
            "dataset_registry_metadata_hash": dataset.registry_metadata_hash,
            "workspace_revision": sample.workspace_revision,
            "workspace_generation": sample.workspace_generation,
            "semantic_mapping_hash": sample.semantic_mapping_hash,
        },
        "sample_design_ref": sample.to_ref_dict(),
        "sample_context_hash": development.evidence_identity["sample_context_hash"],
        "target_binding": {
            "column": sample.target_col,
            "raw_bad_value": sample.target_bad_value,
            "normalized_bad_value": 1,
            "drop_nan_labels": sample.drop_nan_labels,
            "nan_labels_dropped": dropped,
            "labeled_count": result["population"]["row_count"],
            "sample_partition": sample.reference.partition,
        },
        "observation_bindings": {
            "weight_col": sample.weight_col,
            "amount_col": sample.loan_amount_col,
        },
        "requirement_bindings": (
            None if resolved is None else pool_requirement_bindings_provenance(resolved)
        ),
        "excluded_unsupported_rule_ids": normalized_excluded,
        "lifecycle": dict(_LIFECYCLE),
    }
    return _validate_provenance(value)


def _validate_provenance(value: object) -> dict[str, Any]:
    obj = _json_object(value, "Voting search provenance")
    _exact_fields(obj, _PROVENANCE_FIELDS, "Voting search provenance")
    if (
        obj["schema_version"] != VOTING_CANDIDATE_SEARCH_ARTIFACT_SCHEMA_VERSION
        or obj["producer_version"] != VOTING_CANDIDATE_SEARCH_PRODUCER_VERSION
    ):
        raise StrategyError("Voting search provenance version is invalid")
    result = {
        "schema_version": obj["schema_version"],
        "producer_version": obj["producer_version"],
        "task_id": _text(obj["task_id"], "provenance.task_id"),
        "search_id": _search_id(
            obj["search_id"],
            "provenance.search_id",
        ),
        "search_content_hash": _hash(
            obj["search_content_hash"],
            "provenance.search_content_hash",
        ),
        "request_hash": _hash(
            obj["request_hash"],
            "provenance.request_hash",
        ),
    }
    pool = _json_object(obj["pool_ref"], "provenance.pool_ref")
    _exact_fields(pool, _PROVENANCE_POOL_REF_FIELDS, "provenance.pool_ref")
    result["pool_ref"] = {
        "artifact_id": _hash(
            pool["artifact_id"],
            "provenance.pool_ref.artifact_id",
        ),
        "artifact_content_hash": _hash(
            pool["artifact_content_hash"],
            "provenance.pool_ref.artifact_content_hash",
        ),
        "pool_id": _text(pool["pool_id"], "provenance.pool_ref.pool_id"),
        "strategy_type": _text(
            pool["strategy_type"],
            "provenance.pool_ref.strategy_type",
        ),
        "revision": _positive_int(
            pool["revision"],
            "provenance.pool_ref.revision",
        ),
        "revision_id": _text(
            pool["revision_id"],
            "provenance.pool_ref.revision_id",
        ),
        "snapshot_hash": _hash(
            pool["snapshot_hash"],
            "provenance.pool_ref.snapshot_hash",
        ),
    }
    dataset = _json_object(
        obj["dataset_binding"],
        "provenance.dataset_binding",
    )
    _exact_fields(
        dataset,
        _DATASET_BINDING_FIELDS,
        "provenance.dataset_binding",
    )
    result["dataset_binding"] = {
        "task_id": _text(
            dataset["task_id"],
            "provenance.dataset_binding.task_id",
        ),
        "dataset_id": _text(
            dataset["dataset_id"],
            "provenance.dataset_binding.dataset_id",
        ),
        "dataset_source_path": _text(
            dataset["dataset_source_path"],
            "provenance.dataset_binding.dataset_source_path",
        ),
        "dataset_content_hash": _hash(
            dataset["dataset_content_hash"],
            "provenance.dataset_binding.dataset_content_hash",
        ),
        "dataset_registry_metadata_hash": _hash(
            dataset["dataset_registry_metadata_hash"],
            "provenance.dataset_binding.dataset_registry_metadata_hash",
        ),
        "workspace_revision": _nonnegative_int(
            dataset["workspace_revision"],
            "provenance.dataset_binding.workspace_revision",
        ),
        "workspace_generation": _nonnegative_int(
            dataset["workspace_generation"],
            "provenance.dataset_binding.workspace_generation",
        ),
        "semantic_mapping_hash": _hash(
            dataset["semantic_mapping_hash"],
            "provenance.dataset_binding.semantic_mapping_hash",
        ),
    }
    sample_ref = _json_object(
        obj["sample_design_ref"],
        "provenance.sample_design_ref",
    )
    result["sample_design_ref"] = dict(sample_ref)
    result["sample_context_hash"] = _hash(
        obj["sample_context_hash"],
        "provenance.sample_context_hash",
    )
    target = _json_object(
        obj["target_binding"],
        "provenance.target_binding",
    )
    _exact_fields(
        target,
        _TARGET_BINDING_FIELDS,
        "provenance.target_binding",
    )
    result["target_binding"] = {
        "column": _text(
            target["column"],
            "provenance.target_binding.column",
        ),
        "raw_bad_value": _binary_int(
            target["raw_bad_value"],
            "provenance.target_binding.raw_bad_value",
        ),
        "normalized_bad_value": _binary_int(
            target["normalized_bad_value"],
            "provenance.target_binding.normalized_bad_value",
        ),
        "drop_nan_labels": _strict_bool(
            target["drop_nan_labels"],
            "provenance.target_binding.drop_nan_labels",
        ),
        "nan_labels_dropped": _nonnegative_int(
            target["nan_labels_dropped"],
            "provenance.target_binding.nan_labels_dropped",
        ),
        "labeled_count": _positive_int(
            target["labeled_count"],
            "provenance.target_binding.labeled_count",
        ),
        "sample_partition": _text(
            target["sample_partition"],
            "provenance.target_binding.sample_partition",
        ),
    }
    observations = _json_object(
        obj["observation_bindings"],
        "provenance.observation_bindings",
    )
    _exact_fields(
        observations,
        _OBSERVATION_BINDING_FIELDS,
        "provenance.observation_bindings",
    )
    result["observation_bindings"] = {
        "weight_col": _optional_text(
            observations["weight_col"],
            "provenance.observation_bindings.weight_col",
        ),
        "amount_col": _optional_text(
            observations["amount_col"],
            "provenance.observation_bindings.amount_col",
        ),
    }
    requirements = obj["requirement_bindings"]
    if requirements is not None:
        # The public projector owns strict recursive validation.
        from marvis.packs.strategy.pool_requirement_resolver import (
            validate_pool_requirement_bindings_provenance,
        )

        requirements = validate_pool_requirement_bindings_provenance(requirements)
    result["requirement_bindings"] = requirements
    excluded = _text_array(
        obj["excluded_unsupported_rule_ids"],
        "provenance.excluded_unsupported_rule_ids",
    )
    if excluded != sorted(excluded):
        raise StrategyError(
            "Voting search excluded unsupported rule ids are not canonical"
        )
    result["excluded_unsupported_rule_ids"] = excluded
    lifecycle = _json_object(obj["lifecycle"], "provenance.lifecycle")
    _exact_fields(lifecycle, _LIFECYCLE_FIELDS, "provenance.lifecycle")
    if lifecycle != _LIFECYCLE:
        raise StrategyError("Voting search provenance lifecycle changed")
    result["lifecycle"] = dict(_LIFECYCLE)
    if result["dataset_binding"]["task_id"] != result["task_id"]:
        raise StrategyError("Voting search dataset and task bindings differ")
    if result["pool_ref"]["strategy_type"] not in STRATEGY_TYPES:
        raise StrategyError("Voting search provenance strategy_type is invalid")
    if result["target_binding"]["normalized_bad_value"] != 1:
        raise StrategyError("Voting search normalized target semantics changed")
    return result


def _require_result_pool_projection(
    result: Mapping[str, Any],
    *,
    pool: StrategyCandidatePoolArtifactBinding,
) -> None:
    candidate_ids = result["configuration"]["candidate_ids"]
    entries, _excluded = _searchable_entries(pool)
    expected = sorted(entry["rule_id"] for entry in entries)
    if candidate_ids != expected:
        raise StrategyError("Voting search candidate universe changed from Pool")


def _require_live_bindings(
    conn,
    *,
    development: StrategyPoolDevelopmentExecutionBinding,
    resolved: ResolvedPoolRequirements | None,
) -> None:
    require_strategy_pool_development_execution_binding_on_connection(
        conn,
        development,
    )
    if resolved is not None:
        require_resolved_pool_requirements_on_connection(conn, resolved)
    _require_dataset_bytes(development)


def _require_dataset_bytes(
    development: StrategyPoolDevelopmentExecutionBinding,
) -> None:
    path = development.dataset.path
    root = development.pool.datasets_root
    if not path.is_absolute() or not root.is_absolute() or path.is_symlink():
        raise StrategyError("Voting search dataset path is invalid")
    try:
        path.relative_to(root)
        file_stat = os.lstat(path)
    except (OSError, ValueError) as exc:
        raise StrategyError("Voting search dataset is unavailable") from exc
    if not stat.S_ISREG(file_stat.st_mode) or stat.S_ISLNK(file_stat.st_mode):
        raise StrategyError("Voting search dataset must be a regular file")
    if not hmac.compare_digest(
        sha256_file(path),
        development.dataset.content_hash,
    ):
        raise StrategyError("Voting search dataset content hash changed")


def _prepare_output_directory(
    tasks_dir: Path | str,
    *,
    task_id: str,
) -> Path:
    root = Path(tasks_dir).absolute()
    task = _safe_component(task_id, "task_id")
    out_dir = root / task / "strategy_voting_candidate_searches"
    try:
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink():
            raise StrategyError("Voting search task root cannot be a symlink")
        (root / task).mkdir(exist_ok=True)
        if (root / task).is_symlink():
            raise StrategyError("Voting search task directory cannot be a symlink")
        out_dir.mkdir(exist_ok=True)
        if out_dir.is_symlink():
            raise StrategyError("Voting search output directory cannot be a symlink")
    except OSError as exc:
        raise StrategyError(
            "Voting search output directory could not be prepared"
        ) from exc
    return out_dir


def _expected_search_path(
    tasks_dir: Path | str,
    *,
    task_id: str,
    search_id: str,
    pool_snapshot_hash: str,
) -> Path:
    task = _safe_component(task_id, "task_id")
    search = _search_id(search_id, "search_id")
    snapshot = _hash(pool_snapshot_hash, "pool_snapshot_hash")
    return (
        Path(tasks_dir).absolute()
        / task
        / "strategy_voting_candidate_searches"
        / f"{search}-{snapshot[:16]}.json"
    )


def _require_existing_artifact(
    row,
    *,
    task_id: str,
    path: Path,
    canonical: bytes,
    content_hash: str,
    provenance: Mapping[str, Any],
    root: Path,
) -> None:
    if (
        str(row["task_id"]) != task_id
        or str(row["kind"]) != VOTING_CANDIDATE_SEARCH_ARTIFACT_KIND
        or str(row["path"]) != str(path)
        or not hmac.compare_digest(str(row["content_hash"]), content_hash)
        or str(row["origin_tool"]) != VOTING_CANDIDATE_SEARCH_ORIGIN_TOOL
        or str(row["provenance_json"]) != _canonical_json(provenance)
    ):
        raise StrategyError("existing Voting search registry binding changed")
    _require_exact_file(
        path,
        root=root,
        canonical=canonical,
        content_hash=content_hash,
    )


def _require_exact_file(
    path: Path,
    *,
    root: Path,
    canonical: bytes,
    content_hash: str,
) -> None:
    raw = _read_exact_file(
        path,
        root=root,
        expected_content_hash=content_hash,
    )
    if raw != canonical:
        raise StrategyError("Voting search artifact content changed")


def _read_exact_file(
    path: Path,
    *,
    root: Path,
    expected_content_hash: str,
) -> bytes:
    if not path.is_absolute() or not root.is_absolute():
        raise StrategyError("Voting search artifact path is not canonical")
    try:
        path.relative_to(root)
        opened = os.lstat(path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_ISLNK(opened.st_mode)
            or int(opened.st_size) > MAX_JSON_BYTES
        ):
            raise StrategyError("Voting search artifact file is invalid")
        raw = path.read_bytes()
    except StrategyError:
        raise
    except (OSError, ValueError) as exc:
        raise StrategyError("Voting search artifact is unavailable") from exc
    if len(raw) > MAX_JSON_BYTES or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(),
        expected_content_hash,
    ):
        raise StrategyError("Voting search artifact content hash changed")
    return raw


def _require_artifact_row(
    conn,
    *,
    binding: VotingCandidateSearchArtifactBinding,
) -> None:
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
        raise StrategyError("Voting search artifact is no longer registered")
    if (
        str(row["id"]) != binding.artifact_id
        or str(row["task_id"]) != binding.task_id
        or str(row["kind"]) != VOTING_CANDIDATE_SEARCH_ARTIFACT_KIND
        or str(row["path"]) != str(binding.artifact_path)
        or not hmac.compare_digest(
            str(row["content_hash"]),
            binding.artifact_content_hash,
        )
        or str(row["origin_tool"]) != VOTING_CANDIDATE_SEARCH_ORIGIN_TOOL
        or str(row["provenance_json"]) != binding.artifact_provenance_json
    ):
        raise StrategyError("Voting search artifact registry binding changed")


def _tool_output(
    *,
    task_id: str,
    pool: StrategyCandidatePoolArtifactBinding,
    result: Mapping[str, Any],
    excluded_unsupported_rule_ids: Sequence[str],
    record: Mapping[str, Any],
    path: Path,
) -> dict[str, Any]:
    artifact_id = _hash(record["id"], "artifact_id")
    artifact_hash = _hash(record["content_hash"], "artifact content_hash")
    output = {
        "schema_version": VOTING_CANDIDATE_SEARCH_TOOL_SCHEMA_VERSION,
        "search_id": result["search_id"],
        "request_hash": result["request_hash"],
        "content_hash": result["content_hash"],
        "pool_id": pool.pool["pool_id"],
        "pool_revision": pool.pool["revision"],
        "pool_snapshot_hash": pool.pool["snapshot_hash"],
        "search_space": result["search_space"],
        "evaluated": result["evaluated"],
        "truncated": result["truncated"],
        "eligible": result["eligible"],
        "excluded_unsupported_rule_ids": sorted(
            _text_array(
                excluded_unsupported_rule_ids,
                "excluded_unsupported_rule_ids",
            )
        ),
        "search_result": dict(result),
        "artifacts": [
            {
                "artifact_id": artifact_id,
                "kind": VOTING_CANDIDATE_SEARCH_ARTIFACT_KIND,
                "format": "json",
                "filename": path.name,
                "content_hash": artifact_hash,
                "download_url": (
                    f"/api/tasks/{quote(task_id, safe='')}"
                    f"/task-artifacts/{quote(artifact_id, safe='')}/download"
                    f"?expected_content_hash={artifact_hash}"
                ),
            }
        ],
        "not_mutated_pool": True,
        "not_selected": True,
        "not_admitted": True,
        "not_applied": True,
        "not_adopted": True,
        "not_deployed": True,
    }
    if set(output) != _OUTPUT_FIELDS:
        raise StrategyError("Voting search Tool output fields changed")
    return output


def _expression_fields(value: object) -> set[str]:
    if isinstance(value, Mapping):
        fields = {
            item
            for key, item in value.items()
            if key == "field" and isinstance(item, str) and item
        }
        for item in value.values():
            fields.update(_expression_fields(item))
        return fields
    if isinstance(value, Sequence) and not isinstance(
        value,
        str | bytes | bytearray,
    ):
        fields: set[str] = set()
        for item in value:
            fields.update(_expression_fields(item))
        return fields
    return set()


def _modeling_runtime(runtime):
    if hasattr(runtime, "experiments") and hasattr(runtime, "modeling_repo"):
        return runtime
    proxy = SimpleNamespace(**vars(runtime))
    proxy.experiments = ExperimentStore(runtime.settings.db_path)
    proxy.modeling_repo = ModelingRepository(runtime.settings.db_path)
    return proxy


def _json_object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise StrategyError(f"{name} must be an object")
    return dict(value)


def _array(value: object, name: str) -> list[Any]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(
        value,
        Sequence,
    ):
        raise StrategyError(f"{name} must be an array")
    return list(value)


def _text_array(value: object, name: str) -> list[str]:
    rows = [_text(item, f"{name} item") for item in _array(value, name)]
    if len(rows) != len(set(rows)):
        raise StrategyError(f"{name} must not contain duplicates")
    return rows


def _exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    if set(value) != expected:
        raise StrategyError(f"{name} fields are invalid")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise StrategyError(f"{name} must be non-empty text")
    return value


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _hash(value: object, name: str) -> str:
    text = _text(value, name)
    if _HASH_RE.fullmatch(text) is None:
        raise StrategyError(f"{name} must be a lowercase SHA-256 hash")
    return text


def _search_id(value: object, name: str) -> str:
    text = _text(value, name)
    if _SEARCH_ID_RE.fullmatch(text) is None:
        raise StrategyError(f"{name} is invalid")
    return text


def _combo_id(value: object, name: str) -> str:
    text = _text(value, name)
    if _COMBO_ID_RE.fullmatch(text) is None:
        raise StrategyError(f"{name} is invalid")
    return text


def _strategy_type(value: object, name: str) -> str:
    text = _text(value, name)
    if text not in STRATEGY_TYPES:
        raise StrategyError(f"{name} is invalid")
    return text


def _safe_component(value: object, name: str) -> str:
    text = _text(value, name)
    if _SAFE_COMPONENT_RE.fullmatch(text) is None or text in {".", ".."}:
        raise StrategyError(f"{name} is unsafe")
    return text


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise StrategyError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StrategyError(f"{name} must be a non-negative integer")
    return value


def _bounded_int(value: object, name: str, *, maximum: int) -> int:
    normalized = _positive_int(value, name)
    if normalized > maximum:
        raise StrategyError(f"{name} must not exceed {maximum}")
    return normalized


def _binary_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in {0, 1}:
        raise StrategyError(f"{name} must be integer 0 or 1")
    return value


def _strict_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise StrategyError(f"{name} must be boolean")
    return value


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
        raise StrategyError("Voting search binding is not canonical JSON") from exc


__all__ = [
    "VOTING_CANDIDATE_SEARCH_ARTIFACT_KIND",
    "VOTING_CANDIDATE_SEARCH_ARTIFACT_SCHEMA_VERSION",
    "VOTING_CANDIDATE_SEARCH_ORIGIN_TOOL",
    "VOTING_CANDIDATE_SEARCH_SELECTION_TOOL_SCHEMA_VERSION",
    "VOTING_CANDIDATE_SEARCH_TOOL_SCHEMA_VERSION",
    "VotingCandidateSearchArtifactBinding",
    "VotingCandidateSearchSelectionBinding",
    "load_voting_candidate_search_artifact",
    "require_voting_candidate_search_artifact_binding_on_connection",
    "run_build_voting_candidate_from_search",
    "resolve_voting_candidate_search_inputs",
    "resolve_voting_candidate_search_selection",
    "run_search_voting_candidates",
]
