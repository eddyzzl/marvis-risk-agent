"""Governed Tool boundary for bounded 2D/3D Cross threshold-rule search."""

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

import numpy as np
import pandas as pd

from marvis.artifacts import ArtifactUnitOfWork
from marvis.packs.strategy import candidate_asset_tools
from marvis.packs.strategy.candidate_fragment import (
    sample_context_hash_from_candidate_evidence,
)
from marvis.packs.strategy.cross_candidate_search_tools import (
    _load_sample_binding,
    _read_search_sample,
    _select_ranked_axes,
)
from marvis.packs.strategy.cross_matrix_candidate_tools import _amount_array
from marvis.packs.strategy.cross_rule_candidate import (
    CROSS_RULE_CANDIDATE_PRODUCER_VERSION,
    build_cross_rule_candidate,
    canonical_cross_rule_candidate_json,
    validate_cross_rule_candidate,
)
from marvis.packs.strategy.cross_rule_search import (
    CROSS_RULE_SEARCH_PRODUCER_VERSION,
    CROSS_RULE_SEARCH_REQUEST_SCHEMA_VERSION,
    MAX_ARTIFACT_BYTES,
    MAX_FEATURES,
    MAX_ROW_EVALUATIONS,
    MAX_THRESHOLDS_PER_FEATURE,
    MAX_TRIALS,
    canonical_cross_rule_search_result_json,
    canonical_cross_rule_trial_prefix,
    parse_cross_rule_search_result_json,
    search_cross_threshold_rules,
    validate_cross_rule_search_result,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.evaluator import evaluate_expression_frame
from marvis.packs.strategy.sample_design_execution import (
    StrategyRiskDevelopmentExecutionBinding,
    require_historical_strategy_risk_development_execution_binding_on_connection,
    revalidate_historical_strategy_risk_development_execution_binding,
)


CROSS_RULE_SEARCH_TOOL_SCHEMA_VERSION = (
    "strategy.search-cross-threshold-rules-tool.v1"
)
CROSS_RULE_SEARCH_ARTIFACT_KIND = "strategy_cross_rule_search_json"
CROSS_RULE_SEARCH_ARTIFACT_SCHEMA_VERSION = (
    "strategy.cross-rule-search-artifact.v1"
)
CROSS_RULE_SEARCH_ORIGIN_TOOL = "strategy.search_cross_threshold_rules"
CROSS_RULE_CANDIDATE_SELECTION_TOOL_SCHEMA_VERSION = (
    "strategy.build-cross-rule-candidate-from-search-tool.v1"
)
CROSS_RULE_CANDIDATE_ARTIFACT_KIND = (
    "strategy_cross_rule_candidate_json"
)
CROSS_RULE_CANDIDATE_ARTIFACT_SCHEMA_VERSION = (
    "strategy.cross-rule-candidate-artifact.v1"
)
CROSS_RULE_CANDIDATE_ORIGIN_TOOL = (
    "strategy.build_cross_rule_candidate_from_search"
)

_INPUT_FIELDS = frozenset(
    {
        "source_artifact_id",
        "expected_artifact_content_hash",
        "expected_candidate_id",
        "expected_evidence_hash",
        "features",
        "dimension",
        "constraints",
        "max_trials",
    }
)
_CONSTRAINT_FIELDS = frozenset(
    {
        "min_lift",
        "min_bad_count",
        "max_hit_share",
        "min_amount_lift",
    }
)
_SELECTION_INPUT_FIELDS = frozenset(
    {"search_id", "rule_id", "selection_reason"}
)
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
        "dimension",
        "constraints",
        "max_trials",
        "lifecycle",
    }
)
_CANDIDATE_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "task_id",
        "asset_id",
        "asset_hash",
        "search_artifact_id",
        "search_artifact_content_hash",
        "search_id",
        "search_content_hash",
        "rule_id",
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
_SEARCH_ID_RE = re.compile(r"^cross-rule-search-[0-9a-f]{32}$")
_RULE_ID_RE = re.compile(r"^cross-rule-[0-9a-f]{32}$")
_ASSET_ID_RE = re.compile(r"^cross-rule-asset-[0-9a-f]{32}$")


@dataclass(frozen=True)
class CrossRuleSearchArtifactBinding:
    """Authenticated aggregate rule search and immutable source chain."""

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


@dataclass(frozen=True)
class CrossRuleCandidateArtifactBinding:
    """Authenticated materialized rule plus its exact search lineage."""

    task_id: str
    artifact_id: str
    artifact_path: Path
    artifact_content_hash: str
    artifact_provenance: dict[str, Any]
    artifact_provenance_json: str
    candidate: dict[str, Any]
    search: CrossRuleSearchArtifactBinding
    tasks_root: Path
    db_path: Path


def run_search_cross_threshold_rules(inputs, ctx, runtime) -> dict[str, Any]:
    """Evaluate and persist one exact bounded threshold-rule prefix."""

    request = _validate_inputs(inputs)
    task_id = _text(ctx.task_id, "task_id")
    source = candidate_asset_tools._load_source_artifact(
        runtime,
        task_id=task_id,
        artifact_id=request["source_artifact_id"],
        expected_content_hash=request["expected_artifact_content_hash"],
        expected_candidate_id=request["expected_candidate_id"],
        expected_evidence_hash=request["expected_evidence_hash"],
    )
    evidence = _load_parent_evidence(
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
    features, sentinels = _select_rule_features(
        evidence,
        dataset=dataset,
        requested_features=request["features"],
    )
    governed = sorted(
        set(request["features"]) & set(sample_binding.excluded_feature_columns)
    )
    if governed:
        raise StrategyError(
            "Cross rule search features cannot use target, sample partition, "
            "or population columns: "
            + ", ".join(governed)
        )
    prefix = canonical_cross_rule_trial_prefix(
        features,
        dimension=request["dimension"],
        max_trials=request["max_trials"],
    )
    planned = int(evidence["analysis"]["row_count"]) * len(prefix)
    if planned > MAX_ROW_EVALUATIONS:
        raise StrategyError(
            "Cross rule search row evaluations exceed hard budget "
            f"({planned} > {MAX_ROW_EVALUATIONS})"
        )
    labeled, target, projection = _read_search_sample(
        runtime,
        evidence=evidence,
        source=source,
        dataset=dataset,
        sample_binding=sample_binding,
        features=features,
    )
    vectors = {
        item["feature"]: _numeric_feature(
            labeled[item["feature"]],
            feature=item["feature"],
        )
        for item in features
    }
    loan = _amount_array(
        labeled,
        projection["loan_amount_col"],
        "loan_amount",
    )
    overdue = _amount_array(
        labeled,
        projection["overdue_amount_col"],
        "overdue_amount",
    )
    trials = [
        _measure_trial(
            conditions,
            vectors=vectors,
            sentinels=sentinels,
            target=target,
            loan=loan,
            overdue=overdue,
        )
        for conditions in prefix
    ]
    core_request = {
        "schema_version": CROSS_RULE_SEARCH_REQUEST_SCHEMA_VERSION,
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
            "loan_amount_sum": _amount_sum(loan),
            "overdue_amount_sum": _amount_sum(overdue),
        },
        "dimension": request["dimension"],
        "features": features,
        "constraints": request["constraints"],
        "trials": trials,
        "max_trials": request["max_trials"],
    }
    result = search_cross_threshold_rules(core_request)
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


def run_build_cross_rule_candidate_from_search(
    inputs,
    ctx,
    runtime,
) -> dict[str, Any]:
    """Recompute one exact search and persist the explicitly named rule."""

    request = _validate_selection_inputs(inputs)
    task_id = _text(ctx.task_id, "task_id")
    search, rule = resolve_cross_rule_search_rule(
        runtime,
        task_id=task_id,
        search_id=request["search_id"],
        rule_id=request["rule_id"],
    )
    labeled, target, projection = replay_cross_rule_search_binding(
        runtime,
        search,
    )
    candidate = build_cross_rule_candidate(
        search.result,
        search_artifact_ref={
            "artifact_id": search.artifact_id,
            "artifact_content_hash": search.artifact_content_hash,
        },
        rule_id=rule["rule_id"],
        selection_reason=request["selection_reason"],
    )
    _require_candidate_replays(
        candidate,
        labeled=labeled,
        target=target,
        loan_amount_col=projection["loan_amount_col"],
        overdue_amount_col=projection["overdue_amount_col"],
    )
    return _persist_candidate(
        runtime,
        task_id=task_id,
        search=search,
        candidate=candidate,
    )


def resolve_cross_rule_search_rule(
    runtime,
    *,
    task_id: str,
    search_id: str,
    rule_id: str,
) -> tuple[CrossRuleSearchArtifactBinding, dict[str, Any]]:
    """Resolve one rule from exactly one task-owned authenticated search."""

    task = _text(task_id, "task_id")
    search_value = _search_id(search_id, "search_id")
    rule_value = _rule_id(rule_id, "rule_id")
    matches: list[CrossRuleSearchArtifactBinding] = []
    for record in runtime.task_artifacts.list_for_task(task):
        provenance = (
            record.get("provenance")
            if isinstance(record, Mapping)
            else None
        )
        if (
            isinstance(record, Mapping)
            and record.get("kind") == CROSS_RULE_SEARCH_ARTIFACT_KIND
            and isinstance(provenance, Mapping)
            and provenance.get("search_id") == search_value
        ):
            matches.append(
                load_cross_rule_search_artifact(
                    runtime,
                    task_id=task,
                    artifact_id=record["id"],
                    expected_artifact_content_hash=record["content_hash"],
                    expected_search_id=search_value,
                    expected_search_content_hash=provenance.get(
                        "search_content_hash"
                    ),
                )
            )
    if not matches:
        raise StrategyError(
            "Cross rule search artifact not found; run the search again"
        )
    if len(matches) != 1:
        raise StrategyError("Cross rule search identity is ambiguous")
    search = matches[0]
    selected = next(
        (
            item
            for item in search.result["rules"]
            if hmac.compare_digest(item["rule_id"], rule_value)
        ),
        None,
    )
    if selected is None:
        raise StrategyError(
            "rule_id is not an authenticated evaluated Cross rule"
        )
    return search, dict(selected)


def replay_cross_rule_search_binding(
    runtime,
    binding: CrossRuleSearchArtifactBinding,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Recompute the complete bounded search from immutable source bytes."""

    if not isinstance(binding, CrossRuleSearchArtifactBinding):
        raise StrategyError("Cross rule search binding is invalid")
    result = validate_cross_rule_search_result(binding.result)
    features, sentinels = _select_rule_features(
        binding.evidence,
        dataset=binding.dataset,
        requested_features=[
            item["feature"]
            for item in result["configuration"]["features"]
        ],
    )
    if features != result["configuration"]["features"]:
        raise StrategyError("Cross rule search feature evidence changed")
    prefix = canonical_cross_rule_trial_prefix(
        features,
        dimension=result["configuration"]["dimension"],
        max_trials=result["configuration"]["max_trials"],
    )
    labeled, target, projection = _read_search_sample(
        runtime,
        evidence=binding.evidence,
        source=binding.source,
        dataset=binding.dataset,
        sample_binding=binding.sample_binding,
        features=features,
    )
    vectors = {
        item["feature"]: _numeric_feature(
            labeled[item["feature"]],
            feature=item["feature"],
        )
        for item in features
    }
    loan = _amount_array(
        labeled,
        projection["loan_amount_col"],
        "loan_amount",
    )
    overdue = _amount_array(
        labeled,
        projection["overdue_amount_col"],
        "overdue_amount",
    )
    trials = [
        _measure_trial(
            conditions,
            vectors=vectors,
            sentinels=sentinels,
            target=target,
            loan=loan,
            overdue=overdue,
        )
        for conditions in prefix
    ]
    replayed = search_cross_threshold_rules(
        {
            "schema_version": CROSS_RULE_SEARCH_REQUEST_SCHEMA_VERSION,
            "source": result["source"],
            "population": {
                "row_count": len(labeled),
                "good": int(len(labeled) - target.sum()),
                "bad": int(target.sum()),
                "loan_amount_sum": _amount_sum(loan),
                "overdue_amount_sum": _amount_sum(overdue),
            },
            "dimension": result["configuration"]["dimension"],
            "features": features,
            "constraints": result["configuration"]["constraints"],
            "trials": trials,
            "max_trials": result["configuration"]["max_trials"],
        }
    )
    if replayed != result:
        raise StrategyError(
            "Cross rule search did not replay from immutable evidence"
        )
    return labeled, target, projection


def load_cross_rule_search_artifact(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_artifact_content_hash: str,
    expected_search_id: str | None = None,
    expected_search_content_hash: str | None = None,
) -> CrossRuleSearchArtifactBinding:
    """Authenticate one aggregate search and its historical lineage."""

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
            "Cross rule search exact identity requires both id and hash"
        )
    frozen_id = (
        None
        if expected_search_id is None
        else _search_id(expected_search_id, "expected_search_id")
    )
    frozen_hash = (
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
        or record.get("kind") != CROSS_RULE_SEARCH_ARTIFACT_KIND
        or record.get("origin_tool") != CROSS_RULE_SEARCH_ORIGIN_TOOL
        or not hmac.compare_digest(
            str(record.get("content_hash")),
            artifact_hash,
        )
    ):
        raise StrategyError("Cross rule search registry binding changed")
    tasks_root = Path(runtime.settings.tasks_dir).absolute()
    path = Path(_text(record.get("path"), "Cross rule search path"))
    candidate_asset_tools._require_regular_artifact_path(
        path,
        root=tasks_root,
    )
    candidate_asset_tools._require_file_content_hash(
        path,
        artifact_hash,
        "Cross rule search artifact content changed",
    )
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise StrategyError(
            "Cross rule search artifact could not be read"
        ) from exc
    if len(raw) > MAX_ARTIFACT_BYTES:
        raise StrategyError("Cross rule search artifact exceeds byte budget")
    if not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(),
        artifact_hash,
    ):
        raise StrategyError("Cross rule search artifact changed during read")
    result = parse_cross_rule_search_result_json(raw)
    if raw != canonical_cross_rule_search_result_json(result).encode("utf-8"):
        raise StrategyError("Cross rule search artifact is not canonical")
    if frozen_id is not None and (
        result["search_id"] != frozen_id
        or not hmac.compare_digest(result["content_hash"], frozen_hash)
    ):
        raise StrategyError("Cross rule search identity changed")
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
        raise StrategyError("Cross rule search provenance changed")
    expected_path = _expected_path(
        runtime.settings.tasks_dir,
        task_id=task,
        search_id=result["search_id"],
        artifact_content_hash=artifact_hash,
    )
    if path != expected_path:
        raise StrategyError("Cross rule search path is not canonical")
    source = candidate_asset_tools._load_source_artifact(
        runtime,
        task_id=task,
        artifact_id=provenance["source_artifact_id"],
        expected_content_hash=provenance[
            "source_artifact_content_hash"
        ],
        expected_candidate_id=provenance["candidate_id"],
        expected_evidence_hash=provenance["evidence_hash"],
    )
    evidence = _load_parent_evidence(
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
    selected, _sentinels = _select_rule_features(
        evidence,
        dataset=dataset,
        requested_features=[
            item["feature"]
            for item in result["configuration"]["features"]
        ],
    )
    if selected != result["configuration"]["features"]:
        raise StrategyError(
            "Cross rule search feature evidence changed"
        )
    expected_static = _static_provenance(
        task_id=task,
        source=source,
        dataset=dataset,
        sample_binding=sample_binding,
        evidence=evidence,
        result=result,
        dropped=provenance["nan_labels_dropped"],
    )
    if provenance != expected_static:
        raise StrategyError("Cross rule search provenance changed")
    binding = CrossRuleSearchArtifactBinding(
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
        require_cross_rule_search_artifact_binding_on_connection(
            conn,
            binding,
        )
        conn.commit()
    return binding


def load_cross_rule_candidate_artifact(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_artifact_content_hash: str,
    expected_asset_id: str,
    expected_asset_hash: str,
) -> CrossRuleCandidateArtifactBinding:
    """Authenticate one materialized rule and its exact search pointer."""

    task = _text(task_id, "task_id")
    artifact = _hash(artifact_id, "artifact_id")
    artifact_hash = _hash(
        expected_artifact_content_hash,
        "expected_artifact_content_hash",
    )
    asset_id = _asset_id(expected_asset_id, "expected_asset_id")
    asset_hash = _hash(expected_asset_hash, "expected_asset_hash")
    record = runtime.task_artifacts.get_for_task(task, artifact)
    if (
        not isinstance(record, Mapping)
        or record.get("id") != artifact
        or record.get("task_id") != task
        or record.get("kind") != CROSS_RULE_CANDIDATE_ARTIFACT_KIND
        or record.get("origin_tool") != CROSS_RULE_CANDIDATE_ORIGIN_TOOL
        or not hmac.compare_digest(
            str(record.get("content_hash")),
            artifact_hash,
        )
    ):
        raise StrategyError("Cross rule candidate registry binding changed")
    tasks_root = Path(runtime.settings.tasks_dir).absolute()
    path = Path(_text(record.get("path"), "Cross rule candidate path"))
    candidate_asset_tools._require_regular_artifact_path(
        path,
        root=tasks_root,
    )
    candidate_asset_tools._require_file_content_hash(
        path,
        artifact_hash,
        "Cross rule candidate artifact content changed",
    )
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise StrategyError(
            "Cross rule candidate artifact could not be read"
        ) from exc
    if len(raw) > MAX_ARTIFACT_BYTES:
        raise StrategyError(
            "Cross rule candidate artifact exceeds byte budget"
        )
    if not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(),
        artifact_hash,
    ):
        raise StrategyError(
            "Cross rule candidate artifact changed during read"
        )
    try:
        payload = json.loads(raw)
        candidate = validate_cross_rule_candidate(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, StrategyError) as exc:
        raise StrategyError(
            "Cross rule candidate artifact is invalid"
        ) from exc
    if raw != canonical_cross_rule_candidate_json(candidate).encode("utf-8"):
        raise StrategyError(
            "Cross rule candidate artifact is not canonical"
        )
    if (
        candidate["asset_id"] != asset_id
        or not hmac.compare_digest(candidate["asset_hash"], asset_hash)
    ):
        raise StrategyError("Cross rule candidate identity changed")
    provenance = _validate_candidate_provenance(
        record.get("provenance")
    )
    if (
        provenance["task_id"] != task
        or provenance["asset_id"] != asset_id
        or not hmac.compare_digest(provenance["asset_hash"], asset_hash)
    ):
        raise StrategyError("Cross rule candidate provenance changed")
    expected_path = _expected_candidate_path(
        runtime.settings.tasks_dir,
        task_id=task,
        asset_id=asset_id,
        artifact_content_hash=artifact_hash,
    )
    if path != expected_path:
        raise StrategyError(
            "Cross rule candidate artifact path is not canonical"
        )
    search = load_cross_rule_search_artifact(
        runtime,
        task_id=task,
        artifact_id=provenance["search_artifact_id"],
        expected_artifact_content_hash=provenance[
            "search_artifact_content_hash"
        ],
        expected_search_id=provenance["search_id"],
        expected_search_content_hash=provenance["search_content_hash"],
    )
    rebuilt = build_cross_rule_candidate(
        search.result,
        search_artifact_ref={
            "artifact_id": search.artifact_id,
            "artifact_content_hash": search.artifact_content_hash,
        },
        rule_id=provenance["rule_id"],
        selection_reason=candidate["selection_reason"],
    )
    if rebuilt != candidate:
        raise StrategyError(
            "Cross rule candidate does not rebuild from exact search"
        )
    expected_provenance = _candidate_provenance(
        task_id=task,
        search=search,
        candidate=candidate,
    )
    if provenance != expected_provenance:
        raise StrategyError("Cross rule candidate provenance changed")
    binding = CrossRuleCandidateArtifactBinding(
        task_id=task,
        artifact_id=artifact,
        artifact_path=path,
        artifact_content_hash=artifact_hash,
        artifact_provenance=provenance,
        artifact_provenance_json=_canonical_json(provenance),
        candidate=candidate,
        search=search,
        tasks_root=tasks_root,
        db_path=Path(runtime.settings.db_path).absolute(),
    )
    with runtime.task_artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        require_cross_rule_candidate_artifact_binding_on_connection(
            conn,
            binding,
        )
        conn.commit()
    return binding


def require_cross_rule_candidate_artifact_binding_on_connection(
    conn,
    binding: CrossRuleCandidateArtifactBinding,
) -> None:
    """Recheck the candidate, search lineage, file, and registry row."""

    if not isinstance(binding, CrossRuleCandidateArtifactBinding):
        raise StrategyError("Cross rule candidate binding is invalid")
    if not conn.in_transaction:
        raise StrategyError(
            "Cross rule candidate binding requires a caller-owned transaction"
        )
    if validate_cross_rule_candidate(binding.candidate) != binding.candidate:
        raise StrategyError("Cross rule candidate binding changed")
    provenance = _validate_candidate_provenance(
        binding.artifact_provenance
    )
    if (
        provenance != binding.artifact_provenance
        or _canonical_json(provenance)
        != binding.artifact_provenance_json
    ):
        raise StrategyError(
            "Cross rule candidate provenance binding changed"
        )
    require_cross_rule_search_artifact_binding_on_connection(
        conn,
        binding.search,
    )
    candidate_asset_tools._require_regular_artifact_path(
        binding.artifact_path,
        root=binding.tasks_root,
    )
    candidate_asset_tools._require_file_content_hash(
        binding.artifact_path,
        binding.artifact_content_hash,
        "Cross rule candidate artifact changed",
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
        raise StrategyError(
            "Cross rule candidate artifact is no longer registered"
        )
    if (
        str(row["id"]) != binding.artifact_id
        or str(row["task_id"]) != binding.task_id
        or str(row["kind"]) != CROSS_RULE_CANDIDATE_ARTIFACT_KIND
        or str(row["path"]) != str(binding.artifact_path)
        or not hmac.compare_digest(
            str(row["content_hash"]),
            binding.artifact_content_hash,
        )
        or str(row["origin_tool"]) != CROSS_RULE_CANDIDATE_ORIGIN_TOOL
        or str(row["provenance_json"])
        != binding.artifact_provenance_json
    ):
        raise StrategyError(
            "Cross rule candidate registry binding changed"
        )


def replay_cross_rule_candidate_binding(
    runtime,
    binding: CrossRuleCandidateArtifactBinding,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Replay search and candidate condition from immutable source bytes."""

    if not isinstance(binding, CrossRuleCandidateArtifactBinding):
        raise StrategyError("Cross rule candidate binding is invalid")
    labeled, target, projection = replay_cross_rule_search_binding(
        runtime,
        binding.search,
    )
    _require_candidate_replays(
        binding.candidate,
        labeled=labeled,
        target=target,
        loan_amount_col=projection["loan_amount_col"],
        overdue_amount_col=projection["overdue_amount_col"],
    )
    return labeled, target, projection


def require_cross_rule_search_artifact_binding_on_connection(
    conn,
    binding: CrossRuleSearchArtifactBinding,
) -> None:
    """Recheck search, source, sample, dataset, file, and registry row."""

    if not isinstance(binding, CrossRuleSearchArtifactBinding):
        raise StrategyError("Cross rule search binding is invalid")
    if not conn.in_transaction:
        raise StrategyError(
            "Cross rule search binding requires a caller-owned transaction"
        )
    database = conn.execute(
        "SELECT file FROM pragma_database_list WHERE name = 'main'"
    ).fetchone()
    if (
        database is None
        or not str(database["file"])
        or Path(str(database["file"])).absolute() != binding.db_path
    ):
        raise StrategyError("Cross rule search database binding changed")
    if validate_cross_rule_search_result(binding.result) != binding.result:
        raise StrategyError("Cross rule search result binding changed")
    provenance = _validate_provenance(binding.artifact_provenance)
    if (
        provenance != binding.artifact_provenance
        or _canonical_json(provenance)
        != binding.artifact_provenance_json
    ):
        raise StrategyError("Cross rule search provenance binding changed")
    candidate_asset_tools._require_source_on_connection(
        conn,
        binding.source,
    )
    candidate_asset_tools._require_dataset_on_connection(
        conn,
        binding.dataset,
    )
    require_historical_strategy_risk_development_execution_binding_on_connection(
        conn,
        binding.sample_binding,
    )
    for path, expected_hash, message in (
        (
            binding.source.path,
            binding.source.content_hash,
            "Cross rule search source changed",
        ),
        (
            binding.artifact_path,
            binding.artifact_content_hash,
            "Cross rule search artifact changed",
        ),
    ):
        candidate_asset_tools._require_regular_artifact_path(
            path,
            root=binding.tasks_root,
        )
        candidate_asset_tools._require_file_content_hash(
            path,
            expected_hash,
            message,
        )
    candidate_asset_tools._require_file_content_hash(
        binding.dataset.path,
        binding.dataset.content_hash,
        "Cross rule search dataset changed",
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
        raise StrategyError(
            "Cross rule search artifact is no longer registered"
        )
    if (
        str(row["id"]) != binding.artifact_id
        or str(row["task_id"]) != binding.task_id
        or str(row["kind"]) != CROSS_RULE_SEARCH_ARTIFACT_KIND
        or str(row["path"]) != str(binding.artifact_path)
        or not hmac.compare_digest(
            str(row["content_hash"]),
            binding.artifact_content_hash,
        )
        or str(row["origin_tool"]) != CROSS_RULE_SEARCH_ORIGIN_TOOL
        or str(row["provenance_json"])
        != binding.artifact_provenance_json
    ):
        raise StrategyError("Cross rule search registry binding changed")


def _select_rule_features(
    evidence: Mapping[str, Any],
    *,
    dataset,
    requested_features: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, tuple[float, ...]]]:
    selected, _axes = _select_ranked_axes(
        evidence,
        dataset=dataset,
        requested_features=requested_features,
    )
    analysis_features = evidence["analysis"].get("features")
    if not _sequence(analysis_features):
        raise StrategyError(
            "Cross rule search parent features are invalid"
        )
    result: list[dict[str, Any]] = []
    sentinels: dict[str, tuple[float, ...]] = {}
    for selected_row in selected:
        feature = selected_row["feature"]
        method = selected_row["method"]
        feature_rows = [
            item
            for item in analysis_features
            if isinstance(item, Mapping)
            and item.get("feature") == feature
        ]
        if len(feature_rows) != 1:
            raise StrategyError(
                f"Cross rule search feature evidence is ambiguous: {feature}"
            )
        methods = feature_rows[0].get("methods")
        method_rows = [
            item
            for item in methods
            if isinstance(item, Mapping)
            and item.get("method") == method
        ] if _sequence(methods) else []
        if (
            len(method_rows) != 1
            or method_rows[0].get("status") != "available"
        ):
            raise StrategyError(
                f"Cross rule search method evidence is unavailable: "
                f"{feature}/{method}"
            )
        method_row = method_rows[0]
        metrics = method_row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise StrategyError(
                f"Cross rule search metrics are invalid: {feature}/{method}"
            )
        direction = metrics.get("risk_direction")
        if direction not in {
            "increasing",
            "decreasing",
            "non_monotonic",
            "flat",
        }:
            raise StrategyError(
                f"Cross rule search risk direction is invalid: {feature}"
            )
        bins = method_row.get("bins")
        if not _sequence(bins):
            raise StrategyError(
                f"Cross rule search bins are invalid: {feature}/{method}"
            )
        regular = [
            item
            for item in bins
            if isinstance(item, Mapping)
            and item.get("kind") == "numeric_interval"
        ]
        if len(regular) < 2:
            raise StrategyError(
                f"Cross rule search requires at least two numeric bins: "
                f"{feature}/{method}"
            )
        thresholds = sorted(
            {
                float(bound)
                for item in regular
                for bound in (item.get("lower"), item.get("upper"))
                if isinstance(bound, Real)
                and not isinstance(bound, bool)
                and math.isfinite(float(bound))
            }
        )
        if not 1 <= len(thresholds) <= MAX_THRESHOLDS_PER_FEATURE:
            raise StrategyError(
                "Cross rule search feature thresholds must contain 1..8 "
                f"values: {feature}/{method}"
            )
        missing_rows = [
            item
            for item in bins
            if isinstance(item, Mapping) and item.get("kind") == "missing"
        ]
        if len(missing_rows) > 1:
            raise StrategyError(
                f"Cross rule search missing bin is ambiguous: {feature}"
            )
        missing_count = (
            0
            if not missing_rows
            else _integer(
                missing_rows[0].get("count"),
                f"{feature} missing count",
                minimum=0,
            )
        )
        missing_bad = (
            0
            if not missing_rows
            else _integer(
                missing_rows[0].get("bad"),
                f"{feature} missing bad",
                minimum=0,
            )
        )
        sentinel_values = []
        for item in bins:
            if not isinstance(item, Mapping) or item.get("kind") != "sentinel":
                continue
            value = item.get("value")
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                raise StrategyError(
                    f"Cross rule search sentinel is non-numeric: {feature}"
                )
            sentinel_values.append(float(value))
        sentinels[feature] = tuple(sorted(set(sentinel_values)))
        result.append(
            {
                "feature": feature,
                "method": method,
                "risk_direction": direction,
                "thresholds": thresholds,
                "excluded_values": list(sentinels[feature]),
                "missing_count": missing_count,
                "missing_bad": missing_bad,
            }
        )
    return sorted(result, key=lambda item: item["feature"]), sentinels


def _measure_trial(
    conditions: Sequence[Mapping[str, Any]],
    *,
    vectors: Mapping[str, np.ndarray],
    sentinels: Mapping[str, tuple[float, ...]],
    target: np.ndarray,
    loan: np.ndarray | None,
    overdue: np.ndarray | None,
) -> dict[str, Any]:
    mask = np.ones(len(target), dtype=bool)
    for condition in conditions:
        values = vectors[condition["feature"]]
        missing = ~np.isfinite(values)
        if condition["operator"] == "gte":
            current = values >= condition["threshold"]
        elif condition["operator"] == "lt":
            current = values < condition["threshold"]
        else:
            raise StrategyError(
                "Cross rule search condition operator changed"
            )
        excluded = sentinels[condition["feature"]]
        if excluded:
            current &= ~np.isin(values, np.asarray(excluded, dtype=float))
        if condition["include_missing"]:
            current |= missing
        else:
            current &= ~missing
        mask &= current
    count = int(mask.sum())
    bad = int(target[mask].sum())
    return {
        "conditions": [dict(item) for item in conditions],
        "count": count,
        "good": count - bad,
        "bad": bad,
        "loan_amount_sum": _amount_sum(loan, mask=mask),
        "overdue_amount_sum": _amount_sum(overdue, mask=mask),
    }


def _numeric_feature(series: pd.Series, *, feature: str) -> np.ndarray:
    converted = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    invalid = series.notna().to_numpy() & ~np.isfinite(converted)
    if np.any(invalid):
        raise StrategyError(
            f"Cross rule search feature contains non-numeric values: {feature}"
        )
    return converted


def _amount_sum(
    values: np.ndarray | None,
    *,
    mask: np.ndarray | None = None,
) -> float | None:
    if values is None:
        return None
    selected = values if mask is None else values[mask]
    finite = selected[np.isfinite(selected)]
    total = float(finite.sum()) if len(finite) else 0.0
    if not math.isfinite(total):
        raise StrategyError("Cross rule search amount aggregation overflowed")
    return total


def _require_candidate_replays(
    candidate: Mapping[str, Any],
    *,
    labeled: pd.DataFrame,
    target: np.ndarray,
    loan_amount_col: str | None,
    overdue_amount_col: str | None,
) -> None:
    asset = validate_cross_rule_candidate(candidate)
    mask = evaluate_expression_frame(
        labeled,
        asset["condition"],
    ).to_numpy(dtype=bool, copy=False)
    count = int(mask.sum())
    bad = int(target[mask].sum())
    loan = _amount_array(labeled, loan_amount_col, "loan_amount")
    overdue = _amount_array(
        labeled,
        overdue_amount_col,
        "overdue_amount",
    )
    metrics = asset["metrics"]
    expected = {
        "count": count,
        "good": count - bad,
        "bad": bad,
        "loan_amount_sum": _amount_sum(loan, mask=mask),
        "overdue_amount_sum": _amount_sum(overdue, mask=mask),
    }
    if any(metrics[field] != value for field, value in expected.items()):
        raise StrategyError(
            "Cross rule candidate condition did not replay search metrics"
        )


def _persist_candidate(
    runtime,
    *,
    task_id: str,
    search: CrossRuleSearchArtifactBinding,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    asset = validate_cross_rule_candidate(candidate)
    canonical = canonical_cross_rule_candidate_json(asset).encode("utf-8")
    if len(canonical) > MAX_ARTIFACT_BYTES:
        raise StrategyError(
            "Cross rule candidate artifact exceeds byte budget"
        )
    artifact_hash = hashlib.sha256(canonical).hexdigest()
    tasks_root = Path(runtime.settings.tasks_dir)
    out_dir = tasks_root / task_id / "strategy_cross_rule_candidates"
    candidate_asset_tools._require_output_directory_boundary(
        out_dir,
        root=tasks_root,
    )
    final_path = _expected_candidate_path(
        tasks_root,
        task_id=task_id,
        asset_id=asset["asset_id"],
        artifact_content_hash=artifact_hash,
    )
    provenance = _candidate_provenance(
        task_id=task_id,
        search=search,
        candidate=asset,
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
                require_cross_rule_search_artifact_binding_on_connection(
                    conn,
                    search,
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
                        CROSS_RULE_CANDIDATE_ARTIFACT_KIND,
                        str(final_path),
                    ),
                ).fetchone()
                if row is None:
                    if final_path.exists() or final_path.is_symlink():
                        raise StrategyError(
                            "Cross rule candidate path exists without registry"
                        )
                    uow.promote_all()
                else:
                    if (
                        str(row["task_id"]) != task_id
                        or str(row["kind"])
                        != CROSS_RULE_CANDIDATE_ARTIFACT_KIND
                        or str(row["path"]) != str(final_path)
                        or not hmac.compare_digest(
                            str(row["content_hash"]),
                            artifact_hash,
                        )
                        or str(row["origin_tool"])
                        != CROSS_RULE_CANDIDATE_ORIGIN_TOOL
                        or str(row["provenance_json"])
                        != _canonical_json(provenance)
                    ):
                        raise StrategyError(
                            "Cross rule candidate existing binding changed"
                        )
                    candidate_asset_tools._require_file_content_hash(
                        final_path,
                        artifact_hash,
                        "Cross rule candidate existing content changed",
                    )
                    uow.rollback()
                    reused = True
                candidate_asset_tools._require_file_content_hash(
                    final_path,
                    artifact_hash,
                    "Cross rule candidate changed before registration",
                )
                record = runtime.task_artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=CROSS_RULE_CANDIDATE_ARTIFACT_KIND,
                    path=str(final_path),
                    content_hash=artifact_hash,
                    origin_tool=CROSS_RULE_CANDIDATE_ORIGIN_TOOL,
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
    selection = asset["source_selection"]
    return {
        "schema_version": (
            CROSS_RULE_CANDIDATE_SELECTION_TOOL_SCHEMA_VERSION
        ),
        "source_search_selection": {
            "search_id": selection["search_id"],
            "rule_id": selection["rule_id"],
            "rank": selection["rule_rank"],
            "eligible": selection["eligible"],
            "constraint_failures": selection["constraint_failures"],
        },
        "candidate": asset,
        "artifacts": [
            {
                "artifact_id": str(record["id"]),
                "kind": CROSS_RULE_CANDIDATE_ARTIFACT_KIND,
                "format": "json",
                "filename": final_path.name,
                "content_hash": str(record["content_hash"]),
                "download_url": (
                    f"/api/tasks/{quote(task_id, safe='')}"
                    f"/task-artifacts/{quote(str(record['id']), safe='')}/download"
                ),
            }
        ],
        "not_admitted": True,
        "not_applied": True,
        "not_adopted": True,
        "not_deployed": True,
    }


def _candidate_provenance(
    *,
    task_id: str,
    search: CrossRuleSearchArtifactBinding,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    asset = validate_cross_rule_candidate(candidate)
    selection = asset["source_selection"]
    if (
        selection["search_artifact_id"] != search.artifact_id
        or not hmac.compare_digest(
            selection["search_artifact_content_hash"],
            search.artifact_content_hash,
        )
        or selection["search_id"] != search.result["search_id"]
        or not hmac.compare_digest(
            selection["search_content_hash"],
            search.result["content_hash"],
        )
    ):
        raise StrategyError(
            "Cross rule candidate search selection changed"
        )
    source_provenance = search.artifact_provenance
    value = {
        "schema_version": CROSS_RULE_CANDIDATE_ARTIFACT_SCHEMA_VERSION,
        "producer_version": CROSS_RULE_CANDIDATE_PRODUCER_VERSION,
        "task_id": task_id,
        "asset_id": asset["asset_id"],
        "asset_hash": asset["asset_hash"],
        "search_artifact_id": search.artifact_id,
        "search_artifact_content_hash": search.artifact_content_hash,
        "search_id": search.result["search_id"],
        "search_content_hash": search.result["content_hash"],
        "rule_id": selection["rule_id"],
        "source_artifact_id": search.source.artifact_id,
        "source_artifact_content_hash": search.source.content_hash,
        "candidate_id": search.evidence["candidate_id"],
        "evidence_hash": search.evidence["evidence_hash"],
        "dataset_id": search.dataset.dataset_id,
        "dataset_content_hash": search.dataset.content_hash,
        "registry_metadata_hash": search.dataset.registry_metadata_hash,
        "workspace_revision": search.evidence["identity"][
            "workspace_revision"
        ],
        "workspace_generation": search.evidence["identity"][
            "workspace_generation"
        ],
        "semantic_mapping_hash": search.evidence["identity"][
            "semantic_mapping_hash"
        ],
        "sample_design_ref": search.sample_binding.to_ref_dict(),
        "sample_context_hash": source_provenance["sample_context_hash"],
        "sample_partition": "risk/development",
        "lifecycle": dict(_LIFECYCLE),
    }
    return _validate_candidate_provenance(value)


def _validate_candidate_provenance(value: object) -> dict[str, Any]:
    obj = _object(value, "Cross rule candidate provenance")
    if set(obj) != _CANDIDATE_PROVENANCE_FIELDS:
        raise StrategyError(
            "Cross rule candidate provenance fields changed"
        )
    if (
        obj["schema_version"]
        != CROSS_RULE_CANDIDATE_ARTIFACT_SCHEMA_VERSION
        or obj["producer_version"]
        != CROSS_RULE_CANDIDATE_PRODUCER_VERSION
    ):
        raise StrategyError(
            "Cross rule candidate provenance version changed"
        )
    lifecycle = _object(
        obj["lifecycle"],
        "candidate provenance.lifecycle",
    )
    if lifecycle != _LIFECYCLE:
        raise StrategyError(
            "Cross rule candidate provenance lifecycle changed"
        )
    sample_ref = _object(
        obj["sample_design_ref"],
        "candidate provenance.sample_design_ref",
    )
    if set(sample_ref) != {
        "artifact_id",
        "artifact_content_hash",
        "sample_design_id",
        "sample_design_content_hash",
        "partition",
    }:
        raise StrategyError(
            "Cross rule candidate sample_design_ref changed"
        )
    result = {
        "schema_version": CROSS_RULE_CANDIDATE_ARTIFACT_SCHEMA_VERSION,
        "producer_version": CROSS_RULE_CANDIDATE_PRODUCER_VERSION,
        "task_id": _text(obj["task_id"], "candidate provenance.task_id"),
        "asset_id": _asset_id(
            obj["asset_id"],
            "candidate provenance.asset_id",
        ),
        "asset_hash": _hash(
            obj["asset_hash"],
            "candidate provenance.asset_hash",
        ),
        "search_artifact_id": _hash(
            obj["search_artifact_id"],
            "candidate provenance.search_artifact_id",
        ),
        "search_artifact_content_hash": _hash(
            obj["search_artifact_content_hash"],
            "candidate provenance.search_artifact_content_hash",
        ),
        "search_id": _search_id(
            obj["search_id"],
            "candidate provenance.search_id",
        ),
        "search_content_hash": _hash(
            obj["search_content_hash"],
            "candidate provenance.search_content_hash",
        ),
        "rule_id": _rule_id(
            obj["rule_id"],
            "candidate provenance.rule_id",
        ),
        "source_artifact_id": _hash(
            obj["source_artifact_id"],
            "candidate provenance.source_artifact_id",
        ),
        "source_artifact_content_hash": _hash(
            obj["source_artifact_content_hash"],
            "candidate provenance.source_artifact_content_hash",
        ),
        "candidate_id": _candidate_id(
            obj["candidate_id"],
            "candidate provenance.candidate_id",
        ),
        "evidence_hash": _hash(
            obj["evidence_hash"],
            "candidate provenance.evidence_hash",
        ),
        "dataset_id": _text(
            obj["dataset_id"],
            "candidate provenance.dataset_id",
        ),
        "dataset_content_hash": _hash(
            obj["dataset_content_hash"],
            "candidate provenance.dataset_content_hash",
        ),
        "registry_metadata_hash": _hash(
            obj["registry_metadata_hash"],
            "candidate provenance.registry_metadata_hash",
        ),
        "workspace_revision": _integer(
            obj["workspace_revision"],
            "candidate provenance.workspace_revision",
            minimum=1,
        ),
        "workspace_generation": _integer(
            obj["workspace_generation"],
            "candidate provenance.workspace_generation",
            minimum=0,
        ),
        "semantic_mapping_hash": _hash(
            obj["semantic_mapping_hash"],
            "candidate provenance.semantic_mapping_hash",
        ),
        "sample_design_ref": sample_ref,
        "sample_context_hash": _hash(
            obj["sample_context_hash"],
            "candidate provenance.sample_context_hash",
        ),
        "sample_partition": obj["sample_partition"],
        "lifecycle": dict(_LIFECYCLE),
    }
    if result["sample_partition"] != "risk/development":
        raise StrategyError(
            "Cross rule candidate sample partition changed"
        )
    return result


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
    normalized = validate_cross_rule_search_result(result)
    canonical = canonical_cross_rule_search_result_json(normalized).encode(
        "utf-8"
    )
    if len(canonical) > MAX_ARTIFACT_BYTES:
        raise StrategyError(
            "Cross rule search artifact exceeds byte budget"
        )
    artifact_hash = hashlib.sha256(canonical).hexdigest()
    tasks_root = Path(runtime.settings.tasks_dir)
    out_dir = tasks_root / task_id / "strategy_cross_rule_searches"
    candidate_asset_tools._require_output_directory_boundary(
        out_dir,
        root=tasks_root,
    )
    final_path = _expected_path(
        tasks_root,
        task_id=task_id,
        search_id=normalized["search_id"],
        artifact_content_hash=artifact_hash,
    )
    dropped = (
        sample_binding.development_population_count
        - normalized["population"]["row_count"]
    )
    provenance = _static_provenance(
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
                candidate_asset_tools._require_source_on_connection(
                    conn,
                    source,
                )
                candidate_asset_tools._require_dataset_on_connection(
                    conn,
                    dataset,
                )
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
                        CROSS_RULE_SEARCH_ARTIFACT_KIND,
                        str(final_path),
                    ),
                ).fetchone()
                if row is None:
                    if final_path.exists() or final_path.is_symlink():
                        raise StrategyError(
                            "Cross rule search path exists without registry"
                        )
                    uow.promote_all()
                else:
                    if (
                        str(row["task_id"]) != task_id
                        or str(row["kind"])
                        != CROSS_RULE_SEARCH_ARTIFACT_KIND
                        or str(row["path"]) != str(final_path)
                        or not hmac.compare_digest(
                            str(row["content_hash"]),
                            artifact_hash,
                        )
                        or str(row["origin_tool"])
                        != CROSS_RULE_SEARCH_ORIGIN_TOOL
                        or str(row["provenance_json"])
                        != _canonical_json(provenance)
                    ):
                        raise StrategyError(
                            "Cross rule search existing binding changed"
                        )
                    candidate_asset_tools._require_file_content_hash(
                        final_path,
                        artifact_hash,
                        "Cross rule search existing content changed",
                    )
                    uow.rollback()
                    reused = True
                candidate_asset_tools._require_file_content_hash(
                    final_path,
                    artifact_hash,
                    "Cross rule search content changed before registration",
                )
                record = runtime.task_artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=CROSS_RULE_SEARCH_ARTIFACT_KIND,
                    path=str(final_path),
                    content_hash=artifact_hash,
                    origin_tool=CROSS_RULE_SEARCH_ORIGIN_TOOL,
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
        "schema_version": CROSS_RULE_SEARCH_TOOL_SCHEMA_VERSION,
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
                "kind": CROSS_RULE_SEARCH_ARTIFACT_KIND,
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


def _static_provenance(
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
        raise StrategyError("Cross rule search source identity changed")
    if (
        isinstance(dropped, bool)
        or not isinstance(dropped, int)
        or dropped < 0
        or result["population"]["row_count"] + dropped
        != sample_binding.development_population_count
    ):
        raise StrategyError(
            "Cross rule search labelled population binding changed"
        )
    value = {
        "schema_version": CROSS_RULE_SEARCH_ARTIFACT_SCHEMA_VERSION,
        "producer_version": CROSS_RULE_SEARCH_PRODUCER_VERSION,
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
        "workspace_generation": evidence["identity"][
            "workspace_generation"
        ],
        "semantic_mapping_hash": evidence["identity"][
            "semantic_mapping_hash"
        ],
        "sample_design_ref": sample_binding.to_ref_dict(),
        "sample_context_hash": result["source"]["sample_context_hash"],
        "sample_partition": "risk/development",
        "target_col": sample_binding.target_col,
        "drop_nan_labels": sample_binding.drop_nan_labels,
        "nan_labels_dropped": dropped,
        "labeled_count": result["population"]["row_count"],
        "features": result["configuration"]["features"],
        "dimension": result["configuration"]["dimension"],
        "constraints": result["configuration"]["constraints"],
        "max_trials": result["configuration"]["max_trials"],
        "lifecycle": dict(_LIFECYCLE),
    }
    return _validate_provenance(value)


def _validate_provenance(value: object) -> dict[str, Any]:
    obj = _object(value, "Cross rule search provenance")
    if set(obj) != _PROVENANCE_FIELDS:
        raise StrategyError(
            "Cross rule search provenance fields changed"
        )
    if (
        obj["schema_version"] != CROSS_RULE_SEARCH_ARTIFACT_SCHEMA_VERSION
        or obj["producer_version"] != CROSS_RULE_SEARCH_PRODUCER_VERSION
    ):
        raise StrategyError(
            "Cross rule search provenance version changed"
        )
    if _object(obj["lifecycle"], "provenance.lifecycle") != _LIFECYCLE:
        raise StrategyError(
            "Cross rule search provenance lifecycle changed"
        )
    sample_ref = _object(
        obj["sample_design_ref"],
        "provenance.sample_design_ref",
    )
    if set(sample_ref) != {
        "artifact_id",
        "artifact_content_hash",
        "sample_design_id",
        "sample_design_content_hash",
        "partition",
    }:
        raise StrategyError(
            "Cross rule search sample_design_ref changed"
        )
    # Reuse the domain result validator to normalize features and constraints.
    features = obj["features"]
    constraints = _constraints(obj["constraints"])
    result = {
        "schema_version": CROSS_RULE_SEARCH_ARTIFACT_SCHEMA_VERSION,
        "producer_version": CROSS_RULE_SEARCH_PRODUCER_VERSION,
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
        "dataset_id": _text(
            obj["dataset_id"],
            "provenance.dataset_id",
        ),
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
            minimum=1,
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
        "sample_design_ref": sample_ref,
        "sample_context_hash": _hash(
            obj["sample_context_hash"],
            "provenance.sample_context_hash",
        ),
        "sample_partition": obj["sample_partition"],
        "target_col": _text(
            obj["target_col"],
            "provenance.target_col",
        ),
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
        "features": json.loads(_canonical_json(features)),
        "dimension": _dimension(obj["dimension"]),
        "constraints": constraints,
        "max_trials": _integer(
            obj["max_trials"],
            "provenance.max_trials",
            minimum=1,
            maximum=MAX_TRIALS,
        ),
        "lifecycle": dict(_LIFECYCLE),
    }
    if not isinstance(result["drop_nan_labels"], bool):
        raise StrategyError(
            "Cross rule search drop_nan_labels changed"
        )
    if result["sample_partition"] != "risk/development":
        raise StrategyError(
            "Cross rule search sample partition changed"
        )
    return result


def _load_parent_evidence(
    source,
    *,
    task_id: str,
    expected_candidate_id: str,
    expected_evidence_hash: str,
) -> dict[str, Any]:
    # The existing Cross loader already performs strict canonical report
    # validation and exact task/candidate/evidence binding.
    from marvis.packs.strategy.cross_matrix_candidate_tools import (
        _load_exact_parent_evidence,
    )

    return _load_exact_parent_evidence(
        source,
        task_id=task_id,
        expected_candidate_id=expected_candidate_id,
        expected_evidence_hash=expected_evidence_hash,
    )


def _validate_inputs(value: object) -> dict[str, Any]:
    obj = _object(value, "search_cross_threshold_rules inputs")
    if set(obj) != _INPUT_FIELDS:
        raise StrategyError(
            "search_cross_threshold_rules input fields are invalid"
        )
    features = [
        _text(item, f"features[{index}]")
        for index, item in enumerate(_array(obj["features"], "features"))
    ]
    if not 2 <= len(features) <= MAX_FEATURES:
        raise StrategyError(
            "Cross rule search features must contain 2..12 values"
        )
    if len(set(features)) != len(features):
        raise StrategyError("Cross rule search features must be unique")
    dimension = _dimension(obj["dimension"])
    if len(features) < dimension:
        raise StrategyError(
            f"Cross rule search requires at least {dimension} features"
        )
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
        "dimension": dimension,
        "constraints": _constraints(obj["constraints"]),
        "max_trials": _integer(
            obj["max_trials"],
            "max_trials",
            minimum=1,
            maximum=MAX_TRIALS,
        ),
    }


def _validate_selection_inputs(value: object) -> dict[str, Any]:
    obj = _object(
        value,
        "build_cross_rule_candidate_from_search inputs",
    )
    if set(obj) != _SELECTION_INPUT_FIELDS:
        raise StrategyError(
            "build_cross_rule_candidate_from_search input fields are invalid"
        )
    reason = obj["selection_reason"]
    if reason is not None:
        reason = _text(reason, "selection_reason")
        if len(reason) > 500:
            raise StrategyError(
                "selection_reason exceeds 500 characters"
            )
    return {
        "search_id": _search_id(obj["search_id"], "search_id"),
        "rule_id": _rule_id(obj["rule_id"], "rule_id"),
        "selection_reason": reason,
    }


def _constraints(value: object) -> dict[str, Any]:
    obj = _object(value, "constraints")
    if set(obj) != _CONSTRAINT_FIELDS:
        raise StrategyError("Cross rule search constraint fields are invalid")
    return {
        "min_lift": _finite(
            obj["min_lift"],
            "constraints.min_lift",
            minimum=0.0,
            maximum=1_000.0,
        ),
        "min_bad_count": _integer(
            obj["min_bad_count"],
            "constraints.min_bad_count",
            minimum=0,
        ),
        "max_hit_share": _finite(
            obj["max_hit_share"],
            "constraints.max_hit_share",
            minimum=0.0,
            maximum=1.0,
        ),
        "min_amount_lift": _optional_finite(
            obj["min_amount_lift"],
            "constraints.min_amount_lift",
            minimum=0.0,
            maximum=1_000.0,
        ),
    }


def _expected_path(
    tasks_dir: Path | str,
    *,
    task_id: str,
    search_id: str,
    artifact_content_hash: str,
) -> Path:
    return (
        Path(tasks_dir)
        / task_id
        / "strategy_cross_rule_searches"
        / f"{search_id}_{artifact_content_hash[:12]}.json"
    )


def _expected_candidate_path(
    tasks_dir: Path | str,
    *,
    task_id: str,
    asset_id: str,
    artifact_content_hash: str,
) -> Path:
    return (
        Path(tasks_dir)
        / task_id
        / "strategy_cross_rule_candidates"
        / f"{asset_id}_{artifact_content_hash[:12]}.json"
    )


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise StrategyError(f"{name} must be an object")
    return dict(value)


def _array(value: object, name: str) -> list[Any]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or not value
    ):
        raise StrategyError(f"{name} must be a non-empty array")
    return list(value)


def _sequence(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and bool(value)
    )


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise StrategyError(f"{name} must be non-empty text")
    return value.strip()


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise StrategyError(f"{name} must be a lowercase SHA-256 hash")
    return value


def _candidate_id(value: object, name: str) -> str:
    text = _text(value, name)
    if _CANDIDATE_ID_RE.fullmatch(text) is None:
        raise StrategyError(f"{name} has an invalid candidate id")
    return text


def _search_id(value: object, name: str) -> str:
    text = _text(value, name)
    if _SEARCH_ID_RE.fullmatch(text) is None:
        raise StrategyError(f"{name} has an invalid search id")
    return text


def _rule_id(value: object, name: str) -> str:
    text = _text(value, name)
    if _RULE_ID_RE.fullmatch(text) is None:
        raise StrategyError(f"{name} has an invalid rule id")
    return text


def _asset_id(value: object, name: str) -> str:
    text = _text(value, name)
    if _ASSET_ID_RE.fullmatch(text) is None:
        raise StrategyError(f"{name} has an invalid asset id")
    return text


def _dimension(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) not in {2, 3}
    ):
        raise StrategyError("dimension must be 2 or 3")
    return int(value)


def _integer(
    value: object,
    name: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        upper = "" if maximum is None else f" and {maximum}"
        raise StrategyError(
            f"{name} must be an integer between {minimum}{upper}"
        )
    result = int(value)
    if result < minimum or (
        maximum is not None and result > maximum
    ):
        upper = "" if maximum is None else f" and {maximum}"
        raise StrategyError(
            f"{name} must be between {minimum}{upper}"
        )
    return result


def _finite(
    value: object,
    name: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise StrategyError(f"{name} must be a finite number")
    result = float(value)
    if (
        not math.isfinite(result)
        or result < minimum
        or (maximum is not None and result > maximum)
    ):
        raise StrategyError(f"{name} is outside its allowed range")
    return result


def _optional_finite(
    value: object,
    name: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float | None:
    if value is None:
        return None
    return _finite(
        value,
        name,
        minimum=minimum,
        maximum=maximum,
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


__all__ = [
    "CROSS_RULE_CANDIDATE_ARTIFACT_KIND",
    "CROSS_RULE_CANDIDATE_ARTIFACT_SCHEMA_VERSION",
    "CROSS_RULE_CANDIDATE_ORIGIN_TOOL",
    "CROSS_RULE_CANDIDATE_SELECTION_TOOL_SCHEMA_VERSION",
    "CROSS_RULE_SEARCH_ARTIFACT_KIND",
    "CROSS_RULE_SEARCH_ARTIFACT_SCHEMA_VERSION",
    "CROSS_RULE_SEARCH_ORIGIN_TOOL",
    "CROSS_RULE_SEARCH_TOOL_SCHEMA_VERSION",
    "CrossRuleCandidateArtifactBinding",
    "CrossRuleSearchArtifactBinding",
    "load_cross_rule_candidate_artifact",
    "load_cross_rule_search_artifact",
    "replay_cross_rule_candidate_binding",
    "replay_cross_rule_search_binding",
    "require_cross_rule_candidate_artifact_binding_on_connection",
    "require_cross_rule_search_artifact_binding_on_connection",
    "resolve_cross_rule_search_rule",
    "run_build_cross_rule_candidate_from_search",
    "run_search_cross_threshold_rules",
]
