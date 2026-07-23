"""Governed artifact boundary for univariate Strategy ModelEvidence V2.

The current modeling artifacts do not expose authenticated model/score lineage.
This first vertical therefore translates only the structured, task-owned
``strategy_candidate_json`` report into risk/development univariate evidence.
It never accepts caller-computed bins or metrics and never manufactures model,
monthly, comparison, validation, or OOT evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

from marvis.artifacts import ArtifactUnitOfWork
from marvis.artifacts.transactional import ArtifactTransactionError
from marvis.output.strategy_candidate_report import (
    canonical_strategy_candidate_report_json,
    strategy_candidate_report_from_json,
)
from marvis.packs.strategy.candidate_evidence import validate_candidate_evidence
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.model_evidence import (
    DEFAULT_PRODUCER_VERSION,
    MAX_BINS_PER_EVIDENCE,
    MAX_MODEL_EVIDENCE_JSON_BYTES,
    MAX_OBSERVATIONS_PER_EVIDENCE,
    MAX_UNIVARIATE_EVIDENCE,
    build_artifact_ref,
    build_evidence_source_ref,
    build_strategy_model_evidence_bundle,
    build_univariate_bin_ref,
    build_univariate_evidence,
    build_univariate_observation,
    canonical_strategy_model_evidence_bundle_json,
    strategy_model_evidence_bundle_from_json,
    validate_strategy_model_evidence_bundle,
)
from marvis.packs.strategy.sample_design_binding import StrategySampleDesignRef
from marvis.packs.strategy.sample_design_v2 import (
    validate_strategy_sample_design_v2_bundle,
)
from marvis.packs.strategy.sample_design_v2_tools import (
    StrategySampleDesignV2ArtifactBinding,
    load_strategy_sample_design_v2_artifacts,
    require_strategy_sample_design_v2_artifact_binding_on_connection,
)
from marvis.repositories.task_artifacts import (
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
)


MODEL_EVIDENCE_V2_TOOL_SCHEMA_VERSION = (
    "strategy.materialize-model-evidence-v2-tool.v3"
)
MODEL_EVIDENCE_V2_ARTIFACT_SCHEMA_VERSION = (
    "strategy.model-evidence-v2-artifact.v1"
)
MODEL_EVIDENCE_V2_ARTIFACT_KIND = "strategy_model_evidence_v2_json"
MODEL_EVIDENCE_V2_ORIGIN_TOOL = "strategy.materialize_model_evidence_v2"

_SOURCE_ARTIFACT_KIND = "strategy_candidate_json"
_SOURCE_ORIGIN_TOOL = "strategy.analyze_univariate_candidates"
_SOURCE_PROVENANCE_SCHEMA_VERSION = "strategy.univariate-candidate-artifact.v1"
_SOURCE_PRODUCER_VERSION = "strategy.univariate-candidate/1"
_MAX_CANDIDATE_REPORT_BYTES = 32 * 1024 * 1024
_MAX_CANDIDATE_SOURCE_BYTES_TOTAL = MAX_MODEL_EVIDENCE_JSON_BYTES
# Every accepted source must emit at least one evidence item.  Keep authenticated
# DB/file fan-out well below the bundle's 500-item ceiling as a separate I/O bound.
_MAX_UNIVARIATE_SOURCES = min(100, MAX_UNIVARIATE_EVIDENCE)
_MAX_TRANSLATION_WARNINGS = MAX_UNIVARIATE_EVIDENCE
_MAX_TRANSLATION_METHODS = MAX_UNIVARIATE_EVIDENCE + _MAX_TRANSLATION_WARNINGS
# A global ceiling prevents the per-evidence 50k observation allowance from
# multiplying across 500 evidence items before the final JSON check can run.
_MAX_TRANSLATION_OBSERVATIONS = MAX_OBSERVATIONS_PER_EVIDENCE
_OBSERVATIONS_PER_TRANSLATED_METHOD = 5
_OBSERVATIONS_PER_TRANSLATED_BIN = 8
_MAX_TRANSLATION_BINS = (
    _MAX_TRANSLATION_OBSERVATIONS // _OBSERVATIONS_PER_TRANSLATED_BIN
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_INPUT_FIELDS = frozenset({"sample_design_ref", "univariate_sources"})
_SAMPLE_DESIGN_INPUT_FIELDS = frozenset(
    {
        "membership_artifact_id",
        "expected_membership_artifact_content_hash",
        "bundle_artifact_id",
        "expected_bundle_artifact_content_hash",
        "expected_bundle_id",
        "expected_sample_design_id",
        "expected_sample_design_content_hash",
    }
)
_SOURCE_INPUT_FIELDS = frozenset(
    {
        "artifact_id",
        "expected_artifact_content_hash",
        "expected_candidate_id",
        "expected_evidence_hash",
    }
)
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
_RECORD_FIELDS = frozenset(
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
_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "format",
        "task_id",
        "bundle_id",
        "bundle_content_hash",
        "bundle_artifact_content_hash",
        "sample_design_bundle_content_hash",
        "sample_design_ref",
        "membership_id",
        "membership_content_hash",
        "dataset_ref",
        "dataset_source_path",
        "dataset_registry_metadata_hash",
        "workspace_ref",
        "legacy_sample_design_ref",
        "univariate_sources",
        "request_hash",
        "translation_warnings",
    }
)
_OUTPUT_FIELDS = frozenset(
    {
        "schema_version",
        "content_hash",
        "bundle_id",
        "bundle_content_hash",
        "sample_design_bundle_id",
        "sample_design_id",
        "sample_design_content_hash",
        "sample_design_bundle",
        "bundle",
        "artifact",
        "source_artifacts",
        "univariate_only",
        "not_created_model",
        "not_compared_models",
        "not_adopted",
        "not_deployed",
    }
)
_ARTIFACT_OUTPUT_FIELDS = frozenset(
    {"kind", "format", "filename", "content_hash"}
)
_SOURCE_OUTPUT_FIELDS = frozenset(
    {"artifact_id", "kind", "content_hash"}
)
_BOUNDARY_ERRORS = (
    ArtifactTransactionError,
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
)


@dataclass(frozen=True)
class _CandidateSourceBinding:
    task_id: str
    artifact_id: str
    path: Path
    content_hash: str
    provenance: dict[str, Any]
    provenance_json: str
    report: dict[str, Any]
    request: dict[str, str]
    dataset_source_path: str
    canonical_bytes: int


@dataclass
class _TranslationBudget:
    sources: int = 0
    methods: int = 0
    evidence: int = 0
    bins: int = 0
    observations: int = 0
    warnings: int = 0
    warning_json_bytes: int = 0
    evidence_json_bytes: int = 0

    def begin_source(self) -> None:
        self.sources += 1
        if self.sources > _MAX_UNIVARIATE_SOURCES:
            raise StrategyError("model-evidence translation source budget exceeded")

    def begin_method(self) -> None:
        self.methods += 1
        if self.methods > _MAX_TRANSLATION_METHODS:
            raise StrategyError("model-evidence translation method budget exceeded")

    def begin_available_method(self, *, bin_count: int) -> None:
        if bin_count > MAX_BINS_PER_EVIDENCE:
            raise StrategyError("model-evidence translation per-method bin budget exceeded")
        if self.evidence + 1 > MAX_UNIVARIATE_EVIDENCE:
            raise StrategyError("model-evidence translation evidence budget exceeded")
        if self.bins + bin_count > _MAX_TRANSLATION_BINS:
            raise StrategyError("model-evidence translation bin budget exceeded")
        projected_observations = (
            self.observations
            + _OBSERVATIONS_PER_TRANSLATED_METHOD
            + _OBSERVATIONS_PER_TRANSLATED_BIN * bin_count
        )
        if projected_observations > _MAX_TRANSLATION_OBSERVATIONS:
            raise StrategyError("model-evidence translation observation budget exceeded")
        self.evidence += 1
        self.observations += _OBSERVATIONS_PER_TRANSLATED_METHOD

    def begin_bin(self) -> None:
        self.bins += 1
        self.observations += _OBSERVATIONS_PER_TRANSLATED_BIN
        if self.bins > _MAX_TRANSLATION_BINS:
            raise StrategyError("model-evidence translation bin budget exceeded")
        if self.observations > _MAX_TRANSLATION_OBSERVATIONS:
            raise StrategyError("model-evidence translation observation budget exceeded")

    def add_warning(self, warning: str) -> None:
        self.warnings += 1
        if self.warnings > _MAX_TRANSLATION_WARNINGS:
            raise StrategyError("model-evidence translation warning budget exceeded")
        self.warning_json_bytes += len(_canonical_json(warning).encode("utf-8")) + 1
        if self.warning_json_bytes > MAX_MODEL_EVIDENCE_JSON_BYTES:
            raise StrategyError("model-evidence translation warning byte budget exceeded")

    def add_evidence_json(self, evidence: Mapping[str, Any]) -> None:
        self.evidence_json_bytes += len(_canonical_json(evidence).encode("utf-8")) + 1
        if self.evidence_json_bytes > MAX_MODEL_EVIDENCE_JSON_BYTES:
            raise StrategyError("model-evidence translation JSON byte budget exceeded")


@dataclass(frozen=True)
class StrategyModelEvidenceV2ArtifactBinding:
    """Authenticated model-evidence artifact and all of its dependencies."""

    task_id: str
    artifact_id: str
    path: Path
    artifact_content_hash: str
    provenance: dict[str, Any]
    bundle: dict[str, Any]
    sample_design_binding: StrategySampleDesignV2ArtifactBinding
    sources: tuple[_CandidateSourceBinding, ...]
    warnings: tuple[str, ...]
    tasks_root: Path
    db_path: Path


def run_materialize_model_evidence_v2(inputs, ctx, runtime) -> dict[str, Any]:
    """Translate authenticated candidate reports into one immutable V2 bundle."""

    try:
        request = _validate_inputs(inputs)
        task_id = _text(ctx.task_id, "task_id")
        sample_binding = _load_sample_design(runtime, task_id, request)
        sources = _load_candidate_sources(
            runtime,
            task_id=task_id,
            requests=request["univariate_sources"],
            sample_binding=sample_binding,
        )
        bundle, warnings = _translate_sources(
            sample_binding=sample_binding,
            sources=sources,
        )
        return _persist_bundle(
            runtime,
            task_id=task_id,
            request=request,
            sample_binding=sample_binding,
            sources=sources,
            bundle=bundle,
            warnings=warnings,
        )
    except StrategyError:
        raise
    except _BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc


def validate_materialize_model_evidence_v2_tool_output(
    value: object,
) -> dict[str, Any]:
    """Validate a cached Tool envelope without trusting its display fields."""

    _preflight_array_limit(
        value,
        field="source_artifacts",
        maximum=_MAX_UNIVARIATE_SOURCES,
        error="model-evidence V2 source summaries exceed source budget",
    )
    obj = _json_object(value, "materialize_model_evidence_v2 output")
    _exact_fields(obj, _OUTPUT_FIELDS, "materialize_model_evidence_v2 output")
    output_hash = _hash(obj["content_hash"], "output.content_hash")
    sample_bundle = validate_strategy_sample_design_v2_bundle(
        obj["sample_design_bundle"]
    )
    bundle = validate_strategy_model_evidence_bundle(
        obj["bundle"], sample_design_bundle=sample_bundle
    )
    design = sample_bundle["sample_design"]
    expected_scalars = {
        "schema_version": MODEL_EVIDENCE_V2_TOOL_SCHEMA_VERSION,
        "bundle_id": bundle["bundle_id"],
        "bundle_content_hash": bundle["content_hash"],
        "sample_design_bundle_id": sample_bundle["bundle_id"],
        "sample_design_id": design["sample_design_id"],
        "sample_design_content_hash": design["content_hash"],
    }
    for field, expected in expected_scalars.items():
        if obj[field] != expected:
            raise StrategyError(f"model-evidence V2 output {field} drifted")
    if bundle["model_evidence"] or bundle["comparison_evidence"]:
        raise StrategyError("model-evidence V2 Tool output must remain univariate-only")
    if any(
        item["sample_ref"]["population"] != "risk"
        or item["sample_ref"]["partition"] != "development"
        for item in bundle["univariate_evidence"]
    ):
        raise StrategyError("model-evidence V2 Tool emitted non-development evidence")

    canonical = canonical_strategy_model_evidence_bundle_json(
        bundle, sample_design_bundle=sample_bundle
    ).encode("utf-8")
    artifact_hash = _sha256(canonical)
    artifact = _validate_output_artifact(
        obj["artifact"],
        bundle_id=bundle["bundle_id"],
        expected_content_hash=artifact_hash,
    )
    sources = _validate_source_outputs(obj["source_artifacts"])
    referenced = {
        (
            item["analysis_ref"]["kind"],
            item["analysis_ref"]["ref_id"],
            item["analysis_ref"]["content_hash"],
        )
        for item in bundle["univariate_evidence"]
    }
    declared = {
        (item["kind"], item["artifact_id"], item["content_hash"])
        for item in sources
    }
    if referenced != declared:
        raise StrategyError(
            "model-evidence V2 source summaries do not match bundle analysis refs"
        )
    if any(
        obj[field] is not True
        for field in (
            "univariate_only",
            "not_created_model",
            "not_compared_models",
            "not_adopted",
            "not_deployed",
        )
    ):
        raise StrategyError("model-evidence V2 governance flags must be true")
    obj["sample_design_bundle"] = sample_bundle
    obj["bundle"] = bundle
    obj["artifact"] = artifact
    obj["source_artifacts"] = sources
    addressed = {key: item for key, item in obj.items() if key != "content_hash"}
    expected_hash = _sha256(_canonical_json(addressed).encode("utf-8"))
    if not hmac.compare_digest(output_hash, expected_hash):
        raise StrategyError("model-evidence V2 output content_hash does not match content")
    return obj


def load_strategy_model_evidence_v2_artifact(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_artifact_content_hash: str,
    expected_bundle_id: str,
    expected_bundle_content_hash: str,
    sample_design_ref: Mapping[str, Any],
) -> StrategyModelEvidenceV2ArtifactBinding:
    """Load and deterministically re-authenticate one persisted V2 bundle."""

    try:
        normalized_task = _text(task_id, "task_id")
        normalized_artifact_id = _hash(artifact_id, "artifact_id")
        artifact_hash = _hash(
            expected_artifact_content_hash, "expected_artifact_content_hash"
        )
        bundle_id = _text(expected_bundle_id, "expected_bundle_id")
        bundle_content_hash = _hash(
            expected_bundle_content_hash, "expected_bundle_content_hash"
        )
        request = _validate_inputs(
            {
                "sample_design_ref": sample_design_ref,
                "univariate_sources": _provenance_sources_for_artifact(
                    runtime,
                    task_id=normalized_task,
                    artifact_id=normalized_artifact_id,
                    expected_content_hash=artifact_hash,
                    expected_bundle_id=bundle_id,
                ),
            }
        )
        sample_binding = _load_sample_design(runtime, normalized_task, request)
        record = _registered_output_record(
            runtime,
            task_id=normalized_task,
            artifact_id=normalized_artifact_id,
            expected_content_hash=artifact_hash,
        )
        provenance = _validate_provenance(record["provenance"])
        _require_provenance_binding(
            provenance,
            task_id=normalized_task,
            request=request,
            sample_binding=sample_binding,
            artifact_content_hash=artifact_hash,
            expected_bundle_id=bundle_id,
            expected_bundle_content_hash=bundle_content_hash,
        )
        path = _expected_output_path(
            runtime, task_id=normalized_task, bundle_id=bundle_id
        )
        if Path(str(record["path"])) != path:
            raise StrategyError("model-evidence V2 artifact path is not canonical")
        raw = _read_verified(
            path,
            root=Path(runtime.settings.tasks_dir),
            expected_hash=artifact_hash,
            maximum_bytes=MAX_MODEL_EVIDENCE_JSON_BYTES,
        )
        bundle = strategy_model_evidence_bundle_from_json(
            raw, sample_design_bundle=sample_binding.bundle
        )
        canonical = canonical_strategy_model_evidence_bundle_json(
            bundle, sample_design_bundle=sample_binding.bundle
        ).encode("utf-8")
        if canonical != raw:
            raise StrategyError("model-evidence V2 artifact bytes are not canonical")
        if (
            bundle["bundle_id"] != bundle_id
            or not hmac.compare_digest(bundle["content_hash"], bundle_content_hash)
        ):
            raise StrategyError("model-evidence V2 artifact identity changed")
        sources = _load_candidate_sources(
            runtime,
            task_id=normalized_task,
            requests=request["univariate_sources"],
            sample_binding=sample_binding,
        )
        rebuilt, warnings = _translate_sources(
            sample_binding=sample_binding,
            sources=sources,
        )
        if rebuilt != bundle or list(warnings) != provenance["translation_warnings"]:
            raise StrategyError(
                "model-evidence V2 artifact no longer matches deterministic sources"
            )
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            loaded_binding = StrategyModelEvidenceV2ArtifactBinding(
                task_id=normalized_task,
                artifact_id=normalized_artifact_id,
                path=path,
                artifact_content_hash=artifact_hash,
                provenance=provenance,
                bundle=bundle,
                sample_design_binding=sample_binding,
                sources=tuple(sources),
                warnings=tuple(warnings),
                tasks_root=Path(runtime.settings.tasks_dir).absolute(),
                db_path=Path(runtime.settings.db_path).absolute(),
            )
            require_strategy_model_evidence_v2_artifact_binding_on_connection(
                conn,
                loaded_binding,
            )
            conn.commit()
        return loaded_binding
    except StrategyError:
        raise
    except _BOUNDARY_ERRORS as exc:
        raise StrategyError(str(exc)) from exc


def require_strategy_model_evidence_v2_artifact_binding_on_connection(
    conn,
    binding: StrategyModelEvidenceV2ArtifactBinding,
) -> None:
    """Re-authenticate model evidence while a downstream writer owns the lock."""

    if not isinstance(binding, StrategyModelEvidenceV2ArtifactBinding):
        raise StrategyError("model-evidence V2 artifact binding is invalid")
    if not conn.in_transaction:
        raise StrategyError(
            "model-evidence V2 binding requires a caller-owned transaction"
        )
    database = conn.execute(
        "SELECT file FROM pragma_database_list WHERE name = 'main'"
    ).fetchone()
    if (
        database is None
        or not str(database["file"])
        or Path(str(database["file"])).absolute() != binding.db_path
    ):
        raise StrategyError("model-evidence V2 binding database changed")
    task_id = _text(binding.task_id, "model-evidence binding.task_id")
    artifact_id = _hash(
        binding.artifact_id,
        "model-evidence binding.artifact_id",
    )
    artifact_content_hash = _hash(
        binding.artifact_content_hash,
        "model-evidence binding.artifact_content_hash",
    )
    if binding.sample_design_binding.task_id != task_id:
        raise StrategyError("model-evidence V2 sample belongs to another task")
    if not isinstance(binding.sources, tuple) or not isinstance(binding.warnings, tuple):
        raise StrategyError("model-evidence V2 binding is not immutable")
    if not all(isinstance(source, _CandidateSourceBinding) for source in binding.sources):
        raise StrategyError("model-evidence V2 source binding changed")
    provenance = _validate_provenance(binding.provenance)
    if _canonical_json(provenance) != _canonical_json(binding.provenance):
        raise StrategyError("model-evidence V2 binding provenance changed")
    request = _validate_inputs(
        {
            "sample_design_ref": provenance["sample_design_ref"],
            "univariate_sources": [source.request for source in binding.sources],
        }
    )
    if request["univariate_sources"] != provenance["univariate_sources"]:
        raise StrategyError("model-evidence V2 binding source requests changed")
    bundle = validate_strategy_model_evidence_bundle(
        binding.bundle,
        sample_design_bundle=binding.sample_design_binding.bundle,
    )
    if bundle != binding.bundle:
        raise StrategyError("model-evidence V2 binding bundle changed")
    expected_output_path = (
        binding.tasks_root
        / task_id
        / "strategy_model_evidence"
        / f"{bundle['bundle_id']}.json"
    )
    if (
        not binding.tasks_root.is_absolute()
        or binding.path != expected_output_path
    ):
        raise StrategyError("model-evidence V2 governed task root changed")
    canonical = canonical_strategy_model_evidence_bundle_json(
        bundle,
        sample_design_bundle=binding.sample_design_binding.bundle,
    ).encode("utf-8")
    if not hmac.compare_digest(_sha256(canonical), artifact_content_hash):
        raise StrategyError("model-evidence V2 binding artifact hash changed")
    _require_provenance_binding(
        provenance,
        task_id=task_id,
        request=request,
        sample_binding=binding.sample_design_binding,
        artifact_content_hash=artifact_content_hash,
        expected_bundle_id=bundle["bundle_id"],
        expected_bundle_content_hash=bundle["content_hash"],
    )
    if tuple(provenance["translation_warnings"]) != tuple(binding.warnings):
        raise StrategyError("model-evidence V2 binding warnings changed")

    require_strategy_sample_design_v2_artifact_binding_on_connection(
        conn,
        binding.sample_design_binding,
    )
    normalized_sources = {
        item["artifact_id"]: item for item in request["univariate_sources"]
    }
    for source in binding.sources:
        normalized_source = normalized_sources.get(source.artifact_id)
        if (
            source.task_id != task_id
            or normalized_source is None
            or source.request != normalized_source
            or source.dataset_source_path
            != binding.sample_design_binding.provenance["dataset_source_path"]
        ):
            raise StrategyError("model-evidence V2 source binding changed")
        _require_regular_path(source.path, root=binding.tasks_root)
        source_provenance = _validate_source_provenance(source.provenance)
        if (
            _canonical_json(source_provenance) != _canonical_json(source.provenance)
            or source.provenance_json != _canonical_json(source_provenance)
        ):
            raise StrategyError("model-evidence V2 source provenance changed")
        _require_candidate_binding(
            source.report["candidate_evidence"],
            provenance=source_provenance,
            request=normalized_source,
            task_id=task_id,
            sample_binding=binding.sample_design_binding,
        )
        source_canonical = canonical_strategy_candidate_report_json(
            source.report["candidate_evidence"],
            source.report["univariate_analysis"],
        )
        if source.canonical_bytes != len(source_canonical):
            raise StrategyError("model-evidence V2 source byte binding changed")
        _require_source_on_connection(conn, source)
        _require_exact_file(
            source.path,
            root=binding.tasks_root,
            canonical=source_canonical,
            content_hash=source.content_hash,
            maximum_bytes=_MAX_CANDIDATE_REPORT_BYTES,
        )

    rebuilt, warnings = _translate_sources(
        sample_binding=binding.sample_design_binding,
        sources=binding.sources,
    )
    if rebuilt != bundle or warnings != tuple(binding.warnings):
        raise StrategyError(
            "model-evidence V2 artifact no longer matches deterministic sources"
        )
    _require_output_on_connection(
        conn,
        task_id=task_id,
        artifact_id=artifact_id,
        path=binding.path,
        content_hash=artifact_content_hash,
        provenance=provenance,
    )
    _require_exact_file(
        binding.path,
        root=binding.tasks_root,
        canonical=canonical,
        content_hash=artifact_content_hash,
        maximum_bytes=MAX_MODEL_EVIDENCE_JSON_BYTES,
    )


def _validate_inputs(value: object) -> dict[str, Any]:
    _preflight_array_limit(
        value,
        field="univariate_sources",
        maximum=_MAX_UNIVARIATE_SOURCES,
        error=f"univariate_sources exceeds source budget ({_MAX_UNIVARIATE_SOURCES})",
    )
    obj = _json_object(value, "materialize_model_evidence_v2 inputs")
    _exact_fields(obj, _INPUT_FIELDS, "materialize_model_evidence_v2 inputs")
    sample = _json_object(obj["sample_design_ref"], "sample_design_ref")
    _exact_fields(sample, _SAMPLE_DESIGN_INPUT_FIELDS, "sample_design_ref")
    normalized_sample = {
        "membership_artifact_id": _hash(
            sample["membership_artifact_id"], "membership_artifact_id"
        ),
        "expected_membership_artifact_content_hash": _hash(
            sample["expected_membership_artifact_content_hash"],
            "expected_membership_artifact_content_hash",
        ),
        "bundle_artifact_id": _hash(
            sample["bundle_artifact_id"], "bundle_artifact_id"
        ),
        "expected_bundle_artifact_content_hash": _hash(
            sample["expected_bundle_artifact_content_hash"],
            "expected_bundle_artifact_content_hash",
        ),
        "expected_bundle_id": _text(
            sample["expected_bundle_id"], "expected_bundle_id"
        ),
        "expected_sample_design_id": _text(
            sample["expected_sample_design_id"], "expected_sample_design_id"
        ),
        "expected_sample_design_content_hash": _hash(
            sample["expected_sample_design_content_hash"],
            "expected_sample_design_content_hash",
        ),
    }
    raw_sources = _array(obj["univariate_sources"], "univariate_sources", required=True)
    if len(raw_sources) > _MAX_UNIVARIATE_SOURCES:
        raise StrategyError(
            "univariate_sources exceeds source budget "
            f"({_MAX_UNIVARIATE_SOURCES})"
        )
    sources: list[dict[str, str]] = []
    for index, raw in enumerate(raw_sources):
        item = _json_object(raw, f"univariate_sources[{index}]")
        _exact_fields(item, _SOURCE_INPUT_FIELDS, f"univariate_sources[{index}]")
        sources.append(
            {
                "artifact_id": _hash(item["artifact_id"], "source.artifact_id"),
                "expected_artifact_content_hash": _hash(
                    item["expected_artifact_content_hash"],
                    "source.expected_artifact_content_hash",
                ),
                "expected_candidate_id": _text(
                    item["expected_candidate_id"], "source.expected_candidate_id"
                ),
                "expected_evidence_hash": _hash(
                    item["expected_evidence_hash"],
                    "source.expected_evidence_hash",
                ),
            }
        )
    if len({item["artifact_id"] for item in sources}) != len(sources):
        raise StrategyError("univariate_sources contains duplicate artifact_id values")
    if len({item["expected_candidate_id"] for item in sources}) != len(sources):
        raise StrategyError("univariate_sources contains duplicate candidate_id values")
    sources.sort(key=lambda item: item["artifact_id"])
    request = {"sample_design_ref": normalized_sample, "univariate_sources": sources}
    _require_json_byte_budget(request, "materialize_model_evidence_v2 inputs")
    return request


def _load_sample_design(
    runtime, task_id: str, request: Mapping[str, Any]
) -> StrategySampleDesignV2ArtifactBinding:
    ref = request["sample_design_ref"]
    return load_strategy_sample_design_v2_artifacts(
        runtime,
        task_id=task_id,
        membership_artifact_id=ref["membership_artifact_id"],
        expected_membership_artifact_content_hash=ref[
            "expected_membership_artifact_content_hash"
        ],
        bundle_artifact_id=ref["bundle_artifact_id"],
        expected_bundle_artifact_content_hash=ref[
            "expected_bundle_artifact_content_hash"
        ],
        expected_bundle_id=ref["expected_bundle_id"],
        expected_sample_design_id=ref["expected_sample_design_id"],
        expected_sample_design_content_hash=ref[
            "expected_sample_design_content_hash"
        ],
    )


def _load_candidate_sources(
    runtime,
    *,
    task_id: str,
    requests: Sequence[Mapping[str, str]],
    sample_binding: StrategySampleDesignV2ArtifactBinding,
) -> tuple[_CandidateSourceBinding, ...]:
    result: list[_CandidateSourceBinding] = []
    consumed_bytes = 0
    for request in requests:
        remaining_bytes = _MAX_CANDIDATE_SOURCE_BYTES_TOTAL - consumed_bytes
        if remaining_bytes <= 0:
            raise StrategyError("cumulative candidate source byte budget exceeded")
        source = _load_candidate_source(
            runtime,
            task_id=task_id,
            request=request,
            sample_binding=sample_binding,
            maximum_bytes=remaining_bytes,
        )
        consumed_bytes += source.canonical_bytes
        result.append(source)
    return tuple(result)


def _load_candidate_source(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, str],
    sample_binding: StrategySampleDesignV2ArtifactBinding,
    maximum_bytes: int,
) -> _CandidateSourceBinding:
    record = runtime.task_artifacts.get_for_task(task_id, request["artifact_id"])
    if record is None or not isinstance(record, Mapping) or set(record) != _RECORD_FIELDS:
        raise StrategyError(
            f"source candidate artifact not found: {request['artifact_id']}"
        )
    if (
        record["id"] != request["artifact_id"]
        or record["task_id"] != task_id
        or record["kind"] != _SOURCE_ARTIFACT_KIND
        or record["origin_tool"] != _SOURCE_ORIGIN_TOOL
        or not _matches_hash(
            record["content_hash"], request["expected_artifact_content_hash"]
        )
    ):
        raise StrategyError("source candidate artifact registry binding changed")
    provenance = _validate_source_provenance(record["provenance"])
    path = (
        Path(runtime.settings.tasks_dir).absolute()
        / task_id
        / "strategy_candidates"
        / (
            f"{request['expected_candidate_id']}_"
            f"{request['expected_artifact_content_hash'][:12]}.json"
        )
    )
    if Path(str(record["path"])) != path:
        raise StrategyError("source candidate artifact path is not canonical")
    raw = _read_verified(
        path,
        root=Path(runtime.settings.tasks_dir),
        expected_hash=request["expected_artifact_content_hash"],
        maximum_bytes=min(_MAX_CANDIDATE_REPORT_BYTES, maximum_bytes),
        budget_error="cumulative candidate source byte budget exceeded",
    )
    try:
        report = strategy_candidate_report_from_json(raw)
    except (TypeError, ValueError, StrategyError) as exc:
        raise StrategyError("source candidate report failed strict validation") from exc
    evidence = validate_candidate_evidence(report["candidate_evidence"])
    canonical = canonical_strategy_candidate_report_json(
        evidence, report["univariate_analysis"]
    )
    if canonical != raw:
        raise StrategyError("source candidate report bytes are not canonical")
    _require_candidate_binding(
        evidence,
        provenance=provenance,
        request=request,
        task_id=task_id,
        sample_binding=sample_binding,
    )
    with runtime.task_artifacts.transaction() as conn:
        _require_candidate_dataset_on_connection(
            conn,
            provenance=provenance,
            task_id=task_id,
            expected_source_path=sample_binding.provenance["dataset_source_path"],
        )
    return _CandidateSourceBinding(
        task_id=task_id,
        artifact_id=request["artifact_id"],
        path=path,
        content_hash=request["expected_artifact_content_hash"],
        provenance=provenance,
        provenance_json=_canonical_json(provenance),
        report=report,
        request=dict(request),
        dataset_source_path=sample_binding.provenance["dataset_source_path"],
        canonical_bytes=len(raw),
    )


def _validate_source_provenance(value: object) -> dict[str, Any]:
    obj = _json_object(value, "source candidate artifact provenance")
    _exact_fields(obj, _SOURCE_PROVENANCE_FIELDS, "source candidate artifact provenance")
    if (
        obj["schema_version"] != _SOURCE_PROVENANCE_SCHEMA_VERSION
        or obj["producer_version"] != _SOURCE_PRODUCER_VERSION
        or obj["format"] != "json"
    ):
        raise StrategyError("source candidate artifact provenance contract is invalid")
    for field in (
        "evidence_hash",
        "dataset_content_hash",
        "registry_metadata_hash",
        "semantic_mapping_hash",
    ):
        _hash(obj[field], f"source provenance.{field}")
    for field in ("candidate_id", "dataset_id"):
        _text(obj[field], f"source provenance.{field}")
    _non_negative_int(obj["workspace_revision"], "source provenance.workspace_revision")
    _non_negative_int(
        obj["workspace_generation"], "source provenance.workspace_generation"
    )
    obj["generation_parameters"] = _json_object(
        obj["generation_parameters"], "source provenance.generation_parameters"
    )
    return obj


def _require_candidate_binding(
    evidence: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
    request: Mapping[str, str],
    task_id: str,
    sample_binding: StrategySampleDesignV2ArtifactBinding,
) -> None:
    if (
        evidence["candidate_id"] != request["expected_candidate_id"]
        or not hmac.compare_digest(
            evidence["evidence_hash"], request["expected_evidence_hash"]
        )
    ):
        raise StrategyError("source candidate identity does not match the request")
    identity = evidence["identity"]
    design = sample_binding.bundle["sample_design"]
    dataset_ref = design["identity"]["dataset_ref"]
    workspace_ref = design["identity"]["workspace_ref"]
    parameters = evidence["generation"]["parameters"]
    expected = {
        "candidate_id": evidence["candidate_id"],
        "evidence_hash": evidence["evidence_hash"],
        "dataset_id": identity["dataset_id"],
        "dataset_content_hash": identity["dataset_content_hash"],
        "workspace_revision": identity["workspace_revision"],
        "workspace_generation": identity["workspace_generation"],
        "semantic_mapping_hash": identity["semantic_mapping_hash"],
        "generation_parameters": parameters,
    }
    if any(
        _canonical_json(provenance[field]) != _canonical_json(value)
        for field, value in expected.items()
    ):
        raise StrategyError("source candidate artifact provenance changed")
    if identity["task_id"] != task_id:
        raise StrategyError("source candidate evidence belongs to another task")
    if (
        identity["dataset_id"] != dataset_ref["dataset_id"]
        or identity["dataset_content_hash"] != dataset_ref["content_hash"]
        or identity["workspace_revision"] != workspace_ref["revision"]
        or identity["workspace_generation"] != workspace_ref["generation"]
        or identity["semantic_mapping_hash"] != workspace_ref["semantic_mapping_hash"]
    ):
        raise StrategyError(
            "source candidate dataset/workspace does not match SampleDesign V2"
        )
    try:
        legacy_ref = StrategySampleDesignRef.from_value(
            parameters.get("sample_design_ref")
        ).to_ref_dict()
    except StrategyError as exc:
        raise StrategyError("source candidate legacy sample binding is invalid") from exc
    expected_legacy = design["compatibility"]["legacy_development_ref"]
    if legacy_ref != expected_legacy or design["compatibility"]["maps_to"] != "risk/development":
        raise StrategyError(
            "source candidate legacy sample binding does not equal V2 compatibility"
        )
    analysis = evidence["analysis"]
    _require_candidate_analysis_contract(
        analysis,
        parameters=parameters,
        generation=evidence["generation"],
        target_selector=design["target_selector"],
    )
    dropped = parameters.get("nan_labels_dropped")
    if isinstance(dropped, bool) or not isinstance(dropped, int) or dropped < 0:
        raise StrategyError("source candidate nan_labels_dropped is invalid")
    development_count = _risk_development_row_count(sample_binding.bundle)
    if analysis["row_count"] + dropped != development_count:
        raise StrategyError(
            "source candidate analyzed rows do not reconcile to risk/development"
        )
    if parameters.get("target_col") != analysis["target"]:
        raise StrategyError("source candidate target binding is inconsistent")
    maturity = _risk_maturity(sample_binding.bundle)
    if maturity["status"] == "confirmed_matured":
        statistics = _risk_development_statistics(sample_binding.bundle)
        if statistics.get("labeled_count") != analysis["row_count"]:
            raise StrategyError(
                "source candidate labeled rows do not match V2 risk/development"
            )
        for feature in analysis["features"]:
            for method in feature["methods"]:
                if method["status"] != "available":
                    continue
                bad_count = sum(int(item["bad"]) for item in method["bins"])
                if statistics.get("bad_count") != bad_count:
                    raise StrategyError(
                        "source candidate bad count does not match V2 risk/development"
                    )


def _require_candidate_analysis_contract(
    analysis: Mapping[str, Any],
    *,
    parameters: Mapping[str, Any],
    generation: Mapping[str, Any],
    target_selector: Mapping[str, Any],
) -> None:
    feature_names = [item["feature"] for item in analysis["features"]]
    feature_types = {
        item["feature"]: item["feature_type"] for item in analysis["features"]
    }
    expected = {
        "analysis_schema_version": analysis["schema_version"],
        "target_col": analysis["target"],
        "features": feature_names,
        "feature_types": feature_types,
        "bin_count": analysis["parameters"]["bin_count"],
        "min_bin_pct": analysis["parameters"]["min_bin_pct"],
        "loan_amount_col": analysis["parameters"]["loan_amount"],
        "overdue_amount_col": analysis["parameters"]["overdue_amount"],
    }
    for field, value in expected.items():
        if _canonical_json(parameters.get(field)) != _canonical_json(value):
            raise StrategyError(
                f"source candidate generation {field} does not match analysis"
            )
    if generation["seed"] != analysis["parameters"]["seed"]:
        raise StrategyError("source candidate generation seed does not match analysis")
    if generation["truncated"] != analysis["resource_budget"]["truncated"]:
        raise StrategyError(
            "source candidate generation truncation does not match analysis"
        )
    drop_nan = parameters.get("drop_nan_labels")
    if not isinstance(drop_nan, bool):
        raise StrategyError("source candidate drop_nan_labels is invalid")
    if (
        target_selector["status"] != "resolved"
        or target_selector["column"] != analysis["target"]
        or target_selector["drop_missing"] != drop_nan
    ):
        raise StrategyError(
            "source candidate target contract does not match SampleDesign V2"
        )
    sentinels = parameters.get("sentinel_values")
    if not isinstance(sentinels, Mapping) or any(
        not isinstance(key, str) for key in sentinels
    ):
        raise StrategyError("source candidate sentinel generation binding is invalid")
    if set(sentinels) - set(feature_names):
        raise StrategyError("source candidate sentinel binding names an unknown feature")
    for feature in analysis["features"]:
        configured = sentinels.get(feature["feature"], [])
        if _canonical_json(configured) != _canonical_json(feature["sentinel_values"]):
            raise StrategyError(
                "source candidate sentinel binding does not match analysis"
            )


def _translate_sources(
    *,
    sample_binding: StrategySampleDesignV2ArtifactBinding,
    sources: Sequence[_CandidateSourceBinding],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    evidence_items: list[dict[str, Any]] = []
    budget = _TranslationBudget()
    warnings: set[str] = set()
    for warning in (
        "Omitted model, monthly, and comparison evidence because no authenticated model or period lineage was supplied.",
        "Omitted candidate-only risk_direction, amount_metrics, cumulative_ks, rankings, and flattened metric dimensions because ModelEvidence V2 has no matching univariate contract.",
    ):
        budget.add_warning(warning)
        warnings.add(warning)
    for source in sources:
        budget.begin_source()
        translated, source_warnings = _translate_candidate_source(
            sample_binding.bundle, source, budget=budget
        )
        if not translated:
            raise StrategyError(
                f"source candidate contains no supported structured univariate evidence: {source.artifact_id}"
            )
        evidence_items.extend(translated)
        warnings.update(source_warnings)
    bundle = build_strategy_model_evidence_bundle(
        sample_design_bundle=sample_binding.bundle,
        univariate_evidence=evidence_items,
        model_evidence=(),
        comparison_evidence=(),
        producer_version=DEFAULT_PRODUCER_VERSION,
    )
    canonical = canonical_strategy_model_evidence_bundle_json(
        bundle, sample_design_bundle=sample_binding.bundle
    )
    roundtrip = strategy_model_evidence_bundle_from_json(
        canonical, sample_design_bundle=sample_binding.bundle
    )
    if roundtrip != bundle:
        raise StrategyError("model-evidence V2 canonical roundtrip is unstable")
    return bundle, tuple(_warnings(sorted(warnings)))


def _translate_candidate_source(
    sample_bundle: Mapping[str, Any],
    source: _CandidateSourceBinding,
    *,
    budget: _TranslationBudget,
) -> tuple[list[dict[str, Any]], list[str]]:
    analysis = source.report["univariate_analysis"]
    sample_count = int(analysis["row_count"])
    source_ref = build_evidence_source_ref(
        sample_design_bundle=sample_bundle,
        population="risk",
        partition="development",
        kind=_SOURCE_ARTIFACT_KIND,
        ref_id=source.artifact_id,
        content_hash=source.content_hash,
    )
    artifact_ref = build_artifact_ref(
        kind=_SOURCE_ARTIFACT_KIND,
        ref_id=source.artifact_id,
        content_hash=source.content_hash,
    )
    result: list[dict[str, Any]] = []
    warnings: list[str] = []
    for feature in analysis["features"]:
        feature_name = str(feature["feature"])
        sentinel_configured = bool(feature["sentinel_values"])
        for method in feature["methods"]:
            budget.begin_method()
            method_name = str(method["method"])
            if method["status"] != "available":
                reason = method["evidence"].get("kind", "unavailable")
                warning = (
                    "Omitted unavailable candidate method "
                    f"{source.report['candidate_evidence']['candidate_id']}:"
                    f"{feature_name}/{method_name} ({reason})."
                )
                budget.add_warning(warning)
                warnings.append(warning)
                continue
            budget.begin_available_method(bin_count=len(method["bins"]))
            _require_candidate_special_bins(feature, method)
            bin_ids: dict[str, str] = {}
            bins: list[dict[str, Any]] = []
            for item in method["bins"]:
                budget.begin_bin()
                translated_id = f"{method_name}:{item['id']}"
                bin_ids[item["id"]] = translated_id
                bins.append(
                    _translate_bin(
                        sample_bundle,
                        source_ref=source_ref,
                        categories_ref=artifact_ref,
                        bin_id=translated_id,
                        item=item,
                    )
                )
            observations: list[dict[str, Any]] = []
            metrics = method["metrics"]
            for metric_key, unit in (("iv", "number"), ("ks", "ratio"), ("auc", "ratio")):
                observations.append(
                    _outcome_observation(
                        sample_bundle,
                        source_ref=source_ref,
                        feature=feature_name,
                        metric_key=metric_key,
                        unit=unit,
                        value=metrics[metric_key],
                        sample_count=sample_count,
                    )
                )
            missing_count = sum(
                int(item["count"])
                for item in method["bins"]
                if item["kind"] == "missing"
            )
            if not hmac.compare_digest(
                _canonical_json(metrics["missing_rate"]),
                _canonical_json(missing_count / sample_count),
            ):
                raise StrategyError("candidate missing_rate does not reconcile to bins")
            observations.append(
                _present_observation(
                    sample_bundle,
                    source_ref=source_ref,
                    feature=feature_name,
                    metric_key="missing_rate",
                    unit="ratio",
                    value=metrics["missing_rate"],
                    numerator=missing_count,
                    denominator=sample_count,
                    sample_count=sample_count,
                )
            )
            sentinel_count = sum(
                int(item["count"])
                for item in method["bins"]
                if item["kind"] == "sentinel"
            )
            if sentinel_configured:
                observations.append(
                    _present_observation(
                        sample_bundle,
                        source_ref=source_ref,
                        feature=feature_name,
                        metric_key="sentinel_rate",
                        unit="ratio",
                        value=sentinel_count / sample_count,
                        numerator=sentinel_count,
                        denominator=sample_count,
                        sample_count=sample_count,
                    )
                )
            else:
                if sentinel_count:
                    raise StrategyError(
                        "candidate sentinel bins exist without configured sentinel values"
                    )
                observations.append(
                    _unavailable_observation(
                        sample_bundle,
                        source_ref=source_ref,
                        feature=feature_name,
                        metric_key="sentinel_rate",
                        unit="ratio",
                        reason="candidate source has no configured sentinel definition",
                    )
                )
            for item in method["bins"]:
                translated_id = bin_ids[item["id"]]
                observations.extend(
                    _translate_bin_observations(
                        sample_bundle,
                        source_ref=source_ref,
                        feature=feature_name,
                        bin_id=translated_id,
                        item=item,
                        sample_count=sample_count,
                    )
                )
            expected_observations = (
                _OBSERVATIONS_PER_TRANSLATED_METHOD
                + _OBSERVATIONS_PER_TRANSLATED_BIN * len(method["bins"])
            )
            if len(observations) != expected_observations:
                raise StrategyError(
                    "model-evidence translation observation accounting drifted"
                )
            evidence = build_univariate_evidence(
                sample_design_bundle=sample_bundle,
                population="risk",
                partition="development",
                analysis_ref=source_ref,
                analysis_variant=method_name,
                feature=feature_name,
                bins=bins,
                missing_treatment="separate_bin",
                sentinel_treatment=(
                    "separate_bin" if sentinel_configured else "not_configured"
                ),
                observations=observations,
            )
            budget.add_evidence_json(evidence)
            result.append(evidence)
    return result, warnings


def _require_candidate_special_bins(
    feature: Mapping[str, Any], method: Mapping[str, Any]
) -> None:
    missing_bins = [item for item in method["bins"] if item["kind"] == "missing"]
    if len(missing_bins) > 1 or any(int(item["count"]) <= 0 for item in missing_bins):
        raise StrategyError("candidate missing-bin structure is not producer-canonical")
    configured = list(feature["sentinel_values"])
    if feature["feature_type"] == "numeric":
        configured = [float(item) for item in configured]
    expected_keys = {_canonical_json(item) for item in configured}
    actual_keys = {
        _canonical_json(item["value"])
        for item in method["bins"]
        if item["kind"] == "sentinel"
    }
    if actual_keys != expected_keys:
        raise StrategyError(
            "candidate sentinel bins do not match configured sentinel definitions"
        )


def _translate_bin(
    sample_bundle: Mapping[str, Any],
    *,
    source_ref: Mapping[str, Any],
    categories_ref: Mapping[str, Any],
    bin_id: str,
    item: Mapping[str, Any],
) -> dict[str, Any]:
    kind_map = {
        "numeric_interval": "interval",
        "category": "category",
        "missing": "missing",
        "sentinel": "sentinel",
    }
    kind = kind_map.get(item["kind"])
    if kind is None:
        raise StrategyError(f"unsupported candidate bin kind: {item['kind']}")
    return build_univariate_bin_ref(
        sample_design_bundle=sample_bundle,
        population="risk",
        partition="development",
        ordinal=int(item["index"]),
        bin_id=bin_id,
        kind=kind,
        definition_ref=source_ref,
        lower_bound=item.get("lower") if kind == "interval" else None,
        upper_bound=item.get("upper") if kind == "interval" else None,
        lower_inclusive=item.get("include_lower") if kind == "interval" else None,
        upper_inclusive=item.get("include_upper") if kind == "interval" else None,
        categories_ref=categories_ref if kind == "category" else None,
    )


def _translate_bin_observations(
    sample_bundle: Mapping[str, Any],
    *,
    source_ref: Mapping[str, Any],
    feature: str,
    bin_id: str,
    item: Mapping[str, Any],
    sample_count: int,
) -> list[dict[str, Any]]:
    result = [
        _present_observation(
            sample_bundle,
            source_ref=source_ref,
            feature=feature,
            bin_id=bin_id,
            metric_key="bin_count",
            unit="count",
            value=item["count"],
            sample_count=sample_count,
        ),
        _present_observation(
            sample_bundle,
            source_ref=source_ref,
            feature=feature,
            bin_id=bin_id,
            metric_key="bin_share",
            unit="ratio",
            value=item["share"],
            numerator=item["count"],
            denominator=sample_count,
            sample_count=sample_count,
        ),
    ]
    for metric_key, field, unit in (
        ("bin_good_count", "good", "count"),
        ("bin_bad_count", "bad", "count"),
        ("bin_woe", "woe", "number"),
        ("bin_iv", "iv_contribution", "number"),
    ):
        result.append(
            _outcome_observation(
                sample_bundle,
                source_ref=source_ref,
                feature=feature,
                bin_id=bin_id,
                metric_key=metric_key,
                unit=unit,
                value=item[field],
                sample_count=sample_count,
            )
        )
    if item["bad_rate"] is None:
        result.append(
            _unavailable_observation(
                sample_bundle,
                source_ref=source_ref,
                feature=feature,
                bin_id=bin_id,
                metric_key="bin_bad_rate",
                unit="ratio",
                reason="candidate source has no bad rate for an empty bin",
                outcome=True,
            )
        )
    else:
        result.append(
            _outcome_observation(
                sample_bundle,
                source_ref=source_ref,
                feature=feature,
                bin_id=bin_id,
                metric_key="bin_bad_rate",
                unit="ratio",
                value=item["bad_rate"],
                numerator=item["bad"],
                denominator=item["count"],
                sample_count=sample_count,
            )
        )
    if item["lift"] is None:
        result.append(
            _unavailable_observation(
                sample_bundle,
                source_ref=source_ref,
                feature=feature,
                bin_id=bin_id,
                metric_key="lift",
                unit="multiple",
                reason="candidate source has no lift for an empty bin",
                outcome=True,
            )
        )
    else:
        result.append(
            _outcome_observation(
                sample_bundle,
                source_ref=source_ref,
                feature=feature,
                bin_id=bin_id,
                metric_key="lift",
                unit="multiple",
                value=item["lift"],
                sample_count=sample_count,
            )
        )
    return result


def _outcome_observation(
    sample_bundle: Mapping[str, Any],
    *,
    source_ref: Mapping[str, Any],
    feature: str,
    metric_key: str,
    unit: str,
    value: int | float,
    sample_count: int,
    bin_id: str | None = None,
    numerator: int | float | None = None,
    denominator: int | float | None = None,
) -> dict[str, Any]:
    maturity = _risk_maturity(sample_bundle)
    if maturity["status"] == "confirmed_matured":
        return _present_observation(
            sample_bundle,
            source_ref=source_ref,
            feature=feature,
            metric_key=metric_key,
            unit=unit,
            value=value,
            numerator=numerator,
            denominator=denominator,
            sample_count=sample_count,
            bin_id=bin_id,
        )
    status = "not_matured" if maturity["status"] == "not_matured" else "unavailable"
    reason = maturity["reason"] or "risk/development maturity is unavailable"
    return build_univariate_observation(
        sample_design_bundle=sample_bundle,
        population="risk",
        partition="development",
        metric_key=metric_key,
        status=status,
        value=None,
        numerator=None,
        denominator=None,
        sample_count=None,
        unit=unit,
        source_ref=source_ref,
        feature=feature,
        bin_id=bin_id,
        reason=reason,
    )


def _present_observation(
    sample_bundle: Mapping[str, Any],
    *,
    source_ref: Mapping[str, Any],
    feature: str,
    metric_key: str,
    unit: str,
    value: int | float,
    sample_count: int,
    bin_id: str | None = None,
    numerator: int | float | None = None,
    denominator: int | float | None = None,
) -> dict[str, Any]:
    return build_univariate_observation(
        sample_design_bundle=sample_bundle,
        population="risk",
        partition="development",
        metric_key=metric_key,
        status="present",
        value=value,
        numerator=numerator,
        denominator=denominator,
        sample_count=sample_count,
        unit=unit,
        source_ref=source_ref,
        feature=feature,
        bin_id=bin_id,
        reason=None,
    )


def _unavailable_observation(
    sample_bundle: Mapping[str, Any],
    *,
    source_ref: Mapping[str, Any],
    feature: str,
    metric_key: str,
    unit: str,
    reason: str,
    bin_id: str | None = None,
    outcome: bool = False,
) -> dict[str, Any]:
    if outcome:
        maturity = _risk_maturity(sample_bundle)
        if maturity["status"] == "not_matured":
            status = "not_matured"
            reason = maturity["reason"]
        else:
            status = "unavailable"
    else:
        status = "unavailable"
    return build_univariate_observation(
        sample_design_bundle=sample_bundle,
        population="risk",
        partition="development",
        metric_key=metric_key,
        status=status,
        value=None,
        numerator=None,
        denominator=None,
        sample_count=None,
        unit=unit,
        source_ref=source_ref,
        feature=feature,
        bin_id=bin_id,
        reason=reason,
    )


def _persist_bundle(
    runtime,
    *,
    task_id: str,
    request: Mapping[str, Any],
    sample_binding: StrategySampleDesignV2ArtifactBinding,
    sources: Sequence[_CandidateSourceBinding],
    bundle: Mapping[str, Any],
    warnings: Sequence[str],
) -> dict[str, Any]:
    canonical = canonical_strategy_model_evidence_bundle_json(
        bundle, sample_design_bundle=sample_binding.bundle
    ).encode("utf-8")
    content_hash = _sha256(canonical)
    out_dir = _prepare_output_directory(runtime.settings.tasks_dir, task_id=task_id)
    path = out_dir / f"{bundle['bundle_id']}.json"
    provenance = _artifact_provenance(
        request=request,
        sample_binding=sample_binding,
        bundle=bundle,
        artifact_content_hash=content_hash,
        warnings=warnings,
    )
    uow = ArtifactUnitOfWork()
    staged = uow.stage_file(out_dir, path.name)
    try:
        staged.path.write_bytes(canonical)
    except OSError as exc:
        uow.rollback()
        raise StrategyError("model-evidence V2 artifact could not be staged") from exc
    db_committed = False
    rollback_attempted_under_lock = False
    try:
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                require_strategy_sample_design_v2_artifact_binding_on_connection(
                    conn, sample_binding
                )
                for source in sources:
                    _require_source_on_connection(conn, source)
                    _require_exact_file(
                        source.path,
                        root=Path(runtime.settings.tasks_dir),
                        canonical=canonical_strategy_candidate_report_json(
                            source.report["candidate_evidence"],
                            source.report["univariate_analysis"],
                        ),
                        content_hash=source.content_hash,
                        maximum_bytes=_MAX_CANDIDATE_REPORT_BYTES,
                    )
                row = _select_output_row(
                    conn, task_id=task_id, kind=MODEL_EVIDENCE_V2_ARTIFACT_KIND, path=path
                )
                if row is not None:
                    _require_existing_output_row(
                        row,
                        task_id=task_id,
                        path=path,
                        content_hash=content_hash,
                        provenance=provenance,
                    )
                    _require_exact_file(
                        path,
                        root=Path(runtime.settings.tasks_dir),
                        canonical=canonical,
                        content_hash=content_hash,
                        maximum_bytes=MAX_MODEL_EVIDENCE_JSON_BYTES,
                    )
                    staged.rollback()
                elif path.exists() or path.is_symlink():
                    _require_exact_file(
                        path,
                        root=Path(runtime.settings.tasks_dir),
                        canonical=canonical,
                        content_hash=content_hash,
                        maximum_bytes=MAX_MODEL_EVIDENCE_JSON_BYTES,
                    )
                    staged.rollback()
                else:
                    staged.promote()
                    _require_exact_file(
                        path,
                        root=Path(runtime.settings.tasks_dir),
                        canonical=canonical,
                        content_hash=content_hash,
                        maximum_bytes=MAX_MODEL_EVIDENCE_JSON_BYTES,
                    )
                record = runtime.task_artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=MODEL_EVIDENCE_V2_ARTIFACT_KIND,
                    path=str(path),
                    content_hash=content_hash,
                    origin_tool=MODEL_EVIDENCE_V2_ORIGIN_TOOL,
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
    return validate_materialize_model_evidence_v2_tool_output(
        _tool_output(
            sample_bundle=sample_binding.bundle,
            bundle=bundle,
            record=record,
            sources=sources,
        )
    )


def _artifact_provenance(
    *,
    request: Mapping[str, Any],
    sample_binding: StrategySampleDesignV2ArtifactBinding,
    bundle: Mapping[str, Any],
    artifact_content_hash: str,
    warnings: Sequence[str],
) -> dict[str, Any]:
    design = sample_binding.bundle["sample_design"]
    header = sample_binding.membership["header"]
    provenance = {
        "schema_version": MODEL_EVIDENCE_V2_ARTIFACT_SCHEMA_VERSION,
        "producer_version": DEFAULT_PRODUCER_VERSION,
        "format": "json",
        "task_id": sample_binding.task_id,
        "bundle_id": bundle["bundle_id"],
        "bundle_content_hash": bundle["content_hash"],
        "bundle_artifact_content_hash": artifact_content_hash,
        "sample_design_bundle_content_hash": sample_binding.bundle["content_hash"],
        "sample_design_ref": dict(request["sample_design_ref"]),
        "membership_id": header["membership_id"],
        "membership_content_hash": header["content_hash"],
        "dataset_ref": dict(design["identity"]["dataset_ref"]),
        "dataset_source_path": sample_binding.provenance["dataset_source_path"],
        "dataset_registry_metadata_hash": sample_binding.provenance[
            "dataset_registry_metadata_hash"
        ],
        "workspace_ref": dict(design["identity"]["workspace_ref"]),
        "legacy_sample_design_ref": dict(
            design["compatibility"]["legacy_development_ref"]
        ),
        "univariate_sources": [dict(item) for item in request["univariate_sources"]],
        "request_hash": _sha256(_canonical_json(request).encode("utf-8")),
        "translation_warnings": list(warnings),
    }
    _require_json_byte_budget(provenance, "model-evidence V2 artifact provenance")
    return provenance


def _tool_output(
    *,
    sample_bundle: Mapping[str, Any],
    bundle: Mapping[str, Any],
    record: Mapping[str, Any],
    sources: Sequence[_CandidateSourceBinding],
) -> dict[str, Any]:
    design = sample_bundle["sample_design"]
    artifact_hash = str(record["content_hash"])
    body = {
        "schema_version": MODEL_EVIDENCE_V2_TOOL_SCHEMA_VERSION,
        "bundle_id": bundle["bundle_id"],
        "bundle_content_hash": bundle["content_hash"],
        "sample_design_bundle_id": sample_bundle["bundle_id"],
        "sample_design_id": design["sample_design_id"],
        "sample_design_content_hash": design["content_hash"],
        "sample_design_bundle": dict(sample_bundle),
        "bundle": dict(bundle),
        "artifact": {
            "kind": MODEL_EVIDENCE_V2_ARTIFACT_KIND,
            "format": "json",
            "filename": Path(str(record["path"])).name,
            "content_hash": artifact_hash,
        },
        "source_artifacts": [
            {
                "artifact_id": item.artifact_id,
                "kind": _SOURCE_ARTIFACT_KIND,
                "content_hash": item.content_hash,
            }
            for item in sources
        ],
        "univariate_only": True,
        "not_created_model": True,
        "not_compared_models": True,
        "not_adopted": True,
        "not_deployed": True,
    }
    return {
        **body,
        "content_hash": _sha256(_canonical_json(body).encode("utf-8")),
    }


def _provenance_sources_for_artifact(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_content_hash: str,
    expected_bundle_id: str,
) -> list[dict[str, Any]]:
    record = _registered_output_record(
        runtime,
        task_id=task_id,
        artifact_id=artifact_id,
        expected_content_hash=expected_content_hash,
    )
    provenance = _validate_provenance(record["provenance"])
    if provenance["bundle_id"] != expected_bundle_id:
        raise StrategyError("model-evidence V2 artifact bundle_id changed")
    return list(provenance["univariate_sources"])


def _validate_provenance(value: object) -> dict[str, Any]:
    obj = _json_object(value, "model-evidence V2 artifact provenance")
    _require_json_byte_budget(obj, "model-evidence V2 artifact provenance")
    _exact_fields(obj, _PROVENANCE_FIELDS, "model-evidence V2 artifact provenance")
    if (
        obj["schema_version"] != MODEL_EVIDENCE_V2_ARTIFACT_SCHEMA_VERSION
        or obj["producer_version"] != DEFAULT_PRODUCER_VERSION
        or obj["format"] != "json"
    ):
        raise StrategyError("model-evidence V2 artifact provenance contract is invalid")
    _text(obj["task_id"], "provenance.task_id")
    for field in ("bundle_id", "membership_id", "dataset_source_path"):
        _text(obj[field], f"provenance.{field}")
    for field in (
        "bundle_content_hash",
        "bundle_artifact_content_hash",
        "sample_design_bundle_content_hash",
        "membership_content_hash",
        "dataset_registry_metadata_hash",
        "request_hash",
    ):
        _hash(obj[field], f"provenance.{field}")
    request = _validate_inputs(
        {
            "sample_design_ref": obj["sample_design_ref"],
            "univariate_sources": obj["univariate_sources"],
        }
    )
    if not hmac.compare_digest(
        obj["request_hash"], _sha256(_canonical_json(request).encode("utf-8"))
    ):
        raise StrategyError("model-evidence V2 provenance request_hash changed")
    obj["sample_design_ref"] = request["sample_design_ref"]
    obj["univariate_sources"] = request["univariate_sources"]
    obj["dataset_ref"] = _json_object(obj["dataset_ref"], "provenance.dataset_ref")
    obj["workspace_ref"] = _json_object(
        obj["workspace_ref"], "provenance.workspace_ref"
    )
    obj["legacy_sample_design_ref"] = StrategySampleDesignRef.from_value(
        obj["legacy_sample_design_ref"]
    ).to_ref_dict()
    obj["translation_warnings"] = _warnings(obj["translation_warnings"])
    return obj


def _require_provenance_binding(
    provenance: Mapping[str, Any],
    *,
    task_id: str,
    request: Mapping[str, Any],
    sample_binding: StrategySampleDesignV2ArtifactBinding,
    artifact_content_hash: str,
    expected_bundle_id: str,
    expected_bundle_content_hash: str,
) -> None:
    design = sample_binding.bundle["sample_design"]
    header = sample_binding.membership["header"]
    expected = {
        "task_id": task_id,
        "bundle_id": expected_bundle_id,
        "bundle_content_hash": expected_bundle_content_hash,
        "bundle_artifact_content_hash": artifact_content_hash,
        "sample_design_bundle_content_hash": sample_binding.bundle["content_hash"],
        "sample_design_ref": request["sample_design_ref"],
        "membership_id": header["membership_id"],
        "membership_content_hash": header["content_hash"],
        "dataset_ref": design["identity"]["dataset_ref"],
        "dataset_source_path": sample_binding.provenance["dataset_source_path"],
        "dataset_registry_metadata_hash": sample_binding.provenance[
            "dataset_registry_metadata_hash"
        ],
        "workspace_ref": design["identity"]["workspace_ref"],
        "legacy_sample_design_ref": design["compatibility"][
            "legacy_development_ref"
        ],
        "univariate_sources": request["univariate_sources"],
    }
    for field, value in expected.items():
        if _canonical_json(provenance[field]) != _canonical_json(value):
            raise StrategyError(f"model-evidence V2 provenance {field} changed")


def _registered_output_record(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_content_hash: str,
) -> dict[str, Any]:
    record = runtime.task_artifacts.get_for_task(task_id, artifact_id)
    if record is None or not isinstance(record, Mapping) or set(record) != _RECORD_FIELDS:
        raise StrategyError("model-evidence V2 artifact registry row is invalid")
    if (
        record["id"] != artifact_id
        or record["task_id"] != task_id
        or record["kind"] != MODEL_EVIDENCE_V2_ARTIFACT_KIND
        or record["origin_tool"] != MODEL_EVIDENCE_V2_ORIGIN_TOOL
        or not _matches_hash(record["content_hash"], expected_content_hash)
    ):
        raise StrategyError("model-evidence V2 artifact registry binding changed")
    return dict(record)


def _require_source_on_connection(conn, source: _CandidateSourceBinding) -> None:
    row = conn.execute(
        "SELECT * FROM task_artifacts WHERE task_id = ? AND id = ?",
        (source.task_id, source.artifact_id),
    ).fetchone()
    if row is None:
        raise StrategyError("source candidate artifact disappeared before write")
    if (
        str(row["kind"]) != _SOURCE_ARTIFACT_KIND
        or str(row["path"]) != str(source.path)
        or str(row["content_hash"]) != source.content_hash
        or str(row["origin_tool"]) != _SOURCE_ORIGIN_TOOL
        or str(row["provenance_json"]) != source.provenance_json
    ):
        raise StrategyError("source candidate artifact registry binding changed")
    _require_candidate_dataset_on_connection(
        conn,
        provenance=source.provenance,
        task_id=source.task_id,
        expected_source_path=source.dataset_source_path,
    )


def _require_candidate_dataset_on_connection(
    conn,
    *,
    provenance: Mapping[str, Any],
    task_id: str,
    expected_source_path: str,
) -> None:
    row = conn.execute(
        """
        SELECT task_id, role, row_count, columns_json, has_target, target_col,
               content_hash, source_path
          FROM datasets
         WHERE id = ?
        """,
        (provenance["dataset_id"],),
    ).fetchone()
    if row is None or str(row["task_id"]) != task_id:
        raise StrategyError("source candidate dataset is not task-owned")
    if not _matches_hash(row["content_hash"], provenance["dataset_content_hash"]):
        raise StrategyError("source candidate registered dataset hash changed")
    columns_json = row["columns_json"]
    if not isinstance(columns_json, str):
        raise StrategyError("source candidate dataset schema is invalid")
    try:
        json.loads(columns_json)
    except json.JSONDecodeError as exc:
        raise StrategyError("source candidate dataset schema is invalid") from exc
    payload = {
        "role": str(row["role"]),
        "row_count": int(row["row_count"]),
        "columns_json": columns_json,
        "has_target": int(row["has_target"]),
        "target_col": row["target_col"],
    }
    metadata_hash = _sha256(_canonical_json(payload).encode("utf-8"))
    if not hmac.compare_digest(metadata_hash, provenance["registry_metadata_hash"]):
        raise StrategyError("source candidate dataset registry metadata changed")
    if str(row["source_path"]) != expected_source_path:
        raise StrategyError("source candidate dataset registry path changed")


def _require_output_on_connection(
    conn,
    *,
    task_id: str,
    artifact_id: str,
    path: Path,
    content_hash: str,
    provenance: Mapping[str, Any],
) -> None:
    row = conn.execute(
        "SELECT * FROM task_artifacts WHERE task_id = ? AND id = ?",
        (task_id, artifact_id),
    ).fetchone()
    if row is None:
        raise StrategyError("model-evidence V2 artifact disappeared")
    _require_existing_output_row(
        row,
        task_id=task_id,
        path=path,
        content_hash=content_hash,
        provenance=provenance,
    )


def _select_output_row(conn, *, task_id: str, kind: str, path: Path):
    return conn.execute(
        "SELECT * FROM task_artifacts WHERE task_id = ? AND kind = ? AND path = ?",
        (task_id, kind, str(path)),
    ).fetchone()


def _require_existing_output_row(
    row,
    *,
    task_id: str,
    path: Path,
    content_hash: str,
    provenance: Mapping[str, Any],
) -> None:
    expected = {
        "task_id": task_id,
        "kind": MODEL_EVIDENCE_V2_ARTIFACT_KIND,
        "path": str(path),
        "content_hash": content_hash,
        "origin_tool": MODEL_EVIDENCE_V2_ORIGIN_TOOL,
        "provenance_json": _canonical_json(provenance),
    }
    if any(str(row[field]) != value for field, value in expected.items()):
        raise StrategyError("existing model-evidence V2 artifact registry row changed")


def _risk_development_row_count(bundle: Mapping[str, Any]) -> int:
    for population in bundle["populations"]:
        if population["role"] != "risk":
            continue
        for partition in population["partitions"]:
            if partition["name"] == "development":
                return int(partition["row_count"])
    raise StrategyError("SampleDesign V2 lacks risk/development")


def _risk_maturity(bundle: Mapping[str, Any]) -> Mapping[str, Any]:
    for population in bundle["populations"]:
        if population["role"] == "risk":
            return population["maturity_evidence"]
    raise StrategyError("SampleDesign V2 lacks risk maturity evidence")


def _risk_development_statistics(bundle: Mapping[str, Any]) -> dict[str, int]:
    definitions = {
        item["metric_definition_id"]: item["metric_key"]
        for item in bundle["metric_definitions"]
    }
    result: dict[str, int] = {}
    for item in bundle["metric_observations"]:
        if item["population"] != "risk" or item["partition"] != "development":
            continue
        key = definitions[item["metric_definition_ref"]["metric_definition_id"]]
        if key in {"labeled_count", "bad_count"} and item["status"] == "present":
            result[key] = int(item["value"])
    return result


def _prepare_output_directory(tasks_dir: Path | str, *, task_id: str) -> Path:
    if Path(task_id).name != task_id or task_id in {".", ".."}:
        raise StrategyError("task_id cannot escape task storage")
    root = Path(tasks_dir).absolute()
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise StrategyError("task artifact root must be a regular directory")
    root.mkdir(parents=True, exist_ok=True)
    task_dir = root / task_id
    if task_dir.exists() and (task_dir.is_symlink() or not task_dir.is_dir()):
        raise StrategyError("task artifact directory must be a regular directory")
    task_dir.mkdir(exist_ok=True)
    if task_dir.is_symlink() or task_dir.resolve(strict=True).parent != root.resolve(strict=True):
        raise StrategyError("model-evidence V2 task directory escaped storage")
    out_dir = task_dir / "strategy_model_evidence"
    if out_dir.exists() and (out_dir.is_symlink() or not out_dir.is_dir()):
        raise StrategyError("model-evidence V2 output path must be a regular directory")
    out_dir.mkdir(exist_ok=True)
    if out_dir.is_symlink() or out_dir.resolve(strict=True).parent != task_dir.resolve(strict=True):
        raise StrategyError("model-evidence V2 output directory escaped storage")
    return out_dir


def _expected_output_path(runtime, *, task_id: str, bundle_id: str) -> Path:
    if Path(task_id).name != task_id or Path(bundle_id).name != bundle_id:
        raise StrategyError("model-evidence V2 artifact identity is not path-safe")
    return (
        Path(runtime.settings.tasks_dir).absolute()
        / task_id
        / "strategy_model_evidence"
        / f"{bundle_id}.json"
    )


def _read_verified(
    path: Path,
    *,
    root: Path,
    expected_hash: str,
    maximum_bytes: int,
    budget_error: str = "artifact exceeds byte budget",
) -> bytes:
    _require_regular_path(path, root=root)
    descriptor = -1
    try:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise StrategyError("artifact path is not a regular file")
        if before.st_size < 0 or before.st_size > maximum_bytes:
            raise StrategyError(budget_error)
        chunks: list[bytes] = []
        total = 0
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum_bytes:
                raise StrategyError(budget_error)
            digest.update(chunk)
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise StrategyError("artifact changed while being read")
        live_path = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(live_path.st_mode)
            or (live_path.st_dev, live_path.st_ino)
            != (after.st_dev, after.st_ino)
        ):
            raise StrategyError("artifact registry path changed while being read")
    except OSError as exc:
        raise StrategyError("artifact could not be read") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    raw = b"".join(chunks)
    if (
        len(raw) != before.st_size
        or not hmac.compare_digest(digest.hexdigest(), expected_hash)
    ):
        raise StrategyError("artifact content hash drifted")
    return raw


def _require_exact_file(
    path: Path,
    *,
    root: Path,
    canonical: bytes,
    content_hash: str,
    maximum_bytes: int,
) -> None:
    raw = _read_verified(
        path,
        root=root,
        expected_hash=content_hash,
        maximum_bytes=maximum_bytes,
    )
    if raw != canonical:
        raise StrategyError("artifact bytes are not canonical")


def _require_regular_path(path: Path, *, root: Path) -> None:
    absolute_root = root.absolute()
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise StrategyError("artifact path is not a regular file")
    try:
        path.relative_to(absolute_root)
    except ValueError as exc:
        raise StrategyError("artifact path escapes task storage") from exc
    current = path.parent
    while True:
        if current.is_symlink():
            raise StrategyError("artifact path uses a symlink")
        if current == absolute_root:
            break
        if current == current.parent:
            raise StrategyError("artifact path escapes task storage")
        current = current.parent


def _validate_output_artifact(
    value: object,
    *,
    bundle_id: str,
    expected_content_hash: str,
) -> dict[str, str]:
    obj = _json_object(value, "model-evidence V2 artifact output")
    _exact_fields(obj, _ARTIFACT_OUTPUT_FIELDS, "model-evidence V2 artifact output")
    expected = {
        "kind": MODEL_EVIDENCE_V2_ARTIFACT_KIND,
        "format": "json",
        "filename": f"{bundle_id}.json",
        "content_hash": expected_content_hash,
    }
    if obj != expected:
        raise StrategyError("model-evidence V2 artifact output drifted")
    return expected


def _validate_source_outputs(value: object) -> list[dict[str, str]]:
    raw = _array(value, "source_artifacts", required=True)
    if len(raw) > _MAX_UNIVARIATE_SOURCES:
        raise StrategyError("model-evidence V2 source summaries exceed source budget")
    result: list[dict[str, str]] = []
    for index, item in enumerate(raw):
        obj = _json_object(item, f"source_artifacts[{index}]")
        _exact_fields(obj, _SOURCE_OUTPUT_FIELDS, f"source_artifacts[{index}]")
        if obj["kind"] != _SOURCE_ARTIFACT_KIND:
            raise StrategyError("model-evidence V2 source kind is invalid")
        result.append(
            {
                "artifact_id": _hash(obj["artifact_id"], "source.artifact_id"),
                "kind": _SOURCE_ARTIFACT_KIND,
                "content_hash": _hash(obj["content_hash"], "source.content_hash"),
            }
        )
    if len({item["artifact_id"] for item in result}) != len(result):
        raise StrategyError("model-evidence V2 source outputs contain duplicates")
    return sorted(result, key=lambda item: item["artifact_id"])


def _warnings(value: object) -> list[str]:
    raw = _array(value, "warnings", required=False)
    if len(raw) > _MAX_TRANSLATION_WARNINGS:
        raise StrategyError("model-evidence V2 warnings exceed item budget")
    warnings = [_text(item, "warnings[]") for item in raw]
    if warnings != sorted(set(warnings)):
        raise StrategyError("model-evidence V2 warnings must be sorted and unique")
    _require_json_byte_budget(warnings, "model-evidence V2 warnings")
    return warnings


def _require_json_byte_budget(value: object, name: str) -> None:
    if len(_canonical_json(value).encode("utf-8")) > MAX_MODEL_EVIDENCE_JSON_BYTES:
        raise StrategyError(f"{name} exceeds JSON byte budget")


def _preflight_array_limit(
    value: object,
    *,
    field: str,
    maximum: int,
    error: str,
) -> None:
    if not isinstance(value, Mapping):
        return
    raw = value.get(field)
    if (
        not isinstance(raw, (str, bytes, bytearray))
        and isinstance(raw, Sequence)
        and len(raw) > maximum
    ):
        raise StrategyError(error)


def _json_object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StrategyError(f"{name} must be an object")
    try:
        normalized = json.loads(_canonical_json(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StrategyError(f"{name} must be a finite JSON object") from exc
    if not isinstance(normalized, dict):
        raise StrategyError(f"{name} must be an object")
    return normalized


def _array(value: object, name: str, *, required: bool) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise StrategyError(f"{name} must be an array")
    result = list(value)
    if required and not result:
        raise StrategyError(f"{name} must not be empty")
    return result


def _exact_fields(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    if any(not isinstance(key, str) for key in value):
        raise StrategyError(f"{name} keys must be strings")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unknown:
            details.append("unsupported: " + ", ".join(unknown))
        raise StrategyError(f"{name} fields are invalid ({'; '.join(details)})")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip() or "\x00" in value:
        raise StrategyError(f"{name} must be non-empty canonical text")
    return value


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise StrategyError(f"{name} must be a lowercase SHA-256 hash")
    return value


def _matches_hash(value: object, expected: str) -> bool:
    return (
        isinstance(value, str)
        and _SHA256_RE.fullmatch(value) is not None
        and hmac.compare_digest(value, expected)
    )


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StrategyError(f"{name} must be a non-negative integer")
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


__all__ = [
    "MODEL_EVIDENCE_V2_ARTIFACT_KIND",
    "MODEL_EVIDENCE_V2_ARTIFACT_SCHEMA_VERSION",
    "MODEL_EVIDENCE_V2_ORIGIN_TOOL",
    "MODEL_EVIDENCE_V2_TOOL_SCHEMA_VERSION",
    "StrategyModelEvidenceV2ArtifactBinding",
    "load_strategy_model_evidence_v2_artifact",
    "require_strategy_model_evidence_v2_artifact_binding_on_connection",
    "run_materialize_model_evidence_v2",
    "validate_materialize_model_evidence_v2_tool_output",
]
