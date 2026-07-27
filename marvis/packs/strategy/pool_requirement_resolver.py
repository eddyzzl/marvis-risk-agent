"""Resolve governed Strategy Pool execution requirements.

The resolver is the single bridge between a compiled Strategy Pool and
task-owned model-score vectors.  Consumers receive authenticated bindings and
an in-memory virtual field; no consumer reads score Parquet directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
import re
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from marvis.db import ModelingRepository
from marvis.packs.modeling.errors import ModelingError
from marvis.packs.modeling.evidence import RAW_SCORE_PRODUCT
from marvis.packs.modeling.experiment import ExperimentStore
from marvis.packs.modeling.score_evidence_tools import (
    ModelScoreEvidenceArtifactBinding,
    load_historical_model_score_evidence_artifacts,
    load_model_score_evidence_artifacts,
    require_historical_model_score_evidence_artifact_binding_on_connection,
    require_model_score_evidence_artifact_binding_on_connection,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.sample_design_v2_tools import (
    StrategySampleDesignV2ArtifactBinding,
    resolve_strategy_sample_design_v2_source_mode,
)
from marvis.packs.strategy.sample_design_v2_native_tools import (
    StrategySampleDesignV2NativeArtifactBinding,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MODEL_SCORE_VIRTUAL_FIELD_PREFIX = "__marvis_model_pd_"
_OUTER_REQUIREMENT_FIELDS = frozenset({"rule_id", "fragment_id", "requirement"})
_REQUIREMENT_LINEAGE_ENVELOPE_FIELDS = frozenset(
    {"entry_id", "rule_id", "fragment_id", "requirement"}
)
_MAX_REQUIREMENT_LINEAGE_DEPTH = 8
_MODEL_SCORE_REQUIREMENT_FIELDS = frozenset(
    {
        "type",
        "virtual_field",
        "score_product",
        "score_evidence_artifact_id",
        "score_evidence_artifact_content_hash",
        "score_vector_artifact_id",
        "score_vector_artifact_content_hash",
    }
)
_MODEL_SCORE_REQUIREMENT_TYPE = "model_score_vector.v1"
_REQUIREMENT_BINDINGS_FIELDS = frozenset(
    {
        "requirements_hash",
        "requirements",
        "virtual_fields",
    }
)
StrategySampleDesignV2Binding = (
    StrategySampleDesignV2ArtifactBinding
    | StrategySampleDesignV2NativeArtifactBinding
)


@dataclass(frozen=True, slots=True)
class ResolvedPoolRequirements:
    """Canonical requirements and their authenticated in-memory score bindings."""

    task_id: str
    requirements_hash: str
    requirements: tuple[dict[str, Any], ...]
    field_bindings: tuple[
        tuple[str, ModelScoreEvidenceArtifactBinding],
        ...,
    ]

    @property
    def virtual_fields(self) -> tuple[str, ...]:
        """Return virtual fields in deterministic compiled-requirement order."""

        return tuple(field for field, _binding in self.field_bindings)

    @property
    def evidence_bindings(
        self,
    ) -> tuple[ModelScoreEvidenceArtifactBinding, ...]:
        """Return the de-duplicated authenticated evidence bindings."""

        return tuple(binding for _field, binding in self.field_bindings)


def model_score_virtual_field(score_vector_artifact_id: str) -> str:
    """Return the reserved virtual field for one exact score-vector artifact."""

    if (
        not isinstance(score_vector_artifact_id, str)
        or _SHA256_RE.fullmatch(score_vector_artifact_id) is None
    ):
        raise StrategyError("score-vector artifact id must be a lowercase SHA-256")
    return _MODEL_SCORE_VIRTUAL_FIELD_PREFIX + score_vector_artifact_id[:16]


def normalize_pool_requirements(
    value: object,
) -> tuple[dict[str, Any], ...]:
    """Flatten and validate executable Pool requirements.

    Voting assets retain their selected-member lineage as nested envelopes.
    Downstream execution needs the terminal typed requirement while preserving
    the innermost rule/fragment attribution. Ordering and multiplicity are
    preserved because they are part of the existing provenance hash contract;
    only the authenticated score-vector bindings are de-duplicated later.
    """

    if isinstance(value, str | bytes | bytearray) or not isinstance(
        value,
        Sequence,
    ):
        raise StrategyError("Pool requirements must be an array")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise StrategyError(f"compiled requirement[{index}] must be an object")
        if set(item) != _OUTER_REQUIREMENT_FIELDS:
            raise StrategyError(
                f"compiled requirement[{index}] fields must be exactly "
                "rule_id, fragment_id, requirement"
            )
        rule_id = _text(
            item["rule_id"],
            f"compiled requirement[{index}].rule_id",
        )
        fragment_id = _text(
            item["fragment_id"],
            f"compiled requirement[{index}].fragment_id",
        )
        requirement = item["requirement"]
        depth = 0
        while isinstance(requirement, Mapping) and (
            set(requirement) & _REQUIREMENT_LINEAGE_ENVELOPE_FIELDS
        ):
            if set(requirement) != _REQUIREMENT_LINEAGE_ENVELOPE_FIELDS:
                raise StrategyError(
                    f"compiled requirement[{index}] Voting lineage envelope "
                    "fields are invalid"
                )
            depth += 1
            if depth > _MAX_REQUIREMENT_LINEAGE_DEPTH:
                raise StrategyError(
                    f"compiled requirement[{index}] Voting lineage depth "
                    f"exceeds {_MAX_REQUIREMENT_LINEAGE_DEPTH}"
                )
            _text(
                requirement["entry_id"],
                f"compiled requirement[{index}] Voting entry_id",
            )
            rule_id = _text(
                requirement["rule_id"],
                f"compiled requirement[{index}] Voting rule_id",
            )
            fragment_id = _text(
                requirement["fragment_id"],
                f"compiled requirement[{index}] Voting fragment_id",
            )
            requirement = requirement["requirement"]
        outer = {
            "rule_id": rule_id,
            "fragment_id": fragment_id,
            "requirement": _model_score_requirement(requirement),
        }
        normalized.append(outer)
    return tuple(normalized)


def project_pool_entry_requirements(
    entries: object,
) -> tuple[dict[str, Any], ...]:
    """Project canonical Pool entries into normalized executable requirements."""

    if isinstance(entries, str | bytes | bytearray) or not isinstance(
        entries,
        Sequence,
    ):
        raise StrategyError("Strategy Pool entries must be an array")
    projected: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise StrategyError(f"Strategy Pool entry[{index}] must be an object")
        source = entry.get("source")
        execution = entry.get("execution")
        if not isinstance(source, Mapping) or not isinstance(execution, Mapping):
            raise StrategyError(
                f"Strategy Pool entry[{index}] execution lineage is invalid"
            )
        raw_requirements = execution.get("requirements")
        if isinstance(
            raw_requirements,
            str | bytes | bytearray,
        ) or not isinstance(raw_requirements, Sequence):
            raise StrategyError(
                f"Strategy Pool entry[{index}] requirements must be an array"
            )
        rule_id = _text(
            entry.get("rule_id"),
            f"Strategy Pool entry[{index}].rule_id",
        )
        fragment_id = _text(
            source.get("fragment_id"),
            f"Strategy Pool entry[{index}].source.fragment_id",
        )
        projected.extend(
            {
                "rule_id": rule_id,
                "fragment_id": fragment_id,
                "requirement": requirement,
            }
            for requirement in raw_requirements
        )
    return normalize_pool_requirements(projected)


def resolve_pool_requirements(
    runtime,
    *,
    task_id: str,
    compiled_design: Mapping[str, Any],
    sample_design: StrategySampleDesignV2Binding,
) -> ResolvedPoolRequirements:
    """Authenticate requirements against the selected V2 execution source."""

    return _resolve_pool_requirements(
        runtime,
        task_id=task_id,
        compiled_design=compiled_design,
        sample_design=sample_design,
        require_current_scores=True,
    )


def resolve_historical_pool_requirements(
    runtime,
    *,
    task_id: str,
    compiled_design: Mapping[str, Any],
    sample_design: StrategySampleDesignV2Binding,
) -> ResolvedPoolRequirements:
    """Authenticate requirements without requiring score samples to be head."""

    return _resolve_pool_requirements(
        runtime,
        task_id=task_id,
        compiled_design=compiled_design,
        sample_design=sample_design,
        require_current_scores=False,
    )


def _resolve_pool_requirements(
    runtime,
    *,
    task_id: str,
    compiled_design: Mapping[str, Any],
    sample_design: StrategySampleDesignV2Binding,
    require_current_scores: bool,
) -> ResolvedPoolRequirements:
    task = _text(task_id, "task_id")
    if not isinstance(
        sample_design,
        (
            StrategySampleDesignV2ArtifactBinding,
            StrategySampleDesignV2NativeArtifactBinding,
        ),
    ):
        raise StrategyError("sample-design V2 artifact binding is invalid")
    if sample_design.task_id != task:
        raise StrategyError("Pool requirements and sample design task differ")
    if isinstance(sample_design, StrategySampleDesignV2NativeArtifactBinding):
        source_mode = resolve_strategy_sample_design_v2_source_mode(
            sample_design.bundle["sample_design"],
            capability="physical_v2",
            consumer="strategy_pool_requirement_resolver",
        )
        if source_mode != "native_active_dataset":
            raise StrategyError(
                "native sample-design V2 source mode changed"
            )
    if not isinstance(compiled_design, Mapping):
        raise StrategyError("compiled design must be an object")
    requirements = normalize_pool_requirements(
        compiled_design.get("requirements")
    )
    requirements_hash = _sha256(_canonical_json(list(requirements)))
    if not requirements:
        return ResolvedPoolRequirements(
            task_id=task,
            requirements_hash=requirements_hash,
            requirements=(),
            field_bindings=(),
        )

    physical_fields = {str(column) for column in sample_design.source_binding.columns}
    unique: list[dict[str, str]] = []
    exact_refs: set[str] = set()
    by_virtual_field: dict[str, str] = {}
    for outer in requirements:
        requirement = _model_score_requirement(outer["requirement"])
        field = requirement["virtual_field"]
        canonical = _canonical_json(requirement)
        previous = by_virtual_field.get(field)
        if previous is not None and previous != canonical:
            raise StrategyError(
                "one virtual score field cannot bind different score-vector references"
            )
        by_virtual_field[field] = canonical
        if field in physical_fields:
            raise StrategyError(
                f"virtual score field conflicts with physical dataset column: {field}"
            )
        if canonical not in exact_refs:
            exact_refs.add(canonical)
            unique.append(requirement)

    field_bindings: list[tuple[str, ModelScoreEvidenceArtifactBinding]] = []
    for requirement in unique:
        try:
            score_loader = (
                load_model_score_evidence_artifacts
                if require_current_scores
                else load_historical_model_score_evidence_artifacts
            )
            binding = score_loader(
                _modeling_runtime(runtime),
                task_id=task,
                evidence_artifact_id=requirement[
                    "score_evidence_artifact_id"
                ],
                expected_evidence_artifact_content_hash=requirement[
                    "score_evidence_artifact_content_hash"
                ],
                score_vector_artifact_id=requirement[
                    "score_vector_artifact_id"
                ],
                expected_score_vector_artifact_content_hash=requirement[
                    "score_vector_artifact_content_hash"
                ],
            )
        except ModelingError as exc:
            raise StrategyError(str(exc)) from exc
        _require_exact_sample(binding, sample_design=sample_design, task_id=task)
        field_bindings.append((requirement["virtual_field"], binding))

    return ResolvedPoolRequirements(
        task_id=task,
        requirements_hash=requirements_hash,
        requirements=requirements,
        field_bindings=tuple(field_bindings),
    )


def _modeling_runtime(runtime):
    """Add modeling repositories without mutating the owning pack runtime."""

    if hasattr(runtime, "experiments") and hasattr(runtime, "modeling_repo"):
        return runtime
    proxy = SimpleNamespace(**vars(runtime))
    proxy.experiments = ExperimentStore(runtime.settings.db_path)
    proxy.modeling_repo = ModelingRepository(runtime.settings.db_path)
    return proxy


def hydrate_requirement_fields(
    frame: pd.DataFrame,
    *,
    resolved: ResolvedPoolRequirements,
) -> pd.DataFrame:
    """Return a detached frame with score vectors injected by raw row ordinal."""

    _require_resolved(resolved)
    if not isinstance(frame, pd.DataFrame):
        raise StrategyError("Pool requirement hydration requires a DataFrame")
    hydrated = frame.copy(deep=True)
    if not resolved.field_bindings:
        return hydrated
    expected_index = pd.RangeIndex(start=0, stop=len(frame), step=1)
    if not isinstance(frame.index, pd.RangeIndex) or not frame.index.equals(
        expected_index
    ):
        raise StrategyError(
            "Pool requirement hydration requires an exact zero-based RangeIndex"
        )
    for field, binding in resolved.field_bindings:
        if field in frame.columns:
            raise StrategyError(
                f"virtual score field already exists in dataset frame: {field}"
            )
        vector = binding.vector
        if vector.row_count != len(frame):
            raise StrategyError("model score vector length differs from dataset rows")
        expected_ordinals = np.arange(len(frame), dtype=np.int64)
        if not np.array_equal(vector.row_ordinals, expected_ordinals):
            raise StrategyError("model score vector row order changed")
        hydrated[field] = np.asarray(vector.scores, dtype=np.float64).copy()
    return hydrated


def require_resolved_pool_requirements_on_connection(
    conn,
    resolved: ResolvedPoolRequirements,
) -> None:
    """Re-authenticate all resolved evidence under a downstream transaction."""

    _require_resolved_pool_requirements_on_connection(
        conn,
        resolved,
        require_current_scores=True,
    )


def require_historical_resolved_pool_requirements_on_connection(
    conn,
    resolved: ResolvedPoolRequirements,
) -> None:
    """Re-authenticate requirements without requiring score samples to be head."""

    _require_resolved_pool_requirements_on_connection(
        conn,
        resolved,
        require_current_scores=False,
    )


def _require_resolved_pool_requirements_on_connection(
    conn,
    resolved: ResolvedPoolRequirements,
    *,
    require_current_scores: bool,
) -> None:
    _require_resolved(resolved)
    for binding in resolved.evidence_bindings:
        try:
            if require_current_scores:
                require_model_score_evidence_artifact_binding_on_connection(
                    conn,
                    binding,
                )
            else:
                require_historical_model_score_evidence_artifact_binding_on_connection(
                    conn,
                    binding,
                )
        except ModelingError as exc:
            raise StrategyError(str(exc)) from exc


def pool_requirement_bindings_provenance(
    resolved: ResolvedPoolRequirements,
) -> dict[str, Any]:
    """Project one resolved binding into canonical aggregate-only provenance."""

    _require_resolved(resolved)
    value = {
        "requirements_hash": resolved.requirements_hash,
        "requirements": list(resolved.requirements),
        "virtual_fields": list(resolved.virtual_fields),
    }
    return validate_pool_requirement_bindings_provenance(value)


def validate_pool_requirement_bindings_provenance(
    value: object,
) -> dict[str, Any]:
    """Validate canonical Pool requirement provenance without loading artifacts."""

    if not isinstance(value, Mapping):
        raise StrategyError("Pool requirement bindings provenance must be an object")
    _exact_fields(
        value,
        _REQUIREMENT_BINDINGS_FIELDS,
        "Pool requirement bindings provenance",
    )
    raw_requirements = value["requirements"]
    if isinstance(raw_requirements, str | bytes | bytearray) or not isinstance(
        raw_requirements,
        Sequence,
    ):
        raise StrategyError("Pool requirement bindings requirements must be an array")
    requirements = list(normalize_pool_requirements(raw_requirements))
    if not requirements:
        raise StrategyError("Pool requirement bindings requirements must not be empty")
    requirements_hash = _hash(
        value["requirements_hash"],
        "Pool requirement bindings requirements_hash",
    )
    expected_hash = _sha256(_canonical_json(requirements))
    if not hmac.compare_digest(requirements_hash, expected_hash):
        raise StrategyError("Pool requirement bindings requirements hash changed")

    expected_fields: list[str] = []
    seen_requirements: set[str] = set()
    for outer in requirements:
        requirement = _model_score_requirement(outer["requirement"])
        canonical = _canonical_json(requirement)
        if canonical in seen_requirements:
            continue
        seen_requirements.add(canonical)
        expected_fields.append(requirement["virtual_field"])

    raw_fields = value["virtual_fields"]
    if isinstance(raw_fields, str | bytes | bytearray) or not isinstance(
        raw_fields,
        Sequence,
    ):
        raise StrategyError("Pool requirement virtual_fields must be an array")
    virtual_fields = [
        _text(field, "Pool requirement virtual_field") for field in raw_fields
    ]
    if (
        len(virtual_fields) != len(set(virtual_fields))
        or virtual_fields != expected_fields
    ):
        raise StrategyError(
            "Pool requirement virtual fields differ from compiled requirements"
        )
    return {
        "requirements_hash": requirements_hash,
        "requirements": requirements,
        "virtual_fields": virtual_fields,
    }


def _outer_requirement(value: object, *, index: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StrategyError(f"compiled requirement[{index}] must be an object")
    if set(value) != _OUTER_REQUIREMENT_FIELDS:
        raise StrategyError(
            f"compiled requirement[{index}] fields must be exactly "
            "rule_id, fragment_id, requirement"
        )
    return {
        "rule_id": _text(value["rule_id"], f"compiled requirement[{index}].rule_id"),
        "fragment_id": _text(
            value["fragment_id"],
            f"compiled requirement[{index}].fragment_id",
        ),
        "requirement": _model_score_requirement(value["requirement"]),
    }


def _model_score_requirement(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise StrategyError("model score requirement must be an object")
    if set(value) != _MODEL_SCORE_REQUIREMENT_FIELDS:
        raise StrategyError(
            "model score requirement fields do not match model_score_vector.v1"
        )
    requirement_type = _text(value["type"], "model score requirement.type")
    if requirement_type != _MODEL_SCORE_REQUIREMENT_TYPE:
        raise StrategyError("unsupported Pool requirement type")
    vector_id = _hash(
        value["score_vector_artifact_id"],
        "model score requirement.score_vector_artifact_id",
    )
    vector_hash = _hash(
        value["score_vector_artifact_content_hash"],
        "model score requirement.score_vector_artifact_content_hash",
    )
    evidence_id = _hash(
        value["score_evidence_artifact_id"],
        "model score requirement.score_evidence_artifact_id",
    )
    evidence_hash = _hash(
        value["score_evidence_artifact_content_hash"],
        "model score requirement.score_evidence_artifact_content_hash",
    )
    field = _text(value["virtual_field"], "model score requirement.virtual_field")
    if field != model_score_virtual_field(vector_id):
        raise StrategyError(
            "model score requirement virtual_field does not match score vector"
        )
    score_product = _text(
        value["score_product"], "model score requirement.score_product"
    )
    if score_product != RAW_SCORE_PRODUCT:
        raise StrategyError("model score requirement score_product is unsupported")
    return {
        "type": requirement_type,
        "virtual_field": field,
        "score_product": score_product,
        "score_evidence_artifact_id": evidence_id,
        "score_evidence_artifact_content_hash": evidence_hash,
        "score_vector_artifact_id": vector_id,
        "score_vector_artifact_content_hash": vector_hash,
    }


def _require_exact_sample(
    binding: ModelScoreEvidenceArtifactBinding,
    *,
    sample_design: StrategySampleDesignV2Binding,
    task_id: str,
) -> None:
    if binding.task_id != task_id or binding.training.task_id != task_id:
        raise StrategyError("model score evidence belongs to another task")
    scored_sample = binding.training.sample
    scored_source = scored_sample.source_binding
    selected_source = sample_design.source_binding
    source_fields = (
        "dataset_id",
        "dataset_content_hash",
        "row_count",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
    )
    if any(
        getattr(scored_source, field) != getattr(selected_source, field)
        for field in source_fields
    ):
        raise StrategyError(
            "model score evidence does not bind the selected dataset/workspace"
        )
    if isinstance(sample_design, StrategySampleDesignV2ArtifactBinding):
        if _sample_identity(scored_sample) != _sample_identity(sample_design):
            raise StrategyError(
                "model score evidence does not bind the selected "
                "SampleDesign V2"
            )
    else:
        # Score evidence is a full-universe, zero-based row vector.  Native
        # approval/risk masks are authenticated separately by the consuming
        # workflow, so vector reuse is governed by exact dataset/workspace,
        # target polarity, missing-label policy, row count, and row ordinals.
        # Legacy callers retain their established exact V2-pair identity gate.
        source_mode = resolve_strategy_sample_design_v2_source_mode(
            sample_design.bundle["sample_design"],
            capability="physical_v2",
            consumer="strategy_pool_requirement_resolver",
        )
        if (
            source_mode != "native_active_dataset"
            or scored_source.target_col != selected_source.target_col
            or scored_source.target_bad_value
            != selected_source.target_bad_value
            or scored_source.drop_nan_labels
            is not selected_source.drop_nan_labels
        ):
            raise StrategyError(
                "model score evidence does not bind the selected native "
                "sample semantics"
            )
    header = sample_design.membership["header"]
    if (
        binding.vector.row_count != selected_source.row_count
        or header["row_count"] != selected_source.row_count
        or binding.envelope["score_product"] != RAW_SCORE_PRODUCT
    ):
        raise StrategyError(
            "model score vector length or scoring contract changed"
        )
    expected_ordinals = np.arange(selected_source.row_count, dtype=np.int64)
    if not np.array_equal(binding.vector.row_ordinals, expected_ordinals):
        raise StrategyError("model score vector row order changed")


def _sample_identity(
    binding: StrategySampleDesignV2Binding,
) -> tuple[object, ...]:
    header = binding.membership["header"]
    design = binding.bundle["sample_design"]
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
        _canonical_json(header["dataset_ref"]),
    )


def _require_resolved(value: object) -> None:
    if not isinstance(value, ResolvedPoolRequirements):
        raise StrategyError("resolved Pool requirements binding is invalid")
    _text(value.task_id, "resolved Pool requirements task_id")
    if not isinstance(value.requirements, tuple):
        raise StrategyError("resolved Pool requirements must be a tuple")
    normalized = normalize_pool_requirements(value.requirements)
    if normalized != value.requirements:
        raise StrategyError("resolved Pool requirements are not canonical")
    if _SHA256_RE.fullmatch(value.requirements_hash) is None:
        raise StrategyError("resolved Pool requirements hash is invalid")
    if value.requirements_hash != _sha256(
        _canonical_json(list(normalized))
    ):
        raise StrategyError("resolved Pool requirements changed after resolution")

    unique: list[dict[str, str]] = []
    seen_refs: set[str] = set()
    by_field: dict[str, str] = {}
    for outer in normalized:
        requirement = _model_score_requirement(outer["requirement"])
        canonical = _canonical_json(requirement)
        field = requirement["virtual_field"]
        previous = by_field.get(field)
        if previous is not None and previous != canonical:
            raise StrategyError(
                "resolved Pool virtual field has conflicting references"
            )
        by_field[field] = canonical
        if canonical not in seen_refs:
            seen_refs.add(canonical)
            unique.append(requirement)
    if len(value.field_bindings) != len(unique):
        raise StrategyError(
            "resolved Pool requirement bindings do not match requirements"
        )

    seen: set[str] = set()
    for (field, binding), requirement in zip(
        value.field_bindings,
        unique,
        strict=True,
    ):
        if field in seen:
            raise StrategyError("resolved Pool virtual fields are not unique")
        seen.add(field)
        if not isinstance(binding, ModelScoreEvidenceArtifactBinding):
            raise StrategyError("resolved model score evidence binding is invalid")
        if (
            field != requirement["virtual_field"]
            or binding.task_id != value.task_id
            or str(binding.vector_record.get("id"))
            != requirement["score_vector_artifact_id"]
            or str(binding.vector_record.get("content_hash"))
            != requirement["score_vector_artifact_content_hash"]
            or str(binding.evidence_record.get("id"))
            != requirement["score_evidence_artifact_id"]
            or str(binding.evidence_record.get("content_hash"))
            != requirement["score_evidence_artifact_content_hash"]
            or binding.vector.content_hash
            != requirement["score_vector_artifact_content_hash"]
            or binding.envelope.get("score_product") != RAW_SCORE_PRODUCT
        ):
            raise StrategyError("resolved model score evidence binding changed")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StrategyError(f"{name} must be a non-empty string")
    return value.strip()


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise StrategyError(f"{name} must be a lowercase SHA-256")
    return value


def _exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unsupported: " + ", ".join(unexpected))
        raise StrategyError(f"{name} fields are invalid ({'; '.join(details)})")


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise StrategyError("Pool requirements must be canonical JSON") from exc


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "ResolvedPoolRequirements",
    "hydrate_requirement_fields",
    "model_score_virtual_field",
    "pool_requirement_bindings_provenance",
    "require_historical_resolved_pool_requirements_on_connection",
    "require_resolved_pool_requirements_on_connection",
    "resolve_historical_pool_requirements",
    "resolve_pool_requirements",
    "validate_pool_requirement_bindings_provenance",
]
