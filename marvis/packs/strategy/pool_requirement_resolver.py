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
from typing import Any

import numpy as np
import pandas as pd

from marvis.packs.modeling.errors import ModelingError
from marvis.packs.modeling.evidence import RAW_SCORE_PRODUCT
from marvis.packs.modeling.score_evidence_tools import (
    ModelScoreEvidenceArtifactBinding,
    load_model_score_evidence_artifacts,
    require_model_score_evidence_artifact_binding_on_connection,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.sample_design_v2_tools import (
    StrategySampleDesignV2ArtifactBinding,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MODEL_SCORE_VIRTUAL_FIELD_PREFIX = "__marvis_model_pd_"
_OUTER_REQUIREMENT_FIELDS = frozenset({"rule_id", "fragment_id", "requirement"})
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


def resolve_pool_requirements(
    runtime,
    *,
    task_id: str,
    compiled_design: Mapping[str, Any],
    sample_design: StrategySampleDesignV2ArtifactBinding,
) -> ResolvedPoolRequirements:
    """Authenticate every supported requirement against the exact V2 sample."""

    task = _text(task_id, "task_id")
    if not isinstance(sample_design, StrategySampleDesignV2ArtifactBinding):
        raise StrategyError("sample-design V2 artifact binding is invalid")
    if sample_design.task_id != task:
        raise StrategyError("Pool requirements and sample design task differ")
    if not isinstance(compiled_design, Mapping):
        raise StrategyError("compiled design must be an object")
    raw_requirements = compiled_design.get("requirements")
    if isinstance(raw_requirements, str | bytes | bytearray) or not isinstance(
        raw_requirements, Sequence
    ):
        raise StrategyError("compiled design requirements must be an array")

    requirements = tuple(
        _outer_requirement(item, index=index)
        for index, item in enumerate(raw_requirements)
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
            binding = load_model_score_evidence_artifacts(
                runtime,
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

    _require_resolved(resolved)
    for binding in resolved.evidence_bindings:
        try:
            require_model_score_evidence_artifact_binding_on_connection(
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
    requirements = [
        _outer_requirement(item, index=index)
        for index, item in enumerate(raw_requirements)
    ]
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
    sample_design: StrategySampleDesignV2ArtifactBinding,
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
    if _sample_identity(scored_sample) != _sample_identity(sample_design):
        raise StrategyError(
            "model score evidence does not bind the selected SampleDesign V2"
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
    binding: StrategySampleDesignV2ArtifactBinding,
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
    normalized = tuple(
        _outer_requirement(item, index=index)
        for index, item in enumerate(value.requirements)
    )
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
    "require_resolved_pool_requirements_on_connection",
    "resolve_pool_requirements",
    "validate_pool_requirement_bindings_provenance",
]
