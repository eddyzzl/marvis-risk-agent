from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from marvis.validation.feature_metadata import (
    FeatureMetadataSelection,
    normalize_feature_metadata,
)
from marvis.validation.field_transformations import (
    apply_confirmed_transformations,
    required_transformation_inputs,
    topologically_sorted_transformations,
    validate_transformation_plan,
)
from marvis.validation.input_contracts import (
    FeatureMetadataResolution,
    JsonScalar,
    JsonValue,
    SampleSchema,
    TransformationSpec,
    ValidationInputConfirmation,
    ValidationInputContract,
)
from marvis.validation.pmml_manifest import choose_pmml_output_field
from marvis.validation.sample_schema import iter_sample_projection
from marvis.validation.time_periods import date_key_series, month_key_series


CONFIRMATION_CHUNK_SIZE = 100_000
MAX_OBSERVED_VALUES = 100
MAX_SCALAR_CHARS = 10_000
MAX_CONFIRMATION_ERROR_CHARS = 500
_CANONICAL_SPLITS = ("train", "test", "oot")
_METADATA_CANDIDATE_KEYS = (
    "feature_metadata_selection",
    "metadata_selection",
)


@dataclass(frozen=True)
class ValidatedConfirmation:
    values: ValidationInputConfirmation
    sample_schema: SampleSchema
    feature_metadata: FeatureMetadataResolution


@dataclass(frozen=True)
class ObservedConfirmationValues:
    target_values: tuple[JsonScalar, ...]
    split_values: tuple[JsonScalar, ...]
    row_count: int


def json_scalar_identity(value: object) -> tuple[str, str]:
    normalized = _python_scalar(value)
    if normalized is None:
        raise ValueError("field value is null, not a non-null JSON scalar")
    if type(normalized) not in {str, int, float, bool}:
        raise ValueError("field value is not a JSON scalar")
    if isinstance(normalized, str) and len(normalized) > MAX_SCALAR_CHARS:
        raise ValueError("field value exceeds scalar length limit")
    if isinstance(normalized, float) and not math.isfinite(normalized):
        raise ValueError("field value is not finite")
    return type(normalized).__name__, json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    )


def normalize_binary_target(
    values: pd.Series, *, positive: object, negative: object
) -> pd.Series:
    positive_key = json_scalar_identity(positive)
    negative_key = json_scalar_identity(negative)
    if positive_key == negative_key:
        raise ValueError("positive and negative labels must differ")
    normalized: list[int] = []
    for value in values.tolist():
        if _is_null(value):
            raise ValueError("target contains null values")
        key = json_scalar_identity(value)
        if key == positive_key:
            normalized.append(1)
        elif key == negative_key:
            normalized.append(0)
        elif _numeric_scalar_equal(value, positive) and not _numeric_scalar_equal(
            value, negative
        ):
            normalized.append(1)
        elif _numeric_scalar_equal(value, negative) and not _numeric_scalar_equal(
            value, positive
        ):
            normalized.append(0)
        else:
            raise ValueError("target contains a value outside confirmed labels")
    return pd.Series(normalized, index=values.index, dtype="int8")


def validate_binary_labels(
    observed: Sequence[object], *, positive: object, negative: object | None
) -> JsonScalar:
    observed_by_key = _unique_scalar_mapping(observed, field="target")
    positive_value, positive_key = _resolve_observed_label(
        observed_by_key,
        positive,
        field="positive label",
    )
    if len(observed_by_key) != 2:
        raise ValueError("binary target must contain exactly two typed values")
    if negative is None:
        return next(
            value
            for key, value in observed_by_key.items()
            if key != positive_key
        )
    negative_value, negative_key = _resolve_observed_label(
        observed_by_key,
        negative,
        field="negative label",
    )
    if negative_key == positive_key:
        raise ValueError("positive and negative labels must differ")
    if set(observed_by_key) != {positive_key, negative_key}:
        raise ValueError("target contains a value outside confirmed labels")
    return negative_value


def _resolve_observed_label(
    observed_by_key: Mapping[tuple[str, str], JsonScalar],
    requested: object,
    *,
    field: str,
) -> tuple[JsonScalar, tuple[str, str]]:
    _, requested_key = _json_scalar_with_identity(requested, field=field)
    if requested_key in observed_by_key:
        return observed_by_key[requested_key], requested_key
    matches = [
        (value, key)
        for key, value in observed_by_key.items()
        if _numeric_scalar_equal(value, requested)
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"{field} matches multiple typed target values")
    raise ValueError(f"{field} is not observed in target")


def _numeric_scalar_equal(left: object, right: object) -> bool:
    left = _python_scalar(left)
    right = _python_scalar(right)
    if type(left) not in {int, float} or type(right) not in {int, float}:
        return False
    if isinstance(left, float) and not math.isfinite(left):
        return False
    if isinstance(right, float) and not math.isfinite(right):
        return False
    return left == right


def validate_split_mapping(
    observed: Sequence[object],
    mapping: Mapping[str, object],
    *,
    required_canonical: tuple[str, ...] = _CANONICAL_SPLITS,
) -> dict[str, JsonScalar]:
    if not isinstance(mapping, Mapping) or set(mapping) != set(required_canonical):
        raise ValueError(
            "split mapping keys must be exactly train, test, and oot"
        )
    normalized: dict[str, JsonScalar] = {}
    mapped_by_key: dict[tuple[str, str], str] = {}
    for canonical in required_canonical:
        value, identity = _json_scalar_with_identity(
            mapping[canonical], field=f"split mapping {canonical}"
        )
        if identity in mapped_by_key:
            raise ValueError("split mapping values must be type-stably unique")
        mapped_by_key[identity] = canonical
        normalized[canonical] = value

    observed_by_key = _unique_scalar_mapping(observed, field="split")
    unmapped = set(observed_by_key) - set(mapped_by_key)
    if unmapped:
        raise ValueError("split contains unmapped observed values")
    empty = [
        canonical
        for identity, canonical in mapped_by_key.items()
        if identity not in observed_by_key
    ]
    if empty:
        raise ValueError("required split has no rows: " + ", ".join(empty))
    return normalized


def inspect_confirmation_values(
    sample_path: Path,
    *,
    columns: Sequence[str],
    transformations: Sequence[TransformationSpec],
    target_col: str,
    split_col: str,
    time_col: str,
    time_granularity: str,
    sample_schema: SampleSchema,
    chunk_size: int = CONFIRMATION_CHUNK_SIZE,
) -> ObservedConfirmationValues:
    try:
        return _inspect_confirmation_values(
            sample_path,
            columns=columns,
            transformations=transformations,
            target_col=target_col,
            split_col=split_col,
            time_col=time_col,
            time_granularity=time_granularity,
            sample_schema=sample_schema,
            chunk_size=chunk_size,
        )
    except ValueError as exc:
        raise ValueError(_bounded_text(str(exc))) from None


def _inspect_confirmation_values(
    sample_path: Path,
    *,
    columns: Sequence[str],
    transformations: Sequence[TransformationSpec],
    target_col: str,
    split_col: str,
    time_col: str,
    time_granularity: str,
    sample_schema: SampleSchema,
    chunk_size: int,
) -> ObservedConfirmationValues:
    _validate_time_granularity(time_granularity)
    target_values: dict[tuple[str, str], JsonScalar] = {}
    split_values: dict[tuple[str, str], JsonScalar] = {}
    row_count = 0
    for frame in iter_sample_projection(
        sample_path,
        columns=tuple(columns),
        chunk_size=chunk_size,
        schema=sample_schema,
    ):
        transformed = apply_confirmed_transformations(frame, transformations)
        missing = [
            field
            for field in (target_col, split_col, time_col)
            if field not in transformed.columns
        ]
        if missing:
            raise ValueError(
                "confirmed control fields are unavailable after transformations: "
                + _bounded_join(missing)
            )
        _collect_bounded_json_scalars(
            target_values,
            transformed[target_col],
            limit=MAX_OBSERVED_VALUES,
            field="target",
        )
        _collect_bounded_json_scalars(
            split_values,
            transformed[split_col],
            limit=MAX_OBSERVED_VALUES,
            field="split",
        )
        _validate_time_series(transformed[time_col], time_granularity)
        row_count += len(transformed)
    if row_count == 0:
        raise ValueError("validation sample contains no rows")
    return ObservedConfirmationValues(
        target_values=tuple(target_values.values()),
        split_values=tuple(split_values.values()),
        row_count=row_count,
    )


def validate_confirmation_against_materials(
    *,
    contract: ValidationInputContract,
    sample_path: Path,
    dictionary_path: Path,
    requested: ValidationInputConfirmation,
) -> ValidatedConfirmation:
    try:
        return _validate_confirmation_against_materials(
            contract=contract,
            sample_path=sample_path,
            dictionary_path=dictionary_path,
            requested=requested,
        )
    except ValueError as exc:
        raise ValueError(_bounded_text(str(exc))) from None


def _validate_confirmation_against_materials(
    *,
    contract: ValidationInputContract,
    sample_path: Path,
    dictionary_path: Path,
    requested: ValidationInputConfirmation,
) -> ValidatedConfirmation:
    schema = contract.require_sample_schema()
    manifest = contract.require_pmml_manifest()
    _validate_control_fields(requested, schema=schema)
    if not isinstance(requested.model_params, dict):
        raise ValueError("model_params must be a JSON object")
    _validate_json_value(requested.model_params, field="model_params")

    output = choose_pmml_output_field(
        manifest,
        notebook_hint=None,
        user_confirmation=requested.pmml_output_field,
    )
    if output.needs_confirmation or output.selected is None:
        raise ValueError("PMML output field requires a valid selection")

    selection = feature_metadata_selection_from_confirmation(requested)
    _require_atomic_metadata_candidate(contract, requested)

    requested_transformations = _validated_requested_transformations(
        contract,
        requested=requested.transformations,
        sample_columns=schema.columns,
    )
    required_static_inputs = required_transformation_inputs(
        (
            *manifest.raw_required_fields,
            requested.target_col,
            requested.split_col,
            requested.time_col,
        ),
        requested_transformations,
    )
    static_missing = [
        field for field in required_static_inputs if field not in schema.columns
    ]
    if static_missing:
        raise ValueError(
            "confirmed fields are missing required sample inputs: "
            + _bounded_join(static_missing)
        )

    control_transformations = _transformation_closure(
        (requested.target_col, requested.split_col, requested.time_col),
        requested_transformations,
    )
    projection = required_transformation_inputs(
        (requested.target_col, requested.split_col, requested.time_col),
        control_transformations,
    )
    projection = tuple(dict.fromkeys(projection))
    if not projection:
        if not schema.columns:
            raise ValueError("validation sample schema contains no columns")
        projection = (schema.columns[0],)

    observed = inspect_confirmation_values(
        Path(sample_path),
        columns=projection,
        transformations=control_transformations,
        target_col=requested.target_col,
        split_col=requested.split_col,
        time_col=requested.time_col,
        time_granularity=requested.time_granularity,
        sample_schema=schema,
    )
    if schema.row_count is not None and observed.row_count != schema.row_count:
        raise ValueError(
            "validation sample row count does not match scanned sample schema"
        )
    negative = validate_binary_labels(
        observed.target_values,
        positive=requested.positive_label,
        negative=requested.negative_label,
    )
    split_mapping = validate_split_mapping(
        observed.split_values,
        requested.split_value_mapping,
        required_canonical=_CANONICAL_SPLITS,
    )

    try:
        metadata = normalize_feature_metadata(
            Path(dictionary_path), selection=selection, manifest=manifest
        )
        if not metadata_has_complete_coverage(metadata):
            raise ValueError("coverage must be 100%")
    except ValueError as exc:
        raise ValueError(f"feature metadata: {exc}") from None

    resolved = replace(
        requested,
        positive_label=_required_json_scalar(
            requested.positive_label, field="positive label"
        ),
        negative_label=negative,
        split_value_mapping=split_mapping,
        pmml_output_field=output.selected,
        transformations=requested_transformations,
    )
    return ValidatedConfirmation(
        values=resolved,
        sample_schema=schema,
        feature_metadata=metadata,
    )


def feature_metadata_selection_from_confirmation(
    requested: ValidationInputConfirmation,
) -> FeatureMetadataSelection:
    return FeatureMetadataSelection(
        sheet_name=requested.metadata_sheet,
        feature_col=requested.feature_col,
        category_col=requested.category_col,
        importance_col=requested.importance_col,
    )


def metadata_has_complete_coverage(metadata: FeatureMetadataResolution) -> bool:
    coverage = metadata.coverage
    return (
        not metadata.conflicts
        and coverage.feature == 1.0
        and coverage.category == 1.0
        and coverage.importance == 1.0
        and coverage.stress_unit == 1.0
    )


def _validate_control_fields(
    requested: ValidationInputConfirmation, *, schema: SampleSchema
) -> None:
    fields = (requested.target_col, requested.split_col, requested.time_col)
    if any(not isinstance(field, str) or not field.strip() for field in fields):
        raise ValueError("control columns must be non-empty strings")
    if any(len(field) > MAX_SCALAR_CHARS for field in fields):
        raise ValueError("control column name exceeds scalar length limit")
    if len(set(fields)) != len(fields):
        raise ValueError("target, split, and time control columns must be distinct")
    if not isinstance(schema, SampleSchema):
        raise ValueError("validation input contract has invalid sample schema")
    _validate_time_granularity(requested.time_granularity)


def _validated_requested_transformations(
    contract: ValidationInputContract,
    *,
    requested: Sequence[TransformationSpec],
    sample_columns: Sequence[str],
) -> tuple[TransformationSpec, ...]:
    validated = validate_transformation_plan(
        tuple(requested), sample_columns=sample_columns
    )
    allowed = {
        _transformation_identity(spec) for spec in contract.transformations
    }
    for spec in validated:
        if _transformation_identity(spec) not in allowed:
            raise ValueError(
                "requested transformations must be an exact scanned transformation subset"
            )
    return validated


def _transformation_identity(spec: TransformationSpec) -> tuple[Any, ...]:
    return (
        spec.operation,
        spec.output_field,
        spec.input_fields,
        _typed_json_fingerprint(spec.params),
    )


def _typed_json_fingerprint(
    value: object, *, depth: int = 0
) -> tuple[Any, ...]:
    if depth > 64:
        raise ValueError("transformation value exceeds maximum JSON depth")
    if value is None:
        return ("none",)
    if type(value) in {str, int, float, bool}:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("transformation JSON numbers must be finite")
        return ("scalar", type(value).__name__, value)
    if isinstance(value, list):
        return (
            "list",
            tuple(
                _typed_json_fingerprint(item, depth=depth + 1)
                for item in value
            ),
        )
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return (
            "dict",
            tuple(
                (
                    key,
                    _typed_json_fingerprint(value[key], depth=depth + 1),
                )
                for key in sorted(value)
            ),
        )
    raise ValueError("transformation value is not JSON-compatible")


def _transformation_closure(
    output_fields: Sequence[str], specs: Sequence[TransformationSpec]
) -> tuple[TransformationSpec, ...]:
    ordered = topologically_sorted_transformations(specs)
    by_output = {spec.output_field: spec for spec in ordered}
    needed: set[str] = set()
    stack = list(reversed(tuple(output_fields)))
    while stack:
        field = stack.pop()
        spec = by_output.get(field)
        if spec is None or field in needed:
            continue
        needed.add(field)
        stack.extend(reversed(spec.input_fields))
    return tuple(spec for spec in ordered if spec.output_field in needed)


def _require_atomic_metadata_candidate(
    contract: ValidationInputContract, requested: ValidationInputConfirmation
) -> None:
    requested_value = _metadata_selection_value(requested)
    candidates: list[dict[str, JsonValue]] = []
    for key in _METADATA_CANDIDATE_KEYS:
        for candidate in contract.candidates.get(key, ()):
            value = candidate.value
            if isinstance(value, dict):
                normalized = _normalized_metadata_candidate(value)
                if normalized is not None:
                    candidates.append(normalized)
    if not candidates:
        candidates.extend(_aligned_metadata_candidates(contract))
    if requested_value not in candidates:
        raise ValueError(
            "feature metadata: confirmed selection is not an atomic scanned candidate"
        )


def _metadata_selection_value(
    requested: ValidationInputConfirmation,
) -> dict[str, JsonValue]:
    return {
        "metadata_sheet": requested.metadata_sheet,
        "feature_col": requested.feature_col,
        "category_col": requested.category_col,
        "importance_col": requested.importance_col,
    }


def _normalized_metadata_candidate(
    value: dict[str, JsonValue],
) -> dict[str, JsonValue] | None:
    sheet_key = "metadata_sheet" if "metadata_sheet" in value else "sheet_name"
    expected = {sheet_key, "feature_col", "category_col", "importance_col"}
    if set(value) != expected:
        return None
    sheet = value[sheet_key]
    columns = tuple(
        value[key] for key in ("feature_col", "category_col", "importance_col")
    )
    if sheet is not None and not isinstance(sheet, str):
        return None
    if any(not isinstance(column, str) or not column for column in columns):
        return None
    return {
        "metadata_sheet": sheet,
        "feature_col": cast(str, columns[0]),
        "category_col": cast(str, columns[1]),
        "importance_col": cast(str, columns[2]),
    }


def _aligned_metadata_candidates(
    contract: ValidationInputContract,
) -> list[dict[str, JsonValue]]:
    keys = ("metadata_sheet", "feature_col", "category_col", "importance_col")
    values = [contract.candidates.get(key, ()) for key in keys]
    if not all(values) or len({len(items) for items in values}) != 1:
        return []
    result: list[dict[str, JsonValue]] = []
    for index in range(len(values[0])):
        candidate = {
            key: values[position][index].value
            for position, key in enumerate(keys)
        }
        normalized = _normalized_metadata_candidate(candidate)
        if normalized is not None:
            result.append(normalized)
    return result


def _collect_bounded_json_scalars(
    destination: dict[tuple[str, str], JsonScalar],
    values: pd.Series,
    *,
    limit: int,
    field: str,
) -> None:
    for raw_value in values.tolist():
        if _is_null(raw_value):
            raise ValueError(f"{field} contains null values")
        value, identity = _json_scalar_with_identity(raw_value, field=field)
        if identity in destination:
            continue
        if len(destination) >= limit:
            raise ValueError(f"{field} unique value limit exceeded")
        destination[identity] = value


def _unique_scalar_mapping(
    values: Sequence[object], *, field: str
) -> dict[tuple[str, str], JsonScalar]:
    result: dict[tuple[str, str], JsonScalar] = {}
    for raw_value in values:
        value, identity = _json_scalar_with_identity(raw_value, field=field)
        result.setdefault(identity, value)
    return result


def _required_json_scalar(value: object, *, field: str) -> JsonScalar:
    normalized, _ = _json_scalar_with_identity(value, field=field)
    return normalized


def _json_scalar_with_identity(
    value: object, *, field: str
) -> tuple[JsonScalar, tuple[str, str]]:
    normalized = _python_scalar(value)
    try:
        identity = json_scalar_identity(normalized)
    except ValueError as exc:
        raise ValueError(f"{field} {exc}") from None
    return cast(JsonScalar, normalized), identity


def _python_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_null(value: object) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _validate_time_granularity(value: object) -> None:
    if value not in {"month", "date"}:
        raise ValueError("time_granularity must be month or date")


def _validate_time_series(values: pd.Series, granularity: str) -> None:
    column_name = str(values.name or "time")
    for raw_value in values.tolist():
        value = _python_scalar(raw_value)
        if _is_null(value):
            raise ValueError(f"time column {column_name!r} contains null values")
        if isinstance(value, (datetime, date, pd.Timestamp)):
            continue
        if isinstance(value, str):
            if len(value) > MAX_SCALAR_CHARS:
                raise ValueError(
                    f"time column {column_name!r} contains a value exceeding "
                    "the scalar length limit"
                )
            continue
        if type(value) in {int, float}:
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(
                    f"time column {column_name!r} contains a non-finite value"
                )
            try:
                text = str(value)
            except ValueError:
                raise ValueError(
                    f"time column {column_name!r} contains a numeric value "
                    "exceeding the scalar length limit"
                ) from None
            if len(text) > MAX_SCALAR_CHARS:
                raise ValueError(
                    f"time column {column_name!r} contains a numeric value "
                    "exceeding the scalar length limit"
                )
            continue
        raise ValueError(
            f"time column {column_name!r} contains an unsupported scalar value"
        )
    if granularity == "month":
        month_key_series(values, column_name=column_name)
    else:
        date_key_series(values, column_name=column_name)


def _bounded_text(
    value: str, *, limit: int = MAX_CONFIRMATION_ERROR_CHARS
) -> str:
    if len(value) <= limit:
        return value
    if limit <= 3:
        return value[:limit]
    return value[: limit - 3] + "..."


def _bounded_join(
    values: Sequence[str], *, limit: int = MAX_CONFIRMATION_ERROR_CHARS
) -> str:
    result = ""
    for value in values:
        separator = ", " if result else ""
        remaining = limit - len(result) - len(separator)
        if remaining <= 0:
            return _bounded_text(result + "...", limit=limit)
        if len(value) > remaining:
            suffix = "..." if remaining >= 3 else ""
            result += separator + value[: max(remaining - len(suffix), 0)] + suffix
            return result
        result += separator + value
    return result


def _validate_json_value(value: object, *, field: str, depth: int = 0) -> None:
    if depth > 64:
        raise ValueError(f"{field} exceeds maximum JSON depth")
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(cast(float, value)):
            raise ValueError(f"{field} JSON numbers must be finite")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item, field=field, depth=depth + 1)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _validate_json_value(item, field=field, depth=depth + 1)
        return
    raise ValueError(f"{field} must contain only JSON-compatible values")
