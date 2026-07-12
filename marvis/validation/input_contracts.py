from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Literal, cast

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

INPUT_CONTRACT_SCHEMA = "marvis.validation_input_contract.v1"
FIELD_RECOGNITION_SCHEMA = "marvis.field_recognition.v1"
PMML_INPUT_MANIFEST_SCHEMA = "marvis.pmml_input_manifest.v1"
FEATURE_METADATA_SCHEMA = "marvis.feature_metadata.v1"
MATERIAL_HASH_ROLES = frozenset({"notebook", "sample", "pmml", "dictionary"})
TRANSFORMATION_OPERATIONS = frozenset(
    {
        "copy",
        "rename",
        "date_to_month",
        "constant_threshold",
        "constant_mapping",
        "constant_source_label",
    }
)
CONTRACT_STATUSES = frozenset({"ready", "pending_confirmation", "blocked"})


@dataclass(frozen=True)
class FieldEvidence:
    source_kind: str
    notebook_cell: int | None
    source_excerpt: str
    confidence: float


@dataclass(frozen=True)
class FieldCandidate:
    value: JsonValue
    evidence: tuple[FieldEvidence, ...]


@dataclass(frozen=True)
class TransformationSpec:
    operation: Literal[
        "copy",
        "rename",
        "date_to_month",
        "constant_threshold",
        "constant_mapping",
        "constant_source_label",
    ]
    output_field: str
    input_fields: tuple[str, ...]
    params: dict[str, JsonValue]


@dataclass(frozen=True)
class StressUnit:
    model_feature: str
    raw_input_fields: tuple[str, ...]
    derivation_evidence: tuple[str, ...]


@dataclass(frozen=True)
class PmmlInputManifest:
    schema_version: str
    raw_required_fields: tuple[str, ...]
    derived_fields: tuple[str, ...]
    model_features: tuple[str, ...]
    stress_units: tuple[StressUnit, ...]
    unsupported_derivations: tuple[str, ...]
    output_candidates: tuple[str, ...]
    algorithm: str


@dataclass(frozen=True)
class SampleSchema:
    path: str
    columns: tuple[str, ...]
    dtypes: dict[str, str]
    row_count: int | None
    preview_row_count: int
    encoding: str | None
    sha256: str
    sheet_name: str | None = None


@dataclass(frozen=True)
class FeatureMetadataRow:
    feature: str
    category: str
    importance: float
    source_sheet: str | None
    in_pmml: bool


@dataclass(frozen=True)
class MetadataCoverage:
    feature: float
    category: float
    importance: float
    stress_unit: float


@dataclass(frozen=True)
class FeatureMetadataResolution:
    schema_version: str
    rows: tuple[FeatureMetadataRow, ...]
    coverage: MetadataCoverage
    per_category_raw_fields: dict[str, tuple[str, ...]]
    extra_features: tuple[str, ...]
    conflicts: tuple[str, ...]


@dataclass(frozen=True)
class FieldRecognitionResult:
    schema_version: str
    notebook_sha256: str
    candidates: dict[str, tuple[FieldCandidate, ...]]
    transformations: tuple[TransformationSpec, ...]
    # Reserved for non-confirmable structural/security failures. Competing
    # equal-priority assignments remain separate candidates for user confirmation.
    conflicts: tuple[str, ...]

    @classmethod
    def from_candidates(
        cls,
        *,
        notebook_sha256: str,
        candidates,
        transformations,
        conflicts,
    ) -> FieldRecognitionResult:
        return cls(
            schema_version=FIELD_RECOGNITION_SCHEMA,
            notebook_sha256=notebook_sha256,
            candidates={key: tuple(value) for key, value in candidates.items()},
            transformations=tuple(transformations),
            conflicts=tuple(conflicts),
        )


@dataclass(frozen=True)
class ValidationInputContract:
    schema_version: str
    material_hashes: dict[str, str]
    status: Literal["ready", "pending_confirmation", "blocked"]
    candidates: dict[str, tuple[FieldCandidate, ...]]
    sample_schema: SampleSchema | None = None
    pmml_manifest: PmmlInputManifest | None = None
    feature_metadata: FeatureMetadataResolution | None = None
    confirmed: dict[str, Any] = field(default_factory=dict)
    transformations: tuple[TransformationSpec, ...] = ()
    conflicts: tuple[str, ...] = ()

    @classmethod
    def minimal_for_test(cls, *, material_hashes, target_col):
        return cls(
            schema_version=INPUT_CONTRACT_SCHEMA,
            material_hashes=material_hashes,
            status="pending_confirmation",
            candidates={"target_col": (target_col,)},
        )

    def require_pmml_manifest(self) -> PmmlInputManifest:
        if self.pmml_manifest is None:
            raise ValueError("validation input contract has no PMML manifest")
        return self.pmml_manifest

    def require_sample_schema(self) -> SampleSchema:
        if self.sample_schema is None:
            raise ValueError("validation input contract has no sample schema")
        return self.sample_schema

    def require_feature_metadata(self) -> FeatureMetadataResolution:
        if self.feature_metadata is None:
            raise ValueError(
                "validation input contract has no resolved feature metadata"
            )
        return self.feature_metadata

    def require_output_field(self) -> str:
        value = str(self.confirmed.get("pmml_output_field") or "")
        if not value:
            raise ValueError(
                "validation input contract has no confirmed PMML output field"
            )
        return value

    def require_algorithm(self) -> str:
        manifest = self.require_pmml_manifest()
        value = str(self.confirmed.get("algorithm") or manifest.algorithm or "")
        if not value:
            raise ValueError("validation input contract has no model algorithm")
        return value

    def require_model_params(self) -> dict[str, JsonValue]:
        value = self.confirmed.get("model_params")
        if not isinstance(value, dict):
            raise ValueError(
                "validation input contract has no confirmed model parameters"
            )
        _require_json_value(value)
        return {str(key): cast(JsonValue, item) for key, item in value.items()}


@dataclass(frozen=True)
class ValidationInputConfirmation:
    target_col: str
    positive_label: JsonScalar
    negative_label: JsonScalar
    split_col: str
    split_value_mapping: dict[str, JsonScalar]
    time_col: str
    time_granularity: str
    pmml_output_field: str
    model_params: dict[str, JsonValue]
    metadata_sheet: str | None
    feature_col: str
    category_col: str
    importance_col: str
    transformations: tuple[TransformationSpec, ...]


def input_contract_to_dict(value: ValidationInputContract) -> dict[str, Any]:
    payload = _tuples_to_lists(asdict(value))
    _require_json_value(payload)
    return cast(dict[str, Any], payload)


def input_contract_from_dict(payload: dict[str, Any]) -> ValidationInputContract:
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != INPUT_CONTRACT_SCHEMA
    ):
        raise ValueError("unsupported validation input contract schema")
    return _decode_validation_input_contract(payload)


def transformation_spec_from_dict(payload: dict[str, Any]) -> TransformationSpec:
    _require_exact_keys(
        payload,
        {"operation", "output_field", "input_fields", "params"},
        "transformation",
    )
    operation = _require_non_empty_string(
        payload["operation"], "transformation operation"
    )
    if operation not in TRANSFORMATION_OPERATIONS:
        raise ValueError(f"unsupported transformation operation: {operation}")
    output_field = _require_non_empty_string(
        payload["output_field"], "transformation output field"
    )
    input_fields = _decode_string_tuple(
        payload["input_fields"], "transformation input fields", allow_empty=True
    )
    params = _decode_json_object(payload["params"], "transformation params")
    return TransformationSpec(cast(Any, operation), output_field, input_fields, params)


def validation_confirmation_to_dict(
    value: ValidationInputConfirmation,
) -> dict[str, Any]:
    payload = _tuples_to_lists(asdict(value))
    _require_json_value(payload)
    return cast(dict[str, Any], payload)


def _decode_validation_input_contract(
    payload: dict[str, Any],
) -> ValidationInputContract:
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "material_hashes",
            "status",
            "candidates",
            "sample_schema",
            "pmml_manifest",
            "feature_metadata",
            "confirmed",
            "transformations",
            "conflicts",
        },
        "validation input contract",
    )
    schema_version = _require_non_empty_string(
        payload["schema_version"], "schema version"
    )
    if schema_version != INPUT_CONTRACT_SCHEMA:
        raise ValueError("unsupported validation input contract schema")
    material_hashes = _decode_string_dict(payload["material_hashes"], "material hashes")
    if set(material_hashes) != MATERIAL_HASH_ROLES:
        raise ValueError(
            "invalid material hash roles: "
            f"{sorted(set(material_hashes) ^ MATERIAL_HASH_ROLES)}"
        )
    status = _require_non_empty_string(payload["status"], "contract status")
    if status not in CONTRACT_STATUSES:
        raise ValueError(f"unsupported validation input contract status: {status}")
    candidates_payload = _require_mapping(payload["candidates"], "field candidates")
    candidates: dict[str, tuple[FieldCandidate, ...]] = {}
    for key, values in candidates_payload.items():
        name = _require_non_empty_string(key, "candidate field name")
        candidates[name] = tuple(
            _decode_field_candidate(item)
            for item in _require_sequence(values, f"candidates for {name}")
        )
    sample_schema = (
        None
        if payload["sample_schema"] is None
        else _decode_sample_schema(payload["sample_schema"])
    )
    pmml_manifest = (
        None
        if payload["pmml_manifest"] is None
        else _decode_pmml_input_manifest(payload["pmml_manifest"])
    )
    feature_metadata = (
        None
        if payload["feature_metadata"] is None
        else _decode_feature_metadata_resolution(payload["feature_metadata"])
    )
    confirmed = _decode_json_object(payload["confirmed"], "confirmed values")
    transformations = tuple(
        transformation_spec_from_dict(item)
        for item in _require_sequence(payload["transformations"], "transformations")
    )
    conflicts = _decode_string_tuple(
        payload["conflicts"], "contract conflicts", allow_empty=True
    )
    return ValidationInputContract(
        schema_version=schema_version,
        material_hashes=material_hashes,
        status=cast(Any, status),
        candidates=candidates,
        sample_schema=sample_schema,
        pmml_manifest=pmml_manifest,
        feature_metadata=feature_metadata,
        confirmed=confirmed,
        transformations=transformations,
        conflicts=conflicts,
    )


def _decode_field_evidence(payload: object) -> FieldEvidence:
    value = _require_mapping(payload, "field evidence")
    _require_exact_keys(
        value,
        {"source_kind", "notebook_cell", "source_excerpt", "confidence"},
        "field evidence",
    )
    source_kind = _require_non_empty_string(
        value["source_kind"], "evidence source kind"
    )
    notebook_cell = value["notebook_cell"]
    if notebook_cell is not None:
        notebook_cell = _require_integer(
            notebook_cell, "evidence notebook cell", minimum=0
        )
    source_excerpt = _require_string(value["source_excerpt"], "evidence source excerpt")
    confidence = _require_finite_number(value["confidence"], "evidence confidence")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("evidence confidence must be between 0 and 1")
    return FieldEvidence(source_kind, notebook_cell, source_excerpt, confidence)


def _decode_field_candidate(payload: object) -> FieldCandidate:
    value = _require_mapping(payload, "field candidate")
    _require_exact_keys(value, {"value", "evidence"}, "field candidate")
    _require_json_value(value["value"])
    evidence = tuple(
        _decode_field_evidence(item)
        for item in _require_sequence(value["evidence"], "candidate evidence")
    )
    return FieldCandidate(cast(JsonValue, _tuples_to_lists(value["value"])), evidence)


def _decode_stress_unit(payload: object) -> StressUnit:
    value = _require_mapping(payload, "stress unit")
    _require_exact_keys(
        value,
        {"model_feature", "raw_input_fields", "derivation_evidence"},
        "stress unit",
    )
    return StressUnit(
        _require_non_empty_string(value["model_feature"], "stress model feature"),
        _decode_string_tuple(
            value["raw_input_fields"], "stress raw input fields", allow_empty=True
        ),
        _decode_string_tuple(
            value["derivation_evidence"], "stress derivation evidence", allow_empty=True
        ),
    )


def _decode_pmml_input_manifest(payload: object) -> PmmlInputManifest:
    value = _require_mapping(payload, "PMML input manifest")
    _require_exact_keys(
        value,
        {
            "schema_version",
            "raw_required_fields",
            "derived_fields",
            "model_features",
            "stress_units",
            "unsupported_derivations",
            "output_candidates",
            "algorithm",
        },
        "PMML input manifest",
    )
    schema_version = _require_non_empty_string(
        value["schema_version"], "PMML manifest schema"
    )
    if schema_version != PMML_INPUT_MANIFEST_SCHEMA:
        raise ValueError("unsupported PMML input manifest schema")
    return PmmlInputManifest(
        schema_version=schema_version,
        raw_required_fields=_decode_string_tuple(
            value["raw_required_fields"], "PMML raw required fields", allow_empty=True
        ),
        derived_fields=_decode_string_tuple(
            value["derived_fields"], "PMML derived fields", allow_empty=True
        ),
        model_features=_decode_string_tuple(
            value["model_features"], "PMML model features", allow_empty=True
        ),
        stress_units=tuple(
            _decode_stress_unit(item)
            for item in _require_sequence(value["stress_units"], "PMML stress units")
        ),
        unsupported_derivations=_decode_string_tuple(
            value["unsupported_derivations"],
            "PMML unsupported derivations",
            allow_empty=True,
        ),
        output_candidates=_decode_string_tuple(
            value["output_candidates"], "PMML output candidates", allow_empty=True
        ),
        algorithm=_require_string(value["algorithm"], "PMML algorithm").strip(),
    )


def _decode_sample_schema(payload: object) -> SampleSchema:
    value = _require_mapping(payload, "sample schema")
    _require_exact_keys(
        value,
        {
            "path",
            "columns",
            "dtypes",
            "row_count",
            "preview_row_count",
            "encoding",
            "sha256",
            "sheet_name",
        },
        "sample schema",
    )
    row_count = value["row_count"]
    if row_count is not None:
        row_count = _require_integer(row_count, "sample row count", minimum=0)
    encoding = value["encoding"]
    if encoding is not None:
        encoding = _require_non_empty_string(encoding, "sample encoding")
    sheet_name = value["sheet_name"]
    if sheet_name is not None:
        sheet_name = _require_non_empty_string(sheet_name, "sample sheet name")
    return SampleSchema(
        path=_require_non_empty_string(value["path"], "sample path"),
        columns=_decode_string_tuple(
            value["columns"], "sample columns", allow_empty=True
        ),
        dtypes=_decode_string_dict(
            value["dtypes"], "sample dtypes", allow_empty_values=False
        ),
        row_count=row_count,
        preview_row_count=_require_integer(
            value["preview_row_count"], "sample preview row count", minimum=0
        ),
        encoding=encoding,
        sha256=_require_non_empty_string(value["sha256"], "sample sha256"),
        sheet_name=sheet_name,
    )


def _decode_feature_metadata_row(payload: object) -> FeatureMetadataRow:
    value = _require_mapping(payload, "feature metadata row")
    _require_exact_keys(
        value,
        {"feature", "category", "importance", "source_sheet", "in_pmml"},
        "feature metadata row",
    )
    source_sheet = value["source_sheet"]
    if source_sheet is not None:
        source_sheet = _require_non_empty_string(source_sheet, "metadata source sheet")
    in_pmml = value["in_pmml"]
    if not isinstance(in_pmml, bool):
        raise ValueError("feature metadata in_pmml must be a boolean")
    return FeatureMetadataRow(
        feature=_require_non_empty_string(value["feature"], "metadata feature"),
        category=_require_non_empty_string(value["category"], "metadata category"),
        importance=_require_finite_number(value["importance"], "metadata importance"),
        source_sheet=source_sheet,
        in_pmml=in_pmml,
    )


def _decode_metadata_coverage(payload: object) -> MetadataCoverage:
    value = _require_mapping(payload, "metadata coverage")
    _require_exact_keys(
        value,
        {"feature", "category", "importance", "stress_unit"},
        "metadata coverage",
    )
    decoded = {
        key: _require_finite_number(value[key], f"metadata {key} coverage")
        for key in ("feature", "category", "importance", "stress_unit")
    }
    if any(not 0.0 <= item <= 1.0 for item in decoded.values()):
        raise ValueError("metadata coverage values must be between 0 and 1")
    return MetadataCoverage(**decoded)


def _decode_feature_metadata_resolution(payload: object) -> FeatureMetadataResolution:
    value = _require_mapping(payload, "feature metadata")
    _require_exact_keys(
        value,
        {
            "schema_version",
            "rows",
            "coverage",
            "per_category_raw_fields",
            "extra_features",
            "conflicts",
        },
        "feature metadata",
    )
    schema_version = _require_non_empty_string(
        value["schema_version"], "feature metadata schema"
    )
    if schema_version != FEATURE_METADATA_SCHEMA:
        raise ValueError("unsupported feature metadata schema")
    category_payload = _require_mapping(
        value["per_category_raw_fields"], "per-category raw fields"
    )
    category_fields = {
        _require_non_empty_string(key, "metadata category"): _decode_string_tuple(
            fields, f"raw fields for {key}", allow_empty=True
        )
        for key, fields in category_payload.items()
    }
    return FeatureMetadataResolution(
        schema_version=schema_version,
        rows=tuple(
            _decode_feature_metadata_row(item)
            for item in _require_sequence(value["rows"], "feature metadata rows")
        ),
        coverage=_decode_metadata_coverage(value["coverage"]),
        per_category_raw_fields=category_fields,
        extra_features=_decode_string_tuple(
            value["extra_features"], "extra metadata features", allow_empty=True
        ),
        conflicts=_decode_string_tuple(
            value["conflicts"], "metadata conflicts", allow_empty=True
        ),
    )


def _require_exact_keys(payload: object, expected: set[str], label: str) -> None:
    value = _require_mapping(payload, label)
    if set(value) != expected:
        raise ValueError(f"invalid {label} keys: {sorted(set(value) ^ expected)}")


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object with string keys")
    return value


def _require_sequence(value: object, label: str) -> list[Any] | tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _require_non_empty_string(value: object, label: str) -> str:
    text = _require_string(value, label).strip()
    if not text:
        raise ValueError(f"{label} must be non-empty")
    return text


def _require_integer(value: object, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _require_finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _decode_string_tuple(
    value: object, label: str, *, allow_empty: bool
) -> tuple[str, ...]:
    result = tuple(
        _require_string(item, label).strip() for item in _require_sequence(value, label)
    )
    if any(not item for item in result):
        raise ValueError(f"{label} values must be non-empty")
    if not allow_empty and not result:
        raise ValueError(f"{label} must be non-empty")
    return result


def _decode_string_dict(
    value: object,
    label: str,
    *,
    allow_empty_values: bool = False,
) -> dict[str, str]:
    payload = _require_mapping(value, label)
    result: dict[str, str] = {}
    for key, item in payload.items():
        decoded_key = _require_non_empty_string(key, f"{label} key")
        decoded_value = _require_string(item, f"{label} value")
        if not allow_empty_values and not decoded_value:
            raise ValueError(f"{label} values must be non-empty")
        result[decoded_key] = decoded_value
    return result


def _decode_json_object(value: object, label: str) -> dict[str, Any]:
    result = _require_mapping(value, label)
    _require_json_value(result)
    return cast(dict[str, Any], _tuples_to_lists(result))


def _require_json_value(value: object) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _require_json_value(item)
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        for item in value.values():
            _require_json_value(item)
        return
    raise ValueError(f"value is not JSON-compatible: {type(value).__name__}")


def _tuples_to_lists(value: object) -> object:
    if isinstance(value, tuple):
        return [_tuples_to_lists(item) for item in value]
    if isinstance(value, list):
        return [_tuples_to_lists(item) for item in value]
    if isinstance(value, dict):
        return {key: _tuples_to_lists(item) for key, item in value.items()}
    return value
