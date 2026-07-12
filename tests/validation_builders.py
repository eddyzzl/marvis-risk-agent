from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256

from marvis.validation.input_contracts import (
    INPUT_CONTRACT_SCHEMA,
    FeatureMetadataResolution,
    FeatureMetadataRow,
    FieldCandidate,
    FieldEvidence,
    MetadataCoverage,
    PmmlInputManifest,
    SampleSchema,
    StressUnit,
    ValidationInputConfirmation,
    ValidationInputContract,
)


def make_candidate_contract(
    *, material_hashes: dict[str, str] | None = None
) -> ValidationInputContract:
    evidence = FieldEvidence("rmc_literal", 0, "RMC_TARGET_COL='y'", 1.0)
    manifest = PmmlInputManifest(
        schema_version="marvis.pmml_input_manifest.v1",
        raw_required_fields=("x1", "x2"),
        derived_fields=(),
        model_features=("x1", "x2"),
        stress_units=(
            StressUnit("x1", ("x1",), ()),
            StressUnit("x2", ("x2",), ()),
        ),
        unsupported_derivations=(),
        output_candidates=("probability_1",),
        algorithm="xgb",
    )
    metadata = FeatureMetadataResolution(
        schema_version="marvis.feature_metadata.v1",
        rows=(
            FeatureMetadataRow("x1", "内部", 0.6, "features", True),
            FeatureMetadataRow("x2", "征信", 0.4, "features", True),
        ),
        coverage=MetadataCoverage(1.0, 1.0, 1.0, 1.0),
        per_category_raw_fields={"内部": ("x1",), "征信": ("x2",)},
        extra_features=(),
        conflicts=(),
    )
    return ValidationInputContract(
        schema_version=INPUT_CONTRACT_SCHEMA,
        material_hashes=material_hashes
        or {
            key: sha256(key.encode("utf-8")).hexdigest()
            for key in ("notebook", "sample", "pmml", "dictionary")
        },
        status="pending_confirmation",
        candidates={"target_col": (FieldCandidate("y", (evidence,)),)},
        sample_schema=SampleSchema(
            path="sample.parquet",
            columns=("x1", "x2", "y", "split", "apply_month"),
            dtypes={},
            row_count=4,
            preview_row_count=4,
            encoding=None,
            sha256="s" * 64,
            sheet_name=None,
        ),
        pmml_manifest=manifest,
        feature_metadata=metadata,
    )


def make_validation_confirmation() -> ValidationInputConfirmation:
    return ValidationInputConfirmation(
        target_col="y",
        positive_label=1,
        negative_label=0,
        split_col="split",
        split_value_mapping={"train": "train", "test": "test", "oot": "oot"},
        time_col="apply_month",
        time_granularity="month",
        pmml_output_field="probability_1",
        model_params={},
        metadata_sheet="features",
        feature_col="feature",
        category_col="category",
        importance_col="importance",
        transformations=(),
    )


def make_ready_contract() -> ValidationInputContract:
    contract = make_candidate_contract()
    confirmation = make_validation_confirmation()
    confirmed = asdict(confirmation)
    confirmed.pop("transformations")
    confirmed["algorithm"] = contract.require_pmml_manifest().algorithm
    return replace(
        contract,
        status="ready",
        confirmed=confirmed,
        transformations=confirmation.transformations,
    )
