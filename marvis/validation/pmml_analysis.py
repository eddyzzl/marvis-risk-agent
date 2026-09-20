"""Authenticated PMML analysis input decoding and read-only material checks."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import stat

import numpy as np
import pandas as pd

from marvis.validation.field_transformations import (
    apply_confirmed_transformations, required_transformation_inputs, transformation_closure,
)
from marvis.validation.input_confirmation import json_scalar_identity, normalize_binary_target
from marvis.validation.input_contracts import ValidationInputContract
from marvis.validation.pmml_score_artifacts import sha256_file_cancellable, validate_pmml_score_artifact
from marvis.validation.results import PmmlScoringResult
from marvis.validation.sample_chunks import read_selected_columns


def load_pmml_analysis_frame(
    *,
    sample_path: Path,
    score_path: Path,
    contract: ValidationInputContract,
    scoring_result: PmmlScoringResult,
    cancellation_check: Callable[[], None] | None = None,
) -> pd.DataFrame:
    """Load only control columns plus the verified PMML score sidecar.

    The resulting frame is intentionally narrow. It is still O(rows) because the
    existing deterministic metrics require all labels, splits, months and scores in
    memory, but it never materializes unrelated model features or a sample-provided
    prediction column.
    """

    _check_cancelled(cancellation_check)
    sample_identity = _file_identity(Path(sample_path), label="validation sample")
    score_identity = _file_identity(Path(score_path), label="PMML score sidecar")
    schema = contract.require_sample_schema()
    _validate_pmml_scoring_identity(
        contract=contract,
        scoring_result=scoring_result,
        sample_path=sample_path,
        score_path=score_path,
        cancellation_check=cancellation_check,
    )

    target_col = _confirmed_field(contract, "target_col")
    split_col = _confirmed_field(contract, "split_col")
    time_col = _confirmed_field(contract, "time_col")
    transformations = transformation_closure(
        (target_col, split_col, time_col),
        contract.transformations,
    )
    projection = tuple(
        dict.fromkeys(
            required_transformation_inputs(
                (target_col, split_col, time_col),
                transformations,
            )
        )
    )
    sample = read_selected_columns(
        Path(sample_path),
        columns=projection,
        schema=schema,
        cancellation_check=cancellation_check,
    )
    _check_cancelled(cancellation_check)
    sample = apply_confirmed_transformations(sample, transformations)
    _check_cancelled(cancellation_check)

    scores = pd.read_parquet(
        Path(score_path),
        columns=["row_id", "pmml_score"],
    )
    _check_cancelled(cancellation_check)
    expected_row_ids = np.arange(len(sample), dtype=np.int64)
    if not np.array_equal(
        scores["row_id"].to_numpy(dtype=np.int64),
        expected_row_ids,
    ):
        raise ValueError("PMML score sidecar row_id does not match validation sample")
    if len(scores) != scoring_result.input_row_count:
        raise ValueError("PMML score sidecar row count does not match scoring result")
    score_values = scores["pmml_score"].to_numpy(dtype=float)
    if not np.isfinite(score_values).all():
        raise ValueError("PMML score sidecar contains a non-finite score")

    confirmed = contract.confirmed
    if "positive_label" not in confirmed or "negative_label" not in confirmed:
        raise ValueError("validation input contract has no confirmed binary labels")
    target = normalize_binary_target(
        sample[target_col],
        positive=confirmed["positive_label"],
        negative=confirmed["negative_label"],
    )
    split = canonical_split_series(
        sample[split_col],
        confirmed.get("split_value_mapping"),
    )
    frame = pd.DataFrame(
        {
            "__target__": target.to_numpy(dtype=np.int8),
            "__split__": split.to_numpy(dtype=object),
            "__time__": sample[time_col].to_numpy(copy=True),
            "__pmml_score__": score_values,
        }
    )

    # The strict boundary already hashed both files. Fingerprints close the
    # subsequent read window without hashing a million-row sample a second time.
    _require_file_identity(
        Path(sample_path), sample_identity, label="validation sample"
    )
    _require_file_identity(
        Path(score_path), score_identity, label="PMML score sidecar"
    )
    _check_cancelled(cancellation_check)
    return frame


def _validate_pmml_scoring_identity(
    *,
    contract: ValidationInputContract,
    scoring_result: PmmlScoringResult,
    sample_path: Path,
    score_path: Path,
    cancellation_check: Callable[[], None] | None,
) -> None:
    expected_pmml = contract.material_hashes.get("pmml")
    if not expected_pmml or scoring_result.pmml_sha256 != expected_pmml:
        raise ValueError("PMML scoring result does not match confirmed PMML")
    if scoring_result.output_field != contract.require_output_field():
        raise ValueError("PMML scoring result output does not match input contract")
    schema = contract.require_sample_schema()
    if (
        schema.row_count is not None
        and scoring_result.input_row_count != schema.row_count
    ):
        raise ValueError("PMML scoring result row count does not match sample schema")
    _require_current_sample_hash(
        sample_path=sample_path,
        contract=contract,
        scoring_result=scoring_result,
        cancellation_check=cancellation_check,
    )
    validate_pmml_score_artifact(
        scoring_result,
        Path(score_path),
        cancellation_check=cancellation_check,
    )


def _require_current_sample_hash(
    *,
    sample_path: Path,
    contract: ValidationInputContract,
    scoring_result: PmmlScoringResult,
    cancellation_check: Callable[[], None] | None,
) -> None:
    expected = contract.material_hashes.get("sample")
    if not expected or scoring_result.sample_sha256 != expected:
        raise ValueError("PMML scoring result does not match confirmed sample")
    current = sha256_file_cancellable(
        Path(sample_path),
        cancellation_check=cancellation_check,
    )
    if current != expected or current != contract.require_sample_schema().sha256:
        raise ValueError("current validation sample does not match confirmed SHA-256")


def _confirmed_field(contract: ValidationInputContract, key: str) -> str:
    value = contract.confirmed.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"validation input contract has no confirmed {key}")
    return value


def validated_split_identity_mapping(value: object) -> dict[tuple[str, str], str]:
    if not isinstance(value, dict) or set(value) != {"train", "test", "oot"}:
        raise ValueError("split_value_mapping must define train/test/oot")
    by_identity: dict[tuple[str, str], str] = {}
    for canonical in ("train", "test", "oot"):
        identity = json_scalar_identity(value[canonical])
        if identity in by_identity:
            raise ValueError("split_value_mapping values must be typed-distinct")
        by_identity[identity] = canonical
    return by_identity


def canonical_split_series(values: pd.Series, mapping: object) -> pd.Series:
    by_identity = validated_split_identity_mapping(mapping)
    canonical: list[str] = []
    for value in values.tolist():
        try:
            identity = json_scalar_identity(value)
        except ValueError as exc:
            raise ValueError("split contains an invalid confirmed value") from exc
        if identity not in by_identity:
            raise ValueError("split contains a value outside confirmed mapping")
        canonical.append(by_identity[identity])
    return pd.Series(canonical, index=values.index, dtype="object")


def _check_cancelled(callback: Callable[[], None] | None) -> None:
    if callback is not None:
        callback()


def _file_identity(path: Path, *, label: str) -> tuple[int, int, int, int, int]:
    try:
        current = path.stat()
    except OSError as exc:
        raise ValueError(f"unable to inspect {label}") from exc
    if not stat.S_ISREG(current.st_mode):
        raise ValueError(f"{label} must be a regular file")
    return (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
        current.st_ctime_ns,
    )


def _require_file_identity(
    path: Path,
    expected: tuple[int, int, int, int, int],
    *,
    label: str,
) -> None:
    if _file_identity(path, label=label) != expected:
        raise ValueError(f"{label} changed while metrics were loading")
