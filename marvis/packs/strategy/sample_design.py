"""Deterministic, content-addressed StrategySampleDesign evidence.

This module deliberately owns no persistence and performs no filtering.  Its
input frame is the already-materialized active dataset boundary.  The bundle
freezes that boundary and emits versioned metric definitions and observations
which downstream strategy tools can bind to without asking an LLM to calculate
or reinterpret sample statistics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal
import hashlib
import hmac
import json
import math
from numbers import Real
import re
from typing import Any

import numpy as np
import pandas as pd

from marvis.packs.strategy.errors import StrategyError


STRATEGY_SAMPLE_DESIGN_BUNDLE_SCHEMA_VERSION = "strategy.sample-design-bundle.v1"
STRATEGY_SAMPLE_DESIGN_SCHEMA_VERSION = "strategy.sample-design.v1"
STRATEGY_METRIC_DEFINITION_SCHEMA_VERSION = "strategy.metric-definition.v1"
STRATEGY_METRIC_OBSERVATION_SCHEMA_VERSION = "strategy.metric-observation.v1"
STRATEGY_SAMPLE_DESIGN_PRODUCER_VERSION = "marvis.strategy.sample-design/1"

MAX_SAMPLE_DESIGN_JSON_DEPTH = 24
MAX_SAMPLE_DESIGN_JSON_NODES = 100_000
MAX_SAMPLE_DESIGN_JSON_BYTES = 16 * 1024 * 1024
MAX_SAMPLE_DESIGN_SPLIT_VALUES = 100
MAX_SAMPLE_DESIGN_SPLIT_STRING_LENGTH = 256

METRIC_OBSERVATION_STATUSES = frozenset(
    {
        "present",
        "unavailable",
        "not_applicable",
        "not_matured",
        "insufficient_data",
    }
)
MATURITY_STATUSES = frozenset(
    {"confirmed_matured", "not_matured", "unknown"}
)

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_BUNDLE_ID_RE = re.compile(r"^strategy-sample-design-bundle-[0-9a-f]{24}$")
_DESIGN_ID_RE = re.compile(r"^strategy-sample-design-[0-9a-f]{24}$")
_DEFINITION_ID_RE = re.compile(r"^metric-definition-[0-9a-f]{24}$")
_OBSERVATION_ID_RE = re.compile(r"^metric-observation-[0-9a-f]{24}$")
_MAX_SAFE_JSON_INTEGER = 2**53 - 1

_BUNDLE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "bundle_id",
        "sample_design",
        "metric_definitions",
        "metric_observations",
        "content_hash",
    }
)
_SAMPLE_DESIGN_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "sample_design_id",
        "identity",
        "target_definition",
        "active_dataset_boundary",
        "performance_window",
        "observation_window",
        "split_definition",
        "split_population_counts",
        "optional_fields",
        "maturity",
        "scope",
        "lifecycle",
        "red_flags",
        "content_hash",
    }
)
_IDENTITY_FIELDS = frozenset({"task_id", "dataset_ref", "workspace_ref"})
_DATASET_REF_FIELDS = frozenset({"dataset_id", "content_hash", "role"})
_WORKSPACE_REF_FIELDS = frozenset(
    {"revision", "generation", "semantic_mapping_hash"}
)
_TARGET_FIELDS = frozenset(
    {"column", "good_value", "bad_value", "drop_nan_labels"}
)
_BOUNDARY_FIELDS = frozenset(
    {
        "status",
        "population_count",
        "inclusion_rules",
        "exclusion_rules",
        "applies_filters",
    }
)
_PERFORMANCE_WINDOW_FIELDS = frozenset({"status", "days"})
_OBSERVATION_WINDOW_FIELDS = frozenset({"status", "start", "end"})
_SPLIT_FIELDS = frozenset(
    {
        "status",
        "column",
        "development_values",
        "validation_values",
        "oot_values",
    }
)
_SPLIT_POPULATION_FIELDS = frozenset({"development", "validation", "oot"})
_OPTIONAL_FIELDS = frozenset(
    {"month_field", "weight_field", "loan_amount_field", "overdue_amount_field"}
)
_LIFECYCLE_FIELDS = frozenset(
    {
        "candidate_stage",
        "validation_status",
        "oot_validation_claimed",
        "creates_strategy",
        "adopted",
        "deployed",
    }
)
_RED_FLAG_FIELDS = frozenset({"code", "level", "message"})
_DEFINITION_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "metric_definition_id",
        "metric_key",
        "display_name",
        "metric_family",
        "basis",
        "numerator_definition",
        "denominator_definition",
        "label_semantics",
        "performance_window",
        "maturity_rule",
        "aggregation",
        "direction",
        "unit",
        "precision",
        "content_hash",
    }
)
_OBSERVATION_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "observation_id",
        "metric_definition_ref",
        "dataset_ref",
        "stage",
        "scope",
        "time_window",
        "dimension",
        "maturity",
        "status",
        "source_refs",
        "value",
        "numerator",
        "denominator",
        "sample_count",
        "content_hash",
    }
)
_DEFINITION_REF_FIELDS = frozenset(
    {"metric_definition_id", "content_hash"}
)
_DIMENSION_FIELDS = frozenset({"kind", "value"})
_SOURCE_REF_FIELDS = frozenset({"kind", "ref_id", "content_hash"})

_EXPECTED_LIFECYCLE = {
    "candidate_stage": "development",
    "validation_status": "unvalidated",
    "oot_validation_claimed": False,
    "creates_strategy": False,
    "adopted": False,
    "deployed": False,
}


class StrategySampleDesignError(StrategyError):
    """The sample-design bundle does not satisfy the exact V1 contract."""


def build_strategy_sample_design_bundle(
    *,
    frame: pd.DataFrame,
    task_id: str,
    dataset_id: str,
    dataset_content_hash: str,
    workspace_revision: int,
    workspace_generation: int,
    semantic_mapping_hash: str,
    target_col: str,
    target_bad_value: int,
    drop_nan_labels: bool,
    performance_window: Mapping[str, Any],
    observation_window: Mapping[str, Any],
    split_definition: Mapping[str, Any],
    maturity: str,
    month_col: str | None = None,
    weight_col: str | None = None,
    loan_amount_col: str | None = None,
    overdue_amount_col: str | None = None,
    producer_version: str = STRATEGY_SAMPLE_DESIGN_PRODUCER_VERSION,
) -> dict[str, Any]:
    """Build one immutable sample-design receipt from the active DataFrame.

    ``frame`` is already the active dataset boundary.  Inclusion/exclusion
    expressions are intentionally not accepted by this kernel.
    """

    producer = _producer_version(producer_version)
    working, columns = _working_frame(
        frame,
        target_col=target_col,
        split_definition=split_definition,
        month_col=month_col,
        weight_col=weight_col,
        loan_amount_col=loan_amount_col,
        overdue_amount_col=overdue_amount_col,
    )
    if not isinstance(drop_nan_labels, bool):
        raise StrategySampleDesignError("drop_nan_labels must be a boolean")
    bad_value = _target_bad_value(target_bad_value)
    target = _target_series(
        working,
        columns["target_col"],
        target_bad_value=bad_value,
    )
    if bool(target.isna().any()) and not drop_nan_labels:
        raise StrategySampleDesignError(
            "target contains missing labels; set drop_nan_labels=true explicitly "
            "to exclude them from risk denominators while retaining population"
        )

    normalized_performance = _performance_window(performance_window)
    normalized_observation = _observation_window(observation_window)
    normalized_split = _split_definition(split_definition)
    split_assignments = _split_assignments(working, normalized_split)
    split_population_counts = _split_population_counts(
        split_assignments,
        normalized_split,
    )
    normalized_maturity = _maturity(maturity)
    optional_fields = {
        "month_field": columns["month_col"],
        "weight_field": columns["weight_col"],
        "loan_amount_field": columns["loan_amount_col"],
        "overdue_amount_field": columns["overdue_amount_col"],
    }
    amounts = {
        "loan_amount": _non_negative_numeric_series(
            working, columns["loan_amount_col"], role="loan_amount"
        ),
        "overdue_amount": _non_negative_numeric_series(
            working, columns["overdue_amount_col"], role="overdue_amount"
        ),
    }
    weight = _non_negative_numeric_series(
        working, columns["weight_col"], role="weight"
    )

    scope = _sample_scope(
        normalized_performance,
        normalized_observation,
        normalized_maturity,
    )
    flags = _expected_red_flags(
        performance_window=normalized_performance,
        observation_window=normalized_observation,
        split_definition=normalized_split,
        split_population_counts=split_population_counts,
        maturity=normalized_maturity,
        missing_label_count=int(target.isna().sum()),
    )
    design_body = {
        "schema_version": STRATEGY_SAMPLE_DESIGN_SCHEMA_VERSION,
        "producer_version": producer,
        "identity": {
            "task_id": _text(task_id, "task_id"),
            "dataset_ref": {
                "dataset_id": _text(dataset_id, "dataset_id"),
                "content_hash": _hash(
                    dataset_content_hash, "dataset_content_hash"
                ),
                "role": "active",
            },
            "workspace_ref": {
                "revision": _non_negative_int(
                    workspace_revision, "workspace_revision"
                ),
                "generation": _non_negative_int(
                    workspace_generation, "workspace_generation"
                ),
                "semantic_mapping_hash": _hash(
                    semantic_mapping_hash, "semantic_mapping_hash"
                ),
            },
        },
        "target_definition": {
            "column": columns["target_col"],
            "good_value": 1 - bad_value,
            "bad_value": bad_value,
            "drop_nan_labels": drop_nan_labels,
        },
        "active_dataset_boundary": {
            "status": "materialized_active_dataset",
            "population_count": len(working),
            "inclusion_rules": ["all_rows_in_active_dataset"],
            "exclusion_rules": ["upstream_exclusions_already_materialized"],
            "applies_filters": False,
        },
        "performance_window": normalized_performance,
        "observation_window": normalized_observation,
        "split_definition": normalized_split,
        "split_population_counts": split_population_counts,
        "optional_fields": optional_fields,
        "maturity": normalized_maturity,
        "scope": scope,
        "lifecycle": dict(_EXPECTED_LIFECYCLE),
        "red_flags": flags,
    }
    sample_design = _address_object(
        design_body,
        id_field="sample_design_id",
        id_prefix="strategy-sample-design-",
    )
    definitions = _build_metric_definitions(
        normalized_performance,
        target_definition=sample_design["target_definition"],
        producer_version=producer,
    )
    definitions_by_key = {item["metric_key"]: item for item in definitions}

    dataset_ref = sample_design["identity"]["dataset_ref"]
    dimensions: list[tuple[dict[str, str], pd.Series]] = [
        (
            {"kind": "overall", "value": "overall"},
            pd.Series(True, index=working.index, dtype=bool),
        )
    ]
    if normalized_split["status"] == "available":
        assert split_assignments is not None
        dimensions.extend(
            (
                {"kind": "split", "value": split_name},
                split_assignments.eq(split_name),
            )
            for split_name in ("development", "validation", "oot")
            if normalized_split[f"{split_name}_values"]
        )

    observations: list[dict[str, Any]] = []
    for dimension, mask in dimensions:
        slice_metrics = _slice_metrics(
            mask,
            target=target,
            amounts=amounts,
            weight=weight,
        )
        slice_metrics = _apply_risk_metric_statuses(
            slice_metrics,
            performance_window=normalized_performance,
            observation_window=normalized_observation,
            maturity=normalized_maturity,
        )
        for metric_key in sorted(definitions_by_key):
            status, value, numerator, denominator = slice_metrics[metric_key]
            observations.append(
                _build_metric_observation(
                    definition=definitions_by_key[metric_key],
                    dataset_ref=dataset_ref,
                    sample_design=sample_design,
                    scope=scope,
                    time_window=normalized_observation,
                    dimension=dimension,
                    maturity=normalized_maturity,
                    status=status,
                    value=value,
                    numerator=numerator,
                    denominator=denominator,
                    sample_count=int(mask.sum()),
                    producer_version=producer,
                )
            )
    observations.sort(
        key=lambda item: (
            item["dimension"]["kind"],
            item["dimension"]["value"],
            _metric_key_for_ref(item["metric_definition_ref"], definitions),
        )
    )

    bundle_body = {
        "schema_version": STRATEGY_SAMPLE_DESIGN_BUNDLE_SCHEMA_VERSION,
        "producer_version": producer,
        "sample_design": sample_design,
        "metric_definitions": definitions,
        "metric_observations": observations,
    }
    bundle = _address_object(
        bundle_body,
        id_field="bundle_id",
        id_prefix="strategy-sample-design-bundle-",
    )
    return validate_strategy_sample_design_bundle(bundle)


def validate_strategy_sample_design_bundle(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly validate hashes, exact shapes, and aggregate conservation."""

    if not isinstance(payload, Mapping):
        raise StrategySampleDesignError("sample-design bundle must be an object")
    _preflight_json_tree(payload, name="sample-design bundle")
    _require_exact_fields(payload, _BUNDLE_FIELDS, name="sample-design bundle")
    if payload["schema_version"] != STRATEGY_SAMPLE_DESIGN_BUNDLE_SCHEMA_VERSION:
        raise StrategySampleDesignError("sample-design bundle schema_version is invalid")
    producer = _producer_version(payload["producer_version"])

    sample_design = _validate_sample_design(payload["sample_design"], producer)
    definitions = _validate_metric_definitions(
        payload["metric_definitions"],
        performance_window=sample_design["performance_window"],
        target_definition=sample_design["target_definition"],
        producer_version=producer,
    )
    observations = _validate_metric_observations(
        payload["metric_observations"],
        sample_design=sample_design,
        definitions=definitions,
        producer_version=producer,
    )
    normalized_body = {
        "schema_version": STRATEGY_SAMPLE_DESIGN_BUNDLE_SCHEMA_VERSION,
        "producer_version": producer,
        "sample_design": sample_design,
        "metric_definitions": definitions,
        "metric_observations": observations,
    }
    normalized = _validate_addressed_object(
        payload,
        normalized_body=normalized_body,
        id_field="bundle_id",
        id_pattern=_BUNDLE_ID_RE,
        id_prefix="strategy-sample-design-bundle-",
        name="sample-design bundle",
    )
    _validate_observation_conservation(normalized)
    if len(_canonical_json(normalized).encode("utf-8")) > MAX_SAMPLE_DESIGN_JSON_BYTES:
        raise StrategySampleDesignError("sample-design bundle exceeds byte budget")
    return normalized


def canonical_strategy_sample_design_bundle_json(
    payload: Mapping[str, Any],
) -> str:
    """Return the sole byte-stable JSON representation of a valid bundle."""

    return _canonical_json(validate_strategy_sample_design_bundle(payload))


def strategy_sample_design_bundle_from_json(
    raw: str | bytes | bytearray,
) -> dict[str, Any]:
    """Load canonical evidence while rejecting duplicate keys and JSON bombs."""

    if not isinstance(raw, (str, bytes, bytearray)):
        raise StrategySampleDesignError("sample-design bundle JSON must be text or bytes")
    raw_size = len(raw.encode("utf-8")) if isinstance(raw, str) else len(raw)
    if raw_size > MAX_SAMPLE_DESIGN_JSON_BYTES:
        raise StrategySampleDesignError("sample-design bundle JSON exceeds byte budget")
    try:
        payload = json.loads(raw, object_pairs_hook=_object_without_duplicate_keys)
    except StrategySampleDesignError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, MemoryError) as exc:
        raise StrategySampleDesignError(
            "sample-design bundle is not valid bounded JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise StrategySampleDesignError("sample-design bundle JSON must contain an object")
    return validate_strategy_sample_design_bundle(payload)


def _working_frame(
    frame: pd.DataFrame,
    *,
    target_col: str,
    split_definition: Mapping[str, Any],
    month_col: str | None,
    weight_col: str | None,
    loan_amount_col: str | None,
    overdue_amount_col: str | None,
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    if not isinstance(frame, pd.DataFrame):
        raise StrategySampleDesignError("sample-design source must be a DataFrame")
    if frame.empty:
        raise StrategySampleDesignError("active dataset must not be empty")
    if frame.columns.duplicated().any():
        raise StrategySampleDesignError("active dataset has duplicate columns")
    normalized_split = _split_definition(split_definition)
    columns: dict[str, str | None] = {
        "target_col": _column(target_col, "target_col"),
        "split_col": normalized_split["column"],
        "month_col": _optional_column(month_col, "month_col"),
        "weight_col": _optional_column(weight_col, "weight_col"),
        "loan_amount_col": _optional_column(loan_amount_col, "loan_amount_col"),
        "overdue_amount_col": _optional_column(
            overdue_amount_col, "overdue_amount_col"
        ),
    }
    selected = [value for value in columns.values() if value is not None]
    if len(selected) != len(set(selected)):
        raise StrategySampleDesignError("sample-design column bindings must be distinct")
    missing = sorted(set(selected) - set(frame.columns))
    if missing:
        raise StrategySampleDesignError(
            "active dataset is missing bound columns: " + ", ".join(missing)
        )
    return frame.reset_index(drop=True), columns


def _target_series(
    frame: pd.DataFrame,
    column: str,
    *,
    target_bad_value: int,
) -> pd.Series:
    values: list[float] = []
    for index, raw in enumerate(frame[column].tolist()):
        if _is_missing_scalar(raw):
            values.append(float("nan"))
            continue
        if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, Real):
            raise StrategySampleDesignError(
                f"target row {index} must contain only numeric 0, 1, or missing"
            )
        number = float(raw)
        if not math.isfinite(number) or number not in {0.0, 1.0}:
            raise StrategySampleDesignError(
                f"target row {index} must contain only 0, 1, or missing"
            )
        values.append(1.0 if number == target_bad_value else 0.0)
    return pd.Series(values, dtype=float)


def _target_bad_value(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) not in {0.0, 1.0}
    ):
        raise StrategySampleDesignError("target_bad_value must be integer 0 or 1")
    return int(value)


def _non_negative_numeric_series(
    frame: pd.DataFrame, column: str | None, *, role: str
) -> pd.Series | None:
    if column is None:
        return None
    values: list[float] = []
    for index, raw in enumerate(frame[column].tolist()):
        if _is_missing_scalar(raw):
            values.append(float("nan"))
            continue
        if isinstance(raw, (bool, np.bool_, complex, np.complexfloating)) or not isinstance(
            raw, (Real, Decimal)
        ):
            raise StrategySampleDesignError(
                f"{role} row {index} must be a non-negative finite number or missing"
            )
        number = float(raw)
        if not math.isfinite(number) or number < 0:
            raise StrategySampleDesignError(
                f"{role} row {index} must be a non-negative finite number or missing"
            )
        values.append(number)
    return pd.Series(values, dtype=float)


def _split_assignments(
    frame: pd.DataFrame, split: Mapping[str, Any]
) -> pd.Series | None:
    if split["status"] == "unavailable":
        return None
    lookup: dict[str, str] = {}
    for split_name in ("development", "validation", "oot"):
        for value in split[f"{split_name}_values"]:
            lookup[_scalar_identity(value)] = split_name
    assignments: list[str] = []
    column = split["column"]
    assert isinstance(column, str)
    for index, raw in enumerate(frame[column].tolist()):
        if _is_missing_scalar(raw):
            raise StrategySampleDesignError(
                f"split column {column} has a missing value at row {index}"
            )
        try:
            normalized = _json_scalar(raw, f"split column {column} row {index}")
        except StrategySampleDesignError as exc:
            raise StrategySampleDesignError(
                f"split column {column} has an unsupported value at row {index}"
            ) from exc
        assigned = lookup.get(_scalar_identity(normalized))
        if assigned is None:
            raise StrategySampleDesignError(
                f"split column {column} has an unknown value at row {index}"
            )
        assignments.append(assigned)
    return pd.Series(assignments, dtype="object")


def _split_population_counts(
    assignments: pd.Series | None,
    split: Mapping[str, Any],
) -> dict[str, int] | None:
    if split["status"] == "unavailable":
        if assignments is not None:
            raise StrategySampleDesignError(
                "unavailable split must not have population assignments"
            )
        return None
    if assignments is None:
        raise StrategySampleDesignError("available split requires population assignments")
    counts = {
        name: int(assignments.eq(name).sum())
        for name in ("development", "validation", "oot")
    }
    if counts["development"] == 0:
        raise StrategySampleDesignError(
            "available split requires at least one development row"
        )
    return counts


def _slice_metrics(
    mask: pd.Series,
    *,
    target: pd.Series,
    amounts: Mapping[str, pd.Series | None],
    weight: pd.Series | None,
) -> dict[str, tuple[str, int | float | None, int | float | None, int | float | None]]:
    selected = mask.reset_index(drop=True).astype(bool)
    population = int(selected.sum())
    labelled_mask = selected & target.notna()
    labelled = int(labelled_mask.sum())
    bad = int(target.loc[labelled_mask].eq(1).sum())
    good = labelled - bad
    output = {
        "population_count": ("present", population, None, None),
        "labeled_count": ("present", labelled, None, None),
        "good_count": ("present", good, None, None),
        "bad_count": ("present", bad, None, None),
        "bad_rate": _ratio_observation(bad, labelled),
        "label_coverage": _ratio_observation(labelled, population),
    }
    for metric_prefix in ("loan_amount", "overdue_amount"):
        values = amounts[metric_prefix]
        if values is None:
            output[f"{metric_prefix}_coverage"] = (
                "unavailable",
                None,
                None,
                None,
            )
            output[f"{metric_prefix}_sum"] = (
                "unavailable",
                None,
                None,
                None,
            )
            continue
        covered = selected & values.notna()
        coverage_count = int(covered.sum())
        output[f"{metric_prefix}_coverage"] = _ratio_observation(
            coverage_count, population
        )
        output[f"{metric_prefix}_sum"] = (
            (
                "present",
                float(values.loc[covered].sum()),
                None,
                None,
            )
            if coverage_count
            else ("insufficient_data", None, None, None)
        )
    if weight is None:
        output["weight_coverage"] = ("unavailable", None, None, None)
        output["weight_sum"] = ("unavailable", None, None, None)
        output["weighted_bad_rate"] = ("unavailable", None, None, None)
    else:
        covered = selected & weight.notna()
        coverage_count = int(covered.sum())
        output["weight_coverage"] = _ratio_observation(
            coverage_count, population
        )
        output["weight_sum"] = (
            ("present", float(weight.loc[covered].sum()), None, None)
            if coverage_count
            else ("insufficient_data", None, None, None)
        )
        weighted_labelled = labelled_mask & weight.notna()
        denominator = float(weight.loc[weighted_labelled].sum())
        numerator = float(
            (weight.loc[weighted_labelled] * target.loc[weighted_labelled]).sum()
        )
        output["weighted_bad_rate"] = _ratio_observation(
            numerator, denominator
        )
    return output


_MATURITY_DEPENDENT_METRIC_KEYS = frozenset(
    {
        "good_count",
        "bad_count",
        "bad_rate",
        "overdue_amount_coverage",
        "overdue_amount_sum",
        "weighted_bad_rate",
    }
)


def _risk_metric_status(
    *,
    performance_window: Mapping[str, Any],
    observation_window: Mapping[str, Any],
    maturity: str,
) -> str:
    if (
        performance_window["status"] != "provided"
        or observation_window["status"] != "provided"
    ):
        return "unavailable"
    if maturity != "confirmed_matured":
        return "not_matured"
    return "present"


def _apply_risk_metric_statuses(
    metrics: Mapping[
        str,
        tuple[str, int | float | None, int | float | None, int | float | None],
    ],
    *,
    performance_window: Mapping[str, Any],
    observation_window: Mapping[str, Any],
    maturity: str,
) -> dict[
    str,
    tuple[str, int | float | None, int | float | None, int | float | None],
]:
    expected = _risk_metric_status(
        performance_window=performance_window,
        observation_window=observation_window,
        maturity=maturity,
    )
    output = dict(metrics)
    if expected == "present":
        return output
    for metric_key in _MATURITY_DEPENDENT_METRIC_KEYS:
        if output[metric_key][0] != "unavailable":
            output[metric_key] = (expected, None, None, None)
    return output


def _ratio_observation(
    numerator: int | float, denominator: int | float
) -> tuple[str, float | None, int | float | None, int | float | None]:
    if denominator == 0:
        return "insufficient_data", None, None, None
    return "present", float(numerator / denominator), numerator, denominator


def _build_metric_definitions(
    performance_window: Mapping[str, Any],
    *,
    target_definition: Mapping[str, Any],
    producer_version: str,
) -> list[dict[str, Any]]:
    definitions = []
    for body in _metric_definition_bodies(
        performance_window,
        target_definition=target_definition,
        producer_version=producer_version,
    ):
        definitions.append(
            _address_object(
                body,
                id_field="metric_definition_id",
                id_prefix="metric-definition-",
            )
        )
    return sorted(definitions, key=lambda item: item["metric_key"])


def _metric_definition_bodies(
    performance_window: Mapping[str, Any],
    *,
    target_definition: Mapping[str, Any],
    producer_version: str,
) -> list[dict[str, Any]]:
    good_value = target_definition["good_value"]
    bad_value = target_definition["bad_value"]
    label_semantics = (
        f"target {good_value}=good, {bad_value}=bad; "
        "missing is never treated as good"
    )
    maturity_rule = "sample maturity is explicit; no OOT validation is implied"
    specs = (
        ("population_count", "Population", "volume", "count", "all active rows", None, None, None, "count", "neutral", "rows", 0),
        ("labeled_count", "Labeled population", "volume", "count", "rows with target 0 or 1", None, label_semantics, None, "count", "neutral", "rows", 0),
        ("good_count", "Good count", "risk", "count", f"labeled rows with target {good_value}", None, label_semantics, maturity_rule, "count", "higher_is_better", "rows", 0),
        ("bad_count", "Bad count", "risk", "count", f"labeled rows with target {bad_value}", None, label_semantics, maturity_rule, "count", "higher_is_worse", "rows", 0),
        ("bad_rate", "Bad rate", "risk", "count", "bad_count", "labeled_count", label_semantics, maturity_rule, "ratio", "higher_is_worse", "ratio", 12),
        ("label_coverage", "Label coverage", "stability", "count", "labeled_count", "population_count", label_semantics, None, "ratio", "higher_is_better", "ratio", 12),
        ("loan_amount_coverage", "Loan amount coverage", "exposure", "amount", "rows with loan amount", "population_count", None, None, "ratio", "higher_is_better", "ratio", 12),
        ("loan_amount_sum", "Loan amount sum", "exposure", "amount", "sum of non-missing loan amount", None, None, None, "sum", "neutral", "currency_units", 12),
        ("overdue_amount_coverage", "Overdue amount coverage", "risk", "amount", "rows with overdue amount", "population_count", None, maturity_rule, "ratio", "higher_is_better", "ratio", 12),
        ("overdue_amount_sum", "Overdue amount sum", "risk", "amount", "sum of non-missing overdue amount", None, None, maturity_rule, "sum", "higher_is_worse", "currency_units", 12),
        ("weight_coverage", "Weight coverage", "stability", "count", "rows with weight", "population_count", None, None, "ratio", "higher_is_better", "ratio", 12),
        ("weight_sum", "Weight sum", "volume", "count", "sum of non-missing non-negative weight", None, None, None, "sum", "neutral", "weight_units", 12),
        ("weighted_bad_rate", "Weighted bad rate", "risk", "count", "sum(weight * bad-indicator) over labeled weighted rows", "sum(weight) over labeled weighted rows", label_semantics, maturity_rule, "ratio", "higher_is_worse", "ratio", 12),
    )
    bodies = []
    risk_keys = {
        "good_count",
        "bad_count",
        "bad_rate",
        "overdue_amount_coverage",
        "overdue_amount_sum",
        "weighted_bad_rate",
    }
    for (
        key,
        display,
        family,
        basis,
        numerator,
        denominator,
        semantics,
        rule,
        aggregation,
        direction,
        unit,
        precision,
    ) in specs:
        bodies.append(
            {
                "schema_version": STRATEGY_METRIC_DEFINITION_SCHEMA_VERSION,
                "producer_version": producer_version,
                "metric_key": key,
                "display_name": display,
                "metric_family": family,
                "basis": basis,
                "numerator_definition": numerator,
                "denominator_definition": denominator,
                "label_semantics": semantics,
                "performance_window": dict(performance_window)
                if key in risk_keys
                else None,
                "maturity_rule": rule,
                "aggregation": aggregation,
                "direction": direction,
                "unit": unit,
                "precision": precision,
            }
        )
    return bodies


def _build_metric_observation(
    *,
    definition: Mapping[str, Any],
    dataset_ref: Mapping[str, Any],
    sample_design: Mapping[str, Any],
    scope: str,
    time_window: Mapping[str, Any],
    dimension: Mapping[str, str],
    maturity: str,
    status: str,
    value: int | float | None,
    numerator: int | float | None,
    denominator: int | float | None,
    sample_count: int,
    producer_version: str,
) -> dict[str, Any]:
    body = {
        "schema_version": STRATEGY_METRIC_OBSERVATION_SCHEMA_VERSION,
        "producer_version": producer_version,
        "metric_definition_ref": {
            "metric_definition_id": definition["metric_definition_id"],
            "content_hash": definition["content_hash"],
        },
        "dataset_ref": dict(dataset_ref),
        "stage": "development",
        "scope": scope,
        "time_window": dict(time_window),
        "dimension": dict(dimension),
        "maturity": maturity,
        "status": status,
        "source_refs": [
            {
                "kind": "dataset",
                "ref_id": dataset_ref["dataset_id"],
                "content_hash": dataset_ref["content_hash"],
            },
            {
                "kind": "sample_design",
                "ref_id": sample_design["sample_design_id"],
                "content_hash": sample_design["content_hash"],
            },
        ],
        "value": value,
        "numerator": numerator,
        "denominator": denominator,
        "sample_count": sample_count,
    }
    return _address_object(
        body,
        id_field="observation_id",
        id_prefix="metric-observation-",
    )


def _validate_sample_design(value: object, producer_version: str) -> dict[str, Any]:
    obj = _object(value, "sample_design")
    _require_exact_fields(obj, _SAMPLE_DESIGN_FIELDS, name="sample_design")
    if obj["schema_version"] != STRATEGY_SAMPLE_DESIGN_SCHEMA_VERSION:
        raise StrategySampleDesignError("sample_design schema_version is invalid")
    if obj["producer_version"] != producer_version:
        raise StrategySampleDesignError("sample_design producer_version is inconsistent")
    identity = _identity(obj["identity"])
    target = _target_definition(obj["target_definition"])
    boundary = _active_boundary(obj["active_dataset_boundary"])
    performance = _performance_window(obj["performance_window"])
    observation = _observation_window(obj["observation_window"])
    split = _split_definition(obj["split_definition"])
    split_population_counts = _validate_split_population_counts(
        obj["split_population_counts"],
        split_definition=split,
    )
    optional = _optional_fields(obj["optional_fields"])
    bound_columns = [target["column"]]
    if split["column"] is not None:
        bound_columns.append(split["column"])
    bound_columns.extend(item for item in optional.values() if item is not None)
    if len(bound_columns) != len(set(bound_columns)):
        raise StrategySampleDesignError(
            "sample_design target, split, and optional column bindings must be distinct"
        )
    maturity = _maturity(obj["maturity"])
    scope = obj["scope"]
    expected_scope = _sample_scope(performance, observation, maturity)
    if scope != expected_scope:
        raise StrategySampleDesignError("sample_design scope is inconsistent")
    if obj["lifecycle"] != _EXPECTED_LIFECYCLE:
        raise StrategySampleDesignError(
            "sample_design lifecycle must remain development/unvalidated"
        )
    flags = _red_flags(obj["red_flags"])
    missing_count = _flag_missing_label_count(flags)
    expected_flags = _expected_red_flags(
        performance_window=performance,
        observation_window=observation,
        split_definition=split,
        split_population_counts=split_population_counts,
        maturity=maturity,
        missing_label_count=missing_count,
    )
    if flags != expected_flags:
        raise StrategySampleDesignError("sample_design red_flags are inconsistent")
    body = {
        "schema_version": STRATEGY_SAMPLE_DESIGN_SCHEMA_VERSION,
        "producer_version": producer_version,
        "identity": identity,
        "target_definition": target,
        "active_dataset_boundary": boundary,
        "performance_window": performance,
        "observation_window": observation,
        "split_definition": split,
        "split_population_counts": split_population_counts,
        "optional_fields": optional,
        "maturity": maturity,
        "scope": scope,
        "lifecycle": dict(_EXPECTED_LIFECYCLE),
        "red_flags": flags,
    }
    return _validate_addressed_object(
        obj,
        normalized_body=body,
        id_field="sample_design_id",
        id_pattern=_DESIGN_ID_RE,
        id_prefix="strategy-sample-design-",
        name="sample_design",
    )


def _validate_metric_definitions(
    value: object,
    *,
    performance_window: Mapping[str, Any],
    target_definition: Mapping[str, Any],
    producer_version: str,
) -> list[dict[str, Any]]:
    items = _array(value, "metric_definitions", required=True)
    normalized = [
        _validate_metric_definition(item, producer_version) for item in items
    ]
    keys = [item["metric_key"] for item in normalized]
    if len(keys) != len(set(keys)):
        raise StrategySampleDesignError("metric_definitions contain duplicate keys")
    normalized.sort(key=lambda item: item["metric_key"])
    expected = _build_metric_definitions(
        performance_window,
        target_definition=target_definition,
        producer_version=producer_version,
    )
    if normalized != expected:
        raise StrategySampleDesignError(
            "metric_definitions do not match the sample metric contract"
        )
    return normalized


def _validate_metric_definition(
    value: object, producer_version: str
) -> dict[str, Any]:
    obj = _object(value, "metric_definition")
    _require_exact_fields(obj, _DEFINITION_FIELDS, name="metric_definition")
    if obj["schema_version"] != STRATEGY_METRIC_DEFINITION_SCHEMA_VERSION:
        raise StrategySampleDesignError("metric_definition schema_version is invalid")
    if obj["producer_version"] != producer_version:
        raise StrategySampleDesignError("metric_definition producer_version is inconsistent")
    body = {
        "schema_version": STRATEGY_METRIC_DEFINITION_SCHEMA_VERSION,
        "producer_version": producer_version,
        "metric_key": _text(obj["metric_key"], "metric_definition.metric_key"),
        "display_name": _text(
            obj["display_name"], "metric_definition.display_name"
        ),
        "metric_family": _enum(
            obj["metric_family"],
            {"volume", "approval", "drawdown", "risk", "pricing", "exposure", "cost", "profit", "stability"},
            "metric_definition.metric_family",
        ),
        "basis": _enum(
            obj["basis"], {"count", "amount", "balance"}, "metric_definition.basis"
        ),
        "numerator_definition": _text(
            obj["numerator_definition"], "metric_definition.numerator_definition"
        ),
        "denominator_definition": _optional_text(
            obj["denominator_definition"], "metric_definition.denominator_definition"
        ),
        "label_semantics": _optional_text(
            obj["label_semantics"], "metric_definition.label_semantics"
        ),
        "performance_window": None
        if obj["performance_window"] is None
        else _performance_window(obj["performance_window"]),
        "maturity_rule": _optional_text(
            obj["maturity_rule"], "metric_definition.maturity_rule"
        ),
        "aggregation": _enum(
            obj["aggregation"],
            {"ratio", "sum", "mean", "median", "quantile", "count"},
            "metric_definition.aggregation",
        ),
        "direction": _enum(
            obj["direction"],
            {"higher_is_worse", "higher_is_better", "neutral"},
            "metric_definition.direction",
        ),
        "unit": _text(obj["unit"], "metric_definition.unit"),
        "precision": _non_negative_int(
            obj["precision"], "metric_definition.precision"
        ),
    }
    return _validate_addressed_object(
        obj,
        normalized_body=body,
        id_field="metric_definition_id",
        id_pattern=_DEFINITION_ID_RE,
        id_prefix="metric-definition-",
        name="metric_definition",
    )


def _validate_metric_observations(
    value: object,
    *,
    sample_design: Mapping[str, Any],
    definitions: Sequence[Mapping[str, Any]],
    producer_version: str,
) -> list[dict[str, Any]]:
    items = _array(value, "metric_observations", required=True)
    by_id = {item["metric_definition_id"]: item for item in definitions}
    normalized = [
        _validate_metric_observation(
            item,
            sample_design=sample_design,
            definitions_by_id=by_id,
            producer_version=producer_version,
        )
        for item in items
    ]
    identities = [
        (
            item["metric_definition_ref"]["metric_definition_id"],
            item["dimension"]["kind"],
            item["dimension"]["value"],
        )
        for item in normalized
    ]
    if len(identities) != len(set(identities)):
        raise StrategySampleDesignError("metric_observations contain duplicate identities")
    normalized.sort(
        key=lambda item: (
            item["dimension"]["kind"],
            item["dimension"]["value"],
            by_id[item["metric_definition_ref"]["metric_definition_id"]]["metric_key"],
        )
    )
    expected_dimensions = {("overall", "overall")}
    if sample_design["split_definition"]["status"] == "available":
        split_definition = sample_design["split_definition"]
        expected_dimensions |= {
            ("split", name)
            for name in ("development", "validation", "oot")
            if split_definition[f"{name}_values"]
        }
    expected = {
        (definition["metric_definition_id"], kind, dimension)
        for definition in definitions
        for kind, dimension in expected_dimensions
    }
    if set(identities) != expected:
        raise StrategySampleDesignError(
            "metric_observations do not cover every metric and sample dimension"
        )
    return normalized


def _validate_metric_observation(
    value: object,
    *,
    sample_design: Mapping[str, Any],
    definitions_by_id: Mapping[str, Mapping[str, Any]],
    producer_version: str,
) -> dict[str, Any]:
    obj = _object(value, "metric_observation")
    _require_exact_fields(obj, _OBSERVATION_FIELDS, name="metric_observation")
    if obj["schema_version"] != STRATEGY_METRIC_OBSERVATION_SCHEMA_VERSION:
        raise StrategySampleDesignError("metric_observation schema_version is invalid")
    if obj["producer_version"] != producer_version:
        raise StrategySampleDesignError("metric_observation producer_version is inconsistent")
    ref = _definition_ref(obj["metric_definition_ref"])
    definition = definitions_by_id.get(ref["metric_definition_id"])
    if definition is None or ref["content_hash"] != definition["content_hash"]:
        raise StrategySampleDesignError("metric_observation definition ref is invalid")
    dataset_ref = _dataset_ref(obj["dataset_ref"])
    if dataset_ref != sample_design["identity"]["dataset_ref"]:
        raise StrategySampleDesignError("metric_observation dataset ref is inconsistent")
    if obj["stage"] != "development":
        raise StrategySampleDesignError("metric_observation stage must be development")
    if obj["scope"] != sample_design["scope"]:
        raise StrategySampleDesignError("metric_observation scope is inconsistent")
    time_window = _observation_window(obj["time_window"])
    if time_window != sample_design["observation_window"]:
        raise StrategySampleDesignError("metric_observation time window is inconsistent")
    dimension = _dimension(obj["dimension"])
    maturity = _maturity(obj["maturity"])
    if maturity != sample_design["maturity"]:
        raise StrategySampleDesignError("metric_observation maturity is inconsistent")
    status = _enum(
        obj["status"], METRIC_OBSERVATION_STATUSES, "metric_observation.status"
    )
    metric_value = _optional_finite_number(obj["value"], "metric_observation.value")
    numerator = _optional_finite_number(
        obj["numerator"], "metric_observation.numerator"
    )
    denominator = _optional_finite_number(
        obj["denominator"], "metric_observation.denominator"
    )
    if status == "present":
        if metric_value is None:
            raise StrategySampleDesignError(
                "present metric_observation must have a value"
            )
    elif any(item is not None for item in (metric_value, numerator, denominator)):
        raise StrategySampleDesignError(
            "non-present metric_observation must have null value and operands"
        )
    sample_count = _non_negative_int(
        obj["sample_count"], "metric_observation.sample_count"
    )
    expected_sources = [
        {
            "kind": "dataset",
            "ref_id": dataset_ref["dataset_id"],
            "content_hash": dataset_ref["content_hash"],
        },
        {
            "kind": "sample_design",
            "ref_id": sample_design["sample_design_id"],
            "content_hash": sample_design["content_hash"],
        },
    ]
    sources = _source_refs(obj["source_refs"])
    if sources != expected_sources:
        raise StrategySampleDesignError("metric_observation source_refs are inconsistent")
    body = {
        "schema_version": STRATEGY_METRIC_OBSERVATION_SCHEMA_VERSION,
        "producer_version": producer_version,
        "metric_definition_ref": ref,
        "dataset_ref": dataset_ref,
        "stage": "development",
        "scope": sample_design["scope"],
        "time_window": time_window,
        "dimension": dimension,
        "maturity": maturity,
        "status": status,
        "source_refs": sources,
        "value": metric_value,
        "numerator": numerator,
        "denominator": denominator,
        "sample_count": sample_count,
    }
    return _validate_addressed_object(
        obj,
        normalized_body=body,
        id_field="observation_id",
        id_pattern=_OBSERVATION_ID_RE,
        id_prefix="metric-observation-",
        name="metric_observation",
    )


def _validate_observation_conservation(bundle: Mapping[str, Any]) -> None:
    definitions = {
        item["metric_definition_id"]: item["metric_key"]
        for item in bundle["metric_definitions"]
    }
    grouped: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    for observation in bundle["metric_observations"]:
        dimension = observation["dimension"]
        key = (dimension["kind"], dimension["value"])
        grouped.setdefault(key, {})[
            definitions[observation["metric_definition_ref"]["metric_definition_id"]]
        ] = observation
    sample_design = bundle["sample_design"]
    expected_risk_status = _risk_metric_status(
        performance_window=sample_design["performance_window"],
        observation_window=sample_design["observation_window"],
        maturity=sample_design["maturity"],
    )
    for dimension, metrics in grouped.items():
        population = _present_non_negative_int(metrics["population_count"], dimension)
        labeled = _present_non_negative_int(metrics["labeled_count"], dimension)
        if labeled > population:
            raise StrategySampleDesignError("sample count conservation failed")
        if expected_risk_status == "present":
            good = _present_non_negative_int(metrics["good_count"], dimension)
            bad = _present_non_negative_int(metrics["bad_count"], dimension)
            if good + bad != labeled:
                raise StrategySampleDesignError("sample count conservation failed")
            _require_ratio(metrics["bad_rate"], bad, labeled, "bad_rate")
        else:
            for metric_key in ("good_count", "bad_count", "bad_rate"):
                _require_status(
                    metrics[metric_key],
                    expected_risk_status,
                    metric_key,
                )
        for observation in metrics.values():
            if observation["sample_count"] != population:
                raise StrategySampleDesignError("metric sample_count is inconsistent")
        _require_ratio(
            metrics["label_coverage"], labeled, population, "label_coverage"
        )
        optional = sample_design["optional_fields"]
        _require_optional_coverage_and_sum(
            metrics,
            prefix="loan_amount",
            available=optional["loan_amount_field"] is not None,
            population=population,
        )
        if optional["overdue_amount_field"] is None:
            _require_optional_coverage_and_sum(
                metrics,
                prefix="overdue_amount",
                available=False,
                population=population,
            )
        elif expected_risk_status == "present":
            _require_optional_coverage_and_sum(
                metrics,
                prefix="overdue_amount",
                available=True,
                population=population,
            )
        else:
            _require_status(
                metrics["overdue_amount_coverage"],
                expected_risk_status,
                "overdue_amount_coverage",
            )
            _require_status(
                metrics["overdue_amount_sum"],
                expected_risk_status,
                "overdue_amount_sum",
            )
        _require_optional_weight(
            metrics,
            available=optional["weight_field"] is not None,
            population=population,
            expected_risk_status=expected_risk_status,
        )

    overall = grouped[("overall", "overall")]
    overall_population = int(overall["population_count"]["value"])
    if overall_population != bundle["sample_design"]["active_dataset_boundary"][
        "population_count"
    ]:
        raise StrategySampleDesignError("overall population does not match boundary")
    overall_labeled = int(overall["labeled_count"]["value"])
    missing_labels = overall_population - overall_labeled
    flags = bundle["sample_design"]["red_flags"]
    if _flag_missing_label_count(flags) != missing_labels:
        raise StrategySampleDesignError(
            "missing-label red flag does not match overall label coverage"
        )
    if missing_labels and not bundle["sample_design"]["target_definition"][
        "drop_nan_labels"
    ]:
        raise StrategySampleDesignError(
            "missing labels require the explicit drop_nan_labels policy"
        )
    split_names = ("development", "validation", "oot")
    split_definition = sample_design["split_definition"]
    if split_definition["status"] == "available":
        split_counts = sample_design["split_population_counts"]
        if split_counts is None:
            raise StrategySampleDesignError("available split counts are missing")
        emitted_names = tuple(
            name for name in split_names if split_definition[f"{name}_values"]
        )
        splits = [grouped[("split", name)] for name in emitted_names]
        for name in split_names:
            metrics = grouped.get(("split", name))
            if not split_definition[f"{name}_values"]:
                if split_counts[name] != 0 or metrics is not None:
                    raise StrategySampleDesignError(
                        f"unavailable split {name} must not emit observations"
                    )
                continue
            if metrics is None:
                raise StrategySampleDesignError(
                    f"available split {name} observations are missing"
                )
            if metrics["population_count"]["value"] != split_counts[name]:
                raise StrategySampleDesignError(
                    f"split {name} population does not match sample design"
                )
        if sum(split_counts.values()) != overall_population:
            raise StrategySampleDesignError("split populations do not conserve overall")
        additive = (
            "population_count",
            "labeled_count",
            "good_count",
            "bad_count",
            "loan_amount_sum",
            "overdue_amount_sum",
            "weight_sum",
        )
        for metric_key in additive:
            overall_metric = overall[metric_key]
            if overall_metric["status"] != "present":
                if any(
                    item[metric_key]["status"] != overall_metric["status"]
                    for item in splits
                ):
                    raise StrategySampleDesignError("split availability is inconsistent")
                continue
            _require_additive_values(
                overall_metric,
                [item[metric_key] for item in splits],
                metric_key,
            )
        for metric_key in (
            "label_coverage",
            "loan_amount_coverage",
            "overdue_amount_coverage",
            "weight_coverage",
            "weighted_bad_rate",
        ):
            _require_additive_operands(
                overall[metric_key],
                [item[metric_key] for item in splits],
                metric_key,
            )


def _require_optional_coverage_and_sum(
    metrics: Mapping[str, Mapping[str, Any]],
    *,
    prefix: str,
    available: bool,
    population: int,
) -> None:
    coverage = metrics[f"{prefix}_coverage"]
    total = metrics[f"{prefix}_sum"]
    if not available:
        if coverage["status"] != "unavailable" or total["status"] != "unavailable":
            raise StrategySampleDesignError(f"{prefix} metrics must be unavailable")
        return
    _require_bounded_coverage(coverage, population, f"{prefix}_coverage")
    coverage_count = coverage["numerator"] if coverage["status"] == "present" else 0
    if coverage_count == 0:
        _require_status(total, "insufficient_data", f"{prefix}_sum")
        return
    if (
        total["status"] != "present"
        or float(total["value"]) < 0
        or total["numerator"] is not None
        or total["denominator"] is not None
    ):
        raise StrategySampleDesignError(f"{prefix} sum is invalid")


def _require_optional_weight(
    metrics: Mapping[str, Mapping[str, Any]],
    *,
    available: bool,
    population: int,
    expected_risk_status: str,
) -> None:
    coverage = metrics["weight_coverage"]
    total = metrics["weight_sum"]
    rate = metrics["weighted_bad_rate"]
    if not available:
        if any(
            item["status"] != "unavailable" for item in (coverage, total, rate)
        ):
            raise StrategySampleDesignError("weight metrics must be unavailable")
        return
    _require_bounded_coverage(coverage, population, "weight_coverage")
    coverage_count = coverage["numerator"] if coverage["status"] == "present" else 0
    if coverage_count == 0:
        _require_status(total, "insufficient_data", "weight_sum")
    elif (
        total["status"] != "present"
        or float(total["value"]) < 0
        or total["numerator"] is not None
        or total["denominator"] is not None
    ):
        raise StrategySampleDesignError("weight_sum is invalid")
    if expected_risk_status != "present":
        _require_status(rate, expected_risk_status, "weighted_bad_rate")
        return
    if rate["status"] == "present":
        value = float(rate["value"])
        if not 0 <= value <= 1:
            raise StrategySampleDesignError("weighted_bad_rate must be in [0, 1]")
        if rate["denominator"] is None or float(rate["denominator"]) <= 0:
            raise StrategySampleDesignError("weighted_bad_rate denominator is invalid")
        if rate["numerator"] is None or not 0 <= float(rate["numerator"]) <= float(
            rate["denominator"]
        ):
            raise StrategySampleDesignError("weighted_bad_rate numerator is invalid")
        _require_same_number(
            value,
            float(rate["numerator"]) / float(rate["denominator"]),
            "weighted_bad_rate",
        )
    elif rate["status"] != "insufficient_data":
        raise StrategySampleDesignError("weighted_bad_rate status is invalid")


def _require_status(
    observation: Mapping[str, Any], expected_status: str, name: str
) -> None:
    if observation["status"] != expected_status:
        raise StrategySampleDesignError(
            f"{name} must have status {expected_status}"
        )
    if any(
        observation[field] is not None
        for field in ("value", "numerator", "denominator")
    ):
        raise StrategySampleDesignError(f"{name} non-present operands must be null")


def _require_additive_values(
    overall: Mapping[str, Any],
    splits: Sequence[Mapping[str, Any]],
    name: str,
) -> None:
    if any(item["status"] not in {"present", "insufficient_data"} for item in splits):
        raise StrategySampleDesignError(f"split {name} status is inconsistent")
    values = [
        0 if item["status"] == "insufficient_data" else item["value"]
        for item in splits
    ]
    if any(value is None for value in values) or overall["value"] is None:
        raise StrategySampleDesignError(f"split {name} values are missing")
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        matches = overall["value"] == sum(values)
    else:
        total = math.fsum(float(value) for value in values)
        matches = math.isclose(
            float(overall["value"]),
            total,
            rel_tol=1e-15,
            abs_tol=max(1e-12, math.ulp(total) * 4),
        )
    if not matches:
        raise StrategySampleDesignError(f"split {name} does not conserve overall")


def _require_additive_operands(
    overall: Mapping[str, Any],
    splits: Sequence[Mapping[str, Any]],
    name: str,
) -> None:
    if overall["status"] != "present":
        if any(item["status"] != overall["status"] for item in splits):
            raise StrategySampleDesignError(f"split {name} status is inconsistent")
        return
    if any(item["status"] not in {"present", "insufficient_data"} for item in splits):
        raise StrategySampleDesignError(f"split {name} status is inconsistent")
    for field in ("numerator", "denominator"):
        expected = overall[field]
        values = [
            0 if item["status"] == "insufficient_data" else item[field]
            for item in splits
        ]
        if expected is None or any(value is None for value in values):
            raise StrategySampleDesignError(f"split {name} operands are missing")
        if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            matches = expected == sum(values)
        else:
            total = math.fsum(float(value) for value in values)
            matches = math.isclose(
                float(expected),
                total,
                rel_tol=1e-15,
                abs_tol=max(1e-12, math.ulp(total) * 4),
            )
        if not matches:
            raise StrategySampleDesignError(
                f"split {name} {field} does not conserve overall"
            )


def _require_bounded_coverage(
    observation: Mapping[str, Any], population: int, name: str
) -> None:
    if population == 0:
        if observation["status"] != "insufficient_data":
            raise StrategySampleDesignError(f"{name} must be insufficient_data")
        return
    if observation["status"] != "present":
        raise StrategySampleDesignError(f"{name} must be present")
    numerator = observation["numerator"]
    denominator = observation["denominator"]
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator != population
        or not 0 <= numerator <= population
    ):
        raise StrategySampleDesignError(f"{name} operands are invalid")
    _require_same_number(
        observation["value"], float(numerator) / population, name
    )


def _require_ratio(
    observation: Mapping[str, Any],
    numerator: int | float,
    denominator: int | float,
    name: str,
) -> None:
    if denominator == 0:
        if observation["status"] != "insufficient_data":
            raise StrategySampleDesignError(f"{name} must be insufficient_data")
        return
    if observation["status"] != "present":
        raise StrategySampleDesignError(f"{name} must be present")
    if observation["numerator"] != numerator or observation["denominator"] != denominator:
        raise StrategySampleDesignError(f"{name} operands are inconsistent")
    _require_same_number(observation["value"], numerator / denominator, name)


def _present_non_negative_int(
    observation: Mapping[str, Any], dimension: tuple[str, str]
) -> int:
    if observation["status"] != "present":
        raise StrategySampleDesignError(f"count metric is not present for {dimension}")
    value = observation["value"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StrategySampleDesignError("count metric must be a non-negative integer")
    if observation["numerator"] is not None or observation["denominator"] is not None:
        raise StrategySampleDesignError("count metric operands must be null")
    return value


def _identity(value: object) -> dict[str, Any]:
    obj = _object(value, "sample_design.identity")
    _require_exact_fields(obj, _IDENTITY_FIELDS, name="sample_design.identity")
    return {
        "task_id": _text(obj["task_id"], "sample_design.identity.task_id"),
        "dataset_ref": _dataset_ref(obj["dataset_ref"]),
        "workspace_ref": _workspace_ref(obj["workspace_ref"]),
    }


def _dataset_ref(value: object) -> dict[str, Any]:
    obj = _object(value, "dataset_ref")
    _require_exact_fields(obj, _DATASET_REF_FIELDS, name="dataset_ref")
    if obj["role"] != "active":
        raise StrategySampleDesignError("dataset_ref.role must be active")
    return {
        "dataset_id": _text(obj["dataset_id"], "dataset_ref.dataset_id"),
        "content_hash": _hash(obj["content_hash"], "dataset_ref.content_hash"),
        "role": "active",
    }


def _workspace_ref(value: object) -> dict[str, Any]:
    obj = _object(value, "workspace_ref")
    _require_exact_fields(obj, _WORKSPACE_REF_FIELDS, name="workspace_ref")
    return {
        "revision": _non_negative_int(obj["revision"], "workspace_ref.revision"),
        "generation": _non_negative_int(
            obj["generation"], "workspace_ref.generation"
        ),
        "semantic_mapping_hash": _hash(
            obj["semantic_mapping_hash"], "workspace_ref.semantic_mapping_hash"
        ),
    }


def _target_definition(value: object) -> dict[str, Any]:
    obj = _object(value, "target_definition")
    _require_exact_fields(obj, _TARGET_FIELDS, name="target_definition")
    good_value = _target_bad_value(obj["good_value"])
    bad_value = _target_bad_value(obj["bad_value"])
    if good_value == bad_value or {good_value, bad_value} != {0, 1}:
        raise StrategySampleDesignError(
            "target_definition good_value and bad_value must be complementary 0/1"
        )
    if not isinstance(obj["drop_nan_labels"], bool):
        raise StrategySampleDesignError("target_definition.drop_nan_labels must be boolean")
    return {
        "column": _column(obj["column"], "target_definition.column"),
        "good_value": good_value,
        "bad_value": bad_value,
        "drop_nan_labels": obj["drop_nan_labels"],
    }


def _active_boundary(value: object) -> dict[str, Any]:
    obj = _object(value, "active_dataset_boundary")
    _require_exact_fields(obj, _BOUNDARY_FIELDS, name="active_dataset_boundary")
    expected_constants = {
        "status": "materialized_active_dataset",
        "inclusion_rules": ["all_rows_in_active_dataset"],
        "exclusion_rules": ["upstream_exclusions_already_materialized"],
        "applies_filters": False,
    }
    for key, expected in expected_constants.items():
        if obj[key] != expected:
            raise StrategySampleDesignError(
                "active_dataset_boundary may only describe the already-materialized boundary"
            )
    return {
        "status": "materialized_active_dataset",
        "population_count": _positive_int(
            obj["population_count"], "active_dataset_boundary.population_count"
        ),
        "inclusion_rules": ["all_rows_in_active_dataset"],
        "exclusion_rules": ["upstream_exclusions_already_materialized"],
        "applies_filters": False,
    }


def _performance_window(value: object) -> dict[str, Any]:
    obj = _object(value, "performance_window")
    _require_exact_fields(obj, _PERFORMANCE_WINDOW_FIELDS, name="performance_window")
    status = _enum(obj["status"], {"provided", "unavailable"}, "performance_window.status")
    if status == "provided":
        days = _positive_int(obj["days"], "performance_window.days")
    else:
        if obj["days"] is not None:
            raise StrategySampleDesignError(
                "unavailable performance_window must have null days"
            )
        days = None
    return {"status": status, "days": days}


def _observation_window(value: object) -> dict[str, Any]:
    obj = _object(value, "observation_window")
    _require_exact_fields(obj, _OBSERVATION_WINDOW_FIELDS, name="observation_window")
    status = _enum(obj["status"], {"provided", "unavailable"}, "observation_window.status")
    if status == "unavailable":
        if obj["start"] is not None or obj["end"] is not None:
            raise StrategySampleDesignError(
                "unavailable observation_window must have null bounds"
            )
        return {"status": status, "start": None, "end": None}
    start = _iso_date(obj["start"], "observation_window.start")
    end = _iso_date(obj["end"], "observation_window.end")
    if start > end:
        raise StrategySampleDesignError(
            "observation_window.start must not be after end"
        )
    return {"status": status, "start": start.isoformat(), "end": end.isoformat()}


def _split_definition(value: object) -> dict[str, Any]:
    obj = _object(value, "split_definition")
    _require_exact_fields(obj, _SPLIT_FIELDS, name="split_definition")
    status = _enum(obj["status"], {"available", "unavailable"}, "split_definition.status")
    if status == "unavailable":
        if obj["column"] is not None or any(
            obj[key] != []
            for key in ("development_values", "validation_values", "oot_values")
        ):
            raise StrategySampleDesignError(
                "unavailable split_definition must have null column and empty value lists"
            )
        return {
            "status": "unavailable",
            "column": None,
            "development_values": [],
            "validation_values": [],
            "oot_values": [],
        }
    column = _column(obj["column"], "split_definition.column")
    normalized: dict[str, Any] = {"status": "available", "column": column}
    seen: dict[str, str] = {}
    for split_name in ("development", "validation", "oot"):
        field = f"{split_name}_values"
        raw_values = _array(obj[field], f"split_definition.{field}", required=False)
        if len(raw_values) > MAX_SAMPLE_DESIGN_SPLIT_VALUES:
            raise StrategySampleDesignError(
                f"split_definition.{field} exceeds item budget"
            )
        values = [_json_scalar(item, f"split_definition.{field}[]") for item in raw_values]
        identities = [_scalar_identity(item) for item in values]
        if len(identities) != len(set(identities)):
            raise StrategySampleDesignError(f"split_definition.{field} has duplicates")
        for identity in identities:
            prior = seen.get(identity)
            if prior is not None:
                raise StrategySampleDesignError(
                    f"split values overlap between {prior} and {split_name}"
                )
            seen[identity] = split_name
        normalized[field] = sorted(values, key=_scalar_identity)
    if not normalized["development_values"]:
        raise StrategySampleDesignError(
            "available split_definition requires development_values"
        )
    return normalized


def _validate_split_population_counts(
    value: object,
    *,
    split_definition: Mapping[str, Any],
) -> dict[str, int] | None:
    if split_definition["status"] == "unavailable":
        if value is not None:
            raise StrategySampleDesignError(
                "unavailable split must have null split_population_counts"
            )
        return None
    obj = _object(value, "split_population_counts")
    _require_exact_fields(
        obj,
        _SPLIT_POPULATION_FIELDS,
        name="split_population_counts",
    )
    result = {
        name: _non_negative_int(obj[name], f"split_population_counts.{name}")
        for name in ("development", "validation", "oot")
    }
    if result["development"] == 0:
        raise StrategySampleDesignError(
            "available split requires at least one development row"
        )
    return result


def _optional_fields(value: object) -> dict[str, str | None]:
    obj = _object(value, "optional_fields")
    _require_exact_fields(obj, _OPTIONAL_FIELDS, name="optional_fields")
    result = {
        key: _optional_column(obj[key], f"optional_fields.{key}")
        for key in sorted(_OPTIONAL_FIELDS)
    }
    selected = [item for item in result.values() if item is not None]
    if len(selected) != len(set(selected)):
        raise StrategySampleDesignError("optional_fields must be distinct")
    return {
        "month_field": result["month_field"],
        "weight_field": result["weight_field"],
        "loan_amount_field": result["loan_amount_field"],
        "overdue_amount_field": result["overdue_amount_field"],
    }


def _dimension(value: object) -> dict[str, str]:
    obj = _object(value, "metric_observation.dimension")
    _require_exact_fields(obj, _DIMENSION_FIELDS, name="metric_observation.dimension")
    kind = _enum(obj["kind"], {"overall", "split"}, "metric_observation.dimension.kind")
    dimension_value = _text(obj["value"], "metric_observation.dimension.value")
    if (kind == "overall" and dimension_value != "overall") or (
        kind == "split" and dimension_value not in {"development", "validation", "oot"}
    ):
        raise StrategySampleDesignError("metric_observation dimension is invalid")
    return {"kind": kind, "value": dimension_value}


def _definition_ref(value: object) -> dict[str, str]:
    obj = _object(value, "metric_definition_ref")
    _require_exact_fields(obj, _DEFINITION_REF_FIELDS, name="metric_definition_ref")
    definition_id = _text(obj["metric_definition_id"], "metric_definition_ref.id")
    if _DEFINITION_ID_RE.fullmatch(definition_id) is None:
        raise StrategySampleDesignError("metric_definition_ref id is invalid")
    return {
        "metric_definition_id": definition_id,
        "content_hash": _hash(obj["content_hash"], "metric_definition_ref.content_hash"),
    }


def _source_refs(value: object) -> list[dict[str, str]]:
    items = _array(value, "source_refs", required=True)
    result = []
    for item in items:
        obj = _object(item, "source_ref")
        _require_exact_fields(obj, _SOURCE_REF_FIELDS, name="source_ref")
        result.append(
            {
                "kind": _enum(obj["kind"], {"dataset", "sample_design"}, "source_ref.kind"),
                "ref_id": _text(obj["ref_id"], "source_ref.ref_id"),
                "content_hash": _hash(obj["content_hash"], "source_ref.content_hash"),
            }
        )
    if len({(item["kind"], item["ref_id"]) for item in result}) != len(result):
        raise StrategySampleDesignError("source_refs contain duplicates")
    return result


def _red_flags(value: object) -> list[dict[str, str]]:
    items = _array(value, "red_flags", required=False)
    result = []
    for item in items:
        obj = _object(item, "red_flag")
        _require_exact_fields(obj, _RED_FLAG_FIELDS, name="red_flag")
        result.append(
            {
                "code": _text(obj["code"], "red_flag.code"),
                "level": _enum(obj["level"], {"amber", "red"}, "red_flag.level"),
                "message": _text(obj["message"], "red_flag.message"),
            }
        )
    codes = [item["code"] for item in result]
    if len(codes) != len(set(codes)):
        raise StrategySampleDesignError("red_flags contain duplicate codes")
    return sorted(result, key=lambda item: item["code"])


def _expected_red_flags(
    *,
    performance_window: Mapping[str, Any],
    observation_window: Mapping[str, Any],
    split_definition: Mapping[str, Any],
    split_population_counts: Mapping[str, int] | None,
    maturity: str,
    missing_label_count: int,
) -> list[dict[str, str]]:
    flags = []
    if performance_window["status"] == "unavailable":
        flags.append(
            {
                "code": "performance_window_unavailable",
                "level": "red",
                "message": "Performance window is unavailable; strategy use is exploration only.",
            }
        )
    if observation_window["status"] == "unavailable":
        flags.append(
            {
                "code": "observation_window_unavailable",
                "level": "amber",
                "message": "Observation window is unavailable.",
            }
        )
    if split_definition["status"] == "unavailable":
        flags.append(
            {
                "code": "split_unavailable",
                "level": "amber",
                "message": "Development, validation, and OOT split is unavailable.",
            }
        )
    else:
        assert split_population_counts is not None
        if not split_definition["validation_values"]:
            flags.append(
                {
                    "code": "validation_split_unavailable",
                    "level": "amber",
                    "message": "No validation split values were defined.",
                }
            )
        elif split_population_counts["validation"] == 0:
            flags.append(
                {
                    "code": "validation_split_empty",
                    "level": "amber",
                    "message": "Validation split values were defined but matched no rows.",
                }
            )
        if not split_definition["oot_values"]:
            flags.append(
                {
                    "code": "oot_split_unavailable",
                    "level": "amber",
                    "message": "No OOT split values were defined; OOT validation cannot be claimed.",
                }
            )
        elif split_population_counts["oot"] == 0:
            flags.append(
                {
                    "code": "oot_split_empty",
                    "level": "amber",
                    "message": "OOT split values were defined but matched no rows.",
                }
            )
    if maturity == "not_matured":
        flags.append(
            {
                "code": "sample_not_matured",
                "level": "red",
                "message": "Sample is not matured; strategy use is exploration only.",
            }
        )
    elif maturity == "unknown":
        flags.append(
            {
                "code": "sample_maturity_unknown",
                "level": "red",
                "message": "Sample maturity is unknown; strategy use is exploration only.",
            }
        )
    if missing_label_count:
        flags.append(
            {
                "code": f"missing_labels_excluded_from_risk:{missing_label_count}",
                "level": "amber",
                "message": (
                    f"{missing_label_count} rows with missing labels remain in population "
                    "and are excluded only from risk denominators."
                ),
            }
        )
    return sorted(flags, key=lambda item: item["code"])


def _flag_missing_label_count(flags: Sequence[Mapping[str, str]]) -> int:
    matches = [
        item
        for item in flags
        if item["code"].startswith("missing_labels_excluded_from_risk:")
    ]
    if not matches:
        return 0
    if len(matches) != 1:
        raise StrategySampleDesignError("missing-label red flag is duplicated")
    raw = matches[0]["code"].partition(":")[2]
    if not raw.isdigit() or int(raw) < 1:
        raise StrategySampleDesignError("missing-label red flag count is invalid")
    return int(raw)


def _sample_scope(
    performance_window: Mapping[str, Any],
    observation_window: Mapping[str, Any],
    maturity: str,
) -> str:
    if (
        performance_window["status"] != "provided"
        or observation_window["status"] != "provided"
        or maturity != "confirmed_matured"
    ):
        return "exploration_only"
    return "strategy_development"


def _maturity(value: object) -> str:
    return _enum(value, MATURITY_STATUSES, "maturity")


def _address_object(
    body: Mapping[str, Any], *, id_field: str, id_prefix: str
) -> dict[str, Any]:
    _preflight_json_tree(body, name=id_field)
    object_id = id_prefix + _sha256(_canonical_json(body))[:24]
    without_hash = {**body, id_field: object_id}
    return {**without_hash, "content_hash": _sha256(_canonical_json(without_hash))}


def _validate_addressed_object(
    original: Mapping[str, Any],
    *,
    normalized_body: Mapping[str, Any],
    id_field: str,
    id_pattern: re.Pattern[str],
    id_prefix: str,
    name: str,
) -> dict[str, Any]:
    object_id = original[id_field]
    if not isinstance(object_id, str) or id_pattern.fullmatch(object_id) is None:
        raise StrategySampleDesignError(f"{name} {id_field} is invalid")
    expected_id = id_prefix + _sha256(_canonical_json(normalized_body))[:24]
    if not hmac.compare_digest(object_id, expected_id):
        raise StrategySampleDesignError(f"{name} {id_field} does not match content")
    content_hash = _hash(original["content_hash"], f"{name}.content_hash")
    without_hash = {**normalized_body, id_field: object_id}
    expected_hash = _sha256(_canonical_json(without_hash))
    if not hmac.compare_digest(content_hash, expected_hash):
        raise StrategySampleDesignError(f"{name} content_hash does not match content")
    return {**without_hash, "content_hash": content_hash}


def _metric_key_for_ref(
    ref: Mapping[str, str], definitions: Sequence[Mapping[str, Any]]
) -> str:
    for definition in definitions:
        if definition["metric_definition_id"] == ref["metric_definition_id"]:
            return str(definition["metric_key"])
    raise StrategySampleDesignError("metric definition ref is missing")


def _object(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StrategySampleDesignError(f"{name} must be an object")
    return value


def _array(value: object, name: str, *, required: bool) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise StrategySampleDesignError(f"{name} must be an array")
    result = list(value)
    if required and not result:
        raise StrategySampleDesignError(f"{name} must not be empty")
    return result


def _require_exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], *, name: str
) -> None:
    if any(not isinstance(key, str) for key in value):
        raise StrategySampleDesignError(f"{name} keys must be strings")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unknown:
            details.append("unknown: " + ", ".join(unknown))
        raise StrategySampleDesignError(f"{name} fields are invalid ({'; '.join(details)})")


def _preflight_json_tree(value: object, *, name: str) -> None:
    stack: list[tuple[object, int]] = [(value, 1)]
    seen_containers: set[int] = set()
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > MAX_SAMPLE_DESIGN_JSON_NODES:
            raise StrategySampleDesignError(f"{name} exceeds JSON node budget")
        if depth > MAX_SAMPLE_DESIGN_JSON_DEPTH:
            raise StrategySampleDesignError(f"{name} exceeds JSON depth budget")
        if isinstance(current, Mapping):
            identity = id(current)
            if identity in seen_containers:
                raise StrategySampleDesignError(
                    f"{name} contains a repeated or cyclic container"
                )
            seen_containers.add(identity)
            if any(not isinstance(key, str) for key in current):
                raise StrategySampleDesignError(f"{name} has a non-string JSON key")
            stack.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, Sequence) and not isinstance(
            current, (str, bytes, bytearray)
        ):
            identity = id(current)
            if identity in seen_containers:
                raise StrategySampleDesignError(
                    f"{name} contains a repeated or cyclic container"
                )
            seen_containers.add(identity)
            stack.extend((child, depth + 1) for child in current)
        elif current is None or isinstance(current, (str, bool, int, float)):
            if isinstance(current, float) and not math.isfinite(current):
                raise StrategySampleDesignError(f"{name} contains a non-finite number")
        else:
            raise StrategySampleDesignError(
                f"{name} contains unsupported JSON value {type(current).__name__}"
            )


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise StrategySampleDesignError(
                f"sample-design bundle JSON has duplicate key: {key}"
            )
        result[key] = value
    return result


def _canonical_json(value: object) -> str:
    _preflight_json_tree(value, name="canonical JSON")
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise StrategySampleDesignError("value is not finite canonical JSON") from exc


def _json_scalar(value: object, name: str) -> str | bool | int | float:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bytes, bytearray, complex, Decimal)):
        raise StrategySampleDesignError(f"{name} must be a non-null JSON scalar")
    if not isinstance(value, (str, bool, int, float)):
        raise StrategySampleDesignError(f"{name} must be a non-null JSON scalar")
    if isinstance(value, str):
        if not value or "\x00" in value:
            raise StrategySampleDesignError(f"{name} must be a non-empty JSON scalar")
        if len(value) > MAX_SAMPLE_DESIGN_SPLIT_STRING_LENGTH:
            raise StrategySampleDesignError(f"{name} exceeds string length budget")
    if isinstance(value, int) and not isinstance(value, bool):
        if abs(value) > _MAX_SAFE_JSON_INTEGER:
            raise StrategySampleDesignError(f"{name} exceeds exact JSON numeric range")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StrategySampleDesignError(f"{name} must be finite")
        if abs(value) > _MAX_SAFE_JSON_INTEGER:
            raise StrategySampleDesignError(f"{name} exceeds exact JSON numeric range")
        # JSON has one numeric domain even though Python distinguishes int/float.
        # Canonicalize integral floats so a user value ``1`` still matches a
        # pandas column materialized as ``1.0`` and cannot be placed in another
        # split under a type-only spelling difference.
        if value == 0 or value.is_integer():
            return int(value)
    return value


def _scalar_identity(value: str | bool | int | float) -> str:
    if isinstance(value, bool):
        kind = "bool"
    elif isinstance(value, int):
        kind = "int"
    elif isinstance(value, float):
        kind = "float"
    else:
        kind = "string"
    return _canonical_json([kind, value])


def _is_missing_scalar(value: object) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _column(value: object, name: str) -> str:
    return _text(value, name)


def _optional_column(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _column(value, name)


def _text(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise StrategySampleDesignError(f"{name} must be non-empty canonical text")
    return value


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _producer_version(value: object) -> str:
    return _text(value, "producer_version")


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise StrategySampleDesignError(f"{name} must be a lowercase SHA-256 hash")
    return value


def _enum(value: object, allowed: set[str] | frozenset[str], name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise StrategySampleDesignError(
            f"{name} must be one of {', '.join(sorted(allowed))}"
        )
    return value


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StrategySampleDesignError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: object, name: str) -> int:
    result = _non_negative_int(value, name)
    if result == 0:
        raise StrategySampleDesignError(f"{name} must be a positive integer")
    return result


def _optional_finite_number(value: object, name: str) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StrategySampleDesignError(f"{name} must be a finite number or null")
    if not math.isfinite(float(value)):
        raise StrategySampleDesignError(f"{name} must be a finite number or null")
    return value


def _iso_date(value: object, name: str) -> date:
    if not isinstance(value, str) or value != value.strip():
        raise StrategySampleDesignError(f"{name} must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise StrategySampleDesignError(f"{name} must be an ISO date") from exc
    if parsed.isoformat() != value:
        raise StrategySampleDesignError(f"{name} must be a canonical ISO date")
    return parsed


def _require_same_number(actual: object, expected: float, name: str) -> None:
    if (
        isinstance(actual, bool)
        or not isinstance(actual, (int, float))
        or not math.isfinite(float(actual))
        or not math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise StrategySampleDesignError(f"{name} is inconsistent")


def _sha256(value: str | bytes) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "MATURITY_STATUSES",
    "MAX_SAMPLE_DESIGN_JSON_BYTES",
    "MAX_SAMPLE_DESIGN_JSON_DEPTH",
    "MAX_SAMPLE_DESIGN_JSON_NODES",
    "MAX_SAMPLE_DESIGN_SPLIT_STRING_LENGTH",
    "MAX_SAMPLE_DESIGN_SPLIT_VALUES",
    "METRIC_OBSERVATION_STATUSES",
    "STRATEGY_METRIC_DEFINITION_SCHEMA_VERSION",
    "STRATEGY_METRIC_OBSERVATION_SCHEMA_VERSION",
    "STRATEGY_SAMPLE_DESIGN_BUNDLE_SCHEMA_VERSION",
    "STRATEGY_SAMPLE_DESIGN_PRODUCER_VERSION",
    "STRATEGY_SAMPLE_DESIGN_SCHEMA_VERSION",
    "StrategySampleDesignError",
    "build_strategy_sample_design_bundle",
    "canonical_strategy_sample_design_bundle_json",
    "strategy_sample_design_bundle_from_json",
    "validate_strategy_sample_design_bundle",
]
