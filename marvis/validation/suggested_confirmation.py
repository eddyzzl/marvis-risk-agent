from __future__ import annotations

import re

from marvis.validation.input_contracts import (
    ValidationInputConfirmation,
    ValidationInputContract,
)

_REQUIRED_STRING_FIELDS = (
    "target_col",
    "split_col",
    "time_col",
    "pmml_output_field",
)


def _unique_candidate_value(contract: ValidationInputContract, key: str):
    candidates = contract.candidates.get(key) or ()
    if len(candidates) != 1:
        return None
    return candidates[0].value


def suggested_confirmation_from_contract(
    contract: ValidationInputContract,
) -> ValidationInputConfirmation | None:
    if contract.conflicts:
        return None
    values = {
        key: _unique_candidate_value(contract, key)
        for key in _REQUIRED_STRING_FIELDS
    }
    if any(not isinstance(value, str) or not value.strip() for value in values.values()):
        return None
    metadata = _unique_candidate_value(contract, "feature_metadata_selection")
    if not isinstance(metadata, dict):
        return None
    feature_col = str(metadata.get("feature_col") or "").strip()
    category_col = str(metadata.get("category_col") or "").strip()
    importance_col = str(metadata.get("importance_col") or "").strip()
    if not feature_col or not category_col or not importance_col:
        return None
    model_params = _unique_candidate_value(contract, "model_params")
    if model_params is None:
        model_params = {}
    if not isinstance(model_params, dict):
        return None
    split_mapping = _unique_candidate_value(contract, "split_value_mapping")
    if split_mapping is None:
        split_mapping = {"train": "train", "test": "test", "oot": "oot"}
    if not isinstance(split_mapping, dict):
        return None
    time_col = str(values["time_col"])
    granularity = _unique_candidate_value(contract, "time_granularity")
    if not isinstance(granularity, str) or not granularity.strip():
        granularity = (
            "date"
            if re.search(r"date|day|dt", time_col, flags=re.IGNORECASE)
            else "month"
        )
    metadata_sheet = metadata.get("metadata_sheet")
    return ValidationInputConfirmation(
        target_col=str(values["target_col"]),
        positive_label=1,
        negative_label=0,
        split_col=str(values["split_col"]),
        split_value_mapping={
            str(key): value for key, value in split_mapping.items()
        },
        time_col=time_col,
        time_granularity=str(granularity),
        pmml_output_field=str(values["pmml_output_field"]),
        model_params=model_params,
        metadata_sheet=None if metadata_sheet in {None, ""} else str(metadata_sheet),
        feature_col=feature_col,
        category_col=category_col,
        importance_col=importance_col,
        transformations=contract.transformations,
    )
