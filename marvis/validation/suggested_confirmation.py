from __future__ import annotations

import re

from marvis.repositories.tasks import TaskRepository
from marvis.repositories.validation_batches import ValidationBatchRepository
from marvis.repositories.validation_contracts import ValidationContractRepository
from marvis.validation.input_confirmation import validate_confirmation_against_materials
from marvis.validation.input_contracts import (
    ValidationInputConfirmation,
    ValidationInputContract,
)
from marvis.validation_materials import resolve_selected_validation_materials

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


def confirm_unambiguous_contract(*, db_path, task_id: str) -> bool:
    """Confirm a pending contract when every required field has exactly one candidate.

    Returns False when the contract is missing, not pending, or still ambiguous.
    Raises if unique candidates fail material validation.
    """
    task_repo = TaskRepository(db_path)
    contract_repo = ValidationContractRepository(db_path)
    record = contract_repo.get(task_id)
    if record is None or record.status != "pending_confirmation":
        return False
    suggested = suggested_confirmation_from_contract(record.contract)
    if suggested is None:
        return False
    child = task_repo.get_task(task_id)
    paths = resolve_selected_validation_materials(child)
    validated = validate_confirmation_against_materials(
        contract=record.contract,
        sample_path=paths.sample,
        dictionary_path=paths.dictionary,
        requested=suggested,
    )
    contract_repo.confirm(
        task_id,
        validated.values,
        expected_revision=record.revision,
        resolved_sample_schema=validated.sample_schema,
        resolved_feature_metadata=validated.feature_metadata,
    )
    return True


def confirm_unambiguous_batch_contracts(
    *,
    db_path,
    parent_task_id: str,
) -> dict[str, list[str]]:
    batch_repo = ValidationBatchRepository(db_path)
    confirmed: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    for item in batch_repo.list_items(parent_task_id):
        if item.status != "awaiting_confirmation":
            continue
        try:
            if confirm_unambiguous_contract(db_path=db_path, task_id=item.child_task_id):
                confirmed.append(item.model_name)
            else:
                skipped.append(item.model_name)
        except Exception as exc:
            failed.append(f"{item.model_name}：{exc}")
    return {
        "confirmed": confirmed,
        "skipped": skipped,
        "failed": failed,
    }
