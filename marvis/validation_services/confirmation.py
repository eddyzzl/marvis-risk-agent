"""Revision-checked task and batch confirmation orchestration."""
from __future__ import annotations

from marvis.repositories.tasks import TaskRepository
from marvis.repositories.validation_batches import ValidationBatchRepository
from marvis.repositories.validation_contracts import ValidationContractRepository
from marvis.validation.input_confirmation import validate_confirmation_against_materials
from marvis.validation.suggested_confirmation import suggested_confirmation_from_contract
from marvis.validation_materials import resolve_selected_validation_materials


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
