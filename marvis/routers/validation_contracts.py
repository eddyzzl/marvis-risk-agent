from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from marvis.api_schemas import ValidationInputConfirmationRequest
from marvis.api_stage_helpers import start_task_job
from marvis.api_task_helpers import (
    get_task_or_404,
)
from marvis.repositories.tasks import TaskRepository
from marvis.domain import TASK_TYPE_VALIDATION
from marvis.errors import conflict, not_found, unprocessable
from marvis.job_heartbeat import heartbeat_job
from marvis.repositories.validation_contracts import (
    ValidationContractRepository,
    ValidationContractRevisionConflict,
)
from marvis.validation.input_confirmation import (
    validate_confirmation_against_materials,
)
from marvis.agent.workflow_error_diagnostics import workflow_error_content
from marvis.validation.input_contracts import (
    ValidationInputConfirmation,
    transformation_spec_from_dict,
    validation_confirmation_to_dict,
)
from marvis.validation_materials import resolve_selected_validation_materials


router = APIRouter(prefix="/api", tags=["validation-contracts"])
logger = logging.getLogger(__name__)


def _task_repo(request: Request) -> TaskRepository:
    return TaskRepository(request.app.state.settings.db_path)


def _contract_repo(request: Request) -> ValidationContractRepository:
    return ValidationContractRepository(request.app.state.settings.db_path)


@router.get("/tasks/{task_id}/validation-input-contract")
def get_validation_input_contract(task_id: str, request: Request) -> dict:
    record = _contract_repo(request).get(task_id)
    if record is None:
        raise not_found("validation input contract not found")
    return record.to_api_payload()


def _add_confirmation_failure_card(
    request: Request,
    task_repo: TaskRepository,
    task,
    *,
    error: str,
    requested: ValidationInputConfirmation,
) -> None:
    """Put a classified confirmation failure into the Agent conversation.

    The 422 response is unchanged; the card adds the diagnosis and the
    one-click options (B-line spec 2026-10-16 §5.6). The same failure
    submitted again does not stack a second identical card.
    """

    from marvis.agent.validation_repair import diagnose_validation_failure

    # Batch children get the card too: the batch workbench shows the selected
    # child's conversation and sends its commands to that child
    # (2026-10-23 extension spec §5.4).
    diagnostic = diagnose_validation_failure(
        settings=request.app.state.settings,
        task=task,
        stage="input_confirmation",
        error=error,
        requested_confirmation=validation_confirmation_to_dict(requested),
    )
    if diagnostic is None:
        return
    messages = task_repo.list_agent_messages(task.id)
    latest = next(
        (item for item in reversed(messages) if item.get("role") == "assistant"),
        None,
    )
    previous = ((latest or {}).get("metadata") or {}).get("error_diagnostic") or {}
    if (
        previous.get("code") == diagnostic.get("code")
        and previous.get("technical_detail") == diagnostic.get("technical_detail")
    ):
        return
    task_repo.add_agent_message(
        task.id,
        role="assistant",
        stage="failure",
        content=workflow_error_content(diagnostic),
        metadata={"error": True, "error_diagnostic": diagnostic},
    )


@router.put("/tasks/{task_id}/validation-input-contract")
def confirm_validation_input_contract(
    task_id: str,
    payload: ValidationInputConfirmationRequest,
    request: Request,
) -> dict:
    task_repo = _task_repo(request)
    task = get_task_or_404(task_repo, task_id)
    if (
        task.task_type != TASK_TYPE_VALIDATION
        or task.validation_workflow_version != 2
    ):
        raise unprocessable(
            "validation input contract requires a version 2 validation task"
        )
    contract_repo = _contract_repo(request)
    current = contract_repo.get(task_id)
    if current is None:
        raise not_found("validation input contract not found")
    if current.revision != payload.revision:
        raise conflict(
            "validation input contract revision conflict: expected "
            f"{payload.revision}, found {current.revision}"
        )
    if current.status == "blocked":
        raise unprocessable("blocked validation input contract cannot be confirmed")
    if current.status != "pending_confirmation":
        raise conflict("only a pending validation input contract can be confirmed")

    requested: ValidationInputConfirmation | None = None
    job_id = start_task_job(task_repo, task_id, "validation_input_confirmation")
    if not task_repo.mark_job_running(job_id):
        task_repo.finish_job(job_id, status="failed")
        raise conflict("validation input confirmation job could not be started")
    try:
        with heartbeat_job(task_repo, job_id):
            task = task_repo.get_task(task_id)
            current = contract_repo.get(task_id)
            if current is None:
                raise ValueError("validation input contract not found")
            if current.revision != payload.revision:
                raise ValidationContractRevisionConflict(
                    "validation input contract revision conflict: expected "
                    f"{payload.revision}, found {current.revision}"
                )
            if current.status != "pending_confirmation":
                raise ValueError(
                    "only a pending validation input contract can be confirmed"
                )
            paths = resolve_selected_validation_materials(task)
            from marvis.modeling_holdout.guards import record_exposure
            settings = request.app.state.settings
            for path in (paths.sample, paths.dictionary):
                record_exposure(settings.db_path, settings.workspace, path)
            requested = ValidationInputConfirmation(
                target_col=payload.target_col,
                positive_label=payload.positive_label,
                negative_label=payload.negative_label,
                split_col=payload.split_col,
                split_value_mapping=payload.split_value_mapping,
                time_col=payload.time_col,
                time_granularity=payload.time_granularity,
                pmml_output_field=payload.pmml_output_field,
                model_params=payload.model_params,
                metadata_sheet=payload.metadata_sheet,
                feature_col=payload.feature_col,
                category_col=payload.category_col,
                importance_col=payload.importance_col,
                transformations=tuple(
                    transformation_spec_from_dict(item)
                    for item in payload.transformations
                ),
            )
            validated = validate_confirmation_against_materials(
                contract=current.contract,
                sample_path=paths.sample,
                dictionary_path=paths.dictionary,
                requested=requested,
            )
            record = contract_repo.confirm(
                task_id,
                validated.values,
                expected_revision=payload.revision,
                resolved_sample_schema=validated.sample_schema,
                resolved_feature_metadata=validated.feature_metadata,
            )
    except ValidationContractRevisionConflict as exc:
        task_repo.finish_job(
            job_id,
            status="failed",
            error_name=exc.__class__.__name__,
            error_value=str(exc),
        )
        raise conflict(str(exc)) from exc
    except (TypeError, ValueError) as exc:
        task_repo.finish_job(
            job_id,
            status="failed",
            error_name=exc.__class__.__name__,
            error_value=str(exc),
        )
        if task.run_mode == "agent" and requested is not None:
            try:
                _add_confirmation_failure_card(
                    request, task_repo, task, error=str(exc), requested=requested
                )
            except Exception:  # noqa: BLE001 - the card must never replace the 422
                logger.exception(
                    "validation confirmation failure card failed task_id=%s", task_id
                )
        raise unprocessable(str(exc)) from exc
    except Exception as exc:
        task_repo.finish_job(
            job_id,
            status="failed",
            error_name=exc.__class__.__name__,
            error_value=str(exc),
        )
        raise
    task_repo.finish_job(job_id, status="succeeded")
    return record.to_api_payload()


__all__ = ["router"]
