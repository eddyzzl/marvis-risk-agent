"""Task-scoped asynchronous API for deterministic dataset analysis."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, BackgroundTasks, Header, Request, Response
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr

from marvis.data.analysis_service import (
    DataAnalysisActiveJobError,
    DataAnalysisArtifactError,
    DataAnalysisDispatch,
    DataAnalysisNotFoundError,
    DataAnalysisRequest,
    DataAnalysisRequestError,
    DataAnalysisRetryRequiredError,
    DataAnalysisRunView,
    DataAnalysisService,
    DataAnalysisWorkspacePreconditionError,
)
from marvis.data.descriptive import DescriptiveConfig, DescriptiveConfigError
from marvis.errors import (
    bad_request,
    conflict,
    not_found,
    precondition_failed,
    precondition_required,
    unprocessable,
)
from marvis.repositories.data_analysis import (
    DataAnalysisConflictError,
    DataAnalysisDataError,
    DataAnalysisNotFoundError as RepositoryDataAnalysisNotFoundError,
    DataAnalysisStaleIdentityError,
    DataAnalysisTransitionError,
)
from marvis.settings import Settings


router = APIRouter(prefix="/api", tags=["data-analysis"])

PositiveStrictInt = Annotated[StrictInt, Field(gt=0)]
DataAnalysisSection = Literal[
    "overview",
    "target",
    "missing",
    "distribution",
    "correlation",
]


class DataAnalysisConfigRequest(BaseModel):
    """Public deterministic-analysis limits; all values are strict integers."""

    model_config = ConfigDict(extra="forbid", strict=True)

    max_columns: PositiveStrictInt = 200
    max_numeric_columns: PositiveStrictInt = 64
    max_pairs: PositiveStrictInt = 2016
    frequency_top_k: PositiveStrictInt = 20
    low_cardinality_threshold: PositiveStrictInt = 20
    histogram_bins: PositiveStrictInt = 20
    summary_batch_size: PositiveStrictInt = 16
    correlation_batch_size: PositiveStrictInt = 32

    def to_domain(self) -> DescriptiveConfig:
        return DescriptiveConfig(**self.model_dump())


class DataAnalysisPostRequest(BaseModel):
    """Strict request whose normalized form participates in cache identity."""

    model_config = ConfigDict(extra="forbid", strict=True)

    sections: list[DataAnalysisSection]
    columns: list[StrictStr] | None = None
    config: DataAnalysisConfigRequest = Field(
        default_factory=DataAnalysisConfigRequest
    )
    retry: StrictBool = False

    def to_domain(self) -> DataAnalysisRequest:
        return DataAnalysisRequest(
            sections=self.sections,
            columns=self.columns,
            config=self.config.to_domain(),
            retry=self.retry,
        )


def _parse_if_match(if_match: str | None) -> int:
    if if_match is None:
        raise precondition_required("If-Match header is required")
    if not if_match.isascii() or not if_match.isdecimal():
        raise bad_request("If-Match must be a non-negative integer")
    try:
        expected_revision = int(if_match)
    except ValueError as exc:
        raise bad_request("If-Match must be a non-negative integer") from exc
    if expected_revision < 0:
        raise bad_request("If-Match must be a non-negative integer")
    return expected_revision


@router.post("/tasks/{task_id}/data-analysis")
def request_data_analysis(
    task_id: str,
    payload: DataAnalysisPostRequest,
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    if_match: str | None = Header(default=None, alias="If-Match"),
) -> dict[str, object]:
    expected_revision = _parse_if_match(if_match)
    service = DataAnalysisService(request.app.state.settings)
    try:
        dispatch = service.request_analysis(
            task_id,
            expected_workspace_revision=expected_revision,
            request=payload.to_domain(),
        )
    except DataAnalysisNotFoundError as exc:
        raise not_found("task not found") from exc
    except DataAnalysisWorkspacePreconditionError as exc:
        raise precondition_failed(str(exc)) from exc
    except DataAnalysisStaleIdentityError as exc:
        raise precondition_failed(str(exc)) from exc
    except RepositoryDataAnalysisNotFoundError as exc:
        raise not_found("task not found") from exc
    except DataAnalysisRetryRequiredError as exc:
        record = exc.record
        raise conflict(
            {
                "kind": "data_analysis_retry_required",
                "run_id": record.id,
                "status": record.status,
                "error_kind": record.error_kind,
                "error_message": record.error_message,
            }
        ) from exc
    except DataAnalysisActiveJobError as exc:
        raise conflict(str(exc)) from exc
    except DataAnalysisArtifactError as exc:
        raise conflict(str(exc)) from exc
    except (
        DataAnalysisRequestError,
        DataAnalysisDataError,
        DescriptiveConfigError,
    ) as exc:
        raise unprocessable(str(exc)) from exc
    except (DataAnalysisConflictError, DataAnalysisTransitionError) as exc:
        raise conflict(str(exc)) from exc

    if dispatch.should_execute:
        job_id = dispatch.job_id
        if job_id is None:
            raise conflict("queued data analysis has no task job")
        try:
            background_tasks.add_task(
                _run_data_analysis_job,
                request.app.state.settings,
                task_id,
                dispatch.record.id,
                job_id,
            )
        except Exception as exc:
            service.fail_dispatch(
                task_id=task_id,
                run_id=dispatch.record.id,
                job_id=job_id,
                error_kind="background_registration_failed",
                error_message="data analysis background registration failed",
            )
            raise conflict("data analysis background registration failed") from exc

    response.status_code = dispatch.http_status
    return _dispatch_payload(dispatch)


@router.get("/tasks/{task_id}/data-analysis/{run_id}")
def get_data_analysis(
    task_id: str,
    run_id: str,
    request: Request,
) -> dict[str, object]:
    service = DataAnalysisService(request.app.state.settings)
    try:
        view = service.get_run(task_id, run_id)
    except DataAnalysisArtifactError as exc:
        raise conflict(str(exc)) from exc
    except (DataAnalysisRequestError, DataAnalysisDataError) as exc:
        raise unprocessable(str(exc)) from exc
    if view is None:
        raise not_found("data analysis run not found")
    return _run_payload(view)


def _run_data_analysis_job(
    settings: Settings,
    task_id: str,
    run_id: str,
    job_id: str,
) -> None:
    DataAnalysisService(settings).run_job(
        task_id=task_id,
        run_id=run_id,
        job_id=job_id,
    )


def _dispatch_payload(dispatch: DataAnalysisDispatch) -> dict[str, object]:
    view = DataAnalysisRunView(
        record=dispatch.record,
        result=dispatch.result,
        result_artifact_id=dispatch.result_artifact_id,
        download_url=dispatch.download_url,
    )
    result = _run_payload(view)
    if dispatch.cached:
        result["cached"] = True
    return result


def _run_payload(view: DataAnalysisRunView) -> dict[str, object]:
    record = view.record
    payload: dict[str, object] = {
        "task_id": record.task_id,
        "run_id": record.id,
        "job_id": record.job_id,
        "status": record.status,
    }
    if record.status == "succeeded":
        payload.update(
            {
                "result_artifact_id": view.result_artifact_id,
                "download_url": view.download_url,
                "result": view.result,
            }
        )
    elif record.status in {"failed", "cancelled"}:
        payload["error"] = {
            "kind": record.error_kind or "data_analysis_failed",
            "message": record.error_message or "data analysis failed",
        }
    return payload


__all__ = [
    "DataAnalysisConfigRequest",
    "DataAnalysisPostRequest",
    "router",
]
