from pydantic import BaseModel, Field

from riskmodel_checker.domain import TASK_TYPE_VALIDATION


class CreateTaskRequest(BaseModel):
    task_type: str = TASK_TYPE_VALIDATION
    model_name: str
    model_version: str = ""
    validator: str
    source_dir: str
    algorithm: str = ""
    target_col: str = "y"
    score_col: str = "pred"
    split_col: str = "split"
    time_col: str = "apply_month"
    run_mode: str = "manual"
    feature_columns: list[str] = Field(default_factory=list)
    notebook_path: str | None = None
    sample_path: str | None = None
    pmml_path: str | None = None
    dictionary_path: str | None = None
    report_values: dict[str, str] = Field(default_factory=dict)


class ValidateRequest(BaseModel):
    feature_columns: list[str] | None = None


class ExecutionEnvironmentRequest(BaseModel):
    execution_mode: str = "jupyter_kernel"
    kernel_name: str = "python3"
    conda_env_name: str = ""
    python_executable: str = ""


class LLMSettingsRequest(BaseModel):
    default_model_id: str = ""
    models: list[dict] = Field(default_factory=list)


class ReportFieldsUpdateRequest(BaseModel):
    text_values: dict[str, str] = Field(default_factory=dict)


class AgentMessageRequest(BaseModel):
    content: str
    model_id: str | None = None
    effort: str | None = None
    acceptance_mode: str | None = None


class AgentModelRequest(BaseModel):
    model_id: str | None = None
    effort: str | None = None
    acceptance_mode: str | None = None


class AgentReportDraftConfirmRequest(BaseModel):
    revision: int
    text_values: dict[str, str] = Field(default_factory=dict)


def model_payload(payload: BaseModel) -> dict:
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    return payload.dict()
