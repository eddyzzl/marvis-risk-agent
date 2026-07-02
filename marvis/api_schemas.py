from pydantic import BaseModel, Field

from marvis.domain import TASK_TYPE_VALIDATION


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
    target_type: str = ""
    recipes: list[str] = Field(default_factory=list)
    sample_weight_col: str = ""
    # AGT-4 (optional, modeling tasks only): None/absent → no success criterion is
    # injected into the plan. Never defaulted to a platform-chosen number.
    oot_ks_min: float | None = None
    metrics: list[str] = Field(default_factory=list)
    # Per-task capability tier (conservative/balanced/aggressive); "" → global default.
    capability_tier: str = ""
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
    capability_tier: str = ""
    models: list[dict] = Field(default_factory=list)


class MemoryPolicyRequest(BaseModel):
    reference_cross_task: bool = True
    auto_distill: bool = True


class ReportFieldsUpdateRequest(BaseModel):
    text_values: dict[str, str] = Field(default_factory=dict)


class AgentMessageRequest(BaseModel):
    content: str
    model_id: str | None = None
    effort: str | None = None
    acceptance_mode: str | None = None
    # Optional edited feature set from the §4 interactive screening table; when a
    # screening gate is confirmed this overrides the screen's proposed `selected`.
    selection: list[str] | None = None
    # Optional per-feature dedup strategy map (feature_id -> first|last) from the §4
    # join dedup picker; re-confirms confirm_join to resolve non-unique-key conflicts.
    dedup_strategies: dict[str, str] | None = None
    # Optional structured parameter overrides from manual controls (for example the
    # feature-screening leakage/missing thresholds). These bypass LLM text routing.
    adjust_params: dict[str, object] | None = None
    # Optional optimistic-lock token for structured gate controls. The frontend sends
    # the gate step id it rendered; the backend rejects stale tabs/buttons.
    expected_step_id: str | None = None


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
