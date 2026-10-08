"""Public execution inputs. Expected answers are deliberately a different file/type.

Cases contain business instructions, data identities and predeclared human actions.
They cannot name arbitrary HTTP routes, Python callbacks, profiles or filesystem
outputs. Scoring never supplies an instruction or a gate decision to the runtime.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from marvis.api_schemas import DataSemanticMappingRequest, PortfolioSetupRequest
from .runtime_labeling import LabelingBusinessInput
from .runtime_monitoring import MonitoringBusinessInput

RUNTIME_FINISH_REASONS = frozenset(
    {"stop", "length", "tool_calls", "function_call", "content_filter", "other"}
)
RUNTIME_ATTEMPT_OUTCOMES = frozenset(
    {
        "transport_error",
        "http_error",
        "invalid_response",
        "missing_content",
        "content_filtered",
        "empty_at_output_limit",
        "content_at_output_limit",
        "empty_content",
        "content_present",
        "unknown",
        "incomplete",
    }
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RuntimeBudget(StrictModel):
    wall_seconds: int = Field(default=180, ge=1, le=3600)
    max_llm_attempts: int = Field(default=30, ge=0, le=1000)
    max_http_requests: int = Field(default=150, ge=1, le=5000)
    max_output_tokens_per_attempt: int = Field(default=2048, ge=1, le=32768)
    max_total_tokens: int | None = Field(default=None, ge=0)
    max_cost: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    currency: str | None = Field(default=None, min_length=1, max_length=12)

    @model_validator(mode="after")
    def cost_currency(self):
        if (self.max_cost is None) != (self.currency is None):
            raise ValueError("max_cost and currency must be declared together")
        return self

    @model_serializer(mode="wrap")
    def preserve_legacy_identity(self, handler):
        # Old archived cases must retain their original digest. Only newly
        # declared limits add keys; do not silently rewrite historical budgets.
        value = handler(self)
        for key in ("max_total_tokens", "max_cost", "currency"):
            if value.get(key) is None:
                value.pop(key, None)
        return value


class Material(StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    role: Literal["sample", "feature", "unknown", "notebook", "pmml", "dictionary"] = "sample"
    source_kind: Literal["synthetic", "deidentified_historical"] = "synthetic"

    @model_validator(mode="after")
    def relative_path(self):
        path = Path(self.path)
        allowed = {
            "notebook": {".ipynb"}, "pmml": {".pmml"}, "dictionary": {".csv", ".xlsx"},
        }.get(self.role, {".csv", ".parquet", ".xlsx"})
        if (
            path.is_absolute() or ".." in path.parts or "\\" in self.path
            or ":" in self.path or path.suffix.lower() not in allowed
        ):
            raise ValueError("material path must be relative and match its declared role")
        return self


class RuntimeTask(StrictModel):
    task_type: Literal[
        "data_join", "feature_analysis", "modeling", "strategy", "vintage", "portfolio", "validation"
    ]
    model_name: str = "Runtime benchmark"
    validator: str = "benchmark"
    target_col: str = "y"
    score_col: str = "pred"
    split_col: str = "split"
    time_col: str = "apply_month"
    feature_columns: list[str] = Field(default_factory=list)
    metrics: list[str] | None = None
    target_type: str = ""
    strategy_input: dict | None = None
    algorithm: Literal["lr", "lgb", "xgb", "catboost", "scorecard", "dnn"] | None = None

    @model_serializer(mode="wrap")
    def preserve_legacy_identity(self, handler):
        value = handler(self)
        if value.get("algorithm") is None:
            value.pop("algorithm", None)
        return value


class RuntimeAction(StrictModel):
    kind: Literal[
        "message",
        "approve_step",
        "reject_step",
        "replay_approval",
        "retry_step",
        "stop",
        "select_recommended_experiment",
        "bind_single_strategy_sample",
        "start_validation_workflow",
        "start_validation_agent",
        "confirm_current_validation_report",
        "submit_labeling_request",
        "download_labeling_results",
        "download_portfolio_report",
        "download_feature_report",
        "submit_model_monitoring_request",
        "start_model_monitoring_workflow",
        "confirm_model_monitoring_plan",
    ]
    content: str = ""
    tool: str = ""
    portfolio_request: PortfolioSetupRequest | None = None
    semantic_mapping: DataSemanticMappingRequest | None = None
    labeling_request: LabelingBusinessInput | None = None
    monitoring_request: MonitoringBusinessInput | None = None

    @model_validator(mode="after")
    def required_fields(self):
        if self.kind in {"submit_model_monitoring_request", "start_model_monitoring_workflow"}:
            if self.monitoring_request is None or not self.content.strip() or self.tool:
                raise ValueError("monitoring requires declared data/label choices and human text")
        elif self.monitoring_request is not None:
            raise ValueError("monitoring fields belong only to its declared start action")
        if self.kind == "confirm_model_monitoring_plan" and (self.tool or not self.content.strip()):
            raise ValueError("monitoring confirmation needs human text and its current proposal")
        if self.kind in {"start_validation_workflow", "start_validation_agent", "confirm_current_validation_report"} and (
            self.tool or not self.content.strip() or self.portfolio_request is not None
            or self.semantic_mapping is not None
        ):
            raise ValueError("validation start accepts explicit human text only, no tool or injected identifiers")
        if self.kind == "submit_labeling_request":
            if self.labeling_request is None or not self.content.strip() or self.tool:
                raise ValueError("labeling requires explicit business fields and human text")
        elif self.labeling_request is not None:
            raise ValueError("labeling fields belong only to the labeling proposal action")
        if self.kind == "download_labeling_results" and (self.tool or not self.content.strip()):
            raise ValueError("label downloads require explicit human text and no arbitrary tool")
        if self.kind in {"download_portfolio_report", "download_feature_report"} and (self.tool or not self.content.strip()):
            raise ValueError("report download requires explicit human text and no arbitrary tool")
        if self.kind == "bind_single_strategy_sample":
            if self.tool or not self.content.strip() or self.semantic_mapping is None:
                raise ValueError(
                    "sample binding requires explicit human text and field semantics only"
                )
        elif self.semantic_mapping is not None:
            raise ValueError(
                "field semantics belong to the explicit sample binding action only"
            )
        if self.kind == "select_recommended_experiment":
            if self.tool != "modeling.select_experiment" or not self.content.strip():
                raise ValueError(
                    "recommended selection requires the modeling selection gate and explicit human text"
                )
        if self.portfolio_request is not None and self.kind != "message":
            raise ValueError("portfolio_request belongs to a user message only")
        if (
            self.kind in {"message", "approve_step", "reject_step"}
            and not self.content.strip()
        ):
            raise ValueError(
                "a user message / approval requires explicit business text"
            )
        if (
            self.kind in {"approve_step", "reject_step", "retry_step"}
            and not self.tool.strip()
        ):
            raise ValueError("a step action requires a tool reference")
        return self

    @model_serializer(mode="wrap")
    def preserve_legacy_identity(self, handler):
        value = handler(self)
        if value.get("portfolio_request") is None:
            value.pop("portfolio_request", None)
        if value.get("semantic_mapping") is None:
            value.pop("semantic_mapping", None)
        if value.get("monitoring_request") is None:
            value.pop("monitoring_request", None)
        if value.get("labeling_request") is None:
            value.pop("labeling_request", None)
        return value


class RuntimeCase(StrictModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    revision: str = Field(min_length=1)
    family: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    case_set: Literal[
        "development", "fixed_regression", "independently_held_hidden_acceptance"
    ] = "development"
    scenario: Literal["normal", "clarification", "rejection", "recovery"] = "normal"
    task: RuntimeTask
    materials: list[Material] = Field(default_factory=list)
    initial_message: str | None = None
    acceptance_mode: Literal["auto_accept", "manual_review"] = "auto_accept"
    actions: list[RuntimeAction] = Field(default_factory=list)
    budget: RuntimeBudget = Field(default_factory=RuntimeBudget)
    business_constraints_source: str = Field(min_length=1)

    @model_validator(mode="after")
    def strategy_sample_binding(self):
        validation_roles = {"sample", "notebook", "pmml", "dictionary"}
        if self.task.task_type == "validation":
            compatibility = bool(self.actions and self.actions[0].kind == "start_validation_workflow")
            if (
                self.task.algorithm is None or self.initial_message is not None
                or len(self.materials) != len(validation_roles)
                or {m.role for m in self.materials} != validation_roles
                or len({m.path for m in self.materials}) != len(validation_roles)
                or not self.actions
                or (compatibility and any(a.kind not in {"approve_step", "reject_step"}
                    or a.tool != "v1_compat.render_reports" for a in self.actions[1:]))
                or (not compatibility and [a.kind for a in self.actions] != [
                    "start_validation_agent", "confirm_current_validation_report"])
            ):
                raise ValueError("validation requires unique role-bound files and one declared native entry with its own confirmation")
        elif (self.task.algorithm is not None
              or any(m.role in {"notebook", "pmml", "dictionary"} for m in self.materials)
              or any(a.kind in {"start_validation_workflow", "start_validation_agent", "confirm_current_validation_report"} for a in self.actions)):
            raise ValueError("validation files and start action belong only to validation")
        monitor = [i for i, a in enumerate(self.actions) if a.monitoring_request is not None]
        confirms = [i for i, a in enumerate(self.actions) if a.kind == "confirm_model_monitoring_plan"]
        if monitor or confirms:
            if (len(monitor) != 1 or confirms != [monitor[0] + 1]
                    or self.task.task_type != "modeling" or monitor[0] == 0
                    or len(self.materials) != 2
                    or len({m.path for m in self.materials}) != 2
                    or len([m for m in self.materials if m.role == "sample"]) != 1
                    or not any(m.path == self.actions[monitor[0]].monitoring_request.material_path
                               and m.role == "unknown" for m in self.materials)):
                raise ValueError("monitoring follows modeling with one separate declared new dataset and confirmation")
        labeling = [i for i, a in enumerate(self.actions) if a.kind == "submit_labeling_request"]
        if labeling and (
            labeling != [0] or self.initial_message is not None
            or self.task.task_type != "data_join" or len(self.materials) != 1
            or self.materials[0].role != "sample"
        ):
            raise ValueError("labeling starts with one declared data_join sample and proposal")
        if any(a.kind == "download_labeling_results" for a in self.actions) and not labeling:
            raise ValueError("label downloads require a declared labeling proposal")
        downloads = [i for i, a in enumerate(self.actions) if a.kind == "download_portfolio_report"]
        if downloads:
            declarations = [i for i, a in enumerate(self.actions) if a.portfolio_request is not None]
            if (self.task.task_type != "portfolio" or len(self.materials) != 1
                    or self.materials[0].role != "sample" or len(declarations) != 1
                    or any(i <= declarations[0] for i in downloads)):
                raise ValueError("portfolio download follows one declared portfolio request and sample")
        if any(a.kind == "download_feature_report" for a in self.actions) and self.task.task_type != "feature_analysis":
            raise ValueError("feature report download requires a feature analysis task")
        if any(a.kind == "bind_single_strategy_sample" for a in self.actions):
            if (
                self.task.task_type != "strategy"
                or len(self.materials) != 1
                or self.materials[0].role != "sample"
            ):
                raise ValueError(
                    "strategy sample binding requires exactly one declared strategy sample"
                )
        return self


class RuntimeSuite(StrictModel):
    schema_version: Literal[1] = 1
    cases: list[RuntimeCase] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("case ids must be unique; retries belong in a new run")
        return self


class ModelConnection(StrictModel):
    """Explicit config, never a scan of the user's workspace or saved settings.

    Real credentials are resolved only by the parent through one named environment
    variable. Children receive a private loopback gateway profile. Inline secrets,
    request-field overrides and role routing are not accepted here.
    """

    model_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    model_name: str = Field(min_length=1)
    api_base_url: str
    api_key_env: str | None = Field(
        default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,127}$"
    )
    timeout_seconds: int = Field(default=30, ge=1, le=300)
    context_window: int = Field(default=65536, ge=4096, le=2000000)
    max_output_tokens: int = Field(default=2048, ge=1, le=32768)
    transport_max_retries: int = Field(default=1, ge=0, le=5)
    enable_thinking: bool = False
    reasoning_effort: str = "high"
    structured_output: Literal["json_schema", "json_object", "none"] = "json_object"
    thinking_style: Literal[
        "qwen_chat_template", "openai_reasoning", "anthropic", "none"
    ] = "none"

    @model_validator(mode="after")
    def safe_url(self):
        parsed = urlparse(self.api_base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "model URL must be HTTP(S), without credentials, query or fragment"
            )
        return self

    def profile(self, source: str) -> dict:
        if source == "fixture_model":
            if urlparse(self.api_base_url).hostname not in {
                "127.0.0.1",
                "localhost",
                "::1",
            }:
                raise ValueError("fixture model must use a loopback endpoint")
            key = {"api_key": "runtime-fixture"}
        elif source == "real_model":
            if not self.api_key_env:
                raise ValueError("real model requires an explicit api_key_env")
            key = {"api_key_env": self.api_key_env}
        else:
            raise ValueError("model source must be fixture_model or real_model")
        return {**self.model_dump(exclude_none=True), **key, "enabled": True}


def digest(value: bytes | dict | list) -> str:
    data = (
        value
        if isinstance(value, bytes)
        else json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    )
    return hashlib.sha256(data).hexdigest()


def load_suite(path: Path) -> RuntimeSuite:
    return RuntimeSuite.model_validate_json(path.read_bytes())
