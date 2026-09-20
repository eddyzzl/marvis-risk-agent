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

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class Material(StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    role: Literal["sample", "feature", "unknown"] = "sample"
    source_kind: Literal["synthetic", "deidentified_historical"] = "synthetic"

    @model_validator(mode="after")
    def relative_path(self):
        path = Path(self.path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.suffix.lower() not in {".csv", ".parquet", ".xlsx"}
        ):
            raise ValueError("material must be a relative CSV, Parquet or XLSX path")
        return self


class RuntimeTask(StrictModel):
    task_type: Literal[
        "data_join", "feature_analysis", "modeling", "strategy", "vintage", "portfolio"
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


class RuntimeAction(StrictModel):
    kind: Literal["message", "approve_step", "replay_approval", "retry_step", "stop"]
    content: str = ""
    tool: str = ""

    @model_validator(mode="after")
    def required_fields(self):
        if self.kind in {"message", "approve_step"} and not self.content.strip():
            raise ValueError(
                "a user message / approval requires explicit business text"
            )
        if self.kind in {"approve_step", "retry_step"} and not self.tool.strip():
            raise ValueError("a step action requires a tool reference")
        return self


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
