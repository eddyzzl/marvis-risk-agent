"""Dedicated extension case carrier; lightweight inputs avoid common cycles."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .runtime_contracts import Material, RuntimeBudget, RuntimeTask
from .runtime_extension_inputs import ExtensionBusinessInput


class ExtensionBusinessCase(BaseModel):
    """Temporary dedicated case; the common runner owns formal RuntimeCase wiring."""
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    revision: str = Field(min_length=1, max_length=160)
    family: Literal["online_decision", "connector_antifraud", "collections"]
    case_set: Literal["development", "fixed_regression", "independently_held_hidden_acceptance"] = "development"
    scenario: Literal["normal", "clarification", "rejection", "recovery"] = "normal"
    task: RuntimeTask
    materials: list[Material] = Field(default_factory=list, max_length=32)
    actions: list[ExtensionBusinessInput] = Field(min_length=1, max_length=100)
    budget: RuntimeBudget = Field(default_factory=RuntimeBudget)
    business_constraints_source: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def bounded_task(self):
        if self.task.task_type != "strategy":
            raise ValueError("current extension APIs use a registered strategy task; no invented task type")
        return self
