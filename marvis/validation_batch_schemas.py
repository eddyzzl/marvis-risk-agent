from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class ValidationBatchItemRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(min_length=1)
    model_version: str = ""
    source_dir: str = Field(min_length=1)
    upload_token: str | None = Field(default=None, min_length=1, max_length=64)
    algorithm: str = ""
    target_col: str = "y"
    score_col: str = "pred"
    split_col: str = "split"
    time_col: str = "apply_month"
    notebook_path: str = Field(min_length=1)
    sample_path: str = Field(min_length=1)
    pmml_path: str = Field(min_length=1)
    dictionary_path: str = Field(min_length=1)


class CreateValidationBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_name: str = Field(min_length=1)
    validator: str = Field(min_length=1)
    run_mode: Literal["agent"] = "agent"
    items: Annotated[
        list[ValidationBatchItemRequest],
        Field(min_length=1, max_length=10),
    ]


class StartValidationBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str | None = None
    effort: str | None = None


__all__ = [
    "CreateValidationBatchRequest",
    "StartValidationBatchRequest",
    "ValidationBatchItemRequest",
]
