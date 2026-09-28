from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class DecisionError(ValueError):
    def __init__(self, code: str, status: int = 422):
        self.code, self.status = code, status
        super().__init__(code)


class FeatureField(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(min_length=1, max_length=160)
    type: Literal["number", "integer", "string", "boolean"]
    nullable: bool = False


class PackageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    strategy_id: str = Field(min_length=1, max_length=160)
    strategy_version: int = Field(ge=1)
    model_artifact_id: str = Field(min_length=1, max_length=160)
    decision_node: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    score_field: str = Field(min_length=1, max_length=160)
    score_product: Literal["raw_pd", "calibrated_pd", "scorecard_points"] = "raw_pd"
    raw_schema: list[FeatureField] = Field(min_length=1, max_length=500)
    timeout_seconds: int = Field(default=10, ge=1, le=60)
    # A fallback is part of the package approved through the ordinary promotion.
    failure_action: Literal["review", "reject"] = "review"

    @model_validator(mode="after")
    def names(self):
        names = [f.name for f in self.raw_schema]
        if len(set(names)) != len(names) or self.score_field in names:
            raise ValueError(
                "raw feature names must be unique and exclude computed score"
            )
        if any(n.startswith("__marvis_") for n in names):
            raise ValueError(
                "platform derived fields cannot be supplied as raw features"
            )
        return self


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    request_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]{1,128}$")
    decision_node: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    expected_package_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    features: dict[str, Any] = Field(max_length=500)


def validate_features(features: dict, manifest: dict) -> None:
    fields = manifest["configuration"]["raw_schema"]
    if set(features) != {f["name"] for f in fields}:
        raise DecisionError("raw_schema_mismatch")
    for field in fields:
        value = features[field["name"]]
        if value is None and field["nullable"]:
            continue
        kind = field["type"]
        valid = (
            (kind == "number" and type(value) in (int, float) and math.isfinite(value))
            or (kind == "integer" and type(value) is int)
            or (kind == "string" and isinstance(value, str) and len(value) <= 4096)
            or (kind == "boolean" and type(value) is bool)
        )
        if not valid:
            raise DecisionError("raw_feature_type_mismatch")
    if len(canonical(features).encode()) > 64_000:
        raise DecisionError("feature_payload_too_large", 413)
