from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HoldoutError(RuntimeError):
    def __init__(self, code, status=409):
        super().__init__(code)
        self.code, self.status = code, status


class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class Intake(Contract):
    data_identity: str = Field(min_length=1, max_length=160, pattern=r'^[A-Za-z0-9_.:-]+$')
    developer_id: str = Field(min_length=1, max_length=160)
    evaluator_id: str = Field(min_length=1, max_length=160)
    target_col: str = Field(min_length=1, max_length=240)
    split_col: str = Field(min_length=1, max_length=240)
    target_type: Literal['binary', 'continuous', 'multiclass'] = 'binary'
    authority_basis: str = Field(min_length=1, max_length=2000)
    # User assertions remain declarations; custody cannot prove historical truth.
    historical_availability: Literal['unknown'] = 'unknown'
    label_maturity: Literal['unknown'] = 'unknown'

    @field_validator('authority_basis')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('authority basis required')
        return value.strip()


class Freeze(Contract):
    artifact_id: str = Field(min_length=1, max_length=160)


class Evaluate(Contract):
    freeze_hash: str = Field(pattern=r'^[a-f0-9]{64}$')
    request_id: str = Field(default='evaluation', min_length=1, max_length=160)
    recovery_id: str | None = Field(default=None, min_length=1, max_length=160)


class Recover(Contract):
    mode: Literal['resume_before_read', 'publish_result']
    reason: str = Field(min_length=1, max_length=2000)

    @field_validator('reason')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('recovery reason required')
        return value.strip()
