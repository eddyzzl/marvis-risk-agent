"""Installed HTTP input contracts shared by manual and proposal entry points."""

from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt
from marvis.operations.bindings import MonitoringBinding
from marvis.operations.extension_bindings import (
    PortfolioBinding,
    DecisionFeedbackBinding,
)


class CalendarPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    anchor_at: datetime
    interval_seconds: StrictInt = Field(ge=1)


class MonthlyCalendarPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    anchor_at: datetime
    interval_months: StrictInt = Field(ge=1, le=120)
    timezone: str


class RetryPolicyPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_attempts: StrictInt = Field(ge=1, le=10)
    initial_backoff_seconds: StrictFloat = Field(ge=0)
    multiplier: StrictFloat = Field(gt=0)
    max_backoff_seconds: StrictFloat = Field(ge=0)


class SchedulePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schedule_id: str = Field(min_length=1, max_length=200)
    revision: StrictInt = Field(ge=1)
    monitoring_ref: str = Field(min_length=1, max_length=200)
    active_from: datetime
    calendar: CalendarPayload | MonthlyCalendarPayload
    catch_up_budget: StrictInt = Field(ge=1, le=100)
    lease_seconds: StrictInt = Field(ge=1)
    retry_policy: RetryPolicyPayload
    enabled: StrictBool = True
    monitoring_binding: MonitoringBinding | None = None
    portfolio_binding: PortfolioBinding | None = None
    feedback_binding: DecisionFeedbackBinding | None = None
    active_until: datetime | None = None
