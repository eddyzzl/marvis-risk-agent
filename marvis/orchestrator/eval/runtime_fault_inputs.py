"""Future-only fixed reference faults, without routes, callbacks or code."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_serializer, model_validator


class RuntimeFault(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    kind: Literal["stale_step_cas_once", "discard_success_response_once", "stale_report_revision_once",
                  "worker_crash_after_commit_once", "tool_timeout_once",
                  "service_restart_mid_step_once", "cancel_then_resume_once", "stale_monitoring_revision_once"]
    downstream_tool: Literal["analysis.portfolio_report", "modeling.train_models"] | None = None

    @model_serializer(mode="wrap")
    def preserve_original_fault_wire(self, handler):
        value = handler(self)
        if self.downstream_tool is None:
            value.pop("downstream_tool", None)
        return value

    @model_validator(mode="after")
    def fixed_downstream_commit_boundary(self):
        if self.downstream_tool is not None and self.kind != "worker_crash_after_commit_once":
            raise ValueError("fixed downstream target requires the after-commit worker fault")
        return self
