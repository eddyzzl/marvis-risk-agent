"""Future-only fixed reference faults, without routes, callbacks or code."""
from typing import Literal

from pydantic import BaseModel, ConfigDict


class RuntimeFault(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    kind: Literal["stale_step_cas_once", "discard_success_response_once"]
