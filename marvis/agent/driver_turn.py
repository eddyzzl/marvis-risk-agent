"""Shared driver turn/message DTOs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DriverMessage:
    """One append-only assistant message returned by the plan driver."""

    stage: str
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class DriverTurn:
    plan_id: str
    status: str
    messages: list[DriverMessage] = field(default_factory=list)


__all__ = ["DriverMessage", "DriverTurn"]
