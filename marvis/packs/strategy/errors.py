class StrategyError(ValueError):
    pass


class StrategyNotAdoptedError(StrategyError):
    """S5: raised when strategy monitoring is requested for a strategy that is not
    adopted. Monitoring is only meaningful against an adopted (live) strategy --
    a draft/retired strategy has no live production behaviour to compare a fresh
    run against. Carries ``to_detail()`` so the subprocess boundary tags the tool
    result with error_kind='strategy_not_adopted' (structured, never parsed from
    free text -- the NanLabelNotConfirmedError precedent)."""

    def __init__(self, *, strategy_id: str, status: str | None = None) -> None:
        self.strategy_id = str(strategy_id)
        self.status = str(status) if status else None
        detail = f"（当前状态 {self.status}）" if self.status else ""
        super().__init__(
            f"仅对已采纳策略执行监控：策略 {self.strategy_id} 未采纳{detail}。"
        )

    def to_detail(self) -> dict:
        return {
            "kind": "strategy_not_adopted",
            "strategy_id": self.strategy_id,
            "status": self.status,
        }


__all__ = ["StrategyError", "StrategyNotAdoptedError"]
