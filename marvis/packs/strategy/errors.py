from marvis.error_kinds import ErrorKind


class StrategyError(ValueError):
    pass


class StrategyNotAdoptedError(StrategyError):
    """S5: raised when strategy monitoring is requested for a strategy that is not
    locally adopted. A draft/validated/retired asset is not the local champion.
    Local adoption is deliberately not represented as production deployment.
    Carries ``to_detail()`` so the subprocess boundary tags the tool
    result with error_kind='strategy_not_adopted' (structured, never parsed from
    free text -- the NanLabelNotConfirmedError precedent)."""

    def __init__(
        self,
        *,
        strategy_id: str,
        status: str | None = None,
        asset_status: str | None = None,
    ) -> None:
        self.strategy_id = str(strategy_id)
        self.status = str(status) if status else None
        self.asset_status = str(asset_status) if asset_status else None
        current = self.asset_status or self.status
        detail = f"（当前资产状态 {current}）" if current else ""
        super().__init__(
            f"仅对本地已采纳策略执行监控：策略 {self.strategy_id} 未处于"
            f" adopted_local{detail}。本地已采纳，不代表生产上线。"
        )

    def to_detail(self) -> dict:
        return {
            "kind": ErrorKind.STRATEGY_NOT_ADOPTED,
            "strategy_id": self.strategy_id,
            "status": self.status,
            "asset_status": self.asset_status,
        }


class StrategyPoolLegacyDraftNeedsRebuildError(StrategyError):
    """A v1 draft was archived and cannot be interpreted as a v2 Pool."""

    def __init__(self, archive: dict) -> None:
        self.archive = dict(archive)
        super().__init__(
            "archived Strategy Pool v1 draft requires an explicit v2 rebuild"
        )

    def to_detail(self) -> dict:
        return {
            "kind": ErrorKind.LEGACY_POOL_DRAFT_NEEDS_REBUILD,
            "archive": self.archive,
        }


__all__ = [
    "StrategyError",
    "StrategyNotAdoptedError",
    "StrategyPoolLegacyDraftNeedsRebuildError",
]
