class DraftError(ValueError):
    pass


class DraftStateError(DraftError):
    pass


class DraftNotFound(DraftError):
    pass


class OfflineError(DraftError):
    pass


class FetchError(DraftError):
    pass


class OutboundQueryError(DraftError):
    """A search query failed the outbound content boundary; nothing was sent."""


class AuthoringError(DraftError):
    pass


class PromotionError(DraftError):
    pass


__all__ = [
    "AuthoringError",
    "DraftError",
    "DraftNotFound",
    "DraftStateError",
    "FetchError",
    "OfflineError",
    "OutboundQueryError",
    "PromotionError",
]
