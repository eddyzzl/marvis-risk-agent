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


__all__ = [
    "DraftError",
    "DraftNotFound",
    "DraftStateError",
    "FetchError",
    "OfflineError",
]
