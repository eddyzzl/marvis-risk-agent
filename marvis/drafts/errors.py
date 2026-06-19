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


class AuthoringError(DraftError):
    pass


__all__ = [
    "AuthoringError",
    "DraftError",
    "DraftNotFound",
    "DraftStateError",
    "FetchError",
    "OfflineError",
]
