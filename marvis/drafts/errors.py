class DraftError(ValueError):
    pass


class DraftStateError(DraftError):
    pass


class DraftNotFound(DraftError):
    pass


__all__ = ["DraftError", "DraftNotFound", "DraftStateError"]
