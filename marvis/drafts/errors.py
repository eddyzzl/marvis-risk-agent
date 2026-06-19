class DraftError(ValueError):
    pass


class DraftStateError(DraftError):
    pass


__all__ = ["DraftError", "DraftStateError"]
