from marvis.errors import ErrorKind


class AnalysisError(ValueError):
    """Base error for the portfolio analysis pack (non-typed, plain message)."""


class MissingBaselineError(AnalysisError):
    """S3: raised when a trend tool's experiment has no training-time baseline
    distribution snapshot to compare against.

    Typed (to_detail() -> kind="missing_baseline") like the data-layer errors so
    it surfaces as a structured error_kind through the subprocess runner rather
    than a parsed message.
    """

    def __init__(self, *, experiment_id: str, reason: str) -> None:
        self.experiment_id = str(experiment_id)
        self.reason = str(reason)
        super().__init__(self.reason)

    def to_detail(self) -> dict:
        return {
            "kind": ErrorKind.MISSING_BASELINE,
            "experiment_id": self.experiment_id,
            "reason": self.reason,
        }


__all__ = ["AnalysisError", "MissingBaselineError"]
