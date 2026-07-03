from marvis.errors import ErrorKind


class ModelingError(ValueError):
    pass


class ReportScoreMissingError(ModelingError):
    """Raised when generate_model_report has no way to obtain a real model score.

    Neither an explicit ``score`` column nor a trained ``artifact_id`` is available, so
    there is no model score to report on. Silently substituting the first feature column
    would compute plausible-looking KS/PSI/bin numbers with the wrong semantics (DOM-10) —
    the report is a formal deliverable, so this must fail loudly instead.
    """

    def __init__(self, *, experiment_id: str, dataset_id: str) -> None:
        self.experiment_id = str(experiment_id)
        self.dataset_id = str(dataset_id)
        super().__init__(
            f"experiment {self.experiment_id!r} has no artifact and dataset "
            f"{self.dataset_id!r} has no `score` column; cannot generate score-based "
            "report sections. Train a model for this experiment first, or register a "
            "dataset that already carries a `score` column."
        )

    def to_detail(self) -> dict:
        """Structured diagnostics (never parsed from free text)."""
        return {
            "kind": ErrorKind.REPORT_SCORE_MISSING,
            "experiment_id": self.experiment_id,
            "dataset_id": self.dataset_id,
        }


__all__ = ["ModelingError", "ReportScoreMissingError"]
