from __future__ import annotations


class FeatureError(RuntimeError):
    """Base error for deterministic feature analysis and transformation."""


class BinningError(FeatureError):
    """Raised when feature binning cannot produce valid bin edges."""


class FitRequiresSplitError(FeatureError):
    """Raised when a fit-class transform (WOE/impute/normalize/cap/select) has no
    ``split_col`` to exclude holdout rows from fitting.

    Fitting on the full pool (including future test/OOT rows) leaks distribution —
    and, for WOE/select, label — information into the transform, inflating downstream
    evaluation metrics (PREP-1 / FS-2). By default this stops and asks the caller to
    either split first (``make_split`` / ``prepare_modeling_frame``) or explicitly
    confirm a full-pool fit via ``allow_full_fit=true``. Mirrors
    :class:`marvis.data.errors.NanLabelNotConfirmedError`.
    """

    def __init__(self, *, tool: str, dataset_id: str) -> None:
        self.tool = str(tool)
        self.dataset_id = str(dataset_id)
        super().__init__(
            f"{self.tool} has no split_col to exclude holdout rows from fitting "
            f"(dataset {self.dataset_id!r}); split first (make_split or "
            "prepare_modeling_frame) and pass split_col, or pass allow_full_fit=true "
            "to explicitly confirm fitting on the full pool"
        )

    def to_detail(self) -> dict:
        """Structured diagnostics (never parsed from free text)."""
        return {
            "kind": "fit_requires_split",
            "tool": self.tool,
            "dataset_id": self.dataset_id,
        }


__all__ = ["BinningError", "FeatureError", "FitRequiresSplitError"]
