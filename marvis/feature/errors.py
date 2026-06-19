from __future__ import annotations


class FeatureError(RuntimeError):
    """Base error for deterministic feature analysis and transformation."""


class BinningError(FeatureError):
    """Raised when feature binning cannot produce valid bin edges."""


__all__ = ["BinningError", "FeatureError"]
