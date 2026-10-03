"""Compact source-row usage captured by the selection algorithm itself."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SelectionScope:
    row_count: int
    fit: bytes
    label_diagnostics: bytes
    value_diagnostics: bytes

    @classmethod
    def capture(cls, fit, *, label_diagnostics=None, value_diagnostics=None):
        fit = np.asarray(fit, dtype=bool)
        if fit.ndim != 1:
            raise ValueError("selection membership must be one-dimensional")

        def packed(mask):
            mask = np.zeros(len(fit), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
            if mask.shape != fit.shape:
                raise ValueError("selection memberships must share source row identity")
            return np.packbits(mask, bitorder="little").tobytes()

        return cls(len(fit), packed(fit), packed(label_diagnostics), packed(value_diagnostics))
