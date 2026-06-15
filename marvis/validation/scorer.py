from __future__ import annotations

from typing import Protocol

import pandas as pd


class Scorer(Protocol):
    """Anything that turns a feature dataframe into a list of positive-class scores."""

    def score(self, dataframe: pd.DataFrame) -> list[float | None]: ...
