from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pypmml import Model


def _is_missing_score(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _score_field_aliases(field: str) -> list[str]:
    aliases = [field]
    if field.startswith("probability_"):
        target = field.removeprefix("probability_")
        aliases.append(f"probability({target})")
        try:
            aliases.append(f"probability({float(target):.1f})")
        except ValueError:
            pass
    return list(dict.fromkeys(aliases))


def _prediction_score_value(prediction: Any, preferred_field: str) -> Any:
    aliases = _score_field_aliases(preferred_field)
    try:
        prediction_keys = set(prediction.keys())
    except AttributeError:
        prediction_keys = set()
    missing_value = None
    has_missing_value = False

    for field in aliases:
        if prediction_keys and field not in prediction_keys:
            continue
        try:
            value = prediction[field]
        except (KeyError, TypeError):
            continue
        if _is_missing_score(value):
            missing_value = value
            has_missing_value = True
            continue
        return value

    if has_missing_value:
        return missing_value
    return prediction[preferred_field]


@dataclass(frozen=True)
class PmmlScorer:
    model: Model
    positive_output_field: str

    def score(self, dataframe: pd.DataFrame) -> list[float | None]:
        records = dataframe.to_dict(orient="records")
        scores: list[float | None] = []
        for record in records:
            prediction = self.model.predict(record)
            raw_score = _prediction_score_value(prediction, self.positive_output_field)
            scores.append(None if _is_missing_score(raw_score) else float(raw_score))
        return scores


def load_pmml_scorer(pmml_path: Path, positive_output_field: str = "probability_1") -> PmmlScorer:
    model = Model.fromFile(str(pmml_path))
    return PmmlScorer(model=model, positive_output_field=positive_output_field)
