from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import threading
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

    def score_chunk(self, dataframe: pd.DataFrame) -> pd.Series:
        if len(dataframe) == 0:
            return pd.Series(
                index=dataframe.index,
                dtype="float64",
                name="pmml_score",
            )
        prediction = self.model.predict(dataframe)
        frame = _prediction_frame(
            prediction,
            expected_rows=len(dataframe),
            preferred_field=self.positive_output_field,
        )
        if len(frame) != len(dataframe):
            raise ValueError(
                f"PMML scorer returned {len(frame)} rows for {len(dataframe)} inputs"
            )
        output_column = _resolve_output_column(
            frame.columns, self.positive_output_field
        )
        values = pd.to_numeric(frame[output_column], errors="coerce")
        return pd.Series(
            values.to_numpy(),
            index=dataframe.index,
            name="pmml_score",
        )

    def score(self, dataframe: pd.DataFrame) -> list[float | None]:
        values = self.score_chunk(dataframe)
        return [
            None if _is_missing_score(value) else float(value)
            for value in values.to_numpy()
        ]


class TaskPmmlScorerRegistry:
    """Bounded process-local reuse across baseline and stress stages."""

    def __init__(self, max_tasks: int = 4):
        if isinstance(max_tasks, bool) or not isinstance(max_tasks, int) or max_tasks <= 0:
            raise ValueError("max_tasks must be a positive integer")
        self.max_tasks = max_tasks
        self.lock = threading.RLock()
        self.entries: OrderedDict[
            str, tuple[tuple[str, str], PmmlScorer]
        ] = OrderedDict()

    def get(
        self,
        *,
        task_id: str,
        pmml_path: Path,
        pmml_sha256: str,
        output_field: str,
    ) -> PmmlScorer:
        identity = (pmml_sha256, output_field)
        with self.lock:
            current = self.entries.get(task_id)
            if current is not None and current[0] == identity:
                self.entries.move_to_end(task_id)
                return current[1]

        # Loading starts a JVM-backed model and may be slow. Do not serialize model
        # loads for unrelated tasks behind the registry's global lock.
        loaded = PmmlScorer(
            model=Model.fromFile(str(pmml_path)),
            positive_output_field=output_field,
        )

        with self.lock:
            # A concurrent caller may have populated the same identity while this
            # model was loading. Prefer the already-published scorer in that case.
            current = self.entries.get(task_id)
            if current is not None and current[0] == identity:
                self.entries.move_to_end(task_id)
                return current[1]
            self.entries[task_id] = (identity, loaded)
            self.entries.move_to_end(task_id)
            while len(self.entries) > self.max_tasks:
                self.entries.popitem(last=False)
            return loaded

    def clear(self, task_id: str) -> None:
        with self.lock:
            self.entries.pop(task_id, None)


TASK_PMML_SCORERS = TaskPmmlScorerRegistry()


def _prediction_frame(
    prediction: Any,
    *,
    expected_rows: int,
    preferred_field: str,
) -> pd.DataFrame:
    if isinstance(prediction, pd.DataFrame):
        return prediction.reset_index(drop=True)
    if isinstance(prediction, pd.Series):
        aliases = set(_score_field_aliases(preferred_field))
        output_on_index = any(str(value) in aliases for value in prediction.index)
        output_on_name = (
            prediction.name is not None and str(prediction.name) in aliases
        )
        if expected_rows == 1 and output_on_index and output_on_name:
            raise ValueError(
                _bounded_error(
                    "ambiguous single-row PMML Series prediction: the selected "
                    "output field appears on both the index and the Series name"
                )
            )
        if expected_rows == 1 and not output_on_name:
            frame = prediction.to_frame().T
        else:
            frame = prediction.to_frame()
        return frame.reset_index(drop=True)
    if isinstance(prediction, list):
        try:
            return pd.DataFrame.from_records(prediction).reset_index(drop=True)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "unsupported PMML batch prediction list structure"
            ) from exc
    if isinstance(prediction, Mapping):
        values = dict(prediction)
        column_shapes = [_is_column_value(value) for value in values.values()]
        if any(column_shapes) and not all(column_shapes):
            raise ValueError(
                _bounded_error(
                    "ambiguous PMML mapping prediction: it mixes scalar record "
                    "values with batch column values"
                )
            )
        if not any(column_shapes):
            return pd.DataFrame.from_records([values])
        try:
            return pd.DataFrame(values).reset_index(drop=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                _bounded_error("invalid PMML mapping batch column lengths")
            ) from exc
    raise TypeError(
        "unsupported PMML batch prediction type: "
        f"{type(prediction).__name__[:128]}"
    )


def _is_column_value(value: Any) -> bool:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return False
    return bool(pd.api.types.is_list_like(value))


def _resolve_output_column(columns: Any, preferred_field: str) -> Any:
    available = list(columns)
    for alias in _score_field_aliases(preferred_field):
        matching = [column for column in available if str(column) == alias]
        if len(matching) == 1:
            return matching[0]
        if len(matching) > 1:
            raise ValueError(
                _bounded_error(
                    f"PMML output column {alias!r} is duplicated; "
                    f"available columns: {_column_summary(available)}"
                )
            )
    raise ValueError(
        _bounded_error(
            f"PMML output field {preferred_field!r} was not returned; "
            f"available columns: {_column_summary(available)}"
        )
    )


def _column_summary(columns: list[Any], *, max_columns: int = 24) -> str:
    values = [repr(str(column)[:128]) for column in columns[:max_columns]]
    if len(columns) > max_columns:
        values.append(f"... {len(columns) - max_columns} more")
    return ", ".join(values) if values else "<none>"


def _bounded_error(message: str, *, limit: int = 1_024) -> str:
    if len(message) <= limit:
        return message
    suffix = "... [truncated]"
    return message[: limit - len(suffix)] + suffix


def load_pmml_scorer(pmml_path: Path, positive_output_field: str = "probability_1") -> PmmlScorer:
    model = Model.fromFile(str(pmml_path))
    return PmmlScorer(model=model, positive_output_field=positive_output_field)
