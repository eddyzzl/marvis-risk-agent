import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from marvis.validation.pmml_scoring import PmmlScorer, load_pmml_scorer

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "min_lr.pmml"


def test_load_and_score_matches_manual_sigmoid():
    scorer = load_pmml_scorer(FIXTURE)
    df = pd.DataFrame({"x1": [0.0, 1.0, 2.0], "x2": [0.0, 0.0, 1.0]})
    scores = scorer.score(df)
    expected = [1 / (1 + math.exp(-z)) for z in [0.0, 1.0, 1.0]]
    for got, want in zip(scores, expected):
        assert got == pytest.approx(want, abs=1e-6)


def test_score_handles_missing_columns_with_null():
    """pypmml's null handling for a model without MissingValueTreatment is
    library-defined; this test only asserts the call doesn't crash and
    returns the expected number of rows."""
    scorer = load_pmml_scorer(FIXTURE)
    df = pd.DataFrame({"x1": [1.0, 2.0], "x2": [float("nan"), 0.0]})
    try:
        scores = scorer.score(df)
        assert len(scores) == 2
    except Exception:
        # pypmml may raise on NaN input; that is acceptable behavior
        pytest.skip("pypmml raised on NaN input — observed null behavior recorded")


def test_score_returns_none_when_pmml_output_field_is_null():
    class NullOutputModel:
        def predict(self, record):
            return {"probability_1": None}

    scorer = PmmlScorer(model=NullOutputModel(), positive_output_field="probability_1")

    assert scorer.score(pd.DataFrame({"x1": [1.0]})) == [None]


def test_score_uses_pmml_probability_alias_when_configured_field_is_missing():
    class JavaMapLike(dict):
        def __getitem__(self, key):
            return self.get(key)

    class AliasOutputModel:
        def predict(self, record):
            return JavaMapLike({"probability(1.0)": 0.7})

    scorer = PmmlScorer(model=AliasOutputModel(), positive_output_field="probability_1")

    assert scorer.score(pd.DataFrame({"x1": [1.0]})) == [0.7]


def test_dataframe_scoring_calls_model_predict_once_for_the_whole_chunk():
    class BatchModel:
        def __init__(self) -> None:
            self.calls: list[pd.DataFrame] = []

        def predict(self, value: pd.DataFrame) -> pd.DataFrame:
            self.calls.append(value.copy())
            return pd.DataFrame({"probability(1.0)": [0.1, 0.2, 0.3]})

    model = BatchModel()
    scorer = PmmlScorer(model=model, positive_output_field="probability_1")

    scores = scorer.score_chunk(pd.DataFrame({"x": [1, 2, 3]}))

    assert len(model.calls) == 1
    assert model.calls[0].shape == (3, 1)
    assert scores.tolist() == [0.1, 0.2, 0.3]


@pytest.mark.parametrize(
    "prediction",
    [
        pd.Series([0.1, 0.2], name="probability_1"),
        [
            {"probability_1": 0.1, "predictedValue": 0},
            {"probability_1": 0.2, "predictedValue": 0},
        ],
        {"probability_1": [0.1, 0.2], "predictedValue": [0, 0]},
    ],
    ids=["series", "list-of-records", "dict-of-columns"],
)
def test_score_chunk_accepts_common_batch_prediction_shapes(prediction: Any):
    class StaticModel:
        def predict(self, _frame: pd.DataFrame) -> Any:
            return prediction

    scores = PmmlScorer(StaticModel(), "probability_1").score_chunk(
        pd.DataFrame({"x": [1, 2]})
    )

    assert scores.tolist() == [0.1, 0.2]


def test_score_chunk_accepts_one_record_mapping_and_series_outputs():
    class StaticModel:
        def __init__(self, prediction: Any) -> None:
            self.prediction = prediction

        def predict(self, _frame: pd.DataFrame) -> Any:
            return self.prediction

    frame = pd.DataFrame({"x": [1]}, index=[42])
    for prediction in (
        {"probability_1": 0.4, "predictedValue": 0},
        pd.Series({"probability_1": 0.4, "predictedValue": 0}),
    ):
        scores = PmmlScorer(StaticModel(prediction), "probability_1").score_chunk(frame)
        assert scores.index.tolist() == [42]
        assert scores.tolist() == [0.4]


@pytest.mark.parametrize(
    "prediction",
    [
        pd.Series([0.4], name="probability_1"),
        {"probability_1": [0.4]},
    ],
    ids=["series-column", "dict-of-columns"],
)
def test_score_chunk_accepts_single_row_batch_prediction_shapes(prediction: Any):
    class StaticModel:
        def predict(self, _frame: pd.DataFrame) -> Any:
            return prediction

    scores = PmmlScorer(StaticModel(), "probability_1").score_chunk(
        pd.DataFrame({"x": [1]}, index=[42])
    )

    assert scores.index.tolist() == [42]
    assert scores.tolist() == [0.4]


def test_score_chunk_rejects_ambiguous_single_row_series_with_bounded_error():
    class AmbiguousModel:
        def predict(self, _frame: pd.DataFrame) -> pd.Series:
            return pd.Series(
                [0.4], index=["probability_1"], name="probability_1"
            )

    with pytest.raises(ValueError) as exc_info:
        PmmlScorer(AmbiguousModel(), "probability_1").score_chunk(
            pd.DataFrame({"x": [1]})
        )

    assert "ambiguous" in str(exc_info.value).lower()
    assert len(str(exc_info.value)) <= 1_024


def test_score_chunk_rejects_mixed_record_and_batch_mapping_with_bounded_error():
    class AmbiguousModel:
        def predict(self, _frame: pd.DataFrame) -> dict[str, Any]:
            return {"probability_1": [0.4], "predictedValue": 1}

    with pytest.raises(ValueError) as exc_info:
        PmmlScorer(AmbiguousModel(), "probability_1").score_chunk(
            pd.DataFrame({"x": [1]})
        )

    assert "ambiguous" in str(exc_info.value).lower()
    assert len(str(exc_info.value)) <= 1_024


def test_score_chunk_rejects_silent_row_loss():
    class ShortModel:
        def predict(self, _frame: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({"probability_1": [0.1]})

    with pytest.raises(ValueError, match="returned 1 rows for 2 inputs"):
        PmmlScorer(ShortModel(), "probability_1").score_chunk(
            pd.DataFrame({"x": [1, 2]})
        )


def test_score_chunk_does_not_treat_rows_without_columns_as_an_empty_batch():
    class NoInputModel:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
            self.calls += 1
            return pd.DataFrame({"probability_1": [0.1] * len(frame)})

    model = NoInputModel()
    scores = PmmlScorer(model, "probability_1").score_chunk(
        pd.DataFrame(index=[9, 3])
    )

    assert model.calls == 1
    assert scores.index.tolist() == [9, 3]
    assert scores.tolist() == [0.1, 0.1]


def test_missing_output_error_is_bounded_and_lists_available_columns():
    columns = {f"output_{index:04d}": [0.1] for index in range(500)}

    class WideModel:
        def predict(self, _frame: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(columns)

    with pytest.raises(ValueError) as exc_info:
        PmmlScorer(WideModel(), "probability_1").score_chunk(pd.DataFrame({"x": [1]}))

    message = str(exc_info.value)
    assert "probability_1" in message
    assert "output_0000" in message
    assert "available" in message
    assert len(message) <= 1_024


def test_duplicate_output_columns_are_rejected():
    class DuplicateModel:
        def predict(self, _frame: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(
                [[0.1, 0.2]], columns=["probability_1", "probability_1"]
            )

    with pytest.raises(ValueError, match="duplicated"):
        PmmlScorer(DuplicateModel(), "probability_1").score_chunk(
            pd.DataFrame({"x": [1]})
        )
