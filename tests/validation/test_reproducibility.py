import pandas as pd
import pytest

from marvis.validation.config import ValidationConfig
from marvis.validation.reproducibility import (
    scores_match_at_precision,
    run_reproducibility,
)
from marvis.validation.results import ConsistencyStatus


class _FakeScorer:
    def __init__(self, scores_by_row_index: dict[int, float | None]):
        self._scores = scores_by_row_index

    def score(self, df: pd.DataFrame) -> list[float | None]:
        return [self._scores[idx] for idx in df.index]


def _config(**overrides) -> ValidationConfig:
    base = dict(
        target_col="y",
        score_col="sample_score",
        split_col="split",
        time_col="apply_month",
        feature_columns=["x1", "x2"],
        random_sample_size=3,
        random_seed=42,
        score_decimal_places=6,
    )
    base.update(overrides)
    return ValidationConfig(**base)


def test_three_scores_all_match_returns_pass():
    sample = pd.DataFrame({
        "x1": [0.1, 0.2, 0.3],
        "x2": [0.0, 0.0, 0.0],
        "sample_score": [0.5, 0.55, 0.6],
        "y": [0, 1, 0],
        "split": ["train"] * 3,
        "apply_month": ["202503"] * 3,
    })
    code_scores = pd.Series({0: 0.5, 1: 0.55, 2: 0.6})
    submitted_pmml = _FakeScorer({0: 0.5, 1: 0.55, 2: 0.6})

    result = run_reproducibility(
        sample=sample,
        config=_config(),
        code_scores=code_scores,
        submitted_pmml_scorer=submitted_pmml,
    )

    assert result.sample_size == 3
    assert result.summary.match_count == 3
    assert result.summary.mismatch_count == 0
    assert result.summary.status is ConsistencyStatus.PASS
    assert result.rows[0].score_code_model == 0.5
    assert result.rows[0].score_submitted_pmml == 0.5


def test_mismatch_beyond_decimal_returns_fail():
    sample = pd.DataFrame({
        "x1": [0.1, 0.2, 0.3],
        "x2": [0.0, 0.0, 0.0],
        "sample_score": [0.500000, 0.550000, 0.600000],
        "y": [0, 1, 0],
        "split": ["train"] * 3,
        "apply_month": ["202503"] * 3,
    })
    code_scores = pd.Series({0: 0.5, 1: 0.55, 2: 0.6})
    submitted_pmml = _FakeScorer({0: 0.5, 1: 0.55, 2: 0.7})  # row 2 diverges

    result = run_reproducibility(
        sample=sample,
        config=_config(),
        code_scores=code_scores,
        submitted_pmml_scorer=submitted_pmml,
    )

    assert result.summary.mismatch_count == 1
    assert result.summary.status is ConsistencyStatus.FAIL
    assert result.summary.max_abs_diff > 0.05


def test_small_low_ratio_tiny_rounded_mismatch_returns_pass():
    row_count = 200
    sample = pd.DataFrame({
        "x1": [0.1] * row_count,
        "x2": [0.0] * row_count,
        "sample_score": [0.5] * row_count,
        "y": [0] * row_count,
        "split": ["train"] * row_count,
        "apply_month": ["202503"] * row_count,
    })
    code_scores = pd.Series({idx: 0.5 for idx in range(row_count)})
    submitted_scores = {idx: 0.5 for idx in range(row_count)}
    submitted_scores[0] = 0.50005

    result = run_reproducibility(
        sample=sample,
        config=_config(random_sample_size=row_count),
        code_scores=code_scores,
        submitted_pmml_scorer=_FakeScorer(submitted_scores),
    )

    assert result.summary.mismatch_count == 1
    assert result.summary.max_abs_diff == pytest.approx(0.00005)
    assert result.summary.status is ConsistencyStatus.PASS


def test_ninety_eight_percent_tiny_rounded_match_rate_returns_pass():
    row_count = 1000
    sample = pd.DataFrame({
        "x1": [0.1] * row_count,
        "x2": [0.0] * row_count,
        "sample_score": [0.5] * row_count,
        "y": [0] * row_count,
        "split": ["train"] * row_count,
        "apply_month": ["202503"] * row_count,
    })
    code_scores = pd.Series({idx: 0.5 for idx in range(row_count)})
    submitted_scores = {idx: 0.5 for idx in range(row_count)}
    for idx in range(20):
        submitted_scores[idx] = 0.50005

    result = run_reproducibility(
        sample=sample,
        config=_config(random_sample_size=row_count),
        code_scores=code_scores,
        submitted_pmml_scorer=_FakeScorer(submitted_scores),
    )

    assert result.summary.mismatch_count == 20
    assert result.summary.max_abs_diff == pytest.approx(0.00005)
    assert result.summary.status is ConsistencyStatus.PASS


def test_ninety_five_to_ninety_eight_percent_tiny_rounded_match_rate_returns_review():
    row_count = 1000
    sample = pd.DataFrame({
        "x1": [0.1] * row_count,
        "x2": [0.0] * row_count,
        "sample_score": [0.5] * row_count,
        "y": [0] * row_count,
        "split": ["train"] * row_count,
        "apply_month": ["202503"] * row_count,
    })
    code_scores = pd.Series({idx: 0.5 for idx in range(row_count)})
    submitted_scores = {idx: 0.5 for idx in range(row_count)}
    for idx in range(40):
        submitted_scores[idx] = 0.50005

    result = run_reproducibility(
        sample=sample,
        config=_config(random_sample_size=row_count),
        code_scores=code_scores,
        submitted_pmml_scorer=_FakeScorer(submitted_scores),
    )

    assert result.summary.mismatch_count == 40
    assert result.summary.max_abs_diff == pytest.approx(0.00005)
    assert result.summary.status is ConsistencyStatus.REVIEW


def test_ninety_nine_percent_tiny_rounded_match_rate_returns_pass():
    row_count = 1000
    sample = pd.DataFrame({
        "x1": [0.1] * row_count,
        "x2": [0.0] * row_count,
        "sample_score": [0.5] * row_count,
        "y": [0] * row_count,
        "split": ["train"] * row_count,
        "apply_month": ["202503"] * row_count,
    })
    code_scores = pd.Series({idx: 0.5 for idx in range(row_count)})
    submitted_scores = {idx: 0.5 for idx in range(row_count)}
    for idx in range(10):
        submitted_scores[idx] = 0.50005

    result = run_reproducibility(
        sample=sample,
        config=_config(random_sample_size=row_count),
        code_scores=code_scores,
        submitted_pmml_scorer=_FakeScorer(submitted_scores),
    )

    assert result.summary.mismatch_count == 10
    assert result.summary.max_abs_diff == pytest.approx(0.00005)
    assert result.summary.status is ConsistencyStatus.PASS


def test_null_submitted_pmml_score_is_reported_as_mismatch():
    sample = pd.DataFrame({
        "x1": [0.1, 0.2, 0.3],
        "x2": [0.0, 0.0, 0.0],
        "sample_score": [0.500000, 0.550000, 0.600000],
        "y": [0, 1, 0],
        "split": ["train"] * 3,
        "apply_month": ["202503"] * 3,
    })
    code_scores = pd.Series({0: 0.5, 1: 0.55, 2: 0.6})
    submitted_pmml = _FakeScorer({0: None, 1: 0.55, 2: 0.6})

    result = run_reproducibility(
        sample=sample,
        config=_config(),
        code_scores=code_scores,
        submitted_pmml_scorer=submitted_pmml,
    )

    null_row = next(row for row in result.rows if row.row_index == 0)
    assert null_row.score_submitted_pmml is None
    assert null_row.abs_diff is None
    assert null_row.matched is False
    assert result.summary.match_count == 2
    assert result.summary.mismatch_count == 1
    assert result.summary.status is ConsistencyStatus.FAIL


def test_sampling_is_deterministic_with_seed():
    sample = pd.DataFrame({
        "x1": list(range(10)),
        "x2": [0] * 10,
        "sample_score": [0.5] * 10,
        "y": [0] * 10,
        "split": ["train"] * 10,
        "apply_month": ["202503"] * 10,
    })
    code_scores = pd.Series({idx: 0.5 for idx in range(10)})
    submitted_pmml = _FakeScorer({idx: 0.5 for idx in range(10)})

    result_a = run_reproducibility(
        sample=sample,
        config=_config(random_sample_size=4),
        code_scores=code_scores,
        submitted_pmml_scorer=submitted_pmml,
    )
    result_b = run_reproducibility(
        sample=sample,
        config=_config(random_sample_size=4),
        code_scores=code_scores,
        submitted_pmml_scorer=submitted_pmml,
    )
    assert [row.row_index for row in result_a.rows] == [row.row_index for row in result_b.rows]


def test_reproducibility_uses_positional_rows_for_non_integer_sample_index():
    sample = pd.DataFrame(
        {
            "x1": [0.1, 0.2, 0.3],
            "x2": [0.0, 0.0, 0.0],
            "sample_score": [0.5, 0.55, 0.6],
            "y": [0, 1, 0],
            "split": ["train"] * 3,
            "apply_month": ["202503"] * 3,
        },
        index=["row-a", "row-b", "row-c"],
    )
    code_scores = pd.Series({0: 0.5, 1: 0.55, 2: 0.6})
    submitted_pmml = _FakeScorer({0: 0.5, 1: 0.55, 2: 0.6})

    result = run_reproducibility(
        sample=sample,
        config=_config(),
        code_scores=code_scores,
        submitted_pmml_scorer=submitted_pmml,
    )

    assert result.summary.status is ConsistencyStatus.PASS
    assert {row.row_index for row in result.rows} == {0, 1, 2}


def test_reproducibility_uses_positional_scores_for_filtered_integer_sample_index():
    sample = pd.DataFrame(
        {
            "x1": [0.1, 0.2, 0.3],
            "x2": [0.0, 0.0, 0.0],
            "sample_score": [0.5, 0.55, 0.6],
            "y": [0, 1, 0],
            "split": ["train"] * 3,
            "apply_month": ["202503"] * 3,
        },
        index=[10, 11, 12],
    )
    code_scores = sample["sample_score"].astype(float)
    submitted_pmml = _FakeScorer({0: 0.5, 1: 0.55, 2: 0.6})

    result = run_reproducibility(
        sample=sample,
        config=_config(),
        code_scores=code_scores,
        submitted_pmml_scorer=submitted_pmml,
    )

    assert result.summary.status is ConsistencyStatus.PASS
    assert {row.row_index for row in result.rows} == {0, 1, 2}


def test_reproducibility_uses_raw_dataframe_for_submitted_pmml():
    sample = pd.DataFrame({
        "x1": [0.1, 0.2],
        "x2": [99.0, 99.0],
        "sample_score": [0.1, 0.2],
        "y": [0, 1],
        "split": ["train", "train"],
        "apply_month": ["202503", "202503"],
    })
    code_scores = pd.Series({0: 0.1, 1: 0.2})
    seen_columns = []

    class RawFrameScorer:
        def score(self, df: pd.DataFrame) -> list[float]:
            seen_columns.extend(df.columns.tolist())
            return df["x1"].astype(float).tolist()

    result = run_reproducibility(
        sample=sample,
        config=_config(random_sample_size=2),
        code_scores=code_scores,
        submitted_pmml_scorer=RawFrameScorer(),
    )

    assert result.summary.status is ConsistencyStatus.PASS
    assert seen_columns == ["x1", "x2", "sample_score", "y", "split", "apply_month"]


def test_missing_code_score_for_sampled_row_fails_clearly():
    sample = pd.DataFrame({
        "x1": [0.1],
        "x2": [0.0],
        "sample_score": [0.1],
        "y": [0],
        "split": ["train"],
        "apply_month": ["202503"],
    })

    with pytest.raises(ValueError, match="missing code-model scores"):
        run_reproducibility(
            sample=sample,
            config=_config(random_sample_size=1),
            code_scores=pd.Series(dtype=float),
            submitted_pmml_scorer=_FakeScorer({0: 0.1}),
        )


def test_scores_match_at_precision_uses_rounding():
    assert scores_match_at_precision(0.1234564, 0.1234561, 6) is True
    assert scores_match_at_precision(0.1234564, 0.1234554, 6) is False
