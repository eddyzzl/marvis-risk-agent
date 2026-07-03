from __future__ import annotations

import pandas as pd

from marvis.validation.in_memory_scores import _row_index_values
from marvis.validation.config import ValidationConfig
from marvis.validation.results import (
    ConsistencyStatus,
    ConsistencySummary,
    ReproducibilityResult,
    ScoreCompareRow,
)
from marvis.validation.scorer import Scorer

_REVIEW_MISMATCH_RATIO_THRESHOLD = 0.01
_REVIEW_MAX_ABS_DIFF_THRESHOLD = 1e-4


def scores_match_at_precision(left: float, right: float, decimals: int) -> bool:
    return round(float(left), decimals) == round(float(right), decimals)


def _nullable_score(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except TypeError:
        if pd.isna(value):
            return None
        raise
    if pd.isna(number):
        return None
    return number


def _code_scores_by_row_index(
    code_scores: pd.Series | pd.DataFrame,
    *,
    expected_len: int | None = None,
) -> pd.Series:
    if isinstance(code_scores, pd.DataFrame):
        if "row_index" not in code_scores.columns or "code_model_score" not in code_scores.columns:
            raise ValueError("code scores dataframe must contain row_index and code_model_score")
        return pd.Series(
            code_scores["code_model_score"].astype(float).to_numpy(),
            index=_row_index_values(code_scores["row_index"]),
        )
    series = code_scores.astype(float)
    if expected_len is not None and len(series) == int(expected_len):
        series = pd.Series(series.to_numpy(), index=range(len(series)))
    return series


def run_reproducibility(
    *,
    sample: pd.DataFrame,
    config: ValidationConfig,
    code_scores: pd.Series | pd.DataFrame,
    submitted_pmml_scorer: Scorer,
) -> ReproducibilityResult:
    original_len = len(sample)
    sample = sample.reset_index(drop=True)
    take = min(config.random_sample_size, len(sample))
    drawn = sample.sample(n=take, random_state=config.random_seed)

    code_scores_by_index = _code_scores_by_row_index(code_scores, expected_len=original_len)
    missing_indexes = [idx for idx in drawn.index if idx not in code_scores_by_index.index]
    if missing_indexes:
        preview = ", ".join(str(idx) for idx in missing_indexes[:10])
        raise ValueError(f"missing code-model scores for sampled rows: {preview}")

    scores_code = code_scores_by_index.loc[drawn.index].astype(float).tolist()
    scores_pmml = submitted_pmml_scorer.score(drawn.copy())
    if len(scores_pmml) != len(drawn):
        raise ValueError(
            f"submitted PMML scorer returned {len(scores_pmml)} scores for {len(drawn)} rows"
        )

    rows: list[ScoreCompareRow] = []
    match_count = 0
    mismatch_count = 0
    unknown_diff_mismatch_count = 0
    max_abs_diff = 0.0
    decimals = config.score_decimal_places
    for row_index, code_score, pmml_score in zip(drawn.index, scores_code, scores_pmml):
        code_score_float = float(code_score)
        pmml_score_float = _nullable_score(pmml_score)
        abs_diff = (
            None
            if pmml_score_float is None
            else abs(code_score_float - pmml_score_float)
        )
        matched = (
            False
            if pmml_score_float is None
            else scores_match_at_precision(code_score_float, pmml_score_float, decimals)
        )
        rows.append(
            ScoreCompareRow(
                row_index=row_index,
                score_code_model=code_score_float,
                score_submitted_pmml=pmml_score_float,
                abs_diff=abs_diff,
                matched=matched,
            )
        )
        if abs_diff is not None:
            max_abs_diff = max(max_abs_diff, abs_diff)
        if matched:
            match_count += 1
        else:
            mismatch_count += 1
            if abs_diff is None:
                unknown_diff_mismatch_count += 1

    status = _consistency_status(
        sample_size=take,
        mismatch_count=mismatch_count,
        max_abs_diff=max_abs_diff,
        unknown_diff_mismatch_count=unknown_diff_mismatch_count,
    )
    return ReproducibilityResult(
        sample_size=take,
        seed=config.random_seed,
        rows=rows,
        summary=ConsistencySummary(
            match_count=match_count,
            mismatch_count=mismatch_count,
            max_abs_diff=float(max_abs_diff),
            status=status,
        ),
    )


def _consistency_status(
    *,
    sample_size: int,
    mismatch_count: int,
    max_abs_diff: float,
    unknown_diff_mismatch_count: int,
) -> ConsistencyStatus:
    if mismatch_count == 0:
        return ConsistencyStatus.PASS
    if sample_size <= 0 or unknown_diff_mismatch_count > 0:
        return ConsistencyStatus.FAIL
    mismatch_ratio = mismatch_count / sample_size
    if (
        mismatch_ratio <= _REVIEW_MISMATCH_RATIO_THRESHOLD
        and max_abs_diff <= _REVIEW_MAX_ABS_DIFF_THRESHOLD
    ):
        return ConsistencyStatus.REVIEW
    return ConsistencyStatus.FAIL
