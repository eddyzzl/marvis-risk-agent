from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from marvis.artifacts.model_score_vector import (
    MAX_MODEL_SCORE_VECTOR_ROWS,
    MODEL_SCORE_VECTOR_SCHEMA,
    ModelScoreVectorError,
    validate_model_score_vector,
    write_model_score_vector,
)


def test_model_score_vector_round_trips_exact_float64_rows(tmp_path: Path) -> None:
    first_path = tmp_path / "first.parquet"
    second_path = tmp_path / "second.parquet"
    scores = np.array([0.0, 0.125, 0.5, 1.0], dtype=np.float64)

    first = write_model_score_vector(first_path, scores)
    second = write_model_score_vector(second_path, scores)

    assert first.content_hash == second.content_hash
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first.row_count == 4
    assert first.row_ordinals.tolist() == [0, 1, 2, 3]
    assert first.row_ordinals.flags.writeable is False
    assert first.scores.flags.writeable is False
    assert first.scores.dtype == np.dtype("float64")
    assert first.scores.tolist() == scores.tolist()
    assert (
        validate_model_score_vector(
            first_path,
            expected_content_hash=first.content_hash,
            expected_row_count=4,
        )
        == first
    )


@pytest.mark.parametrize(
    "scores",
    [
        [],
        [0.1, np.nan],
        [0.1, np.inf],
        [-0.01, 0.5],
        [0.5, 1.01],
    ],
)
def test_model_score_vector_rejects_invalid_probability_vectors(
    tmp_path: Path,
    scores: list[float],
) -> None:
    with pytest.raises(ModelScoreVectorError):
        write_model_score_vector(tmp_path / "scores.parquet", scores)


def test_model_score_vector_rejects_noncanonical_schema_and_ordinals(
    tmp_path: Path,
) -> None:
    wrong_schema = tmp_path / "wrong-schema.parquet"
    pq.write_table(
        pa.table(
            {
                "row_ordinal": pa.array([0, 1], type=pa.int32()),
                "score": pa.array([0.1, 0.2], type=pa.float64()),
            }
        ),
        wrong_schema,
    )
    with pytest.raises(ModelScoreVectorError, match="schema"):
        validate_model_score_vector(wrong_schema)

    wrong_ordinals = tmp_path / "wrong-ordinals.parquet"
    pq.write_table(
        pa.Table.from_arrays(
            [
                pa.array([0, 2], type=pa.int64()),
                pa.array([0.1, 0.2], type=pa.float64()),
            ],
            schema=MODEL_SCORE_VECTOR_SCHEMA,
        ),
        wrong_ordinals,
    )
    with pytest.raises(ModelScoreVectorError, match="row_ordinal"):
        validate_model_score_vector(wrong_ordinals)


def test_model_score_vector_enforces_row_budget_before_write(tmp_path: Path) -> None:
    oversized = np.zeros(MAX_MODEL_SCORE_VECTOR_ROWS + 1, dtype=np.float64)

    with pytest.raises(ModelScoreVectorError, match="row budget"):
        write_model_score_vector(tmp_path / "oversized.parquet", oversized)

    assert not (tmp_path / "oversized.parquet").exists()
