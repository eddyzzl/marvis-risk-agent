from pathlib import Path

import pandas as pd
import pytest

from marvis.validation.in_memory_scores import load_code_model_scores


def test_load_code_model_scores_returns_series_indexed_by_row_index(tmp_path: Path):
    path = tmp_path / "scores.csv"
    pd.DataFrame(
        {"row_index": [3, 5], "code_model_score": [0.1, 0.2]}
    ).to_csv(path, index=False)

    scores = load_code_model_scores(path)

    assert scores.to_dict() == {3: 0.1, 5: 0.2}


def test_load_code_model_scores_rejects_duplicate_row_index(tmp_path: Path):
    path = tmp_path / "scores.csv"
    pd.DataFrame(
        {"row_index": [3, 3], "code_model_score": [0.1, 0.2]}
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicate row_index"):
        load_code_model_scores(path)


def test_load_code_model_scores_keeps_string_row_index(tmp_path: Path):
    path = tmp_path / "scores.csv"
    pd.DataFrame(
        {"row_index": ["row-a", "row-b"], "code_model_score": [0.1, 0.2]}
    ).to_csv(path, index=False)

    scores = load_code_model_scores(path)

    assert scores.index.tolist() == ["row-a", "row-b"]
