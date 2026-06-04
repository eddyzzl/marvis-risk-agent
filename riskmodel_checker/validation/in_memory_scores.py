from pathlib import Path

import pandas as pd


def _row_index_values(values: pd.Series) -> list[object]:
    row_indexes: list[object] = []
    for value in values.tolist():
        if pd.isna(value):
            raise ValueError("code-model score artifact contains missing row_index values")
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        row_indexes.append(value if isinstance(value, (int, str)) else str(value))
    return row_indexes


def load_code_model_scores(path: Path) -> pd.Series:
    scores = pd.read_csv(path)
    if "row_index" not in scores.columns or "code_model_score" not in scores.columns:
        raise ValueError("code-model score artifact must contain row_index and code_model_score")
    if scores["row_index"].duplicated().any():
        raise ValueError("code-model score artifact contains duplicate row_index values")
    return pd.Series(
        pd.to_numeric(scores["code_model_score"], errors="raise").to_numpy(dtype=float),
        index=_row_index_values(scores["row_index"]),
        name="code_model_score",
    )
