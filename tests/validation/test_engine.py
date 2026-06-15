import json
from pathlib import Path

import pandas as pd
import pytest

from marvis.validation.config import ValidationConfig
from marvis.validation.engine import EngineInputs, run_validation
from marvis.validation.results import ConsistencyStatus, ValidationResults


class _PassThroughScorer:
    def score(self, df: pd.DataFrame) -> list[float]:
        return df["x1"].astype(float).tolist()


class _NullScorer:
    def score(self, df: pd.DataFrame) -> list[float | None]:
        scores = df["x1"].astype(float).tolist()
        scores[1] = None
        return scores


def _config() -> ValidationConfig:
    return ValidationConfig(
        target_col="y",
        score_col="sample_score",
        split_col="split",
        time_col="apply_month",
        feature_columns=["x1"],
        bin_count=5,
        random_sample_size=8,
    )


def _make_sample() -> pd.DataFrame:
    rows = []
    for split in ("train", "test", "oot"):
        for i in range(10):
            score = (i + 1) / 11
            rows.append({
                "x1": score,
                "sample_score": score,
                "y": int(i >= 5),
                "split": split,
                "apply_month": "202503" if split == "train" else "202505",
            })
    return pd.DataFrame(rows)


def test_engine_returns_validation_results(tmp_path: Path):
    sample = _make_sample()
    dictionary = pd.DataFrame({"特征名": ["x1"], "类别": ["征信"]})
    meta_path = tmp_path / "model_meta.json"
    meta_path.write_text(json.dumps({
        "feature_importance": [{"feature": "x1", "importance": 1.0}],
        "hyperparameters": {"max_depth": 4},
    }), encoding="utf-8")

    inputs = EngineInputs(
        model_name="A卡",
        model_version="v1",
        algorithm="lgb",
        sample=sample,
        data_dictionary=dictionary,
        model_meta_path=meta_path,
        input_scorer=_PassThroughScorer(),
    )
    results = run_validation(inputs=inputs, config=_config())

    assert isinstance(results, ValidationResults)
    assert results.reproducibility.summary.status is ConsistencyStatus.PASS
    assert {row.split for row in results.basic_info.split_summary} == {"train", "test", "oot"}
    assert len(results.effectiveness.overall) == 3
    assert len(results.stress_test.per_category) == 1
    assert results.stress_test.per_category[0].category == "征信"


def test_engine_filters_stress_categories_to_model_features_and_uses_full_oot(tmp_path: Path):
    sample = _make_sample()
    dictionary = pd.DataFrame(
        {"特征名": ["x1", "x_unused"], "类别": ["征信", "交易"]}
    )
    meta_path = tmp_path / "model_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "feature_importance": [{"feature": "x1", "importance": 1.0}],
                "hyperparameters": {"max_depth": 4},
            }
        ),
        encoding="utf-8",
    )

    results = run_validation(
        inputs=EngineInputs(
            model_name="A卡",
            model_version="v1",
            algorithm="lgb",
            sample=sample,
            data_dictionary=dictionary,
            model_meta_path=meta_path,
            input_scorer=_PassThroughScorer(),
        ),
        config=_config(),
    )

    assert results.stress_test.baseline.sample_count == 10
    assert [row.category for row in results.stress_test.per_category] == ["征信"]


def test_engine_rejects_null_pmml_scores_before_metric_artifacts(tmp_path: Path):
    sample = _make_sample()
    dictionary = pd.DataFrame({"特征名": ["x1"], "类别": ["征信"]})
    meta_path = tmp_path / "model_meta.json"
    meta_path.write_text(
        json.dumps({"feature_importance": [{"feature": "x1", "importance": 1.0}]}),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"submitted PMML scorer returned non-finite scores at rows: \[1\]",
    ):
        run_validation(
            inputs=EngineInputs(
                model_name="A卡",
                model_version="v1",
                algorithm="lgb",
                sample=sample,
                data_dictionary=dictionary,
                model_meta_path=meta_path,
                input_scorer=_NullScorer(),
            ),
            config=_config(),
        )


def test_engine_rejects_non_binary_target_values(tmp_path: Path):
    sample = _make_sample()
    sample.loc[sample.index[2], "y"] = 2
    dictionary = pd.DataFrame({"特征名": ["x1"], "类别": ["征信"]})
    meta_path = tmp_path / "model_meta.json"
    meta_path.write_text(
        json.dumps({"feature_importance": [{"feature": "x1", "importance": 1.0}]}),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"binary target 'y' must contain only 0/1 values; found: 2",
    ):
        run_validation(
            inputs=EngineInputs(
                model_name="A卡",
                model_version="v1",
                algorithm="lgb",
                sample=sample,
                data_dictionary=dictionary,
                model_meta_path=meta_path,
                input_scorer=_PassThroughScorer(),
            ),
            config=_config(),
        )
