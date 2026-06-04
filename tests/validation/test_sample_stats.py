import json
from pathlib import Path

import pandas as pd
import pytest

from riskmodel_checker.validation.config import ValidationConfig
from riskmodel_checker.validation.sample_stats import run_basic_info


def _config() -> ValidationConfig:
    return ValidationConfig(
        target_col="y",
        score_col="sample_score",
        split_col="split",
        time_col="apply_month",
        feature_columns=["x1"],
    )


def test_basic_info_split_and_monthly(tmp_path: Path):
    sample = pd.DataFrame({
        "x1": [0.1] * 6,
        "sample_score": [0.5] * 6,
        "y": [0, 1, 0, 1, 0, 0],
        "split": ["train", "train", "test", "test", "oot", "oot"],
        "apply_month": ["202503", "202503", "202504", "202504", "202505", "202505"],
    })
    meta_path = tmp_path / "model_meta.json"
    meta_path.write_text(json.dumps({
        "feature_importance": [{"feature": "x1", "category": "征信", "importance": 0.8}],
        "hyperparameters": {"learning_rate": 0.05, "max_depth": 5},
    }), encoding="utf-8")

    info = run_basic_info(sample=sample, config=_config(), model_meta_path=meta_path)

    assert info.sample_period == ("20250301", "20250501")
    by_split = {row.split: row for row in info.split_summary}
    assert by_split["train"].sample_count == 2
    assert by_split["train"].bad_count == 1
    assert by_split["train"].bad_rate == pytest.approx(0.5)
    assert by_split["train"].period_start == "20250301"
    assert by_split["train"].period_end == "20250301"
    assert by_split["test"].period_start == "20250401"
    assert by_split["test"].period_end == "20250401"
    assert by_split["oot"].period_start == "20250501"
    assert by_split["oot"].period_end == "20250501"
    assert by_split["oot"].bad_count == 0
    months = {row.month: row for row in info.monthly_distribution}
    assert months["202503"].sample_count == 2
    assert months["202504"].bad_rate == pytest.approx(0.5)
    assert info.hyperparameters == {"learning_rate": 0.05, "max_depth": 5}
    assert info.feature_importance[0].rank == 1
    assert info.feature_importance[0].feature == "x1"
    assert info.feature_importance[0].category == "征信"


def test_basic_info_derives_dates_and_months_from_mixed_time_formats(tmp_path: Path):
    sample = pd.DataFrame({
        "x1": [0.1] * 6,
        "sample_score": [0.5] * 6,
        "y": [0, 1, 0, 1, 0, 0],
        "split": ["train", "train", "test", "test", "oot", "oot"],
        "apply_month": [
            "2025/03/01",
            "20250315",
            "2025-04-01",
            "2025-04-30 13:20:00",
            pd.Timestamp("2025-05-01 08:00:00"),
            20250502,
        ],
    })
    meta_path = tmp_path / "model_meta.json"
    meta_path.write_text(json.dumps({
        "feature_importance": [{"feature": "x1", "importance": 0.8}],
        "hyperparameters": {},
    }), encoding="utf-8")

    info = run_basic_info(sample=sample, config=_config(), model_meta_path=meta_path)

    assert info.sample_period == ("20250301", "20250502")
    by_split = {row.split: row for row in info.split_summary}
    assert by_split["train"].period_start == "20250301"
    assert by_split["train"].period_end == "20250315"
    assert by_split["test"].period_start == "20250401"
    assert by_split["test"].period_end == "20250430"
    assert by_split["oot"].period_start == "20250501"
    assert by_split["oot"].period_end == "20250502"
    months = {row.month: row for row in info.monthly_distribution}
    assert set(months) == {"202503", "202504", "202505"}
    assert months["202503"].sample_count == 2
    assert months["202504"].bad_rate == pytest.approx(0.5)


def test_basic_info_counts_string_binary_targets_as_numeric(tmp_path: Path):
    sample = pd.DataFrame({
        "x1": [0.1] * 6,
        "sample_score": [0.5] * 6,
        "y": ["0", "1", "0", "1", "0", "0"],
        "split": ["train", "train", "test", "test", "oot", "oot"],
        "apply_month": ["202503", "202503", "202504", "202504", "202505", "202505"],
    })
    meta_path = tmp_path / "model_meta.json"
    meta_path.write_text(
        json.dumps({"feature_importance": [], "hyperparameters": {}}),
        encoding="utf-8",
    )

    info = run_basic_info(sample=sample, config=_config(), model_meta_path=meta_path)

    by_split = {row.split: row for row in info.split_summary}
    assert by_split["train"].bad_count == 1
    assert by_split["train"].bad_rate == pytest.approx(0.5)
    assert {row.month: row.bad_count for row in info.monthly_distribution} == {
        "202503": 1,
        "202504": 1,
        "202505": 0,
    }


def test_basic_info_rejects_missing_required_split(tmp_path: Path):
    sample = pd.DataFrame({
        "x1": [0.1] * 4,
        "sample_score": [0.5] * 4,
        "y": [0, 1, 0, 1],
        "split": ["train", "train", "oot", "oot"],
        "apply_month": ["202503", "202503", "202505", "202505"],
    })
    meta_path = tmp_path / "model_meta.json"
    meta_path.write_text(
        json.dumps({"feature_importance": [], "hyperparameters": {}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="test"):
        run_basic_info(sample=sample, config=_config(), model_meta_path=meta_path)
