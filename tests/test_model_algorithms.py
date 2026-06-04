import pytest

from riskmodel_checker.model_algorithms import (
    ALLOWED_ALGORITHMS,
    model_training_description,
    normalize_algorithm,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("XGB", "xgb"),
        ("XGBClassifier", "xgb"),
        ("xgboost.XGBClassifier", "xgb"),
        ("XGBoost", "xgb"),
        ("lgbm", "lgb"),
        ("LGBMClassifier", "lgb"),
        ("LightGBM", "lgb"),
        ("Lightgbm", "lgb"),
        ("LR", "lr"),
        ("逻辑回归", "lr"),
        ("LogisticRegression", "lr"),
        ("CatBoostClassifier", "catboost"),
        ("评分卡", "scorecard"),
        ("score_card", "scorecard"),
        ("deep neural network", "dnn"),
        ("神经网络", "dnn"),
    ],
)
def test_normalize_algorithm_accepts_display_names_and_case_variants(raw, expected):
    assert normalize_algorithm(raw) == expected


def test_model_training_description_uses_normalized_algorithm():
    assert "XGBoost" in model_training_description("XGBoost")


def test_model_training_descriptions_are_substantive():
    for algorithm in ALLOWED_ALGORITHMS:
        description = model_training_description(algorithm)
        assert len(description) >= 80
        assert "待补充" not in description


@pytest.mark.parametrize("raw", ["unknown", "random forest", "svm"])
def test_normalize_algorithm_rejects_phase_two_or_unknown_algorithms(raw):
    with pytest.raises(ValueError, match="unsupported model algorithm"):
        normalize_algorithm(raw)


@pytest.mark.parametrize("raw", ["", "   ", None])
def test_normalize_algorithm_rejects_blank_values_by_default(raw):
    with pytest.raises(ValueError, match="model algorithm is required"):
        normalize_algorithm(raw)


def test_normalize_algorithm_can_preserve_empty_create_task_placeholder():
    assert normalize_algorithm("", allow_empty=True) == ""
