import pytest

from marvis.model_algorithms import (
    ALLOWED_ALGORITHMS,
    is_platform_default_training_description,
    model_training_description,
    model_training_report_text,
    normalize_algorithm,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("XGB", "xgb"),
        ("xgbm", "xgb"),
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


def test_model_training_report_text_includes_this_model_hyperparameters():
    text = model_training_report_text(
        "lgb",
        {"max_depth": 1, "learning_rate": 0.022631451052132007, "num_boost_round": 90},
    )
    assert "LightGBM" in text
    assert "max_depth=1" in text
    assert "learning_rate=0.022631" in text
    assert "num_boost_round=90" in text
    assert "直方图分裂、叶子优先生长和特征并行" not in text
    assert is_platform_default_training_description(text, "lgb") is False


def test_platform_default_training_description_detects_pending_and_generic_blurbs():
    assert is_platform_default_training_description(
        "待 Notebook 契约 RMC_ALGORITHM 确认后自动生成模型训练说明。"
    )
    assert is_platform_default_training_description(model_training_description("lgb"))
    assert is_platform_default_training_description(model_training_report_text("lgb"))
    assert is_platform_default_training_description(
        "本模型为浅树 LightGBM，max_depth=1。",
        "lgb",
    ) is False


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
