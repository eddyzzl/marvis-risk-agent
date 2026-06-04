from riskmodel_checker.validation.config import ValidationConfig


def test_validation_config_requires_core_columns():
    config = ValidationConfig(
        target_col="y",
        score_col="score",
        split_col="split",
        time_col="apply_month",
        feature_columns=["x1", "x2"],
    )
    assert config.target_col == "y"
    assert config.bin_count == 10
    assert config.random_sample_size == 1000
    assert config.random_seed == 42
    assert config.score_decimal_places == 6
    assert config.split_values == {"train": "train", "test": "test", "oot": "oot"}


def test_validation_config_is_frozen():
    config = ValidationConfig(
        target_col="y",
        score_col="score",
        split_col="split",
        time_col="apply_month",
        feature_columns=["x1"],
    )
    import dataclasses
    with __import__("pytest").raises(dataclasses.FrozenInstanceError):
        config.target_col = "other"
