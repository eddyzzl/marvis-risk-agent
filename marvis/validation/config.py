from dataclasses import dataclass, field

from marvis.validation.input_contracts import JsonScalar


@dataclass(frozen=True)
class ValidationConfig:
    target_col: str
    score_col: str
    split_col: str
    time_col: str
    feature_columns: list[str] = field(default_factory=list)
    bin_count: int = 10
    random_sample_size: int = 1000
    random_seed: int = 42
    score_decimal_places: int = 6
    split_values: dict[str, JsonScalar] = field(
        default_factory=lambda: {"train": "train", "test": "test", "oot": "oot"}
    )
    data_dict_feature_col: str = "特征名"
    data_dict_category_col: str = "类别"
