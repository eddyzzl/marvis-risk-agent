from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from riskmodel_checker.validation.config import ValidationConfig
from riskmodel_checker.validation.checks import finite_score_series, validate_binary_target
from riskmodel_checker.validation.effectiveness import run_effectiveness
from riskmodel_checker.validation.reproducibility import run_reproducibility
from riskmodel_checker.validation.results import ValidationResults
from riskmodel_checker.validation.sample_stats import run_basic_info
from riskmodel_checker.validation.scorer import Scorer
from riskmodel_checker.validation.stress_test import load_feature_categories, run_stress_test


@dataclass(frozen=True)
class EngineInputs:
    model_name: str
    model_version: str
    algorithm: str
    sample: pd.DataFrame
    data_dictionary: pd.DataFrame
    model_meta_path: Path
    code_scores: pd.Series | None = None
    input_scorer: Scorer | None = None


def run_validation(*, inputs: EngineInputs, config: ValidationConfig) -> ValidationResults:
    if inputs.input_scorer is None:
        raise ValueError("input_scorer is required")
    validate_binary_target(inputs.sample, config.target_col)
    code_scores = inputs.code_scores
    if code_scores is None:
        if config.score_col not in inputs.sample.columns:
            raise ValueError("code_scores is required when sample score_col is absent")
        code_scores = inputs.sample[config.score_col].astype(float)

    sample = inputs.sample.copy()
    pmml_scores = inputs.input_scorer.score(sample.copy())
    if len(pmml_scores) != len(sample):
        raise ValueError(
            f"submitted PMML scorer returned {len(pmml_scores)} scores for {len(sample)} rows"
        )
    sample[config.score_col] = finite_score_series(
        pmml_scores,
        index=sample.index,
        label="submitted PMML scorer",
    )

    reproducibility = run_reproducibility(
        sample=inputs.sample,
        config=config,
        code_scores=code_scores,
        submitted_pmml_scorer=inputs.input_scorer,
    )
    basic_info = run_basic_info(
        sample=sample,
        config=config,
        model_meta_path=inputs.model_meta_path,
    )
    effectiveness = run_effectiveness(sample=sample, config=config)

    oot_sample = inputs.sample[inputs.sample[config.split_col] == config.split_values["oot"]]
    feature_categories = load_feature_categories(
        inputs.data_dictionary,
        feature_col=config.data_dict_feature_col,
        category_col=config.data_dict_category_col,
    )
    feature_categories = _filter_feature_categories(
        feature_categories,
        model_features=_model_features(config, basic_info.feature_importance),
    )
    stress_test = run_stress_test(
        oot_sample=oot_sample,
        config=config,
        feature_categories=feature_categories,
        input_scorer=inputs.input_scorer,
    )

    return ValidationResults(
        model_name=inputs.model_name,
        model_version=inputs.model_version,
        algorithm=inputs.algorithm,
        target_type="binary",
        reproducibility=reproducibility,
        basic_info=basic_info,
        effectiveness=effectiveness,
        stress_test=stress_test,
    )


def _model_features(config: ValidationConfig, feature_importance: list) -> list[str]:
    if config.feature_columns:
        return list(config.feature_columns)
    return [str(row.feature) for row in feature_importance]


def _filter_feature_categories(
    feature_categories: dict[str, list[str]],
    *,
    model_features: list[str],
) -> dict[str, list[str]]:
    if not model_features:
        return feature_categories
    allowed = set(model_features)
    filtered: dict[str, list[str]] = {}
    for category, features in feature_categories.items():
        in_model = [feature for feature in features if feature in allowed]
        if in_model:
            filtered[category] = in_model
    return filtered
