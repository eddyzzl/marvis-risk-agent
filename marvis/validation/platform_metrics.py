from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import pandas as pd

from marvis.domain import TaskRecord
from marvis.model_algorithms import normalize_algorithm
from marvis.notebook_contract import RuntimeContract
from marvis.output.excel import write_validation_excel
from marvis.validation.config import ValidationConfig
from marvis.validation.effectiveness import (
    build_effectiveness_result,
    compute_bin_tables,
    compute_monthly_ks,
    compute_monthly_psi,
    compute_overall_ks,
    compute_overall_psi,
    compute_psi_stability_table,
    compute_roc_ks_curves,
    prepare_effectiveness_context,
)
from marvis.validation.in_memory_scores import load_code_model_scores
from marvis.validation.pmml_scoring import load_pmml_scorer
from marvis.validation.reproducibility import run_reproducibility
from marvis.validation.results import ValidationResults
from marvis.validation.sample_stats import run_basic_info
from marvis.validation.stress_test import (
    STRESS_MISSING_VALUE,
    _filter_feature_categories,
    _model_features,
    load_feature_categories,
    run_stress_test,
)


REPRODUCIBILITY_RESULT_JSON = "reproducibility_result.json"


def load_runtime_sample(
    contract: RuntimeContract,
    *,
    fallback_sample_path: Path | None = None,
) -> pd.DataFrame:
    path = contract.sample_snapshot_path or fallback_sample_path
    if path is None:
        raise ValueError("runtime sample snapshot is missing from notebook contract")
    return _read_table(Path(path)).reset_index(drop=True)


def write_reproducibility_result(
    *,
    task: TaskRecord,
    contract: RuntimeContract,
    settings: Any,
    input_pmml_path: Path,
    output_path: Path,
    fallback_sample_path: Path | None = None,
) -> Path:
    sample = load_runtime_sample(contract, fallback_sample_path=fallback_sample_path)
    config = validation_config_from_contract(
        contract=contract,
        task=task,
        settings=settings,
    )
    result = run_reproducibility(
        sample=sample,
        config=config,
        code_scores=load_code_model_scores(contract.code_model_scores_path),
        submitted_pmml_scorer=load_pmml_scorer(
            input_pmml_path,
            positive_output_field=contract.pmml_output_field,
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def write_platform_validation_metrics(
    *,
    task: TaskRecord,
    contract: RuntimeContract,
    settings: Any,
    dictionary_path: Path,
    model_meta_path: Path,
    reproducibility_json_path: Path,
    results_json_path: Path,
    excel_path: Path,
    stress_scores_path: Path | None = None,
    fallback_sample_path: Path | None = None,
) -> Path:
    sample = load_runtime_sample(contract, fallback_sample_path=fallback_sample_path)
    config = validation_config_from_contract(
        contract=contract,
        task=task,
        settings=settings,
    )
    _validate_sample_columns(sample, config)
    dictionary = _load_dictionary(
        dictionary_path,
        feature_col=config.data_dict_feature_col,
        category_col=config.data_dict_category_col,
    )
    code_scores = load_code_model_scores(contract.code_model_scores_path)
    sample_scored = sample.copy()
    sample_scored[config.score_col] = _scores_for_frame_index(sample_scored, code_scores)

    basic_info = run_basic_info(
        sample=sample_scored,
        config=config,
        model_meta_path=model_meta_path,
    )
    effectiveness_context = prepare_effectiveness_context(
        sample=sample_scored,
        config=config,
    )
    effectiveness_overall = compute_overall_ks(
        sample=sample_scored,
        config=config,
    )
    effectiveness_overall = compute_overall_psi(
        sample=sample_scored,
        config=config,
        context=effectiveness_context,
        overall=effectiveness_overall,
    )
    monthly_ks = compute_monthly_ks(sample=sample_scored, config=config)
    monthly_psi = compute_monthly_psi(
        sample=sample_scored,
        config=config,
        context=effectiveness_context,
    )
    psi_stability_table = compute_psi_stability_table(
        sample=sample_scored,
        config=config,
    )
    bin_tables = compute_bin_tables(
        sample=sample_scored,
        config=config,
        context=effectiveness_context,
    )
    roc_ks_curves = compute_roc_ks_curves(sample=sample_scored, config=config)
    effectiveness = build_effectiveness_result(
        overall=effectiveness_overall,
        bin_tables=bin_tables,
        monthly_ks=monthly_ks,
        monthly_psi=monthly_psi,
        psi_stability_table=psi_stability_table,
        roc_ks_curves=roc_ks_curves,
    )

    feature_categories = load_feature_categories(
        dictionary,
        feature_col=config.data_dict_feature_col,
        category_col=config.data_dict_category_col,
    )
    feature_categories = _filter_feature_categories(
        feature_categories,
        model_features=_model_features(config, basic_info.feature_importance),
    )
    oot_sample = sample[sample[config.split_col] == config.split_values["oot"]]
    stress_scorer = PrecomputedStressScenarioScorer(
        code_scores=code_scores,
        stress_scores_path=stress_scores_path,
        feature_categories=feature_categories,
    )
    stress_test = run_stress_test(
        oot_sample=oot_sample,
        config=config,
        feature_categories=feature_categories,
        input_scorer=stress_scorer,
    )

    results = ValidationResults(
        model_name=task.model_name,
        model_version=task.model_version,
        algorithm=normalize_algorithm(contract.algorithm),
        target_type="binary",
        reproducibility=_reproducibility_from_json(reproducibility_json_path),
        basic_info=basic_info,
        effectiveness=effectiveness,
        stress_test=stress_test,
    )
    results_json_path.parent.mkdir(parents=True, exist_ok=True)
    results_json_path.write_text(
        json.dumps(asdict(results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_validation_excel(results, excel_path)
    return results_json_path


class PrecomputedStressScenarioScorer:
    def __init__(
        self,
        *,
        code_scores: pd.Series,
        stress_scores_path: Path | None,
        feature_categories: dict[str, list[str]],
    ) -> None:
        self._code_scores = code_scores
        self._feature_categories = feature_categories
        self._scenario_scores: dict[str, dict[Any, float]] = {}
        self._scenario_errors: dict[str, str] = {}
        if stress_scores_path and Path(stress_scores_path).exists():
            self._load_scenarios(Path(stress_scores_path))

    def score(self, dataframe: pd.DataFrame) -> list[float | None]:
        category = self._detect_category(dataframe)
        if category is None:
            return _scores_for_frame_index(dataframe, self._code_scores)
        if category in self._scenario_errors:
            raise RuntimeError(self._scenario_errors[category])
        if category not in self._scenario_scores:
            raise RuntimeError(f"missing precomputed stress scores for category: {category}")
        return [
            self._score_from_index(self._scenario_scores[category], index)
            for index in dataframe.index
        ]

    def _load_scenarios(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("categories") or []:
            category = str(row.get("category") or "")
            if not category:
                continue
            error = row.get("error")
            if error:
                self._scenario_errors[category] = str(error)
                continue
            row_indexes = row.get("row_index") or []
            scores = row.get("scores") or []
            self._scenario_scores[category] = {
                _normalize_row_index(index): float(score)
                for index, score in zip(row_indexes, scores, strict=False)
            }

    def _detect_category(self, dataframe: pd.DataFrame) -> str | None:
        for category, features in self._feature_categories.items():
            in_frame = [feature for feature in features if feature in dataframe.columns]
            if in_frame and all(
                (dataframe[feature] == STRESS_MISSING_VALUE).all()
                for feature in in_frame
            ):
                return category
        return None

    def _score_from_index(self, scores: dict[Any, float], index: Any) -> float:
        key = _normalize_row_index(index)
        if key in scores:
            return float(scores[key])
        text_key = str(key)
        if text_key in scores:
            return float(scores[text_key])
        raise RuntimeError(f"missing stress score for row index: {index}")


def validation_config_from_contract(
    *,
    contract: RuntimeContract,
    task: TaskRecord,
    settings: Any,
) -> ValidationConfig:
    return ValidationConfig(
        target_col=contract.target_col,
        score_col="__rmc_submitted_pmml_score__",
        split_col=contract.split_col or task.split_col or "",
        time_col=contract.time_col or task.time_col or "",
        feature_columns=[],
        bin_count=int(settings.bin_count),
        random_sample_size=int(settings.random_sample_size),
        random_seed=int(settings.random_seed),
        score_decimal_places=int(contract.score_decimal_places),
        data_dict_feature_col=settings.data_dict_feature_col,
        data_dict_category_col=settings.data_dict_category_col,
    )


def _scores_for_frame_index(dataframe: pd.DataFrame, scores: pd.Series) -> list[float]:
    output: list[float] = []
    for index in dataframe.index:
        key = _normalize_row_index(index)
        if key in scores.index:
            output.append(float(scores.loc[key]))
            continue
        text_key = str(key)
        if text_key in scores.index:
            output.append(float(scores.loc[text_key]))
            continue
        raise ValueError(f"missing code-model score for row index: {index}")
    return output


def _validate_sample_columns(sample: pd.DataFrame, config: ValidationConfig) -> None:
    missing = [
        f"{label}='{column}'"
        for label, column in {
            "target_col": config.target_col,
            "split_col": config.split_col,
            "time_col": config.time_col,
        }.items()
        if column and column not in sample.columns
    ]
    if missing:
        raise ValueError("sample column check failed: " + ", ".join(missing))


def _load_dictionary(
    path: Path,
    *,
    feature_col: str,
    category_col: str,
) -> pd.DataFrame:
    dictionary = _read_table(path)
    missing = [
        col for col in (feature_col, category_col) if col not in dictionary.columns
    ]
    if missing:
        raise ValueError("data dictionary missing columns: " + ", ".join(sorted(missing)))
    return dictionary


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"unsupported table format: {suffix}")


def _reproducibility_from_json(path: Path):
    from marvis.validation.results import validation_results_from_dict

    payload = json.loads(path.read_text(encoding="utf-8"))
    wrapper = {
        "reproducibility": payload,
        "basic_info": {},
        "effectiveness": {},
        "stress_test": {},
    }
    return validation_results_from_dict(wrapper).reproducibility


def _normalize_row_index(value: Any) -> int | str:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value if isinstance(value, (int, str)) else str(value)
