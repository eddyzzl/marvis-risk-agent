from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict
import json
from pathlib import Path
import stat
from typing import Any

import numpy as np
import pandas as pd

from marvis.domain import TaskRecord
from marvis.model_algorithms import normalize_algorithm
from marvis.notebook_contract import RuntimeContract
from marvis.output.excel import write_validation_excel
from marvis.validation.config import ValidationConfig
from marvis.validation.checks import validate_required_splits
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
from marvis.validation.field_transformations import (
    apply_confirmed_transformations,
    required_transformation_inputs,
    topologically_sorted_transformations,
)
from marvis.validation.feature_categories import (
    FeatureCategoryConflict,
    FeatureCategoryResolution,
    resolve_feature_categories,
)
from marvis.validation.in_memory_scores import load_code_model_scores
from marvis.validation.input_confirmation import (
    json_scalar_identity,
    normalize_binary_target,
)
from marvis.validation.input_contracts import (
    FeatureMetadataResolution,
    TransformationSpec,
    ValidationInputContract,
)
from marvis.validation.pmml_score_artifacts import (
    sha256_file_cancellable,
    validate_pmml_score_artifact,
)
from marvis.validation.pmml_scoring import load_pmml_scorer
from marvis.validation.reproducibility import run_reproducibility
from marvis.validation.results import (
    EffectivenessResult,
    FeatureImportanceRow,
    PmmlScoringResult,
    StressTestResult,
    ValidationResults,
    reproducibility_result_from_dict,
    validation_results_to_dict,
)
from marvis.validation.sample_chunks import read_selected_columns
from marvis.validation.sample_stats import run_basic_info, run_basic_info_from_metadata
from marvis.validation.stress_test import STRESS_MISSING_VALUE, run_stress_test


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

    category_resolution = stress_category_resolution_for_metrics(
        feature_importance=basic_info.feature_importance,
        fallback_model_features=[str(column) for column in sample.columns],
        dictionary=dictionary,
        feature_col=config.data_dict_feature_col,
        category_col=config.data_dict_category_col,
        stress_scores_path=stress_scores_path,
    )
    feature_categories = category_resolution.per_category
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
        unclassified_features=category_resolution.unclassified_features,
        category_source_counts=category_resolution.source_counts,
    )

    results = ValidationResults(
        model_name=task.model_name,
        model_version=task.model_version,
        algorithm=normalize_algorithm(contract.algorithm),
        target_type="binary",
        schema_version="marvis.validation_results.v1",
        reproducibility=_reproducibility_from_json(reproducibility_json_path),
        basic_info=basic_info,
        effectiveness=effectiveness,
        stress_test=stress_test,
    )
    results_json_path.parent.mkdir(parents=True, exist_ok=True)
    results_json_path.write_text(
        json.dumps(validation_results_to_dict(results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_validation_excel(results, excel_path)
    return results_json_path


def load_pmml_analysis_frame(
    *,
    sample_path: Path,
    score_path: Path,
    contract: ValidationInputContract,
    scoring_result: PmmlScoringResult,
    cancellation_check: Callable[[], None] | None = None,
) -> pd.DataFrame:
    """Load only control columns plus the verified PMML score sidecar.

    The resulting frame is intentionally narrow. It is still O(rows) because the
    existing deterministic metrics require all labels, splits, months and scores in
    memory, but it never materializes unrelated model features or a sample-provided
    prediction column.
    """

    _check_cancelled(cancellation_check)
    sample_identity = _file_identity(Path(sample_path), label="validation sample")
    score_identity = _file_identity(Path(score_path), label="PMML score sidecar")
    schema = contract.require_sample_schema()
    _validate_pmml_scoring_identity(
        contract=contract,
        scoring_result=scoring_result,
        sample_path=sample_path,
        score_path=score_path,
        cancellation_check=cancellation_check,
    )

    target_col = _confirmed_field(contract, "target_col")
    split_col = _confirmed_field(contract, "split_col")
    time_col = _confirmed_field(contract, "time_col")
    transformations = _transformation_closure(
        (target_col, split_col, time_col),
        contract.transformations,
    )
    projection = tuple(
        dict.fromkeys(
            required_transformation_inputs(
                (target_col, split_col, time_col),
                transformations,
            )
        )
    )
    sample = read_selected_columns(
        Path(sample_path),
        columns=projection,
        schema=schema,
        cancellation_check=cancellation_check,
    )
    _check_cancelled(cancellation_check)
    sample = apply_confirmed_transformations(sample, transformations)
    _check_cancelled(cancellation_check)

    scores = pd.read_parquet(
        Path(score_path),
        columns=["row_id", "pmml_score"],
    )
    _check_cancelled(cancellation_check)
    expected_row_ids = np.arange(len(sample), dtype=np.int64)
    if not np.array_equal(
        scores["row_id"].to_numpy(dtype=np.int64),
        expected_row_ids,
    ):
        raise ValueError("PMML score sidecar row_id does not match validation sample")
    if len(scores) != scoring_result.input_row_count:
        raise ValueError("PMML score sidecar row count does not match scoring result")
    score_values = scores["pmml_score"].to_numpy(dtype=float)
    if not np.isfinite(score_values).all():
        raise ValueError("PMML score sidecar contains a non-finite score")

    confirmed = contract.confirmed
    if "positive_label" not in confirmed or "negative_label" not in confirmed:
        raise ValueError("validation input contract has no confirmed binary labels")
    target = normalize_binary_target(
        sample[target_col],
        positive=confirmed["positive_label"],
        negative=confirmed["negative_label"],
    )
    split = _canonical_split_series(
        sample[split_col],
        confirmed.get("split_value_mapping"),
    )
    frame = pd.DataFrame(
        {
            "__target__": target.to_numpy(dtype=np.int8),
            "__split__": split.to_numpy(dtype=object),
            "__time__": sample[time_col].to_numpy(copy=True),
            "__pmml_score__": score_values,
        }
    )

    # The strict boundary already hashed both files. Fingerprints close the
    # subsequent read window without hashing a million-row sample a second time.
    _require_file_identity(
        Path(sample_path), sample_identity, label="validation sample"
    )
    _require_file_identity(
        Path(score_path), score_identity, label="PMML score sidecar"
    )
    _check_cancelled(cancellation_check)
    return frame


def validation_config_from_input_contract(
    contract: ValidationInputContract,
    settings: Any,
) -> ValidationConfig:
    mapping = contract.confirmed.get("split_value_mapping")
    _validated_split_identity_mapping(mapping)
    return ValidationConfig(
        target_col="__target__",
        score_col="__pmml_score__",
        split_col="__split__",
        time_col="__time__",
        feature_columns=[],
        bin_count=int(settings.bin_count),
        random_sample_size=int(settings.random_sample_size),
        random_seed=int(settings.random_seed),
        score_decimal_places=6,
        split_values={"train": "train", "test": "test", "oot": "oot"},
    )


def compute_existing_effectiveness(
    sample_scored: pd.DataFrame,
    config: ValidationConfig,
    *,
    cancellation_check: Callable[[], None] | None = None,
) -> EffectivenessResult:
    """Run the unchanged deterministic effectiveness sections with checkpoints."""

    _check_cancelled(cancellation_check)
    validate_required_splits(
        sample_scored,
        split_col=config.split_col,
        split_values=config.split_values,
    )
    context = prepare_effectiveness_context(
        sample=sample_scored,
        config=config,
        cancellation_check=cancellation_check,
    )
    _check_cancelled(cancellation_check)
    overall = compute_overall_ks(
        sample=sample_scored,
        config=config,
        cancellation_check=cancellation_check,
    )
    _check_cancelled(cancellation_check)
    overall = compute_overall_psi(
        sample=sample_scored,
        config=config,
        context=context,
        overall=overall,
        cancellation_check=cancellation_check,
    )
    _check_cancelled(cancellation_check)
    bin_tables = compute_bin_tables(
        sample=sample_scored,
        config=config,
        context=context,
        cancellation_check=cancellation_check,
    )
    _check_cancelled(cancellation_check)
    monthly_ks = compute_monthly_ks(
        sample=sample_scored,
        config=config,
        cancellation_check=cancellation_check,
    )
    _check_cancelled(cancellation_check)
    monthly_psi = compute_monthly_psi(
        sample=sample_scored,
        config=config,
        context=context,
        cancellation_check=cancellation_check,
    )
    _check_cancelled(cancellation_check)
    psi_table = compute_psi_stability_table(
        sample=sample_scored,
        config=config,
        cancellation_check=cancellation_check,
    )
    _check_cancelled(cancellation_check)
    curves = compute_roc_ks_curves(
        sample=sample_scored,
        config=config,
        cancellation_check=cancellation_check,
    )
    _check_cancelled(cancellation_check)
    return build_effectiveness_result(
        overall=overall,
        bin_tables=bin_tables,
        monthly_ks=monthly_ks,
        monthly_psi=monthly_psi,
        psi_stability_table=psi_table,
        roc_ks_curves=curves,
    )


def compute_platform_validation_results(
    *,
    task: TaskRecord,
    contract: ValidationInputContract,
    sample_path: Path,
    score_path: Path,
    scoring_result: PmmlScoringResult,
    metadata_resolution: FeatureMetadataResolution,
    stress_test: StressTestResult,
    settings: Any,
    cancellation_check: Callable[[], None] | None = None,
) -> ValidationResults:
    if contract.status != "ready":
        raise ValueError("validation input contract is not ready for metrics")
    if metadata_resolution != contract.require_feature_metadata():
        raise ValueError("feature metadata does not match validation input contract")
    _check_cancelled(cancellation_check)
    config = validation_config_from_input_contract(contract, settings)
    sample_scored = load_pmml_analysis_frame(
        sample_path=sample_path,
        score_path=score_path,
        contract=contract,
        scoring_result=scoring_result,
        cancellation_check=cancellation_check,
    )
    basic_info = run_basic_info_from_metadata(
        sample=sample_scored,
        config=config,
        model_params=contract.require_model_params(),
        feature_metadata=metadata_resolution.rows,
        cancellation_check=cancellation_check,
    )
    effectiveness = compute_existing_effectiveness(
        sample_scored,
        config,
        cancellation_check=cancellation_check,
    )
    _check_cancelled(cancellation_check)
    return ValidationResults(
        model_name=task.model_name,
        model_version=task.model_version,
        algorithm=normalize_algorithm(contract.require_algorithm()),
        target_type="binary",
        schema_version="marvis.validation_results.v2",
        pmml_scoring=scoring_result,
        basic_info=basic_info,
        effectiveness=effectiveness,
        stress_test=stress_test,
    )


def _validate_pmml_scoring_identity(
    *,
    contract: ValidationInputContract,
    scoring_result: PmmlScoringResult,
    sample_path: Path,
    score_path: Path,
    cancellation_check: Callable[[], None] | None,
) -> None:
    expected_pmml = contract.material_hashes.get("pmml")
    if not expected_pmml or scoring_result.pmml_sha256 != expected_pmml:
        raise ValueError("PMML scoring result does not match confirmed PMML")
    if scoring_result.output_field != contract.require_output_field():
        raise ValueError("PMML scoring result output does not match input contract")
    schema = contract.require_sample_schema()
    if (
        schema.row_count is not None
        and scoring_result.input_row_count != schema.row_count
    ):
        raise ValueError("PMML scoring result row count does not match sample schema")
    _require_current_sample_hash(
        sample_path=sample_path,
        contract=contract,
        scoring_result=scoring_result,
        cancellation_check=cancellation_check,
    )
    validate_pmml_score_artifact(
        scoring_result,
        Path(score_path),
        cancellation_check=cancellation_check,
    )


def _require_current_sample_hash(
    *,
    sample_path: Path,
    contract: ValidationInputContract,
    scoring_result: PmmlScoringResult,
    cancellation_check: Callable[[], None] | None,
) -> None:
    expected = contract.material_hashes.get("sample")
    if not expected or scoring_result.sample_sha256 != expected:
        raise ValueError("PMML scoring result does not match confirmed sample")
    current = sha256_file_cancellable(
        Path(sample_path),
        cancellation_check=cancellation_check,
    )
    if current != expected or current != contract.require_sample_schema().sha256:
        raise ValueError("current validation sample does not match confirmed SHA-256")


def _confirmed_field(contract: ValidationInputContract, key: str) -> str:
    value = contract.confirmed.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"validation input contract has no confirmed {key}")
    return value


def _transformation_closure(
    output_fields: Sequence[str],
    specs: Sequence[TransformationSpec],
) -> tuple[TransformationSpec, ...]:
    ordered = topologically_sorted_transformations(specs)
    by_output = {spec.output_field: spec for spec in ordered}
    needed: set[str] = set()
    stack = list(reversed(tuple(output_fields)))
    while stack:
        field = stack.pop()
        spec = by_output.get(field)
        if spec is None or field in needed:
            continue
        needed.add(field)
        stack.extend(reversed(spec.input_fields))
    return tuple(spec for spec in ordered if spec.output_field in needed)


def _validated_split_identity_mapping(value: object) -> dict[tuple[str, str], str]:
    if not isinstance(value, dict) or set(value) != {"train", "test", "oot"}:
        raise ValueError("split_value_mapping must define train/test/oot")
    by_identity: dict[tuple[str, str], str] = {}
    for canonical in ("train", "test", "oot"):
        identity = json_scalar_identity(value[canonical])
        if identity in by_identity:
            raise ValueError("split_value_mapping values must be typed-distinct")
        by_identity[identity] = canonical
    return by_identity


def _canonical_split_series(values: pd.Series, mapping: object) -> pd.Series:
    by_identity = _validated_split_identity_mapping(mapping)
    canonical: list[str] = []
    for value in values.tolist():
        try:
            identity = json_scalar_identity(value)
        except ValueError as exc:
            raise ValueError("split contains an invalid confirmed value") from exc
        if identity not in by_identity:
            raise ValueError("split contains a value outside confirmed mapping")
        canonical.append(by_identity[identity])
    return pd.Series(canonical, index=values.index, dtype="object")


def _check_cancelled(callback: Callable[[], None] | None) -> None:
    if callback is not None:
        callback()


def _file_identity(path: Path, *, label: str) -> tuple[int, int, int, int, int]:
    try:
        current = path.stat()
    except OSError as exc:
        raise ValueError(f"unable to inspect {label}") from exc
    if not stat.S_ISREG(current.st_mode):
        raise ValueError(f"{label} must be a regular file")
    return (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
        current.st_ctime_ns,
    )


def _require_file_identity(
    path: Path,
    expected: tuple[int, int, int, int, int],
    *,
    label: str,
) -> None:
    if _file_identity(path, label=label) != expected:
        raise ValueError(f"{label} changed while metrics were loading")


def stress_category_resolution_for_metrics(
    *,
    feature_importance: list[FeatureImportanceRow],
    fallback_model_features: list[str] | None = None,
    dictionary: pd.DataFrame,
    feature_col: str,
    category_col: str,
    stress_scores_path: Path | None,
) -> FeatureCategoryResolution:
    model_features = [(row.feature, row.category) for row in feature_importance]
    if not model_features:
        dictionary_features = {
            str(value).strip()
            for value in dictionary[feature_col].tolist()
            if pd.notna(value) and str(value).strip()
        }
        model_features = [
            (feature, "")
            for feature in (fallback_model_features or [])
            if feature in dictionary_features
        ]
    expected = resolve_feature_categories(
        model_features=model_features,
        dictionary=dictionary,
        feature_col=feature_col,
        category_col=category_col,
    )
    _raise_category_conflict(expected)
    if stress_scores_path is None or not Path(stress_scores_path).exists():
        return expected

    payload = json.loads(Path(stress_scores_path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "marvis.validation_stress_scores.v2":
        raise ValueError(
            "stress scenario artifact category mapping does not match model metadata"
        )
    actual = _category_resolution_from_stress_payload(payload)
    _raise_category_conflict(actual)
    if (
        actual.per_category != expected.per_category
        or actual.unclassified_features != expected.unclassified_features
        or actual.source_counts != expected.source_counts
    ):
        raise ValueError(
            "stress scenario artifact category mapping does not match model metadata"
        )
    return actual


def _category_resolution_from_stress_payload(
    payload: dict[str, Any],
) -> FeatureCategoryResolution:
    raw_categories = payload.get("feature_categories") or {}
    per_category = {
        str(category): [str(feature) for feature in features]
        for category, features in raw_categories.items()
    }
    conflicts = [
        FeatureCategoryConflict(
            feature=str(row.get("feature") or ""),
            categories=tuple(str(value) for value in row.get("categories") or []),
            source=str(row.get("source") or ""),
        )
        for row in payload.get("conflicts") or []
    ]
    return FeatureCategoryResolution(
        per_category=per_category,
        unclassified_features=[
            str(feature) for feature in payload.get("unclassified_features") or []
        ],
        conflicts=conflicts,
        source_counts={
            str(source): int(count)
            for source, count in (payload.get("source_counts") or {}).items()
        },
    )


def _raise_category_conflict(resolution: FeatureCategoryResolution) -> None:
    if not resolution.conflicts:
        return
    conflict = resolution.conflicts[0]
    raise ValueError(
        f"stress category conflict for {conflict.feature}: "
        + ", ".join(conflict.categories)
    )


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
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("reproducibility result must be an object")
    return reproducibility_result_from_dict(payload)


def _normalize_row_index(value: Any) -> int | str:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value if isinstance(value, (int, str)) else str(value)
