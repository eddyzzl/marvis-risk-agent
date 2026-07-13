from __future__ import annotations

from copy import deepcopy
import dataclasses
from dataclasses import asdict, dataclass, field
from enum import Enum
import math
import re
import types
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from marvis.compat import StrEnum

_UNSET = object()
PMML_SCORING_RESULT_SCHEMA = "marvis.pmml_scoring.v1"
MAX_PMML_SCORING_ERRORS = 64
MAX_PMML_SCORING_ERROR_CHARS = 500
VALIDATION_RESULTS_SCHEMA_V1 = "marvis.validation_results.v1"
VALIDATION_RESULTS_SCHEMA_V2 = "marvis.validation_results.v2"
VALIDATION_LIFT_ORDER_GOOD_TO_BAD = "good_to_bad"
_VALIDATION_LIFT_ORDER_LEGACY_BAD_TO_GOOD = "bad_to_good"


class ConsistencyStatus(StrEnum):
    PASS = "pass"
    REVIEW = "review"
    FAIL = "fail"


@dataclass(frozen=True, init=False)
class ScoreCompareRow:
    row_index: int | str
    score_code_model: float
    score_submitted_pmml: float | None
    abs_diff: float | None
    matched: bool

    def __init__(
        self,
        row_index: int | str,
        score_code_model: float | None = None,
        score_submitted_pmml: float | None | object = _UNSET,
        abs_diff: float | None = None,
        matched: bool | None = None,
        *,
        score_trained_pmml: float | None = None,
        score_input_pmml: float | None | object = _UNSET,
        score_sample_col: float | None = None,
    ) -> None:
        if score_code_model is None:
            score_code_model = score_trained_pmml
        if score_submitted_pmml is _UNSET:
            score_submitted_pmml = score_input_pmml
        if score_code_model is None or score_submitted_pmml is _UNSET:
            raise TypeError("score_code_model and score_submitted_pmml are required")

        # Backward-compatible construction for older tests and saved callers:
        # ScoreCompareRow(row_index, trained_pmml, input_pmml, sample_score)
        if matched is None and score_sample_col is None and abs_diff is not None:
            score_sample_col = abs_diff
            abs_diff = None

        code_score = float(score_code_model)
        submitted_score = None if score_submitted_pmml is None else float(score_submitted_pmml)
        actual_diff = None if submitted_score is None else abs(code_score - submitted_score)
        object.__setattr__(self, "row_index", _normalise_row_index(row_index))
        object.__setattr__(self, "score_code_model", code_score)
        object.__setattr__(self, "score_submitted_pmml", submitted_score)
        if abs_diff is None:
            object.__setattr__(self, "abs_diff", actual_diff)
        else:
            object.__setattr__(self, "abs_diff", float(abs_diff))
        default_matched = bool(actual_diff == 0.0) if actual_diff is not None else False
        object.__setattr__(self, "matched", bool(default_matched if matched is None else matched))

    @property
    def score_trained_pmml(self) -> float:
        return self.score_code_model

    @property
    def score_input_pmml(self) -> float | None:
        return self.score_submitted_pmml

    @property
    def score_sample_col(self) -> float:
        return self.score_code_model


@dataclass(frozen=True)
class ConsistencySummary:
    match_count: int
    mismatch_count: int
    max_abs_diff: float
    status: ConsistencyStatus


@dataclass(frozen=True)
class ReproducibilityResult:
    sample_size: int
    seed: int
    rows: list[ScoreCompareRow]
    summary: ConsistencySummary


@dataclass(frozen=True)
class PmmlScoringResult:
    schema_version: str
    cache_key: str
    pmml_sha256: str
    sample_sha256: str
    engine: str
    engine_version: str
    output_field: str
    input_row_count: int
    success_count: int
    failure_count: int
    null_count: int
    non_finite_count: int
    elapsed_seconds: float
    rows_per_second: float
    chunk_size: int
    required_input_count: int
    missing_inputs: list[str]
    score_artifact_path: str
    score_artifact_sha256: str
    status: str
    bounded_errors: list[str]


def validate_pmml_scoring_result_fields(
    result: PmmlScoringResult,
) -> PmmlScoringResult:
    """Validate persisted PMML scoring evidence without coercing its values."""
    if not isinstance(result, PmmlScoringResult):
        raise ValueError("PMML scoring result has an invalid type")
    if result.schema_version != PMML_SCORING_RESULT_SCHEMA:
        raise ValueError(
            f"unsupported PMML scoring schema: {result.schema_version!r}"
        )

    integer_fields = (
        "input_row_count",
        "success_count",
        "failure_count",
        "null_count",
        "non_finite_count",
        "chunk_size",
        "required_input_count",
    )
    for name in integer_fields:
        value = getattr(result, name)
        if type(value) is not int or value < 0:
            raise ValueError(f"invalid non-negative PMML scoring integer: {name}")

    for name in ("elapsed_seconds", "rows_per_second"):
        value = getattr(result, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"invalid PMML scoring number: {name}")
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"invalid PMML scoring number: {name}")

    if result.input_row_count <= 0 or result.chunk_size <= 0:
        raise ValueError("PMML scoring row count and chunk size must be positive")
    if result.success_count + result.failure_count != result.input_row_count:
        raise ValueError("PMML scoring success/failure counts do not add to input")
    if result.failure_count != result.null_count + result.non_finite_count:
        raise ValueError("PMML scoring failure detail counts are inconsistent")

    if result.status not in {"pass", "failed"}:
        raise ValueError("invalid PMML scoring status")
    _require_non_empty_string_list(result.missing_inputs, "missing_inputs")
    _validate_bounded_scoring_errors(result.bounded_errors)
    if result.status == "pass" and (
        result.failure_count
        or result.null_count
        or result.non_finite_count
        or result.missing_inputs
        or result.bounded_errors
    ):
        raise ValueError("passing PMML scoring evidence contains failures")
    if result.status == "failed" and not (
        result.failure_count or result.missing_inputs or result.bounded_errors
    ):
        raise ValueError("failed PMML scoring evidence is empty")

    for name in (
        "cache_key",
        "pmml_sha256",
        "sample_sha256",
        "score_artifact_sha256",
    ):
        value = getattr(result, name)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"invalid PMML scoring SHA-256 field: {name}")
    for name in ("engine", "engine_version", "output_field", "score_artifact_path"):
        value = getattr(result, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"missing PMML scoring identity field: {name}")
    return result


def pmml_scoring_result_to_dict(result: PmmlScoringResult) -> dict[str, Any]:
    validate_pmml_scoring_result_fields(result)
    return asdict(result)


def pmml_scoring_result_from_dict(payload: dict[str, Any]) -> PmmlScoringResult:
    if not isinstance(payload, dict):
        raise ValueError("PMML scoring result must be an object")
    expected = {item.name for item in dataclasses.fields(PmmlScoringResult)}
    actual = set(payload)
    unknown = actual - expected
    missing = expected - actual
    if unknown or missing:
        raise ValueError(
            "invalid PMML scoring result; "
            f"missing={_bounded_field_names(missing)}, "
            f"unknown={_bounded_field_names(unknown)}"
        )
    try:
        result = PmmlScoringResult(**payload)
    except TypeError as exc:
        raise ValueError("invalid PMML scoring result fields") from exc
    return validate_pmml_scoring_result_fields(result)


def _require_non_empty_string_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and bool(item.strip()) for item in value
    ):
        raise ValueError(f"invalid PMML scoring {name}")
    return value


def _validate_bounded_scoring_errors(value: object) -> None:
    errors = _require_non_empty_string_list(value, "bounded_errors")
    if len(errors) > MAX_PMML_SCORING_ERRORS:
        raise ValueError("too many PMML scoring bounded_errors")
    if any(len(item) > MAX_PMML_SCORING_ERROR_CHARS for item in errors):
        raise ValueError("PMML scoring bounded_error is too long")


def _bounded_field_names(values: set[object]) -> list[str]:
    return sorted(str(value)[:80] for value in values)[:32]


@dataclass(frozen=True)
class SplitRow:
    split: str
    sample_count: int
    bad_count: int
    bad_rate: float
    period_start: str = ""
    period_end: str = ""


@dataclass(frozen=True)
class MonthlyRow:
    month: str
    sample_count: int
    bad_count: int
    bad_rate: float


@dataclass(frozen=True)
class FeatureImportanceRow:
    rank: int
    feature: str
    importance: float
    category: str = ""


@dataclass(frozen=True)
class BasicInfoResult:
    sample_period: tuple[str, str]
    split_summary: list[SplitRow]
    monthly_distribution: list[MonthlyRow]
    hyperparameters: dict[str, Any]
    feature_importance: list[FeatureImportanceRow]


@dataclass(frozen=True)
class OverallRow:
    split: str
    ks: float
    psi_vs_train: float
    sample_count: int
    bad_rate: float
    bad_count: int = 0
    auc: float = 0.0
    head_lift_5pct: float | None = None
    tail_lift_5pct: float | None = None


@dataclass(frozen=True)
class BinRow:
    bin_index: int
    score_lower: float
    score_upper: float
    sample_count: int
    bad_count: int
    bad_rate: float
    cum_sample_pct: float
    cum_bad_pct: float
    lift: float
    ks: float


@dataclass(frozen=True)
class PsiStabilityRow:
    bin_label: str
    expected_count: int
    expected_pct: float
    actual_count: int
    actual_pct: float
    psi: float


@dataclass(frozen=True)
class RocKsCurve:
    split: str
    fpr: list[float]
    tpr: list[float]
    ks_curve: list[float]
    ks: float
    population_at_ks: float


@dataclass(frozen=True)
class MonthlyKsRow:
    month: str
    ks: float
    sample_count: int
    bad_count: int = 0
    bad_rate: float = 0.0
    auc: float = 0.0
    head_lift_5pct: float | None = None
    tail_lift_5pct: float | None = None


@dataclass(frozen=True)
class MonthlyPsiRow:
    month: str
    psi_vs_train: float
    psi_first_month: float | None = None
    psi_last_month: float | None = None
    psi_mom: float | None = None
    psi_mom_reference_month: str = ""
    psi_mom_has_calendar_gap: bool = False


@dataclass(frozen=True)
class EffectivenessResult:
    overall: list[OverallRow]
    bin_tables: dict[str, list[BinRow]]
    monthly_ks: list[MonthlyKsRow]
    monthly_psi: list[MonthlyPsiRow]
    psi_stability_table: list[PsiStabilityRow] = field(default_factory=list)
    roc_ks_curves: dict[str, RocKsCurve] = field(default_factory=dict)


@dataclass(frozen=True)
class StressBaseline:
    ks: float
    sample_count: int
    bin_table: list[BinRow]


@dataclass(frozen=True)
class StressCategoryResult:
    category: str
    dropped_features: list[str]
    ks_after: float | None
    ks_delta: float | None
    psi_vs_baseline: float | None
    bin_table: list[BinRow]
    error: str | None
    status: str = "completed"


@dataclass(frozen=True)
class StressTestResult:
    baseline: StressBaseline
    per_category: list[StressCategoryResult]
    status: str = "completed"
    unclassified_features: list[str] = field(default_factory=list)
    category_source_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationResults:
    model_name: str
    model_version: str
    algorithm: str
    target_type: Literal["binary"]
    basic_info: BasicInfoResult
    effectiveness: EffectivenessResult
    stress_test: StressTestResult
    # Compatibility-safe default: historical constructors remain legacy until
    # a PMML-only assembly point explicitly opts into v2.
    schema_version: str = VALIDATION_RESULTS_SCHEMA_V1
    pmml_scoring: PmmlScoringResult | None = None
    reproducibility: ReproducibilityResult | None = None


_VALIDATION_RESULTS_BASE_FIELDS = {
    "model_name",
    "model_version",
    "algorithm",
    "target_type",
    "basic_info",
    "effectiveness",
    "stress_test",
}
_VALIDATION_RESULTS_ENVELOPE_FIELDS = _VALIDATION_RESULTS_BASE_FIELDS | {
    "schema_version",
    "lift_order",
    "pmml_scoring",
    "reproducibility",
    # Historical pipeline payloads occasionally wrapped the deterministic
    # result with this known orchestration identity. It is intentionally not
    # part of ValidationResults, but remains a recognized compatibility field.
    "task_id",
}
_DATACLASS_FIELD_ALIASES: dict[type[Any], dict[str, str]] = {
    # Older validation JSON used the dictionary header as the result key.
    FeatureImportanceRow: {"类别": "category"},
}


def validation_results_to_dict(results: ValidationResults) -> dict[str, Any]:
    """Serialize one canonical version of deterministic validation results."""
    if not isinstance(results, ValidationResults):
        raise ValueError("invalid validation results: expected ValidationResults")

    try:
        payload: dict[str, Any] = {
            "model_name": results.model_name,
            "model_version": results.model_version,
            "algorithm": results.algorithm,
            "target_type": results.target_type,
            "basic_info": asdict(results.basic_info),
            "effectiveness": asdict(results.effectiveness),
            "stress_test": asdict(results.stress_test),
            "schema_version": results.schema_version,
            "lift_order": VALIDATION_LIFT_ORDER_GOOD_TO_BAD,
        }
    except (TypeError, AttributeError) as exc:
        raise ValueError("invalid validation results fields") from exc

    if results.schema_version == VALIDATION_RESULTS_SCHEMA_V2:
        if results.pmml_scoring is None or results.reproducibility is not None:
            raise ValueError("v2 validation results require only pmml_scoring")
        payload["pmml_scoring"] = pmml_scoring_result_to_dict(results.pmml_scoring)
    elif results.schema_version == VALIDATION_RESULTS_SCHEMA_V1:
        if results.reproducibility is None or results.pmml_scoring is not None:
            raise ValueError("v1 validation results require only reproducibility")
        try:
            payload["reproducibility"] = asdict(results.reproducibility)
        except (TypeError, AttributeError) as exc:
            raise ValueError("invalid validation results reproducibility") from exc
    else:
        raise ValueError(
            f"unsupported validation results schema: {results.schema_version!r}"
        )

    # Reuse the strict persistence decoder so direct dataclass construction
    # cannot serialize malformed nested evidence.
    validation_results_from_dict(payload)
    return payload


def validation_results_from_dict(payload: dict[str, Any]) -> ValidationResults:
    """Decode persisted validation results without defaulting damaged fields."""
    _require_dict(payload, "root")
    payload = normalize_validation_results_lift_order(payload)
    actual = set(payload)
    unknown = actual - _VALIDATION_RESULTS_ENVELOPE_FIELDS
    missing = _VALIDATION_RESULTS_BASE_FIELDS - actual
    if unknown or missing:
        raise _invalid_validation_results(
            "root",
            f"missing={_bounded_field_names(missing)}, "
            f"unknown={_bounded_field_names(unknown)}",
        )
    if "task_id" in payload:
        _strict_value(payload["task_id"], str, "task_id")

    has_reproducibility = "reproducibility" in payload
    has_pmml_scoring = "pmml_scoring" in payload
    if "schema_version" not in payload:
        # Only historical reproducibility payloads predate the version field.
        # PMML-only results were introduced together with v2 and therefore
        # must never be inferred from a damaged envelope.
        if not has_reproducibility or has_pmml_scoring:
            raise _invalid_validation_results(
                "root", "legacy payload must contain only reproducibility"
            )
        schema = VALIDATION_RESULTS_SCHEMA_V1
    else:
        schema = _strict_value(payload["schema_version"], str, "schema_version")

    lift_order = _strict_value(
        payload.get("lift_order"),
        Literal["good_to_bad"],
        "lift_order",
    )
    if lift_order != VALIDATION_LIFT_ORDER_GOOD_TO_BAD:
        raise ValueError(f"unsupported validation lift order: {lift_order!r}")

    if schema == VALIDATION_RESULTS_SCHEMA_V2:
        if has_reproducibility or not has_pmml_scoring:
            raise ValueError("v2 validation results require only pmml_scoring")
        try:
            scoring = pmml_scoring_result_from_dict(
                _require_dict(payload["pmml_scoring"], "pmml_scoring")
            )
        except ValueError as exc:
            raise _invalid_validation_results("pmml_scoring", str(exc)) from exc
        reproducibility = None
    elif schema == VALIDATION_RESULTS_SCHEMA_V1:
        if has_pmml_scoring:
            raise ValueError("v1 validation results cannot contain pmml_scoring")
        if not has_reproducibility:
            raise ValueError("v1 validation results require only reproducibility")
        scoring = None
        reproducibility = reproducibility_result_from_dict(
            _require_dict(payload["reproducibility"], "reproducibility")
        )
    else:
        raise ValueError(f"unsupported validation results schema: {schema!r}")

    return ValidationResults(
        model_name=_strict_value(payload["model_name"], str, "model_name"),
        model_version=_strict_value(
            payload["model_version"], str, "model_version"
        ),
        algorithm=_strict_value(payload["algorithm"], str, "algorithm"),
        target_type=_strict_value(
            payload["target_type"], Literal["binary"], "target_type"
        ),
        basic_info=_decode_dataclass(payload["basic_info"], BasicInfoResult, "basic_info"),
        effectiveness=_decode_dataclass(
            payload["effectiveness"], EffectivenessResult, "effectiveness"
        ),
        stress_test=_decode_dataclass(
            payload["stress_test"], StressTestResult, "stress_test"
        ),
        schema_version=schema,
        pmml_scoring=scoring,
        reproducibility=reproducibility,
    )


def normalize_validation_results_lift_order(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a detached payload using the current good-head/bad-tail convention.

    Early PMML-only V2 artifacts predate an explicit lift-order marker and stored
    overall/monthly fields as high-risk ``head`` and low-risk ``tail``. Unmarked V1
    artifacts exist in both conventions, so their observed head/tail direction is
    used only as a compatibility heuristic; explicitly marked artifacts are never
    inferred from their values.
    """
    normalized = deepcopy(_require_dict(payload, "root"))
    schema = normalized.get("schema_version")
    if schema not in {None, VALIDATION_RESULTS_SCHEMA_V1, VALIDATION_RESULTS_SCHEMA_V2}:
        return normalized

    lift_order = normalized.get("lift_order")
    if lift_order == VALIDATION_LIFT_ORDER_GOOD_TO_BAD:
        return normalized
    if lift_order not in {None, _VALIDATION_LIFT_ORDER_LEGACY_BAD_TO_GOOD}:
        return normalized

    should_swap = lift_order == _VALIDATION_LIFT_ORDER_LEGACY_BAD_TO_GOOD
    if lift_order is None:
        should_swap = schema == VALIDATION_RESULTS_SCHEMA_V2
        if schema in {None, VALIDATION_RESULTS_SCHEMA_V1}:
            should_swap = _unmarked_v1_uses_legacy_bad_head(normalized)

    if should_swap:
        _swap_validation_head_tail_lift(normalized)
    _normalize_persisted_validation_bin_tables(normalized)
    normalized["lift_order"] = VALIDATION_LIFT_ORDER_GOOD_TO_BAD
    return normalized


def _swap_validation_head_tail_lift(payload: dict[str, Any]) -> None:
    effectiveness = payload.get("effectiveness")
    if not isinstance(effectiveness, dict):
        return
    for section in ("overall", "monthly_ks"):
        rows = effectiveness.get(section)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            if "head_lift_5pct" not in row and "tail_lift_5pct" not in row:
                continue
            head = row.get("head_lift_5pct")
            tail = row.get("tail_lift_5pct")
            row["head_lift_5pct"] = tail
            row["tail_lift_5pct"] = head


def _unmarked_v1_uses_legacy_bad_head(payload: dict[str, Any]) -> bool:
    effectiveness = payload.get("effectiveness")
    if isinstance(effectiveness, dict):
        for section in ("overall", "monthly_ks"):
            rows = effectiveness.get(section)
            if not isinstance(rows, list):
                continue
            direction_votes: list[int] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                head = row.get("head_lift_5pct")
                tail = row.get("tail_lift_5pct")
                if not isinstance(head, (int, float)) or not isinstance(tail, (int, float)):
                    continue
                if not math.isfinite(float(head)) or not math.isfinite(float(tail)):
                    continue
                direction_votes.append((head > tail) - (head < tail))
            vote_total = sum(direction_votes)
            if vote_total:
                return vote_total > 0
    return False


def _normalize_persisted_validation_bin_tables(payload: dict[str, Any]) -> None:
    effectiveness = payload.get("effectiveness")
    if isinstance(effectiveness, dict):
        bin_tables = effectiveness.get("bin_tables")
        if isinstance(bin_tables, dict):
            for split, rows in list(bin_tables.items()):
                if isinstance(rows, list):
                    bin_tables[split] = _ordered_persisted_bin_rows(
                        rows,
                        reverse=_persisted_bin_rows_are_bad_to_good(rows),
                    )

    stress = payload.get("stress_test")
    if not isinstance(stress, dict):
        return
    baseline = stress.get("baseline")
    baseline_rows = baseline.get("bin_table") if isinstance(baseline, dict) else None
    if not isinstance(baseline_rows, list):
        return
    reverse_stress = _persisted_bin_rows_are_bad_to_good(baseline_rows)
    baseline["bin_table"] = _ordered_persisted_bin_rows(
        baseline_rows,
        reverse=reverse_stress,
    )
    categories = stress.get("per_category")
    if not isinstance(categories, list):
        return
    for category in categories:
        if not isinstance(category, dict):
            continue
        rows = category.get("bin_table")
        if isinstance(rows, list):
            category["bin_table"] = _ordered_persisted_bin_rows(
                rows,
                reverse=reverse_stress,
            )


def _persisted_bin_rows_are_bad_to_good(rows: list[object]) -> bool:
    observations: list[tuple[float, float, float]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        count = row.get("sample_count")
        bad_count = row.get("bad_count")
        bad_rate = row.get("bad_rate")
        if not isinstance(count, (int, float)) or float(count) <= 0:
            continue
        if not isinstance(bad_rate, (int, float)):
            if not isinstance(bad_count, (int, float)):
                continue
            bad_rate = float(bad_count) / float(count)
        if not math.isfinite(float(bad_rate)):
            continue
        observations.append((float(index), float(bad_rate), float(count)))
    if len(observations) < 2:
        return False
    weight = sum(item[2] for item in observations)
    mean_index = sum(index * count for index, _, count in observations) / weight
    mean_rate = sum(rate * count for _, rate, count in observations) / weight
    covariance = sum(
        count * (index - mean_index) * (rate - mean_rate)
        for index, rate, count in observations
    )
    return covariance < 0


def _ordered_persisted_bin_rows(
    rows: list[object],
    *,
    reverse: bool,
) -> list[object]:
    if not rows or not all(isinstance(row, dict) for row in rows):
        return rows
    ordered = [dict(row) for row in (reversed(rows) if reverse else rows)]
    try:
        total = sum(int(row["sample_count"]) for row in ordered)
        total_bad = sum(int(row["bad_count"]) for row in ordered)
    except (KeyError, TypeError, ValueError):
        return rows
    total_good = total - total_bad
    overall_bad_rate = float(total_bad / total) if total else 0.0
    cumulative_count = 0
    cumulative_bad = 0
    for bin_index, row in enumerate(ordered, start=1):
        count = int(row["sample_count"])
        bad = int(row["bad_count"])
        cumulative_count += count
        cumulative_bad += bad
        cumulative_good = cumulative_count - cumulative_bad
        bad_rate = float(bad / count) if count else 0.0
        cumulative_bad_pct = float(cumulative_bad / total_bad) if total_bad else 0.0
        cumulative_good_pct = float(cumulative_good / total_good) if total_good else 0.0
        row.update(
            {
                "bin_index": bin_index,
                "bad_rate": bad_rate,
                "cum_sample_pct": float(cumulative_count / total) if total else 0.0,
                "cum_bad_pct": cumulative_bad_pct,
                "lift": float(bad_rate / overall_bad_rate) if overall_bad_rate else 0.0,
                "ks": float(abs(cumulative_bad_pct - cumulative_good_pct)),
            }
        )
    return ordered


def reproducibility_result_from_dict(
    payload: dict[str, Any],
) -> ReproducibilityResult:
    return _decode_dataclass(payload, ReproducibilityResult, "reproducibility")


def _decode_dataclass(payload: object, cls: type[Any], path: str):
    value = dict(_require_dict(payload, path))
    for alias, canonical in _DATACLASS_FIELD_ALIASES.get(cls, {}).items():
        if alias not in value:
            continue
        if canonical in value:
            raise _invalid_validation_results(
                path, f"both {alias!r} and {canonical!r} are present"
            )
        value[canonical] = value.pop(alias)
    fields = {item.name: item for item in dataclasses.fields(cls)}
    unknown = set(value) - set(fields)
    missing = {
        name
        for name, item in fields.items()
        if name not in value
        and item.default is dataclasses.MISSING
        and item.default_factory is dataclasses.MISSING
    }
    if unknown or missing:
        raise _invalid_validation_results(
            path,
            f"missing={_bounded_field_names(missing)}, "
            f"unknown={_bounded_field_names(unknown)}",
        )
    hints = get_type_hints(cls)
    decoded = {
        name: _strict_value(value[name], hints[name], f"{path}.{name}")
        for name in fields
        if name in value
    }
    # Preserve the two status inferences used by pre-versioned result files.
    # Dataclass defaults alone would silently reinterpret an omitted failed or
    # empty stress result as "completed".
    if cls is StressCategoryResult and "status" not in value:
        decoded["status"] = "error" if decoded.get("error") else "completed"
    if cls is StressTestResult and "status" not in value:
        decoded["status"] = _stress_test_status_from_categories(
            decoded.get("per_category", [])
        )
    try:
        return cls(**decoded)
    except (TypeError, ValueError) as exc:
        raise _invalid_validation_results(path, "invalid field values") from exc


def _strict_value(value: object, annotation: object, path: str):
    if annotation is Any:
        return _strict_json_value(value, path)
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Literal:
        if value not in args or type(value) not in {type(item) for item in args}:
            raise _invalid_validation_results(path, f"expected {annotation!r}")
        return value
    if origin in {Union, types.UnionType}:
        for option in args:
            try:
                return _strict_value(value, option, path)
            except ValueError:
                continue
        raise _invalid_validation_results(path, f"expected {annotation!r}")
    if origin is list:
        if type(value) is not list:
            raise _invalid_validation_results(path, "expected list")
        return [
            _strict_value(item, args[0], f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if origin is dict:
        if type(value) is not dict:
            raise _invalid_validation_results(path, "expected object")
        return {
            _strict_value(key, args[0], f"{path}.<key>"): _strict_value(
                item, args[1], f"{path}.{key}"
            )
            for key, item in value.items()
        }
    if origin is tuple:
        if not isinstance(value, (list, tuple)) or len(value) != len(args):
            raise _invalid_validation_results(path, "expected fixed-length array")
        return tuple(
            _strict_value(item, item_type, f"{path}[{index}]")
            for index, (item, item_type) in enumerate(zip(value, args, strict=True))
        )
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        return _decode_dataclass(value, annotation, path)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        if isinstance(value, annotation):
            return value
        if type(value) is not str:
            raise _invalid_validation_results(path, f"expected {annotation.__name__}")
        try:
            return annotation(value)
        except ValueError as exc:
            raise _invalid_validation_results(
                path, f"invalid {annotation.__name__}"
            ) from exc
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _invalid_validation_results(path, "expected number")
        numeric = float(value)
        if math.isnan(numeric) or (
            math.isinf(numeric)
            and not path.endswith((".score_lower", ".score_upper"))
        ):
            raise _invalid_validation_results(path, "expected finite number")
        return numeric
    if annotation is int:
        if type(value) is not int:
            raise _invalid_validation_results(path, "expected integer")
        return value
    if annotation is str:
        if type(value) is not str:
            raise _invalid_validation_results(path, "expected string")
        return value
    if annotation is bool:
        if type(value) is not bool:
            raise _invalid_validation_results(path, "expected boolean")
        return value
    if annotation is type(None):
        if value is not None:
            raise _invalid_validation_results(path, "expected null")
        return None
    raise _invalid_validation_results(path, f"unsupported field type {annotation!r}")


def _strict_json_value(value: object, path: str):
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise _invalid_validation_results(path, "expected finite JSON number")
        return value
    if type(value) is list:
        return [
            _strict_json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if type(value) is dict:
        if not all(type(key) is str for key in value):
            raise _invalid_validation_results(path, "expected string JSON keys")
        return {
            key: _strict_json_value(item, f"{path}.{key}")
            for key, item in value.items()
        }
    raise _invalid_validation_results(path, "expected JSON value")


def _stress_test_status_from_categories(
    per_category: list[StressCategoryResult],
) -> str:
    if not per_category:
        return "skipped"
    statuses = {row.status for row in per_category}
    if statuses == {"completed"}:
        return "completed"
    if statuses == {"skipped"}:
        return "skipped"
    if statuses == {"error"}:
        return "failed"
    return "partial"


def _require_dict(value: object, path: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise _invalid_validation_results(path, "expected object")
    return value


def _invalid_validation_results(path: str, detail: str) -> ValueError:
    return ValueError(f"invalid validation results at {path}: {detail}")


def _normalise_row_index(row_index: object) -> int | str:
    if hasattr(row_index, "item"):
        row_index = row_index.item()
    if isinstance(row_index, float) and row_index.is_integer():
        row_index = int(row_index)
    return row_index if isinstance(row_index, (int, str)) else str(row_index)
