from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
import math
import re
from typing import Any, Literal

from marvis.compat import StrEnum

_UNSET = object()
PMML_SCORING_RESULT_SCHEMA = "marvis.pmml_scoring.v1"
MAX_PMML_SCORING_ERRORS = 64
MAX_PMML_SCORING_ERROR_CHARS = 500


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
    reproducibility: ReproducibilityResult
    basic_info: BasicInfoResult
    effectiveness: EffectivenessResult
    stress_test: StressTestResult


def validation_results_from_dict(payload: dict[str, Any]) -> ValidationResults:
    return ValidationResults(
        model_name=str(payload.get("model_name") or ""),
        model_version=str(payload.get("model_version") or ""),
        algorithm=str(payload.get("algorithm") or ""),
        target_type="binary",
        reproducibility=_reproducibility_from_dict(payload.get("reproducibility") or {}),
        basic_info=_basic_info_from_dict(payload.get("basic_info") or {}),
        effectiveness=_effectiveness_from_dict(payload.get("effectiveness") or {}),
        stress_test=_stress_test_from_dict(payload.get("stress_test") or {}),
    )


def _reproducibility_from_dict(payload: dict[str, Any]) -> ReproducibilityResult:
    summary = payload.get("summary") or {}
    return ReproducibilityResult(
        sample_size=int(payload.get("sample_size") or 0),
        seed=int(payload.get("seed") or 0),
        rows=[
            ScoreCompareRow(
                row_index=row.get("row_index", 0),
                score_code_model=row.get("score_code_model"),
                score_submitted_pmml=row.get("score_submitted_pmml"),
                abs_diff=row.get("abs_diff"),
                matched=bool(row.get("matched")),
            )
            for row in payload.get("rows") or []
        ],
        summary=ConsistencySummary(
            match_count=int(summary.get("match_count") or 0),
            mismatch_count=int(summary.get("mismatch_count") or 0),
            max_abs_diff=float(summary.get("max_abs_diff") or 0.0),
            status=ConsistencyStatus(summary.get("status") or ConsistencyStatus.REVIEW),
        ),
    )


def _normalise_row_index(row_index: object) -> int | str:
    if hasattr(row_index, "item"):
        row_index = row_index.item()
    if isinstance(row_index, float) and row_index.is_integer():
        row_index = int(row_index)
    return row_index if isinstance(row_index, (int, str)) else str(row_index)


def _basic_info_from_dict(payload: dict[str, Any]) -> BasicInfoResult:
    sample_period = list(payload.get("sample_period") or ["", ""])
    while len(sample_period) < 2:
        sample_period.append("")
    return BasicInfoResult(
        sample_period=(str(sample_period[0]), str(sample_period[1])),
        split_summary=[
            SplitRow(
                split=str(row.get("split") or ""),
                sample_count=int(row.get("sample_count") or 0),
                bad_count=int(row.get("bad_count") or 0),
                bad_rate=float(row.get("bad_rate") or 0.0),
                period_start=str(row.get("period_start") or ""),
                period_end=str(row.get("period_end") or ""),
            )
            for row in payload.get("split_summary") or []
        ],
        monthly_distribution=[
            MonthlyRow(
                month=str(row.get("month") or ""),
                sample_count=int(row.get("sample_count") or 0),
                bad_count=int(row.get("bad_count") or 0),
                bad_rate=float(row.get("bad_rate") or 0.0),
            )
            for row in payload.get("monthly_distribution") or []
        ],
        hyperparameters=dict(payload.get("hyperparameters") or {}),
        feature_importance=[
            FeatureImportanceRow(
                rank=int(row.get("rank") or 0),
                feature=str(row.get("feature") or ""),
                importance=float(row.get("importance") or 0.0),
                category=str(row.get("category") or row.get("类别") or ""),
            )
            for row in payload.get("feature_importance") or []
        ],
    )


def _effectiveness_from_dict(payload: dict[str, Any]) -> EffectivenessResult:
    return EffectivenessResult(
        overall=[
            OverallRow(
                split=str(row.get("split") or ""),
                ks=float(row.get("ks") or 0.0),
                psi_vs_train=float(row.get("psi_vs_train") or 0.0),
                sample_count=int(row.get("sample_count") or 0),
                bad_rate=float(row.get("bad_rate") or 0.0),
                bad_count=int(row.get("bad_count") or 0),
                auc=float(row.get("auc") or 0.0),
                head_lift_5pct=_optional_float(row.get("head_lift_5pct")),
                tail_lift_5pct=_optional_float(row.get("tail_lift_5pct")),
            )
            for row in payload.get("overall") or []
        ],
        bin_tables={
            str(split): [_bin_row_from_dict(row) for row in rows]
            for split, rows in (payload.get("bin_tables") or {}).items()
        },
        monthly_ks=[
            MonthlyKsRow(
                month=str(row.get("month") or ""),
                ks=float(row.get("ks") or 0.0),
                sample_count=int(row.get("sample_count") or 0),
                bad_count=int(row.get("bad_count") or 0),
                bad_rate=float(row.get("bad_rate") or 0.0),
                auc=float(row.get("auc") or 0.0),
                head_lift_5pct=_optional_float(row.get("head_lift_5pct")),
                tail_lift_5pct=_optional_float(row.get("tail_lift_5pct")),
            )
            for row in payload.get("monthly_ks") or []
        ],
        monthly_psi=[
            MonthlyPsiRow(
                month=str(row.get("month") or ""),
                psi_vs_train=float(row.get("psi_vs_train") or 0.0),
                psi_first_month=_optional_float(row.get("psi_first_month")),
                psi_last_month=_optional_float(row.get("psi_last_month")),
                psi_mom=_optional_float(row.get("psi_mom")),
                psi_mom_reference_month=str(row.get("psi_mom_reference_month") or ""),
                psi_mom_has_calendar_gap=bool(row.get("psi_mom_has_calendar_gap")),
            )
            for row in payload.get("monthly_psi") or []
        ],
        psi_stability_table=[
            PsiStabilityRow(
                bin_label=str(row.get("bin_label") or ""),
                expected_count=int(row.get("expected_count") or 0),
                expected_pct=float(row.get("expected_pct") or 0.0),
                actual_count=int(row.get("actual_count") or 0),
                actual_pct=float(row.get("actual_pct") or 0.0),
                psi=float(row.get("psi") or 0.0),
            )
            for row in payload.get("psi_stability_table") or []
        ],
        roc_ks_curves={
            str(split): RocKsCurve(
                split=str(row.get("split") or split),
                fpr=[float(value) for value in row.get("fpr") or []],
                tpr=[float(value) for value in row.get("tpr") or []],
                ks_curve=[float(value) for value in row.get("ks_curve") or []],
                ks=float(row.get("ks") or 0.0),
                population_at_ks=float(row.get("population_at_ks") or 0.0),
            )
            for split, row in (payload.get("roc_ks_curves") or {}).items()
            if isinstance(row, dict)
        },
    )


def _stress_test_from_dict(payload: dict[str, Any]) -> StressTestResult:
    baseline = payload.get("baseline") or {}
    per_category = [
        StressCategoryResult(
            category=str(row.get("category") or ""),
            dropped_features=[str(feature) for feature in row.get("dropped_features") or []],
            ks_after=_optional_float(row.get("ks_after")),
            ks_delta=_optional_float(row.get("ks_delta")),
            psi_vs_baseline=_optional_float(row.get("psi_vs_baseline")),
            bin_table=[_bin_row_from_dict(item) for item in row.get("bin_table") or []],
            error=row.get("error"),
            status=str(row.get("status") or ("error" if row.get("error") else "completed")),
        )
        for row in payload.get("per_category") or []
    ]
    return StressTestResult(
        baseline=StressBaseline(
            ks=float(baseline.get("ks") or 0.0),
            sample_count=int(baseline.get("sample_count") or 0),
            bin_table=[_bin_row_from_dict(row) for row in baseline.get("bin_table") or []],
        ),
        per_category=per_category,
        status=str(payload.get("status") or _stress_test_status_from_categories(per_category)),
        unclassified_features=[
            str(feature) for feature in payload.get("unclassified_features") or []
        ],
        category_source_counts={
            str(source): int(count)
            for source, count in (payload.get("category_source_counts") or {}).items()
        },
    )


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


def _bin_row_from_dict(row: dict[str, Any]) -> BinRow:
    return BinRow(
        bin_index=int(row.get("bin_index") or 0),
        score_lower=float(row.get("score_lower") or 0.0),
        score_upper=float(row.get("score_upper") or 0.0),
        sample_count=int(row.get("sample_count") or 0),
        bad_count=int(row.get("bad_count") or 0),
        bad_rate=float(row.get("bad_rate") or 0.0),
        cum_sample_pct=float(row.get("cum_sample_pct") or 0.0),
        cum_bad_pct=float(row.get("cum_bad_pct") or 0.0),
        lift=float(row.get("lift") or 0.0),
        ks=float(row.get("ks") or 0.0),
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
