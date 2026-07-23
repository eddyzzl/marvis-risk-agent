from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path

from marvis.compat import StrEnum


class TaskStatus(StrEnum):
    CREATED = "created"
    SCANNED = "scanned"
    RUNNING = "running"
    EXECUTED = "executed"
    COMPUTING_METRICS = "computing_metrics"
    WRITING_ARTIFACTS = "writing_artifacts"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REVIEW_REQUIRED = "review_required"


TASK_STATUS_REASON_USER_CANCELLED = "user_cancelled"
TASK_STATUS_REASON_SERVER_RESTART = "server_restart_while_running"
TASK_TYPE_VALIDATION = "validation"
TASK_TYPE_FEATURE_ANALYSIS = "feature_analysis"
TASK_TYPE_DATA_JOIN = "data_join"
TASK_TYPE_MODELING = "modeling"
TASK_TYPE_STRATEGY = "strategy"
TASK_TYPE_VINTAGE = "vintage"
TASK_TYPE_PORTFOLIO = "portfolio"
# Known task types. New capabilities (modeling/strategy/...) must register here
# so _normalize_task_type can keep arbitrary strings out of the database.
VALID_TASK_TYPES = frozenset({
    TASK_TYPE_VALIDATION,
    TASK_TYPE_FEATURE_ANALYSIS,
    TASK_TYPE_DATA_JOIN,
    TASK_TYPE_MODELING,
    TASK_TYPE_STRATEGY,
    TASK_TYPE_VINTAGE,
    TASK_TYPE_PORTFOLIO,
})

STRATEGY_ENTRY_MODES = frozenset({"strategy_development", "strategy_analysis"})
STRATEGY_TYPES = frozenset(
    {"approval", "reject", "limit", "pricing", "segmentation"}
)
STRATEGY_OBJECTIVES = frozenset({"", "max_profit", "max_approval"})


class FileRole(StrEnum):
    NOTEBOOK = "notebook"
    SAMPLE = "sample"
    MODEL_PMML = "model_pmml"
    DATA_DICTIONARY = "data_dictionary"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class StrategyProfitInput:
    """Governed business assumptions required by a profit strategy objective.

    The contract is intentionally separate from the strategy Tool's runtime
    ``ProfitParams``: this object belongs to the persisted task boundary and also
    names the EAD/PD columns that make the expected-loss chain reproducible.
    """

    ead_col: str
    pd_col: str
    annual_rate: float
    funding_rate: float
    lgd: float
    operating_cost_per_loan: float
    term_months: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "ead_col", _required_contract_string("ead_col", self.ead_col))
        object.__setattr__(self, "pd_col", _required_contract_string("pd_col", self.pd_col))
        for field_name in ("annual_rate", "funding_rate", "lgd"):
            object.__setattr__(
                self,
                field_name,
                _bounded_contract_number(field_name, getattr(self, field_name), minimum=0.0, maximum=1.0),
            )
        object.__setattr__(
            self,
            "operating_cost_per_loan",
            _bounded_contract_number(
                "operating_cost_per_loan",
                self.operating_cost_per_loan,
                minimum=0.0,
            ),
        )
        if (
            isinstance(self.term_months, bool)
            or not isinstance(self.term_months, int)
            or self.term_months < 1
        ):
            raise ValueError("term_months must be an integer greater than or equal to 1")


@dataclass(frozen=True)
class StrategyTaskInput:
    """Persisted business contract for a strategy task.

    Empty objective/constraints are representable so the conversational setup can
    pause for clarification. They are never replaced here with technical defaults.
    ``strategy_input is None`` on ``TaskRecord`` remains distinct: it denotes a
    legacy task created before this governed contract existed.
    """

    entry_mode: str = "strategy_development"
    strategy_type: str = "approval"
    objective: str = ""
    max_bad_rate: float | None = None
    min_approval_rate: float | None = None
    baseline_strategy_id: str | None = None
    profit: StrategyProfitInput | None = None

    def __post_init__(self) -> None:
        if self.entry_mode not in STRATEGY_ENTRY_MODES:
            allowed = ", ".join(sorted(STRATEGY_ENTRY_MODES))
            raise ValueError(f"entry_mode must be one of: {allowed}")
        if self.strategy_type not in STRATEGY_TYPES:
            allowed = ", ".join(sorted(STRATEGY_TYPES))
            raise ValueError(f"strategy_type must be one of: {allowed}")
        if self.objective not in STRATEGY_OBJECTIVES:
            allowed = ", ".join(repr(item) for item in sorted(STRATEGY_OBJECTIVES))
            raise ValueError(f"objective must be one of: {allowed}")
        if self.max_bad_rate is not None:
            object.__setattr__(
                self,
                "max_bad_rate",
                _bounded_contract_number(
                    "max_bad_rate", self.max_bad_rate, minimum=0.0, maximum=1.0
                ),
            )
        if self.min_approval_rate is not None:
            object.__setattr__(
                self,
                "min_approval_rate",
                _bounded_contract_number(
                    "min_approval_rate",
                    self.min_approval_rate,
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        if self.baseline_strategy_id is not None:
            object.__setattr__(
                self,
                "baseline_strategy_id",
                _required_contract_string("baseline_strategy_id", self.baseline_strategy_id),
            )
        if self.profit is not None and not isinstance(self.profit, StrategyProfitInput):
            raise ValueError("profit must be a StrategyProfitInput or None")


def _required_contract_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _bounded_contract_number(
    name: str,
    value: object,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    if number < minimum or (maximum is not None and number > maximum):
        if maximum is None:
            raise ValueError(f"{name} must be greater than or equal to {minimum:g}")
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
    return number


@dataclass(frozen=True)
class TaskCreate:
    model_name: str
    model_version: str
    validator: str
    source_dir: str
    task_type: str = TASK_TYPE_VALIDATION
    algorithm: str = ""
    run_mode: str = "manual"
    # column mappings provided at task-creation time
    target_col: str = "y"
    score_col: str = "pred"
    split_col: str = "split"
    time_col: str = "apply_month"
    feature_columns: list[str] = field(default_factory=list)
    # Modeling recipes the user picked (manual mode multi-select); empty → the agent
    # recommends / modeling_setup defaults. Multi-element → multi-algorithm compare.
    target_type: str = ""
    recipes: list[str] = field(default_factory=list)
    sample_weight_col: str = ""
    # AGT-4 (optional, modeling tasks only): a user/AUTO-supplied minimum OOT KS the
    # final model must clear. None/absent → no success criterion is injected into the
    # plan (the pre-fix behavior); the platform never hard-codes a default threshold.
    oot_ks_min: float | None = None
    # Strategy-only governed business input. None is preserved for legacy tasks;
    # an explicit StrategyTaskInput may itself contain unanswered fields so setup
    # can pause for clarification without inventing business defaults.
    strategy_input: StrategyTaskInput | None = None
    # None denotes an omitted/legacy metric contract; [] is an explicit choice
    # to calculate no optional metrics (FEATURE §2: 选了才算).
    metrics: list[str] | None = None
    # Per-task capability tier (conservative/balanced/aggressive) — controls only the
    # autonomy budget (max_replan_iterations), never effect/determinism/gates/safety.
    # Empty → the driver falls back to the global settings default.
    capability_tier: str = ""
    notebook_path: str | None = None
    sample_path: str | None = None
    pmml_path: str | None = None
    dictionary_path: str | None = None
    report_values: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskRecord:
    id: str
    model_name: str
    model_version: str
    validator: str
    source_dir: str
    algorithm: str
    run_mode: str
    target_col: str
    score_col: str
    split_col: str
    time_col: str
    feature_columns: list[str]
    notebook_path: str | None
    sample_path: str | None
    pmml_path: str | None
    dictionary_path: str | None
    report_values_revision: int
    status: TaskStatus
    status_message: str
    created_at: str
    updated_at: str
    status_reason_code: str = ""
    task_type: str = TASK_TYPE_VALIDATION
    target_type: str = ""
    recipes: list[str] = field(default_factory=list)
    sample_weight_col: str = ""
    oot_ks_min: float | None = None
    strategy_input: StrategyTaskInput | None = None
    metrics: list[str] | None = None
    capability_tier: str = ""
    validation_workflow_version: int = 0


@dataclass(frozen=True)
class FileArtifact:
    role: FileRole
    path: Path
    size_bytes: int
    sha256: str | None
    risk_notes: list[str] = field(default_factory=list)
