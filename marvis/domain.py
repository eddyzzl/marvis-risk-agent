from __future__ import annotations

from dataclasses import dataclass, field
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


class FileRole(StrEnum):
    NOTEBOOK = "notebook"
    SAMPLE = "sample"
    MODEL_PMML = "model_pmml"
    DATA_DICTIONARY = "data_dictionary"
    UNKNOWN = "unknown"


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
    # Optional feature metrics the user selected at creation (e.g. "vif"); empty → base
    # per-feature metrics only (spec §2: 选了才算). Only used for feature_analysis tasks.
    metrics: list[str] = field(default_factory=list)
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
    metrics: list[str] = field(default_factory=list)
    capability_tier: str = ""


@dataclass(frozen=True)
class FileArtifact:
    role: FileRole
    path: Path
    size_bytes: int
    sha256: str | None
    risk_notes: list[str] = field(default_factory=list)
