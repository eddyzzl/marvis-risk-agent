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
# Known task types. New capabilities (modeling/strategy/...) must register here
# so _normalize_task_type can keep arbitrary strings out of the database.
VALID_TASK_TYPES = frozenset({TASK_TYPE_VALIDATION})


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


@dataclass(frozen=True)
class FileArtifact:
    role: FileRole
    path: Path
    size_bytes: int
    sha256: str | None
    risk_notes: list[str] = field(default_factory=list)
