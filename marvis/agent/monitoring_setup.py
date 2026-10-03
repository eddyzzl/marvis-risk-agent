"""Typed model monitoring intake into the existing governed plan lifecycle."""
from __future__ import annotations

from copy import deepcopy
from pydantic import BaseModel, ConfigDict, Field

from marvis.packs.modeling.errors import ModelingError
from marvis.packs.modeling.monitor_binding import capture_monitoring_binding
from marvis.packs.modeling.monitor_tools import MONITOR_RUN_THRESHOLDS
from marvis.repositories.data_workspace import DataWorkspaceRepository


class ModelMonitoringSetupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    experiment_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    expected_content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    workspace_revision: int = Field(ge=0)
    analysis_generation: int = Field(ge=0)
    target_col: str | None = Field(default=None, min_length=1)


def prepare_model_monitoring(settings, task_id, request):
    """Bind the displayed selection; neither labels nor defaults prove readiness."""
    from marvis.data.backend import DataBackend
    from marvis.data.registry import DatasetRegistry
    from marvis.repositories.datasets import DatasetRepository

    workspace = DataWorkspaceRepository(settings.db_path).get_or_default(task_id)
    expected = {
        "revision": request.workspace_revision,
        "analysis_generation": request.analysis_generation,
        "active_dataset_id": request.dataset_id,
        "active_dataset_content_hash": request.expected_content_hash,
    }
    if any(getattr(workspace, key) != value for key, value in expected.items()):
        raise ModelingError("监控数据选择已变化，请刷新后重新提交当前数据和模型。")
    binding = capture_monitoring_binding(
        settings, task_id, request.experiment_id, request.dataset_id,
        request.expected_content_hash,
    )
    binding["workspace_binding"] = expected
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path), DataBackend(settings.datasets_dir), settings.datasets_dir,
    )
    if request.target_col is not None:
        columns = registry.authenticated_parquet_column_names(
            request.dataset_id,
        )
        if request.target_col not in columns:
            raise ModelingError("声明的监控标签列不存在，请重新确认数据口径。")
    return {
        "request": request.model_dump(),
        "monitoring_binding": binding,
        "threshold_source": "platform_default_technical_thresholds",
        "thresholds": deepcopy(MONITOR_RUN_THRESHOLDS),
        "label_mode": "unlabeled_drift_only" if request.target_col is None else "declared_label_column",
        "label_maturity_assurance": "unknown",
        "business_acceptance": "not_established",
    }
