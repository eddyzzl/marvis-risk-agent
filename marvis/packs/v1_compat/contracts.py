from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from marvis.db import TaskRepository
from marvis.domain import TaskRecord
from marvis.pipeline import PipelineSettings
from marvis.settings import Settings


@dataclass(frozen=True)
class V1TaskContext:
    task_id: str
    workspace: Path
    settings: Settings
    pipeline_settings: PipelineSettings
    repo: TaskRepository
    task: TaskRecord
    task_dir: Path
    execution_dir: Path
    outputs_dir: Path
    images_dir: Path

