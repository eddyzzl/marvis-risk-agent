from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from marvis.safe_paths import assert_within


@dataclass(frozen=True)
class ToolContext:
    task_id: str
    seed: int | None
    datasets_root: Path
    workspace: Path

    def load_dataset_path(self, dataset_id: str) -> Path:
        return assert_within(self.datasets_root, self.datasets_root / dataset_id)
