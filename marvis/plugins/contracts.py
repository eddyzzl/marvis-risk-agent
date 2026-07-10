from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from marvis.safe_paths import assert_within

# ARCH-5: host<->worker subprocess protocol version. Bump whenever the job
# dict schema, the result protocol shape, or guard semantics (network/file/
# process guards, error_kind taxonomy, resource_limits fields) change in a
# way that an old worker paired with a new host (or vice versa) could not
# safely interpret. Lives here rather than in runner.py/subprocess_worker.py
# so both sides of the boundary import the same leaf module with zero
# internal marvis dependencies beyond safe_paths (PERF-5: worker entrypoint
# import must stay dependency-free).
# v2 frames the authoritative result as a sentinel-prefixed JSON line so
# native-library stdout cannot corrupt the protocol payload.
PROTOCOL_VERSION = 2
WORKER_RESULT_SENTINEL = "@@MARVIS_PLUGIN_RESULT@@"


@dataclass(frozen=True)
class ToolContext:
    task_id: str
    seed: int | None
    datasets_root: Path
    workspace: Path

    def load_dataset_path(self, dataset_id: str) -> Path:
        return assert_within(self.datasets_root, self.datasets_root / dataset_id)
