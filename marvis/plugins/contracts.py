from __future__ import annotations

from dataclasses import dataclass
import json
import os
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
# v2 framed the authoritative result as a sentinel-prefixed JSON line so
# native-library stdout cannot corrupt the protocol payload. v3 adds opaque,
# out-of-band effect execution metadata to ToolContext for governed writes.
PROTOCOL_VERSION = 3
WORKER_RESULT_SENTINEL = "@@MARVIS_PLUGIN_RESULT@@"
MAX_PROGRESS_BYTES = 64 * 1024


@dataclass(frozen=True)
class ToolContext:
    task_id: str
    seed: int | None
    datasets_root: Path
    workspace: Path
    effect_execution_id: str | None = None
    runtime_generation: str | None = None
    progress_path: Path | None = None

    def load_dataset_path(self, dataset_id: str) -> Path:
        return assert_within(self.datasets_root, self.datasets_root / dataset_id)

    def report_progress(self, payload: dict) -> bool:
        """Atomically publish best-effort worker progress.

        This observability channel is deliberately non-throwing: invalid JSON,
        I/O failures, or a disappearing host must never change tool results.
        Only host-issued paths inside the workspace are accepted.
        """

        if self.progress_path is None or not isinstance(payload, dict):
            return False
        temporary: Path | None = None
        try:
            target = assert_within(self.workspace, self.progress_path)
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            if len(encoded) > MAX_PROGRESS_BYTES:
                return False
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f"{target.name}.tmp")
            temporary.write_bytes(encoded)
            os.replace(temporary, target)
            return True
        except Exception:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except Exception:
                    pass
            return False
