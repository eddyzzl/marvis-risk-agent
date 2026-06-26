"""Per-workspace memory policy settings.

Governs two automatic agent-memory behaviors that are otherwise unconditional:

- ``reference_cross_task``: whether prior-task agent memory is injected into the
  prompt context for the current turn.
- ``auto_distill``: whether user messages are automatically captured as memory
  candidates on every chat turn.

Both default to ``True`` so that a fresh workspace (no settings file) behaves
identically to the historical, unconditional behavior. Mirrors the per-feature
JSON settings store template in ``marvis.execution_environment``.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path


SETTINGS_FILE = "memory_policy.json"


@dataclass(frozen=True)
class MemoryPolicySettings:
    reference_cross_task: bool = True
    auto_distill: bool = True


def load_memory_policy(workspace: Path) -> MemoryPolicySettings:
    """Return the persisted policy, or defaults if the file is missing/corrupt.

    Never raises on read: a missing file, empty file, or garbage JSON all yield
    the defaults (both flags on) so agent behavior is never accidentally changed
    by an unreadable settings file.
    """
    path = _memory_policy_path(workspace)
    if not path.exists():
        return MemoryPolicySettings()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return MemoryPolicySettings()
    if not isinstance(payload, dict):
        return MemoryPolicySettings()
    return MemoryPolicySettings(
        reference_cross_task=_coerce_bool(
            payload.get("reference_cross_task"), default=True
        ),
        auto_distill=_coerce_bool(payload.get("auto_distill"), default=True),
    )


def save_memory_policy(
    workspace: Path,
    settings: MemoryPolicySettings,
) -> MemoryPolicySettings:
    path = _memory_policy_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: write to a sibling temp file then replace, so a crash mid-write
    # can never leave a half-written (corrupt) settings file behind.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(asdict(settings), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)
    return settings


def _coerce_bool(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _memory_policy_path(workspace: Path) -> Path:
    return Path(workspace) / "settings" / SETTINGS_FILE
