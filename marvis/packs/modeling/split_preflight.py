"""Read-only admission check for ``modeling.make_split`` (B-line extension).

``make_split`` declares ``write:dataset``. When it fails after dispatch the
executor cannot prove nothing was written, so the step becomes an unknown
effect that needs reconciliation, and no modeling verifier exists. A split
that would leave train, test or OOT empty is known before any write: the same
deterministic kernel the tool runs (``_make_split``) decides it from the split
columns alone. Running that kernel on an authenticated read of those columns
lets the executor reject the inputs before dispatch, so the step stays
retryable and the failure card can offer a split repair.

The check only rejects what the tool would reject; it never writes, never
decides success, and returns ``None`` whenever it cannot read the inputs.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import re

from marvis.modeling_split import SplitConfigurationError, validate_split_configuration
from marvis.packs.modeling.defaults import DEFAULT_RANDOM_SEED
from marvis.packs.modeling.errors import ModelingError


_EMPTY_SETS_RE = re.compile(r"切分后这些集合为空:\[(?P<sets>[^\]]*)\]")


def split_config_columns(config: dict) -> list[str]:
    """Columns the split kernel reads besides the target and features."""

    columns: list[str] = []
    if config.get("oot_by_time"):
        columns.append(str(config["oot_by_time"]))
    for column in config.get("group_cols") or []:
        columns.append(str(column))
    for rule in config.get("rules") or []:
        for condition in rule.get("when") or []:
            if isinstance(condition, dict) and condition.get("col") is not None:
                columns.append(str(condition["col"]))
    return list(dict.fromkeys(columns))


def dry_run_split(frame, split_config, seed: int):
    """Run the tool's split kernel on ``frame``; returns the split column."""

    from marvis.packs.modeling.prepare import SPLIT_COLUMN, _make_split

    return _make_split(frame, split_config, int(seed))[SPLIT_COLUMN]


def split_rejection(frame, split_config, seed: int) -> str | None:
    """The tool's own error for this split, when it is seed-independent.

    Only an empty train or OOT, or an invalid configuration, depends on the
    split columns alone. An empty test set can depend on the random draw for
    tiny samples, so that case is left to the tool.
    """

    try:
        dry_run_split(frame, split_config, seed)
    except ModelingError as exc:
        message = str(exc)
        match = _EMPTY_SETS_RE.search(message)
        if match is not None:
            empty = {item.strip(" '\"") for item in match.group("sets").split(",")}
            if not empty & {"train", "oot"}:
                return None
        return message
    return None


def make_split_preflight_check(*, db_path: Path, datasets_root: Path) -> Callable:
    """Build the executor admission check bound to this workspace."""

    def check(inputs: dict, *, task_id: str) -> str | None:
        if str(inputs.get("split_col") or "").strip():
            return None  # an existing split passes through; the tool checks it
        raw_config = inputs.get("split_config")
        try:
            config = validate_split_configuration(raw_config, split_col=None)
        except SplitConfigurationError as exc:
            return str(exc)
        if not config or config.get("method") == "existing":
            return None
        columns = split_config_columns(config)
        if not columns:
            return None
        from marvis.data.backend import DataBackend
        from marvis.data.registry import DatasetRegistry
        from marvis.repositories.datasets import DatasetRepository

        registry = DatasetRegistry(
            DatasetRepository(db_path), DataBackend(datasets_root), datasets_root
        )
        dataset = registry.get(str(inputs.get("dataset_id") or ""))
        if str(dataset.task_id) != str(task_id):
            return None
        available = [profile.name for profile in dataset.columns]
        try:
            validate_split_configuration(raw_config, columns=available, split_col=None)
        except SplitConfigurationError as exc:
            return str(exc)
        frame = registry.read_authenticated_parquet_snapshot(dataset.id, columns=columns)
        seed = inputs.get("seed")
        return split_rejection(
            frame, raw_config, DEFAULT_RANDOM_SEED if seed is None else int(seed)
        )

    return check


def builtin_preflight_checks(*, db_path: Path, datasets_root: Path) -> dict[str, Callable]:
    return {
        "modeling.make_split": make_split_preflight_check(
            db_path=db_path, datasets_root=datasets_root
        ),
    }


__all__ = [
    "builtin_preflight_checks",
    "dry_run_split",
    "make_split_preflight_check",
    "split_config_columns",
    "split_rejection",
]
