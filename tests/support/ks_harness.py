"""T4-2: end-to-end KS-baseline harness core.

A parametrized runner that drives the modeling stack (ingest → split → train →
KS) on ANY tabular dataset and compares the resulting KS to a stored ground-truth
baseline. Two layers share this core:

- ``scripts/ks_baseline.py`` — the CLI a user points at a real public dataset
  (GiveMeSomeCredit / Home Credit) they have dropped into a convention directory.
- ``tests/test_ks_baseline_smoke.py`` — a CI smoke anchor that runs the SAME
  runner on a small, deterministic, signal-bearing synthetic dataset so the
  harness itself is proven end-to-end without any external data.

The runner is intentionally thin and dependency-light: it takes a DataFrame (or a
CSV/parquet path), a feature list, a target column, and a split spec, and returns
a :class:`KSRunResult`. It reuses the platform's own ``DataBackend`` +
``train_<recipe>`` recipes (the SAME code paths the agent uses), so a KS the
harness reports is a KS the product can actually produce.

Baseline comparison contract (see docs/ks_baseline/README.md):

    PASS  when   agent_ks >= baseline_ks - tolerance      (default tolerance 0.005)

The baseline value is a one-off human-tuned KS captured as ground truth in a JSON
file; the runner's KS must not regress below it by more than the tolerance.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from marvis.data.backend import DataBackend
from marvis.packs.modeling.contracts import TrainConfig

#: The default KS tolerance below the baseline that still counts as a pass
#: (docs/plans/v2-trust-first-plan.md §0: "agent KS >= 人工精调基线 − 0.005").
DEFAULT_KS_TOLERANCE = 0.005

#: Recipe entry points the harness can drive. Maps recipe id -> importable trainer.
_RECIPES = {
    "lr": "marvis.packs.modeling.recipes.lr:train_lr",
    "lgb": "marvis.packs.modeling.recipes.lgb:train_lgb",
    "scorecard": "marvis.packs.modeling.recipes.scorecard:train_scorecard",
}


@dataclass(frozen=True)
class SplitSpec:
    """How to carve train/test/oot.

    Either point at an existing ``split_col`` with a ``split_values`` mapping, OR
    ask the harness to derive a deterministic positional split (60/20/20) by
    leaving ``split_col`` None.
    """

    split_col: str | None = None
    split_values: dict[str, Any] | None = None
    seed: int = 20260705


@dataclass(frozen=True)
class KSRunResult:
    dataset: str
    recipe: str
    n_rows: int
    n_features: int
    train_ks: float | None
    test_ks: float | None
    oot_ks: float | None
    test_auc: float | None
    nan_labels_dropped: int

    def to_dict(self) -> dict:
        return asdict(self)


def _load_trainer(recipe: str):
    if recipe not in _RECIPES:
        raise ValueError(f"unknown recipe {recipe!r}; known: {sorted(_RECIPES)}")
    module_path, func_name = _RECIPES[recipe].split(":")
    module = __import__(module_path, fromlist=[func_name])
    return getattr(module, func_name)


def _positional_split(n: int) -> np.ndarray:
    """Deterministic 60/20/20 train/test/oot split by row position (no RNG)."""
    idx = np.arange(n)
    split = np.empty(n, dtype=object)
    split[idx % 5 < 3] = "train"
    split[idx % 5 == 3] = "test"
    split[idx % 5 == 4] = "oot"
    return split


def run_ks(
    data: pd.DataFrame | str | Path,
    *,
    features: list[str],
    target_col: str,
    dataset_name: str,
    recipe: str = "lr",
    split: SplitSpec | None = None,
    workdir: Path,
    drop_nan_labels: bool = False,
    params: dict[str, Any] | None = None,
) -> KSRunResult:
    """Ingest ``data``, train ``recipe``, and return its KS.

    ``data`` may be a DataFrame or a path to a CSV/parquet the harness reads. The
    frame is materialized to parquet under ``workdir`` (the same on-disk contract
    the platform's ``DataBackend`` reads), then trained through the real recipe.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    frame = _read_frame(data)
    split = split or SplitSpec()

    if split.split_col is None:
        frame = frame.copy()
        frame["__split__"] = _positional_split(len(frame))
        split_col = "__split__"
        split_values = {"train": "train", "test": "test", "oot": "oot"}
    else:
        split_col = split.split_col
        if not split.split_values:
            raise ValueError("split_values is required when split_col is set")
        split_values = split.split_values

    dataset_path = workdir / f"{dataset_name}.parquet"
    frame.to_parquet(dataset_path, index=False)

    backend = DataBackend(workdir)
    config = TrainConfig(
        dataset_id=dataset_name,
        features=tuple(features),
        target_col=target_col,
        split_col=split_col,
        split_values=split_values,
        params=dict(params or {}),
        seed=split.seed,
        early_stopping_rounds=None,
        recipe_id=recipe,
        drop_nan_labels=drop_nan_labels,
    )
    trainer = _load_trainer(recipe)
    result = trainer(backend, dataset_path, config, out_dir=workdir / "model")
    metrics = result.metrics
    return KSRunResult(
        dataset=dataset_name,
        recipe=recipe,
        n_rows=int(len(frame)),
        n_features=len(features),
        train_ks=_opt_float(metrics.train_ks),
        test_ks=_opt_float(metrics.test_ks),
        oot_ks=_opt_float(metrics.oot_ks),
        test_auc=_opt_float(metrics.test_auc),
        nan_labels_dropped=int(result.nan_labels_dropped),
    )


def _read_frame(data: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    path = Path(data)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"unsupported dataset file: {path}")


def _opt_float(value) -> float | None:
    return None if value is None else float(value)


# --------------------------------------------------------------------------- #
# baseline comparison                                                           #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BaselineVerdict:
    dataset: str
    baseline_ks: float
    agent_ks: float | None
    tolerance: float
    passed: bool
    margin: float | None  # agent_ks - (baseline_ks - tolerance); >= 0 is a pass

    def render(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        agent = "n/a" if self.agent_ks is None else f"{self.agent_ks:.4f}"
        return (
            f"[{status}] {self.dataset}: agent KS {agent} vs baseline "
            f"{self.baseline_ks:.4f} (floor {self.baseline_ks - self.tolerance:.4f}, "
            f"tolerance {self.tolerance})"
        )


def load_baselines(path: str | Path) -> dict[str, dict]:
    """Load the ground-truth baseline JSON: ``{dataset: {"baseline_ks": float, ...}}``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_to_baseline(
    result: KSRunResult,
    baseline_ks: float,
    *,
    tolerance: float = DEFAULT_KS_TOLERANCE,
    metric: str = "test_ks",
) -> BaselineVerdict:
    """Verdict: agent KS must be >= baseline_ks - tolerance to pass."""
    agent_ks = getattr(result, metric)
    floor = baseline_ks - tolerance
    passed = agent_ks is not None and agent_ks >= floor
    margin = None if agent_ks is None else agent_ks - floor
    return BaselineVerdict(
        dataset=result.dataset,
        baseline_ks=float(baseline_ks),
        agent_ks=_opt_float(agent_ks),
        tolerance=float(tolerance),
        passed=bool(passed),
        margin=None if margin is None else float(margin),
    )


__all__ = [
    "DEFAULT_KS_TOLERANCE",
    "BaselineVerdict",
    "KSRunResult",
    "SplitSpec",
    "compare_to_baseline",
    "load_baselines",
    "run_ks",
]
