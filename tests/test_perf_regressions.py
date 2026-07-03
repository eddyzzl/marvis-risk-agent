"""Performance-regression guardrails (LT-8).

The platform has been through two performance overhauls -- PERF-10 (per-recipe
frame CoW reuse in ``train_models``) and PERF-6 (polling-query convergence) --
plus TST-2 (streaming upload), all verified only by one-off manual checks at
the time. This module locks in deterministic, machine-independent PROXIES for
"did the hot path regress back to a full-frame multi-read" instead of relying
on raw wall-clock, which flakes on shared CI runners:

- Test 1 (large-parquet feature screening): counts ``DataBackend.read_frame``
  calls and asserts every call requests a small column batch, never the full
  80-column frame -- the deterministic signature of ``screen_features``'
  column-batched design (see ``marvis/feature/screen.py``).
- Test 2 (multi-recipe ``train_models``): counts ``DataBackend.read_frame``
  calls and asserts an N-recipe run triggers exactly ONE full dataset load
  (PERF-10's ``_TrainingDatasetBackend`` contract), not N.
- Test 3 (join match-rate smoke): counts ``DataBackend._connect`` calls (one
  DuckDB connection per query) and asserts ``diagnose_join`` for a composite
  key issues a small, FIXED number of queries regardless of feature-table row
  count -- plus that two independent runs (fresh, uncached ``DataBackend``
  instances) produce byte-identical ``match_rate``/``matched_rows``.

Every wall-clock assertion carries a generous (>=3x observed local runtime)
margin specifically to avoid flakiness on a shared/loaded machine; the
call-count proxies are the real regression guard.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from marvis.data.align import ColumnAligner
from marvis.data.backend import DataBackend
from marvis.data.contracts import Dataset, KeyPair
from marvis.data.join_engine import JoinEngine
from marvis.data.registry import DatasetRegistry
from marvis.data.schema_infer import infer_dataset_schema
from marvis.db import DatasetRepository, init_db
from marvis.feature.screen import screen_features
from marvis.packs.modeling import tools as modeling_tools
from marvis.settings import build_settings

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Test 1: large-parquet feature screening stays column-batched, never a
# full-frame multi-read.
# ---------------------------------------------------------------------------


def _synthetic_screen_frame(n_rows: int, n_cols: int, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {f"f{i}": rng.normal(size=n_rows) for i in range(n_cols)}
    # A real, moderately discriminative binary target (not pure noise) so KS/IV
    # computation exercises the real numeric code paths, not degenerate 0-KS cases.
    signal = data["f0"] * 0.6 + data["f1"] * 0.4 + rng.normal(scale=1.5, size=n_rows)
    data["y"] = (signal > np.median(signal)).astype(float)
    data["split"] = np.where(np.arange(n_rows) < int(n_rows * 0.85), "train", "oot")
    return pd.DataFrame(data)


def test_screen_features_on_large_parquet_never_reads_full_frame(tmp_path, monkeypatch):
    """LT-8 #1: screen_features on a 200k-row x 80-col parquet must serve every
    DataBackend.read_frame call from a bounded column batch (batch_size), never
    request all 80 feature columns (+target/split) in one call. That per-call
    column-count bound is the deterministic proxy for "no full-frame re-scan per
    feature" -- a wall-clock check alone would flake on a shared machine, so it
    is used only as a generous (3x observed local runtime) backstop below.
    """
    n_rows, n_cols = 200_000, 80
    frame = _synthetic_screen_frame(n_rows, n_cols, seed=42)
    path = tmp_path / "large_modeling.parquet"
    frame.to_parquet(path, index=False)  # generation/write time is not measured below

    backend = DataBackend(tmp_path)
    features = [f"f{i}" for i in range(n_cols)]
    batch_size = 20

    orig_read_frame = DataBackend.read_frame
    calls: list[int | None] = []

    def counting_read_frame(self, call_path, *, columns=None, nrows=None):
        calls.append(len(columns) if columns is not None else None)
        return orig_read_frame(self, call_path, columns=columns, nrows=nrows)

    monkeypatch.setattr(DataBackend, "read_frame", counting_read_frame)

    started = time.monotonic()
    result = screen_features(
        backend,
        path,
        features=features,
        target_col="y",
        split_col="split",
        holdout_values=("oot",),
        batch_size=batch_size,
        top_k=30,
    )
    elapsed = time.monotonic() - started

    # Proxy 1 (deterministic, machine-independent): NO call ever asked for more
    # than batch_size columns (the base target/split read asks for 2). If
    # screen_features regressed to reading the whole 80-column frame at once,
    # some call would report columns=80 (or None for an unbounded full read).
    assert calls, "screen_features made no read_frame calls at all"
    for n_columns in calls:
        assert n_columns is not None, "a read_frame call requested the full frame (columns=None)"
        assert n_columns <= batch_size, (
            f"read_frame call requested {n_columns} columns, exceeding batch_size={batch_size} "
            "-- screen_features regressed away from column-batched reads"
        )

    # Proxy 2 (deterministic call-count arithmetic): 1 base (target+split) read,
    # ceil(80/20)=4 ranking batches, ceil(30/20)=2 IV-enrichment batches for the
    # top_k=30 selected features -- 7 total, regardless of row count.
    assert len(calls) == 7

    assert result.n_screened == n_cols
    assert len(result.selected) == 30

    # Wall-clock backstop only (3x local-observed ~5s at this size on this
    # machine), never the primary guard -- generous margin to absorb a loaded
    # shared CI runner without flaking.
    assert elapsed < 30.0, f"screen_features took {elapsed:.1f}s, exceeding the 30s backstop"


# ---------------------------------------------------------------------------
# Test 2: multi-recipe train_models triggers exactly one full dataset load
# (PERF-10 counting-guard, exercised through REAL recipes on a REAL
# DataBackend/parquet file -- not the monkeypatched _train_recipe already
# covered by tests/test_modeling_training_dataset.py).
# ---------------------------------------------------------------------------


def _synthetic_training_frame(n_rows: int, n_features: int, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {f"x{i}": rng.normal(size=n_rows) for i in range(n_features)}
    signal = data["x0"] * 0.5 + data["x1"] * 0.3 + rng.normal(scale=2.0, size=n_rows)
    data["y"] = (signal > np.median(signal)).astype(int)
    n_train = int(n_rows * 0.6)
    n_test = int(n_rows * 0.25)
    n_oot = n_rows - n_train - n_test
    data["split"] = ["train"] * n_train + ["test"] * n_test + ["oot"] * n_oot
    return pd.DataFrame(data)


def test_train_models_multi_recipe_reads_dataset_exactly_once(tmp_path, monkeypatch):
    """LT-8 #2: an N-recipe tool_train_models run through the REAL DataBackend
    (real parquet file, real recipes -- lr + scorecard, chosen for speed) must
    trigger exactly ONE DataBackend.read_frame call for the shared modeling
    dataset (PERF-10's _TrainingDatasetBackend contract: TrainingDataset.load
    reads once, every recipe is served from the cached CoW frame). A regression
    back to "each recipe re-reads the dataset" would show read_count == len(recipes).

    This is a counting-guard angle distinct from
    test_modeling_training_dataset.test_train_models_uses_training_dataset_cache_for_multiple_recipes
    (which monkeypatches _train_recipe and never touches a real backend/DuckDB
    parquet file) and from
    test_training_dataset_backend_does_not_copy_full_frame_per_read (which
    exercises the adapter directly, not the tool_train_models entrypoint).
    """
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    data_repo = DatasetRepository(settings.db_path)
    backend = DataBackend(settings.datasets_dir)
    registry = DatasetRegistry(data_repo, backend, settings.datasets_dir)

    n_rows, n_features = 20_000, 15
    frame = _synthetic_training_frame(n_rows, n_features, seed=11)
    path = settings.datasets_dir / "modeling.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    dataset = registry.register_existing(path, task_id="task-1", role="modeling_sample")

    ctx = SimpleNamespace(
        workspace=settings.workspace,
        datasets_root=settings.datasets_dir,
        task_id="task-1",
        seed=7,
    )

    orig_read_frame = DataBackend.read_frame
    read_calls: list[Path] = []

    def counting_read_frame(self, call_path, *, columns=None, nrows=None):
        read_calls.append(Path(call_path))
        return orig_read_frame(self, call_path, columns=columns, nrows=nrows)

    # Only count reads that happen during the train_models call itself --
    # dataset registration above already did its own (unrelated) profiling reads.
    monkeypatch.setattr(DataBackend, "read_frame", counting_read_frame)

    recipes = ["lr", "scorecard"]
    started = time.monotonic()
    out = modeling_tools.tool_train_models(
        {
            "dataset_id": dataset.id,
            "recipes": recipes,
            "features": [f"x{i}" for i in range(n_features)],
            "target_col": "y",
            "split_col": "split",
            "split_values": {"train": "train", "test": "test", "oot": "oot"},
            "params": {},
            "seed": 7,
        },
        ctx,
    )
    elapsed = time.monotonic() - started

    # Deterministic counting-guard proxy: however many recipes ran, the
    # underlying dataset parquet is read exactly ONCE through DataBackend.
    dataset_reads = [call for call in read_calls if call == path]
    assert len(dataset_reads) == 1, (
        f"expected exactly 1 DataBackend.read_frame call for the shared modeling "
        f"dataset across {len(recipes)} recipes, got {len(dataset_reads)} -- PERF-10 regression"
    )

    assert len(out["experiment_ids"]) == len(recipes)
    assert out["best_experiment_id"] in out["experiment_ids"]

    # Wall-clock backstop only (3x local-observed ~8.5s at this size/recipe mix).
    assert elapsed < 30.0, f"train_models took {elapsed:.1f}s, exceeding the 30s backstop"


# ---------------------------------------------------------------------------
# Test 3: join match-rate smoke -- diagnose_join completes in a small, fixed
# number of DuckDB connections regardless of feature-table row count, and is
# deterministic across independent (uncached) runs.
# ---------------------------------------------------------------------------


def _dataset_from_frame(dataset_id: str, frame: pd.DataFrame, path: Path) -> Dataset:
    return Dataset(
        id=dataset_id,
        task_id="task-1",
        role="sample",
        source_path=path.name,
        format="parquet",
        sheet=None,
        row_count=len(frame),
        columns=tuple(infer_dataset_schema(frame)),
        has_target=False,
        target_col=None,
        created_at="2026-01-01T00:00:00Z",
    )


class _StaticJoinRegistry:
    def __init__(self, datasets: dict[str, Dataset], paths: dict[str, Path]):
        self._datasets = datasets
        self._paths = paths

    def get(self, dataset_id: str) -> Dataset:
        return self._datasets[dataset_id]

    def resolve_path(self, dataset_id: str) -> Path:
        return self._paths[dataset_id]

    def register_join_result_with_audit(self, *args, **kwargs):
        raise AssertionError("diagnose_join must not register a join result")


class _NullJoinRepo:
    """Minimal repo satisfying JoinEngine's constructor contract; diagnose_join
    (unlike propose/confirm/execute) never calls into the repo, so every method
    just needs to exist -- none is exercised in this smoke test."""

    def update_join_spec_with_audit(self, *args, **kwargs):
        raise AssertionError("diagnose_join must not write through the repo")

    def set_join_plan_executed_with_audit(self, *args, **kwargs):
        raise AssertionError("diagnose_join must not write through the repo")

    def write_audit(self, **kwargs):
        raise AssertionError("diagnose_join must not write through the repo")


def _synthetic_join_tables(n_rows: int, *, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ids = [f"A{i:08d}" for i in range(n_rows)]
    months = [f"{(i % 12) + 1:02d}" for i in range(n_rows)]
    anchor = pd.DataFrame({
        "acct_id": ids,
        "apply_date": [f"2026-{month}-01" for month in months],
        "score": rng.normal(size=n_rows),
    })
    feature = pd.DataFrame({
        "acct_id": ids,
        "biz_date": [f"2026{month}01" for month in months],
        "credit_limit": rng.integers(1000, 50000, size=n_rows),
    })
    return anchor, feature


def test_join_match_rate_smoke_bounded_queries_and_deterministic(tmp_path, monkeypatch):
    """LT-8 #3: diagnose_join on a 100k-row anchor x 100k-row feature table with a
    composite (acct_id, date) key must complete in a small, FIXED number of
    DuckDB connections (one query per bounded step: key validation, sample_rows,
    match-rate scan, is_key_unique) -- never scaling with feature-table row
    count -- and produce byte-identical match_rate/matched_rows across two
    independent (uncached, fresh DataBackend instance) runs.
    """
    n_rows = 100_000
    anchor_frame, feature_frame = _synthetic_join_tables(n_rows, seed=7)
    anchor_path = tmp_path / "anchor.parquet"
    feature_path = tmp_path / "feature.parquet"
    anchor_frame.to_parquet(anchor_path, index=False)
    feature_frame.to_parquet(feature_path, index=False)

    anchor_ds = _dataset_from_frame("anchor", anchor_frame, anchor_path)
    feature_ds = _dataset_from_frame("feature", feature_frame, feature_path)
    registry = _StaticJoinRegistry(
        {"anchor": anchor_ds, "feature": feature_ds},
        {"anchor": anchor_path, "feature": feature_path},
    )
    repo = _NullJoinRepo()

    key_pairs = [
        KeyPair("acct_id", "acct_id", "exact", "both", match_rate=1.0, resolved_by="test"),
        KeyPair("apply_date", "biz_date", "date", "both", match_rate=1.0, resolved_by="test"),
    ]

    orig_connect = DataBackend._connect
    connect_calls = {"n": 0}

    def counting_connect(self):
        connect_calls["n"] += 1
        return orig_connect(self)

    monkeypatch.setattr(DataBackend, "_connect", counting_connect)

    backend_1 = DataBackend(tmp_path)
    engine_1 = JoinEngine(backend_1, ColumnAligner(backend_1), registry, repo)

    started = time.monotonic()
    diagnostics_1 = engine_1.diagnose_join(
        anchor_ds, anchor_path, feature_ds, feature_path, key_pairs, seed=0,
    )
    elapsed = time.monotonic() - started

    # Proxy 1 (deterministic, machine-independent): a small, fixed number of
    # DuckDB connections -- a single-pass match-rate computation, not one query
    # per row/key. A regression to a per-row or per-key-candidate query pattern
    # would blow this bound up regardless of runtime.
    assert 0 < connect_calls["n"] <= 10, (
        f"diagnose_join issued {connect_calls['n']} DuckDB connections for a single "
        "composite-key match-rate check -- expected a small fixed count independent "
        "of the 100k-row feature table"
    )

    assert diagnostics_1.match_rate == 1.0
    assert diagnostics_1.matched_rows > 0
    assert diagnostics_1.feature_key_unique is True

    # Proxy 2 (determinism): a second, fully independent run (fresh DataBackend
    # instance -- no shared in-process cache) must reproduce the exact same
    # match_rate/matched_rows.
    backend_2 = DataBackend(tmp_path)
    engine_2 = JoinEngine(backend_2, ColumnAligner(backend_2), registry, repo)
    diagnostics_2 = engine_2.diagnose_join(
        anchor_ds, anchor_path, feature_ds, feature_path, key_pairs, seed=0,
    )
    assert diagnostics_2.match_rate == diagnostics_1.match_rate
    assert diagnostics_2.matched_rows == diagnostics_1.matched_rows

    # Wall-clock backstop only (3x local-observed ~0.08s at this size).
    assert elapsed < 5.0, f"diagnose_join took {elapsed:.2f}s, exceeding the 5s backstop"
