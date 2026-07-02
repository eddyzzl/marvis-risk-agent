from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from marvis.packs.modeling import tools as modeling_tools
from marvis.packs.modeling import train_tools as modeling_train_tools
from marvis.packs.modeling.contracts import ModelArtifact, ModelMetrics, TrainConfig, TrainResult
from marvis.packs.modeling.report_compute import BusinessColumns
from marvis.packs.modeling.training_dataset import TrainingDataset


class _CountingBackend:
    def __init__(self, frame):
        self.frame = frame
        self.read_count = 0

    def read_frame(self, path: Path, *, columns=None, nrows=None):
        self.read_count += 1
        frame = self.frame
        if columns is not None:
            frame = frame[list(columns)]
        if nrows is not None:
            frame = frame.head(nrows)
        return frame.copy()

    def column_names(self, path: Path):
        return [str(column) for column in self.frame.columns]


class _FakeRegistry:
    def __init__(self, path: Path):
        self.path = path

    def get(self, dataset_id: str):
        return SimpleNamespace(id=dataset_id)

    def resolve_path(self, dataset_id: str):
        return self.path


class _FakeExperiments:
    def __init__(self):
        self._experiments = {}

    def create(self, task_id, recipe, config):
        experiment_id = f"exp-{recipe}"
        self._experiments[experiment_id] = SimpleNamespace(
            id=experiment_id,
            task_id=task_id,
            recipe_id=recipe,
            config=config,
            metrics=None,
            artifact_id=None,
            status="created",
        )
        return experiment_id

    def attach_result(self, experiment_id, result):
        exp = self._experiments[experiment_id]
        exp.metrics = result.metrics
        exp.artifact_id = result.artifact.id

    def set_status(self, experiment_id, status):
        self._experiments[experiment_id].status = status

    def get(self, experiment_id):
        return self._experiments[experiment_id]


def _metrics(oot_ks: float) -> ModelMetrics:
    return ModelMetrics(
        train_ks=oot_ks,
        test_ks=oot_ks,
        oot_ks=oot_ks,
        train_auc=0.7,
        test_auc=0.7,
        oot_auc=0.7,
        psi_test_vs_train=None,
        psi_oot_vs_train=None,
        overfit_train_test_gap=0.0,
        overfit_train_oot_gap=0.0,
        overfit_flag=False,
    )


def _report_config() -> TrainConfig:
    return TrainConfig(
        dataset_id="dataset-1",
        features=("x1",),
        target_col="y",
        split_col="split",
        split_values={"train": "train", "test": "test", "oot": "oot"},
        params={},
        seed=7,
        early_stopping_rounds=None,
    )


def test_train_models_uses_training_dataset_cache_for_multiple_recipes(tmp_path, monkeypatch):
    frame = pd.DataFrame({
        "x1": [0.1, 0.2, 0.3],
        "y": [0, 1, 0],
        "split": ["train", "test", "oot"],
    })
    backend = _CountingBackend(frame)
    dataset_path = tmp_path / "modeling.parquet"
    experiments = _FakeExperiments()
    runtime = SimpleNamespace(
        registry=_FakeRegistry(dataset_path),
        backend=backend,
        experiments=experiments,
        settings=SimpleNamespace(tasks_dir=tmp_path / "tasks"),
    )

    def fake_runtime(_ctx):
        return runtime

    def fake_train_recipe(recipe, recipe_backend, path, config, *, out_dir):
        # Every recipe still calls read_frame through the historical backend API.
        # The adapter should serve these calls from the already-loaded frame.
        assert recipe_backend.read_frame(path).equals(frame)
        return TrainResult(
            artifact=ModelArtifact(
                id=f"artifact-{recipe}",
                experiment_id="",
                algorithm=recipe,
                model_path=f"{recipe}.pkl",
                pmml_path=None,
                feature_list=("x1",),
                params={},
                woe_maps=None,
                created_at="2026-06-29T00:00:00+00:00",
            ),
            metrics=_metrics(0.31 if recipe == "lgb" else 0.29),
            feature_importance=(("x1", 1.0),),
            experiment_id="",
        )

    monkeypatch.setattr(modeling_train_tools, "_runtime", fake_runtime)
    monkeypatch.setattr(modeling_train_tools, "_train_recipe", fake_train_recipe)

    out = modeling_tools.tool_train_models(
        {
            "dataset_id": "dataset-1",
            "recipes": ["lgb", "lr"],
            "features": ["x1"],
            "target_col": "y",
            "split_col": "split",
            "split_values": {"train": "train", "test": "test", "oot": "oot"},
            "params": {},
            "seed": 7,
        },
        SimpleNamespace(task_id="task-1"),
    )

    assert backend.read_count == 1
    assert out["experiment_ids"] == ["exp-lgb", "exp-lr"]
    assert out["best_experiment_id"] == "exp-lgb"


def test_cached_report_runtime_uses_existing_scored_frame_without_backend_read(tmp_path):
    frame = pd.DataFrame({
        "score": [0.10, 0.20, 0.30, 0.60, 0.80, 0.90],
        "y": [0, 0, 1, 0, 1, 1],
        "split": ["train", "train", "test", "test", "oot", "oot"],
        "amount": [100, 120, 140, 160, 180, 200],
    })
    backend = _CountingBackend(frame)
    dataset_path = tmp_path / "scored.parquet"
    runtime = SimpleNamespace(backend=backend)

    cached_runtime = modeling_tools._cached_dataset_runtime(runtime, dataset_path, frame=frame)

    assert cached_runtime.backend.column_names(dataset_path) == ["score", "y", "split", "amount"]
    score_bands = modeling_tools._score_band_rows(
        cached_runtime,
        dataset_path,
        score_col="score",
        target_col="y",
        config=_report_config(),
        bin_count=2,
    )
    oot_bins = modeling_tools._report_bin_table(
        cached_runtime,
        dataset_path,
        score_col="score",
        target_col="y",
        config=_report_config(),
        business=BusinessColumns(loan_amount_col="amount"),
    )

    assert backend.read_count == 0
    assert {row["split"] for row in score_bands} == {"train", "test", "oot"}
    assert oot_bins


def test_cached_report_runtime_loads_dataset_once_when_frame_not_supplied(tmp_path):
    frame = pd.DataFrame({
        "score": [0.10, 0.20, 0.80, 0.90],
        "y": [0, 0, 1, 1],
        "split": ["train", "test", "oot", "oot"],
    })
    backend = _CountingBackend(frame)
    dataset_path = tmp_path / "scored.parquet"
    runtime = SimpleNamespace(backend=backend)

    cached_runtime = modeling_tools._cached_dataset_runtime(runtime, dataset_path)
    assert cached_runtime.backend.read_frame(dataset_path, columns=["score"]).shape == (4, 1)
    assert cached_runtime.backend.read_frame(dataset_path, columns=["y", "split"]).shape == (4, 2)

    assert backend.read_count == 1


def test_training_dataset_backend_does_not_copy_full_frame_per_read(monkeypatch):
    """PERF-10: an N-recipe train_models run reads the shared modeling frame once
    per recipe through the adapter. Each full-frame read must NOT deep-copy the
    (potentially very wide) frame, so the resident-memory peak stays flat in the
    recipe count instead of scaling linearly with it.

    Proxy 1 (copy-call count): count every ``DataFrame.copy(deep=True)`` triggered
    while serving the reads -- it must stay at zero however many recipes read the
    frame. Proxy 2 (memory sharing): every returned full frame must share its
    column blocks with the cached frame, i.e. it is a view, not a materialised
    duplicate.
    """
    import numpy as np

    frame = pd.DataFrame({f"f{i}": np.arange(50, dtype=float) + i for i in range(20)})
    frame["split"] = (["train"] * 20) + (["test"] * 15) + (["oot"] * 15)
    dataset_path = Path("/cached/modeling.parquet")

    class _NoFallback:
        def read_frame(self, *a, **k):
            raise AssertionError("fallback backend must not be hit for the cached path")

        def column_names(self, *a, **k):
            raise AssertionError("fallback backend must not be hit for the cached path")

    training_dataset = TrainingDataset(path=dataset_path, frame=frame)
    adapter = training_dataset.backend_adapter(_NoFallback())

    real_copy = pd.DataFrame.copy
    deep_copies = {"n": 0}

    def counting_copy(self, deep=True):
        if deep:
            deep_copies["n"] += 1
        return real_copy(self, deep=deep)

    monkeypatch.setattr(pd.DataFrame, "copy", counting_copy)

    # Simulate ten recipes each pulling the full frame (recipes call read_frame
    # with no `columns`, exactly like train_lgb / train_xgb / ...).
    recipe_count = 10
    returned = [adapter.read_frame(dataset_path) for _ in range(recipe_count)]

    # No deep copy of the full frame, regardless of how many recipes read it.
    assert deep_copies["n"] == 0
    # Every read is a distinct object that still shares the cached blocks (a view,
    # not a duplicate) -- memory footprint is flat in the recipe count.
    for got in returned:
        assert got is not training_dataset.frame
        assert got.equals(frame)
        assert np.shares_memory(got["f0"].to_numpy(), training_dataset.frame["f0"].to_numpy())


def test_training_dataset_backend_read_is_isolated_and_stable():
    """PERF-10 safety/determinism lock: dropping the eager per-read deep copy must
    not weaken isolation or change the data each recipe sees. A returned frame is
    a Copy-on-Write view, so writing into it copies-on-write instead of corrupting
    the shared cache, and every recipe still reads byte-identical values."""
    frame = pd.DataFrame(
        {
            "x1": [0.1, 0.2, 0.3, 0.4],
            "x2": [1.0, 2.0, 3.0, 4.0],
            "y": [0, 1, 0, 1],
            "split": ["train", "train", "test", "oot"],
        }
    )
    dataset_path = Path("/cached/modeling.parquet")

    class _NoFallback:
        def read_frame(self, *a, **k):
            raise AssertionError("fallback backend must not be hit for the cached path")

        def column_names(self, *a, **k):
            raise AssertionError("fallback backend must not be hit for the cached path")

    training_dataset = TrainingDataset(path=dataset_path, frame=frame)
    adapter = training_dataset.backend_adapter(_NoFallback())

    first = adapter.read_frame(dataset_path)
    # A consumer that writes into its returned frame must not corrupt the cache.
    first.loc[0, "x1"] = 999.0
    assert training_dataset.frame.loc[0, "x1"] == 0.1

    # Every subsequent recipe still reads the original, byte-identical values and
    # the same column order -- training semantics/feature order are untouched.
    second = adapter.read_frame(dataset_path)
    assert list(second.columns) == ["x1", "x2", "y", "split"]
    assert second.equals(frame)

    # Column-projected reads are likewise isolated from the cache.
    subset = adapter.read_frame(dataset_path, columns=["x2", "split"])
    subset.loc[0, "x2"] = -1.0
    assert training_dataset.frame.loc[0, "x2"] == 1.0
    assert adapter.read_frame(dataset_path, columns=["x2"]).equals(frame[["x2"]])
