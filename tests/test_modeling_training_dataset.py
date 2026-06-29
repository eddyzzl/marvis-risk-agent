from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from marvis.packs.modeling import tools as modeling_tools
from marvis.packs.modeling.contracts import ModelArtifact, ModelMetrics, TrainResult


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

    monkeypatch.setattr(modeling_tools, "_runtime", fake_runtime)
    monkeypatch.setattr(modeling_tools, "_train_recipe", fake_train_recipe)

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
