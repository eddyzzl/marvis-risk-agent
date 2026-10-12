from io import BytesIO
from types import SimpleNamespace

import pytest

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.modeling_holdout.service import HoldoutService
from marvis.packs.modeling._runtime import _Runtime
from marvis.packs.modeling.development_refit import development_refit
from marvis.packs.modeling.errors import ModelingError
from marvis.plugins.manifest import ToolRef
from marvis.repositories.datasets import DatasetRepository
from tests.test_modeling_holdout import (
    _bytes,
    _frame,
    _freeze,
    _seal,
    custody as custody,
)


def test_native_fold_lr_refit_freezes_before_exactly_one_oot_evaluation(
    custody, monkeypatch
):
    app, clients, _, task = custody
    _seal(custody)
    reads = []
    original = HoldoutService._score_sealed

    def observe(self, cohort, closure, scorer):
        reads.append(cohort["id"])
        return original(self, cohort, closure, scorer)

    monkeypatch.setattr(HoldoutService, "_score_sealed", observe)
    response = clients["developer"].post(
        f"/api/tasks/{task}/blind-oot/development",
        files={
            "file": (
                "development.parquet",
                BytesIO(_bytes(_frame())),
                "application/octet-stream",
            )
        },
    )
    assert response.status_code == 201, response.text
    dataset = response.json()["dataset_id"]
    references = []
    for tool, controls in (
        ("screen_features", {"top_k": 2, "leakage_ks": 1.0}),
        ("select_features", {"iv_min": 0.0, "top_k": 2, "vif_max": 1e9}),
    ):
        result = app.state.tool_runner.invoke(
            ToolRef("modeling", tool),
            {
                "dataset_id": dataset,
                "features": ["x", "z"],
                "target_col": "y",
                "split_col": "split",
                "target_type": "binary",
                **controls,
            },
            task_id=task,
        )
        assert result.ok, result.error
        references.append(result.output["selection_evidence_ref"])
    inputs = {
        "dataset_id": dataset,
        "features": ["x", "z"],
        "target_col": "y",
        "split_col": "split",
        "split_values": {"train": "train", "test": "test"},
        "seed": 19,
        "selection_evidence_refs": references,
    }
    tuned = app.state.tool_runner.invoke(
        ToolRef("modeling", "tune_hyperparameters"),
        {
            **inputs,
            "recipe": "lr",
            "n_trials": 1,
            "cv_folds": 3,
            "fold_selection": {"source_dataset_id": dataset, "candidates": ["x", "z"]},
        },
        task_id=task,
    )
    assert tuned.ok, tuned.error
    trained = app.state.tool_runner.invoke(
        ToolRef("modeling", "train_models"),
        {
            **inputs,
            "recipes": ["lr"],
            "params": tuned.output["best_params"],
            "features_by_recipe": tuned.output["features_by_recipe"],
            "fold_tuning_evidence_refs": tuned.output["fold_tuning_evidence_refs"],
        },
        task_id=task,
    )
    assert trained.ok, trained.error
    parent = trained.output["best_experiment_id"]
    selected = app.state.tool_runner.invoke(
        ToolRef("modeling", "select_experiment"),
        {
            "experiment_ids": [parent],
            "selected_experiment_id": parent,
            "refit_on_train_plus_test": True,
        },
        task_id=task,
    )
    assert selected.ok, selected.error
    assert (
        selected.output["refit"]["applied"]
        and selected.output["refit"]["fit_rows"] == 240
    )
    second = app.state.tool_runner.invoke(
        ToolRef("modeling", "train_model"),
        {
            **inputs,
            "recipe": "lr",
            "params": tuned.output["best_params"],
            "features": tuned.output["features_by_recipe"]["lr"],
            "fold_tuning_evidence_ref": tuned.output["fold_tuning_evidence_refs"]["lr"],
        },
        task_id=task,
    )
    assert second.ok, second.error
    assert reads == []
    frozen = _freeze(custody, selected.output)
    assert reads == []
    response = clients["evaluator"].post(
        f"/api/tasks/{task}/blind-oot/evaluate",
        json={"freeze_hash": frozen, "request_id": "first"},
    )
    assert response.status_code == 200, response.text
    again = clients["evaluator"].post(
        f"/api/tasks/{task}/blind-oot/evaluate",
        json={"freeze_hash": frozen, "request_id": "changed"},
    )
    assert again.status_code == 200 and again.json() == response.json()
    assert len(reads) == 1
    assert (
        clients["developer"].get(f"/api/tasks/{task}/blind-oot/result").status_code
        == 403
    )
    registry = DatasetRegistry(
        DatasetRepository(app.state.settings.db_path),
        DataBackend(app.state.settings.datasets_dir),
        app.state.settings.datasets_dir,
    )
    runtime = _Runtime(
        SimpleNamespace(
            workspace=app.state.settings.workspace, datasets_root=registry.datasets_root
        )
    )
    # A separately produced native candidate has no completed refit. Even a
    # direct internal invocation must refuse new fitting after final freeze.
    candidate = second.output["experiment_id"]
    runtime.experiments.set_status(candidate, "selected")
    from sklearn.pipeline import Pipeline

    monkeypatch.setattr(
        Pipeline, "fit", lambda *args, **kwargs: pytest.fail("frozen task refitted")
    )
    with pytest.raises(
        ModelingError, match="development_refit_after_final_freeze_forbidden"
    ):
        development_refit(
            runtime,
            experiment=runtime.experiments.get(candidate),
            recipe="lr",
            pre_refit_metrics={},
        )
    assert len(reads) == 1
