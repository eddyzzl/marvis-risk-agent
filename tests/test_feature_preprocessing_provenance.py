"""Registered transforms remain authentic through preparation and model fitting."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from marvis.feature.errors import FeatureError
from marvis.data.errors import DatasetContentDriftError
from marvis.feature.preprocessing import read_preprocessing_chain, sidecar_path
from marvis.data.preprocessing_evidence import (
    load_preprocessing_state,
    training_preprocessing_state,
)
from marvis.packs.modeling.prepare import prepare_modeling_frame
from marvis.repositories.modeling import ModelingRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.packs.feature.tools import tool_normalize
from marvis.plugins.manifest import ToolRef
from tests.test_feature_pack import _runtime


def scenario(tmp_path, *, full=False):
    runner, registry, _, backend = _runtime(tmp_path)
    frame = pd.DataFrame(
        {
            "x": [float(i % 5) for i in range(40)],
            "y": [i % 2 for i in range(40)],
            "split": ["train"] * 24 + ["test"] * 8 + ["oot"] * 8,
        }
    )
    source_path = tmp_path / "source.csv"
    frame.to_csv(source_path, index=False)
    source = registry.register_from_upload("task-feature", source_path, role="sample")
    result = runner.invoke(
        ToolRef("feature", "normalize"),
        {
            "dataset_id": source.id,
            "columns": ["x"],
            "method": "zscore",
            **({"allow_full_fit": True} if full else {"split_col": "split"}),
        },
        task_id="task-feature",
    )
    assert result.ok, result.error
    return (
        runner,
        registry,
        backend,
        source,
        registry.get(result.output["result_dataset_id"]),
    )


def test_real_transform_prepare_train_preserves_authenticated_parameters(tmp_path):
    runner, registry, backend, _, transformed = scenario(tmp_path)
    state = load_preprocessing_state(registry, transformed.id)
    assert state.assurance == "training_only" and state.artifact_id
    prepared = prepare_modeling_frame(
        registry,
        backend,
        transformed.id,
        target_col="y",
        feature_cols=["x"],
        split_col="split",
        split_config=None,
    )
    final = training_preprocessing_state(
        registry, prepared.id, split_col="split", train_values="train"
    )
    assert final.assurance == "training_only" and final.steps == state.steps
    trained = runner.invoke(
        ToolRef("modeling", "train_model"),
        {
            "dataset_id": prepared.id,
            "recipe": "lr",
            "features": ["x"],
            "target_col": "y",
            "split_col": "split",
            "split_values": {"train": "train", "test": "test", "oot": "oot"},
            "seed": 7,
        },
        task_id="task-feature",
    )
    assert trained.ok, trained.error
    artifact = ModelingRepository(registry._repo.db_path).get_model_artifact(
        trained.output["artifact_id"]
    )
    assert artifact.params["preprocessing_steps"] == final.steps
    assert artifact.params["preprocessing_assurance"] == "training_only"
    assert artifact.params["preprocessing_evidence"]["artifact_id"] == final.artifact_id


@pytest.mark.parametrize("what", ["sidecar", "missing_sidecar", "source", "output"])
def test_changed_evidence_cannot_be_read_as_an_empty_legacy_chain(tmp_path, what):
    _, registry, _, source, transformed = scenario(tmp_path)
    target = registry.resolve_path(source.id if what == "source" else transformed.id)
    if what in {"sidecar", "missing_sidecar"}:
        target = sidecar_path(target)
    if what == "missing_sidecar":
        target.unlink()
    else:
        target.write_bytes(b'{"preprocessing_steps": []}')
    with pytest.raises((FeatureError, DatasetContentDriftError, ValueError, OSError)):
        load_preprocessing_state(registry, transformed.id)


def test_new_split_cannot_put_previous_fit_members_into_evaluation(tmp_path):
    _, registry, backend, _, transformed = scenario(tmp_path)
    prepared = prepare_modeling_frame(
        registry,
        backend,
        transformed.id,
        target_col="y",
        feature_cols=["x"],
        split_col="split",
        split_config={"test_size": 0.3, "oot_size": 0.2},
        seed=21,
    )
    with pytest.raises(FeatureError, match="evaluation rows"):
        training_preprocessing_state(
            registry, prepared.id, split_col="split", train_values="train"
        )


def test_full_pool_exploration_cannot_be_claimed_as_independent_evaluation(tmp_path):
    _, registry, _, _, transformed = scenario(tmp_path, full=True)
    assert load_preprocessing_state(registry, transformed.id).assurance == "exploration"
    with pytest.raises(FeatureError, match="evaluation rows"):
        training_preprocessing_state(
            registry, transformed.id, split_col="split", train_values="train"
        )


@pytest.mark.parametrize(
    "tool", ["train_model", "train_models", "tune_hyperparameters"]
)
def test_training_tools_reject_preprocessing_fit_on_evaluation_members(tmp_path, tool):
    runner, _, _, _, transformed = scenario(tmp_path, full=True)
    inputs = {
        "dataset_id": transformed.id,
        "features": ["x"],
        "target_col": "y",
        "split_col": "split",
        "split_values": {"train": "train", "test": "test", "oot": "oot"},
        "seed": 7,
    }
    inputs.update({"recipes": ["lr"]} if tool == "train_models" else {"recipe": "lr"})
    result = runner.invoke(ToolRef("modeling", tool), inputs, task_id="task-feature")
    assert not result.ok
    assert "evaluation rows" in result.error


def test_registered_receipt_survives_dataset_pin_without_losing_parameters(tmp_path):
    _, registry, _, _, transformed = scenario(tmp_path)
    before = load_preprocessing_state(registry, transformed.id)
    registry.authenticate_dataset_binding(
        transformed.id,
        expected_task_id=transformed.task_id,
        expected_content_hash=transformed.content_hash,
    )
    after = load_preprocessing_state(registry, transformed.id)
    assert after == before


@pytest.mark.parametrize(
    "payload", ["broken", "{}", json.dumps({"preprocessing_steps": [1]})]
)
def test_present_malformed_legacy_sidecar_is_not_absent(tmp_path, payload):
    path = tmp_path / "sample.parquet"
    sidecar_path(path).write_text(payload)
    with pytest.raises(FeatureError):
        read_preprocessing_chain(path)


def test_repeated_transform_keeps_the_first_dataset_and_receipt_immutable(tmp_path):
    runner, registry, _, source, transformed = scenario(tmp_path)
    before = load_preprocessing_state(registry, transformed.id)
    result = runner.invoke(
        ToolRef("feature", "normalize"),
        {
            "dataset_id": source.id,
            "columns": ["x"],
            "method": "minmax",
            "split_col": "split",
        },
        task_id="task-feature",
    )
    assert result.ok, result.error
    assert result.output["result_dataset_id"] != transformed.id
    assert load_preprocessing_state(registry, transformed.id) == before


def test_receipt_registration_failure_rolls_back_dataset_and_parameter_files(
    tmp_path, monkeypatch
):
    _, registry, _, source, transformed = scenario(tmp_path)
    before = {dataset.id for dataset in registry.list_for_task(source.task_id)}
    files = set(registry.datasets_root.rglob("*"))

    def fail(*args, **kwargs):
        raise RuntimeError("receipt unavailable")

    monkeypatch.setattr(TaskArtifactRepository, "register_on_connection", fail)
    ctx = SimpleNamespace(
        task_id=source.task_id,
        seed=7,
        workspace=registry.datasets_root.parent,
        datasets_root=registry.datasets_root,
    )
    with pytest.raises(RuntimeError, match="receipt unavailable"):
        tool_normalize(
            {
                "dataset_id": source.id,
                "columns": ["x"],
                "method": "minmax",
                "split_col": "split",
            },
            ctx,
        )
    assert {dataset.id for dataset in registry.list_for_task(source.task_id)} == before
    assert not [
        path
        for path in registry.datasets_root.rglob("*")
        if path.is_file() and path not in files
    ]
    assert (
        load_preprocessing_state(registry, transformed.id).assurance == "training_only"
    )
