"""Historical input visibility is independent of row-local PIT and fit membership."""

import pandas as pd
import pytest

from marvis.data.asof_join import AsOfJoinEngine
from marvis.data.time_contracts import DatasetTimeContract
from marvis.plugins.manifest import ToolRef
from marvis.repositories.modeling import ModelingRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from tests.test_data_time_contracts import contracts, spec
from tests.test_feature_pack import _runtime


def scenario(tmp_path, *, future=True, available=True):
    runner, registry, repo, _ = _runtime(tmp_path)
    decisions = pd.DataFrame(
        {
            "id": [f"d{i}" for i in range(40)],
            "subject": [f"s{i}" for i in range(40)],
            "decision": ["2026-01-20T00:00:00Z"] * 24 + ["2026-01-10T00:00:00Z"] * 16,
            "y": [i % 2 for i in range(40)],
            "x": [float(i % 7) for i in range(40)],
            "split": ["train"] * 24 + ["test"] * 8 + ["oot"] * 8,
        }
    )
    features = pd.DataFrame(
        {
            "id": [f"f{i}" for i in range(40)],
            "subject": decisions.subject,
            "event": ["2026-01-19T00:00:00Z" if future else "2026-01-08T00:00:00Z"] * 24
            + ["2026-01-09T00:00:00Z"] * 16,
            "version": [1] * 40,
            "amount": [100.0 + i % 5 for i in range(40)],
        }
    )
    features["available"] = features.event
    sources = []
    for name, frame in (("decisions", decisions), ("features", features)):
        path = registry.datasets_root / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        with repo.transaction() as conn:
            sources.append(
                registry.register_existing_on_connection(
                    conn,
                    path,
                    task_id="task-feature",
                    role=name,
                    target_col_override="y" if name == "decisions" else None,
                )
            )
    original = contracts() if available else contracts(available_at=None)
    cs = [
        DatasetTimeContract.model_validate(
            {
                **contract.model_dump(),
                "dataset_id": dataset.id,
                "content_hash": dataset.content_hash,
            }
        )
        for contract, dataset in zip(original, sources, strict=True)
    ]
    joined = AsOfJoinEngine(
        registry,
        TaskArtifactRepository(repo.db_path),
        workspace_root=registry.datasets_root.parent,
    ).execute(
        task_id="task-feature",
        decision_contract=cs[0],
        feature_contract=cs[1],
        spec=spec(mode="verified" if available else "exploration"),
    )
    return runner, registry, repo, joined, sources


def model_inputs(dataset_id, **overrides):
    return {
        "dataset_id": dataset_id,
        "recipe": "lr",
        "features": ["asof__amount"],
        "target_col": "y",
        "split_col": "split",
        "split_values": {"train": "train", "test": "test", "oot": "oot"},
        "seed": 7,
        **overrides,
    }


@pytest.mark.parametrize(
    ("tool", "params"),
    [
        ("normalize", {"columns": ["asof__amount"], "method": "zscore"}),
        ("impute_missing", {"columns": ["asof__amount"], "strategy": "median"}),
        ("cap_outliers", {"columns": ["asof__amount"], "method": "iqr"}),
        ("onehot_encode", {"columns": ["asof__amount"]}),
        ("woe_encode", {"features": ["asof__amount"], "target_col": "y"}),
    ],
)
def test_preprocessing_rejects_authenticated_future_fit_inputs(tmp_path, tool, params):
    runner, _, _, joined, _ = scenario(tmp_path)
    result = runner.invoke(
        ToolRef("feature", tool),
        {
            "dataset_id": joined.dataset.id,
            "split_col": "split",
            **params,
        },
        task_id="task-feature",
    )
    assert not result.ok
    assert "fitted input is available after evaluation decision" in str(result.error)


@pytest.mark.parametrize("tool", ["train_model", "tune_hyperparameters"])
def test_training_internal_fit_rejects_authenticated_future_inputs(tmp_path, tool):
    runner, _, _, joined, _ = scenario(tmp_path)
    result = runner.invoke(
        ToolRef("modeling", tool),
        model_inputs(
            joined.dataset.id,
            **({"n_trials": 1} if tool == "tune_hyperparameters" else {}),
        ),
        task_id="task-feature",
    )
    assert not result.ok
    assert "fitted input is available after evaluation decision" in str(result.error)


def test_past_feature_inputs_do_not_prove_label_or_parameter_history(tmp_path):
    runner, _, repo, joined, _ = scenario(tmp_path, future=False)
    result = runner.invoke(
        ToolRef("modeling", "train_model"),
        model_inputs(joined.dataset.id),
        task_id="task-feature",
    )
    assert result.ok, result.error
    artifact = ModelingRepository(repo.db_path).get_model_artifact(
        result.output["artifact_id"]
    )
    evidence = artifact.params["fitted_input_time_evidence"]
    assert evidence["assurance"] == "unknown"
    assert evidence["feature_input_assurance"] == "verified"
    assert "label_available_at_not_recorded" in evidence["reasons"]


def test_rank_of_a_row_local_descendant_rejects_known_future_input(tmp_path):
    runner, _, _, joined, _ = scenario(tmp_path)
    result = runner.invoke(
        ToolRef("feature", "cross_features"),
        {
            "dataset_id": joined.dataset.id,
            "recipe": [
                {"kind": "cross", "a": "asof__amount", "b": "x", "ops": ["add"]},
                {
                    "kind": "transform",
                    "col": "asof__amount_add_x",
                    "ops": ["rank"],
                    "split_col": "split",
                },
            ],
        },
        task_id="task-feature",
    )
    assert not result.ok
    assert "fitted input is available after evaluation decision" in str(result.error)


@pytest.mark.parametrize(
    ("available", "features"), [(False, ["asof__amount"]), (True, ["x"])]
)
def test_unknown_or_unused_timing_does_not_become_a_verified_training_claim(
    tmp_path, available, features
):
    runner, _, repo, joined, _ = scenario(tmp_path, available=available)
    result = runner.invoke(
        ToolRef("modeling", "train_model"),
        model_inputs(joined.dataset.id, features=features),
        task_id="task-feature",
    )
    assert result.ok, result.error
    artifact = ModelingRepository(repo.db_path).get_model_artifact(
        result.output["artifact_id"]
    )
    evidence = artifact.params["fitted_input_time_evidence"]
    assert evidence["assurance"] == evidence["feature_input_assurance"] == "unknown"
    assert evidence["latest_known_fit_input_at"] is None


def test_native_normalize_chain_retains_input_visibility_without_parameter_history(
    tmp_path,
):
    runner, _, repo, joined, _ = scenario(tmp_path, future=False)
    normalized = runner.invoke(
        ToolRef("feature", "normalize"),
        {
            "dataset_id": joined.dataset.id,
            "columns": ["asof__amount"],
            "method": "zscore",
            "split_col": "split",
        },
        task_id="task-feature",
    )
    assert normalized.ok, normalized.error
    result = runner.invoke(
        ToolRef("modeling", "train_model"),
        model_inputs(normalized.output["result_dataset_id"]),
        task_id="task-feature",
    )
    assert result.ok, result.error
    artifact = ModelingRepository(repo.db_path).get_model_artifact(
        result.output["artifact_id"]
    )
    evidence = artifact.params["fitted_input_time_evidence"]
    assert evidence["feature_input_assurance"] == "verified"
    assert evidence["assurance"] == "unknown"
    assert len(evidence["artifact_ids"]) == 2
    assert "historical_parameter_existence" in evidence["excluded_assurances"]


@pytest.mark.parametrize("split_config", [{}, {"test_size": 0.2, "oot_size": 0.2}])
def test_native_preparation_does_not_erase_known_future_fit_inputs(
    tmp_path, split_config
):
    runner, _, _, joined, _ = scenario(tmp_path)
    prepared = runner.invoke(
        ToolRef("modeling", "prepare_modeling_frame"),
        {
            "dataset_id": joined.dataset.id,
            "feature_cols": ["asof__amount"],
            "target_col": "y",
            "split_col": "split",
            "split_config": split_config,
            "seed": 7,
        },
        task_id="task-feature",
    )
    assert prepared.ok, prepared.error
    result = runner.invoke(
        ToolRef("modeling", "train_model"),
        model_inputs(prepared.output["result_dataset_id"]),
        task_id="task-feature",
    )
    assert not result.ok
    assert "fitted input is available after evaluation decision" in str(result.error)


def test_later_source_bytes_change_is_an_error_not_unknown(tmp_path):
    runner, registry, _, joined, sources = scenario(tmp_path, future=False)
    source = registry.resolve_path(sources[1].id)
    frame = pd.read_parquet(source)
    frame["available"] = "2026-01-19T00:00:00Z"
    source.chmod(0o600)
    frame.to_parquet(source, index=False)
    result = runner.invoke(
        ToolRef("modeling", "train_model"),
        model_inputs(joined.dataset.id),
        task_id="task-feature",
    )
    assert not result.ok
    assert "changed" in str(result.error) or "hash" in str(result.error)


def test_previously_registered_future_normalization_is_rechecked_at_training(
    tmp_path, monkeypatch
):
    from marvis.packs.feature import tools as feature_tools
    from marvis.plugins.contracts import ToolContext

    runner, registry, _, joined, _ = scenario(tmp_path)
    # Reproduce an already-issued artifact from the previous producer version;
    # the consumer must not trust its training-only membership as a time proof.
    with monkeypatch.context() as patch:
        patch.setattr(
            feature_tools, "_check_fit_input_time", lambda *args, **kwargs: None
        )
        normalized = feature_tools.tool_normalize(
            {
                "dataset_id": joined.dataset.id,
                "columns": ["asof__amount"],
                "method": "zscore",
                "split_col": "split",
            },
            ToolContext(
                task_id="task-feature",
                seed=7,
                datasets_root=registry.datasets_root,
                workspace=registry.datasets_root.parent,
            ),
        )
    result = runner.invoke(
        ToolRef("modeling", "train_model"),
        model_inputs(normalized["result_dataset_id"]),
        task_id="task-feature",
    )
    assert not result.ok
    assert "fitted input is available after evaluation decision" in str(result.error)


def test_refit_records_only_formal_oot_as_its_independent_time_window(tmp_path):
    import base64
    import numpy as np
    from marvis.packs.modeling.contracts import TrainConfig
    from marvis.packs.modeling.preprocessing_validation import (
        validate_model_fitted_inputs,
    )

    _, registry, _, joined, _ = scenario(tmp_path, future=False)
    frame = registry.read_authenticated_parquet_snapshot(joined.dataset.id)
    frame.loc[frame.split == "test", "split"] = "train"
    frame.loc[0, "split"] = "refit_holdout"
    config = TrainConfig(
        dataset_id=joined.dataset.id,
        features=("asof__amount",),
        target_col="y",
        split_col="split",
        split_values={"train": "train", "test": "refit_holdout", "oot": "oot"},
        params={},
        seed=7,
        early_stopping_rounds=None,
        recipe_id="lr",
    )
    evidence = validate_model_fitted_inputs(
        registry, config, frame=frame, include_test=False
    )
    assert evidence["fit_rows"] == 31
    assert evidence["evaluation_rows"] == 8
    for key, expected in (
        ("fit", list(range(1, 32))),
        ("evaluation", list(range(32, 40))),
    ):
        bitmap = np.unpackbits(
            np.frombuffer(
                base64.b64decode(evidence[f"{key}_membership"]), dtype=np.uint8
            ),
            bitorder="little",
            count=evidence["row_count"],
        )
        assert np.flatnonzero(bitmap).tolist() == expected
    assert evidence["evaluation_roles"] == ["oot"]
    assert evidence["excluded_evaluation_roles"] == [
        "refit_holdout_non_independent_diagnostic"
    ]
    assert evidence["assurance"] == "unknown"
    frame.loc[frame.split == "oot", "split"] = "unused"
    evidence = validate_model_fitted_inputs(
        registry, config, frame=frame, include_test=False
    )
    assert evidence["evaluation_roles"] == []
    assert evidence["evaluation_rows"] == 0
    assert evidence["feature_input_assurance"] == "unknown"
