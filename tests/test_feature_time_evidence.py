"""Temporal claims stay bound to native fields through actual preparation/training."""

import json
from pathlib import Path

import pandas as pd
import pytest

from marvis.data.asof_join import AsOfJoinEngine
from marvis.data.feature_time import feature_time_evidence
from marvis.data.time_contracts import DatasetTimeContract
from marvis.packs.modeling.prepare import prepare_modeling_frame
from marvis.plugins.manifest import ToolRef
from marvis.repositories.modeling import ModelingRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from tests.test_data_time_contracts import contracts, spec
from tests.test_feature_pack import _runtime


@pytest.fixture
def scenario(tmp_path):
    runner, registry, repo, backend = _runtime(tmp_path)
    decision = pd.DataFrame(
        {
            "id": [f"d{i}" for i in range(40)],
            "subject": [f"s{i}" for i in range(40)],
            "decision": ["2026-01-10T00:00:00Z"] * 40,
            "y": [i % 2 for i in range(40)],
            "split": ["train"] * 24 + ["test"] * 8 + ["oot"] * 8,
            "anchor_payload": [i % 2 for i in range(40)],
        }
    )
    features = pd.DataFrame(
        {
            "id": [f"f{i}" for i in range(40)],
            "subject": decision.subject,
            "event": ["2026-01-09T00:00:00Z"] * 40,
            "available": ["2026-01-09T01:00:00Z"] * 40,
            "version": [1] * 40,
            "amount": [float(i % 7) for i in range(40)],
            "amount_woe": [float(100 + i) for i in range(40)],
        }
    )
    sources = []
    for name, frame in (("decisions", decision), ("features", features)):
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
    dc, fc = [
        DatasetTimeContract.model_validate(
            {
                **contract.model_dump(),
                "dataset_id": dataset.id,
                "content_hash": dataset.content_hash,
            }
        )
        for contract, dataset in zip(contracts(), sources, strict=True)
    ]
    engine = AsOfJoinEngine(
        registry,
        TaskArtifactRepository(repo.db_path),
        workspace_root=registry.datasets_root.parent,
    )
    result = engine.execute(
        task_id="task-feature",
        decision_contract=dc,
        feature_contract=fc,
        spec=spec(feature_columns=("amount", "amount_woe")),
    )
    return runner, registry, backend, result


def test_only_native_selected_fields_are_certified_and_anchor_payload_stays_unknown(
    scenario,
):
    _, registry, _, joined = scenario
    evidence = feature_time_evidence(
        registry, joined.dataset.id, ["asof__amount", "anchor_payload"]
    )
    assert evidence["assurance"] == "unknown"
    assert evidence["fields"]["asof__amount"]["assurance"] == "verified"
    assert evidence["fields"]["anchor_payload"]["assurance"] == "unknown"
    assert evidence["artifact_ids"] == [joined.status.artifact_id]
    assert "source_authenticity" in evidence["excluded_assurances"]
    assert "s0" not in json.dumps(evidence)


def test_actual_prepare_train_keeps_empty_preprocessing_temporal_chain(scenario):
    runner, registry, backend, joined = scenario
    prepared = prepare_modeling_frame(
        registry,
        backend,
        joined.dataset.id,
        target_col="y",
        feature_cols=["asof__amount"],
        split_col="split",
        split_config=None,
    )
    evidence = feature_time_evidence(registry, prepared.id, ["asof__amount"])
    assert evidence["assurance"] == "verified"
    assert joined.status.artifact_id in evidence["artifact_ids"]
    assert len(evidence["artifact_ids"]) == 2
    trained = runner.invoke(
        ToolRef("modeling", "train_model"),
        {
            "dataset_id": prepared.id,
            "recipe": "lr",
            "features": ["asof__amount"],
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
    assert artifact.params["feature_time_evidence"] == evidence
    assert artifact.params["preprocessing_chain_traceable"] is False
    delivered = runner.invoke(
        ToolRef("modeling", "post_training_action"),
        {
            "experiment_id": trained.output["experiment_id"],
            "sample_dataset_id": prepared.id,
            "actions": ["export_pmml"],
        },
        task_id="task-feature",
    )
    assert delivered.ok, delivered.error
    card = json.loads(Path(delivered.output["model_card_path"]).read_text())
    assert card["training"]["feature_time_evidence"] == evidence
    assert not any("历史可得时间" in item for item in card["limitations"])


def test_train_only_fit_is_not_a_recorded_historical_parameter_cutoff(scenario):
    runner, registry, _, joined = scenario
    result = runner.invoke(
        ToolRef("feature", "normalize"),
        {
            "dataset_id": joined.dataset.id,
            "columns": ["asof__amount"],
            "method": "zscore",
            "split_col": "split",
        },
        task_id="task-feature",
    )
    assert result.ok, result.error
    evidence = feature_time_evidence(
        registry, result.output["result_dataset_id"], ["asof__amount"]
    )
    assert evidence["assurance"] == "unknown"
    assert evidence["reasons"] == ["fitted_parameter_availability_not_recorded"]
    assert joined.status.artifact_id in evidence["artifact_ids"]


def test_row_local_derived_feature_inherits_only_its_authenticated_inputs(scenario):
    runner, registry, _, joined = scenario
    result = runner.invoke(
        ToolRef("feature", "cross_features"),
        {
            "dataset_id": joined.dataset.id,
            "recipe": [{"kind": "transform", "col": "asof__amount", "ops": ["log1p"]}],
        },
        task_id="task-feature",
    )
    assert result.ok, result.error
    new_id = result.output["result_dataset_id"]
    evidence = feature_time_evidence(registry, new_id, ["asof__amount__log1p"])
    assert evidence["assurance"] == "verified"


def test_upstream_temporal_evidence_tampering_is_not_downgraded_to_legacy_unknown(
    scenario,
):
    _, registry, backend, joined = scenario
    prepared = prepare_modeling_frame(
        registry,
        backend,
        joined.dataset.id,
        target_col="y",
        feature_cols=["asof__amount"],
        split_col="split",
        split_config=None,
    )
    joined.evidence_path.write_text("{}")
    with pytest.raises(ValueError, match="evidence content changed"):
        feature_time_evidence(registry, prepared.id, ["asof__amount"])


def test_reordered_projection_cannot_reuse_original_row_time_certificate(
    scenario, monkeypatch
):
    _, registry, backend, joined = scenario
    project = backend.project_columns_to_parquet

    def reordered(source, output, columns):
        project(source, output, columns)
        pd.read_parquet(output).iloc[::-1].to_parquet(output, index=False)

    monkeypatch.setattr(backend, "project_columns_to_parquet", reordered)
    prepared = prepare_modeling_frame(
        registry,
        backend,
        joined.dataset.id,
        target_col="y",
        feature_cols=["asof__amount"],
        split_col="split",
        split_config=None,
    )
    evidence = feature_time_evidence(registry, prepared.id, ["asof__amount"])
    assert evidence["assurance"] == "unknown"
    assert evidence["reasons"] == ["projection_changed_feature_values_or_membership"]


def test_legacy_source_never_receives_temporal_assurance(scenario):
    _, registry, _, joined = scenario
    evidence = json.loads(joined.evidence_path.read_text())
    source_id = evidence["feature_contract"]["dataset_id"]
    status = feature_time_evidence(registry, source_id, ["amount"])
    assert status["assurance"] == "unknown" and status["artifact_ids"] == []


@pytest.mark.parametrize("tool", ["woe_encode", "woe_encode_categorical"])
def test_fitted_woe_overwrite_does_not_keep_the_old_native_column_certificate(
    scenario, tool
):
    runner, registry, _, joined = scenario
    before = feature_time_evidence(registry, joined.dataset.id, ["asof__amount_woe"])
    assert before["assurance"] == "verified"
    result = runner.invoke(
        ToolRef("feature", tool),
        {
            "dataset_id": joined.dataset.id,
            "features": ["asof__amount"],
            "target_col": "y",
            "split_col": "split",
            **(
                {"method": "equal_frequency", "max_bins": 2}
                if tool == "woe_encode"
                else {}
            ),
        },
        task_id="task-feature",
    )
    assert result.ok, result.error
    output_id = result.output["result_dataset_id"]
    original = registry.read_authenticated_parquet_snapshot(
        joined.dataset.id, columns=["asof__amount_woe"]
    )
    changed = registry.read_authenticated_parquet_snapshot(
        output_id, columns=["asof__amount_woe"]
    )
    assert not original.equals(changed)
    after = feature_time_evidence(registry, output_id, ["asof__amount_woe"])
    assert after["assurance"] == "unknown"
    assert after["reasons"] == ["fitted_parameter_availability_not_recorded"]
