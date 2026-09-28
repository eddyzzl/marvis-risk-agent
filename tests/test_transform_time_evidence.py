"""Native cleaning consumes immutable timing ancestry, never a caller assertion."""

import pytest

from marvis.data.feature_time import feature_time_evidence
from marvis.data.workspace import DataSemanticMapping, DataWorkspaceDraft
from marvis.feature.errors import FeatureError
from marvis.packs.modeling.prepare import prepare_modeling_frame
from marvis.plugins.manifest import ToolRef
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from tests.test_data_ops_transform_dataset import _inputs
from tests.test_feature_time_evidence import scenario  # noqa: F401


def clean(environment, operations):
    runner, registry, _, source = environment
    workspaces = DataWorkspaceRepository(registry._repo.db_path)
    initial = workspaces.save(
        "task-feature",
        DataWorkspaceDraft(
            active_dataset_id=source.dataset.id,
            active_dataset_content_hash=source.dataset.content_hash,
        ),
        expected_revision=0,
    )
    workspace = workspaces.save(
        "task-feature",
        DataWorkspaceDraft(
            active_dataset_id=source.dataset.id,
            active_dataset_content_hash=source.dataset.content_hash,
            semantic_mapping=DataSemanticMapping(
                target_col="y", field_roles={"y": "target"}
            ),
        ),
        expected_revision=initial.revision,
    )
    result = runner.invoke(
        ToolRef("data_ops", "transform_dataset"),
        _inputs(source.dataset, workspace, operations),
        task_id="task-feature",
    )
    assert result.ok, result.error
    return result.output


def test_native_rename_filter_and_row_local_expression_keep_time_proof(scenario):  # noqa: F811
    _, registry, backend, source = scenario
    result = clean(
        scenario,
        [
            {"op": "rename_columns", "mapping": {"asof__amount": "amount_now"}},
            {
                "op": "derive_columns",
                "derivations": [
                    {
                        "name": "known_plus_one",
                        "to_type": "DOUBLE",
                        "expression": {
                            "op": "add",
                            "left": {"column": "amount_now"},
                            "right": {"literal": 1},
                        },
                    },
                    {
                        "name": "mixed",
                        "to_type": "DOUBLE",
                        "expression": {
                            "op": "add",
                            "left": {"column": "amount_now"},
                            "right": {"column": "anchor_payload"},
                        },
                    },
                ],
            },
            {
                "op": "filter_rows",
                "predicate": {
                    "op": "gte",
                    "left": {"column": "amount_now"},
                    "right": {"literal": 2},
                },
            },
        ],
    )
    output_id = result["result_dataset_id"]
    status = feature_time_evidence(registry, output_id, ["known_plus_one", "mixed"])
    assert status["fields"]["known_plus_one"]["assurance"] == "verified"
    assert status["fields"]["mixed"]["assurance"] == "unknown"
    assert set(status["artifact_ids"]) == {
        source.status.artifact_id,
        result["evidence_artifact_id"],
    }
    assert registry.get(output_id).row_count < source.dataset.row_count
    # Later projection keeps the certified ancestry after the active workspace changes.
    prepared = prepare_modeling_frame(
        registry,
        backend,
        output_id,
        target_col="y",
        feature_cols=["known_plus_one"],
        split_col="split",
        split_config=None,
    )
    assert (
        feature_time_evidence(registry, prepared.id, ["known_plus_one"])["assurance"]
        == "verified"
    )


def test_cleaning_fill_has_no_implicit_historical_fit_time(scenario):  # noqa: F811
    _, registry, _, _ = scenario
    output = clean(
        scenario,
        [
            {
                "op": "fill_missing",
                "fills": [{"column": "asof__amount", "method": "mean"}],
            }
        ],
    )
    status = feature_time_evidence(
        registry, output["result_dataset_id"], ["asof__amount", "asof__amount_woe"]
    )
    assert status["fields"]["asof__amount"]["assurance"] == "unknown"
    assert status["fields"]["asof__amount_woe"]["assurance"] == "verified"
    assert "fill_parameter_availability_not_recorded" in status["reasons"]


def test_cleaning_receipt_tamper_is_not_treated_as_absent_legacy_evidence(scenario):  # noqa: F811
    from pathlib import Path

    _, registry, _, _ = scenario
    result = clean(
        scenario, [{"op": "rename_columns", "mapping": {"asof__amount": "amount_now"}}]
    )
    artifact = TaskArtifactRepository(registry._repo.db_path).get_for_task(
        "task-feature", result["evidence_artifact_id"]
    )
    Path(artifact["path"]).write_text("{}")
    with pytest.raises(FeatureError, match="evidence content changed"):
        feature_time_evidence(registry, result["result_dataset_id"], ["amount_now"])


def test_unknown_source_stays_unknown_after_native_cleaning(scenario):  # noqa: F811
    _, registry, _, _ = scenario
    result = clean(
        scenario,
        [{"op": "rename_columns", "mapping": {"anchor_payload": "renamed_unknown"}}],
    )
    status = feature_time_evidence(
        registry, result["result_dataset_id"], ["renamed_unknown"]
    )
    assert status["assurance"] == "unknown"
