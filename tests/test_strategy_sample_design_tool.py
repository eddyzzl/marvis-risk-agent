from __future__ import annotations

import copy
import json
from pathlib import Path
import sqlite3

import pandas as pd
import pytest

from marvis.data.backend import DataBackend
from marvis.data.errors import NanLabelNotConfirmedError
from marvis.data.registry import DatasetRegistry
from marvis.data.workspace import (
    DataSemanticMapping,
    DataWorkspaceDraft,
    data_semantic_mapping_hash,
)
from marvis.db import DatasetRepository, TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.files import sha256_file
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.sample_design import (
    canonical_strategy_sample_design_bundle_json,
)
from marvis.packs.strategy.sample_design_tools import (
    SAMPLE_DESIGN_ARTIFACT_KIND,
    SAMPLE_DESIGN_ORIGIN_TOOL,
    SAMPLE_DESIGN_TOOL_SCHEMA_VERSION,
    load_strategy_sample_design_artifact,
    run_materialize_sample_design,
    validate_materialize_sample_design_tool_output,
)
import marvis.packs.strategy.sample_design_tools as sample_design_tools
from marvis.plugins.contracts import ToolContext
from marvis.plugins.loader import load_manifest
from marvis.plugins.schema_validation import validate_against_schema
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.settings import build_settings


def _context(settings, task_id: str) -> ToolContext:
    return ToolContext(
        task_id=task_id,
        seed=0,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )


def _setup(
    tmp_path: Path,
    *,
    numeric_split: bool = False,
    target_values: list[object] | None = None,
) -> dict:
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="sample-design",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "source"),
            task_type="strategy",
            target_col="bad",
        )
    )
    frame = pd.DataFrame(
        {
            "sample_split": (
                [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
                if numeric_split
                else ["dev", "dev", "valid", "valid", "oot", "oot"]
            ),
            "apply_month": [
                "202601",
                "202601",
                "202602",
                "202602",
                "202603",
                "202603",
            ],
            "weight": [1.0, 1.5, 1.0, 2.0, 1.0, 1.0],
            "loan_amount": [100.0, 200.0, 150.0, None, 300.0, 250.0],
            "overdue_amount": [0.0, 20.0, 0.0, 10.0, None, 30.0],
            "unused_feature": [11, 12, 13, 14, 15, 16],
            "bad": target_values
            if target_values is not None
            else [0, 1, 0, 1, None, 1],
        }
    )
    source = tmp_path / "sample.parquet"
    frame.to_parquet(source, index=False)
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    dataset = registry.register_existing(source, task_id=task.id, role="derived")
    workspaces = DataWorkspaceRepository(settings.db_path)
    activated = workspaces.save(
        task.id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
        ),
        expected_revision=0,
    )
    mapping = DataSemanticMapping(
        target_col="bad",
        field_roles={
            "sample_split": "segment",
            "apply_month": "month",
            "weight": "weight",
            "loan_amount": "loan_amount",
            "overdue_amount": "overdue_amount",
            "bad": "target",
        },
    )
    workspace = workspaces.save(
        task.id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
            semantic_mapping=mapping,
        ),
        expected_revision=activated.revision,
    )
    ctx = _context(settings, task.id)
    runtime = strategy_tools._runtime(ctx)
    request = {
        "dataset_id": dataset.id,
        "expected_dataset_content_hash": dataset.content_hash,
        "workspace_revision": workspace.revision,
        "workspace_generation": workspace.analysis_generation,
        "semantic_mapping_hash": data_semantic_mapping_hash(mapping),
        "target_col": "bad",
        "target_bad_value": 1,
        "performance_window_status": "provided",
        "performance_window_days": 30,
        "observation_window_status": "provided",
        "observation_window_start": "2026-01-01",
        "observation_window_end": "2026-03-31",
        "maturity_status": "confirmed_matured",
        "split_col": "sample_split",
        "development_values": [1] if numeric_split else ["dev"],
        "validation_values": [2] if numeric_split else ["valid"],
        "oot_values": [3] if numeric_split else ["oot"],
        "month_col": "apply_month",
        "weight_col": "weight",
        "loan_amount_col": "loan_amount",
        "overdue_amount_col": "overdue_amount",
        "drop_nan_labels": True,
    }
    return {
        "settings": settings,
        "task": task,
        "dataset": dataset,
        "workspace": workspace,
        "mapping": mapping,
        "ctx": ctx,
        "runtime": runtime,
        "request": request,
    }


def _sample_artifacts(fx: dict) -> list[dict]:
    return [
        item
        for item in TaskArtifactRepository(fx["settings"].db_path).list_for_task(
            fx["task"].id
        )
        if item["kind"] == SAMPLE_DESIGN_ARTIFACT_KIND
    ]


def test_materialize_sample_design_is_canonical_idempotent_and_strictly_loadable(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)

    first = run_materialize_sample_design(
        fx["request"], fx["ctx"], fx["runtime"]
    )
    second = strategy_tools.tool_materialize_sample_design(
        fx["request"], fx["ctx"]
    )

    assert first == second
    assert first["schema_version"] == SAMPLE_DESIGN_TOOL_SCHEMA_VERSION
    assert first["development"] is True
    assert first["unvalidated"] is True
    assert first["not_created_strategy"] is True
    assert first["not_adopted"] is True
    assert first["not_deployed"] is True
    assert validate_materialize_sample_design_tool_output(first) == first
    artifacts = _sample_artifacts(fx)
    assert len(artifacts) == 1
    record = artifacts[0]
    assert record["origin_tool"] == SAMPLE_DESIGN_ORIGIN_TOOL
    path = Path(record["path"])
    assert path.name == f"{first['sample_design_id']}.json"
    assert path.read_text("utf-8") == canonical_strategy_sample_design_bundle_json(
        first["bundle"]
    )
    assert sha256_file(path) == first["artifact"]["content_hash"]
    loaded = load_strategy_sample_design_artifact(
        fx["runtime"],
        task_id=fx["task"].id,
        artifact_id=first["artifact"]["artifact_id"],
        expected_artifact_content_hash=first["artifact"]["content_hash"],
        expected_sample_design_id=first["sample_design_id"],
        expected_sample_design_content_hash=first["content_hash"],
    )
    assert loaded.bundle == first["bundle"]
    assert fx["runtime"].strategies.list_for_task(fx["task"].id) == []


def test_materialize_sample_design_manifest_contract_and_conditional_inputs(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    manifest = load_manifest(
        Path(__file__).parents[1] / "marvis" / "packs" / "strategy",
        builtin=True,
    )
    tool = next(item for item in manifest.tools if item.name == "materialize_sample_design")
    output = run_materialize_sample_design(
        fx["request"], fx["ctx"], fx["runtime"]
    )

    validate_against_schema(
        fx["request"], tool.input_schema, label="sample-design input"
    )
    validate_against_schema(
        {**fx["request"], "target_bad_value": 1.0},
        tool.input_schema,
        label="sample-design JSON integral polarity input",
    )
    validate_against_schema(output, tool.output_schema, label="sample-design output")
    with pytest.raises(Exception):
        validate_against_schema(
            {
                key: value
                for key, value in fx["request"].items()
                if key != "performance_window_days"
            },
            tool.input_schema,
            label="sample-design invalid input",
        )
    for invalid_development_values in ([""], [2**53]):
        with pytest.raises(Exception):
            validate_against_schema(
                {
                    **fx["request"],
                    "development_values": invalid_development_values,
                },
                tool.input_schema,
                label="sample-design bounded split input",
            )
    raw_manifest = json.loads(
        (Path(__file__).parents[1] / "marvis" / "packs" / "strategy" / "manifest.json").read_text(
            "utf-8"
        )
    )
    raw_tool = next(
        item for item in raw_manifest["tools"] if item["name"] == "materialize_sample_design"
    )
    assert raw_tool["policy"] == {
        "schema_version": "tool-policy.v1",
        "human_decision_gate": "none",
        "effect_authorization": "none",
    }
    assert raw_tool["side_effects"] == [
        "read:task",
        "read:dataset",
        "write:artifact",
    ]


def test_materialize_sample_design_rejects_workspace_and_dataset_drift(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)

    with pytest.raises(StrategyError, match="DataWorkspace binding changed"):
        run_materialize_sample_design(
            {**fx["request"], "workspace_revision": 999},
            fx["ctx"],
            fx["runtime"],
        )

    registered = Path(fx["runtime"].registry.resolve_path(fx["dataset"].id))
    registered.write_bytes(registered.read_bytes() + b"drift")
    with pytest.raises(StrategyError, match="drifted|changed"):
        run_materialize_sample_design(
            fx["request"], fx["ctx"], fx["runtime"]
        )
    assert _sample_artifacts(fx) == []


def test_materialize_sample_design_requires_columns_and_nan_confirmation(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)

    with pytest.raises(StrategyError, match="missing columns: unknown_month"):
        run_materialize_sample_design(
            {**fx["request"], "month_col": "unknown_month"},
            fx["ctx"],
            fx["runtime"],
        )
    with pytest.raises(NanLabelNotConfirmedError) as exc_info:
        run_materialize_sample_design(
            {**fx["request"], "drop_nan_labels": False},
            fx["ctx"],
            fx["runtime"],
        )
    assert exc_info.value.to_detail()["kind"] == "nan_label_not_confirmed"
    assert _sample_artifacts(fx) == []


def test_materialize_sample_design_reads_only_bound_columns(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fx = _setup(tmp_path)
    original_read = fx["runtime"].backend.read_frame
    projections: list[list[str] | None] = []

    def tracked_read(path, columns=None):
        projections.append(columns)
        return original_read(path, columns=columns)

    monkeypatch.setattr(fx["runtime"].backend, "read_frame", tracked_read)

    run_materialize_sample_design(fx["request"], fx["ctx"], fx["runtime"])

    assert len(projections) == 1
    assert projections[0] is not None
    assert set(projections[0]) == {
        "sample_split",
        "apply_month",
        "weight",
        "loan_amount",
        "overdue_amount",
        "bad",
    }
    assert "unused_feature" not in projections[0]


def test_materialize_sample_design_rechecks_workspace_under_writer_lock(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fx = _setup(tmp_path)
    original_persist = sample_design_tools._persist_bundle

    def drift_then_persist(*args, **kwargs):
        with sqlite3.connect(fx["settings"].db_path) as conn:
            conn.execute(
                "UPDATE data_workspaces SET revision = revision + 1 WHERE task_id = ?",
                (fx["task"].id,),
            )
        return original_persist(*args, **kwargs)

    monkeypatch.setattr(sample_design_tools, "_persist_bundle", drift_then_persist)

    with pytest.raises(StrategyError, match="DataWorkspace changed.*registration"):
        run_materialize_sample_design(
            fx["request"], fx["ctx"], fx["runtime"]
        )
    assert _sample_artifacts(fx) == []
    output_dir = (
        fx["settings"].tasks_dir
        / fx["task"].id
        / "strategy_sample_designs"
    )
    assert not list(output_dir.glob("*.json"))


def test_materialize_sample_design_rejects_partial_or_overlapping_split_inputs(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    partial = {
        key: value
        for key, value in fx["request"].items()
        if key not in {"validation_values", "oot_values"}
    }
    with pytest.raises(StrategyError, match="must be supplied together"):
        run_materialize_sample_design(partial, fx["ctx"], fx["runtime"])
    with pytest.raises(StrategyError, match="appears in both"):
        run_materialize_sample_design(
            {
                **fx["request"],
                "validation_values": ["dev"],
            },
            fx["ctx"],
            fx["runtime"],
        )
    assert _sample_artifacts(fx) == []


def test_materialize_sample_design_matches_integer_request_to_float_split_column(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path, numeric_split=True)

    output = run_materialize_sample_design(
        fx["request"], fx["ctx"], fx["runtime"]
    )

    split = output["bundle"]["sample_design"]["split_definition"]
    assert split["development_values"] == [1]
    assert split["validation_values"] == [2]
    assert split["oot_values"] == [3]

    with pytest.raises(StrategyError, match="appears in both"):
        run_materialize_sample_design(
            {
                **fx["request"],
                "validation_values": [1.0],
            },
            fx["ctx"],
            fx["runtime"],
        )


def test_materialize_sample_design_replay_normalizes_split_value_order(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path, numeric_split=True)
    first_request = {
        **fx["request"],
        "development_values": [1, -1, 0.5],
    }
    second_request = {
        **fx["request"],
        "development_values": [0.5, 1, -1],
    }

    first = run_materialize_sample_design(first_request, fx["ctx"], fx["runtime"])
    second = run_materialize_sample_design(second_request, fx["ctx"], fx["runtime"])

    assert first == second
    assert len(_sample_artifacts(fx)) == 1
    loaded = load_strategy_sample_design_artifact(
        fx["runtime"],
        task_id=fx["task"].id,
        artifact_id=first["artifact"]["artifact_id"],
        expected_artifact_content_hash=first["artifact"]["content_hash"],
        expected_sample_design_id=first["sample_design_id"],
        expected_sample_design_content_hash=first["content_hash"],
    )
    assert loaded.bundle == first["bundle"]


def test_materialize_sample_design_recovers_exact_promoted_file_without_registry_row(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    first = run_materialize_sample_design(fx["request"], fx["ctx"], fx["runtime"])
    path = Path(_sample_artifacts(fx)[0]["path"])
    original = path.read_bytes()
    with sqlite3.connect(fx["settings"].db_path) as conn:
        conn.execute(
            "DELETE FROM task_artifacts WHERE task_id = ? AND id = ?",
            (fx["task"].id, first["artifact"]["artifact_id"]),
        )

    recovered = run_materialize_sample_design(
        fx["request"], fx["ctx"], fx["runtime"]
    )

    assert recovered["bundle"] == first["bundle"]
    assert path.read_bytes() == original
    assert len(_sample_artifacts(fx)) == 1


def test_materialize_sample_design_replay_ignores_mutable_registry_role(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    first = run_materialize_sample_design(fx["request"], fx["ctx"], fx["runtime"])

    fx["runtime"].registry.set_role(fx["dataset"].id, "feature")
    second = run_materialize_sample_design(
        fx["request"], fx["ctx"], fx["runtime"]
    )

    assert second == first
    assert len(_sample_artifacts(fx)) == 1


@pytest.mark.parametrize(
    "target_values",
    [
        ["0", "1", "0", "1", None, "1"],
        [0, 1, 0, 2, None, 1],
        [0, 1, 0, float("inf"), None, 1],
    ],
)
def test_materialize_sample_design_rejects_non_numeric_or_nonbinary_targets_before_confirmation(
    tmp_path: Path,
    target_values: list[object],
) -> None:
    fx = _setup(tmp_path, target_values=target_values)

    with pytest.raises(StrategyError, match="numeric 0/1"):
        run_materialize_sample_design(fx["request"], fx["ctx"], fx["runtime"])
    assert _sample_artifacts(fx) == []


@pytest.mark.parametrize("invalid", [True, False, -1, 2, 0.5, float("nan"), "1"])
def test_materialize_sample_design_requires_explicit_integer_target_polarity(
    tmp_path: Path,
    invalid: object,
) -> None:
    fx = _setup(tmp_path)
    with pytest.raises(StrategyError, match="target_bad_value"):
        run_materialize_sample_design(
            {**fx["request"], "target_bad_value": invalid},
            fx["ctx"],
            fx["runtime"],
        )


def test_materialize_sample_design_canonicalizes_json_integral_target_polarity(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    float_output = run_materialize_sample_design(
        {**fx["request"], "target_bad_value": 1.0},
        fx["ctx"],
        fx["runtime"],
    )
    integer_output = run_materialize_sample_design(
        fx["request"], fx["ctx"], fx["runtime"]
    )
    assert float_output == integer_output


def test_materialize_sample_design_rejects_oversized_split_controls(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    with pytest.raises(StrategyError, match="item budget"):
        run_materialize_sample_design(
            {
                **fx["request"],
                "development_values": [f"dev-{index}" for index in range(101)],
            },
            fx["ctx"],
            fx["runtime"],
        )
    with pytest.raises(StrategyError, match="string length budget"):
        run_materialize_sample_design(
            {**fx["request"], "development_values": ["x" * 257]},
            fx["ctx"],
            fx["runtime"],
        )


@pytest.mark.parametrize("tamper", ["file", "registry"])
def test_materialize_sample_design_replay_fails_closed_after_artifact_tamper(
    tmp_path: Path,
    tamper: str,
) -> None:
    fx = _setup(tmp_path)
    output = run_materialize_sample_design(
        fx["request"], fx["ctx"], fx["runtime"]
    )
    record = _sample_artifacts(fx)[0]
    if tamper == "file":
        Path(record["path"]).write_text("{}", "utf-8")
        expected = "content hash|bytes changed"
    else:
        with sqlite3.connect(fx["settings"].db_path) as conn:
            conn.execute("DROP TRIGGER trg_task_artifacts_immutable_update")
            conn.execute(
                "UPDATE task_artifacts SET content_hash = ? WHERE id = ?",
                ("f" * 64, output["artifact"]["artifact_id"]),
            )
        expected = "registry row changed"

    with pytest.raises(StrategyError, match=expected):
        run_materialize_sample_design(
            fx["request"], fx["ctx"], fx["runtime"]
        )
    assert len(_sample_artifacts(fx)) == 1
    assert fx["runtime"].strategies.list_for_task(fx["task"].id) == []


def test_materialize_sample_design_cached_output_tamper_fails_closed(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    output = run_materialize_sample_design(
        fx["request"], fx["ctx"], fx["runtime"]
    )
    tampered = copy.deepcopy(output)
    tampered["bundle"]["sample_design"]["active_dataset_boundary"][
        "population_count"
    ] = 999999

    with pytest.raises(StrategyError, match="content|drifted"):
        validate_materialize_sample_design_tool_output(tampered)


def test_sample_design_provenance_distinguishes_boolean_from_integer_polarity(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    output = run_materialize_sample_design(
        fx["request"], fx["ctx"], fx["runtime"]
    )
    provenance = copy.deepcopy(_sample_artifacts(fx)[0]["provenance"])
    provenance["request"]["target_bad_value"] = True
    provenance["request_hash"] = sample_design_tools.hashlib.sha256(
        sample_design_tools._canonical_json(provenance["request"]).encode("utf-8")
    ).hexdigest()

    with pytest.raises(StrategyError, match="request does not match"):
        sample_design_tools._require_bundle_provenance(output["bundle"], provenance)
