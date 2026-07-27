from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.data.workspace import (
    DataSemanticMapping,
    DataWorkspaceDraft,
    data_semantic_mapping_hash,
)
from marvis.db import DatasetRepository, PluginRepository, init_db
from marvis.db_schema import connect
from marvis.packs.data_ops import tools as data_ops_tools
from marvis.plugins.contracts import ToolContext
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.settings import build_settings


def _profile_tool_spec() -> dict:
    manifest_path = (
        Path(__file__).parents[1]
        / "marvis"
        / "packs"
        / "data_ops"
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return next(tool for tool in manifest["tools"] if tool["name"] == "profile_dataset")


def test_profile_dataset_manifest_is_strict_read_only_and_identity_bound():
    tool = _profile_tool_spec()

    assert tool["determinism"] == "deterministic"
    assert tool["side_effects"] == ["read:dataset"]
    assert tool["entrypoint"] == "tool_profile_dataset"
    assert tool["input_schema"]["additionalProperties"] is False
    assert set(tool["input_schema"]["required"]) == {
        "dataset_id",
        "expected_content_hash",
        "workspace_revision",
        "analysis_generation",
        "semantic_mapping_hash",
    }
    properties = tool["input_schema"]["properties"]
    assert properties["sections"]["items"]["enum"] == [
        "overview",
        "target",
        "missing",
        "distribution",
        "correlation",
    ]
    assert properties["columns"]["uniqueItems"] is True
    assert properties["target_col"]["type"] == ["string", "null"]
    for name in (
        "frequency_top_k",
        "low_cardinality_threshold",
        "histogram_bins",
        "correlation_batch_size",
    ):
        assert properties[name]["type"] == "integer"
        assert properties[name]["minimum"] > 0
    output = tool["output_schema"]
    assert output["additionalProperties"] is False
    assert {
        "dataset_id",
        "dataset_content_hash",
        "expected_content_hash",
        "workspace_revision",
        "analysis_generation",
        "semantic_mapping_hash",
        "scan_scope",
        "row_count",
        "row_count_scanned",
        "result",
        "options_echo",
        "semantics",
    } <= set(output["required"])
    semantics = output["properties"]["semantics"]
    assert semantics["additionalProperties"] is False
    assert set(semantics["required"]) == {
        "target_col",
        "field_roles",
        "business_names",
    }


def test_data_ops_pack_permissions_do_not_expand_for_profile_dataset():
    manifest_path = (
        Path(__file__).parents[1]
        / "marvis"
        / "packs"
        / "data_ops"
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["version"] == "0.4.0"
    assert set(manifest["permissions"]) == {
        "read:dataset",
        "write:dataset",
        "read:join_plan",
        "write:join_plan",
        "read:materials",
    }


def test_profile_request_defaults_cover_every_section_and_core_config():
    assert data_ops_tools._profile_sections(None) == (
        "overview",
        "target",
        "missing",
        "distribution",
        "correlation",
    )
    config = data_ops_tools._build_descriptive_config({})
    assert config.to_dict() == {
        "max_columns": 200,
        "max_numeric_columns": 64,
        "max_pairs": 2016,
        "frequency_top_k": 20,
        "low_cardinality_threshold": 20,
        "histogram_bins": 20,
        "summary_batch_size": 16,
        "correlation_batch_size": 32,
    }


def _seed_task(db_path: Path, task_id: str = "task-1") -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO tasks(
                id, model_name, model_version, validator, source_dir,
                status, status_message, created_at, updated_at
            ) VALUES (?, 'profile task', 'v1', 'tester', '/tmp/source',
                      'created', 'created', ?, ?)
            """,
            (
                task_id,
                "2026-07-19T01:02:03+00:00",
                "2026-07-19T01:02:03+00:00",
            ),
        )


def _profile_runtime(tmp_path: Path) -> SimpleNamespace:
    settings = build_settings(tmp_path / "workspace")
    _seed_task(settings.db_path)
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    source = tmp_path / "sample.csv"
    pd.DataFrame(
        {
            "phone": ["13800138000", "13900139000", None, "13700137000"],
            "customer_name": ["张三", "李四", "王五", "张三"],
            "bad": [0, 1, 1, 0],
            "amount": [100.0, 200.0, None, 400.0],
        }
    ).to_csv(source, index=False)
    dataset = registry.register_from_upload("task-1", source, role="sample")

    workspace_repo = DataWorkspaceRepository(settings.db_path)
    activated = workspace_repo.save(
        "task-1",
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
        ),
        expected_revision=0,
    )
    mapping = DataSemanticMapping(
        target_col="bad",
        field_roles={
            "phone": "phone",
            "customer_name": "name",
            "bad": "target",
            "amount": "amount",
        },
        business_names={"bad": "风险标签", "amount": "授信金额"},
    )
    snapshot = workspace_repo.save(
        "task-1",
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
            page="semantics",
            semantic_mapping=mapping,
        ),
        expected_revision=activated.revision,
    )
    inputs = {
        "dataset_id": dataset.id,
        "expected_content_hash": dataset.content_hash,
        "workspace_revision": snapshot.revision,
        "analysis_generation": snapshot.analysis_generation,
        "semantic_mapping_hash": data_semantic_mapping_hash(mapping),
        "sections": ["missing", "correlation"],
        "columns": ["amount", "phone"],
        "target_col": "bad",
        "frequency_top_k": 7,
        "low_cardinality_threshold": 8,
        "histogram_bins": 9,
        "correlation_batch_size": 10,
    }
    ctx = ToolContext(
        task_id="task-1",
        seed=0,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )
    return SimpleNamespace(
        settings=settings,
        registry=registry,
        dataset=dataset,
        snapshot=snapshot,
        inputs=inputs,
        ctx=ctx,
    )


def _core_result(row_count: int) -> dict:
    return {
        "schema_version": "data-analysis.v1",
        "config": {},
        "dataset": {"row_count": row_count, "column_count": 4},
        "fields": [
            {"name": "amount"},
            {"name": "phone"},
            {"name": "bad"},
        ],
        "target_distribution": None,
        "correlations": {
            "columns": [],
            "values": [],
            "pair_counts": [],
            "reasons": [],
        },
    }


def test_profile_dataset_binds_verified_workspace_and_sensitive_sanitizers(
    tmp_path,
    monkeypatch,
):
    case = _profile_runtime(tmp_path)
    captured = {}
    verified_calls: list[str] = []
    original_verified = DatasetRegistry.resolve_verified_path

    def resolve_verified_spy(registry, dataset_id):
        verified_calls.append(dataset_id)
        return original_verified(registry, dataset_id)

    def reject_unverified_path(*_args, **_kwargs):
        raise AssertionError("profile_dataset must not call resolve_path")

    def fake_analyze(path, **kwargs):
        captured["path"] = Path(path)
        captured.update(kwargs)
        return _core_result(case.dataset.row_count)

    class FakeConfig:
        def to_dict(self):
            return {
                "max_columns": 200,
                "max_numeric_columns": 64,
                "max_pairs": 2016,
                "frequency_top_k": 7,
                "low_cardinality_threshold": 8,
                "histogram_bins": 9,
                "summary_batch_size": 16,
                "correlation_batch_size": 10,
            }

    monkeypatch.setattr(DatasetRegistry, "resolve_verified_path", resolve_verified_spy)
    monkeypatch.setattr(DatasetRegistry, "resolve_path", reject_unverified_path)
    monkeypatch.setattr(data_ops_tools, "_analyze_parquet", fake_analyze)
    monkeypatch.setattr(
        data_ops_tools,
        "_build_descriptive_config",
        lambda _inputs: FakeConfig(),
    )

    output = data_ops_tools.tool_profile_dataset(case.inputs, case.ctx)

    assert verified_calls == [case.dataset.id]
    assert captured["target_column"] == "bad"
    assert captured["columns"] == ("amount", "phone")
    assert captured["path"].is_file()
    assert captured["config"].to_dict()["frequency_top_k"] == 7
    assert captured["config"].to_dict()["low_cardinality_threshold"] == 8
    assert captured["config"].to_dict()["histogram_bins"] == 9
    assert captured["config"].to_dict()["correlation_batch_size"] == 10
    expected_result = _core_result(4)
    expected_result["target_distribution"] = {
        "status": "not_requested",
        "column": "bad",
    }
    expected_result["fields"][1]["sensitive_value_policy"] = (
        "frequency_tokenized_numeric_distribution_suppressed"
    )
    assert output == {
        "dataset_id": case.dataset.id,
        "dataset_content_hash": case.dataset.content_hash,
        "expected_content_hash": case.dataset.content_hash,
        "workspace_revision": case.snapshot.revision,
        "analysis_generation": case.snapshot.analysis_generation,
        "semantic_mapping_hash": case.inputs["semantic_mapping_hash"],
        "scan_scope": "full_dataset",
        "row_count": 4,
        "row_count_scanned": 4,
        "options_echo": {
            "sections": ["missing", "correlation"],
            "columns": ["amount", "phone"],
            "target_col": "bad",
            "frequency_top_k": 7,
            "low_cardinality_threshold": 8,
            "histogram_bins": 9,
            "correlation_batch_size": 10,
        },
        "semantics": {
            "target_col": "bad",
            "field_roles": {
                "amount": "amount",
                "phone": "phone",
                "bad": "target",
            },
            "business_names": {
                "amount": "授信金额",
                "bad": "风险标签",
            },
        },
        "result": expected_result,
    }
    sanitizers = captured["value_sanitizers"]
    assert set(sanitizers) == {"phone", "customer_name"}
    masked = sanitizers["customer_name"]({"type": "string", "value": "张三"})
    assert masked["type"] == "string"
    assert masked["value"] != "张三"


def test_profile_dataset_rejects_target_override(
    tmp_path,
    monkeypatch,
):
    case = _profile_runtime(tmp_path)

    def must_not_analyze(*_args, **_kwargs):
        raise AssertionError("invalid request reached descriptive kernel")

    monkeypatch.setattr(data_ops_tools, "_analyze_parquet", must_not_analyze)

    with pytest.raises(ValueError, match="target_col.*workspace"):
        data_ops_tools.tool_profile_dataset(
            {**case.inputs, "target_col": "amount"},
            case.ctx,
        )


def test_profile_sanitizer_cannot_be_downgraded_by_workspace_role_override():
    dataset = SimpleNamespace(
        columns=(SimpleNamespace(name="phone", semantic_role="phone"),),
    )
    mapping = DataSemanticMapping(field_roles={"phone": "feature"})

    sanitizers = data_ops_tools._profile_value_sanitizers(
        dataset,
        mapping,
        dataset_content_hash="a" * 64,
    )

    assert set(sanitizers) == {"phone"}
    masked = sanitizers["phone"]({"type": "string", "value": "13800138000"})
    assert masked["value"] != "13800138000"


def test_profile_section_projection_keeps_actual_core_type_keys_and_missing_only():
    report = {
        "schema_version": "data-analysis.v1",
        "config": {},
        "dataset": {"row_count": 3},
        "fields": [
            {
                "name": "amount",
                "duckdb_type": "DOUBLE",
                "kind": "numeric",
                "selection_role": "requested",
                "row_count": 3,
                "null_count": 1,
                "null_rate": 1 / 3,
                "distinct_count": 2,
                "numeric": {"mean": 10.0},
                "frequency": {"items": []},
                "histogram": {"bins": []},
            }
        ],
        "target_distribution": {"status": "not_configured", "column": None},
        "correlations": {"columns": ["amount"]},
    }

    selected = data_ops_tools._select_profile_sections(
        report,
        sections=("missing",),
        target_column=None,
    )

    assert selected["fields"] == [
        {
            "name": "amount",
            "duckdb_type": "DOUBLE",
            "kind": "numeric",
            "selection_role": "requested",
            "row_count": 3,
            "null_count": 1,
            "null_rate": 1 / 3,
        }
    ]
    assert selected["target_distribution"]["status"] == "not_requested"
    assert selected["correlations"]["status"] == "not_requested"


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("expected_content_hash", "f" * 64, "content hash"),
        ("workspace_revision", 999, "workspace revision"),
        ("analysis_generation", 999, "analysis generation"),
        ("semantic_mapping_hash", "e" * 64, "semantic mapping hash"),
    ],
)
def test_profile_dataset_rejects_stale_identity_before_analysis(
    tmp_path,
    monkeypatch,
    field,
    replacement,
    message,
):
    case = _profile_runtime(tmp_path)

    def must_not_analyze(*_args, **_kwargs):
        raise AssertionError("stale workspace identity reached descriptive kernel")

    monkeypatch.setattr(data_ops_tools, "_analyze_parquet", must_not_analyze)
    inputs = {**case.inputs, field: replacement}

    with pytest.raises(ValueError, match=message):
        data_ops_tools.tool_profile_dataset(inputs, case.ctx)


def test_profile_dataset_rejects_cross_task_dataset_before_workspace_read(
    tmp_path,
    monkeypatch,
):
    case = _profile_runtime(tmp_path)
    ctx = ToolContext(
        task_id="task-2",
        seed=0,
        datasets_root=case.settings.datasets_dir,
        workspace=case.settings.workspace,
    )

    def must_not_analyze(*_args, **_kwargs):
        raise AssertionError("cross-task dataset reached descriptive kernel")

    monkeypatch.setattr(data_ops_tools, "_analyze_parquet", must_not_analyze)

    with pytest.raises(PermissionError, match="belongs to task"):
        data_ops_tools.tool_profile_dataset(case.inputs, ctx)


def test_profile_dataset_runs_through_tool_runner_without_leaking_sensitive_values(
    tmp_path,
):
    case = _profile_runtime(tmp_path)
    plugin_repo = PluginRepository(case.settings.db_path)
    plugin_registry = PluginRegistry(plugin_repo)
    load_builtin_packs(
        plugin_registry,
        Path(__file__).parents[1] / "marvis" / "packs",
    )
    runner = ToolRunner(
        ToolRegistry(plugin_registry),
        plugin_repo,
        python_executable=sys.executable,
        datasets_root=case.settings.datasets_dir,
        workspace=case.settings.workspace,
    )
    inputs = {
        **case.inputs,
        "sections": [
            "overview",
            "target",
            "missing",
            "distribution",
            "correlation",
        ],
        "columns": ["phone", "customer_name", "amount"],
    }

    result = runner.invoke(
        ToolRef("data_ops", "profile_dataset"),
        inputs,
        task_id="task-1",
    )

    assert result.ok is True, result.error
    assert result.output["dataset_id"] == case.dataset.id
    assert result.output["dataset_content_hash"] == case.dataset.content_hash
    assert result.output["scan_scope"] == "full_dataset"
    assert result.output["row_count_scanned"] == case.dataset.row_count
    assert result.output["result"]["dataset"]["row_count"] == case.dataset.row_count
    assert result.output["semantics"] == {
        "target_col": "bad",
        "field_roles": {
            "phone": "phone",
            "customer_name": "name",
            "amount": "amount",
            "bad": "target",
        },
        "business_names": {
            "amount": "授信金额",
            "bad": "风险标签",
        },
    }
    phone = next(
        field
        for field in result.output["result"]["fields"]
        if field["name"] == "phone"
    )
    assert phone["numeric"] is None
    assert phone["histogram"] is None
    assert phone["sensitive_value_policy"].startswith("frequency_tokenized")
    assert "phone" not in result.output["result"]["correlations"]["columns"]
    serialized = json.dumps(result.output, ensure_ascii=False, allow_nan=False)
    for sensitive in (
        "13800138000",
        "13900139000",
        "13700137000",
        "张三",
        "李四",
        "王五",
    ):
        assert sensitive not in serialized
    assert (
        DataWorkspaceRepository(case.settings.db_path).get_or_default("task-1")
        == case.snapshot
    )
