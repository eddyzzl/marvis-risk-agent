import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from marvis.data.backend import DataBackend
from marvis.data.errors import NanLabelNotConfirmedError
from marvis.data.registry import DatasetRegistry
from marvis.db import (
    DatasetRepository,
    PluginRepository,
    StrategyRepository,
    TaskRepository,
    init_db,
)
from marvis.domain import TaskCreate
from marvis.packs.strategy import tools as strategy_tools
from marvis.plugins.loader import load_builtin_packs, load_manifest
from marvis.plugins.manifest import GovernancePolicy, ToolRef
from marvis.plugins.errors import SchemaValidationError
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner
from marvis.plugins.schema_validation import validate_against_schema
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.settings import build_settings
from tests.strategy_tool_sample_design_support import (
    materialize_strategy_tool_sample_design,
)


def _runtime(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    plugin_repo = PluginRepository(settings.db_path)
    plugin_registry = PluginRegistry(plugin_repo)
    packs_root = Path(__file__).parents[1] / "marvis" / "packs"
    _register_policy_neutral_strategy_pack(plugin_registry, packs_root)
    runner = ToolRunner(
        ToolRegistry(plugin_registry),
        plugin_repo,
        python_executable=sys.executable,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )
    data_repo = DatasetRepository(settings.db_path)
    backend = DataBackend(settings.datasets_dir)
    registry = DatasetRegistry(data_repo, backend, settings.datasets_dir)
    task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="策略能力包样例",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "source"),
            algorithm="lr",
            run_mode="agent",
            target_col="bad",
            score_col="score",
            split_col="split",
            time_col="month",
            feature_columns=["score", "segment"],
        )
    )
    return runner, plugin_registry, registry, task


def _register_policy_neutral_strategy_pack(plugin_registry, packs_root):
    """Register a policy-neutral clone for direct strategy-kernel tests only."""
    manifest = load_manifest(packs_root / "strategy", builtin=True)
    neutral_manifest = replace(
        manifest,
        tools=tuple(
            replace(tool, policy=GovernancePolicy()) for tool in manifest.tools
        ),
    )
    plugin_registry.register(neutral_manifest, enabled=True)


def _real_builtin_registry(tmp_path):
    """Load shipped manifests unchanged for manifest contract assertions."""
    settings = build_settings(tmp_path / "manifest-workspace")
    init_db(settings.db_path)
    plugin_registry = PluginRegistry(PluginRepository(settings.db_path))
    packs_root = Path(__file__).parents[1] / "marvis" / "packs"
    load_builtin_packs(plugin_registry, packs_root)
    return plugin_registry


def _register_strategy_sample(registry, tmp_path, task_id: str):
    frame = pd.DataFrame(
        {
            "customer_id": ["A", "A", "B", "B", "C", "C"],
            "month": ["2026-01", "2026-02", "2026-01", "2026-02", "2026-01", "2026-02"],
            "status": ["C", "M1", "C", "C", "M3+", "M3+"],
            "cohort": ["202601", "202601", "202602", "202602", "202603", "202603"],
            "mob": [0, 1, 0, 1, 0, 1],
            "bad": [1, 1, 0, 0, 1, 1],
            "score": [580, 620, 730, 760, 590, 800],
            "ead": [1000.0, 2000.0, 1000.0, 500.0, 1000.0, 800.0],
            "pd": [0.20, 0.05, 0.02, 0.10, 0.15, 0.03],
            "segment": ["A", "A", "B", "B", "A", "B"],
        }
    )
    path = tmp_path / "strategy_sample.parquet"
    frame.to_parquet(path, index=False)
    return registry.register_existing(path, task_id=task_id, role="strategy_sample")


def test_task_analysis_artifact_identity_includes_producer_version(monkeypatch):
    assumptions = {"dataset_id": "dataset-1", "segment_col": "segment"}
    original_stem = strategy_tools._analysis_artifact_stem(
        "profit",
        "a" * 64,
        assumptions,
    )
    original_provenance = strategy_tools._task_analysis_artifact_provenance(
        analysis_kind="profit",
        source_hash="a" * 64,
        assumptions=assumptions,
    )

    assert original_provenance["schema_version"] == "task-artifact-provenance.v1"
    assert original_provenance["producer_version"] == "strategy.profit_calc.v1"

    monkeypatch.setitem(
        strategy_tools._TASK_ANALYSIS_PRODUCER_VERSIONS,
        "profit",
        "strategy.profit_calc.v2",
    )
    upgraded_stem = strategy_tools._analysis_artifact_stem(
        "profit",
        "a" * 64,
        assumptions,
    )
    upgraded_provenance = strategy_tools._task_analysis_artifact_provenance(
        analysis_kind="profit",
        source_hash="a" * 64,
        assumptions=assumptions,
    )

    assert upgraded_stem != original_stem
    assert upgraded_provenance["producer_version"] == "strategy.profit_calc.v2"


def test_task_analysis_artifact_registration_failure_rolls_back_files_and_rows(
    tmp_path,
    monkeypatch,
):
    settings = build_settings(tmp_path / "artifact-workspace")
    init_db(settings.db_path)
    task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="任务分析产物事务测试",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path),
            task_type="strategy",
        )
    )
    repository = TaskArtifactRepository(settings.db_path)
    runtime = SimpleNamespace(settings=settings, task_artifacts=repository)
    assumptions = {"dataset_id": "dataset-1", "segment_col": "segment"}
    stem = strategy_tools._analysis_artifact_stem(
        "profit",
        "b" * 64,
        assumptions,
    )
    output_dir = settings.tasks_dir / task.id / "strategy_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    markdown_path = output_dir / f"{stem}.md"
    csv_path.write_text("old csv", encoding="utf-8")
    markdown_path.write_text("old markdown", encoding="utf-8")

    original_register = repository.register_on_connection
    call_count = 0

    def fail_on_second_registration(conn, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("simulated second registry failure")
        return original_register(conn, **kwargs)

    monkeypatch.setattr(
        repository,
        "register_on_connection",
        fail_on_second_registration,
    )

    with pytest.raises(RuntimeError, match="simulated second registry failure"):
        strategy_tools._write_task_analysis_artifacts(
            runtime,
            task_id=task.id,
            analysis_kind="profit",
            source_hash="b" * 64,
            assumptions=assumptions,
            files=(
                ("profit_csv", "csv", "new csv"),
                ("profit_markdown", "md", "new markdown"),
            ),
        )

    assert csv_path.read_text(encoding="utf-8") == "old csv"
    assert markdown_path.read_text(encoding="utf-8") == "old markdown"
    assert repository.list_for_task(task.id) == []


def test_report_bundle_manifest_accepts_only_exact_optional_candidate_stability_ref(
    tmp_path,
) -> None:
    manifest = _real_builtin_registry(tmp_path).get("strategy")
    tool = next(
        item for item in manifest.tools if item.name == "build_report_bundle_v2"
    )
    schema = tool.input_schema
    candidate_schema = {
        **schema["$defs"]["candidate_stability_ref"],
        "$defs": {"sha256": schema["$defs"]["sha256"]},
    }
    candidate_ref = {
        "artifact_id": "a" * 64,
        "expected_artifact_content_hash": "b" * 64,
        "expected_stability_id": "candidate-stability-" + ("1" * 24),
        "expected_stability_content_hash": "c" * 64,
    }

    assert schema["properties"]["candidate_stability_ref"] == {
        "$ref": "#/$defs/candidate_stability_ref"
    }
    assert "candidate_stability_ref" not in schema["required"]
    assert set(candidate_schema["properties"]) == set(candidate_ref)
    assert set(candidate_schema["required"]) == set(candidate_ref)
    assert candidate_schema["additionalProperties"] is False
    validate_against_schema(
        candidate_ref,
        candidate_schema,
        label="candidate_stability_ref",
    )
    with pytest.raises(SchemaValidationError):
        validate_against_schema(
            {**candidate_ref, "forged_metric": 0.99},
            candidate_schema,
            label="candidate_stability_ref",
        )


def test_strategy_v2_materializers_are_thin_runtime_forwarders(monkeypatch):
    ctx = object()
    runtime = object()
    inputs = {"nullable_value": None}
    calls = []

    def fake_runtime(actual_ctx):
        assert actual_ctx is ctx
        return runtime

    def fake_sample(actual_inputs, actual_ctx, actual_runtime):
        calls.append(("sample", actual_inputs, actual_ctx, actual_runtime))
        return {"result": "sample"}

    def fake_model(actual_inputs, actual_ctx, actual_runtime):
        calls.append(("model", actual_inputs, actual_ctx, actual_runtime))
        return {"result": "model"}

    monkeypatch.setattr(strategy_tools, "_runtime", fake_runtime)
    monkeypatch.setattr(
        strategy_tools,
        "run_materialize_sample_design_v2",
        fake_sample,
    )
    monkeypatch.setattr(
        strategy_tools,
        "run_materialize_model_evidence_v2",
        fake_model,
    )

    assert strategy_tools.tool_materialize_sample_design_v2(inputs, ctx) == {
        "result": "sample"
    }
    assert strategy_tools.tool_materialize_model_evidence_v2(inputs, ctx) == {
        "result": "model"
    }
    assert calls == [
        ("sample", inputs, ctx, runtime),
        ("model", inputs, ctx, runtime),
    ]


def test_strategy_manifest_registers_expected_tools(tmp_path):
    plugin_registry = _real_builtin_registry(tmp_path)

    manifest = plugin_registry.get("strategy")
    tool_names = {tool.name for tool in manifest.tools}
    build_tool = next(tool for tool in manifest.tools if tool.name == "build_strategy")
    backtest_tool = next(
        tool for tool in manifest.tools if tool.name == "backtest_strategy"
    )
    apply_tool = next(tool for tool in manifest.tools if tool.name == "apply_strategy")
    candidate_tool = next(
        tool for tool in manifest.tools if tool.name == "design_strategy_candidate"
    )
    univariate_tool = next(
        tool for tool in manifest.tools if tool.name == "analyze_univariate_candidates"
    )
    automatic_tree_tool = next(
        tool for tool in manifest.tools if tool.name == "build_automatic_tree_candidate"
    )
    automatic_tree_apply_tool = next(
        tool for tool in manifest.tools if tool.name == "apply_automatic_tree"
    )
    automatic_tree_leaf_tool = next(
        tool
        for tool in manifest.tools
        if tool.name == "materialize_automatic_tree_leaf_fragment"
    )
    voting_tool = next(
        tool for tool in manifest.tools if tool.name == "build_voting_candidate"
    )
    voting_search_tool = next(
        tool for tool in manifest.tools if tool.name == "search_voting_candidates"
    )
    cross_matrix_tool = next(
        tool for tool in manifest.tools if tool.name == "build_cross_matrix_candidate"
    )
    cross_matrix_cell_selection_tool = next(
        tool
        for tool in manifest.tools
        if tool.name == "materialize_cross_matrix_cell_selection"
    )
    refinement_tool = next(
        tool for tool in manifest.tools if tool.name == "refine_univariate_candidate"
    )
    pool_mutation_tools = [
        next(tool for tool in manifest.tools if tool.name == name)
        for name in (
            "add_candidate_to_pool",
            "remove_pool_entry",
            "set_pool_entry_action",
            "reorder_strategy_pool",
        )
    ]
    add_pool_tool = next(
        tool for tool in manifest.tools if tool.name == "add_candidate_to_pool"
    )
    compile_pool_tool = next(
        tool for tool in manifest.tools if tool.name == "compile_strategy_pool"
    )
    measure_pool_impact_tool = next(
        tool for tool in manifest.tools if tool.name == "measure_pool_impact"
    )
    candidate_stability_tool = next(
        tool
        for tool in manifest.tools
        if tool.name == "measure_candidate_monthly_stability"
    )
    measure_pool_validation_tool = next(
        tool
        for tool in manifest.tools
        if tool.name == "measure_strategy_pool_validation"
    )
    measure_impact_cube_tool = next(
        tool
        for tool in manifest.tools
        if tool.name == "measure_strategy_impact_cube"
    )
    delivery_tool = next(
        tool for tool in manifest.tools if tool.name == "export_strategy_delivery"
    )
    report_bundle_tool = next(
        tool for tool in manifest.tools if tool.name == "build_report_bundle_v2"
    )
    run_monitoring_tool = next(
        tool for tool in manifest.tools if tool.name == "run_strategy_monitoring"
    )
    disposition_tool = next(
        tool for tool in manifest.tools if tool.name == "apply_monitoring_disposition"
    )
    challenger_report_tool = next(
        tool for tool in manifest.tools if tool.name == "render_challenger_report"
    )
    project_context_tool = next(
        tool for tool in manifest.tools if tool.name == "materialize_project_context"
    )
    sample_v2_tool = next(
        tool for tool in manifest.tools if tool.name == "materialize_sample_design_v2"
    )
    model_evidence_v2_tool = next(
        tool for tool in manifest.tools if tool.name == "materialize_model_evidence_v2"
    )

    assert tool_names == {
        "vintage_curve",
        "roll_rate_matrix",
        "profit_calc",
        "materialize_project_context",
        "materialize_sample_design",
        "materialize_sample_design_v2",
        "materialize_model_evidence_v2",
        "analyze_univariate_candidates",
        "build_automatic_tree_candidate",
        "apply_automatic_tree",
        "materialize_automatic_tree_leaf_fragment",
        "build_voting_candidate",
        "search_voting_candidates",
        "build_cross_matrix_candidate",
        "materialize_cross_matrix_cell_selection",
        "build_scorecard_band_asset",
        "materialize_scorecard_cutoff_selection",
        "refine_univariate_candidate",
        "add_candidate_to_pool",
        "remove_pool_entry",
        "set_pool_entry_action",
        "reorder_strategy_pool",
        "compile_strategy_pool",
        "measure_pool_impact",
        "measure_candidate_monthly_stability",
        "measure_strategy_pool_validation",
        "measure_strategy_impact_cube",
        "export_strategy_delivery",
        "build_report_bundle_v2",
        "design_strategy_candidate",
        "build_strategy",
        "apply_strategy",
        "backtest_strategy",
        "tradeoff_view",
        "design_cutoff_bands",
        "compare_strategies",
        "adopt_strategy",
        "render_strategy_doc",
        "mine_rules",
        "evaluate_rule_set",
        "limit_pricing_matrix",
        "select_rule_set",
        "limit_pricing_matrix",
        "render_challenger_report",
        "run_strategy_monitoring",
        "apply_monitoring_disposition",
        "render_monitoring_report",
    }
    assert manifest.version == "0.19.0"
    assert delivery_tool.determinism == "deterministic"
    assert delivery_tool.failure_policy == "fail"
    assert delivery_tool.policy.human_decision_gate == "none"
    assert delivery_tool.policy.effect_authorization == "none"
    assert set(delivery_tool.side_effects) == {
        "read:task",
        "read:dataset",
        "read:strategy",
        "write:artifact",
    }
    assert delivery_tool.input_schema["additionalProperties"] is False
    assert delivery_tool.output_schema["additionalProperties"] is False
    report_output_schema = report_bundle_tool.output_schema
    assert report_output_schema["properties"]["schema_version"] == {
        "const": "strategy.build-report-bundle-v2-tool.v3"
    }
    report_artifacts = report_output_schema["properties"]["artifacts"]
    assert report_artifacts["minItems"] == 4
    assert report_artifacts["maxItems"] == 4
    assert set(
        report_artifacts["items"]["properties"]["kind"]["enum"]
    ) == {
        "strategy_report_bundle_json",
        "strategy_report_markdown",
        "strategy_report_xlsx",
        "strategy_report_docx",
    }
    assert set(
        report_artifacts["items"]["properties"]["format"]["enum"]
    ) == {"json", "markdown", "xlsx", "docx"}
    assert set(
        report_artifacts["items"]["properties"]["filename"]["enum"]
    ) == {"report.json", "report.md", "report.xlsx", "report.docx"}
    for tool in (project_context_tool, sample_v2_tool, model_evidence_v2_tool):
        assert tool.determinism == "deterministic"
        assert tool.failure_policy == "fail"
        assert tool.policy.human_decision_gate == "none"
        assert tool.policy.effect_authorization == "none"
    for tool in (sample_v2_tool, model_evidence_v2_tool):
        assert set(tool.side_effects) == {
            "read:task",
            "read:dataset",
            "write:artifact",
        }
        assert tool.input_schema["additionalProperties"] is False
        assert tool.output_schema["additionalProperties"] is False
    assert sample_v2_tool.output_schema["properties"]["schema_version"] == {
        "const": "strategy.materialize-sample-design-v2-tool.v2"
    }
    assert sample_v2_tool.input_schema["properties"]["partitioning"]["oneOf"]
    assert sample_v2_tool.input_schema["$defs"]["predicate"]["oneOf"]
    assert model_evidence_v2_tool.output_schema["properties"]["schema_version"] == {
        "const": "strategy.materialize-model-evidence-v2-tool.v3"
    }
    assert model_evidence_v2_tool.input_schema["properties"][
        "univariate_sources"
    ]["maxItems"] == 100
    assert set(
        model_evidence_v2_tool.output_schema["properties"]["artifact"][
            "required"
        ]
    ) == {"kind", "format", "filename", "content_hash"}
    assert "artifact_id" not in model_evidence_v2_tool.output_schema["properties"][
        "artifact"
    ]["properties"]
    assert "download_url" not in model_evidence_v2_tool.output_schema[
        "properties"
    ]["artifact"]["properties"]
    assert build_tool.determinism == "deterministic"
    assert "write:strategy" in build_tool.side_effects
    assert "write:dataset" in apply_tool.side_effects
    assert "write:backtest" in backtest_tool.side_effects
    assert backtest_tool.policy.human_decision_gate == "none"
    assert candidate_tool.policy.human_decision_gate == "none"
    assert candidate_tool.side_effects == ("read:dataset",)
    assert univariate_tool.policy.human_decision_gate == "none"
    assert set(univariate_tool.side_effects) == {
        "read:task",
        "read:dataset",
        "write:artifact",
    }
    assert univariate_tool.input_schema["properties"]["methods"] == {
        "type": "array",
        "maxItems": 5,
        "uniqueItems": True,
        "items": {
            "type": "string",
            "enum": [
                "equal_frequency",
                "equal_width",
                "chimerge",
                "tree",
                "manual",
            ],
        },
    }
    assert univariate_tool.input_schema["properties"]["manual_breakpoints"][
        "additionalProperties"
    ] == {
        "type": "array",
        "minItems": 1,
        "maxItems": 19,
        "items": {"type": "number"},
    }
    assert univariate_tool.output_schema["properties"]["schema_version"] == {
        "type": "string",
        "enum": [
            "strategy.univariate-candidate-tool.v1",
            "strategy.univariate-candidate-tool.v2",
        ],
    }
    assert len(univariate_tool.output_schema["oneOf"]) == 2
    assert automatic_tree_tool.determinism == "deterministic"
    assert automatic_tree_tool.policy.human_decision_gate == "none"
    assert automatic_tree_tool.policy.effect_authorization == "none"
    assert set(automatic_tree_tool.side_effects) == {
        "read:task",
        "read:dataset",
        "write:artifact",
    }
    assert automatic_tree_tool.input_schema["additionalProperties"] is False
    assert automatic_tree_tool.input_schema["properties"]["features"]["maxItems"] == 50
    assert automatic_tree_tool.input_schema["properties"]["max_depth"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 8,
        "default": 4,
    }
    assert automatic_tree_tool.output_schema["additionalProperties"] is False
    assert automatic_tree_tool.output_schema["properties"]["artifacts"]["minItems"] == 6
    automatic_leaf_schema = automatic_tree_tool.output_schema["properties"][
        "leaf_index"
    ]["items"]
    assert set(automatic_leaf_schema["required"]) == {
        "leaf_id",
        "fragment_id",
        "fragment_hash",
        "rule_id",
        "effect_id",
        "condition",
        "requirements",
        "metric_basis",
        "measurements",
    }
    assert automatic_leaf_schema["additionalProperties"] is False
    assert automatic_leaf_schema["properties"]["requirements"] == {
        "type": "array",
        "maxItems": 0,
    }
    report_gap_schema = automatic_tree_tool.output_schema["properties"][
        "report_info_gaps"
    ]
    assert report_gap_schema["maxItems"] == 3
    assert {
        (
            branch["properties"]["code"]["const"],
            branch["properties"]["context"]["const"],
        )
        for branch in report_gap_schema["items"]["oneOf"]
    } == {
        ("sample_weight_not_provided", "sample_weight"),
        ("loan_amount_not_provided", "loan_amount"),
        ("overdue_amount_not_provided", "overdue_amount"),
    }
    assert all(
        branch["properties"]["blocking"] == {"const": False}
        and set(branch["required"]) == {"code", "context", "blocking"}
        and branch["additionalProperties"] is False
        for branch in report_gap_schema["items"]["oneOf"]
    )
    assert "report_info_gaps" in automatic_tree_tool.output_schema["required"]
    assert automatic_tree_apply_tool.determinism == "deterministic"
    assert automatic_tree_apply_tool.policy.human_decision_gate == "none"
    assert automatic_tree_apply_tool.policy.effect_authorization == "none"
    assert set(automatic_tree_apply_tool.side_effects) == {
        "read:task",
        "read:dataset",
        "write:artifact",
        "write:dataset",
        "write:task",
    }
    assert automatic_tree_apply_tool.input_schema["additionalProperties"] is False
    assert set(automatic_tree_apply_tool.input_schema["required"]) == {
        "source_artifact_id",
        "expected_artifact_content_hash",
        "expected_asset_id",
        "expected_asset_hash",
        "expected_tree_result_hash",
        "dataset_id",
        "expected_content_hash",
        "workspace_revision",
        "analysis_generation",
        "semantic_mapping_hash",
        "activate_result",
    }
    assert (
        automatic_tree_apply_tool.input_schema["properties"]["leaf_id_column"][
            "default"
        ]
        == "automatic_tree_leaf_id"
    )
    assert (
        automatic_tree_apply_tool.input_schema["properties"]["rule_id_column"][
            "default"
        ]
        == "automatic_tree_rule_id"
    )
    assert automatic_tree_apply_tool.output_schema["additionalProperties"] is False
    assert automatic_tree_apply_tool.output_schema["properties"]["schema_version"] == {
        "const": "strategy.apply-automatic-tree-tool.v1"
    }
    assert not (
        {"rank", "action", "path"}
        & set(automatic_tree_apply_tool.output_schema["properties"])
    )
    assert automatic_tree_leaf_tool.determinism == "deterministic"
    assert automatic_tree_leaf_tool.policy.human_decision_gate == "none"
    assert automatic_tree_leaf_tool.policy.effect_authorization == "none"
    assert set(automatic_tree_leaf_tool.side_effects) == {
        "read:task",
        "write:artifact",
    }
    assert automatic_tree_leaf_tool.input_schema["additionalProperties"] is False
    assert set(automatic_tree_leaf_tool.input_schema["required"]) == {
        "source_artifact_id",
        "expected_artifact_content_hash",
        "expected_asset_id",
        "expected_asset_hash",
        "expected_tree_result_hash",
        "leaf_id",
    }
    assert automatic_tree_leaf_tool.input_schema["properties"]["selection_reason"] == {
        "type": ["string", "null"],
        "minLength": 1,
        "maxLength": 500,
        "default": None,
    }
    assert automatic_tree_leaf_tool.output_schema["additionalProperties"] is False
    assert automatic_tree_leaf_tool.output_schema["properties"]["schema_version"] == {
        "const": "strategy.materialize-automatic-tree-leaf-fragment-tool.v1"
    }
    assert (
        automatic_tree_leaf_tool.output_schema["properties"]["artifacts"]["minItems"]
        == 1
    )
    assert voting_tool.determinism == "deterministic"
    assert voting_tool.policy.human_decision_gate == "none"
    assert voting_tool.policy.effect_authorization == "none"
    assert set(voting_tool.side_effects) == {
        "read:task",
        "read:dataset",
        "write:artifact",
    }
    assert voting_tool.input_schema["additionalProperties"] is False
    assert set(voting_tool.input_schema["required"]) == {
        "strategy_type",
        "expected_pool_revision",
        "expected_pool_snapshot_hash",
        "selected_entry_ids",
        "n",
    }
    assert voting_tool.input_schema["properties"]["selected_entry_ids"] == {
        "type": "array",
        "minItems": 2,
        "maxItems": 50,
        "uniqueItems": True,
        "items": {"type": "string", "minLength": 1},
    }
    assert voting_tool.output_schema["additionalProperties"] is False
    assert voting_tool.output_schema["properties"]["schema_version"] == {
        "const": "strategy.build-voting-candidate-tool.v2"
    }
    assert voting_tool.output_schema["properties"]["selected_entries"][
        "items"
    ] == {"$ref": "#/$defs/selected_entry"}
    assert voting_tool.output_schema["properties"]["not_admitted"] == {
        "const": True
    }
    assert voting_tool.output_schema["properties"]["not_applied"] == {
        "const": True
    }
    assert voting_tool.output_schema["properties"]["not_adopted"] == {
        "const": True
    }
    assert voting_tool.output_schema["properties"]["not_deployed"] == {
        "const": True
    }
    assert voting_tool.output_schema["properties"]["artifacts"] == {
        "type": "array",
        "minItems": 1,
        "maxItems": 1,
        "items": {"$ref": "#/$defs/artifact"},
    }
    assert voting_search_tool.determinism == "deterministic"
    assert voting_search_tool.policy.human_decision_gate == "none"
    assert voting_search_tool.policy.effect_authorization == "none"
    assert set(voting_search_tool.side_effects) == {
        "read:task",
        "read:dataset",
        "write:artifact",
    }
    assert voting_search_tool.input_schema["additionalProperties"] is False
    assert voting_search_tool.input_schema["properties"]["member_count"] == {
        "type": "integer",
        "minimum": 2,
        "maximum": 50,
    }
    assert voting_search_tool.input_schema["properties"]["pool_ref"][
        "additionalProperties"
    ] is False
    assert voting_search_tool.output_schema["additionalProperties"] is False
    assert voting_search_tool.output_schema["properties"]["schema_version"] == {
        "const": "strategy.search-voting-candidates-tool.v1"
    }
    assert voting_search_tool.output_schema["properties"][
        "excluded_unsupported_rule_ids"
    ]["uniqueItems"] is True
    for field in (
        "not_mutated_pool",
        "not_selected",
        "not_admitted",
        "not_applied",
        "not_adopted",
        "not_deployed",
    ):
        assert voting_search_tool.output_schema["properties"][field] == {
            "const": True
        }
    assert cross_matrix_tool.determinism == "deterministic"
    assert cross_matrix_tool.policy.human_decision_gate == "none"
    assert cross_matrix_tool.policy.effect_authorization == "none"
    assert set(cross_matrix_tool.side_effects) == {
        "read:task",
        "read:dataset",
        "write:artifact",
    }
    assert cross_matrix_tool.input_schema["additionalProperties"] is False
    assert set(cross_matrix_tool.input_schema["required"]) == {
        "source_artifact_id",
        "expected_artifact_content_hash",
        "expected_candidate_id",
        "expected_evidence_hash",
        "x_feature",
        "x_method",
        "y_feature",
        "y_method",
    }
    assert "budget" not in cross_matrix_tool.input_schema["properties"]
    assert "cells" not in cross_matrix_tool.input_schema["properties"]
    assert cross_matrix_tool.output_schema["additionalProperties"] is False
    assert "manual" in cross_matrix_tool.input_schema["properties"]["x_method"]["enum"]
    assert "manual" in cross_matrix_tool.input_schema["properties"]["y_method"]["enum"]
    assert cross_matrix_tool.output_schema["properties"]["schema_version"] == {
        "type": "string",
        "enum": [
            "strategy.build-cross-matrix-candidate-tool.v1",
            "strategy.build-cross-matrix-candidate-tool.v2",
        ],
    }
    assert len(cross_matrix_tool.output_schema["oneOf"]) == 2
    assert cross_matrix_tool.output_schema["properties"]["cell_count"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 400,
    }
    assert cross_matrix_tool.output_schema["properties"]["asset_id"] == {
        "type": "string",
        "pattern": "^candidate-asset-[0-9a-f]{32}$",
    }
    assert cross_matrix_tool.output_schema["properties"]["candidate_id"] == {
        "type": "string",
        "pattern": "^candidate-[0-9a-f]{32}$",
    }
    assert cross_matrix_tool.output_schema["properties"]["evidence_hash"] == {
        "type": "string",
        "pattern": "^[0-9a-f]{64}$",
    }
    assert {"candidate_id", "evidence_hash"} <= set(
        cross_matrix_tool.output_schema["required"]
    )
    cross_asset_schema = cross_matrix_tool.output_schema["$defs"][
        "cross_matrix_asset"
    ]
    assert cross_asset_schema["additionalProperties"] is False
    assert cross_asset_schema["properties"]["schema_version"]["enum"] == [
        "strategy.cross-matrix-candidate-asset.v1",
        "strategy.cross-matrix-candidate-asset.v2",
    ]
    assert cross_asset_schema["properties"]["producer_version"]["enum"] == [
        "strategy.cross-matrix-candidate-asset/1",
        "strategy.cross-matrix-candidate-asset/2",
    ]
    assert set(cross_asset_schema["required"]) == {
        "schema_version",
        "asset_type",
        "lifecycle",
        "parent",
        "sample_identity",
        "axes",
        "measurement",
        "budget",
        "matrix",
        "summary",
        "producer_version",
        "candidate_evidence",
        "asset_id",
        "asset_hash",
    }
    for boundary in (
        "not_selected",
        "not_admitted",
        "not_applied",
        "not_adopted",
        "not_deployed",
    ):
        assert cross_matrix_tool.output_schema["properties"][boundary] == {
            "const": True
        }
    assert cross_matrix_tool.output_schema["properties"]["artifacts"] == {
        "type": "array",
        "minItems": 1,
        "maxItems": 1,
        "items": {"$ref": "#/$defs/artifact"},
    }
    assert cross_matrix_cell_selection_tool.determinism == "deterministic"
    assert cross_matrix_cell_selection_tool.policy.human_decision_gate == "none"
    assert (
        cross_matrix_cell_selection_tool.policy.effect_authorization == "none"
    )
    assert set(cross_matrix_cell_selection_tool.side_effects) == {
        "read:task",
        "read:dataset",
        "write:artifact",
    }
    assert cross_matrix_cell_selection_tool.input_schema["additionalProperties"] is False
    assert set(cross_matrix_cell_selection_tool.input_schema["required"]) == {
        "source_artifact_id",
        "expected_artifact_content_hash",
        "expected_asset_id",
        "expected_asset_hash",
        "expected_candidate_id",
        "expected_evidence_hash",
        "cell_ids",
    }
    assert cross_matrix_cell_selection_tool.input_schema["properties"]["cell_ids"] == {
        "type": "array",
        "minItems": 1,
        "maxItems": 400,
        "uniqueItems": True,
        "items": {
            "type": "string",
            "pattern": "^cross-cell-[0-9a-f]{32}$",
        },
    }
    assert cross_matrix_cell_selection_tool.output_schema["additionalProperties"] is False
    assert cross_matrix_cell_selection_tool.output_schema["properties"][
        "schema_version"
    ] == {"const": "strategy.materialize-cross-matrix-cell-selection-tool.v1"}
    assert cross_matrix_cell_selection_tool.output_schema["properties"][
        "fragment_id"
    ] == {
        "type": "string",
        "pattern": "^cross-matrix-cell-group-[0-9a-f]{32}$",
    }
    for boundary in (
        "not_admitted",
        "not_applied",
        "not_adopted",
        "not_deployed",
    ):
        assert cross_matrix_cell_selection_tool.output_schema["properties"][
            boundary
        ] == {"const": True}
    assert refinement_tool.policy.human_decision_gate == "none"
    assert refinement_tool.policy.effect_authorization == "none"
    assert set(refinement_tool.side_effects) == {
        "read:task",
        "read:dataset",
        "write:artifact",
    }
    assert "manual" in refinement_tool.input_schema["properties"]["method"]["enum"]
    assert manifest.version == "0.19.0"
    assert "refined univariate asset" in add_pool_tool.summary
    assert "automatic-tree leaf selection" in add_pool_tool.summary
    assert "Voting n-of-k candidate" in add_pool_tool.summary
    assert "Cross Matrix cell selection" in add_pool_tool.summary
    for pool_tool in pool_mutation_tools:
        assert pool_tool.policy.human_decision_gate == "none"
        assert pool_tool.policy.effect_authorization == "none"
        required = set(pool_tool.input_schema["required"])
        assert {
            "expected_pool_revision",
            "expected_pool_snapshot_hash",
        } <= required
    assert compile_pool_tool.policy.human_decision_gate == "none"
    assert compile_pool_tool.policy.effect_authorization == "none"
    assert measure_pool_impact_tool.determinism == "deterministic"
    assert measure_pool_impact_tool.policy.human_decision_gate == "none"
    assert measure_pool_impact_tool.policy.effect_authorization == "none"
    assert set(measure_pool_impact_tool.side_effects) == {
        "read:task",
        "read:dataset",
        "write:artifact",
    }
    assert measure_pool_impact_tool.input_schema["additionalProperties"] is False
    assert set(measure_pool_impact_tool.input_schema["required"]) == {
        "strategy_type",
        "expected_pool_revision",
        "expected_pool_snapshot_hash",
        "dataset_id",
        "expected_dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "target_col",
        "sample_design_ref",
        "comparison_mode",
    }
    assert measure_pool_impact_tool.input_schema["properties"]["strategy_type"] == {
        "type": "string",
        "enum": ["approval", "reject"],
    }
    assert measure_pool_impact_tool.output_schema["additionalProperties"] is False
    assert measure_pool_impact_tool.output_schema["properties"]["schema_version"] == {
        "const": "strategy.measure-pool-impact-tool.v2"
    }
    for boundary in ("not_created_strategy", "not_adopted", "not_deployed"):
        assert measure_pool_impact_tool.output_schema["properties"][boundary] == {
            "const": True
        }
    assert candidate_stability_tool.determinism == "deterministic"
    assert candidate_stability_tool.failure_policy == "fail"
    assert candidate_stability_tool.policy.human_decision_gate == "none"
    assert candidate_stability_tool.policy.effect_authorization == "none"
    assert set(candidate_stability_tool.side_effects) == {
        "read:task",
        "read:dataset",
        "write:artifact",
    }
    stability_inputs = candidate_stability_tool.input_schema["oneOf"]
    assert len(stability_inputs) == 2
    assert {
        branch["properties"]["source_kind"]["const"]
        for branch in stability_inputs
    } == {"univariate_asset", "pool_entry"}
    assert all(
        branch["additionalProperties"] is False
        and "dataset_id" not in branch["properties"]
        and "month_col" not in branch["properties"]
        and "sample_design_ref" not in branch["properties"]
        for branch in stability_inputs
    )
    assert candidate_stability_tool.output_schema["additionalProperties"] is False
    assert candidate_stability_tool.output_schema["properties"][
        "schema_version"
    ] == {"const": "strategy.measure-candidate-monthly-stability-tool.v1"}
    assert candidate_stability_tool.output_schema["properties"]["max_psi"] == {
        "type": "number",
        "minimum": 0,
    }
    for boundary in ("not_created_strategy", "not_adopted", "not_deployed"):
        assert candidate_stability_tool.output_schema["properties"][boundary] == {
            "const": True
        }
    assert measure_pool_validation_tool.determinism == "deterministic"
    assert measure_pool_validation_tool.failure_policy == "fail"
    assert measure_pool_validation_tool.policy.human_decision_gate == "none"
    assert measure_pool_validation_tool.policy.effect_authorization == "none"
    assert set(measure_pool_validation_tool.side_effects) == {
        "read:artifacts",
        "read:task",
        "read:dataset",
        "write:artifact",
    }
    assert measure_pool_validation_tool.input_schema["additionalProperties"] is False
    assert set(measure_pool_validation_tool.input_schema["required"]) == {
        "strategy_type",
        "pool_ref",
        "sample_design_ref",
        "partition",
        "population",
        "comparison_mode",
    }
    assert measure_pool_validation_tool.input_schema["properties"]["partition"] == {
        "type": "string",
        "enum": ["validation", "oot"],
    }
    assert measure_pool_validation_tool.input_schema["properties"]["population"] == {
        "const": "risk"
    }
    assert measure_pool_validation_tool.input_schema["properties"][
        "comparison_mode"
    ] == {"const": "absolute"}
    assert measure_pool_validation_tool.output_schema["additionalProperties"] is False
    assert measure_pool_validation_tool.output_schema["properties"][
        "schema_version"
    ] == {"const": "strategy.measure-pool-validation-tool.v1"}
    assert measure_pool_validation_tool.output_schema["properties"][
        "validation_status"
    ] == {"const": "independent_evidence"}
    for boundary in (
        "not_mutated_pool",
        "not_created_strategy",
        "not_adopted",
        "not_promoted",
        "not_deployed",
    ):
        assert measure_pool_validation_tool.output_schema["properties"][
            boundary
        ] == {"const": True}
    assert measure_impact_cube_tool.determinism == "deterministic"
    assert measure_impact_cube_tool.failure_policy == "fail"
    assert measure_impact_cube_tool.policy.human_decision_gate == "none"
    assert measure_impact_cube_tool.policy.effect_authorization == "none"
    assert set(measure_impact_cube_tool.side_effects) == {
        "read:artifacts",
        "read:task",
        "read:dataset",
        "read:strategy",
        "write:artifact",
    }
    assert measure_impact_cube_tool.input_schema["additionalProperties"] is False
    assert set(measure_impact_cube_tool.input_schema["required"]) == {
        "strategy_type",
        "pool_ref",
        "sample_design_ref",
        "partitions",
        "population",
        "dimension_bindings",
    }
    assert measure_impact_cube_tool.input_schema["properties"][
        "strategy_type"
    ]["enum"] == [
        "approval",
        "reject",
        "limit",
        "pricing",
        "segmentation",
    ]
    assert measure_impact_cube_tool.input_schema["properties"]["population"] == {
        "const": "risk"
    }
    assert measure_impact_cube_tool.input_schema["properties"]["partitions"][
        "uniqueItems"
    ] is True
    assert measure_impact_cube_tool.output_schema["additionalProperties"] is False
    assert measure_impact_cube_tool.output_schema["properties"][
        "schema_version"
    ] == {"const": "strategy.measure-impact-cube-tool.v3"}
    assert measure_impact_cube_tool.output_schema["properties"][
        "producer_run_ref"
    ] == {
        "type": "object",
        "properties": {
            "kind": {"const": "tool_run"},
            "ref_id": {
                "type": "string",
                "pattern": "^strategy-impact-cube-run-[0-9a-f]{24}$",
            },
            "content_hash": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
        },
        "required": ["kind", "ref_id", "content_hash"],
        "additionalProperties": False,
    }
    assert "producer_run_ref" in measure_impact_cube_tool.output_schema[
        "required"
    ]
    for boundary in (
        "not_mutated_pool",
        "not_created_strategy",
        "not_adopted",
        "not_promoted",
        "not_deployed",
    ):
        assert measure_impact_cube_tool.output_schema["properties"][
            boundary
        ] == {"const": True}
    assert set(run_monitoring_tool.side_effects) == {
        "read:task",
        "read:dataset",
        "read:strategy",
        "write:dataset",
        "write:strategy",
    }
    assert set(disposition_tool.side_effects) == {
        "read:task",
        "read:dataset",
        "read:strategy",
        "write:artifact",
        "write:dataset",
        "write:task",
        "write:strategy",
    }
    assert set(challenger_report_tool.side_effects) == {
        "read:dataset",
        "read:strategy",
        "write:artifact",
        "write:strategy",
    }

    valid_disposition = {
        "strategy_id": "strategy-1",
        "monitoring_run_id": "run-1",
        "expected_plan_id": "plan-1",
        "expected_plan_revision": 1,
        "expected_plan_hash": "a" * 64,
        "disposition": "observe",
        "reason": "继续观察一个周期",
        "threshold_patch": None,
    }
    validate_against_schema(
        valid_disposition,
        disposition_tool.input_schema,
        label="monitoring disposition input",
    )
    for invalid_reason in (None, ""):
        with pytest.raises(SchemaValidationError):
            validate_against_schema(
                {**valid_disposition, "reason": invalid_reason},
                disposition_tool.input_schema,
                label="monitoring disposition requires a non-empty reason",
            )

    # A missing champion intentionally degrades to a no-baseline report. Once
    # a champion is supplied, the persisted challenger backtest receipt is
    # mandatory because the tool reloads and recomputes evidence from it.
    validate_against_schema(
        {"strategy_id": "strategy-1"},
        challenger_report_tool.input_schema,
        label="challenger report without baseline",
    )
    validate_against_schema(
        {
            "strategy_id": "strategy-1",
            "champion_strategy_id": "strategy-0",
            "challenger_backtest": {"backtest_id": "backtest-1"},
        },
        challenger_report_tool.input_schema,
        label="evidence-bound challenger report",
    )
    with pytest.raises(SchemaValidationError):
        validate_against_schema(
            {
                "strategy_id": "strategy-1",
                "champion_strategy_id": "strategy-0",
            },
            challenger_report_tool.input_schema,
            label="challenger report missing persisted backtest receipt",
        )

    sample_design_ref = {
        "artifact_id": "a" * 64,
        "artifact_content_hash": "b" * 64,
        "sample_design_id": "sample-design-1",
        "sample_design_content_hash": "c" * 64,
        "partition": "development",
    }
    validate_against_schema(
        {
            "dataset_id": "dataset-1",
            "target_col": "bad",
            "sample_design_ref": sample_design_ref,
            "strategy_type": "segmentation",
            "candidate_design": {
                "method": "single_variable_segmentation",
                "feature_col": "score",
            },
            "candidate_policy_version": "strategy.candidate_policy.v1",
        },
        candidate_tool.input_schema,
        label="segmentation candidate input",
    )
    with pytest.raises(SchemaValidationError):
        validate_against_schema(
            {
                "dataset_id": "dataset-1",
                "target_col": "bad",
                "sample_design_ref": sample_design_ref,
                "strategy_type": "segmentation",
                "candidate_design": {
                    "method": "single_variable_segmentation",
                    "feature_col": "score",
                },
                "candidate_policy_version": "strategy.candidate_policy.v1",
                "strategy_spec": {},
            },
            candidate_tool.input_schema,
            label="caller-supplied candidate result",
        )

    validate_against_schema(
        {
            "strategy_type": "approval",
            "rules": [{"condition": "score < 600", "decision": "reject"}],
            "default_decision": "approve",
        },
        build_tool.input_schema,
        label="legacy strategy input",
    )
    validate_against_schema(
        {
            "strategy_spec": {
                "schema_version": "strategy.dsl.v1",
                "strategy_type": "approval",
                "match_policy": "first_match",
                "default_action": {"type": "approval"},
                "rules": [],
            }
        },
        build_tool.input_schema,
        label="canonical strategy input",
    )
    with pytest.raises(SchemaValidationError):
        validate_against_schema(
            {
                "strategy_type": "approval",
                "rules": [
                    {
                        "condition": "score < 600",
                        "decision": "reject",
                        "priority": 1.5,
                    }
                ],
                "default_decision": "approve",
            },
            build_tool.input_schema,
            label="fractional priority",
        )
    with pytest.raises(SchemaValidationError):
        validate_against_schema(
            {
                "strategy_spec": {
                    "strategy_type": "approval",
                    "default_action": {"type": "approval"},
                    "rules": [
                        {
                            "rule_id": "bad-expression",
                            "priority": 10,
                            "condition": {"op": "python_eval", "code": "score > 1"},
                            "action": {"type": "reject"},
                        }
                    ],
                }
            },
            build_tool.input_schema,
            label="unsupported canonical expression",
        )


def test_sample_bound_tool_manifests_use_one_exact_reference_schema(tmp_path):
    manifest = _real_builtin_registry(tmp_path).get("strategy")
    exact_ref = {
        "type": "object",
        "properties": {
            "artifact_id": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
            "artifact_content_hash": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
            "sample_design_id": {"type": "string", "minLength": 1},
            "sample_design_content_hash": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
            "partition": {"const": "development"},
        },
        "required": [
            "artifact_id",
            "artifact_content_hash",
            "sample_design_id",
            "sample_design_content_hash",
            "partition",
        ],
        "additionalProperties": False,
    }
    by_name = {tool.name: tool for tool in manifest.tools}
    required_input_tools = {
        "analyze_univariate_candidates",
        "build_automatic_tree_candidate",
        "measure_pool_impact",
        "design_strategy_candidate",
        "tradeoff_view",
        "design_cutoff_bands",
        "compare_strategies",
        "mine_rules",
        "evaluate_rule_set",
    }
    for name in required_input_tools:
        schema = by_name[name].input_schema
        assert "sample_design_ref" in schema["required"], name
        assert schema["properties"]["sample_design_ref"] == exact_ref, name

    # The generic backtest retains a direct legacy compatibility entrypoint,
    # but every V2 Workflow requires the slot and its runtime shape is still exact.
    backtest_schema = by_name["backtest_strategy"].input_schema
    assert "sample_design_ref" not in backtest_schema["required"]
    assert backtest_schema["properties"]["sample_design_ref"] == exact_ref

    for name in {
        "design_strategy_candidate",
        "tradeoff_view",
        "design_cutoff_bands",
        "compare_strategies",
        "mine_rules",
        "evaluate_rule_set",
        "limit_pricing_matrix",
        "build_voting_candidate",
    }:
        schema = by_name[name].output_schema
        assert "sample_design_ref" in schema["required"], name
        output_ref = schema["properties"]["sample_design_ref"]
        if "$ref" in output_ref:
            assert output_ref == {"$ref": "#/$defs/sample_design_ref"}
            assert schema["$defs"]["sample_design_ref"] == exact_ref
        else:
            assert output_ref == exact_ref, name


def test_build_strategy_tool_accepts_and_persists_canonical_dsl(tmp_path):
    runner, _, _, task = _runtime(tmp_path)
    spec = {
        "schema_version": "strategy.dsl.v1",
        "strategy_type": "segmentation",
        "match_policy": "first_match",
        "default_action": {"type": "segment", "value": "other"},
        "rules": [
            {
                "rule_id": "prime-vote",
                "priority": 10,
                "condition": {
                    "op": "n_of_k",
                    "n": 2,
                    "args": [
                        {
                            "op": "compare",
                            "field": "score",
                            "operator": ">=",
                            "value": 700,
                        },
                        {
                            "op": "compare",
                            "field": "income",
                            "operator": ">=",
                            "value": 10000,
                        },
                    ],
                },
                "action": {
                    "type": "segment",
                    "value": "prime",
                    "reason_code": "PRIME_VOTE",
                },
            }
        ],
        "metadata": {"lineage": {"source_artifact": "rules-1"}},
    }

    built = runner.invoke(
        ToolRef("strategy", "build_strategy"),
        {"strategy_spec": spec, "description": "canonical candidate"},
        task_id=task.id,
    )

    assert built.ok is True, built.error
    assert built.output["dsl_schema_version"] == "strategy.dsl.v1"
    assert built.output["strategy_spec"]["rules"][0]["rule_id"] == "prime-vote"
    strategy = StrategyRepository(
        build_settings(tmp_path / "workspace").db_path
    ).get_strategy(built.output["strategy_id"])
    assert strategy is not None
    assert strategy.spec.to_dict() == built.output["strategy_spec"]


def test_build_strategy_persistence_identity_is_task_scoped_and_payload_exact(
    tmp_path,
) -> None:
    runner, _, _, task = _runtime(tmp_path)
    settings = build_settings(tmp_path / "workspace")
    spec = {
        "strategy_type": "approval",
        "default_action": {"type": "approval"},
        "rules": [
            {
                "rule_id": "reject-low",
                "priority": 10,
                "condition": {
                    "op": "compare",
                    "field": "score",
                    "operator": "<",
                    "value": 600,
                },
                "action": {"type": "reject"},
            }
        ],
    }
    second_task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="另一策略任务",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "source-2"),
            algorithm="lr",
            run_mode="agent",
        )
    )

    first = runner.invoke(
        ToolRef("strategy", "build_strategy"),
        {"strategy_spec": spec, "description": "方案 A"},
        task_id=task.id,
    )
    retry = runner.invoke(
        ToolRef("strategy", "build_strategy"),
        {"strategy_spec": spec, "description": "方案 A"},
        task_id=task.id,
    )
    renamed = runner.invoke(
        ToolRef("strategy", "build_strategy"),
        {"strategy_spec": spec, "description": "方案 B"},
        task_id=task.id,
    )
    other_task = runner.invoke(
        ToolRef("strategy", "build_strategy"),
        {"strategy_spec": spec, "description": "方案 A"},
        task_id=second_task.id,
    )

    assert all(result.ok for result in (first, retry, renamed, other_task))
    assert retry.output["strategy_id"] == first.output["strategy_id"]
    assert renamed.output["strategy_id"] != first.output["strategy_id"]
    assert other_task.output["strategy_id"] != first.output["strategy_id"]
    repository = StrategyRepository(settings.db_path)
    assert [item.description for item in repository.list_for_task(task.id)] == [
        "方案 A",
        "方案 B",
    ]
    assert [item.description for item in repository.list_for_task(second_task.id)] == [
        "方案 A"
    ]
    cross_task_backtest = runner.invoke(
        ToolRef("strategy", "backtest_strategy"),
        {
            "strategy_id": other_task.output["strategy_id"],
            "dataset_id": "not-needed-before-strategy-authorization",
            "target_col": "bad",
        },
        task_id=task.id,
    )
    assert cross_task_backtest.ok is False
    assert "strategy not found" in cross_task_backtest.error


@pytest.mark.slow
def test_strategy_pack_tools_round_trip_via_runner(tmp_path):
    runner, _plugin_registry, registry, task = _runtime(tmp_path)
    dataset = _register_strategy_sample(registry, tmp_path, task.id)
    sample_design_ref = materialize_strategy_tool_sample_design(
        build_settings(tmp_path / "workspace"),
        task,
        dataset,
    )
    params = {
        "annual_rate": 0.12,
        "funding_rate": 0.03,
        "lgd": 0.5,
        "operating_cost_per_loan": 10.0,
        "term_months": 6,
    }

    vintage = runner.invoke(
        ToolRef("strategy", "vintage_curve"),
        {
            "dataset_id": dataset.id,
            "cohort_col": "cohort",
            "mob_col": "mob",
            "bad_col": "bad",
            "mob_max": 2,
            "label_semantics": "incremental",
        },
        task_id=task.id,
    )
    roll = runner.invoke(
        ToolRef("strategy", "roll_rate_matrix"),
        {
            "dataset_id": dataset.id,
            "id_col": "customer_id",
            "time_col": "month",
            "status_col": "status",
            "states": ["C", "M1", "M3+"],
        },
        task_id=task.id,
    )
    profit = runner.invoke(
        ToolRef("strategy", "profit_calc"),
        {
            "dataset_id": dataset.id,
            "segment_col": "segment",
            "ead_col": "ead",
            "pd_col": "pd",
            "params": params,
        },
        task_id=task.id,
    )
    built = runner.invoke(
        ToolRef("strategy", "build_strategy"),
        {
            "strategy_type": "approval",
            "rules": [{"condition": "score < 600", "decision": "reject"}],
            "score_col": "score",
            "default_decision": "approve",
            "description": "reject low scores",
        },
        task_id=task.id,
    )

    assert vintage.ok is True, vintage.error
    assert vintage.output["cohorts"] == ["2026-01", "2026-02", "2026-03"]
    assert vintage.output["summary"]["trend"] in {
        "deteriorating",
        "stable",
        "improving",
    }
    assert roll.ok is True, roll.error
    assert roll.output["base_counts"] == {"C": 2, "M1": 0, "M3+": 1}
    assert roll.output["observation_semantics"] == "adjacent_observation"
    assert roll.output["period"] == "month"
    assert {item["kind"] for item in roll.output["artifacts"]} == {
        "roll_rate_csv",
        "roll_rate_markdown",
    }
    assert all(
        item.get("artifact_id") and "path" not in item
        for item in roll.output["artifacts"]
    )
    assert profit.ok is True, profit.error
    assert {row["segment"] for row in profit.output["results"]} == {"A", "B"}
    assert profit.output["source_evidence"]["dataset_content_hash"]
    assert {item["kind"] for item in profit.output["artifacts"]} == {
        "profit_csv",
        "profit_markdown",
    }
    assert all(
        item.get("artifact_id") and "path" not in item
        for item in profit.output["artifacts"]
    )
    task_artifacts = TaskArtifactRepository(
        build_settings(tmp_path / "workspace").db_path
    ).list_for_task(task.id)
    assert {record["kind"] for record in task_artifacts} == {
        "strategy_sample_design_json",
        "roll_rate_csv",
        "roll_rate_markdown",
        "profit_csv",
        "profit_markdown",
    }
    assert {record["id"] for record in task_artifacts} == {
        item["artifact_id"]
        for item in [*roll.output["artifacts"], *profit.output["artifacts"]]
    } | {sample_design_ref["artifact_id"]}
    assert built.ok is True, built.error
    assert built.output["strategy_id"]
    assert built.output["dsl_schema_version"] == "strategy.dsl.v1"
    assert built.output["strategy_spec"]["rules"][0]["rule_id"]
    assert (
        built.output["rules"][0]["rule_id"]
        == built.output["strategy_spec"]["rules"][0]["rule_id"]
    )

    backtest = runner.invoke(
        ToolRef("strategy", "backtest_strategy"),
        {
            "dataset_id": dataset.id,
            "strategy_id": built.output["strategy_id"],
            "target_col": "bad",
            "profit_params": params,
            "ead_col": "ead",
            "pd_col": "pd",
        },
        task_id=task.id,
    )
    tradeoff = runner.invoke(
        ToolRef("strategy", "tradeoff_view"),
        {
            "dataset_id": dataset.id,
            "score_col": "score",
            "target_col": "bad",
            "sample_design_ref": sample_design_ref,
            "cutoffs": [600, 700],
            "profit_params": params,
            "ead_col": "ead",
            "pd_col": "pd",
            "max_bad_rate": 0.7,
            "objective": "max_profit",
        },
        task_id=task.id,
    )

    assert backtest.ok is True, backtest.error
    assert backtest.output["backtest_id"]
    assert backtest.output["approval_rate"] == pytest.approx(4 / 6)
    assert "by_segment" in backtest.output
    strategy_audits = PluginRepository(
        build_settings(tmp_path / "workspace").db_path
    ).list_audit()
    assert any(
        audit["kind"] == "strategy.create"
        and audit["target_ref"] == built.output["strategy_id"]
        for audit in strategy_audits
    )
    assert any(
        audit["kind"] == "strategy.backtest"
        and audit["target_ref"] == backtest.output["backtest_id"]
        for audit in strategy_audits
    )
    assert tradeoff.ok is True, tradeoff.error
    assert [point["cutoff"] for point in tradeoff.output["points"]] == [600.0, 700.0]
    assert tradeoff.output["recommended"]["cutoff"] in {600.0, 700.0}


def test_roll_rate_matrix_tool_surfaces_balance_weighting_and_warnings(tmp_path):
    # DOM-8: balance_col weights transitions; a missing-month gap for one id
    # surfaces as a data_quality_warnings entry through the tool boundary.
    runner, _plugin_registry, registry, task = _runtime(tmp_path)
    frame = pd.DataFrame(
        {
            "customer_id": ["A", "A", "B", "B"],
            "month": ["202601", "202603", "202601", "202602"],
            "status": ["C", "M1", "C", "C"],
            "balance": [100.0, 100.0, 300.0, 300.0],
        }
    )
    path = tmp_path / "roll_rate_balance_sample.parquet"
    frame.to_parquet(path, index=False)
    dataset = registry.register_existing(path, task_id=task.id, role="strategy_sample")

    result = runner.invoke(
        ToolRef("strategy", "roll_rate_matrix"),
        {
            "dataset_id": dataset.id,
            "id_col": "customer_id",
            "time_col": "month",
            "status_col": "status",
            "states": ["C", "M1"],
            "balance_col": "balance",
        },
        task_id=task.id,
    )

    assert result.ok is True, result.error
    assert result.output["base_counts"] == {"C": 400.0, "M1": 0.0}
    assert len(result.output["data_quality_warnings"]) == 1
    assert result.output["data_quality_warnings"][0]["id"] == "A"


def _register_strategy_sample_with_nan_label(registry, tmp_path, task_id: str):
    frame = pd.DataFrame(
        {
            "bad": [1.0, 0.0, float("nan"), 0.0, 1.0, 0.0],
            "score": [580, 620, 730, 760, 590, 800],
        }
    )
    path = tmp_path / "strategy_nan_sample.parquet"
    frame.to_parquet(path, index=False)
    return registry.register_existing(path, task_id=task_id, role="strategy_sample")


def test_tradeoff_view_gates_nan_label(tmp_path):
    runner, _plugin_registry, registry, task = _runtime(tmp_path)
    dataset = _register_strategy_sample_with_nan_label(registry, tmp_path, task.id)
    settings = build_settings(tmp_path / "workspace")
    with pytest.raises(NanLabelNotConfirmedError, match="1/6 NaN labels"):
        materialize_strategy_tool_sample_design(
            settings,
            task,
            dataset,
            drop_nan_labels=False,
        )
    dropping_sample_ref = materialize_strategy_tool_sample_design(
        settings,
        task,
        dataset,
        drop_nan_labels=True,
    )
    base_inputs = {
        "dataset_id": dataset.id,
        "score_col": "score",
        "target_col": "bad",
        "sample_design_ref": dropping_sample_ref,
        "cutoffs": [600, 700],
        "drop_nan_labels": True,
    }
    confirmed = runner.invoke(
        ToolRef("strategy", "tradeoff_view"),
        base_inputs,
        task_id=task.id,
    )
    assert confirmed.ok is True, confirmed.error
    assert confirmed.output["nan_labels_dropped"] == 1


def test_vintage_curve_gates_nan_label(tmp_path):
    runner, _plugin_registry, registry, task = _runtime(tmp_path)
    frame = pd.DataFrame(
        {
            "cohort": ["202601", "202601", "202602"],
            "mob": [0, 1, 0],
            "bad": [0.0, float("nan"), 1.0],
        }
    )
    path = tmp_path / "vintage_nan_sample.parquet"
    frame.to_parquet(path, index=False)
    dataset = registry.register_existing(path, task_id=task.id, role="strategy_sample")
    base_inputs = {
        "dataset_id": dataset.id,
        "cohort_col": "cohort",
        "mob_col": "mob",
        "bad_col": "bad",
    }

    # The NaN-label gate fires first (label resolution precedes the semantics check),
    # even though label_semantics is also undeclared here.
    blocked = runner.invoke(
        ToolRef("strategy", "vintage_curve"),
        {**base_inputs, "label_semantics": "incremental"},
        task_id=task.id,
    )
    assert blocked.ok is False
    assert blocked.error_kind == "nan_label_not_confirmed"
    assert blocked.error_detail["n_nan"] == 1

    confirmed = runner.invoke(
        ToolRef("strategy", "vintage_curve"),
        {**base_inputs, "drop_nan_labels": True, "label_semantics": "incremental"},
        task_id=task.id,
    )
    assert confirmed.ok is True, confirmed.error
    assert confirmed.output["nan_labels_dropped"] == 1
    assert "warnings" in confirmed.output


def test_tool_vintage_curve_raises_label_semantics_not_declared(tmp_path):
    runner, _plugin_registry, registry, task = _runtime(tmp_path)
    # Snapshot-flag frame with clean 0/1 labels: no NaN gate, so the undeclared
    # label_semantics gate is what fires. 3 MOBs with non-decreasing bad_count so the
    # monotone snapshot heuristic also trips (advisory hint in the gate detail).
    frame = pd.DataFrame(
        {
            "cohort": ["202601"] * 12,
            "mob": [0, 0, 0, 0] + [1, 1, 1, 1] + [2, 2, 2, 2],
            "bad": [1, 0, 0, 0] + [1, 1, 0, 0] + [1, 1, 0, 0],
        }
    )
    path = tmp_path / "vintage_semantics_sample.parquet"
    frame.to_parquet(path, index=False)
    dataset = registry.register_existing(path, task_id=task.id, role="strategy_sample")

    blocked = runner.invoke(
        ToolRef("strategy", "vintage_curve"),
        {
            "dataset_id": dataset.id,
            "cohort_col": "cohort",
            "mob_col": "mob",
            "bad_col": "bad",
        },
        task_id=task.id,
    )
    assert blocked.ok is False
    assert blocked.error_kind == "label_semantics_not_declared"
    detail = blocked.error_detail
    assert detail["target_col"] == "bad"
    assert set(detail["examples"]) == {"incremental", "snapshot"}
    assert detail["monotone_heuristic"] is True


def test_tool_vintage_curve_surfaces_warnings_in_output(tmp_path):
    runner, _plugin_registry, registry, task = _runtime(tmp_path)
    # Monotone snapshot-flag data declared incremental -> the tool output carries a
    # non-empty 'warnings' list (schema additionalProperties:false compliance).
    frame = pd.DataFrame(
        {
            "cohort": ["202601"] * 12,
            "mob": [0, 0, 0, 0] + [1, 1, 1, 1] + [2, 2, 2, 2],
            "bad": [1, 0, 0, 0] + [1, 1, 0, 0] + [1, 1, 0, 0],
        }
    )
    path = tmp_path / "vintage_warnings_sample.parquet"
    frame.to_parquet(path, index=False)
    dataset = registry.register_existing(path, task_id=task.id, role="strategy_sample")

    result = runner.invoke(
        ToolRef("strategy", "vintage_curve"),
        {
            "dataset_id": dataset.id,
            "cohort_col": "cohort",
            "mob_col": "mob",
            "bad_col": "bad",
            "label_semantics": "incremental",
        },
        task_id=task.id,
    )
    assert result.ok is True, result.error
    assert isinstance(result.output["warnings"], list)
    assert any(
        "snapshot" in w.lower() or "快照" in w for w in result.output["warnings"]
    )
