from pathlib import Path

import pytest

from marvis.db import PluginRepository, init_db
from marvis.orchestrator.planner import Planner
from marvis.orchestrator.templates import get_template, load_builtin_templates
from marvis.orchestrator.validator import PlanValidator
from marvis.plugins.errors import SchemaValidationError
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.schema_validation import validate_against_schema


_CONSUMERS = (
    "tune_hyperparameters",
    "train_model",
    "train_models",
    "train_model_with_evidence_v2",
)
_REF = {"artifact_id": "selection-evidence-1", "content_hash": "a" * 64}


@pytest.fixture(scope="module")
def tool_registry(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("selection-workflow") / "app.sqlite"
    init_db(db_path)
    plugins = PluginRegistry(PluginRepository(db_path))
    load_builtin_packs(plugins, Path(__file__).parents[1] / "marvis" / "packs")
    return ToolRegistry(plugins)


def _training_inputs(tool_name):
    if tool_name == "train_model_with_evidence_v2":
        return {
            "sample_design_ref": {
                "membership_artifact_id": "a" * 64,
                "expected_membership_artifact_content_hash": "b" * 64,
                "bundle_artifact_id": "c" * 64,
                "expected_bundle_artifact_content_hash": "d" * 64,
                "expected_bundle_id": "bundle-1",
                "expected_sample_design_id": "sample-1",
                "expected_sample_design_content_hash": "e" * 64,
            },
            "recipe": "lr",
            "features": ["x"],
            "params": {},
            "seed": 7,
            "early_stopping_rounds": None,
        }
    inputs = {
        "dataset_id": "dataset-1",
        "features": ["x"],
        "target_col": "y",
        "split_col": "split",
        "split_values": {"train": "train", "test": "test", "oot": "oot"},
        "seed": 7,
    }
    inputs["recipes" if tool_name == "train_models" else "recipe"] = (
        ["lr"] if tool_name == "train_models" else "lr"
    )
    return inputs


@pytest.mark.parametrize("tool_name", _CONSUMERS)
def test_selection_refs_are_optional_bounded_tool_inputs(tool_registry, tool_name):
    schema = tool_registry.resolve(ToolRef("modeling", tool_name)).input_schema
    inputs = _training_inputs(tool_name)
    validate_against_schema(inputs, schema, label="inputs")
    for refs in ([], [_REF], [_REF] * 16):
        validate_against_schema(
            {**inputs, "selection_evidence_refs": refs}, schema, label="inputs"
        )


@pytest.mark.parametrize("tool_name", _CONSUMERS)
def test_selection_refs_reject_unbound_or_malformed_references(tool_registry, tool_name):
    schema = tool_registry.resolve(ToolRef("modeling", tool_name)).input_schema
    invalid_refs = (
        None,
        _REF,
        ["selection-evidence-1"],
        [{}],
        [{"artifact_id": "selection-evidence-1"}],
        [{"content_hash": "a" * 64}],
        [{**_REF, "artifact_id": ""}],
        [{**_REF, "artifact_id": 1}],
        [{**_REF, "content_hash": "A" * 64}],
        [{**_REF, "content_hash": "g" * 64}],
        [{**_REF, "content_hash": "a" * 63}],
        [{**_REF, "content_hash": "a" * 64 + "\n"}],
        [{**_REF, "assurance": "verified"}],
        [_REF] * 17,
    )
    for refs in invalid_refs:
        with pytest.raises(SchemaValidationError, match="selection_evidence_refs"):
            validate_against_schema(
                {**_training_inputs(tool_name), "selection_evidence_refs": refs},
                schema,
                label="inputs",
            )


@pytest.mark.parametrize(
    ("template_id", "producer_titles", "consumer_titles"),
    (
        ("standard_modeling", ("筛选特征",), ("训练模型",)),
        ("modeling", ("特征筛选", "精选特征"), ("调参", "训练模型")),
        ("modeling_with_join", ("特征筛选", "精选特征"), ("调参", "训练模型")),
    ),
)
def test_instantiated_workflows_forward_exact_selection_refs(
    tool_registry, template_id, producer_titles, consumer_titles
):
    load_builtin_templates()
    validator = PlanValidator(tool_registry)
    planner = Planner(tool_registry, lambda: None, validator)
    plan = planner.from_template(
        get_template(template_id),
        {
            "dataset_id": "dataset-1",
            "anchor_id": "anchor-1",
            "feature_ids": ["feature-1"],
            "feature_cols": ["x"],
            "target_col": "y",
            "split_col": "split",
            "split_values": {"train": "train", "test": "test", "oot": "oot"},
            "recipe": "lr",
            "recipes": ["lr"],
            "seed": 7,
        },
        task_id="task-1",
    )
    assert validator.validate(plan) == []
    steps = {step.title: step for step in plan.steps}
    expected = [
        f"$ref:{steps[title].id}.output.selection_evidence_ref"
        for title in producer_titles
    ]
    for title in consumer_titles:
        consumer = steps[title]
        assert consumer.inputs["selection_evidence_refs"] == expected
        assert {steps[producer].id for producer in producer_titles} <= set(consumer.depends_on)
        # The consumer schema must accept the object values the executor obtains
        # from these nested output references, not just unresolved strings.
        resolved_refs = [
            {**_REF, "artifact_id": steps[producer].id}
            for producer in producer_titles
        ]
        input_schema = tool_registry.resolve(consumer.tool_ref).input_schema
        validate_against_schema(
            {**_training_inputs(consumer.tool_ref.tool), "selection_evidence_refs": resolved_refs},
            input_schema,
            label="inputs",
        )


@pytest.mark.parametrize("template_id", ["modeling", "modeling_with_join"])
def test_fold_workflows_freeze_full_candidates_and_forward_final_results(template_id):
    load_builtin_templates()
    steps = {step.title: step for step in get_template(template_id).steps}
    tune = steps["调参"].inputs_template
    train = steps["训练模型"].inputs_template
    assert steps["配置调参"].inputs_template["cv_folds"] == 3
    assert tune["fold_selection"] == {
        "source_dataset_id": "$ref:切分样本.output.result_dataset_id",
        "candidates": "$ref:选择建模规格.output.feature_cols",
    }
    assert tune["cv_folds"] == "$ref:配置调参.output.cv_folds"
    assert train["features_by_recipe"] == "$ref:调参.output.features_by_recipe"
    assert train["fold_tuning_evidence_refs"] == "$ref:调参.output.fold_tuning_evidence_refs"
    assert train["seed"] == tune["seed"]
    assert train["recipes"] == tune["recipes"]
