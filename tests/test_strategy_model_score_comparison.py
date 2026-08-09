from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from marvis.db import PluginRepository, TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.orchestrator.planner import Planner
from marvis.orchestrator.templates import get_template, load_builtin_templates
from marvis.orchestrator.templates.sample import BUILTIN_TEMPLATES
from marvis.orchestrator.validator import PlanValidator
from marvis.packs.modeling.score_evidence_tools import (
    MODEL_SCORE_EVIDENCE_ARTIFACT_KIND,
    MODEL_SCORE_VECTOR_ARTIFACT_KIND,
)
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy.model_score_comparison_tools import (
    MODEL_SCORE_COMPARISON_V2_ARTIFACT_KIND,
    model_score_comparison_registry_snapshot_token,
    run_materialize_model_score_comparison_v2,
    validate_materialize_model_score_comparison_v2_tool_output,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.plugins.loader import load_builtin_packs, load_manifest
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.schema_validation import validate_against_schema
from marvis.repositories.task_artifacts import TaskArtifactRepository
from tests.test_model_score_evidence_tool import _run_score
from tests.test_modeling_training_evidence_tool import (
    _fixture,
    _run as run_training,
)


def _score_ref(output: dict) -> dict[str, str]:
    return {
        "evidence_artifact_id": output["artifacts"]["score_evidence"][
            "artifact_id"
        ],
        "expected_evidence_artifact_content_hash": output["artifacts"][
            "score_evidence"
        ]["content_hash"],
        "score_vector_artifact_id": output["artifacts"]["score_vector"][
            "artifact_id"
        ],
        "expected_score_vector_artifact_content_hash": output["artifacts"][
            "score_vector"
        ]["content_hash"],
    }


def _comparison_fixture(tmp_path: Path) -> tuple[dict, dict]:
    fx = _fixture(tmp_path)
    first_training = run_training(fx)
    first_score = _run_score(fx, first_training)
    fx["inputs"] = deepcopy(fx["inputs"])
    fx["inputs"]["seed"] = 47
    second_training = run_training(fx)
    second_score = _run_score(fx, second_training)
    artifacts = TaskArtifactRepository(fx["settings"].db_path).list_for_task(
        fx["task"].id
    )
    inputs = {
        "sample_design_ref": fx["sample_ref"],
        "model_score_evidence_refs": [
            _score_ref(first_score),
            _score_ref(second_score),
        ],
        "population": "risk",
        "partition": "development",
        "expected_registry_token": (
            model_score_comparison_registry_snapshot_token(artifacts)
        ),
    }
    return fx, inputs


@pytest.fixture(scope="module")
def comparison_case(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, dict]:
    return _comparison_fixture(tmp_path_factory.mktemp("model-score-comparison"))


def test_tool_materializes_authenticated_comparison_without_selecting_winner(
    comparison_case: tuple[dict, dict],
) -> None:
    fx, inputs = comparison_case

    output = strategy_tools.tool_materialize_model_score_comparison_v2(
        inputs,
        fx["ctx"],
    )

    assert output["population"] == "risk"
    assert output["partition"] == "development"
    assert output["comparison"]["selection"] == {
        "status": "no_selection",
        "selected_model_evidence_ref": None,
        "metric_key": None,
        "period": None,
        "direction": None,
        "reason": "comparison_evidence_does_not_authorize_selection",
    }
    assert output["governance"] == {
        "selection_status": "no_selection",
        "winner_selected": False,
        "not_adopted": True,
        "not_deployed": True,
    }
    assert len(output["comparison"]["model_evidence_refs"]) == 2
    assert output["comparison"]["metrics"]
    manifest = load_manifest(
        Path(__file__).parents[1] / "marvis" / "packs" / "strategy",
        builtin=True,
    )
    spec = next(
        tool
        for tool in manifest.tools
        if tool.name == "materialize_model_score_comparison_v2"
    )
    validate_against_schema(
        output,
        spec.output_schema,
        label="materialize_model_score_comparison_v2 output",
    )
    assert (
        validate_materialize_model_score_comparison_v2_tool_output(
            output,
            runtime=fx["runtime"],
            task_id=fx["task"].id,
        )
        == output
    )

    records = TaskArtifactRepository(fx["settings"].db_path).list_for_task(
        fx["task"].id
    )
    comparison_records = [
        record
        for record in records
        if record["kind"] == MODEL_SCORE_COMPARISON_V2_ARTIFACT_KIND
    ]
    assert len(comparison_records) == 1
    record = comparison_records[0]
    assert record["id"] == output["artifact"]["artifact_id"]
    assert record["content_hash"] == output["artifact"]["content_hash"]
    document = json.loads(Path(record["path"]).read_text(encoding="utf-8"))
    assert document["comparison"] == output["comparison"]
    assert document["governance"]["winner_selected"] is False
    assert {
        item["kind"]
        for item in records
        if item["id"]
        in {
            ref["evidence_artifact_id"]
            for ref in inputs["model_score_evidence_refs"]
        }
        | {
            ref["score_vector_artifact_id"]
            for ref in inputs["model_score_evidence_refs"]
        }
    } == {MODEL_SCORE_EVIDENCE_ARTIFACT_KIND, MODEL_SCORE_VECTOR_ARTIFACT_KIND}


def test_tool_rejects_registry_drift_without_publishing_artifact(
    comparison_case: tuple[dict, dict],
) -> None:
    fx, base_inputs = comparison_case
    inputs = deepcopy(base_inputs)
    inputs["expected_registry_token"] = "f" * 64
    assert inputs["expected_registry_token"] != (
        model_score_comparison_registry_snapshot_token(
            TaskArtifactRepository(fx["settings"].db_path).list_for_task(
                fx["task"].id
            )
        )
    )
    before = [
        record["id"]
        for record in TaskArtifactRepository(
            fx["settings"].db_path
        ).list_for_task(fx["task"].id)
        if record["kind"] == MODEL_SCORE_COMPARISON_V2_ARTIFACT_KIND
    ]

    with pytest.raises(StrategyError, match="registry snapshot changed"):
        run_materialize_model_score_comparison_v2(
            inputs,
            fx["ctx"],
            fx["runtime"],
        )

    assert [
        record["id"]
        for record in TaskArtifactRepository(
            fx["settings"].db_path
        ).list_for_task(fx["task"].id)
        if record["kind"] == MODEL_SCORE_COMPARISON_V2_ARTIFACT_KIND
    ] == before


def test_tool_reauthenticates_score_hash_and_rejects_caller_selection(
    comparison_case: tuple[dict, dict],
) -> None:
    fx, inputs = comparison_case
    forged_hash = deepcopy(inputs)
    forged_hash["model_score_evidence_refs"][0][
        "expected_evidence_artifact_content_hash"
    ] = "e" * 64
    with pytest.raises(StrategyError, match="binding|hash|not found"):
        run_materialize_model_score_comparison_v2(
            forged_hash,
            fx["ctx"],
            fx["runtime"],
        )

    caller_selected = deepcopy(inputs)
    caller_selected["selected_model_evidence_ref"] = {"evidence_id": "forged"}
    with pytest.raises(StrategyError, match="unsupported"):
        run_materialize_model_score_comparison_v2(
            caller_selected,
            fx["ctx"],
            fx["runtime"],
        )


def test_tool_rejects_cross_task_score_evidence_refs(
    comparison_case: tuple[dict, dict],
) -> None:
    fx, inputs = comparison_case
    other_task = TaskRepository(fx["settings"].db_path).create_task(
        TaskCreate(
            model_name="other-task",
            model_version="dev",
            validator="qa",
            source_dir=str(fx["settings"].workspace.parent / "other-source"),
            task_type="strategy",
        )
    )
    other_ctx = replace(fx["ctx"], task_id=other_task.id)

    with pytest.raises(
        StrategyError,
        match="dispatch binding|not found|another task",
    ):
        strategy_tools.tool_materialize_model_score_comparison_v2(
            inputs,
            other_ctx,
        )

    assert not [
        record
        for record in TaskArtifactRepository(
            fx["settings"].db_path
        ).list_for_task(other_task.id)
        if record["kind"] == MODEL_SCORE_COMPARISON_V2_ARTIFACT_KIND
    ]


def test_exact_replay_is_idempotent_and_cached_winner_tamper_is_rejected(
    comparison_case: tuple[dict, dict],
) -> None:
    fx, inputs = comparison_case
    first = run_materialize_model_score_comparison_v2(
        inputs,
        fx["ctx"],
        fx["runtime"],
    )

    replay = run_materialize_model_score_comparison_v2(
        inputs,
        fx["ctx"],
        fx["runtime"],
    )

    assert replay == first
    records = TaskArtifactRepository(fx["settings"].db_path).list_for_task(
        fx["task"].id
    )
    assert sum(
        record["kind"] == MODEL_SCORE_COMPARISON_V2_ARTIFACT_KIND
        for record in records
    ) == 1
    forged = deepcopy(first)
    forged["governance"]["winner_selected"] = True
    with pytest.raises(StrategyError, match="drifted"):
        validate_materialize_model_score_comparison_v2_tool_output(
            forged,
            runtime=fx["runtime"],
            task_id=fx["task"].id,
        )


def _tool_registry(tmp_path: Path) -> ToolRegistry:
    db_path = tmp_path / "plugins.sqlite"
    init_db(db_path)
    plugins = PluginRegistry(PluginRepository(db_path))
    load_builtin_packs(
        plugins,
        Path(__file__).parents[1] / "marvis" / "packs",
    )
    return ToolRegistry(plugins)


def test_manifest_and_builtin_template_register_closed_nonselecting_tool(
    tmp_path: Path,
) -> None:
    tools = _tool_registry(tmp_path)
    spec = tools.resolve(ToolRef("strategy", "materialize_model_score_comparison_v2"))
    assert spec.entrypoint == "tool_materialize_model_score_comparison_v2"
    assert spec.determinism == "deterministic"
    assert spec.policy.human_decision_gate == "none"
    assert spec.policy.effect_authorization == "none"
    assert {"read:task", "read:dataset", "read:model", "write:artifact"} <= set(
        spec.side_effects
    )
    assert set(spec.input_schema["required"]) == {
        "sample_design_ref",
        "model_score_evidence_refs",
        "population",
        "partition",
        "expected_registry_token",
    }
    assert spec.input_schema["additionalProperties"] is False
    refs = spec.input_schema["properties"]["model_score_evidence_refs"]
    assert refs["minItems"] == 2
    assert refs["maxItems"] == 100
    assert spec.output_schema["properties"]["governance"] == {
        "$ref": "#/$defs/governance"
    }
    governance = spec.output_schema["$defs"]["governance"]
    assert governance["properties"]["selection_status"] == {
        "const": "no_selection"
    }
    assert governance["properties"]["winner_selected"] == {"const": False}

    load_builtin_templates()
    template = get_template("strategy_model_score_comparison_v2")
    assert template in BUILTIN_TEMPLATES
    assert {slot.name for slot in template.slots} == {
        "sample_design_ref",
        "model_score_evidence_refs",
        "population",
        "partition",
        "expected_registry_token",
    }
    assert all(slot.required for slot in template.slots)
    [step] = template.steps
    assert step.tool_ref == ToolRef(
        "strategy",
        "materialize_model_score_comparison_v2",
    )
    assert step.needs_confirmation is False
    assert step.decision_point is False

    sample_ref = {
        "membership_artifact_id": "1" * 64,
        "expected_membership_artifact_content_hash": "2" * 64,
        "bundle_artifact_id": "3" * 64,
        "expected_bundle_artifact_content_hash": "4" * 64,
        "expected_bundle_id": "strategy-sample-design-bundle-" + "5" * 24,
        "expected_sample_design_id": "strategy-sample-design-" + "6" * 24,
        "expected_sample_design_content_hash": "7" * 64,
    }
    refs_input = [
        {
            "evidence_artifact_id": "8" * 64,
            "expected_evidence_artifact_content_hash": "9" * 64,
            "score_vector_artifact_id": "a" * 64,
            "expected_score_vector_artifact_content_hash": "b" * 64,
        },
        {
            "evidence_artifact_id": "c" * 64,
            "expected_evidence_artifact_content_hash": "d" * 64,
            "score_vector_artifact_id": "e" * 64,
            "expected_score_vector_artifact_content_hash": "f" * 64,
        },
    ]
    plan = Planner(tools, lambda: None, PlanValidator(tools)).from_template(
        template,
        {
            "sample_design_ref": sample_ref,
            "model_score_evidence_refs": refs_input,
            "population": "risk",
            "partition": "development",
            "expected_registry_token": "f" * 64,
        },
        task_id="task-1",
    )
    assert PlanValidator(tools).validate(plan) == []
    assert plan.steps[0].inputs == {
        "sample_design_ref": sample_ref,
        "model_score_evidence_refs": refs_input,
        "population": "risk",
        "partition": "development",
        "expected_registry_token": "f" * 64,
    }
