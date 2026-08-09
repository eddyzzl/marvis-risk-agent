from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from marvis.agent import renderers
from marvis.agent.gate_adapters import render_gate_dependencies
from marvis.agent.plan_message_composer import PlanMessageComposer
from marvis.agent.presenters.runtime import build_trusted_presenter_runtime
from marvis.orchestrator.contracts import StepStatus
from marvis.routers import validation_agent


@pytest.mark.parametrize(
    ("tool", "runtime_type"),
    [
        ("train_model_with_evidence_v2", "_Runtime"),
        ("search_cross_matrix_candidates", "_Runtime"),
    ],
)
def test_trusted_presenter_runtime_is_built_from_internal_workspace(
    tmp_path: Path,
    tool: str,
    runtime_type: str,
) -> None:
    runtime = build_trusted_presenter_runtime(
        tool,
        workspace=tmp_path,
        task_id="task-1",
    )

    assert type(runtime).__name__ == runtime_type
    assert runtime.settings.workspace == tmp_path.resolve()
    assert runtime.datasets_root == tmp_path.resolve() / "datasets"


def test_trusted_presenter_runtime_rejects_untrusted_or_unknown_context(
    tmp_path: Path,
) -> None:
    assert (
        build_trusted_presenter_runtime(
            "unknown_tool",
            workspace=tmp_path,
            task_id="task-1",
        )
        is None
    )
    assert (
        build_trusted_presenter_runtime(
            "train_model_with_evidence_v2",
            workspace=None,
            task_id="task-1",
        )
        is None
    )
    assert (
        build_trusted_presenter_runtime(
            "train_model_with_evidence_v2",
            workspace=tmp_path,
            task_id=None,
        )
        is None
    )


def test_runtime_presenter_dispatch_forwards_only_trusted_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    sentinel_runtime = object()

    def presenter(output, **kwargs):
        calls.append({"output": output, **kwargs})
        return "authenticated", [{"title": "evidence", "columns": [], "rows": []}]

    monkeypatch.setitem(
        renderers._TRUSTED_RUNTIME_PRESENTERS,
        "test_governed_tool",
        presenter,
    )
    monkeypatch.setattr(
        renderers,
        "build_trusted_presenter_runtime",
        lambda tool, *, workspace, task_id: (
            sentinel_runtime
            if (tool, Path(workspace), task_id)
            == ("test_governed_tool", tmp_path, "task-1")
            else None
        ),
    )

    rendered = renderers.render_tool_output(
        "test_governed_tool",
        {
            "result": "stored",
            "artifact_id": "artifact-1",
            "content_hash": "d" * 64,
        },
        trusted_task_id="task-1",
        trusted_workspace=tmp_path,
        trusted_output_ref="metrics:step-1:v1",
        trusted_artifacts={"evidence": {"id": "artifact-1"}},
        trusted_inputs={"source": "step-1"},
        trusted_step_evidence={
            "output_ref": "metrics:step-1:v1",
            "step_run_id": "run-1",
            "renderer_hint": "test_governed_tool",
            "input_hash": "sha256:" + "a" * 64,
            "artifact_refs": ["artifact:artifact-1"],
            "artifact_bindings": [
                {
                    "artifact_id": "artifact-1",
                    "content_hash": "d" * 64,
                }
            ],
        },
    )

    assert rendered[0] == "authenticated"
    assert calls == [
        {
            "output": {
                "result": "stored",
                "artifact_id": "artifact-1",
                "content_hash": "d" * 64,
            },
            "runtime": sentinel_runtime,
            "task_id": "task-1",
            "trusted_inputs": {"source": "step-1"},
            "trusted_artifacts": {"evidence": {"id": "artifact-1"}},
        }
    ]


def test_runtime_presenter_missing_context_or_failure_never_falls_back_to_generic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args, **_kwargs):
        raise RuntimeError("must not leak")

    monkeypatch.setitem(
        renderers._TRUSTED_RUNTIME_PRESENTERS,
        "test_governed_tool",
        fail,
    )

    missing_text, missing_tables = renderers.render_tool_output(
        "test_governed_tool",
        {"status": "looks-valid"},
    )
    failed_text, failed_tables = renderers.render_tool_output(
        "test_governed_tool",
        {"status": "looks-valid"},
        trusted_task_id="task-1",
        trusted_workspace=tmp_path,
    )

    assert "完整性校验失败" in missing_text
    assert "完整性校验失败" in failed_text
    assert missing_tables == failed_tables == []
    assert "looks-valid" not in missing_text + failed_text
    assert "must not leak" not in missing_text + failed_text


def test_runtime_presenter_rejects_complete_same_task_artifact_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def presenter(_output, **_kwargs):
        nonlocal called
        called = True
        return "must not render", []

    monkeypatch.setitem(
        renderers._TRUSTED_RUNTIME_PRESENTERS,
        "test_governed_tool",
        presenter,
    )
    monkeypatch.setattr(
        renderers,
        "build_trusted_presenter_runtime",
        lambda *_args, **_kwargs: object(),
    )

    text, tables = renderers.render_tool_output(
        "test_governed_tool",
        {
            "artifacts": [
                {
                    "artifact_id": "artifact-from-another-valid-run",
                    "content_hash": "e" * 64,
                }
            ]
        },
        trusted_task_id="task-1",
        trusted_workspace=tmp_path,
        trusted_output_ref="metrics:step-1:v1",
        trusted_step_evidence={
            "output_ref": "metrics:step-1:v1",
            "step_run_id": "run-1",
            "renderer_hint": "test_governed_tool",
            "input_hash": "sha256:" + "a" * 64,
            "artifact_refs": ["artifact:artifact-from-this-step"],
            "artifact_bindings": [
                {
                    "artifact_id": "artifact-from-this-step",
                    "content_hash": "f" * 64,
                }
            ],
        },
    )

    assert "完整性校验失败" in text
    assert tables == []
    assert called is False


def test_gate_dependency_renderer_accepts_trusted_step_callback() -> None:
    dependency = SimpleNamespace(
        id="step-1",
        title="Authenticated dependency",
        output_ref="output-1",
        tool_ref=SimpleNamespace(tool="test_governed_tool"),
    )
    gate = SimpleNamespace(
        id="step-2",
        depends_on=[dependency.id],
        tool_ref=SimpleNamespace(tool="confirm_test"),
    )
    plan = SimpleNamespace(steps=[dependency, gate])
    calls: list[tuple[object, object, str | None]] = []

    rendered = render_gate_dependencies(
        plan,
        gate,
        lambda _step_id: {"result": "stored"},
        render_output=lambda step, output, presentation_state: (
            calls.append((step, output, presentation_state))
            or ("authenticated dependency", [])
        ),
    )

    assert rendered.parts == ["authenticated dependency"]
    assert calls == [(dependency, {"result": "stored"}, None)]


def test_message_composer_forwards_workspace_and_impact_cube_registry_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = {
        "id": "impact-cube-artifact-1",
        "task_id": "task-1",
        "kind": "strategy_impact_cube",
        "content_hash": "a" * 64,
    }
    captured: dict = {}
    composer = PlanMessageComposer(
        load_output=lambda _step_id, *, version=None: None,
        load_step_evidence=lambda _step_id, *, version=None: {
            "output_ref": f"metrics:step-1:v{version}",
            "step_run_id": "run-1",
            "renderer_hint": "measure_strategy_impact_cube",
            "input_hash": "sha256:" + "b" * 64,
            "artifact_refs": ["artifact:impact-cube-artifact-1"],
        },
        load_task_artifact=lambda task_id, artifact_id: (
            record
            if (task_id, artifact_id) == ("task-1", "impact-cube-artifact-1")
            else None
        ),
        tasks_root=tmp_path / "tasks",
        db_path=tmp_path / "marvis.db",
    )
    plan = SimpleNamespace(task_id="task-1")
    step = SimpleNamespace(
        id="step-1",
        output_ref="metrics:step-1:v3",
        inputs={},
        tool_ref=SimpleNamespace(tool="measure_strategy_impact_cube"),
    )
    output = {"artifact": {"artifact_id": "impact-cube-artifact-1"}}

    def capture(tool, value, **kwargs):
        captured.update({"tool": tool, "output": value, **kwargs})
        return "ok", []

    monkeypatch.setattr(
        "marvis.agent.plan_message_composer.render_tool_output", capture
    )

    assert composer._render_step_output(
        plan,
        step,
        output,
        presentation_state="completed",
    ) == ("ok", [])
    assert captured["trusted_task_id"] == "task-1"
    assert captured["trusted_workspace"] == tmp_path.resolve()
    assert captured["trusted_step_evidence"]["output_ref"] == (
        "metrics:step-1:v3"
    )
    assert captured["trusted_artifacts"] == {"impact_cube": {"record": record}}


def test_result_dataset_control_requires_exact_step_binding_and_live_bytes(
    tmp_path: Path,
) -> None:
    output = {"result_dataset_id": "dataset-1"}
    evidence = {
        "output_ref": "metrics:label:v2",
        "step_run_id": "run-2",
        "renderer_hint": "define_label",
        "input_hash": "sha256:" + "c" * 64,
        "artifact_refs": [],
        "result_dataset_bindings": [
            {"dataset_id": "dataset-1", "content_hash": "d" * 64}
        ],
    }
    dataset = SimpleNamespace(
        id="dataset-1",
        task_id="task-1",
        content_hash="d" * 64,
    )
    binding = {
        "plan_id": "plan-1",
        "task_id": "task-1",
        "step_id": "label",
        "output_ref": "metrics:label:v2",
        "output": output,
        "evidence": evidence,
        "inputs": {"source_dataset_id": "dataset-source"},
    }
    verified: list[str] = []
    composer = PlanMessageComposer(
        load_output=lambda _step_id, *, version=None: output,
        load_step_presentation_binding=lambda _step_id, _output_ref: binding,
        load_dataset=lambda dataset_id: (
            dataset if dataset_id == "dataset-1" else None
        ),
        resolve_verified_dataset_path=lambda dataset_id: (
            verified.append(dataset_id) or tmp_path / "dataset.parquet"
        ),
    )
    step = SimpleNamespace(
        id="label",
        index=0,
        status=StepStatus.DONE,
        output_ref="metrics:label:v2",
        tool_ref=SimpleNamespace(tool="define_label"),
    )
    plan = SimpleNamespace(id="plan-1", task_id="task-1", steps=[step])

    assert composer.latest_result_dataset_metadata(plan) == {
        "dataset_id": "dataset-1",
        "download_url": (
            "/api/tasks/task-1/datasets/dataset-1/download"
            "?plan_id=plan-1&step_id=label"
            "&output_ref=metrics%3Alabel%3Av2"
            f"&expected_content_hash={'d' * 64}"
        ),
        "plan_id": "plan-1",
        "step_id": "label",
        "output_ref": "metrics:label:v2",
        "content_hash": "d" * 64,
        "title": "标签数据集已生成",
        "download_label": "下载标签结果",
    }
    assert verified == ["dataset-1"]

    dataset.content_hash = "e" * 64
    drifted = PlanMessageComposer(
        load_output=lambda _step_id, *, version=None: output,
        load_step_presentation_binding=lambda _step_id, _output_ref: binding,
        load_dataset=lambda dataset_id: (
            dataset if dataset_id == "dataset-1" else None
        ),
        resolve_verified_dataset_path=lambda _dataset_id: (
            tmp_path / "dataset.parquet"
        ),
    )
    assert drifted.latest_result_dataset_metadata(plan) is None


def test_historical_download_uses_message_plan_and_overwrites_cached_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_plan = SimpleNamespace(id="plan-old")
    new_plan = SimpleNamespace(id="plan-new")
    plan_repo = SimpleNamespace(
        list_plans_for_task=lambda _task_id: [old_plan, new_plan]
    )
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(plan_repo=plan_repo))
    )
    expected = {
        "plan-old": {
            "dataset_id": "dataset-old",
            "download_url": "/api/tasks/task-1/datasets/dataset-old/download",
            "title": "结果数据集已生成",
            "download_label": "下载结果数据集",
        },
        "plan-new": {
            "dataset_id": "dataset-new",
            "download_url": "/api/tasks/task-1/datasets/dataset-new/download",
            "title": "结果数据集已生成",
            "download_label": "下载结果数据集",
            "recovered_from_plan": True,
        },
    }

    class Composer:
        def latest_result_dataset_metadata(self, plan):
            if plan is old_plan:
                return {key: value for key, value in expected["plan-old"].items()}
            assert plan is new_plan
            return {
                key: value
                for key, value in expected["plan-new"].items()
                if key != "recovered_from_plan"
            }

    monkeypatch.setattr(
        validation_agent,
        "_plan_message_composer",
        lambda _request: Composer(),
    )
    messages = [
        {
            "role": "assistant",
            "content": "✅ 计划已全部完成。",
            "metadata": {
                "plan_id": "plan-old",
                "result_dataset": {
                    "dataset_id": "forged",
                    "download_url": (
                        "/api/tasks/task-1/datasets/forged/download"
                    ),
                    "title": "forged",
                    "download_label": "forged",
                },
            },
        },
        {
            "role": "assistant",
            "content": "✅ 计划已全部完成。",
            "metadata": {"plan_id": "plan-new"},
        },
    ]

    validation_agent._enrich_historical_result_download(
        request,
        "task-1",
        messages,
    )

    assert messages[0]["metadata"]["result_dataset"] == expected["plan-old"]
    assert messages[1]["metadata"]["result_dataset"] == expected["plan-new"]


def test_historical_download_strips_unbound_cached_action() -> None:
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                plan_repo=SimpleNamespace(
                    list_plans_for_task=lambda _task_id: []
                )
            )
        )
    )
    messages = [
        {
            "role": "assistant",
            "content": "✅ 计划已全部完成。",
            "metadata": {
                "plan_id": "missing-plan",
                "result_dataset": {"dataset_id": "forged"},
            },
        }
    ]

    validation_agent._enrich_historical_result_download(
        request,
        "task-1",
        messages,
    )

    assert "result_dataset" not in messages[0]["metadata"]
