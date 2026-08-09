from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.data.workspace import DataWorkspaceDraft
from marvis.db import DatasetRepository, PluginRepository, TaskRepository, init_db
from marvis.db_schema import connect
from marvis.domain import TASK_TYPE_DATA_JOIN, TASK_TYPE_MODELING, TaskCreate
from marvis.files import sha256_file
from marvis.packs.labeling.contracts import (
    LabelingContractError,
    LabelingRequest,
    build_labeling_proposal,
)
from marvis.packs.labeling.tools import tool_define_label
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.settings import build_settings


def _planner(tmp_path):
    from marvis.db import PluginRepository, init_db
    from marvis.orchestrator.planner import Planner
    from marvis.orchestrator.validator import PlanValidator
    from marvis.plugins.loader import load_builtin_packs
    from marvis.plugins.registry import PluginRegistry, ToolRegistry

    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    plugin_repo = PluginRepository(db_path)
    plugin_registry = PluginRegistry(plugin_repo)
    load_builtin_packs(
        plugin_registry,
        Path(__file__).parents[1] / "marvis" / "packs",
    )
    tool_registry = ToolRegistry(plugin_registry)
    return Planner(tool_registry, lambda: None, PlanValidator(tool_registry))


@dataclass(frozen=True)
class _Column:
    name: str


@dataclass(frozen=True)
class _Dataset:
    id: str = "source-1"
    task_id: str = "task-1"
    source_path: str = "task-1/source.parquet"
    content_hash: str = "a" * 64
    row_count: int = 8
    columns: tuple[_Column, ...] = tuple(
        _Column(name)
        for name in ("loan_id", "mob", "cohort", "snapshot_date", "dpd")
    )


@dataclass(frozen=True)
class _Workspace:
    active_dataset_id: str = "source-1"
    active_dataset_content_hash: str = "a" * 64
    revision: int = 3
    analysis_generation: int = 2


class _Registry:
    def get(self, dataset_id):
        if dataset_id != "source-1":
            raise KeyError(dataset_id)
        return _Dataset()

    def resolve_verified_path(self, dataset_id):
        assert dataset_id == "source-1"
        return "/tmp/source.parquet"

    def read_authenticated_parquet_snapshot(self, dataset_id, *, columns=None):
        assert dataset_id == "source-1"
        return _Backend().read_frame("/tmp/source.parquet", columns=columns)


class _Backend:
    _frame = pd.DataFrame(
        {
            "loan_id": ["A"] * 4 + ["B"] * 4,
            "mob": [0, 2, 4, 6] * 2,
            "cohort": ["2026-01"] * 8,
            "snapshot_date": pd.to_datetime(
                [
                    "2026-01-31",
                    "2026-03-31",
                    "2026-05-31",
                    "2026-07-31",
                ]
                * 2
            ),
            "dpd": [0, 0, 95, 120, 0, 0, 0, 0],
        }
    )

    def column_names(self, path):
        return list(self._frame.columns)

    def read_frame(self, path, *, columns=None):
        frame = self._frame.copy()
        return frame if columns is None else frame[list(columns)].copy()


def _request(**overrides) -> LabelingRequest:
    payload = {
        "dataset_id": "source-1",
        "expected_content_hash": "a" * 64,
        "workspace_revision": 3,
        "analysis_generation": 2,
        "id_col": "loan_id",
        "mob_col": "mob",
        "cohort_col": "cohort",
        "date_col": "snapshot_date",
        "as_of_date": "2026-07-31",
        "target_col": "bad_90d_6m",
        "observation_window": 0,
        "performance_window": 6,
        "at_mob": 6,
        "rule_kind": "dpd",
        "dpd_col": "dpd",
        "threshold_dpd": 90,
    }
    payload.update(overrides)
    return LabelingRequest(**payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_col", ""),
        ("date_col", ""),
        ("as_of_date", ""),
        ("rule_kind", ""),
        ("observation_window", None),
        ("performance_window", None),
        ("at_mob", None),
        ("threshold_dpd", None),
    ],
)
def test_labeling_request_fails_closed_when_business_contract_is_missing(
    field,
    value,
):
    with pytest.raises(LabelingContractError, match=field):
        _request(**{field: value})


def test_status_rule_requires_explicit_order_and_threshold_mapping():
    with pytest.raises(LabelingContractError, match="states"):
        _request(
            rule_kind="status",
            dpd_col=None,
            threshold_dpd=None,
            status_col="bucket",
            threshold_status="M2",
            states=None,
        )

    request = _request(
        rule_kind="status",
        dpd_col=None,
        threshold_dpd=None,
        status_col="bucket",
        threshold_status="M2",
        states=["C", "M1", "M2", "M3+"],
    )
    assert request.states == ("C", "M1", "M2", "M3+")
    assert request.rule_summary == "bucket in [M2, M3+]"


def test_labeling_proposal_binds_source_cutoff_rule_and_maturity_evidence():
    request = _request()
    proposal = build_labeling_proposal(
        _Registry(),
        _Backend(),
        _Workspace(),
        task_id="task-1",
        request=request,
    )

    assert proposal.requires_human_confirmation is True
    assert proposal.contract_hash == request.contract_hash
    assert proposal.source_dataset_id == "source-1"
    assert proposal.source_content_hash == "a" * 64
    assert proposal.as_of_date == "2026-07-31"
    assert proposal.rule_summary == "dpd >= 90"
    assert proposal.maturity["required_mob"] == 6
    assert proposal.maturity["all_matured"] is True
    assert proposal.to_template_slots()["proposal_hash"] == request.contract_hash
    assert proposal.to_template_slots()["target_col"] == "bad_90d_6m"


def test_labeling_template_requires_confirmed_explicit_contract_and_human_gate(
    tmp_path,
):
    from marvis.orchestrator.planner import PlanningError
    from marvis.orchestrator.templates import get_template, load_builtin_templates

    load_builtin_templates()
    planner = _planner(tmp_path)
    slots = {
        **_request().to_dict(),
        "proposal_hash": _request().contract_hash,
        "confirm_immature_cohorts": False,
    }
    template = get_template("label_construction")

    for field in (
        "target_col",
        "date_col",
        "as_of_date",
        "rule_kind",
        "proposal_hash",
    ):
        missing = dict(slots)
        missing.pop(field)
        with pytest.raises(PlanningError, match=field):
            planner.from_template(template, missing, "task-1")

    plan = planner.from_template(template, slots, "task-1")
    define = next(step for step in plan.steps if step.title == "构造标签")
    assert define.inputs["target_col"] == "bad_90d_6m"
    assert define.inputs["date_col"] == "snapshot_date"
    assert define.inputs["as_of_date"] == "2026-07-31"
    assert define.inputs["rule_kind"] == "dpd"
    assert define.inputs["proposal_hash"] == _request().contract_hash
    assert define.needs_confirmation is True
    assert define.policy.human_decision_gate == "required"


def _labeling_runtime(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="数据处理内标签构造",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "source"),
            algorithm="lr",
            run_mode="agent",
            task_type=TASK_TYPE_DATA_JOIN,
        )
    )
    backend = DataBackend(settings.datasets_dir)
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        backend,
        settings.datasets_dir,
    )
    source_path = tmp_path / "label-source.parquet"
    _Backend._frame.to_parquet(source_path, index=False)
    dataset = registry.register_existing(
        source_path,
        task_id=task.id,
        role="sample",
    )
    workspace_repo = DataWorkspaceRepository(settings.db_path)
    workspace = workspace_repo.save_initial_binding(
        task.id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
        ),
        expected_revision=0,
        audit={"actor": "test:labeling"},
    )
    ctx = SimpleNamespace(
        workspace=settings.workspace,
        datasets_root=settings.datasets_dir,
        task_id=task.id,
        seed=0,
    )
    request = LabelingRequest(
        dataset_id=dataset.id,
        expected_content_hash=dataset.content_hash,
        workspace_revision=workspace.revision,
        analysis_generation=workspace.analysis_generation,
        id_col="loan_id",
        mob_col="mob",
        cohort_col="cohort",
        date_col="snapshot_date",
        as_of_date="2026-07-31",
        target_col="bad_90d_6m",
        observation_window=0,
        performance_window=6,
        at_mob=6,
        rule_kind="dpd",
        dpd_col="dpd",
        threshold_dpd=90,
    )
    return settings, registry, backend, workspace_repo, task, ctx, request


def test_define_label_publishes_non_active_dataset_quality_and_downloads(tmp_path):
    (
        settings,
        registry,
        backend,
        workspace_repo,
        task,
        ctx,
        request,
    ) = _labeling_runtime(tmp_path)
    before = workspace_repo.get_or_default(task.id)

    output = tool_define_label(
        {
            **request.to_dict(),
            "proposal_hash": request.contract_hash,
            "confirm_immature_cohorts": False,
        },
        ctx,
    )

    after = workspace_repo.get_or_default(task.id)
    assert after.active_dataset_id == before.active_dataset_id == request.dataset_id
    assert after.active_dataset_content_hash == before.active_dataset_content_hash
    assert after.revision == before.revision
    assert output["schema_version"] == "labeling-tool-result.v1"
    assert output["source_dataset_id"] == request.dataset_id
    assert output["result_dataset_id"] != request.dataset_id
    assert output["workspace"]["active_dataset_changed"] is False
    assert output["quality"] == {
        "n_loans": 2,
        "n_bad": 1,
        "n_good": 1,
        "n_unmatured": 0,
        "label_coverage": 1.0,
        "bad_rate": 0.5,
    }
    result = registry.get(output["result_dataset_id"])
    assert result.role == "derived"
    assert result.target_col == "bad_90d_6m"
    result_frame = backend.read_frame(registry.resolve_verified_path(result.id))
    assert list(result_frame.columns) == ["loan_id", "cohort", "bad_90d_6m"]

    artifacts = TaskArtifactRepository(settings.db_path).list_for_task(task.id)
    by_id = {record["id"]: record for record in artifacts}
    dataset_artifact = by_id[output["dataset_artifact_id"]]
    evidence_artifact = by_id[output["evidence_artifact_id"]]
    assert dataset_artifact["kind"] == "labeling_dataset_csv"
    assert evidence_artifact["kind"] == "labeling_quality_evidence_json"
    assert dataset_artifact["content_hash"] == output["dataset_content_hash"]
    assert evidence_artifact["content_hash"] == output["evidence_content_hash"]
    evidence = json.loads(Path(evidence_artifact["path"]).read_text(encoding="utf-8"))
    assert evidence["proposal_hash"] == request.contract_hash
    assert evidence["quality"] == output["quality"]
    assert evidence["maturity"]["all_matured"] is True
    assert evidence["workspace"]["active_dataset_changed"] is False

    with TestClient(create_app(settings)) as client:
        dataset_download = client.get(output["dataset_download_url"])
        evidence_download = client.get(output["evidence_download_url"])
        assert dataset_download.status_code == 200
        assert evidence_download.status_code == 200
        assert sha256_file(Path(dataset_artifact["path"])) == output["dataset_content_hash"]
        assert json.loads(evidence_download.content)["quality"] == output["quality"]

    audits = PluginRepository(settings.db_path).list_audit()
    assert any(
        row["kind"] == "labeling.dataset.created"
        and row["target_ref"] == output["result_dataset_id"]
        for row in audits
    )


def test_define_label_rolls_back_dataset_and_files_when_evidence_registration_fails(
    tmp_path,
    monkeypatch,
):
    (
        settings,
        registry,
        _backend,
        workspace_repo,
        task,
        ctx,
        request,
    ) = _labeling_runtime(tmp_path)
    original = TaskArtifactRepository.register_on_connection

    def fail_evidence(self, conn, **kwargs):
        if kwargs.get("kind") == "labeling_quality_evidence_json":
            raise RuntimeError("injected evidence registry failure")
        return original(self, conn, **kwargs)

    monkeypatch.setattr(
        TaskArtifactRepository,
        "register_on_connection",
        fail_evidence,
    )

    with pytest.raises(RuntimeError, match="injected evidence"):
        tool_define_label(
            {
                **request.to_dict(),
                "proposal_hash": request.contract_hash,
                "confirm_immature_cohorts": False,
            },
            ctx,
        )

    assert [item.id for item in registry.list_for_task(task.id)] == [request.dataset_id]
    assert TaskArtifactRepository(settings.db_path).list_for_task(task.id) == []
    workspace = workspace_repo.get_or_default(task.id)
    assert workspace.active_dataset_id == request.dataset_id
    task_artifact_dir = settings.tasks_dir / task.id / "labeling"
    assert not list(task_artifact_dir.glob("*.csv"))
    assert not list(task_artifact_dir.glob("*.json"))


def test_define_label_rejects_source_swap_between_verification_and_read(
    tmp_path,
    monkeypatch,
):
    (
        settings,
        registry,
        _backend,
        _workspace_repo,
        task,
        ctx,
        request,
    ) = _labeling_runtime(tmp_path)
    source_path = registry.resolve_verified_path(request.dataset_id)
    original_bytes = source_path.read_bytes()
    substituted_path = tmp_path / "substituted.parquet"
    substituted = _Backend._frame.copy()
    substituted["dpd"] = [0] * len(substituted)
    substituted.to_parquet(substituted_path, index=False)
    substituted_bytes = substituted_path.read_bytes()

    original_resolve = DatasetRegistry.resolve_verified_path
    original_read = DataBackend.read_frame

    def resolve_then_swap(self, dataset_id):
        path = original_resolve(self, dataset_id)
        path.write_bytes(substituted_bytes)
        return path

    def read_then_restore(self, path, *, columns=None):
        frame = original_read(self, path, columns=columns)
        if Path(path) == source_path:
            Path(path).write_bytes(original_bytes)
        return frame

    monkeypatch.setattr(
        DatasetRegistry,
        "resolve_verified_path",
        resolve_then_swap,
    )
    monkeypatch.setattr(DataBackend, "read_frame", read_then_restore)

    with pytest.raises(LabelingContractError, match="authenticated|snapshot|bytes"):
        tool_define_label(
            {
                **request.to_dict(),
                "proposal_hash": request.contract_hash,
                "confirm_immature_cohorts": False,
            },
            ctx,
        )

    source_path.write_bytes(original_bytes)
    assert [item.id for item in registry.list_for_task(task.id)] == [request.dataset_id]
    assert TaskArtifactRepository(settings.db_path).list_for_task(task.id) == []
    assert not any(
        row["kind"] == "labeling.dataset.created"
        for row in PluginRepository(settings.db_path).list_audit()
    )


@pytest.mark.parametrize("drift_kind", ["workspace", "source"])
def test_define_label_fails_closed_if_confirmed_binding_changes_before_execution(
    tmp_path,
    drift_kind,
):
    (
        settings,
        registry,
        _backend,
        workspace_repo,
        task,
        ctx,
        request,
    ) = _labeling_runtime(tmp_path)
    if drift_kind == "workspace":
        before = workspace_repo.get_or_default(task.id)
        workspace_repo.save(
            task.id,
            DataWorkspaceDraft(
                active_dataset_id=before.active_dataset_id,
                active_dataset_content_hash=before.active_dataset_content_hash,
                page="history",
                semantic_mapping=before.semantic_mapping,
            ),
            expected_revision=before.revision,
            audit={"actor": "test:post-confirm-drift"},
        )
    else:
        source_path = registry.resolve_verified_path(request.dataset_id)
        changed = _Backend._frame.copy()
        changed.loc[0, "dpd"] = 1
        changed.to_parquet(source_path, index=False)

    with pytest.raises(LabelingContractError, match="workspace|authenticated"):
        tool_define_label(
            {
                **request.to_dict(),
                "proposal_hash": request.contract_hash,
                "confirm_immature_cohorts": False,
            },
            ctx,
        )

    assert [item.id for item in registry.list_for_task(task.id)] == [request.dataset_id]
    assert TaskArtifactRepository(settings.db_path).list_for_task(task.id) == []
    assert not any(
        row["kind"] == "labeling.dataset.created"
        for row in PluginRepository(settings.db_path).list_audit()
    )


def test_define_label_rechecks_registered_source_identity_before_publication(
    tmp_path,
    monkeypatch,
):
    (
        settings,
        registry,
        _backend,
        _workspace_repo,
        task,
        ctx,
        request,
    ) = _labeling_runtime(tmp_path)
    from marvis.packs.labeling import tools as labeling_tools

    original_construct = labeling_tools.construct_label

    def construct_then_repoint_source(*args, **kwargs):
        result = original_construct(*args, **kwargs)
        with connect(settings.db_path) as conn:
            conn.execute(
                "UPDATE datasets SET source_path = ? WHERE id = ?",
                ("repointed/source.parquet", request.dataset_id),
            )
        return result

    monkeypatch.setattr(
        labeling_tools,
        "construct_label",
        construct_then_repoint_source,
    )

    with pytest.raises(LabelingContractError, match="path changed"):
        tool_define_label(
            {
                **request.to_dict(),
                "proposal_hash": request.contract_hash,
                "confirm_immature_cohorts": False,
            },
            ctx,
        )

    assert [item.id for item in registry.list_for_task(task.id)] == [request.dataset_id]
    assert TaskArtifactRepository(settings.db_path).list_for_task(task.id) == []
    assert not any(
        row["kind"] == "labeling.dataset.created"
        for row in PluginRepository(settings.db_path).list_audit()
    )


def _post_labeling_proposal(client, task_id, request, **extra):
    payload = {
        "content": "提交标签构造口径",
        "labeling_request": request.to_dict(),
    }
    payload.update(extra)
    return client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json=payload,
    )


def _last_assistant(response):
    return [
        message
        for message in response.json()["messages"]
        if message["role"] == "assistant"
    ][-1]


def test_labeling_api_requires_complete_unmixed_typed_contract(tmp_path):
    settings, _registry, _backend, _workspace_repo, task, _ctx, request = (
        _labeling_runtime(tmp_path)
    )
    with TestClient(create_app(settings)) as client:
        missing = request.to_dict()
        missing.pop("target_col")
        response = client.post(
            f"/api/tasks/{task.id}/agent/messages",
            json={"content": "提交标签构造口径", "labeling_request": missing},
        )
        assert response.status_code == 422

        mixed_rule = request.to_dict()
        mixed_rule.update(
            {
                "status_col": "bucket",
                "threshold_status": "M2",
                "states": ["C", "M1", "M2", "M3+"],
            }
        )
        response = client.post(
            f"/api/tasks/{task.id}/agent/messages",
            json={"content": "提交标签构造口径", "labeling_request": mixed_rule},
        )
        assert response.status_code == 422

        response = _post_labeling_proposal(
            client,
            task.id,
            request,
            adjust_params={"threshold_dpd": 60},
        )
        assert response.status_code == 422
        assert "labeling_request" in response.text


def test_labeling_api_rejects_non_data_join_task(tmp_path):
    settings, _registry, _backend, _workspace_repo, _task, _ctx, request = (
        _labeling_runtime(tmp_path)
    )
    task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="非数据处理任务",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "modeling-source"),
            algorithm="lr",
            run_mode="agent",
            task_type=TASK_TYPE_MODELING,
        )
    )
    with TestClient(create_app(settings)) as client:
        response = _post_labeling_proposal(
            client,
            task.id,
            request,
            acceptance_mode="auto_accept",
        )
    assert response.status_code == 422
    assert "data_join" in response.text


def test_labeling_typed_request_builds_proposal_without_llm_or_plan(
    tmp_path,
    monkeypatch,
):
    settings, _registry, _backend, _workspace_repo, task, _ctx, request = (
        _labeling_runtime(tmp_path)
    )

    def fail_if_llm_is_resolved(*_args, **_kwargs):
        raise AssertionError("typed labeling request must not resolve an LLM")

    monkeypatch.setattr(
        "marvis.routers.validation_agent.resolve_driver_agent_client",
        fail_if_llm_is_resolved,
    )
    with TestClient(create_app(settings)) as client:
        response = _post_labeling_proposal(client, task.id, request)
        assert response.status_code == 202, response.text
        assert client.app.state.plan_repo.list_plans_for_task(task.id) == []
        proposal = _last_assistant(response)

    metadata = proposal["metadata"]
    assert metadata["kind"] == "labeling_preplan_confirmation"
    assert metadata["labeling_proposal"]["proposal_hash"] == request.contract_hash
    assert metadata["labeling_proposal"]["requires_human_confirmation"] is True
    assert metadata["labeling_proposal"]["rows_at_as_of"] == 8
    assert metadata["labeling_proposal"]["maturity"]["all_matured"] is True


@pytest.mark.parametrize(
    ("instruction", "review_overrides", "should_create"),
    [
        ("标签口径无误，请按当前提案创建计划。", {}, True),
        ("这个标签口径可以继续吗？", {"is_question": True}, False),
        ("如果成熟度没问题再继续。", {"is_conditional": True}, False),
    ],
)
def test_labeling_agent_uses_two_pass_semantic_preplan_authorization(
    tmp_path,
    monkeypatch,
    instruction,
    review_overrides,
    should_create,
):
    settings, _registry, _backend, _workspace_repo, task, _ctx, request = (
        _labeling_runtime(tmp_path)
    )

    class _SemanticClient:
        route_calls = 0
        review_calls = 0

        def complete(self, *_args, **kwargs):
            if kwargs.get("prompt_name") == "GATE_SEMANTIC_AUTHORIZATION_REVIEW_SYS":
                self.review_calls += 1
                reviewed_instruction = json.loads(kwargs["user_prompt"])["instruction"]
                payload = {
                    "verdict": "authorize",
                    "evidence_quote": reviewed_instruction,
                    "reason": "用户明确授权当前完整标签提案。",
                    "confidence": "high",
                    "is_question": False,
                    "is_conditional": False,
                    "requests_change": False,
                    "withholds_authorization": False,
                }
                payload.update(review_overrides)
                return json.dumps(payload, ensure_ascii=False)
            self.route_calls += 1
            return json.dumps(
                {
                    "action": "confirm",
                    "params": {},
                    "constraint": "",
                    "reason": "用户明确授权当前完整标签提案。",
                    "confidence": "high",
                    "explicit_authorization": True,
                },
                ensure_ascii=False,
            )

    semantic_client = _SemanticClient()
    monkeypatch.setattr(
        "marvis.routers.validation_agent.resolve_driver_agent_client",
        lambda _request, _task, _payload: semantic_client,
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda _request, _task: semantic_client,
    )
    with TestClient(create_app(settings)) as client:
        proposal = _post_labeling_proposal(client, task.id, request)
        assert proposal.status_code == 202, proposal.text

        response = client.post(
            f"/api/tasks/{task.id}/agent/messages",
            json={"content": instruction},
        )

        assert response.status_code == 202, response.text
        plans = client.app.state.plan_repo.list_plans_for_task(task.id)
        assert bool(plans) is should_create
        messages = response.json()["messages"]
        receipts = [
            message
            for message in messages
            if message.get("metadata", {}).get("intent")
            == "labeling_semantic_authorization"
        ]
        assert bool(receipts) is should_create
        if should_create:
            assert plans[-1].status.value == "validated"
            assert receipts[-1]["metadata"]["proposal_hash"] == request.contract_hash
        else:
            assert _last_assistant(response)["metadata"]["code"] == (
                "labeling_human_confirmation_required"
            )
    assert semantic_client.route_calls == 1
    assert semantic_client.review_calls == 1


@pytest.mark.parametrize("drift_kind", ["workspace", "source"])
def test_labeling_preplan_confirmation_rejects_workspace_or_source_drift(
    tmp_path,
    drift_kind,
):
    settings, registry, _backend, workspace_repo, task, _ctx, request = (
        _labeling_runtime(tmp_path)
    )
    with TestClient(create_app(settings)) as client:
        proposal_response = _post_labeling_proposal(client, task.id, request)
        assert proposal_response.status_code == 202, proposal_response.text

        if drift_kind == "workspace":
            before = workspace_repo.get_or_default(task.id)
            workspace_repo.save(
                task.id,
                DataWorkspaceDraft(
                    active_dataset_id=before.active_dataset_id,
                    active_dataset_content_hash=before.active_dataset_content_hash,
                    page="history",
                    semantic_mapping=before.semantic_mapping,
                ),
                expected_revision=before.revision,
                audit={"actor": "test:drift"},
            )
        else:
            source_path = registry.resolve_verified_path(request.dataset_id)
            changed = _Backend._frame.copy()
            changed.loc[0, "dpd"] = 1
            changed.to_parquet(source_path, index=False)

        response = client.post(
            f"/api/tasks/{task.id}/agent/messages",
            json={"content": "确认"},
        )
        assert response.status_code == 202, response.text
        assert response.json()["status"] == "clarification_required"
        rejection = _last_assistant(response)
        assert rejection["metadata"]["code"] == "labeling_proposal_stale"
        assert client.app.state.plan_repo.list_plans_for_task(task.id) == []
        assert TaskArtifactRepository(settings.db_path).list_for_task(task.id) == []


def test_labeling_api_e2e_requires_preplan_confirmation_and_publishes_downloads(
    tmp_path,
):
    settings, registry, _backend, workspace_repo, task, _ctx, request = (
        _labeling_runtime(tmp_path)
    )
    before = workspace_repo.get_or_default(task.id)

    with TestClient(create_app(settings)) as client:
        proposal_response = _post_labeling_proposal(client, task.id, request)
        assert proposal_response.status_code == 202, proposal_response.text
        assert client.app.state.plan_repo.list_plans_for_task(task.id) == []

        confirmed = client.post(
            f"/api/tasks/{task.id}/agent/messages",
            json={"content": "确认"},
        )
        assert confirmed.status_code == 202, confirmed.text
        plans = client.app.state.plan_repo.list_plans_for_task(task.id)
        assert len(plans) == 1
        plan = plans[0]
        assert plan.status.value == "validated"
        overview = _last_assistant(confirmed)
        assert overview["metadata"]["kind"] == "plan_overview"

        started = client.post(
            f"/api/tasks/{task.id}/agent/messages",
            json={
                "content": "开始",
                "ui_action": "start_plan",
                "expected_plan_id": plan.id,
                **overview["metadata"]["confirmation_snapshot"],
            },
        )
        assert started.status_code == 202, started.text
        plan = client.app.state.plan_repo.load_plan(plan.id)
        assert plan.status.value == "awaiting_confirm"
        define_step = next(step for step in plan.steps if step.title == "构造标签")
        define_gate = _last_assistant(started)
        assert define_gate["metadata"]["kind"] == "gate"
        assert define_gate["metadata"]["step_id"] == define_step.id

        completed = client.post(
            f"/api/tasks/{task.id}/agent/messages",
            json={
                "content": "确认",
                "ui_action": "confirm_gate",
                "expected_plan_id": define_gate["metadata"]["plan_id"],
                "expected_step_id": define_gate["metadata"]["step_id"],
                **define_gate["metadata"]["confirmation_snapshot"],
            },
        )
        assert completed.status_code == 202, completed.text
        plan = client.app.state.plan_repo.load_plan(plan.id)
        assert plan.status.value == "done"

        artifact_response = client.get(f"/api/tasks/{task.id}/task-artifacts")
        assert artifact_response.status_code == 200, artifact_response.text
        artifacts = artifact_response.json()["artifacts"]
        assert {artifact["kind"] for artifact in artifacts} == {
            "labeling_dataset_csv",
            "labeling_quality_evidence_json",
        }
        for artifact in artifacts:
            assert artifact["available"] is True
            download = client.get(artifact["download_url"])
            assert download.status_code == 200
            assert sha256_file(Path(TaskArtifactRepository(settings.db_path).get_for_task(
                task.id,
                artifact["id"],
            )["path"])) == artifact["content_hash"]

    after = workspace_repo.get_or_default(task.id)
    assert after.active_dataset_id == before.active_dataset_id == request.dataset_id
    assert after.active_dataset_content_hash == before.active_dataset_content_hash
    assert after.revision == before.revision
    derived = [
        dataset
        for dataset in registry.list_for_task(task.id)
        if dataset.id != request.dataset_id
    ]
    assert len(derived) == 1
    assert derived[0].role == "derived"
    assert derived[0].target_col == request.target_col
