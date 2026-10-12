"""Real signed bytes, HTTP admission and native version execution carriers."""

import base64
from dataclasses import replace
import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient
import pytest

from marvis.app import create_app
from marvis.db_schema import connect
from marvis.domain import TaskCreate
from marvis.orchestrator.contracts import Plan, PlanStep, PlanStatus
from marvis.plugins.manifest import ToolRef, manifest_to_dict, parse_manifest
from marvis.plugins.publisher_trust import (
    SIGNATURE_SCHEMA,
    canonical,
    digest,
    signature_body,
    tree_hash,
)


ROOT = "/api/plugins/governance"


@pytest.fixture
def rt(tmp_path):
    app = create_app(tmp_path / "workspace")
    client = TestClient(app)
    client.get(ROOT)
    private = Ed25519PrivateKey.generate()
    headers = {"x-marvis-plugin-admin": app.state.plugin_admin_token}
    public = base64.b64encode(private.public_key().public_bytes_raw()).decode()
    response = client.post(
        ROOT + "/publishers",
        json={
            "id": "synthetic_vendor",
            "display_name": "Synthetic Vendor",
            "public_key": public,
        },
        headers=headers,
    )
    assert response.status_code == 201, response.text
    return app, client, headers, private, tmp_path


def _plugin(version="1.0.0", permissions=()):
    return {
        "name": "signed_echo",
        "version": version,
        "module": "signed_echo.tools",
        "description": "Synthetic carrier",
        "permissions": list(permissions),
        "hooks": [],
        "tools": [
            {
                "name": "echo",
                "summary": "Echo",
                "entrypoint": "echo",
                "timeout_seconds": 10,
                "input_schema": {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                    "additionalProperties": False,
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "version": {"type": "string"},
                        "message": {"type": "string"},
                    },
                    "required": ["version", "message"],
                    "additionalProperties": False,
                },
                "determinism": "deterministic",
                "failure_policy": "fail",
                "side_effects": [],
            }
        ],
    }


def _archive(
    rt,
    *,
    version="1.0.0",
    permissions=(),
    workflow=None,
    publisher="synthetic_vendor",
    alter=None,
    source=None,
):
    _app, _client, _headers, key, temp = rt
    root = temp / ("pack-" + str(len(list(temp.glob("pack-*")))))
    root.mkdir()
    if workflow is None:
        raw = _plugin(version, permissions)
        (root / "manifest.json").write_text(json.dumps(raw))
        (root / "tools.py").write_text(
            source
            or "def echo(inputs, ctx):\n    return {'version':"
            + repr(version)
            + ",'message':inputs['message']}\n"
        )
        declaration = manifest_to_dict(replace(parse_manifest(raw), checksum=""))
        kind, name = "plugin", "signed_echo"
    else:
        declaration = workflow
        (root / "workflow.json").write_text(json.dumps(workflow))
        kind, name = "workflow", workflow["workflow"]["id"]
    body = signature_body(
        kind=kind,
        name=name,
        release_version=version,
        publisher_id=publisher,
        content_hash=tree_hash(root),
        declaration_hash=digest(declaration),
    )
    signature = base64.b64encode(
        key.sign(SIGNATURE_SCHEMA.encode() + b"\0" + canonical(body))
    ).decode()
    (root / "publisher-signature.json").write_text(
        json.dumps({"body": body, "signature": signature})
    )
    if alter:
        alter(root)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.writestr(
                    name + "/" + path.relative_to(root).as_posix(), path.read_bytes()
                )
    return out.getvalue()


def _propose(rt, data, revision=0, kind="plugin", expected=201):
    _app, client, headers, *_ = rt
    response = client.post(
        ROOT + f"/proposals?kind={kind}&expected_revision={revision}",
        content=data,
        headers=headers | {"content-type": "application/zip"},
    )
    assert response.status_code == expected, response.text
    return response.json()


def _confirm(rt, proposal, permissions=None, expected=201, client=None):
    _app, default, headers, *_ = rt
    response = (client or default).post(
        ROOT + "/proposals/" + proposal["id"] + "/confirm",
        json={
            "proposal_hash": proposal["proposal_hash"],
            "expected_revision": proposal["expected_revision"],
            "approved_permissions": proposal["permissions"]
            if permissions is None
            else permissions,
        },
        headers=headers,
    )
    assert response.status_code == expected, response.text
    return response.json()


def _invoke(rt, version=""):
    app = rt[0]
    return app.state.tool_runner.invoke(
        ToolRef("signed_echo", "echo", version),
        {"message": "synthetic"},
        task_id="task-synthetic",
    )


def _counts(rt):
    with connect(rt[0].state.settings.db_path) as conn:
        return tuple(
            conn.execute(sql).fetchone()[0]
            for sql in (
                "SELECT count(*) FROM plugins WHERE name='signed_echo'",
                "SELECT count(*) FROM capability_versions",
                "SELECT count(*) FROM audit WHERE kind='tool.invoke.started'",
            )
        )


def test_real_http_two_versions_rollback_and_frozen_plan_execute_exact_bytes(rt):
    first = _propose(rt, _archive(rt))
    assert _counts(rt) == (0, 0, 0)
    _confirm(rt, first)
    assert _invoke(rt).output == {"version": "1.0.0", "message": "synthetic"}
    app = rt[0]
    task = app.state.tool_runner._repo.db_path
    from marvis.repositories.tasks import TaskRepository

    created = TaskRepository(task).create_task(
        TaskCreate(
            model_name="Synthetic capability carrier",
            model_version="dev",
            validator="synthetic",
            source_dir=str(app.state.settings.workspace),
            task_type="strategy",
        )
    )
    plan = Plan(
        id="signed-plan",
        task_id=created.id,
        goal="Synthetic",
        source="novel",
        template_id=None,
        steps=[
            PlanStep(
                id="signed-step",
                plan_id="signed-plan",
                index=0,
                title="Echo",
                tool_ref=ToolRef("signed_echo", "echo"),
                inputs={"message": "synthetic"},
                depends_on=[],
                post_checks=[],
            )
        ],
        autonomy_level=1,
        status=PlanStatus.DRAFT,
    )
    app.state.plan_repo.create_plan(plan)
    assert plan.steps[0].tool_ref.version == "1.0.0"
    second = _propose(
        rt, _archive(rt, version="2.0.0", permissions=("read:artifacts",)), revision=1
    )
    assert second["delta"]["added_permissions"] == ["read:artifacts"]
    _confirm(rt, second, permissions=[], expected=403)
    assert _counts(rt)[1] == 1
    _confirm(rt, second)
    assert _invoke(rt).output["version"] == "2.0.0"
    frozen = app.state.plan_repo.load_plan(plan.id).steps[0].tool_ref
    assert _invoke(rt, frozen.version).output["version"] == "1.0.0"
    response = rt[1].post(
        ROOT + "/rollback-proposals",
        json={
            "kind": "plugin",
            "name": "signed_echo",
            "version": "1.0.0",
            "expected_revision": 2,
        },
        headers=rt[2],
    )
    assert response.status_code == 201, response.text
    rollback = response.json()
    receipt = _confirm(rt, rollback)
    assert receipt["revision"] == 3
    assert _invoke(rt).output["version"] == "1.0.0"
    assert _confirm(rt, rollback) == receipt
    restarted = create_app(app.state.settings.workspace)
    result = restarted.state.tool_runner.invoke(
        frozen, {"message": "after restart"}, task_id="task-synthetic"
    )
    assert result.ok, result.error
    assert result.output == {"version": "1.0.0", "message": "after restart"}


@pytest.mark.parametrize("tamper", ["code", "permissions", "tool", "hook", "version"])
def test_signed_content_tampering_has_zero_registration_and_execution(rt, tamper):
    def alter(root):
        if tamper == "code":
            (root / "tools.py").write_text('raise RuntimeError("must never import")')
        else:
            path = root / "manifest.json"
            raw = json.loads(path.read_text())
            if tamper == "permissions":
                raw["permissions"] = ["network:optional"]
            if tamper == "tool":
                raw["tools"][0]["summary"] = "Changed"
            if tamper == "hook":
                raw["hooks"] = [{"event": "task.created", "tool": "echo"}]
            if tamper == "version":
                raw["version"] = "9.0.0"
            path.write_text(json.dumps(raw))

    _propose(rt, _archive(rt, alter=alter), expected=422)
    assert _counts(rt) == (0, 0, 0)


def test_unknown_publisher_and_rejected_permission_never_register(rt):
    _propose(rt, _archive(rt, publisher="untrusted_vendor"), expected=403)
    assert _counts(rt) == (0, 0, 0)
    proposal = _propose(rt, _archive(rt, permissions=("network:optional",)))
    _confirm(rt, proposal, permissions=[], expected=403)
    assert _counts(rt) == (0, 0, 0)


def test_revocation_blocks_old_version_native_dispatch_and_rollback(rt):
    proposal = _propose(rt, _archive(rt))
    _confirm(rt, proposal)
    before = _counts(rt)[2]
    revoked = rt[1].post(
        ROOT + "/publishers/synthetic_vendor/revoke",
        json={"reason": "Synthetic revocation"},
        headers=rt[2],
    )
    assert revoked.status_code == 200, revoked.text
    result = _invoke(rt, "1.0.0")
    assert not result.ok and result.error_kind == "permission"
    assert _counts(rt)[2] == before
    response = rt[1].post(
        ROOT + "/rollback-proposals",
        json={
            "kind": "plugin",
            "name": "signed_echo",
            "version": "1.0.0",
            "expected_revision": 1,
        },
        headers=rt[2],
    )
    assert response.status_code == 403, response.text


def test_revocation_after_effect_reservation_releases_without_worker_dispatch(
    rt, monkeypatch
):
    from types import SimpleNamespace
    from test_plugin_runner import _execution_context

    _confirm(rt, _propose(rt, _archive(rt)))
    events = []

    class Governance:
        def reserve_effect(self, *_args, **_kwargs):
            events.append("reserved")
            response = rt[1].post(
                ROOT + "/publishers/synthetic_vendor/revoke",
                json={"reason": "Synthetic pre-dispatch withdrawal"},
                headers=rt[2],
            )
            assert response.status_code == 200, response.text
            return SimpleNamespace(
                id="synthetic-effect", reservation_id="synthetic-reservation"
            )

        def release_prepared_effect(self, effect_id, *, reservation_id, reason):
            events.append(("released", effect_id, reservation_id, reason))

    runner = rt[0].state.tool_runner
    runner._governance = Governance()
    runner._binding_resolver = lambda **_kwargs: object()
    monkeypatch.setattr(
        "marvis.plugins.runner._run_worker",
        lambda *_args, **_kwargs: pytest.fail("revoked capability must not dispatch"),
    )
    result = runner.invoke(
        ToolRef("signed_echo", "echo", "1.0.0"),
        {"message": "synthetic"},
        task_id="task-synthetic",
        execution_context=_execution_context(),
    )
    assert not result.ok and result.error_kind == "permission"
    assert events[0] == "reserved"
    assert events[1][:3] == ("released", "synthetic-effect", "synthetic-reservation")
    assert len(events) == 2


def test_workspace_governed_mode_blocks_legacy_install_and_adhoc(rt):
    response = rt[1].put(
        ROOT + "/settings",
        json={
            "mode": "governed",
            "expected_revision": 0,
            "reason": "Synthetic governance",
        },
        headers=rt[2],
    )
    assert response.status_code == 200, response.text
    data = _archive(rt)
    legacy = rt[1].post(
        "/api/plugins",
        content=data,
        headers=rt[2] | {"content-type": "application/zip"},
    )
    assert legacy.status_code == 403, legacy.text
    assert _counts(rt) == (0, 0, 0)
    source = rt[4] / "adhoc.py"
    source.write_text('raise RuntimeError("must not execute")')
    result = rt[0].state.tool_runner.invoke_adhoc(
        module=source,
        entrypoint="run",
        inputs={},
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        timeout_seconds=1,
        task_id="synthetic",
    )
    assert not result.ok and result.error_kind == "permission"
    assert _counts(rt)[2] == 0
    _confirm(rt, _propose(rt, data))
    assert _invoke(rt).ok


def test_confirmation_actor_and_head_cas_prevent_second_concurrent_activation(rt):
    one = _propose(rt, _archive(rt))
    two = _propose(rt, _archive(rt, version="2.0.0"))
    other = TestClient(rt[0])
    other.get(ROOT)
    _confirm(rt, one, client=other, expected=403)
    _confirm(rt, one)
    _confirm(rt, two, expected=409)
    assert _counts(rt)[:2] == (1, 1)


def test_existing_version_cannot_be_rewritten_or_bypassed_with_legacy_upload(rt):
    _confirm(rt, _propose(rt, _archive(rt)))
    proposal = _propose(rt, _archive(rt, permissions=("network:optional",)), revision=1)
    _confirm(rt, proposal, expected=409)
    legacy = rt[1].post(
        "/api/plugins", content=_archive(rt, version="2.0.0"), headers=rt[2]
    )
    assert legacy.status_code == 403, legacy.text
    assert _counts(rt)[:2] == (1, 1)


def _workflow(rt, version="1.0.0"):
    manifest, _tool = rt[0].state.tool_registry.resolve_with_manifest(
        ToolRef("signed_echo", "echo", version)
    )
    return {
        "workflow": {
            "id": "signed_flow",
            "title": "Synthetic signed flow",
            "goal_patterns": ["synthetic"],
            "slots": [],
            "steps": [
                {
                    "title": "Echo",
                    "tool": {
                        "plugin": "signed_echo",
                        "tool": "echo",
                        "version": version,
                    },
                    "inputs": {"message": "workflow"},
                    "depends_on": [],
                    "post_checks": [],
                }
            ],
        },
        "tool_bindings": [
            {
                "plugin": "signed_echo",
                "tool": "echo",
                "version": version,
                "declaration_hash": digest(manifest_to_dict(manifest)),
            }
        ],
        "permissions": sorted(manifest.permissions),
    }


def test_governed_workflow_uses_existing_parser_and_retains_two_heads(rt):
    _confirm(rt, _propose(rt, _archive(rt)))
    first = _propose(rt, _archive(rt, workflow=_workflow(rt)), kind="workflow")
    _confirm(rt, first)
    assert "signed_flow" in rt[0].state.skill_report.active
    _confirm(rt, _propose(rt, _archive(rt, version="2.0.0"), revision=1))
    second = _propose(
        rt,
        _archive(rt, version="2.0.0", workflow=_workflow(rt, "2.0.0")),
        revision=1,
        kind="workflow",
    )
    _confirm(rt, second)
    from marvis.orchestrator.templates import get_template

    assert get_template("signed_flow").steps[0].tool_ref.version == "2.0.0"
    response = rt[1].post(
        "/api/skills/governance/rollback-proposals",
        json={
            "kind": "workflow",
            "name": "signed_flow",
            "version": "1.0.0",
            "expected_revision": 2,
        },
        headers=rt[2],
    )
    assert response.status_code == 201, response.text
    _confirm(rt, response.json())
    assert get_template("signed_flow").steps[0].tool_ref.version == "1.0.0"
    create_app(rt[0].state.settings.workspace)
    assert get_template("signed_flow").steps[0].tool_ref.version == "1.0.0"


def test_governed_reload_rejects_unsigned_local_workflow_and_name_shadow(rt):
    _confirm(rt, _propose(rt, _archive(rt)))
    signed = _workflow(rt)
    _confirm(rt, _propose(rt, _archive(rt, workflow=signed), kind="workflow"))
    directory = rt[0].state.settings.workspace / "skills"
    directory.mkdir()
    (directory / "signed_flow.json").write_text(json.dumps(signed["workflow"]))
    response = rt[1].post("/api/skills/reload")
    assert response.status_code == 200, response.text
    assert any(
        item["id"] == "signed_flow" and item["status"] == "rejected"
        for item in response.json()["skills"]
    )
    from marvis.orchestrator.templates import get_template

    assert get_template("signed_flow").steps[0].tool_ref.version == "1.0.0"


def test_real_simultaneous_http_confirmation_has_one_head_and_one_activation(rt):
    first = _propose(rt, _archive(rt))
    second = _propose(rt, _archive(rt, version="2.0.0"))
    cookies = dict(rt[1].cookies)

    def confirm(proposal):
        with TestClient(rt[0]) as client:
            client.cookies.update(cookies)
            return client.post(
                ROOT + "/proposals/" + proposal["id"] + "/confirm",
                headers=rt[2],
                json={
                    "proposal_hash": proposal["proposal_hash"],
                    "expected_revision": 0,
                    "approved_permissions": [],
                },
            ).status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(confirm, [first, second]))
    assert sorted(results) == [201, 409]
    with connect(rt[0].state.settings.db_path) as conn:
        assert (
            conn.execute("SELECT count(*) FROM capability_versions").fetchone()[0] == 1
        )
        assert (
            conn.execute(
                "SELECT count(*) FROM audit WHERE kind='capability.activate'"
            ).fetchone()[0]
            == 1
        )
    result = _invoke(rt)
    assert result.ok, result.error
    assert result.output["version"] in {"1.0.0", "2.0.0"}


def _task(rt, *, run_mode="agent"):
    from marvis.repositories.tasks import TaskRepository

    app = rt[0]
    return TaskRepository(app.state.settings.db_path).create_task(
        TaskCreate(
            model_name="Synthetic signed workflow",
            model_version="dev",
            validator="synthetic",
            source_dir=str(app.state.settings.workspace),
            task_type="strategy",
            run_mode=run_mode,
        )
    )


def test_signed_workflow_reaches_existing_plan_create_confirm_run_http(rt):
    from marvis.orchestrator.reviewer import Reviewer
    from test_orch_executor import FakeLLM

    _confirm(rt, _propose(rt, _archive(rt)))
    workflow = _workflow(rt)
    goal = "运行签名合成工作流载体验证"
    workflow["workflow"]["goal_patterns"] = [goal]
    _confirm(rt, _propose(rt, _archive(rt, workflow=workflow), kind="workflow"))
    app, client, *_ = rt
    task = _task(rt, run_mode="manual")
    routing_calls = []

    def no_provider():
        routing_calls.append(True)
        raise AssertionError("exact installed template must not call a provider")

    app.state.intent_router._llm_factory = no_provider
    # Only the narrative critic is synthetic; HTTP planning, CAS, persisted
    # binding, executor, worker and output lookup use the application runtime.
    app.state.plan_executor._reviewer = Reviewer(lambda: FakeLLM())
    created = client.post(f"/api/tasks/{task.id}/plans", json={"goal": goal})
    assert created.status_code == 201, created.text
    plan = created.json()["plan"]
    assert plan["template_id"] == "signed_flow"
    assert plan["steps"][0]["tool_ref"]["version"] == "1.0.0"
    confirmed = client.post(
        f"/api/plans/{plan['id']}/confirm", json=plan["confirmation_snapshot"]
    )
    assert confirmed.status_code == 200, confirmed.text
    snapshot = confirmed.json()["plan"]["confirmation_snapshot"]
    response = client.post(
        f"/api/plans/{plan['id']}/run",
        json={"expected_plan_fingerprint": snapshot["expected_plan_fingerprint"]},
    )
    assert response.status_code == 202, response.text
    finished = client.get(f"/api/plans/{plan['id']}")
    assert finished.status_code == 200, finished.text
    assert finished.json()["plan"]["status"] == "done"
    output = client.get(f"/api/step-outputs/{plan['steps'][0]['id']}")
    assert output.status_code == 200, output.text
    assert output.json() == {"version": "1.0.0", "message": "workflow"}
    assert not routing_calls


def test_native_workflow_executor_keeps_original_version_after_both_heads_change(rt):
    from marvis.orchestrator.templates import get_template
    from test_orch_executor import _executor

    _confirm(rt, _propose(rt, _archive(rt)))
    _confirm(rt, _propose(rt, _archive(rt, workflow=_workflow(rt)), kind="workflow"))
    app = rt[0]
    task = _task(rt)
    plan = app.state.planner.from_template(get_template("signed_flow"), {}, task.id)
    plan.status = PlanStatus.CONFIRMED
    app.state.plan_repo.create_plan(plan)
    _confirm(rt, _propose(rt, _archive(rt, version="2.0.0"), revision=1))
    _confirm(
        rt,
        _propose(
            rt,
            _archive(rt, version="2.0.0", workflow=_workflow(rt, "2.0.0")),
            revision=1,
            kind="workflow",
        ),
    )
    result = _executor(app.state.plan_repo, app.state.tool_runner).run(plan.id)
    assert result.status == PlanStatus.DONE
    persisted = app.state.plan_repo.load_plan(plan.id)
    assert persisted.steps[0].tool_ref.version == "1.0.0"
    output = app.state.plan_repo.load_step_output(persisted.steps[0].id)
    assert output == {"version": "1.0.0", "message": "workflow"}


def test_revoked_workflow_rejects_before_native_first_step(rt):
    from marvis.orchestrator.templates import get_template
    from test_orch_executor import _executor

    _confirm(rt, _propose(rt, _archive(rt)))
    _confirm(rt, _propose(rt, _archive(rt, workflow=_workflow(rt)), kind="workflow"))
    app = rt[0]
    task = _task(rt)
    plan = app.state.planner.from_template(get_template("signed_flow"), {}, task.id)
    plan.status = PlanStatus.CONFIRMED
    app.state.plan_repo.create_plan(plan)
    response = rt[1].post(
        ROOT + "/publishers/synthetic_vendor/revoke",
        json={"reason": "Synthetic review"},
        headers=rt[2],
    )
    assert response.status_code == 200, response.text
    result = _executor(app.state.plan_repo, app.state.tool_runner).run(plan.id)
    assert result.status == PlanStatus.CANCELLED
    assert _counts(rt)[2] == 0
    assert (
        app.state.plan_repo.load_plan(plan.id).loop_events[-1].type
        == "capability_admission_rejected"
    )


@pytest.mark.parametrize("phase", ["proposal", "installed"])
def test_actual_retained_file_tamper_is_rejected_before_native_read(rt, phase):
    proposal = _propose(rt, _archive(rt))
    if phase == "installed":
        _confirm(rt, proposal)
    with connect(rt[0].state.settings.db_path) as conn:
        material = conn.execute(
            "SELECT material_path FROM capability_proposals WHERE id=?",
            (proposal["id"],),
        ).fetchone()[0]
    from pathlib import Path

    source = Path(material) / "tools.py"
    source.chmod(0o600)
    source.write_text('raise RuntimeError("must never import modified bytes")')
    if phase == "proposal":
        _confirm(rt, proposal, expected=409)
    else:
        result = _invoke(rt)
        assert not result.ok and result.error_kind == "permission"
    assert _counts(rt)[2] == 0


def test_invalid_package_and_foreign_selected_target_are_typed_zero_registration(rt):
    _propose(rt, b"not a ZIP", expected=422)
    response = rt[1].post(
        ROOT + "/proposals?kind=plugin&expected_revision=0&expected_name=other_name",
        content=_archive(rt),
        headers=rt[2],
    )
    assert response.status_code == 422, response.text
    assert response.json()["detail"] == "capability_target_changed"
    assert _counts(rt) == (0, 0, 0)


def test_unsigned_workflow_cannot_reload_in_explicit_governed_mode(rt):
    _confirm(rt, _propose(rt, _archive(rt)))
    response = rt[1].put(
        ROOT + "/settings",
        json={"mode": "governed", "expected_revision": 0, "reason": "Synthetic"},
        headers=rt[2],
    )
    assert response.status_code == 200, response.text
    directory = rt[0].state.settings.workspace / "skills"
    directory.mkdir()
    (directory / "unsigned.json").write_text(
        json.dumps(_workflow(rt)["workflow"] | {"id": "unsigned"})
    )
    response = rt[1].post("/api/skills/reload")
    assert response.status_code == 200
    assert any(
        item["id"] == "unsigned" and item["status"] == "rejected"
        for item in response.json()["skills"]
    )
    assert "unsigned" not in rt[0].state.skill_report.active


def test_governed_workflow_rejects_inline_credentials_and_wrong_dependency_hash(rt):
    _confirm(rt, _propose(rt, _archive(rt)))
    declaration = _workflow(rt)
    declaration["workflow"]["steps"][0]["inputs"]["api_key"] = (
        "synthetic-secret-placeholder"
    )
    _propose(rt, _archive(rt, workflow=declaration), kind="workflow", expected=422)
    declaration = _workflow(rt)
    declaration["tool_bindings"][0]["declaration_hash"] = "0" * 64
    _propose(rt, _archive(rt, workflow=declaration), kind="workflow", expected=409)
    assert _counts(rt) == (1, 1, 0)


def test_active_legacy_plan_cannot_be_silently_rebound_at_first_signed_install(rt):
    from marvis.plugins.loader import install_plugin

    archive = _archive(rt)
    path = rt[4] / "legacy.zip"
    path.write_bytes(archive)
    install_plugin(path, rt[0].state.settings.plugins_dir, rt[0].state.plugin_registry)
    app = rt[0]
    task = _task(rt)
    plan = Plan(
        id="legacy-plan",
        task_id=task.id,
        goal="Synthetic",
        source="novel",
        template_id=None,
        steps=[
            PlanStep(
                id="legacy-step",
                plan_id="legacy-plan",
                index=0,
                title="Echo",
                tool_ref=ToolRef("signed_echo", "echo"),
                inputs={"message": "original"},
                depends_on=[],
                post_checks=[],
            )
        ],
        autonomy_level=1,
    )
    app.state.plan_repo.create_plan(plan)
    proposal = _propose(rt, archive)
    result = _confirm(rt, proposal, expected=409)
    assert result["detail"] == "capability_active_legacy_plan_requires_completion"
    assert _counts(rt)[:2] == (1, 0)


def test_sql_register_failure_rolls_back_version_head_and_confirmation(rt, monkeypatch):
    import marvis.repositories.plugins as repository_module

    proposal = _propose(rt, _archive(rt))
    original = repository_module._upsert_plugin_row

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("synthetic registration failure")

    with monkeypatch.context() as patch:
        patch.setattr(repository_module, "_upsert_plugin_row", fail)
        with pytest.raises(RuntimeError, match="synthetic registration failure"):
            _confirm(rt, proposal)
    assert _counts(rt) == (0, 0, 0)
    assert (
        rt[1].get(ROOT + "/proposals/" + proposal["id"], headers=rt[2]).json()["status"]
        == "pending"
    )
    assert rt[1].get(ROOT).json()["versions"] == []
    _confirm(rt, proposal)
    assert _invoke(rt).ok


def test_confirmation_replay_cannot_change_approved_permission_payload(rt):
    proposal = _propose(rt, _archive(rt, permissions=("network:optional",)))
    receipt = _confirm(rt, proposal)
    _confirm(rt, proposal, permissions=[], expected=403)
    assert _confirm(rt, proposal) == receipt
    with connect(rt[0].state.settings.db_path) as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM audit WHERE kind='capability.activate'"
            ).fetchone()[0]
            == 1
        )


def test_governed_mode_blocks_repository_registration_and_promotion_write(rt):
    from marvis.plugins.publisher_trust import CapabilityError

    response = rt[1].put(
        ROOT + "/settings",
        json={"mode": "governed", "expected_revision": 0, "reason": "Synthetic"},
        headers=rt[2],
    )
    assert response.status_code == 200
    manifest = parse_manifest(_plugin())
    with pytest.raises(CapabilityError, match="signed_capability_required"):
        rt[0].state.plugin_registry.register(manifest)
    with pytest.raises(CapabilityError, match="signed_capability_required"):
        rt[0].state.plugin_repo.upsert_plugin(manifest, enabled=True)
    assert _counts(rt) == (0, 0, 0)


def test_governed_versions_and_publisher_identity_are_immutable_in_actual_sqlite(rt):
    import sqlite3

    _confirm(rt, _propose(rt, _archive(rt)))
    for statement in (
        "UPDATE capability_versions SET version='9.0.0'",
        "DELETE FROM capability_versions",
        "UPDATE capability_publishers SET public_key='changed'",
        "DELETE FROM capability_publishers",
    ):
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            with connect(rt[0].state.settings.db_path) as conn:
                conn.execute(statement)
    assert _invoke(rt).output["version"] == "1.0.0"


def test_builtin_dependency_drift_stops_signed_workflow_before_worker(rt):
    from marvis.orchestrator.templates import get_template
    from test_orch_executor import _executor

    builtin = rt[0].state.plugin_registry.get("_sample")
    ref = ToolRef("_sample", "echo", builtin.version)
    declaration = {
        "workflow": {
            "id": "signed_flow",
            "title": "Synthetic builtin flow",
            "goal_patterns": [],
            "slots": [],
            "steps": [
                {
                    "title": "Echo",
                    "tool": {
                        "plugin": ref.plugin,
                        "tool": ref.tool,
                        "version": ref.version,
                    },
                    "inputs": {"message": "synthetic"},
                    "depends_on": [],
                    "post_checks": [],
                }
            ],
        },
        "tool_bindings": [
            {
                "plugin": ref.plugin,
                "tool": ref.tool,
                "version": ref.version,
                "declaration_hash": digest(manifest_to_dict(builtin)),
            }
        ],
        "permissions": sorted(builtin.permissions),
    }
    _confirm(rt, _propose(rt, _archive(rt, workflow=declaration), kind="workflow"))
    app = rt[0]
    task = _task(rt)
    plan = app.state.planner.from_template(get_template("signed_flow"), {}, task.id)
    plan.status = PlanStatus.CONFIRMED
    app.state.plan_repo.create_plan(plan)
    app.state.plugin_repo.upsert_plugin(
        replace(builtin, description="changed synthetic producer declaration"),
        enabled=True,
    )
    result = _executor(app.state.plan_repo, app.state.tool_runner).run(plan.id)
    assert result.status == PlanStatus.CANCELLED
    assert _counts(rt)[2] == 0
    assert (
        app.state.plan_repo.load_plan(plan.id).loop_events[-1].reason
        == "workflow_tool_declaration_changed"
    )
