import json
import sys

import pytest

import marvis.repositories.drafts as draft_repo_module
import marvis.repositories.plugins as plugin_repo_module
from marvis.db import DraftRepository, PluginRepository, init_db
from marvis.db_schema import connect
from marvis.drafts import DraftTool
from marvis.drafts.promotion import (
    PromotionError,
    promote_draft,
    reject_draft,
    validate_for_promotion,
)
from marvis.plugins.errors import ManifestError, PluginNotFoundError, ToolNotFoundError
from marvis.drafts.registry import DraftRegistry
from marvis.drafts.sandbox import DraftSandbox
from marvis.plugins.manifest import (
    EXECUTION_PROFILE_DRAFT_REPROMOTION_REQUIRED,
    EXECUTION_PROFILE_DRAFT_RESTRICTED_V1,
    EXECUTION_PROFILE_STANDARD,
    ToolRef,
)
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner


def _runtime(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    plugin_repo = PluginRepository(db_path)
    plugin_registry = PluginRegistry(plugin_repo)
    tool_runner = ToolRunner(
        ToolRegistry(plugin_registry),
        plugin_repo,
        python_executable=sys.executable,
        datasets_root=tmp_path / "datasets",
        workspace=tmp_path / "workspace",
        plugin_paths=[tmp_path / "plugins"],
    )
    draft_repo = DraftRepository(db_path)
    drafts = DraftRegistry(draft_repo)
    sandbox = DraftSandbox(tool_runner, drafts, draft_repo)
    return sandbox, drafts, plugin_registry, tmp_path / "plugins"


def _draft(**overrides) -> DraftTool:
    payload = {
        "id": "draft-1",
        "task_id": "task-1",
        "name": "calc_margin",
        "summary": "Calculate margin.",
        "code": "def calc_margin(inputs, ctx):\n    return {'margin': inputs['revenue'] - inputs['cost']}\n",
        "input_schema": {
            "type": "object",
            "properties": {"revenue": {"type": "number"}, "cost": {"type": "number"}},
            "required": ["revenue", "cost"],
            "additionalProperties": False,
        },
        "output_schema": {
            "type": "object",
            "properties": {"margin": {"type": "number"}},
            "required": ["margin"],
            "additionalProperties": False,
        },
        "determinism": "deterministic",
        "source": "hand_written",
        "learning_note_id": None,
        "status": "draft",
        "created_at": "2026-06-19T00:00:00Z",
    }
    payload.update(overrides)
    return DraftTool(**payload)


def test_validate_for_promotion_requires_schema_determinism_and_tests(tmp_path):
    sandbox, _drafts, _plugin_registry, _plugins_dir = _runtime(tmp_path)
    draft = _draft(input_schema={}, output_schema={}, determinism="unknown")

    check = validate_for_promotion(draft, sandbox=sandbox, test_cases=[])

    assert check.passed is False
    assert "missing schema" in check.problems
    assert "determinism not declared" in check.problems
    assert "at least one test case required" in check.problems
    assert check.test_result is None


def test_validate_and_promote_draft_registers_formal_plugin(tmp_path):
    sandbox, drafts, plugin_registry, plugins_dir = _runtime(tmp_path)
    draft = _draft()
    drafts.add(draft)

    check = validate_for_promotion(
        draft,
        sandbox=sandbox,
        test_cases=[{"inputs": {"revenue": 10, "cost": 3}, "expect": {"margin": 7}}],
    )
    manifest = promote_draft(draft, registry=plugin_registry, drafts=drafts, plugins_dir=plugins_dir, check=check)
    catalog = ToolRegistry(plugin_registry).catalog_for_planner()

    assert check.passed is True
    assert check.test_result == {"passed": True, "n": 1}
    assert plugin_registry.get(manifest.name).tools[0].name == "calc_margin"
    assert any(item["plugin"] == manifest.name and item["tool"] == "calc_margin" for item in catalog)
    assert manifest.draft_promotion is not None
    assert manifest.draft_promotion.plugin_name == manifest.name
    assert manifest.draft_promotion.plugin_version == manifest.version
    assert manifest.draft_promotion.plugin_checksum == manifest.checksum
    assert manifest.draft_promotion.tool_name == "calc_margin"
    audit = PluginRepository(plugins_dir.parent / "app.sqlite").list_audit(
        kind="draft.promote",
        limit=None,
    )
    assert audit[-1]["detail"]["draft_promotion"] == manifest.draft_promotion.to_dict()
    assert drafts.get(draft.id).status == "promoted"
    assert (plugins_dir / manifest.name / "manifest.json").exists()
    assert not (plugins_dir / ".staging").exists()


def test_promoted_draft_stays_restricted_when_invoked_by_normal_runner(tmp_path):
    sandbox, drafts, plugin_registry, plugins_dir = _runtime(tmp_path)
    draft = _draft()
    drafts.add(draft)
    check = validate_for_promotion(
        draft,
        sandbox=sandbox,
        test_cases=[{"inputs": {"revenue": 10, "cost": 3}, "expect": {"margin": 7}}],
    )
    manifest = promote_draft(
        draft,
        registry=plugin_registry,
        drafts=drafts,
        plugins_dir=plugins_dir,
        check=check,
    )

    tool = manifest.tools[0]
    assert tool.execution_profile == EXECUTION_PROFILE_DRAFT_RESTRICTED_V1
    first = sandbox._runner.invoke(
        ToolRef(manifest.name, tool.name),
        {"revenue": 10, "cost": 3},
        task_id="task-1",
    )
    assert first.ok is True, first.error
    assert first.output == {"margin": 7}

    # The manifest profile, not a module-name heuristic, picks the restricted
    # loader for the normal runner path too. Changing the plugin-root source to
    # unsupported Python is therefore fail-closed before execution.
    (plugins_dir / manifest.name / "tools.py").write_text(
        "import os\n"
        "def calc_margin(inputs, ctx):\n"
        "    return {'margin': len(os.environ)}\n",
        encoding="utf-8",
    )
    blocked = sandbox._runner.invoke(
        ToolRef(manifest.name, tool.name),
        {"revenue": 10, "cost": 3},
        task_id="task-1",
    )

    assert blocked.ok is False
    assert blocked.error_kind == "execution"
    assert "Draft Language v1 rewrite required" in blocked.error
    assert "import 'os'" in blocked.error


def test_legacy_promoted_draft_is_restricted_from_audit_provenance(tmp_path):
    sandbox, drafts, plugin_registry, plugins_dir = _runtime(tmp_path)
    draft = _draft()
    drafts.add(draft)
    check = validate_for_promotion(
        draft,
        sandbox=sandbox,
        test_cases=[{"inputs": {"revenue": 10, "cost": 3}, "expect": {"margin": 7}}],
    )
    manifest = promote_draft(
        draft,
        registry=plugin_registry,
        drafts=drafts,
        plugins_dir=plugins_dir,
        check=check,
    )
    db_path = plugins_dir.parent / "app.sqlite"
    raw = json.loads(PluginRepository(db_path).get_plugin(manifest.name)["manifest_json"])
    raw.pop("draft_promotion")
    raw["tools"][0].pop("execution_profile")
    with connect(db_path) as connection:
        connection.execute(
            "UPDATE plugins SET manifest_json = ? WHERE name = ?",
            (json.dumps(raw), manifest.name),
        )
        connection.execute(
            "UPDATE audit SET detail_json = ? WHERE kind = ? AND target_ref = ?",
            (
                json.dumps({"plugin": manifest.name, "tests": check.test_result}),
                "draft.promote",
                draft.id,
            ),
        )

    reloaded = PluginRegistry(PluginRepository(db_path))
    reloaded.load_from_db()
    restored = reloaded.get(manifest.name)

    assert restored.tools[0].execution_profile == EXECUTION_PROFILE_DRAFT_RESTRICTED_V1
    assert restored.draft_promotion is not None
    assert restored.draft_promotion.plugin_name == manifest.name
    assert restored.draft_promotion.plugin_version == manifest.version
    assert restored.draft_promotion.plugin_checksum == manifest.checksum
    persisted = json.loads(PluginRepository(db_path).get_plugin(manifest.name)["manifest_json"])
    assert persisted["draft_promotion"] == restored.draft_promotion.to_dict()
    runner = ToolRunner(
        ToolRegistry(reloaded),
        PluginRepository(db_path),
        python_executable=sys.executable,
        datasets_root=tmp_path / "datasets",
        workspace=tmp_path / "workspace",
        plugin_paths=[plugins_dir],
    )
    result = runner.invoke(
        ToolRef(restored.name, restored.tools[0].name),
        {"revenue": 10, "cost": 3},
        task_id="task-1",
    )
    assert result.ok is True, result.error
    assert result.output == {"margin": 7}
    (plugins_dir / manifest.name / "tools.py").write_text(
        "import os\n"
        "def calc_margin(inputs, ctx):\n"
        "    return {'margin': len(os.environ)}\n",
        encoding="utf-8",
    )
    blocked = runner.invoke(
        ToolRef(restored.name, restored.tools[0].name),
        {"revenue": 10, "cost": 3},
        task_id="task-1",
    )
    assert blocked.ok is False
    assert "Draft Language v1 rewrite required" in blocked.error


def test_legacy_draft_with_missing_provenance_is_not_loaded_as_standard(tmp_path):
    sandbox, drafts, plugin_registry, plugins_dir = _runtime(tmp_path)
    draft = _draft()
    drafts.add(draft)
    check = validate_for_promotion(
        draft,
        sandbox=sandbox,
        test_cases=[{"inputs": {"revenue": 10, "cost": 3}, "expect": {"margin": 7}}],
    )
    manifest = promote_draft(
        draft,
        registry=plugin_registry,
        drafts=drafts,
        plugins_dir=plugins_dir,
        check=check,
    )
    db_path = plugins_dir.parent / "app.sqlite"
    raw = json.loads(PluginRepository(db_path).get_plugin(manifest.name)["manifest_json"])
    raw.pop("draft_promotion")
    raw["tools"][0].pop("execution_profile")
    with connect(db_path) as connection:
        connection.execute(
            "UPDATE plugins SET manifest_json = ? WHERE name = ?",
            (json.dumps(raw), manifest.name),
        )
        connection.execute("DELETE FROM audit WHERE kind = ?", ("draft.promote",))

    reloaded = PluginRegistry(PluginRepository(db_path))
    reloaded.load_from_db()
    restored = reloaded.get(manifest.name)

    assert restored.tools[0].execution_profile == EXECUTION_PROFILE_DRAFT_REPROMOTION_REQUIRED
    assert restored.draft_promotion is None
    assert ToolRegistry(reloaded).catalog_for_planner() == []
    runner = ToolRunner(
        ToolRegistry(reloaded),
        PluginRepository(db_path),
        python_executable=sys.executable,
        datasets_root=tmp_path / "datasets",
        workspace=tmp_path / "workspace",
        plugin_paths=[plugins_dir],
    )
    with pytest.raises(ToolNotFoundError, match="re-promoted"):
        runner.invoke(
            ToolRef(restored.name, restored.tools[0].name),
            {"revenue": 10, "cost": 3},
            task_id="task-1",
        )


def test_unreadable_promotion_audit_blocks_profileless_legacy_artifact(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    sandbox, drafts, plugin_registry, plugins_dir = _runtime(tmp_path)
    draft = _draft()
    drafts.add(draft)
    check = validate_for_promotion(
        draft,
        sandbox=sandbox,
        test_cases=[{"inputs": {"revenue": 10, "cost": 3}, "expect": {"margin": 7}}],
    )
    manifest = promote_draft(
        draft,
        registry=plugin_registry,
        drafts=drafts,
        plugins_dir=plugins_dir,
        check=check,
    )
    db_path = plugins_dir.parent / "app.sqlite"
    raw = json.loads(PluginRepository(db_path).get_plugin(manifest.name)["manifest_json"])
    raw.pop("draft_promotion")
    raw["tools"][0].pop("execution_profile")
    with connect(db_path) as connection:
        connection.execute(
            "UPDATE plugins SET manifest_json = ? WHERE name = ?",
            (json.dumps(raw), manifest.name),
        )

    repo = PluginRepository(db_path)

    def fail_list_audit(**_kwargs):
        raise RuntimeError("audit database unavailable")

    monkeypatch.setattr(repo, "list_audit", fail_list_audit)
    with pytest.raises(ManifestError, match="cannot read Draft promotion provenance"):
        PluginRegistry(repo).load_from_db()


def test_later_same_name_ordinary_plugin_is_not_restricted_by_legacy_audit(tmp_path):
    sandbox, drafts, plugin_registry, plugins_dir = _runtime(tmp_path)
    draft = _draft()
    drafts.add(draft)
    check = validate_for_promotion(
        draft,
        sandbox=sandbox,
        test_cases=[{"inputs": {"revenue": 10, "cost": 3}, "expect": {"margin": 7}}],
    )
    manifest = promote_draft(
        draft,
        registry=plugin_registry,
        drafts=drafts,
        plugins_dir=plugins_dir,
        check=check,
    )
    db_path = plugins_dir.parent / "app.sqlite"
    raw = json.loads(PluginRepository(db_path).get_plugin(manifest.name)["manifest_json"])
    raw.pop("draft_promotion")
    raw["version"] = "0.2.0"
    raw["checksum"] = "ordinary-plugin-checksum"
    raw["module"] = "ordinary_plugin.tools"
    raw["tools"][0].pop("execution_profile")
    with connect(db_path) as connection:
        connection.execute(
            """
            UPDATE plugins
               SET version = ?, checksum = ?, module = ?, manifest_json = ?, installed_at = ?
             WHERE name = ?
            """,
            (
                raw["version"],
                raw["checksum"],
                raw["module"],
                json.dumps(raw),
                "2099-01-01T00:00:00+00:00",
                manifest.name,
            ),
        )
        connection.execute(
            "UPDATE audit SET detail_json = ? WHERE kind = ? AND target_ref = ?",
            (
                json.dumps({"plugin": manifest.name, "tests": check.test_result}),
                "draft.promote",
                draft.id,
            ),
        )

    reloaded = PluginRegistry(PluginRepository(db_path))
    reloaded.load_from_db()
    ordinary = reloaded.get(manifest.name)

    assert ordinary.version == "0.2.0"
    assert ordinary.checksum == "ordinary-plugin-checksum"
    assert ordinary.tools[0].execution_profile == EXECUTION_PROFILE_STANDARD
    assert ordinary.draft_promotion is None
    assert ToolRegistry(reloaded).catalog_for_planner()[0]["plugin"] == manifest.name


def test_promote_draft_rolls_back_plugin_and_files_when_draft_audit_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    sandbox, drafts, plugin_registry, plugins_dir = _runtime(tmp_path)
    draft = _draft()
    drafts.add(draft)
    check = validate_for_promotion(
        draft,
        sandbox=sandbox,
        test_cases=[{"inputs": {"revenue": 10, "cost": 3}, "expect": {"margin": 7}}],
    )
    status_before_promote = drafts.get(draft.id).status
    original_write_audit = plugin_repo_module._write_audit_row

    def fail_draft_promote_audit(conn, *args, **kwargs):
        if kwargs.get("kind") == "draft.promote":
            raise RuntimeError("audit down")
        return original_write_audit(conn, *args, **kwargs)

    monkeypatch.setattr(plugin_repo_module, "_write_audit_row", fail_draft_promote_audit)

    with pytest.raises(RuntimeError, match="audit down"):
        promote_draft(draft, registry=plugin_registry, drafts=drafts, plugins_dir=plugins_dir, check=check)

    assert status_before_promote == "tested"
    assert drafts.get(draft.id).status == status_before_promote
    assert PluginRepository(plugins_dir.parent / "app.sqlite").get_plugin("draft_calc_margin") is None
    with pytest.raises(PluginNotFoundError):
        plugin_registry.get("draft_calc_margin")
    assert not (plugins_dir / "draft_calc_margin").exists()
    assert not (plugins_dir / ".staging").exists()
    assert not any(path.name.endswith(".bak") for path in plugins_dir.iterdir())


def test_validate_for_promotion_rejects_malformed_test_case_without_running(tmp_path):
    sandbox, drafts, _plugin_registry, _plugins_dir = _runtime(tmp_path)
    draft = _draft()
    drafts.add(draft)

    check = validate_for_promotion(
        draft,
        sandbox=sandbox,
        test_cases=[{"expect": {"margin": 7}}],
    )

    assert check.passed is False
    assert check.problems == ("test case 1 inputs must be an object",)
    assert check.test_result is None


def test_promote_draft_rejects_failed_check_and_reject_marks_status(tmp_path):
    sandbox, drafts, plugin_registry, plugins_dir = _runtime(tmp_path)
    draft = _draft()
    drafts.add(draft)
    failed = validate_for_promotion(
        draft,
        sandbox=sandbox,
        test_cases=[{"inputs": {"revenue": 10, "cost": 3}, "expect": {"margin": 99}}],
    )

    assert failed.passed is False
    with pytest.raises(PromotionError, match="cannot promote"):
        promote_draft(draft, registry=plugin_registry, drafts=drafts, plugins_dir=plugins_dir, check=failed)

    reject_draft(draft, drafts=drafts, reason="not useful")
    assert drafts.get(draft.id).status == "rejected"


def test_reject_draft_rolls_back_status_when_audit_write_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    _sandbox, drafts, _plugin_registry, _plugins_dir = _runtime(tmp_path)
    draft = _draft()
    drafts.add(draft)

    def fail_audit(*args, **kwargs):
        raise RuntimeError("audit down")

    monkeypatch.setattr(draft_repo_module, "_write_audit_row", fail_audit)

    with pytest.raises(RuntimeError, match="audit down"):
        reject_draft(draft, drafts=drafts, reason="not useful")

    assert drafts.get(draft.id).status == "draft"


def test_draft_repository_rolls_back_promoted_status_when_audit_write_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = DraftRepository(db_path)
    draft = _draft(status="tested")
    repo.save_draft(draft)

    def fail_audit(*args, **kwargs):
        raise RuntimeError("audit down")

    monkeypatch.setattr(draft_repo_module, "_write_audit_row", fail_audit)

    with pytest.raises(RuntimeError, match="audit down"):
        repo.set_status_with_audit(
            draft.id,
            "promoted",
            audit={
                "kind": "draft.promote",
                "target_ref": draft.id,
                "outcome": "succeeded",
                "detail": {"plugin": "draft_calc_margin"},
            },
        )

    assert repo.get_draft(draft.id).status == "tested"
