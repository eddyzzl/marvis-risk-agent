import sys
from pathlib import Path
from types import SimpleNamespace

from marvis.db import DraftRepository, PluginRepository, init_db
from marvis.drafts import DraftTool
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner
from marvis.settings import build_settings


def _ctx(tmp_path, *, task_id="task-1"):
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    return SimpleNamespace(
        task_id=task_id,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )


def _runner(tmp_path):
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    repo = PluginRepository(settings.db_path)
    registry = PluginRegistry(repo)
    load_builtin_packs(registry, _packs_root())
    return ToolRunner(
        ToolRegistry(registry),
        repo,
        python_executable=sys.executable,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
        plugin_paths=[settings.plugins_dir],
    )


def _packs_root() -> Path:
    return Path(__file__).parents[1] / "marvis" / "packs"


def _draft() -> DraftTool:
    return DraftTool(
        id="draft-1",
        task_id="task-1",
        name="calc_margin",
        summary="Calculate margin.",
        code="def calc_margin(inputs, ctx):\n    return {'margin': inputs['revenue'] - inputs['cost']}\n",
        input_schema={
            "type": "object",
            "properties": {"revenue": {"type": "number"}, "cost": {"type": "number"}},
            "required": ["revenue", "cost"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"margin": {"type": "number"}},
            "required": ["margin"],
            "additionalProperties": False,
        },
        determinism="deterministic",
        source="hand_written",
        learning_note_id=None,
        status="draft",
        created_at="2026-06-19T00:00:00Z",
    )


class _FakeLLM:
    def complete(self, **_kwargs):
        return (
            '{"name":"calc_margin","summary":"Calculate margin.",'
            '"code":"def calc_margin(inputs, ctx):\\n    return {'
            "'margin': inputs['revenue'] - inputs['cost']}\\n\","
            '"input_schema":{"type":"object","properties":{"revenue":{"type":"number"},'
            '"cost":{"type":"number"}},"required":["revenue","cost"],'
            '"additionalProperties":false},'
            '"output_schema":{"type":"object","properties":{"margin":{"type":"number"}},'
            '"required":["margin"],"additionalProperties":false},'
            '"determinism":"deterministic"}'
        )


def test_draft_web_search_tool_returns_offline_payload(monkeypatch, tmp_path):
    from marvis.drafts import tools
    from marvis.drafts.errors import OfflineError

    monkeypatch.setattr(tools, "web_search", lambda *_args, **_kwargs: (_ for _ in ()).throw(OfflineError("offline guidance")))

    result = tools.tool_web_search({"query": "scorecard validation"}, _ctx(tmp_path))

    assert result == {"results": [], "offline": True, "guidance": "offline guidance"}


def test_draft_script_tool_persists_generated_draft(monkeypatch, tmp_path):
    from marvis.drafts import tools

    class FakeClient:
        def __init__(self, _profile):
            pass

        def complete(self, **kwargs):
            return _FakeLLM().complete(**kwargs)

    monkeypatch.setattr(tools, "resolve_llm_model", lambda _workspace, _model_id=None: {"model_id": "m1"})
    monkeypatch.setattr(tools, "OpenAICompatibleLLMClient", FakeClient)
    ctx = _ctx(tmp_path)

    result = tools.tool_draft_script({"goal": "build margin calculator"}, ctx)

    repo = DraftRepository(build_settings(tmp_path).db_path)
    draft = repo.get_draft(result["draft_id"])
    assert result["name"] == "calc_margin"
    assert result["has_schema"] is True
    assert draft is not None
    assert draft.status == "draft"
    assert draft.task_id == "task-1"


def test_draft_run_tool_runs_saved_draft_in_subprocess(tmp_path):
    from marvis.drafts import tools

    ctx = _ctx(tmp_path)
    DraftRepository(build_settings(tmp_path).db_path).save_draft(_draft())

    result = tools.tool_run_draft(
        {"draft_id": "draft-1", "inputs": {"revenue": 10, "cost": 3}},
        ctx,
    )

    assert result["ok"] is True
    assert result["output"] == {"margin": 7}
    assert result["error"] is None


def test_builtin_drafts_pack_exposes_offline_search_through_runner(monkeypatch, tmp_path):
    monkeypatch.setenv("MARVIS_PROBE_URL", "http://127.0.0.1:9")
    runner = _runner(tmp_path)

    result = runner.invoke(
        ToolRef("drafts", "web_search"),
        {"query": "scorecard validation"},
        task_id="task-1",
    )

    assert result.ok is True, result.error
    assert result.output["offline"] is True
    assert result.output["results"] == []


def test_builtin_drafts_pack_runs_draft_through_runner(tmp_path):
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    DraftRepository(settings.db_path).save_draft(_draft())
    repo = PluginRepository(settings.db_path)
    registry = PluginRegistry(repo)
    load_builtin_packs(registry, _packs_root())
    runner = ToolRunner(
        ToolRegistry(registry),
        repo,
        python_executable=sys.executable,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
        plugin_paths=[settings.plugins_dir],
    )

    result = runner.invoke(
        ToolRef("drafts", "run_draft"),
        {"draft_id": "draft-1", "inputs": {"revenue": 10, "cost": 3}},
        task_id="task-1",
    )

    assert result.ok is True, result.error
    assert result.output["ok"] is True
    assert result.output["output"] == {"margin": 7}
