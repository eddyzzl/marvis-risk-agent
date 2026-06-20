from __future__ import annotations

from pathlib import Path
import sys

from marvis.db import DraftRepository, PluginRepository
from marvis.drafts.authoring import draft_script
from marvis.drafts.errors import DraftNotFound, OfflineError
from marvis.drafts.learning import distill_learning
from marvis.drafts.registry import DraftRegistry
from marvis.drafts.sandbox import DraftSandbox
from marvis.drafts.web_search import fetch_url, web_search
from marvis.llm_client import OpenAICompatibleLLMClient
from marvis.llm_settings import resolve_llm_model
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner
from marvis.settings import build_settings


def tool_web_search(inputs: dict, ctx) -> dict:
    try:
        results = web_search(
            str(inputs["query"]),
            max_results=int(inputs.get("max_results", 5)),
        )
    except OfflineError as exc:
        return {"results": [], "offline": True, "guidance": str(exc)}
    return {"results": results, "offline": False, "guidance": ""}


def tool_fetch_url(inputs: dict, ctx) -> dict:
    url = str(inputs["url"])
    try:
        content = fetch_url(
            url,
            max_bytes=int(inputs.get("max_bytes", 500_000)),
        )
    except OfflineError as exc:
        return {"url": url, "content": "", "offline": True, "guidance": str(exc)}
    return {"url": url, "content": content, "offline": False, "guidance": ""}


def tool_distill_learning(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    note = distill_learning(
        str(inputs["query"]),
        _string_list(inputs.get("contents") or []),
        _string_list(inputs.get("sources") or []),
        llm_factory=_llm_factory(runtime.workspace, _optional_str(inputs.get("model_id"))),
    )
    runtime.draft_repo.save_learning_note(note)
    return {
        "learning_note_id": note.id,
        "query": note.query,
        "source_count": len(note.sources),
    }


def tool_draft_script(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    learning_note = None
    learning_note_id = inputs.get("learning_note_id")
    if learning_note_id:
        learning_note = runtime.draft_repo.get_learning_note(str(learning_note_id))
        if learning_note is None:
            raise DraftNotFound(f"learning note not found: {learning_note_id}")
    draft = draft_script(
        str(ctx.task_id),
        str(inputs["goal"]),
        learning_note=learning_note,
        llm_factory=_llm_factory(runtime.workspace, _optional_str(inputs.get("model_id"))),
    )
    runtime.drafts.add(draft)
    return {
        "draft_id": draft.id,
        "name": draft.name,
        "has_schema": bool(draft.input_schema and draft.output_schema),
    }


def tool_run_draft(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    draft = runtime.drafts.get(str(inputs["draft_id"]))
    run = runtime.sandbox.run_draft(
        draft.id,
        dict(inputs.get("inputs") or {}),
        task_id=str(ctx.task_id),
    )
    return {
        "run_id": run.id,
        "ok": run.ok,
        "output": run.output,
        "error": run.error,
    }


class _Runtime:
    def __init__(self, ctx):
        settings = build_settings(ctx.workspace)
        self.workspace = settings.workspace
        self.draft_repo = DraftRepository(settings.db_path)
        self.drafts = DraftRegistry(self.draft_repo)
        self.plugin_repo = PluginRepository(settings.db_path)
        self.plugin_registry = PluginRegistry(self.plugin_repo)
        self.plugin_registry.load_from_db()
        self.tool_runner = ToolRunner(
            ToolRegistry(self.plugin_registry),
            self.plugin_repo,
            python_executable=sys.executable,
            datasets_root=Path(ctx.datasets_root),
            workspace=settings.workspace,
            plugin_paths=[settings.plugins_dir],
        )
        self.sandbox = DraftSandbox(self.tool_runner, self.drafts, self.draft_repo)


def _runtime(ctx) -> _Runtime:
    return _Runtime(ctx)


def _llm_factory(workspace: Path, model_id: str | None):
    def factory():
        return OpenAICompatibleLLMClient(resolve_llm_model(workspace, model_id))

    return factory


def _optional_str(value) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _string_list(value) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, list):
        return [str(value)]
    return [str(item) for item in value]


__all__ = [
    "tool_distill_learning",
    "tool_draft_script",
    "tool_fetch_url",
    "tool_run_draft",
    "tool_web_search",
]
