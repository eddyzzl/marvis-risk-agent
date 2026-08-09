"""Build live pack runtimes for authenticated Tool-output presentation.

The caller must supply the workspace and task identity from internal plan state.
Cached Tool output is never allowed to choose either value.  Pack imports remain
lazy so ordinary presentation paths do not pay the Modeling/Strategy import cost.
"""

from __future__ import annotations

from pathlib import Path

from marvis.plugins.contracts import ToolContext


MODELING_RUNTIME_PRESENTER_TOOLS = frozenset(
    {
        "train_model_with_evidence_v2",
        "materialize_model_score_evidence_v2",
    }
)
STRATEGY_RUNTIME_PRESENTER_TOOLS = frozenset(
    {
        "search_cross_matrix_candidates",
        "build_cross_matrix_candidate_from_search",
        "search_interactive_tree_split_candidates",
        "auto_continue_interactive_tree",
        "revise_interactive_tree",
        "measure_strategy_impact_cube",
    }
)
RUNTIME_PRESENTER_TOOLS = (
    MODELING_RUNTIME_PRESENTER_TOOLS | STRATEGY_RUNTIME_PRESENTER_TOOLS
)


def build_trusted_presenter_runtime(
    tool: str,
    *,
    workspace: Path | str | None,
    task_id: str | None,
):
    """Return a pack runtime only for a known governed presenter and trust root."""

    if tool not in RUNTIME_PRESENTER_TOOLS:
        return None
    if not isinstance(task_id, str) or not task_id.strip() or "\x00" in task_id:
        return None
    if workspace is None:
        return None
    try:
        trusted_workspace = Path(workspace).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    if not trusted_workspace.is_dir():
        return None
    context = ToolContext(
        task_id=task_id.strip(),
        seed=None,
        datasets_root=trusted_workspace / "datasets",
        workspace=trusted_workspace,
    )
    if tool in MODELING_RUNTIME_PRESENTER_TOOLS:
        from marvis.packs.modeling._runtime import _runtime

        return _runtime(context)
    from marvis.packs.strategy.tools import _runtime

    return _runtime(context)


__all__ = [
    "MODELING_RUNTIME_PRESENTER_TOOLS",
    "RUNTIME_PRESENTER_TOOLS",
    "STRATEGY_RUNTIME_PRESENTER_TOOLS",
    "build_trusted_presenter_runtime",
]
