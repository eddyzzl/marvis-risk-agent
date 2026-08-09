"""Commit-time authentication for governed canonical Tool results.

The eight tools in :data:`CANONICAL_RESULT_TOOLS` return envelopes that can
name live, task-scoped artifacts.  JSON-schema validation alone cannot prove
that an envelope belongs to the inputs of the current invocation.  This
module is the shared trust choke point used by ``ToolRunner`` before it signs
an invocation receipt; presentation repeats the same live checks as defence in
depth.

The domain checks currently live beside their canonical presenters because
they rebuild the exact display payload from the same authenticated objects.
Imports stay lazy so ordinary plugin discovery does not pull Agent or Strategy
runtime dependencies into process startup.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


CANONICAL_RESULT_TOOLS = frozenset(
    {
        "train_model_with_evidence_v2",
        "materialize_model_score_evidence_v2",
        "search_cross_matrix_candidates",
        "build_cross_matrix_candidate_from_search",
        "search_interactive_tree_split_candidates",
        "auto_continue_interactive_tree",
        "revise_interactive_tree",
        "measure_strategy_impact_cube",
    }
)


class CanonicalResultAuthenticationError(ValueError):
    """A governed Tool result did not authenticate against its invocation."""


def authenticate_canonical_result(
    tool: str,
    output: object,
    *,
    trusted_inputs: Mapping[str, Any],
    task_id: str,
    workspace: Path | str,
) -> bool:
    """Authenticate one governed result against live, task-scoped evidence.

    Returns ``False`` for tools outside the canonical set and ``True`` only
    after a canonical tool's complete domain validator has succeeded.  Any
    canonical mismatch is normalized to a single fail-closed exception so a
    caller cannot accidentally downgrade it to generic success rendering.
    """

    if tool not in CANONICAL_RESULT_TOOLS:
        return False
    if not isinstance(trusted_inputs, Mapping):
        raise CanonicalResultAuthenticationError(
            "canonical Tool result is missing trusted invocation inputs"
        )
    try:
        from marvis.agent.presenters.modeling_evidence import (
            MODELING_EVIDENCE_PRESENTERS,
        )
        from marvis.agent.presenters.runtime import (
            build_trusted_presenter_runtime,
        )
        from marvis.agent.presenters.strategy_exploration import (
            STRATEGY_EXPLORATION_PRESENTERS,
        )

        presenter = {
            **MODELING_EVIDENCE_PRESENTERS,
            **STRATEGY_EXPLORATION_PRESENTERS,
        }[tool]
        runtime = build_trusted_presenter_runtime(
            tool,
            workspace=workspace,
            task_id=task_id,
        )
        if runtime is None:
            raise ValueError("canonical Tool runtime is unavailable")
        presenter(
            output,
            runtime=runtime,
            task_id=task_id,
            trusted_inputs=trusted_inputs,
            trusted_artifacts=_trusted_artifacts(
                tool,
                output,
                runtime=runtime,
                task_id=task_id,
            ),
        )
    except CanonicalResultAuthenticationError:
        raise
    except Exception as exc:
        raise CanonicalResultAuthenticationError(
            "canonical Tool result did not match its live invocation evidence"
        ) from exc
    return True


def _trusted_artifacts(
    tool: str,
    output: object,
    *,
    runtime: object,
    task_id: str,
) -> dict[str, dict] | None:
    if tool != "measure_strategy_impact_cube":
        return None
    if not isinstance(output, Mapping):
        raise ValueError("ImpactCube result is not an object")
    artifact = output.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError("ImpactCube result is missing its artifact binding")
    artifact_id = artifact.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id:
        raise ValueError("ImpactCube artifact id is invalid")
    record = runtime.task_artifacts.get_for_task(task_id, artifact_id)
    if not isinstance(record, Mapping):
        raise ValueError("ImpactCube artifact registry record is unavailable")
    return {"impact_cube": {"record": dict(record)}}


__all__ = [
    "CANONICAL_RESULT_TOOLS",
    "CanonicalResultAuthenticationError",
    "authenticate_canonical_result",
]
