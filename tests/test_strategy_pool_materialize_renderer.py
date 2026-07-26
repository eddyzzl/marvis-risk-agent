"""Renderer truth contract for Pool materialization lifecycle/readiness."""

from __future__ import annotations

from marvis.agent.renderers import render_tool_output


def _output(*, blocked: bool = False, adopted: bool = False) -> dict:
    return {
        "schema_version": "strategy.pool-materialization-tool.v1",
        "materialization_id": "strategy-pool-materialization-" + "1" * 24,
        "strategy_ref": {
            "strategy_id": "strategy-pool-" + "2" * 24,
            "strategy_type": "approval",
            "version": 1,
            "strategy_spec_hash": "a" * 64,
            "strategy_dsl_content_hash": "b" * 64,
        },
        "pool_ref": {
            "pool_id": "strategy-pool-approval",
            "revision_id": "strategy-pool-revision-" + "3" * 32,
            "revision": 7,
            "snapshot_hash": "c" * 64,
            "artifact_id": "d" * 64,
            "artifact_content_hash": "e" * 64,
        },
        "design_hash": "f" * 64,
        "requirements": {
            "requirements_hash": "0" * 64,
            "requirement_count": 1 if blocked else 0,
            "virtual_fields": (
                ["__marvis_model_pd_0123456789abcdef"] if blocked else []
            ),
            "runtime_requirements_supported": not blocked,
            "blocker_code": (
                "strategy_pool_runtime_requirements_not_supported"
                if blocked
                else None
            ),
        },
        "lifecycle": {
            "created_status": "draft",
            "created_asset_status": "draft",
            "current_status": "adopted" if adopted else "draft",
            "current_asset_status": "adopted_local" if adopted else "draft",
            "adopted_by_this_tool": False,
            "deployed_by_this_tool": False,
        },
    }


def test_materialize_renderer_states_draft_lifecycle_and_human_adoption_gate() -> None:
    text, tables = render_tool_output(
        "materialize_strategy_from_pool",
        _output(),
    )

    assert "strategy-pool-" + "2" * 24 in text
    assert "Pool revision 7" in text
    assert "draft / draft" in text
    assert "本 Tool 未采纳、未部署" in text
    assert "人工采纳" in text
    assert "监控" in text
    assert tables
    assert "非 lifecycle readiness" in tables[-1]["title"]


def test_materialize_renderer_surfaces_requirement_blocker_without_private_fields() -> None:
    output = _output(blocked=True)

    text, _ = render_tool_output("materialize_strategy_from_pool", output)

    assert "strategy_pool_runtime_requirements_not_supported" in text
    assert "DSL 交付、回测和监控" in text
    assert "blocked" in text
    assert "__marvis_model_pd_0123456789abcdef" not in text


def test_materialize_renderer_never_attributes_preexisting_adoption_to_this_tool() -> None:
    text, _ = render_tool_output(
        "materialize_strategy_from_pool",
        _output(adopted=True),
    )

    assert "adopted / adopted_local" in text
    assert "本 Tool 未采纳、未部署" in text
    assert "当前已进入 adopted_local" in text
