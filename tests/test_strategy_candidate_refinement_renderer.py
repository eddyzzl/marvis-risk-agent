from __future__ import annotations

from marvis.agent.renderers import render_tool_output


def test_candidate_refinement_renderer_preserves_governance_and_downloads():
    output = {
        "asset_id": "candidate-asset-1234",
        "asset_hash": "a" * 64,
        "asset_type": "univariate_refinement",
        "effect_stage": "development",
        "validation_status": "unvalidated",
        "parent_candidate_id": "candidate-parent",
        "parent_evidence_hash": "b" * 64,
        "feature": "score",
        "method": "tree",
        "selection": {"source_bin_ids": ["regular:1", "regular:2"]},
        "rule": {
            "rule_id": "rule-1234",
            "selected_source_bin_ids": ["regular:1", "regular:2"],
        },
        "effect_id": "effect-1234",
        "effect": {
            "effect_id": "effect-1234",
            "selected_count": 42,
            "selected_share": 0.21,
            "bad_rate": 0.3,
        },
        "artifacts": [
            {
                "artifact_id": "artifact-1",
                "kind": "strategy_candidate_asset_json",
                "filename": "candidate.json",
                "content_hash": "c" * 64,
                "download_url": "/api/tasks/task-1/artifacts/artifact-1/download",
            }
        ],
    }

    text, tables = render_tool_output("refine_univariate_candidate", output)

    assert "候选选择与合并完成" in text
    assert "score" in text and "tree" in text
    assert "development / unvalidated" in text
    assert "candidate-parent" in text
    assert "rule-1234" in text and "effect-1234" in text
    assert "candidate.json" in text
    assert "/api/tasks/task-1/artifacts/artifact-1/download" in text
    assert tables == []
