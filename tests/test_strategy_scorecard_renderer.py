"""Reader-facing boundaries for full Scorecard bands and cutoff pointers."""

from __future__ import annotations

from marvis.agent.renderers import render_tool_output


ASSET_ID = "scorecard-band-asset-" + "a" * 32
CUTOFF_ID = "scorecard-cutoff-" + "b" * 32
SELECTION_ID = "scorecard-cutoff-selection-" + "c" * 32


def _artifact(filename: str, artifact_id: str) -> dict:
    content_hash = "d" * 64
    return {
        "artifact_id": artifact_id,
        "kind": "strategy_scorecard_json",
        "format": "json",
        "filename": filename,
        "content_hash": content_hash,
        "download_url": (
            f"/api/tasks/task-1/task-artifacts/{artifact_id}/download"
            f"?expected_content_hash={content_hash}"
        ),
    }


def test_scorecard_band_renderer_surfaces_full_evidence_without_recommendation() -> (
    None
):
    output = {
        "asset_id": ASSET_ID,
        "asset_hash": "e" * 64,
        "dataset_id": "dataset-1",
        "population_count": 1000,
        "development_count": 600,
        "labeled_count": 590,
        "bad_count": 59,
        "banding": {
            "method": "equal_frequency",
            "requested_bin_count": 10,
            "effective_bin_count": 2,
        },
        "band_count": 2,
        "cutoff_count": 1,
        "performance": {"auc": 0.73, "ks": 0.31},
        "scorecard_band_asset": {
            "bands": [
                {
                    "band_id": "scorecard-band-low",
                    "lower_bound": 0.0,
                    "upper_bound": 0.25,
                    "count": 300,
                    "labeled_count": 295,
                    "bad_count": 9,
                    "bad_rate": 9 / 295,
                    "average_pd": 0.11,
                },
                {
                    "band_id": "scorecard-band-high",
                    "lower_bound": 0.25,
                    "upper_bound": 1.0,
                    "count": 300,
                    "labeled_count": 295,
                    "bad_count": 50,
                    "bad_rate": 50 / 295,
                    "average_pd": 0.48,
                },
            ],
            "cutoffs": [
                {
                    "cutoff_id": CUTOFF_ID,
                    "execution_pd": 0.25,
                    "display_points": 620.0,
                    "lower_risk": {
                        "count": 300,
                        "bad_rate": 9 / 295,
                    },
                    "higher_risk": {
                        "count": 300,
                        "bad_rate": 50 / 295,
                    },
                    "mask_equivalence": True,
                }
            ],
        },
        "artifacts": [_artifact("scorecard-band.json", "artifact-band")],
        "not_selected": True,
        "not_admitted": True,
        "not_applied": True,
        "not_adopted": True,
        "not_deployed": True,
    }

    text, tables = render_tool_output("build_scorecard_band_asset", output)

    assert "Scorecard 完整分数带" in text
    assert "不会自动选择、排名或推荐 cutoff" in text
    assert "未入池" in text
    assert "未应用" in text
    assert "未采纳" in text
    assert "未部署" in text
    assert text.count("scorecard-band.json") == 1
    assert [table["title"] for table in tables] == [
        "Scorecard 分数带",
        "Scorecard cutoff 全量证据",
    ]
    assert tables[0]["rows"][0][0] == "scorecard-band-low"
    assert tables[1]["rows"][0][0] == CUTOFF_ID


def test_scorecard_cutoff_renderer_keeps_selection_pointer_only() -> None:
    output = {
        "selection_id": SELECTION_ID,
        "selection_hash": "f" * 64,
        "source_asset_id": ASSET_ID,
        "source_asset_hash": "e" * 64,
        "cutoff_id": CUTOFF_ID,
        "selection_reason": "人工确认进入后续影响评审",
        "artifacts": [
            _artifact("scorecard-cutoff-selection.json", "artifact-selection")
        ],
        "not_admitted": True,
        "not_applied": True,
        "not_adopted": True,
        "not_deployed": True,
    }

    text, tables = render_tool_output(
        "materialize_scorecard_cutoff_selection",
        output,
    )

    assert "Scorecard cutoff 精确选择" in text
    assert "pointer-only" in text
    assert "不会自动排名或推荐" in text
    assert "未入池" in text
    assert "未应用" in text
    assert "未采纳" in text
    assert "未部署" in text
    assert text.count("scorecard-cutoff-selection.json") == 1
    [identity] = tables
    assert identity["title"] == "Scorecard cutoff 选择引用"
    assert ["Selection ID", SELECTION_ID] in identity["rows"]
    assert ["Source Asset ID", ASSET_ID] in identity["rows"]
    assert ["Cutoff ID", CUTOFF_ID] in identity["rows"]
    assert [
        "Selection Reason",
        "人工确认进入后续影响评审",
    ] in identity["rows"]
