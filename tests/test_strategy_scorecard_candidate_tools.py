from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from marvis.packs.strategy import tools as strategy_tools
import marvis.packs.strategy.scorecard_candidate_tools as scorecard_tools
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.scorecard_candidate_tools import (
    SCORECARD_BAND_ASSET_ARTIFACT_KIND,
    SCORECARD_CUTOFF_SELECTION_ARTIFACT_KIND,
    load_scorecard_band_asset_artifact,
    load_scorecard_cutoff_selection_artifact,
    run_build_scorecard_band_asset,
    run_materialize_scorecard_cutoff_selection,
)
from marvis.repositories.task_artifacts import TaskArtifactRepository
from tests.test_model_score_evidence_tool import _run_score
from tests.test_modeling_training_evidence_tool import (
    _fixture,
    _run as run_training,
)


def _real_scorecard(tmp_path: Path) -> dict:
    fx = _fixture(tmp_path)
    fx["inputs"]["recipe"] = "scorecard"
    fx["inputs"]["params"].update(
        {
            "max_iter": 200,
            "scorecard_max_bins": 3,
        }
    )
    training_output = run_training(fx)
    score_output = _run_score(fx, training_output)
    return {
        "fx": fx,
        "runtime": strategy_tools._runtime(fx["ctx"]),
        "training_output": training_output,
        "score_output": score_output,
    }


def _build_inputs(real: dict) -> dict:
    score = real["score_output"]["artifacts"]
    return {
        "score_evidence_ref": {
            "evidence_artifact_id": score["score_evidence"]["artifact_id"],
            "expected_evidence_artifact_content_hash": score["score_evidence"][
                "content_hash"
            ],
            "score_vector_artifact_id": score["score_vector"]["artifact_id"],
            "expected_score_vector_artifact_content_hash": score["score_vector"][
                "content_hash"
            ],
        },
        "sample_design_ref": dict(real["fx"]["sample_ref"]),
    }


def _records(real: dict, kind: str) -> list[dict]:
    return [
        record
        for record in TaskArtifactRepository(
            real["fx"]["settings"].db_path
        ).list_for_task(real["fx"]["task"].id)
        if record["kind"] == kind
    ]


def test_public_build_rejects_non_increasing_manual_raw_pd_edges() -> None:
    digest = "a" * 64
    with pytest.raises(StrategyError, match="strictly increasing"):
        run_build_scorecard_band_asset(
            {
                "score_evidence_ref": {
                    "evidence_artifact_id": digest,
                    "expected_evidence_artifact_content_hash": digest,
                    "score_vector_artifact_id": digest,
                    "expected_score_vector_artifact_content_hash": digest,
                },
                "sample_design_ref": {
                    "membership_artifact_id": digest,
                    "expected_membership_artifact_content_hash": digest,
                    "bundle_artifact_id": digest,
                    "expected_bundle_artifact_content_hash": digest,
                    "expected_bundle_id": "sample-bundle",
                    "expected_sample_design_id": "sample-design",
                    "expected_sample_design_content_hash": digest,
                },
                "raw_pd_band_edges": [0.0, 0.5, 0.5, 1.0],
            },
            SimpleNamespace(task_id="task-scorecard"),
            SimpleNamespace(),
        )


@pytest.mark.slow
def test_real_scorecard_build_load_and_pointer_selection_are_governed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = _real_scorecard(tmp_path)
    runtime = real["runtime"]
    task_id = real["fx"]["task"].id
    build_inputs = _build_inputs(real)
    dataset_path = (
        real["fx"]["settings"].datasets_dir
        / real["fx"]["dataset"].source_path
    )
    dataset_bytes = dataset_path.read_bytes()
    forged_frame = pd.read_parquet(dataset_path)
    forged_frame["bad"] = 1 - forged_frame["bad"]
    forged_path = tmp_path / "forged-scorecard-labels.parquet"
    forged_frame.to_parquet(forged_path, index=False)
    forged_bytes = forged_path.read_bytes()
    original_read_parquet = scorecard_tools.pd.read_parquet

    def restore_live_path_during_read(source, *args, **kwargs):
        if kwargs.get("columns") != ["bad"]:
            return original_read_parquet(source, *args, **kwargs)
        dataset_path.write_bytes(forged_bytes)
        try:
            return original_read_parquet(source, *args, **kwargs)
        finally:
            dataset_path.write_bytes(dataset_bytes)

    monkeypatch.setattr(
        scorecard_tools.pd,
        "read_parquet",
        restore_live_path_during_read,
    )
    with pytest.raises(StrategyError, match="changed|snapshot|bytes"):
        run_build_scorecard_band_asset(
            build_inputs,
            real["fx"]["ctx"],
            runtime,
        )
    assert _records(real, SCORECARD_BAND_ASSET_ARTIFACT_KIND) == []
    monkeypatch.setattr(
        scorecard_tools.pd,
        "read_parquet",
        original_read_parquet,
    )

    score_record = runtime.task_artifacts.get_for_task(
        task_id,
        build_inputs["score_evidence_ref"]["evidence_artifact_id"],
    )
    assert score_record is not None
    score_path = Path(score_record["path"])
    score_bytes = score_path.read_bytes()
    original_transaction = runtime.task_artifacts.transaction

    @contextmanager
    def drift_score_evidence():
        with original_transaction() as conn:
            score_path.write_bytes(score_bytes + b" ")
            yield conn

    monkeypatch.setattr(
        runtime.task_artifacts,
        "transaction",
        drift_score_evidence,
    )
    with pytest.raises(StrategyError, match="score|evidence|hash|changed"):
        run_build_scorecard_band_asset(
            build_inputs,
            real["fx"]["ctx"],
            runtime,
        )
    assert _records(real, SCORECARD_BAND_ASSET_ARTIFACT_KIND) == []
    score_path.write_bytes(score_bytes)
    monkeypatch.setattr(
        runtime.task_artifacts,
        "transaction",
        original_transaction,
    )

    output = run_build_scorecard_band_asset(
        build_inputs,
        real["fx"]["ctx"],
        runtime,
    )

    assert output["not_admitted"] is True
    assert output["not_applied"] is True
    assert output["not_adopted"] is True
    assert output["not_deployed"] is True
    assert output["scorecard_band_asset"]["governance"][
        "best_cutoff_recommended"
    ] is False
    assert output["banding"]["method"] == "equal_frequency"
    assert output["banding"]["requested_bin_count"] == 10
    assert 2 <= output["banding"]["effective_bin_count"] <= 10
    assert len(output["scorecard_band_asset"]["bands"]) == output["banding"][
        "effective_bin_count"
    ]
    assert len(output["scorecard_band_asset"]["cutoffs"]) == (
        output["banding"]["effective_bin_count"] - 1
    )
    assert all(
        cutoff["mask_equivalence"] is True
        for cutoff in output["scorecard_band_asset"]["cutoffs"]
    )
    artifact = output["artifacts"][0]
    band_record = runtime.task_artifacts.get_for_task(
        task_id,
        artifact["artifact_id"],
    )
    assert band_record is not None
    band_path = Path(band_record["path"])
    exact_band_bytes = band_path.read_bytes()
    with original_transaction() as conn:
        conn.execute(
            "DELETE FROM task_artifacts WHERE task_id = ? AND id = ?",
            (task_id, artifact["artifact_id"]),
        )
        conn.commit()
    assert _records(real, SCORECARD_BAND_ASSET_ARTIFACT_KIND) == []

    # A crash after the exact canonical file is promoted but before its
    # registry row is committed must be recoverable by an idempotent replay.
    assert (
        run_build_scorecard_band_asset(
            build_inputs,
            real["fx"]["ctx"],
            runtime,
        )
        == output
    )
    assert band_path.read_bytes() == exact_band_bytes
    assert len(_records(real, SCORECARD_BAND_ASSET_ARTIFACT_KIND)) == 1

    loaded = load_scorecard_band_asset_artifact(
        runtime,
        task_id=task_id,
        artifact_id=artifact["artifact_id"],
        expected_artifact_content_hash=artifact["content_hash"],
        expected_asset_id=output["asset_id"],
        expected_asset_hash=output["asset_hash"],
    )
    assert loaded.asset == output["scorecard_band_asset"]
    assert loaded.path.read_bytes().decode("utf-8").endswith("}")

    cutoff = output["scorecard_band_asset"]["cutoffs"][0]
    selection_inputs = {
        "source_artifact_id": artifact["artifact_id"],
        "expected_source_artifact_content_hash": artifact["content_hash"],
        "expected_asset_id": output["asset_id"],
        "expected_asset_hash": output["asset_hash"],
        "cutoff_id": cutoff["cutoff_id"],
        "reason": "业务确认先评估该通过线",
    }
    selected = run_materialize_scorecard_cutoff_selection(
        selection_inputs,
        real["fx"]["ctx"],
        runtime,
    )

    assert selected["not_admitted"] is True
    assert selected["not_applied"] is True
    assert selected["not_adopted"] is True
    assert selected["not_deployed"] is True
    assert "bands" not in selected["selection"]
    assert "scorecard_table" not in selected["selection"]
    selection_artifact = selected["artifacts"][0]
    selection_record = runtime.task_artifacts.get_for_task(
        task_id,
        selection_artifact["artifact_id"],
    )
    assert selection_record is not None
    selection_path = Path(selection_record["path"])
    exact_selection_bytes = selection_path.read_bytes()
    with original_transaction() as conn:
        conn.execute(
            "DELETE FROM task_artifacts WHERE task_id = ? AND id = ?",
            (task_id, selection_artifact["artifact_id"]),
        )
        conn.commit()
    selection_path.write_bytes(b"{}")

    # Recovery is exact-only: an unregistered file with different bytes must
    # remain fail-closed and must not acquire a registry row.
    with pytest.raises(StrategyError, match="hash|canonical|changed"):
        run_materialize_scorecard_cutoff_selection(
            selection_inputs,
            real["fx"]["ctx"],
            runtime,
        )
    assert _records(real, SCORECARD_CUTOFF_SELECTION_ARTIFACT_KIND) == []

    selection_path.write_bytes(exact_selection_bytes)
    assert (
        run_materialize_scorecard_cutoff_selection(
            selection_inputs,
            real["fx"]["ctx"],
            runtime,
        )
        == selected
    )
    assert selection_path.read_bytes() == exact_selection_bytes

    selection = load_scorecard_cutoff_selection_artifact(
        runtime,
        task_id=task_id,
        artifact_id=selection_artifact["artifact_id"],
        expected_artifact_content_hash=selection_artifact["content_hash"],
        expected_selection_id=selected["selection_id"],
        expected_selection_hash=selected["selection_hash"],
    )
    assert selection.selection == selected["selection"]
    assert selection.selection["cutoff_id"] == cutoff["cutoff_id"]
    assert (
        selection.selection["source_asset_ref"]["asset_id"]
        == output["asset_id"]
    )
    assert len(_records(real, SCORECARD_BAND_ASSET_ARTIFACT_KIND)) == 1
    assert len(_records(real, SCORECARD_CUTOFF_SELECTION_ARTIFACT_KIND)) == 1

    band_bytes = loaded.path.read_bytes()

    @contextmanager
    def drift_band_asset():
        with original_transaction() as conn:
            loaded.path.write_bytes(band_bytes + b" ")
            yield conn

    monkeypatch.setattr(
        runtime.task_artifacts,
        "transaction",
        drift_band_asset,
    )
    cutoff = output["scorecard_band_asset"]["cutoffs"][0]
    with pytest.raises(StrategyError, match="band|asset|hash|changed"):
        run_materialize_scorecard_cutoff_selection(
            selection_inputs,
            real["fx"]["ctx"],
            runtime,
        )
    assert len(_records(real, SCORECARD_CUTOFF_SELECTION_ARTIFACT_KIND)) == 1

    # A loader must authenticate live bytes, not trust the cached registry row.
    with pytest.raises(StrategyError, match="hash|canonical|changed"):
        load_scorecard_band_asset_artifact(
            runtime,
            task_id=task_id,
            artifact_id=artifact["artifact_id"],
            expected_artifact_content_hash=artifact["content_hash"],
            expected_asset_id=output["asset_id"],
            expected_asset_hash=output["asset_hash"],
        )
