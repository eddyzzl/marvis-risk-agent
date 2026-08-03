from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
import hashlib
from pathlib import Path

import pytest

from marvis.db import TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pool import ABSENT_POOL_SNAPSHOT_HASH
from marvis.packs.strategy.pool_requirement_resolver import (
    model_score_virtual_field,
)
from marvis.packs.strategy.pool_tools import (
    bind_strategy_pool_development_execution,
    load_current_strategy_candidate_pool_artifact,
    run_add_candidate_to_pool,
    run_compile_strategy_pool,
    run_remove_pool_entry,
    run_reorder_strategy_pool,
    run_set_pool_entry_action,
)
from marvis.packs.strategy.sample_design_v2_tools import (
    run_materialize_sample_design_v2,
)
from marvis.packs.strategy.scorecard_candidate import (
    SCORECARD_BAND_ASSET_ARTIFACT_KIND,
    SCORECARD_BAND_ASSET_ARTIFACT_SCHEMA_VERSION,
    SCORECARD_BAND_ASSET_ORIGIN_TOOL,
)
from marvis.packs.strategy.scorecard_candidate_tools import (
    load_scorecard_cutoff_selection_artifact,
    run_build_scorecard_band_asset,
    run_materialize_scorecard_cutoff_selection,
)
from marvis.plugins.contracts import ToolContext
from marvis.repositories.strategy_pool import StrategyCandidatePoolRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.settings import build_settings
from tests.test_model_score_evidence_tool import _run_score
from tests.test_modeling_training_evidence_tool import (
    _fixture,
    _native_fixture,
    _run as run_training,
)


def _action(action_type: str, *, reason: str | None = None) -> dict:
    values = {"approval": "approve", "reject": "reject", "review": "review"}
    return {
        "type": action_type,
        "value": values[action_type],
        "reason_code": reason,
        "stop": True,
    }


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
    runtime = strategy_tools._runtime(fx["ctx"])
    score_artifacts = score_output["artifacts"]
    band = run_build_scorecard_band_asset(
        {
            "score_evidence_ref": {
                "evidence_artifact_id": score_artifacts["score_evidence"][
                    "artifact_id"
                ],
                "expected_evidence_artifact_content_hash": score_artifacts[
                    "score_evidence"
                ]["content_hash"],
                "score_vector_artifact_id": score_artifacts["score_vector"][
                    "artifact_id"
                ],
                "expected_score_vector_artifact_content_hash": score_artifacts[
                    "score_vector"
                ]["content_hash"],
            },
            "sample_design_ref": dict(fx["sample_ref"]),
            "banding": {"method": "equal_frequency", "bin_count": 3},
        },
        fx["ctx"],
        runtime,
    )
    return {
        "fx": fx,
        "runtime": runtime,
        "band": band,
    }


def _real_native_scorecard(tmp_path: Path) -> dict:
    fx = _native_fixture(tmp_path)
    training_output = run_training(fx)
    score_output = _run_score(fx, training_output)
    runtime = strategy_tools._runtime(fx["ctx"])
    score_artifacts = score_output["artifacts"]
    band = run_build_scorecard_band_asset(
        {
            "score_evidence_ref": {
                "evidence_artifact_id": score_artifacts["score_evidence"][
                    "artifact_id"
                ],
                "expected_evidence_artifact_content_hash": score_artifacts[
                    "score_evidence"
                ]["content_hash"],
                "score_vector_artifact_id": score_artifacts["score_vector"][
                    "artifact_id"
                ],
                "expected_score_vector_artifact_content_hash": score_artifacts[
                    "score_vector"
                ]["content_hash"],
            },
            "sample_design_ref": dict(fx["sample_ref"]),
            "banding": {"method": "equal_frequency", "bin_count": 3},
        },
        fx["ctx"],
        runtime,
    )
    return {
        "fx": fx,
        "runtime": runtime,
        "band": band,
    }


def _selection(real: dict, ordinal: int = 0) -> dict:
    band = real["band"]
    artifact = band["artifacts"][0]
    cutoff = band["scorecard_band_asset"]["cutoffs"][ordinal]
    return run_materialize_scorecard_cutoff_selection(
        {
            "source_artifact_id": artifact["artifact_id"],
            "expected_source_artifact_content_hash": artifact["content_hash"],
            "expected_asset_id": band["asset_id"],
            "expected_asset_hash": band["asset_hash"],
            "cutoff_id": cutoff["cutoff_id"],
            "reason": "策略人员明确选择该评分卡切点",
        },
        real["fx"]["ctx"],
        real["runtime"],
    )


def _alternate_sample_scorecard(real: dict) -> dict:
    sample = real["runtime"].task_artifacts.get_for_task(
        real["fx"]["task"].id,
        real["fx"]["sample_ref"]["bundle_artifact_id"],
    )
    assert sample is not None
    request = deepcopy(sample["provenance"]["request"])
    request["policy"]["maximum_group_coverage_gap"] = 0.95
    output = run_materialize_sample_design_v2(
        request,
        real["fx"]["ctx"],
        real["runtime"],
    )
    repository = TaskArtifactRepository(real["fx"]["settings"].db_path)
    records = repository.list_for_task(real["fx"]["task"].id)
    membership_record = next(
        record
        for record in records
        if record["kind"] == output["artifacts"]["membership"]["kind"]
        and Path(record["path"]).name
        == output["artifacts"]["membership"]["filename"]
    )
    bundle_record = next(
        record
        for record in records
        if record["kind"] == output["artifacts"]["bundle"]["kind"]
        and Path(record["path"]).name == output["artifacts"]["bundle"]["filename"]
    )
    sample_ref = {
        "membership_artifact_id": membership_record["id"],
        "expected_membership_artifact_content_hash": membership_record[
            "content_hash"
        ],
        "bundle_artifact_id": bundle_record["id"],
        "expected_bundle_artifact_content_hash": bundle_record["content_hash"],
        "expected_bundle_id": output["bundle_id"],
        "expected_sample_design_id": output["sample_design_id"],
        "expected_sample_design_content_hash": output[
            "sample_design_content_hash"
        ],
    }
    real["fx"]["inputs"]["sample_design_ref"] = sample_ref
    real["fx"]["sample_ref"] = sample_ref
    real["fx"]["inputs"]["seed"] += 1
    training_output = run_training(real["fx"])
    score_output = _run_score(real["fx"], training_output)
    score_artifacts = score_output["artifacts"]
    alternate_band = run_build_scorecard_band_asset(
        {
            "score_evidence_ref": {
                "evidence_artifact_id": score_artifacts["score_evidence"][
                    "artifact_id"
                ],
                "expected_evidence_artifact_content_hash": score_artifacts[
                    "score_evidence"
                ]["content_hash"],
                "score_vector_artifact_id": score_artifacts["score_vector"][
                    "artifact_id"
                ],
                "expected_score_vector_artifact_content_hash": score_artifacts[
                    "score_vector"
                ]["content_hash"],
            },
            "sample_design_ref": sample_ref,
            "banding": {"method": "equal_frequency", "bin_count": 3},
        },
        real["fx"]["ctx"],
        real["runtime"],
    )
    alternate = {
        **real,
        "band": alternate_band,
    }
    return _selection(alternate)


def _add_inputs(
    candidate: dict,
    *,
    expected_revision: int,
    expected_snapshot_hash: str,
) -> dict:
    artifact = candidate["artifacts"][0]
    return {
        "source_artifact_id": artifact["artifact_id"],
        "expected_artifact_content_hash": artifact["content_hash"],
        "expected_asset_id": candidate["source_asset_id"],
        "expected_asset_hash": candidate["source_asset_hash"],
        "strategy_type": "approval",
        "default_action": _action("approval"),
        "action": _action("reject", reason="SCORECARD_CUTOFF"),
        "expected_pool_revision": expected_revision,
        "expected_pool_snapshot_hash": expected_snapshot_hash,
    }


def test_complete_scorecard_band_artifact_requires_pointer_selection(
    tmp_path: Path,
) -> None:
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="scorecard-pool-dispatch",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "source"),
            task_type="strategy",
            target_col="bad",
        )
    )
    ctx = ToolContext(
        task_id=task.id,
        seed=0,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )
    runtime = strategy_tools._runtime(ctx)
    path = (
        settings.tasks_dir
        / task.id
        / "strategy_scorecard_candidates"
        / "complete-band.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = b"{}"
    path.write_bytes(raw)
    record = runtime.task_artifacts.register(
        task_id=task.id,
        kind=SCORECARD_BAND_ASSET_ARTIFACT_KIND,
        path=str(path),
        content_hash=hashlib.sha256(raw).hexdigest(),
        origin_tool=SCORECARD_BAND_ASSET_ORIGIN_TOOL,
        provenance={
            "schema_version": SCORECARD_BAND_ASSET_ARTIFACT_SCHEMA_VERSION
        },
    )

    with pytest.raises(
        StrategyError,
        match="complete scorecard band assets cannot be admitted directly",
    ):
        run_add_candidate_to_pool(
            {
                "source_artifact_id": record["id"],
                "expected_artifact_content_hash": record["content_hash"],
                "expected_asset_id": "scorecard-band-asset-placeholder",
                "expected_asset_hash": "a" * 64,
                "strategy_type": "approval",
                "default_action": _action("approval"),
                "action": _action("reject", reason="SCORECARD_CUTOFF"),
                "expected_pool_revision": 0,
                "expected_pool_snapshot_hash": ABSENT_POOL_SNAPSHOT_HASH,
            },
            ctx,
            runtime,
        )


@pytest.mark.slow
def test_native_scorecard_pool_binds_risk_development_execution(
    tmp_path: Path,
) -> None:
    real = _real_native_scorecard(tmp_path)
    selection = _selection(real)
    added = run_add_candidate_to_pool(
        _add_inputs(
            selection,
            expected_revision=0,
            expected_snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        real["fx"]["ctx"],
        real["runtime"],
    )
    current = load_current_strategy_candidate_pool_artifact(
        real["runtime"],
        task_id=real["fx"]["task"].id,
        strategy_type="approval",
        expected_pool_revision=added["revision"],
        expected_pool_snapshot_hash=added["snapshot_hash"],
    )
    development = bind_strategy_pool_development_execution(
        real["runtime"],
        current,
    )

    assert development.sample_design.source_mode == "native_active_dataset"
    assert development.sample_design.reference.partition == "risk/development"
    assert development.sample_design_v2 is None
    assert development.sample_design.target_col == "bad"
    assert development.sample_design._native is not None


@pytest.mark.slow
def test_only_pointer_selection_enters_pool_and_compiles_exact_score_requirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = _real_scorecard(tmp_path)
    band = real["band"]
    band_artifact = band["artifacts"][0]

    with pytest.raises(
        StrategyError,
        match="complete scorecard band assets cannot be admitted directly",
    ):
        run_add_candidate_to_pool(
            {
                "source_artifact_id": band_artifact["artifact_id"],
                "expected_artifact_content_hash": band_artifact["content_hash"],
                "expected_asset_id": band["asset_id"],
                "expected_asset_hash": band["asset_hash"],
                "strategy_type": "approval",
                "default_action": _action("approval"),
                "action": _action("reject", reason="SCORECARD_CUTOFF"),
                "expected_pool_revision": 0,
                "expected_pool_snapshot_hash": ABSENT_POOL_SNAPSHOT_HASH,
            },
            real["fx"]["ctx"],
            real["runtime"],
        )

    selection = _selection(real)
    added = run_add_candidate_to_pool(
        _add_inputs(
            selection,
            expected_revision=0,
            expected_snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        real["fx"]["ctx"],
        real["runtime"],
    )
    compiled = run_compile_strategy_pool(
        {
            "strategy_type": "approval",
            "expected_pool_revision": added["revision"],
            "expected_pool_snapshot_hash": added["snapshot_hash"],
        },
        real["fx"]["ctx"],
        real["runtime"],
    )
    current = load_current_strategy_candidate_pool_artifact(
        real["runtime"],
        task_id=real["fx"]["task"].id,
        strategy_type="approval",
        expected_pool_revision=added["revision"],
        expected_pool_snapshot_hash=added["snapshot_hash"],
    )

    score_evidence = band["scorecard_band_asset"]["source_refs"][
        "score_evidence"
    ]
    score_vector = band["scorecard_band_asset"]["source_refs"]["score_vector"]
    virtual_field = model_score_virtual_field(score_vector["artifact_id"])
    expected_requirement = {
        "type": "model_score_vector.v1",
        "virtual_field": virtual_field,
        "score_product": "raw_native_uncalibrated_bad_probability",
        "score_evidence_artifact_id": score_evidence["artifact_id"],
        "score_evidence_artifact_content_hash": score_evidence[
            "artifact_content_hash"
        ],
        "score_vector_artifact_id": score_vector["artifact_id"],
        "score_vector_artifact_content_hash": score_vector[
            "artifact_content_hash"
        ],
    }
    [entry] = added["entries"]
    assert entry["source"]["artifact_id"] == selection["artifacts"][0]["artifact_id"]
    assert entry["source"]["asset_id"] == band["asset_id"]
    assert entry["execution"]["requirements"] == [expected_requirement]
    assert compiled["requirements"] == [
        {
            "rule_id": entry["rule_id"],
            "fragment_id": entry["source"]["fragment_id"],
            "requirement": expected_requirement,
        }
    ]
    assert current.pool == added["pool"]
    assert current.compiled_design == compiled["selected_strategy_design"]

    alternate_sample_selection = _alternate_sample_scorecard(real)
    with pytest.raises(StrategyError, match="evidence identity"):
        run_add_candidate_to_pool(
            _add_inputs(
                alternate_sample_selection,
                expected_revision=added["revision"],
                expected_snapshot_hash=added["snapshot_hash"],
            ),
            real["fx"]["ctx"],
            real["runtime"],
        )
    assert (
        StrategyCandidatePoolRepository(
            real["fx"]["settings"].db_path
        ).get_current(real["fx"]["task"].id, "approval")
        == added["pool"]
    )

    selection_artifact = selection["artifacts"][0]
    selection_binding = load_scorecard_cutoff_selection_artifact(
        real["runtime"],
        task_id=real["fx"]["task"].id,
        artifact_id=selection_artifact["artifact_id"],
        expected_artifact_content_hash=selection_artifact["content_hash"],
        expected_selection_id=selection["selection_id"],
        expected_selection_hash=selection["selection_hash"],
    )
    asset_binding = selection_binding.source_asset_binding
    score_binding = asset_binding.score_evidence
    lineage_paths = {
        "selection": selection_binding.path,
        "band asset": asset_binding.path,
        "training evidence": score_binding.training.evidence_path,
        "score evidence": score_binding.evidence_path,
        "score vector": score_binding.vector_path,
        "sample bundle": asset_binding.sample_design.bundle_path,
    }
    transaction = real["runtime"].task_artifacts.transaction
    for path in lineage_paths.values():
        before = path.read_bytes()

        @contextmanager
        def drift_inside_pool_write(
            *,
            _path=path,
            _before=before,
        ):
            with transaction() as conn:
                _path.write_bytes(_before + b"\n")
                try:
                    yield conn
                finally:
                    _path.write_bytes(_before)

        monkeypatch.setattr(
            real["runtime"].task_artifacts,
            "transaction",
            drift_inside_pool_write,
        )
        with pytest.raises(
            StrategyError,
            match=(
                "changed|drift|hash|canonical|verification|binding|"
                "readable|invalid|failed"
            ),
        ):
            run_set_pool_entry_action(
                {
                    "strategy_type": "approval",
                    "rule_id": entry["rule_id"],
                    "action": _action("review", reason="MANUAL_REVIEW"),
                    "expected_pool_revision": added["revision"],
                    "expected_pool_snapshot_hash": added["snapshot_hash"],
                },
                real["fx"]["ctx"],
                real["runtime"],
            )
        monkeypatch.setattr(
            real["runtime"].task_artifacts,
            "transaction",
            transaction,
        )
        assert (
            StrategyCandidatePoolRepository(
                real["fx"]["settings"].db_path
            ).get_current(real["fx"]["task"].id, "approval")
            == added["pool"]
        )

    changed = run_set_pool_entry_action(
        {
            "strategy_type": "approval",
            "rule_id": entry["rule_id"],
            "action": _action("review", reason="MANUAL_REVIEW"),
            "expected_pool_revision": added["revision"],
            "expected_pool_snapshot_hash": added["snapshot_hash"],
        },
        real["fx"]["ctx"],
        real["runtime"],
    )
    assert changed["pool"]["entries"][0]["action"] == _action(
        "review",
        reason="MANUAL_REVIEW",
    )

    reordered = run_reorder_strategy_pool(
        {
            "strategy_type": "approval",
            "ordered_rule_ids": [entry["rule_id"]],
            "expected_pool_revision": changed["revision"],
            "expected_pool_snapshot_hash": changed["snapshot_hash"],
            "reason": "重放唯一评分卡规则",
        },
        real["fx"]["ctx"],
        real["runtime"],
    )
    assert [row["rule_id"] for row in reordered["entries"]] == [entry["rule_id"]]

    removed = run_remove_pool_entry(
        {
            "strategy_type": "approval",
            "rule_id": entry["rule_id"],
            "expected_pool_revision": reordered["revision"],
            "expected_pool_snapshot_hash": reordered["snapshot_hash"],
            "reason": "测试移除评分卡规则",
        },
        real["fx"]["ctx"],
        real["runtime"],
    )
    assert removed["entries"] == []
    assert (
        StrategyCandidatePoolRepository(
            real["fx"]["settings"].db_path
        ).get_current(real["fx"]["task"].id, "approval")
        == removed["pool"]
    )
