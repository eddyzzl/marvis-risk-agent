from __future__ import annotations

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
import threading

import pytest

from marvis.artifacts import ArtifactUnitOfWork
from marvis.packs.strategy import tools as strategy_tools
import marvis.packs.strategy.model_evidence_tools as model_evidence_tools
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.model_evidence_tools import (
    MODEL_EVIDENCE_V2_ARTIFACT_KIND,
    MODEL_EVIDENCE_V2_TOOL_SCHEMA_VERSION,
    load_strategy_model_evidence_v2_artifact,
    run_materialize_model_evidence_v2,
    validate_materialize_model_evidence_v2_tool_output,
)
from marvis.packs.strategy.sample_design_v2_tools import (
    run_materialize_sample_design_v2,
)
from marvis.packs.strategy.sample_design_tools import run_materialize_sample_design
from marvis.db import TaskRepository
from marvis.domain import TaskCreate
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.repositories.task_artifacts import TaskArtifactDataError

from test_strategy_sample_design_v2_tool import _setup as _sample_v2_setup


def _fixture(tmp_path: Path, *, not_matured: bool = False) -> dict:
    fx = _sample_v2_setup(tmp_path)
    if not_matured:
        fx["request"] = deepcopy(fx["request"])
        fx["request"]["scope"] = "exploration_only"
        fx["request"]["maturity"] = {
            "status": "not_matured",
            "performance_window_days": 30,
            "cutoff_date": "2026-02-15",
            "reason": "Later cohorts have not completed the performance window.",
        }
    sample_v2 = run_materialize_sample_design_v2(
        fx["request"], fx["ctx"], fx["runtime"]
    )
    candidate = strategy_tools.tool_analyze_univariate_candidates(
        {
            "dataset_id": fx["dataset"].id,
            "expected_content_hash": fx["dataset"].content_hash,
            "workspace_revision": fx["workspace"].revision,
            "analysis_generation": fx["workspace"].analysis_generation,
            "semantic_mapping_hash": sample_v2["bundle"]["sample_design"][
                "identity"
            ]["workspace_ref"]["semantic_mapping_hash"],
            "target_col": "bad",
            "sample_design_ref": fx["request"]["legacy_sample_design_ref"],
            "drop_nan_labels": True,
            "features": ["legacy_score", "channel"],
            "methods": ["equal_width"],
            "bin_count": 3,
            "loan_amount_col": "loan_amount",
            "overdue_amount_col": "overdue_amount",
        },
        fx["ctx"],
    )
    source = next(
        item
        for item in candidate["artifacts"]
        if item["kind"] == "strategy_candidate_json"
    )
    inputs = {
        "sample_design_ref": {
            "membership_artifact_id": sample_v2["artifacts"]["membership"][
                "artifact_id"
            ],
            "expected_membership_artifact_content_hash": sample_v2["artifacts"][
                "membership"
            ]["content_hash"],
            "bundle_artifact_id": sample_v2["artifacts"]["bundle"]["artifact_id"],
            "expected_bundle_artifact_content_hash": sample_v2["artifacts"][
                "bundle"
            ]["content_hash"],
            "expected_bundle_id": sample_v2["bundle_id"],
            "expected_sample_design_id": sample_v2["sample_design_id"],
            "expected_sample_design_content_hash": sample_v2[
                "sample_design_content_hash"
            ],
        },
        "univariate_sources": [
            {
                "artifact_id": source["artifact_id"],
                "expected_artifact_content_hash": source["content_hash"],
                "expected_candidate_id": candidate["candidate_id"],
                "expected_evidence_hash": candidate["evidence_hash"],
            }
        ],
    }
    return {**fx, "sample_v2": sample_v2, "candidate": candidate, "inputs": inputs}


def _registered_model_evidence(fx: dict) -> dict:
    records = [
        item
        for item in TaskArtifactRepository(
            fx["settings"].db_path
        ).list_for_task(fx["task"].id)
        if item["kind"] == MODEL_EVIDENCE_V2_ARTIFACT_KIND
    ]
    assert len(records) == 1
    return records[0]


def _additional_candidate_source(fx: dict) -> dict[str, str]:
    identity = fx["sample_v2"]["bundle"]["sample_design"]["identity"]
    candidate = strategy_tools.tool_analyze_univariate_candidates(
        {
            "dataset_id": fx["dataset"].id,
            "expected_content_hash": fx["dataset"].content_hash,
            "workspace_revision": fx["workspace"].revision,
            "analysis_generation": fx["workspace"].analysis_generation,
            "semantic_mapping_hash": identity["workspace_ref"][
                "semantic_mapping_hash"
            ],
            "target_col": "bad",
            "sample_design_ref": fx["request"]["legacy_sample_design_ref"],
            "drop_nan_labels": True,
            "features": ["legacy_score"],
            "methods": ["equal_frequency"],
            "bin_count": 4,
            "loan_amount_col": "loan_amount",
            "overdue_amount_col": "overdue_amount",
        },
        fx["ctx"],
    )
    source = next(
        item
        for item in candidate["artifacts"]
        if item["kind"] == "strategy_candidate_json"
    )
    return {
        "artifact_id": source["artifact_id"],
        "expected_artifact_content_hash": source["content_hash"],
        "expected_candidate_id": candidate["candidate_id"],
        "expected_evidence_hash": candidate["evidence_hash"],
    }


def test_materialize_model_evidence_is_univariate_only_idempotent_and_loadable(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)

    first = run_materialize_model_evidence_v2(
        fx["inputs"], fx["ctx"], fx["runtime"]
    )
    repeated = run_materialize_model_evidence_v2(
        fx["inputs"], fx["ctx"], fx["runtime"]
    )

    assert repeated == first
    assert first["schema_version"] == MODEL_EVIDENCE_V2_TOOL_SCHEMA_VERSION
    assert first["schema_version"] == "strategy.materialize-model-evidence-v2-tool.v3"
    assert validate_materialize_model_evidence_v2_tool_output(first) == first
    assert first["artifact"] == {
        "kind": MODEL_EVIDENCE_V2_ARTIFACT_KIND,
        "format": "json",
        "filename": f"{first['bundle_id']}.json",
        "content_hash": _registered_model_evidence(fx)["content_hash"],
    }
    assert first["bundle"]["model_evidence"] == []
    assert first["bundle"]["comparison_evidence"] == []
    assert len(first["bundle"]["univariate_evidence"]) == 2
    assert {
        item["feature"]: item["analysis_variant"]
        for item in first["bundle"]["univariate_evidence"]
    } == {"legacy_score": "equal_width", "channel": "categorical"}
    assert {
        (item["sample_ref"]["population"], item["sample_ref"]["partition"])
        for item in first["bundle"]["univariate_evidence"]
    } == {("risk", "development")}
    assert not any(
        observation["metric_key"].startswith("monthly_")
        for evidence in first["bundle"]["univariate_evidence"]
        for observation in evidence["observations"]
    )
    assert all(
        bin_ref["bin_id"].startswith(("equal_width:", "categorical:"))
        for evidence in first["bundle"]["univariate_evidence"]
        for bin_ref in evidence["bins"]
    )
    assert any(
        observation["metric_key"] == "sentinel_rate"
        and observation["status"] == "unavailable"
        and observation["sample_count"] is None
        for evidence in first["bundle"]["univariate_evidence"]
        for observation in evidence["observations"]
    )
    assert any(
        observation["metric_key"] in {"bin_bad_rate", "lift"}
        and observation["status"] == "unavailable"
        and observation["value"] is None
        for evidence in first["bundle"]["univariate_evidence"]
        for observation in evidence["observations"]
    )
    assert "warnings" not in first
    assert set(first["source_artifacts"][0]) == {
        "artifact_id",
        "kind",
        "content_hash",
    }

    record = _registered_model_evidence(fx)
    loaded = load_strategy_model_evidence_v2_artifact(
        fx["runtime"],
        task_id=fx["task"].id,
        artifact_id=record["id"],
        expected_artifact_content_hash=first["artifact"]["content_hash"],
        expected_bundle_id=first["bundle_id"],
        expected_bundle_content_hash=first["bundle_content_hash"],
        sample_design_ref=fx["inputs"]["sample_design_ref"],
    )
    assert loaded.bundle == first["bundle"]
    assert len(
        [
            item
            for item in TaskArtifactRepository(
                fx["settings"].db_path
            ).list_for_task(fx["task"].id)
            if item["kind"] == MODEL_EVIDENCE_V2_ARTIFACT_KIND
        ]
    ) == 1


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: {**value, "metrics": []}, "unsupported: metrics"),
        (lambda value: {**value, "bins": []}, "unsupported: bins"),
        (lambda value: {**value, "model_refs": []}, "unsupported: model_refs"),
        (
            lambda value: {
                **value,
                "sample_design_ref": {
                    **value["sample_design_ref"],
                    "caller_sample_metrics": {},
                },
            },
            "caller_sample_metrics",
        ),
        (
            lambda value: {
                **value,
                "univariate_sources": [
                    {**value["univariate_sources"][0], "prebuilt_bundle": {}}
                ],
            },
            "prebuilt_bundle",
        ),
    ],
)
def test_materialize_model_evidence_rejects_caller_facts_and_refs(
    tmp_path: Path, mutation, match: str
) -> None:
    fx = _fixture(tmp_path)

    with pytest.raises(StrategyError, match=match):
        run_materialize_model_evidence_v2(
            mutation(deepcopy(fx["inputs"])), fx["ctx"], fx["runtime"]
        )


def test_cached_model_evidence_envelope_rejects_tampering(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    output = run_materialize_model_evidence_v2(
        fx["inputs"], fx["ctx"], fx["runtime"]
    )

    forged_warning = {**deepcopy(output), "warnings": ["forged"]}
    with pytest.raises(StrategyError, match="unsupported: warnings"):
        validate_materialize_model_evidence_v2_tool_output(forged_warning)

    for field, value in (
        ("artifact_id", "0" * 64),
        ("download_url", "/api/tasks/forged/task-artifacts/forged/download"),
    ):
        forged_artifact = deepcopy(output)
        forged_artifact["artifact"][field] = value
        with pytest.raises(StrategyError, match=f"unsupported: {field}"):
            validate_materialize_model_evidence_v2_tool_output(forged_artifact)

    forged_source = deepcopy(output)
    forged_source["source_artifacts"][0]["candidate_id"] = "candidate-forged"
    with pytest.raises(StrategyError, match="unsupported: candidate_id"):
        validate_materialize_model_evidence_v2_tool_output(forged_source)

    forged_source_hash = deepcopy(output)
    forged_source_hash["source_artifacts"][0]["content_hash"] = "0" * 64
    with pytest.raises(StrategyError, match="do not match bundle analysis refs"):
        validate_materialize_model_evidence_v2_tool_output(forged_source_hash)

    excessive_sources = deepcopy(output)
    source = excessive_sources["source_artifacts"][0]
    excessive_sources["source_artifacts"] = [
        {**source, "artifact_id": f"{index + 1:064x}"}
        for index in range(model_evidence_tools._MAX_UNIVARIATE_SOURCES + 1)
    ]
    with pytest.raises(StrategyError, match="source summaries exceed source budget"):
        validate_materialize_model_evidence_v2_tool_output(excessive_sources)

    forged_model = deepcopy(output)
    forged_model["bundle"]["model_evidence"] = [
        forged_model["bundle"]["univariate_evidence"][0]
    ]
    with pytest.raises(StrategyError):
        validate_materialize_model_evidence_v2_tool_output(forged_model)


def test_model_evidence_requires_the_exact_v2_legacy_compatibility_ref(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)
    identity = fx["sample_v2"]["bundle"]["sample_design"]["identity"]
    alternate = run_materialize_sample_design(
        {
            "dataset_id": fx["dataset"].id,
            "expected_dataset_content_hash": fx["dataset"].content_hash,
            "workspace_revision": fx["workspace"].revision,
            "workspace_generation": fx["workspace"].analysis_generation,
            "semantic_mapping_hash": identity["workspace_ref"][
                "semantic_mapping_hash"
            ],
            "target_col": "bad",
            "target_bad_value": 1,
            "performance_window_status": "provided",
            "performance_window_days": 30,
            "observation_window_status": "provided",
            "observation_window_start": "2026-01-01",
            "observation_window_end": "2026-03-31",
            "maturity_status": "confirmed_matured",
            "split_col": "sample_split",
            "development_values": ["dev"],
            "validation_values": ["valid"],
            "oot_values": ["oot"],
            "month_col": "apply_month",
            "weight_col": "weight",
            "loan_amount_col": "loan_amount",
            "overdue_amount_col": "overdue_amount",
            "drop_nan_labels": True,
        },
        fx["ctx"],
        fx["runtime"],
    )
    alternate_ref = {
        "artifact_id": alternate["artifact"]["artifact_id"],
        "artifact_content_hash": alternate["artifact"]["content_hash"],
        "sample_design_id": alternate["sample_design_id"],
        "sample_design_content_hash": alternate["content_hash"],
        "partition": "development",
    }
    candidate = strategy_tools.tool_analyze_univariate_candidates(
        {
            "dataset_id": fx["dataset"].id,
            "expected_content_hash": fx["dataset"].content_hash,
            "workspace_revision": fx["workspace"].revision,
            "analysis_generation": fx["workspace"].analysis_generation,
            "semantic_mapping_hash": identity["workspace_ref"][
                "semantic_mapping_hash"
            ],
            "target_col": "bad",
            "sample_design_ref": alternate_ref,
            "drop_nan_labels": True,
            "features": ["legacy_score"],
            "methods": ["equal_width"],
            "bin_count": 3,
            "loan_amount_col": "loan_amount",
            "overdue_amount_col": "overdue_amount",
        },
        fx["ctx"],
    )
    source = next(
        item
        for item in candidate["artifacts"]
        if item["kind"] == "strategy_candidate_json"
    )
    inputs = deepcopy(fx["inputs"])
    inputs["univariate_sources"] = [
        {
            "artifact_id": source["artifact_id"],
            "expected_artifact_content_hash": source["content_hash"],
            "expected_candidate_id": candidate["candidate_id"],
            "expected_evidence_hash": candidate["evidence_hash"],
        }
    ]

    with pytest.raises(StrategyError, match="legacy sample binding.*V2 compatibility"):
        run_materialize_model_evidence_v2(inputs, fx["ctx"], fx["runtime"])


def test_model_evidence_types_immature_outcomes_without_inventing_values(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path, not_matured=True)

    output = run_materialize_model_evidence_v2(
        fx["inputs"], fx["ctx"], fx["runtime"]
    )

    outcome_keys = {
        "bin_good_count",
        "bin_bad_count",
        "bin_bad_rate",
        "bin_woe",
        "bin_iv",
        "iv",
        "ks",
        "auc",
        "lift",
    }
    outcomes = [
        observation
        for evidence in output["bundle"]["univariate_evidence"]
        for observation in evidence["observations"]
        if observation["metric_key"] in outcome_keys
    ]
    assert outcomes
    assert {item["status"] for item in outcomes} == {"not_matured"}
    assert all(
        item["value"] is None
        and item["numerator"] is None
        and item["denominator"] is None
        and item["sample_count"] is None
        for item in outcomes
    )
    assert any(
        item["metric_key"] == "bin_count" and item["status"] == "present"
        for evidence in output["bundle"]["univariate_evidence"]
        for item in evidence["observations"]
    )


@pytest.mark.parametrize("target", ["bundle", "source"])
def test_verified_model_evidence_loader_rejects_byte_tampering(
    tmp_path: Path, target: str
) -> None:
    fx = _fixture(tmp_path)
    output = run_materialize_model_evidence_v2(
        fx["inputs"], fx["ctx"], fx["runtime"]
    )
    repository = TaskArtifactRepository(fx["settings"].db_path)
    model_record = _registered_model_evidence(fx)
    artifact_id = (
        model_record["id"]
        if target == "bundle"
        else fx["inputs"]["univariate_sources"][0]["artifact_id"]
    )
    record = repository.get_for_task(fx["task"].id, artifact_id)
    assert record is not None
    Path(record["path"]).write_bytes(Path(record["path"]).read_bytes() + b"tampered")

    with pytest.raises(StrategyError, match="hash|bytes|canonical"):
        load_strategy_model_evidence_v2_artifact(
            fx["runtime"],
            task_id=fx["task"].id,
            artifact_id=model_record["id"],
            expected_artifact_content_hash=output["artifact"]["content_hash"],
            expected_bundle_id=output["bundle_id"],
            expected_bundle_content_hash=output["bundle_content_hash"],
            sample_design_ref=fx["inputs"]["sample_design_ref"],
        )


@pytest.mark.parametrize("target", ["bundle", "source"])
def test_model_evidence_rejects_symlink_substitution(
    tmp_path: Path, target: str
) -> None:
    fx = _fixture(tmp_path)
    output = run_materialize_model_evidence_v2(
        fx["inputs"], fx["ctx"], fx["runtime"]
    )
    repository = TaskArtifactRepository(fx["settings"].db_path)
    model_record = _registered_model_evidence(fx)
    artifact_id = (
        model_record["id"]
        if target == "bundle"
        else fx["inputs"]["univariate_sources"][0]["artifact_id"]
    )
    record = repository.get_for_task(fx["task"].id, artifact_id)
    assert record is not None
    path = Path(record["path"])
    backup = path.with_suffix(path.suffix + ".saved")
    path.rename(backup)
    path.symlink_to(backup)

    with pytest.raises(StrategyError, match="regular file|symlink"):
        if target == "bundle":
            load_strategy_model_evidence_v2_artifact(
                fx["runtime"],
                task_id=fx["task"].id,
                artifact_id=model_record["id"],
                expected_artifact_content_hash=output["artifact"]["content_hash"],
                expected_bundle_id=output["bundle_id"],
                expected_bundle_content_hash=output["bundle_content_hash"],
                sample_design_ref=fx["inputs"]["sample_design_ref"],
            )
        else:
            run_materialize_model_evidence_v2(
                fx["inputs"], fx["ctx"], fx["runtime"]
            )


def test_model_evidence_registration_failure_rolls_back_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx = _fixture(tmp_path)
    original = fx["runtime"].task_artifacts.register_on_connection

    def fail_model_artifact(*args, **kwargs):
        if kwargs.get("kind") == MODEL_EVIDENCE_V2_ARTIFACT_KIND:
            raise TaskArtifactDataError("injected model-evidence registration failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        fx["runtime"].task_artifacts,
        "register_on_connection",
        fail_model_artifact,
    )
    with pytest.raises(StrategyError, match="injected model-evidence"):
        run_materialize_model_evidence_v2(
            fx["inputs"], fx["ctx"], fx["runtime"]
        )

    records = TaskArtifactRepository(fx["settings"].db_path).list_for_task(
        fx["task"].id
    )
    assert not any(item["kind"] == MODEL_EVIDENCE_V2_ARTIFACT_KIND for item in records)
    out_dir = (
        fx["settings"].tasks_dir / fx["task"].id / "strategy_model_evidence"
    )
    assert not list(out_dir.glob("*.json"))


def test_model_evidence_post_database_commit_cleanup_failure_keeps_registered_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx = _fixture(tmp_path)
    original_commit = ArtifactUnitOfWork.commit

    def fail_cleanup(_self):
        raise RuntimeError("injected post-commit cleanup failure")

    monkeypatch.setattr(ArtifactUnitOfWork, "commit", fail_cleanup)
    with pytest.raises(RuntimeError, match="injected post-commit cleanup failure"):
        run_materialize_model_evidence_v2(
            fx["inputs"], fx["ctx"], fx["runtime"]
        )

    records = [
        item
        for item in TaskArtifactRepository(
            fx["settings"].db_path
        ).list_for_task(fx["task"].id)
        if item["kind"] == MODEL_EVIDENCE_V2_ARTIFACT_KIND
    ]
    assert len(records) == 1
    assert Path(records[0]["path"]).is_file()

    monkeypatch.setattr(ArtifactUnitOfWork, "commit", original_commit)
    replay = run_materialize_model_evidence_v2(
        fx["inputs"], fx["ctx"], fx["runtime"]
    )
    assert replay["artifact"]["content_hash"] == records[0]["content_hash"]
    assert _registered_model_evidence(fx)["id"] == records[0]["id"]


def test_model_evidence_rejects_source_fanout_before_artifact_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx = _fixture(tmp_path)
    inputs = deepcopy(fx["inputs"])
    source_limit = model_evidence_tools._MAX_UNIVARIATE_SOURCES
    inputs["univariate_sources"] = [
        {
            "artifact_id": f"{index + 1:064x}",
            "expected_artifact_content_hash": f"{index + 2:064x}",
            "expected_candidate_id": f"candidate-{index}",
            "expected_evidence_hash": f"{index + 3:064x}",
        }
        for index in range(source_limit + 1)
    ]
    artifact_io_called = False

    def fail_if_artifact_io(*_args, **_kwargs):
        nonlocal artifact_io_called
        artifact_io_called = True
        raise AssertionError("artifact I/O must not run for oversized source fanout")

    monkeypatch.setattr(
        model_evidence_tools,
        "_load_sample_design",
        fail_if_artifact_io,
    )

    with pytest.raises(StrategyError, match="univariate_sources exceeds source budget"):
        run_materialize_model_evidence_v2(inputs, fx["ctx"], fx["runtime"])
    assert artifact_io_called is False


def test_model_evidence_rejects_cumulative_source_bytes_before_next_read_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx = _fixture(tmp_path)
    inputs = deepcopy(fx["inputs"])
    inputs["univariate_sources"].append(_additional_candidate_source(fx))
    ordered_sources = sorted(
        inputs["univariate_sources"], key=lambda item: item["artifact_id"]
    )
    repository = TaskArtifactRepository(fx["settings"].db_path)
    source_paths = []
    for source in ordered_sources:
        record = repository.get_for_task(fx["task"].id, source["artifact_id"])
        assert record is not None
        source_paths.append(Path(record["path"]))
    source_sizes = [path.stat().st_size for path in source_paths]
    monkeypatch.setattr(
        model_evidence_tools,
        "_MAX_CANDIDATE_SOURCE_BYTES_TOTAL",
        sum(source_sizes) - 1,
    )
    original_read_bytes = Path.read_bytes
    candidate_reads: list[Path] = []

    def track_candidate_reads(path: Path) -> bytes:
        if path in source_paths:
            candidate_reads.append(path)
        return original_read_bytes(path)

    translate_called = False
    persist_called = False

    def fail_if_translated(*_args, **_kwargs):
        nonlocal translate_called
        translate_called = True
        raise AssertionError("translation must not run after source byte exhaustion")

    def fail_if_persisted(*_args, **_kwargs):
        nonlocal persist_called
        persist_called = True
        raise AssertionError("output persistence must not run after byte exhaustion")

    monkeypatch.setattr(Path, "read_bytes", track_candidate_reads)
    monkeypatch.setattr(model_evidence_tools, "_translate_sources", fail_if_translated)
    monkeypatch.setattr(model_evidence_tools, "_persist_bundle", fail_if_persisted)

    with pytest.raises(StrategyError, match="cumulative candidate source byte budget"):
        run_materialize_model_evidence_v2(inputs, fx["ctx"], fx["runtime"])

    assert candidate_reads == [source_paths[0]]
    assert translate_called is False
    assert persist_called is False
    assert not [
        item
        for item in repository.list_for_task(fx["task"].id)
        if item["kind"] == MODEL_EVIDENCE_V2_ARTIFACT_KIND
    ]
    out_dir = fx["settings"].tasks_dir / fx["task"].id / "strategy_model_evidence"
    assert not list(out_dir.glob("*.json"))


def test_model_evidence_translation_observation_budget_fails_before_output_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx = _fixture(tmp_path)
    persist_called = False

    def fail_if_persisted(*_args, **_kwargs):
        nonlocal persist_called
        persist_called = True
        raise AssertionError("output persistence must not run after work exhaustion")

    monkeypatch.setattr(model_evidence_tools, "_MAX_TRANSLATION_OBSERVATIONS", 1)
    monkeypatch.setattr(model_evidence_tools, "_persist_bundle", fail_if_persisted)

    with pytest.raises(StrategyError, match="translation observation budget"):
        run_materialize_model_evidence_v2(
            fx["inputs"], fx["ctx"], fx["runtime"]
        )

    assert persist_called is False
    assert not [
        item
        for item in TaskArtifactRepository(
            fx["settings"].db_path
        ).list_for_task(fx["task"].id)
        if item["kind"] == MODEL_EVIDENCE_V2_ARTIFACT_KIND
    ]
    out_dir = fx["settings"].tasks_dir / fx["task"].id / "strategy_model_evidence"
    assert not list(out_dir.glob("*.json"))


@pytest.mark.parametrize(
    ("limit_name", "match"),
    [
        ("MAX_UNIVARIATE_EVIDENCE", "translation evidence budget"),
        ("_MAX_TRANSLATION_WARNINGS", "translation warning budget"),
    ],
)
def test_model_evidence_translation_item_budgets_fail_before_output_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    match: str,
) -> None:
    fx = _fixture(tmp_path)
    persist_called = False

    def fail_if_persisted(*_args, **_kwargs):
        nonlocal persist_called
        persist_called = True
        raise AssertionError("output persistence must not run after item exhaustion")

    monkeypatch.setattr(model_evidence_tools, limit_name, 1)
    monkeypatch.setattr(model_evidence_tools, "_persist_bundle", fail_if_persisted)

    with pytest.raises(StrategyError, match=match):
        run_materialize_model_evidence_v2(
            fx["inputs"], fx["ctx"], fx["runtime"]
        )

    assert persist_called is False
    assert not [
        item
        for item in TaskArtifactRepository(
            fx["settings"].db_path
        ).list_for_task(fx["task"].id)
        if item["kind"] == MODEL_EVIDENCE_V2_ARTIFACT_KIND
    ]


def test_model_evidence_rechecks_source_under_writer_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx = _fixture(tmp_path)
    original = model_evidence_tools._persist_bundle
    source_id = fx["inputs"]["univariate_sources"][0]["artifact_id"]
    source_record = TaskArtifactRepository(fx["settings"].db_path).get_for_task(
        fx["task"].id, source_id
    )
    assert source_record is not None
    source_path = Path(source_record["path"])

    def drift_then_persist(*args, **kwargs):
        source_path.write_bytes(source_path.read_bytes() + b"drifted-under-lock")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        model_evidence_tools, "_persist_bundle", drift_then_persist
    )
    with pytest.raises(StrategyError, match="hash|canonical"):
        run_materialize_model_evidence_v2(
            fx["inputs"], fx["ctx"], fx["runtime"]
        )
    out_dir = (
        fx["settings"].tasks_dir / fx["task"].id / "strategy_model_evidence"
    )
    assert not list(out_dir.glob("*.json"))


def test_model_evidence_concurrent_replay_publishes_one_artifact(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)
    barrier = threading.Barrier(2)

    def invoke() -> dict:
        barrier.wait()
        return run_materialize_model_evidence_v2(
            fx["inputs"], fx["ctx"], fx["runtime"]
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(invoke)
        second_future = executor.submit(invoke)
        first = first_future.result(timeout=30)
        second = second_future.result(timeout=30)

    assert first == second
    records = TaskArtifactRepository(fx["settings"].db_path).list_for_task(
        fx["task"].id
    )
    assert sum(
        item["kind"] == MODEL_EVIDENCE_V2_ARTIFACT_KIND for item in records
    ) == 1


def test_model_evidence_fails_closed_across_tasks_and_stale_workspace(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)
    output = run_materialize_model_evidence_v2(
        fx["inputs"], fx["ctx"], fx["runtime"]
    )
    model_record = _registered_model_evidence(fx)
    foreign = TaskRepository(fx["settings"].db_path).create_task(
        TaskCreate(
            model_name="foreign",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "foreign"),
            task_type="strategy",
            target_col="bad",
        )
    )
    with pytest.raises(StrategyError, match="registry row|binding"):
        load_strategy_model_evidence_v2_artifact(
            fx["runtime"],
            task_id=foreign.id,
            artifact_id=model_record["id"],
            expected_artifact_content_hash=output["artifact"]["content_hash"],
            expected_bundle_id=output["bundle_id"],
            expected_bundle_content_hash=output["bundle_content_hash"],
            sample_design_ref=fx["inputs"]["sample_design_ref"],
        )

    with sqlite3.connect(fx["settings"].db_path) as conn:
        conn.execute(
            "UPDATE data_workspaces SET revision = revision + 1 WHERE task_id = ?",
            (fx["task"].id,),
        )
    with pytest.raises(StrategyError, match="workspace|binding|DataWorkspace"):
        load_strategy_model_evidence_v2_artifact(
            fx["runtime"],
            task_id=fx["task"].id,
            artifact_id=model_record["id"],
            expected_artifact_content_hash=output["artifact"]["content_hash"],
            expected_bundle_id=output["bundle_id"],
            expected_bundle_content_hash=output["bundle_content_hash"],
            sample_design_ref=fx["inputs"]["sample_design_ref"],
        )
