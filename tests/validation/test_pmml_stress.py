from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import marvis.validation.pmml_stress as stress_module
from marvis.job_cancellation import JobCancelled
from marvis.validation.config import ValidationConfig
from marvis.validation.input_contracts import TransformationSpec
from marvis.validation.pmml_score_artifacts import run_pmml_scoring
from marvis.validation.pmml_stress import (
    ScenarioScoreArtifact,
    load_aligned_scenario_scores,
    load_or_run_stress_scenario,
    materialize_cached_stress_scenario,
    materialize_oot_pmml_inputs,
    run_pmml_stress,
    stress_cache_key,
)
from marvis.validation.sample_schema import inspect_sample_schema
from marvis.validation.stress_test import require_complete_stress_result


PMML_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "min_lr.pmml"


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _spawn_cache_worker(
    cache_dir: str,
    output_path: str,
    counter_path: str,
    barrier,
) -> None:
    cache = Path(cache_dir)
    output = Path(output_path)
    counter = Path(counter_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    row_ids = np.array([8, 9], dtype=np.int64)
    barrier.wait(timeout=15)

    def produce(path: Path) -> ScenarioScoreArtifact:
        with counter.open("a", encoding="utf-8") as handle:
            handle.write("produced\n")
        pq.write_table(
            pa.table(
                {
                    "row_id": pa.array(row_ids, type=pa.int64()),
                    "pmml_score": pa.array([0.2, 0.8], type=pa.float64()),
                },
                schema=stress_module.SCORE_ARTIFACT_SCHEMA,
            ),
            path,
        )
        return ScenarioScoreArtifact("征信", path, 2, _digest(path))

    load_or_run_stress_scenario(
        cache_dir=cache,
        cache_key="a" * 64,
        category="征信",
        expected_row_ids=row_ids,
        output_path=output,
        runner=produce,
    )


def _sample_frame(*, has_oot: bool = True) -> pd.DataFrame:
    splits = (
        ["train"] * 4 + ["test"] * 4 + ["oot"] * 4
        if has_oot
        else ["train"] * 4 + ["test"] * 8
    )
    return pd.DataFrame(
        {
            "x1": [-2.0, -1.0, 1.0, 2.0] * 3,
            "x2": [0.0, 1.0, 0.0, 1.0] * 3,
            "y": [0, 0, 1, 1] * 3,
            "split": splits,
            "apply_month": ["202601"] * 4
            + ["202602"] * 4
            + ["202603"] * 4,
            "poison": ["unused"] * 12,
        }
    )


class _RecordingScorer:
    def __init__(self) -> None:
        self.calls: list[pd.DataFrame] = []

    def score_chunk(self, frame: pd.DataFrame) -> pd.Series:
        self.calls.append(frame.copy())
        linear = np.clip(
            frame["x1"].to_numpy(dtype=float)
            - frame["x2"].to_numpy(dtype=float),
            -20,
            20,
        )
        return pd.Series(1.0 / (1.0 + np.exp(-linear)), dtype="float64")


def _ready_for_sample(ready_contract, sample_path: Path, **changes):
    schema = inspect_sample_schema(sample_path)
    contract = replace(
        ready_contract,
        sample_schema=schema,
        material_hashes={
            **ready_contract.material_hashes,
            "sample": schema.sha256,
            "pmml": _digest(PMML_FIXTURE),
        },
    )
    return replace(contract, **changes) if changes else contract


def _config() -> ValidationConfig:
    return ValidationConfig(
        target_col="__target__",
        score_col="__pmml_score__",
        split_col="__split__",
        time_col="__time__",
        bin_count=2,
    )


def _baseline(tmp_path: Path, contract, sample_path: Path):
    path = tmp_path / "baseline.parquet"
    result = run_pmml_scoring(
        contract=contract,
        sample_path=sample_path,
        pmml_path=PMML_FIXTURE,
        score_path=path,
        chunk_size=3,
        scorer=_RecordingScorer(),
    )
    return path, result


def test_stress_reuses_baseline_and_scores_every_category_on_complete_oot(
    tmp_path: Path, ready_contract
):
    sample_path = tmp_path / "sample.parquet"
    _sample_frame().to_parquet(sample_path, index=False, row_group_size=3)
    contract = _ready_for_sample(ready_contract, sample_path)
    baseline_path, scoring = _baseline(tmp_path, contract, sample_path)
    scorer = _RecordingScorer()

    result = run_pmml_stress(
        contract=contract,
        config=_config(),
        sample_path=sample_path,
        baseline_score_path=baseline_path,
        scoring_result=scoring,
        scenario_dir=tmp_path / "stress",
        feature_categories={"征信": ("x1",), "内部": ("x2",)},
        scorer=scorer,
        chunk_size=2,
        category_source_counts={"notebook": 0, "dictionary": 2, "unresolved": 0},
    )

    assert result.baseline.sample_count == 4
    assert [row.category for row in result.per_category] == ["征信", "内部"]
    assert result.status == "completed"
    assert len(scorer.calls) == 4
    assert sum(len(frame) for frame in scorer.calls[:2]) == 4
    assert sum(len(frame) for frame in scorer.calls[2:]) == 4
    assert all((frame["x1"] == -9999).all() for frame in scorer.calls[:2])
    assert all((frame["x2"] == -9999).all() for frame in scorer.calls[2:])
    assert (tmp_path / "stress" / "category_001.parquet").is_file()
    assert (tmp_path / "stress" / "category_002.parquet").is_file()


def test_stress_reads_and_transforms_the_source_sample_once(
    tmp_path: Path, ready_contract, monkeypatch
):
    sample_path = tmp_path / "sample.parquet"
    _sample_frame().to_parquet(sample_path, index=False)
    unrelated = TransformationSpec("copy", "unused", ("poison",), {})
    contract = _ready_for_sample(
        ready_contract,
        sample_path,
        transformations=(unrelated,),
    )
    baseline_path, scoring = _baseline(tmp_path, contract, sample_path)
    calls = 0
    original = stress_module.iter_sample_chunks

    def record(*args, **kwargs):
        nonlocal calls
        calls += 1
        yield from original(*args, **kwargs)

    monkeypatch.setattr(stress_module, "iter_sample_chunks", record)
    run_pmml_stress(
        contract=contract,
        config=_config(),
        sample_path=sample_path,
        baseline_score_path=baseline_path,
        scoring_result=scoring,
        scenario_dir=tmp_path / "stress",
        feature_categories={"征信": ("x1",), "内部": ("x2",)},
        scorer=_RecordingScorer(),
        chunk_size=2,
    )

    assert calls == 1


def test_oot_materialization_preserves_non_contiguous_source_row_ids(
    tmp_path: Path, ready_contract
):
    sample_path = tmp_path / "sample.parquet"
    _sample_frame().to_parquet(sample_path, index=False, row_group_size=2)
    contract = _ready_for_sample(ready_contract, sample_path)
    output = tmp_path / "oot-inputs.parquet"

    materialize_oot_pmml_inputs(
        contract=contract,
        sample_path=sample_path,
        oot_row_ids=np.array([1, 4, 9], dtype=np.int64),
        output_path=output,
        chunk_size=2,
    )

    selected = pd.read_parquet(output)
    assert selected["__marvis_source_row_id__"].tolist() == [1, 4, 9]
    assert selected["x1"].tolist() == [-1.0, -2.0, -1.0]


def test_pmml_raw_input_named_row_id_does_not_collide_with_internal_alignment(
    tmp_path: Path, ready_contract
):
    sample_path = tmp_path / "sample.parquet"
    frame = _sample_frame()
    frame.insert(0, "row_id", np.arange(100, 112, dtype=np.int64))
    frame.to_parquet(sample_path, index=False)
    contract = _ready_for_sample(ready_contract, sample_path)
    contract = replace(
        contract,
        pmml_manifest=replace(
            contract.require_pmml_manifest(),
            raw_required_fields=("row_id", "x1", "x2"),
        ),
    )
    baseline_path, scoring = _baseline(tmp_path, contract, sample_path)

    result = run_pmml_stress(
        contract=contract,
        config=_config(),
        sample_path=sample_path,
        baseline_score_path=baseline_path,
        scoring_result=scoring,
        scenario_dir=tmp_path / "stress",
        feature_categories={"业务序号": ("row_id",)},
        scorer=_RecordingScorer(),
        chunk_size=2,
    )

    assert result.per_category[0].dropped_features == ["row_id"]
    inputs = pd.read_parquet(tmp_path / "stress" / "oot_pmml_inputs.parquet")
    assert "row_id" in inputs
    assert "__marvis_source_row_id__" in inputs


@pytest.mark.parametrize(
    ("categories", "message"),
    [
        ({}, "at least one"),
        ({"空类别": ()}, "no raw input fields"),
        ({"错误": ("not_a_pmml_input",)}, "invalid raw input fields"),
    ],
)
def test_stress_rejects_missing_or_invalid_categories(
    tmp_path: Path, ready_contract, categories, message
):
    sample_path = tmp_path / "sample.parquet"
    _sample_frame().to_parquet(sample_path, index=False)
    contract = _ready_for_sample(ready_contract, sample_path)
    baseline_path, scoring = _baseline(tmp_path, contract, sample_path)

    with pytest.raises(ValueError, match=message):
        run_pmml_stress(
            contract=contract,
            config=_config(),
            sample_path=sample_path,
            baseline_score_path=baseline_path,
            scoring_result=scoring,
            scenario_dir=tmp_path / "stress",
            feature_categories=categories,
            scorer=_RecordingScorer(),
            chunk_size=2,
        )


def test_stress_requires_nonempty_oot(tmp_path: Path, ready_contract):
    sample_path = tmp_path / "sample.parquet"
    _sample_frame(has_oot=False).to_parquet(sample_path, index=False)
    contract = _ready_for_sample(ready_contract, sample_path)
    baseline_path, scoring = _baseline(tmp_path, contract, sample_path)

    with pytest.raises(ValueError, match="OOT sample is required"):
        run_pmml_stress(
            contract=contract,
            config=_config(),
            sample_path=sample_path,
            baseline_score_path=baseline_path,
            scoring_result=scoring,
            scenario_dir=tmp_path / "stress",
            feature_categories={"征信": ("x1",)},
            scorer=_RecordingScorer(),
            chunk_size=2,
        )


@pytest.mark.parametrize("mode", ["short", "non_finite"])
def test_invalid_category_scores_leave_no_partial_sidecar(
    tmp_path: Path, ready_contract, mode
):
    sample_path = tmp_path / "sample.parquet"
    _sample_frame().to_parquet(sample_path, index=False)
    contract = _ready_for_sample(ready_contract, sample_path)
    baseline_path, scoring = _baseline(tmp_path, contract, sample_path)

    class InvalidScorer:
        def score_chunk(self, frame):
            if mode == "short":
                return pd.Series([0.5] * max(0, len(frame) - 1))
            return pd.Series([np.nan] * len(frame))

    with pytest.raises(ValueError, match="invalid PMML stress scores"):
        run_pmml_stress(
            contract=contract,
            config=_config(),
            sample_path=sample_path,
            baseline_score_path=baseline_path,
            scoring_result=scoring,
            scenario_dir=tmp_path / "stress",
            feature_categories={"征信": ("x1",)},
            scorer=InvalidScorer(),
            chunk_size=2,
        )

    assert not list((tmp_path / "stress").glob(".*.staging"))
    assert not (tmp_path / "stress" / "category_001.parquet").exists()


def test_cancellation_between_category_batches_preserves_old_final(
    tmp_path: Path, ready_contract
):
    sample_path = tmp_path / "sample.parquet"
    _sample_frame().to_parquet(sample_path, index=False)
    contract = _ready_for_sample(ready_contract, sample_path)
    baseline_path, scoring = _baseline(tmp_path, contract, sample_path)
    scenario_dir = tmp_path / "stress"
    scenario_dir.mkdir()
    final = scenario_dir / "category_001.parquet"
    final.write_bytes(b"old-complete")
    scorer = _RecordingScorer()

    def cancel() -> None:
        if scorer.calls:
            raise KeyboardInterrupt("cancelled")

    with pytest.raises(KeyboardInterrupt, match="cancelled"):
        run_pmml_stress(
            contract=contract,
            config=_config(),
            sample_path=sample_path,
            baseline_score_path=baseline_path,
            scoring_result=scoring,
            scenario_dir=scenario_dir,
            feature_categories={"征信": ("x1",)},
            scorer=scorer,
            chunk_size=2,
            cancellation_check=cancel,
        )

    assert final.read_bytes() == b"old-complete"
    assert not list(scenario_dir.glob(".*.staging"))


def test_aligned_loader_rejects_reordered_rows(tmp_path: Path):
    path = tmp_path / "scenario.parquet"
    pd.DataFrame(
        {"row_id": [9, 8], "pmml_score": [0.2, 0.8]}
    ).to_parquet(path, index=False)
    artifact = ScenarioScoreArtifact("征信", path, 2, _digest(path))

    with pytest.raises(ValueError, match="row alignment"):
        load_aligned_scenario_scores(
            artifact,
            np.array([8, 9], dtype=np.int64),
        )


def test_oot_input_publish_preserves_old_final_if_sample_changes_during_hash(
    tmp_path: Path, ready_contract, monkeypatch
):
    sample_path = tmp_path / "sample.parquet"
    _sample_frame().to_parquet(sample_path, index=False)
    contract = _ready_for_sample(ready_contract, sample_path)
    output = tmp_path / "oot_inputs.parquet"
    output.write_bytes(b"old-complete")
    original = stress_module.sha256_file_cancellable

    def mutate(path, cancellation_check=None, **kwargs):
        digest = original(path, cancellation_check, **kwargs)
        if Path(path).name.endswith(".staging"):
            sample_path.write_bytes(sample_path.read_bytes() + b"changed")
        return digest

    monkeypatch.setattr(stress_module, "sha256_file_cancellable", mutate)

    with pytest.raises(ValueError, match="validation sample changed"):
        materialize_oot_pmml_inputs(
            contract=contract,
            sample_path=sample_path,
            oot_row_ids=np.array([8, 9, 10, 11], dtype=np.int64),
            output_path=output,
            chunk_size=2,
        )

    assert output.read_bytes() == b"old-complete"
    assert not list(tmp_path.glob(".*.staging"))


def test_complete_stress_gate_rejects_unclassified_or_empty_results(
    tmp_path: Path, ready_contract
):
    sample_path = tmp_path / "sample.parquet"
    _sample_frame().to_parquet(sample_path, index=False)
    contract = _ready_for_sample(ready_contract, sample_path)
    baseline_path, scoring = _baseline(tmp_path, contract, sample_path)
    result = run_pmml_stress(
        contract=contract,
        config=_config(),
        sample_path=sample_path,
        baseline_score_path=baseline_path,
        scoring_result=scoring,
        scenario_dir=tmp_path / "stress",
        feature_categories={"征信": ("x1",)},
        scorer=_RecordingScorer(),
        chunk_size=2,
    )

    with pytest.raises(ValueError, match="unclassified"):
        require_complete_stress_result(
            replace(result, unclassified_features=["x2"])
        )
    with pytest.raises(ValueError, match="no completed categories"):
        require_complete_stress_result(replace(result, per_category=[]))
    with pytest.raises(ValueError, match="no complete OOT baseline"):
        require_complete_stress_result(
            replace(result, baseline=replace(result.baseline, ks=float("nan")))
        )


def test_stress_cache_key_covers_order_chunk_sentinel_and_schema():
    common = {
        "baseline_cache_key": "a" * 64,
        "category": "征信",
        "raw_fields": ("x1", "x2"),
        "chunk_size": 100,
    }
    original = stress_cache_key(**common)

    assert stress_cache_key(**{**common, "raw_fields": ("x2", "x1")}) != original
    assert stress_cache_key(**{**common, "chunk_size": 101}) != original
    assert stress_cache_key(**common, sentinel=-8888) != original
    assert stress_cache_key(**{**common, "category": "内部"}) != original
    assert stress_cache_key(**{**common, "baseline_cache_key": "b" * 64}) != original


def test_all_stress_cache_hits_skip_oot_inputs_and_pmml_scoring(
    tmp_path: Path, ready_contract, monkeypatch
):
    sample_path = tmp_path / "sample.parquet"
    _sample_frame().to_parquet(sample_path, index=False, row_group_size=2)
    contract = _ready_for_sample(ready_contract, sample_path)
    baseline_path, scoring = _baseline(tmp_path, contract, sample_path)
    cache_dir = tmp_path / "cache"
    categories = {"征信": ("x1",), "内部": ("x2",)}
    first = run_pmml_stress(
        contract=contract,
        config=_config(),
        sample_path=sample_path,
        baseline_score_path=baseline_path,
        scoring_result=scoring,
        scenario_dir=tmp_path / "first",
        feature_categories=categories,
        scorer=_RecordingScorer(),
        chunk_size=2,
        baseline_cache_key=scoring.cache_key,
        cache_dir=cache_dir,
    )

    def forbidden_oot_inputs(**kwargs):
        raise AssertionError("all cache hits must not materialize OOT inputs")

    def forbidden_scorer_factory():
        raise AssertionError("all cache hits must not load the PMML scorer")

    monkeypatch.setattr(
        stress_module, "materialize_oot_pmml_inputs", forbidden_oot_inputs
    )
    second = run_pmml_stress(
        contract=contract,
        config=_config(),
        sample_path=sample_path,
        baseline_score_path=baseline_path,
        scoring_result=scoring,
        scenario_dir=tmp_path / "second",
        feature_categories=categories,
        scorer_factory=forbidden_scorer_factory,
        chunk_size=2,
        baseline_cache_key=scoring.cache_key,
        cache_dir=cache_dir,
    )

    assert second == first
    assert not (tmp_path / "second" / "oot_pmml_inputs.parquet").exists()
    assert (tmp_path / "second" / "category_001.parquet").is_file()
    assert (tmp_path / "second" / "category_002.parquet").is_file()


def test_cache_misses_create_one_shared_scorer_for_all_categories(
    tmp_path: Path, ready_contract
):
    sample_path = tmp_path / "sample.parquet"
    _sample_frame().to_parquet(sample_path, index=False)
    contract = _ready_for_sample(ready_contract, sample_path)
    baseline_path, scoring = _baseline(tmp_path, contract, sample_path)
    scorer = _RecordingScorer()
    factory_calls = 0

    def factory():
        nonlocal factory_calls
        factory_calls += 1
        return scorer

    run_pmml_stress(
        contract=contract,
        config=_config(),
        sample_path=sample_path,
        baseline_score_path=baseline_path,
        scoring_result=scoring,
        scenario_dir=tmp_path / "stress",
        feature_categories={"征信": ("x1",), "内部": ("x2",)},
        scorer_factory=factory,
        chunk_size=2,
        baseline_cache_key=scoring.cache_key,
        cache_dir=tmp_path / "cache",
    )

    assert factory_calls == 1
    assert len(scorer.calls) == 4


def test_corrupt_stress_cache_metadata_is_deleted_and_recomputed(
    tmp_path: Path, ready_contract
):
    sample_path = tmp_path / "sample.parquet"
    _sample_frame().to_parquet(sample_path, index=False)
    contract = _ready_for_sample(ready_contract, sample_path)
    baseline_path, scoring = _baseline(tmp_path, contract, sample_path)
    cache_dir = tmp_path / "cache"
    arguments = dict(
        contract=contract,
        config=_config(),
        sample_path=sample_path,
        baseline_score_path=baseline_path,
        scoring_result=scoring,
        feature_categories={"征信": ("x1",)},
        chunk_size=2,
        baseline_cache_key=scoring.cache_key,
        cache_dir=cache_dir,
    )
    run_pmml_stress(
        **arguments,
        scenario_dir=tmp_path / "first",
        scorer=_RecordingScorer(),
    )
    key = stress_cache_key(
        baseline_cache_key=scoring.cache_key,
        category="征信",
        raw_fields=("x1",),
        chunk_size=2,
    )
    metadata_path = (
        cache_dir / "stress-v1" / "entries" / key / "metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["row_count"] = True
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    scorer = _RecordingScorer()

    run_pmml_stress(
        **arguments,
        scenario_dir=tmp_path / "second",
        scorer=scorer,
    )

    repaired = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert type(repaired["row_count"]) is int
    assert repaired["row_count"] == 4
    assert sum(len(frame) for frame in scorer.calls) == 4


@pytest.mark.parametrize("mode", ["hash", "schema", "row_order", "non_finite"])
def test_corrupt_stress_cache_scores_are_deleted_and_recomputed(
    tmp_path: Path, ready_contract, mode
):
    sample_path = tmp_path / "sample.parquet"
    _sample_frame().to_parquet(sample_path, index=False)
    contract = _ready_for_sample(ready_contract, sample_path)
    baseline_path, scoring = _baseline(tmp_path, contract, sample_path)
    cache_dir = tmp_path / "cache"
    arguments = dict(
        contract=contract,
        config=_config(),
        sample_path=sample_path,
        baseline_score_path=baseline_path,
        scoring_result=scoring,
        feature_categories={"征信": ("x1",)},
        chunk_size=2,
        baseline_cache_key=scoring.cache_key,
        cache_dir=cache_dir,
    )
    run_pmml_stress(
        **arguments,
        scenario_dir=tmp_path / "first",
        scorer=_RecordingScorer(),
    )
    key = stress_cache_key(
        baseline_cache_key=scoring.cache_key,
        category="征信",
        raw_fields=("x1",),
        chunk_size=2,
    )
    entry = cache_dir / "stress-v1" / "entries" / key
    score_path = entry / "scores.parquet"
    metadata_path = entry / "metadata.json"
    if mode == "hash":
        score_path.write_bytes(score_path.read_bytes() + b"corrupt")
    else:
        row_ids = [11, 10, 9, 8] if mode == "row_order" else [8, 9, 10, 11]
        scores = (
            [0.2, float("inf"), 0.4, 0.8]
            if mode == "non_finite"
            else [1, 2, 3, 4]
            if mode == "schema"
            else [0.2, 0.3, 0.4, 0.8]
        )
        if mode == "schema":
            table = pa.table({"row_id": row_ids, "pmml_score": scores})
        else:
            table = pa.table(
                {
                    "row_id": pa.array(row_ids, type=pa.int64()),
                    "pmml_score": pa.array(scores, type=pa.float64()),
                },
                schema=stress_module.SCORE_ARTIFACT_SCHEMA,
            )
        pq.write_table(table, score_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["score_sha256"] = _digest(score_path)
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    scorer = _RecordingScorer()

    run_pmml_stress(
        **arguments,
        scenario_dir=tmp_path / "second",
        scorer=scorer,
    )

    assert sum(len(frame) for frame in scorer.calls) == 4
    repaired = pd.read_parquet(score_path)
    assert repaired["row_id"].tolist() == [8, 9, 10, 11]
    assert np.isfinite(repaired["pmml_score"].to_numpy(dtype=float)).all()


def test_stress_cache_lock_allows_one_spawned_producer(tmp_path: Path):
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    counter = tmp_path / "producer-count.txt"
    workers = [
        context.Process(
            target=_spawn_cache_worker,
            args=(
                str(tmp_path / "cache"),
                str(tmp_path / f"worker-{index}" / "scores.parquet"),
                str(counter),
                barrier,
            ),
        )
        for index in range(2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)

    assert [worker.exitcode for worker in workers] == [0, 0]
    assert counter.read_text(encoding="utf-8").splitlines() == ["produced"]


@pytest.mark.parametrize("phase", ["hash", "batch"])
def test_cached_stress_validation_propagates_job_cancellation(
    tmp_path: Path, ready_contract, monkeypatch, phase
):
    sample_path = tmp_path / "sample.parquet"
    _sample_frame().to_parquet(sample_path, index=False)
    contract = _ready_for_sample(ready_contract, sample_path)
    baseline_path, scoring = _baseline(tmp_path, contract, sample_path)
    cache_dir = tmp_path / "cache"
    run_pmml_stress(
        contract=contract,
        config=_config(),
        sample_path=sample_path,
        baseline_score_path=baseline_path,
        scoring_result=scoring,
        scenario_dir=tmp_path / "first",
        feature_categories={"征信": ("x1",)},
        scorer=_RecordingScorer(),
        chunk_size=2,
        baseline_cache_key=scoring.cache_key,
        cache_dir=cache_dir,
    )
    key = stress_cache_key(
        baseline_cache_key=scoring.cache_key,
        category="征信",
        raw_fields=("x1",),
        chunk_size=2,
    )
    checks = 0
    if phase == "batch":
        monkeypatch.setattr(
            stress_module,
            "sha256_file_cancellable",
            lambda path, cancellation_check=None: _digest(Path(path)),
        )

    def cancel() -> None:
        nonlocal checks
        checks += 1
        threshold = 3
        if checks == threshold:
            raise JobCancelled(f"cancelled during cached {phase}")

    output = tmp_path / f"cancel-{phase}.parquet"
    output.write_bytes(b"old-complete")
    with pytest.raises(JobCancelled, match=f"cached {phase}"):
        materialize_cached_stress_scenario(
            cache_dir=cache_dir,
            cache_key=key,
            category="征信",
            expected_row_ids=np.array([8, 9, 10, 11], dtype=np.int64),
            output_path=output,
            cancellation_check=cancel,
        )

    assert output.read_bytes() == b"old-complete"


def test_aligned_score_loading_propagates_job_cancellation(
    tmp_path: Path, monkeypatch
):
    path = tmp_path / "scores.parquet"
    pq.write_table(
        pa.table(
            {
                "row_id": pa.array([8, 9], type=pa.int64()),
                "pmml_score": pa.array([0.2, 0.8], type=pa.float64()),
            },
            schema=stress_module.SCORE_ARTIFACT_SCHEMA,
        ),
        path,
    )
    artifact = ScenarioScoreArtifact("征信", path, 2, _digest(path))
    monkeypatch.setattr(
        stress_module,
        "sha256_file_cancellable",
        lambda candidate, cancellation_check=None: _digest(Path(candidate)),
    )

    def cancel() -> None:
        raise JobCancelled("cancelled during aligned load")

    with pytest.raises(JobCancelled, match="aligned load"):
        load_aligned_scenario_scores(
            artifact,
            np.array([8, 9], dtype=np.int64),
            cancellation_check=cancel,
        )


def test_cancelled_cache_copy_preserves_old_task_artifact(
    tmp_path: Path, ready_contract, monkeypatch
):
    sample_path = tmp_path / "sample.parquet"
    _sample_frame().to_parquet(sample_path, index=False)
    contract = _ready_for_sample(ready_contract, sample_path)
    baseline_path, scoring = _baseline(tmp_path, contract, sample_path)
    cache_dir = tmp_path / "cache"
    run_pmml_stress(
        contract=contract,
        config=_config(),
        sample_path=sample_path,
        baseline_score_path=baseline_path,
        scoring_result=scoring,
        scenario_dir=tmp_path / "first",
        feature_categories={"征信": ("x1",)},
        scorer=_RecordingScorer(),
        chunk_size=2,
        baseline_cache_key=scoring.cache_key,
        cache_dir=cache_dir,
    )
    key = stress_cache_key(
        baseline_cache_key=scoring.cache_key,
        category="征信",
        raw_fields=("x1",),
        chunk_size=2,
    )
    output = tmp_path / "existing.parquet"
    output.write_bytes(b"old-complete")

    class Cancelled(RuntimeError):
        pass

    def no_hardlink(source, destination):
        raise OSError("cross-device")

    def cancel_copy(source, destination, **kwargs):
        Path(destination).write_bytes(b"partial")
        raise Cancelled("stop")

    monkeypatch.setattr(stress_module.os, "link", no_hardlink)
    monkeypatch.setattr(stress_module, "copy_file_cancellable", cancel_copy)

    with pytest.raises(Cancelled, match="stop"):
        materialize_cached_stress_scenario(
            cache_dir=cache_dir,
            cache_key=key,
            category="征信",
            expected_row_ids=np.array([8, 9, 10, 11], dtype=np.int64),
            output_path=output,
        )

    assert output.read_bytes() == b"old-complete"
    assert not list(tmp_path.glob(".existing.parquet.*.staging"))
