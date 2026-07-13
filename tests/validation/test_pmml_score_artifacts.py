from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from filelock import FileLock

import marvis.validation.pmml_score_artifacts as artifacts
from marvis.validation.pmml_score_artifacts import (
    AtomicScoreWriter,
    SCORE_ARTIFACT_SCHEMA,
    cancellable_file_lock,
    copy_file_cancellable,
    pmml_scoring_cache_key,
    run_pmml_scoring,
    validate_pmml_score_artifact,
)
from marvis.validation.input_contracts import TransformationSpec
from marvis.validation.sample_schema import inspect_sample_schema


PMML_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "min_lr.pmml"


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _ready_for_sample(ready_contract, sample: Path):
    schema = inspect_sample_schema(sample)
    return replace(
        ready_contract,
        sample_schema=schema,
        material_hashes={
            **ready_contract.material_hashes,
            "sample": schema.sha256,
            "pmml": _digest(PMML_FIXTURE),
        },
    )


def _sample(tmp_path: Path, rows: int = 3) -> Path:
    path = tmp_path / "sample.parquet"
    pd.DataFrame(
        {
            "x1": np.arange(rows, dtype=float),
            "x2": np.arange(rows, dtype=float) % 2,
            "y": np.arange(rows) % 2,
            "split": ["oot"] * rows,
            "apply_month": ["2026-01"] * rows,
        }
    ).to_parquet(path, index=False, row_group_size=2)
    return path


def _staging_paths(path: Path) -> list[Path]:
    return list(path.parent.glob(f".{path.name}.*.staging"))


class _StaticScorer:
    def __init__(self, values) -> None:
        self.values = values
        self.calls = 0

    def score_chunk(self, frame: pd.DataFrame) -> pd.Series:
        self.calls += 1
        values = self.values(frame, self.calls) if callable(self.values) else self.values
        if isinstance(values, pd.Series):
            return values
        return pd.Series(values)


def test_run_pmml_scoring_real_model_writes_exact_schema_and_all_rows(
    tmp_path, ready_contract
):
    sample = _sample(tmp_path)
    contract = _ready_for_sample(ready_contract, sample)
    output = tmp_path / "pmml_scores.parquet"

    result = run_pmml_scoring(
        contract=contract,
        sample_path=sample,
        pmml_path=PMML_FIXTURE,
        score_path=output,
        chunk_size=2,
    )

    assert pq.ParquetFile(output).schema_arrow == SCORE_ARTIFACT_SCHEMA
    table = pq.read_table(output)
    assert table.column("row_id").to_pylist() == [0, 1, 2]
    assert all(math.isfinite(value) for value in table.column("pmml_score").to_pylist())
    assert result.input_row_count == result.success_count == 3
    assert result.failure_count == result.null_count == result.non_finite_count == 0
    assert result.status == "pass"
    assert result.score_artifact_sha256 == _digest(output)
    assert validate_pmml_score_artifact(result, output) == result


@pytest.mark.parametrize("destination", ["sample", "pmml"])
def test_run_rejects_score_path_that_would_overwrite_an_input(
    tmp_path, ready_contract, destination
):
    sample = _sample(tmp_path)
    contract = _ready_for_sample(ready_contract, sample)
    score_path = sample if destination == "sample" else PMML_FIXTURE
    before = score_path.read_bytes()

    with pytest.raises(ValueError, match="must differ from input materials"):
        run_pmml_scoring(
            contract=contract,
            sample_path=sample,
            pmml_path=PMML_FIXTURE,
            score_path=score_path,
            chunk_size=2,
            scorer=_StaticScorer([0.1, 0.2, 0.3]),
        )

    assert score_path.read_bytes() == before


def test_run_rehashes_current_materials_even_with_supplied_stage_evidence(
    tmp_path, ready_contract, monkeypatch
):
    sample = _sample(tmp_path)
    contract = _ready_for_sample(ready_contract, sample)
    output = tmp_path / "scores.parquet"
    seen: list[Path] = []
    original = artifacts.sha256_file_cancellable

    def record(path, cancellation_check=None, **kwargs):
        seen.append(Path(path))
        return original(path, cancellation_check, **kwargs)

    monkeypatch.setattr(artifacts, "sha256_file_cancellable", record)
    run_pmml_scoring(
        contract=contract,
        sample_path=sample,
        pmml_path=PMML_FIXTURE,
        score_path=output,
        chunk_size=2,
        scorer=_StaticScorer(lambda frame, _call: [0.2] * len(frame)),
        pmml_sha256=contract.material_hashes["pmml"],
        sample_sha256=contract.material_hashes["sample"],
    )

    assert seen[:2] == [PMML_FIXTURE, sample]
    assert len(seen) == 3
    assert seen[2].name.endswith(".staging")


def test_cache_key_includes_chunk_size_and_complete_runtime_identity(monkeypatch):
    monkeypatch.setattr(
        artifacts,
        "_runtime_identity",
        lambda: {
            "engine": "engine",
            "pypmml": "1",
            "pmml4s": "2",
            "pandas": "3",
        },
    )
    common = dict(
        pmml_sha256="p",
        sample_sha256="s",
        output_field="probability_1",
        engine_version="runtime",
        transformation_sha256="t",
    )

    assert pmml_scoring_cache_key(**common, chunk_size=10) != pmml_scoring_cache_key(
        **common, chunk_size=20
    )
    assert pmml_scoring_cache_key(**common, chunk_size=10) != pmml_scoring_cache_key(
        **{**common, "output_field": "probability_0"}, chunk_size=10
    )


def test_scoring_projects_only_model_transformation_closure(
    tmp_path, ready_contract
):
    sample = tmp_path / "transformed.parquet"
    pd.DataFrame(
        {
            "raw_x1": [1.0, 2.0],
            "x2": [0.0, 1.0],
            "control_date": ["2026-01-01", "2026-02-01"],
        }
    ).to_parquet(sample, index=False)
    contract = _ready_for_sample(ready_contract, sample)
    contract = replace(
        contract,
        transformations=(
            TransformationSpec("copy", "x1", ("raw_x1",), {}),
            TransformationSpec(
                "date_to_month",
                "control_month",
                ("control_date",),
                {"mode": "direct_string_slice"},
            ),
        ),
    )

    class RecordingScorer:
        def __init__(self) -> None:
            self.frames: list[pd.DataFrame] = []

        def score_chunk(self, frame):
            self.frames.append(frame.copy())
            return pd.Series([0.2] * len(frame))

    scorer = RecordingScorer()
    run_pmml_scoring(
        contract=contract,
        sample_path=sample,
        pmml_path=PMML_FIXTURE,
        score_path=tmp_path / "scores.parquet",
        chunk_size=1,
        scorer=scorer,
    )

    assert [frame.columns.tolist() for frame in scorer.frames] == [
        ["x1", "x2"],
        ["x1", "x2"],
    ]
    assert pd.concat(scorer.frames, ignore_index=True)["x1"].tolist() == [1.0, 2.0]


def test_zero_input_model_scores_every_row_with_an_n_by_zero_frame(
    tmp_path, ready_contract
):
    sample = _sample(tmp_path, rows=5)
    contract = _ready_for_sample(ready_contract, sample)
    contract = replace(
        contract,
        pmml_manifest=replace(
            contract.require_pmml_manifest(),
            raw_required_fields=(),
            model_features=(),
            stress_units=(),
        ),
    )
    shapes: list[tuple[int, int]] = []

    class ZeroInputScorer:
        def score_chunk(self, frame):
            shapes.append(frame.shape)
            return pd.Series([0.25] * len(frame))

    output = tmp_path / "zero-input-scores.parquet"
    result = run_pmml_scoring(
        contract=contract,
        sample_path=sample,
        pmml_path=PMML_FIXTURE,
        score_path=output,
        chunk_size=2,
        scorer=ZeroInputScorer(),
    )

    assert shapes == [(2, 0), (2, 0), (1, 0)]
    assert result.input_row_count == 5
    assert pq.read_table(output).column("row_id").to_pylist() == [0, 1, 2, 3, 4]


def test_supplied_stage_hash_must_match_confirmed_contract(
    tmp_path, ready_contract
):
    sample = _sample(tmp_path)
    contract = _ready_for_sample(ready_contract, sample)

    with pytest.raises(ValueError, match="supplied sample SHA-256 does not match"):
        run_pmml_scoring(
            contract=contract,
            sample_path=sample,
            pmml_path=PMML_FIXTURE,
            score_path=tmp_path / "scores.parquet",
            chunk_size=2,
            scorer=_StaticScorer([0.1, 0.2, 0.3]),
            sample_sha256="0" * 64,
            pmml_sha256=contract.material_hashes["pmml"],
        )


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        ([0.1, None, 0.2], "null=1"),
        ([0.1, np.nan, 0.2], "null=1"),
        ([0.1, np.inf, 0.2], "non_finite=1"),
        ([0.1, -np.inf, 0.2], "non_finite=1"),
    ],
)
def test_null_nan_or_infinite_score_is_a_hard_gate(
    tmp_path, ready_contract, invalid, message
):
    sample = _sample(tmp_path)
    contract = _ready_for_sample(ready_contract, sample)
    output = tmp_path / "scores.parquet"

    with pytest.raises(ValueError, match=message):
        run_pmml_scoring(
            contract=contract,
            sample_path=sample,
            pmml_path=PMML_FIXTURE,
            score_path=output,
            chunk_size=10,
            scorer=_StaticScorer(invalid),
        )

    assert not output.exists()
    assert _staging_paths(output) == []


@pytest.mark.parametrize("delta", [-1, 1])
def test_scorer_short_or_extra_rows_are_rejected(tmp_path, ready_contract, delta):
    sample = _sample(tmp_path)
    contract = _ready_for_sample(ready_contract, sample)
    output = tmp_path / "scores.parquet"
    scorer = _StaticScorer(
        lambda frame, _call: [0.2] * (len(frame) + delta)
    )

    with pytest.raises(ValueError, match="row count mismatch"):
        run_pmml_scoring(
            contract=contract,
            sample_path=sample,
            pmml_path=PMML_FIXTURE,
            score_path=output,
            chunk_size=10,
            scorer=scorer,
        )

    assert not output.exists()
    assert _staging_paths(output) == []


def test_writer_construction_failure_cleans_staging_and_preserves_final(
    tmp_path, monkeypatch
):
    final = tmp_path / "scores.parquet"
    final.write_bytes(b"old-complete")

    def fail_after_create(path, _schema):
        Path(path).write_bytes(b"partial")
        raise OSError("construction failed")

    monkeypatch.setattr(artifacts.pq, "ParquetWriter", fail_after_create)
    with pytest.raises(OSError, match="construction failed"):
        AtomicScoreWriter(final)

    assert final.read_bytes() == b"old-complete"
    assert _staging_paths(final) == []


@pytest.mark.parametrize("failure", ["write", "close", "hash", "replace"])
def test_writer_stage_failures_preserve_old_final_and_remove_staging(
    tmp_path, ready_contract, monkeypatch, failure
):
    sample = _sample(tmp_path)
    contract = _ready_for_sample(ready_contract, sample)
    output = tmp_path / "scores.parquet"
    output.write_bytes(b"old-complete")

    if failure == "write":
        original = artifacts.AtomicScoreWriter.write

        def fail_write(self, row_ids, scores):
            original(self, row_ids, scores)
            raise OSError("write failed")

        monkeypatch.setattr(artifacts.AtomicScoreWriter, "write", fail_write)
    elif failure == "close":
        original_close = artifacts.AtomicScoreWriter._close

        def fail_close(self):
            original_close(self)
            raise OSError("close failed")

        monkeypatch.setattr(artifacts.AtomicScoreWriter, "_close", fail_close)
    elif failure == "hash":
        original_hash = artifacts.sha256_file_cancellable

        def fail_staging_hash(path, cancellation_check=None, **kwargs):
            if Path(path).name.endswith(".staging"):
                raise OSError("hash failed")
            return original_hash(path, cancellation_check, **kwargs)

        monkeypatch.setattr(
            artifacts,
            "sha256_file_cancellable",
            fail_staging_hash,
        )
    else:
        monkeypatch.setattr(
            artifacts.os,
            "replace",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("replace failed")),
        )

    with pytest.raises(ValueError, match=f"{failure} failed"):
        run_pmml_scoring(
            contract=contract,
            sample_path=sample,
            pmml_path=PMML_FIXTURE,
            score_path=output,
            chunk_size=10,
            scorer=_StaticScorer([0.1, 0.2, 0.3]),
        )

    assert output.read_bytes() == b"old-complete"
    assert _staging_paths(output) == []


@pytest.mark.parametrize("moment", ["first", "between", "commit", "hash"])
def test_cancellation_never_publishes_or_leaves_staging(
    tmp_path, ready_contract, monkeypatch, moment
):
    sample = _sample(tmp_path)
    contract = _ready_for_sample(ready_contract, sample)
    output = tmp_path / "scores.parquet"
    scorer = _StaticScorer(lambda frame, _call: [0.2] * len(frame))

    if moment == "first":
        def callback():
            raise KeyboardInterrupt("cancelled")

        chunk_size = 2
    elif moment == "between":
        def callback():
            if scorer.calls >= 1:
                raise KeyboardInterrupt("cancelled")

        chunk_size = 2
    elif moment == "commit":
        def callback():
            if scorer.calls >= 1:
                raise KeyboardInterrupt("cancelled")

        chunk_size = 10
    else:
        def callback():
            return None

        chunk_size = 10
        original_hash = artifacts.sha256_file_cancellable

        def cancel_hash(path, cancellation_check=None, **kwargs):
            if Path(path).name.endswith(".staging"):
                raise KeyboardInterrupt("cancelled")
            return original_hash(path, cancellation_check, **kwargs)

        monkeypatch.setattr(artifacts, "sha256_file_cancellable", cancel_hash)

    with pytest.raises(KeyboardInterrupt, match="cancelled"):
        run_pmml_scoring(
            contract=contract,
            sample_path=sample,
            pmml_path=PMML_FIXTURE,
            score_path=output,
            chunk_size=chunk_size,
            scorer=scorer,
            cancellation_check=callback,
        )

    assert not output.exists()
    assert _staging_paths(output) == []


@pytest.mark.parametrize("changed", ["sample", "pmml"])
def test_source_replacement_during_scoring_is_rejected(
    tmp_path, ready_contract, changed
):
    sample = _sample(tmp_path)
    pmml = tmp_path / "model.pmml"
    pmml.write_bytes(PMML_FIXTURE.read_bytes())
    contract = replace(
        _ready_for_sample(ready_contract, sample),
        material_hashes={
            **_ready_for_sample(ready_contract, sample).material_hashes,
            "pmml": _digest(pmml),
        },
    )
    output = tmp_path / "scores.parquet"

    def mutate(frame, call):
        if call == 1:
            selected = sample if changed == "sample" else pmml
            selected.write_bytes(selected.read_bytes() + b"changed")
        return [0.2] * len(frame)

    with pytest.raises(ValueError, match=f"{changed.upper() if changed == 'pmml' else changed} file changed"):
        run_pmml_scoring(
            contract=contract,
            sample_path=sample,
            pmml_path=pmml,
            score_path=output,
            chunk_size=10,
            scorer=_StaticScorer(mutate),
        )

    assert not output.exists()
    assert _staging_paths(output) == []


@pytest.mark.parametrize("changed", ["sample", "pmml"])
def test_source_replacement_while_hashing_staging_is_rejected_before_publish(
    tmp_path, ready_contract, monkeypatch, changed
):
    sample = _sample(tmp_path)
    pmml = tmp_path / "model.pmml"
    pmml.write_bytes(PMML_FIXTURE.read_bytes())
    contract = _ready_for_sample(ready_contract, sample)
    contract = replace(
        contract,
        material_hashes={
            **contract.material_hashes,
            "pmml": _digest(pmml),
        },
    )
    output = tmp_path / "scores.parquet"
    original_hash = artifacts.sha256_file_cancellable

    def mutate_during_staging_hash(path, cancellation_check=None, **kwargs):
        if Path(path).name.endswith(".staging"):
            selected = sample if changed == "sample" else pmml
            selected.write_bytes(selected.read_bytes() + b"changed-during-hash")
        return original_hash(path, cancellation_check, **kwargs)

    monkeypatch.setattr(
        artifacts,
        "sha256_file_cancellable",
        mutate_during_staging_hash,
    )

    with pytest.raises(
        ValueError,
        match=f"{changed.upper() if changed == 'pmml' else changed} file changed",
    ):
        run_pmml_scoring(
            contract=contract,
            sample_path=sample,
            pmml_path=pmml,
            score_path=output,
            chunk_size=10,
            scorer=_StaticScorer([0.1, 0.2, 0.3]),
        )

    assert not output.exists()
    assert _staging_paths(output) == []


@pytest.mark.parametrize("changed", ["sample", "pmml"])
def test_material_changed_after_confirmation_but_stable_during_run_is_rejected(
    tmp_path, ready_contract, changed
):
    sample = _sample(tmp_path)
    pmml = tmp_path / "model.pmml"
    pmml.write_bytes(PMML_FIXTURE.read_bytes())
    contract = _ready_for_sample(ready_contract, sample)
    contract = replace(
        contract,
        material_hashes={
            **contract.material_hashes,
            "pmml": _digest(pmml),
        },
    )
    selected = sample if changed == "sample" else pmml
    selected.write_bytes(selected.read_bytes() + b"changed-before-run")
    output = tmp_path / "scores.parquet"

    with pytest.raises(ValueError, match=f"current {changed} file does not match"):
        run_pmml_scoring(
            contract=contract,
            sample_path=sample,
            pmml_path=pmml,
            score_path=output,
            chunk_size=10,
            scorer=_StaticScorer([0.1, 0.2, 0.3]),
        )

    assert not output.exists()
    assert _staging_paths(output) == []


@pytest.mark.parametrize("changed", ["sample", "pmml"])
def test_supplied_stage_hash_cannot_hide_material_replaced_before_run(
    tmp_path, ready_contract, changed
):
    sample = _sample(tmp_path)
    pmml = tmp_path / "model.pmml"
    pmml.write_bytes(PMML_FIXTURE.read_bytes())
    contract = _ready_for_sample(ready_contract, sample)
    contract = replace(
        contract,
        material_hashes={
            **contract.material_hashes,
            "pmml": _digest(pmml),
        },
    )
    selected = sample if changed == "sample" else pmml
    selected.write_bytes(selected.read_bytes() + b"changed-before-run")
    output = tmp_path / "scores.parquet"

    with pytest.raises(ValueError, match=f"current {changed} file does not match"):
        run_pmml_scoring(
            contract=contract,
            sample_path=sample,
            pmml_path=pmml,
            score_path=output,
            chunk_size=10,
            scorer=_StaticScorer([0.1, 0.2, 0.3]),
            sample_sha256=contract.material_hashes["sample"],
            pmml_sha256=contract.material_hashes["pmml"],
        )

    assert not output.exists()
    assert _staging_paths(output) == []


def _passing_result(tmp_path, ready_contract):
    sample = _sample(tmp_path)
    contract = _ready_for_sample(ready_contract, sample)
    output = tmp_path / "scores.parquet"
    result = run_pmml_scoring(
        contract=contract,
        sample_path=sample,
        pmml_path=PMML_FIXTURE,
        score_path=output,
        chunk_size=2,
        scorer=_StaticScorer(lambda frame, _call: [0.2] * len(frame)),
    )
    return result, output


def _replace_sidecar(output: Path, *, row_ids, scores) -> str:
    pq.write_table(
        pa.table(
            {
                "row_id": pa.array(row_ids, type=pa.int64()),
                "pmml_score": pa.array(scores, type=pa.float64()),
            },
            schema=SCORE_ARTIFACT_SCHEMA,
        ),
        output,
    )
    return _digest(output)


def test_verifier_rejects_hash_tampering(tmp_path, ready_contract):
    result, output = _passing_result(tmp_path, ready_contract)
    with output.open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(ValueError, match="hash mismatch"):
        validate_pmml_score_artifact(result, output)


@pytest.mark.parametrize("mode", ["row_id", "non_finite", "schema", "count"])
def test_verifier_rejects_semantic_sidecar_tampering(
    tmp_path, ready_contract, mode
):
    result, output = _passing_result(tmp_path, ready_contract)
    if mode == "row_id":
        digest = _replace_sidecar(output, row_ids=[0, 2, 3], scores=[0.1] * 3)
        expected = "row_id"
    elif mode == "non_finite":
        digest = _replace_sidecar(output, row_ids=[0, 1, 2], scores=[0.1, np.inf, 0.2])
        expected = "non-finite"
    elif mode == "schema":
        pq.write_table(pa.table({"row_id": [0, 1, 2], "pmml_score": [1, 2, 3]}), output)
        digest = _digest(output)
        expected = "schema mismatch"
    else:
        digest = _replace_sidecar(output, row_ids=[0, 1], scores=[0.1, 0.2])
        expected = "row count mismatch"
    tampered_result = replace(result, score_artifact_sha256=digest)

    with pytest.raises(ValueError, match=expected):
        validate_pmml_score_artifact(tampered_result, output, batch_size=1)


def test_verifier_checks_result_before_reading_artifact(tmp_path, ready_contract, monkeypatch):
    result, output = _passing_result(tmp_path, ready_contract)
    invalid = replace(result, null_count=1, failure_count=1, success_count=2)
    monkeypatch.setattr(
        artifacts,
        "sha256_file_cancellable",
        lambda *_args, **_kwargs: pytest.fail("invalid result must fail before I/O"),
    )

    with pytest.raises(ValueError, match="passing PMML scoring evidence"):
        validate_pmml_score_artifact(invalid, output)


def test_copy_cancellation_removes_partial_destination(tmp_path):
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"a" * 32)
    checks = 0

    def cancel() -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise KeyboardInterrupt("cancelled")

    with pytest.raises(KeyboardInterrupt, match="cancelled"):
        copy_file_cancellable(
            source,
            destination,
            cancellation_check=cancel,
            block_size=8,
        )

    assert not destination.exists()


def test_cancellable_file_lock_stops_waiting_and_keeps_owner_lock(tmp_path):
    lock_path = tmp_path / "cache.lock"
    owner = FileLock(str(lock_path))
    owner.acquire()
    checks = 0

    def cancel() -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise KeyboardInterrupt("cancelled")

    try:
        with pytest.raises(KeyboardInterrupt, match="cancelled"):
            with cancellable_file_lock(
                lock_path,
                cancellation_check=cancel,
                poll_seconds=0.01,
            ):
                pytest.fail("contended lock must not be entered")
        assert owner.is_locked
    finally:
        owner.release()


def test_cancellable_file_lock_releases_after_body_failure(tmp_path):
    lock_path = tmp_path / "cache.lock"

    with pytest.raises(RuntimeError, match="body failed"):
        with cancellable_file_lock(lock_path, poll_seconds=0.01):
            raise RuntimeError("body failed")

    replacement = FileLock(str(lock_path))
    replacement.acquire(timeout=0)
    replacement.release()


def test_external_scoring_error_message_is_bounded(tmp_path, ready_contract):
    sample = _sample(tmp_path)
    contract = _ready_for_sample(ready_contract, sample)

    class ExplodingScorer:
        def score_chunk(self, _frame):
            raise RuntimeError("x" * 20_000)

    with pytest.raises(ValueError) as captured:
        run_pmml_scoring(
            contract=contract,
            sample_path=sample,
            pmml_path=PMML_FIXTURE,
            score_path=tmp_path / "scores.parquet",
            chunk_size=10,
            scorer=ExplodingScorer(),
        )

    assert len(str(captured.value)) <= artifacts.MAX_SCORING_ERROR_CHARS
