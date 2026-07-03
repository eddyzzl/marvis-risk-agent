import pandas as pd

from marvis.data.backend import DataBackend
from marvis.data.contracts import ColumnFingerprint, ColumnProfile, Dataset
from marvis.data.profiler import profile_dataset
from marvis.packs.modeling.readiness import (
    QualityIssue,
    check_data_quality,
    modeling_readiness,
)


def _fingerprint() -> ColumnFingerprint:
    return ColumnFingerprint(
        value_kind="numeric",
        length_mode=None,
        regex_pattern=None,
        is_hashed=False,
        hash_type=None,
        hex_case=None,
        date_format=None,
    )


def _profile(
    name: str,
    *,
    null_rate: float = 0.0,
    cardinality: int = 2,
    semantic_role: str = "numeric",
) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        dtype="float64",
        semantic_role=semantic_role,
        fingerprint=_fingerprint(),
        null_rate=null_rate,
        cardinality=cardinality,
        sample_values=(0, 1),
    )


def _dataset(
    *,
    path: str,
    row_count: int,
    columns: tuple[ColumnProfile, ...],
    target_col: str | None = "y",
) -> Dataset:
    return Dataset(
        id="dataset-1",
        task_id="task-1",
        role="sample",
        source_path=path,
        format="parquet",
        sheet=None,
        row_count=row_count,
        columns=columns,
        has_target=target_col is not None,
        target_col=target_col,
        created_at="2026-06-19T00:00:00Z",
    )


def _profiled_dataset(tmp_path, frame: pd.DataFrame, *, target_col: str = "y") -> tuple[DataBackend, Dataset, object]:
    path = tmp_path / "sample.parquet"
    frame.to_parquet(path, index=False)
    backend = DataBackend(tmp_path)
    profiles = tuple(profile_dataset(backend, path, seed=0))
    dataset = _dataset(
        path=path.name,
        row_count=len(frame),
        columns=profiles,
        target_col=target_col,
    )
    return backend, dataset, path


def test_check_data_quality_detects_blockers_warnings_duplicates_and_leakage(tmp_path):
    frame = pd.DataFrame({
        "y": [0, 1, 0, 1, 0, 1],
        "mostly_missing": [None, None, None, None, None, 1],
        "constant": [1, 1, 1, 1, 1, 1],
        "dup_a": [1, 2, 3, 4, 5, 6],
        "dup_b": [1, 2, 3, 4, 5, 6],
        "leak": [0, 1, 0, 1, 0, 1],
        "category": ["a", "b", "c", "d", "e", "f"],
    })
    path = tmp_path / "quality.parquet"
    frame.to_parquet(path, index=False)
    backend = DataBackend(tmp_path)
    dataset = _dataset(
        path=path.name,
        row_count=len(frame),
        columns=(
            _profile("y", cardinality=2),
            _profile("mostly_missing", null_rate=0.96, cardinality=2),
            _profile("constant", cardinality=1),
            _profile("dup_a", cardinality=6),
            _profile("dup_b", cardinality=6),
            _profile("leak", cardinality=2),
            _profile("category", cardinality=1001, semantic_role="categorical"),
        ),
    )

    issues = check_data_quality(backend, dataset, path, target_col="y")
    by_kind = {(issue.column, issue.kind): issue for issue in issues}

    assert by_kind[("mostly_missing", "missing")].severity == "block"
    assert by_kind[("constant", "constant")].severity == "block"
    assert by_kind[("category", "high_cardinality")].severity == "warn"
    assert by_kind[("dup_b", "duplicate_col")].detail == "duplicates dup_a"
    assert by_kind[("leak", "leakage_suspect")].severity == "block"


def test_modeling_readiness_blocks_non_binary_target_and_small_sample(tmp_path):
    frame = pd.DataFrame({
        "y": [0, 1, 2, 0, 1, 2],
        "x": [10, 11, 12, 13, 14, 15],
        "split": ["train", "train", "test", "test", "oot", "oot"],
        "decision": ["approved", "rejected", "approved", "rejected", "approved", "rejected"],
    })
    backend, dataset, path = _profiled_dataset(tmp_path, frame)

    result = modeling_readiness(backend, dataset, path, target_col="y", split_col="split")

    assert result["ready"] is False
    assert "target must be binary 0/1" in result["blockers"]
    assert "too few samples (<1000)" in result["blockers"]


def test_modeling_readiness_passes_for_binary_sample_with_required_splits(tmp_path):
    rows = 1200
    frame = pd.DataFrame({
        "y": [1 if i % 10 == 0 else 0 for i in range(rows)],
        "x1": [((i * 37) % 101) / 100 for i in range(rows)],
        "x2": [i % 17 for i in range(rows)],
        "split": ["train"] * 700 + ["test"] * 300 + ["oot"] * 200,
        "decision": ["approved" if i % 2 == 0 else "rejected" for i in range(rows)],
    })
    backend, dataset, path = _profiled_dataset(tmp_path, frame)

    result = modeling_readiness(backend, dataset, path, target_col="y", split_col="split")

    assert result["ready"] is True
    assert result["blockers"] == []
    assert result["stats"]["rows"] == rows
    assert result["stats"]["bad_rate"] == 0.1
    assert not any("拒绝推断未实现" in warning for warning in result["warnings"])


def test_modeling_readiness_warns_for_accept_only_samples(tmp_path):
    rows = 1200
    frame = pd.DataFrame({
        "y": [1 if i % 20 == 0 else 0 for i in range(rows)],
        "x1": [((i * 11) % 97) / 100 for i in range(rows)],
        "split": ["train"] * 800 + ["test"] * 400,
        "decision": ["approved"] * rows,
    })
    backend, dataset, path = _profiled_dataset(tmp_path, frame)

    result = modeling_readiness(backend, dataset, path, target_col="y", split_col="split")

    assert result["ready"] is True
    assert any("建议使用 reject_inference 工具" in warning for warning in result["warnings"])


def test_quality_issue_contract_round_trips():
    issue = QualityIssue(
        column="score",
        kind="leakage_suspect",
        detail="abs corr 0.9900 with target y",
        severity="block",
    )

    assert issue.column == "score"
    assert issue.kind == "leakage_suspect"
    assert issue.severity == "block"
