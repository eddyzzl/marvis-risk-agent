from __future__ import annotations

import json
import os
from pathlib import Path
import threading

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import marvis.packs.strategy.automatic_tree_apply as automatic_tree_apply
from marvis.feature.weighted_rule_tree import build_weighted_rule_tree
from marvis.files import sha256_file
from marvis.packs.strategy.automatic_tree_apply import (
    AUTOMATIC_TREE_APPLY_PRODUCER_VERSION,
    AUTOMATIC_TREE_APPLY_SCHEMA_VERSION,
    AUTOMATIC_TREE_APPLY_WRITER_CONTRACT,
    AutomaticTreeApplyBudgetError,
    AutomaticTreeApplyError,
    apply_automatic_tree_to_parquet,
)
from marvis.packs.strategy.automatic_tree_asset import build_automatic_tree_asset


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def _tree() -> dict:
    training = pd.DataFrame(
        {
            "x": np.arange(8, dtype=np.float64),
            "unused": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "bad": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    tree = build_weighted_rule_tree(
        training,
        feature_cols=["x", "unused"],
        target_col="bad",
        max_depth=1,
        min_leaf_count=1,
    )
    assert {node.get("feature") for node in tree["tree"]["nodes"]} - {None} == {"x"}
    return tree


def _asset(tree: dict | None = None) -> dict:
    return build_automatic_tree_asset(
        tree or _tree(),
        task_id="task-automatic-apply",
        dataset_id="dataset-automatic-apply",
        dataset_content_hash=HASH_A,
        workspace_revision=3,
        workspace_generation=4,
        semantic_mapping_hash=HASH_B,
        registry_metadata_hash=HASH_C,
        sample_context_hash=HASH_D,
        source_refs=["dataset:dataset-automatic-apply"],
    )


def _write_source(
    path: Path,
    columns: dict[str, pa.Array | list],
    *,
    metadata: dict[bytes, bytes] | None = None,
) -> Path:
    arrays = {
        name: value if isinstance(value, pa.Array) else pa.array(value)
        for name, value in columns.items()
    }
    table = pa.table(arrays)
    if metadata is not None:
        table = table.replace_schema_metadata(metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    return path


def _open_fd_count() -> int | None:
    for directory in (Path("/proc/self/fd"), Path("/dev/fd")):
        if directory.is_dir():
            return len(os.listdir(directory))
    return None


def _directory_entries(directory: Path) -> set[str]:
    return {entry.name for entry in directory.iterdir()}


def _apply(
    tmp_path: Path,
    columns: dict[str, pa.Array | list],
    *,
    tree_or_asset: dict | None = None,
    leaf_id_column: str = "tree_leaf_id",
    rule_id_column: str = "tree_rule_id",
):
    source = _write_source(tmp_path / "source.parquet", columns)
    output = tmp_path / "output.parquet"
    result = apply_automatic_tree_to_parquet(
        tree_or_asset or _asset(),
        source,
        output,
        leaf_id_column=leaf_id_column,
        rule_id_column=rule_id_column,
    )
    return source, output, result


def test_full_tree_apply_routes_missing_and_boundary_rows_and_preserves_source(
    tmp_path: Path,
) -> None:
    tree = _tree()
    asset = _asset(tree)
    root = tree["tree"]["nodes"][0]
    threshold = root["threshold"]
    missing_leaf = root[f"{root['missing_child']}_child_id"]
    source = _write_source(
        tmp_path / "source.parquet",
        {
            "ordinal": [30, 10, 20, 40],
            "x": pa.array(
                [threshold, np.nextafter(threshold, np.inf), None, -999.0],
                type=pa.float64(),
            ),
            "bad": pa.array([None, 1, None, 0], type=pa.int8()),
            "unused": pa.array([0.0, 1.0, None, 1.0], type=pa.float64()),
            "keep_text": ["third", "first", "second", "fourth"],
        },
        metadata={b"source-contract": b"preserve-me"},
    )
    output = tmp_path / "out" / "result.parquet"

    result = apply_automatic_tree_to_parquet(
        asset,
        source,
        output,
        leaf_id_column="tree_leaf_id",
        rule_id_column="tree_rule_id",
    )

    derived = pq.read_table(output)
    source_table = pq.read_table(source)
    assert derived.column_names == [
        *source_table.column_names,
        "tree_leaf_id",
        "tree_rule_id",
    ]
    assert derived.num_rows == source_table.num_rows == result.source_row_count == 4
    assert derived.column("ordinal").to_pylist() == [30, 10, 20, 40]
    assert derived.column("keep_text").to_pylist() == [
        "third",
        "first",
        "second",
        "fourth",
    ]
    assert derived.column("bad").to_pylist() == [None, 1, None, 0]
    assert derived.schema.metadata == source_table.schema.metadata
    assert list(derived.schema)[: len(source_table.schema)] == list(source_table.schema)
    assert derived.schema.field("tree_leaf_id") == pa.field(
        "tree_leaf_id", pa.string(), nullable=False
    )
    assert derived.schema.field("tree_rule_id") == pa.field(
        "tree_rule_id", pa.string(), nullable=False
    )

    leaf_ids = derived.column("tree_leaf_id").to_pylist()
    assert leaf_ids[0] == root["left_child_id"]
    assert leaf_ids[1] == root["right_child_id"]
    assert leaf_ids[2] == missing_leaf
    rule_by_leaf = {rule["leaf_id"]: rule["rule_id"] for rule in tree["rules"]}
    assert derived.column("tree_rule_id").to_pylist() == [
        rule_by_leaf[leaf_id] for leaf_id in leaf_ids
    ]

    expected_counts = {leaf_id: leaf_ids.count(leaf_id) for leaf_id in rule_by_leaf}
    assert result.leaf_distribution == tuple(
        {
            "leaf_id": rule["leaf_id"],
            "rule_id": rule["rule_id"],
            "row_count": expected_counts[rule["leaf_id"]],
        }
        for rule in tree["rules"]
    )


@pytest.mark.parametrize(
    ("leaf_name", "rule_name", "match"),
    [
        ("leaf id", "rule_id", "ASCII"),
        ("1leaf", "rule_id", "cannot start"),
        ("叶子", "rule_id", "ASCII"),
        ("a" * 65, "rule_id", "64"),
        ("Result", "result", "case-insensitively unique"),
        ("ORDINAL", "rule_id", "already exist"),
    ],
)
def test_output_names_are_strict_and_never_overwrite_source(
    tmp_path: Path,
    leaf_name: str,
    rule_name: str,
    match: str,
) -> None:
    source = _write_source(
        tmp_path / "source.parquet",
        {"ordinal": [1], "x": [1.0], "unused": [0.0]},
    )

    with pytest.raises(AutomaticTreeApplyError, match=match):
        apply_automatic_tree_to_parquet(
            _asset(),
            source,
            tmp_path / "output.parquet",
            leaf_id_column=leaf_name,
            rule_id_column=rule_name,
        )

    assert not (tmp_path / "output.parquet").exists()


def test_all_selected_training_features_are_required_exact_case_even_if_unused(
    tmp_path: Path,
) -> None:
    source = _write_source(
        tmp_path / "source.parquet",
        {"x": [0.0, 9.0], "Unused": [0.0, 1.0]},
    )

    with pytest.raises(AutomaticTreeApplyError, match="unused"):
        apply_automatic_tree_to_parquet(
            _asset(),
            source,
            tmp_path / "output.parquet",
            leaf_id_column="leaf_id",
            rule_id_column="rule_id",
        )


@pytest.mark.parametrize(
    ("feature", "values", "match"),
    [
        ("unused", pa.array([0, 2**53 + 1], type=pa.int64()), "exact DOUBLE"),
        ("x", pa.array([0.0, np.inf], type=pa.float64()), "finite or missing"),
        ("unused", pa.array(["0", "1"], type=pa.string()), "integer/float physical"),
    ],
)
def test_full_delivery_domain_rejects_lossy_infinite_and_nonnumeric_features(
    tmp_path: Path,
    feature: str,
    values: pa.Array,
    match: str,
) -> None:
    columns: dict[str, pa.Array | list] = {
        "x": pa.array([0.0, 1.0], type=pa.float64()),
        "unused": pa.array([0.0, 1.0], type=pa.float64()),
    }
    columns[feature] = values
    source = _write_source(tmp_path / "source.parquet", columns)

    with pytest.raises(AutomaticTreeApplyError, match=match):
        apply_automatic_tree_to_parquet(
            _asset(),
            source,
            tmp_path / "output.parquet",
            leaf_id_column="leaf_id",
            rule_id_column="rule_id",
        )

    assert not (tmp_path / "output.parquet").exists()


def test_hard_row_and_source_column_budgets_have_no_caller_override(
    tmp_path: Path,
) -> None:
    over_rows = 1_000_001
    row_source = _write_source(
        tmp_path / "too_many_rows.parquet",
        {
            "x": pa.array(np.zeros(over_rows, dtype=np.float64)),
            "unused": pa.array(np.zeros(over_rows, dtype=np.float64)),
        },
    )
    with pytest.raises(AutomaticTreeApplyBudgetError) as row_error:
        apply_automatic_tree_to_parquet(
            _asset(),
            row_source,
            tmp_path / "rows-output.parquet",
            leaf_id_column="leaf_id",
            rule_id_column="rule_id",
        )
    assert row_error.value.to_detail() == {
        "kind": "automatic_tree_apply_budget_exceeded",
        "dimension": "source_rows",
        "actual": over_rows,
        "limit": 1_000_000,
    }

    wide_columns: dict[str, list[float]] = {"x": [0.0], "unused": [1.0]}
    wide_columns.update({f"column_{index}": [float(index)] for index in range(499)})
    column_source = _write_source(tmp_path / "too_many_columns.parquet", wide_columns)
    with pytest.raises(AutomaticTreeApplyBudgetError) as column_error:
        apply_automatic_tree_to_parquet(
            _asset(),
            column_source,
            tmp_path / "columns-output.parquet",
            leaf_id_column="leaf_id",
            rule_id_column="rule_id",
        )
    assert column_error.value.dimension == "source_columns"
    assert column_error.value.actual == 501
    assert column_error.value.limit == 500


def test_decoded_batch_cap_is_hard_and_failure_leaves_no_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source(
        tmp_path / "source.parquet",
        {
            "x": [0.0, 1.0],
            "unused": [0.0, 1.0],
            "large_non_feature": ["bounded-writeback", "still-bounded"],
        },
    )
    output = tmp_path / "output.parquet"
    monkeypatch.setattr(automatic_tree_apply, "_MAX_DECODED_BATCH_BYTES", 1)

    with pytest.raises(AutomaticTreeApplyBudgetError) as exc_info:
        apply_automatic_tree_to_parquet(
            _asset(),
            source,
            output,
            leaf_id_column="leaf_id",
            rule_id_column="rule_id",
        )

    assert exc_info.value.dimension == "decoded_batch_bytes"
    assert exc_info.value.limit == 1
    assert not output.exists()


def test_result_and_parquet_bytes_are_deterministic_across_output_directories(
    tmp_path: Path,
) -> None:
    tree = _tree()
    source = _write_source(
        tmp_path / "source.parquet",
        {
            "ordinal": list(range(16_401)),
            "x": pa.array(np.arange(16_401) % 8, type=pa.int64()),
            "unused": pa.array(np.arange(16_401) % 2, type=pa.int64()),
            "bad": pa.array([None] * 16_401, type=pa.int8()),
        },
    )
    first_path = tmp_path / "temp-a" / "applied.parquet"
    second_path = tmp_path / "temp-b" / "applied.parquet"

    first = apply_automatic_tree_to_parquet(
        tree,
        source,
        first_path,
        leaf_id_column="leaf_id",
        rule_id_column="rule_id",
    )
    second = apply_automatic_tree_to_parquet(
        tree,
        source,
        second_path,
        leaf_id_column="leaf_id",
        rule_id_column="rule_id",
    )

    assert first_path.read_bytes() == second_path.read_bytes()
    assert first.to_dict() == second.to_dict()
    assert (
        first.output_content_hash
        == second.output_content_hash
        == sha256_file(first_path)
    )
    assert first.source_content_hash == sha256_file(source)
    assert first.asset_id is first.asset_hash is None
    assert first.writer_contract == {
        "contract": AUTOMATIC_TREE_APPLY_WRITER_CONTRACT,
        "engine": "pyarrow.parquet",
        "engine_version": pa.__version__,
        "threads": 1,
        "preserve_insertion_order": True,
        "batch_rows": 8192,
        "max_decoded_batch_bytes": 256 * 1024 * 1024,
        "row_group_rows": 8192,
        "write_batch_rows": 1024,
        "parquet_version": "2.6",
        "data_page_version": "1.0",
        "compression": "zstd",
        "compression_level": 3,
        "dictionary_encoding": True,
        "dictionary_page_bytes": 1_048_576,
        "write_statistics": True,
        "byte_stream_split": False,
        "use_deprecated_int96_timestamps": False,
        "use_compliant_nested_type": True,
        "store_arrow_schema": True,
        "write_page_index": False,
        "write_page_checksum": False,
        "source_schema_metadata": "preserved",
        "appended_id_type": "utf8_non_null",
    }
    parquet_metadata = pq.ParquetFile(first_path).metadata
    assert [
        parquet_metadata.row_group(index).num_rows
        for index in range(parquet_metadata.num_row_groups)
    ] == [8192, 8192, 17]
    assert {
        parquet_metadata.row_group(row_group).column(column).compression
        for row_group in range(parquet_metadata.num_row_groups)
        for column in range(parquet_metadata.num_columns)
    } == {"ZSTD"}
    assert first.schema_version == AUTOMATIC_TREE_APPLY_SCHEMA_VERSION
    assert first.producer_version == AUTOMATIC_TREE_APPLY_PRODUCER_VERSION
    assert len(first.result_hash) == 64
    assert first.result_id == f"automatic-tree-apply-{first.result_hash[:32]}"
    serialized = json.dumps(first.to_dict(), ensure_ascii=False, sort_keys=True)
    assert str(tmp_path) not in serialized
    forbidden_keys = {"action", "actions", "adoption", "pool"}
    assert all(f'"{key}"' not in serialized.lower() for key in forbidden_keys)


def test_asset_identity_and_strict_canonical_facts_are_projected(
    tmp_path: Path,
) -> None:
    tree = _tree()
    asset = _asset(tree)
    _source, _output, result = _apply(
        tmp_path,
        {"x": [0.0, 9.0], "unused": [0.0, 1.0]},
        tree_or_asset=asset,
    )

    assert result.tree_result_hash == tree["result_hash"]
    assert result.asset_id == asset["asset_id"]
    assert result.asset_hash == asset["asset_hash"]
    assert [item["rule_id"] for item in result.leaf_distribution] == [
        rule["rule_id"] for rule in tree["rules"]
    ]
    result.output_columns["leaf_id"] = "caller_mutation"
    with pytest.raises(AutomaticTreeApplyError, match="mutated"):
        result.to_dict()

    tampered = _asset(tree)
    tampered["tree_result"]["rules"][0]["rule_id"] = "caller-rule"
    with pytest.raises(AutomaticTreeApplyError, match="strict validation"):
        apply_automatic_tree_to_parquet(
            tampered,
            tmp_path / "source.parquet",
            tmp_path / "tampered.parquet",
            leaf_id_column="leaf_id",
            rule_id_column="rule_id",
        )


def test_same_source_and_preexisting_outputs_are_rejected_without_overwrite(
    tmp_path: Path,
) -> None:
    source = _write_source(
        tmp_path / "source.parquet",
        {"x": [0.0], "unused": [1.0]},
    )
    before = source.read_bytes()
    with pytest.raises(AutomaticTreeApplyError, match="different paths"):
        apply_automatic_tree_to_parquet(
            _asset(),
            source,
            source,
            leaf_id_column="leaf_id",
            rule_id_column="rule_id",
        )
    assert source.read_bytes() == before

    output = tmp_path / "existing.parquet"
    output.write_bytes(b"do not overwrite")
    with pytest.raises(AutomaticTreeApplyError, match="must not already exist"):
        apply_automatic_tree_to_parquet(
            _asset(),
            source,
            output,
            leaf_id_column="leaf_id",
            rule_id_column="rule_id",
        )
    assert output.read_bytes() == b"do not overwrite"


def test_source_symlink_is_rejected_before_snapshot(tmp_path: Path) -> None:
    real_source = _write_source(
        tmp_path / "real-source.parquet",
        {"x": [0.0], "unused": [1.0]},
    )
    source = tmp_path / "source.parquet"
    try:
        source.symlink_to(real_source)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")

    with pytest.raises(AutomaticTreeApplyError, match="symlink"):
        apply_automatic_tree_to_parquet(
            _asset(),
            source,
            tmp_path / "output.parquet",
            leaf_id_column="leaf_id",
            rule_id_column="rule_id",
        )

    assert not (tmp_path / "output.parquet").exists()


def test_restored_caller_path_aba_is_rejected_and_output_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source(
        tmp_path / "source.parquet",
        {"x": [0.0, 1.0], "unused": [0.0, 1.0]},
    )
    replacement = _write_source(
        tmp_path / "replacement.parquet",
        {"x": [8.0, 9.0], "unused": [1.0, 0.0]},
    )
    parked_source = tmp_path / "parked-source.parquet"
    output = tmp_path / "output.parquet"
    attack_ready = threading.Event()
    attack_done = threading.Event()
    attack_errors: list[BaseException] = []
    entries_before = _directory_entries(tmp_path)
    fd_count_before = _open_fd_count()
    original_verify = automatic_tree_apply._SourceSnapshot.verify_unchanged

    def synchronized_verify(snapshot) -> None:
        attack_ready.set()
        assert attack_done.wait(timeout=5)
        original_verify(snapshot)

    def replace_then_restore() -> None:
        if not attack_ready.wait(timeout=5):
            attack_errors.append(
                TimeoutError("kernel never reached source verification")
            )
            attack_done.set()
            return
        try:
            os.replace(source, parked_source)
            os.replace(replacement, source)
            os.replace(source, replacement)
            os.replace(parked_source, source)
        except BaseException as exc:  # pragma: no cover - platform-specific denial
            attack_errors.append(exc)
        finally:
            attack_done.set()

    monkeypatch.setattr(
        automatic_tree_apply._SourceSnapshot,
        "verify_unchanged",
        synchronized_verify,
    )
    attacker = threading.Thread(target=replace_then_restore)
    attacker.start()
    raised: AutomaticTreeApplyError | None = None
    try:
        apply_automatic_tree_to_parquet(
            _asset(),
            source,
            output,
            leaf_id_column="leaf_id",
            rule_id_column="rule_id",
        )
    except AutomaticTreeApplyError as exc:
        raised = exc
    finally:
        attacker.join(timeout=5)

    assert _directory_entries(tmp_path) == entries_before
    if fd_count_before is not None:
        assert _open_fd_count() == fd_count_before
    if attack_errors:
        output.unlink(missing_ok=True)
        pytest.skip(f"platform denied deterministic path ABA: {attack_errors[0]}")
    assert raised is not None
    assert "changed or was replaced" in str(raised)
    assert not output.exists()


def test_descriptor_snapshot_leaves_no_path_or_fd_leak_on_success(
    tmp_path: Path,
) -> None:
    source = _write_source(
        tmp_path / "source.parquet",
        {"x": [0.0, 1.0], "unused": [0.0, 1.0]},
    )
    output = tmp_path / "output.parquet"
    entries_before = _directory_entries(tmp_path)
    fd_count_before = _open_fd_count()

    apply_automatic_tree_to_parquet(
        _asset(),
        source,
        output,
        leaf_id_column="leaf_id",
        rule_id_column="rule_id",
    )

    assert _directory_entries(tmp_path) == entries_before | {output.name}
    if fd_count_before is not None:
        assert _open_fd_count() == fd_count_before


def test_descriptor_snapshot_leaves_no_path_or_fd_leak_on_schema_failure(
    tmp_path: Path,
) -> None:
    source = _write_source(
        tmp_path / "source.parquet",
        {"x": [0.0, 1.0]},
    )
    entries_before = _directory_entries(tmp_path)
    fd_count_before = _open_fd_count()

    with pytest.raises(AutomaticTreeApplyError, match="unused"):
        apply_automatic_tree_to_parquet(
            _asset(),
            source,
            tmp_path / "output.parquet",
            leaf_id_column="leaf_id",
            rule_id_column="rule_id",
        )

    assert _directory_entries(tmp_path) == entries_before
    if fd_count_before is not None:
        assert _open_fd_count() == fd_count_before


def test_descriptor_snapshot_leaves_no_path_or_fd_leak_on_execution_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source(
        tmp_path / "source.parquet",
        {"x": [0.0, 1.0], "unused": [0.0, 1.0]},
    )
    entries_before = _directory_entries(tmp_path)
    fd_count_before = _open_fd_count()

    def injected_failure(*_args, **_kwargs):
        raise RuntimeError("injected descriptor snapshot failure")

    monkeypatch.setattr(
        automatic_tree_apply,
        "_canonical_leaf_ids",
        injected_failure,
    )

    with pytest.raises(RuntimeError, match="injected descriptor snapshot failure"):
        apply_automatic_tree_to_parquet(
            _asset(),
            source,
            tmp_path / "output.parquet",
            leaf_id_column="leaf_id",
            rule_id_column="rule_id",
        )

    assert _directory_entries(tmp_path) == entries_before
    if fd_count_before is not None:
        assert _open_fd_count() == fd_count_before
