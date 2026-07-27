"""JOIN setup: anchor/feature role proposal over a task's registered datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from marvis.agent.join_setup import (
    JoinSetupError,
    build_join_proposal,
    discover_join_inputs,
    propose_roles,
)
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, init_db
from marvis.settings import build_settings


def _registry(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    repo = DatasetRepository(settings.db_path)
    backend = DataBackend(settings.datasets_dir)
    return DatasetRegistry(repo, backend, settings.datasets_dir)


def _register_csv(registry, tmp_path, name, frame, *, role):
    path = tmp_path / f"{name}.csv"
    frame.to_csv(path, index=False)
    return registry.register_from_upload("task-1", path, role=role)


def test_discover_picks_target_carrying_dataset_as_anchor(tmp_path):
    registry = _registry(tmp_path)
    # feature table (no target) registered first, but must NOT become the anchor
    feature = _register_csv(registry, tmp_path, "feat", pd.DataFrame({"mobile": ["a", "b"], "bal": [1, 2]}), role="feature")
    sample = _register_csv(
        registry,
        tmp_path,
        "sample",
        pd.DataFrame({"mobile": ["a", "b"], "bad_flag": [0, 1]}),
        role="sample",
    )

    anchor_id, feature_ids = discover_join_inputs(registry, "task-1", source_dir=None)

    assert anchor_id == sample.id  # sample carries the label -> anchor
    assert feature_ids == [feature.id]


def test_discover_falls_back_to_row_count_when_no_target(tmp_path):
    registry = _registry(tmp_path)
    big = _register_csv(
        registry, tmp_path, "big",
        pd.DataFrame({"acct": [10, 11, 12, 13, 14], "amt": [100, 200, 300, 400, 500]}), role="feature")
    small = _register_csv(
        registry, tmp_path, "small",
        pd.DataFrame({"acct": [10, 11], "amt2": [100, 200]}), role="feature")

    anchor_id, feature_ids = discover_join_inputs(registry, "task-1", source_dir=None)

    assert anchor_id == big.id  # no target anywhere -> largest is anchor
    assert feature_ids == [small.id]


def test_discover_requires_at_least_two_data_files(tmp_path):
    registry = _registry(tmp_path)
    _register_csv(registry, tmp_path, "only", pd.DataFrame({"k": [1, 2]}), role="sample")

    with pytest.raises(JoinSetupError):
        discover_join_inputs(registry, "task-1", source_dir=None)


def test_propose_roles_is_deterministic_anchor_first(tmp_path):
    registry = _registry(tmp_path)
    # a carries a target (bad_flag 0/1); b has more rows but no target
    a = _register_csv(registry, tmp_path, "a", pd.DataFrame({"acct": [10, 11], "bad_flag": [0, 1]}), role="sample")
    b = _register_csv(registry, tmp_path, "b", pd.DataFrame({"acct": [10, 11, 12], "amt": [100, 200, 300]}), role="feature")
    ordered = propose_roles([b, a])
    assert ordered[0].id == a.id  # target-carrying first regardless of input order / row count


def test_build_proposal_reconciles_third_source_file_after_two_are_registered(tmp_path):
    registry = _registry(tmp_path)
    source_dir = tmp_path / "materials"
    source_dir.mkdir()
    sample_path = source_dir / "y.csv"
    first_vars_path = source_dir / "vars.csv"
    second_vars_path = source_dir / "vars.xlsx"
    pd.DataFrame({"id": [1, 2], "bad_flag": [0, 1]}).to_csv(sample_path, index=False)
    pd.DataFrame({"id": [1, 2], "x1": [3, 4]}).to_csv(first_vars_path, index=False)
    pd.DataFrame({"id": [1, 2], "x2": [5, 6]}).to_excel(second_vars_path, index=False)
    registry.register_from_upload("task-1", sample_path, role="feature")
    registry.register_from_upload("task-1", first_vars_path, role="feature")

    proposal = build_join_proposal(registry, "task-1", source_dir)
    repeated = build_join_proposal(registry, "task-1", source_dir)

    assert len(proposal.files) == 3
    assert len(repeated.files) == 3


def test_reconcile_uses_exact_source_identity_when_same_stem_xlsx_was_registered_first(
    tmp_path,
):
    registry = _registry(tmp_path)
    source_dir = tmp_path / "materials"
    source_dir.mkdir()
    sample_path = source_dir / "y.csv"
    csv_path = source_dir / "vars.csv"
    xlsx_path = source_dir / "vars.xlsx"
    pd.DataFrame({"id": [1, 2], "bad_flag": [0, 1]}).to_csv(sample_path, index=False)
    pd.DataFrame({"id": [1, 2], "x_csv": [3, 4]}).to_csv(csv_path, index=False)
    pd.DataFrame({"id": [1, 2], "x_xlsx": [5, 6]}).to_excel(xlsx_path, index=False)
    registry.register_from_upload("task-1", sample_path, role="feature")
    registry.register_from_upload("task-1", xlsx_path, role="feature")

    proposal = build_join_proposal(registry, "task-1", source_dir)

    assert len(proposal.files) == 3
    identities = {
        registry.source_identity(item.id)["original_name"]
        for item in registry.list_for_task("task-1")
    }
    assert identities == {"y.csv", "vars.csv", "vars.xlsx"}


def test_reconcile_keeps_repeated_nested_names_and_same_content_aliases(tmp_path):
    registry = _registry(tmp_path)
    source_dir = tmp_path / "materials"
    left = source_dir / "left"
    right = source_dir / "right"
    left.mkdir(parents=True)
    right.mkdir(parents=True)
    first = left / "vars.csv"
    second = right / "vars.csv"
    alias = source_dir / "vars_alias.csv"
    pd.DataFrame({"id": [1, 2], "x_left": [3, 4]}).to_csv(first, index=False)
    pd.DataFrame({"id": [1, 2], "x_right": [5, 6]}).to_csv(second, index=False)
    # Same bytes as ``first`` but a different user-supplied name remains a
    # distinct input table (the normalized parquet may still be deduplicated).
    alias.write_bytes(first.read_bytes())
    registry.register_from_upload("task-1", first, role="feature")

    proposal = build_join_proposal(registry, "task-1", source_dir)
    repeated = build_join_proposal(registry, "task-1", source_dir)

    assert len(proposal.files) == 3
    assert len(repeated.files) == 3
    identities = [
        registry.source_identity(item.id)
        for item in registry.list_for_task("task-1")
    ]
    assert {item["resolved_path"] for item in identities} == {
        str(first.resolve()),
        str(second.resolve()),
        str(alias.resolve()),
    }


def test_reconcile_same_path_with_changed_bytes_registers_new_source_identity(tmp_path):
    registry = _registry(tmp_path)
    source_dir = tmp_path / "materials"
    source_dir.mkdir()
    source = source_dir / "vars.csv"
    pd.DataFrame({"id": [1, 2], "x": [3, 4]}).to_csv(source, index=False)
    first = registry.register_from_upload("task-1", source, role="feature")
    first_identity = registry.source_identity(first.id)

    pd.DataFrame({"id": [1, 2], "x": [30, 40]}).to_csv(source, index=False)
    proposal = build_join_proposal(registry, "task-1", source_dir)

    identities = [
        registry.source_identity(item.id)
        for item in registry.list_for_task("task-1")
    ]
    assert len(proposal.files) == 2
    assert len(identities) == 2
    assert {item["resolved_path"] for item in identities} == {str(source.resolve())}
    assert len({item["sha256"] for item in identities}) == 2
    assert first_identity["sha256"] in {item["sha256"] for item in identities}


def test_wide_proposal_keeps_tail_targets_and_requires_choice_when_ambiguous(tmp_path):
    registry = _registry(tmp_path)
    feature_columns = {
        f"feature_{index:03d}": [index, index + 1]
        for index in range(843)
    }
    sample = _register_csv(
        registry,
        tmp_path,
        "wide_sample",
        pd.DataFrame({
            **feature_columns,
            "label_sqandzy": [0, 1],
            "label_sqandzy_new": [1, 0],
        }),
        role="sample",
    )
    feature = _register_csv(
        registry,
        tmp_path,
        "feature_table",
        pd.DataFrame({"join_id": [1, 2, 3], "x": [4, 5, 6]}),
        role="feature",
    )

    proposal = build_join_proposal(registry, "task-1", source_dir=None)
    anchor = next(item for item in proposal.files if item.dataset_id == sample.id)

    assert proposal.anchor_id == sample.id
    assert proposal.feature_ids == [feature.id]
    assert proposal.target_col is None
    assert anchor.has_target is True
    assert anchor.candidate_target is None
    assert len(anchor.columns) <= 300
    assert "label_sqandzy" in anchor.columns
    assert "label_sqandzy_new" in anchor.columns


def test_files_annotations_are_postponed_for_legacy_tool_workers() -> None:
    """The configured validation worker may be Python 3.7.

    ``scan_source_dir`` uses modern built-in generic annotations; without
    postponed evaluation the data_ops worker fails while importing the module,
    before it can execute a join.
    """
    files_source = (Path(__file__).parents[1] / "marvis" / "files.py").read_text(
        encoding="utf-8"
    )
    assert files_source.startswith("from __future__ import annotations\n")
