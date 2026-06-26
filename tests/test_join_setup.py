"""JOIN setup: anchor/feature role proposal over a task's registered datasets."""

from __future__ import annotations

import pandas as pd
import pytest

from marvis.agent.join_setup import JoinSetupError, discover_join_inputs, propose_roles
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
