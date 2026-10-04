"""Numerical fold selection cannot learn from held-out rows."""

import base64
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

from marvis.packs.modeling.errors import ModelingError, SpecialValueDecisionRequiredError
from marvis.packs.modeling.fold_selection import FoldSelectionSession
from tests.test_feature_pack import _runtime


def _frame():
    rng = np.random.default_rng(3)
    n = 240
    target = rng.binomial(1, .45, n)
    return pd.DataFrame({"x": target * .6 + rng.normal(size=n),
                         "z": rng.normal(size=n), "y": target,
                         "split": np.repeat(["train", "test", "oot"], 80)})


@contextmanager
def _session(tmp_path, frame, **kwargs):
    _, registry, _, _ = _runtime(tmp_path)
    path = tmp_path / "sample.parquet"
    frame.to_parquet(path, index=False)
    source = registry.register_from_upload("task-feature", path, role="sample")
    with FoldSelectionSession(registry, source.id, candidates=["x", "z"], target_col="y",
        control_columns=["split"], screen_parameters={"leakage_ks": 1.0, "top_k": 2},
        selection_parameters={"iv_min": 0.0, "corr_max": 1.0, "vif_max": 1e9, "top_k": 1},
        **kwargs) as session:
        yield session


def _fold(session, **kwargs):
    return session.prepare_fold(**{
        "fit_positions": np.arange(120), "train_positions": np.arange(160),
        "valid_positions": np.arange(120, 160), "test_positions": np.arange(160, 200),
        **kwargs,
    })


def test_validation_values_and_labels_do_not_change_selection(tmp_path):
    baseline = _frame()
    changed = baseline.copy()
    changed.loc[120:, "y"] = 1 - changed.loc[120:, "y"]
    changed.loc[120:, "x"] = np.arange(120) * 100
    changed.loc[120:, "z"] = changed.loc[120:, "y"]
    with _session(tmp_path / "baseline", baseline) as session:
        original = _fold(session)
    with _session(tmp_path / "changed", changed) as session:
        result = _fold(session)
    assert original.features == result.features == ("x",)
    pd.testing.assert_frame_equal(original.fit, result.fit)
    assert not original.test.equals(result.test)
    membership = result.evidence["memberships"]["fit"]
    bits = np.unpackbits(np.frombuffer(base64.b64decode(membership["membership"]), dtype=np.uint8), bitorder="little")
    assert np.array_equal(np.flatnonzero(bits), np.arange(120))
    assert result.evidence["upstream_candidate_selection"] == "unknown"
    assert result.evidence["historical_availability"] == "unknown"


def test_each_fold_reselects_from_complete_candidates(tmp_path):
    frame = _frame()
    rng = np.random.default_rng(82)
    frame.loc[:119, "x"] = frame.loc[:119, "y"] + rng.normal(scale=.7, size=120)
    frame.loc[120:, "x"] = rng.normal(size=120)
    frame.loc[:119, "z"] = rng.normal(size=120)
    frame.loc[120:, "z"] = frame.loc[120:, "y"] + rng.normal(scale=.7, size=120)
    with _session(tmp_path, frame) as session:
        first = _fold(session, train_positions=np.arange(120), valid_positions=np.arange(120, 240), test_positions=np.arange(120, 240))
        second = _fold(session, fit_positions=np.arange(120, 240), train_positions=np.arange(120, 240),
                       valid_positions=np.arange(120), test_positions=np.arange(120))
    assert first.features == ("x",)
    assert second.features == ("z",)
    assert first.evidence["candidates"] == second.evidence["candidates"] == ["x", "z"]


@pytest.mark.parametrize("positions", [[0, 0], [False, True], [0.0, 1.0], [-1, 1], [240], []])
def test_invalid_native_membership_rejected(tmp_path, positions):
    with _session(tmp_path, _frame()) as session, pytest.raises(ModelingError):
        _fold(session, fit_positions=positions)


def test_early_stopping_overlap_rejected_before_selection(tmp_path):
    with _session(tmp_path, _frame()) as session, pytest.raises(ModelingError, match="overlap"):
        _fold(session, valid_positions=np.arange(119, 160))


@pytest.mark.parametrize("target_type", ["binary", "continuous", "multiclass"])
def test_undeclared_sentinels_need_a_decision_and_fixed_mask_replays(tmp_path, target_type):
    frame = _frame()
    frame.loc[::8, "x"] = -999.0
    with _session(tmp_path / "unconfirmed", frame, target_type=target_type) as session, pytest.raises(SpecialValueDecisionRequiredError):
        _fold(session)
    with _session(tmp_path / "mask", frame, target_type=target_type, special_value_policy={"x": {"action": "mask", "values": [-999.0]}}) as session:
        result = _fold(session)
    assert result.features == ("x",)
    assert result.train.loc[::8, "x"].isna().all()
    assert result.test.loc[160:199:8, "x"].isna().all()


def test_feature_column_reads_remain_batched(tmp_path, monkeypatch):
    from marvis.packs.modeling import fold_selection

    frame = _frame()
    for index in range(40):
        frame[f"extra{index}"] = np.random.default_rng(index + 12).normal(size=len(frame))
    _, registry, _, _ = _runtime(tmp_path)
    path = tmp_path / "wide.parquet"
    frame.to_parquet(path, index=False)
    source = registry.register_from_upload("task-feature", path, role="sample")
    observed = []
    original = fold_selection._ProjectedBackend.read_frame

    def observe(self, path, *, columns=None):
        observed.append(list(columns if columns is not None else self.columns))
        return original(self, path, columns=columns)

    monkeypatch.setattr(fold_selection._ProjectedBackend, "read_frame", observe)
    with FoldSelectionSession(registry, source.id, candidates=[c for c in frame if c not in {"y", "split"}],
        target_col="y", control_columns=["split"], screen_parameters={"leakage_ks": 1.0, "top_k": 2, "batch_size": 3},
        selection_parameters={"iv_min": 0.0, "vif_max": 1e9, "top_k": 1}) as session:
        _fold(session)
    assert max(map(len, observed)) <= 3
    assert any("extra39" in columns for columns in observed)


def test_returned_policy_cannot_change_later_folds(tmp_path):
    frame = _frame()
    frame.loc[::8, "x"] = -999.0
    with _session(tmp_path, frame, special_value_policy={"x": {"action": "mask", "values": [-999.0]}}) as session:
        first = _fold(session)
        first.evidence["special_value_policy"]["x"]["values"] = []
        first.evidence["selection_parameters"]["iv_min"] = 1000
        second = _fold(session)
    assert second.features == ("x",)
    assert second.train.loc[::8, "x"].isna().all()
    assert second.evidence["selection_parameters"]["multivariate_sample_rows"] == 50_000
    assert second.evidence["screen_parameters"]["max_missing_rate"] == .95


def test_changed_source_invalidates_session_before_publication(tmp_path):
    with pytest.raises(RuntimeError, match="changed"):
        with _session(tmp_path, _frame()) as session:
            _fold(session)
            path = session.registry.resolve_path(session.source.id)
            path.chmod(0o600)
            path.write_bytes(b"concurrent source replacement")


@pytest.mark.parametrize("target_type", ["continuous", "multiclass"])
def test_nonbinary_folds_use_local_association_and_keep_source_positions(tmp_path, target_type):
    frame = _frame()
    rng = np.random.default_rng(9)
    frame["y"] = rng.normal(size=len(frame)) if target_type == "continuous" else np.arange(len(frame)) % 3
    frame["x"] = frame["y"] + rng.normal(scale=.3, size=len(frame))
    with _session(tmp_path, frame, target_type=target_type) as session:
        result = _fold(session, fit_positions=np.arange(119, -1, -1))
    assert result.features == ("x",)
    assert result.fit.index.tolist() == list(range(119, -1, -1))
    pd.testing.assert_series_equal(result.fit["y"], frame.loc[119::-1, "y"], check_dtype=False)


def test_controls_project_training_rows_before_accessing_labels_and_preserve_groups(tmp_path, monkeypatch):
    from marvis.packs.modeling import fold_selection

    frame = _frame()
    frame["borrower"] = np.arange(len(frame), dtype=np.int64) + 2**53
    observed = []
    original = fold_selection._ProjectedBackend.read_frame

    def observe(self, path, *, columns=None):
        observed.append((self.columns, self.positions.copy()))
        return original(self, path, columns=columns)

    monkeypatch.setattr(fold_selection._ProjectedBackend, "read_frame", observe)
    with _session(tmp_path, frame) as session:
        controls = session.training_controls(split_col="split", train_value="train", group_columns=["borrower", "missing"])
    assert observed[0][0] == ("split",)
    assert observed[1][0] == ("y", "borrower")
    assert np.array_equal(observed[1][1], np.arange(80))
    assert controls["borrower"].dtype == pd.Int64Dtype()
    assert controls["borrower"].nunique() == 80
    assert controls.index.tolist() == list(range(80))


@pytest.mark.parametrize("unsigned", [False, True])
def test_native_nullable_integer_projection_masks_only_the_exact_value(tmp_path, unsigned):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from marvis.packs.modeling.fold_selection import _ProjectedBackend

    base = 2**63 if unsigned else 2**60
    path = tmp_path / "no-pandas-metadata.parquet"
    pq.write_table(pa.table({"x": pa.array([base, base + 1, None, 0],
        type=pa.uint64() if unsigned else pa.int64())}), path)
    projected = _ProjectedBackend(pq.ParquetFile(path), np.arange(4), ["x"],
        {"x": {"action": "mask", "values": [float(base)]}}).read_frame(None)
    assert projected.x.isna().tolist() == [True, False, True, False]
    assert int(projected.x.iloc[1]) == base + 1


@pytest.mark.parametrize("unsigned", [False, True])
def test_full_fold_preserves_declared_large_integer_policy(tmp_path, unsigned):
    import pyarrow as pa
    import pyarrow.parquet as pq

    _, registry, _, _ = _runtime(tmp_path)
    base = 2**63 if unsigned else 2**60
    rng = np.random.default_rng(916)
    values = [int(value) for value in rng.integers(0, 100, size=240)]
    for index in range(240):
        if index % 6 in (0, 1):
            values[index] = base + index % 6
        if index % 23 == 2:
            values[index] = None
    path = tmp_path / "integer-policy.parquet"
    pq.write_table(pa.table({"x": pa.array(values, type=pa.uint64() if unsigned else pa.int64()),
        "y": rng.integers(0, 2, size=240)}), path)
    source = registry.register_from_upload("task-feature", path, role="sample")
    with FoldSelectionSession(registry, source.id, candidates=["x"], target_col="y",
        screen_parameters={"leakage_ks": 1.0},
        selection_parameters={"iv_min": 0.0, "vif_max": 1e9},
        special_value_policy={"x": {"action": "mask", "values": [base + 1]}}) as session:
        result = _fold(session)
    assert result.features == ("x",)
    assert result.evidence["special_value_policy"]["x"]["values"] == [base + 1]
    for frame in (result.fit, result.train, result.valid, result.test):
        for index, value in frame.x.items():
            if values[index] in (None, base + 1):
                assert pd.isna(value)
            else:
                assert int(value) == values[index]
