"""Executable limits of the current shared-file modeling path.

These synthetic probes distinguish numerical fit isolation from blind OOT
custody. They describe the legacy path, which must remain usable; passing them
does not establish a sealed-data service, a final-model freeze, or an evaluation
budget. A future explicit blind mode needs separate acceptance tests.
"""

import hashlib
from io import BytesIO
import os

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd

from marvis.app import create_app
from marvis.data import authenticated_snapshot
from marvis.packs.modeling import fold_selection
from marvis.packs.modeling.fold_selection import FoldSelectionSession
from marvis.plugins.manifest import ToolRef
from tests.test_feature_pack import _runtime


_TASK = "task-feature"
_OOT_MARKER = 987654321.0


def _source(tmp_path):
    runner, registry, repo, backend = _runtime(tmp_path)
    rng = np.random.default_rng(306)
    rows = 360
    y = np.arange(rows) % 2
    frame = pd.DataFrame({
        "x": y * 0.6 + rng.normal(size=rows),
        "z": rng.normal(size=rows),
        "y": y,
        "split": ["train"] * 240 + ["test"] * 60 + ["oot"] * 60,
        "borrower": np.arange(rows),
    })
    frame.loc[300:, "z"] = _OOT_MARKER
    upload = tmp_path / "synthetic.parquet"
    frame.to_parquet(upload, index=False)
    source = registry.register_from_upload(_TASK, upload, role="sample")
    return runner, registry, repo, backend, source, frame


def _session(registry, source):
    return FoldSelectionSession(
        registry, source.id, candidates=["x", "z"], target_col="y",
        control_columns=["split"],
        screen_parameters={"leakage_ks": 1.0, "top_k": 2},
        selection_parameters={"iv_min": 0.0, "vif_max": 1e9, "top_k": 1},
    )


def test_opening_legacy_fold_copies_oot_bytes_before_any_fit(tmp_path, monkeypatch):
    """One retained descriptor authenticates the full mixed physical file."""
    _, registry, _, _, source, _ = _source(tmp_path)
    source_path = registry.resolve_verified_path(source.id)
    identity = source_path.stat()
    copied = []
    real_read = os.read

    def observe_read(fd, count):
        value = real_read(fd, count)
        current = os.fstat(fd)
        if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
            copied.append(value)
        return value

    monkeypatch.setattr(authenticated_snapshot.os, "read", observe_read)
    # No training_controls(), feature selector, learner, or evaluation runs.
    with _session(registry, source):
        bytes_seen = b"".join(copied)
        assert hashlib.sha256(bytes_seen).hexdigest() == source.content_hash
        decoded = pd.read_parquet(BytesIO(bytes_seen))
        assert decoded.loc[decoded["split"].eq("oot"), "z"].eq(_OOT_MARKER).all()


def test_legacy_fold_decodes_oot_batches_but_returns_only_fit_members(tmp_path, monkeypatch):
    """Arrow row projection occurs after shared-file batches are decoded."""
    _, registry, _, _, source, frame = _source(tmp_path)
    real_parquet = fold_selection.pq.ParquetFile
    decoded_columns = set()

    class ObservedParquet:
        def __init__(self, stream):
            self.delegate = real_parquet(stream)

        def __getattr__(self, name):
            return getattr(self.delegate, name)

        def iter_batches(self, **kwargs):
            for batch in self.delegate.iter_batches(**kwargs):
                for column in ("y", "z", "borrower"):
                    if column in batch.schema.names:
                        # These source positions include OOT. This observes the
                        # actual Arrow batch, before _ProjectedBackend.take().
                        values = batch.column(column).to_pylist()
                        if len(values) == len(frame):
                            assert values[300:] == frame.loc[300:, column].tolist()
                            decoded_columns.add(column)
                yield batch

    monkeypatch.setattr(fold_selection.pq, "ParquetFile", ObservedParquet)
    with _session(registry, source) as session:
        controls = session.training_controls(
            split_col="split", train_value="train", group_columns=["borrower"],
        )
        selected = session.prepare_final_fit(np.arange(240))
    assert {"y", "z", "borrower"} <= decoded_columns
    assert controls.index.tolist() == list(range(240))
    assert selected.train.index.tolist() == list(range(240))
    assert "split" not in selected.train
    assert selected.evidence["scope"] == "supplied_candidate_algorithm_fit_only"
    assert selected.evidence["upstream_candidate_selection"] == "unknown"
    assert selected.evidence["historical_availability"] == "unknown"


def test_legacy_dataset_role_does_not_seal_a_shared_source(tmp_path):
    """A role label has no actor/authority or byte access gate today."""
    _, registry, _, backend, source, frame = _source(tmp_path)
    copied = tmp_path / "declared-oot.parquet"
    frame.to_parquet(copied, index=False)
    declared_oot = registry.register_from_upload(_TASK, copied, role="oot")
    assert registry.get(declared_oot.id).role == "oot"
    assert declared_oot.content_hash == source.content_hash
    decoded = backend.read_frame(registry.resolve_verified_path(declared_oot.id))
    assert decoded.loc[decoded["split"].eq("oot"), "z"].eq(_OOT_MARKER).all()


def test_legacy_training_repeatedly_returns_oot_metrics_before_champion_freeze(tmp_path):
    """Real tools and HTTP expose candidate OOT metrics without blind admission."""
    runner, registry, _, _, source, _ = _source(tmp_path)
    inputs = {
        "dataset_id": source.id, "recipe": "lr", "features": ["x"],
        "target_col": "y", "split_col": "split",
        "split_values": {"train": "train", "test": "test", "oot": "oot"},
        "seed": 19, "params": {},
    }
    results = [runner.invoke(ToolRef("modeling", "train_model"), inputs,
                             task_id=_TASK) for _ in range(2)]
    for result in results:
        assert result.ok, result.error
        assert result.output["metrics"]["oot_ks"] is not None
        assert result.output["metrics"]["oot_auc"] is not None
    first, second = (result.output for result in results)
    assert first["experiment_id"] != second["experiment_id"]
    assert first["artifact_id"] != second["artifact_id"]
    assert first["metrics"]["oot_ks"] == second["metrics"]["oot_ks"]
    assert first["metrics"]["oot_auc"] == second["metrics"]["oot_auc"]
    # This is the normal local application, including local-session middleware,
    # and no select_experiment, final freeze, or OOT admission has run.
    with TestClient(create_app(registry.datasets_root.parent)) as client:
        response = client.get(f"/api/tasks/{_TASK}/experiments")
        assert response.status_code == 200, response.text
        candidates = response.json()["experiments"]
        assert len(candidates) == 2
        assert {item["id"] for item in candidates} == {
            first["experiment_id"], second["experiment_id"],
        }
        assert all(item["metrics"]["oot_auc"] is not None for item in candidates)
