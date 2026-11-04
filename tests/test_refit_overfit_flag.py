"""A train+test refit keeps an honest overfit flag (review A1).

The stored flag mixes in the in-distribution refit holdout, so it is dropped as
tainted; the train/OOT trigger is recomputed instead of leaving None.
"""
from types import SimpleNamespace

from marvis.packs.modeling import select_tools
from marvis.packs.modeling.delivery_tools import _model_card_key_metrics, refit_train_oot_overfit_flag


class _Experiments:
    def __init__(self, row):
        self.row = row

    def create(self, *_args):
        return "experiment_refit"

    def attach_result(self, *_args):
        return None

    def get(self, _id):
        return SimpleNamespace(artifact_id="artifact_refit")

    def compare(self, ids):
        return {"experiments": [{"id": ids[0], **self.row}]}


def _refit(monkeypatch, tmp_path, row, pre):
    monkeypatch.setattr(select_tools, "_refit_champion_on_train_plus_test",
                        lambda *a, **k: ({}, SimpleNamespace(artifact=None)))
    runtime = SimpleNamespace(settings=SimpleNamespace(tasks_dir=tmp_path), experiments=_Experiments(row))
    experiment = SimpleNamespace(task_id="task-1", config=SimpleNamespace(params={}))
    return select_tools._apply_champion_refit(
        runtime, experiment=experiment, recipe="lr", requested=True, pre_refit_metrics=pre,
    )


def test_refit_with_train_oot_gap_keeps_the_overfit_flag(monkeypatch, tmp_path):
    # holdout says no overfit; train/OOT gap 0.12 > 0.05 does
    row = {"train_ks": 0.52, "oot_ks": 0.40, "test_ks": 0.51, "overfit_flag": False}
    refit = _refit(monkeypatch, tmp_path, row, {"overfit_flag": False, "oot_ks": 0.41})

    assert refit["metrics"]["overfit_flag"] is True
    assert refit["overfit_flag_basis"] == "refit_train_oot"
    assert "test_ks" not in refit["metrics"]
    assert refit["refit_holdout_metrics"]["refit_holdout_overfit_flag"] is False


def test_refit_without_oot_carries_the_pre_refit_flag(monkeypatch, tmp_path):
    refit = _refit(monkeypatch, tmp_path, {"train_ks": 0.5, "overfit_flag": False}, {"overfit_flag": True})

    assert refit["metrics"]["overfit_flag"] is True
    assert refit["overfit_flag_basis"] == "pre_refit_heldout"


def test_refit_model_card_shows_the_train_oot_flag_not_the_stored_one():
    metrics = {"train_ks": 0.52, "oot_ks": 0.40, "test_ks": 0.51, "overfit_flag": False}
    rows = {row["metric"]: row for row in _model_card_key_metrics(metrics, is_refit=True)}

    assert rows["overfit_flag"] == {"metric": "overfit_flag", "value": True, "basis": "train_oot_only"}
    assert "test_ks" not in rows
    assert refit_train_oot_overfit_flag({"train_ks": 0.5}) is None
    assert refit_train_oot_overfit_flag({"train_ks": 0.45, "oot_ks": 0.42}) is False
