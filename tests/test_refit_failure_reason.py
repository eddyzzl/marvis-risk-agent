"""A failed champion refit is explained in business Chinese.

The card used to read 「未重新拟合：全量重训失败,已保留原候选:<exception text>」.
The raw exception now lives in ``failure_detail``; the reason never carries it.
"""
from types import SimpleNamespace

import pytest

from marvis.agent.renderers import _render_select_experiment
from marvis.packs.modeling import select_tools

_RAW = "ValueError: Input X contains NaN at /opt/site-packages/sklearn/utils/validation.py:147"


def _experiment():
    return SimpleNamespace(task_id="task-1", config=SimpleNamespace(params={}))


def _refit_with(monkeypatch, tmp_path, exc):
    def boom(*_args, **_kwargs):
        raise exc

    monkeypatch.setattr(select_tools, "_refit_champion_on_train_plus_test", boom)
    runtime = SimpleNamespace(settings=SimpleNamespace(tasks_dir=tmp_path))
    return select_tools._apply_champion_refit(
        runtime, experiment=_experiment(), recipe="lr", requested=True, pre_refit_metrics={},
    )


@pytest.mark.parametrize(("exc", "cause"), [
    (ValueError(_RAW), "训练过程出错"),
    (MemoryError("cannot allocate 8.0 GiB"), "内存不足"),
    (TimeoutError("fit exceeded 600s"), "运行超时"),
])
def test_refit_failure_reason_is_chinese_and_raw_text_moves_to_detail(monkeypatch, tmp_path, exc, cause):
    refit = _refit_with(monkeypatch, tmp_path, exc)

    assert refit["applied"] is False and refit["requested"] is True
    assert cause in refit["reason"] and "已保留原候选模型" in refit["reason"]
    assert str(exc) not in refit["reason"] and type(exc).__name__ not in refit["reason"]
    assert refit["failure_detail"] == {"error_type": type(exc).__name__, "message": str(exc)}


def test_selection_card_shows_the_explanation_not_the_exception(monkeypatch, tmp_path):
    refit = _refit_with(monkeypatch, tmp_path, ValueError(_RAW))
    text, _tables = _render_select_experiment({"recipe": "lr", "refit": refit})

    assert "未重新拟合：用 train+test 全量重训时训练过程出错" in text
    assert "sklearn" not in text and "ValueError" not in text


def test_selection_card_hides_raw_text_stored_by_earlier_versions():
    legacy = {"applied": False, "requested": True, "reason": f"全量重训失败,已保留原候选:{_RAW}"}
    text, _tables = _render_select_experiment({"recipe": "lr", "refit": legacy})

    assert "未重新拟合：用 train+test 全量重训时训练过程出错" in text
    assert "sklearn" not in text
