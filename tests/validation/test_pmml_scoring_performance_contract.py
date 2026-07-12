from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import inspect
from pathlib import Path
import threading
from typing import Any

import numpy as np
import pandas as pd

from marvis.validation.pmml_scoring import (
    PmmlScorer,
    TaskPmmlScorerRegistry,
    _prediction_score_value,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "min_lr.pmml"


def test_scorer_source_contains_no_record_loop():
    source = inspect.getsource(PmmlScorer.score_chunk)
    assert 'to_dict(orient="records")' not in source
    assert "for record" not in source


def test_score_chunk_preserves_input_order_and_marks_nulls():
    class StaticBatchModel:
        def predict(self, _frame: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({"probability_1": [0.3, None, float("inf")]})

    scorer = PmmlScorer(model=StaticBatchModel(), positive_output_field="probability_1")
    scores = scorer.score_chunk(
        pd.DataFrame({"x": [30, 10, 20]}, index=[7, 2, 9])
    )

    assert scores.index.tolist() == [7, 2, 9]
    assert scores.iloc[0] == 0.3
    assert pd.isna(scores.iloc[1])
    assert scores.iloc[2] == float("inf")


def test_real_pmml_dataframe_batch_matches_legacy_single_row_semantics():
    from pypmml import Model

    frame = pd.DataFrame(
        {
            "x1": [-2.0, -1.0, 0.0, 1.0, 2.0],
            "x2": [0.0, 1.0, 0.0, 1.0, 0.0],
        },
        index=[8, 3, 7, 1, 9],
    )
    model = Model.fromFile(str(FIXTURE))
    legacy = np.asarray(
        [
            float(_prediction_score_value(model.predict(record), "probability_1"))
            for record in frame.to_dict(orient="records")
        ]
    )

    batch = PmmlScorer(model, "probability_1").score_chunk(frame).to_numpy(dtype=float)

    np.testing.assert_allclose(batch, legacy, rtol=1e-12, atol=1e-12)


def test_real_pmml_empty_dataframe_short_circuits_without_entering_jvm():
    from pypmml import Model

    model = Model.fromFile(str(FIXTURE))
    empty = pd.DataFrame(
        {"x1": pd.Series(dtype="float64"), "x2": pd.Series(dtype="float64")},
        index=pd.Index([], name="source_row"),
    )
    scorer = PmmlScorer(model, "probability_1")

    # pypmml/PMML4S raises while scoring an empty DataFrame, so success here also
    # proves that PmmlScorer did not cross the JVM boundary.
    scores = scorer.score_chunk(empty)

    assert scores.index.equals(empty.index)
    assert scores.name == "pmml_score"
    assert scores.dtype == "float64"
    assert scores.empty


def test_task_scorer_registry_reuses_and_invalidates_hash_and_output(monkeypatch):
    loads: list[str] = []

    class LoadedModel:
        def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({"probability_1": [0.5] * len(frame)})

    def load(path: str) -> LoadedModel:
        loads.append(path)
        return LoadedModel()

    monkeypatch.setattr("marvis.validation.pmml_scoring.Model.fromFile", load)
    registry = TaskPmmlScorerRegistry(max_tasks=2)
    first = registry.get(
        task_id="t1",
        pmml_path=Path("one.pmml"),
        pmml_sha256="a",
        output_field="probability_1",
    )
    second = registry.get(
        task_id="t1",
        pmml_path=Path("one.pmml"),
        pmml_sha256="a",
        output_field="probability_1",
    )
    changed_hash = registry.get(
        task_id="t1",
        pmml_path=Path("one.pmml"),
        pmml_sha256="b",
        output_field="probability_1",
    )
    changed_output = registry.get(
        task_id="t1",
        pmml_path=Path("one.pmml"),
        pmml_sha256="b",
        output_field="probability_0",
    )

    assert first is second
    assert changed_hash is not second
    assert changed_output is not changed_hash
    assert loads == ["one.pmml", "one.pmml", "one.pmml"]


def test_task_scorer_registry_evicts_least_recently_used_entry(monkeypatch):
    loads: list[str] = []

    monkeypatch.setattr(
        "marvis.validation.pmml_scoring.Model.fromFile",
        lambda path: loads.append(path) or object(),
    )
    registry = TaskPmmlScorerRegistry(max_tasks=2)

    def get(task_id: str) -> PmmlScorer:
        return registry.get(
            task_id=task_id,
            pmml_path=Path(f"{task_id}.pmml"),
            pmml_sha256=task_id,
            output_field="probability_1",
        )

    first = get("t1")
    get("t2")
    assert get("t1") is first  # t2 is now least recently used.
    get("t3")
    get("t2")

    assert loads == ["t1.pmml", "t2.pmml", "t3.pmml", "t2.pmml"]


def test_task_scorer_registry_clear_forces_reload(monkeypatch):
    loads: list[str] = []
    monkeypatch.setattr(
        "marvis.validation.pmml_scoring.Model.fromFile",
        lambda path: loads.append(path) or object(),
    )
    registry = TaskPmmlScorerRegistry(max_tasks=2)
    request = {
        "task_id": "t1",
        "pmml_path": Path("one.pmml"),
        "pmml_sha256": "a",
        "output_field": "probability_1",
    }

    first = registry.get(**request)
    registry.clear("t1")
    second = registry.get(**request)

    assert second is not first
    assert loads == ["one.pmml", "one.pmml"]


def test_registry_does_not_hold_global_lock_while_loading_models(monkeypatch):
    barrier = threading.Barrier(2, timeout=2)
    loaded: list[str] = []
    loaded_lock = threading.Lock()

    def load(path: str) -> Any:
        barrier.wait()
        with loaded_lock:
            loaded.append(path)
        return object()

    monkeypatch.setattr("marvis.validation.pmml_scoring.Model.fromFile", load)
    registry = TaskPmmlScorerRegistry(max_tasks=2)

    def get(task_id: str) -> PmmlScorer:
        return registry.get(
            task_id=task_id,
            pmml_path=Path(f"{task_id}.pmml"),
            pmml_sha256=task_id,
            output_field="probability_1",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        scorers = list(executor.map(get, ("t1", "t2")))

    assert len(scorers) == 2
    assert sorted(loaded) == ["t1.pmml", "t2.pmml"]
