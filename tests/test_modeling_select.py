import numpy as np
import pandas as pd

from marvis.data.backend import DataBackend
from marvis.packs.modeling.select import SelectionResult, select_features


def test_select_features_drops_low_iv_and_applies_top_k(tmp_path):
    rows = 200
    target = np.array([0, 1] * (rows // 2))
    frame = pd.DataFrame({
        "strong": target + np.linspace(0, 0.01, rows),
        "medium": np.where(target == 1, 0.7, 0.3) + (np.arange(rows) % 5) * 0.01,
        "weak": (np.arange(rows) * 17 % 101) / 100,
        "y": target,
    })
    path = tmp_path / "select.parquet"
    frame.to_parquet(path, index=False)

    result = select_features(
        DataBackend(tmp_path),
        path,
        features=["strong", "medium", "weak"],
        target_col="y",
        iv_min=0.05,
        corr_max=1.1,
        vif_max=1_000_000,
        top_k=1,
    )

    assert isinstance(result, SelectionResult)
    assert len(result.selected) == 1
    assert "strong" in result.scores
    assert any(feature == "weak" and reason.startswith("low IV") for feature, reason in result.dropped)
    assert any(reason == "outside top_k 1" for _feature, reason in result.dropped)


def test_select_features_removes_lower_iv_collinear_feature(tmp_path):
    rows = 200
    target = np.array([0, 1] * (rows // 2))
    strong = target + np.linspace(0, 0.02, rows)
    frame = pd.DataFrame({
        "strong": strong,
        "strong_copy": strong * 2,
        "independent": (np.arange(rows) * 19 % 97) / 100,
        "y": target,
    })
    path = tmp_path / "collinear.parquet"
    frame.to_parquet(path, index=False)

    result = select_features(
        DataBackend(tmp_path),
        path,
        features=["strong", "strong_copy", "independent"],
        target_col="y",
        iv_min=0.0,
        corr_max=0.8,
        vif_max=1_000_000,
    )

    assert "strong" in result.selected
    assert "strong_copy" not in result.selected
    assert any(
        feature == "strong_copy" and reason.startswith("collinear with strong")
        for feature, reason in result.dropped
    )


def test_select_features_records_high_vif_drop_reason(tmp_path):
    rng = np.random.RandomState(7)
    rows = 200
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    x3 = x1 + x2
    frame = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "y": (x1 + rng.normal(scale=0.2, size=rows) > 0).astype(int),
    })
    path = tmp_path / "vif.parquet"
    frame.to_parquet(path, index=False)

    result = select_features(
        DataBackend(tmp_path),
        path,
        features=["x1", "x2", "x3"],
        target_col="y",
        iv_min=0.0,
        corr_max=0.99,
        vif_max=5.0,
    )

    assert any(reason.startswith("high VIF") for _feature, reason in result.dropped)
    assert all("vif" in result.scores[feature] for feature in ("x1", "x2", "x3"))
