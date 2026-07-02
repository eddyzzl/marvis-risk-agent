import numpy as np
import pandas as pd
import pytest

from marvis.data.backend import DataBackend
from marvis.feature.errors import FitRequiresSplitError
from marvis.feature.iv import compute_woe_iv
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
        allow_full_fit=True,
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
        allow_full_fit=True,
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
        allow_full_fit=True,
    )

    assert any(reason.startswith("high VIF") for _feature, reason in result.dropped)
    assert all("vif" in result.scores[feature] for feature in ("x1", "x2", "x3"))


def test_select_features_without_split_raises_typed_error_unless_allow_full_fit(tmp_path):
    """FS-2: select_features must stop with a typed error when it has no split column
    to exclude holdout rows from fitting, mirroring PREP-1's woe/impute/normalize/cap gate."""
    rows = 40
    target = np.array([0, 1] * (rows // 2))
    frame = pd.DataFrame({
        "strong": target + np.linspace(0, 0.01, rows),
        "y": target,
    })
    path = tmp_path / "no_split.parquet"
    frame.to_parquet(path, index=False)

    with pytest.raises(FitRequiresSplitError):
        select_features(
            DataBackend(tmp_path),
            path,
            features=["strong"],
            target_col="y",
            iv_min=0.0,
        )

    result = select_features(
        DataBackend(tmp_path),
        path,
        features=["strong"],
        target_col="y",
        iv_min=0.0,
        allow_full_fit=True,
    )
    assert result.fit_split == "full"
    assert result.fit_rows == rows


def test_select_features_default_excludes_test_and_oot_from_iv_statistics(tmp_path):
    """FS-2: select_features's default holdout_values=("test","oot") must exclude both
    splits from IV — verified by making the holdout rows carry a strongly *inverted*
    label relationship (test/oot bad rate flipped) that would drag the IV score if
    those rows leaked into the fit; the score must equal a train-only oracle computed
    independently via feature_metrics on the pre-filtered train rows."""
    from marvis.feature.metrics import feature_metrics

    frame = pd.DataFrame({
        # train: feature perfectly separates the label (strong signal).
        # test/oot: label is randomized/inverted relative to feature — if these rows
        # leaked into the IV fit, the pooled IV would drop sharply from the train-only IV.
        "signal": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        "y":      [0,   0,   0,   0,   1,   1,   1,   1,   1,   0,   1,   0],
        "split": ["train"] * 8 + ["test", "test", "oot", "oot"],
    })
    path = tmp_path / "fs2_holdout.parquet"
    frame.to_parquet(path, index=False)

    result = select_features(
        DataBackend(tmp_path),
        path,
        features=["signal"],
        target_col="y",
        iv_min=0.0,
        split_col="split",
    )

    assert result.fit_split == "train"
    assert result.fit_rows == 8
    train_only = frame.iloc[:8]
    expected = feature_metrics(
        train_only["signal"].to_numpy(dtype=float),
        train_only["y"].to_numpy(dtype=float),
        feature="signal",
    )
    assert result.scores["signal"]["iv"] == pytest.approx(expected.iv)
    # Sanity: the train-only IV differs materially from what pooling in the inverted
    # holdout would produce, so this assertion is actually discriminating.
    pooled = feature_metrics(
        frame["signal"].to_numpy(dtype=float),
        frame["y"].to_numpy(dtype=float),
        feature="signal",
    )
    assert result.scores["signal"]["iv"] != pytest.approx(pooled.iv)


def test_select_features_auto_detects_standard_split_column(tmp_path):
    """FS-2: with no explicit split_col, select_features falls back to the platform's
    standard SPLIT_COLUMN ("split") when the dataset already carries one — matching
    prepare_modeling_frame's output without requiring every caller to pass split_col."""
    frame = pd.DataFrame({
        "signal": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
        "y":      [0,   0,   0,   0,   1,   1,   1,   1,   1,   0],
        "split": ["train"] * 8 + ["test", "oot"],
    })
    path = tmp_path / "fs2_auto_split.parquet"
    frame.to_parquet(path, index=False)

    result = select_features(
        DataBackend(tmp_path),
        path,
        features=["signal"],
        target_col="y",
        iv_min=0.0,
    )

    assert result.fit_split == "train"
    assert result.fit_rows == 8
