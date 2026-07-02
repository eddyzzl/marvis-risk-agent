"""Unit tests for marvis.feature.screen library-level behavior (FS-4/FS-6/FS-7/FS-10)."""
import numpy as np
import pandas as pd

from marvis.data.backend import DataBackend
from marvis.feature.screen import (
    LEAKAGE_WATCH_LOW,
    SPLIT_SHIFT_THRESHOLD,
    screen_features,
)


def _write(tmp_path, frame: pd.DataFrame, name: str = "screen.parquet"):
    path = tmp_path / name
    frame.to_parquet(path, index=False)
    return DataBackend(tmp_path), path


def test_screen_flags_split_shift_when_train_test_ks_diverge(tmp_path):
    """FS-4: a feature strongly separating the label in train but not in test (a
    migration-type leak) is flagged in split_shift even though its pooled KS is below the
    hard leakage gate."""
    rows = 400
    rng = np.random.RandomState(0)
    split = np.array((["train"] * 150 + ["test"] * 150 + ["oot"] * 100))
    y = np.array(([0, 1] * 200))
    # train: partial signal (KS ~0.3-0.4); test: pure noise (KS ~0). Pooled KS stays under
    # the 0.40 leakage gate, but |ks_train - ks_test| exceeds the split-shift threshold —
    # exactly the migration-type leak the pooled gate cannot see.
    train_signal = np.where(y == 1, 0.6, 0.4) + rng.normal(scale=0.35, size=rows)
    shifty = np.where(split == "train", train_signal, rng.normal(size=rows))
    frame = pd.DataFrame({"shifty": shifty, "y": y, "split": split})
    backend, path = _write(tmp_path, frame)

    result = screen_features(
        backend, path, features=["shifty"], target_col="y", split_col="split",
    )

    assert result.scores["shifty"]["ks_train"] is not None
    assert result.scores["shifty"]["ks_test"] is not None
    shift_cols = {feature for feature, _delta, _reason in result.split_shift}
    assert "shifty" in shift_cols
    delta = dict((f, d) for f, d, _ in result.split_shift)["shifty"]
    assert delta > SPLIT_SHIFT_THRESHOLD


def test_screen_watch_band_flags_softband_ks_without_blocking(tmp_path):
    """FS-4: a feature whose pooled-dev KS lands in [LEAKAGE_WATCH_LOW, leakage_ks) is
    surfaced in leakage_watch but still kept in the clean/ranked set (not blocked)."""
    rows = 400
    rng = np.random.RandomState(3)
    y = np.array(([0, 1] * 200))
    # Build a feature with KS in the watch band: mostly-signal with added noise.
    noise = rng.normal(scale=1.0, size=rows)
    watch = y.astype(float) + noise  # partial separation
    frame = pd.DataFrame({
        "watch": watch,
        "y": y,
        "split": (["train"] * 200 + ["test"] * 200),
    })
    backend, path = _write(tmp_path, frame)

    result = screen_features(
        backend, path, features=["watch"], target_col="y", split_col="split",
    )

    ks = result.scores["watch"]["ks"]
    if LEAKAGE_WATCH_LOW <= ks < 0.40:
        watch_cols = {feature for feature, _ks, _reason in result.leakage_watch}
        assert "watch" in watch_cols
        # watch-band is informational: the feature is not dropped as leakage.
        assert "watch" not in {c for c, _, _ in result.leakage}
        assert "watch" in {c for c, _ in result.ranked}


def test_screen_no_split_produces_no_split_flags(tmp_path):
    """FS-4: without a usable train/test split, split_shift is empty and ks_train/ks_test
    are absent — never an error."""
    rows = 200
    y = np.array(([0, 1] * 100))
    frame = pd.DataFrame({"f": y.astype(float) + np.linspace(0, 0.5, rows), "y": y})
    backend, path = _write(tmp_path, frame)

    result = screen_features(backend, path, features=["f"], target_col="y")

    assert result.split_shift == ()
    assert "ks_train" not in result.scores["f"]


def test_screen_records_ks_decay_and_flags_only_when_threshold_set(tmp_path):
    """FS-6: per-split KS decay (ks_test/ks_train) is always recorded when a train/test
    split exists; the ks_decay_watch flag only fires when max_ks_decay is set."""
    rows = 400
    rng = np.random.RandomState(11)
    split = np.array((["train"] * 200 + ["test"] * 200))
    y = np.array(([0, 1] * 200))
    # Strong in train, weak in test -> low retention ratio.
    train_signal = np.where(y == 1, 0.7, 0.3) + rng.normal(scale=0.3, size=rows)
    decayer = np.where(split == "train", train_signal, rng.normal(size=rows))
    frame = pd.DataFrame({"decayer": decayer, "y": y, "split": split})
    backend, path = _write(tmp_path, frame)

    display_only = screen_features(
        backend, path, features=["decayer"], target_col="y", split_col="split",
    )
    assert "ks_decay" in display_only.scores["decayer"]
    assert display_only.ks_decay_watch == ()  # default: display-only, no flags

    gated = screen_features(
        backend, path, features=["decayer"], target_col="y", split_col="split",
        max_ks_decay=0.9,
    )
    decay = gated.scores["decayer"]["ks_decay"]
    if decay is not None and decay < 0.9:
        assert "decayer" in {feature for feature, _decay, _reason in gated.ks_decay_watch}
    # gating never drops the feature from the ranked/clean set.
    assert "decayer" in {c for c, _ in gated.ranked}
