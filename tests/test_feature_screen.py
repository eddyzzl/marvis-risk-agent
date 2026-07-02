"""Unit tests for marvis.feature.screen library-level behavior (FS-4/FS-6/FS-7/FS-10)."""
import numpy as np
import pandas as pd
import pytest

from marvis.data.backend import DataBackend
from marvis.feature.binning import equal_frequency_edges
from marvis.feature.metrics import DEFAULT_IV_BINS, feature_psi
from marvis.feature.screen import (
    LEAKAGE_WATCH_LOW,
    SPLIT_SHIFT_THRESHOLD,
    screen_features,
    screen_features_non_binary,
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


def test_screen_surfaces_coverage_and_low_coverage_note_without_changing_rank(tmp_path):
    """FS-7: scores carry explicit coverage (1 - missing_rate); a low-coverage yet
    discriminative column gets a 'missing is informative' note, and ranking stays KS-based."""
    rows = 400
    rng = np.random.RandomState(5)
    y = np.array(([0, 1] * 200))
    # full: high coverage, moderate signal.
    full = np.where(y == 1, 0.6, 0.4) + rng.normal(scale=0.3, size=rows)
    # sparse: ~70% missing but where present, "missing is informative" -> strong signal.
    sparse = np.where(y == 1, 5.0, -5.0)
    mask_missing = rng.rand(rows) < 0.7
    sparse = sparse.astype(float)
    sparse[mask_missing] = np.nan
    frame = pd.DataFrame({"full": full, "sparse": sparse, "y": y})
    backend, path = _write(tmp_path, frame)

    result = screen_features(backend, path, features=["full", "sparse"], target_col="y")

    # coverage is explicit and equals 1 - missing_rate.
    for col in ("full", "sparse"):
        assert result.scores[col]["coverage"] == 1.0 - result.scores[col]["missing_rate"]
    assert result.scores["sparse"]["coverage"] < 0.5
    # sparse is discriminative where present -> gets the low-coverage note.
    assert "note" in result.scores["sparse"]
    # high-coverage column gets no note.
    assert "note" not in result.scores["full"]
    # ranking is still KS descending (annotation must not reorder).
    ks_seq = [ks for _c, ks in result.ranked]
    assert ks_seq == sorted(ks_seq, reverse=True)


def test_screen_records_iv_binning_convention(tmp_path):
    """FS-9: the IV enrichment step always records which binning convention produced it
    (equal-frequency DEFAULT_IV_BINS bins), so callers can tell IV values from different
    tools/paths apart instead of silently comparing incompatible bin counts."""
    rows = 200
    rng = np.random.RandomState(2)
    y = np.array(([0, 1] * 100))
    # Moderate signal (KS well under the 0.40 leakage gate) so the feature reaches the
    # IV enrichment step (selected), not the leakage bucket.
    moderate = np.where(y == 1, 0.55, 0.45) + rng.normal(scale=0.3, size=rows)
    frame = pd.DataFrame({"f": moderate, "y": y})
    backend, path = _write(tmp_path, frame)

    result = screen_features(backend, path, features=["f"], target_col="y")

    assert "f" in result.selected
    assert result.scores["f"]["iv_binning"] == "equal_frequency_10"


def test_screen_non_binary_continuous_ranks_by_spearman(tmp_path):
    """FS-10: continuous-target screening ranks clean features by |Spearman| descending
    instead of leaving top_k a slice of input order — the weakly-associated feature
    listed FIRST in `features` must not out-rank the strongly-associated one listed later."""
    rows = 100
    target = np.linspace(0, 1, rows)
    strong = target + np.random.RandomState(1).normal(scale=0.02, size=rows)  # |corr| ~ 1
    weak = np.random.RandomState(2).permutation(rows).astype(float)           # |corr| ~ 0
    frame = pd.DataFrame({"weak": weak, "strong": strong, "target": target})
    backend, path = _write(tmp_path, frame, name="non_binary.parquet")

    result = screen_features_non_binary(
        backend, path, features=["weak", "strong"], target_col="target",
        target_type="continuous",
    )

    assert [c for c, _ks in result.ranked] == ["strong", "weak"]
    assert result.scores["strong"]["assoc_score"] > result.scores["weak"]["assoc_score"]
    assert result.scores["strong"]["ks"] is None  # ks stays None for non-binary (unchanged)
    # top_k now picks the actually-associated feature, not whichever came first in input.
    capped = screen_features_non_binary(
        backend, path, features=["weak", "strong"], target_col="target",
        target_type="continuous", top_k=1,
    )
    assert capped.selected == ("strong",)


def test_screen_non_binary_multiclass_ranks_by_one_vs_rest_auc(tmp_path):
    """FS-10: multiclass screening ranks by one-vs-rest AUC macro-average descending."""
    rows = 150
    rng = np.random.RandomState(4)
    target = np.array([0, 1, 2] * (rows // 3))
    # informative: distinct level per class -> high macro AUC.
    informative = target.astype(float) + rng.normal(scale=0.1, size=rows)
    # noise: unrelated to class.
    noise = rng.normal(size=rows)
    frame = pd.DataFrame({"noise": noise, "informative": informative, "target": target})
    backend, path = _write(tmp_path, frame, name="non_binary_mc.parquet")

    result = screen_features_non_binary(
        backend, path, features=["noise", "informative"], target_col="target",
        target_type="multiclass",
    )

    assert [c for c, _ks in result.ranked][0] == "informative"
    assert result.scores["informative"]["assoc_score"] > result.scores["noise"]["assoc_score"]


def test_screen_non_binary_ties_preserve_input_order(tmp_path):
    """FS-10: a stable sort on tied association scores must not reorder input — regression
    guard for the existing continuous-screen ranked-order test expectations."""
    target = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    good1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])   # |Spearman| == 1.0
    good2 = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])   # |Spearman| == 1.0 (tied with good1)
    frame = pd.DataFrame({"good1": good1, "good2": good2, "target": target})
    backend, path = _write(tmp_path, frame, name="non_binary_tie.parquet")

    result = screen_features_non_binary(
        backend, path, features=["good1", "good2"], target_col="target",
        target_type="continuous",
    )

    assert [c for c, _ks in result.ranked] == ["good1", "good2"]


def test_screen_records_psi_split_matching_direct_feature_psi_computation(tmp_path):
    """DOM-7b: scores['psi_split'] (train vs the holdout split) matches feature_psi
    computed directly off the same train/holdout arrays with the same train-derived
    equal-frequency edges — no drift between the screen path and the primitive."""
    rng = np.random.RandomState(7)
    train_values = rng.normal(loc=0.0, scale=1.0, size=200)
    # oot: shifted distribution -> non-trivial PSI.
    oot_values = rng.normal(loc=1.5, scale=1.0, size=200)
    values = np.concatenate([train_values, oot_values])
    y = np.concatenate([
        rng.randint(0, 2, size=200),
        np.full(200, np.nan),  # OOT unlabeled -- PSI must not require labels.
    ])
    split = np.array((["train"] * 200) + (["oot"] * 200))
    frame = pd.DataFrame({"drifter": values, "y": y, "split": split})
    backend, path = _write(tmp_path, frame)

    result = screen_features(
        backend, path, features=["drifter"], target_col="y", split_col="split",
    )

    assert "psi_split" in result.scores["drifter"]
    psi_split = result.scores["drifter"]["psi_split"]
    assert psi_split is not None
    edges = equal_frequency_edges(train_values, DEFAULT_IV_BINS)
    expected = feature_psi(train_values, oot_values, edges)
    assert psi_split == pytest.approx(expected)


def test_screen_psi_watch_flags_only_when_max_feature_psi_set(tmp_path):
    """DOM-7b: psi_watch stays empty by default (display-only) and only fires when the
    caller opts in via max_feature_psi — mirrors the ks_decay_watch/max_ks_decay pattern.
    Gating never drops the feature from ranked/clean."""
    rng = np.random.RandomState(9)
    train_values = rng.normal(loc=0.0, scale=1.0, size=200)
    oot_values = rng.normal(loc=3.0, scale=1.0, size=200)  # large shift -> high PSI
    values = np.concatenate([train_values, oot_values])
    y = np.concatenate([rng.randint(0, 2, size=200), rng.randint(0, 2, size=200)])
    split = np.array((["train"] * 200) + (["oot"] * 200))
    frame = pd.DataFrame({"drifter": values, "y": y.astype(float), "split": split})
    backend, path = _write(tmp_path, frame)

    display_only = screen_features(
        backend, path, features=["drifter"], target_col="y", split_col="split",
    )
    assert display_only.psi_watch == ()
    psi_split = display_only.scores["drifter"]["psi_split"]
    assert psi_split is not None and psi_split > 0

    gated = screen_features(
        backend, path, features=["drifter"], target_col="y", split_col="split",
        max_feature_psi=0.05,
    )
    watch_cols = {feature for feature, _psi, _reason in gated.psi_watch}
    assert "drifter" in watch_cols
    # gating never drops the feature from the ranked/clean set (informational only).
    assert "drifter" in {c for c, _ in gated.ranked}


def test_screen_no_holdout_split_produces_no_psi(tmp_path):
    """DOM-7b: without a usable train/holdout split, psi_split is absent from scores and
    psi_watch stays empty — never an error (mirrors test_screen_no_split_produces_no_split_flags)."""
    rows = 200
    y = np.array(([0, 1] * 100))
    frame = pd.DataFrame({"f": y.astype(float) + np.linspace(0, 0.5, rows), "y": y})
    backend, path = _write(tmp_path, frame)

    result = screen_features(backend, path, features=["f"], target_col="y")

    assert "psi_split" not in result.scores["f"]
    assert result.psi_watch == ()
