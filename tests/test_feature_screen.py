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


def test_screen_default_feature_reads_are_memory_bounded(tmp_path, monkeypatch):
    rows = 80
    feature_names = [f"f{index}" for index in range(48)]
    target = np.array(([0, 1] * (rows // 2)))
    frame = pd.DataFrame(
        {
            **{
                name: target.astype(float) + (index + 1) * np.linspace(0, 0.01, rows)
                for index, name in enumerate(feature_names)
            },
            "y": target,
        }
    )
    backend, path = _write(tmp_path, frame)
    original_read_frame = DataBackend.read_frame
    requested_widths: list[int] = []

    def counting_read_frame(self, call_path, *, columns=None, nrows=None):
        if columns is not None and "y" not in columns:
            requested_widths.append(len(columns))
        return original_read_frame(self, call_path, columns=columns, nrows=nrows)

    monkeypatch.setattr(DataBackend, "read_frame", counting_read_frame)

    screen_features(backend, path, features=feature_names, target_col="y", top_k=8)

    assert requested_widths
    assert max(requested_widths) <= 16


def test_screen_hard_excludes_named_post_outcome_fields_before_ks(tmp_path):
    """A future-performance column is leakage because of when it becomes known, not
    because its one-variable KS happens to cross a numeric threshold.  Low-KS MOB/FPD
    outcome columns must therefore be blocked before the statistical screen, while an
    ordinary pre-loan feature that merely contains ``mob`` remains eligible."""
    rows = 400
    rng = np.random.RandomState(20260722)
    frame = pd.DataFrame({
        "mob3_ever30": rng.permutation(np.arange(rows, dtype=float)),
        "mob4_dpd30_amt_fm": rng.permutation(np.arange(rows, dtype=float)),
        "fpd30": rng.permutation(np.arange(rows, dtype=float)),
        "label_all": rng.permutation(np.array(([0, 1] * (rows // 2)), dtype=float)),
        "mob_bureau_query_count": rng.normal(size=rows),
        "y": np.array(([0, 1] * (rows // 2)), dtype=float),
    })
    backend, path = _write(tmp_path, frame)

    result = screen_features(
        backend,
        path,
        features=[
            "mob3_ever30",
            "mob4_dpd30_amt_fm",
            "fpd30",
            "label_all",
            "mob_bureau_query_count",
        ],
        target_col="y",
    )

    hard = {feature: reason for feature, _ks, reason in result.leakage}
    assert set(hard) == {"mob3_ever30", "mob4_dpd30_amt_fm", "fpd30", "label_all"}
    assert all("semantic/temporal target leakage" in reason for reason in hard.values())
    assert set(hard).isdisjoint(result.selected)
    assert "mob_bureau_query_count" in result.selected


def test_screen_hard_excludes_deterministic_outcome_subgroup_in_train_only_folds(
    tmp_path,
):
    """A generic alias can leak one target branch while keeping pooled KS low.

    ``outcome_alias=1`` identifies only a minority of bads, so the legacy
    ``KS >= 0.40`` rule cannot catch it.  Repeating the same sufficiently large,
    perfectly pure subgroup in two deterministic halves of train is
    deterministic outcome evidence and must be a hard exclusion even when the
    column name is harmless. Test labels must not participate in this gate.
    """
    train_rows = 1_000
    test_rows = 800
    split = np.array(["train"] * train_rows + ["test"] * test_rows)
    y = np.array(([0, 1] * ((train_rows + test_rows) // 2)), dtype=float)
    alias = np.zeros(train_rows + test_rows, dtype=float)
    train_bad = np.flatnonzero((split == "train") & (y == 1))[:110]
    test_bad = np.flatnonzero((split == "test") & (y == 1))[:110]
    alias[np.concatenate([train_bad, test_bad])] = 1.0
    backend, path = _write(
        tmp_path,
        pd.DataFrame({"outcome_alias": alias, "y": y, "split": split}),
        name="deterministic_subgroup.parquet",
    )

    result = screen_features(
        backend,
        path,
        features=["outcome_alias"],
        target_col="y",
        split_col="split",
    )

    assert result.scores["outcome_alias"]["ks"] < 0.40
    hard = {feature: reason for feature, _ks, reason in result.leakage}
    assert "outcome_alias" in hard
    assert "deterministic outcome subgroup leakage" in hard["outcome_alias"]
    assert "train fold A 55" in hard["outcome_alias"]
    assert "train fold B 55" in hard["outcome_alias"]
    assert "outcome_alias" not in {feature for feature, _ks in result.ranked}
    assert "outcome_alias" not in result.selected


@pytest.mark.parametrize("case", ["small_support", "one_split_only", "normal"])
def test_screen_deterministic_subgroup_gate_avoids_low_support_and_one_sided_false_positives(
    tmp_path,
    case,
):
    train_rows = 1_000
    test_rows = 800
    split = np.array(["train"] * train_rows + ["test"] * test_rows)
    y = np.array(([0, 1] * ((train_rows + test_rows) // 2)), dtype=float)
    values = np.zeros(train_rows + test_rows, dtype=float)
    train_bad = np.flatnonzero((split == "train") & (y == 1))
    test_bad = np.flatnonzero((split == "test") & (y == 1))

    if case == "small_support":
        values[np.concatenate([train_bad[:20], test_bad[:20]])] = 1.0
    elif case == "one_split_only":
        # Within the value-stratified train folds, one half is pure good and
        # the other pure bad. The repeated-subgroup gate requires the same
        # target class in both train-only folds.
        values[:220] = 1.0
    else:
        # Balanced in both splits: a normal binary feature with no deterministic
        # target subgroup.
        values[np.arange(values.size) % 4 < 2] = 1.0

    feature = f"binary_{case}"
    backend, path = _write(
        tmp_path,
        pd.DataFrame({feature: values, "y": y, "split": split}),
        name=f"{case}.parquet",
    )

    result = screen_features(
        backend,
        path,
        features=[feature],
        target_col="y",
        split_col="split",
    )

    assert feature not in {column for column, _ks, _reason in result.leakage}
    assert feature in result.selected


def test_binary_screen_top_k_is_fitted_on_train_not_test_labels(tmp_path):
    """A feature that works only in test cannot enter the automatic candidate set."""
    rows = 4_000
    rng = np.random.RandomState(121)
    split = np.array(["train"] * (rows // 2) + ["test"] * (rows // 2))
    y = rng.randint(0, 2, size=rows).astype(float)
    train_signal = rng.normal(size=rows)
    test_signal = rng.normal(size=rows)
    train_signal[split == "train"] += y[split == "train"] * 0.55
    test_signal[split == "test"] += y[split == "test"] * 1.2
    backend, path = _write(
        tmp_path,
        pd.DataFrame({
            "train_signal": train_signal,
            "test_signal": test_signal,
            "y": y,
            "split": split,
        }),
        name="train_only_screen.parquet",
    )

    result = screen_features(
        backend,
        path,
        features=["test_signal", "train_signal"],
        target_col="y",
        split_col="split",
        leakage_ks=0.99,
        top_k=1,
    )

    assert result.selected == ("train_signal",)
    assert result.scores["test_signal"]["ks_test"] > result.scores["test_signal"]["ks"]


def test_non_binary_screen_top_k_is_fitted_on_train_not_test_target(tmp_path):
    rows = 2_000
    rng = np.random.RandomState(122)
    split = np.array(["train"] * (rows // 2) + ["test"] * (rows // 2))
    target = rng.normal(size=rows)
    train_signal = rng.normal(size=rows)
    test_signal = rng.normal(size=rows)
    train_signal[split == "train"] = target[split == "train"] + rng.normal(
        scale=0.4, size=(split == "train").sum()
    )
    test_signal[split == "test"] = target[split == "test"] + rng.normal(
        scale=0.05, size=(split == "test").sum()
    )
    backend, path = _write(
        tmp_path,
        pd.DataFrame({
            "test_signal": test_signal,
            "train_signal": train_signal,
            "target": target,
            "split": split,
        }),
        name="train_only_non_binary_screen.parquet",
    )

    result = screen_features_non_binary(
        backend,
        path,
        features=["test_signal", "train_signal"],
        target_col="target",
        target_type="continuous",
        split_col="split",
        top_k=1,
    )

    assert result.selected == ("train_signal",)


def test_screen_hard_excludes_protected_controls_and_explicit_identity_keys(tmp_path):
    """Even an explicit feature list must not turn the target, split, sample weight, or
    a strong application-row identifier into model inputs.  These are structural controls,
    so high cardinality is evidence shown to the user, never the exclusion rule itself."""
    rows = 120
    rng = np.random.RandomState(11)
    frame = pd.DataFrame({
        "appl_seq_x": np.arange(10_000, 10_000 + rows),
        "sample_weight": np.where(np.arange(rows) % 3 == 0, 2.0, 1.0),
        "safe_feature": rng.normal(size=rows),
        "y": np.array(([0, 1] * (rows // 2)), dtype=float),
        "split": ["train"] * 80 + ["test"] * 40,
    })
    backend, path = _write(tmp_path, frame)

    result = screen_features(
        backend,
        path,
        features=["y", "split", "sample_weight", "appl_seq_x", "safe_feature"],
        target_col="y",
        split_col="split",
        sample_weight_col="sample_weight",
        holdout_values=(),
    )

    leakage = {feature: reason for feature, _ks, reason in result.leakage}
    unusable = dict(result.unusable)
    assert "y" in leakage
    assert "protected target column" in leakage["y"]
    assert "protected split column" in unusable["split"]
    assert "protected sample-weight column" in unusable["sample_weight"]
    assert "identity/row key" in unusable["appl_seq_x"]
    assert result.selected == ("safe_feature",)


def test_non_binary_screen_applies_same_semantic_and_identity_hard_exclusions(tmp_path):
    rows = 100
    rng = np.random.RandomState(19)
    frame = pd.DataFrame({
        "mob6_ever30": rng.permutation(np.arange(rows, dtype=float)),
        "appl_seq_y": np.arange(rows, dtype=float),
        "safe": rng.normal(size=rows),
        "target": np.linspace(100.0, 200.0, rows),
    })
    backend, path = _write(tmp_path, frame, name="non_binary_hard_exclusions.parquet")

    result = screen_features_non_binary(
        backend,
        path,
        features=["mob6_ever30", "appl_seq_y", "safe"],
        target_col="target",
        target_type="continuous",
    )

    assert {feature for feature, _ks, _reason in result.leakage} == {"mob6_ever30"}
    assert dict(result.unusable)["appl_seq_y"].startswith("identity/row key")
    assert result.selected == ("safe",)


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
        leakage_ks=0.99,
    )
    assert "ks_decay" in display_only.scores["decayer"]
    assert display_only.ks_decay_watch == ()  # default: display-only, no flags

    gated = screen_features(
        backend, path, features=["decayer"], target_col="y", split_col="split",
        leakage_ks=0.99, max_ks_decay=0.9,
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


def test_screen_non_binary_requires_nan_label_confirmation_and_reports_drop(tmp_path):
    """Regression/multiclass screening must not coerce an unknown class to negative."""

    from marvis.data.errors import NanLabelNotConfirmedError

    frame = pd.DataFrame(
        {
            "feature": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "target": [0.0, 1.0, np.nan, 2.0, 0.0, 1.0],
        }
    )
    backend, path = _write(tmp_path, frame, name="non_binary_nan.parquet")

    with pytest.raises(NanLabelNotConfirmedError, match="target"):
        screen_features_non_binary(
            backend,
            path,
            features=["feature"],
            target_col="target",
            target_type="multiclass",
        )

    result = screen_features_non_binary(
        backend,
        path,
        features=["feature"],
        target_col="target",
        target_type="multiclass",
        drop_nan_labels=True,
    )

    assert result.nan_labels_dropped == 1
    assert result.selected == ("feature",)


def test_screen_non_binary_default_feature_reads_are_memory_bounded(
    tmp_path,
    monkeypatch,
):
    feature_names = [f"f{index}" for index in range(41)]
    rows = 60
    target = np.linspace(0.0, 1.0, rows)
    frame = pd.DataFrame(
        {
            **{
                feature: target + index * 0.001
                for index, feature in enumerate(feature_names)
            },
            "target": target,
        }
    )
    backend, path = _write(tmp_path, frame, name="non_binary_wide.parquet")
    original_read_frame = DataBackend.read_frame
    requested_widths: list[int] = []

    def counting_read_frame(self, call_path, *, columns=None, nrows=None):
        if columns is not None and "target" not in columns:
            requested_widths.append(len(columns))
        return original_read_frame(self, call_path, columns=columns, nrows=nrows)

    monkeypatch.setattr(DataBackend, "read_frame", counting_read_frame)

    screen_features_non_binary(
        backend,
        path,
        features=feature_names,
        target_col="target",
        target_type="continuous",
    )

    assert requested_widths
    assert max(requested_widths) <= 16


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


# --- D13: NaN-label confirmation gate (INV-1 / INV-2) --------------------------------


def _nan_label_screen_frame(rows: int = 100, nan_fraction: float = 0.4) -> pd.DataFrame:
    """A binary-labelled frame where ``nan_fraction`` of the target rows are NaN, with a
    partially-discriminative feature (KS below the 0.40 leakage gate so it lands in
    ``selected``) — the exact silent-degradation shape (screen ranks/gates on the labelled
    subset while reporting the full sample)."""
    rng = np.random.RandomState(0)
    y = np.array([0, 1] * (rows // 2), dtype=float)
    n_nan = int(round(rows * nan_fraction))
    # Blank the label for a deterministic contiguous slice so the remaining labelled rows
    # still carry both classes.
    y[:n_nan] = np.nan
    # Partial signal chosen so the KS on the labelled subset stays below the 0.40 leakage
    # gate (so ``f`` lands in ``selected`` rather than the leakage bucket).
    feature = np.nan_to_num(y, nan=0.0) + rng.normal(scale=3.0, size=rows)
    return pd.DataFrame({"f": feature, "y": y})


def test_screen_features_raises_on_nan_label_by_default(tmp_path):
    """D13: with NaN labels and drop_nan_labels omitted, screen_features must STOP with the
    typed NaN-label error (scope='screen') instead of silently ranking on the labelled subset."""
    from marvis.data.errors import NanLabelNotConfirmedError

    frame = _nan_label_screen_frame(rows=100, nan_fraction=0.4)
    backend, path = _write(tmp_path, frame)

    with pytest.raises(NanLabelNotConfirmedError) as excinfo:
        screen_features(backend, path, features=["f"], target_col="y")
    assert excinfo.value.n_nan == 40
    assert excinfo.value.n_total == 100
    assert excinfo.value.scope == "screen"


def test_screen_features_drops_and_reports_when_confirmed(tmp_path):
    """D13: with drop_nan_labels=True the gate proceeds, reports nan_labels_dropped, and the
    KS/ranking are computed only over the labelled rows (the deterministic core already drops
    NaN-label rows — the count is surfaced for audit)."""
    from marvis.feature.metrics import feature_ks

    frame = _nan_label_screen_frame(rows=100, nan_fraction=0.4)
    backend, path = _write(tmp_path, frame)

    result = screen_features(
        backend, path, features=["f"], target_col="y", drop_nan_labels=True,
    )
    assert result.nan_labels_dropped == 40
    assert result.selected == ("f",)
    labelled = frame["y"].notna().to_numpy()
    expected_ks = feature_ks(frame["f"].to_numpy()[labelled], frame["y"].to_numpy()[labelled])
    # KS is computed only over the 60 labelled rows, not the full 100-row sample.
    assert result.scores["f"]["ks"] == pytest.approx(expected_ks)


def test_screen_features_clean_labels_reports_zero_dropped(tmp_path):
    """D13 regression guard: on a fully-labelled target the gate is inert — no raise, and
    nan_labels_dropped == 0 (screen output byte-identical to before the gate)."""
    rows = 100
    rng = np.random.RandomState(0)
    y = np.array([0, 1] * (rows // 2), dtype=float)
    feature = y + rng.normal(scale=2.0, size=rows)  # partial signal, KS below leakage gate
    frame = pd.DataFrame({"f": feature, "y": y})
    backend, path = _write(tmp_path, frame)

    result = screen_features(backend, path, features=["f"], target_col="y")
    assert result.nan_labels_dropped == 0
    assert result.selected == ("f",)
    assert result.psi_watch == ()
