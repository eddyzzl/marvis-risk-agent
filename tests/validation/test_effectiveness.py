import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from marvis.validation.config import ValidationConfig
from marvis.validation.binning import compute_psi
from marvis.validation.effectiveness import (
    _should_reverse_eval_bins,
    build_effectiveness_result,
    compute_auc,
    compute_bin_tables,
    compute_head_tail_lift,
    compute_monthly_ks,
    compute_monthly_psi,
    compute_overall_ks,
    compute_overall_psi,
    compute_psi_stability_table,
    compute_roc_ks_curves,
    prepare_effectiveness_context,
    run_effectiveness,
)


def test_metrics_imported_effectiveness_postpones_annotations():
    source = Path("marvis/validation/effectiveness.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    first_statement_index = 0
    if (
        module.body
        and isinstance(module.body[0], ast.Expr)
        and isinstance(module.body[0].value, ast.Constant)
        and isinstance(module.body[0].value.value, str)
    ):
        first_statement_index = 1
    future_import = module.body[first_statement_index]

    assert isinstance(future_import, ast.ImportFrom)
    assert future_import.module == "__future__"
    assert any(alias.name == "annotations" for alias in future_import.names)


def _config(bin_count: int = 10) -> ValidationConfig:
    return ValidationConfig(
        target_col="y",
        score_col="sample_score",
        split_col="split",
        time_col="apply_month",
        feature_columns=["x1"],
        bin_count=bin_count,
    )


def _build_sample(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_per_split = 200
    rows = []
    for split, month_start in [("train", 3), ("test", 5), ("oot", 7)]:
        scores = rng.uniform(0, 1, size=n_per_split)
        y = (rng.uniform(0, 1, size=n_per_split) < scores * 0.5).astype(int)
        for s, label in zip(scores, y):
            rows.append({
                "x1": 0.0,
                "sample_score": float(s),
                "y": int(label),
                "split": split,
                "apply_month": f"20250{month_start}",
            })
    return pd.DataFrame(rows)


def _build_ranked_sample() -> pd.DataFrame:
    rows = []
    for split, month in [("train", "202503"), ("test", "202505"), ("oot", "202507")]:
        for index in range(100):
            rows.append({
                "x1": 0.0,
                "sample_score": index / 99,
                "y": int(index >= 80),
                "split": split,
                "apply_month": month,
            })
    return pd.DataFrame(rows)


def test_overall_metrics_cover_all_three_splits():
    sample = _build_sample()
    result = run_effectiveness(sample=sample, config=_config(bin_count=5))
    by_split = {row.split: row for row in result.overall}
    assert set(by_split.keys()) == {"train", "test", "oot"}
    assert set(result.roc_ks_curves.keys()) == {"train", "test", "oot"}
    assert result.roc_ks_curves["train"].ks == pytest.approx(by_split["train"].ks)
    assert by_split["train"].psi_vs_train == pytest.approx(0.0, abs=1e-9)
    assert by_split["test"].psi_vs_train >= 0
    assert all(0.0 <= row.ks <= 1.0 for row in result.overall)


def test_effectiveness_rejects_empty_required_split():
    sample = _build_sample()
    sample = sample[sample["split"] != "test"]

    with pytest.raises(ValueError, match="test"):
        run_effectiveness(sample=sample, config=_config(bin_count=5))


def test_overall_metrics_match_model_analysis_auc_lift_columns():
    sample = _build_ranked_sample()
    result = run_effectiveness(sample=sample, config=_config(bin_count=5))
    by_split = {row.split: row for row in result.overall}

    train = by_split["train"]
    assert train.bad_count == 20
    assert train.auc == pytest.approx(1.0)
    assert train.head_lift_5pct == pytest.approx(5.0)
    assert train.tail_lift_5pct == pytest.approx(0.0)


def test_overall_auc_keeps_declared_positive_score_direction():
    rows = []
    for split, month in [("train", "202503"), ("test", "202505"), ("oot", "202507")]:
        for index in range(100):
            rows.append({
                "x1": 0.0,
                "sample_score": index / 99,
                "y": int(index < 20),
                "split": split,
                "apply_month": month,
            })
    sample = pd.DataFrame(rows)

    result = run_effectiveness(sample=sample, config=_config(bin_count=5))
    train = {row.split: row for row in result.overall}["train"]

    assert train.auc == pytest.approx(0.0)
    # NEW-2 (S1a): head_lift is direction-aware (risk_sign from corr(score, target)),
    # not a hard-coded "highest score = head" -- this sample is negatively correlated
    # (low index/score = high risk, since y = index < 20), so head (high-risk end) is
    # the LOW-score slice, and lift_head/lift_tail flip relative to a naive descending
    # sort. See test_head_tail_lift_flips_for_higher_is_better_score for the isolated
    # regression case.
    assert train.head_lift_5pct == pytest.approx(5.0)
    assert train.tail_lift_5pct == pytest.approx(0.0)


def test_compute_auc_returns_neutral_for_degenerate_labels():
    assert compute_auc([0.1, 0.2, 0.3], [1, 1, 1]) == pytest.approx(0.5)
    assert compute_auc([], []) == pytest.approx(0.5)


def test_roc_ks_curve_population_at_ks_uses_rank_position_not_fpr():
    rows = []
    for split, month in [("train", "202503"), ("test", "202505"), ("oot", "202507")]:
        for index in range(20):
            rows.append({
                "x1": 0.0,
                "sample_score": 1.0 - index / 20,
                "y": int(index == 0),
                "split": split,
                "apply_month": month,
            })
    sample = pd.DataFrame(rows)

    curves = compute_roc_ks_curves(sample=sample, config=_config(bin_count=5))

    assert curves["train"].ks == pytest.approx(1.0)
    assert curves["train"].fpr[1] == pytest.approx(0.0)
    assert curves["train"].population_at_ks == pytest.approx(1 / 20)


def test_split_bin_auto_sort_does_not_reverse_zero_variance_scores():
    sample = pd.DataFrame({
        "sample_score": [0.5, 0.5, 0.5, 0.5],
        "y": [0, 1, 0, 1],
    })

    assert _should_reverse_eval_bins(sample, score_col="sample_score", target_col="y") is False


def test_split_bin_tables_follow_model_analysis_auto_sort_direction():
    rows = []
    for split, month in [("train", "202503"), ("test", "202505"), ("oot", "202507")]:
        for index in range(100):
            rows.append({
                "x1": 0.0,
                "sample_score": index / 99,
                "y": int(index < 20),
                "split": split,
                "apply_month": month,
            })
    sample = pd.DataFrame(rows)

    result = run_effectiveness(sample=sample, config=_config(bin_count=5))
    train_bins = result.bin_tables["train"]

    assert train_bins[0].score_lower > train_bins[-1].score_lower
    assert train_bins[0].bad_rate < train_bins[-1].bad_rate


def test_bin_table_reuses_train_edges_for_each_split():
    sample = _build_sample()
    result = run_effectiveness(sample=sample, config=_config(bin_count=5))
    assert set(result.bin_tables.keys()) == {"train", "test", "oot"}
    edges_train = [row.score_upper for row in result.bin_tables["train"]]
    edges_test = [row.score_upper for row in result.bin_tables["test"]]
    edges_oot = [row.score_upper for row in result.bin_tables["oot"]]
    assert edges_train == edges_test == edges_oot
    assert len(edges_train) == 5


def test_psi_stability_uses_train_test_bins_against_oot_distribution():
    rows = []
    for split, scores in {
        "train": [0.1, 0.2, 0.3, 0.4, 0.5],
        "test": [0.15, 0.25, 0.35, 0.45, 0.55],
        "oot": [0.2, 0.4, 0.6, 0.8, 1.0],
    }.items():
        for score in scores:
            rows.append({
                "x1": 0.0,
                "sample_score": score,
                "y": int(score >= 0.5),
                "split": split,
                "apply_month": "202503",
            })
    sample = pd.DataFrame(rows)

    result = run_effectiveness(sample=sample, config=_config(bin_count=5))

    psi_rows = result.psi_stability_table
    assert len(psi_rows) == 5
    assert sum(row.expected_count for row in psi_rows) == 10
    assert sum(row.actual_count for row in psi_rows) == 5
    assert sum(row.psi for row in psi_rows) > 0


def test_psi_stability_table_uses_shared_compute_psi_smoothing():
    rows = []
    for split, scores in {
        "train": [0.1, 0.2, 0.3, 0.4],
        "test": [0.15, 0.25, 0.35, 0.45],
        "oot": [0.9, 0.95, 0.97, 0.99],
    }.items():
        for score in scores:
            rows.append({
                "x1": 0.0,
                "sample_score": score,
                "y": int(score >= 0.5),
                "split": split,
                "apply_month": "202503",
            })
    sample = pd.DataFrame(rows)

    psi_rows = compute_psi_stability_table(sample=sample, config=_config(bin_count=2))

    assert psi_rows[0].actual_pct == pytest.approx(0.0)
    assert psi_rows[0].psi > 0
    assert sum(row.psi for row in psi_rows) == pytest.approx(
        compute_psi(
            [row.expected_pct for row in psi_rows],
            [row.actual_pct for row in psi_rows],
        )
    )


def test_monthly_metrics_present_for_each_month():
    sample = _build_sample()
    result = run_effectiveness(sample=sample, config=_config(bin_count=5))
    monthly_months = {row.month for row in result.monthly_ks}
    psi_months = {row.month for row in result.monthly_psi}
    assert monthly_months == {"202503", "202505", "202507"}
    assert psi_months == monthly_months


def test_monthly_metrics_include_model_analysis_psi_variants():
    sample = _build_ranked_sample()
    result = run_effectiveness(sample=sample, config=_config(bin_count=5))
    by_month = {row.month: row for row in result.monthly_psi}

    assert by_month["202503"].psi_first_month == pytest.approx(0.0)
    assert by_month["202503"].psi_mom is None
    assert by_month["202503"].psi_mom_reference_month == ""
    assert by_month["202503"].psi_mom_has_calendar_gap is False
    assert by_month["202505"].psi_mom_reference_month == "202503"
    assert by_month["202505"].psi_mom_has_calendar_gap is True
    assert by_month["202507"].psi_last_month == pytest.approx(0.0)
    assert by_month["202507"].psi_mom is not None


def test_monthly_metrics_derive_month_from_datetime_time_col():
    sample = _build_sample()
    sample.loc[sample["split"] == "train", "apply_month"] = "2025/03/15"
    sample.loc[sample["split"] == "test", "apply_month"] = "20250520"
    sample.loc[sample["split"] == "oot", "apply_month"] = "2025-07-01 12:00:00"

    result = run_effectiveness(sample=sample, config=_config(bin_count=5))

    monthly_months = {row.month for row in result.monthly_ks}
    psi_months = {row.month for row in result.monthly_psi}
    assert monthly_months == {"202503", "202505", "202507"}
    assert psi_months == monthly_months


def test_effectiveness_can_be_built_from_separate_ks_psi_and_binning_steps():
    sample = _build_sample()
    config = _config(bin_count=5)
    context = prepare_effectiveness_context(sample=sample, config=config)

    overall = compute_overall_ks(sample=sample, config=config)
    monthly_ks = compute_monthly_ks(sample=sample, config=config)
    overall = compute_overall_psi(
        sample=sample,
        config=config,
        context=context,
        overall=overall,
    )
    monthly_psi = compute_monthly_psi(sample=sample, config=config, context=context)
    bin_tables = compute_bin_tables(sample=sample, config=config, context=context)
    psi_stability_table = compute_psi_stability_table(sample=sample, config=config)
    roc_ks_curves = compute_roc_ks_curves(sample=sample, config=config)
    separate = build_effectiveness_result(
        overall=overall,
        monthly_ks=monthly_ks,
        monthly_psi=monthly_psi,
        bin_tables=bin_tables,
        psi_stability_table=psi_stability_table,
        roc_ks_curves=roc_ks_curves,
    )

    combined = run_effectiveness(sample=sample, config=config)
    assert separate == combined


def test_head_tail_lift_flips_for_higher_is_better_score():
    """NEW-2 (S1a) core regression: compute_head_tail_lift must be direction-aware,
    like feature/metrics.py::head_tail_lift, not hard-coded to "highest score = head".
    Construct a higher_is_better sample (score negatively correlated with bad) and
    assert the high-risk end (head) lands on the LOW-score slice."""
    n = 100
    scores = np.arange(n, dtype=float) / (n - 1)
    labels = (np.arange(n) < 20).astype(int)  # low score/index -> bad -> higher_is_better

    head_lift, tail_lift = compute_head_tail_lift(scores, labels, fraction=0.05)

    assert head_lift == pytest.approx(5.0)
    assert tail_lift == pytest.approx(0.0)


def test_head_tail_lift_matches_naive_descending_sort_for_higher_is_riskier_score():
    """Sanity check the non-flipped case is unaffected: score positively correlated
    with bad (higher_is_riskier) keeps head = highest score, matching the pre-NEW-2
    behavior for this direction."""
    n = 100
    scores = np.arange(n, dtype=float) / (n - 1)
    labels = (np.arange(n) >= 80).astype(int)  # high score/index -> bad -> higher_is_riskier

    head_lift, tail_lift = compute_head_tail_lift(scores, labels, fraction=0.05)

    assert head_lift == pytest.approx(5.0)
    assert tail_lift == pytest.approx(0.0)


def test_head_tail_lift_returns_none_for_empty_scores():
    head_lift, tail_lift = compute_head_tail_lift(np.array([]), np.array([]))
    assert head_lift is None
    assert tail_lift is None
