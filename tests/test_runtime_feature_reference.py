import math

import pytest

from marvis.orchestrator.eval.runtime_feature_reference import feature_reference


def test_tied_scores_have_half_pair_auc_and_complete_group_ks():
    row = feature_reference([1, 1, 2, 2], [0, 1, 0, 1], feature="x")
    assert row["auc"] == .5 and row["ks"] == 0 and row["iv"] == 0
    assert row["std"] == pytest.approx(math.sqrt(1 / 3))
    assert row["q25"] == 1 and row["q75"] == 2


def test_perfect_direction_reverse_and_smoothed_two_group_iv():
    row = feature_reference([2, 2, 1, 1], [0, 0, 1, 1], feature="x")
    assert row["auc"] == row["ks"] == 1
    assert row["iv"] == round(4 / 3 * math.log(5), 6)


def test_missing_values_form_iv_bucket_but_are_excluded_from_auc_ks_and_quality_denominators():
    row = feature_reference([0, 2, None, float("inf")], [0, 1, 1, 0], feature="x")
    assert row["valid_count"] == 2 and row["coverage"] == row["missing_rate"] == .5
    assert row["mean"] == 1 and row["q25"] == .5 and row["q75"] == 1.5
    assert row["auc"] == row["ks"] == 1
    assert row["zero_rate"] == .5
    assert row["iv"] == round(4 / 7 * math.log(3), 6)


def test_constant_column_never_has_perfect_ks_or_iv():
    row = feature_reference([7] * 6, [0, 1, 0, 1, 0, 1], feature="constant")
    assert row["iv"] == row["ks"] == row["std"] == 0
    assert row["auc"] == .5 and row["mode_rate"] == 1


def test_default_fixture_metrics_are_recomputed_from_rows():
    labels = [i % 2 for i in range(80)]
    row = feature_reference([i + 20 * labels[i] for i in range(80)], labels, feature="balance")
    assert row["iv"] == 1.047541 and row["ks"] == pytest.approx(.275)
    assert row["auc"] == .728125 and row["mean"] == 49.5
    assert row["std"] == pytest.approx(25.52239026939466)
    assert row["q25"] == 29.75 and row["q75"] == 69.25


def test_selected_quality_only_does_not_require_labels_or_add_supervised_metrics():
    row = feature_reference([0, 1, None], None, feature="x", selected=["coverage"])
    assert not ({"iv", "ks", "auc"} & row.keys())
    assert row["valid_count"] == 2
