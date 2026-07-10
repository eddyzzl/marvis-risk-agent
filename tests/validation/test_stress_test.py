import pandas as pd
import pytest

from marvis.validation.config import ValidationConfig
from marvis.validation.stress_test import (
    load_feature_categories,
    run_stress_test,
)


def _config() -> ValidationConfig:
    return ValidationConfig(
        target_col="y",
        score_col="sample_score",
        split_col="split",
        time_col="apply_month",
        feature_columns=["x1", "x2", "x3"],
        bin_count=5,
    )


class _ProportionalScorer:
    """Scores = x1 contribution. Setting x1 to stress sentinel collapses scores."""
    def score(self, df: pd.DataFrame) -> list[float]:
        scores = []
        for _, row in df.iterrows():
            x1 = row.get("x1")
            if x1 == -9999:
                scores.append(0.5)
            else:
                scores.append(float(min(max(x1, 0.0), 1.0)))
        return scores


def test_categories_loaded_from_dataframe():
    dictionary = pd.DataFrame({
        "特征名": ["x1", "x2", "x3"],
        "类别":   ["征信", "征信", "支付"],
    })
    categories = load_feature_categories(
        dictionary, feature_col="特征名", category_col="类别"
    )
    assert categories == {"征信": ["x1", "x2"], "支付": ["x3"]}


def test_stress_test_per_category_leave_one_out():
    oot = pd.DataFrame({
        "x1": [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 1.0],
        "x2": [0.0] * 10,
        "x3": [0.0] * 10,
        "sample_score": [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 1.0],
        "y": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        "split": ["oot"] * 10,
        "apply_month": ["202507"] * 10,
    })
    categories = {"征信": ["x1"], "支付": ["x2", "x3"]}

    result = run_stress_test(
        oot_sample=oot,
        config=_config(),
        feature_categories=categories,
        input_scorer=_ProportionalScorer(),
    )

    assert result.baseline.sample_count == 10
    assert result.baseline.ks > 0
    by_category = {row.category: row for row in result.per_category}
    # 征信 类置 -9999 后分数全部 0.5 → KS 应明显下降
    assert by_category["征信"].ks_after is not None
    assert by_category["征信"].ks_after < result.baseline.ks
    assert by_category["征信"].ks_delta == pytest.approx(by_category["征信"].ks_after - result.baseline.ks)
    assert by_category["征信"].dropped_features == ["x1"]
    # 支付 类置 -9999 后 x2,x3 对 _ProportionalScorer 没有影响 → 行为应稳定
    assert by_category["支付"].error is None


def test_stress_test_sets_category_features_to_negative_9999():
    class RecordingScorer:
        def __init__(self):
            self.frames: list[pd.DataFrame] = []

        def score(self, df: pd.DataFrame) -> list[float]:
            self.frames.append(df.copy())
            values = []
            for value in df["x1"].tolist():
                values.append(0.5 if value == -9999 or pd.isna(value) else float(value))
            return values

    oot = pd.DataFrame({
        "x1": [0.1, 0.2, 0.8, 0.9],
        "x2": [1.0, 1.0, 1.0, 1.0],
        "sample_score": [0.1, 0.2, 0.8, 0.9],
        "y": [0, 0, 1, 1],
        "split": ["oot"] * 4,
        "apply_month": ["202507"] * 4,
    })
    scorer = RecordingScorer()

    run_stress_test(
        oot_sample=oot,
        config=_config(),
        feature_categories={"征信": ["x1"]},
        input_scorer=scorer,
    )

    assert len(scorer.frames) == 2
    assert scorer.frames[1]["x1"].tolist() == [-9999, -9999, -9999, -9999]


def test_stress_test_checks_cancellation_between_categories():
    class RecordingScorer:
        def __init__(self):
            self.frames: list[pd.DataFrame] = []

        def score(self, df: pd.DataFrame) -> list[float]:
            self.frames.append(df.copy())
            return [float(value) for value in df["sample_score"].tolist()]

    oot = pd.DataFrame({
        "x1": [0.1, 0.2, 0.8, 0.9],
        "x2": [1.0, 1.0, 1.0, 1.0],
        "sample_score": [0.1, 0.2, 0.8, 0.9],
        "y": [0, 0, 1, 1],
        "split": ["oot"] * 4,
        "apply_month": ["202507"] * 4,
    })
    scorer = RecordingScorer()

    def cancellation_check() -> None:
        if len(scorer.frames) >= 2:
            raise KeyboardInterrupt("stress test cancelled")

    with pytest.raises(KeyboardInterrupt, match="stress test cancelled"):
        run_stress_test(
            oot_sample=oot,
            config=_config(),
            feature_categories={"征信": ["x1"], "支付": ["x2"]},
            input_scorer=scorer,
            cancellation_check=cancellation_check,
        )

    assert len(scorer.frames) == 2


def test_stress_test_rejects_empty_oot_sample_with_clear_error():
    empty = pd.DataFrame(
        {
            "x1": [],
            "x2": [],
            "sample_score": [],
            "y": [],
            "split": [],
            "apply_month": [],
        }
    )

    with pytest.raises(ValueError, match="OOT sample is required for stress test"):
        run_stress_test(
            oot_sample=empty,
            config=_config(),
            feature_categories={"征信": ["x1"]},
            input_scorer=_ProportionalScorer(),
        )


def test_stress_test_rejects_non_finite_baseline_scores():
    class BadBaselineScorer:
        def score(self, df: pd.DataFrame) -> list[float]:
            return [0.1, float("nan"), 0.8, 0.9]

    oot = pd.DataFrame({
        "x1": [0.1, 0.2, 0.8, 0.9],
        "x2": [0.2, 0.3, 0.7, 0.8],
        "sample_score": [0.1, 0.2, 0.8, 0.9],
        "y": [0, 0, 1, 1],
        "split": ["oot"] * 4,
        "apply_month": ["202507"] * 4,
    })

    with pytest.raises(ValueError, match="stress baseline scorer returned non-finite"):
        run_stress_test(
            oot_sample=oot,
            config=_config(),
            feature_categories={"征信": ["x1"]},
            input_scorer=BadBaselineScorer(),
        )


def test_stress_test_records_non_finite_category_scores_as_error():
    class BadCategoryScorer:
        def score(self, df: pd.DataFrame) -> list[float]:
            if (df["x1"] == -9999).any():
                return [0.1, float("inf"), 0.8, 0.9]
            return df["sample_score"].astype(float).tolist()

    oot = pd.DataFrame({
        "x1": [0.1, 0.2, 0.8, 0.9],
        "x2": [0.2, 0.3, 0.7, 0.8],
        "sample_score": [0.1, 0.2, 0.8, 0.9],
        "y": [0, 0, 1, 1],
        "split": ["oot"] * 4,
        "apply_month": ["202507"] * 4,
    })

    result = run_stress_test(
        oot_sample=oot,
        config=_config(),
        feature_categories={"征信": ["x1"], "支付": ["x2"]},
        input_scorer=BadCategoryScorer(),
    )

    by_category = {row.category: row for row in result.per_category}
    assert by_category["征信"].error is not None
    assert "non-finite" in by_category["征信"].error
    assert by_category["支付"].error is None
    assert result.status == "partial"


def test_stress_test_marks_all_missing_categories_as_skipped():
    oot = pd.DataFrame({
        "x1": [0.1, 0.2, 0.8, 0.9],
        "sample_score": [0.1, 0.2, 0.8, 0.9],
        "y": [0, 0, 1, 1],
        "split": ["oot"] * 4,
        "apply_month": ["202507"] * 4,
    })

    result = run_stress_test(
        oot_sample=oot,
        config=_config(),
        feature_categories={"支付": ["x_missing"]},
        input_scorer=_ProportionalScorer(),
    )

    assert result.status == "skipped"
    assert result.per_category[0].status == "skipped"


def test_stress_test_marks_completed_categories_partial_when_features_unclassified():
    oot = pd.DataFrame({
        "x1": [0.1, 0.2, 0.8, 0.9],
        "sample_score": [0.1, 0.2, 0.8, 0.9],
        "y": [0, 0, 1, 1],
        "split": ["oot"] * 4,
        "apply_month": ["202507"] * 4,
    })

    result = run_stress_test(
        oot_sample=oot,
        config=_config(),
        feature_categories={"内部特征": ["x1"]},
        input_scorer=_ProportionalScorer(),
        unclassified_features=["BH_A044_C0580"],
        category_source_counts={"notebook": 1, "dictionary": 0, "unresolved": 1},
    )

    assert result.status == "partial"
    assert result.unclassified_features == ["BH_A044_C0580"]
    assert result.category_source_counts["unresolved"] == 1


def test_stress_test_fails_when_no_model_feature_can_be_classified():
    oot = pd.DataFrame({
        "x1": [0.1, 0.2, 0.8, 0.9],
        "sample_score": [0.1, 0.2, 0.8, 0.9],
        "y": [0, 0, 1, 1],
        "split": ["oot"] * 4,
        "apply_month": ["202507"] * 4,
    })

    result = run_stress_test(
        oot_sample=oot,
        config=_config(),
        feature_categories={},
        input_scorer=_ProportionalScorer(),
        unclassified_features=["BH_A044_C0580"],
    )

    assert result.status == "failed"
    assert result.per_category == []
