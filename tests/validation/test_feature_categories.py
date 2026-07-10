import pandas as pd

from marvis.validation.feature_categories import resolve_feature_categories


def test_notebook_category_keeps_transformed_feature_name():
    dictionary = pd.DataFrame({"特征名": ["BH_A044"], "类别": ["睿智"]})

    result = resolve_feature_categories(
        model_features=[("BH_A044_C0580", "睿智")],
        dictionary=dictionary,
        feature_col="特征名",
        category_col="类别",
    )

    assert result.per_category == {"睿智": ["BH_A044_C0580"]}
    assert result.unclassified_features == []
    assert result.source_counts == {
        "notebook": 1,
        "dictionary": 0,
        "unresolved": 0,
    }


def test_dictionary_only_fills_exact_feature_with_empty_notebook_category():
    dictionary = pd.DataFrame({"特征名": ["income"], "类别": ["内部特征"]})

    result = resolve_feature_categories(
        model_features=[("income", "")],
        dictionary=dictionary,
        feature_col="特征名",
        category_col="类别",
    )

    assert result.per_category == {"内部特征": ["income"]}
    assert result.source_counts == {
        "notebook": 0,
        "dictionary": 1,
        "unresolved": 0,
    }


def test_dictionary_does_not_fuzzy_match_transformed_feature():
    dictionary = pd.DataFrame({"特征名": ["BH_A044"], "类别": ["睿智"]})

    result = resolve_feature_categories(
        model_features=[("BH_A044_C0580", "")],
        dictionary=dictionary,
        feature_col="特征名",
        category_col="类别",
    )

    assert result.per_category == {}
    assert result.unclassified_features == ["BH_A044_C0580"]
    assert result.source_counts["unresolved"] == 1


def test_conflicting_notebook_categories_are_reported():
    result = resolve_feature_categories(
        model_features=[("income", "内部特征"), ("income", "征信")],
        dictionary=None,
        feature_col="特征名",
        category_col="类别",
    )

    assert [
        (row.feature, row.categories, row.source) for row in result.conflicts
    ] == [("income", ("内部特征", "征信"), "notebook")]
    assert result.per_category == {}


def test_conflicting_dictionary_categories_are_reported_for_empty_notebook_category():
    dictionary = pd.DataFrame(
        {
            "特征名": ["income", "income"],
            "类别": ["内部特征", "征信"],
        }
    )

    result = resolve_feature_categories(
        model_features=[("income", None)],
        dictionary=dictionary,
        feature_col="特征名",
        category_col="类别",
    )

    assert [
        (row.feature, row.categories, row.source) for row in result.conflicts
    ] == [("income", ("内部特征", "征信"), "dictionary")]
    assert result.per_category == {}


def test_resolver_preserves_model_and_category_order_without_duplicates():
    result = resolve_feature_categories(
        model_features=[
            ("x2", "征信"),
            ("x1", "内部特征"),
            ("x3", "征信"),
            ("x2", "征信"),
        ],
        dictionary=None,
        feature_col="特征名",
        category_col="类别",
    )

    assert list(result.per_category) == ["征信", "内部特征"]
    assert result.per_category == {"征信": ["x2", "x3"], "内部特征": ["x1"]}
