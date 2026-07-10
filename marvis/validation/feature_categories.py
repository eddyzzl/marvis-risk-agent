from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class FeatureCategoryConflict:
    feature: str
    categories: tuple[str, ...]
    source: str


@dataclass(frozen=True)
class FeatureCategoryResolution:
    per_category: dict[str, list[str]]
    unclassified_features: list[str]
    conflicts: list[FeatureCategoryConflict]
    source_counts: dict[str, int]


def resolve_feature_categories(
    *,
    model_features: Sequence[tuple[str, str | None]],
    dictionary: pd.DataFrame | None,
    feature_col: str,
    category_col: str,
) -> FeatureCategoryResolution:
    """Resolve final model features without guessing transformed names."""
    feature_order: list[str] = []
    notebook_categories: dict[str, list[str]] = {}
    for raw_feature, raw_category in model_features:
        feature = _text(raw_feature)
        if not feature:
            continue
        if feature not in notebook_categories:
            notebook_categories[feature] = []
            feature_order.append(feature)
        category = _text(raw_category)
        if category and category not in notebook_categories[feature]:
            notebook_categories[feature].append(category)

    assignments: dict[str, tuple[str, str]] = {}
    conflicts: list[FeatureCategoryConflict] = []
    for feature in feature_order:
        categories = notebook_categories[feature]
        if len(categories) == 1:
            assignments[feature] = (categories[0], "notebook")
        elif len(categories) > 1:
            conflicts.append(
                FeatureCategoryConflict(
                    feature=feature,
                    categories=tuple(categories),
                    source="notebook",
                )
            )

    unresolved = [
        feature
        for feature in feature_order
        if feature not in assignments
        and not any(row.feature == feature for row in conflicts)
    ]
    dictionary_categories = _dictionary_categories(
        dictionary,
        unresolved_features=unresolved,
        feature_col=feature_col,
        category_col=category_col,
    )
    for feature in unresolved:
        categories = dictionary_categories.get(feature, [])
        if len(categories) == 1:
            assignments[feature] = (categories[0], "dictionary")
        elif len(categories) > 1:
            conflicts.append(
                FeatureCategoryConflict(
                    feature=feature,
                    categories=tuple(categories),
                    source="dictionary",
                )
            )

    per_category: dict[str, list[str]] = {}
    for feature in feature_order:
        assignment = assignments.get(feature)
        if assignment is None:
            continue
        category, _source = assignment
        per_category.setdefault(category, []).append(feature)

    unclassified_features = [
        feature for feature in feature_order if feature not in assignments
    ]
    source_counts = {
        "notebook": sum(source == "notebook" for _, source in assignments.values()),
        "dictionary": sum(source == "dictionary" for _, source in assignments.values()),
        "unresolved": len(unclassified_features),
    }
    return FeatureCategoryResolution(
        per_category=per_category,
        unclassified_features=unclassified_features,
        conflicts=conflicts,
        source_counts=source_counts,
    )


def _dictionary_categories(
    dictionary: pd.DataFrame | None,
    *,
    unresolved_features: list[str],
    feature_col: str,
    category_col: str,
) -> dict[str, list[str]]:
    if dictionary is None or not unresolved_features:
        return {}
    missing = [
        column for column in (feature_col, category_col) if column not in dictionary.columns
    ]
    if missing:
        raise ValueError(
            "data dictionary missing columns: " + ", ".join(sorted(missing))
        )

    allowed = set(unresolved_features)
    categories: dict[str, list[str]] = {}
    for _, row in dictionary.iterrows():
        feature = _text(row[feature_col])
        category = _text(row[category_col])
        if feature not in allowed or not category:
            continue
        values = categories.setdefault(feature, [])
        if category not in values:
            values.append(category)
    return categories


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()
