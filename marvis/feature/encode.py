from __future__ import annotations

import numpy as np
import pandas as pd

from marvis.feature.binning import assign_bins
from marvis.feature.contracts import CategoricalWOECategory, CategoricalWOEResult, WOEResult
from marvis.feature.errors import FeatureError
from marvis.feature.iv import _smoothed_woe_iv

_RARE_CATEGORY = "__rare__"


def onehot_encode(
    df: pd.DataFrame,
    columns: list[str],
    *,
    max_categories: int = 50,
    handle_unknown: str = "ignore",
) -> tuple[pd.DataFrame, dict[str, list[object]]]:
    if handle_unknown not in {"ignore", "error"}:
        raise FeatureError("handle_unknown must be 'ignore' or 'error'")
    mapping = {}
    dummy_frames = []
    for column in columns:
        categories = _ordered_categories(df[column])
        if len(categories) > max_categories:
            raise FeatureError(f"{column} has too many categories")
        mapping[column] = categories
        dummy_frames.append(_dummy_frame(df[column], column, categories))
    encoded = df.drop(columns=columns)
    if dummy_frames:
        encoded = pd.concat([encoded, *dummy_frames], axis=1)
    return encoded, mapping


def label_encode(series: pd.Series) -> tuple[pd.Series, dict[object, int]]:
    categories = _ordered_categories(series)
    mapping = {category: index for index, category in enumerate(categories)}
    encoded = series.map(mapping).fillna(-1).astype(int)
    return encoded.rename(series.name), mapping


def woe_encode(df: pd.DataFrame, feature: str, woe: WOEResult) -> pd.Series:
    if feature not in df.columns:
        raise FeatureError(f"{feature} does not exist")
    edges = np.asarray(woe.edges, dtype=float)
    if len(woe.woe_by_bin) != edges.size - 1:
        raise FeatureError("woe_by_bin length must match edges")
    assigned = assign_bins(df[feature].to_numpy(dtype=float), edges)
    out = np.zeros(assigned.shape, dtype=float)
    for index, bin_index in enumerate(assigned):
        if bin_index >= 0:
            out[index] = float(woe.woe_by_bin[int(bin_index)])
        else:
            out[index] = float(woe.na_woe if woe.na_woe is not None else 0.0)
    return pd.Series(out, index=df.index, name=f"{feature}_woe")


def _ordered_categories(series: pd.Series) -> list[object]:
    return [value for value in pd.unique(series.dropna())]


def _dummy_frame(series: pd.Series, column: str, categories: list[object]) -> pd.DataFrame:
    data = {
        f"{column}_{category}": (series == category).astype(int)
        for category in categories
    }
    return pd.DataFrame(data, index=series.index)


def categorical_woe_encode(
    series: pd.Series,
    target: np.ndarray,
    *,
    feature: str,
    min_count: int | None = None,
    smoothing: float = 0.5,
) -> CategoricalWOEResult:
    """Fit a category -> WOE mapping on a (train-only) fit frame (PREP-3/FS-3).

    Mirrors the platform WOE convention in :func:`marvis.feature.iv.compute_woe_iv`
    (Laplace-smoothed ``WOE = ln(good_dist / bad_dist)``) but keyed by raw category
    value instead of a numeric bin. Categories whose fit-frame frequency is below
    ``min_count`` (default ``max(30, 0.5% of fit rows)``) are pooled into a single
    ``__rare__`` bucket *before* WOE is computed for it, so rare categories share one
    smoothed estimate instead of each getting an unstable one. Missing values get
    their own WOE bucket (na_woe). ``default_woe`` is the global-prior WOE (computed
    from the fit frame's overall good/bad split) used at encode time as the fallback
    for any category unseen during fit — the categorical analogue of an out-of-range
    numeric value falling outside the binning edges.
    """
    values = series.to_numpy(dtype=object)
    tgt = np.asarray(target, dtype=float)
    if values.shape != tgt.shape:
        raise FeatureError("values and target must have the same shape")
    valid_target = np.isfinite(tgt)
    target_values = tgt[valid_target]
    if target_values.size == 0 or not np.all(np.isin(target_values, [0, 1])):
        raise FeatureError("target must be binary 0/1")
    if len(set(target_values.astype(int).tolist())) < 2:
        raise FeatureError("target must contain both good and bad classes")

    fit_values = values[valid_target]
    fit_target = target_values.astype(int)
    is_na = pd.isna(fit_values)
    n_fit = int(fit_target.size)
    resolved_min_count = int(min_count) if min_count is not None else max(30, round(0.005 * n_fit))
    resolved_min_count = max(1, resolved_min_count)

    non_na_values = fit_values[~is_na]
    non_na_target = fit_target[~is_na]
    labels = pd.Series(non_na_values, dtype=object).map(str).to_numpy()
    counts = pd.Series(labels).value_counts()
    rare_categories = tuple(sorted(str(cat) for cat, count in counts.items() if count < resolved_min_count))
    rare_set = set(rare_categories)
    grouped = np.where(np.isin(labels, list(rare_set)), _RARE_CATEGORY, labels)

    total_bad = int(np.sum(fit_target == 1))
    total_good = int(np.sum(fit_target == 0))
    if total_bad == 0 or total_good == 0:
        raise FeatureError("target must contain both good and bad classes")

    group_labels = sorted(set(grouped.tolist()))
    n_groups = len(group_labels) + (1 if is_na.any() else 0)
    n_groups = max(n_groups, 1)

    categories: list[CategoricalWOECategory] = []
    total_iv = 0.0
    for label in group_labels:
        mask = grouped == label
        count = int(np.sum(mask))
        bad = int(np.sum(non_na_target[mask] == 1))
        good = count - bad
        woe, iv_contribution = _smoothed_woe_iv(
            bad, good, total_bad, total_good, n_groups, smoothing=smoothing
        )
        total_iv += iv_contribution
        categories.append(
            CategoricalWOECategory(
                category=str(label),
                count=count,
                bad_count=bad,
                good_count=good,
                bad_rate=(bad / count if count else 0.0),
                woe=woe,
                iv_contribution=iv_contribution,
            )
        )

    na_woe = None
    if is_na.any():
        na_target = fit_target[is_na]
        count = int(na_target.size)
        bad = int(np.sum(na_target == 1))
        good = count - bad
        woe, iv_contribution = _smoothed_woe_iv(
            bad, good, total_bad, total_good, n_groups, smoothing=smoothing
        )
        total_iv += iv_contribution
        na_woe = woe
        categories.append(
            CategoricalWOECategory(
                category="__nan__",
                count=count,
                bad_count=bad,
                good_count=good,
                bad_rate=(bad / count if count else 0.0),
                woe=woe,
                iv_contribution=iv_contribution,
            )
        )

    # Global-prior WOE — the fallback for a category never seen during fit. Uses the
    # *fit frame's* overall good/bad split (still Laplace smoothed, single group) so an
    # unseen category never contributes a leak-free-but-arbitrary signal. Same kernel as
    # the per-group WOE above, evaluated on the single "everything" group (n_groups=1,
    # bad=total_bad, good=total_good) so both distributions are 1.0 → default_woe = 0.0.
    default_woe, _ = _smoothed_woe_iv(
        total_bad, total_good, total_bad, total_good, 1, smoothing=smoothing
    )

    return CategoricalWOEResult(
        feature=feature,
        categories=tuple(categories),
        rare_categories=rare_categories,
        min_count=resolved_min_count,
        smoothing=smoothing,
        default_woe=default_woe,
        na_woe=na_woe,
        total_iv=round(float(total_iv), 6),
    )


def apply_categorical_woe(df: pd.DataFrame, feature: str, woe: CategoricalWOEResult) -> pd.Series:
    """Apply a :class:`CategoricalWOEResult` mapping fitted by :func:`categorical_woe_encode`
    to ``df[feature]``.

    Rare-bucket categories (``woe.rare_categories``) and the fit frame's ``__rare__``
    entry both resolve to the same WOE. A NaN value maps to ``woe.na_woe`` when the fit
    saw NaN, else falls back to ``default_woe`` like any other unseen value; a category
    absent from ``woe.categories`` (never seen during fit) falls back to
    ``woe.default_woe`` — the global-prior WOE, never a leak-prone re-fit."""
    if feature not in df.columns:
        raise FeatureError(f"{feature} does not exist")
    by_category = {item.category: item.woe for item in woe.categories}
    rare_woe = by_category.get(_RARE_CATEGORY)
    rare_set = set(woe.rare_categories)
    na_fallback = woe.na_woe if woe.na_woe is not None else woe.default_woe

    def _lookup(value: object) -> float:
        if pd.isna(value):
            return float(na_fallback)
        label = str(value)
        if label in rare_set and rare_woe is not None:
            return float(rare_woe)
        if label in by_category:
            return float(by_category[label])
        return float(woe.default_woe)

    encoded = df[feature].map(_lookup).astype(float)
    return pd.Series(encoded.to_numpy(), index=df.index, name=f"{feature}_woe")


__all__ = [
    "apply_categorical_woe",
    "categorical_woe_encode",
    "label_encode",
    "onehot_encode",
    "woe_encode",
]
