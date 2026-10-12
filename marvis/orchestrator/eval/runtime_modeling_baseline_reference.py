"""Independent feature and score baselines from authenticated fit/source rows."""
from datetime import datetime
import math

import pandas as pd

from .runtime_archive_domain import numbers
from .runtime_modeling_reference import distribution_reference, edges_reference, quantile_reference


def _finite(values):
    result = []
    for value in values:
        try:
            number = float(value)
        except (ValueError, TypeError):
            continue
        if math.isfinite(number):
            result.append(number)
    return result


def baseline_reference(frame, predict, config, *, development_mask=None, source=None, membership_sha256=None):
    features = config["features"]
    if development_mask is None:
        train = frame.loc[frame[config["split_col"]] == config["split_values"]["train"]]
    else:
        train = frame.iloc[[i for i, selected in enumerate(development_mask) if selected]]
    scores = _finite(predict(train))
    if len(scores) != len(train) or not scores:
        raise ValueError("independent baseline score population invalid")
    edges = edges_reference(scores)
    distribution = {"train": {"sample_count": len(scores), "bin_proportions": distribution_reference(scores, edges)}}
    if development_mask is None:
        for role in ("test", "oot"):
            if config["split_values"].get(role) is not None:
                part = frame.loc[frame[config["split_col"]] == config["split_values"][role]]
                values = _finite(predict(part)) if len(part) else []
                if values:
                    distribution[role] = {"sample_count": len(values), "bin_proportions": distribution_reference(values, edges)}
    feature_baselines = {}
    for feature in features:
        values = _finite(pd.to_numeric(train[feature], errors="coerce").tolist())
        if not values:
            continue
        quantiles = [quantile_reference(values, index / 10) for index in range(11)]
        if development_mask is not None:
            quantiles = sorted(set(quantiles))
            if len(quantiles) < 2:
                quantiles = [-math.inf, math.inf]
        feature_baselines[feature] = {"sample_count": len(values), "missing_rate": 1 - len(values) / len(train),
            "quantile_edges": quantiles, "bin_proportions": distribution_reference(values, quantiles)}
    result = {"schema_version": 1, "bin_count": 10, "score_edges": edges,
        "score_direction": "higher_is_riskier", "score_distribution": distribution,
        "feature_distributions": feature_baselines}
    if development_mask is not None:
        result.update(baseline_scope="development_fit", source_dataset_id=source["id"],
            source_dataset_content_hash=source["content_hash"], fit_membership_sha256=membership_sha256,
            independent_test_evaluation="unavailable_original_test_consumed_in_fit",
            oot_evaluation="not_performed", label_maturity_assurance="unknown")
        result["score_distribution"]["train"]["population_scope"] = "development_fit"
    return result


def validate_baseline(actual, expected, *, tolerance):
    if not isinstance(actual, dict) or set(actual) != set(expected) | {"created_at"}:
        raise ValueError("independent baseline field contract differs")
    stamp = datetime.fromisoformat(actual["created_at"].replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("baseline creation time must be declared with timezone")
    # Infinite endpoints are exact bucket bounds, never valid measured metrics.
    def bounds(value):
        if isinstance(value, float) and not math.isfinite(value):
            if math.isnan(value):
                raise ValueError("baseline NaN cannot establish a distribution")
            return {"fixed_bucket_endpoint": "+inf" if value > 0 else "-inf"}
        if isinstance(value, list):
            return [bounds(item) for item in value]
        if isinstance(value, dict):
            return {key: bounds(item) for key, item in value.items()}
        return value
    numbers(bounds({k: v for k, v in actual.items() if k != "created_at"}), bounds(expected), **tolerance)
