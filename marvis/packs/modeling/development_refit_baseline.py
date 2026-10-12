"""Baseline from the final child's already-fitted development rows only."""
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from marvis.packs.modeling.errors import ModelingError
from marvis.validation.binning import bin_distribution, equal_frequency_bin_edges


def development_baseline(frame, scores, features, *, source, membership):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) != len(frame) or not len(frame) or not np.isfinite(scores).all():
        raise ModelingError("development_refit_baseline_prediction_invalid")
    edges = equal_frequency_bin_edges(scores, 10)
    distributions = {}
    for feature in features:
        values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if not len(finite):
            continue
        quantiles = np.unique(np.quantile(finite, np.linspace(0, 1, 11)))
        if len(quantiles) < 2:
            quantiles = np.array([-np.inf, np.inf])
        distributions[feature] = {
            "sample_count": len(finite), "missing_rate": float(1 - len(finite) / len(values)),
            "quantile_edges": quantiles.tolist(),
            "bin_proportions": bin_distribution(finite, quantiles).tolist(),
        }
    return {
        "schema_version": 1,
        "baseline_scope": "development_fit",
        "source_dataset_id": source.id, "source_dataset_content_hash": source.content_hash,
        "fit_membership_sha256": membership["sha256"],
        "created_at": datetime.now(UTC).isoformat(), "bin_count": 10,
        "score_direction": "higher_is_riskier", "score_edges": edges.tolist(),
        # The historical monitor contract calls its reference bucket 'train'.
        # Its explicit scope is the whole development fit, never held-out test.
        "score_distribution": {"train": {
            "sample_count": len(frame), "population_scope": "development_fit",
            "bin_proportions": bin_distribution(scores, edges).tolist(),
        }},
        "feature_distributions": distributions,
        "independent_test_evaluation": "unavailable_original_test_consumed_in_fit",
        "oot_evaluation": "not_performed",
        "label_maturity_assurance": "unknown",
    }
