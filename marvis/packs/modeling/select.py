from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from marvis.data.labels import require_labels_confirmed
from marvis.feature.correlation import correlation_matrix, find_collinear_pairs, vif
from marvis.feature.metrics import feature_metrics


@dataclass(frozen=True)
class SelectionResult:
    selected: tuple[str, ...]
    dropped: tuple[tuple[str, str], ...]
    scores: dict[str, dict[str, float]]
    nan_labels_dropped: int = 0


def select_features(
    backend,
    dataset_path: Path,
    *,
    features: list[str],
    target_col: str,
    iv_min: float = 0.02,
    corr_max: float = 0.8,
    vif_max: float = 10.0,
    top_k: int | None = None,
    seed: int = 0,
    drop_nan_labels: bool = False,
) -> SelectionResult:
    del seed  # Selection is deterministic; seed is reserved for API symmetry.
    columns = _unique([*features, target_col])
    frame = backend.read_frame(dataset_path, columns=columns)
    nan_labels_dropped = require_labels_confirmed(
        frame, target_col, drop_nan_labels=drop_nan_labels,
    )
    # Pass float so feature_metrics' isfinite guard drops NaN targets (never coerced to 0).
    target = frame[target_col].to_numpy(dtype=float)

    kept: list[str] = []
    dropped: list[tuple[str, str]] = []
    scores: dict[str, dict[str, float]] = {}
    for feature in features:
        metrics = feature_metrics(
            frame[feature].to_numpy(dtype=float),
            target,
            feature=feature,
        )
        scores[feature] = {"iv": float(metrics.iv), "ks": float(metrics.ks)}
        if metrics.iv < iv_min:
            dropped.append((feature, f"low IV {metrics.iv:.3f}"))
        else:
            kept.append(feature)

    for left, right, corr in find_collinear_pairs(
        correlation_matrix(frame, kept),
        kept,
        threshold=corr_max,
    ):
        if left not in kept or right not in kept:
            continue
        loser, winner = (left, right) if scores[left]["iv"] < scores[right]["iv"] else (right, left)
        kept.remove(loser)
        dropped.append((loser, f"collinear with {winner} ({corr:.2f})"))

    vifs = vif(frame, kept)
    for feature in features:
        if feature in vifs:
            scores.setdefault(feature, {})["vif"] = float(vifs[feature])
    for feature, value in vifs.items():
        if value > vif_max and feature in kept:
            kept.remove(feature)
            dropped.append((feature, f"high VIF {value:.1f}"))

    if top_k is not None and top_k > 0 and len(kept) > top_k:
        ranked = sorted(kept, key=lambda feature: (-scores[feature]["iv"], features.index(feature)))
        selected = ranked[:top_k]
        for feature in ranked[top_k:]:
            dropped.append((feature, f"outside top_k {top_k}"))
        kept = selected

    return SelectionResult(tuple(kept), tuple(dropped), scores, nan_labels_dropped)


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


__all__ = ["SelectionResult", "select_features"]
