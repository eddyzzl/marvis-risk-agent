"""Thin trusted worker entry point; metrics and decisions use existing kernels."""

from __future__ import annotations

import math
from pathlib import Path
from time import perf_counter

import pandas as pd

from marvis.feature.preprocessing import apply_preprocessing_steps
from marvis.packs.modeling.contracts import ModelArtifact
from marvis.packs.modeling.scoring import _ModelArtifactScorer
from marvis.packs.strategy.evaluator import evaluate_strategy_row
from marvis.reference_decision.contracts import DecisionError, validate_features
from marvis.reference_decision.packages import PackageStore, decode_parameters
from marvis.settings import build_settings


def evaluate(manifest, directory, features, *, event_material=None):
    started = perf_counter()
    validate_features(features, manifest)
    needs_events = manifest["configuration"].get("event_binding") is not None
    if needs_events != (event_material is not None):
        raise DecisionError("event_evidence_required" if needs_events else "unexpected_event_evidence")
    row_features = {**features, **(event_material.values if event_material else {})}
    evidence = {"event_evidence": event_material.evidence} if event_material else {}
    if manifest["configuration"].get("package_kind") == "rule_only":
        decision = evaluate_strategy_row(row_features, manifest["strategy_spec"])
        return {
            **decision.to_dict(),
            **evidence,
            "score": None,
            "score_product": None,
            "timing_ms": {
                "features": 0.0,
                "scoring": None,
                "rules": (perf_counter() - started) * 1000,
            },
        }
    artifact = ModelArtifact(**decode_parameters(manifest["model"]))
    frame = pd.DataFrame([row_features])
    frame = apply_preprocessing_steps(
        frame, artifact.params.get("preprocessing_steps") or []
    )
    feature_end = perf_counter()
    # Preprocessing has already run, once, for both model and rule inputs.
    scorer = _ModelArtifactScorer(
        artifact, base_dir=Path(directory), replay_preprocessing=False
    )
    product = manifest["configuration"]["score_product"]
    if product == "scorecard_points":
        scores = scorer.scorecard_points(frame)
    else:
        scores = scorer.score(frame, use_calibration=product == "calibrated_pd")
    if scores is None or len(scores) != 1 or not math.isfinite(scores[0]):
        raise DecisionError("invalid_model_output")
    score = float(scores[0])
    if product != "scorecard_points" and not 0 <= score <= 1:
        raise DecisionError("invalid_model_probability")
    scoring_end = perf_counter()
    row = frame.iloc[0].to_dict()
    row[manifest["configuration"]["score_field"]] = score
    decision = evaluate_strategy_row(row, manifest["strategy_spec"])
    return {
        **decision.to_dict(),
        **evidence,
        "score": score,
        "score_product": product,
        "timing_ms": {
            "features": (feature_end - started) * 1000,
            "scoring": (scoring_end - feature_end) * 1000,
            "rules": (perf_counter() - scoring_end) * 1000,
        },
    }


def worker_evaluate(inputs, ctx):
    from marvis.reference_decision.event_binding import materialize

    settings = build_settings(ctx.workspace)
    # Read the workspace secret locally; it never enters a subprocess argument,
    # invocation log, client payload, or decision record.
    secret = settings.plugin_admin_token_path.read_text().strip().encode()
    store = PackageStore(settings, secret)
    try:
        with store.snapshot(inputs["package_hash"]) as (manifest, directory):
            events = materialize(settings, manifest, inputs.get("event_evidence"), inputs.get("actor_id"))
            return evaluate(manifest, directory, inputs["features"], event_material=events)
    except DecisionError as exc:
        # Safe machine codes only; no worker exception or source data is echoed.
        return {"decision_error": exc.code, "status": exc.status}
