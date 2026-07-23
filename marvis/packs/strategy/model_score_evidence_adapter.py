"""Pure reconciliation adapter for governed single-model score evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any

from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.model_evidence import (
    COMPARISON_METRIC_KEYS,
    StrategyModelEvidenceError,
    build_artifact_ref,
    build_evidence_source_ref,
    build_model_comparison_evidence,
    build_model_comparison_metric,
    build_model_evidence_ref,
    build_model_selection,
    build_strategy_model_evidence_bundle,
    validate_single_model_evidence,
)
from marvis.packs.strategy.sample_design_v2 import (
    validate_strategy_sample_design_v2_bundle,
)


class ModelScoreEvidenceComparisonError(ValueError):
    """Authenticated model evidence cannot be reconciled for comparison."""


def build_model_score_comparison(
    *,
    sample_design_bundle: Mapping[str, Any],
    model_evidence: Sequence[Mapping[str, Any]],
    population: str,
    partition: str,
    comparison_ref: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive shared present metrics without accepting values or selecting a model."""

    try:
        bundle = validate_strategy_sample_design_v2_bundle(sample_design_bundle)
        if (
            isinstance(model_evidence, (str, bytes, bytearray))
            or not isinstance(model_evidence, Sequence)
            or len(model_evidence) < 2
        ):
            raise ModelScoreEvidenceComparisonError(
                "model score comparison requires at least two model evidence items"
            )
        models = sorted(
            (
                validate_single_model_evidence(
                    item,
                    sample_design_bundle=bundle,
                )
                for item in model_evidence
            ),
            key=lambda item: item["evidence_id"],
        )
        evidence_refs = [
            build_model_evidence_ref(item, sample_design_bundle=bundle)
            for item in models
        ]
        if len({item["evidence_id"] for item in evidence_refs}) != len(evidence_refs):
            raise ModelScoreEvidenceComparisonError(
                "model score comparison evidence items must be distinct"
            )
        model_identities = {
            (
                item["model_ref"]["kind"],
                item["model_ref"]["ref_id"],
                item["model_ref"]["content_hash"],
            )
            for item in models
        }
        if len(model_identities) != len(models):
            raise ModelScoreEvidenceComparisonError(
                "model score comparison requires distinct authenticated models"
            )
        normalized_comparison_ref = (
            _comparison_ref(
                bundle=bundle,
                evidence_refs=evidence_refs,
                population=population,
                partition=partition,
            )
            if comparison_ref is None
            else _artifact_ref(comparison_ref)
        )
        source_ref = build_evidence_source_ref(
            sample_design_bundle=bundle,
            population=population,
            partition=partition,
            kind=normalized_comparison_ref["kind"],
            ref_id=normalized_comparison_ref["ref_id"],
            content_hash=normalized_comparison_ref["content_hash"],
        )
        indexes = [
            _present_comparison_observations(
                item,
                population=population,
                partition=partition,
            )
            for item in models
        ]
        shared = set(indexes[0])
        for index in indexes[1:]:
            shared.intersection_update(index)
        if not shared:
            raise ModelScoreEvidenceComparisonError(
                "compared models have no shared present metrics on this sample"
            )
        metrics = []
        for metric_key, period, unit in sorted(
            shared,
            key=lambda item: (item[0], item[1] or "", item[2]),
        ):
            values = [
                {
                    "model_evidence_ref": evidence_ref,
                    "value": index[(metric_key, period, unit)]["value"],
                }
                for evidence_ref, index in zip(
                    evidence_refs,
                    indexes,
                    strict=True,
                )
            ]
            numbers = [float(item["value"]) for item in values]
            metrics.append(
                build_model_comparison_metric(
                    sample_design_bundle=bundle,
                    population=population,
                    partition=partition,
                    metric_key=metric_key,
                    status="present",
                    unit=unit,
                    source_ref=source_ref,
                    model_values=values,
                    delta=max(numbers) - min(numbers),
                    period=period,
                )
            )
        comparison = build_model_comparison_evidence(
            sample_design_bundle=bundle,
            population=population,
            partition=partition,
            comparison_ref=normalized_comparison_ref,
            model_evidence_refs=evidence_refs,
            metrics=metrics,
            selection=build_model_selection(
                status="no_selection",
                reason="comparison_evidence_does_not_authorize_selection",
            ),
        )
        # Bundle validation independently reconciles every adapter value to one
        # exact present observation on each authenticated model.
        reconciled = build_strategy_model_evidence_bundle(
            sample_design_bundle=bundle,
            model_evidence=models,
            comparison_evidence=[comparison],
        )
        return reconciled["comparison_evidence"][0]
    except ModelScoreEvidenceComparisonError:
        raise
    except (
        StrategyError,
        StrategyModelEvidenceError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise ModelScoreEvidenceComparisonError(str(exc)) from exc


def _present_comparison_observations(
    evidence: Mapping[str, Any],
    *,
    population: str,
    partition: str,
) -> dict[tuple[str, str | None, str], Mapping[str, Any]]:
    result: dict[tuple[str, str | None, str], Mapping[str, Any]] = {}
    for observation in evidence["observations"]:
        if (
            observation["status"] != "present"
            or observation["bin_id"] is not None
            or observation["metric_key"] not in COMPARISON_METRIC_KEYS
            or observation["sample_ref"]["population"] != population
            or observation["sample_ref"]["partition"] != partition
        ):
            continue
        coordinate = (
            observation["metric_key"],
            observation["period"],
            observation["unit"],
        )
        if coordinate in result:
            raise ModelScoreEvidenceComparisonError(
                "model evidence has duplicate present comparison coordinates"
            )
        result[coordinate] = observation
    return result


def _comparison_ref(
    *,
    bundle: Mapping[str, Any],
    evidence_refs: Sequence[Mapping[str, str]],
    population: str,
    partition: str,
) -> dict[str, str]:
    body = {
        "sample_design_id": bundle["sample_design"]["sample_design_id"],
        "sample_design_content_hash": bundle["sample_design"]["content_hash"],
        "population": population,
        "partition": partition,
        "model_evidence_refs": sorted(
            (dict(item) for item in evidence_refs),
            key=lambda item: item["evidence_id"],
        ),
        "selection": "no_selection",
    }
    digest = hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()
    return build_artifact_ref(
        kind="model_score_comparison_adapter",
        ref_id=f"model-score-comparison-{digest[:24]}",
        content_hash=digest,
    )


def _artifact_ref(value: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {
        "kind",
        "ref_id",
        "content_hash",
    }:
        raise ModelScoreEvidenceComparisonError(
            "comparison_ref must be an artifact reference"
        )
    try:
        return build_artifact_ref(
            kind=value["kind"],
            ref_id=value["ref_id"],
            content_hash=value["content_hash"],
        )
    except (KeyError, StrategyError, TypeError, ValueError) as exc:
        raise ModelScoreEvidenceComparisonError("comparison_ref is invalid") from exc


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


__all__ = [
    "ModelScoreEvidenceComparisonError",
    "build_model_score_comparison",
]
