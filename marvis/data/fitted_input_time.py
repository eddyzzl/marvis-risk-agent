"""Recorded fit-input visibility; never historical parameter existence proof."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib

import numpy as np

from marvis.data.asof_join import AsOfJoinEngine
from marvis.data.feature_time import _verify_dataset_unchanged, feature_time_evidence
from marvis.data.preprocessing_evidence import load_preprocessing_state
from marvis.data.preprocessing_validation import _step_dependencies
from marvis.feature.errors import FeatureError
from marvis.repositories.task_artifacts import TaskArtifactRepository


@dataclass(frozen=True)
class _Bound:
    times: tuple
    complete: bool
    known: bool = False


def validate_fitted_input_time(
    registry,
    dataset_id,
    features,
    *,
    fit_mask,
    evaluation_mask,
    requires_labels=False,
    pending_steps=(),
    pending_fit=(),
    evaluation_roles=("non_fit_rows",),
    excluded_evaluation_roles=(),
):
    """Reject any known future input even when other input times are unknown.

    Masks are native dataset positions. A model/WOE label without a recorded
    available_at keeps total assurance unknown, independently of feature times.
    """
    dataset = registry.get(dataset_id)
    fit = _mask(fit_mask, dataset.row_count)
    evaluation = _mask(evaluation_mask, dataset.row_count)
    if np.any(fit & evaluation):
        raise FeatureError("fitting and evaluation time membership overlap")
    fields, decisions, artifacts = _row_times(registry, dataset_id, ())
    fields = _project_steps(fields, pending_steps, pending_fit, dataset.row_count)
    if not features or not set(features) <= fields.keys():
        raise FeatureError("fitted input time selected fields are missing")
    anchors = [
        value
        for value, included in zip(decisions, evaluation, strict=True)
        if included and value is not None
    ]
    earliest = min(anchors, default=None)
    known = []
    complete = bool(
        fit.any() and evaluation.any() and len(anchors) == int(evaluation.sum())
    )
    for feature in features:
        bound = fields[feature]
        if not bound.known:
            complete = False
            continue
        times = [
            value for value, included in zip(bound.times, fit, strict=True) if included
        ]
        present = [value for value in times if value is not None]
        latest = max(present, default=None)
        if latest is not None and earliest is not None and latest > earliest:
            raise FeatureError(
                "fitted input is available after evaluation decision; "
                "拟合输入晚于评估决策时点。请使用当时已可得的数据重建拟合/训练分区，"
                "或改用更晚且符合实际业务的评估窗口；不能据此声称无时点穿越。"
            )
        if latest is not None:
            known.append(latest)
        complete = complete and bound.complete and len(present) == len(times)
    reasons = [] if complete else ["selected_fit_input_or_evaluation_time_unknown"]
    if requires_labels:
        reasons.append("label_available_at_not_recorded")
    _verify_dataset_unchanged(registry, dataset)
    return {
        "schema_version": "fitted-input-time-evidence.v1",
        "dataset_id": dataset.id,
        "dataset_content_hash": dataset.content_hash,
        "scope": "recorded_fit_inputs_visible_before_evaluation_decisions",
        "assurance": "verified" if complete and not requires_labels else "unknown",
        "feature_input_assurance": "verified" if complete else "unknown",
        "label_input_assurance": "unknown" if requires_labels else "not_required",
        "features": list(features),
        "fit_rows": int(fit.sum()),
        "evaluation_rows": int(evaluation.sum()),
        "evaluation_roles": list(evaluation_roles),
        "excluded_evaluation_roles": list(excluded_evaluation_roles),
        "row_count": dataset.row_count,
        "fit_membership": base64.b64encode(
            np.packbits(fit, bitorder="little").tobytes()
        ).decode("ascii"),
        "evaluation_membership": base64.b64encode(
            np.packbits(evaluation, bitorder="little").tobytes()
        ).decode("ascii"),
        "fit_membership_sha256": _mask_hash(fit),
        "evaluation_membership_sha256": _mask_hash(evaluation),
        "latest_known_fit_input_at": max(known).isoformat() if known else None,
        "earliest_known_evaluation_decision_at": earliest.isoformat()
        if earliest is not None
        else None,
        "artifact_ids": sorted(set(artifacts)),
        "reasons": reasons,
        "excluded_assurances": [
            "historical_parameter_existence",
            "historical_model_existence",
            "source_authenticity",
            "label_maturity",
            "model_evaluation",
        ],
    }


def _mask(value, count):
    mask = np.asarray(value)
    if mask.dtype != np.bool_ or mask.shape != (count,):
        raise FeatureError("fitted input time requires full-dataset boolean membership")
    return mask


def _mask_hash(mask):
    return hashlib.sha256(np.packbits(mask, bitorder="little").tobytes()).hexdigest()


def _combine(bounds):
    bounds = list(bounds)
    if not any(bound.known for bound in bounds):
        # Untimed wide tables must not allocate another row-sized vector for
        # every feature at every preprocessing step.
        return bounds[0]
    if len(bounds) == 1:
        return bounds[0]
    times = tuple(
        max((value for value in row if value is not None), default=None)
        for row in zip(*(bound.times for bound in bounds), strict=True)
    )
    complete = all(
        bound.complete and all(value is not None for value in bound.times)
        for bound in bounds
    )
    return _Bound(times, complete, True)


def _row_times(registry, dataset_id, seen, cache=None):
    if dataset_id in seen or len(seen) >= 64:
        raise FeatureError(
            "fitted input time lineage is cyclic or exceeds 64 transforms"
        )
    dataset = registry.get(dataset_id)
    cache = {} if cache is None else cache
    if dataset_id in cache:
        value, frozen = cache[dataset_id]
        _verify_dataset_unchanged(registry, frozen)
        return value

    def checked(fields, decisions, artifacts):
        _verify_dataset_unchanged(registry, dataset)
        value = fields, decisions, sorted(set(artifacts))
        cache[dataset_id] = value, dataset
        return value

    names = registry.authenticated_parquet_column_names(dataset_id)
    unknown = _Bound((None,) * dataset.row_count, False)
    fields = dict.fromkeys(names, unknown)
    repo = TaskArtifactRepository(registry._repo.db_path)
    native = AsOfJoinEngine(
        registry, repo, workspace_root=registry.datasets_root.parent
    ).row_input_time_evidence(dataset_id)
    if native:
        parents = {}
        artifacts = [native["artifact_id"]]
        for role, source in native["parents"].items():
            binding = registry.authenticate_dataset_binding(
                source["dataset_id"], expected_task_id=dataset.task_id,
                expected_content_hash=source["content_hash"],
            )
            parent, _, parent_artifacts = _row_times(
                registry, source["dataset_id"], (*seen, dataset_id), cache
            )
            registry.verify_dataset_binding(binding)
            parents[role] = parent
            artifacts.extend(parent_artifacts)
        decision_positions, feature_positions = zip(*native["memberships"], strict=True)
        mapped = {}
        for name, bound in parents["decision"].items():
            if name in fields:
                if id(bound) not in mapped:
                    mapped[id(bound)] = _map_bound(bound, decision_positions) if bound.known else unknown
                fields[name] = mapped[id(bound)]
        available = native["available_at"]
        declared = _Bound(
            available, all(time is not None for time in available),
            any(time is not None for time in available),
        )
        projected = {}
        for name, source_name in native["feature_inputs"].items():
            parent = parents["feature"][source_name]
            if id(parent) not in projected:
                inherited = _map_bound(parent, feature_positions) if parent.known else unknown
                # A recorded feature snapshot can time a previously untimed raw
                # field. It cannot erase a known later native dependency.
                projected[id(parent)] = _combine((declared, inherited)) if inherited.known else declared
            fields[name] = projected[id(parent)]
        return checked(fields, native["decision_at"], artifacts)
    state = load_preprocessing_state(registry, dataset_id)
    if not state.artifact_id:
        # Existing readers still authenticate complex JOIN/cleaning lineage and
        # reject corruption. They do not provide a native row-position mapping
        # for this narrower cross-row proof, so no timestamps are inferred.
        feature_time_evidence(registry, dataset_id, names)
        return checked(fields, unknown.times, [])
    proof = repo.get_for_task(dataset.task_id, state.artifact_id)["provenance"]
    source_id = proof["source_dataset_id"]
    parent_state = load_preprocessing_state(registry, source_id)
    parent, decisions, artifacts = _row_times(registry, source_id, (*seen, dataset_id), cache)
    local = state.steps[len(parent_state.steps) :]
    components = proof["fit"]
    components = (
        components
        if isinstance(components, list)
        else [components]
        if components
        else []
    )
    projected = _project_steps(parent, local, components, dataset.row_count)
    if not local:
        retained = [name for name in names if name in parent and parent[name].known]
        if retained:
            before = registry.read_authenticated_parquet_snapshot(
                source_id, columns=retained
            )
            after = registry.read_authenticated_parquet_snapshot(
                dataset_id, columns=retained
            )
            if not before.reset_index(drop=True).equals(after.reset_index(drop=True)):
                return checked(fields, unknown.times, [*artifacts, state.artifact_id])
    fields.update({name: projected[name] for name in names if name in projected})
    return checked(fields, decisions, [*artifacts, state.artifact_id])


def _map_bound(bound, positions):
    times = tuple(bound.times[i] if i is not None else None for i in positions)
    return _Bound(times, bound.complete and all(time is not None for time in times),
                  any(time is not None for time in times))


def _project_steps(parent, local, components, row_count):
    projected = dict(parent)
    fitted = [
        step
        for step in local
        if step["kind"] not in {"derive", "sentinel", "missing_indicator"}
    ]
    fit_index = 0
    for step in local:
        fitted_step = step in fitted
        # Constant imputation keeps its existing authenticated membership slot
        # for compatibility, but its fixed value is not learned across rows.
        # Older receipts without an explicit strategy remain conservative.
        component = (
            components[fit_index]
            if fitted_step and len(components) == len(fitted)
            else None
        )
        for output, inputs in _step_dependencies(step).items():
            if not inputs <= projected.keys():
                raise FeatureError("fitted input time dependency is missing")
            bound = _combine(projected[name] for name in inputs)
            if fitted_step and not (
                step["kind"] == "impute" and step.get("strategy") == "constant"
            ):
                if not bound.known:
                    projected[output] = bound
                    continue
                if component is None:
                    bound = _Bound(bound.times, False, bound.known)
                else:
                    mask = np.unpackbits(
                        np.frombuffer(
                            base64.b64decode(component["membership"]), dtype=np.uint8
                        ),
                        bitorder="little",
                        count=component["row_count"],
                    ).astype(bool)
                    if len(mask) != row_count:
                        raise FeatureError("fitted input time row membership changed")
                    latest = max(
                        (
                            time
                            for time, include in zip(bound.times, mask, strict=True)
                            if include and time is not None
                        ),
                        default=None,
                    )
                    bound = _Bound(
                        tuple(
                            max(
                                (v for v in (time, latest) if v is not None),
                                default=None,
                            )
                            for time in bound.times
                        ),
                        bound.complete
                        and step["kind"] not in {"woe", "categorical_woe"},
                        bound.known,
                    )
            projected[output] = bound
        fit_index += int(fitted_step)
    return projected
