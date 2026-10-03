"""Authenticate supervised fit members needed by selected model inputs."""

from __future__ import annotations

import base64
from dataclasses import dataclass

import numpy as np
import pandas as pd

from marvis.data.preprocessing_evidence import load_preprocessing_state
from marvis.feature.errors import FeatureError
from marvis.feature.preprocessing import apply_preprocessing_steps
from marvis.repositories.task_artifacts import TaskArtifactRepository


_SUPERVISED = frozenset({"woe", "categorical_woe"})
_GUIDANCE = (
    "请改选原始字段并使用在每折内拟合的 recipe（如 scorecard），"
    "或用可认证的原始数据及完整配方重新构建每折预处理；当前冻结映射不能作为独立验证。"
)


@dataclass(frozen=True)
class SupervisedFitMembers:
    masks: tuple[np.ndarray, ...]

    def require_known_heldout(self, *, context):
        if self.masks:
            raise FeatureError(
                f"supervised preprocessing {context} membership is unknown; " + _GUIDANCE
            )

    def check(self, positions, *, context):
        positions = np.asarray(positions)
        if not self.masks:
            return
        if positions.ndim != 1 or not np.issubdtype(positions.dtype, np.integer):
            raise FeatureError("supervised preprocessing row identity is not verifiable; " + _GUIDANCE)
        for mask in self.masks:
            if np.any(positions < 0) or np.any(positions >= len(mask)):
                raise FeatureError("supervised preprocessing row identity is not verifiable; " + _GUIDANCE)
            if mask[positions].any():
                raise FeatureError(
                    f"supervised preprocessing includes {context} heldout members; " + _GUIDANCE
                )


def selected_supervised_fit_members(registry, dataset_id, features):
    state = load_preprocessing_state(registry, dataset_id)
    required = _required_supervised_steps(state.steps, features)
    if not required:
        return SupervisedFitMembers(())
    if not state.artifact_id:
        raise FeatureError("supervised preprocessing fitting membership is unknown; " + _GUIDANCE)
    dataset = registry.get(dataset_id)
    records = {row["id"]: row for row in TaskArtifactRepository(registry._repo.db_path).list_for_task(dataset.task_id)}
    masks = []
    identity = state.artifact_id
    current_step_count = len(state.steps)
    covered = set()
    while identity:
        record = records[identity]
        proof = record["provenance"]
        parent = load_preprocessing_state(registry, proof["source_dataset_id"])
        relevant = required.intersection(range(len(parent.steps), current_step_count))
        if relevant:
            fit = proof["fit"]
            components = fit if isinstance(fit, list) else [fit] if fit else []
            if not components:
                raise FeatureError("supervised preprocessing fitting membership is unknown; " + _GUIDANCE)
            for component in components:
                if component["row_count"] != dataset.row_count:
                    raise FeatureError("supervised preprocessing row identity changed; " + _GUIDANCE)
                mask = np.unpackbits(
                    np.frombuffer(base64.b64decode(component["membership"]), dtype=np.uint8),
                    bitorder="little", count=component["row_count"],
                ).astype(bool)
                mask.flags.writeable = False
                masks.append(mask)
            covered.update(relevant)
        identity = proof["parent_artifact_id"]
        current_step_count = len(parent.steps)
    if covered != required:
        raise FeatureError("supervised preprocessing fitting membership is unknown; " + _GUIDANCE)
    return SupervisedFitMembers(tuple(masks))


def _required_supervised_steps(steps, features):
    needed, supervised = set(features), set()
    for index in range(len(steps) - 1, -1, -1):
        step = steps[index]
        dependencies = _step_dependencies(step)
        used = needed.intersection(dependencies)
        if not used:
            continue
        if step["kind"] in _SUPERVISED:
            supervised.add(index)
        needed.difference_update(used)
        needed.update(name for output in used for name in dependencies[output])
    return supervised


def _step_dependencies(step):
    kind, columns, params = step["kind"], step["columns"], step["params"]
    if kind in {"sentinel", "impute", "cap", "normalize"}:
        return {name: {name} for name in columns}
    if kind in _SUPERVISED:
        return {f"{name}_woe": {name} for name in columns}
    if kind == "missing_indicator":
        return {str(params[name]): {name} for name in columns if params.get(name)}
    if kind == "onehot":
        return {f"{name}_{value}": {name} for name in columns for value in params.get(name, [])}
    if kind == "fitted_rank":
        return {f"{params['column']}__rank": {params["column"]}}
    if kind == "group_aggregate":
        return {f"{params['value']}_by_{params['group']}_{agg}": {params["value"], params["group"]} for agg in params["aggs"]}
    if kind == "derive":
        # Existing row-local replay owns its naming; empty input never refits.
        output = apply_preprocessing_steps(pd.DataFrame(columns=columns), [step])
        return {name: set(columns) for name in set(output.columns) - set(columns)}
    raise FeatureError("unsupported preprocessing dependency kind")
