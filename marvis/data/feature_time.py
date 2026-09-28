"""Read existing producer evidence without promoting unknown feature timing.

The scope is field availability at recorded decisions. Source authenticity,
label maturity and independent model evaluation remain separate contracts.
"""

from __future__ import annotations

from copy import deepcopy

import pandas as pd

from marvis.data.asof_join import AsOfJoinEngine
from marvis.data.preprocessing_evidence import load_preprocessing_state
from marvis.data.predicate_ast import canonicalize_expression
from marvis.data.transform_time import transform_time_parent
from marvis.feature.errors import FeatureError
from marvis.feature.preprocessing import apply_preprocessing_steps
from marvis.repositories.task_artifacts import TaskArtifactRepository


def _unknown(reason):
    return {"assurance": "unknown", "reasons": [reason]}


def _combined(values):
    values = list(values)
    assurances = {item["assurance"] for item in values}
    assurance = (
        "unknown"
        if not values or "unknown" in assurances
        else "inferred"
        if "inferred" in assurances
        else "verified"
    )
    return {
        "assurance": assurance,
        "reasons": sorted({reason for value in values for reason in value["reasons"]}),
    }


def feature_time_evidence(registry, dataset_id, features):
    """Authenticate selected field timing through native row-local transforms.

    An ordinary JOIN or old sidecar cannot acquire a temporal claim. Fitted
    transforms have no recorded fit cutoff yet, so their output timing is
    unknown even when training membership itself is correctly certified.
    """
    features = list(features)
    if (
        not features
        or any(not isinstance(f, str) or not f for f in features)
        or len(features) != len(set(features))
    ):
        raise FeatureError("temporal evidence requires distinct selected fields")
    fields, artifacts, dataset = _fields(registry, dataset_id, ())
    if not set(features) <= fields.keys():
        raise FeatureError("temporal evidence selected fields are missing")
    selected = {name: fields[name] for name in features}
    combined = _combined(selected.values())
    return {
        "schema_version": "feature-time-evidence.v1",
        "dataset_id": dataset.id,
        "dataset_content_hash": dataset.content_hash,
        "scope": "selected_field_availability_at_recorded_decisions",
        **combined,
        "fields": selected,
        "artifact_ids": sorted(set(artifacts)),
        "excluded_assurances": [
            "source_authenticity",
            "label_maturity",
            "model_evaluation",
        ],
    }


def _fields(registry, dataset_id, seen):
    if dataset_id in seen or len(seen) >= 64:
        raise FeatureError("temporal lineage is cyclic or exceeds 64 transforms")
    dataset = registry.get(dataset_id)
    # Reading evidence must not repin the registry's source_path: sample-design
    # artifacts may already freeze that exact path. The retained-descriptor
    # snapshot authenticates bytes without changing an existing producer binding.
    names = registry.authenticated_parquet_column_names(dataset_id)
    fields = {name: _unknown("no_field_availability_evidence") for name in names}
    repo = TaskArtifactRepository(registry._repo.db_path)
    engine = AsOfJoinEngine(
        registry, repo, workspace_root=registry.datasets_root.parent
    )
    native = engine.feature_time_status(dataset_id)
    if native["artifact_id"]:
        if not native["fields"].keys() <= fields.keys():
            raise FeatureError("temporal evidence fields differ from dataset")
        fields.update(native["fields"])
        _verify_dataset_unchanged(registry, dataset)
        return fields, [native["artifact_id"]], dataset
    state = load_preprocessing_state(registry, dataset_id)
    if not state.artifact_id:
        transformed = transform_time_parent(registry, repo, dataset)
        if transformed:
            parent, artifacts, _ = _fields(
                registry, transformed.source_dataset_id, (*seen, dataset_id)
            )
            projected = _project_operations(parent, transformed.operations)
            if set(projected) != set(names):
                raise FeatureError("temporal transform output schema changed")
            _verify_dataset_unchanged(registry, dataset)
            return projected, [*artifacts, transformed.result_artifact_id], dataset
        _verify_dataset_unchanged(registry, dataset)
        return fields, [], dataset
    record = repo.get_for_task(dataset.task_id, state.artifact_id)
    proof = record["provenance"]
    source_id = proof["source_dataset_id"]
    parent_state = load_preprocessing_state(registry, source_id)
    parent, artifacts, _ = _fields(registry, source_id, (*seen, dataset_id))
    projected = deepcopy(parent)
    local_steps = state.steps[len(parent_state.steps) :]
    for step in local_steps:
        _project_step(projected, step)
    if not local_steps:
        retained = [
            name
            for name in names
            if name in parent and parent[name]["assurance"] != "unknown"
        ]
        if retained:
            before = registry.read_authenticated_parquet_snapshot(
                source_id, columns=retained
            )
            after = registry.read_authenticated_parquet_snapshot(dataset_id, columns=retained)
            for name in retained:
                if (
                    not before[name]
                    .reset_index(drop=True)
                    .equals(after[name].reset_index(drop=True))
                ):
                    projected[name] = _unknown(
                        "projection_changed_feature_values_or_membership"
                    )
    # Preparation may project fields and add a split. A new field stays unknown;
    # Unchanged values and order above bind each inherited timing claim.
    fields.update({name: projected[name] for name in names if name in projected})
    _verify_dataset_unchanged(registry, dataset)
    return fields, [*artifacts, state.artifact_id], dataset


def _verify_dataset_unchanged(registry, dataset):
    if registry.get(dataset.id) != dataset:
        raise FeatureError("temporal dataset binding changed during verification")
    registry.resolve_verified_path(dataset.id)


def _project_step(fields, step):
    columns = step["columns"]
    if not columns or not set(columns) <= fields.keys():
        raise FeatureError("temporal transform inputs are missing")
    # Use the existing replay grammar for names, without reading sample values
    # or reimplementing recipe naming. Empty input cannot fit new parameters.
    after = apply_preprocessing_steps(pd.DataFrame(columns=list(fields)), [step])
    if not after.columns.is_unique:
        raise FeatureError("temporal transform produced ambiguous field names")
    produced = set(after.columns) - set(fields)
    kind = step["kind"]
    # Encoders/indicators may overwrite an existing output name. Their write
    # set includes those columns even though the schema did not grow.
    if kind in {"woe", "categorical_woe"}:
        produced |= {f"{name}_woe" for name in columns if step["params"].get(name)}
    elif kind == "missing_indicator":
        produced |= {
            str(step["params"][name]) for name in columns if step["params"].get(name)
        }
    if kind in {"sentinel", "missing_indicator", "derive"}:
        value = _combined(fields[name] for name in columns)
    else:
        value = _unknown("fitted_parameter_availability_not_recorded")
    affected = produced | (
        set(columns) & set(after.columns)
        if kind in {"sentinel", "impute", "cap", "normalize"}
        else set()
    )
    for name in set(fields) - set(after.columns):
        del fields[name]
    for name in affected:
        fields[name] = deepcopy(value)


def _project_operations(parent, operations):
    fields = deepcopy(parent)
    for operation in operations:
        kind = operation["op"]
        if kind == "rename_columns":
            mapping = operation["mapping"]
            fields = {mapping.get(name, name): value for name, value in fields.items()}
        elif kind == "drop_columns":
            for name in operation["columns"]:
                del fields[name]
        elif kind == "fill_missing":
            for item in operation["fills"]:
                fields[item["column"]] = _unknown(
                    "fill_parameter_availability_not_recorded"
                )
        elif kind == "derive_columns":
            added = {}
            for item in operation["derivations"]:
                expression = canonicalize_expression(
                    item["expression"], fields, predicate=False
                )
                added[item["name"]] = (
                    _combined(fields[name] for name in expression.required_columns)
                    if expression.required_columns
                    else _unknown("derived_without_temporal_inputs")
                )
            fields.update(added)
        elif kind not in {"cast_columns", "filter_rows", "deduplicate"}:
            raise FeatureError("unsupported temporal cleaning operation")
    return fields
