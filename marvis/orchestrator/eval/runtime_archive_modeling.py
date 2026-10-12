"""Independent native LR/XGB/LGB fit closure for binary, regression and multiclass."""
import base64
import hashlib
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .runtime_archive_domain import FrozenDomainBinding, DomainArchive, HASH, ID, equal, numbers, one_step
from .runtime_archive_datasets import _dataset, _source_input
from .runtime_archive_reader import _Unsupported, _file
from .runtime_modeling_native_reference import read_linear_reference, independent_linear_fit
from .runtime_modeling_tree_reference import PROFILES, read_tree_reference, independent_tree_fit
from .runtime_modeling_reference import membership_reference, metrics_reference
from .runtime_modeling_baseline_reference import baseline_reference, validate_baseline


# Independent fixed contracts for the native complete metric record and the
# binary comparison projection. Undefined metrics remain explicit None in the
# complete record; only the comparison contract omits its undefined fields.
_FULL_MODEL_METRICS = (
    "train_ks", "test_ks", "oot_ks", "train_auc", "test_auc", "oot_auc",
    "psi_test_vs_train", "psi_oot_vs_train", "overfit_train_test_gap",
    "overfit_train_oot_gap", "overfit_flag",
    "weighted_train_ks", "weighted_test_ks", "weighted_oot_ks",
    "weighted_train_auc", "weighted_test_auc", "weighted_oot_auc",
    "weighted_psi_test_vs_train", "weighted_psi_oot_vs_train",
    "train_rmse", "test_rmse", "oot_rmse", "train_mae", "test_mae", "oot_mae",
    "train_r2", "test_r2", "oot_r2", "train_macro_auc", "test_macro_auc", "oot_macro_auc",
    "train_logloss", "test_logloss", "oot_logloss", "train_accuracy", "test_accuracy", "oot_accuracy",
    "test_ks_ci_low", "test_ks_ci_high", "test_ks_ci_std",
    "oot_ks_ci_low", "oot_ks_ci_high", "oot_ks_ci_std", "ks_ci_n_boot",
    "test_lift_head_5", "test_lift_tail_5", "test_lift_head_10", "test_lift_tail_10",
    "oot_lift_head_5", "oot_lift_tail_5", "oot_lift_head_10", "oot_lift_tail_10",
)
_BINARY_COMPARISON_METRICS = (
    "train_ks", "test_ks", "oot_ks", "train_auc", "test_auc", "oot_auc",
    "test_ks_ci_low", "test_ks_ci_high", "test_ks_ci_std",
    "oot_ks_ci_low", "oot_ks_ci_high", "oot_ks_ci_std",
    "test_lift_head_5", "test_lift_tail_5", "test_lift_head_10", "test_lift_tail_10",
    "oot_lift_head_5", "oot_lift_tail_5", "oot_lift_head_10", "oot_lift_tail_10",
    "psi_test_vs_train", "psi_oot_vs_train", "overfit_flag",
)


def _selection_metrics(actual, values, *, allow_projection=False, rel_tol=1e-9, abs_tol=1e-11):
    if not isinstance(actual, dict) or not isinstance(values, dict):
        raise ValueError("selection metric object mismatch")
    if set(values) - set(_FULL_MODEL_METRICS):
        raise ValueError("independent selection metric contract mismatch")
    complete = {key: values.get(key) for key in _FULL_MODEL_METRICS}
    projection = {key: values[key] for key in _BINARY_COMPARISON_METRICS
                  if key in values and values[key] is not None}
    if set(actual) == set(complete):
        expected = complete
    elif allow_projection and set(actual) == set(projection):
        expected = projection
    else:
        raise ValueError("selection metric key set mismatch")
    if actual.get("overfit_flag") is not None and type(actual["overfit_flag"]) is not bool:
        raise ValueError("selection overfit flag must be a boolean")
    numbers(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol)


def _heldout_binary_comparison(scope, values):
    return (scope.get("profile") in {"lr", "xgb", "lgb"}
            and scope.get("fit_scope") == "outer_train"
            and scope.get("test_evaluation") == "independent_heldout"
            and scope.get("parent_experiment_id") is None
            and all(type(values.get(key)) in (int, float) and math.isfinite(values[key])
                    for key in ("test_auc", "test_ks")))


def _selection_projection_allowed(select, scope, *, values):
    inputs, output = select["input"], select["output"]
    refit = output.get("refit")
    selected = output.get("selected_experiment_id")
    return (_heldout_binary_comparison(scope, values)
            and isinstance(refit, dict) and refit.get("applied") is False
            and refit.get("requested") is False
            and inputs.get("refit_on_train_plus_test") is False
            and output.get("selection_evidence") is None
            and isinstance(selected, str) and bool(selected)
            and inputs.get("selected_experiment_id") == selected
            and selected in inputs.get("experiment_ids", []))


class FrozenFitRepresentation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    representation: Literal["compact_float32", "source_native"]
    noncompact_features: list[str]


class FrozenModelingBinding(FrozenDomainBinding):
    dataset_id: str = Field(pattern=ID)
    dataset_sha256: str = Field(pattern=HASH)
    selected_experiment_id: str = Field(pattern=ID)
    final_artifact_id: str = Field(pattern=ID)
    fit_representations: dict[str, FrozenFitRepresentation]

    @model_validator(mode="after")
    def modeling_contract(self):
        if self.case.family not in {"modeling", "monitoring"} or self.case.task.task_type != "modeling" or not self.case.materials:
            raise ValueError("declared native modeling case required")
        if self.selected_experiment_id not in self.fit_representations:
            raise ValueError("final fit numerical representation must be independently frozen")
        return self


def native_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=True).encode()).hexdigest()


def native_learner_params(params):
    evidence = {"classes", "per_class", "preprocessing_steps", "preprocessing_chain_traceable",
        "preprocessing_assurance", "preprocessing_evidence", "feature_time_evidence",
        "fitted_input_time_evidence", "selection_evidence", "selection_evidence_refs",
        "fold_tuning_evidence", "fold_tuning_evidence_ref", "fold_selection_plan",
        "special_value_governance", "valid_group_cols", "refit_on_train_plus_test", "best_iteration"}
    return {k: v for k, v in params.items() if k not in evidence}


def fit_representation(session, frame, config, experiment_id, *, development=False):
    import pandas as pd
    profiles = session.binding.fit_representations
    if experiment_id not in profiles:
        raise _Unsupported("model fit numerical representation not independently frozen")
    profile = FrozenFitRepresentation.model_validate(profiles[experiment_id])
    if development and profile.representation != "source_native":
        raise ValueError("development refit requires native source representation")
    if set(profile.noncompact_features) - set(config["features"]):
        raise ValueError("noncompact feature outside frozen native inputs")
    if profile.representation == "source_native" and profile.noncompact_features:
        raise ValueError("source native fit cannot declare a compact exclusion")
    projected = frame.copy()
    if profile.representation == "compact_float32":
        for feature in config["features"]:
            if feature not in profile.noncompact_features and pd.api.types.is_numeric_dtype(projected[feature]) and not pd.api.types.is_bool_dtype(projected[feature]):
                projected[feature] = projected[feature].astype("float32")
    return projected


def source_material_reference(session, source, frame, scratch):
    identity = session.archive / "workspace/datasets" / session.binding.task_id / ".source-identities" / f"{source['id']}.json"
    if identity.exists():
        _source_input(session.archive, session.manifest, session.binding, source, frame, scratch)
        return "authenticated_original_upload"
    split = one_step(session.steps, "modeling.make_split")
    inputs, output = split["input"], split["output"]
    if output["result_dataset_id"] != source["id"] or not inputs.get("split_col") or inputs.get("split_config"):
        raise _Unsupported("generated/transformed modeling source requires its independently declared split reference")
    parent = session.conn.execute("SELECT content_hash FROM datasets WHERE id=?", (inputs["dataset_id"],)).fetchone()
    if parent is None:
        raise ValueError("original modeling source missing")
    registered, original = _dataset(session.conn, session.archive, session.binding.task_id, inputs["dataset_id"], parent["content_hash"])
    _source_input(session.archive, session.manifest, session.binding, registered, original, scratch)
    columns = list(dict.fromkeys([*inputs["feature_cols"], inputs["target_col"], inputs["split_col"], *inputs.get("passthrough_cols", [])]))
    import pandas as pd
    pd.testing.assert_frame_equal(frame, original[columns], check_dtype=False, check_exact=True)
    if output["split_col"] != inputs["split_col"]:
        raise ValueError("approved original split column changed")
    return "authenticated_original_upload_and_exact_existing_split_projection"


def artifact_metadata(row):
    return {"id": row["id"], "algorithm": row["algorithm"], "model_path": row["model_path"],
        "feature_list": json.loads(row["feature_list_json"]), "params": json.loads(row["params_json"]),
        "woe_maps": json.loads(row["woe_maps_json"]) if row["woe_maps_json"] else None,
        "created_at": row["created_at"], "scorecard_table": json.loads(row["scorecard_table_json"]),
        "score_direction": row["score_direction"], "points_direction": row["points_direction"]}


def authenticated_model(session, experiment_id, artifact_id):
    conn, binding = session.conn, session.binding
    experiment = conn.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
    artifact = conn.execute("SELECT * FROM model_artifacts WHERE id=?", (artifact_id,)).fetchone()
    if (experiment is None or artifact is None or experiment["task_id"] != binding.task_id
            or experiment["artifact_id"] != artifact_id or artifact["experiment_id"] != experiment_id):
        raise ValueError("native model ownership closure mismatch")
    metadata = artifact_metadata(artifact)
    records = [dict(r) for r in conn.execute("SELECT * FROM task_artifacts WHERE task_id=? AND kind='modeling_native_model_closure'", (binding.task_id,))
               if json.loads(r["provenance_json"]).get("artifact_id") == artifact_id]
    if len(records) != 1:
        raise ValueError("unique native model receipt required")
    record = records[0]
    proof = json.loads(record["provenance_json"])
    if (record["origin_tool"] != "modeling.native_model_producer.v1" or proof.get("schema_version") != record["origin_tool"]
            or proof["experiment_id"] != experiment_id or proof["scoring_metadata_hash"] != native_hash(metadata)):
        raise ValueError("native model metadata does not match closure")
    if metadata["params"].get("calibration") or proof["members"]:
        raise _Unsupported("calibrated/ensemble native profiles require their independent readers")
    if len(proof["files"]) != 1 or proof["files"][0]["path"] != metadata["model_path"]:
        raise ValueError("native linear dependency closure mismatch")
    member = proof["files"][0]
    if Path(member["path"]).name != member["path"]:
        raise ValueError("native model member path invalid")
    path = _file(session.archive, Path("workspace/tasks") / binding.task_id / "modeling_artifacts" / member["path"])
    if (len(path.read_bytes()) != member["size"] or hashlib.sha256(path.read_bytes()).hexdigest() != member["sha256"]
            or record["content_hash"] != member["sha256"] or record["path"] != f"modeling_artifacts/{metadata['model_path']}"):
        raise ValueError("native model bytes differ from registered receipt")
    config, metrics = json.loads(experiment["config_json"]), json.loads(experiment["metrics_json"])
    if metadata["feature_list"] != config["features"] or metadata["params"].get("preprocessing_steps"):
        raise _Unsupported("external preprocessing requires its independent source transformation reader")
    reader = read_tree_reference if metadata["algorithm"] in PROFILES else read_linear_reference
    model, predict = reader(path, metadata["algorithm"], metadata["feature_list"])
    learner = model.steps[-1][1] if hasattr(model, "steps") else model
    for key, value in config["params"].items():
        if hasattr(learner, "get_params") and key in learner.get_params() and learner.get_params()[key] != value:
            raise ValueError("frozen native learner hyperparameters mismatch")
    if config["target_type"] != "continuous" and hasattr(learner, "random_state") and learner.random_state != config["seed"]:
        raise ValueError("native learner seed mismatch")
    if config["early_stopping_rounds"] is not None and metadata["algorithm"] not in PROFILES:
        raise _Unsupported("native early-stopping fit profile requires independent membership replay")
    return dict(experiment), dict(artifact), metadata, config, metrics, record, model, predict


def membership_bytes(mask):
    raw = bytes(sum(int(v) << j for j, v in enumerate(mask[i:i + 8])) for i in range(0, len(mask), 8))
    return {"rows": sum(mask), "membership": base64.b64encode(raw).decode(), "sha256": hashlib.sha256(raw).hexdigest()}


def _registered_json(session, reference, *, kind, origin):
    record = session.conn.execute("SELECT * FROM task_artifacts WHERE id=? AND task_id=?", (reference["artifact_id"], session.binding.task_id)).fetchone()
    if record is None or record["kind"] != kind or record["origin_tool"] != origin or record["content_hash"] != reference["content_hash"]:
        raise ValueError("native domain receipt reference mismatch")
    path = _file(session.archive, Path("workspace") / record["path"], maximum=32 * 1024**2)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["content_hash"]:
        raise ValueError("native domain receipt bytes changed")
    return dict(record), json.loads(raw)


def fold_reference(session, metadata, config, frame):
    ref = metadata["params"].get("fold_tuning_evidence_ref")
    if ref is None:
        if config["params"].get("fold_tuning_evidence_ref") is not None:
            raise ValueError("native model fold ref missing")
        return None
    equal(ref, config["params"].get("fold_tuning_evidence_ref"))
    _, payload = _registered_json(session, ref, kind="modeling_fold_tuning", origin="modeling.tune_hyperparameters")
    evidence = payload["evidence"]
    equal(metadata["params"].get("fold_tuning_evidence"), evidence)
    if (payload["features"] != config["features"] or payload["train_seed"] != config["seed"]
            or payload["recipe"] != config["recipe_id"] or evidence["plan"]["training_dataset_id"] != config["dataset_id"]):
        raise ValueError("native fold learner/source contract mismatch")
    masks = membership_reference(frame, config)
    outer = masks["train"]
    for fold in evidence["folds"]:
        selection = fold["selection"]
        body = {k: v for k, v in selection.items() if k != "sha256"}
        if hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest() != selection["sha256"]:
            raise ValueError("fold selection content digest mismatch")
        members = selection["memberships"]
        decoded = {}
        for role, member in members.items():
            raw = base64.b64decode(member["membership"], validate=True)
            mask = [bool(raw[i // 8] & (1 << (i % 8))) for i in range(len(frame))]
            equal(member, membership_bytes(mask))
            if any(yes and not train for yes, train in zip(mask, outer, strict=True)):
                raise ValueError("fold reads outside outer train")
            decoded[role] = mask
        # Fit, early-stop and validation roles must be disjoint where declared.
        independent_roles = [r for r in decoded if r in {"fit", "valid", "early_stop", "validation"}]
        for i in range(len(frame)):
            if sum(decoded[r][i] for r in independent_roles) > 1:
                raise ValueError("native fold fitting and validation overlap")
    final = evidence["final_selection"]
    equal(final["memberships"]["fit"], membership_bytes(outer))
    equal(final["selected"], config["features"])
    if evidence.get("oot_evaluation") != "not_used_for_selection_or_tuning":
        raise ValueError("native fold OOT selection exclusion missing")
    return payload


def _model_values(session, experiment_id, artifact_id, frame):
    experiment, artifact, metadata, config, metrics, receipt, model, predict = authenticated_model(session, experiment_id, artifact_id)
    if config["dataset_id"] != session.binding.dataset_id:
        raise ValueError("native model source dataset changed")
    proof_id = (metadata["params"].get("fitted_input_time_evidence") or {}).get("development_refit_receipt_id")
    development = proof_id is not None
    original_frame = frame
    frame = fit_representation(session, frame, config, experiment_id, development=development)
    masks = membership_reference(frame, config, development=development)
    if development:
        row = session.conn.execute("SELECT * FROM task_artifacts WHERE id=? AND task_id=?", (proof_id, session.binding.task_id)).fetchone()
        if row is None:
            raise ValueError("development receipt missing")
        _, proof = _registered_json(session, {"artifact_id": proof_id, "content_hash": row["content_hash"]},
            kind="modeling_development_refit", origin="modeling.native_development_refit.v1")
        if (proof["experiment_id"] != experiment_id or proof["artifact_id"] != artifact_id
                or proof["config_sha256"] != native_hash(config) or proof["metrics_sha256"] != native_hash(metrics)
                or proof["model_metadata_sha256"] != native_hash(metadata)):
            raise ValueError("development child closure mismatch")
        expected_masks = {"original_train": masks["train"], "original_test": masks["test"],
            "development_fit": masks["development"], "excluded": [not v for v in masks["development"]]}
        equal(proof["memberships"], {k: membership_bytes(v) for k, v in expected_masks.items()})
        parent_id, parent_artifact = proof["parent"]["experiment_id"], proof["parent"]["artifact_id"]
        pexp, _, pmeta, pconfig, pmetrics, preceipt, _, _ = authenticated_model(session, parent_id, parent_artifact)
        equal(proof["parent"], {"experiment_id": parent_id, "artifact_id": parent_artifact,
            "config_sha256": native_hash(pconfig), "metrics_sha256": native_hash(pmetrics),
            "model_metadata_sha256": native_hash(pmeta), "native_receipt_id": preceipt["id"], "native_receipt_hash": preceipt["content_hash"]})
        if pexp["status"] != "selected" or proof["parent_fold_ref"] != pconfig["params"].get("fold_tuning_evidence_ref"):
            raise ValueError("development selection parent mismatch")
        if proof["preprocessing"]["steps"] or proof["historical_availability"] != "unknown":
            raise _Unsupported("development transformed/historically attested profile requires independent source reader")
        fold_reference(session, pmeta, pconfig, frame)
    else:
        fold_reference(session, metadata, config, frame)
    fit_mask = masks["development"] if development else masks["train"]
    weight = config["params"].get("sample_weight_col")
    if weight:
        raise _Unsupported("weighted native model requires independent weighted metric profile")
    fit_frame = frame.iloc[[i for i, yes in enumerate(fit_mask) if yes]]
    classes = metadata["params"].get("classes", [])
    if metadata["algorithm"] in PROFILES:
        independent_tree_fit(model, fit_frame, config, classes=classes, frozen_params=native_learner_params(metadata["params"]))
    else:
        independent_linear_fit(model, fit_frame, config["features"], config["target_col"], classes=classes)
    values, _ = metrics_reference(frame, config, predict, development=development, classes=classes)
    compact = session.binding.fit_representations[experiment_id]
    compact = FrozenFitRepresentation.model_validate(compact).representation == "compact_float32"
    tolerance = {"rel_tol": 1e-6, "abs_tol": 1e-7} if compact or metadata["algorithm"].startswith("xgb") else {}
    numbers(metrics, values, **tolerance)
    baseline = json.loads(artifact["baseline_distributions_json"]) if artifact["baseline_distributions_json"] else None
    if development:
        if proof.get("baseline_sha256") != native_hash(baseline) or proof.get("baseline_scope") != "development_fit":
            raise ValueError("development baseline is missing or unbound to its producer proof")
    if baseline is not None:
        source = session.conn.execute("SELECT id, content_hash FROM datasets WHERE id=?", (config["dataset_id"],)).fetchone()
        expected_baseline = baseline_reference(original_frame, predict, config,
            development_mask=masks["development"] if development else None,
            source=source, membership_sha256=proof["memberships"]["development_fit"]["sha256"] if development else None)
        validate_baseline(baseline, expected_baseline, tolerance=tolerance)
    return values, {"profile": metadata["algorithm"], "fit_scope": "full_development" if development else "outer_train",
        "fit_rows": sum(fit_mask), "test_evaluation": "unavailable_consumed_in_fit" if development else "independent_heldout",
        "historical_availability": "unknown", "parent_experiment_id": parent_id if development else None,
        "metric_numeric_tolerance": tolerance or {"rel_tol": 1e-9, "abs_tol": 1e-11},
        "baseline_verification": "independently_recomputed" if baseline is not None else "unavailable"}


def revalidate_modeling_archive(archive: Path, *, expected_manifest_sha256: str, frozen_binding: FrozenModelingBinding):
    if not isinstance(frozen_binding, FrozenModelingBinding):
        raise ValueError("explicit FrozenModelingBinding required")
    binding = FrozenModelingBinding.model_validate(frozen_binding.model_dump())
    with DomainArchive(archive, expected_manifest_sha256, binding, family="modeling",
                       checks=("registered_dataset", "source_material", "model_closure_fit_and_metrics", "selection_evidence")) as session:
        session.open()
        session.current = "registered_dataset"
        source, frame = _dataset(session.conn, session.archive, binding.task_id, binding.dataset_id, binding.dataset_sha256)
        session.mark("registered_dataset", rows=len(frame))
        session.current = "source_material"
        import tempfile
        scratch = Path(session.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        source_scope = source_material_reference(session, source, frame, scratch)
        session.mark("source_material", scope=source_scope)
        session.current = "model_closure_fit_and_metrics"
        values, scope = _model_values(session, binding.selected_experiment_id, binding.final_artifact_id, frame)
        session.mark("model_closure_fit_and_metrics", **scope)
        session.current = "selection_evidence"
        select = one_step(session.steps, "modeling.select_experiment")
        if select["output"]["selected_experiment_id"] != binding.selected_experiment_id or select["output"]["artifact_id"] != binding.final_artifact_id:
            raise ValueError("selected model differs from frozen final artifact")
        parent = select["output"].get("selection_evidence")
        if parent is not None and not isinstance(parent, dict):
            raise ValueError("native selection parent object mismatch")
        candidates = select["input"]["experiment_ids"]
        if select["input"].get("selected_experiment_id"):
            original = parent.get("experiment_id", binding.selected_experiment_id) if parent else binding.selected_experiment_id
            if select["input"]["selected_experiment_id"] != original or original not in candidates:
                raise ValueError("native selected candidate was not explicitly authorized")
        else:
            raise _Unsupported("automatic multi-candidate selection requires frozen independent ranking policy")
        _selection_metrics(select["output"]["metrics"], values,
            allow_projection=_selection_projection_allowed(select, scope, values=values),
            **scope["metric_numeric_tolerance"])
        if scope["parent_experiment_id"]:
            if not parent or parent["experiment_id"] != scope["parent_experiment_id"]:
                raise ValueError("pre-refit held-out selection reference missing")
            row = session.conn.execute("SELECT artifact_id FROM experiments WHERE id=?", (parent["experiment_id"],)).fetchone()
            if row is None or parent.get("artifact_id") != row["artifact_id"]:
                raise ValueError("pre-refit held-out artifact reference mismatch")
            parent_values, parent_scope = _model_values(session, parent["experiment_id"], row["artifact_id"], frame)
            if parent_scope["fit_scope"] != "outer_train" or parent_scope["test_evaluation"] != "independent_heldout":
                raise ValueError("refit parent is not independent held-out candidate")
            _selection_metrics(parent["metrics"], parent_values,
                allow_projection=_heldout_binary_comparison(parent_scope, parent_values),
                **parent_scope["metric_numeric_tolerance"])
        elif parent is not None:
            raise ValueError("unexpected pre-refit held-out selection reference")
        session.mark("selection_evidence", scope="explicit_candidate_and_authenticated_pre_refit_heldout_metrics")
        session.result["values"] = {"metrics": values, "scope": scope,
            "selection_evidence": parent if parent else {"experiment_id": binding.selected_experiment_id, "metrics": values}}
        session.result["unsupported"]["model_profiles"] = "weighted_calibrated_transformed_ensemble_catboost_mlp_scorecard_require_declared_independent_profiles"
    return session.result
