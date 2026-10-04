"""Native receipts binding a completed fold search to its final training inputs."""

import hashlib
import json

from filelock import FileLock

from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.preprocessing_evidence import load_preprocessing_state, verify_source_on_connection
from marvis.packs.modeling.errors import ModelingError
from marvis.packs.modeling.fold_policy import plan_digest
from marvis.packs.modeling.selection_evidence import _selection_source_mapping, selection_evidence_for_training
from marvis.repositories.task_artifacts import TaskArtifactRepository


KIND = "modeling_fold_tuning"
VERSION = "marvis.fold_tuning_receipt.v2"
ORIGIN = "modeling.tune_hyperparameters"
MAX_BYTES = 32 * 1024**2


def _verify_plan(registry, task_id, plan):
    body = {key: value for key, value in plan.items() if key != "sha256"}
    if plan_digest(body) != plan.get("sha256"):
        raise ModelingError("fold selection plan digest differs")
    datasets = []
    for prefix in ("source", "training"):
        dataset = registry.get(plan[f"{prefix}_dataset_id"])
        if (dataset.task_id != task_id or dataset.content_hash != plan[f"{prefix}_content_hash"]
                or dataset.row_count != plan["row_count"]):
            raise ModelingError("fold tuning source identity changed")
        registry.resolve_verified_path(dataset.id)
        load_preprocessing_state(registry, dataset.id)
        datasets.append(dataset)
    mapped = _selection_source_mapping(registry, datasets[1].id, datasets[0].id, ())
    if mapped is None or mapped[1] != plan["ancestry_artifacts"]:
        raise ModelingError("fold tuning ancestry changed")
    selection_evidence_for_training(registry, task_id, datasets[1].id, plan["candidates"],
                                    plan["target_col"], plan["references"])
    return datasets


def publish_fold_tuning(runtime, ctx, *, plan, recipe, result, train_seed, search_identity, reusable_result):
    evidence = result.get("fold_selection_evidence") or {}
    features = result.get("selected_features") or []
    if (evidence.get("plan") != plan or evidence.get("recipe") != recipe
            or not features or features != evidence.get("final_selection", {}).get("selected")):
        raise ModelingError("fold tuning worker result does not match its frozen plan")
    datasets = _verify_plan(runtime.registry, ctx.task_id, plan)
    payload = {"schema_version": VERSION, "recipe": recipe, "train_seed": int(train_seed),
        "features": features, "best_params": result["best_params"], "evidence": evidence,
        "search_identity": search_identity, "reusable_result_sha256": plan_digest(reusable_result)}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    if len(raw) > MAX_BYTES:
        raise ModelingError("fold tuning evidence exceeds its size limit")
    digest = hashlib.sha256(raw).hexdigest()
    root = runtime.registry.datasets_root
    task_dir = root / ctx.task_id
    if task_dir.resolve(strict=True) != task_dir:
        raise ModelingError("fold tuning evidence task path is invalid")
    lock_path = task_dir / ".fold-tuning-evidence.lock"
    if lock_path.is_symlink():
        raise ModelingError("fold tuning evidence lock path is invalid")
    destination = task_dir / "fold_tuning" / f"{digest}.json"
    relative = destination.relative_to(root.parent).as_posix()
    repo = TaskArtifactRepository(runtime.registry._repo.db_path)
    with FileLock(str(lock_path), timeout=30, mode=0o600):
        existing = repo.get_for_task_kind_path(ctx.task_id, KIND, relative)
        if existing is not None:
            ref = {"artifact_id": existing["id"], "content_hash": digest}
            load_fold_tuning(runtime.registry, ctx.task_id, ref)
            return ref
        uow = ArtifactUnitOfWork()
        artifact = uow.stage_file(destination.parent, destination.name)
        artifact.path.write_bytes(raw)
        artifact.path.chmod(0o600)

        def commit(conn):
            conn.execute("BEGIN IMMEDIATE")
            for dataset in datasets:
                verify_source_on_connection(runtime.registry, conn, dataset)
            if artifact.final_path.read_bytes() != raw:
                raise ModelingError("fold tuning evidence changed before registration")
            return repo.register_on_connection(conn, task_id=ctx.task_id, kind=KIND,
                path=relative, content_hash=digest, origin_tool=ORIGIN,
                provenance={"schema_version": VERSION, "plan_sha256": plan["sha256"], "recipe": recipe})

        try:
            record = uow.finalize_with_connection(repo.transaction, commit)
        except Exception:
            uow.rollback()
            raise
    return {"artifact_id": record["id"], "content_hash": digest}


def load_fold_tuning(registry, task_id, reference):
    if not isinstance(reference, dict) or set(reference) != {"artifact_id", "content_hash"}:
        raise ModelingError("fold tuning evidence reference is invalid")
    record = TaskArtifactRepository(registry._repo.db_path).get_for_task(task_id, reference["artifact_id"])
    if (record is None or record["kind"] != KIND or record["origin_tool"] != ORIGIN
            or record["content_hash"] != reference["content_hash"]
            or record["provenance"].get("schema_version") != VERSION):
        raise ModelingError("fold tuning evidence identity mismatch")
    path = registry.datasets_root.parent / record["path"]
    if (path.resolve(strict=True) != path or not path.is_relative_to(registry.datasets_root)
            or path.stat().st_size > MAX_BYTES):
        raise ModelingError("fold tuning evidence path is invalid")
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES or hashlib.sha256(raw).hexdigest() != reference["content_hash"]:
        raise ModelingError("fold tuning evidence content changed")
    payload = json.loads(raw)
    plan = payload["evidence"]["plan"]
    if (payload.get("schema_version") != VERSION
            or payload["recipe"] != record["provenance"].get("recipe")
            or plan["sha256"] != record["provenance"].get("plan_sha256")):
        raise ModelingError("fold tuning evidence producer binding changed")
    _verify_plan(registry, task_id, plan)
    return payload


def validate_fold_training(registry, task_id, inputs, recipe, features, params, reference):
    payload = load_fold_tuning(registry, task_id, reference)
    plan = payload["evidence"]["plan"]
    if (payload["recipe"] != recipe or payload["features"] != list(features)
            or payload["best_params"] != params or payload["train_seed"] != int(inputs["seed"])
            or plan["training_dataset_id"] != inputs["dataset_id"]
            or plan["target_col"] != inputs["target_col"] or plan["split_col"] != inputs["split_col"]
            or plan["split_values"] != inputs["split_values"]
            or plan["drop_nan_labels"] != bool(inputs.get("drop_nan_labels"))):
        raise ModelingError("final training inputs differ from their native fold tuning result")
    return payload["evidence"]


def validate_fold_cache(registry, task_id, cached, *, search_identity, plan, recipe):
    payload = load_fold_tuning(registry, task_id, cached["fold_tuning_evidence_ref"])
    reusable = {key: value for key, value in cached.items() if key != "fold_tuning_evidence_ref"}
    if (payload.get("search_identity") != search_identity
            or payload.get("reusable_result_sha256") != plan_digest(reusable)
            or payload["evidence"]["plan"] != plan or payload["recipe"] != recipe):
        raise ModelingError("cached tuning result or search identity differs from its native fold receipt")
    return payload


def validate_fold_experiment(registry, task_id, experiment, artifact):
    """Reauthenticate persisted search evidence before delivery or later refitting."""
    reference = (artifact.params or {}).get("fold_tuning_evidence_ref")
    config = experiment.config
    config_reference = config.params.get("fold_tuning_evidence_ref")
    if reference is None and config_reference is None:
        return None
    if reference != config_reference:
        raise ModelingError("model and experiment fold references differ")
    payload = load_fold_tuning(registry, task_id, reference)
    # Platform evidence is refreshed at final training. All learner parameters
    # and row/label controls must still match the authenticated frozen search.
    from marvis.packs.modeling.recipes.common import model_params, sample_weight_col
    if (model_params(config.params) != model_params(payload["best_params"])
            or config.params.get("valid_group_cols") != payload["best_params"].get("valid_group_cols")
            or sample_weight_col(config) != str(payload["evidence"]["plan"]["sample_weight_col"])
            or list(artifact.feature_list) != payload["features"]
            or artifact.params.get("fold_tuning_evidence") != payload["evidence"]):
        raise ModelingError("model metadata differs from its native fold result")
    validate_fold_training(registry, task_id, {
        "dataset_id": config.dataset_id, "target_col": config.target_col,
        "split_col": config.split_col, "split_values": config.split_values,
        "seed": config.seed, "drop_nan_labels": config.drop_nan_labels,
    }, experiment.recipe_id, config.features, payload["best_params"], reference)
    return payload["evidence"]
