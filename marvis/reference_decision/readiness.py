"""Read-only package preparation from authenticated producer/transform evidence."""

import json

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db_schema import connect
from marvis.feature.errors import FeatureError
from marvis.packs.modeling._runtime import _artifact_base_dir
from marvis.packs.modeling.errors import ModelingError
from marvis.packs.modeling.producer_receipts import KIND, load_receipt
from marvis.packs.strategy.dsl import parse_strategy_spec
from marvis.packs.strategy.evaluator import _expression_fields
from marvis.production_governance.errors import GovernanceConflict, GovernanceNotFound
from marvis.production_governance.repository import _strategy_binding_tx
from marvis.reference_decision.contracts import DecisionError, RulePackageRequest
from marvis.reference_decision.features import feature_outputs
from marvis.repositories.datasets import DatasetRepository
from marvis.repositories.modeling import ModelingRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository


def required_raw_fields(model_fields, steps, source_names):
    """Follow actual replay dependencies, including transforms unused by the model."""
    source_names = set(source_names)
    required, generated = set(), set()
    outputs = source_names
    for index, step in enumerate(steps):
        required.update(set(step.get("columns") or []) - generated)
        outputs, generated = feature_outputs(source_names, steps[: index + 1])
    required.update(set(model_fields) - generated)
    if not required <= source_names or not set(model_fields) <= outputs:
        raise DecisionError("model_or_preprocessing_inputs_unbound")
    outputs, generated = feature_outputs(required, steps)
    return sorted(required), sorted(outputs), sorted(generated)


def _readiness_result():
    return {
        "schema_version": "reference-decision-readiness.v1",
        "state": "blocked",
        "build_ready": False,
        "reason_codes": [],
        "model": None,
        "producer": {"state": "unknown", "artifact_id": None},
        "preprocessing": {
            "state": "unknown",
            "receipt_ids": [],
            "source_binding": None,
        },
        "raw_requirements": [],
        "score_products": [],
        "strategy": None,
        "strategy_input_fields": [],
        "unbound_strategy_fields": [],
        "score_field_candidates": [],
        "build_requires_explicit_declaration": [
            "raw_schema.types",
            "raw_schema.nullable",
            "score_field",
            "decision_node",
        ],
        "declaration_authority": "package_request_requires_user_confirmation",
    }


def package_readiness(
    store, model_artifact_id, *, strategy_id=None, strategy_version=None
):
    result = _readiness_result()
    try:
        repo = ModelingRepository(store.settings.db_path)
        artifact = repo.get_model_artifact(model_artifact_id)
        experiment = repo.get_experiment(artifact.experiment_id) if artifact else None
        if (
            not experiment
            or experiment.artifact_id != artifact.id
            or experiment.status not in {"trained", "selected"}
        ):
            raise DecisionError("completed_registered_model_required")
        result["model"] = {
            "id": artifact.id,
            "experiment_id": experiment.id,
            "task_id": experiment.task_id,
            "status": experiment.status,
            "algorithm": artifact.algorithm,
        }
        if artifact.algorithm not in {
            "lr",
            "lgb",
            "xgb",
            "catboost",
            "mlp",
            "scorecard",
            "ensemble",
        }:
            raise DecisionError("unsupported_score_algorithm")
        if artifact.score_direction != "higher_is_riskier":
            raise DecisionError("score_direction_unknown")
        # Bound the total bytes before the common verifier materializes the closure.
        records = TaskArtifactRepository(store.settings.db_path).list_for_task(
            experiment.task_id
        )
        closures = [
            r
            for r in records
            if r["kind"] == KIND and r["provenance"].get("artifact_id") == artifact.id
        ]
        for record in closures:
            sizes = [f.get("size") for f in record["provenance"].get("files", [])]
            if (
                any(type(s) is not int or s < 0 for s in sizes)
                or sum(sizes) > 512_000_000
            ):
                raise DecisionError("package_size_limit")
        producer, snapshots = load_receipt(
            store.settings.db_path,
            experiment,
            artifact,
            _artifact_base_dir(store.settings, experiment.task_id),
        )
        del (
            snapshots
        )  # Authentication only; never deserialize a model in this endpoint.
        result["producer"] = {"state": "authenticated", "artifact_id": producer["id"]}
        steps = artifact.params.get("preprocessing_steps") or []
        if steps:
            proof = store._preprocessing_receipt(experiment, artifact)
            root = proof["lineage"][-1]["provenance"]
            registry = DatasetRegistry(
                DatasetRepository(store.settings.db_path),
                DataBackend(store.settings.datasets_dir),
                store.settings.datasets_dir,
            )
            binding = registry.authenticate_dataset_binding(
                root["source_dataset_id"],
                expected_task_id=experiment.task_id,
                expected_content_hash=root["source_content_hash"],
            )
            source_names = registry.authenticated_binding_column_names(binding)
            result["preprocessing"] = {
                "state": proof["assurance"],
                "receipt_ids": [r["id"] for r in proof["lineage"]],
                "source_binding": {
                    "dataset_id": binding.dataset_id,
                    "task_id": binding.task_id,
                    "content_hash": binding.content_hash,
                },
            }
        else:
            source_names = list(artifact.feature_list)
            result["preprocessing"] = {
                "state": "not_required",
                "receipt_ids": [],
                "source_binding": None,
            }
        names, outputs, _ = required_raw_fields(
            artifact.feature_list, steps, source_names
        )
        result["raw_requirements"] = [
            {"name": name, "type": None, "nullable": None, "declaration_required": True}
            for name in names
        ]
        calibration = bool((artifact.params.get("calibration") or {}).get("path"))
        result["score_products"] = [
            {"value": "raw_pd", "available": True, "reason_code": None},
            {
                "value": "calibrated_pd",
                "available": calibration,
                "reason_code": None if calibration else "calibration_required",
            },
            {
                "value": "scorecard_points",
                "available": artifact.algorithm == "scorecard",
                "reason_code": None
                if artifact.algorithm == "scorecard"
                else "scorecard_required",
            },
        ]
        if (strategy_id is None) != (strategy_version is None):
            raise DecisionError("strategy_id_and_version_required_together")
        if strategy_id is not None:
            with connect(store.settings.db_path) as conn:
                strategy = _strategy_binding_tx(
                    conn, strategy_id=strategy_id, strategy_version=strategy_version
                )
                row = conn.execute(
                    "SELECT dsl_json FROM strategies WHERE id=?", (strategy_id,)
                ).fetchone()
            spec = parse_strategy_spec(json.loads(row["dsl_json"]))
            if any(
                not a.reason_code
                for a in [spec.default_action, *(r.action for r in spec.rules)]
            ):
                raise DecisionError("strategy_reason_codes_required")
            fields = sorted(
                {f for rule in spec.rules for f in _expression_fields(rule.condition)}
            )
            unbound = sorted(set(fields) - set(outputs))
            result.update(
                strategy={
                    "id": strategy_id,
                    "version": strategy_version,
                    "content_hash": strategy["strategy_content_hash"],
                },
                strategy_input_fields=fields,
                unbound_strategy_fields=unbound,
                score_field_candidates=unbound,
            )
        result["state"] = "authenticated"
        # Sample values/dtypes cannot prove the serving schema's type/null policy.
        result["reason_codes"] = ["serving_input_contract_requires_declaration"]
    except DecisionError as exc:
        result["reason_codes"] = [exc.code]
    except ModelingError as exc:
        code = str(exc)
        result["reason_codes"] = [
            code
            if code.startswith("native_model_")
            else "model_producer_evidence_invalid"
        ]
    except (GovernanceConflict, GovernanceNotFound):
        result["reason_codes"] = ["strategy_not_ready"]
    except (FeatureError, ValueError, RuntimeError, KeyError, OSError, TypeError):
        result["reason_codes"] = ["package_source_evidence_invalid"]
    return result


def rule_package_readiness(store, *, strategy_id, strategy_version):
    result = _readiness_result()
    result.update(
        package_kind="rule_only", producer={"state": "not_required", "artifact_id": None},
        preprocessing={"state": "not_required", "receipt_ids": [], "source_binding": None},
        build_requires_explicit_declaration=["raw_schema.types", "raw_schema.nullable", "decision_node"],
    )
    try:
        request = RulePackageRequest(package_kind="rule_only", strategy_id=strategy_id,
            strategy_version=strategy_version, decision_node="readiness", raw_schema=[])
        strategy, spec = store._strategy(request)
        fields = sorted({f for rule in spec.rules for f in _expression_fields(rule.condition)})
        if any(name.startswith("__marvis_") for name in fields):
            raise DecisionError("rule_only_platform_inputs_require_native_binding")
        result.update(
            state="authenticated", strategy_input_fields=fields,
            raw_requirements=[{"name": name, "type": None, "nullable": None,
                               "declaration_required": True} for name in fields],
            strategy={"id": strategy_id, "version": strategy_version,
                      "content_hash": strategy["strategy_content_hash"]},
            reason_codes=["serving_input_contract_requires_declaration"],
        )
    except DecisionError as exc:
        result["reason_codes"] = [exc.code]
    except (GovernanceConflict, GovernanceNotFound, ValueError):
        result["reason_codes"] = ["strategy_not_ready"]
    return result
