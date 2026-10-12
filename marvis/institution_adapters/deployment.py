"""Bind explicit remote installation to the existing approved native strategy."""
from dataclasses import asdict
import json

from marvis.db_schema import connect
from marvis.packs.strategy.dsl import canonical_strategy_json, parse_strategy_spec
from marvis.packs.strategy.evaluator import _expression_fields
from marvis.production_governance.evidence import ActivationVerificationContext
from marvis.reference_decision.contracts import digest, validate_features

from .contracts import AdapterError


def approved_promotion(settings, config, promotion_id):
    # Authentication is provided by the native read chain and caller role, not
    # by arbitrary caller-provided receipt bytes or approval reasons.
    with connect(settings.db_path) as conn:
        row = conn.execute("SELECT * FROM production_promotion_requests WHERE id=?", (promotion_id,)).fetchone()
        if row is None or row["environment"] != config.environment or row["status"] not in {"approved", "promoted"}:
            raise AdapterError("approved_native_promotion_required", 422)
        from datetime import datetime, UTC
        if row["status"] == "approved" and datetime.fromisoformat(row["expires_at"]) <= datetime.now(UTC):
            raise AdapterError("approved_native_promotion_expired")
        roles = conn.execute("SELECT * FROM production_promotion_approvals WHERE request_id=? ORDER BY stage", (promotion_id,)).fetchall()
        if {r["stage"] for r in roles} != {"checker", "admin"} or len({row["maker_principal_id"], *(r["principal_id"] for r in roles)}) != 3:
            raise AdapterError("independent_native_reviews_required")
        strategy = conn.execute("SELECT * FROM strategies WHERE id=?", (row["strategy_id"],)).fetchone()
        if strategy is None or strategy["task_id"] is None or strategy["version"] != row["strategy_version"]:
            raise AdapterError("approved_strategy_binding_changed")
        spec = parse_strategy_spec(json.loads(strategy["dsl_json"]))
        if digest(json.loads(canonical_strategy_json(spec))) != row["strategy_content_hash"] or strategy["dsl_content_hash"] != row["strategy_content_hash"]:
            raise AdapterError("approved_strategy_content_changed")
        context = ActivationVerificationContext(**{key: row[key] for key in ActivationVerificationContext.__dataclass_fields__ if key != "promotion_request_id"}, promotion_request_id=promotion_id)
        return {"task_id": strategy["task_id"], "context": asdict(context), "strategy_spec": spec.to_dict(), "expected_current_deployment_id": row["expected_current_deployment_id"], "status": row["status"], "promoted_deployment_id": row["promoted_deployment_id"]}


def install_payload(settings, config, task_id, request):
    promotion = approved_promotion(settings, config, request.promotion_id)
    if promotion["task_id"] != task_id:
        raise AdapterError("promotion_task_binding_mismatch", 403)
    schema = [f.model_dump() for f in request.raw_schema]
    declared = promotion["strategy_spec"]["metadata"].get("institution_deployment", {})
    if declared != {"package_kind": "rule_only", "schema_hash": digest(schema)}:
        raise AdapterError("approved_deployment_schema_binding_required", 422)
    if any(f["name"].startswith("__marvis_") for f in schema) or len({f["name"] for f in schema}) != len(schema):
        raise AdapterError("deployment_raw_schema_invalid", 422)
    fields = {field for rule in promotion["strategy_spec"]["rules"] for field in _expression_fields(rule["condition"])}
    if not fields <= {f["name"] for f in schema}:
        raise AdapterError("deployment_all_rule_fields_must_be_declared", 422)
    validate_features(request.probe_features, {"configuration": {"raw_schema": schema}})
    payload = {"context": promotion["context"], "strategy_spec": promotion["strategy_spec"], "raw_schema": schema, "schema_hash": digest(schema), "package_kind": "rule_only"}
    return {"manifest": payload, "bundle_hash": digest(payload), "probe_features": request.probe_features}
