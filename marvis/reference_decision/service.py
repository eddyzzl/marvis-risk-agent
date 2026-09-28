from __future__ import annotations

from datetime import UTC, datetime
import json
import subprocess
import sys
import time

from marvis.db_schema import connect
from marvis.plugins.contracts import PROTOCOL_VERSION
from marvis.plugins.runner import _parse_worker_result, _run_worker
from marvis.reference_decision.contracts import DecisionError, digest, validate_features
from marvis.reference_decision.ledger import DecisionLedger
from marvis.reference_decision.packages import PackageStore
from marvis.reference_decision.schema import initialize


ENVIRONMENT = "local-reference"


class ReferenceDecisionService:
    def __init__(self, settings, secret):
        initialize(settings.db_path)
        self.settings = settings
        self.packages = PackageStore(settings, secret)
        self.ledger = DecisionLedger(settings.db_path)

    def head(self, slot="production"):
        column = (
            "active_deployment_id" if slot == "production" else "shadow_deployment_id"
        )
        with connect(self.settings.db_path) as conn:
            row = conn.execute(
                f"""SELECT d.*, h.revision, m.canonical_json
                FROM production_environment_heads h
                JOIN production_deployments d ON d.id=h.{column}
                JOIN production_deployment_manifests m ON m.id=d.manifest_hash
                WHERE h.environment=?""",
                (ENVIRONMENT,),
            ).fetchone()
            if row is None:
                raise DecisionError("no_active_reference_deployment", 409)
            manifest = json.loads(row["canonical_json"])
            package_hash = manifest.get("decision_package_hash")
            installation = conn.execute(
                "SELECT package_hash FROM reference_installations WHERE promotion_id=?",
                (row["promotion_request_id"],),
            ).fetchone()
        if (
            not package_hash
            or not installation
            or installation["package_hash"] != package_hash
        ):
            raise DecisionError("reference_installation_unverified", 409)
        self.packages.get(package_hash)
        return {
            "environment": ENVIRONMENT,
            "slot": slot,
            "deployment_id": row["id"],
            "package_hash": package_hash,
            "revision": row["revision"],
            "manifest_hash": row["manifest_hash"],
            "state": "serving",
            "assurance": "local_reference_only",
        }

    def evaluate(self, package_hash, features, timeout):
        job = {
            "protocol_version": PROTOCOL_VERSION,
            "builtin": True,
            "module": "marvis.reference_decision.evaluation",
            "entrypoint": "worker_evaluate",
            "workspace": str(self.settings.workspace),
            "datasets_root": str(self.settings.datasets_dir),
            "task_id": "reference-decision",
            "side_effects": [],
            "inputs": {"package_hash": package_hash, "features": features},
            "memory_limit_mb": 1024,
            "file_size_limit_mb": 1,
        }
        try:
            completed = _run_worker(
                sys.executable, job, timeout=timeout, rss_limit_mb=1024
            )
        except subprocess.TimeoutExpired as exc:
            raise DecisionError("scoring_timeout", 503) from exc
        result = _parse_worker_result(completed.stdout)
        if completed.returncode or not result or not result.get("ok"):
            # Do not echo raw feature values or worker exception text into logs/API.
            raise DecisionError("scoring_failed", 503)
        return result["output"]

    def decide(self, request, *, slot="production"):
        started = time.perf_counter()
        ledger_scope = f"{ENVIRONMENT}:{slot}"
        input_hash = digest(request.model_dump())
        existing = self.ledger.existing(ledger_scope, request.request_id, input_hash)
        if existing is not None:
            return existing
        pinned = self.ledger.pinned_package(
            ledger_scope, request.request_id, input_hash
        )
        head = self.head(slot) if pinned is None else None
        package_hash = pinned or head["package_hash"]
        if request.expected_package_hash != package_hash:
            raise DecisionError("expected_package_mismatch", 409)
        manifest = self.packages.get(package_hash)
        config = manifest["configuration"]
        if request.decision_node != config["decision_node"]:
            raise DecisionError("decision_node_mismatch")
        validate_features(request.features, manifest)
        owner, existing = self.ledger.claim(
            ledger_scope,
            request.request_id,
            input_hash,
            package_hash,
            config["timeout_seconds"],
        )
        if existing is not None:
            return existing
        worker_started = time.perf_counter()
        try:
            result = self.evaluate(
                package_hash, request.features, config["timeout_seconds"]
            )
            status, error_code = "decided", None
        except DecisionError as exc:
            result = {
                "action": {
                    "type": config["failure_action"],
                    "value": config["failure_action"],
                    "reason_code": "REFERENCE_EXECUTION_UNAVAILABLE",
                    "stop": True,
                },
                "matched_rule_id": None,
                "score": None,
                "score_product": config["score_product"],
                "timing_ms": {"features": None, "scoring": None, "rules": None},
            }
            status, error_code = "fallback", exc.code
        result["timing_ms"].update(
            worker_wall=(time.perf_counter() - worker_started) * 1000,
            total_before_persistence=(time.perf_counter() - started) * 1000,
        )
        response = {
            **result,
            "decision_id": digest(
                {"scope": ledger_scope, "request_id": request.request_id}
            ),
            "request_id": request.request_id,
            "input_hash": input_hash,
            "package_hash": package_hash,
            "decision_node": request.decision_node,
            "environment": ENVIRONMENT,
            "slot": slot,
            "status": status,
            "error_code": error_code,
            "next_action": "人工复核或修复执行环境后发起新申请请求"
            if error_code
            else None,
            "created_at": datetime.now(UTC).isoformat(),
            "assurance": "local_reference_only",
            "versions": {
                "strategy": manifest["strategy"],
                "model_artifact_id": manifest["model"]["id"],
                "feature_schema_hash": digest(config["raw_schema"]),
                "preprocessing_hash": digest(
                    manifest["model"]["params"].get("preprocessing_steps", [])
                ),
                "platform": manifest["platform_version"],
                "files": manifest["files"],
            },
        }
        return self.ledger.finish(ledger_scope, request.request_id, owner, response)
