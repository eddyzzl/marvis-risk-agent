from __future__ import annotations

from dataclasses import asdict, replace
from contextlib import contextmanager
from datetime import UTC, datetime
import hashlib
import hmac
import json
import math
from pathlib import Path
import tempfile


from marvis import __version__
from marvis.artifacts.transactional import ArtifactUnitOfWork
from marvis.data.backend import DataBackend
from marvis.data.preprocessing_evidence import training_preprocessing_state
from marvis.data.registry import DatasetRegistry
from marvis.db_schema import connect
from marvis.packs.modeling._runtime import _artifact_base_dir
from marvis.packs.strategy.dsl import canonical_strategy_json, parse_strategy_spec
from marvis.packs.strategy.evaluator import _expression_fields
from marvis.production_governance.repository import _strategy_binding_tx
from marvis.reference_decision.contracts import (
    DecisionError,
    PackageRequest,
    RulePackageRequest,
    canonical,
    digest,
)
from marvis.reference_decision.features import feature_outputs
from marvis.repositories.datasets import DatasetRepository
from marvis.repositories.modeling import ModelingRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository


def encode_parameters(value):
    """Lossless JSON for existing WOE infinity edges (never accepted from HTTP)."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"$marvis_float": repr(value)}
    if isinstance(value, dict):
        if "$marvis_float" in value:
            raise DecisionError("reserved_parameter_key")
        return {str(k): encode_parameters(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode_parameters(v) for v in value]
    return value


def decode_parameters(value):
    if isinstance(value, dict):
        if set(value) == {"$marvis_float"}:
            return float(value["$marvis_float"])
        return {k: decode_parameters(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode_parameters(v) for v in value]
    return value


def safe_file(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise DecisionError("unsafe_artifact_path")
    path = root / relative
    try:
        if path.resolve(strict=True) != path.absolute() or not path.is_file():
            raise DecisionError("unsafe_artifact_path")
        if path.stat().st_size > 256_000_000:
            raise DecisionError("artifact_size_limit")
    except OSError as exc:
        raise DecisionError("artifact_unavailable") from exc
    return path


class PackageStore:
    def __init__(self, settings, secret: bytes):
        self.settings, self.secret = settings, secret
        self.root = settings.workspace / "reference_decision" / "packages"

    def _signature(self, package_hash):
        return hmac.new(
            self.secret, f"reference-decision.v1:{package_hash}".encode(), "sha256"
        ).hexdigest()

    def get(self, package_hash: str, *, verify_files=True):
        with connect(self.settings.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM reference_decision_packages WHERE id = ?",
                (package_hash,),
            ).fetchone()
        if row is None:
            raise DecisionError("package_not_found", 404)
        manifest = json.loads(row["canonical_json"])
        if digest(manifest) != package_hash or not hmac.compare_digest(
            row["signature"], self._signature(package_hash)
        ):
            raise DecisionError("package_authentication_failed", 409)
        if manifest["platform_version"] != __version__:
            raise DecisionError("package_platform_version_mismatch", 409)
        if verify_files:
            for entry in manifest["files"]:
                path = safe_file(self.root / package_hash, entry["path"])
                if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                    raise DecisionError("package_file_integrity_failed", 409)
        return manifest

    def list(self, *, task_id=None, limit=100, offset=0):
        with connect(self.settings.db_path) as conn:
            conn.execute("BEGIN")
            where, args = (" WHERE s.task_id=?", [task_id]) if task_id else ("", [])
            query = " FROM reference_decision_packages p JOIN strategies s ON s.id=json_extract(p.canonical_json, '$.strategy.id')"
            total = conn.execute("SELECT count(*)" + query + where, args).fetchone()[0]
            rows = conn.execute(
                "SELECT p.*, s.task_id"
                + query
                + where
                + " ORDER BY p.created_at DESC, p.id DESC LIMIT ? OFFSET ?",
                [*args, limit, offset],
            ).fetchall()
        packages = []
        for row in rows:
            manifest = self.get(row["id"], verify_files=False)
            packages.append(
                {
                    "package_hash": row["id"],
                    "task_id": row["task_id"],
                    "created_at": row["created_at"],
                    "strategy_id": manifest["strategy"]["id"],
                    "strategy_version": manifest["strategy"]["version"],
                    "model_artifact_id": (manifest["model"] or {}).get("id"),
                    "package_kind": manifest["configuration"].get("package_kind", "model"),
                    "decision_node": manifest["configuration"]["decision_node"],
                    "assurance": manifest["assurance"],
                    "file_integrity": "verify_on_detail_read",
                }
            )
        return {
            "packages": packages,
            "count": total,
            "next_offset": offset + len(packages)
            if offset + len(packages) < total
            else None,
        }

    def build(self, request: PackageRequest | RulePackageRequest, *, actor_id: str):
        if isinstance(request, RulePackageRequest):
            return self._build_rules(request, actor_id=actor_id)
        repo = ModelingRepository(self.settings.db_path)
        artifact = repo.get_model_artifact(request.model_artifact_id)
        experiment = repo.get_experiment(artifact.experiment_id) if artifact else None
        if (
            not experiment
            or experiment.artifact_id != artifact.id
            or experiment.status not in {"trained", "selected"}
        ):
            raise DecisionError("completed_registered_model_required")
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
        if (
            request.score_product == "scorecard_points"
            and artifact.algorithm != "scorecard"
        ):
            raise DecisionError("scorecard_required")
        if request.score_product == "calibrated_pd" and not artifact.params.get(
            "calibration"
        ):
            raise DecisionError("calibration_required")
        strategy, spec = self._strategy(request)
        steps = artifact.params.get("preprocessing_steps") or []
        outputs, derived = feature_outputs([f.name for f in request.raw_schema], steps)
        fields = {f for rule in spec.rules for f in _expression_fields(rule.condition)}
        if not set(artifact.feature_list) <= outputs or not fields <= outputs | {
            request.score_field
        }:
            raise DecisionError("model_or_strategy_inputs_unbound")
        if request.score_field in outputs or any(
            f.startswith("__marvis_model_pd_") and f != request.score_field
            for f in fields
        ):
            raise DecisionError("model_score_binding_ambiguous")
        receipt = self._preprocessing_receipt(experiment, artifact) if steps else None
        source = _artifact_base_dir(self.settings, experiment.task_id)
        artifacts = [artifact]
        from marvis.packs.modeling.producer_receipts import load_receipt
        from marvis.packs.modeling.errors import ModelingError

        try:
            producer_record, snapshots = load_receipt(
                self.settings.db_path, experiment, artifact, source
            )
        except ModelingError as exc:
            raise DecisionError(str(exc), 409) from exc
        for member in producer_record["provenance"]["members"]:
            if member["algorithm"] not in {"lr", "lgb", "xgb", "catboost", "mlp"}:
                raise DecisionError("unsupported_ensemble_member")
            artifacts.append(
                replace(
                    artifact,
                    id=member["artifact_id"],
                    algorithm=member["algorithm"],
                    model_path=member["model_path"],
                    params={},
                    woe_maps=None,
                    pmml_path=None,
                )
            )
        files = [
            {
                "path": name,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
            }
            for name, data in snapshots.items()
        ]
        if sum(f["size"] for f in files) > 512_000_000:
            raise DecisionError("package_size_limit")
        manifest = {
            "schema_version": "reference-decision-package.v1",
            "platform_version": __version__,
            "configuration": request.model_dump(),
            "strategy": strategy,
            "strategy_spec": json.loads(canonical_strategy_json(spec)),
            "model": encode_parameters(asdict(artifact)),
            "members": [encode_parameters(asdict(a)) for a in artifacts[1:]],
            "files": files,
            "preprocessing_receipt": receipt,
            "model_producer_receipt": producer_record,
            "derived_fields": sorted(derived),
            "output_fields": sorted(outputs),
            "assurance": "local_reference_only",
        }
        return self._publish(manifest, snapshots, source, actor_id)

    def _strategy(self, request):
        with connect(self.settings.db_path) as conn:
            conn.execute("BEGIN")
            strategy = _strategy_binding_tx(
                conn,
                strategy_id=request.strategy_id,
                strategy_version=request.strategy_version,
            )
            strategy_row = conn.execute(
                "SELECT dsl_json, task_id FROM strategies WHERE id = ?",
                (request.strategy_id,),
            ).fetchone()
        spec = parse_strategy_spec(json.loads(strategy_row["dsl_json"]))
        # Explicit reason codes are required for every outcome, including the default.
        if any(
            not action.reason_code
            for action in [spec.default_action, *(r.action for r in spec.rules)]
        ):
            raise DecisionError("strategy_reason_codes_required")
        return strategy, spec

    def _build_rules(self, request, *, actor_id):
        strategy, spec = self._strategy(request)
        outputs = {field.name for field in request.raw_schema}
        fields = {f for rule in spec.rules for f in _expression_fields(rule.condition)}
        if not fields <= outputs:
            raise DecisionError("strategy_inputs_unbound")
        manifest = {
            "schema_version": "reference-decision-package.v2",
            "platform_version": __version__,
            "configuration": {**request.model_dump(), "score_product": None},
            "strategy": strategy,
            "strategy_spec": json.loads(canonical_strategy_json(spec)),
            "model": None,
            "members": [],
            "files": [],
            "preprocessing_receipt": None,
            "model_producer_receipt": None,
            "derived_fields": [],
            "output_fields": sorted(outputs),
            "assurance": "local_reference_only",
        }
        return self._publish(manifest, {}, None, actor_id)

    def _publish(self, manifest, snapshots, source, actor_id):
        package_hash = digest(manifest)
        with connect(self.settings.db_path) as conn:
            # Serialize identical concurrent builds before filesystem publication.
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute(
                "SELECT 1 FROM reference_decision_packages WHERE id = ?",
                (package_hash,),
            ).fetchone():
                self.get(package_hash)
                return package_hash, manifest
            unit = ArtifactUnitOfWork()
            try:
                stage = unit.stage_directory(self.root, package_hash)
                stage.path.mkdir(parents=True, exist_ok=True)
                for name, data in snapshots.items():
                    target = stage.path / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                # Refuse a source mutation during package construction.
                if any(
                    safe_file(source, f["path"]).read_bytes() != snapshots[f["path"]]
                    for f in manifest["files"]
                ):
                    raise DecisionError("source_artifact_changed")
                unit.promote_all()
                conn.execute(
                    "INSERT INTO reference_decision_packages VALUES (?,?,?,?,?)",
                    (
                        package_hash,
                        canonical(manifest),
                        self._signature(package_hash),
                        actor_id,
                        datetime.now(UTC).isoformat(),
                    ),
                )
                conn.commit()
                unit.commit()
            finally:
                unit.rollback()
        return package_hash, manifest

    @contextmanager
    def snapshot(self, package_hash):
        """Deserialize only private copies of the exact bytes we authenticated."""
        manifest = self.get(package_hash, verify_files=False)
        with tempfile.TemporaryDirectory(prefix="marvis-decision-") as temp:
            directory = Path(temp)
            for entry in manifest["files"]:
                data = safe_file(self.root / package_hash, entry["path"]).read_bytes()
                if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                    raise DecisionError("package_file_integrity_failed", 409)
                destination = directory / entry["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
            yield manifest, directory

    def _preprocessing_receipt(self, experiment, artifact):
        registry = DatasetRegistry(
            DatasetRepository(self.settings.db_path),
            DataBackend(self.settings.datasets_dir),
            self.settings.datasets_dir,
        )
        config = experiment.config
        state = training_preprocessing_state(
            registry,
            config.dataset_id,
            split_col=config.split_col,
            train_values=config.split_values["train"],
        )
        expected = artifact.params.get("preprocessing_evidence") or {}
        if (
            state.assurance not in {"training_only", "row_local"}
            or not state.artifact_id
            or expected
            != {"artifact_id": state.artifact_id, "content_hash": state.content_hash}
            or encode_parameters(state.steps)
            != encode_parameters(artifact.params["preprocessing_steps"])
        ):
            raise DecisionError("authenticated_training_preprocessing_required")
        records = {
            r["id"]: r
            for r in TaskArtifactRepository(self.settings.db_path).list_for_task(
                experiment.task_id
            )
        }
        lineage = []
        identity = state.artifact_id
        while identity:
            record = records[identity]
            lineage.append(encode_parameters(record))
            identity = record["provenance"]["parent_artifact_id"]
        return {"assurance": state.assurance, "lineage": lineage}
