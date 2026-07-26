"""Shared governed artifact binding for deterministic Strategy ImpactCube evidence.

The binding is the persistence boundary shared by reports and downstream
read-only analyses.  It authenticates the TaskArtifact registry row, canonical
artifact bytes, semantic evidence, producer provenance, and the measurement
audit that committed the evidence.  This module deliberately has no report
dependency.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.impact_cube import (
    MAX_IMPACT_CUBE_JSON_BYTES,
    STRATEGY_IMPACT_CUBE_PRODUCER_VERSION,
    canonical_strategy_impact_cube_json,
    validate_strategy_impact_cube,
)
from marvis.packs.strategy.impact_cube_tools import (
    IMPACT_CUBE_ARTIFACT_KIND,
    IMPACT_CUBE_ARTIFACT_SCHEMA_VERSION,
    IMPACT_CUBE_ORIGIN_TOOL,
    IMPACT_CUBE_REQUIREMENTS_ARTIFACT_SCHEMA_VERSION,
    require_impact_cube_measurement_audit_on_connection,
    validate_impact_cube_producer_run,
)
from marvis.packs.strategy.pool_requirement_resolver import (
    validate_pool_requirement_bindings_provenance,
)


_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_TASK_ARTIFACT_RECORD_FIELDS = frozenset(
    {
        "id",
        "task_id",
        "kind",
        "path",
        "content_hash",
        "origin_tool",
        "provenance",
        "created_at",
    }
)
_IMPACT_CUBE_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "task_id",
        "cube_id",
        "cube_content_hash",
        "pool_ref",
        "sample_design_ref",
        "dataset_binding",
        "target_binding",
        "dimension_bindings",
        "current_strategy_ref",
        "economics_inputs",
        "partitions",
        "populations",
        "lifecycle",
        "producer_run",
    }
)
_IMPACT_CUBE_REQUIREMENTS_PROVENANCE_FIELDS = (
    _IMPACT_CUBE_PROVENANCE_FIELDS | {"requirement_bindings"}
)


@dataclass(frozen=True)
class StrategyImpactCubeArtifactBinding:
    """Authenticated immutable ImpactCube source for governed consumers."""

    task_id: str
    artifact_id: str
    artifact_path: Path
    artifact_content_hash: str
    artifact_provenance: dict[str, Any]
    artifact_provenance_json: str
    cube: dict[str, Any]
    tasks_root: Path
    db_path: Path


def validate_strategy_impact_cube_artifact_binding(
    binding: StrategyImpactCubeArtifactBinding,
) -> dict[str, Any]:
    """Revalidate one typed ImpactCube binding without persistence reads."""

    if not isinstance(binding, StrategyImpactCubeArtifactBinding):
        raise StrategyError(
            "ImpactCube must be an authenticated "
            "StrategyImpactCubeArtifactBinding binding"
        )
    cube = validate_strategy_impact_cube(binding.cube)
    if cube != binding.cube or cube["identity"]["task_id"] != binding.task_id:
        raise StrategyError("ImpactCube binding identity changed")
    _require_canonical_artifact_hash(
        binding.artifact_content_hash,
        canonical_strategy_impact_cube_json(cube),
        "ImpactCube",
    )
    _require_impact_cube_provenance(binding, cube)
    return cube


def load_strategy_impact_cube_artifact(
    runtime,
    *,
    task_id: str,
    artifact_id: str,
    expected_artifact_content_hash: str,
    expected_cube_id: str,
    expected_cube_content_hash: str,
) -> StrategyImpactCubeArtifactBinding:
    """Load one exact authenticated ImpactCube from governed task storage."""

    artifact_id = _hash(artifact_id, "impact_cube_ref.artifact_id")
    artifact_hash = _hash(
        expected_artifact_content_hash,
        "impact_cube_ref.expected_artifact_content_hash",
    )
    record = runtime.task_artifacts.get_for_task(task_id, artifact_id)
    if (
        not isinstance(record, Mapping)
        or set(record) != _TASK_ARTIFACT_RECORD_FIELDS
    ):
        raise StrategyError("ImpactCube artifact registry row is invalid")
    if (
        record["id"] != artifact_id
        or record["task_id"] != task_id
        or record["kind"] != IMPACT_CUBE_ARTIFACT_KIND
        or record["origin_tool"] != IMPACT_CUBE_ORIGIN_TOOL
        or not hmac.compare_digest(
            str(record["content_hash"]),
            artifact_hash,
        )
    ):
        raise StrategyError("ImpactCube artifact registry binding changed")

    tasks_root = Path(runtime.settings.tasks_dir).absolute()
    db_path = Path(runtime.settings.db_path).absolute()
    path = Path(str(record["path"]))
    raw = _read_impact_cube_source_file(
        path,
        root=tasks_root,
        expected_hash=artifact_hash,
    )
    try:
        cube = validate_strategy_impact_cube(
            json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_object_without_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        RecursionError,
    ) as exc:
        raise StrategyError("ImpactCube artifact JSON is invalid") from exc
    canonical = canonical_strategy_impact_cube_json(cube).encode("utf-8")
    if raw != canonical:
        raise StrategyError("ImpactCube artifact bytes are not canonical")
    if cube["cube_id"] != expected_cube_id:
        raise StrategyError("ImpactCube cube_id changed")
    if not hmac.compare_digest(
        cube["content_hash"],
        expected_cube_content_hash,
    ):
        raise StrategyError("ImpactCube content_hash changed")

    expected_path = (
        tasks_root
        / task_id
        / "strategy_impact_cubes"
        / f"{cube['cube_id']}.json"
    )
    if path != expected_path:
        raise StrategyError("ImpactCube artifact path is not canonical")
    provenance = _canonical_object(
        record["provenance"],
        "ImpactCube artifact provenance",
    )
    binding = StrategyImpactCubeArtifactBinding(
        task_id=task_id,
        artifact_id=artifact_id,
        artifact_path=path,
        artifact_content_hash=artifact_hash,
        artifact_provenance=provenance,
        artifact_provenance_json=_canonical_json(provenance),
        cube=cube,
        tasks_root=tasks_root,
        db_path=db_path,
    )
    validate_strategy_impact_cube_artifact_binding(binding)
    with runtime.task_artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        require_strategy_impact_cube_artifact_binding_on_connection(
            conn,
            binding,
        )
        conn.commit()
    return binding


def require_strategy_impact_cube_artifact_binding_on_connection(
    conn,
    binding: StrategyImpactCubeArtifactBinding,
) -> None:
    """Recheck a binding, its bytes, and measurement audit in a transaction."""

    if not isinstance(binding, StrategyImpactCubeArtifactBinding):
        raise StrategyError("ImpactCube artifact binding is invalid")
    if not conn.in_transaction:
        raise StrategyError(
            "ImpactCube binding requires a caller-owned transaction"
        )
    database = conn.execute(
        "SELECT file FROM pragma_database_list WHERE name = 'main'"
    ).fetchone()
    if (
        database is None
        or not str(database["file"])
        or Path(str(database["file"])).absolute() != binding.db_path
    ):
        raise StrategyError("ImpactCube binding database changed")
    cube = validate_strategy_impact_cube_artifact_binding(binding)
    if (
        binding.tasks_root != binding.tasks_root.absolute()
        or binding.artifact_path
        != binding.tasks_root
        / binding.task_id
        / "strategy_impact_cubes"
        / f"{cube['cube_id']}.json"
    ):
        raise StrategyError("ImpactCube artifact path changed")
    row = conn.execute(
        """
        SELECT id, task_id, kind, path, content_hash, origin_tool,
               provenance_json
          FROM task_artifacts
         WHERE task_id = ? AND id = ?
        """,
        (binding.task_id, binding.artifact_id),
    ).fetchone()
    if (
        row is None
        or str(row["id"]) != binding.artifact_id
        or str(row["task_id"]) != binding.task_id
        or str(row["kind"]) != IMPACT_CUBE_ARTIFACT_KIND
        or str(row["path"]) != str(binding.artifact_path)
        or not hmac.compare_digest(
            str(row["content_hash"]),
            binding.artifact_content_hash,
        )
        or str(row["origin_tool"]) != IMPACT_CUBE_ORIGIN_TOOL
        or str(row["provenance_json"])
        != binding.artifact_provenance_json
    ):
        raise StrategyError("ImpactCube artifact registry binding changed")
    raw = _read_impact_cube_source_file(
        binding.artifact_path,
        root=binding.tasks_root,
        expected_hash=binding.artifact_content_hash,
    )
    if raw != canonical_strategy_impact_cube_json(cube).encode("utf-8"):
        raise StrategyError("ImpactCube artifact bytes changed")
    require_impact_cube_measurement_audit_on_connection(
        conn,
        binding.artifact_provenance["producer_run"],
    )


def _require_impact_cube_provenance(
    binding: StrategyImpactCubeArtifactBinding,
    cube: Mapping[str, Any],
) -> None:
    try:
        canonical = json.dumps(
            binding.artifact_provenance,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        provenance = json.loads(canonical)
    except (TypeError, ValueError, RecursionError) as exc:
        raise StrategyError(
            "ImpactCube artifact provenance is invalid"
        ) from exc
    schema_version = (
        provenance.get("schema_version")
        if isinstance(provenance, dict)
        else None
    )
    expected_fields = (
        _IMPACT_CUBE_PROVENANCE_FIELDS
        if schema_version == IMPACT_CUBE_ARTIFACT_SCHEMA_VERSION
        else _IMPACT_CUBE_REQUIREMENTS_PROVENANCE_FIELDS
        if schema_version == IMPACT_CUBE_REQUIREMENTS_ARTIFACT_SCHEMA_VERSION
        else None
    )
    if (
        expected_fields is None
        or set(provenance) != expected_fields
        or binding.artifact_provenance_json != canonical
    ):
        raise StrategyError(
            "ImpactCube artifact provenance fields changed"
        )
    if schema_version == IMPACT_CUBE_REQUIREMENTS_ARTIFACT_SCHEMA_VERSION:
        try:
            requirement_bindings = (
                validate_pool_requirement_bindings_provenance(
                    provenance["requirement_bindings"]
                )
            )
        except StrategyError as exc:
            raise StrategyError(
                "ImpactCube requirement bindings are invalid"
            ) from exc
        if not requirement_bindings["requirements"]:
            raise StrategyError(
                "ImpactCube requirements provenance must not be empty"
            )
    if (
        provenance["producer_version"]
        != STRATEGY_IMPACT_CUBE_PRODUCER_VERSION
        or provenance["task_id"] != binding.task_id
        or provenance["cube_id"] != cube["cube_id"]
        or provenance["cube_content_hash"] != cube["content_hash"]
        or provenance["populations"] != ["approval", "risk"]
        or provenance["lifecycle"] != cube["lifecycle"]
    ):
        raise StrategyError(
            "ImpactCube artifact provenance identity changed"
        )
    sources = cube["source_bindings"]
    if (
        provenance["dataset_binding"] != sources["dataset"]
        or provenance["target_binding"] != sources["target"]
        or provenance["dimension_bindings"]
        != {
            key: sources["fields"][key]
            for key in ("month_col", "group_col", "segment_col")
        }
    ):
        raise StrategyError(
            "ImpactCube artifact provenance source binding changed"
        )
    current = sources["current_strategy"]
    expected_current = (
        current["value"]
        if current["availability"] == "present"
        else None
    )
    if provenance["current_strategy_ref"] != expected_current:
        raise StrategyError(
            "ImpactCube artifact provenance current strategy changed"
        )
    economics = sources["economics"]
    expected_economics = provenance["economics_inputs"]
    if expected_economics is None:
        expected_absence = (
            ("not_applicable", "segmentation_has_no_economic_contract")
            if cube["identity"]["strategy_type"] == "segmentation"
            else ("unavailable", "economics_inputs_not_provided")
        )
        if (
            (
                economics["availability"],
                economics["reason"],
            )
            != expected_absence
            or economics["bindings"] != {}
        ):
            raise StrategyError(
                "ImpactCube artifact provenance economics changed"
            )
    elif expected_economics != economics["bindings"]:
        raise StrategyError(
            "ImpactCube artifact provenance economics changed"
        )
    expected_partitions = [
        item["name"]
        for item in cube["partitions"]
        if item["role"] == "risk"
    ]
    if provenance["partitions"] != expected_partitions:
        raise StrategyError(
            "ImpactCube artifact provenance partitions changed"
        )
    identity = cube["identity"]
    pool_artifact = sources["pool_artifact"]
    expected_pool_ref = {
        "artifact_id": pool_artifact["artifact_id"],
        "expected_artifact_content_hash": pool_artifact[
            "artifact_content_hash"
        ],
        "expected_pool_id": identity["pool_id"],
        "expected_revision": identity["revision"],
        "expected_revision_id": identity["revision_id"],
        "expected_snapshot_hash": identity["snapshot_hash"],
    }
    if provenance["pool_ref"] != expected_pool_ref:
        raise StrategyError(
            "ImpactCube artifact provenance Pool binding changed"
        )
    sample = sources["sample_design_v2"]
    expected_sample_ref = {
        "membership_artifact_id": sample["membership_artifact_id"],
        "expected_membership_artifact_content_hash": sample[
            "membership_artifact_content_hash"
        ],
        "bundle_artifact_id": sample["bundle_artifact_id"],
        "expected_bundle_artifact_content_hash": sample[
            "bundle_artifact_content_hash"
        ],
        "expected_bundle_id": sample["bundle_id"],
        "expected_sample_design_id": sample["sample_design_id"],
        "expected_sample_design_content_hash": sample[
            "sample_design_content_hash"
        ],
    }
    if provenance["sample_design_ref"] != expected_sample_ref:
        raise StrategyError(
            "ImpactCube artifact provenance sample-design binding changed"
        )
    request = {
        "strategy_type": cube["identity"]["strategy_type"],
        "pool_ref": provenance["pool_ref"],
        "sample_design_ref": provenance["sample_design_ref"],
        "partitions": provenance["partitions"],
        "population": "risk",
        "dimension_bindings": provenance["dimension_bindings"],
        "current_strategy_ref": (
            None
            if provenance["current_strategy_ref"] is None
            else {
                "strategy_id": provenance["current_strategy_ref"][
                    "strategy_id"
                ],
                "expected_strategy_spec_hash": provenance[
                    "current_strategy_ref"
                ]["strategy_spec_hash"],
            }
        ),
        "economics_inputs": provenance["economics_inputs"],
    }
    try:
        validate_impact_cube_producer_run(
            provenance["producer_run"],
            expected_task_id=binding.task_id,
            expected_request=request,
            expected_cube_id=cube["cube_id"],
            expected_cube_content_hash=cube["content_hash"],
            expected_artifact_id=binding.artifact_id,
            expected_artifact_filename=binding.artifact_path.name,
            expected_artifact_content_hash=(
                binding.artifact_content_hash
            ),
        )
    except StrategyError as exc:
        raise StrategyError(
            f"ImpactCube producer_run is invalid: {exc}"
        ) from exc


def _read_impact_cube_source_file(
    path: Path,
    *,
    root: Path,
    expected_hash: str,
) -> bytes:
    if not path.is_absolute() or path.is_symlink():
        raise StrategyError("ImpactCube artifact must be a regular file")
    current = path.parent
    while current != root:
        if current.is_symlink():
            raise StrategyError(
                "ImpactCube artifact path traverses a symlink"
            )
        if current == current.parent:
            break
        current = current.parent
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise StrategyError(
            "ImpactCube artifact escaped task storage"
        ) from exc
    descriptor = -1
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    total = 0
    before = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise StrategyError(
                "ImpactCube artifact must be a regular file"
            )
        if (
            before.st_size < 0
            or before.st_size > MAX_IMPACT_CUBE_JSON_BYTES
        ):
            raise StrategyError("ImpactCube artifact exceeds byte budget")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_IMPACT_CUBE_JSON_BYTES:
                raise StrategyError(
                    "ImpactCube artifact exceeds byte budget"
                )
            digest.update(chunk)
            chunks.append(chunk)
        after = os.fstat(descriptor)
        live = os.stat(path, follow_symlinks=False)
        if (
            (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            )
            or not stat.S_ISREG(live.st_mode)
            or (live.st_dev, live.st_ino)
            != (after.st_dev, after.st_ino)
        ):
            raise StrategyError(
                "ImpactCube artifact changed while being read"
            )
    except StrategyError:
        raise
    except OSError as exc:
        raise StrategyError("ImpactCube artifact is unavailable") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    assert before is not None
    raw = b"".join(chunks)
    if (
        len(raw) != before.st_size
        or not hmac.compare_digest(digest.hexdigest(), expected_hash)
    ):
        raise StrategyError("ImpactCube artifact bytes or hash changed")
    return raw


def _require_canonical_artifact_hash(
    supplied: object,
    canonical: str,
    name: str,
) -> None:
    if not isinstance(supplied, str):
        raise StrategyError(
            f"{name} artifact content hash is invalid"
        )
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if supplied != expected:
        raise StrategyError(
            f"{name} artifact content hash does not match canonical evidence"
        )


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrategyError(f"JSON contains duplicate key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise StrategyError(f"JSON contains non-finite constant: {value}")


def _canonical_object(value: object, name: str) -> dict[str, Any]:
    try:
        raw = _canonical_json(value)
        normalized = json.loads(raw)
    except (TypeError, ValueError, RecursionError, MemoryError) as exc:
        raise StrategyError(f"{name} must be finite canonical JSON") from exc
    if not isinstance(normalized, dict):
        raise StrategyError(f"{name} must be an object")
    return normalized


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise StrategyError(f"{name} must be a lowercase SHA-256 hash")
    return value


__all__ = [
    "StrategyImpactCubeArtifactBinding",
    "load_strategy_impact_cube_artifact",
    "require_strategy_impact_cube_artifact_binding_on_connection",
    "validate_strategy_impact_cube_artifact_binding",
]
