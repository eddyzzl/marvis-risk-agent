"""Capture native feature-selection provenance without certifying independence.

One authenticated private snapshot serves all column batches. Receipts describe
the actual candidate algorithm's source-row scope and exposed diagnostics; they
do not infer an upstream selection, human reasoning or historical availability.
"""

import base64
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from filelock import FileLock, Timeout as FileLockTimeout

from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.authenticated_snapshot import authenticated_file_snapshot
from marvis.data.dataset_identity import dataset_identity_equal
from marvis.data.preprocessing_evidence import verify_source_on_connection
from marvis.feature.errors import FeatureError
from marvis.packs.modeling._common import _jsonable
from marvis.repositories.task_artifacts import TaskArtifactRepository


KIND = "modeling_feature_selection"
VERSION = "marvis.feature_selection.v1"
MAX_RECEIPT_BYTES = 16 * 1024**2
_ORIGINS = frozenset({"modeling.select_features", "modeling.screen_features", "modeling.screen_features_non_binary"})


class _SnapshotBackend:
    def __init__(self, stream):
        self.stream = stream
        parquet = pq.ParquetFile(stream)
        self.names = tuple(parquet.schema_arrow.names)
        self.row_count = parquet.metadata.num_rows

    def column_names(self, _path):
        return self.names

    def read_frame(self, _path, *, columns=None):
        self.stream.seek(0)
        return pd.read_parquet(self.stream, columns=columns)


def _membership(raw, row_count):
    if len(raw) != (row_count + 7) // 8:
        raise FeatureError("selection membership length differs from source")
    mask = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")
    if mask[row_count:].any():
        raise FeatureError("selection membership contains out-of-range rows")
    return {
        "membership": base64.b64encode(raw).decode("ascii"),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "rows": int(mask[:row_count].sum()),
    }


def run_recorded_selection(runtime, ctx, dataset, function, **parameters):
    # Only platform-selected functions arrive here, never a function named by
    # an evidence file. Freeze their effective defaults alongside user options.
    bound = inspect.signature(function).bind(None, Path("snapshot.parquet"), **parameters)
    bound.apply_defaults()
    effective = {key: _jsonable(value) for key, value in bound.arguments.items()
                 if key not in {"backend", "dataset_path"}}
    path = runtime.registry.resolve_verified_path(dataset.id)
    with authenticated_file_snapshot(
        path, root=runtime.registry.datasets_root, expected_sha256=dataset.content_hash,
    ) as stream:
        backend = _SnapshotBackend(stream)
        if backend.row_count != dataset.row_count:
            raise FeatureError("selection source row count changed")
        result = function(backend, path, **parameters)
    scope = result.selection_scope
    if scope is None or scope.row_count != dataset.row_count:
        raise FeatureError("native selection did not capture its source membership")
    payload = {
        "schema_version": VERSION,
        "tool": f"modeling.{function.__name__}",
        "dataset_id": dataset.id, "source_content_hash": dataset.content_hash,
        "source_target": [dataset.has_target, dataset.target_col],
        "row_count": dataset.row_count,
        "parameters": effective,
        "candidates": list(parameters["features"]),
        "selected": list(result.selected),
        "memberships": {
            "fit": _membership(scope.fit, scope.row_count),
            "label_diagnostics": _membership(scope.label_diagnostics, scope.row_count),
            "value_diagnostics": _membership(scope.value_diagnostics, scope.row_count),
        },
        "scope": "core_candidate_algorithm_and_ks_psi_diagnostics",
        "auxiliary_diagnostics": {
            "assurance": "unknown",
            "excluded": ["candidate_inference", "categorical_hints", "dictionary", "human_decisions"],
            "zero_membership_means": "no_recorded_core_diagnostic_use_only",
        },
        "upstream_selection": "unknown",
        "independent_inner_validation": "not_established",
        "historical_availability": "unknown",
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    if len(raw) > MAX_RECEIPT_BYTES:
        raise FeatureError("selection receipt exceeds size limit")
    content_hash = hashlib.sha256(raw).hexdigest()
    root = runtime.registry.datasets_root
    repo = TaskArtifactRepository(runtime.registry._repo.db_path)
    destination = root / ctx.task_id / "selection" / f"{content_hash}.json"
    relative = destination.relative_to(root.parent).as_posix()
    # Serialize the full filesystem + DB unit, including rollback. A DB lock
    # acquired only after promotion cannot stop a failed publisher from deleting
    # an identical receipt another publisher has already committed.
    task_dir = root / ctx.task_id
    if task_dir.resolve(strict=True) != task_dir or not task_dir.is_dir():
        raise FeatureError("selection evidence task path is invalid")
    lock_path = task_dir / ".selection-evidence.lock"
    if lock_path.is_symlink():
        raise FeatureError("selection evidence lock path is invalid")
    lock = FileLock(str(lock_path), mode=0o600)
    try:
        lock.acquire(timeout=30)
    except FileLockTimeout as exc:
        raise FeatureError("selection evidence publication is busy; retry this task") from exc
    try:
        return result, _publish_receipt(runtime, ctx, dataset, repo, destination, relative,
                                       payload, raw, content_hash)
    finally:
        # Keep the lock file: deleting it can give concurrent processes different
        # inodes and therefore different locks for the same publication.
        lock.release()


def _publish_receipt(runtime, ctx, dataset, repo, destination, relative, payload, raw, content_hash):
    existing = repo.get_for_task_kind_path(ctx.task_id, KIND, relative)
    if existing is not None:
        reference = {"artifact_id": existing["id"], "content_hash": content_hash}
        load_selection_evidence(runtime.registry, ctx.task_id, reference)
        return reference
    uow = ArtifactUnitOfWork()
    artifact = uow.stage_file(destination.parent, destination.name)
    artifact.path.write_bytes(raw)
    artifact.path.chmod(0o600)
    try:
        def commit(conn):
            conn.execute("BEGIN IMMEDIATE")
            verify_source_on_connection(runtime.registry, conn, dataset)
            if artifact.final_path.read_bytes() != raw:
                raise FeatureError("selection receipt changed before registration")
            record = repo.register_on_connection(
                conn, task_id=ctx.task_id, kind=KIND,
                path=relative,
                content_hash=content_hash, origin_tool=payload["tool"],
                provenance={"schema_version": VERSION, "dataset_id": dataset.id,
                            "source_content_hash": dataset.content_hash},
            )
            verify_source_on_connection(runtime.registry, conn, dataset)
            return record

        record = uow.finalize_with_connection(repo.transaction, commit)
    except Exception:
        uow.rollback()
        raise
    return {"artifact_id": record["id"], "content_hash": record["content_hash"]}


def load_selection_evidence(registry, task_id, reference):
    if not isinstance(reference, dict) or set(reference) != {"artifact_id", "content_hash"}:
        raise FeatureError("selection evidence reference is invalid")
    record = TaskArtifactRepository(registry._repo.db_path).get_for_task(task_id, reference["artifact_id"])
    if (record is None or record["kind"] != KIND or record["content_hash"] != reference["content_hash"]
            or record["origin_tool"] not in _ORIGINS
            or record["provenance"].get("schema_version") != VERSION):
        raise FeatureError("selection evidence identity mismatch")
    path = registry.datasets_root.parent / record["path"]
    if (path.resolve(strict=True) != path or not path.is_relative_to(registry.datasets_root)
            or not path.is_file() or path.stat().st_size > MAX_RECEIPT_BYTES):
        raise FeatureError("selection evidence path is invalid")
    with path.open("rb") as stream:
        raw = stream.read(MAX_RECEIPT_BYTES + 1)
    if len(raw) > MAX_RECEIPT_BYTES or hashlib.sha256(raw).hexdigest() != reference["content_hash"]:
        raise FeatureError("selection evidence content changed")
    payload = json.loads(raw)
    if (payload.get("schema_version") != VERSION or payload.get("tool") != record["origin_tool"]
            or payload.get("dataset_id") != record["provenance"].get("dataset_id")
            or payload.get("source_content_hash") != record["provenance"].get("source_content_hash")):
        raise FeatureError("selection evidence producer binding changed")
    source = registry.get(payload["dataset_id"])
    if (source.task_id != task_id or source.content_hash != payload["source_content_hash"]
            or source.row_count != payload["row_count"]
            or [source.has_target, source.target_col] != payload["source_target"]):
        raise FeatureError("selection evidence source changed")
    registry.resolve_verified_path(source.id)
    for entry in payload["memberships"].values():
        raw_mask = base64.b64decode(entry["membership"], validate=True)
        if _membership(raw_mask, source.row_count) != entry:
            raise FeatureError("selection evidence membership changed")
    return payload


CONSUMPTION_VERSION = 'marvis.selection_consumption.v1'


def selection_evidence_for_training(registry, task_id, dataset_id, features, target_col, references):
    """Authenticate selection exposure without certifying inner validation.

    A receipt records only the core algorithm. Human choices, candidate inference
    and auxiliary hints remain unknown even when all source memberships map.
    """
    references = normalize_selection_references(references)
    evidence = unknown_selection_evidence(dataset_id, features)
    if not references:
        return evidence
    dataset = registry.get(dataset_id)
    if dataset.task_id != task_id:
        raise FeatureError('selection training source belongs to another task')
    _verify_selection_dataset(registry, dataset)
    sources = []
    for reference in references:
        source = load_selection_evidence(registry, task_id, reference)
        if source['parameters'].get('target_col') != target_col:
            raise FeatureError('selection evidence target differs from training target')
        mapped = _selection_source_mapping(registry, dataset_id, source['dataset_id'], ())
        if mapped is None:
            raise FeatureError('selection evidence is not from the training source or an authenticated ancestor')
        positions, artifacts = mapped
        memberships = None
        if positions is not None:
            memberships = {}
            for role, entry in source['memberships'].items():
                original = np.unpackbits(np.frombuffer(base64.b64decode(entry['membership']), dtype=np.uint8),
                                         bitorder='little', count=source['row_count']).astype(bool)
                mask = np.zeros(len(positions), dtype=bool)
                present = positions >= 0
                mask[present] = original[positions[present]]
                memberships[role] = _membership(np.packbits(mask, bitorder='little').tobytes(), dataset.row_count)
        sources.append({
            'reference': dict(reference), 'tool': source['tool'],
            'source_dataset_id': source['dataset_id'], 'source_content_hash': source['source_content_hash'],
            'source_row_count': source['row_count'], 'memberships': source['memberships'],
            'candidates': source['candidates'], 'selected': source['selected'], 'parameters': source['parameters'],
            'current_row_mapping': 'recorded' if positions is not None else 'unknown',
            'current_memberships': memberships, 'mapping_artifact_ids': artifacts,
            'current_features_within_recorded_selection': set(features) <= set(source['selected']),
            'auxiliary_diagnostics': source['auxiliary_diagnostics'],
        })
    _verify_selection_dataset(registry, dataset)
    evidence.update({
        'dataset_content_hash': dataset.content_hash, 'references': references, 'sources': sources,
        'core_evidence_assurance': 'recorded',
        'current_row_mapping': 'recorded' if all(s['current_row_mapping'] == 'recorded' for s in sources) else 'unknown',
        'inner_validation': {'assurance': 'not_established', 'mode': 'conditional_on_outer_selection'},
        'reasons': ['outer_selection_is_not_fold_local', 'auxiliary_and_human_selection_not_certified'],
    })
    if not all(s['current_features_within_recorded_selection'] for s in sources):
        evidence['reasons'].append('training_features_differ_from_recorded_selection')
    return evidence


def normalize_selection_references(value):
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 16:
        raise FeatureError('selection_evidence_refs must be a list of at most 16 native references')
    result = []
    for item in value:
        if (not isinstance(item, dict) or set(item) != {'artifact_id', 'content_hash'}
                or not isinstance(item['artifact_id'], str) or not item['artifact_id'].strip()
                or not isinstance(item['content_hash'], str) or len(item['content_hash']) != 64
                or any(c not in '0123456789abcdef' for c in item['content_hash'])):
            raise FeatureError('selection evidence reference is invalid')
        if item not in result:
            result.append(dict(item))
    return result


def unknown_selection_evidence(dataset_id='', features=()):
    return {
        'schema_version': CONSUMPTION_VERSION,
        'scope': 'recorded_core_selection_input_exposure',
        'dataset_id': dataset_id, 'features': list(features),
        'assurance': 'unknown', 'core_evidence_assurance': 'unknown',
        'current_row_mapping': 'unknown', 'references': [], 'sources': [],
        'auxiliary_diagnostics': {'assurance': 'unknown'},
        'inner_validation': {'assurance': 'not_established', 'mode': 'unknown'},
        'reasons': ['selection_evidence_not_recorded'],
    }


def _selection_source_mapping(registry, current_id, source_id, seen):
    """Return current-to-source positions only for authenticated native maps."""
    if current_id in seen or len(seen) >= 64:
        raise FeatureError('selection source lineage is cyclic or exceeds 64 transforms')
    dataset = registry.get(current_id)
    _verify_selection_dataset(registry, dataset)
    result = _selection_source_mapping_checked(registry, dataset, source_id, seen)
    _verify_selection_dataset(registry, dataset)
    return result


def _verify_selection_dataset(registry, dataset):
    # Evidence reads must preserve paths already frozen in SampleDesign.
    if not dataset_identity_equal(registry.get(dataset.id), dataset):
        raise FeatureError('selection dataset binding changed during verification')
    registry.resolve_verified_path(dataset.id)
    if not dataset_identity_equal(registry.get(dataset.id), dataset):
        raise FeatureError('selection dataset binding changed during verification')


def _selection_source_mapping_checked(registry, dataset, source_id, seen):
    from marvis.data.asof_join import AsOfJoinEngine
    from marvis.data.preprocessing_evidence import load_preprocessing_state
    from marvis.data.join_evidence import join_time_parent
    from marvis.data.transform_time import transform_time_parent

    current_id = dataset.id
    if current_id == source_id:
        return np.arange(dataset.row_count), []
    repo = TaskArtifactRepository(registry._repo.db_path)
    state = load_preprocessing_state(registry, current_id)
    if state.artifact_id:
        proof = repo.get_for_task(dataset.task_id, state.artifact_id)['provenance']
        result = _selection_source_mapping(registry, proof['source_dataset_id'], source_id, (*seen, current_id))
        if result is None:
            return None
        positions, artifacts = result
        return positions, [*artifacts, state.artifact_id]
    native = AsOfJoinEngine(registry, repo, workspace_root=registry.datasets_root.parent).row_input_time_evidence(current_id)
    if native:
        matches = []
        for index, role in enumerate(('decision', 'feature')):
            parent = native['parents'][role]
            source = registry.get(parent['dataset_id'])
            if source.task_id != dataset.task_id or source.content_hash != parent['content_hash']:
                raise FeatureError('selection ancestor identity changed')
            result = _selection_source_mapping(registry, parent['dataset_id'], source_id, (*seen, current_id))
            _verify_selection_dataset(registry, source)
            if result is None:
                continue
            positions, artifacts = result
            mapped = None
            if positions is not None:
                mapped = np.array([positions[pair[index]] if pair[index] is not None else -1
                                   for pair in native['memberships']], dtype=np.int64)
            matches.append((mapped, [*artifacts, native['artifact_id']]))
        if len(matches) == 1:
            return matches[0]
        if matches:
            return None, sorted({item for _, artifacts in matches for item in artifacts})
        return None
    transformed = transform_time_parent(registry, repo, dataset)
    joined = join_time_parent(registry, repo, dataset) if transformed is None else None
    parent_id = transformed.source_dataset_id if transformed else joined[0] if joined else None
    if parent_id:
        result = _selection_source_mapping(registry, parent_id, source_id, (*seen, current_id))
        if result is not None:
            # Existing readers certify producer ancestry, but do not expose an
            # unambiguous physical row map for this consumer's narrower scope.
            return None, [*result[1], transformed.result_artifact_id if transformed else joined[2]]
    return None
