from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import hmac
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.parse import quote
import uuid

import pandas as pd

from marvis.data.data_dictionary import first_data_dictionary_id, load_business_names
from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.align import ColumnAligner
from marvis.data.backend import (
    connect_duckdb,
    sql_identifier,
)
from marvis.data.contracts import SMALL_SAMPLE_N
from marvis.data.dedup import two_level_dedup
from marvis.data.errors import (
    DatasetTooLargeError,
    DedupRequiredError,
    KeyDtypeMismatchError,
)
from marvis.data.excel_ingest import (
    ingest_sheet,
    list_sheets,
    new_excel_artifact_name,
)
from marvis.data.dataset_export import export_dataset
from marvis.data.join_engine import JoinEngine, _key_fps
from marvis.data.workspace import data_semantic_mapping_hash
from marvis.data.transform_semantics import (
    effective_transform_semantic_mapping,
    migrate_transform_semantics,
)
from marvis.data.transforms import transform_parquet
from marvis.provenance import NumberProvenance
from marvis.reconcile import EXACT_ABS_TOL, EXACT_REL_TOL, ReconcileReport, reconcile
from marvis.db_schema import connect
from marvis.plugins.sdk import PackRuntime
from marvis.repositories.strategy import _write_audit_row
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.data_transform import (
    DATA_TRANSFORM_ARTIFACT_KIND,
    DATA_TRANSFORM_ORIGIN_TOOL,
    DataTransformIdentity,
    DataTransformRepository,
    data_transform_artifact_provenance,
)
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.files import sha256_file
from marvis.safe_paths import assert_within


def tool_ingest_excel(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    path = _resolve_material_path(str(inputs["path"]), ctx)
    size_bytes = path.stat().st_size
    if size_bytes > runtime.settings.max_excel_upload_bytes:
        raise DatasetTooLargeError(
            reason="Excel 材料文件大小超过上限",
            limit=runtime.settings.max_excel_upload_bytes,
            actual=size_bytes,
        )
    requested_sheets = [str(sheet) for sheet in (inputs.get("sheets") or list_sheets(path))]
    role = str(inputs.get("role") or "feature")
    out_dir = runtime.datasets_root / ctx.task_id / "excel"
    out_dir.mkdir(parents=True, exist_ok=True)
    uow = ArtifactUnitOfWork()
    staged_sheets = []
    reports = []
    try:
        with tempfile.TemporaryDirectory(prefix=".excel_ingest_", dir=out_dir) as scratch:
            scratch_dir = Path(scratch)
            for sheet in requested_sheets:
                parquet_path, report = ingest_sheet(
                    path,
                    sheet,
                    scratch_dir,
                    max_rows=runtime.settings.max_excel_rows,
                )
                artifact = uow.stage_file(
                    out_dir,
                    new_excel_artifact_name(report.sheet),
                )
                shutil.move(parquet_path, artifact.path)
                staged_sheets.append((artifact.final_path, report))
                reports.append({
                    "sheet": report.sheet,
                    "header_rows": report.header_rows,
                    "data_start_row": report.data_start_row,
                    "flattened_columns": report.flattened_columns,
                    "original_shape": list(report.original_shape),
                    "warnings": [],
                })
        registered = uow.finalize_with_connection(
            runtime.registry.transaction,
            lambda conn: [
                runtime.registry.register_existing_on_connection(
                    conn,
                    parquet_path,
                    task_id=ctx.task_id,
                    role=role,
                    seed=_seed(ctx),
                )
                for parquet_path, _report in staged_sheets
            ],
        )
    except Exception:
        uow.rollback()
        raise
    return {
        "datasets": [_dataset_payload(dataset) for dataset in registered],
        "reports": reports,
    }


def _resolve_material_path(raw_path: str, ctx) -> Path:
    path = Path(raw_path).expanduser()
    resolved = path.resolve(strict=True)
    roots = _allowed_material_roots(ctx)
    if any(_path_is_within(root, resolved) for root in roots):
        return resolved
    allowed = ", ".join(str(root) for root in roots)
    raise PermissionError(
        f"Excel path must be under an allowed material root: {allowed}. "
        "Set RMC_MATERIAL_ROOTS to allow another local material directory."
    )


def _allowed_material_roots(ctx) -> tuple[Path, ...]:
    roots = [Path(ctx.workspace), Path.home()]
    extra_roots = os.environ.get("RMC_MATERIAL_ROOTS", "")
    roots.extend(Path(raw).expanduser() for raw in extra_roots.split(os.pathsep) if raw)
    resolved: list[Path] = []
    for root in roots:
        candidate = root.resolve()
        if candidate not in resolved:
            resolved.append(candidate)
    return tuple(resolved)


def _path_is_within(root: Path, candidate: Path) -> bool:
    try:
        assert_within(root, candidate)
    except PermissionError:
        return False
    return True


def tool_infer_schema(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    return {
        "dataset_id": dataset.id,
        "columns": [_column_payload(column) for column in dataset.columns],
        "has_target": dataset.has_target,
        "target_col": dataset.target_col,
    }


_PROFILE_SECTIONS = (
    "overview",
    "target",
    "missing",
    "distribution",
    "correlation",
)
_PROFILE_CONFIG_FIELDS = (
    "frequency_top_k",
    "low_cardinality_threshold",
    "histogram_bins",
    "correlation_batch_size",
)
_PROFILE_SENSITIVE_ROLES = frozenset({"phone", "idcard", "id", "name"})


def tool_profile_dataset(inputs: dict, ctx) -> dict:
    """Run a full deterministic profile of the exact active workspace dataset.

    The five identity inputs are deliberately mandatory even though ``dataset_id``
    alone could locate a registry row.  They bind the result to the user's active
    data-workspace revision, physical bytes, analysis generation, and confirmed
    semantic choices; a stale plan fails before the descriptive kernel runs.
    """

    runtime = _runtime(ctx)
    task_id = str(ctx.task_id)
    dataset_id = str(inputs["dataset_id"])
    dataset = runtime.registry.get(dataset_id)
    if str(dataset.task_id) != task_id:
        raise PermissionError(
            f"dataset {dataset_id} belongs to task {dataset.task_id}, not {task_id}"
        )

    expected_hash = str(inputs["expected_content_hash"])
    registered_hash = getattr(dataset, "content_hash", None)
    if not isinstance(registered_hash, str) or not hmac.compare_digest(
        registered_hash,
        expected_hash,
    ):
        raise ValueError("dataset content hash does not match expected content hash")

    snapshot = DataWorkspaceRepository(runtime.settings.db_path).get_or_default(task_id)
    if snapshot.active_dataset_id != dataset_id:
        raise ValueError("dataset is not the active data-workspace dataset")
    active_hash = snapshot.active_dataset_content_hash
    if not isinstance(active_hash, str) or not hmac.compare_digest(
        active_hash,
        expected_hash,
    ):
        raise ValueError("active data-workspace content hash does not match expected content hash")

    workspace_revision = int(inputs["workspace_revision"])
    if snapshot.revision != workspace_revision:
        raise ValueError(
            "data workspace revision mismatch: "
            f"expected {workspace_revision}, found {snapshot.revision}"
        )
    analysis_generation = int(inputs["analysis_generation"])
    if snapshot.analysis_generation != analysis_generation:
        raise ValueError(
            "data workspace analysis generation mismatch: "
            f"expected {analysis_generation}, found {snapshot.analysis_generation}"
        )

    expected_semantic_hash = str(inputs["semantic_mapping_hash"])
    actual_semantic_hash = data_semantic_mapping_hash(snapshot.semantic_mapping)
    if not hmac.compare_digest(actual_semantic_hash, expected_semantic_hash):
        raise ValueError("data workspace semantic mapping hash mismatch")

    workspace_target = snapshot.semantic_mapping.target_col
    if "target_col" in inputs and inputs["target_col"] != workspace_target:
        raise ValueError("target_col must match the data workspace semantic mapping")

    sections = _profile_sections(inputs.get("sections"))
    columns = _profile_columns(inputs.get("columns"))
    config = _build_descriptive_config(inputs)
    verified_path = runtime.registry.resolve_verified_path(dataset_id)
    value_sanitizers = _profile_value_sanitizers(
        dataset,
        snapshot.semantic_mapping,
        dataset_content_hash=registered_hash,
    )
    report = _analyze_parquet(
        verified_path,
        temp_directory=runtime.backend._temp_directory,
        target_column=workspace_target,
        columns=columns,
        config=config,
        value_sanitizers=value_sanitizers,
    )
    report = _suppress_sensitive_profile_values(
        report,
        sensitive_columns=frozenset(value_sanitizers),
    )
    row_count = _profile_row_count(report)
    registered_row_count = int(dataset.row_count)
    if row_count != registered_row_count:
        raise ValueError(
            "full dataset row count does not match the registered dataset: "
            f"scanned {row_count}, registered {registered_row_count}"
        )

    config_payload = config.to_dict()
    semantics = _profile_semantics(report, snapshot.semantic_mapping)
    result = _select_profile_sections(
        report,
        sections=sections,
        target_column=workspace_target,
    )
    return {
        "dataset_id": dataset_id,
        "dataset_content_hash": registered_hash,
        "expected_content_hash": expected_hash,
        "workspace_revision": workspace_revision,
        "analysis_generation": analysis_generation,
        "semantic_mapping_hash": actual_semantic_hash,
        "scan_scope": "full_dataset",
        "row_count": row_count,
        "row_count_scanned": row_count,
        "options_echo": {
            "sections": list(sections),
            "columns": None if columns is None else list(columns),
            "target_col": workspace_target,
            **{name: int(config_payload[name]) for name in _PROFILE_CONFIG_FIELDS},
        },
        "semantics": semantics,
        "result": result,
    }


def _build_descriptive_config(inputs: dict):
    from marvis.data.descriptive import DescriptiveConfig

    overrides = {
        name: int(inputs[name])
        for name in _PROFILE_CONFIG_FIELDS
        if name in inputs
    }
    return DescriptiveConfig(**overrides)


def _analyze_parquet(path: Path, **kwargs) -> dict:
    from marvis.data.descriptive import analyze_parquet

    return analyze_parquet(path, **kwargs)


def _profile_sections(raw) -> tuple[str, ...]:
    if raw is None:
        return _PROFILE_SECTIONS
    requested = [str(item) for item in raw]
    unknown = sorted(set(requested) - set(_PROFILE_SECTIONS))
    if unknown:
        raise ValueError(f"unsupported profile section(s): {', '.join(unknown)}")
    if not requested:
        raise ValueError("sections must not be empty")
    if len(requested) != len(set(requested)):
        raise ValueError("sections must not contain duplicates")
    requested_set = set(requested)
    return tuple(section for section in _PROFILE_SECTIONS if section in requested_set)


def _profile_columns(raw) -> tuple[str, ...] | None:
    if raw is None:
        return None
    columns = tuple(str(item).strip() for item in raw)
    if not columns or any(not column for column in columns):
        raise ValueError("columns must contain at least one non-empty column name")
    if len(columns) != len(set(columns)):
        raise ValueError("columns must not contain duplicates")
    return columns


def _profile_value_sanitizers(
    dataset,
    semantic_mapping,
    *,
    dataset_content_hash: str,
) -> dict:
    sensitive_columns = {
        str(column.name)
        for column in dataset.columns
        if str(column.semantic_role) in _PROFILE_SENSITIVE_ROLES
    }
    sensitive_columns.update(
        str(name)
        for name, role in semantic_mapping.field_roles.items()
        if str(role) in _PROFILE_SENSITIVE_ROLES
    )
    return {
        column: _profile_value_sanitizer(dataset_content_hash, column)
        for column in sorted(sensitive_columns)
    }


def _profile_value_sanitizer(dataset_content_hash: str, column: str):
    def sanitize(tagged_value: dict) -> dict:
        canonical = json.dumps(
            tagged_value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        token = hashlib.sha256(
            (
                "dataset-profile.v1\0"
                f"{dataset_content_hash}\0{column}\0{canonical}"
            ).encode("utf-8")
        ).hexdigest()[:16]
        return {"type": "string", "value": f"token:{token}"}

    return sanitize


def _profile_row_count(report: dict) -> int:
    try:
        value = report["dataset"]["row_count"]
    except (KeyError, TypeError) as exc:
        raise ValueError("descriptive result is missing dataset.row_count") from exc
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("descriptive result dataset.row_count must be a non-negative integer")
    return value


def _profile_semantics(report: dict, semantic_mapping) -> dict:
    result_columns = [
        str(field["name"])
        for field in report.get("fields") or []
        if isinstance(field, dict) and field.get("name") is not None
    ]
    return {
        "target_col": semantic_mapping.target_col,
        "field_roles": {
            column: str(semantic_mapping.field_roles[column])
            for column in result_columns
            if column in semantic_mapping.field_roles
        },
        "business_names": {
            column: str(semantic_mapping.business_names[column])
            for column in result_columns
            if column in semantic_mapping.business_names
        },
    }


def _suppress_sensitive_profile_values(
    report: dict,
    *,
    sensitive_columns: frozenset[str],
) -> dict:
    if not sensitive_columns:
        return report

    sanitized = deepcopy(report)
    for field in sanitized.get("fields") or []:
        if str(field.get("name")) not in sensitive_columns:
            continue
        field["numeric"] = None
        field["histogram"] = None
        field["sensitive_value_policy"] = (
            "frequency_tokenized_numeric_distribution_suppressed"
        )

    correlations = sanitized.get("correlations")
    if not isinstance(correlations, dict):
        return sanitized
    names = [str(name) for name in correlations.get("columns") or []]
    kept_indices = [
        index for index, name in enumerate(names) if name not in sensitive_columns
    ]
    correlations["columns"] = [names[index] for index in kept_indices]
    for matrix_name in ("values", "pair_counts", "reasons"):
        matrix = correlations.get(matrix_name)
        if not isinstance(matrix, list):
            continue
        correlations[matrix_name] = [
            [row[column_index] for column_index in kept_indices]
            for row_index in kept_indices
            if isinstance((row := matrix[row_index]), list)
        ]
    return sanitized


def _select_profile_sections(
    report: dict,
    *,
    sections: tuple[str, ...],
    target_column: str | None,
) -> dict:
    if sections == _PROFILE_SECTIONS:
        return report

    selected = deepcopy(report)
    requested = set(sections)
    if "target" not in requested:
        selected["target_distribution"] = {
            "status": "not_requested",
            "column": target_column,
        }
    if "correlation" not in requested:
        selected["correlations"] = {
            "status": "not_requested",
            "columns": [],
            "values": [],
            "pair_counts": [],
            "reasons": [],
        }

    field_sections = requested & {"overview", "missing", "distribution"}
    if not field_sections:
        selected["fields"] = []
        return selected

    base_keys = {
        "name",
        "duckdb_type",
        "kind",
        "selection_role",
        "sensitive_value_policy",
    }
    section_keys = {
        "overview": {"row_count", "distinct_count"},
        "missing": {"row_count", "null_count", "null_rate"},
        "distribution": {"distinct_count", "numeric", "frequency", "histogram"},
    }
    allowed_keys = set(base_keys)
    for section in field_sections:
        allowed_keys.update(section_keys[section])
    selected["fields"] = [
        {key: value for key, value in field.items() if key in allowed_keys}
        for field in report.get("fields") or []
    ]
    return selected


def tool_align_columns(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    anchor = runtime.registry.get(str(inputs["anchor_id"]))
    anchor_path = runtime.registry.resolve_path(anchor.id)
    alignments = []
    for feature_id in inputs.get("feature_ids") or []:
        feature = runtime.registry.get(str(feature_id))
        key_pairs = runtime.aligner.align(
            anchor,
            anchor_path,
            feature,
            runtime.registry.resolve_path(feature.id),
            seed=_seed(ctx),
        )
        alignments.append({
            "feature_id": feature.id,
            "key_pairs": [_key_pair_payload(pair) for pair in key_pairs],
        })
    return {"alignments": alignments}


def tool_propose_join(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    anchor_id = str(inputs["anchor_id"])
    feature_ids = [str(item) for item in inputs.get("feature_ids") or []]
    seed = _seed(ctx)
    plan = runtime.join_engine.propose_join_plan(
        anchor_id,
        feature_ids,
        ctx.task_id,
        seed=seed,
    )
    key_overrides = inputs.get("key_overrides")
    if isinstance(key_overrides, dict) and key_overrides:
        anchor = runtime.registry.get(anchor_id)
        anchor_path = runtime.registry.resolve_path(anchor_id)
        for spec in plan.joins:
            requested = key_overrides.get(spec.feature_dataset_id)
            if requested is None:
                continue
            selected = [str(column) for column in requested]
            selected_set = set(selected)
            if not selected or len(selected_set) != len(selected):
                raise ValueError(f"feature {spec.feature_dataset_id} must select one or more unique join keys")
            known = {pair.anchor_col for pair in spec.key_pairs}
            unknown = sorted(selected_set - known)
            if unknown:
                raise ValueError(
                    f"feature {spec.feature_dataset_id} has unknown anchor join keys: {', '.join(unknown)}"
                )
            pairs = [pair for pair in spec.key_pairs if pair.anchor_col in selected_set]
            feature = runtime.registry.get(spec.feature_dataset_id)
            feature_path = runtime.registry.resolve_path(feature.id)
            spec.key_pairs = runtime.join_engine.recompute_key_pairs(
                anchor,
                anchor_path,
                feature,
                feature_path,
                pairs,
                seed=seed,
            )
            spec.diagnostics = runtime.join_engine.diagnose_join(
                anchor,
                anchor_path,
                feature,
                feature_path,
                spec.key_pairs,
                seed=seed,
                recompute_match=True,
            )
            runtime.repo.update_join_spec(plan.id, spec)
        plan = runtime.repo.load_join_plan(plan.id)
    payload = _join_plan_payload(plan)
    for join in payload.get("joins", []):
        join["feature_name"] = _friendly_name(runtime.registry, join.get("feature_id"))
    # GAP-4: {column: business_name} map for every key column in this proposal, so
    # the C1/dedup gate can show a meaning tooltip next to raw column-name codes.
    # Best-effort — {} when the task has no registered data dictionary.
    dictionary = _join_dictionary(runtime, ctx, payload)
    if dictionary:
        payload["dictionary"] = dictionary
    # T3: attach the trust layer — a DuckDB-vs-pandas dual-path reconciliation of each
    # feature's match count (blocking red flag on divergence) plus a minimal provenance
    # tuple per number. Best-effort: a trust-layer failure must never break the proposal
    # itself, so any error degrades to "no trust block" rather than failing the gate.
    try:
        _attach_join_trust_layer(runtime, anchor_id, plan, payload, inputs=inputs, seed=seed)
    except Exception:  # noqa: BLE001 - trust layer is additive, never fatal to the join
        pass
    return payload


_DATA_TRANSFORM_PRODUCER_VERSION = "marvis.data-transform/2"
_DATASET_EXPORT_PRODUCER_VERSION = "marvis.dataset-export/1"
_DATASET_EXPORT_ARTIFACT_SCHEMA_VERSION = "dataset-export-artifact.v1"
_DATASET_EXPORT_ARTIFACT_KIND = "dataset_export"
_DATASET_EXPORT_ORIGIN_TOOL = "data_ops.export_dataset"
_DATASET_EXPORT_TEXT_ROLES = frozenset({"phone", "idcard", "id", "name"})


class _DatasetExportConcurrentReplay(RuntimeError):
    """A concurrent transaction committed this exact export input first."""


def tool_transform_dataset(inputs: dict, ctx) -> dict:
    """Apply a canonical transform and atomically activate its derived dataset.

    The LLM or Workflow may author only the closed operation AST accepted by
    :func:`transform_parquet`.  Physical bytes, dataset registration, semantic
    migration, workspace activation, task artifact, immutable run, lineage and
    audit either commit together or roll back together.
    """

    runtime = _runtime(ctx)
    task_id = str(ctx.task_id)
    dataset_id = str(inputs["dataset_id"])
    dataset = runtime.registry.get(dataset_id)
    if str(dataset.task_id) != task_id:
        raise PermissionError(
            f"dataset {dataset_id} belongs to task {dataset.task_id}, not {task_id}"
        )
    expected_hash = str(inputs["expected_content_hash"])
    registered_hash = getattr(dataset, "content_hash", None)
    if not isinstance(registered_hash, str) or not hmac.compare_digest(
        registered_hash,
        expected_hash,
    ):
        raise ValueError("dataset content hash does not match expected content hash")

    workspace_repo = DataWorkspaceRepository(runtime.settings.db_path)
    snapshot = workspace_repo.get_or_default(task_id)
    if snapshot.active_dataset_id != dataset_id:
        raise ValueError("dataset is not the active data-workspace dataset")
    if not isinstance(snapshot.active_dataset_content_hash, str) or not hmac.compare_digest(
        snapshot.active_dataset_content_hash,
        expected_hash,
    ):
        raise ValueError(
            "active data-workspace content hash does not match expected content hash"
        )
    workspace_revision = int(inputs["workspace_revision"])
    if snapshot.revision != workspace_revision:
        raise ValueError(
            "data workspace revision mismatch: "
            f"expected {workspace_revision}, found {snapshot.revision}"
        )
    analysis_generation = int(inputs["analysis_generation"])
    if snapshot.analysis_generation != analysis_generation:
        raise ValueError(
            "data workspace analysis generation mismatch: "
            f"expected {analysis_generation}, found {snapshot.analysis_generation}"
        )
    expected_semantic_hash = str(inputs["semantic_mapping_hash"])
    actual_semantic_hash = data_semantic_mapping_hash(snapshot.semantic_mapping)
    if not hmac.compare_digest(actual_semantic_hash, expected_semantic_hash):
        raise ValueError("data workspace semantic mapping hash mismatch")
    protected_drop = inputs.get("confirm_protected_drop", False)
    if not isinstance(protected_drop, bool):
        raise ValueError("confirm_protected_drop must be boolean")
    operations = inputs.get("operations")
    if not isinstance(operations, list):
        raise ValueError("operations must be an ordered array")

    source_path = runtime.registry.resolve_verified_path(dataset.id)
    source_columns = tuple(runtime.backend.column_names(source_path))
    output_dir = _safe_dataset_directory(
        runtime.datasets_root,
        task_id,
        "transforms",
    )
    with tempfile.TemporaryDirectory(
        prefix=".transform_compute_",
        dir=output_dir,
    ) as scratch_name:
        computed_path = Path(scratch_name) / "result.parquet"
        core_result = transform_parquet(
            source_path,
            computed_path,
            temp_directory=runtime.datasets_root.parent / ".duckdb_tmp",
            operations=operations,
        )
        canonical_operations = core_result.get("operations")
        if not isinstance(canonical_operations, list):
            raise ValueError("transform kernel returned invalid canonical operations")
        output_columns_payload = core_result.get("output", {}).get("columns")
        if not isinstance(output_columns_payload, list) or not all(
            isinstance(item, dict) and isinstance(item.get("name"), str)
            for item in output_columns_payload
        ):
            raise ValueError("transform kernel returned invalid output schema")
        result_columns = tuple(str(item["name"]) for item in output_columns_payload)
        effective_semantics = effective_transform_semantic_mapping(
            dataset,
            snapshot.semantic_mapping,
            source_columns=source_columns,
        )
        effective_semantic_hash = data_semantic_mapping_hash(effective_semantics)
        semantic_migration = migrate_transform_semantics(
            effective_semantics,
            selected_field=snapshot.selected_field,
            operations=canonical_operations,
            source_columns=source_columns,
            result_columns=result_columns,
            confirm_protected_drop=protected_drop,
        )
        identity = DataTransformIdentity(
            task_id=task_id,
            source_dataset_id=dataset.id,
            source_content_hash=expected_hash,
            workspace_revision=snapshot.revision,
            analysis_generation=snapshot.analysis_generation,
            semantic_mapping_hash=effective_semantic_hash,
            operations=tuple(canonical_operations),
            producer_version=_DATA_TRANSFORM_PRODUCER_VERSION,
        )
        # Protected-field validation must happen before an identical-run cache
        # lookup.  Otherwise a concurrent confirmed call could make an
        # unconfirmed call appear successful by handing it the cached record.
        existing = runtime.transforms.find_by_input_hash(task_id, identity.input_hash)
        if existing is not None:
            _verify_cached_transform_record(runtime, existing)
            return _transform_tool_payload(existing, cached=True)

        uow = ArtifactUnitOfWork()
        attempt_token = uuid.uuid4().hex
        parquet_artifact = uow.stage_file(
            output_dir,
            f"{identity.run_id}.{attempt_token}.parquet",
        )
        # stage_file reserves its path with a placeholder.  The transform core
        # intentionally refuses pre-existing outputs, so move the already
        # computed unique result into that reserved stage path.
        parquet_artifact.path.unlink(missing_ok=True)
        shutil.move(computed_path, parquet_artifact.path)
        evidence_dir = _safe_task_artifact_directory(
            runtime.settings,
            task_id,
            "data_transforms",
        )
        evidence_artifact = uow.stage_file(
            evidence_dir,
            f"{identity.run_id}.{attempt_token}.evidence.json",
        )

        try:
            # Close the compute/commit TOCTOU window before promoting anything.
            verified_source = runtime.registry.resolve_verified_path(dataset.id)
            if verified_source != source_path:
                raise ValueError("source dataset path changed during transform")

            def _commit(conn):
                conn.execute("BEGIN IMMEDIATE")
                if not hmac.compare_digest(sha256_file(source_path), expected_hash):
                    raise ValueError("source dataset changed during transform")
                result_dataset = runtime.registry.register_existing_on_connection(
                    conn,
                    parquet_artifact.final_path,
                    task_id=task_id,
                    role="derived",
                    target_col_override=semantic_migration.semantic_mapping.target_col,
                    seed=_seed(ctx),
                )
                kernel_hash = str(core_result["output"]["content_hash"])
                if not isinstance(result_dataset.content_hash, str) or not hmac.compare_digest(
                    result_dataset.content_hash,
                    kernel_hash,
                ):
                    raise ValueError("registered derived dataset hash mismatch")
                activated = workspace_repo.activate_derived_on_connection(
                    conn,
                    task_id,
                    expected_revision=snapshot.revision,
                    source_dataset_id=dataset.id,
                    source_dataset_content_hash=expected_hash,
                    result_dataset_id=result_dataset.id,
                    result_dataset_content_hash=result_dataset.content_hash,
                    page="history",
                    selected_field=semantic_migration.selected_field,
                    semantic_mapping=semantic_migration.semantic_mapping,
                    audit={
                        "actor": "agent:data-transform",
                        "detail": {"transform_run_id": identity.run_id},
                    },
                )
                evidence = _transform_evidence_payload(
                    identity=identity,
                    core_result=core_result,
                    source_dataset=dataset,
                    result_dataset=result_dataset,
                    source_workspace=snapshot,
                    result_workspace=activated,
                    semantic_migration=semantic_migration,
                )
                evidence_json = json.dumps(
                    evidence,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                evidence_artifact.final_path.write_text(evidence_json, encoding="utf-8")
                evidence_hash = sha256_file(evidence_artifact.final_path)
                artifact_record = runtime.task_artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=DATA_TRANSFORM_ARTIFACT_KIND,
                    path=str(evidence_artifact.final_path),
                    content_hash=evidence_hash,
                    origin_tool=DATA_TRANSFORM_ORIGIN_TOOL,
                    provenance=data_transform_artifact_provenance(
                        identity,
                        result_dataset_id=result_dataset.id,
                        result_content_hash=result_dataset.content_hash,
                    ),
                )
                record = runtime.transforms.record_succeeded_on_connection(
                    conn,
                    identity,
                    result_dataset_id=result_dataset.id,
                    result_content_hash=result_dataset.content_hash,
                    result_artifact_id=artifact_record["id"],
                    result_payload=evidence,
                    result_workspace_revision=activated.revision,
                    result_analysis_generation=activated.analysis_generation,
                )
                runtime.repo.write_audit_on_connection(
                    conn,
                    kind="data.transform.completed",
                    target_ref=identity.run_id,
                    actor="agent:data-transform",
                    inputs_hash=identity.input_hash,
                    outcome="succeeded",
                    detail={
                        "task_id": task_id,
                        "source_dataset_id": dataset.id,
                        "result_dataset_id": result_dataset.id,
                        "result_content_hash": result_dataset.content_hash,
                        "operations_hash": identity.operations_hash,
                        "result_artifact_id": artifact_record["id"],
                        "result_workspace_revision": activated.revision,
                        "result_analysis_generation": activated.analysis_generation,
                    },
                )
                return record

            record = uow.finalize_with_connection(runtime.repo.transaction, _commit)
        except Exception:
            uow.rollback()
            raise
    return _transform_tool_payload(record, cached=False)


def _transform_evidence_payload(
    *,
    identity,
    core_result: dict,
    source_dataset,
    result_dataset,
    source_workspace,
    result_workspace,
    semantic_migration,
) -> dict:
    safe_core = deepcopy(core_result)
    output = safe_core.get("output")
    if isinstance(output, dict):
        output.pop("path", None)
    return {
        "schema_version": "data-transform-evidence.v1",
        "run_id": identity.run_id,
        "producer_version": identity.producer_version,
        "input_hash": identity.input_hash,
        "operations_hash": identity.operations_hash,
        "source": {
            "dataset_id": source_dataset.id,
            "content_hash": source_dataset.content_hash,
            "row_count": source_dataset.row_count,
        },
        "result": {
            "dataset_id": result_dataset.id,
            "content_hash": result_dataset.content_hash,
            "row_count": result_dataset.row_count,
        },
        "transform": safe_core,
        "semantic_migration": {
            "before_hash": identity.semantic_mapping_hash,
            "after_hash": data_semantic_mapping_hash(
                semantic_migration.semantic_mapping
            ),
            "renamed_fields": dict(semantic_migration.renamed_fields),
            "dropped_fields": list(semantic_migration.dropped_fields),
            "dropped_protected_fields": list(
                semantic_migration.dropped_protected_fields
            ),
        },
        "workspace": {
            "source_revision": source_workspace.revision,
            "result_revision": result_workspace.revision,
            "source_analysis_generation": source_workspace.analysis_generation,
            "result_analysis_generation": result_workspace.analysis_generation,
        },
        "lineage": {
            "parent_dataset_id": source_dataset.id,
            "child_dataset_id": result_dataset.id,
            "relation_kind": "transform",
            "edge_order": 0,
        },
    }


def _transform_tool_payload(record, *, cached: bool) -> dict:
    evidence = record.result_payload
    transform = evidence.get("transform") or {}
    summary = transform.get("summary") or {}
    semantic = evidence.get("semantic_migration") or {}
    workspace = evidence.get("workspace") or {}
    return {
        "schema_version": "data-transform-tool-result.v1",
        "run_id": record.id,
        "source_dataset_id": record.source_dataset_id,
        "result_dataset_id": record.result_dataset_id,
        "result_content_hash": record.result_content_hash,
        "row_count_before": int(summary.get("row_count_before") or 0),
        "row_count_after": int(summary.get("row_count_after") or 0),
        "column_count_before": int(summary.get("column_count_before") or 0),
        "column_count_after": int(summary.get("column_count_after") or 0),
        "operations": list(record.operations),
        "steps": list(transform.get("steps") or []),
        "semantic_migration": dict(semantic),
        "workspace": dict(workspace),
        "lineage": dict(evidence.get("lineage") or {}),
        "evidence_artifact_id": record.result_artifact_id,
        "evidence_download_url": (
            f"/api/tasks/{record.task_id}/task-artifacts/"
            f"{record.result_artifact_id}/download"
        ),
        "cached": cached,
    }


def _verify_cached_transform_record(runtime, record) -> None:
    """Fail closed before presenting an immutable transform as verified cache."""

    result_dataset = runtime.registry.get(record.result_dataset_id)
    if str(result_dataset.task_id) != record.task_id:
        raise ValueError("cached transform result dataset ownership mismatch")
    result_hash = getattr(result_dataset, "content_hash", None)
    if not isinstance(result_hash, str) or not hmac.compare_digest(
        result_hash,
        record.result_content_hash,
    ):
        raise ValueError("cached transform result dataset hash mismatch")
    runtime.registry.resolve_verified_path(record.result_dataset_id)

    artifact = runtime.task_artifacts.get_for_task(
        record.task_id,
        record.result_artifact_id,
    )
    if artifact is None:
        raise ValueError("cached transform evidence artifact is missing")
    expected_provenance = data_transform_artifact_provenance(
        DataTransformIdentity(
            task_id=record.task_id,
            source_dataset_id=record.source_dataset_id,
            source_content_hash=record.source_content_hash,
            workspace_revision=record.workspace_revision,
            analysis_generation=record.analysis_generation,
            semantic_mapping_hash=record.semantic_mapping_hash,
            operations=record.operations,
            producer_version=record.producer_version,
        ),
        result_dataset_id=record.result_dataset_id,
        result_content_hash=record.result_content_hash,
    )
    if (
        artifact.get("kind") != DATA_TRANSFORM_ARTIFACT_KIND
        or artifact.get("origin_tool") != DATA_TRANSFORM_ORIGIN_TOOL
        or artifact.get("content_hash") != record.result_hash
        or artifact.get("provenance") != expected_provenance
    ):
        raise ValueError("cached transform evidence registry mismatch")
    artifact_path = Path(str(artifact["path"]))
    try:
        artifact_path.resolve(strict=True).relative_to(
            (runtime.settings.tasks_dir / record.task_id).resolve(strict=True)
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("cached transform evidence path is unavailable or unsafe") from exc
    if not artifact_path.is_file() or not hmac.compare_digest(
        sha256_file(artifact_path),
        record.result_hash,
    ):
        raise ValueError("cached transform evidence integrity check failed")

    active = DataWorkspaceRepository(runtime.settings.db_path).get_or_default(
        record.task_id
    )
    if (
        active.active_dataset_id != record.result_dataset_id
        or active.active_dataset_content_hash != record.result_content_hash
        or active.revision != record.result_workspace_revision
        or active.analysis_generation != record.result_analysis_generation
    ):
        raise ValueError("cached transform workspace evidence is no longer active")


def _safe_task_artifact_directory(settings, task_id: str, child: str) -> Path:
    """Create a task-local artifact directory without accepting symlink hops."""

    if Path(task_id).name != task_id or task_id in {".", ".."}:
        raise ValueError("task id is unsafe for an artifact directory")
    if Path(child).name != child or child in {".", ".."}:
        raise ValueError("artifact directory name is unsafe")
    declared_root = Path(settings.tasks_dir).absolute()
    if declared_root.is_symlink():
        raise ValueError("task artifact root must not be a symlink")
    declared_root.mkdir(parents=True, exist_ok=True)
    resolved_root = declared_root.resolve(strict=True)
    task_root = declared_root / task_id
    if task_root.is_symlink():
        raise ValueError("task artifact directory must not be a symlink")
    task_root.mkdir(exist_ok=True)
    resolved_task = task_root.resolve(strict=True)
    if resolved_task.parent != resolved_root:
        raise ValueError("task artifact directory escaped the task root")
    artifact_dir = task_root / child
    if artifact_dir.is_symlink():
        raise ValueError("task artifact subdirectory must not be a symlink")
    artifact_dir.mkdir(exist_ok=True)
    if artifact_dir.resolve(strict=True).parent != resolved_task:
        raise ValueError("task artifact subdirectory escaped the task directory")
    return artifact_dir


def _safe_dataset_directory(datasets_root: Path, task_id: str, child: str) -> Path:
    """Create a task-local dataset directory without accepting symlink hops."""

    if Path(task_id).name != task_id or task_id in {".", ".."}:
        raise ValueError("task id is unsafe for a dataset directory")
    if Path(child).name != child or child in {".", ".."}:
        raise ValueError("dataset directory name is unsafe")
    declared_root = Path(datasets_root).absolute()
    if declared_root.is_symlink():
        raise ValueError("dataset root must not be a symlink")
    declared_root.mkdir(parents=True, exist_ok=True)
    resolved_root = declared_root.resolve(strict=True)
    task_root = declared_root / task_id
    if task_root.is_symlink():
        raise ValueError("task dataset directory must not be a symlink")
    task_root.mkdir(exist_ok=True)
    resolved_task = task_root.resolve(strict=True)
    if resolved_task.parent != resolved_root:
        raise ValueError("task dataset directory escaped the dataset root")
    dataset_dir = task_root / child
    if dataset_dir.is_symlink():
        raise ValueError("task dataset subdirectory must not be a symlink")
    dataset_dir.mkdir(exist_ok=True)
    if dataset_dir.resolve(strict=True).parent != resolved_task:
        raise ValueError("task dataset subdirectory escaped the task directory")
    return dataset_dir


def tool_export_dataset(inputs: dict, ctx) -> dict:
    """Export the exact active dataset as a safe, task-owned CSV or XLSX."""

    runtime = _runtime(ctx)
    task_id = str(ctx.task_id)
    dataset_id = str(inputs["dataset_id"])
    dataset = runtime.registry.get(dataset_id)
    if str(dataset.task_id) != task_id:
        raise PermissionError(
            f"dataset {dataset_id} belongs to task {dataset.task_id}, not {task_id}"
        )
    expected_hash = str(inputs["expected_content_hash"])
    registered_hash = getattr(dataset, "content_hash", None)
    if not isinstance(registered_hash, str) or not hmac.compare_digest(
        registered_hash,
        expected_hash,
    ):
        raise ValueError("dataset content hash does not match expected content hash")

    workspace_repo = DataWorkspaceRepository(runtime.settings.db_path)
    snapshot = workspace_repo.get_or_default(task_id)
    if snapshot.active_dataset_id != dataset_id:
        raise ValueError("dataset is not the active data-workspace dataset")
    if not isinstance(snapshot.active_dataset_content_hash, str) or not hmac.compare_digest(
        snapshot.active_dataset_content_hash,
        expected_hash,
    ):
        raise ValueError(
            "active data-workspace content hash does not match expected content hash"
        )
    workspace_revision = int(inputs["workspace_revision"])
    if snapshot.revision != workspace_revision:
        raise ValueError(
            "data workspace revision mismatch: "
            f"expected {workspace_revision}, found {snapshot.revision}"
        )
    analysis_generation = int(inputs["analysis_generation"])
    if snapshot.analysis_generation != analysis_generation:
        raise ValueError(
            "data workspace analysis generation mismatch: "
            f"expected {analysis_generation}, found {snapshot.analysis_generation}"
        )
    expected_semantic_hash = str(inputs["semantic_mapping_hash"])
    actual_semantic_hash = data_semantic_mapping_hash(snapshot.semantic_mapping)
    if not hmac.compare_digest(actual_semantic_hash, expected_semantic_hash):
        raise ValueError("data workspace semantic mapping hash mismatch")

    export_format = str(inputs["format"])
    if export_format not in {"csv", "xlsx"}:
        raise ValueError("format must be csv or xlsx")
    source_path = runtime.registry.resolve_verified_path(dataset.id)
    source_columns = tuple(runtime.backend.column_names(source_path))
    text_columns = _dataset_export_text_columns(
        dataset,
        snapshot.semantic_mapping,
        source_columns=source_columns,
        requested=inputs.get("text_columns"),
    )
    identity_payload = {
        "schema_version": "dataset-export-input.v1",
        "task_id": task_id,
        "dataset_id": dataset.id,
        "dataset_content_hash": expected_hash,
        "workspace_revision": snapshot.revision,
        "analysis_generation": snapshot.analysis_generation,
        "semantic_mapping_hash": actual_semantic_hash,
        "format": export_format,
        "text_columns": list(text_columns),
        "producer_version": _DATASET_EXPORT_PRODUCER_VERSION,
    }
    input_json = json.dumps(
        identity_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    input_hash = hashlib.sha256(input_json.encode("utf-8")).hexdigest()
    existing = _find_existing_dataset_export(
        runtime,
        task_id=task_id,
        input_hash=input_hash,
    )
    if existing is not None:
        with runtime.repo.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            _require_dataset_export_binding_on_connection(
                conn,
                task_id=task_id,
                dataset_id=dataset.id,
                expected_content_hash=expected_hash,
                workspace_revision=snapshot.revision,
                analysis_generation=snapshot.analysis_generation,
                semantic_mapping_hash=actual_semantic_hash,
            )
            if not hmac.compare_digest(sha256_file(source_path), expected_hash):
                raise ValueError("source dataset changed before cached export replay")
        return _dataset_export_tool_payload(existing, cached=True)

    export_dir = _safe_task_artifact_directory(
        runtime.settings,
        task_id,
        "data_exports",
    )
    with tempfile.TemporaryDirectory(
        prefix=".dataset_export_compute_",
        dir=export_dir,
    ) as scratch_name:
        computed_path = Path(scratch_name) / f"export.{export_format}"
        core_result = export_dataset(
            source_path,
            computed_path,
            format=export_format,
            temp_directory=runtime.datasets_root.parent / ".duckdb_tmp",
            text_columns=text_columns,
        )
        safe_core = deepcopy(core_result)
        core_output = safe_core.get("output")
        if not isinstance(core_output, dict):
            raise ValueError("dataset export kernel returned invalid output evidence")
        core_output.pop("path", None)
        output_hash = str(core_output.get("content_hash") or "")
        if len(output_hash) != 64:
            raise ValueError("dataset export kernel returned invalid content hash")
        if (
            core_output.get("row_count") != int(dataset.row_count)
            or core_output.get("column_count") != len(source_columns)
        ):
            raise ValueError("dataset export kernel row or column count mismatch")
        artifact_provenance = {
            "schema_version": _DATASET_EXPORT_ARTIFACT_SCHEMA_VERSION,
            "input_schema_version": identity_payload["schema_version"],
            "producer_version": _DATASET_EXPORT_PRODUCER_VERSION,
            "input_hash": input_hash,
            **identity_payload,
            "export": safe_core,
        }
        artifact_provenance["schema_version"] = (
            _DATASET_EXPORT_ARTIFACT_SCHEMA_VERSION
        )
        filename = (
            f"dataset_export_{input_hash[:24]}.{uuid.uuid4().hex}.{export_format}"
        )
        uow = ArtifactUnitOfWork()
        staged = uow.stage_file(export_dir, filename)
        staged.path.unlink(missing_ok=True)
        shutil.move(computed_path, staged.path)

        try:
            verified_source = runtime.registry.resolve_verified_path(dataset.id)
            if verified_source != source_path:
                raise ValueError("source dataset path changed during export")

            def _commit(conn):
                conn.execute("BEGIN IMMEDIATE")
                if _dataset_export_exists_on_connection(
                    conn,
                    task_id=task_id,
                    input_hash=input_hash,
                ):
                    raise _DatasetExportConcurrentReplay(input_hash)
                _require_dataset_export_binding_on_connection(
                    conn,
                    task_id=task_id,
                    dataset_id=dataset.id,
                    expected_content_hash=expected_hash,
                    workspace_revision=snapshot.revision,
                    analysis_generation=snapshot.analysis_generation,
                    semantic_mapping_hash=actual_semantic_hash,
                )
                if not hmac.compare_digest(sha256_file(source_path), expected_hash):
                    raise ValueError("source dataset changed during export")
                if not hmac.compare_digest(sha256_file(staged.final_path), output_hash):
                    raise ValueError("published dataset export hash mismatch")
                artifact = runtime.task_artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=_DATASET_EXPORT_ARTIFACT_KIND,
                    path=str(staged.final_path),
                    content_hash=output_hash,
                    origin_tool=_DATASET_EXPORT_ORIGIN_TOOL,
                    provenance=artifact_provenance,
                )
                runtime.repo.write_audit_on_connection(
                    conn,
                    kind="data.export.completed",
                    target_ref=artifact["id"],
                    actor="agent:data-export",
                    inputs_hash=input_hash,
                    outcome="succeeded",
                    detail={
                        "task_id": task_id,
                        "dataset_id": dataset.id,
                        "dataset_content_hash": expected_hash,
                        "format": export_format,
                        "artifact_id": artifact["id"],
                        "content_hash": output_hash,
                        "workspace_revision": snapshot.revision,
                        "analysis_generation": snapshot.analysis_generation,
                    },
                )
                return artifact

            artifact = uow.finalize_with_connection(
                runtime.repo.transaction,
                _commit,
            )
        except _DatasetExportConcurrentReplay as exc:
            uow.rollback()
            winner = _find_existing_dataset_export(
                runtime,
                task_id=task_id,
                input_hash=input_hash,
            )
            if winner is None:  # pragma: no cover - defensive transaction invariant
                raise ValueError(
                    "concurrent dataset export winner is missing"
                ) from exc
            return _dataset_export_tool_payload(winner, cached=True)
        except Exception:
            uow.rollback()
            raise
    return _dataset_export_tool_payload(artifact, cached=False)


def _dataset_export_text_columns(
    dataset,
    semantic_mapping,
    *,
    source_columns: tuple[str, ...],
    requested,
) -> tuple[str, ...]:
    roles = {
        str(column.name): str(column.semantic_role)
        for column in dataset.columns
        if str(column.semantic_role) in _DATASET_EXPORT_TEXT_ROLES
    }
    roles.update(
        {
            str(name): str(role)
            for name, role in semantic_mapping.field_roles.items()
            if str(role) in _DATASET_EXPORT_TEXT_ROLES
        }
    )
    selected = set(roles)
    if requested is not None:
        if isinstance(requested, (str, bytes)) or not isinstance(requested, list):
            raise ValueError("text_columns must be an ordered array")
        for raw_name in requested:
            if not isinstance(raw_name, str) or raw_name not in source_columns:
                raise ValueError(f"unknown text column: {raw_name}")
            selected.add(raw_name)
    return tuple(name for name in source_columns if name in selected)


def _find_existing_dataset_export(runtime, *, task_id: str, input_hash: str):
    matches = []
    for artifact in runtime.task_artifacts.list_for_task(task_id):
        provenance = artifact.get("provenance") or {}
        if (
            artifact.get("kind") == _DATASET_EXPORT_ARTIFACT_KIND
            and artifact.get("origin_tool") == _DATASET_EXPORT_ORIGIN_TOOL
            and provenance.get("schema_version")
            == _DATASET_EXPORT_ARTIFACT_SCHEMA_VERSION
            and provenance.get("input_hash") == input_hash
        ):
            matches.append(artifact)
    if len(matches) > 1:
        raise ValueError("dataset export input has multiple immutable artifacts")
    if not matches:
        return None
    artifact = matches[0]
    _validate_dataset_export_artifact(
        artifact,
        expected_input_hash=input_hash,
    )
    provenance = artifact["provenance"]
    dataset = runtime.registry.get(str(provenance["dataset_id"]))
    if (
        str(dataset.task_id) != task_id
        or dataset.content_hash != provenance["dataset_content_hash"]
    ):
        raise ValueError("cached dataset export source identity mismatch")
    runtime.registry.resolve_verified_path(dataset.id)
    path = Path(str(artifact["path"]))
    expected_hash = str(artifact["content_hash"])
    try:
        path.resolve(strict=True).relative_to(
            (runtime.settings.tasks_dir / task_id).resolve(strict=True)
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("cached dataset export path is unavailable or unsafe") from exc
    expected_size = artifact["provenance"]["export"]["output"]["size_bytes"]
    if (
        not path.is_file()
        or path.stat().st_size != expected_size
        or not hmac.compare_digest(sha256_file(path), expected_hash)
    ):
        raise ValueError("cached dataset export integrity check failed")
    return artifact


def _validate_dataset_export_artifact(
    artifact,
    *,
    expected_input_hash: str | None = None,
) -> None:
    provenance = artifact.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("dataset export provenance must be an object")
    if (
        provenance.get("schema_version")
        != _DATASET_EXPORT_ARTIFACT_SCHEMA_VERSION
        or provenance.get("input_schema_version") != "dataset-export-input.v1"
        or provenance.get("producer_version")
        != _DATASET_EXPORT_PRODUCER_VERSION
        or artifact.get("kind") != _DATASET_EXPORT_ARTIFACT_KIND
        or artifact.get("origin_tool") != _DATASET_EXPORT_ORIGIN_TOOL
        or provenance.get("task_id") != artifact.get("task_id")
    ):
        raise ValueError("dataset export artifact identity is invalid")
    input_hash = provenance.get("input_hash")
    if (
        not isinstance(input_hash, str)
        or len(input_hash) != 64
        or any(character not in "0123456789abcdef" for character in input_hash)
        or (
            expected_input_hash is not None
            and not hmac.compare_digest(input_hash, expected_input_hash)
        )
    ):
        raise ValueError("dataset export input hash is invalid")
    dataset_hash = provenance.get("dataset_content_hash")
    if (
        not isinstance(dataset_hash, str)
        or len(dataset_hash) != 64
        or any(character not in "0123456789abcdef" for character in dataset_hash)
    ):
        raise ValueError("dataset export source hash is invalid")
    text_columns = provenance.get("text_columns")
    workspace_revision = provenance.get("workspace_revision")
    analysis_generation = provenance.get("analysis_generation")
    semantic_hash = provenance.get("semantic_mapping_hash")
    if (
        not isinstance(provenance.get("task_id"), str)
        or not provenance["task_id"]
        or not isinstance(provenance.get("dataset_id"), str)
        or not provenance["dataset_id"]
        or isinstance(workspace_revision, bool)
        or not isinstance(workspace_revision, int)
        or workspace_revision < 0
        or isinstance(analysis_generation, bool)
        or not isinstance(analysis_generation, int)
        or analysis_generation < 0
        or not isinstance(semantic_hash, str)
        or len(semantic_hash) != 64
        or any(character not in "0123456789abcdef" for character in semantic_hash)
        or not isinstance(text_columns, list)
        or not all(isinstance(item, str) and item for item in text_columns)
        or len(text_columns) != len(set(text_columns))
        or provenance.get("format") not in {"csv", "xlsx"}
    ):
        raise ValueError("dataset export input evidence is invalid")
    identity_payload = {
        "schema_version": provenance["input_schema_version"],
        "task_id": provenance["task_id"],
        "dataset_id": provenance["dataset_id"],
        "dataset_content_hash": dataset_hash,
        "workspace_revision": workspace_revision,
        "analysis_generation": analysis_generation,
        "semantic_mapping_hash": semantic_hash,
        "format": provenance["format"],
        "text_columns": text_columns,
        "producer_version": provenance["producer_version"],
    }
    canonical_identity = json.dumps(
        identity_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    canonical_input_hash = hashlib.sha256(
        canonical_identity.encode("utf-8")
    ).hexdigest()
    if not hmac.compare_digest(canonical_input_hash, input_hash):
        raise ValueError("dataset export input hash does not match provenance")
    export = provenance.get("export")
    if not isinstance(export, dict) or export.get("schema_version") != (
        "dataset-export-result.v1"
    ):
        raise ValueError("dataset export evidence schema is invalid")
    output = export.get("output")
    if not isinstance(output, dict):
        raise ValueError("dataset export output evidence is invalid")
    artifact_hash = artifact.get("content_hash")
    if (
        output.get("content_hash") != artifact_hash
        or output.get("format") != provenance.get("format")
        or provenance.get("format") not in {"csv", "xlsx"}
        or isinstance(output.get("row_count"), bool)
        or not isinstance(output.get("row_count"), int)
        or output["row_count"] < 0
        or isinstance(output.get("column_count"), bool)
        or not isinstance(output.get("column_count"), int)
        or output["column_count"] < 1
        or isinstance(output.get("size_bytes"), bool)
        or not isinstance(output.get("size_bytes"), int)
        or output["size_bytes"] < 0
    ):
        raise ValueError("dataset export output evidence does not match artifact")
    options = export.get("options")
    if (
        not isinstance(options, dict)
        or options.get("text_columns") != provenance.get("text_columns")
        or not isinstance(export.get("safety"), dict)
    ):
        raise ValueError("dataset export options or safety evidence is invalid")


def _dataset_export_exists_on_connection(
    conn,
    *,
    task_id: str,
    input_hash: str,
) -> bool:
    rows = conn.execute(
        """
        SELECT provenance_json
          FROM task_artifacts
         WHERE task_id = ? AND kind = ? AND origin_tool = ?
        """,
        (task_id, _DATASET_EXPORT_ARTIFACT_KIND, _DATASET_EXPORT_ORIGIN_TOOL),
    ).fetchall()
    matches = 0
    for row in rows:
        try:
            provenance = json.loads(str(row["provenance_json"]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("persisted dataset export provenance is invalid") from exc
        if (
            isinstance(provenance, dict)
            and provenance.get("schema_version")
            == _DATASET_EXPORT_ARTIFACT_SCHEMA_VERSION
            and provenance.get("input_hash") == input_hash
        ):
            matches += 1
    if matches > 1:
        raise ValueError("dataset export input has multiple immutable artifacts")
    return matches == 1


def _require_dataset_export_binding_on_connection(
    conn,
    *,
    task_id: str,
    dataset_id: str,
    expected_content_hash: str,
    workspace_revision: int,
    analysis_generation: int,
    semantic_mapping_hash: str,
) -> None:
    dataset_row = conn.execute(
        "SELECT task_id, content_hash FROM datasets WHERE id = ?",
        (dataset_id,),
    ).fetchone()
    if dataset_row is None or str(dataset_row["task_id"]) != task_id:
        raise ValueError("dataset ownership changed during export")
    registered_hash = dataset_row["content_hash"]
    if not isinstance(registered_hash, str) or not hmac.compare_digest(
        registered_hash,
        expected_content_hash,
    ):
        raise ValueError("dataset content hash changed during export")
    workspace_row = conn.execute(
        """
        SELECT active_dataset_id, active_dataset_content_hash, revision,
               analysis_generation, semantic_mapping_json
          FROM data_workspaces
         WHERE task_id = ?
        """,
        (task_id,),
    ).fetchone()
    if workspace_row is None:
        raise ValueError("data workspace disappeared during export")
    active_hash = workspace_row["active_dataset_content_hash"]
    mapping_json = workspace_row["semantic_mapping_json"]
    if not isinstance(mapping_json, str):
        raise ValueError("data workspace semantic mapping is invalid")
    mapping_payload = json.loads(mapping_json)
    from marvis.data.workspace import data_semantic_mapping_from_dict

    live_semantic_hash = data_semantic_mapping_hash(
        data_semantic_mapping_from_dict(mapping_payload)
    )
    if (
        str(workspace_row["active_dataset_id"]) != dataset_id
        or not isinstance(active_hash, str)
        or not hmac.compare_digest(active_hash, expected_content_hash)
        or int(workspace_row["revision"]) != workspace_revision
        or int(workspace_row["analysis_generation"]) != analysis_generation
        or not hmac.compare_digest(live_semantic_hash, semantic_mapping_hash)
    ):
        raise ValueError("data workspace changed during export")


def _dataset_export_tool_payload(artifact, *, cached: bool) -> dict:
    _validate_dataset_export_artifact(artifact)
    provenance = artifact.get("provenance") or {}
    export = provenance.get("export") or {}
    output = export.get("output") or {}
    return {
        "schema_version": "dataset-export-tool-result.v1",
        "input_hash": provenance.get("input_hash"),
        "dataset_id": provenance.get("dataset_id"),
        "dataset_content_hash": provenance.get("dataset_content_hash"),
        "workspace_revision": provenance.get("workspace_revision"),
        "analysis_generation": provenance.get("analysis_generation"),
        "semantic_mapping_hash": provenance.get("semantic_mapping_hash"),
        "format": output.get("format"),
        "row_count": output.get("row_count"),
        "column_count": output.get("column_count"),
        "size_bytes": output.get("size_bytes"),
        "content_hash": output.get("content_hash"),
        "options": dict(export.get("options") or {}),
        "safety": dict(export.get("safety") or {}),
        "artifact_id": artifact.get("id"),
        "download_url": (
            f"/api/tasks/{quote(str(artifact.get('task_id')), safe='')}"
            f"/task-artifacts/{quote(str(artifact.get('id')), safe='')}/download"
        ),
        "cached": cached,
    }


def _attach_join_trust_layer(
    runtime: "_Runtime",
    anchor_id: str,
    plan,
    payload: dict,
    *,
    inputs: dict,
    seed: int,
) -> None:
    """T3-1/T3-2: reconcile each proposed join's match count two independent ways
    (DuckDB SQL vs a forced pandas recount over the same sampled keys) and stamp a
    provenance tuple, attaching both to the per-join payload and a plan-level summary.

    The match count is THE highest-risk headline join number: it decides whether a
    feature is worth joining. Reconciling it against a genuinely separate code path
    turns the human's confirmation from trusting one number into seeing whether two
    paths agree. A divergence beyond the exact-path tolerance (counts, so 1e-9) is a
    BLOCKING typed red flag carried in the payload, not a soft warning.
    """
    anchor = runtime.registry.get(anchor_id)
    anchor_path = runtime.registry.resolve_path(anchor_id)
    payload_by_feature = {
        str(join.get("feature_id")): join for join in payload.get("joins", [])
    }
    plan_results = []
    for spec in plan.joins:
        feature_id = spec.feature_dataset_id
        join_payload = payload_by_feature.get(feature_id)
        if join_payload is None or not spec.key_pairs:
            continue
        feature = runtime.registry.get(feature_id)
        feature_path = runtime.registry.resolve_path(feature_id)
        anchor_keys = [pair.anchor_col for pair in spec.key_pairs]
        feature_keys = [pair.feature_col for pair in spec.key_pairs]
        methods = [pair.match_method for pair in spec.key_pairs]
        fingerprints = _key_fps(anchor, feature, spec.key_pairs)
        label = f"{_friendly_name(runtime.registry, feature_id)} 匹配行数"
        primary_matched, primary_sampled = runtime.backend.match_rate_for_method(
            anchor_path, anchor_keys, feature_path, feature_keys,
            method=methods, key_fingerprints=fingerprints,
            sample_n=SMALL_SAMPLE_N, seed=seed,
        )
        # T3-2: the reconcile is only meaningful when the primary actually ran the DuckDB SQL
        # kernel. When it doesn't (unsupported hash / non-CSV-parquet feature), the "pandas"
        # secondary would be the SAME function the primary fell back to -> they agree by
        # construction. Present that honestly as "not independently verified", NEVER as a
        # passing two-path reconciliation (which would be false assurance).
        independent = runtime.backend.reconcile_paths_are_independent(
            anchor_path, feature_path, methods,
        )
        if independent:
            # Primary = DuckDB SQL path; secondary = pure-pandas set membership over the SAME
            # anchor sample the primary scored (identical subset -> a mismatch is a real
            # implementation divergence, not a sampling artifact).
            secondary_matched, _secondary_sampled = runtime.backend.match_rate_reconcile_secondary(
                anchor_path, anchor_keys, feature_path, feature_keys,
                method=methods, key_fingerprints=fingerprints,
                sample_n=SMALL_SAMPLE_N, seed=seed,
            )
            result = reconcile(
                primary_matched,
                secondary_matched,
                rel_tol=EXACT_REL_TOL,
                abs_tol=EXACT_ABS_TOL,
                label=label,
                primary_path="duckdb_sql",
                secondary_path="pandas",
            )
            plan_results.append(result)
            reconcile_payload = {**result.to_dict(), "sampled": int(primary_sampled)}
        else:
            # No independent second path available. Do NOT append a ReconcileResult (an
            # unverified number is not a divergence, so it must not colour the plan's blocking
            # verdict), and stamp an honest trust status the renderer surfaces as 未独立复核.
            reconcile_payload = {
                "label": label,
                "primary": float(primary_matched),
                "secondary": None,
                "primary_path": "duckdb_sql_or_pandas_fallback",
                "secondary_path": None,
                "consistent": None,
                "trust": "not_independently_verified",
                "sampled": int(primary_sampled),
            }
        provenance = NumberProvenance.build(
            content_hashes=[
                getattr(anchor, "content_hash", None),
                getattr(feature, "content_hash", None),
            ],
            params={
                "anchor_id": anchor_id,
                "feature_id": feature_id,
                "anchor_keys": anchor_keys,
                "feature_keys": feature_keys,
                "methods": methods,
                "sample_n": SMALL_SAMPLE_N,
                "seed": seed,
            },
            seed=seed,
        )
        join_payload["reconcile"] = reconcile_payload
        join_payload["provenance"] = provenance.to_dict()
    report = ReconcileReport(results=tuple(plan_results))
    payload["reconcile_summary"] = report.to_dict()


def _join_dictionary(runtime: "_Runtime", ctx, payload: dict) -> dict:
    dictionary_id = first_data_dictionary_id(runtime.registry.list_for_task(ctx.task_id))
    if not dictionary_id:
        return {}
    names = load_business_names(runtime.backend, runtime.registry, dictionary_id)
    if not names:
        return {}
    columns: set[str] = set()
    for join in payload.get("joins", []):
        for pair in join.get("key_pairs") or []:
            if pair.get("anchor_col"):
                columns.add(str(pair["anchor_col"]))
            if pair.get("feature_col"):
                columns.add(str(pair["feature_col"]))
    return {column: names[column] for column in columns if column in names}


def tool_confirm_join(inputs: dict, ctx) -> dict:
    """Confirm a proposed join plan's feature specs so execute_join is allowed.

    Confirmation is per-feature (the engine's forced-confirmation invariant): a
    feature whose join key is not unique requires a dedup strategy ("first"/"last")
    or the engine refuses. ``dedup_strategies`` maps feature_dataset_id -> strategy.

    A feature needing a strategy that wasn't supplied is reported in ``needs_dedup``
    (status="needs_dedup") rather than HARD-FAILING the plan: the conversational flow
    then reaches the C2 gate (which surfaces the conflicts), where the user supplies the
    strategy and re-confirms. The mutating execute_join still refuses to run until every
    spec is confirmed, so nothing is silently joined.
    """
    runtime = _runtime(ctx)
    join_plan_id = str(inputs["join_plan_id"])
    strategies = inputs.get("dedup_strategies") or {}
    # T1-B8: per-feature acknowledgement that a red (text<->float) key-dtype mismatch is the
    # same identifier. Accept a set/list of feature ids, or a truthy scalar to ack all.
    ack_input = inputs.get("ack_dtype_mismatch")
    if isinstance(ack_input, (list, tuple, set)):
        ack_ids = {str(fid) for fid in ack_input}
        ack_all = False
    else:
        ack_ids = set()
        ack_all = bool(ack_input)
    plan = runtime.repo.load_join_plan(join_plan_id)
    confirmed: list[str] = []
    needs_dedup: list[str] = []
    needs_dtype_ack: list[str] = []
    for spec in plan.joins:
        feature_id = spec.feature_dataset_id
        strategy = strategies.get(feature_id)
        try:
            runtime.join_engine.confirm_join_spec(
                join_plan_id, feature_id, dedup_strategy=strategy,
                ack_dtype_mismatch=ack_all or feature_id in ack_ids,
            )
            confirmed.append(feature_id)
        except KeyDtypeMismatchError:
            needs_dtype_ack.append(feature_id)
        except DedupRequiredError:
            needs_dedup.append(feature_id)
    status = "confirmed"
    if needs_dtype_ack:
        status = "needs_dtype_ack"
    elif needs_dedup:
        status = "needs_dedup"
    return {
        "join_plan_id": join_plan_id,
        "confirmed": confirmed,
        "needs_dedup": needs_dedup,
        # friendly file names for the gate message (raw ids stay in needs_dedup for the picker)
        "needs_dedup_labels": {fid: _friendly_name(runtime.registry, fid) for fid in needs_dedup},
        "needs_dtype_ack": needs_dtype_ack,
        "needs_dtype_ack_labels": {
            fid: _friendly_name(runtime.registry, fid) for fid in needs_dtype_ack
        },
        "status": status,
    }


def tool_execute_join(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    plan = runtime.repo.load_join_plan(str(inputs["join_plan_id"]))
    anchor = runtime.registry.get(plan.anchor_dataset_id)
    result = runtime.join_engine.execute_join_plan(
        plan.id,
        out_dir=runtime.datasets_root / ctx.task_id / "joins",
    )
    # §8 stage-completion summary from real per-table diagnostics (no longer hard-coded).
    per_table = []
    warnings = []
    for spec in plan.joins:
        diag = spec.diagnostics
        per_table.append({
            "feature_id": spec.feature_dataset_id,
            "match_rate": round(float(diag.match_rate), 4),
            "new_columns": int(diag.new_columns),
            "new_columns_null_rate": round(float(diag.new_columns_null_rate), 4),
            "dedup_strategy": spec.dedup_strategy or "无",
        })
        if diag.shrink_detected:
            warnings.append(
                f"{spec.feature_dataset_id}:命中率偏低({diag.match_rate:.2f}),新列缺失较多"
            )
        # conflict_report is a ConflictReport in-memory but an asdict-flattened dict after a
        # DB round-trip (load_join_plan) — handle both so the warning never crashes here.
        report = getattr(diag, "conflict_report", None)
        if isinstance(report, dict):
            conflict_keys = int(report.get("n_conflict_keys") or 0)
        elif report is not None:
            conflict_keys = int(getattr(report, "n_conflict_keys", 0) or 0)
        else:
            conflict_keys = 0
        if conflict_keys and spec.dedup_strategy:
            warnings.append(
                f"{spec.feature_dataset_id}:{conflict_keys} 个同键冲突已按 "
                f"'{spec.dedup_strategy}' 解决"
            )
    return {
        "result_dataset_id": result.id,
        "anchor_rows": anchor.row_count,
        "joined_rows": result.row_count,
        "fan_out": result.row_count > anchor.row_count,
        "warnings": warnings,
        "per_table": per_table,
    }


def tool_clean_format(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    path = runtime.registry.resolve_path(dataset.id)
    frame = runtime.backend.read_frame(path)
    changed_columns = []
    for operation in inputs.get("ops") or []:
        column = str(operation["col"])
        op = str(operation["op"])
        if column not in frame.columns:
            raise KeyError(f"unknown column: {column}")
        frame[column] = _apply_clean_op(frame[column], op)
        changed_columns.append(column)
    result = _register_derived_frame(
        runtime,
        ctx,
        frame,
        subdir="clean",
        filename=f"{dataset.id}_clean.parquet",
        role=dataset.role,
        anchor_target=dataset.id,
    )
    return {"dataset_id": result.id, "changed_columns": changed_columns}


def _register_derived_frame(
    runtime,
    ctx,
    frame: pd.DataFrame,
    *,
    subdir: str,
    filename: str,
    role: str,
    anchor_target: str,
):
    uow = ArtifactUnitOfWork()
    artifact = uow.stage_file(runtime.datasets_root / ctx.task_id / subdir, filename)
    try:
        frame.to_parquet(artifact.path, index=False)
        register_on_connection = getattr(runtime.registry, "register_existing_on_connection", None)
        if callable(register_on_connection):
            return uow.finalize_with_connection(
                runtime.repo.transaction,
                lambda conn: register_on_connection(
                    conn,
                    artifact.final_path,
                    task_id=ctx.task_id,
                    role=role,
                    anchor_target=anchor_target,
                    seed=_seed(ctx),
                ),
            )
        return uow.finalize(
            lambda: runtime.registry.register_existing(
                artifact.final_path,
                task_id=ctx.task_id,
                role=role,
                anchor_target=anchor_target,
                seed=_seed(ctx),
            )
        )
    except Exception:
        uow.rollback()
        raise


def tool_dedup_rows(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    keys = [str(item) for item in inputs.get("keys") or []]
    strategy = inputs.get("strategy")
    path = runtime.registry.resolve_path(dataset.id)
    frame = runtime.backend.read_frame(path)
    missing = sorted(set(keys) - set(frame.columns))
    if missing:
        raise KeyError(f"unknown keys: {', '.join(missing)}")
    before = len(frame)
    # Level-1 safe dedup (always) + level-2 conflict detection (never auto-dropped).
    deduped, report = two_level_dedup(frame, keys)
    # A same-key value-conflict is only resolved on an EXPLICIT, deterministic strategy
    # (spec §6: 告警不静默删). With no strategy, conflicts are surfaced for review.
    needs_conflict_review = report.has_conflicts and not strategy
    if strategy and report.has_conflicts and keys:
        keep = "first" if str(strategy) == "first" else "last"
        deduped = deduped.drop_duplicates(subset=keys, keep=keep, ignore_index=True)
    result = _register_derived_frame(
        runtime,
        ctx,
        deduped,
        subdir="dedup",
        filename=f"{dataset.id}_dedup.parquet",
        role=dataset.role,
        anchor_target=dataset.id,
    )
    return {
        "dataset_id": result.id,
        "removed_rows": before - len(deduped),
        "safe_dropped": report.safe_dropped,
        "needs_conflict_review": needs_conflict_review,
        "conflict_report": _conflict_report_json(report),
    }


def _conflict_report_json(report) -> dict:
    return {
        "key_columns": list(report.key_columns),
        "conflict_columns": list(report.conflict_columns),
        "n_conflict_keys": report.n_conflict_keys,
        "n_conflict_rows": report.n_conflict_rows,
        "safe_dropped": report.safe_dropped,
        "sample_keys": [list(key) for key in report.sample_keys],
    }


# ---------------------------------------------------------------------------
# S6 ad-hoc slice/aggregate: a deterministic, whitelisted group-by aggregate over
# a registered dataset. Every group_by/metric/filter column is validated against
# the dataset's column profile (``sql_identifier`` raises on any unknown name), the
# op->SQL mapping is a fixed dictionary, and a single parameterized DuckDB SQL is
# compiled with an explicit ORDER BY -- so the LLM only ever produces a structured
# spec (it never computes a number), and a `; DROP` style injected column name is
# rejected as a typed error before any SQL runs (INV-1).
# ---------------------------------------------------------------------------

# Whitelisted aggregate operators. Each maps to a DuckDB SQL template that takes a
# single already-quoted column identifier. bad_rate/approval_rate encode the fixed
# credit-risk conventions (mean of a 0/1 target, share of an approve decision).
_AGG_COMPARATORS = {"==": "=", "!=": "<>", ">": ">", ">=": ">=", "<": "<", "<=": "<="}
_MAX_GROUP_BY = 3
_MAX_FILTERS = 8
_DEFAULT_TOP_K = 50


def tool_slice_aggregate(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset_id = str(inputs["dataset_id"])
    dataset = runtime.registry.get(dataset_id)
    path = runtime.registry.resolve_path(dataset.id)
    # The column whitelist IS the dataset profile: only names the backend can see in
    # the physical file are legal anywhere in the spec (group_by/metrics/filters/
    # month_col/sort_by). Anything else -> DataSecurityError from sql_identifier.
    allowed_columns = set(runtime.backend.column_names(path))

    group_by = [str(col) for col in (inputs.get("group_by") or [])]
    if len(group_by) > _MAX_GROUP_BY:
        raise ValueError(f"group_by supports at most {_MAX_GROUP_BY} columns")
    metrics = [dict(metric) for metric in (inputs.get("metrics") or []) if isinstance(metric, dict)]
    if not metrics:
        raise ValueError("slice_aggregate requires at least one metric")
    filters = [dict(f) for f in (inputs.get("filters") or []) if isinstance(f, dict)]
    if len(filters) > _MAX_FILTERS:
        raise ValueError(f"filters supports at most {_MAX_FILTERS} conditions")
    top_k = int(inputs.get("top_k") or _DEFAULT_TOP_K)
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    group_sql = [sql_identifier(col, allowed_columns) for col in group_by]
    metric_selects, metric_labels = _metric_selects(metrics, allowed_columns)
    where_sql, where_params = _filter_clause(filters, allowed_columns)
    month_where_sql, month_params = _month_clause(
        _optional_str(inputs.get("month_col")),
        inputs.get("months"),
        allowed_columns,
    )
    all_where = [clause for clause in (where_sql, month_where_sql) if clause]
    where_clause = f" WHERE {' AND '.join(all_where)}" if all_where else ""

    order_sql = _order_clause(
        _optional_str(inputs.get("sort_by")), group_by, metric_labels, allowed_columns
    )
    rel = runtime.backend._duckdb_rel(path)  # parquet_rel/csv_rel -- read-only scan
    select_parts = [*group_sql, *metric_selects]
    query = (
        f"SELECT {', '.join(select_parts)} FROM {rel}{where_clause}"
        + (f" GROUP BY {', '.join(group_sql)}" if group_sql else "")
        + f" ORDER BY {order_sql}"
        + f" LIMIT {int(top_k) + 1}"  # fetch one extra row to detect truncation
    )
    params = [*where_params, *month_params]
    with connect_duckdb(runtime.backend._temp_directory) as conn:
        scanned_row = conn.execute(f"SELECT count(*) FROM {rel}{where_clause}", params).fetchone()
        frame = conn.execute(query, params).df()

    n_rows_scanned = int(scanned_row[0] or 0)
    truncated = len(frame) > top_k
    if truncated:
        frame = frame.head(top_k)
    columns = [*group_by, *metric_labels]
    rows = [
        {column: _jsonable_cell(value) for column, value in zip(columns, record, strict=True)}
        for record in frame.itertuples(index=False, name=None)
    ]

    red_flags: list[dict] = []
    if not rows:
        red_flags.append({
            "code": "empty_result",
            "level": "amber",
            "message": "当前口径下无匹配样本，请检查筛选条件或时间范围。",
        })
    if truncated:
        red_flags.append({
            "code": "truncated",
            "level": "amber",
            "message": f"结果超过 top_k={top_k} 行已截断，请收窄分组或加筛选。",
        })
    # A4: bad_rate/approval_rate denominators drop unlabeled rows. If any group carries
    # excluded rows, surface it so the reader knows the rate is over labeled samples only
    # rather than silently trusting a deflated column.
    unlabeled_total = sum(
        int(cell)
        for row in rows
        for label, cell in row.items()
        if label.startswith("unlabeled_count_") and isinstance(cell, (int, float)) and cell
    )
    if unlabeled_total:
        red_flags.append({
            "code": "unlabeled_present",
            "level": "amber",
            "message": (
                f"{unlabeled_total} 行标签/决策缺失或无法识别，坏率/批准率仅基于已标注样本，"
                "请对照 unlabeled_count_* 列判断覆盖度。"
            ),
        })

    spec_echo = {
        "dataset_id": dataset_id,
        "group_by": group_by,
        "metrics": [{"op": str(m.get("op")), "col": _optional_str(m.get("col"))} for m in metrics],
        "filters": [
            {"col": str(f.get("col")), "op": str(f.get("op")), "value": _jsonable_cell(f.get("value"))}
            for f in filters
        ],
        "month_col": _optional_str(inputs.get("month_col")),
        "months": [str(month) for month in (inputs.get("months") or [])],
        "top_k": top_k,
        "sort_by": _optional_str(inputs.get("sort_by")),
    }

    with connect(runtime.settings.db_path) as conn:
        _write_audit_row(
            conn,
            kind="data.slice_aggregate",
            target_ref=dataset_id,
            outcome="succeeded",
            detail={
                "task_id": str(ctx.task_id),
                "group_by": group_by,
                "metrics": spec_echo["metrics"],
                "n_rows_scanned": n_rows_scanned,
                "n_rows_returned": len(rows),
                "truncated": truncated,
            },
        )

    return {
        "columns": columns,
        "rows": rows,
        "spec_echo": spec_echo,
        "n_rows_scanned": n_rows_scanned,
        "red_flags": red_flags,
    }


def _metric_selects(metrics: list[dict], allowed_columns: set[str]) -> tuple[list[str], list[str]]:
    """(select_expr, output_label) per metric. The op->SQL mapping is a fixed dict so
    an LLM can only pick an operator name, never inject an expression; the target
    column (when the op needs one) is validated against the profile whitelist."""
    selects: list[str] = []
    labels: list[str] = []
    seen: set[str] = set()
    for metric in metrics:
        op = str(metric.get("op") or "")
        col = _optional_str(metric.get("col"))
        label = _metric_label(op, col)
        if label in seen:
            raise ValueError(f"duplicate metric label: {label}")
        seen.add(label)
        selects.append(f"{_metric_expr(op, col, allowed_columns)} AS {_quote(label)}")
        labels.append(label)
        # A4: bad_rate/approval_rate now exclude unlabeled rows from the denominator.
        # Auto-derive a companion unlabeled_count_<col> so the excluded rows are always
        # visible (the op enum stays stable). The extra select keeps the SELECT arity in
        # step with metric_labels, so the zip(strict=True) row-build stays balanced.
        if op in {"bad_rate", "approval_rate"} and col:
            comp_label = f"unlabeled_count_{col}"
            if comp_label not in seen:
                seen.add(comp_label)
                selects.append(
                    f"{_unlabeled_count_expr(op, col, allowed_columns)} AS {_quote(comp_label)}"
                )
                labels.append(comp_label)
    return selects, labels


def _metric_expr(op: str, col: str | None, allowed_columns: set[str]) -> str:
    if op == "count":
        return "count(*)"
    if op in {"sum", "mean", "min", "max", "distinct"}:
        if not col:
            raise ValueError(f"metric op {op!r} requires a column")
        ident = sql_identifier(col, allowed_columns)
        numeric = f"try_cast({ident} AS DOUBLE)"
        return {
            "sum": f"coalesce(sum({numeric}), 0)",
            "mean": f"avg({numeric})",
            "min": f"min({numeric})",
            "max": f"max({numeric})",
            "distinct": f"count(DISTINCT {ident})",
        }[op]
    if op == "bad_rate":
        if not col:
            raise ValueError("metric op 'bad_rate' requires the target column")
        ident = sql_identifier(col, allowed_columns)
        # A4: only rows whose label casts to exactly 0/1 enter the denominator.
        # NULL / empty / non-castable ("N/A") / non-binary (2, -1, ...) -> NULL, which
        # DuckDB avg() drops — matching report_tools.labeled_count semantics and the
        # sibling mean op, instead of the old ELSE 0.0 that deflated the rate.
        num = f"try_cast({ident} AS DOUBLE)"
        return f"avg(CASE WHEN {num} = 1 THEN 1.0 WHEN {num} = 0 THEN 0.0 ELSE NULL END)"
    if op == "approval_rate":
        if not col:
            raise ValueError("metric op 'approval_rate' requires the decision column")
        ident = sql_identifier(col, allowed_columns)
        # A4/D2: only affirmatively-decided rows (approve/deny vocab) enter the
        # denominator; NULL / blank / unrecognized free text ("pending", "review")
        # -> NULL -> dropped, so an unknown decision is never miscounted as a rejection.
        norm = f"lower(trim(CAST({ident} AS VARCHAR)))"
        approve_in = ", ".join(f"'{tok}'" for tok in _APPROVE_TOKENS)
        deny_in = ", ".join(f"'{tok}'" for tok in _DENY_TOKENS)
        return (
            f"avg(CASE WHEN {norm} IN ({approve_in}) THEN 1.0 "
            f"WHEN {norm} IN ({deny_in}) THEN 0.0 ELSE NULL END)"
        )
    raise ValueError(f"unsupported metric op: {op}")


# A4/D2 approval-decision vocabulary. Only these tokens (after lower/trim) count as an
# affirmative decision; anything else is treated as "decision unknown" (unlabeled) and
# excluded from the approval_rate denominator rather than silently counted as a rejection.
_APPROVE_TOKENS = ("approve", "approved", "1", "y", "yes", "t", "true")
_DENY_TOKENS = ("reject", "rejected", "decline", "declined", "deny", "denied", "0", "n", "no", "f", "false")


def _unlabeled_count_expr(op: str, col: str, allowed_columns: set[str]) -> str:
    """A4: per-group count of rows excluded from a bad_rate/approval_rate denominator
    (label NULL / non-binary, or decision NULL / unrecognized). count(*) minus the
    affirmatively-labeled rows, so a fully-labeled group yields 0."""
    ident = sql_identifier(col, allowed_columns)
    if op == "bad_rate":
        num = f"try_cast({ident} AS DOUBLE)"
        return f"count(*) - count(CASE WHEN {num} IN (0, 1) THEN 1 END)"
    norm = f"lower(trim(CAST({ident} AS VARCHAR)))"
    decided_in = ", ".join(f"'{tok}'" for tok in (*_APPROVE_TOKENS, *_DENY_TOKENS))
    return f"count(*) - count(CASE WHEN {norm} IN ({decided_in}) THEN 1 END)"


def _metric_label(op: str, col: str | None) -> str:
    return op if op == "count" or not col else f"{op}_{col}"


def _filter_clause(filters: list[dict], allowed_columns: set[str]) -> tuple[str, list]:
    """Compile filters into a parameterized WHERE (values bound, never interpolated)."""
    clauses: list[str] = []
    params: list = []
    for f in filters:
        col = str(f.get("col") or "")
        op = str(f.get("op") or "")
        value = f.get("value")
        ident = sql_identifier(col, allowed_columns)
        if op in _AGG_COMPARATORS:
            clauses.append(f"{ident} {_AGG_COMPARATORS[op]} ?")
            params.append(value)
        elif op == "in":
            values = list(value) if isinstance(value, (list, tuple)) else [value]
            if not values:
                raise ValueError("filter op 'in' requires a non-empty value list")
            placeholders = ", ".join("?" for _ in values)
            clauses.append(f"{ident} IN ({placeholders})")
            params.extend(values)
        elif op == "between":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("filter op 'between' requires a [low, high] value pair")
            clauses.append(f"{ident} BETWEEN ? AND ?")
            params.extend([value[0], value[1]])
        else:
            raise ValueError(f"unsupported filter op: {op}")
    return " AND ".join(clauses), params


def _month_clause(month_col: str | None, months, allowed_columns: set[str]) -> tuple[str, list]:
    if not month_col:
        return "", []
    month_values = [str(month) for month in (months or [])]
    if not month_values:
        return "", []
    ident = sql_identifier(month_col, allowed_columns)
    placeholders = ", ".join("?" for _ in month_values)
    return f"CAST({ident} AS VARCHAR) IN ({placeholders})", month_values


def _order_clause(
    sort_by: str | None,
    group_by: list[str],
    metric_labels: list[str],
    allowed_columns: set[str],
) -> str:
    """Explicit deterministic ordering. sort_by may name a group column or a metric
    output label; default is group_by lexicographic (or the first metric when there
    is no group_by), so identical inputs always yield identical row order (INV-1)."""
    if sort_by:
        if sort_by in metric_labels:
            return f"{_quote(sort_by)} DESC"
        # A group column must be whitelisted; sort ascending for stable ordering.
        return f"{sql_identifier(sort_by, allowed_columns)} ASC"
    if group_by:
        return ", ".join(f"{sql_identifier(col, allowed_columns)} ASC" for col in group_by)
    return f"{_quote(metric_labels[0])} DESC"


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _jsonable_cell(value):
    if value is None:
        return None
    try:
        import math

        if isinstance(value, float):
            return value if math.isfinite(value) else None
    except Exception:
        pass
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class _Runtime(PackRuntime):
    def _extend(self, ctx) -> None:
        self.aligner = ColumnAligner(self.backend)
        self.join_engine = JoinEngine(self.backend, self.aligner, self.registry, self.repo)
        self.transforms = DataTransformRepository(self.settings.db_path)
        self.task_artifacts = TaskArtifactRepository(self.settings.db_path)


def _runtime(ctx) -> _Runtime:
    return _Runtime(ctx)


def _seed(ctx) -> int:
    return int(ctx.seed or 0)


def _optional_str(value) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _apply_clean_op(series: pd.Series, op: str) -> pd.Series:
    if op == "strip":
        return series.map(lambda value: value.strip() if isinstance(value, str) else value)
    if op == "lower":
        return series.map(lambda value: value.lower() if isinstance(value, str) else value)
    if op == "upper":
        return series.map(lambda value: value.upper() if isinstance(value, str) else value)
    if op == "to_numeric":
        return pd.to_numeric(series, errors="coerce")
    if op == "to_datetime":
        return pd.to_datetime(series, errors="coerce")
    raise ValueError(f"unsupported clean op: {op}")


def _dataset_payload(dataset) -> dict:
    return {
        "id": dataset.id,
        "task_id": dataset.task_id,
        "role": dataset.role,
        "source_path": dataset.source_path,
        "format": dataset.format,
        "sheet": dataset.sheet,
        "row_count": dataset.row_count,
        "columns": [_column_payload(column) for column in dataset.columns],
        "has_target": dataset.has_target,
        "target_col": dataset.target_col,
    }


def _column_payload(column) -> dict:
    return {
        "name": column.name,
        "dtype": column.dtype,
        "semantic_role": column.semantic_role,
        "null_rate": column.null_rate,
        "cardinality": column.cardinality,
        "sample_values": list(column.sample_values),
        "fingerprint": asdict(column.fingerprint),
    }


def _key_pair_payload(pair) -> dict:
    return {
        "anchor_col": pair.anchor_col,
        "feature_col": pair.feature_col,
        "match_method": pair.match_method,
        "transform_side": pair.transform_side,
        "match_rate": pair.match_rate,
        "resolved_by": pair.resolved_by,
        # T1-B8: dtype provenance for each key side + cross-file divergence flag.
        "anchor_dtype": getattr(pair, "anchor_dtype", ""),
        "feature_dtype": getattr(pair, "feature_dtype", ""),
        "dtype_divergent": getattr(pair, "dtype_divergent", False),
    }


def _diagnostics_payload(diagnostics) -> dict:
    return asdict(diagnostics)


def _friendly_name(registry, dataset_id) -> str:
    """A human-readable file name for a dataset id (e.g. ``features.parquet``) so the
    diagnostics / dedup gate show the source file rather than a raw ``ds_<hash>``."""
    try:
        dataset = registry.get(str(dataset_id))
        source = getattr(dataset, "source_path", None)
        return Path(source).name if source else str(dataset_id)
    except Exception:
        return str(dataset_id)


def _join_plan_payload(plan) -> dict:
    return {
        "join_plan_id": plan.id,
        "anchor_dataset_id": plan.anchor_dataset_id,
        "status": plan.status,
        "joins": [
            {
                "feature_id": spec.feature_dataset_id,
                "key_pairs": [_key_pair_payload(pair) for pair in spec.key_pairs],
                "diagnostics": _diagnostics_payload(spec.diagnostics),
                "dedup_strategy": spec.dedup_strategy,
                "confirmed": spec.confirmed,
            }
            for spec in plan.joins
        ],
    }
