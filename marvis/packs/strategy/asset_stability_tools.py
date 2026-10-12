"""Authenticated two-phase independent stability for frozen candidate assets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from marvis.artifacts import ArtifactUnitOfWork
from marvis.files import sha256_file
from marvis.packs.strategy.asset_stability import (
    FREEZE_SCHEMA,
    MAX_BUCKETS,
    MAX_ROWS,
    SCHEMA,
    canonical,
    measure_asset_stability,
    seal,
    validate_evidence,
    validate_sealed,
)
from marvis.packs.strategy.candidate_fragment import verified_fragment_pool_parts
from marvis.packs.strategy.candidate_stability_tools import (
    _modeling_runtime,
    _read_authenticated_parquet_snapshot,
)
from marvis.packs.strategy.evaluator import evaluate_expression_frame
from marvis.packs.strategy.errors import StrategyError
import marvis.packs.strategy.pool_tools as pool
from marvis.packs.strategy.pool_requirement_resolver import (
    hydrate_requirement_fields,
    project_pool_entry_requirements,
    require_resolved_pool_requirements_on_connection,
    resolve_pool_requirements,
)
from marvis.packs.strategy.pool_validation_tools import _sample_risk_development_ref
from marvis.packs.strategy.sample_design_v2_tools import (
    load_requested_strategy_sample_design_v2_artifacts,
    require_any_strategy_sample_design_v2_artifact_binding_on_connection,
    resolve_strategy_sample_design_v2_source_mode,
)

FREEZE_KIND = "strategy_asset_stability_freeze_json"
ARTIFACT_KIND = "strategy_asset_independent_stability_json"
FREEZE_TOOL = "strategy.freeze_candidate_asset_stability"
ORIGIN_TOOL = "strategy.measure_candidate_asset_stability"
_HASH = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_FIELDS = {"artifact_id", "artifact_content_hash", "asset_id", "asset_hash"}
_SAMPLE_FIELDS = {
    "membership_artifact_id",
    "expected_membership_artifact_content_hash",
    "bundle_artifact_id",
    "expected_bundle_artifact_content_hash",
    "expected_bundle_id",
    "expected_sample_design_id",
    "expected_sample_design_content_hash",
}
_REF_FIELDS = {"artifact_id", "artifact_content_hash", "stability_id", "content_hash"}
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_EVALUATION_CELLS = 50_000_000


@dataclass(frozen=True)
class AssetStabilityBinding:
    task_id: str
    artifact_id: str
    path: Path
    artifact_content_hash: str
    evidence: dict
    provenance: dict
    tasks_root: Path
    frozen_binding: AssetStabilityBinding | None = None

    @property
    def ref(self):
        return {
            "artifact_id": self.artifact_id,
            "artifact_content_hash": self.artifact_content_hash,
            "stability_id": self.evidence["stability_id"],
            "content_hash": self.evidence["content_hash"],
        }


@dataclass
class _Source:
    source_kind: str
    identity: dict
    sample_ref: dict
    target_col: str
    definitions: list[dict]
    requirements: list[dict]
    capability: object
    cas: object
    score_field: str | None = None


def _exact(value, fields, name):
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise StrategyError(
            f"{name} requires exact fields: {', '.join(sorted(fields))}"
        )
    return dict(value)


def _hash(value, name):
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise StrategyError(f"{name} must be a SHA256 identity")
    return value


def _source_ref(value):
    ref = _exact(value, _SOURCE_FIELDS, "asset source reference")
    for name in ("artifact_id", "artifact_content_hash", "asset_hash"):
        _hash(ref[name], name)
    if not isinstance(ref["asset_id"], str) or not 0 < len(ref["asset_id"]) <= 160:
        raise StrategyError("asset_id is invalid")
    return ref


def _sample_ref(value):
    ref = _exact(value, _SAMPLE_FIELDS, "physical SampleDesign V2 reference")
    for name in _SAMPLE_FIELDS - {"expected_bundle_id", "expected_sample_design_id"}:
        _hash(ref[name], name)
    return ref


def _artifact_ref(value):
    ref = _exact(value, _REF_FIELDS, "asset stability reference")
    for name in ("artifact_id", "artifact_content_hash", "content_hash"):
        _hash(ref[name], name)
    return ref


def _record(runtime, task_id, ref):
    record = runtime.task_artifacts.get_for_task(task_id, ref["artifact_id"])
    if record is None or record["content_hash"] != ref["artifact_content_hash"]:
        raise StrategyError("candidate source artifact is missing or changed")
    return record


def _load_source(runtime, *, task_id, ref) -> _Source:
    ref = _source_ref(ref)
    record = _record(runtime, task_id, ref)
    provenance = record["provenance"]
    kind = record["kind"]
    kwargs = dict(
        task_id=task_id,
        artifact_id=ref["artifact_id"],
        expected_content_hash=ref["artifact_content_hash"],
        expected_asset_id=ref["asset_id"],
        expected_asset_hash=ref["asset_hash"],
    )
    if kind == "strategy_interactive_tree_revision_json":
        from marvis.packs.strategy.interactive_tree_tools import (
            load_verified_interactive_tree_revision,
            load_verified_interactive_tree_revision_on_connection,
        )

        revision = load_verified_interactive_tree_revision(
            runtime,
            task_id=task_id,
            revision_id=provenance["revision_id"],
            max_total_bytes=MAX_JSON_BYTES,
        )
        asset = revision.revision
        if (
            revision.artifact_id != ref["artifact_id"]
            or revision.content_hash != ref["artifact_content_hash"]
            or asset["semantic_tree_id"] != ref["asset_id"]
            or asset["tree"]["tree_hash"] != ref["asset_hash"]
        ):
            raise StrategyError("interactive frontier exact revision reference changed")
        tree = revision.automatic_source
        pool._load_automatic_tree_dataset(
            runtime, task_id=task_id, tree=tree, cache=pool._LineageCache.empty()
        )

        def cas(conn):
            current = load_verified_interactive_tree_revision_on_connection(
                conn,
                runtime=runtime,
                task_id=task_id,
                revision_id=provenance["revision_id"],
                max_total_bytes=MAX_JSON_BYTES,
            )
            if current != revision:
                raise StrategyError(
                    "interactive frontier ancestry changed after freeze"
                )

        return _Source(
            "complete_interactive_frontier",
            pool._evidence_identity_from_concrete(tree.asset["identity"]),
            pool.sample_design_ref_from_automatic_tree_source_refs(
                tree.asset["source_refs"]
            ),
            tree.asset["tree_result"]["training"]["target_col"],
            [
                {"bucket_id": row["leaf_id"], "condition": row["condition"]}
                for row in asset["fragments"]
            ],
            [],
            revision,
            cas,
        )
    if kind == "strategy_automatic_tree_asset_json":
        from marvis.packs.strategy.automatic_tree_leaf_tools import (
            load_verified_automatic_tree_source_artifact,
            load_verified_automatic_tree_source_artifact_on_connection,
        )

        kwargs["expected_tree_result_hash"] = provenance["tree_result_hash"]
        tree = load_verified_automatic_tree_source_artifact(runtime, **kwargs)
        pool._load_automatic_tree_dataset(
            runtime, task_id=task_id, tree=tree, cache=pool._LineageCache.empty()
        )
        asset = tree.asset
        definitions = [
            {"bucket_id": row["leaf_id"], "condition": row["condition"]}
            for row in asset["fragments"]
        ]

        def cas(conn):
            current = load_verified_automatic_tree_source_artifact_on_connection(
                conn, tasks_dir=runtime.settings.tasks_dir, **kwargs
            )
            if current != tree:
                raise StrategyError("full tree changed after freeze")

        return _Source(
            "complete_tree",
            pool._evidence_identity_from_concrete(asset["identity"]),
            pool.sample_design_ref_from_automatic_tree_source_refs(
                asset["source_refs"]
            ),
            asset["tree_result"]["training"]["target_col"],
            definitions,
            [],
            tree,
            cas,
        )
    if kind == "strategy_scorecard_band_asset_json":
        from marvis.packs.strategy.scorecard_candidate_tools import (
            load_scorecard_band_asset_artifact,
            require_scorecard_band_asset_artifact_binding_on_connection,
        )
        from marvis.packs.strategy.pool_requirement_resolver import (
            model_score_virtual_field,
        )

        score = load_scorecard_band_asset_artifact(
            _modeling_runtime(runtime),
            task_id=task_id,
            artifact_id=ref["artifact_id"],
            expected_artifact_content_hash=ref["artifact_content_hash"],
            expected_asset_id=ref["asset_id"],
            expected_asset_hash=ref["asset_hash"],
        )
        asset = score.asset
        vector = asset["source_refs"]["score_vector"]
        evidence = asset["source_refs"]["score_evidence"]
        virtual = model_score_virtual_field(vector["artifact_id"])
        requirement = {
            "type": "model_score_vector.v1",
            "virtual_field": virtual,
            "score_product": asset["score_contract"]["score_product"],
            "score_evidence_artifact_id": evidence["artifact_id"],
            "score_evidence_artifact_content_hash": evidence["artifact_content_hash"],
            "score_vector_artifact_id": vector["artifact_id"],
            "score_vector_artifact_content_hash": vector["artifact_content_hash"],
        }
        definitions = []
        for band in asset["bands"]:
            predicates = [
                {
                    "op": "compare",
                    "field": virtual,
                    "operator": op,
                    "value": value,
                    "missing": "no_match",
                }
                for key, op in (("lower_bound", ">="), ("upper_bound", "<"))
                if (value := band[key]) is not None
            ]
            condition = (
                predicates[0]
                if len(predicates) == 1
                else {"op": "and", "args": predicates}
            )
            definitions.append({"bucket_id": band["bin_id"], "condition": condition})
        return _Source(
            "scorecard_bands",
            pool._evidence_identity_from_concrete(asset["identity"]),
            _sample_risk_development_ref(score.sample_design).to_ref_dict(),
            score.sample_design.source_binding.target_col,
            definitions,
            [requirement],
            score,
            lambda conn: require_scorecard_band_asset_artifact_binding_on_connection(
                conn, score
            ),
            score_field=virtual,
        )
    if kind == "strategy_cross_matrix_candidate_json":
        from marvis.packs.strategy.cross_matrix_cell_selection_tools import (
            load_verified_cross_matrix_source_artifact,
            load_verified_cross_matrix_source_artifact_on_connection,
        )
        from marvis.packs.strategy.cross_matrix_candidate import (
            rebuild_cross_matrix_candidate_asset,
        )

        kwargs.update(
            expected_candidate_id=provenance["candidate_id"],
            expected_evidence_hash=provenance["evidence_hash"],
        )
        matrix = load_verified_cross_matrix_source_artifact(runtime, **kwargs)
        parent = matrix.asset["parent"]
        parent_record = pool._load_source_artifact(
            runtime,
            task_id=task_id,
            artifact_id=provenance["source_artifact_id"],
            expected_content_hash=provenance["source_artifact_content_hash"],
            expected_candidate_id=parent["candidate_id"],
            expected_evidence_hash=parent["evidence_hash"],
        )
        evidence = pool._read_canonical_parent_evidence(parent_record.path)
        pool._require_report_binding(
            evidence,
            source=parent_record,
            task_id=task_id,
            expected_candidate_id=parent["candidate_id"],
            expected_evidence_hash=parent["evidence_hash"],
        )
        if rebuild_cross_matrix_candidate_asset(matrix.asset, evidence) != matrix.asset:
            raise StrategyError("Cross matrix source evidence cannot be replayed")
        pool._load_dataset_binding(runtime, evidence=evidence, source=parent_record)

        def cas(conn):
            current = load_verified_cross_matrix_source_artifact_on_connection(
                conn, tasks_dir=runtime.settings.tasks_dir, **kwargs
            )
            pool._require_source_on_connection(conn, parent_record)
            if (
                current != matrix
                or sha256_file(parent_record.path) != parent_record.content_hash
            ):
                raise StrategyError("Cross matrix ancestry changed after freeze")

        identity = {
            **evidence["identity"],
            "sample_context_hash": pool.sample_context_hash_from_candidate_evidence(
                evidence
            ),
        }
        return _Source(
            "cross_matrix",
            pool._evidence_identity_from_concrete(identity),
            evidence["generation"]["parameters"]["sample_design_ref"],
            evidence["analysis"]["target"],
            [
                {"bucket_id": row["cell_id"], "condition": row["rule"]["condition"]}
                for row in matrix.asset["matrix"]["cells"]
            ],
            [],
            matrix,
            cas,
        )
    lineage = pool._load_candidate_lineage(runtime, **kwargs)
    facts = pool._pool_lineage_development_facts(
        lineage, active_voting=set(), voting_nodes=set(), facts_by_lineage={}
    )
    _binding, rule_id, execution = verified_fragment_pool_parts(
        lineage.verified_fragment
    )
    source_kind = lineage.source_binding["asset_type"]
    return _Source(
        source_kind,
        facts.evidence_identity,
        facts.sample_ref.to_ref_dict(),
        facts.target_col,
        [{"bucket_id": rule_id, "condition": execution["condition"]}],
        list(execution["requirements"]),
        lineage,
        lambda conn: pool._require_lineage_on_connection(
            conn, lineage, tasks_root=Path(runtime.settings.tasks_dir)
        ),
    )


def _load_binding(runtime, *, task_id, source_ref, sample_ref, partitions):
    source = _load_source(runtime, task_id=task_id, ref=source_ref)
    sample = load_requested_strategy_sample_design_v2_artifacts(
        runtime, task_id, {"sample_design_ref": _sample_ref(sample_ref)}
    )
    design = sample.bundle["sample_design"]
    resolve_strategy_sample_design_v2_source_mode(
        design, capability="physical_v2", consumer="strategy_pool_validation"
    )
    target = design["target_selector"]
    if target["status"] != "resolved" or source.target_col != target["column"]:
        raise StrategyError("candidate target is not authenticated by SampleDesign V2")
    if design["sample_semantics"]["scope"] != "strategy_development":
        raise StrategyError("candidate stability requires governed strategy scope")
    risk = next(row for row in sample.bundle["populations"] if row["role"] == "risk")
    if risk["maturity_evidence"]["status"] != "confirmed_matured":
        raise StrategyError(
            "unknown outcome maturity cannot establish independent validation"
        )
    fields = design["sample_semantics"]["field_bindings"]
    month_col = fields["month_field"] or fields["time_field"]
    if not month_col:
        raise StrategyError(
            "unknown observation time cannot establish independent validation"
        )
    if source.sample_ref != _sample_risk_development_ref(sample).to_ref_dict():
        raise StrategyError(
            "candidate development sample differs from independent SampleDesign"
        )
    live = sample.source_binding
    for key, value in (
        ("dataset_id", live.dataset_id),
        ("dataset_content_hash", live.dataset_content_hash),
        ("workspace_revision", live.workspace_revision),
        ("workspace_generation", live.workspace_generation),
        ("semantic_mapping_hash", live.semantic_mapping_hash),
    ):
        if source.identity[key] != value:
            raise StrategyError(f"candidate {key} differs from independent source")
    if (
        target["column"] != live.target_col
        or target["bad_value"] != live.target_bad_value
    ):
        raise StrategyError("candidate target polarity differs from independent source")
    if (
        not isinstance(partitions, list)
        or not partitions
        or len(set(partitions)) != len(partitions)
        or any(p not in {"validation", "oot"} for p in partitions)
    ):
        raise StrategyError(
            "explicit independent partitions must be validation and/or oot"
        )
    counts = sample.membership["header"]["counts"]["risk"]
    for p in ["development", *partitions]:
        if counts[p] <= 0:
            raise StrategyError(f"candidate stability risk/{p} partition is empty")
    if live.row_count > MAX_ROWS or len(source.definitions) + 1 > MAX_BUCKETS:
        raise StrategyError("candidate stability exceeds explicit row/bucket budget")
    if live.row_count * len(source.definitions) > MAX_EVALUATION_CELLS:
        raise StrategyError(
            "candidate stability exact row-by-bucket evaluation exceeds cell budget"
        )
    entries = [
        {
            "source": {"fragment_id": "asset-stability"},
            "rule_id": source.definitions[0]["bucket_id"],
            "execution": {"requirements": source.requirements},
        }
    ]
    requirements = (
        project_pool_entry_requirements(entries) if source.requirements else []
    )
    resolved = resolve_pool_requirements(
        _modeling_runtime(runtime),
        task_id=task_id,
        compiled_design={"requirements": list(requirements)},
        sample_design=sample,
    )
    return source, sample, resolved, month_col


def _frozen_manifest(
    *, task_id, source_ref, sample_ref, partitions, source, sample, resolved, month_col
):
    membership = {}
    for p in ["development", *partitions]:
        mask = np.asarray(sample.membership["masks"][f"risk/{p}"], dtype=np.bool_)
        membership[p] = {
            "count": int(mask.sum()),
            "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        }
    body = {
        "schema_version": FREEZE_SCHEMA,
        "task_id": task_id,
        "source_ref": dict(source_ref),
        "source_kind": source.source_kind,
        "identity": source.identity,
        "sample_design_ref": dict(sample_ref),
        "development_ref": source.sample_ref,
        "partitions": list(partitions),
        "membership": membership,
        "bucket_definitions": source.definitions,
        "unmatched_bucket": "unmatched",
        "requirements": source.requirements,
        "requirement_binding": {
            "requirements_hash": resolved.requirements_hash,
            "requirements": list(resolved.requirements),
        },
        "target": dict(sample.bundle["sample_design"]["target_selector"]),
        "month_col": month_col,
        "baseline_policy": "fixed_development_no_independent_fit",
        "maturity": "confirmed_matured",
        "sample_diagnostics": sample.bundle["diagnostics"],
        "score_contract": None
        if source.score_field is None
        else {
            "field": source.score_field,
            "direction": "higher_is_bad",
            "product": "authenticated_raw_bad_probability",
        },
    }
    return seal(body, prefix="asset-freeze")


def run_freeze_candidate_asset_stability(inputs, ctx, runtime):
    request = _exact(
        inputs,
        {"source_ref", "sample_design_ref", "partitions"},
        "freeze candidate asset stability",
    )
    source_ref = _source_ref(request["source_ref"])
    sample_ref = _sample_ref(request["sample_design_ref"])
    source, sample, resolved, month_col = _load_binding(
        runtime,
        task_id=ctx.task_id,
        source_ref=source_ref,
        sample_ref=sample_ref,
        partitions=request["partitions"],
    )
    freeze = _frozen_manifest(
        task_id=ctx.task_id,
        source_ref=source_ref,
        sample_ref=sample_ref,
        partitions=request["partitions"],
        source=source,
        sample=sample,
        resolved=resolved,
        month_col=month_col,
    )
    return _publish(
        runtime,
        task_id=ctx.task_id,
        payload=freeze,
        kind=FREEZE_KIND,
        origin=FREEZE_TOOL,
        guard=lambda conn: _cas(conn, source, sample, resolved),
    )


def _cas(conn, source, sample, resolved):
    source.cas(conn)
    require_any_strategy_sample_design_v2_artifact_binding_on_connection(conn, sample)
    require_resolved_pool_requirements_on_connection(conn, resolved)
    if (
        sha256_file(sample.source_binding.dataset_path)
        != sample.source_binding.dataset_content_hash
    ):
        raise StrategyError("candidate stability dataset bytes changed")


def _fields(value):
    if isinstance(value, Mapping):
        result = {value["field"]} if isinstance(value.get("field"), str) else set()
        for child in value.values():
            result.update(_fields(child))
        return result
    if isinstance(value, list):
        return set().union(*(_fields(child) for child in value)) if value else set()
    return set()


def run_measure_candidate_asset_stability(inputs, ctx, runtime):
    request = _exact(inputs, {"frozen_ref"}, "measure candidate asset stability")
    frozen = load_asset_stability_artifact(
        runtime, task_id=ctx.task_id, ref=request["frozen_ref"], freeze=True
    )
    manifest = frozen.evidence
    source, sample, resolved, month_col = _load_binding(
        runtime,
        task_id=ctx.task_id,
        source_ref=manifest["source_ref"],
        sample_ref=manifest["sample_design_ref"],
        partitions=manifest["partitions"],
    )
    rebuilt = _frozen_manifest(
        task_id=ctx.task_id,
        source_ref=manifest["source_ref"],
        sample_ref=manifest["sample_design_ref"],
        partitions=manifest["partitions"],
        source=source,
        sample=sample,
        resolved=resolved,
        month_col=month_col,
    )
    if rebuilt != manifest:
        raise StrategyError("candidate frozen source/sample/model closure changed")
    with runtime.task_artifacts.transaction() as conn:
        rows = conn.execute(
            "SELECT id, content_hash, provenance_json FROM task_artifacts WHERE task_id=? AND kind=? AND origin_tool=? "
            "AND json_extract(provenance_json, '$.frozen_ref.artifact_id')=?",
            (ctx.task_id, ARTIFACT_KIND, ORIGIN_TOOL, frozen.artifact_id),
        ).fetchall()
    if len(rows) > 1:
        raise StrategyError("frozen asset has ambiguous persisted independent evidence")
    if rows:
        row = rows[0]
        provenance = json.loads(row["provenance_json"])
        cached = load_asset_stability_artifact(
            runtime,
            task_id=ctx.task_id,
            ref={
                "artifact_id": row["id"],
                "artifact_content_hash": row["content_hash"],
                "stability_id": provenance["stability_id"],
                "content_hash": provenance["content_hash"],
            },
        )
        if (
            cached.evidence["frozen_ref"] != frozen.ref
            or cached.evidence["freeze"] != manifest
        ):
            raise StrategyError(
                "persisted asset stability does not match this exact frozen closure"
            )
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            require_asset_stability_binding_on_connection(conn, cached)
            _cas(conn, source, sample, resolved)
        return _output(
            cached.evidence,
            task_id=ctx.task_id,
            artifact_id=cached.artifact_id,
            artifact_content_hash=cached.artifact_content_hash,
            kind=ARTIFACT_KIND,
        )
    physical = _fields(source.definitions) - set(resolved.virtual_fields)
    physical.update([source.target_col, month_col])
    frame = _read_authenticated_parquet_snapshot(
        sample.source_binding.dataset_path,
        root=Path(runtime.settings.datasets_dir).absolute(),
        expected_content_hash=sample.source_binding.dataset_content_hash,
        columns=sorted(physical),
    ).reset_index(drop=True)
    if len(frame) != sample.source_binding.row_count:
        raise StrategyError("candidate raw row universe changed")
    frame = hydrate_requirement_fields(frame, resolved=resolved)
    raw_masks = {
        p: np.asarray(sample.membership["masks"][f"risk/{p}"], dtype=np.bool_)
        for p in ["development", *manifest["partitions"]]
    }
    union = np.logical_or.reduce(list(raw_masks.values()))
    if union.shape != (len(frame),):
        raise StrategyError("candidate raw row membership universe changed")
    # Authenticate/hydrate model vectors on raw ordinals first. Other populations
    # cannot turn an otherwise valid risk observation time/label into a refusal.
    frame = frame.loc[union].reset_index(drop=True)
    masks = {partition: mask[union] for partition, mask in raw_masks.items()}
    assigned = np.full(len(frame), "unmatched", dtype=object)
    matched = np.zeros(len(frame), dtype=np.bool_)
    bucket_ids = []
    for definition in source.definitions:
        hit = evaluate_expression_frame(frame, definition["condition"]).to_numpy(
            dtype=np.bool_
        )
        if np.any(hit & matched):
            raise StrategyError("frozen asset bucket conditions overlap")
        assigned[hit] = definition["bucket_id"]
        matched |= hit
        bucket_ids.append(definition["bucket_id"])
    bucket_ids.append("unmatched")
    numeric = pd.to_numeric(frame[source.target_col], errors="coerce")
    if (frame[source.target_col].notna() & numeric.isna()).any():
        raise StrategyError("candidate labels cannot be coerced to missing outcomes")
    raw = numeric.to_numpy(dtype=float)
    if not np.isin(raw[np.isfinite(raw)], [0, 1]).all():
        raise StrategyError(
            "candidate target labels are outside declared binary semantics"
        )
    labels = raw if sample.source_binding.target_bad_value == 1 else 1 - raw
    evidence = measure_asset_stability(
        freeze=manifest,
        frozen_ref=frozen.ref,
        buckets=assigned,
        bucket_ids=bucket_ids,
        labels=labels,
        months=frame[month_col],
        masks=masks,
        scores=None
        if source.score_field is None
        else frame[source.score_field].to_numpy(dtype=float),
    )

    def guard(conn):
        require_asset_stability_binding_on_connection(conn, frozen)
        _cas(conn, source, sample, resolved)

    return _publish(
        runtime,
        task_id=ctx.task_id,
        payload=evidence,
        kind=ARTIFACT_KIND,
        origin=ORIGIN_TOOL,
        guard=guard,
    )


def _safe_path(path, root):
    path = Path(path).absolute()
    root = Path(root).absolute()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise StrategyError("asset stability path is outside task storage") from exc
    if any(part.is_symlink() for part in [path, *path.parents] if part != root.parent):
        raise StrategyError("asset stability symlink paths are forbidden")
    return path


def _publish(runtime, *, task_id, payload, kind, origin, guard):
    content = canonical(payload)
    digest = hashlib.sha256(content).hexdigest()
    if len(content) > MAX_JSON_BYTES:
        raise StrategyError("asset stability artifact exceeds JSON budget")
    root = Path(runtime.settings.tasks_dir).absolute()
    directory = _safe_path(root / task_id / "outputs" / "asset_stability", root)
    directory.mkdir(parents=True, exist_ok=True)
    path = _safe_path(directory / f"{payload['stability_id']}.json", root)
    provenance = {
        "schema_version": payload["schema_version"],
        "task_id": task_id,
        "stability_id": payload["stability_id"],
        "content_hash": payload["content_hash"],
    }
    if kind == ARTIFACT_KIND:
        provenance["frozen_ref"] = payload["frozen_ref"]
    uow = ArtifactUnitOfWork()
    staged = uow.stage_file(directory, path.name)
    staged.path.write_bytes(content)
    try:
        with runtime.task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            guard(conn)
            row = conn.execute(
                "SELECT * FROM task_artifacts WHERE task_id=? AND path=?",
                (task_id, str(path)),
            ).fetchone()
            if row:
                if (
                    row["kind"] != kind
                    or row["origin_tool"] != origin
                    or row["content_hash"] != digest
                    or json.loads(row["provenance_json"]) != provenance
                    or not path.is_file()
                    or path.read_bytes() != content
                ):
                    raise StrategyError(
                        "asset stability exact retry disagrees with registered evidence"
                    )
                uow.rollback()
            else:
                if path.exists() or path.is_symlink():
                    raise StrategyError(
                        "asset stability output exists without authenticated registry row"
                    )
                uow.promote_all()
            record = runtime.task_artifacts.register_on_connection(
                conn,
                task_id=task_id,
                kind=kind,
                path=str(path),
                content_hash=digest,
                origin_tool=origin,
                provenance=provenance,
            )
            conn.commit()
        uow.commit()
    except Exception:
        uow.rollback()
        raise
    return _output(
        payload,
        task_id=task_id,
        artifact_id=record["id"],
        artifact_content_hash=digest,
        kind=kind,
    )


def _output(payload, *, task_id, artifact_id, artifact_content_hash, kind):
    ref = {
        "artifact_id": artifact_id,
        "artifact_content_hash": artifact_content_hash,
        "stability_id": payload["stability_id"],
        "content_hash": payload["content_hash"],
    }
    return {
        "schema_version": payload["schema_version"],
        "ref": ref,
        "evidence": payload,
        "artifacts": [
            {
                "artifact_id": artifact_id,
                "kind": kind,
                "content_hash": artifact_content_hash,
                "url": f"/api/tasks/{task_id}/task-artifacts/{artifact_id}/download?expected_content_hash={artifact_content_hash}",
            }
        ],
    }


def load_asset_stability_artifact(runtime, *, task_id, ref, freeze=False):
    ref = _artifact_ref(ref)
    record = _record(runtime, task_id, ref)
    schema, prefix, kind, origin = (
        (FREEZE_SCHEMA, "asset-freeze", FREEZE_KIND, FREEZE_TOOL)
        if freeze
        else (SCHEMA, "asset-stability", ARTIFACT_KIND, ORIGIN_TOOL)
    )
    if record["kind"] != kind or record["origin_tool"] != origin:
        raise StrategyError("asset stability artifact provenance is unsupported")
    path = _safe_path(record["path"], runtime.settings.tasks_dir)
    if not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise StrategyError("asset stability artifact is missing or exceeds budget")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != ref["artifact_content_hash"]:
        raise StrategyError("asset stability artifact bytes changed")
    try:
        payload = json.loads(content)
    except (ValueError, TypeError) as exc:
        raise StrategyError("asset stability artifact JSON is invalid") from exc
    payload = (
        validate_sealed(payload, schema=schema, prefix=prefix)
        if freeze
        else validate_evidence(payload)
    )
    provenance = {
        "schema_version": schema,
        "task_id": task_id,
        "stability_id": ref["stability_id"],
        "content_hash": ref["content_hash"],
    }
    if not freeze:
        provenance["frozen_ref"] = payload["frozen_ref"]
    if (
        canonical(payload) != content
        or payload["stability_id"] != ref["stability_id"]
        or payload["content_hash"] != ref["content_hash"]
        or record["provenance"] != provenance
    ):
        raise StrategyError("asset stability exact identity/provenance changed")
    if (payload if freeze else payload["freeze"])["task_id"] != task_id:
        raise StrategyError("asset stability belongs to another task")
    frozen_binding = None
    if not freeze:
        frozen_binding = load_asset_stability_artifact(
            runtime, task_id=task_id, ref=payload["frozen_ref"], freeze=True
        )
        if frozen_binding.evidence != payload["freeze"]:
            raise StrategyError(
                "asset stability frozen evidence differs from its authenticated record"
            )
    return AssetStabilityBinding(
        task_id,
        ref["artifact_id"],
        path,
        ref["artifact_content_hash"],
        payload,
        provenance,
        Path(runtime.settings.tasks_dir),
        frozen_binding,
    )


def require_asset_stability_binding_on_connection(conn, binding):
    row = conn.execute(
        "SELECT * FROM task_artifacts WHERE task_id=? AND id=?",
        (binding.task_id, binding.artifact_id),
    ).fetchone()
    schema = binding.evidence["schema_version"]
    kind, origin = (
        (FREEZE_KIND, FREEZE_TOOL)
        if schema == FREEZE_SCHEMA
        else (ARTIFACT_KIND, ORIGIN_TOOL)
    )
    if (
        row is None
        or row["kind"] != kind
        or row["origin_tool"] != origin
        or row["path"] != str(binding.path)
        or row["content_hash"] != binding.artifact_content_hash
        or json.loads(row["provenance_json"]) != binding.provenance
    ):
        raise StrategyError("asset stability registry changed before publication")
    path = _safe_path(binding.path, binding.tasks_root)
    if not path.is_file() or path.read_bytes() != canonical(binding.evidence):
        raise StrategyError("asset stability evidence changed before publication")
    if binding.frozen_binding is not None:
        require_asset_stability_binding_on_connection(conn, binding.frozen_binding)


def candidate_asset_stability_catalog(runtime, *, task_id):
    """List exact registered refs, never infer a latest candidate for execution."""
    rows = runtime.task_artifacts.list_for_task(task_id)
    if len(rows) > 4096:
        raise StrategyError("candidate asset catalogue exceeds bounded history window")
    sources = []
    samples = []
    evidence = []
    from marvis.packs.strategy.sample_design_v2_tools import (
        SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
    )

    allowed = {
        "strategy_automatic_tree_asset_json",
        "strategy_automatic_tree_leaf_fragment_json",
        "strategy_interactive_tree_frontier_selection_json",
        "strategy_interactive_tree_frontier_group_selection_json",
        "strategy_interactive_tree_revision_json",
        "strategy_scorecard_band_asset_json",
        "strategy_scorecard_cutoff_selection_json",
        "strategy_voting_candidate_json",
        "strategy_cross_matrix_candidate_json",
        "strategy_cross_matrix_cell_selection_json",
        "strategy_cross_rule_candidate_json",
        "strategy_candidate_asset_json",
    }
    for row in rows:
        p = row["provenance"]
        if row["kind"] in allowed:
            aid = p.get("asset_id") or p.get("semantic_tree_id")
            ahash = p.get("asset_hash") or p.get("tree_hash")
            if aid and ahash:
                ref = {
                    "artifact_id": row["id"],
                    "artifact_content_hash": row["content_hash"],
                    "asset_id": aid,
                    "asset_hash": ahash,
                }
                # Full loader proves kind/origin/schema/source closure before visibility.
                try:
                    source = _load_source(runtime, task_id=task_id, ref=ref)
                except (StrategyError, KeyError, ValueError, TypeError):
                    continue
                sources.append(
                    {
                        "ref": ref,
                        "source_kind": source.source_kind,
                        "development_ref": source.sample_ref,
                    }
                )
        if row["kind"] == SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND:
            membership = [
                m
                for m in rows
                if m["id"] == p.get("membership_artifact_id")
                and "membership" in m["kind"]
                and m["origin_tool"] == row["origin_tool"]
            ]
            if len(membership) == 1:
                m = membership[0]
                ref = {
                    "membership_artifact_id": m["id"],
                    "expected_membership_artifact_content_hash": m["content_hash"],
                    "bundle_artifact_id": row["id"],
                    "expected_bundle_artifact_content_hash": row["content_hash"],
                    "expected_bundle_id": p["bundle_id"],
                    "expected_sample_design_id": p["sample_design_id"],
                    "expected_sample_design_content_hash": p[
                        "sample_design_content_hash"
                    ],
                }
                try:
                    sample = load_requested_strategy_sample_design_v2_artifacts(
                        runtime, task_id, {"sample_design_ref": ref}
                    )
                except (StrategyError, KeyError, ValueError, TypeError):
                    continue
                samples.append(
                    {
                        "ref": ref,
                        "counts": sample.membership["header"]["counts"]["risk"],
                    }
                )
        if row["kind"] in {FREEZE_KIND, ARTIFACT_KIND}:
            ref = {
                "artifact_id": row["id"],
                "artifact_content_hash": row["content_hash"],
                "stability_id": p["stability_id"],
                "content_hash": p["content_hash"],
            }
            bound = load_asset_stability_artifact(
                runtime, task_id=task_id, ref=ref, freeze=row["kind"] == FREEZE_KIND
            )
            evidence.append(
                {"ref": bound.ref, "kind": row["kind"], "evidence": bound.evidence}
            )
    return {"sources": sources, "samples": samples, "evidence": evidence}
