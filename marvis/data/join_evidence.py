"""Native ordinary-JOIN provenance; this receipt is not point-in-time evidence.

The existing dataset/artifact transaction owns the receipt. Physical memberships
come from the actual JOIN relation, with no customer keys in the receipt JSON.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile

from marvis.data.dataset_identity import dataset_identity_equal
from marvis.data.authenticated_snapshot import materialize_authenticated_file_snapshot
from marvis.data.backend import DataBackend, indexed_parquet_relation, sql_string_literal
from marvis.feature.errors import FeatureError
from marvis.files import sha256_file
from marvis.repositories.task_artifacts import TaskArtifactRepository

KIND = "dataset_join_evidence_v1"
MEMBERS_KIND = "dataset_join_membership_v1"
ORIGIN = "data_ops.execute_join"


def plan_contract(plan):
    return {
        "join_plan_id": plan.id, "task_id": plan.task_id,
        "anchor_dataset_id": plan.anchor_dataset_id,
        "joins": [{
            "feature_dataset_id": spec.feature_dataset_id,
            "keys": [asdict(pair) for pair in spec.key_pairs],
            "dedup_strategy": spec.dedup_strategy, "confirmed": spec.confirmed,
        } for spec in plan.joins],
    }


def stable_dataset_path(registry, dataset):
    """Freeze bytes without changing the registered path frozen by sample design."""
    return materialize_authenticated_file_snapshot(
        registry.resolve_verified_path(dataset.id), root=registry.datasets_root,
        expected_sha256=dataset.content_hash,
        destination=registry.datasets_root / "_cas" / dataset.content_hash / f"{dataset.content_hash}.parquet",
    )


def _binding(dataset, names):
    return {
        "dataset_id": dataset.id, "content_hash": dataset.content_hash,
        "row_count": dataset.row_count, "columns": list(names),
        "has_target": dataset.has_target, "target_col": dataset.target_col,
    }


class JoinEvidenceWriter:
    def __init__(self, registry, plan, uow, directory):
        self.registry, self.plan, self.uow = registry, plan_contract(plan), uow
        self.root = registry.datasets_root.parent
        self.directory = directory
        self.sources = [registry.get(plan.anchor_dataset_id)] + [
            registry.get(spec.feature_dataset_id) for spec in plan.joins
        ]
        if any(ds.task_id != plan.task_id for ds in self.sources):
            raise FeatureError("JOIN source belongs to another task")
        self.paths = [stable_dataset_path(registry, ds) for ds in self.sources]
        self.bindings = [
            _binding(ds, registry.authenticated_parquet_column_names(ds.id)) for ds in self.sources
        ]
        self.steps = []
        self.members = []
        self.proof = None

    def stage_members(self):
        member = self.uow.stage_file(self.directory, f"members-{len(self.members)}.parquet")
        self.members.append(member)
        return member

    def record_step(self, index, input_columns, output_path):
        spec = self.plan["joins"][index]
        source = self.bindings[index + 1]
        mapping = DataBackend.feature_column_mapping(
            source["columns"], [key["feature_col"] for key in spec["keys"]], input_columns,
        )
        member = self.members[index]
        self.steps.append({
            "feature_dataset_id": source["dataset_id"],
            "feature_columns": mapping,
            "member_path": member.final_path.relative_to(self.root).as_posix(),
            "member_hash": sha256_file(member.path),
            "output_hash": sha256_file(output_path),
        })
        return [*input_columns, *mapping.values()]

    def finish(self, columns, output_path):
        _verify_copied_rows(
            self.registry, self.paths[0], output_path, self.sources,
            columns, [member.path for member in self.members],
        )
        self.payload = {
            "schema_version": "ordinary-join-evidence.v1", "assurance": "unknown",
            "reason": "ordinary_join_does_not_establish_point_in_time_availability",
            "contract": self.plan, "sources": self.bindings, "steps": self.steps,
            "output_hash": sha256_file(output_path), "output_columns": columns,
        }
        self.proof = self.uow.stage_file(self.directory, "evidence.json")
        raw = json.dumps(self.payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(raw.encode()) > 20_000_000:
            raise FeatureError("JOIN evidence exceeds its byte budget")
        self.proof.path.write_text(raw, encoding="utf-8")
        self.proof_hash = sha256_file(self.proof.path)

    def verify_sources(self, conn):
        for dataset, snapshot in zip(self.sources, self.paths, strict=True):
            if not dataset_identity_equal(self.registry._repo.get_dataset_on_connection(conn, dataset.id), dataset):
                raise FeatureError("JOIN source binding changed before commit")
            self.registry.resolve_verified_path(dataset.id)
            if sha256_file(snapshot) != dataset.content_hash:
                raise FeatureError("JOIN authenticated source snapshot changed")

    def register(self, conn, result):
        self.verify_sources(conn)
        if (result.content_hash != self.payload["output_hash"]
                or result.row_count != self.sources[0].row_count):
            raise FeatureError("JOIN output changed before evidence registration")
        repo = TaskArtifactRepository(self.registry._repo.db_path)
        for member, step in zip(self.members, self.steps, strict=True):
            if sha256_file(member.final_path) != step["member_hash"]:
                raise FeatureError("JOIN membership changed before registration")
            repo.register_on_connection(
                conn, task_id=result.task_id, kind=MEMBERS_KIND,
                path=step["member_path"], content_hash=step["member_hash"], origin_tool=ORIGIN,
                provenance={"output_dataset_id": result.id, "evidence_hash": self.proof_hash},
            )
        if sha256_file(self.proof.final_path) != self.proof_hash:
            raise FeatureError("JOIN evidence changed before registration")
        repo.register_on_connection(
            conn, task_id=result.task_id, kind=KIND,
            path=self.proof.final_path.relative_to(self.root).as_posix(),
            content_hash=self.proof_hash, origin_tool=ORIGIN,
            provenance={"output_dataset_id": result.id, "output_hash": result.content_hash},
        )
        self.verify_sources(conn)
        for member, step in zip(self.members, self.steps, strict=True):
            if sha256_file(member.final_path) != step["member_hash"]:
                raise FeatureError("JOIN membership changed during registration")
        if sha256_file(self.proof.final_path) != self.proof_hash:
            raise FeatureError("JOIN evidence changed during registration")


def _artifact_path(registry, record, task_id):
    relative = Path(record["path"])
    path = registry.datasets_root.parent / relative
    resolved = path.resolve(strict=True)
    allowed = registry.datasets_root / task_id
    if relative.is_absolute() or path != resolved or not resolved.is_relative_to(allowed):
        raise FeatureError("JOIN artifact path is unsafe")
    if sha256_file(resolved) != record["content_hash"]:
        raise FeatureError("JOIN artifact bytes changed")
    return resolved


def join_time_parent(registry, artifacts, dataset):
    """Return the authenticated left parent after validating actual copied fields.

    Missing legacy receipts stay unknown. A present invalid receipt is an error,
    not a reason to silently fall back to weaker provenance.
    """
    records = artifacts.list_for_task(dataset.task_id)
    matches = [r for r in records if r["kind"] == KIND and r["provenance"].get("output_dataset_id") == dataset.id]
    if not matches:
        return None
    if len(matches) != 1:
        raise FeatureError("ambiguous JOIN evidence")
    record = matches[0]
    if record["origin_tool"] != ORIGIN or record["provenance"] != {
        "output_dataset_id": dataset.id, "output_hash": dataset.content_hash,
    }:
        raise FeatureError("JOIN producer binding changed")
    path = _artifact_path(registry, record, dataset.task_id)
    with path.open("rb") as stream:
        raw = stream.read(20_000_001)
    if len(raw) > 20_000_000 or hashlib.sha256(raw).hexdigest() != record["content_hash"]:
        raise FeatureError("JOIN evidence bytes changed")
    try:
        proof = json.loads(raw)
        _verify_proof(registry, dataset, proof, record, records)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise FeatureError("JOIN evidence is invalid") from exc
    if sha256_file(path) != record["content_hash"]:
        raise FeatureError("JOIN evidence changed during verification")
    return proof["sources"][0]["dataset_id"], proof["sources"][0]["columns"], record["id"]


def _verify_proof(registry, dataset, proof, record, records):
    if (proof["schema_version"] != "ordinary-join-evidence.v1"
            or proof["assurance"] != "unknown"
            or proof["contract"]["task_id"] != dataset.task_id
            or proof["output_hash"] != dataset.content_hash
            or not proof["steps"]
            or len(proof["sources"]) != len(proof["steps"]) + 1
            or len(proof["contract"]["joins"]) != len(proof["steps"])):
        raise FeatureError("JOIN evidence contract differs from output")
    sources, source_paths = [], []
    for expected in proof["sources"]:
        source = registry.get(expected["dataset_id"])
        if source.task_id != dataset.task_id or _binding(
            source, registry.authenticated_parquet_column_names(source.id)
        ) != expected:
            raise FeatureError("JOIN source binding changed")
        sources.append(source)
        source_paths.append(stable_dataset_path(registry, source))
    if proof["contract"]["anchor_dataset_id"] != sources[0].id:
        raise FeatureError("JOIN anchor binding changed")
    columns = proof["sources"][0]["columns"]
    members = []
    for index, step in enumerate(proof["steps"]):
        spec = proof["contract"]["joins"][index]
        if (not spec["confirmed"] or spec["feature_dataset_id"] != sources[index + 1].id
                or step["feature_dataset_id"] != sources[index + 1].id):
            raise FeatureError("JOIN feature binding changed")
        expected = DataBackend.feature_column_mapping(
            proof["sources"][index + 1]["columns"],
            [key["feature_col"] for key in spec["keys"]], columns,
        )
        if step["feature_columns"] != expected:
            raise FeatureError("JOIN column mapping changed")
        columns = [*columns, *expected.values()]
        matches = [r for r in records if r["kind"] == MEMBERS_KIND and r["path"] == step["member_path"]]
        if len(matches) != 1:
            raise FeatureError("JOIN membership artifact is missing or ambiguous")
        member = matches[0]
        if (member["origin_tool"] != ORIGIN or member["content_hash"] != step["member_hash"]
                or member["provenance"] != {"output_dataset_id": dataset.id, "evidence_hash": record["content_hash"]}):
            raise FeatureError("JOIN membership binding changed")
        members.append(_artifact_path(registry, member, dataset.task_id))
    if (columns != proof["output_columns"]
            or columns != list(registry.authenticated_parquet_column_names(dataset.id))
            or proof["steps"][-1]["output_hash"] != dataset.content_hash):
        raise FeatureError("JOIN output column mapping changed")
    result_path = stable_dataset_path(registry, dataset)
    _verify_copied_rows(registry, source_paths[0], result_path, sources, columns, members)
    # Source metadata or a file swap during verification cannot certify the read.
    for source in [*sources, dataset]:
        if not dataset_identity_equal(registry.get(source.id), source):
            raise FeatureError("JOIN binding changed during verification")
        registry.resolve_verified_path(source.id)
    for step, member in zip(proof["steps"], members, strict=True):
        if sha256_file(member) != step["member_hash"]:
            raise FeatureError("JOIN membership changed during verification")


def _verify_copied_rows(registry, anchor_path, result_path, sources, columns, members):
    """Check row positions and copied left values in spillable SQL, not pandas RAM."""
    backend = DataBackend(registry.datasets_root)
    ordinal = "__marvis_verify_row"
    while ordinal in columns:
        ordinal += "_"
    with tempfile.TemporaryDirectory(prefix="join-verify-", dir=registry.datasets_root) as temp:
        temp = Path(temp)
        left = indexed_parquet_relation(anchor_path, ordinal, temp / "left.parquet")
        result = indexed_parquet_relation(result_path, ordinal, temp / "result.parquet")
        count = sources[0].row_count
        with backend._connect() as conn:
            for index, member in enumerate(members):
                relation = f"read_parquet({sql_string_literal(member.as_posix())})"
                schema = [(r[0], r[1]) for r in conn.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()]
                if schema != [("output_row", "BIGINT"), ("left_row", "BIGINT"), ("right_rows", "BIGINT[]")]:
                    raise FeatureError("JOIN membership schema changed")
                invalid = conn.execute(
                    f"SELECT count(*) != ? OR count(DISTINCT output_row) != ? OR "
                    f"count(DISTINCT left_row) != ? OR coalesce(bool_or(output_row < 0 OR output_row >= ? "
                    f"OR left_row < 0 OR left_row >= ? OR output_row IS NULL OR left_row IS NULL), false) FROM {relation}",
                    [count] * 5,
                ).fetchone()[0]
                # Unmatched rows are NULL; an empty member list is never a match.
                invalid_right = conn.execute(
                    f"SELECT EXISTS (SELECT 1 FROM {relation} WHERE len(right_rows)=0) OR EXISTS "
                    f"(SELECT 1 FROM {relation}, UNNEST(right_rows) t(n) WHERE n IS NULL OR n<0 OR n>=?)",
                    [sources[index + 1].row_count],
                ).fetchone()[0]
                if invalid or invalid_right:
                    raise FeatureError("JOIN membership positions changed")
            # Compose each step's actual previous-left row, rather than assume
            # DuckDB's join order or infer identity from a business key.
            chain = f"read_parquet({sql_string_literal(members[-1].as_posix())}) m{len(members)-1}"
            for i in range(len(members) - 2, -1, -1):
                chain += f" JOIN read_parquet({sql_string_literal(members[i].as_posix())}) m{i} ON m{i}.output_row=m{i+1}.left_row"
            def quote(value):
                return '"' + value.replace('"', '""') + '"'
            different = " OR ".join(
                f"a.{quote(name)} IS DISTINCT FROM z.{quote(name)}"
                for name in registry.authenticated_parquet_column_names(sources[0].id)
            )
            matched, bad = conn.execute(
                f"SELECT count(*), coalesce(bool_or({different}), false) FROM {chain} "
                f"JOIN {left} a ON a.{ordinal}=m0.left_row "
                f"JOIN {result} z ON z.{ordinal}=m{len(members)-1}.output_row"
            ).fetchone()
            if bad or matched != count:
                raise FeatureError("JOIN copied left values differ from their native members")
