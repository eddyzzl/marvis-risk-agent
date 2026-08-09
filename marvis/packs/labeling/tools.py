"""Labeling pack tool entrypoints (C1 标签构造与成熟度工具).

三个工具：

- ``define_label``：从 DPD 长表构造 0/1 坏标签，落一个带 target 的衍生数据集 +
  定坏口径元数据。成熟度确认门内置——表现期未闭合的 cohort 默认阻断，须显式确认。
- ``check_cohort_maturity``：只读，按 vintage cohort 报告表现期是否闭合。
- ``suggest_bad_definition``：纯桥接，从既有 roll_rate_matrix 输出推定坏口径建议。
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from copy import deepcopy
import hmac
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any
from urllib.parse import quote
import uuid

from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.authenticated_snapshot import (
    AuthenticatedSnapshotError,
    read_authenticated_parquet_snapshot,
)
from marvis.data.dataset_export import export_dataset
from marvis.data.errors import (
    CohortMaturityNotConfirmedError,
    DatasetContentDriftError,
)
from marvis.data.label_construction import (
    check_cohort_maturity,
    construct_label,
    suggest_bad_definition,
)
from marvis.files import sha256_file
from marvis.packs.labeling.contracts import (
    LabelingContractError,
    LabelingRequest,
    build_authenticated_labeling_source,
    frame_at_as_of,
)
from marvis.plugins.sdk import PackRuntime
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository


_TOOL_RESULT_SCHEMA_VERSION = "labeling-tool-result.v1"
_EVIDENCE_SCHEMA_VERSION = "labeling-quality-evidence.v1"
_DATASET_ARTIFACT_KIND = "labeling_dataset_csv"
_EVIDENCE_ARTIFACT_KIND = "labeling_quality_evidence_json"
_ORIGIN_TOOL = "labeling.define_label"
_PRODUCER_VERSION = "labeling.define_label.v1"


def tool_define_label(inputs: dict, ctx) -> dict:
    """从 DPD 长表构造 0/1 坏标签，落衍生数据集 + 定坏口径元数据。

    成熟度确认门：给了 ``cohort_col`` 时先按 vintage 判定表现期是否闭合到定坏 MOB；
    有未成熟 cohort 且未 ``confirm_immature_cohorts`` -> 抛 CohortMaturityNotConfirmedError。
    """
    runtime = _runtime(ctx)
    task_id = str(ctx.task_id)
    request = _labeling_request(inputs)
    proposal_hash = str(inputs.get("proposal_hash") or "").strip().lower()
    if not hmac.compare_digest(proposal_hash, request.contract_hash):
        raise LabelingContractError(
            "proposal_hash does not match the explicit labeling contract"
        )
    workspace = runtime.workspaces.get_or_default(task_id)
    authenticated_source = build_authenticated_labeling_source(
        runtime.registry,
        runtime.backend,
        workspace,
        task_id=task_id,
        request=request,
    )
    proposal = authenticated_source.proposal
    dataset = runtime.registry.get(request.dataset_id)
    source_path = authenticated_source.source_path
    source_frame = authenticated_source.frame
    frame, _excluded = frame_at_as_of(
        source_frame,
        date_col=request.date_col,
        as_of_date=request.as_of_date,
    )

    # 成熟度确认门（cohort_col 给定时）：表现期未闭合的 cohort 不静默纳入。
    maturity_payload = dict(proposal.maturity)
    immature_cohorts = list(maturity_payload["immature_cohorts"])
    if immature_cohorts and not bool(inputs.get("confirm_immature_cohorts")):
        raise CohortMaturityNotConfirmedError(
            required_mob=int(maturity_payload["required_mob"]),
            immature_cohorts=immature_cohorts,
            cohort_diagnostics=list(maturity_payload["cohorts"]),
        )

    result = construct_label(
        frame,
        id_col=request.id_col,
        mob_col=request.mob_col,
        observation_window=request.observation_window,
        performance_window=request.performance_window,
        dpd_col=request.dpd_col,
        threshold_dpd=request.threshold_dpd,
        status_col=request.status_col,
        threshold_status=request.threshold_status,
        states=request.states,
        at_mob=request.at_mob,
        cohort_col=request.cohort_col,
        target_col=request.target_col,
    )
    red_flags: list[dict] = []
    if result.n_unmatured:
        red_flags.append({
            "code": "unmatured_loans",
            "level": "amber",
            "message": (
                f"{result.n_unmatured}/{result.n_loans} 笔贷款表现期未闭合到 mob"
                f"{result.definition.at_mob}，标签为 NaN（下游 NaN 标签门决定丢弃或补数据）。"
            ),
        })
    if maturity_payload["immature_cohorts"]:
        red_flags.append({
            "code": "immature_cohorts_included",
            "level": "amber",
            "message": (
                f"已按确认纳入 {len(maturity_payload['immature_cohorts'])} 个未成熟 cohort，"
                f"坏率可能被低估。"
            ),
        })

    return _publish_label_result(
        runtime,
        ctx,
        request=request,
        proposal=proposal,
        source_dataset=dataset,
        source_path=source_path,
        registered_source_path=authenticated_source.registered_source_path,
        result=result,
        maturity=maturity_payload,
        red_flags=red_flags,
    )


def tool_check_cohort_maturity(inputs: dict, ctx) -> dict:
    """只读成熟度检查：按 vintage cohort 报告表现期是否闭合到定坏 MOB。"""
    runtime = _runtime(ctx)
    task_id = str(ctx.task_id)
    dataset_id = str(inputs["dataset_id"])
    id_col = str(inputs["id_col"])
    mob_col = str(inputs["mob_col"])
    cohort_col = str(inputs["cohort_col"])
    date_col = str(inputs["date_col"])
    as_of_date = str(inputs["as_of_date"])
    expected_content_hash = str(inputs["expected_content_hash"])
    workspace = runtime.workspaces.get_or_default(task_id)
    _require_workspace_values(
        workspace,
        dataset_id=dataset_id,
        expected_content_hash=expected_content_hash,
        workspace_revision=int(inputs["workspace_revision"]),
        analysis_generation=int(inputs["analysis_generation"]),
    )
    columns = _unique([id_col, mob_col, cohort_col, date_col])
    dataset = runtime.registry.get(dataset_id)
    if str(dataset.task_id) != task_id:
        raise LabelingContractError("dataset_id is not owned by this task")
    if not hmac.compare_digest(
        str(dataset.content_hash or ""),
        expected_content_hash,
    ):
        raise LabelingContractError("expected_content_hash does not match the dataset")
    try:
        frame = runtime.registry.read_authenticated_parquet_snapshot(
            dataset.id,
            columns=columns,
        )
    except DatasetContentDriftError as exc:
        raise LabelingContractError(
            f"labeling source authenticated snapshot failed: {exc.reason}"
        ) from exc
    frame, _excluded = frame_at_as_of(
        frame,
        date_col=date_col,
        as_of_date=as_of_date,
    )

    # required_mob 优先取显式值；否则由 obs+perf 推出（与 define_label 口径一致）。
    required_mob = int(inputs["required_mob"])

    report = check_cohort_maturity(
        frame,
        id_col=id_col,
        mob_col=mob_col,
        cohort_col=cohort_col,
        required_mob=required_mob,
    )
    payload = _maturity_to_payload(report)
    if report.immature_cohorts:
        payload["red_flags"] = [{
            "code": "immature_cohorts",
            "level": "amber",
            "message": (
                f"{len(report.immature_cohorts)} 个 cohort 表现期未闭合到 mob{required_mob}："
                f"{', '.join(report.immature_cohorts[:5])}；纳入建模将低估坏率。"
            ),
        }]
    else:
        payload["red_flags"] = []
    return payload


def tool_suggest_bad_definition(inputs: dict, ctx) -> dict:
    """从既有 roll_rate_matrix 输出推定坏口径建议（纯桥接，不读数据集）。"""
    states = [str(state) for state in inputs["states"]]
    matrix = [[float(value) for value in row] for row in inputs["matrix"]]
    at_mob = int(inputs["at_mob"])
    threshold = inputs.get("roll_back_threshold")
    suggestion = suggest_bad_definition(
        states=states,
        matrix=matrix,
        at_mob=at_mob,
        roll_back_threshold=float(threshold) if threshold is not None else 0.10,
    )
    if suggestion is None:
        return {
            "suggestion": None,
            "message": (
                "在给定 roll_rate 矩阵与回滚率阈值下，没有回滚率足够低的逾期桶可作稳定定坏点；"
                "请手动指定定坏口径或放宽阈值。"
            ),
        }
    return {"suggestion": suggestion.to_dict(), "message": suggestion.rationale}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _maturity_to_payload(report) -> dict:
    return {
        "required_mob": report.required_mob,
        "cohorts": [_jsonable(cohort) for cohort in report.cohorts],
        "immature_cohorts": list(report.immature_cohorts),
        "all_matured": report.all_matured,
    }


def _labeling_request(inputs: dict) -> LabelingRequest:
    return LabelingRequest(
        dataset_id=inputs.get("dataset_id"),
        expected_content_hash=inputs.get("expected_content_hash"),
        workspace_revision=inputs.get("workspace_revision"),
        analysis_generation=inputs.get("analysis_generation"),
        id_col=inputs.get("id_col"),
        mob_col=inputs.get("mob_col"),
        cohort_col=inputs.get("cohort_col"),
        date_col=inputs.get("date_col"),
        as_of_date=inputs.get("as_of_date"),
        target_col=inputs.get("target_col"),
        observation_window=inputs.get("observation_window"),
        performance_window=inputs.get("performance_window"),
        at_mob=inputs.get("at_mob"),
        rule_kind=inputs.get("rule_kind"),
        dpd_col=inputs.get("dpd_col"),
        threshold_dpd=inputs.get("threshold_dpd"),
        status_col=inputs.get("status_col"),
        threshold_status=inputs.get("threshold_status"),
        states=inputs.get("states"),
    )


def _publish_label_result(
    runtime,
    ctx,
    *,
    request: LabelingRequest,
    proposal,
    source_dataset,
    source_path: Path,
    registered_source_path: str,
    result,
    maturity: dict,
    red_flags: list[dict],
) -> dict:
    task_id = str(ctx.task_id)
    attempt = uuid.uuid4().hex[:12]
    dataset_dir = runtime.datasets_root / task_id / "labeling"
    artifact_dir = Path(runtime.settings.tasks_dir) / task_id / "labeling"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    uow = ArtifactUnitOfWork()
    parquet_artifact = uow.stage_file(
        dataset_dir,
        f"{source_dataset.id}_{request.contract_hash[:12]}_{attempt}.parquet",
    )
    csv_artifact = uow.stage_file(
        artifact_dir,
        f"labels_{request.contract_hash[:12]}_{attempt}.csv",
    )
    evidence_artifact = uow.stage_file(
        artifact_dir,
        f"labeling_evidence_{request.contract_hash[:12]}_{attempt}.json",
    )
    try:
        result.frame.to_parquet(parquet_artifact.path, index=False)
        with tempfile.TemporaryDirectory(
            prefix=".labeling-export-",
            dir=artifact_dir,
        ) as temp_dir:
            export_path = Path(temp_dir) / "labels.csv"
            export_evidence = export_dataset(
                parquet_artifact.path,
                export_path,
                format="csv",
                temp_directory=Path(temp_dir),
                text_columns=(request.id_col, request.cohort_col),
            )
            shutil.copyfile(export_path, csv_artifact.path)
        evidence_artifact.path.write_text("{}", encoding="utf-8")
        parquet_hash = sha256_file(parquet_artifact.path)
        csv_hash = sha256_file(csv_artifact.path)

        def _commit(conn):
            conn.execute("BEGIN IMMEDIATE")
            _require_source_and_workspace_on_connection(
                conn,
                task_id=task_id,
                request=request,
                source_path=source_path,
                registered_source_path=registered_source_path,
                datasets_root=runtime.datasets_root,
            )
            if not hmac.compare_digest(
                sha256_file(parquet_artifact.final_path),
                parquet_hash,
            ):
                raise LabelingContractError(
                    "published labeled dataset hash changed before registration"
                )
            if not hmac.compare_digest(
                sha256_file(csv_artifact.final_path),
                csv_hash,
            ):
                raise LabelingContractError(
                    "published label export hash changed before registration"
                )
            registered = runtime.registry.register_existing_on_connection(
                conn,
                parquet_artifact.final_path,
                task_id=task_id,
                role="derived",
                anchor_target=source_dataset.id,
                target_col_override=request.target_col,
                seed=int(ctx.seed or 0),
            )
            if not hmac.compare_digest(
                str(registered.content_hash or ""),
                parquet_hash,
            ):
                raise LabelingContractError(
                    "registered labeled dataset hash does not match the published file"
                )
            quality = _quality_payload(result)
            lineage = {
                "parent_dataset_id": source_dataset.id,
                "child_dataset_id": registered.id,
                "relation_kind": "label_construction",
                "edge_order": 0,
            }
            workspace_payload = {
                "revision": request.workspace_revision,
                "analysis_generation": request.analysis_generation,
                "active_dataset_id": source_dataset.id,
                "active_dataset_content_hash": request.expected_content_hash,
                "active_dataset_changed": False,
            }
            safe_export = deepcopy(export_evidence)
            output = safe_export.get("output")
            if isinstance(output, dict):
                output.pop("path", None)
            evidence = {
                "schema_version": _EVIDENCE_SCHEMA_VERSION,
                "producer_version": _PRODUCER_VERSION,
                "proposal_hash": request.contract_hash,
                "source": {
                    "dataset_id": source_dataset.id,
                    "content_hash": request.expected_content_hash,
                    "row_count": proposal.source_row_count,
                    "rows_at_as_of": proposal.rows_at_as_of,
                    "rows_excluded_after_as_of": proposal.rows_excluded_after_as_of,
                    "date_col": request.date_col,
                    "as_of_date": request.as_of_date,
                },
                "result": {
                    "dataset_id": registered.id,
                    "content_hash": registered.content_hash,
                    "target_col": request.target_col,
                    "row_count": registered.row_count,
                },
                "contract": request.to_dict(),
                "bad_definition": result.definition.to_dict(),
                "quality": quality,
                "maturity": maturity,
                "red_flags": red_flags,
                "lineage": lineage,
                "workspace": workspace_payload,
                "csv_export": safe_export,
            }
            evidence_json = json.dumps(
                evidence,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            evidence_artifact.final_path.write_text(evidence_json, encoding="utf-8")
            evidence_hash = sha256_file(evidence_artifact.final_path)
            dataset_record = runtime.task_artifacts.register_on_connection(
                conn,
                task_id=task_id,
                kind=_DATASET_ARTIFACT_KIND,
                path=str(csv_artifact.final_path),
                content_hash=csv_hash,
                origin_tool=_ORIGIN_TOOL,
                provenance={
                    "schema_version": "labeling-dataset-export.v1",
                    "producer_version": _PRODUCER_VERSION,
                    "proposal_hash": request.contract_hash,
                    "source_dataset_id": source_dataset.id,
                    "source_content_hash": request.expected_content_hash,
                    "result_dataset_id": registered.id,
                    "result_content_hash": registered.content_hash,
                    "target_col": request.target_col,
                },
            )
            evidence_record = runtime.task_artifacts.register_on_connection(
                conn,
                task_id=task_id,
                kind=_EVIDENCE_ARTIFACT_KIND,
                path=str(evidence_artifact.final_path),
                content_hash=evidence_hash,
                origin_tool=_ORIGIN_TOOL,
                provenance={
                    "schema_version": _EVIDENCE_SCHEMA_VERSION,
                    "producer_version": _PRODUCER_VERSION,
                    "proposal_hash": request.contract_hash,
                    "source_dataset_id": source_dataset.id,
                    "source_content_hash": request.expected_content_hash,
                    "result_dataset_id": registered.id,
                    "result_content_hash": registered.content_hash,
                },
            )
            runtime.repo.write_audit_on_connection(
                conn,
                kind="labeling.dataset.created",
                target_ref=registered.id,
                actor="agent:labeling",
                inputs_hash=request.contract_hash,
                outcome="succeeded",
                detail={
                    "task_id": task_id,
                    "source_dataset_id": source_dataset.id,
                    "source_content_hash": request.expected_content_hash,
                    "result_dataset_id": registered.id,
                    "result_content_hash": registered.content_hash,
                    "target_col": request.target_col,
                    "dataset_artifact_id": dataset_record["id"],
                    "evidence_artifact_id": evidence_record["id"],
                    "active_dataset_changed": False,
                },
            )
            return registered, dataset_record, evidence_record, evidence, evidence_hash

        (
            registered,
            dataset_record,
            evidence_record,
            evidence,
            evidence_hash,
        ) = uow.finalize_with_connection(runtime.repo.transaction, _commit)
    except Exception:
        uow.rollback()
        raise

    quality = dict(evidence["quality"])
    return {
        "schema_version": _TOOL_RESULT_SCHEMA_VERSION,
        "source_dataset_id": source_dataset.id,
        "source_content_hash": request.expected_content_hash,
        "result_dataset_id": registered.id,
        "result_content_hash": registered.content_hash,
        "target_col": result.target_col,
        "bad_definition": result.definition.to_dict(),
        "n_loans": result.n_loans,
        "n_bad": result.n_bad,
        "n_good": result.n_good,
        "n_unmatured": result.n_unmatured,
        "bad_rate": quality["bad_rate"],
        "maturity": maturity,
        "quality": quality,
        "proposal_hash": request.contract_hash,
        "lineage": dict(evidence["lineage"]),
        "workspace": dict(evidence["workspace"]),
        "dataset_artifact_id": str(dataset_record["id"]),
        "dataset_content_hash": csv_hash,
        "dataset_download_url": _task_artifact_download_url(
            task_id,
            str(dataset_record["id"]),
            csv_hash,
        ),
        "evidence_artifact_id": str(evidence_record["id"]),
        "evidence_content_hash": evidence_hash,
        "evidence_download_url": _task_artifact_download_url(
            task_id,
            str(evidence_record["id"]),
            evidence_hash,
        ),
        "red_flags": red_flags,
    }


class _Runtime(PackRuntime):
    """Labeling pack repositories sharing the task SQLite unit of work."""

    def _extend(self, ctx) -> None:
        self.workspaces = DataWorkspaceRepository(self.settings.db_path)
        self.task_artifacts = TaskArtifactRepository(self.settings.db_path)


def _runtime(ctx) -> _Runtime:
    return _Runtime(ctx)


def _jsonable(value: Any):
    if value is None:
        return None
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _quality_payload(result) -> dict:
    matured = int(result.n_bad) + int(result.n_good)
    return {
        "n_loans": int(result.n_loans),
        "n_bad": int(result.n_bad),
        "n_good": int(result.n_good),
        "n_unmatured": int(result.n_unmatured),
        "label_coverage": _safe_ratio(matured, int(result.n_loans)),
        "bad_rate": _safe_ratio(int(result.n_bad), matured),
    }


def _require_workspace_values(
    workspace,
    *,
    dataset_id: str,
    expected_content_hash: str,
    workspace_revision: int,
    analysis_generation: int,
) -> None:
    expected = {
        "active_dataset_id": str(dataset_id),
        "active_dataset_content_hash": str(expected_content_hash),
        "revision": int(workspace_revision),
        "analysis_generation": int(analysis_generation),
    }
    for field_name, value in expected.items():
        if getattr(workspace, field_name, None) != value:
            raise LabelingContractError(
                f"data workspace {field_name} changed before label execution"
            )


def _require_source_and_workspace_on_connection(
    conn,
    *,
    task_id: str,
    request: LabelingRequest,
    source_path: Path,
    registered_source_path: str,
    datasets_root: Path,
) -> None:
    dataset_row = conn.execute(
        """
        SELECT task_id, content_hash, source_path
          FROM datasets
         WHERE id = ?
        """,
        (request.dataset_id,),
    ).fetchone()
    if dataset_row is None or str(dataset_row["task_id"]) != task_id:
        raise LabelingContractError("source dataset ownership changed before execution")
    if not hmac.compare_digest(
        str(dataset_row["content_hash"] or ""),
        request.expected_content_hash,
    ):
        raise LabelingContractError("source dataset hash changed before execution")
    if str(dataset_row["source_path"] or "") != registered_source_path:
        raise LabelingContractError("source dataset path changed before execution")
    try:
        read_authenticated_parquet_snapshot(
            source_path,
            root=datasets_root,
            expected_sha256=request.expected_content_hash,
            columns=[],
        )
    except AuthenticatedSnapshotError as exc:
        raise LabelingContractError(
            "source dataset authenticated bytes changed before execution: "
            f"{exc.reason.value}"
        ) from exc
    workspace_row = conn.execute(
        """
        SELECT active_dataset_id, active_dataset_content_hash, revision,
               analysis_generation
          FROM data_workspaces
         WHERE task_id = ?
        """,
        (task_id,),
    ).fetchone()
    if workspace_row is None:
        raise LabelingContractError("active data workspace disappeared before execution")
    expected = {
        "active_dataset_id": request.dataset_id,
        "active_dataset_content_hash": request.expected_content_hash,
        "revision": request.workspace_revision,
        "analysis_generation": request.analysis_generation,
    }
    for field_name, value in expected.items():
        if workspace_row[field_name] != value:
            raise LabelingContractError(
                f"data workspace {field_name} changed before label publication"
            )


def _task_artifact_download_url(
    task_id: str,
    artifact_id: str,
    content_hash: str,
) -> str:
    return (
        f"/api/tasks/{quote(task_id, safe='')}/task-artifacts/"
        f"{quote(artifact_id, safe='')}/download"
        f"?expected_content_hash={quote(content_hash, safe='')}"
    )


def _unique(values: list[str | None]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out
