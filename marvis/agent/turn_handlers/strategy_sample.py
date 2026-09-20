"""Strategy sample for governed Agent turns."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from marvis.agent.strategy_request_compiler import CompiledStrategyRequestDraft
from marvis.agent.strategy_request_compiler import StandardWorkflowRequestDraft
from marvis.agent.strategy_setup import StrategySetupError
from marvis.agent.strategy_setup import build_strategy_dataset_context
from marvis.agent.strategy_setup import preview_strategy_dataset_context
from marvis.agent.strategy_workflows._foundation_delivery import (
    select_sample_design_v2_template,
)
from marvis.data.backend import DataBackend
from marvis.data.errors import DatasetContentDriftError
from marvis.data.registry import DatasetRegistry
from marvis.data.workspace import DataSemanticMapping
from marvis.data.workspace import DataWorkspaceDraft
from marvis.data.workspace import data_semantic_mapping_hash
from marvis.domain import TaskRecord
from marvis.files import sha256_file
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.errors import StrategySampleDesignScopeIneligibleError
from marvis.packs.strategy.sample_design_binding import (
    load_strategy_sample_design_execution_binding,
)
from marvis.packs.strategy.sample_design_execution import (
    load_strategy_risk_development_execution_binding,
)
from marvis.packs.strategy.sample_design_tools import SAMPLE_DESIGN_ARTIFACT_KIND
from marvis.packs.strategy.sample_design_tools import SAMPLE_DESIGN_ORIGIN_TOOL
from marvis.packs.strategy.sample_design_v2_native_tools import (
    SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL,
)
from marvis.packs.strategy.sample_design_v2_native_tools import (
    authenticate_native_strategy_sample_design_v2_bundle_record,
)
from marvis.packs.strategy.sample_design_v2_tools import (
    SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
)
from marvis.packs.strategy.sample_design_v2_tools import SAMPLE_DESIGN_V2_ORIGIN_TOOL
from marvis.packs.strategy.sample_design_v2_tools import (
    load_any_strategy_sample_design_v2_artifacts,
)
from marvis.repositories.data_workspace import DataWorkspaceDataError
from marvis.repositories.data_workspace import DataWorkspaceDatasetNotFound
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.data_workspace import DataWorkspaceRevisionConflict
from marvis.repositories.datasets import DatasetRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.repositories.tasks import TaskRepository
from pathlib import Path
from types import SimpleNamespace
import json
import math
from . import contracts as contracts_lane
from . import data_context as data_context_lane
from . import responses as responses_lane
from . import strategy_contracts as strategy_contracts_lane
from . import strategy_evidence as strategy_evidence_lane


def _strategy_sample_design_plan_slots(
    runtime: contracts_lane.DriverTurnRuntime,
    task: TaskRecord,
    draft: StandardWorkflowRequestDraft,
    *,
    context,
    drop_nan_labels: bool,
) -> dict[str, object]:
    """Bind user-owned sample facts to the exact active workspace snapshot."""

    try:
        workspace = _require_strategy_sample_design_workspace(runtime, task)
    except StrategySetupError:
        raise
    if (
        workspace.active_dataset_id != context.dataset_id
        or workspace.active_dataset_content_hash != context.dataset_content_hash
        or workspace.revision != context.workspace_revision
        or workspace.analysis_generation != context.analysis_generation
    ):
        raise StrategySetupError(
            "活动 DataWorkspace 在样本设计计划创建前发生变化；请基于当前版本重试。"
        )
    target_col = workspace.semantic_mapping.target_col
    if (
        not isinstance(target_col, str)
        or not target_col
        or target_col != context.target_col
        or target_col not in context.columns
    ):
        raise StrategySetupError(
            "策略样本设计只能使用 DataWorkspace 中已确认的二元目标列。"
        )
    content_hash = workspace.active_dataset_content_hash
    semantic_hash = data_semantic_mapping_hash(workspace.semantic_mapping)
    if not isinstance(content_hash, str) or not content_hash:
        raise StrategySetupError("活动数据集缺少内容 hash，不能固化样本设计。")
    if semantic_hash != context.semantic_mapping_hash:
        raise StrategySetupError(
            "活动 DataWorkspace 的语义映射已变化；请重新发起样本设计。"
        )

    inputs = draft.to_dict()["workflow_inputs"]
    for field in (
        "split_col",
        "month_col",
        "weight_col",
        "loan_amount_col",
        "overdue_amount_col",
    ):
        column = inputs.get(field)
        if column is not None and column not in context.columns:
            raise StrategySetupError(
                f"策略样本设计显式字段 {field} 不在当前活动数据集中。"
            )
    slots: dict[str, object] = {
        "dataset_id": workspace.active_dataset_id,
        "expected_dataset_content_hash": content_hash,
        "workspace_revision": workspace.revision,
        "workspace_generation": workspace.analysis_generation,
        "semantic_mapping_hash": semantic_hash,
        "target_col": target_col,
        "drop_nan_labels": bool(drop_nan_labels),
    }
    slots.update(
        {key: value for key, value in inputs.items() if key != "drop_nan_labels"}
    )
    return slots


def _strategy_sample_design_v2_plan_slots(
    runtime: contracts_lane.DriverTurnRuntime,
    task: TaskRecord,
    draft: StandardWorkflowRequestDraft,
    *,
    context,
    drop_nan_labels: bool,
) -> dict[str, object]:
    """Bind platform context, adding compatibility slots only when lossless."""

    workspace = _require_strategy_sample_design_workspace(runtime, task)
    if (
        workspace.active_dataset_id != context.dataset_id
        or workspace.active_dataset_content_hash != context.dataset_content_hash
        or workspace.revision != context.workspace_revision
        or workspace.analysis_generation != context.analysis_generation
    ):
        raise StrategySetupError(
            "活动 DataWorkspace 在 V2 样本设计计划创建前发生变化；请基于当前版本重试。"
        )
    target_col = workspace.semantic_mapping.target_col
    semantic_hash = data_semantic_mapping_hash(workspace.semantic_mapping)
    if (
        not isinstance(target_col, str)
        or not target_col
        or target_col != context.target_col
        or target_col not in context.columns
    ):
        raise StrategySetupError(
            "V2 策略样本设计只能使用 DataWorkspace 中已确认的二元目标列。"
        )
    if (
        not isinstance(workspace.active_dataset_content_hash, str)
        or not workspace.active_dataset_content_hash
        or semantic_hash != context.semantic_mapping_hash
    ):
        raise StrategySetupError(
            "活动 DataWorkspace 的数据 hash 或语义映射已变化；请重新发起 V2 样本设计。"
        )

    inputs = draft.to_dict()["workflow_inputs"]
    fields = inputs["field_bindings"]
    maturity = inputs["maturity"]
    performance = inputs["performance_window"]
    observation = inputs["observation_window"]
    scope = (
        "strategy_development"
        if maturity["status"] == "confirmed_matured"
        and performance["status"] == "provided"
        and observation["status"] == "provided"
        else "exploration_only"
    )
    compatibility_maturity = (
        "unknown" if maturity["status"] == "unavailable" else maturity["status"]
    )
    policy = {
        **strategy_contracts_lane._STRATEGY_SAMPLE_V2_POLICY,
        "diagnostic_severities": dict(
            strategy_contracts_lane._STRATEGY_SAMPLE_V2_POLICY["diagnostic_severities"]
        ),
    }
    slots: dict[str, object] = {
        "dataset_id": workspace.active_dataset_id,
        "expected_dataset_content_hash": workspace.active_dataset_content_hash,
        "workspace_revision": workspace.revision,
        "workspace_generation": workspace.analysis_generation,
        "semantic_mapping_hash": semantic_hash,
        "target_col": target_col,
        "relationship": inputs["relationship"],
        "scope": scope,
        "policy": policy,
        "target_bad_value": inputs["target_bad_value"],
        "drop_nan_labels": bool(drop_nan_labels),
        "approval_population": inputs["approval_population"],
        "risk_population": inputs["risk_population"],
        "partitioning": inputs["partitioning"],
        "maturity": maturity,
        "performance_window": performance,
        "observation_window": observation,
        "field_bindings": fields,
        "historical_score": inputs["historical_score"],
    }
    if select_sample_design_v2_template(inputs) == "strategy_sample_design_v2_native":
        return slots

    split_col, split_values = _strategy_sample_v2_simple_split_projection(
        inputs["partitioning"]
    )
    compatibility_columns = [
        fields.get("month_field"),
        fields.get("weight_field"),
        fields.get("loan_amount_field"),
        fields.get("overdue_amount_field"),
    ]
    present_columns = [
        str(column) for column in compatibility_columns if column is not None
    ]
    if (
        split_col == target_col
        or split_col not in context.columns
        or any(column not in context.columns for column in present_columns)
        or target_col in present_columns
        or split_col in present_columns
        or len(present_columns) != len(set(present_columns))
    ):
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_sample_design_v2_native_source_unsupported",
            "当前 V2 字段绑定无法由活动数据集安全执行；请调整重复或冲突字段。",
        )
    slots.update(
        {
            "compatibility_performance_window_status": performance["status"],
            "compatibility_performance_window_days": performance["days"],
            "compatibility_observation_window_status": observation["status"],
            "compatibility_observation_start": observation["start"],
            "compatibility_observation_end": observation["end"],
            "compatibility_maturity_status": compatibility_maturity,
            "compatibility_split_col": split_col,
            "compatibility_development_values": [split_values["development"]],
            "compatibility_validation_values": [split_values["validation"]],
            "compatibility_oot_values": [split_values["oot"]],
            "compatibility_month_col": fields.get("month_field"),
            "compatibility_weight_col": fields.get("weight_field"),
            "compatibility_loan_amount_col": fields.get("loan_amount_field"),
            "compatibility_overdue_amount_col": fields.get("overdue_amount_field"),
        }
    )
    return slots


def _strategy_sample_v2_simple_split_projection(
    partitioning: object,
) -> tuple[str, dict[str, object]]:
    if (
        not isinstance(partitioning, Mapping)
        or set(partitioning) != {"method", "selectors"}
        or partitioning.get("method") != "predicate_ast"
        or not isinstance(partitioning.get("selectors"), Mapping)
    ):
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_sample_design_v2_native_bootstrap_required",
            "当前 compatibility anchor 只支持同一列上的三组简单等值切分；"
            "本次未创建计划。",
        )
    selectors = partitioning["selectors"]
    if set(selectors) != {"development", "validation", "oot"}:
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_sample_design_v2_native_bootstrap_required",
            "V2 partitioning 必须完整包含 development、validation 和 OOT。",
        )
    columns: list[str] = []
    values: dict[str, object] = {}
    for partition in ("development", "validation", "oot"):
        predicate = selectors[partition]
        if (
            not isinstance(predicate, Mapping)
            or set(predicate) != {"op", "left", "right"}
            or predicate.get("op") != "eq"
            or not isinstance(predicate.get("left"), Mapping)
            or set(predicate["left"]) != {"column"}
            or not isinstance(predicate.get("right"), Mapping)
            or set(predicate["right"]) != {"literal"}
        ):
            raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
                "strategy_sample_design_v2_native_bootstrap_required",
                "当前 compatibility anchor 只支持 column == literal 的简单切分；"
                "本次未创建计划。",
            )
        column = predicate["left"]["column"]
        literal = predicate["right"]["literal"]
        if not isinstance(column, str) or not column or literal is None:
            raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
                "strategy_sample_design_v2_native_bootstrap_required",
                "V2 compatibility 切分列与三个切分值必须完整。",
            )
        columns.append(column)
        values[partition] = literal
    if (
        len(set(columns)) != 1
        or len(
            {
                json.dumps(value, sort_keys=True, ensure_ascii=False)
                for value in values.values()
            }
        )
        != 3
    ):
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_sample_design_v2_native_bootstrap_required",
            "V2 compatibility 切分必须使用同一列上的三个互异标量值。",
        )
    return columns[0], values


def _latest_verified_strategy_sample_design_v2_binding(
    read_runtime: SimpleNamespace,
    *,
    task_id: str,
    artifacts: Sequence[Mapping],
):
    # TaskArtifactRepository.list_for_task is deterministic
    # ORDER BY created_at, id; the final row is therefore the latest published
    # V2 bundle, and a drifted latest bundle is never bypassed for an older one.
    supported_origins = {
        SAMPLE_DESIGN_V2_ORIGIN_TOOL,
        SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL,
    }
    bundles = [
        artifact
        for artifact in artifacts
        if artifact.get("kind") == SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND
        and artifact.get("origin_tool") in supported_origins
    ]
    if not bundles:
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_model_evidence_v2_sample_required",
            "当前任务还没有 StrategySampleDesign V2 双总体样本证据；"
            "请先用自然语言固化 V2 样本设计。",
        )
    newest = bundles[-1]
    provenance = newest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_model_evidence_v2_sample_invalid",
            "最新 StrategySampleDesign V2 bundle 缺少完整 provenance，本次未创建计划。",
        )
    try:
        return load_any_strategy_sample_design_v2_artifacts(
            read_runtime,
            task_id=task_id,
            membership_artifact_id=provenance.get("membership_artifact_id"),
            expected_membership_artifact_content_hash=provenance.get(
                "membership_artifact_content_hash"
            ),
            bundle_artifact_id=newest.get("id"),
            expected_bundle_artifact_content_hash=newest.get("content_hash"),
            expected_bundle_id=provenance.get("bundle_id"),
            expected_sample_design_id=provenance.get("sample_design_id"),
            expected_sample_design_content_hash=provenance.get(
                "sample_design_content_hash"
            ),
        )
    except (
        StrategyError,
        TypeError,
        ValueError,
        *strategy_contracts_lane._STRATEGY_V2_ARTIFACT_ERRORS,
    ) as exc:
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_model_evidence_v2_sample_invalid",
            "最新 StrategySampleDesign V2 membership/bundle pair 未通过"
            "文件、registry、provenance 或数据漂移复核；请重新固化样本设计。",
        ) from exc


def _inherit_strategy_sample_drop_nan_policy(
    runtime: contracts_lane.DriverTurnRuntime,
    task: TaskRecord,
    *,
    context,
) -> bool:
    """Reuse the exact missing-label policy already fixed by SampleDesign.

    Candidate Lab forms do not ask the user to restate this governed sample
    policy.  Probe both boolean identities through the normal authenticated
    loader and return the one owned by the newest exact sample design.  A
    corrupt or unsupported newest native record remains fail-closed.
    """

    failures: list[StrategySetupError] = []
    for candidate_policy in (False, True):
        try:
            _latest_matching_strategy_sample_design_ref(
                runtime,
                task,
                context=context,
                drop_nan_labels=candidate_policy,
                allow_native_risk_development=True,
            )
        except StrategySetupError as exc:
            failures.append(exc)
            continue
        return candidate_policy

    for failure in failures:
        if isinstance(
            failure, strategy_contracts_lane._StrategySampleDesignPolicyMismatchError
        ):
            continue
        if not isinstance(
            failure, strategy_contracts_lane._StrategySampleDesignRequiredError
        ):
            raise failure
    return False


def _latest_matching_strategy_sample_design_ref(
    runtime: contracts_lane.DriverTurnRuntime,
    task: TaskRecord,
    *,
    context,
    drop_nan_labels: bool,
    allow_native_risk_development: bool = False,
    month_col: str | None = None,
    weight_col: str | None = None,
    loan_amount_col: str | None = None,
    overdue_amount_col: str | None = None,
) -> dict[str, str]:
    """Bind downstream execution to the newest exact governed sample design.

    The language model never supplies artifact ids or hashes.  Selection is
    deterministic over task-owned registry rows and only considers designs
    whose immutable provenance already matches the active data/workspace/label
    boundary.  The selected artifact is then fully reloaded and authenticated.
    """

    if not isinstance(context.target_col, str) or not context.target_col:
        raise StrategySetupError("策略开发需要先在 DataWorkspace 中确认二元目标列。")
    expected = {
        "task_id": task.id,
        "dataset_id": context.dataset_id,
        "dataset_content_hash": context.dataset_content_hash,
        "workspace_revision": context.workspace_revision,
        "workspace_generation": context.analysis_generation,
        "semantic_mapping_hash": context.semantic_mapping_hash,
        "target_col": context.target_col,
    }
    matches: list[Mapping] = []
    latest_legacy_position = -1
    latest_native_position = -1
    latest_native_authenticated = None
    latest_invalid_native_position = -1
    latest_invalid_native_cause: Exception | None = None
    latest_native_policy_mismatch_position = -1
    artifact_repository = TaskArtifactRepository(runtime.settings.db_path)
    native_read_runtime = SimpleNamespace(
        settings=runtime.settings,
        task_artifacts=artifact_repository,
    )
    try:
        artifacts = artifact_repository.list_for_task(task.id)
    except Exception as exc:
        raise StrategySetupError(
            "无法读取当前任务的策略样本设计登记，不能安全继续策略开发。"
        ) from exc
    for position, artifact in enumerate(artifacts):
        if (
            artifact.get("kind") == SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND
            and artifact.get("origin_tool") == SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL
        ):
            try:
                authenticated = (
                    authenticate_native_strategy_sample_design_v2_bundle_record(
                        native_read_runtime,
                        task_id=task.id,
                        record=artifact,
                    )
                )
            except (StrategyError, TypeError, ValueError) as exc:
                latest_invalid_native_position = position
                latest_invalid_native_cause = exc
                continue
            relation = _native_sample_design_v2_context_relation(
                authenticated.source_provenance,
                expected=expected,
                drop_nan_labels=drop_nan_labels,
            )
            if relation == "invalid":
                latest_invalid_native_position = position
                latest_invalid_native_cause = None
                continue
            if relation == "policy_mismatch":
                latest_native_policy_mismatch_position = position
                continue
            if relation == "current":
                latest_native_position = position
                latest_native_authenticated = authenticated
            continue
        provenance = artifact.get("provenance")
        if (
            artifact.get("kind") != SAMPLE_DESIGN_ARTIFACT_KIND
            or artifact.get("origin_tool") != SAMPLE_DESIGN_ORIGIN_TOOL
            or not isinstance(provenance, Mapping)
            or any(provenance.get(field) != value for field, value in expected.items())
        ):
            continue
        request = provenance.get("request")
        if not isinstance(request, Mapping) or request.get(
            "drop_nan_labels"
        ) is not bool(drop_nan_labels):
            continue
        matches.append(artifact)
        latest_legacy_position = position
    latest_valid_position = max(
        latest_legacy_position,
        latest_native_position,
    )
    latest_blocking_native_position = max(
        latest_invalid_native_position,
        latest_native_policy_mismatch_position,
        *(() if allow_native_risk_development else (latest_native_position,)),
    )
    blocking_boundary = (
        latest_valid_position
        if allow_native_risk_development
        else latest_legacy_position
    )
    if latest_blocking_native_position > blocking_boundary:
        if (
            allow_native_risk_development
            and latest_native_policy_mismatch_position
            == latest_blocking_native_position
        ):
            raise strategy_contracts_lane._StrategySampleDesignPolicyMismatchError(
                "当前最新原生 StrategySampleDesign V2 的缺失标签政策与本轮"
                "执行口径不同；不会回退到更旧样本。"
            )
        error = strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_sample_design_v2_native_source_unsupported",
            "当前执行口径的最新相关 StrategySampleDesign V2 来自原生"
            "来源，或其 registry、文件、provenance/source identity "
            "无法认证；这个下游 Workflow 不会静默回退到更旧的 V1 "
            "compatibility 样本。",
        )
        if (
            latest_invalid_native_position == latest_blocking_native_position
            and latest_invalid_native_cause is not None
        ):
            raise error from latest_invalid_native_cause
        raise error
    select_native = (
        allow_native_risk_development
        and latest_native_position > latest_legacy_position
    )
    if not select_native and not matches:
        raise strategy_contracts_lane._StrategySampleDesignRequiredError(
            "当前活动数据和标签口径没有可执行的成熟策略样本设计。"
            "请先用自然语言说明坏样本值、表现窗、观察窗、成熟度及可选切分，"
            "让 MARVIS 固化样本设计。"
        )

    if select_native:
        if latest_native_authenticated is None:
            raise StrategySetupError(
                "当前原生 StrategySampleDesign V2 选择状态不完整，"
                "请重新固化样本设计后再执行。"
            )
        reference = {
            "artifact_id": latest_native_authenticated.artifact_id,
            "artifact_content_hash": (
                latest_native_authenticated.artifact_content_hash
            ),
            "sample_design_id": latest_native_authenticated.provenance[
                "sample_design_id"
            ],
            "sample_design_content_hash": (
                latest_native_authenticated.provenance["sample_design_content_hash"]
            ),
            "partition": "risk/development",
        }
    else:
        artifact = matches[-1]
        provenance = artifact["provenance"]
        reference = {
            "artifact_id": artifact.get("id"),
            "artifact_content_hash": artifact.get("content_hash"),
            "sample_design_id": provenance.get("sample_design_id"),
            "sample_design_content_hash": provenance.get("sample_design_content_hash"),
            "partition": "development",
        }
    backend = DataBackend(runtime.settings.datasets_dir)
    read_runtime = SimpleNamespace(
        settings=runtime.settings,
        backend=backend,
        registry=DatasetRegistry(
            DatasetRepository(runtime.settings.db_path),
            backend,
            runtime.settings.datasets_dir,
        ),
        task_artifacts=TaskArtifactRepository(runtime.settings.db_path),
    )
    try:
        loader = (
            load_strategy_risk_development_execution_binding
            if allow_native_risk_development
            else load_strategy_sample_design_execution_binding
        )
        binding = loader(
            read_runtime,
            task_id=task.id,
            sample_design_ref=reference,
            dataset_id=context.dataset_id,
            dataset_content_hash=context.dataset_content_hash,
            workspace_revision=context.workspace_revision,
            workspace_generation=context.analysis_generation,
            semantic_mapping_hash=context.semantic_mapping_hash,
            target_col=context.target_col,
            drop_nan_labels=bool(drop_nan_labels),
            month_col=month_col,
            weight_col=weight_col,
            loan_amount_col=loan_amount_col,
            overdue_amount_col=overdue_amount_col,
        )
    except StrategySampleDesignScopeIneligibleError as exc:
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            exc.code,
            "当前最新原生 StrategySampleDesign V2 已通过 task、registry、"
            "文件、hash、provenance/source identity 和活动 DataWorkspace "
            f"复核，但 scope 为 `{exc.scope}`，不能用于单变量或其他策略开发"
            "执行。请重新固化样本设计：确认成熟度和表现窗，并提供观察窗；"
            "平台不会把 exploration-only 证据静默升级或回退到旧 V1 样本。",
        ) from exc
    except StrategyError as exc:
        raise StrategySetupError(
            "当前最新策略样本设计未通过完整性、成熟度或字段口径校验；"
            "请重新固化样本设计后再执行。"
        ) from exc
    return binding.to_ref_dict()


def _native_sample_design_v2_context_relation(
    source_provenance: Mapping[str, object],
    *,
    expected: Mapping[str, object],
    drop_nan_labels: bool,
) -> str:
    """Classify one already-authenticated native source against execution."""

    if source_provenance.get("task_id") != expected["task_id"]:
        return "invalid"
    if source_provenance.get("dataset_id") != expected["dataset_id"]:
        return "other"
    if (
        source_provenance.get("dataset_content_hash")
        != expected["dataset_content_hash"]
    ):
        # Dataset ids are immutable; a different hash under the same id is
        # corruption, not another legitimate context.
        return "invalid"
    workspace_identity = (
        source_provenance.get("workspace_revision"),
        source_provenance.get("workspace_generation"),
    )
    expected_workspace_identity = (
        expected["workspace_revision"],
        expected["workspace_generation"],
    )
    if workspace_identity != expected_workspace_identity:
        return "other"
    if (
        source_provenance.get("semantic_mapping_hash")
        != expected["semantic_mapping_hash"]
        or source_provenance.get("target_col") != expected["target_col"]
    ):
        return "invalid"
    if source_provenance.get("drop_nan_labels") is not bool(drop_nan_labels):
        return "policy_mismatch"
    return "current"


def _strategy_sample_design_dataset_context(
    runtime: contracts_lane.DriverTurnRuntime,
    task: TaskRecord,
):
    """Resolve the sample-design source only from confirmed workspace state."""

    _require_strategy_sample_design_workspace(runtime, task)
    backend, registry = data_context_lane._modeling_data_runtime(runtime.settings)
    context = build_strategy_dataset_context(
        registry,
        backend,
        task.id,
        task.source_dir,
        target_col=None,
        require_target=True,
    )
    _validate_strategy_sample_design_target(
        registry,
        backend,
        dataset_id=context.dataset_id,
        target_col=context.target_col,
    )
    return context


def _strategy_sample_design_dataset_preview(
    runtime: contracts_lane.DriverTurnRuntime,
    task: TaskRecord,
):
    """Preview the exact active sample and its confirmed workspace target."""

    _ensure_strategy_sample_design_active_workspace(runtime, task)
    _require_strategy_sample_design_workspace(runtime, task)
    backend, registry = data_context_lane._modeling_data_runtime(runtime.settings)
    preview = preview_strategy_dataset_context(
        registry,
        backend,
        task.id,
        task.source_dir,
        target_col=None,
    )
    _validate_strategy_sample_design_target(
        registry,
        backend,
        dataset_id=preview.dataset_id,
        target_col=preview.target_col,
    )
    return preview


def _confirm_manual_sample_design_time_semantics(
    runtime: contracts_lane.DriverTurnRuntime,
    task: TaskRecord,
    *,
    draft: StandardWorkflowRequestDraft,
    preview,
) -> None:
    """Persist the date role explicitly confirmed by the Candidate Lab form."""

    bindings = draft.workflow_inputs.get("field_bindings")
    time_field = bindings.get("time_field") if isinstance(bindings, Mapping) else None
    if time_field is None:
        return
    if (
        not isinstance(time_field, str)
        or not time_field
        or preview is None
        or time_field not in tuple(preview.columns)
    ):
        raise StrategySetupError(
            "双人群样本设计的时间字段不在当前活动样本中，请重新选择。"
        )

    repository = DataWorkspaceRepository(runtime.settings.db_path)
    try:
        snapshot = repository.get_or_default(task.id)
        if (
            snapshot.active_dataset_id is None
            or snapshot.active_dataset_content_hash is None
            or not snapshot.semantic_mapping.target_col
        ):
            raise StrategySetupError("双人群样本设计需要已绑定活动样本和二元目标列。")
        if time_field == snapshot.semantic_mapping.target_col:
            raise StrategySetupError("双人群样本设计的时间字段不能与目标列相同。")
        roles = dict(snapshot.semantic_mapping.field_roles)
        roles[time_field] = "date"
        repository.save(
            task.id,
            DataWorkspaceDraft(
                active_dataset_id=snapshot.active_dataset_id,
                active_dataset_content_hash=(snapshot.active_dataset_content_hash),
                page=snapshot.page,
                selected_field=snapshot.selected_field,
                semantic_mapping=DataSemanticMapping(
                    target_col=snapshot.semantic_mapping.target_col,
                    field_roles=roles,
                    business_names=(snapshot.semantic_mapping.business_names),
                ),
            ),
            expected_revision=snapshot.revision,
            audit={
                "actor": "user:strategy-candidate-lab",
                "detail": {
                    "reason": (
                        "confirm explicit sample-design time field "
                        "with date semantic role"
                    ),
                    "time_field": time_field,
                },
            },
        )
    except StrategySetupError:
        raise
    except (
        DataWorkspaceDataError,
        DataWorkspaceDatasetNotFound,
        DataWorkspaceRevisionConflict,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise StrategySetupError(
            "双人群样本设计的时间字段确认期间 DataWorkspace 发生变化，请刷新后重试。"
        ) from exc


def _ensure_strategy_sample_design_active_workspace(
    runtime: contracts_lane.DriverTurnRuntime,
    task: TaskRecord,
) -> None:
    """Atomically bind one unambiguous sample and binary target on a fresh task."""

    repository = DataWorkspaceRepository(runtime.settings.db_path)
    try:
        snapshot = repository.get_or_default(task.id)
    except (DataWorkspaceDataError, KeyError, TypeError, ValueError) as exc:
        raise StrategySetupError(
            "策略样本设计需要有效且已确认的活动 DataWorkspace。"
        ) from exc
    if snapshot.active_dataset_id is not None:
        if not snapshot.semantic_mapping.target_col:
            raise StrategySetupError(
                "策略样本设计要求先在 DataWorkspace 确认二元目标列。"
            )
        return

    backend, registry = data_context_lane._modeling_data_runtime(runtime.settings)
    registered = [
        dataset
        for dataset in registry.list_for_task(task.id)
        if str(dataset.task_id) == task.id
        and dataset.role in {"sample", "strategy_sample"}
    ]
    if len(registered) > 1:
        raise StrategySetupError(
            "策略样本设计需要一个明确的活动样本；当前存在多个已注册数据集，"
            "请先在 DataWorkspace 选择并保存本次样本。"
        )

    try:
        preview = strategy_evidence_lane._strategy_dataset_preview(runtime, task)
    except StrategySetupError as exc:
        raise StrategySetupError(
            f"策略样本设计要求先在 DataWorkspace 选择唯一活动样本：{exc}"
        ) from exc
    if not preview.target_col:
        raise StrategySetupError(
            "策略样本设计要求可唯一确定的二元目标列，请先确认 target_col。"
        )
    try:
        context = strategy_evidence_lane._strategy_dataset_context(
            runtime, task, require_target=True
        )
    except StrategySetupError as exc:
        raise StrategySetupError(
            f"策略样本设计无法从当前 DataWorkspace 候选建立稳定绑定：{exc}"
        ) from exc
    registered = [
        dataset
        for dataset in registry.list_for_task(task.id)
        if str(dataset.task_id) == task.id
        and dataset.role in {"sample", "strategy_sample"}
    ]
    if (
        len(registered) != 1
        or registered[0].id != context.dataset_id
        or (preview.dataset_id is not None and preview.dataset_id != context.dataset_id)
        or tuple(preview.columns) != tuple(context.columns)
        or preview.target_col != context.target_col
    ):
        raise StrategySetupError(
            "策略样本设计需要一个明确且稳定的活动样本；"
            "当前存在多个或变化中的数据集，请先在 DataWorkspace 明确选择。"
        )
    if (
        not isinstance(context.dataset_content_hash, str)
        or not context.dataset_content_hash
        or not context.target_col
    ):
        raise StrategySetupError(
            "策略样本设计无法绑定样本哈希或二元目标列，请先确认数据与 target_col。"
        )
    _validate_strategy_sample_design_target(
        registry,
        backend,
        dataset_id=context.dataset_id,
        target_col=context.target_col,
    )

    try:
        pinned_dataset = registry.pin_authenticated_snapshot(context.dataset_id)
        if pinned_dataset.content_hash != context.dataset_content_hash:
            raise StrategySetupError(
                "策略样本设计的数据身份在绑定前发生变化，请重新确认样本。"
            )
        repository.save_initial_binding(
            task.id,
            DataWorkspaceDraft(
                active_dataset_id=context.dataset_id,
                active_dataset_content_hash=context.dataset_content_hash,
                semantic_mapping=DataSemanticMapping(
                    target_col=context.target_col,
                    field_roles={context.target_col: "target"},
                ),
            ),
            expected_revision=snapshot.revision,
            audit={
                "actor": "agent:strategy-sample-design",
                "detail": {
                    "reason": (
                        "atomically bind sole task sample and target "
                        "for strategy sample design"
                    )
                },
            },
        )
    except (
        DataWorkspaceDataError,
        DataWorkspaceDatasetNotFound,
        DataWorkspaceRevisionConflict,
        DatasetContentDriftError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        raise StrategySetupError(
            "策略样本设计的数据工作区在计划创建前发生变化，请重新确认活动样本和目标列。"
        ) from exc


def _validate_strategy_sample_design_target(
    registry,
    backend,
    *,
    dataset_id: object,
    target_col: object,
) -> tuple[int, int]:
    """Accept only native numeric 0/1 plus genuine null labels.

    Numeric strings are intentionally rejected even when pandas could coerce
    them. Infinite values and other finite numbers are hard errors, never NaN
    confirmation candidates.
    """

    if not isinstance(dataset_id, str) or not dataset_id:
        raise StrategySetupError(
            "策略样本设计必须绑定已注册的活动数据集后才能校验目标列。"
        )
    if not isinstance(target_col, str) or not target_col:
        raise StrategySetupError("策略样本设计要求已确认的二元目标列。")
    try:
        path = registry.resolve_path(dataset_id)
        frame = backend.read_frame(path, columns=[target_col])
        target = frame[target_col]
    except Exception as exc:
        raise StrategySetupError(
            f"目标列 `{target_col}` 无法从当前活动数据集读取。"
        ) from exc
    dtype_kind = getattr(target.dtype, "kind", None)
    if dtype_kind not in {"i", "u", "f"}:
        raise StrategySetupError(
            f"目标列 `{target_col}` 必须是数值 0/1 或真实空值；"
            "字符串 '0'/'1'、布尔值和其他编码不接受。"
        )
    null_mask = target.isna()
    for value in target.loc[~null_mask].tolist():
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise StrategySetupError(
                f"目标列 `{target_col}` 必须是数值 0/1 或真实空值。"
            ) from exc
        if not math.isfinite(number) or number not in {0.0, 1.0}:
            raise StrategySetupError(
                f"目标列 `{target_col}` 必须是数值 0/1 或真实空值；"
                "inf、-inf 和 0/1 之外的值不能进入样本设计。"
            )
    return int(len(target)), int(null_mask.sum())


def _require_strategy_sample_design_workspace(
    runtime: contracts_lane.DriverTurnRuntime,
    task: TaskRecord,
):
    try:
        snapshot = DataWorkspaceRepository(runtime.settings.db_path).get_or_default(
            task.id
        )
    except (DataWorkspaceDataError, KeyError, TypeError, ValueError) as exc:
        raise StrategySetupError(
            "策略样本设计需要有效且已确认的活动 DataWorkspace。"
        ) from exc
    if snapshot.active_dataset_id is None:
        raise StrategySetupError(
            "策略样本设计要求先在 DataWorkspace 选择并保存活动数据集。"
        )
    if not snapshot.semantic_mapping.target_col:
        raise StrategySetupError("策略样本设计要求先在 DataWorkspace 确认二元目标列。")
    return snapshot


def _require_strategy_pool_impact_workspace(
    runtime: contracts_lane.DriverTurnRuntime,
    task: TaskRecord,
):
    try:
        snapshot = DataWorkspaceRepository(runtime.settings.db_path).get_or_default(
            task.id
        )
    except (DataWorkspaceDataError, KeyError, TypeError, ValueError) as exc:
        raise StrategySetupError(
            "Strategy Pool 影响测算需要有效的活动 DataWorkspace。"
        ) from exc
    if snapshot.active_dataset_id is None:
        raise StrategySetupError(
            "Strategy Pool 影响测算要求先在 DataWorkspace 选择活动数据集。"
        )
    if not snapshot.semantic_mapping.target_col:
        raise StrategySetupError(
            "Strategy Pool 影响测算要求先在 DataWorkspace 确认二元目标列。"
        )
    return snapshot


def _append_strategy_nan_label_clarification(
    repo: TaskRepository,
    task: TaskRecord,
    state: dict,
) -> dict:
    n_nan = int(state.get("n_nan") or 0)
    n_total = int(state.get("n_total") or 0)
    target_col = str(state.get("target_col") or "")
    payload = state.get("draft")
    is_pool_impact = (
        isinstance(payload, Mapping)
        and payload.get("workflow")
        in strategy_contracts_lane._STRATEGY_POOL_MEASUREMENT_WORKFLOWS
    )
    is_sample_design = isinstance(payload, Mapping) and payload.get("workflow") in {
        "strategy_sample_design",
        "strategy_sample_design_v2",
    }
    if is_pool_impact or is_sample_design:
        missing_description = "空标签" if is_sample_design else "空或非有限标签"
        retained_statistics = (
            "总体、金额和权重统计" if is_sample_design else "总体、动作和金额统计"
        )
        message = (
            f"目标列 `{target_col}` 有 {n_nan}/{n_total} 行{missing_description}。"
            f"这些样本行仍会保留在{retained_statistics}中，只从坏账率/风险率分母中排除；"
            "本次尚未创建计划，平台不会默认采用该口径。"
            "如果确实允许，请明确回复「确认将空标签仅从风险分母排除并继续」；"
            "仅回复「确认」不会执行。"
        )
    else:
        message = (
            f"目标列 `{target_col}` 有 {n_nan}/{n_total} 行空或非有限标签。"
            "本次尚未创建计划，平台不会默认丢弃。"
            "如果确实允许，请明确回复「确认丢弃空标签并继续」；"
            "仅回复「确认」不会执行。"
        )
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=message,
        metadata={
            "intent": "strategy_drop_nan_labels_confirmation",
            "kind": "clarification",
            "code": "strategy_drop_nan_labels_confirmation_required",
            "fields": ["drop_nan_labels"],
            strategy_contracts_lane._STRATEGY_NAN_LABEL_META_KEY: state,
        },
    )
    return {
        "task_id": task.id,
        "status": "clarification_required",
        "code": "strategy_drop_nan_labels_confirmation_required",
        "fields": ["drop_nan_labels"],
        "label_quality": {
            "target_col": target_col,
            "n_total": n_total,
            "n_nan": n_nan,
        },
        "messages": repo.list_agent_messages(task.id),
    }


def _repeat_strategy_nan_label_clarification(
    repo: TaskRepository,
    task: TaskRecord,
    state: dict,
) -> dict:
    return _append_strategy_nan_label_clarification(repo, task, dict(state))


def _latest_strategy_nan_label_confirmation(
    conversation: list[dict],
) -> dict | None:
    last_assistant = next(
        (
            message
            for message in reversed(conversation)
            if message.get("role") == "assistant"
        ),
        None,
    )
    if last_assistant is None:
        return None
    state = (last_assistant.get("metadata") or {}).get(
        strategy_contracts_lane._STRATEGY_NAN_LABEL_META_KEY
    )
    return state if isinstance(state, dict) else None


def _strategy_pool_impact_dataset_context(
    runtime: contracts_lane.DriverTurnRuntime,
    task: TaskRecord,
):
    """Resolve target only from confirmed DataWorkspace semantics for impact."""

    _require_strategy_pool_impact_workspace(runtime, task)
    backend, registry = data_context_lane._modeling_data_runtime(runtime.settings)
    return build_strategy_dataset_context(
        registry,
        backend,
        task.id,
        task.source_dir,
        target_col=None,
        require_target=True,
    )


def _strategy_pool_impact_dataset_preview(
    runtime: contracts_lane.DriverTurnRuntime,
    task: TaskRecord,
):
    """Preview the active sample using only its confirmed workspace target."""

    _require_strategy_pool_impact_workspace(runtime, task)
    backend, registry = data_context_lane._modeling_data_runtime(runtime.settings)
    return preview_strategy_dataset_context(
        registry,
        backend,
        task.id,
        task.source_dir,
        target_col=None,
    )


def _strategy_impact_cube_dataset_preview(
    runtime: contracts_lane.DriverTurnRuntime,
    task: TaskRecord,
):
    """Expose compiler columns from the exact latest authenticated V2 sample."""

    read_runtime = strategy_evidence_lane._strategy_v2_read_runtime(runtime)
    try:
        artifacts = tuple(read_runtime.task_artifacts.list_for_task(task.id))
        sample = _latest_verified_strategy_sample_design_v2_binding(
            read_runtime,
            task_id=task.id,
            artifacts=artifacts,
        )
        target = sample.bundle["sample_design"]["target_selector"]["column"]
    except (
        KeyError,
        TypeError,
        strategy_contracts_lane._StrategyV2EvidenceSetupError,
        *strategy_contracts_lane._STRATEGY_V2_ARTIFACT_ERRORS,
    ) as exc:
        raise StrategySetupError(
            "ImpactCube 无法从最新 StrategySampleDesign V2 认证编译字段；"
            "请先重新固化样本设计。"
        ) from exc
    if (
        not isinstance(target, str)
        or not target
        or target not in sample.source_binding.columns
    ):
        raise StrategySetupError("最新 StrategySampleDesign V2 的目标列绑定无效。")
    return SimpleNamespace(
        dataset_id=sample.source_binding.dataset_id,
        columns=sample.source_binding.columns,
        target_col=target,
        identity={
            "kind": "strategy_sample_design_v2",
            "sample_design_ref": strategy_evidence_lane._strategy_report_sample_ref(
                sample
            ),
            "dataset_id": sample.source_binding.dataset_id,
            "dataset_content_hash": sample.source_binding.dataset_content_hash,
        },
    )


def _strategy_dataset_binding_matches(
    runtime: contracts_lane.DriverTurnRuntime,
    task: TaskRecord,
    *,
    preview,
    context,
    use_confirmed_workspace_target: bool = False,
    use_sample_design_workspace: bool = False,
) -> bool:
    """Verify the registered snapshot still represents the compiled preview."""

    if (
        tuple(context.columns) != tuple(preview.columns)
        or context.target_col != preview.target_col
    ):
        return False
    try:
        if use_sample_design_workspace:
            refreshed = _strategy_sample_design_dataset_preview(runtime, task)
        elif use_confirmed_workspace_target:
            refreshed = _strategy_pool_impact_dataset_preview(runtime, task)
        else:
            refreshed = strategy_evidence_lane._strategy_dataset_preview(runtime, task)
    except StrategySetupError:
        return False
    if (
        refreshed.dataset_id != context.dataset_id
        or tuple(refreshed.columns) != tuple(context.columns)
        or refreshed.target_col != context.target_col
    ):
        return False

    identity = preview.identity if isinstance(preview.identity, dict) else {}
    refreshed_identity = (
        refreshed.identity if isinstance(refreshed.identity, dict) else {}
    )
    context_fields = {
        "workspace_revision": getattr(context, "workspace_revision", None),
        "analysis_generation": getattr(context, "analysis_generation", None),
        "semantic_mapping_hash": getattr(context, "semantic_mapping_hash", None),
    }
    for field, context_value in context_fields.items():
        if field in identity and (
            identity[field] != refreshed_identity.get(field)
            or identity[field] != context_value
        ):
            return False
    if identity.get("kind") == "registered":
        return (
            identity.get("dataset_id") == refreshed_identity.get("dataset_id")
            and identity.get("content_hash") == refreshed_identity.get("content_hash")
            and identity.get("content_hash")
            == getattr(context, "dataset_content_hash", None)
        )
    if identity.get("kind") != "source":
        return False
    source_path = identity.get("source_path")
    expected_hash = identity.get("sha256")
    if not source_path or not expected_hash:
        return False
    try:
        # CSV/XLSX source registration may normalize bytes into Parquet.  The
        # confirmation binds the original source here; the registered Parquet
        # hash is bound separately in the plan/tool inputs.
        return sha256_file(Path(str(source_path))) == str(expected_hash)
    except OSError:
        return False


def _strategy_nan_label_clarification_response(
    runtime: contracts_lane.DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    draft: CompiledStrategyRequestDraft,
    context,
    n_total: int,
    n_nan: int,
) -> dict:
    is_pool_impact = (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow
        in strategy_contracts_lane._STRATEGY_POOL_MEASUREMENT_WORKFLOWS
    )
    is_sample_design = isinstance(
        draft, StandardWorkflowRequestDraft
    ) and draft.workflow in {"strategy_sample_design", "strategy_sample_design_v2"}
    refreshed = (
        _strategy_pool_impact_dataset_preview(runtime, task)
        if is_pool_impact
        else (
            _strategy_sample_design_dataset_preview(runtime, task)
            if is_sample_design
            else strategy_evidence_lane._strategy_dataset_preview(runtime, task)
        )
    )
    state = {
        "draft": draft.to_dict(),
        "dataset_id": context.dataset_id,
        "dataset_identity": dict(refreshed.identity),
        "target_col": context.target_col,
        "n_total": int(n_total),
        "n_nan": int(n_nan),
    }
    if is_pool_impact:
        try:
            _pool, pool_binding = (
                strategy_evidence_lane._strategy_pool_impact_pool_binding(
                    runtime,
                    task,
                    str(draft.workflow_inputs.get("strategy_type") or ""),
                )
            )
        except StrategySetupError as exc:
            return responses_lane._strategy_request_clarification_response(
                repo,
                task,
                code="strategy_pool_impact_binding_required",
                message=str(exc),
            )
        state["pool_binding"] = pool_binding
    return _append_strategy_nan_label_clarification(repo, task, state)
