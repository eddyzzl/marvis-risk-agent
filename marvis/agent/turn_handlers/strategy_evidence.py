"""Strategy evidence for governed Agent turns."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from marvis.agent.strategy_request_compiler import CompiledStrategyRequestDraft
from marvis.agent.strategy_request_compiler import StandardWorkflowRequestDraft
from marvis.agent.strategy_request_compiler import StrategyRequestDraft
from marvis.agent.strategy_setup import StrategySetupError
from marvis.agent.strategy_setup import build_strategy_dataset_context
from marvis.agent.strategy_setup import preview_strategy_dataset_context
from marvis.agent.strategy_workflows import migrated_workflow_requirements
from marvis.data.labels import nan_label_mask
from marvis.domain import StrategyProfitInput
from marvis.domain import StrategyTaskInput
from marvis.domain import TaskRecord
from marvis.packs.modeling.errors import ModelingError
from marvis.packs.modeling.evidence import MODELING_TRAINING_EVIDENCE_ARTIFACT_KIND
from marvis.packs.modeling.evidence_tools import build_training_evidence_ref
from marvis.packs.modeling.evidence_tools import (
    load_modeling_training_evidence_artifacts,
)
from marvis.packs.modeling.experiment import ExperimentStore
from marvis.packs.modeling.score_evidence import MODEL_SCORE_EVIDENCE_ARTIFACT_KIND
from marvis.packs.modeling.score_evidence_tools import (
    load_model_score_evidence_artifacts,
)
from marvis.packs.strategy.candidate_stability_tools import (
    ARTIFACT_KIND as CANDIDATE_STABILITY_ARTIFACT_KIND,
)
from marvis.packs.strategy.candidate_stability_tools import (
    load_candidate_stability_artifact,
)
from marvis.packs.strategy.cross_candidate_search_tools import (
    CROSS_CANDIDATE_SEARCH_ARTIFACT_KIND,
)
from marvis.packs.strategy.cross_candidate_search_tools import (
    load_cross_candidate_search_artifact,
)
from marvis.packs.strategy.cross_rule_search_tools import (
    CROSS_RULE_SEARCH_ARTIFACT_KIND,
)
from marvis.packs.strategy.cross_rule_search_tools import (
    load_cross_rule_search_artifact,
)
from marvis.packs.strategy.dsl import strategy_spec_hash
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.impact_cube_tools import IMPACT_CUBE_ARTIFACT_KIND
from marvis.packs.strategy.model_evidence_tools import MODEL_EVIDENCE_V2_ARTIFACT_KIND
from marvis.packs.strategy.model_evidence_tools import (
    derive_strategy_model_evidence_candidate_execution_ref,
)
from marvis.packs.strategy.model_evidence_tools import (
    load_strategy_model_evidence_v2_artifact,
)
from marvis.packs.strategy.model_evidence_tools import (
    strategy_model_evidence_registry_snapshot_token,
)
from marvis.packs.strategy.pool_impact_tools import POOL_IMPACT_ARTIFACT_KIND
from marvis.packs.strategy.pool_impact_tools import (
    load_historical_strategy_pool_impact_artifact,
)
from marvis.packs.strategy.pool_requirement_resolver import (
    pool_requirement_bindings_provenance,
)
from marvis.packs.strategy.pool_requirement_resolver import (
    project_pool_entry_requirements,
)
from marvis.packs.strategy.pool_requirement_resolver import resolve_pool_requirements
from marvis.packs.strategy.pool_stability_tools import POOL_STABILITY_ARTIFACT_KIND
from marvis.packs.strategy.pool_stability_tools import (
    load_strategy_pool_stability_artifact,
)
from marvis.packs.strategy.pool_tools import bind_strategy_pool_development_execution
from marvis.packs.strategy.pool_tools import (
    load_current_strategy_candidate_pool_artifact,
)
from marvis.packs.strategy.report_bundle_adapters import (
    validate_candidate_stability_report_compatibility,
)
from marvis.packs.strategy.report_bundle_adapters import (
    validate_cross_candidate_search_report_compatibility,
)
from marvis.packs.strategy.report_bundle_adapters import (
    validate_cross_rule_search_report_compatibility,
)
from marvis.packs.strategy.report_bundle_tools import (
    authenticate_strategy_report_identity_for_pool_on_connection,
)
from marvis.packs.strategy.report_bundle_tools import load_strategy_impact_cube_artifact
from marvis.packs.strategy.sample_design_v2_native_tools import (
    SAMPLE_DESIGN_V2_NATIVE_MEMBERSHIP_ARTIFACT_KIND,
)
from marvis.packs.strategy.sample_design_v2_tools import (
    SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
)
from marvis.packs.strategy.sample_design_v2_tools import (
    SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND,
)
from marvis.packs.strategy.sample_design_v2_tools import (
    load_any_strategy_sample_design_v2_artifacts,
)
from marvis.packs.strategy.voting_candidate import VOTING_CANDIDATE_ASSET_TYPE
from marvis.packs.strategy.voting_candidate_search_tools import (
    VOTING_CANDIDATE_SEARCH_ARTIFACT_KIND,
)
from marvis.packs.strategy.voting_candidate_search_tools import (
    load_historical_voting_candidate_search_artifact,
)
from marvis.repositories.modeling import ModelingRepository
from marvis.repositories.strategy import StrategyRepository
from marvis.repositories.strategy_pool import StrategyCandidatePoolRepository
from marvis.repositories.strategy_pool import strategy_pool_snapshot_hash
from marvis.repositories.task_artifacts import TaskArtifactRepository
from types import SimpleNamespace
import hashlib
import json
import re
from . import contracts as contracts_lane
from . import data_context as data_context_lane
from . import strategy_contracts as strategy_contracts_lane
from . import strategy_report as strategy_report_lane


def _strategy_pool_impact_pool_binding(
    runtime: contracts_lane.DriverTurnRuntime,
    task: TaskRecord,
    strategy_type: str,
) -> tuple[Mapping, dict[str, object]]:
    """Load one non-empty Pool and return its exact confirmation binding."""

    if strategy_type not in {"approval", "reject"}:
        raise StrategySetupError(
            "Strategy Pool 影响测算首个 V2 纵切只支持 approval/reject；"
            "其他策略类型需要后续类型专属口径。"
        )
    try:
        pool = StrategyCandidatePoolRepository(runtime.settings.db_path).get_current(
            task.id, strategy_type
        )
    except Exception as exc:
        raise StrategySetupError(
            "当前 Strategy Pool 状态无法通过完整性校验，不能执行影响测算。"
        ) from exc
    if pool is None:
        raise StrategySetupError(
            f"当前任务没有 {strategy_type} Strategy Pool，无法测算影响。"
        )
    if not _strategy_pool_entries(pool):
        raise StrategySetupError(
            f"当前 {strategy_type} Strategy Pool 为空；请先加入候选规则再测算影响。"
        )
    try:
        binding = {
            "strategy_type": strategy_type,
            "expected_pool_revision": int(pool["revision"]),
            "expected_pool_snapshot_hash": strategy_pool_snapshot_hash(pool),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise StrategySetupError(
            "当前 Strategy Pool revision/hash 绑定不完整，不能执行影响测算。"
        ) from exc
    return pool, binding


def _strategy_dsl_delivery_strategy_ref(
    snapshot: object,
    *,
    task_id: str,
) -> dict[str, object]:
    if not isinstance(snapshot, Mapping):
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_dsl_delivery_strategy_invalid",
            "策略缺少可认证的原子 snapshot，或不属于当前任务。",
        )
    try:
        strategy = snapshot["strategy"]
        metadata = snapshot["metadata"]
        spec_hash = snapshot["strategy_spec_hash"]
        strategy_id = str(metadata["id"])
        strategy_type = str(metadata["strategy_type"])
        version = metadata["version"]
        canonical_spec_hash = (
            strategy_spec_hash(strategy.spec)
            if getattr(strategy, "spec", None) is not None
            else None
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_dsl_delivery_strategy_invalid",
            "策略 snapshot 的 identity、type、version 或 spec hash 不完整。",
        ) from exc
    if (
        metadata.get("task_id") != task_id
        or getattr(strategy, "id", None) != strategy_id
        or getattr(strategy, "strategy_type", None) != strategy_type
        or strategy_type
        not in {"approval", "reject", "limit", "pricing", "segmentation"}
        or isinstance(version, bool)
        or not isinstance(version, int)
        or version < 1
        or not isinstance(spec_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", spec_hash) is None
        or canonical_spec_hash != spec_hash
    ):
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_dsl_delivery_strategy_invalid",
            "strategy_id 必须属于当前任务，并带有一致的五类 type、正版本和"
            " canonical Strategy DSL/spec hash；历史兼容行需先迁移。",
        )
    return {
        "strategy_id": strategy_id,
        "expected_strategy_type": strategy_type,
        "expected_version": version,
        "expected_spec_hash": spec_hash,
    }


def _strategy_impact_cube_partitions(
    inputs: Mapping,
    *,
    sample,
) -> list[str]:
    order = ("development", "validation", "oot")
    try:
        counts = sample.membership["header"]["counts"]
        approval_counts = counts["approval"]
        risk_counts = counts["risk"]
    except (KeyError, TypeError) as exc:
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_impact_cube_sample_invalid",
            "StrategySampleDesign V2 缺少 approval/risk 分区计数。",
        ) from exc
    for population_counts in (approval_counts, risk_counts):
        if not isinstance(population_counts, Mapping) or any(
            isinstance(population_counts.get(partition), bool)
            or not isinstance(population_counts.get(partition), int)
            or population_counts[partition] < 0
            for partition in order
        ):
            raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
                "strategy_impact_cube_sample_invalid",
                "StrategySampleDesign V2 的分区计数无效。",
            )

    requested = inputs.get("partitions")
    if requested is None:
        selected = [
            partition
            for partition in order
            if approval_counts[partition] > 0 and risk_counts[partition] > 0
        ]
    else:
        if not isinstance(requested, Sequence) or isinstance(
            requested, str | bytes | bytearray
        ):
            raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
                "strategy_impact_cube_partitions_invalid",
                "ImpactCube partitions 必须是明确的分区列表。",
            )
        requested_set = set(requested)
        if (
            not requested_set
            or len(requested_set) != len(requested)
            or not requested_set.issubset(order)
        ):
            raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
                "strategy_impact_cube_partitions_invalid",
                "ImpactCube 只接受不重复的 development、validation、oot 分区。",
            )
        selected = [partition for partition in order if partition in requested_set]
    empty = [
        partition
        for partition in selected
        if approval_counts[partition] == 0 or risk_counts[partition] == 0
    ]
    if not selected or empty:
        detail = "、".join(empty) if empty else "全部"
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_impact_cube_partition_empty",
            f"所选分区 {detail} 没有同时具备 approval 与 risk 总体；"
            "请调整样本设计或选择非空分区。",
        )
    return selected


def _strategy_impact_cube_dimensions(
    inputs: Mapping,
    *,
    sample,
) -> dict[str, str | None]:
    columns = tuple(sample.source_binding.columns)
    roles = dict(sample.source_binding.semantic_field_roles)
    provenance_request = sample.provenance.get("request")
    field_bindings = (
        provenance_request.get("field_bindings")
        if isinstance(provenance_request, Mapping)
        else None
    )
    if not isinstance(field_bindings, Mapping):
        field_bindings = {}

    def unique_role(role: str) -> str | None:
        matches = sorted(
            column
            for column, assigned in roles.items()
            if assigned == role and column in columns
        )
        if len(matches) > 1:
            raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
                "strategy_impact_cube_dimension_ambiguous",
                f"当前样本有多个 `{role}` 语义字段：{'、'.join(matches)}；"
                "请在请求中明确列名。",
            )
        return matches[0] if matches else None

    defaults = {
        "month_col": field_bindings.get("month_field") or unique_role("month"),
        "group_col": field_bindings.get("group_field"),
        "segment_col": unique_role("segment"),
    }
    result: dict[str, str | None] = {}
    used: set[str] = set()
    for field in ("month_col", "group_col", "segment_col"):
        explicit = inputs.get(field)
        selected = explicit if explicit is not None else defaults[field]
        if selected is not None and (
            not isinstance(selected, str) or selected not in columns
        ):
            raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
                "strategy_impact_cube_dimension_invalid",
                f"ImpactCube 维度 {field} 不在最新样本绑定的数据列中。",
            )
        if selected is not None and roles.get(selected) in {"id", "target"}:
            raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
                "strategy_impact_cube_dimension_sensitive",
                f"字段 `{selected}` 的语义角色是 {roles[selected]}，"
                "不能作为 ImpactCube 聚合维度。",
            )
        if selected is not None and selected in used:
            if explicit is not None:
                raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
                    "strategy_impact_cube_dimension_duplicate",
                    "月份、分组和分群维度必须使用不同字段。",
                )
            selected = None
        result[field] = selected
        if selected is not None:
            used.add(selected)
    return result


def _strategy_impact_cube_economics(
    value: object,
    *,
    sample,
) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_impact_cube_economics_invalid",
            "ImpactCube economics_inputs 必须是 typed column/scalar 映射。",
        )
    columns = set(sample.source_binding.columns)
    roles = dict(sample.source_binding.semantic_field_roles)
    result: dict[str, object] = {}
    for component, raw_binding in sorted(value.items()):
        if not isinstance(component, str) or not isinstance(raw_binding, Mapping):
            raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
                "strategy_impact_cube_economics_invalid",
                "ImpactCube economics_inputs 组件或绑定结构无效。",
            )
        binding = dict(raw_binding)
        if binding.get("kind") == "column":
            column = binding.get("column")
            if (
                not isinstance(column, str)
                or column not in columns
                or roles.get(column) in {"id", "target"}
            ):
                raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
                    "strategy_impact_cube_economics_column_invalid",
                    f"经济参数 {component} 未绑定到当前样本中的非敏感业务列。",
                )
        result[component] = binding
    return result


def _strategy_impact_cube_current_strategy_ref(
    runtime: contracts_lane.DriverTurnRuntime,
    *,
    task_id: str,
    strategy_type: str,
    requested_id: object,
) -> dict[str, str] | None:
    if requested_id is None:
        return None
    if not isinstance(requested_id, str) or not requested_id:
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_impact_cube_current_strategy_invalid",
            "当前策略比较需要完整 strategy_id。",
        )
    repository = StrategyRepository(runtime.settings.db_path)
    try:
        snapshot = repository.get_strategy_snapshot(requested_id)
    except Exception as exc:
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_impact_cube_current_strategy_invalid",
            "当前策略的 canonical StrategySpec 无法通过完整性校验。",
        ) from exc
    if snapshot is None:
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_impact_cube_current_strategy_invalid",
            "current_strategy_id 必须属于当前任务、类型一致并带有完整 canonical "
            "StrategySpec；平台不会跨任务或跨类型比较。",
        )
    meta = snapshot["metadata"]
    strategy = snapshot["strategy"]
    spec_hash = snapshot["strategy_spec_hash"]
    if (
        strategy.spec is None
        or meta.get("task_id") != task_id
        or meta.get("strategy_type") != strategy_type
        or strategy.strategy_type != strategy_type
        or not isinstance(spec_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", spec_hash) is None
    ):
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_impact_cube_current_strategy_invalid",
            "current_strategy_id 必须属于当前任务、类型一致并带有完整 canonical "
            "StrategySpec；平台不会跨任务或跨类型比较。",
        )
    return {
        "strategy_id": requested_id,
        "expected_strategy_spec_hash": spec_hash,
    }


def _strategy_impact_cube_registry_token(
    artifacts: Sequence[Mapping],
    *,
    selected_artifact_ids: set[str],
) -> str:
    relevant = [
        {
            "id": item.get("id"),
            "kind": item.get("kind"),
            "content_hash": item.get("content_hash"),
            "origin_tool": item.get("origin_tool"),
            "provenance": item.get("provenance"),
        }
        for item in artifacts
        if item.get("id") in selected_artifact_ids
        or item.get("kind")
        in {
            SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND,
            SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
        }
    ]
    return hashlib.sha256(
        json.dumps(
            relevant,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _strategy_v2_read_runtime(
    runtime: contracts_lane.DriverTurnRuntime,
) -> SimpleNamespace:
    backend, registry = data_context_lane._modeling_data_runtime(runtime.settings)
    return SimpleNamespace(
        settings=runtime.settings,
        backend=backend,
        registry=registry,
        task_artifacts=TaskArtifactRepository(runtime.settings.db_path),
    )


def _strategy_v2_artifact_snapshot(
    read_runtime: SimpleNamespace,
    *,
    task_id: str,
) -> tuple[dict, ...]:
    """Read one deterministic registry snapshot or expose a governed error."""

    try:
        return tuple(read_runtime.task_artifacts.list_for_task(task_id))
    except strategy_contracts_lane._STRATEGY_V2_ARTIFACT_ERRORS as exc:
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_model_evidence_v2_registry_unavailable",
            "无法读取当前任务的 StrategySampleDesign V2 artifact registry。",
        ) from exc


def _strategy_v2_registry_token(artifacts: Sequence[Mapping]) -> str:
    """CAS token for evidence rows relevant to one ModelEvidence V2 plan."""

    try:
        return strategy_model_evidence_registry_snapshot_token(artifacts)
    except StrategyError as exc:
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_model_evidence_v2_registry_unavailable",
            "Strategy ModelEvidence V2 registry snapshot 无法规范化。",
        ) from exc


def _strategy_report_read_runtime(
    runtime: contracts_lane.DriverTurnRuntime,
) -> SimpleNamespace:
    backend, registry = data_context_lane._modeling_data_runtime(runtime.settings)
    task_artifacts = TaskArtifactRepository(runtime.settings.db_path)
    return SimpleNamespace(
        settings=runtime.settings,
        backend=backend,
        registry=registry,
        task_artifacts=task_artifacts,
        strategies=StrategyRepository(runtime.settings.db_path),
        experiments=ExperimentStore(runtime.settings.db_path),
        modeling_repo=ModelingRepository(runtime.settings.db_path),
    )


def _strategy_report_artifact_window(
    read_runtime: SimpleNamespace,
    *,
    task_id: str,
    kind: str,
    limit: int,
    unavailable_code: str,
    invalid_code: str,
    label: str,
) -> tuple[tuple[Mapping, ...], int]:
    """Read one exact newest-first artifact window without full-task allocation."""

    try:
        records, total = (
            read_runtime.task_artifacts.list_recent_for_task_kind_with_count(
                task_id,
                kind,
                limit=limit,
            )
        )
    except strategy_contracts_lane._STRATEGY_V2_ARTIFACT_ERRORS as exc:
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            unavailable_code,
            f"无法读取当前任务最新的 {label} artifact 窗口。",
        ) from exc
    try:
        if (
            not isinstance(records, Sequence)
            or isinstance(records, str | bytes | bytearray)
            or isinstance(total, bool)
            or not isinstance(total, int)
            or total < 0
        ):
            raise ValueError(f"{label} artifact window is invalid")
        window = tuple(records)
        if len(window) != min(total, limit) or any(
            not isinstance(item, Mapping) or item.get("kind") != kind for item in window
        ):
            raise ValueError(f"{label} artifact window is inconsistent")
        return window, total
    except (TypeError, ValueError) as exc:
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            invalid_code,
            f"{label} artifact 窗口与精确总数或 kind 不一致；"
            "其 newest-first 选择边界无法确认。",
        ) from exc


def _strategy_report_requested_pool_type(
    source_message: Mapping | None,
) -> str | None:
    text = (
        str(source_message.get("content") or "")
        if isinstance(source_message, Mapping)
        else ""
    )
    masked = list(text)
    for match in strategy_contracts_lane._STRATEGY_REPORT_POOL_TITLE_RE.finditer(text):
        masked[match.start() : match.end()] = " " * (match.end() - match.start())
    command_text = "".join(masked)
    actions = tuple(
        strategy_contracts_lane._STRATEGY_REPORT_POOL_COMMAND_RE.finditer(command_text)
    )
    selected: list[str] = []
    negated: list[str] = []
    for (
        strategy_type,
        pattern,
    ) in strategy_contracts_lane._STRATEGY_REPORT_POOL_TYPE_PATTERNS.items():
        for match in pattern.finditer(command_text):
            prefix = command_text[max(0, match.start() - 40) : match.start()]
            if strategy_contracts_lane._STRATEGY_REPORT_POOL_TYPE_NEGATION_RE.search(
                prefix
            ):
                if strategy_type not in negated:
                    negated.append(strategy_type)
                continue
            clause_start = max(
                command_text.rfind(separator, 0, match.start())
                for separator in (
                    "，",
                    ",",
                    "；",
                    ";",
                    "。",
                    ".",
                    "！",
                    "!",
                    "？",
                    "?",
                    "\n",
                )
            )
            clause_end_candidates = [
                position
                for separator in (
                    "，",
                    ",",
                    "；",
                    ";",
                    "。",
                    ".",
                    "！",
                    "!",
                    "？",
                    "?",
                    "\n",
                )
                if (position := command_text.find(separator, match.end())) >= 0
            ]
            clause_end = (
                min(clause_end_candidates)
                if clause_end_candidates
                else len(command_text)
            )
            local_clause = command_text[clause_start + 1 : clause_end]
            if strategy_contracts_lane._STRATEGY_REPORT_POOL_HISTORY_RE.search(
                local_clause
            ):
                continue

            inside_creation = any(
                action.start() <= match.start() and match.end() <= action.end()
                for action in actions
            )
            explicitly_selected = (
                strategy_contracts_lane._STRATEGY_REPORT_POOL_SELECTOR_RE.search(prefix)
                is not None
                and _strategy_report_pool_selector_shares_command(
                    command_text,
                    mention_start=match.start(),
                    mention_end=match.end(),
                    actions=actions,
                )
            )
            if inside_creation or explicitly_selected:
                if strategy_type not in selected:
                    selected.append(strategy_type)
                break
    if len(selected) > 1:
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_pool_type_ambiguous",
            "同一报告请求同时点名多个 Strategy Pool 类型；请只选择一种策略类型。",
        )
    if not selected and negated:
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_pool_type_required",
            "报告请求只排除了 Pool 类型，没有明确肯定选择要使用的"
            "审批/准入、拒绝、额度、定价或分群 Pool；平台不会绑定"
            "被否定的 Pool。",
        )
    return selected[0] if selected else None


def _strategy_report_pool_selector_shares_command(
    utterance: str,
    *,
    mention_start: int,
    mention_end: int,
    actions: Sequence[re.Match[str]],
) -> bool:
    sentence_start = max(
        utterance.rfind(separator, 0, mention_start)
        for separator in ("。", ".", "！", "!", "？", "?", "；", ";", "\n")
    )
    sentence_end_candidates = [
        position
        for separator in ("。", ".", "！", "!", "？", "?", "；", ";", "\n")
        if (position := utterance.find(separator, mention_end)) >= 0
    ]
    sentence_end = (
        min(sentence_end_candidates) if sentence_end_candidates else len(utterance)
    )
    return any(
        sentence_start < action.start() and action.end() <= sentence_end
        for action in actions
    )


def _strategy_report_current_pool_binding(
    read_runtime: SimpleNamespace,
    *,
    task_id: str,
    requested_type: str | None,
):
    repository = StrategyCandidatePoolRepository(read_runtime.settings.db_path)
    current: dict[str, Mapping] = {}
    strategy_types = (
        (requested_type,)
        if requested_type is not None
        else (
            "approval",
            "reject",
            "limit",
            "pricing",
            "segmentation",
        )
    )
    try:
        for strategy_type in strategy_types:
            pool = repository.get_current(task_id, strategy_type)
            if pool is not None and pool.get("entries"):
                current[strategy_type] = pool
    except Exception as exc:
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_pool_invalid",
            "当前 Strategy Pool head/revision 无法通过完整性复核。",
        ) from exc

    if requested_type is not None:
        selected_type = requested_type
        if selected_type not in current:
            raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
                "strategy_report_bundle_v2_pool_required",
                f"当前任务没有非空 {selected_type} Strategy Pool。",
            )
    elif len(current) == 1:
        selected_type = next(iter(current))
    elif not current:
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_pool_required",
            "当前任务没有可用于报告的非空 Strategy Pool。",
        )
    else:
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_pool_type_required",
            "当前同时存在多个非空 Strategy Pool；请在报告请求中明确"
            "选择审批/准入、拒绝、额度、定价或分群 Pool，平台不会猜测。",
        )

    selected = current[selected_type]
    try:
        return load_current_strategy_candidate_pool_artifact(
            read_runtime,
            task_id=task_id,
            strategy_type=selected_type,
            expected_pool_revision=selected["revision"],
            expected_pool_snapshot_hash=strategy_pool_snapshot_hash(selected),
        )
    except (
        StrategyError,
        *strategy_contracts_lane._STRATEGY_V2_ARTIFACT_ERRORS,
    ) as exc:
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_pool_invalid",
            f"当前 {selected_type} Strategy Pool 的 artifact、来源或数据绑定"
            "未通过完整性复核。",
        ) from exc


def _strategy_report_latest_sample_binding(
    read_runtime: SimpleNamespace,
    *,
    task_id: str,
):
    bundles, _total = _strategy_report_artifact_window(
        read_runtime,
        task_id=task_id,
        kind=SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
        limit=1,
        unavailable_code="strategy_report_bundle_v2_sample_registry_unavailable",
        invalid_code="strategy_report_bundle_v2_sample_invalid",
        label="StrategySampleDesign V2 bundle",
    )
    if not bundles:
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_sample_required",
            "当前任务没有 StrategySampleDesign V2 membership/bundle 证据。",
        )
    newest = bundles[0]
    provenance = newest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_sample_invalid",
            "最新 StrategySampleDesign V2 bundle provenance 已损坏；"
            "平台不会回退到旧样本设计。",
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
            "strategy_report_bundle_v2_sample_invalid",
            "最新 StrategySampleDesign V2 membership/bundle 未通过文件、"
            "registry、provenance 或数据漂移复核；平台不会回退到旧版本。",
        ) from exc


def _strategy_report_sample_ref(sample) -> dict[str, object]:
    design = sample.bundle["sample_design"]
    return {
        "membership_artifact_id": sample.membership_artifact_id,
        "expected_membership_artifact_content_hash": (
            sample.membership_artifact_content_hash
        ),
        "bundle_artifact_id": sample.bundle_artifact_id,
        "expected_bundle_artifact_content_hash": (sample.bundle_artifact_content_hash),
        "expected_bundle_id": sample.bundle["bundle_id"],
        "expected_sample_design_id": design["sample_design_id"],
        "expected_sample_design_content_hash": design["content_hash"],
    }


def _strategy_report_latest_candidate_stability_binding(
    read_runtime: SimpleNamespace,
    *,
    task_id: str,
    sample,
    pool,
):
    """Select the newest authenticated stability evidence for current sources.

    Every stability artifact is authenticated before its source identity is
    inspected.  A valid artifact for another Pool/SampleDesign is skipped; a
    corrupt candidate fails closed because its actual source cannot be trusted
    and the selector must not silently fall back to older evidence.
    """

    records, total = _strategy_report_artifact_window(
        read_runtime,
        task_id=task_id,
        kind=CANDIDATE_STABILITY_ARTIFACT_KIND,
        limit=strategy_contracts_lane._STRATEGY_REPORT_EVIDENCE_REPLAY_LIMIT,
        unavailable_code=(
            "strategy_report_bundle_v2_candidate_stability_registry_unavailable"
        ),
        invalid_code="strategy_report_bundle_v2_candidate_stability_invalid",
        label="candidate stability",
    )
    for item in records:
        provenance = item.get("provenance")
        try:
            binding = load_candidate_stability_artifact(
                read_runtime,
                task_id=task_id,
                artifact_id=item.get("id"),
                expected_artifact_content_hash=item.get("content_hash"),
                expected_stability_id=(
                    provenance.get("stability_id")
                    if isinstance(provenance, Mapping)
                    else None
                ),
                expected_stability_content_hash=(
                    provenance.get("stability_content_hash")
                    if isinstance(provenance, Mapping)
                    else None
                ),
            )
        except (
            StrategyError,
            TypeError,
            ValueError,
            *strategy_contracts_lane._STRATEGY_V2_ARTIFACT_ERRORS,
        ) as exc:
            raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
                "strategy_report_bundle_v2_candidate_stability_invalid",
                "最新待判定的候选逐月稳定性 artifact 未通过文件、registry、"
                "provenance 或内容完整性复核；其真实 Pool/SampleDesign "
                "身份无法确认，平台不会回退到旧稳定性证据。",
            ) from exc
        try:
            validate_candidate_stability_report_compatibility(
                candidate_stability=binding,
                sample_design=sample,
                candidate_pool=pool,
            )
        except StrategyError:
            continue
        return binding
    if total > strategy_contracts_lane._STRATEGY_REPORT_EVIDENCE_REPLAY_LIMIT:
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_candidate_stability_selection_window_exhausted",
            "已完整认证最新 "
            f"{strategy_contracts_lane._STRATEGY_REPORT_EVIDENCE_REPLAY_LIMIT} 个候选逐月稳定性 "
            "artifact，但 registry 仍有更早记录；平台无法证明窗口外"
            "不存在与当前 Pool/SampleDesign 完全一致的稳定性证据，"
            "本次未创建报告计划。",
        )
    return None


def _strategy_report_latest_voting_search_binding(
    read_runtime: SimpleNamespace,
    *,
    task_id: str,
    sample,
    sample_ref: Mapping[str, object],
    pool,
):
    """Select newest-to-oldest fully authenticated exact Voting search evidence."""

    records, total = _strategy_report_artifact_window(
        read_runtime,
        task_id=task_id,
        kind=VOTING_CANDIDATE_SEARCH_ARTIFACT_KIND,
        limit=strategy_contracts_lane._STRATEGY_REPORT_VOTING_SEARCH_REPLAY_LIMIT,
        unavailable_code=(
            "strategy_report_bundle_v2_voting_candidate_search_registry_unavailable"
        ),
        invalid_code=("strategy_report_bundle_v2_voting_candidate_search_invalid"),
        label="Voting candidate search",
    )
    if not records:
        return None
    try:
        current_development = bind_strategy_pool_development_execution(
            read_runtime,
            pool,
        )
        entries = [
            dict(entry)
            for entry in pool.pool["entries"]
            if entry["enabled"] is True
            and entry["source"]["asset_type"] != VOTING_CANDIDATE_ASSET_TYPE
        ]
        candidate_ids = sorted(str(entry["rule_id"]) for entry in entries)
        requirements = project_pool_entry_requirements(entries)
        if requirements:
            resolved = resolve_pool_requirements(
                read_runtime,
                task_id=task_id,
                compiled_design={"requirements": list(requirements)},
                sample_design=sample,
            )
            requirement_bindings = pool_requirement_bindings_provenance(resolved)
        else:
            requirement_bindings = None
        execution_sample_ref = derive_strategy_model_evidence_candidate_execution_ref(
            sample
        )
        if (
            _strategy_report_sample_ref(sample) != dict(sample_ref)
            or current_development.sample_design.to_ref_dict() != execution_sample_ref
        ):
            return None
    except (
        KeyError,
        ModelingError,
        StrategyError,
        TypeError,
        ValueError,
        *strategy_contracts_lane._STRATEGY_V2_ARTIFACT_ERRORS,
    ) as exc:
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_voting_candidate_search_invalid",
            "当前 Strategy Pool 的 Voting 搜索匹配身份无法通过完整认证；"
            "本次未创建报告计划。",
        ) from exc

    for item in records:
        try:
            binding = load_historical_voting_candidate_search_artifact(
                read_runtime,
                task_id=task_id,
                artifact_id=item.get("id"),
                expected_artifact_content_hash=item.get("content_hash"),
            )
        except (
            KeyError,
            ModelingError,
            StrategyError,
            TypeError,
            ValueError,
            *strategy_contracts_lane._STRATEGY_V2_ARTIFACT_ERRORS,
        ) as exc:
            raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
                "strategy_report_bundle_v2_voting_candidate_search_invalid",
                "最新待判定的 Voting 候选搜索 artifact 未通过历史安全的"
                "文件、registry、provenance、Pool 或数据绑定复核；其真实"
                "身份无法确认，平台不会回退到旧搜索证据。",
            ) from exc
        if _strategy_report_voting_search_matches(
            binding,
            task_id=task_id,
            pool=pool,
            current_development=current_development,
            execution_sample_ref=execution_sample_ref,
            candidate_ids=candidate_ids,
            requirement_bindings=requirement_bindings,
        ):
            return binding
    if total > strategy_contracts_lane._STRATEGY_REPORT_VOTING_SEARCH_REPLAY_LIMIT:
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_voting_candidate_search_"
            "selection_window_exhausted",
            "已完整认证最新 "
            f"{strategy_contracts_lane._STRATEGY_REPORT_VOTING_SEARCH_REPLAY_LIMIT} 个 Voting 候选"
            "搜索 artifact，但 registry 仍有更早的搜索记录；平台无法证明"
            "窗口外不存在与当前 Pool/SampleDesign 完全一致的搜索证据，"
            "本次未创建报告计划。",
        )
    return None


def _strategy_report_voting_search_matches(
    binding,
    *,
    task_id: str,
    pool,
    current_development,
    execution_sample_ref: Mapping[str, object],
    candidate_ids: Sequence[str],
    requirement_bindings: Mapping[str, object] | None,
) -> bool:
    provenance = binding.artifact_provenance
    historical_development = binding.pool_development
    historical_pool = historical_development.pool
    if (
        binding.task_id != task_id
        or historical_pool.artifact_id != pool.artifact_id
        or historical_pool.artifact_content_hash != pool.artifact_content_hash
        or historical_pool.pool != pool.pool
        or provenance["task_id"] != task_id
        or provenance["pool_ref"]
        != {
            "artifact_id": pool.artifact_id,
            "artifact_content_hash": pool.artifact_content_hash,
            "pool_id": pool.pool["pool_id"],
            "strategy_type": pool.pool["strategy_type"],
            "revision": pool.pool["revision"],
            "revision_id": pool.pool["revision_id"],
            "snapshot_hash": pool.pool["snapshot_hash"],
        }
    ):
        return False
    if historical_development.sample_design.to_ref_dict() != dict(execution_sample_ref):
        return False

    dataset = current_development.dataset
    execution_sample = current_development.sample_design
    expected_dataset = {
        "task_id": dataset.task_id,
        "dataset_id": dataset.dataset_id,
        "dataset_source_path": dataset.source_path,
        "dataset_content_hash": dataset.content_hash,
        "dataset_registry_metadata_hash": dataset.registry_metadata_hash,
        "workspace_revision": execution_sample.workspace_revision,
        "workspace_generation": execution_sample.workspace_generation,
        "semantic_mapping_hash": execution_sample.semantic_mapping_hash,
    }
    target = provenance["target_binding"]
    expected_target_identity = {
        "column": execution_sample.target_col,
        "raw_bad_value": execution_sample.target_bad_value,
        "normalized_bad_value": 1,
        "drop_nan_labels": execution_sample.drop_nan_labels,
        "sample_partition": execution_sample.reference.partition,
    }
    if (
        provenance["dataset_binding"] != expected_dataset
        or provenance["sample_design_ref"] != execution_sample.to_ref_dict()
        or provenance["sample_context_hash"]
        != current_development.evidence_identity["sample_context_hash"]
        or any(
            target.get(field) != expected
            for field, expected in expected_target_identity.items()
        )
        or target["labeled_count"] + target["nan_labels_dropped"]
        != execution_sample.development_population_count
        or (target["nan_labels_dropped"] > 0 and not execution_sample.drop_nan_labels)
        or provenance["observation_bindings"]
        != {
            "weight_col": execution_sample.weight_col,
            "amount_col": execution_sample.loan_amount_col,
        }
        or provenance["requirement_bindings"]
        != (None if requirement_bindings is None else dict(requirement_bindings))
        or binding.result["configuration"]["candidate_ids"] != list(candidate_ids)
    ):
        return False
    return True


def _strategy_report_latest_cross_search_binding(
    read_runtime: SimpleNamespace,
    *,
    task_id: str,
    sample,
):
    """Select the newest fully authenticated Cross search for this V2 sample."""

    records, total = _strategy_report_artifact_window(
        read_runtime,
        task_id=task_id,
        kind=CROSS_CANDIDATE_SEARCH_ARTIFACT_KIND,
        limit=strategy_contracts_lane._STRATEGY_REPORT_CROSS_SEARCH_REPLAY_LIMIT,
        unavailable_code=(
            "strategy_report_bundle_v2_cross_candidate_search_registry_unavailable"
        ),
        invalid_code=("strategy_report_bundle_v2_cross_candidate_search_invalid"),
        label="Cross candidate search",
    )
    for item in records:
        try:
            binding = load_cross_candidate_search_artifact(
                read_runtime,
                task_id=task_id,
                artifact_id=item.get("id"),
                expected_artifact_content_hash=item.get("content_hash"),
            )
        except (
            KeyError,
            ModelingError,
            StrategyError,
            TypeError,
            ValueError,
            *strategy_contracts_lane._STRATEGY_V2_ARTIFACT_ERRORS,
        ) as exc:
            raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
                "strategy_report_bundle_v2_cross_candidate_search_invalid",
                "最新待判定的 Cross 候选搜索 artifact 未通过文件、registry、"
                "provenance 或样本绑定复核；其真实身份无法确认，平台不会"
                "回退到旧搜索证据。",
            ) from exc
        if _strategy_report_cross_search_matches(
            binding,
            sample=sample,
        ):
            return binding
    if total > strategy_contracts_lane._STRATEGY_REPORT_CROSS_SEARCH_REPLAY_LIMIT:
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_cross_candidate_search_"
            "selection_window_exhausted",
            "已完整认证最新 "
            f"{strategy_contracts_lane._STRATEGY_REPORT_CROSS_SEARCH_REPLAY_LIMIT} 个 Cross 候选搜索 "
            "artifact，但 registry 仍有更早记录；平台无法证明窗口外"
            "不存在与当前 SampleDesign 完全一致的搜索证据，本次未创建"
            "报告计划。",
        )
    return None


def _strategy_report_cross_search_matches(
    binding,
    *,
    sample,
) -> bool:
    try:
        validate_cross_candidate_search_report_compatibility(
            cross_candidate_search=binding,
            sample_design=sample,
        )
    except StrategyError:
        return False
    return True


def _strategy_report_latest_cross_rule_search_binding(
    read_runtime: SimpleNamespace,
    *,
    task_id: str,
    sample,
):
    """Select the newest authenticated Cross rule search for this V2 sample."""

    records, total = _strategy_report_artifact_window(
        read_runtime,
        task_id=task_id,
        kind=CROSS_RULE_SEARCH_ARTIFACT_KIND,
        limit=strategy_contracts_lane._STRATEGY_REPORT_CROSS_SEARCH_REPLAY_LIMIT,
        unavailable_code=(
            "strategy_report_bundle_v2_cross_rule_search_registry_unavailable"
        ),
        invalid_code="strategy_report_bundle_v2_cross_rule_search_invalid",
        label="Cross rule search",
    )
    for item in records:
        try:
            binding = load_cross_rule_search_artifact(
                read_runtime,
                task_id=task_id,
                artifact_id=item.get("id"),
                expected_artifact_content_hash=item.get("content_hash"),
            )
        except (
            KeyError,
            ModelingError,
            StrategyError,
            TypeError,
            ValueError,
            *strategy_contracts_lane._STRATEGY_V2_ARTIFACT_ERRORS,
        ) as exc:
            raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
                "strategy_report_bundle_v2_cross_rule_search_invalid",
                "最新待判定的 Cross 阈值规则搜索 artifact 未通过文件、"
                "registry、provenance 或样本绑定复核；平台不会回退到旧证据。",
            ) from exc
        try:
            validate_cross_rule_search_report_compatibility(
                cross_rule_search=binding,
                sample_design=sample,
            )
        except StrategyError:
            continue
        return binding
    if total > strategy_contracts_lane._STRATEGY_REPORT_CROSS_SEARCH_REPLAY_LIMIT:
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_cross_rule_search_selection_window_exhausted",
            "已完整认证最新 "
            f"{strategy_contracts_lane._STRATEGY_REPORT_CROSS_SEARCH_REPLAY_LIMIT} 个 Cross 阈值"
            "规则搜索 artifact，但 registry 仍有更早记录；平台无法证明"
            "窗口外不存在兼容证据，本次未创建报告计划。",
        )
    return None


def _strategy_report_latest_impact_cube_binding(
    read_runtime: SimpleNamespace,
    *,
    task_id: str,
    pool,
    sample_ref: Mapping[str, object],
):
    """Prefer the newest exact Pool + SampleDesign ImpactCube.

    The repository window is newest-first. Authenticate each candidate before
    inspecting its embedded Pool/SampleDesign identity. A valid unrelated cube
    can be skipped, while an unauthenticatable candidate fails closed because
    its raw provenance cannot safely prove that it was unrelated.
    """

    expected_pool_ref = {
        "artifact_id": pool.artifact_id,
        "expected_artifact_content_hash": pool.artifact_content_hash,
        "expected_pool_id": pool.pool["pool_id"],
        "expected_revision": pool.pool["revision"],
        "expected_revision_id": pool.pool["revision_id"],
        "expected_snapshot_hash": pool.pool["snapshot_hash"],
    }
    same_kind, total = _strategy_report_artifact_window(
        read_runtime,
        task_id=task_id,
        kind=IMPACT_CUBE_ARTIFACT_KIND,
        limit=strategy_contracts_lane._STRATEGY_REPORT_EVIDENCE_REPLAY_LIMIT,
        unavailable_code=("strategy_report_bundle_v2_impact_cube_registry_unavailable"),
        invalid_code="strategy_report_bundle_v2_impact_cube_invalid",
        label="ImpactCube",
    )
    for item in same_kind:
        provenance = item.get("provenance")
        try:
            binding = load_strategy_impact_cube_artifact(
                read_runtime,
                task_id=task_id,
                artifact_id=item.get("id"),
                expected_artifact_content_hash=item.get("content_hash"),
                expected_cube_id=(
                    provenance.get("cube_id")
                    if isinstance(provenance, Mapping)
                    else None
                ),
                expected_cube_content_hash=(
                    provenance.get("cube_content_hash")
                    if isinstance(provenance, Mapping)
                    else None
                ),
            )
            cube = binding.cube
            identity = cube["identity"]
            sources = cube["source_bindings"]
            pool_artifact = sources["pool_artifact"]
            sample = sources["sample_design_v2"]
            authenticated_pool_ref = {
                "artifact_id": pool_artifact["artifact_id"],
                "expected_artifact_content_hash": pool_artifact[
                    "artifact_content_hash"
                ],
                "expected_pool_id": identity["pool_id"],
                "expected_revision": identity["revision"],
                "expected_revision_id": identity["revision_id"],
                "expected_snapshot_hash": identity["snapshot_hash"],
            }
            authenticated_sample_ref = {
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
        except (
            KeyError,
            StrategyError,
            TypeError,
            ValueError,
            *strategy_contracts_lane._STRATEGY_V2_ARTIFACT_ERRORS,
        ) as exc:
            raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
                "strategy_report_bundle_v2_impact_cube_invalid",
                "最新待判定的 ImpactCube 候选未通过文件、registry、"
                "provenance、producer-run 或 audit 复核；其真实 Pool/"
                "SampleDesign 身份无法确认，平台不会回退到旧 ImpactCube "
                "或 PoolImpact。",
            ) from exc
        if (
            authenticated_pool_ref == expected_pool_ref
            and authenticated_sample_ref == dict(sample_ref)
        ):
            return binding
    if total > strategy_contracts_lane._STRATEGY_REPORT_EVIDENCE_REPLAY_LIMIT:
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_impact_cube_selection_window_exhausted",
            "已完整认证最新 "
            f"{strategy_contracts_lane._STRATEGY_REPORT_EVIDENCE_REPLAY_LIMIT} 个 ImpactCube artifact，"
            "但 registry 仍有更早记录；平台无法证明窗口外不存在与当前 "
            "Pool/SampleDesign 完全一致的 ImpactCube，本次未创建报告计划。",
        )
    return None


def _strategy_report_latest_pool_stability_binding(
    read_runtime: SimpleNamespace,
    *,
    task_id: str,
    impact_cube_ref: Mapping[str, object],
):
    """Select the newest authenticated stability for the exact report cube."""

    records, total = _strategy_report_artifact_window(
        read_runtime,
        task_id=task_id,
        kind=POOL_STABILITY_ARTIFACT_KIND,
        limit=strategy_contracts_lane._STRATEGY_REPORT_POOL_STABILITY_REPLAY_LIMIT,
        unavailable_code=(
            "strategy_report_bundle_v2_pool_stability_registry_unavailable"
        ),
        invalid_code="strategy_report_bundle_v2_pool_stability_invalid",
        label="PoolStability",
    )
    for item in records:
        provenance = item.get("provenance")
        try:
            binding = load_strategy_pool_stability_artifact(
                read_runtime,
                task_id=task_id,
                artifact_id=item.get("id"),
                expected_artifact_content_hash=item.get("content_hash"),
                expected_stability_id=(
                    provenance.get("stability_id")
                    if isinstance(provenance, Mapping)
                    else None
                ),
                expected_stability_content_hash=(
                    provenance.get("stability_content_hash")
                    if isinstance(provenance, Mapping)
                    else None
                ),
            )
            source_ref = binding.stability["source_bindings"]["impact_cube"]
        except (
            KeyError,
            StrategyError,
            TypeError,
            ValueError,
            *strategy_contracts_lane._STRATEGY_V2_ARTIFACT_ERRORS,
        ) as exc:
            raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
                "strategy_report_bundle_v2_pool_stability_invalid",
                "最新待判定的 PoolStability artifact 未通过文件、registry、"
                "provenance、producer-run、唯一 audit 或 embedded "
                "ImpactCube 复核；其真实来源无法确认，平台不会回退到旧"
                "稳定性证据。",
            ) from exc
        if source_ref == dict(impact_cube_ref):
            return binding
    if total > strategy_contracts_lane._STRATEGY_REPORT_POOL_STABILITY_REPLAY_LIMIT:
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_pool_stability_selection_window_exhausted",
            "已完整认证最新 "
            f"{strategy_contracts_lane._STRATEGY_REPORT_POOL_STABILITY_REPLAY_LIMIT} 个 "
            "PoolStability artifact，但 registry 仍有更早记录；平台无法"
            "证明窗口外不存在与当前 exact ImpactCube 一致的稳定性证据，"
            "本次未创建报告计划。",
        )
    return None


def _strategy_report_latest_pool_impact_binding(
    read_runtime: SimpleNamespace,
    *,
    task_id: str,
    pool,
):
    same_kind, total = _strategy_report_artifact_window(
        read_runtime,
        task_id=task_id,
        kind=POOL_IMPACT_ARTIFACT_KIND,
        limit=strategy_contracts_lane._STRATEGY_REPORT_EVIDENCE_REPLAY_LIMIT,
        unavailable_code=("strategy_report_bundle_v2_pool_impact_registry_unavailable"),
        invalid_code="strategy_report_bundle_v2_pool_impact_invalid",
        label="PoolImpact",
    )
    for item in same_kind:
        provenance = item.get("provenance")
        try:
            binding = load_historical_strategy_pool_impact_artifact(
                read_runtime,
                task_id=task_id,
                artifact_id=item.get("id"),
                expected_artifact_content_hash=item.get("content_hash"),
                expected_assessment_id=(
                    provenance.get("assessment_id")
                    if isinstance(provenance, Mapping)
                    else None
                ),
                expected_assessment_content_hash=(
                    provenance.get("assessment_content_hash")
                    if isinstance(provenance, Mapping)
                    else None
                ),
            )
        except (
            StrategyError,
            TypeError,
            ValueError,
            *strategy_contracts_lane._STRATEGY_V2_ARTIFACT_ERRORS,
        ) as exc:
            raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
                "strategy_report_bundle_v2_pool_impact_invalid",
                "最新待判定的 PoolImpact 未通过历史安全的文件、registry、"
                "provenance、Pool 或样本绑定复核；其真实身份无法确认，"
                "平台不会回退到旧影响证据。",
            ) from exc
        if (
            binding.stage != "development_backtest"
            or binding.pool.artifact_id != pool.artifact_id
            or binding.pool.artifact_content_hash != pool.artifact_content_hash
            or binding.pool.pool != pool.pool
        ):
            continue
        return binding
    if total > strategy_contracts_lane._STRATEGY_REPORT_EVIDENCE_REPLAY_LIMIT:
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_pool_impact_selection_window_exhausted",
            "已检查最新 "
            f"{strategy_contracts_lane._STRATEGY_REPORT_EVIDENCE_REPLAY_LIMIT} 个 PoolImpact artifact，"
            "但 registry 仍有更早记录；平台无法证明窗口外不存在当前 "
            "Pool revision/snapshot 的精确 development 证据，"
            "本次未创建报告计划。",
        )
    raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
        "strategy_report_bundle_v2_pool_impact_required",
        "当前非空 Strategy Pool 没有同 revision/snapshot 的 development "
        "PoolImpact；请先单独完成影响测算。",
    )


def _strategy_report_optional_model_evidence(
    read_runtime: SimpleNamespace,
    *,
    task_id: str,
    sample_ref: Mapping[str, object],
) -> tuple[object | None, dict[str, object] | None]:
    records, _total = _strategy_report_artifact_window(
        read_runtime,
        task_id=task_id,
        kind=MODEL_EVIDENCE_V2_ARTIFACT_KIND,
        limit=1,
        unavailable_code=(
            "strategy_report_bundle_v2_optional_evidence_registry_unavailable"
        ),
        invalid_code="strategy_report_bundle_v2_optional_evidence_invalid",
        label="ModelEvidence",
    )
    if not records:
        return None, None
    newest = records[0]
    provenance = newest.get("provenance")
    if not isinstance(provenance, Mapping):
        strategy_report_lane._raise_corrupt_report_optional("ModelEvidence")
    try:
        binding = load_strategy_model_evidence_v2_artifact(
            read_runtime,
            task_id=task_id,
            artifact_id=newest.get("id"),
            expected_artifact_content_hash=newest.get("content_hash"),
            expected_bundle_id=provenance.get("bundle_id"),
            expected_bundle_content_hash=provenance.get("bundle_content_hash"),
            sample_design_ref=provenance.get("sample_design_ref"),
        )
    except (
        ModelingError,
        StrategyError,
        TypeError,
        ValueError,
        *strategy_contracts_lane._STRATEGY_V2_ARTIFACT_ERRORS,
    ) as exc:
        strategy_report_lane._raise_corrupt_report_optional("ModelEvidence", cause=exc)
    reference = {
        "artifact_id": binding.artifact_id,
        "expected_artifact_content_hash": binding.artifact_content_hash,
        "expected_bundle_id": binding.bundle["bundle_id"],
        "expected_bundle_content_hash": binding.bundle["content_hash"],
    }
    if _strategy_report_sample_ref(binding.sample_design_binding) != dict(sample_ref):
        return None, None
    return binding, reference


def _strategy_report_optional_training_evidence(
    read_runtime: SimpleNamespace,
    *,
    task_id: str,
    sample_ref: Mapping[str, object],
) -> tuple[object | None, dict[str, object] | None]:
    records, _total = _strategy_report_artifact_window(
        read_runtime,
        task_id=task_id,
        kind=MODELING_TRAINING_EVIDENCE_ARTIFACT_KIND,
        limit=1,
        unavailable_code=(
            "strategy_report_bundle_v2_optional_evidence_registry_unavailable"
        ),
        invalid_code="strategy_report_bundle_v2_optional_evidence_invalid",
        label="training evidence",
    )
    if not records:
        return None, None
    newest = records[0]
    try:
        reference = _strategy_report_training_ref(
            read_runtime,
            task_id=task_id,
            record=newest,
        )
        binding = load_modeling_training_evidence_artifacts(
            read_runtime,
            task_id=task_id,
            **reference,
        )
        reference = build_training_evidence_ref(binding)
    except (
        ModelingError,
        StrategyError,
        TypeError,
        ValueError,
        *strategy_contracts_lane._STRATEGY_V2_ARTIFACT_ERRORS,
    ) as exc:
        strategy_report_lane._raise_corrupt_report_optional(
            "training evidence", cause=exc
        )
    if reference["sample_design_ref"] != dict(sample_ref):
        return None, None
    return binding, reference


def _strategy_report_training_ref(
    read_runtime: SimpleNamespace,
    *,
    task_id: str,
    record: Mapping,
) -> dict[str, object]:
    provenance = record.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("training evidence provenance is invalid")
    sample_ref = _strategy_report_sample_ref_from_registry(
        read_runtime,
        task_id=task_id,
        membership_artifact_id=provenance.get("sample_membership_artifact_id"),
        bundle_artifact_id=provenance.get("sample_bundle_artifact_id"),
    )
    return {
        "sample_design_ref": sample_ref,
        "model_binary_artifact_id": provenance.get("model_binary_artifact_id"),
        "expected_model_binary_artifact_content_hash": provenance.get(
            "model_binary_artifact_content_hash"
        ),
        "evidence_artifact_id": record.get("id"),
        "expected_evidence_artifact_content_hash": record.get("content_hash"),
        "expected_experiment_id": provenance.get("experiment_id"),
        "expected_model_artifact_id": provenance.get("model_artifact_id"),
        "expected_evidence_id": provenance.get("evidence_id"),
        "expected_evidence_content_hash": provenance.get("evidence_content_hash"),
    }


def _strategy_report_sample_ref_from_registry(
    read_runtime: SimpleNamespace,
    *,
    task_id: str,
    membership_artifact_id: object,
    bundle_artifact_id: object,
) -> dict[str, object]:
    membership = read_runtime.task_artifacts.get_for_task(
        task_id,
        membership_artifact_id,
    )
    bundle = read_runtime.task_artifacts.get_for_task(
        task_id,
        bundle_artifact_id,
    )
    if (
        not isinstance(membership, Mapping)
        or membership.get("kind")
        not in {
            SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND,
            SAMPLE_DESIGN_V2_NATIVE_MEMBERSHIP_ARTIFACT_KIND,
        }
        or not isinstance(bundle, Mapping)
        or bundle.get("kind") != SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND
    ):
        raise ValueError("training evidence sample artifact pair is missing")
    provenance = bundle.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("training evidence sample bundle provenance is invalid")
    return {
        "membership_artifact_id": membership.get("id"),
        "expected_membership_artifact_content_hash": membership.get("content_hash"),
        "bundle_artifact_id": bundle.get("id"),
        "expected_bundle_artifact_content_hash": bundle.get("content_hash"),
        "expected_bundle_id": provenance.get("bundle_id"),
        "expected_sample_design_id": provenance.get("sample_design_id"),
        "expected_sample_design_content_hash": provenance.get(
            "sample_design_content_hash"
        ),
    }


def _strategy_report_optional_score_evidence(
    read_runtime: SimpleNamespace,
    *,
    task_id: str,
    sample_ref: Mapping[str, object],
    training_ref: Mapping[str, object] | None,
) -> tuple[object | None, dict[str, object] | None]:
    records, _total = _strategy_report_artifact_window(
        read_runtime,
        task_id=task_id,
        kind=MODEL_SCORE_EVIDENCE_ARTIFACT_KIND,
        limit=1,
        unavailable_code=(
            "strategy_report_bundle_v2_optional_evidence_registry_unavailable"
        ),
        invalid_code="strategy_report_bundle_v2_optional_evidence_invalid",
        label="score evidence",
    )
    if not records:
        return None, None
    newest = records[0]
    provenance = newest.get("provenance")
    if not isinstance(provenance, Mapping):
        strategy_report_lane._raise_corrupt_report_optional("score evidence")
    reference = {
        "evidence_artifact_id": newest.get("id"),
        "expected_evidence_artifact_content_hash": newest.get("content_hash"),
        "score_vector_artifact_id": provenance.get("score_vector_artifact_id"),
        "expected_score_vector_artifact_content_hash": provenance.get(
            "score_vector_artifact_content_hash"
        ),
    }
    try:
        binding = load_model_score_evidence_artifacts(
            read_runtime,
            task_id=task_id,
            **reference,
        )
    except (
        ModelingError,
        StrategyError,
        TypeError,
        ValueError,
        *strategy_contracts_lane._STRATEGY_V2_ARTIFACT_ERRORS,
    ) as exc:
        strategy_report_lane._raise_corrupt_report_optional("score evidence", cause=exc)
    bound_training_ref = build_training_evidence_ref(binding.training)
    if bound_training_ref["sample_design_ref"] != dict(sample_ref) or (
        training_ref is not None and bound_training_ref != dict(training_ref)
    ):
        return None, None
    return binding, {
        "evidence_artifact_id": binding.evidence_record["id"],
        "expected_evidence_artifact_content_hash": binding.evidence_record[
            "content_hash"
        ],
        "score_vector_artifact_id": binding.vector_record["id"],
        "expected_score_vector_artifact_content_hash": binding.vector_record[
            "content_hash"
        ],
    }


def _strategy_report_identity(
    runtime: contracts_lane.DriverTurnRuntime,
    *,
    task_id: str,
    candidate_pool,
) -> dict[str, str] | None:
    repository = StrategyRepository(runtime.settings.db_path)
    try:
        with repository.transaction() as conn:
            authenticated = (
                authenticate_strategy_report_identity_for_pool_on_connection(
                    repository,
                    conn,
                    task_id=task_id,
                    candidate_pool=candidate_pool,
                )
            )
    except Exception as exc:
        raise strategy_contracts_lane._StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_strategy_identity_invalid",
            "当前 Strategy Pool 的物化策略身份、不可变账本或生命周期"
            "未通过完整性复核；本次未创建计划。",
        ) from exc
    return None if authenticated is None else dict(authenticated["identity"])


def _strategy_pool_impact_column(
    inputs: Mapping,
    *,
    field: str,
    role: str,
    columns: tuple[str, ...],
    field_roles: Mapping,
) -> str | None:
    """Prefer an explicit validated column, else require a unique semantic role."""

    explicit = inputs.get(field)
    if explicit is not None:
        if not isinstance(explicit, str) or explicit not in columns:
            raise StrategySetupError(f"影响测算显式字段 {field} 不在当前活动数据集中。")
        return explicit
    matches = [
        column
        for column, assigned_role in field_roles.items()
        if assigned_role == role and column in columns
    ]
    if len(matches) > 1:
        raise StrategySetupError(
            f"DataWorkspace 有多个 `{role}` 语义字段：{'、'.join(sorted(matches))}；"
            f"请在请求中明确指定 {field}，平台不会任意选择。"
        )
    return matches[0] if matches else None


def _strategy_pool_entries(pool: Mapping) -> list[Mapping]:
    entries = pool.get("entries")
    if not isinstance(entries, Sequence) or isinstance(
        entries, str | bytes | bytearray
    ):
        raise StrategySetupError("当前 Strategy Pool entries 无效。")
    if any(not isinstance(entry, Mapping) for entry in entries):
        raise StrategySetupError("当前 Strategy Pool entry 结构无效。")
    return list(entries)


def _strategy_pool_rule_id(pool: Mapping, identifier: str) -> str:
    matches = [
        entry
        for entry in _strategy_pool_entries(pool)
        if identifier in {str(entry.get("entry_id")), str(entry.get("rule_id"))}
    ]
    if len(matches) != 1:
        raise StrategySetupError(
            f"当前 Strategy Pool 中没有唯一匹配的 rule_id/entry_id：{identifier}。"
        )
    rule_id = matches[0].get("rule_id")
    if not isinstance(rule_id, str) or not rule_id:
        raise StrategySetupError("当前 Strategy Pool entry 缺少完整 rule_id。")
    return rule_id


def _strategy_pool_complete_rule_order(
    pool: Mapping,
    ordered_ids: object,
) -> list[str]:
    entries = _strategy_pool_entries(pool)
    if not isinstance(ordered_ids, Sequence) or isinstance(
        ordered_ids, str | bytes | bytearray
    ):
        raise StrategySetupError("Strategy Pool reorder 必须提供完整 ID 列表。")
    resolved = [_strategy_pool_rule_id(pool, str(item)) for item in ordered_ids]
    current_rule_ids = [str(entry.get("rule_id") or "") for entry in entries]
    if (
        len(resolved) != len(current_rule_ids)
        or len(set(resolved)) != len(resolved)
        or set(resolved) != set(current_rule_ids)
    ):
        raise StrategySetupError(
            "Strategy Pool reorder 必须提供当前全部 rule_id/entry_id 的完整、无重复排列；"
            "遗漏 ID 不会被解释为删除。"
        )
    return resolved


def _strategy_dataset_context(
    runtime: contracts_lane.DriverTurnRuntime,
    task: TaskRecord,
    *,
    require_target: bool = True,
):
    backend, registry = data_context_lane._modeling_data_runtime(runtime.settings)
    return build_strategy_dataset_context(
        registry,
        backend,
        task.id,
        task.source_dir,
        target_col=getattr(task, "target_col", "") or None,
        require_target=require_target,
    )


def _strategy_dataset_preview(
    runtime: contracts_lane.DriverTurnRuntime, task: TaskRecord
):
    backend, registry = data_context_lane._modeling_data_runtime(runtime.settings)
    return preview_strategy_dataset_context(
        registry,
        backend,
        task.id,
        task.source_dir,
        target_col=getattr(task, "target_col", "") or None,
    )


def _strategy_target_nan_stats(
    runtime: contracts_lane.DriverTurnRuntime, context
) -> tuple[int, int]:
    backend, registry = data_context_lane._modeling_data_runtime(runtime.settings)
    target_col = str(context.target_col or "").strip()
    if not target_col:
        raise StrategySetupError("当前策略操作需要明确的二元目标列。")
    path = registry.resolve_path(context.dataset_id)
    try:
        frame = backend.read_frame(path, columns=[target_col])
        mask = nan_label_mask(frame, target_col)
    except Exception as exc:
        raise StrategySetupError(
            f"目标列 `{target_col}` 必须只包含 0/1 和可显式处理的空标签。"
        ) from exc
    return int(len(frame)), int(mask.sum())


def _strategy_request_allowed_columns(preview) -> tuple[str, ...]:
    if preview is None:
        return ()
    # The observed target is evidence, never a deployable strategy feature or
    # an input to an LLM-authored profit contract.
    return tuple(column for column in preview.columns if column != preview.target_col)


def _strategy_request_requires_dataset(
    draft: CompiledStrategyRequestDraft,
) -> bool:
    if isinstance(draft, StandardWorkflowRequestDraft):
        migrated = migrated_workflow_requirements(
            draft.workflow,
            draft.workflow_inputs,
        )
        if migrated is not None:
            return migrated[0]
        if draft.workflow in {
            *strategy_contracts_lane._STRATEGY_POOL_WORKFLOWS,
            "strategy_project_context",
            "strategy_model_evidence_v2",
            "strategy_report_bundle_v2",
            "strategy_impact_cube",
            "strategy_pool_stability",
            "strategy_pool_apply",
            "strategy_pool_materialize",
            "strategy_pool_validation",
            "automatic_tree_leaf_materialization",
            "interactive_tree_split_search",
            "interactive_tree_auto_continuation",
            "interactive_tree_revision",
            "cross_matrix_cell_selection",
            "voting_candidate_search",
            "voting_candidate_build_from_search",
            "voting_candidate_build",
            "cross_matrix_candidate_search",
            "cross_matrix_candidate_build_from_search",
            "cross_rule_search",
            "cross_rule_candidate_build_from_search",
        }:
            return False
        return True
    return not (draft.strategy_spec is None and draft.operation == "report")


def _strategy_request_requires_target(
    draft: CompiledStrategyRequestDraft,
) -> bool:
    if isinstance(draft, StandardWorkflowRequestDraft):
        migrated = migrated_workflow_requirements(
            draft.workflow,
            draft.workflow_inputs,
        )
        if migrated is not None:
            return migrated[1]
        if draft.workflow == "strategy_project_context":
            return False
        return draft.workflow in {
            "strategy_sample_design",
            "strategy_sample_design_v2",
            "automatic_tree_candidate_build",
            "cross_matrix_analysis",
            "strategy_pool_impact",
        }
    if draft.operation in {"apply", "report", "monitor"}:
        return False
    if draft.operation == "develop" and draft.strategy_spec is not None:
        return False
    return True


def _strategy_request_requires_complete_labels(
    draft: CompiledStrategyRequestDraft,
) -> bool:
    """Whether execution would otherwise exclude missing supervision rows."""

    if isinstance(draft, StandardWorkflowRequestDraft):
        migrated = migrated_workflow_requirements(
            draft.workflow,
            draft.workflow_inputs,
        )
        if migrated is not None:
            return migrated[2]
        if draft.workflow == "strategy_project_context":
            return False
        return draft.workflow in {
            "strategy_sample_design",
            "strategy_sample_design_v2",
            "automatic_tree_candidate_build",
            "cross_matrix_analysis",
            "strategy_pool_impact",
        }
    if draft.operation in {"apply", "report", "monitor"}:
        return False
    if draft.operation == "develop" and draft.strategy_spec is not None:
        return False
    return True


def _strategy_slots_with_drop_nan(slots: dict, confirmed: bool) -> dict:
    if not confirmed:
        return slots
    return {**slots, "drop_nan_labels": True}


def _strategy_contract_from_draft(draft: StrategyRequestDraft) -> StrategyTaskInput:
    profit = None
    if draft.profit is not None:
        profit = StrategyProfitInput(**dict(draft.profit))
    return StrategyTaskInput(
        strategy_type=draft.strategy_type,
        objective=draft.objective or "",
        max_bad_rate=draft.max_bad_rate,
        min_approval_rate=draft.min_approval_rate,
        baseline_strategy_id=draft.baseline_strategy_id,
        profit=profit,
    )


def _strategy_request_success_criteria(
    draft: StrategyRequestDraft,
) -> list[dict] | None:
    if draft.strategy_type not in {"approval", "reject"}:
        return None
    criteria: list[dict] = []
    if draft.max_bad_rate is not None:
        criteria.append({"metric": "approved_bad_rate", "max": draft.max_bad_rate})
    if draft.min_approval_rate is not None:
        criteria.append({"metric": "approval_rate", "min": draft.min_approval_rate})
    return criteria or None
