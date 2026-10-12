"""Explicit exact-identity Agent commands without an LLM schema extension."""

from __future__ import annotations

import re
from types import SimpleNamespace

from marvis.packs.strategy import tools
from marvis.packs.strategy.asset_stability_tools import (
    FREEZE_KIND,
    candidate_asset_stability_catalog,
    run_freeze_candidate_asset_stability,
    run_measure_candidate_asset_stability,
)
from marvis.packs.strategy.errors import StrategyError

_PREFIXES = ("列出候选资产稳定性来源", "冻结候选资产稳定性", "测量候选资产稳定性")
_FREEZE = re.compile(
    r"^冻结候选资产稳定性\s+([0-9a-f]{64})\s+([0-9a-f]{64})\s+(validation|oot|validation,oot|oot,validation)$"
)
_MEASURE = re.compile(r"^测量候选资产稳定性\s+([0-9a-f]{64})$")


def is_asset_stability_command(text):
    return isinstance(text, str) and text.strip().startswith(_PREFIXES)


def handle_asset_stability_command(runtime, repo, task, text):
    text = str(text or "").strip()
    if not is_asset_stability_command(text):
        return None
    if task.task_type != "strategy":
        return None
    repo.add_agent_message(
        task.id,
        role="user",
        stage="chat",
        content=text,
        metadata={"intent": "asset_independent_stability"},
    )
    from marvis.agent.turn_handlers.shared import _active_plan

    active = _active_plan(runtime.plan_repo, task.id)
    if active is not None:
        return _response(
            repo,
            task,
            "clarification_required",
            "请先完成或取消当前计划，再冻结独立候选资产。",
            {"code": "asset_stability_active_plan"},
        )
    ctx = SimpleNamespace(
        task_id=task.id,
        workspace=runtime.settings.workspace,
        datasets_root=runtime.settings.datasets_dir,
    )
    pack = tools._runtime(ctx)
    try:
        catalog = candidate_asset_stability_catalog(pack, task_id=task.id)
        if text == _PREFIXES[0]:
            return _response(
                repo,
                task,
                "done",
                "已列出可认证的候选资产与样本设计。请明确指定来源后冻结。",
                {"asset_stability_catalog": catalog},
            )
        freeze = _FREEZE.fullmatch(text)
        measure = _MEASURE.fullmatch(text)
        if freeze:
            source_id, sample_id, partitions = freeze.groups()
            sources = [
                x for x in catalog["sources"] if x["ref"]["artifact_id"] == source_id
            ]
            samples = [
                x
                for x in catalog["samples"]
                if x["ref"]["bundle_artifact_id"] == sample_id
            ]
            if len(sources) != 1 or len(samples) != 1:
                raise StrategyError(
                    "explicit source or sample identity is not current/authenticated"
                )
            output = run_freeze_candidate_asset_stability(
                {
                    "source_ref": sources[0]["ref"],
                    "sample_design_ref": samples[0]["ref"],
                    "partitions": partitions.split(","),
                },
                ctx,
                pack,
            )
            message = "已冻结明确指定的候选和样本版本。下一步可测量此冻结证据。"
        elif measure:
            frozen = [
                x
                for x in catalog["evidence"]
                if x["kind"] == FREEZE_KIND
                and x["ref"]["artifact_id"] == measure.group(1)
            ]
            if len(frozen) != 1:
                raise StrategyError("explicit frozen identity is not authenticated")
            output = run_measure_candidate_asset_stability(
                {"frozen_ref": frozen[0]["ref"]}, ctx, pack
            )
            message = "独立分区稳定性测量完成；development 基线与来源引用保持固定。候选状态未改变。"
        else:
            return _response(
                repo,
                task,
                "clarification_required",
                "请使用完整明确的命令：冻结候选资产稳定性 <候选artifact_id> <样本bundle_artifact_id> validation,oot；随后测量候选资产稳定性 <冻结artifact_id>。",
                {"code": "asset_stability_exact_ids_required"},
            )
        return _response(repo, task, "done", message, {"asset_stability": output})
    except (StrategyError, ValueError, KeyError, TypeError) as exc:
        return _response(
            repo,
            task,
            "clarification_required",
            f"独立候选稳定性未执行或未发布：{exc}",
            {"code": "asset_stability_source_rejected"},
        )


def _response(repo, task, status, message, metadata):
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=message,
        metadata={"intent": "asset_independent_stability", **metadata},
    )
    return {
        "task_id": task.id,
        "status": status,
        "messages": repo.list_agent_messages(task.id),
        **metadata,
    }
