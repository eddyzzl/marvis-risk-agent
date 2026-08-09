"""Strategy Pool execution, replay, stability, and impact presenters."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote

from marvis.agent.presenters._shared import (
    format_number as _num,
    format_percent as _pct,
    format_value as _fmt,
)
from marvis.repositories.task_artifacts import stable_task_artifact_id


def _render_strategy_pool_mutation(o: dict):
    entries = [entry for entry in (o.get("entries") or []) if isinstance(entry, dict)]
    artifacts = [item for item in (o.get("artifacts") or []) if isinstance(item, dict)]
    revision = o.get("revision")
    snapshot_hash = str(o.get("snapshot_hash") or "")
    operation = str(o.get("operation") or "update")
    text = (
        f"**Strategy Pool 已更新**：操作 `{operation}`，Pool `{o.get('pool_id', '')}`，"
        f"revision {revision}，snapshot hash `{snapshot_hash}`。"
        f"当前完整有序条目 **{len(entries)}** 条；所有候选证据保持 "
        "`development / unvalidated`。这是 task 内 draft Pool，**未采纳、未部署**。"
    )
    if operation == "insert_candidate_before_entries":
        text += " Voting 已放在所选成员中最早位置之前，原成员保留为后续规则。"
    elif operation == "replace_entries_with_candidate":
        text += " Voting 已在一个原子 revision 中替代所选成员。"
    links = [
        f"[{str(item.get('filename') or item.get('kind') or '下载')}]"
        f"({str(item.get('download_url'))})"
        for item in artifacts
        if item.get("download_url")
    ]
    if links:
        text += "\n\n**Pool revision artifact**：" + "；".join(links)

    rows = []
    for index, entry in enumerate(entries, start=1):
        action = entry.get("action") if isinstance(entry.get("action"), dict) else {}
        source = entry.get("source") if isinstance(entry.get("source"), dict) else {}
        effect = entry.get("effect") if isinstance(entry.get("effect"), dict) else {}
        asset_ref = (
            entry.get("candidate_asset_ref")
            if isinstance(entry.get("candidate_asset_ref"), dict)
            else {}
        )
        asset_id = (
            entry.get("candidate_asset_id")
            or entry.get("asset_id")
            or asset_ref.get("asset_id")
            or source.get("asset_id")
            or ""
        )
        effect_stage = str(
            entry.get("effect_stage") or source.get("effect_stage") or "development"
        )
        validation_status = str(
            entry.get("validation_status")
            or source.get("validation_status")
            or "unvalidated"
        )
        rows.append(
            [
                str(index),
                str(entry.get("rule_id") or ""),
                str(entry.get("entry_id") or ""),
                str(asset_id),
                str(action.get("type") or action.get("value") or ""),
                _pct(effect.get("selected_share")),
                _pct(effect.get("bad_rate")),
                _num(effect.get("lift")),
                f"{effect_stage} / {validation_status}",
            ]
        )
    tables = [
        {
            "title": "Strategy Pool 完整顺序",
            "columns": [
                "#",
                "rule_id",
                "entry_id",
                "candidate_asset_id",
                "action",
                "selected_share",
                "bad_rate",
                "lift",
                "evidence status",
            ],
            "rows": rows,
        }
    ]
    return text, tables


def _render_compile_strategy_pool(o: dict):
    spec = o.get("strategy_spec") if isinstance(o.get("strategy_spec"), dict) else {}
    rules = [rule for rule in (spec.get("rules") or []) if isinstance(rule, dict)]
    artifacts = [item for item in (o.get("artifacts") or []) if isinstance(item, dict)]
    text = (
        f"**Strategy Pool 编译完成**：Pool `{o.get('pool_id', '')}` revision "
        f"{o.get('revision')} 已只读编译为 canonical `StrategySpec`；design hash "
        f"`{o.get('design_hash', '')}`，snapshot hash `{o.get('snapshot_hash', '')}`。"
        "该结果只是**只读草案**，未创建已采纳策略，**未采纳、未部署**。"
    )
    requirements = o.get("requirements") or []
    if requirements:
        rendered_requirements = "；".join(
            str(item.get("message") or item.get("code") or item)
            if isinstance(item, dict)
            else str(item)
            for item in requirements
        )
        text += f"\n\n**尚待满足的要求**：{rendered_requirements}"
    links = [
        f"[{str(item.get('filename') or item.get('kind') or '下载')}]"
        f"({str(item.get('download_url'))})"
        for item in artifacts
        if item.get("download_url")
    ]
    if links:
        text += "\n\n**来源 Pool artifact**：" + "；".join(links)
    return text, [
        {
            "title": "编译后的 StrategySpec 规则",
            "columns": ["#", "rule_id", "priority", "action", "condition"],
            "rows": [
                [
                    str(index),
                    str(rule.get("rule_id") or ""),
                    _fmt(rule.get("priority")),
                    str(
                        (rule.get("action") or {}).get("type")
                        if isinstance(rule.get("action"), dict)
                        else ""
                    ),
                    str(rule.get("condition") or ""),
                ]
                for index, rule in enumerate(rules, start=1)
            ],
        }
    ]


def _render_materialize_strategy_from_pool(o: dict):
    strategy_ref = (
        o.get("strategy_ref") if isinstance(o.get("strategy_ref"), dict) else {}
    )
    pool_ref = o.get("pool_ref") if isinstance(o.get("pool_ref"), dict) else {}
    requirements = (
        o.get("requirements") if isinstance(o.get("requirements"), dict) else {}
    )
    lifecycle = o.get("lifecycle") if isinstance(o.get("lifecycle"), dict) else {}

    current_status = str(lifecycle.get("current_status") or "unknown")
    current_asset_status = str(lifecycle.get("current_asset_status") or "unknown")
    created_status = str(lifecycle.get("created_status") or "unknown")
    created_asset_status = str(lifecycle.get("created_asset_status") or "unknown")
    blocker_code = str(requirements.get("blocker_code") or "")

    text = (
        "**Strategy Pool 已物化为 canonical Strategy**："
        f"`{strategy_ref.get('strategy_id', '')}`（"
        f"{strategy_ref.get('strategy_type', '')} v"
        f"{strategy_ref.get('version', '')}）。"
        f"来源：Pool revision {pool_ref.get('revision', '')}，"
        f"revision id `{pool_ref.get('revision_id', '')}`。\n\n"
        f"**生命周期**：创建时 {created_status} / {created_asset_status}；"
        f"当前 lifecycle：{current_status} / {current_asset_status}。"
        "**本 Tool 未采纳、未部署**。"
    )

    if blocker_code:
        text += (
            f"\n\n**requirements blocker**：`{blocker_code}`。"
            "当前 Pool 依赖尚不能由 canonical Strategy runtime 安全承载，"
            "因此 DSL 交付、回测和监控 readiness 均为 **blocked**。"
        )
    else:
        text += (
            "\n\n**requirements compatibility**：未发现 runtime blocker。"
            "这只说明当前 requirements 可被持久化，并不替代下游数据、"
            "独立证据或 lifecycle 检查。"
        )

    if current_asset_status == "adopted_local":
        text += (
            "\n\n当前已进入 adopted_local，但这是既有 lifecycle 状态，"
            "不是本 Tool 的动作；监控仍需有效 monitoring plan 和独立执行检查。"
        )
    elif current_status == "draft" and current_asset_status == "draft":
        text += (
            "\n\n当前仍是 draft；进入 adopted_local 前必须完成独立回测证据"
            "和**人工采纳**。监控不能仅凭本次物化结果启动。"
        )
    else:
        text += (
            f"\n\n当前已进入 {current_status} / {current_asset_status}，"
            "但这是既有 lifecycle 状态，不是本 Tool 的动作。后续采纳或监控"
            "必须继续通过各自的人工决策、证据与执行检查。"
        )

    requirement_support = (
        "supported"
        if requirements.get("runtime_requirements_supported") is True
        else "blocked"
    )
    tables = [
        {
            "title": "Strategy 物化身份",
            "columns": ["字段", "值"],
            "rows": [
                ["strategy_id", str(strategy_ref.get("strategy_id") or "")],
                [
                    "strategy_spec_hash",
                    str(strategy_ref.get("strategy_spec_hash") or ""),
                ],
                ["pool_id", str(pool_ref.get("pool_id") or "")],
                ["pool revision", _fmt(pool_ref.get("revision"))],
                ["design_hash", str(o.get("design_hash") or "")],
            ],
        },
        {
            "title": "运行时 requirements 兼容性（非 lifecycle readiness）",
            "columns": ["检查范围", "状态"],
            "rows": [["Strategy runtime requirements", requirement_support]],
        },
    ]
    return text, tables


def _strategy_pool_apply_integrity_failure() -> tuple[str, list[dict]]:
    return (
        "**Strategy Pool 应用结果完整性校验失败**：计划缓存中的 Pool、"
        "派生数据集、逐行分布、requirements、workspace 或 evidence 绑定"
        "不一致，已停止展示结果。请基于当前 Pool 重新执行应用。",
        [],
    )


def _strategy_pool_validation_integrity_failure() -> tuple[str, list[dict]]:
    return (
        "**Strategy Pool 独立样本回放验证结果完整性校验失败**：计划缓存中的"
        " Pool、StrategySampleDesign V2、独立分区、canonical evidence 或"
        " artifact 摘要不一致，已停止展示动作、风险、金额和逐月结果。"
        "请基于当前证据重新执行独立回放。",
        [],
    )


def _validated_strategy_pool_validation_output(
    value: object,
    *,
    trusted_task_id: str | None,
    trusted_inputs: Mapping[str, Any] | None,
    trusted_artifacts: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Authenticate canonical evidence and its exact terminal plan inputs."""

    from marvis.packs.strategy.pool_validation_tools import (
        authenticate_strategy_pool_validation_artifact_record,
        validate_measure_strategy_pool_validation_tool_output,
    )

    if (
        not isinstance(value, Mapping)
        or not isinstance(trusted_task_id, str)
        or not trusted_task_id
        or not isinstance(trusted_inputs, Mapping)
        or not isinstance(trusted_artifacts, Mapping)
        or set(trusted_artifacts) != {"pool_validation"}
    ):
        raise ValueError("Pool validation trusted terminal context is unavailable")
    trusted = trusted_artifacts["pool_validation"]
    if (
        not isinstance(trusted, Mapping)
        or set(trusted) != {"record", "tasks_root"}
        or not isinstance(trusted["record"], Mapping)
        or not isinstance(trusted["tasks_root"], str)
        or not trusted["tasks_root"]
    ):
        raise ValueError("Pool validation registry context is unavailable")
    record = trusted["record"]
    artifact_id = record.get("id")
    if not isinstance(artifact_id, str) or not artifact_id:
        raise ValueError("Pool validation registry artifact id is unavailable")
    output = validate_measure_strategy_pool_validation_tool_output(
        value,
        expected_task_id=trusted_task_id,
        expected_artifact_id=artifact_id,
    )
    authenticated = authenticate_strategy_pool_validation_artifact_record(
        task_id=trusted_task_id,
        record=record,
        evidence=output["evidence"],
        tasks_root=trusted["tasks_root"],
    )
    if authenticated != output["evidence"]:
        raise ValueError("Pool validation registered evidence changed")
    evidence = output["evidence"]
    identity = evidence["identity"]
    sources = evidence["source_bindings"]
    sample = sources["sample_design_v2"]
    expected_inputs = {
        "strategy_type": identity["strategy_type"],
        "partition": evidence["partition"],
        "pool_ref": {
            "artifact_id": sources["pool_artifact"]["artifact_id"],
            "expected_artifact_content_hash": sources["pool_artifact"][
                "artifact_content_hash"
            ],
            "expected_pool_id": identity["pool_id"],
            "expected_revision": identity["revision"],
            "expected_revision_id": identity["revision_id"],
            "expected_snapshot_hash": identity["snapshot_hash"],
        },
        "sample_design_ref": {
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
            "expected_sample_design_content_hash": sample["sample_design_content_hash"],
        },
        "population": "risk",
        "comparison_mode": "absolute",
    }
    if dict(trusted_inputs) != expected_inputs:
        raise ValueError("Pool validation terminal inputs changed")
    return output


def _replay_amount_text(effect: object, name: str, field: str) -> str:
    if not isinstance(effect, Mapping):
        return "n/a"
    amounts = effect.get("amounts")
    if not isinstance(amounts, Mapping):
        return "n/a"
    item = amounts.get(name)
    if not isinstance(item, Mapping) or item.get("status") != "available":
        return "n/a"
    value = item.get(field)
    return _pct(value) if field.endswith("rate") else _num(value)


def _replay_summary_row(scope: str, effect: Mapping[str, Any]) -> list[str]:
    return [
        scope,
        _num(effect.get("population_count")),
        _num(effect.get("labelled_count")),
        _pct(effect.get("bad_rate")),
        _replay_amount_text(effect, "loan_amount", "sum"),
        _replay_amount_text(effect, "overdue_amount", "sum"),
        _replay_amount_text(effect, "paired", "overdue_rate"),
    ]


def _render_measure_strategy_pool_validation(
    o: dict,
    *,
    trusted_task_id: str | None,
    trusted_inputs: Mapping[str, Any] | None,
    trusted_artifacts: Mapping[str, Any] | None = None,
):
    """Render independent replay evidence without inventing drift semantics."""

    try:
        o = _validated_strategy_pool_validation_output(
            o,
            trusted_task_id=trusted_task_id,
            trusted_inputs=trusted_inputs,
            trusted_artifacts=trusted_artifacts,
        )
    except Exception:
        return _strategy_pool_validation_integrity_failure()

    evidence = o["evidence"]
    if evidence["schema_version"] == "strategy.pool-validation-evidence.v2":
        try:
            return _render_typed_strategy_pool_validation(o)
        except Exception:
            return _strategy_pool_validation_integrity_failure()
    identity = evidence["identity"]
    population = evidence["population_metrics"]
    overall = evidence["overall"]
    overall_effect = overall["effect"]
    actions = overall["actions"]["breakdown"]
    monthly = evidence["monthly"]
    periods = monthly["periods"] if monthly["status"] == "available" else []
    month_text = (
        f"**{len(periods)}** 个月"
        if monthly["status"] == "available"
        else "月份维度不可用"
    )
    artifact = o["artifact"]
    text = (
        f"**Strategy Pool 独立样本回放验证完成**：已在 "
        f"`{evidence['partition']}` 的 `risk` 独立分区回放当前 "
        f"`{identity['strategy_type']}` Pool `{identity['pool_id']}` "
        f"revision {identity['revision']}。\n"
        f"- independent replay evidence 覆盖 **{population['population_count']}** "
        f"行，其中 **{population['labelled_count']}** 行进入风险分母、"
        f"**{population['unlabelled_count']}** 行保留但不进入风险分母；"
        f"{month_text}。\n"
        "- 这里只呈现该独立分区实际观察到的动作、风险、金额与逐月回放证据，"
        "不外推为未计算的结论。\n"
        "- 本步骤不会修改 Pool、不会创建策略，也不晋级、不采纳、不部署。\n\n"
        f"**独立回放 evidence artifact**：[{artifact['artifact_id']}]"
        f"({artifact['download_url']})"
    )
    if o["warnings"]:
        text += "\n\n**证据提醒**：" + "；".join(o["warnings"])

    action_rows = [
        [
            str(item["action"]),
            _num(item["count"]),
            _pct(item["rate"]),
            _num(item["bad_count"]),
            _pct(item["bad_rate"]),
        ]
        for item in actions
    ]
    monthly_rows = []
    for item in periods:
        effect = item["effect"]
        metrics = item["actions"]["metrics"]
        monthly_rows.append(
            [
                str(item["period"]),
                _num(effect["population_count"]),
                _num(effect["labelled_count"]),
                _pct(metrics["approve_rate"]),
                _pct(metrics["reject_rate"]),
                _pct(metrics["review_rate"]),
                _pct(effect["bad_rate"]),
                _replay_amount_text(effect, "loan_amount", "sum"),
                _replay_amount_text(effect, "overdue_amount", "sum"),
                _replay_amount_text(effect, "paired", "overdue_rate"),
            ]
        )
    return text, [
        {
            "title": "独立回放总体风险与金额",
            "columns": [
                "范围",
                "样本数",
                "有标签样本",
                "总体坏率",
                "放款金额",
                "逾期金额",
                "逾期金额率",
            ],
            "rows": [_replay_summary_row("overall", overall_effect)],
        },
        {
            "title": "独立回放总体动作",
            "columns": ["动作", "样本数", "占比", "坏样本数", "坏率"],
            "rows": action_rows,
        },
        {
            "title": "独立回放逐月证据",
            "columns": [
                "月份",
                "样本数",
                "有标签样本",
                "通过占比",
                "拒绝占比",
                "复核占比",
                "总体坏率",
                "放款金额",
                "逾期金额",
                "逾期金额率",
            ],
            "rows": monthly_rows,
        },
    ]


def _render_typed_strategy_pool_validation(
    output: Mapping[str, Any],
) -> tuple[str, list[dict]]:
    evidence = output["evidence"]
    identity = evidence["identity"]
    population = evidence["population_metrics"]
    typed = evidence["typed_backtest"]
    strategy_type = identity["strategy_type"]
    labels = {
        "limit": ("额度", "assigned_limit"),
        "pricing": ("定价", "assigned_rate"),
        "segmentation": ("分群", "segment"),
    }
    label, value_field = labels[strategy_type]
    monthly = evidence["monthly"]
    periods = monthly["periods"] if monthly["status"] == "available" else []
    month_text = (
        f"**{len(periods)}** 个月"
        if monthly["status"] == "available"
        else "月份维度不可用"
    )
    artifact = output["artifact"]
    text = (
        f"**Strategy Pool 独立样本回放验证完成**：已在 "
        f"`{evidence['partition']}` 的 `risk` 独立分区回放当前 "
        f"`{strategy_type}` Pool `{identity['pool_id']}` revision "
        f"{identity['revision']}。\n"
        f"- independent replay evidence 覆盖 "
        f"**{population['population_count']}** 行，其中 "
        f"**{population['labelled_count']}** 行进入风险分母、"
        f"**{population['unlabelled_count']}** 行保留但不进入风险分母；"
        f"{month_text}。\n"
        f"- 结果保留原生{label}输出及其分布、标签风险和可用经济指标，"
        "不把它改写成审批动作，也不外推为 PSI、稳定性或漂移结论。\n"
        "- 本步骤不会修改 Pool、不会创建策略，也不晋级、不采纳、不部署。\n\n"
        f"**独立回放 evidence artifact**：[{artifact['artifact_id']}]"
        f"({artifact['download_url']})"
    )
    if output["warnings"]:
        text += "\n\n**证据提醒**：" + "；".join(output["warnings"])

    value_formatter = _pct if strategy_type == "pricing" else _num
    distribution_rows = [
        [
            (
                str(item[value_field])
                if strategy_type == "segmentation"
                else value_formatter(item[value_field])
            ),
            _num(item["count"]),
            _pct(item["share"]),
            _num(item["labeled_count"]),
            _num(item["bad_count"]),
            _pct(item["bad_rate"]),
        ]
        for item in typed["breakdown"]
    ]
    metric_rows = [
        [key, _typed_validation_metric_text(key, value)]
        for key, value in sorted(typed["metrics"].items())
    ]
    tables = [
        {
            "title": f"独立回放{label}指标",
            "columns": ["指标", "值"],
            "rows": metric_rows,
        },
        {
            "title": f"独立回放{label}分布",
            "columns": [
                label,
                "样本数",
                "占比",
                "有标签样本",
                "坏样本数",
                "坏率",
            ],
            "rows": distribution_rows,
        },
    ]
    if typed["economics"]:
        tables.append(
            {
                "title": f"独立回放{label}经济指标",
                "columns": ["指标", "值"],
                "rows": [
                    [key, _typed_validation_metric_text(key, value)]
                    for key, value in sorted(typed["economics"].items())
                    if key != "by_row"
                ],
            }
        )
    if monthly["status"] == "available":
        headline_key = {
            "limit": "mean_limit",
            "pricing": "mean_rate",
            "segmentation": "segment_count",
        }[strategy_type]
        tables.append(
            {
                "title": f"独立回放逐月{label}证据",
                "columns": [
                    "月份",
                    "样本数",
                    "有标签样本",
                    headline_key,
                ],
                "rows": [
                    [
                        str(item["value"]),
                        _num(item["typed_backtest"]["population_count"]),
                        _num(item["typed_backtest"]["labeled_count"]),
                        _typed_validation_metric_text(
                            headline_key,
                            item["typed_backtest"]["metrics"][headline_key],
                        ),
                    ]
                    for item in periods
                ],
            }
        )
    return text, tables


def _typed_validation_metric_text(key: str, value: object) -> str:
    if value is None:
        return "n/a"
    if key.endswith("_rate") or key in {
        "mean_rate",
        "ead_weighted_rate",
        "roa",
    }:
        return _pct(value)
    if isinstance(value, int | float):
        return _num(value)
    return str(value)


def _strategy_pool_stability_integrity_failure() -> tuple[str, list[dict]]:
    return (
        "**Strategy Pool 跨分区稳定性结果完整性校验失败**：计划缓存中的 "
        "exact ImpactCube 引用、稳定性 evidence、TaskArtifact 或 measurement "
        "audit 摘要不一致，已停止展示 PSI 与分布变化。请从当前已认证 "
        "ImpactCube 重新测量。",
        [],
    )


def _validated_strategy_pool_stability_output(
    value: object,
    *,
    trusted_task_id: str | None,
    trusted_inputs: Mapping[str, Any] | None,
    trusted_artifacts: Mapping[str, Any] | None,
) -> dict[str, Any]:
    from marvis.packs.strategy.pool_stability_tools import (
        authenticate_strategy_pool_stability_artifact_record,
        load_strategy_pool_stability_artifact,
        validate_measure_strategy_pool_stability_tool_output,
    )
    from marvis.repositories.task_artifacts import TaskArtifactRepository
    from types import SimpleNamespace

    if (
        not isinstance(value, Mapping)
        or not isinstance(trusted_task_id, str)
        or not trusted_task_id
        or not isinstance(trusted_inputs, Mapping)
        or not isinstance(trusted_artifacts, Mapping)
        or set(trusted_artifacts) != {"pool_stability"}
    ):
        raise ValueError("Pool stability trusted terminal context is unavailable")
    trusted = trusted_artifacts["pool_stability"]
    if (
        not isinstance(trusted, Mapping)
        or set(trusted) != {"record", "tasks_root", "db_path"}
        or not isinstance(trusted["record"], Mapping)
        or not isinstance(trusted["tasks_root"], str)
        or not trusted["tasks_root"]
        or not isinstance(trusted["db_path"], str)
        or not trusted["db_path"]
    ):
        raise ValueError("Pool stability trusted registry context is unavailable")
    stability = value.get("stability")
    if not isinstance(stability, Mapping):
        raise ValueError("Pool stability evidence is unavailable")
    authenticated = authenticate_strategy_pool_stability_artifact_record(
        task_id=trusted_task_id,
        record=trusted["record"],
        stability=stability,
        tasks_root=trusted["tasks_root"],
    )
    source_ref = authenticated["stability"]["source_bindings"]["impact_cube"]
    if dict(trusted_inputs) != source_ref:
        raise ValueError("Pool stability terminal ImpactCube input changed")
    tasks_root = Path(trusted["tasks_root"])
    db_path = Path(trusted["db_path"])
    if (
        not tasks_root.is_absolute()
        or not db_path.is_absolute()
        or db_path.parent != tasks_root.parent
    ):
        raise ValueError("Pool stability governed storage root changed")
    runtime = SimpleNamespace(
        settings=SimpleNamespace(
            tasks_dir=tasks_root,
            db_path=db_path,
        ),
        task_artifacts=TaskArtifactRepository(db_path),
    )
    binding = load_strategy_pool_stability_artifact(
        runtime,
        task_id=trusted_task_id,
        artifact_id=authenticated["artifact_id"],
        expected_artifact_content_hash=authenticated["artifact_content_hash"],
        expected_stability_id=authenticated["stability"]["stability_id"],
        expected_stability_content_hash=authenticated["stability"]["content_hash"],
    )
    if (
        binding.stability != authenticated["stability"]
        or binding.artifact_provenance["producer_run"] != authenticated["producer_run"]
    ):
        raise ValueError("Pool stability live evidence changed")
    producer_run = binding.artifact_provenance["producer_run"]
    output = validate_measure_strategy_pool_stability_tool_output(
        value,
        trusted_task_id=trusted_task_id,
        trusted_artifact_id=binding.artifact_id,
        trusted_artifact_content_hash=binding.artifact_content_hash,
        trusted_producer_run_id=producer_run["run_id"],
        trusted_producer_run_content_hash=producer_run["content_hash"],
    )
    if output["stability"] != binding.stability:
        raise ValueError("Pool stability registered evidence changed")
    return output


def _render_measure_strategy_pool_stability(
    o: dict,
    *,
    trusted_task_id: str | None,
    trusted_inputs: Mapping[str, Any] | None,
    trusted_artifacts: Mapping[str, Any] | None = None,
):
    """Render authenticated PSI evidence without claiming effect validation."""

    try:
        o = _validated_strategy_pool_stability_output(
            o,
            trusted_task_id=trusted_task_id,
            trusted_inputs=trusted_inputs,
            trusted_artifacts=trusted_artifacts,
        )
    except Exception:
        return _strategy_pool_stability_integrity_failure()

    stability = o["stability"]
    identity = stability["identity"]
    comparisons = "、".join(stability["comparison_partitions"])
    artifact = o["artifact"]
    text = (
        f"**Strategy Pool 跨分区稳定性测量完成**：以 `development` 为固定"
        f"基线，对 `{comparisons}` 比较 `{identity['strategy_type']}` Pool "
        f"`{identity['pool_id']}` revision {identity['revision']} 的 first-match "
        "分配与 typed action 分布。\n"
        f"- approval 与 risk 两类 population 分开计算；全局最大 PSI 为 "
        f"**{float(o['max_psi']):.4f}**。\n"
        "- 这是只读分布漂移证据，**不是策略效果验证**；PSI 较低也不代表"
        "风险、通过率或收益效果相同。\n"
        "- 本步骤未修改 Pool、未晋级、未采纳、未部署，也不会创建策略。\n\n"
        f"**Pool stability evidence**：[{artifact['artifact_id']}]"
        f"({artifact['download_url']})"
    )
    if o["warnings"]:
        text += "\n\n**漂移提醒**：" + "；".join(o["warnings"])
    rows = []
    for population in stability["populations"]:
        for comparison in population["comparisons"]:
            for distribution in comparison["distributions"]:
                rows.append(
                    [
                        str(population["population_role"]),
                        str(comparison["partition"]),
                        str(distribution["basis"]),
                        _num(distribution["development_sample_count"]),
                        _num(distribution["comparison_sample_count"]),
                        f"{float(distribution['psi']):.4f}",
                        _pct(distribution["max_abs_share_delta"]),
                        str(distribution["severity"]),
                    ]
                )
    return text, [
        {
            "title": "Pool 跨分区分布稳定性",
            "columns": [
                "Population",
                "比较分区",
                "分布口径",
                "Development 样本",
                "比较样本",
                "PSI",
                "最大占比变化",
                "严重度",
            ],
            "rows": rows,
        }
    ]


def _render_apply_strategy_pool(o: dict):
    """Render only the strictly validated, non-activating Pool application."""

    from marvis.packs.strategy.pool_apply_tools import (
        validate_apply_strategy_pool_tool_output,
    )

    try:
        o = validate_apply_strategy_pool_tool_output(o)
        source = o["source"]
        result = o["result"]
        columns = o["columns"]
        evidence = o["evidence"]
        action_counts = o["action_counts"]
        rule_counts = o["rule_counts"]
        entry_counts = o["entry_counts"]
    except Exception:
        return _strategy_pool_apply_integrity_failure()

    cached_note = "（命中已认证缓存）" if o["cached"] else ""
    text = (
        f"**Strategy Pool 应用完成{cached_note}**：当前 `{source['pool_id']}` "
        f"revision {source['revision']} 已确定性应用到源数据集 "
        f"`{source['dataset_id']}`，生成不可变派生数据集 "
        f"`{result['dataset_id']}`；保留 **{result['row_count']}** 行。\n"
        f"当前 workspace 未切换，仍指向源数据集 "
        f"`{o['workspace']['active_dataset_id']}`。结果**未激活、未采纳、"
        "未部署**，不会修改当前 Strategy Pool，也不会替换当前样本。\n\n"
        f"**应用证据**：[{evidence['artifact_id']}]"
        f"({evidence['download_url']})"
    )
    identity_rows = [
        ["Pool ID", str(source["pool_id"])],
        ["Pool Revision", str(source["revision"])],
        ["Pool Revision ID", str(source["revision_id"])],
        ["Pool Snapshot Hash", str(source["snapshot_hash"])],
        ["Source Dataset", str(source["dataset_id"])],
        ["Result Dataset", str(result["dataset_id"])],
        ["Rows", str(result["row_count"])],
        ["Design Hash", str(source["design_hash"])],
        ["StrategySpec Hash", str(source["strategy_spec_hash"])],
        ["Requirements Hash", str(o["requirements"]["requirements_hash"])],
        ["Evidence Artifact", str(evidence["artifact_id"])],
    ]
    distribution_rows = [
        ["action", str(key), str(value)] for key, value in action_counts.items()
    ]
    distribution_rows.extend(
        ["rule", str(key), str(value)] for key, value in rule_counts.items()
    )
    distribution_rows.extend(
        ["entry", str(key), str(value)] for key, value in entry_counts.items()
    )
    distribution_rows.append(["default", "unmatched", str(o["default_count"])])
    return text, [
        {
            "title": "Strategy Pool 应用身份",
            "columns": ["字段", "值"],
            "rows": identity_rows,
        },
        {
            "title": "派生数据集输出列",
            "columns": ["语义", "列名"],
            "rows": [[str(key), str(value)] for key, value in columns.items()],
        },
        {
            "title": "Strategy Pool 应用分布",
            "columns": ["类别", "标识", "行数"],
            "rows": distribution_rows,
        },
    ]


def _pool_impact_amount_delta_rows(
    amount_deltas: object,
    *,
    period: str | None = None,
) -> list[list[str]]:
    """Format Tool-owned amount deltas; never derive unavailable values."""

    if not isinstance(amount_deltas, dict):
        return []
    rows: list[list[str]] = []
    for action, amounts in amount_deltas.items():
        if not isinstance(amounts, dict):
            continue
        for amount_name, item in amounts.items():
            if not isinstance(item, dict) or item.get("status") != "available":
                continue
            row = [
                str(action),
                str(amount_name),
                "available",
                _num(item.get("coverage_count")),
                _pct(item.get("coverage_rate")),
                _num(item.get("sum")),
                _num(item.get("loan_amount_sum")),
                _num(item.get("overdue_amount_sum")),
                _pct(item.get("overdue_rate")),
            ]
            rows.append(([period] if period is not None else []) + row)
    return rows


def _pool_impact_action_amount_rows(
    action_breakdown: object,
    *,
    period: str | None = None,
) -> list[list[str]]:
    """Format Tool-owned per-action amounts without deriving any observation."""

    if not isinstance(action_breakdown, list):
        return []
    rows: list[list[str]] = []
    for action_row in action_breakdown:
        if not isinstance(action_row, dict):
            continue
        action = str(action_row.get("action") or "")
        amounts = action_row.get("amounts")
        if not isinstance(amounts, dict):
            continue
        for amount_name in ("loan_amount", "overdue_amount", "paired"):
            item = amounts.get(amount_name)
            if not isinstance(item, dict) or item.get("status") != "available":
                continue
            row = [
                action,
                amount_name,
                "available",
                _num(item.get("coverage_count")),
                _pct(item.get("coverage_rate")),
                _num(item.get("sum")),
                _num(item.get("loan_amount_sum")),
                _num(item.get("overdue_amount_sum")),
                _pct(item.get("overdue_rate")),
            ]
            rows.append(([period] if period is not None else []) + row)
    return rows


def _pool_impact_integrity_failure() -> tuple[str, list[dict]]:
    return (
        "**Strategy Pool 影响测算结果完整性校验失败**：计划缓存与 canonical "
        "assessment/artifact 摘要不一致，已停止展示指标。请重新运行测算；"
        "下载接口仍会按 TaskArtifact 注册 hash 校验产物。",
        [],
    )


def _candidate_stability_integrity_failure() -> tuple[str, list[dict]]:
    return (
        "**候选逐月稳定性结果完整性校验失败**：计划缓存与 canonical "
        "stability evidence 或 TaskArtifact 摘要不一致，已停止展示来源、"
        "月份、PSI、样本指标和下载链接。请重新运行候选逐月稳定性测算。",
        [],
    )


def _validate_candidate_stability_tool_output(
    value: object,
    *,
    trusted_task_id: str | None,
    trusted_inputs: Mapping[str, Any] | None,
    trusted_artifacts: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Authenticate Tool output against plan inputs and the artifact registry."""

    from marvis.packs.strategy.candidate_stability import (
        canonical_candidate_stability_artifact_json,
        validate_candidate_stability_artifact,
    )
    from marvis.packs.strategy.candidate_stability_tools import (
        ARTIFACT_KIND,
        ARTIFACT_SCHEMA_VERSION,
        ORIGIN_TOOL,
        TOOL_SCHEMA_VERSION,
    )
    from marvis.packs.strategy.pool import strategy_pool_id

    if not isinstance(value, Mapping):
        raise ValueError("candidate stability Tool output must be an object")
    expected_fields = {
        "schema_version",
        "stability_id",
        "content_hash",
        "basis",
        "source_kind",
        "month_col",
        "population_count",
        "month_count",
        "max_psi",
        "stability",
        "warnings",
        "artifacts",
        "not_created_strategy",
        "not_adopted",
        "not_deployed",
    }
    if set(value) != expected_fields:
        raise ValueError("candidate stability Tool output fields changed")
    if value["schema_version"] != TOOL_SCHEMA_VERSION:
        raise ValueError("candidate stability Tool output schema changed")

    stability = validate_candidate_stability_artifact(value["stability"])
    source = stability["source_ref"]
    summary = stability["summary"]
    lifecycle = stability["lifecycle"]
    if (
        not isinstance(trusted_task_id, str)
        or not trusted_task_id
        or stability["identity"]["task_id"] != trusted_task_id
    ):
        raise ValueError("candidate stability task identity changed")
    if not isinstance(trusted_inputs, Mapping):
        raise ValueError("candidate stability terminal inputs are unavailable")
    if source["source_kind"] == "univariate_asset":
        expected_inputs = {
            "source_kind": "univariate_asset",
            "source_artifact_id": source["artifact_id"],
            "expected_artifact_content_hash": source["artifact_content_hash"],
            "expected_asset_id": source["asset_id"],
            "expected_asset_hash": source["asset_hash"],
        }
        if (
            dict(trusted_inputs) != expected_inputs
            or re.fullmatch(
                r"[0-9a-f]{64}",
                expected_inputs["source_artifact_id"],
            )
            is None
        ):
            raise ValueError("candidate stability terminal asset inputs changed")
    else:
        strategy_type = trusted_inputs.get("strategy_type")
        expected_inputs = {
            "source_kind": "pool_entry",
            "strategy_type": strategy_type,
            "expected_pool_revision": source["revision"],
            "expected_pool_snapshot_hash": source["snapshot_hash"],
            "entry_id": source["entry_id"],
        }
        if (
            not isinstance(strategy_type, str)
            or dict(trusted_inputs) != expected_inputs
            or strategy_pool_id(trusted_task_id, strategy_type) != source["pool_id"]
        ):
            raise ValueError("candidate stability terminal Pool inputs changed")
    expected_outer = {
        "stability_id": stability["stability_id"],
        "content_hash": stability["content_hash"],
        "basis": stability["basis"],
        "source_kind": source["source_kind"],
        "month_col": stability["bindings"]["month_col"],
        "population_count": summary["population_count"],
        "month_count": summary["month_count"],
        "max_psi": summary["max_psi"],
        "not_created_strategy": lifecycle["not_created_strategy"],
        "not_adopted": lifecycle["not_adopted"],
        "not_deployed": lifecycle["not_deployed"],
    }
    for field, expected in expected_outer.items():
        if value[field] != expected:
            raise ValueError(f"candidate stability Tool output {field} changed")
    for field in ("population_count", "month_count"):
        if type(value[field]) is not int:
            raise ValueError(
                f"candidate stability Tool output {field} must be an integer"
            )
    if type(value["max_psi"]) is not float:
        raise ValueError("candidate stability Tool output max_psi must be a float")
    for field in (
        "not_created_strategy",
        "not_adopted",
        "not_deployed",
    ):
        if value[field] is not True:
            raise ValueError(f"candidate stability Tool output {field} changed")

    expected_warnings = [
        (
            f"month {flag['month']} has {flag['observed_rows']} rows, "
            f"below minimum {flag['minimum_rows']}"
        )
        for flag in stability["red_flags"]
    ]
    if (
        not isinstance(value["warnings"], list)
        or value["warnings"] != expected_warnings
    ):
        raise ValueError("candidate stability Tool warnings changed")

    artifacts = value["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise ValueError("candidate stability Tool output must contain one artifact")
    artifact = artifacts[0]
    expected_artifact_fields = {
        "artifact_id",
        "kind",
        "format",
        "filename",
        "content_hash",
        "download_url",
    }
    if not isinstance(artifact, Mapping) or set(artifact) != expected_artifact_fields:
        raise ValueError("candidate stability artifact fields changed")
    if any(not isinstance(artifact[field], str) for field in artifact):
        raise ValueError("candidate stability artifact fields must be text")

    artifact_id = artifact["artifact_id"]
    if re.fullmatch(r"[0-9a-f]{64}", artifact_id) is None:
        raise ValueError("candidate stability artifact id changed")
    canonical = canonical_candidate_stability_artifact_json(stability).encode("utf-8")
    expected_file_hash = hashlib.sha256(canonical).hexdigest()
    expected_download_url = (
        f"/api/tasks/{quote(stability['identity']['task_id'], safe='')}"
        f"/task-artifacts/{quote(artifact_id, safe='')}/download"
    )
    expected_artifact = {
        "artifact_id": artifact_id,
        "kind": ARTIFACT_KIND,
        "format": "json",
        "filename": f"{stability['stability_id']}.json",
        "content_hash": expected_file_hash,
        "download_url": expected_download_url,
    }
    if dict(artifact) != expected_artifact:
        raise ValueError("candidate stability artifact summary changed")

    if not isinstance(trusted_artifacts, Mapping) or set(trusted_artifacts) != {
        "stability"
    }:
        raise ValueError("candidate stability registry artifact is unavailable")
    record = trusted_artifacts["stability"]
    expected_record_fields = {
        "id",
        "task_id",
        "kind",
        "path",
        "content_hash",
        "origin_tool",
        "provenance",
        "created_at",
    }
    if (
        not isinstance(record, Mapping)
        or set(record) != expected_record_fields
        or not isinstance(record["path"], str)
        or not isinstance(record["created_at"], str)
        or not record["created_at"]
    ):
        raise ValueError("candidate stability registry record fields changed")
    path = Path(record["path"])
    canonical_path = Path(os.path.abspath(path))
    if (
        not path.is_absolute()
        or path != canonical_path
        or tuple(canonical_path.parts[-3:])
        != (
            trusted_task_id,
            "strategy_candidate_stability",
            artifact["filename"],
        )
    ):
        raise ValueError("candidate stability registry path changed")
    expected_registry_id = stable_task_artifact_id(
        task_id=trusted_task_id,
        kind=ARTIFACT_KIND,
        path=str(canonical_path),
    )
    if (
        record["id"] != artifact["artifact_id"]
        or record["id"] != expected_registry_id
        or record["task_id"] != trusted_task_id
        or record["kind"] != ARTIFACT_KIND
        or record["origin_tool"] != ORIGIN_TOOL
        or record["content_hash"] != expected_file_hash
    ):
        raise ValueError("candidate stability registry identity changed")

    identity = stability["identity"]
    bindings = stability["bindings"]
    sample_ref = stability["sample_design_ref"]
    expected_provenance = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "producer_version": stability["producer_version"],
        "task_id": trusted_task_id,
        "stability_id": stability["stability_id"],
        "stability_content_hash": stability["content_hash"],
        "basis": stability["basis"],
        "source_kind": source["source_kind"],
        "source_artifact_id": source["artifact_id"],
        "source_artifact_content_hash": source["artifact_content_hash"],
        "source_id": (
            source["asset_id"]
            if source["source_kind"] == "univariate_asset"
            else source["pool_id"]
        ),
        "source_hash": (
            source["asset_hash"]
            if source["source_kind"] == "univariate_asset"
            else source["snapshot_hash"]
        ),
        "rule_id": source["rule_id"],
        "entry_id": source.get("entry_id"),
        "pool_id": source.get("pool_id"),
        "pool_revision": source.get("revision"),
        "pool_revision_id": source.get("revision_id"),
        "dataset_id": identity["dataset_id"],
        "dataset_content_hash": identity["dataset_content_hash"],
        "workspace_revision": identity["workspace_revision"],
        "workspace_generation": identity["workspace_generation"],
        "semantic_mapping_hash": identity["semantic_mapping_hash"],
        "target_col": bindings["target_col"],
        "month_col": bindings["month_col"],
        "sample_design_ref": sample_ref,
        "sample_context_hash": identity["sample_context_hash"],
        "sample_partition": sample_ref["partition"],
    }
    if (
        not isinstance(record["provenance"], Mapping)
        or dict(record["provenance"]) != expected_provenance
    ):
        raise ValueError("candidate stability registry provenance changed")
    return stability, dict(artifact)


def _render_measure_candidate_monthly_stability(
    o: dict,
    *,
    trusted_task_id: str | None,
    trusted_inputs: Mapping[str, Any] | None,
    trusted_artifacts: Mapping[str, Any] | None,
):
    """Render only canonical candidate stability evidence and its artifact."""

    from marvis.packs.strategy.errors import StrategyError

    try:
        stability, artifact = _validate_candidate_stability_tool_output(
            o,
            trusted_task_id=trusted_task_id,
            trusted_inputs=trusted_inputs,
            trusted_artifacts=trusted_artifacts,
        )
    except (StrategyError, TypeError, ValueError, RecursionError):
        return _candidate_stability_integrity_failure()

    source = stability["source_ref"]
    summary = stability["summary"]
    bindings = stability["bindings"]
    if source["source_kind"] == "univariate_asset":
        source_text = (
            f"单变量候选资产 `{source['asset_id']}`（rule `{source['rule_id']}`；"
            "直接规则命中/未命中分布）"
        )
    else:
        source_text = (
            f"Strategy Pool `{source['pool_id']}` revision "
            f"{source['revision']} 的当前顺序条目 `{source['entry_id']}`"
            f"（rule `{source['rule_id']}`；增量 first-match 命中/未命中分布）"
        )

    text = (
        f"**候选逐月稳定性测算完成**：来源为{source_text}。\n"
        f"- 月份列 `{bindings['month_col']}`；完整 development 样本 "
        f"**{_num(summary['population_count'])}** 行，共 "
        f"**{_num(summary['month_count'])}** 个月。\n"
        f"- 最大逐月 PSI **{_num(summary['max_psi'])}**，发生在 "
        f"`{summary['max_psi_month']}`。\n"
        "- PSI 基线是完整 development 样本的命中/未命中分布；这里只展示"
        "确定性分布漂移证据，不据此推断风险、收益或采纳结论。"
    )
    red_flags = stability["red_flags"]
    if red_flags:
        text += (
            "\n- **低样本提醒**："
            + "；".join(
                f"`{flag['month']}` {flag['observed_rows']} 行 < "
                f"minimum_rows={flag['minimum_rows']}"
                for flag in red_flags
            )
            + "。低样本只标记证据强度，不夸大为业务风险结论。"
        )
    else:
        text += (
            "\n- **低样本提醒**：无；各月样本数均达到 "
            f"minimum_rows={bindings['minimum_month_rows']}。"
        )

    lifecycle = stability["lifecycle"]
    text += (
        f"\n- 这是只读 `{lifecycle['candidate_stage']} / "
        f"{lifecycle['observation_stage']} / "
        f"{lifecycle['validation_status']}` 证据；**未创建策略、未修改 "
        "Strategy Pool、未采纳、未部署。**\n\n"
        f"**候选稳定性 artifact**："
        f"[{artifact['filename']}]({artifact['download_url']})"
        f"（registry SHA-256 `{artifact['content_hash']}`）"
    )

    monthly_rows = [
        [
            row["month"],
            _num(row["sample_count"]),
            _num(row["hit_count"]),
            _num(row["not_hit_count"]),
            _pct(row["hit_share"]),
            _pct(row["label_coverage"]),
            _pct(row["hit_bad_rate"]),
            _num(row["psi_vs_development"]),
        ]
        for row in stability["monthly"]
    ]
    tables: list[dict] = [
        {
            "title": "候选逐月稳定性（完整 development 基线）",
            "columns": [
                "月份",
                "样本数",
                "命中数",
                "未命中数",
                "命中占比",
                "标签覆盖率",
                "命中坏账率",
                "PSI vs development",
            ],
            "rows": monthly_rows,
        }
    ]
    if red_flags:
        tables.append(
            {
                "title": "候选逐月稳定性低样本提醒",
                "columns": ["月份", "观测行数", "最低行数"],
                "rows": [
                    [
                        flag["month"],
                        _num(flag["observed_rows"]),
                        _num(flag["minimum_rows"]),
                    ]
                    for flag in red_flags
                ],
            }
        )
    return text, tables


def _render_measure_pool_impact(o: dict):
    """Render immutable Pool impact evidence without deriving any new metric."""

    from marvis.packs.strategy.errors import StrategyError
    from marvis.packs.strategy.pool_impact_tools import (
        validate_measure_pool_impact_tool_output,
    )

    try:
        o = validate_measure_pool_impact_tool_output(o)
    except (StrategyError, RecursionError):
        return _pool_impact_integrity_failure()

    assessment = o.get("assessment") if isinstance(o.get("assessment"), dict) else {}
    identity = (
        assessment.get("identity")
        if isinstance(assessment.get("identity"), dict)
        else {}
    )
    population = (
        assessment.get("population")
        if isinstance(assessment.get("population"), dict)
        else {}
    )
    overall = (
        assessment.get("overall") if isinstance(assessment.get("overall"), dict) else {}
    )
    effect = overall.get("effect") if isinstance(overall.get("effect"), dict) else {}
    actions = overall.get("actions") if isinstance(overall.get("actions"), dict) else {}
    action_rows = [
        item for item in (actions.get("breakdown") or []) if isinstance(item, dict)
    ]
    red_flags = [
        item for item in (assessment.get("red_flags") or []) if isinstance(item, dict)
    ]
    warnings = [str(item) for item in (o.get("warnings") or []) if str(item)]
    text = (
        f"**Strategy Pool 影响测算完成**：`{identity.get('strategy_type', '')}` Pool "
        f"`{identity.get('pool_id', '')}` revision {identity.get('revision')}，"
        f"assessment `{assessment.get('assessment_id', '')}`。"
        f"总体样本 **{_num(population.get('population_count'))}**，"
        f"`unlabeled_rows` **{_num(population.get('unlabelled_count'))}**，"
        f"`nan_labels_excluded` **{_num(o.get('nan_labels_excluded'))}**，"
        f"标签覆盖率 **{_pct(population.get('label_coverage'))}**，"
        f"观测坏账率 **{_pct((actions.get('metrics') or {}).get('overall_bad_rate'))}**。"
    )
    if action_rows:
        text += "\n\n**动作影响**：" + "；".join(
            f"{row.get('action', '')} {_num(row.get('count'))} 笔 / "
            f"{_pct(row.get('rate'))}，坏账率 {_pct(row.get('bad_rate'))}"
            for row in action_rows
        )

    amounts = effect.get("amounts") if isinstance(effect.get("amounts"), dict) else {}
    unavailable: list[str] = []
    amount_rows: list[list[str]] = []
    for key, label in (
        ("loan_amount", "放款金额"),
        ("overdue_amount", "逾期金额"),
        ("paired", "配对逾期金额率"),
    ):
        item = amounts.get(key)
        if isinstance(item, dict) and item.get("status") == "unavailable":
            unavailable.append(f"{label} unavailable（未绑定对应确认语义列）")
        elif isinstance(item, dict) and item.get("status") == "available":
            amount_rows.append(
                [
                    label,
                    "available",
                    _num(item.get("coverage_count")),
                    _pct(item.get("coverage_rate")),
                    (
                        _pct(item.get("overdue_rate"))
                        if key == "paired"
                        else _num(item.get("sum"))
                    ),
                ]
            )
    monthly = (
        assessment.get("monthly") if isinstance(assessment.get("monthly"), dict) else {}
    )
    if monthly.get("status") != "available":
        unavailable.append(
            "逐月结果 unavailable（"
            + str(monthly.get("reason") or "month column unavailable")
            + "）"
        )
    if unavailable:
        text += "\n\n**不可用项**：" + "；".join(unavailable) + "。"
    if red_flags:
        text += "\n\n**Assessment red flags**：" + "；".join(
            f"[{str(item.get('level') or '')}/{str(item.get('code') or '')}] "
            f"{str(item.get('message') or '')}"
            for item in red_flags
        )
        text += "。"
    if warnings:
        text += "\n\n**Tool warnings**：" + "；".join(warnings) + "。"

    baseline = (
        assessment.get("baseline")
        if isinstance(assessment.get("baseline"), dict)
        else {}
    )
    baseline_status = str(baseline.get("status") or "unavailable")
    if baseline_status == "available":
        binding = (
            baseline.get("binding") if isinstance(baseline.get("binding"), dict) else {}
        )
        text += (
            f"\n\n**基线对比 available**：`{binding.get('strategy_id', '')}`；"
            "下表中的 delta 均为 Tool 已计算的当前 Pool 减基线值。"
        )
    elif baseline_status == "not_requested":
        text += "\n\n**基线对比**：未请求（absolute）。"
    else:
        text += (
            "\n\n**基线对比 unavailable**："
            + str(baseline.get("reason") or "baseline evidence unavailable")
            + "。"
        )

    lifecycle = (
        assessment.get("lifecycle")
        if isinstance(assessment.get("lifecycle"), dict)
        else {}
    )
    text += (
        "\n\n这是只读 `development / backtested / unvalidated` 影响证据；"
        f"creates_strategy={lifecycle.get('creates_strategy', False)}，"
        f"adopted={lifecycle.get('adopted', False)}，"
        f"deployed={lifecycle.get('deployed', False)}。"
        "**未创建或修改策略、未采纳、未部署。**"
    )

    artifact = o.get("artifact") if isinstance(o.get("artifact"), dict) else None
    artifacts = [item for item in (o.get("artifacts") or []) if isinstance(item, dict)]
    if artifact is not None:
        artifacts.insert(0, artifact)
    links = [
        f"[{str(item.get('filename') or item.get('kind') or '影响测算 JSON')}]"
        f"({str(item.get('download_url'))})"
        for item in artifacts
        if item.get("download_url")
    ]
    if links:
        text += "\n\n**影响测算 artifact**：" + "；".join(links)

    tables: list[dict] = []
    if red_flags:
        tables.append(
            {
                "title": "Pool Impact 红旗",
                "columns": ["等级", "code", "说明"],
                "rows": [
                    [
                        str(item.get("level") or ""),
                        str(item.get("code") or ""),
                        str(item.get("message") or ""),
                    ]
                    for item in red_flags
                ],
            }
        )
    if warnings:
        tables.append(
            {
                "title": "Pool Impact Tool 警告",
                "columns": ["警告"],
                "rows": [[warning] for warning in warnings],
            }
        )
    if action_rows:
        tables.append(
            {
                "title": "总体动作与风险影响",
                "columns": ["action", "count", "rate", "labelled", "bad", "bad_rate"],
                "rows": [
                    [
                        str(row.get("action") or ""),
                        _num(row.get("count")),
                        _pct(row.get("rate")),
                        _num(row.get("labelled_count")),
                        _num(row.get("bad_count")),
                        _pct(row.get("bad_rate")),
                    ]
                    for row in action_rows
                ],
            }
        )
        action_amount_rows = _pool_impact_action_amount_rows(action_rows)
        if action_amount_rows:
            tables.append(
                {
                    "title": "总体动作金额影响",
                    "columns": [
                        "action",
                        "amount",
                        "status",
                        "coverage_count",
                        "coverage_rate",
                        "sum",
                        "loan_amount_sum",
                        "overdue_amount_sum",
                        "overdue_rate",
                    ],
                    "rows": action_amount_rows,
                }
            )
    if amount_rows:
        tables.append(
            {
                "title": "整体金额影响",
                "columns": ["口径", "状态", "覆盖样本", "覆盖率", "观测值"],
                "rows": amount_rows,
            }
        )

    waterfall = [
        item for item in (assessment.get("waterfall") or []) if isinstance(item, dict)
    ]
    default_unmatched = (
        assessment.get("default_unmatched")
        if isinstance(assessment.get("default_unmatched"), dict)
        else {}
    )
    default_effect = (
        default_unmatched.get("effect")
        if isinstance(default_unmatched.get("effect"), dict)
        else {}
    )
    if waterfall or default_unmatched:
        waterfall_rows = [
            [
                _num(row.get("position")),
                str(row.get("rule_id") or ""),
                str(
                    (row.get("action") or {}).get("type")
                    if isinstance(row.get("action"), dict)
                    else ""
                ),
                _num((row.get("standalone") or {}).get("population_count")),
                _num((row.get("incremental") or {}).get("population_count")),
                _pct((row.get("incremental") or {}).get("population_share")),
                _pct((row.get("incremental") or {}).get("bad_rate")),
                _num((row.get("shadowed") or {}).get("population_count")),
                _num((row.get("remaining_after") or {}).get("population_count")),
            ]
            for row in waterfall
        ]
        if default_unmatched:
            default_action = (
                default_unmatched.get("action")
                if isinstance(default_unmatched.get("action"), dict)
                else {}
            )
            waterfall_rows.append(
                [
                    "default",
                    "default_unmatched",
                    str(default_action.get("type") or ""),
                    "n/a",
                    _num(default_effect.get("population_count")),
                    _pct(default_effect.get("population_share")),
                    _pct(default_effect.get("bad_rate")),
                    "n/a",
                    "n/a",
                ]
            )
        tables.append(
            {
                "title": "Strategy Pool 级联 Waterfall",
                "columns": [
                    "#",
                    "rule_id",
                    "action",
                    "standalone",
                    "incremental",
                    "incremental_share",
                    "incremental_bad_rate",
                    "shadowed",
                    "remaining_after",
                ],
                "rows": waterfall_rows,
            }
        )

    if monthly.get("status") == "available":
        period_rows = [
            item for item in (monthly.get("periods") or []) if isinstance(item, dict)
        ]
        tables.append(
            {
                "title": "逐月影响",
                "columns": [
                    "period",
                    "population",
                    "label_coverage",
                    "bad_rate",
                    "approve_rate",
                    "reject_rate",
                    "review_rate",
                ],
                "rows": [
                    [
                        str(row.get("period") or ""),
                        _num((row.get("effect") or {}).get("population_count")),
                        _pct((row.get("effect") or {}).get("label_coverage")),
                        _pct(
                            ((row.get("actions") or {}).get("metrics") or {}).get(
                                "overall_bad_rate"
                            )
                        ),
                        _pct(
                            ((row.get("actions") or {}).get("metrics") or {}).get(
                                "approve_rate"
                            )
                        ),
                        _pct(
                            ((row.get("actions") or {}).get("metrics") or {}).get(
                                "reject_rate"
                            )
                        ),
                        _pct(
                            ((row.get("actions") or {}).get("metrics") or {}).get(
                                "review_rate"
                            )
                        ),
                    ]
                    for row in period_rows
                ],
            }
        )
        monthly_action_amount_rows: list[list[str]] = []
        for row in period_rows:
            period_actions = row.get("actions")
            breakdown = (
                period_actions.get("breakdown")
                if isinstance(period_actions, dict)
                else None
            )
            monthly_action_amount_rows.extend(
                _pool_impact_action_amount_rows(
                    breakdown,
                    period=str(row.get("period") or ""),
                )
            )
        if monthly_action_amount_rows:
            tables.append(
                {
                    "title": "逐月动作金额影响",
                    "columns": [
                        "period",
                        "action",
                        "amount",
                        "status",
                        "coverage_count",
                        "coverage_rate",
                        "sum",
                        "loan_amount_sum",
                        "overdue_amount_sum",
                        "overdue_rate",
                    ],
                    "rows": monthly_action_amount_rows,
                }
            )

    baseline_overall = (
        baseline.get("overall") if isinstance(baseline.get("overall"), dict) else {}
    )
    deltas = (
        baseline_overall.get("metric_deltas")
        if isinstance(baseline_overall.get("metric_deltas"), dict)
        else {}
    )
    if baseline_status == "available" and deltas:
        tables.append(
            {
                "title": "相对基线指标变化（当前 - 基线）",
                "columns": ["metric", "delta"],
                "rows": [
                    [key, _pct(value) if key.endswith("_rate") else _num(value)]
                    for key, value in deltas.items()
                ],
            }
        )
    overall_amount_delta_rows = _pool_impact_amount_delta_rows(
        baseline_overall.get("amount_deltas")
    )
    if baseline_status == "available" and overall_amount_delta_rows:
        tables.append(
            {
                "title": "相对基线金额变化（当前 - 基线）",
                "columns": [
                    "action",
                    "amount",
                    "status",
                    "coverage_count_delta",
                    "coverage_rate_delta",
                    "sum_delta",
                    "loan_amount_sum_delta",
                    "overdue_amount_sum_delta",
                    "overdue_rate_delta",
                ],
                "rows": overall_amount_delta_rows,
            }
        )
    baseline_monthly = (
        baseline.get("monthly") if isinstance(baseline.get("monthly"), dict) else {}
    )
    baseline_monthly_deltas: list[list[str]] = []
    baseline_monthly_amount_deltas: list[list[str]] = []
    if baseline_monthly.get("status") == "available":
        for row in baseline_monthly.get("periods") or []:
            if not isinstance(row, dict):
                continue
            period = str(row.get("period") or "")
            period_deltas = row.get("metric_deltas")
            if isinstance(period_deltas, dict):
                baseline_monthly_deltas.extend(
                    [
                        period,
                        str(key),
                        _pct(value) if str(key).endswith("_rate") else _num(value),
                    ]
                    for key, value in period_deltas.items()
                )
            baseline_monthly_amount_deltas.extend(
                _pool_impact_amount_delta_rows(
                    row.get("amount_deltas"),
                    period=period,
                )
            )
    if baseline_status == "available" and baseline_monthly_deltas:
        tables.append(
            {
                "title": "逐月相对基线指标变化（当前 - 基线）",
                "columns": ["period", "metric", "delta"],
                "rows": baseline_monthly_deltas,
            }
        )
    if baseline_status == "available" and baseline_monthly_amount_deltas:
        tables.append(
            {
                "title": "逐月相对基线金额变化（当前 - 基线）",
                "columns": [
                    "period",
                    "action",
                    "amount",
                    "status",
                    "coverage_count_delta",
                    "coverage_rate_delta",
                    "sum_delta",
                    "loan_amount_sum_delta",
                    "overdue_amount_sum_delta",
                    "overdue_rate_delta",
                ],
                "rows": baseline_monthly_amount_deltas,
            }
        )
    return text, tables


POOL_PRESENTERS = {
    "add_candidate_to_pool": _render_strategy_pool_mutation,
    "remove_pool_entry": _render_strategy_pool_mutation,
    "set_pool_entry_action": _render_strategy_pool_mutation,
    "reorder_strategy_pool": _render_strategy_pool_mutation,
    "compile_strategy_pool": _render_compile_strategy_pool,
    "materialize_strategy_from_pool": _render_materialize_strategy_from_pool,
    "apply_strategy_pool": _render_apply_strategy_pool,
    "measure_strategy_pool_validation": _render_measure_strategy_pool_validation,
    "measure_strategy_pool_stability": _render_measure_strategy_pool_stability,
    "measure_candidate_monthly_stability": _render_measure_candidate_monthly_stability,
    "measure_pool_impact": _render_measure_pool_impact,
}

INTEGRITY_FAILURES = {
    "measure_pool_impact": _pool_impact_integrity_failure,
    "measure_candidate_monthly_stability": _candidate_stability_integrity_failure,
    "measure_strategy_pool_validation": _strategy_pool_validation_integrity_failure,
    "measure_strategy_pool_stability": _strategy_pool_stability_integrity_failure,
    "apply_strategy_pool": _strategy_pool_apply_integrity_failure,
}

__all__ = ["POOL_PRESENTERS", "INTEGRITY_FAILURES"]
