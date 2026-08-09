"""Candidate, tree, Voting, Cross, and scorecard presenters."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import re
from typing import Any

from marvis.agent.presenters._shared import (
    format_number as _num,
    format_percent as _pct,
    format_value as _fmt,
)


def _scorecard_download_text(o: Mapping, *, fallback: str) -> str:
    artifacts = [
        item for item in (o.get("artifacts") or []) if isinstance(item, Mapping)
    ]
    artifact = next(
        (item for item in artifacts if item.get("download_url")),
        None,
    )
    if artifact is None:
        return ""
    label = str(artifact.get("filename") or artifact.get("kind") or fallback)
    return f"\n\n**受治理 JSON**：[{label}]({artifact['download_url']})"


def _render_design_strategy_candidate(o: dict):
    evidence = o.get("design_evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    strategy_type = str(o.get("strategy_type") or evidence.get("strategy_type") or "")
    bands = [band for band in (evidence.get("bands") or []) if isinstance(band, dict)]
    objective = str(evidence.get("objective") or "")
    source_hash = str(o.get("source_dataset_content_hash") or "")
    text = (
        f"**{strategy_type or '非审批'}策略候选已确定性生成**:"
        f"共 {len(bands)} 个有效分箱，目标 `{objective or '-'}`，"
        f"policy `{o.get('candidate_policy_version') or '-'}`。"
        "这是可复核草稿，尚未采纳；采纳仍需人工确认。"
    )
    if source_hash:
        text += f" 数据证据 `{source_hash[:12]}…`。"
    assumptions = [str(item) for item in (evidence.get("assumptions") or [])]
    if assumptions:
        text += "\n" + "\n".join(f"- 口径:{item}" for item in assumptions)
    red_flags = [
        flag for flag in (evidence.get("red_flags") or []) if isinstance(flag, dict)
    ]
    if red_flags:
        text += "\n" + "\n".join(
            f"- {str(flag.get('level') or 'warning').upper()}:"
            f"{flag.get('message') or flag.get('kind') or flag.get('code')}"
            for flag in red_flags
        )

    rows = []
    for band in bands:
        action = band.get("selected_action")
        action = action if isinstance(action, dict) else {}
        lower = "-∞" if band.get("lower") is None else _fmt(band.get("lower"))
        upper = "+∞" if band.get("upper") is None else _fmt(band.get("upper"))
        rows.append(
            [
                str(band.get("band_id") or ""),
                f"{lower} ~ {upper}",
                _fmt(band.get("count")),
                _pct(band.get("population_share")),
                _pct(band.get("bad_rate")),
                _fmt(band.get("risk_estimate")),
                str(action.get("type") or ""),
                _fmt(action.get("value")),
            ]
        )
    tables = []
    if rows:
        tables.append(
            {
                "title": "确定性候选分箱与动作",
                "columns": [
                    "分箱",
                    "范围",
                    "样本数",
                    "占比",
                    "观测坏率",
                    "风险估计",
                    "动作",
                    "值",
                ],
                "rows": rows,
            }
        )
    return text, tables


def _render_analyze_univariate_candidates(o: dict):
    rankings = [item for item in (o.get("rankings") or []) if isinstance(item, dict)]
    red_flags = [str(item) for item in (o.get("red_flags") or [])]
    artifacts = [item for item in (o.get("artifacts") or []) if isinstance(item, dict)]
    text = (
        f"**单变量候选分析完成**：已分析 {o.get('feature_count', 0)} 个字段，"
        f"得到 {o.get('available_method_count', 0)} 个可用字段/分箱方法组合。"
        f"候选证据 `{o.get('candidate_id', '')}` 仅处于 "
        "`development / unvalidated`，不代表独立验证、采纳或上线。"
    )
    if rankings:
        top = rankings[0]
        text += (
            f" 当前 IV 排名首位是 `{top.get('feature', '')}` / "
            f"`{top.get('method', '')}`；指标均由平台确定性计算。"
        )
    if o.get("nan_labels_dropped"):
        text += (
            f"\n- 已按你的确认排除 {o['nan_labels_dropped']} 行空标签；"
            "候选证据记录了这一口径。"
        )
    if any(flag.startswith("loan_amount_metrics_unavailable") for flag in red_flags):
        text += "\n- 尚未配置放款金额列；如能提供，我可以补做金额口径影响分析。"
    if any(flag.startswith("overdue_amount_metrics_unavailable") for flag in red_flags):
        text += "\n- 尚未配置逾期金额列；如能提供，我可以补做逾期金额口径分析。"
    links = [
        f"[{str(item.get('filename') or item.get('kind') or '下载')}]"
        f"({str(item.get('download_url'))})"
        for item in artifacts
        if item.get("download_url")
    ]
    if links:
        text += "\n\n**候选报告**：" + "；".join(links)

    tables = []
    if rankings:
        tables.append(
            {
                "title": "单变量候选排名（前20）",
                "columns": ["特征", "分箱方法", "IV", "KS", "AUC"],
                "rows": [
                    [
                        str(item.get("feature") or ""),
                        str(item.get("method") or ""),
                        _num(item.get("iv")),
                        _num(item.get("ks")),
                        _num(item.get("auc")),
                    ]
                    for item in rankings[:20]
                ],
            }
        )
    material_flags = [
        flag
        for flag in red_flags
        if not flag.startswith("loan_amount_metrics_unavailable")
        and not flag.startswith("overdue_amount_metrics_unavailable")
    ]
    if material_flags:
        tables.append(
            {
                "title": "候选分析提示",
                "columns": ["提示"],
                "rows": [[flag] for flag in material_flags[:30]],
            }
        )
    return text, tables


def _automatic_tree_condition_text(value: object) -> str:
    if value in (None, {}, []):
        return "全部样本"
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError):
            return "n/a"
    return str(value)


def _render_build_automatic_tree_candidate(o: dict):
    """Render canonical leaves without deriving a ranking or recommendation."""

    summary = o.get("summary") if isinstance(o.get("summary"), dict) else {}
    leaves = [item for item in (o.get("leaf_index") or []) if isinstance(item, dict)]
    artifacts = [item for item in (o.get("artifacts") or []) if isinstance(item, dict)]
    red_flags = [item for item in (o.get("red_flags") or []) if isinstance(item, dict)]
    gaps = [
        item
        for item in (o.get("report_info_gaps") or [])
        if isinstance(item, dict) and item.get("blocking") is False
    ]
    lifecycle = " / ".join(
        str(summary.get(field) or "unknown")
        for field in ("candidate_stage", "observation_stage", "validation_status")
    )
    text = (
        f"**自动树候选构建完成**：资产 `{summary.get('asset_id', '')}`，"
        f"asset hash `{summary.get('asset_hash', '')}`，"
        f"tree result hash `{summary.get('tree_result_hash', '')}`。"
        f"当前状态 `{lifecycle}`；指标、条件和叶节点顺序均来自平台确定性结果。\n"
        "**尚未选叶、未入池、未配置动作，也未采纳、未部署。**"
    )

    links = [
        f"[{str(item.get('filename') or item.get('kind') or '下载')}]"
        f"({str(item.get('download_url'))})"
        for item in artifacts
        if item.get("download_url")
    ]
    if links:
        text += "\n\n**自动树交付件**：" + "；".join(links)

    if gaps:
        labels = {
            "sample_weight": "样本权重",
            "loan_amount": "放款金额",
            "overdue_amount": "逾期金额",
        }
        missing = "、".join(
            labels.get(
                str(item.get("context") or ""),
                str(item.get("context") or item.get("code") or "补充信息"),
            )
            for item in gaps
        )
        text += (
            f"\n\n报告补充信息当前缺少：{missing}。如有可补充；"
            "暂时没有可跳过，最终报告留空。此项不阻塞继续做策略。"
        )

    if red_flags:
        text += (
            "\n\n**风险方向红旗**：存在分裂节点与期望风险方向不一致。"
            "`directions` 在当前自动树中是诊断期望，不是强制分裂约束；"
            "该候选仍为 development / unvalidated，选叶前必须逐项复核。"
        )

    rows = []
    for leaf in leaves:
        basis = (
            leaf.get("metric_basis")
            if isinstance(leaf.get("metric_basis"), dict)
            else {}
        )
        primary = str(basis.get("primary") or "unweighted")
        measurements = (
            leaf.get("measurements")
            if isinstance(leaf.get("measurements"), dict)
            else {}
        )
        metrics = (
            measurements.get(primary)
            if isinstance(measurements.get(primary), dict)
            else {}
        )
        rows.append(
            [
                str(leaf.get("leaf_id") or ""),
                str(leaf.get("rule_id") or ""),
                primary,
                _num(metrics.get("total")),
                _pct(metrics.get("share")),
                _pct(metrics.get("bad_rate")),
                _pct(metrics.get("bad_capture")),
                _num(metrics.get("lift")),
                _automatic_tree_condition_text(leaf.get("condition")),
            ]
        )
    tables = []
    if red_flags:
        direction_labels = {
            "increasing": "递增",
            "decreasing": "递减",
        }
        tables.append(
            {
                "title": "自动树风险方向红旗",
                "columns": ["类型", "Node ID", "特征", "期望风险方向"],
                "rows": [
                    [
                        str(flag.get("code") or ""),
                        str(flag.get("node_id") or ""),
                        str(flag.get("feature") or ""),
                        direction_labels.get(
                            str(flag.get("expected_direction") or ""),
                            str(flag.get("expected_direction") or ""),
                        ),
                    ]
                    for flag in red_flags
                ],
            }
        )
    if rows:
        tables.append(
            {
                "title": "自动树完整叶节点清单",
                "columns": [
                    "Leaf ID",
                    "Rule ID",
                    "指标口径",
                    "样本数",
                    "样本占比",
                    "坏率",
                    "坏样本捕获率",
                    "Lift",
                    "条件",
                ],
                "rows": rows,
            }
        )
    return text, tables


def _render_materialize_automatic_tree_leaf_fragment(o: dict):
    artifacts = [item for item in (o.get("artifacts") or []) if isinstance(item, dict)]
    artifact = next(
        (
            item
            for item in artifacts
            if item.get("kind") == "strategy_automatic_tree_leaf_fragment_json"
            and item.get("format") == "json"
            and item.get("download_url")
        ),
        None,
    )
    text = (
        "**自动树精确叶节点引用已物化。**"
        "该产物仅是 `pointer-only` 引用，没有复制规则、指标或动作；"
        "未入池、未配置动作、未采纳、未部署。"
    )
    if artifact is not None:
        label = str(
            artifact.get("filename")
            or artifact.get("kind")
            or "automatic-tree-leaf-selection.json"
        )
        text += f"\n\n**叶节点引用 JSON**：[{label}]({artifact['download_url']})"

    reason = o.get("selection_reason")
    reason_text = str(reason) if reason is not None else "未提供"
    artifact_id = str(artifact.get("artifact_id") or "") if artifact else ""
    artifact_content_hash = str(artifact.get("content_hash") or "") if artifact else ""
    rows = [
        ["Selection ID", str(o.get("selection_id") or "")],
        ["Selection Hash", str(o.get("selection_hash") or "")],
        ["Tree Asset ID", str(o.get("tree_asset_id") or "")],
        ["Tree Asset Hash", str(o.get("tree_asset_hash") or "")],
        ["Tree Result Hash", str(o.get("tree_result_hash") or "")],
        ["Leaf ID", str(o.get("leaf_id") or "")],
        ["Fragment ID", str(o.get("fragment_id") or "")],
        ["Fragment Hash", str(o.get("fragment_hash") or "")],
        ["Rule ID", str(o.get("rule_id") or "")],
        ["Effect ID", str(o.get("effect_id") or "")],
        ["Artifact ID", artifact_id],
        ["Artifact Content Hash", artifact_content_hash],
        ["Selection Reason", reason_text],
    ]
    return text, [
        {
            "title": "自动树精确叶节点引用",
            "columns": ["字段", "值"],
            "rows": rows,
        }
    ]


def _render_materialize_interactive_tree_frontier_selection(o: dict):
    """Render one pointer-only frontier selection without implying admission."""

    artifacts = [item for item in (o.get("artifacts") or []) if isinstance(item, dict)]
    artifact = next(
        (
            item
            for item in artifacts
            if item.get("kind") == "strategy_interactive_tree_frontier_selection_json"
            and item.get("format") == "json"
            and item.get("download_url")
        ),
        None,
    )
    text = (
        "**交互式决策树精确前沿节点引用已物化。**"
        "该 `pointer-only` 产物绑定精确修订版本与其中一个前沿节点，"
        "没有复制条件、指标或动作；"
        "未入池、未配置动作、未采纳、未部署。"
    )
    if artifact is not None:
        label = str(
            artifact.get("filename")
            or artifact.get("kind")
            or "interactive-tree-frontier-selection.json"
        )
        text += f"\n\n**前沿节点引用 JSON**：[{label}]({artifact['download_url']})"

    reason = o.get("selection_reason")
    artifact_id = str(artifact.get("artifact_id") or "") if artifact else ""
    artifact_content_hash = str(artifact.get("content_hash") or "") if artifact else ""
    rows = [
        ["Selection ID", str(o.get("selection_id") or "")],
        ["Selection Hash", str(o.get("selection_hash") or "")],
        ["Revision ID", str(o.get("revision_id") or "")],
        ["Semantic Tree ID", str(o.get("semantic_tree_id") or "")],
        ["Tree Hash", str(o.get("tree_hash") or "")],
        ["Source Node ID", str(o.get("source_node_id") or "")],
        ["Leaf ID", str(o.get("leaf_id") or "")],
        ["Fragment ID", str(o.get("fragment_id") or "")],
        ["Fragment Hash", str(o.get("fragment_hash") or "")],
        ["Rule ID", str(o.get("rule_id") or "")],
        ["Effect ID", str(o.get("effect_id") or "")],
        ["Artifact ID", artifact_id],
        ["Artifact Content Hash", artifact_content_hash],
        ["Selection Reason", str(reason) if reason is not None else "未提供"],
    ]
    return text, [
        {
            "title": "交互式决策树前沿节点引用",
            "columns": ["字段", "值"],
            "rows": rows,
        }
    ]


def _render_materialize_interactive_tree_frontier_group_selection(o: dict):
    """Render one pointer-only frontier OR group without implying admission."""

    artifacts = [item for item in (o.get("artifacts") or []) if isinstance(item, dict)]
    artifact = next(
        (
            item
            for item in artifacts
            if item.get("kind")
            == "strategy_interactive_tree_frontier_group_selection_json"
            and item.get("format") == "json"
            and item.get("download_url")
        ),
        None,
    )
    source_node_ids = [
        str(item)
        for item in (o.get("source_node_ids") or [])
        if isinstance(item, str) and item
    ]
    member_count = o.get("member_count")
    text = (
        "**交互式决策树前沿 OR 分组引用已物化。**"
        f"该 `pointer-only` 产物绑定精确修订版本中的 {member_count} 个前沿"
        "节点，语义为任一成员命中（OR），没有复制条件、指标或动作；"
        "未入池、未配置动作、未应用、未采纳、未部署。"
    )
    if artifact is not None:
        label = str(
            artifact.get("filename")
            or artifact.get("kind")
            or "interactive-tree-frontier-group-selection.json"
        )
        text += f"\n\n**前沿 OR 分组引用 JSON**：[{label}]({artifact['download_url']})"

    reason = o.get("selection_reason")
    artifact_id = str(artifact.get("artifact_id") or "") if artifact else ""
    artifact_content_hash = str(artifact.get("content_hash") or "") if artifact else ""
    rows = [
        ["Selection ID", str(o.get("selection_id") or "")],
        ["Selection Hash", str(o.get("selection_hash") or "")],
        ["Group ID", str(o.get("group_id") or "")],
        ["Revision ID", str(o.get("revision_id") or "")],
        ["Semantic Tree ID", str(o.get("semantic_tree_id") or "")],
        ["Tree Hash", str(o.get("tree_hash") or "")],
        ["Member Count", str(member_count if member_count is not None else "")],
        ["Source Node IDs", "、".join(source_node_ids)],
        ["Fragment ID", str(o.get("fragment_id") or "")],
        ["Rule ID", str(o.get("rule_id") or "")],
        ["Effect ID", str(o.get("effect_id") or "")],
        ["Artifact ID", artifact_id],
        ["Artifact Content Hash", artifact_content_hash],
        ["Selection Reason", str(reason) if reason is not None else "未提供"],
    ]
    return text, [
        {
            "title": "交互式决策树前沿 OR 分组引用",
            "columns": ["字段", "值"],
            "rows": rows,
        }
    ]


def _automatic_tree_apply_integrity_failure() -> tuple[str, list[dict]]:
    return (
        "**自动树全量写回结果完整性校验失败**：计划缓存中的 source tree、"
        "派生数据集、行数或 evidence 绑定不一致，已停止展示结果。请重新执行写回；"
        "下载接口仍会按 TaskArtifact 注册 hash 校验产物。",
        [],
    )


def _validate_automatic_tree_apply_renderer_output(o: object) -> dict:
    """Validate the exact Tool envelope before rendering any cached value."""

    import re

    if not isinstance(o, dict):
        raise ValueError("automatic-tree apply output must be an object")

    def exact(value: object, fields: set[str], name: str) -> dict:
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError(f"{name} fields are invalid")
        return value

    def text(value: object, name: str) -> str:
        if (
            not isinstance(value, str)
            or not value
            or "\x00" in value
            or "\n" in value
            or "\r" in value
        ):
            raise ValueError(f"{name} is invalid")
        return value

    hash_pattern = re.compile(r"^[0-9a-f]{64}$")
    asset_pattern = re.compile(r"^candidate-asset-[0-9a-f]{32}$")
    run_pattern = re.compile(r"^atar_[0-9a-f]{32}$")
    column_pattern = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")

    def digest(value: object, name: str) -> str:
        normalized = text(value, name)
        if hash_pattern.fullmatch(normalized) is None:
            raise ValueError(f"{name} is invalid")
        return normalized

    def count(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} is invalid")
        return value

    top = exact(
        o,
        {
            "schema_version",
            "run_id",
            "input_hash",
            "cached",
            "activated",
            "source",
            "result",
            "columns",
            "leaf_distribution",
            "workspace",
            "evidence",
        },
        "output",
    )
    if top["schema_version"] != "strategy.apply-automatic-tree-tool.v1":
        raise ValueError("schema version is invalid")
    if run_pattern.fullmatch(text(top["run_id"], "run_id")) is None:
        raise ValueError("run_id is invalid")
    digest(top["input_hash"], "input_hash")
    if type(top["cached"]) is not bool or type(top["activated"]) is not bool:
        raise ValueError("apply flags are invalid")

    source = exact(
        top["source"],
        {
            "tree_artifact_id",
            "tree_artifact_content_hash",
            "asset_id",
            "asset_hash",
            "tree_result_hash",
            "dataset_id",
            "dataset_content_hash",
            "row_count",
        },
        "source",
    )
    text(source["tree_artifact_id"], "source.tree_artifact_id")
    digest(source["tree_artifact_content_hash"], "source.tree_artifact_content_hash")
    if asset_pattern.fullmatch(text(source["asset_id"], "source.asset_id")) is None:
        raise ValueError("source.asset_id is invalid")
    digest(source["asset_hash"], "source.asset_hash")
    digest(source["tree_result_hash"], "source.tree_result_hash")
    text(source["dataset_id"], "source.dataset_id")
    digest(source["dataset_content_hash"], "source.dataset_content_hash")
    source_rows = count(source["row_count"], "source.row_count")

    result = exact(
        top["result"],
        {"dataset_id", "dataset_content_hash", "row_count", "result_hash"},
        "result",
    )
    text(result["dataset_id"], "result.dataset_id")
    digest(result["dataset_content_hash"], "result.dataset_content_hash")
    digest(result["result_hash"], "result.result_hash")
    if (
        count(result["row_count"], "result.row_count") != source_rows
        or result["dataset_id"] == source["dataset_id"]
    ):
        raise ValueError("derived dataset identity is invalid")

    columns = exact(top["columns"], {"leaf_id", "rule_id"}, "columns")
    for field in ("leaf_id", "rule_id"):
        if column_pattern.fullmatch(text(columns[field], f"columns.{field}")) is None:
            raise ValueError(f"columns.{field} is invalid")
    if columns["leaf_id"].casefold() == columns["rule_id"].casefold():
        raise ValueError("output columns are not distinct")

    distribution = top["leaf_distribution"]
    if not isinstance(distribution, list) or not distribution:
        raise ValueError("leaf_distribution is invalid")
    leaf_ids: set[str] = set()
    rule_ids: set[str] = set()
    distributed_rows = 0
    for index, item in enumerate(distribution):
        row = exact(item, {"leaf_id", "rule_id", "row_count"}, f"leaf[{index}]")
        leaf_id = text(row["leaf_id"], f"leaf[{index}].leaf_id")
        rule_id = text(row["rule_id"], f"leaf[{index}].rule_id")
        if leaf_id in leaf_ids or rule_id in rule_ids:
            raise ValueError("leaf distribution identities are not unique")
        leaf_ids.add(leaf_id)
        rule_ids.add(rule_id)
        distributed_rows += count(row["row_count"], f"leaf[{index}].row_count")
    if distributed_rows != source_rows:
        raise ValueError("leaf distribution does not conserve rows")

    workspace = exact(
        top["workspace"],
        {
            "source_revision",
            "source_analysis_generation",
            "source_semantic_mapping_hash",
            "result_revision",
            "result_analysis_generation",
            "result_semantic_mapping_hash",
            "active_dataset_id",
        },
        "workspace",
    )
    count(workspace["source_revision"], "workspace.source_revision")
    count(
        workspace["source_analysis_generation"],
        "workspace.source_analysis_generation",
    )
    digest(
        workspace["source_semantic_mapping_hash"],
        "workspace.source_semantic_mapping_hash",
    )
    digest(
        workspace["result_semantic_mapping_hash"],
        "workspace.result_semantic_mapping_hash",
    )
    text(workspace["active_dataset_id"], "workspace.active_dataset_id")
    if top["activated"]:
        count(workspace["result_revision"], "workspace.result_revision")
        count(
            workspace["result_analysis_generation"],
            "workspace.result_analysis_generation",
        )
        if workspace["active_dataset_id"] != result["dataset_id"]:
            raise ValueError("activated workspace binding is invalid")
    elif (
        workspace["result_revision"] is not None
        or workspace["result_analysis_generation"] is not None
        or workspace["active_dataset_id"] != source["dataset_id"]
    ):
        raise ValueError("inactive workspace binding is invalid")

    evidence = exact(
        top["evidence"],
        {"artifact_id", "content_hash", "download_url"},
        "evidence",
    )
    text(evidence["artifact_id"], "evidence.artifact_id")
    digest(evidence["content_hash"], "evidence.content_hash")
    download_url = text(evidence["download_url"], "evidence.download_url")
    if not (
        download_url.startswith("/api/tasks/")
        and download_url.endswith("/download")
        and "/task-artifacts/" in download_url
    ):
        raise ValueError("evidence.download_url is invalid")
    return top


def _render_apply_automatic_tree(o: dict):
    try:
        o = _validate_automatic_tree_apply_renderer_output(o)
    except (TypeError, ValueError, RecursionError):
        return _automatic_tree_apply_integrity_failure()

    source = o["source"]
    result = o["result"]
    columns = o["columns"]
    evidence = o["evidence"]
    workspace_note = (
        f"当前 workspace 已切换到派生数据集 `{result['dataset_id']}`。"
        if o["activated"]
        else (f"当前 workspace 未切换，仍指向原始数据集 `{source['dataset_id']}`。")
    )
    text = (
        f"**自动树全量写回完成**：完整树资产 `{source['asset_id']}`（source "
        f"artifact `{source['tree_artifact_id']}`）已确定性应用到原始数据集 "
        f"`{source['dataset_id']}`，生成不可变派生数据集 `{result['dataset_id']}`；"
        f"保留 **{source['row_count']}** 行。叶节点输出列 `{columns['leaf_id']}`，"
        f"规则输出列 `{columns['rule_id']}`。{workspace_note}\n"
        "结果边界为 **development / unvalidated**；这是可逆的数据派生，"
        "**未入池、未采纳、未部署**，也未生成业务动作。\n\n"
        f"**写回证据**：[{evidence['artifact_id']}]({evidence['download_url']})"
    )
    identity_rows = [
        ["Source Tree Asset", source["asset_id"]],
        ["Source Tree Artifact", source["tree_artifact_id"]],
        ["Source Dataset", source["dataset_id"]],
        ["Result Dataset", result["dataset_id"]],
        ["Leaf ID Column", columns["leaf_id"]],
        ["Rule ID Column", columns["rule_id"]],
        ["Evidence Artifact", evidence["artifact_id"]],
    ]
    distribution_rows = [
        [item["leaf_id"], item["rule_id"], str(item["row_count"])]
        for item in o["leaf_distribution"]
    ]
    return text, [
        {
            "title": "自动树全量写回身份",
            "columns": ["字段", "值"],
            "rows": identity_rows,
        },
        {
            "title": "自动树叶节点写回分布",
            "columns": ["Leaf ID", "Rule ID", "行数"],
            "rows": distribution_rows,
        },
    ]


def _render_refine_univariate_candidate(o: dict):
    rule = o.get("rule") if isinstance(o.get("rule"), dict) else {}
    effect = o.get("effect") if isinstance(o.get("effect"), dict) else {}
    artifacts = [item for item in (o.get("artifacts") or []) if isinstance(item, dict)]
    text = (
        f"**候选选择与合并完成**：`{o.get('feature', '')}` / "
        f"`{o.get('method', '')}` 已生成候选资产 `{o.get('asset_id', '')}`。"
        "该资产仅处于 `development / unvalidated`，不代表独立验证、采纳或上线。"
    )
    if o.get("parent_candidate_id"):
        text += f" 来源候选证据为 `{o['parent_candidate_id']}`。"
    rule_id = rule.get("rule_id")
    effect_id = o.get("effect_id") or effect.get("effect_id")
    if rule_id or effect_id:
        text += (
            f" 规则 ID `{rule_id or '-'}`，效果 ID `{effect_id or '-'}`；"
            "条件与指标均由平台从绑定样本确定性重放和计算。"
        )
    links = [
        f"[{str(item.get('filename') or item.get('kind') or '下载')}]"
        f"({str(item.get('download_url'))})"
        for item in artifacts
        if item.get("download_url")
    ]
    if links:
        text += "\n\n**候选资产**：" + "；".join(links)
    return text, []


def _render_build_voting_candidate(o: dict):
    """Render one governed n-of-k candidate without implying adoption."""

    selected_entries = [
        item for item in (o.get("selected_entries") or []) if isinstance(item, dict)
    ]
    distribution = [
        item for item in (o.get("hit_distribution") or []) if isinstance(item, dict)
    ]
    artifacts = [item for item in (o.get("artifacts") or []) if isinstance(item, dict)]
    observations = [
        item for item in (o.get("metric_observations") or []) if isinstance(item, dict)
    ]
    effect = o.get("effect") if isinstance(o.get("effect"), dict) else {}
    revision = o.get("pool_revision", o.get("revision"))
    snapshot_hash = str(o.get("pool_snapshot_hash") or o.get("snapshot_hash") or "")
    n = o.get("n")
    k = o.get("k")
    lifecycle = " / ".join(
        str(o.get(field) or "unknown")
        for field in ("candidate_stage", "observation_stage", "validation_status")
    )
    dataset_id = str(o.get("dataset_id") or "n/a")
    target_col = str(o.get("target_col") or "n/a")
    population_count = o.get("population_count", effect.get("population_count"))
    labeled_count = o.get("labeled_count", effect.get("labeled_count"))
    nan_labels_dropped = o.get("nan_labels_dropped")
    drop_nan_labels = o.get("drop_nan_labels")
    population_text = "n/a" if population_count is None else _fmt(population_count)
    labeled_text = "n/a" if labeled_count is None else _fmt(labeled_count)
    dropped_text = "n/a" if nan_labels_dropped is None else _fmt(nan_labels_dropped)
    drop_text = (
        "true"
        if drop_nan_labels is True
        else "false"
        if drop_nan_labels is False
        else "n/a"
    )
    text = (
        f"**Voting n-of-k 策略候选构建完成**：资产 `{o.get('asset_id', '')}`，"
        f"asset hash `{o.get('asset_hash', '')}`；组合为 **{n}-of-{k}**。"
        f"来源 Pool `{o.get('pool_id', '')}` revision {revision}，"
        f"snapshot hash `{snapshot_hash}`；状态 `{lifecycle}`。\n"
        "**本步骤仅生成候选，尚未入池；未应用写回、未采纳、未部署。**"
    )
    text += (
        "\n\n**绑定样本口径**："
        f"dataset `{dataset_id}`，target `{target_col}`；"
        f"population {population_text}，labeled {labeled_text}；"
        f"drop_nan_labels `{drop_text}`，nan_labels_dropped {dropped_text}。"
    )
    text += (
        "\n\n**绑定样本观测（未独立验证）**："
        "以下数值由平台在上述绑定样本上确定性计算，不代表独立验证或上线效果。"
        f"命中率 {_pct(effect.get('matched_rate'))}，"
        f"命中坏率 {_pct(effect.get('matched_bad_rate'))}，"
        f"坏样本捕获率 {_pct(effect.get('bad_capture_rate'))}，"
        f"Lift {_num(effect.get('lift'))}。"
    )

    links = [
        f"[{str(item.get('filename') or item.get('kind') or '下载')}]"
        f"({str(item.get('download_url'))})"
        for item in artifacts
        if item.get("download_url")
    ]
    if links:
        text += "\n\n**Voting 候选资产**：" + "；".join(links)

    tables = []
    if selected_entries:
        tables.append(
            {
                "title": "Voting 成员规则（按 Pool position）",
                "columns": ["Pool position", "rule_id", "entry_id"],
                "rows": [
                    [
                        str(item.get("pool_position") or 0),
                        str(item.get("rule_id") or ""),
                        str(item.get("entry_id") or ""),
                    ]
                    for item in selected_entries
                ],
            }
        )
    if distribution:
        tables.append(
            {
                "title": "Voting 命中数分布",
                "columns": [
                    "命中规则数",
                    "样本数",
                    "样本占比",
                    "坏样本数",
                    "坏率",
                    "Lift",
                ],
                "rows": [
                    [
                        str(item.get("hit_count") or 0),
                        _fmt(item.get("count")),
                        _pct(item.get("share")),
                        _fmt(item.get("bad_count")),
                        _pct(item.get("bad_rate")),
                        _num(item.get("lift")),
                    ]
                    for item in distribution
                ],
            }
        )
    observation_by_identity = {
        (str(item.get("metric_name") or ""), str(item.get("dimension") or "")): item
        for item in observations
    }
    amount_rows = []
    for dimension, dimension_label in (
        ("loan_amount", "放款金额"),
        ("overdue_amount", "逾期金额"),
    ):
        for metric_name, metric_label in (
            ("voting.hit_share", "Voting 命中金额占比"),
            ("voting.bad_capture_rate", "坏样本捕获金额占比"),
        ):
            observation = observation_by_identity.get((metric_name, dimension))
            if observation is None:
                continue
            status = str(observation.get("status") or "unavailable")
            value = _pct(observation.get("value")) if status == "observed" else "n/a"
            amount_rows.append([dimension_label, metric_label, status, value])
    if amount_rows:
        text += "\n\n**金额维度观测**：金额维度观测状态和值见下表。"
        tables.append(
            {
                "title": "Voting 金额维度关键观测",
                "columns": ["金额维度", "指标", "状态", "观测值"],
                "rows": amount_rows,
            }
        )
    else:
        text += "\n\n**金额维度观测**：本次输出未提供可展示的金额维度观测。"
    return text, tables


def _render_build_voting_candidate_from_search(
    o: dict,
    *,
    trusted_inputs: Mapping[str, Any] | None,
):
    """Project an exact search pointer without implying automatic selection."""

    validated = _validated_build_voting_candidate_from_search_output(
        o,
        trusted_inputs=trusted_inputs,
    )
    if validated is None:
        return (
            "**Voting 搜索结果候选完整性校验失败**：平台不会展示未经严格"
            " pointer、候选结构和生命周期一致性校验的搜索来源、约束或候选"
            "事实；请重新执行该构建步骤。",
            [],
        )
    source, candidate = validated

    search_id = source["search_id"]
    combo_id = source["combo_id"]
    eligibility = source.get("eligible")
    prefix = (
        f"**Voting 搜索组合精确构建来源**：用户精确点名 search "
        f"`{search_id}` 中的 combo `{combo_id}`。"
    )
    if eligibility is True:
        prefix += (
            " 该组合满足搜索时的资格约束；这只是对用户 pointer 的证据说明，"
            "不代表平台自动选择或推荐。"
        )
    elif eligibility is False:
        failures = [
            item
            for item in (source.get("constraint_failures") or [])
            if isinstance(item, dict)
        ]
        rendered_failures = "；".join(
            f"{item.get('metric', '-')}"
            f" {item.get('operator', '-')} {_num(item.get('threshold'))}"
            f"（actual {_num(item.get('actual'))}）"
            for item in failures
        )
        prefix += (
            " **警告：该组合未满足搜索时的资格约束。**"
            + (f" 失败约束：{rendered_failures}。" if rendered_failures else "")
            + " 仍继续构建是因为用户精确点名了该 combo，不代表平台推荐。"
        )
    else:
        prefix += " 搜索时的资格状态不可用；平台不会据此声称该组合满足约束或受到推荐。"

    candidate_text, tables = _render_build_voting_candidate(candidate)
    return f"{prefix}\n\n{candidate_text}", tables


def _validated_build_voting_candidate_from_search_output(
    output: object,
    *,
    trusted_inputs: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Validate the wrapper before projecting any search-selection claim."""

    top_fields = {
        "schema_version",
        "source_search_selection",
        "voting_candidate",
        "not_mutated_pool",
        "not_admitted",
        "not_applied",
        "not_adopted",
        "not_deployed",
    }
    source_fields = {
        "search_id",
        "combo_id",
        "strategy_type",
        "rank",
        "member_rule_ids",
        "n",
        "eligible",
        "constraint_failures",
    }
    candidate_fields = {
        "schema_version",
        "asset_id",
        "asset_hash",
        "candidate_id",
        "evidence_hash",
        "rule_id",
        "rule_hash",
        "fragment_id",
        "fragment_hash",
        "effect_id",
        "effect_hash",
        "pool_id",
        "revision",
        "snapshot_hash",
        "selected_entries",
        "n",
        "k",
        "dataset_id",
        "target_col",
        "sample_design_ref",
        "drop_nan_labels",
        "nan_labels_dropped",
        "population_count",
        "labeled_count",
        "candidate_stage",
        "observation_stage",
        "validation_status",
        "effect",
        "metrics",
        "hit_distribution",
        "metric_observations",
        "not_admitted",
        "not_applied",
        "not_adopted",
        "not_deployed",
        "artifacts",
    }
    hash_fields = {
        "asset_hash",
        "evidence_hash",
        "rule_hash",
        "fragment_hash",
        "effect_hash",
        "snapshot_hash",
    }
    id_patterns = {
        "asset_id": r"candidate-asset-[0-9a-f]{32}",
        "candidate_id": r"candidate-[0-9a-f]{32}",
        "rule_id": r"candidate-rule-[0-9a-f]{32}",
        "fragment_id": r"candidate-fragment-[0-9a-f]{32}",
        "effect_id": r"candidate-effect-[0-9a-f]{32}",
    }
    try:
        if not isinstance(output, dict) or set(output) != top_fields:
            return None
        if output["schema_version"] != (
            "strategy.build-voting-candidate-from-search-tool.v1"
        ):
            return None
        if any(
            output[field] is not True
            for field in (
                "not_mutated_pool",
                "not_admitted",
                "not_applied",
                "not_adopted",
                "not_deployed",
            )
        ):
            return None
        source = output["source_search_selection"]
        candidate = output["voting_candidate"]
        if (
            not isinstance(trusted_inputs, Mapping)
            or set(trusted_inputs)
            not in (
                {"search_id", "combo_id"},
                {"search_id", "combo_id", "strategy_type"},
            )
            or not isinstance(source, dict)
            or set(source) != source_fields
            or not isinstance(candidate, dict)
            or set(candidate) != candidate_fields
        ):
            return None
        if (
            trusted_inputs["search_id"] != source["search_id"]
            or trusted_inputs["combo_id"] != source["combo_id"]
            or (
                "strategy_type" in trusted_inputs
                and trusted_inputs["strategy_type"] != source["strategy_type"]
            )
        ):
            return None
        if (
            re.fullmatch(
                r"voting-search-[0-9a-f]{32}",
                str(source["search_id"]),
            )
            is None
            or re.fullmatch(
                r"voting-combo-[0-9a-f]{32}",
                str(source["combo_id"]),
            )
            is None
            or source["strategy_type"]
            not in {"approval", "reject", "limit", "pricing", "segmentation"}
            or isinstance(source["rank"], bool)
            or not isinstance(source["rank"], int)
            or not 1 <= source["rank"] <= 10_000
        ):
            return None
        members = source["member_rule_ids"]
        if (
            not isinstance(members, list)
            or not 2 <= len(members) <= 50
            or len(set(members)) != len(members)
            or any(
                not isinstance(rule_id, str)
                or re.fullmatch(r"candidate-rule-[0-9a-f]{32}", rule_id) is None
                for rule_id in members
            )
        ):
            return None
        source_n = source["n"]
        failures = source["constraint_failures"]
        if (
            isinstance(source_n, bool)
            or not isinstance(source_n, int)
            or not 1 <= source_n <= len(members)
            or not isinstance(source["eligible"], bool)
            or not isinstance(failures, list)
            or len(failures) > 32
            or source["eligible"] is not (len(failures) == 0)
        ):
            return None
        for failure in failures:
            if (
                not isinstance(failure, dict)
                or set(failure) != {"metric", "operator", "threshold", "actual"}
                or not isinstance(failure["metric"], str)
                or not failure["metric"]
                or failure["operator"] not in {"gte", "lte"}
                or isinstance(failure["threshold"], bool)
                or not isinstance(failure["threshold"], (int, float))
                or isinstance(failure["actual"], bool)
                or not isinstance(failure["actual"], (int, float))
            ):
                return None
        if candidate["schema_version"] != "strategy.build-voting-candidate-tool.v2":
            return None
        if any(
            re.fullmatch(pattern, str(candidate[field])) is None
            for field, pattern in id_patterns.items()
        ) or any(
            re.fullmatch(r"[0-9a-f]{64}", str(candidate[field])) is None
            for field in hash_fields
        ):
            return None
        candidate_n = candidate["n"]
        candidate_k = candidate["k"]
        selected_entries = candidate["selected_entries"]
        if (
            not isinstance(candidate["pool_id"], str)
            or not candidate["pool_id"]
            or isinstance(candidate["revision"], bool)
            or not isinstance(candidate["revision"], int)
            or candidate["revision"] < 1
            or isinstance(candidate_n, bool)
            or not isinstance(candidate_n, int)
            or isinstance(candidate_k, bool)
            or not isinstance(candidate_k, int)
            or not 2 <= candidate_k <= 50
            or not 1 <= candidate_n <= candidate_k
            or candidate_n != source_n
            or not isinstance(selected_entries, list)
            or len(selected_entries) != candidate_k
        ):
            return None
        selected_rule_ids: list[str] = []
        selected_entry_ids: list[str] = []
        selected_pool_positions: list[int] = []
        for entry in selected_entries:
            if (
                not isinstance(entry, dict)
                or set(entry) != {"pool_position", "entry_id", "rule_id"}
                or isinstance(entry["pool_position"], bool)
                or not isinstance(entry["pool_position"], int)
                or entry["pool_position"] < 0
                or not isinstance(entry["entry_id"], str)
                or not entry["entry_id"]
                or not isinstance(entry["rule_id"], str)
                or not entry["rule_id"]
            ):
                return None
            selected_pool_positions.append(entry["pool_position"])
            selected_entry_ids.append(entry["entry_id"])
            selected_rule_ids.append(entry["rule_id"])
        if (
            len(set(selected_pool_positions)) != len(selected_pool_positions)
            or len(set(selected_entry_ids)) != len(selected_entry_ids)
            or len(set(selected_rule_ids)) != len(selected_rule_ids)
            or set(selected_rule_ids) != set(members)
        ):
            return None
        if (
            candidate["candidate_stage"] != "development"
            or candidate["observation_stage"] != "backtested"
            or candidate["validation_status"] != "unvalidated"
            or any(
                candidate[field] is not True or candidate[field] is not output[field]
                for field in (
                    "not_admitted",
                    "not_applied",
                    "not_adopted",
                    "not_deployed",
                )
            )
            or not isinstance(candidate["dataset_id"], str)
            or not candidate["dataset_id"]
            or not isinstance(candidate["target_col"], str)
            or not candidate["target_col"]
            or not isinstance(candidate["effect"], dict)
            or not isinstance(candidate["metrics"], dict)
            or not isinstance(candidate["hit_distribution"], list)
            or not isinstance(candidate["metric_observations"], list)
        ):
            return None
        sample_ref = candidate["sample_design_ref"]
        if (
            not isinstance(sample_ref, dict)
            or set(sample_ref)
            != {
                "artifact_id",
                "artifact_content_hash",
                "sample_design_id",
                "sample_design_content_hash",
                "partition",
            }
            or sample_ref["partition"] not in {"development", "risk/development"}
            or any(
                re.fullmatch(r"[0-9a-f]{64}", str(sample_ref[field])) is None
                for field in (
                    "artifact_id",
                    "artifact_content_hash",
                    "sample_design_content_hash",
                )
            )
        ):
            return None
        artifacts = candidate["artifacts"]
        if (
            not isinstance(artifacts, list)
            or len(artifacts) != 1
            or not isinstance(artifacts[0], dict)
            or set(artifacts[0])
            != {
                "artifact_id",
                "kind",
                "format",
                "filename",
                "content_hash",
                "download_url",
            }
            or artifacts[0]["kind"] != "strategy_voting_candidate_json"
            or artifacts[0]["format"] != "json"
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(artifacts[0]["content_hash"]),
            )
            is None
        ):
            return None
    except Exception:
        return None
    return source, candidate


def _render_search_voting_candidates(o: dict):
    """Render a bounded aggregate projection without implying a selection."""

    validated = _validated_voting_search_tool_output(o)
    if validated is None:
        return (
            "**Voting 搜索结果完整性校验失败**：平台不会展示未经 canonical "
            "hash、计数守恒和 artifact hash 认证的搜索计数、排名或指标；"
            "请重新执行 Voting 组合搜索。",
            [],
        )
    result, artifact_hash = validated
    search_id = result["search_id"]
    raw_combinations = result.get("combinations")
    combinations = (
        [
            item
            for item in raw_combinations
            if isinstance(item, dict)
            and item.get("eligible") is True
            and re.fullmatch(
                r"voting-combo-[0-9a-f]{32}",
                str(item.get("combo_id") or ""),
            )
            is not None
            and isinstance(item.get("member_ids"), list)
            and all(
                re.fullmatch(
                    r"candidate-rule-[0-9a-f]{32}",
                    str(rule_id),
                )
                is not None
                for rule_id in item["member_ids"]
            )
        ][:10]
        if isinstance(raw_combinations, list)
        else []
    )
    configuration = result.get("configuration")
    objective = (
        configuration.get("objective")
        if isinstance(configuration, dict)
        and isinstance(configuration.get("objective"), dict)
        else {}
    )
    objective_metric = str(objective.get("metric") or "")
    metric_order = [
        objective_metric,
        "hit_share",
        "bad_rate",
        "bad_capture_rate",
        "lift",
        "weighted_hit_share",
        "weighted_bad_rate",
        "weighted_bad_capture_rate",
        "hit_amount_share",
        "bad_amount_rate",
        "bad_amount_capture_rate",
    ]
    displayed_metrics: list[str] = []
    for metric in metric_order:
        if (
            metric
            and metric not in displayed_metrics
            and any(
                isinstance(item.get("metrics"), dict)
                and item["metrics"].get(metric) is not None
                for item in combinations
            )
        ):
            displayed_metrics.append(metric)

    text = (
        f"**Voting 组合只读搜索完成**：search ID `{search_id}`；"
        f"search_space {_voting_search_count_text(result['search_space'])}，"
        f"evaluated {_voting_search_count_text(result['evaluated'])}，"
        f"eligible {_voting_search_count_text(result['eligible'])}。"
    )
    if result["truncated"] is True:
        text += (
            "\n\n**搜索预算已截断**：按 canonical 组合顺序评估预算前缀，"
            "以下只是已评估范围内 Top-N；不得外推为未评估范围的结论。"
        )
    else:
        text += "\n\n已完整评估当前搜索空间。"
    text += (
        "\n\n**边界**：本步骤未修改 Pool、未选择任何组合、未构建候选、"
        "未入池、未应用、未采纳、未部署。后续如需构建，必须另行引用明确的"
        "组合成员并发起受治理请求。"
    )
    text += (
        "\n\n**完整聚合结果 artifact**：内容 SHA-256 "
        f"`{artifact_hash}`；请从当前任务的统一产物栏下载。"
        "Pool 身份、排除明细和下载路由不从未认证的 Tool 外层字段投影。"
    )

    rows = []
    for item in combinations:
        metrics = item.get("metrics")
        assert isinstance(metrics, dict)
        rows.append(
            [
                str(item.get("rank") or ""),
                str(item["combo_id"]),
                "、".join(str(rule_id) for rule_id in item["member_ids"]),
                str(item.get("n") or ""),
                *[
                    _voting_search_metric_text(metric, metrics.get(metric))
                    for metric in displayed_metrics
                ],
            ]
        )
    return text, [
        {
            "title": "Voting 搜索已评估范围内符合约束的 Top-10",
            "columns": [
                "Rank",
                "Combo ID",
                "成员 Rule IDs",
                "n",
                *displayed_metrics,
            ],
            "rows": rows,
        }
    ]


def _validated_voting_search_tool_output(
    output: object,
) -> tuple[dict[str, Any], str] | None:
    """Authenticate every renderer-owned Voting-search fact before projection."""

    from marvis.packs.strategy.voting_candidate_search import (
        canonical_voting_candidate_search_result_json,
        validate_voting_candidate_search_result,
    )

    expected_fields = {
        "schema_version",
        "search_id",
        "request_hash",
        "content_hash",
        "pool_id",
        "pool_revision",
        "pool_snapshot_hash",
        "search_space",
        "evaluated",
        "truncated",
        "eligible",
        "excluded_unsupported_rule_ids",
        "search_result",
        "artifacts",
        "not_mutated_pool",
        "not_selected",
        "not_admitted",
        "not_applied",
        "not_adopted",
        "not_deployed",
    }
    try:
        if not isinstance(output, dict) or set(output) != expected_fields:
            return None
        if output["schema_version"] != "strategy.search-voting-candidates-tool.v1":
            return None
        result = validate_voting_candidate_search_result(output["search_result"])
        for field in (
            "search_id",
            "request_hash",
            "content_hash",
            "search_space",
            "evaluated",
            "truncated",
            "eligible",
        ):
            if output[field] != result[field]:
                return None
        if (
            not isinstance(output["pool_id"], str)
            or not output["pool_id"]
            or isinstance(output["pool_revision"], bool)
            or not isinstance(output["pool_revision"], int)
            or output["pool_revision"] < 1
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(output["pool_snapshot_hash"]),
            )
            is None
        ):
            return None
        if any(
            output[field] is not True
            for field in (
                "not_mutated_pool",
                "not_selected",
                "not_admitted",
                "not_applied",
                "not_adopted",
                "not_deployed",
            )
        ):
            return None
        excluded = output["excluded_unsupported_rule_ids"]
        if (
            not isinstance(excluded, list)
            or excluded != sorted(set(excluded))
            or any(
                not isinstance(rule_id, str)
                or re.fullmatch(r"candidate-rule-[0-9a-f]{32}", rule_id) is None
                for rule_id in excluded
            )
        ):
            return None
        artifacts = output["artifacts"]
        if (
            not isinstance(artifacts, list)
            or len(artifacts) != 1
            or not isinstance(artifacts[0], dict)
            or set(artifacts[0])
            != {
                "artifact_id",
                "kind",
                "format",
                "filename",
                "content_hash",
                "download_url",
            }
        ):
            return None
        artifact = artifacts[0]
        canonical = canonical_voting_candidate_search_result_json(result).encode(
            "utf-8"
        )
        artifact_hash = hashlib.sha256(canonical).hexdigest()
        if (
            re.fullmatch(r"[0-9a-f]{64}", str(artifact["artifact_id"])) is None
            or artifact["kind"] != "strategy_voting_candidate_search_json"
            or artifact["format"] != "json"
            or not isinstance(artifact["filename"], str)
            or not artifact["filename"].endswith(".json")
            or artifact["content_hash"] != artifact_hash
            or not isinstance(artifact["download_url"], str)
            or not artifact["download_url"].startswith("/api/tasks/")
            or f"expected_content_hash={artifact_hash}" not in artifact["download_url"]
        ):
            return None
    except Exception:
        return None
    return result, artifact_hash


def _voting_search_metric_text(metric: str, value: object) -> str:
    if value is None:
        return "n/a"
    if metric in {
        "hit_share",
        "bad_rate",
        "bad_capture_rate",
        "weighted_hit_share",
        "weighted_bad_rate",
        "weighted_bad_capture_rate",
        "hit_amount_share",
        "bad_amount_rate",
        "bad_amount_capture_rate",
    }:
        return _pct(value)
    return _fmt(value)


def _voting_search_count_text(value: object) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    return _fmt(value)


def _cross_matrix_cell_metric(cell: dict, name: str):
    effect = cell.get("effect") if isinstance(cell.get("effect"), dict) else {}
    return effect.get(name, cell.get(name))


def _cross_matrix_observation_value(observation: dict, field: str, *, pct=False):
    status = str(observation.get("status") or "unavailable")
    value = observation.get(field)
    if status in {"unavailable", "insufficient_data", "not_applicable"}:
        return "n/a"
    return _pct(value) if pct else _num(value)


def _render_build_cross_matrix_candidate(o: dict):
    """Render a complete matrix in canonical axis order without selecting cells."""

    row_axis = o.get("row_axis") if isinstance(o.get("row_axis"), dict) else {}
    column_axis = o.get("column_axis") if isinstance(o.get("column_axis"), dict) else {}
    asset = (
        o.get("cross_matrix_candidate")
        if isinstance(o.get("cross_matrix_candidate"), dict)
        else {}
    )
    matrix = asset.get("matrix") if isinstance(asset.get("matrix"), dict) else {}
    cells = [item for item in (matrix.get("cells") or []) if isinstance(item, dict)]
    axes = [item for item in (asset.get("axes") or []) if isinstance(item, dict)]
    source_bin_by_id = {
        str(bin_row.get("bin_id") or ""): str(bin_row.get("source_bin_id") or "")
        for axis in axes
        for bin_row in (axis.get("bins") or [])
        if isinstance(bin_row, dict) and bin_row.get("bin_id")
    }
    artifacts = [item for item in (o.get("artifacts") or []) if isinstance(item, dict)]
    lifecycle = " / ".join(
        str(o.get(field) or "unknown")
        for field in ("candidate_stage", "observation_stage", "validation_status")
    )
    drop_text = (
        "true"
        if o.get("drop_nan_labels") is True
        else "false"
        if o.get("drop_nan_labels") is False
        else "n/a"
    )
    text = (
        f"**二维 Cross Matrix 候选构建完成**：资产 `{o.get('asset_id', '')}`，"
        f"asset hash `{o.get('asset_hash', '')}`；矩阵候选证据 "
        f"`{o.get('candidate_id', '')}` / `{o.get('evidence_hash', '')}`；"
        "来源单变量候选证据 "
        f"`{o.get('parent_candidate_id', '')}` / `{o.get('parent_evidence_hash', '')}`。\n"
        f"- X 轴：{row_axis.get('feature', '')} / {row_axis.get('method', '')} / "
        f"{row_axis.get('bin_count', 0)} bins\n"
        f"- Y 轴：{column_axis.get('feature', '')} / "
        f"{column_axis.get('method', '')} / {column_axis.get('bin_count', 0)} bins\n"
        f"- 已固化 {o.get('cell_count', 0)} 个完整单元格；状态 `{lifecycle}`。\n"
        "**本步骤只生成完整矩阵 evidence：未选择格子、未入池、未应用写回、"
        "未采纳、未部署。**"
    )
    text += (
        "\n\n**绑定样本口径**："
        f"dataset `{o.get('dataset_id', 'n/a')}`，target `{o.get('target_col', 'n/a')}`；"
        f"population {_fmt(o.get('population_count'))}，"
        f"labeled {_fmt(o.get('labeled_count'))}；"
        f"drop_nan_labels `{drop_text}`，"
        f"nan_labels_dropped {_fmt(o.get('nan_labels_dropped'))}。"
    )
    text += (
        "\n\n**绑定样本观测（未独立验证）**："
        "下表保持 X/Y 分箱及完整 Cartesian 单元格的资产顺序，不按坏率、Lift "
        "或其他指标重排，也不构成格子选择建议。"
    )
    links = [
        f"[{str(item.get('filename') or item.get('kind') or '下载')}]"
        f"({str(item.get('download_url'))})"
        for item in artifacts
        if item.get("download_url")
    ]
    if links:
        text += "\n\n**Cross Matrix 完整 JSON**：" + "；".join(links)

    metric_rows = []
    amount_rows = []
    for cell in cells:
        row_bin_id = str(cell.get("row_bin_id") or "")
        column_bin_id = str(cell.get("column_bin_id") or "")
        row_bin = source_bin_by_id.get(
            row_bin_id,
            str(cell.get("row_source_bin_id") or row_bin_id),
        )
        column_bin = source_bin_by_id.get(
            column_bin_id,
            str(cell.get("column_source_bin_id") or column_bin_id),
        )
        cell_id = str(cell.get("cell_id") or f"{row_bin} × {column_bin}")
        metric_rows.append(
            [
                cell_id,
                row_bin,
                column_bin,
                _fmt(_cross_matrix_cell_metric(cell, "count")),
                _pct(_cross_matrix_cell_metric(cell, "share")),
                _fmt(_cross_matrix_cell_metric(cell, "good")),
                _fmt(_cross_matrix_cell_metric(cell, "bad")),
                _pct(_cross_matrix_cell_metric(cell, "bad_rate")),
                _num(_cross_matrix_cell_metric(cell, "lift")),
                _num(_cross_matrix_cell_metric(cell, "woe")),
                _num(_cross_matrix_cell_metric(cell, "iv_contribution")),
            ]
        )
        effect = cell.get("effect") if isinstance(cell.get("effect"), dict) else {}
        amounts = (
            effect.get("amount_metrics")
            if isinstance(effect.get("amount_metrics"), dict)
            else {}
        )
        for dimension, label in (
            ("loan_amount", "放款金额"),
            ("overdue_amount", "逾期金额"),
            ("overdue_rate", "配对逾期率"),
        ):
            observation = (
                amounts.get(dimension)
                if isinstance(amounts.get(dimension), dict)
                else None
            )
            if observation is None:
                continue
            status = str(observation.get("status") or "unavailable")
            covered = observation.get("covered_count")
            amount_rows.append(
                [
                    cell_id,
                    row_bin,
                    column_bin,
                    label,
                    status,
                    "n/a" if covered is None else _fmt(covered),
                    _cross_matrix_observation_value(
                        observation,
                        "coverage_rate",
                        pct=True,
                    ),
                    _cross_matrix_observation_value(
                        observation,
                        "value",
                        pct=dimension == "overdue_rate",
                    ),
                    str(observation.get("reason") or ""),
                ]
            )

    tables = []
    if metric_rows:
        tables.append(
            {
                "title": "二维 Cross Matrix 全量单元格（保持 X/Y 分箱顺序）",
                "columns": [
                    "Cell ID",
                    "X source bin",
                    "Y source bin",
                    "样本数",
                    "样本占比",
                    "好样本",
                    "坏样本",
                    "坏率",
                    "Lift",
                    "WOE",
                    "IV",
                ],
                "rows": metric_rows,
            }
        )
    if amount_rows:
        tables.append(
            {
                "title": "二维 Cross Matrix 金额观测",
                "columns": [
                    "Cell ID",
                    "X source bin",
                    "Y source bin",
                    "维度",
                    "状态",
                    "覆盖样本",
                    "覆盖率",
                    "观测值",
                    "原因",
                ],
                "rows": amount_rows,
            }
        )
    return text, tables


def _render_materialize_cross_matrix_cell_selection(o: dict):
    """Render pointer-only Cross cells in the tool's canonical source order."""

    artifacts = [item for item in (o.get("artifacts") or []) if isinstance(item, dict)]
    artifact = next(
        (
            item
            for item in artifacts
            if item.get("format") == "json" and item.get("download_url")
        ),
        None,
    )
    cell_ids = [str(value) for value in (o.get("cell_ids") or []) if value]
    lifecycle = " / ".join(
        str(o.get(field) or "unknown")
        for field in ("candidate_stage", "observation_stage", "validation_status")
    )
    text = (
        "**Cross Matrix 精确单元格选择已物化。**"
        f"已按源矩阵顺序固化 **{len(cell_ids)}** 个 cell pointer；多个 cell "
        "采用确定性 OR 语义。该产物仅保存对完整矩阵的不可变引用，不复制观测"
        "指标，不执行排名或推荐，也不生成业务动作；未入池、未应用、未采纳、未部署。"
        f"当前状态 `{lifecycle}`。"
    )
    if artifact is not None:
        label = str(
            artifact.get("filename")
            or artifact.get("kind")
            or "cross-matrix-cell-selection.json"
        )
        text += f"\n\n**Cross Matrix 单元格选择 JSON**：[{label}]({artifact['download_url']})"

    reason = o.get("selection_reason")
    details = [
        ["Selection ID", str(o.get("selection_id") or "")],
        ["Selection Hash", str(o.get("selection_hash") or "")],
        ["Group ID", str(o.get("group_id") or "")],
        ["Source Asset ID", str(o.get("source_asset_id") or "")],
        ["Source Asset Hash", str(o.get("source_asset_hash") or "")],
        ["Source Candidate ID", str(o.get("source_candidate_id") or "")],
        ["Source Evidence Hash", str(o.get("source_evidence_hash") or "")],
        ["Fragment ID", str(o.get("fragment_id") or "")],
        ["Fragment Type", str(o.get("fragment_type") or "")],
        ["Rule ID", str(o.get("rule_id") or "")],
        ["Effect ID", str(o.get("effect_id") or "")],
        ["Selection Reason", str(reason) if reason is not None else "未提供"],
        ["Not Admitted", str(o.get("not_admitted"))],
        ["Not Applied", str(o.get("not_applied"))],
        ["Not Adopted", str(o.get("not_adopted"))],
        ["Not Deployed", str(o.get("not_deployed"))],
    ]
    tables = [
        {
            "title": "Cross Matrix 精确单元格选择引用",
            "columns": ["字段", "值"],
            "rows": details,
        }
    ]
    if cell_ids:
        tables.append(
            {
                "title": "已选择 Cell IDs（源矩阵顺序）",
                "columns": ["顺序", "Cell ID"],
                "rows": [
                    [str(index), cell_id]
                    for index, cell_id in enumerate(cell_ids, start=1)
                ],
            }
        )
    return text, tables


def _render_build_scorecard_band_asset(o: dict):
    """Render the complete measured band/cutoff set without choosing one."""

    asset = (
        o.get("scorecard_band_asset")
        if isinstance(o.get("scorecard_band_asset"), Mapping)
        else {}
    )
    bands = [item for item in (asset.get("bands") or []) if isinstance(item, Mapping)]
    cutoffs = [
        item for item in (asset.get("cutoffs") or []) if isinstance(item, Mapping)
    ]
    performance = (
        o.get("performance") if isinstance(o.get("performance"), Mapping) else {}
    )
    banding = o.get("banding") if isinstance(o.get("banding"), Mapping) else {}
    text = (
        f"**Scorecard 完整分数带已构建**：资产 `{o.get('asset_id', '')}`，"
        f"asset hash `{o.get('asset_hash', '')}`；已固化 "
        f"**{_fmt(o.get('band_count', len(bands)))}** 个分数带和 "
        f"**{_fmt(o.get('cutoff_count', len(cutoffs)))}** 个内部 cutoff。\n"
        f"- 分档方法 `{banding.get('method', 'n/a')}`；请求 "
        f"{_fmt(banding.get('requested_bin_count', 'n/a'))} 档，实际 "
        f"{_fmt(banding.get('effective_bin_count', len(bands)))} 档。\n"
        f"- risk/development 样本 {_fmt(o.get('development_count'))} 行，"
        f"有标签 {_fmt(o.get('labeled_count'))} 行，坏样本 "
        f"{_fmt(o.get('bad_count'))}；AUC {_num(performance.get('auc'))}，"
        f"KS {_num(performance.get('ks'))}。\n"
        "**本步骤只生成完整、确定性的分数带证据：不会自动选择、排名或推荐 "
        "cutoff；未入池、未应用、未采纳、未部署。**"
    )
    text += _scorecard_download_text(o, fallback="scorecard-band.json")
    band_rows = [
        [
            str(band.get("band_id") or band.get("bin_id") or ""),
            _num(band.get("lower_bound")),
            _num(band.get("upper_bound")),
            _fmt(band.get("count")),
            _pct(band.get("share")),
            _fmt(band.get("labeled_count")),
            _fmt(band.get("bad_count")),
            _pct(band.get("bad_rate")),
            _num(band.get("average_pd")),
        ]
        for band in bands
    ]
    cutoff_rows = []
    for cutoff in cutoffs:
        lower = (
            cutoff.get("lower_risk")
            if isinstance(cutoff.get("lower_risk"), Mapping)
            else {}
        )
        higher = (
            cutoff.get("higher_risk")
            if isinstance(cutoff.get("higher_risk"), Mapping)
            else {}
        )
        cutoff_rows.append(
            [
                str(cutoff.get("cutoff_id") or ""),
                _num(cutoff.get("execution_pd")),
                _num(cutoff.get("display_points")),
                _fmt(lower.get("count")),
                _pct(lower.get("bad_rate")),
                _fmt(higher.get("count")),
                _pct(higher.get("bad_rate")),
                "是" if cutoff.get("mask_equivalence") is True else "否",
            ]
        )
    tables = []
    if band_rows:
        tables.append(
            {
                "title": "Scorecard 分数带",
                "columns": [
                    "Band ID",
                    "Raw PD 下界",
                    "Raw PD 上界",
                    "样本数",
                    "占比",
                    "有标签",
                    "坏样本",
                    "坏率",
                    "平均 PD",
                ],
                "rows": band_rows,
            }
        )
    if cutoff_rows:
        tables.append(
            {
                "title": "Scorecard cutoff 全量证据",
                "columns": [
                    "Cutoff ID",
                    "执行 PD",
                    "展示分",
                    "低风险样本",
                    "低风险坏率",
                    "高风险样本",
                    "高风险坏率",
                    "PD/Points 等价",
                ],
                "rows": cutoff_rows,
            }
        )
    return text, tables


def _render_materialize_scorecard_cutoff_selection(o: dict):
    """Render one explicit pointer without implying Pool admission or action."""

    reason = o.get("selection_reason")
    text = (
        "**Scorecard cutoff 精确选择已物化。**"
        f"已从完整分数带 `{o.get('source_asset_id', '')}` 创建 cutoff "
        f"`{o.get('cutoff_id', '')}` 的 pointer-only 不可变引用。"
        "该步骤不会自动排名或推荐，不复制完整分数带或指标，也不生成业务动作；"
        "**未入池、未应用、未采纳、未部署。**"
    )
    text += _scorecard_download_text(
        o,
        fallback="scorecard-cutoff-selection.json",
    )
    return text, [
        {
            "title": "Scorecard cutoff 选择引用",
            "columns": ["字段", "值"],
            "rows": [
                ["Selection ID", str(o.get("selection_id") or "")],
                ["Selection Hash", str(o.get("selection_hash") or "")],
                ["Source Asset ID", str(o.get("source_asset_id") or "")],
                ["Source Asset Hash", str(o.get("source_asset_hash") or "")],
                ["Cutoff ID", str(o.get("cutoff_id") or "")],
                [
                    "Selection Reason",
                    str(reason) if reason is not None else "未提供",
                ],
                ["Not Admitted", str(o.get("not_admitted"))],
                ["Not Applied", str(o.get("not_applied"))],
                ["Not Adopted", str(o.get("not_adopted"))],
                ["Not Deployed", str(o.get("not_deployed"))],
            ],
        }
    ]


def _render_search_cross_threshold_rules(o: dict):
    result = o.get("search_result") if isinstance(o.get("search_result"), dict) else {}
    configuration = (
        result.get("configuration")
        if isinstance(result.get("configuration"), dict)
        else {}
    )
    rules = [
        item for item in (result.get("rules") or [])[:20] if isinstance(item, dict)
    ]
    text = (
        "**2D/3D Cross 阈值规则搜索完成。**"
        f"搜索 `{result.get('search_id', '')}` 使用 "
        f"{_fmt(configuration.get('dimension'))}D 组合，"
        f"已评估 {_fmt(result.get('evaluated'))} / "
        f"{_fmt(result.get('search_space'))} 条试验，"
        f"其中 {_fmt(result.get('eligible'))} 条满足约束。"
        "**排序只用于展示证据；平台没有自动选择、构建、入池、应用、"
        "采纳或部署任何规则。**"
    )
    return text, [
        {
            "title": "Cross 阈值规则搜索结果（最多展示 20 条）",
            "columns": [
                "Rank",
                "Rule ID",
                "Conditions",
                "Hit Share",
                "Bad Rate",
                "Lift",
                "Amount Lift",
                "Eligible",
                "Constraint Failures",
            ],
            "rows": [
                [
                    _fmt(rule.get("rank")),
                    str(rule.get("rule_id") or ""),
                    " AND ".join(
                        (
                            f"{item.get('feature', '')} "
                            f"{item.get('operator', '')} "
                            f"{_fmt(item.get('threshold'))}"
                            + (
                                " OR missing"
                                if item.get("include_missing") is True
                                else ""
                            )
                        )
                        for item in (rule.get("conditions") or [])
                        if isinstance(item, dict)
                    ),
                    _pct((rule.get("metrics") or {}).get("hit_share")),
                    _pct((rule.get("metrics") or {}).get("bad_rate")),
                    _fmt((rule.get("metrics") or {}).get("lift")),
                    _fmt((rule.get("metrics") or {}).get("amount_lift")),
                    str(rule.get("eligible")),
                    "、".join(rule.get("constraint_failures") or []),
                ]
                for rule in rules
            ],
        }
    ]


def _render_build_cross_rule_candidate_from_search(o: dict):
    candidate = o.get("candidate") if isinstance(o.get("candidate"), dict) else {}
    selection = (
        o.get("source_search_selection")
        if isinstance(o.get("source_search_selection"), dict)
        else {}
    )
    metrics = (
        candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    )
    text = (
        "**Cross 阈值规则候选已精确物化。**"
        f"候选 `{candidate.get('asset_id', '')}` 来自搜索 "
        f"`{selection.get('search_id', '')}` 的规则 "
        f"`{selection.get('rule_id', '')}`；原始 rank "
        f"{_fmt(selection.get('rank'))}、eligible "
        f"`{selection.get('eligible')}` 仅作为来源证据。"
        "**候选仍未独立验证、未入池、未应用、未采纳、未部署。**"
    )
    return text, [
        {
            "title": "Cross 阈值规则候选",
            "columns": ["字段", "值"],
            "rows": [
                ["Asset ID", str(candidate.get("asset_id") or "")],
                ["Rule ID", str(selection.get("rule_id") or "")],
                ["Dimension", _fmt(candidate.get("dimension"))],
                ["Hit Share", _pct(metrics.get("hit_share"))],
                ["Bad Rate", _pct(metrics.get("bad_rate"))],
                ["Lift", _fmt(metrics.get("lift"))],
                ["Amount Lift", _fmt(metrics.get("amount_lift"))],
                [
                    "Selection Reason",
                    str(candidate.get("selection_reason") or "未提供"),
                ],
            ],
        }
    ]


CANDIDATE_PRESENTERS = {
    "design_strategy_candidate": _render_design_strategy_candidate,
    "analyze_univariate_candidates": _render_analyze_univariate_candidates,
    "build_automatic_tree_candidate": _render_build_automatic_tree_candidate,
    "apply_automatic_tree": _render_apply_automatic_tree,
    "materialize_automatic_tree_leaf_fragment": _render_materialize_automatic_tree_leaf_fragment,
    "materialize_interactive_tree_frontier_selection": _render_materialize_interactive_tree_frontier_selection,
    "materialize_interactive_tree_frontier_group_selection": _render_materialize_interactive_tree_frontier_group_selection,
    "search_voting_candidates": _render_search_voting_candidates,
    "build_voting_candidate_from_search": _render_build_voting_candidate_from_search,
    "build_voting_candidate": _render_build_voting_candidate,
    "build_cross_matrix_candidate": _render_build_cross_matrix_candidate,
    "search_cross_threshold_rules": _render_search_cross_threshold_rules,
    "build_cross_rule_candidate_from_search": _render_build_cross_rule_candidate_from_search,
    "materialize_cross_matrix_cell_selection": _render_materialize_cross_matrix_cell_selection,
    "build_scorecard_band_asset": _render_build_scorecard_band_asset,
    "materialize_scorecard_cutoff_selection": _render_materialize_scorecard_cutoff_selection,
    "refine_univariate_candidate": _render_refine_univariate_candidate,
}

INTEGRITY_FAILURES = {
    "apply_automatic_tree": _automatic_tree_apply_integrity_failure,
}

__all__ = ["CANDIDATE_PRESENTERS", "INTEGRITY_FAILURES"]
