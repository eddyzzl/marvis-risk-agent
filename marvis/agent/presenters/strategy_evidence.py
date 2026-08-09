"""Strategy project, sample, model, report, and delivery presenters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from marvis.agent.presenters._shared import (
    format_number as _num,
    format_percent as _pct,
    format_value as _fmt,
)


def _strategy_report_integrity_failure() -> tuple[str, list[dict]]:
    return (
        "**StrategyReportBundle V2 结果完整性校验失败**：计划缓存与 canonical "
        "report bundle、状态或四个 TaskArtifact 摘要不一致，已停止展示报告"
        "身份、警告和下载链接。请重新生成报告。",
        [],
    )


def _strategy_delivery_integrity_failure() -> tuple[str, list[dict]]:
    return (
        "**策略交付结果完整性校验失败**：计划缓存与受信任的任务/步骤输入、"
        "canonical Strategy DSL、等价性证据或 TaskArtifact registry 摘要"
        "不一致，已停止展示交付身份、样本数量和下载链接。请重新生成离线"
        "交付包；下载接口仍会按注册 hash 复核文件。",
        [],
    )


def _sample_design_integrity_failure() -> tuple[str, list[dict]]:
    return (
        "**策略样本设计结果完整性校验失败**：计划缓存与 canonical sample-design "
        "bundle/artifact 摘要不一致，已停止展示任何样本指标。请重新运行样本设计；"
        "下载接口仍会按 TaskArtifact 注册 hash 校验产物。",
        [],
    )


def _sample_design_v2_integrity_failure() -> tuple[str, list[dict]]:
    return (
        "**策略样本设计 V2 结果完整性校验失败**：计划缓存与 canonical "
        "sample-design bundle、membership 或 artifact 摘要不一致，已停止展示"
        "所有样本数量、身份与下载链接。请重新运行策略样本设计；统一产物栏只会"
        "提供通过 TaskArtifact 注册 hash 校验的产物。",
        [],
    )


def _model_evidence_v2_integrity_failure() -> tuple[str, list[dict]]:
    return (
        "**策略分析证据 V2 结果完整性校验失败**：计划缓存与 canonical "
        "sample-design/model-evidence bundle 或 artifact 摘要不一致，已停止展示"
        "变量、分箱、观测、身份与产物信息。请重新运行策略分析证据生成；统一"
        "产物栏只会展示注册并校验通过的 TaskArtifact。",
        [],
    )


def _model_score_comparison_integrity_failure() -> tuple[str, list[dict]]:
    return (
        "**模型评分比较证据完整性校验失败**：计划缓存未能证明结果保持"
        " `no_selection`、未采纳和未部署状态，已停止展示模型、指标与产物信息。"
        "请重新生成比较证据；平台不会从异常载荷推断或回显冠军。",
        [],
    )


def _project_context_integrity_failure() -> tuple[str, list[dict]]:
    return (
        "**策略项目上下文完整性校验失败**：计划缓存与 immutable revision/artifact "
        "不一致，已停止展示项目现状、历史和缺失信息。请重新刷新项目上下文。",
        [],
    )


def _render_materialize_project_context(o: dict):
    """Render only the authenticated project-context revision projection."""

    from marvis.packs.strategy.errors import StrategyError
    from marvis.packs.strategy.project_context_tools import (
        validate_materialize_project_context_tool_output,
    )

    try:
        o = validate_materialize_project_context_tool_output(o)
    except (StrategyError, RecursionError):
        return _project_context_integrity_failure()

    revision = o["revision"]
    state = revision["state"]
    current = state["current_project_snapshot"]
    histories = state["historical_strategy_reviews"]
    missing = state["missing_information_records"]
    artifact = o["context_artifact"]
    action = "已创建" if o["created"] else "未变化，已复用"
    text = (
        f"**策略项目上下文{action}**：revision **{revision['revision']}**，"
        f"截止 `{state['as_of']}`；绑定 {len(state['source_refs'])} 个可校验证据引用、"
        f"{len(histories)} 个历史策略版本，当前有 {len(missing)} 项缺失信息。\n"
        f"上下文 artifact：[{artifact['filename']}]({artifact['download_url']})，"
        f"content hash `{artifact['content_hash']}`。"
    )
    if o["external_artifacts"]:
        text += (
            f"\n- 已将 {len(o['external_artifacts'])} 份外部材料按原始字节快照；"
            "平台未从中提取或猜测任何业务指标。"
        )
    if missing:
        text += (
            "\n- 需要补充的问题已按字段和阻断级别记录；用户说明“暂时没有”后，"
            "报告会保留为空并标记 unavailable，不会填成 0。"
        )

    status_rows = []
    for field_name in ("volume", "approval", "risk", "economics"):
        field = current["status_fields"][field_name]
        status_rows.append(
            [
                field_name,
                field["availability"],
                _fmt(field["value"]) if field["availability"] == "present" else "—",
                field.get("as_of") or "—",
                field.get("note") or "",
            ]
        )
    tables = [
        {
            "title": "当前项目状态证据",
            "columns": ["字段", "状态", "值", "截至", "说明"],
            "rows": status_rows,
        }
    ]
    if histories:

        def history_value(item, field_name):
            field = item[field_name]
            return field["value"] if field["availability"] == "present" else "—"

        tables.append(
            {
                "title": "历史策略版本",
                "columns": ["版本", "资产状态", "生效期", "可用性", "范围"],
                "rows": [
                    [
                        item.get("version") if item.get("version") is not None else "—",
                        history_value(item, "asset_status"),
                        (
                            f"{history_value(item, 'effective_period')['start']} ~ "
                            f"{history_value(item, 'effective_period')['end'] or '持续'}"
                            if history_value(item, "effective_period") != "—"
                            else "—"
                        ),
                        item["availability"],
                        history_value(item, "scope"),
                    ]
                    for item in histories
                ],
            }
        )
    if missing:
        tables.append(
            {
                "title": "待补充信息",
                "columns": ["字段", "阻断级别", "状态", "问题"],
                "rows": [
                    [
                        item["field_path"],
                        item["blocking"],
                        item["status"],
                        item["question"],
                    ]
                    for item in missing
                ],
            }
        )
    return text, tables


def _sample_design_metric_value(value: object, *, unit: str, status: str) -> str:
    if status != "present":
        return "n/a"
    if unit == "ratio":
        return _pct(value)
    return _num(value)


def _render_materialize_sample_design(o: dict):
    """Render only observations from the strictly validated Tool envelope."""

    from marvis.packs.strategy.errors import StrategyError
    from marvis.packs.strategy.sample_design_tools import (
        validate_materialize_sample_design_tool_output,
    )

    try:
        o = validate_materialize_sample_design_tool_output(o)
    except (StrategyError, RecursionError):
        return _sample_design_integrity_failure()

    bundle = o["bundle"]
    design = bundle["sample_design"]
    boundary = design["active_dataset_boundary"]
    performance = design["performance_window"]
    observation = design["observation_window"]
    target = design["target_definition"]
    split = design["split_definition"]
    optional = design["optional_fields"]
    lifecycle = design["lifecycle"]
    artifact = o["artifact"]
    performance_text = (
        f"provided / {performance['days']} 天"
        if performance["status"] == "provided"
        else "unavailable"
    )
    observation_text = (
        f"provided / {observation['start']} 至 {observation['end']}"
        if observation["status"] == "provided"
        else "unavailable"
    )
    text = (
        f"**策略样本设计已固化**：`{o['sample_design_id']}`，content hash "
        f"`{o['content_hash']}`；活动样本 **{_num(boundary['population_count'])}** 行。"
        f"表现窗 `{performance_text}`，观察窗 `{observation_text}`，"
        f"成熟度 `{design['maturity']}`，目标列 `{target['column']}`（坏样本值 "
        f"`{target['bad_value']}`，好样本值 `{target['good_value']}`），"
        f"样本切分 `{split['status']}`。\n"
        f"当前生命周期 `{lifecycle['candidate_stage']} / "
        f"{lifecycle['validation_status']}`；**只生成样本证据，未创建或修改策略、"
        "未建模、未建树、未入池、未采纳、未部署。**"
    )
    if design["scope"] == "exploration_only":
        text += "\n- 当前为 `exploration-only`：不能声称样本已成熟或已完成独立验证。"
    if split["status"] == "unavailable":
        text += "\n- 开发/验证/OOT 切分 unavailable；仅展示 overall 样本证据。"
    missing_optional = [
        label
        for field, label in (
            ("month_field", "月份列"),
            ("weight_field", "权重列"),
            ("loan_amount_field", "放款金额列"),
            ("overdue_amount_field", "逾期金额列"),
        )
        if optional[field] is None
    ]
    if missing_optional:
        text += "\n- 可选口径 unavailable：" + "、".join(missing_optional) + "。"
    if artifact.get("download_url"):
        text += (
            "\n\n**样本设计证据**："
            f"[{artifact['filename']}]({artifact['download_url']})"
        )

    definitions = {
        item["metric_definition_id"]: item for item in bundle["metric_definitions"]
    }
    rows_by_kind: dict[str, list[list[str]]] = {"overall": [], "split": []}
    for item in bundle["metric_observations"]:
        definition = definitions[item["metric_definition_ref"]["metric_definition_id"]]
        dimension = item["dimension"]
        rows_by_kind[dimension["kind"]].append(
            [
                str(dimension["value"]),
                str(definition["metric_key"]),
                str(definition["display_name"]),
                str(item["status"]),
                _sample_design_metric_value(
                    item["value"],
                    unit=str(definition["unit"]),
                    status=str(item["status"]),
                ),
                _num(item["numerator"]),
                _num(item["denominator"]),
                _num(item["sample_count"]),
            ]
        )
    columns = [
        "样本维度",
        "指标键",
        "指标",
        "状态",
        "值",
        "分子",
        "分母",
        "维度样本数",
    ]
    tables: list[dict] = []
    if rows_by_kind["overall"]:
        tables.append(
            {
                "title": "策略样本设计 Overall 指标",
                "columns": columns,
                "rows": rows_by_kind["overall"],
            }
        )
    if rows_by_kind["split"]:
        tables.append(
            {
                "title": "策略样本设计开发/验证/OOT 指标",
                "columns": columns,
                "rows": rows_by_kind["split"],
            }
        )
    red_flags = design["red_flags"]
    if red_flags:
        tables.append(
            {
                "title": "策略样本设计红旗",
                "columns": ["级别", "代码", "说明"],
                "rows": [
                    [flag["level"], flag["code"], flag["message"]] for flag in red_flags
                ],
            }
        )
    if o["warnings"]:
        text += "\n\n**样本设计提示**：" + "；".join(o["warnings"])
    return text, tables


def _strategy_sample_v2_population_rows(bundle: dict) -> list[list[str]]:
    rows: list[list[str]] = []
    for population in bundle["populations"]:
        partitions = {
            item["name"]: item["row_count"] for item in population["partitions"]
        }
        maturity = population["maturity_evidence"]
        rows.append(
            [
                population["role"],
                _num(population["total_count"]),
                _num(partitions["development"]),
                _num(partitions["validation"]),
                _num(partitions["oot"]),
                maturity["status"],
            ]
        )
    return rows


def _render_materialize_sample_design_v2(o: dict):
    """Render only facts validated from the cached V2 sample envelope."""

    from marvis.packs.strategy.errors import StrategyError
    from marvis.packs.strategy.sample_design_v2_tools import (
        validate_materialize_sample_design_v2_tool_output,
    )

    try:
        o = validate_materialize_sample_design_v2_tool_output(o)
    except (StrategyError, RecursionError):
        return _sample_design_v2_integrity_failure()
    return _render_validated_sample_design_v2(o)


def _render_materialize_sample_design_v2_native(o: dict):
    """Render only facts validated from the native V2 sample envelope."""

    from marvis.packs.strategy.errors import StrategyError
    from marvis.packs.strategy.sample_design_v2_native_tools import (
        validate_materialize_sample_design_v2_native_tool_output,
    )

    try:
        o = validate_materialize_sample_design_v2_native_tool_output(o)
    except (StrategyError, RecursionError):
        return _sample_design_v2_integrity_failure()
    return _render_validated_sample_design_v2(
        o,
        native_source=True,
    )


def _render_validated_sample_design_v2(
    o: dict,
    *,
    native_source: bool = False,
):
    """Render one already authenticated V2 envelope."""

    bundle = o["bundle"]
    design = bundle["sample_design"]
    semantics = design["sample_semantics"]
    historical = bundle["historical_score"]
    risk_population = next(
        item for item in bundle["populations"] if item["role"] == "risk"
    )
    maturity = risk_population["maturity_evidence"]
    score_detail = historical["status"]
    if historical["status"] == "available":
        score_detail += f" / {historical['column']} / {historical['direction']}"
    else:
        score_detail += f" / {historical['reason']}"

    text = (
        f"**策略样本设计 V2 已固化**：范围 `{semantics['scope']}`，"
        f"人群关系 `{design['relationship']}`；已锁定 approval/risk 两个人群的 "
        "development、validation、OOT 分区。\n"
        f"- 风险样本成熟度：`{maturity['status']}`；历史评分：`{score_detail}`。\n"
        "- 本步骤只固化样本与诊断证据，未创建策略、未采纳、未部署。"
    )
    if native_source:
        text += f"\n- 样本来源：原生活动数据集 `{o['source_binding']['source_mode']}`。"
    if semantics["scope"] == "exploration_only":
        text += "\n- 当前范围为 exploration-only，不可声称已完成成熟样本验证。"
    if o["warnings"]:
        text += "\n- 诊断提示：" + "；".join(o["warnings"]) + "。"
    else:
        text += "\n- 诊断提示：无。"

    bundle_artifact = o["artifacts"]["bundle"]
    membership_artifact = o["artifacts"]["membership"]
    text += (
        "\n\n**样本产物摘要（非 registry 身份认证）**："
        f"bundle `{bundle_artifact['filename']}`，canonical content hash "
        f"`{bundle_artifact['content_hash']}`；membership "
        f"`{membership_artifact['filename']}`，membership semantic content hash "
        f"`{o['membership_content_hash']}`。\n"
        "- 下载请使用任务页统一产物栏。当前 Tool v2 envelope 不包含可信 "
        "download_url、registry artifact id 或 membership binary artifact hash，"
        "本渲染器不会自行拼接链接。"
    )

    tables: list[dict] = [
        {
            "title": "双人群样本分区",
            "columns": [
                "人群",
                "总样本数",
                "开发集",
                "验证集",
                "OOT",
                "成熟度",
            ],
            "rows": _strategy_sample_v2_population_rows(bundle),
        },
        {
            "title": "样本设计口径",
            "columns": ["项目", "状态/值", "补充说明"],
            "rows": [
                ["scope", semantics["scope"], "样本使用范围"],
                ["relationship", design["relationship"], "双人群关系"],
                ["risk maturity", maturity["status"], maturity["reason"] or ""],
                ["historical score", historical["status"], score_detail],
            ],
        },
        {
            "title": "样本诊断",
            "columns": ["类别", "代码", "状态", "说明"],
            "rows": [
                [
                    item["category"],
                    item["code"],
                    item["status"],
                    item["message"],
                ]
                for item in bundle["diagnostics"]
            ],
        },
        {
            "title": "样本产物摘要",
            "columns": ["角色", "类型", "文件", "cached 可验证 hash"],
            "rows": [
                [
                    "sample-design bundle",
                    bundle_artifact["kind"],
                    bundle_artifact["filename"],
                    bundle_artifact["content_hash"],
                ],
                [
                    "membership",
                    membership_artifact["kind"],
                    membership_artifact["filename"],
                    f"semantic:{o['membership_content_hash']}",
                ],
            ],
        },
    ]
    return text, tables


def _strategy_model_v2_evidence_rows(bundle: dict) -> list[list[str]]:
    rows: list[list[str]] = []
    for evidence in bundle["univariate_evidence"]:
        statuses = {
            "present": 0,
            "unavailable": 0,
            "not_matured": 0,
            "not_applicable": 0,
        }
        for observation in evidence["observations"]:
            statuses[observation["status"]] += 1
        rows.append(
            [
                evidence["feature"],
                evidence["analysis_variant"],
                _num(len(evidence["bins"])),
                _num(len(evidence["observations"])),
                _num(statuses["present"]),
                _num(statuses["unavailable"]),
                _num(statuses["not_matured"]),
                _num(statuses["not_applicable"]),
            ]
        )
    return rows


def _render_materialize_model_evidence_v2(o: dict):
    """Render authenticated V2 univariate evidence without inventing a download."""

    from marvis.packs.strategy.errors import StrategyError
    from marvis.packs.strategy.model_evidence_tools import (
        validate_materialize_model_evidence_v2_tool_output,
    )

    try:
        o = validate_materialize_model_evidence_v2_tool_output(o)
    except (StrategyError, RecursionError):
        return _model_evidence_v2_integrity_failure()

    bundle = o["bundle"]
    evidence = bundle["univariate_evidence"]
    artifact = o["artifact"]
    text = (
        f"**策略分析证据 V2 已固化**：已认证 **{len(evidence)}** 条单变量证据，"
        "证据范围为 `risk/development`。\n"
        "- 当前是 **univariate-only**：未创建模型、未进行模型比较、未采纳、"
        "未部署。\n"
        f"- 证据文件 `{artifact['filename']}`，content hash "
        f"`{artifact['content_hash']}`；下载请使用任务页统一产物栏。当前 Tool v3 "
        "envelope 不包含可信 download_url 或 registry artifact id，本渲染器不会"
        "自行拼接链接。"
    )
    tables = [
        {
            "title": "单变量分析证据",
            "columns": [
                "变量",
                "分析方法",
                "分箱数",
                "观测数",
                "present",
                "unavailable",
                "not_matured",
                "not_applicable",
            ],
            "rows": _strategy_model_v2_evidence_rows(bundle),
        }
    ]
    return text, tables


def _render_materialize_model_score_comparison_v2(o: dict):
    """Present deterministic same-sample evidence without implying selection."""

    comparison = o.get("comparison")
    governance = o.get("governance")
    artifact = o.get("artifact")
    if not all(
        isinstance(value, Mapping) for value in (comparison, governance, artifact)
    ):
        raise ValueError("model-score comparison presenter requires its envelope")
    assert isinstance(comparison, Mapping)
    assert isinstance(governance, Mapping)
    assert isinstance(artifact, Mapping)
    selection = comparison.get("selection")
    if (
        not isinstance(selection, Mapping)
        or selection.get("status") != "no_selection"
        or governance.get("selection_status") != "no_selection"
        or governance.get("winner_selected") is not False
        or governance.get("not_adopted") is not True
        or governance.get("not_deployed") is not True
    ):
        raise ValueError("model-score comparison governance is not nonselecting")

    metrics = [
        metric
        for metric in (comparison.get("metrics") or [])
        if isinstance(metric, Mapping)
    ]
    model_ids: list[str] = []
    for metric in metrics:
        for item in metric.get("model_values") or []:
            if not isinstance(item, Mapping):
                continue
            ref = item.get("model_evidence_ref")
            evidence_id = ref.get("evidence_id") if isinstance(ref, Mapping) else None
            if isinstance(evidence_id, str) and evidence_id not in model_ids:
                model_ids.append(evidence_id)

    rows = []
    for metric in metrics:
        values = {}
        for item in metric.get("model_values") or []:
            if not isinstance(item, Mapping):
                continue
            ref = item.get("model_evidence_ref")
            evidence_id = ref.get("evidence_id") if isinstance(ref, Mapping) else None
            if isinstance(evidence_id, str):
                values[evidence_id] = item.get("value")
        rows.append(
            [
                str(metric.get("metric_key") or ""),
                str(metric.get("period") or "整体"),
                str(metric.get("unit") or ""),
                *[_num(values.get(evidence_id)) for evidence_id in model_ids],
                _num(metric.get("delta")),
            ]
        )

    text = (
        "**模型评分比较证据已生成**："
        f"同一认证样本 `{o.get('population', '')} / {o.get('partition', '')}`，"
        f"比较 **{len(model_ids)}** 个模型的 **{len(metrics)}** 组共同指标。\n"
        "- 治理状态：**未选择冠军，未采纳、未部署**；本结果仅提供确定性比较证据。"
    )
    filename = artifact.get("filename")
    download_url = artifact.get("download_url")
    if isinstance(filename, str) and isinstance(download_url, str):
        text += f"\n- 证据文件：[{filename}]({download_url})"

    tables = []
    if rows:
        tables.append(
            {
                "title": "同样本模型评分指标",
                "columns": ["指标", "期间", "单位", *model_ids, "差值"],
                "rows": rows,
            }
        )
    return text, tables


def _render_build_strategy_report_bundle_v2(o: dict):
    """Render only a fully validated governed report publication envelope."""

    from marvis.packs.strategy.errors import StrategyError
    from marvis.packs.strategy.report_bundle_tools import (
        validate_build_strategy_report_bundle_v2_tool_output,
    )

    try:
        o = validate_build_strategy_report_bundle_v2_tool_output(o)
    except (StrategyError, RecursionError):
        return _strategy_report_integrity_failure()

    warnings = o["warnings"]
    warning_text = "；".join(warnings) if warnings else "无"
    download_labels = {
        "json": "JSON",
        "markdown": "Markdown",
        "xlsx": "XLSX",
        "docx": "DOCX",
    }
    downloads = " · ".join(
        f"[{download_labels[artifact['format']]}]({artifact['download_url']})"
        for artifact in o["artifacts"]
    )
    text = (
        f"**StrategyReportBundle V2 已生成**：报告 ID `{o['report_id']}`，"
        f"revision **{o['report_revision']}**，状态 **{o['status']}**。\n"
        "- 本报告生成步骤未创建或变更策略资产，也未执行采纳、部署或上线；"
        "报告中的生命周期状态来自已认证证据。\n"
        f"- 完整性警告：{warning_text}\n"
        f"- 下载：{downloads}"
    )
    return text, []


def _render_export_strategy_delivery(
    o: dict,
    *,
    trusted_task_id: str | None,
    trusted_inputs: Mapping[str, Any] | None,
    trusted_artifacts: Mapping[str, Any] | None,
):
    """Render the internally bound, hash-pinned Strategy DSL delivery."""

    from marvis.packs.strategy.dsl_delivery_tools import (
        DELIVERY_ARTIFACT_KINDS,
        StrategyDeliveryToolError,
        validate_export_strategy_delivery_tool_output,
        validate_strategy_delivery_artifact_records,
    )

    try:
        if (
            not isinstance(trusted_task_id, str)
            or not trusted_task_id
            or not isinstance(trusted_inputs, Mapping)
            or set(trusted_inputs)
            != {
                "strategy_ref",
                "dataset_ref",
                "workspace_ref",
                "maximum_equivalence_rows",
            }
            or not isinstance(trusted_artifacts, Mapping)
            or o.get("maximum_equivalence_rows")
            != trusted_inputs["maximum_equivalence_rows"]
        ):
            raise StrategyDeliveryToolError(
                "renderer is missing authenticated delivery context"
            )
        expected_artifacts = validate_strategy_delivery_artifact_records(
            trusted_artifacts,
            expected_task_id=trusted_task_id,
            expected_delivery_id=o.get("delivery_id"),
            expected_strategy_ref=trusted_inputs["strategy_ref"],
            expected_dataset_ref=trusted_inputs["dataset_ref"],
            expected_workspace_ref=trusted_inputs["workspace_ref"],
            expected_maximum_equivalence_rows=trusted_inputs[
                "maximum_equivalence_rows"
            ],
            expected_equivalence=o.get("equivalence"),
        )
        o = validate_export_strategy_delivery_tool_output(
            o,
            expected_task_id=trusted_task_id,
            expected_strategy_ref=trusted_inputs["strategy_ref"],
            expected_dataset_ref=trusted_inputs["dataset_ref"],
            expected_workspace_ref=trusted_inputs["workspace_ref"],
            expected_artifacts=expected_artifacts,
        )
    except (
        AttributeError,
        IndexError,
        KeyError,
        RecursionError,
        StrategyDeliveryToolError,
        TypeError,
    ):
        return _strategy_delivery_integrity_failure()

    download_labels = {
        DELIVERY_ARTIFACT_KINDS["python"]: "Python",
        DELIVERY_ARTIFACT_KINDS["sql"]: "DuckDB SQL",
        DELIVERY_ARTIFACT_KINDS["strategy_json"]: "Strategy JSON",
        DELIVERY_ARTIFACT_KINDS["equivalence_json"]: "Equivalence JSON",
    }
    downloads = " · ".join(
        f"[{download_labels[artifact['kind']]}]({artifact['download_url']})"
        for artifact in o["artifacts"]
    )
    equivalence = o["equivalence"]
    scope_status = "bounded" if equivalence["bounded"] else "full"
    text = (
        f"**Strategy DSL 离线交付已生成**：交付 ID `{o['delivery_id']}`，"
        f"策略类型 **{o['strategy_type']}**，版本 **{o['strategy_version']}**。\n"
        f"- 等价性校验：sample_count **{equivalence['sample_count']}** / "
        f"source_row_count **{o['source_row_count']}**（**{scope_status}**）。\n"
        "- 执行边界：**offline-only**；"
        "**not_applied=true / not_adopted=true / not_deployed=true**"
        "（未应用、未采纳、未部署）。\n"
        f"- 内容哈希固定下载：{downloads}"
    )
    return text, []


EVIDENCE_PRESENTERS = {
    "build_report_bundle_v2": _render_build_strategy_report_bundle_v2,
    "export_strategy_delivery": _render_export_strategy_delivery,
    "materialize_project_context": _render_materialize_project_context,
    "materialize_sample_design": _render_materialize_sample_design,
    "materialize_sample_design_v2": _render_materialize_sample_design_v2,
    "materialize_sample_design_v2_native": _render_materialize_sample_design_v2_native,
    "materialize_model_evidence_v2": _render_materialize_model_evidence_v2,
    "materialize_model_score_comparison_v2": _render_materialize_model_score_comparison_v2,
}

INTEGRITY_FAILURES = {
    "build_report_bundle_v2": _strategy_report_integrity_failure,
    "export_strategy_delivery": _strategy_delivery_integrity_failure,
    "materialize_sample_design": _sample_design_integrity_failure,
    "materialize_sample_design_v2": _sample_design_v2_integrity_failure,
    "materialize_sample_design_v2_native": _sample_design_v2_integrity_failure,
    "materialize_model_evidence_v2": _model_evidence_v2_integrity_failure,
    "materialize_model_score_comparison_v2": _model_score_comparison_integrity_failure,
    "materialize_project_context": _project_context_integrity_failure,
}

__all__ = ["EVIDENCE_PRESENTERS", "INTEGRITY_FAILURES"]
