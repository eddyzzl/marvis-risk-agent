"""Conversation-first setup for the user-facing risk-analysis task.

The public task type remains ``vintage`` for compatibility, but an agent task
must not guess which business report the user wants.  This module persists a
small intake state in assistant-message metadata and only returns a driver
template after the selected report's deterministic data contract is satisfied.

The state is deliberately stored on assistant messages (the same pattern as
the join C1 and portfolio-state gates).  ``_run_driver_turn`` stores the current
user message before calling setup, so state lookup must ignore user messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

from marvis.agent.vintage_setup import VintageSetupError, build_vintage_proposal
from marvis.domain import FileRole
from marvis.files import scan_source_dir


RISK_ANALYSIS_INTAKE_META_KEY = "risk_analysis_intake"

VTG_TERMINAL = "vtg_terminal"
PROFITABILITY = "profitability"
STANDARD_VINTAGE = "standard_vintage"

_REPORT_TEMPLATE_ID = "risk_analysis_report"


@dataclass(frozen=True)
class RiskAnalysisSetupDecision:
    """One setup transition, either a chat pause or a ready driver contract."""

    content: str
    metadata: dict
    template_id: str | None = None
    slots: dict | None = None

    @property
    def intake_state(self) -> dict:
        state = self.metadata.get(RISK_ANALYSIS_INTAKE_META_KEY)
        return dict(state) if isinstance(state, dict) else {}


@dataclass(frozen=True)
class _MaterialContract:
    analysis_kind: str
    table_name: str
    grain: str
    required: tuple[str, ...]
    optional: tuple[str, ...]
    one_of: tuple[str, ...] = ()
    alternative_groups: tuple[tuple[str, ...], ...] = ()
    conditional_groups: Mapping[str, tuple[tuple[str, ...], ...]] | None = None
    aliases: Mapping[str, tuple[str, ...]] | None = None

    @property
    def all_columns(self) -> tuple[str, ...]:
        alternatives = tuple(
            column for group in self.alternative_groups for column in group
        )
        conditional = tuple(
            column
            for groups in (self.conditional_groups or {}).values()
            for group in groups
            for column in group
        )
        return tuple(
            dict.fromkeys(
                (
                    *self.required,
                    *self.one_of,
                    *alternatives,
                    *conditional,
                    *self.optional,
                )
            )
        )


_VTG_CONTRACT = _MaterialContract(
    analysis_kind=VTG_TERMINAL,
    table_name="canonical VTG input",
    grain="汇总模式每个 product × cohort 一行；原始余额曲线模式每个 product × cohort × MOB 一行",
    required=(
        "product",
        "as_of_date",
        "cohort",
        "amount_unit",
        "disbursement_amount",
        "mob14_bad_rate",
    ),
    one_of=(
        "terminal_bad_rate",
        "long_term_recovery_rate",
        "auxiliary_terminal_bad_rate",
    ),
    alternative_groups=(
        ("turnover",),
        ("mob", "mob_days", "day_count_basis", "mob_balance_rate"),
        ("mob", "mob_days", "day_count_basis", "mob_balance_amount"),
    ),
    optional=(
        "avg_daily_balance",
        "scenario",
        "channel",
        "tenor_months",
        "terminal_method",
        "selection_rule",
        "previous_mob14_bad_rate",
        "previous_terminal_bad_rate",
        "previous_turnover",
        "previous_annualized_bad_rate",
        "previous_disbursement_amount",
        "previous_avg_daily_balance",
        "auxiliary_terminal_bad_rate",
    ),
    aliases={
        "product": ("产品", "产品名称", "product_name"),
        "as_of_date": ("截面日", "数据日期", "测算日期", "as_of_period"),
        "cohort": ("放款月", "放款月份", "cohort_month", "vintage"),
        "amount_unit": ("金额单位", "币种单位"),
        "disbursement_amount": ("放款金额", "放款额", "disbursed_amount"),
        "mob14_bad_rate": (
            "mob14不良率",
            "mob14_bad",
            "14mob_bad_rate",
            "MOB14 vtg30+",
            "MOB14_vtg30+",
        ),
        "turnover": ("周转", "周转次数", "年周转", "annual_turnover"),
        "mob": ("MOB", "账龄", "月龄", "mob_month"),
        "mob_days": ("MOB天数", "当期天数", "天数", "day_weight", "days_in_mob"),
        "day_count_basis": ("计息天数基础", "年化天数", "day_basis"),
        "mob_balance_rate": (
            "MOB余额率",
            "余额率",
            "balance_ratio",
            "balance_curve_rate",
        ),
        "mob_balance_amount": ("MOB余额", "余额金额", "balance_amount", "mob_balance"),
        "terminal_bad_rate": ("终值不良率", "vtg终值", "terminal_rate"),
        "long_term_recovery_rate": ("长期回收率", "终期回收率"),
        "avg_daily_balance": ("日均余额", "平均日余额", "月日均余额", "年日均余额"),
        "terminal_method": ("终值方法", "终值口径", "terminal_basis"),
        "selection_rule": ("选择规则", "终值选择规则"),
        "scenario": ("方案", "场景", "版本", "对比口径", "scenario_name"),
        "channel": ("渠道", "channel_name"),
        "tenor_months": ("期限", "期数", "loan_tenor_months"),
        "previous_mob14_bad_rate": ("上期MOB14不良率",),
        "previous_terminal_bad_rate": ("上期终值不良率",),
        "previous_turnover": ("上期周转",),
        "previous_annualized_bad_rate": ("上期年化不良率",),
        "previous_disbursement_amount": ("上期放款额", "上期放款金额"),
        "previous_avg_daily_balance": ("上期日均余额",),
        "auxiliary_terminal_bad_rate": ("辅助终值", "辅助终值不良率"),
    },
)

_PROFITABILITY_CONTRACT = _MaterialContract(
    analysis_kind=PROFITABILITY,
    table_name="canonical economics",
    grain="每个 product × as_of_period × scenario × asset_class 一行",
    required=(
        "product",
        "as_of_period",
        "asset_class",
        "weight",
        "weight_basis",
        "customer_rate",
        "acquisition_cost_rate",
        "payment_cost_rate",
        "collection_cost_rate",
        "funding_cost_rate",
        "other_cost_rate",
    ),
    conditional_groups={
        "风险成本": (
            ("risk_cost_rate",),
            ("terminal_vintage_rate", "risk_turnover"),
        ),
        "息费损失": (
            ("interest_loss_rate",),
            ("loss_timing_factor",),
        ),
        "分润成本": (
            ("revenue_share_rate",),
            ("profit_share_ratio",),
        ),
        "数据成本": (
            ("data_cost_rate",),
            (
                "per_application_cost",
                "credit_approval_rate",
                "draw_initiation_rate",
                "draw_approval_rate",
                "average_ticket",
                "data_annualization_factor",
                "amount_unit",
            ),
        ),
        "税费成本": (
            ("tax_rate",),
            ("tax_method", "tax_inclusive_divisor", "tax_combined_rate"),
        ),
    },
    optional=(
        "scenario",
        "amount_unit",
        "customer_stage",
        "transaction_weight",
    ),
    aliases={
        "product": ("产品", "产品名称", "product_name"),
        "as_of_period": ("as_of_date", "截面日", "截面期", "测算期", "数据日期"),
        "scenario": ("方案", "场景", "版本", "对比口径", "scenario_name"),
        "asset_class": ("资产类型", "资产类别", "asset_type"),
        "weight": ("权重", "占比"),
        "weight_basis": ("权重口径", "weight_type"),
        "customer_rate": ("客户利率", "客群利率"),
        "risk_cost_rate": ("风险成本率", "风险成本"),
        "funding_cost_rate": ("资金成本率", "资金成本"),
        "interest_loss_rate": ("息损率",),
        "revenue_share_rate": (
            "年化分润成本率",
            "资产口径分润成本率",
            "分润成本率",
        ),
        "acquisition_cost_rate": ("获客成本率",),
        "data_cost_rate": ("数据成本率",),
        "payment_cost_rate": ("支付成本率",),
        "collection_cost_rate": ("催收成本率",),
        "tax_rate": ("年化税费率", "资产口径税费率", "税费成本率"),
        "other_cost_rate": ("其他成本率",),
        "terminal_vintage_rate": ("VTG终值", "终值不良率", "terminal_bad_rate"),
        "risk_turnover": ("风险周转次数", "年周转次数", "risk_annual_turnover"),
        "loss_timing_factor": ("损失时点系数", "loss_factor"),
        "profit_share_ratio": ("合同分润比例", "分润比例", "contract_share_ratio"),
        "per_application_cost": ("单笔申请成本", "申请成本", "application_unit_cost"),
        "credit_approval_rate": ("授信通过率",),
        "draw_initiation_rate": ("用信发起率",),
        "draw_approval_rate": ("用信通过率",),
        "average_ticket": ("件均", "平均件均", "average_loan_amount"),
        "data_annualization_factor": ("数据成本年化系数", "年化系数"),
        "tax_method": ("税费方法", "税费口径"),
        "tax_inclusive_divisor": ("含税除数", "价税分离系数"),
        "tax_combined_rate": ("综合税费率", "附加税率"),
        "amount_unit": ("金额单位", "币种单位"),
        "customer_stage": ("客户阶段", "首复购阶段", "stage"),
        "transaction_weight": ("笔数权重", "交易权重", "stage_weight"),
    },
)

_CONTRACTS = {
    VTG_TERMINAL: _VTG_CONTRACT,
    PROFITABILITY: _PROFITABILITY_CONTRACT,
}


def advance_risk_analysis_setup(
    registry,
    backend,
    task_id: str,
    source_dir,
    *,
    user_text: str | None,
    conversation: Sequence[dict],
    target_col: str | None = None,
    time_col: str | None = None,
) -> RiskAnalysisSetupDecision:
    """Advance the persisted intake state by one user turn.

    No-data and bad-contract conditions are normal chat pauses, not setup
    exceptions.  A ready decision is the only result carrying a template id.
    """

    previous = latest_risk_analysis_intake(conversation)
    if previous is None:
        return _ask_goal()

    phase = str(previous.get("phase") or "")
    if phase == "ask_goal":
        analysis_kind = parse_analysis_kind(user_text)
        if analysis_kind is None:
            return _ask_goal(clarify=True)
        return _request_materials(
            analysis_kind,
            analysis_scope=_bounded_scope(user_text),
        )

    analysis_kind = str(previous.get("analysis_kind") or "")
    analysis_scope = _bounded_scope(previous.get("analysis_scope"))
    changed_kind = _parse_explicit_analysis_switch(user_text)
    if changed_kind is not None and changed_kind != analysis_kind:
        return _request_materials(
            changed_kind,
            analysis_scope=_bounded_scope(user_text),
        )
    if analysis_kind not in {VTG_TERMINAL, PROFITABILITY, STANDARD_VINTAGE}:
        return _ask_goal(clarify=True)

    datasets = _ensure_registered_datasets(registry, task_id, source_dir)
    if not datasets:
        return _await_no_data(analysis_kind, analysis_scope=analysis_scope)

    if analysis_kind == STANDARD_VINTAGE:
        return _prepare_standard_vintage(
            registry,
            backend,
            task_id,
            source_dir,
            target_col=target_col,
            time_col=time_col,
            analysis_scope=analysis_scope,
        )

    contract = _CONTRACTS[analysis_kind]
    match = _best_contract_match(registry, backend, datasets, contract)
    if match["missing_columns"]:
        return _await_missing_columns(
            analysis_kind,
            contract,
            match,
            analysis_scope=analysis_scope,
        )
    return _ready_report(
        analysis_kind,
        contract,
        match,
        analysis_scope=analysis_scope,
    )


def latest_risk_analysis_intake(conversation: Sequence[dict]) -> dict | None:
    """Return the latest assistant-owned intake state.

    User messages are intentionally ignored: the current user turn has already
    been appended when setup runs.
    """

    for message in reversed(conversation):
        if message.get("role") != "assistant":
            continue
        state = (message.get("metadata") or {}).get(RISK_ANALYSIS_INTAKE_META_KEY)
        if isinstance(state, dict):
            return dict(state)
    return None


def parse_analysis_kind(user_text: str | None) -> str | None:
    text = _normalize_text(user_text)
    if not text:
        return None
    # Specific terminal/annualized requests must win over the generic
    # ``vintage`` token.
    if (
        "vtg终值" in text
        or "终值与年化不良" in text
        or ("终值" in text and "年化不良" in text)
        or "vtgterminal" in text
    ):
        return VTG_TERMINAL
    if any(
        token in text
        for token in ("收益测算", "盈利测算", "利润测算", "profitability", "economics")
    ):
        return PROFITABILITY
    if any(
        token in text
        for token in (
            "标准vintage",
            "vintage分析",
            "标准账龄",
            "账龄曲线",
            "standardvintage",
        )
    ):
        return STANDARD_VINTAGE
    return None


def _parse_explicit_analysis_switch(user_text: str | None) -> str | None:
    text = _normalize_text(user_text)
    if not text or not any(
        token in text
        for token in ("改做", "改为", "切换到", "切换为", "换成", "重新选择")
    ):
        return None
    return parse_analysis_kind(user_text)


def material_contract(analysis_kind: str) -> dict:
    """Expose a JSON-safe contract for UI/tests without leaking internals."""

    if analysis_kind == STANDARD_VINTAGE:
        return {
            "table": "vintage panel",
            "grain": "每个 cohort × MOB 一行（或账户 × MOB 明细）",
            "required_columns": ["cohort", "mob", "bad"],
            "one_of_columns": [],
            "optional_columns": ["product", "loan_id", "exposure"],
        }
    contract = _CONTRACTS[analysis_kind]
    return {
        "table": contract.table_name,
        "grain": contract.grain,
        "required_columns": list(contract.required),
        "one_of_columns": list(contract.one_of),
        "alternative_column_groups": [
            list(group) for group in contract.alternative_groups
        ],
        "conditional_column_groups": {
            label: [list(group) for group in groups]
            for label, groups in (contract.conditional_groups or {}).items()
        },
        "optional_columns": list(contract.optional),
    }


def _ask_goal(*, clarify: bool = False) -> RiskAnalysisSetupDecision:
    prefix = (
        "我还不能确定你要做哪一种风险分析。"
        if clarify
        else "先确认一下：你想分析什么？"
    )
    content = (
        f"{prefix}\n"
        "1. **VTG终值与年化不良**：按产品/cohort 测算终值、周转与年化不良；\n"
        "2. **收益测算**：按产品和资产类型汇总收入、风险/资金/运营成本与收益；\n"
        "3. **标准 Vintage**：按 cohort × MOB 生成坏账曲线。\n"
        "请直接回复其中一种，并补充分析范围或对比口径（如产品、观察期、基准版本）。"
    )
    return _chat_decision(content, {"phase": "ask_goal"})


def _request_materials(
    analysis_kind: str,
    *,
    analysis_scope: str | None = None,
) -> RiskAnalysisSetupDecision:
    scope_line = (
        f"\n- 已记录你的目标/范围：{analysis_scope}；报表覆盖上传表中的全部对应行。"
        if analysis_scope
        else ""
    )
    if analysis_kind == VTG_TERMINAL:
        content = (
            "目标已确认：**VTG终值与年化不良**。请提供一张 canonical VTG input：\n"
            "- 汇总模式粒度：每个 product × cohort 一行，并提供 turnover；\n"
            "- 原始余额曲线模式粒度：每个 product × cohort × MOB 一行，提供 "
            "mob, mob_days, day_count_basis，以及 mob_balance_rate 或 mob_balance_amount；"
            "这两个余额字段必须是每个 MOB 的平均日余额率/金额，不是月末余额；系统会按你提供的"
            " day_count_basis 自行推导年日均余额和周转次数；\n"
            "- 两种模式共同必需列：product, as_of_date, cohort, amount_unit, "
            "disbursement_amount, mob14_bad_rate；放款金额必须大于 0；\n"
            "- 终值输入三选一：terminal_bad_rate；或 long_term_recovery_rate；或同时提供 "
            "terminal_method=min_mob14_auxiliary 与 auxiliary_terminal_bad_rate；\n"
            "- 可选列：scenario, channel, tenor_months, selection_rule, avg_daily_balance, "
            "previous_terminal_bad_rate, previous_turnover, previous_annualized_bad_rate, "
            "auxiliary_terminal_bad_rate, terminal_method；\n"
            "- 一份 VTG 组合报告只能包含一个 as_of_date 和一个 scenario；"
            "多截面/多场景请分表上传并分别生成，避免重复计入组合；\n"
            "- 口径：所有百分比用小数（例如 2.5% 填 0.025），turnover 填年周转；"
            "mob14_bad_rate 必须已经是实际表现到 MOB14 或用明确方法投影到 MOB14 的 VTG30+，"
            "不能直接填尚只表现到 MOB3/MOB6 的当前值；"
            "若辅助终值与由长期回收率推导的终值都提供，还必须显式提供 "
            "selection_rule=min_auxiliary_recovery，才会采用 terminal = min(辅助终值, "
            "由回收率推导的终值)。不同产品的终值方法可能不同，"
            "请用 terminal_method 说明，不会由系统猜测。"
            f"{scope_line}\n"
            "上传后回复“材料已上传”，我会逐表检查列契约，缺列时会明确指出。"
        )
    elif analysis_kind == PROFITABILITY:
        content = (
            "目标已确认：**收益测算**。请提供一张规范化数据表：\n"
            "- 表：canonical economics；\n"
            "- 粒度：每个 product × as_of_period × scenario × asset_class 一行；单一方案可省略 "
            "scenario，系统记为“基准”；\n"
            "- 共同必需列：product, as_of_period, asset_class, weight, weight_basis, "
            "customer_rate, acquisition_cost_rate, payment_cost_rate, collection_cost_rate, "
            "funding_cost_rate, other_cost_rate；weight_basis 必须是 average_balance；\n"
            "- 风险成本二选一：risk_cost_rate；或 terminal_vintage_rate + risk_turnover；\n"
            "- 息费损失二选一：interest_loss_rate；或 loss_timing_factor（系统按 "
            "customer_rate × risk_cost_rate × factor 推导）；\n"
            "- 分润成本二选一：revenue_share_rate（已年化成本率）；或 "
            "profit_share_ratio（合同分润比例，系统换算为 revenue_share_rate）；\n"
            "- acquisition_cost_rate 只用于与上述合同分润不同的独立获客成本；"
            "若没有必须显式填 0，不能把同一笔合同分润同时填入两个字段；\n"
            "- 数据成本二选一：data_cost_rate；或 per_application_cost, credit_approval_rate, "
            "draw_initiation_rate, draw_approval_rate, average_ticket, data_annualization_factor；\n"
            "  若首笔/复购分行提供，再增加 customer_stage 与 transaction_weight；系统先逐阶段"
            "推导，再按笔数权重汇总为资产类别数据成本（样例 1:9 不会硬编码）；\n"
            "- 税费二选一：tax_rate；或 tax_method=sample_net_revenue_vat_surcharge, "
            "tax_inclusive_divisor, tax_combined_rate；该样例方法的税基明确为对客利率减息费损失、"
            "分润、独立获客、数据、支付和催收成本，不扣风险/资金成本；样例中的 /2、1.06、6.72% "
            "等常数必须作为输入，不会埋成默认值；\n"
            "- 可选列：scenario, amount_unit, customer_stage, transaction_weight；原始金额驱动项"
            "必须说明 amount_unit。\n"
            "- 口径：每行都要使用一致的计量单位和年化期间；所有比例用小数；"
            "weight 是同一产品/期间/方案下的资产权重并应合计为 1；请说明税费是含税、"
            "税前扣除还是税后口径；revenue_share_rate 必须是已换算到资产收益率口径的"
            "分润成本率，不能直接填合同分润比例；原始合同分润请填 profit_share_ratio。"
            "没有发生且没有推导驱动项的成本必须显式填 0，"
            "系统不会把缺失成本静默当成 0。"
            f"{scope_line}\n"
            "上传后回复“材料已上传”，我会逐表检查列契约，缺列时会明确指出。"
        )
    else:
        content = (
            "目标已确认：**标准 Vintage**。请提供 Vintage panel：\n"
            "- 表：vintage panel；\n"
            "- 粒度：每个 cohort × MOB 一行，或账户 × MOB 明细；\n"
            "- 必需列：cohort（放款批次/月）、mob（月龄）、bad（坏账标签或坏账增量）；\n"
            "- 可选列：product, loan_id, exposure；\n"
            "- 口径：请说明 bad 是 incremental（当期新增）还是 snapshot（截至当期状态），"
            "不能由系统猜测。\n"
            f"{scope_line}\n"
            "上传后回复“材料已上传”，我会检查字段并生成标准 Vintage 计划。"
        )
    state = {"phase": "request_materials", "analysis_kind": analysis_kind}
    if analysis_scope:
        state["analysis_scope"] = analysis_scope
    return _chat_decision(
        content,
        state,
        contract=material_contract(analysis_kind),
    )


def _await_no_data(
    analysis_kind: str,
    *,
    analysis_scope: str | None = None,
) -> RiskAnalysisSetupDecision:
    state = {"phase": "await_materials", "analysis_kind": analysis_kind}
    if analysis_scope:
        state["analysis_scope"] = analysis_scope
    return _chat_decision(
        "还没有检测到已登记的数据表。请按上面的表/粒度/字段清单上传 CSV、XLSX 或 Parquet；"
        "上传完成后回复“材料已上传”，我再做契约检查。",
        state,
        contract=material_contract(analysis_kind),
    )


def _await_missing_columns(
    analysis_kind: str,
    contract: _MaterialContract,
    match: dict,
    *,
    analysis_scope: str | None = None,
) -> RiskAnalysisSetupDecision:
    missing = list(match["missing_columns"])
    observed = ", ".join(match["observed_columns"]) or "（无可读列）"
    content = (
        f"已检查最接近契约的数据表 `{match['dataset_name']}`，目前还不能启动测算。\n"
        f"缺少：{', '.join(missing)}。\n"
        f"已识别列：{observed}。\n"
        "请补齐这些列后重新上传；我不会用相似但含义不明的字段代替，也不会在缺列时启动计划。"
    )
    state = {
        "phase": "await_materials",
        "analysis_kind": analysis_kind,
        "dataset_id": match["dataset_id"],
        "missing_columns": missing,
    }
    if analysis_scope:
        state["analysis_scope"] = analysis_scope
    return _chat_decision(
        content, state, contract=material_contract(contract.analysis_kind)
    )


def _ready_report(
    analysis_kind: str,
    contract: _MaterialContract,
    match: dict,
    *,
    analysis_scope: str | None = None,
) -> RiskAnalysisSetupDecision:
    column_map = dict(match["column_map"])
    state = {
        "phase": "ready",
        "analysis_kind": analysis_kind,
        "dataset_id": match["dataset_id"],
        "column_map": column_map,
    }
    if analysis_scope:
        state["analysis_scope"] = analysis_scope
    scope_confirmation = (
        f"已保留目标/范围“{analysis_scope}”；具体覆盖范围以通过契约的上传表全部行为准。"
        if analysis_scope
        else "具体覆盖范围以上传表全部行为准。"
    )
    return RiskAnalysisSetupDecision(
        content=(
            f"材料契约已通过：将使用 `{match['dataset_name']}` 生成"
            f"{'VTG终值与年化不良' if analysis_kind == VTG_TERMINAL else '收益测算'}报告，"
            f"并在结果中列出关键指标、异常点、假设和可下载文件。{scope_confirmation}"
        ),
        metadata={
            "intent": "vintage",
            RISK_ANALYSIS_INTAKE_META_KEY: state,
            "risk_analysis_contract": material_contract(contract.analysis_kind),
        },
        template_id=_REPORT_TEMPLATE_ID,
        slots={
            "analysis_kind": analysis_kind,
            "dataset_id": match["dataset_id"],
            "column_map": column_map,
        },
    )


def _prepare_standard_vintage(
    registry,
    backend,
    task_id: str,
    source_dir,
    *,
    target_col: str | None,
    time_col: str | None,
    analysis_scope: str | None = None,
) -> RiskAnalysisSetupDecision:
    try:
        proposal = build_vintage_proposal(
            registry,
            backend,
            task_id,
            source_dir,
            target_col=target_col,
            time_col=time_col,
        )
    except VintageSetupError as exc:
        state = {
            "phase": "await_materials",
            "analysis_kind": STANDARD_VINTAGE,
            "validation_error": str(exc),
        }
        if analysis_scope:
            state["analysis_scope"] = analysis_scope
        return _chat_decision(
            f"标准 Vintage 材料还未通过字段检查：{exc} 请补充或重命名字段后重新上传。",
            state,
            contract=material_contract(STANDARD_VINTAGE),
        )
    state = {
        "phase": "ready",
        "analysis_kind": STANDARD_VINTAGE,
        "dataset_id": proposal.dataset_id,
        "column_map": {
            "cohort": proposal.cohort_col,
            "mob": proposal.mob_col,
            "bad": proposal.bad_col,
        },
    }
    if analysis_scope:
        state["analysis_scope"] = analysis_scope
    return RiskAnalysisSetupDecision(
        content=(
            f"标准 Vintage 材料已通过：样本 `{proposal.dataset_name}`，cohort "
            f"`{proposal.cohort_col}`，MOB `{proposal.mob_col}`，坏账列 `{proposal.bad_col}`。"
        ),
        metadata={
            "intent": "vintage",
            RISK_ANALYSIS_INTAKE_META_KEY: state,
            "risk_analysis_contract": material_contract(STANDARD_VINTAGE),
        },
        template_id=proposal.template_id,
        slots=proposal.template_slots(),
    )


def _chat_decision(
    content: str,
    state: dict,
    *,
    contract: dict | None = None,
) -> RiskAnalysisSetupDecision:
    metadata = {"intent": "vintage", RISK_ANALYSIS_INTAKE_META_KEY: dict(state)}
    if contract is not None:
        metadata["risk_analysis_contract"] = dict(contract)
    return RiskAnalysisSetupDecision(content=content, metadata=metadata)


def _ensure_registered_datasets(registry, task_id: str, source_dir) -> list:
    datasets = list(registry.list_for_task(task_id))
    if datasets:
        return datasets
    raw_source = str(source_dir or "").strip()
    if not raw_source:
        return []
    source_path = Path(raw_source)
    if not source_path.is_dir():
        return []
    try:
        artifacts = scan_source_dir(source_path)
    except (OSError, ValueError):
        return []
    for artifact in artifacts:
        if artifact.role != FileRole.SAMPLE:
            continue
        try:
            registry.register_from_upload(task_id, Path(artifact.path), role="sample")
        except Exception:
            # Intake remains a recoverable chat interaction. The normal upload
            # endpoint returns detailed ingest failures to the user; a stale or
            # unreadable source-dir file must not turn setup into a 500/error gate.
            continue
    return list(registry.list_for_task(task_id))


def _best_contract_match(
    registry, backend, datasets: Iterable, contract: _MaterialContract
) -> dict:
    matches = []
    for upload_order, dataset in enumerate(datasets):
        match = _contract_match(registry, backend, dataset, contract)
        match["upload_order"] = upload_order
        matches.append(match)
    return max(
        matches,
        key=lambda item: (
            -len(item["missing_columns"]),
            int(item["upload_order"]),
        ),
    )


def _contract_match(registry, backend, dataset, contract: _MaterialContract) -> dict:
    columns = _dataset_columns(registry, backend, dataset)
    lookup = {
        _normalize_column(column): column for column in columns if str(column).strip()
    }
    aliases = contract.aliases or {}
    column_map: dict[str, str] = {}
    for canonical in contract.all_columns:
        candidates = (canonical, *aliases.get(canonical, ()))
        actual = next(
            (
                lookup.get(_normalize_column(candidate))
                for candidate in candidates
                if _normalize_column(candidate) in lookup
            ),
            None,
        )
        if actual is not None:
            column_map[canonical] = actual
    missing = [column for column in contract.required if column not in column_map]
    if contract.one_of and not any(column in column_map for column in contract.one_of):
        missing.append(" 或 ".join(contract.one_of) + "（至少一个）")
    if (
        contract.analysis_kind == VTG_TERMINAL
        and "auxiliary_terminal_bad_rate" in column_map
        and "terminal_bad_rate" not in column_map
        and "long_term_recovery_rate" not in column_map
        and "terminal_method" not in column_map
    ):
        missing.append(
            "terminal_method（仅用 auxiliary_terminal_bad_rate 推导终值时必需）"
        )
    if (
        contract.analysis_kind == VTG_TERMINAL
        and "long_term_recovery_rate" in column_map
        and "auxiliary_terminal_bad_rate" in column_map
        and "selection_rule" not in column_map
    ):
        missing.append("selection_rule（回收推导终值与辅助终值同时存在时必需）")
    if contract.alternative_groups and not any(
        all(column in column_map for column in group)
        for group in contract.alternative_groups
    ):
        group_labels = [
            group[0] if len(group) == 1 else "(" + " + ".join(group) + ")"
            for group in contract.alternative_groups
        ]
        missing.append(" 或 ".join(group_labels) + "（满足一组）")
    for label, groups in (contract.conditional_groups or {}).items():
        if any(all(column in column_map for column in group) for group in groups):
            continue
        group_labels = [
            group[0] if len(group) == 1 else "(" + " + ".join(group) + ")"
            for group in groups
        ]
        missing.append(f"{label}: " + " 或 ".join(group_labels) + "（满足一组）")
    if (
        contract.analysis_kind == PROFITABILITY
        and "customer_stage" in column_map
        and "transaction_weight" not in column_map
    ):
        missing.append("transaction_weight（customer_stage 分行时必需）")
    return {
        "dataset_id": str(getattr(dataset, "id", "")),
        "dataset_name": _dataset_name(dataset),
        "row_count": int(getattr(dataset, "row_count", 0) or 0),
        "observed_columns": columns,
        "column_map": column_map,
        "missing_columns": missing,
    }


def _dataset_columns(registry, backend, dataset) -> list[str]:
    try:
        return [
            str(column)
            for column in backend.column_names(registry.resolve_path(dataset.id))
        ]
    except Exception:
        profiles = getattr(dataset, "columns", ()) or ()
        return [str(getattr(profile, "name", profile)) for profile in profiles]


def _dataset_name(dataset) -> str:
    sheet = str(getattr(dataset, "sheet", "") or "").strip()
    if sheet:
        return sheet
    source = str(getattr(dataset, "source_path", "") or "").strip()
    return Path(source).stem if source else str(getattr(dataset, "id", ""))


def _normalize_column(value: str) -> str:
    return re.sub(r"[\s_\-./（）()]+", "", str(value).strip().casefold())


def _normalize_text(value: str | None) -> str:
    return re.sub(r"[\s_\-./（）()]+", "", str(value or "").strip().casefold())


def _bounded_scope(value: object, *, max_chars: int = 240) -> str | None:
    text = " ".join(str(value or "").split())
    if not text:
        return None
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


__all__ = [
    "PROFITABILITY",
    "RISK_ANALYSIS_INTAKE_META_KEY",
    "RiskAnalysisSetupDecision",
    "STANDARD_VINTAGE",
    "VTG_TERMINAL",
    "advance_risk_analysis_setup",
    "latest_risk_analysis_intake",
    "material_contract",
    "parse_analysis_kind",
]
