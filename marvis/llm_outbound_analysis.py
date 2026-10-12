"""Analysis DTOs: complete active questions, column metadata and measured aggregates."""
from marvis.llm_outbound import OutboundContextError, _finite, _mint_prompt
from marvis.llm_outbound_instructions import column_identifiers, user_instruction


def field_names(value):
    if not isinstance(value, (list, tuple)) or not value or len(value) > 2000:
        raise OutboundContextError("分析字段列表为空或超限，未截断字段或发送请求。")
    names = column_identifiers(value)
    if len(set(value)) != len(value):
        raise OutboundContextError("分析字段列表存在重复项，请核对当前数据集。")
    return names


def slice_question_prompt(question, columns):
    names = field_names(columns)
    instruction = user_instruction(question)
    return _mint_prompt({"stage": "adhoc_slice_spec", "question": instruction,
        "columns": [{"name": name} for name in names],
        "instruction": "完整保留用户问题的分组、筛选、时间、否定和口径约束。只解析 schema 内规格或澄清，不计算数据结果；未知列或缺失口径必须澄清。未发送样本值和类别枚举，不能猜测用户未给出的筛选值。"})


# These exact governed labels are normalized locally. No arbitrary fragment of a
# dictionary description is promoted to an allowed role or direction.
ROLE_ALIASES = {
    "delinquency_count": {"历史逾期次数", "逾期次数", "overdue count", "delinquency count"},
    "delinquency_amount": {"逾期金额", "overdue amount"},
    "credit_score": {"征信评分", "信用评分", "credit score", "bureau score"},
    "income_amount": {"收入", "收入金额", "income"},
    "debt_amount": {"负债", "负债金额", "debt"},
    "used_credit": {"已用额度", "used credit"},
    "credit_limit": {"授信额度", "approved credit", "credit limit"},
    "deposit_balance": {"存款余额", "deposit balance"},
    "loan_balance": {"贷款余额", "loan balance"},
    "account_age": {"账户年龄", "账户存续月数", "account age"},
    "explicit_negative": {"余额越高风险越低"},
    "explicit_positive": {"余额越高风险越高"},
}
ROLES = set(ROLE_ALIASES) | {"unknown"}


def business_role(value):
    if isinstance(value, dict):
        role = value.get("semantic_role")
        if role in ROLES:
            return role
        value = next((value[key] for key in ("meaning", "business_name", "business_meaning") if isinstance(value.get(key), str)), None)
    if not isinstance(value, str):
        return "unknown"
    label = value.strip().casefold()
    return next((role for role, aliases in ROLE_ALIASES.items() if label in {item.casefold() for item in aliases}), "unknown")


def feature_direction_prompt(entries, *, target_col):
    names = field_names([name for name, _meaning in entries])
    field_names([target_col])
    return _mint_prompt({"stage": "feature_meaning_direction", "target": {"column": target_col,
        "role": "binary_risk_event", "risk_event_value": 1},
        "features": [{"feature": name, "business_role": business_role(meaning)} for name, meaning in entries],
        "instruction": "仅根据已声明的业务角色判断风险事件的预期方向；unknown 必须 uncertain/low，不按字段名或未发送的字典正文猜测含义。positive=越大风险越高；negative=越大风险越低；u_shape=两端风险高。不得生成或猜测任何统计指标。",
        "feature_count": len(names)})


def feature_cross_prompt(dictionary, metrics, max_candidates):
    if not isinstance(dictionary, dict):
        raise OutboundContextError("特征交叉需要当前字段字典结构，请核对当前数据集。")
    names = field_names(list(dictionary))
    if type(max_candidates) is not int or not 1 <= max_candidates <= 100:
        raise OutboundContextError("特征交叉候选数量超限，未发送请求。")
    allowed_metrics = {"iv", "ks", "auc", "coverage", "missing_rate", "correlation", "psi", "unique_count"}
    aggregates = []
    source = metrics if isinstance(metrics, dict) else {}
    for name in names:
        row = source.get(name)
        if isinstance(row, dict):
            clean = {key: value for key, value in row.items() if key in allowed_metrics and _finite(value)}
            if clean:
                aggregates.append({"feature": name, **clean})
    return _mint_prompt({"stage": "feature_cross_recommendations", "max_candidates": max_candidates,
        "features": [{"feature": name, "business_role": business_role(dictionary[name])} for name in names],
        "existing_metrics": aggregates,
        "instruction": "Recommend feature crosses only. Do not calculate or output new IV, KS, AUC, PSI, lift, or any derived metric. 只依据已提供的角色及聚合指标提出候选；未知业务角色不编造业务含义，明确候选需要确定性计算和验证。未发送原始字典描述、类别值或样本行。"})
