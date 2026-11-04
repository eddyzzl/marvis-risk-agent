"""Strategy request compiler prompts (plan 5.2: the compiler no longer mints).

The user's request reaches the model verbatim through ``user_instruction``;
every column name passes ``column_identifiers``.  The installed contracts are
the compiler's own fixed text, passed in unchanged.  Rejected provider prose,
source code and exception bodies never enter a repair prompt.
"""

from marvis.llm_outbound import OutboundContextError, StructuralPrompt, _mint_prompt
from marvis.llm_outbound_instructions import column_identifiers, user_instruction


def _columns(whitelist, target_col):
    columns = column_identifiers(list(whitelist))
    if target_col is not None:
        column_identifiers([target_col])
    return columns


def strategy_request_prompt(
    utterance, whitelist, *, target_col, contract, aliases=None, directions=None
):
    return _mint_prompt({
        "instruction": str(user_instruction(utterance)),
        "instruction_origin": "interactive_user",
        "allowed_columns": _columns(whitelist, target_col),
        "target_col": target_col,
        **({"field_aliases": aliases} if aliases else {}),
        **({"field_directions": directions} if directions is not None else {}),
        "contract": contract,
    })


def strategy_sample_design_correction_prompt(
    utterance, whitelist, *, target_col, format_repair, contract, aliases=None
):
    return _mint_prompt({
        "instruction": str(user_instruction(utterance)),
        "instruction_origin": "interactive_user",
        "allowed_columns": _columns(whitelist, target_col),
        "target_col": target_col,
        **({"field_aliases": aliases} if aliases else {}),
        "repair": "format" if format_repair else "sample_design_semantics",
        "contract": contract,
    })


_REPAIR_INSTRUCTION = (
    "这是唯一一次修复机会。按完整原始指令、字段目录和输出契约重新生成；"
    "不要猜测缺失值。不能确定时只返回 clarification。禁止输出指标。"
)


def strategy_repair_prompt(prompt, *, failure_code):
    """Re-send the frozen structural request with only a platform failure code."""

    import json

    if not isinstance(prompt, StructuralPrompt):
        raise OutboundContextError("策略修复缺少冻结结构请求，未发送。")
    payload = json.loads(prompt)
    payload["repair"] = {"failure_code": failure_code, "instruction": _REPAIR_INSTRUCTION}
    return _mint_prompt(payload)


__all__ = [
    "strategy_repair_prompt",
    "strategy_request_prompt",
    "strategy_sample_design_correction_prompt",
]
