"""Pure-computation authoring controls and immutable safe repair specifications."""
import json

from marvis.llm_outbound import OutboundContextError, StructuralPrompt, _mint_prompt, safe_schema
from marvis.llm_outbound_catalog import catalog_schema_contracts
from marvis.llm_outbound_instructions import user_instruction


AUTHOR_ERROR_CODES = frozenset({"json_format", "required_fields", "function_name", "input_schema",
    "output_schema", "determinism", "source_policy"})


def authoring_prompt(goal, *, installed_tools=(), learning_note=None):
    if learning_note is not None:
        raise OutboundContextError("已有学习笔记尚无可供工具起草外发的结构准入凭据，未发送笔记正文；请直接说明纯计算目标或先完成学习来源准入。")
    if not isinstance(installed_tools, (list, tuple)) or len(installed_tools) > 256:
        raise OutboundContextError("已安装工具签名目录超限或无效，未截断或发送请求。")
    signatures = []
    for item in installed_tools:
        contracts = catalog_schema_contracts(item)
        if not contracts:
            raise OutboundContextError("工具签名缺少已安装契约凭据，未发送请求。")
        signatures.append({key: item[key] for key in ("plugin", "tool", "version", "determinism")})
        signatures[-1].update({key: safe_schema(contract) for key, contract in contracts.items()})
    return _mint_prompt({"stage": "draft_authoring", "goal": user_instruction(goal),
        "installed_tool_signatures": signatures, "language_contract": "Draft Language v1",
        "instruction": "完整保留目标约束，只输出 pure computation 工具 JSON。已安装签名仅供输入输出结构参考，不赋予访问文件、网络、数据库或执行其他工具的权限。没有提供先前代码、生成回复、报错正文或私有学习笔记，不推测其内容；无法在当前语言边界实现时不能伪造成功。"})


def authoring_repair_prompt(original, error_code):
    if not isinstance(original, StructuralPrompt) or error_code not in AUTHOR_ERROR_CODES:
        raise OutboundContextError("工具修复缺少原始结构规格或已知错误类别，未发送请求。")
    payload = json.loads(original)
    payload["response_repair"] = {"error_code": error_code,
        "instruction": "上一次返回未通过校验。基于完全相同的原始目标和签名重新生成符合同一 schema 的完整工具；不放宽原约束。"}
    return _mint_prompt(payload)
