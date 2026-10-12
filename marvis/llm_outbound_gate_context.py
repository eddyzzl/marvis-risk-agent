"""Installed authorization purposes; UI prose never becomes an outbound contract."""
from dataclasses import dataclass
import hashlib
import json

from marvis.llm_outbound import OutboundContextError, structural_summary

_TOKEN = object()
_CONTRACTS = {
    "portfolio_order": "当前节点只授权接受界面已展示的完整逾期桶顺序。用户无需重述顺序；重排或改变口径必须澄清。",
    "labeling_proposal": "当前节点只授权按界面已展示的完整标签提案和成熟度决定创建计划。用户无需重述提案；不得从授权句修改字段。",
    "role_assignment": "当前节点只授权采用平台已绑定的文件角色和目标列方案。用户无需重述已展示的方案；不得猜测或更换材料。",
    "modeling_setup": "当前节点是在创建计划前收集建模规格，只抽取已声明控件；不新增、删除或重排步骤。",
    "create_workflow_plan": "用户已经选择当前工作流。此处只授权创建可审阅计划，尚不执行计划。已声明的参数可被抽取；问题、条件、拒绝或切换工作流必须澄清。",
    "execute_plan": "当前唯一动作是立即执行当前已验证计划。简短而明确的接受当前方案可构成授权；不得猜测新参数或改变计划。",
    "current_step": "当前唯一动作是执行已展示的当前确认节点，参数只能来自控件与 proposed_params。仅保留已有风险注记不改变动作。",
    "validation_stage": "当前唯一动作是立即执行平台指定的模型验证阶段；不是修改指标、材料或报告口径。",
}


@dataclass(frozen=True)
class ApprovedGateContext:
    name: str
    binding_json: str
    _token: object

    def payload(self):
        if self._token is not _TOKEN or self.name not in _CONTRACTS:
            raise OutboundContextError("授权节点来源无效，未发送请求。")
        return {"purpose": self.name, "application_contract": _CONTRACTS[self.name],
                "binding": json.loads(self.binding_json)}


def authorization_gate(name, *, bindings=None):
    if name not in _CONTRACTS:
        raise OutboundContextError("授权节点未声明，未发送请求。")
    raw = bindings or {}
    if name == "create_workflow_plan":
        family = raw.get("workflow_family") if isinstance(raw, dict) else None
        if not isinstance(family, str) or family not in {"modeling", "feature_analysis"}:
            raise OutboundContextError("创建计划的工作流类型未安装或未明确，未发送请求。")
    encoded = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    projected = structural_summary(raw)
    if name == "create_workflow_plan":
        # This constructor knows the selected installed workflow family.
        # Generic summaries must not bless arbitrary caller labels or prose.
        projected["workflow_family"] = family
    projected["complete_binding_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    return ApprovedGateContext(name, json.dumps(projected, ensure_ascii=False, sort_keys=True), _TOKEN)
