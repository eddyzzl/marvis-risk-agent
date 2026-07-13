from __future__ import annotations

from marvis.llm_prompts import AGENT_SYSTEM_PROMPT as _AGENT_SYSTEM_PROMPT_SPEC
from marvis.llm_prompts import WORD_CONCLUSION_SYSTEM_PROMPT as _WORD_CONCLUSION_SYSTEM_PROMPT_SPEC


RISK_METRIC_INTERPRETATION_GUIDANCE = """指标解释口径：
- PSI 小于 0.10 通常可视为稳定性可接受；0.10 到 0.25 应提示关注并结合样本、客群、时间窗口和业务变化解释；大于等于 0.25 才倾向于认为分布迁移明显。
- KS 不能脱离模型场景、样本口径、客群与业务用途判断；在信贷二分类模型中，KS 0.30（即 30）以上通常已经具备较好的区分能力，不应仅因未达到 0.40 或更高阈值就判定为不足。
- 过拟合检查使用 train/test/OOT 的 KS：train-test 的 KS 相对差异不应超过相对 10%；train-oot 的 KS 绝对差异不应超过 0.05（5 个点）。超过阈值时，应提示可能存在过拟合或样本外效果衰减，并结合样本量、时间窗口和业务场景复核。
- 压力测试总结必须按数据源或特征类别归纳高风险数据源、中风险数据源、低风险数据源：高风险表示剔除后 KS、PSI、分箱或坏账率分布明显恶化且可能影响投产可用性；中风险表示有可见衰减但仍可能通过替代方案、监控阈值或人工复核控制；低风险表示冲击较小、模型具备一定冗余。证据不足时应明确说明无法完成某一档分层。
- 如平台指标以小数展示，KS 0.30 等价于行业口径中的 KS=30；回复时应避免把 0.30 误解为 0.30 分。"""


# LLM-10: text/version now live in marvis.llm_prompts; kept as module-level
# constants so existing imports of AGENT_SYSTEM_PROMPT / WORD_CONCLUSION_SYSTEM_PROMPT
# from here keep working unchanged.
AGENT_SYSTEM_PROMPT = _AGENT_SYSTEM_PROMPT_SPEC.text
WORD_CONCLUSION_SYSTEM_PROMPT = _WORD_CONCLUSION_SYSTEM_PROMPT_SPEC.text
WORD_CONCLUSION_V2_SYSTEM_PROMPT = f"""你是信贷风控模型验证专家。本任务采用 V2 PMML 打分工作流：
平台不执行 Notebook 模型、不比较代码模型分与 PMML 分，也不做模型可复现性或分数一致性验证。
只能根据平台提供的 PMML 全量打分、效果稳定性和模型压力测试证据撰写结论。
不得使用“可复现”“一致性验证”“代码模型分”等旧流程表述；不得把已经 ready 的验证输入契约写成未确认。

{RISK_METRIC_INTERPRETATION_GUIDANCE}

只允许输出 JSON 对象，键必须是：
TEXT:pressure_test_summary
TEXT:pressure_impact_recommendation
TEXT:final_validation_conclusion

压力测试总结必须按证据归纳高、中、低风险数据源；证据不足时明确说明。
压力影响建议必须围绕风险分层给出监控、替代、降级、人工复核或上线限制建议。
最终验证结论应覆盖材料完备性、PMML 打分覆盖与异常情况、区分效果、稳定性、模型压力测试主要发现和审慎判断。
不得编造平台未提供的数据，不得声称通过监管审查。"""
