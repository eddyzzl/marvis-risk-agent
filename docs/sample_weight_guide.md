# 样本权重使用指南（leakage-risk / business-rationale）

`sample_weight_col`（建模设置里的"样本权重列"）用于在训练时给不同样本不同的拟合权重。权重列**不会作为特征入模**，但会改变模型的拟合目标——用错权重比不用权重更危险，因为它不会在特征重要性里现形，只会悄悄扭曲 KS/AUC 等评估指标和最终打分。

本指南解释：什么时候应该用样本权重、什么信号说明权重列存在泄漏风险、以及审批/评审时可以引用的业务理由模板。对应追踪项：LT-14。

## 什么时候应该使用样本权重

样本权重的合法用途只有三类，且都应该能独立于标签本身给出理由：

1. **抽样权重（sampling weight）**：训练样本不是简单随机抽样得到的（例如按分层抽样、按时间段不均匀抽样、对稀有分群过采样/欠采样），权重用于把样本分布纠正回总体分布。理由应该能说清楚"抽样设计是什么"，而不是"这个样本表现如何"。
2. **拒绝推断权重（reject inference weight）**：把被拒绝、未放款、或结果未观测到的样本纳入训练时，用权重体现这些样本的可信度/覆盖度调整（例如 parceling、fuzzy augmentation 产生的权重）。理由应该引用拒绝推断方法本身，而不是标签的强弱。
3. **业务策略权重（business weight）**：业务明确要求模型对某类客群/渠道/产品更敏感（例如新客户获取策略要求过采样某个渠道），权重来自业务规则文档，而不是从结果数据里反推出来的系数。

如果说不清楚权重来自上述哪一类、或者权重的取值本身就是"用标签计算出来的某个函数"，就不应该使用样本权重。

## 泄漏风险信号

以下信号出现任意一条，都应该被当作**贷后结果泄漏**对待，而不是简单的"数据质量问题"：

- **权重与标签强相关**：权重列的取值和目标列（好坏标签、逾期天数等）高度相关。这通常说明权重不是独立设定的，而是从结果本身推导出来的——例如直接拿逾期天数、罚息金额做权重，等于把标签信息又喂回了模型的拟合过程一次。
  - 平台侧信号：`choose_modeling_spec` 的建模设置门（modeling setup gate）会计算候选权重列与目标列的样本相关系数，写入 `sample_weight_diagnostics[].leakage_risk`（`"high"` / `"low"`）与 `target_correlation`；相关系数绝对值 ≥ 0.3 时标记为 `"high"`，并在门的 `override_guidance` 里给出中文提示（见下方"平台内置提示"）。这是一个粗粒度信号，不是终审结论——`"low"` 不代表一定安全，`"high"` 也不代表一定要拒绝，只是提示需要人工确认业务理由。
- **权重来自贷后信息**：权重列的取值依赖于放款之后才产生的字段（逾期状态、催收结果、实际还款表现、贷后风险处置动作等）。任何用贷后才存在的信息构造的权重，都会让模型在训练时"看到"未来信息。
- **权重列名称/口径不可追溯**：权重列没有对应的抽样设计文档、拒绝推断方法说明、或业务规则出处，只是数据里凭空多出来的一列数字。
- **权重随时间漂移且与最近的坏账率同向变化**：如果权重是按月/按批次生成的，且其分布随时间的变化趋势和实际坏账率高度一致，通常说明生成权重的过程本身就参考了结果。

## 业务理由模板

在建模设置确认样本权重列时，建议按以下模板补充业务理由（写入任务备注或评审记录）：

```
权重列：<column_name>
权重类型：抽样权重 / 拒绝推断权重 / 业务策略权重（三选一）
权重来源：<抽样设计文档编号 / 拒绝推断方法说明 / 业务规则文档链接>
是否使用贷后字段：否（如为是，必须说明为何在拟合阶段可以使用该信息）
与目标列相关系数：<平台诊断给出的 target_correlation，如无则说明计算方式>
评审结论：批准使用 / 需要更换权重列 / 不使用样本权重
```

没有清晰理由的权重列，应当在建模设置阶段直接改为不使用样本权重（`sample_weight_col` 留空），而不是"先训练看看效果"。

## 平台内置提示（现状）

- `choose_modeling_spec`（`marvis/packs/modeling/feature_tools.py`）在权重列同时出现在候选特征列表时会自动剔除并提示"样本权重列已从入模特征中移除"。
- 建模设置门（`_modeling_override_guidance`，`marvis/agent/gate_payloads.py`）在用户选择了权重列时，始终提示"该权重列会改变拟合目标且不会入模，需要确认来源"；当诊断信号判定为高泄漏风险（权重与目标高度相关）时，会升级为 `level: "warning"` 的提示，并在消息中带出相关系数，要求用户更换权重列或改为不使用样本权重。
- 权重列的有效性诊断（缺失值、非正权重等）在 `_sample_weight_diagnostics`（`marvis/agent/modeling_setup.py`）里计算，`leakage_risk`/`target_correlation` 字段与有效性诊断一起返回，供前端和门文案复用。

## Fixtures（测试覆盖）

以下 fixtures 覆盖了上述判定逻辑，位于：

- `tests/test_modeling_recipes.py::test_build_modeling_proposal_flags_high_leakage_risk_when_weight_correlates_with_target` — 权重与标签强相关时，`sample_weight_diagnostics[].leakage_risk == "high"`。
- `tests/test_modeling_recipes.py::test_build_modeling_proposal_leakage_risk_low_when_weight_independent_of_target` — 权重与标签独立（业务/抽样权重的典型情形）时，`leakage_risk == "low"`。
- `tests/test_plan_driver.py::test_modeling_screen_gate_warns_when_selected_sample_weight_has_high_leakage_risk` — 建模设置门在选中高泄漏风险权重列时，把提示升级为 `level: "warning"` 并带出相关系数。

新增权重相关的判定逻辑或提示文案时，请在这三个测试文件里补充对应 fixture，保持"指南随 fixtures 持续丰富"（LT-14 原始诉求）。
