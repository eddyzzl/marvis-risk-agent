# V2 策略开发全量 Code Review（2026-07-28）

## 结论

本轮审查覆盖 `origin/main@a498f3ca` 至
`codex/v2-strategy-foundation` 的初始审查头 `55da78bb` 所包含的完整策略开发
增量，以及审查中追加的修复。策略核心七步、自然语言 Agent、Manual Workbench、证据治理、候选开发、
Strategy Pool、影响测算、代码交付和四格式七步报告的实现边界一致；没有把已确定
能力推迟到 V3/V4。

审查发现的生产缺陷均已修复。最终发布结论以本文件末尾列出的验证门禁全部通过为准。

## 审查范围

- 129 个增量提交，631 个变更文件；其中策略请求编译、Agent turn、Workflow
  模板、Plugin manifest、确定性 Tool、仓储/迁移、报告渲染、Candidate Lab
  前后端和测试为重点。
- 核对 `docs/roadmap.md`、`DESIGN.md`、Notebook 契约及 V2 策略改造计划，
  检查产品承诺、治理边界与实现是否一致。
- 检查 task ownership、dataset/workspace/hash 绑定、writer-lock 二次认证、
  artifact 原子发布、幂等身份、人工采纳门和审计链。
- 检查 Python/JavaScript 语法与 lint、JSON/YAML manifest、依赖漏洞和高危
  静态安全问题。
- 对策略 fast 测试全集分三片执行，再对修复涉及的文件组和被 fast 排除的
  slow/e2e 层单独回归，避免每个小修复重复全仓回归。

## 已修复问题

### High：记忆审计失败后仍可能继续自动执行

`agent_autodrive_turn` 原先吞掉记忆使用审计异常，后续仍可能确认当前治理门。
现在改为 fail closed：写入结构化 `memory_use_audit_failed` 治理阻断消息并停止
自动执行；新增审计失败注入测试，确保没有无审计决策。

### High：缺少成熟样本时先发送了误导性的“开始执行”

完整策略、快速策略和规则策略入口原先在解析受治理样本引用前发送开始消息；当
样本缺失时，用户会先看到已开始、随后才收到通用失败。现在先认证精确
StrategySampleDesign，再发送开始消息；缺失时返回
`strategy_sample_design_required`，并完整列出 V2 双总体样本口径字段。

### Medium：批准后 DSL 漂移泄漏内部异常

受治理策略采纳在 canonical DSL/hash 被篡改或漂移时抛出内部 `ValueError`，
没有稳定落到治理冲突边界。现在将该完整性失败转换为 `ConflictError`，明确说明
批准后的 strategy spec 已变化，且不产生采纳、副作用或审计写入。

### Medium：评分卡 Pool 来源没有服从统一截断上限

Candidate Lab 的 Pool 来源窗口对评分卡使用独立上限，导致统一上限收紧时仍返回
更多评分卡来源。现在取统一 Pool 上限与评分卡专项上限的较小值，同时保留真实
`total` 和 `truncated` 语义。

### Medium：Voting 搜索组合被展示层按成员顺序误拒绝

Voting 搜索组合成员是无序集合，而候选构建器会按 Pool 位置规范化
`selected_entries`；展示层原先要求规范化后的列表与搜索来源列表顺序完全一致，
导致合法候选被误报为完整性校验失败。现在验证成员集合一一对应且 Pool 位置、
entry id、rule id 均无重复，同时允许顺序规范化；真实自然语言构建和恶意重复成员
均有回归覆盖。

### Low：新契约落地后仍有旧测试断言

同步修正以下已落地契约的旧断言：

- fresh 自然语言样本设计必须生成 V2 双总体证据，不再新建 legacy-only 请求；
- Evidence Drawer 是唯一允许投影认证 hash 的区域；
- 交互树 frontier Tool 同时可读 v1/v2 输出；
- 报告源新增 Cross Rule Search，当前 schema 已升级到 22；
- 交互树搜索/续建 Tool、手工切点、原子 artifact API 的测试使用当前接口。
- 直接启动旧策略模板的 slow E2E 先物化当前受治理的成熟样本设计凭据。

### Low：静态质量债

补齐成对迭代的显式 `zip(..., strict=...)`、异常链、循环闭包绑定和未使用变量，
移除重复集合项及无效循环；不改变确定性计算或公开契约。

## 验证证据

- 策略 fast 全集首次三片：`4793 passed / 17 failed / 106 deselected`；
  17 个失败全部完成根因分类和修复，随后定点复测 `17 passed`。
- 审查修复的 10 个高风险文件组：`162 passed`。
- 策略 slow/e2e/llm/pmml_runtime 层分片收敛：`106 passed`。
- Ruff、全部前端 JavaScript 语法、`git diff --check`：通过。
- `pip-audit`：锁定依赖未发现已知漏洞。
- Bandit high-severity：未发现高危问题。
- JSON/YAML manifests：解析通过。

## 保留边界

- 策略采纳仍是人工责任门；本地采纳不等于生产部署。
- LLM 只负责理解、澄清、规划和解释；指标、样本 membership、规则执行、影响
  测算和报告数值由确定性平台代码计算。
- SQL/Hive/Impala 数据连接、跨设备多用户同步、交互 revision 独立图形包和更
  完整的 browser 自动化属于 V2.x 平台增强，不改变本地文件驱动的策略核心七步
  已完成结论，也不得移入 V3/V4。
