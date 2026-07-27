# E2E 数据处理 / 特征分析 / 模型开发完整 Code Review

- 审查日期：2026-07-28
- 固定基线：`51e8a4f44d0205ab53573d557f2d502ece32d700`
- 审查对象：`codex/e2e-data-feature-model`
- 规模：约 240 个文件、36939 行新增、2689 行删除，另含本轮新增的共享 XLSX
  安全边界与本审查文档
- 结论：**PASS_FOR_PR**

两名独立审查者分别按 Standards 和 Spec 两个轴审查；两个轴的严重级别与结论
保持独立，不合并重排。独立审查和随后的 PR CI 共发现 1 个 Blocker、8 个 High
和 5 个 Medium，均已修复并有定点回归。当前没有遗留 Critical、Blocker、High
或 Medium 代码问题。真实材料业务验收仍有外部前置条件，不能由本代码审查代签。

## Standards 轴（独立审查）

### High S1：结果数据集下载可跨任务引用

旧的 `/api/datasets/{dataset_id}/download` 只凭 dataset id 下载，没有 task
ownership，也没有下载前内容漂移校验。

修复后只保留
`/api/tasks/{task_id}/datasets/{dataset_id}/download`：任务和数据集必须同属一个
task，跨任务返回 404，注册文件漂移返回 409。Agent 消息、历史结果恢复和前端
下载链接全部改为 task-scoped；前端只在当前任务明确时兼容重写旧消息链接。

### High S2：Feature / Modeling Tool 可读取其他任务的引用

多项 Feature Tool 原先直接按 dataset id 读取；建模报告、交付、监控和 Champion
引用也存在只校验 id、不校验 task 的入口。

修复后 Feature 的主数据集、对比数据集和变换输入统一校验 `ctx.task_id`；Modeling
新增 task-scoped dataset、experiment、artifact 解析边界，并覆盖训练、调参、
选择、校准、报告、监控和交付。显式 Champion experiment/artifact 必须互相绑定，
交付样本和 Challenger 样本也必须属于当前任务。

### High S3：LGB 回归/多分类 early stopping 使用正式 test 集

回归和多分类配方原先把正式 test 集作为 early-stopping validation，导致训练轮数
间接利用最终测试集。

修复后 early-stopping fold 只从 train 内部切出；fit 与 valid 的并集等于原 train，
valid 与正式 test 完全不相交。正式 test/OOT 只用于最终指标。

## Spec 轴（独立审查）

### Blocker P1：共享 XLSX 安全模块未进入提交

Feature/Model 报告已经引用 `marvis/output/xlsx_safety.py`，但文件曾处于 untracked
状态，提交后会直接导入失败。该模块现已纳入本次提交清单，并统一处理：

- `= + - @` 公式前缀；
- XML 非法字符；
- Excel 32767 字符单元格上限；
- 数值、布尔和空值的原类型保留。

### High P2：T4-3 把 A2/A4/A5/B5 固定写成 N/A

旧验收脚本允许缺少 Vintage 语义、长 ID/空白键诊断和 JOIN 匹配率对账时继续，
与真实材料验收标准冲突。

修复后：

- A2 必须提供已完成 Vintage 任务、明确 `incremental`/`snapshot` 并生成非空曲线；
- A4/A5 必须有已完成 JOIN 的精度、dtype、键和执行证据；
- B5-INTERNAL 必须有有限匹配率、独立双路对账、无 fan-out 且保持 anchor 行数；
- B5-EXTERNAL 仍保留人工抽样签字，平台不伪造；
- 缺少 companion JOIN/Vintage 任务时机器判定为 FAIL，而不是 N/A。

CLI 和 checklist 已增加 `--join-task-id`、`--vintage-task-id`。

### High P3：没有证明自然语言 Agent 能完成全链路

原 smoke 的 JOIN、Feature、Modeling 主要是手动 gate；多文件建模只跑到 split，
不足以支持“用户用自然语言让 Agent 完成全部工作”的产品结论。

新增 Agent 模式 E2E 使用自然语言确认和开始指令，在全新工作区完整执行：

`多文件识别 → JOIN → split → screen → 特殊值治理 → 精选特征 → 调参 → 训练
→ 选模 → 报告 → PMML/验证移交/Challenger 交付`

测试使用本地非网络 gate client，验证受治理的文本到动作链路；它不把外部大模型
生成质量冒充为确定性指标验收。

### High P4：最终审查结论与保存的机器报告互相矛盾

旧文档称实现完成且只剩外部输入，但保存的机器报告自身是 `BLOCKED_MACHINE`，
同时记录 A1 失败和监控 fail。

相关文档现已改为 `CONDITIONAL`，历史机器报告明确标记为失效失败快照。只有修复
真实任务、补齐同材料 JOIN/Vintage 证据并重新生成报告后，才能进入 B1-B5 外部
签字；不得引用历史报告宣称结项。

## PR CI 补充发现

### High C1：未配置 metrics 的策略任务无法创建新版本

DFM 把任务的指标契约升级为三态：`None` 表示未配置、空列表表示用户明确不选、
非空列表表示已选择。策略监控的红灯交接仍按旧契约执行
`list(source_task.metrics)`，因此自然语言“起新版本”在未配置指标时会以
`NoneType is not iterable` 失败。

修复后新版本交接显式保留三态，不把 `None` 偷换为空列表。新增仓储层快速回归
先在旧实现上稳定复现，再在修复后通过；PR 中原失败的完整监控 E2E 和全部策略
product smoke 均已通过。

### Medium C2：Candidate Lab 列表查询遗漏策略描述

策略元数据行解析新增 `description` 后，最近策略和本地 Champion 两条分页查询
仍使用旧字段列表。Candidate Lab 加载这些结果时会因缺列触发 `IndexError`，
导致候选策略实验台无法打开。

修复后两条查询都显式选择 `description`，保持分页总数、任务隔离和 Champion
筛选语义不变；Candidate Lab API 全文件回归通过。

### Medium C3：成功恢复计划后可选运行时能力导致响应失败

Driver turn 已成功恢复计划后，消息追加阶段直接读取 `llm_client` 和
`hook_dispatcher`；精简运行时适配器没有这些可选能力时，会把成功执行改写成
错误响应。内部 `_run_driver_turn` 同时把公共包装器原本可省略的 `ui_action`
变成必填，破坏了非策略流程的旧调用契约。

修复后消息阶段通过缺省 `None` 使用可选能力，`ui_action` 恢复为可省略参数；
计划状态和确定性执行结果不受可选 Agent 集成影响。

### High C4：数据预览测试错误要求回传原始个人信息

两条数据预览测试曾把手机号、姓名和长卡号的脱敏契约改成“原样返回”，与生产
实现和本地优先的数据治理边界相反。虽然运行时代码仍保持掩码/不可逆 token，
错误测试会阻止隐私保护实现通过门禁，并为未来回归提供错误规范。

修复后测试重新要求手机号掩码、姓名 token 化和长标识有界掩码，同时断言完整
原值不会出现在响应 JSON。

### Medium C5：训练缓存测试桩没有任务所有权

Modeling Tool 已增加 task-scoped dataset 校验，但三个训练/调参缓存测试的伪
registry 仍只返回 dataset id，缺少生产记录必有的 `task_id`。修复后测试桩显式
携带所属任务，继续验证缓存、分组列投影和逐配方内存释放，同时不绕过跨任务边界。

### Medium C6：报告下载前端测试仍要求右栏旧交互

宽屏信息架构已把下载和定位动作收敛到中间工作区，右侧 Plan Rail 只保留
checker、步骤文案和运行进度；旧测试仍要求右栏 `plan-step-ready` 和定位入口。
修复后契约明确断言右栏不含下载/定位控件，并保留状态 checker 与调参进度。

### Medium C7：v15 迁移测试使用当前仓储写入旧表

候选池 v15→v16 迁移测试先构造真实旧库，却调用当前 TaskRepository 写入。新增
`metrics_configured` 后当前 writer 合法地要求新列，测试因此在迁移开始前失败。
修复后 fixture 按 v15 自身 SQL 契约播种任务，再由当前 reader 验证旧行兼容并
执行真实升级；没有给生产 writer 增加伪旧库分支。

## 实现者复审补充修复

- 非二分类 screen 增加 NaN 标签确认、丢弃计数和有界列批次读取。
- 调参拒绝 `cv_folds` 大于可用分组数，避免空 fold 或伪交叉验证。
- PlanExecutor 与 ToolRunner 在工具已成功提交后以完成协议为准，迟到的取消信号
  不再把成功结果改写为 cancelled。
- `train_models` 全部命中缓存时不再重读宽表数据。
- 模型报告按 task/experiment/dataset 隔离，多版本同配方报告文件名不互相覆盖。
- 用户在 screen gate 删除的特征不会被已完成的特殊值 no-op 步骤重新引入。
- 数据预览恢复脱敏显示；长标识、手机号、证件号和类别值不直接回传原文。
- 恢复宽屏 Plan Rail 的预期状态展示，同时删除已废弃或禁止的控制入口。
- Prompt 内容变更同步升级版本与锁定哈希；旧策略 schema/version 断言同步到当前
  数据库版本，不通过放宽生产校验换取测试通过。

## 验证证据

- fresh-workspace closure smoke：`3/3 PASS`
  - Data JOIN：1 passed，9.37 秒；
  - Feature Analysis：1 passed，9.12 秒；
  - Agent 多文件完整建模：1 passed，32.79 秒。
- 本轮关键影响面组合：`121 passed, 46 deselected`，覆盖 Feature、数据预览/
  下载、CV、LGB early stopping、缓存、取消边界、XLSX 和跨任务隔离。
- 跨任务 Modeling Tool 组：`13 passed`。
- closure + Agent 完整建模：`64 passed`。
- closure + Vintage + JOIN 定点组：`66 passed`。
- 策略 PR smoke：`5 passed`，包含自然语言候选策略、标准策略工作流，以及
  `红灯 → 起新版本 → 报告` 完整链路。
- Candidate Lab API 与计划迁移定点组：`95 passed`。
- 第二轮 PR CI 根因定点组：`7 passed`，覆盖训练任务隔离、预览脱敏、宽屏报告
  入口和 v15→v16 迁移。
- 上述第二轮受影响文件完整回归：`83 passed`。
- 先前建模受影响组：`341 passed`；Prompt/策略编译契约组：`1007 passed`。
- 全仓 Ruff、全部前端 JavaScript 语法、`git diff --check`：通过。
- Bandit high-severity：无高危发现。

PR fast 层当前收集 10018 个用例；四片分别为 2522、2460、2547、2489，合计
无遗漏。鉴于单 runner 已在风险分析 PR 上超过 100 分钟被取消，DFM 不再重复
串行跑一轮本地全仓 fast；PR 使用 4 个稳定哈希分片并行执行全部 fast 用例，
汇总门禁要求四片全绿，不删除覆盖。slow/E2E 由上面的定点闭环和最终
main/release gate 覆盖。

## 仍需真实业务材料完成的事项

以下不是可以由代码或 Agent 编造的指标：

1. 至少一份真实材料的最新机器预检必须通过 A1-A6、B4-INTERNAL、
   B5-INTERNAL、C/D；
2. B1-B5 的外部财务/风险口径、独立复算、真实键抽样及责任人签字；
3. 两个公开信用数据集的 KS 基线材料。

这些事项限制“真实业务验收已完成”的表述，但不构成当前 PR 的代码缺失。PR 只有
在并行 CI 全绿后才允许合并。
