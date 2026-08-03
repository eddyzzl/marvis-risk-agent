---
status: clean
reviewed_at: 2026-08-03T05:34:11Z
base_sha: 9845e907c702f65edc77a9d0121f0967f52488c2
staged_file_count: 130
staged_diff_sha256: 2a10220fe8502b79a1f005ef93ac06651c4b35aaca5ae523a5437518dce2edcf
followup_base_sha: 58ce9ec6de52f7df835daec7fe72cdbbf42eceb6
followup_staged_file_count: 7
followup_staged_diff_sha256: 3ef1ede5259eab91739c23eafcd717d27342869b40b64a3fd931c957b3a53d2e
review_depth: standard_domain_split_plus_ci_failure_root_cause_and_two_axis_re_review
findings:
  blocker: 0
  warning: 0
  info: 0
  total: 0
---

# MARVIS 六大功能 UI 验收改动：最终 Code Review

## 结论

**CLEAN — 可以进入完整发布验证。**

本报告绑定基线
`9845e907c702f65edc77a9d0121f0967f52488c2` 与 130 个代码/测试文件组成的
精确 staged 候选；二进制 diff SHA-256 为
`2a10220fe8502b79a1f005ef93ac06651c4b35aaca5ae523a5437518dce2edcf`。

Agent/治理/编排、Feature/Modeling/Strategy workflow、前端三个领域分别审查，
所有已证实问题均已修复并复审关闭。最后一轮独立复审重新计算并匹配上述
base、文件数与 diff 哈希，未发现 blocker 或 warning。

首个干净 release clone 的完整 CI 随后暴露一个跨 ToolRunner 子进程的产物暂存
竞态。修复增量相对提交 `58ce9ec6` 包含 7 个文件，staged diff SHA-256 为
`3ef1ede5259eab91739c23eafcd717d27342869b40b64a3fd931c957b3a53d2e`；
Standards 与 Spec 两个独立审查轴均复审为 **CLEAN**，0 blocker、0 warning。

## 已关闭的审查问题

1. **强制人工/效果门禁可被自由文本 replan 删除 — 已关闭。**
   最终写边界拒绝删除或弱化 mandatory gate；冲突发生在任何计划、loop event
   或审计写入之前。正常保留门禁的 replan 仍可执行。
2. **多分类数值 `+/-inf` 被编码成合法类别 — 已关闭。**
   数值无穷值在 factorize 前被 mask，并复用既有非有限标签确认门；合法字符串
   类别仍保留。
3. **首轮自然语言调参预算被静默丢弃 — 已关闭。**
   `n_trials` 已成为 proposal/template slot，单表 `modeling` 和多表
   `modeling_with_join` 均把用户预算传到规格选择。
4. **完整目标列中的稀有第三类可能被 4,000 行抽样遗漏 — 已关闭。**
   目标类型使用完整目标列精确判断；不能确定时 fail closed。
5. **未恢复的历史 workflow 失败在新计划中消失 — 已关闭。**
   历史错误保留为默认折叠、只读的审计卡，不抢占当前 gate 交互。
6. **后续调参入口可绕过 200 轮上限 — 已关闭。**
   共享约束统一拒绝 bool、float、string、0、201，并覆盖首轮 setup、两种模板、
   gate adjustment、choose/configure/tune 的 scalar 与 per-recipe 输入。
   三个 manifest 输入 schema 均声明 `minimum: 1`、`maximum: 200`；输出
   schema 保留非调参算法的 0 与多算法合计可超过 200 的兼容语义。
7. **并发导出偶发删除共享 `.staging` — 已关闭。**
   失败由完整 CI 的真实 ToolRunner 双子进程用例捕获：一方删除自己的暂存
   占位后，winner 清理空目录，loser 写入时报 `FileNotFoundError`。现在文件
   stage 在跨进程 `mkdir -> touch` 窗口遇到父目录消失时有界重试；数据导出、
   数据转换和策略自动树都保留占位直至同盘 `Path.replace()` 原子覆盖，并把
   覆盖放进 UOW rollback 边界。原先对子进程无效的 monkeypatch 用例也已拆成
   真实 ToolRunner 并发测试与直接 reservation contract 测试。

## 最终候选验证

- 边界与修复定向测试：**46 passed in 14.00s**
- 扩大受影响集（建模 API/pack/select、模板、Plan Driver、Agent API、
  Plugin Runner）：**548 passed, 4 warnings in 756.94s**
- warnings：4 条 LightGBM `eval_set` 上游弃用提示；无项目测试失败
- 全量 Ruff：通过
- Node 语法检查：`app.js`、`driver_manual_analysis.js`、
  `plan_rail_controller.js` 通过
- Modeling/Strategy manifest JSON：通过
- `git diff --cached --check`：通过
- 并发暂存修复相关集合：**59 passed in 70.52s**
- 修复后重复并发压测：数据导出 **30/30**、数据转换 **30/30**、策略自动树
  **10/10**；winner Parquet 可读，dataset/evidence/result 哈希一致
- 并发修复增量复审：Standards **CLEAN**；Spec **CLEAN**

完整、无 fast/affected 限制的 `scripts/check` 按发布流程在本报告提交后的
干净 release clone 上从头执行；本报告不提前声称该后续 gate 已通过。此前
中止的运行仅证明了 `1 failed, 1750 passed` 时捕获上述竞态，不能作为最终通过。

## UI / 运行时证据范围

真实可见界面验收覆盖数据 JOIN、特征分析/分箱、模型开发、模型验证、
Vintage/风险分析、策略六大流程，并覆盖 LR/XGB/LightGBM 等单独与多算法建模、
二分类/回归/多分类、时间 OOT/随机 OOT/无 OOT、多种筛变量与策略方法、
Manual/Agent 分支、确认/调参/重规划/报告下载等交互。

运行时 task、plan、报告、截图与操作记录保存在
`workspace/e2e_ui_20260729/`；代表性语义意图任务为
`f4ab1e9a2bad4da286e08a0809915c53`，Feature Binning UI 修复任务为
`67c777...`。

## 审查边界

- 本报告不包含与本任务无关的 `docs/deploy-linux-env-checklist.md`。
- 原始分域审查记录保留在隔离候选工作区；发布提交只纳入本统一报告。
- 确定性指标与 pass/fail 来自平台代码及测试，不来自 reviewer 文本。
