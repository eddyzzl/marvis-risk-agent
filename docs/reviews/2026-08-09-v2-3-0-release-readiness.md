# V2.3.0 发布就绪复审

> 日期：2026-08-09
> 结论：本地候选发布门 `PASS`；commit、tag、release push 与 GitHub CI 在本文写入时仍为 `NOT_PROVEN`。

## 复审范围

本轮复审覆盖 V2.x 八个正式桌面入口、批量模型验证、真实 LLM 语义路由、
Planner/Validator/Eval、策略 Candidate Lab、认证证据投影、发布清单、包数据、
前端样式与 CI 配置。验收按实现、测试、API/Agent、真实浏览器、真实模型评测和
发布流程分层，不以代码形状或单一测试替代最终用户体验。

## 最终 verdict

| 门 | 结果 | 证据 |
|---|---|---|
| 最终集成 code review | PASS | 只读复审未发现可复现 P0/P1/P2；semantic intent、批量 ingress、Candidate Lab 认证边界、Planner/Eval、默认收起状态栏与发布清单均复核 |
| LLM / Planner / Eval 专项复审 | PASS | 未发现剩余 P0-P3；317 个聚焦测试通过，Ruff 与 diff check 通过 |
| 受影响集成回归 | PASS | 378 passed、1 skipped；跳过项仅为需要显式真实 DeepSeek 工作区的网络诊断 |
| 完整本地 release gate | PASS | `MARVIS_RUN_PLAYWRIGHT_SMOKE=1 PYTHON=/opt/miniconda3/envs/py_313/bin/python scripts/check` 退出码 0；12135 passed、8 skipped、32 warnings，耗时 1:51:34；Ruff、Node 语法与 diff check 同步通过 |
| 真实浏览器验收 | PASS | 真实 in-app browser 完成八入口、批量验证、主要分支、typed gate、下载、刷新、负向阻断与窄容器响应式验收 |
| 真实 LLM 安全评测 | PASS（单一推荐层） | `marvis.eval.report.v2` 为 COMPLETE，0 LLM/harness error；仅 `autonomous` 达到 13/13、guardrail 3/3、两项通过率 1.00 并可推荐 |
| Commit / release push | NOT_PROVEN | 必须在最终 manifest 审核、完整暂存并形成业务 commit 后由 `scripts/release_push.py` 完成 |
| GitHub CI | NOT_PROVEN | 必须核对发布 SHA 的 push CI，并额外手工触发包含完整 `scripts/check` 的 workflow_dispatch CI |

## 用户旅程证据

真实浏览器记录保留在候选外的本地发布证据目录
`workspace/release_evidence/20260808-v2.3.0-candidate/`：

- `UI-ACCEPTANCE.md`：逐入口任务 ID、Agent 话术、分支、可见结果与负向阻断。
- `screenshots/`：入口完成、语义路由、错误门、默认收起状态栏、Feature Binning、
  Strategy Candidate Lab 响应式布局与性能复核。
- `downloads/`：JOIN Parquet、模型/验证/风险/Portfolio 报告、批量单模型报告与汇总
  Excel、策略最终 JSON/Markdown/XLSX/DOCX。
- `app-workspace/`：本轮任务、消息、计划、证据与结果制品，可由平台和后台继续核查。
- `logs/final-full-scripts-check-round3.log`：最终完整本地门禁日志。

已走通的关键分支包括：Feature 全指标/选定分箱/跳过分箱；LR 无 OOT、XGB 随机
OOT、LGB 时间 OOT；模型验证手动/Agent/批量/无效材料阻断；Standard Vintage、
VTG、base-only VTG、利润分析和不兼容混合场景；Portfolio 完整 no-trend；策略
四种单变量分箱、Cross Matrix/阈值/自动搜索、自动树/剪枝/前沿、Scorecard、Pool、
稳定性、Validation/OOT 重放、编译/应用、ProjectContext 与四格式最终报告。

## 真实 LLM 评测

报告：
`workspace/release_evidence/20260808-v2.3.0-candidate/app-workspace/eval/model-452c506756-20260809T033952254625Z-391b8fbb758e.json`

SHA-256：
`96ea196acd4ae071fdb37072b2b1aaefa98b73e3c3783f14e4c5a028186af549`

| Tier | Pass rate | Guardrail pass rate | Guardrail intact | 可推荐 |
|---|---:|---:|---|---|
| autonomous | 1.00 | 1.00 | 是 | 是 |
| balanced | 0.7692 | 0.3333 | 否 | 否 |
| conservative | 0.9231 | 0.6667 | 否 | 否 |

因此，本轮可以证明 `autonomous` 在当前 13-case corpus、prompt snapshot、模型配置和
治理阈值下达到推荐门；不能声称三个 tier 都通过，也不能把该推荐自动解释为部署、
审批或生产授权。

## 本轮关闭的高风险问题

- 语义路由由两次独立 LLM 判定和逐字证据约束负责，失败关闭；不再依赖“确认”等
  关键词命中来决定下一步。
- C1 目标、忽略文件、幂等确认、策略嵌套样本绑定和失败消息原子性保持一致。
- Planner 目录按任务受限检索，约束 required/forbidden/literal/granted tool，
  replan 与 Explore 同样 fail-closed；上下文按模型预算收敛。
- Eval 可区分 harness、planning 与 typed LLM 错误，推荐门要求零错误、最低通过率、
  guardrail 完整及关键案例通过。
- Candidate Lab 请求内复用已认证证据并一次索引 ProjectContext；每个新请求仍重新
  认证，无跨请求 TTL。真实任务切换由约 29.8 秒降至 10.687 秒。
- Feature Binning 卡片、七阶段 spine、参数 grid、报告下载换行与 Candidate Lab
  容器响应式布局完成修复；顶部状态栏默认收起且 ARIA/inert 状态一致。
- Playwright 被纳入锁定 dev 依赖，CI smoke fixture 与生产 task scope 对齐。

## 发布清单边界

最终提交必须纳入两个新增实现模块及七个对应回归文件；漏掉任何实现模块都会使
远端导入或测试收集失败：

- `marvis/agent/semantic_intent.py`
- `marvis/validation_batch_ingress.py`
- `tests/test_semantic_authorization_live_profile.py`
- `tests/test_semantic_intent_live_profile.py`
- `tests/test_semantic_intent_routing.py`
- `tests/test_semantic_join_intent.py`
- `tests/test_specialized_workflow_semantic_delegation.py`
- `tests/test_strategy_sample_binding_semantic_delegation.py`
- `tests/test_validation_batch_ingress.py`

本地 PASS 不关闭以下边界：T4-2 公开数据参考门、真实机构材料对账与责任签字、
原生 Windows/NTFS、真实身份源与 maker-checker、生产执行器/实时决策集成、长期调度
告警、故障演练和运营签字。这些仍按能力矩阵保持 `NOT_PROVEN`、`BLOCKED` 或
`NOT_DELIVERED`。
