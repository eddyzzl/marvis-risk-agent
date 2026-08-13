# MARVIS 历史文档归档（只读）

> 2026-08-13 文档清理的整合产物。本文件取代此前散落在 `docs/plans/`、
> `docs/plans/specs/`、`docs/releases/`、`docs/reviews/`、`docs/superpowers/`
> 与仓库根目录的历史计划、审查与裁决文档：每个被移除的文档在下方保留一行
> 摘要与处置结论，不再单独维护原文。
>
> 从现在开始：
> - **开发执行**以 [docs/superpowers/plans/2026-08-13-next-90-days-development-plan.md](superpowers/plans/2026-08-13-next-90-days-development-plan.md) 为准；
> - **产品范围与术语**以 [docs/roadmap.md](roadmap.md) 为准；
> - **分层验收状态**以 [docs/capability-status.md](capability-status.md) 为准；
> - **发布规则**以 [docs/versioning.md](versioning.md) 为准。
>
> 所有被移除文档的完整原文仍在 git 历史中，需要时可
> `git log --all -- <path>` 找回。

## 1. 处置口径

| 处置 | 含义 |
|---|---|
| 已交付 | 文档规划的工程已落地并进入 V2.1–V2.3 发布线，原文作废 |
| 被取代 | 后续计划/审查/文档已覆盖其全部内容 |
| 已失效 | 文档中的决定已被后续范围变更推翻（如 LT-9 / LT-18） |
| 保留 | 仍处于生效状态（见 §6 清单） |

## 2. 历史计划与 Spec（已移除）

### 2.1 `docs/plans/`（计划主轴与阶段 spec）

| 文件 | 主题 | 处置 | 现况承接处 |
|---|---|---|---|
| modeling-agent-roadmap.md | 模型开发 Agent 对话式全流程打磨路线 | 已交付 | capability-status「模型开发」行 |
| settings-ia-refactor.md | 设置弹窗信息架构重构 | 已交付 | 当前前端实现 |
| v2-completion-plan.md | V2 整体架构与路线主轴（12 条决策，JOIN 先行） | 已交付 | roadmap / capability-status |
| v2-comprehensive-improvement-plan.md | 2026-06-29 实施轮的验证/证据记录 | 被取代 | 其追踪职能已移交 v2-master-backlog（同已归档） |
| v2-feature-phase-spec.md | FEATURE 阶段详细 spec（特征分析+筛选） | 已交付 | capability-status「特征分析」行 |
| v2-frontend-layout-spec.md | 三区前端布局 spec | 已交付 | DESIGN.md + 当前前端实现 |
| v2-join-phase-spec.md | JOIN 阶段详细 spec（锚定左连接等安全不变量） | 已交付 | capability-status「数据处理 / JOIN」行 |
| v2-longtail-adjudications.md | LT-9/10/12/16/17/18 裁决 | 部分已失效 | 裁决要点保留在本文件 §4 |
| v2-master-backlog.md | 历史实施追踪（115 条审查项汇总） | 被取代 | roadmap + capability-status 成为唯一口径 |
| v2-modeling-phase-spec.md | MODELING 阶段详细 spec | 已交付 | capability-status「模型开发」行 |
| v2-plan-driver-spec.md | 通用计划对话驱动 spec（agent 层） | 已交付 | marvis/agent/plan_driver.py |
| v2-strategy-risk-analysis-plan.md | 策略+风险分析能力地图（S1a–S6） | 已交付 | 策略七步 2026-07-27 收口；roadmap §当前状态 |
| v2-trust-first-plan.md | Trust-First 计划（三层数据判据 T4） | 被取代 | T4 门由 capability-status「当前阻断门」承接 |

### 2.2 `docs/plans/specs/`（函数级 spec）

| 文件 | 主题 | 处置 |
|---|---|---|
| v2-s1a-score-direction-spec.md | 分数方向制度化 | 已交付 |
| v2-s2-strategy-development-spec.md | 策略开发主线 | 已交付 |
| v2-s3-portfolio-analysis-spec.md | 组合分析套件 | 已交付（no-trend 纵切） |
| v2-s4-rule-strategy-spec.md | 规则策略 | 已交付 |
| v2-s5-monitoring-closure-spec.md | 监控闭环与定期报告 | 部分交付（单次诊断已交付；调度仍 PARTIAL，见新计划 B-15） |
| v2-s6-adhoc-pricing-compare-spec.md | 即席分析+额度/定价+challenger 呈现 | 部分交付（即席分析已交付；额度/定价 typed 口径在新计划 B-10） |
| v2-t1-semantic-correctness-spec.md | 语义正确性修复包 | 已交付 |

### 2.3 `docs/superpowers/`（蓝图、阶段设计与后续 plan/spec）

| 文件 | 主题 | 处置 |
|---|---|---|
| plans/2026-06-04-v1-1-agent-memory-foundation.md | V1.1 Agent 记忆基础计划 | 已交付（记忆作为 V2 兼容能力保留） |
| specs/2026-06-04-v1-1-agent-memory-foundation-design.md | V1.1 Agent 记忆设计 | 已交付 |
| specs/2026-06-13-marvis-platform-blueprint.md | V2 平台蓝图（Plugin/Tool/Hook/Workflow 架构） | 已交付 |
| specs/2026-06-13-phase-0-foundation.md … phase-8-draft-zone.md | Phase 0–8 阶段详细 spec | 已交付（逐阶段进入发布线） |
| specs/2026-06-13-phase-frontend-v2.md | 前端 V2 spec | 已交付 |
| plans/2026-07-10-notebook-stress-category-continuity.md | Notebook 压力分类连续性计划 | 已交付 |
| specs/2026-07-10-notebook-stress-category-continuity-design.md | 同上设计 | 已交付 |
| plans/2026-07-12-pmml-scoring-validation-non-regression.md | PMML 打分验证非回归计划 | 已交付 |
| specs/2026-07-12-pmml-scoring-validation-non-regression-design.md | 同上设计 | 已交付 |
| plans/2026-07-17-strategy-platform-gap-analysis-and-roadmap.md | 策略平台差距分析与路线 | 被取代（策略七步 07-27 收口） |
| specs/2026-07-19-strategy-report-bundle-spec.md | 七步报告 bundle spec | 已交付 |

### 2.4 `docs/releases/`

| 文件 | 主题 | 处置 |
|---|---|---|
| 2026-07-02-intermediate-pr.md | 中间 PR 记录 | 被取代（发布历史见 git tag 与 versioning.md） |
| 2026-07-03-v2-mainline-pr.md | V2 mainline PR 记录 | 被取代（同上） |

## 3. 历史审查报告（已移除）

| 文件 | 结论摘要 | 处置 |
|---|---|---|
| 2026-06-21-v2-full-code-review.md | V2 全面 code review 与不变量速查 | 被后续审查覆盖 |
| 2026-06-28-v2-improvement-proposals.md | 多专家前瞻性改进提案 | 被 07-02 综合审查吸收 |
| 2026-06-28-v2-plan-code-review.md | 早期 plan/code 审查快照，finding 已修复 | 被取代 |
| 2026-06-28-v2-runtime-deep-review.md | 分支深度审查（对抗验证受服务限流，余 9 条待最终验证） | 被后续审查覆盖 |
| 2026-07-02-v2-comprehensive-improvement-review.md | 115 条全方位审查，产出 master-backlog | 已被实施与后续审查收口 |
| 2026-07-03-fin1-landing-verification.md | FIN-1 落地核验：182/182 全部落地 | 历史证据 |
| 2026-07-03-fin2-fin3-closing-review.md | FIN-2 全量审查 + FIN-3 修复循环：停机判定达成 | 历史证据 |
| 2026-07-03-vd11-design-token-inventory.md | 设计 Token 收敛清单（radius-pill） | 已完成 |
| 2026-07-04-full-read-and-owner-qa.md | 两轮全量读码 + 所有者五问，产出 Trust-First 计划 | 已被实施 |
| 2026-07-24-data-feature-model-final-review.md | 数据/特征/模型全流程最终审查（07-28 复核修订） | 被 E2E 复审覆盖 |
| 2026-07-24-data-feature-ui-code-review.md | 提交前初始审查快照，finding 已修复 | 被取代 |
| 2026-07-28-e2e-dfm-full-code-review.md | 数据处理/特征/模型 E2E：PASS_FOR_PR（Standards+Spec 双轴） | 历史证据 |
| 2026-07-28-e2e-risk-analysis-full-code-review.md | 风险分析 E2E：独立审查通过 | 历史证据 |
| 2026-07-28-v2-strategy-full-code-review.md | 策略开发全量审查：独立审查通过 | 历史证据 |
| 2026-07-31-project-agent-architecture-comprehensive-audit.md | 项目/Agent/架构健康综合审计（含全流程能力六问） | 结论并入 08-01 收口 |
| 2026-08-01-comprehensive-audit-final-closure.md | 综合审计最终收口：高置信僵尸已清理；仍有技术债，适合定向治理，不宣称"已无屎山" | 历史结论（已被 08-09/08-10 发布复审超越） |
| 2026-08-01-comprehensive-audit-remediation.md | 综合审计整改记录 | 已完成并复审 |
| 2026-08-01-final-local-closure-code-review.md | Final local closure 复审：恢复绑定冲突 BLOCKER 已关闭 | 历史证据 |
| 2026-08-01-presenter-trust-code-review.md | Governed Tool Presenter 信任审查：CR-01/02、WR-01 已关闭 | 历史证据 |
| 2026-08-01-remediation-code-review.md | 整改审查的历史 finding 快照 | 被取代 |

## 4. 历史裁决要点（`v2-longtail-adjudications.md` 全文要点，原文已移除）

- **LT-9（OS 级沙箱 vs subprocess+护栏）**：2026-07-03 裁决——V2 威胁模型为单机单用户，subprocess+护栏（真 OOM 进程树击杀、环境白名单、kernel 真杀，均经 TST-4 验证）已足够；沙箱升级曾定为 V3+ 准入项。**2026-07-17 已重开**：多用户/生产执行隔离迁入 V2 实施轨。
- **LT-10（legacy live-session 路径）**：定为 triple-opt-in legacy-only 终态（`notebook_isolated_execution=False` + `allow_legacy_live_notebook_execution=True` + 环境变量三重门槛），只保安全修复，语义=调试后门、生产禁用。
- **LT-12（row-level 聚合去重）**：裁决不触发、维持现状；未来出现"必须保留真实行"的业务要求时以新 spec 重开。
- **LT-16（roadmap Phase 3 开放式打磨）**：仓内 EXC 建模极致清单已清零；真实数据对照实验记为**等待用户输入的显式外部依赖**（= 现在的新计划 A-3/T4-3）。
- **LT-17（定期复审机制）**：已定型并执行三轮（2026-06-13 / 06-21 / 07-02）；节奏 = 每完成 2–3 个阶段跑一轮聚焦审查。
- **LT-18（V3+ 方向蓄水池）**：2026-07-03 曾把多用户/部署打包/决策引擎/多机调度/实时监控接入列入 V3+。**2026-07-17 已撤销**：全部迁入 V2 实施轨。当前口径见 roadmap：V3/V4 只留给未来需要 major 兼容断裂的决定，不作为 V2 backlog 蓄水池。

## 5. 仓库根目录本地工作产物（从未被 git 跟踪，已删除）

| 文件 | 主题 | 处置 |
|---|---|---|
| CODE_REVIEW_2026-06-13.md / -round2 / -round3 | 2026-06-13 代码审查三轮 | 被 docs/reviews/ 系列取代 |
| CODEBASE_DESIGN_AUDIT_2026-06-13.md | 代码库设计审计 | 被后续综合审计取代 |
| REVIEW.md | V2.1.19 六大功能 UI 验收最终 review（CLEAN，绑定 base `9845e907`） | 发布证据由 release tag 与 docs/reviews/ 承接 |
| marvis/agent/word_conclusion_writer.md | Word 结论草稿模板（代码未引用） | 死文件，删除 |

## 6. 保留文档地图（当前仍生效）

**权威文档（保持维护）**：

- `README.md` / `README.zh-CN.md`、`DESIGN.md`、`docs/roadmap.md`、`docs/versioning.md`、`docs/capability-status.md`、`docs/runbook.md`、`docs/notebook_contract.md`、`docs/对notebook的要求.md`、`docs/branding.md`、`docs/sample_weight_guide.md`、`docs/deploy-linux-env-checklist.md`、`docs/ks_baseline/README.md`、`CONTEXT.md`（领域词汇表）。

**仍生效的计划与证据**：

| 文件 | 为什么保留 |
|---|---|
| docs/superpowers/plans/2026-08-13-next-90-days-development-plan.md | 当前 90 天执行计划（开发工作以此为准） |
| docs/plans/v2-real-materials-reconciliation-checklist.md | T4-3 真实材料对账的人工步骤，仍然生效 |
| docs/plans/specs/v2-validation-batch-spec.md | 被 roadmap 引用的批量模型验证规格 |
| docs/reviews/2026-08-09-v2-3-0-release-readiness.md | 被 capability-status 引用的发布就绪复审 |
| docs/reviews/2026-08-10-comprehensive-code-review-remediation.md | 最新整改复审记录 |
| docs/reviews/closure-public-ks-2026-07-24.md | T4-2 公开 KS 门证据 |
| docs/reviews/closure-real-materials-machine-check-2026-07-24.md | T4-3 机器预检证据 |
| docs/reviews/closure-smoke-2026-07-24.md | Smoke 收口证据 |

## 7. 文档规则重申

- `docs/superpowers/specs/` 与 `docs/superpowers/plans/` 仍是未来功能设计与实施计划的输出位置；本次归档不废弃该工作流，新计划继续写在那里。
- 不要再在 `README.md` / `AGENTS.md` / `CLAUDE.md` 复制完整路线；链接到 `docs/roadmap.md`。
- 任何已交付能力的验收状态以 `docs/capability-status.md` 的分层矩阵为准；历史文档结论不构成当前承诺。
