# PR: MARVIS V2.0 — full credit-risk agent platform to mainline

Branch: `codex/v2-plugin-tool-runtime` → `main` · Version: **1.1.9 → 2.0.0**
Merge method: **Create a merge commit**（勿 squash——backlog/审查报告/记忆引用的 713+ 个 commit hash 证据链依赖完整历史）。

## What this PR is

V2 从"运行时外壳+建模深度"完成到**全功能信贷风控 agent 平台**，并按收官三环（落地核验→全量审查→修复循环收敛）验证完毕。`docs/plans/v2-master-backlog.md` 的 183 项全 ✅（每行带 commit hash 证据）；`origin/main` 的 V1.1.9 发布线（conda 启动加固、任务创建/停止生命周期修复）已语义合并进来。

能力面（全部 agent/手动双模式、确认门带红旗清单、审计完整）：
- **数据**：JOIN 引擎（键字典/指纹/强制确认）、流式上传护栏、本地路径注册、数据字典全链路、content-hash 去重
- **特征**：train-only 预处理链（sidecar 重放）、类别 WOE、筛选/IV/相关/VIF、泄漏信号
- **建模**：9 recipe 全配方两阶段调参、时间外推 OOT 默认、校准、champion 护栏与披露、模型卡、PMML/PKL 交付、验证交接
- **打分与监控**：训练期基准快照、逐字节预处理重放打分、PSI/CSI/漂移分级告警门、监控计划闭环+逾期可见
- **策略**：tradeoff 可行域、分数带设计、规则挖掘瀑布、版本化采纳（原子 CAS+自动退役）、决策表/策略文档交付物、challenger swap 报告、额度×定价 EL 矩阵
- **组合分析**：流量/桶迁徙热力、细分画像 HHI、马尔可夫吸收链 EL、稳定性趋势、组合报告
- **即席问数**：LLM 出 spec→白名单算子编译参数化 SQL→口径确认门先行（LLM 不算数）
- **平台**：插件协议版本握手、schema 迁移版本化、错误分类学、pack SDK 基座、UnitOfWork 事务化、真进程隔离验证、AUTO 安全门旗矩阵、记忆注入定界、性能回归计数守卫

## Verification（本 PR 树）

- 六轮全量门禁零失败，规模 1988 → **2779+ passed**（py_313 权威环境；合并 origin/main 后的门禁数字见 PR 评论区更新）
- FIN-1 落地核验：182/182 行 plan-vs-code 证据齐全（`docs/reviews/2026-07-03-fin1-landing-verification.md`）
- FIN-2/FIN-3：8 镜头审查+三席对抗表决 → 15 个修复 commit → 第二轮复审零新 critical/high 收敛（`docs/reviews/2026-07-03-fin2-fin3-closing-review.md`）；bandit 42 中危全量裁决误报
- 真 e2e：`marvis serve` 子进程全旅程（JOIN→建模门→PMML）三连跑确定性；并发注入测试（双确认/上传竞态 40 连跑）
- 真浏览器 Playwright smoke 3 通过

## Known limits（记录在案，非阻塞）

- 等用户输入项：真实数据对照实验（脱敏样本到位即跑）、VD 取值类视觉拍板（radius/spacing 对比稿、发光动画默认开）
- 低风险残余：TransactionalDirectoryStore 目录级并发暂存窗口（drafts 单用户手动路径）；materials 上传 symlink 窗口（uuid4 临时目录）
- V3+ 显式不做项与六项长线裁决：`docs/plans/v2-longtail-adjudications.md`

## Post-merge

本地 `main` 所在 worktree 执行 `git pull --ff-only` 即完成同步；分支保留（hash 证据链已随 merge commit 并入 main 历史，删除亦安全，但建议保留至下个版本周期）。
