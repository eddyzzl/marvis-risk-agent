# 策略分析平台对标审计与 MARVIS V2 全功能改造计划

> 日期：2026-07-17
> MARVIS 基线：V2.1.14，`main@a498f3ca`
> 对标项目：`/Users/eddyz/Downloads/业务学习/策略/risk_analyzer_platform(1)/risk_analyzer_platform`
> 文档性质：现状审计、产品边界判断与后续实施计划；执行 2026-07-17“全部已列能力归入 V2.x”的产品决定，并与 `docs/roadmap.md` 和 `docs/plans/v2-master-backlog.md` 同步
> 版本承诺：本文全部 Phase 均属于 V2.x；V3/V4 不承载本文 backlog
> 验证方式：参考项目做只读源码审计；MARVIS 做源码、契约、入口和测试审计，并运行策略相关回归测试

## 一、结论先行

### 1. MARVIS 是否已经包含参考平台的全部策略功能？

**没有。**

MARVIS 已经具备较扎实的策略确定性内核和治理底座，包括策略构造、回测、分数带、规则挖掘、规则集评估、champion/challenger 对比、采纳、版本、监控、额度定价、审计和 Agent Workflow。它在“可治理、可追溯、工具隔离、确定性计算”方面明显强于参考平台。

但参考平台面向策略分析师的完整交互工作台仍有一批 MARVIS 未覆盖或只部分覆盖的能力，主要是：

- 受治理 SQL 导入、完整数据预览/修改/派生、字段语义与中英文映射、风险方向覆盖；
- 加权自动决策树和可逐节点编辑的交互树；
- 标准 WOE-LR 评分卡工作台、分值写回和策略桥接；
- 投票池及受控组合搜索；
- 二维/三维自动交叉规则，以及二维矩阵切点和选中单元格入池；
- 策略池的手工增删、排序、单规则回测和可视化漏斗；
- 件数与金额双口径、逐月、分群、分群逐月回测；
- 独立策略验证集对策略池、已应用交互树、评分卡、已应用 voting 组合和二维 cross matrix group 的复用；
- 多 sheet 策略 Excel 报告及 Python/SQL 代码生成；参考实现没有 JSON 包或逐行等价性验证，这两项是 MARVIS 应增加的治理能力；
- 面向策略人员的统一 Strategy Workbench，而不是目前的通用计划轨道。

此外，MARVIS 还有若干“工具已经实现，但标准产品入口不可达”的缺口，不能把注册了 Tool 等同于用户已经能完整使用。

### 2. Agent 是否已经能自动代替策略人员完成这些工作？

**不能，也不应把“完全替代策略人员”设为近期产品目标。**

合理目标是“受监督的策略 Copilot/Operator”：Agent 自动完成确定性计算、候选生成、受约束搜索、回测、证据整理、报告生成和监控检查；策略人员负责业务目标、数据口径、风险偏好、规则取舍、例外政策、最终采纳、生产上线和异常处置。

**V2 最终产品形态（2026-07-18 补充决定）**：MARVIS 是自然语言优先的 Strategy Operator，而不是参考平台页面的机械复刻。参考平台中策略人员能够操作的每项能力，都必须能由用户用自然语言提出目标后，由 Agent 自动选择并编排受信任 Tool/Workflow 完成；Manual UI 是可选的审阅、调整和证据视图，不是使用完整能力的前置条件。Agent 默认推进到真正需要人承担责任的业务取舍、人工决策门或受治理副作用门，再请求明确确认。这里的“Agent 能做全部操作”不等于转移人的最终业务责任。

对标采用**能力覆盖并超越**而不是设计照抄：参考平台做不到、做不全或做得不安全的部分，MARVIS 必须补成统一 DSL、严格失败语义、可解释 lineage、确定性验证、受控执行、版本治理和可恢复闭环；不得为了界面相似而保留静默跳过、语义漂移、自由 `exec`、重复算法或必须逐页手工操作等缺陷。

即使后续补齐参考平台全部功能，也不能消除以下人工责任：

- 确认目标标签、表现窗、样本边界、金额口径和数据泄漏风险；
- 明确坏率、通过率、利润、资本、运营容量等经营约束及其优先级；
- 判断候选规则是否合理、稳定、可解释、可执行，是否存在不当客群影响；
- 选择 champion/challenger，并记录采纳或否决理由；
- 批准本地采纳、生产发布、回滚和监控红灯后的处置。

### 3. 当前最重要的不是先堆新算法，而是先修两个 P0

1. **完整策略开发入口未接通。** 前端“开始策略开发”实际进入轻量 `strategy_analysis`，只完成构造、回测和权衡；已实现的 `strategy_development` 采纳、版本、文档和监控计划流程没有从标准入口进入。
2. **策略强制确认门只是模板约定，不是运行时不变量。** novel plan 或 decision-point replan 可以生成不带 `needs_confirmation` 的高风险步骤；当前 `PlanValidator` 和 `ToolRunner` 没有对策略采纳等副作用要求绑定批准凭证。

在这两个问题修复前，不应宣传“Agent 可自动完成完整策略开发闭环”。

**2026-07-17 产品决定**：本审计识别的全部参考平台功能、持续经营闭环及组织/生产治理均在 V2.x 内交付。工程量通过多个 V2 minor / prerelease 消化，不再分配到 V3/V4。

**完成审计硬标准**：参考平台的 13 个页面逐项映射到 MARVIS 能力清单，每项同时证明 Tool/Workflow 可执行、自然语言 Agent 可达、结构化证据可审阅、必要门禁不可绕过、结果可持久化/导出；只存在代码函数、只存在 Manual 页面或只能通过测试直接调用均不算完成。

---

## 二、审计范围与判断口径

### 2.1 参考平台的真实定位

参考平台是一个 React/Vite + FastAPI + pandas/sklearn 的**本地单用户策略分析工作台**，不是 Agent 系统。其 13 个页面构成了较完整的人工分析链路：

1. 数据导入
2. 数据预览
3. 数据修改
4. 描述统计
5. 单规则分析
6. 自动决策树
7. 交互式决策树
8. 评分卡
9. 投票池
10. 交叉分析
11. 策略池
12. 策略报告
13. 验证集

参考价值主要在产品工作流和分析师交互，而不是其底层架构。它以全局 pickle/JSON、进程内 DataFrame 和本地任务文件快照保存状态；策略选择、参数、节点、矩阵单元和规则顺序仍由人决定。

### 2.2 不能照搬的参考实现

以下实现不能直接迁入 MARVIS：

- 新数据导入没有完整清理旧策略和旧分析状态，存在跨数据污染风险；
- 评分卡不是标准 WOE-LR：它在原始特征上拟合 LR 并报告训练内 AUC/KS，却用原始尺度系数乘分箱 WOE 生成 points，并用总体坏率替代模型截距；报告指标不代表最终 points，只能视为启发式评分；apply/export 还会重新训练或重新分箱；
- 候选规则、树和评分多数在同一训练样本内搜索并报告，没有强制 OOT/holdout；
- 用户 Python 派生通过服务进程内 `exec` 执行，不是真正安全沙箱；
- 单用户全局状态不适合多任务并发、审计复现或多用户治理；
- 投票池所谓“穷举”实际有 Top10 和每层最多 200 组合的截断；
- 交叉规则枚举没有完整的运行量预算，存在组合爆炸；
- 交互树叶规则进入策略池时会丢弃类别 `!=` 和纯空值条件，页面树与入池规则可能不等价；
- 策略池 reorder 以请求 ID 子集覆盖原池，遗漏 ID 会静默删除规则；
- 验证数据和训练分布不进入任务快照，切换任务会清除，不能形成稳定复现链；
- 参考项目主要业务页面缺少规范自动化测试和 CI。

因此，本计划借鉴“功能闭环和人机交互”，不复制其状态管理、代码执行、算法口径或测试方式。

### 2.3 MARVIS 的审计口径

本次按四层分别判断，避免把“代码存在”误判为“产品可用”：

1. **Kernel**：是否有确定性算法或 Tool；
2. **Workflow**：是否进入受验证的 Workflow 和确认门；
3. **Product reachability**：普通策略任务能否从真实 UI/API 到达；
4. **Governance closure**：采纳、审批、产物、监控和新版本是否形成可信闭环。

---

## 三、MARVIS 当前策略能力实况

### 3.1 已有确定性内核

`marvis/packs/strategy/manifest.json` 当前注册 34 个 Tool：

| 能力族 | 已实现 Tool | 当前情况 |
|---|---|---|
| 组合风险 | `vintage_curve`、`roll_rate_matrix` | Vintage 有入口；roll-rate 无内置策略 Workflow |
| 收益 | `profit_calc` | Tool 可用；标准策略任务的利润输入契约未完整接通 |
| 策略构造、应用与回测 | `build_strategy`、`apply_strategy`、`backtest_strategy`、`tradeoff_view`、`design_cutoff_bands`、`measure_pool_impact` | 五类 DSL 可确定性构造、逐行应用和回测；approval/reject Pool 已有 first-match、整体/逐月及可用金额影响证据；类型化额度/定价/分群、分群×月和 OOT 仍在 Phase 4/5 |
| 版本与挑战者 | `compare_strategies`、`adopt_strategy`、`render_challenger_report`、`render_strategy_doc` | 完整模板可用；普通入口与下载消费面不足 |
| 规则策略 | `mine_rules`、`evaluate_rule_set`、`select_rule_set` | 有独立规则策略 Workflow，包含 waterfall/overlap 和确认门 |
| 监控 | `run_strategy_monitoring`、`render_monitoring_report` | 可人工发起一次监控；未形成调度和处置闭环 |
| 额度定价 | `limit_pricing_matrix` | Tool 可用；无内置 Workflow 和标准入口 |

此外，`analysis`、`feature`、`modeling`、`data_ops` 已提供描述统计、分箱、IV/KS/AUC/PSI、相关性/VIF、评分卡建模、数据打分、稳定性、分群和即席聚合等可复用能力。缺口不应通过在 strategy pack 内复制这些算法解决，而应通过跨 pack Workflow 和统一 Strategy Workbench 组合。

### 3.2 已有 Workflow

| Workflow | 当前步骤 | 产品可达性 |
|---|---|---|
| `strategy_analysis` | 构造 → 回测确认 → 权衡 | 标准策略入口默认进入；只是轻量分析 |
| `strategy_development` | 权衡 → 分数带 → 构造 → 回测 → challenger → 采纳 → 文档 | 后端已注册；标准策略入口未接通 |
| `rule_strategy` | 规则挖掘 → 规则选择 → 评估 → 构造 → 回测 → 采纳 → 文档 | 特定规则意图可达 |
| `strategy_monitoring` | 运行监控 → 告警处置门 → 监控报告 | 特定监控意图可达；仅当前 task 已采纳策略 |
| `vintage_analysis` | Vintage 分析 | 风险分析入口可达 |

### 3.3 当前真实用户链路

前端策略卡的初始目标是“开始策略开发”，但 `StrategyProposal.template_id` 默认仍是 `strategy_analysis`；turn handler 只对“规则策略”和“策略监控”做特殊路由。因此普通用户看到的是“策略开发”，实际完成的是“策略分析”。

现有完整 E2E 测试直接调用 `driver.start(template_id="strategy_development")`，没有覆盖“前端/API 创建策略任务 → setup 路由 → 完整策略开发 → 采纳 → 产物下载”这一真实旅程。

### 3.4 当前产物与消费面

已能落盘：

- decision table CSV；
- monitoring plan JSON；
- strategy/challenger/monitoring Markdown；
- limit-pricing CSV。

但通用“报告已就绪/下载报告”主要识别模型和特征 XLSX；策略产物位于 task 的 `strategy/` 路径，当前完成态 UI 没有稳定地把 artifact ref 变成可见下载动作。也没有参考平台式的多 sheet Excel 策略报告，或经过等价性验证的 Python/SQL 决策代码包。

---

## 四、逐项覆盖矩阵

状态说明：**覆盖**表示当前可直接完成主要目标；**部分**表示内核存在但 Workflow、交互、口径或交付不完整；**缺失**表示没有等价能力；**MARVIS 优势**表示超出参考平台。

| 能力 | 参考平台 | MARVIS 当前状态 | 判断与缺口 |
|---|---|---|---|
| 任务、版本和审计 | 本地文件快照 | DB task、artifact、工具证据、策略版本和 audit | **MARVIS 优势**；仍缺真实 actor、maker-checker 和 production 状态 |
| 文件与 SQL 导入 | CSV/Excel + 多种 SQL | 文件/Excel 和数据集治理较完整；无同等策略侧 SQL 工作台 | **部分**；后续应做受治理 connector，不复制任意 SQL/全量 pandas 路径 |
| 数据预览、清洗、派生 | 丰富手工页面和任意 Python 函数 | data/feature Workflow 有受控工具，无同等策略工作台和任意代码 | **部分**；拒绝引入进程内 `exec` |
| 字段语义与中文映射 | target/金额/月字段、中英映射、风险方向 | semantic roles、数据字典、目标/分数方向已有 | **部分**；缺策略任务专用字段契约和统一映射 UI |
| 描述统计和相关性 | 全变量统计、分布和相关性 | feature/analysis 的 Kernel 可复用，策略入口未整合 | **部分**；跨 Pack 内核覆盖，产品可达性未完成 |
| 单规则分箱与指标 | tree/quantile/equal/chi，KS/IV/WOE/LIFT，件数+金额 | `bin_feature`、feature metrics、规则挖掘 | **部分**；缺金额维度、交互选箱入池及策略语境导出 |
| 加权自动决策树 | 特征/权重/深度/叶约束，叶规则和代码 | 建模树模型和浅树规则挖掘，但无等价加权规则树产品 | **部分** |
| 交互式决策树 | 节点统计、最佳切点、手动分裂/删节点、自动续建 | 无 | **缺失** |
| 评分卡 | 简化评分卡、写回和代码 | modeling 已有更规范的 scorecard/打分/PMML | **平台覆盖、策略集成部分**；需 scorecard-to-strategy bridge |
| 投票池 | 单规则候选、组合搜索、自定义规则、命中分 | 无独立 n-of-k/voting 工具和 Workflow | **缺失** |
| 交叉分析 | 2D/3D 自动交叉规则；2D 切点矩阵和 cell 选择入池 | 有 cross feature，但无等价规则枚举、规则矩阵和 cell 选择 | **缺失**；3D matrix 若实现属于 MARVIS 扩展，不是对标必需项 |
| 策略池手工编排 | 增删、排序、单条回测、级联规则 | 已有稳定 rule/entry id、task-scoped Pool、自然语言增删/改动作/完整重排、CAS 和只读编译 | **主体覆盖**；不要求照抄手工 UI，仍缺单规则独立回测视图和完整 Workbench 展示 |
| 规则挖掘与规则集评估 | 多页面人工组合 | `mine/evaluate/select_rule_set`、waterfall、overlap、确定性排序 | **覆盖/优势**；候选空间仍小于参考交互面 |
| 策略回测 | 整体、漏斗、逐月、分群、金额 | approval/reject Pool 已有 first-match waterfall、总体/逐月动作风险、标签覆盖、可用金额和同任务基线 delta；另有既有 swap、分群、利润能力 | **部分**；缺分群×月、策略专用 OOT、类型化额度/定价/分群影响和完整报告 |
| 独立验证集 | 策略池、已应用交互树、评分卡、已应用 voting 组合、2D cross matrix group；PSI、月度/分群 | 有通用模型验证和监控底座 | **部分**；缺策略专用 OOT、多 artifact 一致应用和持久化复现 Workflow |
| Excel/代码交付 | 多 sheet Excel、Python/SQL | CSV/JSON/Markdown 和 artifact 下载底座 | **部分**；缺 Excel 以及 Python/SQL/JSON 等价性测试包 |
| champion/version/monitor/pricing | 参考平台较弱 | compare、adopt、version、monitor、limit/pricing Tool | **MARVIS 优势但产品未闭环**；入口、阈值、调度和发布仍缺 |
| Agent 规划和治理 | 无 Agent | PlanDriver/Validator/Executor/ToolRunner、确认门、记忆和证据 | **MARVIS 优势但有 P0 门禁绕过面** |

综合判断：MARVIS 已经超过“demo 级策略模块”，但还不是参考平台那种分析师功能全集，更不是可无人值守替代策略团队的生产决策系统。

---

## 五、Agent 自动化边界

### 5.1 当前可以自动完成

- 数据集和字段候选探测；
- 分数方向诊断与规则 AST 校验；
- 分箱、cutoff/分数带扫描和受预算约束的候选搜索；
- 规则命中、waterfall、overlap、通过率、坏率、swap、分群和收益计算；
- champion/challenger 指标对比；
- schema、post-check、红旗和证据记录；
- 报告草稿、决策表和监控结果生成；
- 在已经批准的输入和约束内重跑可逆分析步骤。

### 5.2 当前可以自动准备，但必须人工确认

- 计划总览及现有策略步骤的推进；
- 分数方向/阈值、分数带、规则集和回测结果选择；
- challenger 与 champion 对比后的本地策略采纳；
- 已采纳策略的一次监控，以及红黄灯处置意图选择。

这里描述的是内置模板的当前行为；在 P0 门禁不变量修复前，novel plan/replan 仍存在结构性绕过面。

### 5.3 尚未实现，但目标态仍须人工批准

- 业务目标和经营约束的结构化定稿；
- development 到 validation/OOT 的策略晋级；
- 真实监控阈值变更并重跑；
- 监控红灯后创建新版本任务；
- 本地版本回滚、production promotion、shadow 和 production rollback。

这些不能作为当前 Agent 已有能力宣传。

### 5.4 始终必须由人承担

- 数据、标签、表现窗和金额口径的业务真实性；
- 风险偏好与经营约束冲突时的优先级；
- 政策合理性、可解释性、例外处置和最终责任；
- 生产环境发布、回滚及外部系统变更授权。

### 5.5 建议的自动化目标

| 层级 | 含义 | MARVIS 判断 |
|---|---|---|
| L1 辅助分析 | Agent 解释和建议，人操作所有步骤 | 已具备 |
| L2 受监督执行 | Agent 执行计算，在关键门停下 | 已部分具备；需先修 P0 门禁 |
| L3 有边界自治 | 在已批准目标、数据和预算内自动搜索、验证、报告；采纳/发布仍人工 | V2.x 完整交付目标 |
| L4 无人授权的生产决策 | 自动采纳、发布和处置生产策略 | 永久治理非目标；V2 支持人工授权下的 production 操作，但不取消责任边界 |

---

## 六、问题清单与优先级

本节的 P0/P1/P2 表示缺陷优先级，不是产品版本号。

### P0：发布承诺和治理安全

#### P0-1 标准入口没有进入完整策略开发

- 前端初始目标：`marvis/static/js/task-types.js`；
- 默认 proposal：`marvis/agent/strategy_setup.py`；
- setup 路由：`marvis/agent/turn_handlers.py`；
- 轻量/完整模板：`marvis/orchestrator/templates/strategy.py`。

必须同时补齐策略任务业务输入 contract：`objective`、`max_bad_rate`、`min_approval_rate`、`baseline_strategy_id`，以及请求利润目标时条件必填的 `ead_col`、`pd_col` 和 `profit_params`。否则接到完整模板后仍只能用 20% 分位数等技术默认值替代经营决策。

`review_capacity` 应等真正实现 review-band 决策语义后再加入，不能先放一个无人消费的字段。`adoption_reason` 也不属于任务创建输入；它必须在用户看完最终回测、红旗和策略内容后的采纳门现场填写，并绑定最终 strategy hash、backtest id 和批准记录。空值或“待确认”占位不得通过。

#### P0-2 高风险策略步骤没有运行时强制确认不变量

**2026-07-18 决策覆盖（supersession）**：`human_decision_gate=required` 只保护需要自然人承担最终责任的状态选择，例如策略采纳、监控结果处置、阈值版本变更、生产 promotion/deploy/rollback 和 break-glass。分数方向诊断、候选 cutoff/分数带设计、规则集搜索/排序、回测、比较和本地 artifact 导出属于可逆、可重算的确定性分析；当自然语言目标和业务约束完整时，Agent 必须自动推进，不再设置强制人工门。缺经营目标、风险/通过率约束、经济口径、字段角色、标签处理或标签语义时仍须先澄清；Manual UI 也可提供可选审阅/调整，但不能把可选审阅伪装成 Agent 必经门。`decision_point=True` 继续用于自动复核和 replan，不等同于人工确认。本文及旧 S2/S4 spec 中与此冲突的 cutoff、规则集、回测 mandatory-gate 表述，以本决定为准。

内置模板和 AUTO 风险映射会停下，但底层仍有结构性缺口：

- `PlanValidator` 只对 Join 和 draft-run 强制确认；
- LLM plan/replan 的 `needs_confirmation` 缺省为 false；
- replan 可以替换剩余步骤；
- ToolRunner 校验 side-effect 声明，但不验证绑定 plan/step/input-hash 的批准凭证。

这里需要两套正交 policy，不能把所有确认都等同于副作用授权：

1. `human_decision_gate=required`：保护策略采纳、监控处置、生产 promotion/deploy/rollback 等自然人责任决策；由 PlanValidator、GateEnvelope 和 AUTO 强制。纯计算 Tool 只有在其结果本身触发上述责任状态转换时才可要求此门，普通候选设计、回测、比较和导出不得滥用；
2. `effect_authorization=required`：保护策略采纳/替换、监控阈值写入、未来发布/回滚等状态转换；由 ToolRunner fail closed 校验带外执行授权。

模板只能提高、不能降低 policy 规定的级别。最终采纳授权还必须绑定已审阅规则集 hash、最终 backtest id、目标策略版本和 expected current status，避免绕过前置决策门直接 build/adopt。

### P1：现有能力的闭环缺口

1. `roll_rate_matrix`、`profit_calc`、`limit_pricing_matrix` 没有内置 Workflow；
2. 完整模板主要仍按 approval 策略构造，limit/pricing/reject/segmentation 没有真实产品路径；
3. 利润计算需要的 EAD/PD/成本参数没有从标准策略任务完整传入；
4. 策略 artifact 已生成，但完成态 UI 缺少稳定的展示和下载；
5. 监控计划写入自定义 warn/fail 阈值，策略侧判级仍固定使用对称 ±5pp/±10pp；
6. “调阈值重跑”只返回提示，不更新版本化 monitoring plan；
7. “起新版本”只建议 `strategy_development`，不会创建任务或调用 `new_version_from`；
8. 当前审计 actor 主要是 `system`，无法表达真实操作者和审批者；
9. 旧计划 `docs/plans/v2-strategy-risk-analysis-plan.md` 仍写“规划中、未开发”，与当前实现状态不一致。

### P2-A：V2 策略分析师完整工作台能力

- 文件/SQL 导入、数据预览/修改/派生/导出、修改历史和分析状态恢复；
- 字段语义、中英文/变量节点映射、风险方向建议与人工覆盖；
- 描述统计、相关矩阵和分布；
- 金额口径单规则分析；
- 加权/约束自动规则树；
- 交互树；
- voting/n-of-k 组合；
- 二维/三维交叉规则、二维矩阵切点和 cell 选择；
- Agent 自然语言 Strategy Pool CRUD/order 已完成；后续补完整 Workbench 展示，不以复制参考平台手工页面为目标；
- approval/reject Pool 的 first-match、逐月、件数和可用金额首纵切已完成；后续补分群、分群逐月、swap、OOT 和其余类型化口径；
- 策略专用 OOT 验证；
- Excel 和等价代码交付；
- 分析产物列写回与修改后数据集导出；
- Manual/Agent 共用的 Strategy Workbench。

### P2-B：V2 持续经营、组织与生产治理

- 用户身份、RBAC、maker-checker 和多级审批；
- 定时监控、告警通知和失败重试；
- production deployment/promotion/rollback；
- shadow/challenger 运行和决策引擎对接；
- 实时评分 API、第三方 Plugin/Workflow 治理和生产执行隔离；
- 周期化组合经营、策略收益风险复盘和红灯到新版本闭环。

以上全部属于 V2.x 承诺范围。P2 只表示实施优先级较 P0/P1 晚，不表示延后到其他 major。

---

## 七、目标架构与关键决策

### D1：新增统一 Strategy DSL/IR，作为所有策略产物的唯一事实源

DSL 至少包含：

- 稳定 `rule_id` 和版本；
- typed condition：数值、类别、缺失、集合、区间；
- 显式 `AND`、`OR`、`NOT`、`n-of-k`、first-match/priority 语义；
- approval/reject/review/limit/pricing/segmentation 输出；
- 空值、边界包含关系和默认决策；
- source artifact、训练/验证数据引用和生成参数。

同一个 evaluator 驱动回测、验证和生产交付；Python/SQL/JSON 都由 DSL 生成，并用 golden dataset 做逐行等价性测试。禁止页面、Agent 和报告各自重写一套规则解释器。

### D2：Strategy Workbench 是编排和证据视图，不新建重复算法层

- 数据清洗放 `data_ops`；
- 分箱、树和交叉特征内核放 `feature`/`modeling`；
- 规则组合、策略语义、回测、采纳和定价放 `strategy`；
- 组合风险、趋势和分群分析复用 `analysis`；
- Workbench 只消费结构化 payload、artifact 和 gate contract。

Manual 和 Agent 使用同一 Workflow、Tool、证据与确认门，只是控件皮肤不同。

### D3：候选搜索必须有预算和验证制度

每个树、投票、交叉枚举工具必须声明：

- 最大特征数、深度、候选切点数和组合数；
- timeout/memory budget；
- 确定性随机种子和 tie-break；
- train/validation/OOT 边界；
- 多重搜索后的验证要求；
- 不可行、截断和降级状态，不能把 truncated 称为 exhaustive。

### D4：V2 提供安全等价的自定义派生能力，不引入服务进程内 `exec`

常规派生使用受限表达式 DSL；高级自定义函数通过受信任 Plugin Tool 或隔离子进程/容器执行，带权限声明、资源限制、网络/文件边界、审计和显式批准。不能因为拒绝参考平台的不安全 `exec` 就删除这项用户能力。

### D5：V2 分离策略资产生命周期与环境部署生命周期

V2 通过兼容迁移把当前 `adopted` 扩展为策略版本自身的 `draft → validated → adopted_local → retired`；本地 champion 切换和 rollback 记录为独立、可审计的指针变更/事件，而不是伪造成生产部署状态。旧记录迁移为 `adopted_local`，且不会凭空生成 deployment。

生产侧另建 environment-scoped `DeploymentRecord`，至少覆盖 `pending → shadow → active/superseded/failed/rolled_back`，并绑定 environment、strategy version、manifest、外部 deployment ref、健康状态和 expected current status。同一策略版本可以同时在测试环境 shadow、在生产环境 active；单一环境的 promotion/rollback 不得改变策略资产状态或其他环境记录。API、报告和 UI 必须明确本地采纳与各环境部署的差异。

### D6：人工决策门与副作用授权门正交，并使用持久化批准状态机

`human_decision_gate` 的确认记录与 `effect_authorization` 的执行授权分别保存。执行授权不是严格业务 inputs 的一个字段，而是独立持久化 `ApprovalRecord`，由专用人工批准入口在服务端签发；AUTO/LLM 永远不能签发。

Phase 0B 可先用服务端派生的本地 session principal 关闭当前绕过面；V2 后续治理阶段必须升级为真实用户身份、RBAC 和 maker-checker。客户端或 LLM 永远不能自填 actor，也不能替人工签发批准。

PlanExecutor 通过带外 `ExecutionContext(plan_id, plan_revision, step_id, decision_id, approval_id?, runtime_generation, human_decision_required, effect_authorization_required)` 把人工决策证明和可选的一次性副作用授权传给 ToolRunner。纯人工决策门必须验证 DecisionRecord；受保护副作用还必须 reserve/consume 对应 ApprovalRecord。任何直接调用受治理 Runner 的路径缺少有效、完整绑定的证明时必须 fail closed。

批准记录至少绑定：

- task、plan、`plan_revision/replan_count`、step 和 tool；
- tool manifest hash 与 policy schema/version/hash；
- `$ref` 解析后的 normalized business input hash；
- 被审批 dependency output/evidence 的版本或 hash；
- effect target、目标策略版本和 expected current status；
- 本地 session principal、人工理由、签发时间、一次性 nonce 和过期时间。

状态机至少包含 `issued → reserved → consumed`，以及 `expired/revoked`。规范必须定义 Tool 失败、超时、重试、Executor 恢复和 reserve 后进程崩溃的处理，既不能重复消费，也不能因可恢复失败永久锁死。

任何 replan、依赖输出重算、输入、manifest、policy 或 effect target 改变都使旧批准失效。确认纯业务决策时同样要绑定所审阅 evidence，但不向只读 Tool 发放副作用执行授权。

---

## 八、分阶段开发计划

以下全部 Phase 都属于 **V2.x 承诺范围**。Phase 只表示依赖和实施顺序，可以拆成多个 V2 minor / prerelease 交付；任何阶段都不得重新挂到 V3/V4。

策略最终报告统一遵守 [`Strategy Report Bundle 契约`](../specs/2026-07-19-strategy-report-bundle-spec.md)。实施顺序坚持 **strategy-first**：报告渲染不成为策略引擎的前置依赖；缺少可选报告资料时继续执行可逆分析，用户明确表示暂缺后在任务内持久化 `unavailable` 并将报告值留空；缺少会改变策略语义或确定性结果的信息时仍必须 fail closed。

### Phase 0A：接通真实入口和业务 contract（V2.x P0，3-5 人日）

**目标**：让标准策略入口真实进入完整开发，且没有业务口径时不会静默使用技术默认值。

**报告契约**：[`Strategy Report Bundle 契约`](../specs/2026-07-19-strategy-report-bundle-spec.md)；本 Phase 负责缺失信息状态和 task-scoped report context，不实现报告渲染器。

交付：

1. 新增 StrategyTaskInput contract：`objective`、坏率/通过率约束、baseline，以及利润目标条件下的 `ead_col`、`pd_col`、成本/期限等参数；
2. 新增 `MissingInformationRecord` 和 task-scoped report context，区分 `strategy/impact/validation/report_optional`；可选信息只主动询问一次，用户回答暂缺后持久化 `unavailable`，同一任务不重复询问；
3. “开始策略开发”默认进入 `strategy_development`，保留显式“快速策略分析”进入 `strategy_analysis`；
4. 为轻量分析、完整开发、规则策略、监控、额度定价和组合分析建立明确 intent taxonomy；
5. 缺 objective 或必要约束时暂停澄清；只有用户显式选择“快速分析”才允许技术默认值；
6. adoption reason 在最终采纳门现场必填，不从任务创建或占位文本继承；
7. 增加真实 API E2E：创建策略任务 → `/agent/start` → 完整模板自动推进到采纳门 → 人工确认本地采纳 → 文档/artifact。

退出标准：

- 普通“开始策略开发”不再落入轻量模板；
- 完整开发缺经营约束时返回澄清，不用 20% 分位数等默认值代替用户决策；
- 请求利润但缺 EAD/PD/必要成本参数时返回结构化缺口，不把利润记为 0；
- 报告可选信息缺失不阻塞策略主链；用户表示暂缺后重启/replan 不再追问，报告字段保持空白；
- 标签语义、样本边界、动作单位等策略正确性 blocker 即使用户暂缺也不能被 `unavailable` 绕过；
- 采纳理由为空、“待确认”或来自预填占位时不能采纳；
- 轻量入口保持向后兼容；API E2E 不直接调用 `driver.start(template_id=...)` 绕过产品入口。

### Phase 0B：把门禁升级为运行时不变量（V2.x P0，7-12 人日，含 1-2 人日 spike）

**目标**：任何模板、novel plan、generic plan API、decision/failure replan 或直接 Runner 调用都不能绕过人工决策门和副作用授权门。

交付：

1. 在统一 policy schema 中分别声明 `human_decision_gate` 和 `effect_authorization`；
2. PlanValidator、GateEnvelope、AUTO、UI 文案和 ToolRunner 从同一 policy 来源派生行为；
3. 持久化 DecisionRecord/ApprovalRecord 和 V2 本地 session principal；
4. 增加人工签发入口、带外 ExecutionContext、`issued/reserved/consumed/expired/revoked` 状态机；
5. ToolRunner 对采纳、替换和其他受保护副作用 fail closed；AUTO 永远不能签发授权；
6. 现有 replan 校验路径消费新增 policy，不允许重规划降低门禁；
7. 修复前端 AUTO 文案，明确哪些决策永远需要人确认；
8. 补齐授权并发、失败、超时、重试、进程崩溃和恢复规范。

退出标准使用策略矩阵测试证明：

- `strategy_analysis` 回测、`strategy_development` cutoff/回测、`rule_strategy` 规则选择/回测和普通 monitoring run 均不要求强制人工门；`strategy_development`/`rule_strategy` 采纳、monitoring disposition 及未来生产动作遵守对应人工决策与副作用 policy；
- template、novel plan、generic plan API、decision replan 和 failure replan 删除或降低强制门时均失败；
- AUTO 对所有 `human_decision_gate=required` 只能 halt，且不能签发 `effect_authorization`；
- Runner 无凭证、错 plan/step/tool、跨任务、过期、重复消费或 policy/manifest 不匹配时 fail closed；
- 输入未变但 plan revision 改变、依赖 evidence 重算、目标策略状态改变时旧批准失效；
- 并发双执行只能有一个消费成功；reserve 后崩溃按规范恢复，不重复副作用。

Phase 0B 的状态机和调用链影响面大，**7-12 人日包含 1-2 人日 spike，但仍须在 spike 后重估，不作为承诺**。

**实施状态（2026-07-18）**：Phase 0B 的治理运行时基础已完成代码收口，包含统一 policy snapshot、validator/replan 防降级、AUTO 强制停机、服务端本地 session principal、不可变 DecisionRecord、一次性 ApprovalRecord、effect execution ledger、带外 ExecutionContext、Runner/Executor fail-closed、策略采纳原子 receipt、启动恢复，以及当前已有 monitoring disposition/report gate 的强制人工决策约束。专项回归覆盖治理、策略开发、规则策略、监控和 API 链路，提交前仍以全量 `scripts/check` 结果为准；本地 session 仍是 V2 单机阶段的过渡身份，真实用户、RBAC 和 maker-checker 继续按后续治理阶段实施。

Phase 0B 的完成结论只覆盖上述治理底座及当时已有监控门禁。其后 Phase 1 已完成自定义阈值真实判级、阈值变更生成版本化 monitoring plan 并重跑，以及监控红灯到新策略版本的 handoff；后续范围继续保留在 V2.x，不迁移到 V3/V4。

### Phase 1：现有策略闭环和统一 DSL（V2.x P1，9-14 人日）

**目标**：把现有 18 个 Tool 变成真实可交付产品链路，并建立后续扩展的唯一策略语义。

**实施状态（2026-07-19）：Phase 1 已完成。** 统一 Strategy DSL 覆盖 approval、reject、limit、pricing、segmentation；五类策略均可由平台确定性设计候选、构造、回测、逐行应用和本地采纳。类型化采纳绑定 task、dataset/hash、strategy effect、回测证据和经济口径，并原子提交生命周期、effect receipt、decision table、monitoring plan、artifact 与 audit。自然语言编译器可把开发、分析/回测、应用、比较、候选设计、规则挖掘、标准利润/roll-rate/额度定价分析、监控和报告请求实例化为受信任 Workflow；可逆步骤自动执行，只有本地采纳和监控处置保留人工责任门。

版本化监控账本已真实执行计划阈值和额度/定价经济绑定；模型 PSI/CSI/KS/AUC 阈值与策略阈值分流但共同可版本化调整，调阈值会追加 plan revision 并用原 evidence 重算，红灯“起新版本”会创建 task-scoped 子版本与数据 handoff。监控 run、处置 receipt 和报告均做 hash、ownership、CAS、语义一致性和不可变 lineage 校验。完成态 UI/API 可下载 task-owned CSV/JSON/Markdown artifact，并明确区分 `draft/validated/adopted_local/retired` 与尚未发生的生产部署。上传材料格式错配和 CSV 坏行保留结构化恢复链。当前 challenger 比较保留 approval/reject 兼容指标；limit/pricing/segmentation 的类型化 champion/challenger 指标属于 Phase 4 的 report-ready 比较，不用 approval/reject 口径冒充。完整 Strategy Workbench、Candidate Lab 和七步最终报告仍按 Phase 2-6 实施，不反向算入 Phase 1。

先做逐类型 contract/design spike，分别钉死 approval、reject、limit、pricing、segmentation 的：输入、rule value、默认决策、核心回测指标、采纳产物和监控基线；未完成设计的类型不能只靠修改 `strategy_type` 字符串宣称可用。

交付：

1. Strategy DSL v1、schema migration、稳定 rule id 和 evaluator；
2. 现有 approval/规则策略迁移到 DSL，保持旧策略可读；
3. `profit_calc`、`roll_rate_matrix`、`limit_pricing_matrix` 内置 Workflow；
4. 按已评审的逐类型 contract 接通 approval/reject/limit/pricing/segmentation；review-band 和 `review_capacity` 只在语义完整后加入；
5. 把 EAD/PD/成本、baseline 等 task inputs 真实传入对应 Tool；
6. 修复 strategy monitoring 按 monitoring plan 的自定义阈值判级；
7. “调阈值”生成版本化 plan 并重跑；“起新版本”经确认创建 task 并调用 `new_version_from`；
8. 完成态 UI 显示并下载 CSV/JSON/Markdown artifact；
9. 为当前 `adopted` 增加向完整生命周期迁移所需的兼容字段，在 API/artifact/UI 明确本地采纳和未部署状态；
10. 更新旧 specs/plan 的实现状态，消除“代码已实现、文档仍未开发”的漂移。

退出标准：

- evaluator 与迁移前所有现有策略决策逐行一致；
- 定制监控阈值能真实改变判级，且有边界测试；
- 每个宣称可用的 strategy type 都有 contract、Tool/Workflow、产物、监控口径和标准入口 E2E；
- 用户能从完成态 UI 下载全部策略产物，并清楚看到本地采纳、验证和部署状态的差异。

### Phase 2：Data & Semantics Workbench（V2.x，10-16 人日）

**目标**：补齐参考平台的数据准备、语义配置和描述分析功能，并以 MARVIS 数据集、artifact 和审计体系持久化。

**报告契约**：[`Strategy Report Bundle 契约`](../specs/2026-07-19-strategy-report-bundle-spec.md)；本 Phase 负责 report-ready 指标、当前状况、样本和外部历史资料的结构化语义。

**首个纵切（已完成）**：已建立 task-scoped `DataWorkspaceSnapshot`，用 revision/`If-Match` 保存 active dataset、dataset hash、analysis generation、页面选择和字段语义；新增 task-owned preview，并复用现有 CSV/XLSX 导入。切换 active dataset 必须清空旧字段选择和语义，旧 dataset/artifact 仍保留为历史证据；底层 parquet 若与注册 hash 漂移，工作区读取、保存和预览均失败关闭；保存请求在途时不允许用本地 discard 冒充服务端写入已撤销。

**旧版 Excel 纵切（已完成）**：HTTP 上传、本地路径注册和 Agent `data_ops.ingest_excel` 均可按真实文件内容读取 BIFF `.xls`，并与 `.xlsx/.xlsm` 共用工作表、精确数据行数和文件体积门禁；扩展名伪装不能绕过 Excel 上限，损坏或 HTML 伪装文件会显式失败且不登记半成品。测试使用固定真实 BIFF fixture，不引入过时的写入依赖。统计、受限变换和导出已由后续纵切完成；SQL connector 仍在本 Phase 和 V2.x 内继续交付。

**报告级描述分析纵切（已完成）**：新增自然语言可达的 `dataset_descriptive_analysis` Workflow 和 `data_ops.profile_dataset`，对活动数据集做全量、确定性的概览、target、缺失、低基数频数/Top-K、高基数等宽直方图及完整 Pearson 相关矩阵；支持显式字段范围和资源预算，超限或不可安全表示时给出类型化原因，不静默截断或把不可用相关性伪装成 0。超出 JavaScript 安全整数范围的值以无损 `bigint` 字符串输出，高精度数值若无法安全进入 DOUBLE 指标则显式 unavailable。分析绑定 task、dataset hash、workspace revision、analysis generation 和 semantic mapping hash；敏感字段按数据集推断与用户语义的并集做稳定 token，并抑制数值分布和相关性。Agent 可用自然语言直接选择分析范围并展示确定性表格；手动 API 提供异步 job、显式 retry、page-only revision 缓存复用、不可变 JSON artifact 和 task-scoped 下载，数据/语义漂移、artifact/job 冒充、进程中断后的失联 job 均失败关闭并可审计恢复。

**受治理变换与导出纵切（已完成）**：新增自然语言可达的 `dataset_transform` 和 `dataset_export` Workflow。变换只接受 rename/drop/cast/fill/filter/derive/dedup 的封闭 AST，不执行任意 SQL/Python；每次执行生成新 Parquet 数据集、版本化 transform run、逐边 lineage 和 task-owned JSON evidence，并以 dataset/hash、workspace revision、analysis generation、semantic hash 和规范化输入做缓存身份。字段语义随 rename/drop/cast 确定性迁移，registry 已识别但尚未写入 workspace 的 target 也纳入有效语义；删除 target、ID、手机号等受保护字段必须在 Agent 对话中绑定原请求二次确认，确认期间数据或语义漂移即失效。CSV/XLSX 导出流式读取活动数据集，自动保护敏感文本列、长整数/高精度数值和公式注入，CSV 使用 UTF-8 BOM，XLSX 固定文档与 ZIP 元数据以获得稳定内容 hash；导出 artifact 同样绑定完整数据身份、事务内复核、文件 hash、task 路径和下载权限，并拒绝 symlink 跳转。SQL connector、隔离自定义派生、风险方向、`CurrentProjectSnapshot` 和外部历史资料映射仍是本 Phase 后续交付，不能据此把 Phase 2 整体标为完成。

**`StrategySampleDesign` 与下游强绑定纵切（已完成，2026-07-22）**：用户可通过自然语言把活动 DataWorkspace、dataset/hash、明确的二元目标与 `target_bad_value=0|1`、表现窗/观察窗/成熟度、可选 development/validation/OOT 切分，以及可选月、权重、放款金额、逾期金额字段，固化为不可变、task-owned 的样本设计和版本化 `MetricDefinition`/`MetricObservation`。风险指标依赖已成熟样本；空标签只有经明确同意才从风险分母排除，未成熟设计保持 `exploration_only / unvalidated`，不能进入策略开发主链。

下游单变量、automatic tree、Voting、Cross/refinement lineage、既有 candidate/tradeoff/bands/rule 工具、typed V2 Workflow backtest 和 Pool impact 现在统一要求精确的成熟 development `StrategySampleDesignRef`，其规范字段为 artifact id/hash、sample design id/hash 与 `partition=development`。平台解析并复核同一 task、活动 dataset/hash、workspace revision/generation、semantic mapping、target 与空标签策略，在落盘前再次检查漂移；`target_bad_value=0` 由确定性内核归一为同一坏样本语义。旧的未绑定 active plan 不得继续用于这些 V2 路径，必须 fail closed。底层 direct `backtest_strategy` 仍为既有 V1/外部调用保留未绑定兼容边界，但所有 V2 策略开发 Workflow 都必须解析并注入该 ref，不能借兼容入口跳过样本设计。当前样本设计仍把同一上游数据边界同时用于风险与通过率观测；风险/通过率双样本、渠道/客群纳排、历史回溯打分、泄漏/选择偏差检测、`CurrentProjectSnapshot` 和历史资料映射仍是 V2 Phase 2 待办；独立 OOT、七步最终报告和统一 Strategy Workbench 也仍分别按 Phase 4-6 继续开发，未因本纵切完成而提前标记完成。

交付：

1. 策略任务创建、列表、加载、删除、显式保存、dirty 切换保护和分析状态恢复；
2. CSV/XLS/XLSX 与受治理 SQL connector；SQL 至少覆盖 SQLite、MySQL、PostgreSQL、Hive、Impala 的连接测试、参数校验和查询导入；
3. 数据概览、字段详情、target 分布、前 N 行、空值分析和分布图；
4. 缺失填充、删列、类型转换、受限表达式派生、过滤、重命名和修改历史；
5. 隔离自定义派生 Tool：权限/资源/网络/文件限制、审计、超时和显式批准；
6. target、loan amount、overdue amount、month、中英文名称、变量节点和大小写规则映射；
7. 风险方向映射导入、p25/p75 确定性建议、逐变量人工覆盖和版本化保存；
8. 描述统计、完整相关矩阵、低基数频数、高基数直方图；
9. 修改后数据集的 CSV/Excel 导出，所有派生数据集保留 lineage；
10. 新增版本化 `MetricDefinition`/`MetricObservation`，显式支持件数、金额、余额、表现窗和成熟度；生成 `CurrentProjectSnapshot`、`StrategySampleDesign`，并把外部历史版本资料映射为带来源的 artifact，而不是自由文本事实。

退出标准：

- 新数据导入不会继承旧策略、旧树或旧页面选择；
- SQL 凭证不进入 task artifact、日志、LLM context 或记忆；
- 自定义派生无法访问未授权文件、网络或环境变量，超时/OOM 可终止并审计；
- 字段映射、显示语言、风险方向、修改历史和页面选择随 task 保存/恢复；
- 描述统计和导出结果与数据集版本/hash 对应，不读取自由文本状态。
- 同名风险指标的件数/金额/余额口径不可混用；未成熟 observation 的值为空且状态为 `not_matured`，不能假报 0。

### Phase 3：完整 Candidate Lab（V2.x，18-28 人日）

**目标**：完整覆盖单规则、自动树、交互树、标准评分卡、voting 和 cross 分析，而不是只提供高层搜索内核。

**报告契约**：[`Strategy Report Bundle 契约`](../specs/2026-07-19-strategy-report-bundle-spec.md)；本 Phase 只产出 report-ready candidate evidence，不在报告层复制算法。

**统一样本 lineage 与 automatic-tree apply 纵切（已完成，2026-07-22）**：单变量 evidence、候选选择/合并、Cross、automatic tree、Voting 及其 Pool lineage 都沿 Phase 2 的同一精确 `StrategySampleDesignRef` 传播，混用不同样本设计、旧未绑定候选或漂移后的活动计划时失败关闭。完整 automatic-tree asset 已支持自然语言触发的确定性逐行 apply；产出是带 lineage 的非活动派生数据集，状态保持 `development / unvalidated`，不会因 apply 自动入 Pool、采纳或部署。该纵切没有完成交互树、评分卡、Voting 自动搜索、Cross 自动搜索、逐月稳定性、代码生成或列写回。

**单变量候选分析首个纵切（已完成，2026-07-19）**：新增自然语言可达的 `strategy_univariate_candidate_analysis` Workflow 和 `strategy.analyze_univariate_candidates`。数值字段可按等频、等距、ChiMerge 和受约束决策树分箱，类别字段使用类型保持的等值箱；统一输出缺失/哨兵箱、左闭右开 DSL 条件、count/good/bad、占比、坏率、WOE、IV、Lift、累计 KS、方法级 KS/AUC、风险方向，以及放款金额、逾期金额和配对逾期率口径。执行绑定确认时的 task、dataset/hash、workspace revision、analysis generation 和 semantic mapping hash，计算后在事务内再次复核；敏感/标识字段不会进入候选，资源超限和不可用方法显式失败或记录 typed evidence，不静默换方法。结果固化为自认证 `CandidateEvidence`，明确 `development/unvalidated`，并生成字节稳定、公式安全、task-owned 的 JSON/XLSX（Summary、Rankings、Bins、Metrics、Red Flags、Lineage）供下载和后续七步报告组装。该纵切尚不等于 Phase 3 完成；后续完成情况以下方纵切状态为准，Strategy Pool 入池、逐月稳定性、树、评分卡、Voting、Cross、代码生成与列写回仍按本 Phase 继续交付。

**候选选择与合并纵切（已完成，2026-07-19）**：新增自然语言可达的 `strategy_univariate_candidate_refinement` Workflow、已有证据专用的 `strategy_univariate_candidate_refinement_existing` Workflow 和 `strategy.refine_univariate_candidate`。Agent 只提取用户明确给出的 source bin id、合并组、观测坏率门槛和选择理由；平台还会把门槛、比较符、bin id 和操作词与用户原话确定性核对，未给定可执行标准的“选最好”会先澄清，不能由 LLM 生成指标、边界或规则。只使用坏率门槛时可在同一两步 Workflow 中先分析再筛选；一旦点名 source bin id 或合并组，用户必须同时引用其实际查看结果中的完整 candidate ID，平台直接解析同 task 的不可变 `strategy_candidate_json`，不允许重新分箱后用 `regular:n` 序号重绑。Tool 校验预期 artifact/candidate/evidence hash，从父证据恢复 dataset/workspace/target/金额口径，投影读取所需字段，使用同一 DSL evaluator 逐行重放父箱并核对分区；数值普通箱只允许相邻合并，类别值保持严格 JSON 类型，missing/sentinel 不会静默并入普通箱。合并后全部 WOE/IV/KS/AUC/Lift、件数和金额指标从绑定行重新计算，不聚合旧摘要；结果固化为 `strategy.candidate-asset.v1`，带稳定 candidate-rule/candidate-effect/candidate-asset id、父级 lineage、`development/unvalidated` 状态和 task-owned 内容寻址 JSON 下载。该资产尚未等于 Strategy Pool 条目，入池、逐月稳定性及其他 Candidate Lab 能力继续按本 Phase 交付。

**显式 Voting 首纵切（已完成，2026-07-19）**：新增自然语言可达的 `strategy_voting_candidate_build` Workflow 和 `strategy.build_voting_candidate`。用户必须点名当前 Pool 中 2 至 50 个完整 rule id、唯一 Pool 类型和 n；平台按 Pool 顺序绑定 entry，拒绝嵌套 Voting、重复条件、模糊推荐和复合操作，只加载所选 K 条 lineage，并在共享绑定样本上逐行重放 canonical `n_of_k`。结果包含完整 0..K 命中数分布、件数与可用金额 `MetricObservation`、样本/标签口径、稳定 rule/fragment/effect/asset id、父 Pool artifact lineage 和 task-owned canonical JSON，状态保持 `development/backtested/unvalidated`。入池必须另发请求并明确选择 `before_selected_members`（保留成员作为后续回退）或 `replace_selected_members`（一个 CAS revision 内原子替代成员）；普通 append、无授权全局置顶、重复等价条件和 first-match 遮蔽均 fail closed。当前仍未完成自动组合搜索、人工自定义成员编辑、预算截断、命中数列写回与 Python/SQL 代码，因此不能把 Voting 整项标记为完成。

**2D Cross Matrix evidence 首纵切（已完成，2026-07-19）**：新增自然语言可达的 `strategy_cross_matrix_analysis` Workflow 和 `strategy.build_cross_matrix_candidate`。用户必须在唯一、当前、肯定式命令中明确 X/Y 字段及各自分箱方法；平台先生成 task-owned 单变量 `CandidateEvidence`，再精确校验其 artifact/candidate/evidence hash，从同一绑定样本逐行回放两轴一次，生成包含空单元格在内的完整 Cartesian matrix。每个 cell 固化类型化 `AND` 规则、稳定 rule/effect/cell id、count/good/bad、share、bad rate、Lift、WOE、IV contribution 及可用金额观测；矩阵同时带独立 candidate/evidence id、父级单变量 lineage、样本/语义 hash、有限 cell budget 和 task-owned canonical JSON，状态保持 `development/backtested/unvalidated`。该首纵切自身不自动推荐或选择 cell，不生成 group，不入 Strategy Pool，不写回列，也不采纳或部署；显式 cell/group 物化与入池由下方后续纵切承接，人工切点、2D/3D 自动交叉搜索、代码与列写回仍需继续完成。

**2D Cross Matrix 显式 cell group 与 Pool 纵切（已完成，2026-07-19）**：新增自然语言可达的 `strategy_cross_matrix_cell_selection` Workflow 和 `strategy.materialize_cross_matrix_cell_selection`。用户必须引用唯一完整 matrix asset ID，并逐字给出一个或多个完整 cell ID；平台拒绝别名、模糊推荐、否定式和同轮复合操作，按源矩阵行主序规范化选择，固化只保存 selection/source 指针和 cell ID 的不可变 artifact，不复制 predicate、指标、动作或生命周期。group id、fragment、规则和 effect 与可选选择理由无关；group 条件是各 cell 类型化 `AND` 规则的 canonical `OR`，件数/风险/金额先聚合绑定主样本的原始观测，WOE/IV 按“所选 cells 合并为一个分组、其余 cells 保持独立”的完整分区重算。用户必须另发请求并用 selection ID 入 Pool；整张 matrix asset 不能直接入池，同一 matrix 在同一 Pool 中允许多个互斥 group，但重复 group 或任何 cell 重叠均无写入失败。Pool 在持锁事务内重新校验 selection、matrix、父 CandidateEvidence、dataset registry/path/hash，并重放 fragment 后才持久化；结果仍是 `draft/development/backtested/unvalidated`，不采纳、不部署。Cross 的人工切点、2D/3D 自动交叉搜索、代码与列写回仍需继续完成。

**Candidate Lab Manual/Agent parity 首纵切（已完成，2026-07-23）**：策略任务宽桌面工作区新增 Candidate Lab，展示 task-owned、重新验真的单变量、Cross Matrix、自动树及当前 Strategy Pool；可直接启动单变量、单变量 fresh/existing refinement、Cross Matrix 和自动树。Manual 表单提交与自然语言共用 `strategy_request`、严格 compiler、PlanValidator、模板和 Tool，不维护第二套指标内核；existing refinement 的 candidate、feature/method 和 bin 只能从服务端投影选择，artifact/hash 不进入用户输入。当前活动 DataWorkspace 改变或不可用时，已有不可变 candidate 的 refinement 仍从父 artifact 恢复数据/sample lineage，并在创建计划前深验 canonical bytes、provenance、feature/method/bin；错配、损坏或跨任务引用只返回 clarification，不创建计划。投影使用 task/kind 的 `COUNT + DESC LIMIT`、总字节预算、来源 canonical/provenance 重验、重复 source cache、最新非终态 plan/assistant message 的单行查询；前端同任务 single-flight，任务切换取消旧请求，不进入轮询 tick。该纵切只覆盖已经存在的四个启动器和证据摘要，不提前宣称 Pool 全操作、交互树、评分卡、自动搜索、稳定性、代码/列写回或完整 evidence drawer 已完成。

交付：

1. **单规则**：tree/quantile/equal-width/chi 四类分箱、类别等值箱、3-20 箱、最小样本、KS/IV/AUC/WOE/LIFT、件数+金额、批量排序、人工选箱/合并入池和 Excel；
2. **自动规则树**：特征/权重/深度/最小叶、风险方向违例、节点/叶指标、路径规则、SVG/PNG、Python/SQL 和叶 ID 写回；
3. **交互树**：节点统计、全特征最佳分裂、单变量候选、手工分裂、删节点、自动续建、缺失/类别分支、可视树、代码、叶 ID 写回和叶规则入池；
4. **标准评分卡 Workbench**：特征选择与 KS Top-N、可复现 WOE-LR、KS/AUC/IV/系数、分值表、分布、评分列写回、Python/SQL 和 strategy bridge；
5. **Voting**：自动候选、人工增删、自定义规则预览、2..K 受预算组合、明确 truncated、n-of-k 阈值、分数分布、命中数写回、入池和代码；
6. **Cross**：2D/3D 自动规则；2D matrix 切点建议/人工调整/空值归属/cell 入池；规则和矩阵代码生成。3D matrix 不属于对标必需项；
7. 所有候选输出统一 Strategy DSL fragment、稳定 artifact id、development evidence 和 lineage；Phase 5 完成独立验证后再关联 validation evidence；
8. 单变量、模型、树、评分卡、Voting 和 Cross 统一输出 `CandidateEvidence`、`MetricObservation`、生成参数、搜索预算、截断状态和来源引用，供七步策略报告直接组装。

退出标准：

- tree/voting/cross 的搜索可复现、有限额、可取消，截断时不得称为 exhaustive；
- 类别 `!=`、纯空值、边界包含关系和缺失分支有 golden tests，树到策略池不得丢条件；
- 页面预览、DSL evaluator、写回列和生成代码逐行同义；
- 评分卡训练指标必须代表最终 points，apply/export 不得重新训练或重新分箱；
- 任一规则进入 draft 策略池前展示 development evidence；尚无匹配 validation evidence 时必须显式标记 `unvalidated`，不得冒充已验证或已采纳。
- 同一 candidate 在页面、DSL evaluator、写回列和报告中的 rule id、指标及 effect stage 完全一致。

### Phase 4：Strategy Pool、回测与交付（V2.x，12-18 人日）

**目标**：把候选 artifact 组合成可版本化策略，并完整覆盖级联回测、报告页面和部署交付。

**报告契约**：[`Strategy Report Bundle 契约`](../specs/2026-07-19-strategy-report-bundle-spec.md)；本 Phase 实现七步报告组装和主报告 artifact，但报告失败不能回滚已完成的策略 evidence。

**approval/reject Pool 影响测算首纵切（已完成，2026-07-19；样本强绑定于 2026-07-22 补齐）**：新增自然语言可达的 `strategy_pool_impact` Workflow 和 `strategy.measure_pool_impact`。用户只需说明要测算的 approval/reject Pool、可选基线及可选精确月份/金额列；Pool revision/hash、候选 lineage、精确且成熟的 development `StrategySampleDesignRef`、活动 dataset/hash、workspace revision/generation、确认 target 和 semantic mapping hash 均由平台绑定，不接受 LLM 注入。Tool 按样本设计的 `target_bad_value` 在同一 development 样本上重放 canonical first-match StrategySpec，输出总体动作/风险、每条规则 standalone/incremental/shadowed/remaining、默认未命中、标签覆盖、可用放款/逾期/配对金额观测、可选逐月及同任务同类型基线件数、风险和金额 delta；逐月件数、标签、风险、金额、动作与规则 incremental 均须回卷总体。结果以 canonical、内容寻址的 `strategy.impact-assessment.v2` JSON 和通用 TaskArtifact registry hash 双层固化，计算后在写入事务内再次复核 Sample Design、Pool/candidate lineage、dataset registry/path/bytes、DataWorkspace 与 baseline；空标签未确认、旧未绑定 active plan、任何漂移或守恒失败都不落盘。artifact 内 hash 用于与可信 expected hash 对账，不是数字签名；脱离原始 frame/spec 的离线 validator 只证明 schema、派生字段和内部守恒，消费持久化 evidence 时必须先核对 TaskArtifact registry 中的 content hash，不能把任意重写后再哈希的 JSON 当成平台 provenance。该纵切只产生 `development / backtested / unvalidated` 证据，不创建、修改、采纳或部署策略。它不是 Phase 4 完成：limit/pricing/segmentation 专属影响语义、分群/分群×月、swap、OOT、代码与列写回仍须继续交付。

**七步 Strategy Report Bundle 纵切（已完成，2026-07-23）**：`strategy_report_bundle_v2` 会从 task-owned project context、历史策略、`StrategySampleDesign`、单变量/模型证据、候选/Pool、ImpactCube 和用户补充的 report fields 组装固定七节、不可变 revision。可选信息缺失或用户明确“暂缺”时保留 typed availability 并在读者报告中留空，空白与数值 0 严格区分；缺少会改变策略语义、样本或确定性结果的 binding 时失败关闭。输出为同一 revision 的 canonical JSON、Markdown、模块化 XLSX 与可解析 DOCX，参考用户提供的迭代评审模板和两份项目报告，但不复制其人工操作界面；四个 artifact 使用内容 hash、task ownership、canonical path、provenance 和审计绑定原子登记，任一格式渲染/登记失败会回滚整套报告登记而不回滚上游策略 evidence。DOCX 使用固定业务简报结构、可审计 evidence 标识和安全文本投影，不允许外链、字段代码或用户文本注入媒体。额度/定价专属扩展、独立 OOT 章节和完整 browser/API 旅程仍须继续完成。

交付：

1. 单规则、组合规则、n-of-k、稳定 rule id、增删、删除、完整 reorder、指标预览和单规则回测；
2. first-match/priority 级联 waterfall：每条规则只消费上一条后剩余样本；
3. 整体、单规则逐月、全策略逐月、件数、金额、分群、分群×月、标签/金额覆盖率；
4. 分群过滤支持 AND/OR、`>`、`>=`、`=`、`<=`、`<`、`in`、`not in`；
5. 为 approval、reject、limit、pricing、segmentation 实现类型化 champion/challenger 比较，分别输出适用的件数、金额、余额、风险和经济指标；不得把额度、定价或分群策略降格套用 approval/reject 二分类口径；
6. 策略、漏斗、月度、分群、Python/SQL/JSON tabs 和 artifact 下载；
7. 实现 `StrategyReportBundle`、`ReportField` 和不可变报告 revision，固化“现状 → 历史 → 样本 → 单变量/模型 → 候选组合 → 影响 → 文档”七步输出；
8. 生成模块化多 Sheet Excel、机器可读 JSON manifest 和 Markdown 执行摘要，覆盖口径、字段映射、修改史、策略、waterfall/swap、逐月、分群、树、评分卡、Voting、Cross、数据成本和代码；`unavailable` 留空，`not_applicable` 与 `not_matured` 按契约展示；
9. 为额度/定价报告增加临额/固额、期限、提降额人数、户均变化、总敞口、使用率、T30 行为、Cap、层级风险、年化风险和利差扩展；
10. DSL→Python/SQL/JSON codegen、逐行 equivalence report 和不支持语义的 fail-closed 清单；
11. tree node、scorecard score、voting hit count、cross group 等列写回及修改后数据集导出。

退出标准：

- reorder 必须是完整、无重复的 rule id 排列，遗漏/未知 ID 返回 typed error；
- 件数与金额 totals 和原始样本对账，级联漏斗各层与剩余样本守恒；
- 平台 evaluator、Python、支持的 SQL dialect 和 JSON fixture 逐行一致；
- 每个 Excel sheet 的指标和代码都能追溯到结构化 Tool output，不由 LLM 重算。
- 相同 evidence 幂等重建同一 content hash；用户补充资料后生成新 revision，不覆盖旧报告；报告渲染失败不破坏策略、回测或验证 artifact。

### Phase 5：独立策略验证集（V2.x，10-16 人日）

**目标**：完整覆盖参考平台验证集交互，同时修复其验证数据和训练 artifact 易失问题。

**报告契约**：[`Strategy Report Bundle 契约`](../specs/2026-07-19-strategy-report-bundle-spec.md)；本 Phase 负责 OOT、成熟度和验证 evidence，不把开发回测包装成 OOT。

交付：

1. 独立 validation/OOT 数据的文件和受治理 SQL 导入、清空和重导；
2. binary label 与无标签触发率模式，可选 loan amount、overdue amount、month；
3. development→validation 字段映射、自动建议、人工覆盖、缺字段显式跳过和红旗；
4. 可选择并一次 apply：策略池、交互树、评分卡、voting、2D cross matrix group；MARVIS DSL 额外支持其他已登记规则 artifact；
5. 策略逐规则/漏斗/月度/分群/分群×月、树节点分布/PSI、评分分布/PSI、voting 规则与命中分布、cross heatmap/group；
6. 每个维度独立 Excel，并在 Phase 4 总报告上追加验证 sheet；PSI 阈值、标签覆盖率和无标签限制显式解释；
7. validation data、训练分布、字段映射、选择维度和 artifact version 随 task/version 持久化；
8. 所有效果 observation 显式区分 `estimated`、`backtested`、`oot_validated` 和未来 `post_launch_observed`；追加验证 Sheet 时生成新报告 revision，不覆盖开发报告。

退出标准：

- development/validation/OOT 不可静默混用，字段缺失不会静默改变策略语义；
- 无标签模式不生成坏率、KS/AUC 等伪指标；
- 任务切换和进程重启后验证数据、训练分布和选择状态可复现；
- 每个验证结果页和 Excel 与同一 evaluator/apply 输出一致；
- 规则或策略只有在引用与当前 dataset/artifact/version 匹配的 validation evidence 后才能晋级 `validated`；`adopted_local` 还必须经过 Phase 0B 的人工决策和副作用授权门。
- 缺 OOT、标签未成熟或字段缺失时相应值保持空白并保留结构化状态，不得把 `backtested` 改写为 OOT。

### Phase 6：统一 Strategy Workbench 与 Agent parity（V2.x，8-14 人日）

**目标**：让策略人员在 Manual 和 Agent 模式下使用同一套完整功能、状态、证据和门禁。

**报告契约**：[`Strategy Report Bundle 契约`](../specs/2026-07-19-strategy-report-bundle-spec.md)；本 Phase 让 Manual 和 Agent 共用缺失信息、七步 Workflow、证据和报告 revision。

信息架构：Data & Semantics → Candidate Lab → Strategy Pool → Backtest & Validation → Champion/Challenger → Adoption & Artifacts → Monitoring & Iteration。

**首个工作台纵切（已完成，2026-07-23）**：Candidate Lab 已实现上述四个 Manual 启动器与受认证结果/Pool 摘要，Manual 和 Agent 已共用同一 deterministic execution kernel；任务切换、active plan/open gate、澄清、失败保留输入、请求去重和投影截断均有独立前后端覆盖。Phase 6 尚未完成，其余区域、完整状态持久化、全 evidence drawer、长任务控制及七步 browser E2E 仍按本 Phase 交付。

交付：

- 每个区域均有完整 Manual 操作面；Agent 调整同一 Workflow/DSL，不维护隐藏副本；
- 结构化 gate 控件覆盖阈值、分数带、规则、顺序、验证晋级、采纳和 production 操作；
- evidence drawer 显示 dataset/artifact/tool/version/input hash/红旗/memory 引用；
- 节点、切点、matrix cell、选中规则、页面位置、语言和字段别名随 task 保存；
- `MissingInformationRecord`、`ReportField` availability、effect stage、报告完整度和 revision 在 Manual/Agent 间完全一致；可选资料只问一次，用户暂缺后任务内不重复询问；
- 长任务进度、取消、恢复、失败重试和可访问性；
- Manual/Agent/browser/API E2E 覆盖完整旅程和七步策略报告。

退出标准：

- Manual 与 Agent 对相同输入和选择产生同一 DSL、指标、写回列和 artifact；
- 页面不解析 LLM 自由文本作为业务事实；
- 缺关键业务口径时 Agent 必须澄清，不用技术默认值静默替代；
- 缺可选报告资料时策略引擎继续，报告对应字段留空；缺策略正确性信息时 Manual 和 Agent 都 fail closed；
- 用户可在一个任务中完成“数据 → 候选 → 策略池 → 回测 → OOT → 采纳 → 监控计划”，刷新/重启后可恢复。

### Phase 7：持续监控与本地经营闭环（V2.x，10-17 人日）

**目标**：把一次性监控升级为可运营的周期闭环。

交付：

1. 本地 scheduler、运行日历、失败重试、去重锁和错过周期补跑；
2. webhook/邮件等通知出口、脱敏 payload、发送审计和重试策略；
3. `draft/validated/adopted_local/retired` 策略资产生命周期、本地 champion 指针和原子 rollback event；
4. monitoring plan 阈值版本、真实调阈值重跑、红灯经确认创建新版本任务；
5. champion/challenger 周期跟踪、shadow-ready 对比和收益风险复盘报告；
6. 组合、Vintage、迁徙、Expected Loss、额度/定价的周期化运行。

退出标准：

- 并发 scheduler、重复触发和错过周期补跑对同一计划周期恰好产生一次有效运行；失败/崩溃可恢复，重启后日历、锁和审计连续；
- 通知 payload 完成脱敏，发送、失败、重试和最终状态都有审计；通知失败不改变监控结论；
- 自定义阈值真实控制判级，每次重跑绑定 monitoring-plan version、dataset version 和 evidence；
- rollback 恢复正确版本、artifact、监控计划和审计链；
- champion/challenger、组合、Vintage、迁徙、Expected Loss、额度/定价的周期任务都产生版本化 evidence 和可追溯复盘报告；
- 红灯只能建议或经人工授权起新版本，不能无人采纳。

### Phase 8：组织与生产治理（V2.x，28-52 人日）

**目标**：在 V2 内完成此前被分配到 V3 的生产化和扩展治理。

交付：

1. 真实身份、session、RBAC、maker-checker、多级审批、职责分离和审批导出；
2. 策略资产状态与 environment-scoped DeploymentRecord 分离，以及当前 `adopted` 和 Phase 0B 本地 principal/ApprovalRecord 的不可伪造兼容迁移；
3. environment promotion、deployment manifest、外部 ref、健康检查、单环境 rollback 和 break-glass 双重授权/事后复核；
4. shadow/challenger runtime、实时评分 API、批量评分和决策引擎 adapter；
5. 第三方 Plugin/Workflow 安装、签名、权限、版本、撤销和回滚治理；
6. 容器/系统沙箱或远程 worker 隔离、secret 边界、网络策略、配额和多租户资源控制；
7. 生产 scheduler、告警、SLO、幂等、灾难恢复和端到端演练。

退出标准：

- 角色矩阵和审批级次按顺序执行，maker 不能批准自己的高风险变更；审批导出完整可验，break-glass 需要双重授权和事后复核；越权、过期、重放和跨环境授权全部 fail closed；
- Phase 0B 本地 principal 和历史 ApprovalRecord 迁移到真实身份后保持签发人、理由、hash、时间和消费状态，不可由客户端伪造或改写；
- 同一策略可在不同环境处于不同 deployment 状态；单环境 promotion/rollback 不改变策略资产状态、本地采纳或其他环境，shadow 与 production 结果可对账，并有真实 adapter E2E；
- 实时、批量和离线 DSL evaluator/模型打分在 golden 数据上等价且请求幂等；决策引擎 adapter 超时、部分失败和恢复不产生重复决策；
- 未签名、已撤销或越权 Plugin 不能安装/执行，已安装版本可审计回滚；资源耗尽和跨租户访问测试证明 worker 无法越过 secret、文件、网络和 quota 边界；
- 生产 scheduler 故障转移、告警 SLO、备份恢复和灾难恢复演练达到定义的 RTO/RPO，且 evidence 可复核；
- 全自动无人工授权的采纳和生产变更继续禁止。

### V2.x 交付批次与估算

- **Foundation train**：Phase 0A/0B/1，约 19-31 人日；
- **Analyst train**：Phase 2/3，约 28-44 人日；
- **Strategy train**：Phase 4/5/6，约 30-48 人日；
- **Operations train**：Phase 7/8，约 38-69 人日。

原始合计约 **115-192 人日**；考虑跨阶段集成、迁移和生产演练，建议用 **130-220 人日**作为 V2 全功能规划区间。Phase 0B、SQL connector、交互树和生产 adapter 各自先做 spike，再校准对应 train。全部 train 都属于 V2.x，最后一个 train 完成前不得把缺项重新标成 V3/V4。

---

## 九、建议的原子提交序列

每个提交都应保持测试可运行，避免把入口、安全策略、DSL 和 UI 混成一次大改。

### Phase 0A

1. `test(strategy): expose failing product-route e2e`
2. `feat(strategy): add governed strategy task and report-input contracts`
3. `feat(strategy): persist one-shot missing-information decisions`
4. `fix(strategy): route development intent to full workflow`
5. `feat(strategy): capture evidence-bound adoption reason at gate`

### Phase 0B

6. `test(governance): define decision and effect policy matrix`
7. `feat(governance): add orthogonal gate policy schema`
8. `feat(governance): persist local principal and approval state machine`
9. `feat(governance): pass execution authorization out of band`
10. `feat(governance): fail closed in runner for protected effects`
11. `test(governance): cover replan replay concurrency and recovery`
12. `fix(ui): derive truthful AUTO behavior from gate policy`

### Phase 1

13. `docs(strategy): specify per-type execution contracts`
14. `feat(strategy): add canonical strategy DSL v1`
15. `refactor(strategy): migrate existing evaluators to DSL`
16. `feat(strategy): wire profit roll-rate and pricing workflows`
17. `feat(strategy): implement reviewed strategy-type paths`
18. `fix(strategy): honor versioned monitoring thresholds`
19. `feat(strategy): close monitoring-to-new-version handoff`
20. `feat(ui): surface local-adoption state and strategy artifacts`
21. `docs(strategy): reconcile specs with implemented state`

### Phase 2：Data & Semantics

22. task snapshot、dirty guard 和分析状态 contract；
23. CSV/XLS/XLSX 受治理导入；
24. SQL connector 公共 contract、secret boundary 和连接测试；
25. SQLite/MySQL/PostgreSQL connector；
26. Hive/Impala connector；
27. 数据预览、字段详情、target/空值分布；
28. 受控填充、删列、转换、派生、过滤、重命名和历史；
29. 隔离自定义派生 Tool 与资源/权限护栏；
30. 字段语义、中英文/节点映射和风险方向；
31. 描述统计、相关矩阵、分布、当前项目快照、历史资料映射和 `MetricDefinition/MetricObservation` 数据导出（描述分析、`StrategySampleDesign`、版本化指标及其下游成熟 development 强绑定已完成；风险/通过率双样本、纳排/泄漏检测、`CurrentProjectSnapshot` 与历史资料映射待续）。

### Phase 3：Candidate Lab

32. 单规则四分箱、类别箱、全指标和选箱入池（确定性分析、证据与 JSON/XLSX、证据绑定的选箱/合并、不可变候选资产及其 Strategy Pool 入池已完成；候选级逐月稳定性待续）；
33. 加权自动规则树、方向检查、树图和写回（受限完整树、叶选择/入池及自然语言 full-tree apply 已完成；交互展示、代码和列写回待续）；
34. 交互树节点/候选/手工分裂内核；
35. 交互树删节点、自动续建、可视化、代码和入池；
36. 标准 WOE-LR 评分卡 Workbench；
37. voting 候选、自定义规则、受预算组合和 n-of-k（显式 n-of-k 候选、同样本 lineage 及受治理入池已完成；自动搜索、自定义编辑、代码和列写回待续）；
38. 2D/3D 自动 cross rules 与 2D matrix/cell（2D matrix、显式 cell group、同样本 lineage 及入池已完成；人工切点、自动搜索、代码和列写回待续）；
39. Candidate artifact、代码、写回列和 report-ready evidence 统一到 DSL。

### Phase 4：Strategy Pool、回测与交付

40. 稳定 rule id、pool CRUD 和完整 reorder（Agent 自然语言纵切已完成，Workbench 展示待续）；
41. first-match 级联 waterfall（approval/reject 当前 Pool 的 standalone/incremental/shadowed/remaining 已完成；其余类型待专属口径）；
42. 单规则/策略逐月、件数和金额回测（approval/reject 当前 Pool 的总体、逐月、规则 incremental、标签/金额覆盖、基线 delta、成熟 development Sample Design 强绑定及 `target_bad_value=0|1` 已完成；direct backtest 仅保留兼容边界，V2 Workflow 必须注入 ref；单规则独立视图、分群×月及其余类型待续）；
43. 分群操作符与分群×月；
44. 策略/漏斗/月度/分群/code tabs；
45. `StrategyReportBundle`、七步 Workflow、模块化 Excel/JSON/Markdown/DOCX、额度定价扩展与结构化 provenance（固定七节、不可变 revision、四格式原子登记已完成；额度/定价专属扩展和独立 OOT 章节待续）；
46. Python/SQL/JSON codegen 和逐行 equivalence；
47. 分析产物列写回与修改后数据导出。

### Phase 5：独立验证集

48. validation/OOT 文件/SQL 导入、清空与重导；
49. label/金额/月和字段映射 contract；
50. 多 artifact 选择与统一 apply；
51. 逐规则/漏斗/月度/分群/PSI/voting/cross 结果页；
52. OOT/effect-stage observation、验证 Sheet 新 revision、task 持久化和重启恢复。

### Phase 6：Strategy Workbench parity

53. Workbench shell 与七区信息架构（Candidate Lab 首个宽桌面区域已接通，其余区域待续）；
54. 完整 Manual 控件和 evidence drawer（单变量/Cross/自动树/refinement 启动器及受认证摘要已完成，Pool 全操作、其他候选和完整 drawer 待续）；
55. 节点/切点/cell/页面/语言状态持久化；
56. Agent 与 Manual 共用 Workflow/DSL/gate、缺失信息状态和报告 revision；
57. 完整七步报告 browser/API E2E、进度、取消和恢复。

### Phase 7：持续经营闭环

58. 本地 scheduler、日历、锁、补跑和失败恢复；
59. 通知 adapter、脱敏、审计和重试；
60. 策略资产生命周期、本地 champion 指针、rollback event 和恢复测试；
61. 版本化监控阈值、真实重跑和红灯到新版本；
62. challenger 跟踪、周期组合分析和收益风险复盘。

### Phase 8：组织与生产治理

63. 真实身份、session 和 RBAC；
64. maker-checker、多级审批和职责分离；
65. 策略资产状态、environment-scoped DeploymentRecord、promotion manifest 和单环境 rollback；
66. shadow、批量/实时评分和决策引擎 adapter；
67. 第三方 Plugin 签名、权限、撤销和回滚；
68. 容器/系统沙箱或远程 worker、secret/network/quota 边界；
69. 生产 SLO、灾难恢复和端到端演练。

每个算法提交应同时包含 contract、typed errors、manifest、ToolRunner 真实调用测试和确定性回归，不先堆页面后补内核。

---

## 十、验证计划

### 10.1 每个阶段的最低验证

```bash
conda run -n py_313 python -m pytest -q <最小相关测试>
conda run -n py_313 python -m ruff check marvis tests --extend-exclude '*.ipynb'
node --check marvis/static/app.js
git diff --check
```

### 10.2 必须新增的测试层

| 层 | 必测内容 |
|---|---|
| Contract | StrategyTaskInput、逐类型策略 contract、DSL、artifact、monitoring 和 gate policy schema |
| Data | 文件/SQL 导入、secret boundary、修改历史、语义映射、风险方向和 task snapshot |
| Kernel | 空值、类别、边界、金额、权重、组合截断、不可行约束 |
| Equivalence | evaluator vs Python vs SQL/JSON；旧策略迁移前后 |
| Governance | template/novel/generic plan、decision/failure replan、plan revision、resolved refs/evidence 改变、过期/跨域/重复批准、并发消费、崩溃恢复、直接 Runner 越权副作用 |
| Workflow | 轻量、完整、规则、定价、监控、新版本和 OOT |
| API E2E | 从真实任务创建和 `/agent/start` 进入，不直接调用模板 |
| Browser E2E | Manual/Agent、gate、artifact 下载、长任务取消与恢复 |
| Lifecycle | strategy asset status、本地 champion/rollback event、environment-scoped deployment、调度 exactly-once、通知脱敏/重试和重启恢复 |
| Production | RBAC/maker-checker/多级审批/break-glass、单环境 promotion/shadow/rollback、实时/批量/离线等价和幂等、Plugin 签名/撤销/回滚、租户隔离、SLO 与灾难恢复演练 |
| Regression | V1 手动模式、Agent P1、Notebook/PMML 确定性边界保持稳定 |

### 10.3 本次审计的动态验证结果

本次已运行：

```bash
conda run -n py_313 python -m pytest -q \
  tests/test_strategy_*.py \
  tests/test_rule_strategy_*.py \
  tests/test_portfolio_api.py \
  tests/test_agent_autodrive.py
```

结果：**266 passed in 234.06s**。

这证明当前已覆盖路径的策略内核和相关 Agent 回归为绿，但不证明标准前端入口已经进入完整策略开发；现有测试正缺少这一真实产品旅程。

参考平台未运行其有副作用的脚本：部分脚本会覆盖 `data_temp`、修改当前策略池或依赖本地 8000 服务，启动脚本还会强制清理端口。对其结论来自只读源码审计，不把 `BUG_REPORT.md` 中历史“全部通过”视为当前可复现验证。

---

## 十一、风险、依赖和非目标

### 主要风险

- **Strategy DSL 语义迁移**：边界、空值、类别和 first-match 任一差异都会改变历史策略决策；必须双跑对账；
- **组合搜索爆炸**：voting/cross/tree 必须先有预算和取消机制；
- **金额口径缺失**：没有可靠申请金额/EAD 时不能伪造金额收益；应返回 N/A 和红旗；
- **OOT 数据不可得**：没有独立时间外样本时只能标记验证不足，不能把训练内最优包装成已验证；
- **UI 与内核分叉**：Workbench 只能消费结构化 Tool payload，禁止页面自己重算；
- **文档漂移**：每阶段交付时同步 specs 状态和 roadmap 链接，不复制完整路线到多个入口文档。

### 本轮非目标

- 不复制参考平台的全局 pickle 状态；
- 不支持服务进程内任意 Python `exec`；V2 必须提供受限表达式和隔离自定义派生的安全等价能力；
- 不把本地采纳宣称为生产部署；只有目标 environment 存在 `active` DeploymentRecord 且有 deployment ref 才能显示该环境已上线；
- 不实现参考平台没有的 3D matrix；保留 2D matrix 和 2D/3D 自动交叉规则；
- 不把无人授权的自动采纳、自动 promotion 或自动生产处置当作任何版本目标；
- 反欺诈、征信报文解析等与本次策略平台对标无关的独立业务域不因“V2 全功能”自动纳入；
- 不让 LLM 计算坏率、KS、AUC、PSI、利润或其他确定性指标；
- 不因新增策略能力破坏 V1 模型验证、Notebook 和 PMML 一致性边界。

---

## 十二、证据索引

### MARVIS

- 产品与版本边界：`docs/roadmap.md`、`docs/versioning.md`
- 旧策略计划：`docs/plans/v2-strategy-risk-analysis-plan.md`
- 策略 Tool：`marvis/packs/strategy/manifest.json`
- 轻量/完整/规则模板：`marvis/orchestrator/templates/strategy.py`
- 监控模板：`marvis/orchestrator/templates/monitoring.py`
- 标准策略 setup：`marvis/agent/strategy_setup.py`、`marvis/agent/turn_handlers.py`
- 前端策略入口：`marvis/static/js/task-types.js`
- Validator/replan/safety：`marvis/orchestrator/validator.py`、`planner.py`、`executor.py`、`safety.py`
- AUTO gate：`marvis/agent/gates/contracts.py`、`marvis/agent/auto_drive.py`
- ToolRunner：`marvis/plugins/runner.py`
- 策略监控：`marvis/packs/strategy/monitoring_plan.py`、`monitor_tools.py`
- 版本/采纳：`marvis/repositories/strategy.py`、`marvis/packs/strategy/tools.py`
- 策略路由与 E2E：`tests/test_agent_task_routing.py`、`tests/test_strategy_development_e2e.py`

### 参考平台

- 13 个页面：`risk_analyzer_frontend/src/components/common/Layout.jsx`
- 数据和任务状态：`risk_analyzer_backend/app/api/data.py`、`task.py`
- 单规则、树、评分卡和 voting：`risk_analyzer_backend/app/api/analysis.py`
- 交互树：`risk_analyzer_backend/app/api/interactive_tree.py`
- 交叉分析：`risk_analyzer_backend/app/api/cross_analysis.py`
- 策略池、回测和 Excel：`risk_analyzer_backend/app/api/strategy.py`
- 独立验证：`risk_analyzer_backend/app/api/validation.py`、`validation_core.py`
- 已知问题记录：`BUG_REPORT.md`

## 十三、最终产品定义

这次对标不应把 MARVIS 改造成参考平台的翻版。目标应是：

> **MARVIS V2 Strategy 是一个覆盖数据准备、候选分析、策略池、回测、独立验证、交付、持续监控和生产治理的完整策略平台；Agent 负责把分析、搜索、验证和交付自动化，平台保证确定性、等价性和审计，策略人员负责目标、取舍、授权与责任。**
