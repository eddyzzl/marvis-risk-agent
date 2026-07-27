# MARVIS 路线图

本文档是 MARVIS 产品阶段、运行时术语和能力边界的统一来源。历史实施文档可以继续使用 P1/P2/P3、Phase 1/2/3 或批次编号；当前所有已确定产品能力统一归入 V2.x，阶段编号只表示实施顺序，不对应 V3/V4。

## 当前状态

- 当前主线：**MARVIS-Agent V2.x**，本地优先、可治理的多工作流信贷风控 Agent 平台。
- 当前范围决定：**V2.x 承担全部已确定产品能力**；不得用 V3/V4 作为延后 V2 功能的蓄水池。V2.x 可以通过多个 minor / prerelease 逐批交付，但必须在 V2 major 内收口。
- 当前产品面：数据处理、特征分析、模型开发、模型验证、策略开发、Vintage/风险分析、监控、组合分析、额度/定价和即席问数等信贷风控 workflow。
- 模型验证是稳定兼容工作流之一：继续保留 V1.1 手动模式和 Agent 辅助验证能力，Notebook 契约、PMML 对比、确定性验证指标和 Excel/Word 产出必须保持兼容。
- V2 不是只有 Plugin/Tool runtime 外壳；欢迎页展示的入口必须对应真实可用的端到端 workflow：人在环确认、受控工具执行、结构化结果、下载/报告或可审计产物。
- Portfolio / 组合分析能力已有后端工具、模板和测试覆盖；具体是否作为首屏入口或 Agent start allowlist 暴露，以当前代码和 UI 为准。
- 策略平台改造的 Phase 0A（真实完整开发入口与业务 contract）、Phase 0B（运行时门禁、人工决策证明和一次性副作用授权）和 Phase 1（五类统一 DSL、自然语言可逆执行、标准分析 Workflow、版本化监控/处置、新版本 handoff、生命周期与可下载 artifact）已完成。策略采纳和监控处置仍是人工责任门，本地采纳不等于生产部署。Phase 2 及全部后续范围继续在 V2.x 内交付，不得推到 V3/V4。
- Phase 2 的 task-scoped 数据工作区、CSV/XLS/XLSX 导入、报告级描述分析、自然语言受限数据变换、不可变派生 lineage，以及安全 CSV/XLSX 导出已经完成纵向闭环。`StrategySampleDesign` 首个纵切也已完成：用户可通过自然语言把活动 DataWorkspace、数据内容 hash、明确的二元目标好坏极性、表现窗/观察窗/成熟度、可选 development/validation/OOT 切分及月、权重、放款金额、逾期金额字段固化为不可变 task-owned JSON；平台确定性计算 overall/各切分的件数、好坏样本、标签覆盖、金额覆盖与权重观测，并输出版本化、逐对象自认证的 `MetricDefinition` / `MetricObservation`。空标签只有在用户明确同意时才从风险分母排除且仍保留总体样本；表现窗或观察窗缺失、样本未确认成熟时强制标成 `exploration_only / unvalidated`，依赖成熟度的风险指标只返回 `unavailable/not_matured`，不创建策略、不建模、不入池、不采纳、不部署。下游强绑定纵切也已接通：单变量、自动树、Voting、Cross/候选 refinement、既有 candidate/tradeoff/bands/rule 工具、V2 Workflow 回测和 Pool 影响测算，都必须解析由 artifact id/hash、sample design id/hash 和 `partition=development` 组成的精确、已成熟 `StrategySampleDesignRef`，并在执行及落盘前复核 task、活动 dataset/hash、workspace revision/generation、semantic mapping、target 和空标签策略；`target_bad_value=0` 会由确定性内核统一归一为坏样本语义。旧的未绑定活动计划不能继续用于这些 V2 开发路径，必须失败关闭。底层 direct `backtest_strategy` 仍保留给既有 V1/外部调用的兼容路径，但所有 V2 策略开发 Workflow 必须解析并注入样本 ref，不得借兼容路径绕过样本门禁。风险/通过率双样本、渠道/客群纳排、历史回溯打分和泄漏/选择偏差检测仍需继续补齐；SQL connector、隔离自定义派生、风险方向、完整版本化指标语义、`CurrentProjectSnapshot` 和历史资料映射也仍在 Phase 2 内继续交付；Phase 2 尚未整体完成。
- Phase 2 的原生 approval/risk 双人群 `StrategySampleDesign V2` 纵切已完成：用户可在自然语言或 Manual 请求中分别描述通过率总体和风险总体的纳入/排除条件，明确二者是同 cohort 嵌套关系还是平行时间 cohort，并使用受限的一层条件组合或显式三分区时间窗；平台直接针对活动 DataWorkspace 生成物理 membership 和 V2 bundle，不再要求先伪造 V1 development ref。请求只能引用当前字段、目标列不能参与人群或切分条件、所有数据/workspace/语义/目标绑定在计算前后及登记锁内重验，原生 membership 路径还绑定 dataset、workspace 和目标口径，避免相同掩码在不同口径下发生 registry 冲突。旧 compatibility 产物的规范 JSON、ID 和 hash 保持不变。Agent 选择最新样本时会先认证原生 evidence；损坏或下游尚未支持的原生样本必须类型化失败，不能静默回退旧 V1 样本。当前这一步完成的是双人群样本定义与物理证据底座；仍依赖 V1 development execution binding 的单变量、树、Pool、ImpactCube、独立验证和报告消费者要逐项迁移后，才算原生双人群策略开发主链完成。上条中“风险/通过率双样本待补齐”的状态以本条为准；渠道/客群产品化纳排、历史回溯打分、泄漏/选择偏差检测仍待续。
- Phase 2/3 的首个原生样本消费纵切已完成：平台新增来源无关的风险开发执行绑定，既保持旧 V1 `partition=development` 的 ref、token 和选样行为不变，也能认证原生 V2 bundle 与 membership，并严格按持久化的 `risk/development` mask 保序取样；它不会重算人群谓词、不会把风险总体与审批总体做隐式交集，`target_bad_value=0` 仍统一归一为内部 `1=bad`。current 执行会绑定活动 DataWorkspace，historical lineage 可在 workspace head 切换后从不可变 dataset registry、文件字节、membership 与 bundle 完整重放；两者在落盘 writer lock 内分别重验。自然语言和 Manual 单变量分析已迁移到该边界，自动排除目标、分区及双人群筛选字段，并把原生五字段 ref 原样写入 evidence 与 provenance。Automatic Tree、Cross/refinement、Voting、Pool、稳定性、ImpactCube、独立验证和报告仍保持类型化阻断，必须逐项迁移，不能因单变量已接通而宣称原生双人群主链完成。
- Phase 2/3 的原生候选开发链已继续接通 refinement、Automatic Tree 与 2D Cross Matrix：单变量候选可在 workspace head 前进后仍按不可变 parent evidence、原 dataset bytes 和 historical membership 精确选择/合并；Cross 同样以 historical lineage 重放双轴并可继续物化 cell selection；Automatic Tree 作为直接开发请求保持 current workspace 强绑定，生成的 native source token 会严格配对 `strategy_sample_design_v2 + risk/development`，并可继续物化叶节点。三条链都只消费持久化 risk mask，统一归一坏样本极性，拒绝把 target、partition 或 approval/risk population 字段作为特征/轴；legacy `strategy_sample_design + development` 的 sample-context hash、token 和 canonical candidate bytes 由 golden contract 保持不变。Agent 仅在对应核心、lineage 和 manifest 全部可执行后逐 Workflow 开放，损坏的最新原生 evidence 仍不回退旧样本。Voting、交互树后续编辑、ModelEvidence、Strategy Pool、稳定性、影响、验证和报告仍需继续迁移。
- Phase 3 Candidate Lab 已完成单变量分析及候选选择/合并两个纵切：Agent 可通过自然语言对精确绑定的成熟 development 样本执行数值四类分箱和类别等值箱，确定性产出 IV/KS/AUC/WOE/Lift、件数及可用金额口径、typed DSL 条件和 `development/unvalidated` 候选证据；随后可按用户明确的 source bin id 或观测坏率门槛合并、选择候选，平台会沿同一 `StrategySampleDesignRef` 和父级 artifact lineage 逐行重放并重算指标，生成带稳定 rule/effect/asset id 的不可变 JSON 候选资产。模糊的“选最好”仍须澄清；门槛和 bin id 必须能从用户原话确定性核对，已有 bin 的选择/合并还必须引用用户实际查看的完整 candidate ID，并直接消费该不可变证据，不能重新分箱后按序号重绑。LLM 不参与指标或条件计算。
- Phase 3 的加权自动规则树、标准评分卡、Strategy Pool、显式 Voting 和 2D Cross Matrix 已完成首批 Agent 纵切：用户可以先用自然语言指定特征及受限参数，在精确绑定的成熟 development 样本上构建完整候选树，平台确定性生成全量叶节点清单、指标和不可变 full-tree asset；完整树也已支持自然语言触发的逐行 apply，生成的新数据集保持非活动的 `development / unvalidated` 派生物，不自动入 Pool、采纳或部署。用户还可用完整 tree asset ID 与 leaf ID 物化 pointer-only 叶选择，再用 selection ID、显式 Pool 类型、默认动作和命中动作把该叶加入 task-scoped `draft / unvalidated` Pool。标准评分卡纵切会从受治理的训练与逐行打分证据生成 band/cutoff 资产，用户只能从已查看的当前投影选择切点再入 Pool；原始模型、分数向量、SampleDesign V2 和 cutoff lineage 在执行及落盘时重新认证。Pool 对单变量候选、自动树叶、Cross cell group、评分卡 cutoff 和 Voting 统一重放 artifact 与同一 `StrategySampleDesignRef` lineage，使用 revision/hash 并发保护，并支持受治理的删除、动作修改、完整重排和只读编译；同树不同叶可以分别入池，同一叶不能重复。当前 Pool 的一致 apply 也已接通自然语言和 Manual typed request：用户只指定五类 Pool 类型及可选安全输出前缀，平台恢复当前非空 Pool 的 revision/hash、完整候选 lineage、活动 dataset/workspace、SampleDesign 和模型分数 requirement，在最多 1,000,000 行、500 个原始字段的全量零基行序上确定性执行 first-match，并生成保留全部原始字段的非活动派生 Parquet；六个新增字段分别记录 action、typed value、value type、rule id、entry id 和 reason code，模型分数虚拟字段只驻留内存。结果数据集和 canonical JSON evidence 在同一事务中登记，精确重试复用同一结果，写入锁内再次认证 Pool、样本、分数和源文件；该动作不修改 Pool、不创建或采纳 Strategy、不部署，也不切换 workspace。Voting 首纵切要求用户逐字点名当前 Pool 的 2 至 50 个 rule id 和唯一 n，平台只回放这些成员在同一绑定样本上的 canonical `n_of_k` 条件，输出完整命中数分布、件数及可用金额观测和不可变候选；它不会自动选择“最好规则”，也不会在同一轮入池。Voting 自动组合搜索也已接入自然语言 Agent：用户明确 K/n、目标、约束、include/exclude 和硬预算后，平台只在当前 Pool 的已启用非 Voting 规则全集上做确定性、有限组合枚举；搜索会认证 Pool、样本、数据和评分 requirement，只固化聚合 JSON，显式报告搜索空间、已评估前缀、截断状态和符合约束数量，不保存逐行矩阵，也不自动宣称冠军、构建候选、修改 Pool、采纳或部署。搜索结果的 pointer-only 构建也已接通：用户必须在后续单独一轮逐字提供完整 search_id 与已评估 combo_id；平台只从当前 Pool 和认证搜索证据恢复成员与 n，并在候选落盘事务中复核完整搜索 artifact、Pool、样本、数据及全搜索 requirement。启发式的“第一名/最好/刚才那个”、用户注入 rule/entry/n/hash，以及同轮重新搜索、入池、应用、采纳或部署都被拒绝；eligible=false 的组合仅在用户精确点名时允许构建并保留失败约束，不代表推荐。用户另行要求入池时必须明确选择保留成员作为后续回退，或由 Voting 原子替代成员；普通末尾追加和无授权全局置顶均被拒绝，编译还会阻止 first-match 遮蔽。2D Cross Matrix 首纵切要求用户明确唯一 X/Y 字段和分箱方法，先生成同一样本的单变量证据，再逐行回放两轴并固化包含空单元格在内的完整 Cartesian matrix、件数/风险及可用金额观测和独立 candidate evidence；第二纵切允许用户在后续自然语言请求中逐字指定完整 matrix asset ID 和一个或多个 cell ID，平台按源矩阵行主序物化 pointer-only cell group，再在单独一轮用 selection ID 入 Pool。同一矩阵的多个 group 必须互不重叠，整张矩阵不得直接入池，group 规则按所选 cell 的类型化 `AND` 条件做 canonical `OR`，件数及风险指标从原始主样本观测聚合并按合并后分区重算。交互树首纵切也已完成：用户可以通过自然语言或 Candidate Lab 对受认证的自动树/上一版 revision 执行明确的 `prune_subtree`，平台逐层重放同一 development 样本并产出 task-owned 不可变 revision；用户必须在后续单独一步从当前认证投影逐字选择一个 frontier 节点，物化不携带条件副本的 pointer-only singleton selection，再在另一轮明确加入 Pool。显式 OR 分组也已接通：用户可逐字选择同一认证 revision 的 2 至 50 个 frontier 节点，平台按 live frontier 顺序规范化为不复制条件或指标的 pointer-only group，运行时再从 revision 重放 canonical `OR`；`group_id` 不受理由影响，selection 审计身份绑定理由。完整 revision、模糊“最好/全部节点”、同轮物化并入池以及绕过 selection 的直接入池均被拒绝；Pool 会以稳定 semantic tree id、tree hash、revision/selection 双 lineage 和 64 MiB 聚合读取预算重新认证，同一棵语义树内 singleton/group 或 group/group 的节点集合不得重叠，不相交的节点或分组可以分别入池，理由别名不能制造重复规则。所有入池都不会连带采纳、部署或生产变更。交互树更多编辑操作、节点命中数列写回与代码、Cross 人工切点、2D/3D 自动交叉搜索、完整代码交付和候选级专项列写回仍在 V2.x Phase 3 内继续交付，不能据此宣称完整 Candidate Lab 已完成。
- Phase 3 的 2D/3D Cross 阈值规则挖掘纵切已完成：用户可通过自然语言或 Candidate Lab 明确选择 2 至 12 个字段、2D/3D 维度、每字段阈值上限、风险方向、坏率/金额约束和最多 5,000 次试验；平台在受认证的成熟 development 样本上按固定组合顺序与有限预算确定性搜索，保存完整聚合 evidence、搜索空间、已评估数、截断状态、约束结果及稳定 rule id，不保存逐行命中矩阵。搜索结果只按确定性指标排序，不声明冠军、推荐或已选择；用户必须在后续独立请求中精确点名完整 search id 与 rule id，平台重放源证据和样本后才物化不可变 Cross 候选。该候选可作为第六类来源加入既有五类 Strategy Pool，并沿 Pool apply、影响测算、独立回放和报告治理链继续执行；搜索本身不改 Pool、不采纳、不部署。七步报告会冻结并重新认证与当前 SampleDesign、dataset/workspace、target 和观测字段完全一致的最新搜索，输出开发回测候选表和约束失败明细，但不会把排序写成业务选择。
- Phase 3 的交互树手工换分裂字段纵切已完成：除剪枝和改阈值外，用户现在可通过自然语言或 Candidate Lab 从来源自动树的认证特征全集中明确选择一个不同字段，并同时给出有限阈值，对当前可见 split node 执行 `replace_split_feature`。平台不会接受“最佳字段”或自动推荐作为写操作，也不允许目标、样本分区或来源树之外的字段；Tool 会重放完整可见拓扑，重新计算目标子树全部节点/前沿的件数、风险、金额、缺失归属和方向诊断，执行最小叶与 exactly-once 守恒后发布不可变 revision，旧树、父 revision 和 sibling 分支保持不变。该 revision 可继续选前沿、入 Pool 和走下游证据链；全特征候选指标搜索与自动续建仍在 V2.x 内继续完成。
- Phase 3/6 的 Candidate Lab 工作台已接通单变量、Cross Matrix、2D/3D Cross 阈值规则搜索及精确构建、自动树、交互树修订/单节点及 OR 分组选择、评分卡 band/cutoff、Voting 自动组合搜索/精确构建、当前 Strategy Pool、五类当前 Pool 的受治理直接应用、Pool 编译/删除/动作修改/完整重排、候选逐月稳定性，以及 approval/reject Pool 的 validation/OOT 独立样本回放验证纵切：策略任务可在宽桌面工作区查看 task-owned、重新验真的投影，并直接启动对应的确定性开发动作。Pool 应用表单只能从当前非空 Pool 投影选择类型并可填写安全 ASCII 输出前缀；Pool revision/hash、样本、数据、workspace 和 requirement 继续由服务端恢复，结果是非 active 派生数据集，不修改 Pool、不采纳或部署。Pool 操作表单只使用完整、非空、未截断的当前认证投影：编译只提交类型，删除和动作修改必须明确选择 Entry，重排始终提交 1 至 200 个 Entry 的完整无重复顺序；操作 settle 后静默刷新投影，均不连带采纳、发布或部署。approval/reject 独立回放表单也改为只显示当前认证非空 Pool；唯一 Pool 自动选择，多个 Pool 必须明确选择，没有兼容 Pool 时失败关闭。Voting 搜索表单只允许从当前 Pool 投影选择已启用且非 Voting 的规则，再提交 K/n、目标、约束与硬预算；Cross 阈值规则搜索也只接受显式字段、维度、方向、约束和硬预算，构建表单只能从受认证投影明确选择 search/rule 指针；两类搜索都不能按名次、最好或冠军自动代选。交互树单节点表单只允许从当前 revision 的认证 frontier 拣选 singleton 指针；OR 分组表单只允许明确多选 2 至 50 个同 revision frontier 指针，点击或多选只负责创建 pointer-only selection，不会在同一动作中入池。独立回放表单只接受 Pool 类型和 validation/OOT 分区，Pool revision/hash、样本、数据及模型分数 requirement 全部由服务端恢复；它输出 absolute replay 证据，不计算或声称 PSI、稳定性或漂移，也不修改 Pool、采纳或部署。手工表单与自然语言 Agent 共用同一 typed request compiler、PlanValidator、Workflow 和确定性 Tool；已有候选只能从当前投影选择 candidate/feature/method/bin、交互树 revision/frontier、band/cutoff、Voting search/combo、Cross search/rule 或 Pool entry，artifact/hash、Pool revision/hash、样本和数据绑定继续由服务端恢复。投影查询在数据库层按类型计数并只读取最新窗口，使用总字节预算、来源 artifact canonical/provenance 重验和同源缓存；前端按任务 single-flight 刷新，切换任务会取消旧请求，不进入每秒轮询。该纵切不是完整 Workbench：交互树更多编辑操作、development 对 validation/OOT 的 PSI/漂移比较、代码和完整 evidence drawer 仍须在 V2.x 内完成。
- Phase 3 的候选逐月稳定性已覆盖完整单变量候选 asset，以及任一当前 Strategy Pool（五种策略类型）内由单变量、自动树叶、交互树前沿选择、Cross cell group、评分卡 cutoff 或 Voting 产生的精确 entry。用户只需用自然语言或 Candidate Lab 表单点名当前可见的 asset/entry；平台自动恢复 candidate artifact/hash、当前 Pool revision/hash、成熟 development `StrategySampleDesignRef`、活动 dataset/workspace、月份、目标和可执行 requirement。评分卡/Voting 的模型分数 requirement 会在完整零基行序上重新认证并注入内存，再切 development 样本，既不信任调用方字段，也不把逐行分数复制进稳定性产物。确定性内核以完整 development 样本的“命中/未命中”分布作为固定基线，对直接候选规则计算 rule hit，对 Pool 条目按当前完整 waterfall 计算 incremental first-match hit，逐月输出样本、命中、标签覆盖、命中坏率和 PSI，并以 30 行固定阈值标记低样本月份。读取前执行 1,000,000 行预算，最多接受 240 个月；结果固化为 task-owned canonical JSON，登记前再次复核候选、Pool、Sample Design、workspace、requirement 和数据文件，状态保持只读 `development / backtested / unvalidated`，不创建策略、不改 Pool、不采纳、不部署。当前尚未覆盖非单变量独立 asset 的直接稳定性和独立 OOT 稳定性。
- Phase 4 的当前 Pool → canonical Strategy 桥接纵切已完成：用户可以用自然语言或严格 Manual typed request 指定五类之一，把当前非空 Pool 的精确 revision、snapshot、artifact、compiled design 和 requirements 在平台侧恢复并再次认证，随后把编译得到的 `StrategySpec` 原样持久化为 root `draft / draft` Strategy。Strategy、唯一创建审计和不可变物化 ledger 在同一个 `BEGIN IMMEDIATE` 事务中提交；精确重试复用同一 Strategy，即使它后来已被人工采纳，也只返回当前 lifecycle，不把既有采纳误报为本 Tool 动作。ledger 同时绑定 Pool revision/artifact、design hash、requirements hash、Strategy semantic hash 和完整 DSL hash，公共读取会重新认证全部证据；任务级清理仍可安全级联删除业务行并保留全局 audit。该 Tool 永不采纳、部署或启动监控；含模型分数等尚无通用 Strategy runtime binding 的 requirements 会按 typed blocker 持久化但禁止宣称 DSL 交付、回测或监控 ready，无 requirements 也只表示 requirement compatibility，不替代独立证据和 lifecycle 门禁。
- Phase 4 的 approval/reject Strategy Pool 影响测算首个纵切已完成：用户可直接用自然语言要求测算当前非空 Pool，Agent 绑定 Pool revision/hash、候选 lineage、精确且成熟的 development `StrategySampleDesignRef`、活动 DataWorkspace、数据内容 hash、确认 target 和语义版本，平台按样本设计的好坏极性确定性输出 first-match waterfall（standalone/incremental/shadowed/remaining）、总体动作与风险、标签覆盖、可用件数/金额观测、可选逐月结果，以及同任务同类型 canonical baseline 的件数、风险和金额 delta，并固化由 TaskArtifact registry hash 锚定的 task-owned canonical JSON。approval/reject Pool 的 validation/OOT 独立回放纵切也已接通自然语言和 Candidate Lab：平台从当前 Pool 与成熟 `risk` 独立分区恢复全部治理绑定，确定性输出动作、风险、金额和可选逐月 absolute replay evidence；该证据与 development 影响测算不同，不计算跨分区 PSI，也不自动晋级、采纳或部署。PoolValidation、ImpactCube、PoolImpact 与全量 apply 已共用完整 Pool development binding，并会在交互树前沿、评分卡或 Voting Pool 上重新认证完整候选 lineage，按需注入模型分数 requirement；自定义 Parquet 索引先还原为受控零基行序，逐行分数仍只驻留内存。缺失的月份或金额语义保持 `unavailable`，空标签必须明确确认，Sample Design/Pool/requirement/数据/登记路径/baseline 漂移均 fail closed；测算结果保持 `development / backtested / unvalidated`，apply 结果保持非活动派生数据集，两者都不创建策略、不采纳、不部署。limit/pricing/segmentation 类型化影响口径、分群与分群×月、swap、development 对 validation/OOT 的 PSI/漂移比较和等价代码仍在 V2.x Phase 4-6 内继续交付。
- 七步 `StrategyReportBundle` 首个完整纵切已完成：Agent 可按“当前项目 → 历史版本 → 样本设计 → 单变量/模型 → 候选组合 → 影响测算 → 结论与行动”固定顺序组装 task-owned、不可变 revision；可选资料暂缺时持久化 availability 并在报告留空，不把空白写成 0，缺少会改变策略语义的关键证据仍失败关闭。报告从受认证的 project/sample/candidate/model/Pool/ImpactCube evidence 组装，并在计划阶段把与当前 Pool、SampleDesign、dataset、target 和 requirement 完全一致的最新 validation/OOT PoolValidation artifact 固化为 exact refs；执行阶段只按这些 ref 重验和读取，不因后来出现的新证据漂移。独立回放证据进入影响测算表、最终结论与 XLSX `10_validation`，但有项目成熟度或验证阻断时只展示证据并明确抑制 `oot_validated` 声明，不自动晋级、采纳或部署，也不把 absolute replay 写成 PSI。平台确定性输出 canonical JSON、Markdown、模块化 Excel 和可解析 DOCX；四种格式同 revision 原子登记、hash 绑定和审计，任一渲染或登记失败不会留下半套报告，也不会回滚已完成的策略证据。额度/定价专属报告扩展、候选/树/评分卡/Voting/Cross 的独立 OOT 稳定性章节和完整 browser 旅程仍在 V2.x 内继续完成。
- 候选逐月稳定性已接入七步报告：报告计划会从新到旧逐个认证稳定性 artifact，只自动选择与当前 Pool entry、当前 SampleDesign V2 `risk/development`、dataset/workspace、目标和月份语义完全一致的最新证据；认证失败时不会回退旧结果，没有兼容证据则把该节作为可选信息省略。报告在“候选组合与策略设计”中加入 development 基线和全月份命中/未命中、标签覆盖、命中坏率与 PSI 表，低样本月份转为 amber 提醒；独立 asset 测算的回测 stage 绑定精确候选 asset artifact，Pool-entry 测算才绑定当前 Pool revision，禁止把独立规则命中伪装成 Pool waterfall 回测。XLSX 使用独立 `appendix_candidate_stability` sheet，JSON/Markdown/DOCX 消费同一 canonical 表。稳定性文件在报告发布事务前后都再次校验 registry、双 hash、provenance 和文件字节，审计同时记录 registry hash 与领域 evidence hash。
- 当前 Pool 的跨分区分布稳定性也已接入七步报告：Agent 先冻结当前 Pool 对应的 exact ImpactCube，再在最多 64 条最新记录中逐条完整认证 PoolStability，只选择 source ref 与该 ImpactCube 四字段完全一致的最新证据；合法但无关的记录可跳过，损坏记录或无法证明窗口外不存在匹配证据时失败关闭，不回退旧结果。发布工具只接受四字段 exact ref，并在四格式原子发布事务前后同时复核 PoolStability、嵌套 ImpactCube、registry、唯一成功 audit、producer run、canonical path 和文件字节；旧计划缺少该可选字段时仍按 `None` 等价重放。报告在影响测算章节和 XLSX `10_validation` 增加最多 8 行 approval/risk × validation/OOT × waterfall/new-action PSI 摘要及分布漂移提醒，但保持 `effect_stage=None`，不改变既有 stage evidence、OOT 门禁、策略状态、采纳或部署结论。
- Voting 自动组合搜索证据已接入七步报告：报告计划只从有界的最新历史窗口中逐个完整认证 search artifact、不可变 Pool revision、SampleDesign V2、dataset、target、观测字段和模型分数 requirement，合法但不属于当前 Pool/样本的搜索会跳过，损坏证据或窗口耗尽则失败关闭，不会把未认证 provenance 当作筛选依据。精确匹配的搜索只在“候选组合与策略设计”中展示开发回测 Top 20，包含 `eligible=false` 及约束失败明细；JSON、Markdown、XLSX `appendix_voting_search` 和 DOCX 共享同一 canonical 行，DOCX 的紧凑表仍受全局事实行预算约束。报告不会据此宣称冠军、最佳、已选择、已构建或已入池，搜索证据在四种输出原子发布事务前后均再次认证并写入审计。
- Candidate Lab 的 Strategy Pool 入池表单已完成受治理纵切：单变量 refinement asset、自动树已物化叶选择、交互树已物化单节点/OR 分组、Cross 已物化 cell group、评分卡 cutoff selection 和已构建 Voting candidate 都只能从当前 task 的重新认证投影选择，前端不接受手输 ID，也不提交 artifact/hash/revision/dataset/sample 等平台绑定。已有 Pool 的完整 typed 默认动作由当前认证 Pool 投影恢复并锁定；首个 Pool 必须显式填写兼容默认动作和命中动作。Voting 还必须显式选择 `before_selected_members` 或 `replace_selected_members`，非 Voting 禁止该字段。提交前会重新核对来源成员资格和 Pool 默认动作，沿用 active-plan/open-gate/submitting 单飞门禁，成功 settle 后静默刷新；该动作只生成可逆的 `draft / unvalidated` Pool revision，不采纳、不发布、不应用、不部署。

## 术语

- **Plugin（插件）**：可安装或内置的能力包。包含 manifest、代码、权限、版本、测试、展示声明、tools 和 hooks。
- **Tool（工具）**：Plugin 内 Agent 可以主动调用的具体动作，有输入 schema、输出 schema、权限、失败策略和审计记录。
- **Hook（钩子）**：Plugin 内由平台事件自动触发的动作，例如任务扫描完成、验证完成、报告生成前后。
- **Workflow（流程）**：Agent 生成、平台内置、或用户可编写的一组 Tool/Hook 编排计划（模板），用来完成端到端任务。
- **Skill（技能）**：SOP / Playbook / 方法论型知识，落地为「用户可编写的 Workflow 模板」——声明式、只编排已信任工具、过 `PlanValidator` 校验，不直接执行 Python 代码、不另立 runtime。历史文档里的 skill runtime 是旧称；MARVIS 执行能力统一是 Plugin / Tool / Hook / Workflow。

示例：

```text
Plugin: credit_modeling
  Tool: check_data_quality
  Tool: train_lgb_model
  Tool: export_pmml
  Hook: validation.completed -> summarize_historical_comparison

Workflow: "做一个 LGB A 卡模型"
  check_data_quality -> train_lgb_model -> evaluate_model -> export_pmml -> validate_model
```

## V1.0.x：上一条稳定模型验证线

V1.0.x 是上一条稳定线，目标是让 MARVIS 第一个完整工作流可靠、可演示、可回滚。

已实现范围：

- 创建模型验证任务。
- 扫描提交的 Notebook、样本、PMML、数据字典等材料。
- 执行 Notebook，并保留 live kernel 供下游验证复用。
- 对比 Notebook 内存模型分和提交 PMML 分。
- 计算 KS、AUC、PSI、分箱、稳定性、压力测试等验证证据。
- 生成 Excel 和 Word 报告。
- 支持手动模式和 Agent P1 模式。
- 支持从本地 workspace 配置运行时 branding。

V1.0.x 只接受必要缺陷修复、兼容性修复、文档修正和发布流程修正。

## V1.1：Agent Memory Foundation

V1.1 给模型验证工作流加入长期、可审计的 Agent 记忆。它现在作为 V2 平台里的兼容基础能力继续保留，而不是当前产品中心。

参考原则：

- 参考 OpenClaw 的本地优先、可查看文件、短期工作层到长期紧凑层的蒸馏、action-sensitive memory 和人工可复核的记忆管理。
- 参考 Hermes 的用户画像 / Agent 经验分层、紧凑上下文注入、会话搜索、外部 provider 预留和写入前安全扫描。
- MARVIS 不照搬通用 agent 记忆。记忆必须适配信贷风控：确定性指标隔离、敏感材料禁存、来源 task_id 可审计、历史模型效果可比性置信度。

目标：

- 让 Agent 能跨任务记住验证、建模、数据处理、策略和分析相关经验，而不是只记住当前对话。
- 记忆只辅助解释、参数建议、风险提醒、历史对比、报告口径和后续 workflow 编排，不直接改变确定性结果。
- 所有记忆必须可查看、可禁用、可删除、可审计。
- 记忆应自然体现在 Agent 对话、阶段分析、报告草稿建议和 workflow 选择中，不新增常驻前端灰块展示匹配记忆。
- 验证完成、建模完成、JOIN 执行、策略采纳、任务失败、用户纠正和字段识别时，系统可以提取候选记忆；候选记忆必须经过分类、安全过滤、压缩和来源记录。

允许保存：

- **用户偏好**：报告措辞、解释详细程度、常用输出风格、用户明确纠正过的表达禁忌。
- **字段口径**：常见字段别名、渠道字段、时间字段、样本分组字段、目标字段和分数字段习惯。
- **验证/建模/策略坑点**：某类 Notebook、PMML、字段、执行环境、数据字典、训练配置、策略口径或报告问题的摘要和修复建议。
- **任务经验**：历史任务的非敏感摘要、失败原因、复核提醒、报告确认口径和人工复核结论摘要。
- **模型经验**：KS、AUC、PSI、月份、渠道、模型名称、模型版本、适用范围、来源 task_id、重要特征的数据源；可对比多个模型、多个版本、多个月份、多个渠道和多个指标。
- **Workflow 经验**：用户编写或系统内置 Workflow 模板的非敏感执行经验摘要。

禁止保存：

- 原始样本数据、客户明细、完整 Notebook 源码。
- PMML 文件内容、模型文件内容、API key、数据库连接。
- 未脱敏报告全文、机构敏感信息、私有 branding 内容。
- 会直接改变 KS/PSI/AUC/分数一致性等确定性指标的内容。
- 无来源、无置信度或无法审计的自动推断结论。

记忆生命周期：

- **候选提取**：从结构化工具结果、Agent 消息、任务失败原因、用户明确偏好和报告确认中生成候选记忆。
- **安全过滤**：拒绝敏感内容、过长内容、源码/数据/密钥、提示注入和无来源结论。
- **压缩保存**：保存短摘要、结构化字段、来源、置信度、创建/更新时间、禁用状态和审计事件。
- **检索使用**：Agent 在阶段分析或聊天前按任务上下文检索相关记忆，生成 bounded memory context，不把全部记忆塞进提示词。
- **审计管理**：用户可以查看、禁用、删除记忆；系统记录读、写、禁用、删除和用于回复的引用。

前端体验：

- 不新增任务顶部固定记忆区域，不用灰块列出“匹配到的记忆”。
- 对话中自然体现记忆价值，例如：“上一版分润 A 卡模型在 2026 年 2 月样本上的 KS 高于当前模型，需要关注。”
- Agent 消息可带可展开的“记忆引用”，展示来源 task_id、类别、置信度和用途。
- 记忆管理入口放在设置或审计管理视图中，用于查看、禁用、删除和导出审计，不作为每个任务的常驻内容。

## V2：当前主线 Agent 平台

V2 是当前主线。它把信贷风控任务纳入统一 Plugin / Tool / Hook / Workflow 运行时，并用受控工具、计划校验、人在环确认、审计证据和结构化产物来约束 Agent 行为。

运行时职责：

- Agent 理解用户目标，补齐任务上下文，选择可用 Plugin/Tool，生成或实例化 Workflow。
- `PlanValidator` 校验工具存在性、输入 schema、DAG、post-check、确认门、指标范围和权限边界。
- `PlanExecutor` 执行步骤、维护状态、暂停确认、重试、replan、loop events、hooks 和 evidence envelope。
- `ToolRunner` 做权限校验、参数校验、确定性 seed、子进程隔离、超时、资源限制、输出 schema 和审计记录。
- AUTO 模式只能在 gate envelope 里选择有限动作，并受 schema、confidence、预算和安全规则约束。

当前产品入口：

- **数据处理 / Data Join**：识别主表/特征表/键，诊断命中率、膨胀、去重和键格式风险，确认后执行 join 并产出拼接数据。
- **特征分析**：计算 IV/KS/AUC/PSI/coverage/lift/共线等指标，输出可下载特征分析报告；被建模或策略调用时可进入筛选确认门。
- **模型开发**：读样本、确认目标和切分、做泄漏感知筛选、调参训练、比较实验并输出模型开发报告、打分产物和交接材料。
- **模型验证**：保持 V1.1 既有手动/Agent 验证能力可用，并可通过 `v1_compat` 作为 Workflow 里的稳定工具包调用。
- **策略开发**：构造规则、回测策略，计算通过率、坏账率、swap、利润或收益权衡，关键上线类动作保留人工确认。
- **Vintage / 风险分析**：计算 vintage、roll-rate、稳定性观察和相关分析，输出可复核图表、表格和报告材料。
- **监控与组合分析**：围绕评分、策略、组合表现、迁移矩阵、Expected Loss、限额/定价和 ad-hoc slice analytics 提供工具、模板和报告能力；首屏暴露范围以当前代码和产品选择为准。

不可用或未接通的入口不得作为“可用”入口展示；如果保留占位，必须明确标记为未开放或实验中。

V2.x 完整交付范围还包括当前尚未全部接通的能力：

- **Data & Semantics Workbench**：文件与受治理 SQL 导入、数据预览/修改/派生/导出、字段语义与中英文映射、风险方向、描述统计和分析状态持久化。
- **Strategy Lab**：单规则、加权自动规则树、交互树、标准评分卡、voting/n-of-k、2D/3D 自动交叉规则和 2D cross matrix。
- **Strategy Pool / Backtest / Validation**：规则池 CRUD/order、级联 waterfall、逐月/金额/分群回测、独立验证集、多 artifact 一致 apply、PSI 和 OOT 晋级。
- **Strategy Delivery**：多 sheet Excel、Python/SQL/JSON 交付、逐行等价验证、产物列写回、策略版本和 artifact 下载。
- **持续经营闭环**：监控 scheduler、通知/重试/运行日历、challenger 跟踪、红灯到新版本、周期化组合分析和收益风险复盘。
- **组织与生产治理**：真实用户身份、RBAC、maker-checker、多级审批；分离 `draft/validated/adopted_local/retired` 策略资产状态与逐环境 deployment 状态，并提供 promotion、shadow、单环境 rollback、实时/批量评分和决策引擎对接。
- **训练与模型治理**：完整训练平台、实验追踪、模型注册、跨实例同步、模型交接和复核流。
- **扩展与执行治理**：第三方 Plugin/Workflow 安装、签名、权限和回滚治理，以及支持多用户/生产运行所需的更强隔离和 worker 边界。

“全部功能进入 V2”不代表取消人工责任：确定性指标仍由平台计算，策略采纳、生产 promotion、rollback 和高风险处置必须保留人工授权，Agent 不得无人授权地修改生产状态。

最小平台 Hook：

```text
task.created
task.scanned
notebook.completed
validation.completed
report.before_generate
report.after_generate
memory.before_save
memory.after_save
workflow.completed
feature.computed
step.completed
plan.replanned
```

## 后续 major 版本原则

当前不为 V3/V4 分配任何已确定 backlog。未来只有出现需要打破 V2 兼容边界的新产品或架构决定时，才重新定义新的 major；不得因为工程量大、需要多批次或尚未实现，就把已经确认的 V2 能力改挂到 V3/V4。

## 文档模型

- `README.md` / `README.zh-CN.md`：公开入口。
- `docs/roadmap.md`：产品阶段、术语和能力边界。
- `docs/versioning.md`：版本号、tag、发布、forward-port 规则。
- `DESIGN.md`：产品体验、信息架构、视觉和交互约束。
- `docs/notebook_contract.md`：Notebook 运行契约。
- `docs/对notebook的要求.md`：给模型开发人员看的 Notebook 提交要求。
