# 模型开发 Agent 改进路线图

> 目标(/goal）：把 MARVIS「模型开发」agent 打磨到能**通过对话**跑通交互式建模全流程（点欢迎页"模型开发"→ 读样本 → 和用户确定并切分 train/test/oot → 确认特征集 → 筛选 → 调参 → 训练），并产出**高质量、稳定、可上线、可解释的模型**。哪里做不到/做不好就改 agent，持续优化。**不设针对某个指标的固定数值目标**——以模型质量、稳定性、可解释性综合判断，参考模型仅作为可行性 sanity 参照。

## 参考数据与可行性（已确认）
- 参考：`/Users/eddyz/Downloads/11_分润通用A卡_mob3_v202604/`，LightGBM，目标 `long_y`，209245×4865，`model_flag`=train/test/oot。
- 已用 lightgbm 4.6.0 在参考数据上复现出参考模型，作为可行性 sanity 参照；**不把任何具体指标数值设为固定目标/验收线**。
- 可行性：干净特征集（排泄漏）+ 粗调即可达到优于参考的水平（有过拟合，调参后更稳）。

## 现状诊断（已用真实工具跑过）
1. **对话 agent 错位**：点"模型开发"进的 `/agent/messages` 是 V1**验证** agent，零建模工具、从不建/跑 Plan。
2. **能建模的是非交互编排器**（`/plans` + STANDARD_MODELING 固定 DAG，后台跑）——上一轮 IA 重构已退役其 UI 入口；且不对话、用户无法中途确认切分/特征。
3. `train_lgb` 默认只 20 棵树/单线程/无真参数 → 开箱即过拟合、效果偏弱；传强参数才能达到参考水平。**无任何调参工具**。
4. `select_features` IV 地板 0.02 太狠，把参考 40 特征砍到 23（树模型不该按单变量 IV 砍）。
5. **无规模化筛选 / 无泄漏检测**：4865 列盲选会命中 `max_overdue_his`（单变量 KS 极高、疑似泄漏）和模型输出列(predprob/pred_pmml) → 灾难性泄漏。
6. **无"源目录→dataset"接线**；唯一入口是上传整文件进内存(峰值~17GB)，且自动选错目标列(mob2_max_overdue 而非 long_y)。

## 路线（分层，先工具后对话）

### Phase 1 — 建模工具层（可离线验证，地基）
- **1a 泄漏感知 + 规模化特征筛选**（新工具 `screen_features`）：对上千列做单变量 KS/IV（分批，内存可控），并**标记疑似泄漏**（单变量 KS≥阈值、模型输出/分数列、与目标近重复），产出"干净候选集 + 泄漏告警"供用户确认。← 支撑"读样本/确认特征集/筛选"。
- **1b 调参能力**（新工具 `tune_hyperparameters` 或训练内置搜索）：随机/贝叶斯小搜索 + 早停，按 test/oot KS 选最优，控过拟合。← 用户的"调参"。
- **1c 强化 `train_lgb`**：真实默认参数、多线程、默认早停、与其它 recipe 一致走 NaN 标签门。
- **1d 改进 `select_features`**：放宽/可配 IV 地板、加基于模型重要性的选择、保留树模型有价值的低-IV 特征。
- **1e 样本摄入**：源目录→dataset 免整文件进内存（pyarrow 转 parquet）、允许显式指定 target_col、不靠错误的自动目标检测。
- **验收**：pack 工具链在参考数据上跑通并产出**优于参考、过拟合可控的模型**（不设固定指标阈值）。

### Phase 2 — 交互式对话建模 agent（需 LLM）
- 让"模型开发"任务的对话真正驱动建模工具，并在**切分 / 特征集 / 最终模型**三处把决定交还用户确认（"propose → 等用户 yes/no → 继续"）。
- 读样本→提议 target/split/候选特征（带泄漏告警）→ 用户确认 → 筛选 → 调参 → 训练 → 报告 KS。
- 需要：把对话 agent（或新建模 controller）接到建模工具 + 人在环确认机制；配置 LLM。

### Phase 3 — 打磨与迭代
- 健壮性、报告、PMML 导出、与验证 agent 的衔接；持续按"用→找问题→改"迭代。

## 当前进度
- ✅ 参考模型复现、数据/泄漏摸清、可行性确认、pack 工具真实跑过、缺口量化。
- ✅ **Phase 1a `screen_features`**(泄漏感知规模化筛选):模块+入口+manifest,真实数据验证(4807 列 32s,max_overdue_his 进硬泄漏,19 疑似输出列标记)。
- ✅ **Phase 1b `tune_hyperparameters`**(随机搜索,按 test KS 选优+过拟合惩罚,OOT 留作无偏判定):模块+入口+manifest。
- ✅ **里程碑:pack 链路 `screen→tune` 已产出优于参考、过拟合可控的稳健模型。** "能开发出更好的模型"已通过工具链证明。
- ⏭ Phase 1 剩余(较低优先,agent 现可传 tuned params):1c 强化 train_lgb 默认、1d 放宽 select_features、1e 样本摄入修复。
- ⏭ **Phase 2(核心剩余):造交互式对话建模 agent** —— 见下,需配 LLM。

## Phase 2 已完成（对话式建模 agent 已接入）
- ✅ **建模会话控制器** `marvis/packs/modeling/session.py`：确定性编排 `读样本→(确认切分)→screen(确认特征)→tune→train→report`，每步返回 `SessionStep(message, awaiting, data)`，可离线端到端验证。参考数据验证：端到端产出优于参考的模型。
- ✅ **对话驱动器** `marvis/agent/modeling_agent.py`：把控制器包成跨轮、状态从消息元数据重建（HTTP 处理器无状态）、意图解析两道确认门（切分/特征）、消息组织。**LLM 可选**——每步都有完整中文 canned 消息，`polish` 回调仅做润色且有兜底，故**无 LLM 也能跑**。自动探测 target/split 用随机抽样+整列分类（样本常按 split 排序，头部抽样会漏 oot——已修）。
- ✅ **API 接线** `marvis/api.py`：`task_type=='modeling'` 在 `/agent/start` 与 `/agent/messages` 顶部分流到 `_dispatch_modeling_agent_job` → 后台 job `_run_modeling_agent_job`（镜像验证 agent 的 job/状态/落消息机制）。样本发现 `_modeling_dataset`（已注册样本优先，否则 scan source_dir→register_from_upload 归一化为 parquet）。`_resolve_agent_model_optional` 让建模路径**不因缺 LLM 而 409**。
- ✅ **前端** `app.js`：建模任务发送时**放行 LLM 可用性门**（建模 agent 不需 LLM）；更新建模 `initialGoal` 为交互式开场文案。
- ✅ **测试**：`test_modeling_session.py`、`test_modeling_agent.py`（逐轮+状态重建）、`test_modeling_agent_api.py`（真实 HTTP `/agent/start`+`/agent/messages` 端到端，无 LLM 配置场景下走 canned 消息成功产出 OOT KS）；manifest 工具集断言已加 screen/tune。共 103 项相关测试全绿。
- ⏭ 剩余（增强项，非阻塞）：LLM `polish` 接线（有 LLM 时润色对话措辞）；基准 KS 经 `report_values.baseline_oot_ks` 传入以在报告里做对比；1c/1d/1e（train_lgb 默认、select_features 地板、样本摄入）。
- ⚠️ 预存测试债（本会话早期 IA 重构遗留，与本阶段无关）：`test_frontend_static_v2.py` 中 6 项断言旧 UI 结构（`data-agent-memory-mode="raw"`、`openV2WorkspaceWithGoal`、执行环境面板、品牌色按钮等）需按 IA 重构后的新结构更新。

## Phase 2 原设计（对话式建模 agent）〔历史/已被上方 as-built 取代〕
> ⚠️ 本节是**实现前的原设计**，仅作历史记录。其中"**硬依赖** LLM"已被上方 as-built（[本文件 Phase 2 已完成](#phase-2-已完成对话式建模-agent-已接入)）推翻——实际 `modeling_agent.py` 是 **LLM 可选**（每步有 canned 中文兜底，无 LLM 也能跑）。如有冲突，**以上方 as-built 为准**。
现状:点"模型开发"→对话进的是 V1 验证 agent(agent/service.py 的 scan/metrics/word 阶段机)。要造一个**建模阶段机**镜像它,但阶段为:
`read_sample`(读样本+profile,提议 target/split/候选)→ **await 用户确认切分** → `propose_features`(screen_features,带泄漏告警)→ **await 用户确认特征集** → `select`(可选)→ `tune` → `train` → `report`(KS,对比基准)。
- 阶段逻辑 + 工具调用**确定性**,可离线测;LLM 只负责把每阶段结果组织成对话回复(像验证 agent 一样有 canned fallback)。
- 入口:`task_type='modeling'` 的对话路由到建模阶段机(而非验证)。
- **硬依赖**:对话回复需配 LLM(`<workspace>/settings/llm.json`);预览 workspace 当前未配——需用户配置一个 OpenAI 兼容模型才能真正"对话"。工具层与阶段逻辑可无 LLM 验证。
