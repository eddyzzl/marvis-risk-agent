# V2 plans/code review - 2026-06-28

> **状态更新（2026-06-28 后续修复后）**：本文主体是较早一轮 plan/code review 快照，下面的部分 finding 已在后续修复中关闭。当前最终验证基线为 `1641 passed, 2 warnings`，strategy / vintage 已接入真实 PlanDriver，不再是 `data-coming-soon` / 501 占位路径；保留原文是为了追溯当时的计划差距与修复动机。

## 0. 审查范围

- 仓库：`/Users/eddyz/zzl/projects/ai/agent/risk_manager-v2`
- 分支：`codex/v2-plugin-tool-runtime`，当前 `HEAD=66e1c76a`，领先远端 1 个提交。
- 当前未提交修改：
  - `marvis/packs/feature/tools.py`
  - `marvis/packs/modeling/tools.py`
  - `tests/test_feature_pack.py`
  - `tests/test_modeling_pack.py`
- 对照计划：
  - `docs/plans/v2-completion-plan.md`
  - `docs/plans/v2-join-phase-spec.md`
  - `docs/plans/v2-feature-phase-spec.md`
  - `docs/plans/v2-modeling-phase-spec.md`
  - `docs/plans/v2-plan-driver-spec.md`
  - `docs/plans/v2-frontend-layout-spec.md`
  - `docs/plans/settings-ia-refactor.md`
  - `docs/roadmap.md`
- 方法：本地源码审查 + 当前 diff 审查 + 2 个只读子 agent 交叉审查 + 相关测试/静态检查。

## 1. 验证结果

本轮修复后新增验证：

- `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_feature_pack.py tests/test_modeling_pack.py tests/test_modeling_recipes.py tests/test_modeling_report.py tests/test_frontend_static_v2.py::test_modeling_create_dialog_has_algorithm_selector -q`
  - 结果：`62 passed, 2 warnings`
- `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_modeling_pack.py tests/test_modeling_artifact.py tests/test_modeling_handoff.py tests/test_modeling_recipes.py tests/test_modeling_api.py::test_modeling_persists_explicit_target_type_and_defaults_recipe tests/test_modeling_api.py::test_modeling_multiple_files_requires_data_join_first tests/test_feature_analysis_api.py::test_feature_analysis_multiple_files_requires_data_join_first tests/test_frontend_static_v2.py::test_modeling_create_dialog_has_algorithm_selector tests/test_frontend_smoke.py tests/test_frontend_v2_api_state.py::test_v2_mount_creates_stable_panels_idempotently tests/test_frontend_v2_api_state.py::test_v2_mount_registers_delegated_handlers_once_and_cleans_up tests/test_frontend_v2_api_state.py::test_v2_mount_initially_loads_governance_panels tests/test_frontend_v2_api_state.py::test_v2_mount_wires_plugin_and_skill_refresh_actions tests/test_frontend_v2_api_state.py::test_v2_mount_fetches_capability_tiers_into_panel_and_state tests/test_frontend_static_v2.py::test_system_settings_center_keeps_extensions_without_runtime_workbench -q`
  - 结果：`46 passed, 2 warnings`
- `CONDA_NO_PLUGINS=true conda run -n py_313 ruff check marvis tests`
  - 结果：通过，`All checks passed!`
- `node --check marvis/static/app.js && node --check marvis/static/js/v2/main_v2.js && git diff --check`
  - 结果：通过。
- `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest -q`
  - 结果：`1570 passed, 2 warnings in 326.87s (0:05:26)`
  - warning：`tests/test_modeling_recipes.py::test_train_mlp_writes_artifact_and_is_seed_reproducible` 中 sklearn MLP 达到 `max_iter=60` 未完全收敛；不是失败。

已完成：

- `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_feature_pack.py tests/test_modeling_pack.py tests/test_modeling_api.py tests/test_data_join_api.py tests/test_feature_analysis_api.py tests/test_orch_templates.py -q`
  - 结果：`50 passed in 138.19s`
- `CONDA_NO_PLUGINS=true conda run -n py_313 ruff check marvis tests`
  - 结果：通过，`All checks passed!`
- `node --check marvis/static/app.js && node --check marvis/static/js/v2/main_v2.js`
  - 结果：通过。
- `git diff --check`
  - 结果：通过。

- `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest -q`
  - 结果：`1560 passed, 2 warnings in 321.44s (0:05:21)`
  - warning：`tests/test_modeling_recipes.py::test_train_mlp_writes_artifact_and_is_seed_reproducible` 中 sklearn MLP 达到 `max_iter=60` 未完全收敛；不是失败。

## 2. 总体结论

本轮已处理：

- P1-1/P2-2/P3-1：非二分类 `screen_features` 抽到共享 helper，按 dev rows 排除 OOT/holdout，`ranked` 保留全部 clean features，feature/modeling 两个 pack 共用实现并补 top_k/OOT 测试。
- P1-3：新增任务级 `target_type` 字段，从 API/schema/DB 到 modeling setup 全链路持久化；服务端拒绝跨目标类型 recipe；前端按算法 family 互斥并随创建任务提交 `payload.target_type`。
- P1-4/G4：`train_models` 的 best selection 和候选表改成 target-type-aware；`compare_experiments` 增加训练后动作能力矩阵，明确 PMML/验证移交当前只支持 LR，其它模型保留原生模型/报告。
- P1-5：manifest 加入 `mlp`，`load_model()` 支持 MLP joblib artifact。
- P2-1/P3-2：原审查时 strategy/vintage 被改成 unavailable；后续修复已进一步接入真实 PlanDriver，欢迎卡恢复为可用入口并移除 `data-coming-soon` 残留。
- P2-4：非二分类模型报告保留固定 workbook skeleton，binary-only sheet 标 `n/a/非二分类不适用`，额外写目标类型指标 sheet。
- P2-5：设置弹窗退役旧 `计划与执行` runtime workbench 挂载；`mountV2` 只保留插件、Workflow 模板、能力档位三个设置型面板，改用 `governanceExtensionMount`。
- P2-6：`make_split` 样本分析只读取月/渠道等必要分组列，不再整表读入。
- P1-2/P2-3：在 PlanDriver 尚无条件 step 语义前，Feature/Modeling 遇到多个数据文件改为 fail-fast，明确要求先走数据拼接并使用拼接结果，避免静默选单表训练/分析。

当前后续修复后的代码比上一轮明显更接近计划：`data_join / feature_analysis / modeling / strategy / vintage` 已经走通用 `PlanDriver`，JOIN 的 C1/C2、强确认、去重策略和 1:1 保护基本成型；manual 模式下的 feature 勾选、join 去重、步骤栏确认、strategy 回测确认、vintage 曲线也已经接上后端。

但当前仍不能按 `docs/roadmap.md` 宣称完整 V2。它更准确的状态是：5 个入口切片可用，JOIN/PlanDriver 底座较稳；Feature/Modeling 的多表自动组合仍需要 PlanDriver 支持条件 step 后再做完整 JOIN→FEATURE→MODELING 编排，本轮已改成 fail-fast 防止错误使用单表。

这次用户修改主要集中在“非二分类 target 时的 screen_features 不再保留常量/全缺失列”。方向正确；本轮已补齐 dev-row/OOT 排除和 `ranked`/`top_k` 契约。

## 3. 已经对齐或不再作为问题的部分

### 3.1 JOIN 和通用 PlanDriver 基本对齐

- `data_join` 模板已有 `propose_join -> confirm_join -> execute_join`，`execute_join` 是 `needs_confirmation=True`，并且引擎层还有未确认不执行的兜底。
- `PlanDriver` 已支持计划级确认、步骤门、自由文本 adjust/replan、manual 控件提交、screen selection、dedup strategies。
- `marvis/agent/plan_driver.py:371-416` 会把 `screen` 和 `dedup` 放入 gate message metadata，前端 `marvis/static/app.js:6843-7047` 已接入“确认所选特征”“应用去重并确认”“确认继续”。

处理建议：保留这条路径，不再重复做一套任务专用 driver；后续缺口应往模板/阶段组合/工具能力里补。

### 3.2 Strategy/Vintage 后续已接入 PlanDriver

- `marvis/static/index.html` 中 strategy/vintage 卡片为 `welcome-task-card available`，不再有 `data-coming-soon`。
- `marvis/static/app.js` 的 plan rail task types 包含 `strategy` / `vintage`。
- 后端 agent allowlist 已包含 `strategy` / `vintage`，真实 driver turn 与 manual/agent-mode 测试均已覆盖。

处理建议：保留后端 allowlist 作为最终防线；对外口径可以说 strategy/vintage 已有 MVP 工作流，但滚动率、FPD、入催回收率等更完整风险分析仍属于后续扩展。

## 4. P1 / 高优先级问题

以下 P1/P2/P3 条目保留为原始 code review 发现记录；本轮修复后的实际状态以上文“本轮已处理”和最终验证结果为准。

### P1-1. 非二分类筛选没有排除 OOT/holdout，OOT 会影响特征可用性判断

计划/既有契约：

- 二分类 screen 明确通过 `split_col + holdout_values` 排除 holdout 行：`marvis/feature/screen.py:91-130`。
- `MODELING` 模板会把 `split_col` 和 `holdout_values` 传给 `screen_features`：`marvis/orchestrator/templates/sample.py:321-331`。

当前实现：

- `marvis/packs/feature/tools.py:142-188` 的 `_screen_features_non_binary()` 只读取 feature columns。
- `marvis/packs/modeling/tools.py:224-267` 的 `_screen_features_non_binary()` 同样只读取 feature columns。
- 两处都用全量数据计算 `missing_rate` / `unique_count`，没有按 split 排除 OOT。

影响：

- 一个特征在 train/test 可用，但在 OOT 全缺失或常量，会被错误剔除。
- 反过来，一个特征只有 OOT 有变化，也可能被错误保留。
- 这把 OOT 分布带进了筛选门，和“训练内筛选、OOT 只做无偏评估”的建模不变量冲突。

处理方法：

1. 抽一个共享 helper，例如 `_non_binary_screen_stats(runtime, dataset_id, features, target_col, split_col, holdout_values, max_missing_rate)`。
2. 读取 `split_col` 时复用 `marvis.feature.screen._dev_mask()` 或等价逻辑：默认 holdout 为 `("oot",)`，只在 dev rows 上算缺失率和唯一值。
3. 在 `tests/test_feature_pack.py` 和 `tests/test_modeling_pack.py` 增加用例：train/test 非缺失且有多值，OOT 全缺失，期望仍 selected。
4. 两个 pack 的实现保持一致，避免 feature/modeling 分叉。

### P1-2. Phase composition 没有实现，Feature/Modeling 还不是计划里的组合模块

计划要求：

- `docs/plans/v2-completion-plan.md:23`：`JOIN / FEATURE / MODELING` 是可组合阶段模块；`feature=[JOIN,FEATURE]`，`modeling=[JOIN,FEATURE,MODELING]`。
- `docs/plans/v2-modeling-phase-spec.md:3-5`：模型开发应支持“甩几个 csv 进来即可拼、析、筛、训”。

当前实现：

- `marvis/agent/feature_setup.py:59-72` 只从 task datasets 里挑一个目标数据集。
- `marvis/agent/modeling_setup.py:205-217` 也只挑一个数据集。
- `marvis/orchestrator/templates/sample.py:275-415` 的 `MODELING` 从 `make_split` 开始，不包含 JOIN phase。
- `marvis/orchestrator/templates/sample.py:420-462` 的 `FEATURE_ANALYSIS` 是单表指标 + 报告，不包含 JOIN phase。

影响：

- 用户上传多个 csv 时，Feature/Modeling 不会进入 JOIN，和 “甩 csv 出模型” 目标不一致。
- JOIN 的改进不能自动惠及上层任务，违背“模块一处改进惠及上层”的计划。

处理方法：

1. 明确阶段编排入口：在 setup 阶段判断 `source_dir`/registry 是否为多表。
2. 多表时先实例化 JOIN phase，产出 `joined_dataset_id`，再传给 FEATURE 或 MODELING。
3. 单表且已满足样本条件时，显式生成一个 skipped JOIN phase/消息，避免静默跳过。
4. 在 API 层增加多文件 Feature/Modeling 的集成测试：上传 sample + feature table，期望 plan 里出现 JOIN phase，后续用 join result 训练。

### P1-3. Modeling G2 算法/任务类型门没有按计划实现，UI 和后端都可能产生混合目标类型

计划要求：

- `docs/plans/v2-modeling-phase-spec.md:44-49`：任务类型必须单选，算法按类型选择；进入任务后校验 y 是否符合目标类型；G2 是用户确认/修改的门。

当前实现：

- UI 用 checkbox 同时展示二分类、MLP、回归、多分类：`marvis/static/index.html:211-246`。
- 文案提示“请单选其一、勿混选”，但控件本身没有强约束。
- `marvis/agent/modeling_setup.py:140-157` 只拒绝“回归 + 多分类”混用；`binary + regression` 或 `binary + multiclass` 会被归成 continuous/multiclass，二分类 recipe 仍可能混进去。
- `marvis/agent/modeling_setup.py:87-120` 通过 recipe 推导 target_type，不是由目标类型单选驱动，也没有独立 G2 确认步骤。

影响：

- 用户可能勾选 `lgb + lgb_regressor`，target_type 被推为 continuous，但训练 recipes 仍包含 binary `lgb`。
- 多算法比较、指标渲染、报告和 export 都会在这种混合目标形态下变得不可预测。

处理方法：

1. UI 改为 `target_type` radio：`binary / continuous / multiclass`。
2. 算法选择按 target_type 动态过滤；二分类组和非二分类组不可混选。
3. 后端 `TaskCreate`/setup 保存并校验 `target_type`，不要只从 recipe 推导。
4. 在 `MODELING` 模板里加入 G2 gate，显示目标类型检测结果、可选算法、用户确认后的 recipes。
5. 服务端拒绝任何跨 target_type recipe family 的组合，并补 API 测试。

### P1-4. Modeling G4/G5 还没有达到计划：选模、导出、移交都是部分实现

计划要求：

- `docs/plans/v2-modeling-phase-spec.md:57-63`：G4 要让用户在 trials/候选模型排行中选择模型，可多版出报告。
- `docs/plans/v2-modeling-phase-spec.md:78-80`：G5 要提供 `compare_experiments / export_pmml / handoff_to_validation` 动作。
- `docs/plans/v2-modeling-phase-spec.md:102-103`：`lr/scorecard/lgb/xgb` 都应导 PMML，DNN 走原生模型和打分脚本。

当前实现：

- `marvis/packs/modeling/tools.py:411-422` 的 `_pick_best_experiment()` 只按 `oot_ks/test_ks` 自动选 best。
- `marvis/agent/plan_driver.py:691-721` 的候选模型表只展示 KS/AUC，并写死“按 OOT KS”。
- `marvis/orchestrator/templates/sample.py:394-413` 的最后一个 gate 是“生成模型开发报告”，模板没有 export/handoff 动作 step。
- `marvis/packs/modeling/artifact.py:78-86` 的 `export_pmml()` 只支持 `lr`。
- `marvis/packs/modeling/handoff.py:38-41` 的 validation handoff 只允许 `lr`。

影响：

- 用户训练出更好的 LGB/XGB/scorecard 后，导出或移交验证会硬失败。
- 回归/多分类的 best model 会因为 KS 为空而退化成“第一个”，不是按 RMSE/macro-AUC/logloss 选择。
- “用户拍板选模”没有真正落地。

处理方法：

1. `_pick_best_experiment()` 改成 target_type-aware：
   - binary：最大化 OOT/test KS；
   - continuous：最小化 OOT/test RMSE；
   - multiclass：优先最大化 OOT/test macro-AUC，或最小化 logloss。
2. `PlanDriver` 渲染候选模型表时按 target_type 切列，不要所有模型都显示 KS/AUC。
3. 增加 G4 选模 gate，允许用户覆盖 auto best，并把选中的 experiment_id 传给报告步骤。
4. G5 先做能力矩阵：不支持 PMML 的 recipe 在 UI 上禁用/说明；支持的 recipe 补实现和测试。
5. 如果短期不实现树模型 PMML，文档和 UI 必须明确“当前仅 LR 支持”，不能和计划声明冲突。

### P1-5. MLP 在 UI/setup 中可选，但 manifest schema 拒绝，实际会在工具运行前失败

当前实现：

- UI 有 MLP checkbox：`marvis/static/index.html:231-234`。
- setup 支持 `mlp`：`marvis/agent/modeling_setup.py:64-67`。
- `_train_recipe()` 也支持 `mlp`：`marvis/packs/modeling/tools.py:635-657`。
- 但 `train_model.recipe` schema 不包含 `mlp`：`marvis/packs/modeling/manifest.json:583-592`。
- `train_models.recipes.items.enum` 也不包含 `mlp`：`marvis/packs/modeling/manifest.json:695-706`。
- `ToolRunner.invoke()` 会先校验 schema：`marvis/plugins/runner.py:71-74`。

影响：

- 用户选择 MLP 后，plan 会到 `train_models` 前被 schema validation 拒绝，训练函数根本不会运行。
- 这是一个真正的用户路径 bug。

处理方法：

1. 在 manifest 的 `train_model` / `train_models` enum 中加入 `mlp`。
2. 补 MLP artifact load/report 支持；如果只生成 minimal report，也要在输出中明确。
3. 增加端到端测试：创建 modeling task，recipes 包含 `mlp`，至少确认 schema validation 不阻断并能生成可下载报告。
4. 如果 MLP 仍未达到可用标准，先从 UI/setup 可选项移除或标灰。

## 5. P2 / 中优先级问题

### P2-1. V2 完成口径仍冲突：当前是 3 入口切片，不是 roadmap 的 V2 complete

计划/roadmap：

- `docs/roadmap.md:130-160` 明确：欢迎页露出的每个入口都必须是真实闭环，不可用入口不得作为可用入口展示。
- `docs/plans/v2-completion-plan.md:97-101` 同时说本轮做 data_join / feature_analysis / modeling，strategy/vintage 不开发，欢迎页入口保留。

当前实现：

- `marvis/api.py:3930-3953` 只 wired `validation/modeling/data_join/feature_analysis`。
- `marvis/api.py:3956-3964` 对 strategy/vintage 返回 501。
- strategy/vintage 卡片仍有 `welcome-task-card available`：`marvis/static/index.html:985-990`、`marvis/static/index.html:1081-1086`，但已通过 `data-coming-soon` 点击阻断。

影响：

- 如果对外宣称 V2 complete，会和 roadmap 完成标准不一致。
- 用户视觉上仍可能把 strategy/vintage 理解为“可用入口”，因为样式是 available。

处理方法：

1. 文档里把当前状态命名为 “V2 三入口切片” 或 “V2 platform slice”，不要写 complete。
2. 未开发卡片从 `available` 改为明确 disabled/coming-soon 样式和 aria-disabled。
3. 所有 task type availability 集中到 `taskTypeDefinitions` 或统一配置，不要由 HTML class、dataset、backend allowlist 三处各管一半。

### P2-2. `ranked` 在非二分类 `top_k` 下被截短，用户无法复核 clean-but-not-selected 特征

既有契约：

- 二分类 screen 中 `ranked` 是所有 clean 特征，`selected` 才受 `top_k` 影响：`marvis/feature/screen.py:128-130`。

当前实现：

- `marvis/packs/feature/tools.py:177-182` 先截短 `selected`，再用 `selected` 构造 `ranked`。
- `marvis/packs/modeling/tools.py:256-260` 同样如此。
- 当前新增测试只断言 `selected` 长度：`tests/test_feature_pack.py:424-437`，没有断言 `ranked` 保留完整 clean list。

影响：

- 手动筛选门无法显示 top_k 之外的 clean 特征，用户也不能加回。
- 和计划里“阈值与取舍交还用户”的 FEATURE 哲学冲突。

处理方法：

1. 在截断前保存 `clean = list(selected)`。
2. 返回 `ranked = [[feature, None] for feature in clean]`。
3. `selected = clean[:top_k]`。
4. 增加 `top_k=1` 测试，期望 `selected` 为 1 个，但 `ranked` 仍包含全部 clean 特征。
5. 前端筛选表也需要考虑渲染 ranked clean rows，而不仅是 selected/leakage/suspected/unusable。

### P2-3. Standalone FEATURE 可用，但“可复用 FEATURE 筛选模块”还不完整

计划要求：

- `docs/plans/v2-feature-phase-spec.md:17-23`：独立特征分析是报告模式；被 Modeling/Strategy 调用时才进入筛选门并交 selected features。
- `docs/plans/v2-feature-phase-spec.md:61-80`：筛选门需要宽表、阈值/勾选、硬剔原因、最终 selected artifact。

当前实现：

- 独立 `feature_analysis` 模板只做 `compute_feature_metrics -> generate_feature_report`：`marvis/orchestrator/templates/sample.py:420-462`。
- `screen_features` 存在，但没有作为独立可复用 FEATURE phase 被上层组合；Modeling 用的是 modeling pack 自己的 `screen_features`。
- `PlanDriver` 和前端已有 screen metadata/勾选提交能力，这是可复用 FEATURE 的基础。

影响：

- Modeling/Strategy 不能共享同一个 FEATURE phase 的指标宽表、筛选门、artifact。
- 计划中的“FEATURE 一处改进惠及上层”没有完全成立。

处理方法：

1. 抽出真正的 FEATURE module template：metrics + optional correlation/importance/lift + screen gate + selected artifact。
2. standalone feature_analysis 使用 report-only 版本；被 modeling/strategy 调用时使用 screen version。
3. 让 Modeling 的 screen step 复用 feature pack 输出或至少共享同一 helper，减少两套逻辑漂移。

### P2-4. 非二分类报告不符合“固定 workbook skeleton，缺项 n/a”的规格

计划要求：

- `docs/plans/v2-modeling-phase-spec.md:64-76`：详细报告固定 sheet 可增不可删，缺数据标 n/a。
- `docs/plans/v2-modeling-phase-spec.md:102-105`：多分类报告本版 binary-only sheet 标 n/a，不是删掉。

当前实现：

- `marvis/packs/modeling/tools.py:495-509` 非 binary target 直接走 `render_minimal_model_report()`，只返回“汇总/模型指标”两节。

影响：

- 回归/多分类报告结构和计划、验证报告下载心智不一致。
- 下游如果按固定 sheet 消费，会缺 sheet。

处理方法：

1. `render_minimal_model_report()` 改为固定 workbook skeleton。
2. binary-only sheet 保留，内容写明确的 `n/a` / `非二分类不适用`。
3. 回归补 RMSE/MAE/R2 和残差类 sheet；多分类补 macro-AUC/logloss/混淆矩阵。
4. 增加 workbook sheet 名测试，避免以后又被简化掉。

### P2-5. Settings IA 是“功能隐藏”，不是“旧 workbench 退役”

计划要求：

- `docs/plans/settings-ia-refactor.md:6-13`、`docs/plans/settings-ia-refactor.md:107-114`：旧 V2 计划与执行/运行审计 workbench 退役；设置只保留设置/扩展/记忆等。

当前实现：

- `marvis/static/index.html:553-561` 仍有 `governance-runtime-panel`、标题 `计划与执行` 和 `#v2RuntimeMount`。
- `marvis/static/js/v2/main_v2.js:15-27` 仍创建 `goalPanel/planPanel/joinPanel/subAgentPanel/loopPanel/artifactPanel` 等旧 runtime 面板。
- `marvis/static/css/v2-workbench.css:144-172` 只是隐藏 `.v2-panel`，只在 plugins/workflows/capabilities 三个 view 显示对应 panel。

影响：

- DOM/JS 复杂度和历史入口仍在，后续维护容易误用旧 goal composer / plan panel。
- “设置弹窗不装运行工作台”的 IA 决议没有从代码层完成。

处理方法：

1. 删除或拆分 `mountV2()` 的旧运行面板定义，仅保留插件/模板/能力/记忆/草稿等设置相关面板。
2. 把 `#v2RuntimeMount` 重命名为扩展设置根，例如 `#governanceExtensionMount`。
3. 删除 `goalPanel/planPanel/joinPanel/subAgentPanel/loopPanel/artifactPanel` 的 settings 挂载路径；如右栏仍需 `plan_view`，只在任务工作区单独挂载。
4. 清掉 `showV2WorkspaceDialog/openV2WorkspaceWithGoal/seedV2GoalComposer` 等无用入口。

### P2-6. `tool_make_split` 会整表读入做样本分析，宽表/大表上有内存风险

当前实现：

- `marvis/packs/modeling/tools.py:126-131` 在 `make_split` 后读取原始 dataset 的整张表，只为了做 split sample analysis。

影响：

- 对 20 万行、几千列的建模样本，G1 切分门会重复读整表，和 V2 大表可用性目标冲突。
- 这个步骤发生在建模最前面，容易让用户误以为整个 Modeling 不可用。

处理方法：

1. 只读取 sample analysis 需要的列：target、split、时间/月、渠道、可能的 group/id 列。
2. `_GROUP_COLUMN_HINTS` 命中的列先从 schema/profile 判断，再按需读列。
3. 对超宽表增加列数/行数采样上限，并在 gate message 中说明“样本分析基于抽样/必要列”。

## 6. P3 / 清理和优化

### P3-1. 当前 tests 对非二分类 screen 的负向场景不够

已有测试：

- `tests/test_feature_pack.py:391-437` 覆盖 constant/all-missing 和 `selected` top_k。
- `tests/test_modeling_pack.py:465` 也覆盖 modeling pack 的 constant/all-missing。

缺少测试：

- OOT 全缺失但 train/test 可用，不应被剔除。
- `ranked` 在 `top_k` 下保留完整 clean list。
- feature pack 与 modeling pack 输出一致。

处理方法：

1. 增加 3 个单测，先在 feature pack 上覆盖，再在 modeling pack 上覆盖。
2. 最好把非二分类 screen helper 抽到共享模块后，只留薄包装测试，减少重复。

### P3-2. 可用性定义分散在 HTML class、dataset、JS task definitions、backend allowlist

当前实现：

- HTML card 用 `available` 和 `data-coming-soon`。
- JS `taskTypeDefinitions` 写业务文案和默认 run mode。
- 后端 `_WIRED_AGENT_TASK_TYPES` 决定是否 501。

影响：

- Strategy/Vintage 这类“展示但未接入”的状态容易再次漂移。
- 测试也容易只断言其中一层。

处理方法：

1. 建一个前端 task availability map：`enabled / comingSoon / hidden / backendWired`。
2. 从同一份定义渲染 card 样式、点击行为、创建弹窗可用性。
3. 后端 allowlist 保持最终防线；前端测试和 API 测试分别覆盖。

### P3-3. 子 agent 审查结果需要沉淀成回归测试清单

本次两个子 agent 各自指出的高价值问题：

- 非二分类 screen 的 holdout 排除和 ranked/top_k 契约。
- phase composition 和 G2/G4/G5 的 plan 差异。
- settings IA 的隐藏式残留。

处理方法：

1. 把每个 P1/P2 转成一个 failing test 或 UI snapshot 检查，再修实现。
2. 修复顺序建议先做“当前改动相关 bug”，再做“用户入口可见的 plan gap”。

## 7. 建议修复顺序

1. 修 `feature/tools.py` 和 `modeling/tools.py` 的非二分类 screen：
   - dev mask 排除 OOT；
   - ranked 保留完整 clean list；
   - 增加缺失测试。
2. 修 Modeling 算法/目标类型：
   - `target_type` 单选；
   - 服务端拒绝跨目标类型 recipe；
   - manifest 加/删 `mlp`，保证 UI 和 schema 一致。
3. 明确当前 release 口径：
   - 若只交付三入口切片，改 roadmap/README/界面文案；
   - strategy/vintage 改 disabled 样式。
4. 做 phase composition：
   - 多文件 Feature/Modeling 先 JOIN；
   - 单表则显式 skip JOIN。
5. 补 Modeling G4/G5：
   - target-type-aware best model；
   - 用户选模 gate；
   - export/handoff 能力矩阵或完整实现。
6. 清 settings IA：
   - 移除旧 runtime workbench 残留面板和挂载根。

## 8. 本次审查不建议立即删除的内容

- 不建议删除 `PlanDriver`、JOIN 的 C1/C2 或 manual gate 控件；这些是当前最接近计划的部分。
- 不建议为了让测试过而移除非二分类 Modeling；应该先把 schema、best selection、报告和导出能力按支持矩阵收紧。
- 不建议把 Strategy/Vintage 后端 501 移除；在未实现闭环前，后端 allowlist 是正确防线。
