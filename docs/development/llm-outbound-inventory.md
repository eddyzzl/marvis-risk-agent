# LLM 出站调用清单

本文列出 `marvis/` 里每一处模型调用（`.complete(`）和它使用的出站适配器。
`tests/test_llm_outbound_inventory.py` 会扫描代码，新增、移动或删除调用点而本表没有同步时测试失败。

## 规则

- 发请求前由 `marvis/llm_client.py` 的 `OpenAICompatibleLLMClient.complete` 强制检查：
  - `user_prompt` 必须是 `StructuralPrompt`；
  - `StructuralPrompt` 只能由 `marvis/llm_outbound*.py` 的适配器（内部调用 `_mint_prompt`）生成；
  - 普通字符串在发请求前被拒绝，不产生任何网络请求。
- 新增调用点的步骤：
  1. 用已有适配器生成提示，或在 `marvis/llm_outbound*.py` 里新增适配器；不要在调用方直接调用 `_mint_prompt`。
  2. 给调用传一个固定的 `caller=`，便于审计和计费归类。
  3. 在下表补一行：文件、所在函数、`caller`、适配器。
- 测试只核对“文件 + 所在函数 + 调用次数”。适配器一列可以写适配器名，也可以写 mint 函数，改适配器不需要改测试。
- 同一函数里有几处调用，就写几行。

## 调用清单

| 文件 | 所在函数 | caller | 出站适配器（mint 函数） |
| --- | --- | --- | --- |
| `marvis/agent/adhoc_analysis.py` | `_invoke_llm` | 由参数传入 | `llm_outbound_analysis.slice_question_prompt` |
| `marvis/agent/auto_drive.py` | `decide_gate` | `gate` | `llm_outbound_gates.auto_gate_prompt` |
| `marvis/agent/auto_drive.py` | `decide_gate` | `gate`（重试） | `llm_outbound.gate_review_prompt` |
| `marvis/agent/feature_setup.py` | `_call_meaning_direction_llm` | `feature_meaning_direction` | `llm_outbound_analysis.feature_direction_prompt` |
| `marvis/agent/instruction_router.py` | `route_instruction` | `router` | `llm_outbound.gate_instruction_prompt` |
| `marvis/agent/instruction_router.py` | `route_instruction` | `router`（重试） | `llm_outbound.gate_review_prompt` |
| `marvis/agent/instruction_router.py` | `_recover_declared_parameter_adjustment` | `router` | `llm_outbound.gate_review_prompt` |
| `marvis/agent/instruction_router.py` | `_recover_declared_selection_decision` | `router` | `llm_outbound.gate_review_prompt` |
| `marvis/agent/semantic_authorization.py` | `review_semantic_authorization` | `semantic_authorization_reviewer` | `llm_outbound.gate_instruction_prompt` |
| `marvis/agent/semantic_intent.py` | `_invoke_and_parse` | 由参数传入 | `llm_outbound_intent.semantic_intent_prompt` |
| `marvis/agent/semantic_intent.py` | `_invoke_and_parse` | `<caller>_repair` | `llm_outbound_intent.semantic_intent_repair_prompt` |
| `marvis/agent/service.py` | `_summarize_stage` | `validation_stage_<stage>` | 按 stage/version 使用 `llm_outbound_scan.scan_prompt`、`llm_outbound_validation.pmml_scoring_prompt` / `reproducibility_prompt` / `validation_metrics_summary_prompt`；只发送类型化检查状态或聚合指标 |
| `marvis/agent/service.py` | `_summarize_stage` | `validation_stage_<stage>_repair` | `llm_outbound_validation.validation_metrics_summary_prompt`：沿用聚合投影，只追加固定缺失章节与经过历史上下文适配器的摘要 |
| `marvis/agent/service.py` | `compose_agent_start_message` | `validation_agent_start` | `llm_outbound_validation.validation_start_prompt` |
| `marvis/agent/service.py` | `failure_summary` | `validation_failure_summary` | `llm_outbound_failure.failure_prompt` |
| `marvis/agent/service.py` | `answer_chat_message` | `agent_chat` | `llm_outbound_chat.chat_prompt` |
| `marvis/agent/service.py` | `generate_word_conclusions` | `validation_v1_report` | `llm_outbound_revision.report_revision_prompt` |
| `marvis/agent/sql_import.py` | `propose_sql_import` | `sql_select_compile` | `llm_outbound_sql.sql_query_prompt` |
| `marvis/agent/strategy_request_compiler/core.py` | `_complete` | 由参数传入 | `llm_outbound_strategy.strategy_request_prompt` / `strategy_repair_prompt`（由本模块包装函数调用，禁止直接 mint） |
| `marvis/agent/strategy_request_compiler/core.py` | `_complete_sample_design_v2_correction` | 由参数传入 | `llm_outbound_strategy.strategy_sample_design_correction_prompt`（由本模块包装函数调用，禁止直接 mint） |
| `marvis/agent/validation_narrative.py` | `_generate_section` | `validation_<kind>_<topic>[_repair]` | `llm_outbound_validation.validation_metric_section_prompt` / `validation_report_section_prompt`，`llm_outbound_revision.report_revision_prompt` |
| `marvis/agent/workflow_insights.py` | `_llm_insight` | `workflow_insight` | `llm_outbound_workflows.workflow_insight_prompt` |
| `marvis/agent/workflow_recovery.py` | `answer_workflow_recovery_message` | `workflow_recovery_chat` | `llm_outbound_recovery.recovery_question_prompt` |
| `marvis/agent_memory/distillation.py` | `_summarize` | `memory_distillation` | `llm_outbound_distillation.distillation_prompt` |
| `marvis/drafts/authoring.py` | `draft_script` | `author` | `llm_outbound_drafts.authoring_prompt` |
| `marvis/drafts/learning.py` | `distill_learning` | `learning_note` | `llm_outbound_learning.public_learning_prompt` |
| `marvis/feature/derive.py` | `recommend_feature_crosses` | `feature_cross_recommendations` | `llm_outbound_analysis.feature_cross_prompt` |
| `marvis/operations/conversation_schedule.py` | `_llm_fill` | `monitoring_schedule_form` | `llm_outbound_operations.schedule_form_prompt` |
| `marvis/orchestrator/intent.py` | `_llm_classify` | `router_intent` | `llm_outbound_intent.workflow_intent_prompt` |
| `marvis/orchestrator/planner.py` | `generate` | `planner` | `llm_outbound.planning_prompt` |
| `marvis/orchestrator/planner.py` | `replan` | `planner` | `llm_outbound.planning_prompt` |
| `marvis/orchestrator/planner.py` | `next_explore_segment` | `planner` | `llm_outbound.planning_prompt` |
| `marvis/orchestrator/reviewer.py` | `llm_critique` | `critic` | `llm_outbound.outbound_prompt` |
| `marvis/orchestrator/reviewer.py` | `llm_critique` | `critic`（修复） | `llm_outbound.review_repair_prompt` |
| `marvis/orchestrator/reviewer.py` | `_llm_summarize` | `reviewer_summary` | `llm_outbound_business.business_review_prompt` |
| `marvis/orchestrator/reviewer.py` | `_llm_summarize` | `reviewer_summary`（修复） | `llm_outbound.review_repair_prompt` |
| `marvis/packs/modeling/report_tools.py` | `_draft_report_narratives` | `modeling_report` | `llm_outbound.report_prompt` |

共 37 处（2026-11-04 核对；`summarize_stage` 包装函数委托给 `_summarize_stage`，调用总数不变）。

`_summarize_stage` 的首个直接调用覆盖材料扫描、V2 PMML 打分、V1 一致性与 V1 指标摘要；未提供类型化适配器的 stage 直接返回本地 fallback。V2 指标走 `validation_narrative._generate_section` 的分节路径，已单独登记。上述扫描适配器不发送文件名、源码、错误原文或样本，评分/一致性/指标适配器只投影允许的聚合信息；客户端在发请求前仍强制要求 `StructuralPrompt`，同时检查 system prompt 与 schema。此处描述的是结构化出站边界，不把本地检查等同于外部服务或组织数据授权。

## 不在本表的出站

- 网页搜索 Tool 发出的查询不经过 `.complete(`，所以不在本表。2026-10-23 核对时它还没有做敏感检查，由 V3 计划 5.2 补上。
