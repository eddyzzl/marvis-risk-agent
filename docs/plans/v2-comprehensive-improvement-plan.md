# V2 Comprehensive Improvement Plan

Last updated: 2026-06-30

This document consolidates the remaining V2 work, previous review findings, and a fresh code review across backend architecture, Agent orchestration, modeling workflow, frontend UX, reliability, and release gates. It is now also used as the execution tracker for the implementation pass started on 2026-06-29.

## Review Scope

- Reviewed current repo state, existing plan docs, and review docs under `docs/plans/` and `docs/reviews/`.
- Reconciled the earlier "not fully finished" items: cross-repository transactional writes, Notebook/plugin sandboxing, `api.py` / `db.py` split, visual token system, and final total review.
- Ran four read-only expert reviews:
  - Backend architecture, persistence, transactions, performance.
  - Agent loop, AUTO decisions, gates, evidence, retry contracts.
  - Modeling lifecycle, PMML/PKL/report/handoff, sample weights.
  - Frontend workspace, visual system, user experience.
- Latest known full-suite evidence from the preceding runtime hardening pass: `1752 passed, 2 warnings`. This must be rerun after the implementation pass because the current working tree is dirty.
- 2026-06-29 follow-up review evidence:
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes with `git diff --check`, ruff, and `node --check`.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_plan_driver.py tests/test_data_join_api.py tests/test_artifacts_transactional.py tests/test_orch_db.py tests/test_orch_executor.py tests/test_modeling_artifact.py tests/test_modeling_pack.py tests/test_frontend_screen_table.py -q`: `127 passed`.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_release_push.py tests/test_modeling_api.py::test_modeling_multiple_files_runs_join_then_modeling_setup -q`: `11 passed`.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check`: passes with `1804 passed, 2 warnings` in `442.76s` after this document/update pass.
- Current focused verification from this implementation pass:
  - `scripts/check --skip-pytest` passes with `git diff --check`, ruff, and `node --check`.
  - `tests/test_plan_driver.py`: `25 passed` after report section-status renderer changes.
  - `tests/test_orch_templates.py tests/test_modeling_pack.py tests/test_plan_driver.py`: `57 passed`.
  - `tests/test_modeling_api.py`: `8 passed` after the G2/G3/selection/report/delivery confirmation chain update.
  - `tests/test_agent_gate_contracts.py tests/test_agent_autodrive.py tests/test_plan_driver.py tests/test_orch_db.py tests/test_orch_executor.py tests/test_orch_templates.py tests/test_modeling_training_dataset.py`: `101 passed`.
  - `tests/test_agent_gate_contracts.py tests/test_agent_autodrive.py tests/test_plan_driver.py tests/test_orch_db.py tests/test_orch_executor.py tests/test_orch_templates.py tests/test_modeling_training_dataset.py tests/test_modeling_api.py`: `109 passed`.
  - `tests/test_frontend_screen_table.py tests/test_frontend_static_v2.py::test_modeling_create_dialog_has_algorithm_selector`: `13 passed` after the sample-weight gate UI/control update.
  - `tests/test_plan_driver.py tests/test_agent_autodrive.py tests/test_modeling_api.py tests/test_orch_templates.py tests/test_modeling_pack.py`: `89 passed` after the sample-weight adjust/rerun path update.
  - `tests/test_artifacts_transactional.py tests/test_join_engine.py tests/test_data_repository_registry.py`: `23 passed` after adding `TransactionalArtifactStore` and migrating join execution output staging.
  - `tests/test_data_ops_pack.py tests/test_data_join_api.py`: `12 passed` after the join staging migration.
  - `tests/test_artifacts_transactional.py tests/test_data_ops_pack.py tests/test_join_engine.py tests/test_data_repository_registry.py`: `32 passed` after migrating `clean_format` / `dedup_rows` derived outputs to staging and tightening artifact path traversal checks.
  - `tests/test_data_join_api.py tests/test_data_ops_pack.py`: `13 passed` after the data_ops clean/dedup staging migration.
  - `tests/test_modeling_prepare.py tests/test_modeling_pack.py::test_reject_inference_tool_registers_augmented_dataset tests/test_modeling_reject_inference.py`: `15 passed` after migrating modeling derived parquet outputs to staging.
  - `tests/test_modeling_api.py tests/test_orch_templates.py tests/test_modeling_pack.py tests/test_modeling_prepare.py`: `51 passed` after the modeling derived parquet staging migration.
  - `tests/test_modeling_report.py`: `23 passed` after migrating `model_report_scored.parquet` and final xlsx report rendering to staging.
  - `tests/test_artifacts_transactional.py tests/test_modeling_artifact.py tests/test_modeling_recipes.py tests/test_modeling_pack.py tests/test_modeling_report.py tests/test_modeling_handoff.py tests/test_plugin_loader.py tests/test_drafts_promotion.py`: `103 passed, 2 warnings` after migrating model binary/meta/PMML/calibration files, plugin install/promote directories, and validation handoff materials to staged transactions.
  - `tests/test_artifacts_recovery.py tests/test_artifacts_transactional.py`: `16 passed` after adding startup artifact reconciliation for orphan `.staging` directories, plugin backups, and validation handoff material directories.
  - `tests/test_notebooks.py`: `21 passed` after adding an isolated notebook worker for ordinary full-notebook execution, worker error propagation, and parent timeout artifact preservation.
  - `tests/test_orch_db.py tests/test_orch_executor.py tests/test_recovery.py tests/test_artifacts_recovery.py`: `59 passed` after adding running step-run recovery for persisted outputs and interrupted runs.
  - `tests/test_modeling_api.py tests/test_agent_autodrive.py tests/test_data_join_api.py tests/test_feature_analysis_api.py tests/test_orch_api.py`: `60 passed` after moving driver turn orchestration into `marvis/agent/turn_handlers.py`.
  - `tests/test_data_api.py tests/test_data_join_api.py`: `14 passed` after extracting data/join API payload and preview masking helpers into `marvis/api_data_payloads.py`.
  - `tests/test_api_v2.py tests/test_orch_api.py`: `85 passed` after the data payload helper extraction kept broader API wrappers import-compatible.
  - `tests/test_agent_memory_api.py tests/test_memory_policy.py tests/test_memory_determinism_guard.py`: `24 passed` after moving Agent Memory payload/context/audit helpers to `marvis/agent_memory/api_support.py` and `/api/agent-memory...` routes to `marvis/routers/agent_memory.py`.
  - `tests/test_agent_api.py`: `86 passed` after the Agent Memory router extraction kept validation-agent memory capture/context/audit wrappers import-compatible.
  - `tests/test_api_v2.py tests/test_orch_api.py tests/test_data_api.py tests/test_data_join_api.py`: `99 passed` after mounting the Agent Memory router separately from the legacy API router.
  - `tests/test_frontend_shell_static.py::test_app_entry_is_split_into_frontend_modules tests/test_frontend_shell_static.py::test_unselected_workspace_shows_centered_welcome_only tests/test_frontend_screen_table.py tests/test_frontend_static_v2.py::test_frontend_uses_v2_task_actions_only`: `15 passed` after adding semantic visual tokens and extracting theme handling to `static/js/theme.js`.
  - `tests/test_frontend_shell_static.py tests/test_frontend_v2_workflow_create.py tests/test_frontend_static_v2.py`: `224 passed` after extracting create-dialog task type definitions and task-type display order into `static/js/task-types.js`.
  - `tests/test_frontend_shell_static.py tests/test_frontend_v2_workflow_create.py tests/test_frontend_static_v2.py`: `224 passed` after extracting the create-dialog controller behavior, payload assembly, material upload, run-mode cards, algorithm/sample-weight controls, and report defaults into `static/js/create-task-dialog.js`.
  - `tests/test_db.py tests/test_orch_db.py tests/test_plugin_db.py tests/test_modeling_db.py tests/test_strategy_db.py tests/test_drafts_db.py`: `72 passed` after extracting schema/connection setup into `marvis/db_schema.py`.
  - `tests/test_plugin_db.py tests/test_plugin_registry.py tests/test_plugin_hooks.py tests/test_db.py::test_update_report_values_with_audit_records_changed_keys tests/test_db.py::test_update_report_values_with_audit_rolls_back_when_audit_fails tests/test_db.py::test_update_agent_report_conclusions_with_audit_rolls_back_when_audit_fails`: `31 passed` after making `PluginRepository.write_audit/list_audit` reuse the shared audit row helpers.
  - `tests/test_agent_gate_contracts.py tests/test_plan_driver.py`: `35 passed` after enriching failure envelopes with editable input defaults and explicit downstream reset step ids.
  - `tests/test_plan_driver.py`: `38 passed` after propagating the latest failed step-run `error_kind` into `FailureEnvelope` so validation/postcheck/runtime failures are distinguishable in retry UX.
  - `tests/test_orch_executor.py tests/test_orch_db.py`: `44 passed` after adding tool version, manifest hash, source dataset refs, and artifact refs to persisted step evidence.
  - `tests/test_agent_autodrive.py`: `24 passed` after adding a deterministic AUTO low-risk control allowlist for declared-but-expensive tuning and delivery actions.
  - `tests/test_orch_api.py tests/test_frontend_v2_plan.py tests/test_frontend_static_v2.py`: `238 passed` after exposing failure envelopes on plan step API payloads and rendering retry defaults/reset scope in both V2 plan views.
  - `tests/test_modeling_recipes.py tests/test_modeling_pack.py tests/test_plan_driver.py tests/test_frontend_screen_table.py tests/test_agent_autodrive.py`: `116 passed, 2 warnings` after adding sample-weight quality diagnostics to modeling setup, gate metadata, frontend controls, and AUTO prompts.
  - `tests/test_orch_db.py tests/test_orch_api.py tests/test_orch_executor.py tests/test_plan_driver.py`: `96 passed` after making step confirmation require the persisted step to still be `awaiting_confirm`.
  - `tests/test_agent_gate_contracts.py tests/test_plan_driver.py tests/test_agent_autodrive.py`: `63 passed` after making AUTO halt on gate-level high-risk flags and wide downstream reset policies.
  - `tests/test_plan_driver.py tests/test_frontend_screen_table.py tests/test_agent_autodrive.py`: `72 passed` after expanding the modeling setup panel/contract with target type, algorithms, tuning budget, PMML support, split/OOT diagnostics, and AUTO context.
  - `tests/test_frontend_screen_table.py tests/test_frontend_v2_api_state.py tests/test_frontend_shell_static.py::test_app_entry_is_split_into_frontend_modules`: `23 passed` after extracting `renderModelingSetupPanel` into `static/js/v2/modeling_setup_panel.js`.
  - `tests/test_frontend_screen_table.py tests/test_frontend_v2_api_state.py tests/test_frontend_shell_static.py::test_app_entry_is_split_into_frontend_modules`: `23 passed` after moving the modeling sample-weight adjust controller into `static/js/v2/modeling_setup_panel.js`.
  - `tests/test_plan_driver.py tests/test_frontend_screen_table.py tests/test_frontend_v2_api_state.py`: `58 passed` after adding structured model comparison/delivery readiness metadata and the `ModelDeliveryPanel` frontend module.
  - `tests/test_plan_driver.py tests/test_frontend_screen_table.py tests/test_frontend_v2_api_state.py`: `59 passed` after merging model-report path and section readiness into the delivery panel for both final gates and done messages.
  - `tests/test_plan_driver.py tests/test_frontend_screen_table.py tests/test_frontend_v2_api_state.py`: `60 passed` after adding editable target type, algorithm, tuning-trial, and sample-weight controls to `ModelingSetupPanel` with required override reasons for structural changes.
  - `tests/test_plan_driver.py tests/test_frontend_screen_table.py tests/test_frontend_v2_api_state.py`: `60 passed` after allowing family-mismatch recipes to be selected when switching target type and blocking target/algorithm family mismatches before submission.
  - `tests/test_plan_driver.py tests/test_frontend_screen_table.py tests/test_frontend_v2_api_state.py`: `60 passed` after adding delivery-panel business signals for stability, feature count, calibration state, and handoff readiness.
  - `tests/test_frontend_screen_table.py tests/test_frontend_v2_api_state.py`: `25 passed` after adding a lightweight DOM-structure smoke for the modeling setup and delivery panels.
  - `tests/test_plan_driver.py tests/test_frontend_screen_table.py tests/test_frontend_v2_api_state.py`: `61 passed` after adding backend modeling override guidance and frontend risk-guidance rendering for target type, algorithms, tuning budget, sample weight, and split quality.
  - `CONDA_NO_PLUGINS=true MARVIS_RUN_PLAYWRIGHT_SMOKE=1 conda run -n py_313 python -m pytest tests/test_frontend_playwright_smoke.py -q`: `1 passed` after adding optional real-browser desktop/mobile smoke for the modeling setup and delivery panels.
  - `CONDA_NO_PLUGINS=true MARVIS_RUN_PLAYWRIGHT_SMOKE=1 conda run -n py_313 python -m pytest tests/test_frontend_playwright_smoke.py -q`: `3 passed` after expanding optional browser smoke to the real app welcome shell, modeling create dialog, plan rail, screen selector table, desktop/mobile, and light/dark startup paths.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_frontend_playwright_smoke.py tests/test_frontend_screen_table.py tests/test_frontend_v2_api_state.py tests/test_frontend_shell_static.py::test_unselected_workspace_shows_centered_welcome_only -q`: `26 passed, 3 skipped` after the broader Playwright smoke stayed opt-in for default CI.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_plan_driver.py tests/test_frontend_screen_table.py tests/test_frontend_v2_api_state.py -q`: `61 passed` after adding model delivery policy signals for scorecard status, monotonicity evidence, approval recommendation, and approval-policy readiness.
  - `CONDA_NO_PLUGINS=true MARVIS_RUN_PLAYWRIGHT_SMOKE=1 conda run -n py_313 python -m pytest tests/test_frontend_playwright_smoke.py -q`: `3 passed` after extending the browser smoke to cover delivery policy cards.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_modeling_pack.py tests/test_orch_templates.py tests/test_plan_driver.py -q`: `71 passed` after turning model-delivery policy signals into executable `select_experiment.selection_policy` gates with required PMML/handoff, scorecard preference, monotonicity checks, feature/PSI limits, audited override reasons, and robust string-boolean normalization.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_modeling_pack.py::test_policy_selection_prefers_compliant_scorecard_candidate tests/test_modeling_pack.py::test_train_models_supports_catboost_and_sample_weight_col tests/test_orch_templates.py::test_standard_modeling_template_instantiates_valid_report_plan tests/test_orch_templates.py::test_modeling_template_phases_gates_and_refs -q`: `4 passed` after the final policy helper/static-check fix.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_plan_driver.py::test_modeling_selection_gate_carries_delivery_payload tests/test_frontend_screen_table.py::test_model_delivery_panel_renderer_and_branch_are_wired tests/test_frontend_screen_table.py::test_model_delivery_panel_renders_selection_and_actions tests/test_frontend_screen_table.py::test_modeling_panels_combined_dom_smoke_contract -q`: `4 passed` after wiring executable policy decisions into model-delivery metadata/readiness and the frontend delivery panel.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_plan_driver.py tests/test_frontend_screen_table.py tests/test_frontend_v2_api_state.py -q`: `61 passed` after the model-delivery policy-decision panel wiring.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_modeling_pack.py::test_modeling_manifest_registers_expected_tools tests/test_modeling_pack.py::test_train_models_supports_catboost_and_sample_weight_col tests/test_orch_templates.py::test_standard_modeling_template_instantiates_valid_report_plan tests/test_orch_templates.py::test_modeling_template_phases_gates_and_refs tests/test_plan_driver.py::test_done_message_carries_post_training_delivery_payload tests/test_frontend_screen_table.py::test_model_delivery_panel_renders_selection_and_actions tests/test_frontend_screen_table.py::test_modeling_panels_combined_dom_smoke_contract -q`: `7 passed` after adding the post-training approval package artifact and wiring it into delivery metadata, renderers, templates, and frontend artifact lists.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_modeling_pack.py::test_modeling_manifest_registers_expected_tools tests/test_modeling_pack.py::test_train_models_supports_catboost_and_sample_weight_col tests/test_plan_driver.py::test_done_message_carries_post_training_delivery_payload tests/test_frontend_screen_table.py::test_model_delivery_panel_renders_selection_and_actions tests/test_frontend_screen_table.py::test_modeling_panels_combined_dom_smoke_contract -q`: `5 passed` after adding the human-readable Markdown approval package and exposing it through delivery metadata/readiness/front-end artifacts while keeping the JSON evidence package.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_modeling_pack.py tests/test_plan_driver.py tests/test_frontend_screen_table.py tests/test_frontend_v2_api_state.py -q`: `84 passed` after the Markdown approval package update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the Markdown approval package update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_modeling_handoff.py tests/test_modeling_pack.py::test_modeling_manifest_registers_expected_tools tests/test_modeling_pack.py::test_train_models_supports_catboost_and_sample_weight_col tests/test_orch_templates.py::test_standard_modeling_template_instantiates_valid_report_plan tests/test_plan_driver.py::test_done_message_carries_post_training_delivery_payload tests/test_plan_driver.py::test_render_registry_has_modeling_renderers_and_generic_fallback tests/test_frontend_screen_table.py::test_model_delivery_panel_renders_selection_and_actions tests/test_frontend_screen_table.py::test_modeling_panels_combined_dom_smoke_contract -q`: `13 passed` after adding the post-training challenger/backtest task package and delivery-panel wiring.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_modeling_handoff.py tests/test_modeling_pack.py tests/test_orch_templates.py tests/test_plan_driver.py tests/test_frontend_screen_table.py tests/test_frontend_v2_api_state.py tests/test_db.py tests/test_modeling_db.py -q`: `134 passed` after fixing select-experiment readiness so challenger/backtest readiness appears only once the G5 action/outputs exist.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the challenger/backtest task package update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_modeling_handoff.py::test_create_challenger_backtest_task_writes_materials_task_and_audit tests/test_modeling_pack.py::test_modeling_manifest_registers_expected_tools tests/test_modeling_pack.py::test_train_models_supports_catboost_and_sample_weight_col tests/test_plan_driver.py::test_done_message_carries_post_training_delivery_payload tests/test_plan_driver.py::test_render_registry_has_modeling_renderers_and_generic_fallback tests/test_frontend_screen_table.py::test_model_delivery_panel_renders_selection_and_actions tests/test_frontend_screen_table.py::test_modeling_panels_combined_dom_smoke_contract -q`: `7 passed` after adding versioned monitoring-policy JSON/Markdown artifacts, approval-package evidence, delivery readiness, and frontend artifact rendering.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_modeling_handoff.py tests/test_modeling_pack.py tests/test_orch_templates.py tests/test_plan_driver.py tests/test_frontend_screen_table.py tests/test_frontend_v2_api_state.py tests/test_db.py tests/test_modeling_db.py -q`: `134 passed` after the monitoring-policy update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the monitoring-policy update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_modeling_handoff.py::test_create_challenger_backtest_task_writes_materials_task_and_audit tests/test_modeling_pack.py::test_modeling_manifest_registers_expected_tools tests/test_modeling_pack.py::test_train_models_supports_catboost_and_sample_weight_col tests/test_orch_templates.py::test_standard_modeling_template_instantiates_valid_report_plan tests/test_plan_driver.py::test_done_message_carries_post_training_delivery_payload tests/test_frontend_screen_table.py::test_model_delivery_panel_renders_selection_and_actions -q`: `6 passed` after adding optional prior Champion/Challenger comparison artifacts, template-slot wiring, previous-selected auto-resolution, and delivery-panel risk counting.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_modeling_handoff.py tests/test_modeling_pack.py tests/test_orch_templates.py tests/test_plan_driver.py tests/test_frontend_screen_table.py tests/test_frontend_v2_api_state.py tests/test_db.py tests/test_modeling_db.py -q`: `134 passed` after the Champion/Challenger comparison update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the Champion/Challenger comparison update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_modeling_pack.py tests/test_orch_templates.py tests/test_plan_driver.py tests/test_frontend_screen_table.py tests/test_frontend_v2_api_state.py -q`: `96 passed` after the approval-package update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_notebooks.py::test_live_notebook_session_reuses_kernel_for_appended_cells tests/test_notebooks.py::test_live_notebook_session_rejects_appended_cells_by_default tests/test_pipeline_v2.py::test_staged_metrics_use_live_notebook_sample_without_rerunning_notebook tests/test_pipeline_v2.py::test_completed_task_cannot_rerun_metrics_after_live_notebook_session_closed -q`: `4 passed` after making live notebook appended-cell execution opt-in per session.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_notebooks.py -q`: `22 passed` after the live notebook appended-cell safety update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_pipeline_v2.py::test_metrics_stage_marks_sample_column_failure_as_metrics_failure tests/test_pipeline_v2.py::test_metrics_stage_success_captures_model_experience_memory tests/test_pipeline_v2.py::test_metrics_stage_cancel_returns_to_executed_status -q`: `3 passed` after confirming the V1 metrics stage still works with explicitly authorized live notebook sessions/fakes.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_notebooks.py::test_appended_system_cells_are_visible_before_execution tests/test_notebooks.py::test_live_notebook_session_reuses_kernel_for_appended_cells tests/test_notebooks.py::test_live_notebook_appended_execution_requires_policy tests/test_notebooks.py::test_live_notebook_appended_policy_rejects_unlisted_marvis_kind tests/test_notebooks.py::test_live_notebook_session_rejects_appended_cells_by_default tests/test_pipeline_v2.py::test_v1_validation_appended_policy_matches_generated_cell_kinds -q`: `6 passed` after replacing naked live appended-cell permission with an explicit V1 validation policy and generated-cell-kind allowlist.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_notebooks.py tests/test_pipeline_v2.py -q`: `51 passed` after the explicit appended-cell policy update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_artifacts_transactional.py tests/test_join_engine.py tests/test_data_ops_pack.py tests/test_data_join_api.py -q`: `33 passed` after adding `ArtifactUnitOfWork` and migrating join result registration to the reusable artifact+DB/audit callback boundary.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes with `git diff --check`, ruff, and `node --check` after the artifact/directory transaction migration.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the step-run recovery update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the API/DB/frontend split updates.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the `api_data_payloads.py` extraction.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the failure-envelope retry contract update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the failure-envelope `error_kind` propagation update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the evidence lineage update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the AUTO high-risk control guard update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the frontend failure retry contract update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the sample-weight diagnostics update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the step confirmation state guard update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the AUTO gate-risk policy update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the plugin/draft `process:spawn` sandbox guard update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the modeling setup panel expansion.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the `ModelingSetupPanel` renderer extraction.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the `ModelingSetupPanel` controller extraction.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the `task-types.js` create-dialog config extraction.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the `CreateTaskDialog` controller extraction.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_frontend_static_v2.py::test_v2_plan_rail_fetch_errors_are_visible_and_retryable tests/test_frontend_static_v2.py::test_plan_rail_matches_validation_stepper_with_nested_subtasks tests/test_frontend_static_v2.py::test_plan_rail_retry_step_posts_edited_inputs tests/test_frontend_static_v2.py::test_plan_rail_artifact_preview_is_wired_to_real_app_shell tests/test_frontend_static_v2.py::test_strategy_and_vintage_welcome_cards_are_enabled tests/test_frontend_static_v2.py::test_agent_message_meta_label_includes_plan_step_context -q`: `6 passed` after extracting `PlanRailController` into `static/js/v2/plan_rail_controller.js`.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_frontend_shell_static.py tests/test_frontend_v2_workflow_create.py tests/test_frontend_static_v2.py -q`: `224 passed` after updating app-shell/static contracts for the extracted `PlanRailController`.
  - `CONDA_NO_PLUGINS=true MARVIS_RUN_PLAYWRIGHT_SMOKE=1 conda run -n py_313 python -m pytest tests/test_frontend_playwright_smoke.py -q`: `4 passed` after the `PlanRailController` extraction.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the `PlanRailController` extraction.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_frontend_shell_static.py tests/test_frontend_v2_workflow_create.py tests/test_frontend_static_v2.py tests/test_frontend_screen_table.py -q`: `240 passed` after extracting agent conversation/timeline pure helpers into `static/js/agent-conversation-view.js`.
  - `CONDA_NO_PLUGINS=true MARVIS_RUN_PLAYWRIGHT_SMOKE=1 conda run -n py_313 python -m pytest tests/test_frontend_playwright_smoke.py -q`: `4 passed` after the conversation helper extraction.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the conversation helper extraction.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_frontend_shell_static.py tests/test_frontend_v2_workflow_create.py tests/test_frontend_static_v2.py tests/test_frontend_screen_table.py -q`: `240 passed` after extracting task workspace storage/greeting helpers into `static/js/task-workspace-state.js`.
  - `CONDA_NO_PLUGINS=true MARVIS_RUN_PLAYWRIGHT_SMOKE=1 conda run -n py_313 python -m pytest tests/test_frontend_playwright_smoke.py -q`: `4 passed` after the task workspace state helper extraction.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the task workspace state helper extraction.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_frontend_screen_table.py::test_screen_table_renderer_and_manual_branch_are_wired tests/test_frontend_screen_table.py::test_screen_confirm_posts_edited_selection tests/test_frontend_screen_table.py::test_screen_threshold_adjust_posts_structured_params tests/test_frontend_screen_table.py::test_screen_table_only_latest_gate_is_interactive tests/test_frontend_screen_table.py::test_screen_threshold_adjust_rejects_empty_and_posts_valid_payload -q`: `5 passed` after extracting the feature-screen renderer/controller into `static/js/v2/screen_gate_controller.js`.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_frontend_shell_static.py tests/test_frontend_v2_workflow_create.py tests/test_frontend_static_v2.py tests/test_frontend_screen_table.py -q`: `240 passed` after updating the app-shell/static contracts for the extracted screen gate controller.
  - `CONDA_NO_PLUGINS=true MARVIS_RUN_PLAYWRIGHT_SMOKE=1 conda run -n py_313 python -m pytest tests/test_frontend_playwright_smoke.py -q`: `4 passed` after the screen gate controller extraction.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the screen gate controller extraction.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_frontend_screen_table.py::test_dedup_picker_renderer_and_branch_are_wired tests/test_frontend_screen_table.py::test_dedup_picker_posts_strategies tests/test_frontend_screen_table.py::test_join_c1_form_renderer_and_submit_are_wired tests/test_frontend_screen_table.py::test_join_gate_controller_posts_c1_and_dedup_payloads -q`: `4 passed` after extracting C1/dedup data-join gate rendering and submit handling into `static/js/v2/join_gate_controller.js`.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_frontend_shell_static.py tests/test_frontend_v2_workflow_create.py tests/test_frontend_static_v2.py tests/test_frontend_screen_table.py -q`: `242 passed` after updating app-shell/static contracts for the extracted join gate controller.
  - `CONDA_NO_PLUGINS=true MARVIS_RUN_PLAYWRIGHT_SMOKE=1 conda run -n py_313 python -m pytest tests/test_frontend_playwright_smoke.py -q`: `4 passed` after the join gate controller extraction.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the join gate controller extraction.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_frontend_screen_table.py::test_screen_table_renderer_and_manual_branch_are_wired tests/test_frontend_screen_table.py::test_screen_table_only_latest_gate_is_interactive tests/test_frontend_static_v2.py::test_driver_manual_analysis_omits_plan_overview_messages -q`: `3 passed` after extracting manual driver analysis branching into `static/js/v2/driver_manual_analysis.js`.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_frontend_shell_static.py tests/test_frontend_v2_workflow_create.py tests/test_frontend_static_v2.py tests/test_frontend_screen_table.py -q`: `242 passed` after updating frontend contracts for the extracted manual analysis module.
  - `CONDA_NO_PLUGINS=true MARVIS_RUN_PLAYWRIGHT_SMOKE=1 conda run -n py_313 python -m pytest tests/test_frontend_playwright_smoke.py -q`: `4 passed` after the manual analysis extraction.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the manual analysis extraction.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_frontend_screen_table.py::test_driver_gate_confirm_controller_renders_and_posts_confirm tests/test_frontend_static_v2.py::test_agent_mode_creation_and_stepper_hide_manual_buttons -q`: `2 passed` after extracting generic driver gate confirmation into `static/js/v2/driver_gate_confirm.js`.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_frontend_shell_static.py tests/test_frontend_v2_workflow_create.py tests/test_frontend_static_v2.py tests/test_frontend_screen_table.py -q`: `243 passed` after updating app-shell/static contracts for the extracted driver gate confirm controller.
  - `CONDA_NO_PLUGINS=true MARVIS_RUN_PLAYWRIGHT_SMOKE=1 conda run -n py_313 python -m pytest tests/test_frontend_playwright_smoke.py -q`: `4 passed` after the driver gate confirm extraction.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the driver gate confirm extraction.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_driver_report_download_frontend.py -q`: `1 passed` after making driver-report downloads use an absolute `/api/...` URL and updating the PlanRailController affordance test.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the driver-report download URL fix.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the Agent Memory router/support extraction.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the `PluginRepository` audit helper de-duplication.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the `ModelDeliveryPanel` metadata/rendering update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the model-report readiness merge into delivery metadata.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the editable modeling setup controls update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the modeling setup algorithm-family guard update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the modeling override-guidance and optional Playwright smoke update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the broader browser smoke update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after adding executable model-selection policy gates.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check`: passes with `1794 passed, 2 warnings` after fixing the theme-module test contract and live-notebook session parameter.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_modeling_pack.py::test_modeling_manifest_registers_expected_tools tests/test_modeling_pack.py::test_train_models_supports_catboost_and_sample_weight_col tests/test_plan_driver.py::test_done_message_carries_post_training_delivery_payload tests/test_frontend_screen_table.py::test_model_delivery_panel_renders_selection_and_actions -q`: `4 passed` after adding JSON/Markdown model-card artifacts to G5 delivery.
  - `CONDA_NO_PLUGINS=true MARVIS_RUN_PLAYWRIGHT_SMOKE=1 conda run -n py_313 python -m pytest tests/test_frontend_playwright_smoke.py -q`: `4 passed` after adding a real app-shell modeling-task smoke that loads `index.html`, task list, plan rail, and model-delivery message metadata in Chromium.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_modeling_handoff.py tests/test_modeling_pack.py tests/test_orch_templates.py tests/test_plan_driver.py tests/test_frontend_screen_table.py tests/test_frontend_v2_api_state.py tests/test_db.py tests/test_modeling_db.py -q`: `134 passed` after the model-card and real-task smoke update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the model-card and real-task smoke update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_modeling_artifact.py tests/test_modeling_pack.py tests/test_plan_driver.py tests/test_frontend_screen_table.py -q`: `81 passed` after surfacing calibrated-score PMML limitations in capabilities, readiness, model cards, and approval packages.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_modeling_pack.py::test_policy_selection_prefers_compliant_scorecard_candidate tests/test_modeling_pack.py::test_selection_policy_rejects_partial_scorecard_monotonicity tests/test_modeling_pack.py::test_selection_policy_rejects_zero_monotone_constraints tests/test_plan_driver.py::test_model_delivery_policy_signals_warn_on_partial_scorecard_monotonicity -q`: `4 passed` after tightening scorecard/monotonicity policy evidence so partial scorecard directions and all-zero tree constraints no longer pass.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_modeling_pack.py tests/test_plan_driver.py tests/test_frontend_screen_table.py -q`: `77 passed` after the stricter scorecard/monotonicity policy fixture update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_modeling_handoff.py tests/test_modeling_artifact.py tests/test_modeling_pack.py::test_calibrate_model_records_diagnostics_and_report_sheet tests/test_modeling_pack.py::test_train_models_supports_catboost_and_sample_weight_col tests/test_plan_driver.py::test_done_message_carries_post_training_delivery_payload -q`: `17 passed` after the native Booster edge fixture.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the native Booster edge fixture.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_agent_autodrive.py -q`: `30 passed` after extending AUTO risk flags to halt on strategy/vintage `manual_review` and `approval` domain markers.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_agent_autodrive.py tests/test_agent_gate_contracts.py tests/test_plan_driver.py::test_screen_gate_carries_structured_screen_payload tests/test_plan_driver.py::test_plan_overview_message_carries_gate_envelope tests/test_plan_driver.py::test_resume_structured_screen_control_rejects_stale_or_missing_gate_token -q`: `37 passed` after the AUTO domain-risk fixture update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the AUTO domain-risk fixture update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_frontend_v2_plan.py tests/test_frontend_v2_plan_confirm.py tests/test_frontend_static_v2.py::test_plan_rail_retry_step_posts_edited_inputs tests/test_frontend_v2_api_state.py -q`: `31 passed` after adding schema-driven retry fields to both the modular V2 plan view and main plan rail while preserving JSON textarea fallback.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the schema-driven retry fields update.
  - `CONDA_NO_PLUGINS=true MARVIS_RUN_PLAYWRIGHT_SMOKE=1 conda run -n py_313 python -m pytest tests/test_frontend_playwright_smoke.py -q`: `4 passed` after the latest modeling/AUTO/retry updates, covering the current real-browser Chromium smoke suite.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_frontend_static_v2.py::test_metric_overview_uses_semantic_visual_tokens tests/test_frontend_static_v2.py::test_metric_overview_dark_theme_keeps_hover_and_chart_text_readable -q`: `2 passed` after centralizing metric/report/KPI/ROC chart tokens.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_frontend_static_v2.py -q`: `211 passed` after the visual-token contract update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the metric/report/KPI/ROC visual-token update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_frontend_static_v2.py -q`: `212 passed` after tokenizing generic download actions, send-stop composer controls, disabled/focus states, and user-message composer bubbles.
  - `CONDA_NO_PLUGINS=true MARVIS_RUN_PLAYWRIGHT_SMOKE=1 conda run -n py_313 python -m pytest tests/test_frontend_playwright_smoke.py -q`: `4 passed` after the generic action/composer token update, confirming the current Chromium smoke suite still loads the app shell, task list, plan rail, and modeling delivery metadata.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_frontend_screen_table.py::test_modeling_panels_use_semantic_model_visual_tokens tests/test_frontend_screen_table.py::test_modeling_setup_weight_picker_renderer_and_branch_are_wired tests/test_frontend_screen_table.py::test_model_delivery_panel_renderer_and_branch_are_wired -q`: `3 passed` after centralizing modeling setup/delivery panel state tokens.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_modeling_api.py tests/test_orch_templates.py tests/test_modeling_recipes.py::test_build_modeling_proposal_derives_continuous_target_type_from_regressor tests/test_modeling_recipes.py::test_build_modeling_proposal_uses_explicit_target_type_default_recipe tests/test_modeling_recipes.py::test_build_modeling_proposal_stays_binary_for_classification_recipes tests/test_modeling_recipes.py::test_build_modeling_proposal_derives_multiclass_target_type_from_recipe -q`: `24 passed` after making default selection policies target-type-aware so continuous/multiclass workflows do not hard-block on PMML/handoff requirements they cannot satisfy.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/test_artifacts_transactional.py::test_artifact_unit_of_work_commits_artifacts_after_db_context_succeeds tests/test_artifacts_transactional.py::test_artifact_unit_of_work_rolls_back_artifact_when_db_context_fails tests/test_data_repository_registry.py::test_dataset_repository_connection_scoped_join_result_rolls_back_with_transaction tests/test_data_repository_registry.py::test_join_engine_uses_connection_scoped_artifact_unit_of_work tests/test_data_repository_registry.py::test_join_engine_rolls_back_result_dataset_and_file_when_executed_audit_fails -q`: `5 passed` after making join-result registration use a SQLite connection-scoped artifact unit of work.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 pytest tests/test_modeling_api.py::test_modeling_business_materials_flow_into_report_and_delivery -q`: `1 passed` after adding source-dir business-column and feature-dictionary auto-detection, preserving business passthrough columns through `make_split`, injecting task project metadata into report slots, and proving the report/delivery chain creates available business sections plus model-card/approval/handoff outputs.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 pytest tests/test_modeling_recipes.py -q`: `27 passed, 2 warnings` after the modeling setup inference update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 pytest tests/test_modeling_api.py -q`: `10 passed` after the real business-material smoke was added to the conversational modeling API suite and expanded to no-split auto-split preservation.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 pytest tests/test_modeling_handoff.py::test_post_training_action_skips_native_tree_booster_without_failing -q`: `2 passed` after extending G5 native Booster skip coverage from LightGBM to both LightGBM and XGBoost.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 pytest tests/test_modeling_api.py::test_modeling_business_materials_without_split_survive_auto_split tests/test_modeling_api.py::test_modeling_business_materials_flow_into_report_and_delivery tests/test_modeling_handoff.py::test_post_training_action_skips_native_tree_booster_without_failing -q`: `4 passed` after preserving business and sample-weight passthrough columns in the setup-time auto-split derived dataset and fixing XGBoost native Booster experiment metadata.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 pytest tests/test_modeling_api.py tests/test_modeling_recipes.py tests/test_modeling_handoff.py -q`: `45 passed, 2 warnings` after the subagent review fixes.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 pytest tests/test_modeling_handoff.py::test_post_training_action_skips_mlp_without_creating_validation_task tests/test_modeling_handoff.py -q`: `9 passed` after adding the MLP/DNN native-only G5 delivery fixture.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 pytest tests/test_modeling_handoff.py::test_post_training_action_writes_sample_weight_governance_artifacts -q`: `1 passed` after adding structured sample-weight governance to model-card, approval-package, and monitoring-policy artifacts.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 pytest tests/test_modeling_handoff.py -q`: `10 passed` after the sample-weight governance artifact update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 pytest tests/test_modeling_handoff.py tests/test_modeling_pack.py::test_train_models_supports_catboost_and_sample_weight_col tests/test_plan_driver.py::test_done_message_carries_post_training_delivery_payload -q`: `12 passed` after confirming downstream modeling delivery contracts still consume the updated post-training payload.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the sample-weight governance artifact update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 pytest tests/test_modeling_pack.py::test_selection_policy_rejects_missing_feature_and_psi_evidence tests/test_plan_driver.py::test_driver_n_trials_only_adjust_requires_fresh_modeling_gate_token tests/test_modeling_handoff.py::test_post_training_action_skips_malformed_scorecard_without_pmml_failure -q`: `3 passed` after blocking selection-policy candidates with missing feature-count/OOT-PSI evidence, making `n_trials`-only modeling setup adjusts require a fresh gate token, and aligning scorecard PMML capability checks with real WOE export validation.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 pytest tests/test_plan_driver.py::test_driver_modeling_setup_adjust_reruns_spec_and_downstream_screen tests/test_plan_driver.py::test_driver_n_trials_only_adjust_requires_fresh_modeling_gate_token tests/test_plan_driver.py::test_driver_sample_weight_adjust_rejects_unknown_candidate_without_reset tests/test_modeling_pack.py::test_policy_selection_prefers_compliant_scorecard_candidate tests/test_modeling_pack.py::test_selection_policy_rejects_partial_scorecard_monotonicity tests/test_modeling_pack.py::test_selection_policy_rejects_zero_monotone_constraints tests/test_modeling_pack.py::test_selection_policy_rejects_missing_feature_and_psi_evidence tests/test_modeling_handoff.py -q`: `18 passed` after the AUTO/modeling edge-fixture update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 pytest tests/test_modeling_artifact.py tests/test_modeling_handoff.py tests/test_modeling_pack.py::test_policy_selection_prefers_compliant_scorecard_candidate tests/test_modeling_pack.py::test_selection_policy_rejects_missing_feature_and_psi_evidence tests/test_plan_driver.py::test_driver_n_trials_only_adjust_requires_fresh_modeling_gate_token -q`: `21 passed` after rechecking the shared scorecard PMML validator against artifact export and post-training delivery paths.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the AUTO/modeling edge-fixture update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check`: passes with `1871 passed, 4 skipped, 2 warnings` after splitting `n_trials` gate validation across modeling setup and tuning gates.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 pytest tests/test_plan_driver.py -q`: `38 passed` after extracting gate-control validation into `marvis/agent/gate_response_adapter.py`.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the `GateResponseAdapter` extraction.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 pytest tests/test_modeling_report.py tests/test_modeling_training_dataset.py -q`: `26 passed` after reusing `TrainingDataset` as a cached report/scoring backend for repeated scored-dataset reads.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the cached report/scoring backend update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 pytest tests/test_plan_driver.py -q`: `38 passed` after extracting gate adjust/replan execution into `marvis/agent/gate_execution_adapter.py` and shared plan/turn DTO helpers.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 pytest tests/test_modeling_pack.py -q`: `28 passed` after making selection-policy OOT PSI gates prefer `weighted_psi_oot_vs_train` when sample-weight metrics are available.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 pytest tests/test_plan_driver.py -q`: `38 passed` after moving screen selection and join dedup gate controls into `GateExecutionAdapter`.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 pytest tests/test_data_repository_registry.py::test_dataset_registry_register_existing_on_connection tests/test_data_ops_pack.py::test_data_ops_derived_frame_registration_failure_rolls_back_staged_file tests/test_data_ops_pack.py::test_data_ops_derived_frame_connection_failure_rolls_back_promoted_file tests/test_data_ops_pack.py::test_data_ops_clean_format_and_dedup_rows_via_runner -q`: `4 passed` after moving data_ops derived dataset registration onto the connection-scoped artifact unit-of-work path.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 pytest tests/test_modeling_prepare.py tests/test_modeling_pack.py::test_reject_inference_tool_registers_augmented_dataset tests/test_modeling_pack.py::test_reject_inference_audit_failure_rolls_back_dataset_and_file -q`: `14 passed` after moving modeling prepare/reject-inference derived parquet registration onto connection-scoped artifact unit-of-work paths.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 pytest tests/test_plugin_loader.py tests/test_plugin_runner.py -q`: `42 passed` after requiring untrusted plugin/draft code to declare `process:spawn` before starting child processes.

## Executive Summary

V2 is materially stronger than the original runtime: PlanDriver is the common execution path, modeling has real recipes and PMML/report/handoff tools, evidence output versioning exists, plugin/tool started checkpoints exist, and the old settings-level V2 workbench has been removed from the UI.

It is not ready to call "complete" yet. The main remaining gap is not one isolated bug; it is that several capabilities are working as primitives but are not yet productized as reliable, typed, user-facing workflows. The highest-value next work is:

1. Finish gate adapters, stale-token coverage, and structured failure/retry UX on top of the new `GateEnvelope`/`FailureEnvelope` base. The failure envelope now flows through the plan API into retry panels, and retry panels now render schema-driven fields with a JSON fallback; deeper per-tool schema/adapter registries remain.
2. Broaden AUTO coverage and safety tests around the new structured `confirm|adjust|replan|clarify|halt` path.
3. Finish the remaining modeling productization with the native/export edge-case fixtures and broader domain-policy coverage. Real business-material smoke now exists for conversational modeling: source-dir business columns and feature dictionary are auto-detected, preserved through existing and auto-generated split frames, and verified through report/delivery outputs. JSON and Markdown model cards, approval packages, monitoring policies, prior Champion comparison, PMML/native delivery metadata, PMML-backed challenger/backtest packages, stricter scorecard/monotonicity policy fixtures, native LightGBM/XGBoost Booster and MLP/DNN skip coverage, and the current Playwright Chromium smoke now exist.
4. Add a deeper DB+file `UnitOfWork` over the staged artifact stores and recovered `StepRunLedger`.
5. Continue deeper `api.py`, `db.py`, and `app.js` decomposition after the first stable splits.
6. Keep CI and local `scripts/check` green while making the remaining large refactors.

## 2026-06-29 Follow-up Review Implementation Update

The follow-up review confirmed several earlier concerns were still real and fixed them in the current working tree:

- Persisted plan step outputs/evidence are now redacted before storage; sensitive JSON-style keys and bearer tokens are masked while alphanumeric business ids such as join plan ids are preserved.
- Stale structured dedup controls now require `expected_step_id`; old or missing dedup gate tokens are rejected instead of applying to the current gate accidentally.
- Multi-file modeling now reuses the C1 file-role/target confirmation gate before creating the `modeling_with_join` plan. The previous behavior guessed anchor/features from heuristics and immediately started modeling.
- PMML export now receives the explicit modeling target column, so a sample-weight column that appears before the label cannot be misidentified as the PMML target.
- Automatic experiment selection now prefers delivery-ready candidates when any candidate supports the requested post-training path, instead of choosing solely by metric.
- `post_training_action` has a dedicated renderer so export/handoff success, skip, and reason states are visible in the conversation.
- `AUTO` gate formatting now includes modeling setup controls such as sample-weight candidates.
- Transactional artifact rollback now restores a previous final file if promotion already replaced it and a later failure occurs.
- Step-run recovery now handles the crash window where a succeeded run/output version exists but the step row has not yet been updated.
- Create-task tier values now match backend capability tiers (`autonomous`, not the stale `aggressive` value), and the create dialog follows the globally selected tier by default.
- CI now installs from `uv.lock`, uses a real diff range, and `scripts/check` falls back to checking `HEAD` when the configured diff base is unavailable.
- `scripts/release_push.py` now accepts supported prerelease tags such as `V2.0.0-alpha.1` and writes Python metadata in PEP 440 form such as `2.0.0a1`.
- Continuation update after commit `0b54507b`:
  - The real `app.js` workspace shell now imports and mounts `artifact_view.js`; right-rail plan steps with `output_ref` render a `查看输出` action into `#artifactPanel`, so dataset/metrics/artifact refs are no longer only covered by isolated module tests.
  - AUTO decisions now pass a deterministic safety policy before reaching `PlanDriver`: `adjust` may only operate controls explicitly declared by the current `GateEnvelope`; undeclared parameters such as expensive tuning changes are converted to `halt`.
  - Screen gates now explicitly declare a `selection` control, so AUTO can safely return feature selections only at gates that expose that control.
  - Draft code safety scanning now includes AST checks for dangerous imports and file APIs such as `Path.read_text()`, closing a previously noted draft sandbox escape before subprocess execution.
  - Failure messages now carry a richer `FailureEnvelope`: retryability, editable input JSON schema with current defaults, stale token, and the exact failed/downstream steps that a retry will reset.
  - Step evidence now records tool version, manifest hash, source dataset refs, artifact refs, input hash/summary, parent output refs, seed, and renderer hint in one persisted envelope.
  - AUTO adjust now has a deterministic low-risk control allowlist: undeclared controls still halt, and declared high-risk controls such as expensive tuning budgets or post-training handoff/export actions also halt for human review.
  - Failed plan steps returned by the plan API now include `failure_envelope`; the main right rail and modular V2 plan view both use its editable input defaults and reset-scope metadata in retry panels, and now render schema-derived retry fields for string/number/integer/boolean/enum/object/array inputs while preserving the JSON textarea fallback.
  - Modeling setup now diagnoses sample-weight candidates for numeric validity, missingness, range, mean, feature exclusion, and exposes those diagnostics in gate metadata, front-end controls, renderer tables, and AUTO prompts.
  - Step confirmation is now guarded at the repository write boundary: only steps still persisted as `awaiting_confirm` can set `confirmed = 1`; stale or non-gate confirm calls return API 409 and record the spawned job as failed instead of silently mutating a pending/failed step.
  - AUTO safety policy now also reads `GateEnvelope.risk_flags` and `downstream_reset_policy`; `adjust`/`replan` decisions halt when a gate declares handoff/export/destructive/manual-review risk or a broad reset scope/count, even if the requested control itself is otherwise low risk.
  - The modeling setup gate now renders a real setup panel rather than only a weight picker: it stays visible without weight candidates, shows target type, selected algorithms, primary tuning recipe, candidate feature count, tuning trials, metric policy, algorithm/PMML support, split/OOT counts and warnings, feeds the same context into AUTO prompts, and its renderer plus setup-adjust controller are extracted to `static/js/v2/modeling_setup_panel.js`. Target type, algorithms, tuning trials, and sample weight can now be adjusted through structured controls; structural changes require a human override reason and rerun the spec plus downstream screening. Family-mismatch recipes such as `lgb_regressor`/`lgb_multiclass` are selectable when changing target type, and invalid target/algorithm combinations are blocked before submission.
  - Model comparison, final experiment selection, and post-training delivery outputs now carry a shared `model_delivery` metadata payload; `static/js/v2/model_delivery_panel.js` renders selected experiment, candidate metrics, stability/feature-count/calibration/delivery business signals, model-report readiness/section coverage, PMML/native/handoff readiness, action status, artifacts, and skip/unsupported states in both manual analysis and agent-message views.

Items confirmed still not complete and therefore still part of the plan:

- OS-level sandboxing is not complete. Plugin/draft tools and ordinary notebook execution have subprocess/resource-limit isolation; live keep-alive notebook appended-cell execution is now disabled by default and requires an explicit policy plus V1 validation injected-cell allowlist, but that live path still needs either worker RPC redesign or removal from safety-critical flows.
- DB plus filesystem writes are staged and recoverable for many high-risk artifact paths. `ArtifactUnitOfWork` now supports SQLite connection-scoped finalization, and join result registration, data_ops clean/dedup derived dataset registration, modeling prepare outputs, and reject-inference augmented samples use that boundary; a full repository-wide `UnitOfWork` is still not complete across all tool execution.
- `api.py`, `db.py`, and `app.js` remain large after the first splits. `turn_handlers.py`, `api_data_payloads.py`, `agent_memory/api_support.py`, `routers/agent_memory.py`, `db_schema.py`, `theme.js`, `task-types.js`, `create-task-dialog.js`, renderers, gate payloads, and adjust specs are good first cuts, not the final architecture.
- The frontend now has richer modeling setup and model delivery/readiness panels, including editable setup controls, setup override guidance, core business signals, scorecard/monotonicity/approval policy signals, executable policy-decision status/violations/override reasons, lightweight Node DOM-structure smoke, and optional Playwright desktop/mobile smoke for setup/delivery panels, the real welcome shell, modeling create dialog, plan rail, screen selector table, light/dark startup paths, and a real app-shell modeling task with task list, plan rail, and delivery metadata. The modeling pack now enforces `select_experiment.selection_policy` for PMML/handoff, scorecard preference, monotonicity, feature count, OOT PSI, and override reasons. `post_training_action` writes JSON evidence, human-readable Markdown approval packages, JSON/Markdown model cards, versioned monitoring-policy JSON/Markdown artifacts, prior Champion/Challenger comparison JSON/Markdown artifacts from explicit references or earlier selected experiments, and can create a PMML-backed challenger/backtest validation task package with JSON/Markdown plan evidence. Calibrated models now explicitly state when PMML does not include the calibration layer while validation handoff notebooks load `calibration.joblib`. Scorecard/monotonicity policy now rejects partial scorecard direction evidence and all-zero tree constraints, native LightGBM/XGBoost Booster plus MLP/DNN artifacts now have G5 skip/model-card coverage, and the real business-material smoke now verifies source business columns/dictionaries through existing-split and auto-split report/delivery outputs. The remaining gap is native/export edge-case and broader domain-policy coverage.
- The visual token system is mostly complete for the primary workspace surfaces: semantic task/surface/status tokens, metric/report/KPI/ROC chart tokens, modeling setup/delivery state tokens, generic download action tokens, send-stop composer tokens, disabled/focus tokens, and user-message bubble tokens now exist. Remaining local palettes are mostly older non-modeling utility/settings surfaces.
- Broader AUTO safety fixtures now cover declared destructive/export/handoff/manual-review/approval risk flags and wide downstream resets. More fixtures are still useful for future extracted modeling setup panels and additional business-domain controls.

Current merge stance: this branch is not "V2 complete" yet. It can become an intermediate PR only after full `scripts/check` from the final tree and a PR description that explicitly lists the remaining items above. Direct merge to `main` as a finished V2 release is still too risky.

## Status Against Earlier 10 Recommendations

| # | Item | Status | Evidence / Gap | Next Action |
|---|---|---|---|---|
| 1 | AUTO structured decisions | Mostly done | `auto_drive.py` now parses `confirm|adjust|replan|clarify|halt` with params, selection, dedup strategies, replan goal, clarifying question, confidence, current-gate allowed action enforcement, and a low-risk control allowlist that blocks expensive tuning and delivery actions even if declared by a gate. Safety fixtures now cover destructive/export/handoff, wide downstream resets, strategy/vintage manual-review or approval markers, and the `n_trials`-only modeling setup stale-token path. | Add more fixtures for future extracted modeling setup gates and frontend stale-control paths. |
| 2 | Modeling business lifecycle | Partial | `choose_modeling_spec`, `configure_tuning`, `select_experiment`, and `post_training_action` are now tools/template steps; `TrainingDataset` caching is wired for multi-recipe train; the modeling setup panel shows and edits target type, algorithms, tuning budget, and sample-weight controls with override reasons, target/algorithm family guards, and risk guidance for structural changes. Model comparison/final-selection/post-training outputs now render through a dedicated delivery readiness panel, including report section coverage, stability/calibration/feature-count/delivery signals, scorecard/monotonicity/approval policy signals, executable policy-decision status, violations, override reasons, and prior Champion comparison status. `select_experiment.selection_policy` now makes those final-model policy signals executable and marks the selected experiment for later Champion resolution: default delivery templates require PMML and validation handoff, hard policy can require scorecard/monotonicity/feature-count/PSI limits, and override requires a reason; stricter policy fixtures now reject partial scorecard monotonic directions, all-zero tree monotone constraints, missing feature-count/OOT-PSI evidence when those policy limits are requested, and now prefer weighted OOT PSI evidence when sample-weight metrics exist. `post_training_action` now writes JSON evidence, human-readable Markdown approval packages, JSON/Markdown model cards, versioned monitoring-policy JSON/Markdown artifacts with target-type-aware threshold checks, optional prior Champion/Challenger comparison JSON/Markdown artifacts from explicit references or earlier selected experiments, exposes the Markdown artifacts in readiness/artifact lists while preserving JSON for machines, can create a PMML-backed challenger/backtest validation task package with copied sample/model/PMML/notebook/dictionary material plus JSON/Markdown plan evidence, and flags calibrated-score PMML limitations in capabilities/readiness/model cards/approval packages. Malformed scorecard WOE payloads now fail PMML readiness during capability precheck and skip delivery actions cleanly with model-card limitations instead of failing during export. Lightweight DOM smoke plus optional Playwright smoke cover setup/delivery, welcome/create, plan rail, screen table, and a real app-shell modeling task with delivery metadata. Conversational API smoke now covers real business materials: source-dir business columns and feature dictionaries flow into report sections and delivery artifacts, including no-split samples that require setup-time auto split. | Expand remaining broader domain-policy fixtures before calling the workflow complete. |
| 3 | PlanDriver decomposition | Partial | Tool output renderers moved to `marvis/agent/renderers.py`; structured gate payload builders moved to `marvis/agent/gate_payloads.py`; gate dependency rendering moved to `marvis/agent/gate_adapters.py`; gate response/control validation moved to `marvis/agent/gate_response_adapter.py`; basic adjust parameter specs moved to `marvis/agent/adjust_specs.py`; adjust/replan execution, downstream reset logic, screen selection, and join dedup apply moved to `marvis/agent/gate_execution_adapter.py`; plan inspection helpers and driver turn DTOs now live outside `PlanDriver`. `PlanDriver` still owns the main turn loop, instruction routing handoff, and message composition. | Continue moving message composition into dedicated composers where it reduces coupling. |
| 4 | V2 turn orchestration out of `api.py` | Mostly done | Driver turn handlers for data join, feature analysis, modeling, strategy, vintage now live in `marvis/agent/turn_handlers.py`; data/join payload and preview masking helpers now live in `marvis/api_data_payloads.py`; Agent Memory payload/context/audit helpers now live in `marvis/agent_memory/api_support.py`, and `/api/agent-memory...` routes plus message-memory-reference lookup live in `marvis/routers/agent_memory.py`; `api.py` keeps the validation-agent wrappers plus LLM/tier resolution. | Continue moving validation-agent stage orchestration and data upload/join routers out of `api.py`. |
| 5 | Modeling data loaded once | Partial | `TrainingDataset` adapter and read-count tests exist for `train_models`; report generation now reuses the same adapter as a cached report/scoring backend after scored dataset creation, so score bands, OOT binning, univariate rows, split profile, and product-removal stress can share the already-scored frame. Some non-report auxiliary paths still read independently. | Expand adapter opportunistically to remaining auxiliary paths where profiling shows repeated reads. |
| 6 | Evidence versioning | Mostly done | `EvidenceEnvelope` is stored beside raw output and includes input summary/hash, parent refs, source dataset refs, artifact refs, tool version, manifest hash, seed, and renderer hint; raw output compatibility is preserved. Running step-runs now recover persisted outputs or finalize as interrupted after restart. `ArtifactUnitOfWork` now gives artifact promotion plus SQLite connection-scoped DB/audit commit semantics for the join result path, data_ops clean/dedup derived dataset registration, modeling prepare outputs, and reject-inference augmented samples; promoted parquet files roll back when connection-scoped dataset insertion or audit write fails. | Expand the DB+file `UnitOfWork` to remaining multi-write modeling/reporting/artifact tools and keep expanding domain-specific lineage where tools expose richer refs. |
| 7 | Sample-weight gate | Done | Backend detects/validates candidates; create dialog now distinguishes no weight vs explicit column; the extracted modeling setup panel renders detected candidates, diagnostics, target/split/tuning context, risk guidance, and posts structured setup adjusts that rerun `choose_modeling_spec` and downstream screening. Structural setup edits now require an override reason. Post-training now writes structured sample-weight governance into model-card, approval-package, and monitoring-policy JSON/Markdown artifacts, including approval review items and monitoring defaults for availability/non-positive-rate/distribution drift. | Keep regression coverage green as modeling delivery evolves. |
| 8 | Frontend task workspace split | Partial | Some V2 modules exist; theme handling is now in `static/js/theme.js`; create-dialog task type definitions live in `static/js/task-types.js`; create-dialog behavior/payload/material upload/run-mode/algorithm/sample-weight/report-default control now lives in `static/js/create-task-dialog.js`; plan fetch/cache/retry/render, status mapping, right-rail artifact preview wiring, and plan-step lookup now live in `static/js/v2/plan_rail_controller.js`; agent conversation/timeline pure helpers now live in `static/js/agent-conversation-view.js`; manual driver analysis branching and latest screen-gate interactivity live in `static/js/v2/driver_manual_analysis.js`; generic manual gate confirm rendering/submit handling lives in `static/js/v2/driver_gate_confirm.js`; task selection storage, result-scroll storage, and greeting rules now live in `static/js/task-workspace-state.js`; modeling setup renderer/controller live in `static/js/v2/modeling_setup_panel.js`; feature-screen renderer/controller live in `static/js/v2/screen_gate_controller.js`; data-join C1/dedup gate renderer/controllers live in `static/js/v2/join_gate_controller.js`; task/welcome tones, metric/report/KPI/ROC chart palettes, and modeling setup/delivery state palettes use semantic tokens. `app.js` still owns transcript DOM mounting and workspace DOM state. | Finish transcript DOM extraction, then extract the DOM-owning `TaskWorkspace`. |
| 9 | PMML manifest contract | Done | Manifest now advertises `lr/lgb/xgb/scorecard`, matching current PMML-supported list. | Keep regression test. |
| 10 | CI gate | Done | `.github/workflows/ci.yml` and `scripts/check` exist; `docs/versioning.md` references the local gate. | Keep full CI green after remaining refactors. |

## Code Review Findings

### P0: Merge And Release Risk

1. Automated CI was missing; it is now added.
   - Impact: every PR/merge relies on local discipline; large refactors can silently break frontend syntax, pytest groups, or docs formatting.
   - Implemented: `scripts/check` plus `.github/workflows/ci.yml` now run `git diff --check`, ruff, `node --check`, and pytest; `docs/versioning.md` documents the local command.
   - Remaining acceptance: CI must pass on the final committed branch after all remaining refactors.

2. Current branch has many uncommitted changes.
   - Impact: final review and release risk are hard to reason about unless changes are staged by topic.
   - Fix: before merge/release, group changes into explicit commits: runtime/audit, modeling, frontend, docs/CI.
   - Acceptance: `git status --short` is clean after commit, then smoke tests are rerun from the committed tree.

### P1: Agent Intelligence And Gate Contracts

1. AUTO is now structured, but needs deeper safety coverage.
   - Current behavior: `auto_drive.py` accepts bounded `confirm|adjust|replan|clarify|halt` decisions and only executes actions allowed by the current gate envelope.
   - Remaining problem: AUTO can now carry structured fields, but the policy layer still needs broader fixtures proving which adjustments are safe, bounded, and non-destructive.
   - Implemented:
     - Define `AutoDecisionV2` with `action`, `reason`, `params`, `selection`, `dedup_strategies`, `replan_goal`, `clarifying_question`, `confidence`.
     - Allow only actions present in the current `GateEnvelope.allowed_actions`.
   - Remaining:
     - Add safety policy: AUTO may adjust low-risk thresholds within declared bounds; destructive or broad changes require halt/clarify.
   - Tests:
     - Parse valid/invalid JSON.
     - Disallow undeclared actions.
     - Verify AUTO can adjust screen thresholds and halt on high-risk actions.

2. Gate metadata now has a typed base, but adapters are still incomplete.
   - Current behavior: gate messages still carry bespoke metadata such as `tables`, `screen`, `dedup`, and `output_refs`, plus a `gate_envelope` typed envelope.
   - Problem: backend, frontend, and AUTO do not share one typed contract for allowed actions, stale controls, retryability, and rendering.
   - Implemented: introduce `GateEnvelope`:
     - `schema_version`, `kind`, `target_step_id`, `stale_token`.
     - `source_output_refs`, `allowed_actions`, `controls`, `render_blocks`.
     - `risk_flags`, `retry_policy`, `downstream_reset_policy`.
   - Tests:
     - Snapshot envelopes for plan overview, join C1/C2, feature screen, modeling setup/tune/train/compare/report, strategy, vintage.
     - Frontend stale-gate tests for screen, dedup, and generic confirm.

3. PlanDriver decomposition has started, but is not finished.
   - Current behavior: the driver loop and adjust/replan logic remain in `plan_driver.py`; markdown/table renderers live in `marvis/agent/renderers.py`, and screen/dedup gate payload builders live in `marvis/agent/gate_payloads.py`.
   - Problem: every new domain step increases risk of regressions in unrelated tasks.
   - Implemented:
     - `marvis/agent/gates/contracts.py`: `GateEnvelope`, `GateAction`, `GateControl`.
     - `marvis/agent/renderers.py`: tool output to render blocks.
   - Remaining:
     - expand `marvis/agent/gate_adapters.py` into per-tool/per-step gate adapters.
     - expand `marvis/agent/adjust_specs.py` into tool/step schema-driven specs.
     - Leave `PlanDriver` as orchestration: start, resume, route instruction, persist messages.
   - Acceptance: `PlanDriver` no longer imports task-specific screen/dedup/model renderer details.

4. Failure and retry UX is not a first-class contract.
   - Current behavior: failure transcript is mostly plain text with limited metadata; plan rail has retry UI.
   - Problem: user and AUTO need one consistent contract for what is retryable, what inputs can be edited, how typed controls map back to JSON, and what downstream steps reset.
   - Fix: add `FailureEnvelope` with `failed_step_id`, `error_kind`, `retryable`, `editable_input_schema`, `suggested_actions`, `stale_token`, `downstream_reset`.
   - Tests: force tool failure, edit retry inputs, verify downstream reset and recovered completion.
   - Current update: modular V2 plan view and main plan rail render schema-derived retry fields and submit typed values before falling back to raw JSON; failure envelopes now source `error_kind` from the latest failed/interrupted step-run ledger row instead of defaulting every failure to `execution`.

### P1: Evidence, Transactions, And Runtime Safety

1. Tool side effects now have a recoverable run ledger, but not full DB+file transactionality.
   - Current behavior: executor records a `plan_step_runs` attempt before invocation, finalizes it after output storage, and startup/run recovery now closes in-flight step runs. If a persisted output version exists for the current running attempt, recovery attaches the latest output ref, marks the run succeeded, and completes post-checks without rerunning the tool. If no output exists, recovery marks the run interrupted and fails the step for explicit retry.
   - Risk: the system is recoverable, but file side effects, output version storage, step state, and run finalization are still not one atomic unit across SQLite plus filesystem.
   - Implemented:
     - Add `step_runs` table with `run_id`, `step_id`, `attempt`, `started_at`, `tool_result_ref`, `side_effects`, `finalized_at`, `error_kind`.
     - Record run start before invocation.
   - Remaining:
     - Finalize output version, step state, and run state in one SQLite transaction where possible.
     - Continue expanding `ArtifactUnitOfWork.finalize_with_connection` to additional staged file plus DB commit paths.
     - Surface interrupted runs in the plan rail as deterministic retry/repair state.
   - Tests: done for crash after output ref before step update, no-output interruption, stale output after reset, and no unsafe replan/rerun on recovered running failure.

2. File and DB writes still need recovery semantics, but the main artifact paths now use staged promotion.
   - Current state: join execution output, data_ops clean/dedup derived outputs, modeling derived parquet/model/meta/PMML/calibration outputs, report scored parquet output, final xlsx reports, plugin install/promote directories, and validation handoff materials now use staged file or directory promotion.
   - Fix:
     - Done for join execution output: add `TransactionalArtifactStore` with stage, promote, rollback, and orphan cleanup; `JoinEngine.execute_join_plan` writes to `.staging` and promotes only the final artifact.
     - Done for data_ops derived outputs: `clean_format` and `dedup_rows` write parquet files through staging, register derived datasets on a shared SQLite connection boundary when available, and roll back promoted parquet files if connection-scoped dataset insertion fails.
     - Done for modeling derived parquet outputs: `prepare_modeling_frame` / `make_split` and `reject_inference` write through staging, register datasets/audits on a shared SQLite connection boundary when available, and roll back promoted parquet files if registration/audit fails.
     - Done for report outputs: `model_report_scored.parquet`, `render_model_report`, and `render_minimal_model_report` write through staging.
     - Done for model artifacts: native binaries, model meta, PMML export, and calibration payloads write through staged files with rollback on writer/validation failure.
     - Done for directory artifacts: plugin install, promoted draft plugin directories, and validation handoff materials use `TransactionalDirectoryStore` with backup restore on DB/audit failure.
     - Done for startup reconciliation: `create_app` runs artifact recovery and stores the report in `app.state.artifact_recovery_report`; orphan `.staging` directories are removed, plugin backups are restored or cleaned by DB checksum, and validation handoff material directories are reconciled against validation task `source_dir`.
     - Done for step-run reconciliation: executor recovery finalizes running step attempts as succeeded when the current run has a persisted output version, or interrupted when no output was persisted.
     - Done for SQLite-scoped UOW slices: `ArtifactUnitOfWork.finalize_with_connection` promotes staged artifacts, runs DB writes inside a repository connection context, rolls back promoted artifacts if the callback or DB commit fails, lets join execution register result datasets/plan status/audit on the same SQLite connection, and now covers data_ops clean/dedup plus modeling prepare/reject-inference derived dataset registration.
     - Remaining: expand connection-scoped DB+filesystem unit-of-work semantics to broader multi-write tool execution.
   - Tests: done for store promote/rollback/orphan cleanup, sibling staged-file promotion, directory backup restore, writer failure rollback, plugin/draft audit-failure rollback, validation handoff audit-failure rollback, model artifact staging, app startup recovery, connection-scoped artifact+DB commit, connection failure rollback, join-result transaction rollback, data_ops derived parquet rollback on connection-scoped dataset insertion failure, and modeling prepare/reject-inference rollback on connection-scoped dataset/audit failure.

3. Evidence output refs are versioned but not semantically complete.
   - Current behavior: `plan_step_output_versions` preserves versions and refs.
   - Gap: missing normalized input snapshot, dataset ids, artifact paths, seed, parent refs, tool/plugin identity, manifest hash, and renderer hints.
   - Fix:
     - Add `EvidenceEnvelope` while preserving raw-output compatibility.
     - Store `input_hash`, `input_summary`, `source_dataset_refs`, `artifact_refs`, `parent_output_refs`, `tool_name`, `tool_version`, `manifest_hash`, `random_seed`.
   - Tests: v1 raw output can still load; v2 evidence can drive renderer and audit.

4. Notebook/plugin sandboxing is not OS-level.
   - Current state: plugin and draft tools run in one-shot subprocess workers with resource limits, timeout kill, network guard, process-spawn guard for non-builtin code, and audited stdout/stderr tails. Ordinary full-notebook execution now has an optional isolated worker with parent timeout kill and artifact preservation. The live keep-alive notebook kernel path still exists for PMML/reproducibility appended cells; appended-cell execution is opt-in per session, requires an explicit policy plus generated V1 validation cell-kind allowlist, and is protected by RSS monitoring/interrupt/shutdown rather than full subprocess session RPC.
   - Fix:
     - Done for plugin/draft tools: run in subprocess with memory/CPU/file-size limits, restricted worker environment, local-only network by default, and `process:spawn` required before untrusted non-builtin code can start child processes.
     - Done for ordinary notebook execution: add `marvis.notebook_worker`, `run_notebook(..., isolated=True)`, worker error propagation, and parent timeout artifact preservation.
     - Remaining for live notebook sessions: either replace keep-alive kernel mutation with a worker RPC protocol, or split PMML/reproducibility appended-cell work into explicit non-live notebook/tool steps so the whole validation flow can use isolated execution.
     - Add slow/OOM integration tests at the pipeline/job layer once the live-session boundary is removed or explicitly downgraded.

### P1: Modeling Workflow And Business Closure

1. G2 algorithm/task selection now has a typed backend step and editable UI controls.
   - Current behavior: `choose_modeling_spec` normalizes target type, recipe family, eligible/disabled algorithms, metric policy, sample-weight policy, tuning budget, fixed params, and exposes a rendered gate table before feature screening.
   - Current update: sample-weight candidates now render inside a fuller setup panel; changing the selected candidate posts structured `adjust_params.sample_weight_col` and reruns the modeling spec plus downstream screening. The panel also shows target type, split/OOT diagnostics, algorithm family, PMML support, tuning budget, metric policy, and warnings. The create dialog also distinguishes "no weight" from an explicit column.
   - Current update: the same panel now lets users change target type, algorithm set, and tuning trials through structured controls. Target/algorithm/trial changes require an explicit reason, post `adjust_params`, and rerun `choose_modeling_spec` plus downstream screening under the current gate token. Target/algorithm family mismatches are blocked in the UI while still exposing regression/multiclass recipes when switching target type.
   - Current update: model comparison, final selection, and post-training delivery now share a structured delivery-readiness panel for candidate metrics, selected experiment, PMML/native/handoff support, model-card/approval/monitoring artifacts, action status, artifacts, and unsupported/skip reasons.
   - Current update: the new setup/delivery surfaces have lightweight DOM smoke plus optional Playwright smoke for setup/delivery, welcome/create, plan rail, screen table, and a real app-shell modeling task that loads task list, plan, and delivery metadata.
   - Remaining problem: real business-material smoke now exists; remaining native/export and broader domain-policy fixtures still need coverage.
   - Remaining fix:
     - Done: expand scorecard/monotonicity/approval signals into enforceable scorecard policy gates, including partial scorecard-direction and all-zero tree-constraint failures.
     - Keep the typed modeling spec as the single downstream contract.

2. G3 tuning needs a typed control surface.
   - Current behavior: tuning exists, but configuration is not a first-class gate.
   - Fix:
     - Add `configure_tuning` gate with skip/tune choice, search space, metric, time budget, sample weight usage, random seed, and agent recommendation.
     - Allow AUTO to propose bounded changes, but require human confirmation for expensive searches.

3. G4 model selection is automatic.
   - Current behavior: `train_models` still computes a best experiment, and `select_experiment` now stores/announces the final chosen experiment with candidate readiness context and executable delivery/model-risk policy checks.
   - Problem: production credit/risk modeling still needs richer business choice controls beyond the current PMML/handoff/scorecard/monotonicity/feature-count/PSI policy set.
   - Fix:
     - Done: add `select_experiment` tool/gate and structured delivery panel.
     - Done: surface stability, calibration, feature count, and delivery readiness in comparison/selection panels.
     - Done: enforce PMML/handoff, scorecard preference, monotonicity, feature-count, OOT PSI, and override-reason requirements in `selection_policy`.
     - Done: add versioned monitoring-threshold policy artifacts with target-type-aware checks and configurable thresholds.
     - Done: add real app-shell Playwright smoke for a modeling task with plan rail and delivery metadata.
     - Done: calibrated-score PMML limitation is explicit in delivery capabilities, PMML readiness reason, model card, and approval package.
     - Done: default selection policy is now target-type-aware; binary remains PMML/handoff strict, while continuous/multiclass flows select a model and surface unsupported delivery actions as skip/limitations rather than failing selection.
     - Done: run real business-material smoke through the conversational modeling API, including business columns, feature dictionary, report sections, delivery artifacts, and no-split auto-split preservation.
     - Remaining: broaden native/export and domain-policy fixtures.

4. G5 post-training closure is not a workflow.
   - Current behavior: report generation and `post_training_action` are workflow steps; PMML/handoff success, skip, and reason states are now visible in a dedicated delivery panel.
   - Fix:
     - Done: add `post_training_action` gate with actions:
       - export `.pkl` native artifact.
       - export `.pmml` when supported.
       - generate model report.
       - hand off to validation.
       - generate JSON and Markdown approval packages.
       - generate JSON and Markdown model cards.
       - create PMML-backed challenger/backtest validation task packages with JSON and Markdown plans.
       - generate versioned monitoring-policy JSON and Markdown artifacts.
     - Done: cover native LightGBM/XGBoost Booster and MLP/DNN G5 delivery close so unsupported PMML/handoff actions skip with explicit limitations and no validation task side effects.
     - Done: cover multiclass end-to-end close so non-PMML/non-handoff candidates do not trigger a false selection-policy failure before report/delivery.
     - Remaining:
       - real business-material smoke is now covered by `test_modeling_business_materials_flow_into_report_and_delivery` and `test_modeling_business_materials_without_split_survive_auto_split`.
       - expand remaining native/unsupported exporter edge-case fixtures beyond the current PMML-compatible, calibrated-model, native Booster, and MLP/DNN paths.
     - Show unsupported PMML states and calibrated-score limitations clearly.

5. `TrainingDataset` adapter is missing.
   - Current behavior: preparation, split, tuning, recipes, report scoring, and artifact schema inference repeatedly read full frames.
   - Fix:
     - Add `TrainingDataset` with cached train/test/OOT frames or lazy references, label, features, weight, schema, split masks, and bounded sample.
     - Update recipes to consume `TrainingDataset` rather than calling `read_frame`.
   - Tests: backend read-count test proves each dataset partition is loaded once per training run.

6. Sample weight support is backend-capable but not user-grade.
   - Current behavior: explicit sample weight works; candidates are detected; create-time input is policy-based; the first modeling gate can choose "no weight" or a detected candidate and rerun the dependent steps.
   - Fix:
     - Done: move detected-candidate choice into the modeling gate.
     - Remaining: validate/display positive numeric values, missingness, leakage risk, and richer business rationale; keep excluding weight from features.

7. Hard-coded metric gates conflict with "no fixed metric target".
   - Current behavior: the modeling template no longer hard-fails on `oot_ks >= 0.3331`; fixed thresholds now live only in generic reviewer/post-check fixtures or configurable modeling policy artifacts.
   - Fix:
     - Done: replace the fixed modeling-template success gate with explicit selection/delivery policy and monitoring-policy artifacts.
     - Remaining: broaden real business-policy fixtures and report language so pass/warn/fail thresholds are always visibly configurable rather than implied by one universal metric target.

8. Reports should surface missing business context.
   - Current behavior: binary reports are richer; non-binary reports are minimal; chat renderer now shows unavailable business sections from `section_status`.
   - Fix:
     - Done: report message shows generated sections, skipped sections, and missing inputs.
     - Add non-binary report sections where applicable.
     - Add business decision summary: threshold, approval recommendation, reject inference status, monitoring plan.

### P2: API And Database Architecture

1. `api.py` remains a bottleneck.
   - Current state: driver turn orchestration for data_join, feature_analysis, modeling, strategy, and vintage has moved to `marvis/agent/turn_handlers.py`; Agent Memory HTTP routes are now in `marvis/routers/agent_memory.py`; memory payload/context/audit support is in `marvis/agent_memory/api_support.py`. `api.py` still owns validation/data/stage job routes and validation-agent compatibility wrappers.
   - Fix order:
     - Done: `marvis/agent/turn_handlers.py`: data_join, feature_analysis, modeling, strategy, vintage.
     - Done: `marvis/api_data_payloads.py`: data/join response payloads and dataset preview masking.
     - `marvis/routers/data.py`: data upload/join routes.
     - `marvis/routers/validation.py`: validation stages and reports.
     - `marvis/routers/agent.py`: agent messages/tasks/driver turns.
     - `marvis/routers/memory.py`: memory endpoints.
   - Acceptance: `api.py` is mostly app-level compatibility glue and imports no domain-heavy execution code.

2. `db.py` is still repository-heavy, but schema/connection setup has started moving out.
   - Fix order:
     - Done: `marvis/db_schema.py`: schema constants/migrations plus connection setup/pragmas/row factory, re-exported from `marvis.db` for compatibility.
     - `marvis/repositories/tasks.py`, `plans.py`, `datasets.py`, `plugins.py`, `drafts.py`, `audit.py`, `modeling.py`.
     - `UnitOfWork`: one transaction-scoped object exposing repositories.
   - Acceptance: new domains do not add hundreds of lines to `db.py`.

3. Cross-repository writes need a common transaction pattern.
   - Current state: several `*_with_audit` helpers exist, but transaction boundaries are bespoke.
   - Fix:
     - Document write categories: DB-only, file+DB, external side effect+DB.
     - Migrate one domain at a time to `UnitOfWork` and artifact staging.
   - First candidate: dataset/join, because it exercises file output plus DB registration and is easier than plugin install.

4. Several list APIs are unbounded or weakly bounded.
   - Fix: repository-level pagination/cursors for tasks, messages, audits, drafts, artifacts.
   - Tests: default limit, max limit, stable ordering, cursor continuation.

### P2: Frontend UX, Visual System, And Product Depth

1. `app.js` decomposition is unfinished.
   - Current state: V2 modules exist, theme handling has moved to `static/js/theme.js`; create-dialog task type definitions/display ordering live in `static/js/task-types.js`; create-dialog behavior/payload/material upload/run-mode/algorithm/sample-weight/report-default logic lives in `static/js/create-task-dialog.js`; plan fetch/cache/retry/render, status mapping, right-rail artifact preview wiring, and plan-step lookup live in `static/js/v2/plan_rail_controller.js`; conversation/timeline pure helpers live in `static/js/agent-conversation-view.js`; manual driver analysis branching and latest screen-gate interactivity live in `static/js/v2/driver_manual_analysis.js`; generic manual gate confirm rendering and submit handling live in `static/js/v2/driver_gate_confirm.js`; task selection storage, result-scroll storage, and greeting rules live in `static/js/task-workspace-state.js`; feature-screen gate rendering and threshold/selection submit handling live in `static/js/v2/screen_gate_controller.js`; data-join C1/dedup gate rendering and submit handling live in `static/js/v2/join_gate_controller.js`. Conversation DOM mounting and workspace DOM state still live in one global controller.
   - Fix order:
     - Done: `CreateTaskDialog` config and controller behavior are extracted; `app.js` keeps the post-create workspace refresh and Agent composer seeding.
     - Done: `PlanRailController` owns plan fetch/cache/retry/render, status mapping, gated actions, artifact preview wiring, and plan-step lookup.
     - Done: `AgentConversationView` pure helpers for timeline stages, frozen snapshot re-anchoring, message bucketing, report-message filtering, and advance-intent detection are extracted.
     - Done: manual driver analysis branching, overview filtering, latest screen-gate interactivity, and renderer injection are extracted to `DriverManualAnalysis`.
     - Done: generic manual gate confirmation button rendering and submit handling are extracted to `DriverGateConfirm`.
     - Done: `TaskWorkspace` pure state helpers for selected-task storage, result-scroll storage, and empty-workspace greeting rules are extracted.
     - Done: feature-screen gate rendering, latest-gate readonly behavior, threshold adjust submit, and feature-selection submit are extracted to `ScreenGateController`.
     - Done: data-join C1 role assignment and dedup strategy rendering/submit handling are extracted to `JoinGateController`.
     - `DriverConversationView`: manual analysis rendering, generic driver confirmation, agent transcript DOM mounting.
     - `TaskWorkspace`: task shell, active task state, right rail coordination.

2. Modeling needs a dedicated setup and analysis surface.
   - Fix:
     - Done: `ModelingSetupPanel` with target/split counts, target type, algorithm choices, detected sample weights, split/OOT warnings.
     - Done: `ModelDeliveryPanel` with selected experiment, candidate metrics, PMML/native/handoff readiness, action state, artifact refs, and unsupported/skip reasons.
     - Screen table as a real modeling selector: threshold sliders plus numeric inputs, `top_k`, sort/filter chips, selected-count summary, leakage override reason, reset-to-proposal.
     - Done: extend model comparison with stability, calibration, feature count, and delivery-readiness business signals.
     - Remaining: native/export and broader domain-policy edge-case coverage.

3. Visual tokens are partial, not a system.
   - Current state: semantic task tones, surface/border/status tokens, welcome/task icon palettes, metric cards, report section tones, KPI cards, PSI bands, ROC chart palettes, modeling setup/delivery status tokens, generic download action tokens, send-stop composer tokens, disabled/focus tokens, and user-message bubble tokens are centralized. Older non-modeling utility/settings areas still contain local palette constants.
   - Fix:
     - Done for task types and core surface/status tokens.
     - Done for metric cards, report section tones, KPI cards, PSI bands, ROC chart lines/axis/grid/legend, and dark/light static parity checks.
     - Done for modeling setup/delivery panel surface, text, signal, readiness, warning, and error tokens.
     - Done for generic download actions, send-stop composer controls, disabled/focus states, and user-message composer bubbles.
     - Replace remaining hard-coded local hex colors in older non-modeling utility/settings surfaces.

4. UX should communicate business readiness, not just execution progress.
   - Fix:
     - Plan rail statuses: "needs decision", "running", "blocked", "ready for handoff", "needs business input".
     - Done: modeling delivery card shows report section coverage, PMML support, validation handoff state, and selected experiment.
     - Clear stale-gate warnings and retry contracts.

5. Accessibility and visual smoke should become test gates.
   - Tests:
     - `node --check` for extracted modules.
     - Static tests for required controls and stale tokens.
     - Optional Playwright smoke now exists for setup/delivery panels, welcome shell, modeling create dialog, plan rail, screen table, desktop/mobile, light/dark startup paths, and a real app-shell modeling task with delivery metadata.

### P2: Performance And Data Scale

1. Full-frame pandas reads remain common.
   - Fix:
     - Use `TrainingDataset` for modeling.
     - Add DuckDB/query-backed helpers for feature screening and large summaries where feasible.
     - Keep bounded samples for UI previews.

2. Expensive jobs are mixed across direct request handlers and background tasks.
   - Fix:
     - Define job policy: quick synchronous route vs background job vs subprocess sandbox.
     - Apply to join, validation stages, modeling train/tune/report, notebook/plugin execution.

3. Add performance regression tests.
   - Tests:
     - Large parquet feature screening smoke.
     - Multi-recipe modeling read-count and runtime smoke.
     - Join match-rate performance smoke.

### P3: Practical Business Problem Solving

1. Add a model approval package.
   - Contents: selected experiment, metrics by split, stability, calibration, reject inference status, excluded features and reasons, sample-weight choice, PMML/native artifact state, validation handoff link, monitoring plan.

2. Add challenger and monitoring workflows.
   - Done: create PMML-backed challenger/backtest validation tasks directly from G5, with copied model/sample/PMML/notebook/dictionary material and JSON/Markdown plan evidence.
   - Done: store monitoring thresholds and drift checks as versioned JSON/Markdown policy artifacts from G5.
   - Done: generate optional prior Champion/Challenger comparison JSON/Markdown artifacts from an existing experiment id, explicit champion metrics, or an earlier selected experiment in the same task; carry the comparison into approval packages, challenger/backtest plans, delivery readiness, and the frontend artifact list.

3. Improve Agent recommendations.
   - Recommendations should cite evidence refs and show tradeoffs:
     - metric gain vs feature count.
     - PMML support vs native model performance.
     - stability vs raw OOT KS/AUC.
     - sample weight use vs data quality.
   - AUTO should propose a bounded action and explain why it is safe.

4. Add domain-specific defaults without hard-coding outcomes.
   - Default recipes and metrics should reflect credit/risk modeling, but thresholds should be configurable policy, not fixed magic numbers.

## Implementation Roadmap

### Phase A: Stabilize Review And CI Gates

Goal: make every later change safer to merge.

Tasks:
- Done: add `scripts/check` as the local canonical command.
- Done: add GitHub Actions CI for `git diff --check`, Python lint, `node --check`, and pytest.
- Done: update docs with current required local checks.
- Done: add this status table that tracks which V2 items are done/partial/not done.

Acceptance:
- CI runs on PR and branch push.
- Local command reproduces CI.
- No implementation phase starts without a passing baseline.

### Phase B: Gate, AUTO, Evidence, Retry Contracts

Goal: turn Agent execution into a typed, inspectable, recoverable loop.

Tasks:
- Done: add `GateEnvelope`, `FailureEnvelope`, and `EvidenceEnvelope`.
- Partial: add output renderers, gate payload helpers, dependency gate adapters, and basic adjust specs; per-tool gate adapters and schema-driven adjust specs remain.
- Done: extend AUTO to structured bounded decisions.
- Partial: add stale-token style `expected_step_id` enforcement for structured screen controls; expand to all gate actions.
- Mostly done: add retry/failure contract metadata, downstream reset behavior, real step-run `error_kind` propagation, and first schema-driven retry fields; deeper per-tool form adapters remain.

Acceptance:
- Existing manual driver flows still work.
- AUTO can safely adjust a declared screen gate in tests.
- Invalid/undeclared AUTO actions halt with a reason.
- Evidence refs remain backward compatible.

### Phase C: Modeling Lifecycle Closure

Goal: make modeling a complete business workflow, not only train/report primitives.

Tasks:
- Done: add G2 modeling spec step/gate (`choose_modeling_spec`).
- Done: add G3 tuning configuration gate.
- Done: add `select_experiment`.
- Done: add G5 `post_training_action` gate.
- Done: add G5 challenger/backtest task package creation for PMML-capable final models.
- Done: add G5 versioned monitoring-policy artifact generation and delivery readiness.
- Mostly done for modeling setup gate: add sample-weight propagation, G2 spec output, create-time no-weight/explicit policy, detected-candidate gate adjust/rerun, target/algorithm/tuning/split/PMML display, editable setup controls with override reasons, AUTO context, extracted setup renderer/controller, setup override guidance, lightweight DOM smoke, optional Playwright smoke for setup/delivery plus workspace surfaces, stricter model-policy fixtures, and real business-material smoke through report/delivery.
- Done for `train_models`: add `TrainingDataset` adapter and read-count tests.
- Done: remove hard-coded `oot_ks >= 0.3331` as a universal success gate.
- Done for chat renderer: improve report renderer with section status and missing inputs; broader report content remains.

Acceptance:
- A user can create a modeling task, confirm algorithms/weights, tune/train, select an experiment, export `.pkl` and `.pmml` when supported, generate report, hand off to validation, and create a PMML-backed challenger/backtest validation task package.
- Tests cover PMML support boundaries, report missing-section visibility, sample-weight propagation, and read-count regression.

### Phase D: Transactional Runtime Hardening

Goal: close remaining crash windows and file/DB inconsistency risks.

Tasks:
- Done: add `StepRunLedger`.
- Done for first slice: add `TransactionalArtifactStore`.
- Done for join execution: migrate join output persistence to staged write, final promote, and rollback on audit/DB failure.
- Done for data_ops derived outputs: migrate `clean_format` and `dedup_rows` parquet outputs to staged write/final promote/rollback.
- Done for modeling derived parquet outputs: migrate `prepare_modeling_frame` / `make_split` and `reject_inference` output datasets to staged write/final promote/rollback.
- Done for report outputs: migrate `model_report_scored.parquet` and final xlsx reports to staged write/final promote.
- Done for modeling model artifacts: migrate native binaries, meta files, PMML exports, and calibration payloads to staged write/final promote/rollback.
- Done for validation handoff materials: migrate material directory activation to `TransactionalDirectoryStore`.
- Done for plugin install/promote paths: migrate zip install and draft promotion directory swaps to `TransactionalDirectoryStore`.
- Done: add orphan reconciliation on startup with `app.state.artifact_recovery_report`.
- Done: recover in-flight `plan_step_runs` by finalizing current persisted-output attempts as succeeded and no-output attempts as interrupted without unsafe reruns.
- Partial: add OS-level subprocess sandbox for plugin/draft tools and ordinary full-notebook execution; untrusted plugin/draft child process spawning now requires explicit `process:spawn`; live keep-alive notebook execution is policy-gated to V1 validation injected cells but still needs a worker RPC redesign or step split.

Acceptance:
- Crash-window tests pass.
- Staged artifacts are cleaned or promoted deterministically.
- OOM/timeout notebook/plugin tests leave a recoverable task state.

### Phase E: Architecture Split

Goal: reduce merge risk and make future domains cheaper to add.

Tasks:
- Done: move driver turn orchestration out of `api.py` into `marvis/agent/turn_handlers.py`.
- Split data/validation/agent/memory routers.
- Partial: split `db.py` schema/connection setup into `marvis/db_schema.py`; repositories still need module extraction.
- Continue introducing `UnitOfWork` semantics and migrate one domain at a time.
- Add pagination to high-volume list endpoints.

Acceptance:
- `api.py` becomes app compatibility and route registration, not domain orchestration.
- `db.py` becomes compatibility exports or a thin package entrypoint.
- New tests assert route registration and repository behavior.

### Phase F: Frontend Workspace And Visual System

Goal: make V2 feel like a professional task workspace rather than a generic chat plus tables.

Tasks:
- Extracted `CreateTaskDialog`: task type definitions/display order live in `static/js/task-types.js`, and controller behavior/payload/material upload/run-mode/algorithm/sample-weight/report-default logic lives in `static/js/create-task-dialog.js`.
- Extracted `PlanRailController`: plan fetch/cache/retry/render, status mapping, plan-step context lookup, and right-rail artifact preview wiring live in `static/js/v2/plan_rail_controller.js`.
- Started `DriverConversationView`: timeline stages, frozen snapshot re-anchoring, message bucketing, report-message filtering, advance-intent detection, manual analysis branching, latest screen-gate interactivity, and generic gate confirmation live outside `app.js`; transcript DOM mounting remains in `app.js`.
- Started `TaskWorkspace`: selected-task storage, result-scroll storage, and empty-workspace greeting rules live in `static/js/task-workspace-state.js`; task shell and active task DOM state remain in `app.js`.
- Extracted `DriverManualAnalysis`: manual mode overview filtering, analysis-section branching, latest screen-gate interactivity, and injected renderer orchestration live in `static/js/v2/driver_manual_analysis.js`.
- Extracted `DriverGateConfirm`: generic manual gate confirmation button rendering and POST handling live in `static/js/v2/driver_gate_confirm.js`.
- Extracted `ScreenGateController`: feature-screen table rendering, latest-gate readonly behavior, threshold-adjust submit, and selected-feature submit live in `static/js/v2/screen_gate_controller.js`.
- Extracted `JoinGateController`: C1 file-role assignment and dedup strategy rendering/submit handling live in `static/js/v2/join_gate_controller.js`.
- Add `ModelingSetupPanel` and model comparison/post-training panels.
- Mostly done: implement semantic task/surface/status, metric/report/chart, modeling setup/delivery state tokens, generic download/composer action tokens, and extract theme controller; older non-modeling utility/settings tokens remain.
- Expand screen selector controls beyond the extracted controller: sliders, `top_k`, filters, reset, and override reasons.

Acceptance:
- `app.js` shrinks materially and no longer owns all task responsibilities.
- Frontend tests cover extracted modules and modeling gate controls.
- Playwright smoke verifies setup/delivery, welcome shell, modeling create dialog, plan rail, screen table, desktop/mobile, light/dark startup paths, and a real app-shell modeling task with task list, plan, and delivery metadata.

### Phase G: Business Deliverables And Agent Quality

Goal: improve actual business usefulness and decision quality.

Tasks:
- Add model approval package output.
- Add challenger/monitoring workflow entrypoints.
- Add Agent recommendation templates that cite evidence refs.
- Add policy-driven thresholds for modeling acceptance.
- Add eval fixtures for join C2, feature screening, modeling selection, PMML handoff, and retry recovery.

Acceptance:
- Generated outputs answer "can I use this model in business?" with evidence and next actions.
- Agent recommendations are traceable to concrete output refs.

### Phase H: Final Review Before Merge

Goal: decide whether the branch can merge to `main`.

Checklist:
- `git status --short` clean except intentional release artifacts.
- `scripts/check` passes.
- Full pytest passes in the intended env.
- Frontend smoke passes.
- Manual smoke:
  - Create data join task.
  - Create feature analysis task.
  - Create modeling task through G2-G5.
  - Export PMML/PKL for supported algorithms.
  - Handoff selected model to validation.
  - Force a retryable failure and recover.
- Review docs updated with remaining known limitations.
- Release/tag flow documented and chosen before merge.

## Suggested Commit Breakdown

1. `ci: add v2 check workflow`
2. `agent: introduce gate and evidence envelopes`
3. `agent: add structured auto decisions`
4. `modeling: add lifecycle gates and experiment selection`
5. `modeling: add training dataset adapter`
6. `runtime: add step run ledger and artifact staging`
7. `api: extract v2 turn handlers`
8. `db: split schema connection and repositories`
9. `frontend: extract task workspace controllers`
10. `frontend: add modeling controls and visual tokens`
11. `docs: update v2 completion status and merge checklist`

## Open Decisions

- AUTO autonomy level: default recommendation is "bounded low-risk adjustments only"; high-cost tuning, destructive resets, handoff, and export should require user confirmation.
- Notebook/plugin sandbox mechanism: default recommendation is subprocess with resource limits first; containerization can be evaluated later if local developer friction is acceptable.
- PMML promise: keep `.pkl` as the source-of-truth native artifact for every algorithm; `.pmml` is a compatibility artifact only when exporter and validation loader are proven.
- Visual redesign depth: default recommendation is semantic token consolidation plus task workspace polish, not a full product redesign before runtime contracts are stable.

## Merge Risk Assessment

Current state should be treated as not ready for direct `main` merge if the goal is "V2 complete". It may be mergeable as an intermediate PR only if the PR description clearly labels the remaining work above and final CI/local checks pass from the committed tree.

The highest risks before a production-style merge are:

- CI gate exists, but full CI/full pytest still must pass from the final committed tree.
- AUTO can apply structured decisions, but broader safe-remediation policy fixtures are still needed.
- Modeling final handoff is now workflow-capable for G2-G5 backend steps, sample-weight/setup adjustment has a working gate with algorithm-family guards and risk guidance, delivery readiness has a dedicated UI with core business signals, scorecard/monotonicity/approval policy signals, validation handoff, model cards, approval packages, monitoring-policy artifacts, prior Champion comparison artifacts, and challenger/backtest task packages, and real business-material smoke now verifies the report/delivery chain; remaining export/domain edge-case fixtures still remain.
- Runtime crash windows are recoverable for staged artifacts and running step attempts; join result registration, data_ops clean/dedup derived registration, and modeling prepare/reject-inference derived registration now have SQLite connection-scoped DB+filesystem boundaries, but this is not yet repository-wide.
- `api.py`, `db.py`, and `app.js` are still large, but the first stable splits have landed (`turn_handlers`, `api_data_payloads`, `agent_memory/api_support`, `routers/agent_memory`, `db_schema`, `theme.js`, `task-types.js`, `create-task-dialog.js`); remaining risk is deeper router/repository/workspace-controller extraction.
- OS-level sandboxing for notebook/plugin execution is not complete; live appended notebook execution is policy/allowlist gated, not yet subprocess-session isolated.

## Definition Of Done For V2 Complete

V2 can be called complete only when:

- All core task types run through typed gates and evidence envelopes.
- AUTO can perform bounded structured actions and stops safely outside declared permissions.
- Modeling covers G2-G5 with explicit user or AUTO decisions, selected experiment, PMML/PKL/report/handoff closure.
- Runtime side effects are recoverable through step-run ledger recovery and artifact staging, with any remaining non-atomic boundaries explicitly documented.
- API, DB, and frontend controllers are split enough that new domains do not expand monolith files.
- CI, focused modeling API smoke, and real business-material smoke are green from a clean committed tree.
- Final review documents remaining limitations as product choices, not unfinished core architecture.
