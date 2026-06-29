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
  - `tests/test_frontend_shell_static.py::test_app_entry_is_split_into_frontend_modules tests/test_frontend_shell_static.py::test_unselected_workspace_shows_centered_welcome_only tests/test_frontend_screen_table.py tests/test_frontend_static_v2.py::test_frontend_uses_v2_task_actions_only`: `15 passed` after adding semantic visual tokens and extracting theme handling to `static/js/theme.js`.
  - `tests/test_db.py tests/test_orch_db.py tests/test_plugin_db.py tests/test_modeling_db.py tests/test_strategy_db.py tests/test_drafts_db.py`: `72 passed` after extracting schema/connection setup into `marvis/db_schema.py`.
  - `tests/test_agent_gate_contracts.py tests/test_plan_driver.py`: `35 passed` after enriching failure envelopes with editable input defaults and explicit downstream reset step ids.
  - `tests/test_orch_executor.py tests/test_orch_db.py`: `44 passed` after adding tool version, manifest hash, source dataset refs, and artifact refs to persisted step evidence.
  - `tests/test_agent_autodrive.py`: `24 passed` after adding a deterministic AUTO low-risk control allowlist for declared-but-expensive tuning and delivery actions.
  - `tests/test_orch_api.py tests/test_frontend_v2_plan.py tests/test_frontend_static_v2.py`: `238 passed` after exposing failure envelopes on plan step API payloads and rendering retry defaults/reset scope in both V2 plan views.
  - `tests/test_modeling_recipes.py tests/test_modeling_pack.py tests/test_plan_driver.py tests/test_frontend_screen_table.py tests/test_agent_autodrive.py`: `116 passed, 2 warnings` after adding sample-weight quality diagnostics to modeling setup, gate metadata, frontend controls, and AUTO prompts.
  - `tests/test_orch_db.py tests/test_orch_api.py tests/test_orch_executor.py tests/test_plan_driver.py`: `96 passed` after making step confirmation require the persisted step to still be `awaiting_confirm`.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes with `git diff --check`, ruff, and `node --check` after the artifact/directory transaction migration.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the step-run recovery update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the API/DB/frontend split updates.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the failure-envelope retry contract update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the evidence lineage update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the AUTO high-risk control guard update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the frontend failure retry contract update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the sample-weight diagnostics update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check --skip-pytest`: passes after the step confirmation state guard update.
  - `CONDA_NO_PLUGINS=true conda run -n py_313 scripts/check`: passes with `1794 passed, 2 warnings` after fixing the theme-module test contract and live-notebook session parameter.

## Executive Summary

V2 is materially stronger than the original runtime: PlanDriver is the common execution path, modeling has real recipes and PMML/report/handoff tools, evidence output versioning exists, plugin/tool started checkpoints exist, and the old settings-level V2 workbench has been removed from the UI.

It is not ready to call "complete" yet. The main remaining gap is not one isolated bug; it is that several capabilities are working as primitives but are not yet productized as reliable, typed, user-facing workflows. The highest-value next work is:

1. Finish gate adapters, stale-token coverage, and structured failure/retry UX on top of the new `GateEnvelope`/`FailureEnvelope` base. The failure envelope now flows through the plan API into retry panels; richer schema-to-form controls remain.
2. Broaden AUTO coverage and safety tests around the new structured `confirm|adjust|replan|clarify|halt` path.
3. Finish the modeling lifecycle by broadening the new sample-weight gate UI into a full modeling setup surface and adding business delivery surfaces on top of the new G2-G5 backend steps.
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
  - Failed plan steps returned by the plan API now include `failure_envelope`; the main right rail and modular V2 plan view both use its editable input defaults and reset-scope metadata in retry panels.
  - Modeling setup now diagnoses sample-weight candidates for numeric validity, missingness, range, mean, feature exclusion, and exposes those diagnostics in gate metadata, front-end controls, renderer tables, and AUTO prompts.
  - Step confirmation is now guarded at the repository write boundary: only steps still persisted as `awaiting_confirm` can set `confirmed = 1`; stale or non-gate confirm calls return API 409 and record the spawned job as failed instead of silently mutating a pending/failed step.

Items confirmed still not complete and therefore still part of the plan:

- OS-level sandboxing is not complete. Plugin/draft tools and ordinary notebook execution have subprocess/resource-limit isolation, but the live keep-alive notebook path still needs either worker RPC redesign or removal from safety-critical flows.
- DB plus filesystem writes are staged and recoverable for many high-risk artifact paths, but there is still no single cross-resource `UnitOfWork` that makes SQLite state and file promotion atomic as one unit.
- `api.py`, `db.py`, and `app.js` remain large after the first splits. `turn_handlers.py`, `db_schema.py`, `theme.js`, renderers, gate payloads, and adjust specs are good first cuts, not the final architecture.
- The frontend still lacks a dedicated modeling setup panel, full model comparison/delivery surface, and Playwright desktop/mobile visual smoke. Right-rail artifact preview is now wired into the real app shell, but still needs browser smoke coverage.
- The visual token system is partial; semantic task/surface/status tokens exist, but chart/KPI/modeling/report palettes still need consolidation.
- Broader AUTO safety fixtures are still needed for destructive actions, export/handoff, and wide downstream resets. Expensive/undeclared tuning adjustments now have a deterministic halt guard.

Current merge stance: this branch is not "V2 complete" yet. It can become an intermediate PR only after full `scripts/check`, manual smoke, and a PR description that explicitly lists the remaining items above. Direct merge to `main` as a finished V2 release is still too risky.

## Status Against Earlier 10 Recommendations

| # | Item | Status | Evidence / Gap | Next Action |
|---|---|---|---|---|
| 1 | AUTO structured decisions | Mostly done | `auto_drive.py` now parses `confirm|adjust|replan|clarify|halt` with params, selection, dedup strategies, replan goal, clarifying question, confidence, current-gate allowed action enforcement, and a low-risk control allowlist that blocks expensive tuning and delivery actions even if declared by a gate. | Add more fixtures for destructive domain gates and frontend stale-control paths. |
| 2 | Modeling business lifecycle | Partial | `choose_modeling_spec`, `configure_tuning`, `select_experiment`, and `post_training_action` are now tools/template steps; `TrainingDataset` caching is wired for multi-recipe train; sample-weight candidate choice now has a gate control. Broader setup UI and business delivery surfaces are still missing. | Expand the modeling setup panel to target type, split warnings, algorithm family, tuning budget, and delivery readiness. |
| 3 | PlanDriver decomposition | Partial | Tool output renderers moved to `marvis/agent/renderers.py`; structured gate payload builders moved to `marvis/agent/gate_payloads.py`; gate dependency rendering moved to `marvis/agent/gate_adapters.py`; basic adjust parameter specs moved to `marvis/agent/adjust_specs.py`. `PlanDriver` still owns adjust/replan routing and step-specific gate validation. | Expand `GateResponseAdapter` and make adjust specs tool/step schema-driven. |
| 4 | V2 turn orchestration out of `api.py` | Mostly done | Driver turn handlers for data join, feature analysis, modeling, strategy, vintage now live in `marvis/agent/turn_handlers.py`; `api.py` keeps the HTTP wrapper plus LLM/tier resolution. | Continue moving validation-agent stage orchestration and memory routes out of `api.py`. |
| 5 | Modeling data loaded once | Partial | `TrainingDataset` adapter and read-count tests exist for `train_models`; reporting and some other paths still read independently. | Expand adapter to report/scoring paths where useful. |
| 6 | Evidence versioning | Mostly done | `EvidenceEnvelope` is stored beside raw output and includes input summary/hash, parent refs, source dataset refs, artifact refs, tool version, manifest hash, seed, and renderer hint; raw output compatibility is preserved. Running step-runs now recover persisted outputs or finalize as interrupted after restart. | Add a deeper DB+file `UnitOfWork` and keep expanding domain-specific lineage where tools expose richer refs. |
| 7 | Sample-weight gate | Mostly done | Backend detects/validates candidates; create dialog now distinguishes no weight vs explicit column; the first modeling gate renders detected candidates and posts a structured `sample_weight_col` adjust that reruns `choose_modeling_spec` and downstream screening. Candidate diagnostics now expose numeric validity, missingness, range, mean, and feature exclusion in gate/UI/AUTO context. | Expose the same decision in the future extracted `ModelingSetupPanel` with richer target/split/tuning controls. |
| 8 | Frontend task workspace split | Partial | Some V2 modules exist; theme handling is now in `static/js/theme.js`, and task/welcome tones use semantic tokens. `app.js` still owns create dialog, rail, transcript, and driver gates. | Extract `CreateTaskDialog`, `PlanRailController`, `DriverConversationView`, `TaskWorkspace`. |
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
   - Problem: user and AUTO cannot reliably understand what is retryable, what inputs can be edited, and what downstream steps reset.
   - Fix: add `FailureEnvelope` with `failed_step_id`, `error_kind`, `retryable`, `editable_input_schema`, `suggested_actions`, `stale_token`, `downstream_reset`.
   - Tests: force tool failure, edit retry inputs, verify downstream reset and recovered completion.

### P1: Evidence, Transactions, And Runtime Safety

1. Tool side effects now have a recoverable run ledger, but not full DB+file transactionality.
   - Current behavior: executor records a `plan_step_runs` attempt before invocation, finalizes it after output storage, and startup/run recovery now closes in-flight step runs. If a persisted output version exists for the current running attempt, recovery attaches the latest output ref, marks the run succeeded, and completes post-checks without rerunning the tool. If no output exists, recovery marks the run interrupted and fails the step for explicit retry.
   - Risk: the system is recoverable, but file side effects, output version storage, step state, and run finalization are still not one atomic unit across SQLite plus filesystem.
   - Implemented:
     - Add `step_runs` table with `run_id`, `step_id`, `attempt`, `started_at`, `tool_result_ref`, `side_effects`, `finalized_at`, `error_kind`.
     - Record run start before invocation.
   - Remaining:
     - Finalize output version, step state, and run state in one SQLite transaction where possible.
     - Add a higher-level `UnitOfWork` that coordinates staged file promotion with DB commit boundaries.
     - Surface interrupted runs in the plan rail as deterministic retry/repair state.
   - Tests: done for crash after output ref before step update, no-output interruption, stale output after reset, and no unsafe replan/rerun on recovered running failure.

2. File and DB writes still need recovery semantics, but the main artifact paths now use staged promotion.
   - Current state: join execution output, data_ops clean/dedup derived outputs, modeling derived parquet/model/meta/PMML/calibration outputs, report scored parquet output, final xlsx reports, plugin install/promote directories, and validation handoff materials now use staged file or directory promotion.
   - Fix:
     - Done for join execution output: add `TransactionalArtifactStore` with stage, promote, rollback, and orphan cleanup; `JoinEngine.execute_join_plan` writes to `.staging` and promotes only the final artifact.
     - Done for data_ops derived outputs: `clean_format` and `dedup_rows` write parquet files through staging and roll back if dataset registration fails.
     - Done for modeling derived parquet outputs: `prepare_modeling_frame` / `make_split` and `reject_inference` write through staging and roll back if registration/audit fails.
     - Done for report outputs: `model_report_scored.parquet`, `render_model_report`, and `render_minimal_model_report` write through staging.
     - Done for model artifacts: native binaries, model meta, PMML export, and calibration payloads write through staged files with rollback on writer/validation failure.
     - Done for directory artifacts: plugin install, promoted draft plugin directories, and validation handoff materials use `TransactionalDirectoryStore` with backup restore on DB/audit failure.
     - Done for startup reconciliation: `create_app` runs artifact recovery and stores the report in `app.state.artifact_recovery_report`; orphan `.staging` directories are removed, plugin backups are restored or cleaned by DB checksum, and validation handoff material directories are reconciled against validation task `source_dir`.
     - Done for step-run reconciliation: executor recovery finalizes running step attempts as succeeded when the current run has a persisted output version, or interrupted when no output was persisted.
     - Remaining: true DB+filesystem unit-of-work semantics for multi-write tool execution.
   - Tests: done for store promote/rollback/orphan cleanup, sibling staged-file promotion, directory backup restore, writer failure rollback, plugin/draft audit-failure rollback, validation handoff audit-failure rollback, model artifact staging, and app startup recovery.

3. Evidence output refs are versioned but not semantically complete.
   - Current behavior: `plan_step_output_versions` preserves versions and refs.
   - Gap: missing normalized input snapshot, dataset ids, artifact paths, seed, parent refs, tool/plugin identity, manifest hash, and renderer hints.
   - Fix:
     - Add `EvidenceEnvelope` while preserving raw-output compatibility.
     - Store `input_hash`, `input_summary`, `source_dataset_refs`, `artifact_refs`, `parent_output_refs`, `tool_name`, `tool_version`, `manifest_hash`, `random_seed`.
   - Tests: v1 raw output can still load; v2 evidence can drive renderer and audit.

4. Notebook/plugin sandboxing is not OS-level.
   - Current state: plugin and draft tools run in one-shot subprocess workers with resource limits, timeout kill, network guard, and audited stdout/stderr tails. Ordinary full-notebook execution now has an optional isolated worker with parent timeout kill and artifact preservation. The live keep-alive notebook kernel path still exists for PMML/reproducibility appended cells and is protected by RSS monitoring/interrupt/shutdown rather than full subprocess session RPC.
   - Fix:
     - Done for plugin/draft tools: run in subprocess with memory/CPU/file-size limits and restricted worker environment.
     - Done for ordinary notebook execution: add `marvis.notebook_worker`, `run_notebook(..., isolated=True)`, worker error propagation, and parent timeout artifact preservation.
     - Remaining for live notebook sessions: either replace keep-alive kernel mutation with a worker RPC protocol, or split PMML/reproducibility appended-cell work into explicit non-live notebook/tool steps so the whole validation flow can use isolated execution.
     - Add slow/OOM integration tests at the pipeline/job layer once the live-session boundary is removed or explicitly downgraded.

### P1: Modeling Workflow And Business Closure

1. G2 algorithm/task selection now has a typed backend step, but the UI is still thin.
   - Current behavior: `choose_modeling_spec` normalizes target type, recipe family, eligible/disabled algorithms, metric policy, sample-weight policy, tuning budget, fixed params, and exposes a rendered gate table before feature screening.
   - Current update: sample-weight candidates now render as a first-gate control; changing the selected candidate posts structured `adjust_params.sample_weight_col` and reruns the modeling spec plus downstream screening. The create dialog also distinguishes "no weight" from an explicit column.
   - Remaining problem: the frontend still lacks a dedicated modeling setup panel for target type, split/OOT diagnostics, algorithm family, tuning budget, and override reasons.
   - Remaining fix:
     - Add `ModelingSetupPanel` / gate controls for target type, algorithm family, OOT/split warnings, tuning budget, and explicit override reasons.
     - Keep the typed modeling spec as the single downstream contract.

2. G3 tuning needs a typed control surface.
   - Current behavior: tuning exists, but configuration is not a first-class gate.
   - Fix:
     - Add `configure_tuning` gate with skip/tune choice, search space, metric, time budget, sample weight usage, random seed, and agent recommendation.
     - Allow AUTO to propose bounded changes, but require human confirmation for expensive searches.

3. G4 model selection is automatic.
   - Current behavior: `train_models` internally picks a best experiment.
   - Problem: production credit/risk modeling usually needs a business choice, not only metric max.
   - Fix:
     - Add `select_experiment` tool/gate.
     - Rank candidates by OOT metrics, stability, calibration, feature count, monotonicity/scorecard constraints, PMML support, and report readiness.
     - Store selected experiment explicitly.

4. G5 post-training closure is not a workflow.
   - Current behavior: report is generated; PMML/handoff tools exist but are not explicit final actions.
   - Fix:
     - Add `post_training_action` gate with actions:
       - export `.pkl` native artifact.
       - export `.pmml` when supported.
       - generate model report.
       - generate model card / approval pack.
       - hand off to validation.
       - create challenger/backtest task.
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
   - Current behavior: template still contains `oot_ks >= 0.3331`.
   - Fix:
     - Replace fixed success gates with configurable acceptance policy.
     - Default to recommendation language: pass/warn/fail based on domain thresholds, but never silently block a valid business model only because one fixed threshold was missed.

8. Reports should surface missing business context.
   - Current behavior: binary reports are richer; non-binary reports are minimal; chat renderer now shows unavailable business sections from `section_status`.
   - Fix:
     - Done: report message shows generated sections, skipped sections, and missing inputs.
     - Add non-binary report sections where applicable.
     - Add business decision summary: threshold, approval recommendation, reject inference status, monitoring plan.

### P2: API And Database Architecture

1. `api.py` remains a bottleneck.
   - Current state: driver turn orchestration for data_join, feature_analysis, modeling, strategy, and vintage has moved to `marvis/agent/turn_handlers.py`. `api.py` still owns validation/data/agent/memory/stage job routes and compatibility wrappers.
   - Fix order:
     - Done: `marvis/agent/turn_handlers.py`: data_join, feature_analysis, modeling, strategy, vintage.
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
   - Current state: V2 modules exist and theme handling has moved to `static/js/theme.js`, but task dialog, plan rail, conversation rendering, and task creation still live in one global controller.
   - Fix order:
     - `CreateTaskDialog`: task type definitions, run mode, algorithm family mutual exclusion, material source, tier picker, payload assembly.
     - `PlanRailController`: plan fetch/cache/retry/render, status mapping, gated actions, downloads.
     - `DriverConversationView`: manual analysis rendering, latest-gate interactivity, screen/dedup/C1 controls, agent transcript rendering.
     - `TaskWorkspace`: task shell, active task state, right rail coordination.

2. Modeling needs a dedicated setup and analysis surface.
   - Fix:
     - `ModelingSetupPanel` with target/split counts, target type, algorithm choices, detected sample weights, split/OOT warnings.
     - Screen table as a real modeling selector: threshold sliders plus numeric inputs, `top_k`, sort/filter chips, selected-count summary, leakage override reason, reset-to-proposal.
     - Model comparison panel: metrics, stability, calibration, artifact support, report readiness, validation handoff readiness.

3. Visual tokens are partial, not a system.
   - Current state: semantic task tones, surface/border/status tokens, and welcome/task icon palettes are centralized. KPI/ROC/modeling report areas still contain local palette constants.
   - Fix:
     - Done for task types and core surface/status tokens.
     - Create semantic tokens for metric cards, charts, model status, and action states.
     - Replace hard-coded local hex colors in welcome/modeling/metric/chart surfaces.
     - Add dark/light parity checks.

4. UX should communicate business readiness, not just execution progress.
   - Fix:
     - Plan rail statuses: "needs decision", "running", "blocked", "ready for handoff", "needs business input".
     - Modeling report card: sections complete/missing, PMML support, validation handoff state, selected experiment.
     - Clear stale-gate warnings and retry contracts.

5. Accessibility and visual smoke should become test gates.
   - Tests:
     - `node --check` for extracted modules.
     - Static tests for required controls and stale tokens.
     - Playwright smoke for welcome, create dialog, plan rail, screen table, desktop/mobile, light/dark.

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
   - Generate challenger comparison from previous selected experiments.
   - Create validation/monitoring tasks directly from G5.
   - Store monitoring thresholds and drift checks as versioned policy.

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
- Partial: add retry/failure contract metadata and downstream reset behavior; richer editable-input retry UX remains.

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
- Mostly done for sample-weight gate: add sample-weight propagation, G2 spec output, create-time no-weight/explicit policy, and detected-candidate gate adjust/rerun; broader setup panel still remains.
- Done for `train_models`: add `TrainingDataset` adapter and read-count tests.
- Done: remove hard-coded `oot_ks >= 0.3331` as a universal success gate.
- Done for chat renderer: improve report renderer with section status and missing inputs; broader report content remains.

Acceptance:
- A user can create a modeling task, confirm algorithms/weights, tune/train, select an experiment, export `.pkl` and `.pmml` when supported, generate report, and hand off to validation.
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
- Partial: add OS-level subprocess sandbox for plugin/draft tools and ordinary full-notebook execution; live keep-alive notebook execution still needs a worker RPC redesign or step split.

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
- Introduce `UnitOfWork` and migrate one domain at a time.
- Add pagination to high-volume list endpoints.

Acceptance:
- `api.py` becomes app compatibility and route registration, not domain orchestration.
- `db.py` becomes compatibility exports or a thin package entrypoint.
- New tests assert route registration and repository behavior.

### Phase F: Frontend Workspace And Visual System

Goal: make V2 feel like a professional task workspace rather than a generic chat plus tables.

Tasks:
- Extract `CreateTaskDialog`.
- Extract `PlanRailController`.
- Extract `DriverConversationView`.
- Add `TaskWorkspace`.
- Add `ModelingSetupPanel` and model comparison/post-training panels.
- Partial: implement semantic task/surface/status tokens and extract theme controller.
- Add screen selector controls for sliders, `top_k`, filters, reset, and override reasons.

Acceptance:
- `app.js` shrinks materially and no longer owns all task responsibilities.
- Frontend tests cover extracted modules and modeling gate controls.
- Playwright smoke verifies desktop/mobile, light/dark, create dialog, plan rail, screen selector.

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

Current state should be treated as not ready for direct `main` merge if the goal is "V2 complete". It may be mergeable as an intermediate PR only if the PR description clearly labels the remaining work above and CI/manual smoke pass.

The highest risks before a production-style merge are:

- CI gate exists, but full CI/full pytest still must pass from the final committed tree.
- AUTO can apply structured decisions, but broader safe-remediation policy fixtures are still needed.
- Modeling final handoff is now workflow-capable for G2-G5 backend steps, and sample-weight candidate adjustment has a working gate; richer modeling setup and business delivery UI still remain.
- Runtime crash windows are recoverable for staged artifacts and running step attempts, but the DB+filesystem boundary is not yet a true atomic unit of work.
- `api.py`, `db.py`, and `app.js` are still large, but the first stable splits have landed (`turn_handlers`, `db_schema`, `theme.js`); remaining risk is deeper router/repository/workspace-controller extraction.
- OS-level sandboxing for notebook/plugin execution is not complete.

## Definition Of Done For V2 Complete

V2 can be called complete only when:

- All core task types run through typed gates and evidence envelopes.
- AUTO can perform bounded structured actions and stops safely outside declared permissions.
- Modeling covers G2-G5 with explicit user or AUTO decisions, selected experiment, PMML/PKL/report/handoff closure.
- Runtime side effects are recoverable through step-run ledger recovery and artifact staging, with any remaining non-atomic boundaries explicitly documented.
- API, DB, and frontend controllers are split enough that new domains do not expand monolith files.
- CI and manual smoke are green from a clean committed tree.
- Final review documents remaining limitations as product choices, not unfinished core architecture.
