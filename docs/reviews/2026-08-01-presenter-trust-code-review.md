---
status: resolved
original_status: issues_found
reviewed: 2026-08-01T00:00:00+08:00
depth: deep
files_reviewed: 12
files_reviewed_list:
  - marvis/agent/renderers.py
  - marvis/agent/gate_adapters.py
  - marvis/agent/plan_message_composer.py
  - marvis/routers/validation_agent.py
  - marvis/agent/presenters/runtime.py
  - marvis/agent/presenters/modeling_evidence.py
  - marvis/agent/presenters/strategy_exploration.py
  - marvis/agent/presenters/strategy.py
  - tests/test_canonical_presenter_dispatch.py
  - tests/test_modeling_evidence_presenters.py
  - tests/test_strategy_exploration_presenters.py
  - tests/test_strategy_presenter_registry.py
findings:
  critical: 2
  warning: 1
  info: 0
  total: 3
---

# Governed Tool Presenter Trust Review

**Reviewed:** 2026-08-01T00:00:00+08:00
**Depth:** deep
**Files Reviewed:** 12 primary files, plus direct runtime, evidence-validation, plan-repository, dataset-download, and template dependencies.
**Status:** resolved (original review: issues_found)

> Closure 2026-08-01：本报告中的 CR-01、CR-02、WR-01 均已关闭；完整复核见
> [Final local closure](2026-08-01-final-local-closure-code-review.md)。下文保留原始发现，
> 作为整改依据，而不是当前未关闭问题清单。

## Summary

The new runtime presenters correctly reject absent runtime/task context and do not fall back to generic rendering after a validator error. The stated trust boundary is nevertheless incomplete: it authenticates that a cached envelope names *some* live object in the same task, not that it is the object produced by the current plan step. Completion and reload dataset actions additionally derive their target from mutable step/message payloads rather than the live dataset registry and the completion message's own plan. These defects can present a different legitimate task artifact or dataset as the current result.

## Critical Issues

### CR-01: Cached canonical envelopes choose the live artifact used as their trust root

**File:** `marvis/agent/plan_message_composer.py:563`
**Related lines:** `marvis/agent/plan_message_composer.py:634`, `marvis/agent/presenters/strategy_exploration.py:650`, `marvis/packs/modeling/evidence_tools.py:497`, `marvis/packs/modeling/score_evidence_tools.py:356`

**Issue:** The composer passes `step.inputs` only to PoolStability. For an ImpactCube it obtains the registry record from `output["artifact"]["artifact_id"]` (lines 641-653); the ImpactCube presenter then authenticates the output against that chosen record (lines 680-689). Modeling and the other exploration presenters likewise derive their sample/artifact/search/revision identifiers from the cached `output` before resolving them through the runtime. Consequently, if a task has two valid ImpactCubes, model-evidence runs, searches, or revisions, replacing a step's stored envelope wholesale with the other valid envelope still passes all current checks and displays it as the current step. The task id/workspace remain live, but the artifact and producer-run roots do not.

This is a provenance/authentication failure, not merely stale display: a user can be shown an authenticated result for the wrong pool revision, sample binding, search request, or model experiment, and may make a governed downstream decision from it.

**Fix:** Bind each persisted step output to the execution's immutable identity (at minimum: plan id, step id, resolved-input hash, and output artifact/producer-run ids plus hashes) when the executor commits it. In `PlanMessageComposer`, obtain the expected binding using `step.id` from repository-owned state, pass that binding to every canonical presenter, and make validators compare cached identifiers to it before resolving any artifact. For ImpactCube, compare the trusted producer run's `input_hash`/request to the resolved inputs of this exact step; apply the equivalent exact-step binding to modeling/search/revision presenters. Fail closed when no exact binding exists.

### CR-02: Done and reload dataset controls trust raw output and stale message metadata instead of the live dataset for the originating plan

**File:** `marvis/agent/plan_message_composer.py:298`
**Related lines:** `marvis/routers/validation_agent.py:334`, `marvis/routers/validation_agent.py:346`

**Issue:** `latest_result_dataset_metadata()` reads `result_dataset_id` directly from stored step output and builds a download URL without querying the dataset registry or binding the dataset to the step. On reload, `_enrich_historical_result_download()` always selects `plans[-1]`, regardless of the completion message's `metadata.plan_id`, then merges the persisted `existing` metadata *after* the recovered live-looking metadata (lines 346-349). A cached output can therefore select any dataset id; a pre-existing completion message can override the recovered id, URL, title, and label; and an older completion message can be enriched with the newest, unrelated plan's dataset.

The download route enforces task ownership, so this does not cross task boundaries. It still misrepresents task evidence and can direct a user to the wrong same-task result after a retry/replan or corrupted cached transcript.

**Fix:** Add a repository-owned result-dataset binding keyed by plan/step (including dataset id and immutable dataset identity/hash) at execution commit. Resolve `result_dataset` exclusively through that binding and `DatasetRegistry`, verifying `dataset.task_id == plan.task_id` and the registered content identity. During historical enrichment, locate the plan by the completion message's trusted `plan_id`; if it is absent or does not have an exact binding, omit the action rather than guessing `plans[-1]`. Treat any persisted `result_dataset` metadata as display-only or overwrite it entirely with reconstructed, verified fields.

## Warnings

### WR-01: Tests prove malformed-envelope rejection but not same-task substitution or reload-plan integrity

**File:** `tests/test_canonical_presenter_dispatch.py:172`
**Related lines:** `tests/test_modeling_evidence_presenters.py:48`, `tests/test_strategy_exploration_presenters.py:141`, `tests/test_data_join_api.py:176`

**Issue:** The tests forge extra/missing fields and injected validator failures, but every fixture has one relevant live artifact. They do not create two valid artifacts in one task and replace one step's cached output with the other's complete canonical envelope. The reload test only removes `result_dataset`; it does not cover forged existing metadata or an older completion message after a newer plan has completed. The current regressions therefore pass while both critical provenance substitutions remain possible.

**Fix:** Add negative integration tests that (1) produce two valid canonical outputs in one task, swap their complete persisted step envelopes, and assert canonical presentation fails; (2) replace an existing completion metadata dataset id/URL and assert reload overwrites or rejects it; and (3) create two completed plans and assert each completion resolves only its own registered result dataset.

---

_Validation run: `conda run -n py_313 python -m pytest -q tests/test_canonical_presenter_dispatch.py tests/test_modeling_evidence_presenters.py tests/test_strategy_exploration_presenters.py tests/test_strategy_presenter_registry.py` — 68 passed._
_Static check: `conda run -n py_313 python -m ruff check` over the 12 scoped files — passed._

## Closure verification

- CR-01：8 个 canonical Tool 在成功前用本次 resolved inputs 与实时领域对象认证；
  ToolRunner 回执绑定 invocation、raw output、实际 Tool version 与 manifest hash；
  Repository 原子持久化 exact output version，Presenter 只接受同 plan/step/run 的绑定。
- CR-02：completion message 按自身 plan 重建结果数据集；下载 URL 绑定
  plan/step/output/hash，服务端重载 exact evidence，并从单 descriptor 复制校验到私有快照；
  registry/文件一致替换和 verify-to-open 竞态均失败关闭，Range 只在冻结快照上处理。
- WR-01：增加同任务合法 envelope 替换、历史多 plan、registry 漂移、下载竞态、
  恢复完整性和回执不可变测试；两轮独立复核最终均为 PASS。
