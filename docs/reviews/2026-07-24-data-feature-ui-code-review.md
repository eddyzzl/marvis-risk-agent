---
phase: e2e-data-feature-ui-closure
reviewed: 2026-07-23T18:20:09Z
depth: deep
status: resolved_in_final_review
files_reviewed: 49
files_reviewed_list:
  - docs/plans/v2-feature-phase-spec.md
  - docs/plans/v2-join-phase-spec.md
  - docs/plans/v2-master-backlog.md
  - marvis/agent/data_setup.py
  - marvis/data/registry.py
  - marvis/agent/feature_setup.py
  - marvis/agent/sample_setup.py
  - marvis/agent/turn_handlers.py
  - marvis/agent/workflow_insights.py
  - marvis/agent/memory_bridge.py
  - marvis/agent/gate_payloads.py
  - marvis/agent/gate_adapters.py
  - marvis/agent/gates/adapters.py
  - marvis/agent_memory/extractors.py
  - marvis/agent_memory/distillation.py
  - marvis/agent_memory/store.py
  - marvis/api_schemas.py
  - marvis/orchestrator/templates/feature.py
  - marvis/packs/feature/tools.py
  - marvis/feature/candidates.py
  - marvis/feature/bin_analysis.py
  - marvis/governance/service.py
  - marvis/repositories/strategy.py
  - marvis/packs/strategy/dsl.py
  - marvis/packs/strategy/legacy_adapter.py
  - marvis/static/index.html
  - marvis/static/app.js
  - marvis/static/js/create-task-dialog.js
  - marvis/static/js/agent-conversation-view.js
  - marvis/static/js/v2/driver_manual_analysis.js
  - marvis/static/js/v2/driver_gate_confirm.js
  - marvis/static/js/v2/plan_rail_controller.js
  - marvis/static/js/v2/join_gate_controller.js
  - marvis/static/js/v2/feature_binning_gate.js
  - scripts/closure_acceptance.py
  - scripts/run_closure_smoke.py
  - scripts/ks_baseline.py
  - tests/test_closure_acceptance.py
  - tests/test_feature_analysis_api.py
  - tests/test_feature_pack.py
  - tests/test_join_setup.py
  - tests/test_join_key_gate.py
  - tests/test_workflow_insights.py
  - tests/test_v2_memory_wiring.py
  - tests/test_frontend_screen_table.py
  - tests/test_frontend_static_v2.py
  - tests/test_strategy_adoption_atomic_repository.py
  - tests/test_strategy_dsl_equivalence.py
  - tests/test_strategy_effect_fence.py
findings:
  critical: 10
  warning: 5
  info: 0
  total: 15
status: issues_found
---

> 这是提交前的初始审查快照；下列 finding 均已修复并复审通过。
> 最终结论见
> [2026-07-24-data-feature-model-final-review.md](2026-07-24-data-feature-model-final-review.md)。

# Data / Feature / Agent / UI / Closure Code Review

**Reviewed:** 2026-07-23T18:20:09Z
**Depth:** deep
**Files Reviewed:** 49
**Status:** issues_found

## Summary

This was an adversarial, read-only review of the non-modeling closure path. It traced source-file reconciliation, feature setup and metrics, Agent insight/memory, manual and Agent gate rendering, the plan rail, strategy effect-fence changes, and the machine-closure script across their callers and tests.

The current tree is **not ready to close**. Ten blockers can produce omitted/duplicated data, dead-end a required manual workflow, silently violate locked feature-analysis requirements, contaminate reusable Agent memory, or issue a machine PASS for evidence that is demonstrably incomplete or internally inconsistent.

No source files were changed by this review. Reproduction snippets were executed against the current worktree.

## Narrative Findings (AI reviewer)

## Blockers

### CR-01: Partial source reconciliation can duplicate one same-stem upload and omit another

**Classification:** BLOCKER
**Files:**

- `marvis/agent/data_setup.py:38-56`
- `marvis/data/registry.py:69-105`

**Issue:** Reconciliation identifies an already registered source only by the normalized stem of its generated parquet name. The registry persists `vars_<random>.parquet`, not the original upload name/suffix. With `vars.csv` and `vars.xlsx`, a partially registered `vars.xlsx` is indistinguishable from a partially registered `vars.csv`. The counter consumes the first artifact with stem `vars` and registers the second, so one source can be duplicated while the other is omitted.

**Reproduction:**

```text
existing normalized row: tasks/t/datasets/vars_deadbeef.parquet
source artifacts: vars.csv, vars.xlsx
registered by reconciliation: ['vars.xlsx']
```

If the existing row came from `vars.xlsx`, the result is two copies of the workbook and no CSV. The existing test covers only the opposite registration order, which happens to pass.

**Fix:** Persist immutable source identity on every dataset registration, at minimum original upload name + suffix and preferably source-file SHA-256. Reconcile by that exact identity/hash, never by stem. Add reverse-order partial-registration tests for `vars.csv`/`vars.xlsx`, repeated names in nested directories, and same-content aliases.

### CR-02: Standalone feature analysis cannot honor or clarify an explicit target column

**Classification:** BLOCKER
**Files:**

- `marvis/agent/turn_handlers.py:508-527`
- `marvis/agent/feature_setup.py:70-94`
- `marvis/agent/sample_setup.py:148-179`

**Issue:** `_run_feature_setup` ignores both `task.target_col` and `user_text`. `build_feature_proposal` calls `detect_setup` without `configured_target`. When two equally ranked valid labels exist, detection correctly returns no target and tells the user to name one, but the next user reply is not parsed into setup state. The workflow repeats the same error/recovery path indefinitely. An API caller that already supplied `target_col` is ignored too.

**Reproduction:**

```text
columns: label_a, label_b, x1
detect_setup(...).target_col                         -> ''
detect_setup(..., configured_target='label_a')      -> 'label_a'
```

The feature setup path never makes the second call.

**Fix:** Thread `task.target_col` into `build_feature_proposal`/`detect_setup`, preserve “unset” separately from the legacy default, and implement a structured target-choice gate for ambiguous standalone samples. Natural-language replies in Agent mode and the manual selector must update the same persisted setup state. Add an end-to-end test that starts with two valid labels, selects one, and completes the report.

### CR-03: The locked “selected metrics only” contract is broken in both run modes

**Classification:** BLOCKER
**Files:**

- `marvis/agent/feature_setup.py:92-101`
- `marvis/api_schemas.py:52-73`
- `marvis/static/js/create-task-dialog.js:77-82`
- `marvis/static/js/create-task-dialog.js:478-480`

**Issue:** `metrics or _DEFAULT_METRICS` turns an explicit empty selection into IV/KS/AUC/coverage/meaning-consistency. This contradicts the locked specification that every metric is optional and unchecked metrics must not run. Separately, the metric selector is hidden in Agent mode and the payload is submitted only for manual mode, so an Agent task cannot express the same metric choice at creation and always falls into the server fallback.

**Reproduction:**

```text
build_feature_proposal(..., metrics=[]).metrics
-> ['iv', 'ks', 'auc', 'coverage', 'meaning_consistency']
```

The pack itself correctly distinguishes an absent key from explicit `[]`; setup destroys that distinction before execution.

**Fix:** Represent “not supplied/legacy task” as `None` and an explicit empty list as `[]` (`metrics: list[str] | None = None`). Apply defaults only to `None`. Expose the same selection contract in Agent mode, either in task creation or through a persisted natural-language slot-filling turn before plan creation. Test absent, empty, and non-empty payloads in both modes.

### CR-04: “Meaning consistency” does not implement the required Agent/LLM or U-shape analysis

**Classification:** BLOCKER
**Files:**

- `docs/plans/v2-feature-phase-spec.md:56-59`
- `docs/plans/v2-feature-phase-spec.md:89-93`
- `marvis/packs/feature/tools.py:462-539`
- `marvis/packs/feature/tools.py:551-577`

**Issue:** The locked design requires an LLM to infer positive/negative/U-shaped/uncertain business direction from the data dictionary and requires binned bad-rate shape analysis for U-shaped expectations. The implementation uses a fixed substring list and a linear correlation threshold. No LLM is called and no U-shape test is performed. Negated meanings such as “无逾期次数” still contain “逾期” and are classified positive. The output source label `agent_dictionary_rule` is therefore misleading.

**Fix:** Add a bounded-schema LLM call that receives only feature code, governed dictionary meaning, target definition, and allowed direction labels. Keep actual metrics deterministic. For an expected U-shape, use the same governed binning kernel to test the bad-rate curve; if support is insufficient, return `uncertain/需人工看`. A no-LLM fallback must be conservative (`uncertain`), not keyword-based. Record prompt/model/version, confidence, and evidence without allowing the LLM to invent measured values.

### CR-05: Neutral “待评估” features become recommendations and are stored as high-confidence memory

**Classification:** BLOCKER
**Files:**

- `marvis/packs/feature/tools.py:580-624`
- `marvis/agent/workflow_insights.py:127-189`
- `marvis/agent/memory_bridge.py:196-244`
- `marvis/agent_memory/extractors.py:62-90`

**Issue:** When IV/KS/AUC were not selected, the deterministic tool correctly labels a feature `待评估`. Both the insight layer and memory capture classify every non-adverse label as recommended. The extractor then writes the unsupported recommendation with `confidence="high"`, contaminating future tasks and distillations.

**Reproduction:**

```text
input row: ['x1', '待评估', '本次未选择 IV、KS 或 AUC...']
recommended_features -> ['x1']
recommendations       -> ['优先评估：x1。']
```

**Fix:** Centralize a closed recommendation-state mapping. Only governed positive states (`推荐`, and if explicitly intended, `候选`) may enter `recommended_features`; `待评估` must remain neutral. Persist the supporting metric names/values and derive confidence from evidence. Add tests for every label, especially `待评估`, empty labels, and contradictory quality flags.

### CR-06: Manual join-key ambiguity is a UI dead end

**Classification:** BLOCKER
**Files:**

- `marvis/static/js/v2/driver_manual_analysis.js:48-95`
- `marvis/static/js/v2/driver_manual_analysis.js:165-204`
- `marvis/static/js/v2/driver_gate_confirm.js:20-31`

**Issue:** `driverGateBodyHtml` knows how to render `meta.join_keys`, but `driverManualAnalysisHtml` never dispatches that metadata branch. It falls through to generic tables. The generic confirm renderer also deliberately returns nothing because `join_keys` is considered a structured widget. Therefore manual mode displays neither the key picker nor an actionable confirm button when the Agent cannot choose a join key.

**Reproduction:**

```json
{
  "renderJoinKeyPicker calls": [],
  "html": "<section ...>choose<TABLE/><CONFIRM/></section>"
}
```

With the real `renderDriverGateButton`, `<CONFIRM/>` is also empty.

**Fix:** Add a `meta.join_keys` branch beside `join_c1` and pass `{ interactive: isPendingGate }` through `driverGateBodyHtml`. Add a frontend test that the latest join-key gate contains the picker and submit action, while historical instances are read-only.

### CR-07: Closure check C2 passes with zero step-run ids, manifest hashes, or input hashes

**Classification:** BLOCKER
**File:** `scripts/closure_acceptance.py:430-453`

**Issue:** `all(... for row in evidence_rows if row.get("step_run_id"))` is vacuously true when no evidence row has a `step_run_id`. Source refs and a seed are only required somewhere in the plan, not on each applicable executed step.

**Reproduction:** Replacing every evidence envelope with only:

```json
{"source_dataset_refs": ["dataset:ds-1"], "random_seed": 23}
```

still returns:

```text
C2 status=PASS
manifest_hash=True; input_hash=True; source_dataset_refs=True; seed=True
```

**Fix:** Build the set of required completed steps and validate each envelope individually. Every executed tool step must have a non-empty step-run id, manifest hash, input hash, and relevant source refs; stochastic steps must also carry their seed. Report missing fields by step id/title. Empty required-step sets must fail closed.

### CR-08: Closure check C3 is hard-coded PASS and does not verify any regression execution

**Classification:** BLOCKER
**File:** `scripts/closure_acceptance.py:455-465`

**Issue:** C3 returns PASS solely because two test filenames are written into a string. It does not run those tests or consume a signed/hashed test result, exit code, commit hash, or timestamp. Missing, stale, or failing tests still produce a machine PASS.

**Fix:** Either run the scoped tests as part of the closure command or consume a machine-generated test-evidence artifact containing exact command, exit code, commit SHA, timestamp, and log hash. If no matching evidence exists for the reviewed commit, C3 must be FAIL (or MANUAL if the product explicitly chooses that policy), never PASS.

### CR-09: Closure verdict remains machine PASS when every delivery artifact is missing

**Classification:** BLOCKER
**File:** `scripts/closure_acceptance.py:512-552`

**Issue:** The script computes booleans for the report, native model, PMML, and model card, but never converts missing mandatory artifacts into failed checks. `machine_failures` only reads the earlier check list.

**Reproduction:**

```text
artifacts = {report: false, native_model: false, pmml: false, model_card: false}
machine_verdict = PASS
machine_failures = []
```

The CLI consequently exits 0.

**Fix:** Add explicit artifact checks to the canonical check list and machine failures. Use a recipe/export applicability matrix so a genuinely unsupported format is explicit `N/A`; report, native model, and model card should be mandatory for a successful model delivery unless the governing contract says otherwise.

### CR-10: Internally inconsistent KS evidence is downgraded to MANUAL and cannot fail closure

**Classification:** BLOCKER
**File:** `scripts/closure_acceptance.py:374-403`

**Issue:** `ks_consistent` is computed, but B4 status is always `MANUAL`. Internal mismatch is therefore excluded from `machine_failures`, even though the message itself says it must be repaired first.

**Reproduction:**

```text
select.train_ks = 0.5
model_card.train_ks = 0.1
B4 status = MANUAL
internal_consistency = false
machine_verdict = PASS
```

**Fix:** Split the check into machine-verifiable internal consistency and external benchmark reconciliation. Internal inconsistency must be `FAIL`; only after it passes may independent historical/ground-truth comparison remain `MANUAL`.

## Warnings

### WR-01: Cross-task memory retrieval ignores the current target/scope

**Classification:** WARNING
**Files:**

- `marvis/agent/memory_bridge.py:229-240`
- `marvis/agent/memory_bridge.py:464-540`
- `marvis/agent_memory/store.py:273-309`
- `marvis/agent_memory/store.py:405-435`

**Issue:** Feature memories are captured with a target-specific scope, but retrieval queries only by category. A task for one target can receive a distillation or raw recommendation from an incompatible target definition and present it as relevant historical memory.

**Fix:** Compute the compatible scope from the current task and filter both raw entries and distillations before ranking. Include scope in visible references. Add negative tests proving a `target=A` task cannot retrieve `target=B` feature advice.

### WR-02: Distillation can recommend and avoid the same feature simultaneously

**Classification:** WARNING
**File:** `marvis/agent_memory/distillation.py:214-248`

**Issue:** Feature distillation independently unions all recommended and all avoid lists. Conflicting history can place `x1` in both lists, and the summary prints both without disclosing disagreement. Support is group-level rather than per-feature.

**Fix:** Aggregate evidence per feature with distinct-task positive/negative support and feedback. Conflicts should become a neutral “evidence inconsistent; re-evaluate” state. Preserve source ids/counts for auditability.

### WR-03: The numeric “meta column” heuristic silently removes ordinary predictive variables

**Classification:** WARNING
**File:** `marvis/feature/candidates.py:11-32`

**Issue:** Broad name tokens such as `loan`, `apply`, `mobile`, and `cust` classify ordinary business features as metadata. Current results include:

```text
loan_amount -> meta
apply_count -> meta
mobile_age  -> meta
cust_income -> meta
```

These numeric columns disappear from both feature analysis and modeling without the explicit excluded-categorical notice.

**Fix:** Restrict automatic hard exclusion to exact/structurally proven identifiers, dates, split fields, and weights. Use schema/cardinality/semantic-role evidence instead of broad business prefixes, surface every numeric exclusion with a reason, and allow an explicit user override.

### WR-04: Historical modeling and dedup gates remain interactive in manual mode

**Classification:** WARNING
**File:** `marvis/static/js/v2/driver_manual_analysis.js:176-183`

**Issue:** Historical `modeling_setup` gates use `meta.kind === "gate"` rather than `isPendingGate`, and dedup gates omit the `interactive` option entirely. A later gate does not disable those old controls.

**Reproduction:**

```json
[["old-model", "model", true], ["old-dedup", "dedup", true]]
```

**Fix:** Pass `{ interactive: isPendingGate }` for both branches and add the same stale-widget regression coverage already present for screen and C1 gates.

### WR-05: The plan rail conflates “Agent is answering” with “workflow step is executing”

**Classification:** WARNING
**Files:**

- `marvis/static/js/v2/plan_rail_controller.js:450-471`
- `marvis/static/app.js:444-467`
- `marvis/static/app.js:7364-7388`

**Issue:** Any send through the Agent composer sets the local busy action to `"agent"`. The rail then promotes the first awaiting-confirm/pending step to running even if the user merely asks “为什么失败？” and the Agent is composing an explanation. This recreates the reported ambiguity between execution and writing a reply.

**Fix:** Track a distinct local driver-execution phase, derived from an explicit execute/confirm/retry dispatch or authoritative server job metadata. Ordinary chat/diagnosis generation must not mutate plan-step status. Never infer `awaiting_confirm` as running solely from a generic Agent busy flag.

---

_Reviewed: 2026-07-23T18:20:09Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: deep_
