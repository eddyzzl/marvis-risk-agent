# MARVIS V2 Comprehensive Improvement Review — 2026-07-02

- **Branch:** `codex/v2-plugin-tool-runtime` (working tree as of 2026-07-02; ~166 commits ahead of the 2026-06-28 review baseline)
- **Scope:** entire codebase — backend (~60k LOC Python across 254 files in `marvis/`), frontend (`static/app.js` 6,416 lines + `static/js/v2/` 22 modules, ~14.6k LOC JS / 9.4k LOC CSS), 159 test files (~1,750 cases), docs and packaging
- **Method:** 10 specialist review lenses ran in parallel, each deep-reading real source with mandatory `file:line` evidence; every high/critical finding with a falsifiable claim was then handed to an independent adversarial verifier instructed to refute it; a completeness critic swept for whole dimensions the lenses missed. 73 sub-agents, ~5.3M tokens, ~1,700 tool calls across two runs.
- **Verification outcome:** 115 findings total. 38 high/critical findings went through the adversarial pass: **30 CONFIRMED, 8 PARTIAL (confirmed with corrections), 0 REFUTED.** The remaining 77 findings are medium/low impact or purely directional proposals and were not independently re-verified (marked as such throughout).
- **Relationship to earlier reviews:** this round deliberately excludes everything the 2026-06-21 and 2026-06-28 reports list as fixed-and-verified. Reviewers were instructed to re-check "claimed fixed" items against the current tree — three of them turned out to be *incompletely* fixed (see below). Items independently re-confirmed as genuinely fixed are listed in the Appendix.

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Top-10 Priorities](#top-10-priorities)
- [Quick Wins (small effort, high impact)](#quick-wins)
- [Roadmap: Six Batches](#roadmap-six-batches)
- 1. Agent Intelligence & Orchestration (AGT)
- 2. Memory System & Experience Reuse (MEM)
- 3. Credit-Risk Domain Depth (DOM)
- 4. LLM Client & Token Engineering (LLM)
- 5. Performance & Efficiency (PERF)
- 6. Product UX & Interaction Flow (UX)
- 7. Visual Design & Brand (VD)
- 8. Architecture & Code Quality (ARCH)
- 9. Reliability & Recovery (REL)
- 10. Testing Strategy & Security (TST)
- 11. Cross-Cutting Gaps (Completeness Critic) (GAP)
- [Appendix: Methodology, Statistics, Confirmed-Fixed Items](#appendix)

---

## Executive Summary

**The deterministic skeleton keeps getting stronger; the layers around it haven't caught up.** Since 6-28 the platform gained transactional staging across write paths, audit-in-same-transaction coverage, a real `api.py` decomposition (922 lines + routers/repositories), tokenized colors with full dark coverage, scorecard points that match the scorer exactly, default monotonic binning, calibration/model-card/monitoring-policy/challenger deliverables, and subprocess-isolated notebooks with RSS soft-monitoring. Nothing in this round contradicts that trajectory — out of 38 adversarially verified claims, zero were refuted.

The review's headline results group into six themes:

**(1) Three "claimed fixed" items are only fixed on the primary path — sibling paths were missed.** This is the most actionable pattern in the whole report:

- **AGT-1** — `is_confirm` got a negation guard in 6-28 (H4), but questions ("这样可以吗？" — "is this OK?") and embedded affirmatives ("结果不是很好的") still match, and the check short-circuits *before* LLM instruction routing, so a user question can silently confirm `execute_join` or model handoff. Empirically demonstrated.
- **DOM-1** — the NaN-label confirmation gate (H1/H2) was wired into all 9 training recipes and `select_features`, but **not** into `tune_hyperparameters`: hyperparameters are still silently selected on NaN-polluted trials (empirically reproduced: LightGBM trains on NaN labels without error, quietly coercing them to 0.0), then consumed by the final training run that *does* pass the gate — "passed the gate, params already poisoned."
- **PERF-2** — the JOIN key-space fix (H3) moved match-rate scanning to the transformed key space, but uniqueness/dedup still run in raw key space; with `exact_lower`/`hash`/`date` transforms a user can confirm the gate, choose dedup, and still hit a guaranteed fan-out hard-failure at execute time (minimal reproduction script confirmed; the INV-3 safety net protects the data but the flow dead-ends).

The systemic fix: when a review item says "wire gate X into path Y," the regression test should enumerate *all* sibling paths (train/tune/select; propose/execute; manual/agent) — three recurrences in one round is a process signal, not three coincidences.

**(2) The intelligence layer is evidence-starved, and its two feedback loops are built but unplugged.** The LLM touchpoints that decide plan fate see almost nothing: `final_review` receives key-name-level output summaries (not a single KS value) yet holds the power to mark a fully-DONE, user-confirmed plan FAILED (AGT-3); every step blocks on a near-zero-signal `llm_critique` call (AGT-6). Meanwhile the two mechanisms that would make the agent *smart* exist but are disconnected: the memory system's `compare_model_experience` API has **zero call sites** — V2 experiments are never written to memory and gate decisions never read historical KS anchors (MEM-1, the single biggest "make the agent smarter" lever) — and all three built-in modeling templates ship `success_criteria = ()` with no task-level injection path, so the "didn't meet the bar → replan" loop can never fire (AGT-4). On top of that, the distillation engine never receives an `llm_factory` so LLM summarization is permanently dormant (MEM-2), and memory capture has no idempotency, so re-runs inflate support/confidence which then dominates retrieval ranking (MEM-3).

**(3) Credit-risk capability now has depth but not the last mile.** The platform can train, calibrate, document and package a model — but has **no tool to apply a trained model to new data**, and the monitoring policy it emits is paper JSON with no execution path (DOM-3). Two methodology contradictions undermine trust: `tradeoff_view` assumes higher score = better customer while `reject_inference` assumes higher score = riskier, neither declares a direction parameter, and native model scores are PD (DOM-2) — with a weak LLM orchestrating, one side *will* be wired backwards and will produce plausible-looking but inverted business recommendations. Calibration evaluates Brier/ECE on its own fitting set (which is also the early-stopping and tuning set), yielding systematically optimistic calibration metrics (DOM-4).

**(4) The single-process synchronous core is now the main UX *and* reliability bottleneck.** JOIN propose/confirm/execute and dataset upload run synchronously inside `async def` endpoints, freezing the entire service — including all polling — during big-table operations (PERF-1, critical). V2 driver turns run in the HTTP request thread with no job row and no lock, so a double-clicked confirm or second browser tab triggers a concurrent `PlanExecutor.run` that mis-flags in-flight steps as restart orphans and FAILs them (REL-1, critical). The same turn is invisible: no busy state, no polling, no way to stop a minutes-long train (UX-1, critical). Restart recovery covers tasks/jobs but not the V2 plan layer — a RUNNING plan spins forever after a crash (REL-4).

**(5) The frontend has two stacks, and the good one is dead code.** Roughly 2,200 lines of v2 workbench modules (join_review, plan_view, loop_progress, subagent_view) are mounted nowhere and kept alive only by tests (UX-8, VD-10), while the *live* driver flow renders gate evidence as bare escaped-HTML tables with none of the chart language (ROC/KS SVG, databars, PSI color bands) already built for the validation flow (VD-1). Confirmation gates look like ordinary chat bubbles with an extra button (VD-2). In agent mode the timeline drops all structured gate controls, forcing every adjustment through weak-LLM text routing — the platform's weakest link (UX-2). Skeletons are still at zero (VD-3), and the real-logo glow animation assets generated for the mascot sit unwired in `scripts/` (VD-5).

**(6) "Run it as a product" dimensions are the biggest whitespace.** The completeness critic found the lenses all stopped at single-task runtime: CSV ingestion hardcodes UTF-8 and lets float64 truncate long ID columns (GBK files fail outright; JOIN keys corrupt silently) (GAP-1); deleting a task leaves raw credit data and orphan rows behind forever, unaudited (GAP-2); the audit log INV-8 pays 20+ transactional call sites for has no read/export surface at all (GAP-3); there is no data dictionary anywhere in the V2 chain (GAP-4), no model registry (GAP-6), no backup story (GAP-9), and 7 of 254 modules have a logger (GAP-10).

**ROI ordering.** Fix the three recurrences and the S-sized quick wins first (they are cheap and carry correctness weight), then stabilize the runtime spine (event loop, driver-turn jobs, restart reclaim), then close the intelligence loops (metric-aware evidence, memory wiring, success criteria, eval harness) — that batch is what moves "the agent gets smarter." Domain last-mile and the visible UX/visual work can proceed in parallel; the ops batch is steady background work.

---

## Top-10 Priorities

| # | ID | Area | Impact / Effort | Verified | One-line problem |
|---|----|------|-----------------|----------|------------------|
| 1 | REL-1 | Reliability | Critical / M | ✅ | Double confirm / second tab → concurrent `PlanExecutor.run` on the same plan; in-flight steps mis-flagged FAILED |
| 2 | PERF-1 | Performance | Critical / S | ⚠️ | JOIN & upload do heavy sync work inside `async def`; whole service freezes during big-table ops |
| 3 | UX-1 | UX | Critical / M | ✅ | Manual gate confirm runs the whole long turn with zero feedback, no busy state, no stop |
| 4 | MEM-1 | Memory | Critical / M | ✅ | Memory ↔ V2 fully disconnected both directions; `compare_model_experience` has zero call sites |
| 5 | AGT-1 | Agent | High / S | ✅ | `is_confirm` treats questions as confirmation and short-circuits before LLM routing (H4 recurrence) |
| 6 | DOM-1 | Domain | High / S | ✅ | `tune_hyperparameters` silently tunes on NaN labels; gate missing on the tuning path (H1/H2 recurrence) |
| 7 | PERF-2 | Performance | High / M | ✅ | Uniqueness/dedup in raw key space vs join in transformed key space → guaranteed execute-time fan-out failure (H3 recurrence) |
| 8 | DOM-2 | Domain | High / M | ✅ | Score-direction convention contradicts itself across tools; no `score_direction` parameter anywhere |
| 9 | DOM-3 | Domain | High / M | ✅ | No tool applies a trained model to new data; monitoring policy is paper JSON with no execution path |
| 10 | AGT-3 | Agent | High / M | ✅ | `final_review` sees key names only (no metric values) yet can FAIL a fully-successful plan or trigger blind replans |

---

## Quick Wins

Effort **S**, impact high/critical — recommended as the first batch (rough order of value):

1. **AGT-1** — add a question guard + full-string anchoring to `is_confirm` (regex change + regression cases).
2. **DOM-1** — wire `resolve_modeling_splits` / the NaN-label gate into `tune.py` and its manifest schema, mirroring the training recipes.
3. **PERF-1** — move JOIN/upload heavy work off the event loop (`def` endpoints or `anyio.to_thread.run_sync`); instantly un-freezes polling during big operations.
4. **PERF-5** — put worker imports on a diet (lazy-import the `marvis.db → packs.modeling → sklearn` chain) to cut the measured 1–2.3 s per-tool-step subprocess cold start.
5. **REL-2** — teach `is_metrics_failure` to recognize the reclaim message (or reclaim `COMPUTING_METRICS` to a metrics-specific failure) so a restart doesn't force a full notebook re-run; wire up the existing-but-dead `last_completed_step()`.
6. **MEM-2** — pass `llm_factory` to both `DistillationEngine` constructors; the LLM summarization path exists and is tested, it's just never enabled in production.
7. **MEM-3** — idempotent memory capture (dedup key on task+kind) so re-runs stop inflating support/confidence and skewing retrieval.
8. **UX-3** — add a "task still current?" guard to gate-widget `setAgentMessages` callbacks to stop cross-task message bleed.
9. **ARCH-3** — replace the 15 remaining `getattr(*_with_audit)` duck-type soft probes with hard requirements (2 of the fallbacks write no audit at all — residual INV-8 risk).

---

## Roadmap: Six Batches

**Batch 1 — Correctness & recurrence debt (mostly S).** The three recurrences (AGT-1, DOM-1, PERF-2) plus the small correctness items: ARCH-3 (audit soft-probes), UX-3 (cross-task bleed), UX-7 (second sample table silently dropped), DOM-9 (champion selection uses OOT KS, contradicting the tune-time "OOT reports only" doctrine), DOM-10 (report scoring column silently falls back to the first feature column), REL-9 (sync `execute_join_plan` branch lacks the job guard). Add sibling-path regression tests as the exit criterion.

**Batch 2 — Runtime spine for a single-machine product.** PERF-1 (event loop), REL-1 + REL-6 (driver turns get a job row, a lock, and progress events), UX-1 (busy state, message polling during turns, stop button), REL-4 (startup reclaim for RUNNING plans + restart notification for driver tasks), REL-2, PERF-5, REL-5 (job heartbeat/watchdog so a hung job can't lock a task forever via `idx_jobs_active_task`).

**Batch 3 — Close the intelligence loops (this is what makes the agent smarter).** AGT-3 (metric-aware output summaries: let critic/final-review actually see train/test/OOT KS), AGT-4 (success-criteria injection at the modeling-setup gate; never hardcode thresholds), MEM-1 (write V2 experiment outcomes into `model_experience`; read top-3 historical anchors into gate prompts as read-only reference — INV-4-safe), MEM-2/3/4, AGT-5 (give `route_instruction` the gate's parameter schema so "adjust" stops guessing parameter names), AGT-9 (deterministic red-flag checklist for modeling gates, not just JOIN/screen), LLM-1 (upgrade `json_object` to `json_schema`/guided decoding where the local stack supports it), LLM-3 (per-call token/latency/success audit), LLM-2 + TST-1 (make the eval framework runnable against a real model; build the degraded-output eval set — fences, prefixes, negation, truncation), AGT-7 (gate budget vs. 9-gate flows), AGT-8.

**Batch 4 — Domain last mile.** DOM-2 (`score_direction` enum + deterministic direction self-check against labels), DOM-3 (a `score_dataset` tool + monitoring-policy execution: PSI against the policy's reference distribution, threshold breach report), DOM-4 (calibrate on a holdout or report split-labeled metrics), DOM-5 (unified score bands with cumulative columns for cutoff work), DOM-6 (wire scenario `eval_metric` into champion selection; add lift/PR metrics), DOM-7 (feature-level PSI/CSI into screening and the univariate report sheet), DOM-8 (roll-rate observation-gap handling + balance-weighted option), DOM-11/12.

**Batch 5 — Experience users can see.** UX-2 (render structured gate controls in the agent timeline instead of text-routing everything), UX-4 (screen-gate table: search/sort/bulk ops/virtual scroll for hundreds of features), VD-1 (sink the existing chart language into V2 gate tables: databars for KS/IV, PSI bands, tabular-nums), VD-2 (a real "gate" visual form: tinted glass card, icon, consequence line, distinct confirm button), VD-3 (skeleton system), VD-4 (calibration reliability curve + score-distribution charts from data the backend already produces), VD-5 (wire the real-logo glow animation into product states), UX-5 (surface replans/subagents/loop events), UX-6 (dedup gate: show conflict examples the backend already computes), PERF-7 (stream LLM output to the UI), UX-10/11/12, VD-6/7/8/9.

**Batch 6 — Run it as a product.** GAP-1 (encoding detection + dtype defenses for ID columns), GAP-2 (task deletion lifecycle with audit), GAP-3 (audit log read/filter/export API + UI), GAP-4 (data dictionary layer feeding gates, reports and LLM context), GAP-6 (model registry), GAP-7 (content-fingerprint dataset reuse), GAP-5/8/9/10 (loopback trust model, LLM precheck, backup command, logging baseline), TST-2 (upload size guards/streaming), TST-3 (one real end-to-end journey against real FastAPI), TST-4 (true kill/OOM isolation tests), TST-5 (test tiering/markers), TST-6/7/8/9, ARCH-2/4/6/9 (tools.py split, turn-handler dedup, pipeline.py split, template file split), ARCH-11 (app.js state ownership), PERF-6/9/10, UX-8/9, VD-10/11, ARCH-10, LLM-4..10.

---

## 1. Agent Intelligence & Orchestration

**Lens verdict.** The V2 orchestration skeleton (template → validator → executor → mandatory confirmation gates) is solid, and the commitments made in the 6-28 report — JSON extraction retry, negation-semantics guard, red-flag checklist, the success_criteria mechanism, and sub-agent pauses no longer faking success — have all been verified as landed. But the intelligence layer still has four classes of structural shortfalls: (1) is_confirm is not fully fixed — interrogative sentences and embedded affirmative words ("这样可以吗？" / "结果不是很好的" — "Is this okay?" / "The result is not very good") are still treated as confirmation, and the check short-circuits before LLM routing, so it can falsely trigger side-effect gates such as execute_join or model delivery; (2) LLM touchpoints are broadly "evidence-starved" — llm_critique and final_review see only key-name summaries of the outputs (not a single KS number), yet hold the power to mark an entirely successful plan FAILED, while every step blocks synchronously on one LLM call and manual mode renders "警告" (warning) noise on every step; (3) two closed loops are "mechanism present, wiring empty" — the memory system's compare_model_experience has zero call sites (planning and gate decisions have no historical anchors), and the built-in modeling templates carry success_criteria=() with no task-level injection point; (4) the AUTO driving loop has three rough edges — the 8-gate budget deterministically and silently exhausts on multi-table modeling flows (9 gates), confidence is parsed but never used, and the replan decision feeds the goal text back as user input, causing a double LLM hop that can be pre-empted by is_confirm. The 10 findings below are ordered by impact, all with file:line evidence from the current working tree.

### AGT-1 · is_confirm still treats interrogatives and embedded affirmative words as confirmation, short-circuiting before LLM routing and directly releasing side-effect gates (H4 not fully fixed)

**Impact:** High · **Effort:** S · **Verification:** ✅ Confirmed adversarially

**Problem.** The 6-28 fix only added the negation guard; it did not implement the "full-string anchoring for short confirmations" recommended in the same report item. When a user raises a question on a needs_confirmation gate such as execute_join, experiment selection, or a model delivery action (export PMML/handoff) — e.g. "这样可以吗？" ("Is this okay?") — or gives a mixed evaluation ("好的地方是命中率，但……" — "The good part is the hit rate, but…"), it is treated as a confirmation and executed directly; in agent mode the LLM instruction routing is short-circuited, so there is no chance to correct course at all.

**Evidence.** marvis/agent/plan_driver.py:32-48: `_CONFIRM` uses `.search` substring matching (including "可以/好的/对/继续" — "okay / fine / right / continue"), `_NEGATED_CONFIRM` only blocks explicit negation words, with no interrogative guard and no full-string anchoring. Empirically tested (py_313 environment): `is_confirm('这样可以吗？')==True`, `is_confirm('结果不是很好的')==True`, `is_confirm('对不起，这个结果有问题')==True`, `is_confirm('KS高吗，可以到0.3吗')==True`. plan_driver.py:162-167 `resume()` checks is_confirm before route_instruction; on a hit it calls `confirm_step(gate.id)` and continues execution; the C1 role confirmation at turn_handlers.py:675 uses the same function.

**Why it matters.** This is the first guard on the confirmation gates; a false release is equivalent to executing a side-effect action the user never approved (join persistence, model delivery/handoff), bypassing the semantics of INV-3 "mandatory confirmation". Interrogative sentences are extremely frequent in a conversational product — this is not a long-tail case.

**How to fix.**
1. Add an interrogative guard: `_QUESTION = re.compile(r'[?？]|吗|吧$|行不行|可不可以|能不能')` — on a hit, return False.
2. Switch short confirmations to full-string anchoring: after stripping whitespace/punctuation, only `re.fullmatch(r'(好的|好|可以|确认|确定|没问题|同意|就这样|继续|开始|对|对的|ok|okay|yes|go|proceed)+', compact, re.I)` counts as a confirmation; abolish the substring search.
3. In agent mode, unmatched text naturally falls through to the existing route_instruction (its confirm classification can catch true confirmations as a backstop); in manual mode it falls to a canned hint — both are safe degradations.
4. Add regression cases: the four empirically tested strings above must be False; '确认' ("confirm") / 'ok 继续' ("ok continue") must be True; `turn_handlers._parse_c1_reply` benefits from the same source.

**Verification note.** The adversarial pass reproduced all four false positives by importing the real module in the py_313 environment ("好的" hits inside "不是很好的" since "不是" is absent from the negation list; "对" hits via "对的?"). It also confirmed there is no upstream backstop — user text reaches `driver.resume` unfiltered (the only interception is the stop-intent check at service.py:437), the plan-level VALIDATED gate at plan_driver.py:137-141 has the same short-circuit, and tests/test_plan_driver.py:1565-1577 covers only negation cases, with no interrogative coverage.

### AGT-3 · final_review's LLM sees only a "key-name-level" summary yet has the power to mark a fully successful plan FAILED or trigger blind replanning

**Impact:** High · **Effort:** M · **Verification:** ✅ Confirmed adversarially

**Problem.** With input consisting of nothing but key names and a four-character goal, a weak model emitting goal_met=false is a high-probability event; once that happens, a plan in which every step passed its deterministic post_check and the user confirmed every gate will either have redundant steps appended and re-run (repeated training/delivery) or be marked FAILED outright. The same shallow summary is also fed to the per-step llm_critique (reviewer.py:60-63), giving the "second pair of eyes" zero signal on the most critical modeling steps.

**Evidence.** marvis/orchestrator/reviewer.py:292-305 `_summarize_output` collapses nested dicts into `{'type':'object','keys':[...]}` and lists into a count — all numeric values such as train_model's `metrics.oot.ks` are invisible; reviewer.py:118-127 `_llm_summarize`'s prompt contains only goal/plan_id/step_count/that summary; planner.py:121 sets template plans' `goal=template.title` (i.e. the four characters "标准建模" — "Standard Modeling"); reviewer.py:104 goal_met requires `llm_goal_met is not False`; executor.py:323-331 on goal_met=False first tries `_try_final_review_replan` (asking the LLM to "fill in remaining steps" for a plan that is entirely DONE), and on failure sets FAILED.

**Why it matters.** This directly determines the task's terminal state and the user's trust in the agent; it is a structural source of "inexplicable failures / inexplicable re-runs" on 32-72B weak models, and it also wastes replanning budget.

**How to fix.**
1. Make `_summarize_output` metric-aware at depth 2: keep real numeric values for dict entries whose key ∈ METRIC_FIELDS (reuse the set at validator.py:16-29) or whose values are numeric (capped at 20 keys / 600 characters), so the critic and final review actually see train/test/oot KS, AUC, PSI.
2. On from_template, splice a summary of the key slots into goal (e.g. "标准建模: dataset=X, target=y, recipe=lgb").
3. Narrow the authority: when llm_goal_met=False and criteria_failures is empty, route through goal_doubt→REVIEW (human re-check; the channel already exists at executor.py:320-322); FAILED and automatic replanning should be triggered only by deterministic success_criteria failures — the LLM's opinion may pause, but never veto (echoing the spirit of INV-1).

**Verification note.** The adversarial pass found the summary is even shallower than claimed: final_review passes `{step_id: output_dict}` (reviewer.py:98/123), so each step's entire output collapses to its first 10 key names — even top-level scalars are invisible. Existing tests (tests/test_orch_executor.py:345-375, tests/test_orch_reviewer.py:186-221) actually lock in the "all steps DONE + LLM goal_met=false → FAILED" behavior, and all three capability tiers have decision_point_replan=True (capability.py:26/36/46), so the LLM veto is replannable by design.

### AGT-2 · Memory system still has zero integration with V2 planning and gate decisions: compare_model_experience has zero call sites repo-wide (re-check confirms the unimplemented item)

**Impact:** High · **Effort:** M · **Verification:** ✅ Confirmed adversarially

**Problem.** Every gate decision (decide_gate) and every planning pass across the entire V2 modeling/JOIN/feature flow is "amnesiac": the agent cannot see historical KS ranges for similar datasets, the tuning conclusions of the last run with the same recipe, or the hit rate of the last join on the same tables. When judging "is this hit rate abnormal / is this KS acceptable" it has no numeric anchor whatsoever and can only rely on the weak model's gut feeling.

**Evidence.** marvis/agent_memory/retrieval.py:126 already implements `compare_model_experience` (with scope/model_family matching and confidence), but apart from the re-export at agent_memory/__init__.py:43 it has no call sites anywhere in the repo; grep for "memory" in marvis/agent/plan_driver.py, auto_drive.py, turn_handlers.py, gate_payloads.py yields 0 hits; orchestrator/subagent.py:131 calls `planner.generate(memory_context={})`; the memory_context at routers/plans.py:63-66 comes only from the HTTP request body; the only real consumer of memory_context left is the V1.1 validation path (agent/validation_stages.py:240-263).

**Why it matters.** This is the biggest lever for "a smarter agent": only gate decisions with historical anchors can evolve from "format correct" to "judgment correct"; it is also the payoff point for the investment in the memory subsystem (store/distillation/retrieval are all fully built).

**How to fix.**
1. When build_modeling_proposal / gate_payloads assemble the "选择实验" ("select experiment") and tuning gates, call `compare_model_experience(store, scope=scenario+target_type, model_family=recipe)`, take the top-3 and generate a read-only table {source task, recipe, oot_ks, oot_auc, confidence} written into gate `metadata['memory_anchor']`.
2. In `auto_drive._format_gate` (auto_drive.py:113-192), append a 【历史同类实验(只读参照)】 ("historical similar experiments — read-only reference") section after the red-flag checklist, with the prompt explicitly stating "for reference only; must not substitute for the platform's numbers".
3. Reuse the `audit_agent_memory_use` paradigm from validation_stages.py:263 to record use audits, with entries annotated with source_task_id+confidence.
4. Cap at 3 entries, each ≤120 characters, passed through fit_to_budget. Read-only throughout, strictly observing INV-4 (no change to deterministic behavior).

**Verification note.** The verifier confirmed zero business callers (only the `__init__.py` re-export and unit tests reference the function) and that `decide_gate`'s signature (auto_drive.py:73) has no memory parameter, while the frontend never populates memory_context (0 grep hits under marvis/static/js). One minor correction: memory consumption is not limited to validation_stages.py:240-263 — there are 3 consumption points in validation_stages.py plus 3 more in routers/validation_agent.py — but all of them belong to the V1.1 validation path, consistent with the finding's thesis that the entire V2 flow is memory-blind.

### AGT-4 · Built-in modeling templates carry success_criteria=() with no task-level injection point — the success-criteria loop is effectively idle for the core modeling flow

**Impact:** High · **Effort:** M · **Verification:** ✅ Confirmed adversarially

**Problem.** The 6-28 update said "success_criteria has entered plans and final review" — the mechanism is indeed there, but all three built-in modeling templates carry an empty tuple, and there is no entry point on the user/task side to write "this project's OOT KS must be at least X" into a plan. As a result, final_review's only deterministic anchor is missing, goal_met degenerates into "all steps DONE + weak-model key-name-level opinion" (see AGT-3), and the "auto-replan on unmet criteria" loop never fires.

**Evidence.** marvis/orchestrator/templates/sample.py:13-17 `_BINARY_MODELING_SUCCESS_CRITERIA = ()` (comment: thresholds belong to a configurable policy layer), referenced at :255 (standard_modeling), :555 (modeling), :801 (modeling_with_join); reviewer.py:97/308-343's `_evaluate_success_criteria` and the executor's criteria-driven replanning chain are complete; the plan table also has the column (db_schema.py:286), and skills.py:50-53 allows user skill templates to carry it; but `PlanDriver.build_plan` (plan_driver.py:222-236) and modeling_setup have no success_criteria injection path at all.

**Why it matters.** This is the closed-loop switch for "better, more accurate results": without configurable success criteria the agent cannot autonomously judge "the model isn't good enough — try another round", and the long-term goal (polishing the modeling agent so it never fails a KS baseline) lacks its execution mechanism.

**How to fix.**
1. Add an optional success_criteria parameter to `PlanDriver.build_plan`, landing on `plan.success_criteria` (field and DB are already in place).
2. Add an optional `oot_ks_min` control to the choose_modeling_spec confirmation gate (the modeling_setup gate), defaulting to empty = no criteria set, and never hard-coding a number — respecting the "early concrete numbers are obsolete" decision; once the user or AUTO fills it in, generate `[{"metric":"oot_ks","min":x,"aggregate":"max","label":"OOT KS"}]`.
3. Once AGT-2 is wired up, the P25 of historical similar experiments can serve as the suggested default (still requiring human confirmation).
4. On unmet criteria, use the executor's existing criteria_failures→replan loop (guarded by max_replan_iterations against metric gaming).

**Verification note.** The verifier confirmed no injection point exists anywhere: modeling_setup's template_slots (modeling_setup.py:58-110), the adjustment whitelist (agent/adjust_specs.py:7-12), and the agent/api layers all lack success_criteria, and user templates cannot shadow builtin template IDs (templates/__init__.py:54-57). It also noted the empty tuple is a deliberate, commented design decision deferring thresholds to a "configurable policy layer" — but the promised layer is only partially present (packs/modeling/tools.py's selection_policy supports metric_thresholds, yet the defaults contain only require_pmml/require_handoff and there is no user entry point to set metric thresholds), so the finding stands.

### AGT-6 · Every step synchronously blocks on a nearly signal-free llm_critique; in manual mode each step renders a "警告 llm critique unavailable" noise line

**Impact:** Medium · **Effort:** M · **Verification:** — Not independently verified

**Problem.** A triple loss: (a) in agent mode, a local 32-72B model incurs one extra 5-20s LLM round-trip per step — a 12-step modeling plan waits an extra 1-4 minutes on average — while the critique, seeing no numeric values at all (same root cause as AGT-3), has essentially no chance of catching a real problem; (b) in manual mode (no LLM), every step deterministically produces a failed verdict, and the plan view hangs "警告 llm critique unavailable: 请先在设置中配置…" ("warning: llm critique unavailable — please configure it in settings first…") on every row, drowning out the deterministic failures that actually deserve attention; (c) that verdict also enters the step.completed hook's review_warnings count (executor.py:846-858), polluting downstream observability signals.

**Evidence.** marvis/orchestrator/executor.py:199-201 unconditionally calls `self._reviewer.llm_critique` (synchronous, serial) for every step that passes the deterministic checks; reviewer.py:60-63 shows the critique input is only goal + step.title + the key-name-level summary; without a configured LLM, app.py:367-371's factory→resolve_llm_model (llm_settings.py:101-102) raises LLMSettingsError, caught at reviewer.py:82-83 into a passed=False verdict; static/js/v2/plan_view.js:104-111 renders it as one "警告" (warning) per step plus English exception text.

**Why it matters.** It directly hurts "higher efficiency" (minute-level latency per plan) and "more professional" (users without an LLM see a screen full of English warnings, damaging trust); the critique's cost/signal ratio is currently near its worst possible.

**How to fix.**
1. Skip when there is no LLM: have llm_critique catch LLMSettingsError separately and return `reviewer='llm_critic', passed=True, reasons=['skipped: no LLM configured']` (or add a `status='skipped'` field); plan_view.js does not render a warning for skipped.
2. Narrow the trigger surface: only critique steps that are decision_point, needs_confirmation, or whose output contains METRIC_FIELDS (add a gate before executor.py:199); profile/read-type deterministic steps skip outright.
3. Share the depth-2 numeric summary with AGT-3 so the critiques that remain can actually see KS/AUC and output valuable reasons.

### AGT-5 · route_instruction's context is only the gate title: the LLM can only blind-guess parameter names, artificially depressing adjust success rates

**Impact:** Medium · **Effort:** S · **Verification:** — Not independently verified

**Problem.** When the user says "阈值放宽到 0.1" ("relax the threshold to 0.1"), the routing LLM cannot see which parameters the current node has, their current values, or their bounds — it can only guess key names; a wrong guess → apply_adjust matches nothing → the reply "没有识别到可调整的参数" ("no adjustable parameter recognized") followed by a listing of the available parameters (gate_execution_adapter.py:137-144) — a parameter list that could have been fed to the routing LLM the first time around. Multi-turn scenarios ("再加严一点" — "tighten it a bit more") also inevitably fall into clarify because there is no conversation history.

**Evidence.** marvis/agent/plan_driver.py:186-187: `context = gate.title if gate is not None else ...; route = route_instruction(self._llm, gate_context=context, instruction=user_text)` — the tables parameter supported by the signature is never passed; instruction_router.py:65-71 `_format` only concatenates the gate title + the instruction. Yet the gate envelope already declares structured controls (gates/contracts.py:279-320 `infer_gate_envelope` generates leakage_ks/max_missing_rate/selection controls for screen gates, with default/bounds), and apply_adjust requires params keys to hit dep.inputs (gate_execution_adapter.py:122).

**Why it matters.** Free-text parameter adjustment on gates is the core interaction of agent mode; a weak model's key-name hit rate without schema context is very low, and the user's felt experience is "it doesn't understand what I say — several rounds of back-and-forth".

**How to fix.**
1. Have `plan_driver._handle_instruction` assemble: `envelope=extract_gate_envelope(...)` (from the gate message metadata), `controls_desc=[{id,label,default,bounds} for c in envelope.controls]`, `dep_current={dep.id: dep.inputs}`; pass these together with the rendered tables and the last 1-2 conversation turns into route_instruction (the signature already reserves tables).
2. Add a constraint to `_SYSTEM`: "params keys may only be taken from the parameter names in the 【可调参数】 (adjustable parameters) list, and values must fall within bounds".
3. After the route returns, run deterministic validation against controls/bounds first; illegal keys immediately echo a hint with the list attached (no longer consuming an execution round-trip).

### AGT-7 · AUTO drive's 8-gate budget deterministically and silently exhausts on multi-table modeling flows (≥9 gates); confidence is parsed but never used

**Impact:** Medium · **Effort:** S · **Verification:** — Not independently verified

**Problem.** In AUTO mode running multi-table modeling, the 9th gate (usually the final model delivery action) inevitably stops at "确认请回复「确认」继续" ("to confirm, reply 「确认」 to continue") with no explanation whatsoever that the autopilot budget is used up — from the user's viewpoint the agent goes on strike for no reason. Meanwhile a weak model's confirm at confidence=0.3 and one at 0.95 are treated with completely equal weight; low-confidence confirmations have no line of defense.

**Evidence.** marvis/agent/turn_handlers.py:38 `AGENT_MAX_GATES=8`; :456 `for _ in range(AGENT_MAX_GATES)` — when the loop exhausts, the function simply ends with no notification message. The modeling_with_join template contains 7 needs_confirmation steps (sample.py:618/675/693/717/760/778/796) + the plan overview gate + the C1 role confirmation gate (turn_handlers.py:354 emits join_c1 first; latest_open_gate:551 counts it as a gate) = at least 9 decisions in a single turn; any adjust/dedup re-stop consumes extra budget. auto_drive.py:319-321 parses confidence and puts it into the decision; turn_handlers.py:473-482 only stores it into message metadata, with no policy consumption of any kind.

**Why it matters.** This is a deterministic break in the AUTO mode's main path (not a long tail), directly manufacturing the "agent inexplicably stops" experience; the blind collection of confidence also wastes a ready-made safety signal.

**How to fix.**
1. When the loop exhausts naturally and latest_open_gate is non-empty, append a message: "已连续自动处理 8 个节点，为安全起见转人工确认；回复任意内容可继续自动执行" ("8 nodes have been auto-processed consecutively; switching to manual confirmation for safety — reply with anything to continue automatic execution") (S, a few lines).
2. Make the budget follow the capability tier (conservative/balanced/autonomous = 6/10/16), placed in capability.py.
3. Consume confidence: `action=='confirm'` with `confidence<0.5` → change to halt (add one policy in `_apply_safety_policy`, configurable threshold); low-confidence confirmations get routed to a human.

### AGT-8 · AUTO's replan decision feeds replan_goal back as user text: a double LLM hop that can be pre-empted by is_confirm into a confirmation

**Impact:** Medium · **Effort:** S · **Verification:** — Not independently verified

**Problem.** Two consequences: (a) the replan_goal written by a weak model often contains phrases like "……并继续调参" ("…and continue tuning"), hitting is_confirm's "继续" ("continue") substring (same root cause as AGT-1) and directly confirming the very gate that was supposed to be restructured; (b) even without a word collision, an extra LLM round-trip is spent letting route_instruction re-classify the replan decision it itself just made, and the second classification may misjudge it as clarify — the autodrive loop exits as soon as it sees a non-gate message, so the replan intent is simply lost.

**Evidence.** marvis/agent/turn_handlers.py:512-519: when `action=='replan'`, `turn_fn(runtime, repo, task, user_text=decision.get('replan_goal') ...)`; that text enters `PlanDriver.resume`, first passes is_confirm (plan_driver.py:162), then falls into `_handle_instruction`→route_instruction for a second LLM classification (plan_driver.py:186-198) before it can possibly reach apply_replan. Yet the structured entry points `executor.replan_from_instruction` (executor.py:573-613) and `gate_execution.apply_replan` (gate_execution_adapter.py:85-107) both already exist.

**Why it matters.** Replanning is the key path for failure recovery and for "a smarter agent"; the decision is already structured yet is deliberately degraded back to free text, adding latency and introducing two failure modes.

**How to fix.**
1. Stop feeding text back in the turn_handlers replan branch: directly take the active plan → `runtime.plan_executor.replan_from_instruction(active.id, decision['replan_goal'])`, and after success call the driver to continue (a thin `resume_structured(action='replan', goal=...)` entry on PlanDriver can reuse apply_replan's message assembly).
2. The adjust branch already goes through structured adjust_params; keep it consistent.
3. Fixing AGT-1 further shrinks the risk here, but the structured direct connection should still be made (saves one LLM round-trip).

### AGT-9 · The gate-decision red-flag checklist covers only JOIN/screening; modeling gates (tuning / experiment selection) have no deterministic red flags at all

**Impact:** Medium · **Effort:** S · **Verification:** — Not independently verified

**Problem.** At the most critical modeling decision gates (which hyperparameter set to pick, which experiment to select), AUTO receives tables of bare numbers; "severe overfitting at train_ks=0.52/oot_ks=0.18" or "near-random oot_auc=0.53" will not show up in the 【平台红旗 checklist】 ("platform red-flag checklist"), so the weak model must perform numeric comparisons itself across a 20-row table — precisely what it is worst at. The red-flag mechanism itself (having the LLM re-check the platform's deterministic outputs rather than invent its own) is right; the coverage merely stopped at JOIN.

**Evidence.** marvis/agent/auto_drive.py:195-246: `_extract_red_flags` only recognizes screen/dedup metadata and the text "行数发生变化/膨胀" ("row count changed / inflated"); `_table_red_flags` only looks for columns whose names contain "命中率/膨胀/去重" ("hit rate / inflation / dedup"); yet the modeling gates' tables explicitly carry train_ks/test_ks/oot_ks/test_auc/oot_auc (renderers.py:196-226 tuning leaderboard, :260-267 experiment comparison), and there is no red-flag generation logic for overfitting gaps, near-random AUC, or PSI overruns.

**Why it matters.** This directly determines the quality of AUTO-mode modeling decisions (choosing the wrong experiment = shipping a bad model); it is also the low-cost, high-return move of extending the "red-flag re-check" paradigm — already validated effective in the 6-28 round — to the core business scenario.

**How to fix.**
1. When gate_payloads/composer assembles a modeling gate, deterministically compute red flags from the structured output (not from the table strings): train_ks-oot_ks>0.08 (configurable), oot_auc<0.55, psi>0.25, any split with samples<500, best-trial-to-runner-up gap smaller than the noise band, etc., written into `meta['red_flags']=[...]`; thresholds live in the platform configuration layer (observing INV-1: the LLM only re-checks, it never computes).
2. Have `_extract_red_flags` read `meta['red_flags']` first, keeping table-string parsing only as a legacy fallback (the existing `_parse_rate` string parsing is fragile anyway).
3. Display alongside AGT-2's historical anchors in the same section: "red flags + historical ranges" is the minimal sufficient evidence for a weak model to make a correct halt/confirm decision.

### AGT-10 · planner's generate/replan/explore paths still use strict json.loads, not reusing load_json_object's fence stripping

**Impact:** Low · **Effort:** S · **Verification:** — Not independently verified

**Problem.** Some local inference runtimes do not strictly enforce `response_format={'type':'json_object'}`; when the weak model outputs code fences or leading/trailing explanatory text, free planning's first attempt necessarily fails: generate has 2 retries so it can self-heal, but wastes two expensive round-trips; replan/explore have only 1 retry (MAX_REPLAN_PARSE_RETRY=1, planner.py:32), so a fence issue appearing twice in a row means ReplanError → the failure-recovery/exploration segment gives up outright.

**Evidence.** marvis/orchestrator/planner.py:296-299 (`_parse_plan_json`), :499-502 (`_parse_steps_json`), :512-517 (`_parse_json_object`) all call `json.loads(raw)` directly and raise `PlanningError('not json')` on failure; whereas decide_gate/route_instruction/reviewer have already been unified onto marvis/agent/json_reply.load_json_object (json_reply.py:9-30, with ```json fence handling and first {..} block extraction).

**Why it matters.** Replanning is the last intelligent line of defense in failure recovery; giving up because of a preventable formatting issue is a real waste, and the fix is a few lines of consistency convergence.

**How to fix.**
1. Replace the three json.loads sites with load_json_object (note that it returns a dict on the steps path: `data,err=load_json_object(raw); if data is None: raise PlanningError(f'not json: {err}')`; for the top-level-list case in `_parse_steps_json`, try json.loads first and fall back to object extraction), keeping the PlanningError semantics and the existing retry re-feeding unchanged.
2. Add a regression case: "a valid steps payload wrapped in ```json fences".

---

## 2. Memory System & Experience Reuse

**Lens verdict.** The memory subsystem itself is of good quality: store / retrieval / distillation / evolution / audit / redaction / dual policy gating (both the reference_cross_task and auto_distill paths are correctly gated) are all in place, and transactions plus rollback have been properly fixed. But today it is merely "an accessory bolted onto the V1.1 validation agent". The core fault line is a bidirectional disconnect between the V2 JOIN/FEATURE/MODELING mainline and memory: V2 experiment results are never written into model_experience, AUTO gate decisions never read historical KS anchors, and the ready-made compare_model_experience API has zero call sites — this is the single biggest idle asset for making a weak model "smarter". The second tier is quality-mechanism gaps: the distillation LLM summary path is permanently dormant because llm_factory is never wired; duplicate capture has no idempotency, inflating support/confidence, which in turn dominates retrieval ranking; scoring has no time decay; raw memories have no low-confidence filter; and there is no negative-feedback loop at all. The third tier is observability and UX fragments: the memory.after_save trigger never fires and distillation swallows errors silently at three layers; the explicit "请记住" ("please remember") entry point is too narrow and messages containing skill/runtime substrings are silently dropped; distillation payloads flow into the prompt with no budget. All recommendations follow "read-only anchors + deterministic scoring + mandatory confirmation unchanged" and do not touch INV-1/INV-3/INV-4.

### MEM-1 · Memory system is bidirectionally disconnected from the V2 three-stage driver: V2 modeling results are never written to memory, gate decisions never read memory, and historical KS anchors remain an idle capability

**Impact:** Critical · **Effort:** M · **Verification:** ✅ Confirmed adversarially

**Problem.** The memory subsystem is roughly 3000 lines (store/retrieval/distillation/consolidation/audit all present) but is wired only into the V1.1 validation agent (validation_stages.py / validation_agent.py). On the core V2 JOIN→FEATURE→MODELING product line: (1) the KS/AUC/PSI produced by every training experiment never lands in model_experience — the more the user runs, the less the platform "remembers"; (2) when AUTO-mode decide_gate judges "is this KS abnormal", it has no historical same-kind model range anchor whatsoever, so the weak model must judge from nothing. This matches the section B proposal of the 6-28 report, and that item is confirmed as not done in the "still not landed" list.

**Evidence.** In marvis/agent/auto_drive.py:73-106, decide_gate's prompt is built solely from _format_gate(gate) (auto_drive.py:113-192, gate content + red flags + tables), with no memory import of any kind; grep 'memory' across marvis/agent/plan_driver.py and marvis/agent/turn_handlers.py yields 0 hits. Write side: the only automatic capture point in the entire repo is the V1.1 pipeline's _capture_agent_memory_for_metrics_success (marvis/pipeline.py:1766-1789, call site pipeline.py:443); the V2 plan-completion path done_message (marvis/agent/plan_message_composer.py:76-102) receives the terminal step's full training-metrics output yet produces no MemoryCandidate. Read side: the ready-made API compare_model_experience (marvis/agent_memory/retrieval.py:126-184, including metric dimensions and usage-guideline text) has zero call sites outside the agent_memory package — a fully implemented dead capability.

**Why it matters.** This is the single trunk path by which "memory makes a weak model smarter". A 32-72B model has no business prior for whether "KS=0.18 is good or bad"; the KS range of historical models with the same scope / same family is the only deterministic anchor the platform can supply. Without it, the quality of AUTO-mode confirm/halt decisions cannot improve, and the experience from the user's repeated experiments all evaporates. It directly affects "cross-task experience reuse" and the long-term modeling-agent goal (never failing a KS baseline).

**How to fix.** Two steps, both under INV-4 (read-only, no change to determinism):

1. Write side: when a V2 modeling plan reaches DONE (at plan_message_composer.done_message, where the terminal step is already located and build_model_delivery_payload already extracts the metrics, or in the api layer after the done message is appended), construct payload={model_name: experiment name from task/slots, scope: template_id + target column, channel/month: fall back to "未标注" ("unlabeled") when unavailable (copy the fallback style of pipeline.py:1844), ks/auc/psi: the oot or test metrics from the terminal output, source_task_id, important_feature_sources: table origins of the selected features}, run it through extract_model_experience→store.create, gated by auto_distill (reuse the gate pattern of api_support.capture).
2. Read side: in turn_handlers.agent_autodrive_turn (before decide_gate is called at turn_handlers.py:461), for modeling tasks call compare_model_experience(current_payload, store.list_entries(memory_type='model_experience')), compress the result to ≤3 items of ≤200 characters each in the form "【历史同类模型参考】KS 区间 x–y（n 个历史任务，来自记忆，仅供风险提醒，不作为指标依据）" ("historical same-kind model reference: KS range x–y, from n historical tasks, sourced from memory, risk reminder only, not a basis for metrics"), and append them to _format_gate's lines; also write memory_references into the decision message metadata via the existing audit_agent_memory_use audit path. The final action still passes through _apply_safety_policy as a backstop — memory only influences explanation and halt inclination and never touches any platform computation.

**Verification note.** The adversarial pass confirmed the bidirectional break in full and found it slightly broader than claimed: extract_memory_candidates (extractors.py:136) is also zero-call dead code, the only store.create write points repo-wide are pipeline.py:1787/1815 and api_support.py:41, memory-context injection happens exclusively in the V1.1 validation flow (api.py:754 used only by routers/validation_agent.py:204/271/309), and the consolidation triggers (consolidation.py:10) provide no fallback ingestion path for V2 results.

### MEM-2 · The distillation LLM summary path is permanently dormant in production: neither DistillationEngine construction passes llm_factory, so all experience sentences are low-quality template sentences

**Impact:** High · **Effort:** S · **Verification:** ✅ Confirmed adversarially

**Problem.** The carefully written DISTILL_SYS (anti-hallucination constraints, "introduce no new facts" rule) and build_distill_prompt have never executed in the product. All "distilled experience" the user sees — and that gets injected into prompts — is template-concatenated sentences; the task_experience ones are even dict literals, neither professional nor carrying semantics a weak model could actually exploit.

**Evidence.** marvis/app.py:277-282 ConsolidationScheduler(DistillationEngine(memory_store), ...) and marvis/routers/agent_memory.py:108-113 (the manual consolidate endpoint) both pass only the store argument; DistillationEngine._summarize (marvis/agent_memory/distillation.py:219-231) always takes _template_summary when self._llm_factory is None. _template_summary (distillation.py:277-294) emits f"任务经验：{structured.get('outcome_tags', {})}" for task_experience (printing a Python dict repr straight into the experience sentence), and for validation_pitfall only emits uninformative sentences like "xx 类问题重复出现" ("problems of type xx keep recurring").

**Why it matters.** Distilled summaries are the main channel by which memory enters the prompt (retrieve_with_distillations prioritizes distillations, retrieval.py:110-116); their readability directly determines whether a weak model can use the experience, and it directly shapes the user's perception of the memory management panel ("the experience this system summarizes reads like logs, not experience").

**How to fix.**

1. S-sized wiring: the assembly site in app.py already has LLM profile loading logic (the OpenAICompatibleLLMClient used in agent mode); construct llm_factory=lambda: OpenAICompatibleLLMClient(load_model_profile(settings)) and pass it into both DistillationEngine constructions. When the LLM is unavailable, the existing try/except already falls back to template sentences — zero risk.
2. Humanize the task_experience branch of _template_summary ("共 N 条任务经验：completed x 次、failed y 次，常见失败类型 z" — "N task-experience entries in total: completed x times, failed y times, common failure type z") as an improved no-LLM fallback.
3. Note that consolidate exists both as a background thread and as a synchronous endpoint; LLM calls must carry a short timeout to avoid blocking manual consolidate requests.

**Verification note.** The verifier additionally established there is no hidden wiring anywhere: app.py:308 does create `_llm_factory(settings)`, but only feeds it to orchestrator components (IntentRouter/Planner/Reviewer), never to DistillationEngine; llm_factory is passed only in tests, and the template output flows verbatim into prompts via retrieval.py:341 → prompting.py:110.

### MEM-3 · Memory capture has no idempotent dedup: rerunning the same task inserts duplicates; distillation support_count counts entries instead of distinct tasks, inflating confidence which in turn dominates retrieval ranking

**Impact:** High · **Effort:** S · **Verification:** ✅ Confirmed adversarially

**Problem.** The same validation task rerun 4 times → 4 model_experience entries with byte-identical payloads → distillation support=4 → confidence='high' → retrieval score +300. An experience backed by only 1 independent data point gets marked as highest-confidence and injected into the prompt with priority — "repetition" is mistaken for "independent corroboration". The same holds for field_convention/task_experience.

**Evidence.** marvis/agent_memory/store.py:66-114 create() INSERTs directly, with no dedup of any kind on (memory_type, source_task_id, payload); marvis/pipeline.py:443 triggers capture on every successful metrics stage (metrics supports retries/reruns). marvis/agent_memory/distillation.py:161 `support = len(members)`, distillation.py:13 CONFIDENCE_THRESHOLDS={'high':4,'medium':2}; yet _merge_model_experience (distillation.py:259) already computes the deduplicated source_task_ids and does not use it for support. Downstream, in marvis/agent_memory/store.py:1022 _distillation_score, confidence contributes high=300/medium=200/low=100 — the absolutely dominant term in the retrieval total score.

**Why it matters.** Confidence is the memory system's core quality metric: it decides retrieval ranking (_distillation_score), decides whether to inject (retrieve_with_distillations skips low), and is the trust label shown to users. Once inflated, both the weak model and the user will overrate single-source experience — in direct conflict with the statistical meaning of the "historical KS range anchor".

**How to fix.** Two small changes:

1. store.create: before INSERT, look up an active entry with the same memory_type + source_task_id + payload_json; on a hit, only touch updated_at and append a 'create' event (details={dedup:true}) without inserting a new row (keeping the audit trail complete).
2. _distill_group: for model_experience/task_experience, compute support as the count of distinct source_task_id values: support=len({_entry_payload(m).get('source_task_id') or _entry_id(m) for m in members}); _merge_model_experience already has this set available for reuse.
3. Add a test: after distilling 4 duplicate entries from the same task, confidence must be 'low', not 'high'.

**Verification note.** The verifier confirmed the entire causal chain against current code with no offsetting mechanism anywhere: the table has no UNIQUE constraint beyond PRIMARY KEY(id) (store.py:862-875), rerun is a real path (_reset_agent_task_for_rerun at api.py:134; "reset task before rerunning pipeline" at pipeline.py:1266/1292), the evolution path also recomputes confidence from the inflated support (store.py:473-489), and the only discrepancy is a one-line drift — `support = len(members)` sits at distillation.py:160, not 161.

### MEM-4 · field_convention and similar convention memories are never fed to the V2 slot detection that needs them most: target/split/key columns are guessed from scratch every time

**Impact:** High · **Effort:** M · **Verification:** ✅ Confirmed adversarially

**Problem.** When a user repeatedly opens JOIN/FEATURE/MODELING tasks over the same batch of data, convention knowledge such as "target field is bad_flag, split field is split, time field is apply_month" is heuristically re-detected every single time; when detection is wrong, the user has to correct it at the gate again — and that correction is not reused next time either. This is the scenario where the memory system could most easily cash in on "higher efficiency", yet it is entirely unwired.

**Evidence.** marvis/agent/modeling_setup.py:180 detects the target column / split column / candidate features purely from the current schema via detect_setup(backend, path, configured_target=...); grep for 'memory'/'field_convention' across the three setup modules marvis/agent/modeling_setup.py, join_setup.py, feature_setup.py yields 0 hits — while on the memory side, field_convention capture (pipeline.py:1784), the allowlist (policy.py:43-51 target_col/score_col/split_col/time_col/channel_col), and distillation merging (distillation.py:181-190, "字段常见取值" — "common field values") are all fully in place.

**Why it matters.** It directly determines the first-shot hit rate at the start of a V2 conversation: with correct detection the user says one "确认" ("confirm") and moves on; when wrong, it takes two or three rounds of back-and-forth. For a single-machine, single-user product, data conventions within the same workspace are highly stable, so the prior value of historical conventions is very high; moreover, this injection only affects the ordering of "default suggestions" — the user/gate still enforces confirmation, naturally preserving INV-3/INV-4.

**How to fix.** Add a read-only memory tie-breaker at detect_setup's candidate-column ordering:

1. Before setup, call retrieve_with_distillations(store, {'keywords':[dataset table name]}, limit=3) restricted to category=field_convention, and take the historical column-name set from the distilled structured['fields'].
2. Boost detected candidates whose names match historical values to the top of the ordering (deterministic string matching, not LLM).
3. Annotate the gate message with "目标列 bad_flag（与历史任务口径一致，来自记忆）" ("target column bad_flag — consistent with historical task convention, sourced from memory") and write memory_references through the existing use audit. The detection logic itself and the final confirmation gate remain unchanged.

**Verification note.** Confirmed, with two minor corrections that do not affect the conclusion: the "字段常见取值" wording actually lives in _template_summary at distillation.py:277-283 (lines 181-190 are the structured merge, without that string), and the capture payload at pipeline.py:1854 only includes target_col/score_col/split_col/time_col — not the allowlist's channel_col. The verifier also found the problem slightly worse than described: capture happens only in the V1 validation pipeline, so V2 gate corrections never even enter memory, and the entire V2 driver chain (driver_turn.py / orchestrator.py / plan_driver.py / auto_drive.py / turn_handlers.py / instruction_router.py / prompts.py / gates/* / gate_adapters.py / gate_payloads.py) greps 0 hits for memory.

### MEM-5 · Retrieval scoring has no time dimension at all: no recency decay, no expiry, no age annotation — credit-metric anchors get injected carrying drift

**Impact:** Medium · **Effort:** S · **Verification:** — Not independently verified

**Problem.** Credit populations and channel mix drift month over month; "A-card mob3 KS 0.32" may be entirely incomparable one year later. Today, a model_experience entry from 2025 competes at the same weight as one from last week, and once injected into the prompt the model has no way to know its freshness — it can easily read a stale anchor as a current baseline for comparison.

**Evidence.** In marvis/agent_memory/retrieval.py:241-277, _score_record's bonus terms are only model/scope/channel/month/keyword matches, with no created_at/updated_at involved; _context_packet (retrieval.py:315-332) also carries no time field, so the LLM cannot tell whether an experience is 3 days or 14 months old; the schema (store.py:859-979) has no TTL/archival mechanism. The only implicit temporal ordering is list_entries' ORDER BY updated_at DESC truncation.

**Why it matters.** Memory's core use is "historical comparison anchors"; anchors without timestamps feed the weak model a prior it cannot discount. This is also a precondition for all "historical KS range" proposals (MEM-1) to land credibly.

**How to fix.** Three deterministic small changes:

1. Add a recency term to _score_record (e.g. ≤90 days +10, ≤365 days +0, >365 days -10, using integers to keep the ordering deterministic).
2. Add 'age_days' and 'observed_at' (taken from entry.created_at) to _context_packet, and pass them through prompting._memory_packet, so each experience in the prompt carries its own "N days ago".
3. Attach a months coverage range (min/max month) to _merge_model_experience's metric_ranges, so distilled anchors carry their own time span. Do not do automatic deletion (preserving audit) — down-weighting plus annotation only.

### MEM-6 · Raw-memory recall ceiling: only the most recent 200 active entries are scanned, with no type targeting — as the store grows, early high-value experience permanently falls out of the recall window

**Impact:** Medium · **Effort:** S · **Verification:** — Not independently verified

**Problem.** However good the scoring function is, it can only score within "the most recent 200". Cross-task experience reuse is exactly about digging up "that same-scope model from half a year ago" — and those are precisely the entries squeezed out of the window first. Distillation partially compensates (full scan of 2000), but distillation preserves only ranges, not single-experiment details, and model_experience distillation groups by exact string concatenation of model+scope+channel (distillation.py:139-146) — the slightest spelling variance breaks grouping.

**Evidence.** marvis/agent_memory/retrieval.py:121 `raw_results = retrieve_relevant_memories(store.list_entries(limit=200), query, limit=remaining)`; list_entries (store.py:218-254) truncates by updated_at DESC, and this call passes no memory_type. Auto-capture writes 2 entries per successful task (model_experience+field_convention, pipeline.py:1780-1787) and 1-2 per failed task, plus user preferences — after roughly a hundred-odd tasks, the early model_experience entries fall out of the window.

**Why it matters.** This is a recall gap that silently degrades with usage volume: the longer the product is used (exactly when memory should be most valuable), the less retrievable raw experience becomes; the user's perception is "it clearly used to know this".

**How to fix.** Push the coarse filter down to SQL and target by type as needed:

1. Split retrieve_with_distillations into two pulls — memory_type='model_experience' alone with limit=200, plus all remaining types with limit=200 — guaranteeing model experience is not crowded out by preference/convention entries.
2. When query.model_name/scope is present, add a SQL prefilter channel of payload_json LIKE '%<normalized model_name>%' (keeping the existing Python scoring for fine ranking).
3. list_entries needs a (memory_type, updated_at) index — it already exists (store.py:879-883), which is sufficient support.

### MEM-7 · Memory quality has no negative-feedback loop whatsoever: user corrections / task failures never write back to memory standing; wrong experience can only be manually disabled entry by entry

**Impact:** Medium · **Effort:** M · **Verification:** — Not independently verified

**Problem.** Once a wrong field_convention (say, a leakage column recorded as a score column) enters the store, it will be retrieved, injected, and keep its standing on every subsequent relevant turn — unless the user proactively opens the memory panel, finds it, and manually disables it. The system is entirely unaware of "did this memory actually help or mislead" — the "memory quality self-assessment" this lens requires currently has zero implementation.

**Evidence.** marvis/agent_memory/store.py:28-36 AUDIT_EVENT_TYPES contains only create/retrieve/use/disable/enable/delete/reject — no event of any kind meaning "challenged / corrected / ineffective" exists; record_use (store.py:622-643) records usage only, never outcomes; retrieval scoring (retrieval.py:241-300) and distillation confidence (distillation.py:81-87) consume neither usage nor challenge history. The message-level memory_references endpoint already exists (routers/agent_memory.py:186-204) but is display-only.

**Why it matters.** A weak model plus a wrong prior is worse than no prior: contaminated memory will systematically lead the agent astray, and the user will struggle to trace it to "the memory's fault". The absence of quality signals also raises the risk of the anchor injection proposed in MEM-1.

**How to fix.** A low-cost loop in three steps:

1. Add 'challenged' to the event model: when a turn's assistant message carries memory_references and the user's next message hits an existing negation/correction lexicon (plan_driver._NEGATED_CONFIRM and the extractors' "纠正一下" ("let me correct that") marker are reusable), append challenged events for those memory ids (a pure audit write, no state change, preserving INV-4/INV-8).
2. Retrieval scoring penalty: entries with challenged_count>=2 get score -20, or directly downgrade the displayed confidence.
3. Memory panel shows use/challenged counts per entry with one-click disable — the data is all in the events table; only the aggregation query is missing.

### MEM-8 · Low-confidence noise goes straight into the prompt: raw memories are injected whenever score>0, while the distillation side filters out low — asymmetric, and distracting for a weak model

**Impact:** Medium · **Effort:** S · **Verification:** — Not independently verified

**Problem.** Up to 6 memories enter the prompt per V1.1 turn, potentially including "some unrelated task experience fished out because your message happened to mention KS". For a 32-72B model, weakly-relevant context is a genuine distraction source and a token waste; worse, low-confidence raw entries get more lenient treatment than low-confidence distillations — a self-contradictory policy.

**Evidence.** marvis/agent_memory/retrieval.py:83-84 `if score <= 0: continue` is the only threshold; a single generic keyword hit (the user's message contains 'KS' → _score_general_record +15, retrieval.py:290-293) is enough to get packed and injected with confidence='low' (_score_confidence<25, retrieval.py:303-308). By contrast, the distillation side at retrieval.py:112-113 explicitly does `if distillation.confidence == 'low': continue`. agent_memory_context_from_store (api_support.py:154-180) applies no further confidence filtering to raw packets.

**Why it matters.** Every memory in the prompt consumes the weak model's limited attention budget; insufficient recall precision can make "with memory" perform worse than "without memory", undermining the perceived value of the entire memory feature.

**How to fix.** One line of policy alignment plus one budget rule:

1. Filter retrieve_relevant_memories results on `_score_confidence(score) == 'low'` (i.e. drop score<25), aligning with the distillation-side policy.
2. In agent_memory_context_from_store, give raw and distillations separate quotas (e.g. distillations ≤3, raw ≤3) so high-scoring distillations cannot squeeze precise single-task experience entirely out of limit=6.
3. For keyword-hit entries scoring below 25, write only the retrieve audit without injecting — so there is data available when tuning the threshold later.

### MEM-9 · User-preference capture entry is too narrow and the 'skill/runtime' substring is a one-vote veto: the user believes it was remembered, but it was silently dropped

**Impact:** Low · **Effort:** S · **Verification:** — Not independently verified

**Problem.** "好的，请记住：以后报告用英文" ("OK, please remember: reports in English from now on" — marker mid-sentence) → not captured; "请记住：训练时用 lightgbm 的 runtime 参数 n_jobs=4" ("please remember: use lightgbm's runtime parameter n_jobs=4 when training") → dropped because it contains 'runtime'. In both cases the user receives no feedback whatsoever, and when they later discover the agent "didn't remember", trust is damaged — and this is the only entry point of the memory system that the user explicitly initiates.

**Evidence.** marvis/agent_memory/extractors.py:227-232 _explicit_preference recognizes only six text.startswith markers ("请记住：" ("please remember:") etc.); if the marker appears mid-sentence it returns ''; extractors.py:241-243 _mentions_reserved_skill_runtime drops the entire message with no receipt whenever the full text contains the substring 'skill' or 'runtime'. capture_user_preference_memory (api_support.py:17-53) silently returns when candidate is None.

**Why it matters.** The explicit "请记住" ("please remember") is the user's direct mental contract with the memory system; silently breaking that contract hurts the experience more than not having the feature at all. The fix cost is minimal.

**How to fix.**

1. Change _explicit_preference to re.search: locate the marker and take the text after it (keeping the explicit-marker threshold — no whole-message LLM judgment, staying conservative).
2. Change the reserved-word check to word-boundary matching, and reject only when the preference's semantics genuinely target the skill runtime (e.g. it contains both '技能'/'skill' and '执行/运行' ("execute/run")); on rejection, reply through the existing add_agent_message with a sentence like "该偏好涉及保留能力，本次未记录" ("this preference touches a reserved capability and was not recorded this time").
3. On successful capture, attach memory_references in the next assistant message's metadata so the frontend can show an "已记住" ("remembered") badge (the reference pipeline already exists).

### MEM-10 · The consolidation trigger memory.after_save never fires, and the distillation chain swallows errors silently end to end: distillation health is unobservable

**Impact:** Low · **Effort:** S · **Verification:** — Not independently verified

**Problem.** For users who only use V2 workflows, their user_preference/field_convention memories may remain raw entries forever and never produce distillations (no event ever triggers, unless they manually click consolidate); and any distillation bug (e.g. an anomalous group payload) manifests as "nothing happened", impossible to troubleshoot.

**Evidence.** marvis/agent_memory/consolidation.py:10 CONSOLIDATION_TRIGGERS contains 'memory.after_save', but the only other occurrence repo-wide is the lexicon declaration at marvis/plugins/manifest.py:53 — there is no dispatch/emit call site anywhere; consolidation of preference-type memories can therefore only piggyback on V1.1's validation.completed / report.after_generate (neither fires in V2 workflows). DistillationEngine.distill_category does `except Exception: continue` per group (distillation.py:116-119), ConsolidationScheduler.consolidate_all records whole-category exceptions as 0 (consolidation.py:47-50), and the _safe_call thread backstop swallows likewise (consolidation.py:94-98) — all three layers with no logging.

**Why it matters.** V2 is the product mainline, yet memory consolidation hangs only off V1.1's hooks; as V1.1's usage share declines, distillation will quietly "grind to a halt" with nobody noticing. The silent error swallowing turns this subsystem into an observability blind spot (consistent with the platform-wide remediation direction that "silent degradation must be observable").

**How to fix.**

1. After a successful AgentMemoryStore.create (active status), have the caller or the store dispatch 'memory.after_save' via app.state.hook_dispatcher (the scheduler has a built-in 300s throttle, so frequency is not a concern); if the store should not depend on the dispatcher, add one dispatch at each of the two capture points — api_support.capture_user_preference_memory and the pipeline.
2. When a V2 plan reaches DONE (same site as the MEM-1 capture point), dispatch an event equivalent to validation.completed, or call scheduler.on_event directly.
3. Change distill_category's except to logger.warning('distill group failed: %s', scope_key, exc_info=True), and have consolidate_all return {category: {count, errors}}, propagated through to the /agent-memory/consolidate response.

### MEM-11 · Distilled structured payload enters the prompt in full: model_experience distillations carry an unbounded source_task_ids list, tokens grow linearly with support count, and it contradicts the distillation discipline of "do not output task IDs"

**Impact:** Low · **Effort:** S · **Verification:** — Not independently verified

**Problem.** A model_experience distillation that has accumulated dozens of tasks will, when injected into the prompt, carry dozens of uuid task IDs plus the full source_memory_ids list — pure noise tokens for a weak model, and in conflict with the distillation system prompt's own discipline. The summary has a 400-character truncation (prompting.py:14), but the payload has no cap of any kind — the only unbudgeted opening on the memory-to-prompt path.

**Evidence.** marvis/agent_memory/prompting.py:106-113 _memory_packet, for kind=='distillation' packets, does `packet['payload'] = payload` directly (no trimming) and additionally carries the full source_memory_ids list (prompting.py:104-105); _merge_model_experience (distillation.py:254-261) puts the complete sorted source_task_ids set into structured. By contrast, DISTILL_SYS (distillation.py:16-19) explicitly requires "不要输出任务 ID" ("do not output task IDs"), and the raw-memory side has _bounded_payload trimming by allowlist (prompting.py:116-121).

**Why it matters.** Local 32-72B models have tight context windows; with up to 6 memories per turn, an unbounded distillation payload will steadily crowd out space for evidence/conversation memory; the ID noise may also induce a weak model to parrot meaningless uuids in its replies.

**How to fix.**

1. Trim for display in _memory_packet's distillation branch: keep only metric_ranges/scopes/channels/support in the payload (drop source_task_ids, or replace it with source_task_count=len(...)).
2. Replace source_memory_ids with a count likewise (the audit reference memory_references keeps the full list for traceability; prompting.memory_references is built separately and is unaffected).
3. Add a total character budget for the entire cross_task_memory section (e.g. 3000 characters, truncating by descending confidence when exceeded) as a final safeguard.

---

## 3. Credit-Risk Domain Depth

**Lens verdict.** Credit-domain depth is markedly better overall than the 6-28 baseline: scorecard points conversion is strictly consistent with the scorer, monotonic binning is the default, feature selection runs in WOE space with coefficient-sign warnings, and the governance artifacts — calibration, model cards, monitoring policies, challenger packages, approval packages — are all genuinely implemented. However, one "claimed fixed but not actually fixed" issue was found: `tune_hyperparameters` still silently tunes on NaN labels (empirically demonstrated that LightGBM silently trains on NaN labels), and the mandatory NaN confirmation gate is entirely missing from the tuning path. There are two cross-cutting methodology gaps: the platform's score-direction convention is self-contradictory (`tradeoff_view` assumes higher score = better customer, `reject_inference` assumes higher score = worse customer, while the model's native score is a PD) — under weak-model agent orchestration one side will inevitably be wired backwards; and probability calibration self-evaluates Brier/ECE on its own fitting set (default `test`, which is also the early-stopping and tuning-selection set), so isotonic regression yields systematically optimistic calibration metrics. The biggest capability gap is the "last mile": the monitoring policy is paper JSON only — the platform has no scoring tool that applies a trained model to new data and no monitoring-execution tool. On the reporting side, the score-band table bins each split independently and lacks cumulative columns, so it cannot support cutoff decisions; the scenario-declared `eval_metric` (`response_lift` etc.) was never wired up; and feature temporal stability (PSI) is absent from screening and reporting. The strategy pack's business semantics (roll-rate blindness to gap months, swap empty sets displayed as 0%, silent fallback at infeasible operating points) remain demo-grade.

### DOM-1 · Regression / incomplete fix: tune_hyperparameters still silently tunes on NaN labels — the NaN confirmation gate is entirely missing from the tuning path

**Impact:** High · **Effort:** S · **Verification:** ✅ Confirmed adversarially

**Problem.** The 6-28 fix-status update claimed "H1/H2: the LightGBM default recipes and the tuning path are wired into the shared NaN-label confirmation gate", but only the training recipes (lgb.py:28-30 and the other 9 recipes) were actually wired in; not a single line of the tuning path was changed. A dataset containing NaN labels goes straight into `lgb.Dataset` for training (LightGBM silently treats the NaN as valid labels), while each trial's train_ks/test_ks passes through `feature_ks`'s isfinite filter and is computed only on the labeled subset — the training sample and the parameter-selection metric use different subsets. `best_params` is selected from contaminated trials, with no confirmation gate and no `nan_labels_dropped` reporting anywhere along the way.

**Evidence.** marvis/packs/modeling/tune.py:100-111 `ytr = train[target_col].to_numpy(dtype=float)` → `dtrain = lgb.Dataset(train[feats], label=ytr, ...)`; the entire file contains no `resolve_modeling_splits`/`require_labels_confirmed` (imports are only metrics/errors, tune.py:25-26). marvis/packs/modeling/tools.py:589-616 `tool_tune_hyperparameters` neither receives nor forwards `drop_nan_labels`. In marvis/packs/modeling/manifest.json the `tune_hyperparameters` input_schema has no `drop_nan_labels` property (train_model/train_models/select_features all have it). Empirically verified in the py_313 environment: `lgb.Dataset` + `lgb.train` on a binary target containing NaN labels trains silently and successfully.

**Why it matters.** This violates the locked-in mandatory NaN-label confirmation gate spec (full coverage of V2 packs, typed-error + drop flag). `best_params` is consumed directly by `tool_train_models` to train the final lgb model: the final training run is blocked by the gate or drops the NaNs, but the parameters were already selected on the wrong sample and the wrong metrics — a hidden quality defect of "looks like it passed the gate, yet the parameters are contaminated", hitting exactly the common credit scenario of a sample table whose performance window has not matured (labels naturally NaN).

**How to fix.**
1. Add a `drop_nan_labels` (boolean) property to `tune_hyperparameters` in manifest.json.
2. In tune.py, after `_split`, copy the training-recipe pattern verbatim: `train, test, oot, oot_has_labels, audit = resolve_modeling_splits(train, test, oot, target_col=target_col, drop_nan_labels=drop_nan_labels)`; when unconfirmed, raise `NanLabelNotConfirmedError` (error_kind=`nan_label_not_confirmed`, going through the existing gate flow); when `oot_has_labels=False`, set the trial's oot_ks/oot_auc to None.
3. Add `nan_labels_dropped` to the TuneResult / tool output.
4. Copy `test_train_model_gates_nan_train_label` into a tune-flavored regression test (first assert blocked, then assert success after drop with count = 1).

**Verification note.** The adversarial pass reproduced the failure empirically: in the py_313 environment, `lgb.Dataset` + `lgb.train` on a binary target with 15% NaN labels trained silently, and `get_label` after Dataset construction showed the NaNs had been quietly converted to 0.0. The verifier also confirmed there is no upstream backstop — `prepare_modeling_frame` does no NaN-label filtering, and the readiness checks at tools.py:3818 (`target.dropna().isin([0,1]).all()`) and tools.py:3957 (all-NaN only) both let partially-NaN labels through.

### DOM-2 · Self-contradictory score-direction conventions within the platform: tradeoff_view assumes higher score = better customer, reject_inference assumes higher score = worse customer, and neither has a direction parameter

**Impact:** High · **Effort:** M · **Verification:** ✅ Confirmed adversarially

**Problem.** Two tools on the same platform adopt opposite implicit conventions for "score direction", and neither declares it in its schema. When a user/agent feeds a model PD score into `tradeoff_view`, `scores>=cutoff` approves the highest-risk customers, and the resulting approval_rate/bad_rate/recommended operating point are all direction-inverted while the numbers look entirely plausible. When scorecard points are fed into `reject_inference`, parceling infers the highest-quality rejects as bad customers, directly poisoning the retraining sample.

**Evidence.** marvis/packs/strategy/tradeoff.py:38 `for approved in [scores >= cutoff]` (above cutoff means approved = higher is better); marvis/packs/modeling/reject_inference.py:203-204 `_risk_order` uses `np.argsort(-safe_scores)` and `_parcel_rejected` (L164-167) marks the top bad_count highest-scored rows as bad=1 (higher is worse). Meanwhile the platform's native model score is a PD (tools.py:4139/4141 — the scorer returns `predict_proba[:,1]`, higher is worse), while scorecard points are higher-is-better (scorecard.py:70-72). The strategy manifest's `tradeoff_view` schema (dataset_id/score_col/.../max_bad_rate) and the `reject_inference` schema have no direction parameter of any kind; the STRATEGY_ANALYSIS template (around orchestrator/templates/sample.py:1039) only passes through the score_col slot.

**Why it matters.** These are the core outputs a chief risk officer uses for cutoff decisions and reject-inference retraining — a wired-backwards direction is not a crash but "silently issuing inverted business recommendations". Under weak-model (32-72B) orchestration one side will almost inevitably be wired backwards, and downstream `recommend_operating_point` will package the wrong point as the "recommended operating point".

**How to fix.**
1. Add a required `score_direction: enum('higher_is_riskier','higher_is_better')` to both tools' schemas; `tradeoff_view` decides `approved = scores<cutoff` or `>=cutoff` by direction, and `reject_inference`'s `_risk_order` sorts by direction.
2. Add a deterministic direction self-check (INV-1-safe): compute corr(score, target) on labeled rows; when it contradicts the declared direction and |corr| exceeds a threshold, raise a typed error requiring user confirmation (reusing the NaN-gate pattern), with the corr value in the diagnostics.
3. In the render layer (renderers.py `_render_tradeoff_view`), explicitly label "score direction: higher = higher/lower risk".
4. Add two regression test groups covering reversed-direction wiring.

**Verification note.** The verifier additionally found that both input schemas declare `additionalProperties: false` (marvis/packs/strategy/manifest.json:217-233, marvis/packs/modeling/manifest.json:120-162), so a direction parameter cannot even be smuggled through today, and that the existing tests actively lock in the two opposite conventions — tests/test_strategy_tradeoff.py:23-45 uses credit-score-style values (500-760, higher = better) while tests/test_modeling_reject_inference.py:12-34 asserts the highest-scored rejects are labeled bad. `score_direction` has zero hits in marvis source code.

### DOM-3 · Missing "last mile": no tool of any kind applies a trained model to new data, and the monitoring policy is paper JSON with no execution path

**Impact:** High · **Effort:** M · **Verification:** ✅ Confirmed adversarially

**Problem.** The MODELING flow terminates in an artifact + reports + a monitoring-policy JSON (thresholds/cadence/owner), but the platform has no tool that can: a) apply the artifact to a newly registered dataset (e.g. next month's applications) to produce a score column; b) execute checks on new data according to the generated monitoring_policy (score PSI vs. the training baseline, feature PSI, KS once labels mature). The warn/fail thresholds in the monitoring policy are only ever "checked" once, against the baseline metrics from training day.

**Evidence.** None of the 20 tools in marvis/packs/modeling/manifest.json (check_data_quality … generate_model_reports) is a score_dataset/apply_model/monitor-type tool. `_ModelArtifactScorer` (tools.py:4110-4160) is used only on internal report paths. The strategy tools (tradeoff_view/backtest_strategy) require the dataset to already carry a score column (strategy/tools.py:169-183). `_monitoring_policy_payload` (tools.py:1678-1716) and `_monitoring_check_payload` (tools.py:1747-1764) only compare the training-time baseline_metrics against the thresholds, producing a one-off snapshot document.

**Why it matters.** This is the gap between "priceable / deployable / approvable" and "operable after deployment". For a single-machine product, the core repeat-use scenario is precisely monthly monitoring — the user comes back each month with new data, and today the only options are rerunning the entire MODELING pipeline or scoring manually outside the platform; the champion/challenger and monitoring-cadence promises all fall through. The strategy pack also struggles to consume the platform's own models for lack of a score column.

**How to fix.** Two steps:
1. Add a new modeling tool `score_dataset(artifact_id, dataset_id, output_col='model_score', use_calibration=true)` — reuse `_ModelArtifactScorer` plus the ArtifactUnitOfWork/register_existing_with_audit pattern to land a derived dataset (audit kind `modeling.dataset.scored`); for scorecards, also land a points column.
2. At training time, persist the train score distribution (equal_frequency boundaries + per-bin proportions) into model_meta as a deterministic baseline; add `run_monitoring_check(artifact_id, dataset_id, monitoring_policy?)`: compute score PSI vs. the training baseline, per-feature PSI for every in-model feature, KS/AUC when labels are present, apply the monitoring_policy's warn/fail thresholds to output pass/warn/fail plus recommendations, and write a `modeling.monitoring.run` audit entry. All computation deterministic, respecting INV-1/INV-8.

**Verification note.** The verifier swept all 7 pack manifests plus app.py/api.py and found no scoring or apply endpoint anywhere on the platform; `_ModelArtifactScorer` has exactly 3 call sites (tools.py:1345, 4010, 4030), all internal to calibration or report generation, and the feature pack's `compute_psi` only compares two filters within a single dataset_id, so it cannot serve as a cross-dataset monitoring backstop.

### DOM-4 · Probability calibration self-evaluated on its own fitting set: fitted on test by default (a set already used for early stopping + tuning selection), so Brier/ECE are entirely in-sample

**Impact:** Medium · **Effort:** S · **Verification:** — Not independently verified

**Problem.** The test set is triple-reused (early stopping, tuning selection, calibration fitting), and the reported brier_calibrated/ece_calibrated/reliability curve are self-evaluations on the calibrator's own fitting set. Isotonic regression achieving ECE near 0 on its fitting set is a mathematical certainty, so the "probability calibration" report sheet presents systematically optimistic calibration quality; a user relying on it for pricing or provisioning will underestimate calibration error.

**Evidence.** marvis/packs/modeling/tools.py:1335 `split_name = str(inputs.get("split") or "test")`; tools.py:1362-1365 `calibrator = _fit_calibrator(method, raw_scores, labels)` is immediately followed by `calibrated_metrics = _calibration_metrics(labels, calibrated_scores, n_bins=n_bins)` — the evaluation uses the very same raw_scores/labels the calibrator was fitted on. Meanwhile the lgb recipe uses test as the early-stopping eval_set (recipes/lgb.py:57) and tune.py selects parameters by test KS (tune.py:92-96).

**Why it matters.** The entire business value of calibration is that PD becomes usable for pricing/provisioning/IFRS 9 — distorted calibration-quality metrics are more dangerous than no calibration at all, because they confer false confidence. This is the mirror image of the very problem the calibration feature was built to solve: "uncalibrated probabilities cannot be used for pricing".

**How to fix.**
1. Keep fitting the calibrator on test (acceptable), but report metrics evaluated on an independent labeled set: when OOT has labels, compute calibrated/raw Brier/ECE/reliability curves on OOT (OOT is evaluation-only, not part of fitting, so this does not violate the no-peeking principle).
2. When OOT is unlabeled, keep the in-sample metrics but explicitly annotate the output and the report sheet with `evaluated_on: 'fit_sample(in-sample)'` plus a warning sentence.
3. Add `fit_split`/`eval_split` fields to the calibration payload; surface both in the model card and the approval package.
4. Add a test: a constructed sample where isotonic has ECE ≈ 0 on the fitting set but clearly larger ECE on OOT, asserting the report takes the OOT value.

### DOM-5 · Score-band table binned per split with no cumulative columns — cannot support cutoff decisions or cross-split comparison

**Impact:** Medium · **Effort:** M · **Verification:** — Not independently verified

**Problem.** The band boundaries of the train/test/oot splits all differ, so the same row bin=3 represents different score ranges in different splits. The table cannot answer "if the cutoff is set at score X, what are OOT's approval rate and bad rate, and how much have they shifted relative to train?" — the core question a score-band table exists to answer. There are also no cumulative columns from which to read off the approval-rate/bad-rate combination at any cut point.

**Evidence.** marvis/packs/modeling/tools.py:3914-3922 `for split_name, split_value in config.split_values.items(): ... edges = equal_frequency_bin_edges(finite_scores, int(bin_count))` — the boundaries are computed independently inside each split's loop iteration. The row fields (tools.py:3932-3942) are only bin/score_lower/score_upper/sample_count/bad_rate/avg_score — no cumulative approval rate, cumulative bad rate, per-band KS, or lift. marvis/output/model_report.py:66 `_write_section_sheet(workbook, "评分分段", payload.score_bands, None)` ("评分分段" = score bands) lands it in Excel as-is.

**Why it matters.** Score bands plus cutoff-decision support is the page a risk committee flips to most often in a model report; the current table only shows within-split rank ordering (KS/AUC already covers that) and offers zero help for setting a cutoff or observing score migration — "a sheet with no decision value".

**How to fix.**
1. Compute the boundaries once on the train split only (for scorecards, use fixed-width points bands, e.g. 20 points per band); test/oot reuse the same edges — migration becomes visible at a glance.
2. Add per-row cum_count_pct (cumulative approval rate from the high-score or low-score direction), cum_bad_rate, cum_bad_capture (cumulative share of bads captured), and lift (= bin bad_rate / overall bad_rate).
3. Determine the cumulation direction from DOM-2's `score_direction` and annotate it in the table header.
4. Add a "worked reading" row to the report sheet: "example: cutoff=xxx → approval rate yy%, bad rate zz%".

### DOM-6 · Scenario-declared eval_metric is dead metadata: marketing/anti-fraud scenarios still pick champions by KS, and the platform computes no lift/PR-type metrics at all

**Impact:** Medium · **Effort:** M · **Verification:** — Not independently verified

**Problem.** The evaluation criteria promised by the scenario templates (response-type models judged by lift, anti-fraud by recall/precision) never enter metric computation, experiment comparison, or champion selection. Train 3 recipes under the marketing scenario and the platform still picks the "best" by OOT KS, and the report still only shows KS/AUC/PSI — in direct contradiction with the scenario's own notes about "not conflating with KS".

**Evidence.** marvis/packs/modeling/scenarios.py:76-77 declares eval_metric='response_lift' for marketing, L88 the transaction scenario notes "focus on recall/precision rather than KS alone"; `apply_scenario` writes config.eval_metric (scenarios.py:161, contracts.py:29). But binary champion selection looks only at oot_ks/test_ks: tools.py:789-797 `_pick_best_experiment` and tools.py:929 `_pick_best_comparison_row` both hardcode `('oot_ks','test_ks')`; `compute_model_metrics` (recipes/common.py:157-213) produces no lift/recall/precision/PR-AUC fields (head_tail_lift exists only in tune trials, tune.py:137). Across the entire repo, the sole consumer of eval_metric is the persistence call at repositories/modeling.py:408.

**Why it matters.** A marketing response model with high KS does not necessarily have high head-of-distribution conversion lift (top-decile lift is what determines campaign ROI); in highly imbalanced anti-fraud scenarios KS is insensitive to head precision. The scenario system is the product's professionalism selling point — declared behavior diverging from actual behavior will make a domain-savvy user lose trust immediately.

**How to fix.**
1. Add deterministic fields to ModelMetrics: lift_head_5/lift_head_10 for test/oot (reusing feature/metrics.py `head_tail_lift`), precision@top10pct, recall@top10pct (for transaction); produce them inside `compute_model_metrics` (no extra scan cost).
2. Have `_pick_best_experiment`/`_pick_best_comparison_row` accept eval_metric: response_lift → select by `('oot_lift_head_10','test_lift_head_10')`; ks_auc keeps current behavior.
3. Have compare_experiments rendering put the scenario's primary metric column first.
4. Write the scenario eval_metric and the actual selection metric into the model card and the selection reason text.

### DOM-7 · Feature temporal stability (PSI/CSI) absent from feature screening, selection, and the model report's univariate sheet

**Impact:** Medium · **Effort:** S · **Verification:** — Not independently verified

**Problem.** A feature with high dev-set KS but violent train→OOT distribution drift (PSI>0.25) sails all the way through screen→select→into the model, and no table in the report can expose "which in-model feature is temporally unstable". The platform already has a ready-made `feature_psi` implementation and a manual `tool_compute_psi` — they are simply not wired into the main flow.

**Evidence.** marvis/feature/screen.py:120 — screen_features' scores are only {ks, missing_rate, unique_count} (iv appended later, L146), with no PSI and no stability-based elimination rule. marvis/packs/modeling/select.py:109/159-165 — the raw/woe-space scores are only iv/ks/vif, and the selection criteria are only iv_min/corr_max/vif_max/top_k. The report's `_univariate_rows` (tools.py:3946-3976) row fields are iv/ks/auc/coverage/missing_rate/unique_count, and the `feature_metrics` call passes no `compare_values`, so its psi branch (feature/metrics.py:217) is always None and never output.

**Why it matters.** Feature-level PSI screening is a standard step in credit modeling (typically PSI>0.1 warn, >0.25 eliminate) and the most common root cause of OOT KS decay. A model report lacking an "in-model feature stability" table is an obvious shortfall from a regulatory/internal-audit perspective, and it directly affects the user's long-term goal of "never failing a KS benchmark".

**How to fix.**
1. Add per-feature dev-vs-holdout PSI to screen_features (holdout_values already exists; reuse `feature_psi` with equal_frequency edges from the dev set), add a `psi` field to scores, and add an `unstable` soft flag (psi >= psi_warn, default 0.25, going onto the confirmation list rather than auto-deleting — same pattern as leakage).
2. Add an optional `psi_max` elimination criterion to select_features (dropped reason worded 'high PSI x.xx train→oot').
3. In `_univariate_rows`, pass `compare_values` per feature with train values as base and OOT values as compare to populate the psi column; add a "PSI(train→OOT)" column to the report's univariate sheet.
4. Do not hardcode business threshold values — everything via parameter defaults with override.

### DOM-8 · Roll-rate matrix blind to missing observation months: cross-month gaps are treated as single-period transitions, and only count-based (no balance-based) rates are supported

**Impact:** Medium · **Effort:** M · **Verification:** — Not independently verified

**Problem.** Monthly credit snapshots routinely have missing months (feed outages, definition switches, frozen accounts): a customer at 202401=M1 and 202403=M3 is recorded as a one-month "M1→M3" transition, compressing two periods of roll into one and systematically overestimating deterioration speed. Meanwhile the collections/provisioning convention for roll rates is balance-weighted (a large account's migration matters far more than a small one's) — the current implementation is count-only and the output does not declare its basis.

**Evidence.** marvis/packs/strategy/roll_rate.py:47-54 `_adjacent_pairs` sorts by id then pairs adjacent observations directly via `zip(statuses[:-1], statuses[1:])` without validating the time gap between the two observations; roll_rate.py:32-37 — the returned RollRateMatrix hardcodes `period="month"`; the matrix and base_counts are all row-count based (`_points_to_matrix`, L77-90) with no balance/EAD-weighting option, even though `compute_vintage_curve` in the same repo already supports a balance denominator (validation/vintage.py:40-44).

**Why it matters.** Roll rate is an input to post-lending strategy and the migration-rate method for provisioning; a wrong period basis directly inflates or distorts key transitions like M1→M2. For a product whose stated scenario is "post-lending (early warning / pre-collections)" (scenarios.py:57-67, notes explicitly say "pair with roll_rate"), this is a core credibility issue.

**How to fix.**
1. Have `_adjacent_pairs` parse month granularity and compute the gap: only pairs with gap==1 period enter the matrix; count gap>1 pairs and add `skipped_gap_pairs`/`gap_ratio` diagnostic fields to the output; when gap_ratio exceeds a threshold (e.g. 5%), add a warning message to the tool output; optionally allow an `allow_gap_periods` parameter to relax.
2. Add an optional `balance_col`: weight the matrix by balance (rate = Σbalance(from→to)/Σbalance(from)); output both count and balance versions, or clearly state the basis.
3. Infer RollRateMatrix.period from the data (month/week) instead of hardcoding it.

### DOM-9 · Multi-recipe champion selected by OOT KS, contradicting tune's explicit "OOT is report-only, not for selection" policy

**Impact:** Medium · **Effort:** S · **Verification:** — Not independently verified

**Problem.** The same platform gives two opposite answers to "may OOT be used for selection": tuning across 40 trials strictly guards against peeking at OOT, then turns around and picks the champion among 4-8 recipes by OOT KS. The selected model's oot_ks carries an upward winner's-curse bias, the OOT metrics in the report/model card are no longer clean out-of-sample estimates, and the protection at the tuning layer is cancelled out.

**Evidence.** marvis/packs/modeling/tune.py:6-8 — the docstring explicitly states "OOT metrics are reported for transparency but are not used for hyperparameter selection" (test_modeling_pack.py:131 even asserts this sentence appears in the summary); yet tools.py:789-797 `_pick_best_experiment` and tools.py:929 `_pick_best_comparison_row` always prefer oot_ks for binary when selecting the champion across recipes/experiments (the tool_train_models docstring at L681-683 likewise self-describes "best by OOT KS").

**Why it matters.** With few candidates the bias is small, but this is a methodological consistency problem: once a reviewer (model validation / regulator) notices that the selection set equals the reporting set, the credibility of all OOT metrics is discounted. It also conflicts with the user's long-term goal of using OOT KS as the acceptance anchor in "never failing a KS benchmark".

**How to fix.**
- Option A (recommended, minimal change): switch the binary selection key to the same test-basis score as tune, `test_ks - 0.5*max(0, train_ks-test_ks)`, with OOT purely reported; change the selection_metric wording to 'test_ks(overfit-penalized)'.
- Option B (if the business insists on selecting by OOT): keep current behavior but explicitly annotate in selection_reason, the model card, and the approval package that "OOT participated in model selection; its metrics contain selection bias", and flag that column in the compare_experiments rendering.
- Either one suffices — the key is eliminating the double standard and making the basis visible.

### DOM-10 · Report score column silently falls back: when the artifact is missing, the first feature column impersonates the model score in a formal report

**Impact:** Medium · **Effort:** S · **Verification:** — Not independently verified

**Problem.** When experiment.artifact_id is None (historical failed experiments, legacy data), generate_model_report does not raise — it takes the first in-model feature as the "model score" and computes a complete, plausible-looking set of KS/binning/stress-test numbers into the formal Excel report. The numbers are all real; the semantics are all wrong.

**Evidence.** marvis/packs/modeling/tools.py:4163-4169 `_report_score_col`: when the dataset has no 'score' column and the artifact is None, it does `return artifact.feature_list[0]`/`config.features[0]`; `_report_scored_dataset` (tools.py:4026-4027) takes this fallback when artifact is None; downstream stress_low_pricing, `_report_bin_table`, and `_score_band_rows` all treat that column as the model score when computing KS/PSI/bins (tools.py:3040-3077), with no annotation anywhere in the report.

**Why it matters.** The model report is a formal deliverable for the risk committee/auditors — a silent semantic substitution is far more dangerous than a crash, and it conflicts with the platform's stance that "every statement is traceable" and "audits are complete" (INV-8).

**How to fix.**
1. When artifact is None and there is no 'score' column, either raise `ModelingError('experiment has no artifact and dataset has no score column; cannot generate score-based sections')`, or mark the score-dependent sections unavailable (reuse resolve_report_sections' missing-column semantics: section_status with reason='缺少模型 artifact' ("model artifact missing")), emitting only the sample/vintage sections that do not depend on scores.
2. If "report on an external score column" genuinely needs supporting, require an explicit score_col input rather than guessing the first feature.
3. Add a test: calling generate_model_report on an artifact-less experiment asserts an error or section-unavailable status.

### DOM-11 · Swap set / bad rate shows 0.0% for empty sets and unlabeled samples; backtest output lacks a label-coverage basis

**Impact:** Low · **Effort:** S · **Verification:** — Not independently verified

**Problem.** The real swap-in set (customers rejected by the old strategy but approved by the new one) mostly has no performance labels — after they are dropped, swap_in may be empty or tiny, and `swap_in_bad_rate=0.0` will be read as "the swap-in population has zero bad debt", the most dangerous possible misreading. The output also never tells the user what fraction of labeled samples the swap statistics are based on.

**Evidence.** marvis/packs/strategy/backtest.py:117-120 `_bad_rate` returns 0.0 on an empty Series (tradeoff.py:93-96 likewise); the swap_in/swap_out bad_rate is computed by it (backtest.py:68-73); tool_backtest_strategy first calls resolve_labeled_frame, dropping all unlabeled rows (strategy/tools.py:131-133), after which the swap statistics cover only the labeled subset, and BacktestResult has no labeled_coverage field.

**Why it matters.** The business purpose of swap-set analysis is exactly to assess "swap-in population risk" — the semantic difference between 0.0% and None is the difference between approving and rejecting a strategy loosening. This is also the interface point where reject inference (the reject_inference output from DOM-2) should link up with backtesting.

**How to fix.**
1. Have `_bad_rate` return None for empty sets; allow the corresponding BacktestResult/TradeoffPoint fields to be null; the render layer displays "no labeled samples".
2. Add `labeled_coverage = labeled_rows/total_rows` and `swap_in_labeled_count` fields to BacktestResult; nan_labels_dropped already exists — connect them in the message text to state "swap statistics are based on labeled samples only".
3. Docs/renderer hint: to evaluate the swap-in population, first run reject_inference to generate an inferred-label dataset, then backtest (closing the tool-chain loop).

### DOM-12 · Operating-point recommendation silently falls back to the lowest-bad-rate point when constraints are infeasible; fuzzy reject inference ignores per-record scores

**Impact:** Low · **Effort:** S · **Verification:** — Not independently verified

**Problem.** a) When the user sets max_bad_rate=2% and every cutoff point exceeds it, the tool still returns a "recommended" point (lowest bad rate, but still violating the constraint), with output structure identical to the feasible case — the agent/user has no way to know the constraint is unsatisfiable. b) The standard fuzzy-augmentation practice splits each reject into weighted good/bad rows by its KGB-score p(bad); the current implementation applies one global bad_rate weight to all rejects (weight*bad_rate / weight*(1-bad_rate)), flattening the risk differences between rejects into "one uniform bad-rate assumption for all rejects" — the method's name does not match its substance.

**Evidence.** marvis/packs/strategy/tradeoff.py:52-58 `recommend_operating_point`: when feasible is empty, `return min(points, key=lambda point: point.bad_rate)` with no infeasibility flag of any kind; marvis/packs/modeling/reject_inference.py:181-196 `_fuzzy_augment_rejected` applies the same global bad_rate weight to every reject (weight*bad_rate / weight*(1-bad_rate)); score_col is used only in the parceling branch (reject_inference.py:86-94).

**Why it matters.** a) is an honesty problem for a decision-support tool — infeasible should be stated as infeasible; b) directly affects the reject-inference-retrained model's ability to rank the rejected population, and makes the professional term "fuzzy_augmentation" a misnomer.

**How to fix.**
1. Add `feasible: bool` and `reason` to the recommend_operating_point return structure (e.g. '所有 cutoff 的 bad_rate 均高于约束 2%，返回坏账最低点仅供参考' — "every cutoff's bad_rate exceeds the 2% constraint; returning the lowest-bad-rate point for reference only"); tool_tradeoff_view passes it through; the render layer presents it in a warning color.
2. Add an optional `score_col` to `_fuzzy_augment_rejected`: when provided, derive per-row weights from per-record scores via a deterministic mapping (e.g. map score quantiles into the [bad_rate*0.5, bad_rate*2] range and renormalize the whole to the target bad_rate); when absent, keep the current global basis and record "全体拒绝件统一坏账假设" ("uniform bad-rate assumption for all rejects") in diagnostics.assumption.
3. Add assertion tests for both.

---

## 4. LLM Client & Token Engineering

**Lens verdict.** The LLM client itself is small and clean (145 lines, OpenAI-compatible, supports streaming/response_format/api_key_env), and the JSON extraction + retry work flagged in the 6-28 report has genuinely landed in decide_gate/route_instruction/planner/reviewer — it is no longer bare parsing. But the entire LLM surface still sits at the weakest constraint level, "json_object + post-hoc parsing", making no use of the json_schema/guided decoding that local inference stacks universally support. There is zero recording of per-call tokens/latency/success (tool calls have an audit trail; LLM calls have none). The model evaluation framework that has already been written (eval cases + calibrate_tier + regression_gate) has no production orchestrator implementing run_eval_case, which means the core question — "how do we verify no regression when swapping in a 32-72B model" — currently cannot be answered. Multi-model configuration has storage but no routing: every orchestration call is hard-wired to the default model. There is no context-window budget defense; blowing the window yields only a detail-stripped "LLM HTTP 400". On top of that: thinking-model `<think>` output receives no handling at all (a correctness risk of extracting a draft JSON from the reasoning scratchpad), draft authoring — the hardest generation task — is the one place with no retry, and streaming persistence performs a full-content UPDATE per delta, an O(n²) write amplification.

### LLM-1 · Structured output stuck at the weak json_object constraint, with json_schema/guided decoding unused — and some JSON call sites don't even enable json_object

**Impact:** High · **Effort:** M · **Verification:** ✅ Confirmed adversarially

**Problem.** The platform's target runtime is local 32-72B weak models (vLLM/SGLang/llama.cpp all natively support json_schema-level guided decoding), yet every critical decision point (gate-decision action enum, instruction-routing action enum, plan steps structure) uses only json_object — which guarantees nothing beyond "it is a JSON", not that fields, enum values, or nested structure are correct. The existing _parse_decision/parse_route/PlanningError retry machinery is entirely mop-up work for this weak constraint: a misspelled action → halt to a human; steps missing a title → burn a retry round. In addition, the reviewer/intent JSON call sites don't even enable json_object, making fenced output even more likely.

**Evidence.** marvis/llm_client.py:50-51 only passes response_format straight through: `if response_format: payload["response_format"] = response_format`; all 14 repo-wide hits pass {"type":"json_object"} (auto_drive.py:87/103, instruction_router.py:43/59, planner.py:162/219/263, authoring.py:95, derive.py:127, modeling/tools.py:4234, service.py:660). Meanwhile reviewer.py:65-69/128-132 (llm_critique/_llm_summarize) and intent.py:78-82 (_llm_classify), which equally expect JSON, pass no response_format at all and rely on load_json_object/regex fallbacks. The model profiles in llm_settings.py:48-69 carry no capability flags of any kind (no supports_json_schema).

**Why it matters.** This directly determines how "smart" the agent is on weak models: under a schema constraint the action field physically cannot emit an illegal value, gate decisions no longer halt to a human over formatting issues, planner first-pass rate rises, and retry rounds (tens of seconds of inference each) drop. It is the single highest-leverage change for the 32-72B target, and it does not violate INV-1 in any way (the LLM still only makes choices; it never computes metrics).

**How to fix.**
1. Add a structured_output field to the llm_settings model profile (enum json_schema|json_object|none, default json_object).
2. Add an optional json_schema parameter to llm_client.complete: when the profile supports it, send {"type":"json_schema","json_schema":{"name":...,"schema":...,"strict":true}}; on a server 4xx, automatically downgrade to json_object and remember it in the in-memory profile state (to avoid hitting the wall on every call).
3. Define schemas for the three high-value points: decide_gate (action as enum:[allowed_actions], required:[action,reason]), route_instruction (action enum + params object), planner steps (a skeleton schema for title/tool/inputs/depends_on, with detail still left to the validator).
4. reviewer.llm_critique/_llm_summarize and intent._llm_classify should at minimum add {"type":"json_object"}.
5. Keep the existing load_json_object + retry as the none/downgrade path.

**Verification note.** The adversarial pass confirmed every claim against live code, with one counting nuance: the "14 occurrences" figure includes the client's own definition/pass-through lines — the actual call sites passing {"type":"json_object"} number 11, which does not change the conclusion. The only theoretical bypass, the profile["extra_request_fields"] pass-through at llm_client.py:47-49, was verified to be stripped by the llm_settings save path and in any case can only inject one global response_format, not per-call-site schemas.

### LLM-2 · The already-built model eval framework cannot run against real models: run_eval_case has no production implementation, so "verify no regression on model swap" remains an open question

**Impact:** High · **Effort:** M · **Verification:** ✅ Confirmed adversarially

**Problem.** The framework, cases, fixtures, scoring, and regression gate are all written; the single missing piece is the executor that wires IntentRouter+Planner+Validator to a real LLM with tool outputs stubbed by fixtures, plus a CLI entry point. As a result, when the user swaps in a 32B/72B model (or a different quantization, or a different inference server), there is no automated way to answer "what is this model's template hit rate on my catalog, its plan first-pass rate, is the guardrail intact, which capability_tier should it be assigned" — the only option is trial-and-error on real tasks. Prompt changes (e.g. rewording PLAN_SYS) likewise have no regression defense.

**Evidence.** marvis/orchestrator/eval/ already contains a complete framework: cases.py:10-189 defines INITIAL_EVAL_CASES (five kinds — template_hit/plan_gen/replan/explore/guardrail — each case carrying offline-fixture tool_outputs); scoring.py:83-106 has calibrate_tier_for_model (runs the full suite per tier and recommends a capability tier), and scoring.py:109-126 has regression_gate (guardrail zero-tolerance + pass_rate drop gate). But scoring.py:78 depends on `orchestrator.run_eval_case(case, ...)`, and a repo-wide grep finds only the fake orchestrators in tests/test_orch_eval.py:162/243 implementing it; the CLI in marvis/__main__.py has only four subcommands — serve/validate/update/version — with no eval entry point.

**Why it matters.** This is the head-on answer to the brief's question "how do we verify a weak-model swap does not regress", and the safety net for all prompt/schema improvements (LLM-1, LLM-10): without it, every prompt tweak is a blind change. capability_tier is currently filled in by hand (llm_settings.py:81-83); calibrate_tier_for_model was designed precisely for automatic tiering.

**How to fix.**
1. Create marvis/orchestrator/eval/runner.py: an EvalOrchestrator holding intent_router/planner/validator plus a FixtureToolRunner (returns preset outputs from case.fixtures.tool_outputs, never actually runs tools, preserving the offline self-containment invariant). run_eval_case dispatches by kind: template_hit→intent.route; plan_gen→planner.generate+validator; replan→inject an observation to trigger planner.replan; guardrail→check validator/safety interception.
2. Add a `marvis eval --model-id X [--baseline path]` subcommand to __main__.py: runs calibrate_tier_for_model, outputs a per_tier report, writes workspace/eval/{model_id}-{date}.json; with a baseline, runs regression_gate and exits non-zero on failure.
3. A results panel can come later; make the CLI usable first.
4. Use the call records from LLM-3 to automatically attach per-case latency/retry counts, so evals also produce a speed profile.

**Verification note.** The adversarial pass confirmed the claim in full and added that the eval package has no production importers at all — run_eval_suite/calibrate_tier_for_model/regression_gate's sole consumer is tests/test_orch_eval.py, and no production path ever constructs PlanRunTrace (contracts.py:30).

### LLM-3 · Zero observability for LLM calls: tokens/latency/success/retry counts land nowhere — slowness cannot be diagnosed and no usage report can be produced

**Impact:** High · **Effort:** M · **Verification:** ✅ Confirmed adversarially

**Problem.** The platform's auditing of deterministic tools already achieves started+final dual checkpoints, yet the one nondeterministic component in the system — the LLM call — leaves no trace whatsoever: how many LLM calls an agent task actually issued, how long each took, how many retries were burned on JSON parse failures, which stage (planner? gate decision? summarization?) consumed 90% of the wait time — none of it can be answered. When the user perceives "the agent is slow / stuck" there is no data to localize the problem.

**Evidence.** complete() in marvis/llm_client.py:62-80 neither times the call nor reads usage; the streaming path _read_completion_content (llm_client.py:83-118) collects only delta.content — the usage terminal chunk of the OpenAI-compatible protocol is discarded, and the request does not carry stream_options={"include_usage":true}; the non-streaming path _content_from_json_response (llm_client.py:139-145) takes only message.content and likewise drops usage. Repo-wide grep for prompt_tokens/completion_tokens/latency = 0 hits. The audit trail has checkpoints such as tool.invoke.started/hook.dispatch.started (see the 06-28 fix list) but no llm.* audit kind of any sort; agent message metadata stores only model_id/display_name/model_name (validation_messages.py:49-54).

**Why it matters.** In a single-machine local-model setting, "cost" means wait time and VRAM residency: without per-call records there is no basis for optimization (nor can the payoff of multi-model routing, LLM-7, be quantified). It is also the natural extension of INV-8 (audit completeness) — gate-decision LLM output is already stored as an agent message, but "which model, which prompt version, how long, after how many retries" is unrecorded; the traceability chain breaks at its most critical link.

**How to fix.**
1. Instrument uniformly inside OpenAICompatibleLLMClient.complete: time with time.monotonic, send stream_options={"include_usage":true}, parse the final usage chunk on the streaming path and read data.usage on the non-streaming path.
2. Add an optional observer callback (or constructor argument) to complete; by default inject a recorder that writes to workspace/llm_calls.sqlite (or reuses an llm_calls table in the main DB): ts/call_kind/model_id/prompt_chars/prompt_tokens/completion_tokens/latency_ms/ok/error_kind/streamed; call_kind is supplied by the caller (plan/replan/gate/route/critique/summary/distill/author/narrative).
3. Migrate call sites through a thin wrapper (e.g. complete_json(kind=...)), which also unifies the schema entry point from LLM-1.
4. Expose /api/llm/usage with a per-call_kind aggregate report (count/avg latency/failure rate/retry rate), displayed on the frontend settings page.
5. Failure and retry counts also feed the speed profile of the LLM-2 evals.

**Verification note.** Confirmed with only minor line drift (complete() actually spans llm_client.py:18-80); the verifier additionally established that repo-wide grep for prompt_tokens/completion_tokens/include_usage/stream_options has zero *.py hits, that the only time.monotonic timers in the repo cover plugin tools and notebook kernels (not LLM calls), and that the planner's retry counter _attempt (planner.py:149/206/252) is never persisted.

### LLM-4 · Multi-model configuration has storage but no routing: every orchestrator LLM call is hard-wired to the default model, with no tiering by task complexity

**Impact:** Medium · **Effort:** M · **Verification:** — Not independently verified

**Problem.** A single V2 agent task mixes two classes of calls: heavy decisions (plan generation/replanning, gate decisions) and light chores (intent classification, one-line memory-distillation summaries, report narrative, learning-note compression). Today they all queue against the same default large model: the light chores waste large-model inference time (on a local single GPU with serial inference this directly slows the main flow), while giving the planner a stronger model on its own is impossible. A user who configures multiple models on the settings page will discover that outside V1.1 chat, the second model is never used anywhere.

**Evidence.** marvis/llm_settings.py supports multiple model profiles + default_model_id + resolution by model_id (resolve_llm_model, llm_settings.py:95-118), but `_llm_factory` in marvis/app.py:367-371 is pinned to `resolve_llm_model(settings.workspace)` without a model_id — planner/reviewer/intent/subagent/distiller all go through the default model; the V2 driver likewise: api.py:618 `resolve_llm_model(..., None)`. Only the V1.1 validation agent supports a per-request model_id (api.py:854-867).

**Why it matters.** On a single machine with one GPU, inference is the serial bottleneck: routing distill/intent/narrative-class calls to a 7B-class small model (or the same model at lower reasoning_effort) directly reduces main-path wait time; conversely, pinning planner/gate decisions to the strongest model raises the quality of the key decisions. This is exactly the gap the brief names: "small models for simple tasks, large models for critical decisions".

**How to fix.**
1. Add role_overrides to llm.json: {"planner":model_id,"gate":model_id,"router":...,"critique":...,"summary":...,"distill":...,"author":...,"narrative":...}, falling back to default_model_id when unset.
2. Add a role parameter to resolve_llm_model (look up the override internally, then resolve); change _llm_factory to _llm_factory(settings, role) and pass a role per component during app.py assembly (Planner uses planner, Reviewer uses critique, IntentRouter uses router, Distiller uses distill).
3. Give each role a dropdown on the settings page (default "follow the default model").
4. Use the call_kind records from LLM-3 to validate the payoff (before/after latency/failure rate per role).
5. Mind INV-4: distill belongs to the memory path, but swapping its model only changes summary wording, not deterministic behavior — safe.

### LLM-5 · No context-window budget defense: the profile doesn't know the model's window size, requests set no max_tokens, and overflow yields only a detail-stripped "LLM HTTP 400"

**Impact:** Medium · **Effort:** M · **Verification:** — Not independently verified

**Problem.** As the plugin ecosystem grows, the planner catalog balloons linearly and gate metadata thickens; a weak model's 8K-32K window will sooner or later be punched through. And the failure mode is the worst kind: the server returns 400, the client throws the body away, and the user/logs see only "LLM HTTP 400" — impossible to distinguish a window overflow from an unsupported response_format from an illegal parameter; nor is there any trimming fallback. Meanwhile, not setting max_tokens means a weak model in thinking mode can generate tens of thousands of tokens on a single gate decision until it times out.

**Evidence.** The profile fields in llm_settings.py:48-69 are only model_id/enabled/display_name/provider/api_base_url/model_name/api_key(_env)/enable_thinking/reasoning_effort/timeout_seconds — no context_window or max_output_tokens; the payload built in llm_client.py:35-51 contains no max_tokens; the HTTPError handler (llm_client.py:65-73) deliberately drains and discards the body (to prevent prompt echo into storage), leaving only "LLM HTTP 400 Bad Request". Budget control is scattered: replan uses fit_to_budget max_chars=4000 (planner.py:198-204), V1.1 conversation uses 32000 characters (service.py:70-72), but build_plan_prompt's catalog/memory_context/task_context has no aggregate cap at all (planner.py:339-357), and the gate decision's _format_gate truncates each table to 20 rows but places no cap on the number of tables (auto_drive.py:184-192).

**Why it matters.** This is the "wall" of a weak-model platform: when you hit it, the diagnostic cost is extreme (the user will only say "the agent is broken"). And with each call site inventing its own budget (4000/32000/20 rows) and no common exchange rate or master valve, any prompt change can overflow someone else's window.

**How to fix.**
1. Add context_window (default 32768) and max_output_tokens (default 2048) to the profile.
2. Add a max_tokens parameter to llm_client.complete and put it in the payload; call sites assign values by type (JSON decisions 512, summaries 1024, planner 4096).
3. Do a client-side token estimate before sending (a conservative mixed estimate of CJK≈1.6 chars/token, ASCII≈4 chars/token is sufficient); when system+user+max_tokens exceeds context_window, raise an LLMClientError with explicit sizes ("prompt ≈ N tokens exceeds model window M") so call sites can take their existing degradation paths.
4. In the HTTPError branch, parse whitelisted error.code/error.type fields from the body (take enums only, never the message — preserving the no-prompt-echo constraint), and report context_length_exceeded as a distinct cause.
5. Put a fit_to_budget master valve over build_plan_prompt's catalog+memory+task_context (priority: instruction > catalog > examples > task_context > memory).

### LLM-6 · No handling of thinking-model output: `<think>` text pollutes messages and JSON extraction may grab the reasoning draft; the enable_thinking wire convention mismatches mainstream local stacks

**Impact:** Medium · **Effort:** M · **Verification:** — Not independently verified

**Problem.** Three compounding consequences: (a) V1.1 streaming chat/stage summaries flush the entire reasoning text to the user verbatim and persist it (_strip_agent_response_preamble does not handle `<think>`); (b) if the gate decision/routing receives `<think>`-bearing text because the server does not support json_object, load_json_object may parse the draft decision inside the reasoning rather than the final decision — a correctness risk, not merely cosmetic; (c) the settings page's "思考模式" (thinking mode) toggle sends fields that the most common vLLM+Qwen stack does not recognize — i.e. it is silently ineffective (the user believes it is on when it is not).

**Evidence.** llm_client.py:44-46: with enable_thinking the client sends `payload["reasoning_effort"]=...` plus `payload["thinking"]={"type":"enabled"}` (the former is the OpenAI o-series convention, the latter Anthropic's; the Qwen3/vLLM convention is chat_template_kwargs={"enable_thinking":true}). Stream parsing reads only delta.content (llm_client.py:121-136), and repo-wide grep for reasoning_content/`<think>` yields 0 hits — meaning when the server has no reasoning parser configured and mixes `<think>...</think>` into content (llama.cpp/ollama/vLLM without --reasoning-parser all do), nothing strips it. _extract_first_object in json_reply.py:33-58 takes the **first** balanced {} block — a JSON the model drafts inside its thinking section will be matched before the final answer.

**Why it matters.** The platform's target models (the Qwen3 32B/72B class) are precisely the ones most afflicted by interleaved thinking output; a gate decision grabbing the wrong JSON can read a drafted "confirm" instead of the final "halt" — a direct reversal of direction. And a silently ineffective toggle is a configuration-trust problem.

**How to fix.**
1. Strip paired `<think>...</think>` uniformly in llm_client before returning (including streaming: either accumulate then strip, or run a state machine over on_delta that skips think segments — the latter protects the streaming UI).
2. Change json_reply.load_json_object to strip think first and reorder candidates to "whole string → last balanced object → first balanced object" (the final answer is almost always at the end).
3. Make the enable_thinking request side provider-aware: add thinking_style to the profile (qwen_chat_template|openai_reasoning|anthropic|none); the qwen style sends chat_template_kwargs, and the existing two fields are sent only under their corresponding styles; document that extra_request_fields can override.
4. Add tests: a fixture containing a `<think>` prefix + draft JSON + final JSON, asserting the final block is extracted.

### LLM-7 · Zero transport-layer retries: a single timeout/network blip terminates AUTO drive or an entire planning round — asymmetric with the JSON-parse-layer retries

**Impact:** Medium · **Effort:** S · **Verification:** — Not independently verified

**Problem.** Local inference servers (vLLM/ollama) routinely produce sporadic timeouts and connection resets under long-task concurrency/VRAM paging. Current behavior: format errors (the model's fault) get gentle feed-back-and-retry, while a network blip (the environment's fault — the failure class most likely to succeed on retry) dies on the first strike. AUTO mode reaching its 5th gate and hitting a single 60s timeout means the user comes back to find the agent parked at the gate awaiting manual takeover.

**Evidence.** llm_client.py:62-79 makes a single request; URLError/Timeout/OSError raise LLMClientError immediately, with no retry/backoff of any kind. All call-site retries target only JSON parse failures (auto_drive.py:90-106, instruction_router.py:46-62, planner.py:149-181's for _attempt loop catches PlanningError, not LLMClientError). turn_handlers.py:461-470: in the AUTO drive loop, an LLMClientError from decide_gate posts "⚠️ 自动决策失败…请手动确认" (auto decision failed… please confirm manually) and returns, terminating the entire automated session. An LLMClientError from planner.generate is not caught by except PlanningError and bubbles up as a whole-round failure.

**Why it matters.** This directly determines the unattended success rate of long tasks — AUTO mode's core selling point is "walk away and it finishes". The fix is extremely cheap (one loop inside a single function) and the benefit covers all 25 call sites.

**How to fix.**
1. Add bounded retries inside OpenAICompatibleLLMClient.complete: retry only on URLError/TimeoutError/OSError/HTTP 5xx (never 4xx), default 2 attempts, backoff 1s/4s, max_retries configurable per profile.
2. Do not retry an interruption after on_delta output has started (to avoid the user seeing content roll back); keep the current raise behavior there.
3. Feed retry counts into the LLM-3 call records.
4. Total wall-clock cap = timeout_seconds × (retries+1); append "automatically retried N times" to the error message in turn_handlers.

### LLM-8 · Draft tool authoring is the hardest single-shot generation task, yet it is the only LLM touchpoint with no JSON tolerance and no error-feedback retry

**Impact:** Medium · **Effort:** S · **Verification:** — Not independently verified

**Problem.** Asking a 32-72B model to produce, in one shot, name+summary+complete Python code that passes the AST safety scan+two JSON schemas+a determinism declaration is the platform's most demanding single-shot output task; and precisely here there is neither fence stripping nor targeted retries of the form "your code contains the banned call open, please rewrite". On weak models this will most likely present as the draft feature "failing on every click", with users abandoning the capability.

**Evidence.** marvis/drafts/authoring.py:92-109: draft_script does a single complete then _safe_json_loads (authoring.py:173-180 uses bare json.loads, not marvis/agent/json_reply.load_json_object); any fence/prefix text goes straight to AuthoringError("LLM output is not valid JSON"); the subsequent _assert_required_keys/_assert_schema/assert_draft_code_safe/function-name checks (authoring.py:99-109) likewise abort on failure — there is no mechanism to feed validation errors back to the model for a retry. Contrast planner.py:149-181, which already has the max_retries+last_error feedback pattern.

**Why it matters.** draft→test→promote is the platform's entry point for self-extending its tool ecosystem (the seed path for a future plugin ecosystem); a low success rate at the entrance leaves the entire web-learning→draft→promote chain existing in name only.

**How to fix.**
1. Replace _safe_json_loads with load_json_object (reusing fence/prefix stripping).
2. Add a retry loop (2 attempts) to draft_script: catch AuthoringError and splice the error text (including the banned-calls hit list, missing keys, schema validation messages) into a retry prompt for targeted correction — the safety floor is unchanged, assert_draft_code_safe remains a hard gate; the retry merely gives the model a chance to fix its mistake.
3. Attach a minimal legal-output few-shot to AUTHOR_SYS (one complete spec for a 10-line pure-pandas tool) — weak models' format adherence improves markedly.
4. Record retry counts and the final failure reason in the draft.author audit metadata (that audit transaction already exists).

### LLM-9 · Every streaming delta triggers a full-content SQLite UPDATE: O(n²) write amplification plus a new connection per delta

**Impact:** Medium · **Effort:** S · **Verification:** — Not independently verified

**Problem.** All streaming messages of the V1.1 validation agent (stage summaries/chat/failure analyses) travel this path, sharing a single SQLite with the frontend's 180ms message-poll reads and the background notebook/metrics stages' business writes. During a long answer the WAL is continuously flooded with full-text rewrites, amplifying busy contention; this is also the highest-frequency trigger of the previously reviewed "PRAGMA handshake re-run on every connect" issue.

**Evidence.** marvis/agent/validation_messages.py:110-119: on_delta, for every streaming fragment received, does parts.append and then executes repo.update_agent_message(message_id, content="".join(parts), ...) — writing **the entire content so far**; update_agent_message in repositories/tasks.py:621-641 opens a fresh connection per call via `with connect(self.db_path)` to run UPDATE+SELECT. vLLM-class servers typically emit deltas per token, so an 800-token Chinese analysis ≈ 800 connections+UPDATEs, with cumulative bytes written being O(n²) (late in the stream, every write rewrites the full text).

**Why it matters.** It hits the exact moment the user is watching (UI polling stutters during streaming output), and the fix is zero-risk — the frontend already polls full messages every 180ms, so per-delta persistence has no consumer value; one only needs to keep the flush throttle interval ≤ the polling interval.

**How to fix.**
1. Change stream_agent_message's on_delta to throttled flushing: if <150ms since the last write and <256 new characters accumulated, only append without writing to the DB (the cancellation check raise_if_cancelled still runs on every delta).
2. Keep the final full update after the producer returns (already present).
3. While at it, put update_agent_message's hot path on in-thread connection reuse (the threading.local scheme already proposed in section G of the earlier report — doing it inside this repository method alone is enough for now).
4. Regression: the existing test_agent_api streaming cases assert the final content is unchanged; add an assertion that "delta count >> flush count".

### LLM-10 · System prompts scattered across 13 modules with no version identifiers: prompt tuning is untraceable and cannot be linked to evals/audit

**Impact:** Low · **Effort:** M · **Verification:** — Not independently verified

**Problem.** On weak models, prompt wording is the primary tuning knob, and Chinese prompts are extremely wording-sensitive. As things stand: change one word in PLAN_SYS, discover two weeks later that plan pass rate has dropped, and there is no way to answer "which change introduced it"; gate-decision messages store the LLM output without knowing which version of the system prompt was in effect; the eval baseline (LLM-2) pass_rate cannot be pinned to a specific prompt version for comparison.

**Evidence.** System prompts are scattered as module-level constants: PLAN_SYS/REPLAN_SYS/EXPLORE_SYS (planner.py:15/22/27), CRITIC_SYS (reviewer.py:22), CLASSIFY_SYS (intent.py:11), _SYSTEM_TEMPLATE (auto_drive.py:59), _SYSTEM (instruction_router.py:23), AGENT_SYSTEM_PROMPT/WORD_CONCLUSION_SYSTEM_PROMPT (agent/prompts.py), DISTILL_SYS (agent_memory/distillation.py), AUTHOR_SYS (drafts/authoring.py:21), LEARN_SYS (drafts/learning.py:12), CROSS_SYS (feature/derive.py), REPORT_NARRATIVE_SYS (packs/modeling/tools.py:4205). There is no version constant anywhere; grep PROMPT_VERSION = 0 hits; the LLM call log (nonexistent, see LLM-3) naturally cannot record prompt versions either.

**Why it matters.** This is the glue layer between LLM-2 (evals) and LLM-3 (observability): only the three together form the closed loop "prompt change → eval → ship → traceable regression". On its own it is engineering hygiene; combined, it is the tuning infrastructure of a weak-model platform.

**How to fix.**
1. Create marvis/llm_prompts.py (or a marvis/prompts/ package): register each prompt as PromptSpec(name, version, text), where version is a manually incremented integer plus modification date; move the existing 13 constants in and re-export them from their original modules so call sites need zero changes.
2. Add prompt_name/prompt_version fields to the LLM-3 call records.
3. Have the LLM-2 eval result JSON carry a snapshot of all prompt versions, so the regression_gate report can show a direct "version diff".
4. Add a lightweight test: bind each registered prompt's text hash to its version; if text changes without a version bump, the test fails (forcing the bump, preventing silent edits).

---

## 5. Performance & Efficiency

**Lens verdict.** Overall diagnosis from the performance lens: the data plane (DuckDB pushdown, sample-level profiling, the 200k-row guardrail, vectorized JOIN match-rate, `after_id` incremental message polling) is clearly better than at the 06-28 review, but three systemic bottleneck classes remain. (1) The heaviest synchronous work on the interactive path (JOIN propose/confirm/execute, dataset upload-to-parquet conversion) hangs inside `async def` endpoints and directly blocks the uvicorn event loop — during large-table operations the entire service (including the 1s task polling and the 180ms message polling) freezes. This is the number-one experience killer for a single-machine product at the target data volume (millions of rows). (2) Repeated computation with no caching: JOIN alignment redoes full-table COUNT/DESCRIBE/reservoir sampling and SELECT DISTINCT for every candidate column × method combination; the V1.1 staged pipeline in isolated mode executes the user notebook twice in full; every tool step pays a 1–2.3 second subprocess cold start (measured); multi-recipe training performs a full-frame copy of the entire frame per recipe. (3) H3, marked "fixed" in the previous report, was measured as not fully fixed — uniqueness determination and deduplication still operate in the raw key space, so in exact_lower/hash/date scenarios the JOIN still hard-fails with fan-out at execute time even after the user confirms and selects deduplication (confirmed with a minimal reproduction script; the INV-3 safety net protected the data, but the flow is a dead end). Items verified as genuinely fixed and not re-reported: the incremental message cursor, match-rate DuckDB pushdown, WAL+busy_timeout, and Notebook RSS soft monitoring.

### PERF-1 · JOIN propose/confirm/execute and dataset upload run heavy work synchronously inside `async def` endpoints, blocking the entire event loop — the whole service freezes during large-table operations

**Impact:** Critical · **Effort:** S · **Verification:** ⚠️ Partially confirmed

**Problem.** FastAPI does not offload `async def` endpoints to a thread pool. During a propose on a million-row table (the alignment scan can take minutes), an execute (DuckDB COPY join), or a 2GB CSV upload, the uvicorn event loop is completely monopolized: the 1s task polling, the 180ms agent message polling, and `/api/health` all become unresponsive, and the UI appears as a site-wide freeze. The upload additionally causes a whole-file in-memory peak.

**Evidence.** marvis/routers/data.py:231 `async def propose_join` → :256 synchronously calls `join_engine.propose_join_plan(...)` (which internally performs full-table alignment scans / COUNT DISTINCT / conflict report); data.py:271 `async def confirm_join_plan` → :290 synchronously runs `diagnose_join` when the user changes key_pairs; data.py:328-366 `async def execute_join_plan` executes the LEFT JOIN synchronously when `_join_async_requested(payload)` is false, while the frontend marvis/static/js/v2/api_v2.js:64 `executeJoin = (joinId) => apiPost(..., {})` sends an empty object and therefore always takes the synchronous path; data.py:117 `async def upload_task_dataset` → :132 `await file.read()` reads the whole file into memory + :180 `register_from_upload` (full `pd.read_csv` → parquet + profile), all running on the event-loop thread.

**Why it matters.** This is the most direct experience killer for a single-machine, single-user product at the target data volume: at the most critical C2 JOIN gate and at upload, what the user sees is "页面卡死" (the page appears frozen), which easily leads them to misjudge the service as crashed and force-kill the process (destroying any training run in flight). Every other optimization is blocked behind this one.

**How to fix.**
1. Minimal change: convert these four endpoints in data.py from `async def` to `def` (FastAPI automatically runs them in a threadpool; replace the in-body `await request.json()` with synchronous `request._body` parsing or switch to a Pydantic body parameter) — a one-line signature change removes the blocking.
2. Frontend: change `executeJoin` at api_v2.js:64 to send `{"async": true}` so it takes the backend's already-implemented 202+job polling path (data.py:343-361); add an async job version for propose as well (reuse the `start_task_job` + BackgroundTasks pattern).
3. Switch the upload to streaming-to-disk: `while chunk := await file.read(1<<20): fh.write(chunk)`, and wrap the register/profile portion in `await run_in_threadpool(...)`.
4. Add a regression test: while a propose is in progress, hit `/api/health` concurrently and assert it returns in <100ms.

**Verification note.** The adversarial pass confirmed the core claim for propose/confirm/upload: they are `async def` endpoints doing heavy synchronous work with zero `run_in_threadpool`/`to_thread` usage anywhere in the repo, on a single-worker uvicorn process — so the event loop is indeed blocked. However, the execute-join sub-claim is outdated: the current frontend (api_v2.js:69-71) defaults `executeJoin` to `{ async_execute: true }`, and the sole caller (join_review.js:438) uses that default, so the real UI flow takes the 202 + BackgroundTasks path (with `_run_join_execute_job` run in Starlette's threadpool) followed by `pollJoinExecution`. The synchronous execute path (data.py:362-366) is only reachable when an API caller explicitly posts a body without any async flag.

### PERF-2 · H3 not fully fixed: key uniqueness and deduplication are still computed in the raw key space while the JOIN matches on transformed keys — in exact_lower/hash/date scenarios, even after the user confirms and selects deduplication, execute inevitably hard-fails with fan-out (reproduced)

**Impact:** High · **Effort:** M · **Verification:** ✅ Confirmed adversarially

**Problem.** The 2026-06-28 runtime review's fix status claimed "H3: uniqueness/deduplication changed to operate in the transformed key space consistent with the actual JOIN", but in the current code only match-rate was moved to the transformed space. The consequence chain: diagnose reports `feature_key_unique=True` → the C2 gate does not require deduplication → or, even if the user proactively selects first/last/agg, partitioning by the raw key keeps both the 'ABC' and 'abc' rows → the 1:1 assertion at execute time throws FanOutError. The data is never corrupted (INV-3 catches it), but this path is a dead end: no matter which dedup strategy the user picks, the JOIN cannot succeed, and each failure wastes a full-table JOIN computation. md5-hashed phone numbers with case differences, mixed-case IDs, and multi-format dates are exactly the most common shapes of credit-risk feature tables.

**Evidence.** marvis/data/backend.py:228-229 `is_key_unique` = raw-column distinct_count==row_count; backend.py:476-479 `_dedup_feature_rel`'s PARTITION/GROUP BY key_sql uses raw column names (:512/:522/:552); whereas the actual JOIN condition backend.py:554-565 `_join_condition` wraps both sides in `_sql_transform` (lower()/hash()/strftime). Reproduced with a minimal script: feature contains 'ABC'/'abc' (unique in raw key space → is_key_unique=True), anchor='abc', method=exact_lower, dedup_strategy='first' → `left_join fan-out: produced 3 rows from 2 anchor rows`. Call sites: marvis/data/join_engine.py:124/237 (uniqueness) and :157 (conflict report also uses raw keys).

**Why it matters.** This directly undermines the core promise — "JOIN never silently misaligns and can be legitimately executed after user confirmation": the user does everything right at the C2 gate and still fails, which reads as "the platform's dedup feature is broken"; it also wastes the entire computation of a large-table JOIN. This belongs to the previous report's category of "claimed fixed but not fully fixed".

**How to fix.**
1. Extract a `_transformed_key_exprs(key_pairs, columns, side)` helper in the backend that reuses `_sql_transform`; add an optional `key_pairs` parameter to `distinct_count`/`is_key_unique`/`_dedup_feature_rel`/`conflict_report`: when provided, COUNT DISTINCT / PARTITION BY on the transformed expressions; when absent, keep raw columns (backwards compatible with existing callers).
2. Have `join_engine.diagnose_join` (:124/:147/:157) and the execute path (backend.py:256/259) uniformly pass the same set of KeyPairs.
3. Keep first/last dedup ordering by file_row_number unchanged; only swap the PARTITION expression.
4. Regression tests: 'ABC'/'abc' + exact_lower asserting is_key_unique=False and that after dedup=first the left_join succeeds and is 1:1; add one case each for md5 case differences and %Y%m%d vs %Y-%m-%d.

**Verification note.** The verifier reproduced the failure end-to-end with a minimal script (feature rows 'ABC'/'abc', anchor 'abc'/'zzz', match_method='exact_lower'): `is_key_unique` returned True, and all four dedup strategies (first/last/agg_max/None) threw the exact error `left_join fan-out: produced 3 rows from 2 anchor rows (must be 1:1)`, empirically confirming the "dead end regardless of dedup choice" claim and the mismatch with the fix claim at docs/reviews/2026-06-28-v2-runtime-deep-review.md:23.

### PERF-3 · In isolated mode the staged pipeline executes the user notebook twice in full: the metrics stage force-closes the session and replays the entire notebook from scratch

**Impact:** High · **Effort:** M · **Verification:** ✅ Confirmed adversarially

**Problem.** The default path of `run_staged_pipeline` (pipeline.py:1261-1279) = the notebook stage runs the full notebook once + the metrics stage runs the full notebook again, plus two jupyter kernel cold starts. User notebooks typically contain heavy computation such as training/scoring, so the total validation wall time doubles outright; this is also the single largest factor behind the user perception that "验证很慢" ("validation is slow").

**Evidence.** marvis/pipeline.py:334-337: in isolated mode, `run_metrics_stage` first runs `close_live_notebook_session(task_id); live_session=None`; then :357-397 the `if live_session is None:` branch calls `_notebook_step_v3(...)`, which re-prepares and executes the complete prepared notebook (all source notebook cells + injected metrics cells); yet `run_notebook_stage` (pipeline.py:173-200, keep_alive=False) had just executed the whole thing once (source cells + injected reproducibility cells). `_notebook_step_v3` (pipeline.py:1965-2001) always starts from `prepare_execution_notebook_v3` and runs every cell.

**Why it matters.** V1.1 validation is one of the product's main paths; for notebooks that include model training, the duplicated execution can waste minutes to tens of minutes of pure compute, and it also exposes a doubled memory/CPU peak window to everything else on the single machine (LLM, training subprocesses).

**How to fix.**
1. In `run_staged_pipeline`, detect the normal path where notebook and metrics stages execute consecutively: in a single kernel run, inject both the reproducibility and metrics cell groups (the two groups are already mutually independent via the `_append_injected_cells` mechanism, each with its own output files); on successful execution, advance state and commit artifacts for both stages in the existing order. Only a metrics-only retry (kernel already gone) should take the "replay notebook" path — this does not violate INV-6, since the isolation boundary remains the subprocess kernel.
2. Alternative: in isolated mode, let the notebook stage keep_alive until the metrics stage of the same pipeline job finishes before closing (the process is still isolated from the API).
3. Add a regression test using a fixture notebook containing a sleep, asserting each source cell executes exactly once.

**Verification note.** The verifier confirmed the double execution and both kernel cold starts, with one nuance: pipeline.py:334-337 is a conditional cleanup of stale sessions (not "unconditional"); the true root cause is that the isolated branch of `run_notebook_stage` uses keep_alive=False and never retains a kernel, so the metrics stage has no state to reuse. Notably, the behavior is locked in by a test (tests/test_pipeline_v2.py:857 `test_metrics_stage_reruns_isolated_notebook_even_with_stale_live_session`) — it is deliberate design consistent with the subprocess-isolation decision, not an accidental regression, but the doubled execution cost is fully real.

### PERF-4 · JOIN alignment/diagnosis repeats full-table COUNT, DESCRIBE, full-table reservoir sampling, and SELECT DISTINCT for every candidate column × method combination, with zero caching — minute-level waits at the C2 gate on large tables

**Impact:** High · **Effort:** M · **Verification:** ✅ Confirmed adversarially

**Problem.** A single `propose_join` on a 5M-row anchor + 10M-row feature produces dozens of full-table scans: the same anchor sample (same path, same seed, same n) is recomputed identically N times, the same file's DESCRIBE/COUNT is repeated dozens of times, and the feature side redoes a full-table DISTINCT for every candidate method. This determines how long the user waits to see results at the JOIN gate, and it is the main latency component of every turn in the agent-driven JOIN stage.

**Evidence.** marvis/data/backend.py:297-331 `match_rate_for_method` executes on every call: :317-318 two `column_names` calls (one DESCRIBE each; for CSV this is full-file inference) + :319 `sample_rows` (internally `row_count` full-table COUNT(*), and for >200k rows a full-table reservoir sample) + :410 a full-table `SELECT DISTINCT` on the feature (for CSV, :440 `read_csv_auto` re-parses the whole file); marvis/data/align.py:80-96 calls it once per key family × per anchor column × per feature candidate column × per method (the hash family can reach 4 algorithms, contracts.py:20); join_engine.py:174-178 relaxation alternatives run another round for each reduced key (:223/:237/:241); the entire marvis/data has no caching whatsoever (grep lru_cache/_cache/mtime = 0 hits).

**Why it matters.** JOIN is the first gate of the V2 three-stage flow — the first screen where the user builds trust; millions of rows is the product's stated target scale. The current implementation makes "smart alignment" slow enough at real scale to be mistaken for a hang (compounded by the event-loop blocking of PERF-1).

**How to fix.**
1. Add an instance-level metadata cache to DataBackend: `self._meta = {}` keyed `(path, mtime_ns, size)` → {column_names, row_count, numeric_columns}; invalidation on file change, so determinism is unaffected.
2. Cache the anchor sample keyed `(path, mtime_ns, size, n, seed)` — same seed + same file should already produce bit-identical samples (upholding INV-1).
3. Merge the multiple candidate methods for the same column pair in `align._resolve_by_data` into one DuckDB query: a single scan computing match counts for multiple transformed keys simultaneously (`SELECT count(*) FILTER(...)`, one column per method).
4. Materialize the feature-side normalized key sets as duckdb temp tables keyed `(path, mtime, col, method)`, reused across diagnose and the relaxation alternatives.
5. Benchmark: full propose flow on a 2-million-row synthetic table in <10s.

**Verification note.** All core facts were confirmed against current code, with one detail corrected in the reviewer's favor: the grep for `lru_cache|_cache|mtime` in marvis/data actually yields 0 hits (not 1 — the claimed backend.py:420 hit was an artifact of a pattern containing 'memo' matching `":memory:"`), and the entire marvis package has no cache/memoization mechanism at all; DataBackend instances are created fresh per request (routers/data.py:53, turn_handlers.py:569).

### PERF-5 · Every tool step pays a 1–2.3 s subprocess cold start (measured), of which ~1 s is purely because the worker's top-level import drags in the entire marvis.db→packs.modeling→sklearn chain

**Impact:** High · **Effort:** S · **Verification:** ✅ Confirmed adversarially

**Problem.** The worker itself only needs the stdlib plus the lightweight ToolContext dataclass, yet importing runner pulls the DB layer and the entire modeling package (including sklearn) into every cold start. A 15–20 step V2 plan pays ~20–45s of pure cold-start overhead; a single tool at an interactive gate (propose_join, compute_feature_metrics) also waits an extra 1–2s each time, stacked on top of LLM latency.

**Evidence.** marvis/plugins/runner.py:636-654: every invoke does `subprocess.Popen([python, '-m', 'marvis.plugins.subprocess_worker'])`; marvis/plugins/subprocess_worker.py:21 top-level `from marvis.plugins.runner import ToolContext`. Measured (/opt/miniconda3/envs/py_313): `python -c pass`=0.01s; `import marvis.plugins.subprocess_worker`=1.08s; `-X importtime` shows 1.02s coming from runner→marvis.db→repositories.modeling→packs.modeling (sklearn/scipy chain); `import marvis.packs.modeling.tools`=2.3s.

**Why it matters.** This directly determines the perceived "responsiveness" of every agent action and the total plan wall time; it is also one of the few spots where a clear win is available without touching INV-6 (subprocess isolation, one process per job).

**How to fix.**
1. Step one (S, immediately actionable): move ToolContext into a dependency-free lightweight module (e.g. marvis/plugins/contracts.py); have both runner and subprocess_worker import it from there; remove all heavy marvis imports from the worker's top level (importlib already loads the tool module on demand inside `_run_tool`) → cold start drops from ~1.1s to ~0.05s + the tool module's own import.
2. Step two (M, optional): a "pre-warmed one-shot standby worker" — immediately after dispatching a job, Popen the next worker in the background so it completes interpreter + common-package imports and blocks reading the job from stdin; each worker still executes exactly one job then exits (preserving INV-6's one-job-one-process semantics), hiding the cold start inside the previous step's execution time. Do not pre-warm third-party plugins, to avoid module pollution.

**Verification note.** The verifier reproduced the measurements (import subprocess_worker=1.01s, `-X importtime` cumulative 1.028s with the runner chain at 1.009s; `import marvis.packs.modeling.tools`=2.11s vs the reported 2.3s, same order of magnitude), with one correction: sklearn is not actually in the top-level cold-start chain — it is lazily imported inside functions (select.py:246, tools.py:3434); the real top-level weight is scipy.stats (0.62s) + pandas (0.32s). The conclusion stands unchanged.

### PERF-6 · The 1-second polling loop = full task list with N+1 queries + a file stat per task + full re-read of 5 evidence JSON files, and every repository call opens a new SQLite connection running 5 PRAGMAs

**Impact:** Medium · **Effort:** M · **Verification:** — Not independently verified

**Problem.** Once task history grows (a few hundred tasks), every polling tick is 1+N SQLite connections+queries + N filesystem stats + multiple JSON parses; meanwhile the 180ms message polling is also opening connections. It won't crush a single machine, but it is a steady fixed cost per second, worsens SQLite busy contention when running in parallel with long tasks, and keeps CPU/battery busy while "nothing is happening".

**Evidence.** marvis/static/app.js:5627-5631 `refreshTasks` calls `api("api/tasks")` every second with no limit; marvis/routers/tasks.py:53-56 calls `task_payload` for every task; marvis/api_task_payloads.py:22 one `repo.get_active_job_kind` per task (repositories/tasks.py:374-386, a separate connect+query), :27 one `.exists()` check for `validation_report.docx` per task; marvis/routers/evidence.py:27-33 re-reads and `json.loads` 5 files every second (validation_results.json can be large); marvis/db_schema.py:676-686 every connect creates a new connection, :648-659 runs 5 PRAGMAs including journal_mode=WAL; the plans side has the same shape: repositories/plans.py:105-113 `list_plans_for_task` calls `load_plan` separately for each plan.

**Why it matters.** This affects background overhead and UI smoothness in all long-task scenarios; it also bounds how far this polling architecture can scale in task count. It belongs to the category of "deterministic gain, zero behavioral risk" cleanups.

**How to fix.**
1. Eliminate the N+1 in `list_tasks`: one `SELECT task_id, kind FROM jobs WHERE status IN ('queued','running')` + one failed-job aggregation, mapped into payloads via a dict (all within a single connection).
2. Change `report_available` to a boolean column written on the tasks row when the report stage completes (or cache by `(task_id, mtime)`), removing the N stats per second.
3. Support ETag on `/evidence` (hash of the 5 files' max(mtime_ns) concatenation), return 304 when unchanged, and have the frontend `api()` pass through If-None-Match.
4. Add a limit to `refreshTasks` (visible list area) or an `updated_since` cursor.
5. Mid-term: a single SSE channel pushing task/plan/message change events, with polling as fallback.

### PERF-7 · Zero real feedback during a local weak model's 10–60 s generation: the LLM client accumulates the entire SSE stream before returning, the driver turn lands one complete message only at the end, and the frontend has only a fake typewriter

**Impact:** Medium · **Effort:** M · **Verification:** — Not independently verified

**Problem.** The target deployment is a local 32–72B model: one driver turn = an instruction-routing LLM call + a gate-decision/summary LLM call, executed serially — easily 20–60s, during which the user sees only a static thinking bubble; the 180ms high-frequency polling spins entirely idle during this window (stacking on top of PERF-6's overhead). The user cannot distinguish "the model is generating" from "it's hung", and in practice tends to resend the message or refresh, creating duplicate turns.

**Evidence.** marvis/llm_client.py:89-112 collects stream_parts per SSE event but ultimately `return "".join(stream_parts)` (no on_delta callback); marvis/agent/auto_drive.py:97/:113 and instruction_router.py:39-45/:55-61 — one gate turn makes 2+ serial `complete(stream=False)` calls; `update_agent_message` is used repo-wide only by validation_messages.py/validation_stages.py (V2 driver turns never write incrementally); frontend app.js:250 AGENT_STREAM_POLL_INTERVAL_MS=180 can only poll "complete messages", and the typewriter at app.js:5101 is a rendering animation, not a real stream.

**Why it matters.** LLM latency is an irreducible physical quantity under weak-model deployment; masking it (real streaming + stage feedback) is the only lever available. This determines the first-impression sense of "does the agent seem smart and reliable".

**How to fix.**
1. Add an optional `on_delta(text)` callback to `llm_client.complete` (the SSE loop at :102-107 already receives increments per event; the change is minimal).
2. At driver-turn start, `add_agent_message` an assistant message with metadata={streaming:true}; inside on_delta, throttle to ~400ms and call `update_agent_message` to overwrite content; clear the streaming flag at turn end — this only writes the conversation table and touches no deterministic outputs (upholding INV-1/INV-4).
3. In the frontend, have `mergeIncrementalAgentMessages` route the "last streaming message" through the existing in-place content update path (`updateAgentMessageContentsInPlace` in agent-conversation-mount.js).
4. Drop a stage hint message at each deterministic checkpoint of the turn (routing done / tool started / tool finished), making "which step am I at" visible.

### PERF-8 · DuckDB globally unconfigured: large JOINs cannot spill to disk, defaults consume all cores and ~80% of memory, competing with training subprocesses / the local LLM — OOM shows up as an opaque crash

**Impact:** Medium · **Effort:** S · **Verification:** — Not independently verified

**Problem.** DuckDB defaults to memory_limit≈80% of physical memory, threads=all cores, and without a temp_directory some operators cannot fully spill. On a single machine, JOIN/feature-engineering work concurrently squeezes training subprocesses and a (possibly co-located) local LLM; under the worker's RLIMIT, a large JOIN ends as a "resource error / process killed", whereas with temp_directory + a conservative memory_limit configured the same JOIN could have succeeded slowly.

**Evidence.** Repo-wide grep for `SET threads|memory_limit|temp_directory|PRAGMA` (duckdb-related) yields 0 hits; the only explicit connection at marvis/data/backend.py:420 is also unconfigured; the large COPY query in left_join (backend.py:275-282) and all `duckdb.sql` calls use the global default connection; on the tool-worker side, subprocess_worker.py:566-571 sets RLIMIT_DATA/AS, after which DuckDB exceeding the limit usually manifests as a native exception / process death rather than a controlled spill.

**Why it matters.** This decides whether "a million-row JOIN is slow but succeeds, or simply fails"; it also determines whether background heavy work starves the UI polling. For the target scenario (LLM + training + JOIN on one machine simultaneously), this is necessary resource governance.

**How to fix.**
1. Create marvis/data/duck.py: `get_conn()` returning a process-level pre-configured connection (`SET memory_limit='50%'`, `SET temp_directory='<workspace>/.duckdb_tmp'`, `SET threads=max(2, cpu//2)`, all overridable via Settings).
2. Route all `duckdb.sql(...)` calls in the backend through `get_conn().sql(...)` (keep preserve_insertion_order default true so row-order determinism is unchanged, upholding INV-1).
3. Call the same configure inside the subprocess worker (subprocesses have independent connections).
4. Test with a memory-constrained fixture: a join larger than memory_limit asserts successful spill rather than OOM.

### PERF-9 · Static-asset cache busting covers only 4 files: 36 ES modules (js/ + js/v2/) have neither a version query param nor Cache-Control — after an upgrade, browsers can keep running old modules indefinitely

**Impact:** Medium · **Effort:** S · **Verification:** — Not independently verified

**Problem.** Change a module file such as js/v2/join_review.js: the version string does not change (it is not in `_STATIC_VERSION_FILES`), the module URL does not change (imports carry no ?v=), and the browser keeps using the old file via heuristic caching — after an upgrade, old and new modules run mixed together, producing "改了没生效" ("my change didn't take effect") and hard-to-reproduce frontend bugs (user memory already records the pain point of "必须 Cmd+Shift+R" — must hard-refresh with Cmd+Shift+R).

**Evidence.** marvis/app.py:73-78 `_STATIC_VERSION_FILES = ("app.js","styles.css","css/welcome.css","css/v2-workbench.css")`; index.html:1425 only app.js carries ?v=; the top of app.js has 30+ `import ... from "./js/*.js"` statements (no version query string); marvis/app.py:226 the StaticFiles mount sets no Cache-Control header (repo-wide grep for 'Cache-Control' = 0 hits).

**Why it matters.** This directly affects development iteration speed and the user's first impression after every release ("wasn't this bug supposed to be fixed?"); the fix cost is extremely low.

**How to fix.**
1. Option A (simplest, recommended): add a response middleware in app.py setting `Cache-Control: no-cache` on /static — ETag/Last-Modified are already provided by StaticFiles, the per-request 304 revalidation cost on localhost is negligible, and staleness is eliminated entirely.
2. Option B: change `_STATIC_VERSION_FILES` to rglob over static/js/**/*.js + css taking max(mtime), and inject a `<script type="importmap">` into index.html mapping ./js/*.js to ?v=-suffixed URLs (module imports honor the importmap).
3. Option A is a one-line middleware and can land first.

### PERF-10 · The multi-recipe training loop deep-copies the entire modeling frame once per recipe, and every recipe reads all columns — memory peaks multiply on wide tables with large samples

**Impact:** Medium · **Effort:** S · **Verification:** — Not independently verified

**Problem.** A "multi-recipe comparison training" run (`train_models` is the product's core comparison workflow) on a 5M-row × several-hundred-column modeling frame: 1 cached copy + 1 full-frame copy per recipe. Although `prepare_modeling_frame` has already pruned columns, passthrough/weight/date columns can still leave the frame very wide; 4 recipes means 4 GB-scale deep copies with stacked peaks (each recipe's split slices again on top). This is also a common source of "inexplicable resource errors" under the tool worker's RLIMIT_DATA.

**Evidence.** marvis/packs/modeling/training_dataset.py:45-50: when columns=None, `frame = self._dataset.frame` → `return frame.copy()` (full-frame deep copy); no recipe passes columns: lgb.py:26 / xgb.py:26 / lr.py:25 / scorecard.py:38 / catboost.py:25 / mlp.py:30 etc. `frame = backend.read_frame(dataset_path)`; tools.py:697 the `train_models` loop chains multiple recipes through the TrainingDataset adapter.

**Why it matters.** Modeling is the platform's core objective (passing the KS benchmarks presupposes training completes successfully); the memory peak determines the sample-size ceiling a single machine can handle, and the wasted copies directly slow down multi-recipe comparison wall time.

**How to fix.**
1. Change each recipe to read only the needed columns: `needed = _unique([*config.features, config.target_col, config.split_col, config.sample_weight_col, <date/monotone-related columns>])` (all obtainable from TrainConfig), then `backend.read_frame(dataset_path, columns=needed)` — on the adapter path, the copy then only duplicates the required columns (training_dataset.py:46-47 already supports column slicing).
2. The three splits returned by `split_modeling_frame` are already new objects; after review confirms this, the adapter's `.copy()` can be downgraded to a copy of the config's feature subset.
3. Add a memory-profiler slow test: 2M rows × 300 columns, 4-recipe `train_models`, asserting peak RSS < ~2.5× the size of a single frame.

---

## 6. Product UX & Interaction Flow

**Lens verdict.** The skeleton capabilities of the V2 frontend — structured gate widgets, failed-step retry with parameter editing, incremental message polling, and a plan rail that reuses the validation stepper — are all in place, and the polish of the legacy validation-task mainline is solid. But the product-experience center of gravity has not caught up with the fact that driver-type tasks are now the primary journey. The three heaviest gaps: first, after a manual-mode confirmation gate is clicked, the backend synchronously runs the entire long turn (including hyperparameter tuning + training) with zero frontend feedback and no way to stop — the core modeling journey becomes a black-box wait. Second, the agent-mode chat timeline discards all of the existing structured widgets (screening table / dedup picker / modeling settings), forcing every adjustment through the weak-LLM free-text routing path — landing squarely on the platform's weakest link. Third, with multiple tasks running in parallel, gate-widget callbacks have no task guard, producing a cross-task conversation-bleed display bug. Below those, execution observability (replan / no_progress / sub-agents / step timing) is entirely invisible in the live UI, while roughly 2,200 lines of v2 workbench modules are a dead-code dual stack referenced only by tests — improvements easily land where users can never see them. Further down sits a batch of S/M-sized decision-support and copy-precision issues: the screening gate table cannot handle real feature scale, the dedup gate lacks decision material, a duplicate C1 anchor table is silently dropped, "计划生成中…" ("plan being generated…") misattributes who is waiting on whom, punctuation conventions are mixed on-screen, and there is no sample data for a first run.

### UX-1 · After clicking a manual-mode confirmation gate, the long task runs with zero feedback: no busy state, no message polling, no plan-rail refresh, and no way to stop

**Impact:** Critical · **Effort:** M · **Verification:** ✅ Confirmed adversarially

**Problem.** In manual mode for driver-type tasks (JOIN / feature / modeling / strategy / vintage), after the user clicks "确认" ("confirm") in the plan rail or a gate widget, the backend synchronously executes every step up to the next gate — in the modeling scenario this includes tune_hyperparameters (n_trials can reach 200) plus train_model, potentially minutes to tens of minutes. During that time the only visible frontend change is the button turning gray: the task hero status bar never enters busy, no new content appears in the center analysis area, the right-hand plan rail does not refresh (steps never show a running ring, because maybeFetchPlan fires only on render events and none occur during the wait), there is no elapsed/progress indicator, and there is no stop button (the composer is hidden in manual mode, app.js:4656). The user's only reasonable inference is "it's frozen", followed by a page refresh or repeated clicks.

**Evidence.** marvis/static/js/v2/driver_gate_confirm.js:35-53 (submitDriverConfirm only sets button.disabled=true then awaits the api; the success path calls neither setActionStatus nor any message polling); the same pattern appears in join_gate_controller.js:91-105, screen_gate_controller.js:120-136/160-176, and modeling_setup_panel.js:214-236. On the backend, marvis/api.py:624-654 `_dispatch_driver_turn` runs the entire turn synchronously inside the HTTP request and never calls `_start_task_job` (grep confirms start_task_job is used only for report/agent/join/notebook/metrics/pipeline), so active_job_kind stays empty → app.js:361-369 taskServerBusyAction returns null → app.js:2371-2376 ensureActiveTaskProgressPolling never claims the per-second polling. app.js:6341-6385: the only global setInterval is a once-per-minute greeting timer; there is no fallback render tick.

**Why it matters.** This hits the product's core journey — modeling is MARVIS's long-term goal (polishing the model-development agent), and manual mode is the control group that proves the agent conversation is not pre-scripted. Psychological management of long tasks is the baseline of waiting UX: a black-box wait beyond 30 seconds triggers hard refreshes or force-kills. A refresh does not actually lose backend progress (the turn keeps running server-side), but the user does not know that, and trust drops straight to zero.

**How to fix.**
1. S: Add "feedback on dispatch" uniformly to the five submit* controllers — on click, immediately call setActionStatus('正在执行下一步…','busy') ("executing the next step…"), and reuse the existing pollAgentMessagesUntilSettled(taskId, requestPromise) in app.js (loadAgentMessages already works for plan-rail tasks; see app.js:5437 where hasConversation includes taskUsesPlanRail), so intermediate step messages appear as the turn runs; during the pending window also call planRailController.resetFetchThrottle + renderWorkflowStepper({force:true}) every 1-2s so the plan rail shows the running ring.
2. M: On the backend, wrap the driver turn in `_start_task_job(repo, task.id, 'driver')` (cleanup in finally), and add a `kind==='driver'` branch to the frontend's taskServerBusyAction, so busy is visible after a refresh or from other entry points and hooks into the global polling and stop semantics.
3. M: Add a started_at field to plan steps in the plan payload, and have planSubstepGroupHtml reuse the validation rail's formatStepElapsed elapsed-time display for running steps.

**Verification note.** The adversarial pass confirmed the entire no-fallback chain and added routing evidence: marvis/routers/validation_agent.py:106-119 dispatches driver task types synchronously inside the HTTP request and marvis/agent/turn_handlers.py:436-449 executes DRIVER_TURN_FUNCS with no background job, while a repo-wide grep found _start_task_job used only at api.py:214 (report) and api.py:672 (agent validation job). The only drift was minor: the lone global setInterval sits at app.js:6375, inside the cited range.

### UX-2 · Agent-mode chat timeline discards all structured gate widgets (screening table / dedup picker / modeling setup panel), forcing every adjustment through the weak-LLM free-text routing path

**Impact:** High · **Effort:** M · **Verification:** ✅ Confirmed adversarially

**Problem.** For the same screen gate, a manual-mode user can directly tick features, drag thresholds, and click "重算" ("recompute"); an agent-mode user sees only a flat read-only table and must type into the composer — "把 xxx 加回来、阈值改 0.35" ("add xxx back, change the threshold to 0.35") — hoping the local 32-72B model routes the free text correctly into adjust_params. The platform's established weakness is precisely the weak model's instruction understanding, yet the structured decisions best suited to direct manipulation (checkboxes / thresholds / single-choice) are pushed into the text channel; the same applies to dedup strategy and modeling settings (algorithm / tuning rounds / weight column). The data (metadata) and the submission endpoints already exist — it is purely the frontend render branch that drops the widgets.

**Evidence.** marvis/static/app.js:5367-5370 (agentMessageHtml renders only the join_c1 form or modelDelivery+tables for assistant messages, and renderDriverGateButton returns empty in agent mode, driver_gate_confirm.js:13); renderScreenGateTable / renderDedupPicker / renderModelingSetupPanel are injected only by driverManualAnalysisHtml (app.js:5006-5018), which runs only in the `showConversation && !isAgent` branch (app.js:4670). On the backend, gate messages carry the full metadata.screen/dedup/modeling_setup in both modes (marvis/agent/plan_message_composer.py:65-72), and the structured submission path for selection/adjust_params/expected_step_id is mode-independent (marvis/routers/validation_agent.py:106-119; gate_response_adapter.py:37-47 has stale-409 protection).

**Why it matters.** This directly determines whether the agent feels smarter and more reliable: structured widgets bypass LLM paraphrasing — one click means deterministic effect — and save an LLM round-trip (local model inference is slow). Cognitive load at the confirmation gate is also lower: the user knows the object of confirmation is exactly the checkbox state on screen, not "whatever the model understood from what I said".

**How to fix.**
1. In agentMessageHtml, for messages with metadata.kind==='gate' that carry screen/dedup/modeling_setup, reuse the existing panels: `if (meta.screen) html += agentMessageModelingSetupHtml(msg,{interactive}) + agentMessageScreenTableHtml(msg,{interactive})`, with the interactive condition reusing latestInteractiveScreenMessageId (driver_manual_analysis.js:15-23) plus "no busy currently active"; widget submissions keep the existing controllers (they already carry acceptance_mode and expected_step_id, with server-side 409 stale protection).
2. Keep free text in the chat as a supplementary channel, and note in the gate message copy: "可直接操作下方控件，或用文字说明" ("you can operate the widgets below directly, or explain in text").
3. Regression point: the agent streaming fast-path's structuralSignature must incorporate the interactive state of the screen/dedup/modeling_setup panels into the signature (app.js:4728-4768), so panels do not get stuck in a stale interactive state.

**Verification note.** The verifier confirmed the gap and noted the omission is partly deliberate — plan_rail_controller.js:327-331 comments that agent mode shows a read-only "待确认" ("pending confirmation") badge because the LLM operates the gate — yet the join_c1 form does remain interactive inside agent chat (app.js:5368), proving inline interactive widgets are architecturally feasible, and the gap contradicts the app.js:4650-4652 comment that both modes show the same conversation + controls.

### UX-3 · Cross-task message bleed: gate-widget callbacks to setAgentMessages have no "task unchanged" guard, so a finishing long turn pours task A's conversation into task B's panel

**Impact:** High · **Effort:** S · **Verification:** ✅ Confirmed adversarially

**Problem.** A driver turn is a synchronous long request (UX-1), and switching to task B while waiting is natural behavior. When task A's confirmation request then resolves, A's full messages array overwrites the global agentMessages, and renderAgentConversation renders A's conversation and gate tables into the conversation panel of the currently selected task B (if B has no polling running, the wrong state persists until the user's next action). The expected_step_id 409 protection blocks mis-execution, but the display-layer bleed is enough to destroy the user's confidence about "which task's results am I looking at" when running tasks in parallel.

**Evidence.** marvis/static/app.js:5329-5331 (driverConfirmControllerContext's setAgentMessages assigns `agentMessages = messages || agentMessages` directly, with no selectedTaskId comparison); the same unguarded wrapper also appears at app.js:5216-5227 (modelingSetup), 5244-5259 (joinGate), and 5280-5296 (screenGate). Control case: loadAgentMessages has an `if (selectedTaskId !== taskId) return;` guard at app.js:5447. Trigger chain: join_gate_controller.js:100-101 and the other controllers unconditionally call setAgentMessages + renderAgentConversation after the POST resolves.

**Why it matters.** Parallel multi-tasking is the daily shape of a single-machine product (modeling running, a feature analysis open next to it); display correctness on context switches is the foundation of the parallel experience — one bleed and the user never dares switch away during a running task again, which effectively disables parallelism.

**How to fix.**
1. In the four controllerContext factories, capture taskId at click time and validate inside the setter: `setAgentMessages: (messages) => { if (selectedTaskId !== capturedTaskId) return; agentMessages = messages || agentMessages; }`.
2. Or, more thoroughly, have the controllers' success paths uniformly call `loadAgentMessages(capturedTaskId)` (which has the guard and incremental merge built in) and discard result.messages.
3. Add a unit test: switch selectedTaskId during a pending confirm, then assert agentMessages is not overwritten.

**Verification note.** The verifier pinned the exact setter locations (app.js:5222-5224, 5250-5252, 5286-5288 — the cited ranges included the surrounding addEventListener blocks) and confirmed there is no fallback: the only self-healing path is the per-second pollValidationProgress refresh (app.js:5739), which fires only when task B has an active poll and is in agent mode.

### UX-4 · The feature screening gate table is unusable at real credit-data scale: hundreds of rows ticked one by one, with no search / sort / per-category summary / bulk actions

**Impact:** High · **Effort:** M · **Verification:** ✅ Confirmed adversarially

**Problem.** After a credit JOIN, candidate features commonly number in the hundreds to thousands of columns. The current screen gate flattens selected + leakage + suspected into one long table with one checkbox per row: removing 20 suspected columns means eyeballing hundreds of rows; verifying by IV in descending order is impossible (row order is fixed as selected→leakage→suspected); "show only selected columns with missing rate >50%" has no tooling at all. This is the gate with the heaviest cognitive load in the entire modeling journey and the biggest influence on model quality, and the current interaction only supports toy data of a few dozen columns.

**Evidence.** marvis/static/js/v2/screen_gate_controller.js:45-48 (selected/leakage/suspected render one row per feature with no cap; only unusable is truncated to 50); the backend payload is passed through in full with no upper bound (marvis/agent/gate_payloads.py:75-79). The table has no filter input, no column sorting, no per-category select-all/clear, and no per-category count summary (screen_gate_controller.js:60-72 contains only a single note line and one confirm button).

**Why it matters.** The screening gate is the user's core control point over model quality (it decides which features enter the model), and it is where the "human confirmation" value of the INV system lives — if the table cannot be reviewed at real scale, users will simply click confirm blindly, the gate degenerates into a rubber stamp, and the model-quality goal (high KS with no leakage) loses its human line of defense.

**How to fix.**
1. Add per-category summary chips to the table header: "入选 N / 泄漏 N / 疑似 N / 不可用 N" ("selected N / leakage N / suspected N / unusable N"); clicking a chip filters to that category.
2. Add a text filter box (feature-name substring match) plus click-to-sort on the KS / IV / missing-rate column headers.
3. Provide per-category select-all/clear buttons (bulk-selecting the leakage category pops a one-time confirmation explaining the consequences).
4. Render only the first 100 rows initially plus a "展示全部 N 行" ("show all N rows") control, or use simple virtual scrolling.
5. Add position:sticky to the thead.

All of this is frontend-only; the payload is unchanged and the confirm submission logic (the selection array) is unaffected.

**Verification note.** The verifier additionally established that no upstream cap exists — top_k defaults to None in marvis/feature/screen.py:79,130 (None means selected = all clean features) and marvis/packs/feature/tools.py:135-145 truncates only when a caller passes top_k explicitly — and that the sole mitigation is a max-height:360px scroll area (v2-workbench.css:1775-1776), which does not solve findability.

### UX-5 · replan / no_progress / exploration branches and sub-agent activity are completely invisible in the live UI: the plan rail ignores loop_events and sub_agents, and the corresponding frontend modules are not wired up

**Impact:** Medium · **Effort:** M · **Verification:** — Not independently verified

**Problem.** In the weak-model setting, replan and no_progress are routine events: a failed LLM gate judgment triggers a replan, and consecutive lack of progress accumulates no_progress. These events are fully persisted and delivered via /api/tasks/{id}/plans, yet the live plan rail shows not a word of them — all the user sees is step rings sitting still or a few steps suddenly appearing, with no way to learn "the agent just replanned, because X" or "there has been no progress for three rounds". Sub-agent spawning and reclamation is equally invisible. An earlier report raised "loop progress upgrades", but it assumed loop_progress.js was live; in fact it was never mounted by the app — a more fundamental disconnection.

**Evidence.** marvis/static/js/v2/plan_rail_controller.js:453-517 (planRailHtml renders only fetchErrorBanner+phases+startControl and never reads plan.loop_events / plan.sub_agents / plan.replan_count); the backend payload explicitly includes these fields (marvis/orchestrator/contracts.py:166 loop_events; marvis/routers/plans.py:254-257 sub_agents). loop_progress.js and subagent_view.js have zero references inside marvis/static other than themselves (a repo-level grep hits only tests/).

**Why it matters.** Explainability of the execution process directly determines the user's trust in the agent and their timing for intervention: without seeing the replan reason, the user cannot judge whether to wait or change instructions; without no_progress surfacing, the user only discovers idle spinning at timeout, wasting local inference compute and time.

**How to fix.**
1. Render a compact event strip at the top of planRailHtml: take the last 3 entries of plan.loop_events, displayed as `重新规划：<reason>` ("replanned: <reason>") / `暂无进展：<reason>` ("no progress: <reason>"); style no_progress in the attention color and attach a "发消息介入" ("send a message to intervene") shortcut button (sendPrompt into the composer).
2. When plan.replan_count>0, add an `已重规划 N 次` ("replanned N times") badge next to the plan title.
3. When sub_agents is non-empty, render "子任务代理 · scope · 状态" ("sub-task agent · scope · status") rows in the row style of subagent_view.js — or, if the decision is not to build this, delete loop_progress.js/subagent_view.js (see UX-8). The data is already in v2PlanCache; this is a pure rendering change.

### UX-6 · The dedup confirmation gate lacks decision material: only two semantically opaque options (first/last); the backend-computed conflict_columns is dropped by the frontend; no conflict samples, no ordering-basis explanation, no "exclude this table" exit

**Impact:** Medium · **Effort:** M · **Verification:** — Not independently verified

**Problem.** This is the confirmation where the user carries the heaviest responsibility in the INV-3 context: choosing first versus last decides which feature record enters the training sample. But "first record / last record" follows file row order — a black box to the user and business-meaningless (it is not application time or update time); the user can see neither which columns are in conflict nor what the conflicting values look like, and there is no "this table's data is too dirty, exclude it for now" exit. With the current information, the user can only pick one of the two at random — the gate assigns responsibility without providing the material for judgment.

**Evidence.** marvis/agent/gate_payloads.py:33 (strategies hard-coded to ["first","last"]); the payload carries conflict_columns per feature (gate_payloads.py:26-27), but the frontend picker shows only the conflict_keys number (marvis/static/js/v2/join_gate_controller.js:125 `${feature.conflict_keys} 个同键冲突` ("N same-key conflicts")), and conflict_columns is never rendered; the option labels are just "保留首条 (first)/保留末条 (last)" ("keep first record / keep last record") (join_gate_controller.js:3). The backend also has sample_keys conflict examples (marvis/data/backend.py:147-154) that are not surfaced to the gate. At the copy layer, renderers.py:688-689 suggests "或排除这些特征后重试" ("or exclude these features and retry") but the picker has no corresponding control.

**Why it matters.** A wrong dedup strategy means systematic bias in feature values (e.g., always picking the stale credit-bureau record), and it is extremely hard to discover afterwards — no metric errors out, just a slightly lower KS or unstable OOT. This is one of the cheapest wins for "more accurate results": the data is already computed; only the last mile of presentation is missing.

**How to fix.**
1. S: Render conflict_columns — add `冲突列：col_a、col_b…` ("conflicting columns: col_a, col_b…") to each row (collapse beyond 5 columns).
2. S: Put the conflict samples already produced by the propose step (sample_keys / two-row comparison) into the picker as an expandable area, so the user can see which row first and last would each keep.
3. M: Add candidate time columns to the backend dedup payload (reusing the date-semantic column detection from the join diagnostics) and offer a third option "按 <time_col> 取最新" ("take the latest by <time_col>"), implemented as order-by dedup in the engine (more business-meaningful than a bare last).
4. S: Add an "排除该特征表" ("exclude this feature table") button whose submission is equivalent to the existing instruction that removes the table from the join.

### UX-7 · The C1 role confirmation form silently drops a second "sample anchor table": no error, not treated as a feature table — the dataset simply vanishes from the join

**Impact:** Medium · **Effort:** S · **Verification:** — Not independently verified

**Problem.** C1 is the first confirmation gate of the JOIN journey; the form has one dropdown per row (sample anchor table / feature table / ignore). When the user mistakenly marks two tables as the anchor (easy to do in multi-file scenarios), the form neither validates nor warns, and the second table is silently treated as "ignore". The user believes they registered two tables, but only one enters the downstream flow — discoverable at best when a table is missing from the diagnostics, and by then the apparent story is "the agent lost my table".

**Evidence.** marvis/static/js/v2/join_gate_controller.js:81-85: `if (select.value === "anchor" && !anchorId) anchorId = datasetId; else if (select.value === "feature") featureIds.push(datasetId);` — the second and later datasets marked anchor go into neither anchorId nor featureIds and are entirely absent from the submitted [C1] payload; submission succeeds with no warning of any kind (join_gate_controller.js:92-101).

**Why it matters.** This violates the product spirit of "JOIN never silent" (albeit not at the engine layer): the user's explicit input is silently rewritten. The damage to trust outweighs the implementation cost — this is a few lines of validation code.

**How to fix.**
1. Add validation after the collection loop: count entries with value==='anchor'; if >1, call setActionStatus('只能有一张样本主表，请把其余表改为「特征表」或「忽略」。','error') ("only one sample anchor table is allowed; change the others to 'feature table' or 'ignore'") and return.
2. In addition, on the change event when the user switches a second dropdown to anchor, immediately flip the previous one back to feature (surfacing the single-select semantics in the UI) — or turn "样本主表" ("sample anchor table") into a radio group on the table column, eliminating the ambiguity at the root.

### UX-8 · 8 v2 frontend modules (~2,200 lines) are dead code in production, referenced only by tests — a dual UI stack where UX improvements land where users can never see them

**Impact:** Medium · **Effort:** M · **Verification:** — Not independently verified

**Problem.** The V2 workbench module group and the actually-mounted conversational UI are two parallel implementations, and the dead-code stack is in places more capable: join_review.js has an abort dedup option and an execution entry point, loop_progress.js has event-stream rendering, subagent_view.js has sub-agent cards. Green tests create the illusion that "the feature already exists" (a sizable share of the 1,750+ test cases exercise UI nobody uses), and future improvements (such as the loop events in UX-5) can easily land in the dead stack again.

**Evidence.** Repo-level grep: join_review.js (406 lines), plan_view.js (401), plan_confirm.js (257), workflow_create.js (201), subagent_view.js (71), loop_progress.js (75), memory_manager.js (273), draft_manager.js (526) have zero imports inside marvis/static other than themselves/each other (the app.js import list at app.js:1-120 does not include them); the only referencers are tests/test_frontend_v2_join.py, test_frontend_v2_plan.py, test_frontend_v2_loop.py, test_frontend_v2_memory.py, test_frontend_v2_subagent.py, and the like. The live equivalents are join_gate_controller / plan_rail_controller / agent-memory-panel / draft-tools-panel.

**Why it matters.** For a solo-maintained 60k-line project, a dual stack is a continuous decision tax: every frontend change must first answer "which stack do I change"; test maintenance cost is spent on code users cannot touch; and product reviews (including this one) must first disambiguate the stacks before discussing experience.

**How to fix.**
1. For modules with incremental value (loop_progress, subagent_view): wire them into plan_rail_controller per UX-5, then delete the standalone modules.
2. For superseded modules (join_review, plan_view, plan_confirm, workflow_create, memory_manager, draft_manager): delete them together with their dedicated tests, after first porting their leading interaction details (join_review's abort option) into the live stack.
3. Declare in a static/js/v2/README or module header comments that "every module in this directory must be imported by app.js directly or indirectly", and add a CI check (a script that greps the import graph) to prevent regrowth.

### UX-9 · Zero first-run experience: no sample data, no one-click trial entry — any trial or demo requires preparing a data directory first

**Impact:** Medium · **Effort:** M · **Verification:** — Not independently verified

**Problem.** Step 0 of the user journey ("let me first see how this thing works") has no entry point: to experience the JOIN confirmation gate or the modeling flow, one must first prepare a compliant dataset with anchor table / feature tables / target column / time split and enter an absolute path. Internal demos (showing leadership/colleagues the confirmation gates and scorecard delivery) equally require scrounging for data. Project memory already holds benchmark datasets with ground truth, but they have not been productized into a built-in sample.

**Evidence.** In marvis/static/index.html, app.js, and js/*.js, grep for '示例|样例数据|demo|新手|引导' ("sample | sample data | demo | newcomer | onboarding") yields zero hits (only the placeholder '例如' ("for example")); the create-task dialog mandates source_dir or manual file upload (create-task-dialog.js:318-341); the six task cards on the welcome page (index.html:935-1129) lead straight to that form with no "try with sample data" bypass. There is also no demo/sample data generation script under scripts/.

**Why it matters.** The differentiating capabilities — confirmation gates, plan rail, scorecard delivery — can only be perceived when running; without sample data, the product's "first-impression cost" is measured in hours rather than minutes. For the developer too, sample data is the fastest reproduction vehicle for UI regressions.

**How to fix.**
1. Write a deterministic synthetic-data generator (fixed seed; a 5k-row anchor table + 2 feature tables; deliberately planting one case-mismatched key, one same-key conflict, one leakage feature, and train/test/oot split columns), shipped in the repo or generated into workspace/samples/ on first launch.
2. Add a secondary link "用示例数据创建" ("create with sample data") to each task card on the welcome page; clicking it creates the task directly with a pre-filled source_dir and pre-filled task name.
3. Have the sample task's first assistant message include "这是内置示例数据，包含 1 个键冲突和 1 个泄漏特征，试试在确认门里找到它们" ("this is built-in sample data containing 1 key conflict and 1 leakage feature — try to find them at the confirmation gates") — turning the sample into confirmation-gate teaching. No analytics needed: single-machine product; the success criterion is walking JOIN→screening→training within a 5-minute demo.

### UX-10 · Waiting/confirmation micro-copy is inaccurate: the plan rail shows "计划生成中…" ("plan being generated…") while actually waiting for the user; confirmation buttons are all a bare "确认" ("confirm") with no consequences stated

**Impact:** Medium · **Effort:** S · **Verification:** — Not independently verified

**Problem.** Three small pieces of copy jointly cause "system-state misattribution": (a) at the C1 gate stage the plan rail says "生成中…" ("generating…"), so the user thinks they are waiting on the system while the system is actually waiting on them — a two-way deadlock of waiting; (b) gate buttons state no consequences, which bears directly on "does the user know what they are confirming" — the execute_join gate's confirmation genuinely writes artifacts to disk, yet the button looks identical to a read-only screening gate's; (c) the phrase "自动审查" ("automatic review") makes it entirely invisible that a weak model will confirm a destructive JOIN on the user's behalf.

**Evidence.** marvis/static/js/v2/plan_rail_controller.js:470-472: whenever there is no plan and no blocking error, it unconditionally returns '计划生成中…' ("plan being generated…") — yet data_join/modeling manual mode has no plan at the C1 role gate stage by design (the plan is built only after C1 confirmation, marvis/agent/turn_handlers.py:96), so at that moment the system is in fact waiting for the user. Confirmation buttons: driver_gate_confirm.js:19 and plan_rail_controller.js:330/512 are both a bare "确认/开始执行" ("confirm / start execution"), with no distinction between "confirm and then execute the left join" and "confirm the screening results"; the acceptance dropdown "默认权限/自动审查" ("default permissions / automatic review") (index.html:1199-1200) has no title/tooltip explaining that auto mode passes every confirmation gate on the user's behalf.

**Why it matters.** Confirmation gates are the user-facing surface of the platform's safety model, and copy precision equals risk-perception precision. These changes are all S-sized, yet they directly raise the sense of professionalism and the gravity of the gates.

**How to fix.**
1. Add one more state to planRailHtml's empty-state logic: when the latest assistant message has metadata.kind==='gate' (getAgentMessages is already injected into the controller), show "等待你在左侧确认后生成计划" ("waiting for your confirmation on the left before generating the plan").
2. Map gate button copy from the gate step's tool_ref: execute_join→"确认并执行拼接" ("confirm and execute the join"), screen_features→"确认所选特征" ("confirm the selected features"), before train→"确认并开始训练" ("confirm and start training"); put the mapping table in driver_gate_confirm.js (metadata.step_id can resolve the tool via planRailController.planStep).
3. Add a title to the acceptance chip: "自动模式下 Agent 将替你确认全部关键节点（含拼接执行与训练）" ("in auto mode the Agent will confirm all critical checkpoints on your behalf, including join execution and training"), and fire setAgentComposerNotice once when switching to auto_accept.

### UX-11 · Inconsistent Chinese punctuation conventions: v2 modules and backend gate copy use half-width commas/semicolons while the main app uses full-width, mixed on the same screen

**Impact:** Low · **Effort:** S · **Verification:** — Not independently verified

**Problem.** The newly added V2 modules (gate widgets, modeling panel) and backend gate copy systematically use half-width commas/semicolons/parentheses, while the V1 mainline copy uses full-width. In a Chinese enterprise product this is the "unprofessional" signal business stakeholders notice most readily, and as V2 panels become the primary interface, the mixed area keeps growing.

**Evidence.** Half-width: screen_gate_controller.js:53 "共筛 N 列;泄漏阈值…勾选=入选,可硬选…" ("screened N columns total; leakage threshold… checked = selected, can hard-select…"), modeling_setup_panel.js:184 "这是历史建模规格,请使用最新待确认步骤调整。" ("this is a historical modeling spec, please adjust via the latest pending-confirmation step."), join_gate_controller.js:140 "拼接键不唯一(同键多行),请选择…" ("join key not unique (multiple rows per key), please choose…"), and backend plan_message_composer.py:55 "确认请回复「确认」继续;要调整可直接说明。" ("to confirm, reply '确认' to continue; to adjust, just say so."). Full-width: app.js:203 "这个入口会继续展示在任务启动页，但当前不会打开创建弹窗。" ("this entry will remain on the task launch page, but currently does not open the creation dialog."), app.js:5679 "Agent 任务已创建，等待你的下一条指令。" ("Agent task created, awaiting your next instruction."). The two conventions appear on the same screen simultaneously (hero status bar full-width + gate panel half-width).

**Why it matters.** Copy consistency is a low-cost, high-perception professionalism item; the target users of a credit-risk tool (risk/validation staff) are sensitive to documentation conventions.

**How to fix.**
1. One-off regex cleanup: in user-visible strings under static/js and marvis/agent, replace `,`→`，`, `;`→`；`, `(`/`)`→`（`/`）` (Chinese context only; code literals/JSON keys excluded).
2. Write a copy rule into DESIGN.md: full-width punctuation within Chinese sentences; half-width space between numbers and units.
3. Add a lightweight lint script (grep for the pattern of a Chinese character immediately adjacent to a half-width comma) hooked into pre-commit to prevent regression.

### UX-12 · Material upload has no progress feedback: large-file uploads show only a static "正在上传材料..." ("uploading materials...")

**Impact:** Low · **Effort:** S · **Verification:** — Not independently verified

**Problem.** Credit sample / bureau feature tables are often CSV/parquet files of hundreds of MB; even on localhost, the browser→FastAPI multipart transfer plus write-to-disk takes seconds to tens of seconds. The user is left staring at a single static line of text inside a modal dialog, with no way to tell which file is being transferred or how much remains, and no way to cancel.

**Evidence.** marvis/static/js/create-task-dialog.js:331-333: setCreateStatus('正在上传材料...') ("uploading materials...") followed by await uploadMaterialFiles → api() (js/api.js:64-90, fetch-based; fetch cannot produce upload progress events); during this there is no percentage, no per-file completion count, and no cancel button.

**Why it matters.** This is the first real waiting point in the new-task journey, occurring in the window where the user forms their first impression of the product.

**How to fix.**
1. Replace uploadMaterialFiles with XMLHttpRequest (onprogress provides loaded/total), or upload files sequentially and update the status as "正在上传 2/5：features.csv (37%)" ("uploading 2/5: features.csv (37%)").
2. Add a cancel button to the dialog (xhr.abort()).
3. After completion, briefly show a per-file size list before proceeding to creation. Roughly 60-80 lines of frontend change; the backend /api/material-uploads endpoint needs no changes.

---

## 7. Visual Design & Brand

**Lens verdict.** After 166 commits, the visual side has clearly improved: color tokens are systematized (~190 tokens including full dark-mode coverage, with metric/chart/model-panel domain-scoped naming), the validation task's metric preview now speaks a real chart language (ROC/KS SVG curves, KPI cards, databars, PSI three-segment color bars, sparklines, tooltips, 16 uses of tabular-nums), focus rings are tokenized, breakpoints have converged to 8, and the glass + tone-glow facade of welcome/task-hero is mature. But one clear main fault line remains: this visualization language has not trickled down at all into the "working area" of the V2 driven flow — gate data tables are bare tables rendered cell-by-cell with escapeHtml, confirmation-gate messages are indistinguishable from ordinary chat bubbles, the feature screening table shows KS/IV as plain decimals, and professional deliverables such as the calibration reliability curve have their data produced but no chart. Meanwhile skeleton loaders are still at zero, dark-mode glass texture is not on par, and the mascot has zero linkage with V2 events (the newly produced true-logo glow animation assets are still sitting in scripts/, unwired). Also discovered: a batch of V2 view modules with zero runtime mounts (join_review/plan_view/loop_progress/subagent_view) carrying a dozen-plus orphan class names with no CSS, kept alive only by node tests — past criticisms of "flat evidence panels" partly landed on this dead code. Suggested priorities: first do gate-table rich rendering and the gate-card visual form (dual business + safety payoff), then the visible quick wins — skeletons, calibration charts, bar entrance animations; token spacing/type consolidation is long-term work, and any radius adjustment must first go through comparison mockups signed off by the user.

### VD-1 · Gate data tables in driver-mode tasks are "bare tables": the metric visualization language already built on the validation side has not been brought down to the V2 decision evidence panels

**Impact:** High · **Effort:** M · **Verification:** ⚠️ Partially confirmed

**Problem.** JOIN match rates, C2 gate diagnostics, feature statistics, model comparisons — the tables where V2 most critically requires "a human making a decision at the gate" — are all left-aligned plain text; users must read 4-decimal numbers row by row to compare tiers. Yet within the same product, the validation metric preview has databars, heat shading, PSI color bars, and ranking tooltips. This is currently the biggest "polished facade, crude working area" fault line, and the repair materials already live in the same file.

**Evidence.** app.js:5174-5198 `agentMessageTablesHtml` renders every cell of the plan driver's metadata.tables as `<td>${escapeHtml(String(cell ?? ""))}</td>` (L5188) — no right-aligned numbers, no databar, no PSI/KS tiering, no tooltips. Meanwhile the same file already contains a full rich-cell renderer suite at app.js:3485-3566 (databar / percent-heat / PSI three-segment color bar / the `metricHeaderShouldRightAlign` right-align whitelist), but it only serves the validation task's metric preview path. Markdown tables in agent replies are equally plain text (styles.css:5871-5900 — no tabular-nums / right alignment).

**Why it matters.** Confirmation decisions at gates (INV-3, leakage screening, model selection) depend on the user quickly scanning numeric tiers; plain-text tables increase the risk of misreading and drag down the professional feel of the entire agent flow. In weak-model scenarios, the more users rely on reading the tables themselves, the higher the cost of the missing visualization.

**How to fix.**

1. Extract renderCellByKind/parseNumeric/columnFractions/psiTier out of app.js into a shared module (render-metrics.js already holds half of it).
2. In agentMessageTablesHtml, infer each column's kind from its header name: `匹配率/命中率/缺失率/占比` (match rate / hit rate / missing rate / share) → databar + percentage; `KS/AUC/IV` → databar + right-aligned; `PSI` → PSI three-segment bar; `行数/样本量` (row count / sample size) → thousands separators + tabular-nums right-aligned (reusing the metricHeaderShouldRightAlign regex is sufficient).
3. Backend driver tables already carry a title; optionally add explicit column_specs to the tables payload (aligned with validation's metadata.sections) — the frontend reads specs first and falls back to header-name inference when no specs are present.
4. Add `td.cell-number{text-align:right;font-variant-numeric:tabular-nums}` to `.agent-inline-table` and `.agent-markdown table`. Uphold INV-1: presentation only — do not change any numeric computation.

**Verification note.** The adversarial pass confirmed the core mechanism — bare cells at marvis/static/app.js:5188 (with a comment at L5169-5173 explicitly noting the validation path is separate), rich renderers (app.js:3502-3549, 3559-3566) reachable only via the validation metricPreview chain, and backend marvis/agent/renderers.py producing genuinely decision-critical tables (propose_join L588-673, execute_join L694+, compare/feature_metrics/tune/train). However, the "no trickle-down at all / all bare tables" framing is overstated: the feature screening gate (marvis/static/js/v2/screen_gate_controller.js:13-73, with `.screen-table td.screen-num` right-aligned + tabular-nums at css/v2-workbench.css:1792-1798) and the model delivery candidate table (model_delivery_panel.js:105-136, `.model-delivery-num` right-aligned + tabular-nums at v2-workbench.css:1656-1659, with signal-colored status chips) already have partial rich rendering. Also, `.agent-inline-table table` already has `font-variant-numeric: tabular-nums` (v2-workbench.css:898-902), though cells remain left-aligned (L904-908). The fault line stands for the generic metadata.tables output — JOIN diagnostics, execute_join contribution tables, compare_experiments/feature_metrics/tune/train — which has no fallback.

### VD-2 · Mandatory confirmation gates have no "gate" visual form in the chat flow: the same bubble as ordinary messages, with just one extra button

**Impact:** High · **Effort:** M · **Verification:** ⚠️ Partially confirmed

**Problem.** The confirmation gate is the single most important interaction moment in the product (the human line of defense for INV-3: destructive JOINs, dedup strategy, feature selection, NaN labels), yet in the chat flow its visual weight is exactly that of a piece of ordinary small talk. When scrolling through a long conversation, users cannot locate at a glance that "the system is blocked waiting for my decision", and the machine red flags (fan-out, low match rate, leakage counts) are just body text.

**Evidence.** app.js:5350-5373 `agentMessageHtml` uses the same `agent-message assistant` class for gate messages and ordinary messages, with no gate-specific class/tone whatsoever; driver_gate_confirm.js:10-22 `renderDriverGateButton` returns only a lone "确认" (Confirm) button in manual mode (and an empty string in agent mode); styles.css:3908-3915 task-hero already has a data-tone glass-glow language and styles.css:4020 already has a pill-pulse animation — neither is used on gates.

**Why it matters.** A gate that gets overlooked = a task silently stalling; a gate confirmed carelessly = a data incident. Making the gate the most prominent object on the page is both a safety design and exactly the kind of "visible change" the user explicitly prefers (glass + tone + animation are all applicable here).

**How to fix.**

1. Wrap gate messages in `.agent-gate-card[data-gate-tone=review|warn]`: a 3px warning-color bar on the left + a tonal glass background of `background: color-mix(in srgb, var(--warning) 8%, var(--surface))` + `backdrop-filter: blur(12px)` (reusing the task-hero material).
2. Put a "⏸ 等待确认" (waiting for confirmation) pill in the header (reusing the pill-pulse breathing animation) + a gate-type badge (JOIN confirmation / feature screening / dedup strategy).
3. Render the machine red flags already present in metadata (fan_out, match_rate, leakage counts) as checklist rows: icon (✓/△/✕) + copy + value, dual-encoded with color and shape.
4. Render the gate-card frame in agent mode as well (buttons still differ by mode), have the mascot switch to its review expression when a gate appears, and highlight the corresponding plan-rail step in the same tone — a consistent cross-zone state language.
5. Degrade the pulse to static under `prefers-reduced-motion`.

**Verification note.** The verifier confirmed the uniform className (app.js:5352) and the empty-string agent-mode button (driver_gate_confirm.js:13), with three corrections. First, the JOIN C1 gate is an exception: in agent mode its bubble renders a dedicated interactive form (app.js:5368 → renderJoinC1Form, join_gate_controller.js:20-71) — the bubble shell class is still identical, but the content is not pure text. Second, the title's "just one extra button" understates it: chat-bubble timelines render only in agent mode (app.js:4649-4675), where the button is always empty — so the manual-mode button never actually appears in the chat flow at all. Third, a partial out-of-flow fallback exists: the plan rail shows an amber 11px "待确认" (awaiting confirmation) pill for awaiting_confirm steps (plan_rail_controller.js:36-37, 327-331; `.plan-step-await` at v2-workbench.css:1029-1038), but it sits outside the chat flow with very small visual weight and gives the gate message itself no visual form — the core finding stands.

### VD-3 · Still no loading skeletons site-wide (zero hits for skeleton/shimmer): during long tasks the evidence area cannot distinguish "loading" from "hung"

**Impact:** High · **Effort:** M · **Verification:** ⚠️ Partially confirmed

**Problem.** The skeleton item from the 6-28 proposal (original report §H, "fill in skeleton/shimmer") has still not landed. JOIN execution, metrics computation, and artifact preview are second-to-minute scale operations; during the wait the container is blank or old content mutates abruptly, and users will misjudge the app as hung and refresh or force-kill it.

**Evidence.** `grep -rn "skeleton\|shimmer" marvis/static/` hits only the message-diff signature variables at app.js:4734/4768 (unrelated to loading UI); styles.css / v2-workbench.css / welcome.css contain no skeleton classes at all. The metric preview, dataset preview, plan rail first load, and artifact panel (artifact_view.js:155-184 — renderArtifact sets innerHTML directly after an async fetch) are all blank before data returns.

**Why it matters.** In a single-machine local-LLM scenario, waiting is the norm, and loading-feedback quality directly shapes the felt "platform credibility"; shimmer is also the lowest-cost item in the user's preferred "keep the animations" direction.

**How to fix.**

1. Add generic utility classes to styles.css: `.skeleton{position:relative;overflow:hidden;background:var(--surface-muted);border-radius:var(--radius-control)} .skeleton::after{content:"";position:absolute;inset:0;background:linear-gradient(90deg,transparent,color-mix(in srgb,var(--text) 6%,transparent),transparent);animation:skeleton-shimmer 1.4s infinite}`, with the animation stopped under `@media (prefers-reduced-motion: reduce)`.
2. Ship skeleton templates for three shapes: table skeleton (header bar + N row bars), KPI card skeleton, message-bubble skeleton.
3. Wire-up points: set `container.innerHTML = tableSkeletonHtml()` before the renderArtifact fetch; render the KPI skeleton when the metric preview enters the computing_metrics state; agent-chat thinking already has dots, no need to duplicate.
4. Use skeletons only on first load / state transitions; keep in-place diffing for polling updates so skeletons do not flash every frame.

**Verification note.** The verifier confirmed the literal grep facts (only app.js:4734/4768, both inside agentStructuralSignature; no skeleton/shimmer CSS anywhere) but refuted the "blank container" claim: all four named surfaces have loading fallbacks. The actually-installed artifact path (plan_rail_controller.js:287-293, renderRightRailArtifact, wired via installArtifactHandlers on document at app.js:5347) writes `<div class="artifact-loading">正在加载输出...</div>` ("loading output...") before awaiting the renderer; the plan rail's first load renders "计划生成中…" ("plan generating...") plus an error state with a retry button (plan_rail_controller.js:465-472); running steps show a rotating spinner (`.check-icon.running` with `animation: step-spin 0.8s linear infinite`, styles.css:4068-4072); task switching shows "正在加载任务内容" ("loading task content"). What remains is a visual-polish gap — text/spinner placeholders instead of skeleton/shimmer — so the severity should be read as substantially lower than stated. Path correction: v2-workbench.css and welcome.css actually live under marvis/static/css/.

### VD-4 · Professional deliverables such as the probability calibration reliability curve and score bands are already produced by the backend, but the frontend renders no chart for them at all

**Impact:** High · **Effort:** M · **Verification:** ✅ Confirmed adversarially

**Problem.** The calibration capability added after 6-28 (the headline professional deliverable of that round) degrades in the UI to a text label. The reliability curve is the core visual evidence for "whether these probabilities can be used for pricing/provisioning", and the monotonicity of bad rates across scorecard score bands likewise exists only as a table. The ROC/KS curve component (renderRocCard — SVG + grid + annotations + tooltip, app.js:3753-3810) has already built all the SVG chart infrastructure, so the extension cost is low.

**Evidence.** marvis/packs/modeling/tools.py:1366, 1404, 1455 produce `reliability_curve` (binned mean prediction vs. actual bad rate) and persist it together with Brier/ECE; marvis/output/model_report.py:67 writes an Excel "概率校准" (probability calibration) sheet. The frontend chart-layout whitelist contains only three kinds — kpi_cards/trend_table/roc_ks_curve (app.js:3568-3581 renderMetricTable switch) — and model_delivery_panel.js:143 shows calibration only as a one-line text chip ("校准: 未校准/需说明" — calibration: uncalibrated / needs explanation).

**Why it matters.** This is the key step toward being "worthy of a professional risk platform": risk reviewers and model committees assess calibration precisely through the reliability-curve chart. The data is already deterministically computed (upholding INV-1); this is a pure frontend presentation gap.

**How to fix.**

1. Add a `reliability_curve` layout: reuse the roc-card plot framework to draw the diagonal (perfect-calibration reference line, dashed) + the binned scatter-connected line, annotate Brier/ECE in the top-right corner, and on point hover show "predicted x% / actual y% / n samples".
2. Add a `score_band_bar` layout: score bands on the horizontal axis + bad-rate bars (reusing the score-precision-bar tier palette and grow animation), with monotonicity-breaking bands marked in the warning color.
3. On the backend, emit the corresponding layout and rows for these two blocks in the metrics/report payload sections (the data already exists — it just never enters sections).
4. Make the model_delivery_panel "校准" (calibration) chip click-expand into that curve card.

**Verification note.** The verifier found the situation slightly worse than stated: the reliability-curve data never enters any payload sent to the frontend at all (marvis/agent/gate_payloads.py:535-544 `_calibration_label` compresses calibration metadata to a text label), and score bands have zero frontend presentation of any kind — grep for score_band/分数分段 under marvis/static/ returns nothing.

### VD-5 · The mascot still only passively follows V1 task status, with zero linkage to V2 gate/alert/metric events; the newly generated true-logo glow animation assets sit in scripts/ unwired into the product

**Impact:** Medium · **Effort:** M · **Verification:** — Not independently verified

**Problem.** The mascot is the brand asset the user explicitly cares about most (the requirement was "animate the real logo pixels"), yet it reacts to none of the product's real highlight or danger moments — it keeps idling while a gate waits for confirmation, shows no worry on a JOIN inflation alert, and does not celebrate when KS hits target. Meanwhile a finished true-logo glow loop animation has entered no interface at all.

**Evidence.** app.js:824-833 `basePetMoodFromTask` reads only `selectedTask.status` (the V1 state machine); app.js:282 reaction moods are only four one-shot reactions — success/failed/complete/review; V2 gate appearance, JOIN fan-out alerts, and KS attainment events have no linkage to the pet whatsoever. The sprite system at styles.css:1592-1692 supports 7-mood multi-frame animation (assets complete, 7 skins under pets/). Additionally, scripts/assets/ already holds marvis-glow-smooth-loop.webp/gif generated from the real logo pixels (full generation pipeline in scripts/generate_marvis_glow_animation.py), but nothing under marvis/static/ references them.

**Why it matters.** A context-aware mascot is a "visible change" achievable at zero new art cost (the sprites exist), and an opportunity to extend the state language from panels up to the brand layer; leaving the glow animation unused means the work was done for nothing.

**How to fix.**

1. Layer V2 signals into the mood source in renderPetState: `plan has an awaiting_confirm step → review`; `latest gate metadata carries fan_out/leakage red flags → failed (worry — restrained, played once)`; `final review KS on target → success`. All the data is already in the polled plan/messages state — a pure frontend mapping, upholding read-only INV-4.
2. After 90s of idle, play a one-shot stretch/peek easter-egg frame (the sprite rows exist; add a timer + data-pet-mood="idle-peek").
3. Wire the scripts/assets glow loop into two places: a hover-swap frame for the welcome page's workspace-brand-logo, and a small-size breathing logo next to the agent thinking indicator (`<img src=pets/…webp>` or a CSS mask).
4. All animations respect prefers-reduced-motion (the sprite system already has this degradation) and keep the off switch in settings.

### VD-6 · State encoding in the modeling setup / model delivery panels is only a 1px border color change: no icons, no fill color — weak hierarchy and indistinguishable for color-blind users

**Impact:** Medium · **Effort:** S · **Verification:** — Not independently verified

**Problem.** Delivery readiness (PMML exportability / calibration status / monotonicity checks) is the conclusive signal for whether a model can go live; right now ready/warning/error differ only by the color of a 1px border — nearly indistinguishable when scanning, and completely indistinguishable for color-blind users. It is also inconsistent with the tonal state language already established in task-hero/welcome.

**Evidence.** v2-workbench.css:1579-1604 `.model-delivery-readiness-card[data-readiness-kind=ready|warning|error]` and `[data-signal-kind]` change only border-color; the card body remains the uniform `border:1px solid; background:var(--model-panel-surface)` (1542-1555). modeling-guidance-item (1335-1358) is slightly better (3px left bar + a fill for warning), but the ready/error states have no icons. The status text in model_delivery_panel.js:128-143 carries no ✓/△/✕ symbols.

**Why it matters.** This is the panel most often screenshotted and shown to others in review/delivery scenarios; dual encoding (color + shape) is also the accessibility baseline.

**How to fix.**

1. Add tonal fills and icons for the three kinds: `[data-readiness-kind="ready"]{background:color-mix(in srgb,var(--model-signal-ready) 7%,var(--model-panel-surface))}`, and likewise for warning/error.
2. Insert a status glyph before the card's strong element (CSS ::before content "✓"/"△"/"✕" with the matching color), or add a `<span class="signal-glyph">` in JS.
3. Add a 3px left status bar aligned with the existing modeling-guidance-item pattern, unified into a single `.signal-card[data-kind]` rule set shared by both panels.
4. Verify contrast in dark mode (--model-signal-* is already mapped to bright variants in dark, styles.css:326-329).

### VD-7 · The feature screening gate table shows KS/IV only as bare 4-decimal numbers: no databar, no IV tier interpretation — mismatched with this gate's decision weight

**Impact:** Medium · **Effort:** S · **Verification:** — Not independently verified

**Problem.** At this gate the user must decide "which features enter the model" and needs to quickly compare the relative magnitudes of KS/IV across dozens of rows — currently readable only as decimals. IV also lacks the industry-convention tier hints (<0.02 no predictive power / 0.02-0.1 weak / 0.1-0.3 medium / >0.3 strong, with very high values warranting leakage suspicion).

**Evidence.** screen_gate_controller.js:37-39 renders KS/IV/missing-rate as `screenNum()` toFixed(4) plain text; the screen-table CSS (v2-workbench.css:1779-1848) has only right alignment + tabular-nums + row-tint categorization, with no relative-magnitude visualization of any kind. The category badges (keep/leak/susp) are already done well.

**Why it matters.** Feature-selection quality directly affects final model KS and OOT stability (the user's core goal); this is the last major gap in copying the PSI tiering benchmark over to IV/KS (the driver-side residue of quick win #5 from 6-28).

**How to fix.**

1. Apply mini databars to the KS/IV cells: `<span class="databar" style="--fraction:…">` (fraction normalized against the column max; reuse columnFractions from render-metrics.js).
2. Add data-tip tier-interpretation copy for IV (write an ivTooltipText following the psiTooltipText pattern; present the 0.02/0.1/0.3 thresholds as reference conventions, not hard gates), and when IV>0.5 add a △ hint "too high — watch for leakage" echoing the leakage badge.
3. Support click-to-sort on headers (keep the current fixed selected→leakage→suspected→unusable order as the default).
4. All of this is presentation-layer only; threshold judgments remain determined by the backend screen results (INV-1).

### VD-8 · Dark-mode glass texture is not on par: the task-hero top highlight is cut from 0.72 to 0.08, and the glass language essentially disappears in dark

**Impact:** Medium · **Effort:** S · **Verification:** — Not independently verified

**Problem.** The 6-28 report's item "bring dark-mode glass/state texture to parity" is still undone (it is not on the fix list). The user's taste is glass + depth, and under the dark theme the facade cards degrade into near-flat translucent gray blocks.

**Evidence.** styles.css:3891-3892 — the light task-hero has `box-shadow: inset 0 1px 0 rgba(255,255,255,0.72)`; styles.css:4253-4262 — dark overrides it to `inset 0 1px 0 rgba(255,255,255,0.08)`; styles.css:238 — the dark value of `--glass-edge` is likewise 0.08. In dark, backdrop-filter is still present but the edge light and layering are nearly invisible.

**Why it matters.** Dark is the common theme for modelers working long hours; brand texture that is inconsistent between the two themes reads as unfinished.

**How to fix.**

1. Rebuild the glass edge for dark task-hero with a double inner shadow: `inset 0 1px 0 rgba(255,255,255,0.16), inset 0 -1px 0 rgba(0,0,0,0.35)`, and brighten the first gradient stop (--surface 76%→82%).
2. Raise `--glass-edge` in dark to 0.14-0.18 and make task-hero reference the token directly (line 3892 is currently a hardcoded rgba — tokenize it while at it).
3. Raise the data-tone glow opacity in dark from 0.10 to 0.16-0.18 (dark backgrounds absorb more chroma).
4. After the change, compare light/dark with the same screenshot flow and let the user decide — a low-risk "visible" change.

### VD-9 · Databars/KPI bars and key numbers have no entrance animation: the repo already has a staggered grow-animation paradigm (score-precision-bar) that goes unreused

**Impact:** Medium · **Effort:** S · **Verification:** — Not independently verified

**Problem.** The metric reveal (first render of the KS/AUC cards) is the product's "report-card moment"; right now every bar appears instantly, and the existing refined animation paradigm is used in only one place — score-precision.

**Evidence.** styles.css:6584-6598 `.databar-fill` declares `transition: width .35s ease-out`, but the width is set to its final value via the inline `--fraction` at innerHTML creation, so the transition never plays; kpi-card-primary-bar (styles.css:6815-6823) is likewise static. By contrast, styles.css:4850-4852 `.score-precision-bar` already implements `animation: precision-bar-grow 0.55s … backwards; animation-delay: calc(var(--bar-index,0)*60ms)` — per-bar staggered growth — plus a `data-animation="none"` degradation switch.

**Why it matters.** The user explicitly likes animation; this is an S-cost quick win that makes "result presentation" feel a notch more premium, with a ready-made pattern to copy and no change whatsoever to data behavior.

**How to fix.**

1. Give `.databar-fill`, `.kpi-card-primary-bar > i`, and `.kpi-card-row-bar > i` a `transform-origin:left; animation: bar-grow-x 0.5s cubic-bezier(0.22,1,0.36,1) backwards; animation-delay: calc(var(--bar-index,0)*50ms)` (keyframes scaleX(0)→1 — cheaper on layout than animating width).
2. Write `--bar-index` for each row at render time.
3. Play only when a data-metric-key appears for the first time (reuse the existing render-signature cache to determine "new content"; polling redraws must not replay).
4. Optionally add a one-shot 300ms count-up to the KPI main value (requestAnimationFrame interpolation, with the final value taken from the backend string to prevent rounding drift from violating deterministic display).
5. Use the same `data-animation="none"` channel as score-precision under `prefers-reduced-motion`.

### VD-10 · Four V2 view modules — join_review/plan_view/loop_progress/subagent_view — have zero runtime mounts, emit a dozen-plus CSS-less class names, and survive only on node tests

**Impact:** Medium · **Effort:** S · **Verification:** — Not independently verified

**Problem.** These modules are residue of the early v2 workbench approach: the real product has moved to the agent-chat + plan-rail route. They mean that most of the flat card styles at v2-workbench.css:191-204 (plan-step/join-card/loop-evt) serve views users will never see, and that past visual reviews (including the 6-28 report's "evidence panels all flat" criticism) partly landed on dead code — wasting review and maintenance bandwidth.

**Evidence.** The v2 imports at app.js:48-84 include only capability/driver_*/governance_extensions/join_gate_controller/modeling_setup_panel/model_delivery_panel/plan_rail_controller/plugin_manager/screen_gate_controller/skill_manager/state_v2; join_review.js, plan_view.js, loop_progress.js, subagent_view.js, workflow_create.js, and plan_confirm.js have no runtime importer at all (a grep across all of static/ finds only one internal link, plan_confirm→plan_view). The class names emitted by join_review.js:185-198 — join-warning/join-key-pair/join-diagnostics/anchor-key/match-rate — and by plan_view.js:88-94 — plan-status-badge/step-status-* — have 0 rules in all CSS files; node unit tests such as tests/test_frontend_v2_join.py import these modules directly, keeping the lights green.

**Why it matters.** Dead views plus orphan class names will keep misdirecting future design investment; and if they are ever mounted by mistake, they will ship bare on screen with browser default styles.

**How to fix.**

1. Make an explicit decision: if these modules have no mounting plan within 6 months, delete them (together with the corresponding v2-workbench.css selectors and node tests); plan rail + chat-embedded controls already cover their responsibilities.
2. If join_review is kept as a "JOIN plan details" dialog (it has product value: a multi-table join overview), then formally mount it at the plan rail's join-step detail entry and fill in the missing styles — join-warning (warning tonal card) / join-key-pair (key-pair chip) / match-rate (databar) — aligned with the VD-1 language.
3. Either way, restrict the node unit tests to modules still used at runtime, to avoid the "tests green = feature exists" illusion.

### VD-11 · Remaining design-token gaps: all four radius tiers collapsed to 16px, spacing/type scales still at zero, 17 distinct font-size literals scattered around

**Impact:** Medium · **Effort:** L · **Verification:** — Not independently verified

**Problem.** The color leg has grown in (the "Tokenize colors" series across the 166 commits), but spacing and font sizes still rely on hand-written literals — "negotiated values" like 12.5/13.5/11.5px will keep multiplying; and radius semantic tiers existing in name only means any future corner-radius adjustment requires whole-file replacement. This was explicitly listed as "still not landed" in the 6-28 report, and the status is unchanged since then.

**Evidence.** styles.css:48-52 — `--radius:16px; --radius-sm:16px; --radius-md:16px; --radius-lg:16px; --radius-control:10px` (the sm/md/lg semantic tiers are fully collapsed); no `--space-*`/`--fs-*` tokens anywhere in static/ (grep: 0 hits); 17 distinct font-size literals (12px×111, 13px×38, 14px×30, 11px×22, 12.5px×9, 13.5px×2, 11.5px×2, etc.). For contrast: color tokenization is done well (~190 token lines in :root, full dark coverage, metric/chart/model-panel domain-scoped naming).

**Why it matters.** V2 panels are still proliferating fast (this round alone added three large CSS blocks: modeling setup / model delivery / screen table); every day the consolidation slips, the migration cost grows. A unified scale is also the prerequisite for the later major visual overhaul (driven by comparison mockups).

**How to fix.**

1. Establish the scale first without moving a pixel: `--space-1..6: 4/8/12/16/20/24px`, `--fs-xs:11px --fs-sm:12px --fs-base:13px --fs-md:14px --fs-lg:15px --fs-xl:18px`; make them mandatory for new code (add a stylelint declaration-property-value-allowed-list rule or a review checklist).
2. Migrate existing usage in three mechanical replacement batches: 12.5→--fs-sm(12) or --fs-base(13), 13.5→--fs-base, 11.5→--fs-xs, running visual screenshot diffs after each batch.
3. Restore radius semantics: --radius-lg=16 (cards), --radius-md=12 (embedded panels), --radius-sm=8 (chips). ⚠️ Smaller corner radii would touch the user's preferred rounded-glass look — comparison mockups of the 3 tiers must be produced first and signed off by the user; if the user does not approve, keep everything at 16 and change only the reference relationships (semantics first, values later).
4. --radius-control=10 is working correctly — do not touch it.

---

## 8. Architecture & Code Quality

**Lens verdict.** The api.py split is 90% of the way there (0 residual routes, 922 lines, the routers/ + repositories/ + api_*_helpers direction is basically right, and the helper files are each cohesive rather than junk drawers), but the last mile has stalled at a "compatibility shim": the validation_agent routes call back into 36 `api._private` functions via `from marvis import api as legacy_api`, 22 test sites import api private symbols, and the newly created seams (DriverTurnRuntime and friends) are all typed `Any`. Meanwhile the god file has migrated rather than disappeared: packs/modeling/tools.py has ballooned to 4386 lines (tool entry points + policy engine + markdown rendering mixed together), pipeline.py is 2090 lines with 0 log statements, app.js is 6416 lines, and templates/sample.py belies its name by holding all 11 production templates. The 6-28 "audit transactionalization" fix left 15 `getattr(*_with_audit)` duck-typed soft probes unclosed (2 of which fall back to writing no audit at all — residual INV-8 risk). Cross-cutting quality debt: the 5 driver turn handlers and 4 pack `_Runtime` classes are wholesale copy-paste, the error taxonomy is scattered across two layers (132 hand-written HTTPExceptions + bare error_kind strings), and the plugin worker protocol has no version handshake while the subprocess reverse-imports the runner, dragging the DB dependency chain into every worker cold start.

### ARCH-1 · api.py split stalled at "relocation": the validation_agent router calls 36 private functions through a legacy_api service locator, and the new seams are all typed Any

**Impact:** High · **Effort:** M · **Verification:** ⚠️ Partially confirmed

**Problem.** api.py has been thinned into a hybrid of "composition root + compatibility re-exports", but the HTTP adapter layer (routers/validation_agent.py) depends on the private symbol table of a legacy module — the dependency direction is route layer → legacy module private API, not route layer → explicit service. Anyone renaming or deleting a `_` function in api.py silently blows up the router and 22 test sites; the newly extracted seams (DriverTurnRuntime, ValidationJobCallbacks) have all-`Any` fields, so mypy performs zero checking on this most central orchestration chain.

**Evidence.** marvis/routers/validation_agent.py:13-19 `def _agent_api(): from marvis import api as legacy_api; return legacy_api`, followed throughout the file by 36 calls to underscore-private functions such as `api._repo(request)`, `api._dispatch_driver_turn(...)`, `api._resolve_agent_model(...)`; marvis/api.py:12-107 contains a dozen-plus `# noqa: F401 - compatibility export` imports, 0 `@router` routes (only L119 include settings_router), and ~50 pure forwarding wrappers (e.g. L403-418 `_run_agent_scan_stage` merely fills in `deps=_validation_stage_dependencies()` and forwards to `_impl`); 22 occurrences of `from marvis.api import _xxx` in tests; marvis/agent/turn_handlers.py:43-50 `DriverTurnRuntime` has all 7 fields typed `Any`.

**Why it matters.** This is the most frequently changed chain in the whole repo (the agent conversation / gate decision entry point). The private-symbol coupling means every further split step has to touch 3 places at once (api.py, router, tests), so split velocity will keep degrading; the `Any` seams mean the type system stays silent during refactors, leaving the 1750 test cases as the only safety net and driving regression cost up.

**How to fix.**
1. Create `marvis/agent/validation_app_service.py`: move the composition-root logic from api.py (`_validation_stage_dependencies`, `_dispatch_agent_validation_job`, `_confirm_agent_report_conclusions`, `_dispatch_driver_turn`, `_resolve_agent_model`, etc.) into it under public names, keeping function signatures unchanged.
2. In routers/validation_agent.py, delete `_agent_api()` and import directly: `from marvis.agent.validation_app_service import dispatch_driver_turn, ...`.
3. Degrade api.py to pure re-exports (one `from ... import x as _x` per line), with a header marking it deprecated plus a planned removal version.
4. Repoint the 22 test imports to the new module (doable with a single sed pass).
5. Give DriverTurnRuntime real field types: `plan_repo: PlanRepository`, `planner: Planner`, `plan_validator: PlanValidator`, `llm_client: OpenAICompatibleLLMClient | None` (app.py already imports these types, no import cycle). Zero behavior change throughout; the 5 steps can land as 5 separately verifiable commits.

**Verification note.** The adversarial pass confirmed the core claim (service-locator reverse dependency on legacy private symbols, 0 `@router` routes in api.py, no mypy configuration anywhere in pyproject.toml) and found the coupling actually worse than stated: 71 `api._xxx` call sites across 25 distinct private symbols, not 36. Several counts were corrected: 8 compatibility-export noqa comments (not a dozen-plus), 43 `def _` functions in api.py (not ~50, and not all pure forwards), and 16 test import statements covering ~20 private symbols (not 22). The type-gap claims were overstated: DriverTurnRuntime has 6 of 7 fields typed `Any` (`tier: str` is typed), and ValidationJobCallbacks' 15 fields are all `Callable`-annotated (6 with parameter-eliding `Callable[..., X]`), not `Any`. "Silently blows up" is also too strong for tests — they fail loudly with AttributeError/ImportError; only the routes fail lazily at runtime.

### ARCH-2 · packs/modeling/tools.py has ballooned to 4386 lines as the new god file: tool entry points, selection-policy engine, monitoring policy, model card, and ~800 lines of markdown rendering mixed together

**Impact:** High · **Effort:** M · **Verification:** ⚠️ Partially confirmed

**Problem.** The modeling pack used to be a "genuinely deep module" (adding a tool touched only 2 files), but the deliverable-governance features (policy / monitoring / model card / approval package) were all piled into tools.py, and the tool entry points are now drowned under 3000 lines of supporting code. The markdown rendering is pure-functional yet lives in the same file as side-effecting tool entry points, and policy decision logic (business rules) is entangled with display formatting (presentation layer).

**Evidence.** marvis/packs/modeling/tools.py totals 4386 lines (the largest Python file in the repo, bigger than any backend file on the pre-split problem list): L1025-1310 selection-policy engine (`_selection_policy_decision`/`_selection_policy_violations` and ~15 sibling functions); L1655-2103 monitoring policy + challenger comparison; L2104-2931 model card payload plus `_approval_package_markdown` (L2427) / `_model_card_markdown` (L2722) and other pure markdown string rendering; L2933-3189 report generation; L3191-3202 `_Runtime`; only L186-1491 holds the actual 25 tool_* entry points. Tests contain 6 occurrences of `from marvis.packs.modeling.tools import _selection_policy_decision`, testing the private implementation directly.

**Why it matters.** The modeling pack is the main battlefield for the product's core goal (the model-development agent); every upcoming deliverable enhancement (governance trio, reason codes, fairness) will keep growing this file. A 4000+ line file is also exactly where a weaker local model doing assisted coding/review loses context focus. Tests importing private functions directly means the module boundary has already been fossilized by tests — the later the split, the more expensive it gets.

**How to fix.** Split mechanically along the already-clear cohesive blocks (pure relocation, zero behavior change):
1. `selection_policy.py` (the `_normalize_selection_policy`/`_selection_policy_*` family at L969-1310, exposing a public `decide_selection_policy` entry point).
2. `monitoring.py` (the `_monitoring_*` family at L1678-1802).
3. `challenger.py` (L1861-2103).
4. `model_card.py` (payload construction at L2104-2289).
5. `delivery_markdown.py` (all `*_markdown` pure functions at L2290-2931); tools.py keeps only the tool_* entry points + `_Runtime` + imports. Repoint the 6 private test imports to public names in selection_policy (leaving a re-export in tools.py as a transition). Post-split, tools.py is expected to land at ~1500 lines.

**Verification note.** The verifier confirmed the 4386-line count, every cited line number (`_approval_package_markdown` at L2427 and `_model_card_markdown` at L2722, both pure dict → str renderers, with the side-effecting write calls at L2374-2405 in the same file), and the 6 private test imports at tests/test_modeling_pack.py:1394,1419,1438,1481,1505,1558. Two facts were corrected: there are 20 tool_* entry points (matching manifest.json), not 25 — 18 in L186-1491 plus `tool_generate_model_report`/`s` at L2933/2940 outside the claimed range; and it is the largest production Python file inside marvis/ (next largest: pipeline.py at 2086 lines), not the largest in the whole repo — tests/test_frontend_static_v2.py is 8858 lines.

### ARCH-3 · 15 residual `getattr(*_with_audit)` duck-typed soft probes (the 6-28 "audit transactionalization" fix was not cleaned up fully), 2 of which fall back to writing no audit at all

**Impact:** High · **Effort:** S · **Verification:** ⚠️ Partially confirmed

**Problem.** The 6-28 report listed "audit hasattr soft degradation" as a fixed item (JoinEngine construction-time enforcement), but the fix consisted of adding `getattr` feature probes at every call site rather than closing the interface contract. This degrades INV-8 (audit completeness) from a type-guaranteed hard constraint into a duck-typed optional: any new code path that passes in a repo-like object missing the method will make critical governance writes such as report-conclusion overrides (api.py:233) silently take the audit-free path, with no warning of any kind.

**Evidence.** A repo-wide grep for `getattr(.*_with_audit` hits 15 sites: marvis/api.py:216 `update_conclusions = getattr(repo, "update_agent_report_conclusions_with_audit", None)`, whose else branch at L233-237 calls `repo.update_agent_report_conclusions(...)` directly with no audit written at all; marvis/agent/validation_stages.py:452/469-474 is the same pattern with the same bare fallback; the fallbacks at join_engine.py:299-303, orchestrator/subagent.py:105-110, plugins/registry.py:46/64/80, agent_memory/evolution.py:15/29, drafts/promotion.py:132/153, routers/report_fields.py:52 do write audit, but the business write and the audit are not in the same transaction. All real Repository classes already implement *_with_audit (27 in total under repositories/); the fallbacks exist only for test fakes.

**Why it matters.** INV-8 is the platform's baseline invariant for passing review/audit; "looks fixed, actually left 15 backdoors" is exactly the recurrence pattern the previous review round called out. The 15 if/else double branches are also ~10 lines of duplicated boilerplate per call site, obscuring the real logic.

**How to fix.**
1. Define `class AuditedTaskRepo(Protocol)` (and siblings) in marvis/repositories/__init__.py, listing the *_with_audit methods as required interface members.
2. Delete all 15 getattr probes and fallback branches and call the audited methods directly (all real repos have them — a zero-risk deletion).
3. Add the corresponding methods to the test fakes (most just need a forward-plus-record on the fake).
4. If compatibility with ultra-minimal fakes is truly needed, provide a shared fake base class in `tests/fakes.py` instead of leaving a degradation path in production code.
5. Add a one-line grep CI check (`getattr(.*with_audit` count must be 0) to prevent recurrence.

**Verification note.** The adversarial pass confirmed all 15 getattr sites and found the situation slightly worse than stated: there are 3 fully audit-free fallbacks, not 2 — marvis/routers/report_fields.py:68-73 also calls `repo.update_report_values` with no `write_audit` at all; all three ultimately reach `_merge_report_values(audit=None)` at repositories/tasks.py:651-657, which writes no audit when audit is None. In addition, the drafts/promotion.py fallbacks at L137-143 and L157-165 are conditional (gated on `hasattr(registry, "_repo")` / `audit_repo is not None`) and can silently skip audit writes under certain assemblies — weaker than the "writes audit but non-transactionally" bucket the finding placed them in. The 27 *_with_audit implementations, the absence of any Protocol/ABC contract, and the absence of any logger warning on the fallback branches were all confirmed.

### ARCH-4 · The five run_*_driver_turn handlers in turn_handlers.py are wholesale copy-paste (~330 lines of isomorphic code); the "adding a stage" tax remains unpaid

**Impact:** Medium · **Effort:** M · **Verification:** — Not independently verified

**Problem.** One goal of the V2 completion plan was to eliminate the "adding a stage means changing 3 contracts" tax, but adding a new PlanDriver stage today still requires copying ~60 lines of turn handler, defining an isolated SetupError, and adding a branch in dispatch_driver_turn (L423). The three copies have already drifted (the [C1] display-replacement logic in modeling/join at L66/L330 is inconsistent with the other three), so the odds of "fix one, forget four" are high going forward.

**Evidence.** marvis/agent/turn_handlers.py:127-314: run_feature_driver_turn / run_strategy_driver_turn / run_vintage_driver_turn are line-for-line isomorphic — append user message (only the intent string differs) → if `_active_plan` exists then `driver.resume` → otherwise `build_*_proposal` + opening message → `driver.start` → the `except XxxSetupError / except DriverError: raise / except Exception` triple; join (L53) / modeling (L317) add one extra C1 pre-flow. The 5 XxxSetupError classes each independently inherit ValueError with no common base (feature_setup.py:23, join_setup.py:32, modeling_setup.py:29, strategy_setup.py:34, vintage_setup.py:32); the response functions shared by all stages are named join_turn_response/append_join_error (L535/539), a naming residue of join-specific semantics.

**Why it matters.** This is the backbone extension point of conversation-driven orchestration; the platform roadmap (more task types in V3/V4) means this code will be copied repeatedly. The naming mismatch (join_turn_response serving all stages) will also mislead future maintainers into thinking it is join-specific logic.

**How to fix.**
1. Define `@dataclass StageSpec: intent: str; template_for(task, proposal); build_proposal: Callable; opening_message: Callable[[proposal], str]; setup_error: type[Exception]; error_label: str; pre_flow: Callable | None` (turn the join/modeling C1 flow into the optional pre_flow).
2. Write one generic `run_stage_driver_turn(spec, runtime, repo, task, ...)`; the 5 existing functions degrade into 5 StageSpec table entries.
3. Give the 5 SetupError classes a common base `StageSetupError(ValueError)` (each module keeps its subclass name so existing excepts are unbroken); the generic runner catches only the base class.
4. Rename join_turn_response/append_join_error to driver_turn_response/append_driver_error (keep the old names as aliases for one release). Net deletion of ~250 lines.

### ARCH-5 · Plugin worker protocol has no version handshake, and the subprocess reverse-imports the runner, dragging the DB/registry dependency chain into every worker cold start

**Impact:** Medium · **Effort:** M · **Verification:** — Not independently verified

**Problem.** The worker protocol is a real cross-process — potentially cross-interpreter — boundary, yet today it works only on the implicit assumption that both ends happen to be the same copy of the code. As soon as an execution environment configures a separate modeling python environment (a capability that already exists), the worker side may load an old marvis: guard semantics (network/file guard), error_kind taxonomy, and resource_limits fields can all drift silently, and failures will only surface as inexplicable protocol/execution errors. The reverse import also makes every tool call's cold start pay the module-loading cost of the DB layer.

**Evidence.** marvis/plugins/subprocess_worker.py:21 `from marvis.plugins.runner import ToolContext` — the worker subprocess imports the entire runner just to get a 4-field dataclass, while runner.py:13-18 top-level imports `marvis.db` (PluginRepository → repositories → db_schema), registry, schema_validation, redaction, safe_paths; runner.py:637 launches the worker with `[python_executable, "-m", "marvis.plugins.subprocess_worker"]`, where python_executable can be a different interpreter (app.py:262 `environment.python_executable or sys.executable`), and the `_worker_env` allowlist (runner.py:22-40) does not include PYTHONPATH — so the worker resolves marvis from the plugin interpreter's own site-packages/CWD, which may not match the host version; neither the job dict (runner.py:298-312) nor the result protocol carries any protocol_version field.

**Why it matters.** The plugin/draft ecosystem is V2's main extension axis (learned skills, third-party packs); an unversioned protocol means any future change to job-field semantics is undetectable, violating the architectural goal of "an evolvable plugin protocol" and indirectly touching INV-6 (reliable subprocess isolation).

**How to fix.**
1. Create `marvis/plugins/contracts.py` (a leaf module with zero internal marvis dependencies) to hold ToolContext, protocol field constants, and `PROTOCOL_VERSION = 1`; both runner and subprocess_worker import from it (runner.py keeps a re-export for compatibility).
2. Add `protocol_version` to the job dict; worker_main validates at the top: unknown version → `_emit({ok: False, error_kind: "protocol", error: "protocol version mismatch: host=1 worker supports=..."})`; the result protocol carries back `worker_marvis_version=__version__`, which the runner persists into the audit detail.
3. Add an optional `manifest_format: 1` field to manifest.py and validate it in the loader, leaving an upgrade channel for future permissions/hook vocabulary evolution.
4. While there, collapse the 6 verbatim-duplicated `self._finalize_audited_result(...)` tail calls inside the runner into a single exit point.

### ARCH-6 · pipeline.py is still a 2090-line god file with 0 log statements: stage running, notebook source codegen, memory distillation, and artifact cleanup are four mixed responsibilities

**Impact:** Medium · **Effort:** M · **Verification:** — Not independently verified

**Problem.** This is the execution core of the V1.1 validation main chain, and having four unrelated responsibilities (orchestration, code generation, memory, cleanup) in one file means: changing the metrics cell template requires locating a string block inside 2000 lines; the cell source generation is pure-functional yet cannot be unit-tested as an independent module (today the only verification is an end-to-end notebook run); and the entire pipeline has zero logging, so production failures can only be reverse-engineered from status_message strings — the zero-behavior-change increment proposed in the 6-28 report ("logger.exception at stage boundaries") was never done either.

**Evidence.** marvis/pipeline.py is 2090 lines with 57 top-level defs: L145-593 three stage runners; L734-1257 roughly 520 lines building notebook cell Python source via string concatenation (`_build_reproducibility_cell_sources`/`_build_metrics_cell_sources`); L1766-1926 agent memory capture (the `_capture_agent_memory_*` family); L1682-1751 artifact cleanup; L2034-2089 state-machine marking. `grep -c "logger\." marvis/pipeline.py` = 0 — the file never even creates a logger instance (contrast api.py:120, which has one).

**Why it matters.** For a single-machine product running a local weaker model, notebook execution is the most failure-prone path (memory/kernel/dependencies), and the absence of logs directly raises the cost of every incident investigation; un-unit-testable cell codegen means template changes can only be regression-tested via slow integration tests.

**How to fix.** Land three zero-behavior-change commits:
1. Extract `marvis/notebook_codegen.py`: move all of L734-1257 (`_build_*_cell_sources`/`_notebook_package_prelude`/`_json_literal` — pure functions, immediately snapshot-testable: given a contract, assert the generated source contains the key lines).
2. Extract `marvis/pipeline_memory.py`: move the memory-capture family at L1766-1926 (code guarding INV-4 is easier to review when concentrated in one place).
3. Create a logger in pipeline.py, add `logger.info(task_id, stage)` at the entry of run_notebook_stage/run_metrics_stage/run_report_stage and `logger.exception` at except boundaries followed by re-raise (changing no raise/state behavior). Post-split, pipeline.py is expected to land at ~1200 lines.

### ARCH-7 · Error taxonomy scattered across two layers: per-package domain exception trees, 132 hand-written HTTPException mappings in routers, and bare error_kind strings with no enum

**Impact:** Medium · **Effort:** M · **Verification:** — Not independently verified

**Problem.** The same domain error can map to different status codes at different endpoints (KeyError is 404 in some places and 422 in others), and adding a new exception type means remembering to add an except in every router; error_kind — the pivot field for audit and frontend triage — has no single source of truth, so misspelling one string (e.g. "time_out") produces no compile-time or test-time signal whatsoever.

**Evidence.** The exception trees are mutually independent: plugins/errors.py (PluginError with 10 subclasses), orchestrator/errors.py (OrchestratorError), data/errors.py (FanOutError etc.), packs/modeling/errors.py, packs/strategy/errors.py, state_machine (ConflictError/IllegalTransition), pipeline.py:77 (PipelineError), agent/plan_driver.py (DriverError), plus the 5 XxxSetupError classes. HTTP mapping is entirely hand-written: 17 files under routers/ total 132 HTTPExceptions (data.py 21, drafts.py 19, plans.py 17, ...), with plans.py:71-336 containing a dozen-plus `except XxxError → HTTPException` boilerplate blocks; app.py:215 registers only 1 exception_handler in the whole app (IllegalTransition); error_kind is a scattering of bare string literals (execution/protocol/resource/paused/hook/audit/postcheck/timeout/schema, spread across runner.py, subprocess_worker.py, executor.py, etc.).

**Why it matters.** Error classification directly determines the tiered display of frontend failure cards and retry semantics (the already-shipped failed-step retry depends on error_kind); mapping drift makes identical failures look inconsistent in the UI and skews audit statistics (aggregation by kind), hurting the analyzability side of INV-8.

**How to fix.**
1. Create `marvis/errors.py` with `class ErrorKind(StrEnum)` to close over all kind literals (start with a mechanical replacement of the strings in runner/subprocess_worker/executor, values unchanged).
2. Define a `class DomainError(Exception): http_status: int` mixin so that ConflictError (409) / PlanNotFoundError (404) / StageSetupError (422) / FanOutError (409) / DedupRequiredError (409) declare their own status.
3. Register one generic `@app.exception_handler(DomainError)` in app.py mapping by http_status while preserving detail.
4. Delete the try/except boilerplate now covered by the handler router by router (one commit per file, guarded by the existing API tests); expected net deletion of 300+ lines. Take care to keep the detail wording unchanged so as not to break frontend string matching.

### ARCH-8 · The four packs' _Runtime bootstrap code and _resolve_feature_cols are wholesale copy-paste; a pack SDK common layer is missing

**Impact:** Medium · **Effort:** S · **Verification:** — Not independently verified

**Problem.** Packs were designed as deep modules (manifest + tools, two files plus tools), but there is no official pack common layer, so every new pack starts by copying an existing one, and platform-level logic — bootstrap code, frame fetching, candidate feature inference — is copied and then evolves independently. Once the DatasetRegistry constructor signature changes, or caching/metrics need to be added uniformly to all packs, 4+ places have to change.

**Evidence.** The isomorphic `_Runtime` + `_runtime(ctx)` pair appears 4 times: data_ops/tools.py:330-342, feature/tools.py:435-445, strategy/tools.py:198-209, modeling/tools.py:3191-3203 — each being build_settings(ctx.workspace) + DatasetRepository + DataBackend + DatasetRegistry plus 1-2 pack-specific repositories, and drift has already appeared (data_ops/feature do not keep the settings reference; strategy/modeling do). `_resolve_feature_cols` is line-for-line identical between feature/tools.py:448-470 and modeling/tools.py:3206-3227, differing only in the final raise (FeatureError vs ModelingError); data-fetching helpers like `_dataset_frame` are likewise written once per pack.

**Why it matters.** The V2 roadmap only grows the pack count (skills learned through drafts will also solidify into packs); the copy-pasted bootstrap layer is the template future third-party pack authors will copy. If it is not consolidated now, it will be unrecoverable once the ecosystem takes off.

**How to fix.**
1. Create `marvis/packs/common.py` (or packs/_sdk/runtime.py): a `class PackRuntime` holding settings/datasets_root/repo/backend/registry and providing `dataset_frame(dataset_id, columns=None)` and `resolve_feature_cols(dataset_id, features, target_col, split_col, error_cls)`.
2. Each pack's `_Runtime` becomes a PackRuntime subclass adding only its own repositories (modeling adds experiments/modeling_repo, strategy adds strategies).
3. Delete one of the two `_resolve_feature_cols` copies in feature/modeling, passing the error type as a parameter.
4. The module ships with the platform and does not enter the plugin worker protocol (builtin packs load in the host's own interpreter, so INV-6 is unaffected). Net deletion of ~120 lines, plus an official bootstrap paradigm for third-party packs.

### ARCH-9 · All 11 production workflow templates crammed into a 1096-line file named templates/sample.py

**Impact:** Medium · **Effort:** S · **Verification:** — Not independently verified

**Problem.** Templates are the "constitution" of plan-driven orchestration (every stage's steps, post_checks, and confirmation gates are declared here), yet they hide in a file called sample.py: a new maintainer or reviewer will skip it by name; stringing 11 templates through one file makes diff review of long templates like MODELING_WITH_JOIN (234 lines) hard, and shared fragments across templates (the join prefix steps are declared redundantly in all 3 *_WITH_JOIN templates) have nowhere to be consolidated.

**Evidence.** marvis/orchestrator/templates/sample.py (1096 lines) defines, beyond SAMPLE_ECHO (L26): MODEL_VALIDATION (L46), STANDARD_MODELING (L102), DATA_JOIN (L259), MODELING (L325), MODELING_WITH_JOIN (L559), FEATURE_ANALYSIS (L805), FEATURE_ANALYSIS_WITH_JOIN (L849), FEATURE_DERIVATION (L924), STRATEGY_ANALYSIS (L992), vintage_analysis (L1054) — i.e. every built-in production template on the platform; the file name and its contents are severely mismatched.

**Why it matters.** Template correctness maps directly to INV-3 / confirmation-gate semantics (`_JOIN_EXECUTE_POST_CHECKS` is defined right here, L19-23); most gaps on the master gap list ultimately land as template changes, so this file's readability is the denominator of orchestration evolution speed.

**How to fix.**
1. Split by domain into templates/{validation,join,modeling,feature,strategy,vintage}.py, leaving only SAMPLE_ECHO in sample.py.
2. Extract a shared-constants module templates/shared.py to hold `_JOIN_EXECUTE_POST_CHECKS` and the join-prefix StepTemplate tuple repeated across the three *_WITH_JOIN templates (reuse via tuple concatenation — still declarative data, no dynamism introduced).
3. Keep the `_register_builtin_template` registration order in templates/__init__.py unchanged (template registration is an explicit call, so this is pure relocation with zero behavior change).
4. Guard with the existing test_orch_* template tests.

### ARCH-11 · app.js at 6416 lines and 431 functions is still a frontend god file: state ownership is undemarcated, and the state_v2 store hosts only the capability tier after being built

**Impact:** Medium · **Effort:** L · **Verification:** — Not independently verified

**Problem.** The current extraction strategy is "extract a controller per widget" (layout-resize, task-search, focus-ring), which only peels off leaves; what actually keeps app.js from shrinking is state ownership: render functions read/write closure state directly and call each other, so any panel modularization must first answer "where does the state live". state_v2.js already provides a get/set/subscribe skeleton (103 lines) but has no migration plan, producing a half-finished "store exists but is unused" state.

**Evidence.** marvis/static/app.js is 6416 lines with 431 functions; although it is already an ES module with ongoing extraction (recent commits: Extract layout resize controller / form control focus ring guard; js/ + js/v2/ already hold 30+ modules), app.js's use of the state_v2.js store is a single line, L84 `import { getSelectedTier, onSelectedTierChange }` — all page state beyond the 9 v2.* state keys defined by the store (current task, message list, polling handles, panel open/close) still lives scattered across app.js closure variables; the largest function blocks sit at L6112 (111 lines), L4290 (97 lines), L4644 (84 lines), and L5715 pollValidationProgress (65 lines).

**Why it matters.** The frontend is the main battlefield for the user's ongoing taste iteration (glass / animation / mascot), and a 6400-line entry file makes the regression surface of every visual redesign uncontrollable; it also blocks the 6-28 proposal's UX items ("incremental polling render, skeletons, status language"), all of which need a clear state→render boundary to diff against.

**How to fix.** Set a "state first" three-step plan (each step independently mergeable):
1. Migrate the task-detail page's core state (currentTaskId, messages, pollHandles, activePanel) into a state_v2.js-style store (a new js/store/task_view.js is fine); switch app.js render functions to subscription style (without touching the render logic itself yet).
2. Extract task_detail_controller.js along the largest blocks (the three 100-line-class functions at L6112, L4290, L4644), allowing dependencies only on the store + api modules.
3. Extract the polling family (pollAgentMessagesUntilSettled L5458, pollValidationProgress L5715) into polling_orchestrator.js and merge with the existing js/polling.js. Target: app.js drops below 3000 lines and retains only assembly and routing.

### ARCH-10 · db_schema.py migrations are unversioned: every startup runs 30+ CREATE TABLE plus 32 add-column probes, and only additive migrations will ever be expressible

**Impact:** Low · **Effort:** S · **Verification:** — Not independently verified

**Problem.** Unversioned migration has two ceilings: (a) it can only express "add table / add column" and cannot safely express transformational migrations — column type changes, backfills, build-index-then-drop-old-column — which are exactly what several performance items in the 6-28 report (agent-message composite index, audit index) require as DDL migrations; (b) the `_ensure_column` list only ever grows, so startup probe overhead and code noise expand linearly with schema evolution (already 32 sites, 687 lines).

**Evidence.** marvis/db_schema.py init_db (from L21) executes all CREATE TABLE IF NOT EXISTS statements plus 32 `_ensure_column` calls (one PRAGMA table_info probe each) on every invocation; the file contains no PRAGMA user_version / schema_version table (grep finds nothing); every repository such as TaskRepository is constructed with db_path, and init_db is called at app startup.

**Why it matters.** On a single-machine single-user setup the startup cost is tolerable, but "migration expressiveness" is a hard gap: the first time a non-additive migration is needed (e.g. building a (task_id,id) composite index on agent_messages and cleaning old data), there will be no safe execution framework — only a hand-written one-off script.

**How to fix.**
1. Introduce `PRAGMA user_version`: `MIGRATIONS: list[tuple[int, Callable[[Connection], None]]]`, with migration 0 = the current init_db full logic (probe-and-stamp the version on pre-existing databases first).
2. At startup, read user_version and sequentially execute only higher-version migrations, each inside a transaction followed by `PRAGMA user_version = n`.
3. Freeze the existing 32 `_ensure_column` calls into migration 0 and stop growing them; all new columns go through new migrations.
4. Add a schema_version field to the sqlite_health output for diagnostics. Mind DDL transaction semantics under WAL mode; avoid mixing DDL with large amounts of DML inside a single migration.

---

## 9. Reliability & Recovery

**Lens verdict.** After 166 commits, the reliability foundation has visibly thickened: staging is transactional (ArtifactUnitOfWork + startup-time reconcile), audit records are written in the same transaction, notebooks have been moved to subprocess isolation + env allowlist + psutil RSS soft monitoring + process-tree SIGKILL, failed plan steps already have a transactional retry endpoint, and the V1.1 pipeline has been split into three independently triggerable stages (notebook/metrics/report). The remaining gaps cluster into three categories: (1) concurrency control completely misses the V2 driver turn — it runs synchronously on the HTTP request thread, creates no job, and takes no lock; a double-sent "确认" (confirm) or two browser tabs triggers a second PlanExecutor.run that mistakes the currently executing step for a "restart orphan" and marks it FAILED — this is the heaviest reliability defect right now; (2) crash recovery does not cover the V2 plan layer (startup only reclaims tasks/jobs; a RUNNING plan spins forever with no restart notice), and an interrupted V1.1 metrics stage forces a full notebook rerun because the reclaim message does not match is_metrics_failure (the already-written last_completed_step is dead code); (3) the guardrails are asymmetric — notebooks have RSS soft monitoring, while the plugin worker that carries the V2 modeling/JOIN main execution path still only has setrlimit, which is largely ineffective on macOS. Further medium/low leaks: jobs have no heartbeat/watchdog (a hung job permanently locks its task via the unique index), and file-level .bak crash residue is neither restored nor cleaned up.

### REL-1 · V2 driver turns bypass the single-active-job guard: a double-sent "确认" (confirm) or dual browser tabs trigger concurrent executor.run on the same plan, and the step that is actually executing gets misjudged as "interrupted by restart" and marked FAILED

**Impact:** Critical · **Effort:** M · **Verification:** ✅ Confirmed adversarially

**Problem.** V2 conversation-driven JOIN/FEATURE/MODELING turns (including execute_join, model training, and other minute-scale operations) run synchronously on the FastAPI request thread. After Tab A sends "确认" (confirm), the step enters RUNNING and invokes the worker subprocess; at that moment Tab B (or the same-tab user who gets impatient and re-sends the confirmation, or a network retry replaying the request) sends another confirmation: inside resume(), `_awaiting_step` no longer finds an AWAITING_CONFIRM step → gate=None → is_confirm is true → executor.run(plan_id) is invoked again → `_recover_inflight_steps` treats the step Tab A is currently running (has a running step_run, no output) as a server-restart orphan and marks it FAILED, after which the main loop marks the whole plan FAILED. When Tab A's tool finishes and writes back DONE/output, it collides with the FAILED status (the CAS in set_plan_status raises ConflictError → 500). Typed text confirmations carry no expected_step_id; gate_response_adapter.py:42 lets plain confirmations without controls straight through, so the step-token mechanism designed against stale confirms cannot intercept this path.

**Evidence.** marvis/routers/validation_agent.py:106-119 — post_agent_message for driver task types synchronously calls _dispatch_driver_turn with no job/lock whatsoever (contrast: the rerun_stage path at L144 in the same file has _reject_if_task_has_active_job); marvis/agent/turn_handlers.py:423-449 — dispatch_driver_turn calls the turn functions directly; marvis/agent/plan_driver.py:239-240 — `self._executor.run(plan_id)` executes synchronously inside `_run_and_handle`; marvis/orchestrator/executor.py:68-76 — a plan with status=RUNNING is allowed into run() and `_recover_inflight_steps` runs first; executor.py:345-368 — RUNNING steps without output_ref are directly marked FAILED ("interrupted during running before output was persisted; explicit retry required") and the step run is marked interrupted. By contrast, all three entry points in marvis/routers/plans.py:150/171/199 first call `_start_plan_job` (backed by the idx_jobs_active_task partial unique index in marvis/db_schema.py:196-202) — only the driver turn creates no job.

**Why it matters.** This is a data-plane incident triggerable by the user's most common action (sending a confirmation) in the most common environment (dual tabs / fast re-sends / frontend retries): an in-flight JOIN/training run gets falsely marked as failed, potentially causing duplicate tool invocations (the same execute_join running twice), a scrambled audit stream, and the trust-destroying experience of "it actually finished, yet shows failed". It simultaneously violates INV-8 (the execution order reflected in the audit is distorted) and turns recovery logic designed for server restarts into a self-inflicted weapon. Single-machine single-user does not mean single-request.

**How to fix.**
1. Reuse the existing single-active-job guard: before the driver branch of post_agent_message / agent/start enters, call `start_task_job(repo, task_id, kind='driver_turn')`, and finish_job at the end of the turn (including on exceptions); on conflict return 409 with an agent message saying "上一轮仍在执行中" (previous turn is still executing), and have the frontend put the input box into a waiting state.
2. Add an in-process in-flight registry to PlanExecutor: at the top of run(), `with _plan_run_guard(plan_id)` (threading.Lock + set); if the plan is already running, return the current status directly instead of going through _recover_inflight_steps.
3. Tighten the trigger condition of `_recover_inflight_steps` to "this plan is not in the in-process registry" (i.e., only genuine restart orphans get reclaimed) — this step is the root of the defense.
4. Add a concurrency regression test: thread A executes a slow tool, thread B calls run() on the same plan; assert B does not reclaim and A finishes DONE normally.

**Verification note.** The adversarial pass confirmed every link of the chain against the current working tree and found no fallback mechanism, including that the typed-confirm bypass at gate_response_adapter.py:35-43 defeats the step-token protection and that no concurrency test exists. One detail correction: the write-back collision may also surface as IllegalPlanTransition (marvis/orchestrator/harness_state.py:64-66) rather than ConflictError — either way, only DriverError maps to 409 (marvis/api.py:653-654), so both surface as a 500.

### REL-2 · A server restart reclaims tasks interrupted during COMPUTING_METRICS with a generic FAILED message that is_metrics_failure does not recognize → users are forced to rerun everything from the notebook stage; the ready-made last_completed_step() is dead code

**Impact:** High · **Effort:** S · **Verification:** ✅ Confirmed adversarially

**Problem.** After the notebook stage (the most time-consuming step, potentially tens of minutes) succeeds, the task enters COMPUTING_METRICS. If the service restarts/crashes at that point, the startup reclaim marks the task FAILED with the generic message "reclaimed: server restart while running". The user then tries to resume metrics: POST /tasks/{id}/metrics → is_metrics_failure=False → 409 "cannot generate metrics in status failed". The only way out is to rerun the notebook stage, even though runtime_contract.json, code_model_scores.csv, and model_meta.json under execution/ are all intact (last_completed_step was written precisely to detect this, but is never wired in).

**Evidence.** marvis/recovery.py:65-67 — reclaim sets status_message to "reclaimed: server restart while running" (reason=SERVER_RESTART); marvis/agent/orchestrator.py:50-54 — is_metrics_failure only recognizes METRICS_STAGE_FAILURE_PREFIX (pipeline.py:99, "模型效果&稳定性验证失败：" — "model performance & stability validation failed:") or the four markers at orchestrator.py:83-88; the restart message matches none of them; marvis/routers/validation_stages.py:148-153 — tasks that are FAILED and not metrics_retry are rejected with a 409 for metrics reruns; marvis/recovery.py:21-37 — last_completed_step() can determine from on-disk artifacts that the notebook stage is complete, but a repo-wide grep finds no caller.

**Why it matters.** This is the most tangible, cheapest segment of the "resume from checkpoint" story: notebook execution is the most expensive stage of the entire V1.1 validation flow, and a single restart invalidates it, halving the resume-value of the stage-split work. The fix is extremely cheap (a decision predicate plus one line of message) and the payoff is one-click continuation from metrics after a restart.

**How to fix.**
1. Distinguish stages at reclaim time: in reclaim_stale_running_tasks, for tasks with status=COMPUTING_METRICS call last_completed_step(task_dir); if it returns "notebook", set status_message to METRICS_STAGE_FAILURE_PREFIX + "服务重启中断，可从指标阶段重试" ("interrupted by server restart; can retry from the metrics stage") — or, more cleanly, have is_metrics_failure additionally accept status_reason_code==TASK_STATUS_REASON_SERVER_RESTART together with last_completed_step()=='notebook'.
2. On the frontend failure card for such tasks, make the "重跑指标" (rerun metrics) button the default highlighted option.
3. Add a regression test: construct COMPUTING_METRICS + complete execution artifacts → reclaim → assert run_task_metrics returns 202 rather than 409. Note that reclaim_stale_running_tasks needs an additional tasks_dir parameter (it currently only receives db_path).

**Verification note.** The verifier confirmed the entire chain and surfaced an aggravating detail: the frontend actually lights up a clickable "retry metrics" button (failure_stage="metrics" derived from the failed job kind via marvis/api_task_payloads.py:60-68 and app.js:1543/1523), but the backend gate only consults is_metrics_failure, so clicking it still yields a 409 — "the UI offers a button that the backend rejects". The agent path is equally blocked (validation_service.py:78/91 also depends on is_metrics_failure).

### REL-3 · The plugin/pack tool subprocess (V2 modeling and JOIN's main execution path) still has only setrlimit as its memory guardrail — largely ineffective on macOS; the psutil RSS soft monitoring already proven on notebooks is not reused in ToolRunner

**Impact:** High · **Effort:** M · **Verification:** ✅ Confirmed adversarially

**Problem.** V2's truly memory-hungry deterministic computations — train_lgb/execute_join/screen_features etc. — all execute through the ToolRunner subprocess (INV-6), and their memory cap relies on setrlimit: on macOS (Darwin, the primary development platform) the kernel mostly does not enforce RLIMIT_AS, and RLIMIT_DATA only constrains brk allocations while modern malloc uses mmap — meaning memory_limit_mb=2048 in the manifest is effectively a comment on macOS. A single large-dataset training run or Cartesian-style JOIN can exhaust physical memory, and the resulting system-wide swap storm drags down the FastAPI main process and SQLite polling, rendering the whole machine unusable; the 6-28 report itself admits the "real OOM slow test remains incomplete".

**Evidence.** marvis/plugins/subprocess_worker.py:566-571 — only setrlimit(RLIMIT_DATA/RLIMIT_AS); :597-599 — on failure it merely appends to meta['errors'] (observable but no fallback); marvis/plugins/runner.py — no psutil reference anywhere in the file (grep: 0 hits); on overrun the only recourse is the tool.timeout_seconds wall-clock (runner.py:141/317). By contrast, marvis/notebooks.py:517-626 — _NotebookResourceMonitor already implements psutil process-tree RSS sampling (:604-620, on limit → on_limit + _terminate_process_tree) and is wired into the pipeline with a default of 4096MB.

**Why it matters.** This is the last remaining gap in INV-6 on the core business path: the notebook path is already covered (subprocess + RSS monitoring + process-tree kill), while the plugin worker that carries the bulk of execution volume is the one running bare. The user's central goal (modeling) runs precisely on the weakest-guarded path; for a single-machine product, the cost of "one runaway takes down everything" is far higher than in a multi-tenant environment.

**How to fix.**
1. Extract _NotebookResourceMonitor into a shared module (e.g., marvis/resource_monitor.py), parameterizing the pid-getter and on_limit.
2. In runner._run_worker (runner.py:636-653), after Popen start a monitor thread on process.pid, with the limit taken from tool.memory_limit_mb (already in the manifest); on overrun → reuse _kill_worker_tree (runner.py:657-669) → return error_kind='resource' with resource_usage carrying peak_rss_mb, written back to the protocol and audit.
3. Keep setrlimit as the first layer of defense in depth; when psutil is unavailable, mark meta as degraded (consistent with notebook semantics).
4. A slow-test group: allocate genuinely large arrays inside the worker, assert error_kind=='resource', the main process survives, and no orphan grandchild processes remain (verified via psutil).

**Verification note.** The verifier confirmed zero psutil hits in the plugins package (psutil exists only in marvis/notebooks.py), that the only effective limit on macOS is the wall-clock timeout in _run_worker's communicate(timeout) → killpg, and that docs/reviews/2026-06-28-v2-improvement-proposals.md:27 and :93 explicitly list the RSS soft monitor and real-OOM slow test as not yet done.

### REL-4 · Startup recovery does not cover the V2 plan layer: after a restart, RUNNING plans/steps are never reclaimed, and driver tasks receive no restart notice (the run_mode/status/jobs trigger conditions all fail to fire)

**Impact:** High · **Effort:** M · **Verification:** ✅ Confirmed adversarially

**Problem.** A user confirms a training step in a V2 modeling task, and the service restarts mid-training: the plans table retains plan=RUNNING and step=RUNNING as-is. After the restart, (a) the conversation stream carries no "interrupted" notice of any kind (none of _add_agent_restart_notices' filter conditions fire); (b) the frontend plan rail reads the RUNNING plan and keeps showing an executing spinner; (c) only when the user sends another message and resume() happens to reach executor.run does the step lazily get marked FAILED — and that FAILED message says "interrupted during running", so the user has no idea a restart happened in between.

**Evidence.** marvis/recovery.py:13-18 — ORPHAN_RECLAIM_STATUSES contains only TaskStatus.RUNNING/COMPUTING_METRICS; recovery.py:101-118 — the restart notice requires run_mode='agent' and (the task in one of those statuses, or a queued/running job) — but V2 driver turns create no job (see REL-1), and marvis/agent/turn_handlers.py never calls update_status anywhere in the file (the task stays in a quiescent state such as scanned); marvis/app.py:163-165 — the startup sequence = init_db + reclaim_stale_running_tasks + reconcile_workspace_artifacts, with no recovery whatsoever for the plans/plan_steps tables; marvis/orchestrator/executor.py:339 — _recover_inflight_steps is only triggered passively on the next run().

**Why it matters.** V2 is the flagship path, and it is precisely the blind spot of the recovery story: after a restart, V1.1 validation tasks get the full three-piece kit (message finalization + restart notice + job marked failed, recovery.py:145-217), while V2 gets nothing. Long training tasks plus local deployment (laptop lid closed, system updates) mean restarts are the norm, not the exception — and "spinning forever" is the most trust-damaging failure mode there is.

**How to fix.**
1. Add plan reclaim at startup: SELECT plans WHERE status='running' → for each plan, run logic equivalent to executor._recover_inflight_steps (CHECKING steps that have output get their review completed; RUNNING steps without output are marked FAILED-retryable and their step_runs finished as interrupted/ServerRestart), and set the plan to FAILED. The executor's recovery logic can be extracted into a standalone function shared by both call sites.
2. For the affected plans' tasks, inject an agent message of the same kind as _add_agent_restart_notices, with copy along the lines of "可点击『重试步骤』从失败步继续（中间产物已保留）" ("click 'Retry step' to continue from the failed step; intermediate artifacts are preserved") — the retry endpoint (plans.py:190-225) already exists; all that is missing is walking the user to the door.
3. Once V2 driver turns are brought into jobs (REL-1), the notice condition here can directly reuse the jobs criterion.

**Verification note.** The verifier confirmed the full chain, with one wording correction: of the three AND-ed trigger conditions, the run_mode='agent' leg does actually match for agent tasks — it is the status and jobs legs that fail; the net effect is identical (no notice is sent). It also verified that the REST job-creating bypass (POST /plans/{id}/run etc.) is not what the V2 frontend uses — gate confirmations go through driver_gate_confirm.js:43 → POST /api/tasks/{id}/agent/messages, the no-job path — and that even on the REST path the RUNNING plan rows would still never be reclaimed at startup.

### REL-5 · No heartbeat/watchdog for long jobs: a job whose process is alive but whose thread is hung permanently locks the task via the idx_jobs_active_task unique index (409 active job), and join/plan jobs have no cancel endpoint

**Impact:** Medium · **Effort:** M · **Verification:** — Not independently verified

**Problem.** The single-active-job constraint of start_job is good design, but it assumes the job will always reach finish_job. If a BackgroundTasks thread hangs (a large DuckDB JOIN with no timeout, disk IO hang, extreme GIL/lock contention), that job stays running forever: from then on the task returns 409 "task already has an active job" for every stage/plan/join operation, and no API can unlock it — the only way out is restarting the entire service (relying on the startup reclaim to clear the field). LLM calls have a 60s timeout (llm_client.py:61-63) and tool workers have a wall-clock timeout, but work running on main-process threads — join execution and report rendering — has no fallback time limit.

**Evidence.** marvis/db_schema.py:196-202 — the idx_jobs_active_task partial unique index guarantees one active job per task; marvis/recovery.py:40-44 — reclaim is called exactly once at startup by marvis/app.py:164, with no reaper at runtime; the jobs table (db_schema.py:173-188) has no heartbeat column; marvis/routers/stage_controls.py has only the three cancel endpoints for notebook/metrics/report; the async join job at marvis/routers/data.py:343-360 and the plan job at plans.py:145-158 have no cancellation path; notebooks have a timeout but DuckDB JOINs (join_engine executed via the backend) have no SQL-level timeout.

**Why it matters.** "Must restart the service to keep using a given task" is a hard flaw for a single-machine product; more insidiously, the user cannot see why — the 409 copy never tells them which job has been stuck for how long. A heartbeat plus watchdog is also the foundation for a future execution-health panel and ETAs.

**How to fix.**
1. Add a heartbeat_at column to jobs; have run_stage_job/_run_plan_job/_run_join_execute_job use a lightweight wrapper thread that UPDATEs heartbeat_at every 15s.
2. An in-app background reaper (daemon thread, 60s period): jobs that are running with a heartbeat older than a threshold (e.g., 10min, configurable per kind) get marked failed(error_name='HeartbeatLost') with an audit record, releasing the unique index.
3. Add POST /tasks/{id}/jobs/cancel, dispatching by kind (join → add cancellation-flag checkpoints to join_engine; plan → cancel_plan plus _kill_worker_tree for in-flight workers).
4. Expose a stuck-job count on /api/health so that "hung" is at least observable.

### REL-6 · Zero observability while a driver turn executes: no job, no streaming placeholder message, no step-progress events — during minute-scale training/JOIN the user sees only their own message, and after a refresh there is no trace at all, directly inviting the duplicate confirmations of REL-1

**Impact:** Medium · **Effort:** M · **Verification:** — Not independently verified

**Problem.** With a weak model plus large local data, a driver turn (post-confirmation JOIN/training execution + LLM review) easily exceeds 30-120 seconds. During that window: the last message in the conversation stream is the user's own "确认" (confirm); the HTTP response has not returned, so the frontend spins or times out; if the user closes and reopens the page, there is not even a spinner. The natural human reaction is to re-send "confirm" or hammer the button — which is exactly the trigger for the REL-1 race. The reliability problem and the observability problem are one and the same here.

**Evidence.** marvis/agent/turn_handlers.py:524-532 — append_driver_messages writes messages in one batch only after the entire turn completes; the driver branch (validation_agent.py:106-119) does not call _add_streaming_agent_message at api.py:874 (V1.1 validation jobs do have this placeholder); the executor's step states do exist in the plan_steps table (visible on the plan rail), but the main conversation stream has no expression of "currently executing step N", and after a page refresh, since no job row exists, the frontend cannot distinguish "turn in progress" from "the service isn't doing anything".

**Why it matters.** "Where am I in this long task" is the intersection of UX and data safety: invisible progress → repeated actions → concurrency incidents. The earlier report's Loop progress-upgrade proposal only covered the notebook rail, not this main V2 driver-turn path.

**How to fix.**
1. At driver-turn entry, first persist a streaming placeholder agent message (metadata: {streaming:true, driver_running:true, plan_id, step_id}); at turn end, replace or finalize it with the final gate/done/failed message — this conveniently reuses the restart-finalization logic in recovery.py:145-179 (which only recognizes messages with streaming:true), killing REL-4's notice blind spot with the same stone.
2. Have the executor emit step.completed through the existing hook_dispatcher when each step finishes → a built-in listener updates the placeholder message content ("已完成 2/5：特征筛选" — "completed 2/5: feature screening").
3. Combined with REL-1's driver_turn job, when the frontend polls an active job of kind='driver_turn' it renders the executing state and disables the confirmation input.

### REL-7 · File-level staging backups (.<name>.<hex>.bak) left behind by crashes: startup reconcile only restores backups in the plugins and validation_handoff directories, while file backups under the datasets/tasks roots are neither restored nor cleaned

**Impact:** Medium · **Effort:** M · **Verification:** — Not independently verified

**Problem.** The ArtifactUnitOfWork window: promote_all has already moved the old file to .bak and placed the new file at the final location → the DB transaction has not yet committed → the process crashes. After restart, SQLite rolls back (the DB returns to the old metadata), but the filesystem is stuck at "new content in the final location, old content in .bak": the file content the DB row points to no longer matches the row count/fingerprint recorded at registration time, and nobody restores the .bak. For datasets (the newly transactionalized paths such as Stage dataset raw uploads / metrics model metadata), this means lineage records and the actual bytes can be misaligned; even when no inconsistency is triggered, .bak files accumulate without bound. The plugins path does the fully correct checksum reconciliation + restore (recovery.py:73-94); the file-level path never received the same treatment.

**Evidence.** marvis/artifacts/transactional.py:177 — StagedArtifact's backup_path = root/f".{final_path.name}.{token}.bak" (RemovedPath uses the same pattern at :267); marvis/artifacts/recovery.py:15 — the name group of _BACKUP_DIR_RE, [A-Za-z_][A-Za-z0-9_-]{0,127}, does not allow '.', so it cannot match file backups like ".model_meta.json.<hex32>.bak"; recovery.py:32-46 — reconcile_workspace_artifacts only does _remove_staging_dirs + plugin directory backup reconcile + validation_handoff directory reconcile, with zero handling of .bak files under datasets/tasks.

**Why it matters.** The entire point of the staging mechanism is "after a crash: either old or new, never an in-between state"; omitting the recovery end means only half the job was done. For a credit-modeling platform, dataset parquet misaligned with DB fingerprints is an INV-3/INV-8-grade hazard (albeit with a small window), and .bak accumulation pollutes the user's data directory.

**How to fix.**
1. Relax _BACKUP_DIR_RE to allow '.' in the name group (or define a separate _BACKUP_FILE_RE for file backups), and scan the datasets/tasks roots at startup.
2. Follow the plugins reconciliation strategy: if a committed record matching the final file can be found in the DB (by path + fingerprint), delete the .bak; otherwise restore (.bak → final). For paths with no fingerprint to check, at minimum move stale .bak files into .staging-trash and count them in the ArtifactRecoveryReport (app.state.artifact_recovery_report is already exposed and visible to the frontend).
3. Add a crash-window test for finalize_with_connection: raise after promote to simulate a crash → run reconcile → assert the old content is back at the final location.

### REL-8 · execution_environment.json is written non-atomically and read with no fault tolerance: if a crash truncates the file, every pipeline dispatch returns a straight 500 with no self-healing path

**Impact:** Low · **Effort:** S · **Verification:** — Not independently verified

**Problem.** A power loss/crash at the instant the execution-environment settings are being saved → truncated file → every subsequent notebook/metrics run raises JSONDecodeError (500) inside load_execution_environment, and the settings page may not open either; the user's only way out is to manually locate and delete workspace/settings/execution_environment.json. This is a leftover of the same class as the recent "Write runtime sidecars atomically" fix (abc9df7e).

**Evidence.** marvis/execution_environment.py:60-63 — save_execution_environment overwrites directly with path.write_text (the repo already has marvis/files.write_text_atomic, in use at notebooks.py:386 and elsewhere); :45 — load_execution_environment uses json.loads(path.read_text()) with no try/except; that load is invoked indirectly via api_stage_helpers.pipeline_settings_from_request on every notebook/metrics/validate dispatch.

**Why it matters.** A corrupted config file taking down the core flow with no UI self-healing is a textbook "low-probability, high-frustration" failure; the fix is under ten lines.

**How to fix.**
1. Switch save to write_text_atomic.
2. Have load catch (json.JSONDecodeError, OSError) → logger.warning + return the ExecutionEnvironmentSettings() defaults (the file is a reconstructible user preference, not authoritative data).
3. While at it, grep the remaining settings/*.json write sites and consolidate them onto write_text_atomic.

### REL-9 · The synchronous branch of execute_join_plan has no job guard (asymmetric with the async branch): two entry points executing the same join plan concurrently leaves a TOCTOU double-execution window

**Impact:** Low · **Effort:** S · **Verification:** — Not independently verified

**Problem.** The final defense against double execution is JoinEngine's internal executed-status transaction plus the 1:1 row-count assertion, but two executions would each write the joined parquet once and race on record_join_result_with_audit — best case, the second one errors out leaving a superfluous parquet; worst case, the audit ends up with two join.executed entries or the artifacts overwrite each other. The async branch already demonstrates the correct approach; the synchronous branch simply never caught up.

**Evidence.** marvis/routers/data.py:328-378 — the async branch (:343-345) first calls start_task_job(kind='join') to claim the single-active index, while the synchronous branch (:362-368) calls join_engine.execute_join_plan directly; the "already executed" check (:340-341) is based on a single read before execution, so two concurrent synchronous requests (or sync + async, or sync + an execute_join step inside a driver turn) can all pass the check and enter execution simultaneously.

**Why it matters.** JOIN is the heart of INV-3 territory; any window in which "the same plan gets executed twice" is worth closing shut, and the fix is one-line-scale.

**How to fix.**
1. Have the synchronous branch likewise call start_task_job(task_repo, plan.task_id, 'join') first, with finish_job in a try/finally.
2. Or, more thoroughly: add claim_join_execution(join_plan_id) to DatasetRepository (UPDATE ... SET status='executing' WHERE status='confirmed', 409 if rowcount=0), so that all entry points (HTTP sync/async, pack tool) share a single CAS claim point.

---

## 10. Testing Strategy & Security

**Lens verdict.** The deterministic guardrails on the testing and security surface are themselves solidly built (uv.lock pinning with hashes, WAL + busy_timeout, plugin worker env allowlist, post-hoc path validation, transactional audit writes, and regression coverage over many transactional write paths), and the 1729 test cases provide broad coverage. Viewed through the "testing pyramid" lens, however, there are three structural gaps: (1) all tests live in a single flat layer — no fast/slow/e2e tiers, no markers at all, and CI runs everything in one pass, dragging down both local iteration speed and signal; (2) the claimed Playwright e2e is in fact a "render smoke test" that feeds canned JSON through a hand-written static server — it never touches the real FastAPI app and is skipped by default in CI, so real user journeys have zero automated coverage; (3) every LLM touchpoint aimed at weak models has only been tested with a FakeLLM that returns perfect JSON: there is no eval set for the degraded outputs that actually break things ("fences/prefixes/negation/truncation"), and the isolation and memory-guardrail tests mostly assert "killpg was called / resource was mocked" rather than a real kill or a real OOM. Two easily overlooked practical risks remain on the security side: dataset uploads are read fully into memory and Excel/CSV are parsed in full with no size guardrail (both a bomb and a legitimately large dataset will take down the single process), and the interactive modeling kernel path still inherits the complete host environment variables (asymmetric with the already-hardened plugin worker / batch notebook). The redaction regexes both over-mask and under-mask, and there is no CVE scanning on the dependency side. Findings below are ordered by impact.

### TST-1 · No eval set for the weak-model LLM touchpoints: only tested with a FakeLLM returning perfect JSON, zero coverage of real degraded outputs

**Impact:** High · **Effort:** M · **Verification:** ⚠️ Partially confirmed

**Problem.** The product's explicit goal is to run on local 32–72B weak models, and the real failure modes of weak models are: ```json fences, prose prefixes/suffixes, field-name casing or Chinese/English mixing, truncated half-JSON, and negation semantics ("不可以继续" — "must not continue"). These are exactly what the newly added extraction/retry/negation guards are supposed to defend against, but in the tests the LLM always emits perfect JSON — which puts the guardrails in a branch that never fires. Any regression in a prompt or in the extraction logic would leave the suite fully green and go undetected.

**Evidence.** tests/test_instruction_router.py:10-18 `_FakeLLM.complete` directly returns preset strings (e.g. '{"action":"confirm",...}'); test_orch_reviewer.py:8-16 is likewise a FakeLLM returning clean canned output; `grep golden/eval_set/prompt_eval/expected_decision tests/` yields 0 hits; tests/fixtures/ contains only contract_notebook_v3.ipynb and min_lr.pmml, with no prompt corpus. decide_gate/route_instruction/reviewer have been changed to JSON extraction + retry (landed status already recorded), but the inputs to these code paths are always clean canned JSON.

**Why it matters.** This directly determines the probability of the agent "getting dumber / stopping inexplicably / treating a negation as a confirmation" on weak models — the product's number-one experience and correctness risk (INV-3 gate decisions). Without an eval set there is no way to quantify whether a model swap or a prompt change made things better or worse, nor to regression-guard against recurrence of the already-fixed negation misreads.

**How to fix.**
1. Build a golden corpus under tests/eval/: for each touchpoint (gate decision / instruction routing / reviewer / planner), prepare 20–50 (raw_model_output → expected_parsed_decision) samples covering fences, prose prefixes, truncation, negation, misspelled field names, and multiple JSON blocks.
2. Write a pure function-level harness that feeds `_extract_json`/`is_confirm`/the router parser directly (no real LLM needed), asserting the parsed result and the safe fallback.
3. Add an optional `@pytest.mark.llm_eval` tier that runs against a real local model and reports a hit-rate baseline (configurable threshold, no hard-coded number). This both regression-guards the guardrails and quantifies the impact of switching models.

**Verification note.** The adversarial pass confirmed the literal facts (canned `_FakeLLM` payloads at tests/test_instruction_router.py:11-18, zero hits for golden/eval_set/prompt_eval/expected_decision in tests/, fixtures limited to contract_notebook_v3.ipynb and min_lr.pmml, and marvis/orchestrator/eval/ only evaluating PlanRunTrace plan behavior rather than LLM text-parsing robustness) — but it refuted the core claim that degraded outputs have zero coverage and that the guards never fire. All three LLM touchpoints already have degraded-input trigger tests: markdown-fence/prose extraction, junk→clarify/halt, unknown-action→halt, and "not json"→retry paths in tests/test_instruction_router.py (:38, :56, :60, :73), tests/test_agent_autodrive.py (:308-320, :339-347, :711-719), and tests/test_orch_reviewer.py (:139-148, :151-159, :224-236); negation handling is not an LLM JSON path at all but a deterministic string guard (marvis/agent/service.py:141-157, marvis/agent/plan_driver.py:38), tested in tests/test_agent_intent.py:50-75 and tests/test_plan_driver.py:1570-1576. The real gap is narrower than claimed: no systematic weak-model degradation eval corpus, no dedicated cases for truncated half-JSON or key casing / language-mixed keys, and no standalone unit tests for the shared extraction helper marvis/agent/json_reply.py (load_json_object / _extract_first_object).

### TST-2 · Dataset uploads read fully into memory + Excel/CSV parsed in full with no size guardrail: both bombs and legitimately large datasets take down the single process

**Impact:** High · **Effort:** M · **Verification:** ✅ Confirmed adversarially

**Problem.** The upload endpoint has no byte limit of any kind, and parsing goes through full pandas materialization. A few-MB xlsx decompression bomb (billions of cells) or a perfectly normal ten-million-row CSV will exhaust memory in the FastAPI main process and/or block the event loop for a long time — a single-machine, single-process product simply becomes unusable. This is both a security DoS and a real business problem: credit sample tables routinely run to millions or tens of millions of rows. And there is no test coverage at all for large-file or bomb uploads.

**Evidence.** marvis/routers/data.py:115 `upload_artifact.path.write_bytes(await file.read())` reads the entire upload body into memory in one go; marvis/data/registry.py:271 `pd.read_csv(source_path, encoding='utf-8-sig')` reads the full file; marvis/data/excel_ingest.py:84 `full = pd.read_excel(path, sheet_name=sheet, header=None, engine='openpyxl')` reads the whole sheet; a repo-wide `grep MAX_UPLOAD|max_upload|content-length|max_request` yields 0 hits; `LARGE_ROW_THRESHOLD=200_000` (data/contracts.py:8) is used only on the JOIN read path and does not apply to upload/ingest.

**Why it matters.** This affects every ordinary user uploading a large dataset (the core entry point is precisely uploading sample/feature tables); one accidental upload drags down the entire platform. It is also an obvious gap for any security audit.

**How to fix.**
1. Add an explicit byte cap to the upload endpoint (env-configurable, e.g. 512MB); change data.py to a chunked, streaming write-to-disk like materials.py:45 `_save_upload_file`, returning 413 on overflow and cleaning up staging.
2. Before ingest, probe dimensions with openpyxl read_only / probe row counts with DuckDB `read_csv_auto ... LIMIT`; above the threshold, use DuckDB streaming conversion to parquet instead of `pd.read_excel(full)`/`pd.read_csv`.
3. Add slow tests: one oversized CSV asserting 413, and one high-compression-ratio xlsx asserting it is either rejected or streamed successfully without OOM.

**Verification note.** Confirmed on every point with minor line drift only: the full-body read is at marvis/routers/data.py:132 (inside async endpoint `upload_task_dataset`, data.py:116) and the full Excel read at excel_ingest.py:82. The verifier additionally established that the pandas parses run as synchronous code inside the async endpoint (directly blocking the event loop), and that no existing limit — materials.py:70 MAX_MATERIAL_UPLOAD_FILES (file count only), plugins/runner.py:132 file_size_limit_mb=2048 (plugin subprocess RLIMIT_FSIZE), web_search.py:42 max_bytes, or the app.py:173 local-client middleware — applies to the dataset upload path, and that tests/ contains no bomb/large-upload coverage.

### TST-3 · The claimed Playwright e2e is really a hand-written static server fed canned JSON — it never touches the real FastAPI, and CI skips it by default

**Impact:** High · **Effort:** M · **Verification:** ✅ Confirmed adversarially

**Problem.** These tests only verify "given canned JSON, the frontend components render the correct DOM" (a render smoke test). They never touch the real backend contract, the real plan_driver, the real DB, or the real confirmation gates. Consequently: if the backend response shape changes, the gate state machine regresses, or the JOIN→FEATURE→MODELING orchestration breaks, the e2e suite stays fully green. The real user journey (create task → upload → JOIN confirmation gate → features → modeling → delivery) has zero automated coverage in CI.

**Evidence.** tests/test_frontend_playwright_smoke.py:24-27 `pytestmark = skipif(MARVIS_RUN_PLAYWRIGHT_SMOKE != '1')` skips by default; :33-100 `_SmokeHandler.do_GET` returns hard-coded fixtures for /api/tasks, /agent/messages, /plans etc. (:64-91); the server is a `ThreadingHTTPServer` (:138), not the app under test. All 4 tests go through this canned server. CI (.github/workflows/ci.yml→scripts/check) does not set that env, so they never run.

**Why it matters.** This is a conversation-driven, multi-stage orchestration product where cross-layer contract drift is the most frequent source of regressions; lacking a real e2e means the "does the whole chain still work" verification is left entirely to manual testing.

**How to fix.**
1. Add 1–2 real e2e critical-journey tests: start the real create_app(tmp_path) via TestClient/uvicorn, have Playwright connect to the real app, and use an injectable deterministic LLM stub (running the real orchestrator/plan_driver, only replacing model output with scripted decisions). Cover "modeling task: confirm the spec gate → train → deliverables appear" and "JOIN: propose → a negation at the confirmation gate is blocked → confirm → execute with row count unchanged".
2. Mark them `@pytest.mark.e2e` and run them in CI after installing chromium (a separate job is fine; slowness is acceptable). Keep the existing canned smoke tests as the fast layer.

**Verification note.** Confirmed with minor line drift: `do_GET` actually starts at :52 (with :61-91 being the canned API fixture branch), and even the test named test_real_modeling_task_delivery_workspace (:534) feeds hand-written fixture dicts — "real" refers only to a real browser. The verifier noted that backend API shapes do have substantial TestClient coverage (test_api_v2.py, test_orch_api.py, test_data_join_api.py), but this does not change the core finding: browser-level front/back integration, real plan_driver orchestration, and confirmation-gate journeys have zero automated end-to-end coverage in CI.

### TST-4 · Isolation/resource guardrail tests assert "it was called / it was mocked" rather than "it really killed / really OOMed" — INV-6 is unverified end to end

**Impact:** High · **Effort:** M · **Verification:** ✅ Confirmed adversarially

**Problem.** Resource limits and process-group kill are the core defense line of INV-6, but the tests verify everything at the mock layer — "the code called killpg / setrlimit was invoked". Not a single test actually forks a grandchild process and asserts with psutil after a timeout that the whole process tree is dead, and none actually allocates a large array to assert error_kind=='resource' while the main process survives. An earlier review already called this out; the situation is unchanged. In effect, the suite has no idea whether the guardrails actually work in production.

**Evidence.** tests/test_plugin_runner.py:1099 test_tool_runner_kills_worker_process_group_on_timeout uses FakeProcess + `monkeypatch.setattr(...os.killpg, lambda pgid,sig: killed.append(...))` (around :1128), merely recording the call; :181 test_worker_resource_limits_apply... uses FakeResource to mock the `resource` module (:201 `monkeypatch.setitem(sys.modules,'resource',fake)`); `grep MemoryError|real forked grandchild tests/` shows no real-OOM or leftover-grandchild assertions. Only test_tool_runner_kills_timed_out_worker (:1090) starts a real sleep worker to verify the parent does not hang.

**Why it matters.** In a single-machine product, one runaway subprocess or OOM can drag down the entire FastAPI main service; if INV-6 is actually breached while the tests stay green, that is the most dangerous kind of "false safety".

**How to fix.**
1. Create a `@pytest.mark.slow`/`robustness` slow-test group (excluded from the fast tier by default, run as a separate CI job): a real worker that forks a detached grandchild and sleeps, asserting via psutil after the timeout that every PID under the pgid is dead.
2. A real worker that allocates an array exceeding the RLIMIT, asserting protocol error_kind=='resource' with the parent process still alive.
3. A notebook kernel that genuinely starts and runs a memory-hungry cell, asserting the RSS monitor triggers and _terminate_process_tree kills everything. Keep the existing mock unit tests as the fast layer.

**Verification note.** Confirmed; the killpg monkeypatch is at :1126 (not ~:1128). The verifier additionally established that the only real-subprocess test (test_tool_runner_kills_timed_out_worker, :1089) asserts only error_kind=='timeout', the sample sleep tool forks no grandchildren, _kill_worker_tree swallows ProcessLookupError (marvis/plugins/runner.py:657-671, killpg at :669), and no test anywhere asserts error_kind=='resource' (the MemoryError→"resource" mapping in marvis/plugins/subprocess_worker.py:42-45 is entirely uncovered; the only psutil hit in tests/ is an unrelated FakePsutil mock at tests/test_notebooks.py:932).

### TST-5 · 1729 test cases form one flat layer: no markers whatsoever, no fast/slow tiers, CI runs everything in a single pass

**Impact:** Medium · **Effort:** S · **Verification:** — Not independently verified

**Problem.** No test tiering means: a one-line local change still requires a full run (including PMML/JVM warm-up, lightgbm/xgboost training, subprocess worker cold starts, and other heavy tests), so feedback is slow; CI cannot run a 30-second fast tier first for an early signal before the slow tier; and there is no way to re-run the isolation/e2e slow group in isolation. The actual testing pyramid is "one big blob of mid-layer integration + slow IO" mixed together, missing a clear boundary between a fast unit layer and a slow integration/e2e layer.

**Evidence.** `grep 'def test_' tests/*.py` = 1729; pyproject.toml `[tool.pytest.ini_options]` contains only testpaths+pythonpath, with no markers/addopts; `grep 'mark.slow|mark.integration|mark.e2e|mark.concurrency' tests/` = 0; scripts/check uses PYTEST_ARGS=('-q'), i.e. a full run.

**Why it matters.** This affects the feedback speed of every development iteration and the CI cost, and it is the precondition for the slow-test groups recommended in TST-3/TST-4 to land at all; the missing tiers push developers toward "just skip running the tests".

**How to fix.**
1. Register markers in pyproject (unit/integration/slow/e2e/llm_eval); tag the training-heavy tests, real subprocess runs, PMML/JVM, and Playwright cases as slow/e2e.
2. Make scripts/check run the fast tier by default with `-m 'not slow and not e2e'`, and add a CI slow/e2e job for the rest.
3. Optionally use pytest-durations to identify the top slow tests and tag those first. Zero behavior change — purely organizational.

### TST-6 · Redaction regexes both over-mask and under-mask, and conversation transcripts (the largest PII surface) are not redacted at all

**Impact:** Medium · **Effort:** M · **Verification:** — Not independently verified

**Problem.** Two classes of issues: (1) over-masking — 13-digit millisecond timestamps, row counts, order numbers and other ordinary large integers get misclassified as bank cards and masked when written into memory/plan evidence, polluting the readability of audit evidence; (2) under-masking — 15-digit legacy national IDs and phone numbers with country codes or spacing separators are not matched. Meanwhile the agent_messages conversation transcript — where PII is actually densest (users routinely paste sample rows containing national IDs / phone numbers) — is not redacted at all. Although the conversation is source data and redacting it would hurt functionality, it is the source for memory distillation, creating a "源不脱、派生才脱" ("source unredacted, only derivatives redacted") asymmetry.

**Evidence.** marvis/redaction.py:11 `_BANK_CARD_RE = r'(?<![0-9A-Za-z])\d{13,19}(?![0-9A-Za-z])'` matches any bare 13–19 digit run; :10 the national-ID regex matches only 18-digit IDs (15-digit legacy IDs are missed); :9 the mobile-number regex matches only undelimited 11-digit numbers. marvis/repositories/tasks.py:505-512 add_agent_message INSERTs content directly with no redaction; redaction is wired into only three places: agent_memory/store.py:72-74, repositories/plans.py:489-491, plugins/runner.py:712.

**Why it matters.** The credibility of INV-5 defense-in-depth depends on coverage and precision; over-masking distorts archived evidence and under-masking makes redaction hollow — both undermine the promise of "observably real redaction".

**How to fix.**
1. For bank cards, switch to stronger contextual/Luhn validation, or require a key-name blocklist hit before masking, so bare digit runs are not blanket-masked; add 15-digit national IDs and separator-formatted phone numbers.
2. Add positive/negative redaction fixture tests (including "a 13-digit timestamp must NOT be masked" and "a 15-digit national ID MUST be masked") to lock in precision regressions.
3. Make a product decision for conversation transcripts: either redact at the export/display layer while storing the original, or structurally redact values at write time — explicitly pick one, test it, and eliminate the source/derivative asymmetry.

### TST-7 · The interactive modeling kernel still inherits the full host environment, asymmetric with the hardened plugin worker / batch notebook

**Impact:** Medium · **Effort:** M · **Verification:** — Not independently verified

**Problem.** H6 already narrowed the plugin/draft worker env to an allowlist (preventing untrusted code from reading OPENAI_API_KEY and the like), and the batch notebook also goes through the allowlist — but the live kernel path actually used for interactive modeling inherits the entire os.environ, with no setrlimit (only after-the-fact soft RSS killing). This is exactly the "weakest isolation path" flagged in the earlier review, and it is the path that runs the user's core modeling code.

**Evidence.** marvis/pipeline.py:1976-1989 defaults keep_alive=True and goes through NotebookExecutionSession; notebooks.py:266-290 `_build_client` uses `NotebookClient(...)` without passing env → jupyter_client starts the kernel with os.environ; by contrast the batch subprocess at notebooks.py:975-985 uses `_notebook_worker_env()` (allowlist, from line 1218) plus start_new_session, and plugins/runner.py:674-682 `_worker_env` is likewise an allowlist. The live kernel has only soft RSS monitoring (notebooks.py:584-620) — no setrlimit, no env filtering.

**Why it matters.** This is a consistency gap in INV-6 subprocess isolation: for the same category of data/modeling code execution, the most-used path has the weakest isolation, exposing the host's LLM/DB secrets to arbitrary code inside the kernel; it also creates the illusion of "looks hardened, but the main path was never hardened" relative to the already-hardened paths.

**How to fix.**
1. Extract a unified "execution guardrail contract" (env allowlist + PYTHONHASHSEED + setrlimit(RLIMIT_DATA/AS/CPU/FSIZE) + start_new_session + pgid tree-kill) shared by the live KernelManager and the subprocess/plugin workers.
2. Pass a filtered env to NotebookExecutionSession's KernelManager (jupyter_client's KernelManager supports an env parameter), apply setrlimit via preexec, and on close/timeout send SIGTERM→SIGKILL to the pgid.
3. On Windows, degrade to RSS soft monitoring (INV-9).

### TST-8 · No dependency CVE scanning in CI: a large ML/Java dependency stack with no pip-audit/safety/bandit

**Impact:** Medium · **Effort:** S · **Verification:** — Not independently verified

**Problem.** Dependencies are pinned with uv.lock including hashes (supply-chain integrity is OK), but there is no automated CVE monitoring at all. Parsers and native libraries like openpyxl/pyarrow/pypmml have historically had deserialization/parsing CVEs — and this product parses user-uploaded Excel/CSV/PMML. No CVE scanning means a known vulnerability could enter with no one alerted.

**Evidence.** .github/workflows/ci.yml runs only `uv sync --locked` + `uv run scripts/check`; scripts/check runs only pytest/ruff/node syntax/diff whitespace (per the option list at the top of the file); a repo-wide `grep pip-audit|safety|bandit` yields 0 hits. Dependencies include pypmml (pulls in a JVM), openpyxl, xgboost/lightgbm/catboost, pandas/pyarrow, etc. (pyproject.toml:8-40).

**Why it matters.** A credit-risk product will almost certainly face security/compliance audits, and "do you have dependency vulnerability management" is a mandatory audit item — especially relevant for an attack surface that parses user files.

**How to fix.**
1. Add `uv run pip-audit` (or `uv pip compile` + osv-scanner) to scripts/check (or a separate CI job) to scan the locked dependencies for CVEs, allowing an allowlist for already-assessed items.
2. Optionally add bandit for static security scanning of marvis/ (subprocess/eval/file writes).
3. Start as non-blocking warnings, then gradually make it blocking. Small effort, high audit value.

### TST-9 · Zero tests on concurrency paths: contention between 180ms high-frequency polling reads and long write transactions under WAL + 5s busy_timeout is uncovered

**Impact:** Medium · **Effort:** M · **Verification:** — Not independently verified

**Problem.** The product is single-user, but the runtime is naturally concurrent: the UI polling thread reads at high frequency, background modeling/JOIN holds long write transactions, the notebook monitoring thread runs, and subprocesses write back audit records. WAL allows read/write concurrency, but write-write contention and long transactions exceeding 5s will trigger `database is locked`. This path has no tests at all; the earlier review also listed "is the high-frequency polling read starved by a long write" as an unverified blind spot.

**Evidence.** marvis/db_schema.py:648-659 configures WAL + `busy_timeout=5000` + synchronous=NORMAL; the frontend polls agent messages every 180ms (earlier review, app.js:246), alongside long JOIN/notebook write transactions; `grep 'busy_timeout|OperationalError|database is locked' tests/` hits only app_security (the rest are HTTP-server threads, not DB concurrency). Not a single test concurrently reads and writes the same SQLite database.

**Why it matters.** A DB lock-contention regression would surface as a frozen UI or failed writes (lost audit/state), immediately visible in a single-process product; without tests there is nothing to intercept the regression.

**How to fix.**
1. Add a concurrency integration test: a background thread holds a >5s write transaction (simulating a long JOIN/modeling persist) while the main thread calls list_agent_messages/list_plans at high frequency, asserting the reads neither raise OperationalError nor fail within the busy_timeout.
2. Add a write-write contention case asserting backoff per busy_timeout rather than immediate failure.
3. While at it, verify that sqlite_health honestly reports wal_degraded on a degraded filesystem. Mark it `@pytest.mark.slow`.

---

## 11. Cross-Cutting Gaps (Completeness Critic)

**Lens verdict.** The 10-lens review already covers runtime correctness, agent intelligence, credit methodology, performance, and frontend experience in considerable depth, but nearly all of it stops at the "single-task runtime" perspective, missing several whole dimensions of "a product that is operated over the long term": ingestion defenses for real Chinese business data (encoding / long numeric ID precision), data lifecycle and privacy (raw credit data permanently residual after task deletion), the compliance delivery surface (audit log is write-only, no export), a column business-semantics layer (the data dictionary is absent across the entire V2 pipeline), the trust model under shared-host deployment, cross-task asset management (both a model registry and dataset reuse are missing), and the most basic operational surfaces such as backup / logging / LLM preflight. Most of these gaps are not arcane bugs — they are shortcomings that directly determine whether a bank risk-control team can actually adopt this as a production tool, and most of the fixes are S/M-sized.

### GAP-1 · CSV ingestion hardcodes UTF-8 with no dtype defenses: GBK files fail outright, and long numeric ID columns are corrupted by float64 precision truncation, breaking JOIN keys

**Impact:** High · **Effort:** M · **Verification:** — Not independently verified (critic sweep)

**Problem.** Two compounding problems: (1) CSVs exported from Chinese bank data warehouses / legacy Excel are very often GBK/GB18030-encoded; ingestion throws a UnicodeDecodeError outright, and the user only sees a 400 error whose message does not even contain the word "encoding". (2) Under pd.read_csv's default type inference, an 18-digit national ID column or a 16–19-digit card number column is read as float64 as soon as it contains any missing value (float64 has only ~15–16 significant digits), so the trailing digits of key values are silently rewritten. Subsequent JOINs then match on corrupted keys, surfacing as an abnormally low match_rate at the C2 gate or as some rows quietly failing to match, with no way for the user to attribute the cause.

**Evidence.** marvis/data/registry.py:271 `frame = pd.read_csv(source_path, encoding="utf-8-sig")` (no encoding fallback, no dtype parameter); the CSV branch of read_frame at marvis/data/backend.py:180-186 likewise hardcodes utf-8-sig; all CSV upload / material-import paths — routers/data.py:180 and agent/modeling_setup.py:349 among others — funnel into this single register_from_upload path.

**Why it matters.** This is the first gate of "can it ingest real business data at all": the target users (in-bank risk teams) will almost inevitably hit one of these two traps with their sample tables and credit-bureau detail files. The encoding failure blocks the first-run experience; the ID precision corruption is a data-correctness incident (the spirit of INV-3 is defeated at the ingestion layer — however rigorous the JOIN engine is, the keys are already broken on entry). The 10 review lenses only covered the memory/size guardrails of uploads and never touched encoding or dtype.

**How to fix.**
1. Encoding fallback chain: after utf-8-sig fails, try gb18030 (a GBK superset) and then latin-1 as the last resort; record the encoding ultimately used in the ingest report and display it on the dataset card. Alternatively, switch uniformly to DuckDB read_csv with an explicit encoding.
2. Key-fidelity defenses: first read with dtype=str, then convert to numeric only those columns that "round-trip losslessly" (str -> number -> str yields equality). At a minimum add a detector: whenever a float64 column's non-null values are all integer-shaped values ≥ 1e15, flag it in red in the C1/C2 gate tables as "suspected long ID truncated by float; recommend re-importing as text".
3. Regression tests: three fixtures — a GBK-encoded CSV, an 18-digit national-ID column containing missing values, and scientific-notation strings.

### GAP-2 · Task deletion only removes the tasks row and tasks directory: raw credit data files and orphan rows in plans/datasets/experiments etc. persist forever, and the deletion itself writes no audit record

**Impact:** High · **Effort:** M · **Verification:** — Not independently verified (critic sweep)

**Problem.** After the user clicks delete on a task: everything under workspace/datasets/<task_id>/ — the raw uploaded files, the per-sheet parquet files, the joined parquet (containing borrower-level detail) — remains on disk; the DB rows in datasets/plans/plan_steps/joins/experiments/strategies all become orphans; and delete_task writes no audit row at any point — all evidence of a modeling task is destroyed without any 'task.deleted' audit record.

**Evidence.** marvis/repositories/tasks.py:125-129 delete_task executes only `DELETE FROM tasks WHERE id=?`; marvis/db_schema.py:274(plans)/440(sub_agents)/455(datasets)/472(joins)/485(experiments)/517(strategies) — none of these tables has a FOREIGN KEY pointing at tasks (contrast with :186 jobs and :213 agent_messages, which have ON DELETE CASCADE); marvis/routers/tasks.py:153-164 only rmtree's settings.tasks_dir/task_id, whereas uploads / Excel conversions / JOIN artifacts are all written under settings.datasets_dir/task_id (settings.py:34-35 = workspace/datasets; routers/data.py:128/145, data/registry.py:301-302).

**Why it matters.** A triple problem: privacy (the user believes sensitive credit data has been deleted while the detail records persist long-term, violating the spirit of INV-5), storage leakage (on a single-machine product, disk usage only grows monotonically), and audit integrity (INV-8 requires side effects to be traceable, and "deleting an entire task" is one of the largest side effects there is). The reliability lens covered crash-residual .bak files, but nobody looked at "the normal deletion path is itself a leak source".

**How to fix.**
1. Change delete_task into a single-transaction multi-table delete: datasets, joins, plans (plan_steps/outputs/runs already cascade with plans), experiments (model_artifacts already cascade), strategies (backtests already cascade), sub_agents; within the same transaction write an audit row with kind='task.deleted' (audit rows themselves are retained, never deleted, preserving the traceability chain).
2. At the router layer, in addition to rmtree(task_dir), also rmtree(settings.datasets_dir/task_id); on failure degrade to a warning and leave it for the startup reconcile to clean up.
3. Add a regression test: after deleting a task, assert the datasets directory no longer exists, the six tables have no residual rows for that task_id, and audit contains a task.deleted record.

### GAP-3 · Audit log is write-only: INV-8 spent transactionalization cost across 20+ sites, yet no API/UI/export can read any of it

**Impact:** High · **Effort:** M · **Verification:** — Not independently verified (critic sweep)

**Problem.** The last two rounds of fixes transactionalized auditing for dozens of side-effect kinds across join/plan/draft/plugin/modeling/report (the 6-28 fix status listed 20+ items), but the audit table is a dead end: there is no per-task viewing endpoint, no frontend timeline, no CSV/JSONL export, and even the repository layer's query capability cannot answer "give me all audit records for this task" (target_ref cannot be filtered on, and there is no index).

**Evidence.** marvis/repositories/audit.py:45-68 _list_audit_rows supports only exact-kind filtering plus limit/offset — no target_ref / task / time-range filters; list_audit is defined only in repositories/plugins.py:133 and repositories/plans.py:832, and a repo-wide grep shows zero calls in marvis/routers/ and marvis/api*.py; the task evidence endpoint at routers/evidence.py:21-50 reads only the evidence JSON files and includes no audit data.

**Why it matters.** "Complete auditability" is one of this product's core selling points for the banking context; what model governance / internal audit needs is precisely the printable record of "for this model, who confirmed each step from data joining to delivery, and on what basis". Right now that value chain is only repaired up to the point of persistence — a large investment was made to build a black box with no reader. It also directly hurts development debugging: when something goes wrong, nobody can conveniently browse the audit trail.

**How to fix.**
1. Repository layer: add target_ref prefix filtering and an at time-range parameter to _list_audit_rows, and add CREATE INDEX ON audit(target_ref, at).
2. Add GET /api/tasks/{task_id}/audit?kind=&after= (rows whose target_ref conventionally embeds the task_id prefix are fetched by prefix, with a fallback match on task_id inside detail_json) and GET /api/tasks/{task_id}/audit/export (streaming JSONL/CSV download).
3. Frontend: add an "审计轨迹" ("audit trail") collapsible panel to the task's right-hand column, rendering kind icon + actor + outcome grouped in reverse-chronological order, with confirmation-gate decision rows highlighted.
4. Longer term: the model report appendix automatically embeds key audit excerpts for the task (gate confirmations, JOIN execution, training, delivery).

### GAP-4 · Data dictionary / column business-semantics layer absent across the entire V2 pipeline: the screening gate, reports, and LLM context all see only bare column-name codes

**Impact:** High · **Effort:** M · **Verification:** — Not independently verified (critic sweep)

**Problem.** The V1 validation flow scans and consumes a data dictionary, but the V2 JOIN→FEATURE→MODELING mainline has nothing end-to-end except bare column-name codes like als_m3_id_nbank_orgnum: the feature screening gate shows hundreds of rows with no business meaning (the UX lens only mentioned search/sort, never semantics), the model report's univariate sheet cannot be delivered (a model committee requires the business definition of each column), and the weak LLM gets none of this cheapest-possible context of "what is this column" during slot probing and gate decisions.

**Evidence.** marvis/routers/data.py:37 `DATASET_ROLES = {"sample", "feature", "derived", "unknown"}` has no dictionary role; the datasets table at db_schema.py:455-470 has only a columns_json column-name list and no column-description field; a data dictionary exists only in the V1 validation flow (domain.py:45 FileRole.DATA_DICTIONARY, pipeline.py:317-322); the only "dictionary" on the V2 side is the handoff-time file generated from feature names plus the fixed placeholder "建模特征" ("modeling feature") in packs/modeling/handoff.py:370-374 _write_dictionary.

**Why it matters.** Credit-bureau / third-party variables routinely run to hundreds or thousands of columns; without a semantics layer, "the user makes decisions by looking at the screening gate" is effectively an impossible task — they can only confirm blindly, which directly hollows out the point of mandatory confirmation gates. Filling this in improves three things users care about at once: gate-decision quality (both humans and the LLM can understand what they see), report professionalism (it can pass review), and the agent appearing smarter (semantics carried in the prompt).

**How to fix.**
1. Data side: allow each task to upload a "dictionary" file (a CSV/Excel with three or four columns: column name / business name / description / category), joined onto dataset profiles by column name and stored either as a new column_meta_json column on the datasets table or as a sidecar JSON.
2. Consumption side, wired in at three places: add a "业务名称" ("business name") column with description tooltips to the feature screening gate table; carry business names in the univariate sheets of feature/model reports; inject a compact "candidate columns: code=business name" mapping into slot-probing and gate-decision prompts (truncated to a length limit, read-only so determinism is untouched, honoring INV-1/INV-4).
3. Change the handoff's _write_dictionary to pass through the real dictionary (use it when present, fall back to the placeholder when absent).

### GAP-5 · Full loopback trust under shared-host deployment: the CLI actively suggests JupyterHub usage, yet any other user on the same machine can operate the platform with full privileges (including installing plugins = arbitrary code execution)

**Impact:** Medium · **Effort:** S · **Verification:** — Not independently verified (critic sweep)

**Problem.** The assumption "local means trusted" holds on a single-person laptop, but the JupyterHub deployment shape the product itself advertises is precisely a multi-user shared server: any other logged-in user's process on that server can reach 127.0.0.1:8000, is judged a trusted local client by _is_local_client, and can read all borrower-level detail, delete tasks, and install plugins (plugins execute arbitrary code under the service account). Meanwhile the two remote-access environment variables are entirely undocumented, so users have no way to configure them correctly.

**Evidence.** marvis/app.py:81-93 _is_local_client returns True for any loopback peer, thereby granting all non-safe-method permissions (including POST /api/plugins install, task deletion, data reads); marvis/__main__.py:145 prints at startup "If running behind JupyterHub, try the matching /proxy/<port>/ URL."; MARVIS_ALLOW_REMOTE_READ and MARVIS_TRUSTED_PROXY_HOSTS (app.py:70-71) have 0 grep hits in docs/ and README.

**Why it matters.** The target deployment (in-bank analytics server / bastion host) is almost certainly a multi-account Linux machine. This does not overturn the established "single-machine single-user trust boundary" decision — rather, that decision's applicability condition (exclusive host) contradicts the deployment mode the product itself advertises (shared host); the testing-security lens covered redaction/CVEs but not this trust-model fissure.

**How to fix.**
1. Add an optional MARVIS_ACCESS_TOKEN: when set, middleware validates Authorization: Bearer or ?token= (a cookie is written after first opening a URL carrying the token); when unset, current behavior is preserved. The startup log warns when multi-user indications are detected.
2. Add a "shared server deployment" section to the runbook: explain the combined usage and risks of the token, MARVIS_TRUSTED_PROXY_HOSTS, and MARVIS_ALLOW_REMOTE_READ.
3. More thorough option (later): support --uds to listen on a Unix socket with 0700 permissions, naturally isolated by file permissions.

### GAP-6 · No model asset registry: experiments do not have a single HTTP endpoint, and cross-task "which models have I delivered" is completely invisible

**Impact:** Medium · **Effort:** M · **Verification:** — Not independently verified (critic sweep)

**Problem.** Trained models (the experiments / model_artifacts tables) can only be seen indirectly through the owning task's plan evidence panel: there is no within-task experiment-list API, let alone a cross-task view. After a few months of use, dozens of tasks and hundreds of experiments will accumulate, and the most basic asset-management questions — "what was the OOT KS of that LGB version we delivered for consumer loans last quarter", "which artifact is running in production right now" — cannot be answered except by manually paging through the task list.

**Evidence.** grep 'experiment' yields 0 hits across marvis/routers/*.py, marvis/api*.py, and marvis/static/app.js; marvis/repositories/modeling.py:50 `def list_experiments(self, task_id: str)` supports only within-task listing, and is consumed only by the tool layer at packs/modeling/experiment.py:98.

**Why it matters.** This is the core instance of the "multi-project management" blind spot: models are the platform's ultimate output asset, yet there is no asset catalog. It is also complementary to the memory lens's historical-KS anchors — those are experience for the LLM, this is a deterministic ledger for humans (a pure tool query, naturally honoring INV-1).

**How to fix.**
1. Repository layer: add list_experiments_all(status=None, limit/offset) (JOIN tasks to fetch task name / scenario).
2. Add two tiers of endpoints: GET /api/experiments?status=trained&scope=all and GET /api/tasks/{id}/experiments.
3. Frontend: add a "模型资产" ("model assets") entry to the welcome page — a table listing task / recipe / training time / train-test-OOT KS / champion flag / handoff-validated flag, with row clicks navigating to the task; reuse the existing metric rendering components.
4. Later, this becomes the vehicle for the scoring and monitoring entry points proposed by the credit-domain lens — the "last mile" carrier.

### GAP-7 · Datasets are copied into per-task silos: the same base table is re-uploaded, re-written to parquet, and re-profiled for every task, with no content-fingerprint reuse

**Impact:** Medium · **Effort:** M · **Verification:** — Not independently verified (critic sweep)

**Problem.** The actual working pattern of a credit team is that the same batch of base tables (sample table, bureau-derived tables, third-party data tables) is reused across many modeling/analysis tasks. Right now, opening each new task means re-uploading the same several-hundred-MB file (the performance lens already noted uploads are read fully into memory), copying another parquet, and running full profiling again; disk usage grows linearly with the number of tasks, and there is no way to confirm across tasks "are we using the same version of the data".

**Evidence.** marvis/data/registry.py:301-302 _dataset_dir = datasets_root/task_id; register_from_upload (registry.py:28-67) writes a fresh parquet (uuid-named) for every upload and runs profile_dataset + sample_rows + target-column probing; the datasets table (db_schema.py:455-470) has no content hash field, and the repo has no cross-task dataset query/reuse entry point anywhere.

**Why it matters.** This is a double tax on efficiency and definitional consistency: repeated waiting (upload + profiling take minutes) directly lengthens every task's cold start; and "did two tasks use the same version of the sample table" is precisely the precondition for model comparison and memory anchors (historical KS references) to be valid — currently it can only be guessed from filenames.

**How to fix.**
1. Compute the file's sha256 at upload time and store it in a new content_hash column on the datasets table (for Excel, hash the converted parquet content).
2. When the hash matches an existing dataset, skip the parquet rewrite and profiling and simply insert a new dataset row for the new task referencing the same source_path (read-only parquet sharing is safe; the invariants are unaffected — reusing profiling results by hash is deterministic).
3. Frontend upload dialog shows on a hit: "检测到与任务 X 的数据集内容一致，已复用" ("detected content identical to task X's dataset; reused").
4. Later evolve into a standalone "dataset library" page (listing hash / earliest importing task / number of referencing tasks), combined with GAP-2's deletion cleanup to do reference counting.

### GAP-8 · Zero LLM configuration preflight: saving always succeeds, /api/health excludes the LLM, there is no test-connection entry, and first-run failures can only be stumbled into inside the agent conversation

**Impact:** Medium · **Effort:** S · **Verification:** — Not independently verified (critic sweep)

**Problem.** Mistyped base_url, port, or model name for the local LLM stack (LM Studio / vLLM / Ollama) is the most common first-run failure, yet a wrong configuration can currently be saved successfully, and only surfaces once the user launches their first agent turn — via indirect symptoms like "routing degraded / critique unavailable / planning failed" (the agent-intelligence lens already noted these degradations are themselves silent). The user cannot distinguish "the model is dumb" from "the model was never connected at all".

**Evidence.** marvis/api_settings.py:82-90 update_llm_settings directly calls save_llm_settings with no connectivity or model-name validation whatsoever; marvis/app.py:228-230 /api/health returns only sqlite_health; a frontend grep for "测试" ("test") in app.js hits only "压力测试" ("stress test") at app.js:2894 — the LLM settings panel has no test button; llm_client.py:28-34 validates the URL format only at actual call time.

**Why it matters.** This is the first-mile experience of a product aimed at weak-model local deployment: every new user, and every model switch, has to pass through this gate. The llm-client lens covered observability and retries, but not "active preflight at configuration time" — the cheapest possible loss-stopper.

**How to fix.**
1. Add POST /api/settings/llm/test: using the profile in the payload, send a 'ping' chat completion with max_tokens=8 (5s timeout), returning {ok, latency_ms, model_echo, error_detail}; connection-refused / 404 / model-not-found each get an actionable Chinese hint ("检查服务是否启动/检查端口/检查模型名" — "check whether the service is running / check the port / check the model name").
2. Frontend LLM settings panel gains a "测试连接" ("Test connection") button showing a result badge, and a test is auto-triggered after a successful save.
3. /api/health gains an llm section (caching the most recent test result — do not hit the LLM in real time inside health).

### GAP-9 · Zero backup/restore story: the single SQLite copy (WAL mode) plus workspace files have no backup command or documentation, and a naive cp still has a data-loss window

**Impact:** Medium · **Effort:** S · **Verification:** — Not independently verified (critic sweep)

**Problem.** All tasks, audit records, experiments, and memory live in a single SQLite database plus a file tree under workspace/, and the product is positioned for long-term single-machine use. The user has no supported way to back up or relocate the workspace: the docs say nothing; a technically-inclined user will cp workspace/ while the service is running, losing the most recent transactions not yet checkpointed out of the WAL and getting a copy that "looks complete but is missing its tail". In a machine-switch / disk-failure / accidental-deletion scenario, months of modeling assets and the audit chain vanish in one stroke.

**Evidence.** grep -rni backup yields 0 hits in docs/*.md and README*.md; the repo contains no sqlite3 backup API calls anywhere; db_schema.py:649 forces journal_mode=WAL (naively copying the db file while running loses transactions not yet checkpointed from the WAL); runbook.md has install / upgrade / multi-worktree sections but no backup/migration section.

**Why it matters.** The value of the audit chain (INV-8) depends on its survivability — an audit that cannot be backed up is an audit that can be lost. This is also the lowest-cost, highest-safety-net item in the deployment-ops blind spot.

**How to fix.**
1. Add `marvis backup [--workspace ./workspace] [--output marvis-backup-<date>.tar.gz] [--include-datasets]`: first use sqlite3 Connection.backup() to produce a consistent snapshot into a temp file, then archive the tasks/plugins/report_templates/branding directories (datasets excluded by default, optionally included), and write a manifest (version + timestamp + file inventory).
2. Restore = unpack into a new workspace and serve directly (the existing startup reconcile can already handle leftovers).
3. Add a "backup & migration" section to the runbook, explicitly warning about the WAL risk of a naive cp while running.

### GAP-10 · Server-side logging infrastructure absent: only 7 of 254 modules have a logger, no logging configuration, no on-disk log file — nothing to consult when users report problems

**Impact:** Medium · **Effort:** S · **Verification:** — Not independently verified (critic sweep)

**Problem.** The architecture lens pointed out that pipeline.py is a single file with 0 logging, but the problem is global: the entire backend has no logging initialization, the vast majority of modules do not even have a logger, and uvicorn's default logs go only to the terminal that launched it — close the terminal and all history is gone. When a user reports "task failed inexplicably / agent stalled", the developer has no server-side log to ask for and can only rely on reproduction.

**Evidence.** grep getLogger under marvis/ hits only 7 files (api_task_helpers.py, db_schema.py, api.py, routers/tasks.py, routers/plans.py, agent_memory/api_support.py, plugins/hooks.py); the repo has no logging.basicConfig/FileHandler/dictConfig anywhere; _serve at marvis/__main__.py:132-146 calls uvicorn.run without a log_config, so logs go only to terminal stdout.

**Why it matters.** For a single-machine product delivered to non-developers and running on the user's own machine, on-disk logs are practically the only remote troubleshooting channel; it is also the lowest-cost safety net for the reliability lens's "zero observability during driver-turn execution" (get logs first, then talk about events and panels).

**How to fix.**
1. Initialize logging in create_app or _serve: a RotatingFileHandler writing workspace/logs/marvis.log (10MB × 5), root at INFO with marvis.* at DEBUG, adjustable via environment variable; pass log_config to uvicorn.run so access/error logs land on disk in the same stream.
2. Add stage-boundary logger.info/exception calls to the core modules (pipeline, plan_driver, executor, join_engine, runner, llm_client) — merge this work with the architecture lens's pipeline-logging item.
3. Frontend settings page gains an "open log directory / download recent logs" entry for one-click evidence collection when reporting issues. Note: log writes must pass through marvis.redaction for scrubbing first (honoring INV-5).

---

## Appendix

### A. Methodology

1. **Phase 1 — parallel lens reviews.** Ten specialist lenses (agent intelligence, memory, credit-risk domain, LLM client, performance, UX, visual design, architecture, reliability, testing & security) each deep-read the relevant source in the current working tree. Ground rules: read the 2026-06-21 / 2026-06-28 review status sections first and do not re-report confirmed-fixed items; every finding needs current `file:line` evidence; each finding carries a falsifiable `verify_claim` where applicable; max 15 findings per lens, depth over breadth.
2. **Phase 2 — adversarial verification.** Every high/critical finding with a falsifiable claim was handed to an independent skeptic agent instructed to *refute* it: re-locate evidence by content (not stale line numbers), and actively search for backstops (another module, a helper, a covering test) that would invalidate the claim. Verdicts: CONFIRMED / PARTIAL / REFUTED. Several verifiers ran empirical checks (e.g., LightGBM NaN-label training behavior in the canonical `py_313` environment; a minimal JOIN key-space reproduction).
3. **Phase 3 — completeness critic.** A final agent reviewed the full finding list against the request ("any aspect: efficiency, accuracy, agent intelligence, UX, visuals, professionalism, business value, bugs") and swept for missed whole dimensions, contributing the GAP findings (not adversarially verified).
4. **Report assembly.** Findings were translated and organized by theme; the executive summary, priority ranking, quick wins and roadmap were synthesized across lenses.

Run statistics: 73 review/verify/critic sub-agents (~5.3M tokens, ~1,700 tool calls) across two runs (the first run was interrupted mid-flight by a session usage limit; the second run completed the remaining five lenses, all 38 verifications, and the critic against the same working tree).

### B. Verification statistics

| Verdict | Count | Meaning |
|---------|-------|---------|
| ✅ CONFIRMED | 30 | Claim held under adversarial re-derivation from current source; several with empirical reproduction |
| ⚠️ PARTIAL | 8 | Substantially correct; verifier corrected details or narrowed scope (noted inline) |
| ❌ REFUTED | 0 | — |
| — Not independently verified | 77 | Medium/low impact or purely directional proposals (including the 10 GAP critic findings) |

A zero refutation rate across 38 adversarial checks suggests the confirmed findings can be acted on without much re-litigation; the PARTIAL notes are worth reading before implementing those specific items.

### C. Items re-checked and confirmed genuinely fixed (not re-reported)

Reviewers spot-checked a sample of 6-28 "fixed" claims against the current tree and confirmed these landed for real (partial list, as noted by the lenses):

- JSON extraction + retry in `decide_gate` / `route_instruction` / planner / reviewer (no longer bare `json.loads` on those paths; AGT-10 notes three planner sub-paths still pending).
- Negation guard in confirmation/dedup instructions (the *question-form* bypass is the AGT-1 residue).
- Machine red-flag checklist injected into JOIN/screen gate prompts; `success_criteria` machinery in plan/DB/reviewer (the *empty-tuple templates* are the AGT-4 residue).
- Sub-agent inner plans only report success on `DONE`; paused/review states surface as paused/failure.
- Scorecard points ↔ scorer consistency, default monotonic binning, WOE-space feature selection with coefficient-sign warnings, calibration tool, model card, monitoring policy JSON, challenger package, approval package.
- Message polling `after_id` incremental cursor; JOIN match-rate scan pushed down to DuckDB; WAL + busy_timeout; notebook subprocess isolation + env allowlist + psutil RSS soft-monitor with process-tree kill.
- Transactional audit writes across JOIN / draft / plugin / strategy / modeling / report-override repositories (the 15 `getattr(*_with_audit)` soft probes are the ARCH-3 residue).
- `api.py` reduced to 922 lines with zero routes remaining; routers/repositories/helpers split is directionally sound (the `legacy_api` service-locator is the ARCH-1 residue).

### D. Coverage limits

- Medium/low findings and all GAP findings were not adversarially re-verified; treat their `file:line` evidence as reviewed-once.
- Line numbers reference the working tree of 2026-07-02 (branch `codex/v2-plugin-tool-runtime`, HEAD `a3c32a23` + staged/untracked work); they will drift.
- The review is static-analysis-first: apart from targeted empirical checks (LightGBM NaN behavior, JOIN key-space repro, subprocess cold-start timing), no full test-suite run or live-server session was part of this round.
- Frontend findings were derived from source reading, not from a driven browser session; visual findings should be validated against a running UI before large CSS refactors (per the project's own "mockup first, then implement" preference).

### E. Related documents

- `docs/reviews/2026-06-21-v2-full-code-review.md` — correctness-focused deep review (baseline).
- `docs/reviews/2026-06-28-v2-runtime-deep-review.md` — runtime deep review + fix-status ledger.
- `docs/reviews/2026-06-28-v2-improvement-proposals.md` — prior improvement round + landed-status ledger.
- `docs/plans/v2-completion-plan.md`, `docs/plans/*-spec.md` — governing plans/specs referenced by findings.

---
