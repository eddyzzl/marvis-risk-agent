# Intermediate PR: V2 Platform — plugin/tool runtime, driver workflows, modeling depth

Branch: `codex/v2-plugin-tool-runtime` → `main`

## What this PR contains

The V2 platform in its current state (~475 files changed vs `main`, +120k lines): plugin/pack tool runtime with subprocess isolation, template→validator→executor plan orchestration with mandatory confirmation gates, the conversational PlanDriver for JOIN / feature-analysis / modeling / strategy / vintage task types (agent + manual modes), the deterministic JOIN engine, modeling packs (LGB/XGB/CatBoost/LR/scorecard/MLP recipes, tuning, calibration, model cards, monitoring policy, approval/challenger packages, PMML/PKL delivery, validation handoff), agent memory subsystem, transactional artifact staging with same-transaction audit writes, and the V2 frontend (task workspace, plan rail, gate controls, tokenized theming).

Key hardening landed since the 2026-06-28 reviews (verified fixed by the 2026-07-02 review, see its Appendix C): JSON extraction+retry on LLM touchpoints, negation guard in confirmations, red-flag checklists on JOIN/screen gates, sub-agent success semantics, scorecard points↔scorer consistency, monotonic binning default, WOE-space selection, calibration tool, reject inference, `after_id` incremental message polling, DuckDB match-rate pushdown, notebook RSS soft-monitoring with process-tree kill, transactional audit across JOIN/draft/plugin/strategy/modeling/report repositories, `api.py` decomposed to 922 lines with routers/repositories split.

## Verification

- `scripts/check` (git diff --check, ruff, node --check on all static JS, full pytest) from the committed tree `5f6bb17c`: **PASS — 1988 passed, 4 skipped, 2 warnings (sklearn MLP ConvergenceWarning) in 8m37s** (2026-07-02).
- Canonical environment: `py_313` (`/opt/miniconda3/envs/py_313/bin/python`).
- Manual smoke (six journeys, live server + synthetic credit data, manual mode): **ALL PASS** (2026-07-02) — JOIN 2000==2000 rows (INV-3); feature report on disk; modeling gates G2–G5 through selected experiment (test_ks 0.31); `.pmml` (valid PMML 4.4) + `.pkl` on disk; validation handoff task + 5 material files; forced failure recovered via step retry endpoint. Two product findings recorded in the backlog: `time_col` alone does not trigger a time-based split (falls back to random 75/25 — confirms SEL-1), and step-retry `inputs` are full-replacement rather than merge semantics.

## Remaining risks and known limitations (tracked, not blocking this intermediate PR)

Single source of truth for all remaining work: [`docs/plans/v2-master-backlog.md`](../plans/v2-master-backlog.md) (179 items, staged; DoD for "V2 complete" in its §0). Highest-priority open items:

**Landed after the gate run above (backlog stage 1, all merged with targeted regressions):** the three recurrence fixes (AGT-1 question-guarded anchored `is_confirm`; DOM-1 NaN-label gate in tuning; PERF-2 transformed-key-space uniqueness/dedup), the long-standing fake-cumulative `cum_bad_rate` (NEW-1), audit soft-probe removal (ARCH-3), cross-task message bleed guard (UX-3), duplicate sample-primary rejection (UX-7), champion-by-test-KS (DOM-9), report score-column hard error (DOM-10), atomic env config + sync join guard (REL-8/9), and a smoke-discovered join+no-split-column 409 (NEW-3). A follow-up full `scripts/check` on the merged tree is the stage-1 exit gate.

Still open (highest priority):

1. **Four criticals**: concurrent `PlanExecutor.run` via double-confirm/second tab mis-fails running steps (REL-1); heavy sync work inside `async def` endpoints freezes the service during big-table ops (PERF-1); manual-mode gate confirm runs long turns with zero feedback and no stop (UX-1); memory ↔ V2 fully disconnected (MEM-1).
2. **Modeling methodology gaps vs "same sample, same label → KS ceiling" bar** (backlog §3): default flow has no multivariate feature selection; only LGB is tuned (12 random trials); default split builds no OOT (`oot_by_time` is dead code — corroborated live by the smoke run); FEATURE-stage fit-transforms lack train-only discipline; preprocessors are not persisted for scoring replay; categorical features silently dropped.

Open decisions resolved for v1 (backlog §1 PR-5): AUTO stays bounded low-risk; subprocess sandbox is the v1 endstate (OS-level containment tracked long-line); `.pkl` is the source of truth with `.pmml` as compatibility artifact; visual work proceeds via token consolidation, taste-level changes gated on user-approved mockups.

## Merge stance

This is an **intermediate** PR: the branch is coherent, fully tested, and substantially hardened, but it is not "V2 complete" (see backlog DoD). Direct release from this merge is not claimed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
