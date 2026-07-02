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

1. **Three incomplete fixes from earlier reviews** (adversarially confirmed): `is_confirm` still treats questions as confirmation and short-circuits before LLM routing (AGT-1); `tune_hyperparameters` still tunes silently on NaN labels (DOM-1); JOIN uniqueness/dedup computed in raw key space vs matching in transformed key space (PERF-2, reproduced).
2. **Four criticals**: concurrent `PlanExecutor.run` via double-confirm/second tab mis-fails running steps (REL-1); heavy sync work inside `async def` endpoints freezes the service during big-table ops (PERF-1); manual-mode gate confirm runs long turns with zero feedback and no stop (UX-1); memory ↔ V2 fully disconnected (MEM-1).
3. **Modeling methodology gaps vs "same sample, same label → KS ceiling" bar** (8 confirmed-high, backlog §3): default flow has no multivariate feature selection; only LGB is tuned (12 random trials); default split builds no OOT (`oot_by_time` is dead code); FEATURE-stage fit-transforms lack train-only discipline; preprocessors are not persisted for scoring replay; categorical features silently dropped.
4. **Known limitation recorded per DoD-7**: raw typed manual replies have no browser-side stale-token; fix scheduled with AGT-1 (backlog PR-6).
5. `cum_bad_rate` in `marvis/validation/vintage.py` is per-MOB, not cumulative (NEW-1; known since 06-21, scheduled in backlog stage 1).

Open decisions resolved for v1 (backlog §1 PR-5): AUTO stays bounded low-risk; subprocess sandbox is the v1 endstate (OS-level containment tracked long-line); `.pkl` is the source of truth with `.pmml` as compatibility artifact; visual work proceeds via token consolidation, taste-level changes gated on user-approved mockups.

## Merge stance

This is an **intermediate** PR: the branch is coherent, fully tested, and substantially hardened, but it is not "V2 complete" (see backlog DoD). Direct release from this merge is not claimed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
