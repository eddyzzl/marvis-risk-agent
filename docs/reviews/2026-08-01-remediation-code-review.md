---
reviewed: 2026-08-01T10:58:00+08:00
depth: deep
status: issues_found
remediation_status: closed_in_current_worktree
remediation_record: 2026-08-01-comprehensive-audit-remediation.md
files_reviewed: 68
files_reviewed_list:
  - marvis/app.py
  - marvis/db_schema.py
  - marvis/repositories/tasks.py
  - marvis/api_schemas.py
  - marvis/routers/validation_agent.py
  - marvis/agent/validation_app_service.py
  - marvis/agent/turn_handlers.py
  - marvis/agent/portfolio_setup.py
  - marvis/agent/strategy_workflows/__init__.py
  - marvis/agent/strategy_workflows/_analytics.py
  - marvis/agent/strategy_workflows/_catalog.py
  - marvis/agent/strategy_workflows/_model_score_comparison.py
  - marvis/agent/strategy_workflows/contracts.py
  - marvis/data/authenticated_snapshot.py
  - marvis/governed_json.py
  - marvis/spreadsheet_safety.py
  - marvis/packs/labeling/__init__.py
  - marvis/packs/labeling/contracts.py
  - marvis/packs/labeling/manifest.json
  - marvis/packs/labeling/tools.py
  - marvis/packs/strategy/model_score_comparison_tools.py
  - marvis/packs/strategy/pool_evidence_verifier.py
  - marvis/orchestrator/templates/portfolio.py
  - marvis/operations/__init__.py
  - marvis/operations/contracts.py
  - marvis/operations/integration.py
  - marvis/operations/notifications.py
  - marvis/operations/repository.py
  - marvis/operations/router.py
  - marvis/operations/scheduler.py
  - marvis/operations/schema.py
  - marvis/production_governance/__init__.py
  - marvis/production_governance/errors.py
  - marvis/production_governance/repository.py
  - marvis/production_governance/router.py
  - marvis/decision_twin/__init__.py
  - marvis/decision_twin/_canonical.py
  - marvis/decision_twin/artifacts.py
  - marvis/decision_twin/comparison.py
  - marvis/decision_twin/contracts.py
  - marvis/decision_twin/reconciliation.py
  - marvis/decision_twin/replay.py
  - marvis/static/js/v2/labeling_setup_panel.js
  - marvis/static/js/v2/portfolio_setup_panel.js
  - marvis/static/app.js
  - tests/test_decision_twin_artifacts.py
  - tests/test_decision_twin_comparison.py
  - tests/test_decision_twin_manifest.py
  - tests/test_decision_twin_reconciliation.py
  - tests/test_decision_twin_replay.py
  - tests/test_frontend_labeling_setup_panel.py
  - tests/test_frontend_portfolio_setup_panel.py
  - tests/test_frontend_strategy_candidate_lab_pool_operations.py
  - tests/test_labeling_workflow.py
  - tests/test_operations_api.py
  - tests/test_operations_contracts.py
  - tests/test_operations_integration.py
  - tests/test_operations_notifications.py
  - tests/test_operations_repository.py
  - tests/test_operations_scheduler.py
  - tests/test_portfolio_agent_wiring.py
  - tests/test_portfolio_api.py
  - tests/test_portfolio_setup.py
  - tests/test_production_governance_api.py
  - tests/test_production_governance_schema.py
  - tests/test_strategy_model_score_comparison.py
  - tests/test_strategy_model_score_comparison_vertical_slice.py
  - tests/test_strategy_workflow_spec.py
findings:
  critical: 3
  warning: 2
  info: 0
  total: 5
---

# Remediation Code Review — historical finding snapshot

**Scope:** current uncommitted remediation boundary only; root `REVIEW.md` was not read or changed.

> `status: issues_found` and the 3 Critical / 2 Warning counts above describe the
> review-time snapshot. Current closure is recorded in the follow-up at the end;
> do not interpret the historical frontmatter as the current worktree verdict.

## Summary at review time

The scheduler, request schemas, and most artifact write paths use sensible role, transaction, and optimistic-lock controls. However, three high-risk flows still let a confirmed or recorded result claim stronger evidence than the runtime has authenticated: portfolio analysis is not pinned to source bytes, label construction retains a source-file TOCTOU despite the newly added authenticated-snapshot primitive, and production activation accepts self-asserted deployment/health evidence.

The direct relevant test suite passed (149 tests), the additional Candidate Lab pool-operation test passed (7 tests), Ruff passed for the reviewed Python source, and `git diff --check` was clean. Those checks do not cover the adversarial races and false-evidence cases below.

## Critical Issues

### CR-01: Portfolio confirmation is not bound to an immutable dataset version

**File:** `marvis/agent/portfolio_setup.py:54-69, 155-157, 192-215`; `marvis/agent/turn_handlers.py:1727-1772`; `marvis/orchestrator/templates/portfolio.py:26-40, 43-122`

**Issue:** `PortfolioProposal` and the persisted `portfolio_states` gate contain only `dataset_id`, column names, and user semantics. They do not carry `content_hash`, workspace revision, or a source snapshot. The setup reads `registry.resolve_path()` and then the live file. On the confirmation turn, `turn_handlers` reconstructs the proposal entirely from that unversioned message state and passes only `performance_dataset_id` to every analysis tool. A replacement of the registered file (or a dataset registry/path change) after human state-order confirmation therefore makes the plan analyse different bytes under the old confirmed state order, LGD, and loss-state evidence. No final source reauthentication closes this gap.

**Fix:** Make portfolio setup source-bound like the labeling request: capture `{dataset_id, expected_content_hash, workspace_revision, analysis_generation}` in the proposal/gate/template slots; use `resolve_verified_path` plus `read_authenticated_parquet_snapshot` (or an equivalent retained-descriptor reader); and revalidate all bindings in each execution tool's transaction before publishing output. Add an E2E test that mutates/replaces the dataset after the state gate and expects the confirmation/execution to fail closed.

### CR-02: Label construction bypasses the authenticated source snapshot and can publish a result from swapped bytes

**File:** `marvis/packs/labeling/contracts.py:261-280`; `marvis/packs/labeling/tools.py:67-91, 323-360`

**Issue:** The remediation adds `read_authenticated_parquet_snapshot`, but no labeling code calls it. Proposal generation verifies/resolves the path and then separately calls `column_names` and `read_frame`; execution repeats the same pattern. The commit-time check verifies the current path only after the derived Parquet/CSV have already been built. An attacker or concurrent writer can replace the source between `resolve_verified_path()` and `read_frame()`, let label construction consume the substituted contents, then restore the original bytes before `_require_source_and_workspace_on_connection()`. The final evidence will state the original `expected_content_hash`, while the registered output and its maturity/quality metrics were computed from different data.

**Fix:** Read the required Parquet columns once through `read_authenticated_parquet_snapshot(path, root=runtime.datasets_root, expected_sha256=request.expected_content_hash, columns=...)`; derive both proposal maturity and labels from that returned private frame. Preserve the source identity from that authenticated read and recheck the workspace/registry transaction before publish. Add a deterministic hook/race test that swaps the source between resolution and read, and assert no derived dataset, artifact, or audit row is committed.

### CR-03: A production activation can be recorded from arbitrary manifest and health claims

**File:** `marvis/production_governance/repository.py:120-175, 390-405, 454-486`; `marvis/production_governance/router.py:51-58, 177-197`

**Issue:** `manifest_hash`, `external_deployment_ref`, `health_evidence_ref`, and `health_status="healthy"` are accepted from request bodies. Creation stores any syntactically valid manifest hash; activation only compares `observed_manifest_hash` with that same caller-supplied stored value, and then persists the two opaque references. No registered manifest is loaded and hashed, no deployment adapter is queried, and no health evidence is verified against the strategy content hash/environment. A maker/checker/admin sequence can consequently produce an immutable audit trail that says a particular strategy is active in production even when the referenced deployment, manifest, or healthy result never existed or describes different bytes.

**Fix:** Replace free-text activation evidence with server-resolved, immutable records: register a deployment manifest whose canonical content includes strategy id/version/content hash; require an allowlisted deployment/health verifier to return a signed or content-addressed result; compare the verifier result to the registered manifest and target environment inside the activation transaction; persist the verified artifact ids/hashes, not caller strings. Test that fabricated but matching body hashes/URIs are rejected.

## Warnings

### WR-01: A failed queued-to-running transition leaks the active driver job until watchdog cleanup

**File:** `marvis/agent/validation_app_service.py:309-315`; `marvis/repositories/tasks.py:615-634, 677-807`

**Issue:** `start_job()` inserts an active `queued` row. If `mark_job_running()` returns `False` (its contract explicitly permits cancellation, watchdog recovery, or another callback to have moved the row), the service raises immediately before installing its `try/finally` and never calls `finish_job`. If the state transition was not already terminal, the task remains behind `idx_jobs_active_task` and subsequent user actions return 409 until the delayed heartbeat watchdog releases it. This is precisely a failure path that is not synchronously closed or given a task-visible reason.

**Fix:** Enter a cleanup guard immediately after `start_job`; on a false mark result, read the job status and call a compare-and-set terminal failure/cancellation transition when still queued/running, then return a typed conflict explaining the actual state. Add a test monkeypatching `mark_job_running` to return `False` and assert there is no active job afterward.

### WR-02: Decision-twin replay hashes are self-asserted, not authenticated against stored artifacts or trusted adapter code

**File:** `marvis/decision_twin/replay.py:490-556`; `marvis/decision_twin/contracts.py:22-58, 69-106`

**Issue:** `ReplayEngine` validates timestamp ordering and hash syntax, but it accepts caller-constructed `ReplayManifest`, `ReplayFacts`, every field `source_sha256`, and the adapter's claimed `TrustedAdapterIdentity`. It neither resolves those hashes from `ContentAddressedAuditStore` nor verifies that facts match their asserted source bytes; `input_sha256` is only a hash of the already trusted caller object. Thus a consumer can create a formally valid, point-in-time replay report for fabricated facts and a substituted adapter while retaining a plausible lineage record. The component is presently a library rather than an HTTP path, so this is a future-use boundary failure rather than an exposed remote bypass.

**Fix:** Provide a repository-backed replay entry point that accepts only immutable artifact receipts/ids, loads and verifies all manifest/fact/adapter materials from the audit store, and checks the adapter binary/package digest against an allowlist before invoking it. Reject direct arbitrary dataclass inputs at the production-facing boundary and add forged-hash/adaptor substitution tests.

---

_Review depth: deep_
_Source files modified: none_

## Remediation follow-up — 2026-08-01

This file preserves the original independent findings above as forensic review
evidence. The five original findings are no longer open in the current worktree:

| Finding | Closure |
|---|---|
| CR-01 Portfolio source binding | Proposal/gate/tool inputs now bind dataset id, content hash and authenticated snapshot; execution revalidates the binding and adversarial source drift fails closed. |
| CR-02 Labeling read TOCTOU | Proposal, maturity and label derivation use one retained-descriptor authenticated frame; swap/restore tests commit no derived dataset, artifact or audit. |
| CR-03 self-asserted production activation | Deployment manifests are server-generated; activation requires allowlisted verifier evidence and reauthenticates inside the transaction; no verifier means fail closed. |
| WR-01 leaked active driver job | Failed queued-to-running claims are synchronously terminated with a typed conflict; no active job remains. |
| WR-02 self-asserted Decision Twin replay | The production entry resolves CAS receipts and frozen trusted adapters, binds manifest/record/facts, and recursively fingerprints/freezes reachable helper/default/closure/global code before execution. |

Deeper follow-up reviews also found and closed Strategy DSL/cache activation drift,
generic report-gate bypass, C1 content-addressed pin/GC races, and stale-report fallback.
The final independent attack replay passed all three last trust-boundary harnesses
(Decision Twin helper substitution, C1 source/CAS races, and failed/running latest
report attempts); the exact status and remaining open acceptance gates are recorded in
[the comprehensive remediation record](2026-08-01-comprehensive-audit-remediation.md).

Across the original and follow-up rounds, the consolidated ledger contains
5 Critical, 10 Warning and 1 Info finding (16 total). All 16 locally actionable
findings are closed in the current worktree; the current local verdict is
`LOCAL_FINDINGS_CLOSED_WITH_OPEN_ACCEPTANCE_GATES`, not a release or production verdict.
