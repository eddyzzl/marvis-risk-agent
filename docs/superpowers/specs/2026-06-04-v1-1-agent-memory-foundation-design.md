# V1.1 Agent Memory Foundation Design

## Status

- Status: Ready for implementation planning
- Date: 2026-06-04
- Scope: V1.1 documentation and development preparation only
- Code status: no implementation code in this spec

## Why V1.1 Was Reset

The first V1.1 attempt was rolled back because it treated memory as a narrow model-metric record and surfaced it as a fixed frontend block. That is not the product target.

V1.1 must build a memory foundation for the Agent. Model metrics are one memory category, not the whole system. Memory should improve Agent analysis, field interpretation, historical comparison, report wording, risk reminders, and later workflow planning. It must not appear as a permanent "matched memories" block at the top of the task UI.

## External References

Sources checked on 2026-06-04:

- OpenClaw memory docs: https://github.com/openclaw/openclaw/blob/main/docs/concepts/memory.md
- Hermes Agent repository and README: https://github.com/NousResearch/hermes-agent
- Hermes persistent memory docs: https://github.com/NousResearch/hermes-agent/blob/main/website/docs/user-guide/features/memory.md

OpenClaw practices to adapt:

- Local-first memory that is inspectable by humans.
- Layered memory: compact long-term memory, richer daily/session notes, optional review diary.
- Promotion and distillation instead of saving every detail permanently.
- Action-sensitive memories that record when a remembered fact is safe to act on.
- Search/get tools and optional richer memory-wiki style provenance.
- Background consolidation with review surfaces rather than blind promotion.

Hermes practices to adapt:

- Separate Agent notes from user profile memory.
- Keep injected memory compact and bounded.
- Treat memory as agent-curated, not raw transcript storage.
- Store full sessions separately and search them on demand.
- Provide add/replace/remove memory operations with size limits.
- Scan memory candidates for sensitive or adversarial content before saving.
- Keep external providers additive, not replacements for built-in memory.

MARVIS decisions:

- Use structured, auditable local storage rather than plain Markdown as the source of truth, because model comparison needs typed fields such as KS, AUC, PSI, month, channel, model name, version, scope, and source task id.
- Generate bounded memory context packets for Agent prompts instead of injecting all memory into every request.
- Keep deterministic validation outputs independent from memory. Memory can explain or warn, but cannot change KS, AUC, PSI, PMML consistency, report evidence, or validation status.
- Add management and audit views, but keep memory value inside Agent analysis rather than as a fixed UI panel.

## Goals

- Let Agent remember useful cross-task knowledge beyond the current conversation.
- Support user preferences, field conventions, validation pitfalls, task experience, model experience, and future skill experience.
- Let Agent compare current model results with historical comparable models across multiple metrics, months, channels, and scopes.
- Keep all memory inspectable, disableable, deletable, and auditable.
- Record which memory entries influenced an Agent message.
- Preserve V1 validation behavior and report determinism.

## Non-Goals

- Do not implement V2 Plugin/Tool/Hook runtime.
- Do not let memory auto-correct or override validation metrics.
- Do not save raw samples, customer-level rows, complete notebook source, PMML/model file contents, secrets, database connections, private branding, or non-desensitized reports.
- Do not add a permanent memory display block to the task workbench.
- Do not require external vector databases or cloud memory providers for V1.1.

## Memory Categories

### User Preference

Purpose: improve phrasing and interaction style.

Allowed examples:

- Report wording preference.
- Preferred explanation depth.
- Common output structure.
- Explicit user correction such as "do not describe KS 0.30 as 0.30 points."

Required metadata:

- `memory_type = "user_preference"`
- `content`
- `source`
- `confidence`
- `created_at`
- `updated_at`

### Field Convention

Purpose: help Agent recognize recurring validation field口径 without changing platform checks.

Allowed examples:

- Channel field aliases.
- Time/month field aliases.
- Sample split field aliases.
- Target and score field naming habits.

Required metadata:

- `memory_type = "field_convention"`
- `field_role`
- `aliases`
- `scope`
- `source_task_id`
- `confidence`

### Validation Pitfall

Purpose: help Agent explain repeated failures and suggest targeted checks.

Allowed examples:

- Notebook kernel died because pyzmq/conda native library signatures were blocked.
- PMML scoring failed due to unsupported transform.
- Data dictionary was missing required variable descriptions.

Required metadata:

- `memory_type = "validation_pitfall"`
- `failure_stage`
- `symptom`
- `likely_cause_summary`
- `recommended_check`
- `source_task_id`
- `confidence`

### Task Experience

Purpose: preserve non-sensitive lessons from completed or failed tasks.

Allowed examples:

- A task failed at notebook reproducibility due to environment setup.
- A report conclusion required a stricter PSI caveat after human review.
- A certain material package structure was accepted.

Required metadata:

- `memory_type = "task_experience"`
- `task_summary`
- `outcome`
- `review_note`
- `source_task_id`
- `confidence`

### Model Experience

Purpose: compare current model behavior with historical comparable models and summarize scenario-level经验.

Required fields:

- `memory_type = "model_experience"`
- `ks`
- `auc`
- `psi`
- `month`
- `channel`
- `model_name`
- `model_version`
- `scope`
- `source_task_id`
- `important_feature_sources`

Optional fields:

- `model_family`: A card, B card, amount, rate, pre-screening, C card, or another normalized scene label.
- `sample_window`
- `metric_direction_note`
- `human_review_note`

Comparison rules:

- Compare multiple metrics when available, not just KS.
- Compare multiple historical candidates when confidence is high or medium.
- High-confidence comparisons can state improvement or decline.
- Medium-confidence comparisons must say the candidate is "possibly comparable" and requires human confirmation.
- Low-confidence candidates should not appear in Agent conclusions.
- Memory can say "current KS is lower than previous comparable A-card version" only when both current and historical values come from structured platform results.

### Skill Experience Reserved

Purpose: reserve future V2 skill/SOP/playbook learning without implementing runtime execution in V1.1.

Allowed in V1.1:

- Schema field reservation.
- Disabled storage category.
- Documentation of future relationship to V2 workflow execution.

Not allowed in V1.1:

- Executing plugin/tool code based on skill memory.
- Auto-generating workflow code from memory.

## Storage Model

Primary storage should be local SQLite because the existing platform already uses SQLite for tasks and agent messages, and V1.1 needs typed filtering and audit history.

Recommended tables:

- `agent_memory_entries`: memory id, type, status, content summary, structured payload JSON, source task id, confidence, created/updated timestamps.
- `agent_memory_events`: append-only audit events for create, update, retrieve, use, disable, delete, and safety rejection.
- `agent_memory_links`: optional relation table for memories linked to tasks, models, fields, or report revisions.

Entry status:

- `active`: can be retrieved.
- `disabled`: preserved for audit but not retrieved by default.
- `deleted`: tombstoned; content removed or redacted, audit event retained.
- `rejected`: candidate blocked by safety or policy.

Memory content should be compact. Richer details should stay in task artifacts, validation result files, or agent message history and be retrieved only when allowed.

## Candidate Extraction

Candidate memories can be proposed from these events:

- Task validation completed.
- Task failed.
- Agent report draft confirmed or corrected.
- User explicitly says to remember a preference.
- Agent detects repeated field conventions from accepted tasks.
- Future V2 skill/workflow completed, reserved only.

Extraction must produce a structured candidate with:

- category,
- compact summary,
- structured payload,
- source task id or source message id,
- confidence,
- safety classification,
- reason for saving.

## Safety and Policy Gate

Reject a candidate if it contains or appears to contain:

- raw sample rows or customer-level records,
- long notebook code,
- PMML or model file contents,
- API keys or database credentials,
- full report text without desensitization,
- private institution identifiers beyond what is already safe in task metadata,
- prompt injection instructions that try to override platform policy,
- claims without source evidence.

Rejected candidates should produce an audit event but not an active memory entry.

## Retrieval and Ranking

Retrieval happens before Agent stage analysis or free-form Agent chat. It should be task-aware and bounded.

Inputs:

- current task metadata,
- current stage,
- validation results when available,
- user message,
- model name/version/scope/channel/month,
- field names and source material scan result.

Ranking signals:

- exact model name and version,
- fuzzy model name keywords,
- normalized model family such as A card or amount model,
- scope similarity,
- channel match,
- month/sample-window match,
- metric availability,
- source reliability,
- recency,
- user-disabled state.

Output:

- a bounded memory context packet for LLM use,
- a list of memory references that must be persisted into Agent message metadata if used,
- comparison confidence per candidate.

## Agent Behavior

Agent may use memory to:

- remind the user of relevant historical issues,
- compare current and previous model performance,
- suggest fields to inspect,
- explain recurring notebook or PMML failures,
- adapt report wording to user preferences,
- prepare future workflow/tool selection context.

Agent must not:

- change platform-calculated metrics,
- mark validation passed/failed based on memory alone,
- hide uncertainty in medium-confidence comparisons,
- cite memory without a memory reference,
- treat deleted or disabled memory as active.

Example desired behavior:

> 当前分润 A 卡模型在 2026 年 1 月样本的 KS 为 30。历史记忆中，上一版分润 A 卡模型在同月同渠道的 KS 为 20，当前模型区分能力有提升；但 PSI 仍需要结合稳定性结果复核。

Example undesired behavior:

> 找到 5 条记忆：...

The first example uses memory inside analysis. The second exposes memory as a separate display artifact without decision value.

## Frontend Experience

Task workbench:

- no fixed memory block,
- no top gray memory summary,
- no separate "matched memory" panel in the main task view.

Agent messages:

- memory-aware statements appear naturally in the content,
- expandable references can show memory id, type, source task id, confidence, and use reason,
- memory references should not dominate the message.

Memory management:

- settings or audit view lists memories,
- filters by type, status, source task, model name, channel, month, confidence,
- actions: disable, re-enable, delete, inspect audit,
- deleted entries should preserve audit tombstones.

## API Surface

V1.1 should expose minimal local API endpoints:

- list memory entries with filters,
- get one memory entry and audit events,
- disable / re-enable memory,
- delete memory,
- list memory references attached to an Agent message,
- optionally export memory audit for local review.

Write APIs should be internal at first, called from pipeline and Agent flows. User-facing manual create/edit can wait unless implementation finds it necessary for correction flows.

## Test Requirements

Storage:

- CRUD and audit event behavior.
- Disabled/deleted memories not retrieved.
- Safety rejection blocks forbidden content.
- Structured model experience fields round-trip.

Retrieval:

- high-confidence model match.
- medium-confidence fuzzy match.
- low-confidence candidates excluded from conclusions.
- multiple metrics and multiple historical candidates handled.

Agent integration:

- memory context is included only where needed.
- Agent message metadata records memory references.
- memory cannot modify validation results payload.

Frontend:

- no fixed memory block appears in task header/body.
- Agent message references can expand.
- management view supports disable/delete.

Regression:

- existing V1 manual mode and Agent mode still work.
- existing report generation and deterministic validation tests still pass.

## Development Boundary

Implementation should start from tests and storage/retrieval contracts, then integrate Agent prompts and UI. Do not start by building a visual memory panel. Do not resurrect the rolled-back narrow metric-only implementation.
