# V1.1 Agent Memory Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build MARVIS V1.1 Agent Memory Foundation so Agent can use auditable cross-task memory for preferences, field conventions, validation pitfalls, task experience, model experience, and future skill-experience hooks without changing deterministic validation results.

**Architecture:** Store typed local memories in SQLite with append-only audit events. Retrieve bounded task-aware memory context before Agent analysis/chat, attach used memory references to Agent message metadata, and surface memory through inline Agent statements plus a management/audit view. Keep validation metrics and report evidence independent from memory.

**Tech Stack:** Python 3.11+ runtime boundary, FastAPI, SQLite, Pydantic-style request models, plain HTML/CSS/JS frontend, pytest, existing MARVIS task/Agent architecture.

---

## Scope Guard

This plan prepares implementation. It does not contain implementation code because the current instruction is to update documents and get ready for development first.

Do not implement:

- V2 Plugin/Tool/Hook runtime.
- External memory providers.
- A fixed top/center memory display block.
- Any mechanism that changes KS, AUC, PSI, PMML consistency, validation status, or report evidence based on memory.

## File Map

Create during implementation:

- `marvis/agent_memory/__init__.py`: public memory package exports.
- `marvis/agent_memory/models.py`: memory types, statuses, structured payload helpers.
- `marvis/agent_memory/policy.py`: allow/deny rules and safety classification.
- `marvis/agent_memory/store.py`: SQLite CRUD and audit event access.
- `marvis/agent_memory/extractors.py`: candidate extraction from task, validation, failure, report, and user preference events.
- `marvis/agent_memory/retrieval.py`: task-aware filtering, fuzzy matching, confidence scoring, bounded context packets.
- `marvis/agent_memory/prompting.py`: conversion of retrieved memories into Agent prompt/evidence payloads.
- `tests/test_agent_memory_store.py`: storage, audit, status, policy tests.
- `tests/test_agent_memory_retrieval.py`: ranking and comparison-confidence tests.
- `tests/test_agent_memory_extractors.py`: extraction tests for model, task, field, pitfall, and preference candidates.
- `tests/test_agent_memory_api.py`: memory management endpoints and message reference tests.

Modify during implementation:

- `marvis/db.py`: create memory tables and repository hooks or delegate to `agent_memory.store`.
- `marvis/domain.py`: add memory reference payloads only if shared typed records are needed.
- `marvis/api.py`: expose memory management endpoints and include message memory references where needed.
- `marvis/pipeline.py`: trigger candidate extraction after validation completion, failure, and report stages.
- `marvis/agent/service.py`: retrieve bounded memory context for Agent analysis/chat and persist used references.
- `marvis/agent/prompts.py`: tell Agent how memory may and may not be used.
- `marvis/static/app.js`: render expandable memory references on Agent messages and management view actions.
- `marvis/static/styles.css`: style inline references and management view with existing platform tokens.
- `marvis/static/index.html`: add management view/modal shell only if needed.
- `tests/test_agent_service.py`: memory context and deterministic-result guard tests.
- `tests/test_agent_api.py`: Agent message metadata tests.
- `tests/test_frontend_static_v2.py`: static UI contract tests, including no permanent memory block.

## Task 1: Memory Schema and Policy Tests

**Files:**
- Create: `tests/test_agent_memory_store.py`
- Create: `tests/test_agent_memory_extractors.py`
- Create: `tests/test_agent_memory_retrieval.py`
- Create: `marvis/agent_memory/models.py`
- Create: `marvis/agent_memory/policy.py`

- [ ] Write tests for memory types: `user_preference`, `field_convention`, `validation_pitfall`, `task_experience`, `model_experience`, `skill_experience_reserved`.
- [ ] Write tests for statuses: `active`, `disabled`, `deleted`, `rejected`.
- [ ] Write tests that forbidden content is rejected: raw sample rows, notebook source blocks, PMML/model file contents, API keys, DB connection strings, non-desensitized report text.
- [ ] Write tests that model experience requires KS/AUC/PSI/month/channel/model name/model version/scope/source task id/important feature sources.
- [ ] Run targeted tests and confirm they fail because modules do not exist yet.
- [ ] Implement minimal models and policy helpers.
- [ ] Run targeted tests and confirm they pass.

## Task 2: SQLite Store and Audit Events

**Files:**
- Create: `marvis/agent_memory/store.py`
- Modify: `marvis/db.py`
- Test: `tests/test_agent_memory_store.py`

- [ ] Add failing tests for creating active memory entries with structured payload JSON.
- [ ] Add failing tests for append-only audit events on create, retrieve, use, disable, re-enable, delete, and reject.
- [ ] Add failing tests that disabled/deleted entries are excluded from default retrieval.
- [ ] Add failing tests that deleted entries redact content but preserve tombstone audit.
- [ ] Implement memory table initialization.
- [ ] Implement store methods.
- [ ] Run store tests.
- [ ] Run `conda run -n py_313 python -m pytest tests/test_db.py tests/test_agent_memory_store.py -q`.

## Task 3: Candidate Extraction

**Files:**
- Create: `marvis/agent_memory/extractors.py`
- Modify: `marvis/pipeline.py`
- Test: `tests/test_agent_memory_extractors.py`

- [ ] Add failing tests for extracting model experience from validation results.
- [ ] Add failing tests for extracting validation pitfall from notebook, PMML, field, execution environment, and report failures.
- [ ] Add failing tests for extracting task experience from completed/failed task summaries.
- [ ] Add failing tests for extracting user preference from explicit remember/correction messages.
- [ ] Add failing tests that skill experience is only a reserved inactive category.
- [ ] Implement extractors without wiring them into runtime side effects yet.
- [ ] Run extractor tests.

## Task 4: Retrieval, Fuzzy Matching, and Comparison Confidence

**Files:**
- Create: `marvis/agent_memory/retrieval.py`
- Test: `tests/test_agent_memory_retrieval.py`

- [ ] Add failing tests for high-confidence exact model/scope/channel/month matches.
- [ ] Add failing tests for medium-confidence fuzzy model keyword and scope matches.
- [ ] Add failing tests that low-confidence candidates do not appear in comparison context.
- [ ] Add failing tests for comparing multiple models, months, channels, and metrics.
- [ ] Add failing tests for normalized model families: A card, B card, amount, rate, pre-screening, C card.
- [ ] Implement ranking and bounded context packet generation.
- [ ] Run retrieval tests.

## Task 5: Agent Prompt Integration

**Files:**
- Create: `marvis/agent_memory/prompting.py`
- Modify: `marvis/agent/service.py`
- Modify: `marvis/agent/prompts.py`
- Test: `tests/test_agent_service.py`
- Test: `tests/test_agent_api.py`

- [ ] Add failing tests that Agent receives bounded memory context when relevant.
- [ ] Add failing tests that used memory references are recorded in Agent message metadata.
- [ ] Add failing tests that Agent can say historical KS improved/declined only from structured memory and current structured validation results.
- [ ] Add failing tests that memory does not mutate validation result payloads.
- [ ] Update prompts with memory use rules and forbidden behavior.
- [ ] Wire retrieval into Agent analysis/chat.
- [ ] Run Agent service/API tests.

## Task 6: Runtime Memory Capture Hooks

**Files:**
- Modify: `marvis/pipeline.py`
- Modify: `marvis/api.py`
- Test: `tests/test_agent_memory_extractors.py`
- Test: `tests/test_pipeline_v2.py`

- [ ] Add failing tests that validation completion creates safe candidate model experience entries.
- [ ] Add failing tests that validation failure creates pitfall/task experience candidates when safe.
- [ ] Add failing tests that report confirmation can create task/user preference candidates when explicit and safe.
- [ ] Add failing tests that rejected candidates create audit events but no active memory.
- [ ] Wire extractors to pipeline events.
- [ ] Run pipeline and memory tests.

## Task 7: Memory Management API

**Files:**
- Modify: `marvis/api.py`
- Test: `tests/test_agent_memory_api.py`

- [ ] Add failing tests for listing memories with type/status/source/model/channel/month filters.
- [ ] Add failing tests for getting one memory with audit events.
- [ ] Add failing tests for disable, re-enable, and delete.
- [ ] Add failing tests for listing memory references attached to an Agent message.
- [ ] Implement local API endpoints.
- [ ] Run API tests.

## Task 8: Frontend Inline References and Management View

**Files:**
- Modify: `marvis/static/app.js`
- Modify: `marvis/static/styles.css`
- Modify: `marvis/static/index.html`
- Test: `tests/test_frontend_static_v2.py`

- [ ] Add failing static tests that no permanent task-level memory block exists.
- [ ] Add failing static tests for expandable memory references on Agent messages.
- [ ] Add failing static tests for memory management actions: inspect, disable, re-enable, delete.
- [ ] Implement inline references and management UI.
- [ ] Keep memory UI compact and consistent with existing workbench styling.
- [ ] Run frontend static tests and `node --check marvis/static/app.js`.

## Task 9: End-to-End Regression

**Files:**
- Modify as needed based on failures.

- [ ] Run targeted V1.1 tests.
- [ ] Run Agent tests.
- [ ] Run pipeline and validation regression tests.
- [ ] Run frontend static tests.
- [ ] Run `git diff --check`.
- [ ] Confirm no deterministic validation outputs are memory-derived.
- [ ] Confirm no fixed memory gray block appears in the task workbench.

## Recommended Implementation Mode

Use subagent-driven development after this plan is accepted:

- Worker 1: store/policy/schema.
- Worker 2: extraction/retrieval.
- Worker 3: Agent prompt/API integration.
- Worker 4: frontend references/management view.
- Leader: integration, audit, deterministic-result regression, final verification.

Each worker should stay inside its assigned files and report conflicts upward before editing shared files such as `api.py`, `pipeline.py`, or `agent/service.py`.
