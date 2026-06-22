# Design

## Source of truth
- Status: Active
- Last refreshed: 2026-06-04
- Primary product surfaces: local MARVIS credit-risk-agent workbench, current model-validation workflow, Agent-assisted explanations and report drafting, generated evidence, Excel/Word reports, and runtime branding.
- Evidence reviewed: current `README.md`, `README.zh-CN.md`, `docs/roadmap.md`, `docs/versioning.md`, `docs/branding.md`, `docs/notebook_contract.md`, `docs/runbook.md`, and the static FastAPI-served frontend.
- Roadmap reference: use `docs/roadmap.md` for version phases and Plugin/Tool/Hook/Workflow terminology. Keep this file focused on product experience and interface decisions.

## Brand
- Personality: restrained local professional tool, closer to Xcode or Finder utility panels than SaaS landing pages.
- Trust signals: clear task state, human-readable execution evidence, stable report output paths, visible errors near the triggering action, and auditable Agent statements.
- Avoid: marketing hero pages, decorative glass effects, heavy shadows, redundant rails, generic AI claims, and JSON as the default user-facing result.
- Default public brand: `MARVIS-全能信贷风控智能体`.
- Branding must be runtime-configurable without source-code edits. The configurable surface is logo, favicon/web logo, primary theme color, platform display name, and browser page title.
- Private/customer branding belongs in `workspace/branding/brand.json` plus sibling assets, and must not be required for the open-source checkout to run.
- When no branding config exists, the platform uses the public MARVIS defaults: black primary color and the built-in MARVIS logo/favicon.

## Product goals
- Goals: make MARVIS a local-first credit-risk agent that assists governed validation, modeling, analysis, strategy, and monitoring work through structured workflows.
- Current V1 goal: keep model validation stable and demonstrable through task creation, material scanning, notebook execution, deterministic validation evidence, Agent explanations, report conclusion confirmation, and Excel/Word output.
- V1.1 experience goal: let Agent use auditable cross-task memory for user preferences, field conventions, validation pitfalls, task experience, model experience, and future skill-experience hooks while preserving deterministic validation results.
- V2+ experience goal: let Agent understand a user goal, plan with available plugins/tools/hooks, execute controlled Python capabilities, and return structured evidence and report content.
- Non-goals: marketing homepage, arbitrary unreviewed code execution, report styling rewrites unrelated to structured task output, hidden memory use, or overclaiming unsupported workflows before their plugins exist.
- Success signals: one obvious next action, evidence readable without code knowledge, Agent explanations traceable to current evidence or memory references, and generated reports grounded in structured results.

## Personas and jobs
- Primary personas: credit-risk modelers, risk analysts, strategy operators, and model validation staff who can operate a local web tool but should not need to read raw JSON or manually wire scripts together.
- Current V1 user jobs: choose a project folder, verify model-validation materials, execute notebook logic, inspect key metrics, complete fixed report fields, export Word/Excel.
- V1.1 user jobs: receive useful historical reminders inside Agent analysis, understand whether a new validation result improved or declined compared with comparable historical models, reuse known field conventions and validation pitfalls, and manage memory entries when needed.
- Future user jobs: prepare modeling evidence, compare risk segments, evaluate strategy changes, monitor portfolio quality, and let governed plugins turn those tasks into structured outputs.
- Key contexts of use: local Jupyter terminal / notebook proxy, internal network, repeated credit-risk analysis and validation tasks.

## Information architecture
- Primary navigation: brand-configurable left task sidebar with platform title, task list, task search, create action, and settings for sort/group/theme.
- Core current screen: single resizable workbench with task management on the left, live task evidence in the center, and report/output canvas on the right.
- Agent surface: center-column conversation that sits with task evidence, not as a detached chatbot unrelated to the current task.
- V1.1 memory surface: memory should appear inside Agent explanations, warnings, comparison summaries, report-draft rationale, and future workflow choices. Do not add a fixed top/center memory block that lists matched memories.
- Memory management surface: settings or audit management view for listing, inspecting, disabling, deleting, and exporting memory audit records. It is not a default task dashboard panel.
- V2+ plugin surface: plugin/tool execution output should appear as native evidence sections, tables, charts, findings, and report sections.

## Design principles
- Principle 1: keep creation-time configuration in the create-task dialog, including manual/Agent mode selection.
- Principle 2: keep real-time validation evidence in the center and the Word report as the right-side working output; report text and project information are edited in the document canvas, not in a separate form page.
- Principle 3: default to human-readable summaries; raw data stays available behind details.
- Principle 4: every Agent statement should be grounded in current task evidence, explicit memory references, or clearly labeled general explanation.
- Principle 5: extension output must look native. A plugin/tool may add a table, chart, finding, or report section, but it should not introduce a separate visual language or bypass platform rendering.
- Tradeoffs: compact internal-tool density over decorative presentation; auditability over conversational smoothness when the two conflict.

## Visual language
- Color: neutral light background and white panels. The default primary color is black; configured primary color drives primary actions such as create-task and Agent send.
- Typography: system font, 12 / 14 / 17 / 22 / 28 size scale, 400 / 500 / 700 weights.
- Spacing/layout rhythm: resizable workbench, central evidence console, right document canvas, 8px grid where practical.
- Shape/radius/elevation: one 8px radius token; border-first surfaces with minimal shadow.
- Motion: no hover translation; loading state changes text and spinner only.
- Imagery/iconography: logo and favicon only for the current internal tool; avoid decorative art.

## Components
- Existing components to reuse: static FastAPI-served HTML/CSS/JS, current task/report API ids, existing task evidence sections, current Agent conversation UI.
- Existing Agent components: LLM settings, center-column conversation, asymmetric user/Agent messages, staged evidence summaries, Word conclusion confirmation gate.
- Branding components: runtime brand loader, brand config schema, sidebar logo/name binding, favicon/title binding, CSS primary-color token, and public default MARVIS logo/favicon.
- V1.1 components to design: inline memory-aware Agent statements, expandable memory references on Agent messages, memory management/audit view, memory disable/delete actions, and memory-use audit metadata.
- V2 components to design later: plugin registry, tool run evidence panel, hook run evidence panel, workflow plan view, extension metric tables, extension report sections, and plugin output display declarations.
- Variants and states: primary/secondary/disabled/loading/error/success buttons; empty/loading/error/success summaries; high/medium/low confidence memory comparison states; disabled memory state; deleted-memory reference state.
- Token/component ownership: `marvis/static/styles.css`.

## Accessibility
- Target standard: practical WCAG AA for contrast and keyboard focus on the local tool.
- Keyboard/focus behavior: visible `:focus-visible`, task list keyboard navigation, Enter task creation, Cmd/Ctrl+S save report text where supported.
- Contrast/readability: muted text must remain readable on white; long names wrap instead of being clipped.
- Screen-reader semantics: status and alert areas near the relevant actions; raw data and memory references are labelled.
- Reduced motion and sensory considerations: no layout-shifting hover motion.

## Responsive behavior
- Supported breakpoints/devices: desktop, 13-inch laptop, tablet/narrow browser, mobile emergency use.
- Layout adaptations: workbench columns on desktop, stacked sections below narrow widths; progress and report panels move below the main evidence flow on small screens.
- Touch/hover differences: controls must work without hover-only affordances.

## Interaction states
- Loading: action buttons show busy text and disable conflicting actions.
- Empty: tells the user what to do next, not just what is missing.
- Error: shown near the action bar and in the affected result section when possible.
- Success: summary panel shows output or evidence in human-readable form.
- Disabled: buttons carry titles explaining why they cannot be used.
- Offline/slow network: long notebook/report actions remain visibly busy.
- Memory comparison uncertainty: medium-confidence matches must be textually marked as requiring human confirmation. Low-confidence matches should not be used for historical comparison.

## Content voice
- Tone: concise operational Chinese in the product UI.
- Terminology: use "材料", "Notebook", "报告文案", "Word 报告", "历史对比", "记忆引用"; avoid backend lifecycle labels as UI labels.
- Microcopy rules: no double-language eyebrow; no README-style explanatory paragraphs in every section; Agent should distinguish current evidence, historical memory, and general domain guidance. Memory should sound like operational guidance, not a separate "AI memory found X" system notice.

## Implementation constraints
- Framework/styling system: plain HTML/CSS/JS served by FastAPI.
- Design-token constraints: no new frontend dependency; keep proxy-safe relative paths.
- Branding config should live outside committed source defaults under `workspace/branding/`.
- Performance constraints: avoid unnecessary `backdrop-filter`, heavy shadows, and repeated DOM rewrites during polling.
- Compatibility constraints: preserve existing API endpoints and legacy DOM ids used by tests unless a task explicitly migrates them.
- Test/screenshot expectations: smoke tests plus browser viewport verification after frontend changes.
- Memory constraints: memory can support explanation, comparison, field suggestions, pitfall warnings, report wording, and future workflow planning, but cannot alter deterministic validation metrics. Memory references must carry source, category, confidence, and audit metadata.
- Plugin constraints: plugin/tool/hook outputs must return structured results; platform owns Word/Excel/chart rendering.

## Open questions
- [ ] V1.1 memory management UI placement: settings modal first, then dedicated audit screen if volume requires it.
- [ ] V2 plugin upload UI: administrator-only, ordinary user with confirmation, or local developer-only in the first release.
- [ ] V2 execution isolation: process-local controlled execution for alpha, subprocess boundary, or container boundary before public use.
