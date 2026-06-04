# Design

## Source of truth
- Status: Active index
- Last refreshed: 2026-06-03
- Primary product surfaces: local MARVIS credit-risk-agent workbench at `/`, Agent-assisted task execution, current built-in model-validation workflow, generated evidence and Excel/Word reports.
- Current architecture references: `AGENTS.md`, `docs/notebook_contract.md`, `docs/superpowers/specs/2026-05-25-agent-mode-p1-design.md`, `docs/superpowers/specs/2026-05-31-agent-skill-runtime-p2-design.md`.
- Historical note: earlier v1/v2 UI goals in this file have been superseded by v3 staged validation and Agent P1. Keep this file as a product/design index, not as the canonical implementation contract.

## Brand
- Personality: restrained local professional tool, closer to Xcode or Finder utility panels than SaaS landing pages.
- Trust signals: clear task state, human-readable execution evidence, stable report output paths, visible errors near the triggering action.
- Avoid: marketing eyebrow text, decorative glass effects, heavy shadows, redundant rails, JSON as the default user-facing result.
- Default public brand: `MARVIS-全能风控智能体`.
- Branding must be runtime-configurable without source-code edits. The configurable surface is logo, favicon/web logo, primary theme color, platform display name, and browser page title.
- Private/customer branding belongs in `workspace/branding/brand.json` plus sibling assets, and must not be required for the open-source checkout to run.
- When no branding config exists, the platform uses the public MARVIS defaults: black primary color and the built-in MARVIS logo/favicon.
- Default logo direction: a simple rounded-square MARVIS risk-agent icon, using a warm dark-gray winking assistant beside a credit-risk report card. The report card shows a red-to-yellow-to-green credit-quality gauge pointing into green, green checklist pass marks, and a small chip/model icon. It should stay friendly, slightly dimensional, not too black-heavy, legible as a sidebar logo and favicon, and not tied to any institution.

## Product goals
- Goals: make MARVIS a local-first credit-risk agent that can assist modeling, analysis, strategy, and validation work through governed task workflows. V1 proves this direction through a stable model-validation workflow: users can create a task, scan supplied files, run Notebook validation in a live kernel, inspect deterministic evidence, use Agent explanations when helpful, confirm report conclusions, and generate Excel/Word outputs.
- P2 goals: let new credit-risk task types be added through governed skill packages rather than Agent code edits; after the runtime is stable, extend from validation-only execution toward modeling helpers, portfolio analysis, strategy evaluation, monitoring, and model-family-aware workflows.
- Non-goals: marketing homepage, arbitrary unreviewed code execution, report styling rewrites unrelated to structured task output, or overclaiming unsupported workflows before their skills exist.
- Success signals: one obvious next action, evidence readable without code knowledge, reports generated from structured results, extension skills auditable by version and output, and validation remaining one workflow under the broader credit-risk-agent product.

## Personas and jobs
- Primary personas: credit-risk modelers, risk analysts, strategy operators, and model validation staff who can operate a local web tool but should not need to read raw JSON or wire together scripts manually.
- Current V1 user jobs: choose a project folder, verify model-validation materials, execute notebook logic, inspect key metrics, complete fixed report fields, export Word.
- Future user jobs: prepare modeling evidence, compare risk segments, evaluate strategy changes, monitor portfolio quality, and let governed skills turn those tasks into structured outputs.
- Key contexts of use: local Jupyter terminal / notebook proxy, internal network, repeated credit-risk analysis and validation tasks.

## Information architecture
- Primary navigation: brand-configurable left task sidebar with platform title, task list, task search, create action, and settings for sort/group/theme.
- Core routes/screens: single resizable three-column workbench: task management on the left, live task evidence in the middle, editable report/output canvas on the right.
- Content hierarchy: current task state and workflow evidence first; Word report stays visible for the current validation workflow and locks during system updates; raw audit payloads are secondary details.

## Design principles
- Principle 1: keep creation-time configuration in the create-task dialog, including manual/Agent mode selection.
- Principle 2: keep real-time validation evidence in the center and the Word report as the right-side working output; report text and project information are edited in the document canvas, not in a separate form page.
- Principle 3: default to human-readable summaries; raw data stays available behind details.
- Principle 4: extension output must look native. A skill may add a table, chart, finding, or report section, but it should not introduce a separate visual language or bypass platform rendering.
- Tradeoffs: compact internal-tool density over decorative Apple-like surface effects.

## Visual language
- Color: neutral light background and white panels. The default primary color is black; configured primary color drives the two create-task buttons and the Agent conversation send button, with future primary actions using the same token.
- Typography: system font, 12 / 14 / 17 / 22 / 28 size scale, 400 / 500 / 700 weights.
- Spacing/layout rhythm: resizable three-column workbench, central evidence console, right document canvas, 8px grid where practical.
- Shape/radius/elevation: one 8px radius token; border-first surfaces with minimal shadow.
- Motion: no hover translation; loading state changes text and spinner only.
- Imagery/iconography: none required for the current internal tool; avoid decorative art.

## Components
- Existing components to reuse: static FastAPI-served HTML/CSS/JS, current task/report API ids.
- Existing Agent components: LLM settings, center-column conversation, asymmetric user/Agent messages, staged evidence summaries, Word conclusion confirmation gate.
- Branding components to add: runtime brand loader, brand config schema, sidebar logo/name binding, favicon/title binding, CSS primary-color token, and a public default MARVIS logo/favicon.
- P2 components to design: skill upload/registry management, skill run evidence panel, extension metric tables, extension report sections, planner-generated execution plan view, model-family-aware result sections for binary classification, multiclass classification, and regression.
- Variants and states: primary/secondary/disabled/loading/error/success buttons; empty/loading/error/success summaries.
- Token/component ownership: `riskmodel_checker/static/styles.css`.

## Accessibility
- Target standard: practical WCAG AA for contrast and keyboard focus on the local tool.
- Keyboard/focus behavior: visible `:focus-visible`, task list arrow navigation, Enter task creation, Cmd/Ctrl+S save report text.
- Contrast/readability: muted text must remain readable on white; long names wrap instead of being clipped.
- Screen-reader semantics: status and alert areas near the relevant actions; raw data is labelled.
- Reduced motion and sensory considerations: no layout-shifting hover motion.

## Responsive behavior
- Supported breakpoints/devices: desktop, 13-inch laptop, tablet/narrow browser, mobile emergency use.
- Layout adaptations: three columns on desktop, stacked sections below narrow widths; the progress rail moves below the document on small screens.
- Touch/hover differences: controls must work without hover-only affordances.

## Interaction states
- Loading: action buttons show busy text and disable conflicting actions.
- Empty: tells the user what to do next, not just what is missing.
- Error: shown near the action bar and in the affected result section when possible.
- Success: summary panel shows output or evidence in human-readable form.
- Disabled: buttons carry titles explaining why they cannot be used.
- Offline/slow network: long notebook/report actions remain visibly busy.

## Content voice
- Tone: concise operational Chinese.
- Terminology: use "材料", "Notebook", "报告文案", "Word 报告"; avoid backend lifecycle labels as UI labels.
- Microcopy rules: no double-language eyebrow; no README-style explanatory paragraphs in every section.

## Implementation constraints
- Framework/styling system: plain HTML/CSS/JS served by FastAPI.
- Design-token constraints: no new frontend dependency; keep proxy-safe relative paths.
- Branding config should live outside committed source defaults, preferably under a workspace-local folder such as `workspace/branding/`. A safe example config may be committed, but private config and assets must be ignored before publishing.
- Performance constraints: avoid unnecessary `backdrop-filter` and heavy shadows.
- Compatibility constraints: preserve existing API endpoints and legacy DOM ids used by tests.
- Test/screenshot expectations: smoke tests plus browser viewport verification after frontend changes.
- Extension constraints: uploaded skills return structured results only; platform owns Word/Excel/chart rendering.

## Open questions
- [ ] P2 skill upload UI 是否只给管理员使用，还是普通验证人员也能上传但需审批。
- [ ] skill 执行隔离第一版是否接受进程内受控执行，还是 P2 上线前必须做子进程/容器隔离。
