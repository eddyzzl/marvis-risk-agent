# Design

## Source of truth
- Status: Active
- Last refreshed: 2026-10-14
- Current redesign: decisions D1-D17 and the batch plan live in `docs/superpowers/plans/2026-10-14-workbench-redesign-plan.md`; the approved mockup is `docs/superpowers/specs/2026-10-14-workbench-mockup/index.html`. Where this file and that plan disagree during the redesign, the plan wins until the batch that touches the area lands.
- Primary product surfaces: local MARVIS credit-risk-agent workbench, V2 workflow task entries, Agent-driven plan/gate execution, model-validation compatibility flow, structured evidence, downloads/reports, audit history, Agent memory, and runtime branding.
- Evidence reviewed: current `README.md`, `README.zh-CN.md`, `docs/roadmap.md`, `docs/versioning.md`, `docs/branding.md`, `docs/notebook_contract.md`, `docs/runbook.md`, and the static FastAPI-served frontend.
- Roadmap reference: use `docs/roadmap.md` for version phases and Plugin/Tool/Hook/Workflow terminology. Keep this file focused on product experience and interface decisions.

## Brand
- Personality: a professional local tool with a macOS division of labour: flat content surfaces, three-dimensional icons, and glass only on controls that float above content.
- Trust signals: clear task state, human-readable execution evidence, stable report output paths, visible errors near the triggering action, and auditable Agent statements.
- Avoid: marketing hero pages, coloured gradient wallpapers, full-screen or decorative glass,
  blur or fade bands over content, neon glows, raised glossy buttons, decorative coloured
  borders or rings (including around confirmation cards), gradient-tile icon backgrounds,
  generic AI claims, and JSON as the default user-facing result.
- Default public brand: `MARVIS-全能风控智能体`.
- Branding must be runtime-configurable without source-code edits. The configurable surface is logo, favicon/web logo, primary theme color, platform display name, and browser page title.
- Private/customer branding belongs in `workspace/branding/brand.json` plus sibling assets, and must not be required for the open-source checkout to run.
- When no branding config exists, the platform uses the public MARVIS defaults: neutral charcoal primary color and the built-in MARVIS logo/favicon.

## Product goals
- Goals: make MARVIS a local-first credit-risk agent that assists governed validation, modeling, analysis, strategy, and monitoring work through structured workflows.
- Current V2 goal: let users run real credit-risk workflows for data processing, feature analysis, model development, model validation, strategy, Vintage/risk analysis, monitoring, and portfolio-style analysis through governed Agent conversations and structured controls.
- Compatibility goal: keep model validation stable and demonstrable through task creation, material scanning, notebook execution, deterministic validation evidence, Agent explanations, report conclusion confirmation, and Excel/Word output.
- Memory goal: let Agent use auditable cross-task memory for user preferences, field conventions, validation/modeling/strategy pitfalls, task experience, model experience, and workflow experience while preserving deterministic results.
- Runtime goal: let Agent understand a user goal, plan with available plugins/tools/hooks, execute controlled Python capabilities, and return structured evidence, artifacts, and report content.
- Non-goals: marketing homepage, arbitrary unreviewed code execution, report styling rewrites unrelated to structured task output, hidden memory use, or overclaiming unsupported workflows before their plugins exist.
- Success signals: one obvious next action, evidence readable without code knowledge, Agent explanations traceable to current evidence or memory references, and generated reports grounded in structured results.

## Personas and jobs
- Primary personas: credit-risk modelers, risk analysts, strategy operators, and model validation staff who can operate a local web tool but should not need to read raw JSON or manually wire scripts together.
- Current user jobs: join and clean datasets, analyze features, train and compare models, validate submitted models, design and backtest strategies, inspect Vintage/risk signals, run monitoring or portfolio-style analysis, and export auditable reports/artifacts.
- Memory-backed user jobs: receive useful historical reminders inside Agent analysis, understand whether a new result improved or declined compared with comparable historical work, reuse known field conventions and pitfalls, and manage memory entries when needed.
- Future user jobs: scheduled monitoring, multi-user review/approval, real-time scoring handoff, stronger portfolio management loops, and governed third-party workflow packs.
- Key contexts of use: local Jupyter terminal / notebook proxy, internal network, repeated credit-risk analysis and validation tasks.

## Information architecture
- Primary navigation: brand-configurable left task sidebar with platform title, task list, task search, create action, and settings for sort/group/theme.
- Core current screen: task navigation on the left (resizable, 200-380px, hover edge shows the handle, double-click resets), the task conversation in the center as the only task view, and a floating glass plan panel on the right that collapses into a progress capsule in the toolbar. There are no Conversation / Evidence / Report tabs; evidence and report drafts are cards inside the conversation. A toolbar switch shows everything or results only (result cards, confirmation cards and artifacts). The manual validation document canvas remains available in its existing flow.
- Agent surface: the center-column conversation, read at 720px or narrower. Agent messages carry the mascot head as avatar; tool activity, self-handled problems, confirmations, results and artifacts are cards in the flow. Governance and extension panels (business acceptance, blind OOT, historical replay, external data, institution adapters, scheduled analysis, collection, model monitoring) live in the Inspector popover, not above the conversation. Setup panels appear in the flow only when the current step needs them.
- Memory surface: memory should appear inside Agent explanations, warnings, comparison summaries, report-draft rationale, and workflow choices. Do not add a fixed top/center memory block that lists matched memories.
- Memory management surface: settings or audit management view for listing, inspecting, disabling, deleting, and exporting memory audit records. It is not a default task dashboard panel.
- Plugin/tool surface: plugin/tool execution output should appear as native evidence sections, tables, charts, findings, and report sections.

## Design principles
- Principle 1: creation starts from the home composer and continues in the task conversation; the Agent infers the task type and asks for missing parameters with confirmation cards. The create-task dialog remains as advanced settings, and manual/Agent mode selection stays there.
- Principle 2: preserve the current model and task context. Report drafts are an editable card in the conversation (most important fields open, the rest folded with a count of empty fields), with a persistent save state, Word preview and the confirmation action in the card; long drafts open a full-text editing layer over the same task. Finished plan steps in the right panel jump to their card in the conversation.
- Principle 3: default to human-readable summaries; raw data stays available behind details.
- Principle 4: every Agent statement should be grounded in current task evidence, explicit memory references, or clearly labeled general explanation.
- Principle 5: extension output must look native. A plugin/tool may add a table, chart, finding, or report section, but it should not introduce a separate visual language or bypass platform rendering.
- Tradeoffs: compact internal-tool density over decorative presentation; auditability over conversational smoothness when the two conflict.

## Visual language
- Color: light theme uses a white canvas (`#ffffff`), grey content boxes and sidebar (`#f4f4f6`), a `#e2e2e7` selected row, and white controls and floating layers; the home composer stays white with a 1px `#e2e2e7` edge. Dark theme keeps `#181818` canvas, `#1f1f1f` sidebar and `#2d2d2d` cards and controls. The default primary color is neutral charcoal (`#343438`); configured primary color drives primary actions such as create-task and Agent send.
- Typography: system font, 12 / 12.5 / 13 / 14 / 14.5 / 15 / 17 / 20 / 24 size scale (tokens `--fs-*`); weights 400 / 500 / 600 / 700. Conversation prose uses 14.5px; report prose 14-15px with comfortable line height; internal template keys belong in expandable details.
- Spacing/layout rhythm: the default wide desktop shell uses a 248px task sidebar (resizable 200-380px) and a 288px floating plan panel; the conversation column centres in the space left of the panel and widens when the panel is collapsed.
- Shape/radius/elevation: radius scale 8 (controls) / 12 (nested boxes) / 14 (cards) / 18 (large cards, dialogs) / 22 (glass capsules) / pill (buttons, chips). Content boxes have no shadow; floating layers use one elevation (`--shadow-pop`); glass controls use `--glass-shadow`.
- Motion: generic surfaces do not move on hover. Icons may lift slightly on hover. The mascot head is the only element that animates on its own (state expressions); a step card may flash briefly when the plan panel jumps to it. Motion stops under `prefers-reduced-motion`.
- Imagery/iconography: logo and favicon, plus the mascot head as the Agent avatar, built from two layers of the real logo pixels (`face-base.png` head without eyes, `face-eyes.png` eyes). CSS transforms only the eye layer to show five states: idle blink, working scan, waiting tilt with wide eyes, done smiling eyes, failed flat red eyes. Keep the product's existing icons: task-type layered glyphs with `--tone-*`, composer chip icons, settings navigation icons, bell, gear, search, create. Sidebar task status uses small dots (working blue pulse, waiting amber, failed red). No redrawn mascot, no glowing orb, no gradient icon tiles.

## Components
- Existing components to reuse: static FastAPI-served HTML/CSS/JS, current task/report API ids, existing task evidence sections, current Agent conversation UI.
- Glass components: only controls that float above the conversation use glass (`.ui-glass`): toolbar capsules, the results filter, Inspector button and popover, the plan panel and its collapsed capsule, reply chips and the composer. Content passes underneath them; there are no blur or fade bands. Glass never wraps Agent prose or content cards.
- Agent components: LLM settings, center-column conversation, mascot avatar, tool activity list (status, duration, progress), self-handled problem card (found / handled / checked, with log and undo), confirmation card (selectable options, a primary action, typing still accepted), picked-answer echo, result cards (task- and step-specific: metric tiles, charts with legend, hover readout and table view, expandable details, file rows), report draft card, artifact rows, reply chips, Word conclusion confirmation gate. Confirmation never requires typing a fixed word such as 继续.
- Reply contract: the backend declares one-click replies as message metadata `quick_replies` (label + the literal command the turn handler already accepts) and Agent-only guidance as `reply_hint` (`marvis/agent/quick_replies.py`). Message content stays mode-neutral, so the frontend never strips or rewrites instructions from free text. High-risk confirmations keep their full explicit phrase as the chip command, so a click authorises exactly what typing would.
- Branding components: runtime brand loader, brand config schema, sidebar logo/name binding, favicon/title binding, CSS primary-color token, and public default MARVIS logo/favicon.
- Memory components: inline memory-aware Agent statements, expandable memory references on Agent messages, memory management/audit view, memory disable/delete actions, and memory-use audit metadata.
- Workflow/runtime components: plugin registry, tool run evidence panel, hook run evidence panel, workflow plan/right-rail view, extension metric tables, extension report sections, and plugin output display declarations.
- Variants and states: primary/secondary/disabled/loading/error/success buttons; empty/loading/error/success summaries; high/medium/low confidence memory comparison states; disabled memory state; deleted-memory reference state.
- Token/component ownership: global tokens and shell components live in
  `marvis/static/styles.css`; V2 workflow components live in
  `marvis/static/css/v2-workbench.css` and consume the same global tokens.

## Accessibility
- Target standard: practical WCAG AA for contrast and keyboard focus on the local tool.
- Keyboard/focus behavior: visible `:focus-visible`, task list keyboard navigation, Enter task creation, Cmd/Ctrl+S save report text where supported.
- Contrast/readability: muted text must remain readable on white. Sidebar task names stay on one line with ellipsis; the full name, owner, and created time appear in a hover/focus preview card to the right of the row.
- Screen-reader semantics: status and alert areas near the relevant actions; raw data and memory references are labelled.
- Reduced motion and sensory considerations: no layout-shifting hover motion;
  entrance and progress animation must stop under `prefers-reduced-motion`.

## Responsive behavior
- Current supported QA scope: wide desktop at 1440, 1600 and 1920 CSS pixels. Narrow/mobile adaptations require a separate explicit product decision; existing fallback CSS is not a claim of verified mobile support.
- Layout adaptations: preserve resizable desktop columns and internal horizontal scrolling for the model switcher. Additional models must not widen the status bar or the page.
- Touch/hover differences: controls must work without hover-only affordances.

## Interaction states
- Loading: action buttons show busy text and disable conflicting actions.
- Empty: tells the user what to do next, not just what is missing.
- Error: shown near the action bar and in the affected result section when possible.
- Success: summary panel shows output or evidence in human-readable form.
- Disabled: buttons carry titles explaining why they cannot be used.
- Offline/slow network: long notebook/report actions remain visibly busy.
- Memory comparison uncertainty: medium-confidence matches must be textually marked as requiring human confirmation. Low-confidence matches should not be used for historical comparison.

## Validation editing and creation
- Each model owns a draft buffer keyed by task ID and draft version. Autosave writes narrative fields to the local task database. It neither confirms the report nor changes metrics, task execution or Agent memory. Never persist report prose in browser storage.
- Distinguish draft saving from report confirmation. Show pending, saving, saved, failed and conflict states. A stale write must fail visibly, preserve edits and offer a version comparison; a draft confirmed in another window must not silently discard this window's changes.
- Confirmation binds the report revision, draft message ID and edit revision. Batch confirmation flushes all edited pending models first; its visible scope includes pending, failed, cancelled and unready models. Report confirmation means consent to generate documents, not that a model passed validation.
- Creation starts from what the user says and the materials they attach; the Agent asks for names and missing parameters in the conversation. Background narratives are optional, empty until supplied; examples are placeholders, never assumed business facts. Keep required data dictionary and execution contracts intact.
- Use one selected execution mode. Single-model creation prefers Agent when a model is configured, otherwise manual. Multiple-model validation always uses the existing automatic Agent workbench and explains that choice.
- Collapse previous model details when adding another model. Keep the total, material-selection summary and create action visible in a fixed footer; validation errors reveal the relevant model without clearing other inputs.
- Completed step groups collapse by default; active and exceptional groups expand. Respect the user's subsequent expansion choice per model. Preserve task-local view and scroll positions during navigation.
- The model switcher is a radiogroup with an accessible model/status label, one checked value, roving focus, and Arrow / Home / End navigation.

## Strategy navigation and visual restraint
- Use the existing seven workflow stages to filter relevant tools. Search addresses all tools; expert access and contextual candidate actions preserve their original governed handlers and evidence bindings.
- Collapse empty analysis categories, empty history and empty evidence details. Keep existing results accessible and expose structural risk flags; empty tasks must not show fictitious execution progress.
- Primary action labels use business language. Precise Tool, version, hash, action and provenance details remain available for inspection.
- Default task rows are compact and quiet; selected state carries the emphasis. Preserve single-line names, full hover/focus previews and keyboard access to delete.
- Radii follow the 8 / 12 / 14 / 18 / 22 / pill scale. Ordinary controls are flat with subtle fills; content boxes are flat; avoid hover translation. During task work, the chosen pet is small, stationary and docked in a reserved place so it never covers sidebar status or menus; retain user branding and welcome-screen preferences.

## Content voice
- Tone: concise operational Chinese in the product UI.
- Terminology: use "材料", "Notebook", "报告文案", "Word 报告", "历史对比", "记忆引用"; avoid backend lifecycle labels as UI labels.
- Internal identifiers stay out of prose: hashes, envelope / registry wording, raw enum values (`nested_same_cohort`) and control paths belong in evidence tables, metadata or the expandable technical detail; the sentence the user reads uses business words.
- Microcopy rules: no double-language eyebrow; no README-style explanatory paragraphs in every section; Agent should distinguish current evidence, historical memory, and general domain guidance. Memory should sound like operational guidance, not a separate "AI memory found X" system notice.

## Implementation constraints
- Framework/styling system: plain HTML/CSS/JS served by FastAPI. `styles.css`
  owns shared tokens and compatibility rules; feature-scoped styles live under
  `static/css/` and are loaded through the tested immutable stylesheet bundle.
- Design-token constraints: no new frontend dependency; keep proxy-safe relative paths.
- Branding config should live outside committed source defaults under `workspace/branding/`.
- Performance constraints: avoid unnecessary `backdrop-filter` and heavy shadows, and avoid
  repeated DOM rewrites during polling. `backdrop-filter` is limited to the glass controls listed
  above (blur 16px) and must keep a readable background where it is unsupported.
- Compatibility constraints: preserve existing API endpoints and legacy DOM ids used by tests unless a task explicitly migrates them.
- Test/screenshot expectations: smoke tests plus browser viewport verification after frontend changes.
- Memory constraints: memory can support explanation, comparison, field suggestions, pitfall warnings, report wording, and future workflow planning, but cannot alter deterministic validation metrics. Memory references must carry source, category, confidence, and audit metadata.
- Plugin constraints: plugin/tool/hook outputs must return structured results; platform owns Word/Excel/chart rendering.

## Open questions
- [ ] Memory management UI placement: settings modal first, then dedicated audit screen if volume requires it.
- [ ] Plugin upload UI: administrator-only, ordinary user with confirmation, or local developer-only for public builds.
- [x] Portfolio exposure: Portfolio Analysis is a governed first-screen task.
  Its currently verified public slice is the no-trend local workflow; production
  monitoring remains embedded in modeling/strategy and must not be advertised
  as a delivered standalone operations service.
- [ ] V2 multi-user governance: design maker-checker, role permissions, approval exports, production-operation confirmation, and recovery UX; the version scope is decided, only the interaction design remains open.
