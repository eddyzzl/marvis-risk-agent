// Home page (2026-10-14 redesign, batch 5, D7 + 5B): an input-first start.
// One sentence (plus optional files) opens a creation card in a short
// conversation: the card shows what the platform understood (task type, name,
// owner, materials, mode) and asks only for what the create contract still
// needs. "创建并开始" creates through the existing create dialog (contract
// unchanged) and hands the sentence to the new task's Agent conversation;
// "高级设置" opens that dialog prefilled. Icons and examples only fill the
// input; nothing here creates a task without the user's click.
import { escapeHtml } from "./ui-utils.js";
import { getUiLocale, localizeText, onUiLocaleChange, rememberTypedLanguage } from "./i18n.js";
import { agentFaceHtml } from "./agent-face.js";
import { droppedFileItems } from "./dialogs.js";
import {
  HOME_EXAMPLES,
  pickHomeExamples,
  HOME_MODEL_ALGORITHMS,
  HOME_RISK_ANALYSES,
  draftMissingFields,
  draftTaskName,
  homeStarterText,
  homeTemplateName,
  homeTemplateTokens,
  inferTaskType,
  parseFeatureMetrics,
  parseModelAlgorithms,
  parseOotKsGoal,
  parseValidatorName,
  parseRiskAnalysis,
  parseScopeNotes,
  parseStrategyLimits,
} from "./home-intent.js";
import {
  VALIDATION_ROLE_LABELS,
  groupValidationModels,
  modelMaterialGaps,
} from "./home-validation-models.js";
import { MAX_VALIDATION_BATCH_ROWS, validationFileRole } from "./validation-batch-create.js";

const MATERIAL_HINTS = {
  validation: "Notebook、样本、PMML 和数据字典",
  modeling: "建模样本（可附数据字典）",
  feature_analysis: "样本或特征表",
  data_join: "主表和要拼接的特征表",
  strategy: "评分或申请数据",
  vintage: "可以先不传，Agent 会按分析口径告诉你要哪些表",
  portfolio: "表现期快照",
};

const MISSING_LABELS = {
  taskType: "工作类型",
  name: "任务名称",
  validator: "负责人",
  files: "材料",
  strategyLimits: "通过率下限或坏率上限",
  algorithms: "建模算法",
  riskAnalysis: "分析类型",
  models: "每个模型的名称和四份材料",
};

const CONTINUE_LIMIT = 3;

// Native selects normally reserve room for their longest option. Measure
// the selected label instead so “High” does not inherit “Medium”'s width.
export function installComposerSelectSizing(doc = globalThis.document) {
  const view = doc?.defaultView;
  if (!doc?.body?.appendChild || !doc?.createElement || !view?.getComputedStyle) return;
  const selects = Array.from(doc.querySelectorAll(".agent-composer-chip select"));
  const sizeSelect = (select) => {
    const option = select.selectedOptions?.[0] || select.options?.[select.selectedIndex];
    if (!option) return;
    const label = option.textContent || "";
    const style = view.getComputedStyle(select);
    const probe = doc.createElement("span");
    probe.setAttribute("translate", "no");
    probe.setAttribute("aria-hidden", "true");
    Object.assign(probe.style, {
      position: "fixed", visibility: "hidden", whiteSpace: "pre",
      font: style.font, letterSpacing: style.letterSpacing,
    });
    probe.textContent = label;
    doc.body.appendChild(probe);
    const horizontal = ["paddingLeft", "paddingRight", "borderLeftWidth", "borderRightWidth"]
      .reduce((sum, key) => sum + (Number.parseFloat(style[key]) || 0), 0);
    const width = `${Math.ceil(probe.getBoundingClientRect().width + horizontal + 1)}px`;
    probe.remove();
    if (select.style.width !== width) select.style.width = width;
    if (select.title !== label) select.title = label;
  };
  const sizeAll = () => selects.forEach(sizeSelect);
  const Observer = view.MutationObserver;
  for (const select of selects) {
    select.addEventListener("change", () => sizeSelect(select));
    if (Observer) new Observer(() => sizeSelect(select)).observe(select, {
      childList: true, characterData: true, subtree: true,
    });
  }
  onUiLocaleChange(sizeAll, doc);
  doc.fonts?.ready?.then?.(sizeAll);
  sizeAll();
}

// Materials are {file, relativePath} items, the create dialog's own shape, so
// a dropped folder keeps its paths through the upload.
function fileItem(entry) {
  const file = entry?.file || entry;
  return { file, relativePath: entry?.relativePath || file?.webkitRelativePath || file?.name || "" };
}

// Two attachments with the same file name end up as one file on disk; say so
// before creating (eval S5 lost four of eight files without a word).
export function duplicateFileNames(items = []) {
  const counts = new Map();
  for (const item of items) {
    const name = String(item?.file?.name || item?.relativePath || "").split("/").pop();
    if (name) counts.set(name, (counts.get(name) || 0) + 1);
  }
  return [...counts].filter(([, count]) => count > 1).map(([name]) => name);
}

function fileKey(item) {
  return `${item.relativePath}:${item.file?.size || 0}:${item.file?.lastModified || 0}`;
}

// Focus survives the card's re-render: the control is found again by its data
// attribute (the 更改 link falls back to the selected type chip).
const FOCUS_ATTRS = [
  "homeDraftType", "homeDraftMode", "homeDraftAlgorithm", "homeDraftAnalysis", "homeDraftField",
  "homeDraftLimit", "homeDraftTypes", "homeDraftAdvanced", "homeDraftCreate", "homeDraftBack",
  "homeDraftSingle", "homeDraftFiles",
];

function focusKeyOf(node) {
  const key = FOCUS_ATTRS.find((attr) => node?.dataset?.[attr] !== undefined);
  return key ? { key, value: node.dataset[key] } : null;
}

function focusSelector({ key, value }) {
  const attr = `data-${key.replace(/[A-Z]/g, (char) => `-${char.toLowerCase()}`)}`;
  return value ? `[${attr}="${String(value).replace(/["\\]/g, "\\$&")}"]` : `[${attr}]`;
}

function formatSize(bytes) {
  const value = Number(bytes) || 0;
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

// 接着做: what is waiting on the user first, then what is still running, then
// failures; recency order (as given) within each group.
export function homeContinueTasks(tasks = [], statusOf = () => null) {
  const rank = { wait: 0, run: 1, bad: 2 };
  return (Array.isArray(tasks) ? tasks : [])
    .map((task) => ({ task, status: statusOf(task) }))
    .filter(({ status }) => status && rank[status.dot] !== undefined)
    .sort((left, right) => rank[left.status.dot] - rank[right.status.dot])
    .slice(0, CONTINUE_LIMIT);
}

// 接着做 progress from a structured step ({number, title}); no step, no text.
export function homeContinueProgress(step, dot = "") {
  const number = Number(step?.number);
  const title = String(step?.title || "").trim();
  if (!Number.isInteger(number) || number < 1 || !title) return "";
  if (dot === "run") return `正在第 ${number} 步：${title}`;
  if (dot === "bad") return `第 ${number} 步失败：${title}`;
  return `停在第 ${number} 步：${title}`;
}

export function createHomeComposer({
  $,
  taskTypeDefinitions,
  taskTypeDisplayOrder,
  hasEnabledAgent = () => false,
  getTasks = () => [],
  statusOf = () => null,
  taskName = (task) => task?.model_name || "",
  // The task's current step as {number, title}, or null when unknown.
  currentStepOf = (task) => task?.current_step || null,
  kindIconHtml = () => "",
  openTask = () => {},
  storage = globalThis.localStorage,
  onCreate = async () => false,
  onAdvanced = () => {},
  // A failure that arrives after the card is gone (the task exists, or the
  // card was left) is reported here instead.
  onError = () => {},
  doc = globalThis.document,
} = {}) {
  let files = [];
  let pickedType = "";
  let starterText = "";
  let draft = null;
  let creating = false;
  let createSeq = 0;
  let lastContinueHtml = null;
  let exampleIndices = [];
  // Work-icon template mode: the editable sentence replaces the textarea's
  // display; roleTarget is the file slot whose picker is open.
  let templateActive = false;
  let roleTarget = null;
  let defaultInputSource = null;
  let modelSeq = 0;

  const definitionOf = (taskType) => taskTypeDefinitions?.[taskType] || null;
  const labelOf = (taskType) => definitionOf(taskType)?.label || "";

  function lastValidator() {
    try {
      return String(storage?.getItem?.("marvis_last_validator") || "");
    } catch (_) {
      return "";
    }
  }

  function defaultRunMode(taskType) {
    const definition = definitionOf(taskType);
    if (definition && definition.manualEnabled === false) return "agent";
    return hasEnabledAgent() ? "agent" : "manual";
  }

  // ---------- home view ----------

  function syncSend() {
    const send = $("homeComposerSend");
    const input = $("homeComposerInput");
    if (send) send.disabled = !(input?.value.trim() || files.length);
  }

  function filesChipsHtml(list, { removable }) {
    return list.map((item, index) => [
      `<span class="home-file-chip" title="${escapeHtml(item.relativePath)}">`,
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/></svg>',
      item.role ? `<span class="home-file-chip-role">${escapeHtml(item.role)}</span>` : "",
      `<span class="home-file-chip-name">${escapeHtml(item.file?.name || item.relativePath)}</span>`,
      `<span class="home-file-chip-size">${escapeHtml(formatSize(item.file?.size))}</span>`,
      removable
        ? `<button type="button" class="home-file-chip-remove" data-home-file-remove="${index}" aria-label="移除 ${escapeHtml(item.relativePath)}">×</button>`
        : "",
      "</span>",
    ].join("")).join("");
  }

  function renderFiles() {
    const box = $("homeComposerFiles");
    if (!box) return;
    box.innerHTML = filesChipsHtml(files, { removable: true });
    box.hidden = !files.length;
    syncSend();
  }

  function addFiles(list, role = "") {
    const known = new Set(files.map(fileKey));
    for (const entry of Array.from(list || [])) {
      const item = fileItem(entry);
      if (!item.file) continue;
      if (known.has(fileKey(item))) {
        // Re-attaching a file through a slot gives it that slot's role.
        if (role) {
          const existing = files.find((candidate) => fileKey(candidate) === fileKey(item));
          if (existing) existing.role = role;
        }
        continue;
      }
      if (role) item.role = role;
      files.push(item);
      known.add(fileKey(item));
    }
    renderFiles();
    refreshFileSlots();
    if (draft) {
      if (draft.models) {
        // Files dropped on the page while the model list is open: regroup
        // when they make a clear set of models, otherwise leave them unplaced.
        const grouped = groupValidationModels(files);
        if (grouped) {
          draft.models = grouped.models.map((model) => newModel(model.name, model.items));
          draft.unassigned = grouped.unassigned;
        } else {
          const placed = new Set(modelPoolItems().map(fileKey));
          draft.unassigned = [...(draft.unassigned || []), ...files.filter((item) => !placed.has(fileKey(item)))];
        }
      }
      draft.files = files.slice();
      syncValidationModels();
      renderDraft();
    }
  }

  function syncExampleScrollEdges() {
    const box = $("homeExamples");
    if (!box) return;
    const left = box.scrollLeft || 0;
    box.dataset.scrollLeft = String(left > 1);
    box.dataset.scrollRight = String(box.scrollWidth - box.clientWidth - left > 1);
  }

  function renderExamples({ shuffle = true } = {}) {
    const box = $("homeExamples");
    if (!box) return;
    if (shuffle || !exampleIndices.length) exampleIndices = pickHomeExamples();
    box.innerHTML = exampleIndices.map((index) => (
      `<button type="button" class="home-example" data-home-example="${index}" data-home-example-type="${escapeHtml(HOME_EXAMPLES[index].taskType)}">${escapeHtml(localizeText(HOME_EXAMPLES[index].text))}</button>`
    )).join("");
    syncExampleScrollEdges();
  }

  function renderContinue() {
    const section = $("homeContinue");
    const list = $("homeContinueList");
    if (!section || !list) return;
    const rows = homeContinueTasks(getTasks(), statusOf);
    const html = rows.map(({ task, status }) => {
      const name = taskName(task);
      const progress = homeContinueProgress(currentStepOf(task), status.dot);
      return [
        `<button type="button" class="home-continue-row" data-home-open-task="${escapeHtml(String(task.id || ""))}">`,
        kindIconHtml(task),
        `<span class="home-continue-name" title="${escapeHtml(name)}">${escapeHtml(name)}</span>`,
        `<span class="home-continue-status" data-dot="${escapeHtml(status.dot)}"${progress ? ` title="${escapeHtml(progress)}"` : ""}>`,
        `<span class="task-status-dot ${escapeHtml(status.dot)}" aria-hidden="true"></span><span class="home-continue-progress">${escapeHtml(progress || status.label)}</span>`,
        "</span>",
        '<svg class="home-continue-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"><path d="M9 6l6 6-6 6"/></svg>',
        "</button>",
      ].join("");
    }).join("");
    if (html === lastContinueHtml) return;
    lastContinueHtml = html;
    list.innerHTML = html;
    section.hidden = !rows.length;
  }

  // ---------- work-icon templates ----------

  const FILE_SLOT_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"><path d="M20.5 11.5 12 20a5.5 5.5 0 0 1-7.8-7.8l8.5-8.5a3.7 3.7 0 0 1 5.2 5.2L9.4 17.4a1.8 1.8 0 0 1-2.6-2.6l7.8-7.8"/></svg>';

  function templateHtml(tokens) {
    return tokens.map((token) => {
      if (typeof token === "string") return escapeHtml(token);
      if (token.slot !== undefined) {
        // The person slot starts from the last name used, when there is one.
        const value = token.key === "validator" ? lastValidator() || token.slot : token.slot;
        return `<span class="home-slot" data-home-slot>${escapeHtml(value)}</span>`;
      }
      const uploadTitle = getUiLocale() === "en" ? `Click to upload ${token.label}` : `点击上传${token.label}`;
      return [
        `<button type="button" class="home-file-slot" contenteditable="false" data-file-role="${escapeHtml(token.role || token.label)}"${token.multiple ? " data-multiple" : ""} title="${escapeHtml(uploadTitle)}">`,
        FILE_SLOT_ICON,
        `<span class="home-file-slot-label">${escapeHtml(token.label)}</span>`,
        '<span class="home-file-slot-names"></span>',
        "</button>",
      ].join("");
    }).join("");
  }

  // A template sentence is named from its value slots, while the picked type
  // still holds.
  function templateDraftName(taskType) {
    const box = $("homeComposerTemplate");
    if (!templateActive || !box || taskType !== pickedType) return "";
    const slots = Array.from(box.querySelectorAll("[data-home-slot]")).map((slot) => slot.textContent);
    return homeTemplateName(taskType, slots, getUiLocale());
  }

  function roleFileNames(role) {
    return files.filter((item) => item.role === role).map((item) => item.file?.name || item.relativePath);
  }

  // The sentence the Agent receives: slots read as their text, file slots as
  // the role plus the attached file names (e.g. Y样本「applications.csv」).
  function serializeTemplate(node) {
    let text = "";
    for (const child of Array.from(node?.childNodes || [])) {
      if (child.nodeType === 3) {
        text += child.nodeValue;
      } else if (child.nodeType === 1) {
        if (child.matches?.(".home-file-slot")) {
          const role = child.dataset.fileRole || "";
          const names = roleFileNames(role);
          const label = child.querySelector?.(".home-file-slot-label")?.textContent || localizeText(role);
          text += names.length ? `${label}「${names.join("、")}」` : label;
        } else if (child.tagName === "BR") {
          text += "\n";
        } else {
          text += serializeTemplate(child);
        }
      }
    }
    return text;
  }

  function refreshFileSlots() {
    const box = $("homeComposerTemplate");
    if (!box || !templateActive) return;
    const untouched = $("homeComposerInput")?.value === starterText;
    for (const slot of Array.from(box.querySelectorAll(".home-file-slot"))) {
      const names = roleFileNames(slot.dataset.fileRole || "");
      const target = slot.querySelector(".home-file-slot-names");
      if (target) target.textContent = names.length ? `「${names.join("、")}」` : "";
      slot.classList.toggle("has-files", names.length > 0);
    }
    syncFromTemplate();
    // Attaching/removing a file changes the serialized sentence, but is not
    // an edit to the platform's default wording.
    if (untouched) starterText = $("homeComposerInput")?.value || "";
  }

  function syncFromTemplate() {
    const box = $("homeComposerTemplate");
    const input = $("homeComposerInput");
    if (!box || !input || !templateActive) return;
    const text = serializeTemplate(box).replace(/\u00a0/g, " ");
    input.value = text;
    if (!text.trim() && !box.querySelector(".home-file-slot")) {
      exitTemplate({ focus: true });
      return;
    }
    syncSend();
  }

  function enterTemplate(taskType, tokens, { focus = true } = {}) {
    const box = $("homeComposerTemplate");
    const input = $("homeComposerInput");
    if (!box || !input) return false;
    box.innerHTML = templateHtml(tokens);
    box.hidden = false;
    input.hidden = true;
    templateActive = true;
    pickedType = taskType;
    defaultInputSource = { kind: "template", taskType };
    refreshFileSlots();
    starterText = input.value;
    rememberTypedLanguage(starterText);
    // V3 UI-20: focus the sentence without preselecting a value (the blue
    // selection box read as a default choice); the caret starts at the front.
    if (focus) box.focus();
    if (focus && typeof doc?.createRange === "function") {
      const range = doc.createRange();
      range.selectNodeContents(box);
      range.collapse(true);
      const selection = doc.getSelection?.();
      selection?.removeAllRanges();
      selection?.addRange(range);
    }
    return true;
  }

  function exitTemplate({ focus = false } = {}) {
    const box = $("homeComposerTemplate");
    const input = $("homeComposerInput");
    templateActive = false;
    if (box) {
      box.hidden = true;
      box.innerHTML = "";
    }
    if (input) {
      input.hidden = false;
      if (focus) input.focus();
    }
  }

  function fillInput(text, taskType = "", source = null) {
    const input = $("homeComposerInput");
    if (!input) return;
    exitTemplate();
    input.value = text;
    pickedType = taskType;
    starterText = text;
    defaultInputSource = source;
    syncSend();
    input.focus();
    input.setSelectionRange?.(text.length, text.length);
  }

  // ---------- creation card ----------

  function setMode(mode) {
    const root = $("workspaceWelcome");
    if (root) root.dataset.homeMode = mode;
    const box = $("homeDraft");
    if (box) box.hidden = mode !== "draft";
  }

  // ---------- several models (batch validation) ----------

  const newModel = (name = "", items = []) => ({ id: `home-model-${++modelSeq}`, name, items: items.slice() });
  const isBatch = () => draft?.taskType === "validation" && Array.isArray(draft?.models) && draft.models.length >= 2;

  // Validation only: files that hold two or more models, or a sentence that
  // asks for several, turn the card into a model list. "只验证一个模型"
  // (draft.single) keeps it a single model.
  function syncValidationModels() {
    if (!draft) return;
    if (draft.taskType !== "validation") {
      draft.models = null;
      draft.unassigned = [];
      return;
    }
    if (draft.models || draft.single) return;
    const grouped = groupValidationModels(draft.files);
    if (grouped) {
      draft.models = grouped.models.map((model) => newModel(model.name, model.items));
      draft.unassigned = grouped.unassigned;
    } else if (draft.multiModel) {
      draft.models = [newModel("", draft.files), newModel()];
      draft.unassigned = [];
    } else {
      return;
    }
    // Several models run as one batch under Agent auto-review.
    draft.runMode = "agent";
  }

  function modelPoolItems() {
    if (!draft?.models) return draft?.files || [];
    const seen = new Set();
    const pool = [];
    for (const item of [...draft.models.flatMap((model) => model.items), ...(draft.unassigned || [])]) {
      const key = fileKey(item);
      if (seen.has(key)) continue;
      seen.add(key);
      pool.push(item);
    }
    return pool;
  }

  // Back to one model: its files (and anything unplaced) become the card's.
  function collapseToSingleModel(model) {
    const kept = model ? [...model.items, ...(draft.unassigned || [])] : modelPoolItems();
    if (model?.name && !draft.nameEdited) draft.name = model.name;
    draft.models = null;
    draft.unassigned = [];
    draft.multiModel = false;
    draft.single = true;
    files = kept.slice();
    draft.files = files.slice();
    renderFiles();
  }

  function addModel() {
    draft.single = false;
    if (!draft.models) {
      draft.models = [newModel(draft.name, draft.files), newModel()];
      draft.unassigned = [];
    } else if (draft.models.length >= MAX_VALIDATION_BATCH_ROWS) {
      draft.error = `一次最多验证 ${MAX_VALIDATION_BATCH_ROWS} 个模型。`;
      return;
    } else {
      draft.models.push(newModel());
    }
    draft.runMode = "agent";
    draft.error = "";
  }

  function addModelFiles(index, list) {
    const model = draft?.models?.[index];
    if (!model) return;
    const known = new Set(model.items.map(fileKey));
    for (const entry of Array.from(list || [])) {
      const item = fileItem(entry);
      if (!item.file || known.has(fileKey(item))) continue;
      model.items.push(item);
      known.add(fileKey(item));
    }
    files = modelPoolItems().slice();
    draft.files = files.slice();
    draft.error = "";
    renderFiles();
    renderDraft();
  }

  function startDraft() {
    const input = $("homeComposerInput");
    const text = String(input?.value || "").trim();
    if (!text && !files.length) return;
    const inference = inferTaskType(text);
    // A picked icon/example holds while its sentence is unchanged. Once the
    // sentence is rewritten it only survives as a guess, and only if the new
    // sentence does not clearly say something else.
    const edited = Boolean(pickedType) && text !== starterText;
    const keepPicked = Boolean(pickedType) && (!edited || (
      inference.confidence !== "high"
      && (!inference.taskType || inference.candidates.includes(pickedType))
    ));
    const taskType = keepPicked ? pickedType : inference.taskType || "";
    const confidence = keepPicked ? (edited ? "low" : "high") : inference.confidence;
    draft = {
      text,
      files: files.slice(),
      taskType,
      typeSource: keepPicked ? "picked" : inference.taskType ? "inferred" : "",
      confidence,
      candidates: inference.candidates,
      multiModel: inference.multiModel,
      name: templateDraftName(taskType) || draftTaskName(text, taskType),
      nameEdited: false,
      // A clear guess shows as one chip with 更改; an unclear one lists all.
      showTypes: !taskType || confidence !== "high",
      validator: parseValidatorName(text) || lastValidator(),
      runMode: defaultRunMode(taskType),
      strategy: parseStrategyLimits(text),
      algorithms: parseModelAlgorithms(text),
      ootKsMin: parseOotKsGoal(text),
      featureMetrics: parseFeatureMetrics(text),
      analysis: parseRiskAnalysis(text),
      ...scopeNotesOf(text),
      models: null,
      unassigned: [],
      single: false,
      error: "",
    };
    syncValidationModels();
    setMode("draft");
    renderDraft();
    focusFirstGap();
  }

  // The card opens on the first thing it asks for (or on 创建并开始).
  function focusFirstGap() {
    const box = $("homeDraft");
    const first = missing()[0];
    const selector = {
      taskType: "[data-home-draft-type]",
      name: '[data-home-draft-field="name"]',
      validator: '[data-home-draft-field="validator"]',
      riskAnalysis: "[data-home-draft-analysis]",
      files: "[data-home-draft-files]",
      strategyLimits: "[data-home-draft-limit]",
      algorithms: "[data-home-draft-algorithm]",
      models: firstModelGapSelector(),
    }[first] || "[data-home-draft-create]";
    box?.querySelector?.(selector)?.focus?.();
  }

  function firstModelGapSelector() {
    if (!draft?.models) return "[data-home-model-add]";
    const unnamed = draft.models.findIndex((model) => !String(model.name || "").trim());
    if (unnamed >= 0) return `[data-home-model-name="${unnamed}"]`;
    const incomplete = draft.models.findIndex((model) => modelMaterialGaps(model.items).length);
    return incomplete >= 0 ? `[data-home-model-files="${incomplete}"]` : "[data-home-model-add]";
  }

  function missing() {
    if (!draft) return [];
    return draftMissingFields(draft, { deferredMaterials: Boolean(definitionOf(draft.taskType)?.deferredMaterials) });
  }

  function typeChipsHtml() {
    if (!draft.showTypes && draft.taskType) {
      return [
        `<button type="button" class="home-draft-chip" aria-pressed="true" data-home-draft-type="${escapeHtml(draft.taskType)}">${escapeHtml(labelOf(draft.taskType))}</button>`,
        '<button type="button" class="home-draft-link" data-home-draft-types>更改</button>',
      ].join("");
    }
    const order = Array.isArray(taskTypeDisplayOrder) && taskTypeDisplayOrder.length
      ? taskTypeDisplayOrder
      : Object.keys(taskTypeDefinitions || {});
    return order
      .filter((taskType) => definitionOf(taskType)?.dialogTitle && definitionOf(taskType)?.available !== false)
      .map((taskType) => (
        `<button type="button" class="home-draft-chip" data-home-draft-type="${escapeHtml(taskType)}" aria-pressed="${draft.taskType === taskType}">${escapeHtml(labelOf(taskType))}</button>`
      )).join("");
  }

  function runModeHtml() {
    const definition = definitionOf(draft.taskType);
    const agentAvailable = hasEnabledAgent() || definition?.manualEnabled === false;
    const options = [
      ["agent", "Agent", agentAvailable || isBatch()],
      ["manual", "手动", definition?.manualEnabled !== false && !isBatch()],
    ];
    return options.map(([mode, label, enabled]) => (
      `<button type="button" class="home-draft-chip" data-home-draft-mode="${mode}" aria-pressed="${draft.runMode === mode}"${enabled ? "" : " disabled"}>${label}</button>`
    )).join("");
  }

  function scopeNotesOf(text) {
    const notes = parseScopeNotes(text);
    return { collections: notes.collections, antiFraud: notes.antiFraud, unsupported: notes.unsupported };
  }

  // D-7 / 5.7: collections and anti-fraud have no task type of their own;
  // the card says where the work happens and what is out of reach.
  function scopeNotesHtml() {
    const notes = [];
    if (draft.collections && draft.taskType === "strategy") {
      notes.push("催收在策略任务里做：创建后打开右上角「检查器」里的「催收参考工作台」，可以登记案件、还款计划和回款流水，按分案规则生成参考批次，复核后做本地参考排队和执行，再核对成本与回收。");
      if (draft.unsupported?.length) {
        draft.unsupported.forEach((label) => notes.push(`现在做不到：${label}。`));
        notes.push("平台不联系客户，也还没有经过评审的催收动作、成本、产能和回收口径。");
      }
    }
    if (draft.antiFraud) {
      notes.push("反欺诈没有单独的工作流：有欺诈标签就按建模做，欺诈规则用策略任务，设备、关系和事件特征在策略任务「检查器」的「外部数据与事件特征」里。平台不做设备指纹、关系图谱识别或实时拦截。");
    }
    return notes.map((note) => `<p class="home-draft-note" role="note">${escapeHtml(note)}</p>`).join("");
  }

  function strategyLimitsHtml() {
    const limit = (key, label, placeholder) => [
      '<label class="home-draft-limit">',
      `<span>${label}</span>`,
      `<input type="number" inputmode="decimal" min="0" max="100" step="0.1" data-home-draft-limit="${key}" value="${escapeHtml(draft.strategy?.[key] ?? "")}" placeholder="${placeholder}" />`,
      '<span aria-hidden="true">%</span>',
      "</label>",
    ].join("");
    return [
      '<div class="home-draft-row"><span class="home-draft-label">策略约束</span><div class="home-draft-limits">',
      limit("minApprovalRate", "通过率不低于", "60"),
      limit("maxBadRate", "坏率不高于", "5"),
      "<small>至少填一个；按“约束坏率下最大化通过率”开发，利润目标请用高级设置。</small>",
      "</div></div>",
    ].join("");
  }

  function algorithmsHtml() {
    const chosen = new Set(draft.algorithms || []);
    return [
      '<div class="home-draft-row"><span class="home-draft-label">建模算法</span>',
      '<div class="home-draft-chips" role="group" aria-label="建模算法">',
      HOME_MODEL_ALGORITHMS.map((item) => (
        `<button type="button" class="home-draft-chip" data-home-draft-algorithm="${item.id}" aria-pressed="${chosen.has(item.id)}">${escapeHtml(item.label)}</button>`
      )).join(""),
      "</div></div>",
    ].join("");
  }

  function riskAnalysisHtml() {
    return [
      '<div class="home-draft-row"><span class="home-draft-label">分析类型</span>',
      '<div class="home-draft-chips" role="group" aria-label="分析类型">',
      HOME_RISK_ANALYSES.map((item) => (
        `<button type="button" class="home-draft-chip" data-home-draft-analysis="${escapeHtml(item.command)}" aria-pressed="${draft.analysis === item.command}">${escapeHtml(item.command)}</button>`
      )).join(""),
      "</div></div>",
    ].join("");
  }

  function leadHtml(gaps) {
    if (!draft.taskType) {
      // Nothing in the sentence names a risk workflow: say what the platform
      // does instead of silently offering the list (eval round 1).
      return draft.confidence === "none"
        ? "这是哪一类工作？选一下，我按它准备。我能做数据处理、特征分析、风险分析、组合分析、模型开发、模型验证和策略开发；订票、写作这类别的事情我做不了。"
        : "这是哪一类工作？选一下，我按它准备。";
    }
    const label = labelOf(draft.taskType);
    const understood = isBatch()
      ? `我理解为「${escapeHtml(label)}」任务，共 ${draft.models.length} 个模型，由 Agent 逐个自动审查，另出一份汇总 Excel。`
      : draft.confidence === "low"
        ? `我猜是「${escapeHtml(label)}」，不对的话换一个。`
        : `我理解为「${escapeHtml(label)}」任务。`;
    const pending = gaps.filter((field) => field !== "taskType").map((field) => MISSING_LABELS[field]);
    const ready = draft.runMode === "manual"
      ? "都齐了，确认后就创建，接下来按步骤逐项确认。"
      : "都齐了，确认后就开始；字段和口径我会在对话里再跟你确认。";
    const ask = pending.length ? `开始前还需要：${pending.join("、")}。` : ready;
    return `${understood}${ask}`;
  }

  function materialsRowHtml(materialHint) {
    const duplicates = duplicateFileNames(draft.files);
    return [
      '<div class="home-draft-row"><span class="home-draft-label">材料</span><div class="home-draft-files">',
      draft.files.length ? `<div class="home-file-chips">${filesChipsHtml(draft.files, { removable: true })}</div>` : "",
      '<div class="home-draft-file-actions">',
      `<label class="home-draft-add">添加文件<input type="file" multiple class="visually-hidden" data-home-draft-files /></label>`,
      draft.taskType === "validation"
        ? '<button type="button" class="home-draft-link" data-home-model-add>添加模型</button>'
        : "",
      "</div>",
      `<small>${escapeHtml(materialHint)}</small>`,
      duplicates.length
        ? `<small class="home-draft-warning" role="note">有同名文件：${escapeHtml(duplicates.join("、"))}。同名文件上传后只会保留一份${draft.taskType === "validation" ? "；不同模型的材料请点「添加模型」分开放" : ""}。</small>`
        : "",
      "</div></div>",
    ].join("");
  }

  function modelFileChipsHtml(items, modelIndex) {
    return items.map((item, index) => {
      const role = VALIDATION_ROLE_LABELS[validationFileRole(item.file)] || "";
      return [
        `<span class="home-file-chip" title="${escapeHtml(item.relativePath)}">`,
        role ? `<span class="home-file-chip-role">${escapeHtml(role)}</span>` : "",
        `<span class="home-file-chip-name">${escapeHtml(item.file?.name || item.relativePath)}</span>`,
        `<button type="button" class="home-file-chip-remove" data-home-model-file-remove="${modelIndex}:${index}" aria-label="移除 ${escapeHtml(item.relativePath)}">×</button>`,
        "</span>",
      ].join("");
    }).join("");
  }

  // Two or more models: one row each (name, its materials, what it still
  // lacks), then 添加模型. The batch is named by date and model count.
  function modelsRowHtml() {
    const rows = draft.models.map((model, index) => {
      const gaps = modelMaterialGaps(model.items);
      return [
        `<div class="home-draft-model" data-home-model="${index}">`,
        '<div class="home-draft-model-head">',
        `<span class="home-draft-model-index" aria-hidden="true">${index + 1}</span>`,
        `<input type="text" data-home-model-name="${index}" value="${escapeHtml(model.name)}" placeholder="模型名称" maxlength="80" aria-label="模型 ${index + 1} 名称" />`,
        `<button type="button" class="home-draft-model-remove" data-home-model-remove="${index}" aria-label="移除模型 ${index + 1}" title="移除模型">×</button>`,
        "</div>",
        model.items.length ? `<div class="home-file-chips">${modelFileChipsHtml(model.items, index)}</div>` : "",
        '<div class="home-draft-model-foot">',
        gaps.length
          ? `<small class="home-draft-model-gaps">还缺：${gaps.map((role) => `<span>${escapeHtml(VALIDATION_ROLE_LABELS[role])}</span>`).join("、")}</small>`
          : "",
        `<label class="home-draft-add">添加文件<input type="file" multiple class="visually-hidden" data-home-model-files="${index}" /></label>`,
        "</div>",
        "</div>",
      ].join("");
    });
    const unassigned = draft.unassigned || [];
    return [
      `<div class="home-draft-row"><span class="home-draft-label">模型（${draft.models.length}）</span><div class="home-draft-models">`,
      rows.join(""),
      unassigned.length
        ? `<small class="home-draft-warning" role="note">没认出属于哪个模型：${escapeHtml(unassigned.map((item) => item.file?.name || item.relativePath).join("、"))}。请在对应模型里添加。</small>`
        : "",
      '<div class="home-draft-file-actions">',
      `<button type="button" class="home-draft-link" data-home-model-add${draft.models.length >= MAX_VALIDATION_BATCH_ROWS ? " disabled" : ""}>添加模型</button>`,
      '<button type="button" class="home-draft-link" data-home-draft-single>只验证一个模型</button>',
      "</div>",
      "</div></div>",
    ].join("");
  }

  function renderDraft() {
    const box = $("homeDraft");
    if (!box || !draft) return;
    const active = doc?.activeElement;
    const focusKey = active && box.contains?.(active) ? focusKeyOf(active) : null;
    const gaps = missing();
    const definition = definitionOf(draft.taskType);
    const materialHint = MATERIAL_HINTS[draft.taskType] || "相关材料";
    const nameLabel = definition?.nameLabel || "任务名称";
    const validatorLabel = definition?.validatorLabel || "负责人";
    box.innerHTML = [
      '<div class="home-draft-thread">',
      draft.text || draft.files.length
        ? [
          '<div class="home-draft-user">',
          draft.files.length ? `<div class="home-file-chips">${filesChipsHtml(draft.files, { removable: false })}</div>` : "",
          draft.text ? `<p>${escapeHtml(draft.text)}</p>` : "",
          "</div>",
        ].join("")
        : "",
      '<div class="home-draft-agent">',
      agentFaceHtml({ face: gaps.length ? "wait" : "idle", label: "Agent" }),
      '<div class="home-draft-body">',
      `<p class="home-draft-lead" aria-live="polite">${leadHtml(gaps)}</p>`,
      '<form class="home-draft-card" data-home-draft-form novalidate>',
      '<div class="home-draft-row"><span class="home-draft-label">工作类型</span>',
      `<div class="home-draft-chips" role="group" aria-label="工作类型">${typeChipsHtml()}</div></div>`,
      draft.taskType
        ? [
          isBatch()
            ? ""
            : [
              `<label class="home-draft-row"><span class="home-draft-label">${escapeHtml(nameLabel)}</span>`,
              `<input type="text" data-home-draft-field="name" value="${escapeHtml(draft.name)}" maxlength="80" /></label>`,
            ].join(""),
          `<label class="home-draft-row"><span class="home-draft-label">${escapeHtml(validatorLabel)}</span>`,
          `<input type="text" data-home-draft-field="validator" value="${escapeHtml(draft.validator)}" placeholder="填写姓名" maxlength="40" /></label>`,
          draft.taskType === "vintage" ? riskAnalysisHtml() : "",
          isBatch() ? modelsRowHtml() : materialsRowHtml(materialHint),
          draft.taskType === "strategy" && !draft.collections ? strategyLimitsHtml() : "",
          scopeNotesHtml(),
          '<div class="home-draft-row"><span class="home-draft-label">执行方式</span>',
          `<div class="home-draft-chips" role="group" aria-label="执行方式">${runModeHtml()}</div></div>`,
          draft.taskType === "modeling" && draft.runMode === "manual" ? algorithmsHtml() : "",
        ].join("")
        : "",
      draft.error ? `<p class="home-draft-error" role="alert">${escapeHtml(draft.error)}</p>` : "",
      '<div class="home-draft-actions">',
      `<button type="submit" class="button primary" data-home-draft-create${creating ? " disabled" : ""}>${creating ? "正在创建…" : "创建并开始"}</button>`,
      `<button type="button" class="button" data-home-draft-advanced${draft.taskType && !creating ? "" : " disabled"}>高级设置</button>`,
      `<button type="button" class="home-draft-back" data-home-draft-back${creating ? " disabled" : ""}>返回修改</button>`,
      "</div>",
      "</form>",
      "</div>",
      "</div>",
      "</div>",
    ].join("");
    if (focusKey) {
      const target = box.querySelector?.(focusSelector(focusKey))
        || (focusKey.key === "homeDraftTypes" ? box.querySelector?.('[data-home-draft-type][aria-pressed="true"]') : null);
      target?.focus?.();
    }
  }

  // Typing in the card keeps the Agent's "还需要" line current without a
  // re-render (which would drop the caret).
  function syncLead() {
    const box = $("homeDraft");
    const lead = box?.querySelector(".home-draft-lead");
    if (!lead || !draft) return;
    const gaps = missing();
    lead.innerHTML = leadHtml(gaps);
    box.querySelector(".home-draft-agent > .agent-face")?.setAttribute("data-face", gaps.length ? "wait" : "idle");
  }

  function backToHome() {
    draft = null;
    creating = false;
    setMode("home");
    $(templateActive ? "homeComposerTemplate" : "homeComposerInput")?.focus();
  }

  function reset() {
    renderExamples(); // a fresh pick each time the home page comes back
    files = [];
    pickedType = "";
    starterText = "";
    defaultInputSource = null;
    draft = null;
    creating = false;
    exitTemplate();
    const input = $("homeComposerInput");
    if (input) input.value = "";
    renderFiles();
    setMode("home");
  }

  async function createFromDraft() {
    if (!draft || creating) return;
    const gaps = missing();
    if (gaps.length) {
      draft.error = `还需要：${gaps.map((field) => MISSING_LABELS[field]).join("、")}。`;
      renderDraft();
      return;
    }
    const current = draft;
    const seq = ++createSeq;
    creating = true;
    current.error = "";
    renderDraft();
    let created = false;
    let failure = null;
    try {
      created = await onCreate({ ...current, name: current.name.trim(), validator: current.validator.trim() });
    } catch (error) {
      failure = error;
    } finally {
      if (seq === createSeq) creating = false;
    }
    // The page has moved on (createTask resets it once the task exists, or the
    // user started over): never write into a newer draft.
    if (draft !== current) {
      if (failure) onError(failure);
      return;
    }
    if (created) {
      reset();
      return;
    }
    if (failure) current.error = failure.message || "创建失败，请重试。";
    renderDraft();
  }

  // ---------- events ----------

  function handleDraftClick(event) {
    const target = event.target;
    if (target.closest?.("[data-home-draft-types]")) {
      draft.showTypes = true;
      renderDraft();
      return;
    }
    const typeButton = target.closest?.("[data-home-draft-type]");
    if (typeButton) {
      const taskType = typeButton.dataset.homeDraftType;
      draft.taskType = taskType;
      draft.typeSource = "picked";
      draft.confidence = "high";
      if (!draft.nameEdited) draft.name = draftTaskName(draft.text, taskType);
      const definition = definitionOf(taskType);
      if (definition?.manualEnabled === false || (draft.runMode === "agent" && !hasEnabledAgent())) {
        draft.runMode = defaultRunMode(taskType);
      }
      draft.error = "";
      syncValidationModels();
      renderDraft();
      return;
    }
    const analysis = target.closest?.("[data-home-draft-analysis]");
    if (analysis) {
      draft.analysis = analysis.dataset.homeDraftAnalysis;
      draft.error = "";
      renderDraft();
      return;
    }
    const algorithm = target.closest?.("[data-home-draft-algorithm]");
    if (algorithm) {
      const id = algorithm.dataset.homeDraftAlgorithm;
      const chosen = new Set(draft.algorithms || []);
      if (chosen.has(id)) chosen.delete(id);
      else chosen.add(id);
      draft.algorithms = HOME_MODEL_ALGORITHMS.map((item) => item.id).filter((item) => chosen.has(item));
      draft.error = "";
      renderDraft();
      return;
    }
    const modeButton = target.closest?.("[data-home-draft-mode]");
    if (modeButton && !modeButton.disabled) {
      draft.runMode = modeButton.dataset.homeDraftMode;
      renderDraft();
      return;
    }
    const remove = target.closest?.("[data-home-file-remove]");
    if (remove) {
      files.splice(Number(remove.dataset.homeFileRemove), 1);
      draft.files = files.slice();
      renderFiles();
      renderDraft();
      return;
    }
    if (target.closest?.("[data-home-draft-single]")) {
      collapseToSingleModel(draft.models?.[0] || null);
      draft.multiModel = false;
      draft.error = "";
      renderDraft();
      return;
    }
    const modelAdd = target.closest?.("[data-home-model-add]");
    if (modelAdd && !modelAdd.disabled) {
      addModel();
      renderDraft();
      draft.models && $("homeDraft")?.querySelector?.(`[data-home-model-name="${draft.models.length - 1}"]`)?.focus?.();
      return;
    }
    const modelRemove = target.closest?.("[data-home-model-remove]");
    if (modelRemove && draft.models) {
      draft.models.splice(Number(modelRemove.dataset.homeModelRemove), 1);
      if (draft.models.length < 2) collapseToSingleModel(draft.models[0] || null);
      else {
        files = modelPoolItems().slice();
        draft.files = files.slice();
        renderFiles();
      }
      draft.error = "";
      renderDraft();
      return;
    }
    const modelFileRemove = target.closest?.("[data-home-model-file-remove]");
    if (modelFileRemove && draft.models) {
      const [modelIndex, fileIndex] = String(modelFileRemove.dataset.homeModelFileRemove).split(":").map(Number);
      draft.models[modelIndex]?.items.splice(fileIndex, 1);
      files = modelPoolItems().slice();
      draft.files = files.slice();
      renderFiles();
      renderDraft();
      return;
    }
    if (creating) return;
    if (target.closest?.("[data-home-draft-advanced]")) {
      onAdvanced({ ...draft });
      return;
    }
    if (target.closest?.("[data-home-draft-back]")) backToHome();
  }

  function bind() {
    installComposerSelectSizing(doc);
    renderExamples();
    renderFiles();
    renderContinue();
    setMode("home");
    onUiLocaleChange(() => {
      renderExamples({ shuffle: false });
      const input = $("homeComposerInput");
      // Locale changes may update platform defaults, never a sentence that
      // the person has edited or a creation card already under review.
      if (draft || !defaultInputSource || input?.value !== starterText) return;
      const { kind, taskType, index } = defaultInputSource;
      if (kind === "template") {
        enterTemplate(taskType, homeTemplateTokens(taskType, getUiLocale()), { focus: false });
      } else if (kind === "example") {
        input.value = localizeText(HOME_EXAMPLES[index].text);
        starterText = input.value;
      } else if (kind === "starter") {
        input.value = homeStarterText(taskType, localizeText(labelOf(taskType)), getUiLocale());
        starterText = input.value;
      }
      rememberTypedLanguage(input.value);
      syncSend();
    }, doc);
    const form = $("homeComposer");
    const input = $("homeComposerInput");
    form?.addEventListener("submit", (event) => {
      event.preventDefault();
      startDraft();
    });
    input?.addEventListener("input", syncSend);
    input?.addEventListener("keydown", (event) => {
      // keyCode 229: Safari reports the IME's confirming Enter this way.
      if (event.key !== "Enter" || event.shiftKey || event.isComposing || event.keyCode === 229) return;
      event.preventDefault();
      startDraft();
    });
    $("homeComposerFileInput")?.addEventListener("change", (event) => {
      addFiles(event.target.files);
      event.target.value = "";
    });
    $("homeComposerFiles")?.addEventListener("click", (event) => {
      const remove = event.target.closest?.("[data-home-file-remove]");
      if (!remove) return;
      files.splice(Number(remove.dataset.homeFileRemove), 1);
      renderFiles();
      refreshFileSlots();
    });
    const template = $("homeComposerTemplate");
    template?.addEventListener("input", syncFromTemplate);
    template?.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" || event.isComposing || event.keyCode === 229) return;
      event.preventDefault();
      if (!event.shiftKey) startDraft();
    });
    // Pasted text stays plain so the sentence keeps its slots and no markup.
    template?.addEventListener("paste", (event) => {
      const text = event.clipboardData?.getData?.("text/plain");
      if (text === undefined) return;
      event.preventDefault();
      doc?.execCommand?.("insertText", false, text);
    });
    template?.addEventListener("click", (event) => {
      const slot = event.target.closest?.(".home-file-slot");
      const picker = $("homeComposerRoleFileInput");
      if (!slot || !picker) return;
      event.preventDefault();
      roleTarget = slot.dataset.fileRole || "";
      picker.multiple = slot.hasAttribute("data-multiple");
      picker.click();
    });
    $("homeComposerRoleFileInput")?.addEventListener("change", (event) => {
      addFiles(event.target.files, roleTarget || "");
      roleTarget = null;
      event.target.value = "";
    });
    // The whole home page takes dropped files and folders ("拖进来"), so a
    // near miss never makes the browser navigate to the file.
    const welcome = $("workspaceWelcome");
    const dropTarget = welcome || form;
    const dropping = (event) => Array.from(event.dataTransfer?.types || []).includes("Files");
    dropTarget?.addEventListener("dragover", (event) => {
      if (!dropping(event)) return;
      event.preventDefault();
      if (form) form.dataset.dragging = "true";
    });
    dropTarget?.addEventListener("dragleave", (event) => {
      if (event.relatedTarget && dropTarget.contains?.(event.relatedTarget)) return;
      if (form) delete form.dataset.dragging;
    });
    dropTarget?.addEventListener("drop", async (event) => {
      if (!dropping(event)) return;
      event.preventDefault();
      if (form) delete form.dataset.dragging;
      addFiles(await droppedFileItems(event.dataTransfer));
    });
    $("homeExamples")?.addEventListener("click", (event) => {
      const button = event.target.closest?.("[data-home-example]");
      if (!button) return;
      const example = HOME_EXAMPLES[Number(button.dataset.homeExample)];
      if (!example) return;
      // D-10: the example fills the input in the language the user reads.
      const text = localizeText(example.text);
      rememberTypedLanguage(text);
      fillInput(text, example.taskType, { kind: "example", taskType: example.taskType, index: Number(button.dataset.homeExample) });
    });
    $("homeExamples")?.addEventListener("wheel", (event) => {
      const box = $("homeExamples");
      if (!box || Math.abs(event.deltaX) >= Math.abs(event.deltaY) || box.scrollWidth <= box.clientWidth) return;
      const previous = box.scrollLeft;
      box.scrollLeft += event.deltaY;
      if (box.scrollLeft !== previous) event.preventDefault();
    }, { passive: false });
    $("homeExamples")?.addEventListener("scroll", syncExampleScrollEdges, { passive: true });
    const ExampleResizeObserver = doc?.defaultView?.ResizeObserver || globalThis.ResizeObserver;
    if (ExampleResizeObserver && $("homeExamples")) {
      new ExampleResizeObserver(syncExampleScrollEdges).observe($("homeExamples"));
    }
    doc?.fonts?.ready?.then?.(syncExampleScrollEdges);
    $("homeContinueList")?.addEventListener("click", (event) => {
      const row = event.target.closest?.("[data-home-open-task]");
      if (row) openTask(row.dataset.homeOpenTask);
    });
    const draftBox = $("homeDraft");
    draftBox?.addEventListener("click", (event) => {
      if (draft) handleDraftClick(event);
    });
    draftBox?.addEventListener("input", (event) => {
      if (!draft) return;
      const limit = event.target.closest?.("[data-home-draft-limit]");
      if (limit) {
        draft.strategy = { ...(draft.strategy || {}), [limit.dataset.homeDraftLimit]: limit.value };
        syncLead();
        return;
      }
      const modelName = event.target.closest?.("[data-home-model-name]");
      if (modelName && draft.models) {
        const model = draft.models[Number(modelName.dataset.homeModelName)];
        if (model) model.name = modelName.value;
        syncLead();
        return;
      }
      const field = event.target.closest?.("[data-home-draft-field]");
      if (!field) return;
      draft[field.dataset.homeDraftField] = field.value;
      if (field.dataset.homeDraftField === "name") draft.nameEdited = true;
      syncLead();
    });
    draftBox?.addEventListener("change", (event) => {
      if (!draft) return;
      if (event.target.matches?.("[data-home-model-files]")) {
        addModelFiles(Number(event.target.dataset.homeModelFiles), event.target.files);
        return;
      }
      if (!event.target.matches?.("[data-home-draft-files]")) return;
      addFiles(event.target.files);
    });
    draftBox?.addEventListener("submit", (event) => {
      event.preventDefault();
      void createFromDraft();
    });
    // Enter in a text field must not create the task and start the Agent run.
    draftBox?.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && event.target.matches?.('input[type="text"], input[type="number"]')) {
        event.preventDefault();
      }
    });
  }

  // A work icon (the seven task cards) fills the input; it never opens a form.
  function pickTaskType(event) {
    const card = event.target.closest?.("[data-task-kind]");
    if (!card) return;
    const taskType = card.dataset.taskKind;
    const input = $("homeComposerInput");
    const current = String(input?.value || "").trim();
    const untouched = !current || current === starterText.trim();
    const tokens = homeTemplateTokens(taskType, getUiLocale());
    if (untouched && tokens && enterTemplate(taskType, tokens)) return;
    const text = untouched ? homeStarterText(taskType, localizeText(labelOf(taskType)), getUiLocale()) : current;
    fillInput(text, taskType, untouched ? { kind: "starter", taskType } : null);
  }

  return {
    bind,
    pickTaskType,
    renderContinue,
    reset,
    startDraft,
    getDraft: () => draft,
  };
}
