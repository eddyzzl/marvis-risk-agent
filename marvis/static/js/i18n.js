// D-10 (batch 1b): platform UI language. The user picks 中文 | English in the
// sidebar; every platform-authored string follows it. Agent replies do not:
// they follow the language the user last typed (sent as a request hint).
//
// The dictionary (i18n-en.js) is keyed by the existing Chinese source string,
// so Chinese stays the source of truth and the fallback. Platform text is
// translated where it lands — text nodes and text attributes — by exact match
// or by a template with placeholders; user data, Agent prose and code are
// skipped. See docs/superpowers/specs/2026-10-28-ui-language-switch.md.
import { EN_UI } from "./i18n-en.js";

export const UI_LOCALE_STORAGE_KEY = "marvis_ui_locale";
export const UI_LOCALE_CHANGE_EVENT = "marvis:ui-locale-change";
export const UI_LOCALES = Object.freeze(["zh", "en"]);

const CJK_RE = /[㐀-䶿一-鿿豈-﫿]/;
const LATIN_RE = /[A-Za-z]/;
const PLACEHOLDER_RE = /\{([A-Za-z0-9_]+)(\|t)?\}/g;
const TEXT_ATTRIBUTES = Object.freeze([
  "placeholder",
  "title",
  "aria-label",
  "aria-description",
  "alt",
  "data-tooltip",
  "data-tip",
  "data-panel-title",
  "data-panel-description",
]);
const VALUE_INPUT_TYPES = new Set(["button", "submit", "reset"]);
// Subtrees whose text is never translated: user data, Agent/LLM prose (its
// language is the backend's concern), code, and the language switch itself.
export const I18N_SKIP_SELECTOR = [
  '[translate="no"]',
  "[data-i18n-skip]",
  "script",
  "style",
  "pre",
  "code",
  "kbd",
  "iframe",
  ".agent-message.user",
  ".agent-markdown",
  ".agent-message-content > :is(p, ul, ol, h1, h2, h3, h4, h5, h6, blockquote, table, hr)",
].join(", ");
// Elements whose text children are user input; their attributes still translate.
const TEXT_SKIP_SELECTOR = 'textarea, [contenteditable=""], [contenteditable="true"], [contenteditable="plaintext-only"]';
const COMPOSER_SELECTOR = "#homeComposerInput, #agentComposerInput, #homeComposerTemplate";
const MAX_MISSING = 4000;

function normalizeKey(text) {
  return String(text ?? "").replace(/\s+/g, " ").trim();
}

function escapeRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function compileDictionary(source) {
  const exact = new Map();
  const templates = [];
  for (const [rawKey, value] of Object.entries(source || {})) {
    if (typeof value !== "string" || !value) continue;
    const key = normalizeKey(rawKey);
    if (!key) continue;
    exact.set(key, value);
    PLACEHOLDER_RE.lastIndex = 0;
    if (!PLACEHOLDER_RE.test(key)) continue;
    const names = [];
    let pattern = "^";
    let literal = "";
    let anchor = "";
    let cursor = 0;
    PLACEHOLDER_RE.lastIndex = 0;
    for (const match of key.matchAll(PLACEHOLDER_RE)) {
      const chunk = key.slice(cursor, match.index);
      pattern += escapeRegExp(chunk);
      literal += chunk;
      if (chunk.trim().length > anchor.length) anchor = chunk.trim();
      pattern += "([\\s\\S]+?)";
      names.push({ name: match[1], translate: Boolean(match[2]) });
      cursor = match.index + match[0].length;
    }
    const tail = key.slice(cursor);
    pattern += `${escapeRegExp(tail)}$`;
    literal += tail;
    if (tail.trim().length > anchor.length) anchor = tail.trim();
    if (!CJK_RE.test(literal)) continue;
    templates.push({ regex: new RegExp(pattern), names, value, anchor, weight: literal.length });
  }
  templates.sort((left, right) => right.weight - left.weight);
  return { exact, templates };
}

let dictionary = compileDictionary(EN_UI);
let currentLocale = "zh";
let replyLanguageHint = "";
let replyLanguageSeedContext = null;
const missing = new Set();

function recordMissing(text) {
  const key = normalizeKey(text);
  if (!key || missing.has(key) || missing.size >= MAX_MISSING) return;
  missing.add(key);
}

export function normalizeUiLocale(value) {
  return /^en\b/i.test(String(value ?? "").trim()) ? "en" : "zh";
}

function defaultStorage() {
  try {
    return globalThis.localStorage || null;
  } catch (_error) {
    return null;
  }
}

export function readStoredUiLocale(storage = defaultStorage()) {
  try {
    const value = storage?.getItem?.(UI_LOCALE_STORAGE_KEY);
    return value ? normalizeUiLocale(value) : "zh";
  } catch (_error) {
    return "zh";
  }
}

export function writeStoredUiLocale(locale, storage = defaultStorage()) {
  try {
    storage?.setItem?.(UI_LOCALE_STORAGE_KEY, normalizeUiLocale(locale));
    return true;
  } catch (_error) {
    return false;
  }
}

export function getUiLocale() {
  return currentLocale;
}

// Same rule as marvis.ui_locale.detect_text_language: any CJK ideograph means
// Chinese; Latin letters without CJK mean English; otherwise unknown.
export function detectTextLanguage(text) {
  const value = String(text ?? "");
  if (CJK_RE.test(value)) return "zh";
  if (LATIN_RE.test(value)) return "en";
  return "";
}

export function rememberTypedLanguage(text) {
  const language = detectTextLanguage(text);
  if (language) replyLanguageHint = language;
  return replyLanguageHint;
}

// The Agent answers in the language the user last typed. "" means unknown:
// no hint is sent and the backend reads the language from the request body.
export function replyLanguage() {
  return replyLanguageHint;
}

// Quick-reply commands are platform phrases (Chinese), not something the user
// wrote, so a send that is exactly one of them carries no language signal.
const quickReplyCommands = new Set();

export function setQuickReplyCommands(commands = []) {
  quickReplyCommands.clear();
  for (const command of commands) {
    const text = String(command ?? "").trim();
    if (text) quickReplyCommands.add(text);
  }
}

export function isQuickReplyCommand(text) {
  return quickReplyCommands.has(String(text ?? "").trim());
}

// Spec §2.4: on a task switch (and after every message load) the hint is the
// language of that task's latest user message that is not a quick reply.
export function seedReplyLanguage(messages = [], { taskId } = {}) {
  let latestTypedMessage = null;
  let language = "";
  if (Array.isArray(messages)) {
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      const message = messages[index];
      if (message?.role !== "user" || isQuickReplyCommand(message?.content)) continue;
      language = detectTextLanguage(message?.content);
      if (language) {
        latestTypedMessage = message;
        break;
      }
    }
  }
  // Polls and composer input both redraw quick replies. Re-seed only when
  // the owner or latest typed message changes, so a redraw cannot erase the
  // language just captured from an unsent draft. A task switch still resets it.
  const context = typeof taskId === "string"
    ? JSON.stringify([taskId, latestTypedMessage?.id || "", latestTypedMessage?.content || ""])
    : null;
  if (context !== null && context === replyLanguageSeedContext) return replyLanguageHint;
  replyLanguageSeedContext = context;
  replyLanguageHint = language;
  return replyLanguageHint;
}

export function uiRequestHeaders(body = undefined) {
  const headers = { "X-MARVIS-UI-Locale": currentLocale };
  if (replyLanguageHint) headers["X-MARVIS-Reply-Language"] = replyLanguageHint;
  let content = "";
  if (typeof body === "string") {
    try {
      const parsed = JSON.parse(body);
      content = parsed && typeof parsed === "object" ? parsed.content : "";
    } catch (_error) {
      content = "";
    }
  }
  // The backend then keeps the hint instead of reading the command's language.
  if (content && isQuickReplyCommand(content)) headers["X-MARVIS-Reply-Source"] = "quick-reply";
  return headers;
}

function lookup(core, active = dictionary) {
  const key = normalizeKey(core);
  if (!key) return null;
  const hit = active.exact.get(key);
  if (hit !== undefined && !/\{[A-Za-z0-9_]+(\|t)?\}/.test(key)) return hit;
  for (const template of active.templates) {
    if (template.anchor && !key.includes(template.anchor)) continue;
    const match = template.regex.exec(key);
    if (!match) continue;
    const captures = {};
    template.names.forEach((slot, index) => {
      const captured = match[index + 1];
      captures[slot.name] = slot.translate ? (lookup(captured, active) ?? captured) : captured;
    });
    return template.value.replace(PLACEHOLDER_RE, (whole, name) => (name in captures ? captures[name] : whole));
  }
  return hit !== undefined ? hit : null;
}

// Translate one text value; null means "not a known platform string".
export function translateText(text) {
  const raw = String(text ?? "");
  if (!CJK_RE.test(raw)) return null;
  const match = /^(\s*)([\s\S]*?)(\s*)$/.exec(raw);
  const translated = lookup(match[2]);
  return translated === null ? null : `${match[1]}${translated}${match[3]}`;
}

export function interpolate(template, params = {}) {
  return String(template ?? "").replace(PLACEHOLDER_RE, (whole, name, flag) => {
    if (!params || !(name in params)) return whole;
    const value = String(params[name] ?? "");
    if (flag && currentLocale === "en") return lookup(value) ?? value;
    return value;
  });
}

// For strings that never reach the DOM (confirm dialogs, clipboard, document.title).
export function t(source, params = {}) {
  const text = String(source ?? "");
  if (currentLocale !== "en" || !CJK_RE.test(text)) return interpolate(text, params);
  const key = normalizeKey(text);
  const entry = dictionary.exact.get(key);
  if (entry !== undefined) return interpolate(entry, params);
  const translated = lookup(key);
  if (translated !== null) return interpolate(translated, params);
  recordMissing(text);
  return interpolate(text, params);
}

// The text a user sees for a Chinese source string in the current locale.
export function localizeText(text) {
  if (currentLocale !== "en") return String(text ?? "");
  return translateText(text) ?? String(text ?? "");
}

export function uiMissingStrings() {
  return [...missing];
}

// ------------------------------------------------------------- DOM translator
const originalText = new WeakMap();
const appliedText = new WeakMap();
const originalAttributes = new WeakMap();
let observer = null;
let observedDocument = null;

function elementOf(node) {
  if (!node) return null;
  return node.nodeType === 1 ? node : node.parentElement || null;
}

function isSkipped(element) {
  try {
    return Boolean(element?.closest?.(I18N_SKIP_SELECTOR));
  } catch (_error) {
    return false;
  }
}

function isTextSkipped(element) {
  try {
    return Boolean(element?.closest?.(TEXT_SKIP_SELECTOR));
  } catch (_error) {
    return false;
  }
}

function translateTextNode(node) {
  const data = node.data;
  if (appliedText.get(node) === data || !CJK_RE.test(data)) return;
  const translated = translateText(data);
  if (translated === null) {
    recordMissing(data);
    return;
  }
  originalText.set(node, data);
  appliedText.set(node, translated);
  node.data = translated;
}

function translateAttribute(element, name) {
  const value = element.getAttribute?.(name);
  if (!value) return;
  const record = originalAttributes.get(element);
  if (record?.get(name)?.applied === value || !CJK_RE.test(value)) return;
  const translated = translateText(value);
  if (translated === null) {
    recordMissing(value);
    return;
  }
  const next = record || new Map();
  next.set(name, { source: value, applied: translated });
  originalAttributes.set(element, next);
  element.setAttribute(name, translated);
}

function translateElementAttributes(element) {
  for (const name of TEXT_ATTRIBUTES) {
    if (element.hasAttribute?.(name)) translateAttribute(element, name);
  }
  if (element.tagName === "INPUT" && VALUE_INPUT_TYPES.has(String(element.type || "").toLowerCase())) {
    translateAttribute(element, "value");
  }
}

function walkTranslate(root) {
  const doc = root.ownerDocument || root;
  const filter = doc.defaultView?.NodeFilter || globalThis.NodeFilter;
  if (!doc.createTreeWalker || !filter) return;
  const walker = doc.createTreeWalker(root, filter.SHOW_ELEMENT | filter.SHOW_TEXT, {
    acceptNode(node) {
      if (node.nodeType === 3) return filter.FILTER_ACCEPT;
      try {
        if (node.matches(I18N_SKIP_SELECTOR)) return filter.FILTER_REJECT;
        if (node.matches(TEXT_SKIP_SELECTOR)) {
          translateElementAttributes(node);
          return filter.FILTER_REJECT;
        }
      } catch (_error) {
        return filter.FILTER_ACCEPT;
      }
      return filter.FILTER_ACCEPT;
    },
  });
  if (root.nodeType === 1) translateElementAttributes(root);
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    if (node.nodeType === 3) translateTextNode(node);
    else translateElementAttributes(node);
  }
}

function translateSubtree(node) {
  if (!node) return;
  if (node.nodeType === 3) {
    const parent = node.parentElement;
    if (!parent || isSkipped(parent) || isTextSkipped(parent)) return;
    translateTextNode(node);
    return;
  }
  if (node.nodeType !== 1 && node.nodeType !== 11) return;
  if (node.nodeType === 1) {
    const parent = node.parentElement;
    if (parent && isSkipped(parent)) return;
    if (isSkipped(node)) return;
    if (parent && isTextSkipped(parent)) return;
  }
  walkTranslate(node);
}

function restoreTree(root) {
  const doc = root.ownerDocument || root;
  const filter = doc.defaultView?.NodeFilter || globalThis.NodeFilter;
  if (!doc.createTreeWalker || !filter) return;
  const walker = doc.createTreeWalker(root, filter.SHOW_ELEMENT | filter.SHOW_TEXT);
  const restore = (node) => {
    if (node.nodeType === 3) {
      if (!appliedText.has(node)) return;
      if (node.data === appliedText.get(node)) node.data = originalText.get(node);
      appliedText.delete(node);
      originalText.delete(node);
      return;
    }
    const record = originalAttributes.get(node);
    if (!record) return;
    for (const [name, { source, applied }] of record) {
      if (node.getAttribute(name) === applied) node.setAttribute(name, source);
    }
    originalAttributes.delete(node);
  };
  restore(root);
  for (let node = walker.nextNode(); node; node = walker.nextNode()) restore(node);
}

function handleMutations(records) {
  if (currentLocale !== "en") return;
  for (const record of records) {
    if (record.type === "childList") {
      record.addedNodes.forEach((node) => translateSubtree(node));
    } else if (record.type === "characterData") {
      const parent = record.target.parentElement;
      if (parent && !isSkipped(parent) && !isTextSkipped(parent)) translateTextNode(record.target);
    } else if (record.type === "attributes") {
      const element = record.target;
      if (element.nodeType !== 1 || isSkipped(element)) continue;
      // value is user data except on button-like inputs (same rule as the initial pass).
      if (record.attributeName === "value" && !(
        element.tagName === "INPUT" && VALUE_INPUT_TYPES.has(String(element.type || "").toLowerCase())
      )) continue;
      translateAttribute(element, record.attributeName);
    }
  }
}

function startObserver(doc) {
  const View = doc?.defaultView?.MutationObserver || globalThis.MutationObserver;
  if (!View || observer) return;
  observer = new View(handleMutations);
  observer.observe(doc.documentElement, {
    subtree: true,
    childList: true,
    characterData: true,
    attributes: true,
    attributeFilter: [...TEXT_ATTRIBUTES, "value"],
  });
  observedDocument = doc;
}

function stopObserver() {
  if (!observer) return;
  observer.disconnect();
  observer = null;
  observedDocument = null;
}

let inertDocument = null;

// What `html` looks like once rendered in the current locale inside `context`.
// Diff-before-write helpers compare against this, so English mode does not
// re-render unchanged markup on every poll.
export function localizeHtml(html, { context = null } = {}) {
  const markup = String(html ?? "");
  if (currentLocale !== "en" || !CJK_RE.test(markup)) return markup;
  if (context && isSkipped(context)) return markup;
  const source = context?.ownerDocument || globalThis.document;
  if (!inertDocument) inertDocument = source?.implementation?.createHTMLDocument?.("") || null;
  if (!inertDocument) return markup;
  const wrapper = inertDocument.createElement(context?.tagName || "div");
  const className = context?.getAttribute?.("class");
  if (className) wrapper.setAttribute("class", className);
  wrapper.innerHTML = markup;
  if (context && isTextSkipped(context)) return wrapper.innerHTML;
  walkTranslate(wrapper);
  return wrapper.innerHTML;
}

// The Chinese source of a node the translator rewrote (for code that compares
// rendered text with a Chinese constant).
export function sourceText(element) {
  if (!element) return "";
  const nodes = element.nodeType === 3 ? [element] : [...(element.childNodes || [])];
  if (nodes.length && nodes.every((node) => node.nodeType === 3)) {
    return nodes.map((node) => (appliedText.get(node) === node.data ? originalText.get(node) : node.data)).join("");
  }
  return element.textContent || "";
}

function applyDocumentLanguage(doc) {
  const root = doc?.documentElement;
  if (!root) return;
  root.setAttribute("lang", currentLocale === "en" ? "en" : "zh-CN");
  root.dataset.uiLocale = currentLocale;
}

function syncToggle(doc) {
  doc?.querySelectorAll?.("[data-ui-locale-option]").forEach((option) => {
    const selected = option.dataset.uiLocaleOption === currentLocale;
    option.setAttribute("aria-checked", selected ? "true" : "false");
    option.classList.toggle("selected", selected);
    option.tabIndex = selected ? 0 : -1;
  });
  doc?.querySelectorAll?.("[data-ui-locale-toggle]").forEach((button) => {
    // Named in the language it switches to, so it reads in either locale.
    const label = currentLocale === "en" ? "切换到中文" : "Switch to English";
    button.setAttribute("aria-label", label);
    button.setAttribute("title", label);
  });
}

export function setUiLocale(next, { document: doc = globalThis.document, storage = defaultStorage() } = {}) {
  const locale = normalizeUiLocale(next);
  writeStoredUiLocale(locale, storage);
  if (locale === currentLocale) {
    syncToggle(doc);
    return locale;
  }
  currentLocale = locale;
  if (doc?.documentElement) {
    applyDocumentLanguage(doc);
    if (locale === "en") {
      walkTranslate(doc.documentElement);
      startObserver(doc);
    } else {
      stopObserver();
      restoreTree(doc.documentElement);
    }
    syncToggle(doc);
    const EventView = doc.defaultView?.CustomEvent || globalThis.CustomEvent;
    if (EventView) doc.dispatchEvent(new EventView(UI_LOCALE_CHANGE_EVENT, { detail: { locale } }));
  }
  return locale;
}

export function onUiLocaleChange(callback, doc = globalThis.document) {
  if (typeof callback !== "function" || !doc?.addEventListener) return () => {};
  const listener = (event) => callback(event?.detail?.locale || currentLocale);
  doc.addEventListener(UI_LOCALE_CHANGE_EVENT, listener);
  return () => doc.removeEventListener(UI_LOCALE_CHANGE_EVENT, listener);
}

function bindToggle(doc) {
  doc.addEventListener("click", (event) => {
    const target = event.target instanceof doc.defaultView.Element ? event.target : null;
    const option = target?.closest?.("[data-ui-locale-option]");
    if (option) {
      event.preventDefault();
      setUiLocale(option.dataset.uiLocaleOption, { document: doc });
      return;
    }
    const toggle = target?.closest?.("[data-ui-locale-toggle]");
    if (toggle) {
      event.preventDefault();
      setUiLocale(currentLocale === "en" ? "zh" : "en", { document: doc });
    }
  });
  doc.addEventListener("keydown", (event) => {
    const target = event.target instanceof doc.defaultView.Element ? event.target : null;
    const option = target?.closest?.("[data-ui-locale-option]");
    if (!option || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) return;
    event.preventDefault();
    const next = currentLocale === "en" ? "zh" : "en";
    setUiLocale(next, { document: doc });
    doc.querySelector(`[data-ui-locale-option="${next}"]`)?.focus?.();
  });
  doc.addEventListener("input", (event) => {
    const target = event.target;
    if (!target?.matches?.(COMPOSER_SELECTOR)) return;
    rememberTypedLanguage("value" in target && target.tagName !== "DIV" ? target.value : target.textContent);
  }, true);
}

let installed = false;

export function installUiLanguage({ document: doc = globalThis.document, storage = defaultStorage() } = {}) {
  currentLocale = readStoredUiLocale(storage);
  if (!doc?.documentElement || installed) return currentLocale;
  installed = true;
  applyDocumentLanguage(doc);
  bindToggle(doc);
  syncToggle(doc);
  if (currentLocale === "en") {
    walkTranslate(doc.documentElement);
    startObserver(doc);
  }
  try {
    globalThis.__marvisI18n = Object.freeze({
      locale: () => currentLocale,
      missing: () => uiMissingStrings(),
      observing: () => Boolean(observer && observedDocument === doc),
    });
  } catch (_error) {
    // Diagnostics only.
  }
  return currentLocale;
}

// Test seam: swap the dictionary and locale without a document.
export function __setI18nStateForTests({ locale, entries } = {}) {
  if (entries) dictionary = compileDictionary(entries);
  if (locale) currentLocale = normalizeUiLocale(locale);
  missing.clear();
  replyLanguageHint = "";
  replyLanguageSeedContext = null;
  quickReplyCommands.clear();
}
