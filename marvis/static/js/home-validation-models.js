// Home page batch validation (2026-11-10): group the materials dropped on the
// home page into models, so several models start as one validation batch from
// the 模型验证 card instead of the create dialog's 高级设置.
//
// Grouping is by evidence only: one sub-folder per model, or else matching
// file-name stems (a_notebook.ipynb + a_model.pmml + a_sample.csv). A single
// leftover sample or dictionary is shared by every model that lacks one;
// anything else stays unassigned for the user to place.

import {
  MAX_VALIDATION_BATCH_ROWS,
  classifyValidationBatchFiles,
  validationFileRole,
} from "./validation-batch-create.js";

export const VALIDATION_MODEL_ROLES = Object.freeze(["notebook", "sample", "pmml", "dictionary"]);
export const VALIDATION_ROLE_LABELS = Object.freeze({
  notebook: "Notebook",
  sample: "样本",
  pmml: "PMML",
  dictionary: "数据字典",
});

// Words that name a file's role, not its model (longer words first).
const ROLE_WORDS = /notebook|ipynb|pmml|models?|模型|样本|samples?|数据字典|字典|dictionary|dict|metadata|meta|features?|train|oot|data|final|最终/g;
const ROLE_TOKEN = /^(?:notebook|ipynb|pmml|models?|模型|样本|samples?|数据字典|字典|dictionary|dict|metadata|meta|features?|train|oot|data|final|最终)$/i;

function itemFile(item) {
  return item?.file || null;
}

function itemPath(item) {
  return String(item?.relativePath || item?.file?.webkitRelativePath || item?.file?.name || "")
    .replace(/\\/g, "/");
}

function stemOf(path) {
  const base = String(path || "").split("/").pop() || "";
  const dot = base.lastIndexOf(".");
  return dot > 0 ? base.slice(0, dot) : base;
}

function squash(text) {
  return String(text || "").toLowerCase().replace(/[\s_\-.()（）[\]【】]+/g, "");
}

// The part of a file name that names its model: "A卡_notebook" and
// "A卡-model" both give "a卡".
export function modelKey(path) {
  const squashed = squash(stemOf(path));
  return squashed.replace(ROLE_WORDS, "") || squashed;
}

// A readable model name from a notebook/PMML name: role words dropped.
function displayName(path) {
  const stem = stemOf(path);
  const kept = stem.split(/[\s_\-.]+/).filter((token) => token && !ROLE_TOKEN.test(token));
  return kept.join("_") || stem;
}

function dirSegments(path) {
  const parts = String(path || "").split("/").filter(Boolean);
  parts.pop();
  return parts;
}

// One sub-folder per model: every notebook/PMML sits one level below the
// folder they share, in at least two different sub-folders.
function groupByFolder(entries, anchors) {
  const dirs = entries.map((entry) => dirSegments(entry.path));
  if (dirs.every((dir) => !dir.length)) return null;
  let prefix = dirs[0].slice();
  for (const dir of dirs) {
    let index = 0;
    while (index < prefix.length && index < dir.length && prefix[index] === dir[index]) index += 1;
    prefix = prefix.slice(0, index);
  }
  const folderOf = (entry) => {
    const dir = dirs[entry.index];
    return dir.length > prefix.length ? dir[prefix.length] : "";
  };
  const folders = new Set();
  for (const anchor of anchors) {
    const folder = folderOf(anchor);
    if (!folder) return null;
    folders.add(folder);
  }
  if (folders.size < 2) return null;
  const groups = new Map();
  const shared = [];
  for (const entry of entries) {
    const folder = folderOf(entry);
    if (!folders.has(folder)) {
      shared.push(entry);
      continue;
    }
    if (!groups.has(folder)) groups.set(folder, { name: folder, entries: [] });
    groups.get(folder).entries.push(entry);
  }
  return { groups: [...groups.values()], shared };
}

// Matching file-name stems: each notebook/PMML key is a model; a table joins
// the model whose key its name contains (the longest one, when unambiguous).
function groupByStem(entries, anchors) {
  const groups = new Map();
  for (const anchor of anchors) {
    const key = modelKey(anchor.path);
    if (!groups.has(key)) groups.set(key, { key, name: displayName(anchor.path), entries: [] });
    groups.get(key).entries.push(anchor);
  }
  if (groups.size < 2) return null;
  const shared = [];
  for (const entry of entries) {
    if (entry.role === "notebook" || entry.role === "pmml") continue;
    const full = squash(stemOf(entry.path));
    const matches = [...groups.values()].filter((group) => group.key && full.includes(group.key));
    const longest = Math.max(0, ...matches.map((group) => group.key.length));
    const best = matches.filter((group) => group.key.length === longest);
    if (best.length === 1) best[0].entries.push(entry);
    else shared.push(entry);
  }
  return { groups: [...groups.values()], shared };
}

// {models: [{name, items}], unassigned: [items]} when the files hold two or
// more models; null for one model (the ordinary single-model card).
export function groupValidationModels(items = []) {
  const entries = Array.from(items || [])
    .filter((item) => itemFile(item))
    .map((item, index) => ({ item, index, path: itemPath(item), role: validationFileRole(itemFile(item)) }));
  const anchors = entries.filter((entry) => entry.role === "notebook" || entry.role === "pmml");
  const notebooks = anchors.filter((entry) => entry.role === "notebook").length;
  if (Math.max(notebooks, anchors.length - notebooks) < 2) return null;
  const grouped = groupByFolder(entries, anchors) || groupByStem(entries, anchors);
  if (!grouped) return null;
  const { groups } = grouped;
  let shared = grouped.shared;
  // A single leftover sample/dictionary is the one every model shares.
  for (const role of ["sample", "dictionary"]) {
    const candidates = shared.filter((entry) => entry.role === role);
    if (candidates.length !== 1) continue;
    const [entry] = candidates;
    let used = false;
    for (const group of groups) {
      if (group.entries.some((member) => member.role === role)) continue;
      group.entries.push(entry);
      used = true;
    }
    if (used) shared = shared.filter((member) => member !== entry);
  }
  const firstIndex = (group) => Math.min(...group.entries.map((entry) => entry.index));
  return {
    models: groups
      .sort((left, right) => firstIndex(left) - firstIndex(right))
      .map((group) => ({
        name: group.name,
        items: group.entries.sort((left, right) => left.index - right.index).map((entry) => entry.item),
      })),
    unassigned: shared.sort((left, right) => left.index - right.index).map((entry) => entry.item),
  };
}

// The roles a model still lacks (empty when its four materials are complete).
export function modelMaterialGaps(items = []) {
  const files = Array.from(items || []).map(itemFile).filter(Boolean);
  if (classifyValidationBatchFiles(files)) return [];
  const roles = new Set(files.map(validationFileRole));
  const gaps = VALIDATION_MODEL_ROLES.filter((role) => !roles.has(role));
  if (gaps.length) return gaps;
  // Two tables that cannot be told apart, or a file used twice.
  return ["dictionary"];
}

// Every model named and complete, between two and the batch limit.
export function validationModelsReady(models = []) {
  const list = Array.isArray(models) ? models : [];
  return list.length >= 2
    && list.length <= MAX_VALIDATION_BATCH_ROWS
    && list.every((model) => String(model?.name || "").trim() && !modelMaterialGaps(model?.items).length);
}
