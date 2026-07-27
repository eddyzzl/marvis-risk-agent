const STATE_SCHEMA_VERSION = "strategy.candidate-lab-view-state.v1";
const STORAGE_PREFIX = "marvis_strategy_candidate_lab_state_v1";
const MAX_FIELDS = 240;
const MAX_VALUES_PER_FIELD = 200;
const MAX_VALUE_LENGTH = 1_000;
const MAX_SERIALIZED_BYTES = 64 * 1024;
const WORKFLOW_RE = /^[a-z][a-z0-9_]{0,95}$/;
const FIELD_RE = /^[a-z][a-z0-9_]{0,127}$/;

function nonEmptyText(value) {
  return typeof value === "string" ? value.trim() : "";
}

function fieldIdentity(form, field) {
  const workflow = nonEmptyText(form?.dataset?.candidateLabWorkflow);
  const fieldName = nonEmptyText(field?.dataset?.candidateLabField);
  if (!WORKFLOW_RE.test(workflow) || !FIELD_RE.test(fieldName)) return null;
  return { workflow, fieldName, key: `${workflow}\u001f${fieldName}` };
}

function fieldType(field) {
  return nonEmptyText(field?.type).toLowerCase();
}

function isCheckable(field) {
  return ["checkbox", "radio"].includes(fieldType(field));
}

function isSelect(field) {
  return nonEmptyText(field?.tagName).toLowerCase() === "select"
    || Array.isArray(field?.options);
}

function isSafeScalarField(fieldName, field) {
  const type = fieldType(field);
  if (["number", "range", "date", "month"].includes(type)) return true;
  return (
    fieldName === "manual_breakpoints"
    || fieldName.endsWith("_manual_breakpoints")
  );
}

function boundedValue(value) {
  const text = String(value ?? "");
  if (text.length > MAX_VALUE_LENGTH || text.includes("\x00")) return null;
  return text;
}

function resolvedStorage(storage) {
  if (storage !== undefined) return storage;
  try {
    return globalThis.localStorage || null;
  } catch (_) {
    return null;
  }
}

function formFields(form) {
  return Array.from(
    form?.querySelectorAll?.("[data-candidate-lab-field]") || [],
  );
}

function selectedValues(field) {
  if (isCheckable(field)) {
    return field.checked === true ? [boundedValue(field.value)] : [];
  }
  if (isSelect(field)) {
    if (field.multiple === true) {
      return Array.from(field.selectedOptions || [])
        .map((option) => boundedValue(option.value))
        .filter((value) => value !== null);
    }
    const value = boundedValue(field.value);
    return value === null ? [] : [value];
  }
  const value = boundedValue(field.value);
  return value === null ? [] : [value];
}

function normalizedSnapshot(value) {
  if (
    !value
    || typeof value !== "object"
    || Array.isArray(value)
    || value.schema_version !== STATE_SCHEMA_VERSION
    || !Array.isArray(value.fields)
    || !Array.isArray(value.open_workflows)
  ) {
    return null;
  }
  const fields = [];
  const seen = new Set();
  for (const raw of value.fields.slice(0, MAX_FIELDS)) {
    const workflow = nonEmptyText(raw?.workflow);
    const fieldName = nonEmptyText(raw?.field);
    const kind = nonEmptyText(raw?.kind);
    const values = Array.isArray(raw?.values)
      ? raw.values.slice(0, MAX_VALUES_PER_FIELD)
        .map(boundedValue)
        .filter((item) => item !== null)
      : [];
    const key = `${workflow}\u001f${fieldName}`;
    if (
      !WORKFLOW_RE.test(workflow)
      || !FIELD_RE.test(fieldName)
      || !["checks", "select", "scalar"].includes(kind)
      || seen.has(key)
    ) {
      continue;
    }
    seen.add(key);
    fields.push({ workflow, field: fieldName, kind, values });
  }
  const openWorkflows = Array.from(new Set(
    value.open_workflows
      .map(nonEmptyText)
      .filter((workflow) => WORKFLOW_RE.test(workflow)),
  )).slice(0, MAX_FIELDS);
  return {
    schema_version: STATE_SCHEMA_VERSION,
    fields,
    open_workflows: openWorkflows,
  };
}

export function strategyCandidateLabStateStorageKey(taskId) {
  const normalized = nonEmptyText(taskId);
  return normalized ? `${STORAGE_PREFIX}:${encodeURIComponent(normalized)}` : "";
}

export function captureStrategyCandidateLabViewState(root) {
  const forms = Array.from(
    root?.querySelectorAll?.("[data-candidate-lab-form]") || [],
  );
  const grouped = new Map();
  const openWorkflows = [];
  for (const form of forms) {
    const workflow = nonEmptyText(form?.dataset?.candidateLabWorkflow);
    if (!WORKFLOW_RE.test(workflow)) continue;
    if (form.closest?.(".candidate-lab-launcher")?.open === true) {
      openWorkflows.push(workflow);
    }
    for (const field of formFields(form)) {
      const identity = fieldIdentity(form, field);
      if (!identity) continue;
      const checkable = isCheckable(field);
      const select = isSelect(field);
      if (!checkable && !select && !isSafeScalarField(identity.fieldName, field)) {
        continue;
      }
      const kind = checkable ? "checks" : select ? "select" : "scalar";
      const current = grouped.get(identity.key) || {
        workflow: identity.workflow,
        field: identity.fieldName,
        kind,
        values: [],
      };
      if (current.kind !== kind) continue;
      for (const value of selectedValues(field)) {
        if (
          value !== null
          && !current.values.includes(value)
          && current.values.length < MAX_VALUES_PER_FIELD
        ) {
          current.values.push(value);
        }
      }
      grouped.set(identity.key, current);
      if (grouped.size >= MAX_FIELDS) break;
    }
    if (grouped.size >= MAX_FIELDS) break;
  }
  return {
    schema_version: STATE_SCHEMA_VERSION,
    fields: Array.from(grouped.values()),
    open_workflows: Array.from(new Set(openWorkflows)).slice(0, MAX_FIELDS),
  };
}

function allowedOptionValues(field) {
  return new Set(
    Array.from(field?.options || [])
      .map((option) => boundedValue(option.value))
      .filter((value) => value !== null),
  );
}

function restoreField(fields, entry) {
  const values = new Set(entry.values);
  if (entry.kind === "checks") {
    for (const field of fields) {
      const value = boundedValue(field.value);
      field.checked = value !== null && values.has(value);
    }
    return;
  }
  const field = fields[0];
  if (!field) return;
  if (entry.kind === "select") {
    const allowed = allowedOptionValues(field);
    const restored = entry.values.filter((value) => allowed.has(value));
    if (field.multiple === true) {
      for (const option of Array.from(field.options || [])) {
        option.selected = restored.includes(String(option.value));
      }
      return;
    }
    if (restored.length === 1) field.value = restored[0];
    return;
  }
  if (entry.kind === "scalar" && isSafeScalarField(entry.field, field)) {
    field.value = entry.values[0] || "";
  }
}

export function restoreStrategyCandidateLabViewState(root, snapshot) {
  const normalized = normalizedSnapshot(snapshot);
  if (!normalized) return false;
  const forms = Array.from(
    root?.querySelectorAll?.("[data-candidate-lab-form]") || [],
  );
  const fieldsByKey = new Map();
  const formsByWorkflow = new Map();
  for (const form of forms) {
    const workflow = nonEmptyText(form?.dataset?.candidateLabWorkflow);
    if (!WORKFLOW_RE.test(workflow)) continue;
    formsByWorkflow.set(workflow, form);
    for (const field of formFields(form)) {
      const identity = fieldIdentity(form, field);
      if (!identity) continue;
      const current = fieldsByKey.get(identity.key) || [];
      current.push(field);
      fieldsByKey.set(identity.key, current);
    }
  }
  for (const entry of normalized.fields) {
    restoreField(
      fieldsByKey.get(`${entry.workflow}\u001f${entry.field}`) || [],
      entry,
    );
  }
  const open = new Set(normalized.open_workflows);
  for (const [workflow, form] of formsByWorkflow) {
    const launcher = form.closest?.(".candidate-lab-launcher");
    if (launcher) launcher.open = open.has(workflow);
  }
  return true;
}

export function loadStrategyCandidateLabViewState(
  taskId,
  storage = undefined,
) {
  const key = strategyCandidateLabStateStorageKey(taskId);
  const target = resolvedStorage(storage);
  if (!key || !target) return null;
  try {
    const raw = target.getItem(key);
    if (!raw || raw.length > MAX_SERIALIZED_BYTES) return null;
    return normalizedSnapshot(JSON.parse(raw));
  } catch (_) {
    return null;
  }
}

export function persistStrategyCandidateLabViewState(
  taskId,
  root,
  storage = undefined,
) {
  const key = strategyCandidateLabStateStorageKey(taskId);
  const target = resolvedStorage(storage);
  if (!key || !root || !target) return false;
  try {
    const payload = captureStrategyCandidateLabViewState(root);
    const raw = JSON.stringify(payload);
    if (raw.length > MAX_SERIALIZED_BYTES) return false;
    target.setItem(key, raw);
    return true;
  } catch (_) {
    return false;
  }
}

export { STATE_SCHEMA_VERSION as STRATEGY_CANDIDATE_LAB_STATE_SCHEMA_VERSION };
