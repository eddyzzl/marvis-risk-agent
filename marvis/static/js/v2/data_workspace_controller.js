import {
  getDataWorkspace as fetchDataWorkspace,
  putDataWorkspace as updateDataWorkspace,
} from "./api_v2.js";

export const DATA_WORKSPACE_EDITABLE_FIELDS = Object.freeze([
  "active_dataset_id",
  "active_dataset_content_hash",
  "page",
  "selected_field",
  "semantic_mapping",
]);

const SHA256_PATTERN = /^[0-9a-f]{64}$/;

function cloneValue(value) {
  if (Array.isArray(value)) {
    return value.map((item) => cloneValue(item));
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [key, cloneValue(item)]),
    );
  }
  return value;
}

function isPlainObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

export function deepStructuralEqual(left, right) {
  if (Object.is(left, right)) return true;
  if (Array.isArray(left) || Array.isArray(right)) {
    if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) {
      return false;
    }
    return left.every((item, index) => deepStructuralEqual(item, right[index]));
  }
  if (!isPlainObject(left) || !isPlainObject(right)) return false;
  const leftKeys = Object.keys(left);
  const rightKeys = Object.keys(right);
  if (leftKeys.length !== rightKeys.length) return false;
  return leftKeys.every(
    (key) => Object.prototype.hasOwnProperty.call(right, key)
      && deepStructuralEqual(left[key], right[key]),
  );
}

export function emptySemanticMapping() {
  return {
    target_col: null,
    field_roles: {},
    business_names: {},
  };
}

function normalizeRecord(value) {
  return isPlainObject(value) ? cloneValue(value) : {};
}

function normalizeSemanticMapping(value) {
  const mapping = isPlainObject(value) ? value : {};
  return {
    target_col: mapping.target_col ?? null,
    field_roles: normalizeRecord(mapping.field_roles),
    business_names: normalizeRecord(mapping.business_names),
  };
}

function editableWorkspace(value) {
  const workspace = isPlainObject(value) ? value : {};
  return {
    active_dataset_id: workspace.active_dataset_id ?? null,
    active_dataset_content_hash: workspace.active_dataset_content_hash ?? null,
    page: typeof workspace.page === "string" && workspace.page ? workspace.page : "overview",
    selected_field: workspace.selected_field ?? null,
    semantic_mapping: normalizeSemanticMapping(workspace.semantic_mapping),
  };
}

function mergeDraft(current, patch) {
  if (!isPlainObject(patch)) {
    throw new TypeError("data workspace edit requires an object or updater function");
  }
  const next = { ...current };
  for (const field of DATA_WORKSPACE_EDITABLE_FIELDS) {
    if (field !== "semantic_mapping" && Object.prototype.hasOwnProperty.call(patch, field)) {
      next[field] = cloneValue(patch[field]);
    }
  }
  if (Object.prototype.hasOwnProperty.call(patch, "semantic_mapping")) {
    const mappingPatch = patch.semantic_mapping;
    if (!isPlainObject(mappingPatch)) {
      next.semantic_mapping = emptySemanticMapping();
    } else {
      next.semantic_mapping = {
        ...current.semantic_mapping,
        ...cloneValue(mappingPatch),
        field_roles: Object.prototype.hasOwnProperty.call(mappingPatch, "field_roles")
          ? normalizeRecord(mappingPatch.field_roles)
          : cloneValue(current.semantic_mapping.field_roles),
        business_names: Object.prototype.hasOwnProperty.call(mappingPatch, "business_names")
          ? normalizeRecord(mappingPatch.business_names)
          : cloneValue(current.semantic_mapping.business_names),
      };
    }
  }
  return editableWorkspace(next);
}

function datasetIdentity(dataset, contentHash) {
  if (isPlainObject(dataset)) {
    const identity = {
      id: dataset.id ?? dataset.dataset_id ?? dataset.active_dataset_id ?? null,
      contentHash: contentHash
        ?? dataset.content_hash
        ?? dataset.active_dataset_content_hash
        ?? null,
    };
    return validatedDatasetIdentity(identity);
  }
  return validatedDatasetIdentity({ id: dataset ?? null, contentHash: contentHash ?? null });
}

function validatedDatasetIdentity(identity) {
  const { id, contentHash } = identity;
  const hasId = typeof id === "string" && id.length > 0;
  const hasHash = typeof contentHash === "string" && SHA256_PATTERN.test(contentHash);
  if (id !== null && !hasId) {
    throw new TypeError("active dataset id must be a non-empty string or null");
  }
  if (contentHash !== null && !hasHash) {
    throw new TypeError("active dataset content hash must be a lowercase SHA-256 digest or null");
  }
  if (hasId !== hasHash) {
    throw new TypeError("active dataset id and content hash must both be null or non-null");
  }
  return identity;
}

function assertSnapshotTask(snapshot, taskId) {
  if (!isPlainObject(snapshot)) {
    throw new TypeError("data workspace response must be an object");
  }
  if (typeof snapshot.task_id !== "string" || snapshot.task_id !== taskId) {
    throw new Error("data workspace response belongs to a different task");
  }
  if (!Number.isInteger(snapshot.revision) || snapshot.revision < 0) {
    throw new TypeError("data workspace response revision must be a non-negative integer");
  }
}

export function createDataWorkspaceController(dependencies = {}) {
  const loadWorkspace = dependencies.getDataWorkspace || fetchDataWorkspace;
  const persistWorkspace = dependencies.putDataWorkspace || updateDataWorkspace;
  const listeners = new Set();

  let taskId = "";
  let serverSnapshot = null;
  let draft = null;
  let loading = false;
  let saving = false;
  let error = null;
  let operation = 0;
  let savePromise = null;

  function isDirty() {
    return Boolean(
      serverSnapshot
      && draft
      && !deepStructuralEqual(draft, editableWorkspace(serverSnapshot)),
    );
  }

  function getState() {
    return {
      taskId,
      serverSnapshot: serverSnapshot ? cloneValue(serverSnapshot) : null,
      draft: draft ? cloneValue(draft) : null,
      dirty: isDirty(),
      loading,
      saving,
      error,
    };
  }

  function emit() {
    const state = getState();
    for (const listener of listeners) listener(state);
  }

  function requireLoaded() {
    if (!taskId || !serverSnapshot || !draft) {
      throw new Error("load a data workspace before editing it");
    }
  }

  async function load(nextTaskId) {
    const requestedTaskId = String(nextTaskId ?? "");
    if (!requestedTaskId) throw new TypeError("taskId is required");
    if (savePromise) {
      throw new Error("cannot load a data workspace while a save is in progress");
    }
    if (taskId && isDirty()) {
      throw new Error(
        "unsaved data workspace changes must be saved or discarded before loading a workspace",
      );
    }

    const loadOperation = ++operation;
    taskId = requestedTaskId;
    serverSnapshot = null;
    draft = null;
    loading = true;
    saving = false;
    savePromise = null;
    error = null;
    emit();

    try {
      const loaded = await loadWorkspace(requestedTaskId);
      if (loadOperation !== operation) return getState();
      assertSnapshotTask(loaded, requestedTaskId);
      serverSnapshot = cloneValue(loaded);
      draft = editableWorkspace(loaded);
      loading = false;
      emit();
      return getState();
    } catch (loadError) {
      if (loadOperation === operation) {
        loading = false;
        error = loadError;
        emit();
      }
      throw loadError;
    }
  }

  function edit(patchOrUpdater) {
    requireLoaded();
    if (typeof patchOrUpdater === "function") {
      const workingDraft = cloneValue(draft);
      const result = patchOrUpdater(workingDraft);
      draft = editableWorkspace(result === undefined ? workingDraft : result);
    } else {
      draft = mergeDraft(draft, patchOrUpdater);
    }
    error = null;
    emit();
    return getDraft();
  }

  function discard() {
    requireLoaded();
    if (savePromise) {
      throw new Error("cannot discard data workspace changes while a save is in progress");
    }
    draft = editableWorkspace(serverSnapshot);
    error = null;
    emit();
    return getDraft();
  }

  function activateDataset(dataset, contentHash) {
    requireLoaded();
    const identity = datasetIdentity(dataset, contentHash);
    draft = {
      active_dataset_id: identity.id,
      active_dataset_content_hash: identity.contentHash,
      page: "overview",
      selected_field: null,
      semantic_mapping: emptySemanticMapping(),
    };
    error = null;
    emit();
    return getDraft();
  }

  function save() {
    requireLoaded();
    if (!isDirty()) return Promise.resolve(getState());
    if (savePromise) return savePromise;

    const saveOperation = operation;
    const expectedRevision = serverSnapshot.revision;
    const submittedDraft = cloneValue(draft);
    saving = true;
    error = null;
    emit();

    const pending = (async () => {
      try {
        const saved = await Promise.resolve().then(
          () => persistWorkspace(taskId, submittedDraft, expectedRevision),
        );
        if (saveOperation !== operation) return getState();
        assertSnapshotTask(saved, taskId);
        serverSnapshot = cloneValue(saved);
        if (deepStructuralEqual(draft, submittedDraft)) {
          draft = editableWorkspace(saved);
        }
        return getState();
      } catch (saveError) {
        if (saveOperation === operation) error = saveError;
        throw saveError;
      } finally {
        if (saveOperation === operation) {
          saving = false;
          if (savePromise === pending) savePromise = null;
          emit();
        }
      }
    })();
    savePromise = pending;
    return pending;
  }

  async function guardNavigation(resolveChoice, navigate) {
    // A dispatched PUT cannot be cancelled safely. Block navigation until it
    // settles so a "discard" choice never pretends that the server write was
    // undone while the response is ignored after a task switch.
    if (savePromise) return false;
    if (!isDirty()) {
      if (typeof navigate === "function") await navigate(getState());
      return true;
    }

    const rawChoice = typeof resolveChoice === "function"
      ? await resolveChoice(getState())
      : await resolveChoice;
    const choice = String(rawChoice ?? "cancel").toLowerCase();
    if (choice === "cancel") return false;
    if (choice === "save") await save();
    else if (choice === "discard") discard();
    else throw new Error(`unknown data workspace navigation choice: ${choice}`);

    // An edit made while a save was in flight remains a draft and still blocks
    // navigation; successful navigation must never imply that draft was saved.
    if (isDirty()) return false;
    if (typeof navigate === "function") await navigate(getState());
    return true;
  }

  function getServerSnapshot() {
    return serverSnapshot ? cloneValue(serverSnapshot) : null;
  }

  function getDraft() {
    return draft ? cloneValue(draft) : null;
  }

  function subscribe(listener) {
    if (typeof listener !== "function") throw new TypeError("listener must be a function");
    listeners.add(listener);
    return () => listeners.delete(listener);
  }

  return {
    activateDataset,
    discard,
    edit,
    getDraft,
    getServerSnapshot,
    getState,
    guardNavigation,
    isDirty,
    load,
    requestNavigation: guardNavigation,
    save,
    subscribe,
    updateDraft: edit,
  };
}

export const createDataWorkspaceSession = createDataWorkspaceController;
