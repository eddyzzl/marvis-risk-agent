const v2Defaults = {
  "v2.currentPlan": null,
  "v2.selectedStepId": "",
  "v2.plugins": [],
  "v2.datasets": [],
  "v2.currentJoin": null,
  "v2.capabilityTiers": [],
  "v2.selectedTier": "",
  "v2.loopEvents": [],
};

const state = new Map();
const subscribers = new Map();

function cloneDefault(value) {
  if (Array.isArray(value)) {
    return [];
  }
  if (value && typeof value === "object") {
    return { ...value };
  }
  return value;
}

function ensureKey(key) {
  if (!Object.prototype.hasOwnProperty.call(v2Defaults, key)) {
    throw new Error(`Unknown v2 state key: ${key}`);
  }
}

function notify(key, next, previous) {
  for (const fn of subscribers.get(key) || []) {
    fn(next, previous);
  }
}

export function resetV2State() {
  for (const [key, value] of Object.entries(v2Defaults)) {
    state.set(key, cloneDefault(value));
  }
}

export function setState(key, value) {
  ensureKey(key);
  const previous = state.get(key);
  state.set(key, value);
  if (previous !== value) {
    notify(key, value, previous);
  }
}

export function getState(key) {
  ensureKey(key);
  return state.get(key);
}

export function subscribe(key, fn) {
  ensureKey(key);
  if (typeof fn !== "function") {
    throw new Error("subscribe requires a function");
  }
  if (!subscribers.has(key)) {
    subscribers.set(key, new Set());
  }
  subscribers.get(key).add(fn);
  return () => subscribers.get(key)?.delete(fn);
}

export const setPlan = (plan) => setState("v2.currentPlan", plan);
export const getPlan = () => getState("v2.currentPlan");
export const onPlanChange = (fn) => subscribe("v2.currentPlan", fn);

export const setSelectedStepId = (stepId) => setState("v2.selectedStepId", stepId);
export const getSelectedStepId = () => getState("v2.selectedStepId");
export const onSelectedStepChange = (fn) => subscribe("v2.selectedStepId", fn);

export const setPlugins = (plugins) => setState("v2.plugins", plugins);
export const getPlugins = () => getState("v2.plugins");
export const onPluginsChange = (fn) => subscribe("v2.plugins", fn);

export const setDatasets = (datasets) => setState("v2.datasets", datasets);
export const getDatasets = () => getState("v2.datasets");
export const onDatasetsChange = (fn) => subscribe("v2.datasets", fn);

export const setCurrentJoin = (joinPlan) => setState("v2.currentJoin", joinPlan);
export const getCurrentJoin = () => getState("v2.currentJoin");
export const onCurrentJoinChange = (fn) => subscribe("v2.currentJoin", fn);

export const setCapabilityTiers = (tiers) => setState("v2.capabilityTiers", tiers);
export const getCapabilityTiers = () => getState("v2.capabilityTiers");
export const onCapabilityTiersChange = (fn) => subscribe("v2.capabilityTiers", fn);

export const setSelectedTier = (tier) => setState("v2.selectedTier", tier);
export const getSelectedTier = () => getState("v2.selectedTier");
export const onSelectedTierChange = (fn) => subscribe("v2.selectedTier", fn);

export const setLoopEvents = (events) => setState("v2.loopEvents", events);
export const getLoopEvents = () => getState("v2.loopEvents");
export const onLoopEventsChange = (fn) => subscribe("v2.loopEvents", fn);

export const v2StateKeys = Object.freeze(Object.keys(v2Defaults));

resetV2State();
