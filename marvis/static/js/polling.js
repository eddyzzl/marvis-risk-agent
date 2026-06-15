export function createProgressPollRegistry() {
  return new Map();
}

export function claimProgressPoll(registry, taskId, { background = false } = {}) {
  const existing = registry.get(taskId);
  if (existing) {
    if (background || !existing.background) return { claimed: false, existing };
    existing.cancelled = true;
  }
  const pollState = { background, cancelled: false, promise: null };
  registry.set(taskId, pollState);
  return { claimed: true, pollState };
}

export function releaseProgressPoll(registry, taskId, pollState) {
  if (registry.get(taskId) === pollState) registry.delete(taskId);
}
