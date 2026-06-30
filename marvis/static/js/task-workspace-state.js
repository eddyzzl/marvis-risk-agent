export function rememberSelectedTaskId(storageKey, taskId, storage = localStorage) {
  try {
    if (taskId) {
      storage.setItem(storageKey, taskId);
    } else {
      storage.removeItem(storageKey);
    }
  } catch (_) {
    // Browser storage can be unavailable in private or embedded contexts.
  }
}

export function storedSelectedTaskId(storageKey, storage = localStorage) {
  try {
    return storage.getItem(storageKey) || "";
  } catch (_) {
    return "";
  }
}

export function loadResultScrollPositions(storageKey, targetMap, storage = localStorage) {
  targetMap?.clear?.();
  let raw = "";
  try {
    raw = storage.getItem(storageKey) || "";
  } catch (_) {
    return;
  }
  if (!raw) return;
  try {
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return;
    Object.entries(parsed).forEach(([taskId, scrollTop]) => {
      const normalized = Number(scrollTop);
      if (taskId && Number.isFinite(normalized) && normalized >= 0) {
        targetMap?.set?.(taskId, normalized);
      }
    });
  } catch (_) {
    // Ignore stale or malformed browser storage.
  }
}

export function persistResultScrollPositions(storageKey, positionsByTask, storage = localStorage) {
  try {
    const payload = {};
    positionsByTask?.forEach?.((scrollTop, taskId) => {
      if (!taskId || !Number.isFinite(scrollTop) || scrollTop < 0) return;
      payload[taskId] = scrollTop;
    });
    if (Object.keys(payload).length === 0) {
      storage.removeItem(storageKey);
      return;
    }
    storage.setItem(storageKey, JSON.stringify(payload));
  } catch (_) {
    // Browser storage can be unavailable in private or embedded contexts.
  }
}

export function workspaceGreetingForHour(hour) {
  if (hour >= 5 && hour < 9) return "早上好";
  if (hour >= 9 && hour < 12) return "上午好";
  if (hour >= 12 && hour < 18) return "下午好";
  return "晚上好";
}
