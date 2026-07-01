export const SIDEBAR_WIDTH_MIN = 314;
export const SIDEBAR_WIDTH_MAX = 520;
export const PROGRESS_WIDTH_MIN = 314;
export const PROGRESS_WIDTH_MAX = 560;

export function createLayoutResizeController({
  body = document.body,
  clamp,
  getComputedStyleFn = getComputedStyle,
  root = document.documentElement,
  storage = localStorage,
  windowObj = window,
} = {}) {
  function setCssNumber(name, value) {
    root.style.setProperty(name, `${Math.round(value)}px`);
  }

  function cssNumber(name) {
    return parseInt(getComputedStyleFn(root).getPropertyValue(name), 10);
  }

  function saveLayoutWidths() {
    try {
      storage.setItem(
        "marvis_layout",
        JSON.stringify({
          sidebar: cssNumber("--sidebar-width"),
          progress: cssNumber("--progress-width"),
        }),
      );
    } catch (_) {
      // Layout persistence is optional in restricted notebook browsers.
    }
  }

  function restoreLayoutWidths() {
    try {
      const stored = JSON.parse(storage.getItem("marvis_layout") || "{}");
      if (stored.sidebar) {
        setCssNumber(
          "--sidebar-width",
          clamp(
            stored.sidebar === 320 ? SIDEBAR_WIDTH_MIN : stored.sidebar,
            SIDEBAR_WIDTH_MIN,
            SIDEBAR_WIDTH_MAX,
          ),
        );
      }
      if (stored.progress) {
        setCssNumber("--progress-width", clamp(stored.progress, PROGRESS_WIDTH_MIN, PROGRESS_WIDTH_MAX));
      }
    } catch (_) {
      // Keep CSS defaults when storage is unavailable or invalid.
    }
  }

  function startResizeDrag(side, event) {
    event.preventDefault();
    const startX = event.clientX;
    const startSidebar = cssNumber("--sidebar-width");
    const startProgress = cssNumber("--progress-width");

    function onPointerMove(moveEvent) {
      const deltaX = moveEvent.clientX - startX;
      if (side === "left") {
        setCssNumber("--sidebar-width", clamp(startSidebar + deltaX, SIDEBAR_WIDTH_MIN, SIDEBAR_WIDTH_MAX));
      } else {
        setCssNumber("--progress-width", clamp(startProgress - deltaX, PROGRESS_WIDTH_MIN, PROGRESS_WIDTH_MAX));
      }
    }

    function onPointerUp() {
      body.classList.remove("is-resizing");
      windowObj.removeEventListener("pointermove", onPointerMove);
      windowObj.removeEventListener("pointerup", onPointerUp);
      saveLayoutWidths();
    }

    body.classList.add("is-resizing");
    windowObj.addEventListener("pointermove", onPointerMove);
    windowObj.addEventListener("pointerup", onPointerUp);
  }

  function handleResizeKey(side, event) {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    const step = event.shiftKey ? 32 : 12;
    const direction = event.key === "ArrowRight" ? 1 : -1;
    if (side === "left") {
      const current = cssNumber("--sidebar-width");
      setCssNumber("--sidebar-width", clamp(current + direction * step, SIDEBAR_WIDTH_MIN, SIDEBAR_WIDTH_MAX));
    } else {
      const current = cssNumber("--progress-width");
      setCssNumber("--progress-width", clamp(current - direction * step, PROGRESS_WIDTH_MIN, PROGRESS_WIDTH_MAX));
    }
    saveLayoutWidths();
  }

  return {
    handleResizeKey,
    restoreLayoutWidths,
    saveLayoutWidths,
    setCssNumber,
    startResizeDrag,
  };
}
