export function createComingSoonToastController({
  body = document.body,
  clearTimeoutFn = clearTimeout,
  createElement = (tagName) => document.createElement(tagName),
  getElementById = (id) => document.getElementById(id),
  setTimeoutFn = setTimeout,
  visibleDurationMs = 2400,
} = {}) {
  let visibleTimer = null;

  function ensureToast() {
    let toast = getElementById("comingSoonToast");
    if (toast) return toast;
    toast = createElement("div");
    toast.id = "comingSoonToast";
    toast.className = "coming-soon-toast";
    toast.setAttribute("role", "status");
    toast.setAttribute("aria-live", "polite");
    body.appendChild(toast);
    return toast;
  }

  function showComingSoonToast(message) {
    const toast = ensureToast();
    toast.textContent = message;
    toast.classList.add("is-visible");
    if (visibleTimer) clearTimeoutFn(visibleTimer);
    visibleTimer = setTimeoutFn(() => {
      toast.classList.remove("is-visible");
      visibleTimer = null;
    }, visibleDurationMs);
  }

  return {
    showComingSoonToast,
  };
}
