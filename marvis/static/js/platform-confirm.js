// Mini line icons for the purge chips, keyed by purge_summary field name.
// Static markup only — labels/counts go through textContent, never innerHTML.
const PURGE_CHIP_ICONS = {
  datasets:
    '<svg viewBox="0 0 24 24" focusable="false" aria-hidden="true"><ellipse cx="12" cy="6" rx="7" ry="2.6"></ellipse><path d="M5 6v12c0 1.4 3.1 2.6 7 2.6s7-1.2 7-2.6V6"></path><path d="M5 12c0 1.4 3.1 2.6 7 2.6s7-1.2 7-2.6"></path></svg>',
  joins:
    '<svg viewBox="0 0 24 24" focusable="false" aria-hidden="true"><rect x="4" y="8.5" width="10" height="10" rx="2.5"></rect><rect x="10" y="5.5" width="10" height="10" rx="2.5"></rect></svg>',
  plans:
    '<svg viewBox="0 0 24 24" focusable="false" aria-hidden="true"><path d="M9 5.5h9.5a1.5 1.5 0 0 1 1.5 1.5v12a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 19V7a1.5 1.5 0 0 1 1.5-1.5H9z"></path><path d="M8 11h8M8 15h5"></path></svg>',
  experiments:
    '<svg viewBox="0 0 24 24" focusable="false" aria-hidden="true"><path d="M9.8 4.5h4.4M10.8 4.5v5.2L6.2 17a2.4 2.4 0 0 0 2.1 3.5h7.4a2.4 2.4 0 0 0 2.1-3.5l-4.6-7.3V4.5"></path><path d="M8.4 14.6h7.2"></path></svg>',
  model_artifacts:
    '<svg viewBox="0 0 24 24" focusable="false" aria-hidden="true"><path d="M12 3.8 19 7.7v8.6L12 20.2 5 16.3V7.7Z"></path><path d="M5.3 7.9 12 11.7l6.7-3.8M12 11.9v8"></path></svg>',
  strategies:
    '<svg viewBox="0 0 24 24" focusable="false" aria-hidden="true"><path d="M5 8h14M5 16h14"></path><circle cx="10" cy="8" r="2.1"></circle><circle cx="15" cy="16" r="2.1"></circle></svg>',
};

export function createPlatformConfirmController({ getElementById } = {}) {
  const $ = typeof getElementById === "function"
    ? getElementById
    : (id) => document.getElementById(id);
  let resolver = null;

  function renderPlatformConfirmMessage(message, messageParts) {
    const node = $("platformConfirmMessage");
    if (!node) return;
    const parts = Array.isArray(messageParts)
      ? messageParts.filter((part) => part && part.text)
      : [];
    if (!parts.length) {
      node.textContent = message;
      return;
    }
    node.textContent = "";
    const doc = node.ownerDocument;
    for (const part of parts) {
      if (part.strong) {
        const strong = doc.createElement("strong");
        strong.textContent = part.text;
        node.appendChild(strong);
      } else {
        node.appendChild(doc.createTextNode(part.text));
      }
    }
  }

  function renderPlatformConfirmPurge(purgeItems) {
    const container = $("platformConfirmPurge");
    if (!container) return;
    for (const chip of Array.from(container.querySelectorAll(".platform-confirm-chip"))) {
      chip.remove();
    }
    const items = Array.isArray(purgeItems)
      ? purgeItems.filter((item) => item && item.label && item.count)
      : [];
    if (!items.length) {
      container.hidden = true;
      return;
    }
    const doc = container.ownerDocument;
    for (const item of items) {
      const chip = doc.createElement("span");
      chip.className = "platform-confirm-chip";
      chip.innerHTML = PURGE_CHIP_ICONS[item.key] || "";
      chip.appendChild(doc.createTextNode(`${item.label} `));
      const count = doc.createElement("b");
      count.textContent = String(item.count);
      chip.appendChild(count);
      container.appendChild(chip);
    }
    container.hidden = false;
  }

  function closePlatformConfirmDialog(confirmed = false) {
    const pending = resolver;
    resolver = null;
    const dialog = $("platformConfirmDialog");
    if (dialog?.open) {
      dialog.close(confirmed ? "confirm" : "cancel");
    }
    if (pending) pending(confirmed);
  }

  function showPlatformConfirm({
    title = "确认操作",
    message = "此操作不能撤销。",
    messageParts = null,
    purgeItems = null,
    confirmText = "确认",
    cancelText = "取消",
    tone = "default",
  } = {}) {
    const dialog = $("platformConfirmDialog");
    if (!dialog) return Promise.resolve(false);
    if (resolver) closePlatformConfirmDialog(false);

    $("platformConfirmTitle").textContent = title;
    renderPlatformConfirmMessage(message, messageParts);
    renderPlatformConfirmPurge(purgeItems);
    $("platformConfirmConfirmButton").textContent = confirmText;
    $("platformConfirmCancelButton").textContent = cancelText;
    dialog.dataset.tone = tone;

    return new Promise((resolve) => {
      resolver = resolve;
      dialog.showModal();
      $("platformConfirmCancelButton").focus({ preventScroll: true });
    });
  }

  function bindPlatformConfirmDialog() {
    const dialog = $("platformConfirmDialog");
    if (!dialog) return;
    $("platformConfirmCancelButton").onclick = () => closePlatformConfirmDialog(false);
    $("platformConfirmConfirmButton").onclick = () => closePlatformConfirmDialog(true);
    dialog.addEventListener("cancel", (event) => {
      event.preventDefault();
      closePlatformConfirmDialog(false);
    });
    dialog.addEventListener("close", () => {
      if (resolver) closePlatformConfirmDialog(false);
    });
  }

  return {
    bindPlatformConfirmDialog,
    closePlatformConfirmDialog,
    showPlatformConfirm,
  };
}
