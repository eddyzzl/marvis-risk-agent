export function createPlatformConfirmController({ getElementById } = {}) {
  const $ = typeof getElementById === "function"
    ? getElementById
    : (id) => document.getElementById(id);
  let resolver = null;

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
    confirmText = "确认",
    cancelText = "取消",
    tone = "default",
  } = {}) {
    const dialog = $("platformConfirmDialog");
    if (!dialog) return Promise.resolve(false);
    if (resolver) closePlatformConfirmDialog(false);

    $("platformConfirmTitle").textContent = title;
    $("platformConfirmMessage").textContent = message;
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
