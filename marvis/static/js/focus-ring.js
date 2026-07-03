export function formControlFocusTarget(target) {
  return target?.closest?.("input, textarea, select") || null;
}

export function installFormControlFocusRingGuard({
  activeElement = () => document.activeElement,
  now = () => performance.now(),
  root = document,
  setTimeoutFn = setTimeout,
  suppressionWindowMs = 750,
} = {}) {
  let lastPointerDownControl = null;
  let lastPointerDownAt = 0;

  function handleFormControlPointerDown(event) {
    const control = formControlFocusTarget(event.target);
    lastPointerDownControl = control;
    lastPointerDownAt = now();
    if (control) control.classList.remove("suppress-pointer-focus-ring");
  }

  function handleFormControlFocusIn(event) {
    const control = formControlFocusTarget(event.target);
    if (!control) return;
    const pointerFocusPending = now() - lastPointerDownAt < suppressionWindowMs;
    control.classList.toggle(
      "suppress-pointer-focus-ring",
      pointerFocusPending && lastPointerDownControl !== control,
    );
    lastPointerDownControl = null;
    lastPointerDownAt = 0;
  }

  function handleFormControlFocusOut(event) {
    const control = formControlFocusTarget(event.target);
    if (control) control.classList.remove("suppress-pointer-focus-ring");
  }

  function handleFormControlLabelClick(event) {
    const clickedControl = formControlFocusTarget(event.target);
    if (clickedControl) {
      clickedControl.classList.remove("suppress-pointer-focus-ring");
      return;
    }
    const label = event.target.closest?.("label");
    if (!label) return;
    setTimeoutFn(() => {
      const focused = formControlFocusTarget(activeElement());
      if (!focused) return;
      const labelTargetsFocusedControl =
        label.contains(focused) || Boolean(label.htmlFor && focused.id === label.htmlFor);
      if (labelTargetsFocusedControl) focused.classList.add("suppress-pointer-focus-ring");
    }, 0);
  }

  root.addEventListener("pointerdown", handleFormControlPointerDown, true);
  root.addEventListener("mousedown", handleFormControlPointerDown, true);
  root.addEventListener("touchstart", handleFormControlPointerDown, true);
  root.addEventListener("click", handleFormControlLabelClick, true);
  root.addEventListener("focusin", handleFormControlFocusIn, true);
  root.addEventListener("focusout", handleFormControlFocusOut, true);
}
