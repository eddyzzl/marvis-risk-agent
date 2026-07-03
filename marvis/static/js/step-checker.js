// Status checker shown before every step / sub-step: a hollow ring (pending),
// a spinning arc (running), a filled green tick (succeeded), or a red cross
// (failed). SVG marks use currentColor so CSS controls the glyph color.
export function stepCheckerHtml(state) {
  if (state === "succeeded") {
    return (
      '<span class="check-icon succeeded" aria-hidden="true">' +
      '<svg viewBox="0 0 16 16" width="11" height="11"><path d="M3 8.4l3 3 7-7" fill="none" ' +
      'stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>' +
      "</span>"
    );
  }
  if (state === "failed") {
    return (
      '<span class="check-icon failed" aria-hidden="true">' +
      '<svg viewBox="0 0 16 16" width="10" height="10"><path d="M4 4l8 8M12 4l-8 8" fill="none" ' +
      'stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/></svg>' +
      "</span>"
    );
  }
  if (state === "stopped") {
    return '<span class="check-icon stopped" aria-hidden="true"></span>';
  }
  if (state === "review") {
    return (
      '<span class="check-icon review" aria-hidden="true">' +
      '<svg viewBox="0 0 16 16" width="10" height="10"><path d="M8 3v6M8 12.5h.01" fill="none" ' +
      'stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/></svg>' +
      "</span>"
    );
  }
  if (state === "running") {
    // Sync the spin phase to a global clock so the ring keeps rotating smoothly
    // even though the stepper is rebuilt from scratch on every poll tick.
    return `<span class="check-icon running" aria-hidden="true" style="animation-delay: -${Date.now() % 800}ms"></span>`;
  }
  return '<span class="check-icon pending" aria-hidden="true"></span>';
}
