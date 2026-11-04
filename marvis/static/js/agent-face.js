// D6 (2026-10-14 redesign): the Agent avatar is the mascot head built from two
// layers of the real logo pixels (brand/marvis-face-base.png for the head,
// brand/marvis-face-eyes.png for the eyes); css/agent-face.css moves the eye
// layer per expression. The expression always comes from the face field of the
// unified status vocabulary (js/task-status.js), never from a status word.
// Pure module: no DOM access at import time, so node test harnesses load it.
import { escapeHtml } from "./ui-utils.js";
import { statusPresentation } from "./task-status.js";

export const AGENT_FACES = Object.freeze(["idle", "busy", "wait", "done", "fail"]);
// sm 22px / md 30px / lg 132px (mockup .face.sm / .face / .face.lg).
const SIZES = new Set(["sm", "md", "lg"]);

function knownFace(face) {
  return AGENT_FACES.includes(face) ? face : "idle";
}

export function agentFaceHtml({ face = "idle", size = "md", label = "" } = {}) {
  const resolvedSize = SIZES.has(size) ? size : "md";
  const a11y = label ? ` role="img" aria-label="${escapeHtml(label)}"` : ' aria-hidden="true"';
  return `<span class="agent-face" data-face="${knownFace(face)}" data-size="${resolvedSize}"${a11y}></span>`;
}

// info is a statusPresentation()/actionStatusPresentation() result; a bare key
// also resolves. No status (an empty title pill) reads as idle.
export function agentFaceForStatus(info) {
  if (!info) return "idle";
  if (typeof info === "string") return knownFace(statusPresentation(info).face);
  return knownFace(info.face || (info.key ? statusPresentation(info.key).face : ""));
}

// The newest run-start avatar mirrors the task state; earlier ones stay idle.
// Only data-face changes, so a status change never re-renders the messages.
export function syncLiveAgentFaces(root, face) {
  const faces = root?.querySelectorAll?.(".agent-message.assistant[data-run-start] > .agent-face");
  if (!faces?.length) return;
  const live = knownFace(face);
  const lastIndex = faces.length - 1;
  Array.from(faces).forEach((node, index) => {
    const next = index === lastIndex ? live : "idle";
    if (node.dataset.face !== next) node.dataset.face = next;
  });
}
