"""Exercise the actual memory controller entry points after facade cleanup."""

from pathlib import Path
import subprocess


def test_memory_view_switch_and_list_inspection_use_current_controller():
    subprocess.run(
        ["node", "--input-type=module", "-e", r'''
import assert from "node:assert/strict";
import { createAgentMemoryPanelController } from "./marvis/static/js/agent-memory-panel.js";
function element(value = "") {
  const classes = new Set();
  return {
    value, innerHTML: "", textContent: "", dataset: {}, attributes: {},
    classList: {
      toggle(name, enabled) { enabled ? classes.add(name) : classes.delete(name); },
      contains(name) { return classes.has(name); },
    },
    setAttribute(name, value) { this.attributes[name] = value; },
  };
}
const elements = new Map([
  "agentMemoryStatus", "agentMemoryStatusFilter", "agentMemoryList", "agentMemoryDetail",
  "agentMemorySourceTaskFilterRow", "agentMemoryModelFilterRow",
].map((id) => [id, element()]));
elements.get("agentMemoryStatusFilter").value = "active";
const tabs = ["raw", "distillation"].map((mode) => {
  const tab = element(); tab.dataset.agentMemoryView = mode; return tab;
});
globalThis.document = { querySelectorAll: () => tabs };
const calls = [];
const pending = [];
const panel = createAgentMemoryPanelController({
  $: (id) => elements.get(id),
  api: async (url) => {
    calls.push(url);
    if (url.includes("?")) return { items: [{ id: "memory/1", title: "样本口径", status: "active" }] };
    return { distillation: { id: "memory/1", title: "已核对的口径", status: "active" }, events: [] };
  },
  runAction: (action) => pending.push(action()),
});
panel.setViewMode("distillation", { reload: false });
assert.equal(panel.viewMode(), "distillation");
assert.equal(tabs[1].attributes["aria-selected"], "true");
assert.equal(tabs[0].attributes["aria-selected"], "false");
assert.ok(elements.get("agentMemorySourceTaskFilterRow").classList.contains("agent-memory-filter-hidden"));
await panel.loadItems();
assert.ok(calls[0].startsWith("api/agent-memory/distillations?"));
assert.ok(panel.hasItems());
assert.match(elements.get("agentMemoryList").innerHTML, /样本口径/);
let prevented = false;
panel.handleListClick({
  preventDefault() { prevented = true; },
  target: { closest: () => ({ dataset: { agentMemoryAction: "inspect", agentMemoryId: "memory/1" } }) },
});
await Promise.all(pending);
assert.equal(prevented, true);
assert.equal(calls.at(-1), "api/agent-memory/distillations/memory%2F1");
assert.match(elements.get("agentMemoryDetail").innerHTML, /已核对的口径/);
panel.setViewMode("raw", { reload: false });
assert.equal(panel.hasItems(), false);
assert.equal(elements.get("agentMemoryDetail").innerHTML, "");
assert.equal(tabs[0].attributes["aria-selected"], "true");
assert.equal(elements.get("agentMemorySourceTaskFilterRow").classList.contains("agent-memory-filter-hidden"), false);
'''],
        cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, text=True,
    )
