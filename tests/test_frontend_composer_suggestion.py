"""V3 UI-6: the composer's grey suggestion is taken with → while empty.

Home: the placeholder example (without 「例如：」). Task: only the reply the
Agent suggested on its latest turn; taking it fills the input, never sends.
"""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_module(script: str) -> dict:
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script], check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(result.stdout)


def test_right_arrow_takes_suggestion_only_into_an_empty_input():
    data = _run_module(
        """
import {
  bindSuggestionAccept, latestSuggestedReply, suggestionFromPlaceholder, SUGGESTED_REPLY_MAX,
} from "./marvis/static/js/composer-suggestion.js";
globalThis.Event = class { constructor(type) { this.type = type; } };
function fakeInput(value = "") {
  const handlers = {};
  return {
    value, events: [],
    addEventListener(type, fn) { handlers[type] = fn; },
    dispatchEvent(event) { this.events.push(event.type); },
    setSelectionRange() {},
    press(key, extra = {}) {
      let prevented = false;
      handlers.keydown({ key, isComposing: false, preventDefault() { prevented = true; }, ...extra });
      return prevented;
    },
  };
}
const empty = fakeInput();
bindSuggestionAccept(empty, () => "确认，按这个口径继续");
const took = empty.press("ArrowRight");
const typed = fakeInput("我自己写");
bindSuggestionAccept(typed, () => "确认");
const keptTyped = !typed.press("ArrowRight") && typed.value === "我自己写";
const none = fakeInput();
bindSuggestionAccept(none, () => "");
const noSuggestion = !none.press("ArrowRight") && none.value === "";
const modded = fakeInput();
bindSuggestionAccept(modded, () => "确认");
const shiftIgnored = !modded.press("ArrowRight", { shiftKey: true }) && modded.value === "";
process.stdout.write(JSON.stringify({
  took, value: empty.value, events: empty.events, keptTyped, noSuggestion, shiftIgnored,
  home: suggestionFromPlaceholder("例如：用这份样本做一个贷前 A 卡"),
  afterAgent: latestSuggestedReply([{ role: "user", content: "x" }, { role: "assistant", metadata: { suggested_reply: " 确认 " } }]),
  afterUser: latestSuggestedReply([{ role: "assistant", metadata: { suggested_reply: "确认" } }, { role: "user" }]),
  tooLong: latestSuggestedReply([{ role: "assistant", metadata: { suggested_reply: "长".repeat(SUGGESTED_REPLY_MAX + 1) } }]),
  missing: latestSuggestedReply([{ role: "assistant", metadata: {} }]),
}));
"""
    )
    assert data["took"] is True
    assert data["value"] == "确认，按这个口径继续"
    assert data["events"] == ["input"]
    assert data["keptTyped"] is True
    assert data["noSuggestion"] is True
    assert data["shiftIgnored"] is True
    assert data["home"] == "用这份样本做一个贷前 A 卡"
    assert data["afterAgent"] == "确认"
    assert data["afterUser"] == ""
    assert data["tooLong"] == ""
    assert data["missing"] == ""
