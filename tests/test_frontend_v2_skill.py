from __future__ import annotations

from pathlib import Path
import subprocess
import textwrap


ROOT = Path(__file__).resolve().parents[1]


def run_node(script: str) -> None:
    subprocess.run(
        ["node", "--input-type=module", "-e", textwrap.dedent(script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def test_skill_manager_html_renders_states_and_escapes_rejected_problems():
    run_node(
        """
        import assert from "node:assert/strict";
        import { skillManagerHtml } from "./marvis/static/js/v2/skill_manager.js";

        const html = skillManagerHtml({
          active: ["safe_skill"],
          disabled: ["old_skill"],
          rejected: [["bad<script>", ["problem <img onerror=alert(1)>"]]],
        });

        assert.ok(html.includes('id="reloadSkills"'));
        assert.ok(html.includes("skill-active"));
        assert.ok(html.includes("safe_skill"));
        assert.ok(html.includes("skill-disabled"));
        assert.ok(html.includes("old_skill"));
        assert.ok(html.includes("skill-rejected"));
        assert.equal(html.includes("bad<script>"), false);
        assert.equal(html.includes("<img onerror"), false);
        assert.ok(html.includes("bad&lt;script&gt;"));
        assert.ok(html.includes("problem &lt;img onerror=alert(1)&gt;"));
        assert.ok(html.includes("data-validate-skill"));
        """
    )


def test_skill_handlers_reload_validate_and_report_local_json_errors():
    run_node(
        """
        import assert from "node:assert/strict";
        import { attachSkillHandlers } from "./marvis/static/js/v2/skill_manager.js";

        const calls = [];
        const resultSlot = { innerHTML: "" };
        const listeners = {};
        const root = {
          addEventListener(type, fn) { listeners[type] = fn; },
          removeEventListener(type, fn) {
            if (listeners[type] === fn) delete listeners[type];
          },
          querySelector(selector) {
            return selector === "[data-skill-validation-result]" ? resultSlot : null;
          },
        };

        const detach = attachSkillHandlers(root, {
          reloadSkills: async () => calls.push(["reloadSkills"]),
          refreshSkills: async () => calls.push(["refreshSkills"]),
          validateSkill: async (skill) => {
            calls.push(["validateSkill", skill]);
            return { problems: [] };
          },
        });

        const reloadTarget = {
          closest(selector) {
            return selector === "#reloadSkills" ? this : null;
          },
        };
        await listeners.click({ target: reloadTarget, preventDefault() {} });
        assert.deepEqual(calls.splice(0), [["reloadSkills"], ["refreshSkills"]]);

        const validTarget = {
          value: '{"id":"preview_echo"}',
          closest(selector) {
            return selector === "[data-validate-skill]" ? this : null;
          },
        };
        await listeners.input({ target: validTarget });
        assert.deepEqual(calls.splice(0), [["validateSkill", { id: "preview_echo" }]]);
        assert.ok(resultSlot.innerHTML.includes("Valid skill"));

        const invalidTarget = {
          value: '{"id":',
          closest(selector) {
            return selector === "[data-validate-skill]" ? this : null;
          },
        };
        await listeners.input({ target: invalidTarget });
        assert.deepEqual(calls, []);
        assert.ok(resultSlot.innerHTML.includes("Invalid JSON"));

        detach();
        assert.equal(listeners.click, undefined);
        assert.equal(listeners.input, undefined);
        """
    )


def test_skill_validation_result_escapes_backend_problem_text():
    run_node(
        """
        import assert from "node:assert/strict";
        import { skillValidationResultHtml } from "./marvis/static/js/v2/skill_manager.js";

        const html = skillValidationResultHtml({
          problems: ["bad <img onerror=alert(1)>"],
        });

        assert.equal(html.includes("<img onerror"), false);
        assert.ok(html.includes("bad &lt;img onerror=alert(1)&gt;"));
        """
    )
