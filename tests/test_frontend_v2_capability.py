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


def test_capability_select_html_uses_backend_tiers_and_escapes_summary():
    run_node(
        """
        import assert from "node:assert/strict";
        import { capabilitySelectHtml } from "./marvis/static/js/v2/capability.js";

        const html = await capabilitySelectHtml({
          listCapabilityTiers: async () => ({
            default: "balanced",
            tiers: [
              { name: "conservative", summary: "Guarded <mode>" },
              { name: "balanced", summary: "Default" },
            ],
          }),
        });

        assert.ok(html.includes('id="tierSelect"'));
        assert.ok(html.includes('value="balanced" selected'));
        assert.equal(html.includes("Guarded <mode>"), false);
        assert.ok(html.includes("Guarded &lt;mode&gt;"));
        """
    )


def test_tier_settings_show_guardrail_constant_note_and_tier_limits():
    run_node(
        """
        import assert from "node:assert/strict";
        import { tierSettingsHtml } from "./marvis/static/js/v2/capability.js";

        const html = tierSettingsHtml({
          default: "balanced",
          tiers: [
            { name: "balanced", summary: "Default", autonomy_level: 2, max_replans: 3 },
          ],
        });

        assert.ok(html.includes("Guardrails remain constant"));
        assert.ok(html.includes("balanced"));
        assert.ok(html.includes("autonomy_level"));
        assert.ok(html.includes("max_replans"));
        """
    )


def test_capability_handlers_update_state_and_storage():
    run_node(
        """
        import assert from "node:assert/strict";
        import { attachCapabilityHandlers } from "./marvis/static/js/v2/capability.js";
        import { getSelectedTier, resetV2State } from "./marvis/static/js/v2/state_v2.js";

        resetV2State();
        const stored = {};
        const listeners = {};
        const root = {
          addEventListener(type, fn) { listeners[type] = fn; },
          removeEventListener(type, fn) {
            if (listeners[type] === fn) delete listeners[type];
          },
        };
        const detach = attachCapabilityHandlers(root, {
          storage: {
            setItem(key, value) { stored[key] = value; },
          },
        });
        const target = {
          value: "autonomous",
          closest(selector) {
            return selector === "#tierSelect" ? this : null;
          },
        };

        await listeners.change({ target });
        assert.equal(getSelectedTier(), "autonomous");
        assert.equal(stored.marvis_v2_selected_tier, "autonomous");

        detach();
        assert.equal(listeners.change, undefined);
        """
    )
