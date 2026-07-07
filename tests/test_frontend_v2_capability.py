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

        assert.ok(html.includes('class="tier-settings-head"'));
        assert.ok(html.includes('class="tier-settings-list"'));
        assert.ok(html.includes("能力档位"));
        assert.ok(html.includes("选择 Agent 默认自治预算"));
        assert.ok(html.includes("安全护栏保持一致"));
        assert.ok(html.includes("均衡"));
        assert.ok(html.includes("autonomy_level"));
        assert.ok(html.includes("最大重规划"));
        """
    )


def test_tier_settings_render_selectable_radios_with_selected_checked():
    run_node(
        """
        import assert from "node:assert/strict";
        import { tierSettingsHtml } from "./marvis/static/js/v2/capability.js";

        const html = tierSettingsHtml({
          default: "balanced",
          selected: "autonomous",
          tiers: [
            { name: "balanced", summary: "Default" },
            { name: "autonomous", summary: "Higher autonomy" },
          ],
        });

        assert.ok(html.includes('name="capabilityTier"'));
        assert.ok(html.includes('value="autonomous" checked'));
        assert.equal(html.includes('value="balanced" checked'), false);
        """
    )


def test_tier_settings_falls_back_to_default_when_no_selected():
    run_node(
        """
        import assert from "node:assert/strict";
        import { tierSettingsHtml } from "./marvis/static/js/v2/capability.js";

        const html = tierSettingsHtml({
          default: "balanced",
          tiers: [
            { name: "balanced", summary: "Default" },
            { name: "autonomous", summary: "Higher autonomy" },
          ],
        });

        assert.ok(html.includes('value="balanced" checked'));
        assert.equal(html.includes('value="autonomous" checked'), false);
        """
    )


def test_capability_handlers_persist_radio_selection():
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
        attachCapabilityHandlers(root, {
          storage: { setItem(key, value) { stored[key] = value; } },
        });
        const target = {
          value: "conservative",
          closest(selector) {
            return selector === 'input[name="capabilityTier"]' ? this : null;
          },
        };

        await listeners.change({ target });
        assert.equal(getSelectedTier(), "conservative");
        assert.equal(stored.marvis_v2_selected_tier, "conservative");
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


def test_render_tier_settings_restores_persisted_selected_tier():
    run_node(
        """
        import assert from "node:assert/strict";
        import { renderTierSettings } from "./marvis/static/js/v2/capability.js";
        import { getSelectedTier, resetV2State } from "./marvis/static/js/v2/state_v2.js";

        resetV2State();
        const container = { innerHTML: "", dataset: {} };
        await renderTierSettings(container, {
          storage: {
            getItem(key) {
              assert.equal(key, "marvis_v2_selected_tier");
              return "autonomous";
            },
          },
          listCapabilityTiers: async () => ({
            default: "balanced",
            tiers: [
              { name: "balanced", summary: "Default" },
              { name: "autonomous", summary: "Higher autonomy" },
            ],
          }),
        });

        assert.equal(getSelectedTier(), "autonomous");
        assert.ok(container.innerHTML.includes('value="autonomous" checked'));
        """
    )


def test_render_tier_settings_ignores_unknown_persisted_tier():
    run_node(
        """
        import assert from "node:assert/strict";
        import { renderTierSettings } from "./marvis/static/js/v2/capability.js";
        import { getSelectedTier, resetV2State } from "./marvis/static/js/v2/state_v2.js";

        resetV2State();
        await renderTierSettings({ innerHTML: "", dataset: {} }, {
          storage: { getItem: () => "removed-tier" },
          listCapabilityTiers: async () => ({
            default: "balanced",
            tiers: [
              { name: "balanced", summary: "Default" },
              { name: "autonomous", summary: "Higher autonomy" },
            ],
          }),
        });

        assert.equal(getSelectedTier(), "balanced");
        """
    )
