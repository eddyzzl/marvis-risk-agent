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


def test_loop_events_html_sorts_events_and_escapes_reasons():
    run_node(
        """
        import assert from "node:assert/strict";
        import { loopEventsHtml } from "./marvis/static/js/v2/loop_progress.js";

        const html = loopEventsHtml([
          { type: "no_progress", reason: "stuck <bad>", at: "2026-06-20T00:03:00Z" },
          { type: "replan", reason: "fan-out", at: "2026-06-20T00:01:00Z" },
          { type: "explore_segment", reason: "next segment", at: "2026-06-20T00:02:00Z" },
        ]);

        assert.ok(html.indexOf("fan-out") < html.indexOf("next segment"));
        assert.ok(html.indexOf("next segment") < html.indexOf("stuck &lt;bad&gt;"));
        assert.equal(html.includes("<bad>"), false);
        assert.ok(html.includes("loop-evt replan"));
        assert.ok(html.includes("loop-evt explore_segment"));
        assert.ok(html.includes("loop-evt no_progress attention"));
        """
    )


def test_render_loop_events_updates_from_state_and_empty_state():
    run_node(
        """
        import assert from "node:assert/strict";
        import { renderLoopEvents } from "./marvis/static/js/v2/loop_progress.js";
        import { resetV2State, setLoopEvents } from "./marvis/static/js/v2/state_v2.js";

        resetV2State();
        const container = { innerHTML: "", dataset: {} };
        const unsubscribe = renderLoopEvents(container);
        assert.ok(container.innerHTML.includes('data-v2-empty="loop-events"'));
        assert.equal(container.dataset.v2LoopEvents, "true");

        setLoopEvents([{ type: "replan", reason: "decision", at: "now" }]);
        assert.ok(container.innerHTML.includes("decision"));
        assert.ok(container.innerHTML.includes("loop-evt replan"));

        const rendered = container.innerHTML;
        unsubscribe();
        setLoopEvents([{ type: "no_progress", reason: "ignored", at: "later" }]);
        assert.equal(container.innerHTML, rendered);
        """
    )


def test_render_loop_events_updates_from_plan_loop_events():
    run_node(
        """
        import assert from "node:assert/strict";
        import { renderLoopEvents } from "./marvis/static/js/v2/loop_progress.js";
        import { resetV2State, setLoopEvents, setPlan } from "./marvis/static/js/v2/state_v2.js";

        resetV2State();
        const container = { innerHTML: "", dataset: {} };
        const unsubscribe = renderLoopEvents(container);

        setLoopEvents([{ type: "replan", reason: "stale state", at: "2026-06-20T00:02:00Z" }]);
        assert.ok(container.innerHTML.includes("stale state"));

        setPlan({
          id: "plan-1",
          loop_events: [
            { type: "explore_segment", reason: "explore", at: "2026-06-20T00:01:00Z" },
            { type: "no_progress", reason: "failure", at: "2026-06-20T00:03:00Z" },
          ],
        });
        assert.ok(container.innerHTML.includes("探索分支：explore"));
        assert.ok(container.innerHTML.includes("暂无进展：failure"));
        assert.equal(container.innerHTML.includes("stale state"), false);

        setPlan({ id: "plan-2", loop_events: [] });
        assert.ok(container.innerHTML.includes('data-v2-empty="loop-events"'));
        assert.equal(container.innerHTML.includes("stale state"), false);

        unsubscribe();
        setPlan({ id: "plan-3", loop_events: [{ type: "replan", reason: "ignored", at: "later" }] });
        assert.equal(container.innerHTML.includes("ignored"), false);
        """
    )
