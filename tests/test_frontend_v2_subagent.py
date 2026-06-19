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


def test_subagent_row_html_escapes_scope_tools_and_result_ref():
    run_node(
        """
        import assert from "node:assert/strict";
        import { subAgentRowHtml } from "./marvis/static/js/v2/subagent_view.js";

        const html = subAgentRowHtml({
          scope: "Investigate <segment>",
          status: "running",
          granted_tools: [
            { plugin: "data<script>", tool: "profile" },
          ],
          result_ref: "artifact:report<script>",
        });

        assert.equal(html.includes("<script>"), false);
        assert.equal(html.includes("<segment>"), false);
        assert.ok(html.includes("Investigate &lt;segment&gt;"));
        assert.ok(html.includes("data&lt;script&gt;.profile"));
        assert.ok(html.includes("subagent-status-running"));
        assert.ok(html.includes('data-artifact="artifact:report&lt;script&gt;"'));
        """
    )


def test_render_subagent_view_updates_from_plan_state_and_empty_state():
    run_node(
        """
        import assert from "node:assert/strict";
        import { renderSubAgentView } from "./marvis/static/js/v2/subagent_view.js";
        import { resetV2State, setPlan } from "./marvis/static/js/v2/state_v2.js";

        resetV2State();
        const container = { innerHTML: "", dataset: {} };
        const unsubscribe = renderSubAgentView(container);
        assert.ok(container.innerHTML.includes('data-v2-empty="subagents"'));
        assert.equal(container.dataset.v2SubAgentView, "true");

        setPlan({
          id: "plan-1",
          sub_agents: [
            {
              scope: "Review join",
              status: "returned",
              granted_tools: [{ plugin: "data_ops", tool: "join" }],
              result_ref: "dataset:joined",
            },
          ],
        });
        assert.ok(container.innerHTML.includes("Review join"));
        assert.ok(container.innerHTML.includes("data_ops.join"));
        assert.ok(container.innerHTML.includes("subagent-status-returned"));

        const rendered = container.innerHTML;
        unsubscribe();
        setPlan({ id: "plan-2", sub_agents: [] });
        assert.equal(container.innerHTML, rendered);
        """
    )
