"""Plan rail phase names for plans saved without them (eval S9: the whole
portfolio and Vintage rail read 「步骤」)."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = (ROOT / "marvis" / "static" / "js" / "v2" / "plan_rail_controller.js").as_uri()


def test_phaseless_steps_take_a_phase_from_their_tool():
    script = "\n".join([
        f"const m = await import({json.dumps(MODULE)});",
        "const steps = (tools) => ({ steps: tools.map((tool, index) => ({ id: `s${index}`, index, status: 'succeeded', tool_ref: { tool } })) });",
        "process.stdout.write(JSON.stringify([",
        "  m.planRailPhaseRows(steps(['flow_rate', 'bucket_migration', 'segment_profile', 'el_estimate', 'portfolio_gate_summary', 'generate_portfolio_report'])).map((g) => [g.phase, g.steps.length]),",
        "  m.planRailPhaseRows(steps(['vintage_curve'])).map((g) => g.phase),",
        "  m.planRailPhaseRows({ steps: [{ id: 'x', index: 0, phase: '拼接', tool_ref: { tool: 'execute_join' } }] }).map((g) => g.phase),",
        "  m.planRailPhaseRows(steps(['something_new'])).map((g) => g.phase),",
        "]));",
    ])
    result = subprocess.run(["node", "--input-type=module", "-e", script], cwd=ROOT, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == [
        [["组合分析", 5], ["报告", 1]],
        ["风险分析"],
        ["拼接"],  # a stored phase always wins
        ["步骤"],
    ]
