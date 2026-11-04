"""Result charts (U13, 2026-10-19 user feedback).

Too many flat tables: a Vintage result now draws its cohort x MOB curves
above the cumulative table, and the migration matrix's declared matrix-heat
cells are shaded by their own rate. Both read the platform's numbers as
given; nothing is computed in the browser.
"""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "marvis" / "static"


def _node(script: str):
    result = subprocess.run(["node", "--input-type=module", "-e", script], cwd=ROOT, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def test_vintage_renderer_attaches_the_tools_curve_values():
    from marvis.agent.renderers import _render_vintage_curve

    _text, tables = _render_vintage_curve({
        "cohorts": ["2025-07", "2025-10"],
        "mob_axis": [0, 1, 2],
        "curves": {"2025-07": [0.0, 0.02, 0.06], "2025-10": [0.01, None, None]},
        "counts": {"2025-07": 50, "2025-10": 40},
        "summary": {"trend": "deteriorating"},
    })
    chart = tables[0]["chart"]
    assert chart == {
        "kind": "vintage_curves",
        "mob_axis": ["0", "1", "2"],
        "series": [
            {"cohort": "2025-07", "values": [0.0, 0.02, 0.06]},
            {"cohort": "2025-10", "values": [0.01, None, None]},
        ],
    }


def test_vintage_curves_card_draws_one_line_per_cohort_and_keeps_gaps():
    module = (STATIC / "js" / "metric-tables.js").as_uri()
    chart = {
        "kind": "vintage_curves",
        "mob_axis": ["0", "1", "2"],
        "series": [
            {"cohort": "2025-07", "values": [0.0, 0.02, 0.06]},
            {"cohort": "2025-10", "values": [0.01, None, 0.03]},
        ],
    }
    data = _node("\n".join([
        f"const m = await import({json.dumps(module)});",
        f"const html = m.renderVintageCurvesCard({json.dumps(chart)});",
        "const paths = [...html.matchAll(/class=\"vintage-line\" d=\"([^\"]*)\"/g)].map((x) => x[1]);",
        "process.stdout.write(JSON.stringify({",
        "  series: (html.match(/class=\"vintage-series\"/g) || []).length,",
        "  paths, legend: [...html.matchAll(/<b>([^<]*)<\\/b>/g)].map((x) => x[1]),",
        "  empty: m.renderVintageCurvesCard({ kind: 'vintage_curves', mob_axis: ['0'], series: [] }),",
        "}));",
    ]))
    assert data["series"] == 2
    # The young cohort's missing MOB1 breaks its line into two segments.
    assert data["paths"][0].count("M") == 1
    assert data["paths"][1].count("M") == 2
    assert data["legend"] == ["6%", "3%"]
    assert data["empty"] == ""


def test_chat_tables_shade_declared_matrix_heat_cells_and_open_charted_tables():
    from tests.test_frontend_result_cards import _tables_harness

    data = _tables_harness("""
const chart = {kind:'vintage_curves',mob_axis:['0','1'],series:[{cohort:'2025-07',values:[0,0.7]}]};
const html = agentMessageTablesHtml({id:'m',metadata:{tables:[
  {title:'概览',columns:['值'],rows:[[1]]},
  {title:'普通明细',columns:['值'],rows:[[2]]},
  {title:'曲线证据',columns:['迁徙率'],column_specs:[{kind:'matrix-heat'}],rows:[[0.7]],chart}
]}});
process.stdout.write(JSON.stringify({html,
  low:driverTableCellHtml(-1,0,0,'迁徙率','matrix-heat',new Map()),
  high:driverTableCellHtml(2,0,0,'迁徙率','matrix-heat',new Map()),
  empty:driverTableCellHtml(null,0,0,'迁徙率','matrix-heat',new Map())}));
""")
    html = data['html']
    assert 'vintage-line' in html and '--heat:0.7000' in html
    assert html.index('曲线证据') < html.index('<details'), 'A chart remains open ahead of folded detail tables'
    assert '--heat:0.0000' in data['low']['html']
    assert '--heat:1.0000' in data['high']['html']
    assert data['empty']['cls'] == 'cell-matrix-heat is-empty'
