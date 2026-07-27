from __future__ import annotations

import json
import subprocess
from pathlib import Path


STATIC = Path(__file__).resolve().parents[1] / "marvis" / "static"


def test_feature_binning_gate_renders_optional_multiselect_and_bin_count():
    module_url = (STATIC / "js" / "v2" / "feature_binning_gate.js").as_uri()
    script = "\n".join([
        f"import {{ renderFeatureBinningGate }} from {json.dumps(module_url)};",
        "const html = renderFeatureBinningGate({metadata:{step_id:'step-bin',feature_binning:{",
        "  features:[{feature:'x1',recommendation:'推荐',recommendation_reason:'区分力较好'},{feature:'x2'}],",
        "  default_bins:10,min_bins:3,max_bins:20",
        "}}});",
        "if (!html.includes('data-feature-binning-pick')) throw new Error('missing multiselect');",
        "if (!html.includes('data-feature-binning-count')) throw new Error('missing bin count');",
        "if (!html.includes('跳过分箱并生成报告')) throw new Error('missing skip action');",
        "if (!html.includes('分析所选特征并生成报告')) throw new Error('missing selected action');",
        "if (!html.includes('x1') || !html.includes('推荐')) throw new Error('missing candidate context');",
        "const readonly = renderFeatureBinningGate({metadata:{feature_binning:{features:[{feature:'x1'}]}}}, {interactive:false});",
        "if (!readonly.includes(' disabled')) throw new Error('agent evidence must be read-only');",
    ])
    subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )


def test_feature_binning_gate_is_wired_to_shared_manual_and_agent_surfaces():
    app = (STATIC / "app.js").read_text(encoding="utf-8")
    manual = (STATIC / "js" / "v2" / "driver_manual_analysis.js").read_text(encoding="utf-8")
    component = (STATIC / "js" / "v2" / "feature_binning_gate.js").read_text(encoding="utf-8")
    assert "renderFeatureBinning: agentMessageFeatureBinningHtml" in app
    assert 'document.addEventListener("click", handleFeatureBinningClick)' in app
    assert "meta.feature_binning" in manual
    assert 'ui_action: "confirm_feature_binning"' in component
    assert 'adjust_params: { features, bins }' in component
