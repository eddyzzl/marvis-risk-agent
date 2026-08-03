from pathlib import Path
import subprocess
import textwrap


ROOT = Path(__file__).resolve().parents[1]


def test_current_task_render_signature_tracks_live_plan_status() -> None:
    app_js = (ROOT / "marvis/static/app.js").read_text(encoding="utf-8")
    start = app_js.index("function currentTaskSignature")
    end = app_js.index("function stepFingerprint", start)
    function_source = app_js[start:end]
    script = textwrap.dedent(
        f"""
        import assert from "node:assert/strict";

        function signatureFromParts(parts) {{
          return JSON.stringify(parts);
        }}

        let planStatus = {{ label: "已完成", tone: "success" }};
        function taskStatusLabel() {{
          return planStatus.label;
        }}
        function taskStatusTone() {{
          return planStatus.tone;
        }}
        function taskStopped() {{
          return false;
        }}

        {function_source}

        const task = {{
          id: "strategy-task",
          name: "Strategy",
          status: "created",
          workflow_status: null,
          status_message: "创建不可变交互式树修订已提交。",
        }};
        const completedSignature = currentTaskSignature(task);
        planStatus = {{ label: "失败", tone: "danger" }};
        const failedSignature = currentTaskSignature(task);

        assert.notEqual(
          completedSignature,
          failedSignature,
          "plan/step failure must invalidate the hero render guard",
        );
        """
    )

    subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
