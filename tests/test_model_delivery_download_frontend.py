from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent


def _run_node(script: str) -> str:
    result = subprocess.run(
        ["node", "--input-type=module", "-e", textwrap.dedent(script)],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_model_delivery_panel_offers_one_safe_download_per_primary_artifact():
    output = _run_node(
        """
        import assert from "node:assert/strict";
        import { renderModelDeliveryPanel } from "./marvis/static/js/v2/model_delivery_panel.js";

        const taskId = "task-42";
        const artifactRoot = `/Users/test/workspace/tasks/${taskId}/modeling_artifacts`;
        const html = renderModelDeliveryPanel({
          task_id: taskId,
          metadata: {
            model_delivery: {
              source_tool: "post_training_action",
              native_model_path: "artifact-1.pkl",
              pmml_path: `${artifactRoot}/artifact-1.pmml`,
              model_card_path: `${artifactRoot}/artifact-1.model_card.json`,
              model_card_markdown_path: `${artifactRoot}/artifact-1.model_card.md`,
              approval_package_path: `${artifactRoot}/artifact-1.approval_package.json`,
              approval_package_markdown_path: `${artifactRoot}/artifact-1.approval_package.md`,
            },
          },
        });

        const expected = {
          native_model: "/api/artifacts/tasks%2Ftask-42%2Fmodeling_artifacts%2Fartifact-1.pkl",
          pmml: "/api/artifacts/tasks%2Ftask-42%2Fmodeling_artifacts%2Fartifact-1.pmml",
          model_card: "/api/artifacts/tasks%2Ftask-42%2Fmodeling_artifacts%2Fartifact-1.model_card.md",
          approval_package: "/api/artifacts/tasks%2Ftask-42%2Fmodeling_artifacts%2Fartifact-1.approval_package.md",
        };
        for (const [kind, href] of Object.entries(expected)) {
          const marker = `data-model-delivery-download="${kind}"`;
          assert.equal(html.split(marker).length - 1, 1, `${kind} should have one download`);
          assert.equal(html.includes(`href="${href}"`), true, href);
        }
        assert.equal(html.includes("下载原生模型"), true);
        assert.equal(html.includes("下载 PMML"), true);
        assert.equal(html.includes("下载模型卡"), true);
        assert.equal(html.includes("下载审批包"), true);
        assert.equal(html.includes('data-model-delivery-download="approval_package_json"'), false);

        const unsafeHtml = renderModelDeliveryPanel({
          task_id: taskId,
          metadata: {
            model_delivery: {
              source_tool: "post_training_action",
              model_card_markdown_path: "../../secret.txt",
            },
          },
        });
        assert.equal(unsafeHtml.includes('data-model-delivery-download="model_card"'), false);
        assert.equal(unsafeHtml.includes("..%2F..%2Fsecret.txt"), false);
        process.stdout.write("ok");
        """
    )

    assert output == "ok"
