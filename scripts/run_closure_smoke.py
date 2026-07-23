#!/usr/bin/env python3
"""Run the three DFM closure flows in isolated fresh pytest workspaces.

The selected HTTP tests instantiate ``create_app(tmp_path)`` and therefore use a
new SQLite database, datasets directory and task directory on every run.  No
port is opened and no currently running MARVIS service is touched.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FLOW_SELECTORS = {
    "data_join": (
        "tests/test_data_join_api.py::"
        "test_data_join_conversation_end_to_end"
    ),
    "feature_analysis": (
        "tests/test_feature_analysis_api.py::"
        "test_feature_analysis_end_to_end"
    ),
    "modeling": (
        "tests/test_modeling_api.py::"
        "test_modeling_business_materials_flow_into_report_and_delivery"
    ),
}


def _git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def run(
    repo_root: Path,
    *,
    python: str = sys.executable,
    flows: list[str] | None = None,
    temp_root: Path | None = None,
) -> dict[str, Any]:
    selected = flows or list(FLOW_SELECTORS)
    unknown = sorted(set(selected) - set(FLOW_SELECTORS))
    if unknown:
        raise ValueError(f"unknown flows: {unknown}")

    if temp_root is None:
        temp_root = Path(tempfile.mkdtemp(prefix="marvis-closure-smoke."))
    temp_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)
    results: list[dict[str, Any]] = []

    for flow in selected:
        basetemp = temp_root / flow
        command = [
            python,
            "-m",
            "pytest",
            "-q",
            "--basetemp",
            str(basetemp),
            FLOW_SELECTORS[flow],
        ]
        started = time.monotonic()
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        results.append(
            {
                "flow": flow,
                "selector": FLOW_SELECTORS[flow],
                "status": "PASS" if completed.returncode == 0 else "FAIL",
                "exit_code": completed.returncode,
                "duration_seconds": round(time.monotonic() - started, 3),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "fresh_workspace_root": str(basetemp),
            }
        )

    return {
        "schema_version": "closure-smoke.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "repo_root": str(repo_root),
        "git_head": _git_head(repo_root),
        "temp_root": str(temp_root),
        "status": "PASS" if all(item["status"] == "PASS" for item in results) else "FAIL",
        "results": results,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# 数据处理 / 特征分析 / 模型开发 Fresh-workspace Smoke",
        "",
        f"- 生成时间：`{result['generated_at']}`",
        f"- Git HEAD：`{result['git_head']}`",
        f"- 临时根目录：`{result['temp_root']}`",
        f"- 总体：**{result['status']}**",
        "",
        "| 流程 | 状态 | 用时（秒） | fresh workspace | 测试入口 |",
        "|---|---|---:|---|---|",
    ]
    for item in result["results"]:
        lines.append(
            "| {flow} | {status} | {duration_seconds} | `{fresh_workspace_root}` | "
            "`{selector}` |".format(**item)
        )
    lines.extend(["", "## 原始 pytest 输出", ""])
    for item in result["results"]:
        lines.extend(
            [
                f"### {item['flow']}",
                "",
                "```text",
                (item["stdout"] + item["stderr"]).strip(),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--flow",
        dest="flows",
        action="append",
        choices=sorted(FLOW_SELECTORS),
        help="flow to run (repeatable; default: all three)",
    )
    parser.add_argument("--temp-root", default=None)
    parser.add_argument("--output", default=None, help="optional Markdown report")
    parser.add_argument("--json-output", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(
            Path(args.repo_root).resolve(),
            python=args.python,
            flows=args.flows,
            temp_root=Path(args.temp_root) if args.temp_root else None,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(render_markdown(result), encoding="utf-8")
    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
