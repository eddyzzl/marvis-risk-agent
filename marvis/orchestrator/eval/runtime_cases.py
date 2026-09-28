"""Create a small, public, synthetic development suite; never hidden acceptance.

Usage: python -m marvis.orchestrator.eval.runtime_cases /new/suite-directory
Then use `marvis eval-agent --cases .../cases.json --expected .../private/expected.json
--dataset-root .../data --profile-workspace /explicit/workspace --model-id MODEL
--output-dir /new/evidence-directory`. This last command makes real model calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .runtime_contracts import digest


def write_synthetic_suite(
    root: Path,
    *,
    include_workflow_families: bool = False,
    normal_modeling_only: bool = False,
    normal_strategy_only: bool = False,
) -> dict[str, Path]:
    if sum((normal_modeling_only, normal_strategy_only, include_workflow_families)) > 1:
        raise ValueError(
            "normal suites are separate, not additions to the archived nine cases"
        )
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    data = root / "data"
    private = root / "private"
    data.mkdir()
    private.mkdir(mode=0o700)
    n = 80
    ids = [f"synthetic-customer-{i:04d}" for i in range(n)]
    frames = {
        "sample.parquet": (
            "sample",
            pd.DataFrame({"id": ids, "y": [i % 2 for i in range(n)]}),
        ),
        "features.parquet": (
            "feature",
            pd.DataFrame({"id": ids, "balance": list(range(n))}),
        ),
        "analysis.parquet": (
            "sample",
            pd.DataFrame(
                {
                    "y": [i % 2 for i in range(n)],
                    "balance": [i + (i % 2) * 20 for i in range(n)],
                    "age": [20 + i % 30 for i in range(n)],
                }
            ),
        ),
    }
    if include_workflow_families:
        from .runtime_family_cases import workflow_frames

        frames.update(workflow_frames())
    if normal_modeling_only:
        from .runtime_family_cases import normal_modeling_frames

        frames = normal_modeling_frames()
    if normal_strategy_only:
        from .runtime_family_cases import normal_strategy_frames

        frames = normal_strategy_frames()
    materials = {}
    for name, (role, frame) in frames.items():
        frame.to_parquet(data / name, index=False)
        materials[name] = {
            "path": name,
            "role": role,
            "sha256": hashlib.sha256((data / name).read_bytes()).hexdigest(),
            "source_kind": "synthetic",
        }
    if normal_modeling_only:
        from .runtime_family_cases import normal_modeling_cases

        cases, expected = normal_modeling_cases(materials)
        return _write_suite_files(root, private, data, cases, expected)
    if normal_strategy_only:
        from .runtime_family_cases import normal_strategy_cases

        cases, expected = normal_strategy_cases(materials)
        return _write_suite_files(root, private, data, cases, expected)
    cases = [
        {
            "id": "synthetic_join",
            "revision": "1",
            "family": "data_join",
            "case_set": "development",
            "scenario": "normal",
            "task": {"task_type": "data_join", "target_col": "y"},
            "materials": [materials["sample.parquet"], materials["features.parquet"]],
            "business_constraints_source": "MARVIS public synthetic fixture: 80 unique identical IDs, no dedup required",
            "actions": [
                {
                    "kind": "approve_step",
                    "tool": "data_ops.execute_join",
                    "content": "两份合成数据各有 80 个相同唯一 ID，批准最终拼接。",
                }
            ],
            "budget": {
                "wall_seconds": 180,
                "max_llm_attempts": 20,
                "max_http_requests": 120,
                "max_output_tokens_per_attempt": 2048,
            },
        },
        {
            "id": "synthetic_feature",
            "revision": "1",
            "family": "feature_analysis",
            "case_set": "development",
            "scenario": "normal",
            "task": {
                "task_type": "feature_analysis",
                "target_col": "y",
                "feature_columns": ["balance", "age"],
            },
            "materials": [materials["analysis.parquet"]],
            "business_constraints_source": "MARVIS public synthetic fixture: balance/age feature report, binary y",
            "actions": [
                {
                    "kind": "approve_step",
                    "tool": "feature.analyze_feature_bins",
                    "content": "保留预设分箱选择，生成这两个合成字段的特征分析报告。",
                }
            ],
            "budget": {
                "wall_seconds": 180,
                "max_llm_attempts": 20,
                "max_http_requests": 120,
                "max_output_tokens_per_attempt": 2048,
            },
        },
    ]
    expected = {
        "synthetic_join": {
            "result": "done",
            "assertions": [
                {"kind": "tool_succeeded", "tool": "data_ops.execute_join"},
                {"kind": "dataset_rows", "tool": "data_ops.execute_join", "value": n},
            ],
        },
        "synthetic_feature": {
            "result": "done",
            "assertions": [
                {"kind": "tool_succeeded", "tool": "feature.compute_feature_metrics"},
                {
                    "kind": "output_length",
                    "tool": "feature.compute_feature_metrics",
                    "path": ["metrics"],
                    "value": 2,
                },
                {"kind": "tool_succeeded", "tool": "feature.generate_feature_report"},
                {
                    "kind": "artifact_exists",
                    "tool": "feature.generate_feature_report",
                    "path": ["report_path"],
                },
            ],
        },
    }
    if include_workflow_families:
        from .runtime_family_cases import workflow_cases

        additional_cases, additional_expected = workflow_cases(materials)
        cases.extend(additional_cases)
        expected.update(additional_expected)
    return _write_suite_files(root, private, data, cases, expected)


def _write_suite_files(root, private, data, cases, expected):
    paths = {
        "cases": root / "cases.json",
        "expected": private / "expected.json",
        "dataset_root": data,
    }
    paths["cases"].write_text(
        json.dumps({"schema_version": 1, "cases": cases}, ensure_ascii=False, indent=2)
    )
    paths["expected"].write_text(
        json.dumps(
            {"schema_version": 1, "cases": expected}, ensure_ascii=False, indent=2
        )
    )
    paths["expected"].chmod(0o600)
    (root / "identities.json").write_text(
        json.dumps(
            {
                "cases_sha256": digest(paths["cases"].read_bytes()),
                "expected_sha256": digest(paths["expected"].read_bytes()),
                "source": "public_synthetic_development_only",
            },
            indent=2,
        )
    )
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--include-workflow-families", action="store_true")
    group.add_argument("--normal-modeling-only", action="store_true")
    group.add_argument("--normal-strategy-only", action="store_true")
    args = parser.parse_args()
    for name, path in write_synthetic_suite(
        args.directory,
        include_workflow_families=args.include_workflow_families,
        normal_modeling_only=args.normal_modeling_only,
        normal_strategy_only=args.normal_strategy_only,
    ).items():
        print(f"{name}={path}")
