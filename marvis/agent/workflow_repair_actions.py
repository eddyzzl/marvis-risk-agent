"""Deterministic data executor for the "改目标定义" one-click action.

Spec: docs/superpowers/specs/2026-10-23-self-repair-extension-spec.md §5.3.

The source dataset and the uploaded file are never modified. A derived
dataset copies every column and replaces only the target column with 0/1
(null stays null), so no raw target value can become a candidate feature. It
is registered together with a platform audit row (who, rule, counts).
"""

from __future__ import annotations

from pathlib import Path
import uuid

import numpy as np
import pandas as pd


def _binary_target(values: pd.Series, rule: dict) -> pd.Series:
    kind = str(rule.get("kind") or "")
    if kind == "ge":
        numeric = pd.to_numeric(values, errors="coerce")
        result = (numeric >= float(rule["threshold"])).astype(float)
        return result.where(numeric.notna(), np.nan)
    if kind == "equals":
        text = values.astype("string")
        result = (text == str(rule["bad_value"])).astype(float)
        return result.where(values.notna(), np.nan)
    raise ValueError(f"unsupported target rule {kind!r}")


def rule_text(target_col: str, rule: dict) -> str:
    if rule.get("kind") == "ge":
        return f"{target_col} ≥ {rule['threshold']} 记为坏（1），其余记为好（0）"
    return f"{target_col} 为 '{rule['bad_value']}' 记为坏（1），其余记为好（0）"


def derive_binary_target(*, settings, task_id: str, dataset_id: str, target_col: str,
                         rule: dict, actor: dict) -> dict:
    """Write and register a derived dataset; returns a verified record or the reason."""

    from marvis.data.backend import DataBackend
    from marvis.data.registry import DatasetRegistry
    from marvis.repositories.datasets import DatasetRepository

    registry = DatasetRegistry(
        DatasetRepository(settings.db_path), DataBackend(settings.datasets_dir), settings.datasets_dir
    )
    source = registry.get(dataset_id)
    if str(source.task_id) != str(task_id):
        raise ValueError("dataset does not belong to this task")
    frame = registry.read_authenticated_parquet_snapshot(source.id)
    if target_col not in frame.columns:
        raise ValueError(f"target column {target_col!r} is missing")
    derived = frame.copy()
    derived[target_col] = _binary_target(frame[target_col], rule)
    labels = derived[target_col].dropna()
    bad_rows = int((labels == 1).sum())
    good_rows = int((labels == 0).sum())
    text = rule_text(target_col, rule)
    log = [f"派生规则：{text}"]
    if not bad_rows or not good_rows or set(labels.unique().tolist()) - {0.0, 1.0}:
        log.append(f"核对没通过：派生后坏样本 {bad_rows} 行、好样本 {good_rows} 行，不能作为二分类目标。")
        return {"verified": False, "log": log, "dataset_id": None, "bad_rows": bad_rows, "changed": text}
    if not derived[target_col].isna().any():
        derived[target_col] = derived[target_col].astype("int64")
    out_dir = Path(settings.datasets_dir) / str(task_id) / "repairs"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{source.id}_target_{uuid.uuid4().hex[:12]}.parquet"
    derived.to_parquet(path, index=False)

    def audit(dataset) -> dict:
        return {
            "kind": "workflow.repair.redefine_target",
            "target_ref": dataset.id,
            "actor": str(actor.get("label") or actor.get("kind") or "user"),
            "outcome": "succeeded",
            "detail": {
                "task_id": str(task_id),
                "source_dataset_id": source.id,
                "source_content_hash": source.content_hash,
                "target_col": target_col,
                "rule": dict(rule),
                "rows": int(len(derived)),
                "bad_rows": bad_rows,
                "good_rows": good_rows,
                "null_rows": int(derived[target_col].isna().sum()),
            },
        }

    dataset = registry.register_existing_with_audit(
        path, audit_factory=audit, task_id=str(task_id), role=str(source.role), anchor_target=source.id,
    )
    rate = bad_rows / max(1, bad_rows + good_rows)
    log.append(f"核对：新数据集目标只有 0/1，坏样本 {bad_rows} 行、好样本 {good_rows} 行（坏率 {rate:.1%}）。")
    return {
        "verified": True,
        "log": log,
        "dataset_id": dataset.id,
        "bad_rows": bad_rows,
        "changed": f"派生数据集 {dataset.id}：{text}（坏率 {rate:.1%}），原数据集 {source.id} 不变",
    }


__all__ = ["derive_binary_target", "rule_text"]
