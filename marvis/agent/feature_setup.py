"""Setup (slot-filling) for the feature_analysis task.

Standalone 特征分析 (spec §1 form A) takes a single dataset (a joined sample or a
plain csv that already carries a target + features) and computes the selected
per-feature metrics — no screening gate, the wide table IS the report. This module
discovers/registers that dataset and proposes the target column + candidate
numeric features, reusing the same deterministic detection as the modeling setup.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from marvis.agent.data_dictionary import resolve_data_dictionary_id
from marvis.agent.join_setup import propose_roles
from marvis.agent.sample_setup import detect_setup
from marvis.domain import FileRole
from marvis.files import scan_source_dir

_DATA_ROLES = frozenset({FileRole.SAMPLE.value, "sample", "feature"})


class FeatureSetupError(ValueError):
    """Raised when the task has no analysable dataset."""


@dataclass
class FeatureProposal:
    dataset_id: str
    dataset_name: str
    target_col: str
    features: list[str]
    notes: list[str]
    metrics: list[str]
    template_id: str = "feature_analysis"
    anchor_id: str | None = None
    feature_ids: list[str] | None = None

    def template_slots(self) -> dict:
        if self.template_id == "feature_analysis_with_join":
            return {
                "anchor_id": self.anchor_id or self.dataset_id,
                "feature_ids": list(self.feature_ids or []),
                "target_col": self.target_col,
                "features": [],
                "metrics": self.metrics,
            }
        return {
            "dataset_id": self.dataset_id,
            "target_col": self.target_col,
            "features": self.features,
            "metrics": self.metrics,
        }


def build_feature_proposal(
    registry, backend, task_id: str, source_dir, *, metrics=None
) -> FeatureProposal:
    datasets = _resolve_datasets(registry, task_id, source_dir)
    # GAP-4: register a data-dictionary material (if present) as a dataset, same
    # detection the modeling setup flow already does. Best-effort/side-effect
    # only — never blocks feature-analysis setup when no dictionary exists.
    resolve_data_dictionary_id(registry, task_id, source_dir)
    joined = len(datasets) > 1
    if joined:
        ranked = propose_roles(datasets)
        dataset = ranked[0]
        feature_ids = [item.id for item in ranked[1:]]
    else:
        dataset = datasets[0]
        feature_ids = []
    path = registry.resolve_path(dataset.id)
    setup = detect_setup(backend, path)
    if not setup.target_col:
        raise FeatureSetupError(
            "未能在数据中识别 0/1 目标列；请确认数据含标签列后重试。"
        )
    # metrics = optional metrics the user selected at creation (spec §2: 选了才算);
    # empty → base per-feature metrics only.
    selected = [str(item).strip() for item in (metrics or []) if str(item).strip()]
    return FeatureProposal(
        dataset_id=dataset.id,
        dataset_name=_dataset_name(dataset),
        target_col=setup.target_col,
        features=list(setup.candidates),
        notes=list(setup.notes),
        metrics=selected,
        template_id="feature_analysis_with_join" if joined else "feature_analysis",
        anchor_id=dataset.id if joined else None,
        feature_ids=feature_ids if joined else None,
    )


def _resolve_datasets(registry, task_id: str, source_dir):
    datasets = [d for d in registry.list_for_task(task_id) if d.role in _DATA_ROLES]
    if not datasets and source_dir is not None:
        for artifact in scan_source_dir(Path(source_dir)):
            if artifact.role == FileRole.SAMPLE:
                registry.register_from_upload(task_id, Path(artifact.path), role="sample")
        datasets = [d for d in registry.list_for_task(task_id) if d.role in _DATA_ROLES]
    if not datasets:
        raise FeatureSetupError(f"特征分析未找到数据文件:{source_dir}")
    # Prefer a target-carrying dataset, else the largest. For multiple files this
    # same order becomes anchor + feature tables for the JOIN-composed template.
    return sorted(
        datasets,
        key=lambda d: (not bool(getattr(d, "has_target", False)), -int(getattr(d, "row_count", 0) or 0)),
    )


def _dataset_name(dataset) -> str:
    source = getattr(dataset, "source_path", None)
    return Path(source).name if source else str(getattr(dataset, "id", ""))


__all__ = ["build_feature_proposal", "FeatureProposal", "FeatureSetupError"]
