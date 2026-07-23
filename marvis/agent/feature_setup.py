"""Setup (slot-filling) for the feature_analysis task.

Standalone 特征分析 (spec §1 form A) takes a single dataset (a joined sample or a
plain csv that already carries a target + features) and computes the selected
per-feature metrics — no screening gate, the wide table IS the report. This module
discovers/registers that dataset and proposes the target column + candidate
numeric features, reusing the same deterministic detection as the modeling setup.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from marvis.agent.data_setup import reconcile_source_data_tables
from marvis.agent.json_reply import load_json_object
from marvis.data.data_dictionary import load_business_names, resolve_data_dictionary_id
from marvis.agent.join_setup import propose_roles
from marvis.agent.sample_setup import detect_setup
from marvis.domain import FileRole

_DATA_ROLES = frozenset({FileRole.SAMPLE.value, "sample", "feature"})
_DEFAULT_METRICS = (
    # FEATURE §2 defaults for omitted/legacy tasks. Meaning consistency remains
    # an explicit Agent-only selection.
    "iv",
    "ks",
    "auc",
    "coverage",
)
_MEANING_DIRECTIONS = frozenset({"positive", "negative", "u_shape", "uncertain"})
_MEANING_CONFIDENCE = frozenset({"high", "medium", "low"})
_MEANING_BATCH_SIZE = 50
_MEANING_PROMPT_NAME = "feature_meaning_direction"
_MEANING_PROMPT_VERSION = 1


class FeatureSetupError(ValueError):
    """Raised when the task has no analysable dataset."""


class FeatureTargetChoiceRequired(FeatureSetupError):
    """The sample has multiple valid targets and needs one exact user choice."""

    def __init__(
        self,
        *,
        dataset_id: str,
        dataset_name: str,
        candidates: list[str],
    ) -> None:
        self.dataset_id = str(dataset_id)
        self.dataset_name = str(dataset_name)
        self.candidates = list(dict.fromkeys(str(item) for item in candidates if str(item)))
        super().__init__(
            "检测到多个合法目标列，请明确选择一个目标列："
            + "、".join(self.candidates)
        )


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
    ingest_notices: list[dict] | None = None
    dictionary_id: str = ""
    meaning_directions: dict[str, dict] | None = None

    def template_slots(self) -> dict:
        if self.template_id == "feature_analysis_with_join":
            return {
                "anchor_id": self.anchor_id or self.dataset_id,
                "feature_ids": list(self.feature_ids or []),
                "target_col": self.target_col,
                "features": [],
                "metrics": self.metrics,
                "meaning_directions": dict(self.meaning_directions or {}),
            }
        return {
            "dataset_id": self.dataset_id,
            "target_col": self.target_col,
            "features": self.features,
            "metrics": self.metrics,
            "meaning_directions": dict(self.meaning_directions or {}),
        }


def build_feature_proposal(
    registry,
    backend,
    task_id: str,
    source_dir,
    *,
    metrics: list[str] | None = None,
    configured_target: str = "",
    configured_features: tuple[str, ...] | list[str] = (),
) -> FeatureProposal:
    datasets = _resolve_datasets(registry, task_id, source_dir)
    # GAP-4: register a data-dictionary material (if present) as a dataset, same
    # detection the modeling setup flow already does. Best-effort/side-effect
    # only — never blocks feature-analysis setup when no dictionary exists.
    dictionary_id = resolve_data_dictionary_id(registry, task_id, source_dir)
    joined = len(datasets) > 1
    if joined:
        ranked = propose_roles(datasets)
        dataset = ranked[0]
        feature_ids = [item.id for item in ranked[1:]]
    else:
        dataset = datasets[0]
        feature_ids = []
    path = registry.resolve_path(dataset.id)
    setup = detect_setup(
        backend,
        path,
        configured_target=str(configured_target or "").strip(),
        include_columns=configured_features,
    )
    if not setup.target_col:
        if setup.target_candidates:
            raise FeatureTargetChoiceRequired(
                dataset_id=dataset.id,
                dataset_name=_dataset_name(dataset),
                candidates=list(setup.target_candidates),
            )
        raise FeatureSetupError(
            "未能在数据中识别 0/1 目标列；请明确提供一个同时含 0 和 1 的目标列。"
        )
    requested = _DEFAULT_METRICS if metrics is None else metrics
    selected = [str(item).strip() for item in requested if str(item).strip()]
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
        ingest_notices=_consume_ingest_notices(registry, task_id),
        dictionary_id=dictionary_id,
    )


def infer_meaning_directions(
    client,
    backend,
    registry,
    proposal: FeatureProposal,
) -> dict[str, dict]:
    """Classify governed dictionary meanings into a bounded direction schema.

    This step interprets text only. Measured correlation/bin bad-rates remain in
    the deterministic feature tool. Missing or failed LLM calls are explicitly
    conservative: ``uncertain`` rather than a keyword guess.
    """

    if "meaning_consistency" not in proposal.metrics or not proposal.dictionary_id:
        return {}
    meanings = load_business_names(
        backend,
        registry,
        proposal.dictionary_id,
    )
    feature_scope = list(proposal.features)
    if proposal.template_id == "feature_analysis_with_join":
        # Join-composed analysis resolves the final feature list only after the
        # join tool runs, so the setup proposal intentionally has no features.
        # Scope semantic classification to columns that actually exist in the
        # selected feature tables; classifying the whole dictionary would send
        # unrelated business metadata to the LLM.
        feature_scope = []
        for dataset_id in proposal.feature_ids or []:
            try:
                dataset = registry.get(dataset_id)
            except (KeyError, ValueError):
                continue
            feature_scope.extend(
                str(getattr(profile, "name", "") or "").strip()
                for profile in getattr(dataset, "columns", ())
            )
    scoped = {
        feature: str(meanings.get(feature) or "").strip()
        for feature in dict.fromkeys(feature_scope)
        if feature
        if str(meanings.get(feature) or "").strip()
    }
    if not scoped:
        return {}
    fallback_source = "no_llm_fallback"
    fallback_reason = "未配置可用 LLM，按保守策略标记为不确定。"
    if client is not None and isinstance(getattr(client, "profile", None), dict):
        fallback_source = "llm_unavailable"
        fallback_reason = "LLM 语义判向失败，按保守策略标记为不确定。"
    output: dict[str, dict] = {}
    items = list(scoped.items())
    for offset in range(0, len(items), _MEANING_BATCH_SIZE):
        batch = items[offset:offset + _MEANING_BATCH_SIZE]
        parsed = _call_meaning_direction_llm(
            client,
            batch,
            target_col=proposal.target_col,
        )
        by_feature = {
            str(item.get("feature") or ""): item
            for item in (parsed or [])
            if isinstance(item, dict)
        }
        for feature, meaning in batch:
            item = by_feature.get(feature) or {}
            direction = str(item.get("direction") or "")
            confidence = str(item.get("confidence") or "")
            valid = (
                direction in _MEANING_DIRECTIONS
                and confidence in _MEANING_CONFIDENCE
            )
            output[feature] = {
                "business_meaning": meaning,
                "expected_direction": direction if valid else "uncertain",
                "confidence": confidence if valid else "low",
                "rationale": (
                    str(item.get("rationale") or "").strip()[:300]
                    if valid
                    else fallback_reason
                ),
                "judgement_source": "llm_semantic_direction" if valid else fallback_source,
                "model": str(
                    (getattr(client, "profile", {}) or {}).get("model_name") or ""
                ) if valid else "",
                "prompt_name": _MEANING_PROMPT_NAME,
                "prompt_version": _MEANING_PROMPT_VERSION,
            }
    return output


def _call_meaning_direction_llm(
    client,
    entries: list[tuple[str, str]],
    *,
    target_col: str,
) -> list[dict] | None:
    if client is None or not isinstance(getattr(client, "profile", None), dict):
        return None
    feature_names = [feature for feature, _meaning in entries]
    schema = {
        "name": "feature_meaning_direction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "directions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "feature": {"type": "string", "enum": feature_names},
                            "direction": {
                                "type": "string",
                                "enum": sorted(_MEANING_DIRECTIONS),
                            },
                            "confidence": {
                                "type": "string",
                                "enum": sorted(_MEANING_CONFIDENCE),
                            },
                            "rationale": {"type": "string", "maxLength": 300},
                        },
                        "required": [
                            "feature",
                            "direction",
                            "confidence",
                            "rationale",
                        ],
                        "additionalProperties": False,
                    },
                    "maxItems": len(entries),
                },
            },
            "required": ["directions"],
            "additionalProperties": False,
        },
    }
    prompt = {
        "target": {
            "column": target_col,
            "definition": "二分类风险事件；平台约定 y=1 表示坏/风险事件",
        },
        "allowed_directions": sorted(_MEANING_DIRECTIONS),
        "features": [
            {"feature": feature, "business_meaning": meaning}
            for feature, meaning in entries
        ],
        "instruction": (
            "仅根据字段含义判断对风险事件的业务预期方向。"
            "positive=特征越大风险越高；negative=越大风险越低；"
            "u_shape=两端风险高、中间低；无法可靠判断必须 uncertain。"
            "不得生成或猜测任何统计指标。"
        ),
    }
    try:
        raw = client.complete(
            system_prompt=(
                "你是受约束的风险变量语义分类器，只能输出给定 JSON schema。"
                "不得计算、补写或猜测 IV、KS、AUC、相关系数和坏率。"
            ),
            user_prompt=json.dumps(prompt, ensure_ascii=False, sort_keys=True),
            temperature=0.0,
            response_format={"type": "json_object"},
            json_schema=schema,
            max_tokens=min(4096, 160 + len(entries) * 90),
            stream=False,
            caller="feature_meaning_direction",
            prompt_name=_MEANING_PROMPT_NAME,
            prompt_version=_MEANING_PROMPT_VERSION,
        )
        payload, _error = load_json_object(raw)
    except Exception:
        return None
    directions = payload.get("directions") if isinstance(payload, dict) else None
    return list(directions) if isinstance(directions, list) else None


def _consume_ingest_notices(registry, task_id: str) -> list[dict]:
    consume = getattr(registry, "consume_ingest_notices", None)
    return list(consume(task_id)) if callable(consume) else []


def _resolve_datasets(registry, task_id: str, source_dir):
    datasets = reconcile_source_data_tables(
        registry,
        task_id,
        source_dir,
        accepted_roles=_DATA_ROLES,
        registered_role="sample",
    )
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


__all__ = [
    "build_feature_proposal",
    "FeatureProposal",
    "FeatureSetupError",
    "FeatureTargetChoiceRequired",
    "infer_meaning_directions",
]
