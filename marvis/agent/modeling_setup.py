"""Setup (slot-filling) for the modeling task.

Discovers the registered sample, detects the target column, the train/test/oot
split column + values, and the numeric candidate features, then fills the
`modeling` template slots. When the sample already carries a split column we use
it. When it does not, we generate a grouped train/test split via ``make_split``
(spec §2 G1): anti-leakage grouping by an identity column when present, fixed
seed, non-empty guards — and crucially NO fabricated OOT (a real out-of-time
holdout needs a time/split column, so downstream OOT metrics simply degrade to
n/a rather than mislabelling a random holdout as out-of-time).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from marvis.agent.join_setup import propose_roles
from marvis.agent.sample_setup import detect_setup
from marvis.domain import FileRole
from marvis.files import scan_source_dir
from marvis.packs.modeling.defaults import DEFAULT_RANDOM_SEED

_DATA_ROLES = frozenset({FileRole.SAMPLE.value, "sample", "feature"})


class ModelingSetupError(ValueError):
    """Raised when the sample can't be set up for modeling."""


@dataclass
class ModelingProposal:
    dataset_id: str
    dataset_name: str
    target_col: str
    feature_cols: list[str]
    split_col: str
    split_values: dict[str, str]
    holdout_values: list[str]
    bad_rate: float | None
    counts: dict[str, int]
    recipe: str = "lgb"  # primary recipe (the one tuned, if lgb is among recipes)
    recipes: list[str] = field(default_factory=lambda: ["lgb"])  # all recipes to train + compare
    seed: int = DEFAULT_RANDOM_SEED
    target_type: str = "binary"  # derived: _regressor⇒continuous, *multiclass*⇒multiclass, else binary
    notes: list[str] = field(default_factory=list)
    template_id: str = "modeling"
    anchor_id: str | None = None
    join_feature_ids: list[str] = field(default_factory=list)
    sample_weight_col: str = ""
    sample_weight_candidates: list[str] = field(default_factory=list)
    sample_weight_diagnostics: list[dict] = field(default_factory=list)

    def template_slots(self) -> dict:
        selection_policy = _default_selection_policy(self.target_type)
        if self.template_id == "modeling_with_join":
            return {
                "anchor_id": self.anchor_id or self.dataset_id,
                "feature_ids": list(self.join_feature_ids),
                "target_col": self.target_col,
                # Empty means: infer candidate numeric features from the joined schema.
                "feature_cols": [],
                "split_col": self.split_col,
                "split_values": self.split_values,
                "recipe": self.recipe,
                "recipes": self.recipes,
                "seed": self.seed,
                "holdout_values": self.holdout_values,
                "target_type": self.target_type,
                "split_config": {},
                "sample_weight_col": self.sample_weight_col,
                "sample_weight_candidates": list(self.sample_weight_candidates),
                "sample_weight_diagnostics": list(self.sample_weight_diagnostics),
                "passthrough_cols": _unique([self.sample_weight_col, *self.sample_weight_candidates]),
                "selection_policy": selection_policy,
            }
        return {
            "dataset_id": self.dataset_id,
            "target_col": self.target_col,
            "feature_cols": self.feature_cols,
            "split_col": self.split_col,
            "split_values": self.split_values,
            "recipe": self.recipe,
            "recipes": self.recipes,
            "seed": self.seed,
            "holdout_values": self.holdout_values,
            "target_type": self.target_type,
            # The G1 make_split gate passes the setup-decided split through unchanged
            # ({} = passthrough); re-splitting with rules/time/group config is an adjust.
            "split_config": {},
            "sample_weight_col": self.sample_weight_col,
            "sample_weight_candidates": list(self.sample_weight_candidates),
            "sample_weight_diagnostics": list(self.sample_weight_diagnostics),
            "passthrough_cols": _unique([self.sample_weight_col, *self.sample_weight_candidates]),
            "selection_policy": selection_policy,
        }


# Recipes selectable for the binary credit-risk default; lgb is the recommended
# starting algorithm. mlp = a sklearn DNN (impute→scale→MLP pipeline).
# lgb_regressor = the continuous-target (regression) recipe.
_SUPPORTED_RECIPES = ("lgb", "xgb", "catboost", "lr", "scorecard", "mlp", "lgb_regressor", "lgb_multiclass")
_BINARY_RECIPES = frozenset({"lgb", "xgb", "catboost", "lr", "scorecard", "mlp"})
_WEIGHT_NAME_HINTS = ("sample_weight", "sampleweight", "weight", "样本权重", "权重")


def build_modeling_proposal(
    registry, backend, task_id: str, source_dir, *, seed: int = DEFAULT_RANDOM_SEED,
    recipe: str | None = None, recipes: list[str] | None = None,
    target_type: str | None = None,
    sample_weight_col: str | None = None,
    anchor_id: str | None = None,
    join_feature_ids: list[str] | None = None,
    target_col: str | None = None,
) -> ModelingProposal:
    datasets = _resolve_datasets(registry, task_id, source_dir)
    by_id = {dataset.id: dataset for dataset in datasets}
    join_feature_ids = [str(item_id) for item_id in (join_feature_ids or []) if str(item_id)]
    if anchor_id:
        if anchor_id not in by_id:
            raise ModelingSetupError("选择的样本主表不存在;请重新确认文件角色。")
        dataset = by_id[anchor_id]
        join_feature_ids = [
            item_id for item_id in join_feature_ids if item_id in by_id and item_id != anchor_id
        ]
        joined = bool(join_feature_ids)
    elif len(datasets) > 1:
        ranked = propose_roles(datasets)
        dataset = ranked[0]
        join_feature_ids = [item.id for item in ranked[1:]]
        joined = bool(join_feature_ids)
    else:
        dataset = datasets[0]
        join_feature_ids = []
        joined = False
    path = registry.resolve_path(dataset.id)
    requested_target_type = _normalize_target_type(target_type)
    if recipes:
        recipe_list = [str(item).strip() for item in recipes]
    elif recipe:
        recipe_list = [str(recipe).strip()]
    else:
        recipe_list = [_default_recipe_for_target_type(requested_target_type or "binary")]
    for item in recipe_list:
        if item not in _SUPPORTED_RECIPES:
            raise ModelingSetupError(
                f"不支持的算法 `{item}`;可选:{', '.join(_SUPPORTED_RECIPES)}。"
            )
    derived_target_type = _derive_target_type(recipe_list)
    if requested_target_type and requested_target_type != derived_target_type:
        raise ModelingSetupError(
            f"目标类型 `{requested_target_type}` 与算法 `{', '.join(recipe_list)}` 不匹配;请重新选择同一目标类型的算法。"
        )
    target_type = requested_target_type or derived_target_type
    setup = detect_setup(backend, path, configured_target=str(target_col or ""), target_type=target_type)
    if not setup.target_col:
        if target_type == "continuous":
            raise ModelingSetupError("未能识别连续型目标列;请确认数据含数值目标列(如 income/amount)后重试。")
        if target_type == "multiclass":
            raise ModelingSetupError("未能识别多分类目标列;请指定 3-20 类的目标列(如 风险等级/评级)后重试。")
        raise ModelingSetupError("未能识别 0/1 目标列;请确认数据含标签列后重试。")
    # The tuner is lgb-specific, so the "primary" recipe (the one tuned) is lgb when
    # it is among the chosen recipes, else the first one (tuning is skipped for it).
    primary_recipe = "lgb" if "lgb" in recipe_list else recipe_list[0]
    notes = list(setup.notes)
    weight_diagnostics = _sample_weight_diagnostics(
        backend,
        path,
        target_col=setup.target_col,
        split_col=setup.split_col,
    )
    weight_candidates = [item["column"] for item in weight_diagnostics if item.get("valid")]
    selected_weight_col = _normalize_sample_weight_col(
        sample_weight_col,
        available_columns=backend.column_names(path),
    )
    if selected_weight_col:
        if selected_weight_col == setup.target_col or selected_weight_col == str(setup.split_col or ""):
            raise ModelingSetupError("样本权重列不能是目标列或切分列。")
        selected_diag = _sample_weight_diagnostics(
            backend,
            path,
            target_col=setup.target_col,
            split_col=setup.split_col,
            explicit_columns=[selected_weight_col],
        )
        if not selected_diag or not selected_diag[0].get("valid"):
            reason = selected_diag[0].get("reason") if selected_diag else "不是数值型权重列"
            raise ModelingSetupError(f"样本权重列 `{selected_weight_col}` 不可用:{reason}。")
        weight_diagnostics = _merge_weight_diagnostics(selected_diag, weight_diagnostics)
    if selected_weight_col:
        notes.append(f"样本权重列:`{selected_weight_col}`(仅作为 sample_weight,不作为入模特征)。")
        weight_candidates = _unique([selected_weight_col, *weight_candidates])
    elif weight_candidates:
        display = "/".join(f"`{col}`" for col in weight_candidates[:3])
        notes.append(f"检测到样本权重候选列:{display};如需启用,请确认 sample_weight_col。")
    if target_type == "continuous":
        notes.append("回归任务（连续型目标）：指标用 RMSE/MAE/R2,不计算坏率/KS/AUC。")
    elif target_type == "multiclass":
        notes.append("多分类任务：指标用 macro-AUC/logloss/准确率,不计算坏率/KS。")
    if setup.split_col:
        dataset_id = dataset.id
        split_col = setup.split_col
        split_values = dict(setup.split_values)
        counts = dict(setup.counts)
    elif joined:
        dataset_id = dataset.id
        split_col = ""
        split_values = {}
        counts = {}
        group_cols = _detect_group_cols(dataset)
        grouping = f"(按 `{group_cols[0]}` 分组防泄漏)" if group_cols else "(逐行随机)"
        notes.append(
            f"多文件建模将在拼接后自动 75/25 分组随机切 train/test{grouping};"
            "未设 OOT(时间外推 OOT 需切分列或日期列),OOT 相关指标将显示 n/a。"
        )
    else:
        dataset_id, split_col, split_values, counts, note = _generate_split(
            registry, backend, dataset, setup, seed
        )
        notes.append(note)
    if len(recipe_list) > 1:
        notes.append(f"算法:{'/'.join(recipe_list)}(多算法训练后按 {_selection_metric_label(target_type)} 取最优)。")
    else:
        notes.append(f"算法:`{recipe_list[0]}`(可选 {'/'.join(_SUPPORTED_RECIPES)})。")
    oot = split_values.get("oot")
    return ModelingProposal(
        dataset_id=dataset_id,
        dataset_name=_dataset_name(dataset),
        target_col=setup.target_col,
        feature_cols=list(setup.candidates),
        split_col=split_col,
        split_values=split_values,
        holdout_values=[oot] if oot else [],
        bad_rate=setup.bad_rate,
        counts=counts,
        recipe=primary_recipe,
        recipes=recipe_list,
        seed=seed,
        target_type=target_type,
        notes=notes,
        template_id="modeling_with_join" if joined else "modeling",
        anchor_id=dataset.id if joined else None,
        join_feature_ids=join_feature_ids,
        sample_weight_col=selected_weight_col,
        sample_weight_candidates=weight_candidates,
        sample_weight_diagnostics=weight_diagnostics,
    )


def _derive_target_type(recipe_list: list[str]) -> str:
    """Derive the task target_type from the chosen recipes.

    A regression recipe (id ends with "_regressor") ⇒ "continuous"; a multiclass recipe
    (id contains "multiclass") ⇒ "multiclass"; otherwise "binary". Recipe families are
    mutually exclusive within one run (different target shapes), so reject any mix rather
    than silently picking one."""
    has_regression = any(item.endswith("_regressor") for item in recipe_list)
    has_multiclass = any("multiclass" in item for item in recipe_list)
    has_binary = any(item in _BINARY_RECIPES for item in recipe_list)
    family_count = sum(1 for flag in (has_binary, has_regression, has_multiclass) if flag)
    if family_count > 1:
        raise ModelingSetupError(
            "二分类、回归与多分类算法不能在同一次训练混用(目标列形态不同);请分别建模。"
        )
    if has_regression:
        return "continuous"
    if has_multiclass:
        return "multiclass"
    return "binary"


def _normalize_target_type(value: str | None) -> str | None:
    if value is None:
        return None
    target_type = str(value).strip().lower()
    if not target_type:
        return None
    if target_type not in {"binary", "continuous", "multiclass"}:
        raise ModelingSetupError(f"不支持的目标类型 `{target_type}`;可选:binary/continuous/multiclass。")
    return target_type


def _default_recipe_for_target_type(target_type: str) -> str:
    if target_type == "continuous":
        return "lgb_regressor"
    if target_type == "multiclass":
        return "lgb_multiclass"
    return "lgb"


def _default_selection_policy(target_type: str) -> dict[str, bool]:
    if target_type == "binary":
        return {"require_pmml": True, "require_handoff": True}
    return {"require_pmml": False, "require_handoff": False}


def _sample_weight_diagnostics(
    backend,
    path: Path,
    *,
    target_col: str,
    split_col: str | None,
    explicit_columns: list[str] | None = None,
    sample_rows: int = 4000,
) -> list[dict]:
    probe = backend.sample_rows(path, sample_rows, seed=0)
    excluded = {str(target_col), str(split_col or "")}
    columns = explicit_columns or [
        str(column)
        for column in probe.columns
        if any(hint in str(column).lower() or hint in str(column) for hint in _WEIGHT_NAME_HINTS)
    ]
    diagnostics: list[dict] = []
    for column in columns:
        name = str(column)
        if name in excluded:
            continue
        if name not in probe.columns:
            continue
        numeric = pd.to_numeric(probe[name], errors="coerce")
        non_missing = numeric.dropna()
        missing_count = int(numeric.isna().sum())
        reason = ""
        valid = True
        if non_missing.empty:
            valid = False
            reason = "全为空或非数值"
        elif missing_count:
            valid = False
            reason = "存在空值或非数值"
        elif (non_missing < 0).any():
            valid = False
            reason = "存在负权重"
        elif float(non_missing.sum()) <= 0:
            valid = False
            reason = "总权重不为正"
        diagnostics.append({
            "column": name,
            "valid": valid,
            "reason": reason,
            "rows_sampled": int(len(probe)),
            "non_missing": int(non_missing.shape[0]),
            "missing_rate": float(missing_count / len(probe)) if len(probe) else 0.0,
            "min": _maybe_float(non_missing.min()) if not non_missing.empty else None,
            "max": _maybe_float(non_missing.max()) if not non_missing.empty else None,
            "mean": _maybe_float(non_missing.mean()) if not non_missing.empty else None,
            "excluded_from_features": True,
            "leakage_risk": "low",
        })
    return diagnostics


def _merge_weight_diagnostics(primary: list[dict], secondary: list[dict]) -> list[dict]:
    by_column: dict[str, dict] = {}
    for item in [*primary, *secondary]:
        column = str(item.get("column") or "")
        if column and column not in by_column:
            by_column[column] = dict(item)
    return list(by_column.values())


def _maybe_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_sample_weight_col(value: str | None, *, available_columns: list[str]) -> str:
    column = str(value or "").strip()
    if not column:
        return ""
    if column not in set(available_columns):
        raise ModelingSetupError(f"样本权重列 `{column}` 不存在;请检查列名。")
    return column


def _selection_metric_label(target_type: str) -> str:
    if target_type == "continuous":
        return "OOT RMSE"
    if target_type == "multiclass":
        return "OOT macro-AUC"
    return "OOT KS"


# Identity-like column names used for anti-leakage grouping (best-effort).
_ID_TOKENS = ("cust_id", "user_id", "id_no", "loan_id", "order_id", "apply_id", "mobile", "phone", "身份证", "手机", "cust")


def _detect_group_cols(dataset) -> list[str]:
    for profile in dataset.columns:
        name = str(profile.name)
        low = name.lower()
        if any(token in low or token in name for token in _ID_TOKENS):
            return [name]
    return []


def _unique(values: list[str]) -> list[str]:
    return [value for value in dict.fromkeys(str(item).strip() for item in values) if value]


def _generate_split(registry, backend, dataset, setup, seed):
    """No split column → build a grouped train/test split (spec §2 G1). No OOT is
    fabricated; downstream OOT metrics degrade to n/a."""
    from marvis.packs.modeling.errors import ModelingError
    from marvis.packs.modeling.prepare import prepare_modeling_frame

    group_cols = _detect_group_cols(dataset)
    try:
        derived = prepare_modeling_frame(
            registry,
            backend,
            dataset.id,
            target_col=setup.target_col,
            feature_cols=list(setup.candidates),
            split_col=None,
            split_config={"test_size": 0.25, "group_cols": group_cols},
            seed=seed,
        )
    except ModelingError as exc:
        raise ModelingSetupError(f"自动切分失败:{exc}") from exc

    split_series = backend.read_frame(registry.resolve_path(derived.id), columns=["split"])["split"]
    counts = {str(key): int(value) for key, value in split_series.value_counts().items()}
    split_values = {role: role for role in counts}
    grouping = f"(按 `{group_cols[0]}` 分组防泄漏)" if group_cols else "(逐行随机)"
    note = (
        f"未提供切分列,已自动 75/25 分组随机切 train/test{grouping};"
        "未设 OOT(时间外推 OOT 需切分列或日期列),OOT 相关指标将显示 n/a。"
    )
    return derived.id, "split", split_values, counts, note


def _resolve_datasets(registry, task_id: str, source_dir):
    datasets = [d for d in registry.list_for_task(task_id) if d.role in _DATA_ROLES]
    if not datasets and source_dir is not None:
        for artifact in scan_source_dir(Path(source_dir)):
            if artifact.role == FileRole.SAMPLE:
                registry.register_from_upload(task_id, Path(artifact.path), role="sample")
        datasets = [d for d in registry.list_for_task(task_id) if d.role in _DATA_ROLES]
    if not datasets:
        raise ModelingSetupError(f"建模未找到样本文件:{source_dir}")
    return sorted(
        datasets,
        key=lambda d: (not bool(getattr(d, "has_target", False)), -int(getattr(d, "row_count", 0) or 0)),
    )


def _dataset_name(dataset) -> str:
    source = getattr(dataset, "source_path", None)
    return Path(source).name if source else str(getattr(dataset, "id", ""))


__all__ = ["build_modeling_proposal", "ModelingProposal", "ModelingSetupError"]
