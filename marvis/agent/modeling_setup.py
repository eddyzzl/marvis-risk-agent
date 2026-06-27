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

from marvis.agent.sample_setup import detect_setup
from marvis.domain import FileRole
from marvis.files import scan_source_dir

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
    seed: int = 23
    target_type: str = "binary"  # derived: _regressor⇒continuous, *multiclass*⇒multiclass, else binary
    notes: list[str] = field(default_factory=list)

    def template_slots(self) -> dict:
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
        }


# Recipes selectable for the binary credit-risk default; lgb is the recommended
# starting algorithm. mlp = a sklearn DNN (impute→scale→MLP pipeline).
# lgb_regressor = the continuous-target (regression) recipe.
_SUPPORTED_RECIPES = ("lgb", "xgb", "lr", "scorecard", "mlp", "lgb_regressor", "lgb_multiclass")


def build_modeling_proposal(
    registry, backend, task_id: str, source_dir, *, seed: int = 23,
    recipe: str | None = None, recipes: list[str] | None = None,
) -> ModelingProposal:
    dataset = _resolve_dataset(registry, task_id, source_dir)
    path = registry.resolve_path(dataset.id)
    if recipes:
        recipe_list = [str(item).strip() for item in recipes]
    elif recipe:
        recipe_list = [str(recipe).strip()]
    else:
        recipe_list = ["lgb"]
    for item in recipe_list:
        if item not in _SUPPORTED_RECIPES:
            raise ModelingSetupError(
                f"不支持的算法 `{item}`;可选:{', '.join(_SUPPORTED_RECIPES)}。"
            )
    # target_type is DERIVED from the chosen recipes (no DB/task field): a regression
    # recipe (id ends with "_regressor") ⇒ continuous; a multiclass recipe (id contains
    # "multiclass") ⇒ multiclass; otherwise binary. Mixing regression and multiclass
    # recipes in one run is contradictory, so reject it.
    target_type = _derive_target_type(recipe_list)
    setup = detect_setup(backend, path, target_type=target_type)
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
    if target_type == "continuous":
        notes.append("回归任务（连续型目标）：指标用 RMSE/MAE/R2,不计算坏率/KS/AUC。")
    elif target_type == "multiclass":
        notes.append("多分类任务：指标用 macro-AUC/logloss/准确率,不计算坏率/KS。")
    if setup.split_col:
        dataset_id = dataset.id
        split_col = setup.split_col
        split_values = dict(setup.split_values)
        counts = dict(setup.counts)
    else:
        dataset_id, split_col, split_values, counts, note = _generate_split(
            registry, backend, dataset, setup, seed
        )
        notes.append(note)
    if len(recipe_list) > 1:
        notes.append(f"算法:{'/'.join(recipe_list)}(多算法训练后按 OOT KS 取最优)。")
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
    )


def _derive_target_type(recipe_list: list[str]) -> str:
    """Derive the task target_type from the chosen recipes.

    A regression recipe (id ends with "_regressor") ⇒ "continuous"; a multiclass recipe
    (id contains "multiclass") ⇒ "multiclass"; otherwise "binary". Regression and
    multiclass recipes are mutually exclusive within one run (different target shapes),
    so reject a mix rather than silently picking one."""
    has_regression = any(item.endswith("_regressor") for item in recipe_list)
    has_multiclass = any("multiclass" in item for item in recipe_list)
    if has_regression and has_multiclass:
        raise ModelingSetupError(
            "回归与多分类算法不能在同一次训练混用(目标列形态不同);请分别建模。"
        )
    if has_regression:
        return "continuous"
    if has_multiclass:
        return "multiclass"
    return "binary"


# Identity-like column names used for anti-leakage grouping (best-effort).
_ID_TOKENS = ("cust_id", "user_id", "id_no", "loan_id", "order_id", "apply_id", "mobile", "phone", "身份证", "手机", "cust")


def _detect_group_cols(dataset) -> list[str]:
    for profile in dataset.columns:
        name = str(profile.name)
        low = name.lower()
        if any(token in low or token in name for token in _ID_TOKENS):
            return [name]
    return []


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


def _resolve_dataset(registry, task_id: str, source_dir):
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
    )[0]


def _dataset_name(dataset) -> str:
    source = getattr(dataset, "source_path", None)
    return Path(source).name if source else str(getattr(dataset, "id", ""))


__all__ = ["build_modeling_proposal", "ModelingProposal", "ModelingSetupError"]
