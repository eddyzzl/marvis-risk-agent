from __future__ import annotations

from marvis.packs.modeling.contracts import ModelRecipe


_RECIPES: dict[str, ModelRecipe] = {}


def register_recipe(recipe: ModelRecipe) -> None:
    _RECIPES[recipe.id] = recipe


def get_recipe(recipe_id: str) -> ModelRecipe:
    return _RECIPES[recipe_id]


def list_recipes() -> list[ModelRecipe]:
    return list(_RECIPES.values())


def _register_builtin_recipes() -> None:
    for recipe in (
        ModelRecipe(
            id="lgb",
            algorithm="lgb",
            default_params={
                "objective": "binary",
                "metric": "auc",
                "verbosity": -1,
            },
            param_space={},
            requires_woe=False,
        ),
        ModelRecipe(
            id="xgb",
            algorithm="xgb",
            default_params={
                "objective": "binary:logistic",
                "eval_metric": "auc",
            },
            param_space={},
            requires_woe=False,
        ),
        ModelRecipe(
            id="lr",
            algorithm="lr",
            default_params={
                "max_iter": 1000,
                "solver": "lbfgs",
            },
            param_space={},
            requires_woe=False,
        ),
        ModelRecipe(
            id="scorecard",
            algorithm="scorecard",
            default_params={
                "max_iter": 1000,
                "solver": "lbfgs",
            },
            param_space={},
            requires_woe=True,
        ),
        ModelRecipe(
            id="lgb_regressor",
            algorithm="lgb_regressor",
            default_params={
                "objective": "regression",
                "metric": "rmse",
                "verbosity": -1,
            },
            param_space={},
            requires_woe=False,
        ),
    ):
        register_recipe(recipe)


_register_builtin_recipes()


__all__ = ["get_recipe", "list_recipes", "register_recipe"]
