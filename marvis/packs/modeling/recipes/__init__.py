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
            id="catboost",
            algorithm="catboost",
            default_params={
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "learning_rate": 0.05,
                "depth": 4,
                "verbose": False,
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
        ModelRecipe(
            id="mlp",
            algorithm="mlp",
            default_params={
                "hidden_layer_sizes": [32, 16],
                "max_iter": 300,
                "alpha": 1e-4,
                "early_stopping": False,
            },
            param_space={},
            requires_woe=False,
        ),
        ModelRecipe(
            id="lgb_multiclass",
            algorithm="lgb_multiclass",
            default_params={
                "num_boost_round": 50,
                "learning_rate": 0.1,
                "num_leaves": 15,
                "verbosity": -1,
            },
            param_space={},
            requires_woe=False,
        ),
        # SEL-6: seed-bagging/blend ensemble -- no fixed default_params of its own
        # (each member recipe supplies its own defaults via its own get_recipe()
        # call inside recipes/ensemble.py); registered mainly for discoverability
        # via list_recipes()/get_recipe("ensemble").
        ModelRecipe(
            id="ensemble",
            algorithm="ensemble",
            default_params={},
            param_space={},
            requires_woe=False,
        ),
    ):
        register_recipe(recipe)


_register_builtin_recipes()


__all__ = ["get_recipe", "list_recipes", "register_recipe"]
