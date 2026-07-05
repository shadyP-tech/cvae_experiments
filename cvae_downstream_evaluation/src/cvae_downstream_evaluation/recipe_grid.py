"""Predeclared real-feature recipe grids."""

from __future__ import annotations

from .real_feature_recipes import ModelSpec, PreprocessingSpec, RecipeSpec, recipe_grid_hash


def build_v3_recipe_grid(*, classifier_seed: int = 23) -> tuple[RecipeSpec, ...]:
    """Return the approved v3 MIDOG++ real-feature recipe grid."""

    std_only = PreprocessingSpec(kind="standardize", random_state=classifier_seed)
    pca128 = PreprocessingSpec(kind="standardize_pca", pca_components=128, random_state=classifier_seed)
    pca256 = PreprocessingSpec(kind="standardize_pca", pca_components=256, random_state=classifier_seed)
    pca512 = PreprocessingSpec(kind="standardize_pca", pca_components=512, random_state=classifier_seed)
    recipes: list[RecipeSpec] = []
    for preprocessing in (std_only, pca128, pca256, pca512):
        for c_value in (0.01, 0.1, 1.0, 10.0, 100.0):
            for max_iter in (2000, 5000):
                for class_weight in (None, "balanced"):
                    recipes.append(
                        RecipeSpec(
                            preprocessing=preprocessing,
                            model=ModelSpec(
                                family="logistic",
                                C=float(c_value),
                                solver="lbfgs",
                                max_iter=int(max_iter),
                                class_weight=class_weight,
                                random_state=int(classifier_seed),
                            ),
                        )
                    )
    for preprocessing in (std_only, pca128, pca256, pca512):
        for c_value in (0.01, 0.1, 1.0):
            for class_weight in (None, "balanced"):
                recipes.append(
                    RecipeSpec(
                        preprocessing=preprocessing,
                        model=ModelSpec(
                            family="linear_svm",
                            C=float(c_value),
                            max_iter=10000,
                            class_weight=class_weight,
                            random_state=int(classifier_seed),
                        ),
                    )
                )
    for n_components in (256, 512):
        for gamma in (1.0 / 256.0, 1.0 / 128.0):
            for c_value in (0.1, 1.0):
                for class_weight in (None, "balanced"):
                    recipes.append(
                        RecipeSpec(
                            preprocessing=pca256,
                            model=ModelSpec(
                                family="nystroem_svm",
                                C=float(c_value),
                                max_iter=10000,
                                class_weight=class_weight,
                                random_state=int(classifier_seed),
                                nystroem_components=int(n_components),
                                gamma=float(gamma),
                            ),
                        )
                    )
    for hidden in ((64,), (128,)):
        for alpha in (1e-4, 1e-3):
            recipes.append(
                RecipeSpec(
                    preprocessing=pca256,
                    model=ModelSpec(
                        family="mlp",
                        hidden_layer_sizes=tuple(hidden),
                        alpha=float(alpha),
                        max_iter=300,
                        class_weight=None,
                        random_state=int(classifier_seed),
                    ),
                )
            )
    return tuple(recipes)


def build_v3_logistic_baseline_grid(*, classifier_seed: int = 23) -> tuple[RecipeSpec, ...]:
    """Return only the logistic current-family subset for same-cache baseline rows."""

    return tuple(recipe for recipe in build_v3_recipe_grid(classifier_seed=classifier_seed) if recipe.family == "logistic")


def v3_recipe_grid_hash(*, classifier_seed: int = 23) -> str:
    return recipe_grid_hash(build_v3_recipe_grid(classifier_seed=classifier_seed))
