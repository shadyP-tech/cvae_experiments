from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402
from cvae_downstream_evaluation.real_feature_recipes import ModelSpec, PreprocessingSpec, RecipeSpec  # noqa: E402
from cvae_downstream_evaluation.source_inner_classifier_tuning import SourceInnerClassifierFold  # noqa: E402
from cvae_downstream_evaluation.source_inner_recipe_tuning import (  # noqa: E402
    RecipeFoldScore,
    select_recipe_source_inner_lodo,
)


def test_source_inner_recipe_selection_ignores_hypothetical_target_metric() -> None:
    low_c = _recipe("logistic", c=0.1)
    high_c = _recipe("logistic", c=10.0)

    result = select_recipe_source_inner_lodo(
        outer_target_center="T",
        folds=[_fold("A", train_centers=("B", "C")), _fold("B", train_centers=("A", "C"))],
        candidate_recipes=[high_c, low_c],
        score_fn=_score_low_c,
    )

    target_bacc_by_hash = {
        low_c.config_hash: 0.60,
        high_c.config_hash: 0.99,
    }
    assert target_bacc_by_hash[high_c.config_hash] > target_bacc_by_hash[low_c.config_hash]
    assert result.selected_config_hash == low_c.config_hash


def test_one_se_rule_prefers_stable_simpler_recipe() -> None:
    logistic = _recipe("logistic", c=1.0)
    mlp = _recipe("mlp")

    result = select_recipe_source_inner_lodo(
        outer_target_center="T",
        folds=[_fold("A", train_centers=("B", "C")), _fold("B", train_centers=("A", "C"))],
        candidate_recipes=[mlp, logistic],
        score_fn=_score_noisy_mlp,
    )

    assert result.selected_config_hash == logistic.config_hash
    selected_rows = [row for row in result.rows if row.selected_by_source_inner_lodo]
    assert selected_rows[0].within_one_se_best is True


def test_nonconverged_recipe_is_ineligible() -> None:
    logistic = _recipe("logistic", c=1.0)
    svm = _recipe("linear_svm", c=1.0)

    result = select_recipe_source_inner_lodo(
        outer_target_center="T",
        folds=[_fold("A", train_centers=("B", "C"))],
        candidate_recipes=[logistic, svm],
        score_fn=_score_logistic_nonconverged,
    )

    assert result.selected_config_hash == svm.config_hash


def test_source_inner_recipe_selection_rejects_outer_target_leakage() -> None:
    try:
        select_recipe_source_inner_lodo(
            outer_target_center="T",
            folds=[_fold("A", train_centers=("B", "T"))],
            candidate_recipes=[_recipe("logistic", c=1.0)],
            score_fn=_score_low_c,
        )
    except ProtocolError:
        pass
    else:
        raise AssertionError("outer target center was accepted in recipe fit centers")


def _recipe(family: str, *, c: float = 1.0) -> RecipeSpec:
    preprocessing = PreprocessingSpec(kind="standardize", random_state=23)
    if family == "mlp":
        model = ModelSpec(family="mlp", hidden_layer_sizes=(64,), alpha=1e-4, max_iter=300, random_state=23)
    else:
        model = ModelSpec(family=family, C=float(c), solver="lbfgs" if family == "logistic" else None, random_state=23)
    return RecipeSpec(preprocessing=preprocessing, model=model)


def _fold(center: str, *, train_centers: tuple[str, ...]) -> SourceInnerClassifierFold:
    return SourceInnerClassifierFold(
        pseudo_target_center=center,
        train_centers=train_centers,
        train_embeddings=(),
        train_labels=(),
        validation_embeddings=(),
        validation_labels=(),
    )


def _score_low_c(recipe: RecipeSpec, fold: SourceInnerClassifierFold) -> RecipeFoldScore:
    del fold
    if recipe.model.C == 0.1:
        return RecipeFoldScore(bacc=0.80, macro_f1=0.78, n_iter=(10,))
    return RecipeFoldScore(bacc=0.70, macro_f1=0.68, n_iter=(10,))


def _score_noisy_mlp(recipe: RecipeSpec, fold: SourceInnerClassifierFold) -> RecipeFoldScore:
    if recipe.family == "mlp":
        return RecipeFoldScore(
            bacc=0.90 if fold.pseudo_target_center == "A" else 0.70,
            macro_f1=0.75,
            n_iter=(10,),
        )
    return RecipeFoldScore(bacc=0.79, macro_f1=0.77, n_iter=(10,))


def _score_logistic_nonconverged(recipe: RecipeSpec, fold: SourceInnerClassifierFold) -> RecipeFoldScore:
    del fold
    if recipe.family == "logistic":
        return RecipeFoldScore(bacc=0.99, macro_f1=0.99, converged=False, n_iter=(2000,))
    return RecipeFoldScore(bacc=0.70, macro_f1=0.68, n_iter=(10,))
