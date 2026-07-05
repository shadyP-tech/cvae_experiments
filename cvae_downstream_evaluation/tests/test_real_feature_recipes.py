from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402
from cvae_downstream_evaluation.real_feature_recipes import (  # noqa: E402
    ModelSpec,
    PreprocessingSpec,
    RecipeSpec,
    fit_recipe,
)


def test_recipe_hash_covers_preprocessing_and_family_fields() -> None:
    pca2 = PreprocessingSpec(kind="standardize_pca", pca_components=2, random_state=23)
    pca3 = PreprocessingSpec(kind="standardize_pca", pca_components=3, random_state=23)
    svm_c1 = RecipeSpec(
        preprocessing=pca2,
        model=ModelSpec(family="linear_svm", C=1.0, max_iter=10000, random_state=23),
    )
    svm_c01 = RecipeSpec(
        preprocessing=pca2,
        model=ModelSpec(family="linear_svm", C=0.1, max_iter=10000, random_state=23),
    )
    svm_pca3 = RecipeSpec(
        preprocessing=pca3,
        model=ModelSpec(family="linear_svm", C=1.0, max_iter=10000, random_state=23),
    )
    nystroem_128 = RecipeSpec(
        preprocessing=pca2,
        model=ModelSpec(
            family="nystroem_svm",
            C=1.0,
            nystroem_components=128,
            gamma=1.0 / 128.0,
            max_iter=10000,
            random_state=23,
        ),
    )
    nystroem_256 = RecipeSpec(
        preprocessing=pca2,
        model=ModelSpec(
            family="nystroem_svm",
            C=1.0,
            nystroem_components=256,
            gamma=1.0 / 128.0,
            max_iter=10000,
            random_state=23,
        ),
    )
    mlp_64 = RecipeSpec(
        preprocessing=pca2,
        model=ModelSpec(family="mlp", hidden_layer_sizes=(64,), alpha=1e-4, max_iter=300, random_state=23),
    )
    mlp_128 = RecipeSpec(
        preprocessing=pca2,
        model=ModelSpec(family="mlp", hidden_layer_sizes=(128,), alpha=1e-4, max_iter=300, random_state=23),
    )

    hashes = {
        svm_c1.config_hash,
        svm_c01.config_hash,
        svm_pca3.config_hash,
        nystroem_128.config_hash,
        nystroem_256.config_hash,
        mlp_64.config_hash,
        mlp_128.config_hash,
    }
    assert len(hashes) == 7


def test_recipe_rejects_non_predict_decision_rule() -> None:
    try:
        ModelSpec(family="linear_svm", C=1.0, decision_rule="fixed_0_5")
    except ProtocolError:
        pass
    else:
        raise AssertionError("recipe accepted a threshold-like decision rule")


def test_linear_svm_fit_uses_decision_function_not_probabilities() -> None:
    recipe = RecipeSpec(
        preprocessing=PreprocessingSpec(kind="standardize", random_state=23),
        model=ModelSpec(family="linear_svm", C=1.0, max_iter=10000, random_state=23),
    )
    fitted = fit_recipe(
        [[0.0, 0.0], [0.2, 0.1], [2.0, 2.0], [2.2, 2.1]],
        [0, 0, 1, 1],
        [[0.1, 0.1], [2.1, 2.0]],
        recipe=recipe,
    )

    assert fitted.predictions == (0, 1)
    assert fitted.score_kind == "decision_function"
    assert fitted.score_pos == (None, None)
    assert all(score is not None for score in fitted.decision_score)
