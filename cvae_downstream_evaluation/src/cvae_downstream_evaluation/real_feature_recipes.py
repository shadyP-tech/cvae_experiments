"""Recipe-level real-feature classifier fitting.

This module is intentionally separate from ``classifiers.py``. The existing
classifier contract is logistic-only and threshold-aware, while this path owns
fold-local real-feature preprocessing and hard-label multi-family recipes.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

from .artifacts import stable_hash
from .protocol import ProtocolError


VALID_MODEL_FAMILIES = frozenset({"logistic", "linear_svm", "nystroem_svm", "mlp"})
VALID_CLASS_WEIGHTS = frozenset({None, "balanced"})
VALID_DECISION_RULES = frozenset({"predict"})
PREPROCESSING_FIT_SCOPE = "fold_train_only"
OUTER_PREPROCESSING_FIT_SCOPE = "outer_source_train_only"
MODEL_FIT_SCOPE = "fold_train_only"
OUTER_MODEL_FIT_SCOPE = "outer_source_train_only"


@dataclass(frozen=True)
class PreprocessingSpec:
    """Fold-local preprocessing identity for real Virchow2 features."""

    kind: str = "standardize"
    pca_components: int | None = None
    random_state: int | None = None
    fit_scope: str = PREPROCESSING_FIT_SCOPE

    def __post_init__(self) -> None:
        if self.kind not in {"standardize", "standardize_pca"}:
            raise ProtocolError(f"Unsupported preprocessing kind: {self.kind!r}")
        if self.fit_scope != PREPROCESSING_FIT_SCOPE:
            raise ProtocolError("Preprocessing fit_scope must be fold_train_only.")
        if self.kind == "standardize":
            if self.pca_components is not None:
                raise ProtocolError("standardize preprocessing cannot declare PCA components.")
        else:
            if self.pca_components is None or int(self.pca_components) <= 0:
                raise ProtocolError("standardize_pca preprocessing requires positive pca_components.")

    @property
    def preprocessing_id(self) -> str:
        if self.kind == "standardize":
            return "std_only"
        return f"std_pca{int(self.pca_components or 0)}"

    @property
    def complexity(self) -> tuple[int, int]:
        return (0, 0) if self.kind == "standardize" else (1, int(self.pca_components or 0))

    def to_payload(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "preprocessing_id": self.preprocessing_id,
            "pca_components": self.pca_components,
            "random_state": self.random_state,
            "fit_scope": self.fit_scope,
        }


@dataclass(frozen=True)
class ModelSpec:
    """Family-specific hard-label classifier identity."""

    family: str
    C: float | None = None
    solver: str | None = None
    max_iter: int = 2000
    class_weight: str | None = None
    random_state: int = 23
    nystroem_components: int | None = None
    gamma: float | None = None
    hidden_layer_sizes: tuple[int, ...] | None = None
    alpha: float | None = None
    decision_rule: str = "predict"

    def __post_init__(self) -> None:
        if self.family not in VALID_MODEL_FAMILIES:
            raise ProtocolError(f"Unsupported recipe family: {self.family!r}")
        if self.decision_rule not in VALID_DECISION_RULES:
            raise ProtocolError(f"Unsupported decision rule: {self.decision_rule!r}")
        if self.max_iter <= 0:
            raise ProtocolError("Model max_iter must be positive.")
        if self.class_weight not in VALID_CLASS_WEIGHTS:
            raise ProtocolError(f"Unsupported class_weight: {self.class_weight!r}")
        if self.family in {"logistic", "linear_svm", "nystroem_svm"}:
            if self.C is None or float(self.C) <= 0.0:
                raise ProtocolError(f"{self.family} requires positive C.")
        if self.family == "logistic" and self.solver != "lbfgs":
            raise ProtocolError("v3 logistic recipes use solver='lbfgs'.")
        if self.family == "nystroem_svm":
            if self.nystroem_components is None or int(self.nystroem_components) <= 0:
                raise ProtocolError("nystroem_svm requires positive nystroem_components.")
            if self.gamma is None or float(self.gamma) <= 0.0:
                raise ProtocolError("nystroem_svm requires positive gamma.")
        if self.family == "mlp":
            if self.class_weight is not None:
                raise ProtocolError("v3 MLP recipes do not support class_weight.")
            if not self.hidden_layer_sizes or any(int(size) <= 0 for size in self.hidden_layer_sizes):
                raise ProtocolError("MLP recipes require positive hidden_layer_sizes.")
            if self.alpha is None or float(self.alpha) < 0.0:
                raise ProtocolError("MLP recipes require non-negative alpha.")

    @property
    def imbalance(self) -> str:
        return "balanced" if self.class_weight == "balanced" else "none"

    @property
    def family_order(self) -> int:
        return {"logistic": 0, "linear_svm": 1, "nystroem_svm": 2, "mlp": 3}[self.family]

    @property
    def complexity(self) -> tuple[object, ...]:
        if self.family == "logistic":
            return (0, float(self.C or 0.0), 0 if self.class_weight is None else 1, int(self.max_iter))
        if self.family == "linear_svm":
            return (1, float(self.C or 0.0), 0 if self.class_weight is None else 1, int(self.max_iter))
        if self.family == "nystroem_svm":
            return (
                2,
                int(self.nystroem_components or 0),
                float(self.gamma or 0.0),
                float(self.C or 0.0),
                0 if self.class_weight is None else 1,
                int(self.max_iter),
            )
        return (3, tuple(int(size) for size in self.hidden_layer_sizes or ()), float(self.alpha or 0.0), int(self.max_iter))

    def to_payload(self) -> dict[str, object]:
        return {
            "family": self.family,
            "C": self.C,
            "solver": self.solver,
            "max_iter": int(self.max_iter),
            "class_weight": self.class_weight,
            "imbalance": self.imbalance,
            "random_state": int(self.random_state),
            "nystroem_components": self.nystroem_components,
            "gamma": self.gamma,
            "hidden_layer_sizes": list(self.hidden_layer_sizes or ()),
            "alpha": self.alpha,
            "decision_rule": self.decision_rule,
        }


@dataclass(frozen=True)
class RecipeSpec:
    """Complete preprocessing plus model recipe selected source-inner."""

    preprocessing: PreprocessingSpec
    model: ModelSpec

    @property
    def config_hash(self) -> str:
        return stable_hash(self.to_payload())

    @property
    def family(self) -> str:
        return self.model.family

    @property
    def preprocessing_id(self) -> str:
        return self.preprocessing.preprocessing_id

    def to_payload(self) -> dict[str, object]:
        return {
            "preprocessing": self.preprocessing.to_payload(),
            "model": self.model.to_payload(),
            "recipe_family": self.family,
            "preprocessing_id": self.preprocessing_id,
        }

    def tie_break_key(self) -> tuple[object, ...]:
        return (
            self.model.family_order,
            *self.preprocessing.complexity,
            *self.model.complexity,
            self.config_hash,
        )


@dataclass(frozen=True)
class FittedRecipeResult:
    predictions: tuple[int, ...]
    score_pos: tuple[float | None, ...]
    decision_score: tuple[float | None, ...]
    classes: tuple[int, ...]
    n_iter: tuple[int, ...]
    converged: bool
    status: str
    error_message: str
    recipe_config_hash: str
    score_kind: str


def recipe_grid_hash(recipes: Sequence[RecipeSpec]) -> str:
    if not recipes:
        raise ProtocolError("Recipe grid must contain at least one recipe.")
    return stable_hash([recipe.to_payload() for recipe in recipes])


def assert_fixed_recipe_random_state(recipes: Sequence[RecipeSpec]) -> int:
    states = {int(recipe.model.random_state) for recipe in recipes}
    states.update(int(recipe.preprocessing.random_state) for recipe in recipes if recipe.preprocessing.random_state is not None)
    if len(states) != 1:
        raise ProtocolError(
            "Do not sweep random_state as a recipe hyperparameter; use one fixed "
            "state or evaluate repeated seeds as a separate predeclared stability axis."
        )
    return states.pop()


def fit_recipe(
    train_embeddings: Sequence[Sequence[float]],
    train_labels: Sequence[int],
    eval_embeddings: Sequence[Sequence[float]],
    *,
    recipe: RecipeSpec,
) -> FittedRecipeResult:
    """Fit a recipe with all preprocessing state learned from train data only."""

    try:
        import numpy as np  # type: ignore
        from sklearn.exceptions import ConvergenceWarning  # type: ignore
        from sklearn.kernel_approximation import Nystroem  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.neural_network import MLPClassifier  # type: ignore
        from sklearn.pipeline import Pipeline  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
        from sklearn.decomposition import PCA  # type: ignore
        from sklearn.svm import LinearSVC  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Real-feature recipe fitting requires numpy and scikit-learn.") from exc

    x_train = np.asarray(train_embeddings, dtype=float)
    y_train = np.asarray(train_labels, dtype=int)
    x_eval = np.asarray(eval_embeddings, dtype=float)
    if x_train.ndim != 2 or x_eval.ndim != 2:
        raise ValueError("Embeddings must be 2D arrays.")
    if x_train.shape[1] != x_eval.shape[1]:
        raise ValueError("Training and evaluation embeddings must share the same feature frame.")
    if sorted(set(int(value) for value in y_train.tolist())) != [0, 1]:
        raise ValueError("Real-feature recipes expect binary training labels 0/1.")

    model = recipe.model
    steps: list[tuple[str, object]] = [("scaler", StandardScaler())]
    if recipe.preprocessing.kind == "standardize_pca":
        steps.append(
            (
                "pca",
                PCA(
                    n_components=int(recipe.preprocessing.pca_components or 0),
                    svd_solver="randomized",
                    random_state=int(recipe.preprocessing.random_state or model.random_state),
                ),
            )
        )
    if model.family == "nystroem_svm":
        steps.append(
            (
                "nystroem",
                Nystroem(
                    kernel="rbf",
                    n_components=int(model.nystroem_components or 0),
                    gamma=float(model.gamma or 0.0),
                    random_state=int(model.random_state),
                ),
            )
        )
    if model.family == "logistic":
        estimator = LogisticRegression(
            C=float(model.C or 0.0),
            solver=str(model.solver),
            penalty="l2",
            max_iter=int(model.max_iter),
            class_weight=model.class_weight,
            random_state=int(model.random_state),
        )
    elif model.family in {"linear_svm", "nystroem_svm"}:
        estimator = LinearSVC(
            C=float(model.C or 0.0),
            max_iter=int(model.max_iter),
            class_weight=model.class_weight,
            random_state=int(model.random_state),
        )
    else:
        estimator = MLPClassifier(
            hidden_layer_sizes=tuple(int(size) for size in model.hidden_layer_sizes or ()),
            alpha=float(model.alpha or 0.0),
            max_iter=int(model.max_iter),
            early_stopping=False,
            random_state=int(model.random_state),
        )
    steps.append(("model", estimator))
    pipeline = Pipeline(steps)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.fit(x_train, y_train)
    convergence_warnings = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
    predictions = tuple(int(value) for value in pipeline.predict(x_eval).tolist())
    fitted_model = pipeline.named_steps["model"]
    classes = tuple(int(value) for value in getattr(fitted_model, "classes_", (0, 1)))
    n_iter = _n_iter_tuple(getattr(fitted_model, "n_iter_", ()))
    converged = not convergence_warnings and all(value < int(model.max_iter) for value in n_iter)
    if hasattr(pipeline, "predict_proba") and model.family in {"logistic", "mlp"}:
        proba = pipeline.predict_proba(x_eval)
        if classes != (0, 1):
            raise ValueError("Probability diagnostics require classes (0, 1).")
        score_pos = tuple(float(value) for value in proba[:, 1].tolist())
        decision_score: tuple[float | None, ...] = tuple(None for _ in predictions)
        score_kind = "predict_proba"
    elif hasattr(pipeline, "decision_function"):
        raw = pipeline.decision_function(x_eval)
        values = raw.tolist()
        if values and isinstance(values[0], list):
            values = [row[-1] for row in values]
        decision_score = tuple(float(value) for value in values)
        score_pos = tuple(None for _ in predictions)
        score_kind = "decision_function"
    else:
        score_pos = tuple(None for _ in predictions)
        decision_score = tuple(None for _ in predictions)
        score_kind = "none"
    return FittedRecipeResult(
        predictions=predictions,
        score_pos=score_pos,
        decision_score=decision_score,
        classes=classes,
        n_iter=n_iter,
        converged=bool(converged),
        status="ok" if converged else "non_converged",
        error_message="",
        recipe_config_hash=recipe.config_hash,
        score_kind=score_kind,
    )


def _n_iter_tuple(raw: object) -> tuple[int, ...]:
    if raw is None:
        return ()
    if isinstance(raw, (list, tuple)):
        return tuple(int(value) for value in raw)
    try:
        return tuple(int(value) for value in raw)  # type: ignore[operator]
    except TypeError:
        return (int(raw),)
