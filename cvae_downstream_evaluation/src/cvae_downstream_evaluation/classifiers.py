"""Validated downstream classifier specs and sklearn construction helpers."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Mapping, Sequence

from .artifacts import stable_hash
from .protocol import ProtocolError


VALID_SOLVERS_BY_PENALTY: Mapping[str, frozenset[str]] = {
    "l1": frozenset({"liblinear", "saga"}),
    "l2": frozenset({"lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"}),
    "elasticnet": frozenset({"saga"}),
}

VALID_CLASS_WEIGHTS = frozenset({None, "balanced"})
VALID_THRESHOLD_POLICIES = frozenset({"predict", "fixed_0_5"})


@dataclass(frozen=True)
class ClassifierSpec:
    """Protocol identity for a downstream logistic-regression classifier."""

    C: float = 1.0
    penalty: str = "l2"
    solver: str = "lbfgs"
    max_iter: int = 2000
    class_weight: str | None = None
    random_state: int = 17
    l1_ratio: float | None = None
    threshold_policy: str = "predict"
    scaler_fit: str = "synthetic_train_only"
    family: str = "sklearn_logistic_regression"

    def __post_init__(self) -> None:
        validate_classifier_spec(self)

    @property
    def config_hash(self) -> str:
        return stable_hash(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "family": self.family,
            "C": float(self.C),
            "penalty": self.penalty,
            "solver": self.solver,
            "max_iter": int(self.max_iter),
            "class_weight": self.class_weight,
            "random_state": int(self.random_state),
            "l1_ratio": self.l1_ratio,
            "threshold_policy": self.threshold_policy,
            "scaler_fit": self.scaler_fit,
        }

    def tie_break_key(self) -> tuple[int, float, int, int, int, str]:
        penalty_order = {"l2": 0, "elasticnet": 1, "l1": 2}
        solver_order = {
            "lbfgs": 0,
            "newton-cg": 1,
            "newton-cholesky": 2,
            "sag": 3,
            "saga": 4,
            "liblinear": 5,
        }
        class_weight_order = {None: 0, "balanced": 1}
        return (
            penalty_order.get(self.penalty, 99),
            float(self.C),
            solver_order.get(self.solver, 99),
            class_weight_order.get(self.class_weight, 99),
            int(self.max_iter),
            self.config_hash,
        )

    def to_sklearn_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "C": float(self.C),
            "penalty": self.penalty,
            "solver": self.solver,
            "max_iter": int(self.max_iter),
            "class_weight": self.class_weight,
            "random_state": int(self.random_state),
        }
        if self.penalty == "elasticnet":
            kwargs["l1_ratio"] = float(self.l1_ratio)  # type: ignore[arg-type]
        return kwargs


@dataclass(frozen=True)
class FittedClassifierResult:
    predictions: object
    probabilities: object
    classes: tuple[int, ...]
    n_iter: tuple[int, ...]
    converged: bool
    classifier_config_hash: str


def validate_classifier_spec(spec: ClassifierSpec) -> None:
    if spec.family != "sklearn_logistic_regression":
        raise ProtocolError(f"Unsupported classifier family: {spec.family!r}")
    if spec.C <= 0.0:
        raise ProtocolError("ClassifierSpec.C must be positive.")
    if spec.max_iter <= 0:
        raise ProtocolError("ClassifierSpec.max_iter must be positive.")
    if spec.scaler_fit != "synthetic_train_only":
        raise ProtocolError("Classifier scaler_fit must remain synthetic_train_only.")
    if spec.class_weight not in VALID_CLASS_WEIGHTS:
        raise ProtocolError(f"Unsupported class_weight: {spec.class_weight!r}")
    if spec.threshold_policy not in VALID_THRESHOLD_POLICIES:
        raise ProtocolError(f"Unsupported threshold_policy: {spec.threshold_policy!r}")
    valid_solvers = VALID_SOLVERS_BY_PENALTY.get(spec.penalty)
    if valid_solvers is None:
        raise ProtocolError(f"Unsupported penalty: {spec.penalty!r}")
    if spec.solver not in valid_solvers:
        raise ProtocolError(f"Solver {spec.solver!r} is invalid for penalty {spec.penalty!r}.")
    if spec.penalty == "elasticnet":
        if spec.l1_ratio is None:
            raise ProtocolError("elasticnet classifier specs require l1_ratio.")
        if not 0.0 <= float(spec.l1_ratio) <= 1.0:
            raise ProtocolError("ClassifierSpec.l1_ratio must be in [0, 1].")
    elif spec.l1_ratio is not None:
        raise ProtocolError("l1_ratio is only allowed when penalty='elasticnet'.")


DEFAULT_LOCKED_CLASSIFIER_SPEC = ClassifierSpec(
    C=1.0,
    penalty="l2",
    solver="lbfgs",
    max_iter=2000,
    class_weight=None,
    random_state=17,
    l1_ratio=None,
    threshold_policy="predict",
)


def classifier_grid_hash(specs: Sequence[ClassifierSpec]) -> str:
    if not specs:
        raise ProtocolError("Classifier grid must contain at least one spec.")
    return stable_hash([spec.to_payload() for spec in specs])


def assert_fixed_random_state_policy(specs: Sequence[ClassifierSpec]) -> int:
    states = {int(spec.random_state) for spec in specs}
    if len(states) != 1:
        raise ProtocolError(
            "Do not sweep random_state as a classifier hyperparameter; use one fixed "
            "state or evaluate repeated seeds as a separate predeclared stability axis."
        )
    return states.pop()


def fit_logistic_classifier(
    train_embeddings: Sequence[Sequence[float]],
    train_labels: Sequence[int],
    eval_embeddings: Sequence[Sequence[float]],
    *,
    spec: ClassifierSpec,
) -> FittedClassifierResult:
    """Fit a validated logistic classifier with scaler fit only on training data."""

    try:
        import numpy as np  # type: ignore
        from sklearn.exceptions import ConvergenceWarning  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Downstream classifier fitting requires numpy and scikit-learn.") from exc

    validate_classifier_spec(spec)
    x_train = np.asarray(train_embeddings, dtype=float)
    y_train = np.asarray(train_labels, dtype=int)
    x_eval = np.asarray(eval_embeddings, dtype=float)
    if x_train.ndim != 2 or x_eval.ndim != 2:
        raise ValueError("Embeddings must be 2D arrays.")
    if x_train.shape[1] != x_eval.shape[1]:
        raise ValueError("Training and evaluation embeddings must share the same projection frame.")
    if sorted(set(int(v) for v in y_train.tolist())) != [0, 1]:
        raise ValueError("Downstream logistic classifier expects binary training labels 0/1.")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_eval_scaled = scaler.transform(x_eval)
    clf = LogisticRegression(**spec.to_sklearn_kwargs())
    convergence_warnings = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clf.fit(x_train_scaled, y_train)
        convergence_warnings = [
            item for item in caught if issubclass(item.category, ConvergenceWarning)
        ]
    proba = clf.predict_proba(x_eval_scaled)
    classes = tuple(int(v) for v in clf.classes_.tolist())
    if spec.threshold_policy == "fixed_0_5":
        if classes != (0, 1):
            raise ValueError("fixed_0_5 threshold policy requires classifier classes (0, 1).")
        pred = (proba[:, 1] >= 0.5).astype(int)
    else:
        pred = clf.predict(x_eval_scaled)
    n_iter = tuple(int(v) for v in getattr(clf, "n_iter_", ()))
    converged = not convergence_warnings and all(value < int(spec.max_iter) for value in n_iter)
    return FittedClassifierResult(
        predictions=pred,
        probabilities=proba,
        classes=classes,
        n_iter=n_iter,
        converged=bool(converged),
        classifier_config_hash=spec.config_hash,
    )
