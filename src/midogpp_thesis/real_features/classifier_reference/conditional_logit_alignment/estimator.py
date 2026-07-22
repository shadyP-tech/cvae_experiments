"""Numerically locked conditional-logit alignment estimator."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from ..artifacts import stable_hash
from ..classifiers import (
    ClassifierSpec,
    StandardizedFitEval,
    _fit_standardized_logistic_classifier,
    standardize_fit_eval,
)
from ..protocol import ProtocolError
from .config import (
    AlignmentOptimizerConfig,
    DEFAULT_OPTIMIZER_CONFIG,
    canonical_classifier_spec,
)
from .folds import ConditionalLogitFold
from .penalty import ConditionalPenaltyOperator, build_conditional_penalty


@dataclass(frozen=True)
class ConditionalObjectiveTerms:
    """Auditable decomposition of the frozen binary-logistic objective."""

    objective: float
    mean_log_loss: float
    l2_penalty: float
    alignment_penalty: float
    unscaled_alignment_value: float
    gradient: object


@dataclass(frozen=True)
class PreparedConditionalLogit:
    """One fit-local scaler, penalty operator, and pooled warm start."""

    fold_data: ConditionalLogitFold
    classifier_spec: ClassifierSpec
    standardized: StandardizedFitEval
    penalty_operator: ConditionalPenaltyOperator
    pooled_fit: object

    @property
    def scaler_state_hash(self) -> str:
        return self.standardized.scaler_state_hash

    @property
    def training_frame_hash(self) -> str:
        return self.fold_data.training_frame_hash

    @property
    def fit_row_hash(self) -> str:
        return self.fold_data.fit_row_hash


@dataclass(frozen=True)
class AlignmentFitResult:
    """Predictions, parameters, objective terms, and solver audit for one gamma."""

    gamma: float
    predictions: object
    probabilities: object
    coefficients: object
    intercept: float
    classes: tuple[int, ...]
    n_iter: tuple[int, ...]
    converged: bool
    backend: str
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    n_function_evaluations: int
    n_gradient_evaluations: int
    objective: float
    mean_log_loss: float
    l2_penalty: float
    alignment_penalty: float
    unscaled_alignment_value: float
    gradient_inf_norm: float
    classifier_config_hash: str
    scaler_state_hash: str
    penalty_operator_hash: str
    fit_identity: str

    @property
    def status(self) -> str:
        return "ok" if self.converged else "optimizer_nonconverged"

    @property
    def objective_value(self) -> float:
        return self.objective

    @property
    def factor_hash(self) -> str:
        return self.penalty_operator_hash

    def solver_audit_payload(self) -> dict[str, object]:
        return {
            "gamma": float(self.gamma),
            "fit_identity": self.fit_identity,
            "backend": self.backend,
            "optimizer_success": bool(self.optimizer_success),
            "optimizer_status": int(self.optimizer_status),
            "optimizer_message": self.optimizer_message,
            "n_iter": list(self.n_iter),
            "n_function_evaluations": int(self.n_function_evaluations),
            "n_gradient_evaluations": int(self.n_gradient_evaluations),
            "converged": bool(self.converged),
            "objective": float(self.objective),
            "mean_log_loss": float(self.mean_log_loss),
            "l2_penalty": float(self.l2_penalty),
            "alignment_penalty": float(self.alignment_penalty),
            "unscaled_alignment_value": float(self.unscaled_alignment_value),
            "gradient_inf_norm": float(self.gradient_inf_norm),
            "classifier_config_hash": self.classifier_config_hash,
            "scaler_state_hash": self.scaler_state_hash,
            "factor_hash": self.penalty_operator_hash,
        }


def prepare_conditional_logit(
    fold_data: ConditionalLogitFold,
    spec: ClassifierSpec,
) -> PreparedConditionalLogit:
    """Fit the fold-local scaler and pooled gamma-zero warm start exactly once."""

    _validate_fold_and_spec(fold_data, spec)
    standardized = standardize_fit_eval(
        fold_data.fit_embeddings,
        fold_data.eval_embeddings,
    )
    penalty = build_conditional_penalty(
        standardized.fit_embeddings,
        fold_data.fit_labels,
        fold_data.fit_domains,
    )
    pooled_fit = _fit_standardized_logistic_classifier(
        standardized,
        fold_data.fit_labels,
        spec=spec,
        sample_weight=None,
    )
    return PreparedConditionalLogit(
        fold_data=fold_data,
        classifier_spec=spec,
        standardized=standardized,
        penalty_operator=penalty,
        pooled_fit=pooled_fit,
    )


def fit_conditional_logit(
    fold_data: ConditionalLogitFold,
    spec: ClassifierSpec,
    gamma: float,
    *,
    optimizer: AlignmentOptimizerConfig = DEFAULT_OPTIMIZER_CONFIG,
) -> AlignmentFitResult:
    """Fit one gamma; gamma zero is always the shared standardized sklearn path."""

    prepared = prepare_conditional_logit(fold_data, spec)
    return fit_prepared_conditional_logit(
        prepared,
        gamma,
        optimizer=optimizer,
    )


def fit_prepared_conditional_logit(
    prepared: PreparedConditionalLogit,
    gamma: float,
    *,
    optimizer: AlignmentOptimizerConfig = DEFAULT_OPTIMIZER_CONFIG,
) -> AlignmentFitResult:
    """Fit one gamma on a precomputed scaler/operator frame."""

    import numpy as np  # type: ignore

    gamma_value = _validated_gamma(gamma)
    _validate_optimizer(optimizer)
    x_fit = np.asarray(prepared.standardized.fit_embeddings, dtype=np.float64)
    x_eval = np.asarray(prepared.standardized.eval_embeddings, dtype=np.float64)
    y_fit = np.asarray(prepared.fold_data.fit_labels, dtype=np.float64)
    pooled = prepared.pooled_fit
    pooled_weights = np.asarray(getattr(pooled, "coefficients"), dtype=np.float64).reshape(-1)
    pooled_intercept_values = np.asarray(
        getattr(pooled, "intercept"), dtype=np.float64
    ).reshape(-1)
    if pooled_weights.shape != (x_fit.shape[1],) or pooled_intercept_values.shape != (1,):
        raise ProtocolError("Shared sklearn pooled fit returned an unexpected parameter shape.")

    if gamma_value == 0.0:
        weights = pooled_weights.copy()
        intercept = float(pooled_intercept_values[0])
        terms = conditional_logit_objective_terms(
            np.concatenate([weights, np.asarray([intercept])]),
            x_fit,
            y_fit,
            prepared.penalty_operator,
            C=prepared.classifier_spec.C,
            gamma=0.0,
        )
        gradient_inf_norm = _gradient_inf_norm(terms.gradient)
        return AlignmentFitResult(
            gamma=0.0,
            predictions=np.asarray(getattr(pooled, "predictions"), dtype=np.int64).copy(),
            probabilities=np.asarray(getattr(pooled, "probabilities"), dtype=np.float64).copy(),
            coefficients=weights,
            intercept=intercept,
            classes=tuple(int(value) for value in getattr(pooled, "classes")),
            n_iter=tuple(int(value) for value in getattr(pooled, "n_iter")),
            converged=bool(getattr(pooled, "converged")),
            backend="sklearn_lbfgs",
            optimizer_success=bool(getattr(pooled, "converged")),
            optimizer_status=0 if bool(getattr(pooled, "converged")) else 1,
            optimizer_message=(
                "shared_sklearn_gamma0_converged"
                if bool(getattr(pooled, "converged"))
                else "shared_sklearn_gamma0_nonconverged"
            ),
            n_function_evaluations=0,
            n_gradient_evaluations=0,
            objective=float(terms.objective),
            mean_log_loss=float(terms.mean_log_loss),
            l2_penalty=float(terms.l2_penalty),
            alignment_penalty=0.0,
            unscaled_alignment_value=float(terms.unscaled_alignment_value),
            gradient_inf_norm=gradient_inf_norm,
            classifier_config_hash=prepared.classifier_spec.config_hash,
            scaler_state_hash=prepared.scaler_state_hash,
            penalty_operator_hash=prepared.penalty_operator.factor_hash,
            fit_identity=_fit_identity(prepared, 0.0),
        )

    try:
        from scipy.optimize import minimize  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - direct dependency
        raise RuntimeError("Positive-gamma conditional-logit fitting requires SciPy.") from exc

    initial = np.concatenate([pooled_weights, pooled_intercept_values])

    def objective(parameters: object) -> tuple[float, object]:
        return conditional_logit_objective_and_gradient(
            parameters,
            x_fit,
            y_fit,
            prepared.penalty_operator,
            C=prepared.classifier_spec.C,
            gamma=gamma_value,
        )

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={
            "ftol": float(optimizer.ftol_float64_eps_multiplier)
            * float(np.finfo(np.float64).eps),
            "gtol": float(optimizer.tol),
            "maxiter": int(optimizer.max_iter),
            "maxls": int(optimizer.max_line_search_steps),
        },
    )
    parameters = np.asarray(result.x, dtype=np.float64)
    weights = parameters[:-1].copy()
    intercept = float(parameters[-1])
    terms = conditional_logit_objective_terms(
        parameters,
        x_fit,
        y_fit,
        prepared.penalty_operator,
        C=prepared.classifier_spec.C,
        gamma=gamma_value,
    )
    gradient_inf_norm = _gradient_inf_norm(terms.gradient)
    finite = (
        np.all(np.isfinite(parameters))
        and math.isfinite(float(terms.objective))
        and math.isfinite(gradient_inf_norm)
    )
    converged = bool(
        result.success
        and finite
        and gradient_inf_norm <= float(optimizer.gradient_inf_norm_max)
        and int(getattr(result, "nit", optimizer.max_iter)) < int(optimizer.max_iter)
    )
    decision = x_eval @ weights + intercept
    probability_positive = _sigmoid(decision)
    probabilities = np.column_stack((1.0 - probability_positive, probability_positive))
    predictions = (decision > 0.0).astype(np.int64)
    return AlignmentFitResult(
        gamma=gamma_value,
        predictions=predictions,
        probabilities=probabilities,
        coefficients=weights,
        intercept=intercept,
        classes=(0, 1),
        n_iter=(int(getattr(result, "nit", 0)),),
        converged=converged,
        backend="scipy_lbfgsb",
        optimizer_success=bool(result.success),
        optimizer_status=int(getattr(result, "status", -1)),
        optimizer_message=str(getattr(result, "message", "")),
        n_function_evaluations=int(getattr(result, "nfev", 0)),
        n_gradient_evaluations=int(getattr(result, "njev", 0)),
        objective=float(terms.objective),
        mean_log_loss=float(terms.mean_log_loss),
        l2_penalty=float(terms.l2_penalty),
        alignment_penalty=float(terms.alignment_penalty),
        unscaled_alignment_value=float(terms.unscaled_alignment_value),
        gradient_inf_norm=gradient_inf_norm,
        classifier_config_hash=prepared.classifier_spec.config_hash,
        scaler_state_hash=prepared.scaler_state_hash,
        penalty_operator_hash=prepared.penalty_operator.factor_hash,
        fit_identity=_fit_identity(prepared, gamma_value),
    )


def conditional_logit_objective_and_gradient(
    parameters: Sequence[float],
    x_fit_scaled: Sequence[Sequence[float]],
    y_fit: Sequence[int],
    penalty_operator: ConditionalPenaltyOperator,
    *,
    C: float,
    gamma: float,
) -> tuple[float, object]:
    """Return the exact mean-loss objective and analytic gradient.

    The last parameter is the unpenalized intercept.  The coefficient terms are
    ``||w||^2 / (2 C N) + gamma * ||R w||^2``.
    """

    terms = conditional_logit_objective_terms(
        parameters,
        x_fit_scaled,
        y_fit,
        penalty_operator,
        C=C,
        gamma=gamma,
    )
    return terms.objective, terms.gradient


def conditional_logit_objective_terms(
    parameters: Sequence[float],
    x_fit_scaled: Sequence[Sequence[float]],
    y_fit: Sequence[int],
    penalty_operator: ConditionalPenaltyOperator,
    *,
    C: float,
    gamma: float,
) -> ConditionalObjectiveTerms:
    """Return the frozen objective decomposition and analytic gradient."""

    import numpy as np  # type: ignore

    x = np.asarray(x_fit_scaled, dtype=np.float64)
    y = np.asarray(y_fit, dtype=np.float64)
    theta = np.asarray(parameters, dtype=np.float64)
    gamma_value = _validated_gamma(gamma)
    if not math.isfinite(float(C)) or float(C) <= 0.0:
        raise ValueError("Conditional-logit C must be finite and positive.")
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("Conditional-logit objective requires a nonempty 2D fit array.")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError("Conditional-logit objective labels must align with fit rows.")
    if theta.ndim != 1 or theta.shape[0] != x.shape[1] + 1:
        raise ValueError(
            "Conditional-logit parameters must contain one coefficient per feature plus intercept."
        )
    if set(int(value) for value in y.tolist()) != {0, 1}:
        raise ValueError("Conditional-logit objective requires binary labels 0/1.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(theta)):
        raise ValueError("Conditional-logit objective inputs must be finite.")
    weights = theta[:-1]
    intercept = float(theta[-1])
    scores = x @ weights + intercept
    mean_log_loss = float(np.mean(np.logaddexp(0.0, scores) - y * scores))
    n_fit = int(x.shape[0])
    l2_penalty = float(weights @ weights) / (2.0 * float(C) * float(n_fit))
    unscaled_alignment = float(penalty_operator.value(weights))
    alignment_penalty = gamma_value * unscaled_alignment
    residual = _sigmoid(scores) - y
    gradient_weights = (
        (x.T @ residual) / float(n_fit)
        + weights / (float(C) * float(n_fit))
        + gamma_value
        * np.asarray(penalty_operator.gradient(weights), dtype=np.float64)
    )
    gradient_intercept = float(np.mean(residual))
    gradient = np.concatenate(
        [gradient_weights, np.asarray([gradient_intercept], dtype=np.float64)]
    )
    objective = mean_log_loss + l2_penalty + alignment_penalty
    if not math.isfinite(objective) or not np.all(np.isfinite(gradient)):
        raise ValueError("Conditional-logit objective or gradient became non-finite.")
    return ConditionalObjectiveTerms(
        objective=float(objective),
        mean_log_loss=mean_log_loss,
        l2_penalty=l2_penalty,
        alignment_penalty=float(alignment_penalty),
        unscaled_alignment_value=unscaled_alignment,
        gradient=gradient,
    )


def _validate_fold_and_spec(
    fold_data: ConditionalLogitFold, spec: ClassifierSpec
) -> None:
    if spec.to_payload() != canonical_classifier_spec().to_payload():
        raise ProtocolError(
            "Conditional-logit alignment requires the frozen C=0.01, unweighted "
            "sklearn-lbfgs classifier specification."
        )
    if fold_data.outer_target_center in set(fold_data.fit_centers):
        raise ProtocolError("Outer target center leaked into a conditional-logit fit.")
    if (
        fold_data.inner_pseudo_target_center is not None
        and fold_data.inner_pseudo_target_center in set(fold_data.fit_centers)
    ):
        raise ProtocolError("Inner pseudo-target center leaked into a conditional-logit fit.")
    if len(fold_data.fit_labels) != len(fold_data.fit_domains):
        raise ProtocolError("Conditional-logit fold labels/domains are misaligned.")
    if set(fold_data.fit_labels) != {0, 1}:
        raise ProtocolError("Conditional-logit fit requires both binary classes.")
    if set(fold_data.eval_labels) != {0, 1}:
        raise ProtocolError("Conditional-logit evaluation requires both binary classes.")


def _validate_optimizer(config: AlignmentOptimizerConfig) -> None:
    values = (
        config.tol,
        config.gradient_inf_norm_max,
        config.objective_atol,
        config.coefficient_atol,
        config.coefficient_rtol,
        config.probability_atol,
        config.probability_rtol,
    )
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in values):
        raise ValueError("Conditional-logit optimizer tolerances must be finite and nonnegative.")
    if (
        config.tol <= 0.0
        or config.gradient_inf_norm_max <= 0.0
        or config.ftol_float64_eps_multiplier <= 0
        or config.max_line_search_steps <= 0
        or config.max_iter <= 0
    ):
        raise ValueError("Conditional-logit optimizer iteration controls must be positive.")


def _validated_gamma(gamma: float) -> float:
    value = float(gamma)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("Conditional-logit gamma must be finite and nonnegative.")
    return value


def _sigmoid(values: object) -> object:
    import numpy as np  # type: ignore

    array = np.asarray(values, dtype=np.float64)
    output = np.empty_like(array)
    positive = array >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exp_values = np.exp(array[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def _gradient_inf_norm(gradient: object) -> float:
    import numpy as np  # type: ignore

    values = np.asarray(gradient, dtype=np.float64)
    return float(np.max(np.abs(values))) if values.size else 0.0


def _fit_identity(prepared: PreparedConditionalLogit, gamma: float) -> str:
    return stable_hash(
        {
            "method": "conditional_logit_alignment",
            "training_frame_hash": prepared.training_frame_hash,
            "fit_row_hash": prepared.fit_row_hash,
            "scaler_state_hash": prepared.scaler_state_hash,
            "factor_hash": prepared.penalty_operator.factor_hash,
            "classifier_config_hash": prepared.classifier_spec.config_hash,
            "gamma": float(gamma),
        }
    )


__all__ = [
    "AlignmentFitResult",
    "ConditionalObjectiveTerms",
    "PreparedConditionalLogit",
    "conditional_logit_objective_and_gradient",
    "conditional_logit_objective_terms",
    "fit_conditional_logit",
    "fit_prepared_conditional_logit",
    "prepare_conditional_logit",
]
