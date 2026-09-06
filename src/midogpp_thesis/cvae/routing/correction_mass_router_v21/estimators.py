"""Small deterministic estimators on an explicitly normalized mean-loss scale.

Intercepts are unpenalized. Candidate rows never change the effective penalty:
weights are normalized before every fit, including conditional magnitude fits.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ...protocol import ProtocolError

DEFAULT_RIDGE_ALPHA = 0.01
CATEGORY_NAMES = ("SAFE_POSITIVE", "HARM", "OTHER")


def _inputs(matrix, response, weights, alpha):
    x, y, w = (np.asarray(value, dtype=np.float64) for value in (matrix, response, weights))
    if (x.ndim != 2 or len(x) == 0 or x.shape[1] < 1 or y.shape != (len(x),)
        or w.shape != (len(x),) or np.any(w < 0) or w.sum() <= 0
        or not np.all(x[:, 0] == 1) or not np.isfinite(alpha) or alpha <= 0
        or not all(np.isfinite(value).all() for value in (x, y, w))):
        raise ProtocolError("HARP v21 mean-loss estimator inputs are malformed.")
    return x, y, w / w.sum()


def standardize_design(matrix, weights):
    """Return transformed matrix, nonintercept means and scales."""
    x, _, w = _inputs(matrix, np.zeros(len(matrix)), weights, DEFAULT_RIDGE_ALPHA)
    means = w @ x[:, 1:]
    scales = np.sqrt(w @ ((x[:, 1:] - means) ** 2))
    scales[scales <= np.sqrt(np.finfo(float).eps)] = 1.0
    result = x.copy()
    result[:, 1:] = (x[:, 1:] - means) / scales
    return result, means, scales


def fit_mean_ridge(matrix, response, weights, *, alpha=DEFAULT_RIDGE_ALPHA):
    """Minimize .5 E_w[(y-X beta)^2] + .5 alpha ||beta_nonintercept||²."""
    x, y, w = _inputs(matrix, response, weights, alpha)
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0.0
    normal = x.T @ (w[:, None] * x) + penalty
    right = x.T @ (w * y)
    try:
        result = np.linalg.solve(normal, right)
    except np.linalg.LinAlgError:
        result = np.linalg.lstsq(normal, right, rcond=None)[0]
    if not np.isfinite(result).all():
        raise ProtocolError("HARP v21 mean-loss ridge fit is nonfinite.")
    return result


def predict_softmax(matrix, coefficients):
    logits = np.asarray(matrix, dtype=float) @ np.asarray(coefficients, dtype=float).T
    if logits.ndim != 2 or logits.shape[1] != 3 or not np.isfinite(logits).all():
        raise ProtocolError("HARP v21 categorical prediction is malformed.")
    logits -= logits.max(axis=1, keepdims=True)
    exponentials = np.exp(logits)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def fit_softmax_ridge(matrix, targets, weights, *, alpha=DEFAULT_RIDGE_ALPHA):
    """Fit one coherent S/H/O distribution, with no synthetic training rows.

    Missing categories receive a fixed numerical floor in the intercept-only
    degenerate case; this is a probability estimate, never a safety bound.
    """
    x, y, w = _inputs(matrix, targets, weights, alpha)
    if not np.isin(y, (0, 1, 2)).all():
        raise ProtocolError("HARP v21 category targets must encode S/H/O.")
    y = y.astype(np.int64)
    prevalence = np.bincount(y, weights=w, minlength=3)
    initial = np.zeros((3, x.shape[1]))
    initial[:, 0] = np.log(np.clip(prevalence, 1e-8, 1.0))
    if np.count_nonzero(prevalence) == 1:
        return initial
    targets_onehot = np.eye(3)[y]

    def objective(flat):
        beta = flat.reshape(initial.shape)
        logits = x @ beta.T
        maximum = logits.max(axis=1)
        logsum = maximum + np.log(np.exp(logits - maximum[:, None]).sum(axis=1))
        value = np.dot(w, logsum - logits[np.arange(len(x)), y])
        value += .5 * alpha * np.sum(beta[:, 1:] ** 2)
        gradient = (w[:, None] * (predict_softmax(x, beta) - targets_onehot)).T @ x
        gradient[:, 1:] += alpha * beta[:, 1:]
        return float(value), gradient.ravel()

    result = minimize(objective, initial.ravel(), jac=True, method="L-BFGS-B",
                      options={"maxiter": 256, "ftol": 1e-11, "gtol": 1e-7, "maxls": 40})
    if not result.success or not np.isfinite(result.x).all():
        raise ProtocolError(f"HARP v21 categorical mean-loss fit did not converge: {result.message}")
    return result.x.reshape(initial.shape)


__all__ = ("CATEGORY_NAMES", "DEFAULT_RIDGE_ALPHA", "fit_mean_ridge",
           "fit_softmax_ridge", "predict_softmax", "standardize_design")
