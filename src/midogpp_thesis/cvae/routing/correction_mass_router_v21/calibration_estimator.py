"""Small binary harm regression with case-normalized mean loss."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from ...protocol import ProtocolError


def fit_harm_logistic(matrix, response, weights, *, alpha):
    x, y, w = (np.asarray(v, dtype=float) for v in (matrix, response, weights))
    if (x.ndim != 2 or y.shape != (len(x),) or w.shape != y.shape
        or not len(x) or not np.isin(y, (0, 1)).all() or np.any(w <= 0)
        or not np.isfinite(x).all() or not np.isfinite(w).all() or alpha <= 0):
        raise ProtocolError("HARP v21 binary calibration inputs are malformed.")
    w = w / w.sum()
    initial = np.zeros(x.shape[1])
    prevalence = np.clip(w @ y, 1e-6, 1-1e-6)
    initial[0] = np.log(prevalence / (1-prevalence))
    if len(np.unique(y)) == 1:
        return initial
    def objective(beta):
        logits = x @ beta
        loss = w @ (np.logaddexp(0, logits)-y*logits) + .5*alpha*(beta[1:] @ beta[1:])
        gradient = x.T @ (w*(expit(logits)-y))
        gradient[1:] += alpha*beta[1:]
        return float(loss), gradient
    result = minimize(objective, initial, jac=True, method="L-BFGS-B",
        options={"maxiter": 256, "ftol": 1e-11, "gtol": 1e-7})
    if not result.success or not np.isfinite(result.x).all():
        raise ProtocolError(f"HARP v21 harm calibration did not converge: {result.message}")
    return result.x
