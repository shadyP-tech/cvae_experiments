"""Bounded-memory ridge fits for posterior, mass allocation and total mass."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from ....protocol import ProtocolError
from ..hashing import canonical_hash
from .design import baseline, case_design, full_scope_case_weights, raw_design, standardized, CLIP
from .targets import correction_targets

RIDGE_ALPHA = .1


def _minimize(objective, width):
    result = minimize(objective, np.zeros(width), jac=True, method='L-BFGS-B',
                      options={'maxiter':128, 'ftol':1e-10, 'gtol':1e-7, 'maxcor':8})
    if not result.success or not np.isfinite(result.x).all():
        raise ProtocolError('HARP v21 correction evidence fit did not converge.')
    return result.x


def fit_correction_evidence(menus, label_reader, *, variant='embedding_residual'):
    from .model import PatchEvidenceModel
    rows = tuple(sorted(menus, key=lambda m:(m.center_id, m.case_id)))
    keys = tuple((m.center_id, m.case_id) for m in rows)
    if (not rows or len(set(keys)) != len(keys)
        or any(m.surface_role.value != 'SOURCE_TRAIN_DEVELOPMENT' for m in rows)):
        raise ProtocolError('HARP v21 correction evidence requires unique source-development menus.')
    # All features are authenticated/validated before the first label callback.
    raw_rows = tuple(raw_design(m, variant) for m in rows)
    labels = tuple(np.asarray(label_reader(m)) for m in rows)
    totals, normalized, v = correction_targets(rows, labels)
    case_w = full_scope_case_weights(rows)
    patch_w = np.concatenate([np.full(len(m.sample_ids), w/len(m.sample_ids))
                              for m, w in zip(rows, case_w, strict=True)])
    raw = np.concatenate(raw_rows)
    means = patch_w @ raw
    scales = np.sqrt(patch_w @ ((raw-means)**2))
    scales[scales < 1e-8] = 1.
    x = standardized(raw, means, scales, variant)
    b = np.concatenate([baseline(m) for m in rows])
    bc = np.clip(b, CLIP, 1-CLIP)
    offsets = np.log(bc/(1-bc))
    y = np.concatenate(labels)
    posterior_x = np.column_stack((np.ones(len(x)), x))
    posterior_beta = np.zeros(posterior_x.shape[1])
    if variant != 'baseline':
        def posterior_objective(beta):
            z = offsets + posterior_x @ beta
            value = patch_w @ (np.logaddexp(0, z)-y*z) + .5*RIDGE_ALPHA*(beta[1:]@beta[1:])
            grad = posterior_x.T @ (patch_w*(expit(z)-y))
            grad[1:] += RIDGE_ALPHA*beta[1:]
            return float(value), grad
        posterior_beta = _minimize(posterior_objective, posterior_x.shape[1])
    lengths = [len(m.sample_ids) for m in rows]
    boundaries = np.cumsum([0, *lengths])
    mass_betas = []
    for k in (0, 1):
        # An additional v_ck weight is essential: u estimates normalized E[T].
        weights = case_w*v[:, k]*(totals[:, k] > 0)
        weights /= weights.sum()
        anchor = np.log(bc if k else 1-bc)
        def mass_objective(beta):
            z = anchor + x @ beta
            residual = np.zeros(len(x))
            value = .5*RIDGE_ALPHA*(beta@beta)
            for i, weight in enumerate(weights):
                if weight == 0:
                    continue
                sl = slice(boundaries[i], boundaries[i+1])
                logu = z[sl]-logsumexp(z[sl])
                q = normalized[i][:, k]
                value -= weight*(q@logu)
                residual[sl] = weight*(np.exp(logu)-q)
            return float(value), x.T@residual+RIDGE_ALPHA*beta
        mass_betas.append(_minimize(mass_objective, x.shape[1]))
    # Only six own-case baseline summaries; no target cohort or identity input.
    case_x = np.stack([case_design(m) for m in rows])
    penalty = np.eye(case_x.shape[1])*RIDGE_ALPHA
    penalty[0, 0] = 0.
    tau = np.linalg.solve(case_x.T@(case_w[:, None]*case_x)+penalty,
                          case_x.T@(case_w[:, None]*totals)).T
    normalization = tuple((c, sum(m.center_id == c for m in rows),
                            int(sum(m.center_id == c and np.any(y == 0) for m,y in zip(rows, labels, strict=True))),
                            int(sum(m.center_id == c and np.any(y == 1) for m,y in zip(rows, labels, strict=True))))
                           for c in sorted({m.center_id for m in rows}))
    return PatchEvidenceModel(keys, tuple(m.menu_hash for m in rows), variant,
        tuple(map(float, means)), tuple(map(float, scales)), tuple(map(float, posterior_beta)),
        tuple(tuple(map(float, a)) for a in mass_betas), tuple(tuple(map(float, a)) for a in tau),
        normalization, tuple(map(float, case_w)), canonical_hash({'normalization':normalization,'case_keys':keys}))
