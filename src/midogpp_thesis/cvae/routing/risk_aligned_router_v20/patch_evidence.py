"""Case-excluded class evidence from a fixed, label-free Virchow2 sketch.

The sketch is not fitted on source or target data. Standardization and the
regularized logistic model are fitted only within the owning training scope.
Evidence is a probability estimate, never an action-level safety certificate.
"""
from dataclasses import dataclass, field
from collections import Counter
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from ...protocol import ProtocolError
from .hashing import canonical_hash

PATCH_DIMENSION = 64
PATCH_SKETCH_SEED = 20020
PATCH_SCHEMA = 'harp_v20_virchow2_fixed_countsketch64'


def sketch_virchow2(features):
    x = np.asarray(features)
    if x.ndim != 2 or x.shape[1] != 3840 or not np.isfinite(x).all():
        raise ProtocolError('HARP v20 patch sketch requires canonical finite Virchow2_3840 rows.')
    rng = np.random.default_rng(PATCH_SKETCH_SEED)
    order = rng.permutation(3840)
    signs = rng.choice(np.array([-1., 1.]), size=3840)
    result = np.empty((len(x), PATCH_DIMENSION), dtype=np.float32)
    for j in range(PATCH_DIMENSION):
        indices = order[j::PATCH_DIMENSION]
        result[:, j] = np.sum(x[:, indices].astype(np.float64) * signs[indices], axis=1) / np.sqrt(len(indices))
    return result


def patch_array(menu):
    x = np.asarray(menu.patch_features, dtype=np.float64)
    if x.shape != (len(menu.sample_ids), PATCH_DIMENSION) or not np.isfinite(x).all():
        raise ProtocolError('HARP v20 requires sample-aligned, sealed patch features.')
    return x


@dataclass(frozen=True, slots=True)
class PatchEvidenceModel:
    training_case_keys: tuple
    training_menu_hashes: tuple
    means: tuple
    scales: tuple
    coefficients: tuple
    model_hash: str = field(init=False)

    def __post_init__(self):
        if (not self.training_case_keys or len(set(self.training_case_keys)) != len(self.training_case_keys)
            or len(self.training_menu_hashes) != len(self.training_case_keys)
            or len(self.means) != PATCH_DIMENSION or len(self.scales) != PATCH_DIMENSION
            or len(self.coefficients) != PATCH_DIMENSION+1
            or not np.isfinite((*self.means,*self.scales,*self.coefficients)).all()
            or min(self.scales) <= 0):
            raise ProtocolError('HARP v20 patch evidence model is malformed.')
        object.__setattr__(self, 'model_hash', canonical_hash(self._payload()))

    def predict(self, menu):
        if (menu.center_id,menu.case_id) in self.training_case_keys:
            raise ProtocolError('HARP v20 patch evidence prediction includes a fitted case.')
        x = (patch_array(menu)-self.means)/self.scales
        return np.clip(expit(self.coefficients[0] + x @ np.asarray(self.coefficients[1:])),1e-6,1-1e-6)

    def _payload(self):
        return dict(schema_version=PATCH_SCHEMA, training_case_keys=self.training_case_keys,
            training_menu_hashes=self.training_menu_hashes, means=self.means, scales=self.scales,
            coefficients=self.coefficients, ridge_alpha=.01, source_only=True,
            raw_labels_persisted=False, model_is_safety_bound=False)

    def public_payload(self):
        return {**self._payload(),'model_hash':self.model_hash}


def fit_patch_evidence(menus, label_reader):
    """Called by the scoped truth capability; labels never escape the fit."""
    rows = tuple(menus)
    if not rows or any(m.surface_role.value != 'SOURCE_TRAIN_DEVELOPMENT' for m in rows):
        raise ProtocolError('HARP v20 class evidence requires source-development menus.')
    counts = Counter(m.center_id for m in rows)
    x = np.concatenate([patch_array(m) for m in rows])
    y = np.concatenate([label_reader(m) for m in rows])
    w = np.concatenate([np.full(len(m.sample_ids),1/len(counts)/counts[m.center_id]/len(m.sample_ids)) for m in rows])
    means = w @ x
    scales = np.sqrt(w @ ((x-means)**2)); scales[scales<1e-8]=1.
    x = np.column_stack((np.ones(len(x)),(x-means)/scales))
    def objective(beta):
        z=x@beta; p=expit(z)
        value=w@(np.logaddexp(0,z)-y*z)+.005*np.dot(beta[1:],beta[1:])
        grad=x.T@(w*(p-y));grad[1:]+=.01*beta[1:]
        return float(value),grad
    beta=np.zeros(x.shape[1]);prev=np.clip(w@y,1e-6,1-1e-6);beta[0]=np.log(prev/(1-prev))
    result=minimize(objective,beta,jac=True,method='L-BFGS-B',options={'maxiter':256,'ftol':1e-11})
    if not result.success or not np.isfinite(result.x).all():
        raise ProtocolError('HARP v20 patch evidence fit did not converge.')
    return PatchEvidenceModel(tuple((m.center_id,m.case_id) for m in rows),tuple(m.menu_hash for m in rows),
        tuple(map(float,means)),tuple(map(float,scales)),tuple(map(float,result.x)))


EVIDENCE_FEATURE_NAMES = ('patch_d01_positive_mean','patch_d10_negative_mean',
    'patch_d01_positive_q10','patch_d10_negative_q10','patch_expected_brier_delta',
    'patch_expected_logloss_delta','patch_baseline_disagreement')


def evidence_descriptor(menu, composite, probability):
    from .contracts import decode_probability_hex
    p=np.asarray(probability,dtype=float)
    if p.shape!=(len(menu.sample_ids),) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ProtocolError('HARP v20 patch evidence probability alignment failed.')
    b=np.asarray(decode_probability_hex(menu.baseline_probability_hex));a=np.asarray(decode_probability_hex(composite.probability_hex))
    d01=(b<.5)&(a>=.5);d10=(b>=.5)&(a<.5)
    def stats(v):return (float(v.mean()),float(np.quantile(v,.1))) if len(v) else (0.,0.)
    m01,q01=stats(p[d01]);m10,q10=stats(1-p[d10])
    ac=np.clip(a,1e-6,1-1e-6);bc=np.clip(b,1e-6,1-1e-6)
    # Exact conditional expectations for a Bernoulli outcome with posterior p.
    # Every probability change contributes, including non-flipping patches.
    brier=np.mean((a-b)*(a+b-2*p))
    logloss=np.mean(-p*np.log(ac/bc)-(1-p)*np.log((1-ac)/(1-bc)))
    return (m01,m10,q01,q10,float(brier),float(logloss),float(((p>=.5)!=(b>=.5)).mean()))

@dataclass(frozen=True, slots=True)
class HeldPatchEvidence:
    case_key: tuple
    menu_hash: str
    training_case_keys: tuple
    model_hash: str
    probabilities: tuple
    prediction_hash: str = field(init=False)

    def __post_init__(self):
        from .hashing import require_sha256
        require_sha256(self.menu_hash, name='patch menu hash')
        require_sha256(self.model_hash, name='patch evidence model hash')
        if (self.case_key in self.training_case_keys or not self.training_case_keys
            or not self.probabilities or not np.isfinite(self.probabilities).all()
            or any(not 0 <= p <= 1 for p in self.probabilities)):
            raise ProtocolError('HARP v20 held patch evidence leaked its case or is malformed.')
        object.__setattr__(self,'prediction_hash',canonical_hash(self.public_body()))

    def public_body(self):
        return dict(case_key=self.case_key,menu_hash=self.menu_hash,training_case_keys=self.training_case_keys,
                    model_hash=self.model_hash,probabilities=self.probabilities)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.probabilities,dtype=dtype)


def seal_patch_evidence(model, menu):
    return HeldPatchEvidence((menu.center_id,menu.case_id),menu.menu_hash,model.training_case_keys,
        model.model_hash,tuple(map(float,model.predict(menu))))
