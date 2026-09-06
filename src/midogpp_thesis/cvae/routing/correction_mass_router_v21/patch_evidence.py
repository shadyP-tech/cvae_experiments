"""Public correction evidence API and sealed held-case control predictions."""
from dataclasses import dataclass, field
import numpy as np
from ...protocol import ProtocolError
from .hashing import canonical_hash
from .evidence.design import PATCH_DIMENSION, PATCH_SCHEMA, patch_array, sketch_virchow2
from .evidence.model import PatchEvidenceModel
from .evidence.fitting import fit_correction_evidence


def fit_patch_evidence(menus, label_reader, *, variant="embedding_residual"):
    """Compatibility API; the model now estimates correction masses as well."""
    return fit_correction_evidence(menus, label_reader, variant=variant)


EVIDENCE_FEATURE_NAMES = ('patch_d01_positive_mean','patch_d10_negative_mean',
    'patch_d01_positive_q10','patch_d10_negative_q10','patch_expected_brier_delta',
    'patch_expected_logloss_delta','patch_baseline_disagreement')


def evidence_descriptor(menu, composite, probability):
    from .contracts import decode_probability_hex
    p=np.asarray(probability,dtype=float)
    if p.shape!=(len(menu.sample_ids),) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ProtocolError('HARP v21 patch evidence probability alignment failed.')
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
    normalized_masses: tuple
    evidence_variant: str
    fit_normalization_hash: str
    prediction_hash: str = field(init=False)

    def __post_init__(self):
        from .hashing import require_sha256
        require_sha256(self.menu_hash, name='patch menu hash')
        require_sha256(self.model_hash, name='patch evidence model hash')
        require_sha256(self.fit_normalization_hash, name='correction fit normalization hash')
        masses = np.asarray(self.normalized_masses)
        if (self.case_key in self.training_case_keys or not self.training_case_keys
            or not self.probabilities or not np.isfinite(self.probabilities).all()
            or any(not 0 <= p <= 1 for p in self.probabilities)
            or masses.shape != (len(self.probabilities), 2) or not np.isfinite(masses).all()
            or np.any(masses < 0)
            or self.evidence_variant not in ('baseline','calibrated_baseline','embedding_residual')):
            raise ProtocolError('HARP v21 held patch evidence leaked its case or is malformed.')
        object.__setattr__(self,'prediction_hash',canonical_hash(self.public_body()))

    def public_body(self):
        return dict(case_key=self.case_key,menu_hash=self.menu_hash,training_case_keys=self.training_case_keys,
                    model_hash=self.model_hash,probabilities=self.probabilities,
                    normalized_masses=self.normalized_masses,evidence_variant=self.evidence_variant,
                    fit_normalization_hash=self.fit_normalization_hash)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.probabilities,dtype=dtype)


def seal_patch_evidence(model, menu):
    return HeldPatchEvidence((menu.center_id,menu.case_id),menu.menu_hash,model.training_case_keys,
        model.model_hash,tuple(map(float,model.predict(menu))),
        tuple(tuple(map(float, row)) for row in model.predict_masses(menu)),
        model.variant,model.normalization_hash)
