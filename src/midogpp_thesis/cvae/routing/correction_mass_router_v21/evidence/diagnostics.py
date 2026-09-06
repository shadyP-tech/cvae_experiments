"""Aggregate post-seal evidence diagnostics; raw labels never leave capability."""
import numpy as np
from ....protocol import ProtocolError
from ..contracts import decode_probability_hex
from ..outcome_model import effect_arrays


def correction_diagnostics(menu, evidence, labels, normalizer, composites=()):
    p = np.asarray(evidence.probabilities)
    m = np.asarray(evidence.normalized_masses)
    b = np.asarray(decode_probability_hex(menu.baseline_probability_hex))
    _, n, s0, s1 = next(row for row in normalizer.center_counts if row[0] == menu.center_id)
    target = np.zeros_like(m)
    for k, support in ((0, s0), (1, s1)):
        mask = labels == k
        if np.any(mask):
            target[mask, k] = n/support/mask.sum()
    pc = np.clip(p, 1e-6, 1-1e-6)
    result = dict(evidence_variant=evidence.evidence_variant,
        held_evidence_prediction_hash=evidence.prediction_hash,
        fit_normalization_hash=evidence.fit_normalization_hash,
        scoring_normalization_hash=normalizer.normalization_hash,
        mass_calibration_population='held_scoring_scope_normalization',
        estimated_mass_by_class=m.sum(axis=0).tolist(),
        observed_mass_by_class=target.sum(axis=0).tolist(),
        normalized_mass_squared_error_by_class=np.sum((m-target)**2, axis=0).tolist(),
        posterior_brier=float(np.mean((p-labels)**2)),
        posterior_logloss=float(-np.mean(labels*np.log(pc)+(1-labels)*np.log1p(-pc))),
        diagnostic_only=True, used_for_policy_selection=False, raw_labels_persisted=False)
    union01, union10 = np.zeros(len(p), dtype=bool), np.zeros(len(p), dtype=bool)
    action_rows = []
    for c in composites:
        if c.menu_hash != menu.menu_hash or c.sample_ids != menu.sample_ids:
            raise ProtocolError('HARP v21 correction diagnostic composite changed its sealed menu.')
        a = np.asarray(decode_probability_hex(c.probability_hex))
        union01 |= (b < .5)&(a >= .5)
        union10 |= (b >= .5)&(a < .5)
        estimated = effect_arrays(b, a[None, :], p, m)
        observed = effect_arrays(b, a[None, :], labels, target)
        action_rows.append(dict(composite_hash=c.composite_hash,
            estimated_gain=float(estimated[0][0]), observed_gain=float(observed[0][0]),
            gain_error=float(estimated[0][0]-observed[0][0]),
            estimated_brier_delta=float(estimated[3][0]), observed_brier_delta=float(observed[3][0]),
            estimated_logloss_delta=float(estimated[4][0]), observed_logloss_delta=float(observed[4][0])))
    def flip_summary(mask, k):
        if not np.any(mask):
            return dict(patch_count=0, mean_predicted_correctness=None, observed_correctness=None)
        return dict(patch_count=int(mask.sum()),
            mean_predicted_correctness=float(np.mean(p[mask] if k else 1-p[mask])),
            observed_correctness=float(np.mean(labels[mask] == k)))
    result['union_d01_flip_calibration'] = flip_summary(union01, 1)
    result['union_d10_flip_calibration'] = flip_summary(union10, 0)
    result['sealed_action_effect_diagnostics'] = action_rows
    return result
