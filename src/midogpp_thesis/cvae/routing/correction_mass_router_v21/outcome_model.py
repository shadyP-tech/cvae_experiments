"""Exact action effects from shared correction masses, without free outcome heads."""
from dataclasses import dataclass, field
import numpy as np
from ...protocol import ProtocolError
from .contracts import CompositeKind, decode_probability_hex
from .hashing import canonical_hash
from .patch_evidence import PatchEvidenceModel


@dataclass(frozen=True, slots=True)
class ActionOutcomePrediction:
    composite_hash: str
    predicted_gain: float
    predicted_harm: float
    predicted_brier_delta: float
    predicted_logloss_delta: float
    safe_positive_probability: float
    predicted_class_0_gain: float
    predicted_class_1_gain: float
    approximate_gain_lower_score: float
    risk_adjusted_score: float = 0.
    remaining_probability: float = 1.
    safe_gain_magnitude: float = 0.
    harm_gain_magnitude: float = 0.

    def public_payload(self):
        return {**{name:getattr(self, name) for name in self.__dataclass_fields__},
                'per_action_safety_guarantee':False, 'lower_score_is_model_based':True,
                'approximate_gain_lower_score_deprecated':True,
                'risk_adjusted_score_is_confidence_bound':False,
                'candidate_harm_probability_estimated':False,
                'auxiliary_category_fields_unused':True,
                'class_gain_fields_are_source_normalized_contributions':True}


def effect_arrays(baseline, actions, posterior, masses):
    """Vectorized mean effects, valid without an independent-patch assumption."""
    b, a, p, m = map(lambda value:np.asarray(value, dtype=float),
                      (baseline, actions, posterior, masses))
    if (b.ndim != 1 or p.shape != b.shape or a.ndim != 2 or a.shape[1:] != b.shape
        or m.shape != (len(b), 2) or not len(b)
        or any(not np.isfinite(x).all() for x in (b, a, p, m))
        or any(np.any((x < 0)|(x > 1)) for x in (b, a, p)) or np.any(m < 0)):
        raise ProtocolError('HARP v21 correction effect inputs are malformed.')
    delta = (a >= .5).astype(float)-(b >= .5).astype(float)
    g0, g1 = -(delta@m[:, 0]), delta@m[:, 1]
    brier = np.mean((a-b)*(a+b-2*p), axis=1)
    ac, bc = np.clip(a, 1e-6, 1-1e-6), np.clip(b, 1e-6, 1-1e-6)
    logloss = np.mean(-p*np.log(ac/bc)-(1-p)*np.log((1-ac)/(1-bc)), axis=1)
    return .5*(g0+g1), g0, g1, brier, logloss


@dataclass(frozen=True, slots=True)
class ActionOutcomeModel:
    patch_model: PatchEvidenceModel
    model_hash: str = field(init=False)

    def __post_init__(self):
        if not isinstance(self.patch_model, PatchEvidenceModel):
            raise ProtocolError('HARP v21 action effects require a frozen correction evidence model.')
        object.__setattr__(self, 'model_hash', canonical_hash(self._payload()))

    @property
    def training_case_keys(self):
        return self.patch_model.training_case_keys

    @property
    def empty_population(self):
        return False  # No candidate fitting population exists in this architecture.

    def _payload(self):
        return dict(schema_version='harp_v21_correction_mass_exact_action_effects',
            patch_evidence_model=self.patch_model.public_payload(),
            gain_formula='0.5*sum(delta_hard*(mass_1-mass_0))',
            proper_loss_population='all_probability_changes',
            candidate_outcome_heads_fitted=False, candidate_harm_probability_estimated=False,
            per_action_safety_guarantee=False, center_id_is_model_feature=False,
            class_gain_fields_are_source_normalized_contributions=True)

    def public_payload(self):
        return {**self._payload(), 'model_hash':self.model_hash}

    @classmethod
    def from_payload(cls, payload):
        try:
            model = cls(PatchEvidenceModel.from_payload(payload['patch_evidence_model']))
            if canonical_hash(model.public_payload()) != canonical_hash(payload):
                raise ProtocolError('HARP v21 action reconstruction seal differs.')
            return model
        except ProtocolError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError('HARP v21 action reconstruction is malformed.') from exc

    def predict_composites(self, menu, composites):
        rows = tuple(composites)
        p = self.patch_model.predict(menu)  # Held-case check applies even to B.
        m = self.patch_model.predict_masses(menu)
        if not rows:
            return ()
        if any(c.menu_hash != menu.menu_hash or c.sample_ids != menu.sample_ids
               or c.baseline_probability_hex != menu.baseline_probability_hex
               or (c.center_id, c.case_id) != (menu.center_id, menu.case_id) for c in rows):
            raise ProtocolError('HARP v21 correction effects crossed a sealed menu.')
        b = np.asarray(decode_probability_hex(menu.baseline_probability_hex))
        a = np.stack([decode_probability_hex(c.probability_hex) for c in rows])
        gain, g0, g1, brier, logloss = effect_arrays(b, a, p, m)
        result = []
        for i, c in enumerate(rows):
            if c.kind is CompositeKind.B:
                result.append(ActionOutcomePrediction(c.composite_hash, 0., 0., 0., 0., 0., 0., 0., 0.))
            else:
                result.append(ActionOutcomePrediction(c.composite_hash, float(gain[i]), 0.,
                    float(brier[i]), float(logloss[i]), 0., float(g0[i]), float(g1[i]),
                    float(gain[i]), float(gain[i])))
        return tuple(result)


def fit_action_outcome_model(menus, composites=(), outcomes=(), *, evidence_model=None,
                             patch_model=None, **unused_legacy_arguments):
    """Wrap evidence; composite outcomes are deliberately no longer fit targets."""
    menus = tuple(menus)
    model = evidence_model if evidence_model is not None else patch_model
    keys = tuple(sorted((m.center_id, m.case_id) for m in menus))
    if not isinstance(model, PatchEvidenceModel) or keys != model.training_case_keys:
        raise ProtocolError('HARP v21 action model requires exact scoped correction evidence.')
    if tuple(m.menu_hash for m in sorted(menus, key=lambda m:(m.center_id,m.case_id))) != model.training_menu_hashes:
        raise ProtocolError('HARP v21 action evidence menu identity differs.')
    return ActionOutcomeModel(model)


__all__ = ('ActionOutcomePrediction', 'ActionOutcomeModel', 'fit_action_outcome_model', 'effect_arrays')
