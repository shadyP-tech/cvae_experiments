"""Immutable source-scoped model with exact-payload reconstruction."""
from dataclasses import dataclass, field
from collections import Counter
import numpy as np
from scipy.special import expit, softmax

from ....protocol import ProtocolError
from ..hashing import canonical_hash, require_sha256
from .design import PATCH_DIMENSION, PATCH_SCHEMA, VARIANTS, baseline, case_design, raw_design, standardized, CLIP


@dataclass(frozen=True, slots=True)
class PatchEvidenceModel:
    training_case_keys: tuple
    training_menu_hashes: tuple
    variant: str
    means: tuple
    scales: tuple
    coefficients: tuple
    mass_coefficients: tuple
    total_mass_coefficients: tuple
    source_support_counts: tuple
    case_weights: tuple
    normalization_hash: str
    model_hash: str = field(init=False)

    def __post_init__(self):
        width = PATCH_DIMENSION+1 if self.variant == 'embedding_residual' else 1
        numbers = (*self.means, *self.scales, *self.coefficients, *self.case_weights,
                   *(x for row in (*self.mass_coefficients, *self.total_mass_coefficients) for x in row))
        if (self.variant not in VARIANTS or not self.training_case_keys
            or len(set(self.training_case_keys)) != len(self.training_case_keys)
            or tuple(sorted(self.training_case_keys)) != self.training_case_keys
            or len(self.training_menu_hashes) != len(self.training_case_keys)
            or len(self.means) != width or len(self.scales) != width
            or len(self.coefficients) != width+1 or len(self.mass_coefficients) != 2
            or any(len(v) != width for v in self.mass_coefficients)
            or len(self.total_mass_coefficients) != 2 or any(len(v) != 6 for v in self.total_mass_coefficients)
            or len(self.case_weights) != len(self.training_case_keys) or min(self.case_weights) <= 0
            or abs(sum(self.case_weights)-1.) > 1e-10 or not np.isfinite(numbers).all()
            or min(self.scales) <= 0 or (self.variant == 'baseline' and any(self.coefficients))):
            raise ProtocolError('HARP v21 correction evidence model is malformed.')
        for value in (*self.training_menu_hashes, self.normalization_hash):
            require_sha256(value, name='correction evidence scope hash')
        counts = Counter(c for c, _ in self.training_case_keys)
        if (tuple(row[0] for row in self.source_support_counts) != tuple(sorted(counts))
            or any(len(row) != 4 or any(type(n) is not int for n in row[1:])
                   or row[1] != counts[row[0]] or not 1 <= row[2] <= row[1]
                   or not 1 <= row[3] <= row[1] for row in self.source_support_counts)
            or any(abs(w-1./(len(counts)*counts[c])) > 1e-12
                   for (c, _), w in zip(self.training_case_keys, self.case_weights, strict=True))):
            raise ProtocolError('HARP v21 correction source normalization or full-scope weights drifted.')
        if self.normalization_hash != canonical_hash({'normalization':self.source_support_counts,'case_keys':self.training_case_keys}):
            raise ProtocolError('HARP v21 correction normalization seal differs.')
        object.__setattr__(self, 'model_hash', canonical_hash(self._payload()))

    def _design(self, menu):
        if (menu.center_id, menu.case_id) in self.training_case_keys:
            raise ProtocolError('HARP v21 correction evidence prediction includes a fitted case.')
        return standardized(raw_design(menu, self.variant), self.means, self.scales, self.variant)

    def predict(self, menu):
        x = self._design(menu)
        b = baseline(menu)
        if self.variant == 'baseline':
            return b  # Preserve exact B probabilities for this ablation.
        bc = np.clip(b, CLIP, 1-CLIP)
        return np.clip(expit(np.log(bc/(1-bc)) + self.coefficients[0] + x@np.asarray(self.coefficients[1:])), CLIP, 1-CLIP)

    def predict_masses(self, menu):
        x = self._design(menu)
        b = np.clip(baseline(menu), CLIP, 1-CLIP)
        tau = np.maximum(0., np.asarray(self.total_mass_coefficients)@case_design(menu))
        return np.column_stack([tau[k]*softmax(np.log(b if k else 1-b)+x@np.asarray(self.mass_coefficients[k]))
                                for k in (0, 1)])

    def _payload(self):
        return dict(schema_version='harp_v21_normalized_correction_evidence', patch_schema=PATCH_SCHEMA,
            training_case_keys=self.training_case_keys, training_menu_hashes=self.training_menu_hashes,
            variant=self.variant, means=self.means, scales=self.scales, coefficients=self.coefficients,
            mass_coefficients=self.mass_coefficients, total_mass_coefficients=self.total_mass_coefficients,
            source_support_counts=self.source_support_counts, case_weights=self.case_weights,
            normalization_hash=self.normalization_hash, ridge_alpha=.1, source_only=True,
            optimizer='L-BFGS-B', optimizer_maxiter=128, optimizer_maxcor=8,
            optimizer_ftol=1e-10, optimizer_gtol=1e-7,
            standardized_embedding_width_divisor=1., total_mass_link='max(0,linear_ridge_mean)',
            posterior_probability_clip=CLIP, mass_allocation='baseline_class_anchored_softmax',
            posterior_baseline_anchored=True, class_mass_support_weight_in_allocation_loss=True,
            full_population_case_weights_before_filtering=True, total_mass_is_probability=False,
            center_id_is_model_feature=False, target_class_counts_used=False,
            estimand='frozen_source_equal_center_equal_class_equal_supporting_case_contribution',
            raw_labels_persisted=False, model_is_safety_bound=False)

    def public_payload(self):
        return {**self._payload(), 'model_hash':self.model_hash}

    @classmethod
    def from_payload(cls, payload):
        try:
            vector = lambda name: tuple(payload[name])
            matrix = lambda name: tuple(tuple(row) for row in payload[name])
            model = cls(matrix('training_case_keys'), vector('training_menu_hashes'), payload['variant'],
                vector('means'), vector('scales'), vector('coefficients'), matrix('mass_coefficients'),
                matrix('total_mass_coefficients'), matrix('source_support_counts'), vector('case_weights'),
                payload['normalization_hash'])
            if canonical_hash(model.public_payload()) != canonical_hash(payload):
                raise ProtocolError('HARP v21 correction evidence reconstruction seal differs.')
            return model
        except ProtocolError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError('HARP v21 correction evidence reconstruction is malformed.') from exc
