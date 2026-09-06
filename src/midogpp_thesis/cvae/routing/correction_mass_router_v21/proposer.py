"""One frozen donor ranker and normalized correction-evidence estimator."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence
from ...protocol import ProtocolError
from .candidate_prediction import CandidatePrediction
from .composition import build_candidate_composites
from .contracts import LabelFreeCaseMenu, RouterFitConfig
from .fit_cache import ScopedFitCache
from .hashing import canonical_hash
from .modeling import fit_proposal_model
from .outcome_model import fit_action_outcome_model
from .splitting import CaseKey
from .truth import SupportTruthCapability


@dataclass(frozen=True, slots=True)
class FittedProposer:
    proposal_model: object
    action_model: object
    training_case_keys: tuple[CaseKey, ...]
    stacking_receipts: tuple[dict[str, object], ...] = ()
    model_hash: str = field(init=False)

    def __post_init__(self):
        if (self.training_case_keys != self.proposal_model.training_case_keys
            or self.training_case_keys != self.action_model.training_case_keys
            or self.stacking_receipts):
            raise ProtocolError("HARP v21 proposer fitting scope is inconsistent.")
        object.__setattr__(self, "model_hash", canonical_hash(self._payload()))

    def predict_menu(self, menu):
        return self.proposal_model.predict_menu(menu)

    def candidate_predictions(self, menu, config):
        candidates = build_candidate_composites(menu, self.predict_menu(menu), config)
        composites = tuple(row.composite for row in candidates if row.eligible and row.duplicate_of is None)
        estimates = self.action_model.predict_composites(menu, composites)
        by_hash = {row.composite_hash: estimate for row, estimate in zip(composites, estimates, strict=True)}
        return tuple(CandidatePrediction(row, None if row.composite is None else by_hash.get(row.composite.composite_hash)) for row in candidates)

    def _payload(self):
        return {"schema_version": "harp_v21_frozen_correction_mass_proposer",
            "proposal_model": self.proposal_model.public_payload(),
            "action_model": self.action_model.public_payload(),
            "training_case_keys": self.training_case_keys,
            "stacking_receipts": self.stacking_receipts,
            "candidate_outcome_regression_fitted": False,
            "candidate_tie_rule": "MAX_PREDICTED_GAIN_THEN_LEXICAL_ARM_ID"}

    def public_payload(self):
        return {**self._payload(), "model_hash": self.model_hash}


def _fit_ranker(menus, capability, config, cache):
    key = cache.key("ranker", menus, capability, config)
    cached = cache.get(key)
    if cached is not None:
        return cached
    profiles, outcomes = capability.derive_training_surface(menus)
    return cache.put(key, fit_proposal_model(menus, profiles, outcomes,
        maximum_numeric_features=config.maximum_numeric_features))


def fit_proposer(menus: Sequence[LabelFreeCaseMenu], capability: SupportTruthCapability,
                 *, config: RouterFitConfig, cache: ScopedFitCache | None = None) -> FittedProposer:
    rows = tuple(sorted(menus, key=lambda row: (row.center_id, row.case_id)))
    keys = tuple((row.center_id, row.case_id) for row in rows)
    if len(keys) < 2:
        raise ProtocolError("HARP v21 proposer requires at least two source fitting cases.")
    scoped = capability.scoped(rows)
    cache = ScopedFitCache() if cache is None else cache
    key = cache.key("proposer", rows, scoped, config)
    cached = cache.get(key)
    if cached is not None:
        return cached
    proposal = _fit_ranker(rows, scoped, config, cache)
    evidence = scoped.fit_correction_evidence(rows, variant=config.evidence_variant)
    action = fit_action_outcome_model(rows, evidence_model=evidence)
    return cache.put(key, FittedProposer(proposal, action, keys))
