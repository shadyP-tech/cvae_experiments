"""Ranker-stacked actual-composite proposer fitted in one exact scope."""
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
from .splitting import CaseKey, _subset_menus, center_stratified_folds
from .truth import SupportTruthCapability


@dataclass(frozen=True, slots=True)
class FittedProposer:
    proposal_model: object
    action_model: object
    training_case_keys: tuple[CaseKey, ...]
    stacking_receipts: tuple[dict[str, object], ...]
    model_hash: str = field(init=False)

    def __post_init__(self):
        if (self.training_case_keys != self.proposal_model.training_case_keys
            or self.training_case_keys != self.action_model.training_case_keys):
            raise ProtocolError("HARP v19 proposer fitting scope is inconsistent.")
        covered = []
        for receipt in self.stacking_receipts:
            training, held = set(receipt["training_case_keys"]), set(receipt["held_case_keys"])
            if training & held or training | held != set(self.training_case_keys):
                raise ProtocolError("HARP v19 ranker receipt crossed its fitting scope.")
            covered.extend(held)
        if len(covered) != len(set(covered)) or set(covered) != set(self.training_case_keys):
            raise ProtocolError("HARP v19 ranker held-case partition is incomplete.")
        object.__setattr__(self, "model_hash", canonical_hash(self._payload()))

    def predict_menu(self, menu):
        return self.proposal_model.predict_menu(menu)

    def candidate_predictions(self, menu, config):
        candidates = build_candidate_composites(menu, self.predict_menu(menu), config)
        composites = tuple(row.composite for row in candidates if row.eligible and row.duplicate_of is None)
        predictions = self.action_model.predict_composites(menu, composites)
        by_hash = {row.composite_hash: prediction for row, prediction in zip(composites, predictions, strict=True)}
        return tuple(CandidatePrediction(row, None if row.composite is None else by_hash.get(row.composite.composite_hash)) for row in candidates)

    def _payload(self):
        return {"schema_version": "harp_v19_ranker_stacked_proposer",
            "proposal_model": self.proposal_model.public_payload(),
            "action_model": self.action_model.public_payload(),
            "training_case_keys": self.training_case_keys,
            "stacking_receipts": self.stacking_receipts,
            "candidate_tie_rule": "MAX_SAFE_BENEFIT_THEN_LEXICAL_ARM_ID"}

    def public_payload(self):
        return {**self._payload(), "model_hash": self.model_hash}


def _fit_ranker(menus, capability, config, cache):
    key = cache.key("ranker", menus, capability, config)
    cached = cache.get(key)
    if cached is not None:
        return cached
    profiles, outcomes = capability.derive_training_surface(menus)
    model = fit_proposal_model(menus, profiles, outcomes, maximum_numeric_features=config.maximum_numeric_features)
    return cache.put(key, model)


def fit_proposer(menus: Sequence[LabelFreeCaseMenu], capability: SupportTruthCapability,
                 *, config: RouterFitConfig, cache: ScopedFitCache | None = None) -> FittedProposer:
    rows = tuple(sorted(menus, key=lambda row: (row.center_id, row.case_id)))
    keys = tuple((row.center_id, row.case_id) for row in rows)
    if len(keys) < 2:
        raise ProtocolError("HARP v19 proposer requires at least two source cases for honest stacking.")
    scoped = capability.scoped(rows)
    cache = ScopedFitCache() if cache is None else cache
    cache_key = cache.key("proposer", rows, scoped, config)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    folds = center_stratified_folds(keys, fold_count=min(config.stack_folds, len(keys)), namespace="HARP_V19_RANKER_STACK")
    composites, receipts = [], []
    for ordinal, held_keys in enumerate(folds):
        train = _subset_menus(rows, set(keys) - set(held_keys))
        held = _subset_menus(rows, set(held_keys))
        proposal = _fit_ranker(train, scoped.scoped(train), config, cache)
        pending = tuple(candidate.composite for menu in held
            for candidate in build_candidate_composites(menu, proposal.predict_menu(menu), config)
            if candidate.eligible and candidate.duplicate_of is None)
        seal = canonical_hash({"composite_hashes": tuple(row.composite_hash for row in pending),
                               "training_case_keys": tuple((row.center_id, row.case_id) for row in train)})
        composites.extend(pending)
        receipts.append({"stack_fold": ordinal, "proposal_model_hash": proposal.model_hash,
            "training_case_keys": tuple((row.center_id, row.case_id) for row in train),
            "held_case_keys": held_keys, "pretruth_composite_seal_hash": seal,
            "candidate_count": len(pending), "held_candidates_sealed_before_truth": True})
    # Full fitting-scope normalizers are derived after every ranker-OOF action
    # is sealed. Empty / probability-only menus still contribute to this scope.
    outcomes = scoped.score_composites(tuple(composites), normalized=True)
    profiles, _ = scoped.derive_training_surface(rows)
    action = fit_action_outcome_model(rows, tuple(composites), outcomes,
        maximum_numeric_features=config.maximum_numeric_features,
        ridge_alpha=config.candidate_ridge_alpha, normalization_profiles=profiles)
    proposal = _fit_ranker(rows, scoped, config, cache)
    return cache.put(cache_key, FittedProposer(proposal, action, keys, tuple(receipts)))
