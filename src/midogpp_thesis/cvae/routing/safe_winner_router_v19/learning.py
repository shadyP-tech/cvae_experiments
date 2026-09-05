"""FitComplete(S): nested proposer OOF winners, gate, then final proposer(S)."""
from __future__ import annotations

from dataclasses import dataclass, field

from ...protocol import ProtocolError
from .fit_cache import ScopedFitCache, with_execution_feature_cache
from .hashing import canonical_hash
from .proposer import fit_proposer
from .splitting import _subset_menus, center_stratified_folds
from .winner_gate import fit_winner_gate
from .winner_records import SealedWinner


@dataclass(frozen=True, slots=True)
class StackedScienceModel:
    proposer: object
    winner_gate: object
    winner_fit_receipts: tuple
    training_case_keys: tuple
    model_hash: str = field(init=False)

    def __post_init__(self):
        if self.training_case_keys != self.proposer.training_case_keys or self.training_case_keys != self.winner_gate.training_case_keys:
            raise ProtocolError("HARP v19 complete learner fitting scopes disagree.")
        covered = []
        for receipt in self.winner_fit_receipts:
            train, held = set(receipt["training_case_keys"]), set(receipt["held_case_keys"])
            if train & held or train | held != set(self.training_case_keys):
                raise ProtocolError("HARP v19 gate receipt crossed its fitting scope.")
            covered.extend(held)
        if len(covered) != len(set(covered)) or set(covered) != set(self.training_case_keys):
            raise ProtocolError("HARP v19 gate held-case partition is incomplete.")
        object.__setattr__(self, "model_hash", canonical_hash(self._payload()))

    @property
    def proposal_model(self):
        return self.proposer.proposal_model

    @property
    def action_model(self):
        return self.proposer.action_model

    @property
    def stacking_receipts(self):
        return self.proposer.stacking_receipts

    @property
    def opportunity_alpha(self):
        return 1.0

    @property
    def ranker_alpha(self):
        return 1.0

    def predict_menu(self, menu):
        return self.proposer.predict_menu(menu)

    def candidate_predictions(self, menu, config):
        return self.proposer.candidate_predictions(menu, config)

    def winner_prediction(self, menu, candidates):
        return self.winner_gate.predict(menu, candidates)

    def _payload(self):
        return {"schema_version": "harp_v19_complete_nested_winner_learner",
            "proposer": self.proposer.public_payload(),
            "proposal_model": self.proposal_model.public_payload(),
            "action_model": self.action_model.public_payload(),
            "winner_gate": self.winner_gate.public_payload(),
            "winner_fit_receipts": self.winner_fit_receipts,
            "stacking_receipts": self.stacking_receipts,
            "training_case_keys": self.training_case_keys,
            "policy_rule": "MAX_NONBASELINE_SAFE_BENEFIT_THEN_POSITIVE_SCORE_AND_ONE_MINUS_WINNER_HARM_GE_TAU_ELSE_EXACT_B",
            "runnerup_after_gate_veto": False, "winner_gate_is_independently_cross_fitted": True,
            "opportunity_alpha": 1.0, "ranker_alpha": 1.0}

    def public_payload(self):
        return {**self._payload(), "model_hash": self.model_hash}


@with_execution_feature_cache
def fit_stacked_science_model(menus, capability, *, config, cache=None):
    rows = tuple(sorted(menus, key=lambda row: (row.center_id, row.case_id)))
    keys = tuple((row.center_id, row.case_id) for row in rows)
    if len(keys) < 3:
        raise ProtocolError("HARP v19 complete learner requires at least three cases for nested proposer and winner folds.")
    scoped = capability.scoped(rows)
    cache = ScopedFitCache() if cache is None else cache
    cache_key = cache.key("complete_learner", rows, scoped, config)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    folds = center_stratified_folds(keys, fold_count=min(config.winner_folds, len(keys)), namespace="HARP_V19_WINNER_STACK")
    pending, receipts = [], []
    for ordinal, held_keys in enumerate(folds):
        training = _subset_menus(rows, set(keys) - set(held_keys))
        held = _subset_menus(rows, set(held_keys))
        proposer = fit_proposer(training, scoped.scoped(training), config=config, cache=cache)
        seals = tuple(SealedWinner(menu, proposer.candidate_predictions(menu, config),
                      proposer.training_case_keys, proposer.model_hash, ordinal) for menu in held)
        pending.extend(seals)
        receipts.append({"winner_fold": ordinal,
            "training_case_keys": proposer.training_case_keys, "held_case_keys": held_keys,
            "proposer_hash": proposer.model_hash,
            "proposal_model_hash": proposer.proposal_model.model_hash,
            "action_model_hash": proposer.action_model.model_hash,
            "ranker_stacking_receipts": proposer.stacking_receipts,
            "pretruth_winner_seal_hashes": tuple(row.winner_seal_hash for row in seals),
            "held_labels_excluded_from_every_upstream_fit": True})
    # Empty menus remain in the complete normalization inventory. Negative
    # winner scores and harmful winners are deliberately not filtered here.
    sealed_winners = tuple(row.winner.candidate.composite for row in pending if row.winner is not None)
    outcomes = scoped.score_composites(sealed_winners, normalized=True) if sealed_winners else ()
    gate = fit_winner_gate(tuple(pending), outcomes, training_case_keys=keys,
                           ridge_alpha=config.winner_gate_ridge_alpha)
    proposer = fit_proposer(rows, scoped, config=config, cache=cache)
    return cache.put(cache_key, StackedScienceModel(proposer, gate, tuple(receipts), keys))
