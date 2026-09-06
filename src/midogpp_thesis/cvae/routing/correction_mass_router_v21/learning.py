"""Fit, freeze, calibrate: the evaluated proposer is never refitted afterwards."""
from __future__ import annotations
from dataclasses import dataclass, field
from ...protocol import ProtocolError
from .calibration_split import proposer_calibration_partition
from .fit_cache import ScopedFitCache, with_execution_feature_cache
from .hashing import canonical_hash
from .risk_selection import POLICY_RULE, selection_contract
from .proposer import fit_proposer
from .splitting import _subset_menus
from .winner_gate import fit_winner_gate
from .winner_records import SealedWinner


@dataclass(frozen=True, slots=True)
class StackedScienceModel:
    proposer: object
    winner_gate: object
    winner_fit_receipts: tuple
    training_case_keys: tuple
    evidence_variant: str = "embedding_residual"
    model_hash: str = field(init=False)

    def __post_init__(self):
        fitting, calibration = proposer_calibration_partition(self.training_case_keys)
        if (self.proposer.training_case_keys != fitting
            or self.winner_gate.training_case_keys != calibration
            or tuple(tuple(k) for k in self.winner_gate.fit_audit["normalization_scope_case_keys"]) != self.training_case_keys
            or len(self.winner_fit_receipts) != 1):
            raise ProtocolError("HARP v21 complete learner fit/calibration scopes disagree.")
        receipt = self.winner_fit_receipts[0]
        if (tuple(tuple(k) for k in receipt["training_case_keys"]) != fitting or tuple(tuple(k) for k in receipt["held_case_keys"]) != calibration
            or receipt["proposer_hash"] != self.proposer.model_hash
            or receipt["proposer_refitted_after_calibration"] is not False):
            raise ProtocolError("HARP v21 calibrated proposer was replaced or crossed its fitting scope.")
        if self.action_model.patch_model.variant != self.evidence_variant:
            raise ProtocolError("HARP v21 fitted correction evidence variant drifted.")
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

    def _require_held_case(self, menu):
        # Role retagging cannot turn a fitting/calibration identity into an
        # evaluation case. Training accesses the frozen proposer directly.
        if (menu.center_id, menu.case_id) in self.training_case_keys:
            raise ProtocolError("HARP v21 complete learner prediction includes a fitting or calibration case.")

    def predict_menu(self, menu):
        self._require_held_case(menu)
        return self.proposer.predict_menu(menu)

    def candidate_predictions(self, menu, config):
        self._require_held_case(menu)
        return self.proposer.candidate_predictions(menu, config)

    def winner_prediction(self, menu, candidates):
        self._require_held_case(menu)
        return self.winner_gate.predict(menu, candidates)

    def _payload(self):
        return {"schema_version": "harp_v21_complete_nested_winner_learner",
            "evidence_variant": self.evidence_variant, "risk_selection": selection_contract(),
            "proposer": self.proposer.public_payload(),
            "proposal_model": self.proposal_model.public_payload(),
            "action_model": self.action_model.public_payload(),
            "winner_gate": self.winner_gate.public_payload(),
            "winner_fit_receipts": self.winner_fit_receipts,
            "stacking_receipts": self.stacking_receipts,
            "training_case_keys": self.training_case_keys,
            "proposer_fit_case_keys": self.proposer.training_case_keys,
            "calibration_case_keys": self.winner_gate.training_case_keys,
            "policy_rule": POLICY_RULE, "runnerup_after_gate_veto": False,
            "winner_gate_is_fitted_on_disjoint_calibration_cases": True,
            "proposer_refitted_after_calibration": False,
            "opportunity_alpha": 1.0, "ranker_alpha": 1.0}

    def public_payload(self):
        return {**self._payload(), "model_hash": self.model_hash}


@with_execution_feature_cache
def fit_stacked_science_model(menus, capability, *, config, cache=None):
    rows = tuple(sorted(menus, key=lambda row: (row.center_id, row.case_id)))
    keys = tuple((row.center_id, row.case_id) for row in rows)
    if len(keys) < 3:
        raise ProtocolError("HARP v21 complete learner requires three fit/calibration cases.")
    scoped = capability.scoped(rows)
    cache = ScopedFitCache() if cache is None else cache
    cache_key = cache.key("complete_learner", rows, scoped, config)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    fit_keys, calibration_keys = proposer_calibration_partition(keys)
    fitting = _subset_menus(rows, set(fit_keys))
    calibration = _subset_menus(rows, set(calibration_keys))
    proposer = fit_proposer(fitting, scoped.scoped(fitting), config=config, cache=cache)
    # Freeze every winner from this very proposer before consuming any
    # calibration outcome. There is no subsequent full-scope proposer refit.
    pending = tuple(SealedWinner(menu, proposer.candidate_predictions(menu, config),
        proposer.training_case_keys, proposer.model_hash, 0) for menu in calibration)
    sealed_winners = tuple(row.winner.candidate.composite for row in pending if row.winner is not None)
    outcomes = scoped.score_composites(sealed_winners, normalized=True) if sealed_winners else ()
    gate = fit_winner_gate(pending, outcomes, training_case_keys=calibration_keys,
        population_case_keys=keys, ridge_alpha=config.winner_gate_ridge_alpha)
    receipt = {"calibration_partition": 0, "training_case_keys": fit_keys,
        "held_case_keys": calibration_keys, "proposer_hash": proposer.model_hash,
        "proposal_model_hash": proposer.proposal_model.model_hash,
        "action_model_hash": proposer.action_model.model_hash,
        "pretruth_winner_seal_hashes": tuple(row.winner_seal_hash for row in pending),
        "held_labels_excluded_from_every_upstream_fit": True,
        "proposer_refitted_after_calibration": False}
    return cache.put(cache_key, StackedScienceModel(proposer, gate, (receipt,), keys, config.evidence_variant))
