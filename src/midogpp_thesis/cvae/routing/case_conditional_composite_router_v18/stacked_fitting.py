"""Fit the complete proposal/outcome stack independently within each scope.

The outcome model sees only composites proposed for cases held out of the
ranker that produced them. No global OOF table crosses an outer boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ...protocol import ProtocolError
from .composition import build_baseline_composite, build_candidate_composites
from .contracts import LabelFreeCaseMenu, RouterFitConfig, SoftTopKComposite
from .hashing import canonical_hash
from .modeling import fit_proposal_model
from .outcome_model import fit_action_outcome_model
from .splitting import CaseKey, _subset_menus, center_stratified_folds
from .truth import SupportTruthCapability


POLICY_ARM_ID = "CASE_CONDITIONAL_ACTION_POLICY"


@dataclass(frozen=True, slots=True)
class CandidatePrediction:
    """One label-free proposed candidate with its actual-composite estimates."""
    candidate: object
    prediction: object | None
    hard_prediction_changed: bool = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "hard_prediction_changed", bool(
            self.candidate.composite is not None and self.candidate.composite.prediction_changed
        ))
        if self.prediction is not None and (
            self.candidate.composite is None
            or self.prediction.composite_hash != self.candidate.composite.composite_hash
        ):
            raise ProtocolError("HARP v18 outcome estimates do not bind their actual composite.")

    @property
    def arm_id(self) -> str:
        return self.candidate.arm_id

    @property
    def screened(self) -> bool:
        p = self.prediction
        return bool(
            self.candidate.eligible and self.candidate.duplicate_of is None
            and self.hard_prediction_changed
            and p is not None and p.predicted_gain > 0.0
            and p.predicted_harm <= 0.25
            and p.predicted_brier_delta <= 0.002
            and p.predicted_logloss_delta <= 0.005
        )

    @property
    def route_score(self) -> float:
        return 0.0 if self.prediction is None else max(0.0, float(self.prediction.predicted_gain))

    def public_payload(self) -> dict[str, object]:
        p = self.prediction
        return {
            "arm_id": self.arm_id,
            "eligible": self.candidate.eligible,
            "ineligible_reason": self.candidate.ineligible_reason,
            "duplicate_of": self.candidate.duplicate_of,
            "composite_hash": None if self.candidate.composite is None else self.candidate.composite.composite_hash,
            "prediction": None if p is None else p.public_payload(),
            "predictive_risk_screen_passes": self.screened,
            "hard_prediction_changed": self.hard_prediction_changed,
        }


@dataclass(frozen=True, slots=True)
class HeldCandidatePrediction:
    fold: int
    menu: LabelFreeCaseMenu
    candidates: tuple[CandidatePrediction, ...]
    training_case_keys: tuple[CaseKey, ...]
    model_hash: str
    prediction_seal_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (self.menu.center_id, self.menu.case_id) in self.training_case_keys:
            raise ProtocolError("HARP v18 held candidate entered ranker/outcome fitting.")
        object.__setattr__(self, "prediction_seal_hash", canonical_hash({
            "schema_version": "harp_v18_held_candidate_prediction_seal",
            "fold": self.fold, "menu_hash": self.menu.menu_hash,
            "training_case_keys": self.training_case_keys,
            "model_hash": self.model_hash,
            "candidates": tuple(row.public_payload() for row in self.candidates),
            "held_truth_joined": False,
        }))


def choose_candidate(
    menu: LabelFreeCaseMenu,
    candidates: Sequence[CandidatePrediction],
    threshold: float,
    *,
    enabled: bool = True,
) -> tuple[SoftTopKComposite, float, str | None]:
    """Choose independently per case; a missing candidate never prunes a policy."""
    if not enabled:
        return build_baseline_composite(menu), 0.0, "NO_SAFE_INNER_OOF_POLICY"
    eligible = tuple(row for row in candidates if row.screened and row.route_score > threshold)
    if not eligible:
        return build_baseline_composite(menu), 0.0, "NO_PREDICTED_SAFE_GAIN_ABOVE_THRESHOLD"
    winner = min(eligible, key=lambda row: (-row.route_score, row.arm_id))
    return winner.candidate.composite, winner.route_score, None


@dataclass(frozen=True, slots=True)
class StackedScienceModel:
    proposal_model: object
    action_model: object
    training_case_keys: tuple[CaseKey, ...]
    stacking_receipts: tuple[dict[str, object], ...]
    model_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (not self.training_case_keys
            or self.training_case_keys != self.proposal_model.training_case_keys
            or self.training_case_keys != self.action_model.training_case_keys):
            raise ProtocolError("HARP v18 stacked model training scope is inconsistent.")
        covered = []
        for receipt in self.stacking_receipts:
            training, held = set(receipt["training_case_keys"]), set(receipt["held_case_keys"])
            if training.intersection(held) or training | held != set(self.training_case_keys):
                raise ProtocolError("HARP v18 ranker stack receipt crossed its fitting scope.")
            covered.extend(held)
        if len(covered) != len(set(covered)) or set(covered) != set(self.training_case_keys):
            raise ProtocolError("HARP v18 ranker stack held-case partition is incomplete.")
        object.__setattr__(self, "model_hash", canonical_hash({
            "schema_version": "harp_v18_independently_nested_stack",
            "proposal_model": self.proposal_model.public_payload(),
            "action_model": self.action_model.public_payload(),
            "training_case_keys": self.training_case_keys,
            "stacking_receipts": self.stacking_receipts,
            "ridge_alpha": 1.0,
        }))

    @property
    def opportunity_alpha(self) -> float:
        return 1.0

    @property
    def ranker_alpha(self) -> float:
        return 1.0

    def predict_menu(self, menu: LabelFreeCaseMenu) -> object:
        return self.proposal_model.predict_menu(menu)

    def candidate_predictions(self, menu: LabelFreeCaseMenu, config: RouterFitConfig) -> tuple[CandidatePrediction, ...]:
        candidates = build_candidate_composites(menu, self.predict_menu(menu), config)
        composites = tuple(row.composite for row in candidates if row.eligible and row.duplicate_of is None)
        predictions = self.action_model.predict_composites(menu, composites)
        by_hash = dict(zip((row.composite_hash for row in composites), predictions, strict=True))
        return tuple(CandidatePrediction(row, None if row.composite is None else by_hash.get(row.composite.composite_hash)) for row in candidates)

    def public_payload(self) -> dict[str, object]:
        return {
            "schema_version": "harp_v18_independently_nested_stack",
            "model_hash": self.model_hash,
            "proposal_model": self.proposal_model.public_payload(),
            "action_model": self.action_model.public_payload(),
            "training_case_keys": [list(key) for key in self.training_case_keys],
            "stacking_receipts": list(self.stacking_receipts),
            "outcome_targets_are_actual_heldout_composites": True,
            "ranker_and_outcome_fits_repeat_inside_each_validation_scope": True,
            "opportunity_alpha": 1.0, "ranker_alpha": 1.0,
        }


def fit_stacked_science_model(
    menus: Sequence[LabelFreeCaseMenu], capability: SupportTruthCapability, *, config: RouterFitConfig,
) -> StackedScienceModel:
    rows = tuple(sorted(menus, key=lambda row: (row.center_id, row.case_id)))
    keys = tuple((row.center_id, row.case_id) for row in rows)
    scoped = capability.scoped(rows)
    folds = center_stratified_folds(keys, fold_count=config.stack_folds, namespace="HARP_V18_STACK")
    composites = []
    receipts = []
    for ordinal, held_keys in enumerate(folds):
        train = _subset_menus(rows, set(keys) - set(held_keys))
        held = _subset_menus(rows, set(held_keys))
        training_truth = scoped.scoped(train)
        profiles, primitive_outcomes = training_truth.derive_training_surface(train)
        proposal = fit_proposal_model(train, profiles, primitive_outcomes, maximum_numeric_features=config.maximum_numeric_features)
        pending = tuple(
            candidate.composite for menu in held
            for candidate in build_candidate_composites(menu, proposal.predict_menu(menu), config)
            if candidate.eligible and candidate.duplicate_of is None
        )
        # The entire held candidate menu is immutable and hashed before truth.
        seal = canonical_hash({"composite_hashes": tuple(row.composite_hash for row in pending), "training_case_keys": tuple((row.center_id, row.case_id) for row in train)})
        composites.extend(pending)
        receipts.append({
            "stack_fold": ordinal, "proposal_model_hash": proposal.model_hash,
            "training_case_keys": tuple((row.center_id, row.case_id) for row in train),
            "held_case_keys": held_keys, "pretruth_composite_seal_hash": seal,
            "candidate_count": len(pending), "held_candidates_sealed_before_truth": True,
        })
    # Joint Fit(S) normalizer is opened only after every ranker-OOF candidate
    # in S is sealed. An outer validation case is absent from S entirely.
    outcomes = scoped.score_composites(tuple(composites), normalized=True)
    action = fit_action_outcome_model(rows, tuple(composites), outcomes, maximum_numeric_features=config.maximum_numeric_features)
    profiles, primitive_outcomes = scoped.derive_training_surface(rows)
    proposal = fit_proposal_model(rows, profiles, primitive_outcomes, maximum_numeric_features=config.maximum_numeric_features)
    return StackedScienceModel(proposal, action, keys, tuple(receipts))
