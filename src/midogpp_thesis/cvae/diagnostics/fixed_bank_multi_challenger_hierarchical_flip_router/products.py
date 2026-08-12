"""Typed products for model, menu, decision, and terminal phases."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from ...routing.hierarchical_multi_challenger import (
    CandidateMenu,
    DirectionalCalibration,
    DirectionalLogitModel,
)
from .hashing import canonical_hash
from .semantic_payloads import decision_semantic_payload


def semantic_decision_payload(row: "MethodDecision") -> dict[str, object]:
    """Keep decision semantics exact while stabilizing derived fit numerics."""

    return decision_semantic_payload(row)


@dataclass(frozen=True, order=True)
class MethodDecision:
    target_center: str
    fold_ordinal: int
    case_id: str
    method_id: str
    action_id: str
    anchor_action_id: str
    best_action_id: str
    runner_up_action_id: str
    predicted_gain: float
    action_margin: float
    epistemic_standard_error: float
    calibration_standard_error: float
    margin_standard_error: float
    margin_lcb: float
    decision_source: str
    menu_hash: str
    evaluation_labels_used: bool = False

    def __post_init__(self) -> None:
        from .constants import CENTERS, PRE_EVALUATION_METHOD_IDS, legal_actions

        if (
            self.target_center not in CENTERS
            or self.fold_ordinal not in range(5)
            or self.method_id not in PRE_EVALUATION_METHOD_IDS
            or self.evaluation_labels_used is not False
            or not self.case_id
            or self.action_id not in legal_actions(self.target_center)
            or self.anchor_action_id not in legal_actions(self.target_center)
            or self.best_action_id not in legal_actions(self.target_center)
            or self.runner_up_action_id not in legal_actions(self.target_center)
            or len(self.menu_hash) != 64
        ):
            raise ProtocolError("Multi-challenger method decision drifted.")
        for role in (
            "predicted_gain",
            "action_margin",
            "epistemic_standard_error",
            "calibration_standard_error",
            "margin_standard_error",
            "margin_lcb",
        ):
            value = float(getattr(self, role))
            if not __import__("math").isfinite(value) or (
                "standard_error" in role and value < 0.0
            ):
                raise ProtocolError("Multi-challenger decision numerics drifted.")

    def to_payload(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class DonorPhaseResult:
    contribution_rows: tuple[Mapping[str, object], ...]
    fit_rows: tuple[Mapping[str, object], ...]
    model_seals: Mapping[str, Mapping[str, object]]
    permutation_provenance: Mapping[str, object]
    models_by_target_family: Mapping[
        str, Mapping[str, Mapping[str, DirectionalLogitModel]]
    ]
    single_models_by_target: Mapping[str, object]


@dataclass(frozen=True)
class FoldPhaseResult:
    menu_rows: tuple[Mapping[str, object], ...]
    calibration_rows: tuple[Mapping[str, object], ...]
    score_rows: tuple[Mapping[str, object], ...]
    decisions: tuple[MethodDecision, ...]
    fold_seal_hashes: Mapping[tuple[str, int], str]
    menu_by_fold: Mapping[tuple[str, int], CandidateMenu]
    calibrations_by_fold_family: Mapping[
        tuple[str, int], Mapping[str, Mapping[str, DirectionalCalibration]]
    ]
    decision_bundle_hash: str

    def __post_init__(self) -> None:
        from .constants import CENTERS, PRE_EVALUATION_METHOD_IDS

        expected_folds = {(target, fold) for target in CENTERS for fold in range(5)}
        if set(self.fold_seal_hashes) != expected_folds:
            raise ProtocolError("Multi-challenger fold seal coverage drifted.")
        expected_decisions = 218 * len(PRE_EVALUATION_METHOD_IDS)
        if len(self.decisions) != expected_decisions:
            raise ProtocolError("Multi-challenger decision coverage drifted.")
        payload = {
            "schema_version": "fixed_bank_multi_challenger_decision_bundle_v1",
            "decisions": [semantic_decision_payload(row) for row in self.decisions],
            "fold_seals": {
                f"{key[0]}::{key[1]}": value
                for key, value in sorted(self.fold_seal_hashes.items())
            },
            "evaluation_labels_used": False,
        }
        if self.decision_bundle_hash != canonical_hash(payload):
            raise ProtocolError("Multi-challenger decision bundle hash drifted.")
        object.__setattr__(
            self, "fold_seal_hashes", MappingProxyType(dict(self.fold_seal_hashes))
        )


__all__ = (
    "DonorPhaseResult",
    "FoldPhaseResult",
    "MethodDecision",
    "semantic_decision_payload",
)
