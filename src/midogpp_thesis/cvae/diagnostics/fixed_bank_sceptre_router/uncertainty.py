"""Paired whole-case uncertainty for the descriptive SCEPTRE diagnostic.

The statistical unit resampled here is a whole target case.  Each action keeps
its exact 3 x 3 training/generation seed grid; metrics are pooled across target
observations *within each physical seed cell* and only then averaged over the
nine cells.  One Dirichlet case-weight draw is shared by every action, including
the exact equal-union B control, so all action-versus-B comparisons are paired.

This pure statistical kernel can construct phase-bound selection, calibration,
or terminal-evaluation surfaces, but only selection and calibration may produce
a route decision.  A capability here is lineage evidence, not a raw-label
firewall: label access must already have been authorized by the phase-owned
reader.  This module does not authorize label access, execution, promotion,
fresh-evidence claims, or downstream reuse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    GENERATION_SEEDS,
    TRAINING_SEEDS,
    legal_routing_sources,
)
from midogpp_thesis.cvae.protocol import ProtocolError

from .hashing import canonical_hash, require_sha256
from .outcome_surface import EXACT_B_CANDIDATE
from .partitions import FOLD_COUNT, ThreeRoleFold
from .phase_contracts import PhaseCapability, TerminalEvaluationCapability
from .policy_contracts import FIXED_ACCEPTANCE_PROBABILITY


SEED_CELL_GRID = tuple(
    (training_seed, generation_seed)
    for training_seed in TRAINING_SEEDS
    for generation_seed in GENERATION_SEEDS
)
SEED_CELL_COUNT = len(SEED_CELL_GRID)
ROUTING_ROLES = frozenset({"SELECTION", "CALIBRATION"})
ALL_SURFACE_ROLES = frozenset({*ROUTING_ROLES, "EVALUATION"})
FIXED_PREDICTION_THRESHOLD = 0.5
FIXED_DIRICHLET_CONCENTRATION = 1.0
FIXED_LOG_LOSS_EPSILON = 1e-15
DEFAULT_BOOTSTRAP_DRAWS = 2048
DEFAULT_BOOTSTRAP_SEED = 27_082_026
PUBLICATION_STATUS = "POST_HOC_CONSUMED_TEST_SENSITIVITY"
TERMINAL_DECISION = "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"

_PHASE_ROLE_BY_SURFACE_ROLE = {
    "SELECTION": "SELECTION_LABELS",
    "CALIBRATION": "CALIBRATION_LABELS",
}


def _identifier(value: object, role: str) -> str:
    text = str(value)
    if not text or text.strip() != text:
        raise ProtocolError(f"SCEPTRE {role} is invalid.")
    return text


def _positive_integer(value: object, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProtocolError(f"SCEPTRE {role} must be a positive integer.")
    return value


def _rng_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProtocolError("SCEPTRE Dirichlet RNG seed must be a nonnegative integer.")
    return value


def _probability(value: object, role: str) -> float:
    if isinstance(value, bool):
        raise ProtocolError(f"SCEPTRE {role} must be a finite probability.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError(f"SCEPTRE {role} must be a finite probability.") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ProtocolError(f"SCEPTRE {role} must be a finite probability.")
    return parsed


def _capability_identity(
    capability: PhaseCapability | TerminalEvaluationCapability,
    *,
    target_center: str,
    fold_ordinal: int,
    partition_hash: str,
    role: str,
) -> tuple[str, str, str | None]:
    """Validate the capability's public scope and hash its phase identity.

    This pure kernel deliberately does not decide whether a capability was
    manager-issued.  The phase manager retains that nonce/identity check when
    the reduced decision is recorded.
    """

    expected_phase_role = _PHASE_ROLE_BY_SURFACE_ROLE.get(role)
    if expected_phase_role is not None:
        if not isinstance(capability, PhaseCapability):
            raise ProtocolError(
                "SCEPTRE labeled routing surface requires a typed phase capability."
            )
        if (
            capability.role != expected_phase_role
            or capability.target_center != target_center
            or capability.fold_ordinal != fold_ordinal
            or capability.partition_hash != partition_hash
        ):
            raise ProtocolError("SCEPTRE phase capability scope or partition drifted.")
        router_bundle = require_sha256(
            capability.router_bundle_hash, "phase-capability router bundle"
        )
        g_proposal = require_sha256(
            capability.g_proposal_hash, "phase-capability G proposal"
        )
        predecessor_decision = require_sha256(
            capability.predecessor_decision_hash,
            "phase-capability predecessor decision",
        )
        predecessor = require_sha256(
            capability.predecessor_seal_hash, "phase-capability predecessor seal"
        )
        nonce = require_sha256(capability.nonce_hash, "phase-capability nonce")
        body = {
            "schema_version": "sceptre_phase_capability_identity_v1",
            "capability_kind": "LABEL_PHASE",
            "capability_role": expected_phase_role,
            "target_center": target_center,
            "fold_ordinal": fold_ordinal,
            "partition_hash": partition_hash,
            "router_bundle_hash": router_bundle,
            "g_proposal_hash": g_proposal,
            "predecessor_decision_hash": predecessor_decision,
            "predecessor_seal_hash": predecessor,
            "nonce_hash": nonce,
            "manager_ownership_check_location": "phase_manager_record",
        }
        return canonical_hash(body), router_bundle, g_proposal

    if role != "EVALUATION" or not isinstance(
        capability, TerminalEvaluationCapability
    ):
        raise ProtocolError(
            "SCEPTRE evaluation surface requires a typed terminal capability."
        )
    if capability.partition_hash != partition_hash:
        raise ProtocolError("SCEPTRE terminal capability partition drifted.")
    router_bundle = require_sha256(
        capability.router_bundle_hash, "terminal-capability router bundle"
    )
    route_policy = require_sha256(
        capability.route_policy_hash, "terminal route-policy artifact"
    )
    policy_seal = require_sha256(
        capability.policy_seal_hash, "terminal-capability policy seal"
    )
    attestation = require_sha256(
        capability.durable_attestation_hash, "terminal-capability attestation"
    )
    terminal_hash = require_sha256(
        capability.capability_hash, "terminal-capability identity"
    )
    expected_terminal_hash = canonical_hash(
        {
            "schema_version": "sceptre_terminal_evaluation_capability_v1",
            "partition_hash": partition_hash,
            "router_bundle_hash": router_bundle,
            "route_policy_hash": route_policy,
            "policy_seal_hash": policy_seal,
            "durable_attestation_hash": attestation,
            "one_shot": True,
            "raw_labels_may_be_persisted": False,
        }
    )
    if terminal_hash != expected_terminal_hash:
        raise ProtocolError("SCEPTRE terminal capability semantic replay drifted.")
    body = {
        "schema_version": "sceptre_phase_capability_identity_v1",
        "capability_kind": "TERMINAL_EVALUATION",
        "partition_hash": partition_hash,
        "router_bundle_hash": router_bundle,
        "route_policy_hash": route_policy,
        "policy_seal_hash": policy_seal,
        "durable_attestation_hash": attestation,
        "terminal_capability_hash": terminal_hash,
        "global_terminal_capability": True,
    }
    return canonical_hash(body), router_bundle, None


@dataclass(frozen=True, slots=True)
class DirichletBootstrapConfig:
    """Deterministic Bayesian-bootstrap configuration with a frozen gate."""

    draw_count: int = DEFAULT_BOOTSTRAP_DRAWS
    rng_seed: int = DEFAULT_BOOTSTRAP_SEED
    acceptance_probability: float = field(
        default=FIXED_ACCEPTANCE_PROBABILITY,
        init=False,
    )
    dirichlet_concentration: float = field(
        default=FIXED_DIRICHLET_CONCENTRATION,
        init=False,
    )
    prediction_threshold: float = field(
        default=FIXED_PREDICTION_THRESHOLD,
        init=False,
    )
    log_loss_epsilon: float = field(
        default=FIXED_LOG_LOSS_EPSILON,
        init=False,
    )
    config_hash: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        draws = _positive_integer(self.draw_count, "Dirichlet draw count")
        seed = _rng_seed(self.rng_seed)
        body = {
            "schema_version": "sceptre_dirichlet_bootstrap_config_v1",
            "draw_count": draws,
            "rng_seed": seed,
            "rng_algorithm": "numpy.PCG64",
            "dirichlet_concentration": FIXED_DIRICHLET_CONCENTRATION,
            "acceptance_probability": FIXED_ACCEPTANCE_PROBABILITY,
            "prediction_threshold": FIXED_PREDICTION_THRESHOLD,
            "log_loss_epsilon": FIXED_LOG_LOSS_EPSILON,
            "resampling_unit": "whole_target_case",
            "weights_shared_across_actions_and_seed_cells": True,
            "seed_cells_resampled": False,
            "descriptive_only": True,
        }
        expected = canonical_hash(body)
        if self.config_hash and self.config_hash != expected:
            raise ProtocolError("SCEPTRE Dirichlet-bootstrap config hash drifted.")
        object.__setattr__(self, "draw_count", draws)
        object.__setattr__(self, "rng_seed", seed)
        object.__setattr__(self, "config_hash", expected)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "sceptre_dirichlet_bootstrap_config_v1",
            "draw_count": self.draw_count,
            "rng_seed": self.rng_seed,
            "rng_algorithm": "numpy.PCG64",
            "dirichlet_concentration": self.dirichlet_concentration,
            "acceptance_probability": self.acceptance_probability,
            "prediction_threshold": self.prediction_threshold,
            "log_loss_epsilon": self.log_loss_epsilon,
            "resampling_unit": "whole_target_case",
            "weights_shared_across_actions_and_seed_cells": True,
            "seed_cells_resampled": False,
            "descriptive_only": True,
            "config_hash": self.config_hash,
        }


@dataclass(frozen=True, slots=True)
class SeedCellPredictions:
    """Predicted positive-class probabilities for one physical seed cell."""

    training_seed: int
    generation_seed: int
    probabilities: tuple[float, ...]
    prediction_hash: str = ""

    def __post_init__(self) -> None:
        if (
            self.training_seed not in TRAINING_SEEDS
            or self.generation_seed not in GENERATION_SEEDS
        ):
            raise ProtocolError("SCEPTRE prediction seed cell is outside the exact grid.")
        probabilities = tuple(
            _probability(value, "seed-cell prediction") for value in self.probabilities
        )
        if not probabilities:
            raise ProtocolError("SCEPTRE seed-cell prediction vector is empty.")
        body = {
            "schema_version": "sceptre_seed_cell_predictions_v1",
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "probabilities": list(probabilities),
            "probability_semantics": "positive_class_probability",
        }
        expected = canonical_hash(body)
        if self.prediction_hash and self.prediction_hash != expected:
            raise ProtocolError("SCEPTRE seed-cell prediction hash drifted.")
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "prediction_hash", expected)


@dataclass(frozen=True, slots=True)
class ActionPredictionSurface:
    """Exact nine-cell prediction tensor for one source-family action or B."""

    action_id: str
    seed_cells: tuple[SeedCellPredictions, ...]
    action_hash: str = ""

    def __post_init__(self) -> None:
        action = _identifier(self.action_id, "action identifier")
        cells = tuple(self.seed_cells)
        if any(not isinstance(cell, SeedCellPredictions) for cell in cells):
            raise ProtocolError("SCEPTRE action prediction surface is untyped.")
        by_key = {
            (cell.training_seed, cell.generation_seed): cell for cell in cells
        }
        if len(cells) != SEED_CELL_COUNT or set(by_key) != set(SEED_CELL_GRID):
            raise ProtocolError("SCEPTRE action lacks the exact nine seed cells.")
        ordered = tuple(by_key[key] for key in SEED_CELL_GRID)
        row_counts = {len(cell.probabilities) for cell in ordered}
        if len(row_counts) != 1:
            raise ProtocolError("SCEPTRE action seed cells have different row counts.")
        body = {
            "schema_version": "sceptre_action_prediction_surface_v1",
            "action_id": action,
            "seed_cell_grid": [list(key) for key in SEED_CELL_GRID],
            "seed_cell_prediction_hashes": [cell.prediction_hash for cell in ordered],
            "seed_cells_are_nuisance_replications": True,
            "seed_cells_are_independent_observations": False,
        }
        expected = canonical_hash(body)
        if self.action_hash and self.action_hash != expected:
            raise ProtocolError("SCEPTRE action prediction-surface hash drifted.")
        object.__setattr__(self, "action_id", action)
        object.__setattr__(self, "seed_cells", ordered)
        object.__setattr__(self, "action_hash", expected)

    @property
    def observation_count(self) -> int:
        return len(self.seed_cells[0].probabilities)


@dataclass(frozen=True, slots=True)
class RolePredictionSurface:
    """Fold-bound action x seed-cell x observation surface with whole-case IDs."""

    target_center: str
    fold: ThreeRoleFold
    partition_hash: str
    role: str
    observation_ids: tuple[str, ...]
    case_ids: tuple[str, ...]
    labels: tuple[int, ...]
    actions: tuple[ActionPredictionSurface, ...]
    candidate_menu_hash: str
    exact_b_control_receipt_hash: str
    prediction_bundle_sha256: str
    phase_capability: PhaseCapability | TerminalEvaluationCapability = field(
        repr=False,
        compare=False,
    )
    phase_capability_identity_hash: str = ""
    publication_status: str = PUBLICATION_STATUS
    terminal_decision: str = TERMINAL_DECISION
    descriptive_only: bool = True
    fresh_evidence: bool = False
    surface_hash: str = ""

    def __post_init__(self) -> None:
        target = _identifier(self.target_center, "target center")
        role = str(self.role).upper()
        if (
            not isinstance(self.fold, ThreeRoleFold)
            or self.fold.target_center != target
            or role not in ALL_SURFACE_ROLES
        ):
            raise ProtocolError("SCEPTRE role prediction surface scope drifted.")
        partition = require_sha256(self.partition_hash, "prediction partition")
        phase_identity, router_bundle, g_proposal = _capability_identity(
            self.phase_capability,
            target_center=target,
            fold_ordinal=self.fold.fold_ordinal,
            partition_hash=partition,
            role=role,
        )
        if (
            self.phase_capability_identity_hash
            and self.phase_capability_identity_hash != phase_identity
        ):
            raise ProtocolError("SCEPTRE phase-capability identity hash drifted.")
        observations = tuple(
            _identifier(value, "observation identifier") for value in self.observation_ids
        )
        cases = tuple(_identifier(value, "case identifier") for value in self.case_ids)
        if (
            not observations
            or len(observations) != len(set(observations))
            or len(cases) != len(observations)
            or len(self.labels) != len(observations)
        ):
            raise ProtocolError("SCEPTRE prediction observation geometry drifted.")
        labels: list[int] = []
        for raw in self.labels:
            if isinstance(raw, bool) or not isinstance(raw, int) or raw not in (0, 1):
                raise ProtocolError("SCEPTRE prediction labels are not exact binary integers.")
            labels.append(raw)
        frozen_labels = tuple(labels)
        if set(frozen_labels) != {0, 1}:
            raise ProtocolError("SCEPTRE prediction role surface lacks one true class.")
        expected_cases = set(_fold_case_ids(self.fold, role))
        if set(cases) != expected_cases:
            raise ProtocolError("SCEPTRE prediction cases differ from the typed fold role.")

        actions = tuple(self.actions)
        expected_actions = (*legal_routing_sources(target), EXACT_B_CANDIDATE)
        if (
            any(not isinstance(action, ActionPredictionSurface) for action in actions)
            or tuple(action.action_id for action in actions) != expected_actions
            or any(action.observation_count != len(observations) for action in actions)
        ):
            raise ProtocolError(
                "SCEPTRE prediction surface must be exact C minus H plus exact B."
            )
        menu_hash = _identifier(self.candidate_menu_hash, "candidate-menu hash")
        control_hash = _identifier(
            self.exact_b_control_receipt_hash, "exact-B control receipt"
        )
        prediction_sha = require_sha256(
            self.prediction_bundle_sha256, "prediction bundle"
        )
        if (
            self.publication_status != PUBLICATION_STATUS
            or self.terminal_decision != TERMINAL_DECISION
            or self.descriptive_only is not True
            or self.fresh_evidence is not False
        ):
            raise ProtocolError("SCEPTRE uncertainty claim boundary drifted.")
        body = {
            "schema_version": "sceptre_role_prediction_surface_v1",
            "target_center": target,
            "fold_ordinal": self.fold.fold_ordinal,
            "fold_hash": self.fold.fold_hash,
            "partition_hash": partition,
            "role": role,
            "role_case_set_hash": self.fold.case_set_hash(role),
            "observation_ids": list(observations),
            "case_ids": list(cases),
            "labels": list(frozen_labels),
            "action_ids": list(expected_actions),
            "action_hashes": [action.action_hash for action in actions],
            "candidate_menu_hash": menu_hash,
            "exact_b_control_receipt_hash": control_hash,
            "prediction_bundle_sha256": prediction_sha,
            "phase_capability_identity_hash": phase_identity,
            "phase_router_bundle_hash": router_bundle,
            "phase_g_proposal_hash": g_proposal,
            "phase_predecessor_decision_hash": (
                self.phase_capability.predecessor_decision_hash
                if isinstance(self.phase_capability, PhaseCapability)
                else None
            ),
            "prediction_tensor_shape": [len(actions), SEED_CELL_COUNT, len(observations)],
            "whole_case_resampling_required": True,
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
            "descriptive_only": True,
            "fresh_evidence": False,
        }
        expected_hash = canonical_hash(body)
        if self.surface_hash and self.surface_hash != expected_hash:
            raise ProtocolError("SCEPTRE role prediction-surface hash drifted.")
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "partition_hash", partition)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "observation_ids", observations)
        object.__setattr__(self, "case_ids", cases)
        object.__setattr__(self, "labels", frozen_labels)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "candidate_menu_hash", menu_hash)
        object.__setattr__(self, "exact_b_control_receipt_hash", control_hash)
        object.__setattr__(self, "prediction_bundle_sha256", prediction_sha)
        object.__setattr__(self, "phase_capability_identity_hash", phase_identity)
        object.__setattr__(self, "surface_hash", expected_hash)

    @property
    def action_ids(self) -> tuple[str, ...]:
        return tuple(action.action_id for action in self.actions)

    @property
    def whole_case_ids(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.case_ids)))

    @property
    def tensor_shape(self) -> tuple[int, int, int]:
        return len(self.actions), SEED_CELL_COUNT, len(self.observation_ids)

    @property
    def router_bundle_hash(self) -> str:
        return self.phase_capability.router_bundle_hash

    @property
    def g_proposal_hash(self) -> str | None:
        capability = self.phase_capability
        return (
            capability.g_proposal_hash
            if isinstance(capability, PhaseCapability)
            else None
        )

    @property
    def predecessor_decision_hash(self) -> str | None:
        capability = self.phase_capability
        return (
            capability.predecessor_decision_hash
            if isinstance(capability, PhaseCapability)
            else None
        )


def _fold_case_ids(fold: ThreeRoleFold, role: str) -> tuple[str, ...]:
    values = {
        "SELECTION": fold.selection_case_ids,
        "CALIBRATION": fold.calibration_case_ids,
        "EVALUATION": fold.evaluation_case_ids,
    }.get(role)
    if values is None:
        raise ProtocolError("SCEPTRE uncertainty role is unknown.")
    return values


def build_role_prediction_surface(
    *,
    target_center: str,
    fold: ThreeRoleFold,
    partition_hash: str,
    role: str,
    observation_ids: Sequence[str],
    case_ids: Sequence[str],
    labels: Sequence[int],
    probabilities_by_action_and_seed: Mapping[
        str,
        Mapping[tuple[int, int], Sequence[float]],
    ],
    candidate_menu_hash: str,
    exact_b_control_receipt_hash: str,
    prediction_bundle_sha256: str,
    phase_capability: PhaseCapability | TerminalEvaluationCapability,
) -> RolePredictionSurface:
    """Build a canonical immutable surface after public capability-scope replay."""

    target = str(target_center)
    normalized_role = str(role).upper()
    if (
        not isinstance(fold, ThreeRoleFold)
        or fold.target_center != target
        or normalized_role not in ALL_SURFACE_ROLES
    ):
        raise ProtocolError("SCEPTRE role prediction surface scope drifted.")
    partition = require_sha256(partition_hash, "prediction partition")
    # Validate the capability's public lineage before transforming any label or
    # prediction values. Manager ownership remains the phase manager's concern.
    _capability_identity(
        phase_capability,
        target_center=target,
        fold_ordinal=fold.fold_ordinal,
        partition_hash=partition,
        role=normalized_role,
    )
    try:
        expected_actions = (*legal_routing_sources(target), EXACT_B_CANDIDATE)
    except ValueError as exc:
        raise ProtocolError("SCEPTRE prediction target is unknown.") from exc
    try:
        raw_actions = dict(probabilities_by_action_and_seed)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("SCEPTRE prediction action mapping is invalid.") from exc
    if set(raw_actions) != set(expected_actions):
        raise ProtocolError("SCEPTRE prediction action mapping is not C minus H plus B.")
    actions: list[ActionPredictionSurface] = []
    for action_id in expected_actions:
        try:
            raw_cells = dict(raw_actions[action_id])
        except (TypeError, ValueError) as exc:
            raise ProtocolError("SCEPTRE prediction seed-cell mapping is invalid.") from exc
        if set(raw_cells) != set(SEED_CELL_GRID):
            raise ProtocolError("SCEPTRE prediction action lacks the exact seed grid.")
        actions.append(
            ActionPredictionSurface(
                action_id=action_id,
                seed_cells=tuple(
                    SeedCellPredictions(
                        training_seed=training_seed,
                        generation_seed=generation_seed,
                        probabilities=tuple(raw_cells[(training_seed, generation_seed)]),
                    )
                    for training_seed, generation_seed in SEED_CELL_GRID
                ),
            )
        )
    return RolePredictionSurface(
        target_center=target,
        fold=fold,
        partition_hash=partition,
        role=normalized_role,
        observation_ids=tuple(observation_ids),
        case_ids=tuple(case_ids),
        labels=tuple(labels),
        actions=tuple(actions),
        candidate_menu_hash=candidate_menu_hash,
        exact_b_control_receipt_hash=exact_b_control_receipt_hash,
        prediction_bundle_sha256=prediction_bundle_sha256,
        phase_capability=phase_capability,
    )


@dataclass(frozen=True, slots=True)
class ActionUncertaintySummary:
    action_id: str
    point_bacc: float
    point_brier: float
    point_log_loss: float
    bootstrap_expected_bacc: float
    bootstrap_expected_brier: float
    bootstrap_expected_log_loss: float
    bacc_superiority_probability: float
    brier_noninferiority_probability: float
    log_loss_noninferiority_probability: float
    joint_acceptance_probability: float
    summary_hash: str = ""

    def __post_init__(self) -> None:
        action = _identifier(self.action_id, "uncertainty action")
        unit_interval_values = (
            self.point_bacc,
            self.point_brier,
            self.bootstrap_expected_bacc,
            self.bootstrap_expected_brier,
            self.bacc_superiority_probability,
            self.brier_noninferiority_probability,
            self.log_loss_noninferiority_probability,
            self.joint_acceptance_probability,
        )
        if any(
            not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
            for value in unit_interval_values
        ):
            raise ProtocolError("SCEPTRE uncertainty summary contains an invalid probability.")
        if any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in (self.point_log_loss, self.bootstrap_expected_log_loss)
        ):
            raise ProtocolError("SCEPTRE uncertainty summary log loss is invalid.")
        body = self._payload_without_hash(action)
        expected = canonical_hash(body)
        if self.summary_hash and self.summary_hash != expected:
            raise ProtocolError("SCEPTRE action uncertainty-summary hash drifted.")
        object.__setattr__(self, "action_id", action)
        object.__setattr__(self, "summary_hash", expected)

    def _payload_without_hash(self, action: str | None = None) -> dict[str, object]:
        return {
            "schema_version": "sceptre_action_uncertainty_summary_v1",
            "action_id": self.action_id if action is None else action,
            "point_bacc": self.point_bacc,
            "point_brier": self.point_brier,
            "point_log_loss": self.point_log_loss,
            "bootstrap_expected_bacc": self.bootstrap_expected_bacc,
            "bootstrap_expected_brier": self.bootstrap_expected_brier,
            "bootstrap_expected_log_loss": self.bootstrap_expected_log_loss,
            "bacc_superiority_probability": self.bacc_superiority_probability,
            "brier_noninferiority_probability": self.brier_noninferiority_probability,
            "log_loss_noninferiority_probability": (
                self.log_loss_noninferiority_probability
            ),
            "joint_acceptance_probability": self.joint_acceptance_probability,
            "reference_action": EXACT_B_CANDIDATE,
            "seed_metric_aggregation": "pool_cases_within_seed_then_mean_nine",
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload_without_hash(), "summary_hash": self.summary_hash}


@dataclass(frozen=True, slots=True)
class UncertaintyRouteDecision:
    target_center: str
    fold_ordinal: int
    fold_hash: str
    partition_hash: str
    role: str
    role_case_set_hash: str
    candidate_menu_hash: str
    exact_b_control_receipt_hash: str
    router_bundle_hash: str
    g_proposal_hash: str
    predecessor_decision_hash: str
    phase_capability_identity_hash: str
    prediction_surface_hash: str
    bootstrap_config_hash: str
    shared_weight_draw_hash: str
    action_summaries: tuple[ActionUncertaintySummary, ...]
    g_proposed_candidate: str
    support_decision_hash: str | None
    support_selected_candidate: str | None
    selected_candidate: str | None
    route: str
    accepted: bool
    acceptance_probability: float
    reason: str
    publication_status: str = PUBLICATION_STATUS
    terminal_decision: str = TERMINAL_DECISION
    descriptive_only: bool = True
    fresh_evidence: bool = False
    decision_hash: str = ""

    def __post_init__(self) -> None:
        target = _identifier(self.target_center, "uncertainty target")
        if (
            self.role not in ROUTING_ROLES
            or isinstance(self.fold_ordinal, bool)
            or self.fold_ordinal not in range(FOLD_COUNT)
        ):
            raise ProtocolError("SCEPTRE uncertainty decisions cannot use evaluation labels.")
        try:
            candidates = legal_routing_sources(target)
        except ValueError as exc:
            raise ProtocolError("SCEPTRE uncertainty target is unknown.") from exc
        expected_actions = (*candidates, EXACT_B_CANDIDATE)
        summaries = tuple(self.action_summaries)
        if (
            any(not isinstance(row, ActionUncertaintySummary) for row in summaries)
            or tuple(row.action_id for row in summaries) != expected_actions
        ):
            raise ProtocolError("SCEPTRE uncertainty summaries are not C minus H plus B.")
        proposed = _identifier(self.g_proposed_candidate, "G-proposed candidate")
        if proposed not in candidates:
            raise ProtocolError("SCEPTRE G proposal is outside exact C minus H.")
        if self.role == "SELECTION":
            if (
                self.support_decision_hash is not None
                or self.support_selected_candidate is not None
            ):
                raise ProtocolError(
                    "SCEPTRE selection uncertainty cannot consume support lineage."
                )
            support_hash = None
            support_selected = None
        else:
            if self.support_decision_hash != self.predecessor_decision_hash:
                raise ProtocolError(
                    "SCEPTRE calibration uncertainty lost its support decision."
                )
            support_hash = require_sha256(
                self.support_decision_hash, "calibration support decision"
            )
            support_selected = _identifier(
                self.support_selected_candidate,
                "support-selected candidate",
            )
            if support_selected != proposed or support_selected not in candidates:
                raise ProtocolError(
                    "SCEPTRE calibration candidate differs from support or G."
                )
        if self.acceptance_probability != FIXED_ACCEPTANCE_PROBABILITY:
            raise ProtocolError("SCEPTRE uncertainty acceptance threshold drifted.")
        proposed_summary = next(row for row in summaries if row.action_id == proposed)
        expected_accepted = (
            proposed_summary.joint_acceptance_probability
            >= FIXED_ACCEPTANCE_PROBABILITY
        )
        expected_selected = proposed if expected_accepted else None
        expected_route = proposed if expected_accepted else EXACT_B_CANDIDATE
        expected_reason = (
            "FIXED_0_8_UPSTREAM_CANDIDATE_PAIRED_GATE_ACCEPT"
            if expected_accepted
            else "FIXED_0_8_UPSTREAM_CANDIDATE_PAIRED_GATE_FALLBACK_TO_B"
        )
        if (
            self.selected_candidate != expected_selected
            or self.accepted is not expected_accepted
            or self.route != expected_route
            or self.reason != expected_reason
        ):
            raise ProtocolError("SCEPTRE uncertainty route semantics drifted.")
        for digest, role in (
            (self.fold_hash, "uncertainty fold"),
            (self.partition_hash, "uncertainty partition"),
            (self.role_case_set_hash, "uncertainty case set"),
            (self.router_bundle_hash, "uncertainty router bundle"),
            (self.g_proposal_hash, "uncertainty G proposal"),
            (
                self.predecessor_decision_hash,
                "uncertainty predecessor decision",
            ),
            (
                self.phase_capability_identity_hash,
                "uncertainty phase-capability identity",
            ),
            (self.prediction_surface_hash, "prediction surface"),
            (self.bootstrap_config_hash, "bootstrap config"),
            (self.shared_weight_draw_hash, "shared bootstrap weights"),
        ):
            require_sha256(digest, role)
        menu_hash = _identifier(self.candidate_menu_hash, "candidate-menu hash")
        control_hash = _identifier(
            self.exact_b_control_receipt_hash, "exact-B control receipt"
        )
        if (
            self.publication_status != PUBLICATION_STATUS
            or self.terminal_decision != TERMINAL_DECISION
            or self.descriptive_only is not True
            or self.fresh_evidence is not False
        ):
            raise ProtocolError("SCEPTRE uncertainty decision claim boundary drifted.")
        body = {
            "schema_version": "sceptre_uncertainty_route_decision_v2",
            "target_center": target,
            "fold_ordinal": self.fold_ordinal,
            "fold_hash": self.fold_hash,
            "partition_hash": self.partition_hash,
            "role": self.role,
            "role_case_set_hash": self.role_case_set_hash,
            "candidate_menu_hash": menu_hash,
            "exact_b_control_receipt_hash": control_hash,
            "router_bundle_hash": self.router_bundle_hash,
            "g_proposal_hash": self.g_proposal_hash,
            "predecessor_decision_hash": self.predecessor_decision_hash,
            "phase_capability_identity_hash": self.phase_capability_identity_hash,
            "prediction_surface_hash": self.prediction_surface_hash,
            "bootstrap_config_hash": self.bootstrap_config_hash,
            "shared_weight_draw_hash": self.shared_weight_draw_hash,
            "action_summary_hashes": [row.summary_hash for row in summaries],
            "full_action_summaries_are_descriptive_only": True,
            "routing_comparison_action_ids": [proposed, EXACT_B_CANDIDATE],
            "g_proposed_candidate": proposed,
            "g_decision_receipt_hash": (
                self.predecessor_decision_hash
                if self.role == "SELECTION"
                else None
            ),
            "support_decision_hash": support_hash,
            "support_selected_candidate": support_selected,
            "selected_candidate": expected_selected,
            "route": expected_route,
            "accepted": expected_accepted,
            "acceptance_probability": FIXED_ACCEPTANCE_PROBABILITY,
            "reason": expected_reason,
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
            "descriptive_only": True,
            "fresh_evidence": False,
        }
        expected_hash = canonical_hash(body)
        if self.decision_hash and self.decision_hash != expected_hash:
            raise ProtocolError("SCEPTRE uncertainty decision hash drifted.")
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "candidate_menu_hash", menu_hash)
        object.__setattr__(self, "exact_b_control_receipt_hash", control_hash)
        object.__setattr__(self, "action_summaries", summaries)
        object.__setattr__(self, "g_proposed_candidate", proposed)
        object.__setattr__(self, "support_decision_hash", support_hash)
        object.__setattr__(self, "support_selected_candidate", support_selected)
        object.__setattr__(self, "selected_candidate", expected_selected)
        object.__setattr__(self, "route", expected_route)
        object.__setattr__(self, "accepted", expected_accepted)
        object.__setattr__(self, "reason", expected_reason)
        object.__setattr__(self, "decision_hash", expected_hash)

    @property
    def summaries_by_action(self) -> Mapping[str, ActionUncertaintySummary]:
        return MappingProxyType({row.action_id: row for row in self.action_summaries})


def paired_dirichlet_route_decision(
    surface: RolePredictionSurface,
    *,
    g_proposed_candidate: str,
    support_selected_candidate: str | None = None,
    config: DirichletBootstrapConfig | None = None,
) -> UncertaintyRouteDecision:
    """Evaluate one predeclared G proposal against exact B with the fixed gate."""

    if not isinstance(surface, RolePredictionSurface):
        raise ProtocolError("SCEPTRE uncertainty requires a typed prediction surface.")
    if surface.role not in ROUTING_ROLES:
        raise ProtocolError("SCEPTRE evaluation labels cannot select or calibrate a route.")
    proposed = _identifier(g_proposed_candidate, "G-proposed candidate")
    if proposed not in legal_routing_sources(surface.target_center):
        raise ProtocolError("SCEPTRE G proposal is outside exact C minus H.")
    if surface.role == "SELECTION":
        if support_selected_candidate is not None:
            raise ProtocolError(
                "SCEPTRE selection uncertainty cannot consume support lineage."
            )
    else:
        support_selected = _identifier(
            support_selected_candidate,
            "support-selected candidate",
        )
        if support_selected != proposed:
            raise ProtocolError(
                "SCEPTRE calibration candidate differs from support or G."
            )
    settings = DirichletBootstrapConfig() if config is None else config
    if not isinstance(settings, DirichletBootstrapConfig):
        raise ProtocolError("SCEPTRE uncertainty config type drifted.")

    probabilities = np.asarray(
        [
            [cell.probabilities for cell in action.seed_cells]
            for action in surface.actions
        ],
        dtype=np.float64,
    )
    expected_shape = surface.tensor_shape
    if probabilities.shape != expected_shape or not np.all(np.isfinite(probabilities)):
        raise ProtocolError("SCEPTRE uncertainty tensor geometry or finiteness drifted.")
    labels = np.asarray(surface.labels, dtype=np.int8)
    case_ids = surface.whole_case_ids
    case_index_by_id = {case_id: index for index, case_id in enumerate(case_ids)}
    observation_case_indices = np.asarray(
        [case_index_by_id[case_id] for case_id in surface.case_ids],
        dtype=np.int64,
    )
    contributions = _case_contributions(
        probabilities,
        labels,
        observation_case_indices,
        len(case_ids),
        settings,
    )

    point_weights = np.ones((1, len(case_ids)), dtype=np.float64)
    point_bacc, point_brier, point_log = _metrics_from_case_weights(
        point_weights,
        contributions,
    )
    generator = np.random.Generator(np.random.PCG64(settings.rng_seed))
    shared_case_weights = generator.dirichlet(
        np.full(
            len(case_ids),
            settings.dirichlet_concentration,
            dtype=np.float64,
        ),
        size=settings.draw_count,
    ).astype(np.float64, copy=False)
    if (
        shared_case_weights.shape != (settings.draw_count, len(case_ids))
        or not np.all(np.isfinite(shared_case_weights))
        or np.any(shared_case_weights <= 0.0)
    ):
        raise ProtocolError("SCEPTRE Dirichlet whole-case weights are invalid.")
    bootstrap_bacc, bootstrap_brier, bootstrap_log = _metrics_from_case_weights(
        shared_case_weights,
        contributions,
    )
    draw_bytes_hash = hashlib.sha256(
        np.ascontiguousarray(shared_case_weights, dtype="<f8").tobytes()
    ).hexdigest()
    shared_weight_draw_hash = canonical_hash(
        {
            "schema_version": "sceptre_shared_dirichlet_case_weights_v1",
            "config_hash": settings.config_hash,
            "case_ids": list(case_ids),
            "draw_shape": list(shared_case_weights.shape),
            "float64_little_endian_sha256": draw_bytes_hash,
            "same_weights_for_all_actions_and_seed_cells": True,
        }
    )

    reference_index = surface.action_ids.index(EXACT_B_CANDIDATE)
    summaries: list[ActionUncertaintySummary] = []
    for action_index, action_id in enumerate(surface.action_ids):
        bacc_better = bootstrap_bacc[:, action_index] > bootstrap_bacc[:, reference_index]
        brier_safe = bootstrap_brier[:, action_index] <= bootstrap_brier[:, reference_index]
        log_safe = bootstrap_log[:, action_index] <= bootstrap_log[:, reference_index]
        joint = bacc_better & brier_safe & log_safe
        summaries.append(
            ActionUncertaintySummary(
                action_id=action_id,
                point_bacc=float(point_bacc[0, action_index]),
                point_brier=float(point_brier[0, action_index]),
                point_log_loss=float(point_log[0, action_index]),
                bootstrap_expected_bacc=float(
                    np.mean(bootstrap_bacc[:, action_index], dtype=np.float64)
                ),
                bootstrap_expected_brier=float(
                    np.mean(bootstrap_brier[:, action_index], dtype=np.float64)
                ),
                bootstrap_expected_log_loss=float(
                    np.mean(bootstrap_log[:, action_index], dtype=np.float64)
                ),
                bacc_superiority_probability=float(np.mean(bacc_better, dtype=np.float64)),
                brier_noninferiority_probability=float(
                    np.mean(brier_safe, dtype=np.float64)
                ),
                log_loss_noninferiority_probability=float(
                    np.mean(log_safe, dtype=np.float64)
                ),
                joint_acceptance_probability=float(np.mean(joint, dtype=np.float64)),
            )
        )
    frozen_summaries = tuple(summaries)
    proposed_summary = next(
        row for row in frozen_summaries if row.action_id == proposed
    )
    accepted = (
        proposed_summary.joint_acceptance_probability
        >= FIXED_ACCEPTANCE_PROBABILITY
    )
    selected = proposed if accepted else None
    route = proposed if accepted else EXACT_B_CANDIDATE
    reason = (
        "FIXED_0_8_UPSTREAM_CANDIDATE_PAIRED_GATE_ACCEPT"
        if accepted
        else "FIXED_0_8_UPSTREAM_CANDIDATE_PAIRED_GATE_FALLBACK_TO_B"
    )
    g_proposal_hash = surface.g_proposal_hash
    predecessor_decision_hash = surface.predecessor_decision_hash
    if g_proposal_hash is None or predecessor_decision_hash is None:
        raise ProtocolError("SCEPTRE routing surface lost its phase lineage.")
    return UncertaintyRouteDecision(
        target_center=surface.target_center,
        fold_ordinal=surface.fold.fold_ordinal,
        fold_hash=surface.fold.fold_hash,
        partition_hash=surface.partition_hash,
        role=surface.role,
        role_case_set_hash=surface.fold.case_set_hash(surface.role),
        candidate_menu_hash=surface.candidate_menu_hash,
        exact_b_control_receipt_hash=surface.exact_b_control_receipt_hash,
        router_bundle_hash=surface.router_bundle_hash,
        g_proposal_hash=g_proposal_hash,
        predecessor_decision_hash=predecessor_decision_hash,
        phase_capability_identity_hash=surface.phase_capability_identity_hash,
        prediction_surface_hash=surface.surface_hash,
        bootstrap_config_hash=settings.config_hash,
        shared_weight_draw_hash=shared_weight_draw_hash,
        action_summaries=frozen_summaries,
        g_proposed_candidate=proposed,
        support_decision_hash=(
            predecessor_decision_hash if surface.role == "CALIBRATION" else None
        ),
        support_selected_candidate=(
            support_selected_candidate if surface.role == "CALIBRATION" else None
        ),
        selected_candidate=selected,
        route=route,
        accepted=accepted,
        acceptance_probability=FIXED_ACCEPTANCE_PROBABILITY,
        reason=reason,
    )


@dataclass(frozen=True, slots=True)
class _CaseContributions:
    true_negative: np.ndarray
    false_positive: np.ndarray
    false_negative: np.ndarray
    true_positive: np.ndarray
    brier_sum: np.ndarray
    log_loss_sum: np.ndarray
    observation_count: np.ndarray


def _case_contributions(
    probabilities: np.ndarray,
    labels: np.ndarray,
    observation_case_indices: np.ndarray,
    case_count: int,
    config: DirichletBootstrapConfig,
) -> _CaseContributions:
    action_count, seed_count, observation_count = probabilities.shape
    predicted = probabilities >= config.prediction_threshold
    truth_positive = labels.astype(bool)[None, None, :]
    truth_negative = ~truth_positive
    clipped = np.clip(
        probabilities,
        config.log_loss_epsilon,
        1.0 - config.log_loss_epsilon,
    )
    labels_float = labels.astype(np.float64)[None, None, :]
    brier = np.square(probabilities - labels_float)
    log_loss = -(
        labels_float * np.log(clipped)
        + (1.0 - labels_float) * np.log1p(-clipped)
    )
    shape = (case_count, action_count, seed_count)

    def aggregate(values: np.ndarray) -> np.ndarray:
        result = np.zeros(shape, dtype=np.float64)
        for case_index in range(case_count):
            mask = observation_case_indices == case_index
            if not np.any(mask):
                raise ProtocolError("SCEPTRE whole-case contribution is empty.")
            result[case_index] = np.sum(values[:, :, mask], axis=2, dtype=np.float64)
        return result

    true_negative = aggregate(truth_negative & ~predicted)
    false_positive = aggregate(truth_negative & predicted)
    false_negative = aggregate(truth_positive & ~predicted)
    true_positive = aggregate(truth_positive & predicted)
    brier_sum = aggregate(brier)
    log_loss_sum = aggregate(log_loss)
    rows = np.zeros(shape, dtype=np.float64)
    per_case_rows = np.bincount(
        observation_case_indices,
        minlength=case_count,
    ).astype(np.float64)
    rows[:] = per_case_rows[:, None, None]
    if (
        observation_count <= 0
        or seed_count != SEED_CELL_COUNT
        or not all(np.all(np.isfinite(value)) for value in (brier_sum, log_loss_sum))
    ):
        raise ProtocolError("SCEPTRE case-level statistical contributions are invalid.")
    return _CaseContributions(
        true_negative,
        false_positive,
        false_negative,
        true_positive,
        brier_sum,
        log_loss_sum,
        rows,
    )


def _metrics_from_case_weights(
    weights: np.ndarray,
    contributions: _CaseContributions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def weighted(values: np.ndarray) -> np.ndarray:
        case_count, action_count, seed_count = values.shape
        flattened = values.reshape(case_count, action_count * seed_count)
        return (weights @ flattened).reshape(
            weights.shape[0],
            action_count,
            seed_count,
        )

    true_negative = weighted(contributions.true_negative)
    false_positive = weighted(contributions.false_positive)
    false_negative = weighted(contributions.false_negative)
    true_positive = weighted(contributions.true_positive)
    negative = true_negative + false_positive
    positive = true_positive + false_negative
    if np.any(negative <= 0.0) or np.any(positive <= 0.0):
        raise ProtocolError("SCEPTRE weighted pooled BACC lacks one true class.")
    seed_bacc = 0.5 * (true_negative / negative + true_positive / positive)
    weighted_rows = weighted(contributions.observation_count)
    if np.any(weighted_rows <= 0.0):
        raise ProtocolError("SCEPTRE weighted proper-loss denominator is empty.")
    seed_brier = weighted(contributions.brier_sum) / weighted_rows
    seed_log = weighted(contributions.log_loss_sum) / weighted_rows
    # Physical seed cells are nuisance replications: never resample or flatten
    # them into the bootstrap observation axis.
    mean_bacc = np.mean(seed_bacc, axis=2, dtype=np.float64)
    mean_brier = np.mean(seed_brier, axis=2, dtype=np.float64)
    mean_log = np.mean(seed_log, axis=2, dtype=np.float64)
    if not all(np.all(np.isfinite(value)) for value in (mean_bacc, mean_brier, mean_log)):
        raise ProtocolError("SCEPTRE uncertainty metrics are non-finite.")
    return mean_bacc, mean_brier, mean_log


# Compact orchestration aliases.
RolePredictionOutcomeSurface = RolePredictionSurface
paired_shared_dirichlet_decision = paired_dirichlet_route_decision


__all__ = (
    "ALL_SURFACE_ROLES",
    "ActionPredictionSurface",
    "ActionUncertaintySummary",
    "DEFAULT_BOOTSTRAP_DRAWS",
    "DEFAULT_BOOTSTRAP_SEED",
    "DirichletBootstrapConfig",
    "FIXED_ACCEPTANCE_PROBABILITY",
    "FIXED_DIRICHLET_CONCENTRATION",
    "FIXED_LOG_LOSS_EPSILON",
    "FIXED_PREDICTION_THRESHOLD",
    "PUBLICATION_STATUS",
    "ROUTING_ROLES",
    "RolePredictionOutcomeSurface",
    "RolePredictionSurface",
    "SEED_CELL_COUNT",
    "SEED_CELL_GRID",
    "SeedCellPredictions",
    "TERMINAL_DECISION",
    "UncertaintyRouteDecision",
    "build_role_prediction_surface",
    "paired_dirichlet_route_decision",
    "paired_shared_dirichlet_decision",
)
