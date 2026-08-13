"""Immutable scientific products for the support-static routing diagnostic."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .constants import (
    B_ACTION_ID,
    CENTERS,
    HARD_THRESHOLD,
    METHOD_IDS,
    OOF_FOLD_COUNT,
    PRE_EVALUATION_METHOD_IDS,
    U_ACTION_ID,
    decision_action_ids,
    physical_action_ids,
    source_from_action,
)
from .hashing import (
    canonical_hash,
    finite,
    nonempty_text,
    nonnegative_int,
    require_sha256,
    require_stable_hash,
)


@dataclass(frozen=True, order=True)
class CaseIdentityRow:
    target_center: str
    case_id: str
    sample_id: str

    def __post_init__(self) -> None:
        if str(self.target_center) not in CENTERS:
            raise ProtocolError("Case identity contains an unknown MIDOG++ center.")
        nonempty_text(self.case_id, "case_id")
        nonempty_text(self.sample_id, "sample_id")

    @property
    def case_key(self) -> tuple[str, str]:
        return self.target_center, self.case_id

    @property
    def sample_key(self) -> tuple[str, str, str]:
        return self.target_center, self.case_id, self.sample_id

    def to_payload(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True, order=True)
class BinaryLabelRow:
    target_center: str
    case_id: str
    sample_id: str
    value: int

    def __post_init__(self) -> None:
        CaseIdentityRow(self.target_center, self.case_id, self.sample_id)
        if isinstance(self.value, bool) or self.value not in (0, 1):
            raise ProtocolError("Labels must be binary integer values.")

    @property
    def label(self) -> int:
        return self.value

    @property
    def case_key(self) -> tuple[str, str]:
        return self.target_center, self.case_id

    @property
    def sample_key(self) -> tuple[str, str, str]:
        return self.target_center, self.case_id, self.sample_id


@dataclass(frozen=True, order=True)
class BinaryPredictionRow:
    target_center: str
    case_id: str
    sample_id: str
    action_id: str
    probability: float
    probability_surface_hash: str

    def __post_init__(self) -> None:
        CaseIdentityRow(self.target_center, self.case_id, self.sample_id)
        if self.action_id not in physical_action_ids(self.target_center):
            raise ProtocolError("Prediction row uses an illegal target action.")
        value = finite(self.probability, "probability")
        if not 0.0 <= value <= 1.0:
            raise ProtocolError("Prediction probability must lie in [0,1].")
        require_sha256(self.probability_surface_hash, "probability_surface_hash")
        object.__setattr__(self, "probability", value)

    @property
    def hard_prediction(self) -> int:
        return int(self.probability >= HARD_THRESHOLD)

    @property
    def sample_key(self) -> tuple[str, str, str]:
        return self.target_center, self.case_id, self.sample_id


@dataclass(frozen=True, order=True)
class CaseActionCounts:
    """Exact additive case/action counts; intentionally no per-case BACC field."""

    target_center: str
    case_id: str
    action_id: str
    n_positive: int
    true_positive: int
    n_negative: int
    true_negative: int
    counts_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        if self.target_center not in CENTERS:
            raise ProtocolError("Case-action counts contain an unknown target center.")
        nonempty_text(self.case_id, "case_id")
        if self.action_id not in physical_action_ids(self.target_center):
            raise ProtocolError("Case-action counts use an illegal action.")
        for name in ("n_positive", "true_positive", "n_negative", "true_negative"):
            nonnegative_int(getattr(self, name), name)
        if self.n_positive + self.n_negative <= 0:
            raise ProtocolError("Case-action counts cannot be empty.")
        if self.true_positive > self.n_positive or self.true_negative > self.n_negative:
            raise ProtocolError("Correct counts exceed class denominators.")
        object.__setattr__(self, "counts_hash", canonical_hash(self._unhashed()))

    @property
    def case_key(self) -> tuple[str, str]:
        return self.target_center, self.case_id

    @property
    def key(self) -> tuple[str, str, str]:
        return self.target_center, self.case_id, self.action_id

    @property
    def class_counts(self) -> tuple[int, int]:
        return self.n_positive, self.n_negative

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_support_static_router_case_action_counts_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "action_id": self.action_id,
            "n_positive": self.n_positive,
            "true_positive": self.true_positive,
            "n_negative": self.n_negative,
            "true_negative": self.true_negative,
            "additive_sufficient_statistics": True,
            "per_case_bacc_computed": False,
            "per_case_bacc_persisted": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "counts_hash": self.counts_hash}


# Compatibility spelling used by older fixed-bank packages.
CaseConfusionCounts = CaseActionCounts


@dataclass(frozen=True)
class PooledBacc:
    action_or_method_id: str
    case_count: int
    n_positive: int
    true_positive: int
    n_negative: int
    true_negative: int
    sensitivity: float
    specificity: float
    exact_bacc: float
    metric_hash: str = field(init=False)

    def __post_init__(self) -> None:
        nonempty_text(self.action_or_method_id, "action_or_method_id")
        if self.case_count <= 0 or self.n_positive <= 0 or self.n_negative <= 0:
            raise ProtocolError("Pooled BACC requires cases and both pooled classes.")
        if not 0 <= self.true_positive <= self.n_positive:
            raise ProtocolError("Pooled true-positive count is invalid.")
        if not 0 <= self.true_negative <= self.n_negative:
            raise ProtocolError("Pooled true-negative count is invalid.")
        sensitivity = finite(self.sensitivity, "sensitivity")
        specificity = finite(self.specificity, "specificity")
        exact_bacc = finite(self.exact_bacc, "exact_bacc")
        expected_sensitivity = self.true_positive / self.n_positive
        expected_specificity = self.true_negative / self.n_negative
        if (
            abs(sensitivity - expected_sensitivity) > 1.0e-12
            or abs(specificity - expected_specificity) > 1.0e-12
            or abs(exact_bacc - 0.5 * (expected_sensitivity + expected_specificity)) > 1.0e-12
        ):
            raise ProtocolError("Pooled BACC differs from its exact sufficient statistics.")
        object.__setattr__(self, "sensitivity", sensitivity)
        object.__setattr__(self, "specificity", specificity)
        object.__setattr__(self, "exact_bacc", exact_bacc)
        object.__setattr__(self, "metric_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_support_static_router_pooled_bacc_v1",
            "action_or_method_id": self.action_or_method_id,
            "case_count": self.case_count,
            "n_positive": self.n_positive,
            "true_positive": self.true_positive,
            "n_negative": self.n_negative,
            "true_negative": self.true_negative,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "exact_bacc": self.exact_bacc,
            "per_case_bacc_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "metric_hash": self.metric_hash}


@dataclass(frozen=True, order=True)
class ActionGain:
    action_id: str
    selected_source: str | None
    action_score: float | None
    baseline_score: float | None
    gain: float | None
    score_type: str
    label_scope: str
    donor_centers: tuple[str, ...]
    label_case_keys: tuple[tuple[str, str], ...]
    contribution_hash: str

    def __post_init__(self) -> None:
        if self.action_id == B_ACTION_ID:
            if self.selected_source is not None:
                raise ProtocolError("Baseline gain cannot select a source.")
        else:
            source = source_from_action(self.action_id)
            if self.selected_source != source:
                raise ProtocolError("Action gain source identity drifted.")
        nonempty_text(self.score_type, "score_type")
        if self.score_type not in {
            "pooled_exact_bacc",
            "equal_center_mean_of_per_q_pooled_exact_bacc",
        }:
            raise ProtocolError("Action gain score type is not predeclared.")
        nonempty_text(self.label_scope, "label_scope")
        centers = tuple(str(value) for value in self.donor_centers)
        case_keys = tuple(sorted((str(center), str(case)) for center, case in self.label_case_keys))
        if (
            not centers
            or len(centers) != len(set(centers))
            or any(center not in CENTERS for center in centers)
            or not case_keys
            or len(case_keys) != len(set(case_keys))
            or {center for center, _case in case_keys} != set(centers)
        ):
            raise ProtocolError("Action gain label provenance is incomplete or duplicated.")
        require_sha256(self.contribution_hash, "contribution_hash")
        object.__setattr__(self, "donor_centers", centers)
        object.__setattr__(self, "label_case_keys", case_keys)
        values = (self.action_score, self.baseline_score, self.gain)
        if all(value is None for value in values):
            return
        if any(value is None for value in values):
            raise ProtocolError("Action gain metrics must be all present or all unavailable.")
        for name in ("action_score", "baseline_score", "gain"):
            object.__setattr__(self, name, finite(getattr(self, name), name))
        if abs(float(self.gain) - (float(self.action_score) - float(self.baseline_score))) > 1.0e-12:
            raise ProtocolError("Action gain is not candidate pooled BACC minus B.")

    def to_payload(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class StaticSelection:
    target_center: str
    method_id: str
    action_id: str
    selected_source: str | None
    selected_gain: float
    baseline_score: float | None
    selected_score: float | None
    score_type: str
    action_gains: tuple[ActionGain, ...]
    label_case_ids: tuple[str, ...]
    label_case_keys: tuple[tuple[str, str], ...]
    label_scope: str
    prerequisite_seal_hash: str
    fallback_reason: str | None
    selection_hash: str = field(init=False)
    held_evaluation_labels_used: bool = False

    def __post_init__(self) -> None:
        if self.target_center not in CENTERS or self.method_id not in {"S4", "G_static"}:
            raise ProtocolError("Static selection identity drifted.")
        legal = decision_action_ids(self.target_center)
        if self.action_id not in legal or self.held_evaluation_labels_used is not False:
            raise ProtocolError("Static selection escaped the legal pre-evaluation action set.")
        expected_source = None if self.action_id == B_ACTION_ID else source_from_action(self.action_id)
        if self.selected_source != expected_source:
            raise ProtocolError("Static selection source identity drifted.")
        gains = tuple(self.action_gains)
        if tuple(row.action_id for row in gains) != legal:
            raise ProtocolError("Static selection must retain B plus all eight A1 gains in order.")
        labels = tuple(sorted(str(value) for value in self.label_case_ids))
        case_keys = tuple(sorted((str(center), str(case)) for center, case in self.label_case_keys))
        expected_labels = (
            tuple(case for _center, case in case_keys)
            if self.method_id == "S4"
            else tuple(f"{center}::{case}" for center, case in case_keys)
        )
        if not case_keys or len(case_keys) != len(set(case_keys)) or labels != tuple(sorted(expected_labels)):
            raise ProtocolError("Static selection label cases must be unique and non-empty.")
        require_sha256(self.prerequisite_seal_hash, "prerequisite_seal_hash")
        object.__setattr__(self, "selected_gain", finite(self.selected_gain, "selected_gain"))
        if self.score_type not in {
            "pooled_exact_bacc",
            "equal_center_mean_of_per_q_pooled_exact_bacc",
        } or any(row.score_type != self.score_type for row in gains):
            raise ProtocolError("Static-selection score type drifted.")
        if (self.baseline_score is None) != (self.selected_score is None):
            raise ProtocolError("Static-selection scores must be jointly available.")
        if self.baseline_score is not None:
            object.__setattr__(self, "baseline_score", finite(self.baseline_score, "baseline_score"))
            object.__setattr__(self, "selected_score", finite(self.selected_score, "selected_score"))
        chosen = gains[legal.index(self.action_id)]
        unavailable = chosen.gain is None
        mismatch = (
            (not unavailable and abs(self.selected_gain - float(chosen.gain)) > 1.0e-12)
            or (not unavailable and abs(float(self.baseline_score) - float(chosen.baseline_score)) > 1.0e-12)
            or (not unavailable and abs(float(self.selected_score) - float(chosen.action_score)) > 1.0e-12)
            or (unavailable and (self.baseline_score is not None or self.selected_score is not None))
            or ((self.action_id == B_ACTION_ID) != (self.fallback_reason is not None))
        )
        if mismatch:
            raise ProtocolError("Static selection does not match its retained action scores.")
        object.__setattr__(self, "action_gains", gains)
        object.__setattr__(self, "label_case_ids", labels)
        object.__setattr__(self, "label_case_keys", case_keys)
        object.__setattr__(self, "selection_hash", canonical_hash(self._unhashed()))

    @property
    def exact_gain(self) -> float:
        return self.selected_gain

    @property
    def fallback_to_b(self) -> bool:
        return self.action_id == B_ACTION_ID

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_support_static_router_static_selection_v1",
            "target_center": self.target_center,
            "method_id": self.method_id,
            "action_id": self.action_id,
            "selected_source": self.selected_source,
            "selected_gain": self.selected_gain,
            "baseline_score": self.baseline_score,
            "selected_score": self.selected_score,
            "score_type": self.score_type,
            "action_gains": [row.to_payload() for row in self.action_gains],
            "label_case_ids": list(self.label_case_ids),
            "label_case_keys": [list(value) for value in self.label_case_keys],
            "label_scope": self.label_scope,
            "prerequisite_seal_hash": self.prerequisite_seal_hash,
            "fallback_reason": self.fallback_reason,
            "held_evaluation_labels_used": False,
            "U_is_control_not_candidate": True,
            "per_case_bacc_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "selection_hash": self.selection_hash}


@dataclass(frozen=True)
class GStaticSeal:
    selections: tuple[StaticSelection, ...]
    probability_seal_hash: str
    seal_hash: str

    def __post_init__(self) -> None:
        values = tuple(self.selections)
        if (
            tuple(row.target_center for row in values) != CENTERS
            or any(row.method_id != "G_static" for row in values)
        ):
            raise ProtocolError("G_static seal must contain nine canonical target selections.")
        require_stable_hash(self.probability_seal_hash, "probability_seal_hash")
        require_sha256(self.seal_hash, "g_static_seal_hash")
        if self.seal_hash != canonical_hash(self._unhashed()):
            raise ProtocolError("G_static seal hash drifted.")
        object.__setattr__(self, "selections", values)

    def selection(self, target_center: object) -> StaticSelection:
        target = str(target_center)
        for row in self.selections:
            if row.target_center == target:
                return row
        raise ProtocolError("G_static target selection is absent.")

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_support_static_router_g_static_seal_v1",
            "probability_seal_hash": self.probability_seal_hash,
            "selections": [row.to_payload() for row in self.selections],
            "source_excluding_target_excluding_LOCO": True,
            "same_H_support_labels_used": False,
            "held_evaluation_labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "seal_hash": self.seal_hash}


@dataclass(frozen=True)
class RouteDecision:
    target_center: str
    fold_ordinal: int
    fold_hash: str
    support_case_ids: tuple[str, ...]
    evaluation_case_ids: tuple[str, ...]
    g_static: StaticSelection
    s4: StaticSelection
    g_static_seal_hash: str
    probability_seal_hash: str
    route_decision_hash: str = field(init=False)
    held_evaluation_labels_used: bool = False

    def __post_init__(self) -> None:
        if (
            self.target_center not in CENTERS
            or isinstance(self.fold_ordinal, bool)
            or self.fold_ordinal not in range(OOF_FOLD_COUNT)
            or self.g_static.target_center != self.target_center
            or self.g_static.method_id != "G_static"
            or self.s4.target_center != self.target_center
            or self.s4.method_id != "S4"
            or self.held_evaluation_labels_used is not False
        ):
            raise ProtocolError("Route decision identity or label boundary drifted.")
        support = tuple(sorted(self.support_case_ids))
        evaluation = tuple(sorted(self.evaluation_case_ids))
        if not support or not evaluation or set(support) & set(evaluation):
            raise ProtocolError("Route decision support/evaluation cases are not disjoint.")
        if support != self.s4.label_case_ids or self.s4.label_case_keys != tuple(
            (self.target_center, case_id) for case_id in support
        ):
            raise ProtocolError("Route decision S4 selection is not bound to its support cases.")
        for row in self.g_static.action_gains:
            if row.action_id == B_ACTION_ID:
                expected_donors = set(CENTERS) - {self.target_center}
            else:
                expected_donors = set(CENTERS) - {
                    self.target_center,
                    source_from_action(row.action_id),
                }
            if set(row.donor_centers) != expected_donors:
                raise ProtocolError("G_static action gain violates H/e donor exclusion.")
        require_sha256(self.fold_hash, "fold_hash")
        require_sha256(self.g_static_seal_hash, "g_static_seal_hash")
        require_stable_hash(self.probability_seal_hash, "probability_seal_hash")
        object.__setattr__(self, "support_case_ids", support)
        object.__setattr__(self, "evaluation_case_ids", evaluation)
        object.__setattr__(self, "route_decision_hash", canonical_hash(self._unhashed()))

    @property
    def method_actions(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                "B": B_ACTION_ID,
                "U": U_ACTION_ID,
                "G_static": self.g_static.action_id,
                "S4": self.s4.action_id,
            }
        )

    @property
    def route_key(self) -> tuple[str, int]:
        return self.target_center, self.fold_ordinal

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_support_static_router_route_decision_v1",
            "target_center": self.target_center,
            "fold_ordinal": self.fold_ordinal,
            "fold_hash": self.fold_hash,
            "support_case_ids": list(self.support_case_ids),
            "evaluation_case_ids": list(self.evaluation_case_ids),
            "method_actions": dict(self.method_actions),
            "g_static_selection_hash": self.g_static.selection_hash,
            "g_static_seal_hash": self.g_static_seal_hash,
            "s4_selection_hash": self.s4.selection_hash,
            "probability_seal_hash": self.probability_seal_hash,
            "held_evaluation_labels_used": False,
            "terminal_oracles_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "route_decision_hash": self.route_decision_hash}


@dataclass(frozen=True)
class DecisionSeal:
    decisions: tuple[RouteDecision, ...]
    partition_hash: str
    probability_seal_hash: str
    decision_seal_hash: str

    def __post_init__(self) -> None:
        decisions = tuple(self.decisions)
        expected = tuple((center, fold) for center in CENTERS for fold in range(OOF_FOLD_COUNT))
        if tuple(row.route_key for row in decisions) != expected:
            raise ProtocolError("Decision seal must contain all 45 routes in canonical order.")
        if any(row.probability_seal_hash != self.probability_seal_hash for row in decisions):
            raise ProtocolError("Decision seal mixes probability surfaces.")
        require_sha256(self.partition_hash, "partition_hash")
        require_stable_hash(self.probability_seal_hash, "probability_seal_hash")
        require_sha256(self.decision_seal_hash, "decision_seal_hash")
        if self.decision_seal_hash != canonical_hash(self._unhashed()):
            raise ProtocolError("Decision seal hash drifted.")
        object.__setattr__(self, "decisions", decisions)

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_support_static_router_all_route_decision_seal_v1",
            "partition_hash": self.partition_hash,
            "probability_seal_hash": self.probability_seal_hash,
            "decisions": [row.to_payload() for row in self.decisions],
            "all_route_decisions_sealed_before_terminal_aggregation": True,
            "each_route_decision_sealed_before_own_evaluation_labels": True,
            "evaluation_labels_used": False,
            "terminal_oracles_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "decision_seal_hash": self.decision_seal_hash}


@dataclass(frozen=True, order=True)
class NullRouteSelection:
    target_center: str
    fold_ordinal: int
    permutation_index: int
    action_id: str
    selected_gain: float

    def __post_init__(self) -> None:
        if (
            self.target_center not in CENTERS
            or self.fold_ordinal not in range(OOF_FOLD_COUNT)
            or isinstance(self.permutation_index, bool)
            or self.permutation_index < 0
            or self.action_id not in decision_action_ids(self.target_center)
        ):
            raise ProtocolError("Null route selection identity drifted.")
        object.__setattr__(self, "selected_gain", finite(self.selected_gain, "selected_gain"))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_support_static_router_null_route_selection_v1",
            **self.__dict__,
            "evaluation_labels_used": False,
        }


@dataclass(frozen=True)
class NullSelectionPlan:
    target_center: str
    fold_ordinal: int
    fold_hash: str
    permutation_seed: int
    permutation_count: int
    support_counts_hash: str
    prerequisite_seal_hash: str
    selections: tuple[NullRouteSelection, ...]
    plan_hash: str

    def __post_init__(self) -> None:
        values = tuple(self.selections)
        if (
            self.target_center not in CENTERS
            or self.fold_ordinal not in range(OOF_FOLD_COUNT)
            or isinstance(self.permutation_seed, bool)
            or not isinstance(self.permutation_seed, int)
            or self.permutation_count <= 0
            or len(values) != self.permutation_count
            or tuple(row.permutation_index for row in values) != tuple(range(self.permutation_count))
            or any(
                row.target_center != self.target_center or row.fold_ordinal != self.fold_ordinal
                for row in values
            )
        ):
            raise ProtocolError("Null selection plan topology drifted.")
        for name in ("fold_hash", "support_counts_hash", "prerequisite_seal_hash", "plan_hash"):
            require_sha256(getattr(self, name), name)
        if self.plan_hash != canonical_hash(self._unhashed()):
            raise ProtocolError("Null selection plan hash drifted.")
        object.__setattr__(self, "selections", values)

    @property
    def selected_action_ids(self) -> tuple[str, ...]:
        return tuple(row.action_id for row in self.selections)

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_support_static_router_null_selection_plan_v1",
            "target_center": self.target_center,
            "fold_ordinal": self.fold_ordinal,
            "fold_hash": self.fold_hash,
            "permutation_seed": self.permutation_seed,
            "permutation_count": self.permutation_count,
            "support_counts_hash": self.support_counts_hash,
            "prerequisite_seal_hash": self.prerequisite_seal_hash,
            "selections": [row.to_payload() for row in self.selections],
            "complete_candidate_blocks_permuted_within_support_case": True,
            "baseline_block_permuted": False,
            "nonzero_cyclic_shifts_only": True,
            "evaluation_labels_used": False,
            "descriptive_exceedance_only": True,
            "p_value_computed": False,
            "gate_computed": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "plan_hash": self.plan_hash}


__all__ = (
    "ActionGain",
    "BinaryLabelRow",
    "BinaryPredictionRow",
    "CaseActionCounts",
    "CaseConfusionCounts",
    "CaseIdentityRow",
    "DecisionSeal",
    "GStaticSeal",
    "NullRouteSelection",
    "NullSelectionPlan",
    "PooledBacc",
    "RouteDecision",
    "StaticSelection",
)
