"""Pure contracts for the utility-aligned fresh Stage-70 evaluation.

The package consumes an already-frozen Stage-60 decision surface.  It never
fits or updates a router.  In particular, the H x e menu is a terminal
diagnostic surface and is deliberately absent from every policy output type.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ....common.hashing import stable_hash
from ...expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ...protocol import ProtocolError


BASE_ACTION_ID = "base_equal_union"
UNIFORM_ACTION_ID = "uniform_residual_topup"
GLOBAL_ACTION_ID = "global_delta_single_source_tail"
ROUTED_ACTION_ID = "utility_aligned_residual_tail"
PERMUTATION_ACTION_ID = "target_feature_permutation_control_tail"
SINGLE_SOURCE_TAIL_PREFIX = "single_source_tail::"

# Report-facing notation.  The long action ids remain the serialized identity.
B_ACTION_ID = BASE_ACTION_ID
U_ACTION_ID = UNIFORM_ACTION_ID
G_DELTA_ACTION_ID = GLOBAL_ACTION_ID
R_ACTION_ID = ROUTED_ACTION_ID
P_ACTION_ID = PERMUTATION_ACTION_ID
ACTION_SYMBOL_BY_ID = MappingProxyType(
    {
        BASE_ACTION_ID: "B",
        UNIFORM_ACTION_ID: "U",
        GLOBAL_ACTION_ID: "G_delta",
        ROUTED_ACTION_ID: "R",
        PERMUTATION_ACTION_ID: "P",
    }
)

CORE_ACTION_IDS = (
    BASE_ACTION_ID,
    UNIFORM_ACTION_ID,
    GLOBAL_ACTION_ID,
    ROUTED_ACTION_ID,
    PERMUTATION_ACTION_ID,
)
BASE_PER_SOURCE = 128
BASE_BUDGET_PER_CLASS = 1024
TOPUP_TOTAL_PER_CLASS = 128
MATCHED_BUDGET_PER_CLASS = 1152
PROBABILITY_THRESHOLD = 0.5
EXPECTED_SEED_CELL_COUNT = len(TRAINING_SEEDS) * len(GENERATION_SEEDS)
EXPECTED_ACTION_COUNT_PER_TARGET = len(CORE_ACTION_IDS) + len(CENTERS) - 1
EXPECTED_LOGICAL_PREDICTION_COUNT = (
    len(CENTERS) * EXPECTED_ACTION_COUNT_PER_TARGET * EXPECTED_SEED_CELL_COUNT
)
EXPECTED_ENSEMBLE_METRIC_COUNT = len(CENTERS) * EXPECTED_ACTION_COUNT_PER_TARGET

PRIMARY_ENDPOINT = "all_nine_seed_probability_ensemble_bacc"
DESCRIPTIVE_SEED_ENDPOINT = "paired_seed_cell_bacc_descriptive_only"
INFERENCE_UNIT = "target_center"
CONFIDENCE_LEVEL = 0.95
TIE_ATOL = 1.0e-12

PRIMARY_CONTRASTS = (
    ("R-B", ROUTED_ACTION_ID, BASE_ACTION_ID),
    ("R-G_delta", ROUTED_ACTION_ID, GLOBAL_ACTION_ID),
    ("R-U", ROUTED_ACTION_ID, UNIFORM_ACTION_ID),
)
PERMUTATION_CONTRAST = (
    "R-P",
    ROUTED_ACTION_ID,
    PERMUTATION_ACTION_ID,
)
SECONDARY_CONTRASTS = (
    ("U-B", UNIFORM_ACTION_ID, BASE_ACTION_ID),
    ("G_delta-B", GLOBAL_ACTION_ID, BASE_ACTION_ID),
)


def legal_sources(target_center: object) -> tuple[str, ...]:
    target = str(target_center)
    if target not in CENTERS:
        raise ProtocolError("Utility-aligned Stage-70 target center is unknown.")
    return tuple(center for center in CENTERS if center != target)


def tail_action_id(source_center: object) -> str:
    source = str(source_center)
    if source not in CENTERS:
        raise ProtocolError("Utility-aligned H x e source center is unknown.")
    return f"{SINGLE_SOURCE_TAIL_PREFIX}{source}"


def tail_source(action_id: object) -> str | None:
    rendered = str(action_id)
    if not rendered.startswith(SINGLE_SOURCE_TAIL_PREFIX):
        return None
    source = rendered[len(SINGLE_SOURCE_TAIL_PREFIX) :]
    if source not in CENTERS:
        raise ProtocolError("Utility-aligned H x e action id is malformed.")
    return source


def expected_action_ids(target_center: object) -> tuple[str, ...]:
    return CORE_ACTION_IDS + tuple(
        tail_action_id(source) for source in legal_sources(target_center)
    )


def _immutable_counts(
    values: Mapping[object, Mapping[object, object]],
) -> Mapping[int, Mapping[str, int]]:
    if not isinstance(values, Mapping):
        raise ProtocolError("Frozen utility-aligned counts must be a mapping.")
    normalized: dict[int, Mapping[str, int]] = {}
    for raw_label, raw_counts in values.items():
        if isinstance(raw_label, bool) or not isinstance(raw_counts, Mapping):
            raise ProtocolError("Frozen utility-aligned class counts are malformed.")
        try:
            label = int(raw_label)
        except (TypeError, ValueError) as exc:
            raise ProtocolError("Frozen utility-aligned class labels must be 0/1.") from exc
        counts: dict[str, int] = {}
        for raw_source, raw_count in raw_counts.items():
            source = str(raw_source)
            if isinstance(raw_count, bool):
                raise ProtocolError("Frozen utility-aligned counts must be integers.")
            try:
                count = int(raw_count)
            except (TypeError, ValueError) as exc:
                raise ProtocolError("Frozen utility-aligned counts must be integers.") from exc
            try:
                exact = float(raw_count) == float(count)
            except (TypeError, ValueError, OverflowError):
                exact = False
            if not source or source in counts or count < 0 or not exact:
                raise ProtocolError("Frozen utility-aligned source counts are malformed.")
            counts[source] = count
        if label in normalized:
            raise ProtocolError("Frozen utility-aligned class labels duplicate.")
        normalized[label] = MappingProxyType(counts)
    if set(normalized) != {0, 1}:
        raise ProtocolError("Frozen utility-aligned actions require both classes.")
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class FrozenActionPayload:
    """One label-free logical action admitted from the Stage-60 lock."""

    target_center: str
    action_id: str
    action_role: str
    source_counts_by_class: Mapping[int, Mapping[str, int]]
    action_hash: str
    selected_source: str | None = None
    abstained_to_base: bool = False
    fallback_reason: str | None = None
    target_labels_used: bool = False
    support_labels_used: bool = False

    def __post_init__(self) -> None:
        target = str(self.target_center)
        action = str(self.action_id)
        selected = None if self.selected_source is None else str(self.selected_source)
        if target not in CENTERS or not action or not self.action_hash:
            raise ProtocolError("Frozen utility-aligned action identity is malformed.")
        if self.target_labels_used is not False or self.support_labels_used is not False:
            raise ProtocolError("Fresh Stage-70 actions cannot consume target labels.")
        if selected is not None and selected not in legal_sources(target):
            raise ProtocolError("Frozen utility-aligned action selected an illegal source.")
        if self.abstained_to_base and selected is not None:
            raise ProtocolError("An abstained utility-aligned action cannot select a source.")
        if self.abstained_to_base and not self.fallback_reason:
            raise ProtocolError("Utility-aligned abstention must preserve its reason.")
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "action_id", action)
        object.__setattr__(self, "selected_source", selected)
        object.__setattr__(
            self,
            "source_counts_by_class",
            _immutable_counts(self.source_counts_by_class),
        )

    @property
    def budget_per_class(self) -> int:
        totals = {
            sum(self.source_counts_by_class[label].values()) for label in (0, 1)
        }
        if len(totals) != 1:
            raise ProtocolError("Frozen utility-aligned class budgets disagree.")
        return totals.pop()

    @property
    def composition_hash(self) -> str:
        return stable_hash(
            {
                "schema_version": "midogpp_utility_aligned_composition_v1",
                "target_center": self.target_center,
                "source_counts_by_class": {
                    str(label): dict(self.source_counts_by_class[label])
                    for label in (0, 1)
                },
            }
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_utility_aligned_fresh_action_v1",
            "target_center": self.target_center,
            "action_id": self.action_id,
            "action_role": self.action_role,
            "selected_source": self.selected_source,
            "abstained_to_base": self.abstained_to_base,
            "fallback_reason": self.fallback_reason,
            "source_counts_by_class": {
                str(label): dict(self.source_counts_by_class[label])
                for label in (0, 1)
            },
            "budget_per_class": self.budget_per_class,
            "action_hash": self.action_hash,
            "composition_hash": self.composition_hash,
            "target_labels_used": False,
            "support_labels_used": False,
            "frozen_before_stage70": True,
        }


@dataclass(frozen=True)
class EvaluationCell:
    target_center: str
    training_seed: int
    generation_seed: int
    action_id: str
    action_hash: str
    composition_hash: str

    @property
    def key(self) -> tuple[str, int, int, str]:
        return (
            self.target_center,
            self.training_seed,
            self.generation_seed,
            self.action_id,
        )


@dataclass(frozen=True)
class CompositionCell:
    target_center: str
    training_seed: int
    generation_seed: int
    composition_hash: str
    representative_action_id: str

    @property
    def key(self) -> tuple[str, int, int, str]:
        return (
            self.target_center,
            self.training_seed,
            self.generation_seed,
            self.composition_hash,
        )


@dataclass(frozen=True)
class EvaluationPlan:
    actions_by_target: Mapping[str, tuple[FrozenActionPayload, ...]]
    logical_cells: tuple[EvaluationCell, ...]
    composition_cells: tuple[CompositionCell, ...]
    evaluation_row_ids_by_target: Mapping[str, tuple[str, ...]]
    plan_hash: str

    def __post_init__(self) -> None:
        if len(self.logical_cells) != EXPECTED_LOGICAL_PREDICTION_COUNT:
            raise ProtocolError("Utility-aligned logical prediction coverage drifted.")
        if len(self.composition_cells) > len(self.logical_cells):
            raise ProtocolError("Utility-aligned composition deduplication drifted.")

    @property
    def cells(self) -> tuple[EvaluationCell, ...]:
        """Compatibility alias for action-agnostic consumers."""

        return self.logical_cells

    def action_for(self, target_center: str, action_id: str) -> FrozenActionPayload:
        for action in self.actions_by_target[str(target_center)]:
            if action.action_id == str(action_id):
                return action
        raise ProtocolError("Utility-aligned action is absent from the frozen plan.")


@dataclass(frozen=True)
class PredictionCell:
    """One sealed logical prediction; arrays remain compact float32."""

    target_center: str
    training_seed: int
    generation_seed: int
    action_id: str
    action_hash: str
    composition_hash: str
    evaluation_row_ids: tuple[str, ...]
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        rows = tuple(str(value) for value in self.evaluation_row_ids)
        probabilities = np.ascontiguousarray(self.probabilities, dtype=np.float32)
        if (
            self.target_center not in CENTERS
            or self.training_seed not in TRAINING_SEEDS
            or self.generation_seed not in GENERATION_SEEDS
            or not self.action_id
            or not self.action_hash
            or not self.composition_hash
            or not rows
            or len(rows) != len(set(rows))
            or probabilities.shape != (len(rows),)
            or not np.isfinite(probabilities).all()
            or bool(np.any(probabilities < 0.0))
            or bool(np.any(probabilities > 1.0))
        ):
            raise ProtocolError("Utility-aligned prediction cell is malformed.")
        probabilities.setflags(write=False)
        object.__setattr__(self, "evaluation_row_ids", rows)
        object.__setattr__(self, "probabilities", probabilities)

    @property
    def key(self) -> tuple[str, int, int, str]:
        return (
            self.target_center,
            self.training_seed,
            self.generation_seed,
            self.action_id,
        )


@dataclass(frozen=True)
class SeedCellMetric:
    target_center: str
    training_seed: int
    generation_seed: int
    action_id: str
    bacc: float
    macro_f1: float
    evaluation_row_count: int
    prediction_seal_hash: str
    endpoint_role: str = DESCRIPTIVE_SEED_ENDPOINT
    descriptive_only: bool = True

    @property
    def key(self) -> tuple[str, int, int, str]:
        return (
            self.target_center,
            self.training_seed,
            self.generation_seed,
            self.action_id,
        )


@dataclass(frozen=True)
class EnsembleMetric:
    target_center: str
    action_id: str
    bacc: float
    macro_f1: float
    evaluation_row_count: int
    seed_cell_count: int
    prediction_seal_hash: str
    endpoint: str = PRIMARY_ENDPOINT
    primary_endpoint: bool = True

    @property
    def key(self) -> tuple[str, str]:
        return self.target_center, self.action_id


@dataclass(frozen=True)
class ScoredEvaluation:
    seed_cell_metrics: tuple[SeedCellMetric, ...]
    ensemble_metrics: tuple[EnsembleMetric, ...]
    prediction_seal_hash: str
    labels_used_for_scoring_only: bool = True

    def __post_init__(self) -> None:
        if (
            len(self.seed_cell_metrics) != EXPECTED_LOGICAL_PREDICTION_COUNT
            or len(self.ensemble_metrics) != EXPECTED_ENSEMBLE_METRIC_COUNT
            or self.labels_used_for_scoring_only is not True
        ):
            raise ProtocolError("Utility-aligned scored coverage drifted.")


@dataclass(frozen=True)
class CenterContrast:
    contrast_id: str
    target_center: str
    left_action_id: str
    right_action_id: str
    probability_ensemble_bacc_delta: float
    descriptive_seed_cell_mean_bacc_delta: float
    contrast_role: str


@dataclass(frozen=True)
class ContrastInference:
    contrast_id: str
    mean_probability_ensemble_bacc_delta: float
    two_sided_95_ci_low: float
    two_sided_95_ci_high: float
    one_sided_95_lcb: float
    wins: int
    ties: int
    losses: int
    center_count: int
    contrast_role: str


@dataclass(frozen=True)
class OracleDiagnostic:
    target_center: str
    routed_source: str | None
    oracle_source: str
    routed_top1_agreement: bool
    base_bacc: float
    routed_bacc: float
    oracle_bacc: float
    oracle_headroom_over_base_bacc: float
    routed_oracle_gap_bacc: float
    normalized_routed_oracle_gap: float
    prediction_seal_hash: str
    diagnostic_only: bool = True
    may_update_frozen_policy: bool = False

    def __post_init__(self) -> None:
        values = (
            self.base_bacc,
            self.routed_bacc,
            self.oracle_bacc,
            self.oracle_headroom_over_base_bacc,
            self.routed_oracle_gap_bacc,
            self.normalized_routed_oracle_gap,
        )
        if not all(math.isfinite(value) for value in values):
            raise ProtocolError("Utility-aligned oracle diagnostics must be finite.")
        if not self.diagnostic_only or self.may_update_frozen_policy:
            raise ProtocolError("H x e diagnostics cannot update the frozen router.")


@dataclass(frozen=True)
class FreshEvaluationReport:
    scored: ScoredEvaluation
    center_contrasts: tuple[CenterContrast, ...]
    contrast_inference: tuple[ContrastInference, ...]
    oracle_diagnostics: tuple[OracleDiagnostic, ...]
    prediction_seal_hash: str
    policy_update_emitted: bool = False

    def __post_init__(self) -> None:
        if self.policy_update_emitted:
            raise ProtocolError("Fresh Stage-70 evaluation cannot update its policy.")


__all__ = (
    "ACTION_SYMBOL_BY_ID",
    "BASE_ACTION_ID",
    "BASE_BUDGET_PER_CLASS",
    "BASE_PER_SOURCE",
    "B_ACTION_ID",
    "CENTERS",
    "CONFIDENCE_LEVEL",
    "CORE_ACTION_IDS",
    "CompositionCell",
    "CenterContrast",
    "ContrastInference",
    "DESCRIPTIVE_SEED_ENDPOINT",
    "EXPECTED_ENSEMBLE_METRIC_COUNT",
    "EXPECTED_ACTION_COUNT_PER_TARGET",
    "EXPECTED_LOGICAL_PREDICTION_COUNT",
    "EXPECTED_SEED_CELL_COUNT",
    "EnsembleMetric",
    "EvaluationCell",
    "EvaluationPlan",
    "FreshEvaluationReport",
    "FrozenActionPayload",
    "G_DELTA_ACTION_ID",
    "GENERATION_SEEDS",
    "GLOBAL_ACTION_ID",
    "INFERENCE_UNIT",
    "MATCHED_BUDGET_PER_CLASS",
    "OracleDiagnostic",
    "PERMUTATION_ACTION_ID",
    "PERMUTATION_CONTRAST",
    "PRIMARY_CONTRASTS",
    "PRIMARY_ENDPOINT",
    "PROBABILITY_THRESHOLD",
    "P_ACTION_ID",
    "PredictionCell",
    "ROUTED_ACTION_ID",
    "R_ACTION_ID",
    "ScoredEvaluation",
    "SECONDARY_CONTRASTS",
    "SINGLE_SOURCE_TAIL_PREFIX",
    "SeedCellMetric",
    "TIE_ATOL",
    "TOPUP_TOTAL_PER_CLASS",
    "TRAINING_SEEDS",
    "UNIFORM_ACTION_ID",
    "U_ACTION_ID",
    "expected_action_ids",
    "legal_sources",
    "tail_action_id",
    "tail_source",
)
