"""Pure contracts for the fresh residual-top-up Stage-70 evaluation.

This package deliberately starts at an already-frozen action menu.  It does
not fit a proxy, choose an action, or expose an oracle-derived replacement
policy.  Target labels are accepted only by :mod:`scoring`, after the complete
prediction menu has been bound to an opaque seal capability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ...protocol import ProtocolError


BASE_ACTION_ID = "base_equal_union"
UNIFORM_ACTION_ID = "uniform_residual_topup"
GLOBAL_ACTION_ID = "global_rank_residual_topup"
SUPPORT_ACTION_ID = "support_rank_residual_topup"
PERMUTATION_ACTION_ID = "support_rank_permutation_control"
SINGLE_SOURCE_TAIL_PREFIX = "single_source_tail::"

# Short aliases match the predeclared B/U/G/S/P notation in reports.
B_ACTION_ID = BASE_ACTION_ID
U_ACTION_ID = UNIFORM_ACTION_ID
G_ACTION_ID = GLOBAL_ACTION_ID
S_ACTION_ID = SUPPORT_ACTION_ID
P_ACTION_ID = PERMUTATION_ACTION_ID

CORE_ACTION_IDS = (
    BASE_ACTION_ID,
    UNIFORM_ACTION_ID,
    GLOBAL_ACTION_ID,
    SUPPORT_ACTION_ID,
    PERMUTATION_ACTION_ID,
)
PRIMARY_ACTION_IDS = (
    BASE_ACTION_ID,
    UNIFORM_ACTION_ID,
    GLOBAL_ACTION_ID,
    SUPPORT_ACTION_ID,
)

BASE_BUDGET_PER_CLASS = 1024
MATCHED_BUDGET_PER_CLASS = 1152
BASE_PER_SOURCE = 128
TOPUP_TOTAL_PER_CLASS = 128
PROBABILITY_THRESHOLD = 0.5

PRIMARY_ENDPOINT = "all_nine_seed_probability_ensemble_bacc"
DESCRIPTIVE_SEED_ENDPOINT = "paired_seed_cell_mean_bacc_descriptive_only"
INFERENCE_UNIT = "target_center"
CONFIDENCE_LEVEL = 0.95
TIE_ATOL = 1.0e-12

PRIMARY_CONTRASTS = (
    ("S-U", SUPPORT_ACTION_ID, UNIFORM_ACTION_ID),
    ("S-G", SUPPORT_ACTION_ID, GLOBAL_ACTION_ID),
)
SECONDARY_CONTRASTS = (
    ("G-U", GLOBAL_ACTION_ID, UNIFORM_ACTION_ID),
    ("U-B", UNIFORM_ACTION_ID, BASE_ACTION_ID),
    ("S-B", SUPPORT_ACTION_ID, BASE_ACTION_ID),
)
PERMUTATION_CONTRAST = (
    "S-P",
    SUPPORT_ACTION_ID,
    PERMUTATION_ACTION_ID,
)

EXPECTED_SEED_CELL_COUNT = len(TRAINING_SEEDS) * len(GENERATION_SEEDS)
EXPECTED_ACTION_COUNT_PER_TARGET = len(CORE_ACTION_IDS) + len(CENTERS) - 1
EXPECTED_PLAN_CELL_COUNT = (
    len(CENTERS) * EXPECTED_ACTION_COUNT_PER_TARGET * EXPECTED_SEED_CELL_COUNT
)
EXPECTED_ENSEMBLE_METRIC_COUNT = len(CENTERS) * EXPECTED_ACTION_COUNT_PER_TARGET


def tail_action_id(source_center: object) -> str:
    """Return the predeclared H x e single-source-tail action identity."""

    source = str(source_center)
    if source not in CENTERS:
        raise ProtocolError("Single-source-tail action uses an unknown source.")
    return f"{SINGLE_SOURCE_TAIL_PREFIX}{source}"


def tail_source(action_id: object) -> str | None:
    action = str(action_id)
    if not action.startswith(SINGLE_SOURCE_TAIL_PREFIX):
        return None
    source = action[len(SINGLE_SOURCE_TAIL_PREFIX) :]
    if source not in CENTERS:
        raise ProtocolError("Single-source-tail action identity is malformed.")
    return source


def legal_sources(target_center: object) -> tuple[str, ...]:
    target = str(target_center)
    if target not in CENTERS:
        raise ProtocolError("Fresh Stage-70 target center is unknown.")
    return tuple(center for center in CENTERS if center != target)


def expected_action_ids(target_center: object) -> tuple[str, ...]:
    return CORE_ACTION_IDS + tuple(
        tail_action_id(source) for source in legal_sources(target_center)
    )


def _immutable_counts(
    values: Mapping[object, Mapping[object, object]],
) -> Mapping[int, Mapping[str, int]]:
    if not isinstance(values, Mapping):
        raise ProtocolError("Frozen action source counts must be a mapping.")
    normalized: dict[int, Mapping[str, int]] = {}
    for raw_label, raw_counts in values.items():
        if isinstance(raw_label, bool):
            raise ProtocolError("Frozen action class labels must be 0 and 1.")
        try:
            label = int(raw_label)
        except (TypeError, ValueError) as exc:
            raise ProtocolError("Frozen action class labels must be 0 and 1.") from exc
        if label in normalized or not isinstance(raw_counts, Mapping):
            raise ProtocolError("Frozen action source-count geometry is malformed.")
        counts: dict[str, int] = {}
        for raw_source, raw_count in raw_counts.items():
            source = str(raw_source)
            if (
                not source
                or source.strip() != source
                or source in counts
                or isinstance(raw_count, bool)
            ):
                raise ProtocolError("Frozen action source counts are malformed.")
            try:
                count = int(raw_count)
            except (TypeError, ValueError) as exc:
                raise ProtocolError("Frozen action source counts must be integers.") from exc
            if count < 0 or float(raw_count) != float(count):
                raise ProtocolError(
                    "Frozen action source counts must be non-negative integers."
                )
            counts[source] = count
        normalized[label] = MappingProxyType(counts)
    if set(normalized) != {0, 1}:
        raise ProtocolError("Frozen actions must bind both class budgets.")
    return MappingProxyType(normalized)


def _immutable_float_mapping(values: Mapping[object, object]) -> Mapping[str, float]:
    if not isinstance(values, Mapping):
        raise ProtocolError("Frozen support scores must be a mapping.")
    output: dict[str, float] = {}
    for raw_key, raw_value in values.items():
        key = str(raw_key)
        if not key or key.strip() != key or key in output or isinstance(raw_value, bool):
            raise ProtocolError("Frozen support scores are malformed.")
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProtocolError("Frozen support scores must be finite.") from exc
        if not math.isfinite(value):
            raise ProtocolError("Frozen support scores must be finite.")
        output[key] = value
    return MappingProxyType(output)


def _immutable_string_mapping(values: Mapping[object, object]) -> Mapping[str, str]:
    if not isinstance(values, Mapping):
        raise ProtocolError("Frozen source-identity permutation must be a mapping.")
    output: dict[str, str] = {}
    for raw_key, raw_value in values.items():
        key = str(raw_key)
        value = str(raw_value)
        if (
            not key
            or key.strip() != key
            or not value
            or value.strip() != value
            or key in output
        ):
            raise ProtocolError("Frozen source-identity permutation is malformed.")
        output[key] = value
    return MappingProxyType(output)


@dataclass(frozen=True)
class FrozenActionPayload:
    """One label-free action frozen before fresh target evaluation.

    ``source_counts_by_class`` is the final synthetic allocation, including
    the equal-union base.  G/S/P carry their generic lower-is-better mean
    normalized-midrank/Borda inputs.  The P action additionally carries the
    fixed source-identity permutation used to produce its counts.
    """

    target_center: str
    action_id: str
    source_counts_by_class: Mapping[int, Mapping[str, int]]
    action_hash: str
    mean_normalized_midrank_by_source: Mapping[str, float] = field(
        default_factory=dict
    )
    source_identity_permutation: Mapping[str, str] = field(default_factory=dict)
    normalized_midrank_semantics: str = "lower_is_better"
    frozen_before_label_access: bool = True

    def __post_init__(self) -> None:
        target = str(self.target_center)
        action = str(self.action_id)
        action_hash = str(self.action_hash)
        if target not in CENTERS:
            raise ProtocolError("Frozen action target center is unknown.")
        if not action or action.strip() != action:
            raise ProtocolError("Frozen action identity is invalid.")
        if not action_hash or action_hash.strip() != action_hash:
            raise ProtocolError("Frozen action hash is required.")
        if self.frozen_before_label_access is not True:
            raise ProtocolError("Stage-70 actions must be frozen before labels.")
        if self.normalized_midrank_semantics != "lower_is_better":
            raise ProtocolError(
                "Fresh Stage-70 normalized midranks must be lower-is-better."
            )
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "action_id", action)
        object.__setattr__(self, "action_hash", action_hash)
        object.__setattr__(
            self,
            "source_counts_by_class",
            _immutable_counts(self.source_counts_by_class),
        )
        object.__setattr__(
            self,
            "mean_normalized_midrank_by_source",
            _immutable_float_mapping(self.mean_normalized_midrank_by_source),
        )
        object.__setattr__(
            self,
            "source_identity_permutation",
            _immutable_string_mapping(self.source_identity_permutation),
        )

    @property
    def budget_per_class(self) -> int:
        totals = {
            sum(self.source_counts_by_class[label].values()) for label in (0, 1)
        }
        if len(totals) != 1:
            raise ProtocolError("Frozen action class budgets disagree.")
        return totals.pop()

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_residual_topup_fresh_action_v1",
            "target_center": self.target_center,
            "action_id": self.action_id,
            "source_counts_by_class": {
                str(label): dict(self.source_counts_by_class[label])
                for label in (0, 1)
            },
            "budget_per_class": self.budget_per_class,
            "action_hash": self.action_hash,
            "mean_normalized_midrank_by_source": dict(
                self.mean_normalized_midrank_by_source
            ),
            "source_identity_permutation": dict(
                self.source_identity_permutation
            ),
            "normalized_midrank_semantics": "lower_is_better",
            "frozen_before_label_access": True,
            "target_labels_used": False,
        }


@dataclass(frozen=True)
class EvaluationCell:
    target_center: str
    training_seed: int
    generation_seed: int
    action_id: str
    action_hash: str

    @property
    def key(self) -> tuple[str, int, int, str]:
        return (
            self.target_center,
            self.training_seed,
            self.generation_seed,
            self.action_id,
        )


@dataclass(frozen=True)
class EvaluationPlan:
    actions_by_target: Mapping[str, tuple[FrozenActionPayload, ...]]
    cells: tuple[EvaluationCell, ...]
    evaluation_row_ids_by_target: Mapping[str, tuple[str, ...]]
    plan_hash: str
    primary_endpoint: str = PRIMARY_ENDPOINT
    seed_cell_endpoint_role: str = DESCRIPTIVE_SEED_ENDPOINT

    def __post_init__(self) -> None:
        if self.primary_endpoint != PRIMARY_ENDPOINT:
            raise ProtocolError("Fresh Stage-70 primary endpoint drifted.")
        if self.seed_cell_endpoint_role != DESCRIPTIVE_SEED_ENDPOINT:
            raise ProtocolError("Fresh Stage-70 seed-cell endpoint role drifted.")
        if len(self.cells) != EXPECTED_PLAN_CELL_COUNT:
            raise ProtocolError("Fresh Stage-70 plan coverage is incomplete.")

    def action_for(self, target_center: str, action_id: str) -> FrozenActionPayload:
        for action in self.actions_by_target[str(target_center)]:
            if action.action_id == str(action_id):
                return action
        raise ProtocolError("Fresh Stage-70 action is absent from the frozen plan.")


@dataclass(frozen=True)
class PredictionCell:
    """One label-free probability vector for a frozen action/seed cell."""

    target_center: str
    training_seed: int
    generation_seed: int
    action_id: str
    action_hash: str
    evaluation_row_ids: tuple[str, ...]
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        row_ids = tuple(str(value) for value in self.evaluation_row_ids)
        probabilities = np.ascontiguousarray(self.probabilities, dtype=np.float64)
        if (
            self.target_center not in CENTERS
            or self.training_seed not in TRAINING_SEEDS
            or self.generation_seed not in GENERATION_SEEDS
            or not self.action_id
            or not self.action_hash
            or not row_ids
            or len(row_ids) != len(set(row_ids))
            or probabilities.shape != (len(row_ids),)
            or not np.isfinite(probabilities).all()
            or bool(np.any(probabilities < 0.0))
            or bool(np.any(probabilities > 1.0))
        ):
            raise ProtocolError("Fresh Stage-70 prediction cell is malformed.")
        if any(not row_id or row_id.strip() != row_id for row_id in row_ids):
            raise ProtocolError("Fresh Stage-70 evaluation row identities are invalid.")
        probabilities.setflags(write=False)
        object.__setattr__(self, "evaluation_row_ids", row_ids)
        object.__setattr__(self, "probabilities", probabilities)

    @property
    def key(self) -> tuple[str, int, int, str]:
        return (
            self.target_center,
            self.training_seed,
            self.generation_seed,
            self.action_id,
        )


ActionPrediction = PredictionCell


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
    probability_aggregation: str = (
        "arithmetic_mean_all_nine_seed_probabilities_no_seed_selection"
    )

    @property
    def key(self) -> tuple[str, str]:
        return self.target_center, self.action_id


@dataclass(frozen=True)
class ScoredEvaluation:
    seed_cell_metrics: tuple[SeedCellMetric, ...]
    ensemble_metrics: tuple[EnsembleMetric, ...]
    prediction_seal_hash: str
    primary_endpoint: str = PRIMARY_ENDPOINT
    labels_used_for_scoring_only: bool = True

    def __post_init__(self) -> None:
        if (
            self.primary_endpoint != PRIMARY_ENDPOINT
            or len(self.seed_cell_metrics) != EXPECTED_PLAN_CELL_COUNT
            or len(self.ensemble_metrics) != EXPECTED_ENSEMBLE_METRIC_COUNT
        ):
            raise ProtocolError("Fresh Stage-70 scored coverage is incomplete.")


@dataclass(frozen=True)
class CenterContrast:
    contrast_id: str
    target_center: str
    left_action_id: str
    right_action_id: str
    probability_ensemble_bacc_delta: float
    descriptive_seed_cell_mean_bacc_delta: float
    contrast_role: str
    primary_endpoint: str = PRIMARY_ENDPOINT
    inference_unit: str = INFERENCE_UNIT


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
    primary_endpoint: str = PRIMARY_ENDPOINT
    inference_unit: str = INFERENCE_UNIT


@dataclass(frozen=True)
class OracleDiagnostic:
    """Scalar, diagnostic-only summary of one sealed H x e utility row.

    Source identities and per-source utilities are intentionally absent.  The
    summary cannot be converted into a replacement frozen action by this API.
    """

    target_center: str
    source_count: int
    support_score_utility_spearman: float
    spearman_defined: bool
    top1_agreement: bool
    oracle_headroom_bacc: float
    normalized_oracle_gap: float
    oracle_utility_range_bacc: float
    prediction_seal_hash: str
    diagnostic_only: bool = True
    may_update_frozen_policy: bool = False


@dataclass(frozen=True)
class FreshEvaluationReport:
    scored: ScoredEvaluation
    center_contrasts: tuple[CenterContrast, ...]
    contrast_inference: tuple[ContrastInference, ...]
    oracle_diagnostics: tuple[OracleDiagnostic, ...]
    prediction_seal_hash: str
    primary_endpoint: str = PRIMARY_ENDPOINT
    policy_update_emitted: bool = False


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ProtocolError("Cannot average an empty fresh Stage-70 sequence.")
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ProtocolError("Fresh Stage-70 metrics must be finite.")
    return float(np.mean(array))


__all__ = (
    "BASE_ACTION_ID",
    "BASE_BUDGET_PER_CLASS",
    "BASE_PER_SOURCE",
    "B_ACTION_ID",
    "CENTERS",
    "CONFIDENCE_LEVEL",
    "CORE_ACTION_IDS",
    "DESCRIPTIVE_SEED_ENDPOINT",
    "EXPECTED_ACTION_COUNT_PER_TARGET",
    "EXPECTED_ENSEMBLE_METRIC_COUNT",
    "EXPECTED_PLAN_CELL_COUNT",
    "EXPECTED_SEED_CELL_COUNT",
    "EvaluationCell",
    "EvaluationPlan",
    "FreshEvaluationReport",
    "FrozenActionPayload",
    "GENERATION_SEEDS",
    "GLOBAL_ACTION_ID",
    "G_ACTION_ID",
    "INFERENCE_UNIT",
    "MATCHED_BUDGET_PER_CLASS",
    "PERMUTATION_ACTION_ID",
    "PERMUTATION_CONTRAST",
    "PRIMARY_ACTION_IDS",
    "PRIMARY_CONTRASTS",
    "PRIMARY_ENDPOINT",
    "PROBABILITY_THRESHOLD",
    "PredictionCell",
    "ActionPrediction",
    "P_ACTION_ID",
    "SINGLE_SOURCE_TAIL_PREFIX",
    "SECONDARY_CONTRASTS",
    "SUPPORT_ACTION_ID",
    "ScoredEvaluation",
    "SeedCellMetric",
    "EnsembleMetric",
    "CenterContrast",
    "ContrastInference",
    "OracleDiagnostic",
    "S_ACTION_ID",
    "TIE_ATOL",
    "TOPUP_TOTAL_PER_CLASS",
    "TRAINING_SEEDS",
    "UNIFORM_ACTION_ID",
    "U_ACTION_ID",
    "expected_action_ids",
    "legal_sources",
    "mean",
    "tail_action_id",
    "tail_source",
)
