"""Terminal center-level endpoint contrasts and Hxe oracle-rank diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from ...metrics import spearman
from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from ...routing.utility_aligned import ProbabilityEnsembleEndpoint, SeedProbabilityVector
from .actions import (
    FrozenEndpointAction,
    FrozenTargetActionLibrary,
)
from .contracts import (
    BASE_ACTION_ID,
    CENTERS,
    EXPECTED_CONTRAST_ROW_COUNT,
    EXPECTED_EVALUATION_CASE_COUNTS_BY_CENTER,
    EXPECTED_EVALUATION_ROW_COUNTS_BY_CENTER,
    EXPECTED_TERMINAL_SCORE_COUNT,
    GLOBAL_ACTION_ID,
    PERMUTATION_ACTION_ID,
    PRIMARY_CONTRASTS,
    ROUTED_ACTION_ID,
    SEED_PAIRS,
    UNIFORM_ACTION_ID,
    candidate_sources,
    expected_target_action_ids,
    h_x_e_action_id,
)
from .endpoint_adapter import score_sealed_probability_ensemble
from .policy import FrozenTargetPolicySet


@dataclass(frozen=True)
class TerminalEndpointScore:
    """One terminal exact-nine endpoint opened after the global prelabel seal."""

    target_id: str
    action_id: str
    action_hash: str
    policy_hash: str
    support_partition_lock_hash: str
    evaluation_partition_hash: str
    global_target_prediction_seal_hash: str
    global_prelabel_seal_hash: str
    evaluation_case_count: int
    observed_class_row_counts: tuple[int, int]
    endpoint: ProbabilityEnsembleEndpoint
    score_hash: str = field(init=False)

    def __post_init__(self) -> None:
        target = str(self.target_id)
        action = str(self.action_id)
        try:
            class_counts = tuple(int(value) for value in self.observed_class_row_counts)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProtocolError("Terminal class sufficient statistics are invalid.") from exc
        endpoint_payload = (
            self.endpoint.to_payload()
            if isinstance(self.endpoint, ProbabilityEnsembleEndpoint)
            else {}
        )
        endpoint_hash = endpoint_payload.pop("endpoint_hash", None)
        endpoint_valid = bool(endpoint_payload) and endpoint_hash == canonical_sha256(
            endpoint_payload
        )
        endpoint_probabilities = np.asarray(
            getattr(self.endpoint, "mean_positive_probabilities", ())
        )
        endpoint_predictions = np.asarray(getattr(self.endpoint, "predictions", ()))
        if (
            target not in CENTERS
            or action not in expected_target_action_ids(target)
            or not isinstance(self.endpoint, ProbabilityEnsembleEndpoint)
            or not endpoint_valid
            or self.endpoint.seed_keys != SEED_PAIRS
            or len(self.endpoint.component_vector_hashes) != len(SEED_PAIRS)
            or len(set(self.endpoint.component_vector_hashes)) != len(SEED_PAIRS)
            or endpoint_probabilities.ndim != 1
            or endpoint_predictions.ndim != 1
            or len(endpoint_probabilities) != len(endpoint_predictions)
            or not np.isfinite(endpoint_probabilities).all()
            or np.any(endpoint_probabilities < 0.0)
            or np.any(endpoint_probabilities > 1.0)
            or not 0.0 <= self.endpoint.balanced_accuracy <= 1.0
            or not np.array_equal(
                endpoint_predictions,
                (endpoint_probabilities >= self.endpoint.threshold).astype(np.uint8),
            )
            or self.evaluation_case_count
            != EXPECTED_EVALUATION_CASE_COUNTS_BY_CENTER[target]
            or self.endpoint.row_count != EXPECTED_EVALUATION_ROW_COUNTS_BY_CENTER[target]
            or len(class_counts) != 2
            or any(value <= 0 for value in class_counts)
            or sum(class_counts) != self.endpoint.row_count
            or self.support_partition_lock_hash == self.evaluation_partition_hash
            or not all(
                _text(value)
                for value in (
                    self.action_hash,
                    self.policy_hash,
                    self.support_partition_lock_hash,
                    self.evaluation_partition_hash,
                    self.global_target_prediction_seal_hash,
                    self.global_prelabel_seal_hash,
                )
            )
        ):
            raise ProtocolError("Terminal endpoint score boundary drifted.")
        object.__setattr__(self, "target_id", target)
        object.__setattr__(self, "action_id", action)
        object.__setattr__(self, "observed_class_row_counts", class_counts)
        object.__setattr__(self, "score_hash", canonical_sha256(self.to_payload()))

    @property
    def balanced_accuracy(self) -> float:
        return self.endpoint.balanced_accuracy

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_consumed_test_terminal_endpoint_score_v1",
            "target_center": self.target_id,
            "action_id": self.action_id,
            "action_hash": self.action_hash,
            "policy_hash": self.policy_hash,
            "support_partition_lock_hash": self.support_partition_lock_hash,
            "evaluation_partition_hash": self.evaluation_partition_hash,
            "global_target_prediction_seal_hash": self.global_target_prediction_seal_hash,
            "global_prelabel_seal_hash": self.global_prelabel_seal_hash,
            "evaluation_row_identity_hash": self.endpoint.row_identity_hash,
            "evaluation_label_hash": self.endpoint.label_hash,
            "endpoint_hash": self.endpoint.endpoint_hash,
            "evaluation_case_count": self.evaluation_case_count,
            "evaluation_row_count": self.endpoint.row_count,
            "observed_class_0_row_count": self.observed_class_row_counts[0],
            "observed_class_1_row_count": self.observed_class_row_counts[1],
            "balanced_accuracy": self.balanced_accuracy,
            "primary_endpoint": "exact_nine_probability_mean_then_threshold_bacc",
            "same_outer_H_evaluation_labels_opened_after_plan_and_global_seal": True,
            "terminal_scores_may_update_plan": False,
            "inference_unit": "target_center",
            "technical_seed_cells_are_independent_units": False,
            "consumed_test_diagnostic_only": True,
        }


@dataclass(frozen=True)
class TerminalEndpointScoreSet:
    rows: tuple[TerminalEndpointScore, ...]
    action_library_hash: str
    policy_set_hash: str
    action_hash_by_key: Mapping[str, str]
    policy_hash_by_target: Mapping[str, str]
    global_target_prediction_seal_hash: str
    global_prelabel_seal_hash: str
    score_set_hash: str

    def __post_init__(self) -> None:
        action_hashes = {
            str(key): str(value) for key, value in self.action_hash_by_key.items()
        }
        policy_hashes = {
            str(target): str(value)
            for target, value in self.policy_hash_by_target.items()
        }
        expected_keys = tuple(
            (target, action_id)
            for target in CENTERS
            for action_id in expected_target_action_ids(target)
        )
        keys = tuple((row.target_id, row.action_id) for row in self.rows)
        encoded_keys = tuple(_action_key(*key) for key in expected_keys)
        if (
            len(self.rows) != EXPECTED_TERMINAL_SCORE_COUNT
            or keys != expected_keys
            or len(set(keys)) != len(keys)
            or tuple(action_hashes) != encoded_keys
            or tuple(policy_hashes) != CENTERS
            or any(
                not _text(action_hashes[_action_key(row.target_id, row.action_id)])
                or row.action_hash
                != action_hashes[_action_key(row.target_id, row.action_id)]
                for row in self.rows
            )
            or any(
                not _text(policy_hashes[target])
                or row.policy_hash != policy_hashes[target]
                for row in self.rows
                for target in (row.target_id,)
            )
            or any(
                row.global_target_prediction_seal_hash
                != self.global_target_prediction_seal_hash
                or row.global_prelabel_seal_hash != self.global_prelabel_seal_hash
                for row in self.rows
            )
            or not _text(self.action_library_hash)
            or not _text(self.policy_set_hash)
            or not _text(self.global_target_prediction_seal_hash)
            or not _text(self.global_prelabel_seal_hash)
        ):
            raise ProtocolError("Terminal endpoint score set is incomplete.")
        if self.score_set_hash != canonical_sha256(self._unhashed_payload()):
            raise ProtocolError("Terminal endpoint score-set hash drifted.")
        object.__setattr__(
            self, "policy_hash_by_target", MappingProxyType(policy_hashes)
        )
        object.__setattr__(self, "action_hash_by_key", MappingProxyType(action_hashes))

    @property
    def by_key(self) -> Mapping[tuple[str, str], TerminalEndpointScore]:
        return MappingProxyType(
            {(row.target_id, row.action_id): row for row in self.rows}
        )

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_consumed_test_terminal_endpoint_score_set_v1",
            "centers": list(CENTERS),
            "score_hashes": [row.score_hash for row in self.rows],
            "action_library_hash": self.action_library_hash,
            "policy_set_hash": self.policy_set_hash,
            "action_hash_by_key": dict(self.action_hash_by_key),
            "policy_hash_by_target": dict(self.policy_hash_by_target),
            "global_target_prediction_seal_hash": self.global_target_prediction_seal_hash,
            "global_prelabel_seal_hash": self.global_prelabel_seal_hash,
            "score_count": len(self.rows),
            "same_outer_H_labels_opened_only_after_global_seal": True,
            "terminal_scores_may_update_plan": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "score_set_hash": self.score_set_hash}


@dataclass(frozen=True)
class CenterBaccContrast:
    target_id: str
    contrast_id: str
    left_action_id: str
    right_action_id: str
    left_bacc: float
    right_bacc: float
    paired_bacc_delta: float
    score_set_hash: str
    contrast_hash: str

    def __post_init__(self) -> None:
        expected = next(
            (
                (left, right)
                for identifier, left, right in PRIMARY_CONTRASTS
                if identifier == self.contrast_id
            ),
            None,
        )
        payload = self.to_payload()
        payload.pop("contrast_hash")
        if (
            self.target_id not in CENTERS
            or expected != (self.left_action_id, self.right_action_id)
            or not all(
                math.isfinite(value)
                for value in (
                    self.left_bacc,
                    self.right_bacc,
                    self.paired_bacc_delta,
                )
            )
            or not math.isclose(
                self.paired_bacc_delta,
                self.left_bacc - self.right_bacc,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            or not _text(self.score_set_hash)
            or self.contrast_hash != canonical_sha256(payload)
        ):
            raise ProtocolError("Center BACC contrast boundary drifted.")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_consumed_test_center_bacc_contrast_v1",
            "target_center": self.target_id,
            "contrast_id": self.contrast_id,
            "left_action_id": self.left_action_id,
            "right_action_id": self.right_action_id,
            "left_bacc": self.left_bacc,
            "right_bacc": self.right_bacc,
            "paired_bacc_delta": self.paired_bacc_delta,
            "score_set_hash": self.score_set_hash,
            "inference_unit": "target_center",
            "terminal_scores_may_update_plan": False,
            "consumed_test_diagnostic_only": True,
            "contrast_hash": self.contrast_hash,
        }


@dataclass(frozen=True)
class AggregateCenterContrast:
    contrast_id: str
    left_action_id: str
    right_action_id: str
    center_count: int
    degrees_of_freedom: int
    equal_center_mean_delta: float
    sample_standard_deviation: float
    standard_error: float
    two_sided_ci95_lower: float
    two_sided_ci95_upper: float
    one_sided_lcb95: float
    two_sided_p_value: float
    center_delta_hash: str
    score_set_hash: str
    summary_hash: str

    def __post_init__(self) -> None:
        expected = canonical_sha256(self._unhashed_payload())
        expected_actions = next(
            (
                (left, right)
                for identifier, left, right in PRIMARY_CONTRASTS
                if identifier == self.contrast_id
            ),
            None,
        )
        finite_values = (
            self.equal_center_mean_delta,
            self.sample_standard_deviation,
            self.standard_error,
            self.two_sided_ci95_lower,
            self.two_sided_ci95_upper,
            self.one_sided_lcb95,
            self.two_sided_p_value,
        )
        if (
            self.center_count != len(CENTERS)
            or self.degrees_of_freedom != len(CENTERS) - 1
            or expected_actions != (self.left_action_id, self.right_action_id)
            or not all(math.isfinite(value) for value in finite_values)
            or self.sample_standard_deviation < 0.0
            or self.standard_error < 0.0
            or not 0.0 <= self.two_sided_p_value <= 1.0
            or not _text(self.center_delta_hash)
            or not _text(self.score_set_hash)
            or self.summary_hash != expected
        ):
            raise ProtocolError("Aggregate center contrast boundary drifted.")

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_consumed_test_aggregate_center_contrast_v1",
            "contrast_id": self.contrast_id,
            "left_action_id": self.left_action_id,
            "right_action_id": self.right_action_id,
            "center_count": self.center_count,
            "degrees_of_freedom": self.degrees_of_freedom,
            "equal_center_mean_delta": self.equal_center_mean_delta,
            "sample_standard_deviation": self.sample_standard_deviation,
            "standard_error": self.standard_error,
            "two_sided_ci95_lower": self.two_sided_ci95_lower,
            "two_sided_ci95_upper": self.two_sided_ci95_upper,
            "one_sided_lcb95": self.one_sided_lcb95,
            "two_sided_p_value": self.two_sided_p_value,
            "center_delta_hash": self.center_delta_hash,
            "score_set_hash": self.score_set_hash,
            "inference_unit": "target_center",
            "technical_seed_cells_are_independent_units": False,
            "terminal_scores_may_update_plan": False,
            "consumed_test_diagnostic_only": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "summary_hash": self.summary_hash}


@dataclass(frozen=True)
class OracleRankDiagnostic:
    target_id: str
    routed_candidate_source: str
    routed_executed_source: str | None
    routed_executed_action_id: str
    oracle_source_ids: tuple[str, ...]
    routed_candidate_oracle_rank: int
    routed_candidate_normalized_rank: float
    routed_top1_exact_agreement: bool
    routed_top1_tie_agreement: bool
    predicted_gain_hxe_bacc_spearman: float | None
    base_bacc: float
    routed_endpoint_bacc: float
    routed_candidate_hxe_bacc: float
    oracle_hxe_bacc: float
    normalized_oracle_gap: float
    policy_hash: str
    score_set_hash: str
    diagnostic_hash: str

    def __post_init__(self) -> None:
        sources = candidate_sources(self.target_id)
        payload = self.to_payload()
        payload.pop("diagnostic_hash")
        if (
            self.target_id not in CENTERS
            or self.routed_candidate_source not in sources
            or self.routed_executed_source is not None
            and self.routed_executed_source not in sources
            or not self.oracle_source_ids
            or any(source not in sources for source in self.oracle_source_ids)
            or self.routed_candidate_oracle_rank not in range(1, len(sources) + 1)
            or not 0.0 <= self.routed_candidate_normalized_rank <= 1.0
            or not 0.0 <= self.normalized_oracle_gap <= 1.0
            or self.predicted_gain_hxe_bacc_spearman is not None
            and not math.isfinite(self.predicted_gain_hxe_bacc_spearman)
            or not all(
                math.isfinite(value)
                for value in (
                    self.base_bacc,
                    self.routed_endpoint_bacc,
                    self.routed_candidate_hxe_bacc,
                    self.oracle_hxe_bacc,
                )
            )
            or not _text(self.policy_hash)
            or not _text(self.score_set_hash)
            or self.diagnostic_hash != canonical_sha256(payload)
        ):
            raise ProtocolError("Oracle rank diagnostic boundary drifted.")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_consumed_test_oracle_rank_diagnostic_v1",
            "target_center": self.target_id,
            "routed_candidate_source": self.routed_candidate_source,
            "routed_executed_source": self.routed_executed_source,
            "routed_executed_action_id": self.routed_executed_action_id,
            "oracle_source_ids": list(self.oracle_source_ids),
            "routed_candidate_oracle_rank": self.routed_candidate_oracle_rank,
            "routed_candidate_normalized_rank": self.routed_candidate_normalized_rank,
            "routed_top1_exact_agreement": self.routed_top1_exact_agreement,
            "routed_top1_tie_agreement": self.routed_top1_tie_agreement,
            "predicted_gain_hxe_bacc_spearman": self.predicted_gain_hxe_bacc_spearman,
            "base_bacc": self.base_bacc,
            "routed_endpoint_bacc": self.routed_endpoint_bacc,
            "routed_candidate_hxe_bacc": self.routed_candidate_hxe_bacc,
            "oracle_hxe_bacc": self.oracle_hxe_bacc,
            "normalized_oracle_gap": self.normalized_oracle_gap,
            "policy_hash": self.policy_hash,
            "score_set_hash": self.score_set_hash,
            "Hxe_may_feed_plan": False,
            "terminal_scores_may_update_plan": False,
            "consumed_test_diagnostic_only": True,
            "diagnostic_hash": self.diagnostic_hash,
        }


@dataclass(frozen=True)
class TerminalInferenceProducts:
    center_contrasts: tuple[CenterBaccContrast, ...]
    aggregate_contrasts: tuple[AggregateCenterContrast, ...]
    oracle_rank_diagnostics: tuple[OracleRankDiagnostic, ...]
    score_set_hash: str
    inference_hash: str

    def __post_init__(self) -> None:
        if (
            len(self.center_contrasts) != EXPECTED_CONTRAST_ROW_COUNT
            or len(self.aggregate_contrasts) != len(PRIMARY_CONTRASTS)
            or len(self.oracle_rank_diagnostics) != len(CENTERS)
            or any(row.score_set_hash != self.score_set_hash for row in self.center_contrasts)
            or any(
                row.score_set_hash != self.score_set_hash
                for row in self.aggregate_contrasts
            )
            or any(
                row.score_set_hash != self.score_set_hash
                for row in self.oracle_rank_diagnostics
            )
        ):
            raise ProtocolError("Terminal inference product coverage drifted.")
        if self.inference_hash != canonical_sha256(self._unhashed_payload()):
            raise ProtocolError("Terminal inference product hash drifted.")

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_consumed_test_terminal_inference_v1",
            "score_set_hash": self.score_set_hash,
            "center_contrast_hashes": [row.contrast_hash for row in self.center_contrasts],
            "aggregate_contrast_hashes": [
                row.summary_hash for row in self.aggregate_contrasts
            ],
            "oracle_rank_diagnostic_hashes": [
                row.diagnostic_hash for row in self.oracle_rank_diagnostics
            ],
            "inference_unit": "target_center",
            "Hxe_may_feed_plan": False,
            "terminal_scores_may_update_plan": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "inference_hash": self.inference_hash}


def score_terminal_target_action(
    *,
    action: FrozenEndpointAction,
    vectors: Sequence[SeedProbabilityVector],
    terminal_evaluation_labels: Sequence[int] | np.ndarray,
    support_partition_lock_hash: str,
    evaluation_partition_hash: str,
    global_target_prediction_seal_hash: str,
    global_prelabel_seal_hash: str,
    evaluation_case_count: int,
    global_target_prediction_seal_verified: bool,
) -> TerminalEndpointScore:
    """Score one already-frozen target action; labels cannot flow backward."""

    if (
        not isinstance(action, FrozenEndpointAction)
        or action.query_id != action.outer_target_id
        or global_target_prediction_seal_verified is not True
    ):
        raise ProtocolError("Terminal labels require a frozen target action and global seal.")
    raw = np.asarray(terminal_evaluation_labels)
    if raw.ndim != 1:
        raise ProtocolError("Terminal evaluation labels must be one-dimensional.")
    try:
        numeric = raw.astype(np.int64)
        floating = raw.astype(np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError("Terminal evaluation labels must be binary.") from exc
    if not np.array_equal(floating, numeric.astype(np.float64)):
        raise ProtocolError("Terminal evaluation labels must be binary integers.")
    counts = (int(np.sum(numeric == 0)), int(np.sum(numeric == 1)))
    if any(value <= 0 for value in counts) or sum(counts) != len(numeric):
        raise ProtocolError("Terminal BACC requires both binary classes only.")
    endpoint = score_sealed_probability_ensemble(vectors, numeric)
    return TerminalEndpointScore(
        target_id=action.outer_target_id,
        action_id=action.action_id,
        action_hash=action.action_hash,
        policy_hash=str(action.policy_hash),
        support_partition_lock_hash=support_partition_lock_hash,
        evaluation_partition_hash=evaluation_partition_hash,
        global_target_prediction_seal_hash=global_target_prediction_seal_hash,
        global_prelabel_seal_hash=global_prelabel_seal_hash,
        evaluation_case_count=evaluation_case_count,
        observed_class_row_counts=counts,
        endpoint=endpoint,
    )


def validate_terminal_endpoint_scores(
    rows: Sequence[TerminalEndpointScore],
    actions: FrozenTargetActionLibrary,
) -> TerminalEndpointScoreSet:
    if not isinstance(actions, FrozenTargetActionLibrary):
        raise ProtocolError("Terminal scores require the frozen action library.")
    values = tuple(rows)
    if any(not isinstance(row, TerminalEndpointScore) for row in values):
        raise ProtocolError("Terminal scores must use their typed contract.")
    ordered = tuple(
        sorted(
            values,
            key=lambda row: (
                CENTERS.index(row.target_id),
                expected_target_action_ids(row.target_id).index(row.action_id),
            ),
        )
    )
    if len(ordered) != EXPECTED_TERMINAL_SCORE_COUNT:
        raise ProtocolError("Terminal endpoint score count drifted.")
    for row in ordered:
        expected_action = actions.by_target[row.target_id].by_action_id[row.action_id]
        if row.action_hash != expected_action.action_hash or row.policy_hash != expected_action.policy_hash:
            raise ProtocolError("Terminal score/action binding drifted.")
    prediction_seals = {row.global_target_prediction_seal_hash for row in ordered}
    prelabel_seals = {row.global_prelabel_seal_hash for row in ordered}
    if len(prediction_seals) != 1 or len(prelabel_seals) != 1:
        raise ProtocolError("Terminal scores require one global pair of prelabel seals.")
    prediction_seal = next(iter(prediction_seals))
    prelabel_seal = next(iter(prelabel_seals))
    payload = {
        "schema_version": "midogpp_consumed_test_terminal_endpoint_score_set_v1",
        "centers": list(CENTERS),
        "score_hashes": [row.score_hash for row in ordered],
        "action_library_hash": actions.action_library_hash,
        "policy_set_hash": actions.policy_set_hash,
        "action_hash_by_key": {
            _action_key(target, action_id):
            actions.by_target[target].by_action_id[action_id].action_hash
            for target in CENTERS
            for action_id in expected_target_action_ids(target)
        },
        "policy_hash_by_target": {
            target: actions.by_target[target].policy_hash for target in CENTERS
        },
        "global_target_prediction_seal_hash": prediction_seal,
        "global_prelabel_seal_hash": prelabel_seal,
        "score_count": len(ordered),
        "same_outer_H_labels_opened_only_after_global_seal": True,
        "terminal_scores_may_update_plan": False,
    }
    return TerminalEndpointScoreSet(
        rows=ordered,
        action_library_hash=actions.action_library_hash,
        policy_set_hash=actions.policy_set_hash,
        action_hash_by_key={
            _action_key(target, action_id):
            actions.by_target[target].by_action_id[action_id].action_hash
            for target in CENTERS
            for action_id in expected_target_action_ids(target)
        },
        policy_hash_by_target={
            target: actions.by_target[target].policy_hash for target in CENTERS
        },
        global_target_prediction_seal_hash=prediction_seal,
        global_prelabel_seal_hash=prelabel_seal,
        score_set_hash=canonical_sha256(payload),
    )


def build_terminal_inference_products(
    scores: TerminalEndpointScoreSet,
    policies: FrozenTargetPolicySet,
) -> TerminalInferenceProducts:
    """Compute terminal contrasts/ranks without returning anything policy-shaped."""

    if (
        not isinstance(scores, TerminalEndpointScoreSet)
        or not isinstance(policies, FrozenTargetPolicySet)
        or scores.policy_set_hash != policies.policy_set_hash
        or any(
            scores.policy_hash_by_target[target]
            != policies.by_target[target].policy_hash
            for target in CENTERS
        )
    ):
        raise ProtocolError("Terminal inference requires aligned score/policy sets.")
    metric = scores.by_key
    contrasts: list[CenterBaccContrast] = []
    aggregate_rows: list[AggregateCenterContrast] = []
    oracle_rows: list[OracleRankDiagnostic] = []
    for target in CENTERS:
        for contrast_id, left, right in PRIMARY_CONTRASTS:
            left_value = metric[(target, left)].balanced_accuracy
            right_value = metric[(target, right)].balanced_accuracy
            delta = left_value - right_value
            payload = {
                "schema_version": "midogpp_consumed_test_center_bacc_contrast_v1",
                "target_center": target,
                "contrast_id": contrast_id,
                "left_action_id": left,
                "right_action_id": right,
                "left_bacc": left_value,
                "right_bacc": right_value,
                "paired_bacc_delta": delta,
                "score_set_hash": scores.score_set_hash,
                "inference_unit": "target_center",
                "terminal_scores_may_update_plan": False,
                "consumed_test_diagnostic_only": True,
            }
            contrasts.append(
                CenterBaccContrast(
                    target_id=target,
                    contrast_id=contrast_id,
                    left_action_id=left,
                    right_action_id=right,
                    left_bacc=left_value,
                    right_bacc=right_value,
                    paired_bacc_delta=delta,
                    score_set_hash=scores.score_set_hash,
                    contrast_hash=canonical_sha256(payload),
                )
            )
        policy = policies.by_target[target]
        sources = candidate_sources(target)
        utility = {
            source: metric[(target, h_x_e_action_id(source))].balanced_accuracy
            for source in sources
        }
        maximum = max(utility.values())
        minimum = min(utility.values())
        oracle_sources = tuple(
            source
            for source in sources
            if math.isclose(utility[source], maximum, rel_tol=0.0, abs_tol=1.0e-15)
        )
        proposal = policy.routed_candidate_source
        rank = 1 + sum(
            value > utility[proposal] + 1.0e-15 for value in utility.values()
        )
        denominator = maximum - minimum
        normalized_gap = (
            0.0
            if denominator <= 0.0
            else (maximum - utility[proposal]) / denominator
        )
        predictions = policy.core_policy.role_prediction_by_source[ROUTED_ACTION_ID]
        rho = spearman(
            [predictions[source] for source in sources],
            [utility[source] for source in sources],
        )
        rho_value = None if not math.isfinite(rho) else float(rho)
        oracle_payload = {
            "schema_version": "midogpp_consumed_test_oracle_rank_diagnostic_v1",
            "target_center": target,
            "routed_candidate_source": proposal,
            "routed_executed_source": policy.executed_routed_source,
            "routed_executed_action_id": policy.selected_action_id,
            "oracle_source_ids": list(oracle_sources),
            "routed_candidate_oracle_rank": rank,
            "routed_candidate_normalized_rank": (rank - 1) / 7.0,
            "routed_top1_exact_agreement": proposal == min(oracle_sources),
            "routed_top1_tie_agreement": proposal in oracle_sources,
            "predicted_gain_hxe_bacc_spearman": rho_value,
            "base_bacc": metric[(target, BASE_ACTION_ID)].balanced_accuracy,
            "routed_endpoint_bacc": metric[(target, ROUTED_ACTION_ID)].balanced_accuracy,
            "routed_candidate_hxe_bacc": utility[proposal],
            "oracle_hxe_bacc": maximum,
            "normalized_oracle_gap": normalized_gap,
            "policy_hash": policy.policy_hash,
            "score_set_hash": scores.score_set_hash,
            "Hxe_may_feed_plan": False,
            "terminal_scores_may_update_plan": False,
            "consumed_test_diagnostic_only": True,
        }
        oracle_rows.append(
            OracleRankDiagnostic(
                target_id=target,
                routed_candidate_source=proposal,
                routed_executed_source=policy.executed_routed_source,
                routed_executed_action_id=policy.selected_action_id,
                oracle_source_ids=oracle_sources,
                routed_candidate_oracle_rank=rank,
                routed_candidate_normalized_rank=(rank - 1) / 7.0,
                routed_top1_exact_agreement=proposal == min(oracle_sources),
                routed_top1_tie_agreement=proposal in oracle_sources,
                predicted_gain_hxe_bacc_spearman=rho_value,
                base_bacc=metric[(target, BASE_ACTION_ID)].balanced_accuracy,
                routed_endpoint_bacc=metric[(target, ROUTED_ACTION_ID)].balanced_accuracy,
                routed_candidate_hxe_bacc=utility[proposal],
                oracle_hxe_bacc=maximum,
                normalized_oracle_gap=normalized_gap,
                policy_hash=policy.policy_hash,
                score_set_hash=scores.score_set_hash,
                diagnostic_hash=canonical_sha256(oracle_payload),
            )
        )
    aggregate_rows.extend(
        summarize_center_contrasts(contrasts, score_set_hash=scores.score_set_hash)
    )
    inference_payload = {
        "schema_version": "midogpp_consumed_test_terminal_inference_v1",
        "score_set_hash": scores.score_set_hash,
        "center_contrast_hashes": [row.contrast_hash for row in contrasts],
        "aggregate_contrast_hashes": [row.summary_hash for row in aggregate_rows],
        "oracle_rank_diagnostic_hashes": [row.diagnostic_hash for row in oracle_rows],
        "inference_unit": "target_center",
        "Hxe_may_feed_plan": False,
        "terminal_scores_may_update_plan": False,
    }
    return TerminalInferenceProducts(
        center_contrasts=tuple(contrasts),
        aggregate_contrasts=tuple(aggregate_rows),
        oracle_rank_diagnostics=tuple(oracle_rows),
        score_set_hash=scores.score_set_hash,
        inference_hash=canonical_sha256(inference_payload),
    )


def summarize_center_contrasts(
    center_contrasts: Sequence[CenterBaccContrast],
    *,
    score_set_hash: str,
) -> tuple[AggregateCenterContrast, ...]:
    """Summarize the nine paired center units without treating seeds as units."""

    rows = tuple(center_contrasts)
    expected_keys = tuple(
        (target, contrast_id)
        for target in CENTERS
        for contrast_id, _left, _right in PRIMARY_CONTRASTS
    )
    keys = tuple((row.target_id, row.contrast_id) for row in rows)
    if (
        len(rows) != EXPECTED_CONTRAST_ROW_COUNT
        or keys != expected_keys
        or any(row.score_set_hash != score_set_hash for row in rows)
        or not _text(score_set_hash)
    ):
        raise ProtocolError("Aggregate center contrast input coverage drifted.")
    aggregate_rows: list[AggregateCenterContrast] = []
    for contrast_id, left, right in PRIMARY_CONTRASTS:
        selected = tuple(
            row for row in rows if row.contrast_id == contrast_id
        )
        if tuple(row.target_id for row in selected) != CENTERS:
            raise ProtocolError("Aggregate center contrast coverage drifted.")
        delta = np.asarray(
            [row.paired_bacc_delta for row in selected], dtype=np.float64
        )
        count = len(delta)
        degrees = count - 1
        mean = float(np.mean(delta, dtype=np.float64))
        standard_deviation = float(np.std(delta, ddof=1, dtype=np.float64))
        standard_error = standard_deviation / math.sqrt(float(count))
        two_sided_critical = float(student_t.ppf(0.975, degrees))
        one_sided_critical = float(student_t.ppf(0.95, degrees))
        two_sided_margin = two_sided_critical * standard_error
        if standard_error == 0.0:
            p_value = 0.0 if mean != 0.0 else 1.0
        else:
            p_value = float(
                2.0 * student_t.sf(abs(mean / standard_error), degrees)
            )
        delta_hash = canonical_sha256(delta.tolist())
        payload = {
            "schema_version": "midogpp_consumed_test_aggregate_center_contrast_v1",
            "contrast_id": contrast_id,
            "left_action_id": left,
            "right_action_id": right,
            "center_count": count,
            "degrees_of_freedom": degrees,
            "equal_center_mean_delta": mean,
            "sample_standard_deviation": standard_deviation,
            "standard_error": standard_error,
            "two_sided_ci95_lower": mean - two_sided_margin,
            "two_sided_ci95_upper": mean + two_sided_margin,
            "one_sided_lcb95": mean - one_sided_critical * standard_error,
            "two_sided_p_value": p_value,
            "center_delta_hash": delta_hash,
            "score_set_hash": score_set_hash,
            "inference_unit": "target_center",
            "technical_seed_cells_are_independent_units": False,
            "terminal_scores_may_update_plan": False,
            "consumed_test_diagnostic_only": True,
        }
        aggregate_rows.append(
            AggregateCenterContrast(
                contrast_id=contrast_id,
                left_action_id=left,
                right_action_id=right,
                center_count=count,
                degrees_of_freedom=degrees,
                equal_center_mean_delta=mean,
                sample_standard_deviation=standard_deviation,
                standard_error=standard_error,
                two_sided_ci95_lower=mean - two_sided_margin,
                two_sided_ci95_upper=mean + two_sided_margin,
                one_sided_lcb95=mean - one_sided_critical * standard_error,
                two_sided_p_value=p_value,
                center_delta_hash=delta_hash,
                score_set_hash=score_set_hash,
                summary_hash=canonical_sha256(payload),
            )
        )
    return tuple(aggregate_rows)


build_center_contrasts_and_oracle_ranks = build_terminal_inference_products


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value.strip() == value


def _action_key(target_id: str, action_id: str) -> str:
    return f"{target_id}::{action_id}"


__all__ = (
    "CenterBaccContrast",
    "AggregateCenterContrast",
    "OracleRankDiagnostic",
    "TerminalEndpointScore",
    "TerminalEndpointScoreSet",
    "TerminalInferenceProducts",
    "build_center_contrasts_and_oracle_ranks",
    "build_terminal_inference_products",
    "score_terminal_target_action",
    "summarize_center_contrasts",
    "validate_terminal_endpoint_scores",
)
