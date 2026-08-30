"""Post-seal action-matrix diagnostics that can never feed HARP routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import re
from typing import Mapping

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...runtime.harp_probability_menu import (
    BASE_ACTION_ID,
    LAMBDA_GRID,
    TARGET_SURFACE,
    UNIFORM_ACTION_ID,
)
from ...runtime.harp_probability_menu.hashing import canonical_sha256
from .metric_primitives import (
    binary_log_loss,
    case_equal_balanced_accuracy,
    case_equal_mean,
)
from .sealing import HarpFreshPrelabelSeal


_TIE_TOLERANCE = 1.0e-12
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, kw_only=True)
class HarpFreshActionMatrixMetric:
    center: str
    action_id: str
    physical_action_id: str
    selected_source_id: str | None
    lambda_value: float
    physical_generated_action: bool
    matched_budget_action: bool
    row_count: int
    case_count: int
    balanced_accuracy: float
    brier: float
    log_loss: float
    balanced_accuracy_delta_vs_b: float
    balanced_accuracy_delta_vs_u: float
    brier_delta_vs_b: float
    brier_delta_vs_u: float
    log_loss_delta_vs_b: float
    log_loss_delta_vs_u: float
    diagnostic_only: bool = True

    def __post_init__(self) -> None:
        if (
            self.center not in CENTERS
            or not self.action_id
            or not self.physical_action_id
            or type(self.row_count) is not int
            or self.row_count <= 0
            or type(self.case_count) is not int
            or self.case_count <= 0
            or self.case_count > self.row_count
            or self.lambda_value not in (0.0, *LAMBDA_GRID)
            or self.diagnostic_only is not True
        ):
            raise ProtocolError("Fresh HARP action-matrix identity drifted.")
        numeric = (
            self.balanced_accuracy,
            self.brier,
            self.log_loss,
            self.balanced_accuracy_delta_vs_b,
            self.balanced_accuracy_delta_vs_u,
            self.brier_delta_vs_b,
            self.brier_delta_vs_u,
            self.log_loss_delta_vs_b,
            self.log_loss_delta_vs_u,
        )
        if any(not math.isfinite(float(value)) for value in numeric):
            raise ProtocolError("Fresh HARP action-matrix metric is nonfinite.")
        if self.action_id == BASE_ACTION_ID:
            valid_identity = (
                self.physical_action_id == BASE_ACTION_ID
                and self.selected_source_id is None
                and self.lambda_value == 0.0
                and self.physical_generated_action is True
                and self.matched_budget_action is False
            )
        elif self.action_id == UNIFORM_ACTION_ID:
            valid_identity = (
                self.physical_action_id == UNIFORM_ACTION_ID
                and self.selected_source_id is None
                and self.lambda_value == 0.0
                and self.physical_generated_action is True
                and self.matched_budget_action is True
            )
        else:
            source = self.selected_source_id
            valid_identity = (
                type(source) is str
                and source in CENTERS
                and source != self.center
                and self.lambda_value in LAMBDA_GRID
                and self.action_id == _action_id(source, self.lambda_value)
                and self.physical_action_id == f"Hxe::{source}"
                and self.physical_generated_action
                is (self.lambda_value == 1.0)
                and self.matched_budget_action is True
            )
        if not valid_identity:
            raise ProtocolError(
                "Fresh HARP action-matrix action semantics drifted."
            )


@dataclass(frozen=True, kw_only=True)
class HarpFreshCenterOracleDiagnostic:
    center: str
    action_count: int
    physical_matched_action_count: int
    budget_bacc_effect_u_minus_b: float
    budget_brier_effect_u_minus_b: float
    budget_log_loss_effect_u_minus_b: float
    best_fixed_action_ids: tuple[str, ...]
    best_fixed_action_bacc: float
    best_physical_action_ids: tuple[str, ...]
    best_physical_action_bacc: float
    best_physical_bacc_effect_vs_u: float
    frozen_lambda_one_policy_balanced_accuracy_delta_vs_u: float
    frozen_lambda_one_policy_brier_delta_vs_u: float
    frozen_lambda_one_policy_log_loss_delta_vs_u: float
    frozen_lambda_one_policy_route_rate: float
    frozen_lambda_one_policy_reference_preserving: bool
    selected_predictive_bacc_effect_vs_u: float
    final_operational_bacc_effect_vs_b: float
    selected_top1_oracle_tie_credit: float
    selected_mean_true_probability_rank: float
    selected_mean_log_loss_regret: float
    selected_mean_brier_regret: float
    best_fixed_action_bacc_minus_policy_bacc: float
    case_equal: bool = True
    labels_used_after_route_seal_only: bool = True
    diagnostic_may_feed_policy: bool = False

    def __post_init__(self) -> None:
        if (
            self.center not in CENTERS
            or self.action_count != 2 + (len(CENTERS) - 1) * len(LAMBDA_GRID)
            or self.physical_matched_action_count != len(CENTERS)
            or not self.best_fixed_action_ids
            or not self.best_physical_action_ids
            or self.case_equal is not True
            or self.labels_used_after_route_seal_only is not True
            or self.diagnostic_may_feed_policy is not False
            or self.frozen_lambda_one_policy_reference_preserving is not True
        ):
            raise ProtocolError("Fresh HARP center-oracle coverage drifted.")
        numeric = tuple(
            float(value)
            for name, value in self.__dict__.items()
            if name
            not in {
                "center",
                "action_count",
                "physical_matched_action_count",
                "best_fixed_action_ids",
                "best_physical_action_ids",
                "case_equal",
                "labels_used_after_route_seal_only",
                "diagnostic_may_feed_policy",
                "frozen_lambda_one_policy_reference_preserving",
            }
        )
        if (
            any(not math.isfinite(value) for value in numeric)
            or not 0.0 <= self.frozen_lambda_one_policy_route_rate <= 1.0
            or not 0.0 <= self.selected_top1_oracle_tie_credit <= 1.0
            or self.selected_mean_true_probability_rank < 1.0
            or self.selected_mean_log_loss_regret < -_TIE_TOLERANCE
            or self.selected_mean_brier_regret < -_TIE_TOLERANCE
        ):
            raise ProtocolError("Fresh HARP center-oracle metric drifted.")


@dataclass(frozen=True, kw_only=True)
class HarpFreshOracleDiagnosticResult:
    prelabel_seal_hash: str
    action_matrix: tuple[HarpFreshActionMatrixMetric, ...]
    center_diagnostics: tuple[HarpFreshCenterOracleDiagnostic, ...]
    diagnostic_only: bool = True
    labels_used_after_route_seal_only: bool = True
    labels_available_to_policy: bool = False
    policy_or_threshold_update_emitted: bool = False
    result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        rows = tuple(self.action_matrix)
        centers = tuple(self.center_diagnostics)
        per_center = 2 + (len(CENTERS) - 1) * len(LAMBDA_GRID)
        expected_action_order = tuple(
            (center, action_id)
            for center in CENTERS
            for action_id in (
                BASE_ACTION_ID,
                UNIFORM_ACTION_ID,
                *(
                    _action_id(source, lam)
                    for source in CENTERS
                    if source != center
                    for lam in LAMBDA_GRID
                ),
            )
        )
        if (
            type(self.prelabel_seal_hash) is not str
            or _SHA256.fullmatch(self.prelabel_seal_hash) is None
            or len(rows) != len(CENTERS) * per_center
            or tuple(row.center for row in rows[::per_center]) != CENTERS
            or tuple((row.center, row.action_id) for row in rows)
            != expected_action_order
            or tuple(row.center for row in centers) != CENTERS
            or self.diagnostic_only is not True
            or self.labels_used_after_route_seal_only is not True
            or self.labels_available_to_policy is not False
            or self.policy_or_threshold_update_emitted is not False
        ):
            raise ProtocolError("Fresh HARP oracle result coverage drifted.")
        payload = {
            "schema_version": "midogpp_harp_fresh_oracle_diagnostics_v2",
            "prelabel_seal_hash": self.prelabel_seal_hash,
            "action_matrix": [asdict(row) for row in rows],
            "center_diagnostics": [asdict(row) for row in centers],
            "diagnostic_only": True,
            "labels_used_after_route_seal_only": True,
            "labels_available_to_policy": False,
            "policy_or_threshold_update_emitted": False,
        }
        object.__setattr__(self, "action_matrix", rows)
        object.__setattr__(self, "center_diagnostics", centers)
        object.__setattr__(self, "result_hash", canonical_sha256(payload))


def _action_id(source: str, lam: float) -> str:
    return f"Hxe::{source}|lambda={lam:.2f}"


def _metric(
    *,
    center: str,
    action_id: str,
    physical_action_id: str,
    source: str | None,
    lam: float,
    physical: bool,
    matched: bool,
    truth: np.ndarray,
    probability: np.ndarray,
    cases: np.ndarray,
    baseline: tuple[float, float, float],
    reference: tuple[float, float, float],
) -> HarpFreshActionMatrixMetric:
    bacc = case_equal_balanced_accuracy(truth, probability, cases)
    brier = case_equal_mean((probability - truth) ** 2, cases)
    loss = case_equal_mean(binary_log_loss(truth, probability), cases)
    return HarpFreshActionMatrixMetric(
        center=center,
        action_id=action_id,
        physical_action_id=physical_action_id,
        selected_source_id=source,
        lambda_value=lam,
        physical_generated_action=physical,
        matched_budget_action=matched,
        row_count=len(truth),
        case_count=len(set(cases.tolist())),
        balanced_accuracy=bacc,
        brier=brier,
        log_loss=loss,
        balanced_accuracy_delta_vs_b=bacc - baseline[0],
        balanced_accuracy_delta_vs_u=bacc - reference[0],
        brier_delta_vs_b=brier - baseline[1],
        brier_delta_vs_u=brier - reference[1],
        log_loss_delta_vs_b=loss - baseline[2],
        log_loss_delta_vs_u=loss - reference[2],
    )


def build_harp_fresh_oracle_diagnostics(
    seal: HarpFreshPrelabelSeal,
    labels_by_row_key: Mapping[tuple[str, str, str], int],
) -> HarpFreshOracleDiagnosticResult:
    """Score all sealed actions after labels open, with no policy callback edge."""

    if not isinstance(seal, HarpFreshPrelabelSeal):
        raise ProtocolError("Fresh HARP oracle diagnostics require the prelabel seal.")
    seal.menu.assert_valid()
    matrix_rows: list[HarpFreshActionMatrixMetric] = []
    center_rows: list[HarpFreshCenterOracleDiagnostic] = []
    for center, vector, physical_policy_vector, physical_policy_or_u in zip(
        CENTERS,
        seal.routed_vectors,
        seal.physical_ablation_vectors,
        seal.physical_ablation_reference_preserving_vectors,
        strict=True,
    ):
        cases = np.asarray(vector.case_ids, dtype=object)
        try:
            truth = np.asarray(
                [
                    labels_by_row_key[(center, case_id, row_id)]
                    for row_id, case_id in zip(
                        vector.row_ids, vector.case_ids, strict=True
                    )
                ],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise ProtocolError("Fresh HARP oracle label coverage is incomplete.") from exc
        if set(int(value) for value in truth.tolist()) != {0, 1}:
            raise ProtocolError("Fresh HARP oracle center lacks both truth classes.")

        baseline_action = seal.menu.action_for(
            surface_kind=TARGET_SURFACE,
            outer_target_id=center,
            query_center_id=center,
            selected_source_id=None,
            action_id=BASE_ACTION_ID,
        )
        reference_action = seal.menu.action_for(
            surface_kind=TARGET_SURFACE,
            outer_target_id=center,
            query_center_id=center,
            selected_source_id=None,
            action_id=UNIFORM_ACTION_ID,
        )
        baseline_probability = seal.menu.exact_nine(baseline_action)
        reference_probability = seal.menu.exact_nine(reference_action)
        if (
            tuple(vector.row_ids) != seal.menu.identities_for(baseline_action)[0]
            or tuple(vector.row_ids) != seal.menu.identities_for(reference_action)[0]
        ):
            raise ProtocolError("Fresh HARP oracle actions escaped target row identity.")
        baseline_metrics = (
            case_equal_balanced_accuracy(truth, baseline_probability, cases),
            case_equal_mean((baseline_probability - truth) ** 2, cases),
            case_equal_mean(binary_log_loss(truth, baseline_probability), cases),
        )
        reference_metrics = (
            case_equal_balanced_accuracy(truth, reference_probability, cases),
            case_equal_mean((reference_probability - truth) ** 2, cases),
            case_equal_mean(binary_log_loss(truth, reference_probability), cases),
        )
        probabilities: list[np.ndarray] = [baseline_probability, reference_probability]
        block: list[HarpFreshActionMatrixMetric] = [
            _metric(
                center=center,
                action_id=BASE_ACTION_ID,
                physical_action_id=BASE_ACTION_ID,
                source=None,
                lam=0.0,
                physical=True,
                matched=False,
                truth=truth,
                probability=baseline_probability,
                cases=cases,
                baseline=baseline_metrics,
                reference=reference_metrics,
            ),
            _metric(
                center=center,
                action_id=UNIFORM_ACTION_ID,
                physical_action_id=UNIFORM_ACTION_ID,
                source=None,
                lam=0.0,
                physical=True,
                matched=True,
                truth=truth,
                probability=reference_probability,
                cases=cases,
                baseline=baseline_metrics,
                reference=reference_metrics,
            ),
        ]
        for source in CENTERS:
            if source == center:
                continue
            physical_action = seal.menu.action_for(
                surface_kind=TARGET_SURFACE,
                outer_target_id=center,
                query_center_id=center,
                selected_source_id=source,
            )
            expert_probability = seal.menu.exact_nine(physical_action)
            for lam in LAMBDA_GRID:
                probability = np.ascontiguousarray(
                    (1.0 - lam) * reference_probability + lam * expert_probability,
                    dtype=np.float64,
                )
                probabilities.append(probability)
                block.append(
                    _metric(
                        center=center,
                        action_id=_action_id(source, lam),
                        physical_action_id=physical_action.action_id,
                        source=source,
                        lam=lam,
                        physical=lam == 1.0,
                        matched=True,
                        truth=truth,
                        probability=probability,
                        cases=cases,
                        baseline=baseline_metrics,
                        reference=reference_metrics,
                    )
                )
        if len(block) != 2 + (len(CENTERS) - 1) * len(LAMBDA_GRID):
            raise ProtocolError("Fresh HARP oracle action matrix is incomplete.")

        stacked = np.stack(probabilities, axis=0)
        true_probability = np.where(truth[None, :] == 1, stacked, 1.0 - stacked)
        selected_true_probability = np.where(
            truth == 1,
            vector.routed_probabilities,
            1.0 - vector.routed_probabilities,
        )
        best_true_probability = np.max(true_probability, axis=0)
        tie_credit = np.asarray(
            selected_true_probability >= best_true_probability - _TIE_TOLERANCE,
            dtype=np.float64,
        )
        better = np.sum(
            true_probability > selected_true_probability[None, :] + _TIE_TOLERANCE,
            axis=0,
        )
        ties = np.sum(
            np.abs(true_probability - selected_true_probability[None, :])
            <= _TIE_TOLERANCE,
            axis=0,
        )
        selected_rank = 1.0 + better + 0.5 * np.maximum(0, ties - 1)
        action_losses = -np.log(np.clip(true_probability, 1.0e-7, 1.0))
        selected_losses = binary_log_loss(truth, vector.routed_probabilities)
        log_regret = selected_losses - np.min(action_losses, axis=0)
        selected_brier = (vector.routed_probabilities - truth) ** 2
        action_brier = (stacked - truth[None, :]) ** 2
        brier_regret = selected_brier - np.min(action_brier, axis=0)

        best_bacc = max(row.balanced_accuracy for row in block)
        best_ids = tuple(
            row.action_id
            for row in block
            if abs(row.balanced_accuracy - best_bacc) <= _TIE_TOLERANCE
        )
        physical_matched = tuple(
            row for row in block if row.matched_budget_action and row.physical_generated_action
        )
        physical_experts = tuple(
            row
            for row in physical_matched
            if row.selected_source_id is not None and row.lambda_value == 1.0
        )
        if len(physical_matched) != len(CENTERS) or len(physical_experts) != len(
            CENTERS
        ) - 1:
            raise ProtocolError(
                "Fresh HARP physical oracle action coverage drifted."
            )
        best_physical_bacc = max(
            row.balanced_accuracy for row in physical_experts
        )
        best_physical_ids = tuple(
            row.action_id
            for row in physical_experts
            if abs(row.balanced_accuracy - best_physical_bacc) <= _TIE_TOLERANCE
        )
        selected_or_u = np.where(
            np.asarray([decision.eligible for decision in vector.decisions], dtype=bool),
            vector.routed_probabilities,
            reference_probability,
        )
        selected_or_u_bacc = case_equal_balanced_accuracy(truth, selected_or_u, cases)
        final_bacc = case_equal_balanced_accuracy(
            truth, vector.routed_probabilities, cases
        )
        physical_policy_bacc = case_equal_balanced_accuracy(
            truth,
            physical_policy_or_u,
            cases,
        )
        physical_policy_brier = case_equal_mean(
            (physical_policy_or_u - truth) ** 2,
            cases,
        )
        physical_policy_log_loss = case_equal_mean(
            binary_log_loss(
                truth, physical_policy_or_u
            ),
            cases,
        )
        physical_policy_route_rate = case_equal_mean(
            np.asarray(
                [
                    decision.eligible
                    for decision in physical_policy_vector.decisions
                ],
                dtype=np.float64,
            ),
            cases,
        )
        center_rows.append(
            HarpFreshCenterOracleDiagnostic(
                center=center,
                action_count=len(block),
                physical_matched_action_count=len(physical_matched),
                budget_bacc_effect_u_minus_b=reference_metrics[0]
                - baseline_metrics[0],
                budget_brier_effect_u_minus_b=reference_metrics[1]
                - baseline_metrics[1],
                budget_log_loss_effect_u_minus_b=reference_metrics[2]
                - baseline_metrics[2],
                best_fixed_action_ids=best_ids,
                best_fixed_action_bacc=best_bacc,
                best_physical_action_ids=best_physical_ids,
                best_physical_action_bacc=best_physical_bacc,
                best_physical_bacc_effect_vs_u=best_physical_bacc
                - reference_metrics[0],
                frozen_lambda_one_policy_balanced_accuracy_delta_vs_u=(
                    physical_policy_bacc - reference_metrics[0]
                ),
                frozen_lambda_one_policy_brier_delta_vs_u=(
                    physical_policy_brier - reference_metrics[1]
                ),
                frozen_lambda_one_policy_log_loss_delta_vs_u=(
                    physical_policy_log_loss - reference_metrics[2]
                ),
                frozen_lambda_one_policy_route_rate=physical_policy_route_rate,
                frozen_lambda_one_policy_reference_preserving=True,
                selected_predictive_bacc_effect_vs_u=selected_or_u_bacc
                - reference_metrics[0],
                final_operational_bacc_effect_vs_b=final_bacc
                - baseline_metrics[0],
                selected_top1_oracle_tie_credit=case_equal_mean(tie_credit, cases),
                selected_mean_true_probability_rank=case_equal_mean(
                    selected_rank, cases
                ),
                selected_mean_log_loss_regret=max(
                    0.0, case_equal_mean(log_regret, cases)
                ),
                selected_mean_brier_regret=max(
                    0.0, case_equal_mean(brier_regret, cases)
                ),
                best_fixed_action_bacc_minus_policy_bacc=best_bacc - final_bacc,
            )
        )
        matrix_rows.extend(block)
    return HarpFreshOracleDiagnosticResult(
        prelabel_seal_hash=seal.seal_hash,
        action_matrix=tuple(matrix_rows),
        center_diagnostics=tuple(center_rows),
    )


__all__ = (
    "HarpFreshActionMatrixMetric",
    "HarpFreshCenterOracleDiagnostic",
    "HarpFreshOracleDiagnosticResult",
    "build_harp_fresh_oracle_diagnostics",
)
