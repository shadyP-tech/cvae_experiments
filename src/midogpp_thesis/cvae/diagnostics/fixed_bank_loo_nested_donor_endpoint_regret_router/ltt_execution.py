"""Outer-center cross-fitted descriptive feasibility computation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    CENTERS,
    HARD_THRESHOLD,
    LOG_LOSS_CLIP_EPSILON,
    LTT_METHOD_ID,
    PORTFOLIO_METHOD_ID,
)
from .contracts import (
    BinaryLabel,
    CandidateDescriptor,
    DonorRegretRow,
    EndpointCasePrediction,
    RouteDecision,
)
from .controls import RoutePolicySpec, select_route_for_policy
from .donor_regret_model import fit_models_for_training_centers
from .hashing import canonical_hash
from .learn_then_test import CenterBlockPolicyEvidence, learn_then_test_center_harm
from .route_worker_runtime import CenterEndpointProducts


@dataclass(frozen=True)
class TargetLTTAuthorization:
    target_center: str
    selected_policy_id: str | None
    decisions: tuple[RouteDecision, ...]
    report: Mapping[str, object]
    authorization_hash: str = field(init=False)

    def __post_init__(self) -> None:
        report = dict(self.report)
        if (
            self.target_center not in CENTERS
            or any(row.target_center != self.target_center for row in self.decisions)
            or any(row.policy_id != LTT_METHOD_ID for row in self.decisions)
            or report.get("outer_target_center") != self.target_center
        ):
            raise ProtocolError("Target center-block feasibility topology drifted.")
        object.__setattr__(self, "report", MappingProxyType(report))
        object.__setattr__(
            self,
            "authorization_hash",
            canonical_hash(
                {
                    "target_center": self.target_center,
                    "selected_policy_id": self.selected_policy_id,
                    "decision_hashes": [row.decision_hash for row in self.decisions],
                    "report": report,
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            **dict(self.report),
            "selected_policy_id": self.selected_policy_id,
            "authorization_hash": self.authorization_hash,
        }


def build_ltt_authorizations(
    *,
    descriptors_by_center: Mapping[str, Sequence[CandidateDescriptor]],
    donor_endpoint_products: Mapping[
        tuple[str, str], CenterEndpointProducts
    ],
    donor_rows_by_outer_target: Mapping[str, Sequence[DonorRegretRow]],
    donor_labels_by_outer_target: Mapping[
        tuple[str, str], Sequence[BinaryLabel]
    ],
    target_decisions_by_policy: Mapping[
        str, Mapping[tuple[str, str], RouteDecision]
    ],
    ltt_policies: Sequence[RoutePolicySpec],
) -> tuple[TargetLTTAuthorization, ...]:
    """Describe eight donor-center blocks without reading target-H labels."""

    policies = tuple(ltt_policies)
    if not policies or len({row.policy_id for row in policies}) != len(policies):
        raise ProtocolError("Center-block feasibility requires one frozen policy menu.")
    outputs: list[TargetLTTAuthorization] = []
    for outer in CENTERS:
        donors = tuple(center for center in CENTERS if center != outer)
        rows = tuple(donor_rows_by_outer_target[outer])
        evidence_values = {policy.policy_id: [] for policy in policies}
        evidence_losses = {policy.policy_id: [] for policy in policies}
        inner_fit_hashes: list[str] = []
        for validation_center in donors:
            validation_products = donor_endpoint_products[
                (outer, validation_center)
            ]
            training = tuple(
                center for center in donors if center != validation_center
            )
            full, deleted = fit_models_for_training_centers(
                rows, training_centers=training
            )
            inner_fit_hashes.extend(
                model.model_hash
                for models in (full, *deleted.values())
                for model in models.values()
            )
            for policy in policies:
                # Preserve the frozen quorum fraction.  Seven-of-eight and
                # eight-of-eight both become seven-of-seven in a seven-center
                # inner fit; the report records this low-power resolution.
                inner_quorum = (
                    0
                    if policy.minimum_delete_donor_positive == 0
                    else math.ceil(
                        policy.minimum_delete_donor_positive
                        * len(training)
                        / len(donors)
                    )
                )
                inner_policy = RoutePolicySpec(
                    policy.policy_id,
                    policy.support_dispersion_multiplier,
                    inner_quorum,
                    policy.require_model,
                    policy.require_support_margin,
                    policy.require_proper_loss,
                )
                decisions = tuple(
                    select_route_for_policy(
                        descriptor,
                        inner_policy,
                        full_models=full if inner_policy.require_model else None,
                        delete_donor_models=(
                            deleted if inner_policy.require_model else None
                        ),
                    )
                    for descriptor in validation_products.descriptors
                )
                gain, loss = _center_contrast(
                    {
                        row.case_id: row
                        for row in validation_products.outer_predictions
                    },
                    decisions,
                    donor_labels_by_outer_target[(outer, validation_center)],
                )
                evidence_values[policy.policy_id].append(gain)
                evidence_losses[policy.policy_id].append(loss)
        evidence = tuple(
            CenterBlockPolicyEvidence(
                policy.policy_id,
                tuple(evidence_values[policy.policy_id]),
                tuple(evidence_losses[policy.policy_id]),
            )
            for policy in policies
        )
        ltt = learn_then_test_center_harm(evidence)
        authorized = tuple(str(value) for value in ltt["authorized_policy_ids"])
        if authorized:
            raise ProtocolError("Descriptive center-block feasibility cannot authorize.")
        selected_policy = next(
            (policy.policy_id for policy in policies if policy.policy_id in authorized),
            None,
        )
        target_rows: list[RouteDecision] = []
        for descriptor in descriptors_by_center[outer]:
            base = target_decisions_by_policy[
                selected_policy or policies[0].policy_id
            ][(outer, descriptor.case_id)]
            target_rows.append(
                _retag_ltt_decision(
                    base,
                    selected_method=(
                        base.selected_method
                        if selected_policy is not None
                        else PORTFOLIO_METHOD_ID
                    ),
                    reason=(
                        f"authorized_by_center_block_feasibility::{selected_policy}"
                        if selected_policy is not None
                        else "fallback_P_center_block_feasibility_has_no_authority"
                    ),
                )
            )
        report = {
            **ltt,
            "outer_target_center": outer,
            "donor_centers": list(donors),
            "inner_training_center_count": 7,
            "effective_calibration_unit_count": 8,
            "case_rows_are_not_inference_units": True,
            "target_center_labels_used": False,
            "target_center_conditional_guarantee_claimed": False,
            "inner_model_hash": canonical_hash(inner_fit_hashes),
        }
        outputs.append(
            TargetLTTAuthorization(
                outer,
                selected_policy,
                tuple(target_rows),
                report,
            )
        )
    return tuple(outputs)


def _center_contrast(
    predictions: Mapping[str, EndpointCasePrediction],
    decisions: Sequence[RouteDecision],
    labels: Sequence[BinaryLabel],
) -> tuple[float, float]:
    by_case = {row.case_id: row for row in decisions}
    label_map = {row.sample_id: row.value for row in labels}
    if (
        set(by_case) != set(predictions)
        or len(label_map) != len(labels)
        or {row.center for row in labels} != {next(iter(predictions.values())).center}
    ):
        raise ProtocolError("Center-block feasibility topology drifted.")
    truth: list[int] = []
    selected_probabilities: list[float] = []
    portfolio_probabilities: list[float] = []
    for case, prediction in predictions.items():
        if set(prediction.sample_ids) - set(label_map):
            raise ProtocolError("Feasibility donor labels do not cover predictions.")
        truth.extend(label_map[sample] for sample in prediction.sample_ids)
        selected_probabilities.extend(
            prediction.probabilities[by_case[case].selected_method]
        )
        portfolio_probabilities.extend(
            prediction.probabilities[PORTFOLIO_METHOD_ID]
        )
    y = np.asarray(truth, dtype=np.int8)
    selected = np.asarray(selected_probabilities, dtype=np.float64)
    portfolio = np.asarray(portfolio_probabilities, dtype=np.float64)
    n_positive = int(np.sum(y == 1, dtype=np.int64))
    n_negative = int(np.sum(y == 0, dtype=np.int64))
    if not n_positive or not n_negative:
        raise ProtocolError("Feasibility donor center lacks both classes.")
    selected_hard = selected >= HARD_THRESHOLD
    portfolio_hard = portfolio >= HARD_THRESHOLD
    gain = 0.5 * (
        (
            np.sum((y == 1) & selected_hard, dtype=np.int64)
            - np.sum((y == 1) & portfolio_hard, dtype=np.int64)
        )
        / n_positive
        + (
            np.sum((y == 0) & (~selected_hard), dtype=np.int64)
            - np.sum((y == 0) & (~portfolio_hard), dtype=np.int64)
        )
        / n_negative
    )
    epsilon = LOG_LOSS_CLIP_EPSILON
    selected = np.clip(selected, epsilon, 1.0 - epsilon)
    portfolio = np.clip(portfolio, epsilon, 1.0 - epsilon)
    selected_loss = -(
        y * np.log(selected) + (1 - y) * np.log(1.0 - selected)
    )
    portfolio_loss = -(
        y * np.log(portfolio) + (1 - y) * np.log(1.0 - portfolio)
    )
    return float(gain), float(
        np.mean(selected_loss - portfolio_loss, dtype=np.float64)
    )


def _retag_ltt_decision(
    source: RouteDecision, *, selected_method: str, reason: str
) -> RouteDecision:
    return RouteDecision(
        source.target_center,
        source.case_id,
        LTT_METHOD_ID,
        source.alternative,
        selected_method,
        source.predicted_bacc_regret,
        source.predicted_log_loss_delta,
        source.delete_bacc_positive_count,
        source.delete_log_loss_safe_count,
        source.support_regret_sum_pp,
        source.support_voter_dispersion_for_sum_pp,
        reason,
        source.descriptor_hash,
        source.model_hashes,
    )


__all__ = (
    "TargetLTTAuthorization",
    "build_ltt_authorizations",
)
