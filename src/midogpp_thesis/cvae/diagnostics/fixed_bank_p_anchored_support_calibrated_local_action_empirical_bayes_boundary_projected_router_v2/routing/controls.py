"""Frozen negative and attribution controls for the direct-action router."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from ..protocol import GovernanceError
from ..physical.contracts import ACTION_IDS, MetricVector
from ..posterior.contracts import LocalResidualObservation, ScaleVector
from ..posterior.empirical_bayes import ActionEstimate


CONTROL_METHOD_IDS = (
    "SCALE_BP_V2_DONOR_ONLY",
    "SCALE_BP_V2_LOCAL_ONLY",
    "SCALE_BP_V2_SUPPORT_LABEL_PERMUTATION",
    "SCALE_BP_V2_CYCLIC_ACTION_IDENTITY_POISON",
    "SCALE_BP_V2_FULL_ENDPOINT_SENSITIVITY",
)


def donor_only_estimates(
    estimates: Sequence[ActionEstimate],
) -> tuple[ActionEstimate, ...]:
    rows = _complete(estimates)
    zero_metric, zero_scale = MetricVector.zeros(), ScaleVector.zeros()
    return tuple(
        replace(
            row,
            mean=zero_metric if row.structural_noop else row.donor_mean,
            local_correction=zero_metric,
            shrinkage_weights=(0.0, 0.0, 0.0),
            local_oof_rmse=zero_scale,
            local_fold_heterogeneity=zero_scale,
            local_estimator_se=zero_scale,
            combined_estimator_se=row.donor_estimator_se,
            local_prediction_hash=None,
        )
        for row in rows
    )


def local_only_estimates(
    estimates: Sequence[ActionEstimate],
) -> tuple[ActionEstimate, ...]:
    rows = _complete(estimates)
    zero_metric, zero_scale = MetricVector.zeros(), ScaleVector.zeros()
    return tuple(
        replace(
            row,
            mean=zero_metric if row.structural_noop else row.local_correction,
            donor_mean=zero_metric,
            shrinkage_weights=(1.0, 1.0, 1.0),
            transport_rmse=zero_scale,
            donor_heterogeneity=zero_scale,
            donor_estimator_se=zero_scale,
            combined_estimator_se=row.local_estimator_se,
        )
        for row in rows
    )


def permute_local_residuals(
    observations: Sequence[LocalResidualObservation],
) -> tuple[LocalResidualObservation, ...]:
    """Cyclically permute response blocks across support cases within action."""

    rows = tuple(observations)
    cases = tuple(sorted({row.support_case_id for row in rows}))
    expected = {(case, action) for case in cases for action in ACTION_IDS}
    if len(cases) < 2 or {(row.support_case_id, row.action_id) for row in rows} != expected:
        raise GovernanceError("SCALE-BP v2 support permutation rectangle drifted.")
    by_key = {(row.support_case_id, row.action_id): row for row in rows}
    predecessor = {case: cases[(index - 1) % len(cases)] for index, case in enumerate(cases)}
    output: list[LocalResidualObservation] = []
    for case in cases:
        for action in ACTION_IDS:
            row = by_key[(case, action)]
            donor = by_key[(predecessor[case], action)]
            output.append(
                LocalResidualObservation(
                    row.target_center,
                    row.route_case_id,
                    row.support_case_id,
                    row.action_id,
                    row.descriptor,
                    donor.residual,
                    donor.donor_prediction_hash,
                    row.support_scope_hash,
                    row.endpoint_plan_hash,
                    row.support_excluded_case_ids,
                    row.outer_held_case_id,
                )
            )
    return tuple(output)


def cyclically_poison_action_identities(
    estimates: Sequence[ActionEstimate],
) -> tuple[ActionEstimate, ...]:
    rows = _complete(estimates)
    return tuple(
        replace(row, action_id=ACTION_IDS[(index + 1) % len(ACTION_IDS)])
        for index, row in enumerate(rows)
    )


def _complete(estimates: Sequence[ActionEstimate]) -> tuple[ActionEstimate, ...]:
    rows = tuple(estimates)
    if tuple(row.action_id for row in rows) != ACTION_IDS:
        raise GovernanceError("SCALE-BP v2 control requires the complete action menu.")
    return rows


__all__ = (
    "CONTROL_METHOD_IDS",
    "cyclically_poison_action_identities",
    "donor_only_estimates",
    "local_only_estimates",
    "permute_local_residuals",
)
