"""Pure action-rectangle posterior orchestration."""

from __future__ import annotations

from ..protocol import GovernanceError
from ..utility.actions import ActionRectangle
from .donor import (
    DonorActionModel,
    assess_donor_support,
    predict_donor_action,
)
from .empirical_bayes import ActionEstimate, combine_empirical_bayes
from .local import LocalResidualModel, predict_local_residual


def estimate_action_rectangle(
    rectangle: ActionRectangle,
    donor_model: DonorActionModel,
    local_model: LocalResidualModel | None,
    *,
    maximum_abs_standardized_feature: float = 4.0,
    minimum_independent_centers: int = 6,
) -> tuple[ActionEstimate, ...]:
    if local_model is not None and (
        local_model.target_center != rectangle.target_center
        or local_model.route_case_id != rectangle.case_id
    ):
        raise GovernanceError("SCALE-BP v2 local model route identity drifted.")
    output: list[ActionEstimate] = []
    for cell in rectangle.cells:
        descriptor = cell.evidence.descriptor
        donor = predict_donor_action(
            donor_model, action_id=cell.action_id, descriptor=descriptor
        )
        within_support, bank_viable = assess_donor_support(
            donor_model,
            descriptor,
            maximum_abs_standardized_feature=maximum_abs_standardized_feature,
            minimum_independent_centers=minimum_independent_centers,
        )
        local = (
            None
            if local_model is None
            else predict_local_residual(
                local_model, action_id=cell.action_id, descriptor=descriptor
            )
        )
        output.append(
            combine_empirical_bayes(
                donor,
                local,
                target_center=rectangle.target_center,
                case_id=rectangle.case_id,
                structural_noop=cell.structural_noop,
                within_support=within_support,
                bank_viable=bank_viable,
            )
        )
    return tuple(output)


__all__ = ("estimate_action_rectangle",)
