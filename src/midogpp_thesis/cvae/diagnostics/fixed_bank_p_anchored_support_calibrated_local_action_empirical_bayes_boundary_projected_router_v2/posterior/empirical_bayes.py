"""Empirical-Bayes shrinkage without collapsing uncertainty components."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..hashing import canonical_hash
from ..protocol import GovernanceError
from ..physical.contracts import ACTION_IDS, MetricVector
from .contracts import ScaleVector
from .donor import DonorPrediction
from .local import LocalResidualPrediction


@dataclass(frozen=True, slots=True)
class ActionEstimate:
    target_center: str
    case_id: str
    action_id: str
    mean: MetricVector
    donor_mean: MetricVector
    local_correction: MetricVector
    shrinkage_weights: tuple[float, float, float]
    transport_rmse: ScaleVector
    donor_heterogeneity: ScaleVector
    donor_estimator_se: ScaleVector
    local_oof_rmse: ScaleVector
    local_fold_heterogeneity: ScaleVector
    local_estimator_se: ScaleVector
    combined_estimator_se: ScaleVector
    structural_noop: bool
    within_support: bool
    bank_viable: bool
    donor_prediction_hash: str
    local_prediction_hash: str | None
    estimate_hash: str = field(init=False)

    def __post_init__(self) -> None:
        weights = tuple(float(value) for value in self.shrinkage_weights)
        if (
            self.action_id not in ACTION_IDS
            or len(weights) != 3
            or any(not 0.0 <= value <= 1.0 for value in weights)
            or not self.donor_prediction_hash
            or (self.structural_noop and self.mean != MetricVector.zeros())
        ):
            raise GovernanceError("SCALE-BP v2 empirical-Bayes estimate drifted.")
        object.__setattr__(self, "shrinkage_weights", weights)
        object.__setattr__(
            self,
            "estimate_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_v2_action_estimate_v2",
                    "target_center": self.target_center,
                    "case_id": self.case_id,
                    "action_id": self.action_id,
                    "mean": self.mean.to_payload(),
                    "donor_mean": self.donor_mean.to_payload(),
                    "local_correction": self.local_correction.to_payload(),
                    "shrinkage_weights": weights,
                    "transport_rmse": self.transport_rmse.to_payload(),
                    "donor_heterogeneity": self.donor_heterogeneity.to_payload(),
                    "donor_estimator_se": self.donor_estimator_se.to_payload(),
                    "local_oof_rmse": self.local_oof_rmse.to_payload(),
                    "local_fold_heterogeneity": self.local_fold_heterogeneity.to_payload(),
                    "local_estimator_se": self.local_estimator_se.to_payload(),
                    "combined_estimator_se": self.combined_estimator_se.to_payload(),
                    "structural_noop": self.structural_noop,
                    "within_support": self.within_support,
                    "bank_viable": self.bank_viable,
                    "donor_prediction_hash": self.donor_prediction_hash,
                    "local_prediction_hash": self.local_prediction_hash,
                    "uncertainty_components_collapsed": False,
                    "shrinkage_signal_variance": "donor_heterogeneity_squared",
                    "transport_rmse_used_as_independent_variance": False,
                    "combined_estimator_variance": "donor_se_squared_plus_w_squared_local_se_squared",
                }
            ),
        )


def combine_empirical_bayes(
    donor: DonorPrediction,
    local: LocalResidualPrediction | None,
    *,
    target_center: object,
    case_id: object,
    structural_noop: bool,
    within_support: bool = True,
    bank_viable: bool = True,
) -> ActionEstimate:
    target, case = str(target_center), str(case_id)
    if local is not None and (
        local.action_id != donor.action_id
        or local.descriptor_hash != donor.descriptor_hash
    ):
        raise GovernanceError("SCALE-BP v2 donor/local prediction identity drifted.")
    if structural_noop:
        zero_metric = MetricVector.zeros()
        zero_scale = ScaleVector.zeros()
        return ActionEstimate(
            target,
            case,
            donor.action_id,
            zero_metric,
            zero_metric,
            zero_metric,
            (0.0, 0.0, 0.0),
            zero_scale,
            zero_scale,
            zero_scale,
            zero_scale,
            zero_scale,
            zero_scale,
            zero_scale,
            True,
            bool(within_support),
            bool(bank_viable),
            donor.prediction_hash,
            None if local is None else local.prediction_hash,
        )
    donor_mean = donor.mean.as_array()
    if local is None:
        correction = np.zeros(3, dtype=np.float64)
        weights = np.zeros(3, dtype=np.float64)
        local_oof = np.zeros(3, dtype=np.float64)
        local_heterogeneity = np.zeros(3, dtype=np.float64)
        local_se = np.zeros(3, dtype=np.float64)
        local_hash = None
    else:
        correction = local.correction.as_array()
        # The EB signal variance is the between-center heterogeneity only.
        # Transport RMSE is a predictive-error diagnostic from the same OOF
        # residuals and must not be added as if it were independent variance.
        prior_variance = np.square(donor.heterogeneity.as_tuple())
        local_oof = np.asarray(local.oof_rmse.as_tuple(), dtype=np.float64)
        local_heterogeneity = np.asarray(
            local.fold_heterogeneity.as_tuple(), dtype=np.float64
        )
        local_se = np.asarray(local.estimator_se.as_tuple(), dtype=np.float64)
        # Local fold heterogeneity and OOF RMSE are descriptive views of the
        # same fold-error surface.  Use OOF RMSE plus estimator uncertainty for
        # shrinkage; retain fold heterogeneity separately for the envelope.
        local_noise = local_oof**2 + local_se**2
        denominator = prior_variance + local_noise
        weights = np.divide(
            prior_variance,
            denominator,
            out=np.zeros_like(prior_variance),
            where=denominator > 0.0,
        )
        local_hash = local.prediction_hash
    mean = donor_mean + weights * correction
    donor_se = np.asarray(donor.estimator_se.as_tuple(), dtype=np.float64)
    # The posterior mean is donor_mean + w * local_correction, so donor_mean
    # retains coefficient one.  Its estimator uncertainty cannot disappear as
    # local weight approaches one.
    combined_se = np.sqrt(donor_se**2 + weights**2 * local_se**2)
    return ActionEstimate(
        target,
        case,
        donor.action_id,
        MetricVector.from_array(mean),
        donor.mean,
        MetricVector.from_array(correction),
        tuple(float(value) for value in weights),
        donor.transport_rmse,
        donor.heterogeneity,
        donor.estimator_se,
        ScaleVector.from_values(local_oof),
        ScaleVector.from_values(local_heterogeneity),
        ScaleVector.from_values(local_se),
        ScaleVector.from_values(combined_se),
        False,
        bool(within_support),
        bool(bank_viable),
        donor.prediction_hash,
        local_hash,
    )


__all__ = ("ActionEstimate", "combine_empirical_bayes")
