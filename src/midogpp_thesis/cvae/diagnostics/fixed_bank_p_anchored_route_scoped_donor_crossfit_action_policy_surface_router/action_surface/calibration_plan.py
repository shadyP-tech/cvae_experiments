"""Workstation-efficient action-calibration fit plan.

Reliability for pseudo center J must score every other center K with a model
that saw neither H, J, nor K.  The numerical fit for exclusions {J, K} is
identical in both scoring orientations, so this module solves it once and
rebinds only the immutable scored-center provenance.  That reduces each outer
H from 195 to 111 three-coordinate ridge solves without changing predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ....expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ....protocol import ProtocolError
from ..identity import METRICS, canonical_hash
from .contracts import ActionCalibrationModel, ActionPrediction, ActionResponse
from .runtime import fit_action_calibration_models


@dataclass(frozen=True)
class ActionCalibrationFamilies:
    outer_center: str
    target_models: tuple[ActionCalibrationModel, ...]
    pseudo_models_by_center: tuple[
        tuple[str, tuple[ActionCalibrationModel, ...]], ...
    ]
    pseudo_reliability_oof_by_context: tuple[
        tuple[
            str,
            tuple[tuple[str, tuple[ActionCalibrationModel, ...]], ...],
        ],
        ...,
    ]
    numerical_metric_fit_count: int
    serialized_model_count: int
    plan_hash: str = field(init=False)

    def __post_init__(self) -> None:
        donors = tuple(center for center in CENTERS if center != self.outer_center)
        pseudo = tuple(center for center, _models in self.pseudo_models_by_center)
        contexts = tuple(
            center for center, _models in self.pseudo_reliability_oof_by_context
        )
        if (
            self.outer_center not in CENTERS
            or tuple(model.metric for model in self.target_models) != METRICS
            or pseudo != donors
            or contexts != donors
            or int(self.numerical_metric_fit_count)
            != len(METRICS) * (1 + len(donors) + len(donors) * (len(donors) - 1) // 2)
            or int(self.serialized_model_count)
            != len(METRICS) * (1 + len(donors) + len(donors) * (len(donors) - 1))
        ):
            raise ProtocolError("P-DCAPS optimized action calibration plan drifted.")
        object.__setattr__(
            self,
            "plan_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_action_calibration_families_v1",
                    "outer_center": self.outer_center,
                    "target_model_hashes": tuple(
                        row.model_hash for row in self.target_models
                    ),
                    "pseudo_model_hashes": tuple(
                        (center, tuple(row.model_hash for row in models))
                        for center, models in self.pseudo_models_by_center
                    ),
                    "pseudo_reliability_model_hashes": tuple(
                        (
                            context,
                            tuple(
                                (
                                    scored,
                                    tuple(row.model_hash for row in models),
                                )
                                for scored, models in rows
                            ),
                        )
                        for context, rows in self.pseudo_reliability_oof_by_context
                    ),
                    "numerical_metric_fit_count": self.numerical_metric_fit_count,
                    "serialized_model_count": self.serialized_model_count,
                    "unordered_exclusion_pair_fit_reuse": True,
                }
            ),
        )

    @property
    def pseudo_models(self) -> Mapping[str, tuple[ActionCalibrationModel, ...]]:
        return dict(self.pseudo_models_by_center)

    @property
    def target_reliability_oof(
        self,
    ) -> Mapping[str, tuple[ActionCalibrationModel, ...]]:
        # Leave-K models used to route pseudo K are exactly the OOF models
        # required to evaluate target-H action reliability.
        return self.pseudo_models

    def pseudo_reliability_oof(
        self, context: str
    ) -> Mapping[str, tuple[ActionCalibrationModel, ...]]:
        try:
            return dict(dict(self.pseudo_reliability_oof_by_context)[str(context)])
        except KeyError as exc:
            raise ProtocolError(
                "P-DCAPS pseudo reliability context is absent."
            ) from exc

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_action_calibration_families_v1",
            "outer_center": self.outer_center,
            "target_models": [row.to_payload() for row in self.target_models],
            "pseudo_models_by_center": [
                [center, [row.to_payload() for row in models]]
                for center, models in self.pseudo_models_by_center
            ],
            "pseudo_reliability_oof_by_context": [
                [
                    context,
                    [
                        [scored, [row.to_payload() for row in models]]
                        for scored, models in rows
                    ],
                ]
                for context, rows in self.pseudo_reliability_oof_by_context
            ],
            "numerical_metric_fit_count": self.numerical_metric_fit_count,
            "serialized_model_count": self.serialized_model_count,
            "unordered_exclusion_pair_fit_reuse": True,
            "plan_hash": self.plan_hash,
        }


def _rebind_scored_center(
    models: Sequence[ActionCalibrationModel],
    *,
    scored_center: str,
) -> tuple[ActionCalibrationModel, ...]:
    rows = tuple(models)
    scored = str(scored_center)
    if (
        tuple(row.metric for row in rows) != METRICS
        or scored not in rows[0].all_excluded_centers
        or scored == rows[0].excluded_outer_center
    ):
        raise ProtocolError("P-DCAPS action model rebind exclusion drifted.")
    return tuple(
        ActionCalibrationModel(
            model.metric,
            model.excluded_outer_center,
            scored,
            model.training_centers,
            model.feature_names,
            model.feature_mean,
            model.feature_scale,
            model.intercept,
            model.coefficients,
            model.ridge_alpha,
            model.training_row_count,
            model.training_response_hash,
            model.weight_audit_hash,
            model.solver,
        )
        for model in rows
    )


def build_optimized_action_calibration_families(
    predictions: Sequence[ActionPrediction],
    responses: Sequence[ActionResponse],
    *,
    outer_center: str,
) -> ActionCalibrationFamilies:
    """Fit target, leave-J, and unordered leave-{J,K} model families."""

    outer = str(outer_center)
    if outer not in CENTERS:
        raise ProtocolError("P-DCAPS optimized action outer center drifted.")
    donors = tuple(center for center in CENTERS if center != outer)
    target = fit_action_calibration_models(
        predictions, responses, outer_center=outer
    )
    pseudo = tuple(
        (
            center,
            fit_action_calibration_models(
                predictions,
                responses,
                outer_center=outer,
                scored_center=center,
            ),
        )
        for center in donors
    )

    oriented: dict[
        tuple[str, str], tuple[ActionCalibrationModel, ...]
    ] = {}
    for left_index, left in enumerate(donors):
        for right in donors[left_index + 1 :]:
            # Score right while additionally excluding left, then reuse the
            # identical numerical fit with right/left provenance reversed.
            right_scored = fit_action_calibration_models(
                predictions,
                responses,
                outer_center=outer,
                scored_center=right,
                additional_excluded_centers=(left,),
            )
            oriented[(left, right)] = right_scored
            oriented[(right, left)] = _rebind_scored_center(
                right_scored, scored_center=left
            )

    reliability = tuple(
        (
            context,
            tuple(
                (scored, oriented[(context, scored)])
                for scored in donors
                if scored != context
            ),
        )
        for context in donors
    )
    numerical_fit_count = len(METRICS) * (
        1 + len(donors) + len(donors) * (len(donors) - 1) // 2
    )
    serialized_model_count = len(METRICS) * (
        1 + len(donors) + len(donors) * (len(donors) - 1)
    )
    return ActionCalibrationFamilies(
        outer,
        target,
        pseudo,
        reliability,
        numerical_fit_count,
        serialized_model_count,
    )


__all__ = (
    "ActionCalibrationFamilies",
    "build_optimized_action_calibration_families",
)
