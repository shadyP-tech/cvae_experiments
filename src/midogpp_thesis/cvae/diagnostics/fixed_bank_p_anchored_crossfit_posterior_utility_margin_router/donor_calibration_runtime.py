"""Outer-center-excluded donor response and margin-calibration phase."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType

from ...protocol import ProtocolError
from .constants import CENTERS
from .contracts import BinaryLabel
from .margin_calibration import calibrate_margin
from .outer_endpoint_runtime import OuterEndpointProducts
from .posterior_contracts import RoutePosteriorEnsemble
from .posterior_utility import score_posterior_utilities
from .utility_contracts import (
    DonorUtilityRow,
    MarginCalibration,
    PosteriorUtilityPrediction,
    UtilityDescriptor,
)
from .utility_features import build_utility_descriptor_surface
from .utility_responses import build_donor_utility_rows


def build_donor_calibrations(
    donor_products: Mapping[tuple[str, str], OuterEndpointProducts],
    donor_labels: Mapping[tuple[str, str], Sequence[BinaryLabel]],
    ensembles: Mapping[
        str, Mapping[tuple[str, str], RoutePosteriorEnsemble]
    ],
    *,
    control_ids: Sequence[str],
) -> tuple[
    Mapping[str, tuple[DonorUtilityRow, ...]],
    Mapping[tuple[str, str], tuple[PosteriorUtilityPrediction, ...]],
    Mapping[tuple[str, str], MarginCalibration],
]:
    """Build realized donor rows, analytic scores, and nested margins."""

    controls = tuple(control_ids)
    donor_rows: dict[str, tuple[DonorUtilityRow, ...]] = {}
    donor_utilities: dict[tuple[str, str], tuple[PosteriorUtilityPrediction, ...]] = {}
    calibrations: dict[tuple[str, str], MarginCalibration] = {}
    for outer in CENTERS:
        rows: list[DonorUtilityRow] = []
        utility_rows: dict[str, list[PosteriorUtilityPrediction]] = {
            control: [] for control in controls
        }
        for donor in CENTERS:
            if donor == outer:
                continue
            labels = tuple(donor_labels[(outer, donor)])
            by_case = {
                case: tuple(row for row in labels if row.case_id == case)
                for case in dict.fromkeys(row.case_id for row in labels)
            }
            n_positive = sum(row.value == 1 for row in labels)
            n_negative = sum(row.value == 0 for row in labels)
            products = donor_products[(outer, donor)]
            descriptors = build_utility_descriptor_surface(products.predictions)
            descriptors_by_case = _group_by_case(descriptors)
            if set(by_case) != {row.case_id for row in products.predictions}:
                raise ProtocolError("PUMR donor cases do not align with endpoints.")
            for prediction in products.predictions:
                case_descriptors = descriptors_by_case[prediction.case_id]
                rows.extend(
                    build_donor_utility_rows(
                        outer_target_center=outer,
                        prediction=prediction,
                        descriptors=case_descriptors,
                        case_labels=by_case[prediction.case_id],
                        center_n_positive=n_positive,
                        center_n_negative=n_negative,
                    )
                )
                for control in controls:
                    utility_rows[control].extend(
                        score_posterior_utilities(
                            prediction,
                            case_descriptors,
                            ensembles[control][(donor, prediction.case_id)],
                        )
                    )
        donor_rows[outer] = tuple(sorted(rows, key=lambda row: row.key))
        for control in controls:
            key = (outer, control)
            donor_utilities[key] = tuple(
                sorted(utility_rows[control], key=lambda row: row.key)
            )
            calibrations[key] = calibrate_margin(
                outer_target_center=outer,
                control_id=control,
                predictions=donor_utilities[key],
                donor_rows=donor_rows[outer],
            )
    return (
        MappingProxyType(donor_rows),
        MappingProxyType(donor_utilities),
        MappingProxyType(calibrations),
    )


def _group_by_case(
    descriptors: Sequence[UtilityDescriptor],
) -> Mapping[str, tuple[UtilityDescriptor, ...]]:
    cases = dict.fromkeys(row.case_id for row in descriptors)
    return MappingProxyType(
        {case: tuple(row for row in descriptors if row.case_id == case) for case in cases}
    )


__all__ = ("build_donor_calibrations",)
