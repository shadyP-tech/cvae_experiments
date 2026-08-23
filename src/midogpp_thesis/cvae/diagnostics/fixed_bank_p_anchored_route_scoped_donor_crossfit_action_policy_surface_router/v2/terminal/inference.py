"""Predeclared descriptive center-block inference for P-DCAPS v2."""

from __future__ import annotations

from itertools import product
from typing import Mapping, Sequence

import numpy as np

from .....expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from .....protocol import ProtocolError
from ...identity import METHOD_MENU, P_METHOD_ID


def exact_shared_center_max_sign_flip(
    center_metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    method_menu: Sequence[str] = METHOD_MENU,
) -> dict[str, object]:
    """Enumerate all 512 shared-center signs with an inclusive MaxT tail."""

    menu = tuple(str(value) for value in method_menu)
    nonreference = tuple(value for value in menu if value != P_METHOD_ID)
    if menu != METHOD_MENU or not nonreference:
        raise ProtocolError("P-DCAPS sign-flip fixed method menu drifted.")
    try:
        reference = center_metrics[P_METHOD_ID]
        deltas = {
            method: np.asarray(
                [
                    float(center_metrics[method][center]["center_bacc"])
                    - float(reference[center]["center_bacc"])
                    for center in CENTERS
                ],
                dtype=np.float64,
            )
            for method in nonreference
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("P-DCAPS sign-flip inputs are incomplete.") from exc
    if any(
        values.shape != (len(CENTERS),) or not np.isfinite(values).all()
        for values in deltas.values()
    ):
        raise ProtocolError("P-DCAPS sign-flip deltas drifted.")

    observed = {
        method: float(np.mean(values, dtype=np.float64))
        for method, values in deltas.items()
    }
    selected = max(
        nonreference,
        key=lambda method: (observed[method], -nonreference.index(method)),
    )
    observed_maximum = observed[selected]
    null_maxima = tuple(
        max(
            float(np.mean(np.asarray(signs) * deltas[method], dtype=np.float64))
            for method in nonreference
        )
        for signs in product((-1.0, 1.0), repeat=len(CENTERS))
    )
    expected = 2 ** len(CENTERS)
    if len(null_maxima) != expected:
        raise ProtocolError("P-DCAPS sign-flip enumeration drifted.")
    exceedance = sum(value >= observed_maximum - 1.0e-15 for value in null_maxima)
    return {
        "schema_version": "pdcaps_v2_shared_center_max_sign_flip_v1",
        "center_order": list(CENTERS),
        "fixed_method_menu": list(nonreference),
        "test_statistic": (
            "maximum_equal_center_mean_BACC_delta_vs_P_over_fixed_method_menu"
        ),
        "tail": "one_sided_improvement_inclusive_ties",
        "observed_mean_center_bacc_delta_vs_P_by_method": observed,
        "observed_selected_method_id": selected,
        "observed_max_statistic": observed_maximum,
        "null_replicate_count": expected,
        "null_exceedance_count": exceedance,
        "selection_aware_descriptive_randomization_p_value": exceedance / expected,
        "minimum_null_max_statistic": min(null_maxima),
        "maximum_null_max_statistic": max(null_maxima),
        "center_signs_shared_across_methods": True,
        "method_identity_reselected_inside_each_null_replicate": True,
        "route_pipeline_refit_inside_null_replicate": False,
        "case_decisions_held_fixed_inside_null_replicate": True,
        "center_blocks_are_exchangeability_assumption": True,
        "descriptive_only": True,
        "formal_claim_authorized": False,
        "nominal_significance_claimed": False,
    }


__all__ = ("exact_shared_center_max_sign_flip",)
