from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from types import MappingProxyType

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_multi_challenger_hierarchical_flip_router.science_donor import (
    _fit_target_job,
    _freeze_received_families,
)
from midogpp_thesis.cvae.routing.hierarchical_multi_challenger import (
    DirectionalDonorRow,
)
from midogpp_thesis.cvae.routing.threshold_flip_case_router import (
    ContributionTarget,
    DonorRow,
)


FEATURE_NAMES = ("flip_count", "margin")


def _fit_job() -> tuple[
    str,
    tuple[DirectionalDonorRow, ...],
    tuple[DonorRow, ...],
    int,
    int,
]:
    directional: list[DirectionalDonorRow] = []
    single: list[DonorRow] = []
    for query, source in (("1", "2"), ("2", "1")):
        action_id = f"A1::source={source}"
        for case_ordinal in range(2):
            case_id = f"q{query}-case-{case_ordinal}"
            values = (
                float(2 + case_ordinal),
                float(case_ordinal + int(query) / 10.0),
            )
            target = ContributionTarget(
                case_id=case_id,
                action_id=action_id,
                delta_tp=case_ordinal,
                delta_tn=1 - case_ordinal,
                n_positive=4,
                n_negative=4,
            )
            single.append(
                DonorRow(
                    model_target="0",
                    query_center=query,
                    candidate_source=source,
                    case_id=case_id,
                    action_id=action_id,
                    feature_case_id=case_id,
                    feature_names=FEATURE_NAMES,
                    values=values,
                    target=target,
                )
            )
            for direction, successes in (
                ("0to1", 1 + case_ordinal),
                ("1to0", 2 - case_ordinal),
            ):
                directional.append(
                    DirectionalDonorRow(
                        model_target="0",
                        query_center=query,
                        candidate_source=source,
                        case_id=case_id,
                        action_id=action_id,
                        feature_case_id=case_id,
                        direction=direction,
                        success_count=successes,
                        trial_count=4,
                        feature_names=FEATURE_NAMES,
                        values=values,
                    )
                )
    return "0", tuple(directional), tuple(single), 90_902_026, 1


def test_donor_fit_result_crosses_real_spawn_boundary_then_freezes() -> None:
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
        result = executor.submit(_fit_target_job, _fit_job()).result(timeout=30)

    target, families, single_model, permutation_hash = result
    assert target == "0"
    assert type(families) is dict
    assert set(families) == {"G", "R", "P"}
    assert all(type(by_direction) is dict for by_direction in families.values())
    assert all(
        set(by_direction) == {"0to1", "1to0"}
        for by_direction in families.values()
    )
    assert single_model.model_target == target
    assert len(permutation_hash) == 64

    frozen = _freeze_received_families(families)
    assert isinstance(frozen, MappingProxyType)
    assert all(isinstance(value, MappingProxyType) for value in frozen.values())
    with pytest.raises(TypeError):
        frozen["G"] = families["G"]  # type: ignore[index]
    with pytest.raises(TypeError):
        frozen["G"]["0to1"] = families["G"]["0to1"]  # type: ignore[index]
