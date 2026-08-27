from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import math

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.donor_prior import (
    CELL_IDS,
    DonorDeleteCenterFold,
    DonorObservation,
    DonorPriorModel,
    DonorPriorPrediction,
    crossfit_donor_prediction,
    fit_final_donor_prior,
    predict_donor_prior,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.donor_contracts import (
    DonorObservation as ContractDonorObservation,
    DonorPriorModel as ContractDonorPriorModel,
    DonorPriorPrediction as ContractDonorPriorPrediction,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.case_inventory import (
    DatasetCaseInventory,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.influence.contracts import (
    ActionDescriptor,
    ActionMetricVector,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.replay_scope import (
    FinalDonorScope,
    PseudoReplayScope,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.identity import (
    ACTION_IDS,
    CELL_IDS as IDENTITY_CELL_IDS,
    CENTERS,
    EXPECTED_CASE_COUNTS_BY_CENTER,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.route_identity import (
    RouteIdentityInventory,
    RouteScopeWitness,
    SampleIdentity,
    build_route_identity_inventory,
)
from midogpp_thesis.cvae.protocol import ProtocolError


SHA = "c" * 64


@lru_cache(maxsize=1)
def _inventory() -> DatasetCaseInventory:
    counts = dict(EXPECTED_CASE_COUNTS_BY_CENTER)
    return DatasetCaseInventory(
        SHA,
        SHA,
        SHA,
        tuple(
            (
                center,
                tuple(f"case-{center}-{index:03d}" for index in range(counts[center])),
            )
            for center in CENTERS
        ),
    )


@lru_cache(maxsize=1)
def _route_identity_inventory() -> RouteIdentityInventory:
    inventory = _inventory()
    return build_route_identity_inventory(
        tuple(
            SampleIdentity(
                center,
                case,
                f"group-{case}",
                f"patient-{case}",
                f"slide-{case}",
                f"sample-{case}",
            )
            for center in CENTERS
            for case in inventory.cases(center)
        ),
        case_inventory=inventory,
    )


def _descriptor(case: str, family: str, direction: str, value: float) -> ActionDescriptor:
    return ActionDescriptor(
        case,
        f"{family}::{direction}",
        family,
        direction,
        ("margin", "shift"),
        (value, 2.0 * value),
        1,
        2,
        SHA,
        SHA,
        SHA,
    )


def _final_scope() -> FinalDonorScope:
    return FinalDonorScope(
        "9",
        "case-9-000",
        RouteScopeWitness("9", "case-9-000", _route_identity_inventory()),
        _inventory(),
    )


def _pseudo_scope() -> PseudoReplayScope:
    return PseudoReplayScope(
        "9",
        "8",
        "case-8-000",
        RouteScopeWitness("8", "case-8-000", _route_identity_inventory()),
        _inventory(),
    )


def _rows(
    scope: FinalDonorScope | PseudoReplayScope,
    *,
    deleted_center: str | None = None,
) -> tuple[DonorObservation, ...]:
    rows: list[DonorObservation] = []
    for center_index, center in enumerate(scope.donor_training_centers):
        if center == deleted_center:
            continue
        sources = tuple(
            value
            for value in scope.donor_training_centers
            if value not in {center, deleted_center}
        )
        for case_index, case_id in enumerate(scope.case_inventory.cases(center)):
            family = ("B", "I", "R")[case_index % 3]
            direction = ("zero_to_one", "one_to_zero")[case_index % 2]
            value = 0.01 * (1 + center_index + case_index)
            descriptor = _descriptor(case_id, family, direction, value)
            rows.append(
                DonorObservation(
                    center,
                    descriptor.case_id,
                    sources,
                    descriptor,
                    ActionMetricVector(value, -0.1 * value, -0.2 * value),
                    scope.scope_hash,
                )
                    )
    return tuple(rows)


def _folds(
    scope: FinalDonorScope | PseudoReplayScope,
) -> tuple[DonorDeleteCenterFold, ...]:
    return tuple(
        DonorDeleteCenterFold(center, _rows(scope, deleted_center=center))
        for center in scope.donor_training_centers
    )


def test_final_donor_prior_is_center_balanced_scope_bound_and_h_excluded() -> None:
    scope = _final_scope()
    model = fit_final_donor_prior(
        _rows(scope), scope=scope, delete_center_folds=_folds(scope)
    )
    assert model.held_center == "9"
    assert model.scope_hash == scope.scope_hash
    assert model.fit_role == "FINAL_H_C"
    assert "9" not in model.training_centers
    assert model.training_centers == tuple(center for center in CENTERS if center != "9")
    prediction = predict_donor_prior(
        model,
        _descriptor("case-9-000", "B", "zero_to_one", 0.25),
        scope=scope,
    )
    assert prediction.scope_hash == scope.scope_hash
    assert all(math.isfinite(value) for value in prediction.mean.as_tuple())
    assert all(
        value >= 0.0
        for value in prediction.between_center_standard_error.as_tuple()
    )


def test_donor_facade_is_backward_compatible_and_repeat_hash_deterministic() -> None:
    assert CELL_IDS == ACTION_IDS == IDENTITY_CELL_IDS
    assert DonorObservation is ContractDonorObservation
    assert DonorPriorModel is ContractDonorPriorModel
    assert DonorPriorPrediction is ContractDonorPriorPrediction
    scope = _final_scope()
    rows = _rows(scope)
    folds = _folds(scope)
    first = fit_final_donor_prior(rows, scope=scope, delete_center_folds=folds)
    second = fit_final_donor_prior(rows, scope=scope, delete_center_folds=folds)
    assert first.model_hash == second.model_hash
    descriptor = _descriptor("case-9-000", "B", "zero_to_one", 0.25)
    assert predict_donor_prior(
        first, descriptor, scope=scope
    ).prediction_hash == predict_donor_prior(
        second, descriptor, scope=scope
    ).prediction_hash


def test_pseudo_crossfit_mechanically_excludes_outer_h_pseudo_j_and_held_d() -> None:
    scope = _pseudo_scope()
    assert scope.route_scope_hash == scope.route_witness.witness_hash
    prediction = crossfit_donor_prediction(
        _rows(scope),
        scope=scope,
        descriptor=_descriptor("case-8-000", "I", "one_to_zero", 0.25),
        delete_center_folds=_folds(scope),
    )
    assert prediction.scope_hash == scope.scope_hash
    assert prediction.fit_role == "PSEUDO_H_J_D"


def test_donor_scope_rejects_pseudo_j_or_held_d_fit_poison() -> None:
    scope = _pseudo_scope()
    rows = list(_rows(scope))
    template = rows[0]
    poisoned_j = DonorObservation(
        "8",
        "case-8-001",
        scope.donor_training_centers,
        _descriptor("case-8-001", "B", "zero_to_one", 0.2),
        ActionMetricVector.zeros(),
        scope.scope_hash,
    )
    with pytest.raises(ProtocolError, match="H/J/d scope lineage"):
        crossfit_donor_prediction(
            (*rows, poisoned_j),
            scope=scope,
            descriptor=_descriptor("case-8-000", "B", "zero_to_one", 0.2),
            delete_center_folds=_folds(scope),
        )

    poisoned_d = DonorObservation(
        template.query_center,
        "case-8-000",
        template.source_centers,
        _descriptor("case-8-000", "B", "zero_to_one", 0.2),
        ActionMetricVector.zeros(),
        scope.scope_hash,
    )
    with pytest.raises(ProtocolError, match="H/J/d scope lineage"):
        crossfit_donor_prediction(
            (poisoned_d, *rows[1:]),
            scope=scope,
            descriptor=_descriptor("case-8-000", "B", "zero_to_one", 0.2),
            delete_center_folds=_folds(scope),
        )


def test_delete_center_fold_rejects_candidate_source_poison() -> None:
    scope = _final_scope()
    fold = _folds(scope)[0]
    row = fold.training_observations[0]
    poisoned = replace(
        row,
        source_centers=tuple(sorted((*row.source_centers, fold.deleted_center))),
    )
    with pytest.raises(ProtocolError, match="delete-center donor fold"):
        DonorDeleteCenterFold(
            fold.deleted_center, (poisoned, *fold.training_observations[1:])
        )
