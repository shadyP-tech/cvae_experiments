"""Pairwise-residual ranker and selected-policy adapter for HARP v9.

This module deliberately contains no endpoint-wise safety-certificate path.
The science core ranks the sealed physical actions relative to the virtual
exact-B control, learns acceptance on cross-fitted *selected* actions, and
calibrates one whole-policy acceptance threshold inside source-center LODO.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ...routing.policy_calibrated_residual_router_v9 import (
    AdmissionConfig,
    CasePrediction,
    EffectiveMenu,
    OuterAdmission,
    PairwiseFitConfig,
    PolicyCalibration,
    PolicyRiskConfig,
    SourceActionOutcome,
    SourceLODOResult,
    calibrate_selected_policy,
    evaluate_outer_admission,
    fit_source_lodo,
    predict_case,
)
from .science_pool import execute_science_jobs, science_pool_plan
from .source_development import SourceDevelopmentState


@dataclass(frozen=True, slots=True)
class OuterRouterBundle:
    outer_target_id: str
    lodo: SourceLODOResult

    def __post_init__(self) -> None:
        if self.lodo.outer_target_id != self.outer_target_id:
            raise ProtocolError("HARP v9 outer model crossed its held target.")


@dataclass(frozen=True, slots=True)
class RouterFitState:
    bundles: tuple[OuterRouterBundle, ...]

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.bundles, key=lambda row: row.outer_target_id))
        if not ordered or len({row.outer_target_id for row in ordered}) != len(ordered):
            raise ProtocolError("HARP v9 fitted outer-router inventory is malformed.")
        object.__setattr__(self, "bundles", ordered)

    def for_outer(self, outer_target_id: str) -> OuterRouterBundle:
        for bundle in self.bundles:
            if bundle.outer_target_id == str(outer_target_id):
                return bundle
        raise ProtocolError("HARP v9 fitted router lacks an outer target.")


@dataclass(frozen=True, slots=True)
class OuterPolicyState:
    outer_target_id: str
    admission: OuterAdmission
    calibration: PolicyCalibration

    def __post_init__(self) -> None:
        if (
            self.admission.outer_target_id != self.outer_target_id
            or self.calibration.outer_target_id != self.outer_target_id
        ):
            raise ProtocolError("HARP v9 local policy crossed its outer target.")

    @property
    def policy_enabled(self) -> bool:
        return self.admission.admitted and self.calibration.calibrated


@dataclass(frozen=True, slots=True)
class RouterAdmissionState:
    by_outer: tuple[OuterPolicyState, ...]

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.by_outer, key=lambda row: row.outer_target_id))
        if not ordered or len({row.outer_target_id for row in ordered}) != len(ordered):
            raise ProtocolError("HARP v9 per-outer policy inventory is malformed.")
        object.__setattr__(self, "by_outer", ordered)

    def for_outer(self, outer_target_id: str) -> OuterPolicyState:
        for value in self.by_outer:
            if value.outer_target_id == str(outer_target_id):
                return value
        raise ProtocolError("HARP v9 local policy lacks an outer target.")


@dataclass(frozen=True, slots=True)
class TargetEvidenceState:
    menus: tuple[EffectiveMenu, ...]
    predictions: tuple[CasePrediction, ...]

    def __post_init__(self) -> None:
        menus = tuple(sorted(self.menus, key=lambda row: (row.outer_target_id, row.case_id)))
        predictions = tuple(
            sorted(self.predictions, key=lambda row: (row.outer_target_id, row.case_id))
        )
        menu_keys = {(row.outer_target_id, row.case_id) for row in menus}
        prediction_keys = {(row.outer_target_id, row.case_id) for row in predictions}
        if (
            not menus
            or menu_keys != prediction_keys
            or len(menu_keys) != len(menus)
            or any(row.query_center_id != row.outer_target_id for row in menus)
        ):
            raise ProtocolError("HARP v9 target evidence inventory drifted.")
        by_key = {(row.outer_target_id, row.case_id): row for row in menus}
        if any(
            row.menu_hash != by_key[(row.outer_target_id, row.case_id)].menu_hash
            for row in predictions
        ):
            raise ProtocolError("HARP v9 target prediction/menu hash drifted.")
        object.__setattr__(self, "menus", menus)
        object.__setattr__(self, "predictions", predictions)

    def case(self, outer: str, case: str) -> tuple[EffectiveMenu, CasePrediction]:
        menu = next(
            (row for row in self.menus if row.outer_target_id == outer and row.case_id == case),
            None,
        )
        prediction = next(
            (
                row
                for row in self.predictions
                if row.outer_target_id == outer and row.case_id == case
            ),
            None,
        )
        if menu is None or prediction is None:
            raise ProtocolError("HARP v9 target case evidence is absent.")
        return menu, prediction


def _fit_grid(model_config: Mapping[str, object]) -> tuple[PairwiseFitConfig, ...]:
    try:
        pairwise = tuple(float(value) for value in model_config["pairwise_alpha_grid"])
        residual = tuple(float(value) for value in model_config["residual_alpha_grid"])
        acceptor = tuple(float(value) for value in model_config["acceptor_alpha_grid"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("HARP v9 nested source fit grid is malformed.") from exc
    grid = tuple(
        PairwiseFitConfig(
            pairwise_alpha=pairwise_alpha,
            residual_alpha=residual_alpha,
            acceptor_alpha=acceptor_alpha,
        )
        for pairwise_alpha, residual_alpha, acceptor_alpha in product(
            pairwise, residual, acceptor
        )
    )
    if len(grid) != 1:
        raise ProtocolError("HARP v9 source fit requires one predeclared fixed configuration.")
    return grid


def _fit_outer_task(
    payload: tuple[
        str,
        tuple[EffectiveMenu, ...],
        tuple[SourceActionOutcome, ...],
        tuple[PairwiseFitConfig, ...],
    ]
) -> OuterRouterBundle:
    outer, menus, outcomes, grid = payload
    result = fit_source_lodo(
        outcomes,
        effective_menus=menus,
        config_grid=grid,
    )
    return OuterRouterBundle(outer, result)


def fit_outer_routers(
    development: SourceDevelopmentState,
    *,
    model_config: Mapping[str, object],
    runtime_config: Mapping[str, object],
) -> RouterFitState:
    if not isinstance(development, SourceDevelopmentState):
        raise ProtocolError("HARP v9 fitting requires a typed development state.")
    science_pool_plan(runtime_config)
    grid = _fit_grid(model_config)
    outers = tuple(sorted({menu.outer_target_id for menu in development.effective_menus}))
    tasks = tuple(
        (
            outer,
            tuple(menu for menu in development.effective_menus if menu.outer_target_id == outer),
            tuple(row for row in development.outcomes if row.action.outer_target_id == outer),
            grid,
        )
        for outer in outers
    )
    receipt = execute_science_jobs(
        tasks,
        _fit_outer_task,
        weights=tuple(len(task[1]) * len(grid) for task in tasks),
        workers=int(runtime_config["science_workers"]),
        threads_per_worker=int(runtime_config["science_blas_threads_per_worker"]),
    )
    return RouterFitState(tuple(receipt.values))


def _admission_config(model_config: Mapping[str, object]) -> AdmissionConfig:
    raw = model_config.get("admission")
    if not isinstance(raw, Mapping):
        raise ProtocolError("HARP v9 admission config is absent.")
    return AdmissionConfig(
        min_pooled_top1_excess=float(raw["min_pooled_top1_excess_over_always_b"]),
        min_delete_center_top1_excess=float(
            raw["min_delete_center_top1_excess_over_always_b"]
        ),
        min_opportunity_top1_accuracy=float(raw["min_opportunity_top1_accuracy"]),
        min_opportunity_cases=int(raw["min_opportunity_cases"]),
    )


def _risk_config(model_config: Mapping[str, object]) -> PolicyRiskConfig:
    raw = model_config.get("policy")
    if not isinstance(raw, Mapping):
        raise ProtocolError("HARP v9 policy risk config is absent.")
    return PolicyRiskConfig(
        acceptance_thresholds=tuple(
            float(value) for value in model_config["acceptance_threshold_grid"]
        ),
        fixed_rank_margin_threshold=float(model_config["rank_margin_fixed_guard"]),
        min_case_equal_bacc_gain=float(raw["min_case_equal_bacc_gain"]),
        min_delete_center_bacc_gain=float(raw["min_delete_center_bacc_gain"]),
        max_routed_harm_rate=float(raw["max_routed_harm_rate"]),
        max_case_equal_brier_delta=float(raw["max_case_equal_brier_delta"]),
        max_case_equal_log_delta=float(raw["max_case_equal_log_loss_delta"]),
        min_coverage=float(raw["min_coverage"]),
        min_routed_cases=int(raw["min_routed_cases"]),
    )


def build_source_only_admission(
    fitted: RouterFitState,
    development: SourceDevelopmentState,
    *,
    model_config: Mapping[str, object],
) -> RouterAdmissionState:
    if not isinstance(fitted, RouterFitState) or not isinstance(
        development, SourceDevelopmentState
    ):
        raise ProtocolError("HARP v9 policy admission requires fitted source state.")
    admission_config = _admission_config(model_config)
    risk_config = _risk_config(model_config)
    policies: list[OuterPolicyState] = []
    for bundle in fitted.bundles:
        menus = tuple(
            row
            for row in development.effective_menus
            if row.outer_target_id == bundle.outer_target_id
        )
        outcomes = tuple(
            row
            for row in development.outcomes
            if row.action.outer_target_id == bundle.outer_target_id
        )
        admission = evaluate_outer_admission(
            bundle.lodo.oof_predictions,
            outcomes,
            config=admission_config,
            effective_menus=menus,
        )
        calibration = calibrate_selected_policy(
            bundle.lodo.oof_predictions,
            outcomes,
            config=risk_config,
            effective_menus=menus,
            nested_policy_folds=bundle.lodo.nested_policy_folds,
        )
        policies.append(OuterPolicyState(bundle.outer_target_id, admission, calibration))
    return RouterAdmissionState(tuple(policies))


def predict_target_evidence(
    menus: Sequence[EffectiveMenu], fitted: RouterFitState
) -> TargetEvidenceState:
    typed = tuple(menus)
    if not typed or any(
        not isinstance(row, EffectiveMenu) or row.query_center_id != row.outer_target_id
        for row in typed
    ):
        raise ProtocolError("HARP v9 target prediction requires sealed target menus.")
    predictions = tuple(
        predict_case(fitted.for_outer(menu.outer_target_id).lodo.final_model, menu)
        for menu in typed
    )
    return TargetEvidenceState(typed, predictions)


def model_manifest(state: RouterFitState) -> dict[str, object]:
    rows = []
    for bundle in state.bundles:
        model = bundle.lodo.final_model
        rows.append(
            {
                "outer_target_id": bundle.outer_target_id,
                "model_hash": model.model_hash,
                "source_lodo_hash": bundle.lodo.result_hash,
                "model": dict(model.public_payload()),
                "heldout_model_hashes": [
                    list(row) for row in bundle.lodo.heldout_model_hashes
                ],
                "numeric_oof": bundle.lodo.numeric_oof_payload(),
            }
        )
    body = {
        "schema_version": "midogpp_harp_v9_pairwise_residual_nested_lodo_model_v1",
        "outer_models": rows,
        "outer_model_count": len(rows),
        "shared_effective_menu_before_labels": True,
        "exact_B_virtual_tie_aware_control": True,
        "budget_residual": "U_MINUS_B",
        "allocation_residual": "HXE_MINUS_U",
        "case_center_balanced_pairwise_ranker": True,
        "cross_fitted_selected_action_acceptor": True,
        "strict_query_and_candidate_center_exclusion": True,
        "per_action_worst_center_certificate_used": False,
        "ranking_scope": "ALL_ACTIVE_PHYSICAL_ACTIONS_PLUS_VIRTUAL_B",
        "hyperparameters_selected_inside_source_lodo": False,
        "regularization_hyperparameters_predeclared_fixed_before_source_lodo": True,
        "numeric_source_oof_persisted": True,
        "evaluation_labels_used": False,
    }
    return {**body, "source_model_manifest_hash": canonical_hash(body)}


def admission_manifest(state: RouterAdmissionState) -> dict[str, object]:
    rows = [
        {
            "outer_target_id": value.outer_target_id,
            "policy_enabled": value.policy_enabled,
            "admission": dict(value.admission.public_payload()),
            "calibration": dict(value.calibration.public_payload()),
        }
        for value in state.by_outer
    ]
    body = {
        "schema_version": "midogpp_harp_v9_per_outer_selected_policy_admission_v1",
        "outer_policies": rows,
        "admitted_outer_count": sum(row.policy_enabled for row in state.by_outer),
        "global_kill_switch_used": False,
        "always_b_tie_aware_null": True,
        "raw_pairwise_rank_skill_checked_separately": True,
        "cross_fitted_selected_action_acceptance": True,
        "whole_policy_oof_risk_coverage": True,
        "whole_policy_surface": "ALL_HELD_SOURCE_CASES_ROUTE_OR_EXACT_B",
        "per_action_worst_center_certificate_used": False,
        "evaluation_labels_used": False,
    }
    return {**body, "source_policy_manifest_hash": canonical_hash(body)}


def target_evidence_manifest(state: TargetEvidenceState) -> dict[str, object]:
    body = {
        "schema_version": "midogpp_harp_v9_target_pairwise_residual_action_scores_v1",
        "case_count": len(state.menus),
        "active_case_count": sum(bool(menu.actions) for menu in state.menus),
        "prediction_rows": [dict(prediction.public_payload()) for prediction in state.predictions],
        "per_action_worst_center_certificate_used": False,
        "evaluation_labels_used": False,
    }
    return {**body, "target_evidence_manifest_hash": canonical_hash(body)}


__all__ = (
    "OuterPolicyState",
    "OuterRouterBundle",
    "RouterAdmissionState",
    "RouterFitState",
    "TargetEvidenceState",
    "admission_manifest",
    "build_source_only_admission",
    "fit_outer_routers",
    "model_manifest",
    "predict_target_evidence",
    "target_evidence_manifest",
)
