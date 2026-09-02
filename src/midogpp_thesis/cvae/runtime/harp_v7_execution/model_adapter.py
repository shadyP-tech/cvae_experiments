"""Source-active nested-LODO model and whole-policy adapter for HARP v7."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ...routing.source_active_selective_router_v7 import (
    AdmissionConfig,
    CasePrediction,
    EffectiveMenu,
    FitConfig,
    OuterAdmission,
    RiskCoverageConfig,
    SelectiveCalibration,
    SourceActionOutcome,
    SourceLODOResult,
    calibrate_policy_risk_coverage,
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
            raise ProtocolError("HARP v7 outer model crossed its held target.")


@dataclass(frozen=True, slots=True)
class RouterFitState:
    bundles: tuple[OuterRouterBundle, ...]

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.bundles, key=lambda row: row.outer_target_id))
        if not ordered or len({row.outer_target_id for row in ordered}) != len(ordered):
            raise ProtocolError("HARP v7 fitted outer-router inventory is malformed.")
        object.__setattr__(self, "bundles", ordered)

    def for_outer(self, outer_target_id: str) -> OuterRouterBundle:
        for bundle in self.bundles:
            if bundle.outer_target_id == str(outer_target_id):
                return bundle
        raise ProtocolError("HARP v7 fitted router lacks an outer target.")


@dataclass(frozen=True, slots=True)
class OuterPolicyState:
    outer_target_id: str
    admission: OuterAdmission
    calibration: SelectiveCalibration

    def __post_init__(self) -> None:
        if (
            self.admission.outer_target_id != self.outer_target_id
            or self.calibration.outer_target_id != self.outer_target_id
        ):
            raise ProtocolError("HARP v7 local policy crossed its outer target.")

    @property
    def policy_enabled(self) -> bool:
        return self.admission.admitted and self.calibration.calibrated


@dataclass(frozen=True, slots=True)
class RouterAdmissionState:
    by_outer: tuple[OuterPolicyState, ...]

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.by_outer, key=lambda row: row.outer_target_id))
        if not ordered or len({row.outer_target_id for row in ordered}) != len(ordered):
            raise ProtocolError("HARP v7 per-outer policy inventory is malformed.")
        object.__setattr__(self, "by_outer", ordered)

    def for_outer(self, outer_target_id: str) -> OuterPolicyState:
        for value in self.by_outer:
            if value.outer_target_id == str(outer_target_id):
                return value
        raise ProtocolError("HARP v7 local policy lacks an outer target.")


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
            raise ProtocolError("HARP v7 target evidence inventory drifted.")
        by_key = {(row.outer_target_id, row.case_id): row for row in menus}
        if any(
            row.menu_hash != by_key[(row.outer_target_id, row.case_id)].menu_hash
            for row in predictions
        ):
            raise ProtocolError("HARP v7 target prediction/menu hash drifted.")
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
            raise ProtocolError("HARP v7 target case evidence is absent.")
        return menu, prediction


def _fit_grid(model_config: Mapping[str, object]) -> tuple[FitConfig, ...]:
    try:
        opportunity = tuple(float(value) for value in model_config["opportunity_alpha_grid"])
        rank = tuple(float(value) for value in model_config["rank_alpha_grid"])
        gain = tuple(float(value) for value in model_config["min_opportunity_gain_grid"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("HARP v7 nested source fit grid is malformed.") from exc
    grid = tuple(
        FitConfig(
            opportunity_alpha=opportunity_alpha,
            rank_alpha=rank_alpha,
            min_opportunity_gain=min_gain,
        )
        for opportunity_alpha, rank_alpha, min_gain in product(opportunity, rank, gain)
    )
    if not grid:
        raise ProtocolError("HARP v7 nested source fit grid is empty.")
    return grid


def _fit_outer_task(
    payload: tuple[
        str,
        tuple[EffectiveMenu, ...],
        tuple[SourceActionOutcome, ...],
        tuple[FitConfig, ...],
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
        raise ProtocolError("HARP v7 fitting requires a typed development state.")
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
        raise ProtocolError("HARP v7 admission config is absent.")
    return AdmissionConfig(
        min_pooled_top1_excess=float(raw["min_pooled_top1_excess_over_always_b"]),
        min_delete_center_top1_excess=float(
            raw["min_delete_center_top1_excess_over_always_b"]
        ),
        min_opportunity_top1_accuracy=float(raw["min_opportunity_top1_accuracy"]),
        min_opportunity_cases=int(raw["min_opportunity_cases"]),
    )


def _risk_config(model_config: Mapping[str, object]) -> RiskCoverageConfig:
    raw = model_config.get("policy")
    if not isinstance(raw, Mapping):
        raise ProtocolError("HARP v7 risk-coverage config is absent.")
    return RiskCoverageConfig(
        opportunity_thresholds=tuple(
            float(value)
            for value in model_config["opportunity_probability_threshold_grid"]
        ),
        rank_margin_thresholds=tuple(
            float(value) for value in model_config["rank_margin_threshold_grid"]
        ),
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
        raise ProtocolError("HARP v7 policy admission requires fitted source state.")
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
        calibration = calibrate_policy_risk_coverage(
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
        raise ProtocolError("HARP v7 target prediction requires sealed target menus.")
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
                "training_center_ids": list(model.training_center_ids),
                "action_feature_names": list(model.action_standardizer.names),
                "action_mean": list(model.action_standardizer.mean),
                "action_scale": list(model.action_standardizer.scale),
                "case_feature_names": list(model.case_standardizer.names),
                "case_mean": list(model.case_standardizer.mean),
                "case_scale": list(model.case_standardizer.scale),
                "opportunity_head": {
                    "intercept": model.opportunity_head.intercept,
                    "coefficients": list(model.opportunity_head.coefficients),
                },
                "d01_rank_head": {
                    "intercept": model.d01_rank_head.intercept,
                    "coefficients": list(model.d01_rank_head.coefficients),
                    "available": model.d01_rank_head.available,
                },
                "d10_rank_head": {
                    "intercept": model.d10_rank_head.intercept,
                    "coefficients": list(model.d10_rank_head.coefficients),
                    "available": model.d10_rank_head.available,
                },
                "heldout_model_hashes": [list(row) for row in bundle.lodo.heldout_model_hashes],
                "numeric_oof": bundle.lodo.numeric_oof_payload(),
            }
        )
    body = {
        "schema_version": "midogpp_harp_v7_source_active_nested_lodo_model_v1",
        "outer_models": rows,
        "outer_model_count": len(rows),
        "shared_effective_menu_before_labels": True,
        "case_opportunity_hurdle": True,
        "conditional_direction_ranker": True,
        "hyperparameters_selected_inside_source_lodo": True,
        "numeric_source_oof_persisted": True,
        "evaluation_labels_used": False,
    }
    return {**body, "source_model_manifest_hash": canonical_hash(body)}


def admission_manifest(state: RouterAdmissionState) -> dict[str, object]:
    rows = []
    for value in state.by_outer:
        admission = value.admission
        calibration = value.calibration
        rows.append(
            {
                "outer_target_id": value.outer_target_id,
                "policy_enabled": value.policy_enabled,
                "admission": {
                    "admitted": admission.admitted,
                    "learned_top1_accuracy": admission.learned_top1_accuracy,
                    "always_b_top1_accuracy": admission.always_b_top1_accuracy,
                    "pooled_top1_excess": admission.pooled_top1_excess,
                    "min_delete_center_top1_excess": admission.min_delete_center_top1_excess,
                    "opportunity_top1_accuracy": admission.opportunity_top1_accuracy,
                    "opportunity_case_count": admission.opportunity_case_count,
                    "case_count": admission.case_count,
                    "reasons": list(admission.reasons),
                    "admission_hash": admission.admission_hash,
                },
                "calibration": {
                    "calibrated": calibration.calibrated,
                    "opportunity_threshold": calibration.opportunity_threshold,
                    "rank_margin_threshold": calibration.rank_margin_threshold,
                    "selected_replay_hash": calibration.selected_replay.replay_hash,
                    "calibration_hash": calibration.calibration_hash,
                    "frontier": [
                        {
                            "opportunity_threshold": row.opportunity_threshold,
                            "rank_margin_threshold": row.rank_margin_threshold,
                            "routed_cases": row.routed_cases,
                            "case_count": row.case_count,
                            "coverage": row.coverage,
                            "case_equal_bacc_gain": row.case_equal_bacc_gain,
                            "min_delete_center_bacc_gain": row.min_delete_center_bacc_gain,
                            "case_equal_brier_delta": row.case_equal_brier_delta,
                            "case_equal_log_delta": row.case_equal_log_delta,
                            "routed_harm_rate": row.routed_harm_rate,
                            "safe": row.safe,
                            "replay_hash": row.replay_hash,
                        }
                        for row in calibration.frontier
                    ],
                },
            }
        )
    body = {
        "schema_version": "midogpp_harp_v7_per_outer_whole_policy_admission_v1",
        "outer_policies": rows,
        "admitted_outer_count": sum(row.policy_enabled for row in state.by_outer),
        "global_kill_switch_used": False,
        "always_b_tie_aware_null": True,
        "conditional_rank_skill_surface": "POSITIVE_OPPORTUNITY_CASES_ONLY",
        "whole_policy_oof_risk_coverage": True,
        "whole_policy_surface": "ALL_HELD_SOURCE_CASES_ROUTE_OR_EXACT_B",
        "evaluation_labels_used": False,
    }
    return {**body, "source_policy_manifest_hash": canonical_hash(body)}


def target_evidence_manifest(state: TargetEvidenceState) -> dict[str, object]:
    body = {
        "schema_version": "midogpp_harp_v7_target_case_predictions_v1",
        "case_count": len(state.menus),
        "active_case_count": sum(bool(menu.actions) for menu in state.menus),
        "rows": [
            {
                "outer_target_id": prediction.outer_target_id,
                "case_id": prediction.case_id,
                "menu_hash": prediction.menu_hash,
                "prediction_hash": prediction.prediction_hash,
                "opportunity_probability": prediction.opportunity_probability,
                "rank_margin": prediction.rank_margin,
                "top_action_id": prediction.top_action_id,
                "action_scores": [
                    {
                        "action_id": score.action_id,
                        "action_hash": score.action_hash,
                        "direction": score.direction.value,
                        "score": score.score,
                    }
                    for score in prediction.action_scores
                ],
            }
            for prediction in state.predictions
        ],
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
