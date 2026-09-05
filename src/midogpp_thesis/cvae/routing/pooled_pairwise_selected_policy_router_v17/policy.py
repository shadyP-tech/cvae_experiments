"""Fit one pooled source policy and apply it label-free to every target H."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

from ...protocol import ProtocolError
from .admission import SourceOnlyAdmission, build_source_only_admission
from .composition import build_baseline_composite
from .contracts import (
    CompositeKind,
    Direction,
    LabelFreeCaseMenu,
    RouterFitConfig,
    SupportActionOutcome,
    SupportCaseClassProfile,
    SurfaceRole,
)
from .crossfit import (
    NestedCrossfitResult,
    _IneligibleArm,
    _build_composite,
    nested_source_crossfit,
    validate_source_inventory,
)
from .hashing import canonical_hash
from .modeling import PooledScienceModel, fit_pooled_science_model
from .records import RouteDecision, SelectedOOFRecord
from .truth import SupportTruthCapability, combine_truth_capabilities


@dataclass(frozen=True, slots=True)
class PooledRouterPolicy:
    model: PooledScienceModel
    crossfit: NestedCrossfitResult
    admission: SourceOnlyAdmission
    config: RouterFitConfig
    source_menu_hash: str
    truth_capability_hash: str
    policy_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.model, PooledScienceModel)
            or not isinstance(self.crossfit, NestedCrossfitResult)
            or not isinstance(self.admission, SourceOnlyAdmission)
            or not isinstance(self.config, RouterFitConfig)
            or self.model.opportunity_alpha != self.crossfit.final_opportunity_alpha
            or self.model.ranker_alpha != self.crossfit.final_ranker_alpha
        ):
            raise ProtocolError("HARP v17 pooled router policy components drifted.")
        object.__setattr__(
            self,
            "policy_hash",
            canonical_hash(
                {
                    "schema_version": "pooled_pairwise_selected_policy_router_v17",
                    "model_hash": self.model.model_hash,
                    "crossfit_hash": self.crossfit.result_hash,
                    "admission_hash": self.admission.admission_hash,
                    "config": self.config.public_payload(),
                    "source_menu_hash": self.source_menu_hash,
                    "truth_capability_hash": self.truth_capability_hash,
                    "selected_arm": self.crossfit.final_arm.public_payload(),
                    "route_threshold": self.crossfit.final_route_threshold,
                    "pooled_known_center_policy_count": 1,
                    "nested_oof_evaluates_selection_algorithm_not_final_refit": True,
                    "target_evaluation_labels_consumed": False,
                }
            ),
        )

    @property
    def model_hash(self) -> str:
        return self.model.model_hash

    @property
    def selected_arm_id(self) -> str:
        return self.crossfit.final_arm.arm_id

    @property
    def route_threshold(self) -> float:
        return self.crossfit.final_route_threshold

    @property
    def oof_records(self) -> tuple[SelectedOOFRecord, ...]:
        return self.crossfit.records

    @property
    def training_case_keys(self) -> tuple[tuple[str, str], ...]:
        return self.model.training_case_keys

    @property
    def admitted(self) -> bool:
        return self.admission.admitted

    def route_menu(self, menu: LabelFreeCaseMenu) -> RouteDecision:
        if (
            not isinstance(menu, LabelFreeCaseMenu)
            or menu.surface_role is not SurfaceRole.TARGET_EVALUATION
        ):
            raise ProtocolError("HARP v17 policy routes only label-free target-evaluation menus.")
        prediction = self.model.predict_menu(menu)
        reason: str | None = None
        if not self.admitted:
            composite = build_baseline_composite(menu)
            reason = f"SOURCE_OOF_ADMISSION_{self.admission.status.value}"
        elif self.crossfit.final_arm.kind is CompositeKind.B:
            composite = build_baseline_composite(menu)
            reason = "FINAL_MODAL_ARM_B"
        else:
            try:
                composite, reason = _build_composite(
                    menu,
                    prediction,
                    self.crossfit.final_arm,
                    self.crossfit.final_route_threshold,
                )
            except _IneligibleArm:
                composite = build_baseline_composite(menu)
                reason = "FEWER_THAN_K_ELIGIBLE"
        return RouteDecision(
            composite=composite,
            requested_arm_id=self.crossfit.final_arm.arm_id,
            route_score=prediction.route_score_for(self.crossfit.final_arm.kind),
            route_threshold=self.crossfit.final_route_threshold,
            policy_hash=self.policy_hash,
            admitted=self.admitted,
            fallback_reason=reason,
        )

    def public_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pooled_pairwise_selected_policy_router_v17",
            "policy_hash": self.policy_hash,
            "model_hash": self.model_hash,
            "model": self.model.public_payload(),
            "crossfit": self.crossfit.public_payload(),
            "admission": self.admission.public_payload(),
            "config": self.config.public_payload(),
            "source_menu_hash": self.source_menu_hash,
            "truth_capability_hash": self.truth_capability_hash,
            "selected_arm_id": self.selected_arm_id,
            "route_threshold": self.route_threshold,
            "training_case_keys": [list(value) for value in self.training_case_keys],
            "source_case_count": len(self.training_case_keys),
            "source_center_count": len({center for center, _ in self.training_case_keys}),
            "pooled_known_center_policy_count": 1,
            "nested_oof_evaluates_selection_algorithm_not_final_refit": True,
            "raw_labels_persisted": False,
            "target_evaluation_labels_consumed": False,
        }


def _normalize_capability(
    value: SupportTruthCapability | Sequence[SupportTruthCapability],
) -> SupportTruthCapability:
    if isinstance(value, SupportTruthCapability):
        return value
    return combine_truth_capabilities(tuple(value))


def fit_source_router(
    menus: Sequence[LabelFreeCaseMenu],
    truth_capabilities: SupportTruthCapability | Sequence[SupportTruthCapability],
    *,
    config: RouterFitConfig | None = None,
    case_profiles: Sequence[SupportCaseClassProfile] | None = None,
    action_outcomes: Sequence[SupportActionOutcome] | None = None,
) -> PooledRouterPolicy:
    """Fit exactly one pooled known-center policy from source-train cases."""

    fit_config = RouterFitConfig() if config is None else config
    if not isinstance(fit_config, RouterFitConfig):
        raise ProtocolError("HARP v17 fit_source_router requires a typed config.")
    capability = _normalize_capability(truth_capabilities)
    menu_rows = validate_source_inventory(menus, capability, config=fit_config)
    if (case_profiles is None) != (action_outcomes is None):
        raise ProtocolError(
            "HARP v17 precomputed profiles and outcomes must be supplied together."
        )
    if case_profiles is None or action_outcomes is None:
        profiles, outcomes = capability.derive_training_surface(menu_rows)
    else:
        if any(
            not isinstance(row, SupportCaseClassProfile) for row in case_profiles
        ) or any(not isinstance(row, SupportActionOutcome) for row in action_outcomes):
            raise ProtocolError("HARP v17 precomputed source outcomes are untyped.")
        profiles = tuple(
            sorted(case_profiles, key=lambda row: (row.center_id, row.case_id))
        )
        outcomes = tuple(
            sorted(
                action_outcomes,
                key=lambda row: (
                    row.action.center_id,
                    row.action.case_id,
                    row.action.arm_id,
                ),
            )
        )
        expected_keys = tuple((row.center_id, row.case_id) for row in menu_rows)
        expected_actions = {
            action.action_hash: menu.menu_hash
            for menu in menu_rows
            for action in menu.actions
        }
        observed_actions = {
            row.action.action_hash: row.menu_hash for row in outcomes
        }
        if (
            tuple((row.center_id, row.case_id) for row in profiles) != expected_keys
            or len(observed_actions) != len(outcomes)
            or observed_actions != expected_actions
        ):
            raise ProtocolError(
                "HARP v17 precomputed source outcome universe is incomplete or drifted."
            )
    crossfit = nested_source_crossfit(
        menu_rows,
        profiles,
        outcomes,
        capability,
        config=fit_config,
    )
    admission = build_source_only_admission(crossfit.records, config=fit_config)
    model = fit_pooled_science_model(
        menu_rows,
        profiles,
        outcomes,
        opportunity_alpha=crossfit.final_opportunity_alpha,
        ranker_alpha=crossfit.final_ranker_alpha,
        maximum_numeric_features=fit_config.maximum_numeric_features,
    )
    source_menu_hash = canonical_hash(
        {
            "schema_version": "pooled_pairwise_source_menu_surface_v17",
            "menu_hashes": tuple(row.menu_hash for row in menu_rows),
            "case_keys": tuple((row.center_id, row.case_id) for row in menu_rows),
            "pooled_policy_count": 1,
        }
    )
    return PooledRouterPolicy(
        model=model,
        crossfit=crossfit,
        admission=admission,
        config=fit_config,
        source_menu_hash=source_menu_hash,
        truth_capability_hash=capability.capability_hash,
    )


def route_target_cases(
    policy: PooledRouterPolicy,
    menus: Sequence[LabelFreeCaseMenu],
) -> tuple[RouteDecision, ...]:
    if not isinstance(policy, PooledRouterPolicy):
        raise ProtocolError("HARP v17 target routing requires a pooled policy.")
    rows = tuple(sorted(menus, key=lambda row: (row.center_id, row.case_id)))
    keys = tuple((row.center_id, row.case_id) for row in rows)
    if (
        not rows
        or len(keys) != len(set(keys))
        or any(row.surface_role is not SurfaceRole.TARGET_EVALUATION for row in rows)
    ):
        raise ProtocolError("HARP v17 target menu inventory is malformed.")
    return tuple(policy.route_menu(menu) for menu in rows)


def route_decision_report(decisions: Sequence[RouteDecision]) -> dict[str, object]:
    rows = tuple(decisions)
    if not rows:
        raise ProtocolError("HARP v17 route report cannot be empty.")
    route_selected = sum(row.route_selected for row in rows)
    probability_changed = sum(row.probability_changed for row in rows)
    prediction_changed = sum(row.prediction_changed for row in rows)
    entropies = tuple(row.donor_entropy for row in rows if row.route_selected)
    payload = {
        "schema_version": "pooled_pairwise_route_report_v17",
        "case_count": len(rows),
        "route_selected_count": route_selected,
        "probability_changed_count": probability_changed,
        "prediction_changed_count": prediction_changed,
        "utility_success_count": None,
        "mean_selected_donor_entropy": (
            0.0 if not entropies else float(sum(entropies) / len(entropies))
        ),
        "route_selected_is_probability_changed": False,
        "probability_changed_is_prediction_changed": False,
        "target_utility_opened": False,
        "decision_hashes": [row.decision_hash for row in rows],
    }
    return {**payload, "report_hash": canonical_hash(payload)}


__all__ = (
    "PooledRouterPolicy",
    "fit_source_router",
    "route_decision_report",
    "route_target_cases",
)
