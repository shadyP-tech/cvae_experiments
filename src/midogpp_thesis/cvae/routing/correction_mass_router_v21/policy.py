"""Fit one pooled source policy and apply it label-free to every target H."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
from .crossfit import NestedCrossfitResult, nested_source_crossfit, validate_source_inventory
from .hashing import canonical_hash
from .fit_cache import ScopedFitCache, with_execution_feature_cache
from .decision_evidence import decision_evidence
from .candidate_prediction import unthresholded_winner
from .stacked_fitting import StackedScienceModel, choose_candidate, fit_stacked_science_model
from .records import RouteDecision, SelectedOOFRecord
from .truth import SupportTruthCapability, combine_truth_capabilities


@dataclass(frozen=True, slots=True)
class PooledRouterPolicy:
    model: StackedScienceModel
    crossfit: NestedCrossfitResult
    admission: SourceOnlyAdmission
    config: RouterFitConfig
    source_menu_hash: str
    truth_capability_hash: str
    policy_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.model, StackedScienceModel)
            or not isinstance(self.crossfit, NestedCrossfitResult)
            or not isinstance(self.admission, SourceOnlyAdmission)
            or not isinstance(self.config, RouterFitConfig)
            or self.model.opportunity_alpha != self.crossfit.final_opportunity_alpha
            or self.model.ranker_alpha != self.crossfit.final_ranker_alpha
            or self.model.evidence_variant != self.crossfit.final_evidence_variant
        ):
            raise ProtocolError("HARP v21 pooled router policy components drifted.")
        object.__setattr__(
            self,
            "policy_hash",
            canonical_hash(
                {
                    "schema_version": "correction_mass_router_v21",
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
        return bool(self.admission.admitted and self.crossfit.final_policy_enabled)

    def route_menu(self, menu: LabelFreeCaseMenu) -> RouteDecision:
        if (
            not isinstance(menu, LabelFreeCaseMenu)
            or menu.surface_role is not SurfaceRole.TARGET_EVALUATION
        ):
            raise ProtocolError("HARP v21 policy routes only label-free target-evaluation menus.")
        candidates = self.model.candidate_predictions(menu, self.config)
        gate = self.model.winner_prediction(menu, candidates)
        composite, score, reason = choose_candidate(
            menu, candidates, self.route_threshold,
            enabled=self.crossfit.final_policy_enabled,
            winner_prediction=gate,
        )
        if not self.admitted:
            composite = build_baseline_composite(menu)
            score = 0.0
            reason = (
                f"SOURCE_OOF_ADMISSION_{self.admission.status.value}"
                if not self.admission.admitted else "FINAL_REFIT_HAS_NO_NONZERO_POLICY"
            )
        return RouteDecision(
            composite=composite, requested_arm_id=self.selected_arm_id,
            route_score=score, route_threshold=self.route_threshold,
            policy_hash=self.policy_hash, admitted=self.admitted, fallback_reason=reason,
            **decision_evidence(unthresholded_winner(candidates), gate),
        )

    def public_payload(self) -> dict[str, object]:
        return {
            "schema_version": "correction_mass_router_v21",
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


@with_execution_feature_cache
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
        raise ProtocolError("HARP v21 fit_source_router requires a typed config.")
    capability = _normalize_capability(truth_capabilities)
    menu_rows = validate_source_inventory(menus, capability, config=fit_config)
    if (case_profiles is None) != (action_outcomes is None):
        raise ProtocolError(
            "HARP v21 precomputed profiles and outcomes must be supplied together."
        )
    # Legacy caller aggregates may be recorded by the runtime, but are never
    # fitting inputs. Every nested ranker derives its own scoped normalizer.
    profiles = () if case_profiles is None else tuple(case_profiles)
    outcomes = () if action_outcomes is None else tuple(action_outcomes)
    if profiles or outcomes:
        expected_keys = {(row.center_id, row.case_id) for row in menu_rows}
        if (
            any(not isinstance(row, SupportCaseClassProfile) for row in profiles)
            or any(not isinstance(row, SupportActionOutcome) for row in outcomes)
            or {(row.center_id, row.case_id) for row in profiles} != expected_keys
            or {row.action.action_hash for row in outcomes} != {action.action_hash for row in menu_rows for action in row.actions}
        ):
            raise ProtocolError("HARP v21 audit-only source aggregates are incomplete or drifted.")
    cache = ScopedFitCache()
    crossfit = nested_source_crossfit(
        menu_rows,
        profiles,
        outcomes,
        capability,
        config=fit_config, cache=cache,
    )
    admission = build_source_only_admission(crossfit.records, config=fit_config)
    model = fit_stacked_science_model(menu_rows, capability, config=replace(fit_config, evidence_variant=crossfit.final_evidence_variant), cache=cache)
    source_menu_hash = canonical_hash(
        {
            "schema_version": "pooled_pairwise_source_menu_surface_v21",
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
        raise ProtocolError("HARP v21 target routing requires a pooled policy.")
    rows = tuple(sorted(menus, key=lambda row: (row.center_id, row.case_id)))
    keys = tuple((row.center_id, row.case_id) for row in rows)
    if (
        not rows
        or len(keys) != len(set(keys))
        or any(row.surface_role is not SurfaceRole.TARGET_EVALUATION for row in rows)
    ):
        raise ProtocolError("HARP v21 target menu inventory is malformed.")
    return tuple(policy.route_menu(menu) for menu in rows)


def route_decision_report(decisions: Sequence[RouteDecision]) -> dict[str, object]:
    rows = tuple(decisions)
    if not rows:
        raise ProtocolError("HARP v21 route report cannot be empty.")
    route_selected = sum(row.route_selected for row in rows)
    probability_changed = sum(row.probability_changed for row in rows)
    prediction_changed = sum(row.prediction_changed for row in rows)
    entropies = tuple(row.donor_entropy for row in rows if row.route_selected)
    payload = {
        "schema_version": "pooled_pairwise_route_report_v21",
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
