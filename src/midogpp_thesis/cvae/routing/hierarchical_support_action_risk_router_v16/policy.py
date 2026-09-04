"""Whole-policy admission and exact-action routing for HARP v16."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

from ...protocol import ProtocolError
from .certificates import (
    ActionRiskCertificate,
    MenuRiskCalibration,
    certify_case_prediction,
    fit_menu_risk_calibration,
)
from .contracts import (
    ActionFamily,
    CasePrediction,
    Direction,
    LabelFreeCaseMenu,
    RouterFitConfig,
    SupportActionOutcome,
    SupportCaseClassProfile,
    SurfaceRole,
    canonical_probability_hex,
)
from .crossfit import (
    SupportCrossfitResult,
    SupportOOFRecord,
    leave_one_case_out_crossfit,
    validate_support_inventory,
)
from .hashing import canonical_hash, require_sha256
from .hierarchical import (
    NullSupportEndpointModel,
    SupportEndpointModel,
    fit_support_endpoint_model,
)
from .outcome_normalization import (
    fit_support_fold_normalizer,
    validate_support_case_profiles,
)


@dataclass(frozen=True, slots=True)
class HierarchyTrace:
    eligible_action_ids: tuple[str, ...]
    selected_direction: Direction | None
    selected_family: ActionFamily | None
    selected_action_id: str | None
    trace_hash: str = field(init=False)

    def __post_init__(self) -> None:
        eligible = tuple(sorted(self.eligible_action_ids))
        if self.selected_action_id is None:
            if self.selected_direction is not None or self.selected_family is not None:
                raise ProtocolError("HARP v16 empty hierarchy trace claims a child stage.")
        elif (
            self.selected_action_id not in eligible
            or not isinstance(self.selected_direction, Direction)
            or not isinstance(self.selected_family, ActionFamily)
        ):
            raise ProtocolError("HARP v16 hierarchy trace is malformed.")
        object.__setattr__(self, "eligible_action_ids", eligible)
        object.__setattr__(
            self,
            "trace_hash",
            canonical_hash(
                {
                    "schema_version": "hierarchical_support_selection_trace_v16",
                    "eligible_action_ids": eligible,
                    "selected_direction": (
                        None if self.selected_direction is None else self.selected_direction.value
                    ),
                    "selected_family": (
                        None if self.selected_family is None else self.selected_family.value
                    ),
                    "selected_action_id": self.selected_action_id,
                    "stages": ("B_VS_ROUTE", "DIRECTION", "FAMILY", "EXPERT"),
                }
            ),
        )

    def public_payload(self) -> dict[str, object]:
        return {
            "eligible_action_ids": list(self.eligible_action_ids),
            "selected_direction": (
                None if self.selected_direction is None else self.selected_direction.value
            ),
            "selected_family": (
                None if self.selected_family is None else self.selected_family.value
            ),
            "selected_action_id": self.selected_action_id,
            "trace_hash": self.trace_hash,
            "stages": ["B_VS_ROUTE", "DIRECTION", "FAMILY", "EXPERT"],
        }


@dataclass(frozen=True, slots=True)
class PolicyAdmission:
    outer_target_id: str
    admitted: bool
    support_case_count: int
    routed_case_count: int
    coverage: float
    case_equal_bacc_gain: float
    routed_harm_rate: float
    case_equal_brier_delta: float
    case_equal_log_loss_delta: float
    support_crossfit_hash: str
    heldout_calibration_hashes: tuple[tuple[str, str], ...]
    policy_config_hash: str
    admission_hash: str = field(init=False)

    def __post_init__(self) -> None:
        values = (
            self.coverage,
            self.case_equal_bacc_gain,
            self.routed_harm_rate,
            self.case_equal_brier_delta,
            self.case_equal_log_loss_delta,
        )
        if (
            self.support_case_count < 1
            or not 0 <= self.routed_case_count <= self.support_case_count
            or any(not math.isfinite(value) for value in values)
            or not 0.0 <= self.coverage <= 1.0
            or not 0.0 <= self.routed_harm_rate <= 1.0
        ):
            raise ProtocolError("HARP v16 whole-policy admission is malformed.")
        support_hash = require_sha256(
            self.support_crossfit_hash, name="admission support crossfit hash"
        )
        calibration_hashes = tuple(sorted(self.heldout_calibration_hashes))
        if (
            len(calibration_hashes) != self.support_case_count
            or len({case for case, _ in calibration_hashes}) != self.support_case_count
        ):
            raise ProtocolError(
                "HARP v16 admission lacks one nested calibration per support case."
            )
        for _, value in calibration_hashes:
            require_sha256(value, name="heldout admission calibration hash")
        config_hash = require_sha256(
            self.policy_config_hash, name="admission policy config hash"
        )
        object.__setattr__(self, "support_crossfit_hash", support_hash)
        object.__setattr__(self, "heldout_calibration_hashes", calibration_hashes)
        object.__setattr__(self, "policy_config_hash", config_hash)
        object.__setattr__(
            self,
            "admission_hash",
            canonical_hash(
                {
                    "schema_version": "hierarchical_support_policy_admission_v16",
                    "outer_target_id": self.outer_target_id,
                    "admitted": bool(self.admitted),
                    "support_case_count": self.support_case_count,
                    "routed_case_count": self.routed_case_count,
                    "coverage": self.coverage,
                    "case_equal_bacc_gain": self.case_equal_bacc_gain,
                    "routed_harm_rate": self.routed_harm_rate,
                    "case_equal_brier_delta": self.case_equal_brier_delta,
                    "case_equal_log_loss_delta": self.case_equal_log_loss_delta,
                    "support_crossfit_hash": support_hash,
                    "heldout_calibration_hashes": calibration_hashes,
                    "policy_config_hash": config_hash,
                    "support_oof_only": True,
                    "heldout_case_excluded_from_model_and_calibration": True,
                    "evaluation_labels_consumed": False,
                }
            ),
        )

    def public_payload(self) -> dict[str, object]:
        return {
            "outer_target_id": self.outer_target_id,
            "admitted": self.admitted,
            "support_case_count": self.support_case_count,
            "routed_case_count": self.routed_case_count,
            "coverage": self.coverage,
            "case_equal_bacc_gain": self.case_equal_bacc_gain,
            "routed_harm_rate": self.routed_harm_rate,
            "case_equal_brier_delta": self.case_equal_brier_delta,
            "case_equal_log_loss_delta": self.case_equal_log_loss_delta,
            "support_crossfit_hash": self.support_crossfit_hash,
            "heldout_calibration_hashes": [
                {"case_id": case, "calibration_hash": value}
                for case, value in self.heldout_calibration_hashes
            ],
            "policy_config_hash": self.policy_config_hash,
            "admission_hash": self.admission_hash,
            "support_oof_only": True,
            "heldout_case_excluded_from_model_and_calibration": True,
            "evaluation_labels_consumed": False,
        }


@dataclass(frozen=True, slots=True)
class RouteDecision:
    outer_target_id: str
    case_id: str
    selected_action_id: str
    probability_hex: tuple[str, ...]
    exact_b_fallback: bool
    reason: str
    menu_hash: str
    prediction_hash: str
    calibration_hash: str
    admission_hash: str
    router_hash: str
    hierarchy_trace: HierarchyTrace
    selected_certificate_hash: str | None
    route_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.exact_b_fallback != (self.selected_action_id == "B"):
            raise ProtocolError("HARP v16 route fallback semantics are malformed.")
        if self.exact_b_fallback != (self.selected_certificate_hash is None):
            raise ProtocolError("HARP v16 fallback/certificate semantics disagree.")
        probability = canonical_probability_hex(self.probability_hex)
        menu_hash = require_sha256(self.menu_hash, name="route menu hash")
        prediction_hash = require_sha256(
            self.prediction_hash, name="route prediction hash"
        )
        calibration_hash = require_sha256(
            self.calibration_hash, name="route calibration hash"
        )
        admission_hash = require_sha256(
            self.admission_hash, name="route admission hash"
        )
        router_hash = require_sha256(self.router_hash, name="route router hash")
        certificate_hash = (
            None
            if self.selected_certificate_hash is None
            else require_sha256(
                self.selected_certificate_hash, name="selected certificate hash"
            )
        )
        object.__setattr__(self, "probability_hex", probability)
        object.__setattr__(self, "menu_hash", menu_hash)
        object.__setattr__(self, "prediction_hash", prediction_hash)
        object.__setattr__(self, "calibration_hash", calibration_hash)
        object.__setattr__(self, "admission_hash", admission_hash)
        object.__setattr__(self, "router_hash", router_hash)
        object.__setattr__(self, "selected_certificate_hash", certificate_hash)
        object.__setattr__(
            self,
            "route_hash",
            canonical_hash(
                {
                    "schema_version": "hierarchical_support_exact_action_route_v16",
                    "outer_target_id": self.outer_target_id,
                    "case_id": self.case_id,
                    "selected_action_id": self.selected_action_id,
                    "probability_hex": probability,
                    "exact_b_fallback": self.exact_b_fallback,
                    "reason": self.reason,
                    "menu_hash": menu_hash,
                    "prediction_hash": prediction_hash,
                    "calibration_hash": calibration_hash,
                    "admission_hash": admission_hash,
                    "router_hash": router_hash,
                    "hierarchy_trace_hash": self.hierarchy_trace.trace_hash,
                    "selected_certificate_hash": certificate_hash,
                    "probability_blending_used": False,
                    "evaluation_labels_consumed": False,
                }
            ),
        )

    def public_payload(self) -> dict[str, object]:
        return {
            "outer_target_id": self.outer_target_id,
            "case_id": self.case_id,
            "selected_action_id": self.selected_action_id,
            "probability_hex": list(self.probability_hex),
            "exact_b_fallback": self.exact_b_fallback,
            "reason": self.reason,
            "menu_hash": self.menu_hash,
            "prediction_hash": self.prediction_hash,
            "calibration_hash": self.calibration_hash,
            "admission_hash": self.admission_hash,
            "router_hash": self.router_hash,
            "hierarchy_trace": self.hierarchy_trace.public_payload(),
            "selected_certificate_hash": self.selected_certificate_hash,
            "route_hash": self.route_hash,
            "probability_blending_used": False,
            "evaluation_labels_consumed": False,
        }


@dataclass(frozen=True, slots=True)
class FittedSupportRouter:
    outer_target_id: str
    support_case_ids: tuple[str, ...]
    endpoint_model: SupportEndpointModel | NullSupportEndpointModel
    risk_calibration: MenuRiskCalibration
    admission: PolicyAdmission
    support_crossfit: SupportCrossfitResult
    config: RouterFitConfig
    router_hash: str = field(init=False)

    def __post_init__(self) -> None:
        cases = tuple(sorted(self.support_case_ids))
        if (
            not cases
            or cases != self.endpoint_model.training_case_ids
            or cases != self.risk_calibration.support_case_ids
            or cases != self.support_crossfit.case_ids
            or any(
                value != self.outer_target_id
                for value in (
                    self.endpoint_model.outer_target_id,
                    self.risk_calibration.outer_target_id,
                    self.admission.outer_target_id,
                    self.support_crossfit.outer_target_id,
                )
            )
            or self.admission.policy_config_hash != canonical_hash(self.config)
        ):
            raise ProtocolError("HARP v16 fitted router crossed a support boundary.")
        object.__setattr__(self, "support_case_ids", cases)
        object.__setattr__(
            self,
            "router_hash",
            canonical_hash(
                {
                    "schema_version": "hierarchical_support_action_risk_router_v16",
                    "outer_target_id": self.outer_target_id,
                    "support_case_ids": cases,
                    "endpoint_model_hash": self.endpoint_model.model_hash,
                    "risk_calibration_hash": self.risk_calibration.calibration_hash,
                    "admission_hash": self.admission.admission_hash,
                    "support_crossfit_hash": self.support_crossfit.result_hash,
                    "config": self.config,
                    "estimand": "known_center_train_support_to_full_test_case_routing",
                    "publication_status": "POST_HOC_CONSUMED_TEST_SENSITIVITY",
                    "terminal_decision": "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE",
                    "evaluation_labels_consumed": False,
                }
            ),
        )

    def route(self, menu: LabelFreeCaseMenu) -> RouteDecision:
        if (
            not isinstance(menu, LabelFreeCaseMenu)
            or menu.surface_role is not SurfaceRole.TARGET_EVALUATION
            or menu.outer_target_id != self.outer_target_id
            or menu.case_id in self.support_case_ids
        ):
            raise ProtocolError(
                "HARP v16 target routing crossed support/evaluation roles or cases."
            )
        prediction = self.endpoint_model.predict_menu(menu, out_of_fold=False)
        certificates = certify_case_prediction(
            prediction,
            self.risk_calibration,
            config=self.config,
        )
        selected, trace = select_hierarchical_certificate(certificates)
        if not menu.actions:
            return _fallback(self, menu, prediction, trace, "EXACT_B_NO_ACTIVE_ACTION")
        if self.endpoint_model.is_null:
            return _fallback(
                self, menu, prediction, trace, "EXACT_B_NULL_SUPPORT_MODEL"
            )
        if not self.admission.admitted:
            return _fallback(
                self, menu, prediction, trace, "EXACT_B_SUPPORT_POLICY_NOT_ADMITTED"
            )
        if selected is None:
            return _fallback(
                self, menu, prediction, trace, "EXACT_B_NO_ACTION_PASSED_DIRECT_CERTIFICATE"
            )
        action = menu.action_for(selected.prediction.action.action_id)
        if action is None or action.action_hash != selected.prediction.action.action_hash:
            raise ProtocolError("HARP v16 selected action drifted from the target menu.")
        return RouteDecision(
            outer_target_id=menu.outer_target_id,
            case_id=menu.case_id,
            selected_action_id=action.action_id,
            probability_hex=action.action_probability_hex,
            exact_b_fallback=False,
            reason="ROUTED_SUPPORT_CERTIFIED_HIERARCHICAL_EXACT_ACTION",
            menu_hash=menu.menu_hash,
            prediction_hash=prediction.prediction_hash,
            calibration_hash=self.risk_calibration.calibration_hash,
            admission_hash=self.admission.admission_hash,
            router_hash=self.router_hash,
            hierarchy_trace=trace,
            selected_certificate_hash=selected.certificate_hash,
        )

    def public_payload(self) -> dict[str, object]:
        return {
            "schema_version": "hierarchical_support_action_risk_router_v16",
            "outer_target_id": self.outer_target_id,
            "support_case_ids": list(self.support_case_ids),
            "endpoint_model": self.endpoint_model.public_payload(),
            "risk_calibration": self.risk_calibration.public_payload(),
            "admission": self.admission.public_payload(),
            "support_crossfit": self.support_crossfit.public_payload(),
            "config": self.config.public_payload(),
            "router_hash": self.router_hash,
            "publication_status": "POST_HOC_CONSUMED_TEST_SENSITIVITY",
            "terminal_decision": "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE",
            "evaluation_labels_consumed": False,
        }


def _stage_sort_key(
    certificate: ActionRiskCertificate,
) -> tuple[float, float, float, float, str]:
    return (
        -certificate.gain_lcb,
        certificate.harm_ucb,
        certificate.brier_delta_ucb,
        certificate.log_loss_delta_ucb,
        certificate.prediction.action.action_id,
    )


def _best_certificate(
    rows: Sequence[ActionRiskCertificate],
) -> ActionRiskCertificate:
    selected = tuple(rows)
    if not selected:
        raise ProtocolError("HARP v16 hierarchy stage cannot rank an empty set.")
    return min(selected, key=_stage_sort_key)


def select_hierarchical_certificate(
    certificates: Sequence[ActionRiskCertificate],
) -> tuple[ActionRiskCertificate | None, HierarchyTrace]:
    eligible = tuple(row for row in certificates if row.passed)
    eligible_ids = tuple(row.prediction.action.action_id for row in eligible)
    if not eligible:
        return None, HierarchyTrace(eligible_ids, None, None, None)

    directions = sorted({row.prediction.action.direction for row in eligible}, key=lambda row: row.value)
    selected_direction = min(
        directions,
        key=lambda direction: (
            _stage_sort_key(
                _best_certificate(
                    tuple(
                        row
                        for row in eligible
                        if row.prediction.action.direction is direction
                    )
                )
            ),
            direction.value,
        ),
    )
    direction_rows = tuple(
        row for row in eligible if row.prediction.action.direction is selected_direction
    )
    families = sorted(
        {row.prediction.action.family for row in direction_rows}, key=lambda row: row.value
    )
    selected_family = min(
        families,
        key=lambda family: (
            _stage_sort_key(
                _best_certificate(
                    tuple(
                        row
                        for row in direction_rows
                        if row.prediction.action.family is family
                    )
                )
            ),
            family.value,
        ),
    )
    family_rows = tuple(
        row for row in direction_rows if row.prediction.action.family is selected_family
    )
    selected = _best_certificate(family_rows)
    return selected, HierarchyTrace(
        eligible_action_ids=eligible_ids,
        selected_direction=selected_direction,
        selected_family=selected_family,
        selected_action_id=selected.prediction.action.action_id,
    )


def _fallback(
    router: FittedSupportRouter,
    menu: LabelFreeCaseMenu,
    prediction: CasePrediction,
    trace: HierarchyTrace,
    reason: str,
) -> RouteDecision:
    decision = RouteDecision(
        outer_target_id=menu.outer_target_id,
        case_id=menu.case_id,
        selected_action_id="B",
        probability_hex=menu.baseline_probability_hex,
        exact_b_fallback=True,
        reason=reason,
        menu_hash=menu.menu_hash,
        prediction_hash=prediction.prediction_hash,
        calibration_hash=router.risk_calibration.calibration_hash,
        admission_hash=router.admission.admission_hash,
        router_hash=router.router_hash,
        hierarchy_trace=trace,
        selected_certificate_hash=None,
    )
    if decision.probability_hex != menu.baseline_probability_hex:
        raise ProtocolError("HARP v16 exact-B fallback is not byte-identical.")
    return decision


def _nested_heldout_calibration(
    menus: Sequence[LabelFreeCaseMenu],
    outcomes: Sequence[SupportActionOutcome],
    *,
    heldout_case_id: str,
    config: RouterFitConfig,
    case_profiles: Sequence[SupportCaseClassProfile],
    candidate_source_ids: Sequence[str],
) -> MenuRiskCalibration:
    inner_menus = tuple(row for row in menus if row.case_id != heldout_case_id)
    inner_outcomes = tuple(
        row for row in outcomes if row.action.case_id != heldout_case_id
    )
    inner_profiles = tuple(
        row for row in case_profiles if row.case_id != heldout_case_id
    )
    if len(inner_menus) < 3:
        raise ProtocolError(
            "HARP v16 nested admission requires at least four support cases."
        )
    inner_crossfit = leave_one_case_out_crossfit(
        inner_menus,
        inner_outcomes,
        config=config,
        minimum_support_cases=len(inner_menus),
        case_profiles=inner_profiles,
        candidate_source_ids=candidate_source_ids,
    )
    return fit_menu_risk_calibration(
        inner_crossfit.records,
        alpha=config.calibration_alpha,
        support_crossfit_hash=canonical_hash(
            {
                "calibration_heldout_case_id": heldout_case_id,
                "inner_crossfit_hash": inner_crossfit.result_hash,
                "heldout_case_labels_used_by_inner_models": False,
            }
        ),
        support_case_ids=inner_crossfit.case_ids,
        outer_target_id=inner_crossfit.outer_target_id,
    )


def evaluate_support_policy_admission(
    crossfit: SupportCrossfitResult,
    menus: Sequence[LabelFreeCaseMenu],
    outcomes: Sequence[SupportActionOutcome],
    *,
    config: RouterFitConfig,
    case_profiles: Sequence[SupportCaseClassProfile],
    candidate_source_ids: Sequence[str],
) -> PolicyAdmission:
    menu_rows, outcome_rows = validate_support_inventory(
        menus,
        outcomes,
        minimum_support_cases=config.minimum_support_cases,
    )
    if tuple(row.case_id for row in menu_rows) != crossfit.case_ids:
        raise ProtocolError("HARP v16 admission surface drifted from its cross-fit.")
    profiles = validate_support_case_profiles(
        menu_rows, case_profiles, require_complete=True
    )
    raw_by_action = {row.action.action_hash: row for row in outcome_rows}
    selected_outcomes: list[SupportActionOutcome | None] = []
    heldout_calibrations: list[tuple[str, str]] = []
    for case_id in crossfit.case_ids:
        records = crossfit.records_for_case(case_id)
        prediction = crossfit.prediction_for_case(case_id).prediction
        calibration = _nested_heldout_calibration(
            menu_rows,
            outcome_rows,
            heldout_case_id=case_id,
            config=config,
            case_profiles=profiles,
            candidate_source_ids=candidate_source_ids,
        )
        heldout_calibrations.append((case_id, calibration.calibration_hash))
        certificates = certify_case_prediction(prediction, calibration, config=config)
        selected, _ = select_hierarchical_certificate(certificates)
        if selected is None:
            selected_outcomes.append(None)
            continue
        outcome = raw_by_action.get(selected.prediction.action.action_hash)
        if outcome is None:
            raise ProtocolError("HARP v16 admitted support action lacks its raw outcome.")
        selected_outcomes.append(outcome)
    # Selection is now sealed for every support case.  Only at this point is
    # the full-support normalizer allowed to aggregate the observed policy
    # utility; it cannot affect any heldout prediction or selection.
    routed_raw = tuple(row for row in selected_outcomes if row is not None)
    full_normalizer = fit_support_fold_normalizer(profiles, crossfit.case_ids)
    routed = tuple(full_normalizer.normalize(row) for row in routed_raw)
    count = len(selected_outcomes)
    routed_count = len(routed)
    coverage = routed_count / count
    gain = sum(row.bacc_gain for row in routed) / count
    brier = sum(row.brier_delta for row in routed) / count
    log_delta = sum(row.log_loss_delta for row in routed) / count
    harm = 0.0 if not routed else sum(float(row.harmed) for row in routed) / routed_count
    admitted = (
        routed_count >= config.minimum_policy_routed_cases
        and coverage >= config.minimum_policy_coverage
        and gain >= config.minimum_policy_gain
        and harm <= config.maximum_policy_harm_rate
        and brier <= config.maximum_policy_brier_delta
        and log_delta <= config.maximum_policy_log_loss_delta
    )
    return PolicyAdmission(
        outer_target_id=crossfit.outer_target_id,
        admitted=admitted,
        support_case_count=count,
        routed_case_count=routed_count,
        coverage=coverage,
        case_equal_bacc_gain=gain,
        routed_harm_rate=harm,
        case_equal_brier_delta=brier,
        case_equal_log_loss_delta=log_delta,
        support_crossfit_hash=crossfit.result_hash,
        heldout_calibration_hashes=tuple(heldout_calibrations),
        policy_config_hash=canonical_hash(config),
    )


def fit_support_router(
    menus: Sequence[LabelFreeCaseMenu],
    outcomes: Sequence[SupportActionOutcome],
    *,
    config: RouterFitConfig | None = None,
    case_profiles: Sequence[SupportCaseClassProfile],
    candidate_source_ids: Sequence[str],
) -> FittedSupportRouter:
    selected_config = RouterFitConfig() if config is None else config
    menu_rows, outcome_rows = validate_support_inventory(
        menus,
        outcomes,
        minimum_support_cases=selected_config.minimum_support_cases,
    )
    profiles = validate_support_case_profiles(
        menu_rows, case_profiles, require_complete=True
    )
    crossfit = leave_one_case_out_crossfit(
        menu_rows,
        outcome_rows,
        config=selected_config,
        case_profiles=profiles,
        candidate_source_ids=candidate_source_ids,
    )
    calibration = fit_menu_risk_calibration(
        crossfit.records,
        alpha=selected_config.calibration_alpha,
        support_crossfit_hash=crossfit.result_hash,
        support_case_ids=crossfit.case_ids,
        outer_target_id=crossfit.outer_target_id,
    )
    full_normalizer = fit_support_fold_normalizer(
        profiles, tuple(row.case_id for row in menu_rows)
    )
    normalized_outcomes = tuple(full_normalizer.normalize(row) for row in outcome_rows)
    model = fit_support_endpoint_model(
        normalized_outcomes,
        config=selected_config,
        candidate_source_ids=candidate_source_ids,
        training_case_ids=tuple(row.case_id for row in menu_rows),
        outer_target_id=menu_rows[0].outer_target_id,
    )
    admission = evaluate_support_policy_admission(
        crossfit,
        menu_rows,
        outcome_rows,
        config=selected_config,
        case_profiles=profiles,
        candidate_source_ids=candidate_source_ids,
    )
    return FittedSupportRouter(
        outer_target_id=menu_rows[0].outer_target_id,
        support_case_ids=tuple(row.case_id for row in menu_rows),
        endpoint_model=model,
        risk_calibration=calibration,
        admission=admission,
        support_crossfit=crossfit,
        config=selected_config,
    )


__all__ = (
    "FittedSupportRouter",
    "HierarchyTrace",
    "PolicyAdmission",
    "RouteDecision",
    "evaluate_support_policy_admission",
    "fit_support_router",
    "select_hierarchical_certificate",
)
