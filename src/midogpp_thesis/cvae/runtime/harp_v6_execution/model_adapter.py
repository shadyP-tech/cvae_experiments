"""Source-only model, uncertainty, and learnability adapter for HARP v6."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ...protocol import ProtocolError
from ...routing.compatibility_conditioned_directional_router import (
    BoundedActionEvidence,
    EndpointCalibration,
    HurdlePairwiseModel,
    LearnabilityAdmission,
    SourceActionObservation,
    SourceAdmissionCandidate,
    SourceAdmissionCase,
    SourceOOFPrediction,
    TargetAction,
    bound_action_vs_baseline,
    build_oof_endpoint_rows,
    calibrate_endpoint_uncertainty,
    crossfit_source_predictions,
    evaluate_source_only_admission,
    fit_hurdle_pairwise_model,
    predict_action,
)
from ...routing.harp_protocol import canonical_hash
from .science_pool import execute_science_jobs, science_pool_plan


@dataclass(frozen=True, slots=True)
class OuterRouterBundle:
    outer_target_id: str
    model: HurdlePairwiseModel
    source_oof: tuple[SourceOOFPrediction, ...]
    calibration: EndpointCalibration

    def __post_init__(self) -> None:
        if (
            self.model.outer_target_id != self.outer_target_id
            or not self.source_oof
            or any(
                row.prediction.feature.outer_target_id != self.outer_target_id
                for row in self.source_oof
            )
        ):
            raise ProtocolError("HARP v6 outer-router bundle crossed target roles.")


@dataclass(frozen=True, slots=True)
class RouterFitState:
    bundles: tuple[OuterRouterBundle, ...]

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.bundles, key=lambda row: row.outer_target_id))
        if not ordered or len({row.outer_target_id for row in ordered}) != len(ordered):
            raise ProtocolError("HARP v6 fitted outer-router inventory is malformed.")
        object.__setattr__(self, "bundles", ordered)

    def for_outer(self, outer_target_id: str) -> OuterRouterBundle:
        for bundle in self.bundles:
            if bundle.outer_target_id == str(outer_target_id):
                return bundle
        raise ProtocolError("HARP v6 fitted router lacks an outer target.")


@dataclass(frozen=True, slots=True)
class RouterAdmissionState:
    by_outer: tuple[tuple[str, LearnabilityAdmission], ...]
    router_admitted: bool

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.by_outer, key=lambda row: row[0]))
        if (
            not ordered
            or len({outer for outer, _ in ordered}) != len(ordered)
            or any(not isinstance(value, LearnabilityAdmission) for _, value in ordered)
            or bool(self.router_admitted) != any(value.passed for _, value in ordered)
        ):
            raise ProtocolError("HARP v6 global learnability admission drifted.")
        object.__setattr__(self, "by_outer", ordered)

    def for_outer(self, outer_target_id: str) -> LearnabilityAdmission:
        for outer, value in self.by_outer:
            if outer == str(outer_target_id):
                return value
        raise ProtocolError("HARP v6 admission lacks an outer target.")


@dataclass(frozen=True, slots=True)
class TargetEvidenceState:
    actions: tuple[TargetAction, ...]
    evidence: tuple[BoundedActionEvidence, ...]
    failed_uncertainty_action_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        actions = tuple(
            sorted(
                self.actions,
                key=lambda row: (
                    row.feature.outer_target_id,
                    row.feature.case_id,
                    row.feature.action_id,
                ),
            )
        )
        evidence = tuple(
            sorted(
                self.evidence,
                key=lambda row: (
                    row.prediction.feature.outer_target_id,
                    row.prediction.feature.case_id,
                    row.prediction.feature.action_id,
                ),
            )
        )
        if (
            not actions
            or len({row.target_action_hash for row in actions}) != len(actions)
            or any(
                row.prediction.feature.feature_hash
                not in {action.feature.feature_hash for action in actions}
                for row in evidence
            )
        ):
            raise ProtocolError("HARP v6 target evidence inventory drifted.")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(
            self,
            "failed_uncertainty_action_ids",
            tuple(sorted(set(self.failed_uncertainty_action_ids))),
        )

    def case_actions(self, outer: str, case: str) -> tuple[TargetAction, ...]:
        return tuple(
            row
            for row in self.actions
            if row.feature.outer_target_id == outer and row.feature.case_id == case
        )

    def case_evidence(
        self, outer: str, case: str
    ) -> tuple[BoundedActionEvidence, ...]:
        return tuple(
            row
            for row in self.evidence
            if row.prediction.feature.outer_target_id == outer
            and row.prediction.feature.case_id == case
        )


def _fit_outer_task(
    payload: tuple[
        str,
        tuple[SourceActionObservation, ...],
        tuple[float, ...],
        float,
        int,
    ]
) -> OuterRouterBundle:
    outer, rows, alpha_grid, quantile, minimum_centers = payload
    model = fit_hurdle_pairwise_model(
        rows, outer_target_id=outer, alpha_grid=alpha_grid
    )
    source_oof = crossfit_source_predictions(rows, model=model)
    calibration = calibrate_endpoint_uncertainty(
        build_oof_endpoint_rows(source_oof),
        quantile=quantile,
        minimum_centers_per_cell=minimum_centers,
    )
    return OuterRouterBundle(outer, model, source_oof, calibration)


def fit_outer_routers(
    observations: Sequence[SourceActionObservation],
    *,
    model_config: Mapping[str, object],
    runtime_config: Mapping[str, object],
) -> RouterFitState:
    rows = tuple(observations)
    if not rows or any(not isinstance(row, SourceActionObservation) for row in rows):
        raise ProtocolError("HARP v6 fitting requires typed source observations.")
    science_pool_plan(runtime_config)
    grouped: dict[str, list[SourceActionObservation]] = defaultdict(list)
    for row in rows:
        grouped[row.feature.outer_target_id].append(row)
    alpha_grid = tuple(float(value) for value in model_config["alpha_grid"])
    quantile = float(model_config["residual_quantile"])
    policy = model_config.get("policy")
    if not isinstance(policy, Mapping):
        raise ProtocolError("HARP v6 model policy is absent.")
    minimum_centers = int(policy["min_donor_count"])
    tasks = tuple(
        (
            outer,
            tuple(
                sorted(
                    scoped,
                    key=lambda row: (
                        row.feature.query_center_id,
                        row.feature.case_id,
                        row.feature.action_id,
                    ),
                )
            ),
            alpha_grid,
            quantile,
            minimum_centers,
        )
        for outer, scoped in sorted(grouped.items())
    )
    receipt = execute_science_jobs(
        tasks,
        _fit_outer_task,
        weights=tuple(len(task[1]) for task in tasks),
        workers=int(runtime_config["science_workers"]),
        threads_per_worker=int(runtime_config["science_blas_threads_per_worker"]),
    )
    return RouterFitState(tuple(receipt.values))


def build_source_only_admission(
    fitted: RouterFitState,
    *,
    opportunity_threshold: float,
) -> RouterAdmissionState:
    if not isinstance(fitted, RouterFitState):
        raise ProtocolError("HARP v6 admission requires fitted outer routers.")
    reports: list[tuple[str, LearnabilityAdmission]] = []
    for bundle in fitted.bundles:
        grouped: dict[tuple[str, str], list[SourceOOFPrediction]] = defaultdict(list)
        for row in bundle.source_oof:
            grouped[(row.held_center_id, row.prediction.feature.case_id)].append(row)
        cases: list[SourceAdmissionCase] = []
        for (query, case), rows in sorted(grouped.items()):
            bounded: dict[str, BoundedActionEvidence] = {}
            for row in rows:
                try:
                    bounded[row.prediction.feature.action_id] = bound_action_vs_baseline(
                        row.prediction, calibration=bundle.calibration
                    )
                except ProtocolError as exc:
                    if "No exact endpoint calibration" not in str(exc):
                        raise
            eligible = tuple(
                row
                for row in rows
                if row.prediction.feature.action_id in bounded
                and bounded[row.prediction.feature.action_id].safe_vs_baseline
                and row.prediction.opportunity_probability >= opportunity_threshold
            )
            selected_id = (
                None
                if not eligible
                else min(
                    eligible,
                    key=lambda row: (
                        -row.prediction.ranking_score,
                        row.prediction.feature.action_id,
                    ),
                ).prediction.feature.action_id
            )
            cases.append(
                SourceAdmissionCase(
                    query_center_id=query,
                    case_id=case,
                    candidates=tuple(
                        SourceAdmissionCandidate(
                            action_id=row.prediction.feature.action_id,
                            predicted_score=row.prediction.ranking_score,
                            opportunity_probability=row.prediction.opportunity_probability,
                            safe_selected=(
                                row.prediction.feature.action_id == selected_id
                            ),
                            observed=row.observed,
                        )
                        for row in rows
                    ),
                )
            )
        reports.append(
            (bundle.outer_target_id, evaluate_source_only_admission(tuple(cases)))
        )
    return RouterAdmissionState(tuple(reports), any(value.passed for _, value in reports))


def predict_target_evidence(
    actions: Sequence[TargetAction], fitted: RouterFitState
) -> TargetEvidenceState:
    typed = tuple(actions)
    if not typed or any(not isinstance(row, TargetAction) for row in typed):
        raise ProtocolError("HARP v6 target prediction requires complete typed actions.")
    evidence: list[BoundedActionEvidence] = []
    failed: list[str] = []
    for action in typed:
        bundle = fitted.for_outer(action.feature.outer_target_id)
        prediction = predict_action(bundle.model, action.feature)
        try:
            evidence.append(
                bound_action_vs_baseline(prediction, calibration=bundle.calibration)
            )
        except ProtocolError as exc:
            if "No exact endpoint calibration" not in str(exc):
                raise
            failed.append(action.feature.action_id)
    return TargetEvidenceState(typed, tuple(evidence), tuple(failed))


def model_manifest(state: RouterFitState) -> dict[str, object]:
    body = {
        "schema_version": "midogpp_harp_v6_source_only_router_v1",
        "outer_models": [
            {
                "outer_target_id": bundle.outer_target_id,
                "inner_model_hash": bundle.model.model_hash,
                "feature_names": list(bundle.model.feature_names),
                "normalization_mean": list(bundle.model.normalization_mean),
                "normalization_scale": list(bundle.model.normalization_scale),
                "design_names": list(bundle.model.design_names),
                "hurdle_coefficients": list(bundle.model.hurdle_coefficients),
                "pairwise_coefficients": list(bundle.model.pairwise_coefficients),
                "endpoint_coefficients": {
                    name: list(values)
                    for name, values in bundle.model.endpoint_coefficients
                },
                "source_oof_hash": canonical_hash(
                    [row.receipt_hash for row in bundle.source_oof]
                ),
                "source_oof_receipt_hashes": [
                    row.receipt_hash for row in bundle.source_oof
                ],
                "calibration_hash": bundle.calibration.calibration_hash,
                "calibration_quantile": bundle.calibration.quantile,
                "calibration_cells": [
                    {
                        "action_key": cell.action_key,
                        "comparator_key": cell.comparator_key,
                        "bacc_overprediction_quantile": (
                            cell.bacc_overprediction_quantile
                        ),
                        "brier_underprediction_quantile": (
                            cell.brier_underprediction_quantile
                        ),
                        "log_underprediction_quantile": (
                            cell.log_underprediction_quantile
                        ),
                        "source_center_ids": list(cell.source_center_ids),
                        "row_count": cell.row_count,
                        "cell_hash": cell.cell_hash,
                    }
                    for cell in bundle.calibration.cells
                ],
                "selected_alpha": bundle.model.selected_alpha,
                "alpha_grid": list(bundle.model.alpha_grid),
                "fold_losses": [
                    {
                        "held_center_id": row.held_center_id,
                        "alpha": row.alpha,
                        "hurdle_log_loss": row.hurdle_log_loss,
                        "pairwise_mse": row.pairwise_mse,
                    }
                    for row in bundle.model.fold_losses
                ],
                "training_query_ids": list(bundle.model.training_query_ids),
                "training_candidate_ids": list(bundle.model.training_candidate_ids),
                "training_case_count": bundle.model.training_case_count,
                "training_row_hash": bundle.model.training_row_hash,
                "calibration_cell_count": len(bundle.calibration.cells),
            }
            for bundle in state.bundles
        ],
        "target_labels_used": False,
        "old_aggregate_utility_surface_used": False,
    }
    return {**body, "inner_fit_hash": canonical_hash(body)}


def admission_manifest(state: RouterAdmissionState) -> dict[str, object]:
    body = {
        "schema_version": "midogpp_harp_v6_source_only_learnability_admission_v1",
        "router_admitted": state.router_admitted,
        "admitted_outer_target_ids": [
            outer for outer, value in state.by_outer if value.passed
        ],
        "rejected_outer_target_ids": [
            outer for outer, value in state.by_outer if not value.passed
        ],
        "outer_admissions": [
            {
                "outer_target_id": outer,
                "passed": value.passed,
                "admission_hash": value.admission_hash,
                "reasons": list(value.reasons),
                "sign_accuracy": value.sign_accuracy,
                "top1_accuracy": value.top1_accuracy,
                "minimum_delete_center_tau": value.minimum_delete_center_tau,
                "safe_coverage": value.safe_coverage,
                "selected_count": value.selected_count,
                "harmful_selected_count": value.harmful_selected_count,
                "proper_loss_violation_count": value.proper_loss_violation_count,
            }
            for outer, value in state.by_outer
        ],
        "per_outer_failure_forces_exact_b": True,
        "zero_admitted_outer_targets_forces_global_exact_b": True,
        "target_labels_used": False,
    }
    return {**body, "inner_admission_hash": canonical_hash(body)}


def target_evidence_manifest(state: TargetEvidenceState) -> dict[str, object]:
    body = {
        "schema_version": "midogpp_harp_v6_complete_target_evidence_v1",
        "action_count": len(state.actions),
        "case_count": len(
            {
                (row.feature.outer_target_id, row.feature.case_id)
                for row in state.actions
            }
        ),
        "target_action_hashes": [row.target_action_hash for row in state.actions],
        "bounded_evidence_hashes": [row.evidence_hash for row in state.evidence],
        "failed_uncertainty_action_ids": list(state.failed_uncertainty_action_ids),
        "complete_actions_retained_when_uncertainty_missing": True,
        "target_labels_used": False,
    }
    return {**body, "inner_target_evidence_hash": canonical_hash(body)}


__all__ = (
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
