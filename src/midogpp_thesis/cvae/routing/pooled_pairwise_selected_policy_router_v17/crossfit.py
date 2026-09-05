"""Deterministic 5x4 center-stratified nested selection for HARP v17."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import statistics
from typing import Iterable, Sequence

import numpy as np

from ...protocol import ProtocolError
from .composition import (
    build_baseline_composite,
    build_exact_u_composite,
    build_soft_topk_composite,
    soft_arm_id,
)
from .contracts import (
    CompositeKind,
    LabelFreeCaseMenu,
    RouterFitConfig,
    SoftTopKComposite,
    SupportActionOutcome,
    SupportCaseClassProfile,
    SurfaceRole,
)
from .hashing import canonical_hash
from .modeling import (
    CaseModelPrediction,
    PooledScienceModel,
    component_validation_losses,
    fit_pooled_science_model,
)
from .records import SealedOOFSelection, SelectedOOFRecord
from .truth import SupportTruthCapability, score_selected_composite


CaseKey = tuple[str, str]


@dataclass(frozen=True, slots=True)
class ArmSpec:
    kind: CompositeKind
    arm_id: str
    k: int | None = None
    mixing_lambda: float | None = None

    @property
    def order_key(self) -> tuple[int, int, float]:
        if self.kind is CompositeKind.B:
            return (0, 0, 0.0)
        if self.kind is CompositeKind.U_FULL:
            return (1, 0, 0.0)
        return (2, int(self.k or 0), float(self.mixing_lambda or 0.0))

    def public_payload(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "arm_id": self.arm_id,
            "k": self.k,
            "mixing_lambda": self.mixing_lambda,
        }


def candidate_arm_specs(config: RouterFitConfig) -> tuple[ArmSpec, ...]:
    return (
        ArmSpec(CompositeKind.B, "B"),
        ArmSpec(CompositeKind.U_FULL, "U_FULL"),
        *(
            ArmSpec(CompositeKind.SOFT_TOPK, soft_arm_id(k, value), k, value)
            for k in config.k_values
            for value in config.lambda_values
        ),
    )


def center_stratified_folds(
    case_keys: Sequence[CaseKey],
    *,
    fold_count: int,
    namespace: str,
) -> tuple[tuple[CaseKey, ...], ...]:
    keys = tuple(sorted(case_keys))
    if (
        type(fold_count) is not int
        or fold_count < 2
        or len(keys) != len(set(keys))
        or len(keys) < fold_count
    ):
        raise ProtocolError("HARP v17 center-stratified fold inventory is malformed.")
    grouped: dict[str, list[CaseKey]] = defaultdict(list)
    for key in keys:
        grouped[key[0]].append(key)
    folds: list[list[CaseKey]] = [[] for _ in range(fold_count)]
    for center, rows in sorted(grouped.items()):
        ordered = sorted(
            rows,
            key=lambda key: (
                canonical_hash(
                    {
                        "schema_version": "pooled_pairwise_center_stratified_fold_key_v17",
                        "namespace": namespace,
                        "center_id": center,
                        "case_id": key[1],
                    }
                ),
                key[1],
            ),
        )
        for ordinal, key in enumerate(ordered):
            folds[ordinal % fold_count].append(key)
    output = tuple(tuple(sorted(rows)) for rows in folds)
    if any(not rows for rows in output) or set(key for rows in output for key in rows) != set(keys):
        raise ProtocolError("HARP v17 center-stratified folds are incomplete.")
    return output


def validate_source_inventory(
    menus: Sequence[LabelFreeCaseMenu],
    capability: SupportTruthCapability,
    *,
    config: RouterFitConfig,
) -> tuple[LabelFreeCaseMenu, ...]:
    rows = tuple(sorted(menus, key=lambda row: (row.center_id, row.case_id)))
    keys = tuple((row.center_id, row.case_id) for row in rows)
    center_counts = Counter(center for center, _ in keys)
    required_per_center = max(
        config.minimum_cases_per_center,
        config.outer_folds,
        config.inner_folds + 1,
    )
    if (
        not rows
        or any(not isinstance(row, LabelFreeCaseMenu) for row in rows)
        or any(row.surface_role is not SurfaceRole.SOURCE_TRAIN_DEVELOPMENT for row in rows)
        or len(keys) != len(set(keys))
        or keys != capability.case_keys
        or min(center_counts.values(), default=0) < required_per_center
        or (
            config.required_source_case_count is not None
            and len(rows) != config.required_source_case_count
        )
        or (
            config.required_source_center_count is not None
            and len(center_counts) != config.required_source_center_count
        )
    ):
        raise ProtocolError(
            "HARP v17 requires the exact center-stratified source-train case inventory."
        )
    return rows


def _subset_menus(rows: Sequence[LabelFreeCaseMenu], keys: set[CaseKey]) -> tuple[LabelFreeCaseMenu, ...]:
    return tuple(row for row in rows if (row.center_id, row.case_id) in keys)


def _subset_profiles(
    rows: Sequence[SupportCaseClassProfile], keys: set[CaseKey]
) -> tuple[SupportCaseClassProfile, ...]:
    return tuple(row for row in rows if (row.center_id, row.case_id) in keys)


def _subset_outcomes(
    rows: Sequence[SupportActionOutcome], keys: set[CaseKey]
) -> tuple[SupportActionOutcome, ...]:
    return tuple(
        row for row in rows if (row.action.center_id, row.action.case_id) in keys
    )


@dataclass(frozen=True, slots=True)
class FoldChoice:
    outer_fold: int
    arm: ArmSpec
    route_threshold: float
    opportunity_alpha: float
    ranker_alpha: float
    outer_training_case_keys: tuple[CaseKey, ...]
    outer_heldout_case_keys: tuple[CaseKey, ...]
    inner_fold_hash: str
    choice_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.outer_fold) is not int
            or self.outer_fold < 0
            or not isinstance(self.arm, ArmSpec)
            or self.route_threshold < 0.0
            or self.opportunity_alpha <= 0.0
            or self.ranker_alpha <= 0.0
            or set(self.outer_training_case_keys).intersection(self.outer_heldout_case_keys)
        ):
            raise ProtocolError("HARP v17 nested fold choice is malformed.")
        object.__setattr__(
            self,
            "choice_hash",
            canonical_hash(
                {
                    "schema_version": "pooled_pairwise_nested_fold_choice_v17",
                    "outer_fold": self.outer_fold,
                    "arm": self.arm.public_payload(),
                    "route_threshold": self.route_threshold,
                    "opportunity_alpha": self.opportunity_alpha,
                    "ranker_alpha": self.ranker_alpha,
                    "outer_training_case_keys": self.outer_training_case_keys,
                    "outer_heldout_case_keys": self.outer_heldout_case_keys,
                    "inner_fold_hash": self.inner_fold_hash,
                    "outer_cases_used_for_selection": False,
                }
            ),
        )

    def public_payload(self) -> dict[str, object]:
        return {
            "outer_fold": self.outer_fold,
            "arm": self.arm.public_payload(),
            "route_threshold": self.route_threshold,
            "opportunity_alpha": self.opportunity_alpha,
            "ranker_alpha": self.ranker_alpha,
            "outer_training_case_keys": [list(value) for value in self.outer_training_case_keys],
            "outer_heldout_case_keys": [list(value) for value in self.outer_heldout_case_keys],
            "inner_fold_hash": self.inner_fold_hash,
            "choice_hash": self.choice_hash,
            "outer_cases_used_for_selection": False,
        }


@dataclass(frozen=True, slots=True)
class NestedCrossfitResult:
    records: tuple[SelectedOOFRecord, ...]
    fold_choices: tuple[FoldChoice, ...]
    outer_fold_case_keys: tuple[tuple[CaseKey, ...], ...]
    final_arm: ArmSpec
    final_route_threshold: float
    final_opportunity_alpha: float
    final_ranker_alpha: float
    result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        rows = tuple(sorted(self.records, key=lambda row: (row.center_id, row.case_id)))
        keys = tuple((row.center_id, row.case_id) for row in rows)
        expected = tuple(sorted(key for fold in self.outer_fold_case_keys for key in fold))
        if (
            not rows
            or len(keys) != len(set(keys))
            or keys != expected
            or len(self.fold_choices) != len(self.outer_fold_case_keys)
            or not isinstance(self.final_arm, ArmSpec)
        ):
            raise ProtocolError("HARP v17 nested crossfit result is incomplete.")
        object.__setattr__(self, "records", rows)
        object.__setattr__(
            self,
            "result_hash",
            canonical_hash(
                {
                    "schema_version": "pooled_pairwise_nested_crossfit_v17",
                    "record_hashes": tuple(row.score_hash for row in rows),
                    "fold_choice_hashes": tuple(row.choice_hash for row in self.fold_choices),
                    "outer_fold_case_keys": self.outer_fold_case_keys,
                    "final_arm": self.final_arm.public_payload(),
                    "final_route_threshold": self.final_route_threshold,
                    "final_opportunity_alpha": self.final_opportunity_alpha,
                    "final_ranker_alpha": self.final_ranker_alpha,
                    "outer_folds": len(self.outer_fold_case_keys),
                    "all_tuning_nested_inside_outer_folds": True,
                    "nested_oof_evaluates_selection_algorithm_not_final_refit": True,
                }
            ),
        )

    def selection_for(self, center_id: str, case_id: str) -> SealedOOFSelection:
        for row in self.records:
            if row.center_id == center_id and row.case_id == case_id:
                return row.selection
        raise ProtocolError("HARP v17 nested OOF selection is absent.")

    def public_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pooled_pairwise_nested_crossfit_v17",
            "records": [row.public_payload() for row in self.records],
            "fold_choices": [row.public_payload() for row in self.fold_choices],
            "outer_fold_case_keys": [
                [list(value) for value in fold] for fold in self.outer_fold_case_keys
            ],
            "final_arm": self.final_arm.public_payload(),
            "final_route_threshold": self.final_route_threshold,
            "final_opportunity_alpha": self.final_opportunity_alpha,
            "final_ranker_alpha": self.final_ranker_alpha,
            "result_hash": self.result_hash,
            "all_tuning_nested_inside_outer_folds": True,
            "nested_oof_evaluates_selection_algorithm_not_final_refit": True,
        }


def _build_composite(
    menu: LabelFreeCaseMenu,
    prediction: CaseModelPrediction,
    arm: ArmSpec,
    threshold: float,
) -> tuple[SoftTopKComposite, str | None]:
    if arm.kind is CompositeKind.B:
        return build_baseline_composite(menu), "POLICY_ARM_B"
    if prediction.route_score_for(arm.kind) <= threshold:
        return build_baseline_composite(menu), "ROUTE_SCORE_BELOW_THRESHOLD"
    if arm.kind is CompositeKind.U_FULL:
        return build_exact_u_composite(menu), None
    if arm.k is None or arm.mixing_lambda is None:
        raise ProtocolError("HARP v17 soft arm specification is incomplete.")
    try:
        return (
            build_soft_topk_composite(
                menu,
                d01_ranked_actions=prediction.d01_ranked_action_ids,
                d10_ranked_actions=prediction.d10_ranked_action_ids,
                k=arm.k,
                mixing_lambda=arm.mixing_lambda,
            ),
            None,
        )
    except ProtocolError as exc:
        if "fewer than K" not in str(exc):
            raise
        raise _IneligibleArm from exc


class _IneligibleArm(Exception):
    pass


@dataclass(frozen=True, slots=True)
class _HeldPrediction:
    inner_fold: int
    menu: LabelFreeCaseMenu
    prediction: CaseModelPrediction
    model: PooledScienceModel


def _equal_center_policy_moments(records: Sequence[SelectedOOFRecord]) -> np.ndarray:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in records:
        route = float(row.route_selected)
        grouped[row.center_id].append(
            np.asarray(
                [
                    route * row.bacc_gain,
                    route * (float(row.harm) - 0.25),
                    route * (row.brier_delta - 0.002),
                    route * (row.log_loss_delta - 0.005),
                ],
                dtype=np.float64,
            )
        )
    return np.mean(
        np.asarray(
            [np.mean(values, axis=0, dtype=np.float64) for _, values in sorted(grouped.items())],
            dtype=np.float64,
        ),
        axis=0,
        dtype=np.float64,
    )


def _select_arm_and_threshold(
    held: Sequence[_HeldPrediction],
    capability: SupportTruthCapability,
    *,
    config: RouterFitConfig,
    outer_fold: int,
) -> tuple[ArmSpec, float]:
    candidates: list[tuple[tuple[object, ...], ArmSpec, float]] = []
    for arm in candidate_arm_specs(config):
        thresholds = (0.0,) if arm.kind is CompositeKind.B else config.route_thresholds
        for threshold in thresholds:
            records: list[SelectedOOFRecord] = []
            try:
                for row in held:
                    composite, _ = _build_composite(row.menu, row.prediction, arm, threshold)
                    seal = SealedOOFSelection(
                        outer_fold=outer_fold,
                        composite=composite,
                        requested_arm_id=arm.arm_id,
                        route_score=row.prediction.route_score_for(arm.kind),
                        route_threshold=float(threshold),
                        training_case_keys=row.model.training_case_keys,
                        model_hash=row.model.model_hash,
                    )
                    records.append(score_selected_composite(capability, seal))
            except _IneligibleArm:
                continue
            moments = _equal_center_policy_moments(records)
            routed = sum(row.route_selected for row in records)
            feasible_positive = bool(
                routed > 0
                and moments[0] > 0.0
                and moments[1] <= 0.0
                and moments[2] <= 0.0
                and moments[3] <= 0.0
            )
            tier = 0 if feasible_positive else 1 if arm.kind is CompositeKind.B else 2
            candidates.append(
                (
                    (
                        tier,
                        -float(moments[0]) if feasible_positive else 0.0,
                        float(moments[1]) if feasible_positive else 0.0,
                        float(moments[2]) if feasible_positive else 0.0,
                        float(moments[3]) if feasible_positive else 0.0,
                        arm.order_key,
                        float(threshold),
                    ),
                    arm,
                    float(threshold),
                )
            )
    if not candidates:
        raise ProtocolError("HARP v17 nested inner selection has no eligible policy arm.")
    _, arm, threshold = min(candidates, key=lambda row: row[0])
    return arm, threshold


def _select_inner_hyperparameters(
    outer_training_menus: tuple[LabelFreeCaseMenu, ...],
    profiles: tuple[SupportCaseClassProfile, ...],
    outcomes: tuple[SupportActionOutcome, ...],
    capability: SupportTruthCapability,
    *,
    config: RouterFitConfig,
    outer_fold: int,
) -> tuple[float, float, ArmSpec, float, str]:
    outer_keys = tuple((row.center_id, row.case_id) for row in outer_training_menus)
    inner_folds = center_stratified_folds(
        outer_keys,
        fold_count=config.inner_folds,
        namespace=f"HARP_V17_OUTER_{outer_fold}_INNER",
    )
    all_keys = set(outer_keys)
    opportunity_losses: dict[float, list[float]] = defaultdict(list)
    for inner_fold, held_keys_tuple in enumerate(inner_folds):
        held_keys = set(held_keys_tuple)
        train_keys = all_keys - held_keys
        train_menus = _subset_menus(outer_training_menus, train_keys)
        held_menus = _subset_menus(outer_training_menus, held_keys)
        train_profiles = _subset_profiles(profiles, train_keys)
        train_outcomes = _subset_outcomes(outcomes, train_keys)
        held_profiles = _subset_profiles(profiles, held_keys)
        held_outcomes = _subset_outcomes(outcomes, held_keys)
        for alpha in config.opportunity_ridge_alphas:
            model = fit_pooled_science_model(
                train_menus,
                train_profiles,
                train_outcomes,
                opportunity_alpha=alpha,
                ranker_alpha=config.ranker_ridge_alphas[0],
                maximum_numeric_features=config.maximum_numeric_features,
            )
            loss, _ = component_validation_losses(model, held_menus, held_profiles, held_outcomes)
            opportunity_losses[alpha].append(loss)
    opportunity_alpha = min(
        config.opportunity_ridge_alphas,
        key=lambda alpha: (float(np.mean(opportunity_losses[alpha], dtype=np.float64)), alpha),
    )
    ranker_losses: dict[float, list[float]] = defaultdict(list)
    models: dict[tuple[int, float], PooledScienceModel] = {}
    held_by_fold: dict[int, tuple[LabelFreeCaseMenu, ...]] = {}
    for inner_fold, held_keys_tuple in enumerate(inner_folds):
        held_keys = set(held_keys_tuple)
        train_keys = all_keys - held_keys
        train_menus = _subset_menus(outer_training_menus, train_keys)
        held_menus = _subset_menus(outer_training_menus, held_keys)
        held_by_fold[inner_fold] = held_menus
        train_profiles = _subset_profiles(profiles, train_keys)
        train_outcomes = _subset_outcomes(outcomes, train_keys)
        held_profiles = _subset_profiles(profiles, held_keys)
        held_outcomes = _subset_outcomes(outcomes, held_keys)
        for alpha in config.ranker_ridge_alphas:
            model = fit_pooled_science_model(
                train_menus,
                train_profiles,
                train_outcomes,
                opportunity_alpha=opportunity_alpha,
                ranker_alpha=alpha,
                maximum_numeric_features=config.maximum_numeric_features,
            )
            models[(inner_fold, alpha)] = model
            _, loss = component_validation_losses(model, held_menus, held_profiles, held_outcomes)
            ranker_losses[alpha].append(loss)
    ranker_alpha = min(
        config.ranker_ridge_alphas,
        key=lambda alpha: (float(np.mean(ranker_losses[alpha], dtype=np.float64)), alpha),
    )
    held_predictions = tuple(
        _HeldPrediction(inner_fold, menu, model.predict_menu(menu), model)
        for inner_fold, menus in sorted(held_by_fold.items())
        for model in (models[(inner_fold, ranker_alpha)],)
        for menu in menus
    )
    arm, threshold = _select_arm_and_threshold(
        held_predictions,
        capability,
        config=config,
        outer_fold=outer_fold,
    )
    inner_hash = canonical_hash(
        {
            "schema_version": "pooled_pairwise_inner_selection_v17",
            "outer_fold": outer_fold,
            "inner_folds": inner_folds,
            "opportunity_losses": tuple(
                (alpha, tuple(opportunity_losses[alpha])) for alpha in config.opportunity_ridge_alphas
            ),
            "ranker_losses": tuple(
                (alpha, tuple(ranker_losses[alpha])) for alpha in config.ranker_ridge_alphas
            ),
            "selected_opportunity_alpha": opportunity_alpha,
            "selected_ranker_alpha": ranker_alpha,
            "selected_arm": arm.public_payload(),
            "selected_route_threshold": threshold,
            "outer_cases_used": False,
            "all_transforms_fit_inside_inner_training": True,
        }
    )
    return opportunity_alpha, ranker_alpha, arm, threshold, inner_hash


def _modal(values: Iterable[object], *, order: dict[object, int] | None = None) -> object:
    counts = Counter(values)
    if not counts:
        raise ProtocolError("HARP v17 modal selection is empty.")
    ordering = {} if order is None else order
    return min(counts, key=lambda value: (-counts[value], ordering.get(value, 0), repr(value)))


def nested_source_crossfit(
    menus: Sequence[LabelFreeCaseMenu],
    profiles: Sequence[SupportCaseClassProfile],
    outcomes: Sequence[SupportActionOutcome],
    capability: SupportTruthCapability,
    *,
    config: RouterFitConfig,
) -> NestedCrossfitResult:
    menu_rows = tuple(sorted(menus, key=lambda row: (row.center_id, row.case_id)))
    profile_rows = tuple(profiles)
    outcome_rows = tuple(outcomes)
    keys = tuple((row.center_id, row.case_id) for row in menu_rows)
    outer_folds = center_stratified_folds(
        keys,
        fold_count=config.outer_folds,
        namespace="HARP_V17_OUTER",
    )
    all_keys = set(keys)
    records: list[SelectedOOFRecord] = []
    choices: list[FoldChoice] = []
    for outer_fold, held_keys_tuple in enumerate(outer_folds):
        held_keys = set(held_keys_tuple)
        training_keys = all_keys - held_keys
        training_menus = _subset_menus(menu_rows, training_keys)
        held_menus = _subset_menus(menu_rows, held_keys)
        training_profiles = _subset_profiles(profile_rows, training_keys)
        training_outcomes = _subset_outcomes(outcome_rows, training_keys)
        opportunity_alpha, ranker_alpha, arm, threshold, inner_hash = _select_inner_hyperparameters(
            training_menus,
            training_profiles,
            training_outcomes,
            capability,
            config=config,
            outer_fold=outer_fold,
        )
        model = fit_pooled_science_model(
            training_menus,
            training_profiles,
            training_outcomes,
            opportunity_alpha=opportunity_alpha,
            ranker_alpha=ranker_alpha,
            maximum_numeric_features=config.maximum_numeric_features,
        )
        choices.append(
            FoldChoice(
                outer_fold=outer_fold,
                arm=arm,
                route_threshold=threshold,
                opportunity_alpha=opportunity_alpha,
                ranker_alpha=ranker_alpha,
                outer_training_case_keys=tuple(sorted(training_keys)),
                outer_heldout_case_keys=tuple(sorted(held_keys)),
                inner_fold_hash=inner_hash,
            )
        )
        for menu in held_menus:
            prediction = model.predict_menu(menu)
            try:
                composite, _ = _build_composite(menu, prediction, arm, threshold)
            except _IneligibleArm:
                composite = build_baseline_composite(menu)
            selection = SealedOOFSelection(
                outer_fold=outer_fold,
                composite=composite,
                requested_arm_id=arm.arm_id,
                route_score=prediction.route_score_for(arm.kind),
                route_threshold=threshold,
                training_case_keys=model.training_case_keys,
                model_hash=model.model_hash,
            )
            records.append(score_selected_composite(capability, selection))
    arm_specs = candidate_arm_specs(config)
    arm_order = {arm.arm_id: ordinal for ordinal, arm in enumerate(arm_specs)}
    modal_arm_id = _modal((row.arm.arm_id for row in choices), order=arm_order)
    final_arm = next(row for row in arm_specs if row.arm_id == modal_arm_id)
    final_threshold = float(statistics.median(row.route_threshold for row in choices))
    final_opportunity_alpha = float(_modal(row.opportunity_alpha for row in choices))
    final_ranker_alpha = float(_modal(row.ranker_alpha for row in choices))
    return NestedCrossfitResult(
        records=tuple(records),
        fold_choices=tuple(choices),
        outer_fold_case_keys=outer_folds,
        final_arm=final_arm,
        final_route_threshold=final_threshold,
        final_opportunity_alpha=final_opportunity_alpha,
        final_ranker_alpha=final_ranker_alpha,
    )


__all__ = (
    "ArmSpec",
    "FoldChoice",
    "NestedCrossfitResult",
    "candidate_arm_specs",
    "center_stratified_folds",
    "nested_source_crossfit",
    "validate_source_inventory",
)
