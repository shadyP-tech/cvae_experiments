"""Nested case-conditional selection; only abstention is tuned."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ...protocol import ProtocolError
from .contracts import CompositeKind, LabelFreeCaseMenu, RouterFitConfig
from .frontier import build_candidate_frontier, failed_constraints, policy_moments, seal_selections
from .hashing import canonical_hash
from .fit_cache import ScopedFitCache, with_execution_feature_cache
from .records import SealedOOFSelection, SelectedOOFRecord
from .splitting import CaseKey, _subset_menus, center_stratified_folds, validate_source_inventory
from .stacked_fitting import HeldCandidatePrediction, POLICY_ARM_ID, fit_stacked_science_model
from .truth import SupportTruthCapability


def _oracle_summary(item):
    body = {key: value for key, value in item.items() if key not in
            ("candidate_prediction_outcome_joins", "winner_gate_diagnostics", "diagnostic_hash")}
    return {**body, "diagnostic_hash": canonical_hash(body)}


@dataclass(frozen=True, slots=True)
class ArmSpec:
    kind: CompositeKind
    arm_id: str
    k: int | None = None
    mixing_lambda: float | None = None

    def public_payload(self) -> dict[str, object]:
        return {"kind": POLICY_ARM_ID if self.arm_id == POLICY_ARM_ID else self.kind.value,
                "arm_id": self.arm_id, "k": self.k, "mixing_lambda": self.mixing_lambda}


def candidate_arm_specs(config: RouterFitConfig) -> tuple[ArmSpec, ...]:
    from .composition import soft_arm_id
    return (
        ArmSpec(CompositeKind.B, "B"), ArmSpec(CompositeKind.U_FULL, "U_FULL"),
        *(ArmSpec(kind, soft_arm_id(k, value, kind=kind), k, value)
          for kind in (CompositeKind.D01_ONLY, CompositeKind.D10_ONLY, CompositeKind.BOTH)
          for k in config.k_values for value in config.lambda_values),
    )


@dataclass(frozen=True, slots=True)
class FoldChoice:
    outer_fold: int
    route_threshold: float
    policy_enabled: bool
    outer_training_case_keys: tuple[CaseKey, ...]
    outer_heldout_case_keys: tuple[CaseKey, ...]
    inner_fold_hash: str
    choice_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if set(self.outer_training_case_keys).intersection(self.outer_heldout_case_keys):
            raise ProtocolError("HARP v19 fold selection includes its outer held cases.")
        object.__setattr__(self, "choice_hash", canonical_hash(self._payload()))

    @property
    def arm(self) -> ArmSpec:
        # Compatibility metadata, never an executed globally selected action.
        return ArmSpec(CompositeKind.B, POLICY_ARM_ID if self.policy_enabled else "B")

    @property
    def opportunity_alpha(self) -> float:
        return 1.0

    @property
    def ranker_alpha(self) -> float:
        return 1.0

    def _payload(self) -> dict[str, object]:
        return {"outer_fold": self.outer_fold, "arm": self.arm.public_payload(),
                "route_threshold": self.route_threshold, "policy_enabled": self.policy_enabled,
                "opportunity_alpha": 1.0, "ranker_alpha": 1.0,
                "outer_training_case_keys": self.outer_training_case_keys,
                "outer_heldout_case_keys": self.outer_heldout_case_keys,
                "inner_fold_hash": self.inner_fold_hash, "outer_cases_used_for_selection": False}

    def public_payload(self) -> dict[str, object]:
        return {**self._payload(), "choice_hash": self.choice_hash}


@dataclass(frozen=True, slots=True)
class NestedCrossfitResult:
    records: tuple[SelectedOOFRecord, ...]
    fold_choices: tuple[FoldChoice, ...]
    outer_fold_case_keys: tuple[tuple[CaseKey, ...], ...]
    final_route_threshold: float
    final_policy_enabled: bool
    final_inner_fold_hash: str
    frontier_rows: tuple[dict[str, object], ...]
    actual_menu_oracle_diagnostics: tuple[dict[str, object], ...]
    all_outer_prediction_seal_hash: str
    result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        keys = tuple(sorted((row.center_id, row.case_id) for row in self.records))
        expected = tuple(sorted(key for fold in self.outer_fold_case_keys for key in fold))
        if not keys or len(keys) != len(set(keys)) or keys != expected or len(self.fold_choices) != len(self.outer_fold_case_keys):
            raise ProtocolError("HARP v19 nested crossfit result is incomplete.")
        object.__setattr__(self, "result_hash", canonical_hash(self._payload(compact=True)))

    @property
    def final_arm(self) -> ArmSpec:
        return ArmSpec(CompositeKind.B, POLICY_ARM_ID if self.final_policy_enabled else "B")

    @property
    def final_opportunity_alpha(self) -> float:
        return 1.0

    @property
    def final_ranker_alpha(self) -> float:
        return 1.0

    def selection_for(self, center_id: str, case_id: str) -> SealedOOFSelection:
        for row in self.records:
            if (row.center_id, row.case_id) == (center_id, case_id):
                return row.selection
        raise ProtocolError("HARP v19 nested OOF selection is absent.")

    def _payload(self, *, compact: bool) -> dict[str, object]:
        return {
            "schema_version": "harp_v19_fully_nested_case_conditional_crossfit",
            "records": [row.score_hash if compact else row.public_payload() for row in self.records],
            "fold_choices": [row.public_payload() for row in self.fold_choices],
            "outer_fold_case_keys": self.outer_fold_case_keys,
            "final_arm": self.final_arm.public_payload(), "final_route_threshold": self.final_route_threshold,
            "final_policy_enabled": self.final_policy_enabled, "final_inner_fold_hash": self.final_inner_fold_hash,
            "final_opportunity_alpha": 1.0, "final_ranker_alpha": 1.0,
            "frontier_rows": [row["frontier_row_hash"] if compact else row for row in self.frontier_rows],
            "actual_menu_oracle_diagnostics": [_oracle_summary(item) for item in self.actual_menu_oracle_diagnostics],
            "candidate_prediction_outcome_joins": [row["join_hash"] if compact else row
                for item in self.actual_menu_oracle_diagnostics for row in item.get("candidate_prediction_outcome_joins", ())],
            "winner_gate_diagnostics": [row["diagnostic_hash"] if compact else row
                for item in self.actual_menu_oracle_diagnostics for row in item.get("winner_gate_diagnostics", ())],
            "all_outer_prediction_seal_hash": self.all_outer_prediction_seal_hash,
            "outer_oof_normalized_only_after_all_selections_sealed": True,
            "all_tuning_nested_inside_outer_folds": True,
            "only_abstention_threshold_tuned": True,
            "final_threshold_selected_by_full_source_inner_cv": True,
            "nested_oof_evaluates_selection_algorithm_not_final_refit": True,
        }

    def public_payload(self) -> dict[str, object]:
        return {**self._payload(compact=False), "result_hash": self.result_hash}


def _held_predictions(model: object, menus: Sequence[LabelFreeCaseMenu], *, fold: int, config: RouterFitConfig) -> tuple[HeldCandidatePrediction, ...]:
    result = []
    for menu in menus:
        candidates = model.candidate_predictions(menu, config)
        result.append(HeldCandidatePrediction(fold, menu, candidates, model.training_case_keys,
                      model.model_hash, model.winner_prediction(menu, candidates)))
    return tuple(result)


def _select_inner_threshold(
    menus: tuple[LabelFreeCaseMenu, ...], capability: SupportTruthCapability,
    *, config: RouterFitConfig, namespace: str, cache: ScopedFitCache,
) -> tuple[float, bool, str, tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    keys = tuple((row.center_id, row.case_id) for row in menus)
    folds = center_stratified_folds(keys, fold_count=config.inner_folds, namespace=namespace)
    held = []
    for fold, held_keys in enumerate(folds):
        training = _subset_menus(menus, set(keys) - set(held_keys))
        validation = _subset_menus(menus, set(held_keys))
        # Rebuild the entire stack. In particular, validation labels cannot
        # shape ranker proposals used to train this validation outcome model.
        model = fit_stacked_science_model(training, capability.scoped(training), config=config, cache=cache)
        held.extend(_held_predictions(model, validation, fold=fold, config=config))
    held = tuple(held)
    # Freeze every threshold's selections before opening the inner OOF scores.
    selections = {threshold: seal_selections(held, threshold) for threshold in config.route_thresholds}
    scoped = capability.scoped(menus)
    evaluated = {threshold: scoped.score_selections(rows) for threshold, rows in selections.items()}
    valid = tuple(threshold for threshold, rows in evaluated.items() if not failed_constraints(rows))
    enabled = bool(valid)
    threshold = min(valid, key=lambda value: (-policy_moments(evaluated[value])["gain"], -value)) if valid else config.route_thresholds[-1]
    frontier = []
    oracle = []
    for fold in range(len(folds)):
        fold_held = tuple(row for row in held if row.fold == fold)
        rows, diagnostic = build_candidate_frontier(fold_held, scoped, thresholds=config.route_thresholds, stage=f"{namespace}_FOLD_{fold}", normalization_menus=menus, include_detailed_joins=False)
        frontier.extend(rows)
        oracle.append(diagnostic)
    rows, diagnostic = build_candidate_frontier(held, scoped, thresholds=config.route_thresholds, stage=f"{namespace}_ALL_INNER_OOF")
    frontier.extend(rows)
    oracle.append(diagnostic)
    selection_hash = canonical_hash({
        "namespace": namespace, "folds": folds,
        "held_prediction_seal_hashes": tuple(row.prediction_seal_hash for row in held),
        "selected_threshold": threshold, "policy_enabled": enabled,
        "threshold_frontier": tuple({"threshold": value, **policy_moments(rows), "failed_constraints": failed_constraints(rows)} for value, rows in evaluated.items()),
        "fit_complete_learner_independently_per_inner_training_scope": True,
        "candidate_ridge_alpha": config.candidate_ridge_alpha,
        "winner_gate_ridge_alpha": config.winner_gate_ridge_alpha,
    })
    return threshold, enabled, selection_hash, tuple(frontier), tuple(oracle)


@with_execution_feature_cache
def nested_source_crossfit(
    menus: Sequence[LabelFreeCaseMenu], profiles: Sequence[object], outcomes: Sequence[object],
    capability: SupportTruthCapability, *, config: RouterFitConfig, cache: ScopedFitCache | None = None,
) -> NestedCrossfitResult:
    # Precomputed full-source profiles/outcomes are intentionally not learned
    # from here: primitive class support must be rederived in each fit scope.
    cache = ScopedFitCache() if cache is None else cache
    rows = validate_source_inventory(menus, capability, config=config)
    keys = tuple((row.center_id, row.case_id) for row in rows)
    folds = center_stratified_folds(keys, fold_count=config.outer_folds, namespace="HARP_V19_OUTER")
    choices, frontier, oracle, pending, outer_held = [], [], [], [], []
    for fold, held_keys in enumerate(folds):
        training = _subset_menus(rows, set(keys) - set(held_keys))
        validation = _subset_menus(rows, set(held_keys))
        threshold, enabled, inner_hash, fold_frontier, fold_oracle = _select_inner_threshold(
            training, capability.scoped(training), config=config, namespace=f"HARP_V19_OUTER_{fold}_INNER", cache=cache)
        model = fit_stacked_science_model(training, capability.scoped(training), config=config, cache=cache)
        held = _held_predictions(model, validation, fold=fold, config=config)
        outer_held.extend(held)
        pending.extend(seal_selections(held, threshold, enabled=enabled))
        choices.append(FoldChoice(fold, threshold, enabled, model.training_case_keys, held_keys, inner_hash))
        frontier.extend(fold_frontier)
        oracle.extend(fold_oracle)
    # Every source case's selected predictions are sealed before deriving the
    # full-source OOF class-support normalizer and scores.
    outer_seal = canonical_hash(tuple(row.selection_hash for row in pending))
    records = capability.scoped(rows).score_selections(tuple(pending))
    for fold in range(len(folds)):
        held = tuple(row for row in outer_held if row.fold == fold)
        values, diagnostic = build_candidate_frontier(held, capability, thresholds=config.route_thresholds, stage=f"OUTER_{fold}_DIAGNOSTIC", normalization_menus=rows, include_detailed_joins=False)
        frontier.extend(values)
        oracle.append(diagnostic)
    values, diagnostic = build_candidate_frontier(tuple(outer_held), capability, thresholds=config.route_thresholds, stage="ALL_OUTER_OOF_DIAGNOSTIC")
    frontier.extend(values)
    oracle.append(diagnostic)
    # One coherent final training procedure. This final tuning cannot change
    # previously sealed outer predictions and does not certify the final refit.
    threshold, enabled, final_hash, values, diagnostics = _select_inner_threshold(rows, capability, config=config, namespace="HARP_V19_FINAL_INNER", cache=cache)
    frontier.extend(values)
    oracle.extend(diagnostics)
    return NestedCrossfitResult(tuple(records), tuple(choices), folds, threshold, enabled, final_hash, tuple(frontier), tuple(oracle), outer_seal)


__all__ = ("ArmSpec", "FoldChoice", "NestedCrossfitResult", "candidate_arm_specs", "center_stratified_folds", "nested_source_crossfit", "validate_source_inventory")
