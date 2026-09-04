"""Strict nested-center fitting for the HARP v14 residual policy router."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .acceptor import (
    SelectedActionAcceptor,
    SelectedActionObservation,
    fit_selected_action_acceptor,
    selected_action_features,
)
from .contracts import (
    ActionScore,
    CasePrediction,
    LabelFreeAction,
    SourceActionOutcome,
    action_group,
)
from .effective_menu import EffectiveMenu
from .hashing import canonical_hash
from .outcome_inventory import SourceOutcomeUniverse
from .ranker import PairwiseRanker, fit_pairwise_ranker
from .residuals import residualize_menu


@dataclass(frozen=True, slots=True)
class PairwiseFitConfig:
    pairwise_alpha: float = 1.0
    residual_alpha: float = 1.0
    acceptor_alpha: float = 1.0
    pairwise_tie_tolerance: float = 1e-12
    max_selected_brier_delta: float = 0.002
    max_selected_log_delta: float = 0.005
    max_irls_iterations: int = 32
    min_source_centers: int = 4

    def __post_init__(self) -> None:
        values = (
            self.pairwise_alpha,
            self.residual_alpha,
            self.acceptor_alpha,
            self.pairwise_tie_tolerance,
            self.max_selected_brier_delta,
            self.max_selected_log_delta,
        )
        if (
            any(not math.isfinite(value) for value in values)
            or self.pairwise_alpha <= 0.0
            or self.residual_alpha <= 0.0
            or self.acceptor_alpha <= 0.0
            or self.pairwise_tie_tolerance < 0.0
            or int(self.max_irls_iterations) < 1
            or int(self.min_source_centers) < 4
        ):
            raise ProtocolError("HARP v14 pairwise fit configuration is malformed.")


@dataclass(frozen=True, slots=True)
class PairwiseResidualRouterModel:
    outer_target_id: str
    training_center_ids: tuple[str, ...]
    training_candidate_ids: tuple[str, ...]
    excluded_center_ids: tuple[str, ...]
    ranker: PairwiseRanker
    acceptor: SelectedActionAcceptor
    fit_config: PairwiseFitConfig
    model_hash: str = field(init=False)

    def __post_init__(self) -> None:
        training = tuple(sorted(self.training_center_ids))
        candidates = tuple(sorted(self.training_candidate_ids))
        excluded = tuple(sorted(self.excluded_center_ids))
        if (
            self.outer_target_id not in excluded
            or set(training) & set(excluded)
            or set(candidates) & set(excluded)
            or training != self.ranker.training_center_ids
            or training != self.acceptor.training_center_ids
            or candidates != self.ranker.training_candidate_ids
            or excluded != self.ranker.excluded_center_ids
            or excluded != self.acceptor.excluded_center_ids
        ):
            raise ProtocolError("HARP v14 model roles crossed an exclusion boundary.")
        object.__setattr__(self, "training_center_ids", training)
        object.__setattr__(self, "training_candidate_ids", candidates)
        object.__setattr__(self, "excluded_center_ids", excluded)
        object.__setattr__(
            self,
            "model_hash",
            canonical_hash(
                {
                    "schema_version": "policy_calibrated_residual_router_model_v14",
                    "outer_target_id": self.outer_target_id,
                    "training_center_ids": training,
                    "training_candidate_ids": candidates,
                    "excluded_center_ids": excluded,
                    "ranker_hash": self.ranker.ranker_hash,
                    "acceptor_hash": self.acceptor.acceptor_hash,
                    "fit_config": self.fit_config,
                    "rank_all_before_acceptance": True,
                    "per_action_certificate_gate": False,
                    "target_evaluation_labels_used": False,
                }
            ),
        )

    def public_payload(self) -> dict[str, object]:
        return {
            "schema_version": "policy_calibrated_residual_router_model_v14",
            "model_hash": self.model_hash,
            "outer_target_id": self.outer_target_id,
            "training_center_ids": list(self.training_center_ids),
            "training_candidate_ids": list(self.training_candidate_ids),
            "excluded_center_ids": list(self.excluded_center_ids),
            "fit_config": {
                name: getattr(self.fit_config, name)
                for name in self.fit_config.__dataclass_fields__
            },
            "ranker": self.ranker.public_payload(),
            "acceptor": self.acceptor.public_payload(),
            "rank_all_before_acceptance": True,
            "per_action_certificate_gate": False,
            "target_evaluation_labels_used": False,
        }


@dataclass(frozen=True, slots=True)
class NestedPolicyFold:
    heldout_center_id: str
    training_center_ids: tuple[str, ...]
    predictions: tuple[CasePrediction, ...]
    heldout_predictions: tuple[CasePrediction, ...]
    fold_hash: str = field(init=False)

    def __post_init__(self) -> None:
        training = tuple(sorted(self.training_center_ids))
        prediction_keys = tuple(
            (row.query_center_id, row.case_id)
            for row in (*self.predictions, *self.heldout_predictions)
        )
        heldout_model_hashes = {
            row.model_hash for row in self.heldout_predictions
        }
        if (
            self.heldout_center_id in training
            or not training
            or not self.predictions
            or not self.heldout_predictions
            or {row.query_center_id for row in self.predictions} != set(training)
            or {row.query_center_id for row in self.heldout_predictions}
            != {self.heldout_center_id}
            or any(
                self.heldout_center_id not in row.excluded_center_ids
                or row.query_center_id not in row.excluded_center_ids
                or row.query_center_id in row.training_candidate_ids
                for row in (*self.predictions, *self.heldout_predictions)
            )
            or len(prediction_keys) != len(set(prediction_keys))
            or len(heldout_model_hashes) != 1
        ):
            raise ProtocolError("HARP v14 nested policy fold leaked or is incomplete.")
        object.__setattr__(self, "training_center_ids", training)
        object.__setattr__(
            self,
            "fold_hash",
            canonical_hash(
                {
                    "schema_version": "policy_calibrated_nested_policy_fold_v14",
                    "heldout_center_id": self.heldout_center_id,
                    "training_center_ids": training,
                    "training_prediction_hashes": tuple(
                        row.prediction_hash for row in self.predictions
                    ),
                    "heldout_prediction_hashes": tuple(
                        row.prediction_hash for row in self.heldout_predictions
                    ),
                    "heldout_query_candidate_and_threshold_excluded": True,
                }
            ),
        )

    @property
    def heldout_model_hash(self) -> str:
        """Return the single model hash used for every prelabel q prediction."""

        return self.heldout_predictions[0].model_hash

    @property
    def outer_target_id(self) -> str:
        return self.heldout_predictions[0].outer_target_id


@dataclass(frozen=True, slots=True)
class SourceLODOResult:
    outer_target_id: str
    final_model: PairwiseResidualRouterModel
    oof_predictions: tuple[CasePrediction, ...]
    heldout_model_hashes: tuple[tuple[str, str], ...]
    nested_policy_folds: tuple[NestedPolicyFold, ...]
    nested_outcome_universes: tuple[tuple[str, SourceOutcomeUniverse], ...]
    config: PairwiseFitConfig
    result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        centers = tuple(sorted({row.query_center_id for row in self.oof_predictions}))
        folds_by_center = {
            row.heldout_center_id: row for row in self.nested_policy_folds
        }
        universe_rows = tuple(
            sorted(self.nested_outcome_universes, key=lambda row: row[0])
        )
        universes_by_center = dict(universe_rows)
        sealed_oof = tuple(
            prediction
            for center in sorted(folds_by_center)
            for prediction in folds_by_center[center].heldout_predictions
        )
        sealed_hashes = tuple(
            (center, folds_by_center[center].heldout_model_hash)
            for center in sorted(folds_by_center)
        )
        if (
            self.final_model.outer_target_id != self.outer_target_id
            or not centers
            or any(
                row.outer_target_id != self.outer_target_id
                or row.query_center_id not in row.excluded_center_ids
                or row.query_center_id in row.training_center_ids
                or row.query_center_id in row.training_candidate_ids
                for row in self.oof_predictions
            )
            or len({(row.query_center_id, row.case_id) for row in self.oof_predictions})
            != len(self.oof_predictions)
            or len(folds_by_center) != len(self.nested_policy_folds)
            or set(folds_by_center) != set(centers)
            or len(universes_by_center) != len(universe_rows)
            or set(universes_by_center) != set(centers)
            or any(
                not isinstance(universe, SourceOutcomeUniverse)
                for universe in universes_by_center.values()
            )
            or {center for center, _ in self.heldout_model_hashes} != set(centers)
            or tuple(row.prediction_hash for row in sealed_oof)
            != tuple(row.prediction_hash for row in self.oof_predictions)
            or sealed_hashes != self.heldout_model_hashes
        ):
            raise ProtocolError("HARP v14 source LODO inventory leaked or is incomplete.")
        for center in centers:
            fold = folds_by_center[center]
            universe = universes_by_center[center]
            universe.bind_predictions(
                (*fold.predictions, *fold.heldout_predictions),
                require_complete=True,
            )
        object.__setattr__(self, "nested_outcome_universes", universe_rows)
        object.__setattr__(
            self,
            "result_hash",
            canonical_hash(
                {
                    "schema_version": "policy_calibrated_source_lodo_v14",
                    "outer_target_id": self.outer_target_id,
                    "final_model_hash": self.final_model.model_hash,
                    "oof_prediction_hashes": tuple(
                        row.prediction_hash for row in self.oof_predictions
                    ),
                    "heldout_model_hashes": self.heldout_model_hashes,
                    "nested_policy_fold_hashes": tuple(
                        row.fold_hash for row in self.nested_policy_folds
                    ),
                    "nested_outcome_universe_hashes": tuple(
                        (center, universe.universe_hash)
                        for center, universe in universe_rows
                    ),
                    "config": self.config,
                    "strict_outer_query_candidate_nested_lodo": True,
                }
            ),
        )

    def numeric_oof_payload(self) -> dict[str, object]:
        return {
            "schema_version": "policy_calibrated_numeric_oof_v14",
            "outer_target_id": self.outer_target_id,
            "result_hash": self.result_hash,
            "config": {
                name: getattr(self.config, name)
                for name in self.config.__dataclass_fields__
            },
            "heldout_model_hashes": [list(row) for row in self.heldout_model_hashes],
            "nested_outcome_universe_hashes": [
                [center, universe.universe_hash]
                for center, universe in self.nested_outcome_universes
            ],
            "nested_policy_folds": [
                {
                    "heldout_center_id": fold.heldout_center_id,
                    "training_center_ids": list(fold.training_center_ids),
                    "fold_hash": fold.fold_hash,
                    "training_rows": [row.public_payload() for row in fold.predictions],
                    "heldout_rows": [row.public_payload() for row in fold.heldout_predictions],
                }
                for fold in self.nested_policy_folds
            ],
            "rows": [row.public_payload() for row in self.oof_predictions],
        }

    def nested_outcome_universe(self, heldout_center_id: str) -> SourceOutcomeUniverse:
        heldout = str(heldout_center_id)
        for center, universe in self.nested_outcome_universes:
            if center == heldout:
                return universe
        raise ProtocolError("HARP v14 nested outcome universe is absent.")

    @property
    def nested_outcome_universe_hashes(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (center, universe.universe_hash)
            for center, universe in self.nested_outcome_universes
        )


@dataclass(frozen=True, slots=True)
class _SourceCase:
    menu: EffectiveMenu
    outcomes: tuple[SourceActionOutcome, ...]


def _source_cases(
    observations: Sequence[SourceActionOutcome],
    effective_menus: Sequence[EffectiveMenu] | None,
    *,
    min_centers: int,
) -> tuple[_SourceCase, ...]:
    rows = tuple(observations)
    if any(not isinstance(row, SourceActionOutcome) for row in rows):
        raise ProtocolError("HARP v14 fitting requires source-development outcomes.")
    if effective_menus is None:
        raise ProtocolError(
            "HARP v14 fitting requires the explicit effective-menu inventory; "
            "outcomes cannot reconstruct exact-B controls."
        )
    menus = tuple(effective_menus)
    by_key = {(menu.query_center_id, menu.case_id): menu for menu in menus}
    if not menus or len(by_key) != len(menus):
        raise ProtocolError("HARP v14 effective-menu inventory is empty or duplicated.")
    by_outcome: dict[tuple[str, str], list[SourceActionOutcome]] = defaultdict(list)
    for row in rows:
        key = (row.action.query_center_id, row.action.case_id)
        menu = by_key.get(key)
        if menu is None or not any(
            action.action_id == row.action.action_id
            and action.action_hash == row.action.action_hash
            for action in menu.actions
        ):
            raise ProtocolError("HARP v14 outcome is absent from its sealed menu.")
        by_outcome[key].append(row)
    cases: list[_SourceCase] = []
    for key in sorted(by_key):
        menu = by_key[key]
        members = tuple(
            sorted(by_outcome.get(key, ()), key=lambda row: row.action.action_id)
        )
        if {row.action.action_id for row in members} != {
            action.action_id for action in menu.actions
        }:
            raise ProtocolError("HARP v14 source outcome/menu inventory is incomplete.")
        cases.append(_SourceCase(menu, members))
    typed = tuple(cases)
    outers = {case.menu.outer_target_id for case in typed}
    schemas = {case.menu.feature_names for case in typed}
    centers = {case.menu.query_center_id for case in typed}
    if (
        len(outers) != 1
        or len(schemas) != 1
        or len(centers) < min_centers
        or next(iter(outers)) in centers
    ):
        raise ProtocolError("HARP v14 source surface crossed outer/schema/center roles.")
    return typed


def _filter_menu(menu: EffectiveMenu, excluded: frozenset[str]) -> EffectiveMenu:
    actions = tuple(
        action for action in menu.actions if action.candidate_source_id not in excluded
    )
    action_ids = {action.action_id for action in actions}
    aliases = tuple(
        pair for pair in menu.duplicate_representatives if pair[1] in action_ids
    )
    return EffectiveMenu(
        outer_target_id=menu.outer_target_id,
        query_center_id=menu.query_center_id,
        case_id=menu.case_id,
        feature_names=menu.feature_names,
        baseline_probability_hex=menu.baseline_probability_hex,
        actions=actions,
        dropped_noop_action_ids=menu.dropped_noop_action_ids,
        duplicate_representatives=aliases,
    )


def _filter_cases(
    cases: Sequence[_SourceCase], *, excluded_center_ids: frozenset[str]
) -> tuple[_SourceCase, ...]:
    output: list[_SourceCase] = []
    for case in cases:
        if case.menu.query_center_id in excluded_center_ids:
            continue
        menu = _filter_menu(case.menu, excluded_center_ids)
        hashes = {action.action_hash for action in menu.actions}
        outcomes = tuple(row for row in case.outcomes if row.action.action_hash in hashes)
        output.append(_SourceCase(menu, outcomes))
    return tuple(output)


@dataclass(slots=True)
class _FixedExclusionFitter:
    """Memoized fitter whose fixed exclusions can never be weakened.

    A pseudo-target fold constructs one instance with ``{H, q}``. Every
    transform, ranker and selected-action acceptor inherits those exclusions;
    an inner calibration center ``r`` is additive.
    """

    all_cases: tuple[_SourceCase, ...]
    config: PairwiseFitConfig
    fixed_excluded_center_ids: frozenset[str]
    ranker_cache: dict[frozenset[str], PairwiseRanker] = field(default_factory=dict)
    model_cache: dict[frozenset[str], PairwiseResidualRouterModel] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.all_cases:
            raise ProtocolError("HARP v14 fixed-exclusion fit has no source cases.")
        outers = {case.menu.outer_target_id for case in self.all_cases}
        if len(outers) != 1:
            raise ProtocolError("HARP v14 fixed-exclusion fit crossed outer targets.")
        outer = next(iter(outers))
        if (
            outer not in self.fixed_excluded_center_ids
            or any(
                case.menu.query_center_id in self.fixed_excluded_center_ids
                for case in self.all_cases
            )
        ):
            raise ProtocolError(
                "HARP v14 fixed-exclusion fit retained an excluded query outcome."
            )

    @property
    def outer_target_id(self) -> str:
        return self.all_cases[0].menu.outer_target_id

    @property
    def training_center_ids(self) -> tuple[str, ...]:
        return tuple(sorted({case.menu.query_center_id for case in self.all_cases}))

    def _exclusions(self, additional: Sequence[str] = ()) -> frozenset[str]:
        return frozenset(
            (*self.fixed_excluded_center_ids, *(str(value) for value in additional))
        )

    def ranker_for(self, additional: Sequence[str] = ()) -> PairwiseRanker:
        excluded = self._exclusions(additional)
        cached = self.ranker_cache.get(excluded)
        if cached is not None:
            return cached
        cases = _filter_cases(self.all_cases, excluded_center_ids=excluded)
        value = fit_pairwise_ranker(
            cases,
            excluded_center_ids=tuple(sorted(excluded)),
            pairwise_alpha=self.config.pairwise_alpha,
            residual_alpha=self.config.residual_alpha,
            pairwise_tie_tolerance=self.config.pairwise_tie_tolerance,
        )
        if not set(value.excluded_center_ids).issuperset(
            self.fixed_excluded_center_ids
        ):
            raise ProtocolError("HARP v14 ranker dropped a fixed exclusion.")
        self.ranker_cache[excluded] = value
        return value

    def model_for(self, additional: Sequence[str] = ()) -> PairwiseResidualRouterModel:
        excluded = self._exclusions(additional)
        cached = self.model_cache.get(excluded)
        if cached is not None:
            return cached
        training_cases = _filter_cases(self.all_cases, excluded_center_ids=excluded)
        training_centers = tuple(
            sorted({case.menu.query_center_id for case in training_cases})
        )
        if len(training_centers) < 2:
            raise ProtocolError("HARP v14 acceptor needs at least two training centers.")
        ranker = self.ranker_for(additional)
        records: list[SelectedActionObservation] = []
        for heldout in training_centers:
            inner_additional = (*additional, heldout)
            inner_excluded = self._exclusions(inner_additional)
            inner_ranker = self.ranker_for(inner_additional)
            heldout_cases = tuple(
                _SourceCase(
                    _filter_menu(case.menu, inner_excluded),
                    tuple(
                        row
                        for row in case.outcomes
                        if row.action.candidate_source_id not in inner_excluded
                    ),
                )
                for case in self.all_cases
                if case.menu.query_center_id == heldout
            )
            records.extend(
                _selection_record(inner_ranker, case) for case in heldout_cases
            )
        acceptor = fit_selected_action_acceptor(
            records,
            excluded_center_ids=tuple(sorted(excluded)),
            ridge_alpha=self.config.acceptor_alpha,
            max_brier_delta=self.config.max_selected_brier_delta,
            max_log_delta=self.config.max_selected_log_delta,
            max_irls_iterations=self.config.max_irls_iterations,
        )
        value = PairwiseResidualRouterModel(
            outer_target_id=self.outer_target_id,
            training_center_ids=ranker.training_center_ids,
            training_candidate_ids=ranker.training_candidate_ids,
            excluded_center_ids=ranker.excluded_center_ids,
            ranker=ranker,
            acceptor=acceptor,
            fit_config=self.config,
        )
        if not set(value.excluded_center_ids).issuperset(
            self.fixed_excluded_center_ids
        ):
            raise ProtocolError("HARP v14 model dropped a fixed exclusion.")
        self.model_cache[excluded] = value
        return value


@dataclass(frozen=True, slots=True)
class _RankedAction:
    action: LabelFreeAction
    budget_gain: float
    allocation_gain: float

    @property
    def score(self) -> float:
        return self.budget_gain + self.allocation_gain


def _rank_menu(ranker: PairwiseRanker, menu: EffectiveMenu) -> tuple[tuple[_RankedAction, ...], str, float]:
    if (
        menu.outer_target_id != ranker.outer_target_id
        or menu.query_center_id not in ranker.excluded_center_ids
        or any(action.candidate_source_id in ranker.excluded_center_ids for action in menu.actions)
    ):
        raise ProtocolError("HARP v14 prediction crossed query/candidate exclusion.")
    ranked = tuple(
        _RankedAction(row.action, *ranker.contributions(row))
        for row in residualize_menu(menu)
    )
    ordered = sorted(ranked, key=lambda row: (-row.score, row.action.action_id))
    tolerance = ranker.pairwise_tie_tolerance
    if not ordered or ordered[0].score <= tolerance:
        top = "B"
        second = ordered[0].score if ordered else 0.0
        margin = max(0.0, -second)
    else:
        top = ordered[0].action.action_id
        runner_up = max(0.0, ordered[1].score if len(ordered) > 1 else 0.0)
        margin = max(0.0, ordered[0].score - runner_up)
    return ranked, top, float(margin)


def _features_for(
    selected: _RankedAction | None,
    ranked: Sequence[_RankedAction],
    *,
    rank_margin: float,
) -> tuple[float, ...]:
    return selected_action_features(
        selected_score=0.0 if selected is None else selected.score,
        budget_gain=0.0 if selected is None else selected.budget_gain,
        allocation_gain=0.0 if selected is None else selected.allocation_gain,
        rank_margin=rank_margin,
        all_scores=tuple(row.score for row in ranked),
        action_kind="B" if selected is None else selected.action.action_kind,
        direction="" if selected is None else selected.action.direction.value,
    )


def _selection_record(
    ranker: PairwiseRanker, case: _SourceCase
) -> SelectedActionObservation:
    ranked, selected_id, margin = _rank_menu(ranker, case.menu)
    selected = next((row for row in ranked if row.action.action_id == selected_id), None)
    if selected is None:
        gain = brier = log_delta = 0.0
    else:
        outcome = next(
            row for row in case.outcomes if row.action.action_id == selected_id
        )
        gain, brier, log_delta = outcome.bacc_gain, outcome.brier_delta, outcome.log_delta
    return SelectedActionObservation(
        outer_target_id=case.menu.outer_target_id,
        query_center_id=case.menu.query_center_id,
        case_id=case.menu.case_id,
        selected_action_id=selected_id,
        feature_values=_features_for(selected, ranked, rank_margin=margin),
        bacc_gain=gain,
        brier_delta=brier,
        log_delta=log_delta,
        selection_excluded_center_ids=ranker.excluded_center_ids,
        selection_ranker_hash=ranker.ranker_hash,
    )


def predict_case(
    model: PairwiseResidualRouterModel, menu: EffectiveMenu
) -> CasePrediction:
    if not isinstance(model, PairwiseResidualRouterModel) or not isinstance(menu, EffectiveMenu):
        raise ProtocolError("HARP v14 prediction requires typed model/menu inputs.")
    ranked, top, margin = _rank_menu(model.ranker, menu)
    all_scores = tuple(row.score for row in ranked)
    action_scores: list[ActionScore] = []
    top_probability = 0.0
    for candidate in ranked:
        competitor = max(
            (0.0, *(row.score for row in ranked if row.action.action_id != candidate.action.action_id))
        )
        candidate_margin = max(0.0, candidate.score - competitor)
        probability, harm, gain, brier, log_delta = model.acceptor.predict(
            _features_for(candidate, ranked, rank_margin=candidate_margin)
        )
        if candidate.action.action_id == top:
            top_probability = probability
        action_scores.append(
            ActionScore(
                action_id=candidate.action.action_id,
                action_hash=candidate.action.action_hash,
                action_group=action_group(candidate.action),
                direction=candidate.action.direction,
                pairwise_score=candidate.score,
                predicted_budget_gain=candidate.budget_gain,
                predicted_allocation_gain=candidate.allocation_gain,
                predicted_total_gain=candidate.score,
                predicted_harm_probability=harm,
                predicted_brier_delta=brier,
                predicted_log_delta=log_delta,
                acceptance_probability=probability,
                model_available=True,
            )
        )
    if top == "B":
        top_probability = 0.0
    return CasePrediction(
        outer_target_id=menu.outer_target_id,
        query_center_id=menu.query_center_id,
        case_id=menu.case_id,
        action_scores=tuple(action_scores),
        raw_top_action_id=top,
        top_action_id=top,
        acceptance_probability=top_probability,
        rank_margin=margin,
        model_hash=model.model_hash,
        ranker_hash=model.ranker.ranker_hash,
        acceptor_hash=model.acceptor.acceptor_hash,
        training_center_ids=model.training_center_ids,
        training_candidate_ids=model.training_candidate_ids,
        excluded_center_ids=model.excluded_center_ids,
        menu_hash=menu.menu_hash,
    )


def _single_config(config_grid: Sequence[PairwiseFitConfig]) -> PairwiseFitConfig:
    grid = tuple(config_grid)
    if len(grid) != 1 or not isinstance(grid[0], PairwiseFitConfig):
        raise ProtocolError("HARP v14 requires one predeclared fit configuration.")
    return grid[0]


def _heldout_menus(
    menus: Sequence[EffectiveMenu],
    *,
    outer_target_id: str,
    heldout_center_id: str,
    excluded_center_ids: frozenset[str],
) -> tuple[EffectiveMenu, ...]:
    typed = tuple(menus)
    keys = {(row.query_center_id, row.case_id) for row in typed}
    if (
        not typed
        or any(not isinstance(row, EffectiveMenu) for row in typed)
        or len(keys) != len(typed)
        or any(
            row.outer_target_id != outer_target_id
            or row.query_center_id != heldout_center_id
            for row in typed
        )
    ):
        raise ProtocolError("HARP v14 heldout-q label-free menu inventory is malformed.")
    return tuple(
        _filter_menu(row, excluded_center_ids)
        for row in sorted(typed, key=lambda value: value.case_id)
    )


def fit_prelabel_pseudo_target_fold(
    observations: Sequence[SourceActionOutcome],
    *,
    heldout_center_id: str,
    heldout_menus: Sequence[EffectiveMenu],
    fixed_excluded_center_ids: Sequence[str],
    effective_menus: Sequence[EffectiveMenu] | None = None,
    config: PairwiseFitConfig = PairwiseFitConfig(),
) -> NestedPolicyFold:
    """Fit and seal one pseudo-target ``q`` fold before opening q outcomes.

    ``observations`` and ``effective_menus`` must contain only centers outside
    the fixed ``{H, q}`` boundary.  The heldout q surface is accepted solely as
    typed, label-free menus.  Calibration predictions for each remaining
    center ``r`` are produced with the additive exclusion ``{H, q, r}``.
    """

    if not isinstance(config, PairwiseFitConfig):
        raise ProtocolError("HARP v14 prelabel fold configuration is malformed.")
    heldout = str(heldout_center_id)
    rows = tuple(observations)
    if any(row.action.query_center_id == heldout for row in rows):
        raise ProtocolError(
            "HARP v14 prelabel fold was given an excluded heldout-q outcome."
        )
    cases = _source_cases(
        rows,
        effective_menus,
        min_centers=config.min_source_centers,
    )
    outer = cases[0].menu.outer_target_id
    raw_fixed = tuple(str(value) for value in fixed_excluded_center_ids)
    fixed = frozenset(raw_fixed)
    if (
        len(raw_fixed) != 2
        or len(fixed) != 2
        or heldout == outer
        or fixed != frozenset((outer, heldout))
    ):
        raise ProtocolError(
            "HARP v14 prelabel fold requires the exact fixed {H,q} exclusions."
        )
    if any(case.menu.query_center_id in fixed for case in cases):
        raise ProtocolError(
            "HARP v14 prelabel fold retained an excluded H/q query outcome."
        )
    training_centers = tuple(
        sorted({case.menu.query_center_id for case in cases})
    )
    fitter = _FixedExclusionFitter(cases, config, fixed)
    q_menus = _heldout_menus(
        heldout_menus,
        outer_target_id=outer,
        heldout_center_id=heldout,
        excluded_center_ids=fixed,
    )
    q_model = fitter.model_for()
    q_predictions = tuple(predict_case(q_model, menu) for menu in q_menus)

    calibration_predictions: list[CasePrediction] = []
    for calibration_center in training_centers:
        excluded = frozenset((*fixed, calibration_center))
        model = fitter.model_for((calibration_center,))
        menus = tuple(
            _filter_menu(case.menu, excluded)
            for case in cases
            if case.menu.query_center_id == calibration_center
        )
        calibration_predictions.extend(predict_case(model, menu) for menu in menus)
    return NestedPolicyFold(
        heldout_center_id=heldout,
        training_center_ids=training_centers,
        predictions=tuple(calibration_predictions),
        heldout_predictions=q_predictions,
    )


def assemble_source_lodo_result(
    observations: Sequence[SourceActionOutcome],
    *,
    presealed_folds: Sequence[NestedPolicyFold],
    nested_outcome_universes: Mapping[str, SourceOutcomeUniverse],
    effective_menus: Sequence[EffectiveMenu] | None = None,
    config_grid: Sequence[PairwiseFitConfig] = (PairwiseFitConfig(),),
) -> SourceLODOResult:
    """Fit only the final target model and bind it to presealed q folds.

    No q OOF or nested calibration prediction is recomputed here.  The full
    source outcomes may therefore be opened only after every q fold has been
    durably sealed by the caller.
    """

    config = _single_config(config_grid)
    all_cases = _source_cases(
        observations,
        effective_menus,
        min_centers=config.min_source_centers,
    )
    outer = all_cases[0].menu.outer_target_id
    source_centers = tuple(
        sorted({case.menu.query_center_id for case in all_cases})
    )
    folds = tuple(sorted(presealed_folds, key=lambda row: row.heldout_center_id))
    nested_universes = {
        str(center): universe
        for center, universe in nested_outcome_universes.items()
    }
    if (
        any(not isinstance(row, NestedPolicyFold) for row in folds)
        or tuple(row.heldout_center_id for row in folds) != source_centers
        or any(row.outer_target_id != outer for row in folds)
        or set(nested_universes) != set(source_centers)
        or any(
            not isinstance(universe, SourceOutcomeUniverse)
            for universe in nested_universes.values()
        )
    ):
        raise ProtocolError("HARP v14 presealed q-fold inventory is incomplete.")
    for fold in folds:
        expected_training = tuple(
            center for center in source_centers if center != fold.heldout_center_id
        )
        excluded = frozenset((outer, fold.heldout_center_id))
        if (
            fold.training_center_ids != expected_training
            or any(
                not set(row.excluded_center_ids).issuperset(excluded)
                for row in (*fold.predictions, *fold.heldout_predictions)
            )
        ):
            raise ProtocolError("HARP v14 presealed q fold is not source-bound.")
        nested_universes[fold.heldout_center_id].bind_predictions(
            (*fold.predictions, *fold.heldout_predictions),
            require_complete=True,
        )

    # This is deliberately the only fit performed by assembly.  Fold models
    # and q predictions are immutable inputs at this boundary.
    final_model = _FixedExclusionFitter(
        all_cases,
        config,
        frozenset((outer,)),
    ).model_for()
    oof_predictions = tuple(
        prediction
        for fold in folds
        for prediction in fold.heldout_predictions
    )
    heldout_hashes = tuple(
        (fold.heldout_center_id, fold.heldout_model_hash) for fold in folds
    )
    return SourceLODOResult(
        outer_target_id=outer,
        final_model=final_model,
        oof_predictions=oof_predictions,
        heldout_model_hashes=heldout_hashes,
        nested_policy_folds=folds,
        nested_outcome_universes=tuple(
            (center, nested_universes[center]) for center in sorted(nested_universes)
        ),
        config=config,
    )


def fit_source_lodo(
    observations: Sequence[SourceActionOutcome],
    *,
    effective_menus: Sequence[EffectiveMenu] | None = None,
    config_grid: Sequence[PairwiseFitConfig] = (PairwiseFitConfig(),),
) -> SourceLODOResult:
    """Reject the projection-based compatibility path removed by HARP v14.

    A valid v14 result requires presealed pseudo-target folds and exact
    per-``q`` outcome universes built from the certified H/q/r physical
    surfaces.  They cannot be reconstructed from the aggregate H/r/r menu,
    so callers must use :func:`assemble_source_lodo_result`.
    """

    del config_grid
    if effective_menus is None:
        raise ProtocolError(
            "HARP v14 fitting requires the explicit effective-menu inventory; "
            "outcomes cannot reconstruct exact-B controls."
        )
    del observations, effective_menus
    raise ProtocolError(
        "HARP v14 fit_source_lodo is disabled; exact presealed H/q folds and "
        "nested outcome universes are required."
    )


def predict_target_actions(
    model: PairwiseResidualRouterModel, menus: Sequence[EffectiveMenu]
) -> tuple[CasePrediction, ...]:
    return tuple(predict_case(model, menu) for menu in menus)


__all__ = (
    "NestedPolicyFold",
    "PairwiseFitConfig",
    "PairwiseResidualRouterModel",
    "SourceLODOResult",
    "assemble_source_lodo_result",
    "fit_prelabel_pseudo_target_fold",
    "fit_source_lodo",
    "predict_case",
    "predict_target_actions",
)
