"""Typed case inventories for label-free predictions and source outcomes.

HARP v14 distinguishes a valid empty effective menu from an incomplete
outcome join.  The former is an exact-B control and has no action outcomes;
the latter is a protocol error.  This module is the single implementation of
that distinction for admission, calibration, and durable OOF replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from types import MappingProxyType
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .contracts import CasePrediction, SourceActionOutcome
from .effective_menu import EffectiveMenu
from .hashing import canonical_hash


CaseKey = tuple[str, str, str]


class CaseInventoryKind(str, Enum):
    """The two legal effective-menu states."""

    EXACT_B_CONTROL = "EXACT_B_CONTROL"
    ACTIVE_MENU = "ACTIVE_MENU"


def _case_key(value: EffectiveMenu | CasePrediction) -> CaseKey:
    return (value.outer_target_id, value.query_center_id, value.case_id)


@dataclass(frozen=True, slots=True)
class LabelFreeCaseContext:
    """One menu/prediction binding created without source or target labels."""

    menu: EffectiveMenu
    prediction: CasePrediction
    kind: CaseInventoryKind
    context_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.menu, EffectiveMenu) or not isinstance(
            self.prediction, CasePrediction
        ):
            raise ProtocolError("HARP v14 case context requires typed menu/prediction.")
        if _case_key(self.menu) != _case_key(self.prediction):
            raise ProtocolError("HARP v14 case context crossed H/q/case identity.")
        excluded = frozenset(self.prediction.excluded_center_ids)
        if any(
            action.candidate_source_id is not None
            and action.candidate_source_id in excluded
            for action in self.menu.actions
        ):
            raise ProtocolError(
                "HARP v14 exact fold-local menu retained an excluded candidate."
            )
        if self.prediction.menu_hash != self.menu.menu_hash:
            raise ProtocolError("HARP v14 case prediction/menu hash drifted.")
        menu_actions = {
            action.action_id: action.action_hash for action in self.menu.actions
        }
        score_actions = {
            score.action_id: score.action_hash
            for score in self.prediction.action_scores
        }
        if score_actions != menu_actions:
            raise ProtocolError(
                "HARP v14 prediction scores do not exactly cover the effective menu."
            )
        expected_kind = (
            CaseInventoryKind.ACTIVE_MENU
            if self.menu.actions
            else CaseInventoryKind.EXACT_B_CONTROL
        )
        if self.kind is not expected_kind:
            raise ProtocolError("HARP v14 case inventory kind disagrees with its menu.")
        if self.kind is CaseInventoryKind.EXACT_B_CONTROL and (
            self.prediction.action_scores
            or self.prediction.raw_top_action_id != "B"
            or self.prediction.top_action_id != "B"
            or self.prediction.acceptance_probability != 0.0
            or self.prediction.rank_margin != 0.0
        ):
            raise ProtocolError("HARP v14 empty menu is not an exact-B control.")
        object.__setattr__(
            self,
            "context_hash",
            canonical_hash(
                {
                    "schema_version": "policy_calibrated_label_free_case_context_v14",
                    "case_key": _case_key(self.menu),
                    "kind": self.kind.value,
                    "effective_menu_hash": self.menu.menu_hash,
                    "action_ids": tuple(menu_actions),
                    "action_hashes": tuple(menu_actions.values()),
                    "prediction_hash": self.prediction.prediction_hash,
                    "source_outcomes_consumed": False,
                    "evaluation_labels_consumed": False,
                }
            ),
        )

    @property
    def key(self) -> CaseKey:
        return _case_key(self.menu)

    @property
    def is_exact_b_control(self) -> bool:
        return self.kind is CaseInventoryKind.EXACT_B_CONTROL

    def public_payload(self) -> dict[str, object]:
        return {
            "outer_target_id": self.key[0],
            "query_center_id": self.key[1],
            "case_id": self.key[2],
            "kind": self.kind.value,
            "effective_menu_hash": self.menu.menu_hash,
            "action_ids": [action.action_id for action in self.menu.actions],
            "action_hashes": [action.action_hash for action in self.menu.actions],
            "prediction_hash": self.prediction.prediction_hash,
            "context_hash": self.context_hash,
        }


@dataclass(frozen=True, slots=True)
class LabelFreeCaseInventory:
    """Deterministic, label-free prediction inventory."""

    contexts: tuple[LabelFreeCaseContext, ...]
    complete_for_menu_universe: bool
    inventory_hash: str = field(init=False)

    def __post_init__(self) -> None:
        contexts = tuple(sorted(self.contexts, key=lambda row: row.key))
        if (
            not contexts
            or contexts != self.contexts
            or len({row.key for row in contexts}) != len(contexts)
        ):
            raise ProtocolError("HARP v14 label-free case inventory is malformed.")
        object.__setattr__(
            self,
            "inventory_hash",
            canonical_hash(
                {
                    "schema_version": "policy_calibrated_label_free_case_inventory_v14",
                    "context_hashes": tuple(row.context_hash for row in contexts),
                    "case_count": len(contexts),
                    "exact_b_control_count": sum(
                        row.is_exact_b_control for row in contexts
                    ),
                    "complete_for_menu_universe": self.complete_for_menu_universe,
                    "source_outcomes_consumed": False,
                    "evaluation_labels_consumed": False,
                }
            ),
        )

    def by_key(self) -> dict[CaseKey, LabelFreeCaseContext]:
        return {row.key: row for row in self.contexts}

    def public_payload(self) -> dict[str, object]:
        return {
            "schema_version": "policy_calibrated_label_free_case_inventory_v14",
            "case_count": len(self.contexts),
            "exact_b_control_count": sum(
                row.is_exact_b_control for row in self.contexts
            ),
            "active_menu_count": sum(
                not row.is_exact_b_control for row in self.contexts
            ),
            "complete_for_menu_universe": self.complete_for_menu_universe,
            "contexts": [row.public_payload() for row in self.contexts],
            "inventory_hash": self.inventory_hash,
            "source_outcomes_consumed": False,
            "evaluation_labels_consumed": False,
        }


@dataclass(frozen=True, slots=True)
class CaseOutcomeContext:
    """One label-free context joined to its exact source action outcomes."""

    label_free: LabelFreeCaseContext
    outcomes: tuple[SourceActionOutcome, ...]
    context_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outcomes = tuple(
            sorted(self.outcomes, key=lambda row: row.action.action_id)
        )
        expected = {
            action.action_id: action.action_hash
            for action in self.label_free.menu.actions
        }
        observed = {
            row.action.action_id: row.action.action_hash for row in outcomes
        }
        if (
            len(observed) != len(outcomes)
            or observed != expected
            or any(
                (
                    row.action.outer_target_id,
                    row.action.query_center_id,
                    row.action.case_id,
                )
                != self.label_free.key
                for row in outcomes
            )
        ):
            raise ProtocolError(
                "HARP v14 source outcomes do not exactly cover the case menu."
            )
        if self.label_free.is_exact_b_control and outcomes:
            raise ProtocolError("HARP v14 exact-B control cannot carry action outcomes.")
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(
            self,
            "context_hash",
            canonical_hash(
                {
                    "schema_version": "policy_calibrated_case_outcome_context_v14",
                    "label_free_context_hash": self.label_free.context_hash,
                    "outcome_hashes": tuple(row.outcome_hash for row in outcomes),
                    "exact_b_gain": 0.0,
                    "evaluation_labels_consumed": False,
                }
            ),
        )

    @property
    def key(self) -> CaseKey:
        return self.label_free.key

    @property
    def menu(self) -> EffectiveMenu:
        return self.label_free.menu

    @property
    def prediction(self) -> CasePrediction:
        return self.label_free.prediction

    @property
    def is_exact_b_control(self) -> bool:
        return self.label_free.is_exact_b_control

    @property
    def best_bacc_gain(self) -> float:
        return max((0.0, *(row.bacc_gain for row in self.outcomes)))

    def outcome_for(self, action_id: str) -> SourceActionOutcome | None:
        if action_id == "B":
            return None
        return next(
            (row for row in self.outcomes if row.action.action_id == action_id),
            None,
        )


@dataclass(frozen=True, slots=True)
class CaseOutcomeInventory:
    """Strict source-label join over a previously label-free inventory."""

    label_free_inventory_hash: str
    contexts: tuple[CaseOutcomeContext, ...]
    inventory_hash: str = field(init=False)

    def __post_init__(self) -> None:
        contexts = tuple(sorted(self.contexts, key=lambda row: row.key))
        if (
            not contexts
            or contexts != self.contexts
            or len({row.key for row in contexts}) != len(contexts)
        ):
            raise ProtocolError("HARP v14 source case-outcome inventory is malformed.")
        object.__setattr__(
            self,
            "inventory_hash",
            canonical_hash(
                {
                    "schema_version": "policy_calibrated_case_outcome_inventory_v14",
                    "label_free_inventory_hash": self.label_free_inventory_hash,
                    "context_hashes": tuple(row.context_hash for row in contexts),
                    "case_count": len(contexts),
                    "exact_b_control_count": sum(
                        row.is_exact_b_control for row in contexts
                    ),
                    "active_menu_count": sum(
                        not row.is_exact_b_control for row in contexts
                    ),
                    "evaluation_labels_consumed": False,
                }
            ),
        )

    def by_key(self) -> dict[CaseKey, CaseOutcomeContext]:
        return {row.key: row for row in self.contexts}


@dataclass(frozen=True, slots=True)
class SourceOutcomeUniverse:
    """One validated base-menu/outcome surface reusable across nested folds."""

    effective_menus: tuple[EffectiveMenu, ...]
    outcomes: tuple[SourceActionOutcome, ...]
    universe_hash: str = field(init=False)
    _outcomes_by_case: Mapping[CaseKey, Mapping[str, SourceActionOutcome]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        menus = tuple(sorted(self.effective_menus, key=_case_key))
        outcomes = tuple(
            sorted(
                self.outcomes,
                key=lambda row: (
                    row.action.outer_target_id,
                    row.action.query_center_id,
                    row.action.case_id,
                    row.action.action_id,
                ),
            )
        )
        indexed = _validated_outcomes_by_case(outcomes, menus)
        immutable = MappingProxyType(
            {
                key: MappingProxyType(dict(values))
                for key, values in indexed.items()
            }
        )
        object.__setattr__(self, "effective_menus", menus)
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "_outcomes_by_case", immutable)
        object.__setattr__(
            self,
            "universe_hash",
            canonical_hash(
                {
                    "schema_version": "policy_calibrated_source_outcome_universe_v14",
                    "effective_menu_hashes": tuple(row.menu_hash for row in menus),
                    "source_outcome_hashes": tuple(row.outcome_hash for row in outcomes),
                    "exact_b_gain": 0.0,
                    "evaluation_labels_consumed": False,
                }
            ),
        )

    def label_free_inventory(
        self,
        predictions: Sequence[CasePrediction],
        *,
        require_complete: bool = False,
    ) -> LabelFreeCaseInventory:
        return build_label_free_case_inventory(
            predictions,
            self.effective_menus,
            require_complete=require_complete,
        )

    def join(self, label_free: LabelFreeCaseInventory) -> CaseOutcomeInventory:
        if not isinstance(label_free, LabelFreeCaseInventory):
            raise ProtocolError("HARP v14 outcome join requires a label-free inventory.")
        contexts: list[CaseOutcomeContext] = []
        for context in label_free.contexts:
            try:
                base = self._outcomes_by_case[context.key]
            except KeyError as exc:
                raise ProtocolError(
                    "HARP v14 prediction escaped the source outcome universe."
                ) from exc
            expected_ids = {action.action_id for action in context.menu.actions}
            projected = tuple(
                row for action_id, row in base.items() if action_id in expected_ids
            )
            contexts.append(CaseOutcomeContext(context, projected))
        return CaseOutcomeInventory(label_free.inventory_hash, tuple(contexts))

    def bind_predictions(
        self,
        predictions: Sequence[CasePrediction],
        *,
        require_complete: bool = False,
    ) -> CaseOutcomeInventory:
        return self.join(
            self.label_free_inventory(
                predictions,
                require_complete=require_complete,
            )
        )


def build_label_free_case_inventory(
    predictions: Sequence[CasePrediction],
    effective_menus: Sequence[EffectiveMenu],
    *,
    require_complete: bool = False,
) -> LabelFreeCaseInventory:
    """Bind predictions to their exact fold-local menus without opening labels.

    The menu supplied for a case must be the same physical H/q/r menu that was
    passed to the predictor.  In particular, this boundary must never attempt
    to derive an H/q/r menu by deleting candidates from an H/r/r menu: menu
    construction also fixes candidate-pool compatibility, physical action
    bytes, no-op removal, and duplicate representatives.  Those semantics are
    not recoverable from ``excluded_center_ids`` after prediction.
    """

    typed_predictions = tuple(predictions)
    typed_menus = tuple(effective_menus)
    if (
        not typed_predictions
        or not typed_menus
        or any(not isinstance(row, CasePrediction) for row in typed_predictions)
        or any(not isinstance(row, EffectiveMenu) for row in typed_menus)
    ):
        raise ProtocolError("HARP v14 case inventory requires typed predictions/menus.")
    menus_by_key = {_case_key(row): row for row in typed_menus}
    prediction_keys = tuple(_case_key(row) for row in typed_predictions)
    if len(menus_by_key) != len(typed_menus) or len(set(prediction_keys)) != len(
        prediction_keys
    ):
        raise ProtocolError("HARP v14 case inventory contains duplicate identities.")
    if any(key not in menus_by_key for key in prediction_keys):
        raise ProtocolError("HARP v14 prediction escaped the sealed menu universe.")
    if require_complete and set(prediction_keys) != set(menus_by_key):
        raise ProtocolError("HARP v14 prediction/menu case inventory is incomplete.")

    contexts: list[LabelFreeCaseContext] = []
    for prediction in sorted(typed_predictions, key=_case_key):
        exact = menus_by_key[_case_key(prediction)]
        kind = (
            CaseInventoryKind.ACTIVE_MENU
            if exact.actions
            else CaseInventoryKind.EXACT_B_CONTROL
        )
        contexts.append(LabelFreeCaseContext(exact, prediction, kind))
    return LabelFreeCaseInventory(
        tuple(contexts), complete_for_menu_universe=require_complete
    )


def _validated_outcomes_by_case(
    observations: Sequence[SourceActionOutcome],
    effective_menus: Sequence[EffectiveMenu],
) -> dict[CaseKey, dict[str, SourceActionOutcome]]:
    menus = tuple(effective_menus)
    outcomes = tuple(observations)
    menus_by_key = {_case_key(row): row for row in menus}
    if not menus or len(menus_by_key) != len(menus):
        raise ProtocolError("HARP v14 source outcome menu universe is malformed.")
    by_case: dict[CaseKey, dict[str, SourceActionOutcome]] = {
        key: {} for key in menus_by_key
    }
    for row in outcomes:
        if not isinstance(row, SourceActionOutcome):
            raise ProtocolError("HARP v14 source outcome inventory is untyped.")
        key = (
            row.action.outer_target_id,
            row.action.query_center_id,
            row.action.case_id,
        )
        scoped = by_case.get(key)
        if scoped is None or row.action.action_id in scoped:
            raise ProtocolError("HARP v14 source outcome identity is extra or duplicated.")
        scoped[row.action.action_id] = row
    for key, menu in menus_by_key.items():
        expected = {
            action.action_id: action.action_hash for action in menu.actions
        }
        observed = {
            action_id: row.action.action_hash
            for action_id, row in by_case[key].items()
        }
        if observed != expected:
            raise ProtocolError(
                "HARP v14 active-menu source outcome inventory is incomplete or drifted."
            )
    return by_case


def join_source_outcomes(
    label_free: LabelFreeCaseInventory,
    observations: Sequence[SourceActionOutcome],
    effective_menus: Sequence[EffectiveMenu],
) -> CaseOutcomeInventory:
    """Join source outcomes after validating the complete base-menu surface."""

    return SourceOutcomeUniverse(tuple(effective_menus), tuple(observations)).join(
        label_free
    )


def build_case_outcome_inventory(
    predictions: Sequence[CasePrediction],
    observations: Sequence[SourceActionOutcome],
    effective_menus: Sequence[EffectiveMenu],
    *,
    require_complete: bool = False,
) -> CaseOutcomeInventory:
    """Convenience constructor used by source-only policy consumers."""

    return SourceOutcomeUniverse(
        tuple(effective_menus), tuple(observations)
    ).bind_predictions(predictions, require_complete=require_complete)


__all__ = (
    "CaseInventoryKind",
    "CaseKey",
    "CaseOutcomeContext",
    "CaseOutcomeInventory",
    "LabelFreeCaseContext",
    "LabelFreeCaseInventory",
    "SourceOutcomeUniverse",
    "build_case_outcome_inventory",
    "build_label_free_case_inventory",
    "join_source_outcomes",
)
