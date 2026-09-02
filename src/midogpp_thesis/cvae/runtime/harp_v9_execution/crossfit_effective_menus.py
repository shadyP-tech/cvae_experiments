"""Label-free H/q/r adapter from physical blocks to case EffectiveMenus."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ...routing.policy_calibrated_residual_router_v9 import (
    Direction,
    EffectiveMenu,
    LabelFreeAction,
    group_effective_menus,
    SourceActionOutcome,
)
from ...routing.harp_protocol import HarpSourceLabelRow
from .contracts import ActionKind, LabelFreeActionBlock
from .crossfit_actions import BASE_ACTION_ID, UNIFORM_ACTION_ID
from .crossfit_contracts import (
    FoldConditionedActionBlock,
    FoldConditionedCompatibility,
    FoldConditionedSourceSurface,
)
from .crossfit_durability import SourceCrossfitLabelCapability
from .directional_surfaces import (
    DIRECTIONS,
    _base_features,
    _case_indices,
    _case_order,
    _probability_cells,
    directional_probability_bytes,
    attach_source_outcomes,
)
from .geometry_features import GEOMETRY_FEATURE_NAMES, geometry_feature_values


_COMPATIBILITY_FEATURE_NAMES = (
    "compatibility_mean_z",
    "compatibility_std_z",
    "compatibility_reciprocal_rank",
    "compatibility_rank_margin",
    "compatibility_available",
)


@dataclass(frozen=True, slots=True)
class FoldConditionedEffectiveMenu:
    """An ordinary EffectiveMenu guarded by its held-out ``q`` identity."""

    outer_target_id: str
    heldout_center_id: str
    current_query_center_id: str
    menu: EffectiveMenu
    candidate_source_ids: tuple[str, ...]
    physical_block_hashes: tuple[str, ...]
    compatibility_receipt_hashes: tuple[str, ...]
    fold_menu_hash: str = field(init=False)

    def __post_init__(self) -> None:
        h = str(self.outer_target_id)
        q = str(self.heldout_center_id)
        r = str(self.current_query_center_id)
        candidates = tuple(str(value) for value in self.candidate_source_ids)
        expected = tuple(center for center in CENTERS if center not in {h, q, r})
        if (
            h not in CENTERS
            or q not in CENTERS
            or r not in CENTERS
            or h == q
            or h == r
            or candidates != expected
            or not isinstance(self.menu, EffectiveMenu)
            or self.menu.outer_target_id != h
            or self.menu.query_center_id != r
            or any(
                action.candidate_source_id is not None
                and action.candidate_source_id not in candidates
                for action in self.menu.actions
            )
        ):
            raise ProtocolError("HARP v9 effective H/q/r menu escaped its fold.")
        block_hashes = tuple(str(value) for value in self.physical_block_hashes)
        compatibility_hashes = tuple(
            str(value) for value in self.compatibility_receipt_hashes
        )
        if (
            len(block_hashes) != 2 + len(candidates)
            or len(set(block_hashes)) != len(block_hashes)
            or len(compatibility_hashes) != len(candidates)
            or len(set(compatibility_hashes)) != len(compatibility_hashes)
            or any(len(value) != 64 for value in (*block_hashes, *compatibility_hashes))
        ):
            raise ProtocolError("HARP v9 effective H/q/r lineage is malformed.")
        body = {
            "schema_version": "midogpp_harp_v9_fold_conditioned_effective_menu_v1",
            "outer_target_id": h,
            "heldout_center_id": q,
            "current_query_center_id": r,
            "case_id": self.menu.case_id,
            "candidate_source_ids": list(candidates),
            "effective_menu_hash": self.menu.menu_hash,
            "physical_block_hashes": list(block_hashes),
            "compatibility_receipt_hashes": list(compatibility_hashes),
            "heldout_q_in_identity": True,
            "source_pool_semantics": "C_MINUS_H_MINUS_Q_MINUS_R",
            "labels_consumed": False,
        }
        object.__setattr__(self, "outer_target_id", h)
        object.__setattr__(self, "heldout_center_id", q)
        object.__setattr__(self, "current_query_center_id", r)
        object.__setattr__(self, "candidate_source_ids", candidates)
        object.__setattr__(self, "physical_block_hashes", block_hashes)
        object.__setattr__(self, "compatibility_receipt_hashes", compatibility_hashes)
        object.__setattr__(self, "fold_menu_hash", canonical_hash(body))

    @property
    def prediction_fold(self) -> bool:
        return self.heldout_center_id == self.current_query_center_id


@dataclass(frozen=True, slots=True)
class FoldConditionedEffectiveSurface:
    source_surface_hash: str
    menus: tuple[FoldConditionedEffectiveMenu, ...]
    adapter_hash: str = field(init=False)

    def __post_init__(self) -> None:
        source_hash = str(self.source_surface_hash)
        menus = tuple(
            sorted(
                self.menus,
                key=lambda row: (
                    row.outer_target_id,
                    row.heldout_center_id,
                    row.current_query_center_id,
                    row.menu.case_id,
                ),
            )
        )
        if (
            len(source_hash) != 64
            or menus != self.menus
            or len({row.fold_menu_hash for row in menus}) != len(menus)
        ):
            raise ProtocolError("HARP v9 effective crossfit surface drifted.")
        body = {
            "schema_version": "midogpp_harp_v9_fold_conditioned_effective_surface_v1",
            "source_surface_hash": source_hash,
            "fold_menu_hashes": [row.fold_menu_hash for row in menus],
            "heldout_identity_preserved": True,
            "labels_consumed": False,
        }
        object.__setattr__(self, "source_surface_hash", source_hash)
        object.__setattr__(self, "menus", menus)
        object.__setattr__(self, "adapter_hash", canonical_hash(body))

    def prediction_menus(
        self, outer_target_id: str, heldout_center_id: str
    ) -> tuple[FoldConditionedEffectiveMenu, ...]:
        key = (str(outer_target_id), str(heldout_center_id))
        rows = tuple(
            row
            for row in self.menus
            if (row.outer_target_id, row.heldout_center_id) == key
            and row.current_query_center_id == row.heldout_center_id
        )
        if not rows:
            raise ProtocolError(f"HARP v9 prediction fold is absent: {key}.")
        return rows

    def fitting_menus(
        self, outer_target_id: str, heldout_center_id: str
    ) -> tuple[FoldConditionedEffectiveMenu, ...]:
        key = (str(outer_target_id), str(heldout_center_id))
        rows = tuple(
            row
            for row in self.menus
            if (row.outer_target_id, row.heldout_center_id) == key
            and row.current_query_center_id != row.heldout_center_id
        )
        if not rows:
            raise ProtocolError(f"HARP v9 fitting fold is absent: {key}.")
        if any(
            row.heldout_center_id in row.candidate_source_ids
            or row.current_query_center_id in row.candidate_source_ids
            for row in rows
        ):
            raise ProtocolError("HARP v9 fitting menus leaked q/r candidates.")
        return rows


@dataclass(frozen=True, slots=True)
class FoldConditionedSourceOutcomeSet:
    outer_target_id: str
    heldout_center_id: str
    source_surface_hash: str
    effective_adapter_hash: str
    fold_menu_hashes: tuple[str, ...]
    label_capability_hash: str
    outcomes: tuple[SourceActionOutcome, ...]
    outcome_set_hash: str = field(init=False)

    def __post_init__(self) -> None:
        h = str(self.outer_target_id)
        q = str(self.heldout_center_id)
        outcomes = tuple(
            sorted(
                self.outcomes,
                key=lambda row: (
                    row.action.query_center_id,
                    row.action.case_id,
                    row.action.action_id,
                ),
            )
        )
        if (
            h not in CENTERS
            or q not in CENTERS
            or h == q
            or not outcomes
            or any(
                row.action.outer_target_id != h
                or row.action.query_center_id in {h, q}
                for row in outcomes
            )
            or any(
                len(value) != 64
                for value in (
                    self.source_surface_hash,
                    self.effective_adapter_hash,
                    *self.fold_menu_hashes,
                    self.label_capability_hash,
                )
            )
        ):
            raise ProtocolError("HARP v9 source outcome fold escaped H/q.")
        body = {
            "schema_version": "midogpp_harp_v9_fold_conditioned_source_outcomes_v1",
            "outer_target_id": h,
            "heldout_center_id": q,
            "source_surface_hash": self.source_surface_hash,
            "effective_adapter_hash": self.effective_adapter_hash,
            "fold_menu_hashes": list(self.fold_menu_hashes),
            "label_capability_hash": self.label_capability_hash,
            "outcome_hashes": [row.outcome_hash for row in outcomes],
            "label_scope": "C_MINUS_H_MINUS_Q",
            "prediction_q_labels_consumed": False,
            "evaluation_labels_consumed": False,
        }
        object.__setattr__(self, "outer_target_id", h)
        object.__setattr__(self, "heldout_center_id", q)
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "outcome_set_hash", canonical_hash(body))


def build_fold_conditioned_effective_surface(
    surface: FoldConditionedSourceSurface,
) -> FoldConditionedEffectiveSurface:
    """Compile case menus without losing the held-out fold identity."""

    if not isinstance(surface, FoldConditionedSourceSurface):
        raise ProtocolError("HARP v9 effective adapter requires a typed source surface.")
    by_context: dict[
        tuple[str, str, str], tuple[FoldConditionedActionBlock, ...]
    ] = {}
    for h in surface.outer_target_ids:
        for q in CENTERS:
            if q == h:
                continue
            for r in CENTERS:
                if r == h:
                    continue
                by_context[(h, q, r)] = surface.blocks_for(h, q, r)
    compat_by_context: dict[
        tuple[str, str, str, str], dict[str, FoldConditionedCompatibility]
    ] = {}
    for row in surface.compatibility:
        compat_by_context.setdefault(
            (
                row.outer_target_id,
                row.heldout_center_id,
                row.current_query_center_id,
                row.case_id,
            ),
            {},
        )[row.candidate_source_id] = row

    output: list[FoldConditionedEffectiveMenu] = []
    for key in sorted(by_context):
        blocks = by_context[key]
        typed_by_action = {row.action.action_id: row for row in blocks}
        baseline = typed_by_action.get(BASE_ACTION_ID)
        if baseline is None:
            raise ProtocolError("HARP v9 crossfit effective context lacks B.")
        physical = {
            action_id: _physical_block(row) for action_id, row in typed_by_action.items()
        }
        baseline_physical = physical[BASE_ACTION_ID]
        candidates = tuple(center for center in CENTERS if center not in set(key))
        raw_actions: list[LabelFreeAction] = []
        for case_id in _case_order(baseline_physical):
            compatibility = compat_by_context.get((*key, case_id), {})
            if tuple(sorted(compatibility)) != candidates:
                raise ProtocolError(
                    "HARP v9 case-local crossfit compatibility pool drifted."
                )
            indices = _case_indices(baseline_physical, case_id)
            baseline_cells = _probability_cells(
                baseline_physical.probabilities[indices]
            )
            for action_id in sorted(
                action_id for action_id in typed_by_action if action_id != BASE_ACTION_ID
            ):
                typed = typed_by_action[action_id]
                challenger = physical[action_id]
                kind = "U" if action_id == UNIFORM_ACTION_ID else "HXE"
                candidate = typed.action.selected_source_id
                receipt = compatibility.get(candidate) if candidate is not None else None
                for direction in DIRECTIONS:
                    _, routed = directional_probability_bytes(
                        baseline_physical,
                        challenger,
                        case_id=case_id,
                        direction=direction,
                    )
                    base = dict(
                        _base_features(
                            baseline_physical,
                            challenger,
                            case_id=case_id,
                            direction=direction,
                            surface=routed,
                        )
                    )
                    density = geometry_feature_values(
                        candidate_count=len(candidates),
                        action_kind=kind,
                        context_kind=(
                            "source_prediction"
                            if key[1] == key[2]
                            else "source_calibration"
                        ),
                    )
                    compatibility_values = (
                        (0.0, 0.0, 0.0, 0.0, 0.0)
                        if receipt is None
                        else (
                            receipt.mean_z,
                            receipt.std_z,
                            1.0 / float(receipt.rank),
                            receipt.rank_margin,
                            1.0,
                        )
                    )
                    names = (
                        *tuple(base),
                        *GEOMETRY_FEATURE_NAMES,
                        *_COMPATIBILITY_FEATURE_NAMES,
                    )
                    values = (
                        *tuple(base.values()),
                        *density,
                        *compatibility_values,
                    )
                    raw_actions.append(
                        LabelFreeAction(
                            outer_target_id=key[0],
                            query_center_id=key[2],
                            case_id=case_id,
                            action_id=(
                                f"U:{direction.value}"
                                if kind == "U"
                                else f"HXE:{candidate}:{direction.value}"
                            ),
                            action_kind=kind,
                            direction=Direction(direction),
                            candidate_source_id=candidate,
                            feature_names=names,
                            feature_values=values,
                            baseline_probability_hex=tuple(
                                value.hex() for value in baseline_cells
                            ),
                            action_probability_hex=tuple(value.hex() for value in routed),
                        )
                    )
        effective = group_effective_menus(raw_actions)
        block_hashes = tuple(row.block_hash for row in blocks)
        output.extend(
            FoldConditionedEffectiveMenu(
                outer_target_id=key[0],
                heldout_center_id=key[1],
                current_query_center_id=key[2],
                menu=menu,
                candidate_source_ids=candidates,
                physical_block_hashes=block_hashes,
                compatibility_receipt_hashes=tuple(
                    compat_by_context[(*key, menu.case_id)][source].receipt_hash
                    for source in candidates
                ),
            )
            for menu in effective
        )
    return FoldConditionedEffectiveSurface(
        source_surface_hash=surface.surface_hash,
        menus=tuple(output),
    )


def attach_fold_conditioned_source_outcomes(
    effective_surface: FoldConditionedEffectiveSurface,
    physical_surface: FoldConditionedSourceSurface,
    *,
    outer_target_id: str,
    heldout_center_id: str,
    label_capability: SourceCrossfitLabelCapability,
    source_label_rows: Sequence[HarpSourceLabelRow],
) -> FoldConditionedSourceOutcomeSet:
    """Join only C-{H,q} source labels to physical r!=q fitting menus."""

    if (
        not isinstance(effective_surface, FoldConditionedEffectiveSurface)
        or not isinstance(physical_surface, FoldConditionedSourceSurface)
        or effective_surface.source_surface_hash != physical_surface.surface_hash
    ):
        raise ProtocolError("HARP v9 source outcome join crossed physical surfaces.")
    h = str(outer_target_id)
    q = str(heldout_center_id)
    _validate_label_capability(
        label_capability,
        physical_surface=physical_surface,
        outer_target_id=h,
        heldout_center_id=q,
    )
    allowed = tuple(center for center in CENTERS if center not in {h, q})
    rows = tuple(source_label_rows)
    if (
        h not in CENTERS
        or q not in CENTERS
        or h == q
        or not rows
        or any(not isinstance(row, HarpSourceLabelRow) for row in rows)
        or any(row.center not in allowed for row in rows)
        or len({row.row_key for row in rows}) != len(rows)
    ):
        raise ProtocolError("HARP v9 source labels escaped exact C-{H,q} scope.")
    label_index = {row.row_key: row.label for row in rows}
    expected_keys: set[tuple[str, str, str]] = set()
    outcomes: list[SourceActionOutcome] = []
    fitting = effective_surface.fitting_menus(h, q)
    fold_hashes = tuple(row.fold_menu_hash for row in fitting)
    for query in allowed:
        physical_blocks = physical_surface.blocks_for(h, q, query)
        baseline_rows = tuple(
            row for row in physical_blocks if row.action.action_id == BASE_ACTION_ID
        )
        if len(baseline_rows) != 1:
            raise ProtocolError("HARP v9 fitting fold lacks exact physical B.")
        baseline = _physical_block(baseline_rows[0])
        keys = {
            (query, case_id, sample_id)
            for case_id, sample_id in zip(
                baseline.case_ids, baseline.sample_ids, strict=True
            )
        }
        expected_keys.update(keys)
        try:
            scoped = {
                (case_id, sample_id): label_index[(query, case_id, sample_id)]
                for _, case_id, sample_id in keys
            }
        except KeyError as exc:
            raise ProtocolError(
                "HARP v9 source labels do not cover an H/q/r physical fold."
            ) from exc
        menus = tuple(
            row.menu for row in fitting if row.current_query_center_id == query
        )
        outcomes.extend(
            attach_source_outcomes(menus, baseline, source_labels=scoped)
        )
    if set(label_index) != expected_keys:
        raise ProtocolError("HARP v9 source labels exceed or omit C-{H,q} folds.")
    return FoldConditionedSourceOutcomeSet(
        outer_target_id=h,
        heldout_center_id=q,
        source_surface_hash=physical_surface.surface_hash,
        effective_adapter_hash=effective_surface.adapter_hash,
        fold_menu_hashes=fold_hashes,
        label_capability_hash=label_capability.capability_hash,
        outcomes=tuple(outcomes),
    )


def _validate_label_capability(
    capability: object,
    *,
    physical_surface: FoldConditionedSourceSurface,
    outer_target_id: str,
    heldout_center_id: str,
) -> None:
    if (
        not isinstance(capability, SourceCrossfitLabelCapability)
        or capability.surface_receipt.surface_hash != physical_surface.surface_hash
        or capability.outer_target_id != outer_target_id
        or capability.heldout_center_id != heldout_center_id
        or capability.authorized_source_center_ids
        != tuple(
            center
            for center in CENTERS
            if center not in {outer_target_id, heldout_center_id}
        )
    ):
        raise ProtocolError("HARP v9 source label capability is absent or cross-bound.")


def _physical_block(row: FoldConditionedActionBlock) -> LabelFreeActionBlock:
    action = row.action
    kind = (
        ActionKind.B
        if action.action_id == BASE_ACTION_ID
        else ActionKind.U
        if action.action_id == UNIFORM_ACTION_ID
        else ActionKind.HXE
    )
    return LabelFreeActionBlock(
        surface_role="development",
        outer_target_id=action.outer_target_id,
        query_center_id=action.current_query_center_id,
        action_kind=kind,
        selected_source_id=action.selected_source_id,
        sample_ids=row.sample_ids,
        case_ids=row.case_ids,
        probabilities=row.probabilities,
        seed_dispersion=row.seed_dispersion,
    )


__all__ = (
    "FoldConditionedEffectiveMenu",
    "FoldConditionedEffectiveSurface",
    "FoldConditionedSourceOutcomeSet",
    "attach_fold_conditioned_source_outcomes",
    "build_fold_conditioned_effective_surface",
)
