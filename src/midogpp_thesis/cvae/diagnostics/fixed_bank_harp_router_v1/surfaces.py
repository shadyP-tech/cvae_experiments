"""Neutral HARP surface construction for the consumed-test sensitivity."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.harp_action_surface import (
    HarpActionFeatureSurface,
    HarpProbabilityRow,
    build_action_feature_surface,
    build_probability_ensemble_surface,
    build_probability_surface,
)
from ...runtime.harp_probability_menu import (
    BASE_ACTION_ID,
    DEVELOPMENT_SURFACE,
    EXACT_NINE_SEED_PAIRS,
    UNIFORM_ACTION_ID,
    HarpPredictionMenuSeal,
)
from .target_action_compiler import build_target_actions


def _seed_id(pair: tuple[int, int]) -> str:
    return f"train={pair[0]:010d}::generation={pair[1]:010d}"


SEED_IDS = tuple(sorted(_seed_id(pair) for pair in EXACT_NINE_SEED_PAIRS))


def build_development_feature_surface(
    menu: HarpPredictionMenuSeal,
) -> HarpActionFeatureSurface:
    """Build sample-level exact-nine features before any labels are opened."""

    menu.assert_valid()
    rows: list[HarpProbabilityRow] = []
    by_context: dict[tuple[str, str], list[object]] = defaultdict(list)
    for action in menu.actions:
        if action.surface_kind == DEVELOPMENT_SURFACE:
            by_context[(action.outer_target_id, action.query_center_id)].append(action)
    expected_contexts = {(outer, query) for outer in CENTERS for query in CENTERS if outer != query}
    if set(by_context) != expected_contexts:
        raise ProtocolError("HARP Stage-90 development menu lacks all outer/query contexts.")
    for (outer, query), actions in sorted(by_context.items()):
        exact_b = next((action for action in actions if action.action_id == BASE_ACTION_ID), None)
        reference = next(
            (action for action in actions if action.action_id == UNIFORM_ACTION_ID), None
        )
        if exact_b is None or reference is None:
            raise ProtocolError("HARP Stage-90 development context lacks B or U.")
        exact_b_cells = {
            (cell.training_seed, cell.generation_seed): cell
            for cell in menu.cells_for(exact_b)
        }
        reference_cells = {
            (cell.training_seed, cell.generation_seed): cell
            for cell in menu.cells_for(reference)
        }
        for action in actions:
            if action.selected_source_id is None:
                continue
            assert action.selected_source_id is not None
            if outer in {query, action.selected_source_id} or query == action.selected_source_id:
                raise ProtocolError("HARP Stage-90 outer H escaped role exclusion.")
            for candidate in menu.cells_for(action):
                pair = (candidate.training_seed, candidate.generation_seed)
                base = exact_b_cells[pair]
                ref = reference_cells[pair]
                if (
                    candidate.row_ids != ref.row_ids
                    or candidate.case_ids != ref.case_ids
                    or base.row_ids != ref.row_ids
                    or base.case_ids != ref.case_ids
                ):
                    raise ProtocolError("HARP Stage-90 development B/U/e rows are misaligned.")
                for ordinal, (sample, case) in enumerate(
                    zip(ref.row_ids, ref.case_ids, strict=True)
                ):
                    rows.append(
                        HarpProbabilityRow(
                            outer_target=outer,
                            pseudo_query=query,
                            candidate_source=action.selected_source_id,
                            inner_donor=None,
                            case_id=case,
                            sample_id=sample,
                            seed_id=_seed_id(pair),
                            # The feature/response contract calls this field
                            # baseline; HARP v1 binds it to matched-budget U.
                            baseline_probability=float(ref.probabilities[ordinal]),
                            expert_probability=float(candidate.probabilities[ordinal]),
                            prediction_seal_hash=menu.seal_hash,
                        )
                    )
    seed_surface = build_probability_surface(rows)
    ensemble = build_probability_ensemble_surface(seed_surface, expected_seed_ids=SEED_IDS)
    return build_action_feature_surface(ensemble)


__all__ = ("SEED_IDS", "build_development_feature_surface", "build_target_actions")
