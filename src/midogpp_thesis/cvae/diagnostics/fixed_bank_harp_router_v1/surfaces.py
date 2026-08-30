"""Neutral HARP surface construction for the consumed-test sensitivity."""

from __future__ import annotations

from collections import defaultdict
import statistics
import struct

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.harp_action_model import HarpTargetAction, LAMBDA_GRID
from ...routing.harp_action_surface import (
    ACTION_FEATURE_NAMES,
    HarpActionFeatureSurface,
    HarpProbabilityRow,
    build_action_feature_surface,
    build_probability_ensemble_surface,
    build_probability_surface,
)
from ...routing.harp_protocol import canonical_hash
from ...runtime.harp_probability_menu import (
    BASE_ACTION_ID,
    DEVELOPMENT_SURFACE,
    EXACT_NINE_SEED_PAIRS,
    TARGET_SURFACE,
    UNIFORM_ACTION_ID,
    HarpPredictionMenuSeal,
)
from ...runtime.harp_probability_menu.hashing import raw_array_sha256


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


def _feature_values(
    baseline_members: np.ndarray,
    expert_members: np.ndarray,
    baseline: float,
    expert: float,
    lam: float,
) -> tuple[float, ...]:
    action = (1.0 - lam) * baseline + lam * expert
    member_actions = (1.0 - lam) * baseline_members + lam * expert_members
    expert_flips = float(
        np.mean((expert_members >= 0.5) != (baseline_members >= 0.5), dtype=np.float64)
    )
    action_flips = float(
        np.mean((member_actions >= 0.5) != (baseline_members >= 0.5), dtype=np.float64)
    )
    dispersion = float(
        statistics.pstdev(
            tuple(float(value) for value in expert_members - baseline_members)
        )
    )
    return (
        baseline,
        expert,
        action,
        abs(baseline - 0.5),
        abs(expert - 0.5),
        abs(action - 0.5),
        expert - baseline,
        abs(expert - baseline),
        action - baseline,
        abs(action - baseline),
        expert_flips,
        action_flips,
        dispersion,
        lam,
    )


def _direction(baseline: float, action: float) -> str:
    before, after = baseline >= 0.5, action >= 0.5
    if not before and after:
        return "D01"
    if before and not after:
        return "D10"
    return "ALL_MARGINS"


def _ensemble_receipt(
    menu: HarpPredictionMenuSeal,
    *,
    center: str,
    ordinal: int,
    sample_id: str,
    case_id: str,
) -> str:
    actions = tuple(
        action
        for action in menu.actions
        if action.surface_kind == TARGET_SURFACE
        and action.outer_target_id == center
        and action.query_center_id == center
    )
    members = []
    for action in actions:
        values = np.asarray(
            [cell.probabilities[ordinal] for cell in menu.cells_for(action)],
            dtype=np.float32,
        )
        members.append(
            {
                "action_hash": action.action_hash,
                "member_probability_bytes_sha256": raw_array_sha256(values),
            }
        )
    return canonical_hash(
        {
            "schema_version": "midogpp_harp_stage90_target_exact_nine_receipt_v1",
            "prediction_menu_seal_hash": menu.seal_hash,
            "outer_target_id": center,
            "sample_id": sample_id,
            "case_id": case_id,
            "seed_pairs": [list(pair) for pair in EXACT_NINE_SEED_PAIRS],
            "actions": members,
            "labels_consumed": False,
        }
    )


def build_target_actions(menu: HarpPredictionMenuSeal) -> tuple[HarpTargetAction, ...]:
    """Build the full candidate-by-lambda target grid without target truth."""

    menu.assert_valid()
    output: list[HarpTargetAction] = []
    for center in CENTERS:
        actions = tuple(
            action
            for action in menu.actions
            if action.surface_kind == TARGET_SURFACE
            and action.outer_target_id == center
            and action.query_center_id == center
        )
        baseline_action = next(
            (action for action in actions if action.action_id == BASE_ACTION_ID), None
        )
        reference_action = next(
            (action for action in actions if action.action_id == UNIFORM_ACTION_ID), None
        )
        if baseline_action is None or reference_action is None:
            raise ProtocolError("HARP Stage-90 target context lacks exact B or U.")
        source_actions = tuple(action for action in actions if action.selected_source_id is not None)
        fallback_cells = menu.cells_for(baseline_action)
        reference_cells = menu.cells_for(reference_action)
        fallback = menu.exact_nine(baseline_action)
        reference = menu.exact_nine(reference_action)
        row_ids, case_ids = menu.identities_for(reference_action)
        if menu.identities_for(baseline_action) != (row_ids, case_ids):
            raise ProtocolError("HARP Stage-90 target B/U rows are misaligned.")
        for ordinal, (sample_id, case_id) in enumerate(zip(row_ids, case_ids, strict=True)):
            reference_members = np.asarray(
                [cell.probabilities[ordinal] for cell in reference_cells], dtype=np.float64
            )
            # Force the operational B ensemble through the same row/cell
            # alignment check even though only its exact-nine mean is routed.
            if any(
                cell.row_ids != reference_cells[0].row_ids
                or cell.case_ids != reference_cells[0].case_ids
                for cell in fallback_cells
            ):
                raise ProtocolError("HARP Stage-90 target exact-B cells are misaligned.")
            reference_probability = float(reference[ordinal])
            reference_bytes = struct.pack("<d", reference_probability)
            fallback_bytes = struct.pack("<d", float(fallback[ordinal]))
            receipt = _ensemble_receipt(
                menu,
                center=center,
                ordinal=ordinal,
                sample_id=sample_id,
                case_id=case_id,
            )
            for action in source_actions:
                assert action.selected_source_id is not None
                if action.selected_source_id == center:
                    raise ProtocolError("HARP Stage-90 target expert entered its own action menu.")
                expert_cells = menu.cells_for(action)
                if any(
                    cell.row_ids != reference_cells[0].row_ids
                    or cell.case_ids != reference_cells[0].case_ids
                    for cell in expert_cells
                ):
                    raise ProtocolError(
                        "HARP Stage-90 target candidate cells are misaligned."
                    )
                expert_members = np.asarray(
                    [cell.probabilities[ordinal] for cell in expert_cells], dtype=np.float64
                )
                expert_probability = float(menu.exact_nine(action)[ordinal])
                for lam in LAMBDA_GRID:
                    features = _feature_values(
                        reference_members,
                        expert_members,
                        reference_probability,
                        expert_probability,
                        lam,
                    )
                    output.append(
                        HarpTargetAction(
                            outer_target_id=center,
                            target_query_id=center,
                            candidate_source_id=action.selected_source_id,
                            case_id=case_id,
                            sample_id=sample_id,
                            lambda_value=lam,
                            direction=_direction(reference_probability, features[2]),
                            feature_names=ACTION_FEATURE_NAMES,
                            feature_values=features,
                            baseline_probability_bytes=reference_bytes,
                            operational_fallback_probability_bytes=fallback_bytes,
                            expert_probability=expert_probability,
                            ensemble_size=len(EXACT_NINE_SEED_PAIRS),
                            ensemble_receipt_hash=receipt,
                            prediction_seal_hash=menu.seal_hash,
                            compatibility_shrinkage=1.0,
                        )
                    )
    return tuple(output)


__all__ = ("SEED_IDS", "build_development_feature_surface", "build_target_actions")
