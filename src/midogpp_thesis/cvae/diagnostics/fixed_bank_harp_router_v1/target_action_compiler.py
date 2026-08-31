"""Compiled, label-free construction of the HARP target action surface."""

from __future__ import annotations

import statistics
import struct

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.harp_action_model import HarpTargetAction, LAMBDA_GRID
from ...routing.harp_action_surface import ACTION_FEATURE_NAMES
from ...routing.harp_protocol import canonical_hash
from ...runtime.harp_probability_menu import (
    BASE_ACTION_ID,
    EXACT_NINE_SEED_PAIRS,
    UNIFORM_ACTION_ID,
    HarpPredictionMenuSeal,
    HarpActionSpec,
    HarpPredictionCell,
)
from ...runtime.harp_probability_menu.indexed import (
    HarpValidatedTargetMenuView,
    validated_target_menu_view,
)
from ...runtime.harp_probability_menu.hashing import raw_array_sha256


def _feature_values(
    baseline_members: np.ndarray,
    expert_members: np.ndarray,
    baseline: float,
    expert: float,
    lam: float,
) -> tuple[float, ...]:
    """Retain the frozen scalar operation order of the legacy constructor."""

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
    view: HarpValidatedTargetMenuView,
    *,
    center: str,
    ordinal: int,
    sample_id: str,
    case_id: str,
) -> str:
    members = []
    for action in view.actions_for_center(center):
        cells = view.cells_for(action)
        values = np.asarray(
            [cell.probabilities[ordinal] for cell in cells], dtype=np.float32
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
            "prediction_menu_seal_hash": view.seal_hash,
            "outer_target_id": center,
            "sample_id": sample_id,
            "case_id": case_id,
            "seed_pairs": [list(pair) for pair in EXACT_NINE_SEED_PAIRS],
            "actions": members,
            "labels_consumed": False,
        }
    )


def _build_from_view(
    view: HarpValidatedTargetMenuView,
) -> tuple[HarpTargetAction, ...]:
    output: list[HarpTargetAction] = []
    for center in CENTERS:
        actions = view.actions_for_center(center)
        baseline_action = next(
            (action for action in actions if action.action_id == BASE_ACTION_ID), None
        )
        reference_action = next(
            (action for action in actions if action.action_id == UNIFORM_ACTION_ID), None
        )
        if baseline_action is None or reference_action is None:
            raise ProtocolError("HARP Stage-90 target context lacks exact B or U.")

        source_actions = tuple(
            action for action in actions if action.selected_source_id is not None
        )
        fallback_cells = view.cells_for(baseline_action)
        reference_cells = view.cells_for(reference_action)
        fallback = view.exact_nine(baseline_action)
        reference = view.exact_nine(reference_action)
        row_ids, case_ids = view.identities_for(reference_action)
        if view.identities_for(baseline_action) != (row_ids, case_ids):
            raise ProtocolError("HARP Stage-90 target B/U rows are misaligned.")

        reference_identity = (reference_cells[0].row_ids, reference_cells[0].case_ids)
        if any(
            (cell.row_ids, cell.case_ids) != reference_identity
            for cell in fallback_cells
        ):
            raise ProtocolError("HARP Stage-90 target exact-B cells are misaligned.")

        compiled_sources: list[
            tuple[HarpActionSpec, tuple[HarpPredictionCell, ...], np.ndarray]
        ] = []
        for action in source_actions:
            if action.selected_source_id == center:
                raise ProtocolError("HARP Stage-90 target expert entered its own action menu.")
            expert_cells = view.cells_for(action)
            if any(
                (cell.row_ids, cell.case_ids) != reference_identity
                for cell in expert_cells
            ):
                raise ProtocolError("HARP Stage-90 target candidate cells are misaligned.")
            compiled_sources.append((action, expert_cells, view.exact_nine(action)))

        for ordinal, (sample_id, case_id) in enumerate(
            zip(row_ids, case_ids, strict=True)
        ):
            reference_members = np.asarray(
                [cell.probabilities[ordinal] for cell in reference_cells],
                dtype=np.float64,
            )
            reference_probability = float(reference[ordinal])
            reference_bytes = struct.pack("<d", reference_probability)
            fallback_bytes = struct.pack("<d", float(fallback[ordinal]))
            receipt = _ensemble_receipt(
                view,
                center=center,
                ordinal=ordinal,
                sample_id=sample_id,
                case_id=case_id,
            )
            for action, expert_cells, exact_expert in compiled_sources:
                selected_source = action.selected_source_id
                assert selected_source is not None
                expert_members = np.asarray(
                    [cell.probabilities[ordinal] for cell in expert_cells],
                    dtype=np.float64,
                )
                expert_probability = float(exact_expert[ordinal])
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
                            candidate_source_id=selected_source,
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
                            prediction_seal_hash=view.seal_hash,
                            compatibility_shrinkage=1.0,
                        )
                    )
    return tuple(output)


def build_target_actions(
    menu: HarpPredictionMenuSeal,
) -> tuple[HarpTargetAction, ...]:
    """Compile the complete target grid with two full fail-closed validations."""

    view = validated_target_menu_view(menu)
    output = _build_from_view(view)
    # No target surface escapes if raw menu bytes changed during construction.
    view.assert_fully_valid()
    return output


__all__ = ("build_target_actions",)
