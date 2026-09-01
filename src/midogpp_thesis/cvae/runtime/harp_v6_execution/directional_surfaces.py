"""Case-level directional surfaces for the HARP v6 production adapter.

This module is the narrow bridge between the physical B/U/Hxe probability
menu and the stage-neutral compatibility-conditioned router.  Surface
construction is label-free and byte-preserving.  Source labels enter only in
``build_source_directional_observations`` after the development menu has been
sealed; the target builder has no label argument.

The three directional surfaces form a deterministic partition of each
physical challenger relative to protected B:

* ``D01`` contains only threshold crossings from B < .5 to action >= .5;
* ``D10`` contains only threshold crossings from B >= .5 to action < .5;
* ``ALL_MARGINS`` contains the remaining, non-crossing margin changes.

Every cell outside the relevant mask reuses the exact float32 B bytes.  This
means a structural no-op remains byte-identical to B and can be removed later
by the neutral opportunity primitive without numerical tolerance choices.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from types import MappingProxyType

import numpy as np

from ...protocol import ProtocolError
from ...routing.case_equal_metrics import case_class_support_counts, case_metrics
from ...routing.compatibility_conditioned_directional_router import (
    ActionKind as RouterActionKind,
    CandidatePoolReceipt,
    CompatibilityReceipt,
    Direction,
    EndpointEffects,
    SourceActionObservation,
    TargetAction,
    build_candidate_feature,
    compatibility_by_candidate,
    probability_hash,
)
from .contracts import (
    ActionKind as PhysicalActionKind,
    LabelFreeActionBlock,
    LabelFreeOuterMenu,
)


DIRECTIONS = (Direction.D01, Direction.D10, Direction.ALL)


def _probability_cells(values: np.ndarray) -> tuple[bytes, ...]:
    """Return one exact little-endian float32 byte cell per probability."""

    raw = np.asarray(values)
    if (
        raw.dtype != np.dtype("float32")
        or raw.ndim != 1
        or not len(raw)
        or not np.isfinite(raw).all()
        or np.any((raw < 0.0) | (raw > 1.0))
    ):
        raise ProtocolError("HARP v6 directional probabilities are malformed.")
    packed = np.ascontiguousarray(raw, dtype="<f4").tobytes(order="C")
    return tuple(packed[index : index + 4] for index in range(0, len(packed), 4))


def directional_probability_bytes(
    baseline: LabelFreeActionBlock,
    challenger: LabelFreeActionBlock,
    *,
    case_id: str,
    direction: Direction,
) -> tuple[tuple[str, ...], tuple[bytes, ...]]:
    """Build one case surface while preserving exact physical probability bytes.

    The returned sample identities retain the physical block order.  No
    probability is averaged, cast, or recomputed here: each output cell is
    selected directly from either the protected B vector or the physical
    challenger vector.
    """

    _validate_aligned_blocks(baseline, challenger)
    try:
        selected_direction = Direction(direction)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("HARP v6 directional surface has an unknown direction.") from exc
    indices = _case_indices(baseline, case_id)
    base = np.asarray(baseline.probabilities[indices], dtype=np.float32)
    action = np.asarray(challenger.probabilities[indices], dtype=np.float32)
    base_bytes = _probability_cells(base)
    action_bytes = _probability_cells(action)
    base_positive = base >= np.float32(0.5)
    action_positive = action >= np.float32(0.5)
    if selected_direction is Direction.D01:
        active = (~base_positive) & action_positive
    elif selected_direction is Direction.D10:
        active = base_positive & (~action_positive)
    else:
        active = base_positive == action_positive
    output = tuple(
        action_bytes[index] if bool(active[index]) else base_bytes[index]
        for index in range(len(indices))
    )
    sample_ids = tuple(baseline.sample_ids[index] for index in indices.tolist())
    return sample_ids, output


def build_target_directional_actions(
    menu: LabelFreeOuterMenu,
    *,
    candidate_pool: CandidatePoolReceipt,
    compatibility_receipts: Sequence[CompatibilityReceipt],
) -> tuple[TargetAction, ...]:
    """Compile the complete label-free target directional action inventory.

    There is intentionally no label-bearing parameter.  Compatibility is a
    typed, all-seed, support-only proxy bound to the exact target C-minus-H
    candidate pool.
    """

    blocks, compatibility = _validated_context(
        menu,
        candidate_pool=candidate_pool,
        compatibility_receipts=compatibility_receipts,
        expected_role="target",
    )
    baseline = _single_block(blocks, PhysicalActionKind.B)
    actions: list[TargetAction] = []
    for case_id in _case_order(baseline):
        for challenger in _challengers(blocks):
            router_kind, candidate = _router_action_identity(challenger)
            receipt = compatibility.get(candidate) if candidate is not None else None
            for direction in DIRECTIONS:
                sample_ids, surface = directional_probability_bytes(
                    baseline, challenger, case_id=case_id, direction=direction
                )
                feature = build_candidate_feature(
                    candidate_pool=candidate_pool,
                    case_id=case_id,
                    action_id=_action_id(router_kind, candidate, direction),
                    action_kind=router_kind,
                    direction=direction,
                    candidate_source_id=candidate,
                    base_features=_base_features(
                        baseline,
                        challenger,
                        case_id=case_id,
                        direction=direction,
                        surface=surface,
                    ),
                    probability_hash=probability_hash(surface),
                    compatibility=receipt,
                )
                actions.append(
                    TargetAction(
                        feature=feature,
                        candidate_pool=candidate_pool,
                        sample_ids=sample_ids,
                        probability_bytes=surface,
                        prediction_seal_hash=menu.menu_hash,
                    )
                )
    return tuple(sorted(actions, key=lambda row: (row.feature.case_id, row.feature.action_id)))


def build_source_directional_observations(
    menu: LabelFreeOuterMenu,
    *,
    candidate_pool: CandidatePoolReceipt,
    compatibility_receipts: Sequence[CompatibilityReceipt],
    source_labels: Mapping[tuple[str, str], int],
) -> tuple[SourceActionObservation, ...]:
    """Attach source-development endpoint effects to sealed directional rows.

    ``source_labels`` is keyed by ``(case_id, sample_id)`` and must equal the
    complete physical development context exactly.  Extra rows, missing rows,
    or labels for another query are rejected before a response is computed.
    The BACC effect is the normalized additive case contribution whose mean
    reconstructs the query-center case-equal BACC estimand.
    """

    blocks, compatibility = _validated_context(
        menu,
        candidate_pool=candidate_pool,
        compatibility_receipts=compatibility_receipts,
        expected_role="development",
    )
    if candidate_pool.target_scope:
        raise ProtocolError("HARP v6 source responses cannot use target-scope labels.")
    baseline = _single_block(blocks, PhysicalActionKind.B)
    labels = _validated_source_labels(baseline, source_labels)
    cases = _case_order(baseline)
    label_rows = tuple(
        np.asarray(
            [
                labels[(case_id, baseline.sample_ids[index])]
                for index in _case_indices(baseline, case_id)
            ],
            dtype=np.int64,
        )
        for case_id in cases
    )
    support_counts = case_class_support_counts(label_rows)
    observations: list[SourceActionObservation] = []
    for case_id, truth in zip(cases, label_rows, strict=True):
        indices = _case_indices(baseline, case_id)
        base_probability = np.asarray(baseline.probabilities[indices], dtype=np.float64)
        base_metrics = case_metrics(
            base_probability,
            truth,
            total_case_count=len(cases),
            class_support_case_counts=support_counts,
        )
        for challenger in _challengers(blocks):
            router_kind, candidate = _router_action_identity(challenger)
            receipt = compatibility.get(candidate) if candidate is not None else None
            for direction in DIRECTIONS:
                _, surface = directional_probability_bytes(
                    baseline, challenger, case_id=case_id, direction=direction
                )
                feature = build_candidate_feature(
                    candidate_pool=candidate_pool,
                    case_id=case_id,
                    action_id=_action_id(router_kind, candidate, direction),
                    action_kind=router_kind,
                    direction=direction,
                    candidate_source_id=candidate,
                    base_features=_base_features(
                        baseline,
                        challenger,
                        case_id=case_id,
                        direction=direction,
                        surface=surface,
                    ),
                    probability_hash=probability_hash(surface),
                    compatibility=receipt,
                )
                action_probability = np.asarray(
                    [np.frombuffer(value, dtype="<f4")[0] for value in surface],
                    dtype=np.float64,
                )
                action_metrics = case_metrics(
                    action_probability,
                    truth,
                    total_case_count=len(cases),
                    class_support_case_counts=support_counts,
                )
                observations.append(
                    SourceActionObservation(
                        feature=feature,
                        candidate_pool=candidate_pool,
                        effects=EndpointEffects(
                            bacc_gain=(
                                action_metrics.case_equal_bacc_contribution
                                - base_metrics.case_equal_bacc_contribution
                            ),
                            brier_delta=action_metrics.brier - base_metrics.brier,
                            log_delta=action_metrics.log_loss - base_metrics.log_loss,
                        ),
                    )
                )
    return tuple(
        sorted(
            observations,
            key=lambda row: (row.feature.case_id, row.feature.action_id),
        )
    )


def _validated_context(
    menu: LabelFreeOuterMenu,
    *,
    candidate_pool: CandidatePoolReceipt,
    compatibility_receipts: Sequence[CompatibilityReceipt],
    expected_role: str,
) -> tuple[tuple[LabelFreeActionBlock, ...], Mapping[str, CompatibilityReceipt]]:
    if not isinstance(menu, LabelFreeOuterMenu) or not isinstance(
        candidate_pool, CandidatePoolReceipt
    ):
        raise ProtocolError("HARP v6 directional compilation requires typed menu and pool.")
    if (
        menu.outer_target_id != candidate_pool.outer_target_id
        or (expected_role == "target") != candidate_pool.target_scope
    ):
        raise ProtocolError("HARP v6 directional menu and candidate-pool roles disagree.")
    blocks = tuple(
        block
        for block in menu.blocks
        if block.surface_role == expected_role
        and block.query_center_id == candidate_pool.query_center_id
    )
    if not blocks:
        raise ProtocolError("HARP v6 directional menu lacks its requested H/q context.")
    baseline = _single_block(blocks, PhysicalActionKind.B)
    _single_block(blocks, PhysicalActionKind.U)
    expert_sources = tuple(
        sorted(
            block.selected_source_id
            for block in blocks
            if block.action_kind is PhysicalActionKind.HXE
            and block.selected_source_id is not None
        )
    )
    if expert_sources != candidate_pool.candidate_center_ids:
        raise ProtocolError("HARP v6 physical Hxe menu is not the exact candidate pool.")
    if any(
        block.sample_ids != baseline.sample_ids or block.case_ids != baseline.case_ids
        for block in blocks
    ):
        raise ProtocolError("HARP v6 physical context sample/case alignment drifted.")
    typed_receipts = tuple(compatibility_receipts)
    if any(not isinstance(row, CompatibilityReceipt) for row in typed_receipts):
        raise ProtocolError("HARP v6 compatibility inventory contains an untyped row.")
    by_candidate = compatibility_by_candidate(typed_receipts)
    if (
        tuple(sorted(by_candidate)) != candidate_pool.candidate_center_ids
        or any(
            row.outer_target_id != candidate_pool.outer_target_id
            or row.query_center_id != candidate_pool.query_center_id
            or row.candidate_pool_hash != candidate_pool.pool_hash
            for row in typed_receipts
        )
    ):
        raise ProtocolError("HARP v6 compatibility escaped the exact H/q candidate pool.")
    return blocks, MappingProxyType(by_candidate)


def _single_block(
    blocks: Sequence[LabelFreeActionBlock], kind: PhysicalActionKind
) -> LabelFreeActionBlock:
    matched = tuple(block for block in blocks if block.action_kind is kind)
    if len(matched) != 1:
        raise ProtocolError(f"HARP v6 context lacks exactly one {kind.value} block.")
    return matched[0]


def _challengers(
    blocks: Sequence[LabelFreeActionBlock],
) -> tuple[LabelFreeActionBlock, ...]:
    return tuple(
        sorted(
            (block for block in blocks if block.action_kind is not PhysicalActionKind.B),
            key=lambda row: (row.action_kind.value, row.selected_source_id or ""),
        )
    )


def _router_action_identity(
    block: LabelFreeActionBlock,
) -> tuple[RouterActionKind, str | None]:
    if block.action_kind is PhysicalActionKind.U:
        return RouterActionKind.U, None
    if block.action_kind is PhysicalActionKind.HXE and block.selected_source_id is not None:
        return RouterActionKind.HXE, block.selected_source_id
    raise ProtocolError("HARP v6 protected B cannot become a challenger action.")


def _action_id(
    kind: RouterActionKind, candidate: str | None, direction: Direction
) -> str:
    if kind is RouterActionKind.U:
        return f"U:{direction.value}"
    if candidate is None:
        raise ProtocolError("HARP v6 HXE directional action lacks a candidate.")
    return f"HXE:{candidate}:{direction.value}"


def _case_order(block: LabelFreeActionBlock) -> tuple[str, ...]:
    return tuple(dict.fromkeys(block.case_ids))


def _case_indices(block: LabelFreeActionBlock, case_id: str) -> np.ndarray:
    if type(case_id) is not str or not case_id or case_id.strip() != case_id:
        raise ProtocolError("HARP v6 directional case identity is malformed.")
    indices = np.flatnonzero(np.asarray(block.case_ids, dtype=object) == case_id)
    if not len(indices):
        raise ProtocolError("HARP v6 directional case is absent from its physical block.")
    return indices


def _validate_aligned_blocks(
    baseline: LabelFreeActionBlock, challenger: LabelFreeActionBlock
) -> None:
    if (
        not isinstance(baseline, LabelFreeActionBlock)
        or not isinstance(challenger, LabelFreeActionBlock)
        or baseline.action_kind is not PhysicalActionKind.B
        or challenger.action_kind is PhysicalActionKind.B
        or baseline.surface_role != challenger.surface_role
        or baseline.outer_target_id != challenger.outer_target_id
        or baseline.query_center_id != challenger.query_center_id
        or baseline.sample_ids != challenger.sample_ids
        or baseline.case_ids != challenger.case_ids
    ):
        raise ProtocolError("HARP v6 baseline/challenger physical blocks are misaligned.")


def _surface_values(surface: Sequence[bytes]) -> np.ndarray:
    cells = tuple(surface)
    if not cells or any(type(value) is not bytes or len(value) != 4 for value in cells):
        raise ProtocolError("HARP v6 directional surface bytes are malformed.")
    return np.asarray(
        [np.frombuffer(value, dtype="<f4")[0] for value in cells], dtype=np.float64
    )


def _base_features(
    baseline: LabelFreeActionBlock,
    challenger: LabelFreeActionBlock,
    *,
    case_id: str,
    direction: Direction,
    surface: Sequence[bytes],
) -> Mapping[str, float]:
    indices = _case_indices(baseline, case_id)
    base = np.asarray(baseline.probabilities[indices], dtype=np.float64)
    physical = np.asarray(challenger.probabilities[indices], dtype=np.float64)
    routed = _surface_values(surface)
    if routed.shape != base.shape:
        raise ProtocolError("HARP v6 directional surface feature geometry drifted.")
    delta = routed - base
    changed = delta != 0.0
    base_class = base >= 0.5
    routed_class = routed >= 0.5
    flips = base_class != routed_class
    base_dispersion = np.asarray(baseline.seed_dispersion[indices], dtype=np.float64)
    physical_dispersion = np.asarray(challenger.seed_dispersion[indices], dtype=np.float64)
    surface_dispersion = np.where(changed, physical_dispersion, base_dispersion)
    if direction is Direction.D01:
        aligned_mass = np.maximum(delta, 0.0)
    elif direction is Direction.D10:
        aligned_mass = np.maximum(-delta, 0.0)
    else:
        aligned_mass = np.abs(delta)
    values = {
        "case_sample_count_log1p": math.log1p(float(len(base))),
        "baseline_probability_mean": float(np.mean(base, dtype=np.float64)),
        "surface_probability_mean": float(np.mean(routed, dtype=np.float64)),
        "baseline_negative_branch_fraction": float(np.mean(~base_class, dtype=np.float64)),
        "baseline_positive_branch_fraction": float(np.mean(base_class, dtype=np.float64)),
        "active_mask_fraction": float(np.mean(changed, dtype=np.float64)),
        "threshold_flip_fraction": float(np.mean(flips, dtype=np.float64)),
        "signed_branch_mass": float(np.mean(delta, dtype=np.float64)),
        "direction_aligned_branch_mass": float(np.mean(aligned_mass, dtype=np.float64)),
        "action_delta_mean": float(np.mean(delta, dtype=np.float64)),
        "action_delta_std": float(np.std(delta, dtype=np.float64)),
        "action_delta_abs_mean": float(np.mean(np.abs(delta), dtype=np.float64)),
        "action_delta_min": float(np.min(delta)),
        "action_delta_max": float(np.max(delta)),
        "baseline_boundary_distance_mean": float(
            np.mean(np.abs(base - 0.5), dtype=np.float64)
        ),
        "baseline_boundary_distance_min": float(np.min(np.abs(base - 0.5))),
        "surface_boundary_distance_mean": float(
            np.mean(np.abs(routed - 0.5), dtype=np.float64)
        ),
        "surface_boundary_distance_min": float(np.min(np.abs(routed - 0.5))),
        "boundary_distance_change_mean": float(
            np.mean(np.abs(routed - 0.5) - np.abs(base - 0.5), dtype=np.float64)
        ),
        "baseline_seed_dispersion_mean": float(
            np.mean(base_dispersion, dtype=np.float64)
        ),
        "baseline_seed_dispersion_max": float(np.max(base_dispersion)),
        "physical_action_seed_dispersion_mean": float(
            np.mean(physical_dispersion, dtype=np.float64)
        ),
        "physical_action_seed_dispersion_max": float(np.max(physical_dispersion)),
        "surface_seed_dispersion_mean": float(
            np.mean(surface_dispersion, dtype=np.float64)
        ),
        "surface_seed_dispersion_max": float(np.max(surface_dispersion)),
        "surface_seed_dispersion_change_mean": float(
            np.mean(surface_dispersion - base_dispersion, dtype=np.float64)
        ),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ProtocolError("HARP v6 directional base features are nonfinite.")
    return MappingProxyType(values)


def _validated_source_labels(
    baseline: LabelFreeActionBlock,
    source_labels: Mapping[tuple[str, str], int],
) -> Mapping[tuple[str, str], int]:
    if not isinstance(source_labels, Mapping):
        raise ProtocolError("HARP v6 source labels must be an exact keyed mapping.")
    expected = tuple(zip(baseline.case_ids, baseline.sample_ids, strict=True))
    normalized: dict[tuple[str, str], int] = {}
    for raw_key, raw_value in source_labels.items():
        if (
            type(raw_key) is not tuple
            or len(raw_key) != 2
            or any(type(value) is not str or not value for value in raw_key)
            or isinstance(raw_value, (bool, np.bool_))
            or not isinstance(raw_value, (int, np.integer))
        ):
            raise ProtocolError("HARP v6 source label keys or values are malformed.")
        value = int(raw_value)
        if raw_value != value or value not in (0, 1):
            raise ProtocolError("HARP v6 source labels must be binary integers.")
        normalized[(raw_key[0], raw_key[1])] = value
    if set(normalized) != set(expected) or len(normalized) != len(source_labels):
        raise ProtocolError(
            "HARP v6 source-label keys differ from the sealed development context."
        )
    return MappingProxyType(normalized)


__all__ = (
    "DIRECTIONS",
    "build_source_directional_observations",
    "build_target_directional_actions",
    "directional_probability_bytes",
)
