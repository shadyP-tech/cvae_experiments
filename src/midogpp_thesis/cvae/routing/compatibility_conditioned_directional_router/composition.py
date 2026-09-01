"""Direction-specific top-K soft composition anchored to protected B."""

from __future__ import annotations

import math
import struct
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import (
    CompositionReceipt,
    CompositionResult,
    Direction,
    RoutingDecision,
    TargetAction,
    canonical_probability_bytes,
    canonical_text,
)
from .hashing import probability_bytes_hash


def _unpack(values: Sequence[bytes]) -> np.ndarray:
    return np.asarray([struct.unpack("<f", value)[0] for value in values], dtype=np.float64)


def compose_directional_probability_bytes(
    baseline_probability_bytes: Sequence[bytes],
    candidate_probability_bytes: Sequence[Sequence[bytes]],
    *,
    weights: Sequence[float],
    mixture_lambda: float,
    direction: Direction,
) -> tuple[bytes, ...]:
    """Low-level deterministic directional composition referenced to B.

    D01 may only alter rows on B's negative branch; D10 may only alter rows on
    B's positive branch.  The opposite branch reuses the original byte objects.
    ``mixture_lambda == 0`` is an exact byte-for-byte B identity.
    """

    baseline_raw = canonical_probability_bytes(baseline_probability_bytes)
    candidates = tuple(canonical_probability_bytes(values) for values in candidate_probability_bytes)
    normalized_weights = tuple(float(value) for value in weights)
    try:
        route_direction = Direction(direction)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("Directional composition received an unknown direction.") from exc
    route_lambda = float(mixture_lambda)
    if (
        not 0.0 <= route_lambda <= 1.0
        or any(not math.isfinite(value) or value <= 0.0 for value in normalized_weights)
        or len(candidates) != len(normalized_weights)
        or (candidates and not math.isclose(sum(normalized_weights), 1.0, rel_tol=1e-12, abs_tol=1e-12))
        or any(len(values) != len(baseline_raw) for values in candidates)
    ):
        raise ProtocolError("Directional composition weights or probability geometry are invalid.")
    if route_lambda == 0.0:
        return baseline_raw
    if not candidates:
        raise ProtocolError("An enabled directional composition requires at least one candidate.")
    baseline = _unpack(baseline_raw)
    matrix = np.asarray([_unpack(values) for values in candidates], dtype=np.float64)
    candidate_mean = np.sum(
        np.asarray(normalized_weights, dtype=np.float64)[:, None] * matrix,
        axis=0,
        dtype=np.float64,
    )
    mixed = (1.0 - route_lambda) * baseline + route_lambda * candidate_mean
    mixed = np.clip(mixed, 0.0, 1.0)
    if route_direction is Direction.D01:
        active = baseline < 0.5
    elif route_direction is Direction.D10:
        active = baseline >= 0.5
    else:
        active = np.ones(len(baseline), dtype=bool)
    output: list[bytes] = []
    for ordinal, is_active in enumerate(active.tolist()):
        if not is_active:
            output.append(baseline_raw[ordinal])
        else:
            output.append(struct.pack("<f", float(np.float32(mixed[ordinal]))))
    return tuple(output)


def compose_route(
    *,
    decision: RoutingDecision,
    baseline_sample_ids: Sequence[str],
    baseline_probability_bytes: Sequence[bytes],
    actions: Sequence[TargetAction],
) -> CompositionResult:
    """Materialize only the actions named by a frozen pre-label decision."""

    if not isinstance(decision, RoutingDecision):
        raise ProtocolError("Route composition requires a typed frozen decision.")
    samples = tuple(canonical_text(value, name="baseline sample") for value in baseline_sample_ids)
    baseline = canonical_probability_bytes(baseline_probability_bytes)
    if len(samples) != len(baseline) or len(set(samples)) != len(samples):
        raise ProtocolError("Baseline sample identities and probabilities are misaligned.")
    action_by_id = {row.feature.action_id: row for row in actions}
    if len(action_by_id) != len(tuple(actions)):
        raise ProtocolError("Target action inventory contains duplicate action ids.")
    if not decision.enabled:
        output = baseline
        receipt = CompositionReceipt(
            decision_hash=decision.decision_hash,
            baseline_probability_hash=probability_bytes_hash(baseline),
            selected_probability_hashes=(),
            output_probability_hash=probability_bytes_hash(output),
            exact_baseline_fallback=True,
            opposite_branch_preserved=True,
        )
        return CompositionResult(
            sample_ids=samples,
            output_probability_bytes=output,
            receipt=receipt,
        )
    try:
        selected = tuple(action_by_id[action_id] for action_id in decision.selected_action_ids)
    except KeyError as exc:
        raise ProtocolError("Frozen route names an absent target action.") from exc
    if (
        any(row.sample_ids != samples for row in selected)
        or any(row.feature.outer_target_id != decision.outer_target_id for row in selected)
        or any(row.feature.case_id != decision.case_id for row in selected)
        or any(row.feature.direction is not decision.selected_direction for row in selected)
    ):
        raise ProtocolError("Selected target actions crossed route case or direction.")
    assert decision.selected_direction is not None
    output = compose_directional_probability_bytes(
        baseline,
        tuple(row.probability_bytes for row in selected),
        weights=decision.selected_weights,
        mixture_lambda=decision.mixture_lambda,
        direction=decision.selected_direction,
    )
    baseline_values = _unpack(baseline)
    if decision.selected_direction is Direction.D01:
        opposite = baseline_values >= 0.5
    elif decision.selected_direction is Direction.D10:
        opposite = baseline_values < 0.5
    else:
        opposite = np.zeros(len(baseline), dtype=bool)
    opposite_preserved = all(
        output[index] == baseline[index] for index in np.flatnonzero(opposite).tolist()
    )
    receipt = CompositionReceipt(
        decision_hash=decision.decision_hash,
        baseline_probability_hash=probability_bytes_hash(baseline),
        selected_probability_hashes=tuple(
            probability_bytes_hash(row.probability_bytes) for row in selected
        ),
        output_probability_hash=probability_bytes_hash(output),
        exact_baseline_fallback=False,
        opposite_branch_preserved=opposite_preserved,
    )
    return CompositionResult(
        sample_ids=samples,
        output_probability_bytes=output,
        receipt=receipt,
    )


__all__ = ("compose_directional_probability_bytes", "compose_route")
