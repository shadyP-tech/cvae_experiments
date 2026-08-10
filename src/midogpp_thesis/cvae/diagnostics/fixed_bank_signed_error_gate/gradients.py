"""Strictly OOF class-balanced proper-loss gradient targets."""

from __future__ import annotations

from typing import Sequence

from ...protocol import ProtocolError
from ..fixed_bank_hierarchical_residual_stacker.contracts import (
    BinaryLabel,
    SampleActionProbability,
)
from ..fixed_bank_hierarchical_residual_stacker.scientific_constants import (
    BASELINE_ACTION_ID,
)
from .contracts import GradientTargetRow


def build_gradient_targets(
    probabilities: Sequence[SampleActionProbability],
    labels: Sequence[BinaryLabel],
    *,
    heldout_target: str,
) -> tuple[GradientTargetRow, ...]:
    """Return the negative log-loss logit gradient, balanced 0.5/0.5 by class."""

    label_rows = tuple(labels)
    if (
        not label_rows
        or any(row.label_scope != "loco_donor" for row in label_rows)
        or any(row.target_center == str(heldout_target) for row in label_rows)
    ):
        raise ProtocolError(
            "Signed-error gradient fitting accepts only outer-target-excluded LOCO labels."
        )
    label_by_key = {row.sample_key: row.label for row in label_rows}
    baseline_rows = tuple(
        row
        for row in probabilities
        if row.action_id == BASELINE_ACTION_ID and row.sample_key in label_by_key
    )
    baseline = {row.sample_key: row.probability for row in baseline_rows}
    if (
        not baseline
        or len(baseline_rows) != len(baseline)
        or len(label_by_key) != len(label_rows)
        or set(baseline) != set(label_by_key)
    ):
        raise ProtocolError("Gradient targets require unique, aligned baseline rows and labels.")
    center_counts = {
        center: {
            label: sum(
                key[0] == center and value == label
                for key, value in label_by_key.items()
            )
            for label in (0, 1)
        }
        for center in {key[0] for key in label_by_key}
    }
    if any(not counts[0] or not counts[1] for counts in center_counts.values()):
        raise ProtocolError("Each donor center needs both classes for balanced gradients.")
    output: list[GradientTargetRow] = []
    for target, case, sample in sorted(label_by_key):
        key = (target, case, sample)
        label = label_by_key[key]
        probability = baseline[key]
        counts = center_counts[target]
        center_total = counts[0] + counts[1]
        weight = 0.5 * center_total / counts[label]
        output.append(
            GradientTargetRow(
                target,
                case,
                sample,
                label,
                probability,
                weight,
                weight * (label - probability),
            )
        )
    return tuple(output)


__all__ = ("build_gradient_targets",)
