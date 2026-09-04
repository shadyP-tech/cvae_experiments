"""Source-label joins for the HARP v14 source-crossfit lifecycle.

The fold-scoped join accepts only ``C-{H,q}`` rows inside its isolated worker.
The aggregate join binds each already-sealed q prediction surface to its exact
q outcomes after the complete fold seal set exists.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.harp_protocol import HarpSourceLabelRow
from ...routing.policy_calibrated_residual_router_v14 import (
    EffectiveMenu,
    SourceActionOutcome,
)
from ...runtime.harp_v14_execution.contracts import (
    ActionKind,
    LabelFreeActionBlock,
)
from ...runtime.harp_v14_execution.directional_surfaces import attach_source_outcomes

if TYPE_CHECKING:
    from .source_crossfit_orchestration import (
        FoldFitTask,
        LabelFreeSourceCrossfitBundle,
    )


def join_scoped_worker_outcomes(
    task: FoldFitTask,
    labels: Sequence[HarpSourceLabelRow],
) -> tuple[SourceActionOutcome, ...]:
    """Join C-{H,q} labels inside the one-task child and nowhere else."""

    allowed = tuple(
        center
        for center in CENTERS
        if center not in {task.outer_target_id, task.heldout_center_id}
    )
    rows = tuple(labels)
    if (
        not rows
        or any(not isinstance(row, HarpSourceLabelRow) for row in rows)
        or {row.center for row in rows} != set(allowed)
        or any(row.center in {task.outer_target_id, task.heldout_center_id} for row in rows)
    ):
        raise ProtocolError("HARP v14 isolated worker labels escaped C-{H,q}.")
    label_index = {row.row_key: row.label for row in rows}
    if len(label_index) != len(rows):
        raise ProtocolError("HARP v14 isolated worker labels duplicate an identity.")
    expected_keys: set[tuple[str, str, str]] = set()
    outcomes: list[SourceActionOutcome] = []
    for query, baseline in task.baseline_blocks:
        keys = {
            (query, case, sample)
            for case, sample in zip(baseline.case_ids, baseline.sample_ids, strict=True)
        }
        expected_keys.update(keys)
        try:
            scoped = {
                (case, sample): label_index[(query, case, sample)]
                for _, case, sample in keys
            }
        except KeyError as exc:
            raise ProtocolError(
                "HARP v14 isolated worker labels omit a physical row."
            ) from exc
        menus = tuple(row for row in task.fitting_menus if row.query_center_id == query)
        outcomes.extend(attach_source_outcomes(menus, baseline, source_labels=scoped))
    if set(label_index) != expected_keys:
        raise ProtocolError("HARP v14 isolated worker label scope exceeds its surface.")
    return tuple(outcomes)


def attach_prediction_outcomes(
    bundle: LabelFreeSourceCrossfitBundle,
    labels: Sequence[HarpSourceLabelRow],
) -> tuple[tuple[SourceActionOutcome, ...], tuple[EffectiveMenu, ...]]:
    """Bind every sealed q prediction menu to its exact aggregate labels."""

    by_center = {
        center: tuple(row for row in labels if row.center == center)
        for center in CENTERS
    }
    output: list[SourceActionOutcome] = []
    menus: list[EffectiveMenu] = []
    for h in CENTERS:
        for q in CENTERS:
            if q == h:
                continue
            matches = tuple(
                row
                for row in bundle.physical_surface.blocks_for(h, q, q)
                if row.action.action_id == "B"
            )
            if len(matches) != 1:
                raise ProtocolError("HARP v14 aggregate q join lacks exact physical B.")
            raw = matches[0]
            baseline = LabelFreeActionBlock(
                surface_role="development",
                outer_target_id=h,
                query_center_id=q,
                action_kind=ActionKind.B,
                selected_source_id=None,
                sample_ids=raw.sample_ids,
                case_ids=raw.case_ids,
                probabilities=raw.probabilities,
                seed_dispersion=raw.seed_dispersion,
            )
            q_rows = by_center[q]
            label_index = {row.row_key: row.label for row in q_rows}
            expected = {
                (q, case, sample)
                for case, sample in zip(
                    baseline.case_ids, baseline.sample_ids, strict=True
                )
            }
            if set(label_index) != expected:
                raise ProtocolError(
                    "HARP v14 aggregate q labels exceed or omit the prediction fold."
                )
            scoped = {
                (case, sample): label_index[(q, case, sample)]
                for _, case, sample in expected
            }
            q_menus = tuple(
                row.menu for row in bundle.effective_surface.prediction_menus(h, q)
            )
            output.extend(
                attach_source_outcomes(q_menus, baseline, source_labels=scoped)
            )
            menus.extend(q_menus)
    return tuple(output), tuple(menus)


__all__ = ("attach_prediction_outcomes", "join_scoped_worker_outcomes")
