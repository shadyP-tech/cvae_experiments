"""Deterministic complete-case feature-block permutation control."""

from __future__ import annotations

import random
from typing import Sequence

from ...protocol import ProtocolError
from .contracts import DirectionalDonorRow


def permute_complete_case_feature_blocks(
    rows: Sequence[DirectionalDonorRow],
    *,
    seed: int,
) -> tuple[DirectionalDonorRow, ...]:
    """Derange complete action/direction feature blocks within each donor q."""

    supplied = tuple(rows)
    if not supplied:
        raise ProtocolError("Permutation control requires donor rows.")
    groups: dict[tuple[str, str], list[DirectionalDonorRow]] = {}
    for row in supplied:
        groups.setdefault((row.model_target, row.query_center), []).append(row)
    result: list[DirectionalDonorRow] = []
    generator = random.Random(int(seed))
    for key in sorted(groups):
        group = tuple(groups[key])
        cases = sorted({row.case_id for row in group})
        if len(cases) < 2:
            raise ProtocolError("Permutation group requires two complete cases.")
        expected_cells = {
            (row.action_id, row.direction) for row in group if row.case_id == cases[0]
        }
        if any(
            {(row.action_id, row.direction) for row in group if row.case_id == case}
            != expected_cells
            for case in cases
        ):
            raise ProtocolError("Permutation feature blocks are incomplete.")
        shuffled = list(cases)
        generator.shuffle(shuffled)
        mapped = shuffled[1:] + shuffled[:1]
        mapping = dict(zip(shuffled, mapped, strict=True))
        feature_index = {
            (row.case_id, row.action_id, row.direction): row for row in group
        }
        for row in group:
            feature = feature_index[(mapping[row.case_id], row.action_id, row.direction)]
            result.append(
                DirectionalDonorRow(
                    model_target=row.model_target,
                    query_center=row.query_center,
                    candidate_source=row.candidate_source,
                    case_id=row.case_id,
                    action_id=row.action_id,
                    feature_case_id=feature.case_id,
                    direction=row.direction,
                    success_count=row.success_count,
                    trial_count=row.trial_count,
                    feature_names=row.feature_names,
                    values=feature.values,
                )
            )
    ordered = tuple(
        sorted(
            result,
            key=lambda row: (
                row.model_target,
                row.query_center,
                row.case_id,
                row.action_id,
                row.direction,
            ),
        )
    )
    if any(row.case_id == row.feature_case_id for row in ordered):
        raise ProtocolError("Permutation control failed to derange a case block.")
    return ordered


__all__ = ("permute_complete_case_feature_blocks",)
