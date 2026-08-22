"""Adapter from the neutral fixed-bank store to successor-owned arrays."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType

import numpy as np

from ...protocol import ProtocolError
from .constants import CENTERS, SEED_PAIR_COUNT, physical_action_ids
from .contracts import CenterProbabilitySurface, PhysicalProbabilitySurface
from .row_order import canonical_center_row_order


def build_physical_probability_surface(
    prediction_store: object,
    *,
    strict_canonical_topology: bool = True,
) -> PhysicalProbabilitySurface:
    """Build a label-free surface without importing a Stage-90 predecessor."""

    try:
        cells = tuple(getattr(prediction_store, "cells"))
        rows_by_center = getattr(prediction_store, "rows_by_center")
        cases_by_center = getattr(prediction_store, "case_ids_by_center")
        store_hash = str(getattr(prediction_store, "store_hash"))
    except (AttributeError, TypeError) as exc:
        raise ProtocolError("Neutral prediction store contract is incomplete.") from exc
    output: dict[str, CenterProbabilitySurface] = {}
    for target in CENTERS:
        samples = tuple(str(value) for value in rows_by_center[target])
        cases = tuple(str(value) for value in cases_by_center[target])
        raw_arrays: dict[str, np.ndarray] = {}
        for action in physical_action_ids(target):
            selected = tuple(
                cell
                for cell in cells
                if str(getattr(cell, "target_center")) == target
                and str(getattr(cell, "action_id")) == action
            )
            if len(selected) != SEED_PAIR_COUNT:
                raise ProtocolError("Physical action lacks exact-nine seed cells.")
            try:
                raw_arrays[action] = np.stack(
                    [
                        np.asarray(
                            getattr(cell, "probabilities"), dtype=np.float32
                        )
                        for cell in selected
                    ]
                )
            except ValueError as exc:
                raise ProtocolError(
                    "Physical action probability columns are ragged."
                ) from exc
        samples, cases, arrays = _canonicalize_center_columns(
            samples, cases, raw_arrays
        )
        output[target] = CenterProbabilitySurface(
            target,
            samples,
            cases,
            MappingProxyType(arrays),
            store_hash,
        )
    return PhysicalProbabilitySurface(
        MappingProxyType(output),
        store_hash,
        strict_canonical_topology=strict_canonical_topology,
    )


def _canonicalize_center_columns(
    sample_ids: Sequence[str],
    case_ids: Sequence[str],
    action_arrays: Mapping[str, np.ndarray],
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    Mapping[str, np.ndarray],
]:
    """Apply one label-free canonical identity permutation to every action."""

    rows = canonical_center_row_order(sample_ids, case_ids)
    canonical_arrays: dict[str, np.ndarray] = {}
    for action, value in action_arrays.items():
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (SEED_PAIR_COUNT, len(rows.sample_ids)):
            raise ProtocolError("Physical action column topology drifted.")
        canonical_arrays[str(action)] = np.ascontiguousarray(
            array[:, rows.permutation], dtype=np.float32
        )
    return (
        rows.sample_ids,
        rows.case_ids,
        MappingProxyType(canonical_arrays),
    )


__all__ = ("build_physical_probability_surface",)
