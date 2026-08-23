"""Exact float32 composition of sealed target actions over protected P."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .action_surface import SealedRouteActionSurface, probability_sha256
from .identity import METHOD_MENU, P_METHOD_ID, canonical_hash


@dataclass(frozen=True)
class ComposedCenterPrediction:
    center: str
    method_id: str
    sample_ids: tuple[str, ...]
    probabilities: np.ndarray
    protected_probability_hash: str
    selected_action_hashes: tuple[str, ...]
    selection_enabled: bool
    composition_hash: str = field(init=False)

    def __post_init__(self) -> None:
        values = np.ascontiguousarray(self.probabilities, dtype=np.float32)
        selected = tuple(str(value) for value in self.selected_action_hashes)
        if (
            self.method_id not in METHOD_MENU
            or values.shape != (len(self.sample_ids),)
            or not np.isfinite(values).all()
            or np.any((values < 0.0) | (values > 1.0))
            or len(selected) != len(set(selected))
            or (
                (self.method_id == P_METHOD_ID or not self.selection_enabled)
                and selected
            )
        ):
            raise ProtocolError("P-DCAPS composed center prediction drifted.")
        values.setflags(write=False)
        object.__setattr__(self, "probabilities", values)
        object.__setattr__(self, "selected_action_hashes", selected)
        object.__setattr__(
            self,
            "composition_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_composed_center_prediction_v2",
                    "center": self.center,
                    "method_id": self.method_id,
                    "sample_ids": self.sample_ids,
                    "probability_hash": probability_sha256(values),
                    "protected_probability_hash": self.protected_probability_hash,
                    "selected_action_hashes": selected,
                    "selection_enabled": self.selection_enabled,
                    "target_labels_used": False,
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_composed_center_prediction_v2",
            "center": self.center,
            "method_id": self.method_id,
            "sample_ids": list(self.sample_ids),
            "probability_hash": probability_sha256(self.probabilities),
            "protected_probability_hash": self.protected_probability_hash,
            "selected_action_hashes": list(self.selected_action_hashes),
            "selection_enabled": self.selection_enabled,
            "target_labels_used": False,
            "composition_hash": self.composition_hash,
        }


def compose_center_prediction(
    routes: Sequence[SealedRouteActionSurface],
    *,
    center_sample_order: Sequence[str],
    selected_action_hashes: Sequence[str],
    method_id: str,
    selection_enabled: bool,
) -> ComposedCenterPrediction:
    """Apply at most one selected action per case, preserving all other P bytes."""

    rows = tuple(sorted(tuple(routes), key=lambda row: row.route_key))
    order = tuple(str(value) for value in center_sample_order)
    selected = tuple(str(value) for value in selected_action_hashes)
    if (
        not rows
        or len(order) != len(set(order))
        or any(row.route_key.surface_role != "target" for row in rows)
        or len({row.route_key.held_case_id for row in rows}) != len(rows)
    ):
        raise ProtocolError("P-DCAPS target composition topology drifted.")
    centers = {row.route_key.route_center for row in rows}
    route_samples = tuple(sample for row in rows for sample in row.sample_ids)
    if len(centers) != 1 or set(route_samples) != set(order) or len(route_samples) != len(order):
        raise ProtocolError("P-DCAPS target composition row inventory drifted.")

    action_by_hash = {
        cell.prediction.key.action_key_hash: (route, cell)
        for route in rows
        for cell in route.cells
    }
    if not set(selected).issubset(action_by_hash):
        raise ProtocolError("P-DCAPS selected target action is absent from its seal.")
    selected_routes = [action_by_hash[value][0].route_key for value in selected]
    if len(selected_routes) != len(set(selected_routes)):
        raise ProtocolError("P-DCAPS selected more than one action for a target case.")
    effective_selected = selected if selection_enabled and method_id != P_METHOD_ID else ()
    selected_by_route = {
        action_by_hash[value][0].route_key: action_by_hash[value][1]
        for value in effective_selected
    }

    probability_by_sample: dict[str, np.float32] = {}
    protected_by_sample: dict[str, np.float32] = {}
    for route in rows:
        baseline = route.baseline_probabilities
        cell = selected_by_route.get(route.route_key)
        values = baseline if cell is None else cell.action_probabilities
        for sample_id, protected, value in zip(
            route.sample_ids, baseline, values, strict=True
        ):
            probability_by_sample[sample_id] = np.float32(value)
            protected_by_sample[sample_id] = np.float32(protected)
    probabilities = np.ascontiguousarray(
        [probability_by_sample[value] for value in order], dtype=np.float32
    )
    protected = np.ascontiguousarray(
        [protected_by_sample[value] for value in order], dtype=np.float32
    )
    if not effective_selected and not np.array_equal(probabilities, protected):
        raise ProtocolError("P-DCAPS exact-P fallback changed a probability byte.")
    return ComposedCenterPrediction(
        next(iter(centers)),
        str(method_id),
        order,
        probabilities,
        probability_sha256(protected),
        effective_selected,
        bool(selection_enabled),
    )


__all__ = ("ComposedCenterPrediction", "compose_center_prediction")
