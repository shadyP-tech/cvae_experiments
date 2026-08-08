"""Fixed additive residual top-up actions.

The Stage-90 calibrated-energy actions below are intentionally kept separate
from the normalized-midrank/Borda policy actions.  A rank ballot is not a
calibrated energy, even when both ultimately provide weights to the same
Hamilton allocation primitive.
"""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ...protocol import ProtocolError
from .allocation import build_hamilton_topup_allocation
from .contracts import (
    MAX_FINAL_SOURCE_WEIGHT,
    MIN_FINAL_EFFECTIVE_SOURCES,
    ResidualTopupAction,
    SourceClassWindows,
    TopupGeometry,
    immutable_nested_mapping,
)
from .hashing import canonical_sha256


UNIFORM_ACTION_KIND = "exact_uniform_topup_control"
ENERGY_ACTION_KIND = "fixed_calibrated_energy_rank_directed_topup"
SOFTMAX_ENERGY_ACTION_KIND = "nondefault_softmax_energy_directed_topup"
BORDA_ACTION_KIND = "fixed_normalized_midrank_borda_directed_topup"
SINGLE_SOURCE_TAIL_ACTION_KIND = "fixed_single_source_tail_topup_diagnostic"
UNIFORM_DIRECTION_SEMANTICS = "exact_equal_topup_weight"
ENERGY_RANK_DIRECTION_SEMANTICS = (
    "lower_energy_first_canonical_source_ties_linear_k_minus_rank_plus_one"
)
SOFTMAX_DIRECTION_SEMANTICS = "nondefault_softmax_negative_energy"
BORDA_DIRECTION_SEMANTICS = (
    "lower_mean_normalized_midrank_higher_weight_explicit_one_minus_b"
)
SINGLE_SOURCE_TAIL_DIRECTION_SEMANTICS = (
    "all_topup_weight_to_one_declared_source"
)


def build_uniform_topup_action(geometry: TopupGeometry) -> ResidualTopupAction:
    """Build the exact uniform top-up control for a frozen geometry."""

    if geometry.topup_total_per_class % geometry.source_count != 0:
        raise ProtocolError(
            "Exact uniform top-up requires an integer count for every source."
        )
    uniform = 1.0 / float(geometry.source_count)
    return _build_action(
        geometry=geometry,
        action_kind=UNIFORM_ACTION_KIND,
        direction_semantics=UNIFORM_DIRECTION_SEMANTICS,
        direction_weights={source: uniform for source in geometry.source_order},
        calibrated_energy_by_source={},
        temperature=None,
    )


def build_energy_directed_topup_action(
    calibrated_energy_by_source: Mapping[object, float],
    *,
    geometry: TopupGeometry,
) -> ResidualTopupAction:
    """Build the frozen parameter-free lower-energy rank action.

    Sources are sorted by ``(energy, canonical_source_id)``.  Rank one receives
    raw priority ``K`` and rank ``K`` receives raw priority one.  There is no
    temperature, fitted strength, or searched hyperparameter in this action.
    """

    energies, values = _validated_energies(
        calibrated_energy_by_source, geometry=geometry
    )
    ranked_indices = tuple(
        sorted(
            range(geometry.source_count),
            key=lambda index: (
                float(values[index]),
                geometry.source_order[index],
            ),
        )
    )
    raw_priorities = np.zeros(geometry.source_count, dtype=np.float64)
    for rank, index in enumerate(ranked_indices, start=1):
        raw_priorities[index] = float(geometry.source_count - rank + 1)
    direction = raw_priorities / float(raw_priorities.sum())
    return _build_action(
        geometry=geometry,
        action_kind=ENERGY_ACTION_KIND,
        direction_semantics=ENERGY_RANK_DIRECTION_SEMANTICS,
        direction_weights={
            source: float(value)
            for source, value in zip(
                geometry.source_order, direction, strict=True
            )
        },
        calibrated_energy_by_source=energies,
        temperature=None,
    )


def build_softmax_energy_topup_action(
    calibrated_energy_by_source: Mapping[object, float],
    *,
    geometry: TopupGeometry,
    temperature: float,
) -> ResidualTopupAction:
    """Build a non-default softmax action for bounded method diagnostics.

    This helper is intentionally named and tagged as non-default.  The frozen
    experiment-facing energy action is :func:`build_energy_directed_topup_action`.
    """

    if isinstance(temperature, bool):
        raise ProtocolError("Residual top-up temperature must be finite and positive.")
    try:
        tau = float(temperature)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError(
            "Residual top-up temperature must be finite and positive."
        ) from exc
    if not math.isfinite(tau) or tau <= 0.0:
        raise ProtocolError("Residual top-up temperature must be finite and positive.")
    energies, values = _validated_energies(
        calibrated_energy_by_source, geometry=geometry
    )
    if bool(np.all(values == values[0])):
        direction = np.full(
            geometry.source_count,
            1.0 / float(geometry.source_count),
            dtype=np.float64,
        )
    else:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            logits = -(values - float(values.min())) / tau
            exponentials = np.exp(logits)
        normalizer = float(exponentials.sum())
        if (
            not np.isfinite(exponentials).all()
            or not math.isfinite(normalizer)
            or normalizer <= 0.0
        ):
            raise ProtocolError("Residual top-up energy softmax failed closed.")
        direction = exponentials / normalizer
    return _build_action(
        geometry=geometry,
        action_kind=SOFTMAX_ENERGY_ACTION_KIND,
        direction_semantics=SOFTMAX_DIRECTION_SEMANTICS,
        direction_weights={
            source: float(value)
            for source, value in zip(
                geometry.source_order, direction, strict=True
            )
        },
        calibrated_energy_by_source=energies,
        temperature=tau,
    )


def build_borda_directed_topup_action(
    mean_normalized_midrank_by_source: Mapping[object, float],
    *,
    geometry: TopupGeometry,
) -> ResidualTopupAction:
    """Build a fixed Borda action from mean normalized-midrank ballots.

    ``b=0`` is best and ``b=1`` is worst.  Hamilton therefore receives the
    explicit nonnegative priority ``1-b``.  Equal ballot values remain equal
    priorities; canonical source identifiers are used only by Hamilton when an
    indivisible integer remainder must be assigned.

    Ballot values are deliberately *not* written to
    ``calibrated_energy_by_source``.  That field remains reserved for the
    immutable Stage-90 energy actions.
    """

    midranks = _validated_mean_normalized_midranks(
        mean_normalized_midrank_by_source,
        geometry=geometry,
    )
    raw_priorities = {
        source: 1.0 - midranks[source] for source in geometry.source_order
    }
    normalizer = sum(raw_priorities.values())
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise ProtocolError(
            "Mean normalized-midrank ballots must leave positive Borda weight."
        )
    direction_weights = {
        source: raw_priorities[source] / normalizer
        for source in geometry.source_order
    }
    return _build_action(
        geometry=geometry,
        action_kind=BORDA_ACTION_KIND,
        direction_semantics=BORDA_DIRECTION_SEMANTICS,
        direction_weights=direction_weights,
        calibrated_energy_by_source={},
        temperature=None,
    )


def build_single_source_tail_action(
    selected_source: object,
    *,
    geometry: TopupGeometry,
) -> ResidualTopupAction:
    """Build one sealed diagnostic action that assigns the tail to one source."""

    try:
        source = str(selected_source)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError("Single-source tail source is invalid.") from exc
    if (
        not source
        or source.strip() != source
        or source not in geometry.source_order
    ):
        raise ProtocolError(
            "Single-source tail source must be present in the geometry."
        )
    return _build_action(
        geometry=geometry,
        action_kind=SINGLE_SOURCE_TAIL_ACTION_KIND,
        direction_semantics=SINGLE_SOURCE_TAIL_DIRECTION_SEMANTICS,
        direction_weights={
            candidate: float(candidate == source)
            for candidate in geometry.source_order
        },
        calibrated_energy_by_source={},
        temperature=None,
    )


def _build_action(
    *,
    geometry: TopupGeometry,
    action_kind: str,
    direction_semantics: str,
    direction_weights: Mapping[str, float],
    calibrated_energy_by_source: Mapping[str, float],
    temperature: float | None,
) -> ResidualTopupAction:
    hamilton = build_hamilton_topup_allocation(
        direction_weights,
        topup_total_per_class=geometry.topup_total_per_class,
    )
    if hamilton.source_order != geometry.source_order:
        raise ProtocolError("Residual top-up action source order drifted.")
    topup_counts = dict(hamilton.counts)
    final_counts: dict[int, dict[str, int]] = {}
    final_weights: dict[int, dict[str, float]] = {}
    windows: dict[int, dict[str, SourceClassWindows]] = {}
    effective: dict[int, float] = {}
    maximum = 0.0
    for class_label in geometry.class_labels:
        class_counts = {
            source: geometry.base_per_source + topup_counts[source]
            for source in geometry.source_order
        }
        if (
            any(value < geometry.base_per_source for value in class_counts.values())
            or sum(class_counts.values()) != geometry.final_total_per_class
        ):
            raise ProtocolError("Residual top-up final class counts drifted.")
        class_weights = {
            source: value / float(geometry.final_total_per_class)
            for source, value in class_counts.items()
        }
        concentration = sum(value * value for value in class_weights.values())
        class_effective = 1.0 / concentration
        class_maximum = max(class_weights.values())
        if (
            class_maximum > MAX_FINAL_SOURCE_WEIGHT + 1.0e-12
            or class_effective < MIN_FINAL_EFFECTIVE_SOURCES - 1.0e-10
        ):
            raise ProtocolError(
                "Additive residual top-up final density invariants failed."
            )
        final_counts[class_label] = class_counts
        final_weights[class_label] = class_weights
        effective[class_label] = class_effective
        maximum = max(maximum, class_maximum)
        windows[class_label] = {
            source: SourceClassWindows(
                base_start=0,
                base_stop=geometry.base_per_source,
                topup_start=geometry.base_per_source,
                topup_stop=geometry.base_per_source + topup_counts[source],
            )
            for source in geometry.source_order
        }
    final_counts_immutable = immutable_nested_mapping(final_counts)
    final_weights_immutable = immutable_nested_mapping(final_weights)
    windows_immutable = immutable_nested_mapping(windows)
    window_payload = {
        str(label): {
            source: windows[label][source].to_payload()
            for source in geometry.source_order
        }
        for label in geometry.class_labels
    }
    window_hash = canonical_sha256(
        {
            "schema_version": "midogpp_residual_topup_windows_v1",
            "geometry": geometry.to_payload(),
            "windows_by_class": window_payload,
        }
    )
    payload_without_hash = {
        "schema_version": "midogpp_residual_topup_action_v1",
        "geometry": geometry.to_payload(),
        "action_kind": action_kind,
        "direction_semantics": direction_semantics,
        "temperature": temperature,
        "calibrated_energy_by_source": dict(calibrated_energy_by_source),
        "direction_weights": dict(direction_weights),
        "topup_counts": topup_counts,
        "final_counts_by_class": {
            str(label): final_counts[label] for label in geometry.class_labels
        },
        "allocation_hash": hamilton.allocation_hash,
        "window_hash": window_hash,
    }
    return ResidualTopupAction(
        geometry=geometry,
        action_kind=action_kind,
        direction_semantics=direction_semantics,
        temperature=temperature,
        calibrated_energy_by_source=MappingProxyType(
            {
                source: float(calibrated_energy_by_source[source])
                for source in geometry.source_order
                if source in calibrated_energy_by_source
            }
        ),
        direction_weights=MappingProxyType(
            {
                source: float(direction_weights[source])
                for source in geometry.source_order
            }
        ),
        topup_counts=MappingProxyType(topup_counts),
        final_counts_by_class=final_counts_immutable,  # type: ignore[arg-type]
        final_weights_by_class=final_weights_immutable,  # type: ignore[arg-type]
        windows_by_class=windows_immutable,  # type: ignore[arg-type]
        effective_source_count_by_class=MappingProxyType(effective),
        maximum_source_weight=maximum,
        allocation_hash=hamilton.allocation_hash,
        window_hash=window_hash,
        action_hash=canonical_sha256(payload_without_hash),
    )


def _validated_energies(
    calibrated_energy_by_source: Mapping[object, float],
    *,
    geometry: TopupGeometry,
) -> tuple[dict[str, float], np.ndarray]:
    if not isinstance(calibrated_energy_by_source, Mapping):
        raise ProtocolError("Calibrated residual top-up energies must be a mapping.")
    energies: dict[str, float] = {}
    try:
        for raw_source, raw_energy in calibrated_energy_by_source.items():
            source = str(raw_source)
            if not source or source.strip() != source or source in energies:
                raise ProtocolError(
                    "Calibrated residual top-up energy source keys are invalid."
                )
            energies[source] = float(raw_energy)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError("Calibrated residual top-up energies are invalid.") from exc
    if set(energies) != set(geometry.source_order):
        raise ProtocolError(
            "Calibrated residual top-up energies must match the geometry exactly."
        )
    values = np.asarray(
        [energies[source] for source in geometry.source_order], dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise ProtocolError("Calibrated residual top-up energies must be finite.")
    return energies, values


def _validated_mean_normalized_midranks(
    mean_normalized_midrank_by_source: Mapping[object, float],
    *,
    geometry: TopupGeometry,
) -> dict[str, float]:
    if not isinstance(mean_normalized_midrank_by_source, Mapping):
        raise ProtocolError("Mean normalized-midrank ballots must be a mapping.")
    midranks: dict[str, float] = {}
    try:
        for raw_source, raw_midrank in mean_normalized_midrank_by_source.items():
            source = str(raw_source)
            if not source or source.strip() != source or source in midranks:
                raise ProtocolError(
                    "Mean normalized-midrank ballot source keys are invalid."
                )
            if isinstance(raw_midrank, bool):
                raise ProtocolError(
                    "Mean normalized-midrank ballot values are invalid."
                )
            midranks[source] = float(raw_midrank)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError(
            "Mean normalized-midrank ballot values are invalid."
        ) from exc
    if set(midranks) != set(geometry.source_order):
        raise ProtocolError(
            "Mean normalized-midrank ballots must match the geometry exactly."
        )
    values = np.asarray(
        [midranks[source] for source in geometry.source_order],
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ProtocolError(
            "Mean normalized-midrank ballot values must be finite in [0, 1]."
        )
    return midranks


__all__ = (
    "BORDA_ACTION_KIND",
    "BORDA_DIRECTION_SEMANTICS",
    "ENERGY_ACTION_KIND",
    "ENERGY_RANK_DIRECTION_SEMANTICS",
    "SINGLE_SOURCE_TAIL_ACTION_KIND",
    "SINGLE_SOURCE_TAIL_DIRECTION_SEMANTICS",
    "SOFTMAX_DIRECTION_SEMANTICS",
    "SOFTMAX_ENERGY_ACTION_KIND",
    "UNIFORM_ACTION_KIND",
    "UNIFORM_DIRECTION_SEMANTICS",
    "build_borda_directed_topup_action",
    "build_energy_directed_topup_action",
    "build_single_source_tail_action",
    "build_softmax_energy_topup_action",
    "build_uniform_topup_action",
)
