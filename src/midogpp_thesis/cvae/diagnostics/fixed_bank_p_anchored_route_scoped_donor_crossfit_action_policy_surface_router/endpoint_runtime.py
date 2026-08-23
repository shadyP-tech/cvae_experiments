"""Package-local B/I/R/P endpoint reconstruction from the fresh physical bank."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...runtime.artifact_io import sha256_array
from .identity import ACTION_FAMILIES, DIRECTIONS, canonical_hash, require_sha256
from .label_firewall import (
    SupportLabelCapability,
    require_support_label_capability,
)
from .physical_adapter import (
    B_ACTION_ID,
    CenterPhysicalSurface,
    a1_action_id,
    candidate_sources,
)


PORTFOLIO_IDENTIFICATION_WEIGHT = 3.0 / 5.0
PORTFOLIO_ROBUST_WEIGHT = 2.0 / 5.0
ROBUST_K_GRID = (4, 5, 6)
ROBUST_W_GRID = (0.5, 0.6, 0.7)


@dataclass(frozen=True)
class EndpointPrediction:
    center: str
    case_id: str
    sample_ids: tuple[str, ...]
    probabilities: tuple[tuple[str, np.ndarray], ...]
    support_case_ids: tuple[str, ...]
    support_capability_hash: str
    excluded_source_centers: tuple[str, ...]
    physical_surface_hash: str
    center_surface_hash: str
    endpoint_hash: str = field(init=False)

    def __post_init__(self) -> None:
        arrays: list[tuple[str, np.ndarray]] = []
        for method, values in self.probabilities:
            array = np.ascontiguousarray(values, dtype=np.float32)
            if array.shape != (len(self.sample_ids),) or not np.isfinite(array).all():
                raise ProtocolError("P-DCAPS endpoint probability drifted.")
            array.setflags(write=False)
            arrays.append((str(method), array))
        if (
            tuple(method for method, _ in arrays)
            != (*ACTION_FAMILIES, "P_PROTECTED")
            or self.case_id in self.support_case_ids
        ):
            raise ProtocolError("P-DCAPS endpoint topology drifted.")
        require_sha256(self.support_capability_hash, "support capability")
        object.__setattr__(self, "probabilities", tuple(arrays))
        object.__setattr__(
            self,
            "endpoint_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_endpoint_prediction_v1",
                    "center": self.center,
                    "case_id": self.case_id,
                    "sample_ids": self.sample_ids,
                    "probabilities": [
                        [method, sha256_array(values)] for method, values in arrays
                    ],
                    "support_case_ids": self.support_case_ids,
                    "support_capability_hash": self.support_capability_hash,
                    "excluded_source_centers": self.excluded_source_centers,
                    "physical_surface_hash": self.physical_surface_hash,
                    "center_surface_hash": self.center_surface_hash,
                    "held_case_labels_used": False,
                }
            ),
        )

    def probability(self, method: str) -> np.ndarray:
        try:
            return dict(self.probabilities)[str(method)]
        except KeyError as exc:
            raise ProtocolError("P-DCAPS endpoint method is absent.") from exc


def _support_label_vector(
    surface: CenterPhysicalSurface,
    held_case_id: str,
    support_capability: SupportLabelCapability,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    held = str(held_case_id)
    positions = np.flatnonzero(np.asarray(surface.case_ids) != held)
    support_cases = tuple(sorted(set(surface.case_ids).difference({held})))
    expected = {
        (surface.center, surface.case_ids[position], surface.sample_ids[position])
        for position in positions
    }
    expected_keys = tuple(
        (surface.center, surface.case_ids[position], surface.sample_ids[position])
        for position in positions
    )
    rows = require_support_label_capability(
        support_capability,
        center=surface.center,
        held_case_id=held,
        expected_keys=expected_keys,
    )
    labels = {row.key: row.value for row in rows}
    if (
        not support_cases
        or len(rows) != len(labels)
        or set(labels) != expected
        or any(row.case_id == held for row in rows)
    ):
        raise ProtocolError("P-DCAPS endpoint support label scope drifted.")
    truth = np.asarray(
        [
            labels[(surface.center, surface.case_ids[position], surface.sample_ids[position])]
            for position in positions
        ],
        dtype=np.int8,
    )
    return positions, truth, support_cases


def _direction_gain(
    baseline: np.ndarray,
    candidate: np.ndarray,
    truth: np.ndarray,
    direction: str,
) -> float:
    b = baseline >= 0.5
    a = candidate >= 0.5
    selected = (~b & a) if direction == DIRECTIONS[0] else (b & ~a)
    n_positive = int(np.sum(truth == 1))
    n_negative = int(np.sum(truth == 0))
    if n_positive <= 0 or n_negative <= 0:
        raise ProtocolError("P-DCAPS endpoint support lacks both classes.")
    if direction == DIRECTIONS[0]:
        favorable = np.sum(selected & (truth == 1), dtype=np.int64)
        harmful = np.sum(selected & (truth == 0), dtype=np.int64)
        favorable_denominator = n_positive
        harmful_denominator = n_negative
    else:
        favorable = np.sum(selected & (truth == 0), dtype=np.int64)
        harmful = np.sum(selected & (truth == 1), dtype=np.int64)
        favorable_denominator = n_negative
        harmful_denominator = n_positive
    return 0.5 * (
        float(favorable) / favorable_denominator
        - float(harmful) / harmful_denominator
    )


def build_case_endpoints(
    surface: CenterPhysicalSurface,
    *,
    physical_surface_hash: str,
    held_case_id: str,
    support_capability: SupportLabelCapability,
    donor_priors: Mapping[tuple[str, str], float] | None = None,
    excluded_source_centers: Sequence[str] = (),
) -> EndpointPrediction:
    """Build B/I/R/P with whole-case support and optional outer-source exclusion.

    The method intentionally keeps endpoint construction independent from the
    learned P-DCAPS response models.  Donor priors are frozen before the held
    case is scored and may not contain the outer or scored center.
    """

    positions, truth, support_cases = _support_label_vector(
        surface, held_case_id, support_capability
    )
    raw_excluded = tuple(str(value) for value in excluded_source_centers)
    excluded = tuple(center for center in candidate_sources(surface.center) if center in set(raw_excluded))
    if len(excluded) != len(raw_excluded):
        raise ProtocolError("P-DCAPS endpoint source exclusion drifted.")
    sources = tuple(
        source for source in candidate_sources(surface.center) if source not in excluded
    )
    if not sources:
        raise ProtocolError("P-DCAPS endpoint source exclusion drifted.")
    expected_prior_keys = {
        (source, direction) for source in sources for direction in DIRECTIONS
    }
    if donor_priors is not None and (
        set(donor_priors) != expected_prior_keys
        or any(not np.isfinite(float(value)) for value in donor_priors.values())
    ):
        raise ProtocolError("P-DCAPS donor-prior capability escaped its route scope.")
    priors = {
        (source, direction): float(
            0.0 if donor_priors is None else donor_priors[(source, direction)]
        )
        for source in sources
        for direction in DIRECTIONS
    }
    baseline_all = surface.exact_nine_mean(B_ACTION_ID)
    support_baseline = baseline_all[positions]
    gains: dict[tuple[str, str], float] = {}
    for source in sources:
        support_candidate = surface.exact_nine_mean(a1_action_id(source))[positions]
        for direction in DIRECTIONS:
            gains[(source, direction)] = _direction_gain(
                support_baseline, support_candidate, truth, direction
            )

    held_positions = surface.positions(held_case_id)
    baseline = np.asarray(baseline_all[held_positions], dtype=np.float64)
    baseline_hard = baseline >= 0.5
    selected_identity: dict[str, str | None] = {}
    for direction in DIRECTIONS:
        case_scale = float(np.mean([abs(gains[(source, direction)]) for source in sources]))
        prior_scale = float(np.mean([abs(priors[(source, direction)]) for source in sources]))
        scores = {
            source: (
                0.8 * (0.0 if case_scale == 0.0 else gains[(source, direction)] / case_scale)
                + 0.2 * (0.0 if prior_scale == 0.0 else priors[(source, direction)] / prior_scale)
            )
            for source in sources
            if gains[(source, direction)] > 0.0
        }
        if not scores or max(scores.values()) <= 1.0e-12:
            selected_identity[direction] = None
        else:
            maximum = max(scores.values())
            selected_identity[direction] = min(
                (source for source, value in scores.items() if maximum - value <= 1.0e-12),
                key=int,
            )

    identification = baseline.copy()
    for branch, direction in ((False, DIRECTIONS[0]), (True, DIRECTIONS[1])):
        source = selected_identity[direction]
        if source is not None:
            mask = baseline_hard == branch
            identification[mask] = surface.exact_nine_mean(a1_action_id(source))[held_positions][mask]

    arms: list[np.ndarray] = []
    for k in ROBUST_K_GRID:
        for blend in ROBUST_W_GRID:
            selections: dict[str, str | None] = {}
            for direction in DIRECTIONS:
                ranked = tuple(
                    sorted(
                        sources,
                        key=lambda source: (-priors[(source, direction)], int(source)),
                    )
                )[: min(k, len(sources))]
                scores: dict[str | None, float] = {None: 0.0}
                scores.update(
                    {
                        source: blend * gains[(source, direction)]
                        + (1.0 - blend) * priors[(source, direction)]
                        for source in ranked
                    }
                )
                maximum = max(scores.values())
                selections[direction] = min(
                    (
                        source
                        for source, value in scores.items()
                        if maximum - value <= 1.0e-12
                    ),
                    key=lambda source: -1 if source is None else int(source),
                )
            values = baseline.copy()
            for branch, direction in ((False, DIRECTIONS[0]), (True, DIRECTIONS[1])):
                source = selections[direction]
                if source is not None:
                    mask = baseline_hard == branch
                    values[mask] = surface.exact_nine_mean(a1_action_id(source))[held_positions][mask]
            arms.append(values)
    robust = np.mean(np.stack(arms), axis=0, dtype=np.float64)
    portfolio = (
        PORTFOLIO_IDENTIFICATION_WEIGHT * identification
        + PORTFOLIO_ROBUST_WEIGHT * robust
    )
    endpoints = tuple(
        (
            method,
            np.ascontiguousarray(values, dtype=np.float32),
        )
        for method, values in (
            (ACTION_FAMILIES[0], baseline),
            (ACTION_FAMILIES[1], identification),
            (ACTION_FAMILIES[2], robust),
            ("P_PROTECTED", portfolio),
        )
    )
    return EndpointPrediction(
        surface.center,
        str(held_case_id),
        tuple(surface.sample_ids[position] for position in held_positions),
        endpoints,
        support_cases,
        support_capability.capability_hash,
        excluded,
        physical_surface_hash,
        surface.center_surface_hash,
    )


__all__ = (
    "EndpointPrediction",
    "build_case_endpoints",
)
