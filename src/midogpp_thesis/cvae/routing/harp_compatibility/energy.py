"""Neutral, label-free variational-energy primitives for HARP.

The inputs are intentionally model-agnostic.  A producer supplies a per-row,
per-class reconstruction *distortion* and prior-rate term.  HARP combines the
two fixed class hypotheses without accepting a target label.  Unless the
producer separately proves a normalized observation likelihood, this is an
energy and not an exact NELBO.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError


ENERGY_SEMANTICS = (
    "label_free_fixed_class_prior_log_marginal_of_"
    "reconstruction_distortion_plus_beta_times_prior_rate"
)


@dataclass(frozen=True)
class VariationalEnergySurface:
    """One immutable label-free energy surface for an expert replica."""

    source_center: str
    training_seed: int
    row_ids: tuple[str, ...]
    case_ids: tuple[str, ...]
    per_class_energy: Mapping[int, np.ndarray]
    per_row: np.ndarray
    beta: float
    class_prior: tuple[float, float]
    energy_semantics: str = ENERGY_SEMANTICS
    exact_nelbo: bool = False
    labels_consumed: bool = False

    def __post_init__(self) -> None:
        source = str(self.source_center)
        rows = tuple(str(value) for value in self.row_ids)
        cases = tuple(str(value) for value in self.case_ids)
        values = {
            int(label): _readonly(array) for label, array in self.per_class_energy.items()
        }
        marginal = _readonly(self.per_row)
        prior = tuple(float(value) for value in self.class_prior)
        if (
            not source
            or isinstance(self.training_seed, bool)
            or int(self.training_seed) < 0
            or not rows
            or len(rows) != len(set(rows))
            or len(cases) != len(rows)
            or any(not value for value in (*rows, *cases))
            or set(values) != {0, 1}
            or any(value.shape != (len(rows),) for value in values.values())
            or marginal.shape != (len(rows),)
            or len(prior) != 2
            or any(not np.isfinite(value) or value <= 0.0 for value in prior)
            or not np.isclose(sum(prior), 1.0, rtol=0.0, atol=1e-12)
            or not np.isfinite(float(self.beta))
            or float(self.beta) < 0.0
            or self.energy_semantics != ENERGY_SEMANTICS
            or bool(self.exact_nelbo)
            or bool(self.labels_consumed)
        ):
            raise ProtocolError("HARP variational-energy surface is invalid.")
        object.__setattr__(self, "source_center", source)
        object.__setattr__(self, "training_seed", int(self.training_seed))
        object.__setattr__(self, "row_ids", rows)
        object.__setattr__(self, "case_ids", cases)
        object.__setattr__(self, "per_class_energy", MappingProxyType(values))
        object.__setattr__(self, "per_row", marginal)
        object.__setattr__(self, "beta", float(self.beta))
        object.__setattr__(self, "class_prior", prior)

    @property
    def case_equal_mean(self) -> float:
        means = []
        case_array = np.asarray(self.case_ids, dtype=object)
        for case_id in sorted(set(self.case_ids)):
            means.append(float(np.mean(self.per_row[case_array == case_id])))
        return float(np.mean(means, dtype=np.float64))


def class_marginal_variational_energy(
    *,
    source_center: str,
    training_seed: int,
    row_ids: Sequence[str],
    case_ids: Sequence[str],
    reconstruction_distortion: Sequence[Sequence[float]] | np.ndarray,
    prior_rate: Sequence[Sequence[float]] | np.ndarray,
    beta: float = 1.0,
    class_prior: Sequence[float] = (0.5, 0.5),
) -> VariationalEnergySurface:
    """Combine fixed class hypotheses into a label-free energy.

    Arrays have shape ``[row, class]`` in canonical class order ``(0, 1)``.
    This function deliberately has no label argument.
    """

    distortion = np.asarray(reconstruction_distortion, dtype=np.float64)
    rate = np.asarray(prior_rate, dtype=np.float64)
    rows = tuple(str(value) for value in row_ids)
    cases = tuple(str(value) for value in case_ids)
    weight = float(beta)
    prior = np.asarray(tuple(float(value) for value in class_prior), dtype=np.float64)
    if (
        distortion.ndim != 2
        or distortion.shape != (len(rows), 2)
        or rate.shape != distortion.shape
        or len(cases) != len(rows)
        or not len(rows)
        or not np.isfinite(distortion).all()
        or not np.isfinite(rate).all()
        or float(np.min(distortion)) < 0.0
        or float(np.min(rate)) < -1e-10
        or not np.isfinite(weight)
        or weight < 0.0
        or prior.shape != (2,)
        or not np.isfinite(prior).all()
        or np.any(prior <= 0.0)
        or not np.isclose(float(prior.sum()), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise ProtocolError("HARP variational-energy inputs are invalid or misaligned.")
    energy = distortion + weight * np.maximum(rate, 0.0)
    logits = np.log(prior)[None, :] - energy
    maximum = np.max(logits, axis=1)
    marginal = -(maximum + np.log(np.exp(logits - maximum[:, None]).sum(axis=1)))
    if not np.isfinite(marginal).all():
        raise ProtocolError("HARP class-marginal energy is non-finite.")
    return VariationalEnergySurface(
        source_center=str(source_center),
        training_seed=int(training_seed),
        row_ids=rows,
        case_ids=cases,
        per_class_energy={0: energy[:, 0], 1: energy[:, 1]},
        per_row=marginal,
        beta=weight,
        class_prior=(float(prior[0]), float(prior[1])),
    )


def _readonly(value: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ProtocolError("HARP variational energy must be finite.")
    array.setflags(write=False)
    return array


__all__ = (
    "ENERGY_SEMANTICS",
    "VariationalEnergySurface",
    "class_marginal_variational_energy",
)
