"""Rectangular conditional-centroid contrast operator for CLA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..artifacts import stable_hash
from ..protocol import ProtocolError
from .config import MIN_PENALTY_SCALE, TRACE_ATOL, TRACE_RTOL


@dataclass(frozen=True)
class ConditionalPenaltyOperator:
    """Low-rank rectangular factor ``R`` with ``trace(R.T @ R) == 1``."""

    factor: object
    centers: tuple[str, ...]
    row_keys: tuple[tuple[str, int], ...]
    cell_counts: Mapping[str, int]
    t: float
    trace: float
    rank: int
    factor_hash: str
    centroid_hash: str

    @property
    def operator_hash(self) -> str:
        return self.factor_hash

    @property
    def R(self) -> object:
        """Mathematical-name alias used in numerical checks."""

        return self.factor

    @property
    def n_domains(self) -> int:
        return len(self.centers)

    @property
    def n_cells(self) -> int:
        return len(self.row_keys)

    @property
    def n_features(self) -> int:
        return int(getattr(self.factor, "shape", (0, 0))[1])

    @property
    def maximum_rank(self) -> int:
        return min(self.n_features, 2 * (self.n_domains - 1))

    def apply(self, weights: Sequence[float]) -> object:
        """Return ``R @ w`` without materializing ``R.T @ R``."""

        import numpy as np  # type: ignore

        w = np.asarray(weights, dtype=np.float64)
        r = np.asarray(self.factor, dtype=np.float64)
        if w.ndim != 1 or w.shape[0] != r.shape[1]:
            raise ValueError(
                "Conditional penalty weights must be a 1D vector matching the feature dimension."
            )
        if not np.all(np.isfinite(w)):
            raise ValueError("Conditional penalty weights must contain only finite values.")
        return r @ w

    def value(self, weights: Sequence[float]) -> float:
        """Return the unscaled quadratic value ``||R w||^2``."""

        import numpy as np  # type: ignore

        applied = np.asarray(self.apply(weights), dtype=np.float64)
        return float(applied @ applied)

    def gradient(self, weights: Sequence[float]) -> object:
        """Return ``2 R.T (R w)`` for the unscaled quadratic value."""

        import numpy as np  # type: ignore

        r = np.asarray(self.factor, dtype=np.float64)
        applied = np.asarray(self.apply(weights), dtype=np.float64)
        return 2.0 * (r.T @ applied)

    def audit_payload(self) -> dict[str, object]:
        """Return the immutable numerical construction facts for artifact rows."""

        return {
            "factor_representation": "rectangular_contrast_factor",
            "dense_matrix_materialized": False,
            "normalization": "unit_trace",
            "centroid_weighting": "equal_domain_class_cells",
            "class_centering": "equal_domain_mean_within_class",
            "n_domains": self.n_domains,
            "required_cell_count": 2 * self.n_domains,
            "observed_cell_count": self.n_cells,
            "n_features": self.n_features,
            "factor_shape": [self.n_cells, self.n_features],
            "factor_rank": int(self.rank),
            "maximum_factor_rank": int(self.maximum_rank),
            "factor_trace": float(self.trace),
            "penalty_scale_t": float(self.t),
            "factor_hash": self.factor_hash,
            "operator_hash": self.operator_hash,
            "centroid_hash": self.centroid_hash,
            "centers": list(self.centers),
            "row_order": [
                {"center": center, "class_label": int(label)}
                for center, label in self.row_keys
            ],
            "cell_counts": dict(self.cell_counts),
            "all_cells_present": self.n_cells == 2 * self.n_domains,
        }


def build_conditional_penalty(
    x_fit_scaled: Sequence[Sequence[float]],
    y_fit: Sequence[int],
    centers_fit: Sequence[str],
    *,
    trace_atol: float = TRACE_ATOL,
    trace_rtol: float = TRACE_RTOL,
) -> ConditionalPenaltyOperator:
    """Build the unit-trace rectangular factor from all ``D x 2`` fit cells.

    For every center/class cell, ``c[d,y] = mu[d,y] - mu[y]`` where the
    within-class reference mean gives each fit domain equal mass.  Rows are
    ordered by numeric center and then class ``0, 1``.
    """

    import numpy as np  # type: ignore

    x = np.asarray(x_fit_scaled, dtype=np.float64)
    y = np.asarray(y_fit, dtype=np.int64)
    domains = np.asarray(tuple(str(value) for value in centers_fit), dtype=object)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ProtocolError("Conditional penalty requires a nonempty 2D fit array.")
    if y.ndim != 1 or domains.ndim != 1:
        raise ProtocolError("Conditional penalty labels and centers must be 1D.")
    if x.shape[0] != y.shape[0] or x.shape[0] != domains.shape[0]:
        raise ProtocolError("Conditional penalty inputs must align row-for-row.")
    if not np.all(np.isfinite(x)):
        raise ProtocolError("Conditional penalty fit embeddings contain non-finite values.")
    if set(int(value) for value in y.tolist()) != {0, 1}:
        raise ProtocolError("Conditional penalty requires binary fit labels 0/1.")
    if any(not str(value) for value in domains.tolist()):
        raise ProtocolError("Conditional penalty requires nonempty center IDs.")
    centers = tuple(sorted(set(domains.tolist()), key=_numeric_center_key))
    if len(centers) < 2:
        raise ProtocolError("Conditional penalty requires at least two fit centers.")

    cell_means: dict[tuple[str, int], object] = {}
    cell_counts: dict[str, int] = {}
    for center in centers:
        for label in (0, 1):
            mask = (domains == center) & (y == label)
            count = int(np.count_nonzero(mask))
            key = _cell_key(center, label)
            cell_counts[key] = count
            if count <= 0:
                raise ProtocolError(
                    "Conditional penalty is undefined because a required "
                    f"center x class cell is missing: center={center!r}, class={label}."
                )
            mean = np.asarray(x[mask].mean(axis=0), dtype=np.float64)
            if not np.all(np.isfinite(mean)):
                raise ProtocolError("Conditional penalty cell centroid is non-finite.")
            cell_means[(center, label)] = mean

    class_means = {
        label: np.mean(
            np.stack([cell_means[(center, label)] for center in centers], axis=0),
            axis=0,
        )
        for label in (0, 1)
    }
    row_keys = tuple((center, label) for center in centers for label in (0, 1))
    contrasts = np.stack(
        [
            np.asarray(cell_means[(center, label)], dtype=np.float64)
            - np.asarray(class_means[label], dtype=np.float64)
            for center, label in row_keys
        ],
        axis=0,
    )
    d = len(centers)
    t = float(np.sum(contrasts * contrasts) / float(2 * d))
    if not np.isfinite(t) or not t > MIN_PENALTY_SCALE:
        raise ProtocolError(
            "Conditional penalty is degenerate: normalized centroid scale "
            f"t must be finite and > {MIN_PENALTY_SCALE:.0e}, got {t!r}."
        )
    factor = contrasts / np.sqrt(float(2 * d) * t)
    if not np.all(np.isfinite(factor)):
        raise ProtocolError("Conditional penalty factor contains non-finite values.")
    trace = float(np.sum(factor * factor))
    if not np.isclose(trace, 1.0, atol=float(trace_atol), rtol=float(trace_rtol)):
        raise ProtocolError(
            "Conditional penalty unit-trace normalization failed: "
            f"trace={trace!r}."
        )
    rank = int(np.linalg.matrix_rank(factor))
    if rank <= 0:
        raise ProtocolError("Conditional penalty factor must have positive rank.")
    maximum_rank = min(int(x.shape[1]), 2 * (d - 1))
    if rank > maximum_rank:
        raise ProtocolError(
            "Conditional penalty rank exceeds the two class-centered contrast "
            f"subspaces: rank={rank}, maximum={maximum_rank}."
        )

    centroid_payload = {
        _cell_key(center, label): np.asarray(
            cell_means[(center, label)], dtype=np.float64
        ).tolist()
        for center, label in row_keys
    }
    centroid_hash = stable_hash(
        {
            "cell_centroids": centroid_payload,
            "class_centroids": {
                str(label): np.asarray(class_means[label], dtype=np.float64).tolist()
                for label in (0, 1)
            },
            "row_order": [[center, label] for center, label in row_keys],
        }
    )
    factor_hash = stable_hash(
        {
            "representation": "rectangular_contrast_factor",
            "factor": np.asarray(factor, dtype=np.float64).tolist(),
            "row_order": [[center, label] for center, label in row_keys],
            "t": t,
            "trace": trace,
            "rank": rank,
            "centroid_hash": centroid_hash,
        }
    )
    return ConditionalPenaltyOperator(
        factor=np.asarray(factor, dtype=np.float64),
        centers=centers,
        row_keys=row_keys,
        cell_counts=cell_counts,
        t=t,
        trace=trace,
        rank=rank,
        factor_hash=factor_hash,
        centroid_hash=centroid_hash,
    )


def _numeric_center_key(center: str) -> tuple[int, str]:
    try:
        return int(str(center)), str(center)
    except ValueError as exc:
        raise ProtocolError(
            f"Conditional penalty center IDs must be numeric: {center!r}"
        ) from exc


def _cell_key(center: str, label: int) -> str:
    return f"center={center}|class={int(label)}"


__all__ = ["ConditionalPenaltyOperator", "build_conditional_penalty"]
