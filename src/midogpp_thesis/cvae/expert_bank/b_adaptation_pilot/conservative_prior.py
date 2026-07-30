"""Fixed source-fit class-conditional diagonal shrinkage diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ....real_features.classifier_reference.artifacts import stable_hash
from ....real_features.classifier_reference.protocol import ProtocolError


CONDITIONAL_PRIOR_FAMILY = "class_conditional_diagonal_standard_shrinkage_rho025_v1"
STANDARD_PRIOR_FAMILY = "standard_normal"


@dataclass(frozen=True)
class ShrunkDiagonalPrior:
    requested_family: str
    realized_family: str
    means: object
    variances: object
    n_rows_by_class: Mapping[str, int]
    n_cases_by_class: Mapping[str, int]
    condition_number_by_class: Mapping[str, float]
    fallback_reason: str
    source_state_hash: str

    @property
    def state_hash(self) -> str:
        def convert(value: object) -> object:
            return value.tolist() if hasattr(value, "tolist") else value

        return stable_hash(
            {
                "schema_version": "midogpp_b_adaptation_shrunk_prior_v1",
                "requested_family": self.requested_family,
                "realized_family": self.realized_family,
                "means": convert(self.means),
                "variances": convert(self.variances),
                "n_rows_by_class": dict(self.n_rows_by_class),
                "n_cases_by_class": dict(self.n_cases_by_class),
                "condition_number_by_class": dict(self.condition_number_by_class),
                "fallback_reason": self.fallback_reason,
                "source_state_hash": self.source_state_hash,
            }
        )


def fit_shrunk_diagonal_prior(
    mu: Sequence[Sequence[float]],
    logvar: Sequence[Sequence[float]],
    labels: Sequence[int],
    case_ids: Sequence[str],
    *,
    rho: float = 0.25,
    variance_clip: tuple[float, float] = (0.25, 4.0),
    min_rows: int = 64,
    min_cases: int = 5,
    max_condition_number: float = 1e4,
    source_state_hash: str,
) -> ShrunkDiagonalPrior:
    import numpy as np

    mu_np = np.asarray(mu, dtype=np.float64)
    logvar_np = np.asarray(logvar, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    cases = np.asarray(case_ids, dtype=str)
    if (
        mu_np.ndim != 2
        or logvar_np.shape != mu_np.shape
        or len(y) != len(mu_np)
        or len(cases) != len(mu_np)
        or not np.isfinite(mu_np).all()
        or not np.isfinite(logvar_np).all()
    ):
        raise ProtocolError("Conditional-prior inputs must be finite aligned arrays.")
    if rho != 0.25 or variance_clip != (0.25, 4.0):
        raise ProtocolError("Pilot conditional-prior shrinkage constants are frozen.")
    latent_dim = int(mu_np.shape[1])
    means = np.zeros((2, latent_dim), dtype=np.float64)
    variances = np.ones((2, latent_dim), dtype=np.float64)
    n_rows: dict[str, int] = {}
    n_cases: dict[str, int] = {}
    conditions: dict[str, float] = {}
    failures: list[str] = []
    for label in (0, 1):
        mask = y == label
        count = int(mask.sum())
        distinct_cases = len(set(cases[mask].tolist()))
        n_rows[str(label)] = count
        n_cases[str(label)] = distinct_cases
        if count < min_rows:
            failures.append(f"class_{label}_rows_below_{min_rows}")
            conditions[str(label)] = 1.0
            continue
        if distinct_cases < min_cases:
            failures.append(f"class_{label}_cases_below_{min_cases}")
            conditions[str(label)] = 1.0
            continue
        class_mu = mu_np[mask]
        class_logvar = logvar_np[mask]
        empirical_mean = class_mu.mean(axis=0)
        total_variance = class_mu.var(axis=0) + np.exp(class_logvar).mean(axis=0)
        shrunk_mean = rho * empirical_mean
        shrunk_variance = np.clip(
            (1.0 - rho) + rho * total_variance,
            variance_clip[0],
            variance_clip[1],
        )
        condition = float(shrunk_variance.max() / shrunk_variance.min())
        conditions[str(label)] = condition
        if not np.isfinite(shrunk_mean).all() or not np.isfinite(shrunk_variance).all():
            failures.append(f"class_{label}_nonfinite")
        elif condition > max_condition_number:
            failures.append(f"class_{label}_condition_number")
        else:
            means[label] = shrunk_mean
            variances[label] = shrunk_variance
    fallback_reason = ";".join(failures)
    realized = CONDITIONAL_PRIOR_FAMILY if not failures else STANDARD_PRIOR_FAMILY
    if failures:
        means.fill(0.0)
        variances.fill(1.0)
    return ShrunkDiagonalPrior(
        requested_family=CONDITIONAL_PRIOR_FAMILY,
        realized_family=realized,
        means=means,
        variances=variances,
        n_rows_by_class=n_rows,
        n_cases_by_class=n_cases,
        condition_number_by_class=conditions,
        fallback_reason=fallback_reason,
        source_state_hash=str(source_state_hash),
    )


def sample_prior(
    prior: ShrunkDiagonalPrior | None,
    labels: Sequence[int],
    *,
    latent_dim: int,
    seed: int,
) -> object:
    import numpy as np

    y = np.asarray(labels, dtype=np.int64)
    if not set(y.tolist()).issubset({0, 1}):
        raise ProtocolError("Pilot prior supports only labels 0/1.")
    rng = np.random.default_rng(int(seed))
    epsilon = rng.normal(size=(len(y), int(latent_dim)))
    if prior is None:
        return epsilon.astype(np.float32)
    means = np.asarray(prior.means, dtype=np.float64)
    variances = np.asarray(prior.variances, dtype=np.float64)
    return (means[y] + epsilon * np.sqrt(variances[y])).astype(np.float32)
