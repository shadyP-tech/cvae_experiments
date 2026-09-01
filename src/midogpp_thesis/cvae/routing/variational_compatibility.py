"""Label-blind variational compatibility and source-local calibration.

This module intentionally does not call the model's NELBO helpers.  The score
combines reconstruction MSE in the shared 3840-D feature space with an
analytic KL to the promoted aggregate-posterior (PS) prior.  It is therefore a
variational compatibility *energy*, not an exact NELBO or downstream utility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..generation_samplers import FULL_SAMPLER
from ..protocol import ProtocolError


CLASS_PRIOR = (0.5, 0.5)
ENERGY_SEMANTICS = (
    "class_marginalized_common_space_posterior_mean_reconstruction_mse_plus_"
    "latent_dim_normalized_analytic_ps_kl_fixed_class_prior_half"
)
CALIBRATION_SEMANTICS = (
    "own_source_case_equal_median_mad_robust_z_then_fixed_three_seed_mean"
)
DEFAULT_TRAINING_SEEDS = (17, 42, 101)
DEFAULT_SCALE_FLOOR = 1e-6


@dataclass(frozen=True, order=True)
class ReplicaKey:
    source_center: str
    training_seed: int


@dataclass(frozen=True)
class CompatibilityEnergy:
    """Row- and case-level label-blind energy for one expert replica."""

    source_center: str
    training_seed: int
    case_order: tuple[str, ...]
    per_case: Mapping[str, float]
    per_row: np.ndarray
    per_class_energy: Mapping[int, np.ndarray]
    per_class_reconstruction_mse: Mapping[int, np.ndarray]
    per_class_normalized_ps_kl: Mapping[int, np.ndarray]
    case_equal_mean: float
    energy_semantics: str = ENERGY_SEMANTICS
    class_prior: tuple[float, float] = CLASS_PRIOR
    exact_nelbo: bool = False
    labels_consumed: bool = False


@dataclass(frozen=True)
class ReplicaCalibration:
    source_center: str
    training_seed: int
    query_case_equal_mean: float
    own_source_location: float
    own_source_raw_mad: float
    own_source_sample_std: float
    scale: float
    scale_source: str
    calibrated_z: float
    query_case_count: int
    own_source_case_count: int


@dataclass(frozen=True)
class OwnSourceCalibration:
    """Fixed-seed robust calibration with no replica or seed selection."""

    replicas: tuple[ReplicaCalibration, ...]
    mean_z_by_source: Mapping[str, float]
    candidate_sources: tuple[str, ...]
    training_seeds: tuple[int, ...]
    scale_floor: float
    calibration_semantics: str = CALIBRATION_SEMANTICS


def gaussian_kl_diagonal_to_full(
    posterior_mean: Sequence[Sequence[float]] | np.ndarray,
    posterior_log_variance: Sequence[Sequence[float]] | np.ndarray,
    prior_mean: Sequence[float] | np.ndarray,
    prior_covariance: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Return analytic ``KL(q_diag || p_full)`` for every posterior row.

    ``q`` is parameterized by a row-wise mean and log variance.  ``p`` is one
    positive-definite full Gaussian.  A Cholesky solve is used instead of a
    direct matrix inverse for the quadratic term and log determinant.
    """

    mu_q = np.asarray(posterior_mean, dtype=np.float64)
    logvar_q = np.asarray(posterior_log_variance, dtype=np.float64)
    mu_p = np.asarray(prior_mean, dtype=np.float64)
    covariance_p = np.asarray(prior_covariance, dtype=np.float64)
    if (
        mu_q.ndim != 2
        or not len(mu_q)
        or logvar_q.shape != mu_q.shape
        or mu_p.shape != (mu_q.shape[1],)
        or covariance_p.shape != (mu_q.shape[1], mu_q.shape[1])
    ):
        raise ProtocolError("Analytic PS KL inputs have incompatible geometry.")
    if not all(
        np.isfinite(value).all()
        for value in (mu_q, logvar_q, mu_p, covariance_p)
    ):
        raise ProtocolError("Analytic PS KL inputs must be finite.")
    if not np.allclose(covariance_p, covariance_p.T, rtol=1e-10, atol=1e-12):
        raise ProtocolError("Analytic PS KL prior covariance must be symmetric.")
    try:
        cholesky = np.linalg.cholesky(covariance_p)
    except np.linalg.LinAlgError as exc:
        raise ProtocolError(
            "Analytic PS KL prior covariance must be positive definite."
        ) from exc

    latent_dim = int(mu_q.shape[1])
    inverse_cholesky = np.linalg.solve(
        cholesky, np.eye(latent_dim, dtype=np.float64)
    )
    precision_diagonal = np.sum(inverse_cholesky * inverse_cholesky, axis=0)
    posterior_variance = np.exp(logvar_q)
    if not np.isfinite(posterior_variance).all():
        raise ProtocolError("Analytic PS KL posterior variance overflowed.")
    trace_term = posterior_variance @ precision_diagonal
    delta = mu_q - mu_p[None, :]
    whitened_delta = np.linalg.solve(cholesky, delta.T).T
    quadratic_term = np.sum(whitened_delta * whitened_delta, axis=1)
    logdet_p = 2.0 * float(np.log(np.diag(cholesky)).sum())
    logdet_q = np.sum(logvar_q, axis=1)
    kl = 0.5 * (
        trace_term
        + quadratic_term
        - float(latent_dim)
        + logdet_p
        - logdet_q
    )
    if not np.isfinite(kl).all() or float(kl.min()) < -1e-8:
        raise ProtocolError("Analytic PS KL produced an invalid value.")
    return np.maximum(kl, 0.0)


def score_variational_compatibility(
    expert: object,
    common_embeddings: Sequence[Sequence[float]] | np.ndarray,
    case_ids: Sequence[str],
) -> CompatibilityEnergy:
    """Score unlabeled query rows under both fixed class hypotheses.

    Each class energy is shared-space posterior-mean reconstruction MSE plus
    ``KL(q_diag || p_PS_full) / latent_dim``.  The two hypotheses are combined
    as ``-log(sum_y 0.5 * exp(-energy_y))``.  Target labels are neither accepted
    nor inferred.
    """

    import torch

    required = ("source_center", "training_seed", "model", "source_frame", "sampler")
    if any(not hasattr(expert, field) for field in required):
        raise ProtocolError("Compatibility scoring requires one promoted expert replica.")
    common = np.asarray(common_embeddings, dtype=np.float32)
    cases = tuple(str(value) for value in case_ids)
    source_frame = expert.source_frame
    frame = getattr(source_frame, "frame", None)
    model = expert.model
    sampler = expert.sampler
    if (
        common.ndim != 2
        or not len(common)
        or len(cases) != len(common)
        or any(not value for value in cases)
        or frame is None
        or not np.isfinite(common).all()
    ):
        raise ProtocolError("Compatibility query rows are invalid or misaligned.")
    if int(getattr(frame, "input_dim", -1)) != int(common.shape[1]):
        raise ProtocolError("Compatibility query is outside the expert common frame.")
    if int(getattr(model, "n_classes", -1)) != 2:
        raise ProtocolError("Compatibility energy requires the frozen binary CVAE.")
    latent_dim = int(getattr(model, "latent_dim", -1))
    if latent_dim <= 0 or int(getattr(sampler, "latent_dim", -1)) != latent_dim:
        raise ProtocolError("Compatibility model and promoted PS prior disagree.")
    projected = np.asarray(frame.transform(common), dtype=np.float32)
    if projected.shape != (len(common), int(getattr(model, "input_dim", -1))):
        raise ProtocolError("Compatibility expert frame produced invalid coordinates.")
    try:
        device = next(model.parameters()).device
    except (StopIteration, AttributeError) as exc:
        raise ProtocolError("Compatibility CVAE has no model parameters.") from exc
    x = torch.as_tensor(projected, dtype=torch.float32, device=device)

    class_energy: dict[int, np.ndarray] = {}
    reconstruction_mse: dict[int, np.ndarray] = {}
    normalized_kl: dict[int, np.ndarray] = {}
    model.eval()
    with torch.no_grad():
        for class_label in (0, 1):
            state = sampler.classes.get(class_label)
            if state is None or state.realized_family != FULL_SAMPLER:
                raise ProtocolError(
                    "Compatibility energy requires the promoted full PS Gaussian."
                )
            y = torch.full(
                (len(x),), class_label, dtype=torch.long, device=device
            )
            mu, logvar = model.encode(x, y)
            decoded_projected = model.decode(mu, y).detach().cpu().numpy()
            mu_np = mu.detach().cpu().numpy()
            logvar_np = logvar.detach().cpu().numpy()
            decoded_common = np.asarray(
                frame.inverse_transform(decoded_projected), dtype=np.float64
            )
            if decoded_common.shape != common.shape or not np.isfinite(decoded_common).all():
                raise ProtocolError(
                    "Compatibility posterior-mean reconstruction left the common frame."
                )
            mse = np.mean(
                (decoded_common - common.astype(np.float64)) ** 2,
                axis=1,
                dtype=np.float64,
            )
            kl = gaussian_kl_diagonal_to_full(
                mu_np,
                logvar_np,
                state.mean,
                state.covariance,
            ) / float(latent_dim)
            energy = mse + kl
            if not np.isfinite(energy).all():
                raise ProtocolError("Compatibility class energy is non-finite.")
            reconstruction_mse[class_label] = _readonly(mse)
            normalized_kl[class_label] = _readonly(kl)
            class_energy[class_label] = _readonly(energy)

    log_half = float(np.log(CLASS_PRIOR[0]))
    per_row = -np.logaddexp(
        log_half - class_energy[0],
        log_half - class_energy[1],
    )
    case_order = tuple(sorted(set(cases)))
    per_case = {
        case_id: float(
            np.mean(per_row[np.fromiter((value == case_id for value in cases), bool)])
        )
        for case_id in case_order
    }
    case_equal_mean = float(np.mean(tuple(per_case.values()), dtype=np.float64))
    return CompatibilityEnergy(
        source_center=str(expert.source_center),
        training_seed=int(expert.training_seed),
        case_order=case_order,
        per_case=per_case,
        per_row=_readonly(per_row),
        per_class_energy=class_energy,
        per_class_reconstruction_mse=reconstruction_mse,
        per_class_normalized_ps_kl=normalized_kl,
        case_equal_mean=case_equal_mean,
    )


def calibrate_own_source_energies(
    query_case_energies_by_replica: Mapping[
        ReplicaKey | tuple[str, int], Mapping[str, float] | Sequence[float]
    ],
    own_source_case_energies_by_replica: Mapping[
        ReplicaKey | tuple[str, int], Mapping[str, float] | Sequence[float]
    ],
    *,
    candidate_sources: Sequence[str],
    training_seeds: Sequence[int] = DEFAULT_TRAINING_SEEDS,
    scale_floor: float = DEFAULT_SCALE_FLOOR,
) -> OwnSourceCalibration:
    """Robustly calibrate each replica, then mean all retained seeds.

    For replica ``(e, s)``, the own-source location is the case median and the
    scale is ``1.4826 * MAD``.  A degenerate MAD falls back to sample standard
    deviation, then to the fixed positive floor.  The query statistic is its
    case-equal mean.  Exactly the declared source/seed Cartesian product must
    be present, which makes post-hoc seed selection impossible.
    """

    candidates = tuple(sorted(str(value) for value in candidate_sources))
    seeds = tuple(int(value) for value in training_seeds)
    floor = float(scale_floor)
    if (
        not candidates
        or len(candidates) != len(set(candidates))
        or seeds != DEFAULT_TRAINING_SEEDS
        or not np.isfinite(floor)
        or floor <= 0.0
    ):
        raise ProtocolError(
            "Own-source calibration requires the frozen three-seed replicate policy."
        )
    query = _normalize_replica_mapping(query_case_energies_by_replica)
    own = _normalize_replica_mapping(own_source_case_energies_by_replica)
    expected = {
        ReplicaKey(source_center, seed)
        for source_center in candidates
        for seed in seeds
    }
    if set(query) != expected or set(own) != expected:
        raise ProtocolError(
            "Own-source calibration requires every declared source/seed replica exactly once."
        )

    rows: list[ReplicaCalibration] = []
    for key in sorted(expected):
        query_values = _case_values(query[key], role="query")
        own_values = _case_values(own[key], role="own-source")
        query_mean = float(np.mean(query_values, dtype=np.float64))
        location = float(np.median(own_values))
        raw_mad = float(np.median(np.abs(own_values - location)))
        robust_scale = 1.4826 * raw_mad
        sample_std = (
            float(np.std(own_values, ddof=1, dtype=np.float64))
            if len(own_values) >= 2
            else 0.0
        )
        if np.isfinite(robust_scale) and robust_scale >= floor:
            scale = robust_scale
            scale_source = "scaled_mad"
        elif np.isfinite(sample_std) and sample_std >= floor:
            scale = sample_std
            scale_source = "sample_std_fallback"
        else:
            scale = floor
            scale_source = "fixed_floor_fallback"
        calibrated_z = (query_mean - location) / scale
        if not np.isfinite(calibrated_z):
            raise ProtocolError("Own-source calibration produced a non-finite z score.")
        rows.append(
            ReplicaCalibration(
                source_center=key.source_center,
                training_seed=key.training_seed,
                query_case_equal_mean=query_mean,
                own_source_location=location,
                own_source_raw_mad=raw_mad,
                own_source_sample_std=sample_std,
                scale=scale,
                scale_source=scale_source,
                calibrated_z=float(calibrated_z),
                query_case_count=len(query_values),
                own_source_case_count=len(own_values),
            )
        )
    mean_z_by_source = {
        source: float(
            np.mean(
                [row.calibrated_z for row in rows if row.source_center == source],
                dtype=np.float64,
            )
        )
        for source in candidates
    }
    return OwnSourceCalibration(
        replicas=tuple(rows),
        mean_z_by_source=mean_z_by_source,
        candidate_sources=candidates,
        training_seeds=seeds,
        scale_floor=floor,
    )


def _normalize_replica_mapping(
    values: Mapping[
        ReplicaKey | tuple[str, int], Mapping[str, float] | Sequence[float]
    ],
) -> dict[ReplicaKey, Mapping[str, float] | Sequence[float]]:
    normalized: dict[ReplicaKey, Mapping[str, float] | Sequence[float]] = {}
    for raw_key, raw_values in values.items():
        if isinstance(raw_key, ReplicaKey):
            key = raw_key
        elif isinstance(raw_key, tuple) and len(raw_key) == 2:
            key = ReplicaKey(str(raw_key[0]), int(raw_key[1]))
        else:
            raise ProtocolError("Calibration replica keys must be (source, seed).")
        if key in normalized:
            raise ProtocolError("Calibration replica key was duplicated.")
        normalized[key] = raw_values
    return normalized


def _case_values(
    raw: Mapping[str, float] | Sequence[float],
    *,
    role: str,
) -> np.ndarray:
    if isinstance(raw, Mapping):
        values = np.asarray(
            [
                float(value)
                for _, value in sorted(raw.items(), key=lambda item: str(item[0]))
            ],
            dtype=np.float64,
        )
    else:
        values = np.asarray(tuple(float(value) for value in raw), dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ProtocolError(f"Calibration {role} case energies must be finite and nonempty.")
    return values


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=np.float64)
    result.setflags(write=False)
    return result


__all__ = (
    "CALIBRATION_SEMANTICS",
    "CLASS_PRIOR",
    "DEFAULT_SCALE_FLOOR",
    "DEFAULT_TRAINING_SEEDS",
    "ENERGY_SEMANTICS",
    "CompatibilityEnergy",
    "OwnSourceCalibration",
    "ReplicaCalibration",
    "ReplicaKey",
    "calibrate_own_source_energies",
    "gaussian_kl_diagonal_to_full",
    "score_variational_compatibility",
)
