"""Analytic additive expected utility relative to immutable protected P."""

from __future__ import annotations

from functools import reduce
from typing import Iterable, Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import (
    ExpectedDenominators,
    NormalizedUtility,
    PrimitiveUtility,
    RowPosteriorPrediction,
    canonical_sha256,
)
from .surface_hashing import probability_hash


DEFAULT_PROBABILITY_EPSILON = 1.0e-7


def _probability_vector(values: Sequence[float], *, role: str) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=np.float64)
    if (
        result.ndim != 1
        or result.size == 0
        or not np.isfinite(result).all()
        or np.any(result < 0.0)
        or np.any(result > 1.0)
    ):
        raise ProtocolError(f"{role} must be a finite non-empty probability vector.")
    return result


def _typed_eta_vector(
    values: Sequence[RowPosteriorPrediction],
    *,
    expected_length: int | None = None,
) -> tuple[np.ndarray, str, str]:
    rows = tuple(values)
    if not rows or any(not isinstance(value, RowPosteriorPrediction) for value in rows):
        raise ProtocolError("Expected-label probabilities require typed row-posterior predictions.")
    model_hashes = {value.model_hash for value in rows}
    scope_hashes = {value.source_scope_receipt_hash for value in rows}
    if len(model_hashes) != 1 or len(scope_hashes) != 1:
        raise ProtocolError("Row-posterior predictions mixed model or scope lineage.")
    eta = np.ascontiguousarray([value.eta for value in rows], dtype=np.float64)
    if (
        eta.ndim != 1
        or eta.size == 0
        or (expected_length is not None and eta.size != expected_length)
        or not np.isfinite(eta).all()
        or np.any(eta <= 0.0)
        or np.any(eta >= 1.0)
    ):
        raise ProtocolError("Expected-label probabilities must be finite and strictly inside (0, 1).")
    return eta, next(iter(model_hashes)), next(iter(scope_hashes))


def build_expected_denominators(
    eta: Sequence[RowPosteriorPrediction],
    *,
    scope_id: object,
    row_manifest_hash: object,
) -> ExpectedDenominators:
    """Freeze a single expected class-total normalization for every action.

    Callers must construct this once from the complete action-invariant row
    scope.  Candidate-specific rows never alter these denominators.
    """

    values, model_hash, receipt_hash = _typed_eta_vector(eta)
    eta_hash = canonical_sha256(
        {
            "schema": "action_invariant_expected_denominators_v1",
            "eta": tuple(float(value).hex() for value in values),
            "row_count": len(values),
        }
    )
    return ExpectedDenominators(
        scope_id=str(scope_id),
        expected_positive=float(np.sum(values, dtype=np.float64)),
        expected_negative=float(np.sum(1.0 - values, dtype=np.float64)),
        row_count=len(values),
        eta_hash=eta_hash,
        row_manifest_hash=str(row_manifest_hash),
        posterior_model_hash=model_hash,
        posterior_scope_receipt_hash=receipt_hash,
    )


def expected_additive_utility(
    protected_p: Sequence[float],
    candidate_probability: Sequence[float],
    eta: Sequence[RowPosteriorPrediction],
    *,
    action_id: object,
    scope_id: object,
    row_manifest_hash: object,
    epsilon: float = DEFAULT_PROBABILITY_EPSILON,
) -> PrimitiveUtility:
    """Compute exact expected candidate-minus-P additive primitives.

    The function uses no realized target labels.  ``eta`` is the source-only
    row posterior.  Brier and log components are *sums*, while class-count
    components are expected count changes; normalization is a separate step.
    """

    baseline = _probability_vector(protected_p, role="Protected P")
    candidate = _probability_vector(candidate_probability, role="Candidate")
    if candidate.shape != baseline.shape:
        raise ProtocolError("Candidate and protected P probability vectors are misaligned.")
    posterior, model_hash, receipt_hash = _typed_eta_vector(
        eta, expected_length=len(baseline)
    )
    baseline_hash = probability_hash(baseline)
    candidate_hash = probability_hash(candidate)
    if not 0.0 < float(epsilon) < 0.5:
        raise ProtocolError("Log-loss epsilon must lie strictly inside (0, 0.5).")
    if np.array_equal(candidate, baseline):
        return PrimitiveUtility.zeros(
            len(baseline),
            action_id=str(action_id),
            baseline_probability_hash=baseline_hash,
            candidate_probability_hash=candidate_hash,
            scope_id=str(scope_id),
            row_manifest_hash=str(row_manifest_hash),
            posterior_model_hash=model_hash,
            posterior_scope_receipt_hash=receipt_hash,
        )

    baseline_hard = baseline >= 0.5
    candidate_hard = candidate >= 0.5
    hard_delta = candidate_hard.astype(np.float64) - baseline_hard.astype(np.float64)
    delta_tp = float(np.sum(posterior * hard_delta, dtype=np.float64))
    delta_tn = float(np.sum(-(1.0 - posterior) * hard_delta, dtype=np.float64))

    # E[(q-Y)^2 - (p-Y)^2] = q^2 - p^2 - 2 E[Y] (q-p).
    delta_brier_sum = float(
        np.sum(
            candidate * candidate
            - baseline * baseline
            - 2.0 * posterior * (candidate - baseline),
            dtype=np.float64,
        )
    )

    clipped_candidate = np.clip(candidate, float(epsilon), 1.0 - float(epsilon))
    clipped_baseline = np.clip(baseline, float(epsilon), 1.0 - float(epsilon))
    # E[L_log(q,Y)-L_log(p,Y)] in a cancellation-resistant ratio form.
    delta_log_sum = float(
        np.sum(
            -posterior * np.log(clipped_candidate / clipped_baseline)
            - (1.0 - posterior)
            * np.log((1.0 - clipped_candidate) / (1.0 - clipped_baseline)),
            dtype=np.float64,
        )
    )
    return PrimitiveUtility(
        delta_tp=delta_tp,
        delta_tn=delta_tn,
        delta_brier_sum=delta_brier_sum,
        delta_log_sum=delta_log_sum,
        row_count=len(baseline),
        action_id=str(action_id),
        baseline_probability_hash=baseline_hash,
        candidate_probability_hash=candidate_hash,
        scope_id=str(scope_id),
        row_manifest_hash=str(row_manifest_hash),
        posterior_model_hash=model_hash,
        posterior_scope_receipt_hash=receipt_hash,
    )


def sum_primitives(values: Iterable[PrimitiveUtility]) -> PrimitiveUtility:
    """Add case-level primitives before applying any metric denominator."""

    rows = tuple(values)
    if not rows:
        raise ProtocolError("Primitive aggregation requires at least one case.")
    scopes = {row.scope_id for row in rows}
    action_ids = {row.action_id for row in rows}
    baseline_hashes = tuple(sorted(row.baseline_probability_hash for row in rows))
    candidate_hashes = tuple(sorted(row.candidate_probability_hash for row in rows))
    posterior_receipts = {row.posterior_scope_receipt_hash for row in rows}
    posterior_models = {row.posterior_model_hash for row in rows}
    manifests = tuple(sorted(row.row_manifest_hash for row in rows))
    if (
        len(scopes) != 1
        or len(action_ids) != 1
        or len(posterior_models) != 1
        or len(posterior_receipts) != 1
        or len(set(manifests)) != len(manifests)
    ):
        raise ProtocolError("Primitive aggregation requires one scope and unique row manifests.")
    return PrimitiveUtility(
        delta_tp=float(sum(row.delta_tp for row in rows)),
        delta_tn=float(sum(row.delta_tn for row in rows)),
        delta_brier_sum=float(sum(row.delta_brier_sum for row in rows)),
        delta_log_sum=float(sum(row.delta_log_sum for row in rows)),
        row_count=sum(row.row_count for row in rows),
        action_id=next(iter(action_ids)),
        baseline_probability_hash=canonical_sha256(
            {
                "schema": "aggregated_protected_probability_surface_v1",
                "member_hashes": baseline_hashes,
            }
        ),
        candidate_probability_hash=canonical_sha256(
            {
                "schema": "aggregated_candidate_probability_surface_v1",
                "member_hashes": candidate_hashes,
            }
        ),
        scope_id=next(iter(scopes)),
        row_manifest_hash=canonical_sha256(
            {"schema": "aggregated_primitive_row_manifest_v1", "member_hashes": manifests}
        ),
        posterior_model_hash=next(iter(posterior_models)),
        posterior_scope_receipt_hash=next(iter(posterior_receipts)),
    )


def normalize_expected_utility(
    primitive: PrimitiveUtility,
    denominators: ExpectedDenominators,
) -> NormalizedUtility:
    """Apply one action-invariant expected-denominator policy."""

    if (
        primitive.row_count != denominators.row_count
        or primitive.scope_id != denominators.scope_id
        or primitive.row_manifest_hash != denominators.row_manifest_hash
        or primitive.posterior_model_hash != denominators.posterior_model_hash
        or primitive.posterior_scope_receipt_hash
        != denominators.posterior_scope_receipt_hash
    ):
        raise ProtocolError(
            "Primitive and expected denominators must share exact scope, row manifest, and count."
        )
    return NormalizedUtility(
        bacc_gain=0.5
        * (
            primitive.delta_tp / denominators.expected_positive
            + primitive.delta_tn / denominators.expected_negative
        ),
        brier_loss_delta=primitive.delta_brier_sum / denominators.row_count,
        log_loss_delta=primitive.delta_log_sum / denominators.row_count,
        action_id=primitive.action_id,
        baseline_probability_hash=primitive.baseline_probability_hash,
        candidate_probability_hash=primitive.candidate_probability_hash,
        denominator_scope_id=denominators.scope_id,
        denominator_eta_hash=denominators.eta_hash,
        row_manifest_hash=denominators.row_manifest_hash,
        primitive_response_hash=primitive.response_hash,
        posterior_model_hash=denominators.posterior_model_hash,
        posterior_scope_receipt_hash=denominators.posterior_scope_receipt_hash,
    )


__all__ = (
    "DEFAULT_PROBABILITY_EPSILON",
    "build_expected_denominators",
    "expected_additive_utility",
    "normalize_expected_utility",
    "sum_primitives",
)
