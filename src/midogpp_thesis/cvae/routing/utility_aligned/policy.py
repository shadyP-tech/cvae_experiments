"""Conservative utility-aligned policy construction."""

from __future__ import annotations

from collections import defaultdict
import math
from numbers import Integral
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from ..residual_topup.hashing import canonical_sha256
from .policy_contracts import (
    ABSTENTION_SEMANTICS,
    BASE_ACTION_ID,
    GLOBAL_ACTION_ID,
    PERMUTATION_ACTION_ID,
    ROUTED_ACTION_ID,
    UtilityAlignedPolicy,
)
from .result_contracts import CardinalityTransferResult, UtilityAlignedModels
from .row_contracts import (
    MIN_SUPPORT_BOOTSTRAP_REPLICATES,
    MIN_TARGET_SUPPORT_CASES,
    SEED_PAIR_COUNT,
    TARGET_CANDIDATE_COUNT,
    TARGET_ROLE,
    CaseBootstrapPlan,
)
from .features import permute_interaction_features
from .surface_contracts import FeatureSurface


def build_utility_aligned_policy(
    models: UtilityAlignedModels,
    target_features: FeatureSurface,
    transfer_eligibility: CardinalityTransferResult,
    *,
    global_only: bool = False,
    confidence_multiplier: float = 1.96,
    minimum_gain: float = 0.0,
    minimum_support_case_count: int = MIN_TARGET_SUPPORT_CASES,
    support_bootstrap_features: Sequence[FeatureSurface] = (),
    case_bootstrap_plan: CaseBootstrapPlan | None = None,
    minimum_support_bootstrap_replicates: int = MIN_SUPPORT_BOOTSTRAP_REPLICATES,
) -> UtilityAlignedPolicy:
    """Choose one exact additive tail or abstain bit-exactly to B.

    The target decision averages all nine frozen seed pairs for every source.
    Model covariance is combined conservatively with the *full* across-seed
    prediction standard deviation, rather than treating seed cells as
    independent samples.  Source-inner 6->7 success is only eligibility to
    attempt a sealed fresh 7->8 evaluation; it is not 7->8 routing evidence.
    Failed eligibility, fewer than eight independent target-support cases, or
    a nonpositive lower confidence bound yields exact B and no selected source.
    """

    if not isinstance(models, UtilityAlignedModels):
        raise ProtocolError("Utility-aligned policy requires fitted models.")
    if (
        not isinstance(target_features, FeatureSurface)
        or target_features.role != TARGET_ROLE
        or target_features.permutation_seed is not None
        or target_features.case_bootstrap_replicate is not None
    ):
        raise ProtocolError("Policy construction requires an unpermuted target surface.")
    if not isinstance(transfer_eligibility, CardinalityTransferResult):
        raise ProtocolError("Policy construction requires a cardinality eligibility result.")
    if (
        models.outer_target_id != target_features.outer_target_id
        or models.candidate_sources != target_features.candidate_sources
        or transfer_eligibility.outer_target_id != models.outer_target_id
        or transfer_eligibility.candidate_sources != models.candidate_sources
    ):
        raise ProtocolError("Policy target/candidate geometry drifted.")
    if len(models.candidate_sources) != TARGET_CANDIDATE_COUNT:
        raise ProtocolError("Target policy requires exactly eight candidate sources.")
    if models.permutation_seed is None:
        if transfer_eligibility.model_hash != models.model_hash:
            raise ProtocolError("Transfer eligibility does not belong to this model.")
    elif global_only:
        raise ProtocolError("Global-only policy must use the unpermuted model bundle.")
    multiplier = _nonnegative_finite(confidence_multiplier, "confidence_multiplier")
    threshold = _nonnegative_finite(minimum_gain, "minimum_gain")
    if (
        isinstance(minimum_support_case_count, bool)
        or not isinstance(minimum_support_case_count, Integral)
        or int(minimum_support_case_count) < MIN_TARGET_SUPPORT_CASES
    ):
        raise ProtocolError("Target support-case requirement cannot be relaxed below eight.")
    minimum_cases = int(minimum_support_case_count)
    if (
        isinstance(minimum_support_bootstrap_replicates, bool)
        or not isinstance(minimum_support_bootstrap_replicates, Integral)
        or int(minimum_support_bootstrap_replicates)
        < MIN_SUPPORT_BOOTSTRAP_REPLICATES
    ):
        raise ProtocolError(
            "Support bootstrap replicate requirement cannot be relaxed below 32."
        )
    minimum_bootstraps = int(minimum_support_bootstrap_replicates)
    support_counts = {row.support_case_count for row in target_features.rows}
    if len(support_counts) != 1:
        raise ProtocolError("Target support case count drifted across feature rows.")
    support_case_count = next(iter(support_counts))

    if global_only:
        routed_surface = target_features
        matrix = routed_surface.global_values
        model = models.global_model
        router_kind = "global_source_quality_only"
        proposed_action = GLOBAL_ACTION_ID
        permutation_seed = None
    elif models.permutation_seed is not None:
        routed_surface = permute_interaction_features(
            target_features, permutation_seed=models.permutation_seed
        )
        matrix = routed_surface.interaction_values
        model = models.interaction_model
        router_kind = "cyclic_feature_permutation_control"
        proposed_action = PERMUTATION_ACTION_ID
        permutation_seed = models.permutation_seed
    else:
        routed_surface = target_features
        matrix = routed_surface.interaction_values
        model = models.interaction_model
        router_kind = "target_source_interaction"
        proposed_action = ROUTED_ACTION_ID
        permutation_seed = None
    if tuple(model.feature_names) != (
        routed_surface.global_feature_names
        if global_only
        else routed_surface.interaction_feature_names
    ):
        raise ProtocolError("Target feature columns do not match the fitted model.")
    prediction = model.predict_with_uncertainty(
        matrix, include_residual_variance=False
    )
    by_source: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(routed_surface.rows):
        by_source[row.candidate_source].append(index)
    aggregate: dict[str, tuple[float, float, float]] = {}
    for source in routed_surface.candidate_sources:
        indices = tuple(by_source[source])
        if len(indices) != SEED_PAIR_COUNT:
            raise ProtocolError("Target policy requires all nine seed pairs per source.")
        means = np.asarray(prediction.mean[list(indices)], dtype=np.float64)
        mean_gain = float(np.mean(means, dtype=np.float64))
        averaging = np.full(SEED_PAIR_COUNT, 1.0 / SEED_PAIR_COUNT, dtype=np.float64)
        covariance = prediction.covariance[np.ix_(indices, indices)]
        # Coefficient uncertainty may be averaged with its full covariance,
        # but out-of-query residual uncertainty is a domain-level component.
        # It is added exactly once and must never be divided by nine technical
        # seed cells as though those cells were independent domains.
        model_variance = max(0.0, float(averaging @ covariance @ averaging))
        model_variance += float(model.residual_variance)
        # Seeds are repeated model/generation realizations, not independent
        # domains.  Retain their full spread instead of dividing by sqrt(9).
        replicate_std = float(np.std(means, ddof=1))
        combined_standard_error = float(
            np.sqrt(model_variance + replicate_std * replicate_std)
        )
        aggregate[source] = (mean_gain, combined_standard_error, replicate_std)
    proposed_source = min(
        routed_surface.candidate_sources,
        key=lambda source: (-aggregate[source][0], source),
    )
    predicted_gain, standard_error, replicate_std = aggregate[proposed_source]
    bootstrap_means: list[float] = []
    if any(
        not isinstance(surface, FeatureSurface)
        for surface in support_bootstrap_features
    ):
        raise ProtocolError("Support-bootstrap inputs must be feature surfaces.")
    bootstrap_hashes = tuple(sorted(surface.surface_hash for surface in support_bootstrap_features))
    bootstrap_replicate_hashes = tuple(
        sorted(
            surface.case_bootstrap_replicate.replicate_hash
            if surface.case_bootstrap_replicate is not None
            else ""
            for surface in support_bootstrap_features
        )
    )
    if len(set(bootstrap_hashes)) != len(bootstrap_hashes):
        raise ProtocolError("Support-bootstrap feature surfaces must have unique hashes.")
    if (
        any(not value for value in bootstrap_replicate_hashes)
        or len(set(bootstrap_replicate_hashes)) != len(bootstrap_replicate_hashes)
    ):
        raise ProtocolError(
            "Support-bootstrap surfaces require unique typed case replicates."
        )
    if target_features.surface_hash in set(bootstrap_hashes):
        raise ProtocolError("The point-estimate surface cannot masquerade as a bootstrap.")
    if not global_only and support_bootstrap_features and case_bootstrap_plan is None:
        raise ProtocolError("Hash-only bootstrap surfaces are forbidden; provide a typed plan.")
    if not global_only and case_bootstrap_plan is not None:
        if (
            not isinstance(case_bootstrap_plan, CaseBootstrapPlan)
            or case_bootstrap_plan.target_id != models.outer_target_id
            or len(case_bootstrap_plan.support_case_ids) != support_case_count
        ):
            raise ProtocolError("Case-bootstrap plan target/support geometry drifted.")
        if any(
            row.support_partition_hash != case_bootstrap_plan.support_partition_hash
            for row in target_features.rows
        ):
            raise ProtocolError("Point target surface does not match the bootstrap parent cases.")
        expected_replicate_hashes = tuple(
            sorted(item.replicate_hash for item in case_bootstrap_plan.replicates)
        )
        if bootstrap_replicate_hashes != expected_replicate_hashes:
            raise ProtocolError("Bootstrap surfaces do not exactly cover the typed plan.")
    if not global_only:
        for bootstrap_surface in sorted(
            support_bootstrap_features,
            key=lambda surface: (
                surface.case_bootstrap_replicate.replicate_index
                if surface.case_bootstrap_replicate is not None
                else -1
            ),
        ):
            if (
                not isinstance(bootstrap_surface, FeatureSurface)
                or bootstrap_surface.role != TARGET_ROLE
                or bootstrap_surface.permutation_seed is not None
                or bootstrap_surface.outer_target_id != target_features.outer_target_id
                or bootstrap_surface.candidate_sources != target_features.candidate_sources
                or bootstrap_surface.row_keys != target_features.row_keys
                or bootstrap_surface.case_bootstrap_replicate is None
            ):
                raise ProtocolError("Support-bootstrap feature surface geometry drifted.")
            routed_bootstrap = (
                permute_interaction_features(
                    bootstrap_surface,
                    permutation_seed=models.permutation_seed,
                )
                if models.permutation_seed is not None
                else bootstrap_surface
            )
            bootstrap_prediction = model.predict(routed_bootstrap.interaction_values)
            source_indices = tuple(
                index
                for index, row in enumerate(routed_bootstrap.rows)
                if row.candidate_source == proposed_source
            )
            if len(source_indices) != SEED_PAIR_COUNT:
                raise ProtocolError("Support-bootstrap seed geometry drifted.")
            bootstrap_means.append(
                float(np.mean(bootstrap_prediction[list(source_indices)], dtype=np.float64))
            )
    support_bootstrap_std = (
        float(np.std(np.asarray(bootstrap_means, dtype=np.float64), ddof=1))
        if len(bootstrap_means) >= 2
        else 0.0
    )
    standard_error = float(
        np.sqrt(standard_error * standard_error + support_bootstrap_std**2)
    )
    lower_bound = predicted_gain - multiplier * standard_error
    fallback_reason: str | None = None
    if not global_only and support_case_count < minimum_cases:
        fallback_reason = "insufficient_independent_support_cases_exact_base"
    elif not global_only and len(bootstrap_means) < minimum_bootstraps:
        fallback_reason = "support_case_bootstrap_uncertainty_missing_exact_base"
    elif global_only and not transfer_eligibility.global_gate_passed:
        fallback_reason = "global_source_quality_gate_failed_exact_base"
    elif not global_only and not transfer_eligibility.eligibility_passed:
        fallback_reason = "cardinality_transfer_eligibility_failed_exact_base"
    elif lower_bound <= threshold:
        fallback_reason = "nonpositive_or_uncertain_additive_gain_exact_base"
    used_fallback = fallback_reason is not None
    selected_source = None if used_fallback else proposed_source
    action_id = BASE_ACTION_ID if used_fallback else proposed_action
    payload = {
        "schema_version": "midogpp_utility_aligned_policy_v1",
        "target_id": models.outer_target_id,
        "candidate_sources": list(models.candidate_sources),
        "router_kind": router_kind,
        "proposed_action_id": proposed_action,
        "action_id": action_id,
        "proposed_source": proposed_source,
        "selected_source": selected_source,
        "predicted_gain": predicted_gain,
        "standard_error": standard_error,
        "lower_confidence_bound": lower_bound,
        "confidence_multiplier": multiplier,
        "minimum_gain": threshold,
        "support_case_count": support_case_count,
        "minimum_support_case_count": minimum_cases,
        "seed_pair_count": SEED_PAIR_COUNT,
        "replicate_standard_deviation": replicate_std,
        "support_bootstrap_replicates": len(bootstrap_means),
        "support_bootstrap_standard_deviation": support_bootstrap_std,
        "support_bootstrap_surface_hashes": list(bootstrap_hashes),
        "case_bootstrap_replicate_hashes": list(bootstrap_replicate_hashes),
        "minimum_support_bootstrap_replicates": minimum_bootstraps,
        "used_exact_base_fallback": used_fallback,
        "fallback_reason": fallback_reason,
        "global_only": global_only,
        "permutation_seed": permutation_seed,
        "model_hash": models.model_hash,
        "feature_surface_hash": routed_surface.surface_hash,
        "cardinality_eligibility_hash": transfer_eligibility.result_hash,
        "case_bootstrap_plan_hash": (
            case_bootstrap_plan.plan_hash if case_bootstrap_plan is not None else None
        ),
        "target_support_labels_used": False,
        "target_evaluation_used": False,
        "seed_selection_performed": False,
        "abstention_semantics": ABSTENTION_SEMANTICS,
    }
    return UtilityAlignedPolicy(
        target_id=models.outer_target_id,
        candidate_sources=models.candidate_sources,
        router_kind=router_kind,
        proposed_action_id=proposed_action,
        action_id=action_id,
        proposed_source=proposed_source,
        selected_source=selected_source,
        predicted_gain=predicted_gain,
        standard_error=standard_error,
        lower_confidence_bound=lower_bound,
        confidence_multiplier=multiplier,
        minimum_gain=threshold,
        support_case_count=support_case_count,
        minimum_support_case_count=minimum_cases,
        seed_pair_count=SEED_PAIR_COUNT,
        replicate_standard_deviation=replicate_std,
        support_bootstrap_replicates=len(bootstrap_means),
        minimum_support_bootstrap_replicates=minimum_bootstraps,
        support_bootstrap_standard_deviation=support_bootstrap_std,
        support_bootstrap_surface_hashes=bootstrap_hashes,
        case_bootstrap_replicate_hashes=bootstrap_replicate_hashes,
        used_exact_base_fallback=used_fallback,
        fallback_reason=fallback_reason,
        global_only=global_only,
        permutation_seed=permutation_seed,
        model_hash=models.model_hash,
        feature_surface_hash=routed_surface.surface_hash,
        cardinality_eligibility_hash=transfer_eligibility.result_hash,
        case_bootstrap_plan_hash=(
            case_bootstrap_plan.plan_hash if case_bootstrap_plan is not None else None
        ),
        target_support_labels_used=False,
        target_evaluation_used=False,
        seed_selection_performed=False,
        abstention_semantics=ABSTENTION_SEMANTICS,
        policy_hash=canonical_sha256(payload),
    )


def _nonnegative_finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ProtocolError(f"{name} must be finite and nonnegative.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError(f"{name} must be finite and nonnegative.") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ProtocolError(f"{name} must be finite and nonnegative.")
    return number


__all__ = ("build_utility_aligned_policy",)
