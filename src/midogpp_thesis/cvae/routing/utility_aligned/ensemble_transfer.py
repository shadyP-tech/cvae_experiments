"""Source-inner query bootstrap and cardinality-transfer authorization gate."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ..residual_topup.hashing import array_sha256, canonical_sha256
from .ensemble_endpoint import validate_ensemble_utility_responses
from .ensemble_utility_contracts import (
    EnsembleUtilityResponse,
    EnsembleUtilitySurface,
    ScoredEnsembleUtilityResponse,
)
from .ensemble_model_contracts import (
    EnsembleCardinalityTransferResult,
    EnsembleUtilityModel,
)
from .ensemble_policy_contracts import (
    ENSEMBLE_GLOBAL_ROLE,
    ENSEMBLE_PERMUTATION_ROLE,
    ENSEMBLE_QUERY_BOOTSTRAP_DRAWS,
    ENSEMBLE_QUERY_BOOTSTRAP_SEED,
    ENSEMBLE_ROUTED_ROLE,
)
from .ensemble_policy_metrics import quantile, routing_metric_vectors
from .result_contracts import (
    MAX_AUTHORIZED_NORMALIZED_ORACLE_GAP,
    SOURCE_INNER_TOP1_CHANCE,
)
from .row_contracts import (
    INNER_CANDIDATE_COUNT,
    TARGET_CANDIDATE_COUNT,
)


def evaluate_ensemble_cardinality_transfer(
    global_model: EnsembleUtilityModel,
    routed_model: EnsembleUtilityModel,
    permutation_model: EnsembleUtilityModel,
    utility: EnsembleUtilitySurface
    | Sequence[
        EnsembleUtilityResponse | ScoredEnsembleUtilityResponse | Mapping[str, object]
    ],
) -> EnsembleCardinalityTransferResult:
    """Bind G/R/P strict crossfits to source-inner routing/regret gates."""

    models = {
        ENSEMBLE_GLOBAL_ROLE: global_model,
        ENSEMBLE_ROUTED_ROLE: routed_model,
        ENSEMBLE_PERMUTATION_ROLE: permutation_model,
    }
    if any(not isinstance(model, EnsembleUtilityModel) for model in models.values()):
        raise ProtocolError("Ensemble cardinality transfer requires three typed models.")
    targets = {model.outer_target_id for model in models.values()}
    row_key_sets = {model.crossfit_row_keys for model in models.values()}
    if len(targets) != 1 or len(row_key_sets) != 1:
        raise ProtocolError("G/R/P ensemble model bindings drifted.")
    if (
        len(global_model.feature_names) != 1
        or len(routed_model.feature_names) != 2
        or len(permutation_model.feature_names) != 2
        or global_model.permutation_seed is not None
        or routed_model.permutation_seed is not None
        or permutation_model.permutation_seed is None
    ):
        raise ProtocolError("Ensemble transfer requires G=M0, R=M1, and P=permuted M1.")
    outer = next(iter(targets))
    utility_surface = (
        utility
        if isinstance(utility, EnsembleUtilitySurface)
        else validate_ensemble_utility_responses(utility)
    )
    utility_rows = utility_surface.rows_for_outer_target(outer)
    utility_by_key = {row.row_key: row.utility_delta for row in utility_rows}
    keys = global_model.crossfit_row_keys
    if set(keys) != set(utility_by_key):
        raise ProtocolError("Ensemble transfer utility/crossfit keys do not align.")
    truth = np.asarray([utility_by_key[key] for key in keys], dtype=np.float64)
    metric_vectors = {
        role: routing_metric_vectors(keys, model.crossfit_predictions, truth)
        for role, model in models.items()
    }
    metrics = {
        role: {
            "query_count": float(len(next(iter(vectors.values())))),
            "top1_oracle_agreement": float(
                np.mean(vectors["top1"], dtype=np.float64)
            ),
            "mean_spearman": float(
                np.mean(vectors["spearman"], dtype=np.float64)
            ),
            "mean_normalized_oracle_gap": float(
                np.mean(vectors["normalized_gap"], dtype=np.float64)
            ),
        }
        for role, vectors in metric_vectors.items()
    }
    query_count = len(metric_vectors[ENSEMBLE_ROUTED_ROLE]["top1"])
    if query_count != TARGET_CANDIDATE_COUNT:
        raise ProtocolError("Ensemble transfer requires eight independent query domains.")
    rng = np.random.Generator(np.random.PCG64(ENSEMBLE_QUERY_BOOTSTRAP_SEED))
    bootstrap_indices = rng.integers(
        0,
        query_count,
        size=(ENSEMBLE_QUERY_BOOTSTRAP_DRAWS, query_count),
        dtype=np.int64,
    )
    bootstrap_indices_hash = array_sha256(bootstrap_indices)
    bootstrap_draws: dict[str, dict[str, np.ndarray]] = {}
    bootstrap_bounds: dict[str, dict[str, float]] = {}
    for role, vectors in metric_vectors.items():
        draws = {
            name: np.mean(values[bootstrap_indices], axis=1, dtype=np.float64)
            for name, values in vectors.items()
        }
        bootstrap_draws[role] = draws
        bootstrap_bounds[role] = {
            "top1_lower_bound": quantile(draws["top1"], 0.025),
            "top1_upper_bound": quantile(draws["top1"], 0.975),
            "spearman_lower_bound": quantile(draws["spearman"], 0.025),
            "spearman_upper_bound": quantile(draws["spearman"], 0.975),
            "normalized_gap_lower_bound": quantile(
                draws["normalized_gap"], 0.025
            ),
            "normalized_gap_upper_bound": quantile(
                draws["normalized_gap"], 0.975
            ),
        }
    paired_bounds: dict[str, dict[str, float]] = {}
    routed_draws = bootstrap_draws[ENSEMBLE_ROUTED_ROLE]
    for comparator_role in (ENSEMBLE_GLOBAL_ROLE, ENSEMBLE_PERMUTATION_ROLE):
        comparator_draws = bootstrap_draws[comparator_role]
        paired_bounds[comparator_role] = {
            "top1_improvement_lower_bound": quantile(
                routed_draws["top1"] - comparator_draws["top1"], 0.025
            ),
            "spearman_improvement_lower_bound": quantile(
                routed_draws["spearman"] - comparator_draws["spearman"], 0.025
            ),
            "normalized_gap_reduction_lower_bound": quantile(
                comparator_draws["normalized_gap"]
                - routed_draws["normalized_gap"],
                0.025,
            ),
        }
    capacity_hashes = {
        role: tuple(
            model.candidate_capacity_reports[source].report_hash
            for source in sorted(model.candidate_capacity_reports)
        )
        for role, model in models.items()
    }
    all_capacity = all(
        report.gate_passed
        for model in models.values()
        for report in model.candidate_capacity_reports.values()
    )
    routed = metrics[ENSEMBLE_ROUTED_ROLE]
    global_metrics = metrics[ENSEMBLE_GLOBAL_ROLE]
    permutation = metrics[ENSEMBLE_PERMUTATION_ROLE]
    routed_bounds = bootstrap_bounds[ENSEMBLE_ROUTED_ROLE]
    failures: list[str] = []
    if not all_capacity:
        failures.append("capacity_gate_failed")
    if routed["top1_oracle_agreement"] <= SOURCE_INNER_TOP1_CHANCE:
        failures.append("routed_top1_not_above_chance")
    if routed["mean_normalized_oracle_gap"] > MAX_AUTHORIZED_NORMALIZED_ORACLE_GAP:
        failures.append("routed_normalized_oracle_gap_above_gate")
    if routed["top1_oracle_agreement"] < global_metrics["top1_oracle_agreement"]:
        failures.append("routed_top1_below_global")
    if routed["top1_oracle_agreement"] < permutation["top1_oracle_agreement"]:
        failures.append("routed_top1_below_permutation")
    if routed["mean_spearman"] < global_metrics["mean_spearman"]:
        failures.append("routed_spearman_below_global")
    if routed["mean_spearman"] < permutation["mean_spearman"]:
        failures.append("routed_spearman_below_permutation")
    if routed["mean_normalized_oracle_gap"] > global_metrics["mean_normalized_oracle_gap"]:
        failures.append("routed_regret_above_global")
    if routed["mean_normalized_oracle_gap"] > permutation["mean_normalized_oracle_gap"]:
        failures.append("routed_regret_above_permutation")
    if routed_bounds["top1_lower_bound"] <= SOURCE_INNER_TOP1_CHANCE:
        failures.append("routed_top1_lower_bound_not_above_chance")
    if routed_bounds["spearman_lower_bound"] <= 0.0:
        failures.append("routed_spearman_lower_bound_not_positive")
    if (
        routed_bounds["normalized_gap_upper_bound"]
        >= MAX_AUTHORIZED_NORMALIZED_ORACLE_GAP
    ):
        failures.append("routed_normalized_gap_upper_bound_not_below_gate")
    for comparator_role in (ENSEMBLE_GLOBAL_ROLE, ENSEMBLE_PERMUTATION_ROLE):
        comparator_bounds = paired_bounds[comparator_role]
        if comparator_bounds["top1_improvement_lower_bound"] <= 0.0:
            failures.append(f"routed_top1_paired_bound_not_above_{comparator_role}")
        if comparator_bounds["spearman_improvement_lower_bound"] <= 0.0:
            failures.append(
                f"routed_spearman_paired_bound_not_above_{comparator_role}"
            )
        if comparator_bounds["normalized_gap_reduction_lower_bound"] <= 0.0:
            failures.append(
                f"routed_gap_reduction_paired_bound_not_above_{comparator_role}"
            )
    unhashed = {
        "schema_version": "midogpp_utility_aligned_ensemble_cardinality_transfer_v1",
        "outer_target_id": outer,
        "independent_query_count": len({key[1] for key in keys}),
        "source_inner_candidate_count": INNER_CANDIDATE_COUNT,
        "deployment_candidate_count": TARGET_CANDIDATE_COUNT,
        "metrics_by_role": metrics,
        "bootstrap_bounds_by_role": bootstrap_bounds,
        "paired_improvement_bounds": paired_bounds,
        "capacity_report_hashes_by_role": {
            role: list(values) for role, values in capacity_hashes.items()
        },
        "all_capacity_gates_passed": all_capacity,
        "query_bootstrap_seed": ENSEMBLE_QUERY_BOOTSTRAP_SEED,
        "query_bootstrap_draw_count": ENSEMBLE_QUERY_BOOTSTRAP_DRAWS,
        "query_bootstrap_indices_hash": bootstrap_indices_hash,
        "authorized_for_target_policy": not failures,
        "authorization_failures": failures,
        "claim_role": "source_inner_ensemble_eligibility_only_not_target_utility",
    }
    return EnsembleCardinalityTransferResult(
        outer_target_id=outer,
        independent_query_count=len({key[1] for key in keys}),
        source_inner_candidate_count=INNER_CANDIDATE_COUNT,
        deployment_candidate_count=TARGET_CANDIDATE_COUNT,
        metrics_by_role=metrics,
        bootstrap_bounds_by_role=bootstrap_bounds,
        paired_improvement_bounds=paired_bounds,
        capacity_report_hashes_by_role=capacity_hashes,
        query_bootstrap_seed=ENSEMBLE_QUERY_BOOTSTRAP_SEED,
        query_bootstrap_draw_count=ENSEMBLE_QUERY_BOOTSTRAP_DRAWS,
        query_bootstrap_indices_hash=bootstrap_indices_hash,
        all_capacity_gates_passed=all_capacity,
        authorized_for_target_policy=not failures,
        authorization_failures=tuple(failures),
        transfer_hash=canonical_sha256(unhashed),
    )



__all__ = ("evaluate_ensemble_cardinality_transfer",)
