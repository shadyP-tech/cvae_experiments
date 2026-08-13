"""Frozen science protocol manifest for runtime and independent replay."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...protocol import ProtocolError
from .constants import (
    CANDIDATE_FEATURE_PERMUTATION_ALGORITHM,
    CANDIDATE_FEATURE_PERMUTATION_SEED,
    CASE_PROXY_WEIGHT_DENOMINATOR,
    CASE_PROXY_WEIGHT_NUMERATOR,
    DESCRIPTIVE_METHOD_IDS,
    FEATURE_NAMES,
    IRLS_CONVERGENCE_TOLERANCE,
    IRLS_ETA_CLIP,
    IRLS_MAX_ITERATIONS,
    IRLS_PROBABILITY_CLIP,
    PRE_TERMINAL_METHOD_IDS,
    PRIOR_WEIGHT_DENOMINATOR,
    PRIOR_WEIGHT_NUMERATOR,
    PUBLICATION_STATUS,
    RIDGE_ALPHA,
    TERMINAL_DECISION,
    TIE_TOLERANCE,
)
from .hashing import canonical_hash, require_sha256


def frozen_science_protocol_payload() -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_cdca_science_protocol_v1",
        "feature_names": list(FEATURE_NAMES),
        "feature_formulas": {
            "directional_flip_rate": "m_over_case_size",
            "baseline_abs_margin_on_directional_flips": "mean_abs_B_mean_minus_0.5_on_direction_flips",
            "candidate_abs_margin_on_directional_flips": "mean_abs_A1_mean_minus_0.5_on_direction_flips",
            "directional_probability_shift_on_flips": "mean_A1_probability_minus_B_probability_on_directional_flips",
            "seed_directional_flip_robustness": "mean_fraction_of_nine_seed_pairs_reproducing_mean_directional_flip",
            "candidate_seed_disagreement_on_directional_flips": "mean_two_r_times_one_minus_r_where_r_is_A1_positive_seed_fraction",
        },
        "feature_labels_used": False,
        "response": "directional_flip_correctness_binomial_successes_over_trials",
        "fit_unit": "one_ephemeral_model_per_H_c_e_direction",
        "fit_scope": "same_H_whole_cases_except_c_only",
        "standardization": "H_minus_c_unweighted_mean_population_sd",
        "model_family": "pure_numpy_ridge_binomial_logistic_newton_irls_v1",
        "ridge_alpha": RIDGE_ALPHA,
        "intercept_penalized": False,
        "max_iterations": IRLS_MAX_ITERATIONS,
        "convergence_tolerance": IRLS_CONVERGENCE_TOLERANCE,
        "eta_clip": [-IRLS_ETA_CLIP, IRLS_ETA_CLIP],
        "probability_clip": [
            IRLS_PROBABILITY_CLIP,
            1.0 - IRLS_PROBABILITY_CLIP,
        ],
        "initialization": "all_zero_coefficients",
        "invalid_model_case_proxy": 0.0,
        "support_denominators": "H_minus_c_labels_only",
        "zero_to_one_case_proxy": "m*pi_over_2Npos_minus_m*one_minus_pi_over_2Nneg",
        "one_to_zero_case_proxy": "m*pi_over_2Nneg_minus_m*one_minus_pi_over_2Npos",
        "case_proxy_weight": [
            CASE_PROXY_WEIGHT_NUMERATOR,
            CASE_PROXY_WEIGHT_DENOMINATOR,
        ],
        "donor_prior_weight": [
            PRIOR_WEIGHT_NUMERATOR,
            PRIOR_WEIGHT_DENOMINATOR,
        ],
        "donor_query_scope": "q_not_in_H_or_e",
        "donor_grant_count": 72,
        "all_donor_grants_complete_before_route_support": True,
        "candidate_pool": "all_eight_non_target_sources_plus_OFF",
        "off_score": 0.0,
        "tie_tolerance": TIE_TOLERANCE,
        "tie_order": "OFF_then_numeric_source",
        "composition": "B_probability_replaced_by_selected_A1_on_B_hard_direction",
        "pre_terminal_method_ids": list(PRE_TERMINAL_METHOD_IDS),
        "descriptive_method_ids": list(DESCRIPTIVE_METHOD_IDS),
        "feature_permutation": {
            "seed": CANDIDATE_FEATURE_PERMUTATION_SEED,
            "algorithm": CANDIDATE_FEATURE_PERMUTATION_ALGORITHM,
            "route_seed_derivation": "configured_seed_xor_first_64_bits_of_sha256_canonical_route_identity",
            "shuffle": "descending_fisher_yates_with_splitmix64_draw_modulo",
            "unit": "complete_candidate_feature_vector_within_H_c_direction",
            "outcomes_denominators_G_and_candidate_identities_fixed": True,
            "refit_and_reselect_same_pipeline": True,
            "descriptive_only": True,
        },
        "terminal_barrier": "all_218_route_seals_plus_readback_aggregate",
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
    }


@dataclass(frozen=True)
class FrozenScienceProtocol:
    payload: dict[str, object] = field(
        default_factory=frozen_science_protocol_payload
    )
    protocol_hash: str = ""

    def __post_init__(self) -> None:
        canonical = frozen_science_protocol_payload()
        if self.payload != canonical:
            raise ProtocolError("Abstention-router frozen science protocol drifted.")
        expected = canonical_hash(canonical)
        if self.protocol_hash and require_sha256(
            self.protocol_hash, "protocol_hash"
        ) != expected:
            raise ProtocolError("Abstention-router science protocol hash drifted.")
        object.__setattr__(self, "payload", canonical)
        object.__setattr__(self, "protocol_hash", expected)

    def to_payload(self) -> dict[str, object]:
        return {**self.payload, "protocol_hash": self.protocol_hash}


def build_frozen_science_protocol() -> FrozenScienceProtocol:
    return FrozenScienceProtocol()


__all__ = (
    "FrozenScienceProtocol",
    "build_frozen_science_protocol",
    "frozen_science_protocol_payload",
)
