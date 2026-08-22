"""Utility input manifest for the analytic, feature-free response rule."""

from __future__ import annotations

from .hashing import canonical_hash


def analytic_utility_input_manifest() -> dict[str, object]:
    """Describe the complete inputs; no transferred feature model is fitted."""

    payload: dict[str, object] = {
        "schema_version": "fixed_bank_cbpupr_analytic_utility_inputs_v1",
        "inputs": [
            "byte_exact_P_probabilities",
            "complete_candidate_probabilities",
            "held_case_excluded_posterior_eta",
            "posterior_augmented_center_denominators",
        ],
        "learned_utility_features": [],
        "donor_response_regression_used": False,
    }
    return {**payload, "manifest_hash": canonical_hash(payload)}


__all__ = ("analytic_utility_input_manifest",)
