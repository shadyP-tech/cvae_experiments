"""Focused Stage-70 exact-nine aggregation contract tests."""

from __future__ import annotations

import numpy as np
import pytest

from midogpp_thesis.cvae.frozen_policy_downstream.utility_aligned_residual_fresh.ensemble_adapter import (
    mean_exact_nine_probabilities,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.frozen_policy_downstream.utility_aligned_residual_fresh.contracts import (
    CENTERS,
    legal_sources,
)
from midogpp_thesis.cvae.frozen_policy_downstream.utility_aligned_residual_fresh.policy_target_validation import (
    _validate_ensemble_policy,
    _validate_policy_grid,
)


def test_stage70_exact_nine_adapter_matches_direct_probability_mean() -> None:
    vectors = tuple(
        np.asarray([0.05 * index, 1.0 - 0.05 * index], dtype=np.float32)
        for index in range(9)
    )

    observed = mean_exact_nine_probabilities(vectors)

    expected = np.mean(np.stack(vectors, axis=0), axis=0, dtype=np.float64)
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)
    assert observed.dtype == np.float64
    assert observed.flags.writeable is False


@pytest.mark.parametrize(
    "vectors",
    [
        tuple(np.asarray([0.5], dtype=np.float64) for _ in range(8)),
        tuple(np.asarray([0.5, 0.6], dtype=np.float64) for _ in range(8))
        + (np.asarray([0.5], dtype=np.float64),),
        tuple(np.asarray([1.1], dtype=np.float64) for _ in range(9)),
    ],
)
def test_stage70_exact_nine_adapter_rejects_tampered_probability_geometry(
    vectors: tuple[np.ndarray, ...],
) -> None:
    with pytest.raises(ProtocolError, match="exact-nine probability geometry"):
        mean_exact_nine_probabilities(vectors)


def _ensemble_target_payload(*, target: str = "0", support_case_count: int = 10):
    candidates = list(legal_sources(target))
    surface_hash = "a" * 64
    bootstrap_hashes = [f"{index:064x}" for index in range(1, 33)]
    predictions = {
        role: {source: 0.1 + ordinal * 0.01 for ordinal, source in enumerate(candidates)}
        for role in ("G", "R", "P")
    }
    errors = {
        role: {source: 0.01 for source in candidates} for role in ("G", "R", "P")
    }
    model_errors = {
        role: {source: 0.006 for source in candidates}
        for role in ("G", "R", "P")
    }
    bootstrap_errors = {
        role: {source: 0.008 for source in candidates}
        for role in ("G", "R", "P")
    }
    seed_diagnostics = {
        role: {source: 0.02 for source in candidates}
        for role in ("G", "R", "P")
    }
    bounds = {
        role: {
            source: predictions[role][source] - 1.96 * errors[role][source]
            for source in candidates
        }
        for role in ("G", "R", "P")
    }
    selected = candidates[-1]
    policy_hash = "b" * 64
    transfer_hash = "c" * 64
    nested = {
        "schema_version": "midogpp_utility_aligned_ensemble_policy_v1",
        "target_id": target,
        "role_prediction_by_source": predictions,
        "role_model_standard_error_by_source": model_errors,
        "role_bootstrap_standard_deviation_by_source": bootstrap_errors,
        "role_target_scalar_seed_standard_deviation_by_source": seed_diagnostics,
        "role_combined_standard_error_by_source": errors,
        "role_lower_confidence_bound_by_source": bounds,
        "role_selected_action": {"G": "G", "R": "R", "P": "P"},
        "role_selected_source": {"G": selected, "R": selected, "P": selected},
        "point_feature_surface_hash": surface_hash,
        "bootstrap_feature_surface_hashes": bootstrap_hashes,
        "cardinality_transfer_hash": transfer_hash,
        "authorization_uncertainty_components": [
            "model_covariance_and_residual",
            "independent_whole_case_bootstrap",
        ],
        "target_scalar_seed_spread_role": "descriptive_only_non_decision",
        "target_scalar_seed_spread_enters_combined_standard_error": False,
        "policy_hash": policy_hash,
    }
    payload = {
        "schema_version": "midogpp_utility_aligned_ensemble_policy_v1",
        "target_id": target,
        "role": "R",
        "candidate_sources": candidates,
        "proposed_action_id": "R",
        "action_id": "R",
        "proposed_source": selected,
        "selected_source": selected,
        "predicted_gain": predictions["R"][selected],
        "standard_error": errors["R"][selected],
        "lower_confidence_bound": bounds["R"][selected],
        "support_case_count": support_case_count,
        "support_bootstrap_replicates": 32,
        "used_exact_base_fallback": False,
        "fallback_reason": None,
        "model_hash": "d" * 64,
        "feature_surface_hash": surface_hash,
        "cardinality_eligibility_hash": transfer_hash,
        "policy_hash": policy_hash,
        "ensemble_policy": nested,
    }
    feature_lock = {
        "target_feature_surface_hash": surface_hash,
        "bootstrap_surface_hashes": bootstrap_hashes,
        "support_case_count": support_case_count,
    }
    frozen = {
        "selected_source": selected,
        "abstained_to_base": False,
        "fallback_reason": None,
    }
    return payload, feature_lock, frozen


def test_stage70_ensemble_policy_accepts_more_than_minimum_support_cases() -> None:
    payload, feature_lock, frozen = _ensemble_target_payload(support_case_count=10)
    _validate_ensemble_policy(
        payload,
        target="0",
        proposed="R",
        feature_lock=feature_lock,
        frozen_action=frozen,
    )


def test_stage70_ensemble_policy_rejects_cross_role_action_swap() -> None:
    payload, feature_lock, frozen = _ensemble_target_payload()
    payload["proposed_action_id"] = "G_delta"
    with pytest.raises(ProtocolError, match="ensemble target policy drifted"):
        _validate_ensemble_policy(
            payload,
            target="0",
            proposed="R",
            feature_lock=feature_lock,
            frozen_action=frozen,
        )


def test_stage70_ensemble_policy_rejects_seed_spread_in_authorization_se() -> None:
    payload, feature_lock, frozen = _ensemble_target_payload()
    nested = payload["ensemble_policy"]
    selected = payload["proposed_source"]
    nested["role_combined_standard_error_by_source"]["R"][selected] = 0.03
    payload["standard_error"] = 0.03
    payload["lower_confidence_bound"] = payload["predicted_gain"] - 1.96 * 0.03
    nested["role_lower_confidence_bound_by_source"]["R"][selected] = payload[
        "lower_confidence_bound"
    ]
    with pytest.raises(ProtocolError, match="ensemble target policy drifted"):
        _validate_ensemble_policy(
            payload,
            target="0",
            proposed="R",
            feature_lock=feature_lock,
            frozen_action=frozen,
        )


def test_stage70_ensemble_family_rejects_legacy_policy_schema() -> None:
    legacy = [{"schema_version": "midogpp_utility_aligned_policy_v1"}] * (
        len(CENTERS) * 3
    )
    with pytest.raises(ProtocolError, match="only the ensemble target-policy schema"):
        _validate_policy_grid(
            legacy,
            feature_locks={},
            frozen_actions={},
        )
