from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
import struct

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.harp_action_model import (
    LAMBDA_GRID,
    HarpTargetAction,
    HarpTrainingObservation,
    fit_harp_action_model_bank,
    model_bank_collection_from_payload,
    model_bank_collection_payload,
    score_harp_actions,
)
from midogpp_thesis.cvae.routing.harp_protocol.hashing import canonical_hash


DONORS = ("0", "1", "2", "3", "4")
FEATURES = ("margin_gap", "seed_dispersion")
SEAL = "a" * 64
RESPONSE = "b" * 64
ENSEMBLE = "c" * 64


def _observations() -> tuple[HarpTrainingObservation, ...]:
    rows: list[HarpTrainingObservation] = []
    for q_index, query in enumerate(DONORS):
        for source_index, source in enumerate(DONORS):
            if query == source:
                continue
            for case_index in range(4):
                for sample_index in range(2):
                    feature = (0.1 * (source_index - q_index) + 0.01 * case_index + 0.001 * sample_index, 0.02 + 0.001 * sample_index)
                    truth = sample_index
                    for lam in LAMBDA_GRID:
                        rows.append(
                            HarpTrainingObservation(
                                outer_target_id="H",
                                pseudo_query_id=query,
                                candidate_source_id=source,
                                case_id=f"case-{case_index}",
                                sample_id=f"sample-{sample_index}",
                                lambda_value=lam,
                                direction="ALL_MARGINS",
                                feature_names=FEATURES,
                                feature_values=feature,
                                weighted_correctness_surrogate=0.03 + 0.04 * lam + 0.01 * feature[0],
                                brier_delta=-0.02 * lam + 0.001 * feature[0],
                                log_loss_delta=-0.03 * lam + 0.001 * feature[0],
                                truth_class=truth,
                                ensemble_size=9,
                                ensemble_receipt_hash=ENSEMBLE,
                                case_aggregation_receipt_hash="8" * 64,
                                prediction_seal_hash=SEAL,
                                response_receipt_hash=RESPONSE,
                            )
                        )
    return tuple(rows)


@pytest.fixture(scope="module")
def bank():
    return fit_harp_action_model_bank(_observations(), outer_target_id="H", alphas=(0.1,))


def test_outer_h_and_exact_nine_are_structural_hard_stops() -> None:
    row = _observations()[0]
    with pytest.raises(ProtocolError, match="Outer H"):
        replace(row, pseudo_query_id="H")
    with pytest.raises(ProtocolError, match="exact-nine"):
        replace(row, ensemble_size=8)


def test_nested_and_delete_donor_models_exclude_query_and_candidate_roles(bank) -> None:
    for outcome in bank.models:
        for fold in outcome.nested_lodo_audit:
            assert fold.heldout_donor_id not in fold.training_query_ids
            assert fold.heldout_donor_id not in fold.training_source_ids
        for donor, model in outcome.delete_donor_models:
            assert donor not in model.training_query_ids
            assert donor not in model.training_source_ids
            assert model.excluded_donor_ids == (donor,)


def test_sample_identity_rejection_and_equal_case_clone_invariance(bank) -> None:
    original = _observations()
    with pytest.raises(ProtocolError, match="sample-level.*duplicate identities"):
        fit_harp_action_model_bank((*original, original[0]), outer_target_id="H", alphas=(0.1,))

    clones = tuple(
        replace(row, sample_id=f"{row.sample_id}-clone", ensemble_receipt_hash="9" * 64)
        for row in original
        if row.case_id == "case-0" and row.sample_id == "sample-0"
    )
    cloned_bank = fit_harp_action_model_bank((*original, *clones), outer_target_id="H", alphas=(0.1,))
    left = bank.model("gain", "ALL_MARGINS").full_model
    right = cloned_bank.model("gain", "ALL_MARGINS").full_model
    np.testing.assert_allclose(left.feature_mean, right.feature_mean, rtol=0, atol=1e-12)
    np.testing.assert_allclose(left.coefficients, right.coefficients, rtol=0, atol=1e-11)


def test_scoring_uses_delete_donor_predictions_and_no_seed_index(bank) -> None:
    action = HarpTargetAction(
        outer_target_id="H",
        target_query_id="H",
        candidate_source_id="0",
        case_id="target-case",
        sample_id="target-sample",
        lambda_value=0.5,
        direction="ALL_MARGINS",
        feature_names=FEATURES,
        feature_values=(0.1, 0.02),
        baseline_probability_bytes=struct.pack("<d", 0.4),
        expert_probability=0.8,
        ensemble_size=9,
        ensemble_receipt_hash=ENSEMBLE,
        prediction_seal_hash="d" * 64,
    )
    score = score_harp_actions(bank, (action,))[0]
    assert score.delete_donors == DONORS
    assert len(score.gain_predictions) == len(DONORS)
    assert action.group_key == ("H", "target-case", "target-sample")


def test_model_bank_round_trip_reconstructs_and_rejects_tampered_geometry(bank) -> None:
    payload = model_bank_collection_payload((bank,))
    rebuilt = model_bank_collection_from_payload(payload)
    assert model_bank_collection_payload(rebuilt) == payload
    assert all(not model.coefficients.flags.writeable for outcome in rebuilt[0].models for _donor, model in outcome.delete_donor_models)

    tampered = deepcopy(payload)
    tampered["banks"][0]["models"][0]["full_model"]["coefficients"].pop()
    bank_payload = tampered["banks"][0]
    bank_payload["model_bank_hash"] = canonical_hash({key: value for key, value in bank_payload.items() if key != "model_bank_hash"})
    tampered["collection_hash"] = canonical_hash({key: value for key, value in tampered.items() if key != "collection_hash"})
    with pytest.raises(ProtocolError, match="model-bank values|ridge model state"):
        model_bank_collection_from_payload(tampered)
