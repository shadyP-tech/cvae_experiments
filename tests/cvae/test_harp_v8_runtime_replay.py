from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.harp_v8_execution import validation
from midogpp_thesis.cvae.runtime.harp_v8_execution.contracts import ArtifactValue


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _model_and_admission() -> tuple[ArtifactValue, ArtifactValue]:
    certificate = {
        "action_id": "U:D01",
        "action_hash": SHA_A,
        "direction": "D01",
        "action_group": "U:D01",
        "predicted_bacc_gain": 0.04,
        "predicted_harm_probability": 0.2,
        "predicted_brier_delta": -0.01,
        "predicted_log_delta": -0.02,
        "model_available": True,
        "gain_lcb": 0.01,
        "harm_probability_ucb": 0.3,
        "brier_delta_ucb": 0.0,
        "log_delta_ucb": 0.0,
        "harm_brier_risk": 0.1,
        "harm_log_loss_risk": 0.5,
        "calibration_cell_hash": SHA_C,
        "safe": True,
        "failed_gates": [],
        "certificate_hash": SHA_D,
    }
    prediction = {
        "query_center_id": "C",
        "case_id": "case-1",
        "rank_margin": 0.0,
        "safe_action_ids": ["U:D01"],
        "model_hash": SHA_A,
        "menu_hash": SHA_C,
        "prediction_hash": SHA_B,
        "action_certificates": [certificate],
    }
    nested_prediction = {
        **prediction,
        "model_hash": SHA_C,
        "prediction_hash": SHA_D,
    }
    model = ArtifactValue(
        state=None,
        manifest={
            "outer_models": [
                {
                    "outer_target_id": "H",
                    "numeric_oof": {
                        "rows": [prediction],
                        "nested_policy_folds": [
                            {
                                "heldout_center_id": "C",
                                "training_center_ids": ["A", "B"],
                                "fold_hash": SHA_A,
                                "heldout_rows": [nested_prediction],
                            }
                        ]
                    },
                }
            ]
        },
        arrays={
            "oof_case_values": np.asarray(
                [[0.8, 0.0, 1.0, 1.0]], dtype=np.float64
            ),
            "oof_action_certificates": np.asarray(
                [
                    [
                        0.04,
                        0.2,
                        -0.01,
                        -0.02,
                        0.01,
                        0.3,
                        0.0,
                        0.0,
                        0.1,
                        0.5,
                        1.0,
                        1.0,
                    ]
                ],
                dtype=np.float64,
            ),
            "oof_action_certificate_offsets": np.asarray([0, 1], dtype=np.int64),
        },
    )
    row = {
        "outer_target_id": "H",
        "query_center_id": "C",
        "case_id": "case-1",
        "selected_certificate_confidence": 0.7,
        "rank_margin": 0.0,
        "selected_action_id": "U:D01",
        "observed_bacc_gain": 0.03,
        "observed_brier_delta": -0.01,
        "observed_log_delta": -0.02,
        "best_observed_bacc_gain": 0.03,
        "regret": 0.0,
        "nested_certificate_confidence": 0.65,
        "nested_rank_margin": 0.0,
        "nested_selected_action_id": "U:D01",
        "nested_observed_bacc_gain": 0.03,
        "nested_observed_brier_delta": -0.01,
        "nested_observed_log_delta": -0.02,
        "nested_regret": 0.0,
        "nested_certificate_confidence_threshold": 0.6,
        "nested_rank_margin_threshold": 0.0,
        "nested_threshold_training_center_ids": ["A", "B"],
        "nested_policy_fold_hash": SHA_A,
        "nested_policy_replay_hash": SHA_B,
        "nested_heldout_model_hash": SHA_C,
        "nested_prediction_hash": SHA_D,
        "heldout_model_hash": SHA_A,
        "prediction_hash": SHA_B,
        "menu_hash": SHA_C,
        "safe_action_ids": ["U:D01"],
        "nested_safe_action_ids": ["U:D01"],
        "action_certificates": [certificate],
    }
    admission = ArtifactValue(
        state=None,
        manifest={
            "outer_policies": [
                {
                    "outer_target_id": "H",
                    "calibration": {
                        "nested_replay_hash": SHA_B,
                        "heldout_thresholds": [
                            {
                                "heldout_center_id": "C",
                                "certificate_confidence_threshold": 0.6,
                                "rank_margin_threshold": 0.0,
                            }
                        ],
                    },
                }
            ],
            "source_policy_oof_rows": [row],
            "source_policy_oof_case_count": 1,
            "nested_held_source_threshold_policy_replayed": True,
        },
        arrays={
            "source_policy_oof_values": np.asarray(
                [[0.7, 0.0, 1.0, 0.03, -0.01, -0.02, 0.03, 0.0]],
                dtype=np.float64,
            ),
            "nested_source_policy_oof_values": np.asarray(
                [[0.65, 0.0, 1.0, 0.03, -0.01, -0.02, 0.03, 0.0]],
                dtype=np.float64,
            ),
        },
    )
    return model, admission


def test_numeric_certificate_and_nested_policy_replay_are_byte_bound() -> None:
    model, admission = _model_and_admission()
    validation._validate_numeric_oof(model, admission)

    changed = {
        name: np.asarray(value).copy() for name, value in model.arrays.items()
    }
    changed["oof_action_certificates"][0, 4] += 0.001
    with pytest.raises(ProtocolError, match="certificate binding"):
        validation._validate_numeric_oof(
            ArtifactValue(state=None, manifest=model.manifest, arrays=changed),
            admission,
        )

    bad_row = dict(admission.manifest["source_policy_oof_rows"][0])
    bad_row["nested_threshold_training_center_ids"] = ["A", "C"]
    with pytest.raises(ProtocolError, match="threshold provenance"):
        validation._validate_numeric_oof(
            model,
            ArtifactValue(
                state=None,
                manifest={
                    **dict(admission.manifest),
                    "source_policy_oof_rows": [bad_row],
                },
                arrays=admission.arrays,
            ),
        )


def _target_artifact() -> tuple[
    ArtifactValue,
    MappingProxyType[tuple[str, str, str], tuple[str, str, bytes]],
]:
    values = np.asarray([0.7, 0.3], dtype=np.float32)
    effective = MappingProxyType(
        {("H", "case-1", "U:D01"): (SHA_A, SHA_B, values.tobytes(order="C"))}
    )
    certificate = {
        "action_id": "U:D01",
        "action_hash": SHA_A,
        "direction": "D01",
        "predicted_bacc_gain": 0.04,
        "predicted_harm_probability": 0.2,
        "predicted_brier_delta": -0.01,
        "predicted_log_delta": -0.02,
        "model_available": True,
        "gain_lcb": 0.01,
        "harm_probability_ucb": 0.3,
        "brier_delta_ucb": 0.0,
        "log_delta_ucb": 0.0,
        "harm_brier_risk": 0.1,
        "harm_log_loss_risk": 0.5,
        "calibration_cell_hash": SHA_C,
        "safe": True,
        "failed_gates": [],
        "certificate_hash": SHA_D,
    }
    target = ArtifactValue(
        state=None,
        manifest={
            "prediction_rows": [
                {
                    "outer_target_id": "H",
                    "case_id": "case-1",
                    "rank_margin": 0.0,
                    "raw_top_action_id": "U:D01",
                    "top_action_id": "U:D01",
                    "safe_action_ids": ["U:D01"],
                    "training_center_ids": ["A", "B"],
                    "training_candidate_ids": ["A", "B"],
                    "excluded_center_ids": ["H"],
                    "action_certificates": [certificate],
                }
            ]
        },
        arrays={
            "probabilities": values,
            "probability_offsets": np.asarray([0, 2], dtype=np.int64),
            "case_prediction_values": np.asarray(
                [[0.8, 0.0, 1.0, 1.0]], dtype=np.float64
            ),
            "action_certificate_values": np.asarray(
                [
                    [
                        0.04,
                        0.2,
                        -0.01,
                        -0.02,
                        0.01,
                        0.3,
                        0.0,
                        0.0,
                        0.1,
                        0.5,
                        1.0,
                        1.0,
                    ]
                ],
                dtype=np.float64,
            ),
            "action_certificate_offsets": np.asarray([0, 1], dtype=np.int64),
        },
    )
    return target, effective


def test_target_certificate_projection_rejects_numeric_tampering() -> None:
    target, effective = _target_artifact()
    validation._validate_target_certificate_projection(target, effective=effective)

    changed = {
        name: np.asarray(value).copy() for name, value in target.arrays.items()
    }
    changed["action_certificate_values"][0, 5] += 0.01
    with pytest.raises(ProtocolError, match="certificate row/value binding"):
        validation._validate_target_certificate_projection(
            ArtifactValue(state=None, manifest=target.manifest, arrays=changed),
            effective=effective,
        )
