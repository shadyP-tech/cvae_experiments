from __future__ import annotations

import numpy as np

from midogpp_thesis.cvae.block_frame import (
    PCAState,
    PilotFeatureFrame,
    bridge_a_prefix,
    fit_pilot_frame,
)
from midogpp_thesis.cvae.case_split import CaseHoldout, deterministic_case_holdout
from midogpp_thesis.cvae.diagnostics.b_paired_reparameterization_audit import (
    snapshot_builder,
)
from midogpp_thesis.cvae.expert_bank.b_adaptation_pilot.block_frame import (
    PCAState as PilotPCAState,
)
from midogpp_thesis.cvae.expert_bank.b_adaptation_pilot.block_frame import (
    PilotFeatureFrame as LegacyPilotFeatureFrame,
)
from midogpp_thesis.cvae.expert_bank.b_adaptation_pilot.block_frame import (
    bridge_a_prefix as pilot_bridge_a_prefix,
)
from midogpp_thesis.cvae.expert_bank.b_adaptation_pilot.block_frame import (
    fit_pilot_frame as pilot_fit_pilot_frame,
)
from midogpp_thesis.cvae.expert_bank.b_adaptation_pilot.case_split import (
    CaseHoldout as PilotCaseHoldout,
)
from midogpp_thesis.cvae.expert_bank.b_adaptation_pilot.case_split import (
    deterministic_case_holdout as pilot_deterministic_case_holdout,
)


def test_pilot_preparation_modules_are_thin_compatibility_exports() -> None:
    assert PilotPCAState is PCAState
    assert LegacyPilotFeatureFrame is PilotFeatureFrame
    assert pilot_bridge_a_prefix is bridge_a_prefix
    assert pilot_fit_pilot_frame is fit_pilot_frame
    assert PilotCaseHoldout is CaseHoldout
    assert pilot_deterministic_case_holdout is deterministic_case_holdout


def test_paired_audit_uses_neutral_preparation_primitives() -> None:
    assert snapshot_builder.fit_pilot_frame is fit_pilot_frame
    assert snapshot_builder.deterministic_case_holdout is deterministic_case_holdout


def test_neutral_block_frame_preserves_payload_and_round_trip() -> None:
    blocks = (
        PCAState(
            start=0,
            stop=1,
            output_dim=1,
            scaler_mean=np.asarray([1.0]),
            scaler_scale=np.asarray([2.0]),
            pca_mean=np.asarray([0.5]),
            pca_components=np.asarray([[1.0]]),
            explained_variance=np.asarray([1.0]),
            explained_variance_ratio_sum=1.0,
        ),
        PCAState(
            start=1,
            stop=2,
            output_dim=1,
            scaler_mean=np.asarray([-1.0]),
            scaler_scale=np.asarray([4.0]),
            pca_mean=np.asarray([-0.25]),
            pca_components=np.asarray([[1.0]]),
            explained_variance=np.asarray([1.0]),
            explained_variance_ratio_sum=1.0,
        ),
    )
    frame = PilotFeatureFrame(
        arm="b_block_pca96_32",
        input_dim=2,
        output_dim=2,
        blocks=blocks,
        fit_sample_hash="fit-rows",
    )
    embeddings = np.asarray([[1.0, -1.0], [3.0, 3.0]], dtype=np.float32)

    projected = frame.transform(embeddings)
    reconstructed = frame.inverse_transform(projected)

    assert frame.to_payload()["schema_version"] == (
        "midogpp_b_adaptation_feature_frame_v1"
    )
    assert len(frame.state_hash) == 16
    np.testing.assert_allclose(reconstructed, embeddings)


def test_neutral_case_holdout_is_deterministic_and_case_disjoint() -> None:
    case_ids = tuple(
        case
        for index in range(10)
        for case in (f"case-{index}", f"case-{index}")
    )
    labels = (0, 1) * 10

    first = deterministic_case_holdout(
        case_ids,
        labels,
        validation_fraction=0.2,
        seed=42,
    )
    second = deterministic_case_holdout(
        case_ids,
        labels,
        validation_fraction=0.2,
        seed=42,
    )

    assert first == second
    assert not set(first.fit_cases).intersection(first.eval_cases)
    assert {labels[index] for index in first.fit_indices} == {0, 1}
    assert {labels[index] for index in first.eval_indices} == {0, 1}


def test_neutral_a_prefix_bridge_preserves_legacy_schema() -> None:
    a_embeddings = np.ones((2, 2560), dtype=np.float32)
    b_embeddings = np.concatenate(
        (a_embeddings, np.zeros((2, 1280), dtype=np.float32)),
        axis=1,
    )

    result = bridge_a_prefix(b_embeddings, a_embeddings)

    assert result["schema_version"] == "midogpp_uniform_b_a_prefix_bridge_v1"
    assert result["status"] == "PASS"
