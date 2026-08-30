from __future__ import annotations

import pickle
import struct
from dataclasses import replace

import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.harp_portfolio import HarpPortfolioDecision
from midogpp_thesis.cvae.routing.harp_replay import (
    evaluate_harp_replay,
    freeze_harp_predictions,
    issue_harp_replay_capability,
)


def _decisions() -> tuple[HarpPortfolioDecision, ...]:
    rows: list[HarpPortfolioDecision] = []
    for center, case_count in (("A", 1), ("B", 3)):
        for case_index in range(case_count):
            for truth in (0, 1):
                baseline = 0.6 if truth == 0 else 0.4
                routed = center == "A"
                output = (0.2 if truth == 0 else 0.8) if routed else baseline
                rows.append(
                    HarpPortfolioDecision(
                        center, f"case-{case_index}", f"sample-{truth}",
                        struct.pack("<d", baseline), struct.pack("<d", output),
                        "0" if routed else None, 0.5 if routed else None, routed,
                        "CONSERVATIVE_ACTION_ADMITTED" if routed else "EXACT_B_FALLBACK_NO_ADMISSIBLE_ACTION",
                        0.1 if routed else None, -0.01 if routed else None,
                        -0.01 if routed else None, "a" * 64, "b" * 64,
                    )
                )
    return tuple(rows)


def _seal():
    return freeze_harp_predictions(
        _decisions(),
        prediction_surface_hash="c" * 64,
        policy_hash="d" * 64,
        durable_bundle_hash="e" * 64,
        independent_validation_hashes=("f" * 64, "1" * 64),
    )


def test_target_truth_cannot_open_before_durable_seal() -> None:
    with pytest.raises(ProtocolError, match="before the durable prediction seal"):
        issue_harp_replay_capability(  # type: ignore[arg-type]
            _decisions(), target_truth={}, authorization_hash="2" * 64
        )


def test_replay_capability_is_one_shot_and_nonserializable() -> None:
    seal = _seal()
    truth = {row.row_key: int(row.sample_id.rsplit("-", 1)[1]) for row in seal.decisions}
    capability = issue_harp_replay_capability(seal, target_truth=truth, authorization_hash="2" * 64)
    with pytest.raises(ProtocolError, match="nonserializable"):
        pickle.dumps(capability)
    result = evaluate_harp_replay(seal, capability)
    assert result.metrics.aggregation_unit == "equal_target_center"
    assert result.metrics.baseline_balanced_accuracy == 0.0
    assert result.metrics.routed_balanced_accuracy == 0.5
    assert result.descriptive_row_metrics.routed_balanced_accuracy == 0.25
    assert result.metrics.brier_delta < 0
    assert result.metrics.log_loss_delta < 0
    with pytest.raises(ProtocolError, match="already been consumed"):
        evaluate_harp_replay(seal, capability)


def test_two_independent_durable_validations_are_required() -> None:
    with pytest.raises(ProtocolError, match="two independent"):
        freeze_harp_predictions(
            _decisions(),
            prediction_surface_hash="c" * 64,
            policy_hash="d" * 64,
            durable_bundle_hash="e" * 64,
            independent_validation_hashes=("f" * 64,),
        )


def test_replay_primary_is_invariant_to_complete_within_case_replication() -> None:
    original = _decisions()
    selected = tuple(
        row for row in original
        if row.outer_target_id == "B" and row.case_id == "case-0"
    )
    clones = tuple(
        replace(row, sample_id=f"{row.sample_id}-replica") for row in selected
    )
    seal = freeze_harp_predictions(
        (*original, *clones),
        prediction_surface_hash="c" * 64,
        policy_hash="d" * 64,
        durable_bundle_hash="e" * 64,
        independent_validation_hashes=("f" * 64, "1" * 64),
    )
    truth = {
        row.row_key: int(row.sample_id.split("-")[1])
        for row in seal.decisions
    }
    replicated = evaluate_harp_replay(
        seal,
        issue_harp_replay_capability(
            seal, target_truth=truth, authorization_hash="2" * 64
        ),
    )
    base_seal = _seal()
    base_truth = {
        row.row_key: int(row.sample_id.split("-")[1])
        for row in base_seal.decisions
    }
    base = evaluate_harp_replay(
        base_seal,
        issue_harp_replay_capability(
            base_seal, target_truth=base_truth, authorization_hash="2" * 64
        ),
    )
    assert replace(replicated.metrics, row_count=base.metrics.row_count) == base.metrics
