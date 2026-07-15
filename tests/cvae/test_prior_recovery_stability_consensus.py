from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from midogpp_thesis.cvae.generation_samplers import (
    DIAGONAL_SAMPLER,
    FULL_SAMPLER,
    STANDARD_SAMPLER,
)
from midogpp_thesis.cvae.objectives import ISOTROPIC_OBJECTIVE, TASK_FISHER_OBJECTIVE
from midogpp_thesis.cvae.preservation.prior_recovery_stability_consensus import (
    TrainingSeedRecipeLock,
    select_training_seed_consensus,
)
from midogpp_thesis.cvae.preservation.prior_recovery_provenance import (
    ProvenanceRecorder,
)
from midogpp_thesis.cvae.preservation.source_inner_selection import RecipeLock
from midogpp_thesis.real_features.classifier_reference.protocol import ProtocolError


SEEDS = (17, 42, 101)


def test_consensus_truth_table() -> None:
    all_a = _consensus([_wrapped(seed, "A", STANDARD_SAMPLER) for seed in SEEDS])
    assert (all_a.primary_arm, all_a.stability_status, all_a.recipe_export_ready) == (
        "A",
        "STABLE_STANDARD_FALLBACK",
        True,
    )

    mixed_a = _consensus(
        [
            _wrapped(17, "A", STANDARD_SAMPLER),
            _wrapped(42, "C", DIAGONAL_SAMPLER),
            _wrapped(101, "D", DIAGONAL_SAMPLER),
        ]
    )
    assert mixed_a.primary_arm == "A"
    assert mixed_a.consensus_origin == "CONSERVATIVE_DIVERGENCE_FALLBACK"
    assert mixed_a.stability_status == "CROSS_SEED_DISAGREEMENT"

    family_disagreement = _consensus(
        [
            _wrapped(17, "C", DIAGONAL_SAMPLER),
            _wrapped(42, "C", FULL_SAMPLER),
            _wrapped(101, "D", FULL_SAMPLER),
        ]
    )
    assert family_disagreement.primary_arm == "A"
    assert family_disagreement.reason == "conditional_sampler_family_disagreement"

    mixed_objective = _consensus(
        [
            _wrapped(17, "C", FULL_SAMPLER),
            _wrapped(42, "D", FULL_SAMPLER),
            _wrapped(101, "D", FULL_SAMPLER),
        ]
    )
    assert mixed_objective.primary_arm == "C"
    assert mixed_objective.objective_id == ISOTROPIC_OBJECTIVE
    assert mixed_objective.stability_status == "STABLE_SAMPLER_OBJECTIVE_DIVERGENCE"

    all_d = _consensus([_wrapped(seed, "D", FULL_SAMPLER) for seed in SEEDS])
    assert all_d.primary_arm == "D"
    assert all_d.objective_id == TASK_FISHER_OBJECTIVE
    assert all_d.stability_status == "STABLE_CONDITIONAL"


def test_invalid_seed_lock_disables_export() -> None:
    locks = [_wrapped(seed, "A", STANDARD_SAMPLER) for seed in SEEDS]
    locks[1] = replace(
        locks[1],
        recipe_lock=replace(locks[1].recipe_lock, status="INVALID"),
    )
    consensus = _consensus(locks)
    assert consensus.integrity_status == "INVALID"
    assert consensus.recipe_export_ready is False


def test_consensus_rejects_cross_center_or_missing_seed() -> None:
    locks = [_wrapped(seed, "A", STANDARD_SAMPLER) for seed in SEEDS]
    with pytest.raises(ProtocolError, match="different outer centers"):
        _consensus([locks[0], replace(locks[1], outer_target_center="1"), locks[2]])
    with pytest.raises(ProtocolError, match="coverage/order"):
        _consensus(locks[:2])


def test_stability_provenance_preserves_distinct_training_identities_for_shared_bytes(
    tmp_path: Path,
) -> None:
    first = {"checkpoint_hash": "a" * 64, "training_key_hash": "1" * 16}
    second = {"checkpoint_hash": "a" * 64, "training_key_hash": "2" * 16}
    recorder = ProvenanceRecorder(tmp_path, allow_shared_checkpoint_hashes=True)
    recorder._record_checkpoint(first)
    recorder._record_checkpoint(second)
    assert len(recorder.checkpoint_records) == 2

    strict = ProvenanceRecorder(tmp_path)
    strict._record_checkpoint(first)
    with pytest.raises(ProtocolError, match="collision"):
        strict._record_checkpoint(second)


def _consensus(locks: list[TrainingSeedRecipeLock]):
    return select_training_seed_consensus(
        locks,
        outer_target_center="0",
        training_seeds=SEEDS,
        parent_protocol_hash="parent",
        parent_selection_bundle_hash="bundle",
    )


def _wrapped(seed: int, arm: str, family: str) -> TrainingSeedRecipeLock:
    objective = TASK_FISHER_OBJECTIVE if arm == "D" else ISOTROPIC_OBJECTIVE
    summary = (
        {"sampler": {"mean_delta": 0.1, "strict_wins": 7}}
        if arm in {"C", "D"}
        else {"sampler_gate": "NO_PASS"}
    )
    lock = RecipeLock(
        outer_target_center="0",
        status="VALID",
        primary_arm=arm,
        objective_id=objective,
        sampler_family=family,
        alpha=1.0 if arm == "D" else 0.0,
        beta_final=0.001,
        generation_seeds=SEEDS,
        inner_centers=("1", "2"),
        gate_summary=summary,
        classifier_grid_hash="grid",
        protocol_hash=f"protocol-{seed}",
        source_metric_table_hash=f"metrics-{seed}",
        fit_center_sets_hash="fits",
        recipe_contract_hash=f"contract-{seed}",
        selection_bundle_hash=f"evidence-{seed}",
    )
    return TrainingSeedRecipeLock(
        training_seed=seed,
        outer_target_center="0",
        recipe_lock=lock,
        seed_evidence_hash=f"evidence-{seed}",
        per_seed_contract_hash=f"contract-{seed}",
        parent_protocol_hash="parent",
        checkpoint_hashes=(f"checkpoint-{seed}",),
        sampler_state_hashes=(f"sampler-{seed}",),
    )
