from __future__ import annotations

import inspect
import struct

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.compatibility_conditioned_directional_router import (
    Direction,
    ReplicaEnergyInput,
    SupportPartitionReceipt,
    build_compatibility_receipts,
    build_source_candidate_pool,
    build_target_candidate_pool,
)
from midogpp_thesis.cvae.runtime.harp_v5_execution.contracts import (
    ActionKind,
    LabelFreeActionBlock,
    LabelFreeOuterMenu,
)
from midogpp_thesis.cvae.runtime.harp_v5_execution.directional_surfaces import (
    build_source_directional_observations,
    build_target_directional_actions,
    directional_probability_bytes,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
CENTERS = ("C", "H", "Q")
SAMPLES = ("s0", "s1", "s2", "s3")
CASES = ("case-0", "case-0", "case-1", "case-1")


def _block(
    *,
    role: str,
    query: str,
    kind: ActionKind,
    source: str | None,
    values: tuple[float, ...],
    dispersion: tuple[float, ...] = (0.01, 0.02, 0.03, 0.04),
) -> LabelFreeActionBlock:
    return LabelFreeActionBlock(
        surface_role=role,
        outer_target_id="H",
        query_center_id=query,
        action_kind=kind,
        selected_source_id=source,
        sample_ids=SAMPLES,
        case_ids=CASES,
        probabilities=np.asarray(values, dtype=np.float32),
        seed_dispersion=np.asarray(dispersion, dtype=np.float32),
    )


def _menu() -> LabelFreeOuterMenu:
    base = (0.2, 0.8, 0.2, 0.8)
    uniform = (0.6, 0.4, 0.1, 0.9)
    expert_c = (0.7, 0.3, 0.1, 0.9)
    expert_q = (0.3, 0.7, 0.8, 0.2)
    blocks = [
        _block(role="development", query="Q", kind=ActionKind.B, source=None, values=base),
        _block(role="development", query="Q", kind=ActionKind.U, source=None, values=uniform),
        _block(
            role="development", query="Q", kind=ActionKind.HXE, source="C", values=expert_c
        ),
        _block(role="target", query="H", kind=ActionKind.B, source=None, values=base),
        _block(role="target", query="H", kind=ActionKind.U, source=None, values=uniform),
        _block(role="target", query="H", kind=ActionKind.HXE, source="C", values=expert_c),
        _block(role="target", query="H", kind=ActionKind.HXE, source="Q", values=expert_q),
    ]
    return LabelFreeOuterMenu(
        outer_target_id="H",
        blocks=tuple(sorted(blocks, key=lambda row: row.key)),
        lineage={"physical": True},
    )


def _compatibility(pool):
    partition = SupportPartitionReceipt(
        center_id=pool.query_center_id,
        support_case_ids=(f"{pool.query_center_id}-support",),
        evaluation_case_ids=(f"{pool.query_center_id}-evaluation",),
        support_manifest_hash=SHA_A,
        evaluation_manifest_hash=SHA_B,
    )
    replicas = tuple(
        ReplicaEnergyInput(
            candidate_source_id=source,
            training_seed=seed,
            query_case_equal_energy=1.0 + 0.1 * source_index + 0.001 * seed_index,
            own_source_location=1.0,
            own_source_scale=0.5,
            checkpoint_hash=SHA_A,
            source_frame_hash=SHA_B,
            sampler_hash=SHA_C,
        )
        for source_index, source in enumerate(pool.candidate_center_ids)
        for seed_index, seed in enumerate((17, 42, 101))
    )
    return build_compatibility_receipts(
        candidate_pool=pool,
        support_partition=partition,
        replica_energies=replicas,
    )


def _unpack(values: tuple[bytes, ...]) -> tuple[float, ...]:
    return tuple(struct.unpack("<f", value)[0] for value in values)


def test_directional_masks_partition_crossings_and_margins_with_exact_b_bytes() -> None:
    menu = _menu()
    baseline = next(
        row
        for row in menu.blocks
        if row.surface_role == "target" and row.action_kind is ActionKind.B
    )
    challenger = next(
        row
        for row in menu.blocks
        if row.surface_role == "target"
        and row.action_kind is ActionKind.HXE
        and row.selected_source_id == "C"
    )
    _, d01 = directional_probability_bytes(
        baseline, challenger, case_id="case-0", direction=Direction.D01
    )
    _, d10 = directional_probability_bytes(
        baseline, challenger, case_id="case-0", direction=Direction.D10
    )
    _, margins = directional_probability_bytes(
        baseline, challenger, case_id="case-1", direction=Direction.ALL
    )

    assert _unpack(d01) == pytest.approx((0.7, 0.8))
    assert _unpack(d10) == pytest.approx((0.2, 0.3))
    assert _unpack(margins) == pytest.approx((0.1, 0.9))
    baseline_bytes = tuple(
        np.asarray([value], dtype="<f4").tobytes()
        for value in baseline.probabilities
    )
    assert d01[1] == baseline_bytes[1]
    assert d10[0] == baseline_bytes[0]


def test_target_builder_is_label_free_complete_and_candidate_aware() -> None:
    assert "label" not in inspect.signature(build_target_directional_actions).parameters
    pool = build_target_candidate_pool(
        outer_target_id="H", all_center_ids=CENTERS, bank_lock_hash=SHA_A
    )
    actions = build_target_directional_actions(
        _menu(), candidate_pool=pool, compatibility_receipts=_compatibility(pool)
    )

    # Two cases x (U + two legal experts) x three complete directions.
    assert len(actions) == 18
    assert {row.feature.direction for row in actions} == {
        Direction.D01,
        Direction.D10,
        Direction.ALL,
    }
    assert all(row.feature.outer_target_id == "H" for row in actions)
    assert all(row.feature.query_center_id == "H" for row in actions)
    assert all(row.prediction_seal_hash == _menu().menu_hash for row in actions)
    expert = next(
        row
        for row in actions
        if row.feature.action_id == "HXE:C:D01"
        and row.feature.case_id == "case-0"
    )
    feature = dict(zip(expert.feature.feature_names, expert.feature.feature_values, strict=True))
    assert feature["threshold_flip_fraction"] == pytest.approx(0.5)
    assert feature["signed_branch_mass"] > 0.0
    assert feature["surface_seed_dispersion_mean"] > 0.0
    assert feature["compatibility_available"] == 1.0
    uniform = next(
        row
        for row in actions
        if row.feature.action_id == "U:D01" and row.feature.case_id == "case-0"
    )
    uniform_feature = dict(
        zip(uniform.feature.feature_names, uniform.feature.feature_values, strict=True)
    )
    assert uniform_feature["compatibility_available"] == 0.0


def test_source_builder_uses_case_equal_contributions_and_exact_label_keys() -> None:
    pool = build_source_candidate_pool(
        outer_target_id="H",
        pseudo_query_id="Q",
        all_center_ids=CENTERS,
        bank_lock_hash=SHA_A,
    )
    labels = {
        ("case-0", "s0"): 1,
        ("case-0", "s1"): 0,
        ("case-1", "s2"): 0,
        ("case-1", "s3"): 1,
    }
    observations = build_source_directional_observations(
        _menu(),
        candidate_pool=pool,
        compatibility_receipts=_compatibility(pool),
        source_labels=labels,
    )

    # Two cases x (U + one C-minus-H-minus-Q expert) x three directions.
    assert len(observations) == 12
    d01 = next(
        row
        for row in observations
        if row.feature.case_id == "case-0" and row.feature.action_id == "HXE:C:D01"
    )
    # B is wrong for both case-0 classes; D01 repairs its positive sample.
    # With two cases supporting each class, the additive case contribution is .5.
    assert d01.effects.bacc_gain == pytest.approx(0.5)
    assert d01.effects.brier_delta < 0.0
    assert d01.effects.log_delta < 0.0

    with pytest.raises(ProtocolError, match="source-label keys differ"):
        build_source_directional_observations(
            _menu(),
            candidate_pool=pool,
            compatibility_receipts=_compatibility(pool),
            source_labels={**labels, ("foreign-case", "foreign-sample"): 1},
        )


def test_surface_builders_reject_compatibility_from_another_hq_pool() -> None:
    source_pool = build_source_candidate_pool(
        outer_target_id="H",
        pseudo_query_id="Q",
        all_center_ids=CENTERS,
        bank_lock_hash=SHA_A,
    )
    target_pool = build_target_candidate_pool(
        outer_target_id="H", all_center_ids=CENTERS, bank_lock_hash=SHA_A
    )
    labels = dict(zip(zip(CASES, SAMPLES, strict=True), (1, 0, 0, 1), strict=True))

    with pytest.raises(ProtocolError, match="compatibility"):
        build_source_directional_observations(
            _menu(),
            candidate_pool=source_pool,
            compatibility_receipts=_compatibility(target_pool),
            source_labels=labels,
        )
