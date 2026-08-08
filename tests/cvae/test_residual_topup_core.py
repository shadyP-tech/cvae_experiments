from __future__ import annotations

import json

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.residual_topup import (
    BORDA_ACTION_KIND,
    BORDA_DIRECTION_SEMANTICS,
    ENERGY_ACTION_KIND,
    ENERGY_RANK_DIRECTION_SEMANTICS,
    SINGLE_SOURCE_TAIL_ACTION_KIND,
    SINGLE_SOURCE_TAIL_DIRECTION_SEMANTICS,
    SOFTMAX_ENERGY_ACTION_KIND,
    SourceClassWindows,
    build_borda_directed_topup_action,
    build_energy_directed_topup_action,
    build_hamilton_topup_allocation,
    build_softmax_energy_topup_action,
    build_single_source_tail_action,
    build_topup_geometry,
    build_uniform_topup_action,
    compose_equal_union_base_blocks,
    compose_residual_topup_blocks,
    hamilton_topup_allocate,
    inner_topup_geometry,
    target_topup_geometry,
)


TARGET_SOURCES = tuple(str(index) for index in range(8))
INNER_SOURCES = tuple(str(index) for index in range(7))


def _small_geometry():
    return build_topup_geometry(
        TARGET_SOURCES,
        base_per_source=8,
        topup_total_per_class=8,
    )


def _block(source_index: int, *, capacity_per_class: int = 16):
    def rows(class_label: int) -> np.ndarray:
        return np.asarray(
            [
                (float(source_index), float(class_label), float(index))
                for index in range(capacity_per_class)
            ],
            dtype=np.float32,
        )

    return {
        "embeddings": np.concatenate((rows(0), rows(1)), axis=0),
        "labels": np.asarray(
            [0] * capacity_per_class + [1] * capacity_per_class,
            dtype=np.int64,
        ),
    }


def _blocks(*, capacity_per_class: int = 16):
    return {
        source: _block(index, capacity_per_class=capacity_per_class)
        for index, source in enumerate(TARGET_SOURCES)
    }


def test_target_and_inner_geometries_have_exact_locked_budgets() -> None:
    target = target_topup_geometry(reversed(TARGET_SOURCES))
    assert target.source_order == TARGET_SOURCES
    assert target.base_per_source == 128
    assert target.base_total_per_class == 1024
    assert target.topup_total_per_class == 128
    assert target.final_total_per_class == 1152
    target_uniform = build_uniform_topup_action(target)
    assert dict(target_uniform.topup_counts) == {
        source: 16 for source in TARGET_SOURCES
    }
    assert all(
        dict(target_uniform.final_counts_by_class[label])
        == {source: 144 for source in TARGET_SOURCES}
        for label in (0, 1)
    )

    inner = inner_topup_geometry(reversed(INNER_SOURCES))
    assert inner.source_order == INNER_SOURCES
    assert inner.base_per_source == 144
    assert inner.base_total_per_class == 1008
    assert inner.topup_total_per_class == 126
    assert inner.final_total_per_class == 1134
    inner_uniform = build_uniform_topup_action(inner)
    assert dict(inner_uniform.topup_counts) == {
        source: 18 for source in INNER_SOURCES
    }
    assert all(
        dict(inner_uniform.final_counts_by_class[label])
        == {source: 162 for source in INNER_SOURCES}
        for label in (0, 1)
    )


def test_rank_energy_action_is_parameter_free_canonical_and_deterministic() -> None:
    geometry = target_topup_geometry(TARGET_SOURCES)
    energies = {
        "7": 4.0,
        "6": 3.0,
        "5": 3.0,
        "4": 2.0,
        "3": 2.0,
        "2": 1.0,
        "1": 1.0,
        "0": 0.0,
    }
    first = build_energy_directed_topup_action(energies, geometry=geometry)
    second = build_energy_directed_topup_action(
        dict(reversed(tuple(energies.items()))), geometry=geometry
    )
    assert first.action_kind == ENERGY_ACTION_KIND
    assert first.direction_semantics == ENERGY_RANK_DIRECTION_SEMANTICS
    assert first.temperature is None
    assert first.action_hash == second.action_hash
    assert first.allocation_hash == second.allocation_hash
    assert dict(first.topup_counts) == dict(second.topup_counts)
    priorities = [first.direction_weights[source] for source in TARGET_SOURCES]
    assert priorities == sorted(priorities, reverse=True)
    assert first.direction_weights["1"] > first.direction_weights["2"]
    assert first.direction_weights["3"] > first.direction_weights["4"]
    json.dumps(first.to_payload(), sort_keys=True)


def test_existing_stage90_action_hashes_remain_byte_stable() -> None:
    geometry = target_topup_geometry(TARGET_SOURCES)
    energies = {source: float(index) for index, source in enumerate(TARGET_SOURCES)}
    assert (
        build_uniform_topup_action(geometry).action_hash
        == "87d543a558fa1e0042e220b6309367463829886eb9cb9790bed3f3323e4e8fb4"
    )
    assert (
        build_energy_directed_topup_action(
            energies, geometry=geometry
        ).action_hash
        == "218e19a3bec21beea69985ccde1b709b59380ee0b11eae3fd72d609228157b9a"
    )
    assert (
        build_softmax_energy_topup_action(
            energies, geometry=geometry, temperature=1.0
        ).action_hash
        == "981b5d14c96cb37b283583c5a26423cea514870cc7b54f92d646897ce0ea1199"
    )


def test_borda_action_uses_one_minus_midrank_and_preserves_true_ties() -> None:
    geometry = target_topup_geometry(TARGET_SOURCES)
    ballots = {
        "7": 1.0,
        "6": 0.75,
        "5": 0.75,
        "4": 0.5,
        "3": 0.5,
        "2": 0.25,
        "1": 0.25,
        "0": 0.0,
    }
    first = build_borda_directed_topup_action(ballots, geometry=geometry)
    second = build_borda_directed_topup_action(
        dict(reversed(tuple(ballots.items()))), geometry=geometry
    )

    assert first.action_kind == BORDA_ACTION_KIND
    assert first.direction_semantics == BORDA_DIRECTION_SEMANTICS
    assert first.temperature is None
    assert dict(first.calibrated_energy_by_source) == {}
    assert sum(first.direction_weights.values()) == pytest.approx(1.0)
    assert first.direction_weights["0"] > first.direction_weights["1"]
    assert first.direction_weights["1"] == first.direction_weights["2"]
    assert first.direction_weights["3"] == first.direction_weights["4"]
    assert first.direction_weights["5"] == first.direction_weights["6"]
    assert first.direction_weights["6"] > first.direction_weights["7"]
    assert first.action_hash == second.action_hash
    assert first.allocation_hash == second.allocation_hash
    assert first.window_hash == second.window_hash
    assert first.action_hash != build_energy_directed_topup_action(
        ballots, geometry=geometry
    ).action_hash


def test_single_source_tail_is_distinct_and_uses_shared_action_invariants() -> None:
    geometry = target_topup_geometry(TARGET_SOURCES)
    first = build_single_source_tail_action("3", geometry=geometry)
    repeated = build_single_source_tail_action(3, geometry=geometry)
    same_counts_energy = build_softmax_energy_topup_action(
        {
            source: (0.0 if source == "3" else 1000.0)
            for source in TARGET_SOURCES
        },
        geometry=geometry,
        temperature=1.0,
    )

    assert first.action_kind == SINGLE_SOURCE_TAIL_ACTION_KIND
    assert first.direction_semantics == SINGLE_SOURCE_TAIL_DIRECTION_SEMANTICS
    assert dict(first.calibrated_energy_by_source) == {}
    assert dict(first.topup_counts) == {
        source: (128 if source == "3" else 0) for source in TARGET_SOURCES
    }
    assert first.action_hash == repeated.action_hash
    assert first.allocation_hash == same_counts_energy.allocation_hash
    assert first.window_hash == same_counts_energy.window_hash
    assert first.action_hash != same_counts_energy.action_hash
    assert first.maximum_source_weight <= 0.25
    assert all(
        value >= 6.0 for value in first.effective_source_count_by_class.values()
    )


def test_softmax_helper_is_explicitly_nondefault_and_lower_energy_gets_more() -> None:
    geometry = target_topup_geometry(TARGET_SOURCES)
    energies = {source: float(index) for index, source in enumerate(TARGET_SOURCES)}
    action = build_softmax_energy_topup_action(
        energies, geometry=geometry, temperature=1.0
    )
    assert action.action_kind == SOFTMAX_ENERGY_ACTION_KIND
    assert action.temperature == 1.0
    assert action.direction_weights["0"] > action.direction_weights["1"]
    assert action.topup_counts["0"] > action.topup_counts["7"]

    tied = build_softmax_energy_topup_action(
        {source: 2.0 for source in TARGET_SOURCES},
        geometry=geometry,
        temperature=0.5,
    )
    assert dict(tied.topup_counts) == {
        source: 16 for source in TARGET_SOURCES
    }


def test_hamilton_ties_and_hashes_are_canonical() -> None:
    first = build_hamilton_topup_allocation(
        {"c": 1.0, "a": 1.0, "b": 1.0},
        topup_total_per_class=8,
    )
    second = build_hamilton_topup_allocation(
        {"b": 1.0, "c": 1.0, "a": 1.0},
        topup_total_per_class=8,
    )
    assert dict(first.counts) == {"a": 3, "b": 3, "c": 2}
    assert first.remainder_order == ("a", "b", "c")
    assert first.allocation_hash == second.allocation_hash
    assert hamilton_topup_allocate(
        {"b": 0.0, "a": 1.0}, topup_total_per_class=3
    ) == {"a": 3, "b": 0}


def test_invalid_fraction_temperature_energy_and_hamilton_fail_closed() -> None:
    with pytest.raises(ProtocolError, match="1/8 exactly"):
        build_topup_geometry(
            TARGET_SOURCES,
            base_per_source=128,
            topup_total_per_class=127,
        )
    geometry = target_topup_geometry(TARGET_SOURCES)
    energies = {source: float(index) for index, source in enumerate(TARGET_SOURCES)}
    for bad_temperature in (0.0, -1.0, float("nan"), float("inf"), True):
        with pytest.raises(ProtocolError, match="temperature"):
            build_softmax_energy_topup_action(
                energies,
                geometry=geometry,
                temperature=bad_temperature,
            )
    invalid = dict(energies)
    invalid["4"] = float("nan")
    with pytest.raises(ProtocolError, match="finite"):
        build_energy_directed_topup_action(invalid, geometry=geometry)
    with pytest.raises(ProtocolError, match="Hamilton contract"):
        hamilton_topup_allocate(
            {"a": 1.0, "b": -0.1}, topup_total_per_class=8
        )
    with pytest.raises(ProtocolError, match="eight sources"):
        target_topup_geometry(INNER_SOURCES)


@pytest.mark.parametrize(
    "invalid_ballots",
    [
        {source: 0.5 for source in TARGET_SOURCES[:-1]},
        {**{source: 0.5 for source in TARGET_SOURCES}, "8": 0.5},
        {**{source: 0.5 for source in TARGET_SOURCES}, "3": float("nan")},
        {**{source: 0.5 for source in TARGET_SOURCES}, "3": -0.1},
        {**{source: 0.5 for source in TARGET_SOURCES}, "3": 1.1},
        {**{source: 0.5 for source in TARGET_SOURCES}, "3": True},
        {source: 1.0 for source in TARGET_SOURCES},
    ],
)
def test_borda_action_rejects_invalid_ballots(invalid_ballots) -> None:
    with pytest.raises(ProtocolError, match="normalized-midrank|Borda"):
        build_borda_directed_topup_action(
            invalid_ballots,
            geometry=target_topup_geometry(TARGET_SOURCES),
        )


@pytest.mark.parametrize("selected_source", ["", " 3", "missing", None])
def test_single_source_tail_rejects_invalid_source(selected_source) -> None:
    with pytest.raises(ProtocolError, match="Single-source tail source"):
        build_single_source_tail_action(
            selected_source,
            geometry=target_topup_geometry(TARGET_SOURCES),
        )


def test_final_counts_balance_and_density_invariants_hold_without_projection() -> None:
    target = target_topup_geometry(TARGET_SOURCES)
    extreme = build_softmax_energy_topup_action(
        {source: (0.0 if source == "0" else 1000.0) for source in TARGET_SOURCES},
        geometry=target,
        temperature=1.0,
    )
    assert dict(extreme.topup_counts) == {
        "0": 128,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
    }
    for class_label in (0, 1):
        counts = extreme.final_counts_by_class[class_label]
        weights = extreme.final_weights_by_class[class_label]
        assert sum(counts.values()) == 1152
        assert sum(weights.values()) == pytest.approx(1.0)
        assert max(weights.values()) <= 0.25
        assert extreme.effective_source_count_by_class[class_label] >= 6.0
    assert extreme.maximum_source_weight == pytest.approx(2.0 / 9.0)
    assert (
        extreme.density_constraint_semantics
        == "validated_after_addition_without_projection"
    )


def test_composition_uses_disjoint_base_and_topup_windows_and_exact_balance() -> None:
    geometry = _small_geometry()
    action = build_energy_directed_topup_action(
        {source: float(index) for index, source in enumerate(TARGET_SOURCES)},
        geometry=geometry,
    )
    composition = compose_residual_topup_blocks(
        _blocks(), action, shuffle_seed_by_class={0: 17, 1: 42}
    )
    assert composition.embeddings.shape == (144, 3)
    assert composition.labels.shape == (144,)
    assert int(np.sum(composition.labels == 0)) == 72
    assert int(np.sum(composition.labels == 1)) == 72
    assert composition.component_by_row.count("base") == 128
    assert composition.component_by_row.count("topup") == 16
    assert composition.embeddings.flags.writeable is False
    assert composition.labels.flags.writeable is False
    assert len(composition.allocation_hash) == 64
    assert len(composition.window_hash) == 64
    assert len(composition.pre_shuffle_sha256_by_class[0]) == 64
    assert len(composition.post_shuffle_sha256_by_class[1]) == 64

    # Each selected source/class index appears exactly once: base is [0, 8)
    # and top-up begins at 8, so no row can occur in both components.
    for class_label in (0, 1):
        class_rows = composition.embeddings[composition.labels == class_label]
        assert len(np.unique(class_rows, axis=0)) == len(class_rows)


def test_composition_rejects_insufficient_capacity_and_overlapping_windows() -> None:
    geometry = _small_geometry()
    action = build_softmax_energy_topup_action(
        {source: (0.0 if source == "0" else 1000.0) for source in TARGET_SOURCES},
        geometry=geometry,
        temperature=1.0,
    )
    with pytest.raises(ProtocolError, match="insufficient class capacity"):
        compose_residual_topup_blocks(
            _blocks(capacity_per_class=15),
            action,
            shuffle_seed_by_class={0: 1, 1: 2},
        )
    with pytest.raises(ProtocolError, match="contiguous and disjoint"):
        SourceClassWindows(
            base_start=0,
            base_stop=8,
            topup_start=7,
            topup_stop=9,
        )


def test_uniform_composition_is_byte_identical_under_mapping_permutations() -> None:
    geometry_a = _small_geometry()
    geometry_b = build_topup_geometry(
        reversed(TARGET_SOURCES),
        base_per_source=8,
        topup_total_per_class=8,
    )
    action_a = build_uniform_topup_action(geometry_a)
    action_b = build_uniform_topup_action(geometry_b)
    blocks_a = _blocks()
    blocks_b = {
        source: _block(int(source)) for source in reversed(TARGET_SOURCES)
    }
    first = compose_residual_topup_blocks(
        blocks_a,
        action_a,
        shuffle_seed_by_class={0: 101, 1: 202},
    )
    second = compose_residual_topup_blocks(
        blocks_b,
        action_b,
        shuffle_seed_by_class={"1": 202, "0": 101},
    )
    assert action_a.action_hash == action_b.action_hash
    np.testing.assert_array_equal(first.embeddings, second.embeddings)
    np.testing.assert_array_equal(first.labels, second.labels)
    assert first.source_by_row == second.source_by_row
    assert first.component_by_row == second.component_by_row
    assert first.output_sha256 == second.output_sha256
    assert first.composition_hash == second.composition_hash
    json.dumps(first.to_payload(), sort_keys=True)


def test_shuffle_permutation_is_fixed_by_class_seed() -> None:
    geometry = _small_geometry()
    action = build_uniform_topup_action(geometry)
    blocks = _blocks()
    first = compose_residual_topup_blocks(
        blocks, action, shuffle_seed_by_class={0: 9, 1: 10}
    )
    repeated = compose_residual_topup_blocks(
        blocks, action, shuffle_seed_by_class={1: 10, 0: 9}
    )
    changed = compose_residual_topup_blocks(
        blocks, action, shuffle_seed_by_class={0: 19, 1: 20}
    )
    np.testing.assert_array_equal(
        first.permutation_by_class[0], repeated.permutation_by_class[0]
    )
    np.testing.assert_array_equal(first.embeddings, repeated.embeddings)
    assert first.composition_hash == repeated.composition_hash
    assert not np.array_equal(first.embeddings, changed.embeddings)
    assert first.composition_hash != changed.composition_hash


def test_base_only_arm_is_exact_byte_stable_and_contains_no_topup_rows() -> None:
    geometry = _small_geometry()
    blocks = _blocks()
    first = compose_equal_union_base_blocks(
        blocks,
        geometry,
        shuffle_seed_by_class={0: 31, 1: 32},
    )
    rebuilt = compose_equal_union_base_blocks(
        {source: _block(int(source)) for source in reversed(TARGET_SOURCES)},
        build_topup_geometry(
            reversed(TARGET_SOURCES),
            base_per_source=8,
            topup_total_per_class=8,
        ),
        shuffle_seed_by_class={"1": 32, "0": 31},
    )
    assert first.total_per_class == 64
    assert first.embeddings.shape == (128, 3)
    assert int(np.sum(first.labels == 0)) == 64
    assert int(np.sum(first.labels == 1)) == 64
    assert set(first.component_by_row) == {"base"}
    assert len(first.component_by_row) == 128
    assert all(first.source_by_row.count(source) == 16 for source in TARGET_SOURCES)
    np.testing.assert_array_equal(first.embeddings, rebuilt.embeddings)
    np.testing.assert_array_equal(first.labels, rebuilt.labels)
    assert first.output_sha256 == rebuilt.output_sha256
    assert first.allocation_hash == rebuilt.allocation_hash
    assert first.window_hash == rebuilt.window_hash
    assert first.composition_hash == rebuilt.composition_hash

    matched_budget = compose_residual_topup_blocks(
        blocks,
        build_uniform_topup_action(geometry),
        shuffle_seed_by_class={0: 31, 1: 32},
    )
    assert first.composition_semantics != matched_budget.composition_semantics
    assert first.composition_hash != matched_budget.composition_hash
    assert first.total_per_class < matched_budget.total_per_class
    json.dumps(first.to_payload(), sort_keys=True)


def test_base_only_arm_rejects_capacity_shortfall_and_ignores_suffix() -> None:
    geometry = _small_geometry()
    with pytest.raises(ProtocolError, match="insufficient class capacity"):
        compose_equal_union_base_blocks(
            _blocks(capacity_per_class=7),
            geometry,
            shuffle_seed_by_class={0: 3, 1: 4},
        )

    original = _blocks()
    changed = _blocks()
    for block in changed.values():
        embeddings = block["embeddings"]
        embeddings[8:16] += 9999.0
        embeddings[24:32] += 9999.0
    first = compose_equal_union_base_blocks(
        original,
        geometry,
        shuffle_seed_by_class={0: 3, 1: 4},
    )
    suffix_changed = compose_equal_union_base_blocks(
        changed,
        geometry,
        shuffle_seed_by_class={0: 3, 1: 4},
    )
    np.testing.assert_array_equal(first.embeddings, suffix_changed.embeddings)
    assert first.output_sha256 == suffix_changed.output_sha256
    assert first.composition_hash == suffix_changed.composition_hash
